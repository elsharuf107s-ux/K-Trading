from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import time
from typing import Any
from urllib.parse import urlparse

import websockets

from .auth import auth_headers
from .config import settings
from .models import MarketSnapshot


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ''):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_levels(payload: dict[str, Any], side: str) -> dict[float, float]:
    raw = (
        payload.get(f'{side}_dollars_fp')
        or payload.get(f'{side}_dollars')
        or payload.get(side)
        or []
    )
    levels: dict[float, float] = {}
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        price = _float(row[0])
        qty = _float(row[1])
        if price > 1:
            price /= 100.0
        if 0 < price < 1 and qty > 0:
            levels[round(price, 6)] = qty
    return levels


@dataclass
class QuoteSample:
    ts: float
    ticker: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float


@dataclass
class QuoteMove:
    ticker: str
    horizon_ms: int
    elapsed_ms: float
    yes_ask_move: float
    no_ask_move: float
    captured_at: str


class OrderBookStream:
    """Authenticated real-time Kalshi orderbook with empirical quote-move sampling.

    The channel sends one snapshot followed by sequenced deltas. A sequence gap
    invalidates local state and forces a reconnect so we do not silently trade
    from a corrupted book.
    """

    def __init__(self):
        self.target_ticker = ''
        self.active_ticker = ''
        self.yes: dict[float, float] = {}
        self.no: dict[float, float] = {}
        self.seq: int | None = None
        self.connected = False
        self.synced = False
        self.error: str | None = None
        self.last_message_at: datetime | None = None
        self.sequence_gaps = 0
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._quotes: deque[QuoteSample] = deque(maxlen=4096)
        self._moves: deque[QuoteMove] = deque(maxlen=100_000)

    @property
    def enabled(self) -> bool:
        return settings.enable_kalshi_ws and settings.has_kalshi_credentials

    @property
    def horizons_ms(self) -> list[int]:
        out: list[int] = []
        for raw in str(settings.empirical_quote_horizons_ms).split(','):
            try:
                value = int(raw.strip())
            except ValueError:
                continue
            if value > 0:
                out.append(value)
        return sorted(set(out)) or [250, 500, 1000]

    async def start(self) -> None:
        if self.enabled and self._task is None:
            self._task = asyncio.create_task(self._run(), name='orderbook-stream')

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task

    def set_ticker(self, ticker: str) -> None:
        ticker = str(ticker or '')
        if ticker != self.target_ticker:
            self.target_ticker = ticker
            self.synced = False

    def fresh(self, ticker: str) -> bool:
        if not self.synced or ticker != self.active_ticker or not self.last_message_at:
            return False
        age = (datetime.now(timezone.utc) - self.last_message_at).total_seconds()
        return age <= settings.orderbook_stream_stale_seconds

    def status(self) -> dict[str, Any]:
        age = None
        if self.last_message_at:
            age = (datetime.now(timezone.utc) - self.last_message_at).total_seconds()
        return {
            'enabled': self.enabled,
            'connected': self.connected,
            'synced': self.synced,
            'target_ticker': self.target_ticker,
            'active_ticker': self.active_ticker,
            'last_message_at': self.last_message_at.isoformat() if self.last_message_at else None,
            'age_seconds': age,
            'sequence': self.seq,
            'sequence_gaps': self.sequence_gaps,
            'error': self.error,
            'quote_samples': len(self._quotes),
            'pending_move_samples': len(self._moves),
        }

    def apply_to_market(self, market: MarketSnapshot) -> MarketSnapshot:
        if not self.fresh(market.ticker) or not self.yes or not self.no:
            return market
        yes_levels = sorted(self.yes.items())
        no_levels = sorted(self.no.items())
        yes_bid = yes_levels[-1][0]
        no_bid = no_levels[-1][0]
        yes_ask = 1.0 - no_bid
        no_ask = 1.0 - yes_bid
        depth_levels = 5
        yes_depth = sum(q for _, q in yes_levels[-depth_levels:])
        no_depth = sum(q for _, q in no_levels[-depth_levels:])
        total = yes_depth + no_depth
        imbalance = (yes_depth - no_depth) / total if total > 0 else 0.0
        return market.model_copy(update={
            'yes_bid': yes_bid,
            'yes_ask': yes_ask,
            'no_bid': no_bid,
            'no_ask': no_ask,
            'yes_depth': yes_depth,
            'no_depth': no_depth,
            'book_imbalance': max(-1.0, min(1.0, imbalance)),
            'yes_book': yes_levels,
            'no_book': no_levels,
            'captured_at': self.last_message_at or market.captured_at,
        })

    def drain_moves(self, max_items: int = 10_000) -> list[QuoteMove]:
        result: list[QuoteMove] = []
        while self._moves and len(result) < max_items:
            result.append(self._moves.popleft())
        return result

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            ticker = self.target_ticker
            if not ticker:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=0.25)
                except asyncio.TimeoutError:
                    pass
                continue
            try:
                await self._session(ticker)
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.connected = False
                self.synced = False
                self.error = f'{type(exc).__name__}: {exc}'
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    backoff = min(backoff * 2, 15.0)

    async def _session(self, ticker: str) -> None:
        url = settings.kalshi_ws_url
        path = urlparse(url).path or '/trade-api/ws/v2'
        headers = auth_headers('GET', path)
        self.yes.clear()
        self.no.clear()
        self.seq = None
        self.active_ticker = ticker
        self.synced = False

        async with websockets.connect(
            url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=20,
            max_queue=8192,
        ) as ws:
            self.connected = True
            self.error = None
            await ws.send(json.dumps({
                'id': 101,
                'cmd': 'subscribe',
                'params': {
                    'channels': ['orderbook_delta'],
                    'market_tickers': [ticker],
                    'use_yes_price': False,
                },
            }))

            while not self._stop.is_set() and self.target_ticker == ticker:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                self._handle(message)

        self.connected = False
        self.synced = False

    def _check_seq(self, message: dict[str, Any]) -> None:
        raw_seq = message.get('seq')
        if raw_seq is None:
            return
        seq = int(raw_seq)
        if self.seq is not None and seq != self.seq + 1:
            self.sequence_gaps += 1
            self.synced = False
            raise RuntimeError(f'orderbook sequence gap: expected {self.seq + 1}, got {seq}')
        self.seq = seq

    def _handle(self, message: dict[str, Any]) -> None:
        msg_type = str(message.get('type') or '')
        if msg_type == 'error':
            self.error = str(message.get('msg') or message)
            return
        if msg_type not in ('orderbook_snapshot', 'orderbook_delta'):
            return

        msg = message.get('msg') or {}
        ticker = str(msg.get('market_ticker') or '')
        if ticker and ticker != self.active_ticker:
            return
        self._check_seq(message)

        if msg_type == 'orderbook_snapshot':
            self.yes = _parse_levels(msg, 'yes')
            self.no = _parse_levels(msg, 'no')
            self.synced = bool(self.yes or self.no)
        else:
            side = str(msg.get('side') or '').lower()
            if side not in ('yes', 'no'):
                return
            price = _float(msg.get('price_dollars'))
            if price <= 0:
                price = _float(msg.get('price'))
                if price > 1:
                    price /= 100.0
            delta = _float(msg.get('delta_fp'))
            if delta == 0:
                delta = _float(msg.get('delta'))
            if not (0 < price < 1):
                return
            book = self.yes if side == 'yes' else self.no
            key = round(price, 6)
            new_qty = book.get(key, 0.0) + delta
            if new_qty <= 1e-9:
                book.pop(key, None)
            else:
                book[key] = new_qty

        now = datetime.now(timezone.utc)
        self.last_message_at = now
        if self.yes and self.no:
            self.synced = True
            self._sample(now.timestamp())

    def _sample(self, ts: float) -> None:
        yes_bid = max(self.yes)
        no_bid = max(self.no)
        current = QuoteSample(
            ts=ts,
            ticker=self.active_ticker,
            yes_bid=yes_bid,
            yes_ask=1.0 - no_bid,
            no_bid=no_bid,
            no_ask=1.0 - yes_bid,
        )

        for horizon_ms in self.horizons_ms:
            target = ts - horizon_ms / 1000.0
            prior = None
            for sample in reversed(self._quotes):
                if sample.ticker != current.ticker:
                    continue
                if sample.ts <= target:
                    prior = sample
                    break
            if prior is None:
                continue
            elapsed_ms = (ts - prior.ts) * 1000.0
            # Reject extremely loose matches so a quiet market does not pretend
            # a 100ms horizon was measured from a quote several seconds old.
            tolerance = max(100.0, horizon_ms * 1.5)
            if elapsed_ms - horizon_ms > tolerance:
                continue
            self._moves.append(QuoteMove(
                ticker=current.ticker,
                horizon_ms=horizon_ms,
                elapsed_ms=elapsed_ms,
                yes_ask_move=current.yes_ask - prior.yes_ask,
                no_ask_move=current.no_ask - prior.no_ask,
                captured_at=datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            ))

        self._quotes.append(current)
        cutoff = ts - 5.0
        while self._quotes and self._quotes[0].ts < cutoff:
            self._quotes.popleft()
