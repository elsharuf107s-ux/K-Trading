from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import httpx

from .config import settings
from .models import MarketSnapshot


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == '':
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _price(m: dict[str, Any], dollar_key: str, legacy_key: str, default: float) -> float:
    if dollar_key in m and m.get(dollar_key) not in (None, ''):
        return max(0.0, min(1.0, _to_float(m.get(dollar_key), default)))
    legacy = _to_float(m.get(legacy_key), default * 100)
    return max(0.0, min(1.0, legacy / 100.0 if legacy > 1 else legacy))


def _dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return None


def _extract_target(m: dict[str, Any]) -> float | None:
    for key in ('floor_strike', 'cap_strike', 'strike', 'strike_value', 'target_price'):
        val = _to_float(m.get(key), 0.0)
        if val > 1000:
            return val
    text = ' '.join(str(m.get(k) or '') for k in (
        'functional_strike', 'subtitle', 'yes_sub_title', 'no_sub_title', 'rules_primary', 'title'
    )).replace(',', '')
    nums = re.findall(r'\$?([0-9]{4,6}(?:\.[0-9]+)?)', text)
    return float(nums[-1]) if nums else None


class KalshiPublicClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or settings.kalshi_base_url).rstrip('/')
        self._client = httpx.AsyncClient(timeout=10.0, headers={'User-Agent': 'K-Trading/0.2'})

    async def close(self) -> None:
        await self._client.aclose()

    async def list_open_markets(self, max_pages: int = 3) -> list[dict[str, Any]]:
        params: dict[str, Any] = {'status': 'open', 'limit': 200}
        if settings.kalshi_series_ticker:
            params['series_ticker'] = settings.kalshi_series_ticker
        cursor = ''
        markets: list[dict[str, Any]] = []
        for _ in range(max_pages):
            if cursor:
                params['cursor'] = cursor
            response = await self._client.get(f'{self.base_url}/markets', params=params)
            response.raise_for_status()
            payload = response.json()
            markets.extend(payload.get('markets', []))
            cursor = payload.get('cursor') or ''
            if not cursor:
                break
        return markets

    async def get_market(self, ticker: str) -> dict[str, Any]:
        response = await self._client.get(f'{self.base_url}/markets/{ticker}')
        response.raise_for_status()
        payload = response.json()
        return payload.get('market', payload)

    async def get_orderbook(self, ticker: str, depth: int = 20) -> dict[str, Any]:
        response = await self._client.get(f'{self.base_url}/markets/{ticker}/orderbook', params={'depth': depth})
        response.raise_for_status()
        payload = response.json()
        return payload.get('orderbook_fp') or payload.get('orderbook') or payload

    async def discover_btc_15m(self) -> MarketSnapshot | None:
        if settings.market_ticker_override:
            raw = await self.get_market(settings.market_ticker_override)
            return self.normalize(raw)
        now = datetime.now(timezone.utc)
        candidates: list[tuple[float, dict[str, Any]]] = []
        for m in await self.list_open_markets():
            ticker = str(m.get('ticker') or '')
            if settings.kalshi_series_ticker and not ticker.upper().startswith(settings.kalshi_series_ticker.upper()):
                continue
            open_time = _dt(m.get('open_time'))
            close_time = _dt(m.get('close_time') or m.get('expected_expiration_time') or m.get('expiration_time'))
            if not close_time:
                continue
            remaining = (close_time - now).total_seconds()
            if remaining <= 0:
                continue
            duration = (close_time - open_time).total_seconds() if open_time else 900.0
            duration_penalty = abs(duration - 900.0) / 900.0
            candidates.append((remaining + duration_penalty * 120.0, m))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return self.normalize(candidates[0][1])

    async def enrich_orderbook(self, market: MarketSnapshot, depth_levels: int = 5) -> MarketSnapshot:
        try:
            book = await self.get_orderbook(market.ticker, depth=max(depth_levels, 10))
        except Exception:
            return market
        def levels(name: str) -> list[tuple[float, float]]:
            out: list[tuple[float, float]] = []
            for row in book.get(name, []) or []:
                if not isinstance(row, (list, tuple)) or len(row) < 2:
                    continue
                price = _to_float(row[0]); qty = _to_float(row[1])
                if 0 <= price <= 1 and qty > 0:
                    out.append((price, qty))
            out.sort(key=lambda x: x[0])
            return out
        yes = levels('yes_dollars') or levels('yes')
        no = levels('no_dollars') or levels('no')
        yes_bid = yes[-1][0] if yes else market.yes_bid
        no_bid = no[-1][0] if no else market.no_bid
        yes_ask = 1.0 - no_bid if no else market.yes_ask
        no_ask = 1.0 - yes_bid if yes else market.no_ask
        yes_depth = sum(q for _, q in yes[-depth_levels:]); no_depth = sum(q for _, q in no[-depth_levels:])
        total = yes_depth + no_depth
        imbalance = (yes_depth - no_depth) / total if total > 0 else 0.0
        return market.model_copy(update={
            'yes_bid': max(0.0, min(1.0, yes_bid)), 'yes_ask': max(0.0, min(1.0, yes_ask)),
            'no_bid': max(0.0, min(1.0, no_bid)), 'no_ask': max(0.0, min(1.0, no_ask)),
            'yes_depth': yes_depth, 'no_depth': no_depth,
            'book_imbalance': max(-1.0, min(1.0, imbalance)), 'captured_at': datetime.now(timezone.utc),
        })

    def normalize(self, m: dict[str, Any]) -> MarketSnapshot:
        now = datetime.now(timezone.utc)
        open_time = _dt(m.get('open_time'))
        close_time = _dt(m.get('close_time') or m.get('expected_expiration_time') or m.get('expiration_time'))
        seconds_remaining = max(0.0, (close_time - now).total_seconds()) if close_time else 0.0
        yes_bid = _price(m, 'yes_bid_dollars', 'yes_bid', 0.0)
        yes_ask = _price(m, 'yes_ask_dollars', 'yes_ask', 1.0)
        no_bid = _price(m, 'no_bid_dollars', 'no_bid', max(0.0, 1 - yes_ask))
        no_ask = _price(m, 'no_ask_dollars', 'no_ask', min(1.0, 1 - yes_bid))
        last = _price(m, 'last_price_dollars', 'last_price', (yes_bid + yes_ask) / 2 if yes_ask < 1 else 0.5)
        return MarketSnapshot(
            ticker=str(m.get('ticker') or ''), event_ticker=str(m.get('event_ticker') or ''),
            title=str(m.get('title') or ''), subtitle=str(m.get('subtitle') or ''), open_time=open_time,
            close_time=close_time, yes_bid=yes_bid, yes_ask=yes_ask, no_bid=no_bid, no_ask=no_ask,
            last_price=last, volume=_to_float(m.get('volume_fp', m.get('volume', 0))),
            open_interest=_to_float(m.get('open_interest_fp', m.get('open_interest', 0))),
            seconds_remaining=seconds_remaining, raw_target=_extract_target(m),
            rules_primary=str(m.get('rules_primary') or ''), captured_at=now,
        )
