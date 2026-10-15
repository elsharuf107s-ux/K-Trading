from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import websockets

from .auth import auth_headers
from .config import settings
from .models import BenchmarkSnapshot


def _float(value: Any) -> float | None:
    try:
        if value in (None, ''):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_numeric(payload: Any) -> float | None:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return _float(payload)
    if isinstance(payload, dict):
        for key in ('value', 'value_usd', 'price', 'index_value', 'rate'):
            val = _float(payload.get(key))
            if val is not None and val > 100:
                return val
        for value in payload.values():
            found = _find_numeric(value)
            if found is not None and found > 100:
                return found
    if isinstance(payload, list):
        for value in payload:
            found = _find_numeric(value)
            if found is not None and found > 100:
                return found
    return None


class CFBenchmarkStream:
    """Authenticated Kalshi CF Benchmarks stream.

    K-Trading subscribes to the once-per-second ``cfbenchmarks_value`` channel
    because it includes the trailing 60-second average and the special
    quarter-hour final-minute average used by KXBTC15M settlement. The 5Hz
    sibling is accepted by the parser as a raw-value fallback, but it does not
    carry those averages.
    """

    def __init__(self):
        self.latest: BenchmarkSnapshot | None = None
        self.connected = False
        self.error: str | None = None
        self.last_message_at: datetime | None = None
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    @property
    def enabled(self) -> bool:
        return settings.enable_kalshi_ws and settings.has_kalshi_credentials

    async def start(self) -> None:
        if self.enabled and self._task is None:
            self._task = asyncio.create_task(self._run(), name='cfbenchmarks-stream')

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task

    def fresh(self) -> BenchmarkSnapshot | None:
        if not self.latest:
            return None
        age = (datetime.now(timezone.utc) - self.latest.captured_at).total_seconds()
        return self.latest if age <= settings.benchmark_stale_seconds else None

    def status(self) -> dict[str, Any]:
        latest = self.latest
        return {
            'enabled': self.enabled,
            'connected': self.connected,
            'channel': 'cfbenchmarks_value',
            'index_id': settings.cfbenchmarks_index_id,
            'last_message_at': self.last_message_at.isoformat() if self.last_message_at else None,
            'error': self.error,
            'latest': latest.model_dump(mode='json') if latest else None,
        }

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                await self._session()
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.connected = False
                self.error = f'{type(exc).__name__}: {exc}'
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    backoff = min(backoff * 2, 30.0)

    async def _session(self) -> None:
        url = settings.kalshi_ws_url
        path = urlparse(url).path or '/trade-api/ws/v2'
        headers = auth_headers('GET', path)
        async with websockets.connect(
            url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=20,
            max_queue=2048,
        ) as ws:
            self.connected = True
            self.error = None
            subscribe = {
                'id': 1,
                'cmd': 'subscribe',
                'params': {
                    'channels': ['cfbenchmarks_value'],
                    'index_ids': [settings.cfbenchmarks_index_id],
                },
            }
            await ws.send(json.dumps(subscribe))
            async for raw in ws:
                if self._stop.is_set():
                    break
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                self._handle(message)
        self.connected = False

    def _handle(self, message: dict[str, Any]) -> None:
        msg_type = str(message.get('type') or '')
        if msg_type == 'error':
            self.error = str(message.get('msg') or message)
            return
        if msg_type not in ('cfbenchmarks_value', 'cfbenchmarks_value_5hz'):
            return

        msg = message.get('msg') or {}
        index_id = str(msg.get('index_id') or '')
        if index_id not in ('', settings.cfbenchmarks_index_id):
            return

        previous = self.latest if self.latest and self.latest.index_id == (index_id or settings.cfbenchmarks_index_id) else None
        avg = msg.get('avg_60s_data') or {}
        settlement = msg.get('last_60s_windowed_average_15min') or {}

        raw_value = _float(msg.get('value_usd')) or _find_numeric(msg.get('data'))
        avg_value = _float(avg.get('value'))
        settlement_value = _float(settlement.get('value'))

        # The 5Hz channel intentionally omits average fields. If one of those
        # frames is ever routed here, preserve the last settlement-aware values
        # instead of replacing them with None.
        if msg_type == 'cfbenchmarks_value_5hz' and previous:
            avg_value = avg_value if avg_value is not None else previous.avg_60s
            settlement_value = settlement_value if settlement_value is not None else previous.settlement_window_avg

        current = raw_value or avg_value or settlement_value
        if current is None:
            return

        now = datetime.now(timezone.utc)
        self.latest = BenchmarkSnapshot(
            index_id=index_id or settings.cfbenchmarks_index_id,
            value=current,
            avg_60s=avg_value,
            settlement_window_avg=settlement_value,
            received_at_ms=int(msg.get('received_at')) if msg.get('received_at') else None,
            captured_at=now,
        )
        self.last_message_at = now
