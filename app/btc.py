from __future__ import annotations

import statistics
import httpx

from .config import settings
from .models import BtcSnapshot


class BtcPriceClient:
    def __init__(self):
        self._client = httpx.AsyncClient(timeout=5.0, headers={'User-Agent': 'K-Trading/0.1'})

    async def close(self) -> None:
        await self._client.aclose()

    async def _coinbase(self) -> float:
        r = await self._client.get(settings.coinbase_spot_url)
        r.raise_for_status()
        return float(r.json()['data']['amount'])

    async def _kraken(self) -> float:
        r = await self._client.get(settings.kraken_ticker_url)
        r.raise_for_status()
        result = r.json()['result']
        row = next(iter(result.values()))
        return float(row['c'][0])

    async def get(self) -> BtcSnapshot:
        sources: dict[str, float] = {}
        for name, fn in (('coinbase', self._coinbase), ('kraken', self._kraken)):
            try:
                value = await fn()
                if value > 0:
                    sources[name] = value
            except Exception:
                continue
        if not sources:
            raise RuntimeError('No BTC proxy price source is currently available')
        return BtcSnapshot(price=statistics.median(sources.values()), sources=sources)
