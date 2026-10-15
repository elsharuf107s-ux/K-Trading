from __future__ import annotations

from collections import deque
import math
import statistics

from .models import BtcSnapshot, FeatureVector, MarketSnapshot


class PriceBuffer:
    def __init__(self, max_seconds: int = 900):
        self.max_seconds = max_seconds
        self.points: deque[tuple[float, float]] = deque()

    def add(self, snapshot: BtcSnapshot) -> None:
        ts = snapshot.captured_at.timestamp()
        self.points.append((ts, snapshot.price))
        cutoff = ts - self.max_seconds
        while self.points and self.points[0][0] < cutoff:
            self.points.popleft()

    def _price_at_or_before(self, seconds_ago: int) -> float | None:
        if not self.points:
            return None
        target = self.points[-1][0] - seconds_ago
        candidates = [p for ts, p in self.points if ts <= target]
        return candidates[-1] if candidates else None

    def momentum(self, seconds: int) -> float:
        if not self.points:
            return 0.0
        current = self.points[-1][1]
        prior = self._price_at_or_before(seconds)
        if prior is None or prior <= 0:
            return 0.0
        return current / prior - 1.0

    def realized_vol(self, seconds: int = 60) -> float:
        if len(self.points) < 3:
            return 0.0
        cutoff = self.points[-1][0] - seconds
        prices = [p for ts, p in self.points if ts >= cutoff]
        if len(prices) < 3:
            return 0.0
        rets = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices)) if prices[i - 1] > 0]
        return statistics.pstdev(rets) if len(rets) >= 2 else 0.0


def build_features(market: MarketSnapshot, btc: BtcSnapshot, buf: PriceBuffer) -> FeatureVector:
    dispersion = 0.0
    if len(btc.sources) > 1 and btc.price > 0:
        dispersion = (max(btc.sources.values()) - min(btc.sources.values())) / btc.price
    distance_bps = None
    if market.raw_target and market.raw_target > 0:
        distance_bps = (btc.price / market.raw_target - 1.0) * 10_000
    return FeatureVector(
        ticker=market.ticker,
        btc_price=btc.price,
        market_probability=market.yes_mid,
        yes_bid=market.yes_bid,
        yes_ask=market.yes_ask,
        no_bid=market.no_bid,
        no_ask=market.no_ask,
        spread=market.spread,
        seconds_remaining=market.seconds_remaining,
        momentum_30s=buf.momentum(30),
        momentum_60s=buf.momentum(60),
        momentum_180s=buf.momentum(180),
        realized_vol_60s=buf.realized_vol(60),
        source_dispersion=dispersion,
        target_price=market.raw_target,
        distance_to_target_bps=distance_bps,
    )
