from __future__ import annotations

from collections import deque
from datetime import datetime
import math
import statistics

from .models import BenchmarkSnapshot, BtcSnapshot, FeatureVector, MarketSnapshot


class PriceBuffer:
    def __init__(self, max_seconds: int = 900):
        self.max_seconds = max_seconds
        self.points: deque[tuple[float, float]] = deque()

    def add(self, snapshot: BtcSnapshot) -> None:
        self.add_point(snapshot.price, snapshot.captured_at)

    def add_point(self, price: float, captured_at: datetime) -> None:
        if price <= 0:
            return
        ts = captured_at.timestamp()
        self.points.append((ts, price))
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


class ReferenceBuffer(PriceBuffer):
    """Time-series buffer for the official BRTI reference feed."""

    def add_benchmark(self, snapshot: BenchmarkSnapshot) -> None:
        if snapshot.value:
            self.add_point(snapshot.value, snapshot.captured_at)


class MarketBuffer:
    def __init__(self, max_seconds: int = 900):
        self.max_seconds = max_seconds
        self.ticker = ''
        self.points: deque[tuple[float, float, float]] = deque()

    def add(self, market: MarketSnapshot) -> None:
        if market.ticker != self.ticker:
            self.ticker = market.ticker
            self.points.clear()
        ts = market.captured_at.timestamp()
        self.points.append((ts, market.yes_mid, market.book_imbalance))
        cutoff = ts - self.max_seconds
        while self.points and self.points[0][0] < cutoff:
            self.points.popleft()

    def momentum(self, seconds: int) -> float:
        if not self.points:
            return 0.0
        now_ts, current, _ = self.points[-1]
        target = now_ts - seconds
        candidates = [p for ts, p, _ in self.points if ts <= target]
        return current - candidates[-1] if candidates else 0.0


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def build_features(
    market: MarketSnapshot,
    btc: BtcSnapshot,
    buf: PriceBuffer,
    market_buf: MarketBuffer | None = None,
    benchmark: BenchmarkSnapshot | None = None,
    reference_buf: ReferenceBuffer | None = None,
) -> FeatureVector:
    dispersion = 0.0
    if len(btc.sources) > 1 and btc.price > 0:
        dispersion = (max(btc.sources.values()) - min(btc.sources.values())) / btc.price

    reference_price = btc.price
    reference_source = 'spot_proxy'
    benchmark_avg = None
    settlement_avg = None
    if benchmark:
        benchmark_avg = benchmark.avg_60s
        settlement_avg = benchmark.settlement_window_avg
        if market.seconds_remaining <= 60 and settlement_avg:
            reference_price = settlement_avg
            reference_source = 'cfbenchmarks_settlement_window'
        elif benchmark.value:
            reference_price = benchmark.value
            reference_source = 'cfbenchmarks'
        elif benchmark_avg:
            reference_price = benchmark_avg
            reference_source = 'cfbenchmarks_avg60'

    distance_bps = (
        (reference_price / market.raw_target - 1.0) * 10_000
        if market.raw_target and market.raw_target > 0
        else None
    )

    mm10 = market_buf.momentum(10) if market_buf else 0.0
    mm30 = market_buf.momentum(30) if market_buf else 0.0

    spot_vol = max(buf.realized_vol(60), 0.00008)
    spot_m30 = buf.momentum(30)

    ref_m10 = reference_buf.momentum(10) if reference_buf and reference_buf.points else 0.0
    ref_m30 = reference_buf.momentum(30) if reference_buf and reference_buf.points else 0.0
    ref_m60 = reference_buf.momentum(60) if reference_buf and reference_buf.points else 0.0
    ref_vol = reference_buf.realized_vol(60) if reference_buf and reference_buf.points else 0.0

    # When BRTI is live, use its own direction as the primary wave input. Spot
    # remains useful as a cross-check and leading microstructure feature.
    if reference_source.startswith('cfbenchmarks') and reference_buf and len(reference_buf.points) >= 2:
        primary_vol = max(ref_vol, 0.00008)
        primary_momentum = ref_m30
    else:
        primary_vol = spot_vol
        primary_momentum = spot_m30

    btc_norm = _clip(primary_momentum / primary_vol, -3.0, 3.0) / 3.0
    market_norm = _clip(mm10 / 0.10, -1.0, 1.0)
    imbalance = _clip(market.book_imbalance, -1.0, 1.0)
    distance_norm = _clip((distance_bps or 0.0) / 30.0, -1.0, 1.0)
    spot_crosscheck = _clip(spot_m30 / spot_vol, -3.0, 3.0) / 3.0

    wave_score = _clip(
        0.31 * btc_norm
        + 0.25 * market_norm
        + 0.20 * imbalance
        + 0.14 * distance_norm
        + 0.10 * spot_crosscheck,
        -1.0,
        1.0,
    )
    wave_state = 'UP' if wave_score >= 0.12 else 'DOWN' if wave_score <= -0.12 else 'FLAT'

    return FeatureVector(
        ticker=market.ticker,
        btc_price=btc.price,
        reference_price=reference_price,
        reference_source=reference_source,
        benchmark_avg_60s=benchmark_avg,
        settlement_window_avg=settlement_avg,
        market_probability=market.yes_mid,
        yes_bid=market.yes_bid,
        yes_ask=market.yes_ask,
        no_bid=market.no_bid,
        no_ask=market.no_ask,
        spread=market.spread,
        seconds_remaining=market.seconds_remaining,
        momentum_10s=buf.momentum(10),
        momentum_30s=spot_m30,
        momentum_60s=buf.momentum(60),
        momentum_180s=buf.momentum(180),
        reference_momentum_10s=ref_m10,
        reference_momentum_30s=ref_m30,
        reference_momentum_60s=ref_m60,
        realized_vol_60s=buf.realized_vol(60),
        reference_vol_60s=ref_vol,
        source_dispersion=dispersion,
        target_price=market.raw_target,
        distance_to_target_bps=distance_bps,
        yes_depth=market.yes_depth,
        no_depth=market.no_depth,
        book_imbalance=market.book_imbalance,
        market_momentum_10s=mm10,
        market_momentum_30s=mm30,
        wave_score=wave_score,
        wave_state=wave_state,
    )
