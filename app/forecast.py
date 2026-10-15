from __future__ import annotations

import math
from .models import FeatureVector, Forecast


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _logit(p: float) -> float:
    p = _clip(p, 0.01, 0.99)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class BaselineForecaster:
    """Conservative data-collection baseline, not a claim of alpha."""

    def predict(self, f: FeatureVector) -> Forecast:
        anchor = _clip(f.market_probability, 0.03, 0.97)
        vol = max(f.realized_vol_60s, 0.00008)
        m30 = _clip(f.momentum_30s / vol, -3, 3)
        m60 = _clip(f.momentum_60s / vol, -3, 3)
        m180 = _clip(f.momentum_180s / (vol * math.sqrt(3)), -3, 3)

        signal = 0.10 * m30 + 0.14 * m60 + 0.08 * m180
        reasons: list[str] = []
        if f.distance_to_target_bps is not None:
            distance_signal = _clip(f.distance_to_target_bps / 25.0, -2.5, 2.5)
            time_weight = _clip(1 - f.seconds_remaining / 900.0, 0.15, 0.9)
            signal += 0.18 * distance_signal * time_weight
            reasons.append(f'BTC is {f.distance_to_target_bps:+.1f} bps from detected target')

        signal *= _clip(1.0 - f.source_dispersion * 150, 0.35, 1.0)
        raw = _sigmoid(_logit(anchor) + signal)
        fair = _clip(raw, anchor - 0.16, anchor + 0.16)

        if f.momentum_60s > 0:
            reasons.append(f'60s BTC momentum positive ({f.momentum_60s*100:+.3f}%)')
        elif f.momentum_60s < 0:
            reasons.append(f'60s BTC momentum negative ({f.momentum_60s*100:+.3f}%)')
        else:
            reasons.append('Not enough local price history yet; staying close to market prior')

        confidence = _clip(0.35 + min(abs(signal), 1.0) * 0.35 - min(f.source_dispersion * 100, 0.2), 0.2, 0.8)
        return Forecast(yes_probability=fair, confidence=confidence, rationale=reasons)
