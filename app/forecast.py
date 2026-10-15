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
    """Conservative wave-aware research baseline; not a claim of alpha."""

    def predict(self, f: FeatureVector) -> Forecast:
        anchor = _clip(f.market_probability, 0.03, 0.97)
        spot_vol = max(f.realized_vol_60s, 0.00008)
        m10 = _clip(f.momentum_10s / spot_vol, -3, 3)
        m30 = _clip(f.momentum_30s / spot_vol, -3, 3)
        m60 = _clip(f.momentum_60s / spot_vol, -3, 3)
        m180 = _clip(f.momentum_180s / (spot_vol * math.sqrt(3)), -3, 3)

        signal = (
            0.03 * m10
            + 0.06 * m30
            + 0.07 * m60
            + 0.04 * m180
            + 0.32 * f.wave_score
            + 0.08 * _clip(f.book_imbalance, -1, 1)
        )
        reasons = [
            f'Wave {f.wave_state} ({f.wave_score:+.2f})',
            f'Book imbalance {f.book_imbalance:+.2f}',
        ]

        official = f.reference_source.startswith('cfbenchmarks')
        if official and f.reference_vol_60s > 0:
            ref_vol = max(f.reference_vol_60s, 0.00008)
            ref30 = _clip(f.reference_momentum_30s / ref_vol, -3, 3)
            ref60 = _clip(f.reference_momentum_60s / ref_vol, -3, 3)
            signal += 0.08 * ref30 + 0.05 * ref60
            reasons.append(f'BRTI 30s momentum {f.reference_momentum_30s*100:+.3f}%')

        if f.distance_to_target_bps is not None:
            distance_signal = _clip(f.distance_to_target_bps / 25.0, -3.0, 3.0)
            time_weight = _clip(1 - f.seconds_remaining / 900.0, 0.15, 1.0)
            signal += (0.24 if official else 0.14) * distance_signal * time_weight
            reasons.append(f'{f.reference_source}: {f.distance_to_target_bps:+.1f} bps from target')

        signal *= _clip(1.0 - f.source_dispersion * 150, 0.35, 1.0)
        raw = _sigmoid(_logit(anchor) + signal)
        max_shift = 0.20 if official else 0.14
        fair = _clip(raw, anchor - max_shift, anchor + max_shift)

        if f.momentum_60s > 0:
            reasons.append(f'60s spot momentum {f.momentum_60s*100:+.3f}%')
        elif f.momentum_60s < 0:
            reasons.append(f'60s spot momentum {f.momentum_60s*100:+.3f}%')
        else:
            reasons.append('Local spot history still warming up')

        confidence = (
            0.34
            + min(abs(signal), 1.0) * 0.32
            + (0.10 if official else 0)
            - min(f.source_dispersion * 100, 0.18)
        )
        return Forecast(
            yes_probability=fair,
            confidence=_clip(confidence, 0.2, 0.85),
            rationale=reasons,
        )
