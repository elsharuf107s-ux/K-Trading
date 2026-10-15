from __future__ import annotations

from collections import defaultdict
from math import inf

from .storage import Storage


def _brier(rows: list[dict], key: str) -> float | None:
    if not rows:
        return None
    return sum((float(row[key]) - float(row['outcome_yes'])) ** 2 for row in rows) / len(rows)


class ResearchAnalytics:
    """Out-of-sample-friendly research summaries from already-recorded observations.

    These metrics do not prove profitability. They tell us whether probability
    forecasts are calibrated, whether they beat the live market probability
    baseline, and how much execution friction the paper strategy is paying.
    """

    def __init__(self, storage: Storage):
        self.storage = storage
        self._labeled_cache: tuple[tuple[int, int], dict] | None = None

    def _labeled_summary(self) -> dict:
        # Brier/calibration read up to 100k labeled snapshots, but they only
        # change when a market settles, so recompute just on label changes.
        version = self.storage.labels_version()
        if self._labeled_cache is not None and self._labeled_cache[0] == version:
            return self._labeled_cache[1]
        snapshots = self.storage.labeled_observations(limit=100_000)
        windows = self.storage.last_labeled_observations(limit=20_000)

        model_brier = _brier(windows, 'model_probability')
        market_brier = _brier(windows, 'market_probability')
        improvement = None
        if model_brier is not None and market_brier not in (None, 0):
            improvement = (market_brier - model_brier) / market_brier
        summary = {
            'completed_windows': len(windows),
            'labeled_snapshots': len(snapshots),
            'model_brier_last_snapshot': model_brier,
            'market_brier_last_snapshot': market_brier,
            'brier_improvement_vs_market': improvement,
            'calibration': self._calibration(snapshots),
        }
        self._labeled_cache = (version, summary)
        return summary

    def summary(self) -> dict:
        trades = self.storage.closed_trade_stats(limit=20_000)
        closed = trades['closed_trades']
        gross_profit = trades['gross_profit']
        gross_loss = abs(trades['gross_loss'])
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (inf if gross_profit > 0 else None)

        return {
            **self._labeled_summary(),
            'trading': {
                'closed_trades': closed,
                'realized_pnl': trades['realized_pnl'],
                'win_rate': trades['wins'] / closed if closed else None,
                'average_win': trades['gross_profit'] / trades['wins'] if trades['wins'] else None,
                'average_loss': trades['gross_loss'] / trades['losses'] if trades['losses'] else None,
                'profit_factor': None if profit_factor is None else (profit_factor if profit_factor != inf else 'inf'),
            },
            'execution': self.storage.execution_metrics(),
            'execution_attribution': self.storage.execution_attribution(),
            'warning': 'Calibration/Brier need many independent completed windows; correlated snapshots are shown for diagnosis only.',
        }

    def _calibration(self, rows: list[dict], bins: int = 10) -> list[dict]:
        grouped: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            p = max(0.0, min(1.0, float(row['model_probability'])))
            idx = min(bins - 1, int(p * bins))
            grouped[idx].append(row)
        result = []
        for idx in range(bins):
            bucket = grouped.get(idx, [])
            if not bucket:
                continue
            result.append({
                'range_low': idx / bins,
                'range_high': (idx + 1) / bins,
                'count': len(bucket),
                'mean_prediction': sum(float(r['model_probability']) for r in bucket) / len(bucket),
                'market_mean': sum(float(r['market_probability']) for r in bucket) / len(bucket),
                'actual_yes_rate': sum(float(r['outcome_yes']) for r in bucket) / len(bucket),
            })
        return result
