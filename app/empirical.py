from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from statistics import mean
from typing import Iterable

from .orderbook_stream import QuoteMove


class EmpiricalExecutionLab:
    """Persists and summarizes observed short-horizon quote movement."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS quote_moves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                captured_at TEXT NOT NULL,
                ticker TEXT NOT NULL,
                horizon_ms INTEGER NOT NULL,
                elapsed_ms REAL NOT NULL,
                yes_ask_move REAL NOT NULL,
                no_ask_move REAL NOT NULL
            )
        ''')
        self.db.execute('CREATE INDEX IF NOT EXISTS idx_quote_moves_horizon ON quote_moves(horizon_ms,id)')
        self.db.commit()

    def save_moves(self, moves: Iterable[QuoteMove]) -> int:
        rows = [
            (
                m.captured_at, m.ticker, m.horizon_ms, m.elapsed_ms,
                m.yes_ask_move, m.no_ask_move,
            )
            for m in moves
        ]
        if not rows:
            return 0
        self.db.executemany(
            '''INSERT INTO quote_moves(captured_at,ticker,horizon_ms,elapsed_ms,yes_ask_move,no_ask_move)
               VALUES(?,?,?,?,?,?)''',
            rows,
        )
        self.db.commit()
        return len(rows)

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        position = max(0.0, min(1.0, quantile)) * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    def horizon_summary(self, horizon_ms: int, limit: int = 50_000) -> dict:
        rows = self.db.execute(
            '''SELECT elapsed_ms,yes_ask_move,no_ask_move FROM quote_moves
               WHERE horizon_ms=? ORDER BY id DESC LIMIT ?''',
            (int(horizon_ms), int(limit)),
        ).fetchall()
        if not rows:
            return {
                'horizon_ms': int(horizon_ms),
                'samples': 0,
                'mean_elapsed_ms': 0.0,
                'mean_yes_ask_move': 0.0,
                'mean_no_ask_move': 0.0,
                'mean_adverse_move': 0.0,
                'p95_adverse_move': 0.0,
                'p99_adverse_move': 0.0,
            }
        elapsed = [float(r[0]) for r in rows]
        yes = [float(r[1]) for r in rows]
        no = [float(r[2]) for r in rows]
        # Positive ask movement is adverse to a marketable buyer. Pool YES and
        # NO because the execution engine can trade either side.
        adverse = [max(0.0, x) for x in yes] + [max(0.0, x) for x in no]
        return {
            'horizon_ms': int(horizon_ms),
            'samples': len(rows),
            'mean_elapsed_ms': mean(elapsed),
            'mean_yes_ask_move': mean(yes),
            'mean_no_ask_move': mean(no),
            'mean_adverse_move': mean(adverse),
            'p95_adverse_move': self._percentile(adverse, 0.95),
            'p99_adverse_move': self._percentile(adverse, 0.99),
        }

    def summary(self, horizons_ms: list[int]) -> dict:
        horizons = [self.horizon_summary(h) for h in sorted(set(horizons_ms))]
        total = sum(int(item['samples']) for item in horizons)
        return {'total_samples': total, 'horizons': horizons}

    def recommended_adverse_slippage(self, latency_ms: float, min_samples: int = 200) -> dict:
        rows = self.db.execute(
            'SELECT DISTINCT horizon_ms FROM quote_moves ORDER BY horizon_ms ASC'
        ).fetchall()
        horizons = [int(r[0]) for r in rows]
        if not horizons:
            return {
                'ready': False,
                'requested_latency_ms': latency_ms,
                'horizon_ms': None,
                'samples': 0,
                'p95_adverse_move': 0.0,
            }
        horizon = min(horizons, key=lambda h: abs(h - latency_ms))
        stats = self.horizon_summary(horizon)
        return {
            'ready': int(stats['samples']) >= int(min_samples),
            'requested_latency_ms': latency_ms,
            'horizon_ms': horizon,
            'samples': stats['samples'],
            'p95_adverse_move': stats['p95_adverse_move'],
        }
