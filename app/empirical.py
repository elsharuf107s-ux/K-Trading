from __future__ import annotations

from dataclasses import dataclass
import sqlite3
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from .orderbook_stream import QuoteMove


@dataclass
class _HorizonCache:
    last_id: int
    moves: np.ndarray  # rows of (elapsed_ms, yes_ask_move, no_ask_move), oldest first
    refreshed_at: float
    summary: dict


class EmpiricalExecutionLab:
    """Persists and summarizes observed short-horizon quote movement.

    Summaries are served from a per-horizon cache that is refreshed at most
    every ``cache_seconds`` and reads only rows added since the last refresh,
    so the 5 Hz fast path never re-scans tens of thousands of samples.
    """

    def __init__(self, path: str, max_samples: int = 50_000, cache_seconds: float = 5.0):
        self.max_samples = int(max_samples)
        self.cache_seconds = float(cache_seconds)
        self._horizon_cache: dict[int, _HorizonCache] = {}
        self._known_horizons: tuple[float, list[int]] | None = None
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
    def _summarize(horizon_ms: int, moves: np.ndarray) -> dict:
        if len(moves) == 0:
            return {
                'horizon_ms': horizon_ms,
                'samples': 0,
                'mean_elapsed_ms': 0.0,
                'mean_yes_ask_move': 0.0,
                'mean_no_ask_move': 0.0,
                'mean_adverse_move': 0.0,
                'p95_adverse_move': 0.0,
                'p99_adverse_move': 0.0,
            }
        elapsed, yes, no = moves[:, 0], moves[:, 1], moves[:, 2]
        # Positive ask movement is adverse to a marketable buyer. Pool YES and
        # NO because the execution engine can trade either side.
        adverse = np.concatenate([np.maximum(yes, 0.0), np.maximum(no, 0.0)])
        p95, p99 = np.percentile(adverse, [95, 99])
        return {
            'horizon_ms': horizon_ms,
            'samples': int(len(moves)),
            'mean_elapsed_ms': float(elapsed.mean()),
            'mean_yes_ask_move': float(yes.mean()),
            'mean_no_ask_move': float(no.mean()),
            'mean_adverse_move': float(adverse.mean()),
            'p95_adverse_move': float(p95),
            'p99_adverse_move': float(p99),
        }

    def horizon_summary(self, horizon_ms: int) -> dict:
        """Summary of the most recent ``max_samples`` moves, at most ``cache_seconds`` old."""
        horizon_ms = int(horizon_ms)
        now = time.monotonic()
        state = self._horizon_cache.get(horizon_ms)
        if state is not None and now - state.refreshed_at < self.cache_seconds:
            return dict(state.summary)

        last_id = state.last_id if state else 0
        moves = state.moves if state else np.empty((0, 3))
        rows = self.db.execute(
            '''SELECT id,elapsed_ms,yes_ask_move,no_ask_move FROM quote_moves
               WHERE horizon_ms=? AND id>? ORDER BY id DESC LIMIT ?''',
            (horizon_ms, last_id, self.max_samples),
        ).fetchall()
        if rows:
            last_id = int(rows[0][0])
            new = np.asarray([row[1:] for row in reversed(rows)], dtype=float)
            moves = np.concatenate([moves, new])[-self.max_samples:]
        summary = self._summarize(horizon_ms, moves)
        self._horizon_cache[horizon_ms] = _HorizonCache(last_id, moves, now, summary)
        return dict(summary)

    def _recorded_horizons(self) -> list[int]:
        now = time.monotonic()
        if self._known_horizons is not None and now - self._known_horizons[0] < self.cache_seconds:
            return self._known_horizons[1]
        rows = self.db.execute(
            'SELECT DISTINCT horizon_ms FROM quote_moves ORDER BY horizon_ms ASC'
        ).fetchall()
        horizons = [int(r[0]) for r in rows]
        self._known_horizons = (now, horizons)
        return horizons

    def summary(self, horizons_ms: list[int]) -> dict:
        horizons = [self.horizon_summary(h) for h in sorted(set(horizons_ms))]
        total = sum(int(item['samples']) for item in horizons)
        return {'total_samples': total, 'horizons': horizons}

    def recommended_adverse_slippage(self, latency_ms: float, min_samples: int = 200) -> dict:
        horizons = self._recorded_horizons()
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
