from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import time
from typing import Any

from .ai_sidecar import AISidecarClient
from .config import settings
from .features import build_features
from .models import BtcSnapshot, Decision, MarketSnapshot
from .risk import AccountRiskState


class FastPathController:
    """Low-latency paper decision loop using only already-synced in-memory data.

    Slow REST discovery, spot refresh, settlement reconciliation, analytics, and
    training remain in TradingEngine's normal loop. This path runs at ~5 Hz by
    default and consumes the latest Kalshi WebSocket book plus BRTI snapshot,
    so it does not increase REST request rate.
    """

    ENTRY_ATTEMPT_COOLDOWN_SECONDS = 1.0

    def __init__(self, engine) -> None:
        self.engine = engine
        self.sidecar = AISidecarClient()
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._started_at: datetime | None = None
        self._last_tick_at: datetime | None = None
        self._last_trade_at: datetime | None = None
        self._last_observation_monotonic = 0.0
        self._last_entry_attempt_monotonic = 0.0
        self._last_reference_ts: float | None = None
        self.cycles = 0
        self.valid_cycles = 0
        self.trades = 0
        self.skipped = 0
        self.errors = 0
        self.last_cycle_ms = 0.0
        self.ewma_cycle_ms = 0.0
        self.last_error: str | None = None
        self.last_skip_reason: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(settings.fast_path_enabled)

    async def start(self) -> None:
        if self.enabled and self._task is None:
            self._started_at = datetime.now(timezone.utc)
            self._task = asyncio.create_task(self._loop(), name='fast-decision-path')

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task
        await self.sidecar.close()

    def status(self) -> dict[str, Any]:
        return {
            'enabled': self.enabled,
            'running': bool(self._task and not self._task.done()),
            'target_interval_ms': settings.fast_path_interval_ms,
            'target_hz': 1000.0 / max(settings.fast_path_interval_ms, 1),
            'cycles': self.cycles,
            'valid_cycles': self.valid_cycles,
            'skipped_cycles': self.skipped,
            'trades': self.trades,
            'errors': self.errors,
            'last_cycle_ms': round(self.last_cycle_ms, 3),
            'ewma_cycle_ms': round(self.ewma_cycle_ms, 3),
            'last_tick_at': self._last_tick_at.isoformat() if self._last_tick_at else None,
            'last_trade_at': self._last_trade_at.isoformat() if self._last_trade_at else None,
            'last_skip_reason': self.last_skip_reason,
            'last_error': self.last_error,
            'sidecar': self.sidecar.status(),
        }

    async def _loop(self) -> None:
        interval = max(0.05, settings.fast_path_interval_ms / 1000.0)
        while not self._stop.is_set():
            started = time.perf_counter()
            try:
                await self.tick()
                self.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.errors += 1
                self.last_error = f'{type(exc).__name__}: {exc}'
            elapsed = time.perf_counter() - started
            self.last_cycle_ms = elapsed * 1000.0
            self.ewma_cycle_ms = (
                self.last_cycle_ms if self.ewma_cycle_ms <= 0
                else 0.90 * self.ewma_cycle_ms + 0.10 * self.last_cycle_ms
            )
            delay = max(0.0, interval - elapsed)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

    def _inputs(self) -> tuple[MarketSnapshot, BtcSnapshot] | None:
        state = self.engine.last_state or {}
        raw_market = state.get('market')
        raw_btc = state.get('btc')
        if not raw_market or not raw_btc:
            self.last_skip_reason = 'slow path has not produced a market/BTC snapshot yet'
            return None
        market = MarketSnapshot.model_validate(raw_market)
        btc = BtcSnapshot.model_validate(raw_btc)

        now = datetime.now(timezone.utc)
        btc_age = max(0.0, (now - btc.captured_at).total_seconds())
        if btc_age > settings.fast_path_max_slow_state_age_seconds:
            self.last_skip_reason = f'slow BTC cross-check is stale ({btc_age:.2f}s)'
            return None
        if settings.fast_path_require_stream and not self.engine.orderbook_stream.fresh(market.ticker):
            self.last_skip_reason = 'synchronized Kalshi WebSocket book is not fresh'
            return None

        if market.close_time:
            remaining = max(0.0, (market.close_time - now).total_seconds())
            market = market.model_copy(update={'seconds_remaining': remaining})
        if self.engine.orderbook_stream.fresh(market.ticker):
            market = self.engine.orderbook_stream.apply_to_market(market)
        return market, btc

    async def tick(self) -> dict[str, Any] | None:
        self.cycles += 1
        self._last_tick_at = datetime.now(timezone.utc)
        inputs = self._inputs()
        if inputs is None:
            self.skipped += 1
            return None
        market, btc = inputs
        self.valid_cycles += 1
        self.last_skip_reason = None

        saved_moves = self.engine.empirical.save_moves(self.engine.orderbook_stream.drain_moves())
        self.engine.market_buffer.add(market)
        benchmark = self.engine.benchmark_stream.fresh()
        if benchmark:
            ref_ts = benchmark.captured_at.timestamp()
            if self._last_reference_ts != ref_ts:
                self.engine.reference_buffer.add_benchmark(benchmark)
                self._last_reference_ts = ref_ts

        features = build_features(
            market,
            btc,
            self.engine.buffer,
            self.engine.market_buffer,
            benchmark,
            self.engine.reference_buffer,
        )
        base_forecast = self.engine.forecaster.predict(features)

        history_limit = max(16, int(settings.ai_sidecar_history_points))
        self.sidecar.schedule(
            market,
            features,
            base_forecast,
            spot_history=self.engine.buffer.recent_points(history_limit),
            reference_history=self.engine.reference_buffer.recent_points(history_limit),
            market_history=self.engine.market_buffer.recent_points(history_limit),
        )
        forecast = self.sidecar.blend(base_forecast, market.ticker)

        now_mono = time.monotonic()
        if now_mono - self._last_observation_monotonic >= settings.fast_path_observation_interval_seconds:
            self.engine.storage.save_observation(market, features, forecast)
            self._last_observation_monotonic = now_mono

        data_health = self.engine._data_health(market, btc, benchmark, features)
        latency_model = self.engine._latency_model()
        latency_override = self.engine._latency_override(latency_model)

        marks: dict[str, float] = {}
        pos = self.engine.paper.positions.get(market.ticker)
        if pos:
            marks[market.ticker] = features.yes_bid if pos.side == 'yes' else features.no_bid
        equity = self.engine.paper.mark_equity(marks)
        account = AccountRiskState(
            cash=self.engine.paper.cash,
            equity=equity,
            starting_equity=self.engine.paper.starting_cash,
            daily_pnl=self.engine.paper.realized_pnl,
            kill_switch=self.engine.paper.kill_switch,
        )

        trade = None
        fill = None
        execution_event = None
        if pos:
            decision = self.engine.risk.evaluate_exit(features, forecast, pos)
            requested_decision = decision
            if settings.auto_paper_trade and decision.action == 'EXIT':
                fill = self.engine._exit_fill(
                    market, decision, pos.quantity, features.wave_score, latency_override,
                )
                if fill.filled_quantity > 0:
                    trade = self.engine.paper.close_fill(fill, f'FAST: {decision.reason}')
                    self.engine.storage.save_trade(trade)
                    self.trades += 1
                    self._last_trade_at = datetime.now(timezone.utc)
                else:
                    decision = decision.model_copy(update={
                        'reason': f'{decision.reason}; no visible fast-path exit liquidity',
                        'risk_status': 'blocked',
                    })
                execution_event = self.engine._record_execution_event(
                    fill, requested_decision, decision, forecast,
                )
        else:
            if data_health['status'] == 'blocked':
                decision = Decision(
                    action='HOLD',
                    reason='Fast data-health gate: ' + '; '.join(data_health['blocking']),
                    risk_status='blocked',
                )
            else:
                decision = self.engine.risk.evaluate_entry(features, forecast, account)
            requested_decision = decision
            can_attempt = now_mono - self._last_entry_attempt_monotonic >= self.ENTRY_ATTEMPT_COOLDOWN_SECONDS
            if (
                settings.auto_paper_trade
                and can_attempt
                and decision.action in ('BUY_YES', 'BUY_NO')
                and decision.quantity > 0
            ):
                self._last_entry_attempt_monotonic = now_mono
                decision, fill = self.engine._entry_fill(
                    market, decision, forecast, features.wave_score, latency_override,
                )
                execution_event = self.engine._record_execution_event(
                    fill, requested_decision, decision, forecast,
                ) if fill else None
                if decision.action in ('BUY_YES', 'BUY_NO') and fill and fill.filled_quantity > 0:
                    trade = self.engine.paper.open_fill(fill, f'FAST: {decision.reason}')
                    self.engine.storage.save_trade(trade)
                    self.trades += 1
                    self._last_trade_at = datetime.now(timezone.utc)

        pos_after = self.engine.paper.positions.get(market.ticker)
        marks_after: dict[str, float] = {}
        if pos_after:
            marks_after[market.ticker] = features.yes_bid if pos_after.side == 'yes' else features.no_bid
        equity_after = self.engine.paper.mark_equity(marks_after)
        if trade or now_mono - self._last_observation_monotonic < 0.05:
            self.engine._persist_paper()

        self.engine.last_state = {
            **self.engine.last_state,
            'status': 'running',
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'market': market.model_dump(mode='json'),
            'btc': btc.model_dump(mode='json'),
            'benchmark': benchmark.model_dump(mode='json') if benchmark else None,
            'features': features.model_dump(mode='json'),
            'forecast': forecast.model_dump(mode='json'),
            'decision': decision.model_dump(mode='json'),
            'execution_fill': fill.model_dump(mode='json') if fill else None,
            'execution_event': execution_event.model_dump(mode='json') if execution_event else None,
            'trade': trade.model_dump(mode='json') if trade else None,
            'data_health': data_health,
            'orderbook_stream': self.engine.orderbook_stream.status(),
            'saved_quote_moves_fast_cycle': saved_moves,
            'fast_path': self.status(),
            'ai_sidecar': self.sidecar.status(),
            'account': {
                'starting_cash': self.engine.paper.starting_cash,
                'cash': round(self.engine.paper.cash, 4),
                'equity': round(equity_after, 4),
                'realized_pnl': round(self.engine.paper.realized_pnl, 4),
                'total_fees': round(self.engine.paper.total_fees, 4),
                'positions': {k: v.model_dump(mode='json') for k, v in self.engine.paper.positions.items()},
                'kill_switch': self.engine.paper.kill_switch,
                'persisted': True,
            },
            'error': None,
        }
        return self.engine.last_state
