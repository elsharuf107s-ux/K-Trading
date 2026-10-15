from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from .btc import BtcPriceClient
from .config import settings
from .execution import ExecutionSimulator
from .features import MarketBuffer, PriceBuffer, ReferenceBuffer, build_features
from .forecast import BaselineForecaster
from .kalshi import KalshiPublicClient
from .model_lab import AdaptiveForecaster, ModelRegistry
from .models import Decision, ExecutionEvent, ExecutionFill
from .paper import PaperBroker
from .risk import AccountRiskState, RiskEngine
from .storage import Storage
from .stream import CFBenchmarkStream


class TradingEngine:
    def __init__(self):
        settings.ensure_dirs()
        self.kalshi = KalshiPublicClient()
        self.btc = BtcPriceClient()
        self.buffer = PriceBuffer()
        self.reference_buffer = ReferenceBuffer()
        self.market_buffer = MarketBuffer()
        self.benchmark_stream = CFBenchmarkStream()
        self.model_registry = ModelRegistry()
        self.forecaster = AdaptiveForecaster(BaselineForecaster(), self.model_registry)
        self.risk = RiskEngine()
        self.execution = ExecutionSimulator()
        self.storage = Storage(settings.database_path)
        self.paper = PaperBroker(settings.paper_starting_cash)
        self.paper.restore(self.storage.load_paper_state())
        self.last_state: dict = {
            'status': 'starting',
            'mode': 'paper',
            'live_trading_enabled': settings.live_trading_enabled,
            'model_registry': self.model_registry.status(),
            'execution': {'mode': 'depth_aware_taker_sim'},
            'warning': 'Paper-only. Authenticated BRTI streaming activates when Kalshi API credentials are configured.',
        }
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    def _persist_paper(self) -> None:
        self.storage.save_paper_state(self.paper.snapshot())

    async def start(self) -> None:
        await self.benchmark_stream.start()
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name='trading-engine')

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task
        self._persist_paper()
        await self.benchmark_stream.stop()
        await self.kalshi.close()
        await self.btc.close()

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.cycle()
            except Exception as exc:
                self.last_state = {
                    **self.last_state,
                    'status': 'error',
                    'error': str(exc),
                    'error_type': type(exc).__name__,
                    'updated_at': datetime.now(timezone.utc).isoformat(),
                }
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=settings.engine_interval_seconds)
            except asyncio.TimeoutError:
                pass

    async def _reconcile_completed_markets(self, current_ticker: str) -> list:
        settlements = []
        candidates = set(self.storage.unresolved_tickers(current_ticker, limit=8))
        candidates.update(t for t in self.paper.positions if t != current_ticker)
        for ticker in candidates:
            try:
                raw = await self.kalshi.get_market(ticker)
            except Exception:
                continue
            result = str(raw.get('result') or '').lower()
            if result not in ('yes', 'no'):
                continue
            self.storage.save_market_result(ticker, result)
            if ticker in self.paper.positions:
                trade = self.paper.settle(ticker, result)
                self.storage.save_trade(trade)
                self._persist_paper()
                settlements.append(trade)
        return settlements

    def _data_health(self, market, btc, benchmark, features) -> dict:
        now = datetime.now(timezone.utc)
        market_age = max(0.0, (now - market.captured_at).total_seconds())
        benchmark_age = None
        if benchmark is not None:
            benchmark_age = max(0.0, (now - benchmark.captured_at).total_seconds())
        book_ok = bool(market.yes_book and market.no_book)
        source_count = len(btc.sources)
        issues: list[str] = []
        blocking: list[str] = []

        if market_age > settings.max_market_snapshot_age_seconds:
            blocking.append(f'market snapshot age {market_age:.1f}s')
        if settings.require_orderbook_for_entry and not book_ok:
            blocking.append('missing executable YES/NO book depth')
        if features.source_dispersion > settings.max_source_dispersion:
            blocking.append(f'spot-source dispersion {features.source_dispersion:.2%}')
        if benchmark is None:
            issues.append('BRTI unavailable; proxy reference mode')
        elif benchmark_age is not None and benchmark_age > settings.benchmark_stale_seconds:
            issues.append(f'BRTI age {benchmark_age:.1f}s')
        if source_count < 2:
            issues.append('only one public BTC spot source available')

        status = 'blocked' if blocking else 'degraded' if issues else 'healthy'
        return {
            'status': status,
            'blocking': blocking,
            'issues': issues,
            'market_age_seconds': market_age,
            'benchmark_age_seconds': benchmark_age,
            'book_ok': book_ok,
            'spot_source_count': source_count,
            'source_dispersion': features.source_dispersion,
        }

    def _entry_fill(self, market, decision: Decision, forecast, wave_score: float) -> tuple[Decision, ExecutionFill | None]:
        requested = decision.quantity
        fill: ExecutionFill | None = None
        while requested > 0:
            candidate = self.execution.simulate(
                market, decision.side, 'buy', requested, wave_score=wave_score,
            )
            if candidate.filled_quantity < 1:
                return Decision(
                    action='HOLD', side=decision.side, usable_edge=decision.usable_edge,
                    reason='Execution simulator found no visible fillable depth', risk_status='blocked',
                ), candidate
            total_cost = candidate.filled_quantity * candidate.average_price + candidate.fee
            if total_cost <= self.paper.cash + 1e-9:
                fill = candidate
                break
            requested -= 1

        if fill is None or fill.filled_quantity < 1:
            return Decision(
                action='HOLD', side=decision.side,
                reason='Simulated execution does not fit available cash after fees', risk_status='blocked',
            ), fill

        fill_fraction = fill.filled_quantity / max(fill.requested_quantity, 1)
        if fill_fraction < settings.sim_min_fill_fraction:
            return Decision(
                action='HOLD', side=decision.side,
                reason=f'Only {fill_fraction:.0%} of desired contracts are visible at current depth', risk_status='blocked',
            ), fill
        if fill.slippage > settings.sim_max_execution_slippage:
            return Decision(
                action='HOLD', side=decision.side,
                reason=f'Simulated slippage {fill.slippage:.1%} exceeds execution limit', risk_status='blocked',
            ), fill

        fair = forecast.yes_probability if decision.side == 'yes' else 1.0 - forecast.yes_probability
        fee_per_contract = fill.fee / max(fill.filled_quantity, 1)
        realized_entry_edge = fair - fill.average_price - fee_per_contract - settings.uncertainty_buffer
        if realized_entry_edge < settings.min_usable_edge:
            return Decision(
                action='HOLD', side=decision.side,
                usable_edge=realized_entry_edge,
                gross_edge=fair - fill.average_price,
                price=fill.average_price,
                quantity=fill.filled_quantity,
                reason='Depth/fees/latency erased the minimum executable edge',
                risk_status='blocked',
            ), fill

        return decision.model_copy(update={
            'price': fill.average_price,
            'quantity': fill.filled_quantity,
            'usable_edge': realized_entry_edge,
            'reason': f'{decision.reason}; executable edge {realized_entry_edge:.1%}',
        }), fill

    def _exit_fill(self, market, decision: Decision, quantity: int, wave_score: float) -> ExecutionFill:
        return self.execution.simulate(market, decision.side, 'sell', quantity, wave_score=wave_score)

    def _record_execution_event(
        self,
        fill: ExecutionFill,
        requested_decision: Decision,
        final_decision: Decision,
        forecast,
    ) -> ExecutionEvent:
        fair = forecast.yes_probability if fill.side == 'yes' else 1.0 - forecast.yes_probability
        if fill.intent == 'buy':
            gross_before = fair - (fill.top_price or requested_decision.price)
            executable_after = (
                fair - fill.average_price - fill.fee / max(fill.filled_quantity, 1) - settings.uncertainty_buffer
                if fill.filled_quantity > 0 else 0.0
            )
        else:
            gross_before = (fill.top_price or requested_decision.price) - fair
            executable_after = (
                fill.average_price - fair - fill.fee / max(fill.filled_quantity, 1)
                if fill.filled_quantity > 0 else 0.0
            )

        if fill.filled_quantity < 1:
            status = 'no_liquidity'
        elif final_decision.action == 'HOLD':
            status = 'rejected'
        elif fill.complete:
            status = 'filled'
        else:
            status = 'partial'

        event = ExecutionEvent(
            ticker=fill.ticker,
            side=fill.side,
            intent=fill.intent,
            status=status,
            requested_quantity=fill.requested_quantity,
            filled_quantity=fill.filled_quantity,
            fair_probability=fair,
            top_price=fill.top_price,
            average_price=fill.average_price,
            gross_edge_before=gross_before,
            executable_edge_after=executable_after,
            fee=fill.fee,
            depth_slippage=fill.depth_slippage,
            latency_slippage=fill.latency_slippage,
            total_slippage=fill.slippage,
            reason=final_decision.reason,
        )
        self.storage.save_execution_event(event)
        return event

    async def cycle(self) -> dict:
        market = await self.kalshi.discover_btc_15m()
        if market is None:
            self.last_state = {
                **self.last_state,
                'status': 'waiting_for_market',
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'feed': self.benchmark_stream.status(),
                'model_registry': self.model_registry.status(),
                'error': None,
            }
            return self.last_state

        market = await self.kalshi.enrich_orderbook(market)
        settlements = await self._reconcile_completed_markets(market.ticker)

        btc = await self.btc.get()
        self.buffer.add(btc)
        self.market_buffer.add(market)
        benchmark = self.benchmark_stream.fresh()
        if benchmark:
            self.reference_buffer.add_benchmark(benchmark)

        features = build_features(
            market, btc, self.buffer, self.market_buffer, benchmark, self.reference_buffer,
        )
        forecast = self.forecaster.predict(features)
        self.storage.save_observation(market, features, forecast)
        data_health = self._data_health(market, btc, benchmark, features)

        marks = {}
        pos = self.paper.positions.get(market.ticker)
        if pos:
            marks[market.ticker] = features.yes_bid if pos.side == 'yes' else features.no_bid
        equity = self.paper.mark_equity(marks)
        account = AccountRiskState(
            cash=self.paper.cash,
            equity=equity,
            starting_equity=self.paper.starting_cash,
            daily_pnl=self.paper.realized_pnl,
            kill_switch=self.paper.kill_switch,
        )

        trade = None
        fill = None
        execution_event = None
        if pos:
            decision = self.risk.evaluate_exit(features, forecast, pos)
            requested_decision = decision
            if settings.auto_paper_trade and decision.action == 'EXIT':
                fill = self._exit_fill(market, decision, pos.quantity, features.wave_score)
                if fill.filled_quantity > 0:
                    trade = self.paper.close_fill(fill, decision.reason)
                    self.storage.save_trade(trade)
                else:
                    decision = decision.model_copy(update={
                        'reason': f'{decision.reason}; no visible exit liquidity',
                        'risk_status': 'blocked',
                    })
                execution_event = self._record_execution_event(fill, requested_decision, decision, forecast)
        else:
            if data_health['status'] == 'blocked':
                decision = Decision(
                    action='HOLD',
                    reason='Data-health gate: ' + '; '.join(data_health['blocking']),
                    risk_status='blocked',
                )
            else:
                decision = self.risk.evaluate_entry(features, forecast, account)
            requested_decision = decision
            if settings.auto_paper_trade and decision.action in ('BUY_YES', 'BUY_NO') and decision.quantity > 0:
                decision, fill = self._entry_fill(market, decision, forecast, features.wave_score)
                execution_event = self._record_execution_event(fill, requested_decision, decision, forecast) if fill else None
                if decision.action in ('BUY_YES', 'BUY_NO') and fill and fill.filled_quantity > 0:
                    trade = self.paper.open_fill(fill, decision.reason)
                    self.storage.save_trade(trade)

        pos_after = self.paper.positions.get(market.ticker)
        marks_after = {}
        if pos_after:
            marks_after[market.ticker] = features.yes_bid if pos_after.side == 'yes' else features.no_bid
        equity_after = self.paper.mark_equity(marks_after)
        self._persist_paper()

        self.last_state = {
            'status': 'running',
            'mode': 'paper',
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'market': market.model_dump(mode='json'),
            'btc': btc.model_dump(mode='json'),
            'benchmark': benchmark.model_dump(mode='json') if benchmark else None,
            'feed': self.benchmark_stream.status(),
            'data_health': data_health,
            'features': features.model_dump(mode='json'),
            'forecast': forecast.model_dump(mode='json'),
            'decision': decision.model_dump(mode='json'),
            'execution_fill': fill.model_dump(mode='json') if fill else None,
            'execution_event': execution_event.model_dump(mode='json') if execution_event else None,
            'execution_metrics': self.storage.execution_metrics(),
            'execution_attribution': self.storage.execution_attribution(),
            'trade': trade.model_dump(mode='json') if trade else None,
            'settlements': [x.model_dump(mode='json') for x in settlements],
            'model_registry': self.model_registry.status(),
            'account': {
                'starting_cash': self.paper.starting_cash,
                'cash': round(self.paper.cash, 4),
                'equity': round(equity_after, 4),
                'realized_pnl': round(self.paper.realized_pnl, 4),
                'total_fees': round(self.paper.total_fees, 4),
                'positions': {k: v.model_dump(mode='json') for k, v in self.paper.positions.items()},
                'kill_switch': self.paper.kill_switch,
                'persisted': True,
            },
            'live_trading_enabled': settings.live_trading_enabled,
            'warning': (
                'Using authenticated CF Benchmarks BRTI. Paper fills include depth, current fee settings and latency attribution.'
                if benchmark
                else 'No fresh BRTI stream: proxy mode. Final-minute entries are blocked. Paper fills include depth, current fee settings and latency attribution.'
            ),
            'error': None,
        }
        return self.last_state

    def set_kill_switch(self, enabled: bool) -> None:
        self.paper.kill_switch = enabled
        self._persist_paper()
