from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from .btc import BtcPriceClient
from .config import settings
from .features import MarketBuffer, PriceBuffer, build_features
from .forecast import BaselineForecaster
from .kalshi import KalshiPublicClient
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
        self.market_buffer = MarketBuffer()
        self.benchmark_stream = CFBenchmarkStream()
        self.forecaster = BaselineForecaster()
        self.risk = RiskEngine()
        self.paper = PaperBroker(settings.paper_starting_cash)
        self.storage = Storage(settings.database_path)
        self.last_state: dict = {
            'status': 'starting',
            'mode': 'paper',
            'live_trading_enabled': settings.live_trading_enabled,
            'warning': 'Paper-only. Authenticated BRTI streaming activates when Kalshi API credentials are configured.',
        }
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        await self.benchmark_stream.start()
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name='trading-engine')

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task
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
                settlements.append(trade)
        return settlements

    async def cycle(self) -> dict:
        market = await self.kalshi.discover_btc_15m()
        if market is None:
            self.last_state = {
                **self.last_state,
                'status': 'waiting_for_market',
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'feed': self.benchmark_stream.status(),
                'error': None,
            }
            return self.last_state

        market = await self.kalshi.enrich_orderbook(market)
        settlements = await self._reconcile_completed_markets(market.ticker)

        btc = await self.btc.get()
        self.buffer.add(btc)
        self.market_buffer.add(market)
        benchmark = self.benchmark_stream.fresh()
        features = build_features(market, btc, self.buffer, self.market_buffer, benchmark)
        forecast = self.forecaster.predict(features)
        self.storage.save_observation(market, features, forecast)

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
        if pos:
            decision = self.risk.evaluate_exit(features, forecast, pos)
            if settings.auto_paper_trade and decision.action == 'EXIT' and decision.price > 0:
                trade = self.paper.close(market.ticker, decision.price, decision.reason)
                self.storage.save_trade(trade)
        else:
            decision = self.risk.evaluate_entry(features, forecast, account)
            if settings.auto_paper_trade and decision.action in ('BUY_YES', 'BUY_NO') and decision.quantity > 0:
                trade = self.paper.open(
                    market.ticker, decision.side, decision.quantity, decision.price, decision.reason
                )
                self.storage.save_trade(trade)

        pos_after = self.paper.positions.get(market.ticker)
        marks_after = {}
        if pos_after:
            marks_after[market.ticker] = features.yes_bid if pos_after.side == 'yes' else features.no_bid
        equity_after = self.paper.mark_equity(marks_after)

        self.last_state = {
            'status': 'running',
            'mode': 'paper',
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'market': market.model_dump(mode='json'),
            'btc': btc.model_dump(mode='json'),
            'benchmark': benchmark.model_dump(mode='json') if benchmark else None,
            'feed': self.benchmark_stream.status(),
            'features': features.model_dump(mode='json'),
            'forecast': forecast.model_dump(mode='json'),
            'decision': decision.model_dump(mode='json'),
            'trade': trade.model_dump(mode='json') if trade else None,
            'settlements': [x.model_dump(mode='json') for x in settlements],
            'account': {
                'starting_cash': self.paper.starting_cash,
                'cash': round(self.paper.cash, 4),
                'equity': round(equity_after, 4),
                'realized_pnl': round(self.paper.realized_pnl, 4),
                'positions': {k: v.model_dump(mode='json') for k, v in self.paper.positions.items()},
                'kill_switch': self.paper.kill_switch,
            },
            'live_trading_enabled': settings.live_trading_enabled,
            'warning': (
                'Using authenticated CF Benchmarks BRTI reference.'
                if benchmark
                else 'No fresh BRTI stream: using Coinbase/Kraken proxy. Final-minute entries are blocked.'
            ),
            'error': None,
        }
        return self.last_state

    def set_kill_switch(self, enabled: bool) -> None:
        self.paper.kill_switch = enabled
