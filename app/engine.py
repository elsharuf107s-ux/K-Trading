from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from .btc import BtcPriceClient
from .config import settings
from .features import PriceBuffer, build_features
from .forecast import BaselineForecaster
from .kalshi import KalshiPublicClient
from .models import Decision
from .paper import PaperBroker
from .risk import AccountRiskState, RiskEngine
from .storage import Storage


class TradingEngine:
    def __init__(self):
        settings.ensure_dirs()
        self.kalshi = KalshiPublicClient()
        self.btc = BtcPriceClient()
        self.buffer = PriceBuffer()
        self.forecaster = BaselineForecaster()
        self.risk = RiskEngine()
        self.paper = PaperBroker(settings.paper_starting_cash)
        self.storage = Storage(settings.database_path)
        self.last_state: dict = {
            'status': 'starting',
            'mode': 'paper',
            'live_trading_enabled': settings.live_trading_enabled,
            'warning': 'V1 is paper-only. BTC spot feeds are proxies; Kalshi settlement rules control.',
        }
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name='trading-engine')

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task
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

    async def cycle(self) -> dict:
        market = await self.kalshi.discover_btc_15m()
        if market is None:
            self.last_state = {
                **self.last_state,
                'status': 'waiting_for_market',
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'error': None,
            }
            return self.last_state

        btc = await self.btc.get()
        self.buffer.add(btc)
        features = build_features(market, btc, self.buffer)
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
                trade = self.paper.open(market.ticker, decision.side, decision.quantity, decision.price, decision.reason)
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
            'features': features.model_dump(mode='json'),
            'forecast': forecast.model_dump(mode='json'),
            'decision': decision.model_dump(mode='json'),
            'trade': trade.model_dump(mode='json') if trade else None,
            'account': {
                'starting_cash': self.paper.starting_cash,
                'cash': round(self.paper.cash, 4),
                'equity': round(equity_after, 4),
                'realized_pnl': round(self.paper.realized_pnl, 4),
                'positions': {k: v.model_dump(mode='json') for k, v in self.paper.positions.items()},
                'kill_switch': self.paper.kill_switch,
            },
            'live_trading_enabled': settings.live_trading_enabled,
            'warning': 'Paper mode only. Spot feeds are features, not the official settlement value.',
            'error': None,
        }
        return self.last_state

    def set_kill_switch(self, enabled: bool) -> None:
        self.paper.kill_switch = enabled
