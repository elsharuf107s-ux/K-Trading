from __future__ import annotations

from dataclasses import dataclass
from .config import settings
from .models import Decision, FeatureVector, Forecast, PaperPosition


@dataclass
class AccountRiskState:
    cash: float
    equity: float
    starting_equity: float
    daily_pnl: float
    kill_switch: bool = False


class RiskEngine:
    def evaluate_entry(self, f: FeatureVector, forecast: Forecast, account: AccountRiskState) -> Decision:
        if account.kill_switch:
            return Decision(action='HOLD', reason='Kill switch is active', risk_status='blocked')
        if settings.live_trading_enabled:
            return Decision(action='HOLD', reason='V1 refuses live execution; paper mode only', risk_status='blocked')
        if f.seconds_remaining < settings.min_seconds_to_expiry:
            return Decision(action='HOLD', reason='Too close to expiry', risk_status='blocked')
        if f.spread > settings.max_spread:
            return Decision(action='HOLD', reason=f'Spread {f.spread:.1%} exceeds limit', risk_status='blocked')
        if account.daily_pnl <= -abs(settings.daily_loss_limit_usd):
            return Decision(action='HOLD', reason='Daily loss limit reached', risk_status='blocked')
        drawdown = 1 - account.equity / max(account.starting_equity, 1e-9)
        if drawdown >= settings.max_drawdown_fraction:
            return Decision(action='HOLD', reason='Maximum account drawdown reached', risk_status='blocked')

        yes_gross = forecast.yes_probability - f.yes_ask
        no_gross = (1 - forecast.yes_probability) - f.no_ask
        friction = settings.estimated_cost_buffer + settings.uncertainty_buffer
        yes_edge = yes_gross - friction
        no_edge = no_gross - friction

        if max(yes_edge, no_edge) < settings.min_usable_edge:
            return Decision(action='HOLD', usable_edge=max(yes_edge, no_edge), gross_edge=max(yes_gross, no_gross), reason='No sufficient after-cost edge', risk_status='approved')

        side = 'yes' if yes_edge >= no_edge else 'no'
        price = f.yes_ask if side == 'yes' else f.no_ask
        usable = yes_edge if side == 'yes' else no_edge
        gross = yes_gross if side == 'yes' else no_gross
        budget = min(settings.max_position_usd, account.equity * settings.max_bankroll_fraction, account.cash)
        quantity = int(budget / max(price, 0.01))
        if quantity < 1:
            return Decision(action='HOLD', reason='Position budget too small', risk_status='blocked')
        return Decision(
            action='BUY_YES' if side == 'yes' else 'BUY_NO', side=side, usable_edge=usable,
            gross_edge=gross, price=price, quantity=quantity,
            reason=f'After-cost edge {usable:.1%} exceeds threshold', risk_status='approved'
        )

    def evaluate_exit(self, f: FeatureVector, forecast: Forecast, pos: PaperPosition) -> Decision:
        bid = f.yes_bid if pos.side == 'yes' else f.no_bid
        fair = forecast.yes_probability if pos.side == 'yes' else 1 - forecast.yes_probability
        remaining_edge = fair - bid - settings.estimated_cost_buffer
        trade_return = bid / max(pos.entry_price, 1e-9) - 1
        if trade_return <= -settings.max_trade_loss_fraction:
            return Decision(action='EXIT', side=pos.side, price=bid, reason='Per-trade stop loss', risk_status='approved')
        if remaining_edge <= settings.exit_edge:
            return Decision(action='EXIT', side=pos.side, price=bid, reason='Forecast edge has decayed', risk_status='approved')
        if f.seconds_remaining < settings.min_seconds_to_expiry:
            return Decision(action='EXIT', side=pos.side, price=bid, reason='Time exit before settlement window', risk_status='approved')
        return Decision(action='HOLD', side=pos.side, price=bid, usable_edge=remaining_edge, reason='Position thesis still active', risk_status='approved')
