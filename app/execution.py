from __future__ import annotations

import math

from .config import settings
from .models import ExecutionFill, MarketSnapshot


def _ceil_centicent(value: float) -> float:
    # $0.0001 = one centicent.
    return math.ceil(max(0.0, value) * 10_000.0 - 1e-12) / 10_000.0


def event_contract_fee(quantity: int, price: float, maker: bool = False) -> float:
    """Approximate Kalshi event-contract fee with current general rounding."""
    if quantity <= 0:
        return 0.0
    p = max(0.0, min(1.0, float(price)))
    if maker:
        rate = settings.sim_maker_fee_rate
        multiplier = settings.sim_maker_fee_multiplier
    else:
        rate = settings.sim_taker_fee_rate
        multiplier = settings.sim_taker_fee_multiplier
    position_cost = quantity * p
    raw_fee = multiplier * rate * quantity * p * (1.0 - p)
    rounded_total = _ceil_centicent(position_cost + raw_fee)
    return max(0.0, rounded_total - position_cost)


class ExecutionSimulator:
    """Depth-aware taker execution model for Kalshi binary event contracts."""

    @staticmethod
    def _latency_penalty(wave_score: float) -> float:
        seconds = max(0.0, settings.sim_latency_ms) / 1000.0
        direction_factor = 0.35 + 0.65 * min(1.0, abs(float(wave_score)))
        raw = seconds * max(0.0, settings.sim_latency_price_drift_per_second) * direction_factor
        return min(max(0.0, settings.sim_max_latency_slippage), raw)

    @staticmethod
    def _book(market: MarketSnapshot, side: str, intent: str) -> list[tuple[float, float]]:
        if intent == 'buy' and side == 'yes':
            levels = [(1.0 - p, q) for p, q in market.no_book]
            return sorted(levels, key=lambda x: x[0])
        if intent == 'buy' and side == 'no':
            levels = [(1.0 - p, q) for p, q in market.yes_book]
            return sorted(levels, key=lambda x: x[0])
        if intent == 'sell' and side == 'yes':
            return sorted(market.yes_book, key=lambda x: x[0], reverse=True)
        if intent == 'sell' and side == 'no':
            return sorted(market.no_book, key=lambda x: x[0], reverse=True)
        return []

    def simulate(
        self,
        market: MarketSnapshot,
        side: str,
        intent: str,
        quantity: int,
        wave_score: float = 0.0,
        latency_penalty_override: float | None = None,
    ) -> ExecutionFill:
        requested = max(0, int(quantity))
        if requested < 1:
            return ExecutionFill(
                ticker=market.ticker,
                side=side,
                intent=intent,
                requested_quantity=requested,
                filled_quantity=0,
                latency_ms=settings.sim_latency_ms,
                reason='No quantity requested',
            )

        book = self._book(market, side, intent)
        if not book:
            return ExecutionFill(
                ticker=market.ticker,
                side=side,
                intent=intent,
                requested_quantity=requested,
                filled_quantity=0,
                latency_ms=settings.sim_latency_ms,
                reason='No visible order-book depth',
            )

        top_price = float(book[0][0])
        remaining = requested
        raw_notional = 0.0
        adjusted_notional = 0.0
        fees = 0.0
        fills = 0
        penalty = (
            self._latency_penalty(wave_score)
            if latency_penalty_override is None
            else max(0.0, min(settings.sim_max_latency_slippage, float(latency_penalty_override)))
        )

        for raw_price, raw_qty in book:
            if remaining <= 0:
                break
            available = max(0, int(math.floor(float(raw_qty))))
            if available < 1:
                continue
            take = min(remaining, available)
            adjusted = raw_price + penalty if intent == 'buy' else raw_price - penalty
            adjusted = max(0.01, min(0.99, adjusted))
            raw_notional += take * raw_price
            adjusted_notional += take * adjusted
            fees += event_contract_fee(take, adjusted, maker=False)
            remaining -= take
            fills += 1

        filled = requested - remaining
        if filled < 1:
            return ExecutionFill(
                ticker=market.ticker,
                side=side,
                intent=intent,
                requested_quantity=requested,
                filled_quantity=0,
                top_price=top_price,
                latency_ms=settings.sim_latency_ms,
                reason='Visible depth was below one whole contract',
            )

        raw_average = raw_notional / filled
        average = adjusted_notional / filled
        if intent == 'buy':
            depth_slippage = max(0.0, raw_average - top_price)
            latency_slippage = max(0.0, average - raw_average)
            total_slippage = max(0.0, average - top_price)
        else:
            depth_slippage = max(0.0, top_price - raw_average)
            latency_slippage = max(0.0, raw_average - average)
            total_slippage = max(0.0, top_price - average)

        complete = filled == requested
        fill_fraction = filled / requested
        reason = 'Complete taker fill' if complete else f'Partial taker fill ({filled}/{requested})'
        if latency_penalty_override is not None:
            reason += '; empirical latency penalty'
        if total_slippage > settings.sim_max_execution_slippage:
            reason += '; execution slippage exceeds configured limit'
        if fill_fraction < settings.sim_min_fill_fraction:
            reason += '; fill fraction below configured minimum'

        return ExecutionFill(
            ticker=market.ticker,
            side=side,
            intent=intent,
            requested_quantity=requested,
            filled_quantity=filled,
            average_price=average,
            top_price=top_price,
            fee=fees,
            slippage=total_slippage,
            depth_slippage=depth_slippage,
            latency_slippage=latency_slippage,
            latency_ms=settings.sim_latency_ms,
            levels_consumed=fills,
            complete=complete,
            reason=reason,
        )
