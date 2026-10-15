from __future__ import annotations

import math

from .config import settings
from .models import ExecutionFill, MarketSnapshot


def _ceil_cent(value: float) -> float:
    return math.ceil(max(0.0, value) * 100.0 - 1e-12) / 100.0


def event_contract_fee(quantity: int, price: float, maker: bool = False) -> float:
    """Approximate Kalshi event-contract transaction fee using configured rate.

    The research build intentionally keeps the rate configurable because fee
    schedules and product exceptions can change. Rounding is applied to the
    order-level fee in cents, matching the published general schedule formula.
    """
    if quantity <= 0:
        return 0.0
    p = max(0.0, min(1.0, float(price)))
    rate = settings.sim_maker_fee_rate if maker else settings.sim_taker_fee_rate
    return _ceil_cent(rate * quantity * p * (1.0 - p))


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
        # Kalshi exposes YES bids and NO bids. Buying YES consumes NO bids at
        # complementary prices; buying NO consumes YES bids. Selling a held
        # side consumes that same side's bids directly.
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
        notional = 0.0
        fills = 0
        penalty = self._latency_penalty(wave_score)

        for raw_price, raw_qty in book:
            if remaining <= 0:
                break
            available = max(0, int(math.floor(float(raw_qty))))
            if available < 1:
                continue
            take = min(remaining, available)
            adjusted = raw_price + penalty if intent == 'buy' else raw_price - penalty
            adjusted = max(0.01, min(0.99, adjusted))
            notional += take * adjusted
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

        average = notional / filled
        # Positive slippage means worse execution than the displayed top quote.
        slippage = average - top_price if intent == 'buy' else top_price - average
        fee = event_contract_fee(filled, average, maker=False)
        complete = filled == requested
        fill_fraction = filled / requested

        reason = 'Complete taker fill' if complete else f'Partial taker fill ({filled}/{requested})'
        if slippage > settings.sim_max_execution_slippage:
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
            fee=fee,
            slippage=max(0.0, slippage),
            latency_ms=settings.sim_latency_ms,
            levels_consumed=fills,
            complete=complete,
            reason=reason,
        )
