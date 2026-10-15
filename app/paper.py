from __future__ import annotations

from typing import Any

from .models import ExecutionFill, PaperPosition, TradeRecord


class PaperBroker:
    def __init__(self, starting_cash: float):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions: dict[str, PaperPosition] = {}
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.trades: list[TradeRecord] = []
        self.kill_switch = False

    def snapshot(self) -> dict[str, Any]:
        return {
            'starting_cash': self.starting_cash,
            'cash': self.cash,
            'realized_pnl': self.realized_pnl,
            'total_fees': self.total_fees,
            'kill_switch': self.kill_switch,
            'positions': {
                ticker: position.model_dump(mode='json')
                for ticker, position in self.positions.items()
            },
        }

    def restore(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        self.starting_cash = float(state.get('starting_cash', self.starting_cash))
        self.cash = float(state.get('cash', self.starting_cash))
        self.realized_pnl = float(state.get('realized_pnl', 0.0))
        self.total_fees = float(state.get('total_fees', 0.0))
        self.kill_switch = bool(state.get('kill_switch', False))
        restored: dict[str, PaperPosition] = {}
        for ticker, raw in (state.get('positions') or {}).items():
            try:
                position = PaperPosition.model_validate(raw)
            except Exception:
                continue
            restored[ticker] = position
        self.positions = restored

    def open_fill(self, fill: ExecutionFill, reason: str) -> TradeRecord:
        if fill.intent != 'buy':
            raise ValueError('Opening fill must be a buy')
        if fill.filled_quantity < 1:
            raise ValueError('Cannot open an empty fill')
        if fill.ticker in self.positions:
            raise ValueError('Position already exists for market')
        cost = fill.filled_quantity * fill.average_price + fill.fee
        if cost > self.cash + 1e-9:
            raise ValueError('Insufficient paper cash after simulated fee')
        self.cash -= cost
        self.total_fees += fill.fee
        self.positions[fill.ticker] = PaperPosition(
            ticker=fill.ticker,
            side=fill.side,
            quantity=fill.filled_quantity,
            entry_price=fill.average_price,
            entry_fee=fill.fee,
            peak_mark=fill.average_price,
        )
        trade = TradeRecord(
            ticker=fill.ticker,
            side=fill.side,
            action='open',
            quantity=fill.filled_quantity,
            requested_quantity=fill.requested_quantity,
            price=fill.average_price,
            fee=fill.fee,
            slippage=fill.slippage,
            reason=f'{reason}; {fill.reason}',
        )
        self.trades.append(trade)
        return trade

    def close_fill(self, fill: ExecutionFill, reason: str) -> TradeRecord:
        if fill.intent != 'sell':
            raise ValueError('Closing fill must be a sell')
        if fill.filled_quantity < 1:
            raise ValueError('Cannot close with an empty fill')
        pos = self.positions.get(fill.ticker)
        if pos is None:
            raise ValueError('No open paper position for market')
        if pos.side != fill.side:
            raise ValueError('Execution side does not match paper position')

        quantity = min(pos.quantity, fill.filled_quantity)
        if quantity < 1:
            raise ValueError('No contracts available to close')
        # If the execution fill is larger than the remaining position, scale fee
        # to the actually closed quantity. Normally the simulator is called with
        # exactly pos.quantity, but this keeps accounting defensive.
        exit_fee = fill.fee * (quantity / max(fill.filled_quantity, 1))
        entry_fee_alloc = pos.entry_fee * (quantity / max(pos.quantity, 1))
        gross_proceeds = quantity * fill.average_price
        cash_proceeds = gross_proceeds - exit_fee
        basis = quantity * pos.entry_price
        pnl = gross_proceeds - basis - entry_fee_alloc - exit_fee

        self.cash += cash_proceeds
        self.realized_pnl += pnl
        self.total_fees += exit_fee
        pos.quantity -= quantity
        pos.entry_fee = max(0.0, pos.entry_fee - entry_fee_alloc)
        if pos.quantity <= 0:
            self.positions.pop(fill.ticker, None)

        trade = TradeRecord(
            ticker=fill.ticker,
            side=fill.side,
            action='close',
            quantity=quantity,
            requested_quantity=fill.requested_quantity,
            price=fill.average_price,
            pnl=pnl,
            fee=exit_fee,
            slippage=fill.slippage,
            reason=f'{reason}; {fill.reason}',
        )
        self.trades.append(trade)
        return trade

    # Backward-compatible helpers used by simple unit tests and manual research.
    def open(self, ticker: str, side: str, quantity: int, price: float, reason: str) -> TradeRecord:
        return self.open_fill(
            ExecutionFill(
                ticker=ticker,
                side=side,
                intent='buy',
                requested_quantity=quantity,
                filled_quantity=quantity,
                average_price=price,
                top_price=price,
                complete=True,
                reason='Legacy zero-fee direct fill',
            ),
            reason,
        )

    def close(self, ticker: str, price: float, reason: str) -> TradeRecord:
        pos = self.positions[ticker]
        return self.close_fill(
            ExecutionFill(
                ticker=ticker,
                side=pos.side,
                intent='sell',
                requested_quantity=pos.quantity,
                filled_quantity=pos.quantity,
                average_price=price,
                top_price=price,
                complete=True,
                reason='Legacy zero-fee direct fill',
            ),
            reason,
        )

    def settle(self, ticker: str, result: str, reason: str = 'Kalshi market settled') -> TradeRecord:
        pos = self.positions.pop(ticker)
        price = 1.0 if result.lower() == pos.side else 0.0
        proceeds = pos.quantity * price
        cost = pos.quantity * pos.entry_price
        pnl = proceeds - cost - pos.entry_fee
        self.cash += proceeds
        self.realized_pnl += pnl
        trade = TradeRecord(
            ticker=ticker,
            side=pos.side,
            action='settle',
            quantity=pos.quantity,
            requested_quantity=pos.quantity,
            price=price,
            pnl=pnl,
            fee=0.0,
            reason=reason,
        )
        self.trades.append(trade)
        return trade

    def mark_equity(self, marks: dict[str, float]) -> float:
        value = self.cash
        for ticker, pos in self.positions.items():
            mark = marks.get(ticker, pos.entry_price)
            pos.peak_mark = max(pos.peak_mark, mark)
            value += pos.quantity * mark
        return value
