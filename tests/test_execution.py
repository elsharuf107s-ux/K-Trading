import sqlite3

from app.config import settings
from app.execution import ExecutionSimulator, event_contract_fee
from app.models import ExecutionFill, MarketSnapshot, TradeRecord
from app.paper import PaperBroker
from app.storage import Storage


def book_market(**overrides):
    base = dict(
        ticker='KXBTC15M-EXEC',
        yes_bid=0.42,
        yes_ask=0.53,
        no_bid=0.47,
        no_ask=0.58,
        last_price=0.50,
        seconds_remaining=600,
        yes_book=[(0.40, 3), (0.42, 5)],
        no_book=[(0.45, 2), (0.47, 4)],
    )
    base.update(overrides)
    return MarketSnapshot(**base)


def test_general_fee_rounds_up_to_cent(monkeypatch):
    monkeypatch.setattr(settings, 'sim_taker_fee_rate', 0.07)
    assert event_contract_fee(1, 0.50) == 0.02
    assert event_contract_fee(100, 0.50) == 1.75
    assert event_contract_fee(10, 0.40) == 0.17


def test_buy_yes_walks_complementary_no_book(monkeypatch):
    monkeypatch.setattr(settings, 'sim_latency_ms', 0.0)
    sim = ExecutionSimulator()
    fill = sim.simulate(book_market(), 'yes', 'buy', 6, wave_score=0.8)
    # Buy YES consumes NO bids: 4 @ YES 0.53, then 2 @ YES 0.55.
    assert fill.complete is True
    assert fill.filled_quantity == 6
    assert fill.levels_consumed == 2
    assert round(fill.top_price, 4) == 0.53
    assert round(fill.average_price, 4) == round((4 * 0.53 + 2 * 0.55) / 6, 4)
    assert fill.slippage > 0
    assert fill.fee > 0


def test_partial_fill_when_visible_depth_is_insufficient(monkeypatch):
    monkeypatch.setattr(settings, 'sim_latency_ms', 0.0)
    sim = ExecutionSimulator()
    fill = sim.simulate(book_market(), 'yes', 'buy', 10)
    assert fill.complete is False
    assert fill.requested_quantity == 10
    assert fill.filled_quantity == 6
    assert 'Partial' in fill.reason


def test_sell_yes_walks_yes_bids(monkeypatch):
    monkeypatch.setattr(settings, 'sim_latency_ms', 0.0)
    sim = ExecutionSimulator()
    fill = sim.simulate(book_market(), 'yes', 'sell', 7)
    # Sell YES hits 5 @ .42, then 2 @ .40.
    assert fill.filled_quantity == 7
    assert round(fill.average_price, 4) == round((5 * 0.42 + 2 * 0.40) / 7, 4)
    assert fill.top_price == 0.42
    assert fill.slippage > 0


def test_fee_aware_round_trip_accounting():
    broker = PaperBroker(1000)
    opening = ExecutionFill(
        ticker='T', side='yes', intent='buy', requested_quantity=10, filled_quantity=10,
        average_price=0.40, top_price=0.40, fee=0.17, complete=True,
    )
    broker.open_fill(opening, 'open')
    assert round(broker.cash, 2) == 995.83

    closing = ExecutionFill(
        ticker='T', side='yes', intent='sell', requested_quantity=10, filled_quantity=10,
        average_price=0.60, top_price=0.60, fee=0.17, complete=True,
    )
    trade = broker.close_fill(closing, 'close')
    assert round(trade.pnl, 2) == 1.66
    assert round(broker.cash, 2) == 1001.66
    assert round(broker.total_fees, 2) == 0.34


def test_partial_exit_preserves_remaining_cost_basis():
    broker = PaperBroker(1000)
    broker.open_fill(
        ExecutionFill(
            ticker='T', side='yes', intent='buy', requested_quantity=10, filled_quantity=10,
            average_price=0.40, top_price=0.40, fee=0.17, complete=True,
        ),
        'open',
    )
    trade = broker.close_fill(
        ExecutionFill(
            ticker='T', side='yes', intent='sell', requested_quantity=10, filled_quantity=4,
            average_price=0.60, top_price=0.60, fee=0.07, complete=False,
        ),
        'partial close',
    )
    assert round(trade.pnl, 3) == 0.662
    assert broker.positions['T'].quantity == 6
    assert round(broker.positions['T'].entry_fee, 3) == 0.102


def test_storage_migrates_old_trade_schema(tmp_path):
    path = tmp_path / 'old.sqlite3'
    db = sqlite3.connect(path)
    db.execute('''CREATE TABLE trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,created_at TEXT NOT NULL,ticker TEXT NOT NULL,side TEXT NOT NULL,
        action TEXT NOT NULL,quantity INTEGER NOT NULL,price REAL NOT NULL,pnl REAL NOT NULL,reason TEXT NOT NULL
    )''')
    db.commit()
    db.close()

    storage = Storage(str(path))
    columns = {row[1] for row in storage.db.execute('PRAGMA table_info(trades)').fetchall()}
    assert {'fee', 'requested_quantity', 'slippage'} <= columns

    storage.save_trade(TradeRecord(
        ticker='T', side='yes', action='open', quantity=2, requested_quantity=3,
        price=0.51, fee=0.02, slippage=0.01, reason='test',
    ))
    row = storage.recent_trades(1)[0]
    assert row['fee'] == 0.02
    assert row['requested_quantity'] == 3
    assert row['slippage'] == 0.01


def test_execution_metrics_count_partial_fills(tmp_path):
    storage = Storage(str(tmp_path / 'metrics.sqlite3'))
    storage.save_trade(TradeRecord(
        ticker='T', side='yes', action='open', quantity=4, requested_quantity=10,
        price=0.55, fee=0.08, slippage=0.02, reason='partial',
    ))
    metrics = storage.execution_metrics()
    assert metrics['fills'] == 1
    assert metrics['filled_contracts'] == 4
    assert metrics['requested_contracts'] == 10
    assert metrics['fill_rate'] == 0.4
    assert metrics['fees'] == 0.08
    assert metrics['average_slippage'] == 0.02
