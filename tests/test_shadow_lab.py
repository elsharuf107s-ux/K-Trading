from app.config import settings
from app.execution import ExecutionSimulator
from app.models import ExecutionEvent, FeatureVector, Forecast, MarketSnapshot
from app.risk import AccountRiskState, RiskEngine
from app.storage import Storage


def features(**overrides):
    base = dict(
        ticker='KXBTC15M-SHADOW',
        btc_price=100100,
        reference_price=100110,
        reference_source='cfbenchmarks',
        market_probability=0.50,
        yes_bid=0.49,
        yes_ask=0.51,
        no_bid=0.49,
        no_ask=0.51,
        spread=0.02,
        seconds_remaining=500,
        momentum_30s=0.001,
        source_dispersion=0.0001,
        target_price=100000,
        distance_to_target_bps=11.0,
        yes_depth=50,
        no_depth=50,
        book_imbalance=0.5,
        wave_score=0.6,
        wave_state='UP',
    )
    base.update(overrides)
    return FeatureVector(**base)


def account():
    return AccountRiskState(cash=1000, equity=1000, starting_equity=1000, daily_pnl=0)


def test_risk_blocks_disagreeing_spot_sources(monkeypatch):
    monkeypatch.setattr(settings, 'max_source_dispersion', 0.003)
    f = features(source_dispersion=0.01)
    decision = RiskEngine().evaluate_entry(
        f,
        Forecast(yes_probability=0.75, confidence=0.8, rationale=['test']),
        account(),
    )
    assert decision.action == 'HOLD'
    assert decision.risk_status == 'blocked'
    assert 'dispersion' in decision.reason.lower()


def test_risk_requires_opposite_side_depth(monkeypatch):
    monkeypatch.setattr(settings, 'require_orderbook_for_entry', True)
    monkeypatch.setattr(settings, 'min_execution_side_depth', 1.0)
    f = features(no_depth=0, wave_score=0.8)
    decision = RiskEngine().evaluate_entry(
        f,
        Forecast(yes_probability=0.80, confidence=0.8, rationale=['test']),
        account(),
    )
    assert decision.action == 'HOLD'
    assert decision.side == 'yes'
    assert decision.risk_status == 'blocked'
    assert 'order-book depth' in decision.reason


def test_execution_splits_depth_and_latency_drag(monkeypatch):
    market = MarketSnapshot(
        ticker='KXBTC15M-DRAG',
        yes_book=[(0.40, 5), (0.42, 5)],
        no_book=[(0.45, 2), (0.47, 4)],
        yes_bid=0.42,
        no_bid=0.47,
        yes_ask=0.53,
        no_ask=0.58,
    )
    monkeypatch.setattr(settings, 'sim_latency_ms', 500.0)
    monkeypatch.setattr(settings, 'sim_latency_price_drift_per_second', 0.01)
    monkeypatch.setattr(settings, 'sim_max_latency_slippage', 0.02)
    fill = ExecutionSimulator().simulate(market, 'yes', 'buy', 6, wave_score=1.0)
    assert fill.filled_quantity == 6
    assert fill.depth_slippage > 0
    assert fill.latency_slippage > 0
    assert abs(fill.slippage - (fill.depth_slippage + fill.latency_slippage)) < 1e-9


def test_zero_latency_leaves_only_depth_drag(monkeypatch):
    market = MarketSnapshot(
        ticker='KXBTC15M-DRAG0',
        yes_book=[(0.40, 5), (0.42, 5)],
        no_book=[(0.45, 2), (0.47, 4)],
        yes_bid=0.42,
        no_bid=0.47,
        yes_ask=0.53,
        no_ask=0.58,
    )
    monkeypatch.setattr(settings, 'sim_latency_ms', 0.0)
    fill = ExecutionSimulator().simulate(market, 'yes', 'buy', 6, wave_score=1.0)
    assert fill.depth_slippage > 0
    assert fill.latency_slippage == 0
    assert abs(fill.slippage - fill.depth_slippage) < 1e-9


def test_execution_events_preserve_rejections_and_no_liquidity(tmp_path):
    storage = Storage(str(tmp_path / 'shadow.sqlite3'))
    storage.save_execution_event(ExecutionEvent(
        ticker='A', side='yes', intent='buy', status='filled',
        requested_quantity=10, filled_quantity=10, fair_probability=0.70,
        top_price=0.55, average_price=0.57, gross_edge_before=0.15,
        executable_edge_after=0.09, fee=0.12, depth_slippage=0.015,
        latency_slippage=0.005, total_slippage=0.02, reason='filled',
    ))
    storage.save_execution_event(ExecutionEvent(
        ticker='B', side='yes', intent='buy', status='rejected',
        requested_quantity=10, filled_quantity=4, fair_probability=0.65,
        top_price=0.55, average_price=0.60, gross_edge_before=0.10,
        executable_edge_after=0.01, fee=0.05, depth_slippage=0.04,
        latency_slippage=0.01, total_slippage=0.05, reason='edge erased',
    ))
    storage.save_execution_event(ExecutionEvent(
        ticker='C', side='no', intent='buy', status='no_liquidity',
        requested_quantity=8, filled_quantity=0, fair_probability=0.60,
        gross_edge_before=0.08, executable_edge_after=0.0, reason='empty book',
    ))

    events = storage.recent_execution_events(10)
    assert len(events) == 3
    stats = storage.execution_attribution()
    assert stats['entry_attempts'] == 3
    # "filled_attempts" is intentionally a fillable-attempt count: it includes
    # a simulated fill that was later rejected by the executable-edge gate.
    assert stats['filled_attempts'] == 2
    assert stats['rejected_attempts'] == 1
    assert stats['no_liquidity_attempts'] == 1
    assert stats['average_edge_drag'] > 0
    assert stats['average_depth_slippage'] > 0
    assert stats['average_latency_slippage'] > 0
