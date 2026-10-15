import json
import math

from app.engine import TradingEngine
from app.main import _json_safe


def test_strategy_json_safe_removes_non_finite_values():
    payload = {
        'ok': 1.25,
        'nan': float('nan'),
        'pos_inf': float('inf'),
        'neg_inf': float('-inf'),
        'nested': [0.5, float('nan')],
    }
    safe = _json_safe(payload)
    assert safe['ok'] == 1.25
    assert safe['nan'] is None
    assert safe['pos_inf'] is None
    assert safe['neg_inf'] is None
    assert safe['nested'][1] is None
    # Strict JSON serialization must succeed after sanitization.
    json.dumps(safe, allow_nan=False)


def test_market_result_prefers_explicit_result():
    result, settled_at = TradingEngine._resolved_market_result({
        'status': 'settled',
        'result': 'yes',
        'settlement_value_dollars': '0.0000',
        'settlement_ts': '2026-10-15T15:15:00Z',
    })
    assert result == 'yes'
    assert settled_at == '2026-10-15T15:15:00Z'


def test_market_result_falls_back_to_binary_settlement_value():
    yes, _ = TradingEngine._resolved_market_result({
        'status': 'settled',
        'result': '',
        'settlement_value_dollars': '1.0000',
    })
    no, _ = TradingEngine._resolved_market_result({
        'status': 'settled',
        'result': None,
        'settlement_value_dollars': '0.0000',
    })
    assert yes == 'yes'
    assert no == 'no'


def test_market_result_does_not_guess_unsettled_or_nonbinary_value():
    unresolved, _ = TradingEngine._resolved_market_result({
        'status': 'closed',
        'settlement_value_dollars': '1.0000',
    })
    nonbinary, _ = TradingEngine._resolved_market_result({
        'status': 'settled',
        'settlement_value_dollars': '0.5000',
    })
    assert unresolved is None
    assert nonbinary is None


def test_daily_loss_limit_resets_on_a_new_utc_day():
    from datetime import datetime, timedelta, timezone

    from app.paper import PaperBroker

    broker = PaperBroker(1000)
    broker.open('T1', 'yes', 100, 0.60, 'test')
    broker.close('T1', 0.05, 'loss')
    assert round(broker.realized_pnl, 2) == -55.00
    assert round(broker.daily_pnl(), 2) == -55.00
    # Lifetime realized P&L must not keep the daily limit tripped forever.
    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
    assert broker.daily_pnl(tomorrow) == 0.0
    assert round(broker.realized_pnl, 2) == -55.00


def test_daily_pnl_survives_restart_and_restores_legacy_state():
    from app.paper import PaperBroker

    first = PaperBroker(1000)
    first.open('T1', 'yes', 10, 0.50, 'test')
    first.settle('T1', 'no')
    second = PaperBroker(1000)
    second.restore(first.snapshot())
    assert round(second.daily_pnl(), 2) == -5.00

    legacy = first.snapshot()
    legacy.pop('daily_pnl_date')
    legacy.pop('daily_realized_pnl')
    third = PaperBroker(1000)
    third.restore(legacy)
    assert third.daily_pnl() == 0.0
    assert round(third.realized_pnl, 2) == -5.00


def test_research_gate_passes_zero_losses_and_zero_drawdown(tmp_path):
    from app.storage import Storage
    from app.strategy_eval import ResearchGate, StrategyEvaluator

    storage = Storage(str(tmp_path / 'gate.sqlite3'))
    gate = ResearchGate(storage, StrategyEvaluator(storage, tmp_path / 'eval.json'))
    perfect = {'metrics': {'traded_windows': 60, 'max_drawdown_fraction': 0.0, 'max_consecutive_losses': 0}}
    gates = {g['name']: g for g in gate.evaluate({}, {}, strategy_report=perfect)['gates']}
    assert gates['loss_streak']['pass'] is True
    assert gates['replay_drawdown']['pass'] is True

    # Missing metrics (no replay run yet) or zero replay trades must still fail closed.
    no_evidence = [{}, {'traded_windows': 0, 'max_drawdown_fraction': 0.0, 'max_consecutive_losses': 0}]
    for metrics in no_evidence:
        gates = {g['name']: g for g in gate.evaluate({}, {}, strategy_report={'metrics': metrics})['gates']}
        assert gates['loss_streak']['pass'] is False
        assert gates['replay_drawdown']['pass'] is False
