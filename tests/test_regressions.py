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
