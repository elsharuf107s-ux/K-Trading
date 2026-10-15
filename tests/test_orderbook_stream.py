from datetime import datetime, timezone

import pytest

from app.config import settings
from app.empirical import EmpiricalExecutionLab
from app.execution import ExecutionSimulator
from app.models import MarketSnapshot
from app.orderbook_stream import OrderBookStream, QuoteMove


def test_orderbook_snapshot_and_delta_build_live_book(monkeypatch):
    monkeypatch.setattr(settings, 'orderbook_stream_stale_seconds', 5.0)
    stream = OrderBookStream()
    stream.active_ticker = 'KXBTC15M-TEST'
    stream.target_ticker = 'KXBTC15M-TEST'
    stream._handle({
        'type': 'orderbook_snapshot',
        'seq': 1,
        'msg': {
            'market_ticker': 'KXBTC15M-TEST',
            'yes_dollars_fp': [['0.4000', '5.00'], ['0.4200', '4.00']],
            'no_dollars_fp': [['0.4500', '2.00'], ['0.4700', '6.00']],
        },
    })
    assert stream.synced is True
    assert stream.yes[0.42] == 4
    assert stream.no[0.47] == 6

    stream._handle({
        'type': 'orderbook_delta',
        'seq': 2,
        'msg': {
            'market_ticker': 'KXBTC15M-TEST',
            'side': 'no',
            'price_dollars': '0.4700',
            'delta_fp': '-2.00',
        },
    })
    assert stream.no[0.47] == 4

    market = MarketSnapshot(ticker='KXBTC15M-TEST')
    updated = stream.apply_to_market(market)
    assert updated.yes_bid == pytest.approx(0.42)
    assert updated.no_bid == pytest.approx(0.47)
    assert updated.yes_ask == pytest.approx(0.53)
    assert updated.no_ask == pytest.approx(0.58)
    assert updated.yes_book
    assert updated.no_book


def test_orderbook_sequence_gap_invalidates_stream():
    stream = OrderBookStream()
    stream.active_ticker = 'KXBTC15M-GAP'
    stream.target_ticker = 'KXBTC15M-GAP'
    stream._handle({
        'type': 'orderbook_snapshot',
        'seq': 10,
        'msg': {
            'market_ticker': 'KXBTC15M-GAP',
            'yes_dollars_fp': [['0.40', '2']],
            'no_dollars_fp': [['0.50', '2']],
        },
    })
    with pytest.raises(RuntimeError, match='sequence gap'):
        stream._handle({
            'type': 'orderbook_delta',
            'seq': 12,
            'msg': {
                'market_ticker': 'KXBTC15M-GAP',
                'side': 'yes',
                'price_dollars': '0.41',
                'delta_fp': '1',
            },
        })
    assert stream.synced is False
    assert stream.sequence_gaps == 1


def test_quote_sampler_measures_configured_horizon(monkeypatch):
    monkeypatch.setattr(settings, 'empirical_quote_horizons_ms', '250')
    stream = OrderBookStream()
    stream.active_ticker = 'KXBTC15M-SAMPLE'
    stream.yes = {0.40: 5}
    stream.no = {0.50: 5}
    stream._sample(1000.0)

    # 300ms later the NO bid falls, so the executable YES ask rises 2 cents.
    stream.no = {0.48: 5}
    stream._sample(1000.3)
    moves = stream.drain_moves()
    assert len(moves) == 1
    move = moves[0]
    assert move.horizon_ms == 250
    assert round(move.elapsed_ms) == 300
    assert round(move.yes_ask_move, 4) == 0.02


def test_empirical_lab_recommends_p95_after_sample_gate(tmp_path):
    lab = EmpiricalExecutionLab(str(tmp_path / 'empirical.sqlite3'))
    moves = [
        QuoteMove(
            ticker=f'T{i}', horizon_ms=250, elapsed_ms=250,
            yes_ask_move=move, no_ask_move=-move / 2,
            captured_at=datetime.now(timezone.utc).isoformat(),
        )
        for i, move in enumerate([0.001, 0.002, 0.003, 0.004, 0.010])
    ]
    assert lab.save_moves(moves) == 5
    summary = lab.horizon_summary(250)
    assert summary['samples'] == 5
    assert summary['p95_adverse_move'] > 0

    recommendation = lab.recommended_adverse_slippage(260, min_samples=5)
    assert recommendation['ready'] is True
    assert recommendation['horizon_ms'] == 250
    assert recommendation['p95_adverse_move'] == summary['p95_adverse_move']


def test_execution_can_use_empirical_latency_override(monkeypatch):
    monkeypatch.setattr(settings, 'sim_max_latency_slippage', 0.02)
    market = MarketSnapshot(
        ticker='KXBTC15M-EMP',
        yes_book=[(0.40, 10)],
        no_book=[(0.47, 10)],
        yes_bid=0.40,
        no_bid=0.47,
        yes_ask=0.53,
        no_ask=0.60,
    )
    fill = ExecutionSimulator().simulate(
        market, 'yes', 'buy', 5,
        wave_score=0.8,
        latency_penalty_override=0.003,
    )
    assert fill.filled_quantity == 5
    assert fill.latency_slippage == pytest.approx(0.003)
    assert 'empirical latency penalty' in fill.reason


def test_empirical_summary_refreshes_incrementally_and_keeps_newest_window(tmp_path):
    lab = EmpiricalExecutionLab(str(tmp_path / 'incremental.sqlite3'), max_samples=4, cache_seconds=0.0)

    def move(value):
        return QuoteMove(
            ticker='T', horizon_ms=250, elapsed_ms=250, yes_ask_move=value, no_ask_move=0.0,
            captured_at=datetime.now(timezone.utc).isoformat(),
        )

    lab.save_moves([move(0.01), move(0.01)])
    assert lab.horizon_summary(250)['samples'] == 2
    lab.save_moves([move(0.03), move(0.03), move(0.03), move(0.03)])
    summary = lab.horizon_summary(250)
    # Only the newest `max_samples` moves count, so the two 0.01 moves age out.
    assert summary['samples'] == 4
    assert summary['mean_yes_ask_move'] == pytest.approx(0.03)

    cached = EmpiricalExecutionLab(str(tmp_path / 'incremental.sqlite3'), cache_seconds=60.0)
    first = cached.horizon_summary(250)
    cached.save_moves([move(0.50)])
    assert cached.horizon_summary(250) == first
