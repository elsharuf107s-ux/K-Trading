from datetime import datetime, timedelta, timezone

from app.config import settings
from app.models import FeatureVector, Forecast, MarketSnapshot
from app.storage import Storage
from app.strategy_eval import ResearchGate, StrategyEvaluator


def _save_window(storage: Storage, index: int, result: str, forecast_probability: float = 0.80) -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index * 15)
    ticker = f'KXBTC15M-EVAL-{index:03d}'
    frames = [
        # seconds remaining, YES bid, NO bid, wave
        (600, 0.48, 0.50, 0.80),
        (360, 0.60, 0.38, 0.75),
        (120, 0.70, 0.28, 0.70),
    ]
    for frame_index, (seconds, yes_bid, no_bid, wave) in enumerate(frames):
        captured = base + timedelta(seconds=frame_index * 240)
        yes_ask = 1.0 - no_bid
        no_ask = 1.0 - yes_bid
        market = MarketSnapshot(
            ticker=ticker,
            title='synthetic strategy-eval window',
            open_time=base,
            close_time=base + timedelta(minutes=15),
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            last_price=(yes_bid + yes_ask) / 2,
            seconds_remaining=seconds,
            raw_target=100000,
            yes_depth=10,
            no_depth=10,
            book_imbalance=0.30,
            yes_book=[(max(0.01, yes_bid - 0.02), 10), (yes_bid, 10)],
            no_book=[(max(0.01, no_bid - 0.02), 10), (no_bid, 10)],
            captured_at=captured,
        )
        features = FeatureVector(
            ticker=ticker,
            btc_price=100150,
            reference_price=100160,
            reference_source='cfbenchmarks',
            market_probability=market.yes_mid,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            spread=yes_ask - yes_bid,
            seconds_remaining=seconds,
            momentum_10s=0.0003,
            momentum_30s=0.0008,
            momentum_60s=0.0012,
            momentum_180s=0.002,
            reference_momentum_10s=0.0003,
            reference_momentum_30s=0.0009,
            reference_momentum_60s=0.0013,
            realized_vol_60s=0.0003,
            reference_vol_60s=0.00025,
            source_dispersion=0.0001,
            target_price=100000,
            distance_to_target_bps=16,
            yes_depth=10,
            no_depth=10,
            book_imbalance=0.30,
            market_momentum_10s=0.02,
            market_momentum_30s=0.04,
            wave_score=wave,
            wave_state='UP',
        )
        storage.save_observation(
            market,
            features,
            Forecast(
                yes_probability=forecast_probability,
                confidence=0.8,
                rationale=['synthetic evaluation signal'],
                model_name='synthetic-eval',
            ),
        )
    storage.save_market_result(ticker, result, settled_at=(base + timedelta(minutes=16)).isoformat())


def test_strategy_replay_uses_recorded_frames_and_scores_settlement(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'sim_latency_ms', 0.0)
    storage = Storage(str(tmp_path / 'eval.sqlite3'))
    _save_window(storage, 0, 'yes')
    evaluator = StrategyEvaluator(storage, tmp_path / 'report.json')
    report = evaluator.run()

    assert report['status'] == 'ok'
    assert report['metrics']['completed_windows'] == 1
    assert report['metrics']['traded_windows'] == 1
    trade = report['trades'][0]
    assert trade['side'] == 'yes'
    assert trade['direction_correct'] is True
    assert trade['exit_reason'] == 'settlement'
    assert trade['pnl'] > 0
    assert report['metrics']['net_pnl_per_contract_strategy'] > 0


def test_settlement_outcome_cannot_change_entry_side(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'sim_latency_ms', 0.0)
    storage = Storage(str(tmp_path / 'no_lookahead.sqlite3'))
    _save_window(storage, 0, 'yes')
    _save_window(storage, 1, 'no')
    report = StrategyEvaluator(storage, tmp_path / 'report.json').run()

    assert report['metrics']['traded_windows'] == 2
    sides = [trade['side'] for trade in report['trades']]
    assert sides == ['yes', 'yes']
    assert report['trades'][0]['direction_correct'] is True
    assert report['trades'][1]['direction_correct'] is False
    assert report['trades'][0]['entry_price'] == report['trades'][1]['entry_price']


def test_strategy_report_contains_stress_and_bucket_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'sim_latency_ms', 0.0)
    storage = Storage(str(tmp_path / 'buckets.sqlite3'))
    for i in range(8):
        _save_window(storage, i, 'yes' if i % 3 else 'no')
    report = StrategyEvaluator(storage, tmp_path / 'report.json').run()
    metrics = report['metrics']

    assert metrics['traded_windows'] == 8
    assert metrics['max_drawdown'] >= 0
    assert metrics['max_drawdown_fraction'] >= 0
    assert metrics['max_consecutive_losses'] >= 1
    assert report['buckets']['entry_edge']
    assert report['buckets']['entry_time']
    assert report['buckets']['wave_strength']
    assert report['buckets']['reference'][0]['bucket'] == 'official'


def test_replay_skips_frames_without_tradeable_edge(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'sim_latency_ms', 0.0)
    storage = Storage(str(tmp_path / 'skip.sqlite3'))
    _save_window(storage, 0, 'yes', forecast_probability=0.51)
    report = StrategyEvaluator(storage, tmp_path / 'report.json').run()
    assert report['metrics']['completed_windows'] == 1
    assert report['metrics']['traded_windows'] == 0
    assert report['metrics']['skipped_windows'] == 1


def test_research_gate_explains_failure_and_can_pass_dev_thresholds(tmp_path, monkeypatch):
    storage = Storage(str(tmp_path / 'gate.sqlite3'))
    evaluator = StrategyEvaluator(storage, tmp_path / 'report.json')
    gate = ResearchGate(storage, evaluator)

    not_ready = gate.evaluate(
        analytics={
            'completed_windows': 10,
            'brier_improvement_vs_market': -0.01,
            'execution': {'fill_rate': 0.20},
        },
        empirical_summary={'total_samples': 20},
        strategy_report={'metrics': {
            'traded_windows': 2,
            'net_pnl_per_contract_strategy': -1,
            'max_drawdown_fraction': 0.20,
            'max_consecutive_losses': 15,
        }},
    )
    assert not_ready['passed'] is False
    assert not_ready['status'] == 'NOT_READY'
    assert any(not item['pass'] for item in not_ready['gates'])

    monkeypatch.setattr(settings, 'soak_min_completed_windows', 4)
    monkeypatch.setattr(settings, 'soak_min_replay_trades', 2)
    monkeypatch.setattr(settings, 'soak_min_brier_gain', 0.0)
    monkeypatch.setattr(settings, 'soak_require_positive_replay_pnl', True)
    monkeypatch.setattr(settings, 'soak_max_replay_drawdown_fraction', 0.10)
    monkeypatch.setattr(settings, 'soak_max_consecutive_losses', 3)
    monkeypatch.setattr(settings, 'soak_min_execution_fill_rate', 0.5)
    monkeypatch.setattr(settings, 'soak_min_empirical_samples', 10)

    passed = gate.evaluate(
        analytics={
            'completed_windows': 10,
            'brier_improvement_vs_market': 0.05,
            'execution': {'fill_rate': 0.90},
        },
        empirical_summary={'total_samples': 100},
        strategy_report={'metrics': {
            'traded_windows': 5,
            'net_pnl_per_contract_strategy': 1.2,
            'max_drawdown_fraction': 0.03,
            'max_consecutive_losses': 2,
        }},
    )
    assert passed['passed'] is True
    assert passed['status'] == 'GO_FOR_MORE_PAPER_RESEARCH'


def test_replay_includes_newest_window_after_long_history(tmp_path, monkeypatch):
    # Regression: the replay used to load only the oldest 100k observation rows,
    # silently dropping every newer window once ~1 day of 1 Hz frames existed.
    monkeypatch.setattr(settings, 'sim_latency_ms', 0.0)
    storage = Storage(str(tmp_path / 'long_history.sqlite3'))
    storage.db.executemany(
        '''INSERT INTO observations(created_at,ticker,btc_price,market_probability,usable_json)
           VALUES('2025-12-31T00:00:00+00:00','KXBTC15M-OLD',100000,0.5,'{}')''',
        [()] * 100_001,
    )
    storage.db.commit()
    storage.save_market_result('KXBTC15M-OLD', 'no')
    storage = Storage(str(tmp_path / 'long_history.sqlite3'))
    _save_window(storage, 0, 'yes')

    report = StrategyEvaluator(storage, tmp_path / 'report.json').run()
    assert report['metrics']['completed_windows'] == 2
    assert report['metrics']['traded_windows'] == 1
    assert report['trades'][0]['ticker'] == 'KXBTC15M-EVAL-000'
