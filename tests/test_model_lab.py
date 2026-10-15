from datetime import datetime, timedelta, timezone

from app.model_lab import ModelLab, ModelRegistry
from app.models import FeatureVector, Forecast, MarketSnapshot
from app.storage import Storage


def _save_window(storage: Storage, index: int, outcome_yes: bool) -> None:
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15 * index)
    ticker = f'KXBTC15M-SYN-{index:03d}'
    result = 'yes' if outcome_yes else 'no'

    # Deliberately make the market baseline lean the wrong direction so the
    # synthetic feature signal has something real to beat out-of-sample.
    market_probability = 0.38 if outcome_yes else 0.62
    direction = 1.0 if outcome_yes else -1.0

    for step in range(3):
        created = base_time + timedelta(minutes=step * 4)
        m = MarketSnapshot(
            ticker=ticker,
            title='synthetic BTC 15m',
            open_time=base_time,
            close_time=base_time + timedelta(minutes=15),
            yes_bid=market_probability - 0.01,
            yes_ask=market_probability + 0.01,
            no_bid=1 - market_probability - 0.01,
            no_ask=1 - market_probability + 0.01,
            last_price=market_probability,
            seconds_remaining=900 - step * 240,
            raw_target=100000,
            yes_depth=150 if outcome_yes else 45,
            no_depth=45 if outcome_yes else 150,
            book_imbalance=0.70 * direction,
            captured_at=created,
        )
        f = FeatureVector(
            ticker=ticker,
            btc_price=100000 + direction * (25 + step * 20),
            reference_price=100000 + direction * (35 + step * 25),
            reference_source='cfbenchmarks',
            benchmark_avg_60s=100000 + direction * 30,
            settlement_window_avg=None,
            market_probability=market_probability,
            yes_bid=m.yes_bid,
            yes_ask=m.yes_ask,
            no_bid=m.no_bid,
            no_ask=m.no_ask,
            spread=0.02,
            seconds_remaining=m.seconds_remaining,
            momentum_10s=direction * 0.0002,
            momentum_30s=direction * 0.0006,
            momentum_60s=direction * 0.0010,
            momentum_180s=direction * 0.0016,
            reference_momentum_10s=direction * 0.00025,
            reference_momentum_30s=direction * 0.0008,
            reference_momentum_60s=direction * 0.0012,
            realized_vol_60s=0.0003,
            reference_vol_60s=0.00025,
            source_dispersion=0.00005,
            target_price=100000,
            distance_to_target_bps=direction * (8 + step * 3),
            yes_depth=m.yes_depth,
            no_depth=m.no_depth,
            book_imbalance=m.book_imbalance,
            market_momentum_10s=direction * 0.03,
            market_momentum_30s=direction * 0.06,
            wave_score=direction * 0.82,
            wave_state='UP' if outcome_yes else 'DOWN',
        )
        storage.save_observation(
            m,
            f,
            Forecast(
                yes_probability=market_probability,
                confidence=0.5,
                rationale=['synthetic'],
                model_name='synthetic-baseline',
            ),
        )
    storage.save_market_result(ticker, result, settled_at=(base_time + timedelta(minutes=16)).isoformat())


def test_model_lab_refuses_small_dataset(tmp_path):
    storage = Storage(str(tmp_path / 'small.sqlite3'))
    for i in range(6):
        _save_window(storage, i, outcome_yes=(i % 2 == 0))
    registry = ModelRegistry(tmp_path / 'models-small')
    report = ModelLab(storage, registry).train()
    assert report['status'] == 'insufficient_data'
    assert report['promoted'] is False
    assert registry.load_champion() is None


def test_model_lab_promotes_only_after_oos_gain(tmp_path, monkeypatch):
    from app import model_lab as model_lab_module

    monkeypatch.setattr(model_lab_module.settings, 'model_min_windows', 24)
    monkeypatch.setattr(model_lab_module.settings, 'model_min_brier_gain', 0.002)

    storage = Storage(str(tmp_path / 'learning.sqlite3'))
    for i in range(32):
        _save_window(storage, i, outcome_yes=(i % 2 == 0))

    registry = ModelRegistry(tmp_path / 'models')
    report = ModelLab(storage, registry).train()

    assert report['windows'] == 32
    assert report['status'] == 'promoted'
    assert report['promoted'] is True
    assert report['selected']['metrics']['brier'] < report['selected']['metrics']['market_brier']
    assert report['selected']['metrics']['brier_gain_vs_market'] >= 0.002

    champion = registry.load_champion()
    assert champion is not None
    assert champion.name in {'logistic', 'hist_gradient_boosting'}
    assert champion.training_windows >= 8
