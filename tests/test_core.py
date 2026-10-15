from datetime import datetime, timedelta, timezone

from app.features import MarketBuffer, PriceBuffer, ReferenceBuffer, build_features
from app.forecast import BaselineForecaster
from app.models import BenchmarkSnapshot, BtcSnapshot, MarketSnapshot
from app.paper import PaperBroker
from app.risk import AccountRiskState, RiskEngine


def market(**kw):
    now = datetime.now(timezone.utc)
    base = dict(
        ticker='KXBTC15M-TEST', title='Bitcoin 15 minute test',
        open_time=now - timedelta(minutes=2), close_time=now + timedelta(minutes=13),
        yes_bid=0.49, yes_ask=0.51, no_bid=0.49, no_ask=0.51,
        last_price=0.50, seconds_remaining=780, raw_target=100000,
        yes_depth=120, no_depth=80, book_imbalance=0.20,
    )
    base.update(kw)
    return MarketSnapshot(**base)


def warm_buffers():
    buf = PriceBuffer()
    mb = MarketBuffer()
    t = datetime.now(timezone.utc)
    prices = [100000, 100020, 100050, 100090]
    for i, p in enumerate(prices):
        snap = BtcSnapshot(price=p, sources={'a': p, 'b': p + 1}, captured_at=t + timedelta(seconds=i * 30))
        buf.add(snap)
        mb.add(market(
            captured_at=t + timedelta(seconds=i * 30),
            yes_bid=0.48 + i * 0.01, yes_ask=0.50 + i * 0.01,
            last_price=0.49 + i * 0.01,
        ))
    return buf, mb, t


def test_forecaster_stays_bounded():
    buf, mb, t = warm_buffers()
    f = build_features(
        market(),
        BtcSnapshot(price=100090, sources={'a': 100090, 'b': 100091}, captured_at=t + timedelta(seconds=90)),
        buf, mb,
    )
    out = BaselineForecaster().predict(f)
    assert 0.01 <= out.yes_probability <= 0.99
    assert abs(out.yes_probability - f.market_probability) <= 0.141


def test_official_benchmark_becomes_reference():
    buf, mb, t = warm_buffers()
    b = BenchmarkSnapshot(index_id='BRTI', value=100123.0, avg_60s=100111.0)
    f = build_features(
        market(),
        BtcSnapshot(price=99900, sources={'a': 99900}, captured_at=t),
        buf, mb, b,
    )
    assert f.reference_source == 'cfbenchmarks'
    assert f.reference_price == 100123.0
    assert f.distance_to_target_bps > 0


def test_brti_momentum_drives_reference_features():
    buf, mb, t = warm_buffers()
    ref = ReferenceBuffer()
    ref.add_benchmark(BenchmarkSnapshot(value=100000, captured_at=t))
    ref.add_benchmark(BenchmarkSnapshot(value=100050, captured_at=t + timedelta(seconds=30)))
    current = BenchmarkSnapshot(value=100120, avg_60s=100050, captured_at=t + timedelta(seconds=60))
    ref.add_benchmark(current)
    f = build_features(
        market(),
        BtcSnapshot(price=99950, sources={'a': 99950}, captured_at=t + timedelta(seconds=60)),
        buf, mb, current, ref,
    )
    assert f.reference_source == 'cfbenchmarks'
    assert f.reference_momentum_30s > 0
    assert f.reference_momentum_60s > 0
    assert f.reference_vol_60s >= 0


def test_final_minute_uses_settlement_window_average():
    buf, mb, t = warm_buffers()
    b = BenchmarkSnapshot(index_id='BRTI', value=100010.0, avg_60s=100020.0, settlement_window_avg=100150.0)
    f = build_features(
        market(seconds_remaining=35),
        BtcSnapshot(price=99900, sources={'a': 99900}, captured_at=t),
        buf, mb, b,
    )
    assert f.reference_source == 'cfbenchmarks_settlement_window'
    assert f.reference_price == 100150.0


def test_risk_rejects_wide_spread():
    f = build_features(
        market(yes_bid=0.30, yes_ask=0.60),
        BtcSnapshot(price=100000, sources={'a': 100000}),
        PriceBuffer(), MarketBuffer(),
    )
    fc = BaselineForecaster().predict(f)
    d = RiskEngine().evaluate_entry(
        f, fc, AccountRiskState(cash=1000, equity=1000, starting_equity=1000, daily_pnl=0)
    )
    assert d.action == 'HOLD'
    assert d.risk_status == 'blocked'


def test_final_minute_blocks_proxy_entry():
    f = build_features(
        market(seconds_remaining=60),
        BtcSnapshot(price=100100, sources={'a': 100100}),
        PriceBuffer(), MarketBuffer(),
    )
    fc = BaselineForecaster().predict(f)
    d = RiskEngine().evaluate_entry(
        f, fc, AccountRiskState(cash=1000, equity=1000, starting_equity=1000, daily_pnl=0)
    )
    assert d.action == 'HOLD'
    assert 'official CF Benchmarks' in d.reason


def test_paper_round_trip():
    broker = PaperBroker(1000)
    broker.open('T', 'yes', 10, 0.40, 'test')
    trade = broker.close('T', 0.60, 'test exit')
    assert round(trade.pnl, 2) == 2.00
    assert round(broker.cash, 2) == 1002.00


def test_paper_settlement():
    broker = PaperBroker(1000)
    broker.open('T', 'yes', 10, 0.40, 'test')
    trade = broker.settle('T', 'yes')
    assert trade.action == 'settle'
    assert round(trade.pnl, 2) == 6.00
    assert round(broker.cash, 2) == 1006.00


def test_paper_state_survives_storage_round_trip(tmp_path):
    from app.storage import Storage
    st = Storage(str(tmp_path / 'state.sqlite3'))
    first = PaperBroker(1000)
    first.open('KXBTC15M-PERSIST', 'yes', 12, 0.42, 'persist me')
    first.kill_switch = True
    st.save_paper_state(first.snapshot())

    second = PaperBroker(1000)
    second.restore(st.load_paper_state())
    assert round(second.cash, 2) == round(first.cash, 2)
    assert second.kill_switch is True
    assert 'KXBTC15M-PERSIST' in second.positions
    assert second.positions['KXBTC15M-PERSIST'].quantity == 12


def test_cfbenchmarks_message_parser():
    from app.stream import CFBenchmarkStream
    stream = CFBenchmarkStream()
    assert stream.status()['channel'] == 'cfbenchmarks_value'
    stream._handle({
        'type': 'cfbenchmarks_value',
        'msg': {
            'index_id': 'BRTI',
            'received_at': 1234567890000,
            'data': '{"value":"100250.50"}',
            'avg_60s_data': {'value': '100200.25'},
            'last_60s_windowed_average_15min': {'value': '100210.75'},
        },
    })
    assert stream.latest is not None
    assert stream.latest.value == 100250.50
    assert stream.latest.avg_60s == 100200.25
    assert stream.latest.settlement_window_avg == 100210.75


def test_5hz_tick_preserves_settlement_averages():
    from app.stream import CFBenchmarkStream
    stream = CFBenchmarkStream()
    stream._handle({
        'type': 'cfbenchmarks_value',
        'msg': {
            'index_id': 'BRTI',
            'received_at': 1234567890000,
            'data': '{"value":"100250.50"}',
            'avg_60s_data': {'value': '100200.25'},
            'last_60s_windowed_average_15min': {'value': '100210.75'},
        },
    })
    stream._handle({
        'type': 'cfbenchmarks_value_5hz',
        'msg': {
            'index_id': 'BRTI',
            'received_at': 1234567890100,
            'value_usd': '100260.75',
            'data': '{"value":"100260.75"}',
        },
    })
    assert stream.latest is not None
    assert stream.latest.value == 100260.75
    assert stream.latest.avg_60s == 100200.25
    assert stream.latest.settlement_window_avg == 100210.75


def test_analytics_brier_and_replay(tmp_path):
    from app.analytics import ResearchAnalytics
    from app.storage import Storage
    path = str(tmp_path / 'research.sqlite3')
    st = Storage(path)
    m = market(ticker='KXBTC15M-LABELED')
    btc = BtcSnapshot(price=100050, sources={'a': 100050})
    f = build_features(m, btc, PriceBuffer(), MarketBuffer())
    fc = BaselineForecaster().predict(f)
    st.save_observation(m, f, fc)
    st.save_market_result(m.ticker, 'yes')
    report = ResearchAnalytics(st).summary()
    assert report['completed_windows'] == 1
    assert report['model_brier_last_snapshot'] is not None
    assert len(report['calibration']) == 1
    replay = st.replay_window(m.ticker)
    assert replay is not None
    assert replay['result'] == 'yes'
    assert len(replay['frames']) == 1


def test_unresolved_market_tracking(tmp_path):
    from app.storage import Storage
    st = Storage(str(tmp_path / 'labels.sqlite3'))
    m = market(ticker='KXBTC15M-OLD')
    f = build_features(m, BtcSnapshot(price=100000, sources={'a': 100000}), PriceBuffer(), MarketBuffer())
    st.save_observation(m, f, BaselineForecaster().predict(f))
    assert m.ticker in st.unresolved_tickers('KXBTC15M-CURRENT')
    st.save_market_result(m.ticker, 'no')
    assert m.ticker not in st.unresolved_tickers('KXBTC15M-CURRENT')


def test_market_windows_backfill_for_existing_databases(tmp_path):
    from app.storage import Storage
    path = str(tmp_path / 'legacy.sqlite3')
    st = Storage(path)
    for ticker in ('KXBTC15M-A', 'KXBTC15M-A', 'KXBTC15M-B', 'KXBTC15M-C'):
        m = market(ticker=ticker)
        f = build_features(m, BtcSnapshot(price=100000, sources={'a': 100000}), PriceBuffer(), MarketBuffer())
        st.save_observation(m, f, BaselineForecaster().predict(f))
    st.save_market_result('KXBTC15M-A', 'yes')
    # Simulate a database written before the market_windows table existed.
    st.db.execute('DROP TABLE market_windows')
    st.db.commit()

    reopened = Storage(path)
    assert reopened.completed_window_count() == 1
    assert reopened.unresolved_tickers('KXBTC15M-C') == ['KXBTC15M-B']
    labeled = reopened.last_labeled_observations()
    assert [row['ticker'] for row in labeled] == ['KXBTC15M-A']
    last_a = reopened.db.execute("SELECT MAX(id) FROM observations WHERE ticker='KXBTC15M-A'").fetchone()[0]
    window = reopened.db.execute(
        "SELECT first_observation_id,last_observation_id FROM market_windows WHERE ticker='KXBTC15M-A'"
    ).fetchone()
    assert window == (1, last_a)
