from datetime import datetime, timedelta, timezone

from app.features import MarketBuffer, PriceBuffer, build_features
from app.forecast import BaselineForecaster
from app.models import BenchmarkSnapshot, BtcSnapshot, MarketSnapshot, PaperPosition
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


def test_cfbenchmarks_message_parser():
    from app.stream import CFBenchmarkStream
    stream = CFBenchmarkStream()
    stream._handle({
        'type': 'cfbenchmarks_value_5hz',
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
