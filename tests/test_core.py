from datetime import datetime, timedelta, timezone

from app.features import PriceBuffer, build_features
from app.forecast import BaselineForecaster
from app.models import BtcSnapshot, MarketSnapshot
from app.paper import PaperBroker
from app.risk import AccountRiskState, RiskEngine


def market(**kw):
    now = datetime.now(timezone.utc)
    base = dict(
        ticker='KXBTC15M-TEST', title='Bitcoin 15 minute test', open_time=now - timedelta(minutes=2),
        close_time=now + timedelta(minutes=13), yes_bid=0.49, yes_ask=0.51, no_bid=0.49,
        no_ask=0.51, last_price=0.50, seconds_remaining=780,
    )
    base.update(kw)
    return MarketSnapshot(**base)


def test_forecaster_stays_bounded():
    buf = PriceBuffer()
    t = datetime.now(timezone.utc)
    for i, p in enumerate([100000, 100020, 100050, 100090]):
        buf.add(BtcSnapshot(price=p, sources={'a': p, 'b': p + 1}, captured_at=t + timedelta(seconds=i * 30)))
    f = build_features(market(), BtcSnapshot(price=100090, sources={'a': 100090, 'b': 100091}, captured_at=t + timedelta(seconds=90)), buf)
    out = BaselineForecaster().predict(f)
    assert 0.01 <= out.yes_probability <= 0.99
    assert abs(out.yes_probability - f.market_probability) <= 0.161


def test_risk_rejects_wide_spread():
    f = build_features(market(yes_bid=0.30, yes_ask=0.60), BtcSnapshot(price=100000, sources={'a': 100000}), PriceBuffer())
    fc = BaselineForecaster().predict(f)
    d = RiskEngine().evaluate_entry(f, fc, AccountRiskState(cash=1000, equity=1000, starting_equity=1000, daily_pnl=0))
    assert d.action == 'HOLD'
    assert d.risk_status == 'blocked'


def test_paper_round_trip():
    broker = PaperBroker(1000)
    broker.open('T', 'yes', 10, 0.40, 'test')
    trade = broker.close('T', 0.60, 'test exit')
    assert round(trade.pnl, 2) == 2.00
    assert round(broker.cash, 2) == 1002.00
