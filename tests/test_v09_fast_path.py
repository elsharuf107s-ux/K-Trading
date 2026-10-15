import asyncio
from datetime import datetime, timedelta, timezone

from app.ai_sidecar import AISidecarClient, SidecarForecast
from app.config import settings
from app.engine import TradingEngine
from app.fast_path import FastPathController
from app.features import PriceBuffer
from app.models import BtcSnapshot, Forecast, MarketSnapshot


def test_price_buffer_recent_points_deduplicates_identical_frame():
    buf = PriceBuffer()
    now = datetime.now(timezone.utc)
    buf.add_point(100_000, now)
    buf.add_point(100_000, now)
    buf.add_point(100_010, now + timedelta(seconds=1))
    assert buf.recent_points(10) == [
        (now.timestamp(), 100_000),
        ((now + timedelta(seconds=1)).timestamp(), 100_010),
    ]


def test_sidecar_blend_is_confidence_gated_and_shift_capped(monkeypatch):
    monkeypatch.setattr(settings, 'ai_sidecar_enabled', True)
    monkeypatch.setattr(settings, 'ai_sidecar_url', 'http://sidecar.invalid/forecast')
    monkeypatch.setattr(settings, 'ai_sidecar_weight', 0.50)
    monkeypatch.setattr(settings, 'ai_sidecar_min_confidence', 0.55)
    monkeypatch.setattr(settings, 'ai_sidecar_max_probability_shift', 0.08)
    client = AISidecarClient()
    base = Forecast(yes_probability=0.50, confidence=0.60, rationale=['base'], model_name='base')

    client.latest = SidecarForecast(
        ticker='T', yes_probability=0.99, confidence=0.95, model='amazon/chronos-2',
    )
    blended = client.blend(base, 'T')
    assert blended.yes_probability == 0.58
    assert 'sidecar:amazon/chronos-2' in blended.model_name

    client.latest = SidecarForecast(
        ticker='T', yes_probability=0.99, confidence=0.20, model='amazon/chronos-2',
    )
    ignored = client.blend(base, 'T')
    assert ignored.yes_probability == base.yes_probability
    asyncio.run(client.close())


def test_fast_path_uses_in_memory_state_without_rest(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'database_path', str(tmp_path / 'fast.sqlite3'))
    monkeypatch.setattr(settings, 'model_dir', str(tmp_path / 'models'))
    monkeypatch.setattr(settings, 'strategy_eval_report_path', str(tmp_path / 'eval.json'))
    monkeypatch.setattr(settings, 'fast_path_enabled', True)
    monkeypatch.setattr(settings, 'fast_path_require_stream', True)
    monkeypatch.setattr(settings, 'fast_path_interval_ms', 200)
    monkeypatch.setattr(settings, 'auto_paper_trade', False)
    monkeypatch.setattr(settings, 'ai_sidecar_enabled', False)

    engine = TradingEngine()
    now = datetime.now(timezone.utc)
    market = MarketSnapshot(
        ticker='KXBTC15M-FAST-TEST',
        title='fast path test',
        open_time=now - timedelta(minutes=2),
        close_time=now + timedelta(minutes=13),
        yes_bid=0.49,
        yes_ask=0.51,
        no_bid=0.49,
        no_ask=0.51,
        last_price=0.50,
        seconds_remaining=780,
        raw_target=100_000,
        yes_depth=100,
        no_depth=100,
        yes_book=[(0.48, 100), (0.49, 100)],
        no_book=[(0.48, 100), (0.49, 100)],
        captured_at=now,
    )
    btc = BtcSnapshot(
        price=100_050,
        sources={'coinbase': 100_050, 'kraken': 100_051},
        captured_at=now,
    )
    engine.buffer.add(btc)
    engine.last_state = {
        'status': 'running',
        'market': market.model_dump(mode='json'),
        'btc': btc.model_dump(mode='json'),
    }

    # A fast-path tick must not invoke market discovery/REST. Make an accidental
    # call fail the test immediately.
    async def forbidden_rest_call(*args, **kwargs):
        raise AssertionError('fast path attempted a REST discovery call')

    engine.kalshi.discover_btc_15m = forbidden_rest_call
    engine.orderbook_stream.fresh = lambda ticker: True
    engine.orderbook_stream.apply_to_market = lambda m: m
    engine.orderbook_stream.drain_moves = lambda max_items=10_000: []

    controller = FastPathController(engine)
    result = asyncio.run(controller.tick())
    assert result is not None
    assert controller.valid_cycles == 1
    assert result['fast_path']['target_interval_ms'] == 200
    assert result['features']['ticker'] == market.ticker
    assert result['decision']['action'] in ('HOLD', 'BUY_YES', 'BUY_NO')

    asyncio.run(controller.sidecar.close())
    asyncio.run(engine.kalshi.close())
    asyncio.run(engine.btc.close())
