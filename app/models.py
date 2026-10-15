from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MarketSnapshot(BaseModel):
    ticker: str
    event_ticker: str = ''
    title: str = ''
    subtitle: str = ''
    open_time: datetime | None = None
    close_time: datetime | None = None
    yes_bid: float = 0.0
    yes_ask: float = 1.0
    no_bid: float = 0.0
    no_ask: float = 1.0
    last_price: float = 0.5
    volume: float = 0.0
    open_interest: float = 0.0
    seconds_remaining: float = 0.0
    raw_target: float | None = None
    rules_primary: str = ''
    yes_depth: float = 0.0
    no_depth: float = 0.0
    book_imbalance: float = 0.0
    # Kalshi publishes YES bids and NO bids. We preserve them so the paper
    # executor can derive asks and walk real visible depth instead of filling
    # the whole order at the top quote.
    yes_book: list[tuple[float, float]] = Field(default_factory=list)
    no_book: list[tuple[float, float]] = Field(default_factory=list)
    captured_at: datetime = Field(default_factory=utc_now)

    @property
    def yes_mid(self) -> float:
        if self.yes_bid > 0 and self.yes_ask < 1:
            return (self.yes_bid + self.yes_ask) / 2
        return self.last_price or 0.5

    @property
    def spread(self) -> float:
        return max(0.0, self.yes_ask - self.yes_bid)


class BtcSnapshot(BaseModel):
    price: float
    sources: dict[str, float]
    captured_at: datetime = Field(default_factory=utc_now)


class BenchmarkSnapshot(BaseModel):
    index_id: str = 'BRTI'
    value: float | None = None
    avg_60s: float | None = None
    settlement_window_avg: float | None = None
    received_at_ms: int | None = None
    captured_at: datetime = Field(default_factory=utc_now)
    source: str = 'cfbenchmarks_ws'


class FeatureVector(BaseModel):
    ticker: str
    btc_price: float
    reference_price: float
    reference_source: str
    benchmark_avg_60s: float | None = None
    settlement_window_avg: float | None = None
    market_probability: float
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    spread: float
    seconds_remaining: float
    momentum_10s: float = 0.0
    momentum_30s: float = 0.0
    momentum_60s: float = 0.0
    momentum_180s: float = 0.0
    reference_momentum_10s: float = 0.0
    reference_momentum_30s: float = 0.0
    reference_momentum_60s: float = 0.0
    realized_vol_60s: float = 0.0
    reference_vol_60s: float = 0.0
    source_dispersion: float = 0.0
    target_price: float | None = None
    distance_to_target_bps: float | None = None
    yes_depth: float = 0.0
    no_depth: float = 0.0
    book_imbalance: float = 0.0
    market_momentum_10s: float = 0.0
    market_momentum_30s: float = 0.0
    wave_score: float = 0.0
    wave_state: Literal['UP', 'DOWN', 'FLAT'] = 'FLAT'


class Forecast(BaseModel):
    yes_probability: float
    confidence: float
    rationale: list[str]
    model_name: str = 'market-anchor-wave-v3'


class Decision(BaseModel):
    action: Literal['BUY_YES', 'BUY_NO', 'HOLD', 'EXIT']
    side: Literal['yes', 'no', 'none'] = 'none'
    usable_edge: float = 0.0
    gross_edge: float = 0.0
    price: float = 0.0
    quantity: int = 0
    reason: str = ''
    risk_status: str = 'not_checked'


class ExecutionFill(BaseModel):
    ticker: str
    side: Literal['yes', 'no']
    intent: Literal['buy', 'sell']
    requested_quantity: int
    filled_quantity: int
    average_price: float = 0.0
    top_price: float = 0.0
    fee: float = 0.0
    slippage: float = 0.0
    latency_ms: float = 0.0
    levels_consumed: int = 0
    complete: bool = False
    reason: str = ''


class PaperPosition(BaseModel):
    ticker: str
    side: Literal['yes', 'no']
    quantity: int
    entry_price: float
    entry_fee: float = 0.0
    opened_at: datetime = Field(default_factory=utc_now)
    peak_mark: float = 0.0


class TradeRecord(BaseModel):
    ticker: str
    side: Literal['yes', 'no']
    action: Literal['open', 'close', 'settle']
    quantity: int
    price: float
    pnl: float = 0.0
    fee: float = 0.0
    requested_quantity: int | None = None
    slippage: float = 0.0
    reason: str = ''
    created_at: datetime = Field(default_factory=utc_now)
