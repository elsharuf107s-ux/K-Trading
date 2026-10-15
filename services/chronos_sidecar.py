from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_ID = os.getenv('CHRONOS_MODEL_ID', 'amazon/chronos-2')
DEVICE = os.getenv('CHRONOS_DEVICE', 'cpu')
FREQ_SECONDS = max(1, int(os.getenv('CHRONOS_FREQ_SECONDS', '5')))
MAX_CONTEXT_POINTS = max(32, int(os.getenv('CHRONOS_MAX_CONTEXT_POINTS', '512')))
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

pipeline = None


class MarketPayload(BaseModel):
    ticker: str
    close_time: str | None = None
    target_price: float | None = None
    seconds_remaining: float
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float


class HistoryPayload(BaseModel):
    spot: list[list[float]] = Field(default_factory=list)
    reference: list[list[float]] = Field(default_factory=list)
    kalshi: list[list[float]] = Field(default_factory=list)


class ForecastRequest(BaseModel):
    model: str | None = None
    market: MarketPayload
    features: dict[str, Any]
    base_forecast: dict[str, Any]
    history: HistoryPayload
    contract: dict[str, Any] | None = None


class ForecastResponse(BaseModel):
    ticker: str
    yes_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    model: str
    generated_at: datetime
    rationale: list[str]


def _regularize(points: list[list[float]], freq_seconds: int = FREQ_SECONDS) -> pd.DataFrame:
    if len(points) < 4:
        raise ValueError('not enough history points')
    clean: list[tuple[float, float]] = []
    for row in points:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        try:
            ts = float(row[0])
            value = float(row[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(ts) and math.isfinite(value) and value > 0:
            clean.append((ts, value))
    if len(clean) < 4:
        raise ValueError('not enough valid history points')

    df = pd.DataFrame(clean, columns=['timestamp', 'target']).drop_duplicates('timestamp').sort_values('timestamp')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(None)
    df = df.set_index('timestamp').resample(f'{freq_seconds}s').last().interpolate(method='time').ffill().bfill()
    df = df.tail(MAX_CONTEXT_POINTS).reset_index()
    df['item_id'] = 'BTC'
    return df[['item_id', 'timestamp', 'target']]


def _monotone_quantiles(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError('non-finite Chronos quantiles')
    # Small quantile crossings can happen numerically. Enforce a monotone CDF
    # before converting the forecast into P(settlement average > target).
    return np.maximum.accumulate(arr)


def _probability_above(target: float, quantile_values: list[float]) -> float:
    values = _monotone_quantiles(quantile_values)
    q = np.asarray(QUANTILES, dtype=float)
    if target <= values[0]:
        cdf = max(0.01, q[0] * 0.5)
    elif target >= values[-1]:
        cdf = min(0.99, q[-1] + (1.0 - q[-1]) * 0.5)
    else:
        cdf = float(np.interp(target, values, q))
    return float(max(0.01, min(0.99, 1.0 - cdf)))


def _confidence(current: float, q10: float, q90: float, history_points: int, seconds_remaining: float) -> float:
    width_bps = abs(q90 - q10) / max(current, 1e-9) * 10_000.0
    history_factor = min(1.0, history_points / 128.0)
    horizon_factor = max(0.45, 1.0 - min(seconds_remaining, 900.0) / 1800.0)
    interval_factor = max(0.15, 1.0 - width_bps / 120.0)
    score = 0.25 + 0.30 * history_factor + 0.25 * horizon_factor + 0.20 * interval_factor
    return float(max(0.20, min(0.90, score)))


def _load_pipeline():
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=DEVICE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = _load_pipeline()
    yield
    pipeline = None


app = FastAPI(title='K-Trading Chronos Sidecar', version='0.1.0', lifespan=lifespan)


@app.get('/health')
async def health():
    return {
        'ok': pipeline is not None,
        'model': MODEL_ID,
        'device': DEVICE,
        'frequency_seconds': FREQ_SECONDS,
    }


@app.post('/forecast', response_model=ForecastResponse)
async def forecast(body: ForecastRequest) -> ForecastResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail='Chronos model is not loaded')
    target = body.market.target_price
    if target is None or target <= 0:
        raise HTTPException(status_code=422, detail='target_price is required')

    source = 'reference'
    points = body.history.reference
    if len(points) < 16:
        source = 'spot'
        points = body.history.spot
    if len(points) < 16:
        raise HTTPException(status_code=422, detail='At least 16 reference or spot history points are required')

    try:
        context = _regularize(points)
        seconds = max(1.0, float(body.market.seconds_remaining))
        prediction_length = max(1, min(180, int(math.ceil(seconds / FREQ_SECONDS))))
        pred = pipeline.predict_df(
            context,
            prediction_length=prediction_length,
            quantile_levels=QUANTILES,
            id_column='item_id',
            timestamp_column='timestamp',
            target='target',
            freq=f'{FREQ_SECONDS}s',
        )
        if pred.empty:
            raise ValueError('Chronos returned no forecast rows')

        # KXBTC15M settles from the final-minute reference average. Approximate
        # that object directly by averaging each forecast quantile over the last
        # 60 seconds instead of comparing only the terminal point.
        final_steps = max(1, min(len(pred), int(math.ceil(60 / FREQ_SECONDS))))
        tail = pred.tail(final_steps)
        averaged_quantiles = [float(tail[str(q)].mean()) for q in QUANTILES]
        yes_probability = _probability_above(float(target), averaged_quantiles)
        current = float(context['target'].iloc[-1])
        confidence = _confidence(
            current,
            averaged_quantiles[0],
            averaged_quantiles[-1],
            len(context),
            seconds,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Chronos inference failed: {type(exc).__name__}: {exc}') from exc

    return ForecastResponse(
        ticker=body.market.ticker,
        yes_probability=yes_probability,
        confidence=confidence,
        model=MODEL_ID,
        generated_at=datetime.now(timezone.utc),
        rationale=[
            f'Chronos-2 forecasted {source} at {FREQ_SECONDS}s resolution',
            f'Probability derived from forecasted final-60s average vs target ${target:,.2f}',
            f'10–90% final-minute-average band: ${averaged_quantiles[0]:,.2f}–${averaged_quantiles[-1]:,.2f}',
            'Sidecar vote is advisory and remains bounded by K-Trading risk/ensemble caps',
        ],
    )
