from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import time
from typing import Any

import httpx
from pydantic import BaseModel, Field

from .config import settings
from .models import FeatureVector, Forecast, MarketSnapshot


class SidecarForecast(BaseModel):
    ticker: str
    yes_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    model: str = 'unknown'
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rationale: list[str] = Field(default_factory=list)


class AISidecarClient:
    """Optional asynchronous probability voter.

    The sidecar never receives exchange credentials and never places orders. It
    only receives market/feature history and returns a YES probability. Calls
    run in a background task so a slow AI service can never block the fast
    trading path.

    Recommended sidecar implementation: Amazon Chronos-2 exposed behind a
    small HTTP endpoint. Any service that returns the SidecarForecast schema is
    compatible.
    """

    def __init__(self) -> None:
        self.latest: SidecarForecast | None = None
        self.error: str | None = None
        self.last_request_at: datetime | None = None
        self.last_response_at: datetime | None = None
        self._last_request_monotonic = 0.0
        self._task: asyncio.Task | None = None
        self._client = httpx.AsyncClient(timeout=settings.ai_sidecar_timeout_seconds)

    @property
    def enabled(self) -> bool:
        return settings.has_ai_sidecar

    async def close(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._client.aclose()

    def status(self) -> dict[str, Any]:
        latest = self.latest
        age = None
        if latest:
            age = max(0.0, (datetime.now(timezone.utc) - latest.generated_at).total_seconds())
        return {
            'enabled': self.enabled,
            'configured_model': settings.ai_sidecar_model,
            'url_configured': bool(settings.ai_sidecar_url.strip()),
            'request_in_flight': bool(self._task and not self._task.done()),
            'last_request_at': self.last_request_at.isoformat() if self.last_request_at else None,
            'last_response_at': self.last_response_at.isoformat() if self.last_response_at else None,
            'latest_age_seconds': age,
            'latest': latest.model_dump(mode='json') if latest else None,
            'error': self.error,
            'weight': settings.ai_sidecar_weight,
        }

    def schedule(
        self,
        market: MarketSnapshot,
        features: FeatureVector,
        base_forecast: Forecast,
        *,
        spot_history: list[tuple[float, float]],
        reference_history: list[tuple[float, float]],
        market_history: list[tuple[float, float, float]],
    ) -> bool:
        if not self.enabled:
            return False
        if self._task and not self._task.done():
            return False
        now = time.monotonic()
        if now - self._last_request_monotonic < settings.ai_sidecar_request_interval_seconds:
            return False

        self._last_request_monotonic = now
        self.last_request_at = datetime.now(timezone.utc)
        payload = {
            'model': settings.ai_sidecar_model,
            'market': {
                'ticker': market.ticker,
                'close_time': market.close_time.isoformat() if market.close_time else None,
                'target_price': market.raw_target,
                'seconds_remaining': features.seconds_remaining,
                'yes_bid': features.yes_bid,
                'yes_ask': features.yes_ask,
                'no_bid': features.no_bid,
                'no_ask': features.no_ask,
            },
            'features': features.model_dump(mode='json'),
            'base_forecast': base_forecast.model_dump(mode='json'),
            'history': {
                'spot': spot_history,
                'reference': reference_history,
                'kalshi': market_history,
            },
            'contract': {
                'output': {
                    'ticker': 'string',
                    'yes_probability': 'number 0..1',
                    'confidence': 'number 0..1',
                    'model': 'string',
                    'rationale': ['string'],
                }
            },
        }
        self._task = asyncio.create_task(self._request(payload), name='ai-sidecar-request')
        return True

    async def _request(self, payload: dict[str, Any]) -> None:
        try:
            response = await self._client.post(settings.ai_sidecar_url, json=payload)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and isinstance(data.get('forecast'), dict):
                data = data['forecast']
            forecast = SidecarForecast.model_validate(data)
            expected_ticker = str((payload.get('market') or {}).get('ticker') or '')
            if forecast.ticker != expected_ticker:
                raise ValueError(f'sidecar ticker mismatch: expected {expected_ticker}, got {forecast.ticker}')
            self.latest = forecast
            self.last_response_at = datetime.now(timezone.utc)
            self.error = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'

    def blend(self, base: Forecast, ticker: str) -> Forecast:
        side = self.latest
        if not self.enabled or side is None or side.ticker != ticker:
            return base
        age = max(0.0, (datetime.now(timezone.utc) - side.generated_at).total_seconds())
        if age > settings.ai_sidecar_max_age_seconds:
            return base
        if side.confidence < settings.ai_sidecar_min_confidence:
            return base

        weight = max(0.0, min(0.5, settings.ai_sidecar_weight))
        raw = (1.0 - weight) * base.yes_probability + weight * side.yes_probability
        max_shift = max(0.0, settings.ai_sidecar_max_probability_shift)
        lo = max(0.01, base.yes_probability - max_shift)
        hi = min(0.99, base.yes_probability + max_shift)
        probability = max(lo, min(hi, raw))
        confidence = max(0.0, min(0.95, (1.0 - weight) * base.confidence + weight * side.confidence))

        return Forecast(
            yes_probability=probability,
            confidence=confidence,
            rationale=[
                *base.rationale,
                f'AI sidecar {side.model}: P(YES) {side.yes_probability:.1%} @ {side.confidence:.0%} confidence',
                f'Sidecar blend weight {weight:.0%}; capped shift ±{max_shift:.1%}',
            ],
            model_name=f'{base.model_name}+sidecar:{side.model}',
        )
