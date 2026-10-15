from __future__ import annotations

import asyncio
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .analytics import ResearchAnalytics
from .engine import TradingEngine
from .fast_path import FastPathController
from .model_lab import ModelLab
from .strategy_eval import ResearchGate, StrategyEvaluator

engine = TradingEngine()
fast_path = FastPathController(engine)
analytics = ResearchAnalytics(engine.storage)
model_lab = ModelLab(engine.storage, engine.model_registry)
strategy_eval = StrategyEvaluator(engine.storage)
research_gate = ResearchGate(engine.storage, strategy_eval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.start()
    await fast_path.start()
    yield
    await fast_path.stop()
    await engine.stop()


app = FastAPI(title='K-Trading', version='0.9.0', lifespan=lifespan)

WEB_DIR = Path(__file__).resolve().parent.parent / 'web'
if WEB_DIR.exists():
    app.mount('/static', StaticFiles(directory=WEB_DIR), name='static')


class KillSwitchRequest(BaseModel):
    enabled: bool


@app.get('/')
async def dashboard():
    html = (WEB_DIR / 'index.html').read_text(encoding='utf-8')
    html = html.replace(
        '<span class="pill">V0.8</span>',
        '<span class="pill">V0.9</span><span class="pill" id="fastBadge">FAST —</span>',
        1,
    )
    html = html.replace('</body>', '<script src="/static/v09.js"></script></body>', 1)
    return HTMLResponse(html)


def _json_safe(value: Any) -> Any:
    """Recursively remove values that strict JSON encoders cannot serialize."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _gate_payload(strategy_report=None):
    return research_gate.evaluate(
        analytics.summary(),
        engine.empirical.summary(engine.orderbook_stream.horizons_ms),
        strategy_report=strategy_report,
    )


@app.get('/api/health')
async def health():
    return _json_safe({
        'ok': True,
        'status': engine.last_state.get('status'),
        'mode': 'paper',
        'version': app.version,
        'benchmark_stream': engine.benchmark_stream.status(),
        'orderbook_stream': engine.orderbook_stream.status(),
        'fast_path': fast_path.status(),
        'ai_sidecar': fast_path.sidecar.status(),
        'data_health': engine.last_state.get('data_health'),
        'model_registry': engine.model_registry.status(),
        'auto_training': engine.last_state.get('auto_training'),
        'execution': engine.storage.execution_metrics(),
        'execution_attribution': engine.storage.execution_attribution(),
        'empirical_latency': engine.last_state.get('empirical_latency_model') or engine._latency_model(),
        'research_gate': _gate_payload(),
    })


@app.get('/api/state')
async def state():
    state = dict(engine.last_state)
    state['fast_path'] = fast_path.status()
    state['ai_sidecar'] = fast_path.sidecar.status()
    return _json_safe(state)


@app.get('/api/fast-path')
async def fast_path_status():
    return _json_safe(fast_path.status())


@app.get('/api/ai-sidecar')
async def ai_sidecar_status():
    return _json_safe(fast_path.sidecar.status())


@app.get('/api/trades')
async def trades(limit: int = 50):
    return _json_safe({'trades': engine.storage.recent_trades(max(1, min(limit, 500)))})


@app.get('/api/execution')
async def execution_metrics():
    return _json_safe({
        'fills': engine.storage.execution_metrics(),
        'attribution': engine.storage.execution_attribution(),
        'empirical': engine.empirical.summary(engine.orderbook_stream.horizons_ms),
        'latency_model': engine._latency_model(),
        'empirical_latency_enabled': bool(engine.last_state.get('empirical_latency_enabled')),
    })


@app.get('/api/execution-events')
async def execution_events(limit: int = 100):
    return _json_safe({'events': engine.storage.recent_execution_events(max(1, min(limit, 2000)))})


@app.get('/api/empirical-execution')
async def empirical_execution():
    return _json_safe({
        'stream': engine.orderbook_stream.status(),
        'summary': engine.empirical.summary(engine.orderbook_stream.horizons_ms),
        'latency_model': engine.last_state.get('empirical_latency_model') or engine._latency_model(),
        'enabled_for_simulation': bool(engine.last_state.get('empirical_latency_enabled')),
    })


@app.get('/api/observations')
async def observations(limit: int = 200):
    return _json_safe({'observations': engine.storage.recent_observations(max(1, min(limit, 2000)))})


@app.get('/api/analytics')
async def research_analytics():
    return _json_safe(analytics.summary())


@app.get('/api/model-lab')
async def model_lab_status():
    return _json_safe({
        **engine.model_registry.status(),
        'auto_training': engine.last_state.get('auto_training') or engine._auto_train_state,
    })


@app.post('/api/model-lab/train')
async def train_model_lab():
    try:
        return _json_safe(await asyncio.to_thread(model_lab.train))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Model lab failed: {type(exc).__name__}: {exc}') from exc


@app.get('/api/strategy-eval')
async def strategy_evaluation():
    report = strategy_eval.last_report()
    return _json_safe(report or {
        'status': 'not_run',
        'message': 'Run POST /api/strategy-eval/run after completed windows have been collected.',
    })


@app.post('/api/strategy-eval/run')
async def run_strategy_evaluation():
    try:
        report = await asyncio.to_thread(strategy_eval.run)
        payload = {**report, 'research_gate': _gate_payload(report)}
        return _json_safe(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Strategy evaluation failed: {type(exc).__name__}: {exc}') from exc


@app.get('/api/research-gate')
async def research_gate_status():
    return _json_safe(_gate_payload())


@app.get('/api/replay/{ticker}')
async def replay(ticker: str):
    data = engine.storage.replay_window(ticker)
    if data is None:
        raise HTTPException(status_code=404, detail='No recorded observations for that ticker')
    return _json_safe(data)


@app.post('/api/cycle')
async def cycle():
    try:
        return _json_safe(await engine.cycle())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post('/api/kill-switch')
async def kill_switch(body: KillSwitchRequest):
    engine.set_kill_switch(body.enabled)
    return {'enabled': engine.paper.kill_switch}
