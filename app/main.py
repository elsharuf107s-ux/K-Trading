from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .analytics import ResearchAnalytics
from .engine import TradingEngine
from .model_lab import ModelLab
from .strategy_eval import ResearchGate, StrategyEvaluator

engine = TradingEngine()
analytics = ResearchAnalytics(engine.storage)
model_lab = ModelLab(engine.storage, engine.model_registry)
strategy_eval = StrategyEvaluator(engine.storage)
research_gate = ResearchGate(engine.storage, strategy_eval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.start()
    yield
    await engine.stop()


app = FastAPI(title='K-Trading', version='0.8.0', lifespan=lifespan)

WEB_DIR = Path(__file__).resolve().parent.parent / 'web'
if WEB_DIR.exists():
    app.mount('/static', StaticFiles(directory=WEB_DIR), name='static')


class KillSwitchRequest(BaseModel):
    enabled: bool


@app.get('/')
async def dashboard():
    return FileResponse(WEB_DIR / 'index.html')


def _gate_payload(strategy_report=None):
    return research_gate.evaluate(
        analytics.summary(),
        engine.empirical.summary(engine.orderbook_stream.horizons_ms),
        strategy_report=strategy_report,
    )


@app.get('/api/health')
async def health():
    return {
        'ok': True,
        'status': engine.last_state.get('status'),
        'mode': 'paper',
        'version': app.version,
        'benchmark_stream': engine.benchmark_stream.status(),
        'orderbook_stream': engine.orderbook_stream.status(),
        'data_health': engine.last_state.get('data_health'),
        'model_registry': engine.model_registry.status(),
        'auto_training': engine.last_state.get('auto_training'),
        'execution': engine.storage.execution_metrics(),
        'execution_attribution': engine.storage.execution_attribution(),
        'empirical_latency': engine.last_state.get('empirical_latency_model') or engine._latency_model(),
        'research_gate': _gate_payload(),
    }


@app.get('/api/state')
async def state():
    return engine.last_state


@app.get('/api/trades')
async def trades(limit: int = 50):
    return {'trades': engine.storage.recent_trades(max(1, min(limit, 500)))}


@app.get('/api/execution')
async def execution_metrics():
    return {
        'fills': engine.storage.execution_metrics(),
        'attribution': engine.storage.execution_attribution(),
        'empirical': engine.empirical.summary(engine.orderbook_stream.horizons_ms),
        'latency_model': engine._latency_model(),
        'empirical_latency_enabled': bool(engine.last_state.get('empirical_latency_enabled')),
    }


@app.get('/api/execution-events')
async def execution_events(limit: int = 100):
    return {'events': engine.storage.recent_execution_events(max(1, min(limit, 2000)))}


@app.get('/api/empirical-execution')
async def empirical_execution():
    return {
        'stream': engine.orderbook_stream.status(),
        'summary': engine.empirical.summary(engine.orderbook_stream.horizons_ms),
        'latency_model': engine.last_state.get('empirical_latency_model') or engine._latency_model(),
        'enabled_for_simulation': bool(engine.last_state.get('empirical_latency_enabled')),
    }


@app.get('/api/observations')
async def observations(limit: int = 200):
    return {'observations': engine.storage.recent_observations(max(1, min(limit, 2000)))}


@app.get('/api/analytics')
async def research_analytics():
    return analytics.summary()


@app.get('/api/model-lab')
async def model_lab_status():
    return {
        **engine.model_registry.status(),
        'auto_training': engine.last_state.get('auto_training') or engine._auto_train_state,
    }


@app.post('/api/model-lab/train')
async def train_model_lab():
    try:
        return await asyncio.to_thread(model_lab.train)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Model lab failed: {type(exc).__name__}: {exc}') from exc


@app.get('/api/strategy-eval')
async def strategy_evaluation():
    report = strategy_eval.last_report()
    return report or {
        'status': 'not_run',
        'message': 'Run POST /api/strategy-eval/run after completed windows have been collected.',
    }


@app.post('/api/strategy-eval/run')
async def run_strategy_evaluation():
    try:
        report = await asyncio.to_thread(strategy_eval.run)
        return {**report, 'research_gate': _gate_payload(report)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Strategy evaluation failed: {type(exc).__name__}: {exc}') from exc


@app.get('/api/research-gate')
async def research_gate_status():
    return _gate_payload()


@app.get('/api/replay/{ticker}')
async def replay(ticker: str):
    data = engine.storage.replay_window(ticker)
    if data is None:
        raise HTTPException(status_code=404, detail='No recorded observations for that ticker')
    return data


@app.post('/api/cycle')
async def cycle():
    try:
        return await engine.cycle()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post('/api/kill-switch')
async def kill_switch(body: KillSwitchRequest):
    engine.set_kill_switch(body.enabled)
    return {'enabled': engine.paper.kill_switch}
