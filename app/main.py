from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .engine import TradingEngine

engine = TradingEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.start()
    yield
    await engine.stop()


app = FastAPI(title='K-Trading', version='0.2.0', lifespan=lifespan)

WEB_DIR = Path(__file__).resolve().parent.parent / 'web'
if WEB_DIR.exists():
    app.mount('/static', StaticFiles(directory=WEB_DIR), name='static')


class KillSwitchRequest(BaseModel):
    enabled: bool


@app.get('/')
async def dashboard():
    return FileResponse(WEB_DIR / 'index.html')


@app.get('/api/health')
async def health():
    return {
        'ok': True,
        'status': engine.last_state.get('status'),
        'mode': 'paper',
        'benchmark_stream': engine.benchmark_stream.status(),
    }


@app.get('/api/state')
async def state():
    return engine.last_state


@app.get('/api/trades')
async def trades(limit: int = 50):
    return {'trades': engine.storage.recent_trades(max(1, min(limit, 500)))}


@app.get('/api/observations')
async def observations(limit: int = 200):
    return {'observations': engine.storage.recent_observations(max(1, min(limit, 2000)))}


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
