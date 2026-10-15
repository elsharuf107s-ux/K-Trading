# K-Trading

K-Trading is a **paper-only AI/quant research engine** for Kalshi's rolling `KXBTC15M` Bitcoin markets.

It is built to answer one question honestly: **does a forecast still have edge after the actual quote, order-book depth, fees, latency, slippage, partial fills, calibration, and risk controls are applied?**

> **Current build: V0.9 Fast Path + AI Ensemble. Real-money order submission is still intentionally disabled.**

## V0.9 — Fast Path + AI Ensemble

V0.9 separates slow data/reconciliation work from the trading decision path.

The normal engine loop still handles REST market discovery, Coinbase/Kraken cross-checks, settlement reconciliation, database analytics, and model training. The new fast path consumes already-synchronized in-memory Kalshi WebSocket order-book data plus the latest BRTI reference and evaluates the paper strategy at **200ms / 5Hz by default**.

That means lowering the fast-path interval does **not** multiply REST traffic.

V0.9 adds:

- 200ms in-memory paper decision path
- WebSocket-first quote/depth decisions
- separate fast-path latency telemetry (`last_cycle_ms`, EWMA compute time, cycles, skips, trades)
- 1-second fast observation persistence for higher-resolution training data
- bounded asynchronous AI sidecar interface
- optional Amazon Chronos-2 sidecar service
- AI failure isolation: slow/offline sidecar never blocks the fast trader
- sidecar confidence gate, freshness gate, blend weight, and maximum probability-shift cap
- V0.9 dashboard panel injected without replacing the existing research UI
- `/api/fast-path` and `/api/ai-sidecar` status endpoints

### Default fast path

```env
FAST_PATH_ENABLED=true
FAST_PATH_INTERVAL_MS=200
FAST_PATH_REQUIRE_STREAM=true
FAST_PATH_OBSERVATION_INTERVAL_SECONDS=1
FAST_PATH_MAX_SLOW_STATE_AGE_SECONDS=4
```

The dashboard should report roughly **5 Hz**. `Cycle compute` is the time spent computing each fast decision; it should remain comfortably below the configured 200ms interval.

## Recommended outcome AI: Amazon Chronos-2

The recommended sidecar for the 15-minute BTC outcome problem is **Amazon Chronos-2** (`amazon/chronos-2`). It is a purpose-built probabilistic time-series foundation model, not a general chat model.

K-Trading sends the sidecar recent BRTI/spot/Kalshi history. The included service forecasts the remaining time series, approximates the final 60-second settlement average, and returns a `P(YES)` vote.

Chronos does **not** control execution. Its vote is blended into the local fast model and then still passes through the same risk and execution layers.

Defaults:

```env
AI_SIDECAR_ENABLED=false
AI_SIDECAR_URL=
AI_SIDECAR_MODEL=amazon/chronos-2
AI_SIDECAR_WEIGHT=0.20
AI_SIDECAR_MIN_CONFIDENCE=0.55
AI_SIDECAR_REQUEST_INTERVAL_SECONDS=5
AI_SIDECAR_TIMEOUT_SECONDS=2
AI_SIDECAR_MAX_AGE_SECONDS=15
AI_SIDECAR_MAX_PROBABILITY_SHIFT=0.08
```

Full setup: [`docs/AI_SIDECAR.md`](docs/AI_SIDECAR.md)

The optional model dependencies are deliberately separate from the main trading process:

```bash
pip install -r requirements-chronos.txt
CHRONOS_DEVICE=cpu bash scripts/run_chronos_sidecar.sh
```

Then configure:

```env
AI_SIDECAR_ENABLED=true
AI_SIDECAR_URL=http://127.0.0.1:8010/forecast
```

For a lower-resource CPU experiment, use `CHRONOS_MODEL_ID=autogluon/chronos-2-small`. The full `amazon/chronos-2` model is the quality-oriented default.

## V0.8 — Strategy / Soak Lab

V0.8 adds a chronological strategy evaluator on top of the paper engine. It:

- replays recorded completed-market frames in timestamp order
- keeps final YES/NO settlement out of entry/exit decisions
- sizes replay trades at one contract per completed market
- uses the same depth-aware fee/latency execution simulator
- tracks P&L, win rate, fees, hold time, maximum drawdown, and loss streaks
- breaks performance down by edge bucket, entry time, wave strength, reference source, side, and exit reason
- stores the last report at `data/strategy_eval.json`
- exposes explicit PASS/FAIL research gates

Default gate:

```env
SOAK_MIN_COMPLETED_WINDOWS=200
SOAK_MIN_REPLAY_TRADES=50
SOAK_MIN_BRIER_GAIN=0.0
SOAK_REQUIRE_POSITIVE_REPLAY_PNL=true
SOAK_MAX_REPLAY_DRAWDOWN_FRACTION=0.10
SOAK_MAX_CONSECUTIVE_LOSSES=10
SOAK_MIN_EXECUTION_FILL_RATE=0.50
SOAK_MIN_EMPIRICAL_SAMPLES=500
```

Passing means **GO_FOR_MORE_PAPER_RESEARCH**, not permission to trade real money.

## Earlier layers retained

### V0.7 — empirical execution learning

- authenticated Kalshi `orderbook_delta` WebSocket
- snapshot + sequenced deltas and sequence-gap resync
- measured 100/250/500/1000 ms quote movement
- persistent empirical quote-move distributions
- p95 adverse-movement latency recommendation
- optional empirical latency override
- automatic model-lab retraining

### V0.6 — shadow edge attribution

- records rejected as well as filled entry attempts
- separates gross model edge from executable edge
- measures fee, depth, and latency drag
- blocks unhealthy source/depth conditions

### V0.5 — realistic paper execution

- walks visible book depth
- supports partial fills
- models fees and latency
- rejects trades whose edge disappears after execution friction

### V0.4 — model lab

- trains only on completed markets
- keeps complete market windows together across splits
- uses chronological walk-forward validation
- compares candidates against the Kalshi probability baseline
- promotes only after configured out-of-sample improvement

## Run K-Trading

In GitHub Codespaces or Linux:

```bash
cp .env.example .env
pip install -r requirements.txt
./scripts/run.sh
```

Open forwarded port **8000**.

Direct run:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Useful endpoints:

```text
GET  /api/health
GET  /api/state
GET  /api/fast-path
GET  /api/ai-sidecar
GET  /api/analytics
GET  /api/execution
GET  /api/empirical-execution
GET  /api/model-lab
POST /api/model-lab/train
GET  /api/strategy-eval
POST /api/strategy-eval/run
GET  /api/research-gate
```

## Kalshi authenticated feeds

Keep credentials outside Git:

```env
KALSHI_KEY_ID=your-key-id
KALSHI_PRIVATE_KEY_PATH=/workspaces/K-Trading/kalshi-private-key.pem
ENABLE_KALSHI_WS=true
CFBENCHMARKS_INDEX_ID=BRTI
```

These credentials power authenticated BRTI/reference and order-book market-data streaming. They do **not** enable live order submission.

## Tests

```bash
pytest -q
python -m compileall app tests services
```

The suite covers reference/BRTI handling, order-book streaming and gap invalidation, realistic execution/accounting, empirical latency, grouped model training, chronological replay, settlement no-lookahead, research gates, V0.9 buffer handling, bounded AI sidecar blending, and proof that a fast-path decision does not call REST market discovery.

## Safety / research rules

1. Paper first.
2. Real-money order submission remains disabled.
3. Win rate alone is not the objective; after-cost expected value is.
4. Never use final settlement data to influence a historical decision frame.
5. Never treat a small backtest/replay sample as proof of edge.
6. Keep model error and execution drag separate.
7. Treat stale, missing, disagreeing, or sequence-broken data as a risk condition.
8. A slow or broken AI sidecar must never block or control execution.
9. A passing research gate means continue paper research, not go live.
