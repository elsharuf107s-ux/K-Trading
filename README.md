# K-Trading

K-Trading is a **paper-only AI/quant research engine** for Kalshi's rolling `KXBTC15M` Bitcoin markets.

It is built to answer one question honestly: **does a forecast still have edge after the actual quote, order-book depth, fees, latency, slippage, partial fills, calibration, and risk controls are applied?**

> **Current build: V0.8 Strategy / Soak Lab. Real-money order submission is not implemented.**

## What is finished in V0.8

V0.8 adds a chronological strategy evaluator on top of the existing paper engine.

It replays recorded completed-market frames in timestamp order and:

- uses only the information that existed in each recorded frame
- keeps final YES/NO settlement out of entry/exit decisions
- sizes replay trades at one contract per completed market so strategy quality is not confused with bankroll sizing
- runs the same depth-aware fee/latency execution simulator used by the paper engine
- tracks replay P&L, win rate, direction accuracy, fees, hold time, maximum drawdown, and consecutive losses
- breaks performance down by edge bucket, entry time, wave strength, reference source, side, and exit reason
- stores the last report at `data/strategy_eval.json`
- exposes explicit research gates instead of presenting raw P&L as permission to go live

### Research gate defaults

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

Passing these gates only means **GO_FOR_MORE_PAPER_RESEARCH**. It does not approve or prove safe/profitable live trading.

## Earlier layers retained

### V0.7 — empirical execution learning

- authenticated Kalshi `orderbook_delta` WebSocket for the active contract
- initial snapshot + sequenced deltas
- sequence-gap detection and forced resync
- fresh WebSocket book preferred over REST fallback
- measured 100/250/500/1000 ms quote movement
- SQLite persistence of empirical quote moves
- p95 adverse-movement latency recommendation
- optional empirical latency override, disabled by default until the sample gate is met
- automatic model-lab retraining after configurable newly completed windows

### V0.6 — shadow edge attribution

- records entry execution attempts even when they are rejected
- separates gross model edge from executable edge
- tracks fee drag, depth slippage, latency slippage, rejected attempts, and no-liquidity attempts
- blocks entries when source agreement or executable depth is unhealthy

### V0.5 — realistic paper execution

- walks visible YES/NO depth instead of filling everything at the top quote
- supports partial fills
- models configurable fees and latency
- rejects entries when execution friction destroys the minimum edge
- carries entry fees through paper P&L and partial exits

### V0.4 — model lab

- trains only on completed markets
- keeps whole market windows together across train/test splits
- uses chronological expanding walk-forward validation
- compares learned challengers against the live Kalshi probability baseline
- promotes a model only after configured out-of-sample improvement

## Run it

In GitHub Codespaces or Linux:

```bash
cp .env.example .env
pip install -r requirements.txt
./scripts/run.sh
```

Then open forwarded port **8000**.

Direct run:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The dashboard is at `/`.

## Test V0.8

1. Start the server and open port 8000.
2. Let K-Trading collect observations and completed markets.
3. In the dashboard, click **Run Strategy Eval**.
4. Review the replay metrics and the individual PASS/FAIL research gates.
5. For raw JSON, use:

```text
GET  /api/strategy-eval
POST /api/strategy-eval/run
GET  /api/research-gate
GET  /api/health
GET  /api/state
GET  /api/analytics
GET  /api/execution
GET  /api/empirical-execution
```

If you do not have many completed windows yet, the evaluator can still run, but the gate should remain `NOT_READY`.

## Kalshi authenticated feeds

Keep credentials outside Git:

```env
KALSHI_KEY_ID=your-key-id
KALSHI_PRIVATE_KEY_PATH=/workspaces/K-Trading/kalshi-private-key.pem
ENABLE_KALSHI_WS=true
CFBENCHMARKS_INDEX_ID=BRTI
```

These credentials power BRTI/reference and order-book market-data streaming. They do **not** enable live order submission in this build.

## Important V0.7 settings

```env
ORDERBOOK_STREAM_STALE_SECONDS=2
EMPIRICAL_QUOTE_HORIZONS_MS=100,250,500,1000
EMPIRICAL_LATENCY_MIN_SAMPLES=200
USE_EMPIRICAL_LATENCY=false
AUTO_MODEL_TRAIN=true
AUTO_TRAIN_EVERY_WINDOWS=16
```

Keep `USE_EMPIRICAL_LATENCY=false` until you have enough real measurements and have reviewed them.

## Tests

```bash
pytest -q
python -m compileall app tests
```

The test suite covers:

- reference/BRTI handling
- Kalshi order-book snapshot and delta parsing
- sequence-gap invalidation
- fee rounding and paper accounting
- depth-aware and partial fills
- depth-vs-latency slippage attribution
- shadow execution persistence
- empirical latency sampling and recommendations
- grouped chronological model training
- strategy replay chronology
- settlement no-lookahead behavior
- drawdown/loss-streak calculations
- research-gate pass/fail logic

## Safety / research rules

1. Paper first.
2. Real-money order submission remains disabled.
3. Win rate alone is not the objective; after-cost expected value is.
4. Never use final settlement data to influence a historical decision frame.
5. Never treat a small backtest or replay sample as proof of edge.
6. Keep model error and execution drag separate.
7. Treat stale, missing, disagreeing, or sequence-broken data as a risk condition.
8. Keep empirical latency disabled until the measured sample is large enough.
9. A passing V0.8 gate means continue paper research, not go live.
