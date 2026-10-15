# K-Trading

K-Trading is a **paper-only research and learning engine** for Kalshi's rolling `KXBTC15M` Bitcoin markets.

The goal is not to maximize win rate. The goal is to determine whether a probability model can beat the live Kalshi market **out-of-sample and after spread, fees, depth, latency, slippage, partial fills, and hard risk controls**.

> **Status: V0.5 / paper trading only. Real-money order execution is not implemented.**

## What V0.5 does

### Live research engine
- Discovers the current `KXBTC15M` market.
- Reads Kalshi's public YES/NO bid order book and preserves visible depth.
- Derives opposite-side asks from binary-market complement prices.
- Reads Coinbase + Kraken as spot cross-checks.
- Optionally authenticates to Kalshi WebSocket data and streams CF Benchmarks `BRTI`.
- Uses BRTI as the primary reference and momentum source when available.
- Uses the 15-minute settlement-window average during the final minute when supplied by the feed.
- Builds BTC/BRTI momentum, Kalshi probability momentum, order-book imbalance, target distance, and a wave score.
- Enforces hard spread, expiry, bankroll, drawdown, stop, trailing-profit, wave-alignment, and kill-switch rules.

### V0.5 execution simulator
- Walks visible Kalshi depth instead of filling the full order at the displayed best quote.
- Converts NO bids into executable YES asks and YES bids into executable NO asks.
- Supports partial fills when visible depth is insufficient.
- Applies configurable adverse latency drift.
- Applies configurable general taker/maker fee rates and multipliers.
- Uses the July 2026 general event-contract fee rounding rule: fee plus position cost is rounded up to a centicent (`$0.0001`).
- Re-checks entry edge **after** simulated price impact, fees, and latency.
- Rejects new entries when executable edge falls below the strategy threshold.
- On risk exits, sells only the visible quantity rather than inventing a full fill.
- Tracks requested quantity, filled quantity, average fill price, fee, slippage, fill rate, and cumulative fee drag.
- Keeps entry fees in position cost basis so partial exits and settlement P&L stay internally consistent.

Fee schedules can change and some series have special multipliers. V0.5 therefore exposes fee rates/multipliers through environment settings instead of hard-wiring them into strategy logic.

### V0.4 Model Lab retained
- Builds training data only from **completed** markets.
- Keeps all snapshots from one market window on the same side of a train/test split.
- Uses chronological expanding walk-forward validation rather than random row splitting.
- Gives each market equal total sample weight.
- Trains logistic-regression and histogram-gradient-boosting candidates.
- Scores candidates on Brier score/log loss against Kalshi's market probability.
- Uses a disjoint recent holdout for probability calibration.
- Promotes a model only after configured out-of-sample improvement and champion checks.
- Automatically falls back to the deterministic wave model when there is no eligible champion.

## Core loop

```text
KALSHI KXBTC15M + BRTI + SPOT
              |
              v
         feature engine
              |
              v
       probability model
              |
              v
        hard risk gate
              |
              v
     desired order quantity
              |
              v
      EXECUTION SIMULATOR
       /      |       \
    depth    fees    latency
      |        |        |
      +--------+--------+
              |
              v
      executable edge gate
          /          \
       reject         fill
                       |
                  partial/full
                       |
                       v
                  paper broker
                       |
                       v
              SQLite + analytics
```

## Run in Codespaces / Linux

```bash
cp .env.example .env
pip install -r requirements.txt
./scripts/run.sh
```

Open forwarded port **8000**.

Or run directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Official BRTI stream

Keep the Kalshi private key outside Git:

```env
KALSHI_KEY_ID=your-key-id
KALSHI_PRIVATE_KEY_PATH=/workspaces/K-Trading/kalshi-private-key.pem
ENABLE_KALSHI_WS=true
CFBENCHMARKS_INDEX_ID=BRTI
```

Without a fresh BRTI feed, K-Trading falls back to Coinbase/Kraken for research features and **blocks new final-minute entries**.

## Model Lab settings

```env
MODEL_DIR=data/models
ENABLE_LEARNED_MODEL=true
MODEL_MIN_WINDOWS=24
MODEL_MIN_BRIER_GAIN=0.002
```

`MODEL_MIN_WINDOWS=24` is a development floor, not evidence of durable edge. The research target is hundreds to thousands of independent completed windows.

## V0.5 execution settings

```env
EXECUTION_BOOK_DEPTH=20
SIM_TAKER_FEE_RATE=0.07
SIM_TAKER_FEE_MULTIPLIER=1.0
SIM_MAKER_FEE_RATE=0.0175
SIM_MAKER_FEE_MULTIPLIER=0.0
SIM_LATENCY_MS=250
SIM_LATENCY_PRICE_DRIFT_PER_SECOND=0.004
SIM_MAX_LATENCY_SLIPPAGE=0.01
SIM_MAX_EXECUTION_SLIPPAGE=0.03
SIM_MIN_FILL_FRACTION=0.25
```

The current strategy submits simulated **taker** orders. Maker-rate settings are present for later passive-order experiments; a zero maker multiplier means no generic maker fee is assumed unless the applicable fee schedule says otherwise.

## Main API routes

- `GET /api/health`
- `GET /api/state`
- `GET /api/trades`
- `GET /api/execution`
- `GET /api/observations`
- `GET /api/analytics`
- `GET /api/model-lab`
- `POST /api/model-lab/train`
- `GET /api/replay/{ticker}`
- `POST /api/cycle`
- `POST /api/kill-switch`

## Tests

```bash
pytest -q
```

Coverage includes:
- BRTI/reference and wave features
- settlement-window reference selection
- risk and final-minute guards
- paper accounting, settlement, and restart persistence
- CF Benchmarks payload parsing
- completed-window labeling/replay
- Brier/calibration analytics
- grouped walk-forward model promotion
- centicent fee rounding
- complementary YES/NO order-book execution
- multi-level price impact
- partial fills and partial exits
- fee-aware P&L/cost basis
- SQLite migration for execution fields
- fill-rate/slippage execution metrics

## Safety / research rules

1. **Paper first.**
2. Real-money order execution does not exist in V0.5.
3. Never optimize primarily for win rate.
4. Judge results after fees, spread, latency, slippage, and incomplete fills.
5. Never inject future/settlement data into a historical decision frame.
6. Keep entire market windows isolated across model train/test boundaries.
7. A candidate that cannot beat the live-market probability baseline out-of-sample does not get promoted.
8. A backtest or paper result is not proof of future profitability.
9. Kalshi's actual contract rules and current fee schedule control; simulator settings must be kept current.

## Next milestone — V0.6 shadow/live-readiness lab

- record full-depth book snapshots around every decision
- measure real quote movement over configurable latency horizons
- replace synthetic latency drift with empirical slippage distributions
- model queue position for passive maker orders
- maker-vs-taker strategy experiments
- decompose model alpha vs execution drag per trade
- data-feed health scoring and stale-book rejection
- automatic model retraining schedule after enough new completed windows
- longer paper soak tests and failure-injection tests

Real execution stays gated until the research stack demonstrates durable out-of-sample behavior and the execution simulator survives long-running paper tests.
