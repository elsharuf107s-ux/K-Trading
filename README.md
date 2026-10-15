# K-Trading

K-Trading is a **paper-only AI/quant research engine** for Kalshi's rolling `KXBTC15M` Bitcoin markets.

The system asks a stricter question than “can the model predict BTC?”: **does a forecast retain positive edge after the actual market quote, visible depth, fees, latency, slippage, partial fills, calibration, and hard risk controls are applied?**

> **Status: V0.6 Shadow Lab / paper trading only. Real-money order execution is not implemented.**

## Current stack

```text
KXBTC15M market + full visible order book
             |
             +------------------+
             |                  |
             v                  v
      Kalshi probability      BRTI
             |                  |
             +---- Coinbase/Kraken
                        |
                        v
                  feature engine
                        |
        momentum + target distance
        book imbalance + wave score
                        |
                        v
                 probability model
                        |
                        v
                   hard risk gate
                        |
                        v
               desired order intent
                        |
                        v
              execution simulator
        depth + fees + latency + fills
                        |
                        v
               EXECUTABLE EDGE GATE
                    /         \
                 reject       fill
                   |            |
                   +------v-----+
                          |
                   Shadow Lab log
                          |
              SQLite + analytics + UI
```

## V0.6 Shadow Lab

V0.6 records what happens **between a good-looking forecast and an executable trade**.

For every simulated entry attempt it can store:

- model fair probability
- displayed top-of-book price
- desired contracts
- visible/fillable contracts
- volume-weighted simulated fill price
- gross edge before execution
- executable edge after execution
- taker fee
- depth slippage
- latency slippage
- total slippage
- filled / partial / rejected / no-liquidity status
- final rejection or trade reason

This lets us measure whether the strategy's apparent alpha survives execution instead of judging it from idealized prices.

### Edge attribution

```text
model fair probability
        - top quote
        = gross edge

        - depth impact
        - latency drift
        - transaction fee
        - uncertainty buffer
        = executable edge
```

The dashboard shows average gross edge, executable edge, edge drag, fee drag, depth drag, latency drag, rejected attempts, and no-liquidity attempts.

## Data-health gates

New entries can be blocked when the inputs are not trustworthy enough:

- missing executable YES/NO order-book depth
- excessive disagreement between public BTC spot feeds
- market snapshot older than the configured maximum
- existing final-minute BRTI/reference safeguards
- ordinary spread/risk/drawdown/kill-switch rules

A missing BRTI feed outside the final-minute guard is treated as **degraded proxy mode**, not silently presented as official settlement data.

Default V0.6 settings:

```env
REQUIRE_ORDERBOOK_FOR_ENTRY=true
MIN_EXECUTION_SIDE_DEPTH=1
MAX_SOURCE_DISPERSION=0.003
MAX_MARKET_SNAPSHOT_AGE_SECONDS=5
```

## V0.5 realistic paper execution retained

K-Trading does not assume the full desired order fills at the displayed ask.

It:

- preserves visible YES and NO bid depth
- derives complementary asks for the binary contract
- walks multiple price levels
- supports partial fills
- applies configurable adverse latency drift
- applies configurable fee rates/multipliers
- rejects entries when execution friction destroys the minimum edge
- lets risk exits reduce whatever visible quantity can actually be reached
- carries entry fees in the paper cost basis
- keeps partial-exit P&L internally consistent

Current research defaults:

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

Fee schedules and product-specific multipliers can change, so the values are environment configuration rather than hidden strategy constants.

## V0.4 Model Lab retained

The learning system:

- trains only on completed market windows
- keeps every snapshot from one `KXBTC15M` contract on the same side of a split
- validates chronologically with expanding walk-forward folds
- weights whole markets so densely sampled windows do not dominate
- trains logistic-regression and histogram-gradient-boosting candidates
- compares Brier score and log loss with the live Kalshi probability baseline
- calibrates probabilities on a separate recent holdout
- promotes a challenger only when it beats the configured out-of-sample gate and current champion
- falls back to the deterministic wave model when no champion qualifies

Development defaults:

```env
MODEL_DIR=data/models
ENABLE_LEARNED_MODEL=true
MODEL_MIN_WINDOWS=24
MODEL_MIN_BRIER_GAIN=0.002
```

`24` completed windows is only enough to exercise the pipeline. It is **not evidence of durable edge**. The research target is hundreds to thousands of independent completed windows.

## Market/reference engine

K-Trading currently supports:

- automatic `KXBTC15M` discovery
- public Kalshi order-book reads
- Coinbase + Kraken spot cross-checks
- authenticated Kalshi WebSocket BRTI reference data
- BRTI momentum
- final-minute settlement-window average when supplied by the feed
- target-distance features
- Kalshi probability momentum
- order-book imbalance
- wave-state scoring
- automatic prior-market result reconciliation
- persistent SQLite paper account state
- completed-window replay and labeling
- Brier/calibration analytics

## Run in Codespaces / Linux

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

## BRTI credentials

Keep the private key out of Git:

```env
KALSHI_KEY_ID=your-key-id
KALSHI_PRIVATE_KEY_PATH=/workspaces/K-Trading/kalshi-private-key.pem
ENABLE_KALSHI_WS=true
CFBENCHMARKS_INDEX_ID=BRTI
```

Without a fresh BRTI feed the system can run research features from public spot proxies, but new entries near settlement are blocked by the official-reference guard.

## API

- `GET /api/health`
- `GET /api/state`
- `GET /api/trades`
- `GET /api/execution`
- `GET /api/execution-events`
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

The suite covers, among other things:

- BRTI/reference and wave features
- settlement-window reference selection
- spread/final-minute/data-health risk guards
- paper accounting, settlement, and restart persistence
- current fee-rounding configuration semantics
- complementary YES/NO order-book execution
- multi-level depth impact
- partial fills and partial exits
- depth-vs-latency slippage decomposition
- execution-event persistence and attribution
- completed-window replay and Brier/calibration analytics
- grouped chronological model training
- champion promotion only after out-of-sample improvement

## Research rules

1. **Paper first.**
2. Live-order execution remains disabled.
3. Win rate is not the objective; expected value after costs is.
4. Do not use future/settlement information in a historical decision frame.
5. Keep complete market windows isolated across model train/test boundaries.
6. Reject models that cannot beat simple/live-market baselines out-of-sample.
7. Measure prediction edge separately from execution drag.
8. Treat stale/missing/disagreeing data as a risk condition, not an inconvenience.
9. Backtest or paper profitability does not guarantee future profitability.
10. Kalshi's actual contract rules and current fee schedule control; research settings must be kept current.

## Next milestone — V0.7 empirical execution learning

V0.7 should replace more of the synthetic execution assumptions with measurements from K-Trading's own recorded data:

- sample quote/book movement over 100ms / 250ms / 500ms / 1s horizons
- estimate empirical latency-slippage distributions by regime
- compare predicted vs realized short-horizon order-book movement
- passive maker queue-position experiments
- maker-vs-taker routing research
- automatic retraining trigger after a configurable number of newly settled windows
- long-running soak/failure-injection tests
- execution-drag calibration by liquidity and time-to-expiry

Real execution stays gated until the research and execution stacks survive much larger out-of-sample and paper-soak samples.
