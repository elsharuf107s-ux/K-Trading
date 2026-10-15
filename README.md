# K-Trading

K-Trading is a **paper-only AI/quant research engine** for Kalshi's rolling `KXBTC15M` Bitcoin markets.

The core question is stricter than “can a model predict BTC?”: **does a forecast retain positive expected edge after the actual Kalshi quote, visible depth, fees, latency, slippage, partial fills, calibration, and hard risk controls are applied?**

> **Status: V0.7 Empirical Execution Lab / paper trading only. Real-money order submission is not implemented.**

## V0.7 — empirical execution learning

V0.7 stops treating latency as something we can only guess.

When Kalshi API credentials are configured, K-Trading now opens an authenticated `orderbook_delta` WebSocket subscription for the active `KXBTC15M` contract. The stream receives an initial book snapshot and then sequenced deltas.

The system:

- keeps a synchronized local YES/NO order book
- detects sequence gaps and forces a resync/reconnect instead of trusting corrupted local state
- prefers the fresh streaming book over REST depth
- falls back to REST order-book reads when the stream is not fresh
- records short-horizon executable quote movement
- persists those measurements in SQLite
- summarizes adverse movement at configurable latency horizons
- calculates a p95 adverse-movement recommendation for the configured simulated latency
- keeps empirical latency **disabled by default** until enough real samples exist and the operator explicitly enables it
- can automatically rerun the existing grouped model lab after enough newly settled markets accumulate

Default empirical horizons:

```env
ORDERBOOK_STREAM_STALE_SECONDS=2
EMPIRICAL_QUOTE_HORIZONS_MS=100,250,500,1000
EMPIRICAL_LATENCY_MIN_SAMPLES=200
USE_EMPIRICAL_LATENCY=false
AUTO_MODEL_TRAIN=true
AUTO_TRAIN_EVERY_WINDOWS=16
```

`USE_EMPIRICAL_LATENCY=false` is intentional. Collect and inspect enough observations first. When enabled after the sample gate, V0.7 can substitute the measured p95 adverse quote move for the older synthetic latency penalty.

### Empirical latency loop

```text
Kalshi orderbook_snapshot
          |
          v
orderbook_delta stream
          |
          v
sequenced local book
          |
          +---------------------------+
          |                           |
          v                           v
 current execution book        timestamped quotes
                                      |
                           100/250/500/1000ms moves
                                      |
                                      v
                                 SQLite history
                                      |
                                      v
                              empirical distribution
                                      |
                            p95 adverse quote move
                                      |
                           sample gate + explicit flag
                                      |
                                      v
                           paper execution simulator
```

## V0.6 — shadow edge attribution

Every simulated execution attempt can be recorded even when no paper trade is ultimately opened.

Tracked fields include:

- fair/model probability
- top quote
- requested and fillable contracts
- volume-weighted simulated price
- gross edge before execution
- executable edge after execution
- transaction fee
- depth slippage
- latency slippage
- total slippage
- filled / partial / rejected / no-liquidity status
- rejection/trade reason

This separates **forecast alpha** from **execution drag**.

```text
model fair probability
        - top quote
        = gross edge

        - depth impact
        - latency impact
        - fee
        - uncertainty buffer
        = executable edge
```

## V0.6 data-health gates

New entries can be blocked for:

- missing executable YES/NO depth
- excessive Coinbase/Kraken disagreement
- stale market snapshots
- spread violations
- final-minute missing official reference
- account drawdown / daily loss / kill switch

The dashboard reports whether the current book came from the WebSocket stream or REST fallback, plus any sequence gaps seen by the stream.

## V0.5 — realistic paper fills

The paper broker does **not** assume an entire desired order fills at one displayed price.

It:

- preserves visible YES/NO depth
- derives complementary asks in the binary market
- walks multiple levels
- supports partial fills
- applies configurable fees
- applies latency slippage
- rejects entries when execution friction erases the minimum edge
- permits partial risk exits when only part of the position can be reached
- carries entry fees through cost basis and settlement P&L

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

Fee schedules and series-specific multipliers can change, so fee parameters are environment configuration rather than hidden strategy constants.

## V0.4 — model lab

The learning system:

- trains only on completed market windows
- keeps all observations from one 15-minute contract together across train/test boundaries
- validates chronologically with expanding walk-forward folds
- weights complete windows so high-frequency samples from one market do not dominate
- trains logistic-regression and histogram-gradient-boosting challengers
- compares Brier score/log loss with the Kalshi probability baseline
- calibrates on a disjoint recent holdout
- promotes a challenger only after configured out-of-sample improvement
- falls back to the deterministic wave baseline when no champion qualifies

Development defaults:

```env
MODEL_DIR=data/models
ENABLE_LEARNED_MODEL=true
MODEL_MIN_WINDOWS=24
MODEL_MIN_BRIER_GAIN=0.002
```

`24` windows is a pipeline-development floor, not evidence of durable edge.

## Market / reference engine

K-Trading currently includes:

- automatic current `KXBTC15M` discovery
- Kalshi REST order-book fallback
- authenticated real-time order-book stream
- authenticated CF Benchmarks BRTI stream
- Coinbase + Kraken spot cross-checks
- BRTI momentum
- final-minute settlement-window average when supplied by the feed
- target-distance, market-momentum and order-book features
- wave score / wave state
- automatic prior-market settlement reconciliation
- persistent paper account state
- window replay
- Brier/calibration analytics
- execution-attribution analytics
- empirical quote-movement analytics
- champion/challenger model registry
- automatic retraining trigger after configurable new settled-window counts

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

## Enable BRTI + order-book streaming

Keep the Kalshi private key outside Git:

```env
KALSHI_KEY_ID=your-key-id
KALSHI_PRIVATE_KEY_PATH=/workspaces/K-Trading/kalshi-private-key.pem
ENABLE_KALSHI_WS=true
CFBENCHMARKS_INDEX_ID=BRTI
```

The same authenticated WebSocket capability powers the BRTI reference stream and V0.7 order-book stream. No credential is exposed to the browser UI.

## API

- `GET /api/health`
- `GET /api/state`
- `GET /api/trades`
- `GET /api/execution`
- `GET /api/execution-events`
- `GET /api/empirical-execution`
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

- BRTI reference / momentum handling
- final-minute settlement-window selection
- spread, drawdown, source-dispersion and book-depth risk guards
- paper accounting, settlement and persistence
- fee semantics and cost basis
- complementary YES/NO order-book execution
- multi-level depth impact and partial fills
- depth-vs-latency drag decomposition
- execution-event persistence and attribution
- grouped chronological model training and promotion
- orderbook snapshot/delta parsing
- sequence-gap invalidation
- empirical quote-horizon sampling
- p95 empirical latency recommendations
- empirical-latency execution override

## Research rules

1. **Paper first.**
2. Real-money order execution does not exist in V0.7.
3. Win rate is not the objective; expected value after costs is.
4. Never inject future settlement data into historical decisions.
5. Keep complete market windows isolated across model train/test boundaries.
6. Reject learned models that cannot beat baselines out-of-sample.
7. Measure forecast edge separately from execution drag.
8. Treat stale, missing, disagreeing, or sequence-broken data as risk conditions.
9. Do not switch to an empirical execution assumption until it has enough samples.
10. Paper/backtest profitability does not guarantee future profitability.

## Next milestone — V0.8 strategy evaluation / soak lab

- walk-forward strategy P&L replay using the recorded full decision frames
- regime-specific performance tables
- entry-minute / time-to-expiry performance
- edge-bucket performance
- separate model-error vs execution-error attribution
- maximum drawdown and consecutive-loss stress tests
- failure injection for stale feeds, sequence gaps, REST failures, and reconnects
- multi-day paper soak summary
- automated go/no-go research gates for future development

Live execution stays gated until the probability, execution, and reliability layers all survive substantially larger paper and out-of-sample samples.
