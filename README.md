# K-Trading

K-Trading is a **paper-only research and learning engine** for Kalshi's rolling `KXBTC15M` Bitcoin markets.

The goal is not to maximize win rate. The goal is to determine whether a probability model can beat the live Kalshi probability **out-of-sample and after spread, cost buffers, execution friction, and hard risk controls**.

> **Status: V0.4 / paper trading only. Real-money order execution is not implemented.**

## What V0.4 does

### Live research engine
- Discovers the current `KXBTC15M` market.
- Reads the public Kalshi order book and derives YES/NO bid/ask, depth, spread, and imbalance.
- Reads Coinbase + Kraken as spot cross-checks.
- Optionally authenticates to Kalshi WebSocket data and streams CF Benchmarks `BRTI`.
- Uses BRTI as the primary reference and momentum source when available.
- Uses the 15-minute settlement-window average during the final minute when supplied by the feed.
- Builds BTC/BRTI momentum, Kalshi probability momentum, order-book imbalance, target distance, and a wave score.
- Produces a fair YES probability and compares it with Kalshi after friction buffers.
- Enforces hard spread, expiry, bankroll, drawdown, stop, trailing-profit, wave-alignment, and kill-switch rules.
- Persists paper cash, positions, realized P&L, and kill-switch state through restarts.
- Reconciles completed markets and settles stale paper positions from Kalshi's result.

### Research / replay layer
- Stores point-in-time observations in SQLite.
- Labels completed windows with final YES/NO results.
- Computes model-vs-market Brier score and calibration buckets.
- Replays recorded 15-minute windows frame-by-frame without future-data injection.

### V0.4 Model Lab
- Builds training data only from **completed** markets.
- Keeps all snapshots from one market window on the same side of a train/test split.
- Uses chronological expanding walk-forward validation rather than random row splitting.
- Gives each market equal total sample weight so windows with more captured snapshots do not dominate training.
- Trains a standardized logistic-regression baseline.
- Trains a histogram gradient-boosting challenger.
- Scores both on Brier score and log loss against the live Kalshi market probability baseline.
- Uses a disjoint recent holdout for sigmoid probability calibration.
- Saves candidates locally under `data/models/`.
- Promotes a model to `champion` only when its out-of-sample Brier score clears the configured market-improvement threshold and beats the existing champion.
- Automatically uses the promoted champion in the paper engine; otherwise it falls back to the deterministic wave-aware baseline.

## Learning architecture

```text
COMPLETED KXBTC15M WINDOWS
           |
           v
 Point-in-time snapshots
           |
           v
 Group by market ticker
           |
           v
 chronological walk-forward
           |
      +----+----+
      |         |
      v         v
 Logistic    HistGradient
 baseline     challenger
      |         |
      +----+----+
           |
           v
 OOS Brier / log-loss
           |
     compare to Kalshi
           |
           v
 recent calibration holdout
           |
           v
   candidate registry
           |
   promotion rules pass?
       /          \
     no            yes
     |              |
 candidate only   CHAMPION
                    |
                    v
             paper forecaster
```

## Why the split is by market, not by row

A single 15-minute contract generates many snapshots. Randomly splitting individual rows would leak information because early snapshots from a market could land in training while later snapshots from that exact same market land in testing.

V0.4 orders completed market tickers by time and moves **whole windows** between training and testing. This is intentionally stricter than ordinary random cross-validation.

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

Kalshi WebSocket market/reference data requires API authentication. Keep the private key outside Git.

```env
KALSHI_KEY_ID=your-key-id
KALSHI_PRIVATE_KEY_PATH=/workspaces/K-Trading/kalshi-private-key.pem
ENABLE_KALSHI_WS=true
CFBENCHMARKS_INDEX_ID=BRTI
```

`KALSHI_PRIVATE_KEY_PEM` is supported too, but an external secret/file is preferable.

Without a fresh BRTI feed, K-Trading falls back to Coinbase/Kraken for research features and **blocks new final-minute entries**.

## Model Lab settings

Defaults:

```env
MODEL_DIR=data/models
ENABLE_LEARNED_MODEL=true
MODEL_MIN_WINDOWS=24
MODEL_MIN_BRIER_GAIN=0.002
```

`MODEL_MIN_WINDOWS` is intentionally only a development floor. Twenty-four 15-minute windows is nowhere near enough evidence to conclude that a strategy has durable real-world edge. The purpose of the low initial gate is to exercise the research pipeline while the database grows.

Model artifacts are ignored by Git and stay local to the running Codespace/machine unless you intentionally export them.

## Train the model lab

Once enough completed markets have been recorded:

```bash
curl -X POST http://localhost:8000/api/model-lab/train
```

The response reports each candidate's walk-forward metrics, which candidate won, whether it beat Kalshi, and whether it was promoted.

A failed/rejected candidate does **not** replace the current champion.

## Main API routes

- `GET /api/health`
- `GET /api/state`
- `GET /api/trades`
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

- bounded forecast output
- official BRTI preference and BRTI momentum
- settlement-window reference selection
- spread and final-minute reference guards
- paper accounting, settlement, and restart persistence
- CF Benchmarks WebSocket parsing
- completed-window labeling and replay
- Brier/calibration analytics
- refusal to train on too few completed markets
- synthetic grouped walk-forward training
- champion promotion only after out-of-sample Brier improvement

## Safety / research rules

1. **Paper first.**
2. Real-money order execution does not exist in V0.4.
3. Never optimize primarily for win rate.
4. Track net P&L after costs, Brier score, calibration, drawdown, edge capture, and regime performance.
5. Never use future or settlement data in a historical decision frame.
6. Keep complete market windows isolated across training/testing boundaries.
7. A candidate that cannot beat the live-market probability baseline out-of-sample does not get promoted.
8. A backtest or paper result is not proof of future profitability.
9. Kalshi's actual contract rules control settlement.

## Next milestone — V0.5 execution simulator

- order-book-depth-aware fills
- partial fills
- configurable API/network latency
- current Kalshi fee model
- slippage curves
- maker vs taker simulation where applicable
- execution-quality attribution
- strategy P&L separated from fill/slippage P&L

Real execution remains gated until the research stack demonstrates durable out-of-sample behavior across a much larger number of completed markets.
