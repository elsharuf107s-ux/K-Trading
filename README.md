# K-Trading

K-Trading is a **paper-trading research engine** for Kalshi's rolling `KXBTC15M` Bitcoin markets.

The goal is to test one thing honestly: whether a short-horizon model can estimate a fair probability better than the live Kalshi market **after spread, fees/cost buffers, execution friction, and risk controls**.

> **V0.2 is paper-only. Real-money order execution is not implemented.**

## What V0.2 does

- Discovers the current `KXBTC15M` market automatically.
- Reads the live public Kalshi order book and derives YES/NO bid/ask, top depth, spread, and imbalance.
- Reads Coinbase + Kraken BTC spot as public cross-check/momentum feeds.
- Optionally connects to Kalshi's authenticated WebSocket and streams CF Benchmarks `BRTI`.
- Uses BRTI as the primary reference when available.
- Uses Kalshi's `last_60s_windowed_average_15min` field in the final minute when present.
- Builds short-term BTC momentum, Kalshi probability momentum, order-book imbalance, target distance, and a wave score.
- Produces a conservative model fair probability anchored to the market.
- Trades only when after-cost edge and wave direction agree.
- Exits on edge decay, wave reversal, stop loss, trailing-profit giveback, or time/reference safety gates.
- Settles stale paper positions once Kalshi reports the prior market result.
- Journals observations and trades to SQLite.
- Exposes a localhost/Codespaces dashboard and JSON API.

## Architecture

```text
KXBTC15M market + order book
            |
            +--------------------+
            |                    |
            v                    v
      Kalshi probability     CF Benchmarks BRTI
                                  |
Coinbase + Kraken ----------------+
            |
            v
       Feature engine
            |
     BTC momentum
     market momentum
     order-book imbalance
     target distance
     wave score
            |
            v
    Probability forecaster
            |
            v
 model fair value - market price
       - friction buffers
            |
            v
       hard risk engine
            |
     BUY / HOLD / EXIT
            |
            v
       paper broker
            |
            v
      SQLite research log
```

## Run in Codespaces / Linux

```bash
cp .env.example .env
pip install -r requirements.txt
./scripts/run.sh
```

Open forwarded port **8000**.

You can also run directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Enable the official BRTI stream

Kalshi's WebSocket requires API authentication even for public market-data channels.

Create a Kalshi API key, keep the **private key out of Git**, then set:

```env
KALSHI_KEY_ID=your-key-id
KALSHI_PRIVATE_KEY_PATH=/workspaces/K-Trading/kalshi-private-key.pem
ENABLE_KALSHI_WS=true
CFBENCHMARKS_INDEX_ID=BRTI
```

`KALSHI_PRIVATE_KEY_PEM` is also supported, but a secret/file outside the repository is preferable.

The dashboard will show:

- `LIVE` when the authenticated BRTI feed is healthy.
- `NO KEY` when no credentials are configured.
- proxy mode using Coinbase/Kraken if BRTI is unavailable.

When BRTI is unavailable, K-Trading **blocks new final-minute entries** by default.

## Main API routes

- `GET /api/health`
- `GET /api/state`
- `GET /api/trades`
- `GET /api/observations`
- `POST /api/cycle`
- `POST /api/kill-switch`

## Tests

```bash
pytest -q
```

Current V0.2 test coverage includes:

- bounded forecast output
- official BRTI reference preference
- final-minute settlement-window reference selection
- spread risk rejection
- final-minute proxy guard
- paper round-trip accounting
- paper market settlement
- CF Benchmarks WebSocket payload parsing

## Safety / research rules

1. **Paper first.**
2. No live-order executor exists in V0.2.
3. Do not judge the system by win rate alone.
4. Track net P&L after costs, calibration, Brier score, drawdown, edge capture, and performance by regime.
5. Never use later/final data in historical decisions.
6. A model that cannot beat simple baselines out-of-sample does not have proven edge.
7. Kalshi's actual market rules control settlement.

## Next milestones

### V0.3 — data collector + replay
- second-level observation capture
- historical market ingestion
- deterministic window replay
- calibration buckets
- Brier score / expected-value reporting
- regime performance report

### V0.4 — model lab
- logistic-regression baseline
- LightGBM/XGBoost candidate models
- walk-forward validation
- model registry / champion-vs-challenger
- no-lookahead test suite

### V0.5 — execution simulator
- order-book-depth fills
- partial fills
- latency model
- fee model from current Kalshi schedule
- slippage curves

Real execution remains gated until the research stack demonstrates durable out-of-sample behavior.
