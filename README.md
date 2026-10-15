# K-Trading — Kalshi BTC 15-Minute AI Lab

K-Trading is a paper-first research and execution platform for Kalshi's short-duration Bitcoin markets. V1 ingests the live market, estimates fair YES probability from market + BTC micro-momentum, trades only when after-cost edge clears hard risk rules, and journals every observation so calibrated models can replace the baseline later.

> **Status: V1 foundation / paper trading only.** `LIVE_TRADING_ENABLED=true` is intentionally blocked by the risk engine in this version.

## Working now

- Discovers an open short-duration BTC market through Kalshi's public V2 market API.
- Reads YES/NO prices and contract timing.
- Pulls BTC spot proxy prices from Coinbase + Kraken and uses the median.
- Tracks 30s/60s/180s momentum and realized volatility.
- Produces a conservative fair-probability estimate anchored to Kalshi's implied probability.
- Calculates YES and NO edge after uncertainty + cost buffers.
- Enforces spread, expiry, bankroll, max-position, loss, drawdown, stop-loss and kill-switch limits.
- Paper opens/closes positions and writes observations/trades to SQLite.
- Serves a browser trading terminal on port `8000`.
- Includes a GitHub Codespaces devcontainer.

## Run in Codespaces

```bash
cp .env.example .env
./scripts/run.sh
```

Open forwarded port **8000**.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
./scripts/run.sh
```

Or:

```bash
cp .env.example .env
docker compose up --build
```

## Architecture

```text
Kalshi public markets ──┐
                       ├──> Feature Engine ──> Probability Model
BTC proxy feeds ───────┘                           │
                                                   v
                                    Fair probability vs market
                                                   │
                                                   v
                                      After-cost usable edge
                                                   │
                                                   v
                                          HARD RISK ENGINE
                                                   │
                                                   v
                                             PAPER BROKER
                                                   │
                                SQLite journal + dashboard
```

## Settlement/reference-data warning

The Coinbase/Kraken spot feeds are **features only**, not settlement truth. Kalshi states that most crypto price markets use CF Benchmarks indexes and that a 15-minute market compares the average from the minute before close with the average from the minute before open. Exact market rules and contract terms control. Live trading must stay locked until authenticated CF Benchmarks/reference data and historical point-in-time validation are added.

## API

- `GET /api/health`
- `GET /api/state`
- `GET /api/trades`
- `POST /api/cycle`
- `POST /api/kill-switch` with `{ "enabled": true }`

## Strategy settings

See `.env.example`. Important defaults include:

```env
AUTO_PAPER_TRADE=true
PAPER_STARTING_CASH=1000
MIN_USABLE_EDGE=0.04
MAX_POSITION_USD=15
MAX_BANKROLL_FRACTION=0.015
MAX_SPREAD=0.12
MIN_SECONDS_TO_EXPIRY=45
LIVE_TRADING_ENABLED=false
```

If discovery misses the live contract:

```env
MARKET_TICKER_OVERRIDE=<kalshi-market-ticker>
```

## Model philosophy

The current `BaselineForecaster` is deliberately conservative and is **not a claim of profitable AI**. The first objective is clean point-in-time data collection. Once enough completed windows exist, replace it behind the same interface with calibrated logistic regression / LightGBM, walk-forward validation, probability calibration and regime-specific ensembles.

## Next milestones

1. **Data integrity:** Kalshi WebSocket book/trade stream, exact series discovery, event live data, CF Benchmarks pass-through, second-level persistence.
2. **Backtesting:** replay completed windows without look-ahead; include bid/ask fills, fees and latency; score calibration/Brier/P&L.
3. **Learning:** dataset builder, walk-forward splits, logistic baseline, LightGBM candidate, calibration and model registry.
4. **Wave management:** separate entry/exit models, trailing profit, regime detection and order-book imbalance.
5. **Live gate:** only after paper/out-of-sample thresholds—authenticated V2 orders, idempotency, reconciliation, freshness shutdowns and tiny capped position sizing.

## Tests

```bash
pytest -q
```

The initial core currently passes three local unit tests covering bounded forecasts, hard spread rejection and paper round-trip accounting.
