# K-Trading V0.9 AI Sidecar

K-Trading keeps the low-latency trading path lightweight. The optional AI model runs as a separate service and only returns a probability vote. It never receives Kalshi credentials and cannot place orders.

## Recommended model

Use **Amazon Chronos-2** (`amazon/chronos-2`) for the outcome/time-series sidecar. Chronos-2 is built for multivariate/covariate time-series forecasting and is Apache-2.0 licensed. K-Trading converts its probabilistic forecast into an estimate of `P(YES)` for the 15-minute BTC contract by comparing the model's forecasted final-minute average with the Kalshi target.

For a smaller CPU-only experiment, set `CHRONOS_MODEL_ID=autogluon/chronos-2-small`. The full `amazon/chronos-2` model is the default quality-oriented choice.

## Why the sidecar does not control trades

The main prediction stack remains the fast calibrated K-Trading model because it can run inside the ~200ms decision budget. Chronos is slower, so its result is cached asynchronously and blended into the fast model only when it is fresh and confident.

Default safeguards:

- Chronos weight: 20%
- Minimum Chronos confidence: 55%
- Maximum probability shift caused by the sidecar: 8 percentage points
- Maximum sidecar age: 15 seconds
- Sidecar failure/timeout: ignored; K-Trading continues with its local model
- Hard risk and execution gates always run after the ensemble

## Run locally / Codespaces

Install the optional sidecar dependencies in a separate environment if possible:

```bash
python -m venv .venv-chronos
source .venv-chronos/bin/activate
pip install -r requirements-chronos.txt
```

Start the sidecar on port 8010:

```bash
CHRONOS_DEVICE=cpu CHRONOS_MODEL_ID=amazon/chronos-2 \
  bash scripts/run_chronos_sidecar.sh
```

The first start downloads the model weights from Hugging Face and can take a while. A GPU endpoint is preferable for lower inference latency, but K-Trading itself does not block while waiting for this service.

Check it:

```bash
curl http://127.0.0.1:8010/health
```

Then enable the client in K-Trading's `.env`:

```env
AI_SIDECAR_ENABLED=true
AI_SIDECAR_URL=http://127.0.0.1:8010/forecast
AI_SIDECAR_MODEL=amazon/chronos-2
AI_SIDECAR_WEIGHT=0.20
AI_SIDECAR_MIN_CONFIDENCE=0.55
AI_SIDECAR_REQUEST_INTERVAL_SECONDS=5
AI_SIDECAR_TIMEOUT_SECONDS=2
AI_SIDECAR_MAX_AGE_SECONDS=15
AI_SIDECAR_MAX_PROBABILITY_SHIFT=0.08
```

Restart K-Trading. The V0.9 dashboard panel will show the sidecar as `READY` and then `LIVE` after a valid forecast arrives.

## Fast path

The V0.9 fast path defaults to 200ms / 5Hz:

```env
FAST_PATH_ENABLED=true
FAST_PATH_INTERVAL_MS=200
FAST_PATH_REQUIRE_STREAM=true
```

This does **not** run REST discovery five times per second. Market discovery, Coinbase/Kraken refresh, settlement reconciliation, and training stay on the slower loop. The fast path only consumes the in-memory Kalshi WebSocket book, the latest BRTI value, and the most recent slow-loop BTC cross-check.

## Model roles

```text
Kalshi WS + BRTI
       |
       v
Fast local model / promoted champion  ----+
       |                                   |
       |                         Chronos-2 sidecar
       |                                   |
       +----------- bounded blend <---------+
                         |
                         v
                  Hard risk engine
                         |
                         v
               Execution simulator
                         |
                         v
                    Paper broker
```

Do not interpret a sidecar forecast or a passing research gate as proof of profitability. Keep live-money execution disabled until the strategy has enough out-of-sample evidence and execution/reconciliation behavior has been validated.
