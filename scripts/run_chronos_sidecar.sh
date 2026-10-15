#!/usr/bin/env bash
set -euo pipefail

HOST="${CHRONOS_HOST:-0.0.0.0}"
PORT="${CHRONOS_PORT:-8010}"

exec uvicorn services.chronos_sidecar:app --host "$HOST" --port "$PORT"
