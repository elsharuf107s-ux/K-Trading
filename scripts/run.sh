#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
exec uvicorn app.main:app --host "${APP_HOST:-0.0.0.0}" --port "${APP_PORT:-8000}" --reload
