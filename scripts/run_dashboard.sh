#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-8080}
LOG=${LOG:-dashboard.out}

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "Starting Performance Dashboard on 0.0.0.0:${PORT}..."
python - <<PY >"${LOG}" 2>&1 &
from utils.performance_dashboard import start_performance_dashboard
start_performance_dashboard(port=int("${PORT}"), background=False)
PY
echo $! > dashboard.pid
sleep 1
echo "Logs: ${LOG} | PID: $(cat dashboard.pid)"
