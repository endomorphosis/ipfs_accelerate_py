#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$ROOT_DIR/state/smoketest_logs"

# Deterministic local-only networking
export IPFS_ACCEL_SKIP_CORE=1
export PYTHONUNBUFFERED=1
export IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP=127.0.0.1
export IPFS_ACCELERATE_PY_TASK_P2P_SESSION=smoketest
export IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS=0
export IPFS_ACCELERATE_PY_TASK_P2P_DHT=0
export IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS=0
export IPFS_ACCELERATE_PY_TASK_P2P_MDNS=0

# Use a dedicated queue file so the test is reproducible.
export IPFS_ACCELERATE_PY_TASK_QUEUE_PATH="$LOG_DIR/queue.duckdb"
QUEUE_PATH="$IPFS_ACCELERATE_PY_TASK_QUEUE_PATH"

# Use a dedicated announce file so the driver dials exactly this service.
export IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE="$LOG_DIR/announce.json"
ANNOUNCE_FILE="$IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"

# Avoid needing a real Copilot installation: make provider 'copilot_cli' just echo a stable string.
# (The routing still goes through llm_router, and workers still enforce the copilot_cli gating.)
export ipfs_accelerate_py_COPILOT_CLI_CMD='bash -lc "echo OK"'

# Required for copilot_cli tasks in worker.
export IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_COPILOT_CLI=1

# Choose a free port via Python so we don't depend on a fixed one.
PORT="$($ROOT_DIR/.venv/bin/python - <<'PY'
import socket
s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()
PY
)"
export IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_HOST=127.0.0.1
export IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT="$PORT"

mkdir -p "$LOG_DIR"
rm -f "$ANNOUNCE_FILE" "$LOG_DIR"/*.log

echo "=== starting controller (p2p service + 2 worker threads) on port $PORT ==="
"$ROOT_DIR/.venv/bin/python" -c "
import os, threading
from ipfs_accelerate_py.p2p_tasks.worker import run_autoscaled_workers

stop = threading.Event()

run_autoscaled_workers(
    queue_path=os.environ['IPFS_ACCELERATE_PY_TASK_QUEUE_PATH'],
    base_worker_id='pool',
    min_workers=2,
    max_workers=2,
    poll_interval_s=0.25,
    p2p_service=True,
    p2p_listen_port=int(os.environ['IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT']),
    supported_task_types=['llm.generate'],
    stop_event=stop,
)
" >"$LOG_DIR/controller.log" 2>&1 &
CONTROLLER_PID=$!

cleanup() {
  set +e
  if kill -0 "$CONTROLLER_PID" 2>/dev/null; then kill "$CONTROLLER_PID"; fi
  wait "$CONTROLLER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for announce file
for i in {1..200}; do
  if [[ -s "$ANNOUNCE_FILE" ]]; then break; fi
  sleep 0.05
done

if [[ ! -s "$ANNOUNCE_FILE" ]]; then
  echo "ERROR: announce file not created: $ANNOUNCE_FILE"
  echo "--- controller.log tail ---"
  tail -n 200 "$LOG_DIR/controller.log" || true
  exit 3
fi

echo "=== announce ==="
cat "$ANNOUNCE_FILE"

# Give controller a moment to start polling
sleep 0.25

echo "=== running driver ==="
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/dev_tools/mesh_pooling_smoketest.py" \
  --announce "$ANNOUNCE_FILE" \
  --jobs 12 --concurrency 6 --timeout-s 60 \
  --session-id "${IPFS_ACCELERATE_PY_TASK_P2P_SESSION}" \
  --transcript-jsonl "$LOG_DIR/transcript.jsonl" \
  >"$LOG_DIR/driver.out" 2>&1 || {
    code=$?
    echo "Driver exit code: $code"
    echo "--- driver.out ---"; cat "$LOG_DIR/driver.out" || true
    echo "--- controller.log tail ---"; tail -n 200 "$LOG_DIR/controller.log" || true
    exit "$code"
  }

echo "=== driver output ==="
cat "$LOG_DIR/driver.out"

echo "=== log locations ==="
echo "  $LOG_DIR/controller.log"
echo "  $LOG_DIR/driver.out"
echo "  $LOG_DIR/transcript.jsonl"
