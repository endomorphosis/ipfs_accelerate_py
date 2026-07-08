#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_gpt2_p2p_regression_loop.sh [options]

Runs the gated GPT-2 MCP+p2p regression pair repeatedly and prints a summary.

Options:
  --runs N               Number of loop iterations (default: 3)
  --timeout-sec S        Per-iteration timeout in seconds (default: 0, disabled)
  --run-root DIR         Directory for logs/artifacts (default: scripts/ops/.runs/gpt2_p2p_<timestamp>)
  --pytest-args "ARGS"   Extra args appended to pytest command (default: "")
  --help                 Show this help

Environment:
  IPFS_ACCELERATE_PY_RUN_GPT2_E2E=1
  IPFS_ACCELERATE_PY_RUN_GPT2_P2P_REGRESSION=1

Examples:
  scripts/ops/run_gpt2_p2p_regression_loop.sh --runs 3
  scripts/ops/run_gpt2_p2p_regression_loop.sh --runs 2 --timeout-sec 1200 --pytest-args "-s"
EOF
}

RUNS=3
TIMEOUT_SEC=0
RUN_ROOT=""
EXTRA_PYTEST_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs)
      RUNS="${2:-}"; shift 2 ;;
    --timeout-sec)
      TIMEOUT_SEC="${2:-}"; shift 2 ;;
    --run-root)
      RUN_ROOT="${2:-}"; shift 2 ;;
    --pytest-args)
      EXTRA_PYTEST_ARGS="${2:-}"; shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || (( RUNS <= 0 )); then
  echo "--runs must be a positive integer." >&2
  exit 2
fi

if ! [[ "$TIMEOUT_SEC" =~ ^[0-9]+$ ]]; then
  echo "--timeout-sec must be a non-negative integer." >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="${ROOT_DIR}/scripts/ops/.runs/gpt2_p2p_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$RUN_ROOT"
RUN_LOG="$RUN_ROOT/run.log"
SUMMARY_JSON="$RUN_ROOT/summary.json"

PYTEST_BIN="${ROOT_DIR}/.venv/bin/pytest"
if [[ ! -x "$PYTEST_BIN" ]]; then
  echo "pytest binary not found at: $PYTEST_BIN" >&2
  exit 127
fi

TEST_ARGS=(
  -q
  ipfs_accelerate_py/mcp/tests/test_p2p_call_tool_bridge.py::test_p2p_call_tool_runs_gpt2_inference_over_libp2p
  test/api/test_task_p2p_textgen_two_peers_regression.py::test_task_p2p_two_peers_textgen_regression_50
  -vv
)

EXTRA_ARGS=()
if [[ -n "${EXTRA_PYTEST_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${EXTRA_PYTEST_ARGS} )
fi

PASS_COUNT=0
FAIL_COUNT=0
FAILED_RUNS=()

export IPFS_ACCELERATE_PY_RUN_GPT2_E2E="${IPFS_ACCELERATE_PY_RUN_GPT2_E2E:-1}"
export IPFS_ACCELERATE_PY_RUN_GPT2_P2P_REGRESSION="${IPFS_ACCELERATE_PY_RUN_GPT2_P2P_REGRESSION:-1}"

echo "RUN_ROOT=${RUN_ROOT}"
echo "RUNS=${RUNS} TIMEOUT_SEC=${TIMEOUT_SEC}" >"$RUN_LOG"

for i in $(seq 1 "$RUNS"); do
  echo "=== GPT2_P2P_RUN ${i}/${RUNS} started_at=$(date -Iseconds) ===" | tee -a "$RUN_LOG"
  ITER_LOG="$RUN_ROOT/run_${i}.log"

  run_cmd=("$PYTEST_BIN" "${TEST_ARGS[@]}" "${EXTRA_ARGS[@]}")
  run_rc=0

  if (( TIMEOUT_SEC > 0 )); then
    if command -v timeout >/dev/null 2>&1; then
      if timeout "${TIMEOUT_SEC}s" "${run_cmd[@]}" 2>&1 | tee "$ITER_LOG"; then
        run_rc=0
      else
        run_rc=$?
      fi
    else
      echo "WARN: timeout command not found; running without timeout" >&2
      if "${run_cmd[@]}" 2>&1 | tee "$ITER_LOG"; then
        run_rc=0
      else
        run_rc=$?
      fi
    fi
  else
    if "${run_cmd[@]}" 2>&1 | tee "$ITER_LOG"; then
      run_rc=0
    else
      run_rc=$?
    fi
  fi

  cat "$ITER_LOG" >>"$RUN_LOG"
  if (( run_rc == 0 )); then
    PASS_COUNT=$((PASS_COUNT + 1))
    echo "=== GPT2_P2P_RUN ${i}/${RUNS} exit=0 ended_at=$(date -Iseconds) ===" | tee -a "$RUN_LOG"
  else
    echo "=== GPT2_P2P_RUN ${i}/${RUNS} exit=${run_rc} ended_at=$(date -Iseconds) ===" | tee -a "$RUN_LOG"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    FAILED_RUNS+=("$i")
  fi
done

FAILED_RUNS_TEXT="${FAILED_RUNS[*]:-}"

cat >"$SUMMARY_JSON" <<EOF
{
  "runs": ${RUNS},
  "passed": ${PASS_COUNT},
  "failed": ${FAIL_COUNT},
  "failed_runs": "${FAILED_RUNS_TEXT}",
  "run_root": "${RUN_ROOT}",
  "run_log": "${RUN_LOG}"
}
EOF

echo "SUMMARY runs=${RUNS} passed=${PASS_COUNT} failed=${FAIL_COUNT} run_root=${RUN_ROOT}"
if (( FAIL_COUNT > 0 )); then
  echo "FAILED_RUNS=${FAILED_RUNS[*]}" >&2
  echo "SUMMARY_JSON=${SUMMARY_JSON}" >&2
  exit 1
fi

echo "SUMMARY_JSON=${SUMMARY_JSON}"
exit 0
