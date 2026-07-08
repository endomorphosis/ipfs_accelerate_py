#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${REPO_ROOT}"

REPORT_PATH=""
if [[ -f "implementation_plan/docs/31-ipfs-accelerate-model-serving-readiness-report.json" ]]; then
  REPORT_PATH="implementation_plan/docs/31-ipfs-accelerate-model-serving-readiness-report.json"
elif [[ -f "../implementation_plan/docs/31-ipfs-accelerate-model-serving-readiness-report.json" ]]; then
  REPORT_PATH="../implementation_plan/docs/31-ipfs-accelerate-model-serving-readiness-report.json"
fi

if [[ -n "${REPORT_PATH}" ]]; then
  echo "==> Validating readiness report JSON (${REPORT_PATH})"
  "${PYTHON_BIN}" -m json.tool "${REPORT_PATH}" >/dev/null
else
  echo "==> Readiness report JSON not found in repository layout; skipping JSON validation"
fi

echo "==> Running serving contract gates"
"${PYTHON_BIN}" -m pytest \
  test/api/test_serving_call_matrix_enforcement.py \
  test/api/test_task_worker_backend_manager_required.py \
  test/api/test_serving_readiness_contracts.py \
  test/api/test_serving_e2e_call_chain_ordering.py

echo "==> Serving contract gates passed"
