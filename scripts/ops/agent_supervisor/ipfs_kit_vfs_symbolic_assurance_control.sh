#!/usr/bin/env bash
#
# Operate the isolated Grok/Codex supervisor lanes for the IPFS Kit VFS
# symbolic-assurance program. Runtime state is intentionally kept outside the
# repository so a clean integration checkout remains the only merge authority.

set -Eeuo pipefail

readonly CONTROL_COMMAND="${1:-status}"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
readonly PLAN_PATH="docs/architecture/IPFS_KIT_VFS_SYMBOLIC_ASSURANCE_PLAN.md"
readonly OBJECTIVE_PATH="docs/architecture/ipfs_kit_vfs_symbolic_assurance.objectives.md"
readonly TODO_PATH="docs/architecture/ipfs_kit_vfs_symbolic_assurance.todo.md"
readonly VALIDATOR_PATH="scripts/ops/agent_supervisor/validate_ipfs_kit_vfs_symbolic_assurance.py"
readonly CONTROL_PATH="scripts/ops/agent_supervisor/ipfs_kit_vfs_symbolic_assurance_control.sh"
readonly TARGET_BRANCH="${IPFS_KIT_VFS_ASSURANCE_BRANCH:-agent/swissknife-contract-audit}"

if [[ -z "${HOME:-}" ]]; then
  echo "HOME must be set so the external supervisor state root can be resolved" >&2
  exit 2
fi

readonly DEFAULT_STATE_BASE="${XDG_STATE_HOME:-${HOME}/.local/state}"
readonly STATE_ROOT="${IPFS_KIT_VFS_ASSURANCE_STATE_ROOT:-${DEFAULT_STATE_BASE}/ipfs_accelerate_py/ipfs_kit_vfs_symbolic_assurance}"
readonly STATE_DIR="${STATE_ROOT}/state"
readonly RUNTIME_DIR="${STATE_ROOT}/runtime"
readonly LOG_DIR="${STATE_ROOT}/logs"
readonly PROJECTION_DIR="${STATE_ROOT}/projection"
readonly WORKTREE_DIR="${STATE_ROOT}/worktrees"
readonly MERGE_QUEUE_DIR="${STATE_ROOT}/merge-queue"
readonly OBJECTIVE_ABS="${REPO_ROOT}/${OBJECTIVE_PATH}"
readonly TODO_ABS="${REPO_ROOT}/${TODO_PATH}"
readonly VALIDATOR_ABS="${REPO_ROOT}/${VALIDATOR_PATH}"

choose_python() {
  local candidate
  if [[ -n "${IPFS_ACCELERATE_AGENT_PYTHON:-}" ]]; then
    candidate="${IPFS_ACCELERATE_AGENT_PYTHON}"
    if [[ -x "${candidate}" ]] && \
      PYTHONDONTWRITEBYTECODE=1 "${candidate}" -c 'import duckdb' \
        >/dev/null 2>&1
    then
      printf '%s\n' "${candidate}"
      return 0
    fi
    echo "Configured IPFS_ACCELERATE_AGENT_PYTHON cannot import DuckDB: ${candidate}" >&2
    return 2
  fi
  for candidate in \
    "${REPO_ROOT}/.venv/bin/python" \
    "${REPO_ROOT}/../.venv/bin/python" \
    "${REPO_ROOT}/../../.venv/bin/python"
  do
    if [[ -x "${candidate}" ]] && \
      PYTHONDONTWRITEBYTECODE=1 "${candidate}" -c 'import duckdb' \
        >/dev/null 2>&1
    then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  candidate="$(command -v python3)"
  if [[ -x "${candidate}" ]] && \
    PYTHONDONTWRITEBYTECODE=1 "${candidate}" -c 'import duckdb' \
      >/dev/null 2>&1
  then
    printf '%s\n' "${candidate}"
    return 0
  fi
  echo "No Python interpreter with a working DuckDB extension was found" >&2
  return 2
}

readonly PYTHON_BIN="$(choose_python)"
readonly PYTHONPATH_VALUE="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
readonly -a RUNTIME_ENV=(
  "PYTHONDONTWRITEBYTECODE=1"
  "PYTHONHASHSEED=0"
  "PYTHONPATH=${PYTHONPATH_VALUE}"
  "IPFS_ACCELERATE_DUCKDB_ONLY=1"
  "IPFS_ACCEL_SKIP_CORE=1"
  "IPFS_KIT_DISABLE=1"
  "IPFS_DATASETS_AUTO_INSTALL=false"
  "IPFS_AUTO_INSTALL=false"
  "IPFS_DATASETS_PY_MINIMAL_IMPORTS=1"
)
readonly -a PROTECTED_ARGS=(
  "--implementation-protected-path" "${PLAN_PATH}"
  "--implementation-protected-path" "${OBJECTIVE_PATH}"
  "--implementation-protected-path" "${TODO_PATH}"
  "--implementation-protected-path" "${VALIDATOR_PATH}"
)
readonly -a COMMON_ARGS=(
  "--todo-path" "${TODO_ABS}"
  "--state-dir" "${STATE_DIR}"
  "--task-prefix" "## VFS-"
  "--implement"
  "--max-task-attempts" "3"
  "--implementation-retry-budget" "3"
  "--validation-retry-budget" "3"
  "--merge-retry-budget" "3"
  "--implementation-timeout" "3600"
  "--implementation-max-timeout" "7200"
  "--implementation-log-stall-seconds" "1200"
  "--daemon-interval" "60"
  "--check-interval" "30"
  "--stale-seconds" "1800"
  "--watchdog-startup-grace-seconds" "300"
  "--task-shard-count" "2"
  "--worktree-root" "${WORKTREE_DIR}"
  "--worktree-submodule-path" "ipfs_accelerate_py/mcplusplus"
  "--worktree-submodule-path" "ipfs_datasets_py"
  "--worktree-submodule-path" "ipfs_kit_py"
  "--merge-target-branch" "${TARGET_BRANCH}"
  "--merge-queue-dir" "${MERGE_QUEUE_DIR}"
  "${PROTECTED_ARGS[@]}"
)

prepare_state_dirs() {
  umask 077
  mkdir -p \
    "${STATE_DIR}" \
    "${RUNTIME_DIR}" \
    "${LOG_DIR}" \
    "${PROJECTION_DIR}/discovery" \
    "${PROJECTION_DIR}/bundles" \
    "${PROJECTION_DIR}/datasets" \
    "${WORKTREE_DIR}" \
    "${MERGE_QUEUE_DIR}"
}

require_isolated_clean_checkout() {
  local branch
  local dirty
  branch="$(git -C "${REPO_ROOT}" branch --show-current)"
  if [[ "${branch}" != "${TARGET_BRANCH}" ]]; then
    echo "Refusing to launch from branch '${branch}'; expected '${TARGET_BRANCH}'" >&2
    return 2
  fi
  dirty="$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)"
  if [[ -n "${dirty}" ]]; then
    echo "Refusing to launch from a dirty integration checkout:" >&2
    printf '%s\n' "${dirty}" >&2
    return 2
  fi
}

pid_file_for_lane() {
  printf '%s/%s_supervisor.pid\n' "${RUNTIME_DIR}" "$1"
}

status_file_for_lane() {
  printf '%s/%s_supervisor_status.json\n' "${STATE_DIR}" "$1"
}

lane_pid() {
  local lane="$1"
  local path
  path="$(pid_file_for_lane "${lane}")"
  if [[ -f "${path}" ]]; then
    tr -d '[:space:]' < "${path}"
  fi
}

lane_process_is_owned() {
  local lane="$1"
  local pid="$2"
  local command_line
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
  [[ "${command_line}" == *"implementation_supervisor"* ]] || return 1
  [[ "${command_line}" == *"--state-prefix ${lane}"* ]] || return 1
  [[ "${command_line}" == *"${TODO_ABS}"* ]]
}

provider_preflight() {
  env \
    "${RUNTIME_ENV[@]}" \
    "IPFS_ACCELERATE_AGENT_GROK_BIN=${IPFS_ACCELERATE_AGENT_GROK_BIN:-${HOME}/.local/bin/grok}" \
    "${PYTHON_BIN}" - <<'PY'
import json
import shutil
import sys
import duckdb
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    _grok_binary,
    _grok_cli_available,
)

result = {
    "codex": shutil.which("codex") or "",
    "duckdb": duckdb.__version__,
    "grok": _grok_binary() or "",
    "grok_authenticated": bool(_grok_cli_available()),
    "python": sys.executable,
}
print(json.dumps(result, indent=2, sort_keys=True))
if not result["codex"] or not result["grok"] or not result["grok_authenticated"]:
    raise SystemExit(2)
PY
}

project_objectives() {
  env "${RUNTIME_ENV[@]}" "${PYTHON_BIN}" "${VALIDATOR_ABS}" \
    > "${PROJECTION_DIR}/native_board_preflight.json"
  env "${RUNTIME_ENV[@]}" "${PYTHON_BIN}" \
    -m ipfs_accelerate_py.agent_supervisor.objective_daemon \
    --repo-root "${REPO_ROOT}" \
    --objective-path "${OBJECTIVE_ABS}" \
    --todo-path "${TODO_ABS}" \
    --discovery-dir "${PROJECTION_DIR}/discovery" \
    --bundle-dir "${PROJECTION_DIR}/bundles" \
    --dataset-dir "${PROJECTION_DIR}/datasets" \
    --graph-path "${PROJECTION_DIR}/objective_graph.json" \
    --todo-vector-index-path "${PROJECTION_DIR}/todo_vector_index.json" \
    --analysis-escalation-path "${PROJECTION_DIR}/analysis_escalation.json" \
    --plan-evaluation-path "${PROJECTION_DIR}/plan_evaluations.json" \
    --objective-generation-path "${PROJECTION_DIR}/objective_generation.json" \
    --task-prefix "VFS-" \
    --max-findings "0" \
    --no-generate-bounded-work \
    --no-reconcile-goal-completion \
    > "${PROJECTION_DIR}/objective_daemon_receipt.json"
}

lane_args() {
  local lane="$1"
  local shard="$2"
  printf '%s\0' \
    "${COMMON_ARGS[@]}" \
    "--state-prefix" "${lane}" \
    "--task-shard-index" "${shard}"
}

reconciliation_preflight() {
  local lane="$1"
  local shard="$2"
  local -a args=()
  while IFS= read -r -d '' item; do
    args+=("${item}")
  done < <(lane_args "${lane}" "${shard}")
  (
    cd "${REPO_ROOT}"
    env "${RUNTIME_ENV[@]}" "${PYTHON_BIN}" \
      -m ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor \
      "${args[@]}" \
      --once \
      --reconciliation-only \
      --log-level INFO
  ) > "${LOG_DIR}/${lane}_preflight.log" 2>&1
}

launch_lane() {
  local lane="$1"
  local shard="$2"
  local provider="$3"
  local pid_path
  local existing_pid
  local -a args=()
  local -a provider_env=()
  pid_path="$(pid_file_for_lane "${lane}")"
  existing_pid="$(lane_pid "${lane}")"
  if [[ -n "${existing_pid}" ]] && lane_process_is_owned "${lane}" "${existing_pid}"; then
    echo "${lane} supervisor is already running as PID ${existing_pid}"
    return 0
  fi
  while IFS= read -r -d '' item; do
    args+=("${item}")
  done < <(lane_args "${lane}" "${shard}")

  if [[ "${provider}" == "grok-build" ]]; then
    provider_env=(
      "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER=grok-build"
      "IPFS_ACCELERATE_AGENT_GROK_BIN=${IPFS_ACCELERATE_AGENT_GROK_BIN:-${HOME}/.local/bin/grok}"
      "IPFS_ACCELERATE_AGENT_GROK_MODEL=${IPFS_ACCELERATE_AGENT_GROK_MODEL:-grok-4.5}"
    )
    args+=(
      "--objective-refill-scan"
      "--objective-path" "${OBJECTIVE_ABS}"
      "--objective-graph-path" "${PROJECTION_DIR}/objective_graph.json"
      "--objective-bundle-dir" "${PROJECTION_DIR}/bundles"
      "--objective-dataset-dir" "${PROJECTION_DIR}/datasets"
      "--objective-discovery-dir" "${PROJECTION_DIR}/discovery"
      "--objective-discovery-output-path" "data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance/objective_gap.json"
      "--objective-todo-vector-index-path" "${PROJECTION_DIR}/todo_vector_index.json"
      "--objective-scan-min-open-tasks" "4"
      "--objective-scan-max-findings" "8"
      "--objective-scan-cooldown-seconds" "3600"
      "--objective-refill-timeout-seconds" "900"
      "--objective-max-refinement-children" "3"
      "--objective-max-refinement-depth" "4"
      "--objective-surplus-findings-per-goal" "2"
      "--objective-mission-term" "virtual filesystem"
      "--objective-mission-term" "contract"
      "--objective-mission-term" "MCP++"
      "--no-objective-goal-completion-reconcile"
      "--no-objective-goal-migration"
      "--codebase-refill-scan"
      "--codebase-scan-discovery-dir" "${PROJECTION_DIR}/discovery"
      "--codebase-scan-discovery-output-path" "data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance/codebase_finding.json"
      "--codebase-scan-min-open-tasks" "2"
      "--codebase-scan-max-findings" "5"
      "--codebase-scan-cooldown-seconds" "21600"
      "--codebase-refill-timeout-seconds" "600"
      "--codebase-scan-skip-prefix" ".git"
      "--codebase-scan-skip-prefix" ".worktrees"
    )
  else
    provider_env=(
      "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER=codex"
    )
    args+=(
      "--no-retry-budget-guardrail"
      "--no-dependency-guardrail"
      "--no-reconciliation-guardrail"
      "--no-objective-task-janitor"
      "--no-objective-goal-completion-reconcile"
      "--no-objective-goal-migration"
    )
  fi

  (
    cd "${REPO_ROOT}"
    nohup setsid env \
      "${RUNTIME_ENV[@]}" \
      "${provider_env[@]}" \
      "${PYTHON_BIN}" \
      -m ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor \
      "${args[@]}" \
      --log-level INFO \
      > "${LOG_DIR}/${lane}_supervisor.log" 2>&1 \
      < /dev/null \
      9>&- &
    printf '%s\n' "$!" > "${pid_path}"
  )
}

verify_lane_started() {
  local lane="$1"
  local pid
  local status_path
  local daemon_pid
  local attempt
  local healthy_observations=0
  local status
  pid="$(lane_pid "${lane}")"
  status_path="$(status_file_for_lane "${lane}")"
  for attempt in $(seq 1 55); do
    if ! lane_process_is_owned "${lane}" "${pid}"; then
      echo "${lane} supervisor PID ${pid:-missing} exited during startup" >&2
      tail -n 80 "${LOG_DIR}/${lane}_supervisor.log" >&2 || true
      return 1
    fi
    if [[ -s "${status_path}" ]]; then
      read -r status daemon_pid < <(
        "${PYTHON_BIN}" - "${status_path}" <<'PY'
import json
import sys
from pathlib import Path

try:
    value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, ValueError):
    value = {}
print(value.get("status") or "", value.get("daemon_pid") or "")
PY
      )
      if [[ "${status}" == "running" ]] && \
        [[ "${daemon_pid}" =~ ^[1-9][0-9]*$ ]] && \
        kill -0 "${daemon_pid}" 2>/dev/null && \
        [[ -s "${STATE_DIR}/${lane}_task_state.json" ]]
      then
        healthy_observations=$((healthy_observations + 1))
        if (( healthy_observations >= 3 )); then
          echo "${lane} supervisor PID ${pid}, managed daemon PID ${daemon_pid}"
          return 0
        fi
      else
        healthy_observations=0
      fi
    fi
    sleep 1
  done
  echo "${lane} supervisor did not publish a live managed daemon within 55 seconds" >&2
  tail -n 80 "${LOG_DIR}/${lane}_supervisor.log" >&2 || true
  return 1
}

stop_lane() {
  local lane="$1"
  local pid
  local attempt
  pid="$(lane_pid "${lane}")"
  if [[ -z "${pid}" ]]; then
    echo "${lane} supervisor has no recorded PID"
    return 0
  fi
  if ! lane_process_is_owned "${lane}" "${pid}"; then
    echo "${lane} PID ${pid} is not a live owned supervisor; leaving it untouched"
    return 0
  fi
  kill -TERM "${pid}"
  for attempt in $(seq 1 30); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${lane} supervisor PID ${pid} stopped"
      return 0
    fi
    sleep 1
  done
  echo "${lane} supervisor PID ${pid} did not stop after SIGTERM" >&2
  return 1
}

show_status() {
  env "${RUNTIME_ENV[@]}" "${PYTHON_BIN}" - "${STATE_ROOT}" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
result = {
    "schema": "ipfs_accelerate_py/vfs-symbolic-assurance-control-status@1",
    "state_root": str(root),
    "lanes": {},
}
for lane in ("vfs_grok", "vfs_codex"):
    pid_path = root / "runtime" / f"{lane}_supervisor.pid"
    status_path = root / "state" / f"{lane}_supervisor_status.json"
    try:
        supervisor_pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        supervisor_pid = 0
    supervisor_alive = False
    if supervisor_pid > 0:
        try:
            os.kill(supervisor_pid, 0)
            supervisor_alive = True
        except OSError:
            pass
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        status = {}
    daemon_pid = status.get("daemon_pid")
    daemon_alive = False
    if isinstance(daemon_pid, int) and daemon_pid > 0:
        try:
            os.kill(daemon_pid, 0)
            daemon_alive = True
        except OSError:
            pass
    result["lanes"][lane] = {
        "supervisor_pid": supervisor_pid or None,
        "supervisor_alive": supervisor_alive,
        "status_path": str(status_path),
        "status": status.get("status"),
        "updated_at": status.get("updated_at"),
        "daemon_pid": daemon_pid,
        "daemon_pid_alive": daemon_alive,
        "active_task_id": status.get("active_task_id"),
        "last_agentic_maintenance_phase": status.get(
            "last_agentic_maintenance_phase"
        ),
        "last_agentic_maintenance_error": status.get(
            "last_agentic_maintenance_error"
        ),
        "last_log_path": status.get("last_log_path"),
    }
print(json.dumps(result, indent=2, sort_keys=True))
PY
}

start_all() {
  local lock_path="${STATE_ROOT}/control.lock"
  prepare_state_dirs
  exec 9> "${lock_path}"
  if ! flock -n 9; then
    echo "Another VFS assurance control operation owns ${lock_path}" >&2
    return 2
  fi
  require_isolated_clean_checkout
  provider_preflight
  project_objectives
  reconciliation_preflight "vfs_grok" "0"
  reconciliation_preflight "vfs_codex" "1"
  launch_lane "vfs_grok" "0" "grok-build"
  launch_lane "vfs_codex" "1" "codex"
  if ! verify_lane_started "vfs_grok"; then
    stop_lane "vfs_codex" || true
    return 1
  fi
  if ! verify_lane_started "vfs_codex"; then
    stop_lane "vfs_grok" || true
    return 1
  fi
  show_status
}

stop_all() {
  local lock_path="${STATE_ROOT}/control.lock"
  prepare_state_dirs
  exec 9> "${lock_path}"
  if ! flock -n 9; then
    echo "Another VFS assurance control operation owns ${lock_path}" >&2
    return 2
  fi
  stop_lane "vfs_grok"
  stop_lane "vfs_codex"
  show_status
}

case "${CONTROL_COMMAND}" in
  start)
    start_all
    ;;
  status)
    prepare_state_dirs
    show_status
    ;;
  preflight)
    prepare_state_dirs
    require_isolated_clean_checkout
    provider_preflight
    project_objectives
    reconciliation_preflight "vfs_grok" "0"
    reconciliation_preflight "vfs_codex" "1"
    ;;
  stop)
    stop_all
    ;;
  *)
    echo "Usage: ${CONTROL_PATH} {start|status|preflight|stop}" >&2
    exit 2
    ;;
esac
