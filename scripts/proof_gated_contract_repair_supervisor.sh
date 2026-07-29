#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${RPR_PYTHON_BIN:-python3}"
CODEX_BIN="${RPR_CODEX_BIN:-/usr/local/bin/codex}"
TARGET_BRANCH="agent/proof-gated-contract-repair"

PLAN_PATH="docs/architecture/AGENT_SUPERVISOR_PROOF_GATED_CONTRACT_REPAIR_PLAN.md"
OBJECTIVE_PATH="docs/architecture/agent_supervisor_proof_gated_contract_repair.objectives.md"
TODO_PATH="docs/architecture/agent_supervisor_proof_gated_contract_repair.todo.md"
SCHEDULER_PATH="config/agent_supervisor_proof_gated_contract_repair_scheduler.json"
LAUNCHER_PATH="scripts/proof_gated_contract_repair_supervisor.sh"

DEFAULT_STATE_BASE="${XDG_STATE_HOME:-${HOME}/.local/state}"
PROGRAM_ROOT="${RPR_STATE_ROOT:-${DEFAULT_STATE_BASE}/ipfs_accelerate_py/proof_gated_contract_repair}"
RUNTIME_ROOT="${PROGRAM_ROOT}/runtime"
STATE_ROOT="${PROGRAM_ROOT}/state"
PREFLIGHT_ROOT="${PROGRAM_ROOT}/preflight"
WORKTREE_ROOT="${PROGRAM_ROOT}/worktrees"
MERGE_QUEUE_ROOT="${PROGRAM_ROOT}/merge-queue"
MASTER_PID_PATH="${RUNTIME_ROOT}/master.pid"
MASTER_LOG_PATH="${RUNTIME_ROOT}/master.log"
LANE_COUNT="${RPR_LANE_COUNT:-4}"
DURATION_SECONDS="${RPR_DURATION_SECONDS:-28800}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/ipfs_datasets_py${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER="${IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER:-codex}"
export IPFS_ACCELERATE_AGENT_CODEX_MODEL="${IPFS_ACCELERATE_AGENT_CODEX_MODEL:-gpt-5.6-terra}"
export IPFS_DATASETS_AUTO_INSTALL=false
export IPFS_AUTO_INSTALL=false
export IPFS_DATASETS_PY_MINIMAL_IMPORTS=1

bind_managed_contract_repair_toolchain() {
  local bindings=""
  local key=""
  local value=""
  bindings="$(
    "${PYTHON_BIN}" \
      -m ipfs_accelerate_py.agent_supervisor.integrations.contract_repair_dependencies \
      --print-env 2>/dev/null || true
  )"
  while IFS='=' read -r key value; do
    case "${key}" in
      PATH)
        export PATH="${value}"
        ;;
      TYPESCRIPT_PATH)
        export TYPESCRIPT_PATH="${value}"
        ;;
    esac
  done <<<"${bindings}"
}

require_file() {
  local path="$1"
  if [[ ! -f "${REPO_ROOT}/${path}" ]]; then
    echo "required file is missing: ${path}" >&2
    return 2
  fi
}

live_pid_from_file() {
  local path="$1"
  local pid=""
  if [[ -f "${path}" ]]; then
    pid="$(tr -dc '0-9' <"${path}")"
  fi
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    printf '%s\n' "${pid}"
    return 0
  fi
  return 1
}

verify_branch_and_sources() {
  local current_branch=""
  local expected_datasets=""
  local actual_datasets=""
  current_branch="$(git -C "${REPO_ROOT}" branch --show-current)"
  if [[ "${current_branch}" != "${TARGET_BRANCH}" ]]; then
    echo "expected branch ${TARGET_BRANCH}; found ${current_branch:-detached}" >&2
    return 2
  fi
  if [[ ! -f "${REPO_ROOT}/ipfs_datasets_py/ipfs_datasets_py/logic/__init__.py" ]]; then
    echo "ipfs_datasets_py gitlink is not initialized" >&2
    return 2
  fi
  expected_datasets="$(git -C "${REPO_ROOT}" rev-parse HEAD:ipfs_datasets_py)"
  actual_datasets="$(git -C "${REPO_ROOT}/ipfs_datasets_py" rev-parse HEAD)"
  if [[ "${expected_datasets}" != "${actual_datasets}" ]]; then
    echo "ipfs_datasets_py revision mismatch: expected ${expected_datasets}, found ${actual_datasets}" >&2
    return 2
  fi
  "${PYTHON_BIN}" - "${REPO_ROOT}" "${expected_datasets}" <<'PY'
import pathlib
import subprocess
import sys

import ipfs_accelerate_py
import ipfs_datasets_py.logic

root = pathlib.Path(sys.argv[1]).resolve()
expected = sys.argv[2]
accelerator_file = pathlib.Path(ipfs_accelerate_py.__file__).resolve()
datasets_file = pathlib.Path(ipfs_datasets_py.logic.__file__).resolve()
if root not in accelerator_file.parents:
    raise SystemExit(f"accelerator import escaped target checkout: {accelerator_file}")
if root.joinpath("ipfs_datasets_py") not in datasets_file.parents:
    raise SystemExit(f"datasets import escaped exact gitlink: {datasets_file}")
actual = subprocess.check_output(
    ["git", "-C", str(root / "ipfs_datasets_py"), "rev-parse", "HEAD"],
    text=True,
).strip()
if actual != expected:
    raise SystemExit(f"datasets import revision mismatch: {actual} != {expected}")
print(f"accelerator_module={accelerator_file}")
print(f"datasets_logic_module={datasets_file}")
print(f"datasets_revision={actual}")
PY
}

validate_control_plane() {
  "${PYTHON_BIN}" - \
    "${REPO_ROOT}/${OBJECTIVE_PATH}" \
    "${REPO_ROOT}/${TODO_PATH}" \
    "${REPO_ROOT}/${SCHEDULER_PATH}" <<'PY'
import json
import pathlib
import sys

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    parse_task_file,
)

objective_path = pathlib.Path(sys.argv[1])
todo_path = pathlib.Path(sys.argv[2])
scheduler_path = pathlib.Path(sys.argv[3])

goals = parse_goal_heap(objective_path.read_text(encoding="utf-8"))
goal_ids = {goal.goal_id for goal in goals}
if "RPR-G000" not in goal_ids:
    raise SystemExit("RPR-G000 is missing")
for goal in goals:
    for dependency in goal.dependencies:
        if dependency not in goal_ids:
            raise SystemExit(
                f"unknown objective dependency: {goal.goal_id}->{dependency}"
            )

tasks = parse_task_file(todo_path, task_header_prefix="RPR-")
task_ids = {task.task_id for task in tasks}
if len(tasks) != len(task_ids):
    raise SystemExit("duplicate task id")
if "RPR-000" not in task_ids:
    raise SystemExit("RPR-000 is missing")
for task in tasks:
    for dependency in task.depends_on:
        if dependency not in task_ids:
            raise SystemExit(
                f"unknown task dependency: {task.task_id}->{dependency}"
            )
    goal_id = task.metadata.get("goal id", "")
    if goal_id and goal_id not in goal_ids:
        raise SystemExit(f"unknown task goal: {task.task_id}->{goal_id}")

completed = {task.task_id for task in tasks if task.status == "completed"}
ready = sorted(
    task.task_id
    for task in tasks
    if task.status == "todo" and set(task.depends_on).issubset(completed)
)
if len(ready) < 4:
    raise SystemExit(f"expected at least four ready tasks, found {ready}")

scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
if scheduler.get("task_prefix") != "RPR-":
    raise SystemExit("scheduler task prefix mismatch")
if scheduler.get("merge_target_branch") != "agent/proof-gated-contract-repair":
    raise SystemExit("scheduler merge target mismatch")
if scheduler.get("objective_refill_enabled") is not False:
    raise SystemExit("objective refill must be disabled at launch")
if scheduler.get("codebase_refill_enabled") is not False:
    raise SystemExit("codebase refill must be disabled at launch")
proof_policy = scheduler.get("proof_policy") or {}
if proof_policy.get("datasets_logic_required_before_target_admission") is not True:
    raise SystemExit("datasets logic gate is not enabled")
if proof_policy.get("vector_semantic_authority") is not False:
    raise SystemExit("vector semantic authority must be false")

print(
    json.dumps(
        {
            "goal_count": len(goals),
            "task_count": len(tasks),
            "completed_count": len(completed),
            "ready_task_ids": ready,
        },
        sort_keys=True,
    )
)
PY
}

doctor() {
  bind_managed_contract_repair_toolchain
  require_file "${PLAN_PATH}"
  require_file "${OBJECTIVE_PATH}"
  require_file "${TODO_PATH}"
  require_file "${SCHEDULER_PATH}"
  require_file "${LAUNCHER_PATH}"
  verify_branch_and_sources
  validate_control_plane
  "${PYTHON_BIN}" -m json.tool "${REPO_ROOT}/${SCHEDULER_PATH}" >/dev/null
  if [[ ! -x "${CODEX_BIN}" ]]; then
    echo "Codex executable is unavailable: ${CODEX_BIN}" >&2
    return 2
  fi
  "${CODEX_BIN}" login status
  if command -v cvc5 >/dev/null 2>&1; then
    cvc5 --version | head -n 1
  else
    echo "cvc5 unavailable; supported SMT obligations will remain non-conclusive"
  fi
  if command -v z3 >/dev/null 2>&1; then
    z3 --version | head -n 1
  else
    echo "z3 unavailable (recorded capability; not a launch blocker)"
  fi
  if command -v mypy >/dev/null 2>&1; then
    mypy --version
  else
    echo "mypy unavailable (recorded capability; not a launch blocker)"
  fi
  if command -v ruff >/dev/null 2>&1; then
    ruff --version
  else
    echo "ruff unavailable (worker validation may be reduced)"
  fi
  if command -v tsc >/dev/null 2>&1; then
    tsc --version
  else
    echo "managed TypeScript unavailable; run ipfs-accelerate-contract-repair-deps --install typescript"
  fi
  if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=normal)" ]]; then
    echo "target checkout is dirty; commit the control plane before start" >&2
    return 2
  fi
  echo "doctor: healthy"
}

preflight() {
  mkdir -p "${PREFLIGHT_ROOT}" "${WORKTREE_ROOT}" "${MERGE_QUEUE_ROOT}"
  (
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" \
      -m ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon \
      --once \
      --todo-path "${TODO_PATH}" \
      --task-prefix "RPR-" \
      --state-dir "${PREFLIGHT_ROOT}" \
      --state-prefix rpr \
      --worktree-root "${WORKTREE_ROOT}/preflight" \
      --worktree-submodule-path ipfs_datasets_py \
      --merge-target-branch "${TARGET_BRANCH}" \
      --merge-queue-dir "${MERGE_QUEUE_ROOT}" \
      --implementation-protected-path "${PLAN_PATH}" \
      --implementation-protected-path "${OBJECTIVE_PATH}" \
      --implementation-protected-path "${TODO_PATH}" \
      --implementation-protected-path "${SCHEDULER_PATH}" \
      --implementation-protected-path "${LAUNCHER_PATH}" \
      --log-level INFO
  )
  "${PYTHON_BIN}" - "${PREFLIGHT_ROOT}/rpr_task_state.json" <<'PY'
import json
import os
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
task_count = int(payload.get("task_count") or 0)
ready_count = int(payload.get("eligible_ready_count") or 0)
blocked = payload.get("blocked_task_ids") or []
reason = str(payload.get("selection_idle_reason") or "")
if task_count <= 0:
    raise SystemExit(f"preflight parsed no tasks: {payload}")
if ready_count <= 0 and int(payload.get("completed_count") or 0) < task_count:
    raise SystemExit(f"preflight has no eligible ready work: {payload}")
if blocked:
    raise SystemExit(f"preflight found blocked tasks: {blocked}")
if reason and ready_count:
    raise SystemExit(f"unexpected selection idle reason with ready work: {reason}")
age = time.time() - path.stat().st_mtime
if age > 120:
    raise SystemExit(f"preflight task state is stale: {age:.1f}s")
print(
    json.dumps(
        {
            "task_count": task_count,
            "eligible_ready_count": ready_count,
            "blocked_task_ids": blocked,
            "selection_idle_reason": reason,
        },
        sort_keys=True,
    )
)
PY
}

start() {
  if pid="$(live_pid_from_file "${MASTER_PID_PATH}")"; then
    echo "proof-gated contract-repair supervisor is already running with master pid ${pid}" >&2
    return 2
  fi
  doctor
  preflight
  mkdir -p "${RUNTIME_ROOT}" "${STATE_ROOT}" "${WORKTREE_ROOT}" "${MERGE_QUEUE_ROOT}"
  (
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" \
      -m ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner \
      --repo-root "${REPO_ROOT}" \
      --duration-seconds "${DURATION_SECONDS}" \
      --master-dir "${RUNTIME_ROOT}" \
      --master-log "${MASTER_LOG_PATH}" \
      --master-pid-path "${MASTER_PID_PATH}" \
      --label proof-gated-contract-repair \
      --implementation-track "rpr|scripts/ops/agent_supervisor/implementation_supervisor_entry.py|${STATE_ROOT}|rpr" \
      --implementation-supervisor-lanes-per-track "${LANE_COUNT}" \
      --implementation-supervisor-strict-task-sharding \
      --common-arg=--todo-path \
      --common-arg="${TODO_PATH}" \
      --common-arg=--task-prefix \
      --common-arg=RPR- \
      --common-arg=--implement \
      --common-arg=--max-task-attempts \
      --common-arg=3 \
      --common-arg=--implementation-retry-budget \
      --common-arg=3 \
      --common-arg=--validation-retry-budget \
      --common-arg=3 \
      --common-arg=--merge-retry-budget \
      --common-arg=3 \
      --common-arg=--implementation-timeout \
      --common-arg=3600 \
      --common-arg=--implementation-max-timeout \
      --common-arg=7200 \
      --common-arg=--implementation-log-stall-seconds \
      --common-arg=1200 \
      --common-arg=--daemon-interval \
      --common-arg=60 \
      --common-arg=--check-interval \
      --common-arg=30 \
      --common-arg=--stale-seconds \
      --common-arg=1800 \
      --common-arg=--watchdog-startup-grace-seconds \
      --common-arg=300 \
      --common-arg=--worktree-root \
      --common-arg="${WORKTREE_ROOT}" \
      --common-arg=--worktree-submodule-path \
      --common-arg=ipfs_datasets_py \
      --common-arg=--merge-target-branch \
      --common-arg="${TARGET_BRANCH}" \
      --common-arg=--merge-queue-dir \
      --common-arg="${MERGE_QUEUE_ROOT}" \
      --common-arg=--implementation-protected-path \
      --common-arg="${PLAN_PATH}" \
      --common-arg=--implementation-protected-path \
      --common-arg="${OBJECTIVE_PATH}" \
      --common-arg=--implementation-protected-path \
      --common-arg="${TODO_PATH}" \
      --common-arg=--implementation-protected-path \
      --common-arg="${SCHEDULER_PATH}" \
      --common-arg=--implementation-protected-path \
      --common-arg="${LAUNCHER_PATH}" \
      --common-arg=--no-objective-task-janitor \
      --common-arg=--no-objective-goal-refinement \
      --common-arg=--no-objective-goal-migration \
      --common-arg=--log-level \
      --common-arg=INFO \
      --detach
  )
  local attempt=""
  for attempt in {1..90}; do
    if live_pid_from_file "${MASTER_PID_PATH}" >/dev/null \
      && status >/dev/null 2>&1; then
      status
      return 0
    fi
    sleep 1
  done
  status || true
  echo "master and all lanes did not become healthy within 90 seconds" >&2
  return 1
}

status() {
  local master_pid=""
  if master_pid="$(live_pid_from_file "${MASTER_PID_PATH}")"; then
    echo "master: running pid=${master_pid}"
  else
    echo "master: stopped"
  fi
  "${PYTHON_BIN}" - "${STATE_ROOT}" "${LANE_COUNT}" <<'PY'
import json
import os
import pathlib
import sys
import time

root = pathlib.Path(sys.argv[1])
lane_count = int(sys.argv[2])
failed = False
for lane in range(lane_count):
    state_dir = root / f"lane-{lane}"
    prefix = f"rpr_lane_{lane}"
    supervisor_path = state_dir / f"{prefix}_supervisor_status.json"
    task_path = state_dir / f"{prefix}_task_state.json"
    values = {"lane": lane, "supervisor": "missing", "task_state": "missing"}
    if supervisor_path.exists():
        try:
            supervisor = json.loads(supervisor_path.read_text(encoding="utf-8"))
            pid = int(supervisor.get("pid") or supervisor.get("supervisor_pid") or 0)
            alive = pid > 0
            if alive:
                try:
                    os.kill(pid, 0)
                except OSError:
                    alive = False
            values["supervisor"] = supervisor.get("status", "unknown")
            values["supervisor_pid"] = pid
            values["supervisor_pid_alive"] = alive
            values["restart_count"] = supervisor.get("restart_count", 0)
            if values["supervisor"] != "running" or not alive:
                failed = True
        except Exception as exc:
            values["supervisor_error"] = str(exc)
            failed = True
    else:
        failed = True
    if task_path.exists():
        try:
            task = json.loads(task_path.read_text(encoding="utf-8"))
            values["task_state"] = task.get("status", "available")
            values["active_task_id"] = task.get("active_task_id", "")
            values["active_phase"] = task.get("active_phase", "")
            values["eligible_ready_count"] = int(task.get("eligible_ready_count") or 0)
            values["blocked_count"] = int(task.get("blocked_count") or 0)
            values["selection_idle_reason"] = task.get("selection_idle_reason", "")
            values["heartbeat_age_seconds"] = round(
                time.time() - task_path.stat().st_mtime, 1
            )
            if values["heartbeat_age_seconds"] > 180:
                failed = True
            if values["blocked_count"]:
                failed = True
        except Exception as exc:
            values["task_state_error"] = str(exc)
            failed = True
    else:
        failed = True
    print(json.dumps(values, sort_keys=True))
if failed:
    raise SystemExit(1)
PY
}

stop() {
  local pid=""
  if ! pid="$(live_pid_from_file "${MASTER_PID_PATH}")"; then
    echo "master is already stopped"
    return 0
  fi
  kill "${pid}"
  local attempt=""
  for attempt in {1..30}; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "master stopped"
      return 0
    fi
    sleep 1
  done
  echo "master did not stop within 30 seconds; no forced termination was attempted" >&2
  return 1
}

case "${1:-}" in
  doctor)
    doctor
    ;;
  preflight)
    verify_branch_and_sources
    validate_control_plane
    preflight
    ;;
  start)
    start
    ;;
  status)
    status
    ;;
  stop)
    stop
    ;;
  *)
    echo "usage: $0 {doctor|preflight|start|status|stop}" >&2
    exit 2
    ;;
esac
