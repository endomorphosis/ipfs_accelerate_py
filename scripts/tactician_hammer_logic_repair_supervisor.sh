#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${LPR_PYTHON_BIN:-python3}"
DEFAULT_RUNNER_MODULE="ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner"
RUNNER_MODULE="${LPR_RUNNER_MODULE:-${DEFAULT_RUNNER_MODULE}}"
TEST_MODE="${LPR_TEST_MODE:-0}"
TARGET_BRANCH="agent/proof-gated-contract-repair"
LABEL="tactician-hammer-logic-repair"

PLAN_PATH="docs/architecture/AGENT_SUPERVISOR_TACTICIAN_HAMMER_LOGIC_REPAIR_PLAN.md"
OBJECTIVE_PATH="docs/architecture/agent_supervisor_tactician_hammer_logic_repair.objectives.md"
TODO_PATH="docs/architecture/agent_supervisor_tactician_hammer_logic_repair.todo.md"
SCHEDULER_PATH="config/agent_supervisor_tactician_hammer_logic_repair_scheduler.json"
VALIDATOR_PATH="scripts/validate_tactician_hammer_logic_repair_board.py"
LAUNCHER_PATH="scripts/tactician_hammer_logic_repair_supervisor.sh"

DEFAULT_STATE_BASE="$(git -C "${REPO_ROOT}" rev-parse --git-path agent-supervisor-state)"
if [[ "${DEFAULT_STATE_BASE}" != /* ]]; then
  DEFAULT_STATE_BASE="${REPO_ROOT}/${DEFAULT_STATE_BASE}"
fi
PROGRAM_ROOT="${LPR_STATE_ROOT:-${DEFAULT_STATE_BASE}/tactician_hammer_logic_repair}"
RUNTIME_ROOT="${PROGRAM_ROOT}/runtime"
STATE_ROOT="${PROGRAM_ROOT}/state"
PREFLIGHT_ROOT="${PROGRAM_ROOT}/preflight"
WORKTREE_ROOT="${PROGRAM_ROOT}/worktrees"
MERGE_QUEUE_ROOT="${PROGRAM_ROOT}/merge-queue"
MASTER_PID_PATH="${RUNTIME_ROOT}/master.pid"
MASTER_IDENTITY_PATH="${RUNTIME_ROOT}/master.identity.json"
MASTER_LOG_PATH="${RUNTIME_ROOT}/master.log"
LAUNCH_RECEIPT_PATH="${RUNTIME_ROOT}/launch-receipt.json"
LANE_COUNT="${LPR_LANE_COUNT:-4}"
DURATION_SECONDS="${LPR_DURATION_SECONDS:-28800}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/ipfs_datasets_py${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER="${IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER:-auto}"
export IPFS_ACCELERATE_AGENT_CODEX_MODEL="${IPFS_ACCELERATE_AGENT_CODEX_MODEL:-gpt-5.6-terra}"
export IPFS_ACCELERATE_AGENT_RECLAIM_DEAD_WORKTREE_LEASES_ON_STARTUP="${IPFS_ACCELERATE_AGENT_RECLAIM_DEAD_WORKTREE_LEASES_ON_STARTUP:-1}"
export IPFS_DATASETS_AUTO_INSTALL=false
export IPFS_AUTO_INSTALL=false
export IPFS_DATASETS_PY_MINIMAL_IMPORTS=1

CONTROL_PATHS=(
  "${PLAN_PATH}"
  "${OBJECTIVE_PATH}"
  "${TODO_PATH}"
  "${SCHEDULER_PATH}"
  "${VALIDATOR_PATH}"
  "${LAUNCHER_PATH}"
)

validate_test_mode() {
  if [[ "${TEST_MODE}" != "1" ]]; then
    return 0
  fi
  if [[ "${RUNNER_MODULE}" == "${DEFAULT_RUNNER_MODULE}" ]]; then
    echo "LPR_TEST_MODE requires an explicit fake LPR_RUNNER_MODULE" >&2
    return 2
  fi
  if [[ -z "${LPR_STATE_ROOT:-}" ]]; then
    echo "LPR_TEST_MODE requires an explicit isolated LPR_STATE_ROOT" >&2
    return 2
  fi
}

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

validate_runtime_root() {
  "${PYTHON_BIN}" - "${PROGRAM_ROOT}" "${REPO_ROOT}" <<'PY'
import pathlib
import sys

state = pathlib.Path(sys.argv[1]).expanduser().resolve(strict=False)
repo = pathlib.Path(sys.argv[2]).resolve()
for forbidden in (pathlib.Path("/"), repo):
    if state == forbidden:
        raise SystemExit(f"unsafe supervisor state root: {state}")
if len(state.parts) < 4:
    raise SystemExit(f"supervisor state root is too broad: {state}")
print(f"state_root={state}")
PY
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

master_state_json() {
  "${PYTHON_BIN}" - \
    "${MASTER_PID_PATH}" \
    "${MASTER_IDENTITY_PATH}" \
    "${REPO_ROOT}" \
    "${RUNNER_MODULE}" \
    "${LABEL}" <<'PY'
import json
import os
import pathlib
import sys

pid_path = pathlib.Path(sys.argv[1])
identity_path = pathlib.Path(sys.argv[2])
repo = pathlib.Path(sys.argv[3]).resolve()
module = sys.argv[4]
label = sys.argv[5]

def result(status, **extra):
    print(json.dumps({"status": status, **extra}, sort_keys=True))

def process(pid):
    stat = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    close = stat.rfind(")")
    fields = stat[close + 2:].split()
    if not fields or fields[0] == "Z":
        raise ProcessLookupError(pid)
    start_time = int(fields[19])
    cwd = pathlib.Path(os.readlink(f"/proc/{pid}/cwd")).resolve()
    argv = tuple(
        item.decode("utf-8", "surrogateescape")
        for item in pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        if item
    )
    env = {}
    for item in pathlib.Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        key, separator, value = item.partition(b"=")
        if separator:
            env[key.decode("utf-8", "surrogateescape")] = value.decode("utf-8", "surrogateescape")
    return start_time, cwd, argv, env

try:
    raw_pid = pid_path.read_text(encoding="ascii").strip()
    pid = int(raw_pid)
except FileNotFoundError:
    result("missing", pid=None)
    raise SystemExit(0)
except (OSError, ValueError):
    result("invalid", pid=None)
    raise SystemExit(0)
if pid <= 0:
    result("invalid", pid=None)
    raise SystemExit(0)
try:
    current_start, cwd, argv, env = process(pid)
except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
    result("stale", pid=pid)
    raise SystemExit(0)
try:
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
except Exception:
    result("foreign", pid=pid, reason="missing_or_invalid_identity")
    raise SystemExit(0)

def pair(flag, value):
    return any(argv[index:index + 2] == (flag, value) for index in range(len(argv) - 1))

checks = {
    "pid": identity.get("pid") == pid,
    "start_time": identity.get("start_time_ticks") == current_start,
    "repo": cwd == repo and identity.get("repo_root") == str(repo),
    "module": pair("-m", module) and identity.get("runner_module") == module,
    "label": pair("--label", label) and identity.get("label") == label,
    "repo_arg": pair("--repo-root", str(repo)),
    "pid_arg": pair("--master-pid-path", str(pid_path)),
    "run_id": bool(identity.get("run_id"))
        and env.get("IPFS_ACCELERATE_LPR_RUN_ID") == identity.get("run_id"),
}
failed = sorted(name for name, okay in checks.items() if not okay)
if failed:
    result("foreign", pid=pid, reason="identity_mismatch", failed_checks=failed)
else:
    result("owned", pid=pid, start_time_ticks=current_start, run_id=identity["run_id"])
PY
}

master_state_field() {
  local field="$1"
  "${PYTHON_BIN}" -c \
    'import json,sys; print(json.load(sys.stdin).get(sys.argv[1], ""))' \
    "${field}"
}

cleanup_dead_master_markers() {
  local state_json=""
  local state=""
  state_json="$(master_state_json)"
  state="$(master_state_field status <<<"${state_json}")"
  if [[ "${state}" == "foreign" || "${state}" == "owned" ]]; then
    echo "refusing to remove a live ${state} master marker: ${state_json}" >&2
    return 2
  fi
  "${PYTHON_BIN}" - "${MASTER_PID_PATH}" "${MASTER_IDENTITY_PATH}" <<'PY'
import pathlib
import sys
for raw in sys.argv[1:]:
    pathlib.Path(raw).unlink(missing_ok=True)
PY
}

capture_master_identity() {
  local run_id="$1"
  "${PYTHON_BIN}" - \
    "${MASTER_PID_PATH}" \
    "${MASTER_IDENTITY_PATH}" \
    "${REPO_ROOT}" \
    "${RUNNER_MODULE}" \
    "${LABEL}" \
    "${run_id}" <<'PY'
import json
import os
import pathlib
import sys
import tempfile
import time

pid_path = pathlib.Path(sys.argv[1])
identity_path = pathlib.Path(sys.argv[2])
repo = pathlib.Path(sys.argv[3]).resolve()
module, label, run_id = sys.argv[4:7]

deadline = time.monotonic() + 10.0
while True:
    try:
        pid = int(pid_path.read_text(encoding="ascii").strip())
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        close = stat.rfind(")")
        fields = stat[close + 2:].split()
        if fields and fields[0] != "Z":
            break
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, ValueError):
        pass
    if time.monotonic() >= deadline:
        raise SystemExit("detached master did not publish a live PID")
    time.sleep(0.05)

start_time = int(fields[19])
cwd = pathlib.Path(os.readlink(f"/proc/{pid}/cwd")).resolve()
argv = tuple(
    item.decode("utf-8", "surrogateescape")
    for item in pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    if item
)
env = {}
for item in pathlib.Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
    key, separator, value = item.partition(b"=")
    if separator:
        env[key.decode("utf-8", "surrogateescape")] = value.decode("utf-8", "surrogateescape")
def pair(flag, value):
    return any(argv[index:index + 2] == (flag, value) for index in range(len(argv) - 1))
if cwd != repo or not pair("-m", module) or not pair("--label", label):
    raise SystemExit("detached master command identity mismatch")
if not pair("--repo-root", str(repo)) or not pair("--master-pid-path", str(pid_path)):
    raise SystemExit("detached master path binding mismatch")
if env.get("IPFS_ACCELERATE_LPR_RUN_ID") != run_id:
    raise SystemExit("detached master run identity mismatch")
payload = {
    "schema": "ipfs_accelerate_py.agent_supervisor.tactician_hammer_logic_repair.master_identity@1",
    "pid": pid,
    "start_time_ticks": start_time,
    "repo_root": str(repo),
    "runner_module": module,
    "label": label,
    "run_id": run_id,
}
identity_path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=identity_path.name + ".", dir=identity_path.parent)
try:
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, identity_path)
finally:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
print(json.dumps(payload, sort_keys=True))
PY
}

record_launch_receipt() {
  local run_id="$1"
  local accelerator_revision=""
  local datasets_revision=""
  accelerator_revision="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  datasets_revision="$(git -C "${REPO_ROOT}/ipfs_datasets_py" rev-parse HEAD)"
  "${PYTHON_BIN}" - \
    "${LAUNCH_RECEIPT_PATH}" \
    "${run_id}" \
    "${accelerator_revision}" \
    "${datasets_revision}" \
    "${TARGET_BRANCH}" <<'PY'
import datetime
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "ipfs_accelerate_py.agent_supervisor.tactician_hammer_logic_repair.launch_receipt@1",
    "run_id": sys.argv[2],
    "accelerator_revision": sys.argv[3],
    "datasets_revision": sys.argv[4],
    "accelerator_branch": sys.argv[5],
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
try:
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
finally:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
print(json.dumps(payload, sort_keys=True))
PY
}

doctor() {
  validate_test_mode
  if [[ "${TEST_MODE}" == "1" ]]; then
    validate_runtime_root
    echo "doctor: healthy (isolated fake-process mode)"
    return 0
  fi
  bind_managed_contract_repair_toolchain
  local path=""
  for path in "${CONTROL_PATHS[@]}"; do
    require_file "${path}"
  done
  validate_runtime_root
  verify_branch_and_sources
  "${PYTHON_BIN}" "${REPO_ROOT}/${VALIDATOR_PATH}" --check-all
  "${PYTHON_BIN}" -m json.tool "${REPO_ROOT}/${SCHEDULER_PATH}" >/dev/null
  "${PYTHON_BIN}" - "${REPO_ROOT}" <<'PY'
import pathlib
import sys

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
)

repo_root = pathlib.Path(sys.argv[1]).resolve()
probe = object.__new__(PortalImplementationDaemon)
probe.implementation_command = ""
try:
    command = PortalImplementationDaemon._build_implementation_command(probe, repo_root)
except Exception as exc:
    raise SystemExit(f"selected implementation provider is unavailable: {exc}") from exc
if not command:
    raise SystemExit("selected implementation provider produced no command")
print(f"implementation_provider_launcher={pathlib.Path(command[0]).name}")
PY
  if command -v cvc5 >/dev/null 2>&1; then
    cvc5 --version | head -n 1
  else
    echo "cvc5 unavailable; Hammer execution remains disabled in shadow mode"
  fi
  if command -v z3 >/dev/null 2>&1; then
    z3 --version | head -n 1
  else
    echo "z3 unavailable; Hammer execution remains disabled in shadow mode"
  fi
  echo "hammer_import_isolation=unsafe_process_global_environment; native execution remains disabled in shadow mode"
  echo "doctor: healthy"
}

require_committed_clean_control_plane() {
  if [[ "${TEST_MODE}" == "1" ]]; then
    return 0
  fi
  local path=""
  for path in "${CONTROL_PATHS[@]}"; do
    if ! git -C "${REPO_ROOT}" ls-files --error-unmatch -- "${path}" >/dev/null 2>&1; then
      echo "control artifact is not tracked: ${path}" >&2
      return 2
    fi
    if ! git -C "${REPO_ROOT}" diff --quiet HEAD -- "${path}"; then
      echo "control artifact differs from HEAD: ${path}" >&2
      return 2
    fi
  done
  if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=normal)" ]]; then
    echo "target checkout is dirty; commit the bootstrap before start" >&2
    return 2
  fi
}

require_bootstrap_completed() {
  "${PYTHON_BIN}" - "${REPO_ROOT}/${TODO_PATH}" <<'PY'
import pathlib
import sys

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    parse_task_file,
)

tasks = parse_task_file(pathlib.Path(sys.argv[1]), task_header_prefix="LPR-")
bootstrap = next((task for task in tasks if task.task_id == "LPR-000"), None)
if bootstrap is None or bootstrap.status != "completed":
    raise SystemExit("LPR-000 must be completed before the sealed board is launched")
print("bootstrap_status=completed")
PY
}

preflight() {
  validate_test_mode
  if [[ "${TEST_MODE}" == "1" ]]; then
    echo '{"completed_count":1,"drained":false,"eligible_ready_count":4,"task_count":21}'
    return 0
  fi
  verify_branch_and_sources
  "${PYTHON_BIN}" "${REPO_ROOT}/${VALIDATOR_PATH}" --check-all
  mkdir -p "${PREFLIGHT_ROOT}" "${WORKTREE_ROOT}" "${MERGE_QUEUE_ROOT}"
  (
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" \
      -m ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon \
      --once \
      --todo-path "${TODO_PATH}" \
      --task-prefix "LPR-" \
      --state-dir "${PREFLIGHT_ROOT}" \
      --state-prefix lpr \
      --worktree-root "${WORKTREE_ROOT}/preflight" \
      --worktree-submodule-path ipfs_datasets_py \
      --merge-target-branch "${TARGET_BRANCH}" \
      --merge-queue-dir "${MERGE_QUEUE_ROOT}" \
      --implementation-protected-path "${PLAN_PATH}" \
      --implementation-protected-path "${OBJECTIVE_PATH}" \
      --implementation-protected-path "${TODO_PATH}" \
      --implementation-protected-path "${SCHEDULER_PATH}" \
      --implementation-protected-path "${VALIDATOR_PATH}" \
      --implementation-protected-path "${LAUNCHER_PATH}" \
      --log-level INFO
  )
  "${PYTHON_BIN}" - "${PREFLIGHT_ROOT}/lpr_task_state.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
task_count = int(payload.get("task_count") or 0)
completed = int(payload.get("completed_count") or 0)
ready = int(payload.get("eligible_ready_count") or 0)
blocked = tuple(payload.get("blocked_task_ids") or ())
if task_count != 21:
    raise SystemExit(f"preflight parsed unexpected task count: {payload}")
if blocked:
    raise SystemExit(f"preflight found blocked tasks: {blocked}")
if completed < task_count and ready <= 0:
    raise SystemExit(f"preflight has no ready work while incomplete: {payload}")
print(json.dumps({
    "task_count": task_count,
    "completed_count": completed,
    "eligible_ready_count": ready,
    "drained": completed == task_count,
}, sort_keys=True))
PY
}

launch_master() {
  local run_id="$1"
  mkdir -p "${RUNTIME_ROOT}" "${STATE_ROOT}" "${WORKTREE_ROOT}" "${MERGE_QUEUE_ROOT}"
  export IPFS_ACCELERATE_LPR_RUN_ID="${run_id}"
  (
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" -m "${RUNNER_MODULE}" \
      --repo-root "${REPO_ROOT}" \
      --duration-seconds "${DURATION_SECONDS}" \
      --heartbeat-interval-seconds 5 \
      --exit-when-all-tracks-terminal \
      --master-dir "${RUNTIME_ROOT}" \
      --master-log "${MASTER_LOG_PATH}" \
      --master-pid-path "${MASTER_PID_PATH}" \
      --label "${LABEL}" \
      --implementation-track "lpr|scripts/ops/agent_supervisor/implementation_supervisor_entry.py|${STATE_ROOT}|lpr" \
      --implementation-supervisor-lanes-per-track "${LANE_COUNT}" \
      --implementation-supervisor-strict-task-sharding \
      --common-arg=--todo-path \
      --common-arg="${TODO_PATH}" \
      --common-arg=--task-prefix \
      --common-arg=LPR- \
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
      --common-arg="${VALIDATOR_PATH}" \
      --common-arg=--implementation-protected-path \
      --common-arg="${LAUNCHER_PATH}" \
      --common-arg=--no-objective-task-janitor \
      --common-arg=--no-objective-goal-refinement \
      --common-arg=--no-objective-goal-migration \
      --common-arg=--log-level \
      --common-arg=INFO \
      --detach
  )
  capture_master_identity "${run_id}"
  record_launch_receipt "${run_id}"
}

start() {
  local state_json=""
  local state=""
  local preflight_json=""
  local drained=""
  local run_id=""
  local attempt=""
  state_json="$(master_state_json)"
  state="$(master_state_field status <<<"${state_json}")"
  case "${state}" in
    owned)
      echo "Tactician-Hammer logic-repair supervisor is already running"
      status
      return 0
      ;;
    foreign)
      echo "refusing to start over an unowned live PID: ${state_json}" >&2
      return 2
      ;;
    stale|invalid)
      cleanup_dead_master_markers
      ;;
  esac
  doctor
  require_committed_clean_control_plane
  require_bootstrap_completed
  preflight_json="$(preflight)"
  echo "${preflight_json}"
  drained="$(master_state_field drained <<<"${preflight_json##*$'\n'}")"
  if [[ "${drained}" == "True" || "${drained}" == "true" ]]; then
    echo "board is already drained; no supervisor processes were started"
    return 0
  fi
  run_id="$("${PYTHON_BIN}" -c 'import uuid; print(uuid.uuid4().hex)')"
  launch_master "${run_id}"
  for attempt in {1..90}; do
    if status >/dev/null 2>&1; then
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
  local master_json=""
  local master_state=""
  master_json="$(master_state_json)"
  master_state="$(master_state_field status <<<"${master_json}")"
  case "${master_state}" in
    owned)
      echo "master: running pid=$(master_state_field pid <<<"${master_json}")"
      ;;
    foreign)
      echo "master: foreign ${master_json}"
      return 2
      ;;
    *)
      echo "master: stopped state=${master_state}"
      ;;
  esac
  "${PYTHON_BIN}" - "${STATE_ROOT}" "${LANE_COUNT}" "${master_state}" <<'PY'
import json
import os
import pathlib
import sys
import time

root = pathlib.Path(sys.argv[1])
lane_count = int(sys.argv[2])
master_state = sys.argv[3]
failed = False
for lane in range(lane_count):
    state_dir = root / f"lane-{lane}"
    prefix = f"lpr_lane_{lane}"
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
                    stat = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
                    if stat[stat.rfind(")") + 2:].split()[0] == "Z":
                        alive = False
                except (OSError, IndexError):
                    alive = False
            reported = str(supervisor.get("status") or "unknown")
            effective = reported if alive else "stopped"
            values.update({
                "supervisor": effective,
                "reported_supervisor_status": reported,
                "supervisor_pid": pid or None,
                "supervisor_pid_alive": alive,
                "restart_count": supervisor.get("restart_count", 0),
                "supervisor_heartbeat_age_seconds": round(time.time() - supervisor_path.stat().st_mtime, 1),
            })
            if master_state == "owned" and (effective != "running" or not alive):
                failed = True
        except Exception as exc:
            values["supervisor_error"] = str(exc)
            if master_state == "owned":
                failed = True
    elif master_state == "owned":
        failed = True
    if task_path.exists():
        try:
            task = json.loads(task_path.read_text(encoding="utf-8"))
            values.update({
                "task_state": task.get("status", "available"),
                "active_task_id": task.get("active_task_id", ""),
                "active_phase": task.get("active_phase", ""),
                "eligible_ready_count": int(task.get("eligible_ready_count") or 0),
                "blocked_count": int(task.get("blocked_count") or 0),
                "selection_idle_reason": task.get("selection_idle_reason", ""),
                "heartbeat_age_seconds": round(time.time() - task_path.stat().st_mtime, 1),
            })
            if values["blocked_count"]:
                failed = True
        except Exception as exc:
            values["task_state_error"] = str(exc)
            if master_state == "owned":
                failed = True
    print(json.dumps(values, sort_keys=True))
if failed:
    raise SystemExit(1)
PY
}

signal_owned_master() {
  "${PYTHON_BIN}" - \
    "${MASTER_PID_PATH}" \
    "${MASTER_IDENTITY_PATH}" \
    "${REPO_ROOT}" \
    "${RUNNER_MODULE}" \
    "${LABEL}" <<'PY'
import json
import os
import pathlib
import signal
import sys

pid_path = pathlib.Path(sys.argv[1])
identity_path = pathlib.Path(sys.argv[2])
repo = pathlib.Path(sys.argv[3]).resolve()
module, label = sys.argv[4:6]
identity = json.loads(identity_path.read_text(encoding="utf-8"))
pid = int(pid_path.read_text(encoding="ascii").strip())
if pid != identity.get("pid") or pid <= 0:
    raise SystemExit("master PID identity mismatch")
pidfd = os.pidfd_open(pid, 0)
try:
    stat = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    close = stat.rfind(")")
    fields = stat[close + 2:].split()
    if not fields or fields[0] == "Z" or int(fields[19]) != identity.get("start_time_ticks"):
        raise SystemExit("master process was replaced before signal")
    cwd = pathlib.Path(os.readlink(f"/proc/{pid}/cwd")).resolve()
    argv = tuple(
        item.decode("utf-8", "surrogateescape")
        for item in pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        if item
    )
    env = {}
    for item in pathlib.Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        key, separator, value = item.partition(b"=")
        if separator:
            env[key.decode("utf-8", "surrogateescape")] = value.decode("utf-8", "surrogateescape")
    def pair(flag, value):
        return any(argv[index:index + 2] == (flag, value) for index in range(len(argv) - 1))
    if cwd != repo or not pair("-m", module) or not pair("--label", label):
        raise SystemExit("refusing to signal unowned master command")
    if not pair("--repo-root", str(repo)) or not pair("--master-pid-path", str(pid_path)):
        raise SystemExit("refusing to signal master with mismatched path binding")
    if env.get("IPFS_ACCELERATE_LPR_RUN_ID") != identity.get("run_id"):
        raise SystemExit("refusing to signal master with mismatched run identity")
    signal.pidfd_send_signal(pidfd, signal.SIGTERM)
finally:
    os.close(pidfd)
print(f"signalled owned master pid={pid}")
PY
}

stop() {
  local state_json=""
  local state=""
  local attempt=""
  state_json="$(master_state_json)"
  state="$(master_state_field status <<<"${state_json}")"
  case "${state}" in
    missing)
      echo "master is already stopped"
      return 0
      ;;
    stale|invalid)
      cleanup_dead_master_markers
      echo "removed stale master runtime markers"
      return 0
      ;;
    foreign)
      echo "refusing to stop an unowned live PID: ${state_json}" >&2
      return 2
      ;;
  esac
  signal_owned_master
  for attempt in {1..30}; do
    state_json="$(master_state_json)"
    state="$(master_state_field status <<<"${state_json}")"
    if [[ "${state}" == "stale" || "${state}" == "missing" ]]; then
      cleanup_dead_master_markers
      echo "master stopped"
      return 0
    fi
    if [[ "${state}" == "foreign" ]]; then
      echo "master PID identity changed during stop; no further signal sent" >&2
      return 1
    fi
    sleep 1
  done
  echo "master did not stop within 30 seconds; no forced termination was attempted" >&2
  return 1
}

restart() {
  stop
  start
}

case "${1:-}" in
  doctor)
    doctor
    ;;
  preflight)
    preflight
    ;;
  start)
    start
    ;;
  status)
    status
    ;;
  restart)
    restart
    ;;
  stop)
    stop
    ;;
  *)
    echo "usage: $0 {doctor|preflight|start|status|restart|stop}" >&2
    exit 2
    ;;
esac
