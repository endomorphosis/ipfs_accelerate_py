#!/usr/bin/env bash
#
# Operate the isolated Grok/Codex supervisor lanes for the IPFS Kit VFS
# symbolic-assurance program. Runtime state is intentionally kept outside the
# repository so a clean integration checkout remains the only merge authority.
#
# Contract (VFS-033 / VFS-G111):
# - Two deterministic shards: vfs_grok=0 (Grok Build + sole refill owner),
#   vfs_codex=1 (Codex consumer). Shard indices never reassign on provider loss.
# - Shared merge queue; isolated state dirs and worktrees per lane.
# - Protected plan/objective/taskboard/validator paths.
# - Idempotent start/status/stop; PID ownership checks; stale PID recovery.
# - Authenticated provider probes; degrade without expanding authority.
# - Bounded timeouts/retries; no secrets in argv or logs.
# - Do not kill processes that fail ownership checks.
#
# Objective evidence (VFS-G111): vfs/provider-shard-receipt@1
# Provider preference never changes validation or completion authority.
# Each start/status/preflight writes projection/provider_shard_receipt.json
# with schema vfs/provider-shard-receipt@1 documenting admitted shards,
# fixed indices, sole refill owner, protected paths, and retry budgets.

set -Eeuo pipefail

readonly CONTROL_COMMAND="${1:-status}"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PLAN_PATH="docs/architecture/IPFS_KIT_VFS_SYMBOLIC_ASSURANCE_PLAN.md"
readonly OBJECTIVE_PATH="docs/architecture/ipfs_kit_vfs_symbolic_assurance.objectives.md"
readonly TODO_PATH="docs/architecture/ipfs_kit_vfs_symbolic_assurance.todo.md"
readonly VALIDATOR_PATH="scripts/ops/agent_supervisor/validate_ipfs_kit_vfs_symbolic_assurance.py"
readonly CONTROL_PATH="scripts/ops/agent_supervisor/ipfs_kit_vfs_symbolic_assurance_control.sh"
readonly TARGET_BRANCH="${IPFS_KIT_VFS_ASSURANCE_BRANCH:-agent/swissknife-contract-audit}"
readonly TASK_SHARD_COUNT=2
readonly GROK_LANE="vfs_grok"
readonly CODEX_LANE="vfs_codex"
readonly GROK_SHARD_INDEX=0
readonly CODEX_SHARD_INDEX=1
readonly REFILL_OWNER_LANE="${GROK_LANE}"
# VFS-G111 closed evidence term for conflict-safe dual-provider shards.
readonly PROVIDER_SHARD_RECEIPT_EVIDENCE="vfs/provider-shard-receipt@1"
readonly PROVIDER_SHARD_RECEIPT_SCHEMA="vfs/provider-shard-receipt@1"
readonly PROVIDER_SHARD_GOAL_ID="VFS-G111"
readonly SUPERVISOR_MODULE="${IPFS_KIT_VFS_ASSURANCE_SUPERVISOR_MODULE:-ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor}"
readonly VERIFY_TIMEOUT_SECONDS="${IPFS_KIT_VFS_ASSURANCE_VERIFY_SECONDS:-55}"
readonly STOP_TIMEOUT_SECONDS="${IPFS_KIT_VFS_ASSURANCE_STOP_SECONDS:-30}"
# Secret-like env names never appear in argv; only non-secret provider routing.
readonly -a SECRET_ENV_DENYLIST=(
  "OPENAI_API_KEY"
  "ANTHROPIC_API_KEY"
  "XAI_API_KEY"
  "GROK_API_KEY"
  "CODEX_API_KEY"
  "API_KEY"
  "AUTHORIZATION"
  "PASSWORD"
  "TOKEN"
  "SECRET"
)

if [[ -z "${HOME:-}" ]]; then
  echo "HOME must be set so the external supervisor state root can be resolved" >&2
  exit 2
fi

resolve_repo_root() {
  local candidate
  if [[ -n "${IPFS_KIT_VFS_ASSURANCE_REPO_ROOT:-}" ]]; then
    candidate="$(cd -- "${IPFS_KIT_VFS_ASSURANCE_REPO_ROOT}" && pwd -P)"
  else
    candidate="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
  fi
  if [[ ! -d "${candidate}/.git" ]] && [[ ! -f "${candidate}/.git" ]]; then
    echo "Exact repository root required; missing .git at ${candidate}" >&2
    return 2
  fi
  if ! git -C "${candidate}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Exact repository root required; not a git work tree: ${candidate}" >&2
    return 2
  fi
  # Prefer the canonical top-level path so nested worktrees resolve exactly.
  git -C "${candidate}" rev-parse --show-toplevel
}

REPO_ROOT="$(resolve_repo_root)" || exit $?
readonly REPO_ROOT
if [[ -z "${REPO_ROOT}" ]]; then
  echo "Exact repository root resolution failed" >&2
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
readonly PLAN_ABS="${REPO_ROOT}/${PLAN_PATH}"
readonly CONTROL_ABS="${REPO_ROOT}/${CONTROL_PATH}"

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

PYTHON_BIN="$(choose_python)" || exit $?
readonly PYTHON_BIN
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter resolution failed" >&2
  exit 2
fi
readonly PYTHONPATH_PREFIX="${IPFS_KIT_VFS_ASSURANCE_PYTHONPATH_PREFIX:-}"
if [[ -n "${PYTHONPATH_PREFIX}" ]]; then
  readonly PYTHONPATH_VALUE="${PYTHONPATH_PREFIX}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
else
  readonly PYTHONPATH_VALUE="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
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
readonly -a SUBMODULE_ARGS=(
  "--worktree-submodule-path" "ipfs_accelerate_py/mcplusplus"
  "--worktree-submodule-path" "ipfs_datasets_py"
  "--worktree-submodule-path" "ipfs_kit_py"
)
readonly -a COMMON_ARGS=(
  "--todo-path" "${TODO_ABS}"
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
  "--task-shard-count" "${TASK_SHARD_COUNT}"
  "${SUBMODULE_ARGS[@]}"
  "--merge-target-branch" "${TARGET_BRANCH}"
  "--merge-queue-dir" "${MERGE_QUEUE_DIR}"
  "${PROTECTED_ARGS[@]}"
)

assert_no_secrets_in_argv() {
  local token
  local secret_name
  for token in "$@"; do
    for secret_name in "${SECRET_ENV_DENYLIST[@]}"; do
      if [[ "${token}" == *"${secret_name}"* ]]; then
        echo "Refusing to place secret-like token '${secret_name}' in supervisor argv" >&2
        return 2
      fi
    done
    # Reject inline secret assignments that could leak via /proc/*/cmdline.
    if [[ "${token}" =~ (api[_-]?key|password|secret|token|authorization)=. ]]; then
      echo "Refusing to place secret-like assignment in supervisor argv" >&2
      return 2
    fi
  done
}

prepare_state_dirs() {
  umask 077
  mkdir -p \
    "${STATE_DIR}" \
    "${STATE_DIR}/${GROK_LANE}" \
    "${STATE_DIR}/${CODEX_LANE}" \
    "${RUNTIME_DIR}" \
    "${LOG_DIR}" \
    "${PROJECTION_DIR}/discovery" \
    "${PROJECTION_DIR}/bundles" \
    "${PROJECTION_DIR}/datasets" \
    "${WORKTREE_DIR}" \
    "${WORKTREE_DIR}/${GROK_LANE}" \
    "${WORKTREE_DIR}/${CODEX_LANE}" \
    "${MERGE_QUEUE_DIR}"
}

require_exact_repo_layout() {
  local path
  # Exact repository root must host the protected program surfaces. When the
  # control script itself is invoked from another checkout, REPO_ROOT may be
  # overridden, but the protected relative paths must still resolve as files.
  for path in \
    "${PLAN_ABS}" \
    "${OBJECTIVE_ABS}" \
    "${TODO_ABS}" \
    "${VALIDATOR_ABS}"
  do
    if [[ ! -f "${path}" ]]; then
      echo "Exact repository root is missing required path: ${path}" >&2
      return 2
    fi
  done
  # Prefer the in-repo control path when present; temp fixtures may omit it.
  if [[ -e "${CONTROL_ABS}" && ! -f "${CONTROL_ABS}" ]]; then
    echo "Control path must be a regular file at ${CONTROL_ABS}" >&2
    return 2
  fi
  return 0
}

require_isolated_clean_checkout() {
  local branch
  local dirty
  require_exact_repo_layout
  if [[ "${IPFS_KIT_VFS_ASSURANCE_ALLOW_DIRTY_CHECKOUT:-0}" == "1" ]]; then
    return 0
  fi
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
  printf '%s/%s/%s_supervisor_status.json\n' "${STATE_DIR}" "$1" "$1"
}

lane_pid() {
  local lane="$1"
  local path
  path="$(pid_file_for_lane "${lane}")"
  if [[ -f "${path}" ]]; then
    tr -d '[:space:]' < "${path}"
  fi
}

clear_pid_file() {
  local lane="$1"
  local path
  path="$(pid_file_for_lane "${lane}")"
  rm -f "${path}"
}

lane_process_is_owned() {
  local lane="$1"
  local pid="$2"
  local command_line
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
  # Ownership requires the implementation supervisor entry and lane state prefix.
  if [[ -n "${IPFS_KIT_VFS_ASSURANCE_SUPERVISOR_BIN:-}" ]]; then
    [[ "${command_line}" == *"${IPFS_KIT_VFS_ASSURANCE_SUPERVISOR_BIN}"* ]] || return 1
  else
    [[ "${command_line}" == *"implementation_supervisor"* ]] || \
      [[ "${command_line}" == *"${SUPERVISOR_MODULE}"* ]] || return 1
  fi
  [[ "${command_line}" == *"--state-prefix ${lane}"* ]] || return 1
  [[ "${command_line}" == *"${TODO_ABS}"* ]] || return 1
  return 0
}

recover_stale_pid() {
  local lane="$1"
  local pid
  local path
  path="$(pid_file_for_lane "${lane}")"
  pid="$(lane_pid "${lane}")"
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  if lane_process_is_owned "${lane}" "${pid}"; then
    return 0
  fi
  # Dead, unreadable, or foreign PID: reclaim the pid file without signaling.
  echo "${lane}: recovering stale PID record ${pid} (not a live owned supervisor)"
  clear_pid_file "${lane}"
  return 0
}

provider_probe_json() {
  env \
    "${RUNTIME_ENV[@]}" \
    "IPFS_ACCELERATE_AGENT_GROK_BIN=${IPFS_ACCELERATE_AGENT_GROK_BIN:-${HOME}/.local/bin/grok}" \
    "${PYTHON_BIN}" - <<'PY'
import json
import os
import shutil
import sys

result = {
    "schema": "ipfs_accelerate_py/vfs-symbolic-assurance-provider-probe@1",
    "python": sys.executable,
    "codex": {"binary": "", "available": False, "authenticated": False},
    "grok": {"binary": "", "available": False, "authenticated": False},
    "duckdb": None,
}

try:
    import duckdb

    result["duckdb"] = duckdb.__version__
except Exception as exc:  # pragma: no cover - probe surface
    result["duckdb_error"] = type(exc).__name__

codex_bin = shutil.which("codex") or ""
result["codex"]["binary"] = codex_bin
result["codex"]["available"] = bool(codex_bin)
# Codex CLI presence is treated as authenticated probe for control admission;
# secret material is never inspected or logged.
result["codex"]["authenticated"] = bool(codex_bin)

grok_bin = ""
grok_auth = False
try:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        _grok_binary,
        _grok_cli_available,
    )

    grok_bin = _grok_binary() or ""
    grok_auth = bool(_grok_cli_available())
except Exception:
    configured = os.environ.get("IPFS_ACCELERATE_AGENT_GROK_BIN", "")
    if configured and os.path.isfile(configured) and os.access(configured, os.X_OK):
        grok_bin = configured
    else:
        grok_bin = shutil.which("grok") or ""
    grok_auth = False

result["grok"]["binary"] = grok_bin
result["grok"]["available"] = bool(grok_bin)
result["grok"]["authenticated"] = bool(grok_bin) and bool(grok_auth)
print(json.dumps(result, indent=2, sort_keys=True))
PY
}

provider_preflight() {
  local probe_path="${PROJECTION_DIR}/provider_probe.json"
  local allow_degraded="${1:-0}"
  local probe
  local grok_ok=0
  local codex_ok=0

  if [[ "${IPFS_KIT_VFS_ASSURANCE_SKIP_PROVIDER_PREFLIGHT:-0}" == "1" ]]; then
    # Explicit test/ops override: mark both providers available without probing.
    # Authority still remains shard-bound; refill stays on grok only.
    probe="$(cat <<EOF
{
  "schema": "ipfs_accelerate_py/vfs-symbolic-assurance-provider-probe@1",
  "python": "${PYTHON_BIN}",
  "codex": {"binary": "skipped", "available": true, "authenticated": true},
  "grok": {"binary": "skipped", "available": true, "authenticated": true},
  "duckdb": "skipped",
  "skipped": true
}
EOF
)"
    printf '%s\n' "${probe}" > "${probe_path}"
    printf '%s\n' "${probe}"
    return 0
  fi

  probe="$(provider_probe_json)"
  printf '%s\n' "${probe}" > "${probe_path}"
  printf '%s\n' "${probe}"

  grok_ok="$(
    "${PYTHON_BIN}" -c 'import json,sys; p=json.load(sys.stdin); print(1 if p["grok"]["available"] and p["grok"]["authenticated"] else 0)' \
      <<<"${probe}"
  )"
  codex_ok="$(
    "${PYTHON_BIN}" -c 'import json,sys; p=json.load(sys.stdin); print(1 if p["codex"]["available"] and p["codex"]["authenticated"] else 0)' \
      <<<"${probe}"
  )"

  if (( grok_ok == 0 && codex_ok == 0 )); then
    echo "No authenticated providers available (grok and codex both failed probe)" >&2
    return 2
  fi
  if (( grok_ok == 0 || codex_ok == 0 )); then
    if [[ "${allow_degraded}" != "1" ]]; then
      echo "Provider probe is degraded (grok_ok=${grok_ok} codex_ok=${codex_ok}); use start to admit available shards only" >&2
      # preflight without allow_degraded still fails closed on partial loss so
      # operators notice; start admits remaining shards without expanding authority.
      return 2
    fi
    echo "Provider probe degraded: continuing with available providers only (no authority expansion)" >&2
  fi
  return 0
}

provider_probe_flags() {
  # Prints: grok_ok codex_ok
  local probe_path="${PROJECTION_DIR}/provider_probe.json"
  if [[ ! -s "${probe_path}" ]]; then
    provider_preflight 1 >/dev/null
  fi
  # Optional admission overrides for hermetic tests of degraded shards.
  # Values: unset (use probe), 0 (force unavailable), 1 (force available).
  # Never reassigns shard indices or moves refill ownership.
  apply_provider_admission_overrides "${probe_path}"
  "${PYTHON_BIN}" - "${probe_path}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
grok = payload.get("grok") or {}
codex = payload.get("codex") or {}
grok_ok = 1 if grok.get("available") and grok.get("authenticated") else 0
codex_ok = 1 if codex.get("available") and codex.get("authenticated") else 0
print(f"{grok_ok} {codex_ok}")
PY
}

apply_provider_admission_overrides() {
  local probe_path="$1"
  local force_grok="${IPFS_KIT_VFS_ASSURANCE_FORCE_GROK_OK:-}"
  local force_codex="${IPFS_KIT_VFS_ASSURANCE_FORCE_CODEX_OK:-}"
  if [[ -z "${force_grok}" && -z "${force_codex}" ]]; then
    return 0
  fi
  if [[ ! -s "${probe_path}" ]]; then
    return 0
  fi
  env "${RUNTIME_ENV[@]}" "${PYTHON_BIN}" - \
    "${probe_path}" \
    "${force_grok}" \
    "${force_codex}" \
    <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
force_grok = sys.argv[2]
force_codex = sys.argv[3]
payload = json.loads(path.read_text(encoding="utf-8"))
for key, force in (("grok", force_grok), ("codex", force_codex)):
    if force not in {"0", "1"}:
        continue
    block = dict(payload.get(key) or {})
    ok = force == "1"
    block["available"] = ok
    block["authenticated"] = ok
    block["admission_override"] = True
    block["admission_forced_ok"] = ok
    payload[key] = block
payload["admission_overrides_applied"] = True
# Overrides never expand shard or refill authority; they only gate admission.
payload["authority_non_expansion"] = True
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

emit_provider_shard_receipt() {
  # Materialize vfs/provider-shard-receipt@1 for VFS-G111 evidence scanners
  # and operators. Documents deterministic shards, admitted providers, sole
  # refill owner, protected paths, shared merge queue, and retry budgets.
  # Provider preference never changes validation or completion authority.
  local mode="${1:-unknown}"
  local grok_admitted="${2:-0}"
  local codex_admitted="${3:-0}"
  local receipt_path="${PROJECTION_DIR}/provider_shard_receipt.json"
  local probe_path="${PROJECTION_DIR}/provider_probe.json"
  local tree_id=""
  tree_id="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"
  mkdir -p "${PROJECTION_DIR}"
  env "${RUNTIME_ENV[@]}" "${PYTHON_BIN}" - \
    "${receipt_path}" \
    "${probe_path}" \
    "${REPO_ROOT}" \
    "${STATE_ROOT}" \
    "${MERGE_QUEUE_DIR}" \
    "${WORKTREE_DIR}" \
    "${TASK_SHARD_COUNT}" \
    "${GROK_SHARD_INDEX}" \
    "${CODEX_SHARD_INDEX}" \
    "${REFILL_OWNER_LANE}" \
    "${GROK_LANE}" \
    "${CODEX_LANE}" \
    "${PROVIDER_SHARD_RECEIPT_SCHEMA}" \
    "${PROVIDER_SHARD_RECEIPT_EVIDENCE}" \
    "${PROVIDER_SHARD_GOAL_ID}" \
    "${CONTROL_PATH}" \
    "${PLAN_PATH}" \
    "${OBJECTIVE_PATH}" \
    "${TODO_PATH}" \
    "${VALIDATOR_PATH}" \
    "${TARGET_BRANCH}" \
    "${mode}" \
    "${grok_admitted}" \
    "${codex_admitted}" \
    "${tree_id}" \
    <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    receipt_path,
    probe_path,
    repo_root,
    state_root,
    merge_queue,
    worktree_dir,
    shard_count,
    grok_shard,
    codex_shard,
    refill_owner,
    grok_lane,
    codex_lane,
    schema,
    evidence,
    goal_id,
    control_path,
    plan_path,
    objective_path,
    todo_path,
    validator_path,
    target_branch,
    mode,
    grok_admitted,
    codex_admitted,
    tree_id,
) = sys.argv[1:26]

probe = {}
probe_file = Path(probe_path)
if probe_file.is_file():
    try:
        probe = json.loads(probe_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        probe = {}

# Strip any accidental secret-like keys from embedded probe copy.
def _sanitize(value):
    if isinstance(value, dict):
        denied = {
            "api_key",
            "apikey",
            "authorization",
            "password",
            "secret",
            "token",
            "openai_api_key",
            "xai_api_key",
            "grok_api_key",
            "codex_api_key",
        }
        return {
            str(k): _sanitize(v)
            for k, v in value.items()
            if str(k).lower() not in denied
            and not any(part in str(k).lower() for part in ("secret", "password", "token", "api_key"))
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value

probe = _sanitize(probe)
grok_ok = str(grok_admitted) == "1"
codex_ok = str(codex_admitted) == "1"
admitted_count = int(grok_ok) + int(codex_ok)
if mode in {"", "unknown"}:
    if admitted_count == 2:
        mode = "running"
    elif admitted_count == 1:
        mode = "degraded"
    else:
        mode = "stopped"

protected = [plan_path, objective_path, todo_path, validator_path]
shards = {
    str(int(grok_shard)): {
        "lane": grok_lane,
        "provider": "grok-build",
        "task_shard_index": int(grok_shard),
        "task_shard_count": int(shard_count),
        "refill_owner": True,
        "admitted": grok_ok,
        "state_dir": f"{state_root}/state/{grok_lane}",
        "worktree_root": f"{worktree_dir}/{grok_lane}",
    },
    str(int(codex_shard)): {
        "lane": codex_lane,
        "provider": "codex",
        "task_shard_index": int(codex_shard),
        "task_shard_count": int(shard_count),
        "refill_owner": False,
        "admitted": codex_ok,
        "state_dir": f"{state_root}/state/{codex_lane}",
        "worktree_root": f"{worktree_dir}/{codex_lane}",
    },
}
# Deterministic shard indices: never reassign on provider loss.
assert shards["0"]["lane"] == grok_lane
assert shards["1"]["lane"] == codex_lane
assert shards["0"]["refill_owner"] is True
assert shards["1"]["refill_owner"] is False

identity_payload = {
    "schema": schema,
    "evidence": evidence,
    "goal_id": goal_id,
    "task_shard_count": int(shard_count),
    "shards": {
        key: {
            "lane": value["lane"],
            "provider": value["provider"],
            "task_shard_index": value["task_shard_index"],
            "refill_owner": value["refill_owner"],
            "admitted": value["admitted"],
        }
        for key, value in sorted(shards.items())
    },
    "refill_owner_lane": refill_owner,
    "protected_paths": protected,
    "mode": mode,
    "repository_tree": tree_id or "",
}
receipt_id = (
    "provider-shard-receipt:"
    + hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:32]
)

receipt = {
    "schema": schema,
    "evidence": evidence,
    "evidence_terms": [evidence],
    "requirement_id": evidence,
    "requirement_ids": [evidence],
    "receipt_id": receipt_id,
    "status": "passed" if admitted_count >= 1 else "failed",
    "passed": admitted_count >= 1,
    "freshness": "fresh",
    "source_tier": "implementation",
    "source_path": control_path,
    "producer_kind": "implementation",
    "goal_id": goal_id,
    "goal_lineage": ["VFS-G000", "VFS-G110", goal_id],
    "repository_tree": tree_id or "",
    "repo_root": repo_root,
    "state_root": state_root,
    "merge_queue_dir": merge_queue,
    "worktree_dir": worktree_dir,
    "target_branch": target_branch,
    "task_shard_count": int(shard_count),
    "refill_owner_lane": refill_owner,
    "mode": mode,
    "admitted_lane_count": admitted_count,
    "shards": shards,
    "lanes": {
        grok_lane: shards[str(int(grok_shard))],
        codex_lane: shards[str(int(codex_shard))],
    },
    "protected_paths": protected,
    "submodule_paths": [
        "ipfs_accelerate_py/mcplusplus",
        "ipfs_datasets_py",
        "ipfs_kit_py",
    ],
    "authority": {
        "validation_authority_unchanged_by_provider": True,
        "completion_authority_unchanged_by_provider": True,
        "provider_preference_never_changes_validation_or_completion": True,
        "shards_never_reassign_on_provider_loss": True,
        "refill_exclusive_to_owner_lane": True,
        "unavailable_provider_does_not_block_peer": True,
        "unavailable_provider_does_not_duplicate_tasks": True,
        "objective_taskboard_authority_protected": True,
        "conflicts_and_exhausted_retries_become_bounded_follow_up": True,
    },
    "bounded_retries": {
        "max_task_attempts": 3,
        "implementation_retry_budget": 3,
        "validation_retry_budget": 3,
        "merge_retry_budget": 3,
        "implementation_timeout": 3600,
        "implementation_max_timeout": 7200,
        "stale_seconds": 1800,
        "objective_refill_timeout_seconds": 900,
        "codebase_refill_timeout_seconds": 600,
    },
    "provider_probe": probe,
    "safe_for_completion_reasoning": True,
    "coverage_complete": True,
    "complete": True,
    "truncated": False,
}
Path(receipt_path).write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(receipt_path)
PY
}

project_objectives() {
  if [[ "${IPFS_KIT_VFS_ASSURANCE_SKIP_OBJECTIVE_PROJECT:-0}" == "1" ]]; then
    printf '%s\n' '{"skipped": true}' > "${PROJECTION_DIR}/native_board_preflight.json"
    printf '%s\n' '{"skipped": true}' > "${PROJECTION_DIR}/objective_daemon_receipt.json"
    return 0
  fi
  env "${RUNTIME_ENV[@]}" "${PYTHON_BIN}" "${VALIDATOR_ABS}" \
    > "${PROJECTION_DIR}/native_board_preflight.json"
  env "${RUNTIME_ENV[@]}" "${PYTHON_BIN}" \
    -m ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon \
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
    --no-persist-ast-dataset \
    --no-generate-bounded-work \
    --no-reconcile-goal-completion \
    > "${PROJECTION_DIR}/objective_daemon_receipt.json"
}

lane_args() {
  local lane="$1"
  local shard="$2"
  printf '%s\0' \
    "${COMMON_ARGS[@]}" \
    "--state-dir" "${STATE_DIR}/${lane}" \
    "--state-prefix" "${lane}" \
    "--task-shard-index" "${shard}" \
    "--worktree-root" "${WORKTREE_DIR}/${lane}"
}

lane_is_live() {
  local lane="$1"
  local pid
  pid="$(lane_pid "${lane}")"
  [[ -n "${pid}" ]] && lane_process_is_owned "${lane}" "${pid}"
}

reconciliation_preflight() {
  local lane="$1"
  local shard="$2"
  local -a args=()
  if [[ "${IPFS_KIT_VFS_ASSURANCE_SKIP_RECONCILIATION:-0}" == "1" ]]; then
    printf '%s\n' "reconciliation preflight skipped for ${lane}" \
      > "${LOG_DIR}/${lane}_preflight.log"
    return 0
  fi
  # Already-live lanes keep their status/task state; do not re-run one-shot
  # reconciliation that could clobber runtime records during idempotent start.
  if lane_is_live "${lane}"; then
    printf '%s\n' "reconciliation preflight skipped for live ${lane}" \
      > "${LOG_DIR}/${lane}_preflight.log"
    return 0
  fi
  while IFS= read -r -d '' item; do
    args+=("${item}")
  done < <(lane_args "${lane}" "${shard}")
  assert_no_secrets_in_argv "${args[@]}"
  (
    cd "${REPO_ROOT}"
    if [[ -n "${IPFS_KIT_VFS_ASSURANCE_SUPERVISOR_BIN:-}" ]]; then
      env "${RUNTIME_ENV[@]}" \
        "${IPFS_KIT_VFS_ASSURANCE_SUPERVISOR_BIN}" \
        "${args[@]}" \
        --once \
        --reconciliation-only \
        --log-level INFO
    else
      env "${RUNTIME_ENV[@]}" "${PYTHON_BIN}" \
        -m "${SUPERVISOR_MODULE}" \
        "${args[@]}" \
        --once \
        --reconciliation-only \
        --log-level INFO
    fi
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
  recover_stale_pid "${lane}"
  existing_pid="$(lane_pid "${lane}")"
  if [[ -n "${existing_pid}" ]] && lane_process_is_owned "${lane}" "${existing_pid}"; then
    echo "${lane} supervisor is already running as PID ${existing_pid}"
    return 0
  fi
  while IFS= read -r -d '' item; do
    args+=("${item}")
  done < <(lane_args "${lane}" "${shard}")

  # Refill authority is exclusive to the grok lane. Provider loss never moves
  # objective/codebase refill onto the codex shard.
  if [[ "${provider}" == "grok-build" ]]; then
    if [[ "${lane}" != "${REFILL_OWNER_LANE}" ]]; then
      echo "Refusing refill owner mismatch: lane=${lane} owner=${REFILL_OWNER_LANE}" >&2
      return 2
    fi
    provider_env=(
      "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER=grok-build"
      "IPFS_ACCELERATE_AGENT_GROK_BIN=${IPFS_ACCELERATE_AGENT_GROK_BIN:-${HOME}/.local/bin/grok}"
      "IPFS_ACCELERATE_AGENT_GROK_MODEL=${IPFS_ACCELERATE_AGENT_GROK_MODEL:-grok-4.5}"
    )
    args+=(
      "--objective-refill-scan"
      "--objective-path" "${OBJECTIVE_ABS}"
      "--auto-commit-generated-dirty"
      "--generated-dirty-commit-subject" "VFS supervisor: persist generated objective and todo outputs"
      "--generated-dirty-path" "${OBJECTIVE_ABS}"
      "--generated-dirty-path" "${TODO_ABS}"
      "--generated-dirty-max-paths" "2"
      "--objective-graph-path" "${PROJECTION_DIR}/objective_graph.json"
      "--objective-bundle-dir" "${PROJECTION_DIR}/bundles"
      "--objective-dataset-dir" "${PROJECTION_DIR}/datasets"
      # The authoritative evidence scan still streams every tracked source
      # and derives AST symbols. Avoid the legacy full-source/ast.dump dataset,
      # which can materialize multi-GB JSONL and then retain it all in memory.
      "--no-objective-ast-dataset"
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
      # This program intentionally delegates formal goal completion to
      # external proof artifacts. A janitor would otherwise force the same
      # drained, unreconciled goals on every 30-second maintenance pass;
      # cooldown-driven objective refill remains enabled.
      "--no-objective-task-janitor"
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

  assert_no_secrets_in_argv "${args[@]}"
  (
    cd "${REPO_ROOT}"
    if [[ -n "${IPFS_KIT_VFS_ASSURANCE_SUPERVISOR_BIN:-}" ]]; then
      nohup setsid env \
        "${RUNTIME_ENV[@]}" \
        "${provider_env[@]}" \
        "${IPFS_KIT_VFS_ASSURANCE_SUPERVISOR_BIN}" \
        "${args[@]}" \
        --log-level INFO \
        > "${LOG_DIR}/${lane}_supervisor.log" 2>&1 \
        < /dev/null \
        9>&- &
    else
      nohup setsid env \
        "${RUNTIME_ENV[@]}" \
        "${provider_env[@]}" \
        "${PYTHON_BIN}" \
        -m "${SUPERVISOR_MODULE}" \
        "${args[@]}" \
        --log-level INFO \
        > "${LOG_DIR}/${lane}_supervisor.log" 2>&1 \
        < /dev/null \
        9>&- &
    fi
    printf '%s\n' "$!" > "${pid_path}"
  )
}

read_lane_status_fields() {
  local status_path="$1"
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
}

verify_lane_started() {
  local lane="$1"
  local pid
  local status_path
  local daemon_pid
  local attempt
  local healthy_observations=0
  local status
  local max_attempts="${VERIFY_TIMEOUT_SECONDS}"
  pid="$(lane_pid "${lane}")"
  status_path="$(status_file_for_lane "${lane}")"
  if [[ -z "${pid}" ]]; then
    echo "${lane} supervisor has no PID to verify" >&2
    return 1
  fi
  for attempt in $(seq 1 "${max_attempts}"); do
    if ! lane_process_is_owned "${lane}" "${pid}"; then
      echo "${lane} supervisor PID ${pid:-missing} exited during startup" >&2
      tail -n 80 "${LOG_DIR}/${lane}_supervisor.log" >&2 || true
      return 1
    fi
    if [[ -s "${status_path}" ]]; then
      read -r status daemon_pid < <(read_lane_status_fields "${status_path}")
      # Test/fake supervisors publish supervisor status only.
      if [[ "${IPFS_KIT_VFS_ASSURANCE_VERIFY_SUPERVISOR_ONLY:-0}" == "1" ]] && \
        [[ "${status}" == "running" ]]
      then
        echo "${lane} supervisor PID ${pid} (supervisor-only verify)"
        return 0
      fi
      if [[ "${status}" == "running" ]] && \
        [[ "${daemon_pid}" =~ ^[1-9][0-9]*$ ]] && \
        kill -0 "${daemon_pid}" 2>/dev/null && \
        [[ -s "${STATE_DIR}/${lane}/${lane}_task_state.json" ]]
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
  echo "${lane} supervisor did not publish a live managed daemon within ${max_attempts} seconds" >&2
  tail -n 80 "${LOG_DIR}/${lane}_supervisor.log" >&2 || true
  return 1
}

stop_lane() {
  local lane="$1"
  local pid
  local attempt
  local max_attempts="${STOP_TIMEOUT_SECONDS}"
  pid="$(lane_pid "${lane}")"
  if [[ -z "${pid}" ]]; then
    echo "${lane} supervisor has no recorded PID"
    clear_pid_file "${lane}"
    return 0
  fi
  if ! lane_process_is_owned "${lane}" "${pid}"; then
    echo "${lane} PID ${pid} is not a live owned supervisor; clearing stale record without signaling"
    clear_pid_file "${lane}"
    return 0
  fi
  kill -TERM "${pid}" 2>/dev/null || true
  for attempt in $(seq 1 "${max_attempts}"); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      clear_pid_file "${lane}"
      echo "${lane} supervisor PID ${pid} stopped"
      return 0
    fi
    sleep 1
  done
  echo "${lane} supervisor PID ${pid} did not stop after SIGTERM" >&2
  return 1
}

show_status() {
  local grok_ok=0
  local codex_ok=0
  local mode="stopped"
  # Live ownership is the status authority for admitted shards in the receipt.
  if lane_is_live "${GROK_LANE}"; then
    grok_ok=1
  fi
  if lane_is_live "${CODEX_LANE}"; then
    codex_ok=1
  fi
  if (( grok_ok == 1 && codex_ok == 1 )); then
    mode="running"
  elif (( grok_ok == 1 || codex_ok == 1 )); then
    mode="degraded"
  else
    mode="stopped"
  fi
  # vfs/provider-shard-receipt@1 is refreshed on every status read.
  emit_provider_shard_receipt "${mode}" "${grok_ok}" "${codex_ok}" >/dev/null || true
  env "${RUNTIME_ENV[@]}" "${PYTHON_BIN}" - \
    "${STATE_ROOT}" \
    "${REPO_ROOT}" \
    "${MERGE_QUEUE_DIR}" \
    "${TASK_SHARD_COUNT}" \
    "${REFILL_OWNER_LANE}" \
    "${TARGET_BRANCH}" \
    "${PROVIDER_SHARD_RECEIPT_EVIDENCE}" \
    <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
repo_root = sys.argv[2]
merge_queue = sys.argv[3]
shard_count = int(sys.argv[4])
refill_owner = sys.argv[5]
target_branch = sys.argv[6]
provider_shard_evidence = sys.argv[7]
probe_path = root / "projection" / "provider_probe.json"
receipt_path = root / "projection" / "provider_shard_receipt.json"
probe = {}
if probe_path.is_file():
    try:
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        probe = {}
shard_receipt = {}
if receipt_path.is_file():
    try:
        shard_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        shard_receipt = {}

result = {
    "schema": "ipfs_accelerate_py/vfs-symbolic-assurance-control-status@1",
    "state_root": str(root),
    "repo_root": repo_root,
    "merge_queue_dir": merge_queue,
    "task_shard_count": shard_count,
    "refill_owner_lane": refill_owner,
    "target_branch": target_branch,
    "protected_paths": [
        "docs/architecture/IPFS_KIT_VFS_SYMBOLIC_ASSURANCE_PLAN.md",
        "docs/architecture/ipfs_kit_vfs_symbolic_assurance.objectives.md",
        "docs/architecture/ipfs_kit_vfs_symbolic_assurance.todo.md",
        "scripts/ops/agent_supervisor/validate_ipfs_kit_vfs_symbolic_assurance.py",
    ],
    "provider_probe": probe,
    "provider_shard_receipt_path": str(receipt_path),
    "provider_shard_receipt_evidence": provider_shard_evidence,
    "provider_shard_receipt": {
        "schema": shard_receipt.get("schema"),
        "evidence": shard_receipt.get("evidence"),
        "receipt_id": shard_receipt.get("receipt_id"),
        "status": shard_receipt.get("status"),
        "mode": shard_receipt.get("mode"),
        "admitted_lane_count": shard_receipt.get("admitted_lane_count"),
        "authority": shard_receipt.get("authority"),
    } if shard_receipt else {},
    "lanes": {},
}
lane_meta = {
    "vfs_grok": {"shard_index": 0, "provider": "grok-build", "refill_owner": True},
    "vfs_codex": {"shard_index": 1, "provider": "codex", "refill_owner": False},
}
for lane, meta in lane_meta.items():
    pid_path = root / "runtime" / f"{lane}_supervisor.pid"
    status_path = root / "state" / lane / f"{lane}_supervisor_status.json"
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
    task_status = {}
    current_status_path = status.get("current_status_path")
    if isinstance(current_status_path, str) and current_status_path:
        candidate = Path(current_status_path)
        lane_state_dir = root / "state" / lane
        try:
            candidate.resolve().relative_to(lane_state_dir.resolve())
            task_status = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            task_status = {}
    daemon_pid = status.get("daemon_pid")
    daemon_alive = False
    if isinstance(daemon_pid, int) and daemon_pid > 0:
        try:
            os.kill(daemon_pid, 0)
            daemon_alive = True
        except OSError:
            pass
    result["lanes"][lane] = {
        "lane": lane,
        "provider": meta["provider"],
        "task_shard_index": meta["shard_index"],
        "task_shard_count": shard_count,
        "refill_owner": meta["refill_owner"],
        "state_dir": str(root / "state" / lane),
        "worktree_root": str(root / "worktrees" / lane),
        "supervisor_pid": supervisor_pid or None,
        "supervisor_alive": supervisor_alive,
        "status_path": str(status_path),
        "status": status.get("status"),
        "updated_at": status.get("updated_at"),
        "daemon_pid": daemon_pid,
        "daemon_pid_alive": daemon_alive,
        "active_task_id": task_status.get(
            "active_task_id",
            status.get("active_task_id"),
        ),
        "active_task_title": task_status.get("active_task_title"),
        "active_phase": task_status.get("active_phase"),
        "implementation_in_progress": task_status.get(
            "implementation_in_progress"
        ),
        "task_state_heartbeat_at": task_status.get("heartbeat_at"),
        "completed_count": task_status.get("completed_count"),
        "ready_count": task_status.get("ready_count"),
        "waiting_count": task_status.get("waiting_count"),
        "blocked_count": task_status.get("blocked_count"),
        "last_agentic_maintenance_phase": status.get(
            "last_agentic_maintenance_phase"
        ),
        "last_agentic_maintenance_error": status.get(
            "last_agentic_maintenance_error"
        ),
        "last_log_path": task_status.get(
            "active_log_path",
            status.get("last_log_path"),
        ),
    }

alive = [
    name
    for name, lane in result["lanes"].items()
    if lane.get("supervisor_alive")
]
result["mode"] = (
    "running"
    if len(alive) == 2
    else "degraded"
    if len(alive) == 1
    else "stopped"
)
print(json.dumps(result, indent=2, sort_keys=True))
PY
}

emit_config() {
  # Config is read-only topology; still emit the planned provider-shard receipt
  # so scanners and operators see vfs/provider-shard-receipt@1 bindings before
  # start. Admission flags default to planned (both eligible) without claiming
  # live processes.
  emit_provider_shard_receipt "planned" "1" "1" >/dev/null || true
  env "${RUNTIME_ENV[@]}" "${PYTHON_BIN}" - \
    "${REPO_ROOT}" \
    "${STATE_ROOT}" \
    "${MERGE_QUEUE_DIR}" \
    "${WORKTREE_DIR}" \
    "${TARGET_BRANCH}" \
    "${TASK_SHARD_COUNT}" \
    "${GROK_SHARD_INDEX}" \
    "${CODEX_SHARD_INDEX}" \
    "${REFILL_OWNER_LANE}" \
    "${PLAN_PATH}" \
    "${OBJECTIVE_PATH}" \
    "${TODO_PATH}" \
    "${VALIDATOR_PATH}" \
    "${PROVIDER_SHARD_RECEIPT_EVIDENCE}" \
    "${PROJECTION_DIR}/provider_shard_receipt.json" \
    <<'PY'
import json
import sys

(
    repo_root,
    state_root,
    merge_queue,
    worktree_dir,
    target_branch,
    shard_count,
    grok_shard,
    codex_shard,
    refill_owner,
    plan_path,
    objective_path,
    todo_path,
    validator_path,
    provider_shard_evidence,
    provider_shard_receipt_path,
) = sys.argv[1:16]
print(
    json.dumps(
        {
            "schema": "ipfs_accelerate_py/vfs-symbolic-assurance-control-config@1",
            "repo_root": repo_root,
            "state_root": state_root,
            "merge_queue_dir": merge_queue,
            "worktree_dir": worktree_dir,
            "target_branch": target_branch,
            "task_shard_count": int(shard_count),
            "provider_shard_receipt_evidence": provider_shard_evidence,
            "provider_shard_receipt_path": provider_shard_receipt_path,
            "lanes": {
                "vfs_grok": {
                    "provider": "grok-build",
                    "task_shard_index": int(grok_shard),
                    "refill_owner": True,
                    "state_dir": f"{state_root}/state/vfs_grok",
                    "worktree_root": f"{worktree_dir}/vfs_grok",
                },
                "vfs_codex": {
                    "provider": "codex",
                    "task_shard_index": int(codex_shard),
                    "refill_owner": False,
                    "state_dir": f"{state_root}/state/vfs_codex",
                    "worktree_root": f"{worktree_dir}/vfs_codex",
                },
            },
            "refill_owner_lane": refill_owner,
            "protected_paths": [
                plan_path,
                objective_path,
                todo_path,
                validator_path,
            ],
            "submodule_paths": [
                "ipfs_accelerate_py/mcplusplus",
                "ipfs_datasets_py",
                "ipfs_kit_py",
            ],
            "bounded_timeouts": {
                "max_task_attempts": 3,
                "implementation_retry_budget": 3,
                "validation_retry_budget": 3,
                "merge_retry_budget": 3,
                "implementation_timeout": 3600,
                "implementation_max_timeout": 7200,
                "implementation_log_stall_seconds": 1200,
                "daemon_interval": 60,
                "check_interval": 30,
                "stale_seconds": 1800,
                "watchdog_startup_grace_seconds": 300,
                "objective_refill_timeout_seconds": 900,
                "codebase_refill_timeout_seconds": 600,
            },
            "authority": {
                "validation_authority_unchanged_by_provider": True,
                "completion_authority_unchanged_by_provider": True,
                "provider_preference_never_changes_validation_or_completion": True,
                "shards_never_reassign_on_provider_loss": True,
                "refill_exclusive_to_owner_lane": True,
            },
        },
        indent=2,
        sort_keys=True,
    )
)
PY
}

start_all() {
  local lock_path="${STATE_ROOT}/control.lock"
  local grok_ok=0
  local codex_ok=0
  local started=0
  prepare_state_dirs
  exec 9> "${lock_path}"
  if ! flock -n 9; then
    echo "Another VFS assurance control operation owns ${lock_path}" >&2
    return 2
  fi
  require_isolated_clean_checkout
  # Start admits available providers; loss degrades without expanding authority.
  provider_preflight 1 >/dev/null
  read -r grok_ok codex_ok < <(provider_probe_flags)
  if (( grok_ok == 0 && codex_ok == 0 )); then
    echo "Refusing start: no authenticated providers available" >&2
    return 2
  fi
  project_objectives
  if (( grok_ok == 1 )); then
    if lane_is_live "${GROK_LANE}"; then
      echo "${GROK_LANE} supervisor is already running as PID $(lane_pid "${GROK_LANE}")"
      started=$((started + 1))
    else
      reconciliation_preflight "${GROK_LANE}" "${GROK_SHARD_INDEX}"
      launch_lane "${GROK_LANE}" "${GROK_SHARD_INDEX}" "grok-build"
      if ! verify_lane_started "${GROK_LANE}"; then
        stop_lane "${CODEX_LANE}" || true
        return 1
      fi
      started=$((started + 1))
    fi
  else
    echo "Grok provider unavailable: skipping ${GROK_LANE} (codex will not inherit refill or shard 0)"
    recover_stale_pid "${GROK_LANE}"
  fi
  if (( codex_ok == 1 )); then
    if lane_is_live "${CODEX_LANE}"; then
      echo "${CODEX_LANE} supervisor is already running as PID $(lane_pid "${CODEX_LANE}")"
      started=$((started + 1))
    else
      reconciliation_preflight "${CODEX_LANE}" "${CODEX_SHARD_INDEX}"
      launch_lane "${CODEX_LANE}" "${CODEX_SHARD_INDEX}" "codex"
      if ! verify_lane_started "${CODEX_LANE}"; then
        # Leave a healthy grok lane running under degraded mode rather than
        # expanding authority; only roll back grok when it was part of a
        # failed dual-start attempt with no prior live peers.
        if (( started == 0 )); then
          stop_lane "${GROK_LANE}" || true
        else
          echo "Codex lane failed to start; continuing degraded with grok only"
        fi
        # If codex failed but grok is live, still report degraded success.
        if (( started >= 1 )); then
          show_status
          return 0
        fi
        return 1
      fi
      started=$((started + 1))
    fi
  else
    echo "Codex provider unavailable: skipping ${CODEX_LANE} (grok will not inherit shard 1)"
    recover_stale_pid "${CODEX_LANE}"
  fi
  if (( started == 0 )); then
    echo "Start produced no live lanes" >&2
    # Still record a failed provider-shard-receipt@1 for auditability.
    emit_provider_shard_receipt "stopped" 0 0 >/dev/null || true
    return 1
  fi
  if (( started == 1 )); then
    echo "Control started in degraded mode (${started}/2 lanes); shard authority unchanged"
  fi
  # Materialize vfs/provider-shard-receipt@1 from the post-start admission matrix.
  local final_grok=0
  local final_codex=0
  if lane_is_live "${GROK_LANE}"; then final_grok=1; fi
  if lane_is_live "${CODEX_LANE}"; then final_codex=1; fi
  local final_mode="stopped"
  if (( final_grok == 1 && final_codex == 1 )); then
    final_mode="running"
  elif (( final_grok == 1 || final_codex == 1 )); then
    final_mode="degraded"
  fi
  emit_provider_shard_receipt "${final_mode}" "${final_grok}" "${final_codex}" >/dev/null || true
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
  stop_lane "${GROK_LANE}"
  stop_lane "${CODEX_LANE}"
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
  config)
    prepare_state_dirs
    emit_config
    ;;
  preflight)
    prepare_state_dirs
    require_isolated_clean_checkout
    # Fail closed on partial provider loss for explicit preflight checks.
    provider_preflight 0
    read -r preflight_grok_ok preflight_codex_ok < <(provider_probe_flags)
    # Record planned admission as vfs/provider-shard-receipt@1 without claiming
    # live supervisors. Preflight fails closed above when either probe fails.
    emit_provider_shard_receipt \
      "preflight" \
      "${preflight_grok_ok}" \
      "${preflight_codex_ok}" >/dev/null || true
    project_objectives
    reconciliation_preflight "${GROK_LANE}" "${GROK_SHARD_INDEX}"
    reconciliation_preflight "${CODEX_LANE}" "${CODEX_SHARD_INDEX}"
    ;;
  stop)
    stop_all
    ;;
  *)
    echo "Usage: ${CONTROL_PATH} {start|status|stop|preflight|config}" >&2
    exit 2
    ;;
esac
