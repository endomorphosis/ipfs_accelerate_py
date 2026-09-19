#!/bin/bash
# SAWM exclusive owner + launch, with supervisor heals injected into the
# one extra-gate package. Do not start a second exclusive owner.
# Overlay-first is only for quack-start/ready. Sealed launch keeps the
# board validators on the sealed package; overlay-first launch fails
# "sealed dependency or board validation failed" and SIGKILLs extra-gate.
set -euo pipefail
OVERLAY=${1:?overlay root}
SOURCE_ROOT=${2:?sealed source root}
cd "$SOURCE_ROOT"
SCRIPT=scripts/ops/agent_supervisor/semantic_addressed_world_model.py
# Overlay-first for quack-start so the exclusive owner binds the typed
# command socket. Sealed `launch` below stays pin-free on the sealed package.
LAUNCH=(
  /usr/bin/python3 -P
  "$OVERLAY/ipfs_accelerate_py/agent_supervisor/rescue/sealed_board_supervisor_launch.py"
  --overlay "$OVERLAY"
  --source-root "$SOURCE_ROOT"
  --
  "$SCRIPT"
)
SEALED=(/usr/bin/python3 -P "$SCRIPT")
OWNER_READY=(
  /usr/bin/python3 -P
  "$OVERLAY/ipfs_accelerate_py/agent_supervisor/rescue/live_owner_ready.py"
)
owner_ready() {
  "${OWNER_READY[@]}" ready "$SOURCE_ROOT" "${quack_pid:-}"
}
"${LAUNCH[@]}" quack-start &
quack_pid=$!
ready=0
for _ in $(seq 1 90); do
  if owner_ready; then
    ready=1
    break
  fi
  if ! kill -0 "$quack_pid" 2>/dev/null; then
    wait "$quack_pid" || true
    echo "quack-start exited before ready" >&2
    exit 1
  fi
  sleep 2
done
if [ "$ready" != 1 ]; then
  echo "quack-start did not become ready" >&2
  kill "$quack_pid" 2>/dev/null || true
  exit 1
fi
STATE_ROOT=$("${OWNER_READY[@]}" state-root "$SOURCE_ROOT" "${quack_pid:-}" || true)
lanes_attached() {
  [ -n "$STATE_ROOT" ] || return 1
  python3 - "$STATE_ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
for status in root.glob("lane-*/*_supervisor_status.json"):
    try:
        payload = json.loads(status.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        continue
    pid = payload.get("supervisor_pid")
    if isinstance(pid, int) and pid > 1 and Path(f"/proc/{pid}").exists():
        raise SystemExit(0)
# Live scheduler/supervisors may exist before status files are rewritten.
# Match the run directory, not only .../state, so configured_board_scheduler
# argv (worktrees/merge-queue) counts as attached.
needle = str(root.parent if root.name == "state" else root)
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        continue
    if needle not in cmdline:
        continue
    if "configured_board_scheduler" in cmdline or "implementation_supervisor" in cmdline:
        raise SystemExit(0)
raise SystemExit(1)
PY
}
# Extra-gate is ready. Retry sealed launch forever; never SIGKILL extra-gate.
# If supervisors later die, relaunch them. Do not wait on extra-gate just
# because lanes were attached once — that trapped a dead scheduler.
launch_pid=""
while kill -0 "$quack_pid" 2>/dev/null; do
  if lanes_attached; then
    sleep 2
    continue
  fi
  if [ -z "$launch_pid" ] || ! kill -0 "$launch_pid" 2>/dev/null; then
    if [ -n "$launch_pid" ]; then
      wait "$launch_pid" || true
      launch_rc=$?
      if [ "$launch_rc" -eq 0 ]; then
        echo "sealed launch admitted; waiting for lane supervisors without retiring another token" >&2
        for _ in $(seq 1 45); do
          if lanes_attached; then
            break
          fi
          sleep 2
        done
        if ! lanes_attached; then
          vault="${STATE_ROOT%/state}/quack-owner/env___SAWM_QUACK_TOKEN.quack-token"
          if [ -f "$vault" ]; then
            echo "sealed launch admitted but lanes did not attach; client vault rearmed, retrying without stopping extra-gate" >&2
            launch_pid=""
          else
            echo "sealed launch admitted but lanes did not attach; holding extra-gate (gen-48 token is one-time)" >&2
          fi
        fi
        continue
      fi
      echo "sealed launch returned without lane supervisors; retrying without stopping extra-gate" >&2
    fi
    "${SEALED[@]}" launch &
    launch_pid=$!
  fi
  sleep 2
done
echo "extra-gate exited before lane supervisors attached" >&2
if [ -n "$launch_pid" ]; then
  kill "$launch_pid" 2>/dev/null || true
fi
exit 1
