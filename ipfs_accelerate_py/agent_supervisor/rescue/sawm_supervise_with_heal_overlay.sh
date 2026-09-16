#!/bin/bash
# SAWM exclusive owner + launch, with supervisor heals injected into the
# one extra-gate package. Do not start a second exclusive owner.
set -euo pipefail
OVERLAY=${1:?overlay root}
SOURCE_ROOT=${2:?sealed source root}
cd "$SOURCE_ROOT"
LAUNCH=(
  /usr/bin/python3 -P
  "$OVERLAY/ipfs_accelerate_py/agent_supervisor/rescue/sealed_board_supervisor_launch.py"
  --overlay "$OVERLAY"
  --source-root "$SOURCE_ROOT"
  --
  scripts/ops/agent_supervisor/semantic_addressed_world_model.py
)
"${LAUNCH[@]}" quack-start &
quack_pid=$!
ready=0
for _ in $(seq 1 90); do
  if "${LAUNCH[@]}" quack-ready >/dev/null 2>&1; then
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
"${LAUNCH[@]}" launch
wait "$quack_pid"
