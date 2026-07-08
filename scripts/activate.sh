#!/usr/bin/env bash
# Convenience helper for interactive shells.
# Usage: source scripts/activate.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PATH="$REPO_ROOT/bin:$PATH"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
else
  echo "[activate] Missing .venv. Run: scripts/zero_touch_install.sh" >&2
  return 1
fi
