#!/usr/bin/env python3
"""Stable module entry point for multi-supervisor implementation tracks."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.runtime.process_security import (
    harden_state_authority_process,
)

# Redeem the one-shot state-authority handoff before importing the supervisor.
# The implementation module is intentionally large, so delaying redemption
# until ``main`` makes cold-import latency part of the parent's fail-closed
# handoff deadline.
harden_state_authority_process()

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (  # noqa: E402, I001
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
