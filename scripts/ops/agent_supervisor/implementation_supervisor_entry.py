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


def main() -> int:
    """Redeem authority before importing the large implementation module."""

    harden_state_authority_process()
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        main as implementation_main,
    )

    return implementation_main()


if __name__ == "__main__":
    raise SystemExit(main())
