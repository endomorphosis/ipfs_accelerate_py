#!/usr/bin/env python3
"""Stable module entry point for multi-supervisor implementation tracks."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
