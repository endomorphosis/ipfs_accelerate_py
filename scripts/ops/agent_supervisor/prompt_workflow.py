#!/usr/bin/env python3
"""Thin ops wrapper for the prompt-workflow module entry.

This script intentionally only adjusts ``sys.path`` for a source checkout and
delegates to
``python -m ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow``.  It does not
import providers, open DuckDB, or start a supervisor process on its own.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
        run_prompt_workflow_cli,
    )

    return int(run_prompt_workflow_cli(argv))


if __name__ == "__main__":
    raise SystemExit(main())
