#!/usr/bin/env python3
"""Ops wrapper for Meta Spark + goose implementation runner.

Delegates to
``python -m ipfs_accelerate_py.agent_supervisor.integrations.meta_spark_goose_runner``.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    from ipfs_accelerate_py.agent_supervisor.integrations.meta_spark_goose_runner import main as run

    return int(run(argv))


if __name__ == "__main__":
    raise SystemExit(main())
