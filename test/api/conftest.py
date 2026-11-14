"""Test API fixtures and path setup."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


@pytest.fixture(autouse=True)
def _release_agent_supervisor_duckdb_tmp_path(request: pytest.FixtureRequest):
    """Drop DuckDB handles before pytest tries to delete ``tmp_path``.

    Agent-supervisor tests write ``control.duckdb`` into the default
    ``/tmp/pytest-of-*`` tree. Unclosed connections keep ``.lock`` / ``.wal``
    files open, pytest fails to rmtree the session, and the directory is
    renamed to ``garbage-*``.
    """

    yield
    tmp_path = request.node.funcargs.get("tmp_path")
    if tmp_path is None:
        return
    gc.collect()
    try:
        from ipfs_accelerate_py.agent_supervisor.duckdb_state import (
            release_duckdb_tree,
        )
    except Exception:
        return
    release_duckdb_tree(tmp_path)
