"""Regression tests for cold-import-safe todo-daemon entry points."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "ipfs_accelerate_py.agent_supervisor.todo_daemon"
DAEMON_MODULE = f"{PACKAGE}.implementation_daemon"
SUPERVISOR_MODULE = f"{PACKAGE}.implementation_supervisor"


def _run_python(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_package_import_defers_implementation_modules_until_export_access() -> None:
    script = f"""
import sys
import {PACKAGE} as package

assert {DAEMON_MODULE!r} not in sys.modules
assert {SUPERVISOR_MODULE!r} not in sys.modules
assert package.TodoImplementationDaemon.__module__ == {DAEMON_MODULE!r}
assert {DAEMON_MODULE!r} in sys.modules
assert {SUPERVISOR_MODULE!r} not in sys.modules
assert package.TodoImplementationSupervisor.__module__ == {SUPERVISOR_MODULE!r}
assert {SUPERVISOR_MODULE!r} in sys.modules
"""
    result = _run_python("-W", "error::RuntimeWarning", "-c", script)

    assert result.returncode == 0, result.stderr


def test_implementation_module_entry_points_have_no_runpy_warning() -> None:
    for module_name in (DAEMON_MODULE, SUPERVISOR_MODULE):
        result = _run_python(
            "-W",
            "error::RuntimeWarning",
            "-m",
            module_name,
            "--help",
        )

        assert result.returncode == 0, result.stderr
        assert "found in sys.modules" not in result.stderr
