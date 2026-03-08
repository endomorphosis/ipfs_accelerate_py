"""Regression tests for lazy unified MCP package exports."""

from __future__ import annotations

import subprocess
import sys


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )


def test_importing_logger_submodule_does_not_load_unified_server_module() -> None:
    result = _run_python(
        "import sys; import ipfs_accelerate_py.mcp_server.logger; "
        "print('ipfs_accelerate_py.mcp_server.server' in sys.modules)"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


def test_importing_logger_export_from_package_does_not_load_unified_server_module() -> None:
    result = _run_python(
        "import sys; from ipfs_accelerate_py import mcp_server; "
        "_ = mcp_server.configure_root_logging; "
        "print('ipfs_accelerate_py.mcp_server.server' in sys.modules)"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"