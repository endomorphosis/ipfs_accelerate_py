#!/usr/bin/env python3
"""Regression tests for the canonical CLI entry point."""

from __future__ import annotations

import subprocess
import sys


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Execute the CLI module directly using the active interpreter."""
    return subprocess.run(
        [sys.executable, "-m", "ipfs_accelerate_py.cli_entry", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_module_entry_point_help() -> None:
    result = _run_cli("--help")

    assert result.returncode == 0, result.stderr
    assert "IPFS Accelerate CLI" in result.stdout


def test_mcp_command_help() -> None:
    result = _run_cli("mcp", "--help")

    assert result.returncode == 0, result.stderr
    assert "start" in result.stdout
    assert "dashboard" in result.stdout


def test_mcp_start_help() -> None:
    result = _run_cli("mcp", "start", "--help")

    assert result.returncode == 0, result.stderr
    assert "--dashboard" in result.stdout
    assert "--port" in result.stdout