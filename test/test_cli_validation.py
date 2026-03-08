#!/usr/bin/env python3
"""Smoke tests for the canonical `ipfs_accelerate_py.cli` module."""

from __future__ import annotations

import importlib
import subprocess
import sys


def _run_cli_module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ipfs_accelerate_py.cli", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cli_module_exports_main_entry_point() -> None:
    cli_module = importlib.import_module("ipfs_accelerate_py.cli")

    assert hasattr(cli_module, "main")
    assert callable(cli_module.main)
    assert hasattr(cli_module, "IPFSAccelerateCLI")


def test_cli_module_help() -> None:
    result = _run_cli_module("--help")

    assert result.returncode == 0, result.stderr
    assert "IPFS Accelerate CLI - Unified interface" in result.stdout
    assert "mcp" in result.stdout
    assert "models" in result.stdout


def test_cli_module_mcp_start_help() -> None:
    result = _run_cli_module("mcp", "start", "--help")

    assert result.returncode == 0, result.stderr
    assert "--dashboard" in result.stdout
    assert "--port" in result.stdout