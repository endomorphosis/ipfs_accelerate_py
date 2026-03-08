#!/usr/bin/env python3
"""Smoke tests for MCP setup entry surfaces and assets."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
PACKAGE_STATIC_DIR = REPO_ROOT / "ipfs_accelerate_py" / "static" / "js"
VSCODE_WRAPPER_PATH = SCRIPTS_DIR / "vscode_mcp_server.py"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mcp_setup_assets_exist() -> None:
    assert (SCRIPTS_DIR / "comprehensive_mcp_server.py").is_file()
    assert VSCODE_WRAPPER_PATH.is_file()
    assert (PACKAGE_STATIC_DIR / "mcp-sdk.js").is_file()
    assert (PACKAGE_STATIC_DIR / "kitchen-sink-sdk.js").is_file()


def test_comprehensive_mcp_server_help() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "comprehensive_mcp_server.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=20,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert "Run the Comprehensive AI Model MCP Server" in result.stdout
    assert "--transport" in result.stdout


def test_standalone_server_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ipfs_accelerate_py.mcp.standalone", "--help"],
        capture_output=True,
        text=True,
        timeout=20,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert "IPFS Accelerate MCP Standalone Server" in result.stdout
    assert "--fastapi" in result.stdout


def test_vscode_wrapper_points_to_live_server_script(monkeypatch) -> None:
    module = _load_module("vscode_mcp_server", VSCODE_WRAPPER_PATH)

    recorded: dict[str, object] = {}

    class _FakeProcess:
        def terminate(self) -> None:
            recorded["terminated"] = True

        def wait(self) -> int:
            recorded["waited"] = True
            return 0

    def fake_popen(cmd, stdin=None, stdout=None, stderr=None, cwd=None, **_kwargs):
        recorded["cmd"] = cmd
        recorded["stdin"] = stdin
        recorded["stdout"] = stdout
        recorded["stderr"] = stderr
        recorded["cwd"] = cwd
        return _FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module.signal, "signal", lambda *_args, **_kwargs: None)

    module.run_mcp_server()

    cmd = recorded["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == sys.executable
    assert Path(cmd[1]).resolve() == (SCRIPTS_DIR / "comprehensive_mcp_server.py").resolve()
    assert cmd[2:] == ["--transport", "stdio"]
    assert Path(recorded["cwd"]).resolve() == SCRIPTS_DIR.resolve()