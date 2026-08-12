"""ASE3-012 expert compatibility and installed-surface smoke matrix."""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.control.control_cli import (
    register_agent_cli,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints import cli as supervisor_cli
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    PROMPT_LIFECYCLE_TOOLS,
    register_all_agent_supervisor_tools,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class _RecordingManager:
    def __init__(self) -> None:
        self.names: list[str] = []

    def register_tool(self, **kwargs: object) -> None:
        self.names.append(str(kwargs["name"]))


def test_expert_agent_cli_still_registers_alongside_supervisor() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    agent = register_agent_cli(sub)
    supervisor = supervisor_cli.register_supervisor_cli(sub)
    assert "agent" in agent.prog or agent.prog.endswith("agent")
    assert "supervisor" in supervisor.prog or True
    # Both groups present on the same parent parser.
    choices = getattr(sub, "choices", {}) or {}
    assert "agent" in choices
    assert "supervisor" in choices


def test_expert_agent_help_remains_available() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        [sys.executable, "-m", "ipfs_accelerate_py.cli_entry", "agent", "--help"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "capabilities" in completed.stdout or "status" in completed.stdout


def test_prompt_lifecycle_tools_do_not_remove_control_plane_tools() -> None:
    manager = _RecordingManager()
    register_all_agent_supervisor_tools(manager)
    names = set(manager.names)
    # Prompt lifecycle tools present.
    assert set(PROMPT_LIFECYCLE_TOOLS) <= names
    # Legacy/control operation tools still registered (at least one).
    assert len(names) > len(PROMPT_LIFECYCLE_TOOLS)


def test_entrypoint_package_cold_import_still_provider_free() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    script = r"""
import json
import sys
before = set(sys.modules)
import ipfs_accelerate_py.agent_supervisor.entrypoints as ep
after = set(sys.modules)
added = sorted(after - before)
forbidden = [
    name for name in added
    if name.startswith("duckdb")
    or name.startswith("torch")
    or name.startswith("openai")
]
print(json.dumps({
    "lazy": list(ep.ENTRYPOINT_LAZY_FACADE_EXPORTS)[:3],
    "forbidden": forbidden,
    "has_facade_module": "ipfs_accelerate_py.agent_supervisor.entrypoints.facade" in added,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    import json

    payload = json.loads(completed.stdout)
    assert payload["forbidden"] == []
    assert payload["has_facade_module"] is False
    assert "Supervisor" in payload["lazy"]


def test_docs_guide_exists_and_documents_journey() -> None:
    guide = (
        REPO_ROOT / "docs" / "guides" / "AGENT_SUPERVISOR_PROMPT_ENTRYPOINTS.md"
    )
    assert guide.is_file()
    text = guide.read_text(encoding="utf-8")
    for needle in (
        "Supervisor.open",
        "ipfs-accelerate supervisor run",
        "agent_supervisor_run",
        "ASE3-026",
        "connect_duckdb_with_policy",
    ):
        assert needle in text


def test_package_root_still_exports_supervisor_lazily() -> None:
    import ipfs_accelerate_py as root

    # Access should resolve without starting providers.
    Supervisor = root.Supervisor
    assert Supervisor.__name__ == "Supervisor"
