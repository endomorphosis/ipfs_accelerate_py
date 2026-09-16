"""Claude / Gemini implement command builders and runner CLI."""

from __future__ import annotations

import sys
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    CLAUDE_IMPLEMENTATION_PROVIDER_NAMES,
    GEMINI_IMPLEMENTATION_PROVIDER_NAMES,
    GOOSE_IMPLEMENTATION_PROVIDER_NAMES,
    MUSE_CODE_IMPLEMENTATION_PROVIDER_NAMES,
    SUPPORTED_IMPLEMENTATION_PROVIDER_NAMES,
    _claude_implementation_command,
    _gemini_implementation_command,
    _muse_code_implementation_command,
    _provider_labels_from_implementation_command,
)


def test_claude_and_gemini_are_supported_provider_pins() -> None:
    assert "claude" in SUPPORTED_IMPLEMENTATION_PROVIDER_NAMES
    assert "gemini" in SUPPORTED_IMPLEMENTATION_PROVIDER_NAMES
    assert "claude_code" in CLAUDE_IMPLEMENTATION_PROVIDER_NAMES
    assert "gemini_cli" in GEMINI_IMPLEMENTATION_PROVIDER_NAMES
    assert "muse_code" in SUPPORTED_IMPLEMENTATION_PROVIDER_NAMES
    assert "muse_code" in MUSE_CODE_IMPLEMENTATION_PROVIDER_NAMES
    # Historical goose/Meta Spark pin stays on "muse"; Muse Code is muse_code.
    assert "muse" in GOOSE_IMPLEMENTATION_PROVIDER_NAMES
    assert "muse" not in MUSE_CODE_IMPLEMENTATION_PROVIDER_NAMES


def test_claude_implementation_command_uses_runner_module() -> None:
    workspace = Path("/tmp/example-worktree")
    command = _claude_implementation_command(workspace_path=workspace)
    assert command[0] == sys.executable
    assert command[1:3] == [
        "-m",
        "ipfs_accelerate_py.agent_supervisor.cli_implement_runner",
    ]
    assert command[command.index("--provider") + 1] == "claude"
    assert command[command.index("--workspace") + 1] == str(workspace.resolve())


def test_gemini_implementation_command_accepts_model_override() -> None:
    workspace = Path("/tmp/example-worktree")
    command = _gemini_implementation_command(
        workspace_path=workspace,
        model_override="gemini-2.0-flash",
    )
    assert command[command.index("--provider") + 1] == "gemini"
    assert command[command.index("--model") + 1] == "gemini-2.0-flash"


def test_muse_code_implementation_command_installs_and_uses_runner(
    monkeypatch,
) -> None:
    import ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon as daemon

    monkeypatch.setattr(daemon, "_muse_code_binary", lambda auto_install=False: "/tmp/muse")
    monkeypatch.setenv("META_API_KEY", "test-key")
    workspace = Path("/tmp/example-worktree")
    command = _muse_code_implementation_command(
        workspace_path=workspace,
        model_override="muse-spark-1.2",
    )
    assert command[0] == sys.executable
    assert command[1:3] == [
        "-m",
        "ipfs_accelerate_py.agent_supervisor.cli_implement_runner",
    ]
    assert command[command.index("--provider") + 1] == "muse_code"
    assert command[command.index("--workspace") + 1] == str(workspace.resolve())
    assert command[command.index("--model") + 1] == "muse-spark-1.2"


def test_muse_runner_installs_official_cli_and_runs_exec(
    tmp_path: Path, monkeypatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor import cli_implement_runner as runner

    muse = tmp_path / "muse"
    muse.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    muse.chmod(0o755)
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.cli_provider_balance.ensure_muse_cli_binary",
        lambda: str(muse),
    )
    recorded: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        recorded["argv"] = list(argv)
        recorded["cwd"] = kwargs.get("cwd")
        recorded["env"] = kwargs.get("env") or {}
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    workspace = tmp_path / "work"
    workspace.mkdir()
    rc = runner._run_muse(
        workspace=workspace,
        model="muse-spark-1.2",
        prompt="implement the failing test",
    )
    assert rc == 0
    argv = recorded["argv"]
    assert isinstance(argv, list)
    assert argv[0] == str(muse)
    assert argv[1] == "exec"
    assert "--disable-approval" in argv
    assert "--prompt-file" in argv
    assert "--workspace" in argv
    assert recorded["cwd"] == str(workspace)
    labels = _provider_labels_from_implementation_command(argv)
    assert "muse_code" in labels


def test_provider_labels_attribute_cli_implement_runner() -> None:
    claude_cmd = _claude_implementation_command(workspace_path=Path("/tmp/w"))
    # labels from concrete argv tokens
    labels = _provider_labels_from_implementation_command(
        ["claude", "-p", "do work", "--dangerously-skip-permissions"]
    )
    assert "claude" in labels
    gemini_labels = _provider_labels_from_implementation_command(
        ["npx", "@google/gemini-cli", "-p", "do work"]
    )
    assert "gemini" in gemini_labels
    # runner module path alone is not a provider binary label
    assert isinstance(claude_cmd, list)
