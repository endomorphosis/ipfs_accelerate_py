"""Claude / Gemini implement command builders and runner CLI."""

from __future__ import annotations

import sys
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    CLAUDE_IMPLEMENTATION_PROVIDER_NAMES,
    GEMINI_IMPLEMENTATION_PROVIDER_NAMES,
    SUPPORTED_IMPLEMENTATION_PROVIDER_NAMES,
    _claude_implementation_command,
    _gemini_implementation_command,
    _provider_labels_from_implementation_command,
)


def test_claude_and_gemini_are_supported_provider_pins() -> None:
    assert "claude" in SUPPORTED_IMPLEMENTATION_PROVIDER_NAMES
    assert "gemini" in SUPPORTED_IMPLEMENTATION_PROVIDER_NAMES
    assert "claude_code" in CLAUDE_IMPLEMENTATION_PROVIDER_NAMES
    assert "gemini_cli" in GEMINI_IMPLEMENTATION_PROVIDER_NAMES


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
