"""Campaign's explicit Grok pin uses the supported native quota-gated runner.

No generation or provider process is allowed in these command-assembly tests.
The positive integration check uses installed binary/auth readiness and skips
only when that real prerequisite is absent on another host.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_provider_auto


@pytest.fixture
def pinned_daemon(tmp_path, monkeypatch):
    for key in (
        "IMPLEMENTATION_DAEMON_COMMAND",
        implementation_daemon.IMPLEMENTATION_FALLBACK_PROVIDER_ENV,
        implementation_daemon.IMPLEMENTATION_FALLBACK_TRIGGER_ENV,
        implementation_daemon._GROK_MODEL_ENV,
        implementation_daemon._CODEX_MODEL_ENV,
        implementation_daemon._CODEX_REASONING_EFFORT_ENV,
        implementation_daemon._ROUTE_BOARD_NAMESPACE_ENV,
        implementation_daemon._ROUTE_AUTHORIZATION_PATH_ENV,
        implementation_daemon._ROUTE_AUTHORIZATION_SHA256_ENV,
        implementation_daemon._ROUTE_AUTHORIZATION_ID_ENV,
        implementation_daemon._ROUTE_AUTHORIZATION_KIND_ENV,
        implementation_daemon._ROUTE_SOURCE_HEAD_ENV,
        implementation_daemon._ROUTE_SOURCE_TREE_ENV,
        implementation_daemon._ROUTE_ID_ENV,
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(implementation_daemon.IMPLEMENTATION_PROVIDER_ENV, "grok")
    board = tmp_path / "tasks.todo.md"
    board.write_text("# Agent Todos\n")
    worktrees = tmp_path / "worktrees"
    workspace = worktrees / "worker"
    workspace.mkdir(parents=True)
    daemon = implementation_daemon.TodoImplementationDaemon(
        todo_path=board,
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
        worktree_root=worktrees,
    )
    task = implementation_daemon.PortalTask(
        task_id="AF-001",
        title="Recover editable manuscript and missing reference artifacts",
        status="ready",
        completion="auto",
        priority="P0",
        track="autoformalization",
        outputs=["papers/completion/autoformalization/manuscript/main.tex"],
        validation=[
            "python3 scripts/paper_supervisors.py verify-task "
            "--paper autoformalization --task AF-001"
        ],
        acceptance="Preserve original source content and evidence.",
    )
    real_popen = subprocess.Popen

    def prohibit_provider_process(argv, *args, **kwargs):
        if isinstance(argv, (list, tuple)) and argv:
            executable = Path(str(argv[0])).name
            assert executable not in {"grok", "codex", "claude", "copilot"}, (
                "Command assembly must not start a provider process"
            )
        return real_popen(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", prohibit_provider_process)
    return daemon, workspace, task


def test_real_grok_pin_assembles_quota_gated_runner(pinned_daemon, monkeypatch):
    daemon, workspace, task = pinned_daemon
    if not implementation_daemon._grok_cli_available():
        pytest.skip("Installed authenticated Grok CLI is required for this integration check")

    def reject_legacy_auto(*args, **kwargs):
        raise AssertionError("An explicit pin must not enter the obsolete auto adapter")

    monkeypatch.setattr(
        implementation_provider_auto,
        "select_auto_implementation_provider",
        reject_legacy_auto,
    )
    command = daemon._build_implementation_command(workspace, task=task)
    assert command[:3] == [
        implementation_daemon.sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
    ]
    assert command[command.index("--workspace") + 1] == str(workspace)
    assert command[command.index("--grok-bin") + 1] == implementation_daemon._grok_binary()
    assert command[command.index("--model") + 1] == "grok-4.6"
    trusted_codex = grok_cli_runner.resolve_codex_quota_fallback_executable(
        workspace=workspace,
        configured=str(implementation_daemon.shutil.which("codex") or ""),
    )
    if trusted_codex:
        assert grok_cli_runner.CANONICAL_LEGACY_PREFLIGHT_ROUTE_FLAG in command
        fallback = json.loads(command[command.index("--codex-fallback-command-json") + 1])
        assert fallback[:2] == [trusted_codex, "exec"]
        assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
        assert 'model_reasoning_effort="medium"' in fallback
    else:
        assert "--codex-fallback-command-json" not in command
    assert "--agent-implementation-route-json" not in command
    assert "--grok-failure-receipt-nonce" not in command


def test_grok_pin_does_not_bypass_authentication(pinned_daemon, monkeypatch):
    daemon, workspace, task = pinned_daemon
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires the Grok Build CLI"):
        daemon._build_implementation_command(workspace, task=task)


def test_grok_pin_rejects_incomplete_ordered_route(pinned_daemon, monkeypatch):
    daemon, workspace, task = pinned_daemon
    monkeypatch.setenv(
        implementation_daemon.IMPLEMENTATION_FALLBACK_TRIGGER_ENV,
        "primary_quota_or_auth_unavailable",
    )
    with pytest.raises(
        implementation_daemon.ImplementationRetryDeferred,
        match="invalid sealed implementation route",
    ):
        daemon._build_implementation_command(workspace, task=task)
