"""Focused DCR-000 checks for the ordered Grok quota fallback contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon


def test_builder_binds_high_for_dcr_and_keeps_legacy_medium(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: "/usr/local/bin/codex",
    )
    high = grok_cli_runner.build_grok_quota_routed_agent_command(
        workspace=tmp_path,
        python_executable="/usr/bin/python3",
        grok_bin="/usr/bin/grok",
        codex_bin="/usr/local/bin/codex",
        fallback_reasoning_effort="high",
    )
    assert high[high.index("--codex-fallback-reasoning-effort") + 1] == "high"
    high_fallback = json.loads(high[high.index("--codex-fallback-command-json") + 1])
    assert 'model_reasoning_effort="high"' in high_fallback
    assert (
        grok_cli_runner._parse_codex_fallback_command(
            json.dumps(high_fallback),
            expected_fallback_reasoning_effort="high",
        )
        == high_fallback
    )

    medium = grok_cli_runner.build_grok_quota_routed_agent_command(
        workspace=tmp_path,
        codex_bin="/usr/local/bin/codex",
    )
    assert medium[medium.index("--codex-fallback-reasoning-effort") + 1] == "medium"
    with pytest.raises(ValueError, match="must be medium or high"):
        grok_cli_runner.build_grok_quota_routed_agent_command(
            workspace=tmp_path,
            fallback_reasoning_effort="low",
        )


def test_daemon_passes_configured_high_effort_to_exact_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_environment = {
        implementation_daemon.IMPLEMENTATION_PROVIDER_ENV: "grok_cli",
        implementation_daemon._GROK_MODEL_ENV: "grok-4.6",
        implementation_daemon.IMPLEMENTATION_FALLBACK_PROVIDER_ENV: "codex",
        implementation_daemon._CODEX_MODEL_ENV: "gpt-5.6-terra",
        implementation_daemon.IMPLEMENTATION_FALLBACK_TRIGGER_ENV: (
            "primary_quota_exhausted"
        ),
        implementation_daemon._CODEX_REASONING_EFFORT_ENV: "high",
    }
    for name, value in route_environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: True)
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: "/opt/providers/codex",
    )

    board = tmp_path / "tasks.todo.md"
    board.write_text("# Tasks\n", encoding="utf-8")
    daemon = implementation_daemon.TodoImplementationDaemon(
        todo_path=board,
        state_path=tmp_path / "state" / "task-state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
        worktree_root=tmp_path,
    )

    command = daemon._build_implementation_command(tmp_path)
    assert command[command.index("--codex-fallback-reasoning-effort") + 1] == "high"
    fallback = json.loads(command[command.index("--codex-fallback-command-json") + 1])
    assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="high"' in fallback
    assert "--agent-implementation-route-json" in command
    assert "--grok-failure-receipt-nonce" in command
    assert "--canonical-legacy-preflight-route" not in command


def test_quota_only_route_requires_authenticated_pinned_grok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_environment = {
        implementation_daemon.IMPLEMENTATION_PROVIDER_ENV: "grok_cli",
        implementation_daemon._GROK_MODEL_ENV: "grok-4.6",
        implementation_daemon.IMPLEMENTATION_FALLBACK_PROVIDER_ENV: "codex",
        implementation_daemon._CODEX_MODEL_ENV: "gpt-5.6-terra",
        implementation_daemon.IMPLEMENTATION_FALLBACK_TRIGGER_ENV: (
            "primary_quota_exhausted"
        ),
        implementation_daemon._CODEX_REASONING_EFFORT_ENV: "high",
    }
    for name, value in route_environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: False)
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )

    board = tmp_path / "tasks.todo.md"
    board.write_text("# Tasks\n", encoding="utf-8")
    daemon = implementation_daemon.TodoImplementationDaemon(
        todo_path=board,
        state_path=tmp_path / "state" / "task-state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
        worktree_root=tmp_path,
    )

    with pytest.raises(
        implementation_daemon.ImplementationRetryDeferred,
        match="requires an authenticated pinned Grok CLI",
    ):
        daemon._build_implementation_command(tmp_path)


def test_quota_only_route_requires_trusted_codex_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_environment = {
        implementation_daemon.IMPLEMENTATION_PROVIDER_ENV: "grok_cli",
        implementation_daemon._GROK_MODEL_ENV: "grok-4.6",
        implementation_daemon.IMPLEMENTATION_FALLBACK_PROVIDER_ENV: "codex",
        implementation_daemon._CODEX_MODEL_ENV: "gpt-5.6-terra",
        implementation_daemon.IMPLEMENTATION_FALLBACK_TRIGGER_ENV: (
            "primary_quota_exhausted"
        ),
        implementation_daemon._CODEX_REASONING_EFFORT_ENV: "high",
    }
    for name, value in route_environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: True)
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: "",
    )

    board = tmp_path / "tasks.todo.md"
    board.write_text("# Tasks\n", encoding="utf-8")
    daemon = implementation_daemon.TodoImplementationDaemon(
        todo_path=board,
        state_path=tmp_path / "state" / "task-state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
        worktree_root=tmp_path,
    )

    with pytest.raises(
        RuntimeError,
        match="sealed Grok/Codex route requires a trusted Codex CLI",
    ):
        daemon._build_implementation_command(tmp_path)
