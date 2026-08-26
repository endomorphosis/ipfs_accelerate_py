"""Focused DCR-000 checks for the ordered Grok quota fallback contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
)


def _configure_quota_high_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_values = {
        implementation_daemon.IMPLEMENTATION_PROVIDER_ENV: "grok_cli",
        implementation_daemon._GROK_MODEL_ENV: "grok-4.6",
        implementation_daemon.IMPLEMENTATION_FALLBACK_PROVIDER_ENV: "codex",
        implementation_daemon._CODEX_MODEL_ENV: "gpt-5.6-terra",
        implementation_daemon.IMPLEMENTATION_FALLBACK_TRIGGER_ENV: (
            "primary_quota_exhausted"
        ),
        implementation_daemon._CODEX_REASONING_EFFORT_ENV: "high",
    }
    for name, value in route_values.items():
        monkeypatch.setenv(name, value)
    for name in (
        implementation_daemon._ROUTE_BOARD_NAMESPACE_ENV,
        implementation_daemon._ROUTE_AUTHORIZATION_PATH_ENV,
        implementation_daemon._ROUTE_AUTHORIZATION_SHA256_ENV,
        implementation_daemon._ROUTE_AUTHORIZATION_ID_ENV,
        implementation_daemon._ROUTE_AUTHORIZATION_KIND_ENV,
        implementation_daemon._ROUTE_SOURCE_HEAD_ENV,
        implementation_daemon._ROUTE_SOURCE_TREE_ENV,
        implementation_daemon._ROUTE_ID_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        implementation_daemon.secrets,
        "token_hex",
        lambda _size: "ab" * 32,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_available",
        lambda: True,
    )
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


def _daemon(
    root: Path,
    *,
    implementation_command: str = "",
) -> TodoImplementationDaemon:
    board = root / "tasks.todo.md"
    board.write_text("# Tasks\n", encoding="utf-8")
    return TodoImplementationDaemon(
        todo_path=board,
        state_path=root / "state" / "task-state.json",
        strategy_path=root / "state" / "strategy.json",
        events_path=root / "state" / "events.jsonl",
        repo_root=root,
        worktree_root=root,
        implementation_command=implementation_command,
    )


def _assert_quota_high_runner(command: list[str]) -> None:
    assert Path(command[1]).name == "provider_fallback_runner.py"
    assert command[command.index("--primary-provider") + 1] == "grok"
    assert command[command.index("--fallback-provider") + 1] == "codex"
    assert command[command.index("--fallback-policy") + 1] == "grok_quota_only"
    primary = json.loads(command[command.index("--primary-command-json") + 1])
    fallback = json.loads(command[command.index("--fallback-command-json") + 1])
    assert primary[primary.index("--model") + 1] == "grok-4.6"
    assert "--require-terminal-quota-frame" in primary
    assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="high"' in fallback


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
    with pytest.raises(ValueError, match="medium or high"):
        grok_cli_runner.build_grok_quota_routed_agent_command(
            workspace=tmp_path,
            fallback_reasoning_effort="low",
        )


def test_daemon_passes_configured_high_effort_to_exact_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_quota_high_route(monkeypatch)
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
    daemon = _daemon(tmp_path)

    command = daemon._build_implementation_command(tmp_path)

    _assert_quota_high_runner(command)
    assert "--canonical-legacy-preflight-route" not in command


@pytest.mark.parametrize("override_source", ("constructor", "environment"))
def test_quota_high_route_rejects_raw_command_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override_source: str,
) -> None:
    _configure_quota_high_route(monkeypatch)
    raw_command = "codex exec -"
    if override_source == "constructor":
        monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
        daemon = _daemon(tmp_path, implementation_command=raw_command)
    else:
        monkeypatch.setenv("IMPLEMENTATION_DAEMON_COMMAND", raw_command)
        daemon = _daemon(tmp_path)

    for action in (
        lambda: daemon._require_primary_provider_readiness(None),
        lambda: daemon._build_implementation_command(tmp_path),
    ):
        with pytest.raises(
            implementation_daemon.ImplementationRetryDeferred,
            match="sealed Grok/Codex route rejects",
        ):
            action()


def test_grok_implement_role_keeps_quota_high_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_quota_high_route(monkeypatch)
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
    task = PortalTask(
        task_id="DCR-ROUTE-001",
        title="Use the ordered Grok-first route",
        status="ready",
        completion="manual",
        priority="P0",
        track="provider",
        outputs=["src/provider.py"],
        metadata={"Provider role": "grok-implement"},
    )
    daemon = _daemon(tmp_path)

    daemon._require_primary_provider_readiness(task)
    command = daemon._build_implementation_command(tmp_path, task=task)

    assert "--canonical-legacy-preflight-route" not in command
    _assert_quota_high_runner(command)


def test_grok_only_role_stays_grok_only_with_ordered_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_quota_high_route(monkeypatch)
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
    task = PortalTask(
        task_id="DCR-ROUTE-002",
        title="Stay pinned to Grok",
        status="ready",
        completion="manual",
        priority="P0",
        track="provider",
        outputs=["src/provider.py"],
        metadata={"Provider role": "grok-only"},
    )
    daemon = _daemon(tmp_path)

    daemon._require_primary_provider_readiness(task)
    command = daemon._build_implementation_command(tmp_path, task=task)

    assert command[command.index("--model") + 1] == "grok-4.6"
    assert "--codex-fallback-command-json" not in command
    assert "--grok-failure-receipt-nonce" not in command
    assert "--agent-implementation-route-json" not in command
