"""Keep an explicit quota/high route intact through daemon command building."""

from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon


@pytest.fixture
def quota_high_daemon(tmp_path, monkeypatch):
    route = llm_router.resolve_agent_implementation_route(
        primary_provider_id="grok_cli", primary_model_id="grok-4.6",
        fallback_provider_id="codex", fallback_model_id="gpt-5.6-terra",
        fallback_trigger="primary_quota_exhausted", fallback_reasoning_effort="high",
    )
    for name, value in route.as_environment().items():
        monkeypatch.setenv(name, value)
    for name in (
        "IMPLEMENTATION_DAEMON_COMMAND",
        implementation_daemon.PRODUCTION_PROVIDER_ROUTE_ENABLED_ENV,
        implementation_daemon.PRODUCTION_PROVIDER_ALLOW_RAW_COMMAND_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: True)
    monkeypatch.setattr(implementation_daemon, "_grok_binary", lambda: "/opt/providers/grok")
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: "/opt/providers/grok")
    monkeypatch.setattr(implementation_daemon, "_goose_meta_spark_available", lambda: False)
    monkeypatch.setattr(
        grok_cli_runner, "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: "/opt/providers/codex",
    )
    todo = tmp_path / "tasks.todo.md"
    todo.write_text("# Tasks\n")
    daemon = implementation_daemon.TodoImplementationDaemon(
        todo_path=todo, state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json", events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path, worktree_root=tmp_path,
    )
    return daemon, route, tmp_path


def test_quota_high_command_preserves_effort_and_route(quota_high_daemon):
    daemon, route, workspace = quota_high_daemon
    command = daemon._build_implementation_command(workspace)
    fallback = json.loads(command[command.index("--codex-fallback-command-json") + 1])
    assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="high"' in fallback
    assert 'model_reasoning_effort="medium"' not in fallback
    binding = json.loads(command[command.index("--agent-implementation-route-json") + 1])
    assert binding == route.as_binding_dict()
    assert binding["fallback_trigger"] == "primary_quota_exhausted"
    nonce = command[command.index("--grok-failure-receipt-nonce") + 1]
    assert len(nonce) == 64 and set(nonce) <= set("0123456789abcdef")
    assert "--canonical-legacy-preflight-route" not in command


def test_quota_high_command_does_not_admit_authentication_failure(
    quota_high_daemon, monkeypatch,
):
    daemon, _route, workspace = quota_high_daemon
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: False)
    with pytest.raises(RuntimeError, match="not authenticated"):
        daemon._build_implementation_command(workspace)


@pytest.mark.parametrize("override", ["constructor", "environment"])
def test_quota_high_command_denies_raw_override(quota_high_daemon, monkeypatch, override):
    daemon, _route, workspace = quota_high_daemon
    if override == "constructor":
        daemon.implementation_command = "arbitrary-command"
    else:
        monkeypatch.setenv("IMPLEMENTATION_DAEMON_COMMAND", "arbitrary-command")
    with pytest.raises(implementation_daemon.ImplementationRetryDeferred, match="override"):
        daemon._build_implementation_command(workspace)


def test_quota_high_command_keeps_explicit_grok_only_task(quota_high_daemon, monkeypatch):
    daemon, _route, workspace = quota_high_daemon
    monkeypatch.setattr(daemon, "_task_declared_implementation_provider", lambda _task: "grok")
    command = daemon._build_implementation_command(workspace)
    assert "--codex-fallback-command-json" not in command
    assert "--agent-implementation-route-json" not in command


def test_quota_high_command_preserves_independent_review(quota_high_daemon, monkeypatch):
    daemon, _route, workspace = quota_high_daemon
    monkeypatch.setattr(daemon, "_task_declares_independent_codex_review", lambda _task: True)
    with pytest.raises(implementation_daemon.ImplementationRetryDeferred, match="independent Codex review"):
        daemon._build_implementation_command(workspace)
