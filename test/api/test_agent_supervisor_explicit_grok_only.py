"""An explicit Grok pin must not grant an implicit Codex quota fallback."""

from __future__ import annotations

import json
import socket
import subprocess
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    allocation_compat,
    cli_provider_balance,
    implementation_daemon,
    implementation_provider_auto,
)


@pytest.fixture
def command_only_daemon(tmp_path, monkeypatch):
    """Exercise native routing and command assembly without process or I/O probes."""

    def reject_external(*_args, **_kwargs):
        pytest.fail("command assembly attempted a process or network connection")

    monkeypatch.setattr(subprocess, "Popen", reject_external)
    monkeypatch.setattr(socket.socket, "connect", reject_external)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_external)
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
    monkeypatch.setenv(implementation_daemon._GROK_MODEL_ENV, "grok-4.7")
    monkeypatch.delenv(implementation_daemon._GROK_MAX_TURNS_ENV, raising=False)
    monkeypatch.setattr(allocation_compat, "ensure_allocation_schema", lambda: None)
    monkeypatch.setattr(implementation_daemon, "_grok_binary", lambda: "/fixture/grok")
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: True)
    monkeypatch.setattr(implementation_daemon, "_goose_meta_spark_available", lambda: False)
    monkeypatch.setattr(implementation_daemon, "_configured_agent_implementation_route_plan", lambda _root: None)
    monkeypatch.setattr(llm_router, "_grok_cli_auth_available", lambda: True)
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda _provider: object())
    monkeypatch.setattr(cli_provider_balance, "probe_all_cli_provider_readiness", lambda: {})
    monkeypatch.setattr(grok_cli_runner, "resolve_codex_quota_fallback_executable", lambda **_kwargs: "/fixture/codex")

    daemon = object.__new__(implementation_daemon.PortalImplementationDaemon)
    daemon.repo_root = tmp_path
    daemon.implementation_command = None
    daemon.manual_completion_authority_revalidation_only = False
    daemon._scoped_control_plane_launch = None
    monkeypatch.setattr(daemon, "_require_implement_workspace_not_merge_target", lambda workspace, **_kwargs: workspace)
    monkeypatch.setattr(daemon, "_protected_effect_recovery_command", lambda **_kwargs: None)
    monkeypatch.setattr(daemon, "_task_declared_implementation_provider", lambda _task: "")
    monkeypatch.setattr(daemon, "_task_uses_typed_local_execution", lambda _task: False)
    monkeypatch.setattr(daemon, "_task_declares_independent_codex_review", lambda _task: False)
    monkeypatch.setattr(daemon, "_provider_capacity_latch_states", lambda: {})
    monkeypatch.setattr(daemon, "_record_event", lambda *_args, **_kwargs: None)
    return daemon


@pytest.mark.parametrize("provider", sorted(implementation_daemon.GROK_IMPLEMENTATION_PROVIDER_NAMES))
def test_explicit_grok_alias_has_no_codex_fallback(command_only_daemon, tmp_path, monkeypatch, provider):
    monkeypatch.setenv(implementation_daemon.IMPLEMENTATION_PROVIDER_ENV, provider)
    command = command_only_daemon._build_implementation_command(tmp_path)
    assert command[:3] == [
        implementation_daemon.sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner",
    ]
    assert command[command.index("--model") + 1] == "grok-4.7"
    assert "--codex-fallback-command-json" not in command
    assert grok_cli_runner.CANONICAL_LEGACY_PREFLIGHT_ROUTE_FLAG not in command


def test_task_grok_pin_has_no_codex_fallback(command_only_daemon, tmp_path, monkeypatch):
    monkeypatch.setenv(implementation_daemon.IMPLEMENTATION_PROVIDER_ENV, "auto")
    monkeypatch.setattr(command_only_daemon, "_task_declared_implementation_provider", lambda _task: "grok")
    command = command_only_daemon._build_implementation_command(tmp_path)
    assert "--codex-fallback-command-json" not in command


def test_grok_pin_still_requires_authentication(command_only_daemon, tmp_path, monkeypatch):
    monkeypatch.setenv(implementation_daemon.IMPLEMENTATION_PROVIDER_ENV, "grok")
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires the Grok Build CLI"):
        command_only_daemon._build_implementation_command(tmp_path)


def test_auto_route_keeps_its_explicit_quota_fallback(command_only_daemon, tmp_path, monkeypatch):
    monkeypatch.setenv(implementation_daemon.IMPLEMENTATION_PROVIDER_ENV, "auto")
    monkeypatch.setattr(
        implementation_provider_auto,
        "select_auto_implementation_provider",
        lambda **_kwargs: SimpleNamespace(
            decision=implementation_provider_auto.AutoProviderDecision.GROK,
            to_dict=lambda: {"decision": "grok"},
        ),
    )
    command = command_only_daemon._build_implementation_command(tmp_path)
    fallback = json.loads(command[command.index("--codex-fallback-command-json") + 1])
    assert fallback[:2] == ["/fixture/codex", "exec"]
    assert fallback[fallback.index("-m") + 1] == "gpt-6.1-sol"
    assert "--agent-implementation-route-json" in command


@pytest.mark.parametrize("reasoning_effort", ("medium", "high"))
def test_configured_quota_route_is_preserved(command_only_daemon, tmp_path, monkeypatch, reasoning_effort):
    monkeypatch.setenv(implementation_daemon.IMPLEMENTATION_PROVIDER_ENV, "grok")
    route = llm_router.resolve_agent_implementation_route(
        primary_provider_id="grok_cli",
        primary_model_id="grok-4.7",
        fallback_provider_id="codex",
        fallback_model_id="gpt-6.1-sol",
        fallback_trigger="primary_quota_exhausted",
        fallback_reasoning_effort=reasoning_effort,
    )
    monkeypatch.setattr(implementation_daemon, "_configured_agent_implementation_route_plan", lambda _root: route)
    command = command_only_daemon._build_implementation_command(tmp_path)
    fallback = json.loads(command[command.index("--codex-fallback-command-json") + 1])
    assert fallback[:2] == ["/fixture/codex", "exec"]
    assert f'model_reasoning_effort="{reasoning_effort}"' in fallback
    binding = json.loads(command[command.index("--agent-implementation-route-json") + 1])
    assert binding == route.as_binding_dict()
    assert "--grok-failure-receipt-nonce" in command
