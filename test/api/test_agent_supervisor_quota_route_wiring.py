from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    GROK_QUOTA_ONLY_FALLBACK_POLICY,
    IMPLEMENTATION_FALLBACK_TRIGGER_ENV,
    TodoImplementationDaemon,
)

from ipfs_accelerate_py import llm_router

ROOT = Path(__file__).resolve().parents[2]
CASF_CONFIG = ROOT / "config/agent_supervisor_causal_event_federation_scheduler.json"
FALLBACK_RUNNER = (
    Path(daemon_module.__file__).resolve().parents[1]
    / "provider_fallback_runner.py"
)
GROK_ADAPTER = FALLBACK_RUNNER.with_name("grok_cli_runner.py")


def _daemon(tmp_path: Path) -> TodoImplementationDaemon:
    todo_path = tmp_path / "todo.md"
    todo_path.write_text("# Tasks\n", encoding="utf-8")
    return TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=tmp_path / "state" / "task_state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
        worktree_root=tmp_path,
    )


def _casf_route_environment() -> dict[str, str]:
    payload = json.loads(CASF_CONFIG.read_text(encoding="utf-8"))
    provider = payload["provider"]
    route = llm_router.resolve_agent_implementation_route(
        primary_provider_id=provider["primary_provider_id"],
        primary_model_id=provider["primary_model_id"],
        fallback_provider_id=provider["fallback_provider_id"],
        fallback_model_id=provider["fallback_model_id"],
        fallback_trigger=provider["fallback_trigger"],
        fallback_reasoning_effort=provider["fallback_reasoning_effort"],
    )
    assert route.authorization is None
    return route.as_environment()


def _configure_ready_casf_route(
    monkeypatch: pytest.MonkeyPatch,
    *,
    grok: str,
    codex: str,
) -> None:
    for name, value in _casf_route_environment().items():
        monkeypatch.setenv(name, value)
    for name in (
        daemon_module.PROVIDER_FALLBACK_POLICY_ENV,
        daemon_module._ROUTE_BOARD_NAMESPACE_ENV,
        daemon_module._ROUTE_AUTHORIZATION_PATH_ENV,
        daemon_module._ROUTE_AUTHORIZATION_SHA256_ENV,
        daemon_module._ROUTE_AUTHORIZATION_ID_ENV,
        daemon_module._ROUTE_AUTHORIZATION_KIND_ENV,
        daemon_module._ROUTE_SOURCE_HEAD_ENV,
        daemon_module._ROUTE_SOURCE_TREE_ENV,
        daemon_module._ROUTE_ID_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(daemon_module, "_grok_binary", lambda: grok)
    monkeypatch.setattr(daemon_module, "_grok_cli_available", lambda: True)
    monkeypatch.setattr(
        daemon_module.shutil,
        "which",
        lambda name: codex if name == "codex" else None,
    )
    monkeypatch.setattr(
        daemon_module,
        "_grok_codex_agent_route_readiness",
        lambda *, codex: pytest.fail(
            "quota-only route must not derive authority from a models probe"
        ),
    )


def _json_argv(command: list[str], flag: str) -> list[str]:
    return json.loads(command[command.index(flag) + 1])


def test_casf_launch_route_builds_exact_ordered_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_ready_casf_route(
        monkeypatch,
        grok="/provider/grok",
        codex="/provider/codex",
    )

    command = _daemon(tmp_path)._build_implementation_command(tmp_path)

    assert command[:2] == [sys.executable, str(FALLBACK_RUNNER)]
    assert command[command.index("--fallback-policy") + 1] == (
        GROK_QUOTA_ONLY_FALLBACK_POLICY
    )
    primary = _json_argv(command, "--primary-command-json")
    assert primary[:2] == [sys.executable, str(GROK_ADAPTER)]
    assert primary[primary.index("--model") + 1] == "grok-4.6"
    assert "--codex-fallback-command-json" not in primary
    fallback = _json_argv(command, "--fallback-command-json")
    assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="high"' in fallback
    assert 'model_reasoning_effort="medium"' not in fallback
    assert "--primary-unavailable-kind" not in command


def test_authentication_unavailable_route_stays_reviewer_gated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _casf_route_environment()
    environment[IMPLEMENTATION_FALLBACK_TRIGGER_ENV] = (
        "primary_quota_or_auth_unavailable"
    )
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match="invalid sealed implementation route"):
        _daemon(tmp_path)._build_implementation_command(tmp_path)


def _write_provider(path: Path, source: str) -> None:
    path.write_text("#!/usr/bin/env python3\n" + source, encoding="utf-8")
    path.chmod(0o700)


@pytest.mark.parametrize(
    ("diagnostic", "fallback_expected"),
    (
        ('{"error":{"type":"insufficient_quota","message":"no capacity"}}', True),
        ('{"error":{"message":"authentication failed"},"http_status":401}', False),
    ),
)
def test_actual_runner_falls_back_only_after_trusted_terminal_quota(
    diagnostic: str,
    fallback_expected: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grok = tmp_path / "grok"
    codex = tmp_path / "codex"
    fallback_marker = tmp_path / "fallback.txt"
    _write_provider(
        grok,
        "import sys\n"
        f"print({diagnostic!r}, file=sys.stderr)\n"
        "raise SystemExit(19)\n",
    )
    _write_provider(
        codex,
        "import pathlib, sys\n"
        f"pathlib.Path({str(fallback_marker)!r}).write_text("
        "sys.stdin.read(), encoding='utf-8')\n",
    )
    _configure_ready_casf_route(
        monkeypatch,
        grok=str(grok),
        codex=str(codex),
    )
    command = _daemon(tmp_path)._build_implementation_command(tmp_path)
    environment = dict(os.environ)
    environment["IPFS_ACCELERATE_AGENT_PROOF_REUSE_STATE_ROOT"] = ""
    environment["IPFS_ACCELERATE_AGENT_PROVIDER_PROTECTED_STATE_ROOT"] = ""

    completed = subprocess.run(
        command,
        cwd=tmp_path,
        input="implement CASF model task\n",
        text=True,
        capture_output=True,
        check=False,
        env=environment,
        timeout=30,
    )

    assert fallback_marker.exists() is fallback_expected
    if fallback_expected:
        assert completed.returncode == 0
        assert "grok_quota_exhausted" in completed.stderr
    else:
        assert completed.returncode == 19
        assert "failure_not_fallback_eligible" in completed.stderr
