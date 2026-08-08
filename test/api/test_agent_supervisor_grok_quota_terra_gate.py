"""Contracts for Grok hard-quota → Codex Terra/medium fallback authority."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    TodoImplementationDaemon,
)
import ipfs_accelerate_py.llm_router as llm_router


_GROK_1_SPENDING_LIMIT_MESSAGE = (
    "API error (status 403 Forbidden): personal-team-blocked:spending-limit: "
    "You have run out of credits or need a Grok subscription. Add credits at "
    "https://grok.com/?_s=usage or upgrade at https://grok.com/supergrok."
)


def _daemon(root: Path) -> TodoImplementationDaemon:
    board = root / "tasks.todo.md"
    board.write_text("# Tasks\n", encoding="utf-8")
    return TodoImplementationDaemon(
        todo_path=board,
        state_path=root / "state" / "task-state.json",
        strategy_path=root / "state" / "strategy.json",
        events_path=root / "state" / "events.jsonl",
        repo_root=root,
    )


def _terra_fallback_command(codex: str, workspace: str | Path) -> list[str]:
    return [
        str(codex),
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "-s",
        "workspace-write",
        "-C",
        str(workspace),
        "-m",
        "gpt-5.6-terra",
        "-c",
        'model_reasoning_effort="medium"',
        "-",
    ]


def test_daemon_auto_route_embeds_strict_terra_fallback_when_codex_resolves(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv(implementation_daemon.IMPLEMENTATION_PROVIDER_ENV, raising=False)
    monkeypatch.delenv(implementation_daemon._CODEX_REASONING_EFFORT_ENV, raising=False)
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
    monkeypatch.delenv(
        implementation_daemon.PRODUCTION_PROVIDER_ROUTE_ENABLED_ENV, raising=False
    )
    monkeypatch.delenv(
        implementation_daemon.PRODUCTION_PROVIDER_ALLOW_RAW_COMMAND_ENV, raising=False
    )
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: True)
    monkeypatch.setattr(implementation_daemon, "_grok_binary", lambda: "/opt/providers/grok")
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: "/opt/providers/grok")
    monkeypatch.setattr(
        implementation_daemon, "_goose_meta_spark_available", lambda: False
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

    command = implementation_daemon._grok_cli_command(workspace_path=tmp_path)
    assert "--codex-fallback-command-json" in command
    fallback = json.loads(command[command.index("--codex-fallback-command-json") + 1])
    assert fallback[0] == "/opt/providers/codex"
    assert fallback[1] == "exec"
    assert "--ignore-user-config" in fallback
    assert "--ignore-rules" in fallback
    assert "--ephemeral" in fallback
    assert fallback[fallback.index("-s") + 1] == "workspace-write"
    assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="medium"' in fallback
    head = " ".join(command[: command.index("--codex-fallback-command-json")])
    assert "/opt/providers/codex" not in head


def test_quota_fallback_command_rejects_model_or_effort_drift() -> None:
    valid = _terra_fallback_command("/usr/local/bin/codex", "/repo")
    assert grok_cli_runner._parse_codex_fallback_command(json.dumps(valid)) == valid
    model_drift = list(valid)
    model_drift[model_drift.index("-m") + 1] = "gpt-5.6-sol"
    effort_drift = list(valid)
    effort_idx = next(
        i for i, item in enumerate(effort_drift) if "model_reasoning_effort=" in item
    )
    effort_drift[effort_idx] = 'model_reasoning_effort="high"'
    for drifted in (model_drift, effort_drift):
        with pytest.raises(ValueError):
            grok_cli_runner._parse_codex_fallback_command(json.dumps(drifted))


def test_quota_fallback_rejects_workspace_codex_executable(tmp_path: Path) -> None:
    attacker = tmp_path / "codex"
    attacker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    attacker.chmod(0o700)
    fallback = _terra_fallback_command(str(attacker), tmp_path)
    with pytest.raises(ValueError):
        grok_cli_runner._validate_codex_quota_fallback_command(
            fallback, workspace=tmp_path
        )


def test_quota_classifier_is_fail_closed_for_incomplete_diagnostics() -> None:
    assert (
        grok_cli_runner._grok_quota_exhausted(
            "\n".join(
                [
                    "Grok implementation failed",
                    "usage balance exhausted",
                    "402 Payment Required",
                ]
            )
        )
        is False
    )
    assert (
        grok_cli_runner._grok_quota_exhausted(
            '{"provider":"xAI","error":{"type":"insufficient_quota"}}'
        )
        is False
    )


def test_quota_classifier_accepts_exact_balance_exhausted_envelope() -> None:
    transcript = (
        'Internal error: {"message":"API error (status 402 Payment Required): '
        'Grok Build usage balance exhausted","http_status":402}'
    )
    assert grok_cli_runner._grok_quota_exhausted(transcript) is True
    parsed = grok_cli_runner.parse_grok_quota_error(transcript)
    assert parsed["kind"] == "usage_balance_exhausted"
    assert parsed["http_status"] == 402


def _write_native_grok_1_spending_limit_session(
    grok_home: Path,
    *,
    message: str = _GROK_1_SPENDING_LIMIT_MESSAGE,
    terminal_message: str | None = None,
) -> str:
    session_id = str(uuid.uuid4())
    session = grok_home / "sessions" / "%2Frepo" / session_id
    session.mkdir(parents=True)
    updates = (
        {
            "method": "_x.ai/session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "retry_state",
                    "type": "failed",
                    "error_type": "api",
                    "message": message,
                },
            },
        },
        {
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "user_message_chunk",
                    "content": {"type": "text", "text": "quota probe"},
                    "_meta": {"modelId": "grok-4.5", "promptIndex": 0},
                },
            },
        },
        {
            "method": "_x.ai/session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "turn_completed",
                    "stop_reason": "error",
                    "agent_result": (
                        message if terminal_message is None else terminal_message
                    ),
                },
            },
        },
    )
    (session / "updates.jsonl").write_text(
        "".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
            for item in updates
        ),
        encoding="utf-8",
    )
    (session / "summary.json").write_text(
        json.dumps(
            {
                "info": {"id": session_id, "cwd": "/repo"},
                "current_model_id": "grok-4.5",
                "grok_home": str(grok_home.resolve()),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return session_id


def test_native_grok_1_spending_limit_authorizes_only_exact_terminal_record(
    tmp_path: Path,
) -> None:
    grok_home = tmp_path / "grok-home"
    session_id = _write_native_grok_1_spending_limit_session(grok_home)

    assert (
        grok_cli_runner._terminal_grok_failure_type_from_isolated_home(
            grok_home,
            expected_session_id=session_id,
        )
        == "usage_pool_exhausted"
    )


def test_native_grok_1_spending_limit_rejects_session_and_model_drift(
    tmp_path: Path,
) -> None:
    grok_home = tmp_path / "grok-home"
    session_id = _write_native_grok_1_spending_limit_session(grok_home)

    assert not grok_cli_runner._terminal_grok_failure_type_from_isolated_home(
        grok_home,
        expected_session_id=str(uuid.uuid4()),
    )
    assert not grok_cli_runner._terminal_grok_failure_type_from_isolated_home(
        grok_home,
        expected_model="grok-4",
        expected_session_id=session_id,
    )

    terminal_drift_home = tmp_path / "terminal-drift-home"
    terminal_drift_session = _write_native_grok_1_spending_limit_session(
        terminal_drift_home,
        terminal_message=_GROK_1_SPENDING_LIMIT_MESSAGE + " ",
    )
    assert not grok_cli_runner._terminal_grok_failure_type_from_isolated_home(
        terminal_drift_home,
        expected_session_id=terminal_drift_session,
    )

    summary_path = next((grok_home / "sessions").rglob("summary.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["grok_home"] = str((tmp_path / "wrong-home").resolve())
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    assert not grok_cli_runner._terminal_grok_failure_type_from_isolated_home(
        grok_home,
        expected_session_id=session_id,
    )


@pytest.mark.parametrize(
    "message",
    [
        "API error (status 403 Forbidden): authentication failed",
        _GROK_1_SPENDING_LIMIT_MESSAGE.replace("403", "401"),
        _GROK_1_SPENDING_LIMIT_MESSAGE.replace("spending-limit", "rate-limit"),
        _GROK_1_SPENDING_LIMIT_MESSAGE + " ",
        _GROK_1_SPENDING_LIMIT_MESSAGE + " retry later",
    ],
)
def test_native_grok_1_spending_limit_near_matches_fail_closed(
    tmp_path: Path,
    message: str,
) -> None:
    grok_home = tmp_path / "grok-home"
    session_id = _write_native_grok_1_spending_limit_session(
        grok_home,
        message=message,
    )

    assert not grok_cli_runner._terminal_grok_failure_type_from_isolated_home(
        grok_home,
        expected_session_id=session_id,
    )


def test_build_grok_quota_routed_agent_command_embeds_terra_shape(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_k: "/usr/local/bin/codex",
    )
    command = grok_cli_runner.build_grok_quota_routed_agent_command(
        workspace=tmp_path,
        python_executable="/usr/bin/python3",
        grok_bin="/usr/bin/grok",
        codex_bin="/usr/local/bin/codex",
    )
    assert command[:3] == [
        "/usr/bin/python3",
        "-m",
        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
    ]
    assert command[command.index("--model") + 1] == "grok-4.5"
    fallback = json.loads(command[command.index("--codex-fallback-command-json") + 1])
    assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="medium"' in fallback
    assert command[command.index("--codex-fallback-reasoning-effort") + 1] == "medium"
    assert "--ephemeral" in fallback


def test_dcr_reasoning_effort_reaches_terra_fallback_and_binds_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_k: "/usr/local/bin/codex",
    )
    command = grok_cli_runner.build_grok_quota_routed_agent_command(
        workspace=tmp_path,
        python_executable="/usr/bin/python3",
        grok_bin="/usr/bin/grok",
        codex_bin="/usr/local/bin/codex",
        fallback_reasoning_effort="high",
    )
    assert command[command.index("--codex-fallback-reasoning-effort") + 1] == "high"
    fallback = json.loads(command[command.index("--codex-fallback-command-json") + 1])
    assert 'model_reasoning_effort="high"' in fallback
    assert grok_cli_runner._parse_codex_fallback_command(
        json.dumps(fallback), expected_fallback_reasoning_effort="high"
    ) == fallback
    with pytest.raises(ValueError, match="exactly medium"):
        grok_cli_runner._parse_codex_fallback_command(json.dumps(fallback))
    with pytest.raises(ValueError, match="must be one of"):
        grok_cli_runner.build_grok_quota_routed_agent_command(
            workspace=tmp_path,
            fallback_reasoning_effort="low",
        )


def test_daemon_passes_configured_dcr_reasoning_effort_to_quota_route(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(implementation_daemon._CODEX_REASONING_EFFORT_ENV, "high")
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: True)
    monkeypatch.setattr(implementation_daemon, "_grok_binary", lambda: "/opt/providers/grok")
    monkeypatch.setattr(implementation_daemon.shutil, "which", lambda name: "/opt/providers/codex" if name == "codex" else None)
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_k: "/opt/providers/codex",
    )

    command = implementation_daemon._grok_cli_command(workspace_path=tmp_path)
    assert command[command.index("--codex-fallback-reasoning-effort") + 1] == "high"
    fallback = json.loads(command[command.index("--codex-fallback-command-json") + 1])
    assert 'model_reasoning_effort="high"' in fallback
