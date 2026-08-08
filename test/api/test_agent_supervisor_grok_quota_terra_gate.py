"""Contracts for Grok hard-quota → Codex Terra/medium fallback authority."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import ipfs_accelerate_py.llm_router as llm_router
import pytest
from ipfs_accelerate_py.agent_supervisor import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    TodoImplementationDaemon,
)

_NATIVE_SESSION_ID = "00000000-0000-4000-8000-000000000001"
_SPENDING_LIMIT_MESSAGE = (
    "API error (status 403 Forbidden): personal-team-blocked:spending-limit: "
    "You have run out of credits or need a Grok subscription. Add credits at "
    "https://grok.com/?_s=usage or upgrade at https://grok.com/supergrok."
)


def _native_update(
    update: dict[str, object],
    *,
    session_id: str = _NATIVE_SESSION_ID,
) -> dict[str, object]:
    return {
        "method": "_x.ai/session/update",
        "params": {
            "sessionId": session_id,
            "update": update,
        },
    }


def _write_native_session_home(
    grok_home: Path,
    updates: list[dict[str, object]],
    *,
    session_id: str = _NATIVE_SESSION_ID,
) -> Path:
    session = grok_home / "sessions" / session_id
    session.mkdir(parents=True)
    (session / "updates.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in updates),
        encoding="utf-8",
    )
    (session / "summary.json").write_text(
        json.dumps(
            {
                "info": {"id": session_id},
                "current_model_id": "grok-4.5",
                "grok_home": str(grok_home),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return grok_home


def _write_native_session(
    root: Path,
    updates: list[dict[str, object]],
) -> Path:
    return _write_native_session_home(root / "grok-home", updates)


def _spending_limit_retry(
    *, session_id: str = _NATIVE_SESSION_ID
) -> dict[str, object]:
    return _native_update(
        {
            "sessionUpdate": "retry_state",
            "type": "failed",
            "error_type": "api",
            "message": _SPENDING_LIMIT_MESSAGE,
        },
        session_id=session_id,
    )


def _spending_limit_terminal(
    *,
    message: str = _SPENDING_LIMIT_MESSAGE,
    session_id: str = _NATIVE_SESSION_ID,
) -> dict[str, object]:
    return _native_update(
        {
            "sessionUpdate": "turn_completed",
            "stop_reason": "error",
            "agent_result": message,
        },
        session_id=session_id,
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
        worktree_root=root,
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

    command = _daemon(tmp_path)._build_implementation_command(tmp_path)
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
    assert "codex" not in head


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


def test_native_terminal_spending_limit_authorizes_quota_classification(
    tmp_path: Path,
) -> None:
    retry = _spending_limit_retry()
    assert (
        grok_cli_runner._grok_failure_type_from_stream_event(json.dumps(retry))
        == "spending_limit_exhausted"
    )
    grok_home = _write_native_session(
        tmp_path,
        [
            retry,
            _native_update(
                {
                    "sessionUpdate": "user_message_chunk",
                    "content": "fixed verifier prompt",
                }
            ),
            _spending_limit_terminal(),
        ],
    )

    assert (
        grok_cli_runner._terminal_grok_failure_type_from_isolated_home(
            grok_home,
            expected_session_id=_NATIVE_SESSION_ID,
        )
        == "spending_limit_exhausted"
    )


@pytest.mark.parametrize(
    "retry_update",
    (
        {
            "sessionUpdate": "retry_state",
            "type": "failed",
            "error_type": "api",
            "message": _SPENDING_LIMIT_MESSAGE.replace(
                "personal-team-blocked:spending-limit",
                "personal-team-blocked:other-limit",
            ),
        },
        {
            "sessionUpdate": "retry_state",
            "type": "failed",
            "error_type": "network",
            "message": _SPENDING_LIMIT_MESSAGE,
        },
        {
            "sessionUpdate": "retry_state",
            "type": "pending",
            "error_type": "api",
            "message": _SPENDING_LIMIT_MESSAGE,
        },
    ),
)
def test_native_spending_limit_rejects_nonexact_retry_evidence(
    tmp_path: Path,
    retry_update: dict[str, object],
) -> None:
    grok_home = _write_native_session(
        tmp_path,
        [_native_update(retry_update), _spending_limit_terminal()],
    )

    assert (
        grok_cli_runner._terminal_grok_failure_type_from_isolated_home(
            grok_home,
            expected_session_id=_NATIVE_SESSION_ID,
        )
        == ""
    )


def test_native_spending_limit_requires_matching_terminal_correlation(
    tmp_path: Path,
) -> None:
    grok_home = _write_native_session(
        tmp_path,
        [
            _spending_limit_retry(),
            _spending_limit_terminal(message="different terminal result"),
        ],
    )

    assert (
        grok_cli_runner._terminal_grok_failure_type_from_isolated_home(
            grok_home,
            expected_session_id=_NATIVE_SESSION_ID,
        )
        == ""
    )


def test_spending_limit_prompt_or_projected_error_prose_never_grants_authority() -> None:
    projected_error = json.dumps(
        {"type": "error", "message": _SPENDING_LIMIT_MESSAGE},
        sort_keys=True,
    )
    prompt_chunk = _native_update(
        {
            "sessionUpdate": "user_message_chunk",
            "content": _SPENDING_LIMIT_MESSAGE,
        }
    )

    assert grok_cli_runner._grok_failure_type_from_stream_event(projected_error) == ""
    assert (
        grok_cli_runner._grok_failure_type_from_stream_event(json.dumps(prompt_chunk))
        == ""
    )
    assert grok_cli_runner.parse_grok_quota_error(projected_error) == {}


@pytest.mark.parametrize(
    ("verifier_failure_type", "expected_returncode", "expected_fallback_count"),
    (
        ("spending_limit_exhausted", 0, 1),
        ("usage_pool_exhausted", 23, 0),
    ),
)
def test_main_route_requires_matching_native_spending_limit_before_terra(
    tmp_path: Path,
    monkeypatch,
    capsys,
    verifier_failure_type: str,
    expected_returncode: int,
    expected_fallback_count: int,
) -> None:
    workspace = tmp_path / "workspace"
    provider_bin = tmp_path / "provider-bin"
    workspace.mkdir()
    provider_bin.mkdir()
    grok = provider_bin / "grok"
    codex = provider_bin / "codex"
    for executable in (grok, codex):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)

    fallback = _terra_fallback_command(str(codex), workspace)
    prompt = "repair the failed implementation"
    fallback_calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_primary(command, *, env) -> int:
        session_id = command[command.index("--session-id") + 1]
        _write_native_session_home(
            Path(env["GROK_HOME"]),
            [
                _spending_limit_retry(session_id=session_id),
                _native_update(
                    {
                        "sessionUpdate": "user_message_chunk",
                        "content": prompt,
                    },
                    session_id=session_id,
                ),
                _spending_limit_terminal(session_id=session_id),
            ],
            session_id=session_id,
        )
        return 23

    def fake_fallback(command, **kwargs):
        fallback_calls.append((list(command), dict(kwargs)))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO(prompt))
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_trusted_grok_bin",
        lambda **_kwargs: str(grok),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_select_grok_isolation_backend",
        lambda **_kwargs: grok_cli_runner.GROK_ISOLATION_GROK_SANDBOX,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        fake_primary,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_independently_verify_grok_quota",
        lambda **_kwargs: verifier_failure_type,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_codex_quota_fallback_env",
        lambda **_kwargs: {"PATH": "/usr/bin:/bin"},
    )
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_fallback)
    monkeypatch.chdir(workspace)

    returncode = grok_cli_runner.main(
        [
            "--workspace",
            str(workspace),
            "--grok-bin",
            str(grok),
            "--model",
            "grok-4.5",
            "--codex-fallback-command-json",
            json.dumps(fallback),
        ]
    )

    assert returncode == expected_returncode
    assert len(fallback_calls) == expected_fallback_count
    if fallback_calls:
        command, kwargs = fallback_calls[0]
        assert command == fallback
        assert command[command.index("-m") + 1] == "gpt-5.6-terra"
        assert 'model_reasoning_effort="medium"' in command
        assert kwargs["cwd"] == workspace.resolve()
        assert kwargs["input"] == prompt
        assert kwargs["text"] is True
    else:
        assert "did not confirm the same quota failure" in capsys.readouterr().err


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
    assert "--ephemeral" in fallback
