"""Contracts for typed Grok failure → Codex Terra/high fallback authority."""

from __future__ import annotations

import io
import json
import os
import re
import shlex
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.entrypoints import provider_attempt_store
from ipfs_accelerate_py.agent_supervisor.integrations import (
    llm_merge_resolver_fallback as merge_resolver_fallback,
)
from ipfs_accelerate_py.agent_supervisor.runtime import provider_failure_policy
from ipfs_accelerate_py.agent_supervisor.runtime.provider_failure_policy import (
    GROK_NOT_SIGNED_IN_GUIDANCE,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
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
                "current_model_id": "grok-4.6",
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


def _terra_fallback_command(
    codex: str,
    workspace: str | Path,
    *,
    reasoning_effort: str = "high",
) -> list[str]:
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
        f'model_reasoning_effort="{reasoning_effort}"',
        "-",
    ]


def _seal_auth_or_quota_route(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        implementation_daemon.IMPLEMENTATION_PROVIDER_ENV: "grok_cli",
        implementation_daemon.IMPLEMENTATION_FALLBACK_PROVIDER_ENV: "codex",
        implementation_daemon.IMPLEMENTATION_FALLBACK_TRIGGER_ENV: (
            "primary_quota_or_auth_unavailable"
        ),
        implementation_daemon._GROK_MODEL_ENV: "grok-4.6",
        implementation_daemon._CODEX_MODEL_ENV: "gpt-5.6-terra",
        implementation_daemon._CODEX_REASONING_EFFORT_ENV: "high",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        implementation_daemon,
        "_configured_agent_implementation_route_plan",
        lambda _repo_root: llm_router._AUTH_OR_QUOTA_AGENT_IMPLEMENTATION_ROUTE,
    )


def _install_fake_grok_docker_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    create_returncode: int = 0,
    create_stdout: bytes | None = None,
) -> dict[str, object]:
    workspace = tmp_path / "workspace"
    provider_home = tmp_path / "asref-grok-home-test"
    lease_root = tmp_path / "asref-grok-container-test"
    workspace.mkdir()
    provider_home.mkdir(mode=0o700)
    lease_root.mkdir(mode=0o700)
    docker_config = lease_root / "docker-config"
    docker_config.mkdir(mode=0o700)
    policy_path = provider_home / "sandbox.toml"
    policy_path.write_text("[profiles.test]\n", encoding="utf-8")
    grok = tmp_path / "grok"
    codex = tmp_path / "codex"
    for executable in (grok, codex):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)
    container_id = "d" * 64
    close_calls: list[bool] = []
    create_calls: list[tuple[list[str], dict[str, object]]] = []

    class FakeHome:
        name = str(provider_home)

        def cleanup(self) -> None:
            return None

    class FakeCommandEnvironment:
        wrapper_path = "/opt/provider-command-wrapper"
        contract_sha256 = "sha256:" + "1" * 64
        formal_toolchain_contract_sha256 = "sha256:" + "2" * 64

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    class FakeLease:
        docker_bin = "/usr/bin/docker"
        container_name = "ipfs-accelerate-grok-1-" + "c" * 32

        def close(self, *, docker_run_finished: bool) -> None:
            close_calls.append(docker_run_finished)

    FakeLease.docker_config = docker_config
    FakeLease.cidfile = lease_root / "container.cid"
    FakeLease.lease_root = lease_root

    def fake_create_run(command, **kwargs):
        create_calls.append((list(command), dict(kwargs)))
        return subprocess.CompletedProcess(
            command,
            create_returncode,
            stdout=(
                (container_id + "\n").encode("ascii")
                if create_stdout is None
                else create_stdout
            ),
            stderr=(b"create failed" if create_returncode else b""),
        )

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("implement"))
    monkeypatch.setattr(
        grok_cli_runner,
        "sealed_provider_command_environment",
        lambda *_args, **_kwargs: FakeCommandEnvironment(),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_isolated_grok_home",
        lambda **kwargs: (
            FakeHome(),
            {
                **kwargs["child_env"],
                "GROK_HOME": str(provider_home),
                "HOME": str(provider_home),
            },
            policy_path,
            (provider_home,),
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_select_grok_isolation_backend",
        lambda **_kwargs: grok_cli_runner.GROK_ISOLATION_DOCKER,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_isolation_binary",
        lambda: "/usr/bin/docker",
    )
    monkeypatch.setattr(
        grok_cli_runner._DockerContainerLease,
        "create",
        lambda *_args, **_kwargs: FakeLease(),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_isolation_image_id",
        lambda *_args, **_kwargs: "sha256:" + "e" * 64,
    )
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_create_run)
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_trusted_grok_bin",
        lambda **_kwargs: str(grok),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: str(codex),
    )
    monkeypatch.chdir(workspace)
    return {
        "workspace": workspace,
        "grok": grok,
        "codex": codex,
        "container_id": container_id,
        "close_calls": close_calls,
        "create_calls": create_calls,
    }


def test_daemon_auth_or_quota_route_embeds_strict_terra_high_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _seal_auth_or_quota_route(monkeypatch)
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
    monkeypatch.delenv(
        implementation_daemon.PRODUCTION_PROVIDER_ROUTE_ENABLED_ENV, raising=False
    )
    monkeypatch.delenv(
        implementation_daemon.PRODUCTION_PROVIDER_ALLOW_RAW_COMMAND_ENV, raising=False
    )
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: False)
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

    daemon = _daemon(tmp_path)
    daemon._require_primary_provider_readiness(None)
    command = daemon._build_implementation_command(tmp_path)
    assert "--codex-fallback-command-json" in command
    fallback = json.loads(command[command.index("--codex-fallback-command-json") + 1])
    assert fallback[0] == "/opt/providers/codex"
    assert fallback[1] == "exec"
    assert "--ignore-user-config" in fallback
    assert "--ignore-rules" in fallback
    assert "--ephemeral" in fallback
    assert fallback[fallback.index("-s") + 1] == "workspace-write"
    assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="high"' in fallback
    nonce = command[command.index("--grok-failure-receipt-nonce") + 1]
    assert len(nonce) == 64
    assert set(nonce) <= set("0123456789abcdef")
    head = " ".join(command[: command.index("--codex-fallback-command-json")])
    assert "codex" not in head


@pytest.mark.parametrize("override_source", ("constructor", "environment"))
def test_auth_or_quota_route_rejects_raw_command_override_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override_source: str,
) -> None:
    _seal_auth_or_quota_route(monkeypatch)
    command = "codex exec -"
    if override_source == "environment":
        monkeypatch.setenv("IMPLEMENTATION_DAEMON_COMMAND", command)
        constructor_command = ""
    else:
        monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
        constructor_command = command
    board = tmp_path / "tasks.todo.md"
    board.write_text("# Tasks\n", encoding="utf-8")
    daemon = TodoImplementationDaemon(
        todo_path=board,
        state_path=tmp_path / "state" / "task-state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
        worktree_root=tmp_path,
        implementation_command=constructor_command,
    )

    with pytest.raises(
        implementation_daemon.ImplementationRetryDeferred,
        match="sealed Grok/Codex route rejects",
    ):
        daemon._require_primary_provider_readiness(None)
    with pytest.raises(
        implementation_daemon.ImplementationRetryDeferred,
        match="sealed Grok/Codex route rejects",
    ):
        daemon._build_implementation_command(tmp_path)


def test_auth_or_quota_route_keeps_explicit_grok_task_grok_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seal_auth_or_quota_route(monkeypatch)
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
    task = PortalTask(
        task_id="AUTH-ROUTE-001",
        title="Stay on Grok",
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


def test_auth_or_quota_route_denies_independent_codex_review_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seal_auth_or_quota_route(monkeypatch)
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: f"/opt/providers/{name}",
    )
    task = PortalTask(
        task_id="AUTH-ROUTE-REVIEW-001",
        title="Keep review independent",
        status="ready",
        completion="manual",
        priority="P0",
        track="provider",
        outputs=["src/provider.py"],
        metadata={"Provider role": "codex-review"},
    )
    daemon = _daemon(tmp_path)

    for action in (
        lambda: daemon._require_primary_provider_readiness(task),
        lambda: daemon._build_implementation_command(tmp_path, task=task),
    ):
        with pytest.raises(
            implementation_daemon.ImplementationRetryDeferred,
            match="requires independent Codex review",
        ):
            action()


def test_typed_preflight_process_uses_its_isolated_workspace_as_os_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    probe_home = tmp_path / "isolated-grok-home"

    class FakeIsolatedHome:
        name = str(probe_home)

        def cleanup(self) -> None:
            return None

    def fake_isolated_home(**kwargs):
        probe_home.mkdir(mode=0o700)
        captured["isolated_workspace"] = kwargs["workspace"]
        return (
            FakeIsolatedHome(),
            {"GROK_HOME": str(probe_home)},
            probe_home / "settings.json",
            (),
        )

    def fake_probe(command, *, env, cwd):
        captured["command"] = list(command)
        captured["env"] = dict(env)
        captured["cwd"] = cwd
        prompt_path = Path(command[command.index("--prompt-file") + 1])
        captured["prompt"] = prompt_path.read_text(encoding="utf-8")
        return 41, "Error: Not signed in", len("Error: Not signed in"), False

    monkeypatch.setattr(
        grok_cli_runner,
        "_isolated_grok_home",
        fake_isolated_home,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_isolated_grok_quota_probe",
        fake_probe,
    )

    returncode, receipt, overflow = grok_cli_runner._run_typed_grok_preflight(
        grok_bin="/usr/local/bin/grok",
        base_env={},
        nonce="a" * 64,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert returncode == 41
    assert overflow is False
    assert receipt["failure_class"] == "authentication_unavailable"
    assert captured["cwd"] == captured["isolated_workspace"]
    assert captured["env"]["PWD"] == str(captured["cwd"])
    assert "OLDPWD" not in captured["env"]
    assert command[command.index("--cwd") + 1] == str(captured["cwd"])
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--max-turns") + 1] == "1"
    assert command[command.index("--permission-mode") + 1] == "dontAsk"
    assert captured["prompt"] == grok_cli_runner.GROK_QUOTA_PROBE_PROMPT
    assert not Path(captured["cwd"]).exists()


def _typed_preflight_attempt(
    stderr_text: str,
    *,
    nonce: str = "a" * 64,
    returncode: int = 41,
) -> tuple[int, dict[str, object], bool, str]:
    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text=stderr_text,
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=returncode,
        primary_dispatched=False,
    )
    return returncode, receipt, False, stderr_text


def test_typed_preflight_retries_exact_max_turns_artifact_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    results = iter(
        (
            _typed_preflight_attempt("Error: max turns reached\n"),
            (0, {}, False, ""),
        )
    )

    def fake_attempt(**kwargs):
        calls.append(dict(kwargs))
        return next(results)

    monkeypatch.setattr(
        grok_cli_runner,
        "_run_typed_grok_preflight_once",
        fake_attempt,
    )

    result = grok_cli_runner._run_typed_grok_preflight(
        grok_bin="/usr/local/bin/grok",
        base_env={},
        nonce="a" * 64,
    )

    assert result == (0, {}, False)
    assert len(calls) == 2
    assert calls[0] == calls[1]


@pytest.mark.parametrize(
    "stderr_text",
    (
        "Error: max turns reached",
        "error: max turns reached\n",
        "Error: max turns reached\n\n",
        "Error: Not signed in",
        "Grok Build usage balance exhausted",
    ),
)
def test_typed_preflight_does_not_retry_near_match_auth_or_quota(
    monkeypatch: pytest.MonkeyPatch,
    stderr_text: str,
) -> None:
    calls = 0
    attempt = _typed_preflight_attempt(stderr_text)

    def fake_attempt(**_kwargs):
        nonlocal calls
        calls += 1
        return attempt

    monkeypatch.setattr(
        grok_cli_runner,
        "_run_typed_grok_preflight_once",
        fake_attempt,
    )

    result = grok_cli_runner._run_typed_grok_preflight(
        grok_bin="/usr/local/bin/grok",
        base_env={},
        nonce="a" * 64,
    )

    assert result == attempt[:3]
    assert calls == 1


def test_repeated_exact_max_turns_is_one_unknown_denial_without_terra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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

    nonce = "a" * 64
    attempt = _typed_preflight_attempt(
        "Error: max turns reached\n",
        nonce=nonce,
    )
    calls = 0

    def fake_attempt(**_kwargs):
        nonlocal calls
        calls += 1
        return attempt

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("implement"))
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_trusted_grok_bin",
        lambda **_kwargs: str(grok),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_typed_grok_preflight_once",
        fake_attempt,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_workspace_content_fingerprint",
        lambda _workspace: "clean",
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_repository_head",
        lambda _workspace: "b" * 40,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_independently_verify_grok_quota",
        lambda **_kwargs: pytest.fail("unknown evidence must not run verifier"),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_codex_quota_fallback_in_docker",
        lambda *_args, **_kwargs: pytest.fail("Terra must remain forbidden"),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_select_grok_isolation_backend",
        lambda **_kwargs: pytest.fail("task Grok must not run after denial"),
    )
    route_plan = llm_router._AUTH_OR_QUOTA_AGENT_IMPLEMENTATION_ROUTE
    monkeypatch.setattr(
        llm_router,
        "resolve_agent_implementation_route_binding",
        lambda *_args, **_kwargs: route_plan,
    )

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(workspace),
            "--grok-bin",
            str(grok),
            "--model",
            "grok-4.6",
            "--codex-fallback-command-json",
            json.dumps(_terra_fallback_command(str(codex), workspace)),
            "--grok-failure-receipt-nonce",
            nonce,
            "--agent-implementation-route-json",
            json.dumps(route_plan.as_binding_dict()),
        ]
    )

    assert result == 41
    assert calls == 2
    rendered = capsys.readouterr().err
    assert rendered.count(grok_cli_runner.GROK_FAILURE_RECEIPT_PREFIX) == 1
    outcomes = provider_failure_policy.extract_grok_route_outcomes(rendered)
    assert len(outcomes) == 1
    assert outcomes[0]["decision"] == "denied"
    assert outcomes[0]["failure_class"] == "unknown"
    assert outcomes[0]["fallback_dispatched"] is False


def test_scoped_route_rejects_prompt_cid_before_grok_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    grok = tmp_path / "grok"
    codex = tmp_path / "codex"
    for executable in (grok, codex):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)
    invocation = SimpleNamespace(
        prompt_cid=grok_cli_runner._agent_prompt_cid("signed prompt"),
        control_plane=object(),
        provider_attempt_store=str(tmp_path / "attempt-store"),
        provider_attempt_store_identity="sha256:" + "a" * 64,
        logical_attempt_id="sha256:" + "b" * 64,
    )
    route_plan = SimpleNamespace(
        invocation_binding=invocation,
        fallback_reasoning_effort="high",
    )

    class EmptyAttemptStore:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def read(self, _logical_attempt_id: str) -> None:
            return None

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("wrong prompt"))
    monkeypatch.setattr(grok_cli_runner.sys, "argv", ["/proc/self/fd/71"])
    monkeypatch.setattr(
        llm_router,
        "resolve_agent_implementation_route_binding",
        lambda *_args, **_kwargs: route_plan,
    )
    monkeypatch.setattr(
        llm_router,
        "verify_agent_implementation_sealed_control_plane",
        lambda _pin, _descriptor: "/proc/self/fd/71",
    )
    monkeypatch.setattr(
        provider_attempt_store,
        "DurableProviderAttemptCAS",
        EmptyAttemptStore,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: str(codex),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_trusted_grok_bin",
        lambda **_kwargs: str(grok),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_repository_head",
        lambda _workspace: "c" * 40,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_typed_grok_preflight",
        lambda **_kwargs: pytest.fail(
            "prompt CID mismatch must stop before Grok preflight"
        ),
    )

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(workspace),
            "--grok-bin",
            str(grok),
            "--model",
            "grok-4.6",
            "--codex-fallback-command-json",
            json.dumps(_terra_fallback_command(str(codex), workspace)),
            "--grok-failure-receipt-nonce",
            "a" * 64,
            "--agent-implementation-route-json",
            "{}",
        ]
    )

    assert result == 2
    assert "does not match the task prompt" in capsys.readouterr().err


def test_typed_preflight_probe_overflow_is_measured_fail_closed(
    tmp_path: Path,
) -> None:
    guidance = GROK_NOT_SIGNED_IN_GUIDANCE.encode("utf-8")
    code = (
        "import os\n"
        "payload = b'HTTP 429\\n' + "
        f"(b'x' * {grok_cli_runner.MAX_GROK_FAILURE_EVIDENCE_BYTES + 4096}) + "
        f"b'\\n' + {guidance!r}\n"
        "os.write(2, payload)\n"
        "raise SystemExit(41)\n"
    )

    returncode, retained, evidence_size, overflow = (
        grok_cli_runner._run_isolated_grok_quota_probe(
            [grok_cli_runner.sys.executable, "-c", code],
            env=os.environ.copy(),
            cwd=tmp_path,
        )
    )
    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text=retained,
        nonce="f" * 64,
        model="grok-4.6",
        probe_returncode=returncode,
        primary_dispatched=False,
        evidence_size=evidence_size,
        evidence_overflow=overflow,
    )

    assert returncode == 41
    assert evidence_size > grok_cli_runner.MAX_GROK_FAILURE_EVIDENCE_BYTES
    assert overflow is True
    assert receipt["evidence_overflow"] is True
    assert len(retained.encode("utf-8")) <= (
        grok_cli_runner.MAX_GROK_FAILURE_EVIDENCE_BYTES
    )


def test_independent_quota_verifier_uses_isolated_os_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    verifier_home = tmp_path / "verifier-home"

    class FakeIsolatedHome:
        name = str(verifier_home)

        def cleanup(self) -> None:
            return None

    def fake_isolated_home(**_kwargs):
        verifier_home.mkdir(mode=0o700)
        return (
            FakeIsolatedHome(),
            {"GROK_HOME": str(verifier_home), "OLDPWD": "/task/worktree"},
            verifier_home / "settings.json",
            (),
        )

    def fake_run(command, **kwargs):
        captured["cwd"] = kwargs["cwd"]
        captured["env"] = dict(kwargs["env"])
        session_id = command[command.index("--session-id") + 1]
        _write_native_session_home(
            verifier_home,
            [
                _spending_limit_retry(session_id=session_id),
                _spending_limit_terminal(session_id=session_id),
            ],
            session_id=session_id,
        )
        return subprocess.CompletedProcess(command, 23)

    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text="Grok Build usage balance exhausted",
        nonce="1" * 64,
        model="grok-4.6",
        probe_returncode=41,
        primary_dispatched=False,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_isolated_grok_home",
        fake_isolated_home,
    )
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)

    evidence = grok_cli_runner._independently_verify_grok_quota(
        grok_bin="/opt/providers/grok",
        base_env={},
        failure_receipt=receipt,
    )

    assert evidence is not None
    assert captured["env"]["PWD"] == str(captured["cwd"])
    assert "OLDPWD" not in captured["env"]


def test_quota_fallback_command_rejects_model_or_effort_drift() -> None:
    valid = _terra_fallback_command("/usr/local/bin/codex", "/repo")
    assert grok_cli_runner._parse_codex_fallback_command(json.dumps(valid)) == valid
    model_drift = list(valid)
    model_drift[model_drift.index("-m") + 1] = "gpt-5.6-sol"
    effort_drift = list(valid)
    effort_idx = next(
        i for i, item in enumerate(effort_drift) if "model_reasoning_effort=" in item
    )
    effort_drift[effort_idx] = 'model_reasoning_effort="low"'
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
            "Grok implementation failed\n"
            "usage balance exhausted\n"
            "402 Payment Required"
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


def test_direct_no_nonce_native_quota_cannot_cross_providers(
    tmp_path: Path,
    monkeypatch,
    capsys,
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

    fallback = _terra_fallback_command(
        str(codex),
        workspace,
        reasoning_effort="medium",
    )
    prompt = "repair the failed implementation"

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
        lambda **_kwargs: pytest.fail(
            "direct no-nonce route must not run a fallback verifier"
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_codex_quota_fallback_in_docker",
        lambda *_args, **_kwargs: pytest.fail(
            "direct no-nonce route must not cross providers"
        ),
    )
    monkeypatch.chdir(workspace)

    returncode = grok_cli_runner.main(
        [
            "--workspace",
            str(workspace),
            "--grok-bin",
            str(grok),
            "--model",
            "grok-4.6",
            "--codex-fallback-command-json",
            json.dumps(fallback),
        ]
    )

    assert returncode == 23
    assert "Direct no-nonce Grok failure cannot authorize" in capsys.readouterr().err


def test_merge_resolver_marker_mints_fresh_legacy_preflight_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: str(codex),
    )
    wrapper_command = shlex.split(
        merge_resolver_fallback.llm_merge_resolver_fallback_command(
            python_executable="python-test"
        )
    )
    assert wrapper_command == [
        "python-test",
        "-m",
        (
            "ipfs_accelerate_py.agent_supervisor.integrations."
            "llm_merge_resolver_fallback"
        ),
    ]
    prompt = "resolve the merge conflict"
    preflight_nonces: list[str] = []
    fallback_calls: list[list[str]] = []

    def fake_preflight(**kwargs):
        nonce = str(kwargs["nonce"])
        preflight_nonces.append(nonce)
        receipt = grok_cli_runner.build_grok_failure_receipt(
            probe_stderr_text="Grok Build usage balance exhausted",
            nonce=nonce,
            model="grok-4.6",
            probe_returncode=41,
            primary_dispatched=False,
        )
        return 41, receipt, False

    def fake_verifier(**kwargs) -> object:
        verifier_home = _write_native_session(
            tmp_path / "legacy-independent-verifier",
            [_spending_limit_retry(), _spending_limit_terminal()],
        )
        return llm_router.validate_agent_implementation_quota_evidence(
            grok_home=verifier_home,
            expected_session_id=_NATIVE_SESSION_ID,
            verifier_returncode=41,
            failure_receipt=kwargs["failure_receipt"],
        )

    def fake_fallback(command, **kwargs) -> int:
        fallback_calls.append(list(command))
        kwargs["pre_effect_validator"]()
        return 0

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO(prompt))
    monkeypatch.setenv("AGENT_RESOLVER_LOCK_BYPASS", "1")
    monkeypatch.setenv(merge_resolver_fallback._INVOCATION_DEPTH_ENV, "0")
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_trusted_grok_bin",
        lambda **_kwargs: str(grok),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_typed_grok_preflight",
        fake_preflight,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_independently_verify_grok_quota",
        fake_verifier,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_workspace_content_fingerprint",
        lambda _workspace: "clean",
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_repository_head",
        lambda _workspace: "a" * 40,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        lambda *_args, **_kwargs: pytest.fail(
            "legacy fallback must run before the task Grok process"
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_codex_quota_fallback_in_docker",
        fake_fallback,
    )

    assert merge_resolver_fallback.main([str(workspace)]) == 0
    assert len(preflight_nonces) == 1
    assert re.fullmatch(r"[0-9a-f]{64}", preflight_nonces[0])
    assert len(fallback_calls) == 1
    assert 'model_reasoning_effort="medium"' in fallback_calls[0]
    assert "quota is exhausted" in capsys.readouterr().err


@pytest.mark.parametrize(
    "external_route_arguments",
    (
        ("--grok-failure-receipt-nonce", "a" * 64),
        ("--agent-implementation-route-json", "{}"),
    ),
)
def test_legacy_preflight_marker_rejects_external_route_authority(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    external_route_arguments: tuple[str, str],
) -> None:
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex.chmod(0o700)
    assert (
        grok_cli_runner.main(
            [
                "--workspace",
                str(tmp_path),
                "--codex-fallback-command-json",
                json.dumps(
                    _terra_fallback_command(
                        str(codex),
                        tmp_path,
                        reasoning_effort="medium",
                    )
                ),
                grok_cli_runner.CANONICAL_LEGACY_PREFLIGHT_ROUTE_FLAG,
                *external_route_arguments,
            ]
        )
        == 2
    )
    error = capsys.readouterr().err
    if external_route_arguments[0] == "--grok-failure-receipt-nonce":
        assert "cannot be combined with an external nonce" in error
    else:
        assert "legacy quota route forbids an auth/high route binding" in error


@pytest.mark.parametrize(
    (
        "probe_stderr",
        "fingerprints",
        "verifier_failure_type",
        "expected_returncode",
        "expected_fallback_count",
        "expected_verifier_count",
    ),
    (
        (
            "Error: Not signed in",
            ("clean", "clean", "clean", "clean"),
            "",
            0,
            1,
            0,
        ),
        (
            GROK_NOT_SIGNED_IN_GUIDANCE,
            ("clean", "clean", "clean", "clean"),
            "",
            0,
            1,
            0,
        ),
        ("Error: Not signed in", ("clean", "mutated"), "", 41, 0, 0),
        ("HTTP 429 Too Many Requests", ("clean",), "", 41, 0, 0),
        ("HTTP 403", ("clean",), "", 41, 0, 1),
        ("Forbidden", ("clean",), "", 41, 0, 1),
        ("Not signed in\nHTTP 429", ("clean",), "", 41, 0, 1),
        ("Not signed in\nHTTP 403", ("clean",), "", 41, 0, 1),
        ("Forbidden\nNot signed in", ("clean",), "", 41, 0, 1),
        (
            GROK_NOT_SIGNED_IN_GUIDANCE + "\nHTTP 429",
            ("clean",),
            "",
            41,
            0,
            1,
        ),
        (
            "HTTP 429\n" + GROK_NOT_SIGNED_IN_GUIDANCE,
            ("clean",),
            "",
            41,
            0,
            1,
        ),
        (
            "Not signed in\nGrok Build usage balance exhausted",
            ("clean",),
            "",
            41,
            0,
            1,
        ),
        (
            _SPENDING_LIMIT_MESSAGE,
            ("clean", "clean", "clean", "clean"),
            "spending_limit_exhausted",
            0,
            1,
            1,
        ),
        (
            "Grok Build usage balance exhausted",
            ("clean", "clean", "clean", "clean"),
            "spending_limit_exhausted",
            0,
            1,
            1,
        ),
        (
            "Grok Build usage balance exhausted",
            ("clean",),
            "",
            41,
            0,
            1,
        ),
    ),
)
def test_typed_preflight_requires_independent_quota_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    probe_stderr: str,
    fingerprints: tuple[str, ...],
    verifier_failure_type: str,
    expected_returncode: int,
    expected_fallback_count: int,
    expected_verifier_count: int,
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

    nonce = "a" * 64
    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text=probe_stderr,
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=41,
        primary_dispatched=False,
    )
    fallback = _terra_fallback_command(str(codex), workspace)
    fallback_calls: list[list[str]] = []
    verifier_calls: list[dict[str, object]] = []
    preflight_calls: list[dict[str, object]] = []
    fingerprint_values = iter(fingerprints)
    route_plan = llm_router._AUTH_OR_QUOTA_AGENT_IMPLEMENTATION_ROUTE

    class PreflightOrderedStdin(io.StringIO):
        def read(self, *args, **kwargs) -> str:
            assert preflight_calls, "task prompt was read before Grok preflight"
            return super().read(*args, **kwargs)

    monkeypatch.setattr(
        grok_cli_runner.sys,
        "stdin",
        PreflightOrderedStdin("implement"),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_trusted_grok_bin",
        lambda **_kwargs: str(grok),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_select_grok_isolation_backend",
        lambda **_kwargs: pytest.fail(
            "authorized typed preflight must skip task-Grok isolation"
        ),
    )
    def fake_preflight(**kwargs):
        preflight_calls.append(dict(kwargs))
        return 41, receipt, False

    monkeypatch.setattr(
        grok_cli_runner,
        "_run_typed_grok_preflight",
        fake_preflight,
    )

    def fake_verifier(**kwargs) -> object:
        verifier_calls.append(dict(kwargs))
        if verifier_failure_type != "spending_limit_exhausted":
            return None
        verifier_home = _write_native_session(
            tmp_path / "independent-verifier",
            [_spending_limit_retry(), _spending_limit_terminal()],
        )
        return llm_router.validate_agent_implementation_quota_evidence(
            grok_home=verifier_home,
            expected_session_id=_NATIVE_SESSION_ID,
            verifier_returncode=41,
            failure_receipt=kwargs["failure_receipt"],
        )

    monkeypatch.setattr(
        grok_cli_runner,
        "_independently_verify_grok_quota",
        fake_verifier,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_workspace_content_fingerprint",
        lambda _workspace: next(fingerprint_values),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        lambda *_args, **_kwargs: pytest.fail(
            "task Grok must not run after a failed typed preflight"
        ),
    )

    def fake_fallback(command, **kwargs) -> int:
        fallback_calls.append(list(command))
        kwargs["pre_effect_validator"]()
        return 0

    monkeypatch.setattr(
        grok_cli_runner,
        "_run_codex_quota_fallback_in_docker",
        fake_fallback,
    )
    monkeypatch.setattr(
        llm_router,
        "resolve_agent_implementation_route_binding",
        lambda *_args, **_kwargs: route_plan,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_repository_head",
        lambda _workspace: "b" * 40,
    )

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(workspace),
            "--grok-bin",
            str(grok),
            "--model",
            "grok-4.6",
            "--codex-fallback-command-json",
            json.dumps(fallback),
            "--grok-failure-receipt-nonce",
            nonce,
            "--agent-implementation-route-json",
            json.dumps(route_plan.as_binding_dict()),
        ]
    )

    assert result == expected_returncode
    assert len(fallback_calls) == expected_fallback_count
    assert len(verifier_calls) == expected_verifier_count
    rendered = capsys.readouterr().err
    assert grok_cli_runner.GROK_FAILURE_RECEIPT_PREFIX in rendered
    if expected_fallback_count:
        assert 'model_reasoning_effort="high"' in fallback_calls[0]
        expected_reason = (
            "authentication is unavailable"
            if receipt["failure_class"] == "authentication_unavailable"
            else "quota is exhausted"
        )
        assert expected_reason in rendered
    else:
        assert "Codex fallback is forbidden" in rendered


def test_terminal_route_outcome_is_bound_to_receipt_route_and_runner_exit() -> None:
    nonce = "c" * 64
    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text="Grok Build usage balance exhausted",
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=41,
        primary_dispatched=False,
    )
    route = llm_router.resolve_agent_implementation_route(
        default_route="legacy"
    ).as_binding_dict()
    evidence_id = "sha256:" + "d" * 64
    cases = (
        (
            provider_failure_policy.build_grok_route_outcome(
                receipt=receipt,
                route_plan=route,
                quota_evidence_id=evidence_id,
                decision="fallback_succeeded",
                verifier_status="confirmed_quota",
                fallback_dispatched=True,
                fallback_returncode=0,
            ),
            0,
        ),
        (
            provider_failure_policy.build_grok_route_outcome(
                receipt=receipt,
                route_plan=route,
                quota_evidence_id=evidence_id,
                decision="fallback_failed",
                verifier_status="confirmed_quota",
                fallback_dispatched=True,
                fallback_returncode=17,
            ),
            17,
        ),
        (
            provider_failure_policy.build_grok_route_outcome(
                receipt=receipt,
                route_plan=route,
                decision="denied",
                verifier_status="not_confirmed",
                fallback_dispatched=False,
                fallback_returncode=None,
            ),
            41,
        ),
    )

    for outcome, runner_returncode in cases:
        assert provider_failure_policy.valid_grok_route_outcome(
            outcome,
            receipt=receipt,
            route_plan=route,
            runner_returncode=runner_returncode,
        )
        rendered = provider_failure_policy.render_grok_route_outcome(outcome)
        assert provider_failure_policy.extract_grok_route_outcomes(
            rendered + "\n"
        ) == (outcome,)

    tampered = dict(cases[0][0])
    tampered["fallback_returncode"] = 9
    assert not provider_failure_policy.valid_grok_route_outcome(
        tampered,
        receipt=receipt,
        route_plan=route,
        runner_returncode=0,
    )
    duplicate_log = "\n".join(
        provider_failure_policy.render_grok_route_outcome(cases[1][0])
        for _ in range(2)
    )
    assert len(
        provider_failure_policy.extract_grok_route_outcomes(duplicate_log)
    ) == 2
    assert provider_failure_policy.extract_grok_route_outcomes(
        provider_failure_policy.GROK_ROUTE_OUTCOME_PREFIX
        + '{"schema":"first","schema":"last"}\n'
    ) == ()
    assert provider_failure_policy.extract_grok_failure_receipts(
        provider_failure_policy.GROK_FAILURE_RECEIPT_PREFIX
        + '{"schema":"first","schema":"last"}\n'
    ) == ()


@pytest.mark.parametrize("outcome_case", ("valid", "missing", "forged", "duplicate"))
def test_nonce_route_nonzero_never_restores_provider_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome_case: str,
) -> None:
    classifier_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        implementation_daemon,
        "classify_provider_capacity_failure",
        lambda *args, **_kwargs: classifier_calls.append(args),
    )
    nonce = "e" * 64
    route_plan = llm_router._AUTH_OR_QUOTA_AGENT_IMPLEMENTATION_ROUTE
    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text="Error: Not signed in",
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=41,
        primary_dispatched=False,
    )
    outcome = provider_failure_policy.build_grok_route_outcome(
        receipt=receipt,
        route_plan=route_plan.as_binding_dict(),
        decision="fallback_failed",
        verifier_status="not_required_exact_auth",
        fallback_dispatched=True,
        fallback_returncode=17,
    )
    if outcome_case == "forged":
        outcome = {**outcome, "fallback_returncode": 16}
    records = [provider_failure_policy.render_grok_failure_receipt(receipt)]
    if outcome_case != "missing":
        records.append(provider_failure_policy.render_grok_route_outcome(outcome))
    if outcome_case == "duplicate":
        records.append(provider_failure_policy.render_grok_route_outcome(outcome))
    log_path = tmp_path / "route.log"
    log_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        implementation_daemon,
        "resolve_agent_implementation_route_binding",
        lambda *_args, **_kwargs: route_plan,
    )
    command = [
        "/usr/bin/python3",
        "grok_cli_runner.py",
        "--model",
        "grok-4.6",
        "--grok-failure-receipt-nonce",
        nonce,
        "--agent-implementation-route-json",
        json.dumps(route_plan.as_binding_dict()),
    ]

    capacity = _daemon(tmp_path)._provider_capacity_failure_from_log(
        log_path,
        command=command,
        returncode=17,
    )

    assert capacity["exhausted"] is False
    assert capacity["providers"] == []
    assert classifier_calls == []
    assert route_plan.invocation_binding is None
    assert "route_outcome_id" not in capacity


def test_legacy_non_route_capacity_classification_remains_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_classifier = (
        implementation_daemon.classify_provider_capacity_failure
    )
    classifier_calls: list[tuple[object, ...]] = []

    def classify(*args: object, **kwargs: object) -> dict[str, object]:
        classifier_calls.append(args)
        return original_classifier(*args, **kwargs)

    monkeypatch.setattr(
        implementation_daemon,
        "classify_provider_capacity_failure",
        classify,
    )
    log_path = tmp_path / "legacy-capacity.log"
    log_path.write_text(
        "Grok quota exhausted before implementation\n",
        encoding="utf-8",
    )

    capacity = _daemon(tmp_path)._provider_capacity_failure_from_log(
        log_path,
        command=["/usr/local/bin/grok"],
        returncode=23,
    )

    assert capacity["exhausted"] is True
    assert capacity["providers"] == ["grok"]
    assert len(classifier_calls) == 1


def test_docker_codex_boundary_transforms_only_validated_sandbox(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    provider_bin = tmp_path / "provider-bin"
    docker_config = tmp_path / "docker-config"
    workspace.mkdir()
    provider_bin.mkdir()
    docker_config.mkdir()
    codex = provider_bin / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex.chmod(0o700)
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}\n", encoding="utf-8")
    source_auth.chmod(0o600)
    fallback = _terra_fallback_command(str(codex), workspace)
    image = grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID
    container_name = "ipfs-accelerate-codex-1-" + "b" * 32
    child_env = grok_cli_runner._codex_task_container_environment()

    command = grok_cli_runner._docker_codex_fallback_command(
        codex_command=fallback,
        workspace=workspace,
        source_auth=source_auth,
        child_env=child_env,
        docker_config=docker_config,
        container_name=container_name,
        cidfile=tmp_path / "container.cid",
        docker_bin="/usr/bin/docker",
        isolation_image=image,
    )

    assert fallback[fallback.index("-s") + 1] == "workspace-write"
    assert command[0] == "/usr/bin/docker"
    assert f"--host={grok_cli_runner._DOCKER_LOCAL_HOST}" in command
    assert "--pull=never" in command
    assert "--read-only" in command
    assert "--network=bridge" in command
    assert "--runtime=runc" in command
    assert "--entrypoint=/usr/bin/env" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--device" not in command
    assert "ipfs_accelerate.codex_fallback_isolation=true" in command
    assert image in command
    docker_env = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--env"
    ]
    assert docker_env == list(grok_cli_runner._CODEX_DOCKER_IMAGE_ENV_OVERRIDES)
    assert "NVIDIA_VISIBLE_DEVICES=void" in docker_env
    assert "BASH_ENV=" in docker_env
    assert "ENV=" in docker_env

    mounts = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--mount"
    ]
    writable_mounts = [mount for mount in mounts if "readonly" not in mount]
    assert writable_mounts == [
        f"type=bind,src={workspace},dst={workspace}"
    ]
    assert "type=bind,src=/usr,dst=/usr,readonly" in mounts
    assert (
        "type=bind,src=/etc/ssl/certs,dst=/etc/ssl/certs,readonly" in mounts
    )
    assert (
        f"type=bind,src={grok_cli_runner._HOST_CODEX_TASK_TOOLCHAIN_PYTHON},"
        f"dst={grok_cli_runner._CODEX_TASK_TOOLCHAIN_PYTHON},readonly"
        in mounts
    )
    assert (
        f"type=bind,src={source_auth},"
        f"dst={grok_cli_runner._CODEX_CONTAINER_AUTH_PATH},readonly"
        in mounts
    )
    assert not any("/var/run/docker.sock" in mount for mount in mounts)
    assert not any("/home/" in mount for mount in mounts)

    inner = command[command.index(image) + 1 :]
    expected_inner = list(fallback)
    expected_inner[expected_inner.index("-s") + 1] = "danger-full-access"
    expected_environment = [
        f"{name}={value}" for name, value in sorted(child_env.items())
    ]
    assert inner == ["-i", *expected_environment, *expected_inner]
    assert not any("/home/barberb" in item for item in command)
    assert "--dangerously-bypass-approvals-and-sandbox" not in inner


@pytest.mark.parametrize("valid_label", (True, False))
def test_codex_task_toolchain_image_requires_exact_identity_and_label(
    tmp_path: Path,
    monkeypatch,
    valid_label: bool,
) -> None:
    expected_line = (
        f"{grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID}|linux|arm64|"
        f"{grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_LABEL}"
    )
    observed_commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        observed_commands.append(list(command))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                expected_line
                if valid_label
                else expected_line.removesuffix(
                    grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_LABEL
                )
                + "untrusted"
            ),
        )

    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)

    resolved = grok_cli_runner._docker_codex_task_toolchain_image_id(
        "/usr/bin/docker",
        docker_config=tmp_path,
    )

    assert resolved == (
        grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID if valid_label else ""
    )
    assert observed_commands[0][-1] == (
        grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID
    )


@pytest.mark.parametrize(
    "invalid_case",
    ("symlink", "group_readable", "hardlink", "wrong_owner"),
)
def test_codex_auth_boundary_rejects_ambient_or_mutable_authority(
    tmp_path: Path,
    monkeypatch,
    invalid_case: str,
) -> None:
    workspace = tmp_path / "workspace"
    codex_home = tmp_path / "codex-home"
    workspace.mkdir()
    codex_home.mkdir()
    auth_path = codex_home / "auth.json"
    auth_path.write_text("{}\n", encoding="utf-8")
    auth_path.chmod(0o600)

    if invalid_case == "symlink":
        target = tmp_path / "real-auth.json"
        target.write_text("{}\n", encoding="utf-8")
        target.chmod(0o600)
        auth_path.unlink()
        auth_path.symlink_to(target)
    elif invalid_case == "group_readable":
        auth_path.chmod(0o640)
    elif invalid_case == "hardlink":
        os.link(auth_path, tmp_path / "auth-alias.json")
    else:
        current_uid = os.getuid()
        monkeypatch.setattr(
            grok_cli_runner.os,
            "getuid",
            lambda: current_uid + 1,
        )

    with pytest.raises(ValueError, match="private, owned, regular"):
        grok_cli_runner._codex_quota_fallback_env(
            workspace=workspace,
            base_env={
                "HOME": str(tmp_path),
                "CODEX_HOME": str(codex_home),
            },
        )


def test_codex_isolated_home_seals_environment_without_host_toolchain(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    codex_home = tmp_path / "codex-home"
    workspace.mkdir()
    codex_home.mkdir()
    auth_path = codex_home / "auth.json"
    auth_path.write_text("{}\n", encoding="utf-8")
    auth_path.chmod(0o600)

    temporary_home, environment, source_auth = (
        grok_cli_runner._isolated_codex_quota_fallback_home(
            workspace=workspace,
            base_env={
                "CODEX_HOME": str(codex_home),
                "BASH_ENV": "/workspace/untrusted-hook",
                "PYTHONPATH": "/home/barberb/.local/lib/python3.12/site-packages",
            },
        )
    )
    try:
        assert environment == (
            grok_cli_runner._codex_task_container_environment()
        )
        assert source_auth == auth_path
        assert not any("/home/barberb" in value for value in environment.values())
        assert not any(Path(temporary_home.name).iterdir())
    finally:
        grok_cli_runner._robust_remove_runner_temp_tree(
            Path(temporary_home.name)
        )
        temporary_home.cleanup()


def test_grok_docker_create_binds_exact_id_to_attached_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_id = "d" * 64
    docker_environment = {"PATH": "/usr/bin"}
    create_command = ["/usr/bin/docker", "create", "sealed-grok"]
    calls: list[tuple[list[str], dict[str, object]]] = []

    class FakeLease:
        docker_bin = "/usr/bin/docker"
        docker_config = tmp_path / "docker-config"

    def fake_run(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(container_id + "\n").encode("ascii"),
            stderr=b"",
        )

    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)

    start_command = (
        grok_cli_runner._create_grok_container_and_build_start_command(
            create_command,
            workspace=tmp_path,
            docker_environment=docker_environment,
            docker_lease=FakeLease(),
        )
    )

    assert start_command == [
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "--config",
        str(FakeLease.docker_config),
        "start",
        "--attach",
        "--interactive",
        container_id,
    ]
    assert calls[0][0] == create_command
    assert calls[0][1]["env"] is docker_environment
    assert calls[0][1]["stdin"] is subprocess.DEVNULL
    assert calls[0][1]["stdout"] is subprocess.PIPE
    assert calls[0][1]["stderr"] is subprocess.PIPE


@pytest.mark.parametrize(
    "create_stdout",
    (
        b"container-name\n",
        ("a" * 64 + "\n" + "b" * 64 + "\n").encode("ascii"),
        ("A" * 64 + "\n").encode("ascii"),
    ),
)
def test_grok_docker_create_rejects_untrusted_container_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    create_stdout: bytes,
) -> None:
    class FakeLease:
        docker_bin = "/usr/bin/docker"
        docker_config = tmp_path / "docker-config"

    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=create_stdout,
            stderr=b"",
        ),
    )

    with pytest.raises(ValueError, match="container identity is invalid"):
        grok_cli_runner._create_grok_container_and_build_start_command(
            ["/usr/bin/docker", "create", "sealed-grok"],
            workspace=tmp_path,
            docker_environment={},
            docker_lease=FakeLease(),
        )


@pytest.mark.parametrize("typed_route", (False, True))
def test_grok_docker_primary_parses_attached_start_not_create_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    typed_route: bool,
) -> None:
    harness = _install_fake_grok_docker_primary(tmp_path, monkeypatch)
    provider_calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_bounded(command, *, env):
        provider_calls.append((list(command), dict(env)))
        return 0, b"", 0, False

    def fake_typed(command, *, env):
        provider_calls.append((list(command), dict(env)))
        return 0

    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_bounded_stderr",
        fake_bounded,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        fake_typed,
    )
    argv = [
        "--workspace",
        str(harness["workspace"]),
        "--grok-bin",
        str(harness["grok"]),
        "--model",
        "grok-4.6",
    ]
    if typed_route:
        argv.extend(
            [
                "--codex-fallback-command-json",
                json.dumps(
                    _terra_fallback_command(
                        str(harness["codex"]),
                        harness["workspace"],
                        reasoning_effort="medium",
                    )
                ),
            ]
        )

    assert grok_cli_runner.main(argv) == 0

    create_calls = harness["create_calls"]
    assert isinstance(create_calls, list)
    assert len(create_calls) == 1
    assert "create" in create_calls[0][0]
    assert len(provider_calls) == 1
    assert provider_calls[0][0] == [
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "--config",
        str(tmp_path / "asref-grok-container-test" / "docker-config"),
        "start",
        "--attach",
        "--interactive",
        harness["container_id"],
    ]
    assert harness["close_calls"] == [True]


@pytest.mark.parametrize(
    ("create_returncode", "create_stdout"),
    (
        (125, b""),
        (0, b"container-name\n"),
    ),
)
def test_grok_docker_create_failure_cleans_without_provider_or_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    create_returncode: int,
    create_stdout: bytes,
) -> None:
    harness = _install_fake_grok_docker_primary(
        tmp_path,
        monkeypatch,
        create_returncode=create_returncode,
        create_stdout=create_stdout,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        lambda *_args, **_kwargs: pytest.fail(
            "failed container creation must never start Grok"
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_codex_quota_fallback_in_docker",
        lambda *_args, **_kwargs: pytest.fail(
            "failed container creation must not fall through to Terra"
        ),
    )

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(harness["workspace"]),
            "--grok-bin",
            str(harness["grok"]),
            "--model",
            "grok-4.6",
            "--codex-fallback-command-json",
            json.dumps(
                _terra_fallback_command(
                    str(harness["codex"]),
                    harness["workspace"],
                    reasoning_effort="medium",
                )
            ),
        ]
    )

    assert result == 2
    assert harness["close_calls"] == [False]


def test_grok_docker_start_failure_cleans_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_fake_grok_docker_primary(tmp_path, monkeypatch)
    start_calls: list[list[str]] = []

    def fail_start(command, *, env):
        del env
        start_calls.append(list(command))
        raise OSError("docker start failed")

    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        fail_start,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_codex_quota_fallback_in_docker",
        lambda *_args, **_kwargs: pytest.fail(
            "failed Docker start must not fall through to Terra"
        ),
    )

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(harness["workspace"]),
            "--grok-bin",
            str(harness["grok"]),
            "--model",
            "grok-4.6",
            "--codex-fallback-command-json",
            json.dumps(
                _terra_fallback_command(
                    str(harness["codex"]),
                    harness["workspace"],
                    reasoning_effort="medium",
                )
            ),
        ]
    )

    assert result == 127
    assert len(start_calls) == 1
    assert "start" in start_calls[0]
    assert "create" not in start_calls[0]
    assert harness["close_calls"] == [False]


@pytest.mark.parametrize(
    "invalid_case",
    (
        "mutable_image",
        "unapproved_image",
        "wrong_provider_name",
        "workspace_mismatch",
        "ambient_environment",
    ),
)
def test_docker_codex_boundary_rejects_unpinned_or_mismatched_authority(
    tmp_path: Path,
    invalid_case: str,
) -> None:
    workspace = tmp_path / "workspace"
    other_workspace = tmp_path / "other-workspace"
    workspace.mkdir()
    other_workspace.mkdir()
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex.chmod(0o700)
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}\n", encoding="utf-8")
    source_auth.chmod(0o600)
    fallback_workspace = (
        other_workspace if invalid_case == "workspace_mismatch" else workspace
    )
    image = {
        "mutable_image": "ubuntu:24.04",
        "unapproved_image": "sha256:" + "a" * 64,
    }.get(invalid_case, grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID)
    container_name = (
        "ipfs-accelerate-grok-1-" + "b" * 32
        if invalid_case == "wrong_provider_name"
        else "ipfs-accelerate-codex-1-" + "b" * 32
    )
    child_env = grok_cli_runner._codex_task_container_environment()
    if invalid_case == "ambient_environment":
        child_env["BASH_ENV"] = "/workspace/untrusted-hook"

    with pytest.raises(ValueError):
        grok_cli_runner._docker_codex_fallback_command(
            codex_command=_terra_fallback_command(
                str(codex),
                fallback_workspace,
            ),
            workspace=workspace,
            source_auth=source_auth,
            child_env=child_env,
            docker_config=tmp_path,
            container_name=container_name,
            cidfile=tmp_path / "container.cid",
            docker_bin="/usr/bin/docker",
            isolation_image=image,
        )


@pytest.mark.parametrize(
    ("outcome", "expected_finished"),
    (
        ("success", True),
        ("signal", True),
        ("error", False),
        ("auth_swap", False),
    ),
)
def test_docker_codex_fallback_always_closes_its_separate_lease(
    tmp_path: Path,
    monkeypatch,
    outcome: str,
    expected_finished: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex.chmod(0o700)
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}\n", encoding="utf-8")
    source_auth.chmod(0o600)
    provider_home = tmp_path / "asref-codex-home-test"
    provider_home.mkdir()
    prompt_path = tmp_path / "asref-grok-prompt-test.txt"
    prompt_path.write_text("repair", encoding="utf-8")
    close_calls: list[bool] = []
    create_kwargs: list[dict[str, object]] = []
    create_commands: list[list[str]] = []
    popen_calls: list[bool] = []
    boundary_events: list[str] = []
    created_container_id = "d" * 64

    class FakeHome:
        name = str(provider_home)

        def cleanup(self) -> None:
            return None

    class FakeLease:
        docker_config = tmp_path / "docker-config"
        container_name = "ipfs-accelerate-codex-1-" + "c" * 32
        cidfile = tmp_path / "container.cid"

        def close(self, *, docker_run_finished: bool) -> None:
            close_calls.append(docker_run_finished)

    FakeLease.docker_config.mkdir()

    def fake_create(*_args, **kwargs):
        create_kwargs.append(dict(kwargs))
        return FakeLease()

    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")

        def wait(self) -> int:
            return 0 if outcome == "success" else -15

    def fake_run(command, **_kwargs):
        create_command = list(command)
        create_commands.append(create_command)
        boundary_events.append("create")
        assert create_command == ["docker", "create"]
        return subprocess.CompletedProcess(
            create_command,
            0,
            stdout=(created_container_id + "\n").encode("ascii"),
            stderr=b"",
        )

    def fake_popen(command, **_kwargs):
        boundary_events.append("popen")
        assert list(command) == [
            "/usr/bin/docker",
            "--host=unix:///var/run/docker.sock",
            "--config",
            str(FakeLease.docker_config),
            "start",
            "--attach",
            "--interactive",
            created_container_id,
        ]
        if outcome == "error":
            raise OSError("docker launch failed")
        popen_calls.append(True)
        return FakeProcess()

    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: str(codex),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_isolation_binary",
        lambda: "/usr/bin/docker",
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_isolated_codex_quota_fallback_home",
        lambda **_kwargs: (
            FakeHome(),
            grok_cli_runner._codex_task_container_environment(),
            source_auth,
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner._DockerContainerLease,
        "create",
        fake_create,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_codex_task_toolchain_image_id",
        lambda *_args, **_kwargs: (
            grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_codex_fallback_command",
        lambda **_kwargs: ["docker", "create"],
    )
    validate_auth = grok_cli_runner._validated_codex_auth_path

    def record_and_validate_auth(**kwargs):
        try:
            validated = validate_auth(**kwargs)
        except (OSError, ValueError):
            boundary_events.append("auth-fail")
            raise
        boundary_events.append("auth")
        return validated

    monkeypatch.setattr(
        grok_cli_runner,
        "_validated_codex_auth_path",
        record_and_validate_auth,
    )
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(grok_cli_runner.subprocess, "Popen", fake_popen)

    def invocation() -> int:
        def swap_auth() -> None:
            boundary_events.append("route")
            replacement = tmp_path / "replacement-auth.json"
            replacement.write_text("{}\n", encoding="utf-8")
            replacement.chmod(0o600)
            source_auth.unlink()
            source_auth.symlink_to(replacement)

        def validate_route() -> None:
            boundary_events.append("route")

        return grok_cli_runner._run_codex_quota_fallback_in_docker(
            _terra_fallback_command(str(codex), workspace),
            workspace=workspace,
            prompt="repair",
            prompt_path=prompt_path,
            base_env={},
            pre_effect_validator=(
                swap_auth if outcome == "auth_swap" else validate_route
            ),
        )
    if outcome in {"error", "auth_swap"}:
        with pytest.raises((OSError, ValueError)):
            invocation()
    else:
        expected_returncode = 0 if outcome == "success" else -15
        assert invocation() == expected_returncode

    assert close_calls == [expected_finished]
    if outcome == "auth_swap":
        assert popen_calls == []
        assert create_commands == []
        assert boundary_events == ["route", "auth-fail"]
    else:
        assert create_commands == [["docker", "create"]]
        assert popen_calls == ([] if outcome == "error" else [True])
        assert boundary_events == [
            "route",
            "auth",
            "route",
            "create",
            "popen",
        ]
    assert create_kwargs == [
        {
            "provider": "codex",
            "provider_home": provider_home,
            "prompt_path": prompt_path,
        }
    ]


def test_real_disposable_codex_container_and_board_toolchain_probe(
    tmp_path: Path,
) -> None:
    workspace = Path(__file__).resolve().parents[2]
    docker_bin = grok_cli_runner._docker_isolation_binary()
    codex = grok_cli_runner.resolve_codex_quota_fallback_executable(
        workspace=workspace,
    )
    if not docker_bin or not codex:
        pytest.skip("trusted local Docker/Codex boundary is unavailable")
    docker_config = tmp_path / "docker-config"
    docker_config.mkdir()
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}\n", encoding="utf-8")
    source_auth.chmod(0o600)
    image = grok_cli_runner._docker_codex_task_toolchain_image_id(
        docker_bin,
        docker_config=docker_config,
    )
    if not image:
        pytest.skip("pinned local Docker image is unavailable")
    assert image == grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID
    child_env = grok_cli_runner._codex_task_container_environment()

    def create_start_wait_and_cleanup(
        create_command: list[str],
        *,
        container_name: str,
        cidfile: Path,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        assert create_command[4] == "create"
        assert "run" not in create_command
        assert "--rm" not in create_command
        assert "--pull=never" in create_command
        label_index = create_command.index("--label")
        assert create_command[label_index + 1] == (
            "ipfs_accelerate.codex_fallback_isolation=true"
        )
        assert create_command.index(image) > label_index
        container_id = ""
        try:
            created = subprocess.run(
                create_command,
                cwd=workspace,
                env=grok_cli_runner._docker_control_env(child_env),
                input="",
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            assert created.returncode == 0, created.stderr
            created_fields = created.stdout.split()
            assert len(created_fields) == 1
            container_id = created_fields[0]
            assert re.fullmatch(r"[0-9a-f]{64}", container_id)
            assert cidfile.read_text(encoding="ascii").strip() == container_id

            inspected = subprocess.run(
                [
                    docker_bin,
                    "--host=unix:///var/run/docker.sock",
                    "--config",
                    str(docker_config),
                    "container",
                    "inspect",
                    container_id,
                ],
                env=grok_cli_runner._docker_control_env(child_env),
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            assert inspected.returncode == 0, inspected.stderr
            inspection = json.loads(inspected.stdout)
            assert isinstance(inspection, list) and len(inspection) == 1
            record = inspection[0]
            assert record["Id"] == container_id
            assert record["Name"] == "/" + container_name
            assert record["Image"] == image
            assert record["State"]["Status"] == "created"
            assert record["Config"]["Labels"].get(
                "ipfs_accelerate.codex_fallback_isolation"
            ) == "true"

            started = subprocess.run(
                [
                    docker_bin,
                    "--host=unix:///var/run/docker.sock",
                    "--config",
                    str(docker_config),
                    "start",
                    "--attach",
                    "--interactive",
                    container_id,
                ],
                cwd=workspace,
                env=grok_cli_runner._docker_control_env(child_env),
                input="",
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            waited = subprocess.run(
                [
                    docker_bin,
                    "--host=unix:///var/run/docker.sock",
                    "--config",
                    str(docker_config),
                    "container",
                    "wait",
                    container_id,
                ],
                env=grok_cli_runner._docker_control_env(child_env),
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            assert waited.returncode == 0, waited.stderr
            assert waited.stdout.split() == [str(started.returncode)]
            return started
        finally:
            grok_cli_runner._remove_exact_docker_container(
                docker_bin=docker_bin,
                docker_config=docker_config,
                container_name=container_name,
                settle_for_creation=False,
            )
            absent = subprocess.run(
                [
                    docker_bin,
                    "--host=unix:///var/run/docker.sock",
                    "--config",
                    str(docker_config),
                    "container",
                    "ls",
                    "--all",
                    "--no-trunc",
                    "--filter",
                    f"name=^/{container_name}$",
                    "--format",
                    "{{.Names}}",
                ],
                env=grok_cli_runner._docker_control_env(child_env),
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            assert absent.returncode == 0, absent.stderr
            assert absent.stdout.strip() == ""

    version_container_name = (
        f"ipfs-accelerate-codex-{os.getpid()}-{uuid.uuid4().hex}"
    )
    version_cidfile = tmp_path / "version-container.cid"
    command = grok_cli_runner._docker_codex_fallback_command(
        codex_command=_terra_fallback_command(codex, workspace),
        workspace=workspace,
        source_auth=source_auth,
        child_env=child_env,
        docker_config=docker_config,
        container_name=version_container_name,
        cidfile=version_cidfile,
        docker_bin=docker_bin,
        isolation_image=image,
    )
    image_index = command.index(image)
    codex_index = command.index(codex, image_index + 1)
    probe_command = [*command[:codex_index], codex, "--version"]

    completed = create_start_wait_and_cleanup(
        probe_command,
        container_name=version_container_name,
        cidfile=version_cidfile,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("codex-cli ")
    assert "bwrap:" not in completed.stderr

    validation_container_name = (
        f"ipfs-accelerate-codex-{os.getpid()}-{uuid.uuid4().hex}"
    )
    validation_cidfile = tmp_path / "validation-container.cid"
    validation_command = grok_cli_runner._docker_codex_fallback_command(
        codex_command=_terra_fallback_command(codex, workspace),
        workspace=workspace,
        source_auth=source_auth,
        child_env=child_env,
        docker_config=docker_config,
        container_name=validation_container_name,
        cidfile=validation_cidfile,
        docker_bin=docker_bin,
        isolation_image=image,
    )
    validation_image_index = validation_command.index(image)
    validation_codex_index = validation_command.index(
        codex,
        validation_image_index + 1,
    )
    validation_command[validation_codex_index:] = [
        "python",
        "-m",
        "pytest",
        "test/api/test_agent_supervisor_prompt_v3_convergence.py",
        "-q",
    ]
    validation = create_start_wait_and_cleanup(
        validation_command,
        container_name=validation_container_name,
        cidfile=validation_cidfile,
        timeout=120,
    )

    assert validation.returncode == 0, validation.stdout + validation.stderr
    assert "passed" in validation.stdout


def test_daemon_liveness_accepts_exact_codex_fallback_container_label(
    tmp_path: Path,
    monkeypatch,
) -> None:
    daemon = _daemon(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        if command[:2] == ["docker", "ps"]:
            label = command[command.index("--filter") + 1]
            stdout = "codex-container\n" if "codex_fallback" in label else ""
            return subprocess.CompletedProcess(command, 0, stdout=stdout)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                [{"Source": str(tmp_path), "Destination": str(tmp_path)}]
            ),
        )

    monkeypatch.setattr(implementation_daemon.subprocess, "run", fake_run)

    assert daemon._docker_isolation_active_for_worktree(str(tmp_path)) is True
    filters = [
        call[call.index("--filter") + 1]
        for call in calls
        if call[:2] == ["docker", "ps"]
    ]
    assert filters == [
        "label=ipfs_accelerate.grok_isolation=true",
        "label=ipfs_accelerate.codex_fallback_isolation=true",
    ]


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
    assert command[command.index("--model") + 1] == "grok-4.6"
    fallback = json.loads(command[command.index("--codex-fallback-command-json") + 1])
    assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="medium"' in fallback
    assert "--ephemeral" in fallback
