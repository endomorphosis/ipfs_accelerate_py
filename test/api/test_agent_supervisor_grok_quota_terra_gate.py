"""Contracts for Grok hard-quota → Codex Terra/medium fallback authority."""

from __future__ import annotations

import io
import json
import os
import subprocess
import uuid
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

    def fake_fallback(command, **kwargs) -> int:
        fallback_calls.append((list(command), dict(kwargs)))
        return 0

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
        "_run_codex_quota_fallback_in_docker",
        fake_fallback,
    )
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
        assert kwargs["workspace"] == workspace.resolve()
        assert kwargs["prompt"] == prompt
        assert Path(kwargs["prompt_path"]).name.startswith("asref-grok-prompt-")
    else:
        assert "did not confirm the same quota failure" in capsys.readouterr().err


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
    image = "sha256:" + "a" * 64
    container_name = "ipfs-accelerate-codex-1-" + "b" * 32

    command = grok_cli_runner._docker_codex_fallback_command(
        codex_command=fallback,
        workspace=workspace,
        source_auth=source_auth,
        child_env={
            "HOME": str(grok_cli_runner._CODEX_CONTAINER_HOME),
            "CODEX_HOME": str(grok_cli_runner._CODEX_CONTAINER_HOME),
            "PATH": "/usr/bin:/bin",
            "LANG": "opaque-child-value",
        },
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
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--device" not in command
    assert "ipfs_accelerate.codex_fallback_isolation=true" in command
    assert image in command
    assert command[command.index("--env") + 1] == "CODEX_HOME"
    assert "LANG" in [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--env"
    ]
    assert "opaque-child-value" not in command

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
        f"type=bind,src={source_auth},"
        f"dst={grok_cli_runner._CODEX_CONTAINER_AUTH_PATH},readonly"
        in mounts
    )
    assert not any("/var/run/docker.sock" in mount for mount in mounts)
    assert not any("/home/" in mount for mount in mounts)

    inner = command[command.index(image) + 1 :]
    expected_inner = list(fallback)
    expected_inner[expected_inner.index("-s") + 1] = "danger-full-access"
    assert inner == expected_inner
    assert "--dangerously-bypass-approvals-and-sandbox" not in inner


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


@pytest.mark.parametrize(
    "invalid_case",
    ("mutable_image", "wrong_provider_name", "workspace_mismatch"),
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
    image = (
        "ubuntu:24.04"
        if invalid_case == "mutable_image"
        else "sha256:" + "a" * 64
    )
    container_name = (
        "ipfs-accelerate-grok-1-" + "b" * 32
        if invalid_case == "wrong_provider_name"
        else "ipfs-accelerate-codex-1-" + "b" * 32
    )

    with pytest.raises(ValueError):
        grok_cli_runner._docker_codex_fallback_command(
            codex_command=_terra_fallback_command(
                str(codex),
                fallback_workspace,
            ),
            workspace=workspace,
            source_auth=source_auth,
            child_env={"PATH": "/usr/bin:/bin"},
            docker_config=tmp_path,
            container_name=container_name,
            cidfile=tmp_path / "container.cid",
            docker_bin="/usr/bin/docker",
            isolation_image=image,
        )


@pytest.mark.parametrize(
    ("outcome", "expected_finished"),
    (("success", True), ("signal", True), ("error", False)),
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

    def fake_run(*_args, **_kwargs):
        if outcome == "error":
            raise OSError("docker launch failed")
        return subprocess.CompletedProcess(
            ["docker", "run"],
            0 if outcome == "success" else -15,
        )

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
            {
                "HOME": str(grok_cli_runner._CODEX_CONTAINER_HOME),
                "CODEX_HOME": str(grok_cli_runner._CODEX_CONTAINER_HOME),
                "PATH": "/usr/bin:/bin",
            },
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
        "_docker_isolation_image_id",
        lambda *_args, **_kwargs: "sha256:" + "d" * 64,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_codex_fallback_command",
        lambda **_kwargs: ["docker", "run"],
    )
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)

    def invocation() -> int:
        return grok_cli_runner._run_codex_quota_fallback_in_docker(
            _terra_fallback_command(str(codex), workspace),
            workspace=workspace,
            prompt="repair",
            prompt_path=prompt_path,
            base_env={},
        )
    if outcome == "error":
        with pytest.raises(OSError, match="docker launch failed"):
            invocation()
    else:
        expected_returncode = 0 if outcome == "success" else -15
        assert invocation() == expected_returncode

    assert close_calls == [expected_finished]
    assert create_kwargs == [
        {
            "provider": "codex",
            "provider_home": provider_home,
            "prompt_path": prompt_path,
        }
    ]


def test_real_disposable_codex_container_boundary_probe(tmp_path: Path) -> None:
    docker_bin = grok_cli_runner._docker_isolation_binary()
    codex = grok_cli_runner.resolve_codex_quota_fallback_executable(
        workspace=tmp_path,
    )
    if not docker_bin or not codex:
        pytest.skip("trusted local Docker/Codex boundary is unavailable")
    workspace = tmp_path / "workspace"
    docker_config = tmp_path / "docker-config"
    workspace.mkdir()
    docker_config.mkdir()
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}\n", encoding="utf-8")
    source_auth.chmod(0o600)
    image = grok_cli_runner._docker_isolation_image_id(
        docker_bin,
        docker_config=docker_config,
    )
    if not image:
        pytest.skip("pinned local Docker image is unavailable")
    child_env = {
        "HOME": str(grok_cli_runner._CODEX_CONTAINER_HOME),
        "CODEX_HOME": str(grok_cli_runner._CODEX_CONTAINER_HOME),
        "PATH": "/usr/bin:/bin",
    }
    command = grok_cli_runner._docker_codex_fallback_command(
        codex_command=_terra_fallback_command(codex, workspace),
        workspace=workspace,
        source_auth=source_auth,
        child_env=child_env,
        docker_config=docker_config,
        container_name=(
            f"ipfs-accelerate-codex-{os.getpid()}-{uuid.uuid4().hex}"
        ),
        cidfile=tmp_path / "container.cid",
        docker_bin=docker_bin,
        isolation_image=image,
    )
    image_index = command.index(image)
    probe_command = [*command[: image_index + 1], codex, "--version"]

    completed = subprocess.run(
        probe_command,
        cwd=workspace,
        env=grok_cli_runner._docker_control_env(child_env),
        input="",
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("codex-cli ")
    assert "bwrap:" not in completed.stderr


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
    assert command[command.index("--model") + 1] == "grok-4.5"
    fallback = json.loads(command[command.index("--codex-fallback-command-json") + 1])
    assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
    assert "--ephemeral" in fallback
