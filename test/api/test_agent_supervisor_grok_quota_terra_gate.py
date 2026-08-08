"""Contracts for Grok hard-quota → Codex Terra/medium fallback authority."""

from __future__ import annotations

import json
import os
import signal
import uuid
from pathlib import Path

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    TodoImplementationDaemon,
)

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


def _terra_fallback_command(
    codex: str,
    workspace: str | Path,
    *,
    reasoning_effort: str = "medium",
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
        "-c",
        'web_search="disabled"',
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
    assert 'web_search="disabled"' in fallback
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
    assert 'web_search="disabled"' in fallback
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


def _containerized_fallback_inputs(tmp_path: Path) -> dict[str, Path]:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    git_control = workspace / ".git"
    git_control.mkdir()
    (git_control / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_control / "objects").mkdir()
    (git_control / "refs").mkdir()
    auth_path = (tmp_path / "auth.json").resolve()
    auth_path.write_text("{}", encoding="utf-8")
    auth_path.chmod(0o600)
    package_root = (tmp_path / "codex-linux-arm64").resolve()
    package_root.mkdir()
    bwrap_path = (tmp_path / "bwrap").resolve()
    bwrap_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    bwrap_path.chmod(0o700)
    checkpoint_path = (tmp_path / "checkpoint").resolve()
    checkpoint_path.mkdir(mode=0o700)
    docker_config = (tmp_path / "docker-config").resolve()
    docker_config.mkdir()
    return {
        "workspace": workspace,
        "git_control": git_control,
        "auth_path": auth_path,
        "package_root": package_root,
        "bwrap_path": bwrap_path,
        "checkpoint_path": checkpoint_path,
        "docker_config": docker_config,
        "cidfile": (tmp_path / "container.cid").resolve(),
    }


def test_terra_high_quota_fallback_requires_the_pinned_docker_boundary(
    tmp_path: Path,
) -> None:
    """The sealed high route can only become the exact nested boundary."""

    paths = _containerized_fallback_inputs(tmp_path)
    workspace = paths["workspace"]
    auth_path = paths["auth_path"]
    package_root = paths["package_root"]
    bwrap_path = paths["bwrap_path"]
    checkpoint_path = paths["checkpoint_path"]
    docker_config = paths["docker_config"]
    cidfile = paths["cidfile"]
    git_control = paths["git_control"]
    container_name = "ipfs-accelerate-codex-1-" + "a" * 32
    host_route = _terra_fallback_command(
        "/usr/local/bin/codex", workspace, reasoning_effort="high"
    )
    command = grok_cli_runner._build_containerized_codex_quota_fallback_command(
        host_fallback_command=host_route,
        workspace=workspace,
        auth_path=auth_path,
        package_root=package_root,
        bwrap_path=bwrap_path,
        checkpoint_path=checkpoint_path,
        docker_config=docker_config,
        container_name=container_name,
        cidfile=cidfile,
        git_controls=(git_control,),
    )

    assert command[:5] == [
        "/usr/bin/docker",
        "--host=unix:///run/docker.sock",
        "--config",
        str(docker_config),
        "run",
    ]
    assert "--pull=never" in command
    assert "--rm" in command
    assert "--init" in command
    assert "--network=bridge" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert command.count("--cap-add=SYS_ADMIN") == 1
    assert command.count("--cap-add=SYS_CHROOT") == 1
    assert command.count("--cap-add=SETUID") == 1
    assert command.count("--cap-add=SETGID") == 1
    assert command.count("--cap-add=SYS_PTRACE") == 1
    assert command.count("--cap-add=NET_ADMIN") == 1
    assert command.count("--cap-add=NET_RAW") == 1
    assert "--security-opt=seccomp=unconfined" in command
    assert "--security-opt=apparmor=unconfined" in command
    assert "--security-opt=systempaths=unconfined" in command
    assert "--security-opt=no-new-privileges:true" not in command
    assert command[command.index("--user") + 1] == "0:0"
    assert command[command.index("--entrypoint") + 1] == "/bin/sh"
    toolchain_path = (
        f"PATH={grok_cli_runner.TYPESCRIPT_TOOLCHAIN_BIN}:"
        "/usr/local/bin:/usr/bin:/bin"
    )
    assert command[command.index(toolchain_path) - 1] == "--env"
    assert (
        f"NODE_PATH={grok_cli_runner.TYPESCRIPT_NODE_MODULES}"
        in command
    )
    assert (
        "IPFS_ACCELERATE_TYPESCRIPT_JS="
        f"{grok_cli_runner.TYPESCRIPT_COMPILER_JS}"
        in command
    )
    assert (
        "IPFS_ACCELERATE_TYPESCRIPT_PACKAGE_JSON="
        f"{grok_cli_runner.TYPESCRIPT_PACKAGE_JSON}"
        in command
    )
    assert (
        "IPFS_ACCELERATE_TYPESCRIPT_VERSION="
        f"{grok_cli_runner.TYPESCRIPT_VERSION}"
        in command
    )
    assert (
        f"{grok_cli_runner._CODEX_FALLBACK_CHECKPOINT_ENV}={checkpoint_path}"
        in command
    )
    assert "/usr/local/bin/codex" not in command
    image_index = command.index(grok_cli_runner._CODEX_FALLBACK_IMAGE)
    assert command[image_index + 1] == "-ec"
    wrapper = command[image_index + 2]
    assert "/opt/host-bwrap /usr/local/bin/bwrap" in wrapper
    assert "/bin/chmod 4755 /usr/local/bin/bwrap" in wrapper
    assert "--clear-groups -- /opt/codex/" in wrapper
    profile = next(
        item
        for item in command
        if item.startswith('permissions.dcr_fallback={extends=":workspace"')
    )
    assert '"/opt/ipfs-accelerate-codex-home/auth.json"="deny"' in profile
    assert str(auth_path) not in profile
    assert '"/opt/host-bwrap"="deny"' in profile
    assert '"/usr/local/bin/bwrap"="deny"' in profile
    assert '"/opt/codex"="read"' in profile
    assert f'"{checkpoint_path}"="write"' in profile
    assert f'"{git_control}"="read"' in profile
    assert 'default_permissions="dcr_fallback"' in command
    assert "-s" not in command[image_index + 4 :]
    assert 'web_search="disabled"' in command
    tmpfs = [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "--tmpfs"
    ]
    assert any(item.startswith("/tmp:rw,nosuid,nodev,exec,") for item in tmpfs)
    assert any(
        item.startswith(
            "/opt/ipfs-accelerate-codex-home:rw,nosuid,nodev,exec,"
        )
        for item in tmpfs
    )
    assert "/usr/local/bin:rw,suid,nodev,exec,mode=0755,size=16m" in tmpfs
    mounts = [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "--mount"
    ]
    assert mounts == [
        f"type=bind,src={workspace},dst={workspace}",
        f"type=bind,src={git_control},dst={git_control},readonly",
        (
            f"type=bind,src={auth_path},dst="
            f"{grok_cli_runner._CODEX_FALLBACK_AUTH_DESTINATION},readonly"
        ),
        f"type=bind,src={package_root},dst=/opt/codex,readonly",
        f"type=bind,src={bwrap_path},dst=/opt/host-bwrap,readonly",
        (
            "type=bind,src=/dev/null,dst="
            f"{grok_cli_runner._CODEX_FALLBACK_CONTAINER_BUNDLED_BWRAP},readonly"
        ),
        f"type=bind,src={checkpoint_path},dst={checkpoint_path}",
    ]

    def validate(candidate: list[str]) -> None:
        grok_cli_runner._validate_containerized_codex_quota_fallback_command(
            candidate,
            workspace=workspace,
            auth_path=auth_path,
            package_root=package_root,
            bwrap_path=bwrap_path,
            checkpoint_path=checkpoint_path,
            docker_config=docker_config,
            container_name=container_name,
            cidfile=cidfile,
            git_controls=(git_control,),
            host_fallback_command=host_route,
        )

    wrong_image = list(command)
    wrong_image[image_index] = "sha256:" + "0" * 64
    wrong_network = list(command)
    wrong_network[wrong_network.index("--network=bridge")] = "--network=none"
    wrong_mount = list(command)
    auth_mount_index = next(
        index
        for index, item in enumerate(wrong_mount)
        if item == "--mount"
        and f"dst={grok_cli_runner._CODEX_FALLBACK_AUTH_DESTINATION}"
        in wrong_mount[index + 1]
    )
    wrong_mount[auth_mount_index + 1] = wrong_mount[auth_mount_index + 1].replace(
        str(grok_cli_runner._CODEX_FALLBACK_AUTH_DESTINATION), "/tmp/other.json"
    )
    wrong_profile = list(command)
    wrong_profile[wrong_profile.index('default_permissions="dcr_fallback"')] = (
        'default_permissions=":workspace"'
    )
    legacy_sandbox = list(command)
    legacy_sandbox.insert(legacy_sandbox.index("-"), "-s")
    legacy_sandbox.insert(legacy_sandbox.index("-") + 1, "danger-full-access")
    missing_cleanup = [item for item in command if item != "--rm"]
    duplicate_environment = list(command)
    duplicate_environment[duplicate_environment.index("--env") : duplicate_environment.index("--env")] = [
        "--env",
        "EXTRA=1",
    ]
    unknown_docker_flag = list(command)
    unknown_docker_flag.insert(image_index, "--privileged")
    duplicate_network = list(command)
    duplicate_network.insert(duplicate_network.index("--network=bridge"), "--network=bridge")
    missing_helper_suid = list(command)
    helper_tmpfs_index = missing_helper_suid.index(
        "/usr/local/bin:rw,suid,nodev,exec,mode=0755,size=16m"
    )
    missing_helper_suid[helper_tmpfs_index] = (
        "/usr/local/bin:rw,nodev,exec,mode=0755,size=16m"
    )
    for drifted in (
        wrong_image,
        wrong_network,
        wrong_mount,
        wrong_profile,
        legacy_sandbox,
        missing_cleanup,
        duplicate_environment,
        unknown_docker_flag,
        duplicate_network,
        missing_helper_suid,
    ):
        with pytest.raises(ValueError):
            validate(drifted)

    comma_workspace = (tmp_path / "workspace,option=escape").resolve()
    comma_workspace.mkdir()
    with pytest.raises(ValueError, match="mount inputs"):
        grok_cli_runner._build_containerized_codex_quota_fallback_command(
            host_fallback_command=_terra_fallback_command(
                "/usr/local/bin/codex", comma_workspace, reasoning_effort="high"
            ),
            workspace=comma_workspace,
            auth_path=auth_path,
            package_root=package_root,
            bwrap_path=bwrap_path,
            checkpoint_path=checkpoint_path,
            docker_config=docker_config,
            container_name=container_name,
            cidfile=cidfile,
            git_controls=(),
        )
    with pytest.raises(ValueError, match="checkpoint authority"):
        grok_cli_runner._build_containerized_codex_quota_fallback_command(
            host_fallback_command=host_route,
            workspace=workspace,
            auth_path=auth_path,
            package_root=package_root,
            bwrap_path=bwrap_path,
            checkpoint_path=None,
            docker_config=docker_config,
            container_name=container_name,
            cidfile=cidfile,
            git_controls=(git_control,),
        )
    with pytest.raises(ValueError, match="Git metadata"):
        grok_cli_runner._build_containerized_codex_quota_fallback_command(
            host_fallback_command=host_route,
            workspace=workspace,
            auth_path=auth_path,
            package_root=package_root,
            bwrap_path=bwrap_path,
            checkpoint_path=checkpoint_path,
            docker_config=docker_config,
            container_name=container_name,
            cidfile=cidfile,
            git_controls=(tmp_path.resolve(),),
        )


def test_medium_legacy_fallback_has_no_checkpoint_authority(tmp_path: Path) -> None:
    paths = _containerized_fallback_inputs(tmp_path)
    workspace = paths["workspace"]
    command = grok_cli_runner._build_containerized_codex_quota_fallback_command(
        host_fallback_command=_terra_fallback_command("/usr/local/bin/codex", workspace),
        workspace=workspace,
        auth_path=paths["auth_path"],
        package_root=paths["package_root"],
        bwrap_path=paths["bwrap_path"],
        checkpoint_path=None,
        docker_config=paths["docker_config"],
        container_name="ipfs-accelerate-codex-1-" + "d" * 32,
        cidfile=paths["cidfile"],
        git_controls=(paths["git_control"],),
    )
    assert str(paths["checkpoint_path"]) not in command
    assert not any(
        item.startswith(grok_cli_runner._CODEX_FALLBACK_CHECKPOINT_ENV + "=")
        for item in command
    )
    profile = next(
        item for item in command if item.startswith("permissions.dcr_fallback=")
    )
    assert str(paths["checkpoint_path"]) not in profile
    assert '"/opt/ipfs-accelerate-codex-home/auth.json"="deny"' in profile


def test_high_checkpoint_requires_safe_owner_writable_directory(
    tmp_path: Path,
) -> None:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    checkpoint = (tmp_path / "checkpoint").resolve()
    checkpoint.mkdir(mode=0o500)
    with pytest.raises(ValueError, match="checkpoint"):
        grok_cli_runner._resolve_codex_quota_fallback_checkpoint_path(
            workspace=workspace,
            base_env={
                grok_cli_runner._CODEX_FALLBACK_CHECKPOINT_ENV: str(checkpoint)
            },
        )


def test_git_control_scan_rejects_walk_errors_and_unbounded_markers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()

    def unreadable_walk(*_args: object, **kwargs: object):
        onerror = kwargs["onerror"]
        assert callable(onerror)
        onerror(OSError("permission denied"))
        if False:
            yield "", [], []

    monkeypatch.setattr(grok_cli_runner.os, "walk", unreadable_walk)
    with pytest.raises(ValueError, match="could not inspect"):
        grok_cli_runner._codex_fallback_git_controls(workspace)

    def oversized_walk(*_args: object, **_kwargs: object):
        for _ in range(grok_cli_runner._CODEX_FALLBACK_MAX_GIT_MARKERS + 1):
            yield str(workspace), [".git"], []

    monkeypatch.setattr(grok_cli_runner.os, "walk", oversized_walk)
    with pytest.raises(ValueError, match="too many Git controls"):
        grok_cli_runner._codex_fallback_git_controls(workspace)


def test_container_auth_must_be_private_to_its_owner(tmp_path: Path) -> None:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    codex_home = (tmp_path / "codex-home").resolve()
    codex_home.mkdir()
    auth_path = codex_home / "auth.json"
    auth_path.write_text("{}", encoding="utf-8")
    auth_path.chmod(0o644)
    with pytest.raises(ValueError, match="auth"):
        grok_cli_runner._resolve_codex_quota_fallback_auth_path(
            workspace=workspace,
            base_env={"CODEX_HOME": str(codex_home)},
        )


def test_watchdog_defers_reentrant_sigterm_until_exact_cleanup_finishes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A SIGTERM during removal must not interrupt the sole Docker reaper."""

    lease_root = Path(
        grok_cli_runner.tempfile.mkdtemp(prefix="asref-codex-container-")
    ).resolve()
    docker_config = lease_root / "docker-config"
    docker_config.mkdir(mode=0o700)
    cidfile = lease_root / "container.cid"
    ready_fifo = tmp_path / "watchdog-ready"
    os.mkfifo(ready_fifo)
    ready_fd = os.open(ready_fifo, os.O_RDWR | os.O_NONBLOCK)
    handlers: dict[int, object] = {}
    removal: list[str] = []

    class _Input:
        class buffer:
            @staticmethod
            def read(_size: int) -> bytes:
                return b""

    def capture_signal(signum: int, handler: object) -> None:
        handlers[signum] = handler

    def interrupted_remove(**_kwargs: object) -> None:
        removal.append("started")
        handler = handlers[signal.SIGTERM]
        assert callable(handler)
        handler(signal.SIGTERM, None)
        removal.append("finished")

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", _Input())
    monkeypatch.setattr(grok_cli_runner.signal, "signal", capture_signal)
    monkeypatch.setattr(
        grok_cli_runner,
        "_remove_exact_docker_container",
        interrupted_remove,
    )
    try:
        assert grok_cli_runner._docker_cleanup_watchdog_main(
            [
                "--docker-bin",
                "/usr/bin/docker",
                "--container-name",
                "ipfs-accelerate-codex-1-" + "e" * 32,
                "--cidfile",
                str(cidfile),
                "--lease-root",
                str(lease_root),
                "--ready-fd",
                str(ready_fd),
                "--codex-fallback",
            ]
        ) == 0
    finally:
        try:
            os.close(ready_fd)
        except OSError:
            pass
    assert removal == ["started", "finished"]


def test_terra_quota_fallback_invokes_docker_not_host_codex(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _containerized_fallback_inputs(tmp_path)
    workspace = paths["workspace"]

    class Lease:
        container_name = "ipfs-accelerate-codex-1-" + "b" * 32
        closed = False

        def __init__(self) -> None:
            self.docker_config = paths["docker_config"]
            self.cidfile = paths["cidfile"]

        def close(self, *, docker_run_finished: bool) -> None:
            assert docker_run_finished is True
            self.closed = True

    lease = Lease()
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_containerized_codex_fallback_assets",
        lambda **_kwargs: (
            Path("/usr/bin/docker"),
            paths["auth_path"],
            paths["package_root"],
            paths["bwrap_path"],
            None,
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner._DockerCodexFallbackLease,
        "create",
        classmethod(lambda _cls: lease),
    )
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> object:
        calls.append(command)
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)
    result = grok_cli_runner._run_containerized_codex_quota_fallback(
        host_fallback_command=_terra_fallback_command("/usr/local/bin/codex", workspace),
        workspace=workspace,
        base_env={},
        prompt="no provider call in this test",
    )

    assert result.returncode == 0
    assert lease.closed is True
    assert len(calls) == 1
    assert calls[0][0] == "/usr/bin/docker"
    assert "/usr/local/bin/codex" not in calls[0]


def test_terra_quota_fallback_closes_its_lease_on_launch_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _containerized_fallback_inputs(tmp_path)
    workspace = paths["workspace"]

    class Lease:
        container_name = "ipfs-accelerate-codex-1-" + "c" * 32
        closed_with: bool | None = None

        def __init__(self) -> None:
            self.docker_config = paths["docker_config"]
            self.cidfile = paths["cidfile"]

        def close(self, *, docker_run_finished: bool) -> None:
            self.closed_with = docker_run_finished

    lease = Lease()
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_containerized_codex_fallback_assets",
        lambda **_kwargs: (
            Path("/usr/bin/docker"),
            paths["auth_path"],
            paths["package_root"],
            paths["bwrap_path"],
            None,
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner._DockerCodexFallbackLease,
        "create",
        classmethod(lambda _cls: lease),
    )
    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("launch failed")),
    )

    with pytest.raises(OSError, match="launch failed"):
        grok_cli_runner._run_containerized_codex_quota_fallback(
            host_fallback_command=_terra_fallback_command(
                "/usr/local/bin/codex", workspace
            ),
            workspace=workspace,
            base_env={},
            prompt="no provider call in this test",
        )
    assert lease.closed_with is False
