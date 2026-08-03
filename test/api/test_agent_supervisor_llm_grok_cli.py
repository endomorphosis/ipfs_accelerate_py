"""End-to-end supervisor subprocess coverage for the Grok CLI provider."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import uuid
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm import (
    LlmRouterInvocation,
    call_llm_router,
)


def test_supervisor_child_routes_grok_through_datasets_router(monkeypatch, tmp_path) -> None:
    fake_grok = tmp_path / "grok"
    fake_grok.write_text(
        """#!/usr/bin/env python3
import json
import pathlib
import sys

args = sys.argv[1:]
prompt_path = pathlib.Path(args[args.index("--prompt-file") + 1])
prompt = prompt_path.read_text(encoding="utf-8")
model = args[args.index("--model") + 1]
print(json.dumps({
    "text": f"supervisor:{model}:{prompt}",
    "stopReason": "EndTurn",
    "sessionId": "supervisor-session",
    "requestId": "supervisor-request",
}))
""",
        encoding="utf-8",
    )
    fake_grok.chmod(0o700)

    monkeypatch.setenv("IPFS_DATASETS_PY_GROK_CLI_CMD", str(fake_grok))
    monkeypatch.setenv("IPFS_DATASETS_PY_ROUTER_CACHE", "0")
    monkeypatch.setenv("IPFS_DATASETS_PY_ROUTER_RESPONSE_CACHE", "0")

    config = LlmRouterInvocation(
        repo_root=Path(__file__).resolve().parents[2],
        provider="grok",
        model_name="grok-4.5",
        allow_local_fallback=False,
        timeout_seconds=15,
        timeout_grace_seconds=2,
        max_new_tokens=16,
        python_executable=sys.executable,
        required_effective_providers=("grok",),
    )

    assert call_llm_router("child-smoke", config) == "supervisor:grok-4.5:child-smoke"


def test_grok_agent_runner_forwards_resolved_launch_policy(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    fake_codex = tmp_path / "codex"
    fake_codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_codex.chmod(0o700)
    fake_copilot = tmp_path / "copilot"
    fake_copilot.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_copilot.chmod(0o700)
    fake_goose = tmp_path / "goose"
    fake_goose.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_goose.chmod(0o700)
    fake_openai = tmp_path / "openai"
    fake_openai.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_openai.chmod(0o700)
    fake_docker = tmp_path / "docker"
    fake_docker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_docker.chmod(0o700)
    fake_codex_home = tmp_path / "codex-home"
    fake_codex_home.mkdir()
    fake_goose_store = tmp_path / "goose-store"
    fake_goose_store.mkdir()
    fake_vibe_store = tmp_path / "vibe-store"
    fake_vibe_store.mkdir()
    fake_docker_socket = tmp_path / "docker.sock"
    fake_docker_socket.write_text("socket sentinel\n", encoding="utf-8")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(fake_codex_home))
    monkeypatch.setenv("OPENAI_API_KEY", "parent-only-openai-authority")
    monkeypatch.setenv("GOOSE_CONFIG_DIR", str(fake_goose_store))
    monkeypatch.setenv("VIBE_HOME", str(fake_vibe_store))
    monkeypatch.setenv("DOCKER_HOST", f"unix://{fake_docker_socket}")
    monkeypatch.setattr(
        grok_cli_runner,
        "_select_grok_isolation_backend",
        lambda **_kwargs: grok_cli_runner.GROK_ISOLATION_GROK_SANDBOX,
    )

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs["env"])
        prompt_path = Path(cmd[cmd.index("--prompt-file") + 1])
        captured["prompt"] = prompt_path.read_text(encoding="utf-8")
        policy_path = Path(kwargs["env"]["GROK_HOME"]) / "sandbox.toml"
        captured["sandbox_policy"] = tomllib.loads(
            policy_path.read_text(encoding="utf-8")
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("repair the board"))
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(tmp_path),
            "--grok-bin",
            "/bin/true",
            "--model",
            "grok-4.5",
            "--max-turns",
            "1234",
            "--permission-mode",
            "acceptEdits",
            "--mode",
            "agent",
        ]
    )

    assert result == 0
    assert captured["prompt"] == "repair the board"
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[cmd.index("--model") + 1] == "grok-4.5"
    assert cmd[cmd.index("--max-turns") + 1] == "1234"
    assert cmd[cmd.index("--permission-mode") + 1] == "acceptEdits"
    assert cmd[cmd.index("--output-format") + 1] == "plain"
    assert "--always-approve" in cmd
    assert cmd[cmd.index("--sandbox") + 1] == (
        grok_cli_runner.GROK_PRIMARY_SANDBOX_PROFILE
    )
    assert cmd.count("--deny") >= len(
        grok_cli_runner.GROK_ISOLATION_DENY_RULES
    )
    assert "CODEX_HOME" not in captured["env"]
    assert "OPENAI_API_KEY" not in captured["env"]
    profile = captured["sandbox_policy"]["profiles"][
        grok_cli_runner.GROK_PRIMARY_SANDBOX_PROFILE
    ]
    assert profile["extends"] == "workspace"
    assert profile["restrict_network"] is True
    denied = set(profile["deny"])
    assert str(fake_codex) in denied
    assert str(fake_copilot) in denied
    assert str(fake_goose) in denied
    assert str(fake_openai) in denied
    assert str(fake_docker) in denied
    assert str(fake_docker_socket) in denied
    assert str(fake_codex_home) in denied
    assert str(fake_goose_store) in denied
    assert str(fake_vibe_store) in denied
    assert "Bash(grok *)" in cmd
    assert "Bash(/opt/ipfs-accelerate/grok *)" in cmd
    assert str(Path(captured["env"]["GROK_HOME"])) in denied
    assert "/proc" in denied
    assert "/dev" in denied
    assert cmd[cmd.index("--tools") + 1] == grok_cli_runner._SEALED_GROK_TOOLS
    disallowed = cmd[cmd.index("--disallowed-tools") + 1]
    assert "run_terminal_cmd" in disallowed
    assert "search_tool" in disallowed
    assert "use_tool" in disallowed
    assert "call_mcp_tool" in disallowed
    assert "list_mcp_resources" in disallowed
    assert "list_mcp_resource_templates" in disallowed
    assert "read_mcp_resource" in disallowed
    assert "fetch_mcp_resource" in disallowed
    assert "Agent" in disallowed


def test_isolated_grok_home_uses_private_profile_and_preserves_parent_env(
    tmp_path,
) -> None:
    source = {
        "HOME": str(tmp_path),
        "PATH": "/usr/bin",
        "CODEX_HOME": str(tmp_path / "missing-codex-home"),
        "OPENAI_API_KEY": "parent-only",
    }
    child = {"HOME": str(tmp_path), "PATH": "/usr/bin"}

    temporary_home, isolated, policy_path, denied_paths = (
        grok_cli_runner._isolated_grok_home(
            base_env=source,
            child_env=child,
            codex_fallback_command=(),
        )
    )
    try:
        assert Path(isolated["GROK_HOME"]) == policy_path.parent
        assert policy_path.parent != tmp_path / ".grok"
        profile = tomllib.loads(policy_path.read_text(encoding="utf-8"))[
            "profiles"
        ][grok_cli_runner.GROK_PRIMARY_SANDBOX_PROFILE]
        assert profile["restrict_network"] is True
        assert policy_path.parent in denied_paths
        assert Path("/proc") in denied_paths
        assert Path("/dev") in denied_paths
        assert source["OPENAI_API_KEY"] == "parent-only"
    finally:
        temporary_home.cleanup()


def test_docker_grok_command_masks_providers_and_mounts_only_workspace_rw(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    git_marker = workspace / ".git"
    git_marker.write_text("gitdir: /tmp/read-only-metadata\n", encoding="utf-8")
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("implement", encoding="utf-8")
    grok_bin = tmp_path / "grok"
    grok_bin.write_text("binary", encoding="utf-8")
    grok_bin.chmod(0o700)
    grok_home = tmp_path / "isolated-grok-home"
    grok_home.mkdir()
    mask_root = tmp_path / "unmounted-provider-masks"
    docker_config = tmp_path / "docker-config"
    docker_config.mkdir()
    (grok_home / "alternate-provider-deny-sentinel").write_text(
        "sentinel\n",
        encoding="utf-8",
    )
    codex_entrypoint = tmp_path / "providers" / "codex"
    codex_entrypoint.parent.mkdir()
    codex_entrypoint.write_text("provider", encoding="utf-8")
    codex_package = tmp_path / "provider-packages" / "codex"
    codex_package.mkdir(parents=True)
    copilot_store = tmp_path / "home" / ".copilot"
    copilot_store.mkdir(parents=True)
    grok_auth = tmp_path / "home" / ".grok" / "auth.json"
    grok_auth.parent.mkdir()
    grok_auth.write_text("{}\n", encoding="utf-8")
    peer_library = tmp_path / "home" / ".local" / "lib" / "python" / "openai"
    peer_library.mkdir(parents=True)
    (peer_library / "__init__.py").write_text("peer payload\n", encoding="utf-8")
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_isolation_binary",
        lambda: "/usr/bin/docker",
    )

    child_env = {
        "HOME": str(tmp_path / "home"),
        "GROK_HOME": str(grok_home),
        "PATH": "/usr/local/bin:/usr/bin",
        "XAI_API_KEY": "grok-only",
    }
    command = grok_cli_runner._docker_grok_command(
        grok_command=[
            str(grok_bin),
            "--model",
            "grok-4.5",
            "--prompt-file",
            str(prompt_path),
        ],
        grok_bin=grok_bin,
        workspace=workspace,
        prompt_path=prompt_path,
        grok_home=grok_home,
        base_env={
            "HOME": str(tmp_path / "home"),
            "PATH": child_env["PATH"],
        },
        child_env=child_env,
        denied_paths=(
            grok_home / "alternate-provider-deny-sentinel",
            codex_entrypoint,
            codex_package,
            copilot_store,
        ),
        mask_root=mask_root,
        docker_config=docker_config,
        container_name="ipfs-accelerate-grok-123-" + "a" * 32,
        cidfile=tmp_path / "container.cid",
        isolation_image="sha256:" + "b" * 64,
    )

    assert command[:7] == [
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "--config",
        str(docker_config),
        "run",
        "--pull=never",
        "--rm",
    ]
    assert command[command.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    tmpfs_specs = {
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "--tmpfs"
    }
    assert tmpfs_specs == {
        (
            "/tmp:rw,nosuid,nodev,noexec,mode=0700,"
            f"uid={os.getuid()},gid={os.getgid()}"
        ),
        (
            "/var/tmp:rw,nosuid,nodev,noexec,mode=0700,"
            f"uid={os.getuid()},gid={os.getgid()}"
        ),
    }
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert command[command.index("--name") + 1].startswith(
        "ipfs-accelerate-grok-"
    )
    assert command[command.index("--cidfile") + 1] == str(
        tmp_path / "container.cid"
    )
    assert "--sandbox" not in command
    mount_specs = {
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "--mount"
    }
    assert any(
        f"src={workspace}" in spec
        and f"dst={workspace}" in spec
        and "readonly" not in spec
        for spec in mount_specs
    )
    assert any(
        f"src={git_marker}" in spec
        and f"dst={git_marker}" in spec
        and "readonly" in spec
        for spec in mount_specs
    )
    assert any(
        f"dst={codex_entrypoint}" in spec and "readonly" in spec
        for spec in mount_specs
    )
    assert any(
        f"dst={codex_package}" in spec and "readonly" in spec
        for spec in mount_specs
    )
    assert any(
        f"dst={copilot_store}" in spec and "readonly" in spec
        for spec in mount_specs
    )
    assert any(
        f"src={grok_auth}" in spec
        and f"dst={grok_auth}" in spec
        and "readonly" in spec
        for spec in mount_specs
    )
    assert not any("src=" + str(copilot_store) in spec for spec in mount_specs)
    assert not any(
        "src=" + str(tmp_path / "home" / ".local" / "lib") in spec
        for spec in mount_specs
    )
    assert "OPENAI_API_KEY" not in command
    assert "grok-only" not in command
    assert not any(
        f"src={mask_root}" in spec and "readonly" not in spec
        for spec in mount_specs
    )
    image_index = command.index("sha256:" + "b" * 64)
    assert command[image_index + 1] == "/opt/ipfs-accelerate/grok"
    grok_cli_runner._restore_mask_permissions(mask_root)
    shutil.rmtree(mask_root)


def test_docker_command_reads_mode_0600_prompt_as_runtime_uid(tmp_path) -> None:
    docker_bin = grok_cli_runner._docker_isolation_binary()
    if not docker_bin:
        pytest.skip("pinned local Docker isolation image is unavailable")

    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    grok_home = tmp_path / "grok-home"
    grok_home.mkdir(mode=0o700)
    docker_config = tmp_path / "docker-config"
    docker_config.mkdir(mode=0o700)
    mask_root = tmp_path / "provider-masks"
    cidfile = tmp_path / "container.cid"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=tmp_path,
        prefix="asref-grok-prompt-",
        suffix=".txt",
        delete=False,
    ) as handle:
        handle.write("private prompt is readable\n")
        prompt_path = Path(handle.name)
    assert prompt_path.stat().st_mode & 0o777 == 0o600

    child_env = {
        "HOME": str(grok_home),
        "GROK_HOME": str(grok_home),
        "PATH": "/usr/bin:/bin",
    }
    image_id = grok_cli_runner._docker_isolation_image_id(
        docker_bin,
        docker_config=docker_config,
    )
    command = grok_cli_runner._docker_grok_command(
        grok_command=["/usr/bin/cat", str(prompt_path)],
        grok_bin=Path("/usr/bin/cat"),
        workspace=workspace,
        prompt_path=prompt_path,
        grok_home=grok_home,
        base_env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        child_env=child_env,
        denied_paths=(),
        mask_root=mask_root,
        docker_config=docker_config,
        container_name=(
            f"ipfs-accelerate-grok-{os.getpid()}-{uuid.uuid4().hex}"
        ),
        cidfile=cidfile,
        docker_bin=docker_bin,
        isolation_image=image_id,
    )
    try:
        completed = subprocess.run(
            command,
            env=grok_cli_runner._docker_control_env(child_env),
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout == "private prompt is readable\n"
    finally:
        grok_cli_runner._restore_mask_permissions(mask_root)
        shutil.rmtree(mask_root, ignore_errors=True)
        prompt_path.unlink(missing_ok=True)
        cidfile.unlink(missing_ok=True)


def test_docker_control_plane_ignores_hostile_redirect_and_image_override(
    tmp_path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    docker_config = tmp_path / "docker-config"
    docker_config.mkdir()

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["env"] = dict(kwargs["env"])
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="sha256:" + "c" * 64 + "\n",
            stderr="",
        )

    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)
    image_id = grok_cli_runner._docker_isolation_image_id(
        "/usr/bin/docker",
        docker_config=docker_config,
        base_env={
            "DOCKER_HOST": "tcp://attacker.invalid:2376",
            "DOCKER_CONTEXT": "attacker",
            "DOCKER_TLS_VERIFY": "1",
            "DOCKER_CERT_PATH": "/peer/certs",
            "IPFS_ACCELERATE_AGENT_GROK_ISOLATION_IMAGE": "attacker/image:latest",
        },
    )

    assert image_id == "sha256:" + "c" * 64
    command = captured["command"]
    assert command[:4] == [
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "--config",
        str(docker_config),
    ]
    assert grok_cli_runner.DEFAULT_GROK_ISOLATION_IMAGE in command
    assert "attacker/image:latest" not in command
    assert captured["env"] == {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"}


def test_trusted_grok_binary_requires_versioned_download_anchor(
    tmp_path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    download = home / ".grok" / "downloads" / "grok-0.2.118-linux-aarch64"
    download.parent.mkdir(parents=True)
    download.write_bytes(b"standalone grok")
    download.chmod(0o700)
    entrypoint = home / ".local" / "bin" / "grok"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.symlink_to(download)
    untrusted = tmp_path / "grok"
    untrusted.write_bytes(b"workspace-adjacent executable")
    untrusted.chmod(0o700)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    hostile_home = tmp_path / "hostile-grok-home"
    hostile_download = (
        hostile_home / "downloads" / "grok-0.2.118-linux-aarch64"
    )
    hostile_download.parent.mkdir(parents=True)
    hostile_download.write_bytes(b"forged grok")
    hostile_download.chmod(0o700)
    hostile_entrypoint = tmp_path / "hostile-bin" / "grok"
    hostile_entrypoint.parent.mkdir()
    hostile_entrypoint.symlink_to(hostile_download)
    monkeypatch.setattr(
        grok_cli_runner,
        "_operating_system_account_home",
        lambda: home,
    )
    monkeypatch.setenv("HOME", str(hostile_home))
    monkeypatch.setenv("GROK_HOME", str(hostile_home))

    assert grok_cli_runner._resolve_trusted_grok_bin(
        configured=str(entrypoint),
        workspace=workspace,
    ) == str(download)
    assert not grok_cli_runner._resolve_trusted_grok_bin(
        configured=str(untrusted),
        workspace=workspace,
    )
    assert not grok_cli_runner._resolve_trusted_grok_bin(
        configured=str(hostile_entrypoint),
        workspace=workspace,
    )


def test_codex_quota_executable_rejects_hostile_path(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    hostile_codex = tmp_path / "codex"
    hostile_codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    hostile_codex.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))

    assert not grok_cli_runner.resolve_codex_quota_fallback_executable(
        workspace=workspace,
    )
    assert not grok_cli_runner.resolve_codex_quota_fallback_executable(
        workspace=workspace,
        configured=str(hostile_codex),
    )


def test_grok_isolation_selection_fails_closed_without_kernel_backend(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        grok_cli_runner,
        "_grok_custom_sandbox_available",
        lambda: False,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_isolation_binary",
        lambda: "",
    )

    with pytest.raises(ValueError, match="provider isolation unavailable"):
        grok_cli_runner._select_grok_isolation_backend()


def test_quota_route_requires_docker_even_when_native_sandbox_exists(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        grok_cli_runner,
        "_grok_custom_sandbox_available",
        lambda: True,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_isolation_binary",
        lambda: "",
    )
    with pytest.raises(ValueError, match="requires the pinned local Docker"):
        grok_cli_runner._select_grok_isolation_backend(
            require_container_boundary=True,
        )

    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_isolation_binary",
        lambda: "/usr/bin/docker",
    )
    assert (
        grok_cli_runner._select_grok_isolation_backend(
            require_container_boundary=True,
        )
        == grok_cli_runner.GROK_ISOLATION_DOCKER
    )
