from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.runtime.provider_isolation import (
    CLI_LOG_COLLECTION_SCHEMA,
    KUBERNETES_LOG_VOLUME_MOUNT,
    PROVIDER_ISOLATION_BACKEND_ENV,
    PROVIDER_ISOLATION_DOCKER,
    PROVIDER_ISOLATION_GROK_SANDBOX,
    PROVIDER_ISOLATION_KUBERNETES,
    PROVIDER_ISOLATION_WORKTREE,
    collect_container_cli_logs,
    docker_logs_command,
    extract_cli_error_snippets,
    kubernetes_cluster_log_spec,
    kubectl_logs_command,
    kubectl_logs_selector_command,
    load_provider_cli_receipts,
    publish_provider_cli_logs,
    requested_provider_isolation_backend,
    select_provider_isolation_backend,
    supervisor_cli_failure_projection,
)


def test_default_isolation_is_worktree_sandbox_not_docker(monkeypatch) -> None:
    monkeypatch.delenv(PROVIDER_ISOLATION_BACKEND_ENV, raising=False)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    assert requested_provider_isolation_backend() == PROVIDER_ISOLATION_WORKTREE
    assert (
        select_provider_isolation_backend(
            docker_available=True,
            sandbox_available=True,
        )
        == PROVIDER_ISOLATION_GROK_SANDBOX
    )
    assert (
        select_provider_isolation_backend(
            docker_available=True,
            sandbox_available=False,
        )
        == PROVIDER_ISOLATION_WORKTREE
    )


def test_docker_isolation_is_opt_in(monkeypatch) -> None:
    monkeypatch.setenv(PROVIDER_ISOLATION_BACKEND_ENV, "docker")
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    assert (
        select_provider_isolation_backend(
            docker_available=True,
            sandbox_available=True,
        )
        == PROVIDER_ISOLATION_DOCKER
    )


def test_kubernetes_in_cluster_stays_on_worktree(monkeypatch) -> None:
    monkeypatch.delenv(PROVIDER_ISOLATION_BACKEND_ENV, raising=False)
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    assert (
        select_provider_isolation_backend(
            docker_available=True,
            sandbox_available=True,
            kubernetes_available=True,
        )
        == PROVIDER_ISOLATION_GROK_SANDBOX
    )
    monkeypatch.setenv(PROVIDER_ISOLATION_BACKEND_ENV, "kubernetes")
    assert (
        select_provider_isolation_backend(
            docker_available=True,
            sandbox_available=False,
            kubernetes_available=True,
        )
        == PROVIDER_ISOLATION_WORKTREE
    )


def test_quota_route_does_not_override_worktree_default(monkeypatch) -> None:
    monkeypatch.delenv(PROVIDER_ISOLATION_BACKEND_ENV, raising=False)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    assert (
        select_provider_isolation_backend(
            docker_available=True,
            sandbox_available=False,
            require_container_boundary=True,
        )
        == PROVIDER_ISOLATION_WORKTREE
    )


def test_quota_route_can_still_opt_in_docker(monkeypatch) -> None:
    monkeypatch.setenv(PROVIDER_ISOLATION_BACKEND_ENV, "docker")
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    assert (
        select_provider_isolation_backend(
            docker_available=True,
            sandbox_available=False,
            require_container_boundary=True,
        )
        == PROVIDER_ISOLATION_DOCKER
    )


def test_collect_container_cli_logs_extracts_errors(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_runner(command, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="ok\nERROR quota exhausted\nFatal: provider denied\n",
            stderr="",
        )

    receipt = collect_container_cli_logs(
        backend="docker",
        provider="grok",
        identity={"container_id": "abc123", "attempt_id": "attempt-1"},
        log_dir=tmp_path / "logs",
        returncode=86,
        log_command=docker_logs_command(
            docker_bin="/usr/bin/docker",
            docker_host="unix:///var/run/docker.sock",
            docker_config=str(tmp_path / "config"),
            container_id="sha256:abc123",
        ),
        runner=fake_runner,
    )
    assert receipt["schema"] == CLI_LOG_COLLECTION_SCHEMA
    assert receipt["backend"] == "docker"
    assert receipt["provider"] == "grok"
    assert receipt["returncode"] == 86
    assert receipt["error_count"] == 2
    assert "quota exhausted" in receipt["error_snippets"][0]
    log_text = (tmp_path / "logs" / "grok-attempt-1.cli.log").read_text(
        encoding="utf-8"
    )
    assert "Fatal: provider denied" in log_text
    stored = json.loads(
        (tmp_path / "logs" / "grok-attempt-1.cli-receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored["error_count"] == 2


def test_kubectl_logs_command_uses_pod_identity(monkeypatch) -> None:
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setenv("HOSTNAME", "aseh-lane-1")
    monkeypatch.setenv("KUBERNETES_NAMESPACE", "agent-supervisor")
    command = kubectl_logs_command(container="grok")
    assert command[:5] == [
        "kubectl",
        "--namespace",
        "agent-supervisor",
        "logs",
        "aseh-lane-1",
    ]
    assert command[-2:] == ["-c", "grok"]


def test_extract_cli_error_snippets_ignores_noise() -> None:
    text = "hello\ninfo ready\nTraceback (most recent call last):\nValueError: boom\n"
    snippets = extract_cli_error_snippets(text)
    assert snippets[0].startswith("Traceback")
    assert "ValueError: boom" in snippets[1]


def test_worktree_cli_logs_are_supervisor_visible(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_TASK_ID", "ASEH-011")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_TASK_ATTEMPT", "3")
    receipt = publish_provider_cli_logs(
        backend="worktree",
        provider="codex",
        identity={},
        returncode=1,
        captured_output="ERROR: disk quota exceeded\nFatal: write failed\n",
        log_dir=tmp_path / "logs",
    )
    assert receipt["backend"] == "worktree"
    assert receipt["provider"] == "codex"
    assert receipt["identity"]["task_id"] == "ASEH-011"
    assert receipt["error_count"] == 2
    loaded = load_provider_cli_receipts(
        tmp_path / "logs",
        task_id="ASEH-011",
        attempt="3",
    )
    assert len(loaded) == 1
    projection = supervisor_cli_failure_projection(loaded)
    assert projection["cli_error_count"] == 2
    assert "disk quota exceeded" in projection["cli_error_snippets"][0]


def test_kubernetes_cluster_log_spec_uses_shared_volume(monkeypatch) -> None:
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.96.0.1")
    monkeypatch.setenv("HOSTNAME", "aseh-lane-2")
    monkeypatch.setenv("KUBERNETES_NAMESPACE", "agent-supervisor")
    spec = kubernetes_cluster_log_spec(
        provider="claude-code",
        task_id="ASEH-021",
        attempt="2",
    )
    assert spec["in_cluster"] is True
    assert spec["log_volume"]["mount_path"] == KUBERNETES_LOG_VOLUME_MOUNT
    assert spec["log_volume"]["name"] == "agent-supervisor-provider-cli-logs"
    assert spec["labels"]["app.kubernetes.io/name"] == "agent-supervisor-provider"
    assert "ASEH-021" in spec["label_selector"]
    command = kubectl_logs_selector_command(
        namespace="agent-supervisor",
        selector=spec["label_selector"],
        container="claude-code",
    )
    assert "--prefix" in command
    assert "-l" in command
    assert command[-2:] == ["-c", "claude-code"]


def test_publish_kubernetes_cli_logs_uses_kubectl(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.96.0.1")
    monkeypatch.setenv("HOSTNAME", "aseh-lane-3")
    monkeypatch.setenv("KUBERNETES_NAMESPACE", "agent-supervisor")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_TASK_ID", "ASEH-031")
    seen: list[list[str]] = []

    def fake_runner(command, **_kwargs):
        seen.append(list(command))
        return SimpleNamespace(
            returncode=0,
            stdout="ERROR gemini quota denied\n",
            stderr="",
        )

    receipt = publish_provider_cli_logs(
        backend="kubernetes",
        provider="gemini",
        identity={"attempt": "4"},
        returncode=86,
        log_dir=tmp_path / "logs",
        runner=fake_runner,
    )
    assert receipt["provider"] == "gemini"
    assert receipt["backend"] == "kubernetes"
    assert receipt["error_count"] == 1
    assert seen
    assert seen[0][:5] == [
        "kubectl",
        "--namespace",
        "agent-supervisor",
        "logs",
        "aseh-lane-3",
    ]
    assert "quota denied" in receipt["error_snippets"][0]


def test_primary_grok_command_preserves_custom_sandbox_and_sealed_tools(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        grok_cli_runner,
        "_grok_custom_sandbox_available",
        lambda: True,
    )
    command = grok_cli_runner.build_grok_agent_command(
        workspace=Path("/tmp/workspace"),
        prompt_file=Path("/tmp/prompt.txt"),
        model="grok-4.6",
        max_turns=10,
        permission_mode="bypassPermissions",
        grok_bin="/usr/bin/grok",
    )
    assert command[command.index("--sandbox") + 1] == (
        grok_cli_runner.GROK_PRIMARY_SANDBOX_PROFILE
    ) == "ipfs-accelerate-provider-isolated"
    assert command[command.index("--tools") + 1] == (
        grok_cli_runner._SEALED_GROK_TOOLS
    )
    assert command[command.index("--disallowed-tools") + 1] == (
        grok_cli_runner._SEALED_GROK_DISALLOWED_TOOLS
    )
    denied = [command[index + 1] for index, arg in enumerate(command) if arg == "--deny"]
    assert denied == list(grok_cli_runner.GROK_ISOLATION_DENY_RULES)
    assert "--no-subagents" in command


def test_native_sandbox_unavailable_selects_pinned_docker_boundary(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        grok_cli_runner,
        "_grok_custom_sandbox_available",
        lambda: False,
    )
    monkeypatch.setattr(
        grok_cli_runner, "_docker_isolation_binary", lambda: "/usr/bin/docker"
    )
    assert grok_cli_runner._select_grok_isolation_backend() == (
        grok_cli_runner.GROK_ISOLATION_DOCKER
    ) == "docker"


def test_missing_native_and_docker_boundaries_refuses_execution(monkeypatch) -> None:
    monkeypatch.setattr(grok_cli_runner, "_grok_custom_sandbox_available", lambda: False)
    monkeypatch.setattr(grok_cli_runner, "_docker_isolation_binary", lambda: "")

    def no_host_execution(*args, **kwargs):
        raise AssertionError("An unavailable boundary must not retry a host process")

    monkeypatch.setattr(grok_cli_runner.subprocess, "run", no_host_execution)
    with pytest.raises(ValueError, match="Grok provider isolation unavailable"):
        grok_cli_runner._select_grok_isolation_backend()


def test_native_sandbox_probe_fails_closed_with_bounded_host_failures(
    monkeypatch,
) -> None:
    monkeypatch.setattr(grok_cli_runner.sys, "platform", "linux")
    monkeypatch.setattr(grok_cli_runner.shutil, "which", lambda name: "/reviewed/bwrap")
    calls = []
    outcomes = (
        SimpleNamespace(returncode=1),
        OSError("bwrap: setting up uid map: Permission denied"),
        subprocess.TimeoutExpired(["/reviewed/bwrap"], 5),
    )

    for outcome in outcomes:
        def probe(command, **kwargs):
            calls.append(command)
            assert command == ["/reviewed/bwrap", "--ro-bind", "/", "/", "--", "/bin/true"]
            assert kwargs == {
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "timeout": 5,
                "check": False,
            }
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(grok_cli_runner.subprocess, "run", probe)
        assert grok_cli_runner._grok_custom_sandbox_available() is False
    assert len(calls) == len(outcomes)


def test_container_required_route_never_downgrades_to_native_sandbox(
    monkeypatch,
) -> None:
    def no_native_downgrade():
        raise AssertionError("A container-required route must not consult native fallback")

    monkeypatch.setattr(grok_cli_runner, "_grok_custom_sandbox_available", no_native_downgrade)
    monkeypatch.setattr(grok_cli_runner, "_docker_isolation_binary", lambda: "")
    with pytest.raises(ValueError, match="requires the pinned local Docker"):
        grok_cli_runner._select_grok_isolation_backend(require_container_boundary=True)
    monkeypatch.setattr(
        grok_cli_runner, "_docker_isolation_binary", lambda: "/usr/bin/docker"
    )
    assert grok_cli_runner._select_grok_isolation_backend(
        require_container_boundary=True
    ) == grok_cli_runner.GROK_ISOLATION_DOCKER == "docker"


def test_unauthenticated_private_grok_home_keeps_custom_deny_profile(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    source_home = tmp_path / "empty-grok-home"
    source_home.mkdir()
    temporary_home, env, policy_path, denied = grok_cli_runner._isolated_grok_home(
        base_env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path / "sealed"),
            "GROK_HOME": str(source_home),
        },
        child_env={"PATH": "/usr/bin:/bin"},
        codex_fallback_command=(),
        workspace=workspace,
    )
    try:
        home = Path(temporary_home.name)
        sentinel = home / "alternate-provider-deny-sentinel"
        policy = tomllib.loads(Path(policy_path).read_text(encoding="utf-8"))
        profile = policy["profiles"]["ipfs-accelerate-provider-isolated"]
        assert profile["extends"] == "workspace"
        assert profile["restrict_network"] is True
        assert str(sentinel) in profile["deny"]
        assert home in denied and sentinel in denied
        assert env["GROK_HOME"] == temporary_home.name
        assert env["HOME"] == temporary_home.name
        assert not (home / "auth.json").exists()
    finally:
        temporary_home.cleanup()


def test_bwrap_deny_mounts_skip_symlink_entrypoints(tmp_path: Path) -> None:
    real = tmp_path / "downloads" / "grok-real"
    real.parent.mkdir()
    real.write_text("#!/bin/sh\n", encoding="utf-8")
    real.chmod(0o755)
    link_dir = tmp_path / ".local" / "bin"
    link_dir.mkdir(parents=True)
    link = link_dir / "grok"
    link.symlink_to(real)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    state = tmp_path / "state"
    state.mkdir()
    sentinel = state / "sentinel"
    sentinel.write_text("sentinel\n", encoding="utf-8")
    denied = grok_cli_runner._grok_isolation_deny_paths(
        base_env={"PATH": str(link_dir), "HOME": str(tmp_path)},
        codex_fallback_command=(),
        grok_home=state,
        sentinel_path=sentinel,
        workspace=workspace,
    )
    mounted = tuple(
        path for path in denied if grok_cli_runner._bwrap_exact_mount_path(path)
    )
    assert link.resolve(strict=True) == real.resolve(strict=True)
    assert real.resolve(strict=True) in mounted
    assert link not in mounted
    assert Path("/opt/ipfs-accelerate/grok") in denied
    assert not grok_cli_runner._bwrap_exact_mount_path(
        Path("/opt/ipfs-accelerate/grok")
    )
    assert not grok_cli_runner._bwrap_exact_mount_path(link)


def test_sandbox_profile_mounts_sentinel_not_private_home(tmp_path: Path) -> None:
    real = tmp_path / "downloads" / "grok-real"
    real.parent.mkdir()
    real.write_text("#!/bin/sh\n", encoding="utf-8")
    real.chmod(0o755)
    link_dir = tmp_path / ".local" / "bin"
    link_dir.mkdir(parents=True)
    link = link_dir / "grok"
    link.symlink_to(real)
    user = tmp_path / "user"
    auth = user / ".grok" / "auth.json"
    auth.parent.mkdir(parents=True)
    auth.write_text("{}\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    temporary_home, _env, policy_path, denied = grok_cli_runner._isolated_grok_home(
        base_env={"PATH": str(link_dir), "HOME": str(user)},
        child_env={"PATH": str(link_dir)},
        codex_fallback_command=(),
        workspace=workspace,
    )
    try:
        grok_home = Path(temporary_home.name)
        sentinel = grok_home / "alternate-provider-deny-sentinel"
        text = Path(policy_path).read_text(encoding="utf-8")
        assert json.dumps(str(grok_home)) not in text
        assert json.dumps(str(sentinel)) in text
        assert json.dumps(str(real.resolve(strict=True))) not in text
        assert json.dumps("/proc") not in text
        assert json.dumps("/dev") not in text
        assert json.dumps(str(auth.resolve(strict=True))) not in text
        assert grok_home in denied
        assert sentinel in denied
        assert real.resolve(strict=True) in denied
        assert auth.resolve(strict=True) in denied
    finally:
        temporary_home.cleanup()
