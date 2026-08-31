from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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


def test_worktree_isolation_uses_builtin_workspace_sandbox() -> None:
    assert (
        grok_cli_runner.grok_sandbox_cli_profile(
            grok_cli_runner.GROK_ISOLATION_WORKTREE
        )
        == grok_cli_runner.GROK_WORKTREE_SANDBOX_PROFILE
    )
    assert grok_cli_runner.GROK_WORKTREE_SANDBOX_PROFILE == "workspace"
    assert (
        grok_cli_runner.grok_sandbox_cli_profile(
            grok_cli_runner.GROK_ISOLATION_GROK_SANDBOX
        )
        == grok_cli_runner.GROK_PRIMARY_SANDBOX_PROFILE
    )
    command = grok_cli_runner.build_grok_agent_command(
        workspace=Path("/tmp/workspace"),
        prompt_file=Path("/tmp/prompt.txt"),
        model="grok-4.6",
        max_turns=10,
        permission_mode="bypassPermissions",
        grok_bin="/usr/bin/grok",
    )
    assert command[command.index("--sandbox") + 1] == "workspace"
    assert grok_cli_runner.GROK_PRIMARY_SANDBOX_PROFILE not in command


def test_grok_stderr_detects_bwrap_host_failure() -> None:
    assert grok_cli_runner.grok_stderr_is_sandbox_host_failure(
        "bwrap: setting up uid map: Permission denied"
    )
    assert grok_cli_runner.grok_stderr_is_sandbox_host_failure(
        "bwrap: setting up gid map: Permission denied"
    )
    assert not grok_cli_runner.grok_stderr_is_sandbox_host_failure(
        "quota exhausted"
    )


def test_rewrite_grok_sandbox_profile_swaps_custom_to_workspace() -> None:
    command = [
        "grok",
        "--sandbox",
        grok_cli_runner.GROK_PRIMARY_SANDBOX_PROFILE,
        "--prompt-file",
        "prompt.txt",
    ]
    rewritten = grok_cli_runner._rewrite_grok_sandbox_profile(
        command,
        grok_cli_runner.GROK_WORKTREE_SANDBOX_PROFILE,
    )
    assert rewritten[rewritten.index("--sandbox") + 1] == "workspace"


def test_isolated_grok_home_worktree_skips_custom_deny_profile(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    temporary_home, env, policy_path, _denied = grok_cli_runner._isolated_grok_home(
        base_env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path / "sealed")},
        child_env={"PATH": "/usr/bin:/bin"},
        codex_fallback_command=(),
        workspace=workspace,
        populate_credentials=False,
        isolation_backend=grok_cli_runner.GROK_ISOLATION_WORKTREE,
    )
    try:
        text = Path(policy_path).read_text(encoding="utf-8")
        assert grok_cli_runner.GROK_PRIMARY_SANDBOX_PROFILE not in text
        assert "deny = [" not in text
        assert env["GROK_HOME"] == temporary_home.name
    finally:
        temporary_home.cleanup()
