"""No live Docker: merged cleanup factories and native effect boundaries."""
from __future__ import annotations

import inspect
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import process_security








@pytest.mark.parametrize("observation", [None, {}, {"logical_attempt_id": "attempt"}])
def test_missing_effect_observation_denies_before_docker_lookup(tmp_path, monkeypatch, observation):
    monkeypatch.setattr(runner, "_docker_isolation_binary", lambda: pytest.fail("Docker lookup before authority"))
    with pytest.raises(ValueError, match="exact durable observation"):
        runner._run_codex_quota_fallback_in_docker(
            ["codex", "exec"], workspace=tmp_path, prompt="test",
            prompt_path=tmp_path / "prompt", base_env={}, effect_observation=observation,
            effect_claim=lambda value: None, effect_terminal=lambda code: None,
        )


def test_native_create_adapter_rejects_legacy_or_unprepared_leases(tmp_path):
    with pytest.raises(ValueError, match="native prepared cleanup binding"):
        runner._create_bound_provider_container(
            SimpleNamespace(), ["docker", "create"], cwd=tmp_path, env={},
        )
    lease = object.__new__(runner._DockerContainerLease)
    lease.cleanup_binding_record = None
    with pytest.raises(ValueError, match="native prepared cleanup binding"):
        runner._create_bound_provider_container(lease, ["docker", "create"], cwd=tmp_path, env={})


def test_old_signed_entrypoint_cannot_acquire_native_start_authority(tmp_path, monkeypatch):
    # Deliberately isolate command-layout admission from separately tested
    # binding validation; no immutable receipt or effect is produced.
    lease = object.__new__(runner._DockerContainerLease)
    lease.cleanup_binding_record = tmp_path / "binding"
    lease._cleanup_binding_identity = {"inode": 1}
    lease._cleanup_binding_value = {"binding_state": "prepared_no_dispatch"}
    monkeypatch.setattr(lease, "create_inert_container", lambda *a, **kw: pytest.fail("legacy create dispatched"))
    with pytest.raises(ValueError, match="lacks qualified native provider-start custody"):
        runner._create_bound_provider_container(
            lease, ["docker", "create", "--entrypoint=/usr/bin/env"], cwd=tmp_path, env={},
        )


def test_unfenced_cleanup_observes_absence_without_removal(tmp_path, monkeypatch):
    commands = []
    def run(argv, **kwargs):
        commands.append(argv)
        assert "rm" not in argv
        return subprocess.CompletedProcess(argv, 0, stdout=b"")
    monkeypatch.setattr(runner.subprocess, "run", run)
    runner._remove_exact_docker_container(
        docker_bin="/usr/bin/docker", docker_config=tmp_path,
        container_name="ipfs-accelerate-codex-123-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", settle_for_creation=False,
    )
    assert len(commands) == 2


def test_malformed_termination_fence_never_issues_docker_command(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: pytest.fail("unverified removal"))
    with pytest.raises(ValueError):
        runner._remove_exact_docker_container(
            docker_bin="/usr/bin/docker", docker_config=tmp_path,
            container_name="ipfs-accelerate-codex-123-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", settle_for_creation=False,
            termination_fence={"provider": "codex"}, issue_removal=True,
        )


def test_cleanup_constructor_and_removal_signatures_share_native_contract():
    parameters = inspect.signature(runner._DockerContainerLease).parameters
    assert "write_fd" not in parameters
    assert {"control_socket", "watchdog", "effect_observation", "cleanup_binding_record"} <= parameters.keys()
    removal = inspect.signature(runner._remove_exact_docker_container).parameters
    assert {"termination_fence", "issue_removal", "deadline", "pass_fds", "engine_endpoint"} <= removal.keys()


@pytest.mark.parametrize("missing_callback", ["claim", "terminal"])
def test_complete_observation_still_requires_both_effect_callbacks(tmp_path, monkeypatch, missing_callback):
    monkeypatch.setattr(runner, "_docker_isolation_binary", lambda: pytest.fail("Docker lookup before authority"))
    observation = {name: "test-binding" for name in runner._DOCKER_EFFECT_OBSERVATION_FIELDS}
    with pytest.raises(ValueError, match="exact durable observation"):
        runner._run_codex_quota_fallback_in_docker(
            ["codex", "exec"], workspace=tmp_path, prompt="test",
            prompt_path=tmp_path / "prompt", base_env={}, effect_observation=observation,
            effect_claim=None if missing_callback == "claim" else lambda value: None,
            effect_terminal=None if missing_callback == "terminal" else lambda code: None,
        )


@pytest.mark.parametrize("lease_kind", ["absent", "fabricated", "different_coordinates"])
def test_typed_create_path_requires_actual_matching_native_lease(tmp_path, monkeypatch, lease_kind):
    monkeypatch.setattr(runner, "_create_grok_container_and_build_start_command", lambda *a, **kw: pytest.fail("unbound create"))
    lease = None
    if lease_kind == "fabricated":
        lease = SimpleNamespace(docker_bin="/usr/bin/docker", docker_config=tmp_path, cidfile=tmp_path / "cid")
    elif lease_kind == "different_coordinates":
        lease = object.__new__(runner._DockerContainerLease)
        lease.docker_bin = "/usr/bin/docker"
        lease.docker_config = tmp_path / "other"
        lease.cidfile = tmp_path / "cid"
    with pytest.raises(ValueError, match="exact native cleanup lease"):
        runner._run_created_grok_container_with_typed_failure_capture(
            ["docker", "create"], docker_bin="/usr/bin/docker", docker_config=tmp_path,
            cidfile=tmp_path / "cid", workspace=tmp_path, env={}, docker_lease=lease,
        )


def test_incompatible_signed_profile_denies_before_effect_claim_in_actual_caller(tmp_path, monkeypatch):
    home = tmp_path / "provider-home"
    home.mkdir()
    events = []
    lease = object.__new__(runner._DockerContainerLease)
    lease.cleanup_binding_record = tmp_path / "binding"
    lease._cleanup_binding_identity = {"inode": 1}
    lease._cleanup_binding_value = {"binding_state": "prepared_no_dispatch"}
    lease.docker_config = tmp_path / "docker-config"
    lease.lease_root = tmp_path / "lease"
    lease.container_name = "ipfs-accelerate-codex-123-" + "a" * 32
    lease.cidfile = tmp_path / "cid"
    lease.preserve_for_recovery = False
    monkeypatch.setattr(lease, "bind_isolation_image", lambda image: events.append("image_validated"))
    monkeypatch.setattr(lease, "create_inert_container", lambda *a, **kw: pytest.fail("incompatible create dispatched"))
    monkeypatch.setattr(lease, "close", lambda **kw: events.append(("closed", kw["docker_run_finished"])))
    monkeypatch.setattr(lease, "mark_cas_owned", lambda: pytest.fail("unstarted effect claimed"))
    monkeypatch.setattr(runner._DockerContainerLease, "create", lambda *a, **kw: lease)
    monkeypatch.setattr(runner, "resolve_codex_quota_fallback_executable", lambda **kw: "/trusted/codex")
    monkeypatch.setattr(runner, "_docker_isolation_binary", lambda: "/usr/bin/docker")
    monkeypatch.setattr(runner, "_isolated_codex_quota_fallback_home", lambda **kw: (
        SimpleNamespace(name=str(home), cleanup=lambda: None), {}, tmp_path / "auth",
    ))
    monkeypatch.setattr(runner, "_inspect_signed_worker_network", lambda **kw: None)
    monkeypatch.setattr(runner, "_docker_codex_task_toolchain_image_id", lambda *a, **kw: "sha256:" + "a" * 64)
    monkeypatch.setattr(runner, "_docker_codex_fallback_command", lambda **kw: ["docker", "create", "--entrypoint=/usr/bin/env"])
    monkeypatch.setattr(runner, "_codex_provider_argv_receipt", lambda *a: ["/trusted/codex", "exec"])
    monkeypatch.setattr(runner, "_validated_codex_auth_path", lambda **kw: None)
    monkeypatch.setattr(runner, "_robust_remove_runner_temp_tree", lambda path: None)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: pytest.fail("unqualified Docker effect"))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: pytest.fail("unqualified provider start"))
    profile = SimpleNamespace(container_name=lease.container_name, lease_root=lease.lease_root, authorization=None)
    observation = {name: "test-binding" for name in runner._DOCKER_EFFECT_OBSERVATION_FIELDS}
    with pytest.raises(ValueError, match="lacks qualified native provider-start custody"):
        runner._run_codex_quota_fallback_in_docker(
            ["/trusted/codex", "exec"], workspace=tmp_path, prompt="test",
            prompt_path=tmp_path / "prompt", base_env={}, network_profile=profile,
            effect_observation=observation,
            effect_claim=lambda context: pytest.fail("effect-start callback preceded profile denial"),
            effect_terminal=lambda code: pytest.fail("unstarted effect received a terminal callback"),
        )
    assert events == ["image_validated", ("closed", False)]


def test_native_lease_rejects_unbound_rootless_engine_before_dispatch(tmp_path, monkeypatch):
    import os

    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: pytest.fail("foreign engine queried"))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: pytest.fail("watchdog dispatched"))
    with pytest.raises(ValueError, match="engine endpoint is not locally admitted"):
        runner._validated_local_docker_engine_endpoint(
            f"unix:///run/user/{os.geteuid()}/docker.sock"
        )


def test_ordinary_compatibility_route_cannot_complete_protected_effect(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_docker_isolation_binary", lambda: pytest.fail("Docker lookup before authority"))
    with pytest.raises(ValueError, match="protected Codex effect requires native cleanup custody"):
        runner._run_legacy_codex_quota_fallback_in_docker(
            ["codex", "exec"], workspace=tmp_path, prompt="test",
            prompt_path=tmp_path / "prompt", base_env={},
            effect_claim=lambda value: pytest.fail("unqualified effect claim"),
            effect_terminal=lambda value: pytest.fail("unqualified completion"),
        )


def test_current_codex_builder_mounts_repository_git_directory_once(tmp_path, monkeypatch):
    from test.api.test_terminal_cleanup_observer import _current_launch_context
    from test.api.test_llm_router_agent_supervisor_fallback_route import _discard_live_cleanup_inputs

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(["git", "-C", str(workspace), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm", "fixture"], check=True)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    monkeypatch.setattr(runner.tempfile, "tempdir", str(private))
    context, paths = _current_launch_context(tmp_path, workspace)
    try:
        mounts = context["mount_receipt"]
        expected = f"type=bind,src={workspace / '.git'},dst={workspace / '.git'},readonly"
        assert mounts.count(expected) == 1
        assert len(mounts) == len(set(mounts))
    finally:
        _discard_live_cleanup_inputs(paths)


@pytest.mark.parametrize("prompt", ["ordinary prompt", "ASEH_PROVIDER_START_FENCE_V2\nrun task"])
def test_created_recovery_cannot_reinterpret_prompt_as_start_authority(tmp_path, monkeypatch, prompt):
    monkeypatch.setattr(runner, "_docker_isolation_binary", lambda: pytest.fail("recovery reached Docker"))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: pytest.fail("recovery started provider"))
    with pytest.raises(ValueError, match="lacks qualified native adoption custody"):
        runner._start_recorded_codex_effect({"container_id": "sha256:" + "a" * 64}, prompt=prompt)


@pytest.mark.parametrize("provider", ["codex", "grok"])
def test_native_factory_is_disabled_before_paths_sockets_or_processes(tmp_path, monkeypatch, provider):
    monkeypatch.setattr(runner.tempfile, "mkdtemp", lambda *a, **kw: pytest.fail("lease directory created"))
    monkeypatch.setattr(runner.socket, "socketpair", lambda *a, **kw: pytest.fail("native socket created"))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: pytest.fail("watchdog dispatched"))
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: pytest.fail("Docker dispatched"))
    for name in runner._DOCKER_WATCHDOG_LIFECYCLE_ENV_NAMES:
        monkeypatch.setenv(name, "syntactically-present-but-not-owner-authority")
    with pytest.raises(RuntimeError, match="disabled until supervisor owner STOP"):
        runner._DockerContainerLease.create(
            "/usr/bin/docker", provider=provider,
            provider_home=tmp_path / "home", prompt_path=tmp_path / "prompt",
            authorized_lease_root=tmp_path / "lease",
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mode", [
    runner._DOCKER_REMOVAL_ISSUER_LAUNCHER_ARG,
    runner._DOCKER_REMOVAL_ISSUER_ARG,
    runner._DOCKER_CLEANUP_WATCHDOG_LAUNCHER_ARG,
    runner._DOCKER_CLEANUP_WATCHDOG_ARG,
])
def test_native_cli_modes_are_disabled_before_parsing_or_dispatch(tmp_path, mode):
    import sys

    result = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(runner.__file__).resolve()), mode,
         "--lease-root", str(tmp_path / "lease"), "--container-name", "never-dispatch"],
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert result.returncode == 125
    assert "disabled until supervisor owner STOP" in result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


def test_native_inert_create_rejects_unbound_lease_before_any_effect(tmp_path, monkeypatch):
    lease = object.__new__(runner._DockerContainerLease)
    lease.cleanup_binding_record = None
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: pytest.fail("unbound Docker create dispatched"))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: pytest.fail("unbound create issuer dispatched"))
    with pytest.raises(ValueError, match="requires native prepared cleanup binding"):
        lease.create_inert_container(["docker", "create"], cwd=tmp_path, env={})
    assert list(tmp_path.iterdir()) == []
