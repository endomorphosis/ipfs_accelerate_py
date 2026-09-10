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


@pytest.fixture
def factory_case(tmp_path, monkeypatch):
    home = tmp_path / "provider-home"
    home.mkdir(mode=0o700)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("test prompt")
    prompt.chmod(0o600)
    captured = {"spawns": [], "waits": []}
    class Launcher:
        def wait(self, timeout=None):
            return 0
        def poll(self):
            return 0
        def kill(self):
            pytest.fail("successful launcher was killed")
    class Watchdog:
        pid = 424242
        start_ticks = 123
        def wait(self, timeout=None):
            captured["waits"].append(timeout)
            return 0
        def poll(self):
            return 0
        def kill(self):
            pytest.fail("prepared watchdog was killed")
        def terminate(self):
            pytest.fail("prepared watchdog was terminated")
    def popen(argv, **kwargs):
        captured["spawns"].append((list(argv), kwargs))
        return Launcher()
    real_resolve, real_stat = Path.resolve, Path.stat
    monkeypatch.setattr(Path, "resolve", lambda p, **kw: p if str(p) == "/usr/bin/docker" else real_resolve(p, **kw))
    monkeypatch.setattr(Path, "stat", lambda p, **kw: SimpleNamespace(st_uid=0, st_mode=stat.S_IFREG | 0o755) if str(p) == "/usr/bin/docker" else real_stat(p, **kw))
    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: pytest.fail("unqualified Docker effect"))
    monkeypatch.setattr(runner, "_docker_cleanup_binding_path", lambda name: None)
    monkeypatch.setattr(runner, "_docker_cleanup_watchdog_env", lambda: {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"})
    monkeypatch.setattr(runner, "_read_detached_docker_cleanup_watchdog", lambda *a, **kw: Watchdog())
    monkeypatch.setattr(process_security, "require_state_authority_handoff_ptrace_protection", lambda: None)
    return home, prompt, captured


def test_factory_preserves_signed_coordinates_with_private_watchdog(tmp_path, factory_case):
    home, prompt, captured = factory_case
    lease_root = tmp_path / "asref-codex-container-signed"
    name = "ipfs-accelerate-codex-123-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    lease = runner._DockerContainerLease.create(
        "/usr/bin/docker", provider="codex", provider_home=home, prompt_path=prompt,
        authorized_container_name=name, authorized_lease_root=lease_root,
    )
    try:
        assert len(captured["spawns"]) == 1
        argv, options = captured["spawns"][0]
        assert argv[4] == runner._DOCKER_CLEANUP_WATCHDOG_LAUNCHER_ARG
        assert argv[argv.index("--container-name") + 1] == name
        assert argv[argv.index("--lease-root") + 1] == str(lease_root)
        assert argv[argv.index("--engine-endpoint") + 1] == lease.engine_endpoint
        control_fd = int(argv[argv.index("--control-fd") + 1])
        assert control_fd in options["pass_fds"]
        assert options["stdin"] == subprocess.DEVNULL
        assert not hasattr(lease, "_write_fd")
        assert lease._watchdog.start_ticks == 123
        assert lease._control_socket.fileno() >= 0
    finally:
        lease._control_socket.close()
        lease._abort_provider_start()


def test_constructor_denial_closes_socket_but_preserves_watchdog(factory_case, monkeypatch):
    home, prompt, captured = factory_case
    sockets = []
    def deny(self, **kwargs):
        sockets.append(kwargs["control_socket"])
        raise ValueError("prepared binding denied")
    monkeypatch.setattr(runner._DockerContainerLease, "__init__", deny)
    with pytest.raises(ValueError, match="prepared binding denied"):
        runner._DockerContainerLease.create(
            "/usr/bin/docker", provider="codex", provider_home=home, prompt_path=prompt,
        )
    assert len(captured["spawns"]) == 1
    assert sockets[0].fileno() == -1
    assert captured["waits"]


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
