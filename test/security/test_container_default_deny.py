"""EAAEF-051: default-deny OCI run specs never invoke a live engine."""

from __future__ import annotations

import os

import pytest

from ipfs_accelerate_py.agent_supervisor.containers.contracts import (
    ContainerExecutionProfile,
    ContainerTrustError,
    IsolationPolicy,
    ResourceBounds,
)
from ipfs_accelerate_py.agent_supervisor.containers.oci_runner import (
    DEFAULT_NONROOT_USER,
    DEFAULT_PIDS_LIMIT,
    ROOTFUL_FALLBACK_ADMITTED_BY_DEFAULT,
    EngineAdmission,
    EngineMode,
    OciEngineAdmissionError,
    OciRunner,
    OciRunnerTrustError,
    build_oci_run_spec,
    default_engine_admission,
    select_engine_mode,
)


IMAGE_DIGEST = "sha256:" + ("a" * 64)
WORKTREE_ID = "worktree:eaaef-051"
TASK_ID = "task:EAAEF-051"
AUTHORITY_ID = "authority:supervisor"
DAEMON_IDENTITY = "sha256:" + ("c" * 64)
DAEMON_POLICY = "sha256:" + ("d" * 64)

_RESOURCES = ResourceBounds(
    cpu_millicores=4000,
    ram_mib=8192,
    disk_mib=16384,
    timeout_seconds=7200,
    gpu_count=0,
)


def _profile(**changes: object) -> ContainerExecutionProfile:
    values: dict[str, object] = {
        "image_digest": IMAGE_DIGEST,
        "worktree_id": WORKTREE_ID,
        "task_id": TASK_ID,
        "authority_id": AUTHORITY_ID,
        "resources": _RESOURCES,
        "policy": IsolationPolicy(),
    }
    values.update(changes)
    return ContainerExecutionProfile(**values)  # type: ignore[arg-type]


def _admitted_rootful() -> EngineAdmission:
    return EngineAdmission(
        rootless_supported=False,
        rootless_verified=False,
        rootful_fallback_admitted=True,
        independent_security_approval=True,
        daemon_identity_cid=DAEMON_IDENTITY,
        daemon_policy_cid=DAEMON_POLICY,
    )


def test_default_deny_run_spec() -> None:
    spec = build_oci_run_spec(_profile())
    argv = spec.argv
    assert spec.live_engine_invoked is False
    assert spec.network == "none"
    assert "--network=none" in argv
    assert spec.read_only is True
    assert "--read-only" in argv
    assert spec.cap_drop == ("ALL",)
    assert "--cap-drop=ALL" in argv
    assert spec.no_new_privileges is True
    assert "--security-opt=no-new-privileges" in argv
    assert spec.user == DEFAULT_NONROOT_USER
    assert spec.user != "0:0"
    assert f"--user={DEFAULT_NONROOT_USER}" in argv
    assert spec.privileged is False
    assert "--privileged" not in argv
    assert spec.docker_socket_mounted is False
    assert spec.inherit_host_environment is False
    assert spec.host_pid is False
    assert spec.host_ipc is False
    assert spec.rootless_preferred is True
    assert spec.engine_mode is EngineMode.ROOTLESS
    assert spec.rootful_fallback_admitted is False
    assert ROOTFUL_FALLBACK_ADMITTED_BY_DEFAULT is False
    assert default_engine_admission().rootful_fallback_admitted is False
    assert spec.engine_endpoint == f"unix:///run/user/{os.geteuid()}/docker.sock"
    assert not any(
        part.startswith("-v") or part.startswith("--volume") for part in argv
    )
    assert all("type=bind" not in part for part in argv)
    assert not any(
        "docker.sock" in part and not part.startswith("--host=") for part in argv
    )
    assert argv[:3] == ("docker", f"--host={spec.engine_endpoint}", "create")
    assert "run" not in argv
    runner = OciRunner()
    planned = runner.build_spec(_profile())
    assert planned.live_engine_invoked is False
    assert planned.network == "none"
    assert planned.docker_socket_mounted is False


def test_reject_docker_socket() -> None:
    profile = _profile()
    with pytest.raises(OciRunnerTrustError, match="docker.sock"):
        build_oci_run_spec(
            profile,
            extra_mounts=(
                {
                    "source": "/var/run/docker.sock",
                    "target": "/var/run/docker.sock",
                    "read_only": True,
                    "kind": "other",
                },
            ),
        )
    with pytest.raises(OciRunnerTrustError, match="docker.sock"):
        build_oci_run_spec(profile, worktree_source="/run/docker.sock")
    with pytest.raises(OciRunnerTrustError, match="docker.sock"):
        build_oci_run_spec(
            profile,
            extra_args=("-v", "/var/run/docker.sock:/var/run/docker.sock"),
        )
    with pytest.raises(ContainerTrustError, match="docker.sock"):
        IsolationPolicy(docker_socket_mounted=True)
    spec = build_oci_run_spec(profile)
    assert spec.docker_socket_mounted is False
    assert spec.mounts == ()


def test_reject_privileged() -> None:
    profile = _profile()
    with pytest.raises(OciRunnerTrustError, match="privileged"):
        build_oci_run_spec(profile, privileged=True)
    with pytest.raises(OciRunnerTrustError, match="privileged"):
        build_oci_run_spec(profile, extra_args=("--privileged",))
    with pytest.raises(ContainerTrustError, match="privileged"):
        IsolationPolicy(privileged=True)
    spec = build_oci_run_spec(profile)
    assert spec.privileged is False
    assert "--privileged" not in spec.argv
    assert spec.cap_drop == ("ALL",)
    assert spec.no_new_privileges is True


def test_resource_bounds_in_spec() -> None:
    spec = build_oci_run_spec(
        _profile(),
        pids_limit=256,
        command=("python3", "-c", "pass"),
    )
    assert spec.pids_limit == DEFAULT_PIDS_LIMIT
    assert spec.cpu_millicores == 4000
    assert spec.ram_mib == 8192
    assert spec.disk_mib == 16384
    assert spec.timeout_seconds == 7200
    assert spec.gpu_count == 0
    assert "--pids-limit=256" in spec.argv
    assert "--cpus=4" in spec.argv
    assert f"--memory={8192 * 1024 * 1024}" in spec.argv
    assert f"--memory-swap={8192 * 1024 * 1024}" in spec.argv
    assert f"--storage-opt=size={16384 * 1024 * 1024}" in spec.argv
    assert "--stop-timeout=7200" in spec.argv
    assert "--gpus=" not in "".join(spec.argv)
    assert spec.argv[-4:] == (IMAGE_DIGEST, "python3", "-c", "pass")
    bounded = build_oci_run_spec(
        _profile(
            resources=ResourceBounds(
                cpu_millicores=1500,
                ram_mib=512,
                disk_mib=1024,
                timeout_seconds=60,
                gpu_count=1,
            )
        )
    )
    assert bounded.cpu_millicores == 1500
    assert bounded.ram_mib == 512
    assert bounded.disk_mib == 1024
    assert bounded.timeout_seconds == 60
    assert bounded.gpu_count == 1
    assert "--cpus=1.5" in bounded.argv
    assert "--gpus=1" in bounded.argv
    assert bounded.live_engine_invoked is False


def test_rootful_fallback_requires_independent_admission_off_by_default() -> None:
    assert ROOTFUL_FALLBACK_ADMITTED_BY_DEFAULT is False
    assert EngineAdmission().rootful_fallback_admitted is False
    assert select_engine_mode().value == "rootless_engine"
    unsupported = EngineAdmission(rootless_supported=False, rootless_verified=False)
    with pytest.raises(OciEngineAdmissionError, match="independent policy admission"):
        select_engine_mode(unsupported)
    with pytest.raises(OciEngineAdmissionError, match="independent policy admission"):
        build_oci_run_spec(_profile(), engine_admission=unsupported)
    with pytest.raises(OciEngineAdmissionError, match="independent policy admission"):
        build_oci_run_spec(
            _profile(),
            engine_admission=EngineAdmission(
                rootless_supported=False,
                rootless_verified=False,
                rootful_fallback_admitted=True,
            ),
        )
    fallback = build_oci_run_spec(_profile(), engine_admission=_admitted_rootful())
    assert fallback.engine_mode is EngineMode.ROOTFUL_DAEMON_NONROOT_WORKER
    assert fallback.rootful_fallback_admitted is True
    assert fallback.user == DEFAULT_NONROOT_USER
    assert fallback.docker_socket_mounted is False
    assert fallback.privileged is False
    assert fallback.network == "none"
    assert fallback.cap_drop == ("ALL",)
    assert fallback.live_engine_invoked is False
    preferred = build_oci_run_spec(
        _profile(),
        engine_admission=EngineAdmission(
            rootless_supported=True,
            rootless_verified=True,
            rootful_fallback_admitted=True,
            independent_security_approval=True,
            daemon_identity_cid=DAEMON_IDENTITY,
            daemon_policy_cid=DAEMON_POLICY,
        ),
    )
    assert preferred.engine_mode is EngineMode.ROOTLESS
    assert preferred.rootful_fallback_admitted is False
