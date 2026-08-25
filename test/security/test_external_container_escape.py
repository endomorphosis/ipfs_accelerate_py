"""EAAEF-123: container escape attempts fail closed on the admitted profile.

evidence_mode: contract_fail_closed

IsolationPolicy and build_oci_run_spec refuse docker.sock, privileged, host PID,
cap-add, device, cgroup and symlink escapes.  A live engine is not admitted or
invoked.  This is not a claim that a live escape-harness ran.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.containers.contracts import (
    ContainerExecutionProfile,
    ContainerTrustError,
    IsolationPolicy,
    ResourceBounds,
)
from ipfs_accelerate_py.agent_supervisor.containers.oci_runner import (
    OciRunnerTrustError,
    build_oci_run_spec,
)

SECCOMP = Path("containers/external-agent/seccomp.json")
IMAGE_DIGEST = "sha256:" + ("a" * 64)


def _profile() -> ContainerExecutionProfile:
    return ContainerExecutionProfile(
        image_digest=IMAGE_DIGEST,
        worktree_id="worktree:eaaef-123",
        task_id="task:EAAEF-123",
        authority_id="authority:supervisor",
        resources=ResourceBounds(
            cpu_millicores=1000,
            ram_mib=512,
            disk_mib=1024,
            timeout_seconds=60,
            gpu_count=0,
        ),
        policy=IsolationPolicy(),
    )


def test_seccomp_profile_is_default_deny() -> None:
    payload = json.loads(SECCOMP.read_text(encoding="utf-8"))
    assert payload["defaultAction"] == "SCMP_ACT_ERRNO"
    names = payload["syscalls"][0]["names"]
    assert "ptrace" not in names
    assert "mount" not in names
    assert "reboot" not in names
    assert "openat" in names


def test_escape_vectors_are_refused_without_live_engine() -> None:
    with pytest.raises(ContainerTrustError, match="docker.sock"):
        IsolationPolicy(docker_socket_mounted=True)
    with pytest.raises(ContainerTrustError, match="privileged"):
        IsolationPolicy(privileged=True)
    spec = build_oci_run_spec(_profile())
    assert spec.live_engine_invoked is False
    assert spec.privileged is False
    assert spec.docker_socket_mounted is False
    with pytest.raises(OciRunnerTrustError):
        build_oci_run_spec(
            _profile(),
            extra_mounts=(
                {
                    "source": "/var/run/docker.sock",
                    "target": "/var/run/docker.sock",
                    "read_only": True,
                    "kind": "other",
                },
            ),
        )
    with pytest.raises(OciRunnerTrustError, match="privileged"):
        build_oci_run_spec(_profile(), privileged=True)
    with pytest.raises(OciRunnerTrustError, match="isolation escape"):
        build_oci_run_spec(_profile(), extra_args=("--pid=host",))
    with pytest.raises(OciRunnerTrustError, match="isolation escape"):
        build_oci_run_spec(_profile(), extra_args=("--cap-add=SYS_ADMIN",))
    with pytest.raises(OciRunnerTrustError, match="isolation escape"):
        build_oci_run_spec(_profile(), extra_args=("--device=/dev/kmsg",))
    with pytest.raises(OciRunnerTrustError, match="bind mounts"):
        build_oci_run_spec(
            _profile(),
            extra_args=("--mount", "type=bind,src=/sys/fs/cgroup,dst=/host-cgroup"),
        )
    with pytest.raises(OciRunnerTrustError, match="mount path is not admitted"):
        build_oci_run_spec(_profile(), worktree_source="/workspace/../escape")
    spec = build_oci_run_spec(_profile())
    assert spec.live_engine_invoked is False
    assert spec.host_pid is False
