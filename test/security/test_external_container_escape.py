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
    assert payload["schema"].endswith("external-agent-seccomp-profile@1")
    assert payload["task_id"] == "EAAEF-123"
    assert payload["evidence_mode"] == "contract_fail_closed"
    assert payload["qualification_scope"] == "offline_profile_contract_only"
    assert payload["task_completion_claimed"] is False
    assert payload["production_qualification_claimed"] is False
    assert payload["live_escape_harness_ran"] is False
    assert payload["live_runtime_invoked"] is False
    assert payload["defaultAction"] == "SCMP_ACT_ERRNO"
    allowed = {
        name
        for rule in payload["syscalls"]
        if rule["action"] == "SCMP_ACT_ALLOW"
        for name in rule["names"]
    }
    assert {
        "add_key",
        "bpf",
        "delete_module",
        "finit_module",
        "init_module",
        "kexec_file_load",
        "kexec_load",
        "keyctl",
        "mount",
        "open_by_handle_at",
        "perf_event_open",
        "pivot_root",
        "process_vm_readv",
        "process_vm_writev",
        "ptrace",
        "reboot",
        "request_key",
        "setns",
        "umount2",
        "unshare",
        "userfaultfd",
    }.isdisjoint(allowed)
    assert "openat" in allowed

    clone_rules = [rule for rule in payload["syscalls"] if "clone" in rule["names"]]
    assert len(clone_rules) == 1
    clone_rule = clone_rules[0]
    assert clone_rule["names"] == ["clone"]
    assert clone_rule["args"] == [
        {
            "index": 0,
            "value": 0x7E020000,
            "valueTwo": 0,
            "op": "SCMP_CMP_MASKED_EQ",
        }
    ]


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
