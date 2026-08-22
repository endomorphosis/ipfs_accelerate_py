"""Deterministic tests for EAAEF-050 container execution contracts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from ipfs_accelerate_py.agent_supervisor.containers.contracts import (
    ABSOLUTE_MAX_CPU_MILLICORES,
    ABSOLUTE_MAX_DISK_MIB,
    ABSOLUTE_MAX_GPU_COUNT,
    ABSOLUTE_MAX_RAM_MIB,
    ABSOLUTE_MAX_TIMEOUT_SECONDS,
    ARTIFACT_MANIFEST_INTERFACE,
    CONTAINER_CHECKPOINT_INTERFACE,
    CONTAINER_CONTRACT_FAMILY,
    CONTAINER_CONTRACT_VERSION,
    CONTAINER_EXECUTION_PROFILE_INTERFACE,
    CONTAINER_RECEIPT_INTERFACE,
    CONTRACT_VERSION,
    SCHEMA_VERSION,
    WORKER_LEASE_INTERFACE,
    ArtifactManifest,
    ContainerBoundsError,
    ContainerCheckpoint,
    ContainerContractError,
    ContainerExecutionProfile,
    ContainerIdentityError,
    ContainerOutcome,
    ContainerReceipt,
    ContainerTrustError,
    ContainerVersionError,
    IsolationPolicy,
    NetworkPolicy,
    ResourceBounds,
    ResourceUse,
    WorkerLease,
    bind_container_execution,
    canonical_container_json_bytes,
    decode_container_contract,
)


IMAGE_DIGEST = "sha256:" + ("a" * 64)
ARTIFACT_CID = "sha256:" + ("b" * 64)
WORKTREE_ID = "worktree:eaaef-050"
TASK_ID = "task:EAAEF-050"
AUTHORITY_ID = "authority:supervisor"
WORKER_ID = "worker:container-1"
FIXED_MS = 1_700_000_000_000

_RESOURCES = ResourceBounds(
    cpu_millicores=4000,
    ram_mib=8192,
    disk_mib=16384,
    timeout_seconds=7200,
    gpu_count=0,
)


def _bind(**changes: object):
    values: dict[str, object] = {
        "image_digest": IMAGE_DIGEST,
        "worktree_id": WORKTREE_ID,
        "task_id": TASK_ID,
        "authority_id": AUTHORITY_ID,
        "worker_id": WORKER_ID,
        "resources": _RESOURCES,
        "artifacts": (
            {
                "path": "out/result.json",
                "content_id": ARTIFACT_CID,
                "byte_count": 32,
            },
        ),
        "created_at_ms": FIXED_MS,
    }
    values.update(changes)
    return bind_container_execution(**values)  # type: ignore[arg-type]


def test_frozen_contract_family_is_strictly_versioned_at_1() -> None:
    assert CONTAINER_CONTRACT_VERSION == 1
    assert CONTRACT_VERSION == 1
    assert SCHEMA_VERSION == 1
    assert dict(CONTAINER_CONTRACT_FAMILY) == {
        "profile": CONTAINER_EXECUTION_PROFILE_INTERFACE,
        "lease": WORKER_LEASE_INTERFACE,
        "artifact_manifest": ARTIFACT_MANIFEST_INTERFACE,
        "checkpoint": CONTAINER_CHECKPOINT_INTERFACE,
        "receipt": CONTAINER_RECEIPT_INTERFACE,
    }
    for name in CONTAINER_CONTRACT_FAMILY.values():
        assert name.endswith("@1")


def test_valid_bind_image_worktree_task_authority_resources_and_policy() -> None:
    bound = _bind()
    profile = bound.profile
    assert profile.image_digest == IMAGE_DIGEST
    assert profile.worktree_id == WORKTREE_ID
    assert profile.task_id == TASK_ID
    assert profile.authority_id == AUTHORITY_ID
    assert profile.resources.cpu_millicores == 4000
    assert profile.resources.ram_mib == 8192
    assert profile.resources.disk_mib == 16384
    assert profile.resources.timeout_seconds == 7200
    assert profile.resources.gpu_count == 0
    assert profile.policy.network_policy is NetworkPolicy.DENY
    assert profile.policy.docker_socket_mounted is False
    assert profile.policy.no_new_privileges is True
    assert bound.lease.profile_id == profile.profile_id
    assert bound.lease.image_digest == IMAGE_DIGEST
    assert bound.lease.worktree_id == WORKTREE_ID
    assert bound.lease.task_id == TASK_ID
    assert bound.lease.authority_id == AUTHORITY_ID
    assert bound.lease.worker_id == WORKER_ID
    assert bound.artifact_manifest.lease_id == bound.lease.lease_id
    assert bound.artifact_manifest.artifacts[0].content_id == ARTIFACT_CID
    assert bound.checkpoint.restart_safe is True
    assert bound.checkpoint.artifact_manifest_id == bound.artifact_manifest.manifest_id
    assert bound.receipt.checkpoint_id == bound.checkpoint.checkpoint_id
    assert bound.receipt.outcome is ContainerOutcome.COMPLETED
    for record in (
        profile,
        bound.lease,
        bound.artifact_manifest,
        bound.checkpoint,
        bound.receipt,
    ):
        payload = record.to_dict()
        assert payload["contract_version"] == 1
        assert payload["schema"].endswith("@1")
        assert payload["interface"].endswith("@1")
        for forbidden in (
            "accepted",
            "acceptor_id",
            "host_acceptance",
            "host_acceptance_authority",
            "completion_eligible",
        ):
            assert forbidden not in payload
        restored = type(record).from_json(record.to_json())
        assert restored == record
        assert restored.content_id == record.content_id
        assert decode_container_contract(json.loads(record.to_json())) == record


def test_reject_missing_image_digest() -> None:
    with pytest.raises(ContainerContractError, match="image_digest is required"):
        _bind(image_digest="")
    with pytest.raises(ContainerContractError, match="image_digest is required"):
        ContainerExecutionProfile(
            image_digest=None,  # type: ignore[arg-type]
            worktree_id=WORKTREE_ID,
            task_id=TASK_ID,
            authority_id=AUTHORITY_ID,
            resources=_RESOURCES,
        )
    with pytest.raises(ContainerContractError, match="sha256"):
        _bind(image_digest="ubuntu:latest")
    payload = _bind().profile.to_dict()
    payload.pop("image_digest")
    with pytest.raises(ContainerContractError, match="image_digest"):
        ContainerExecutionProfile.from_dict(payload)


def test_reject_docker_sock_mount() -> None:
    with pytest.raises(ContainerTrustError, match="docker.sock"):
        _bind(
            policy={
                "network_policy": "deny",
                "docker_socket_mounted": False,
                "no_new_privileges": True,
                "mounts": [
                    {
                        "source": "/var/run/docker.sock",
                        "target": "/var/run/docker.sock",
                        "read_only": True,
                        "kind": "other",
                    }
                ],
            }
        )
    with pytest.raises(ContainerTrustError, match="docker.sock"):
        IsolationPolicy(docker_socket_mounted=True)
    with pytest.raises(ContainerTrustError, match="docker.sock"):
        IsolationPolicy(
            mounts=(
                {
                    "source": "unix:///run/user/1000/docker.sock",
                    "target": "/run/docker.sock",
                    "read_only": True,
                    "kind": "other",
                },
            )
        )


def test_reject_worker_as_acceptor() -> None:
    bound = _bind()
    payload = bound.receipt.to_dict()
    payload["acceptor_id"] = WORKER_ID
    with pytest.raises(ContainerTrustError, match="self-approve"):
        ContainerReceipt.from_dict(payload)
    with pytest.raises(ContainerTrustError, match="host acceptance"):
        _bind(acceptor_id=WORKER_ID)
    with pytest.raises(ContainerTrustError, match="self-approve"):
        _bind(accepted=True)
    lease_payload = bound.lease.to_dict()
    lease_payload["host_acceptance_authority"] = WORKER_ID
    with pytest.raises(ContainerTrustError, match="host acceptance"):
        WorkerLease.from_dict(lease_payload)
    with pytest.raises(ContainerTrustError, match="self-approve"):
        _bind(worker_id=AUTHORITY_ID)


def test_resource_bounds() -> None:
    valid = ResourceBounds(
        cpu_millicores=1,
        ram_mib=1,
        disk_mib=1,
        timeout_seconds=1,
        gpu_count=0,
    )
    assert valid.admits(ResourceUse())
    with pytest.raises(ContainerContractError, match="cpu_millicores"):
        ResourceBounds(
            cpu_millicores=0,
            ram_mib=1,
            disk_mib=1,
            timeout_seconds=1,
        )
    with pytest.raises(ContainerContractError, match="ram_mib"):
        ResourceBounds(
            cpu_millicores=1,
            ram_mib=-1,
            disk_mib=1,
            timeout_seconds=1,
        )
    with pytest.raises(ContainerBoundsError, match="cpu_millicores"):
        ResourceBounds(
            cpu_millicores=ABSOLUTE_MAX_CPU_MILLICORES + 1,
            ram_mib=1,
            disk_mib=1,
            timeout_seconds=1,
        )
    with pytest.raises(ContainerBoundsError, match="ram_mib"):
        ResourceBounds(
            cpu_millicores=1,
            ram_mib=ABSOLUTE_MAX_RAM_MIB + 1,
            disk_mib=1,
            timeout_seconds=1,
        )
    with pytest.raises(ContainerBoundsError, match="disk_mib"):
        ResourceBounds(
            cpu_millicores=1,
            ram_mib=1,
            disk_mib=ABSOLUTE_MAX_DISK_MIB + 1,
            timeout_seconds=1,
        )
    with pytest.raises(ContainerBoundsError, match="timeout_seconds"):
        ResourceBounds(
            cpu_millicores=1,
            ram_mib=1,
            disk_mib=1,
            timeout_seconds=ABSOLUTE_MAX_TIMEOUT_SECONDS + 1,
        )
    with pytest.raises(ContainerBoundsError, match="gpu_count"):
        ResourceBounds(
            cpu_millicores=1,
            ram_mib=1,
            disk_mib=1,
            timeout_seconds=1,
            gpu_count=ABSOLUTE_MAX_GPU_COUNT + 1,
        )
    with pytest.raises(ContainerBoundsError, match="resource_use exceeds"):
        _bind(resource_use={"cpu_millicores": 4001, "ram_mib": 1})
    reserved = _bind(
        resources={
            "cpu_millicores": 500,
            "ram_mib": 256,
            "disk_mib": 512,
            "timeout_seconds": 30,
            "gpu_count": 0,
        },
        resource_use={
            "cpu_millicores": 500,
            "ram_mib": 256,
            "disk_mib": 512,
            "elapsed_seconds": 30,
            "gpu_count": 0,
        },
    )
    assert reserved.profile.resources.admits(reserved.receipt.resource_use)


def test_records_are_frozen_and_round_trip_identities() -> None:
    bound = _bind()
    with pytest.raises(FrozenInstanceError):
        bound.profile.task_id = "mutated"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        bound.lease.active = False  # type: ignore[misc]
    first = bound.profile.to_json()
    second = ContainerExecutionProfile.from_dict(
        {
            "authority_id": AUTHORITY_ID,
            "task_id": TASK_ID,
            "worktree_id": WORKTREE_ID,
            "image_digest": IMAGE_DIGEST,
            "resources": {
                "gpu_count": 0,
                "timeout_seconds": 7200,
                "disk_mib": 16384,
                "ram_mib": 8192,
                "cpu_millicores": 4000,
            },
            "created_at_ms": FIXED_MS,
        }
    ).to_json()
    assert first == second
    assert bound.profile.content_id == ContainerExecutionProfile.from_json(first).content_id
    assert canonical_container_json_bytes(bound.profile) == bound.profile.canonical_bytes()


def test_unknown_schema_version_and_forged_identity_are_rejected() -> None:
    payload = _bind().profile.to_dict()
    payload["schema"] = "ipfs_accelerate_py/agent-supervisor/container-execution-profile@2"
    with pytest.raises(ContainerVersionError):
        ContainerExecutionProfile.from_dict(payload)
    payload = _bind().lease.to_dict()
    payload["interface"] = "WorkerLease@2"
    with pytest.raises(ContainerVersionError):
        WorkerLease.from_dict(payload)
    payload = _bind().receipt.to_dict()
    payload["contract_version"] = 2
    with pytest.raises(ContainerVersionError):
        ContainerReceipt.from_dict(payload)
    with pytest.raises(ContainerVersionError):
        decode_container_contract({"schema": "UnknownRecord@1", "contract_version": 1})
    payload = _bind().checkpoint.to_dict()
    payload["content_id"] = IMAGE_DIGEST
    with pytest.raises(ContainerIdentityError):
        ContainerCheckpoint.from_dict(payload)
    payload = _bind().artifact_manifest.to_dict()
    payload["extra"] = "nope"
    with pytest.raises(ContainerContractError, match="unsupported fields"):
        ArtifactManifest.from_dict(payload)


def test_network_policy_and_no_new_privileges_are_required() -> None:
    with pytest.raises(ContainerContractError, match="network_policy"):
        IsolationPolicy(network_policy="allow")  # type: ignore[arg-type]
    with pytest.raises(ContainerTrustError, match="no-new-privileges"):
        IsolationPolicy(no_new_privileges=False)
    with pytest.raises(ContainerTrustError, match="privileged"):
        IsolationPolicy(privileged=True)
    payload = IsolationPolicy().to_dict()
    payload["network_policy"] = "bridge"
    with pytest.raises(ContainerContractError, match="network_policy"):
        IsolationPolicy.from_dict(payload)
