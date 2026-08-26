"""EAAEF-055: container isolation contracts; never invoke a live engine."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.containers import checkpoint as checkpoint_mod
from ipfs_accelerate_py.agent_supervisor.containers.checkpoint import (
    CHECKPOINT_SCHEMA,
    CheckpointError,
    ContainerCheckpoint,
    recover,
)
from ipfs_accelerate_py.agent_supervisor.containers.contracts import (
    ContainerExecutionProfile,
    ContainerTrustError,
    IsolationPolicy,
    ResourceBounds,
)
from ipfs_accelerate_py.agent_supervisor.containers.oci_runner import (
    EngineAdmission,
    OciRunner,
    OciRunnerTrustError,
    build_oci_run_spec,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)

IMAGE_DIGEST = "sha256:" + ("a" * 64)
_RESOURCES = ResourceBounds(
    cpu_millicores=4000,
    ram_mib=8192,
    disk_mib=16384,
    timeout_seconds=7200,
    gpu_count=0,
)
RECEIPT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "architecture"
    / "external_agent_autonomous_execution_fabric"
    / "receipts"
    / "container.json"
)
ARTIFACT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-offline-qualification-artifact@1"
)
PRODUCER_ARGV = (
    "python3",
    "-m",
    "pytest",
    "-q",
    "test/security/test_external_agent_container_isolation.py",
)
RECEIPT_FIELDS = {
    "artifact_cid",
    "checkpoint_contract_validated",
    "cleanup_observed_on_live_runtime",
    "default_deny_contract",
    "evidence_mode",
    "host_engine_probe_invoked",
    "live_engine_invoked",
    "live_runtime_invoked",
    "producer_argv",
    "producer_source_cid",
    "production_qualification_claimed",
    "qualification_scope",
    "qualification_status",
    "resource_bounds",
    "schema",
    "task_completion_claimed",
    "task_id",
}


def _producer_source_cid() -> str:
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _validate_receipt(payload: dict[str, object]) -> None:
    assert set(payload) == RECEIPT_FIELDS
    assert payload["schema"] == ARTIFACT_SCHEMA
    assert payload["task_id"] == "EAAEF-055"
    assert payload["evidence_mode"] == "contract_fail_closed"
    assert payload["qualification_scope"] == "offline_container_isolation_contract_only"
    assert payload["qualification_status"] == "not_live_qualified"
    assert payload["task_completion_claimed"] is False
    assert payload["production_qualification_claimed"] is False
    assert payload["live_runtime_invoked"] is False
    assert payload["live_engine_invoked"] is False
    assert payload["host_engine_probe_invoked"] is False
    assert payload["cleanup_observed_on_live_runtime"] is False
    assert payload["producer_argv"] == list(PRODUCER_ARGV)
    assert payload["producer_source_cid"] == _producer_source_cid()
    unsealed = dict(payload)
    artifact_cid = unsealed.pop("artifact_cid")
    assert artifact_cid == content_identity(unsealed)


def _write_receipt(payload: dict[str, object]) -> dict[str, object]:
    sealed = {
        **payload,
        "producer_argv": list(PRODUCER_ARGV),
        "producer_source_cid": _producer_source_cid(),
    }
    sealed["artifact_cid"] = content_identity(sealed)
    _validate_receipt(sealed)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sealed


def _profile(**changes: object) -> ContainerExecutionProfile:
    values: dict[str, object] = {
        "image_digest": IMAGE_DIGEST,
        "worktree_id": "worktree:eaaef-055",
        "task_id": "task:EAAEF-055",
        "authority_id": "authority:supervisor",
        "resources": _RESOURCES,
        "policy": IsolationPolicy(),
    }
    values.update(changes)
    return ContainerExecutionProfile(**values)  # type: ignore[arg-type]


def test_default_deny_isolation_without_live_engine() -> None:
    spec = build_oci_run_spec(_profile())
    assert spec.live_engine_invoked is False
    assert spec.docker_socket_mounted is False
    assert spec.privileged is False
    assert spec.network == "none"
    assert spec.read_only is True
    assert spec.cap_drop == ("ALL",)
    assert spec.no_new_privileges is True
    assert "--network=none" in spec.argv
    assert "--read-only" in spec.argv
    assert "--cap-drop=ALL" in spec.argv
    assert "--security-opt=no-new-privileges" in spec.argv
    assert "--privileged" not in spec.argv
    assert not any(
        "docker.sock" in part and not part.startswith("--host=") for part in spec.argv
    )
    assert spec.cpu_millicores == 4000
    assert spec.ram_mib == 8192
    assert spec.disk_mib == 16384
    assert spec.timeout_seconds == 7200
    runner = OciRunner()
    planned = runner.build_spec(_profile())
    assert planned.live_engine_invoked is False
    assert planned.docker_socket_mounted is False
    assert isinstance(EngineAdmission().rootful_fallback_admitted, bool)


def test_isolation_policy_rejects_docker_socket_and_privileged() -> None:
    with pytest.raises(ContainerTrustError, match="docker.sock"):
        IsolationPolicy(docker_socket_mounted=True)
    with pytest.raises(ContainerTrustError, match="privileged"):
        IsolationPolicy(privileged=True)
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
    with pytest.raises(OciRunnerTrustError, match="privileged"):
        build_oci_run_spec(profile, privileged=True)
    policy = IsolationPolicy()
    assert policy.docker_socket_mounted is False
    assert policy.privileged is False
    assert policy.no_new_privileges is True
    assert policy.read_only_base is True


def test_checkpoint_contracts_exist_without_starting_containers() -> None:
    assert checkpoint_mod.CHECKPOINT_SCHEMA == CHECKPOINT_SCHEMA
    assert CHECKPOINT_SCHEMA.endswith("@1")
    checkpoint = ContainerCheckpoint(
        attempt_id="attempt:1",
        worktree_id="worktree:eaaef-055",
        fence_token=1,
        lane_id="lane:0",
        owner_alive=False,
    )
    assert checkpoint.to_dict()["schema"] == CHECKPOINT_SCHEMA
    recovered = recover(checkpoint, next_fence=2)
    assert recovered.fence_token == 2
    assert recovered.owner_alive is True
    live = ContainerCheckpoint(
        attempt_id="attempt:1",
        worktree_id="worktree:eaaef-055",
        fence_token=2,
        lane_id="lane:0",
        owner_alive=True,
    )
    with pytest.raises(CheckpointError, match="live owner"):
        recover(live, next_fence=3)
    with pytest.raises(CheckpointError, match="later fence"):
        recover(checkpoint, next_fence=1)
    assert not hasattr(recover, "start_container")
    assert build_oci_run_spec(_profile()).live_engine_invoked is False


def test_write_offline_container_isolation_receipt() -> None:
    spec = build_oci_run_spec(_profile())
    assert spec.live_engine_invoked is False
    assert spec.network == "none"
    assert spec.read_only is True
    assert spec.cap_drop == ("ALL",)
    assert spec.no_new_privileges is True
    checkpoint = ContainerCheckpoint(
        attempt_id="attempt:offline-receipt",
        worktree_id="worktree:eaaef-055",
        fence_token=1,
        lane_id="lane:0",
        owner_alive=False,
    )
    assert recover(checkpoint, next_fence=2).fence_token == 2

    receipt = _write_receipt(
        {
            "schema": ARTIFACT_SCHEMA,
            "task_id": "EAAEF-055",
            "evidence_mode": "contract_fail_closed",
            "qualification_scope": "offline_container_isolation_contract_only",
            "qualification_status": "not_live_qualified",
            "task_completion_claimed": False,
            "production_qualification_claimed": False,
            "live_runtime_invoked": False,
            "live_engine_invoked": False,
            "host_engine_probe_invoked": False,
            "cleanup_observed_on_live_runtime": False,
            "checkpoint_contract_validated": True,
            "default_deny_contract": {
                "cap_drop": ["ALL"],
                "docker_socket_mounted": False,
                "network": "none",
                "no_new_privileges": True,
                "privileged": False,
                "read_only": True,
            },
            "resource_bounds": {
                "cpu_millicores": spec.cpu_millicores,
                "disk_mib": spec.disk_mib,
                "ram_mib": spec.ram_mib,
                "timeout_seconds": spec.timeout_seconds,
            },
        }
    )
    _validate_receipt(receipt)
