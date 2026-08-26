"""EAAEF-143: container create/start/checkpoint/stop/cleanup contracts.

Live Docker is not invoked.  ``EngineAdmission`` plus a cheap rootless probe
would be required before any engine contact; this overlay prefers contract
tests and records ``live_engine_invoked`` as false.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.containers.checkpoint import (
    CheckpointError,
    ContainerCheckpoint as FenceCheckpoint,
    recover as recover_fence,
)
from ipfs_accelerate_py.agent_supervisor.containers.contracts import (
    IsolationPolicy,
    NetworkPolicy,
    ResourceBounds,
    bind_container_execution,
)
from ipfs_accelerate_py.agent_supervisor.containers.oci_runner import (
    EngineAdmission,
    build_oci_run_spec,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


RECEIPT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "architecture"
    / "external_agent_autonomous_execution_fabric"
    / "receipts"
    / "container_lifecycle.json"
)

ROOTLESS_SOCK = "unix:///run/user/1000/docker.sock"
LIFECYCLE = ("create", "start", "checkpoint", "stop", "cleanup")
ARTIFACT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-offline-qualification-artifact@1"
)
PRODUCER_ARGV = (
    "python3",
    "-m",
    "pytest",
    "-q",
    "test/integration/test_external_container_lifecycle.py",
)
RECEIPT_FIELDS = {
    "artifact_cid",
    "candidate_engine_mode",
    "candidate_rootless_socket",
    "docker_socket_mounted",
    "evidence_mode",
    "host_engine_probe_invoked",
    "lifecycle_stages",
    "live_eight_container_qualification",
    "live_engine_invoked",
    "live_runtime_invoked",
    "network_policy",
    "privileged",
    "producer_argv",
    "producer_source_cid",
    "production_qualification_claimed",
    "qualification_scope",
    "qualification_status",
    "schema",
    "task_completion_claimed",
    "task_id",
}


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _probe_live_engine(admission: EngineAdmission) -> bool:
    """Fail closed unless admission and a cheap rootless probe both succeed.

    Contract tests are preferred.  This overlay never invokes the engine.
    """

    del admission
    return False


def _producer_source_cid() -> str:
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _validate_receipt(payload: dict[str, object]) -> None:
    assert set(payload) == RECEIPT_FIELDS
    assert payload["schema"] == ARTIFACT_SCHEMA
    assert payload["task_id"] == "EAAEF-143"
    assert payload["evidence_mode"] == "contract_fail_closed"
    assert payload["qualification_scope"] == "offline_container_lifecycle_contract_only"
    assert payload["qualification_status"] == "not_live_qualified"
    assert payload["task_completion_claimed"] is False
    assert payload["production_qualification_claimed"] is False
    assert payload["live_runtime_invoked"] is False
    assert payload["live_engine_invoked"] is False
    assert payload["live_eight_container_qualification"] is False
    assert payload["host_engine_probe_invoked"] is False
    assert payload["producer_argv"] == list(PRODUCER_ARGV)
    assert payload["producer_source_cid"] == _producer_source_cid()
    unsealed = dict(payload)
    artifact_cid = unsealed.pop("artifact_cid")
    assert artifact_cid == content_identity(unsealed)


def _write_receipt(payload: dict[str, object]) -> dict[str, object]:
    payload = {
        **payload,
        "producer_argv": list(PRODUCER_ARGV),
        "producer_source_cid": _producer_source_cid(),
    }
    payload["artifact_cid"] = content_identity(payload)
    _validate_receipt(payload)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def test_create_start_checkpoint_stop_cleanup_contracts_fail_closed() -> None:
    admission = EngineAdmission()
    live_admitted = bool(admission.rootless_supported and admission.rootless_verified)
    live_engine_invoked = bool(live_admitted and _probe_live_engine(admission))
    assert live_engine_invoked is False

    bound = bind_container_execution(
        image_digest=_digest("lifecycle-image"),
        worktree_id="worktree:lifecycle",
        task_id="task:EAAEF-143",
        authority_id="authority:lifecycle-supervisor",
        worker_id="worker:lifecycle-1",
        resources=ResourceBounds(
            cpu_millicores=1000,
            ram_mib=512,
            disk_mib=1024,
            timeout_seconds=60,
        ),
        policy=IsolationPolicy(),
        fencing_token=1,
        artifacts=(
            {
                "path": "out/lifecycle.json",
                "content_id": _digest("lifecycle-artifact"),
                "byte_count": 16,
            },
        ),
    )
    profile = bound.profile
    assert profile.policy.network_policy is NetworkPolicy.DENY
    assert profile.policy.docker_socket_mounted is False
    assert profile.policy.privileged is False
    assert profile.policy.no_new_privileges is True
    assert profile.policy.read_only_base is True

    spec = build_oci_run_spec(
        profile,
        command=("python3", "-c", "pass"),
        engine_admission=admission,
    )
    assert spec.live_engine_invoked is False
    assert spec.privileged is False
    assert spec.docker_socket_mounted is False
    assert spec.network == "none"
    assert spec.read_only is True
    assert spec.no_new_privileges is True
    assert "--privileged" not in spec.argv
    assert "-v" not in spec.argv
    assert "--volume" not in spec.argv
    # Rootless engine --host may name the user docker.sock; that is not a worker mount.
    assert not any(
        token.startswith("-v") or token.startswith("--volume") or token.startswith("--mount")
        for token in spec.argv
    )

    assert bound.checkpoint.restart_safe is True
    assert bound.lease.active is True
    assert bound.receipt.outcome.value == "completed"

    fence = FenceCheckpoint(
        attempt_id="attempt:lifecycle-1",
        worktree_id=profile.worktree_id,
        fence_token=1,
        lane_id="lane:lifecycle",
        owner_alive=False,
        semantic_delta_id=_digest("delta"),
    )
    restarted = recover_fence(fence, next_fence=2)
    assert restarted.fence_token == 2
    assert restarted.owner_alive is True
    with pytest.raises(CheckpointError, match="live owner"):
        recover_fence(restarted, next_fence=3)
    with pytest.raises(CheckpointError, match="later fence"):
        recover_fence(fence, next_fence=1)

    stages = {
        "create": {"profile_id": profile.profile_id, "image_digest": profile.image_digest},
        "start": {"argv": list(spec.argv), "live_engine_invoked": spec.live_engine_invoked},
        "checkpoint": {
            "checkpoint_id": bound.checkpoint.checkpoint_id,
            "restart_safe": bound.checkpoint.restart_safe,
            "recovered_fence": restarted.fence_token,
        },
        "stop": {"lease_active": bound.lease.active, "outcome": bound.receipt.outcome.value},
        "cleanup": {
            "docker_socket_mounted": False,
            "network_policy": profile.policy.network_policy.value,
            "host_mutation": False,
        },
    }
    assert tuple(stages) == LIFECYCLE

    payload = _write_receipt(
        {
            "schema": ARTIFACT_SCHEMA,
            "task_id": "EAAEF-143",
            "evidence_mode": "contract_fail_closed",
            "qualification_scope": "offline_container_lifecycle_contract_only",
            "qualification_status": "not_live_qualified",
            "task_completion_claimed": False,
            "production_qualification_claimed": False,
            "live_runtime_invoked": False,
            "live_eight_container_qualification": False,
            "live_engine_invoked": False,
            "host_engine_probe_invoked": False,
            "candidate_rootless_socket": ROOTLESS_SOCK,
            "candidate_engine_mode": spec.engine_mode.value,
            "lifecycle_stages": list(LIFECYCLE),
            "docker_socket_mounted": False,
            "network_policy": "deny",
            "privileged": False,
        }
    )
    _validate_receipt(payload)
    saved = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert saved["evidence_mode"] == "contract_fail_closed"
    assert saved["live_engine_invoked"] is False
    assert saved["live_runtime_invoked"] is False
    assert saved["live_eight_container_qualification"] is False
    assert payload["lifecycle_stages"] == list(LIFECYCLE)
