from __future__ import annotations

import base64
import json
import os
from copy import deepcopy
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_implementation_route import (
    AgentImplementationControlPlanePin,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.validation import (
    external_agent_configured_board_capsule as capsule,
)
from ipfs_accelerate_py.agent_supervisor.validation.eaaef_authority_registry import (
    EAAEFAuthorityRegistry,
)

NOW_MS = 1_800_000_000_000
REPO_ROOT = Path(__file__).resolve().parents[2]


def _sibling_authority_root(tmp_path: Path) -> Path:
    return tmp_path.parent / f"{tmp_path.name}-authority"


def _sha(token: str) -> str:
    return "sha256:" + token * 64


def _pin() -> AgentImplementationControlPlanePin:
    return AgentImplementationControlPlanePin(
        schema="ipfs_accelerate_py.agent_supervisor.accepted-control-plane@2",
        runner_path="/sealed/ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py",
        runner_sha256=_sha("1"),
        capsule_root="/sealed",
        capsule_id=_sha("2"),
        source_head="3" * 40,
        source_tree="4" * 40,
        archive_sha256=_sha("5"),
    )


def _frontier(task: str = "EAAEF-001", token: str = "a") -> dict[str, object]:
    return {
        "task_id": task,
        "task_cid": _sha(token),
        "dependencies": [_sha("9")],
        "read_scope": [f"read/{task}"],
        "write_scope": [f"write/{task}"],
        "effect_scope": [f"effect/{task}"],
        "container_slots": 1,
        "provider_slots": 1,
    }


def _network_policy() -> dict[str, object]:
    return {
        "schema": capsule.EAAEF_WORKER_NETWORK_DISPATCH_POLICY_SCHEMA,
        "authorization_schema": (
            "ipfs_accelerate_py/eaaef-worker-network-authorization@1"
        ),
        "verifier_interface": "verify_worker_network_authorization@1",
        "artifact_path_authority": "verified_invocation_profile_dir",
        "artifact_relative_path_template": (
            "network-authorizations/<sha256(invocation_id)>/<provider>.json"
        ),
        "dynamic_caller_path_allowed": False,
        "expected_artifact_cid_required": True,
        "expected_worker_principal_did_required": True,
        "expected_provider_principal_did_required": True,
        "control_plane_capsule_binding_required": True,
        "task_plan_source_worktree_effect_binding_required": True,
        "container_and_lease_binding_required": True,
        "create_start_restart_reverification_required": True,
        "supported_providers": ["codex", "grok"],
        "child_propagation_status": "unavailable_fail_closed",
    }


def _command_fabric_profile() -> tuple[dict[str, object], dict[str, object]]:
    payload = json.loads(
        (
            REPO_ROOT
            / "config/external_agent_autonomous_execution_fabric_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    return (
        dict(payload["operational_command_fabric"]),
        dict(payload["database_program"]),
    )


def test_operational_command_fabric_v2_binds_exact_board_and_shard() -> None:
    profile, operational = _command_fabric_profile()

    result = capsule.validate_eaaef_operational_command_fabric_profile(
        profile,
        operational_program=operational,
        expected_board_namespace="external-agent-autonomous-execution-fabric-v1",
        expected_shard_id="control-shard-0",
    )

    assert result["schema"] == capsule.EAAEF_SIGNED_COMMAND_FABRIC_PROFILE_SCHEMA
    assert result["board_namespace"] == "external-agent-autonomous-execution-fabric-v1"
    assert result["shard_id"] == "control-shard-0"


@pytest.mark.parametrize("field", ["board_namespace", "shard_id"])
def test_operational_command_fabric_v2_rejects_missing_identity_field(
    field: str,
) -> None:
    profile, operational = _command_fabric_profile()
    profile.pop(field)

    with pytest.raises(
        capsule.ExternalAgentConfiguredBoardCapsuleError,
        match="shape is not canonical",
    ):
        capsule.validate_eaaef_operational_command_fabric_profile(
            profile,
            operational_program=operational,
            expected_board_namespace="external-agent-autonomous-execution-fabric-v1",
            expected_shard_id="control-shard-0",
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("board_namespace", "other-board"), ("shard_id", "control-shard-1")],
)
def test_operational_command_fabric_v2_rejects_profile_identity_mismatch(
    field: str,
    replacement: str,
) -> None:
    profile, operational = _command_fabric_profile()
    profile[field] = replacement

    with pytest.raises(
        capsule.ExternalAgentConfiguredBoardCapsuleError,
        match="policy is not fail closed",
    ):
        capsule.validate_eaaef_operational_command_fabric_profile(
            profile,
            operational_program=operational,
            expected_board_namespace="external-agent-autonomous-execution-fabric-v1",
            expected_shard_id="control-shard-0",
        )


@pytest.mark.parametrize(
    ("expected_board_namespace", "expected_shard_id"),
    [
        ("other-board", "control-shard-0"),
        ("external-agent-autonomous-execution-fabric-v1", "control-shard-1"),
    ],
)
def test_operational_command_fabric_v2_rejects_outer_identity_mismatch(
    expected_board_namespace: str,
    expected_shard_id: str,
) -> None:
    profile, operational = _command_fabric_profile()

    with pytest.raises(
        capsule.ExternalAgentConfiguredBoardCapsuleError,
        match="expected identity is not admitted",
    ):
        capsule.validate_eaaef_operational_command_fabric_profile(
            profile,
            operational_program=operational,
            expected_board_namespace=expected_board_namespace,
            expected_shard_id=expected_shard_id,
        )


def _statement() -> dict[str, object]:
    pin = _pin()
    frontier = [_frontier()]
    satisfied = [_sha("9")]
    value: dict[str, object] = {
        "schema": capsule.EAAEF_CONFIGURED_BOARD_CAPSULE_STATEMENT_SCHEMA,
        "board_namespace": "external-agent-autonomous-execution-fabric-v1",
        "board_cid": _sha("6"),
        "source_head": pin.source_head,
        "source_tree": pin.source_tree,
        "source_generation_cid": _sha("7"),
        "configuration_root": _sha("8"),
        "materialization_receipt_cid": _sha("8"),
        "materialization_store_generation": "eaaef-test-run-v5",
        "materialization_database_program_binding_cid": _sha("4"),
        "materialization_bootstrap_profile_cid": _sha("5"),
        "materialization_operational_profile_cid": _sha("7"),
        "population_cid": _sha("a"),
        "plan_root_cid": _sha("b"),
        "control_projection_root": _sha("c"),
        "coordination_projection_root": _sha("d"),
        "execution_projection_root": _sha("e"),
        "bootstrap_admission_receipt_cid": _sha("f"),
        "admission_operator_did": "did:key:zOperator",
        "admission_security_reviewer_did": "did:key:zSecurity",
        "provider_container_qualification_cid": _sha("0"),
        "qualified_worker_image_digest": _sha("f"),
        "qualified_worker_container_profile_cid": _sha("e"),
        "provider_maximum_parallel_workers": 5,
        "provider_maximum_parallel_containers": 5,
        "provider_worker_principal_did": "did:key:zWorker",
        "provider_principal_did": "did:key:zProviderService",
        "provider_task_dispatch_admitted": True,
        "provider_workload_class": "agent_worker",
        "provider_qualification_signer_did": "did:key:zProvider",
        "image_qualification_reviewer_did": "did:key:zImage",
        "quack_owner_qualification_cid": _sha("1"),
        "quack_owner_verification_cid": _sha("2"),
        "quack_qualification_reviewer_did": "did:key:zQuack",
        "quack_owner_principal_did": "did:key:zOwner",
        "quack_shard_id": "eaaef-control",
        "quack_epoch": 3,
        "quack_fence": 4,
        "accepted_control_plane_pin": pin.as_dict(),
        "accepted_control_plane_pin_cid": capsule._cid(pin.as_dict()),
        "worker_network_authorization_policy": _network_policy(),
        "control_plane_promotion": {
            "mode": "owner_only_r1",
            "promotion_receipt_cid": "",
            "promotion_verification_cid": "",
            "base_owner_qualification_cid": _sha("1"),
            "bootstrap_admission_cid": "",
            "dispatcher_interface": "",
            "command_fabric_interface": "",
            "operation_vocabulary_cid": "",
            "plan_r2_operational_capability_cid": "",
            "command_fabric_qualification_cid": "",
            "authorization_policy_cid": "",
            "store_id": "eaaef-control-run-v5",
            "owner_generation": 1,
            "generic_daemon_gateway_admitted": False,
        },
        "active_plan": {
            "revision_alias": "EAAEF-PLAN-R1",
            "plan_root_cid": _sha("b"),
            "population_cid": _sha("a"),
            "semantic_state_root": _sha("3"),
            "revision": 1,
            "event_cursor": "bootstrap-event-cursor-1",
        },
        "plan_transition": {
            "mode": "bootstrap_r1",
            "authorization_cid": "",
            "transition_receipt_cid": "",
            "state_observation_cid": _sha("8"),
            "before_plan_root_cid": _sha("b"),
            "after_plan_root_cid": _sha("b"),
            "before_plan_revision": 1,
            "after_plan_revision": 1,
            "before_event_cursor": "bootstrap-event-cursor-1",
            "after_event_cursor": "bootstrap-event-cursor-1",
            "before_semantic_root_cid": _sha("3"),
            "after_semantic_root_cid": _sha("3"),
        },
        "satisfied_task_cids": satisfied,
        "frontier": frontier,
        "frontier_cid": capsule._cid(
            {
                "schema": "EAAEFConflictFreeFrontier@1",
                "tasks": frontier,
                "satisfied_task_cids": satisfied,
            }
        ),
        "authority": {
            "launch_mode": "configured_board_multi_supervisor",
            "maximum_lanes": 5,
            "actual_lane_count": 1,
            "one_fenced_quack_owner": True,
            "direct_duckdb_file_open": False,
            "child_birth_requires_sealed_descriptor": True,
            "restart_requires_reverification": True,
        },
        "issued_at_ms": NOW_MS - 1000,
        "expires_at_ms": NOW_MS + 100_000,
    }
    value["statement_cid"] = capsule._cid(value)
    return value


def _signed_capsule(monkeypatch: pytest.MonkeyPatch):
    pin = _pin()
    monkeypatch.setattr(
        capsule,
        "build_agent_implementation_control_plane_pin",
        lambda **_kwargs: pin,
    )
    statement = _statement()
    key = Ed25519PrivateKey.generate()
    reviewer = ed25519_did_key(key.public_key())
    approval = capsule.prepare_external_agent_capsule_approval(
        statement,
        identity_did=reviewer,
        issued_at_ms=NOW_MS - 500,
        expires_at_ms=NOW_MS + 50_000,
    )
    approval["signature"] = base64.b64encode(
        key.sign(capsule._canonical_bytes(approval))
    ).decode("ascii")
    value = capsule.assemble_external_agent_configured_board_capsule(
        statement,
        reviewer_approval=approval,
        trusted_reviewer_dids=[reviewer],
        now_ms=NOW_MS,
    )
    return value, reviewer


def test_capsule_binds_sealed_control_plane_and_exact_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, reviewer = _signed_capsule(monkeypatch)
    result = capsule.verify_external_agent_configured_board_capsule(
        value,
        trusted_reviewer_dids=[reviewer],
        now_ms=NOW_MS,
    )
    assert result["valid"] is True
    assert result["frontier_task_cids"] == [_sha("a")]
    assert result["actual_lane_count"] == 1
    assert result["maximum_lanes"] == 5
    assert result["authority_mutated"] is False
    assert result["process_started"] is False


def test_overlapping_frontier_and_unknown_scopes_fail_closed() -> None:
    left = _frontier("EAAEF-001", "a")
    right = _frontier("EAAEF-002", "b")
    right["write_scope"] = list(left["read_scope"])
    with pytest.raises(capsule.ExternalAgentConfiguredBoardCapsuleError, match="overlapping"):
        capsule._validate_frontier(
            [left, right], satisfied_task_cids=frozenset({_sha("9")})
        )

    left["write_scope"] = []
    with pytest.raises(capsule.ExternalAgentConfiguredBoardCapsuleError, match="scope"):
        capsule._validate_frontier(
            [left], satisfied_task_cids=frozenset({_sha("9")})
        )


def test_capsule_tamper_is_rejected_before_pin_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, reviewer = _signed_capsule(monkeypatch)
    tampered = deepcopy(value)
    tampered["quack_fence"] = 99
    with pytest.raises(capsule.ExternalAgentConfiguredBoardCapsuleError, match="self-address"):
        capsule.verify_external_agent_configured_board_capsule(
            tampered,
            trusted_reviewer_dids=[reviewer],
            now_ms=NOW_MS,
        )


def test_direct_capsule_verification_rejects_provider_principal_reviewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verification must enforce the same independence rule as assembly."""

    pin = _pin()
    monkeypatch.setattr(
        capsule,
        "build_agent_implementation_control_plane_pin",
        lambda **_kwargs: pin,
    )
    statement = _statement()
    key = Ed25519PrivateKey.generate()
    provider_principal = ed25519_did_key(key.public_key())
    statement["provider_principal_did"] = provider_principal
    statement["statement_cid"] = capsule._cid(
        {key: value for key, value in statement.items() if key != "statement_cid"}
    )
    approval = capsule.prepare_external_agent_capsule_approval(
        statement,
        identity_did=provider_principal,
        issued_at_ms=NOW_MS - 500,
        expires_at_ms=NOW_MS + 50_000,
    )
    approval["signature"] = base64.b64encode(
        key.sign(capsule._canonical_bytes(approval))
    ).decode("ascii")
    # Assemble the closed capsule bytes directly to exercise the public verifier
    # rather than relying on the stricter assembly helper.
    value = {
        **statement,
        "schema": capsule.EAAEF_CONFIGURED_BOARD_CAPSULE_SCHEMA,
        "reviewer_approval": approval,
    }
    value["capsule_cid"] = capsule._cid(value)

    with pytest.raises(
        capsule.ExternalAgentConfiguredBoardCapsuleError,
        match="independent of execution principals",
    ):
        capsule.verify_external_agent_configured_board_capsule(
            value,
            trusted_reviewer_dids=[provider_principal],
            now_ms=NOW_MS,
        )


def test_capsule_publication_is_create_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, reviewer = _signed_capsule(monkeypatch)
    authority_root = _sibling_authority_root(tmp_path)
    logical_path = capsule.external_agent_configured_board_launch_capsule_relative_path(
        str(value["source_head"]),
        str(value["active_plan"]["plan_root_cid"]),
    )
    registry = EAAEFAuthorityRegistry(
        repo_root=tmp_path,
        authority_root=authority_root,
    )
    target = registry.physical_path(logical_path)
    capsule.publish_external_agent_configured_board_capsule(
        tmp_path,
        value,
        trusted_reviewer_dids=[reviewer],
        now_ms=NOW_MS,
        authority_root=authority_root,
    )
    original = target.read_bytes()
    capsule.publish_external_agent_configured_board_capsule(
        tmp_path,
        value,
        trusted_reviewer_dids=[reviewer],
        now_ms=NOW_MS,
        authority_root=authority_root,
    )
    assert target.read_bytes() == original

    conflicting, conflicting_reviewer = _signed_capsule(monkeypatch)
    with pytest.raises(
        capsule.ExternalAgentConfiguredBoardCapsuleError,
        match="overwrite",
    ):
        capsule.publish_external_agent_configured_board_capsule(
            tmp_path,
            conflicting,
            trusted_reviewer_dids=[conflicting_reviewer],
            now_ms=NOW_MS,
            authority_root=authority_root,
        )
    assert target.read_bytes() == original


def test_live_seal_uses_paths_not_cyclic_cids_and_rejects_post_parent_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, reviewer = _signed_capsule(monkeypatch)
    authority_root = _sibling_authority_root(tmp_path)
    registry = EAAEFAuthorityRegistry(
        repo_root=tmp_path,
        authority_root=authority_root,
    )
    admission_path = capsule.external_agent_bootstrap_admission_relative_path(
        str(value["source_head"])
    )
    capsule_logical_path = capsule.external_agent_configured_board_launch_capsule_relative_path(
        str(value["source_head"]),
        str(value["active_plan"]["plan_root_cid"]),
    )
    registry.publish_json(admission_path, {})
    registry.publish_json(capsule_logical_path, value)
    capsule_path = registry.physical_path(capsule_logical_path)
    statement = _statement()
    monkeypatch.setattr(
        capsule,
        "verify_external_agent_bootstrap_admission",
        lambda *_args, **_kwargs: {
            "source_head": statement["source_head"],
            "source_tree": statement["source_tree"],
            "receipt_cid": statement["bootstrap_admission_receipt_cid"],
            "board_cid": statement["board_cid"],
            "materialization_receipt_cid": statement[
                "materialization_receipt_cid"
            ],
            "materialization_store_generation": statement[
                "materialization_store_generation"
            ],
            "materialization_database_program_binding_cid": statement[
                "materialization_database_program_binding_cid"
            ],
            "materialization_bootstrap_profile_cid": statement[
                "materialization_bootstrap_profile_cid"
            ],
            "materialization_operational_profile_cid": statement[
                "materialization_operational_profile_cid"
            ],
            "population_cid": statement["population_cid"],
            "plan_root_cid": statement["plan_root_cid"],
            "control_projection_root": statement["control_projection_root"],
            "coordination_projection_root": statement[
                "coordination_projection_root"
            ],
            "execution_projection_root": statement[
                "execution_projection_root"
            ],
            "provider_container_qualification_cid": statement[
                "provider_container_qualification_cid"
            ],
            "image_digest": statement["qualified_worker_image_digest"],
            "container_profile_cid": statement[
                "qualified_worker_container_profile_cid"
            ],
            "provider_maximum_parallel_workers": statement[
                "provider_maximum_parallel_workers"
            ],
            "provider_maximum_parallel_containers": statement[
                "provider_maximum_parallel_containers"
            ],
            "provider_worker_principal_did": statement[
                "provider_worker_principal_did"
            ],
            "provider_principal_did": statement["provider_principal_did"],
            "provider_task_dispatch_admitted": statement[
                "provider_task_dispatch_admitted"
            ],
            "provider_workload_class": statement["provider_workload_class"],
            "quack_owner_qualification_cid": statement[
                "quack_owner_qualification_cid"
            ],
            "quack_owner_verification_cid": statement[
                "quack_owner_verification_cid"
            ],
            "quack_shard_id": statement["quack_shard_id"],
            "quack_epoch": statement["quack_epoch"],
            "quack_fence": statement["quack_fence"],
            "quack_owner_principal_did": statement[
                "quack_owner_principal_did"
            ],
        },
    )
    live_config = {
        "schema": capsule.EAAEF_LIVE_SEAL_CONFIG_SCHEMA,
        "authority_registry_prefix": capsule.EAAEF_AUTHORITY_REGISTRY_PREFIX,
        "bootstrap_admission_schema": (
            "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-admission@1"
        ),
        "configured_board_launch_capsule_schema": (
            capsule.EAAEF_CONFIGURED_BOARD_CAPSULE_SCHEMA
        ),
        "trusted_operator_dids": ["did:key:zOperator"],
        "trusted_security_reviewer_dids": ["did:key:zSecurity"],
        "trusted_capsule_reviewer_dids": [reviewer],
        "worker_network_authorization_policy": _network_policy(),
        "maximum_lanes": 5,
    }
    # The tracked configuration intentionally has no receipt/capsule CID, so
    # source freeze precedes signing without a cryptographic fixed-point cycle.
    assert not any(key.endswith("_cid") for key in live_config)
    first = capsule.verify_external_agent_configured_board_live_seal(
        live_config,
        repo_root=tmp_path,
        configuration_root=statement["configuration_root"],
        expected_source_head=statement["source_head"],
        expected_source_tree=statement["source_tree"],
        accepted_control_plane_pin=_pin(),
        now_ms=NOW_MS,
        authority_root=authority_root,
    )
    assert first["valid"] is True

    tampered = deepcopy(value)
    tampered["quack_fence"] = 99
    os.chmod(capsule_path, 0o600)
    capsule_path.write_text(json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(capsule_path, 0o400)
    with pytest.raises(capsule.ExternalAgentConfiguredBoardCapsuleError, match="self-address"):
        capsule.verify_external_agent_configured_board_live_seal(
            live_config,
            repo_root=tmp_path,
            configuration_root=statement["configuration_root"],
            expected_source_head=statement["source_head"],
            expected_source_tree=statement["source_tree"],
            accepted_control_plane_pin=_pin(),
            now_ms=NOW_MS,
            authority_root=authority_root,
        )
