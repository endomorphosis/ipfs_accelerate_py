from __future__ import annotations

import base64
from copy import deepcopy

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    external_agent_control_plane_promotion as promotion,
)

NOW_MS = 1_800_000_000_000


def _sha(token: str) -> str:
    return "sha256:" + token * 64


def _did() -> str:
    return ed25519_did_key(Ed25519PrivateKey.generate().public_key())


def _board() -> dict[str, object]:
    return {
        "board_namespace": promotion.EAAEF_BOARD_NAMESPACE,
        "board_cid": _sha("1"),
        "operational_command_fabric": {
            "store_id": "eaaef-control-run-v5",
            "schema_revision": "datasets-authoritative-operational-v1",
        },
    }


def _materialization() -> dict[str, object]:
    return {
        "receipt_cid": _sha("2"),
        "namespace_claim_cid": _sha("3"),
        "population_cid": _sha("4"),
        "plan_root_cid": _sha("5"),
        "source_generation": {"source_generation_cid": _sha("6")},
        "controls": {"board_cid": _sha("1")},
        "database_program_bindings": {
            "bootstrap": {"store_generation": "eaaef-run-v5"}
        },
        "control_projection": {"projection_root": _sha("7")},
        "coordination_projection": {"projection_root": _sha("8")},
        "execution_projection": {"projection_root": _sha("9")},
    }


def _signed_v1_receipt() -> tuple[dict[str, object], str]:
    key = Ed25519PrivateKey.generate()
    reviewer = ed25519_did_key(key.public_key())
    receipt: dict[str, object] = {
        "schema": promotion.EAAEF_QUACK_OWNER_QUALIFICATION_SCHEMA,
        "board_namespace": promotion.EAAEF_BOARD_NAMESPACE,
        "board_cid": _sha("1"),
        "source": {
            "repository": "ipfs_accelerate_py",
            "commit": "a" * 40,
            "tree": "b" * 40,
            "source_generation_cid": _sha("6"),
        },
        "materialization": {
            "generation": "eaaef-run-v5",
            "receipt_cid": _sha("2"),
            "namespace_claim_cid": _sha("3"),
            "population_cid": _sha("4"),
            "plan_root_cid": _sha("5"),
            "control_projection_root": _sha("7"),
            "coordination_projection_root": _sha("8"),
            "execution_projection_root": _sha("9"),
        },
        "profile": {
            "profile_id": promotion.EAAEF_PROFILE_ID,
            "platform": "linux_arm64",
            "duckdb_version": promotion.REQUIRED_DUCKDB_VERSION,
            "duckdb_artifact_sha256": _sha("a"),
            "quack_build": promotion.REQUIRED_QUACK_BUILD,
            "quack_extension_sha256": _sha("b"),
            "schema_revision": "datasets-authoritative-operational-v1",
            "schema_fingerprint": _sha("c"),
        },
        "owner": {
            "shard_id": "eaaef-control",
            "store_id": "eaaef-control-run-v5",
            "database_uuid": "db-1234",
            "server_id": "server-1",
            "process_birth_id": "birth-1",
            "owner_generation": 1,
            "epoch": 2,
            "fence": 3,
            "lease_id": "lease-1",
            "owner_principal_did": "did:key:zOwner",
        },
        "transport": {
            "mode": "quack",
            "authenticated": True,
            "typed_requests_only": True,
            "raw_sql_allowed": False,
            "direct_file_fallback": False,
            "maximum_file_owners": 1,
            "multi_reader_writer_qualified": True,
        },
        "qualification": {
            "status": "accepted",
            "readiness_receipt_cid": _sha("d"),
            "stale_fence_test_cid": _sha("e"),
            "idempotency_test_cid": _sha("f"),
            "failover_test_cid": _sha("0"),
            "qualified_at_ms": NOW_MS - 1000,
            "expires_at_ms": NOW_MS + 1000,
            "ducklake_required": False,
            "ducklake_authority": False,
        },
        "reviewer": {
            "identity_did": reviewer,
            "role": "independent_control_plane_reviewer",
        },
    }
    receipt["reviewer_signature"] = base64.b64encode(
        key.sign(promotion.external_agent_control_plane_promotion_signing_bytes(receipt))
    ).decode("ascii")
    receipt["receipt_cid"] = promotion._cid(receipt)
    return receipt, reviewer


def _signed_capability(
    *,
    key: Ed25519PrivateKey,
    owner_did: str,
    command_fabric_cid: str,
) -> dict[str, object]:
    reviewer = ed25519_did_key(key.public_key())
    value: dict[str, object] = {
        "schema": promotion.PLAN_R2_OPERATIONAL_CAPABILITY_SCHEMA,
        "allowed": True,
        "blockers": [],
        "source_head": "a" * 40,
        "source_tree": "b" * 40,
        "bootstrap_admission_cid": _sha("c"),
        "quack_owner_qualification_cid": _sha("d"),
        "quack_command_fabric_qualification_cid": command_fabric_cid,
        "owner_principal_did": owner_did,
        "shard_id": "eaaef-control",
        "owner_generation": 4,
        "epoch": 5,
        "fence": 6,
        "duckdb_version": promotion.REQUIRED_DUCKDB_VERSION,
        "quack_build": promotion.REQUIRED_QUACK_BUILD,
        "authorized_state_command_schema": (
            "ipfs_accelerate_py/agent-supervisor/authorized-state-command@1"
        ),
        "ingress_authenticated": True,
        "ingress_append_only_single_relation": True,
        "ingress_accepts_signed_envelope_only": True,
        "bare_state_command_rejected": True,
        "owner_verifies_authorized_state_command": True,
        "authority_ref_binds_transition_authorization": True,
        "local_owner_verifies_transition_authorization": True,
        "operational_database_private": True,
        "one_mutable_owner": True,
        "atomic_plan_population_cas": True,
        "egress_read_only": True,
        "egress_append_denied": True,
        "durable_idempotent_receipts": True,
        "protected_full_rows_bound": True,
        "reviewer_identity_did": reviewer,
        "issued_at_ms": NOW_MS - 900,
        "expires_at_ms": NOW_MS + 50_000,
    }
    from ipfs_accelerate_py.agent_supervisor.planning import (
        external_agent_plan_r2 as plan_r2,
    )

    value["reviewer_signature"] = base64.b64encode(
        key.sign(plan_r2._canonical_bytes(value))
    ).decode("ascii")
    value["capability_cid"] = plan_r2._cid(value)
    return value


def _signed_v2_promotion() -> tuple[
    dict[str, object],
    dict[str, str],
    dict[str, Ed25519PrivateKey],
]:
    keys = {
        "reviewer": Ed25519PrivateKey.generate(),
        "operator": Ed25519PrivateKey.generate(),
        "security": Ed25519PrivateKey.generate(),
    }
    reviewer_did = ed25519_did_key(keys["reviewer"].public_key())
    identities = {
        "reviewer": reviewer_did,
        "operator": ed25519_did_key(keys["operator"].public_key()),
        "security": ed25519_did_key(keys["security"].public_key()),
        "owner": _did(),
        "approver": _did(),
        "principal": _did(),
    }
    command_fabric_cid = _sha("e")
    capability = _signed_capability(
        key=keys["reviewer"],
        owner_did=identities["owner"],
        command_fabric_cid=command_fabric_cid,
    )
    policy: dict[str, object] = {
        "schema": promotion.QUACK_COMMAND_AUTHORIZATION_POLICY_SCHEMA,
        "board_namespace": promotion.EAAEF_BOARD_NAMESPACE,
        "shard_id": "eaaef-control",
        "store_id": "eaaef-control-run-v5",
        "authority_ref_cid": _sha("f"),
        "owner_principal_did": identities["owner"],
        "owner_generation": 4,
        "fence_epoch": 6,
        "trusted_approver_dids": [identities["approver"]],
        "authorized_principal_dids": [identities["principal"]],
        "allowed_command_kinds": ["migrate", "observe"],
        "maximum_authorization_lifetime_ms": 300_000,
    }
    policy["policy_cid"] = promotion._cid(policy)
    receipt: dict[str, object] = {
        "schema": promotion.EAAEF_CONTROL_PLANE_PROMOTION_SCHEMA_V2,
        "board_namespace": promotion.EAAEF_BOARD_NAMESPACE,
        "board_cid": _sha("1"),
        "source": {
            "repository": "ipfs_accelerate_py",
            "commit": "a" * 40,
            "tree": "b" * 40,
            "source_generation_cid": _sha("6"),
        },
        "materialization": {
            "generation": "eaaef-run-v5",
            "receipt_cid": _sha("2"),
            "namespace_claim_cid": _sha("3"),
            "population_cid": _sha("4"),
            "plan_root_cid": _sha("5"),
            "control_projection_root": _sha("7"),
            "coordination_projection_root": _sha("8"),
            "execution_projection_root": _sha("9"),
        },
        "profile": {
            "profile_id": promotion.EAAEF_PROFILE_ID,
            "platform": "linux_arm64",
            "duckdb_version": promotion.REQUIRED_DUCKDB_VERSION,
            "duckdb_artifact_sha256": _sha("a"),
            "quack_build": promotion.REQUIRED_QUACK_BUILD,
            "quack_extension_sha256": _sha("b"),
            "quack_lockfile_cid": _sha("0"),
            "schema_revision": "datasets-authoritative-operational-v1",
            "schema_fingerprint": _sha("1"),
        },
        "owner": {
            "shard_id": "eaaef-control",
            "store_id": "eaaef-control-run-v5",
            "database_uuid": "db-1234",
            "server_id": "server-1",
            "process_birth_id": "birth-1",
            "owner_generation": 4,
            "epoch": 5,
            "fence": 6,
            "lease_id": "lease-1",
            "owner_principal_did": identities["owner"],
        },
        "transport": {
            "mode": "quack",
            "authenticated": True,
            "typed_requests_only": True,
            "raw_sql_allowed": False,
            "direct_file_fallback": False,
            "maximum_file_owners": 1,
            "multi_reader_writer_qualified": True,
        },
        "dispatcher": {
            "dispatcher_interface": promotion.PLAN_R2_OWNER_GATEWAY_INTERFACE,
            "command_fabric_interface": promotion.QUACK_COMMAND_FABRIC_INTERFACE,
            "operation_schema": promotion.PLAN_R2_OWNER_OPERATION_SCHEMA,
            "authorized_state_command_interface": (
                promotion.AUTHORIZED_STATE_COMMAND_INTERFACE
            ),
            "state_command_interface": promotion.STATE_COMMAND_INTERFACE,
            "operation_vocabulary": promotion.exact_plan_r2_operation_vocabulary(),
            "operation_vocabulary_cid": (
                promotion.plan_r2_operation_vocabulary_cid()
            ),
            "plan_r2_operational_capability_cid": capability["capability_cid"],
            "command_fabric_qualification_cid": command_fabric_cid,
            "authorization_policy_cid": policy["policy_cid"],
            "generic_daemon_gateway_admitted": False,
        },
        "authorization_policy": policy,
        "plan_r2_operational_capability": capability,
        "atomicity": dict(promotion._EXPECTED_ATOMICITY),
        "evidence": {
            "apply_readback_cid": _sha("2"),
            "rollback_cid": _sha("3"),
            "replay_cid": _sha("4"),
            "stale_fence_cid": _sha("5"),
            "revoked_lease_cid": _sha("6"),
            "crash_ambiguity_cid": _sha("7"),
            "distinct_shard_store_cid": _sha("8"),
            "gateway_forgery_cid": _sha("9"),
        },
        "qualification": {
            "status": "accepted",
            "qualified_at_ms": NOW_MS - 100,
            "expires_at_ms": NOW_MS + 40_000,
        },
        "one_use_nonce": "promotion-nonce-1",
        "reviewer": {
            "identity_did": identities["reviewer"],
            "role": "independent_control_plane_reviewer",
        },
        "operator": {
            "identity_did": identities["operator"],
            "role": "independent_operator",
        },
        "security_reviewer": {
            "identity_did": identities["security"],
            "role": "independent_security_reviewer",
        },
    }
    _resign_v2(receipt, keys)
    return receipt, identities, keys


def _resign_v2(
    receipt: dict[str, object],
    keys: dict[str, Ed25519PrivateKey],
) -> None:
    for field in (
        "reviewer_signature",
        "operator_signature",
        "security_reviewer_signature",
    ):
        receipt.pop(field, None)
    receipt.pop("receipt_cid", None)
    signing_bytes = promotion.external_agent_control_plane_promotion_signing_bytes(
        receipt
    )
    for role, field in (
        ("reviewer", "reviewer_signature"),
        ("operator", "operator_signature"),
        ("security", "security_reviewer_signature"),
    ):
        receipt[field] = base64.b64encode(keys[role].sign(signing_bytes)).decode(
            "ascii"
        )
    receipt["receipt_cid"] = promotion._cid(receipt)


def _verify_v2(receipt: object, identities: dict[str, str]):
    return promotion.verify_external_agent_control_plane_promotion(
        qualification_receipt=receipt,
        board=_board(),
        materialization_receipt=_materialization(),
        expected_source_commit="a" * 40,
        expected_source_tree="b" * 40,
        trusted_reviewer_dids=[identities["reviewer"]],
        trusted_operator_dids=[identities["operator"]],
        trusted_security_reviewer_dids=[identities["security"]],
        now_ms=NOW_MS,
    )


def test_v1_remains_historical_but_cannot_promote_dispatcher() -> None:
    receipt, reviewer = _signed_v1_receipt()
    historical = promotion.verify_external_agent_quack_owner_qualification_v1(
        qualification_receipt=receipt,
        board=_board(),
        materialization_receipt=_materialization(),
        expected_source_commit="a" * 40,
        expected_source_tree="b" * 40,
        trusted_reviewer_dids=[reviewer],
        now_ms=NOW_MS,
    )
    assert historical["allowed"] is True
    assert historical["historical_only"] is True
    assert historical["promotion_allowed"] is False

    promoted = promotion.verify_external_agent_control_plane_promotion(
        qualification_receipt=receipt,
        board=_board(),
        materialization_receipt=_materialization(),
        expected_source_commit="a" * 40,
        expected_source_tree="b" * 40,
        trusted_reviewer_dids=[reviewer],
        now_ms=NOW_MS,
    )
    assert promoted["allowed"] is False
    assert promoted["blockers"] == ["quack_control_plane_promotion_v2_required"]


def test_exact_signed_v2_promotion_is_effect_free_and_cycle_free() -> None:
    receipt, identities, _keys = _signed_v2_promotion()
    result = _verify_v2(receipt, identities)

    assert result["allowed"] is True
    assert result["promotion_allowed"] is True
    assert result["blockers"] == []
    assert result["dispatcher_interface"] == promotion.PLAN_R2_OWNER_GATEWAY_INTERFACE
    assert result["command_fabric_interface"] == promotion.QUACK_COMMAND_FABRIC_INTERFACE
    assert result["base_owner_qualification_cid"] == _sha("d")
    assert result["bootstrap_admission_cid"] == _sha("c")
    assert result["authority_mutated"] is False
    assert result["process_started"] is False


def test_vocabulary_drift_and_forged_capability_fail_closed() -> None:
    receipt, identities, keys = _signed_v2_promotion()
    drift = deepcopy(receipt)
    drift_vocabulary = drift["dispatcher"]["operation_vocabulary"]
    drift_vocabulary[0]["command_kind"] = "migrate"
    drift["dispatcher"]["operation_vocabulary_cid"] = promotion._cid(
        {
            "schema": promotion.EAAEF_PLAN_R2_OPERATION_VOCABULARY_SCHEMA,
            "operations": drift_vocabulary,
        }
    )
    _resign_v2(drift, keys)
    result = _verify_v2(drift, identities)
    assert result["allowed"] is False
    assert "quack_plan_r2_operation_vocabulary_invalid" in result["blockers"]

    forged = deepcopy(receipt)
    forged["plan_r2_operational_capability"]["atomic_plan_population_cas"] = False
    _resign_v2(forged, keys)
    result = _verify_v2(forged, identities)
    assert result["allowed"] is False
    assert "quack_plan_r2_operational_capability_invalid" in result["blockers"]


def test_stale_owner_fence_and_reviewer_conflicts_fail_closed() -> None:
    receipt, identities, keys = _signed_v2_promotion()
    stale = deepcopy(receipt)
    stale["owner"]["fence"] = 7
    _resign_v2(stale, keys)
    result = _verify_v2(stale, identities)
    assert result["allowed"] is False
    assert "quack_command_authorization_policy_invalid" in result["blockers"]
    assert "quack_plan_r2_capability_identity_mismatch" in result["blockers"]

    conflict = deepcopy(receipt)
    conflict["operator"]["identity_did"] = identities["reviewer"]
    _resign_v2(conflict, keys)
    result = _verify_v2(conflict, identities)
    assert result["allowed"] is False
    assert "quack_control_plane_reviewers_not_independent" in result["blockers"]


def test_named_operator_and_security_reviewer_must_both_participate() -> None:
    receipt, identities, keys = _signed_v2_promotion()

    missing_operator = deepcopy(receipt)
    missing_operator["operator_signature"] = ""
    missing_operator["receipt_cid"] = promotion._cid(
        {key: value for key, value in missing_operator.items() if key != "receipt_cid"}
    )
    result = _verify_v2(missing_operator, identities)
    assert result["allowed"] is False
    assert "quack_control_plane_promotion_operator_unsigned" in result["blockers"]

    forged_security = deepcopy(receipt)
    forged_security["security_reviewer_signature"] = base64.b64encode(
        keys["operator"].sign(
            promotion.external_agent_control_plane_promotion_signing_bytes(
                forged_security
            )
        )
    ).decode("ascii")
    forged_security["receipt_cid"] = promotion._cid(
        {key: value for key, value in forged_security.items() if key != "receipt_cid"}
    )
    result = _verify_v2(forged_security, identities)
    assert result["allowed"] is False
    assert (
        "quack_control_plane_promotion_security_reviewer_signature_invalid"
        in result["blockers"]
    )
