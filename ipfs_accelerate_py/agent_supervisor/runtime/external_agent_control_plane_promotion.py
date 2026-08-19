"""Read-only verification for EAAEF DuckDB/Quack promotion evidence.

``eaaef-quack-owner-qualification@1`` predates the production Plan-R2 owner
dispatcher.  It remains verifiable as historical, read-only owner evidence,
but it cannot promote that dispatcher.  ``eaaef-control-plane-promotion@2``
adds the closed operation vocabulary, independently signed Plan-R2
capability, exact authorization policy, immutable build identity, and current
failure evidence required for promotion.

This module has no signing or runtime effects.  It never opens DuckDB, loads
Quack, starts a listener, mints a capability, or mutates authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    LocalProfileTampered,
    verify_did_key_signature,
)

EAAEF_QUACK_OWNER_QUALIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-quack-owner-qualification@1"
)
EAAEF_QUACK_OWNER_VERIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-quack-owner-verification@1"
)
EAAEF_CONTROL_PLANE_PROMOTION_SCHEMA_V2: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-control-plane-promotion@2"
)
EAAEF_CONTROL_PLANE_PROMOTION_VERIFICATION_SCHEMA_V2: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-control-plane-promotion-verification@2"
)
EAAEF_PLAN_R2_OPERATION_VOCABULARY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-plan-r2-operation-vocabulary@1"
)
EAAEF_BOARD_NAMESPACE: Final = "external-agent-autonomous-execution-fabric-v1"
EAAEF_PROFILE_ID: Final = "eaaef-duckdb-quack-1.5.5"
REQUIRED_DUCKDB_VERSION: Final = "1.5.5"
REQUIRED_QUACK_BUILD: Final = "quack@1.5.5+core"
PLAN_R2_OWNER_GATEWAY_INTERFACE: Final = "AuthorizedStateCommandPlanR2OwnerGateway@1"
QUACK_COMMAND_FABRIC_INTERFACE: Final = "QuackCommandFabric@1"
PLAN_R2_OWNER_OPERATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/authorized-plan-r2-owner-operation@1"
)
AUTHORIZED_STATE_COMMAND_INTERFACE: Final = "AuthorizedStateCommand@1"
STATE_COMMAND_INTERFACE: Final = "StateCommand@1"
PLAN_R2_OPERATIONAL_CAPABILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-plan-r2-operational-capability@1"
)
QUACK_COMMAND_AUTHORIZATION_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-command-authorization-policy@1"
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,511}\Z")
_DID = re.compile(r"did:key:z[A-Za-z0-9]+\Z")

_V1_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema",
        "board_namespace",
        "board_cid",
        "source",
        "materialization",
        "profile",
        "owner",
        "transport",
        "qualification",
        "reviewer",
        "reviewer_signature",
        "receipt_cid",
    }
)
_V2_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema",
        "board_namespace",
        "board_cid",
        "source",
        "materialization",
        "profile",
        "owner",
        "transport",
        "dispatcher",
        "authorization_policy",
        "plan_r2_operational_capability",
        "atomicity",
        "evidence",
        "qualification",
        "one_use_nonce",
        "reviewer",
        "operator",
        "security_reviewer",
        "reviewer_signature",
        "operator_signature",
        "security_reviewer_signature",
        "receipt_cid",
    }
)
_SOURCE_FIELDS: Final = frozenset(
    {"repository", "commit", "tree", "source_generation_cid"}
)
_MATERIALIZATION_FIELDS: Final = frozenset(
    {
        "generation",
        "receipt_cid",
        "namespace_claim_cid",
        "population_cid",
        "plan_root_cid",
        "control_projection_root",
        "coordination_projection_root",
        "execution_projection_root",
    }
)
_V1_PROFILE_FIELDS: Final = frozenset(
    {
        "profile_id",
        "platform",
        "duckdb_version",
        "duckdb_artifact_sha256",
        "quack_build",
        "quack_extension_sha256",
        "schema_revision",
        "schema_fingerprint",
    }
)
_V2_PROFILE_FIELDS: Final = frozenset({*_V1_PROFILE_FIELDS, "quack_lockfile_cid"})
_OWNER_FIELDS: Final = frozenset(
    {
        "shard_id",
        "store_id",
        "database_uuid",
        "server_id",
        "process_birth_id",
        "owner_generation",
        "epoch",
        "fence",
        "lease_id",
        "owner_principal_did",
    }
)
_TRANSPORT_FIELDS: Final = frozenset(
    {
        "mode",
        "authenticated",
        "typed_requests_only",
        "raw_sql_allowed",
        "direct_file_fallback",
        "maximum_file_owners",
        "multi_reader_writer_qualified",
    }
)
_V1_QUALIFICATION_FIELDS: Final = frozenset(
    {
        "status",
        "readiness_receipt_cid",
        "stale_fence_test_cid",
        "idempotency_test_cid",
        "failover_test_cid",
        "qualified_at_ms",
        "expires_at_ms",
        "ducklake_required",
        "ducklake_authority",
    }
)
_V2_QUALIFICATION_FIELDS: Final = frozenset(
    {"status", "qualified_at_ms", "expires_at_ms"}
)
_REVIEWER_FIELDS: Final = frozenset({"identity_did", "role"})
_DISPATCHER_FIELDS: Final = frozenset(
    {
        "dispatcher_interface",
        "command_fabric_interface",
        "operation_schema",
        "authorized_state_command_interface",
        "state_command_interface",
        "operation_vocabulary",
        "operation_vocabulary_cid",
        "plan_r2_operational_capability_cid",
        "command_fabric_qualification_cid",
        "authorization_policy_cid",
        "generic_daemon_gateway_admitted",
    }
)
_OPERATION_FIELDS: Final = frozenset(
    {
        "operation",
        "command_kind",
        "payload_fields",
        "command_parameter_fields",
        "response_schema",
    }
)
_AUTHORIZATION_POLICY_FIELDS: Final = frozenset(
    {
        "schema",
        "board_namespace",
        "shard_id",
        "store_id",
        "authority_ref_cid",
        "owner_principal_did",
        "owner_generation",
        "fence_epoch",
        "trusted_approver_dids",
        "authorized_principal_dids",
        "allowed_command_kinds",
        "maximum_authorization_lifetime_ms",
        "policy_cid",
    }
)
_ATOMICITY_FIELDS: Final = frozenset(
    {
        "single_owner_transaction",
        "plan_task_dependency_frontier_atomic",
        "command_receipt_same_transaction",
        "idempotency_nonce_same_transaction",
        "compare_and_swap_required",
        "live_lease_required",
        "stale_fence_rejected",
        "revoked_lease_rejected",
        "commit_ambiguity_readback",
        "projection_derived_after_commit",
        "projection_is_authority",
        "direct_duckdb_file_open",
        "worker_self_approval",
    }
)
_EVIDENCE_FIELDS: Final = frozenset(
    {
        "apply_readback_cid",
        "rollback_cid",
        "replay_cid",
        "stale_fence_cid",
        "revoked_lease_cid",
        "crash_ambiguity_cid",
        "distinct_shard_store_cid",
        "gateway_forgery_cid",
    }
)
_COMMAND_PARAMETER_FIELDS: Final = (
    "authorization_cid",
    "expected_event_cursor",
    "interface",
    "operation",
    "operation_payload_cid",
    "population_cid",
    "prepared_projection_cid",
    "protected_tasks_root_cid",
    "shard_id",
    "statement_cid",
    "store_id",
    "transition_receipt_cid",
)
_EXACT_OPERATION_VOCABULARY: Final = (
    {
        "operation": "plan_r2.prepare",
        "command_kind": "observe",
        "payload_fields": ["authorization", "operation", "schema"],
        "command_parameter_fields": list(_COMMAND_PARAMETER_FIELDS),
        "response_schema": (
            "ipfs_accelerate_py/agent-supervisor/eaaef-plan-r2-prepared-projection@1"
        ),
    },
    {
        "operation": "plan_r2.apply",
        "command_kind": "migrate",
        "payload_fields": [
            "authorization",
            "operation",
            "prepared_projection",
            "schema",
        ],
        "command_parameter_fields": list(_COMMAND_PARAMETER_FIELDS),
        "response_schema": (
            "ipfs_accelerate_py/agent-supervisor/eaaef-plan-r2-transition-receipt@1"
        ),
    },
    {
        "operation": "plan_r2.observe",
        "command_kind": "observe",
        "payload_fields": [
            "authorization",
            "operation",
            "schema",
            "transition_receipt",
        ],
        "command_parameter_fields": list(_COMMAND_PARAMETER_FIELDS),
        "response_schema": (
            "ipfs_accelerate_py/agent-supervisor/eaaef-plan-r2-state-observation@1"
        ),
    },
)
_EXPECTED_ATOMICITY: Final = {
    "single_owner_transaction": True,
    "plan_task_dependency_frontier_atomic": True,
    "command_receipt_same_transaction": True,
    "idempotency_nonce_same_transaction": True,
    "compare_and_swap_required": True,
    "live_lease_required": True,
    "stale_fence_rejected": True,
    "revoked_lease_rejected": True,
    "commit_ambiguity_readback": True,
    "projection_derived_after_commit": True,
    "projection_is_authority": False,
    "direct_duckdb_file_open": False,
    "worker_self_approval": False,
}
_EXPECTED_TRANSPORT: Final = {
    "mode": "quack",
    "authenticated": True,
    "typed_requests_only": True,
    "raw_sql_allowed": False,
    "direct_file_fallback": False,
    "maximum_file_owners": 1,
    "multi_reader_writer_qualified": True,
}


class ExternalAgentControlPlanePromotionError(ValueError):
    """The qualification receipt cannot be interpreted canonically."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExternalAgentControlPlanePromotionError(
            "Quack qualification is not canonical JSON"
        ) from exc


def _cid(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def external_agent_control_plane_promotion_signing_bytes(
    receipt: Mapping[str, Any],
) -> bytes:
    """Return the exact three-party approval statement; never sign it locally."""

    body = dict(receipt)
    for field in (
        "reviewer_signature",
        "operator_signature",
        "security_reviewer_signature",
    ):
        body.pop(field, None)
    body.pop("receipt_cid", None)
    return _canonical_bytes(body)


def exact_plan_r2_operation_vocabulary() -> list[dict[str, Any]]:
    """Return a detached copy of the only dispatcher vocabulary promoted by v2."""

    return json.loads(json.dumps(_EXACT_OPERATION_VOCABULARY))


def plan_r2_operation_vocabulary_cid() -> str:
    return _cid(
        {
            "schema": EAAEF_PLAN_R2_OPERATION_VOCABULARY_SCHEMA,
            "operations": exact_plan_r2_operation_vocabulary(),
        }
    )


def _closed(value: object, fields: frozenset[str]) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) and set(value) == fields else None


def _positive_int(value: object) -> bool:
    return type(value) is int and int(value) > 0


def _materialization_projection(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    controls = value.get("controls")
    source_generation = value.get("source_generation")
    database_bindings = value.get("database_program_bindings")
    control = value.get("control_projection")
    coordination = value.get("coordination_projection")
    execution = value.get("execution_projection")
    if not all(
        isinstance(item, Mapping)
        for item in (
            controls,
            source_generation,
            database_bindings,
            control,
            coordination,
            execution,
        )
    ):
        return {}
    bootstrap = database_bindings.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        return {}
    return {
        "generation": str(bootstrap.get("store_generation") or ""),
        "receipt_cid": str(value.get("receipt_cid") or ""),
        "namespace_claim_cid": str(value.get("namespace_claim_cid") or ""),
        "population_cid": str(value.get("population_cid") or ""),
        "plan_root_cid": str(value.get("plan_root_cid") or ""),
        "control_projection_root": str(control.get("projection_root") or ""),
        "coordination_projection_root": str(
            coordination.get("projection_root") or ""
        ),
        "execution_projection_root": str(execution.get("projection_root") or ""),
        "source_generation_cid": str(
            source_generation.get("source_generation_cid") or ""
        ),
        "board_cid": str(controls.get("board_cid") or ""),
    }


def _base_identity_checks(
    *,
    receipt: Mapping[str, Any],
    source: Mapping[str, Any] | None,
    materialization: Mapping[str, Any] | None,
    profile: Mapping[str, Any] | None,
    owner: Mapping[str, Any] | None,
    transport: Mapping[str, Any] | None,
    board: Mapping[str, Any],
    materialization_receipt: Mapping[str, Any],
    expected_source_commit: str,
    expected_source_tree: str,
    profile_fields: frozenset[str],
) -> tuple[list[str], Mapping[str, Any], dict[str, str]]:
    blockers: list[str] = []
    projection = _materialization_projection(materialization_receipt)
    operational = board.get("operational_command_fabric")
    if not isinstance(operational, Mapping):
        blockers.append("quack_owner_operational_profile_missing")
        operational = {}
    if (
        board.get("board_namespace") != EAAEF_BOARD_NAMESPACE
        or not _SHA256.fullmatch(str(board.get("board_cid") or ""))
        or receipt.get("board_namespace") != board.get("board_namespace")
        or receipt.get("board_cid") != board.get("board_cid")
    ):
        blockers.append("quack_owner_board_identity_mismatch")
    if source is None or (
        source.get("repository") != "ipfs_accelerate_py"
        or source.get("commit") != expected_source_commit
        or source.get("tree") != expected_source_tree
        or not _GIT_OBJECT.fullmatch(expected_source_commit)
        or not _GIT_OBJECT.fullmatch(expected_source_tree)
        or source.get("source_generation_cid")
        != projection.get("source_generation_cid")
    ):
        blockers.append("quack_owner_source_identity_mismatch")
    if materialization is None or not projection or any(
        materialization.get(field) != projection.get(field)
        for field in _MATERIALIZATION_FIELDS
    ):
        blockers.append("quack_owner_materialization_identity_mismatch")
    if projection.get("board_cid") != str(board.get("board_cid") or ""):
        blockers.append("quack_owner_materialization_board_mismatch")
    if profile is None or set(profile) != profile_fields or (
        profile.get("profile_id") != EAAEF_PROFILE_ID
        or profile.get("duckdb_version") != REQUIRED_DUCKDB_VERSION
        or profile.get("quack_build") != REQUIRED_QUACK_BUILD
        or profile.get("schema_revision") != operational.get("schema_revision")
        or not _SAFE_ID.fullmatch(str(profile.get("platform") or ""))
        or any(
            not _SHA256.fullmatch(str(profile.get(field) or ""))
            for field in (
                "duckdb_artifact_sha256",
                "quack_extension_sha256",
                "schema_fingerprint",
            )
        )
    ):
        blockers.append("quack_owner_exact_profile_invalid")
    if profile_fields == _V2_PROFILE_FIELDS and (
        profile is None
        or not _SHA256.fullmatch(str(profile.get("quack_lockfile_cid") or ""))
    ):
        blockers.append("quack_owner_exact_profile_invalid")
    if owner is None or (
        not _SAFE_ID.fullmatch(str(owner.get("shard_id") or ""))
        or not _SAFE_ID.fullmatch(str(owner.get("store_id") or ""))
        or owner.get("store_id") != operational.get("store_id")
        or owner.get("shard_id") == owner.get("store_id")
        or not _SAFE_ID.fullmatch(str(owner.get("database_uuid") or ""))
        or not _SAFE_ID.fullmatch(str(owner.get("server_id") or ""))
        or not _SAFE_ID.fullmatch(str(owner.get("process_birth_id") or ""))
        or not _SAFE_ID.fullmatch(str(owner.get("lease_id") or ""))
        or not _DID.fullmatch(str(owner.get("owner_principal_did") or ""))
        or any(
            not _positive_int(owner.get(field))
            for field in ("owner_generation", "epoch", "fence")
        )
    ):
        blockers.append("quack_owner_identity_invalid")
    if transport is None or dict(transport) != _EXPECTED_TRANSPORT:
        blockers.append("quack_owner_transport_not_qualified")
    return list(dict.fromkeys(blockers)), operational, projection


def verify_external_agent_quack_owner_qualification_v1(
    *,
    qualification_receipt: object,
    board: Mapping[str, Any],
    materialization_receipt: Mapping[str, Any],
    expected_source_commit: str,
    expected_source_tree: str,
    trusted_reviewer_dids: Sequence[str],
    now_ms: int,
) -> dict[str, Any]:
    """Verify historical owner evidence without promoting a dispatcher."""

    receipt = (
        dict(qualification_receipt)
        if isinstance(qualification_receipt, Mapping)
        else {}
    )
    source = _closed(receipt.get("source"), _SOURCE_FIELDS)
    materialization = _closed(receipt.get("materialization"), _MATERIALIZATION_FIELDS)
    profile = _closed(receipt.get("profile"), _V1_PROFILE_FIELDS)
    owner = _closed(receipt.get("owner"), _OWNER_FIELDS)
    transport = _closed(receipt.get("transport"), _TRANSPORT_FIELDS)
    qualification = _closed(receipt.get("qualification"), _V1_QUALIFICATION_FIELDS)
    reviewer = _closed(receipt.get("reviewer"), _REVIEWER_FIELDS)
    blockers, _operational, _projection = _base_identity_checks(
        receipt=receipt,
        source=source,
        materialization=materialization,
        profile=profile,
        owner=owner,
        transport=transport,
        board=board,
        materialization_receipt=materialization_receipt,
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
        profile_fields=_V1_PROFILE_FIELDS,
    )
    if set(receipt) != _V1_RECEIPT_FIELDS:
        blockers.append("quack_owner_qualification_shape_invalid")
    if receipt.get("schema") != EAAEF_QUACK_OWNER_QUALIFICATION_SCHEMA:
        blockers.append("quack_owner_qualification_schema_invalid")
    claimed_cid = str(receipt.get("receipt_cid") or "")
    cid_body = dict(receipt)
    cid_body.pop("receipt_cid", None)
    if not _SHA256.fullmatch(claimed_cid) or claimed_cid != _cid(cid_body):
        blockers.append("quack_owner_qualification_self_address_invalid")
    if qualification is None or (
        qualification.get("status") != "accepted"
        or qualification.get("ducklake_required") is not False
        or qualification.get("ducklake_authority") is not False
        or any(
            not _SHA256.fullmatch(str(qualification.get(field) or ""))
            for field in (
                "readiness_receipt_cid",
                "stale_fence_test_cid",
                "idempotency_test_cid",
                "failover_test_cid",
            )
        )
        or not _positive_int(qualification.get("qualified_at_ms"))
        or not _positive_int(qualification.get("expires_at_ms"))
        or not _positive_int(now_ms)
        or int(qualification.get("qualified_at_ms") or 0) > now_ms
        or now_ms >= int(qualification.get("expires_at_ms") or 0)
    ):
        blockers.append("quack_owner_qualification_not_current")
    reviewer_did = str((reviewer or {}).get("identity_did") or "")
    if reviewer is None or (
        reviewer.get("role") != "independent_control_plane_reviewer"
        or reviewer_did not in frozenset(trusted_reviewer_dids)
        or not _DID.fullmatch(reviewer_did)
        or reviewer_did == str((owner or {}).get("owner_principal_did") or "")
    ):
        blockers.append("quack_owner_reviewer_not_independent")
    signature = str(receipt.get("reviewer_signature") or "")
    if not signature or not reviewer_did:
        blockers.append("quack_owner_qualification_unsigned")
    else:
        try:
            verify_did_key_signature(
                identity_did=reviewer_did,
                payload={
                    key: value
                    for key, value in receipt.items()
                    if key not in {"reviewer_signature", "receipt_cid"}
                },
                signature=signature,
            )
        except (LocalProfileTampered, ValueError):
            blockers.append("quack_owner_qualification_signature_invalid")

    blockers = list(dict.fromkeys(blockers))
    report: dict[str, Any] = {
        "schema": EAAEF_QUACK_OWNER_VERIFICATION_SCHEMA,
        "allowed": not blockers,
        "historical_only": True,
        "promotion_allowed": False,
        "blockers": blockers,
        "receipt_cid": claimed_cid,
        "board_cid": str(receipt.get("board_cid") or ""),
        "source_head": str((source or {}).get("commit") or ""),
        "source_tree": str((source or {}).get("tree") or ""),
        "materialization_receipt_cid": str(
            (materialization or {}).get("receipt_cid") or ""
        ),
        "shard_id": str((owner or {}).get("shard_id") or ""),
        "store_id": str((owner or {}).get("store_id") or ""),
        "owner_principal_did": str((owner or {}).get("owner_principal_did") or ""),
        "owner_generation": int((owner or {}).get("owner_generation") or 0),
        "epoch": int((owner or {}).get("epoch") or 0),
        "fence": int((owner or {}).get("fence") or 0),
        "expires_at_ms": int((qualification or {}).get("expires_at_ms") or 0),
        "authority_mutated": False,
        "process_started": False,
    }
    report["decision_cid"] = _cid(report)
    return report


def _v2_rejection_report(receipt: Mapping[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": EAAEF_CONTROL_PLANE_PROMOTION_VERIFICATION_SCHEMA_V2,
        "allowed": False,
        "historical_only": False,
        "promotion_allowed": False,
        "blockers": ["quack_control_plane_promotion_v2_required"],
        "receipt_cid": str(receipt.get("receipt_cid") or ""),
        "board_cid": str(receipt.get("board_cid") or ""),
        "source_head": "",
        "source_tree": "",
        "materialization_receipt_cid": "",
        "base_owner_qualification_cid": "",
        "bootstrap_admission_cid": "",
        "shard_id": "",
        "store_id": "",
        "owner_principal_did": "",
        "owner_generation": 0,
        "epoch": 0,
        "fence": 0,
        "dispatcher_interface": "",
        "command_fabric_interface": "",
        "operation_vocabulary_cid": "",
        "plan_r2_operational_capability_cid": "",
        "command_fabric_qualification_cid": "",
        "authorization_policy_cid": "",
        "qualification_reviewer_did": "",
        "operator_identity_did": "",
        "security_reviewer_identity_did": "",
        "expires_at_ms": 0,
        "authority_mutated": False,
        "process_started": False,
    }
    report["decision_cid"] = _cid(report)
    return report


def verify_external_agent_control_plane_promotion(
    *,
    qualification_receipt: object,
    board: Mapping[str, Any],
    materialization_receipt: Mapping[str, Any],
    expected_source_commit: str,
    expected_source_tree: str,
    trusted_reviewer_dids: Sequence[str],
    now_ms: int,
    trusted_operator_dids: Sequence[str] = (),
    trusted_security_reviewer_dids: Sequence[str] = (),
) -> dict[str, Any]:
    """Verify the exact v2 Plan-R2 dispatcher promotion, read-only.

    A v1 owner receipt deliberately returns a typed no-go here.  Call
    :func:`verify_external_agent_quack_owner_qualification_v1` only when the
    caller needs the historical base-owner decision for the pre-Promotion R1
    bootstrap phase.
    """

    receipt = (
        dict(qualification_receipt)
        if isinstance(qualification_receipt, Mapping)
        else {}
    )
    if receipt.get("schema") != EAAEF_CONTROL_PLANE_PROMOTION_SCHEMA_V2:
        return _v2_rejection_report(receipt)

    blockers: list[str] = []
    source = _closed(receipt.get("source"), _SOURCE_FIELDS)
    materialization = _closed(receipt.get("materialization"), _MATERIALIZATION_FIELDS)
    profile = _closed(receipt.get("profile"), _V2_PROFILE_FIELDS)
    owner = _closed(receipt.get("owner"), _OWNER_FIELDS)
    transport = _closed(receipt.get("transport"), _TRANSPORT_FIELDS)
    dispatcher = _closed(receipt.get("dispatcher"), _DISPATCHER_FIELDS)
    policy = _closed(receipt.get("authorization_policy"), _AUTHORIZATION_POLICY_FIELDS)
    atomicity = _closed(receipt.get("atomicity"), _ATOMICITY_FIELDS)
    evidence = _closed(receipt.get("evidence"), _EVIDENCE_FIELDS)
    qualification = _closed(receipt.get("qualification"), _V2_QUALIFICATION_FIELDS)
    reviewer = _closed(receipt.get("reviewer"), _REVIEWER_FIELDS)
    operator = _closed(receipt.get("operator"), _REVIEWER_FIELDS)
    security = _closed(receipt.get("security_reviewer"), _REVIEWER_FIELDS)
    capability = receipt.get("plan_r2_operational_capability")
    blockers.extend(
        _base_identity_checks(
            receipt=receipt,
            source=source,
            materialization=materialization,
            profile=profile,
            owner=owner,
            transport=transport,
            board=board,
            materialization_receipt=materialization_receipt,
            expected_source_commit=expected_source_commit,
            expected_source_tree=expected_source_tree,
            profile_fields=_V2_PROFILE_FIELDS,
        )[0]
    )
    if set(receipt) != _V2_RECEIPT_FIELDS:
        blockers.append("quack_control_plane_promotion_shape_invalid")
    one_use_nonce = str(receipt.get("one_use_nonce") or "")
    if not _SAFE_ID.fullmatch(one_use_nonce):
        blockers.append("quack_control_plane_promotion_nonce_invalid")
    claimed_cid = str(receipt.get("receipt_cid") or "")
    cid_body = dict(receipt)
    cid_body.pop("receipt_cid", None)
    if not _SHA256.fullmatch(claimed_cid) or claimed_cid != _cid(cid_body):
        blockers.append("quack_control_plane_promotion_self_address_invalid")

    vocabulary = (dispatcher or {}).get("operation_vocabulary")
    if (
        dispatcher is None
        or dispatcher.get("dispatcher_interface") != PLAN_R2_OWNER_GATEWAY_INTERFACE
        or dispatcher.get("command_fabric_interface") != QUACK_COMMAND_FABRIC_INTERFACE
        or dispatcher.get("operation_schema") != PLAN_R2_OWNER_OPERATION_SCHEMA
        or dispatcher.get("authorized_state_command_interface")
        != AUTHORIZED_STATE_COMMAND_INTERFACE
        or dispatcher.get("state_command_interface") != STATE_COMMAND_INTERFACE
        or dispatcher.get("generic_daemon_gateway_admitted") is not False
        or not isinstance(vocabulary, list)
        or vocabulary != exact_plan_r2_operation_vocabulary()
        or any(
            not isinstance(item, Mapping) or set(item) != _OPERATION_FIELDS
            for item in (vocabulary if isinstance(vocabulary, list) else ())
        )
        or dispatcher.get("operation_vocabulary_cid")
        != plan_r2_operation_vocabulary_cid()
    ):
        blockers.append("quack_plan_r2_operation_vocabulary_invalid")

    policy_cid = ""
    if policy is not None:
        policy_body = dict(policy)
        policy_cid = str(policy_body.pop("policy_cid", ""))
    owner_did = str((owner or {}).get("owner_principal_did") or "")
    reviewer_did = str((reviewer or {}).get("identity_did") or "")
    operator_did = str((operator or {}).get("identity_did") or "")
    security_did = str((security or {}).get("identity_did") or "")
    trusted_reviewers = frozenset(str(item) for item in trusted_reviewer_dids)
    trusted_operators = frozenset(str(item) for item in trusted_operator_dids)
    trusted_security = frozenset(
        str(item) for item in trusted_security_reviewer_dids
    )
    privileged = {owner_did, reviewer_did, operator_did, security_did}
    approvers = (
        policy.get("trusted_approver_dids") if policy is not None else None
    )
    principals = (
        policy.get("authorized_principal_dids") if policy is not None else None
    )
    if (
        policy is None
        or policy.get("schema") != QUACK_COMMAND_AUTHORIZATION_POLICY_SCHEMA
        or policy_cid != _cid(policy_body if policy is not None else {})
        or policy.get("board_namespace") != EAAEF_BOARD_NAMESPACE
        or policy.get("shard_id") != (owner or {}).get("shard_id")
        or policy.get("store_id") != (owner or {}).get("store_id")
        or policy.get("shard_id") == policy.get("store_id")
        or policy.get("owner_principal_did") != owner_did
        or policy.get("owner_generation") != (owner or {}).get("owner_generation")
        or policy.get("fence_epoch") != (owner or {}).get("fence")
        or not _SHA256.fullmatch(str(policy.get("authority_ref_cid") or ""))
        or policy.get("allowed_command_kinds") != ["migrate", "observe"]
        or not _positive_int(policy.get("maximum_authorization_lifetime_ms"))
        or int(policy.get("maximum_authorization_lifetime_ms") or 0) > 300_000
        or not isinstance(approvers, list)
        or not isinstance(principals, list)
        or not approvers
        or not principals
        or approvers != sorted(set(approvers))
        or principals != sorted(set(principals))
        or any(not _DID.fullmatch(str(item)) for item in [*approvers, *principals])
        or bool(set(approvers).intersection(principals))
        or bool(privileged.intersection({*approvers, *principals}))
        or (dispatcher or {}).get("authorization_policy_cid") != policy_cid
    ):
        blockers.append("quack_command_authorization_policy_invalid")

    if atomicity is None or dict(atomicity) != _EXPECTED_ATOMICITY:
        blockers.append("quack_plan_r2_atomic_guarantees_invalid")
    if evidence is None or any(
        not _SHA256.fullmatch(str(evidence.get(field) or ""))
        for field in _EVIDENCE_FIELDS
    ) or (evidence is not None and len(set(evidence.values())) != len(evidence)):
        blockers.append("quack_plan_r2_qualification_evidence_invalid")

    role_identities = (reviewer_did, operator_did, security_did)
    if (
        reviewer is None
        or operator is None
        or security is None
        or reviewer.get("role") != "independent_control_plane_reviewer"
        or operator.get("role") != "independent_operator"
        or security.get("role") != "independent_security_reviewer"
        or reviewer_did not in trusted_reviewers
        or operator_did not in trusted_operators
        or security_did not in trusted_security
        or any(not _DID.fullmatch(item) for item in role_identities)
        or len(set(role_identities)) != 3
        or owner_did in set(role_identities)
        or bool(trusted_reviewers.intersection(trusted_operators))
        or bool(trusted_reviewers.intersection(trusted_security))
        or bool(trusted_operators.intersection(trusted_security))
    ):
        blockers.append("quack_control_plane_reviewers_not_independent")

    verified_capability: Mapping[str, Any] = {}
    try:
        from ipfs_accelerate_py.agent_supervisor.planning.external_agent_plan_r2 import (
            ExternalAgentPlanR2Error,
            verify_plan_r2_operational_capability,
        )

        verified_capability = verify_plan_r2_operational_capability(
            capability,
            trusted_reviewer_dids=[reviewer_did],
            now_ms=now_ms,
        )
    except (ExternalAgentPlanR2Error, ValueError, TypeError):
        blockers.append("quack_plan_r2_operational_capability_invalid")
    capability_cid = str(verified_capability.get("capability_cid") or "")
    command_fabric_cid = str(
        verified_capability.get("quack_command_fabric_qualification_cid") or ""
    )
    base_owner_cid = str(
        verified_capability.get("quack_owner_qualification_cid") or ""
    )
    bootstrap_admission_cid = str(
        verified_capability.get("bootstrap_admission_cid") or ""
    )
    capability_exact = {
        "source_head": str((source or {}).get("commit") or ""),
        "source_tree": str((source or {}).get("tree") or ""),
        "owner_principal_did": owner_did,
        "shard_id": str((owner or {}).get("shard_id") or ""),
        "owner_generation": int((owner or {}).get("owner_generation") or 0),
        "epoch": int((owner or {}).get("epoch") or 0),
        "fence": int((owner or {}).get("fence") or 0),
        "duckdb_version": REQUIRED_DUCKDB_VERSION,
        "quack_build": REQUIRED_QUACK_BUILD,
    }
    if (
        not verified_capability
        or any(verified_capability.get(key) != value for key, value in capability_exact.items())
        or verified_capability.get("schema") != PLAN_R2_OPERATIONAL_CAPABILITY_SCHEMA
        or verified_capability.get("reviewer_identity_did") != reviewer_did
        or (dispatcher or {}).get("plan_r2_operational_capability_cid")
        != capability_cid
        or (dispatcher or {}).get("command_fabric_qualification_cid")
        != command_fabric_cid
        or any(
            not _SHA256.fullmatch(value)
            for value in (capability_cid, command_fabric_cid, base_owner_cid, bootstrap_admission_cid)
        )
    ):
        blockers.append("quack_plan_r2_capability_identity_mismatch")

    if qualification is None or (
        qualification.get("status") != "accepted"
        or not _positive_int(qualification.get("qualified_at_ms"))
        or not _positive_int(qualification.get("expires_at_ms"))
        or not _positive_int(now_ms)
        or int(qualification.get("qualified_at_ms") or 0) > now_ms
        or now_ms >= int(qualification.get("expires_at_ms") or 0)
        or int(qualification.get("expires_at_ms") or 0)
        > int(verified_capability.get("expires_at_ms") or 0)
    ):
        blockers.append("quack_control_plane_promotion_not_current")

    # Promotion is post-bootstrap.  None of the precursor identities may point
    # back to the promotion receipt; this makes the signing order acyclic.
    precursor_cids = {
        base_owner_cid,
        bootstrap_admission_cid,
        capability_cid,
        command_fabric_cid,
        policy_cid,
        str((dispatcher or {}).get("operation_vocabulary_cid") or ""),
    }
    if claimed_cid in precursor_cids or len({item for item in precursor_cids if item}) != len(
        [item for item in precursor_cids if item]
    ):
        blockers.append("quack_control_plane_promotion_cid_cycle_or_alias")

    signing_payload = {
        key: value
        for key, value in receipt.items()
        if key
        not in {
            "reviewer_signature",
            "operator_signature",
            "security_reviewer_signature",
            "receipt_cid",
        }
    }
    for role, identity_did, signature_field in (
        ("reviewer", reviewer_did, "reviewer_signature"),
        ("operator", operator_did, "operator_signature"),
        ("security_reviewer", security_did, "security_reviewer_signature"),
    ):
        signature = str(receipt.get(signature_field) or "")
        if not signature or not identity_did:
            blockers.append(f"quack_control_plane_promotion_{role}_unsigned")
            continue
        try:
            verify_did_key_signature(
                identity_did=identity_did,
                payload=signing_payload,
                signature=signature,
            )
        except (LocalProfileTampered, ValueError):
            blockers.append(
                f"quack_control_plane_promotion_{role}_signature_invalid"
            )

    blockers = list(dict.fromkeys(blockers))
    report: dict[str, Any] = {
        "schema": EAAEF_CONTROL_PLANE_PROMOTION_VERIFICATION_SCHEMA_V2,
        "allowed": not blockers,
        "historical_only": False,
        "promotion_allowed": not blockers,
        "blockers": blockers,
        "receipt_cid": claimed_cid,
        "board_cid": str(receipt.get("board_cid") or ""),
        "source_head": str((source or {}).get("commit") or ""),
        "source_tree": str((source or {}).get("tree") or ""),
        "materialization_receipt_cid": str(
            (materialization or {}).get("receipt_cid") or ""
        ),
        "base_owner_qualification_cid": base_owner_cid,
        "bootstrap_admission_cid": bootstrap_admission_cid,
        "shard_id": str((owner or {}).get("shard_id") or ""),
        "store_id": str((owner or {}).get("store_id") or ""),
        "owner_principal_did": owner_did,
        "owner_generation": int((owner or {}).get("owner_generation") or 0),
        "epoch": int((owner or {}).get("epoch") or 0),
        "fence": int((owner or {}).get("fence") or 0),
        "dispatcher_interface": str(
            (dispatcher or {}).get("dispatcher_interface") or ""
        ),
        "command_fabric_interface": str(
            (dispatcher or {}).get("command_fabric_interface") or ""
        ),
        "operation_vocabulary_cid": str(
            (dispatcher or {}).get("operation_vocabulary_cid") or ""
        ),
        "plan_r2_operational_capability_cid": capability_cid,
        "command_fabric_qualification_cid": command_fabric_cid,
        "authorization_policy_cid": policy_cid,
        "qualification_reviewer_did": reviewer_did,
        "operator_identity_did": operator_did,
        "security_reviewer_identity_did": security_did,
        "expires_at_ms": int((qualification or {}).get("expires_at_ms") or 0),
        "authority_mutated": False,
        "process_started": False,
    }
    report["decision_cid"] = _cid(report)
    return report


__all__ = [
    "AUTHORIZED_STATE_COMMAND_INTERFACE",
    "EAAEF_CONTROL_PLANE_PROMOTION_SCHEMA_V2",
    "EAAEF_CONTROL_PLANE_PROMOTION_VERIFICATION_SCHEMA_V2",
    "EAAEF_PLAN_R2_OPERATION_VOCABULARY_SCHEMA",
    "EAAEF_QUACK_OWNER_QUALIFICATION_SCHEMA",
    "EAAEF_QUACK_OWNER_VERIFICATION_SCHEMA",
    "ExternalAgentControlPlanePromotionError",
    "PLAN_R2_OPERATIONAL_CAPABILITY_SCHEMA",
    "PLAN_R2_OWNER_GATEWAY_INTERFACE",
    "QUACK_COMMAND_FABRIC_INTERFACE",
    "exact_plan_r2_operation_vocabulary",
    "external_agent_control_plane_promotion_signing_bytes",
    "plan_r2_operation_vocabulary_cid",
    "verify_external_agent_control_plane_promotion",
    "verify_external_agent_quack_owner_qualification_v1",
]
