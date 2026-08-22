"""Host-only, create-once admission for the EAAEF bootstrap.

The provider/container and Quack receipts are necessary evidence, but neither
is authority to start a supervisor.  This module joins their independent
read-only verifications with two effect-bound human approvals.  Preparing and
verifying are pure.  Publication is an explicit, create-once host effect.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    LocalProfileTampered,
    verify_did_key_signature,
)

EAAEF_BOOTSTRAP_ADMISSION_STATEMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-bootstrap-admission-statement@1"
)
EAAEF_BOOTSTRAP_APPROVAL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-approval@1"
)
EAAEF_BOOTSTRAP_ADMISSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-admission@1"
)
EAAEF_BOARD_NAMESPACE: Final = "external-agent-autonomous-execution-fabric-v1"
EAAEF_TASK_ID: Final = "EAAEF-000"
EAAEF_MAXIMUM_LANES: Final = 5
EAAEF_AUTHORITY_REGISTRY_PREFIX: Final = (
    "data/agent_supervisor/external_agent_autonomous_execution_fabric/authority"
)
EAAEF_MATERIALIZATION_RECEIPT_SCHEMA_V2: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-autonomous-execution-fabric-materialization@2"
)
EAAEF_DATABASE_PROGRAM_BINDINGS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-database-program-bindings@1"
)
EAAEF_SIGNED_COMMAND_FABRIC_PROFILE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-signed-command-fabric-profile@2"
)
EAAEF_OPERATIONAL_COMMAND_FABRIC_SHARD_ID: Final = "control-shard-0"

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}")
_APPROVAL_ROLES: Final = (
    "independent_operator",
    "independent_security_reviewer",
)
_STATEMENT_FIELDS: Final = frozenset(
    {
        "schema",
        "task_id",
        "board_namespace",
        "decision",
        "outcome",
        "blockers",
        "board_cid",
        "source_head",
        "source_tree",
        "source_generation_cid",
        "materialization_receipt_cid",
        "materialization_store_generation",
        "materialization_database_program_binding_cid",
        "materialization_bootstrap_profile_cid",
        "materialization_operational_profile_cid",
        "population_cid",
        "plan_root_cid",
        "control_projection_root",
        "coordination_projection_root",
        "execution_projection_root",
        "provider_container_qualification_cid",
        "provider_container_verification_cid",
        "provider_qualification_signer_did",
        "image_qualification_reviewer_did",
        "provider_qualification_expires_at_ms",
        "provider_maximum_parallel_workers",
        "provider_maximum_parallel_containers",
        "provider_worker_principal_did",
        "provider_principal_did",
        "provider_task_dispatch_admitted",
        "provider_workload_class",
        "quack_owner_qualification_cid",
        "quack_owner_verification_cid",
        "quack_qualification_reviewer_did",
        "quack_qualification_expires_at_ms",
        "quack_owner_principal_did",
        "container_profile_cid",
        "image_digest",
        "quack_shard_id",
        "quack_epoch",
        "quack_fence",
        "authority",
        "one_use_nonce",
        "issued_at_ms",
        "expires_at_ms",
        "statement_cid",
    }
)
_EXPECTED_AUTHORITY: Final = {
    "launch_mode": "configured_board_multi_supervisor",
    "maximum_lanes": EAAEF_MAXIMUM_LANES,
    "actual_lanes_bounded_by_qualified_resources": True,
    "mutable_coordination_authority": "one_fenced_quack_owner",
    "direct_duckdb_file_open": False,
    "ducklake_current_authority": False,
    "automatic_protected_branch_merge": False,
}
_APPROVAL_FIELDS: Final = frozenset(
    {
        "schema",
        "role",
        "identity_did",
        "statement_cid",
        "one_use_nonce",
        "issued_at_ms",
        "expires_at_ms",
        "signature",
    }
)
_RECEIPT_FIELDS: Final = frozenset(
    {*_STATEMENT_FIELDS, "operator_approval", "security_approval", "receipt_cid"}
)
_DATABASE_PROGRAM_BINDING_FIELDS: Final = frozenset(
    {
        "schema",
        "bootstrap",
        "bootstrap_source_cid",
        "bootstrap_profile_cid",
        "operational",
        "operational_source_cid",
        "operational_database_program_profile_cid",
        "operational_command_fabric",
        "operational_profile_cid",
        "operational_child_adapter_status",
        "materializer_opens_operational_profile",
        "direct_file_fallback",
        "binding_cid",
    }
)
_SIGNED_COMMAND_FABRIC_PROFILE_FIELDS: Final = frozenset(
    {
        "schema",
        "transport_kind",
        "board_namespace",
        "shard_id",
        "ingress_endpoint",
        "ingress_secret_handle",
        "projection_endpoint",
        "projection_secret_handle",
        "store_id",
        "store_generation",
        "schema_revision",
        "owner_qualification_schema",
        "command_envelope_schema",
        "state_command_schema",
        "ingress_relation",
        "ingress_append_only",
        "ingress_accepts_signed_envelopes_only",
        "operational_database_private",
        "operational_tables_remotely_exposed",
        "one_mutable_owner",
        "owner_verifies_signed_envelopes",
        "projection_read_only",
        "projection_append_allowed",
        "atomic_plan_r2_required",
        "direct_file_fallback",
        "failover_policy",
        "child_adapter_status",
    }
)


class ExternalAgentBootstrapAdmissionError(RuntimeError):
    """Fail-closed final bootstrap admission error."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _cid(value: Any) -> str:
    raw = value if isinstance(value, bytes) else _canonical_bytes(value)
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _positive_int(value: object) -> bool:
    return type(value) is int and int(value) > 0


def _decision_payload(value: object) -> Mapping[str, Any]:
    if hasattr(value, "as_dict"):
        payload = value.as_dict()
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise ExternalAgentBootstrapAdmissionError(
            "qualification verifier returned no canonical decision"
        )
    if not isinstance(payload, Mapping):
        raise ExternalAgentBootstrapAdmissionError(
            "qualification verifier returned no canonical decision"
        )
    return payload


def _materialization_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    source = value.get("source_generation")
    board_validation = value.get("board_validation")
    control = value.get("control_projection")
    coordination = value.get("coordination_projection")
    execution = value.get("execution_projection")
    bindings = value.get("database_program_bindings")
    body = dict(value)
    receipt_cid = str(body.pop("receipt_cid", ""))
    if (
        value.get("schema") != EAAEF_MATERIALIZATION_RECEIPT_SCHEMA_V2
        or not _SHA256.fullmatch(receipt_cid)
        or receipt_cid != _cid(body)
        or not isinstance(source, Mapping)
        or not isinstance(board_validation, Mapping)
        or not isinstance(control, Mapping)
        or not isinstance(coordination, Mapping)
        or not isinstance(execution, Mapping)
        or not isinstance(bindings, Mapping)
    ):
        raise ExternalAgentBootstrapAdmissionError(
            "materialization_receipt_missing_or_invalid"
        )
    binding_body = dict(bindings)
    binding_cid = str(binding_body.pop("binding_cid", ""))
    bootstrap_program = bindings.get("bootstrap")
    operational_program = bindings.get("operational")
    command_fabric = bindings.get("operational_command_fabric")
    if (
        set(bindings) != _DATABASE_PROGRAM_BINDING_FIELDS
        or bindings.get("schema") != EAAEF_DATABASE_PROGRAM_BINDINGS_SCHEMA
        or binding_cid != _cid(binding_body)
        or not isinstance(bootstrap_program, Mapping)
        or not isinstance(operational_program, Mapping)
        or not isinstance(command_fabric, Mapping)
        or set(command_fabric) != _SIGNED_COMMAND_FABRIC_PROFILE_FIELDS
        or bindings.get("materializer_opens_operational_profile") is not False
        or bindings.get("direct_file_fallback") is not False
        or bootstrap_program.get("authority_mode") != "embedded"
        or bootstrap_program.get("task_source_kind") != "duckdb"
        or operational_program.get("authority_mode") != "quack"
        or operational_program.get("task_source_kind") != "duckdb"
        or operational_program.get("failover_policy") != "fail_closed"
        or not str(operational_program.get("quack_endpoint") or "")
        or not str(operational_program.get("endpoint_secret_handle") or "")
        or "/" in str(operational_program.get("store_id") or "")
        or "\\" in str(operational_program.get("store_id") or "")
        or command_fabric.get("schema")
        != EAAEF_SIGNED_COMMAND_FABRIC_PROFILE_SCHEMA
        or command_fabric.get("transport_kind") != "signed_command_fabric"
        or command_fabric.get("board_namespace") != EAAEF_BOARD_NAMESPACE
        or command_fabric.get("board_namespace")
        != board_validation.get("board_namespace")
        or command_fabric.get("shard_id")
        != EAAEF_OPERATIONAL_COMMAND_FABRIC_SHARD_ID
        or command_fabric.get("store_id") != operational_program.get("store_id")
        or command_fabric.get("store_generation")
        != operational_program.get("store_generation")
        or command_fabric.get("schema_revision")
        != operational_program.get("schema_revision")
        or command_fabric.get("ingress_endpoint")
        == command_fabric.get("projection_endpoint")
        or command_fabric.get("ingress_append_only") is not True
        or command_fabric.get("ingress_accepts_signed_envelopes_only") is not True
        or command_fabric.get("operational_database_private") is not True
        or command_fabric.get("operational_tables_remotely_exposed") is not False
        or command_fabric.get("one_mutable_owner") is not True
        or command_fabric.get("owner_verifies_signed_envelopes") is not True
        or command_fabric.get("projection_read_only") is not True
        or command_fabric.get("projection_append_allowed") is not False
        or command_fabric.get("atomic_plan_r2_required") is not True
        or command_fabric.get("direct_file_fallback") is not False
        or command_fabric.get("child_adapter_status")
        not in {
            "implemented_unqualified_fail_closed",
            "admitted",
        }
        or bindings.get("operational_child_adapter_status")
        not in {
            "implemented_unqualified_fail_closed",
            "admitted",
        }
        or command_fabric.get("child_adapter_status")
        != bindings.get("operational_child_adapter_status")
    ):
        raise ExternalAgentBootstrapAdmissionError(
            "materialization_database_program_binding_invalid"
        )
    store_generation = str(bootstrap_program.get("store_generation") or "")
    if (
        not store_generation
        or store_generation != operational_program.get("store_generation")
        or bindings.get("bootstrap_profile_cid") != _cid(dict(bootstrap_program))
        or bindings.get("operational_database_program_profile_cid")
        != _cid(dict(operational_program))
        or bindings.get("operational_profile_cid")
        != _cid(dict(command_fabric))
    ):
        raise ExternalAgentBootstrapAdmissionError(
            "materialization_database_program_binding_invalid"
        )
    projection = {
        "source_head": str(value.get("source_head") or ""),
        "source_tree": str(value.get("source_tree") or ""),
        "source_generation_cid": str(source.get("source_generation_cid") or ""),
        "materialization_receipt_cid": receipt_cid,
        "materialization_store_generation": store_generation,
        "materialization_database_program_binding_cid": binding_cid,
        "materialization_bootstrap_profile_cid": str(
            bindings.get("bootstrap_profile_cid") or ""
        ),
        "materialization_operational_profile_cid": str(
            bindings.get("operational_profile_cid") or ""
        ),
        "population_cid": str(value.get("population_cid") or ""),
        "plan_root_cid": str(value.get("plan_root_cid") or ""),
        "board_cid": str(board_validation.get("board_cid") or ""),
        "control_projection_root": str(control.get("projection_root") or ""),
        "coordination_projection_root": str(
            coordination.get("projection_root") or ""
        ),
        "execution_projection_root": str(execution.get("projection_root") or ""),
    }
    if (
        not _GIT_OBJECT.fullmatch(projection["source_head"])
        or not _GIT_OBJECT.fullmatch(projection["source_tree"])
        or any(
            not _SHA256.fullmatch(projection[name])
            for name in projection
            if name
            not in {"source_head", "source_tree", "materialization_store_generation"}
        )
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", store_generation)
    ):
        raise ExternalAgentBootstrapAdmissionError(
            "materialization_receipt_missing_or_invalid"
        )
    return projection


def prepare_external_agent_bootstrap_admission(
    *,
    board: Mapping[str, Any],
    materialization_receipt: Mapping[str, Any],
    provider_container_qualification: object,
    route_plan: object,
    image_qualification: object,
    container_profile: object,
    quack_owner_qualification: object,
    trusted_provider_signer_dids: Sequence[str],
    trusted_image_reviewer_dids: Sequence[str],
    trusted_container_profile_reviewer_dids: Sequence[str],
    trusted_quack_reviewer_dids: Sequence[str],
    expected_worker_principal_did: str,
    expected_provider_principal_did: str,
    expected_source_commit: str,
    expected_source_tree: str,
    one_use_nonce: str,
    issued_at_ms: int,
    expires_at_ms: int,
) -> dict[str, Any]:
    """Prepare an unsigned admission/no-go statement without authority effects."""

    if (
        not isinstance(board, Mapping)
        or board.get("board_namespace") != EAAEF_BOARD_NAMESPACE
        or not _SHA256.fullmatch(str(board.get("board_cid") or ""))
    ):
        raise ExternalAgentBootstrapAdmissionError("board_missing_or_invalid")
    materialization = _materialization_projection(materialization_receipt)
    if (
        materialization["board_cid"] != board.get("board_cid")
        or materialization["source_head"] != expected_source_commit
        or materialization["source_tree"] != expected_source_tree
    ):
        raise ExternalAgentBootstrapAdmissionError(
            "materialization_source_or_board_mismatch"
        )
    if (
        not isinstance(one_use_nonce, str)
        or not one_use_nonce
        or len(one_use_nonce.encode("utf-8")) > 512
        or not _positive_int(issued_at_ms)
        or not _positive_int(expires_at_ms)
        or issued_at_ms >= expires_at_ms
    ):
        raise ExternalAgentBootstrapAdmissionError("admission_time_or_nonce_invalid")
    principal_blockers: list[str] = []
    if (
        not expected_worker_principal_did.startswith("did:key:z")
        or not expected_provider_principal_did.startswith("did:key:z")
        or expected_worker_principal_did == expected_provider_principal_did
    ):
        principal_blockers.append("worker_network_runtime_principals_unavailable")

    # Lazy imports keep statement preparation free of provider/Quack side
    # effects and make this module consume, rather than reimplement, those
    # independently owned verifiers.
    from ipfs_accelerate_py.agent_supervisor.runtime.external_agent_control_plane_promotion import (
        verify_external_agent_quack_owner_qualification_v1,
    )
    from ipfs_accelerate_py.agent_supervisor.validation.external_agent_fabric_bootstrap import (
        verify_eaaef_provider_container_qualification,
    )

    provider = _decision_payload(
        verify_eaaef_provider_container_qualification(
            qualification=provider_container_qualification,
            board=board,
            materialization_receipt=materialization_receipt,
            route_plan=route_plan,
            image_qualification=image_qualification,
            container_profile=container_profile,
            trusted_qualification_signer_dids=trusted_provider_signer_dids,
            trusted_image_reviewer_dids=trusted_image_reviewer_dids,
            trusted_container_profile_reviewer_dids=(
                trusted_container_profile_reviewer_dids
            ),
            expected_worker_principal_did=expected_worker_principal_did,
            expected_provider_principal_did=expected_provider_principal_did,
            now_ms=issued_at_ms,
        )
    )
    quack = _decision_payload(
        verify_external_agent_quack_owner_qualification_v1(
            qualification_receipt=quack_owner_qualification,
            board=board,
            materialization_receipt=materialization_receipt,
            expected_source_commit=expected_source_commit,
            expected_source_tree=expected_source_tree,
            trusted_reviewer_dids=trusted_quack_reviewer_dids,
            now_ms=issued_at_ms,
        )
    )
    provider_valid = provider.get("valid") is True
    quack_valid = quack.get("allowed") is True
    blockers = list(principal_blockers)
    blockers.extend(str(item) for item in provider.get("blockers") or () if str(item))
    blockers.extend(str(item) for item in quack.get("blockers") or () if str(item))
    if not provider_valid and not blockers:
        blockers.append("provider_container_qualification_not_admitted")
    if not quack_valid and not blockers:
        blockers.append("quack_owner_qualification_not_admitted")
    blockers = list(dict.fromkeys(blockers))
    provider_task_dispatch_admitted = (
        provider.get("task_dispatch_admitted") is True
    )
    provider_workload_class = str(provider.get("workload_class") or "")
    if provider_valid and not provider_task_dispatch_admitted:
        blockers.append("provider_task_dispatch_not_admitted")
    blockers = list(dict.fromkeys(blockers))
    provider_cid = str(provider.get("qualification_cid") or "")
    provider_verification_cid = str(provider.get("verifier_cid") or "")
    quack_cid = str(quack.get("receipt_cid") or "")
    quack_verification_cid = str(quack.get("decision_cid") or "")
    profile_cid = str(provider.get("container_profile_cid") or "")
    provider_maximum_parallel_workers = provider.get("maximum_parallel_workers")
    provider_maximum_parallel_containers = provider.get(
        "maximum_parallel_containers"
    )
    provider_worker_principal_did = str(
        provider.get("worker_principal_did") or ""
    )
    provider_principal_did = str(provider.get("provider_principal_did") or "")
    quack_owner_principal_did = str(quack.get("owner_principal_did") or "")
    image_digest = ""
    image_reviewer_did = ""
    if isinstance(image_qualification, Mapping):
        image_digest = str(image_qualification.get("image_digest") or "")
        image_reviewer_did = str(
            image_qualification.get("reviewer_identity_did") or ""
        )
    provider_signer_did = ""
    provider_expires_at_ms = 0
    if isinstance(provider_container_qualification, Mapping):
        provider_signer_did = str(
            provider_container_qualification.get("signer_identity_did") or ""
        )
        raw_provider_expiry = provider_container_qualification.get("expires_at_ms")
        if _positive_int(raw_provider_expiry):
            provider_expires_at_ms = int(raw_provider_expiry)
    quack_reviewer_did = ""
    quack_expires_at_ms = 0
    if isinstance(quack_owner_qualification, Mapping):
        raw_reviewer = quack_owner_qualification.get("reviewer")
        raw_qualification = quack_owner_qualification.get("qualification")
        if isinstance(raw_reviewer, Mapping):
            quack_reviewer_did = str(raw_reviewer.get("identity_did") or "")
        if isinstance(raw_qualification, Mapping) and _positive_int(
            raw_qualification.get("expires_at_ms")
        ):
            quack_expires_at_ms = int(raw_qualification["expires_at_ms"])
    if provider_valid and provider_task_dispatch_admitted and (
        not provider_worker_principal_did.startswith("did:key:z")
        or not provider_principal_did.startswith("did:key:z")
        or provider_worker_principal_did == provider_principal_did
        or provider_worker_principal_did != expected_worker_principal_did
        or provider_principal_did != expected_provider_principal_did
    ):
        blockers.append("worker_network_runtime_principals_unavailable")
    blockers = list(dict.fromkeys(blockers))
    accepted = provider_valid and quack_valid and not blockers
    for field, identity in (
        ("provider qualification", provider_cid),
        ("provider verification", provider_verification_cid),
        ("Quack qualification", quack_cid),
        ("Quack verification", quack_verification_cid),
        ("container profile", profile_cid),
        ("image", image_digest),
    ):
        if accepted and not _SHA256.fullmatch(identity):
            raise ExternalAgentBootstrapAdmissionError(
                f"accepted {field} identity is missing or invalid"
            )
    evidence_reviewers = (
        provider_signer_did,
        image_reviewer_did,
        quack_reviewer_did,
    )
    if accepted and (
        any(not item.startswith("did:key:z") for item in evidence_reviewers)
        or not _positive_int(provider_expires_at_ms)
        or not _positive_int(quack_expires_at_ms)
        or expires_at_ms > min(provider_expires_at_ms, quack_expires_at_ms)
        or not _positive_int(provider_maximum_parallel_workers)
        or not _positive_int(provider_maximum_parallel_containers)
        or int(provider_maximum_parallel_workers)
        > int(provider_maximum_parallel_containers)
        or int(provider_maximum_parallel_containers) > EAAEF_MAXIMUM_LANES
        or not provider_worker_principal_did.startswith("did:key:z")
        or not provider_principal_did.startswith("did:key:z")
        or not quack_owner_principal_did.startswith("did:key:z")
        or len(
            {
                provider_worker_principal_did,
                provider_principal_did,
                quack_owner_principal_did,
            }
        )
        != 3
    ):
        raise ExternalAgentBootstrapAdmissionError(
            "accepted evidence reviewers or lifetime are not independent and bounded"
        )

    statement: dict[str, Any] = {
        "schema": EAAEF_BOOTSTRAP_ADMISSION_STATEMENT_SCHEMA,
        "task_id": EAAEF_TASK_ID,
        "board_namespace": EAAEF_BOARD_NAMESPACE,
        "decision": "admitted" if accepted else "no_go",
        "outcome": "accepted" if accepted else "mutation_not_admitted",
        "blockers": blockers,
        **materialization,
        "provider_container_qualification_cid": provider_cid,
        "provider_container_verification_cid": provider_verification_cid,
        "provider_qualification_signer_did": provider_signer_did,
        "image_qualification_reviewer_did": image_reviewer_did,
        "provider_qualification_expires_at_ms": provider_expires_at_ms,
        "provider_maximum_parallel_workers": (
            int(provider_maximum_parallel_workers)
            if _positive_int(provider_maximum_parallel_workers)
            else 0
        ),
        "provider_maximum_parallel_containers": (
            int(provider_maximum_parallel_containers)
            if _positive_int(provider_maximum_parallel_containers)
            else 0
        ),
        "provider_worker_principal_did": provider_worker_principal_did,
        "provider_principal_did": provider_principal_did,
        "provider_task_dispatch_admitted": provider_task_dispatch_admitted,
        "provider_workload_class": provider_workload_class,
        "quack_owner_qualification_cid": quack_cid,
        "quack_owner_verification_cid": quack_verification_cid,
        "quack_qualification_reviewer_did": quack_reviewer_did,
        "quack_qualification_expires_at_ms": quack_expires_at_ms,
        "quack_owner_principal_did": quack_owner_principal_did,
        "container_profile_cid": profile_cid,
        "image_digest": image_digest,
        "quack_shard_id": str(quack.get("shard_id") or ""),
        "quack_epoch": int(quack.get("epoch") or 0),
        "quack_fence": int(quack.get("fence") or 0),
        "authority": dict(_EXPECTED_AUTHORITY),
        "one_use_nonce": one_use_nonce,
        "issued_at_ms": issued_at_ms,
        "expires_at_ms": expires_at_ms,
    }
    statement["statement_cid"] = _cid(statement)
    return statement


def prepare_external_agent_bootstrap_approval(
    statement: Mapping[str, Any],
    *,
    role: str,
    identity_did: str,
    issued_at_ms: int,
    expires_at_ms: int,
) -> dict[str, Any]:
    """Return the exact payload an external reviewer must sign."""

    _validate_statement_shape(statement)
    if role not in _APPROVAL_ROLES or not identity_did.startswith("did:key:"):
        raise ExternalAgentBootstrapAdmissionError("approval principal is invalid")
    if (
        not _positive_int(issued_at_ms)
        or not _positive_int(expires_at_ms)
        or issued_at_ms >= expires_at_ms
        or issued_at_ms < int(statement["issued_at_ms"])
        or expires_at_ms > int(statement["expires_at_ms"])
    ):
        raise ExternalAgentBootstrapAdmissionError("approval lifetime is invalid")
    return {
        "schema": EAAEF_BOOTSTRAP_APPROVAL_SCHEMA,
        "role": role,
        "identity_did": identity_did,
        "statement_cid": statement["statement_cid"],
        "one_use_nonce": statement["one_use_nonce"],
        "issued_at_ms": issued_at_ms,
        "expires_at_ms": expires_at_ms,
    }


def _validate_statement_shape(statement: Mapping[str, Any]) -> None:
    body = dict(statement)
    statement_cid = str(body.pop("statement_cid", ""))
    if (
        set(statement) != _STATEMENT_FIELDS
        or statement.get("schema") != EAAEF_BOOTSTRAP_ADMISSION_STATEMENT_SCHEMA
        or statement.get("task_id") != EAAEF_TASK_ID
        or statement.get("board_namespace") != EAAEF_BOARD_NAMESPACE
        or statement.get("decision") not in {"admitted", "no_go"}
        or statement.get("outcome") not in {"accepted", "mutation_not_admitted"}
        or not isinstance(statement.get("blockers"), list)
        or any(
            not isinstance(item, str) or not item
            for item in statement.get("blockers", ())
        )
        or statement.get("authority") != _EXPECTED_AUTHORITY
        or not _GIT_OBJECT.fullmatch(str(statement.get("source_head") or ""))
        or not _GIT_OBJECT.fullmatch(str(statement.get("source_tree") or ""))
        or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
            str(statement.get("materialization_store_generation") or ""),
        )
        or any(
            not _SHA256.fullmatch(str(statement.get(field) or ""))
            for field in (
                "materialization_receipt_cid",
                "materialization_database_program_binding_cid",
                "materialization_bootstrap_profile_cid",
                "materialization_operational_profile_cid",
            )
        )
        or not isinstance(statement.get("one_use_nonce"), str)
        or not statement.get("one_use_nonce")
        or len(str(statement.get("one_use_nonce")).encode("utf-8")) > 512
        or not _positive_int(statement.get("issued_at_ms"))
        or not _positive_int(statement.get("expires_at_ms"))
        or int(statement.get("issued_at_ms") or 0)
        >= int(statement.get("expires_at_ms") or 0)
        or statement_cid != _cid(body)
    ):
        raise ExternalAgentBootstrapAdmissionError("admission statement is invalid")
    admitted = statement.get("decision") == "admitted"
    if admitted != (statement.get("outcome") == "accepted") or admitted == bool(
        statement.get("blockers")
    ):
        raise ExternalAgentBootstrapAdmissionError("admission decision is contradictory")
    if admitted:
        cid_fields = _STATEMENT_FIELDS - {
            "schema",
            "task_id",
            "board_namespace",
            "decision",
            "outcome",
            "blockers",
            "source_head",
            "source_tree",
            "materialization_store_generation",
            "provider_qualification_signer_did",
            "image_qualification_reviewer_did",
            "provider_qualification_expires_at_ms",
            "provider_maximum_parallel_workers",
            "provider_maximum_parallel_containers",
            "provider_worker_principal_did",
            "provider_principal_did",
            "provider_task_dispatch_admitted",
            "provider_workload_class",
            "quack_qualification_reviewer_did",
            "quack_qualification_expires_at_ms",
            "quack_owner_principal_did",
            "quack_shard_id",
            "quack_epoch",
            "quack_fence",
            "authority",
            "one_use_nonce",
            "issued_at_ms",
            "expires_at_ms",
        }
        evidence_reviewers = (
            str(statement["provider_qualification_signer_did"]),
            str(statement["image_qualification_reviewer_did"]),
            str(statement["quack_qualification_reviewer_did"]),
        )
        service_principals = (
            str(statement["provider_worker_principal_did"]),
            str(statement["provider_principal_did"]),
            str(statement["quack_owner_principal_did"]),
        )
        if (
            any(not _SHA256.fullmatch(str(statement[field])) for field in cid_fields)
            or any(not item.startswith("did:key:z") for item in evidence_reviewers)
            or not str(statement.get("quack_shard_id") or "")
            or not _positive_int(statement.get("quack_epoch"))
            or not _positive_int(statement.get("quack_fence"))
            or not _positive_int(statement.get("provider_qualification_expires_at_ms"))
            or not _positive_int(statement.get("provider_maximum_parallel_workers"))
            or not _positive_int(
                statement.get("provider_maximum_parallel_containers")
            )
            or int(statement["provider_maximum_parallel_workers"])
            > int(statement["provider_maximum_parallel_containers"])
            or int(statement["provider_maximum_parallel_containers"])
            > EAAEF_MAXIMUM_LANES
            or any(not item.startswith("did:key:z") for item in service_principals)
            or len(service_principals) != len(set(service_principals))
            or bool(set(evidence_reviewers).intersection(service_principals))
            or statement.get("provider_task_dispatch_admitted") is not True
            or not str(statement.get("provider_workload_class") or "")
            or not _positive_int(statement.get("quack_qualification_expires_at_ms"))
            or int(statement["expires_at_ms"])
            > min(
                int(statement["provider_qualification_expires_at_ms"]),
                int(statement["quack_qualification_expires_at_ms"]),
            )
        ):
            raise ExternalAgentBootstrapAdmissionError(
                "admitted statement identities or evidence bounds are invalid"
            )


def _safe_repo_relative_path(value: str | Path) -> Path:
    relative = Path(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.name in {"", ".", ".."}
    ):
        raise ExternalAgentBootstrapAdmissionError(
            "immutable publication path is not repository-relative"
        )
    return relative


def external_agent_bootstrap_admission_relative_path(
    source_head: str,
    *,
    registry_prefix: str = EAAEF_AUTHORITY_REGISTRY_PREFIX,
) -> Path:
    """Return the sole create-once EAAEF-000 path for one source commit."""

    if registry_prefix != EAAEF_AUTHORITY_REGISTRY_PREFIX:
        raise ExternalAgentBootstrapAdmissionError(
            "authority registry prefix is not the reviewed EAAEF prefix"
        )
    if not _GIT_OBJECT.fullmatch(str(source_head or "")):
        raise ExternalAgentBootstrapAdmissionError(
            "bootstrap admission source identity is invalid"
        )
    return _safe_repo_relative_path(registry_prefix) / (
        f"bootstrap-admission--{source_head}.json"
    )


def _runtime_principal_dids(statement: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(
        {
            str(statement.get("provider_worker_principal_did") or ""),
            str(statement.get("provider_principal_did") or ""),
            str(statement.get("quack_owner_principal_did") or ""),
        }
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        metadata.st_mode,
    )


def _open_secure_publication_parent(
    repo_root: str | Path,
    relative: Path,
) -> tuple[int, int, tuple[tuple[str, tuple[int, int, int, int]], ...]]:
    """Open an owner-only parent chain without following any component."""

    root_path = Path(repo_root)
    try:
        root_lstat = os.lstat(root_path)
    except OSError as exc:
        raise ExternalAgentBootstrapAdmissionError(
            "immutable publication root is unavailable"
        ) from exc
    if stat.S_ISLNK(root_lstat.st_mode) or not stat.S_ISDIR(root_lstat.st_mode):
        raise ExternalAgentBootstrapAdmissionError(
            "immutable publication root is linked or non-directory"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        root_fd = os.open(root_path, flags)
    except OSError as exc:
        raise ExternalAgentBootstrapAdmissionError(
            "immutable publication root cannot be opened"
        ) from exc
    current_fd = os.dup(root_fd)
    identities: list[tuple[str, tuple[int, int, int, int]]] = []
    try:
        root_fstat = os.fstat(root_fd)
        if _directory_identity(root_fstat) != _directory_identity(root_lstat):
            raise ExternalAgentBootstrapAdmissionError(
                "immutable publication root changed while opening"
            )
        for part in relative.parts[:-1]:
            try:
                next_fd = os.open(part, flags, dir_fd=current_fd)
            except OSError as exc:
                raise ExternalAgentBootstrapAdmissionError(
                    "immutable publication parent is unavailable"
                ) from exc
            os.close(current_fd)
            current_fd = next_fd
            metadata = os.fstat(current_fd)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) & 0o077
            ):
                raise ExternalAgentBootstrapAdmissionError(
                    "immutable publication parent is not an owner-only directory"
                )
            identities.append((part, _directory_identity(metadata)))
        return root_fd, current_fd, tuple(identities)
    except Exception:
        os.close(current_fd)
        os.close(root_fd)
        raise


def _publication_parent_is_stable(
    root_fd: int,
    identities: Sequence[tuple[str, tuple[int, int, int, int]]],
    parent_fd: int,
) -> bool:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    current = os.dup(root_fd)
    try:
        for part, expected in identities:
            next_fd = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = next_fd
            if _directory_identity(os.fstat(current)) != expected:
                return False
        return _directory_identity(os.fstat(current)) == _directory_identity(
            os.fstat(parent_fd)
        )
    except OSError:
        return False
    finally:
        os.close(current)


def _publish_create_once_repo_json(
    repo_root: str | Path,
    relative_path: str | Path,
    value: Mapping[str, Any],
    *,
    noun: str,
) -> None:
    """Publish canonical JSON through an anchored, no-follow directory walk."""

    relative = _safe_repo_relative_path(relative_path)
    root_fd, parent_fd, identities = _open_secure_publication_parent(
        repo_root, relative
    )
    raw = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    temporary = f".{relative.name}.{secrets.token_hex(16)}.tmp"
    temp_created = False
    target_created = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        temp_created = True
        try:
            offset = 0
            while offset < len(raw):
                written = os.write(descriptor, raw[offset:])
                if written <= 0:
                    raise OSError(f"short immutable {noun} write")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(
                temporary,
                relative.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            target_created = True
        except FileExistsError as exc:
            raise ExternalAgentBootstrapAdmissionError(
                f"refusing to overwrite immutable {noun}"
            ) from exc
        if not _publication_parent_is_stable(
            root_fd, identities, parent_fd
        ):
            raise ExternalAgentBootstrapAdmissionError(
                "immutable publication parent changed during commit"
            )
        os.fsync(parent_fd)
    except Exception:
        if target_created:
            try:
                os.unlink(relative.name, dir_fd=parent_fd)
            except OSError:
                pass
        raise
    finally:
        if temp_created:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except OSError:
                pass
        os.close(parent_fd)
        os.close(root_fd)


def _verify_approval(
    approval: object,
    *,
    expected_role: str,
    statement: Mapping[str, Any],
    trusted_dids: frozenset[str],
    now_ms: int,
) -> str:
    if not isinstance(approval, Mapping) or set(approval) != _APPROVAL_FIELDS:
        raise ExternalAgentBootstrapAdmissionError(f"{expected_role} approval is invalid")
    identity = str(approval.get("identity_did") or "")
    if (
        approval.get("schema") != EAAEF_BOOTSTRAP_APPROVAL_SCHEMA
        or approval.get("role") != expected_role
        or identity not in trusted_dids
        or approval.get("statement_cid") != statement.get("statement_cid")
        or approval.get("one_use_nonce") != statement.get("one_use_nonce")
        or not _positive_int(approval.get("issued_at_ms"))
        or not _positive_int(approval.get("expires_at_ms"))
        or int(approval["issued_at_ms"]) < int(statement["issued_at_ms"])
        or int(approval["issued_at_ms"]) > now_ms
        or now_ms >= int(approval["expires_at_ms"])
        or int(approval["expires_at_ms"]) > int(statement["expires_at_ms"])
    ):
        raise ExternalAgentBootstrapAdmissionError(f"{expected_role} approval is invalid")
    payload = dict(approval)
    signature = payload.pop("signature", None)
    if not isinstance(signature, str) or not signature:
        raise ExternalAgentBootstrapAdmissionError(f"{expected_role} approval is unsigned")
    try:
        verify_did_key_signature(
            identity_did=identity,
            payload=payload,
            signature=signature,
        )
    except (LocalProfileTampered, ValueError) as exc:
        raise ExternalAgentBootstrapAdmissionError(
            f"{expected_role} approval signature is invalid"
        ) from exc
    return identity


def assemble_external_agent_bootstrap_admission(
    statement: Mapping[str, Any],
    *,
    operator_approval: Mapping[str, Any],
    security_approval: Mapping[str, Any],
    trusted_operator_dids: Sequence[str],
    trusted_security_reviewer_dids: Sequence[str],
    now_ms: int,
) -> dict[str, Any]:
    """Assemble and verify the signed receipt without publishing it."""

    _validate_statement_shape(statement)
    operator = _verify_approval(
        operator_approval,
        expected_role="independent_operator",
        statement=statement,
        trusted_dids=frozenset(trusted_operator_dids),
        now_ms=now_ms,
    )
    security = _verify_approval(
        security_approval,
        expected_role="independent_security_reviewer",
        statement=statement,
        trusted_dids=frozenset(trusted_security_reviewer_dids),
        now_ms=now_ms,
    )
    if (
        operator == security
        or operator in _runtime_principal_dids(statement)
        or security in _runtime_principal_dids(statement)
    ):
        raise ExternalAgentBootstrapAdmissionError(
            "operator/security reviewers must be independent of runtime principals"
        )
    receipt = {
        **dict(statement),
        "schema": EAAEF_BOOTSTRAP_ADMISSION_SCHEMA,
        "operator_approval": dict(operator_approval),
        "security_approval": dict(security_approval),
    }
    receipt["receipt_cid"] = _cid(receipt)
    return receipt


def verify_external_agent_bootstrap_admission(
    receipt: object,
    *,
    trusted_operator_dids: Sequence[str],
    trusted_security_reviewer_dids: Sequence[str],
    now_ms: int,
    require_admitted: bool = True,
) -> dict[str, Any]:
    """Read-only verification of one final signed authority receipt."""

    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_FIELDS:
        raise ExternalAgentBootstrapAdmissionError("bootstrap admission receipt is invalid")
    body = dict(receipt)
    receipt_cid = str(body.pop("receipt_cid", ""))
    operator_approval = body.pop("operator_approval", None)
    security_approval = body.pop("security_approval", None)
    statement = dict(body)
    statement["schema"] = EAAEF_BOOTSTRAP_ADMISSION_STATEMENT_SCHEMA
    if receipt_cid != _cid({**body, "operator_approval": operator_approval, "security_approval": security_approval}):
        raise ExternalAgentBootstrapAdmissionError(
            "bootstrap admission receipt self-address is invalid"
        )
    _validate_statement_shape(statement)
    operator = _verify_approval(
        operator_approval,
        expected_role="independent_operator",
        statement=statement,
        trusted_dids=frozenset(trusted_operator_dids),
        now_ms=now_ms,
    )
    security = _verify_approval(
        security_approval,
        expected_role="independent_security_reviewer",
        statement=statement,
        trusted_dids=frozenset(trusted_security_reviewer_dids),
        now_ms=now_ms,
    )
    if (
        operator == security
        or operator in _runtime_principal_dids(statement)
        or security in _runtime_principal_dids(statement)
    ):
        raise ExternalAgentBootstrapAdmissionError(
            "operator/security reviewers must be independent of runtime principals"
        )
    if now_ms >= int(statement["expires_at_ms"]):
        raise ExternalAgentBootstrapAdmissionError("bootstrap admission receipt expired")
    if require_admitted and statement["decision"] != "admitted":
        raise ExternalAgentBootstrapAdmissionError("bootstrap admission is a typed no-go")
    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "eaaef-bootstrap-admission-verification@1"
        ),
        "valid": True,
        "admitted": statement["decision"] == "admitted",
        "receipt_cid": receipt_cid,
        "statement_cid": statement["statement_cid"],
        "board_cid": statement["board_cid"],
        "source_head": statement["source_head"],
        "source_tree": statement["source_tree"],
        "materialization_receipt_cid": statement["materialization_receipt_cid"],
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
        "provider_container_qualification_cid": statement[
            "provider_container_qualification_cid"
        ],
        "provider_qualification_signer_did": statement[
            "provider_qualification_signer_did"
        ],
        "image_qualification_reviewer_did": statement[
            "image_qualification_reviewer_did"
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
        "quack_qualification_reviewer_did": statement[
            "quack_qualification_reviewer_did"
        ],
        "quack_owner_principal_did": statement[
            "quack_owner_principal_did"
        ],
        "population_cid": statement["population_cid"],
        "plan_root_cid": statement["plan_root_cid"],
        "control_projection_root": statement["control_projection_root"],
        "coordination_projection_root": statement[
            "coordination_projection_root"
        ],
        "execution_projection_root": statement["execution_projection_root"],
        "quack_shard_id": statement["quack_shard_id"],
        "quack_epoch": statement["quack_epoch"],
        "quack_fence": statement["quack_fence"],
        "operator_identity_did": operator,
        "security_reviewer_identity_did": security,
        "maximum_lanes": statement["authority"]["maximum_lanes"],
        "authority_mutated": False,
        "process_started": False,
    }


def publish_external_agent_bootstrap_admission(
    repo_root: str | Path,
    receipt: Mapping[str, Any],
    *,
    trusted_operator_dids: Sequence[str],
    trusted_security_reviewer_dids: Sequence[str],
    now_ms: int,
) -> dict[str, Any]:
    """Verify and atomically publish one immutable receipt without overwrite."""

    verification = verify_external_agent_bootstrap_admission(
        receipt,
        trusted_operator_dids=trusted_operator_dids,
        trusted_security_reviewer_dids=trusted_security_reviewer_dids,
        now_ms=now_ms,
        require_admitted=False,
    )
    relative_path = external_agent_bootstrap_admission_relative_path(
        str(verification["source_head"])
    )
    _publish_create_once_repo_json(
        repo_root,
        relative_path,
        receipt,
        noun="bootstrap admission receipt",
    )
    return verification


__all__ = (
    "EAAEF_BOOTSTRAP_ADMISSION_SCHEMA",
    "EAAEF_BOOTSTRAP_ADMISSION_STATEMENT_SCHEMA",
    "EAAEF_BOOTSTRAP_APPROVAL_SCHEMA",
    "EAAEF_AUTHORITY_REGISTRY_PREFIX",
    "EAAEF_BOARD_NAMESPACE",
    "EAAEF_MAXIMUM_LANES",
    "EAAEF_OPERATIONAL_COMMAND_FABRIC_SHARD_ID",
    "EAAEF_SIGNED_COMMAND_FABRIC_PROFILE_SCHEMA",
    "ExternalAgentBootstrapAdmissionError",
    "assemble_external_agent_bootstrap_admission",
    "external_agent_bootstrap_admission_relative_path",
    "prepare_external_agent_bootstrap_admission",
    "prepare_external_agent_bootstrap_approval",
    "publish_external_agent_bootstrap_admission",
    "verify_external_agent_bootstrap_admission",
)
