"""Signed admission for the process-remote Plan-R2 owner boundary.

Plan-R2 is deliberately not part of the bootstrap R1 daemon vocabulary.  This
module admits only the three already-authorized Plan-R2 owner operations and
binds them to a canonical, bounded request/response channel.  A capability
cannot authorize merge, provider, process-birth, generic ``StateCommand``, or
R1 operations, and it never carries a filesystem path, transport token,
callback, Portal, or database handle.

The verifier joins three independently signed facts: the Plan-R2 transition,
the existing atomic-owner operational capability, and this remote-transport
capability.  It performs no I/O and starts no process.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

from ..control.profile_authority import LocalProfileTampered, verify_did_key_signature
from ..planning.external_agent_plan_r2 import (
    ExternalAgentPlanR2Error,
    verify_plan_r2_operational_capability,
    verify_plan_r2_transition_authorization,
)
from ..task_sources.external_agent_state_repository import (
    APPLY_PLAN_R2_OPERATION,
    OBSERVE_PLAN_R2_OPERATION,
    PREPARE_PLAN_R2_OPERATION,
)

PLAN_R2_REMOTE_OWNER_CAPABILITY_INTERFACE: Final = "PlanR2ProcessRemoteOwnerCapability@1"
PLAN_R2_REMOTE_OWNER_CAPABILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/plan-r2-process-remote-owner-capability@1"
)
PLAN_R2_REMOTE_OWNER_REVIEW_ROLE: Final = "independent_plan_r2_remote_transport_reviewer"
PLAN_R2_REMOTE_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/plan-r2-remote-owner-request@1"
)
PLAN_R2_REMOTE_RESPONSE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/plan-r2-remote-owner-response@1"
)
PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE: Final = "PlanR2CanonicalWireChannel@1"
PLAN_R2_REMOTE_OWNER_SERVICE_INTERFACE: Final = "PlanR2RemoteOwnerService@1"
PLAN_R2_REMOTE_CLIENT_GATEWAY_INTERFACE: Final = "PlanR2ProcessRemoteOwnerGateway@1"

PLAN_R2_REMOTE_OPERATIONS: Final = (
    PREPARE_PLAN_R2_OPERATION,
    APPLY_PLAN_R2_OPERATION,
    OBSERVE_PLAN_R2_OPERATION,
)
PLAN_R2_REMOTE_EXCLUDED_OPERATION_CLASSES: Final = (
    "bootstrap_r1",
    "generic_state_command",
    "merge",
    "process_birth",
    "provider_effect",
)
PLAN_R2_REMOTE_AUTHORITY_PARTITION: Final = "plan_r2_remaining_population_only"
PLAN_R2_REMOTE_TRANSPORT_KIND: Final = "qualified_process_remote_canonical_exchange"
MAX_PLAN_R2_REMOTE_CAPABILITY_BYTES: Final = 65_536
MAX_PLAN_R2_REMOTE_REQUEST_BYTES: Final = 786_432
MAX_PLAN_R2_REMOTE_RESPONSE_BYTES: Final = 262_144
MAX_PLAN_R2_REMOTE_WAIT_MS: Final = 60_000
MAX_PLAN_R2_REMOTE_CAPABILITY_LIFETIME_MS: Final = 15 * 60 * 1000

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,511}\Z")
_DID = re.compile(r"did:key:z[A-Za-z0-9]{8,511}\Z")

_CAPABILITY_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "allowed",
        "blockers",
        "authority_partition",
        "allowed_operations",
        "excluded_operation_classes",
        "transport_kind",
        "wire_channel_interface",
        "request_schema",
        "response_schema",
        "owner_service_interface",
        "client_gateway_interface",
        "source_head",
        "source_tree",
        "board_namespace",
        "plan_root_cid",
        "population_cid",
        "plan_r2_authorization_cid",
        "plan_r2_operational_capability_cid",
        "quack_command_fabric_qualification_cid",
        "owner_principal_did",
        "shard_id",
        "store_id",
        "owner_generation",
        "epoch",
        "fence",
        "authorized_principal_did",
        "independent_approver_did",
        "request_channel_id",
        "response_channel_id",
        "maximum_request_bytes",
        "maximum_response_bytes",
        "maximum_wait_ms",
        "canonical_bytes_only",
        "exact_envelope_replay_required",
        "durable_client_journal_required",
        "owner_durable_receipt_adoption_required",
        "r1_operations_allowed",
        "merge_operations_allowed",
        "generic_state_command_allowed",
        "process_birth_allowed",
        "database_authority_crossing_allowed",
        "filesystem_path_authority_crossing_allowed",
        "transport_token_authority_crossing_allowed",
        "callback_authority_crossing_allowed",
        "portal_authority_crossing_allowed",
        "reviewer_role",
        "reviewer_did",
        "issued_at_ms",
        "expires_at_ms",
        "issuance_nonce",
        "reviewer_signature",
        "capability_cid",
    }
)
_STATEMENT_FIELDS: Final = _CAPABILITY_FIELDS - {
    "reviewer_signature",
    "capability_cid",
}


class PlanR2RemoteOwnerAdmissionError(RuntimeError):
    """The distinct remote Plan-R2 authority failed closed."""


def _canonical_bytes(value: Any, *, noun: str) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PlanR2RemoteOwnerAdmissionError(f"{noun} is not canonical JSON") from exc
    if len(encoded) > MAX_PLAN_R2_REMOTE_CAPABILITY_BYTES:
        raise PlanR2RemoteOwnerAdmissionError(f"{noun} exceeds its byte bound")
    return encoded


def _cid(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(_canonical_bytes(value, noun="content identity payload")).hexdigest()
    )


def _sha(value: object, noun: str) -> str:
    text = str(value or "")
    if not _SHA256.fullmatch(text):
        raise PlanR2RemoteOwnerAdmissionError(f"{noun} is not a full sha256 identity")
    return text


def _did(value: object, noun: str) -> str:
    text = str(value or "")
    if not _DID.fullmatch(text):
        raise PlanR2RemoteOwnerAdmissionError(f"{noun} is not an Ed25519 did:key")
    return text


def _identifier(value: object, noun: str) -> str:
    text = str(value or "")
    if not _SAFE_ID.fullmatch(text):
        raise PlanR2RemoteOwnerAdmissionError(f"{noun} is not a bounded identifier")
    return text


def _positive(value: object, noun: str, *, maximum: int | None = None) -> int:
    if type(value) is not int or int(value) < 1:
        raise PlanR2RemoteOwnerAdmissionError(f"{noun} is not a positive integer")
    number = int(value)
    if maximum is not None and number > maximum:
        raise PlanR2RemoteOwnerAdmissionError(f"{noun} exceeds its bound")
    return number


_VERIFIED_REMOTE_OWNER_ADMISSION_TOKEN = object()


class VerifiedPlanR2RemoteOwnerAdmission(Mapping[str, Any]):
    """Exact, immutable join used by both sides of the remote boundary."""

    __slots__ = ("_value",)

    def __init__(self, token: object, value: Mapping[str, Any]) -> None:
        if token is not _VERIFIED_REMOTE_OWNER_ADMISSION_TOKEN:
            raise TypeError(
                "verified remote Plan-R2 admissions come from the signature verifier"
            )
        self._value = MappingProxyType(
            json.loads(
                _canonical_bytes(
                    dict(value), noun="verified remote Plan-R2 admission"
                )
            )
        )

    def __getitem__(self, key: str) -> Any:
        return self._value[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    @property
    def capability_cid(self) -> str:
        return str(self._value["capability_cid"])

    def to_dict(self) -> dict[str, Any]:
        return json.loads(_canonical_bytes(dict(self._value), noun="verified remote admission"))


def plan_r2_remote_owner_capability_signing_payload(
    *,
    source_head: str,
    source_tree: str,
    board_namespace: str,
    plan_root_cid: str,
    population_cid: str,
    plan_r2_authorization_cid: str,
    plan_r2_operational_capability_cid: str,
    quack_command_fabric_qualification_cid: str,
    owner_principal_did: str,
    shard_id: str,
    store_id: str,
    owner_generation: int,
    epoch: int,
    fence: int,
    authorized_principal_did: str,
    independent_approver_did: str,
    request_channel_id: str,
    response_channel_id: str,
    reviewer_did: str,
    issued_at_ms: int,
    expires_at_ms: int,
    issuance_nonce: str,
    maximum_request_bytes: int = MAX_PLAN_R2_REMOTE_REQUEST_BYTES,
    maximum_response_bytes: int = MAX_PLAN_R2_REMOTE_RESPONSE_BYTES,
    maximum_wait_ms: int = 30_000,
) -> Mapping[str, Any]:
    """Build public bytes for an independent remote-transport reviewer."""

    value: dict[str, Any] = {
        "schema": PLAN_R2_REMOTE_OWNER_CAPABILITY_SCHEMA,
        "interface": PLAN_R2_REMOTE_OWNER_CAPABILITY_INTERFACE,
        "allowed": True,
        "blockers": [],
        "authority_partition": PLAN_R2_REMOTE_AUTHORITY_PARTITION,
        "allowed_operations": list(PLAN_R2_REMOTE_OPERATIONS),
        "excluded_operation_classes": list(PLAN_R2_REMOTE_EXCLUDED_OPERATION_CLASSES),
        "transport_kind": PLAN_R2_REMOTE_TRANSPORT_KIND,
        "wire_channel_interface": PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE,
        "request_schema": PLAN_R2_REMOTE_REQUEST_SCHEMA,
        "response_schema": PLAN_R2_REMOTE_RESPONSE_SCHEMA,
        "owner_service_interface": PLAN_R2_REMOTE_OWNER_SERVICE_INTERFACE,
        "client_gateway_interface": PLAN_R2_REMOTE_CLIENT_GATEWAY_INTERFACE,
        "source_head": source_head,
        "source_tree": source_tree,
        "board_namespace": board_namespace,
        "plan_root_cid": plan_root_cid,
        "population_cid": population_cid,
        "plan_r2_authorization_cid": plan_r2_authorization_cid,
        "plan_r2_operational_capability_cid": (plan_r2_operational_capability_cid),
        "quack_command_fabric_qualification_cid": (quack_command_fabric_qualification_cid),
        "owner_principal_did": owner_principal_did,
        "shard_id": shard_id,
        "store_id": store_id,
        "owner_generation": owner_generation,
        "epoch": epoch,
        "fence": fence,
        "authorized_principal_did": authorized_principal_did,
        "independent_approver_did": independent_approver_did,
        "request_channel_id": request_channel_id,
        "response_channel_id": response_channel_id,
        "maximum_request_bytes": maximum_request_bytes,
        "maximum_response_bytes": maximum_response_bytes,
        "maximum_wait_ms": maximum_wait_ms,
        "canonical_bytes_only": True,
        "exact_envelope_replay_required": True,
        "durable_client_journal_required": True,
        "owner_durable_receipt_adoption_required": True,
        "r1_operations_allowed": False,
        "merge_operations_allowed": False,
        "generic_state_command_allowed": False,
        "process_birth_allowed": False,
        "database_authority_crossing_allowed": False,
        "filesystem_path_authority_crossing_allowed": False,
        "transport_token_authority_crossing_allowed": False,
        "callback_authority_crossing_allowed": False,
        "portal_authority_crossing_allowed": False,
        "reviewer_role": PLAN_R2_REMOTE_OWNER_REVIEW_ROLE,
        "reviewer_did": reviewer_did,
        "issued_at_ms": issued_at_ms,
        "expires_at_ms": expires_at_ms,
        "issuance_nonce": issuance_nonce,
    }
    _validate_statement_shape(value)
    return MappingProxyType(value)


def _validate_statement_shape(value: Mapping[str, Any]) -> None:
    if set(value) != _STATEMENT_FIELDS:
        raise PlanR2RemoteOwnerAdmissionError(
            "remote Plan-R2 capability statement shape is not exact"
        )
    _canonical_bytes(dict(value), noun="remote Plan-R2 capability statement")
    if (
        value.get("schema") != PLAN_R2_REMOTE_OWNER_CAPABILITY_SCHEMA
        or value.get("interface") != PLAN_R2_REMOTE_OWNER_CAPABILITY_INTERFACE
        or value.get("allowed") is not True
        or value.get("blockers") != []
        or value.get("authority_partition") != PLAN_R2_REMOTE_AUTHORITY_PARTITION
        or value.get("allowed_operations") != list(PLAN_R2_REMOTE_OPERATIONS)
        or value.get("excluded_operation_classes")
        != list(PLAN_R2_REMOTE_EXCLUDED_OPERATION_CLASSES)
        or value.get("transport_kind") != PLAN_R2_REMOTE_TRANSPORT_KIND
        or value.get("wire_channel_interface") != PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE
        or value.get("request_schema") != PLAN_R2_REMOTE_REQUEST_SCHEMA
        or value.get("response_schema") != PLAN_R2_REMOTE_RESPONSE_SCHEMA
        or value.get("owner_service_interface") != PLAN_R2_REMOTE_OWNER_SERVICE_INTERFACE
        or value.get("client_gateway_interface") != PLAN_R2_REMOTE_CLIENT_GATEWAY_INTERFACE
    ):
        raise PlanR2RemoteOwnerAdmissionError("remote Plan-R2 authority partition is invalid")
    required_true = (
        "canonical_bytes_only",
        "exact_envelope_replay_required",
        "durable_client_journal_required",
        "owner_durable_receipt_adoption_required",
    )
    required_false = (
        "r1_operations_allowed",
        "merge_operations_allowed",
        "generic_state_command_allowed",
        "process_birth_allowed",
        "database_authority_crossing_allowed",
        "filesystem_path_authority_crossing_allowed",
        "transport_token_authority_crossing_allowed",
        "callback_authority_crossing_allowed",
        "portal_authority_crossing_allowed",
    )
    if any(value.get(field) is not True for field in required_true) or any(
        value.get(field) is not False for field in required_false
    ):
        raise PlanR2RemoteOwnerAdmissionError("remote Plan-R2 authority isolation is invalid")
    if not _GIT_OBJECT.fullmatch(str(value.get("source_head") or "")) or not (
        _GIT_OBJECT.fullmatch(str(value.get("source_tree") or ""))
    ):
        raise PlanR2RemoteOwnerAdmissionError("remote Plan-R2 source binding is invalid")
    for field in (
        "plan_root_cid",
        "population_cid",
        "plan_r2_authorization_cid",
        "plan_r2_operational_capability_cid",
        "quack_command_fabric_qualification_cid",
    ):
        _sha(value.get(field), field)
    for field in (
        "owner_principal_did",
        "authorized_principal_did",
        "independent_approver_did",
        "reviewer_did",
    ):
        _did(value.get(field), field)
    for field in (
        "board_namespace",
        "shard_id",
        "store_id",
        "request_channel_id",
        "response_channel_id",
        "issuance_nonce",
    ):
        _identifier(value.get(field), field)
    if (
        value["shard_id"] == value["store_id"]
        or value["request_channel_id"] == value["response_channel_id"]
    ):
        raise PlanR2RemoteOwnerAdmissionError(
            "remote Plan-R2 owner and channel identities must remain distinct"
        )
    for field in ("owner_generation", "epoch", "fence", "issued_at_ms", "expires_at_ms"):
        _positive(value.get(field), field)
    _positive(
        value.get("maximum_request_bytes"),
        "maximum_request_bytes",
        maximum=MAX_PLAN_R2_REMOTE_REQUEST_BYTES,
    )
    _positive(
        value.get("maximum_response_bytes"),
        "maximum_response_bytes",
        maximum=MAX_PLAN_R2_REMOTE_RESPONSE_BYTES,
    )
    _positive(
        value.get("maximum_wait_ms"),
        "maximum_wait_ms",
        maximum=MAX_PLAN_R2_REMOTE_WAIT_MS,
    )
    if value.get("reviewer_role") != PLAN_R2_REMOTE_OWNER_REVIEW_ROLE:
        raise PlanR2RemoteOwnerAdmissionError("remote Plan-R2 reviewer role is invalid")


def seal_plan_r2_remote_owner_capability(
    statement: Mapping[str, Any], *, reviewer_signature: str
) -> Mapping[str, Any]:
    """Attach externally produced review evidence without loading a key."""

    if not isinstance(statement, Mapping):
        raise PlanR2RemoteOwnerAdmissionError(
            "remote Plan-R2 capability statement is not an object"
        )
    value = dict(statement)
    _validate_statement_shape(value)
    signature = str(reviewer_signature or "")
    if not signature:
        raise PlanR2RemoteOwnerAdmissionError(
            "remote Plan-R2 capability reviewer signature is absent"
        )
    signed = {**value, "reviewer_signature": signature}
    capability = {**signed, "capability_cid": _cid(signed)}
    _canonical_bytes(capability, noun="remote Plan-R2 capability")
    return MappingProxyType(capability)


def verify_plan_r2_remote_owner_admission(
    capability: object,
    *,
    plan_r2_operational_capability: Mapping[str, Any],
    authorization: Mapping[str, Any],
    trusted_remote_reviewer_dids: Sequence[str],
    trusted_plan_r2_capability_reviewer_dids: Sequence[str],
    trusted_operator_dids: Sequence[str],
    trusted_security_reviewer_dids: Sequence[str],
    now_ms: int,
    forbidden_reviewer_dids: Sequence[str] = (),
) -> VerifiedPlanR2RemoteOwnerAdmission:
    """Verify the mutually exclusive remote path and all identity joins."""

    if not isinstance(capability, Mapping) or set(capability) != _CAPABILITY_FIELDS:
        raise PlanR2RemoteOwnerAdmissionError("remote Plan-R2 capability shape is not exact")
    value = dict(capability)
    statement = {
        key: item
        for key, item in value.items()
        if key not in {"reviewer_signature", "capability_cid"}
    }
    _validate_statement_shape(statement)
    signature = str(value.get("reviewer_signature") or "")
    signed = {**statement, "reviewer_signature": signature}
    if not signature or value.get("capability_cid") != _cid(signed):
        raise PlanR2RemoteOwnerAdmissionError("remote Plan-R2 capability identity is invalid")
    reviewer = _did(value.get("reviewer_did"), "reviewer_did")
    trusted_remote = frozenset(str(item) for item in trusted_remote_reviewer_dids)
    forbidden = frozenset(str(item) for item in forbidden_reviewer_dids)
    if reviewer not in trusted_remote or reviewer in forbidden:
        raise PlanR2RemoteOwnerAdmissionError(
            "remote Plan-R2 capability reviewer is not independently trusted"
        )
    issued = _positive(value.get("issued_at_ms"), "issued_at_ms")
    expires = _positive(value.get("expires_at_ms"), "expires_at_ms")
    if (
        issued > now_ms
        or now_ms >= expires
        or issued >= expires
        or expires - issued > MAX_PLAN_R2_REMOTE_CAPABILITY_LIFETIME_MS
    ):
        raise PlanR2RemoteOwnerAdmissionError("remote Plan-R2 capability lifetime is invalid")
    try:
        verify_did_key_signature(
            identity_did=reviewer,
            payload=statement,
            signature=signature,
        )
        verified_plan = verify_plan_r2_operational_capability(
            plan_r2_operational_capability,
            trusted_reviewer_dids=trusted_plan_r2_capability_reviewer_dids,
            now_ms=now_ms,
        )
        verified_authorization = verify_plan_r2_transition_authorization(
            authorization,
            trusted_operator_dids=trusted_operator_dids,
            trusted_security_reviewer_dids=trusted_security_reviewer_dids,
            now_ms=now_ms,
        )
    except (LocalProfileTampered, ExternalAgentPlanR2Error, ValueError) as exc:
        raise PlanR2RemoteOwnerAdmissionError(
            "remote Plan-R2 signed authority chain is invalid"
        ) from exc
    plan_reviewer = str(verified_plan.get("reviewer_identity_did") or "")
    identities = {
        reviewer,
        plan_reviewer,
        str(value["owner_principal_did"]),
        str(value["authorized_principal_did"]),
        str(value["independent_approver_did"]),
        str(verified_authorization["operator_identity_did"]),
        str(verified_authorization["security_reviewer_identity_did"]),
    }
    if len(identities) != 7 or not identities.isdisjoint(forbidden):
        raise PlanR2RemoteOwnerAdmissionError(
            "remote Plan-R2 reviewers and command principals are not independent"
        )
    comparisons = {
        "source_head": authorization.get("source_head"),
        "source_tree": authorization.get("source_tree"),
        "board_namespace": authorization.get("board_namespace"),
        "plan_root_cid": authorization.get("plan_root_cid"),
        "population_cid": authorization.get("population_cid"),
        "plan_r2_authorization_cid": authorization.get("authorization_cid"),
        "plan_r2_operational_capability_cid": verified_plan.get("capability_cid"),
        "quack_command_fabric_qualification_cid": authorization.get(
            "quack_command_fabric_qualification_cid"
        ),
        "owner_principal_did": authorization.get("owner_principal_did"),
        "shard_id": authorization.get("shard_id"),
        "store_id": authorization.get("store_id"),
        "owner_generation": authorization.get("owner_generation"),
        "epoch": authorization.get("expected_epoch"),
        "fence": authorization.get("fencing_token"),
    }
    mismatched = sorted(
        field for field, expected in comparisons.items() if value.get(field) != expected
    )
    plan_comparisons = {
        "source_head": value["source_head"],
        "source_tree": value["source_tree"],
        "quack_command_fabric_qualification_cid": value["quack_command_fabric_qualification_cid"],
        "owner_principal_did": value["owner_principal_did"],
        "shard_id": value["shard_id"],
        "owner_generation": value["owner_generation"],
        "epoch": value["epoch"],
        "fence": value["fence"],
    }
    mismatched.extend(
        f"plan_r2.{field}"
        for field, expected in plan_comparisons.items()
        if verified_plan.get(field) != expected
    )
    if expires > int(verified_plan["expires_at_ms"]) or expires > int(
        authorization["expires_at_ms"]
    ):
        mismatched.append("expires_at_ms")
    if mismatched:
        raise PlanR2RemoteOwnerAdmissionError(
            "remote Plan-R2 capability differs from signed owner authority: "
            + ", ".join(mismatched)
        )
    _canonical_bytes(value, noun="remote Plan-R2 capability")
    detached = json.loads(_canonical_bytes(value, noun="remote Plan-R2 verified admission"))
    return VerifiedPlanR2RemoteOwnerAdmission(
        _VERIFIED_REMOTE_OWNER_ADMISSION_TOKEN,
        detached,
    )


__all__ = (
    "MAX_PLAN_R2_REMOTE_REQUEST_BYTES",
    "MAX_PLAN_R2_REMOTE_RESPONSE_BYTES",
    "MAX_PLAN_R2_REMOTE_WAIT_MS",
    "PLAN_R2_REMOTE_CLIENT_GATEWAY_INTERFACE",
    "PLAN_R2_REMOTE_OPERATIONS",
    "PLAN_R2_REMOTE_OWNER_CAPABILITY_INTERFACE",
    "PLAN_R2_REMOTE_OWNER_CAPABILITY_SCHEMA",
    "PLAN_R2_REMOTE_OWNER_REVIEW_ROLE",
    "PLAN_R2_REMOTE_OWNER_SERVICE_INTERFACE",
    "PLAN_R2_REMOTE_REQUEST_SCHEMA",
    "PLAN_R2_REMOTE_RESPONSE_SCHEMA",
    "PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE",
    "PlanR2RemoteOwnerAdmissionError",
    "VerifiedPlanR2RemoteOwnerAdmission",
    "plan_r2_remote_owner_capability_signing_payload",
    "seal_plan_r2_remote_owner_capability",
    "verify_plan_r2_remote_owner_admission",
)
