"""Canonical no-go request for independent review of a future owner transition.

This prepares public bytes and attaches an external signature.  It does not
inspect a broker, authenticate lifecycle artifacts, establish trust, consume
a nonce, or admit production.  Even when signed, the request remains a typed
no-go until a broker-local verifier joins every blocker named below.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

from ..control.profile_authority import (
    LocalProfileTampered,
    verify_did_key_signature,
)
from .plan_r2_remote_owner_admission import (
    MAX_PLAN_R2_REMOTE_REQUEST_BYTES,
    MAX_PLAN_R2_REMOTE_RESPONSE_BYTES,
    MAX_PLAN_R2_REMOTE_WAIT_MS,
    PLAN_R2_REMOTE_OPERATIONS,
    PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE,
)

EAAEF_SINGLE_OWNER_REVIEW_REQUEST_INTERFACE: Final = (
    "EAAEFSingleOwnerPlanR2ReviewRequest@1"
)
EAAEF_SINGLE_OWNER_REVIEW_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-single-owner-plan-r2-review-request@1"
)
EAAEF_SINGLE_OWNER_REVIEW_ROLE: Final = (
    "independent_eaaef_single_owner_transition_reviewer"
)
EAAEF_SINGLE_OWNER_REVIEW_BLOCKERS: Final = (
    "exact_ready_quack_state_server_join_absent",
    "exclusive_owner_lease_and_marker_join_absent",
    "broker_process_birth_join_absent",
    "authenticated_lifecycle_cid_preimages_absent",
    "owner_sealed_cutover_trust_roots_absent",
    "durable_atomic_cutover_replay_journal_absent",
    "prelaunch_r1_authority_contract_absent",
    "plan_r2_management_exchange_absent",
)
MAX_EAAEF_SINGLE_OWNER_REVIEW_REQUEST_BYTES: Final = 65_536
MAX_EAAEF_SINGLE_OWNER_REVIEW_LIFETIME_MS: Final = 15 * 60 * 1000

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_OID_RE = re.compile(r"^[0-9a-f]{40}$")
_DID_RE = re.compile(r"^did:key:z[A-Za-z0-9]{8,511}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,511}$")
_CID_FIELDS = frozenset(
    """source_forest_root management_binding_cid
    management_snapshot_bindings_cid management_capsule_cid
    broker_process_birth_cid owner_start_receipt_cid owner_commit_receipt_cid
    r1_merge_admission_cid r1_operational_capability_cid
    plan_r2_authorization_cid plan_r2_operational_capability_cid
    plan_r2_remote_capability_cid plan_r2_authority_bundle_cid
    plan_r2_trust_bundle_cid active_plan_root_cid
    active_plan_revision_cid""".split()
)
_INTEGER_FIELDS = frozenset(
    """owner_generation plan_r2_epoch fence_epoch active_plan_revision
    maximum_wait_ms""".split()
)
_IDENTIFIER_FIELDS = frozenset(
    """board_namespace management_generation_id shard_id store_id
    r1_service_interface r1_service_schema plan_r2_service_interface
    plan_r2_service_schema plan_r2_gateway_interface request_channel_id
    response_channel_id""".split()
)
_BINDING_FIELDS = (
    _CID_FIELDS
    | _INTEGER_FIELDS
    | _IDENTIFIER_FIELDS
    | {"source_head", "source_tree", "owner_principal_did"}
)
_STATEMENT_FIELDS = _BINDING_FIELDS | set(
    """schema interface allowed production_admitted blockers proposed_operations
    wire_channel_interface maximum_request_bytes maximum_response_bytes
    same_owner_gateway_required same_open_connection_required
    same_transaction_lock_required opens_database_allowed closes_database_allowed
    attach_database_allowed sidecar_allowed raw_database_authority_allowed
    filesystem_path_authority_allowed transport_token_authority_allowed
    sql_authority_allowed reviewer_role reviewer_did issued_at_ms expires_at_ms
    issuance_nonce""".split()
)


class EAAEFSingleOwnerReviewRequestError(RuntimeError):
    """The non-authoritative review request is malformed."""


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(type(name) is str for name in value):
            raise EAAEFSingleOwnerReviewRequestError(
                "single-owner review request contains a non-string field"
            )
        return {name: _canonical_value(item) for name, item in value.items()}
    if type(value) in {list, tuple}:
        return [_canonical_value(item) for item in value]
    return value


def canonical_eaaef_single_owner_review_request_bytes(value: Any) -> bytes:
    try:
        raw = json.dumps(
            _canonical_value(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise EAAEFSingleOwnerReviewRequestError(
            "single-owner review request is not canonical JSON"
        ) from exc
    if not raw or len(raw) > MAX_EAAEF_SINGLE_OWNER_REVIEW_REQUEST_BYTES:
        raise EAAEFSingleOwnerReviewRequestError(
            "single-owner review request exceeds its byte bound"
        )
    return raw


def _cid(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        canonical_eaaef_single_owner_review_request_bytes(value)
    ).hexdigest()


def _immutable_request(value: Mapping[str, Any]) -> Mapping[str, Any]:
    detached = dict(value)
    detached["blockers"] = tuple(detached["blockers"])
    detached["proposed_operations"] = tuple(detached["proposed_operations"])
    return MappingProxyType(detached)


def _validate_statement(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(canonical_eaaef_single_owner_review_request_bytes(dict(raw)))
    if set(value) != _STATEMENT_FIELDS:
        raise EAAEFSingleOwnerReviewRequestError(
            "single-owner review request shape is not exact"
        )
    constants = {
        "schema": EAAEF_SINGLE_OWNER_REVIEW_REQUEST_SCHEMA,
        "interface": EAAEF_SINGLE_OWNER_REVIEW_REQUEST_INTERFACE,
        "allowed": False,
        "production_admitted": False,
        "blockers": list(EAAEF_SINGLE_OWNER_REVIEW_BLOCKERS),
        "proposed_operations": list(PLAN_R2_REMOTE_OPERATIONS),
        "wire_channel_interface": PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE,
        "maximum_request_bytes": MAX_PLAN_R2_REMOTE_REQUEST_BYTES,
        "maximum_response_bytes": MAX_PLAN_R2_REMOTE_RESPONSE_BYTES,
        "reviewer_role": EAAEF_SINGLE_OWNER_REVIEW_ROLE,
    }
    if any(value.get(name) != expected for name, expected in constants.items()):
        raise EAAEFSingleOwnerReviewRequestError(
            "single-owner review request constants differ"
        )
    required_true = {
        "same_owner_gateway_required",
        "same_open_connection_required",
        "same_transaction_lock_required",
    }
    authority_flags = {
        "opens_database_allowed",
        "closes_database_allowed",
        "attach_database_allowed",
        "sidecar_allowed",
        "raw_database_authority_allowed",
        "filesystem_path_authority_allowed",
        "transport_token_authority_allowed",
        "sql_authority_allowed",
    }
    if any(value[name] is not True for name in required_true) or any(
        value[name] is not False for name in authority_flags
    ):
        raise EAAEFSingleOwnerReviewRequestError(
            "single-owner review request isolation differs"
        )
    if any(
        type(value[name]) is not str or not _SHA256_RE.fullmatch(value[name])
        for name in _CID_FIELDS
    ):
        raise EAAEFSingleOwnerReviewRequestError(
            "single-owner review content identity differs"
        )
    if any(
        type(value[name]) is not str or not _SAFE_ID_RE.fullmatch(value[name])
        for name in _IDENTIFIER_FIELDS
    ):
        raise EAAEFSingleOwnerReviewRequestError(
            "single-owner review bounded identity differs"
        )
    if any(
        type(value[name]) is not str or not _GIT_OID_RE.fullmatch(value[name])
        for name in ("source_head", "source_tree")
    ) or any(
        type(value[name]) is not str or not _DID_RE.fullmatch(value[name])
        for name in ("owner_principal_did", "reviewer_did")
    ):
        raise EAAEFSingleOwnerReviewRequestError(
            "single-owner review source or DID identity differs"
        )
    if value["reviewer_did"] == value["owner_principal_did"]:
        raise EAAEFSingleOwnerReviewRequestError(
            "single-owner review reviewer is not independent of the owner"
        )
    positive = _INTEGER_FIELDS | {"issued_at_ms", "expires_at_ms"}
    if any(type(value[name]) is not int or value[name] < 1 for name in positive):
        raise EAAEFSingleOwnerReviewRequestError(
            "single-owner review integer binding differs"
        )
    if (
        value["maximum_wait_ms"] > MAX_PLAN_R2_REMOTE_WAIT_MS
        or value["issued_at_ms"] >= value["expires_at_ms"]
        or value["expires_at_ms"] - value["issued_at_ms"]
        > MAX_EAAEF_SINGLE_OWNER_REVIEW_LIFETIME_MS
        or type(value["issuance_nonce"]) is not str
        or not _SAFE_ID_RE.fullmatch(value["issuance_nonce"])
    ):
        raise EAAEFSingleOwnerReviewRequestError(
            "single-owner review lifetime, wait, or nonce differs"
        )
    return value


def prepare_eaaef_single_owner_review_request(
    bindings: Mapping[str, Any],
    *,
    reviewer_did: str,
    issued_at_ms: int,
    expires_at_ms: int,
    issuance_nonce: str,
) -> Mapping[str, Any]:
    """Prepare public no-go bytes; no key or trust root is accepted."""

    if not isinstance(bindings, Mapping) or set(bindings) != _BINDING_FIELDS:
        raise EAAEFSingleOwnerReviewRequestError(
            "single-owner review binding shape is not exact"
        )
    value = {
        "schema": EAAEF_SINGLE_OWNER_REVIEW_REQUEST_SCHEMA,
        "interface": EAAEF_SINGLE_OWNER_REVIEW_REQUEST_INTERFACE,
        "allowed": False,
        "production_admitted": False,
        "blockers": list(EAAEF_SINGLE_OWNER_REVIEW_BLOCKERS),
        **dict(bindings),
        "proposed_operations": list(PLAN_R2_REMOTE_OPERATIONS),
        "wire_channel_interface": PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE,
        "maximum_request_bytes": MAX_PLAN_R2_REMOTE_REQUEST_BYTES,
        "maximum_response_bytes": MAX_PLAN_R2_REMOTE_RESPONSE_BYTES,
        "same_owner_gateway_required": True,
        "same_open_connection_required": True,
        "same_transaction_lock_required": True,
        **{
            name: False
            for name in (
                "opens_database_allowed closes_database_allowed attach_database_allowed "
                "sidecar_allowed raw_database_authority_allowed "
                "filesystem_path_authority_allowed transport_token_authority_allowed "
                "sql_authority_allowed"
            ).split()
        },
        "reviewer_role": EAAEF_SINGLE_OWNER_REVIEW_ROLE,
        "reviewer_did": reviewer_did,
        "issued_at_ms": issued_at_ms,
        "expires_at_ms": expires_at_ms,
        "issuance_nonce": issuance_nonce,
    }
    return _immutable_request(_validate_statement(value))


def seal_eaaef_single_owner_review_request(
    statement: Mapping[str, Any], *, reviewer_signature: str
) -> Mapping[str, Any]:
    """Authenticate signature possession without establishing signer trust.

    A valid signature proves only that the public ``did:key`` signed these
    exact canonical request bytes.  It does not trust the reviewer, consume
    the nonce, verify the lifecycle claims, or admit production.
    """

    value = _validate_statement(statement)
    if type(reviewer_signature) is not str:
        raise EAAEFSingleOwnerReviewRequestError(
            "single-owner review signature syntax differs"
        )
    signature = reviewer_signature
    try:
        signature_bytes = base64.b64decode(signature, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise EAAEFSingleOwnerReviewRequestError(
            "single-owner review signature syntax differs"
        ) from exc
    if len(signature_bytes) != 64 or base64.b64encode(signature_bytes).decode() != signature:
        raise EAAEFSingleOwnerReviewRequestError(
            "single-owner review signature syntax differs"
        )
    try:
        verify_did_key_signature(
            identity_did=str(value["reviewer_did"]),
            payload=value,
            signature=signature,
        )
    except (LocalProfileTampered, ValueError) as exc:
        raise EAAEFSingleOwnerReviewRequestError(
            "single-owner review signature authenticity differs"
        ) from exc
    signed = {**value, "reviewer_signature": signature}
    sealed = {**signed, "request_cid": _cid(signed)}
    canonical_eaaef_single_owner_review_request_bytes(sealed)
    return _immutable_request(sealed)


__all__ = (
    "EAAEF_SINGLE_OWNER_REVIEW_BLOCKERS",
    "EAAEF_SINGLE_OWNER_REVIEW_REQUEST_INTERFACE",
    "EAAEF_SINGLE_OWNER_REVIEW_REQUEST_SCHEMA",
    "EAAEF_SINGLE_OWNER_REVIEW_ROLE",
    "EAAEFSingleOwnerReviewRequestError",
    "MAX_EAAEF_SINGLE_OWNER_REVIEW_LIFETIME_MS",
    "MAX_EAAEF_SINGLE_OWNER_REVIEW_REQUEST_BYTES",
    "canonical_eaaef_single_owner_review_request_bytes",
    "prepare_eaaef_single_owner_review_request",
    "seal_eaaef_single_owner_review_request",
)
