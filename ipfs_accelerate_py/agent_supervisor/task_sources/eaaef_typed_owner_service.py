"""Owner-local EAAEF commands on one already-open state connection.

This module is deliberately smaller than the historical command fabric.  It
does not open or close a database, publish a listener, accept SQL or an
operation callback, create a projection, or load a downstream catalog.  The
owner binds an already-open operational connection and its one transaction
lock, then exposes only exact authorized-command submit and durable-receipt
lookup operations.

The current signed EAAEF lane artifacts still bind the historical Quack
transport.  Consequently this source implementation remains fail-closed for
production transport admission until an independently signed typed-owner
transport cutover is available.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..validation.eaaef_bootstrap_gateway_launch import (
    EAAEFBootstrapGatewayLaunchError,
    VerifiedEAAEFBootstrapOperationalCapability,
    verify_eaaef_bootstrap_operation_submission,
)
from ..validation.eaaef_lane_gateway_admission import (
    EAAEFLaneGatewayAdmissionError,
    VerifiedEAAEFLaneRuntimeAdmissionV2,
)
from .control_plane_contracts import (
    CommandKind,
    CommandOutcome,
    StateCommand,
    canonical_json_bytes,
)
from .control_plane_transactions import StateTransaction
from .eaaef_bootstrap_daemon_gateway import EAAEF_BOOTSTRAP_DAEMON_OPERATIONS
from .eaaef_borrowed_transaction import (
    EAAEF_BORROWED_TRANSACTION_HANDLER_INTERFACE,
    EAAEFBootstrapBorrowedTransactionOperationHandler,
    eaaef_bootstrap_handler_source_evidence,
)
from .quack_command_authorization import (
    AuthorizedStateCommand,
    QuackCommandAuthorizationError,
    QuackCommandAuthorizationPolicy,
)
from .quack_daemon_gateway import (
    QuackDaemonGatewayError,
    quack_daemon_operation_intent_from_envelope,
)

EAAEF_TYPED_OWNER_COMMAND_SERVICE_INTERFACE: Final = (
    "EAAEFTypedOwnerCommandService@1"
)
EAAEF_TYPED_OWNER_COMMAND_SERVICE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-typed-owner-command-service@1"
)
EAAEF_TYPED_OWNER_COMMAND_SUBMIT_OPERATION: Final = "eaaef.command.submit"
EAAEF_TYPED_OWNER_COMMAND_LOOKUP_OPERATION: Final = "eaaef.command.lookup"
EAAEF_TYPED_OWNER_COMMAND_OPERATIONS: Final = frozenset(
    {
        EAAEF_TYPED_OWNER_COMMAND_SUBMIT_OPERATION,
        EAAEF_TYPED_OWNER_COMMAND_LOOKUP_OPERATION,
    }
)
EAAEF_TYPED_OWNER_SERVICE_QUALIFICATION_STATUS: Final = (
    "implemented_unqualified_fail_closed"
)
EAAEF_TYPED_OWNER_TRANSPORT_PRODUCTION_BLOCKER: Final = (
    "independently_signed_typed_owner_transport_cutover_absent"
)

_PRIVATE_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/private-authorized-command-receipt@1"
)
EAAEF_TYPED_OWNER_PUBLIC_RECEIPT_FIELDS: Final = frozenset(
    {
        "submission_id",
        "envelope_cid",
        "request_id",
        "principal_did",
        "approver_did",
        "authority_ref_cid",
        "lease_id",
        "one_use_nonce",
        "command_id",
        "idempotency_key",
        "outcome",
        "changed",
        "revision",
        "generation",
        "fence_epoch",
        "result_json",
        "error",
        "submitted_at",
        "applied_at",
    }
)
_PRIVATE_RECEIPT_FIELDS: Final = EAAEF_TYPED_OWNER_PUBLIC_RECEIPT_FIELDS | {
    "schema",
    "scope_id",
    "effect",
    "daemon_operation",
    "daemon_operation_intent_cid",
}
_LOCK_TYPE = type(threading.Lock())
_SERVICE_FACTORY_TOKEN = object()


class EAAEFTypedOwnerServiceError(RuntimeError):
    """The closed owner-local EAAEF service rejected an operation."""


class EAAEFTypedOwnerServiceDiverged(EAAEFTypedOwnerServiceError):
    """A durable receipt differs from the exact transported envelope."""


def _receipt_event_id(submission_id: str) -> str:
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": "AuthorizedStateCommandSubmissionIdentity@1",
                "submission_id": str(submission_id),
            }
        )
    ).hexdigest()
    return f"authorized-command-receipt:sha256:{digest}"


def _policy_from_capability(
    capability: VerifiedEAAEFBootstrapOperationalCapability,
) -> QuackCommandAuthorizationPolicy:
    if type(capability) is not VerifiedEAAEFBootstrapOperationalCapability:
        raise EAAEFTypedOwnerServiceError(
            "typed owner service requires an exact verified operational capability"
        )
    raw = capability.get("authorization_policy")
    if not isinstance(raw, Mapping):
        raise EAAEFTypedOwnerServiceError(
            "verified operational capability has no authorization policy"
        )
    value = dict(raw)
    try:
        policy = QuackCommandAuthorizationPolicy(
            board_namespace=value.get("board_namespace"),
            shard_id=value.get("shard_id"),
            store_id=value.get("store_id"),
            authority_ref_cid=value.get("authority_ref_cid"),
            owner_principal_did=value.get("owner_principal_did"),
            owner_generation=value.get("owner_generation"),
            fence_epoch=value.get("fence_epoch"),
            trusted_approver_dids=frozenset(
                value.get("trusted_approver_dids") or ()
            ),
            authorized_principal_dids=frozenset(
                value.get("authorized_principal_dids") or ()
            ),
            allowed_command_kinds=frozenset(
                CommandKind(item)
                for item in value.get("allowed_command_kinds") or ()
            ),
            maximum_authorization_lifetime_ms=value.get(
                "maximum_authorization_lifetime_ms"
            ),
        )
    except (QuackCommandAuthorizationError, TypeError, ValueError) as exc:
        raise EAAEFTypedOwnerServiceError(
            "verified operational authorization policy is invalid"
        ) from exc
    if policy.to_dict() != value:
        raise EAAEFTypedOwnerServiceError(
            "verified operational authorization policy is not canonical"
        )
    return policy


def _handler_from_capability(
    capability: VerifiedEAAEFBootstrapOperationalCapability,
) -> EAAEFBootstrapBorrowedTransactionOperationHandler:
    policy = _policy_from_capability(capability)
    handler = EAAEFBootstrapBorrowedTransactionOperationHandler(
        board_namespace=policy.board_namespace,
        shard_id=policy.shard_id,
        owner_principal_did=policy.owner_principal_did,
        command_principal_did=str(capability["command_principal_did"]),
        owner_session_id=str(capability["owner_session_id"]),
        owner_generation=policy.owner_generation,
        fence_epoch=policy.fence_epoch,
        gateway_binding_cid=str(capability["gateway_binding_cid"]),
        control_plane_schema_version=str(
            capability["control_plane_schema_version"]
        ),
        state_schema_revision=str(capability["state_schema_revision"]),
    )
    evidence = handler.evidence()
    source = eaaef_bootstrap_handler_source_evidence(
        board_namespace=policy.board_namespace,
        shard_id=policy.shard_id,
    )
    if (
        handler.INTERFACE != EAAEF_BORROWED_TRANSACTION_HANDLER_INTERFACE
        or evidence.get("handler_source_evidence_cid")
        != capability.get("borrowed_transaction_handler_source_evidence_cid")
        or evidence.get("handler_source_evidence_cid")
        != source["handler_source_evidence_cid"]
        or evidence.get("operation_count") != 31
        or evidence.get("production_admitted") is not False
        or capability.get("operations")
        != sorted(EAAEF_BOOTSTRAP_DAEMON_OPERATIONS)
    ):
        raise EAAEFTypedOwnerServiceError(
            "verified capability differs from the closed 31-operation handler"
        )
    return handler


def _exact_envelope(envelope: AuthorizedStateCommand) -> None:
    if type(envelope) is not AuthorizedStateCommand:
        raise QuackCommandAuthorizationError(
            "typed owner EAAEF envelope is not exact AuthorizedStateCommand@1"
        )
    if type(envelope.command) is not StateCommand:
        raise QuackCommandAuthorizationError(
            "typed owner EAAEF command is not exact StateCommand@1"
        )


class EAAEFTypedOwnerCommandService:
    """Two-operation service borrowing the sole owner's connection and lock."""

    INTERFACE: ClassVar[str] = EAAEF_TYPED_OWNER_COMMAND_SERVICE_INTERFACE
    SCHEMA: ClassVar[str] = EAAEF_TYPED_OWNER_COMMAND_SERVICE_SCHEMA
    __slots__ = (
        "_connection",
        "_transaction_lock",
        "_admission",
        "_capability",
        "_policy",
        "_handler",
        "_closed",
    )

    def __init__(
        self,
        token: object,
        *,
        connection: Any,
        transaction_lock: Any,
        admission: VerifiedEAAEFLaneRuntimeAdmissionV2,
    ) -> None:
        if token is not _SERVICE_FACTORY_TOKEN:
            raise TypeError(
                "typed owner EAAEF services come from the owner-local binder"
            )
        if (
            type(admission) is not VerifiedEAAEFLaneRuntimeAdmissionV2
            or type(transaction_lock) is not _LOCK_TYPE
            or not callable(getattr(connection, "execute", None))
            or not callable(getattr(connection, "commit", None))
            or not callable(getattr(connection, "rollback", None))
        ):
            raise EAAEFTypedOwnerServiceError(
                "owner service requires one exact admission, owner lock, and open connection"
            )
        try:
            checked = admission.reverify(now_ms=time.time_ns() // 1_000_000)
        except EAAEFLaneGatewayAdmissionError as exc:
            raise EAAEFTypedOwnerServiceError(
                "owner service lane admission failed source re-verification"
            ) from exc
        capability = checked.operational_capability
        self._connection = connection
        self._transaction_lock = transaction_lock
        self._admission = checked
        self._capability = capability
        self._policy = _policy_from_capability(capability)
        self._handler = _handler_from_capability(capability)
        self._closed = False

    @property
    def admission_cid(self) -> str:
        return str(self._admission["merge_admission_cid"])

    @property
    def operational_capability_cid(self) -> str:
        return str(self._capability["capability_cid"])

    def _require_open(self) -> None:
        if self._closed:
            raise EAAEFTypedOwnerServiceError("typed owner EAAEF service is closed")

    def _new_transaction(self, *, session_id: str = "") -> StateTransaction:
        return StateTransaction(
            self._connection,
            store_id=self._policy.store_id,
            session_id=session_id,
        )

    def _lookup_private_receipt(
        self, envelope: AuthorizedStateCommand
    ) -> Mapping[str, Any] | None:
        transaction = self._new_transaction(session_id=envelope.command.session_id)
        try:
            transaction.begin()
            receipt = transaction.lookup_authorized_command_receipt(
                receipt_event_id=_receipt_event_id(envelope.submission_id)
            )
            transaction.commit()
            return receipt
        except Exception:
            if transaction.active:
                transaction.rollback()
            raise

    @staticmethod
    def _public_receipt(
        receipt: Mapping[str, Any],
        *,
        envelope: AuthorizedStateCommand,
        operation: str,
        intent_cid: str,
    ) -> Mapping[str, Any]:
        if set(receipt) != _PRIVATE_RECEIPT_FIELDS:
            raise EAAEFTypedOwnerServiceDiverged(
                "durable EAAEF receipt shape is not exact"
            )
        exact_joins = {
            "schema": _PRIVATE_RECEIPT_SCHEMA,
            "submission_id": envelope.submission_id,
            "envelope_cid": envelope.envelope_cid,
            "request_id": envelope.request_id,
            "principal_did": envelope.principal_did,
            "approver_did": envelope.approver_did,
            "authority_ref_cid": envelope.authority_ref_cid,
            "lease_id": envelope.lease_id,
            "scope_id": envelope.scope_id,
            "effect": envelope.effect,
            "one_use_nonce": envelope.one_use_nonce,
            "command_id": envelope.command.command_id,
            "idempotency_key": envelope.command.idempotency_key,
            "fence_epoch": envelope.command.fence_epoch,
            "daemon_operation": operation,
            "daemon_operation_intent_cid": intent_cid,
        }
        if any(receipt.get(name) != value for name, value in exact_joins.items()):
            raise EAAEFTypedOwnerServiceDiverged(
                "durable EAAEF receipt differs from its exact envelope"
            )
        if receipt.get("outcome") not in {
            CommandOutcome.ACCEPTED.value,
            CommandOutcome.IDEMPOTENT_REPLAY.value,
        } or receipt.get("error"):
            raise EAAEFTypedOwnerServiceDiverged(
                "durable EAAEF receipt is not an accepted owner result"
            )
        integer_fields = (
            "revision",
            "generation",
            "fence_epoch",
            "submitted_at",
            "applied_at",
        )
        if (
            type(receipt.get("changed")) is not bool
            or any(
                isinstance(receipt.get(name), bool)
                or not isinstance(receipt.get(name), int)
                or int(receipt[name]) < 0
                for name in integer_fields
            )
            or receipt.get("generation")
            != envelope.command.expected_generation
            or int(receipt["submitted_at"]) < 1
            or int(receipt["applied_at"]) < int(receipt["submitted_at"])
            or not isinstance(receipt.get("result_json"), str)
            or not isinstance(receipt.get("error"), str)
        ):
            raise EAAEFTypedOwnerServiceDiverged(
                "durable EAAEF receipt result metadata is invalid"
            )
        try:
            result = json.loads(str(receipt.get("result_json") or ""))
        except (TypeError, ValueError) as exc:
            raise EAAEFTypedOwnerServiceDiverged(
                "durable EAAEF receipt result is corrupt"
            ) from exc
        if (
            not isinstance(result, Mapping)
            or set(result) != {"daemon_operation", "intent_cid", "value"}
            or result.get("daemon_operation") != operation
            or result.get("intent_cid") != intent_cid
        ):
            raise EAAEFTypedOwnerServiceDiverged(
                "durable EAAEF receipt does not bind the exact operation intent"
            )
        projected = {
            name: receipt[name]
            for name in EAAEF_TYPED_OWNER_PUBLIC_RECEIPT_FIELDS
        }
        canonical_json_bytes(projected)
        return MappingProxyType(projected)

    def _checked_intent(
        self, envelope: AuthorizedStateCommand
    ) -> tuple[dict[str, Any], str, str]:
        _exact_envelope(envelope)
        try:
            intent = dict(quack_daemon_operation_intent_from_envelope(envelope))
        except QuackDaemonGatewayError as exc:
            raise EAAEFTypedOwnerServiceError(
                "typed owner EAAEF envelope carries a malformed operation intent"
            ) from exc
        operation = str(intent.get("operation") or "")
        intent_cid = str(intent.get("intent_cid") or "")
        if operation not in EAAEF_BOOTSTRAP_DAEMON_OPERATIONS:
            raise EAAEFTypedOwnerServiceError(
                "typed owner operation is outside the exact EAAEF vocabulary"
            )
        self._handler.require_operation(operation)
        return intent, operation, intent_cid

    def lookup_authorized_operation_receipt(
        self, envelope: AuthorizedStateCommand
    ) -> Mapping[str, Any] | None:
        """Adopt one durable receipt without reopening expired authority."""

        self._require_open()
        intent, operation, intent_cid = self._checked_intent(envelope)
        del intent
        with self._transaction_lock:
            receipt = self._lookup_private_receipt(envelope)
        if receipt is None:
            return None
        return self._public_receipt(
            receipt,
            envelope=envelope,
            operation=operation,
            intent_cid=intent_cid,
        )

    def submit_authorized_operation(
        self, envelope: AuthorizedStateCommand
    ) -> Mapping[str, Any]:
        """Verify and atomically apply one exact EAAEF operation."""

        self._require_open()
        intent, operation, intent_cid = self._checked_intent(envelope)
        with self._transaction_lock:
            prior = self._lookup_private_receipt(envelope)
            if prior is not None:
                return self._public_receipt(
                    prior,
                    envelope=envelope,
                    operation=operation,
                    intent_cid=intent_cid,
                )

            now_ms = time.time_ns() // 1_000_000
            try:
                checked = self._admission.reverify(now_ms=now_ms)
            except EAAEFLaneGatewayAdmissionError as exc:
                raise EAAEFTypedOwnerServiceError(
                    "typed owner EAAEF lane admission is stale or changed"
                ) from exc
            capability = checked.operational_capability
            if (
                checked["merge_admission_cid"] != self.admission_cid
                or dict(capability) != dict(self._capability)
            ):
                raise EAAEFTypedOwnerServiceError(
                    "typed owner EAAEF authority changed after service binding"
                )
            try:
                authorization = verify_eaaef_bootstrap_operation_submission(
                    envelope,
                    intent,
                    verified_capability=capability,
                    authorization_policy=self._policy,
                    now_ms=now_ms,
                )
            except EAAEFBootstrapGatewayLaunchError as exc:
                raise QuackCommandAuthorizationError(
                    "typed owner EAAEF operation authorization failed"
                ) from exc
            if (
                authorization.get("operation") != operation
                or authorization.get("scope_id") != envelope.scope_id
                or authorization.get("lease_id") != envelope.lease_id
                or authorization.get("effect") != envelope.effect
            ):
                raise QuackCommandAuthorizationError(
                    "typed owner EAAEF authorization projection diverged"
                )

            transaction = self._new_transaction(
                session_id=envelope.command.session_id
            )
            submitted_at = time.time_ns()
            try:
                transaction.begin()
                lease = transaction.assert_live_authorized_command_lease(
                    lease_id=envelope.lease_id,
                    scope_id=envelope.scope_id,
                    principal_did=envelope.principal_did,
                    effect=envelope.effect,
                    command_kind=envelope.command.command_kind,
                    fence_epoch=envelope.command.fence_epoch,
                    now_ms=now_ms,
                )
                if int(lease.get("fencing_token") or 0) != int(
                    authorization["fencing_token"]
                ):
                    raise QuackCommandAuthorizationError(
                        "typed owner EAAEF task fencing token is stale"
                    )
                transaction.consume_authorized_command_replay_claims(
                    request_id=envelope.request_id,
                    one_use_nonce=envelope.one_use_nonce,
                    scope_id=envelope.scope_id,
                    effect=envelope.effect,
                )

                def apply_owner_operation(
                    active_transaction: StateTransaction,
                    command: StateCommand,
                    _live: Any,
                ) -> Mapping[str, Any]:
                    owner_result = self._handler.apply_authorized_daemon_operation(
                        operation=operation,
                        arguments=dict(authorization["arguments"]),
                        transaction=active_transaction,
                        command=command,
                        lease=dict(lease),
                    )
                    if not isinstance(owner_result, Mapping) or set(
                        owner_result
                    ) != {"value"}:
                        raise EAAEFTypedOwnerServiceError(
                            "closed EAAEF handler returned a non-exact result"
                        )
                    body = {
                        "daemon_operation": operation,
                        "intent_cid": intent_cid,
                        "value": owner_result["value"],
                    }
                    canonical_json_bytes(body)
                    return body

                result = transaction.execute_command(
                    envelope.command,
                    apply=apply_owner_operation,
                    auto_commit=False,
                )
                receipt = {
                    "schema": _PRIVATE_RECEIPT_SCHEMA,
                    "submission_id": envelope.submission_id,
                    "envelope_cid": envelope.envelope_cid,
                    "request_id": envelope.request_id,
                    "principal_did": envelope.principal_did,
                    "approver_did": envelope.approver_did,
                    "authority_ref_cid": envelope.authority_ref_cid,
                    "lease_id": envelope.lease_id,
                    "scope_id": envelope.scope_id,
                    "effect": envelope.effect,
                    "one_use_nonce": envelope.one_use_nonce,
                    "command_id": envelope.command.command_id,
                    "idempotency_key": envelope.command.idempotency_key,
                    "outcome": result.outcome.value,
                    "changed": bool(result.changed),
                    "revision": int(result.revision),
                    "generation": int(result.generation),
                    "fence_epoch": int(result.fence_epoch),
                    "result_json": canonical_json_bytes(dict(result.result)).decode(
                        "utf-8"
                    ),
                    "error": "",
                    "submitted_at": submitted_at,
                    "applied_at": time.time_ns(),
                    "daemon_operation": operation,
                    "daemon_operation_intent_cid": intent_cid,
                }
                transaction.record_authorized_command_receipt(
                    receipt_event_id=_receipt_event_id(envelope.submission_id),
                    stream_id=f"authorized-command:{self._policy.shard_id}",
                    task_cid=envelope.scope_id,
                    session_id=envelope.command.session_id,
                    receipt=receipt,
                )
                transaction.commit()
            except Exception:
                if transaction.active:
                    transaction.rollback()
                recovered = self._lookup_private_receipt(envelope)
                if recovered is None:
                    raise
                return self._public_receipt(
                    recovered,
                    envelope=envelope,
                    operation=operation,
                    intent_cid=intent_cid,
                )
            return self._public_receipt(
                receipt,
                envelope=envelope,
                operation=operation,
                intent_cid=intent_cid,
            )

    def close(self) -> None:
        """Retire only the adapter; the outer owner retains its resources."""

        self._closed = True

    def evidence(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "interface": self.INTERFACE,
                "schema": self.SCHEMA,
                "qualification_status": (
                    EAAEF_TYPED_OWNER_SERVICE_QUALIFICATION_STATUS
                ),
                "production_admitted": False,
                "production_blocker": (
                    EAAEF_TYPED_OWNER_TRANSPORT_PRODUCTION_BLOCKER
                ),
                "operations": sorted(EAAEF_TYPED_OWNER_COMMAND_OPERATIONS),
                "operation_count": 31,
                "admission_cid": self.admission_cid,
                "operational_capability_cid": self.operational_capability_cid,
                "borrows_open_connection": True,
                "shares_owner_transaction_lock": True,
                "opens_database": False,
                "closes_database": False,
                "arbitrary_sql_enabled": False,
                "local_sidecar_enabled": False,
                "downstream_catalog_enabled": False,
            }
        )


def bind_eaaef_typed_owner_command_service(
    *,
    connection: Any,
    transaction_lock: Any,
    admission: VerifiedEAAEFLaneRuntimeAdmissionV2,
) -> EAAEFTypedOwnerCommandService:
    """Bind the closed service to resources already owned by CASF."""

    return EAAEFTypedOwnerCommandService(
        _SERVICE_FACTORY_TOKEN,
        connection=connection,
        transaction_lock=transaction_lock,
        admission=admission,
    )


__all__ = [
    "EAAEF_TYPED_OWNER_COMMAND_LOOKUP_OPERATION",
    "EAAEF_TYPED_OWNER_COMMAND_OPERATIONS",
    "EAAEF_TYPED_OWNER_COMMAND_SERVICE_INTERFACE",
    "EAAEF_TYPED_OWNER_COMMAND_SERVICE_SCHEMA",
    "EAAEF_TYPED_OWNER_COMMAND_SUBMIT_OPERATION",
    "EAAEF_TYPED_OWNER_PUBLIC_RECEIPT_FIELDS",
    "EAAEF_TYPED_OWNER_SERVICE_QUALIFICATION_STATUS",
    "EAAEF_TYPED_OWNER_TRANSPORT_PRODUCTION_BLOCKER",
    "EAAEFTypedOwnerCommandService",
    "EAAEFTypedOwnerServiceDiverged",
    "EAAEFTypedOwnerServiceError",
    "bind_eaaef_typed_owner_command_service",
]
