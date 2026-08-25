"""Authorized Plan-R2 application adapter for the fenced Quack owner.

This module is intentionally not a repository, scheduler, or DuckDB client.
It converts an already admitted ``PlanR2TransitionAuthorization@1`` into an
independently signed ``AuthorizedStateCommand@1`` and submits that envelope to
the sole mutable owner.  The owner-side operation is responsible for verifying
both signatures and applying the complete plan population in one transaction.

The command fabric may supply the required callback only through its exact,
independently qualified Plan-R2 owner gateway.  Consequently this adapter has
no direct-file or bare-``StateCommand`` fallback.  Production construction
fails closed unless the owner component implements the narrow interface
declared below and binds the signed operational capability.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Protocol, runtime_checkable

from ..planning.external_agent_plan_r2 import (
    AUTHORIZED_PLAN_R2_REPOSITORY_INTERFACE,
    PLAN_R2_PREPARED_PROJECTION_SCHEMA,
    PLAN_R2_STATE_OBSERVATION_SCHEMA,
    PLAN_R2_TRANSITION_RECEIPT_SCHEMA,
)
from .control_plane_contracts import (
    CommandKind,
    StateAuthorityClass,
    StateCommand,
)
from .quack_command_authorization import (
    AuthorizedStateCommand,
    authorized_state_command_signing_payload,
    seal_authorized_state_command,
)

PLAN_R2_OWNER_GATEWAY_INTERFACE: Final = "AuthorizedStateCommandPlanR2OwnerGateway@1"
PLAN_R2_OWNER_OPERATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/authorized-plan-r2-owner-operation@1"
)
PLAN_R2_GATEWAY_COMPONENT_INTERFACE: Final = "QuackDaemonGatewayComponent@1"

PREPARE_PLAN_R2_OPERATION: Final = "plan_r2.prepare"
APPLY_PLAN_R2_OPERATION: Final = "plan_r2.apply"
OBSERVE_PLAN_R2_OPERATION: Final = "plan_r2.observe"

_OPERATIONS = frozenset(
    {PREPARE_PLAN_R2_OPERATION, APPLY_PLAN_R2_OPERATION, OBSERVE_PLAN_R2_OPERATION}
)
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,511}\Z")
_AUTHORIZATION_REQUIRED = frozenset(
    {
        "authorization_cid",
        "statement_cid",
        "board_namespace",
        "source_head",
        "source_tree",
        "owner_principal_did",
        "shard_id",
        "store_id",
        "owner_generation",
        "expected_epoch",
        "fencing_token",
        "lease_id",
        "expected_version",
        "expected_event_cursor",
        "request_id",
        "idempotency_key",
        "one_use_nonce",
        "deadline_ms",
        "expires_at_ms",
        "quack_command_fabric_qualification_cid",
        "population_cid",
        "plan_root_cid",
        "task_population_cid",
        "dependency_population_cid",
        "protected_tasks_root_cid",
        "frontier_cid",
        "new_plan",
        "tasks",
        "dependencies",
        "protected_tasks",
        "frontier_task_cids",
    }
)
_PREPARED_FIELDS = frozenset(
    {
        "schema",
        "authorization_cid",
        "statement_cid",
        "capability_cid",
        "authorized_prepare_command_cid",
        "source_head",
        "source_tree",
        "shard_id",
        "owner_generation",
        "epoch",
        "fence",
        "before_plan_cid",
        "before_plan_root_cid",
        "before_plan_revision",
        "before_version",
        "before_event_cursor",
        "before_semantic_root_cid",
        "population_cid",
        "plan_root_cid",
        "protected_tasks_root_cid",
        "frontier_cid",
        "prepared_at_ms",
        "expires_at_ms",
        "authority_mutated",
        "process_started",
        "projection_cid",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "authorization_cid",
        "statement_cid",
        "capability_cid",
        "authorized_prepare_command_cid",
        "authorized_apply_command_cid",
        "prepared_projection_cid",
        "source_head",
        "source_tree",
        "shard_id",
        "owner_generation",
        "epoch",
        "fence",
        "before_plan_cid",
        "after_plan_cid",
        "before_plan_root_cid",
        "after_plan_root_cid",
        "before_plan_revision",
        "after_plan_revision",
        "before_version",
        "after_version",
        "before_event_cursor",
        "after_event_cursor",
        "before_semantic_root_cid",
        "after_semantic_root_cid",
        "population_cid",
        "task_population_cid",
        "dependency_population_cid",
        "protected_tasks_root_cid",
        "frontier_cid",
        "frontier_task_cids",
        "protected_tasks_unchanged",
        "transaction_cid",
        "replayed",
        "committed_at_ms",
        "receipt_cid",
    }
)
_OBSERVATION_FIELDS = frozenset(
    {
        "schema",
        "authorization_cid",
        "transition_receipt_cid",
        "transaction_cid",
        "authorized_prepare_command_cid",
        "authorized_apply_command_cid",
        "quack_command_fabric_qualification_cid",
        "source_head",
        "source_tree",
        "owner_principal_did",
        "shard_id",
        "owner_generation",
        "epoch",
        "fence",
        "store_version",
        "active_plan_cid",
        "active_plan_root_cid",
        "active_plan_revision",
        "event_cursor",
        "semantic_root_cid",
        "population_cid",
        "task_population_cid",
        "dependency_population_cid",
        "protected_tasks_root_cid",
        "frontier_cid",
        "frontier_task_cids",
        "captured_at_ms",
        "authority_mutated",
        "process_started",
        "observation_cid",
    }
)


class PlanR2OwnerAdapterError(RuntimeError):
    """The signed Plan-R2 owner path failed closed."""


class PlanR2OwnerAdapterUnavailable(PlanR2OwnerAdapterError):
    """The sole owner lacks the required atomic Plan-R2 operation."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise PlanR2OwnerAdapterError("value is not canonical JSON") from exc


def _cid(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _mapping(value: object, noun: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise PlanR2OwnerAdapterError(f"{noun} must be an object")
    return dict(value)


def _positive(value: object, noun: str) -> int:
    if type(value) is not int or int(value) < 1:
        raise PlanR2OwnerAdapterError(f"{noun} must be a positive integer")
    return int(value)


def _safe_id(value: object, noun: str) -> str:
    text = str(value or "")
    if not _SAFE_ID.fullmatch(text):
        raise PlanR2OwnerAdapterError(f"{noun} must be a bounded identifier")
    return text


def _sha(value: object, noun: str) -> str:
    text = str(value or "")
    if not _SHA256.fullmatch(text):
        raise PlanR2OwnerAdapterError(f"{noun} must be a full sha256 identity")
    return text


@runtime_checkable
class AuthorizedPlanR2OwnerGateway(Protocol):
    """Exact missing owner API; implementations own the private transaction."""

    INTERFACE: str

    def submit_authorized_plan_r2_operation(
        self,
        envelope: AuthorizedStateCommand,
        operation_payload: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


@dataclass(slots=True)
class ExternalAgentStateRepository:
    """Concrete Plan-R2 component over the sole AuthorizedStateCommand owner.

    ``envelope_signer`` is an independently configured approval signer.  The
    adapter never selects trust roots or receives a DuckDB path/connection.
    """

    owner_gateway: AuthorizedPlanR2OwnerGateway
    board_namespace: str
    shard_id: str
    store_id: str
    owner_principal_did: str
    owner_generation: int
    owner_epoch: int
    fence_epoch: int
    capability_cid: str
    command_fabric_qualification_cid: str
    principal_did: str
    approver_did: str
    envelope_signer: Callable[[Mapping[str, Any]], str] = field(repr=False)
    ingress_slot_allocator: Callable[[], int] = field(repr=False)
    gateway_binding_cid: str = ""
    clock_ms: Callable[[], int] = field(default=lambda: time.time_ns() // 1_000_000, repr=False)

    INTERFACE: ClassVar[str] = AUTHORIZED_PLAN_R2_REPOSITORY_INTERFACE
    GATEWAY_COMPONENT_INTERFACE: ClassVar[str] = PLAN_R2_GATEWAY_COMPONENT_INTERFACE

    def __post_init__(self) -> None:
        if getattr(
            self.owner_gateway, "INTERFACE", ""
        ) != PLAN_R2_OWNER_GATEWAY_INTERFACE or not isinstance(
            self.owner_gateway, AuthorizedPlanR2OwnerGateway
        ):
            raise PlanR2OwnerAdapterUnavailable(
                "typed_quack_plan_transition_atomic_owner_operation_unavailable: "
                "requires AuthorizedStateCommandPlanR2OwnerGateway@1."
                "submit_authorized_plan_r2_operation(envelope, operation_payload)"
            )
        self.board_namespace = _safe_id(self.board_namespace, "board_namespace")
        self.shard_id = _safe_id(self.shard_id, "shard_id")
        self.store_id = _safe_id(self.store_id, "store_id")
        if self.shard_id == self.store_id:
            raise PlanR2OwnerAdapterError("shard_id and store_id must remain distinct")
        for name in ("owner_principal_did", "principal_did", "approver_did"):
            value = str(getattr(self, name) or "")
            if not value.startswith("did:key:z"):
                raise PlanR2OwnerAdapterError(f"{name} must be an Ed25519 did:key")
        if len({self.owner_principal_did, self.principal_did, self.approver_did}) != 3:
            raise PlanR2OwnerAdapterError(
                "owner, command principal, and independent approver must be distinct"
            )
        self.owner_generation = _positive(self.owner_generation, "owner_generation")
        self.owner_epoch = _positive(self.owner_epoch, "owner_epoch")
        self.fence_epoch = _positive(self.fence_epoch, "fence_epoch")
        self.capability_cid = _sha(self.capability_cid, "capability_cid")
        self.command_fabric_qualification_cid = _sha(
            self.command_fabric_qualification_cid,
            "command_fabric_qualification_cid",
        )
        if self.gateway_binding_cid:
            self.gateway_binding_cid = _sha(self.gateway_binding_cid, "gateway_binding_cid")

    def attach(self) -> None:
        attach = getattr(self.owner_gateway, "attach", None)
        if callable(attach):
            attach()

    def close(self) -> None:
        close = getattr(self.owner_gateway, "close", None)
        if callable(close):
            close()

    # Closed component aliases used by QuackDaemonCommandGateway@1.
    def prepare(self, authorization: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.prepare_authorized_plan_r2_transition(authorization)

    def apply(
        self,
        authorization: Mapping[str, Any],
        prepared_projection: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self.apply_authorized_plan_r2_transition(authorization, prepared_projection)

    def observe(
        self,
        authorization: Mapping[str, Any],
        transition_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self.observe_authorized_plan_r2_transition(authorization, transition_receipt)

    def prepare_authorized_plan_r2_transition(
        self, authorization: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return self._submit(
            PREPARE_PLAN_R2_OPERATION,
            authorization,
            context={},
            response_schema=PLAN_R2_PREPARED_PROJECTION_SCHEMA,
            response_fields=_PREPARED_FIELDS,
            response_cid_field="projection_cid",
            command_cid_field="authorized_prepare_command_cid",
        )

    def apply_authorized_plan_r2_transition(
        self,
        authorization: Mapping[str, Any],
        prepared_projection: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        prepared = _mapping(prepared_projection, "prepared_projection")
        return self._submit(
            APPLY_PLAN_R2_OPERATION,
            authorization,
            context={"prepared_projection": prepared},
            response_schema=PLAN_R2_TRANSITION_RECEIPT_SCHEMA,
            response_fields=_RECEIPT_FIELDS,
            response_cid_field="receipt_cid",
            command_cid_field="authorized_apply_command_cid",
        )

    def observe_authorized_plan_r2_transition(
        self,
        authorization: Mapping[str, Any],
        transition_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        receipt = _mapping(transition_receipt, "transition_receipt")
        return self._submit(
            OBSERVE_PLAN_R2_OPERATION,
            authorization,
            context={"transition_receipt": receipt},
            response_schema=PLAN_R2_STATE_OBSERVATION_SCHEMA,
            response_fields=_OBSERVATION_FIELDS,
            response_cid_field="observation_cid",
            command_cid_field="authorized_apply_command_cid",
            require_command_cid_match=False,
        )

    def _submit(
        self,
        operation: str,
        authorization: Mapping[str, Any],
        *,
        context: Mapping[str, Any],
        response_schema: str,
        response_fields: frozenset[str],
        response_cid_field: str,
        command_cid_field: str,
        require_command_cid_match: bool = True,
    ) -> Mapping[str, Any]:
        auth = self._validate_authorization_binding(authorization)
        if operation not in _OPERATIONS:
            raise PlanR2OwnerAdapterError("Plan-R2 operation is not allowlisted")
        payload = {
            "schema": PLAN_R2_OWNER_OPERATION_SCHEMA,
            "operation": operation,
            "authorization": auth,
            **dict(context),
        }
        payload_cid = _cid(payload)
        recover_envelope = getattr(
            self.owner_gateway,
            "recover_exact_authorized_plan_r2_envelope",
            None,
        )
        recovered = recover_envelope(payload) if callable(recover_envelope) else None
        if recovered is not None:
            if type(recovered) is not AuthorizedStateCommand:
                raise PlanR2OwnerAdapterError(
                    "recoverable Plan-R2 owner returned an untyped envelope"
                )
            ingress_slot = _positive(recovered.ingress_slot, "recovered ingress_slot")
            expected_command = self._build_state_command(
                operation=operation,
                authorization=auth,
                operation_payload_cid=payload_cid,
                context=context,
                ingress_slot=ingress_slot,
            )
            if (
                recovered.command.to_record() != expected_command.to_record()
                or recovered.authority_ref_cid != auth["authorization_cid"]
                or recovered.board_namespace != self.board_namespace
                or recovered.shard_id != self.shard_id
                or recovered.owner_principal_did != self.owner_principal_did
                or recovered.principal_did != self.principal_did
                or recovered.approver_did != self.approver_did
                or recovered.scope_id != auth["plan_root_cid"]
            ):
                raise PlanR2OwnerAdapterError(
                    "recoverable Plan-R2 owner envelope differs from the exact operation"
                )
            envelope = recovered
        else:
            ingress_slot = _positive(self.ingress_slot_allocator(), "ingress_slot")
            command = self._build_state_command(
                operation=operation,
                authorization=auth,
                operation_payload_cid=payload_cid,
                context=context,
                ingress_slot=ingress_slot,
            )
            envelope = self._build_envelope(
                operation=operation,
                authorization=auth,
                command=command,
                operation_payload_cid=payload_cid,
                ingress_slot=ingress_slot,
            )
        result = _mapping(
            self.owner_gateway.submit_authorized_plan_r2_operation(envelope, payload),
            f"{operation} result",
        )
        if set(result) != response_fields or result.get("schema") != response_schema:
            raise PlanR2OwnerAdapterError(f"{operation} owner result does not use its exact schema")
        body = dict(result)
        claimed_cid = str(body.pop(response_cid_field, ""))
        if claimed_cid != _cid(body):
            raise PlanR2OwnerAdapterError(f"{operation} owner result CID is invalid")
        if result.get("authorization_cid") != auth["authorization_cid"]:
            raise PlanR2OwnerAdapterError(f"{operation} result is joined to another authorization")
        if require_command_cid_match and result.get(command_cid_field) != envelope.envelope_cid:
            raise PlanR2OwnerAdapterError(
                f"{operation} result is joined to another authorized command"
            )
        if operation == OBSERVE_PLAN_R2_OPERATION:
            receipt = _mapping(context.get("transition_receipt"), "transition_receipt")
            if result.get("authorized_apply_command_cid") != receipt.get(
                "authorized_apply_command_cid"
            ) or result.get("transition_receipt_cid") != receipt.get("receipt_cid"):
                raise PlanR2OwnerAdapterError(
                    "Plan-R2 observation is joined to another apply receipt"
                )
        return result

    def _validate_authorization_binding(self, authorization: Mapping[str, Any]) -> dict[str, Any]:
        auth = _mapping(authorization, "authorization")
        missing = sorted(_AUTHORIZATION_REQUIRED - set(auth))
        if missing:
            raise PlanR2OwnerAdapterError(
                "Plan-R2 authorization lacks required fields: " + ", ".join(missing)
            )
        now_ms = _positive(self.clock_ms(), "clock_ms")
        if now_ms >= _positive(auth["expires_at_ms"], "expires_at_ms"):
            raise PlanR2OwnerAdapterError("Plan-R2 authorization expired")
        if now_ms >= _positive(auth["deadline_ms"], "deadline_ms"):
            raise PlanR2OwnerAdapterError("Plan-R2 command deadline expired")
        exact = {
            "board_namespace": self.board_namespace,
            "shard_id": self.shard_id,
            "store_id": self.store_id,
            "owner_principal_did": self.owner_principal_did,
            "owner_generation": self.owner_generation,
            "expected_epoch": self.owner_epoch,
            "fencing_token": self.fence_epoch,
            "quack_command_fabric_qualification_cid": (self.command_fabric_qualification_cid),
        }
        mismatched = sorted(name for name, expected in exact.items() if auth.get(name) != expected)
        if mismatched:
            raise PlanR2OwnerAdapterError(
                "Plan-R2 authorization differs from owner binding: " + ", ".join(mismatched)
            )
        _sha(auth["authorization_cid"], "authorization_cid")
        _sha(auth["statement_cid"], "statement_cid")
        _safe_id(auth["expected_event_cursor"], "expected_event_cursor")
        return auth

    def _build_state_command(
        self,
        *,
        operation: str,
        authorization: Mapping[str, Any],
        operation_payload_cid: str,
        context: Mapping[str, Any],
        ingress_slot: int,
    ) -> StateCommand:
        prepared_cid = ""
        receipt_cid = ""
        if "prepared_projection" in context:
            prepared_cid = _sha(
                _mapping(context["prepared_projection"], "prepared_projection").get(
                    "projection_cid"
                ),
                "prepared_projection.projection_cid",
            )
        if "transition_receipt" in context:
            receipt_cid = _sha(
                _mapping(context["transition_receipt"], "transition_receipt").get("receipt_cid"),
                "transition_receipt.receipt_cid",
            )
        suffix = operation.replace(".", "-")
        if operation == OBSERVE_PLAN_R2_OPERATION:
            suffix = f"{suffix}:{ingress_slot}"
        return StateCommand(
            command_id=_safe_id(f"{authorization['request_id']}:{suffix}", "command_id"),
            command_kind=(
                CommandKind.MIGRATE if operation == APPLY_PLAN_R2_OPERATION else CommandKind.OBSERVE
            ),
            store_id=self.store_id,
            session_id=_safe_id(authorization["lease_id"], "lease_id"),
            expected_generation=_positive(authorization["owner_generation"], "owner_generation"),
            expected_revision=int(authorization["expected_version"]),
            fence_epoch=_positive(authorization["fencing_token"], "fencing_token"),
            idempotency_key=_safe_id(
                f"{authorization['idempotency_key']}:{suffix}",
                "idempotency_key",
            ),
            authority_class=StateAuthorityClass.AUTHORITATIVE,
            parameters={
                "interface": self.INTERFACE,
                "operation": operation,
                "authorization_cid": authorization["authorization_cid"],
                "statement_cid": authorization["statement_cid"],
                "operation_payload_cid": operation_payload_cid,
                "shard_id": self.shard_id,
                "store_id": self.store_id,
                "expected_event_cursor": authorization["expected_event_cursor"],
                "population_cid": authorization["population_cid"],
                "protected_tasks_root_cid": authorization["protected_tasks_root_cid"],
                "prepared_projection_cid": prepared_cid,
                "transition_receipt_cid": receipt_cid,
            },
        )

    def _build_envelope(
        self,
        *,
        operation: str,
        authorization: Mapping[str, Any],
        command: StateCommand,
        operation_payload_cid: str,
        ingress_slot: int,
    ) -> AuthorizedStateCommand:
        now_ms = _positive(self.clock_ms(), "clock_ms")
        deadline_ms = min(
            _positive(authorization["deadline_ms"], "deadline_ms"),
            _positive(authorization["expires_at_ms"], "expires_at_ms"),
        )
        suffix = operation.replace(".", "-")
        if operation == OBSERVE_PLAN_R2_OPERATION:
            suffix = f"{suffix}:{ingress_slot}"
        unsigned = authorized_state_command_signing_payload(
            request_id=_safe_id(f"{authorization['request_id']}:{suffix}", "request_id"),
            submission_id=_safe_id(
                f"{authorization['request_id']}:{suffix}:{operation_payload_cid}",
                "submission_id",
            ),
            ingress_slot=ingress_slot,
            principal_did=self.principal_did,
            approver_did=self.approver_did,
            authority_ref_cid=authorization["authorization_cid"],
            board_namespace=self.board_namespace,
            shard_id=self.shard_id,
            owner_principal_did=self.owner_principal_did,
            lease_id=_safe_id(authorization["lease_id"], "lease_id"),
            scope_id=_safe_id(authorization["plan_root_cid"], "scope_id"),
            effect=f"control-plane/{command.command_kind.value}",
            issued_at_ms=now_ms,
            expires_at_ms=min(deadline_ms, now_ms + 5 * 60 * 1000),
            deadline_ms=deadline_ms,
            one_use_nonce=_safe_id(f"{authorization['one_use_nonce']}:{suffix}", "one_use_nonce"),
            command=command,
        )
        signature = self.envelope_signer(unsigned)
        if not isinstance(signature, str) or not signature:
            raise PlanR2OwnerAdapterError("independent command approver did not return a signature")
        return seal_authorized_state_command(unsigned, approver_signature=signature)


__all__ = (
    "APPLY_PLAN_R2_OPERATION",
    "AuthorizedPlanR2OwnerGateway",
    "ExternalAgentStateRepository",
    "OBSERVE_PLAN_R2_OPERATION",
    "PLAN_R2_OWNER_GATEWAY_INTERFACE",
    "PLAN_R2_OWNER_OPERATION_SCHEMA",
    "PREPARE_PLAN_R2_OPERATION",
    "PlanR2OwnerAdapterError",
    "PlanR2OwnerAdapterUnavailable",
)
