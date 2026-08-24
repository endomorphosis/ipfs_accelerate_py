"""Fail-closed process-remote EAAEF execution-repository boundary.

This module implements the source-safe R1 prerequisites without pretending the
whole daemon launch is production-admitted.  It never accepts a dispatch
callback, database path, Portal, raw token, or untyped mapping as a live
gateway.  Positive construction requires a source-reverified signed per-birth
lane admission, independent verifier and gateway-source merge evidence, an
exact command-authorizer client, fixed Quack clients, and a durable
PlanRevisionStore exact-envelope journal.

The pure projection helpers are still useful at that future factory boundary:
they define the exact lane, task-operation, and dead-lane-recovery shapes that
the frozen borrowed-transaction adapter cross-joins at the sole Quack owner.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import socket
import stat
import struct
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ...llm_router import (
    AgentSupervisorNativeDependencyLaunch,
    verify_agent_supervisor_native_dependency_sealed_fd,
)
from ..control.profile_authority import LocalProfileTampered, verify_did_key_signature
from ..task_sources.control_plane_contracts import CommandKind, CommandOutcome, StateCommand
from ..task_sources.eaaef_borrowed_transaction import (
    EAAEF_DAEMON_LANE_BINDING_SCHEMA,
    EAAEF_DEAD_LANE_RECOVERY_AUTHORITY_SCHEMA,
    EAAEF_TASK_OPERATION_AUTHORITY_SCHEMA,
)
from ..task_sources.eaaef_typed_owner_service import (
    EAAEF_TYPED_OWNER_COMMAND_LOOKUP_OPERATION,
    EAAEF_TYPED_OWNER_COMMAND_SUBMIT_OPERATION,
    EAAEF_TYPED_OWNER_PUBLIC_RECEIPT_FIELDS,
    EAAEF_TYPED_OWNER_TRANSPORT_PRODUCTION_BLOCKER,
)
from ..task_sources.plan_revision_store import (
    PlanRevisionStore,
    PlanRevisionStoreError,
)
from ..task_sources.quack_command_authorization import (
    AuthorizedStateCommand,
    QuackCommandAuthorizationError,
    QuackCommandAuthorizationPolicy,
)
from ..task_sources.quack_command_fabric import QuackCommandClient, QuackReadClient
from ..task_sources.quack_daemon_gateway import (
    QUACK_DAEMON_GATEWAY_COMPONENT_INTERFACE,
    QuackDaemonCommandGateway,
    QuackDaemonGatewayError,
    quack_daemon_operation_command_vocabulary,
    quack_daemon_operation_intent,
    quack_daemon_operation_intent_from_envelope,
)
from ..task_sources.typed_state_owner import TypedStateOwnerConnection
from ..todo_daemon.external_agent_container_dispatcher import (
    ExternalAgentContainerDispatchError,
    ExternalAgentContainerWorkerDispatcher,
    ExternalAgentContainerWorkPacket,
)
from ..validation.agent_native_dependency_admission import (
    AgentSupervisorNativeDependencyAdmissionError,
    VerifiedAgentSupervisorNativeDependencyAdmission,
)
from ..validation.eaaef_bootstrap_gateway_launch import (
    EAAEFCommandAuthorizationServiceClient,
    VerifiedEAAEFBootstrapOperationalCapability,
    verify_eaaef_bootstrap_operation_submission,
)
from ..validation.eaaef_lane_gateway_admission import (
    EAAEFLaneGatewayAdmissionError,
    VerifiedEAAEFContainerDispatcherFactoryQualification,
    VerifiedEAAEFProcessBirth,
    VerifiedEAAEFQuackClientFactoryQualification,
)
from ..validation.eaaef_lane_gateway_admission import (
    VerifiedEAAEFExpiredLaneRecoveryAdmissionV2 as VerifiedEAAEFExpiredLaneRecoveryAdmission,
)
from ..validation.eaaef_lane_gateway_admission import (
    VerifiedEAAEFLaneRuntimeAdmissionV2 as VerifiedEAAEFLaneRuntimeAdmission,
)

EAAEF_BOOTSTRAP_EXECUTION_REPOSITORY_PROXY_INTERFACE: Final = (
    "EAAEFBootstrapExecutionRepositoryProxy@2"
)
EAAEF_BOOTSTRAP_EXECUTION_REPOSITORY_PROXY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-execution-repository-proxy@2"
)
EAAEF_BOOTSTRAP_RUNTIME_GATEWAY_QUALIFICATION_STATUS: Final = (
    "r1_source_verified_runtime_factory_implemented"
)
EAAEF_BOOTSTRAP_RUNTIME_GATEWAY_PRODUCTION_BLOCKERS: Final = (
    "signed_quack_client_factory_qualification_artifact_absent",
    "signed_dynamic_dispatcher_service_qualification_artifact_absent",
    "independently_signed_per_birth_lane_runtime_artifact_absent",
)

EAAEF_EXACT_ENVELOPE_JOURNAL_INTERFACE: Final = "EAAEFExactEnvelopeJournal@1"
EAAEF_EXACT_ENVELOPE_JOURNAL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-exact-envelope-journal@1"
)
EAAEF_BOOTSTRAP_COMMAND_TRANSPORT_INTERFACE: Final = "EAAEFBootstrapAuthorizedCommandTransport@1"
EAAEF_TYPED_OWNER_COMMAND_CLIENT_INTERFACE: Final = (
    "EAAEFTypedOwnerCommandClient@1"
)
EAAEF_TYPED_OWNER_COMMAND_TRANSPORT_INTERFACE: Final = (
    "EAAEFTypedOwnerCommandTransport@1"
)
EAAEF_BOOTSTRAP_COMMAND_GATEWAY_INTERFACE: Final = "EAAEFBootstrapCommandGateway@1"
EAAEF_SEALED_QUACK_SECRET_INTERFACE: Final = "EAAEFSealedQuackSecret@1"
EAAEF_SEALED_QUACK_SECRET_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-sealed-quack-secret@1"
)
EAAEF_SEALED_QUACK_CLIENT_DESCRIPTORS_INTERFACE: Final = (
    "EAAEFSealedQuackClientDescriptors@1"
)
EAAEF_CONTAINER_DYNAMIC_SERVICE_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-container-dynamic-service-request@1"
)
EAAEF_CONTAINER_DYNAMIC_SERVICE_RESPONSE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-container-dynamic-service-response@1"
)
EAAEF_CONTAINER_DISPATCHER_FACTORY_INTERFACE: Final = (
    "EAAEFContainerDispatcherFactory@1"
)

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,511}\Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_RECOVERY_LANES = 5
_MAX_RECOVERY_LIMIT = 1_000
_DAEMON_LANE_FIELDS: Final = frozenset(
    {
        "schema",
        "gateway_binding_cid",
        "owner_principal_did",
        "owner_session_id",
        "owner_generation",
        "lane_session_id",
        "lane_generation",
        "process_instance_id",
        "fence_epoch",
    }
)
_TASK_AUTHORITY_FIELDS: Final = frozenset(
    {
        "schema",
        "task_cid",
        "claim_id",
        "attempt_id",
        "attempt_number",
        "lease_id",
        "owner_session_id",
        "fencing_token",
        "fence_epoch",
        "daemon_lane_binding",
    }
)
_RECOVERY_AUTHORITY_FIELDS: Final = frozenset(
    {"schema", "purpose", "lane_bindings", "limit", "now_ms"}
)


class EAAEFBootstrapRuntimeGatewayError(RuntimeError):
    """A runtime proxy projection or construction attempt failed closed."""


class EAAEFBootstrapRuntimeGatewayNoGo(EAAEFBootstrapRuntimeGatewayError):
    """Positive runtime construction is not yet independently qualified."""

    def __init__(self) -> None:
        super().__init__(
            "eaaef_bootstrap_runtime_gateway_no_go:"
            + ",".join(EAAEF_BOOTSTRAP_RUNTIME_GATEWAY_PRODUCTION_BLOCKERS)
        )


class EAAEFBootstrapRuntimeGatewayAmbiguous(EAAEFBootstrapRuntimeGatewayError):
    """An exact envelope may have reached the owner but has no receipt yet."""


class EAAEFBootstrapRuntimeGatewayDiverged(EAAEFBootstrapRuntimeGatewayError):
    """Durable replay state differs from the requested command identity."""


class EAAEFBootstrapExcludedOperation(EAAEFBootstrapRuntimeGatewayError):
    """A generic/merge/Plan-R2 operation is outside bootstrap R1."""


def _canonical_detached(value: Any, noun: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise EAAEFBootstrapRuntimeGatewayError(f"{noun} is not canonical JSON") from exc


def _exact(value: object, fields: frozenset[str], noun: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EAAEFBootstrapRuntimeGatewayError(f"{noun} shape is not exact")
    detached = _canonical_detached(dict(value), noun)
    if not isinstance(detached, dict):
        raise EAAEFBootstrapRuntimeGatewayError(f"{noun} is not an object")
    return detached


def _identifier(value: object, noun: str) -> str:
    text = str(value or "")
    if _SAFE_ID.fullmatch(text) is None:
        raise EAAEFBootstrapRuntimeGatewayError(f"{noun} is not a bounded identifier")
    return text


def _sha(value: object, noun: str) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise EAAEFBootstrapRuntimeGatewayError(f"{noun} is not a full sha256 identity")
    return text


def _positive(value: object, noun: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EAAEFBootstrapRuntimeGatewayError(f"{noun} must be a positive integer")
    if maximum is not None and value > maximum:
        raise EAAEFBootstrapRuntimeGatewayError(f"{noun} exceeds its bound")
    return value


def eaaef_daemon_lane_binding_projection(
    value: Mapping[str, Any],
    *,
    verified_capability: VerifiedEAAEFBootstrapOperationalCapability,
) -> Mapping[str, Any]:
    """Validate one exact lane projection against capability @2.

    This is structural projection validation, not a signature or launch
    admission.  The positive runtime factory remains intentionally absent.
    """

    if type(verified_capability) is not VerifiedEAAEFBootstrapOperationalCapability:
        raise EAAEFBootstrapRuntimeGatewayError(
            "daemon lane binding requires a typed verified capability"
        )
    lane = _exact(value, _DAEMON_LANE_FIELDS, "daemon lane binding")
    capability = dict(verified_capability)
    normalized = {
        "schema": str(lane["schema"]),
        "gateway_binding_cid": _sha(lane["gateway_binding_cid"], "gateway_binding_cid"),
        "owner_principal_did": _identifier(lane["owner_principal_did"], "owner_principal_did"),
        "owner_session_id": _identifier(lane["owner_session_id"], "owner_session_id"),
        "owner_generation": _positive(lane["owner_generation"], "owner_generation"),
        "lane_session_id": _identifier(lane["lane_session_id"], "lane_session_id"),
        "lane_generation": _positive(lane["lane_generation"], "lane_generation"),
        "process_instance_id": _identifier(lane["process_instance_id"], "process_instance_id"),
        "fence_epoch": _positive(lane["fence_epoch"], "fence_epoch"),
    }
    if (
        normalized["schema"] != EAAEF_DAEMON_LANE_BINDING_SCHEMA
        or normalized["gateway_binding_cid"] != capability.get("gateway_binding_cid")
        or normalized["owner_principal_did"] != capability.get("owner_principal_did")
        or normalized["owner_session_id"] != capability.get("owner_session_id")
        or normalized["owner_generation"] != capability.get("owner_generation")
        or normalized["fence_epoch"] != capability.get("fence_epoch")
        or normalized["lane_session_id"] == normalized["owner_session_id"]
    ):
        raise EAAEFBootstrapRuntimeGatewayError(
            "daemon lane binding differs from the verified gateway owner"
        )
    return MappingProxyType(normalized)


def eaaef_task_operation_authority_projection(
    value: Mapping[str, Any],
    *,
    verified_capability: VerifiedEAAEFBootstrapOperationalCapability,
) -> Mapping[str, Any]:
    """Validate the exact task/claim/attempt/lease/lane projection."""

    authority = _exact(value, _TASK_AUTHORITY_FIELDS, "task operation authority")
    lane = eaaef_daemon_lane_binding_projection(
        authority["daemon_lane_binding"],
        verified_capability=verified_capability,
    )
    normalized = {
        "schema": str(authority["schema"]),
        "task_cid": _identifier(authority["task_cid"], "task_cid"),
        "claim_id": _identifier(authority["claim_id"], "claim_id"),
        "attempt_id": _identifier(authority["attempt_id"], "attempt_id"),
        "attempt_number": _positive(authority["attempt_number"], "attempt_number"),
        "lease_id": _identifier(authority["lease_id"], "lease_id"),
        "owner_session_id": _identifier(authority["owner_session_id"], "owner_session_id"),
        "fencing_token": _positive(authority["fencing_token"], "fencing_token"),
        "fence_epoch": _positive(authority["fence_epoch"], "fence_epoch"),
        "daemon_lane_binding": dict(lane),
    }
    if (
        normalized["schema"] != EAAEF_TASK_OPERATION_AUTHORITY_SCHEMA
        or normalized["owner_session_id"] != lane["lane_session_id"]
        or normalized["fence_epoch"] != lane["fence_epoch"]
    ):
        raise EAAEFBootstrapRuntimeGatewayError(
            "task operation authority differs from its daemon lane"
        )
    return MappingProxyType(normalized)


def eaaef_dead_lane_recovery_arguments(
    *,
    lane_bindings: Sequence[Mapping[str, Any]],
    limit: int,
    now_ms: int,
    verified_capability: VerifiedEAAEFBootstrapOperationalCapability,
) -> Mapping[str, Any]:
    """Build only the admitted read-only expired-lane listing arguments."""

    lanes = [
        dict(
            eaaef_daemon_lane_binding_projection(
                item,
                verified_capability=verified_capability,
            )
        )
        for item in lane_bindings
    ]
    if not 1 <= len(lanes) <= _MAX_RECOVERY_LANES:
        raise EAAEFBootstrapRuntimeGatewayError(
            "dead-lane recovery requires between one and five exact lanes"
        )
    lane_session_ids = {item["lane_session_id"] for item in lanes}
    if len(lane_session_ids) != len(lanes):
        raise EAAEFBootstrapRuntimeGatewayError(
            "dead-lane recovery contains a duplicate lane_session_id"
        )
    recovery = {
        "schema": EAAEF_DEAD_LANE_RECOVERY_AUTHORITY_SCHEMA,
        "purpose": "expired_lane_retirement",
        "lane_bindings": lanes,
        "limit": _positive(limit, "recovery limit", maximum=_MAX_RECOVERY_LIMIT),
        "now_ms": _positive(now_ms, "recovery now_ms"),
    }
    if set(recovery) != _RECOVERY_AUTHORITY_FIELDS:
        raise EAAEFBootstrapRuntimeGatewayError("dead-lane recovery authority shape changed")
    return MappingProxyType({"recovery_authority": recovery})


def _canonical_bytes(value: Any, noun: str) -> bytes:
    detached = _canonical_detached(value, noun)
    return json.dumps(
        detached,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _content_cid(value: Any, noun: str) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value, noun)).hexdigest()


def _policy_from_capability(
    capability: VerifiedEAAEFBootstrapOperationalCapability,
) -> QuackCommandAuthorizationPolicy:
    raw = capability.get("authorization_policy")
    if not isinstance(raw, Mapping):
        raise EAAEFBootstrapRuntimeGatewayError("verified capability has no authorization policy")
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
            trusted_approver_dids=frozenset(value.get("trusted_approver_dids") or ()),
            authorized_principal_dids=frozenset(value.get("authorized_principal_dids") or ()),
            allowed_command_kinds=frozenset(
                CommandKind(item) for item in value.get("allowed_command_kinds") or ()
            ),
            maximum_authorization_lifetime_ms=value.get("maximum_authorization_lifetime_ms"),
        )
    except (QuackCommandAuthorizationError, TypeError, ValueError) as exc:
        raise EAAEFBootstrapRuntimeGatewayError(
            "verified capability authorization policy is invalid"
        ) from exc
    if policy.to_dict() != value:
        raise EAAEFBootstrapRuntimeGatewayError(
            "verified capability authorization policy is not canonical"
        )
    return policy


_JOURNAL_FACTORY_TOKEN = object()


class EAAEFExactEnvelopeJournal:
    """Durable create-once envelope and receipt journal for one lane birth.

    ``PlanRevisionStore`` supplies its crash-safe CAS and continuation files.
    Every read/write is additionally serialized by that store's process/file
    locks and the store directories are rechecked against their initial inode
    identities before use.  A continuation is never overwritten by a
    different intent or envelope.
    """

    INTERFACE: ClassVar[str] = EAAEF_EXACT_ENVELOPE_JOURNAL_INTERFACE
    SCHEMA: ClassVar[str] = EAAEF_EXACT_ENVELOPE_JOURNAL_SCHEMA

    __slots__ = (
        "_store",
        "_admission_cid",
        "_lane_authority_cid",
        "_journal_namespace",
        "_path_identities",
    )

    def __init__(
        self,
        token: object,
        *,
        store: PlanRevisionStore,
        admission: VerifiedEAAEFLaneRuntimeAdmission,
    ) -> None:
        if token is not _JOURNAL_FACTORY_TOKEN:
            raise TypeError("EAAEF journals come from the safe store factory")
        if type(store) is not PlanRevisionStore:
            raise EAAEFBootstrapRuntimeGatewayError(
                "exact PlanRevisionStore is required for the envelope journal"
            )
        if type(admission) is not VerifiedEAAEFLaneRuntimeAdmission:
            raise EAAEFBootstrapRuntimeGatewayError(
                "typed source-verified lane admission is required"
            )
        self._store = store
        self._admission_cid = str(admission["merge_admission_cid"])
        self._lane_authority_cid = str(admission["lane_authority_cid"])
        self._journal_namespace = str(admission["journal_namespace"])
        paths = (store.root, store.cas_dir, store.continuations_dir)
        self._path_identities = tuple(self._directory_identity(path) for path in paths)

    @staticmethod
    def _directory_identity(path: Path) -> tuple[int, int, int, int]:
        try:
            metadata = os.lstat(path)
        except OSError as exc:
            raise EAAEFBootstrapRuntimeGatewayError(
                "envelope journal directory is unavailable"
            ) from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "envelope journal directory ownership is unsafe"
            )
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_uid,
            stat.S_IFMT(metadata.st_mode),
        )

    def _assert_store_identity(self) -> None:
        observed = tuple(
            self._directory_identity(path)
            for path in (
                self._store.root,
                self._store.cas_dir,
                self._store.continuations_dir,
            )
        )
        if observed != self._path_identities:
            raise EAAEFBootstrapRuntimeGatewayDiverged(
                "envelope journal directory identity changed"
            )

    def _continuation_key(self, operation_key: str) -> str:
        key = _sha(operation_key, "operation_key")
        digest = hashlib.sha256(f"{self._lane_authority_cid}\0{key}".encode()).hexdigest()
        return f"eaaef-envelope-{digest}"

    def _pending_key(self) -> str:
        digest = hashlib.sha256(f"{self._lane_authority_cid}\0pending".encode()).hexdigest()
        return f"eaaef-pending-{digest}"

    def _validate_state(
        self,
        value: Mapping[str, Any],
        *,
        operation_key: str,
        operation: str,
        intent_cid: str,
    ) -> dict[str, Any]:
        fields = {
            "schema",
            "lane_authority_cid",
            "admission_cid",
            "journal_namespace",
            "operation_key",
            "operation",
            "intent_cid",
            "envelope_cid",
            "envelope_cas_cid",
            "phase",
            "receipt_cas_cid",
        }
        state = _exact(value, frozenset(fields), "exact-envelope continuation")
        if (
            state["schema"] != self.SCHEMA
            or state["lane_authority_cid"] != self._lane_authority_cid
            or state["admission_cid"] != self._admission_cid
            or state["journal_namespace"] != self._journal_namespace
            or state["operation_key"] != operation_key
            or state["operation"] != operation
            or state["intent_cid"] != intent_cid
            or state["phase"] not in {"eaaef_prepared", "eaaef_committed"}
            or (state["phase"] == "eaaef_prepared" and state["receipt_cas_cid"])
            or (state["phase"] == "eaaef_committed" and not state["receipt_cas_cid"])
        ):
            raise EAAEFBootstrapRuntimeGatewayDiverged(
                "durable exact-envelope continuation diverged"
            )
        _sha(state["operation_key"], "journal operation_key")
        _identifier(state["operation"], "journal operation")
        _sha(state["intent_cid"], "journal intent_cid")
        _sha(state["envelope_cid"], "journal envelope_cid")
        _identifier(state["envelope_cas_cid"], "journal envelope_cas_cid")
        if state["receipt_cas_cid"]:
            _identifier(state["receipt_cas_cid"], "journal receipt_cas_cid")
        return state

    def _envelope_from_state(self, state: Mapping[str, Any]) -> AuthorizedStateCommand:
        try:
            envelope = AuthorizedStateCommand.from_dict(
                self._store.get_cas(str(state["envelope_cas_cid"]))
            )
        except (
            KeyError,
            PlanRevisionStoreError,
            QuackCommandAuthorizationError,
        ) as exc:
            raise EAAEFBootstrapRuntimeGatewayDiverged(
                "journaled command envelope is corrupt"
            ) from exc
        if (
            type(envelope) is not AuthorizedStateCommand
            or type(envelope.command) is not StateCommand
            or envelope.envelope_cid != state["envelope_cid"]
        ):
            raise EAAEFBootstrapRuntimeGatewayDiverged(
                "journaled command envelope identity changed"
            )
        return envelope

    def _load_pending_locked(
        self,
    ) -> tuple[AuthorizedStateCommand, dict[str, Any]] | None:
        continuation = self._store.load_continuation(self._pending_key())
        if continuation is None:
            return None
        operation_key = str(continuation.get("operation_key") or "")
        operation = str(continuation.get("operation") or "")
        intent_cid = str(continuation.get("intent_cid") or "")
        state = self._validate_state(
            continuation,
            operation_key=operation_key,
            operation=operation,
            intent_cid=intent_cid,
        )
        if state["phase"] != "eaaef_prepared":
            raise EAAEFBootstrapRuntimeGatewayDiverged(
                "lane pending pointer is not a prepared exact envelope"
            )
        return self._envelope_from_state(state), state

    def _load_locked(
        self,
        *,
        operation_key: str,
        operation: str,
        intent_cid: str,
    ) -> tuple[AuthorizedStateCommand, Mapping[str, Any] | None] | None:
        continuation = self._store.load_continuation(self._continuation_key(operation_key))
        if continuation is None:
            return None
        state = self._validate_state(
            continuation,
            operation_key=operation_key,
            operation=operation,
            intent_cid=intent_cid,
        )
        envelope = self._envelope_from_state(state)
        receipt = None
        if state["phase"] == "eaaef_committed":
            try:
                receipt = self._store.get_cas(str(state["receipt_cas_cid"]))
            except (KeyError, PlanRevisionStoreError) as exc:
                raise EAAEFBootstrapRuntimeGatewayDiverged(
                    "journaled command receipt is corrupt"
                ) from exc
        return envelope, receipt

    def lookup(
        self,
        *,
        operation_key: str,
        operation: str,
        intent_cid: str,
    ) -> tuple[AuthorizedStateCommand, Mapping[str, Any] | None] | None:
        self._assert_store_identity()
        with self._store._thread_lock, self._store._guard():
            existing = self._load_locked(
                operation_key=operation_key,
                operation=operation,
                intent_cid=intent_cid,
            )
            pending = self._load_pending_locked()
            if existing is not None:
                if pending is not None:
                    pending_envelope, pending_state = pending
                    pending_is_current = (
                        pending_state["operation_key"] == operation_key
                        and pending_state["operation"] == operation
                        and pending_state["intent_cid"] == intent_cid
                    )
                    if existing[1] is None and (
                        not pending_is_current
                        or pending_envelope.to_dict() != existing[0].to_dict()
                    ):
                        raise EAAEFBootstrapRuntimeGatewayDiverged(
                            "lane pending pointer differs from its operation record"
                        )
                    if existing[1] is not None and pending_is_current:
                        if pending_envelope.to_dict() != existing[0].to_dict():
                            raise EAAEFBootstrapRuntimeGatewayDiverged(
                                "lane pending pointer differs from committed envelope"
                            )
                        self._store.clear_continuation(self._pending_key())
                return existing
            if pending is None:
                return None
            pending_envelope, pending_state = pending
            pending_record = self._load_locked(
                operation_key=str(pending_state["operation_key"]),
                operation=str(pending_state["operation"]),
                intent_cid=str(pending_state["intent_cid"]),
            )
            if pending_record is not None:
                if pending_record[0].to_dict() != pending_envelope.to_dict():
                    raise EAAEFBootstrapRuntimeGatewayDiverged(
                        "lane pending pointer differs from its operation record"
                    )
            if pending_record is not None and pending_record[1] is not None:
                # Heal a crash after the committed record reached disk but
                # before the secondary lane pointer was cleared.
                self._store.clear_continuation(self._pending_key())
                return None
            if (
                pending_state["operation_key"] != operation_key
                or pending_state["operation"] != operation
                or pending_state["intent_cid"] != intent_cid
            ):
                raise EAAEFBootstrapRuntimeGatewayAmbiguous(
                    "another exact lane envelope remains unresolved"
                )
            # Heal a crash between the lane pointer and per-operation record.
            self._store.put_continuation(self._continuation_key(operation_key), pending_state)
            return pending_envelope, None

    def pending(
        self,
    ) -> tuple[str, str, str, AuthorizedStateCommand] | None:
        """Return the sole unresolved lane envelope, healing stale pointers."""

        self._assert_store_identity()
        with self._store._thread_lock, self._store._guard():
            pending = self._load_pending_locked()
            if pending is None:
                return None
            envelope, state = pending
            operation_key = str(state["operation_key"])
            operation = str(state["operation"])
            intent_cid = str(state["intent_cid"])
            indexed = self._load_locked(
                operation_key=operation_key,
                operation=operation,
                intent_cid=intent_cid,
            )
            if indexed is not None:
                if indexed[0].to_dict() != envelope.to_dict():
                    raise EAAEFBootstrapRuntimeGatewayDiverged(
                        "lane pending pointer differs from its operation record"
                    )
                if indexed[1] is not None:
                    self._store.clear_continuation(self._pending_key())
                    return None
            return operation_key, operation, intent_cid, envelope

    def prepare(
        self,
        *,
        operation_key: str,
        operation: str,
        intent_cid: str,
        envelope: AuthorizedStateCommand,
    ) -> AuthorizedStateCommand:
        if type(envelope) is not AuthorizedStateCommand:
            raise EAAEFBootstrapRuntimeGatewayError(
                "journal requires exact AuthorizedStateCommand base type"
            )
        self._assert_store_identity()
        with self._store._thread_lock, self._store._guard():
            existing = self._load_locked(
                operation_key=operation_key,
                operation=operation,
                intent_cid=intent_cid,
            )
            pending = self._load_pending_locked()
            if existing is not None:
                prior, prior_receipt = existing
                if prior.to_dict() != envelope.to_dict():
                    raise EAAEFBootstrapRuntimeGatewayDiverged(
                        "operation key already owns a different exact envelope"
                    )
                if (
                    prior_receipt is None
                    and pending is not None
                    and pending[0].to_dict() != prior.to_dict()
                ):
                    raise EAAEFBootstrapRuntimeGatewayDiverged(
                        "lane pending pointer owns a different exact envelope"
                    )
                return prior
            if pending is not None:
                pending_envelope, pending_state = pending
                if (
                    pending_state["operation_key"] != operation_key
                    or pending_state["operation"] != operation
                    or pending_state["intent_cid"] != intent_cid
                ):
                    raise EAAEFBootstrapRuntimeGatewayAmbiguous(
                        "another exact lane envelope remains unresolved"
                    )
                if pending_envelope.to_dict() != envelope.to_dict():
                    raise EAAEFBootstrapRuntimeGatewayDiverged(
                        "prepared operation tried to replace its exact envelope"
                    )
                self._store.put_continuation(self._continuation_key(operation_key), pending_state)
                return pending_envelope
            envelope_cas_cid = self._store.put_cas(envelope.to_dict())
            state = {
                "schema": self.SCHEMA,
                "lane_authority_cid": self._lane_authority_cid,
                "admission_cid": self._admission_cid,
                "journal_namespace": self._journal_namespace,
                "operation_key": _sha(operation_key, "operation_key"),
                "operation": _identifier(operation, "operation"),
                "intent_cid": _sha(intent_cid, "intent_cid"),
                "envelope_cid": envelope.envelope_cid,
                "envelope_cas_cid": envelope_cas_cid,
                # Namespaced phases are deliberately outside PlanRevisionStore's
                # own apply-state recovery vocabulary.
                "phase": "eaaef_prepared",
                "receipt_cas_cid": "",
            }
            # The lane pointer is first: a crash can lose the secondary index,
            # never the fact that this exact envelope is already authoritative.
            self._store.put_continuation(self._pending_key(), state)
            self._store.put_continuation(self._continuation_key(operation_key), state)
            return envelope

    def commit_receipt(
        self,
        *,
        operation_key: str,
        operation: str,
        intent_cid: str,
        envelope: AuthorizedStateCommand,
        receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        detached = _canonical_detached(dict(receipt), "owner command receipt")
        if (
            detached.get("submission_id") != envelope.submission_id
            or detached.get("envelope_cid") != envelope.envelope_cid
        ):
            raise EAAEFBootstrapRuntimeGatewayDiverged(
                "receipt identity differs from its prepared exact envelope"
            )
        self._assert_store_identity()
        with self._store._thread_lock, self._store._guard():
            existing = self._load_locked(
                operation_key=operation_key,
                operation=operation,
                intent_cid=intent_cid,
            )
            if existing is None or existing[0].to_dict() != envelope.to_dict():
                raise EAAEFBootstrapRuntimeGatewayDiverged("receipt has no exact prepared envelope")
            prior_receipt = existing[1]
            pending = self._load_pending_locked()
            pending_is_current = pending is not None and (
                pending[1]["operation_key"] == operation_key
                and pending[1]["operation"] == operation
                and pending[1]["intent_cid"] == intent_cid
            )
            if (
                prior_receipt is None
                and pending is not None
                and (not pending_is_current or pending[0].to_dict() != envelope.to_dict())
            ):
                raise EAAEFBootstrapRuntimeGatewayDiverged(
                    "lane pending pointer differs from prepared envelope"
                )
            if prior_receipt is not None:
                if dict(prior_receipt) != detached:
                    raise EAAEFBootstrapRuntimeGatewayDiverged("owner receipt replay diverged")
                if pending_is_current:
                    assert pending is not None
                    if pending[0].to_dict() != envelope.to_dict():
                        raise EAAEFBootstrapRuntimeGatewayDiverged(
                            "lane pending pointer differs from committed envelope"
                        )
                    self._store.clear_continuation(self._pending_key())
                return prior_receipt
            receipt_cas_cid = self._store.put_cas(detached)
            state = {
                "schema": self.SCHEMA,
                "lane_authority_cid": self._lane_authority_cid,
                "admission_cid": self._admission_cid,
                "journal_namespace": self._journal_namespace,
                "operation_key": operation_key,
                "operation": operation,
                "intent_cid": intent_cid,
                "envelope_cid": envelope.envelope_cid,
                "envelope_cas_cid": self._store.put_cas(envelope.to_dict()),
                "phase": "eaaef_committed",
                "receipt_cas_cid": receipt_cas_cid,
            }
            self._store.put_continuation(self._continuation_key(operation_key), state)
            if pending is not None:
                self._store.clear_continuation(self._pending_key())
            return MappingProxyType(detached)

    def clear_observation(
        self,
        *,
        operation_key: str,
        operation: str,
        intent_cid: str,
    ) -> None:
        self._assert_store_identity()
        with self._store._thread_lock, self._store._guard():
            existing = self._load_locked(
                operation_key=operation_key,
                operation=operation,
                intent_cid=intent_cid,
            )
            if existing is None or existing[1] is None:
                raise EAAEFBootstrapRuntimeGatewayDiverged(
                    "observation cannot clear before its receipt is durable"
                )
            self._store.clear_continuation(self._continuation_key(operation_key))


def eaaef_exact_envelope_journal_relative_path(
    admission: VerifiedEAAEFLaneRuntimeAdmission,
) -> Path:
    """Derive the only journal path for one signed lane birth."""

    if type(admission) is not VerifiedEAAEFLaneRuntimeAdmission:
        raise EAAEFBootstrapRuntimeGatewayError(
            "journal factory requires exact source-verified lane admission"
        )
    namespace = _identifier(admission["journal_namespace"], "journal_namespace")
    lane_cid = _sha(admission["lane_authority_cid"], "lane_authority_cid")
    return Path(
        "eaaef-envelope-journal-" + hashlib.sha256(f"{namespace}\0{lane_cid}".encode()).hexdigest()
    )


def open_eaaef_exact_envelope_journal(
    parent_directory: str | Path,
    *,
    admission: VerifiedEAAEFLaneRuntimeAdmission,
) -> EAAEFExactEnvelopeJournal:
    """Open one owner-only per-lane PlanRevisionStore journal."""

    directory_name = eaaef_exact_envelope_journal_relative_path(admission).name
    parent = Path(os.path.abspath(os.fspath(parent_directory)))
    try:
        metadata = os.lstat(parent)
    except OSError as exc:
        raise EAAEFBootstrapRuntimeGatewayError("envelope journal parent is unavailable") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise EAAEFBootstrapRuntimeGatewayError("envelope journal parent ownership is unsafe")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_flag is None:
        raise EAAEFBootstrapRuntimeGatewayError(
            "envelope journal requires nofollow directory support"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC | nofollow | directory_flag

    def validate_existing_tree(descriptor: int, *, depth: int = 0) -> int:
        if depth > 8:
            raise EAAEFBootstrapRuntimeGatewayError("envelope journal tree exceeds its depth bound")
        count = 0
        for name in os.listdir(descriptor):
            count += 1
            if count > 100_000:
                raise EAAEFBootstrapRuntimeGatewayError(
                    "envelope journal tree exceeds its entry bound"
                )
            try:
                child = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as exc:
                raise EAAEFBootstrapRuntimeGatewayError(
                    "envelope journal entry changed during preflight"
                ) from exc
            if (
                stat.S_ISLNK(child.st_mode)
                or child.st_uid != os.geteuid()
                or stat.S_IMODE(child.st_mode) & 0o022
            ):
                raise EAAEFBootstrapRuntimeGatewayError("envelope journal contains an unsafe entry")
            if stat.S_ISDIR(child.st_mode):
                try:
                    child_fd = os.open(name, flags, dir_fd=descriptor)
                except OSError as exc:
                    raise EAAEFBootstrapRuntimeGatewayError(
                        "envelope journal directory changed during preflight"
                    ) from exc
                try:
                    count += validate_existing_tree(child_fd, depth=depth + 1)
                finally:
                    os.close(child_fd)
            elif not stat.S_ISREG(child.st_mode) or child.st_nlink != 1:
                raise EAAEFBootstrapRuntimeGatewayError(
                    "envelope journal contains a non-regular entry"
                )
        return count

    try:
        parent_fd = os.open("/", flags)
        for part in parent.parts[1:]:
            try:
                next_fd = os.open(part, flags, dir_fd=parent_fd)
            except OSError:
                os.close(parent_fd)
                raise
            os.close(parent_fd)
            parent_fd = next_fd
    except OSError as exc:
        raise EAAEFBootstrapRuntimeGatewayError(
            "envelope journal parent cannot be opened safely"
        ) from exc
    try:
        opened_parent = os.fstat(parent_fd)
        current_parent = os.stat(parent, follow_symlinks=False)
        parent_identity = lambda item: (  # noqa: E731 - immutable stat identity.
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_uid,
        )
        if (
            parent_identity(metadata) != parent_identity(opened_parent)
            or parent_identity(metadata) != parent_identity(current_parent)
            or stat.S_ISLNK(current_parent.st_mode)
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "envelope journal parent changed during safe open"
            )
        try:
            os.mkdir(directory_name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        try:
            journal_fd = os.open(directory_name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise EAAEFBootstrapRuntimeGatewayError(
                "envelope journal lane directory is unavailable"
            ) from exc
        try:
            lane_metadata = os.fstat(journal_fd)
            pathname = os.stat(directory_name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISDIR(lane_metadata.st_mode)
                or stat.S_ISLNK(pathname.st_mode)
                or lane_metadata.st_dev != pathname.st_dev
                or lane_metadata.st_ino != pathname.st_ino
                or lane_metadata.st_uid != os.geteuid()
                or stat.S_IMODE(lane_metadata.st_mode) & 0o077
            ):
                raise EAAEFBootstrapRuntimeGatewayError(
                    "envelope journal lane directory ownership is unsafe"
                )
            validate_existing_tree(journal_fd)
        finally:
            os.close(journal_fd)
    finally:
        os.close(parent_fd)
    lane_path = parent / directory_name
    try:
        lock_flags = os.O_WRONLY | os.O_CREAT | os.O_CLOEXEC | nofollow
        lock_fd = os.open(
            lane_path / ".plan-revision-store.lock",
            lock_flags,
            0o600,
        )
        try:
            os.fchmod(lock_fd, 0o600)
        finally:
            os.close(lock_fd)
        store = PlanRevisionStore(lane_path)
        for child in (
            store.cas_dir,
            store.intents_dir,
            store.continuations_dir,
            store.quarantine_dir,
            store.backups_dir,
        ):
            os.chmod(child, 0o700)
    except (OSError, PlanRevisionStoreError) as exc:
        raise EAAEFBootstrapRuntimeGatewayError(
            "PlanRevisionStore envelope journal failed to open"
        ) from exc
    return EAAEFExactEnvelopeJournal(
        _JOURNAL_FACTORY_TOKEN,
        store=store,
        admission=admission,
    )


_SEALED_SECRET_DESCRIPTOR_TOKEN = object()
_SEALED_CLIENT_DESCRIPTORS_TOKEN = object()


class EAAEFSealedQuackSecretDescriptor:
    """Parent-owned write-sealed memfd; its public surface has no token."""

    __slots__ = ("_descriptor", "_sha256", "_purpose", "_closed")

    def __init__(self, token: object, descriptor: int, sha256: str, purpose: str) -> None:
        if token is not _SEALED_SECRET_DESCRIPTOR_TOKEN:
            raise TypeError("sealed Quack descriptors come from the exact memfd creator")
        self._descriptor = descriptor
        self._sha256 = _sha(sha256, "sealed secret descriptor sha256")
        self._purpose = purpose
        self._closed = False

    @property
    def descriptor(self) -> int:
        if self._closed:
            raise EAAEFBootstrapRuntimeGatewayError("sealed Quack descriptor is closed")
        return self._descriptor

    @property
    def sha256(self) -> str:
        return self._sha256

    @property
    def purpose(self) -> str:
        return self._purpose

    @property
    def pass_fds(self) -> tuple[int, ...]:
        return (self.descriptor,)

    def close(self) -> None:
        if not self._closed:
            os.close(self._descriptor)
            self._closed = True


def _transport_token(value: object) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not 32 <= len(value.encode("utf-8")) <= 512
        or any(character.isspace() for character in value)
    ):
        raise EAAEFBootstrapRuntimeGatewayError(
            "Quack token must be high-entropy transport material"
        )
    return value


def create_eaaef_sealed_quack_secret_descriptor(
    *,
    operational_capability: VerifiedEAAEFBootstrapOperationalCapability,
    purpose: str,
    lane_session_id: str,
    lane_generation: int,
    process_instance_id: str,
    process_birth_nonce: str,
    secret_generation: int,
    token: str,
) -> EAAEFSealedQuackSecretDescriptor:
    """Seal one parent-resolved token for inherited-FD delivery only."""

    if type(operational_capability) is not VerifiedEAAEFBootstrapOperationalCapability:
        raise EAAEFBootstrapRuntimeGatewayError(
            "sealed secret creation requires a verified operational capability"
        )
    if purpose not in {"command", "state"}:
        raise EAAEFBootstrapRuntimeGatewayError("sealed secret purpose is invalid")
    endpoint_field = "command_endpoint" if purpose == "command" else "state_endpoint"
    handle_field = "command_secret_handle" if purpose == "command" else "state_secret_handle"
    body = {
        "schema": EAAEF_SEALED_QUACK_SECRET_SCHEMA,
        "interface": EAAEF_SEALED_QUACK_SECRET_INTERFACE,
        "purpose": purpose,
        "operational_capability_cid": _sha(
            operational_capability["capability_cid"], "operational capability CID"
        ),
        "gateway_binding_cid": _sha(
            operational_capability["gateway_binding_cid"], "gateway binding CID"
        ),
        "lane_session_id": _identifier(lane_session_id, "lane_session_id"),
        "lane_generation": _positive(lane_generation, "lane_generation"),
        "process_instance_id": _identifier(process_instance_id, "process_instance_id"),
        "process_birth_nonce": _identifier(process_birth_nonce, "process_birth_nonce"),
        "endpoint": str(operational_capability[endpoint_field]),
        "secret_handle": _identifier(
            operational_capability[handle_field], f"{purpose} secret handle"
        ),
        "secret_generation": _positive(secret_generation, "secret_generation"),
        "descriptor_nonce": secrets.token_hex(32),
        "token": _transport_token(token),
    }
    raw = _canonical_bytes(body, "sealed Quack secret")
    if not hasattr(os, "memfd_create") or not hasattr(fcntl, "F_ADD_SEALS"):
        raise EAAEFBootstrapRuntimeGatewayError(
            "sealed Quack descriptor delivery requires Linux memfd sealing"
        )
    flags = int(getattr(os, "MFD_CLOEXEC", 0)) | int(getattr(os, "MFD_ALLOW_SEALING", 0))
    descriptor = os.memfd_create(f"eaaef-quack-{purpose}-secret", flags=flags)
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.lseek(descriptor, 0, os.SEEK_SET)
        seals = (
            fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_SEAL
        )
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
    except BaseException:
        os.close(descriptor)
        raise
    return EAAEFSealedQuackSecretDescriptor(
        _SEALED_SECRET_DESCRIPTOR_TOKEN,
        descriptor,
        "sha256:" + hashlib.sha256(raw).hexdigest(),
        purpose,
    )


def _read_sealed_quack_secret(descriptor: int, *, expected_sha256: str) -> dict[str, Any]:
    if isinstance(descriptor, bool) or not isinstance(descriptor, int) or descriptor < 0:
        raise EAAEFBootstrapRuntimeGatewayError("sealed secret descriptor must be an fd")
    required_seals = (
        fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
    )
    try:
        metadata = os.fstat(descriptor)
        observed_seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
        duplicate = os.dup(descriptor)
    except OSError as exc:
        raise EAAEFBootstrapRuntimeGatewayError(
            "sealed secret descriptor is unavailable"
        ) from exc
    try:
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_size <= 0
            or metadata.st_size > 16_384
            or observed_seals & required_seals != required_seals
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "sealed secret descriptor is not an immutable owner memfd"
            )
        os.lseek(duplicate, 0, os.SEEK_SET)
        raw = b""
        while len(raw) < metadata.st_size:
            chunk = os.read(duplicate, metadata.st_size - len(raw))
            if not chunk:
                break
            raw += chunk
    finally:
        os.close(duplicate)
    if len(raw) != metadata.st_size or (
        "sha256:" + hashlib.sha256(raw).hexdigest()
    ) != _sha(expected_sha256, "expected sealed secret descriptor sha256"):
        raise EAAEFBootstrapRuntimeGatewayError(
            "sealed secret descriptor identity differs from signed v2 admission"
        )
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EAAEFBootstrapRuntimeGatewayError(
            "sealed secret descriptor is not canonical JSON"
        ) from exc
    expected = {
        "schema",
        "interface",
        "purpose",
        "operational_capability_cid",
        "gateway_binding_cid",
        "lane_session_id",
        "lane_generation",
        "process_instance_id",
        "process_birth_nonce",
        "endpoint",
        "secret_handle",
        "secret_generation",
        "descriptor_nonce",
        "token",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or raw != _canonical_bytes(value, "sealed Quack secret")
        or value.get("schema") != EAAEF_SEALED_QUACK_SECRET_SCHEMA
        or value.get("interface") != EAAEF_SEALED_QUACK_SECRET_INTERFACE
    ):
        raise EAAEFBootstrapRuntimeGatewayError(
            "sealed secret descriptor shape is not exact"
        )
    return value


class EAAEFSealedQuackClientDescriptors:
    """Child-side exact handle resolution; token values remain private."""

    __slots__ = ("_command", "_state", "_admission_cid", "_birth_cid", "_used")

    def __init__(
        self,
        token: object,
        *,
        command: Mapping[str, Any],
        state: Mapping[str, Any],
        admission: VerifiedEAAEFLaneRuntimeAdmission,
        process_birth: VerifiedEAAEFProcessBirth,
    ) -> None:
        if token is not _SEALED_CLIENT_DESCRIPTORS_TOKEN:
            raise TypeError("sealed Quack client descriptors come from the exact binder")
        self._command = dict(command)
        self._state = dict(state)
        self._admission_cid = str(admission["merge_admission_cid"])
        self._birth_cid = str(process_birth["birth_cid"])
        self._used = False

    def close(self) -> None:
        """Erase resolved token material when a factory is abandoned."""

        self._command.clear()
        self._state.clear()
        self._used = True


def bind_eaaef_sealed_quack_client_descriptors(
    *,
    admission: VerifiedEAAEFLaneRuntimeAdmission,
    process_birth: VerifiedEAAEFProcessBirth,
    command_descriptor: int,
    state_descriptor: int,
) -> EAAEFSealedQuackClientDescriptors:
    """Resolve only the two signed opaque handles from immutable inherited FDs."""

    if (
        type(admission) is not VerifiedEAAEFLaneRuntimeAdmission
        or type(process_birth) is not VerifiedEAAEFProcessBirth
        or process_birth.admission_cid != admission["merge_admission_cid"]
        or command_descriptor == state_descriptor
    ):
        raise EAAEFBootstrapRuntimeGatewayError(
            "sealed client descriptor binder requires one exact verified birth"
        )
    command = _read_sealed_quack_secret(
        command_descriptor,
        expected_sha256=str(admission["command_secret_descriptor_sha256"]),
    )
    state = _read_sealed_quack_secret(
        state_descriptor,
        expected_sha256=str(admission["state_secret_descriptor_sha256"]),
    )
    capability = admission.operational_capability
    common = {
        "operational_capability_cid": admission["operational_capability_cid"],
        "gateway_binding_cid": admission["gateway_binding_cid"],
        "lane_session_id": admission["lane_session_id"],
        "lane_generation": admission["lane_generation"],
        "process_instance_id": admission["process_instance_id"],
        "process_birth_nonce": admission["process_birth_nonce"],
    }
    expectations = {
        "command": {
            **common,
            "purpose": "command",
            "endpoint": capability["command_endpoint"],
            "secret_handle": capability["command_secret_handle"],
            "secret_generation": admission["command_secret_generation"],
        },
        "state": {
            **common,
            "purpose": "state",
            "endpoint": capability["state_endpoint"],
            "secret_handle": capability["state_secret_handle"],
            "secret_generation": admission["state_secret_generation"],
        },
    }
    for purpose, value in (("command", command), ("state", state)):
        if any(value.get(name) != expected for name, expected in expectations[purpose].items()):
            raise EAAEFBootstrapRuntimeGatewayError(
                f"sealed {purpose} descriptor differs from signed lane/capability"
            )
        _identifier(value.get("descriptor_nonce"), f"{purpose} descriptor nonce")
        _transport_token(value.get("token"))
    if command["token"] == state["token"] or command["secret_handle"] == state["secret_handle"]:
        raise EAAEFBootstrapRuntimeGatewayError(
            "command and state sealed handles must be distinct"
        )
    return EAAEFSealedQuackClientDescriptors(
        _SEALED_CLIENT_DESCRIPTORS_TOKEN,
        command=command,
        state=state,
        admission=admission,
        process_birth=process_birth,
    )


_QUALIFIED_QUACK_CLIENTS_TOKEN = object()


class EAAEFQualifiedQuackClients:
    """Exact client pair backed by signed inputs and sealed native bytes."""

    __slots__ = (
        "_command_client",
        "_read_client",
        "_extension_descriptor",
        "_admission_cid",
        "_birth_cid",
        "_native_admission_cid",
        "_qualification_cid",
        "_consumed",
    )

    def __init__(
        self,
        token: object,
        *,
        command_client: QuackCommandClient,
        read_client: QuackReadClient,
        extension_descriptor: int,
        admission: VerifiedEAAEFLaneRuntimeAdmission,
        process_birth: VerifiedEAAEFProcessBirth,
        native_admission: VerifiedAgentSupervisorNativeDependencyAdmission,
        qualification: VerifiedEAAEFQuackClientFactoryQualification,
    ) -> None:
        if token is not _QUALIFIED_QUACK_CLIENTS_TOKEN:
            raise TypeError("qualified Quack clients come from the exact factory")
        self._command_client = command_client
        self._read_client = read_client
        self._extension_descriptor = extension_descriptor
        self._admission_cid = str(admission["merge_admission_cid"])
        self._birth_cid = str(process_birth["birth_cid"])
        self._native_admission_cid = str(native_admission["admission_cid"])
        self._qualification_cid = qualification.qualification_cid
        self._consumed = False


def _sealed_quack_extension(raw: bytes, expected_sha256: str) -> int:
    if (
        not raw
        or "sha256:" + hashlib.sha256(raw).hexdigest()
        != _sha(expected_sha256, "qualified Quack extension sha256")
        or not hasattr(os, "memfd_create")
        or not hasattr(fcntl, "F_ADD_SEALS")
    ):
        raise EAAEFBootstrapRuntimeGatewayError(
            "qualified Quack extension cannot be materialized safely"
        )
    flags = int(getattr(os, "MFD_CLOEXEC", 0)) | int(getattr(os, "MFD_ALLOW_SEALING", 0))
    descriptor = os.memfd_create("eaaef-quack.duckdb_extension", flags=flags)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise EAAEFBootstrapRuntimeGatewayError(
                    "qualified Quack extension memfd write failed"
                )
            view = view[written:]
        os.fchmod(descriptor, 0o500)
        seals = (
            fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_SEAL
        )
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        os.lseek(descriptor, 0, os.SEEK_SET)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _fixed_sql_literal(value: str) -> str:
    """Quote one factory-owned fixed-template value, never caller SQL."""

    return "'" + value.replace("'", "''") + "'"


def _clients_from_sealed_quack_extension(
    *,
    native_module: object,
    extension_descriptor: int,
    command_endpoint: str,
    command_token: str,
    state_endpoint: str,
    state_token: str,
    alias: str,
) -> tuple[QuackCommandClient, QuackReadClient]:
    """Create exact clients without resolving away the sealed proc-fd path."""

    if not alias.replace("_", "a").isalnum():
        raise EAAEFBootstrapRuntimeGatewayError("qualified Quack alias is invalid")
    extension_path = f"/proc/self/fd/{extension_descriptor}"
    command_connection: Any | None = None
    read_connection: Any | None = None
    try:
        command_connection = native_module.connect(database=":memory:")
        command_connection.execute(f"LOAD {_fixed_sql_literal(extension_path)}")
        command_connection.execute(
            f"ATTACH {_fixed_sql_literal(command_endpoint)} AS {alias} "
            f"(TOKEN {_fixed_sql_literal(command_token)})"
        )
        read_connection = native_module.connect(database=":memory:")
        read_connection.execute(f"LOAD {_fixed_sql_literal(extension_path)}")
    except BaseException:
        for connection in (read_connection, command_connection):
            if connection is not None:
                try:
                    connection.close()
                except BaseException:
                    pass
        raise
    command = object.__new__(QuackCommandClient)
    command._connection = command_connection
    command._alias = alias
    command._endpoint = command_endpoint
    command._closed = False
    read = object.__new__(QuackReadClient)
    read._connection = read_connection
    read._endpoint = state_endpoint
    read._token = state_token
    read._closed = False
    return command, read


def create_eaaef_qualified_quack_clients(
    *,
    admission: VerifiedEAAEFLaneRuntimeAdmission,
    process_birth: VerifiedEAAEFProcessBirth,
    native_admission: VerifiedAgentSupervisorNativeDependencyAdmission,
    native_launch: AgentSupervisorNativeDependencyLaunch,
    native_module: object,
    qualification: VerifiedEAAEFQuackClientFactoryQualification,
    sealed_descriptors: EAAEFSealedQuackClientDescriptors,
) -> EAAEFQualifiedQuackClients:
    """Construct fixed clients without accepting paths, tokens, or callbacks."""

    exact_types = (
        type(admission) is VerifiedEAAEFLaneRuntimeAdmission,
        type(process_birth) is VerifiedEAAEFProcessBirth,
        type(native_admission) is VerifiedAgentSupervisorNativeDependencyAdmission,
        type(native_launch) is AgentSupervisorNativeDependencyLaunch,
        type(qualification) is VerifiedEAAEFQuackClientFactoryQualification,
        type(sealed_descriptors) is EAAEFSealedQuackClientDescriptors,
    )
    if not all(exact_types):
        raise EAAEFBootstrapRuntimeGatewayError(
            "qualified client factory rejects mappings and substitute dependencies"
        )
    now_ms = time.time_ns() // 1_000_000
    checked_lane = admission.reverify(now_ms=now_ms)
    try:
        checked_native = native_admission.reverify(now_ms=now_ms)
    except AgentSupervisorNativeDependencyAdmissionError as exc:
        raise EAAEFBootstrapRuntimeGatewayError(
            "native dependency admission failed source re-verification"
        ) from exc
    checked_qualification = qualification.reverify(now_ms=now_ms)
    if (
        process_birth.admission_cid != checked_lane["merge_admission_cid"]
        or sealed_descriptors._admission_cid != checked_lane["merge_admission_cid"]
        or sealed_descriptors._birth_cid != process_birth["birth_cid"]
        or checked_native["admission_cid"]
        != checked_lane["native_dependency_admission_cid"]
        or checked_qualification.qualification_cid
        != checked_lane["quack_client_factory_qualification_cid"]
        or native_launch.accepted_authorization_id != checked_native["admission_cid"]
        or native_launch.pin != checked_native.native_dependency_pin
        or sealed_descriptors._used
    ):
        raise EAAEFBootstrapRuntimeGatewayError(
            "qualified client dependencies differ from one exact signed birth"
        )
    try:
        native_path = verify_agent_supervisor_native_dependency_sealed_fd(native_launch)
    except ValueError as exc:
        raise EAAEFBootstrapRuntimeGatewayError(
            "native dependency sealed descriptor is invalid"
        ) from exc
    pin = checked_native.native_dependency_pin
    if (
        sys.modules.get(pin.module_name) is not native_module
        or sys.modules.get(pin.public_alias) is not native_module
        or getattr(native_module, "__name__", None) != pin.module_name
        or getattr(native_module, "__file__", None) != native_path
        or getattr(native_module, "__version__", None) != pin.distribution_version
        or not callable(getattr(native_module, "connect", None))
    ):
        raise EAAEFBootstrapRuntimeGatewayError(
            "preloaded native module differs from its signed sealed admission"
        )
    sealed_descriptors._used = True
    extension_descriptor = _sealed_quack_extension(
        checked_qualification._extension_bytes,
        str(checked_qualification["quack_extension_sha256"]),
    )
    command: QuackCommandClient | None = None
    read: QuackReadClient | None = None
    try:
        command, read = _clients_from_sealed_quack_extension(
            native_module=native_module,
            extension_descriptor=extension_descriptor,
            command_endpoint=str(
                checked_lane.operational_capability["command_endpoint"]
            ),
            command_token=str(sealed_descriptors._command["token"]),
            state_endpoint=str(checked_lane.operational_capability["state_endpoint"]),
            state_token=str(sealed_descriptors._state["token"]),
            alias=(
                "eaaef_ingress_"
                + str(checked_lane["lane_authority_cid"])
                .removeprefix("sha256:")[:16]
            ),
        )
    except BaseException:
        for client in (read, command):
            if client is not None:
                try:
                    client.close()
                except BaseException:
                    pass
        os.close(extension_descriptor)
        sealed_descriptors._command.clear()
        sealed_descriptors._state.clear()
        raise
    sealed_descriptors._command.clear()
    sealed_descriptors._state.clear()
    return EAAEFQualifiedQuackClients(
        _QUALIFIED_QUACK_CLIENTS_TOKEN,
        command_client=command,
        read_client=read,
        extension_descriptor=extension_descriptor,
        admission=checked_lane,
        process_birth=process_birth,
        native_admission=checked_native,
        qualification=checked_qualification,
    )


_TRANSPORT_FACTORY_TOKEN = object()


class EAAEFBootstrapCommandTransport:
    """Exact append/fixed-receipt channel; never accepts callbacks or SQL."""

    INTERFACE: ClassVar[str] = EAAEF_BOOTSTRAP_COMMAND_TRANSPORT_INTERFACE
    __slots__ = (
        "_command_client",
        "_read_client",
        "_admission_cid",
        "_birth_cid",
        "_native_admission_cid",
        "_qualification_cid",
        "_extension_descriptor",
        "_maximum_wait_ms",
        "_poll_interval_ms",
        "_closed",
    )

    def __init__(
        self,
        token: object,
        *,
        command_client: QuackCommandClient,
        read_client: QuackReadClient,
        admission: VerifiedEAAEFLaneRuntimeAdmission,
        maximum_wait_ms: int,
        poll_interval_ms: int,
        qualified_clients: EAAEFQualifiedQuackClients | None = None,
    ) -> None:
        if token is not _TRANSPORT_FACTORY_TOKEN:
            raise TypeError("EAAEF command transports come from the exact client binder")
        if (
            type(command_client) is not QuackCommandClient
            or type(read_client) is not QuackReadClient
            or type(admission) is not VerifiedEAAEFLaneRuntimeAdmission
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "command transport rejects callbacks, duck types, and mappings"
            )
        capability = admission.operational_capability
        if (
            command_client.endpoint != capability["command_endpoint"]
            or read_client.endpoint != capability["state_endpoint"]
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "command transport endpoints differ from its exact admission"
            )
        self._command_client = command_client
        self._read_client = read_client
        self._admission_cid = str(admission["merge_admission_cid"])
        if qualified_clients is None:
            self._birth_cid = ""
            self._native_admission_cid = ""
            self._qualification_cid = ""
            self._extension_descriptor = None
        else:
            if (
                type(qualified_clients) is not EAAEFQualifiedQuackClients
                or qualified_clients._consumed
                or qualified_clients._command_client is not command_client
                or qualified_clients._read_client is not read_client
                or qualified_clients._admission_cid != self._admission_cid
            ):
                raise EAAEFBootstrapRuntimeGatewayError(
                    "transport received divergent qualified clients"
                )
            qualified_clients._consumed = True
            self._birth_cid = qualified_clients._birth_cid
            self._native_admission_cid = qualified_clients._native_admission_cid
            self._qualification_cid = qualified_clients._qualification_cid
            self._extension_descriptor = qualified_clients._extension_descriptor
        self._maximum_wait_ms = _positive(maximum_wait_ms, "maximum receipt wait", maximum=60_000)
        self._poll_interval_ms = _positive(
            poll_interval_ms,
            "receipt poll interval",
            maximum=self._maximum_wait_ms,
        )
        self._closed = False

    @property
    def admission_cid(self) -> str:
        return self._admission_cid

    def append(self, envelope: AuthorizedStateCommand) -> None:
        if self._closed:
            raise EAAEFBootstrapRuntimeGatewayError("command transport is closed")
        self._command_client.append(envelope)

    def receipts(self) -> tuple[Mapping[str, Any], ...]:
        if self._closed:
            raise EAAEFBootstrapRuntimeGatewayError("command transport is closed")
        return self._read_client.list_recent_receipts()

    def lookup_receipt(
        self, envelope: AuthorizedStateCommand
    ) -> Mapping[str, Any] | None:
        """Find one exact receipt in the legacy bounded projection."""

        if type(envelope) is not AuthorizedStateCommand:
            raise EAAEFBootstrapRuntimeGatewayError(
                "command receipt lookup requires an exact envelope"
            )
        matched: Mapping[str, Any] | None = None
        receipts = self.receipts()
        if type(receipts) is not tuple:
            raise EAAEFBootstrapRuntimeGatewayDiverged(
                "owner receipt projection is not an exact tuple"
            )
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                raise EAAEFBootstrapRuntimeGatewayDiverged(
                    "owner receipt projection contains a non-record"
                )
            if receipt.get("submission_id") != envelope.submission_id:
                continue
            if receipt.get("envelope_cid") != envelope.envelope_cid:
                raise EAAEFBootstrapRuntimeGatewayDiverged(
                    "ordinary submission_id collision has a different envelope"
                )
            if matched is not None and dict(matched) != dict(receipt):
                raise EAAEFBootstrapRuntimeGatewayDiverged(
                    "duplicate owner receipts diverged"
                )
            matched = receipt
        return matched

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        for client in (self._read_client, self._command_client):
            try:
                client.close()
            except BaseException as exc:
                errors.append(exc)
        if self._extension_descriptor is not None:
            try:
                os.close(self._extension_descriptor)
            except OSError as exc:
                errors.append(exc)
            self._extension_descriptor = None
        self._closed = True
        if errors:
            raise EAAEFBootstrapRuntimeGatewayError(
                "EAAEF command transport failed close"
            ) from errors[0]


def bind_eaaef_bootstrap_command_transport(
    *,
    command_client: QuackCommandClient,
    read_client: QuackReadClient,
    admission: VerifiedEAAEFLaneRuntimeAdmission,
    maximum_wait_ms: int = 30_000,
    poll_interval_ms: int = 10,
) -> EAAEFBootstrapCommandTransport:
    """Bind exact fixed-template clients to one source-verified lane."""

    if (
        type(command_client) is not QuackCommandClient
        or type(read_client) is not QuackReadClient
        or type(admission) is not VerifiedEAAEFLaneRuntimeAdmission
    ):
        raise EAAEFBootstrapRuntimeGatewayError(
            "transport binder requires exact Quack clients and lane admission"
        )
    capability = admission.operational_capability
    if (
        command_client.endpoint != capability["command_endpoint"]
        or read_client.endpoint != capability["state_endpoint"]
    ):
        raise EAAEFBootstrapRuntimeGatewayError(
            "transport endpoints differ from the source-verified capability"
        )
    return EAAEFBootstrapCommandTransport(
        _TRANSPORT_FACTORY_TOKEN,
        command_client=command_client,
        read_client=read_client,
        admission=admission,
        maximum_wait_ms=maximum_wait_ms,
        poll_interval_ms=poll_interval_ms,
    )


def bind_eaaef_qualified_bootstrap_command_transport(
    *,
    clients: EAAEFQualifiedQuackClients,
    admission: VerifiedEAAEFLaneRuntimeAdmission,
    process_birth: VerifiedEAAEFProcessBirth,
    maximum_wait_ms: int = 30_000,
    poll_interval_ms: int = 10,
) -> EAAEFBootstrapCommandTransport:
    """Consume one exact qualified client bundle into the R1 transport."""

    if (
        type(clients) is not EAAEFQualifiedQuackClients
        or type(admission) is not VerifiedEAAEFLaneRuntimeAdmission
        or type(process_birth) is not VerifiedEAAEFProcessBirth
        or clients._consumed
        or clients._admission_cid != admission["merge_admission_cid"]
        or clients._birth_cid != process_birth["birth_cid"]
        or process_birth.admission_cid != admission["merge_admission_cid"]
    ):
        raise EAAEFBootstrapRuntimeGatewayError(
            "qualified transport requires one exact client/lane/process birth"
        )
    return EAAEFBootstrapCommandTransport(
        _TRANSPORT_FACTORY_TOKEN,
        command_client=clients._command_client,
        read_client=clients._read_client,
        admission=admission,
        maximum_wait_ms=maximum_wait_ms,
        poll_interval_ms=poll_interval_ms,
        qualified_clients=clients,
    )


_TYPED_OWNER_CLIENT_FACTORY_TOKEN = object()
_TYPED_OWNER_TRANSPORT_FACTORY_TOKEN = object()


class EAAEFTypedOwnerCommandClient:
    """Borrow one authenticated typed-owner channel; never expose it onward."""

    INTERFACE: ClassVar[str] = EAAEF_TYPED_OWNER_COMMAND_CLIENT_INTERFACE
    __slots__ = (
        "_owner_connection",
        "_admission_cid",
        "_operational_capability_cid",
        "_closed",
    )

    def __init__(
        self,
        token: object,
        *,
        owner_connection: TypedStateOwnerConnection,
        admission: VerifiedEAAEFLaneRuntimeAdmission,
    ) -> None:
        if token is not _TYPED_OWNER_CLIENT_FACTORY_TOKEN:
            raise TypeError(
                "typed-owner EAAEF clients come from the exact channel binder"
            )
        if (
            type(owner_connection) is not TypedStateOwnerConnection
            or type(admission) is not VerifiedEAAEFLaneRuntimeAdmission
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "typed-owner EAAEF client rejects mappings, callbacks, and substitutes"
            )
        self._owner_connection = owner_connection
        self._admission_cid = str(admission["merge_admission_cid"])
        self._operational_capability_cid = str(
            admission["operational_capability_cid"]
        )
        self._closed = False

    @property
    def admission_cid(self) -> str:
        return self._admission_cid

    def _request(
        self,
        operation: str,
        envelope: AuthorizedStateCommand,
        *,
        receipt_required: bool,
    ) -> Mapping[str, Any] | None:
        if self._closed:
            raise EAAEFBootstrapRuntimeGatewayError(
                "typed-owner EAAEF client is closed"
            )
        if (
            operation
            not in {
                EAAEF_TYPED_OWNER_COMMAND_SUBMIT_OPERATION,
                EAAEF_TYPED_OWNER_COMMAND_LOOKUP_OPERATION,
            }
            or type(envelope) is not AuthorizedStateCommand
            or type(envelope.command) is not StateCommand
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "typed-owner EAAEF request is outside the exact wire vocabulary"
            )
        response = self._owner_connection._request(  # noqa: SLF001
            operation,
            envelope=envelope.to_dict(),
            merge_admission_cid=self._admission_cid,
            operational_capability_cid=self._operational_capability_cid,
        )
        receipt = response.get("receipt")
        if receipt is None and not receipt_required:
            return None
        if not isinstance(receipt, Mapping) or set(receipt) != set(
            EAAEF_TYPED_OWNER_PUBLIC_RECEIPT_FIELDS
        ):
            raise EAAEFBootstrapRuntimeGatewayDiverged(
                "typed owner returned a non-exact EAAEF receipt"
            )
        detached = _canonical_detached(dict(receipt), "typed owner EAAEF receipt")
        return MappingProxyType(detached)

    def submit(
        self, envelope: AuthorizedStateCommand
    ) -> Mapping[str, Any]:
        receipt = self._request(
            EAAEF_TYPED_OWNER_COMMAND_SUBMIT_OPERATION,
            envelope,
            receipt_required=True,
        )
        if receipt is None:
            raise EAAEFBootstrapRuntimeGatewayDiverged(
                "typed owner submit returned no durable receipt"
            )
        return receipt

    def lookup(
        self, envelope: AuthorizedStateCommand
    ) -> Mapping[str, Any] | None:
        return self._request(
            EAAEF_TYPED_OWNER_COMMAND_LOOKUP_OPERATION,
            envelope,
            receipt_required=False,
        )

    def close(self) -> None:
        # The connection belongs to the surrounding typed-owner session.  A
        # component adapter must never close or replace that shared channel.
        self._closed = True


class EAAEFTypedOwnerCommandTransport:
    """Exact receipt transport over an already-open typed-owner channel."""

    INTERFACE: ClassVar[str] = EAAEF_TYPED_OWNER_COMMAND_TRANSPORT_INTERFACE
    __slots__ = (
        "_client",
        "_admission_cid",
        "_maximum_wait_ms",
        "_poll_interval_ms",
        "_last_receipt",
        "_closed",
    )

    def __init__(
        self,
        token: object,
        *,
        client: EAAEFTypedOwnerCommandClient,
        admission: VerifiedEAAEFLaneRuntimeAdmission,
        maximum_wait_ms: int,
        poll_interval_ms: int,
    ) -> None:
        if token is not _TYPED_OWNER_TRANSPORT_FACTORY_TOKEN:
            raise TypeError(
                "typed-owner EAAEF transports come from the exact binder"
            )
        if (
            type(client) is not EAAEFTypedOwnerCommandClient
            or type(admission) is not VerifiedEAAEFLaneRuntimeAdmission
            or client.admission_cid != admission["merge_admission_cid"]
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "typed-owner transport requires one exact client/lane binding"
            )
        self._client = client
        self._admission_cid = str(admission["merge_admission_cid"])
        self._maximum_wait_ms = _positive(
            maximum_wait_ms, "maximum receipt wait", maximum=60_000
        )
        self._poll_interval_ms = _positive(
            poll_interval_ms,
            "receipt poll interval",
            maximum=self._maximum_wait_ms,
        )
        self._last_receipt: Mapping[str, Any] | None = None
        self._closed = False

    @property
    def admission_cid(self) -> str:
        return self._admission_cid

    def append(self, envelope: AuthorizedStateCommand) -> None:
        if self._closed:
            raise EAAEFBootstrapRuntimeGatewayError(
                "typed-owner EAAEF transport is closed"
            )
        self._last_receipt = self._client.submit(envelope)

    def lookup_receipt(
        self, envelope: AuthorizedStateCommand
    ) -> Mapping[str, Any] | None:
        if self._closed:
            raise EAAEFBootstrapRuntimeGatewayError(
                "typed-owner EAAEF transport is closed"
            )
        cached = self._last_receipt
        if (
            cached is not None
            and cached.get("submission_id") == envelope.submission_id
        ):
            if cached.get("envelope_cid") != envelope.envelope_cid:
                raise EAAEFBootstrapRuntimeGatewayDiverged(
                    "ordinary submission_id collision has a different envelope"
                )
            return cached
        receipt = self._client.lookup(envelope)
        if receipt is not None:
            self._last_receipt = receipt
        return receipt

    def receipts(self) -> tuple[Mapping[str, Any], ...]:
        """Retain a bounded compatibility projection for source diagnostics."""

        if self._closed:
            raise EAAEFBootstrapRuntimeGatewayError(
                "typed-owner EAAEF transport is closed"
            )
        return () if self._last_receipt is None else (self._last_receipt,)

    def close(self) -> None:
        if self._closed:
            return
        self._client.close()
        self._last_receipt = None
        self._closed = True


def bind_eaaef_typed_owner_command_client(
    *,
    owner_connection: TypedStateOwnerConnection,
    admission: VerifiedEAAEFLaneRuntimeAdmission,
) -> EAAEFTypedOwnerCommandClient:
    """Bind a borrowed authenticated typed-owner connection to one lane."""

    return EAAEFTypedOwnerCommandClient(
        _TYPED_OWNER_CLIENT_FACTORY_TOKEN,
        owner_connection=owner_connection,
        admission=admission,
    )


def bind_eaaef_typed_owner_command_transport(
    *,
    owner_connection: TypedStateOwnerConnection,
    admission: VerifiedEAAEFLaneRuntimeAdmission,
    maximum_wait_ms: int = 30_000,
    poll_interval_ms: int = 10,
) -> EAAEFTypedOwnerCommandTransport:
    """Build the source-only typed transport without claiming cutover."""

    client = bind_eaaef_typed_owner_command_client(
        owner_connection=owner_connection,
        admission=admission,
    )
    return EAAEFTypedOwnerCommandTransport(
        _TYPED_OWNER_TRANSPORT_FACTORY_TOKEN,
        client=client,
        admission=admission,
        maximum_wait_ms=maximum_wait_ms,
        poll_interval_ms=poll_interval_ms,
    )


_EAAEF_COMMAND_TRANSPORT_TYPES: Final = (
    EAAEFBootstrapCommandTransport,
    EAAEFTypedOwnerCommandTransport,
)


def _process_start_time_ticks(pid: int) -> int:
    try:
        raw = Path(f"/proc/{_positive(pid, 'process pid')}/stat").read_text(
            encoding="ascii"
        )
        closing = raw.rfind(")")
        fields = raw[closing + 2 :].split()
        result = int(fields[19])
    except (OSError, UnicodeError, ValueError, IndexError) as exc:
        raise EAAEFBootstrapRuntimeGatewayError(
            "dynamic service process birth is unavailable"
        ) from exc
    return _positive(result, "dynamic service process birth")


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    remaining = _positive(size, "dynamic service response size", maximum=8 * 1024 * 1024)
    chunks: list[bytes] = []
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EAAEFBootstrapRuntimeGatewayError(
                "dynamic service closed before its complete response"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class _EAAEFContainerDynamicServiceClient:
    """One fixed signed Unix service; no callback or caller-selected endpoint."""

    __slots__ = ("_admission", "_birth", "_qualification", "_service", "_descriptor")

    def __init__(
        self,
        *,
        admission: VerifiedEAAEFLaneRuntimeAdmission,
        process_birth: VerifiedEAAEFProcessBirth,
        qualification: VerifiedEAAEFContainerDispatcherFactoryQualification,
        service: str,
    ) -> None:
        if (
            type(admission) is not VerifiedEAAEFLaneRuntimeAdmission
            or type(process_birth) is not VerifiedEAAEFProcessBirth
            or type(qualification)
            is not VerifiedEAAEFContainerDispatcherFactoryQualification
            or service not in {"worker", "verifier", "merge", "host_source"}
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "dynamic service client requires exact signed dependencies"
            )
        services = qualification["services"]
        descriptor = services.get(service) if isinstance(services, Mapping) else None
        if not isinstance(descriptor, Mapping):
            raise EAAEFBootstrapRuntimeGatewayError(
                "dynamic service qualification lost its exact descriptor"
            )
        self._admission = admission
        self._birth = process_birth
        self._qualification = qualification
        self._service = service
        self._descriptor = MappingProxyType(
            _canonical_detached(dict(descriptor), "dynamic service descriptor")
        )

    def _connect(self) -> socket.socket:
        endpoint = str(self._descriptor["endpoint"])
        path = endpoint.removeprefix("unix:")
        expected_uid = int(self._descriptor["expected_server_uid"])
        try:
            status = os.lstat(path)
        except OSError as exc:
            raise EAAEFBootstrapRuntimeGatewayError(
                "signed dynamic service endpoint is unavailable"
            ) from exc
        if (
            not stat.S_ISSOCK(status.st_mode)
            or status.st_uid != expected_uid
            or status.st_mode & stat.S_IWOTH
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "signed dynamic service endpoint identity is invalid"
            )
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.settimeout(int(self._descriptor["request_timeout_ms"]) / 1000)
            connection.connect(path)
            peer = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            peer_pid, peer_uid, _peer_gid = struct.unpack("3i", peer)
            if (
                peer_uid != expected_uid
                or peer_pid != int(self._descriptor["expected_server_pid"])
                or _process_start_time_ticks(peer_pid)
                != int(self._descriptor["expected_server_process_start_time_ticks"])
            ):
                raise EAAEFBootstrapRuntimeGatewayError(
                    "connected dynamic service differs from its signed process birth"
                )
            return connection
        except BaseException:
            connection.close()
            raise

    def request(self, method: str, arguments: Mapping[str, Any]) -> Any:
        now_ms = time.time_ns() // 1_000_000
        try:
            checked_lane = self._admission.reverify(now_ms=now_ms)
            checked_qualification = self._qualification.reverify(now_ms=now_ms)
        except EAAEFLaneGatewayAdmissionError as exc:
            raise EAAEFBootstrapRuntimeGatewayError(
                "dynamic service sources failed live re-verification"
            ) from exc
        if (
            self._birth.admission_cid != checked_lane["merge_admission_cid"]
            or checked_qualification.qualification_cid
            != checked_lane["container_dispatcher_factory_qualification_cid"]
            or checked_qualification["services"].get(self._service)
            != dict(self._descriptor)
            or method not in self._descriptor["methods"]
            or not isinstance(arguments, Mapping)
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "dynamic service request differs from its exact signed lane"
            )
        body = {
            "schema": EAAEF_CONTAINER_DYNAMIC_SERVICE_REQUEST_SCHEMA,
            "interface": "EAAEFContainerDynamicService@1",
            "service": self._service,
            "method": method,
            "lane_authority_cid": checked_lane["lane_authority_cid"],
            "lane_merge_admission_cid": checked_lane["merge_admission_cid"],
            "lane_session_id": checked_lane["lane_session_id"],
            "lane_generation": checked_lane["lane_generation"],
            "process_instance_id": checked_lane["process_instance_id"],
            "process_birth_nonce": checked_lane["process_birth_nonce"],
            "process_birth_cid": self._birth["birth_cid"],
            "request_nonce": secrets.token_hex(32),
            "arguments": _canonical_detached(dict(arguments), "dynamic service arguments"),
        }
        request = {**body, "request_cid": _content_cid(body, "dynamic service request")}
        raw = _canonical_bytes(request, "dynamic service request")
        if len(raw) > int(self._descriptor["maximum_request_bytes"]):
            raise EAAEFBootstrapRuntimeGatewayError(
                "dynamic service request exceeds its signed bound"
            )
        connection = self._connect()
        try:
            connection.sendall(struct.pack("!I", len(raw)) + raw)
            header = _receive_exact(connection, 4)
            response_size = struct.unpack("!I", header)[0]
            if not 0 < response_size <= int(self._descriptor["maximum_response_bytes"]):
                raise EAAEFBootstrapRuntimeGatewayError(
                    "dynamic service response exceeds its signed bound"
                )
            response_raw = _receive_exact(connection, response_size)
        except (OSError, TimeoutError) as exc:
            raise EAAEFBootstrapRuntimeGatewayError(
                "dynamic service request has no complete response"
            ) from exc
        finally:
            connection.close()
        try:
            response = json.loads(response_raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise EAAEFBootstrapRuntimeGatewayError(
                "dynamic service response is not canonical JSON"
            ) from exc
        fields = {
            "schema",
            "interface",
            "service",
            "method",
            "request_cid",
            "service_principal_did",
            "result",
            "issued_at_ms",
            "expires_at_ms",
            "response_nonce",
            "service_signature",
            "response_cid",
        }
        if (
            not isinstance(response, dict)
            or set(response) != fields
            or response_raw != _canonical_bytes(response, "dynamic service response")
            or response.get("schema") != EAAEF_CONTAINER_DYNAMIC_SERVICE_RESPONSE_SCHEMA
            or response.get("interface") != "EAAEFContainerDynamicService@1"
            or response.get("service") != self._service
            or response.get("method") != method
            or response.get("request_cid") != request["request_cid"]
            or response.get("service_principal_did")
            != self._descriptor["service_principal_did"]
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "dynamic service response is not exactly bound to its request"
            )
        issued = _positive(response.get("issued_at_ms"), "service response issued_at_ms")
        expires = _positive(response.get("expires_at_ms"), "service response expires_at_ms")
        verified_at_ms = time.time_ns() // 1_000_000
        signed = dict(response)
        claimed_cid = signed.pop("response_cid", None)
        signature = signed.pop("service_signature", None)
        if (
            issued > verified_at_ms
            or verified_at_ms >= expires
            or issued >= expires
            or expires - issued > 60_000
            or claimed_cid
            != _content_cid({**signed, "service_signature": signature}, "service response")
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "dynamic service response lifetime or content identity is invalid"
            )
        _identifier(response.get("response_nonce"), "dynamic service response nonce")
        try:
            verify_did_key_signature(
                identity_did=str(self._descriptor["service_principal_did"]),
                payload=signed,
                signature=str(signature or ""),
            )
        except (LocalProfileTampered, ValueError) as exc:
            raise EAAEFBootstrapRuntimeGatewayError(
                "dynamic service response signature is invalid"
            ) from exc
        return _canonical_detached(response["result"], "dynamic service result")


_CONTAINER_DISPATCHER_FACTORY_TOKEN = object()


class EAAEFContainerDispatcherFactory:
    """Lazy exact wrapper around signed per-attempt remote services."""

    INTERFACE: ClassVar[str] = EAAEF_CONTAINER_DISPATCHER_FACTORY_INTERFACE
    __slots__ = (
        "_admission",
        "_process_birth",
        "_native_admission",
        "_quack_qualification",
        "_qualification",
        "_services",
        "_created",
    )

    def __init__(
        self,
        token: object,
        *,
        admission: VerifiedEAAEFLaneRuntimeAdmission,
        process_birth: VerifiedEAAEFProcessBirth,
        native_admission: VerifiedAgentSupervisorNativeDependencyAdmission,
        quack_qualification: VerifiedEAAEFQuackClientFactoryQualification,
        qualification: VerifiedEAAEFContainerDispatcherFactoryQualification,
    ) -> None:
        if token is not _CONTAINER_DISPATCHER_FACTORY_TOKEN:
            raise TypeError("container dispatcher factories come from the exact binder")
        self._admission = admission
        self._process_birth = process_birth
        self._native_admission = native_admission
        self._quack_qualification = quack_qualification
        self._qualification = qualification
        self._services = MappingProxyType(
            {
                name: _EAAEFContainerDynamicServiceClient(
                    admission=admission,
                    process_birth=process_birth,
                    qualification=qualification,
                    service=name,
                )
                for name in ("worker", "verifier", "merge", "host_source")
            }
        )
        self._created = False

    @property
    def qualification_cid(self) -> str:
        return self._qualification.qualification_cid

    @staticmethod
    def _attempt_arguments(attempt: Any) -> dict[str, Any]:
        fields = (
            "task_cid",
            "task_alias",
            "attempt_id",
            "claim_id",
            "attempt_number",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
            "lease_id",
        )
        return _canonical_detached(
            {name: getattr(attempt, name, None) for name in fields},
            "container attempt projection",
        )

    def create(
        self,
        *,
        execution_repository: EAAEFBootstrapExecutionRepositoryProxy,
    ) -> ExternalAgentContainerWorkerDispatcher:
        """Create callbacks internally; no caller callback enters this boundary."""

        if self._created:
            raise EAAEFBootstrapRuntimeGatewayError(
                "container dispatcher factory is single-use"
            )
        if type(execution_repository) is not EAAEFBootstrapExecutionRepositoryProxy:
            raise EAAEFBootstrapRuntimeGatewayError(
                "container dispatcher requires the exact EAAEF execution proxy"
            )
        now_ms = time.time_ns() // 1_000_000
        checked_lane = self._admission.reverify(now_ms=now_ms)
        checked_quack = self._quack_qualification.reverify(now_ms=now_ms)
        checked_dispatcher = self._qualification.reverify(now_ms=now_ms)
        try:
            checked_native = self._native_admission.reverify(now_ms=now_ms)
        except AgentSupervisorNativeDependencyAdmissionError as exc:
            raise EAAEFBootstrapRuntimeGatewayError(
                "dispatcher native admission failed source re-verification"
            ) from exc
        if (
            self._process_birth.admission_cid != checked_lane["merge_admission_cid"]
            or checked_native["admission_cid"]
            != checked_lane["native_dependency_admission_cid"]
            or checked_quack.qualification_cid
            != checked_lane["quack_client_factory_qualification_cid"]
            or checked_dispatcher.qualification_cid
            != checked_lane["container_dispatcher_factory_qualification_cid"]
            or execution_repository.gateway_binding_cid
            != checked_lane["gateway_binding_cid"]
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "dispatcher factory dependencies differ from one exact lane birth"
            )
        worker = self._services["worker"]
        verifier = self._services["verifier"]
        merge = self._services["merge"]
        host_source = self._services["host_source"]
        task_pairs = dict(zip(checked_lane["task_ids"], checked_lane["task_cids"], strict=True))
        worker_principal = str(
            checked_dispatcher["services"]["worker"]["service_principal_did"]
        )
        verifier_principal = str(
            checked_dispatcher["services"]["verifier"]["service_principal_did"]
        )
        merge_principal = str(
            checked_dispatcher["services"]["merge"]["service_principal_did"]
        )

        def packet_provider(attempt: Any) -> ExternalAgentContainerWorkPacket:
            result = worker.request("packet", self._attempt_arguments(attempt))
            if not isinstance(result, Mapping):
                raise ExternalAgentContainerDispatchError(
                    "signed packet service returned a non-object"
                )
            packet = ExternalAgentContainerWorkPacket.from_mapping(result)
            if (
                task_pairs.get(packet.task_id) != packet.task_cid
                or packet.plan_revision_cid != checked_lane["active_plan_revision_cid"]
                or packet.gateway_binding_cid != checked_lane["gateway_binding_cid"]
                or packet.worker_principal_did != worker_principal
            ):
                raise ExternalAgentContainerDispatchError(
                    "dynamic work packet differs from the signed lane task population"
                )
            return packet

        def qualification_guard(packet: ExternalAgentContainerWorkPacket) -> Mapping[str, Any]:
            result = worker.request("qualify", {"packet": packet.to_dict()})
            if not isinstance(result, Mapping):
                raise ExternalAgentContainerDispatchError(
                    "signed qualification service returned a non-object"
                )
            return result

        def container_launcher(
            packet: ExternalAgentContainerWorkPacket,
            reservation: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            result = worker.request(
                "launch", {"packet": packet.to_dict(), "reservation": dict(reservation)}
            )
            if not isinstance(result, Mapping):
                raise ExternalAgentContainerDispatchError(
                    "signed worker service returned a non-object proposal"
                )
            return result

        def independent_verifier(
            packet: ExternalAgentContainerWorkPacket,
            proposal: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            result = verifier.request(
                "verify", {"packet": packet.to_dict(), "proposal": dict(proposal)}
            )
            if (
                not isinstance(result, Mapping)
                or result.get("verifier_principal_did") != verifier_principal
            ):
                raise ExternalAgentContainerDispatchError(
                    "dynamic verification receipt differs from its signed service"
                )
            return result

        def merge_observer(
            packet: ExternalAgentContainerWorkPacket,
            effect: Mapping[str, Any],
        ) -> Mapping[str, Any] | None:
            result = merge.request(
                "observe_merge", {"packet": packet.to_dict(), "effect": dict(effect)}
            )
            if result is None:
                return None
            if (
                not isinstance(result, Mapping)
                or result.get("reviewer_principal_did") != merge_principal
            ):
                raise ExternalAgentContainerDispatchError(
                    "dynamic merge admission differs from its signed service"
                )
            return result

        def host_source_observer() -> str:
            result = host_source.request("observe_source", {})
            identity = str(result or "") if isinstance(result, str) else ""
            if _SHA256.fullmatch(identity) is None and re.fullmatch(
                r"[0-9a-f]{40}", identity
            ) is None:
                raise ExternalAgentContainerDispatchError(
                    "dynamic host-source observer returned an invalid identity"
                )
            return identity

        self._created = True
        return ExternalAgentContainerWorkerDispatcher(
            execution_repository=execution_repository,
            packet_provider=packet_provider,
            qualification_guard=qualification_guard,
            container_launcher=container_launcher,
            independent_verifier=independent_verifier,
            merge_admission_observer=merge_observer,
            host_source_observer=host_source_observer,
            now_ms=lambda: time.time_ns() // 1_000_000,
        )


def create_eaaef_container_dispatcher_factory(
    *,
    admission: VerifiedEAAEFLaneRuntimeAdmission,
    process_birth: VerifiedEAAEFProcessBirth,
    native_admission: VerifiedAgentSupervisorNativeDependencyAdmission,
    quack_qualification: VerifiedEAAEFQuackClientFactoryQualification,
    qualification: VerifiedEAAEFContainerDispatcherFactoryQualification,
) -> EAAEFContainerDispatcherFactory:
    """Bind exact signed services without opening a socket or launching work."""

    if (
        type(admission) is not VerifiedEAAEFLaneRuntimeAdmission
        or type(process_birth) is not VerifiedEAAEFProcessBirth
        or type(native_admission) is not VerifiedAgentSupervisorNativeDependencyAdmission
        or type(quack_qualification) is not VerifiedEAAEFQuackClientFactoryQualification
        or type(qualification)
        is not VerifiedEAAEFContainerDispatcherFactoryQualification
    ):
        raise EAAEFBootstrapRuntimeGatewayError(
            "dispatcher factory binder rejects mappings, callbacks, and substitutes"
        )
    now_ms = time.time_ns() // 1_000_000
    checked_lane = admission.reverify(now_ms=now_ms)
    checked_quack = quack_qualification.reverify(now_ms=now_ms)
    checked_dispatcher = qualification.reverify(now_ms=now_ms)
    try:
        checked_native = native_admission.reverify(now_ms=now_ms)
    except AgentSupervisorNativeDependencyAdmissionError as exc:
        raise EAAEFBootstrapRuntimeGatewayError(
            "dispatcher native admission failed source re-verification"
        ) from exc
    if (
        process_birth.admission_cid != checked_lane["merge_admission_cid"]
        or checked_native["admission_cid"] != checked_lane["native_dependency_admission_cid"]
        or checked_quack.qualification_cid
        != checked_lane["quack_client_factory_qualification_cid"]
        or checked_dispatcher.qualification_cid
        != checked_lane["container_dispatcher_factory_qualification_cid"]
    ):
        raise EAAEFBootstrapRuntimeGatewayError(
            "dispatcher factory artifacts do not share one exact signed birth"
        )
    return EAAEFContainerDispatcherFactory(
        _CONTAINER_DISPATCHER_FACTORY_TOKEN,
        admission=checked_lane,
        process_birth=process_birth,
        native_admission=checked_native,
        quack_qualification=checked_quack,
        qualification=checked_dispatcher,
    )


class _GatewayRecord:
    __slots__ = ("_record",)

    def __init__(self, record: Mapping[str, Any]) -> None:
        self._record = _canonical_detached(dict(record), "gateway record")

    def __getattr__(self, name: str) -> Any:
        try:
            return self._record[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        return dict(self._record)


class _GatewayTaskPage:
    __slots__ = ("tasks",)

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.tasks = tuple(_GatewayRecord(item) for item in records)


_BOARD_OPERATIONS: Final = frozenset(
    {
        "task.ready",
        "coordination.register_task",
        "coordination.claim_ready",
        "coordination.list_unsettled_completions",
        "coordination.reconcile_promoted_completion",
        "coordination.recover_prepared_completion",
        "coordination.abort_prepared_completion",
        "coordination.expire_claim",
        "execution.bind_daemon",
        "execution.list_running_attempts",
        "execution.commit_reconciled_attempt",
    }
)
_LANE_BOUND_OPERATIONS: Final = frozenset(
    {
        "coordination.claim_ready",
        "execution.bind_daemon",
        "execution.list_running_attempts",
    }
)
_EXCLUDED_OPERATIONS: Final = frozenset(
    {
        "task.materialize",
        "task.list",
        "merge.enqueue",
        "merge.observe",
        "merge.accept",
        "plan_r2.prepare",
        "plan_r2.apply",
        "plan_r2.observe",
    }
)


def _plain_mapping(value: Any, noun: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _canonical_detached(dict(value), noun)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return _canonical_detached(dict(result), noun)
    raise EAAEFBootstrapRuntimeGatewayError(f"{noun} is not a closed record")


class _EAAEFCommandDispatcher:
    """Shared R1 authority registry and journal-first append/poll dispatcher."""

    __slots__ = (
        "_admission",
        "_authorization_client",
        "_transport",
        "_journal",
        "_policy",
        "_lane_binding",
        "_recovery_admissions",
        "_recovery_lane_bindings",
        "_authorities_by_attempt",
        "_authorities_by_claim",
        "_authorities_by_task",
        "_lock",
        "_attached",
    )

    def __init__(
        self,
        *,
        admission: VerifiedEAAEFLaneRuntimeAdmission,
        authorization_client: EAAEFCommandAuthorizationServiceClient,
        transport: EAAEFBootstrapCommandTransport
        | EAAEFTypedOwnerCommandTransport,
        journal: EAAEFExactEnvelopeJournal,
        recovery_admissions: Sequence[
            VerifiedEAAEFLaneRuntimeAdmission | VerifiedEAAEFExpiredLaneRecoveryAdmission
        ],
    ) -> None:
        if (
            type(admission) is not VerifiedEAAEFLaneRuntimeAdmission
            or type(authorization_client) is not EAAEFCommandAuthorizationServiceClient
            or type(transport) not in _EAAEF_COMMAND_TRANSPORT_TYPES
            or type(journal) is not EAAEFExactEnvelopeJournal
            or transport.admission_cid != admission["merge_admission_cid"]
            or journal._admission_cid != admission["merge_admission_cid"]
            or journal._lane_authority_cid != admission["lane_authority_cid"]
            or authorization_client._operational_capability_cid
            != admission["operational_capability_cid"]
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "dispatcher dependencies are not one exact typed lane bundle"
            )
        if any(
            type(item)
            not in {
                VerifiedEAAEFLaneRuntimeAdmission,
                VerifiedEAAEFExpiredLaneRecoveryAdmission,
            }
            for item in recovery_admissions
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "dispatcher recovery artifacts are not exact typed admissions"
            )
        self._admission = admission
        self._authorization_client = authorization_client
        self._transport = transport
        self._journal = journal
        self._policy = _policy_from_capability(admission.operational_capability)
        lane = {
            "schema": EAAEF_DAEMON_LANE_BINDING_SCHEMA,
            "gateway_binding_cid": admission["gateway_binding_cid"],
            "owner_principal_did": admission["owner_principal_did"],
            "owner_session_id": admission["owner_session_id"],
            "owner_generation": admission["owner_generation"],
            "lane_session_id": admission["lane_session_id"],
            "lane_generation": admission["lane_generation"],
            "process_instance_id": admission["process_instance_id"],
            "fence_epoch": admission["fence_epoch"],
        }
        self._lane_binding = dict(
            eaaef_daemon_lane_binding_projection(
                lane, verified_capability=admission.operational_capability
            )
        )
        self._recovery_admissions = tuple(recovery_admissions)
        recovery_lanes: list[dict[str, Any]] = []
        for prior in recovery_admissions:
            prior_lane = {
                "schema": EAAEF_DAEMON_LANE_BINDING_SCHEMA,
                "gateway_binding_cid": prior["gateway_binding_cid"],
                "owner_principal_did": prior["owner_principal_did"],
                "owner_session_id": prior["owner_session_id"],
                "owner_generation": prior["owner_generation"],
                "lane_session_id": prior["lane_session_id"],
                "lane_generation": prior["lane_generation"],
                "process_instance_id": prior["process_instance_id"],
                "fence_epoch": prior["fence_epoch"],
            }
            recovery_lanes.append(
                dict(
                    eaaef_daemon_lane_binding_projection(
                        prior_lane,
                        verified_capability=admission.operational_capability,
                    )
                )
            )
        self._recovery_lane_bindings = tuple(recovery_lanes)
        self._authorities_by_attempt: dict[str, dict[str, Any]] = {}
        self._authorities_by_claim: dict[str, dict[str, Any]] = {}
        self._authorities_by_task: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._attached = False

    @property
    def gateway_binding_cid(self) -> str:
        return str(self._admission["gateway_binding_cid"])

    @property
    def attached(self) -> bool:
        return self._attached

    def attach(self) -> None:
        with self._lock:
            if self._transport._closed:
                raise EAAEFBootstrapRuntimeGatewayError(
                    "closed EAAEF command transport cannot be reattached"
                )
            self._reverify_admission()
            self._attached = True

    def close(self) -> None:
        with self._lock:
            try:
                # A fully qualified transport owns live client/extension
                # descriptors before the daemon attaches the gateway.  Close
                # those resources on every lifecycle exit, including a child
                # startup failure between bundle construction and attach.
                self._transport.close()
            finally:
                self._attached = False

    def _reverify_admission(self) -> VerifiedEAAEFLaneRuntimeAdmission:
        try:
            checked = self._admission.reverify(now_ms=time.time_ns() // 1_000_000)
        except EAAEFLaneGatewayAdmissionError as exc:
            raise EAAEFBootstrapRuntimeGatewayError(
                "lane runtime admission re-verification failed"
            ) from exc
        if (
            checked["lane_authority_cid"] != self._admission["lane_authority_cid"]
            or checked["merge_admission_cid"] != self._admission["merge_admission_cid"]
        ):
            raise EAAEFBootstrapRuntimeGatewayDiverged(
                "lane runtime admission changed after factory construction"
            )
        return checked

    @staticmethod
    def _identity_candidate(value: Mapping[str, Any]) -> dict[str, Any] | None:
        required = {
            "task_cid",
            "claim_id",
            "attempt_id",
            "attempt_number",
            "lease_id",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
        }
        if not required.issubset(value):
            return None
        return {name: value[name] for name in required}

    def _register_identity(self, value: Any) -> None:
        if isinstance(value, Mapping):
            plain = dict(value)
            candidate = self._identity_candidate(plain)
            if (
                candidate is not None
                and candidate["owner_session_id"] == self._lane_binding["lane_session_id"]
            ):
                authority = {
                    "schema": EAAEF_TASK_OPERATION_AUTHORITY_SCHEMA,
                    **candidate,
                    "daemon_lane_binding": dict(self._lane_binding),
                }
                checked = dict(
                    eaaef_task_operation_authority_projection(
                        authority,
                        verified_capability=self._admission.operational_capability,
                    )
                )
                self._authorities_by_attempt[checked["attempt_id"]] = checked
                self._authorities_by_claim[checked["claim_id"]] = checked
                self._authorities_by_task[checked["task_cid"]] = checked
            for nested in plain.values():
                self._register_identity(nested)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for nested in value:
                self._register_identity(nested)

    def _authority_from_arguments(
        self, operation: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        if "task_authority_binding" in arguments:
            raise EAAEFBootstrapRuntimeGatewayError("callers cannot inject task authority bindings")
        for name in ("claim", "lease", "attempt"):
            nested = arguments.get(name)
            if isinstance(nested, Mapping):
                self._register_identity(nested)
        for field, registry in (
            ("attempt_id", self._authorities_by_attempt),
            ("claim_id", self._authorities_by_claim),
            ("task_cid", self._authorities_by_task),
        ):
            identity = str(arguments.get(field) or "")
            if identity and identity in registry:
                return dict(registry[identity])
        for name in ("claim", "lease", "attempt"):
            nested = arguments.get(name)
            if isinstance(nested, Mapping):
                for field, registry in (
                    ("attempt_id", self._authorities_by_attempt),
                    ("claim_id", self._authorities_by_claim),
                    ("task_cid", self._authorities_by_task),
                ):
                    identity = str(nested.get(field) or "")
                    if identity and identity in registry:
                        return dict(registry[identity])
        raise EAAEFBootstrapRuntimeGatewayError(
            f"{operation} has no exact current claim/attempt authority"
        )

    def _authorized_arguments(self, operation: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        plain = _canonical_detached(dict(arguments), f"{operation} arguments")
        if operation in _EXCLUDED_OPERATIONS:
            raise EAAEFBootstrapExcludedOperation(
                f"{operation} is outside the exact EAAEF bootstrap R1 vocabulary"
            )
        operations = frozenset(self._admission.operational_capability["operations"])
        if operation not in operations:
            raise EAAEFBootstrapExcludedOperation(
                f"{operation} is outside the signed 31-operation vocabulary"
            )
        unscoped_event = (
            operation == "execution.record_event"
            and not str(plain.get("task_cid") or "")
            and not str(plain.get("attempt_id") or "")
        )
        if operation in _BOARD_OPERATIONS or unscoped_event:
            if operation in _LANE_BOUND_OPERATIONS:
                if "recovery_authority" in plain:
                    if operation != "execution.list_running_attempts" or set(plain) != {
                        "recovery_authority"
                    }:
                        raise EAAEFBootstrapRuntimeGatewayError(
                            "recovery authority is outside its exact read operation"
                        )
                else:
                    if "daemon_lane_binding" in plain:
                        raise EAAEFBootstrapRuntimeGatewayError(
                            "callers cannot inject daemon lane bindings"
                        )
                    plain["daemon_lane_binding"] = dict(self._lane_binding)
            return plain
        authority = self._authority_from_arguments(operation, plain)
        plain["task_authority_binding"] = authority
        return plain

    @staticmethod
    def _receipt_fields() -> frozenset[str]:
        return frozenset(
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

    def _decode_receipt(
        self,
        receipt: Mapping[str, Any],
        *,
        envelope: AuthorizedStateCommand,
        operation: str,
        intent_cid: str,
    ) -> Any:
        row = _exact(receipt, self._receipt_fields(), "owner command receipt")
        command = envelope.command
        exact_joins = {
            "submission_id": envelope.submission_id,
            "envelope_cid": envelope.envelope_cid,
            "request_id": envelope.request_id,
            "principal_did": envelope.principal_did,
            "approver_did": envelope.approver_did,
            "authority_ref_cid": envelope.authority_ref_cid,
            "lease_id": envelope.lease_id,
            "one_use_nonce": envelope.one_use_nonce,
            "command_id": command.command_id,
            "idempotency_key": command.idempotency_key,
            "fence_epoch": command.fence_epoch,
        }
        if any(row.get(name) != value for name, value in exact_joins.items()):
            raise EAAEFBootstrapRuntimeGatewayDiverged(
                "owner receipt differs from its exact authorized envelope"
            )
        integer_fields = (
            "revision",
            "generation",
            "fence_epoch",
            "submitted_at",
            "applied_at",
        )
        if (
            type(row["changed"]) is not bool
            or any(
                isinstance(row[name], bool) or not isinstance(row[name], int) or row[name] < 0
                for name in integer_fields
            )
            or row["generation"] != command.expected_generation
            or row["submitted_at"] < 1
            or row["applied_at"] < row["submitted_at"]
            or not isinstance(row["result_json"], str)
            or not isinstance(row["error"], str)
        ):
            raise EAAEFBootstrapRuntimeGatewayDiverged("owner receipt result metadata is invalid")
        outcome = str(row["outcome"] or "")
        if outcome not in {
            CommandOutcome.ACCEPTED.value,
            CommandOutcome.IDEMPOTENT_REPLAY.value,
        }:
            raise EAAEFBootstrapRuntimeGatewayError(
                "owner rejected EAAEF operation: "
                + str(row.get("error") or "unspecified owner rejection")
            )
        if row["error"]:
            raise EAAEFBootstrapRuntimeGatewayDiverged(
                "accepted owner receipt unexpectedly carries an error"
            )
        try:
            result = json.loads(str(row["result_json"] or "{}"))
        except (TypeError, ValueError) as exc:
            raise EAAEFBootstrapRuntimeGatewayDiverged("owner receipt result is corrupt") from exc
        if (
            not isinstance(result, dict)
            or set(result) != {"daemon_operation", "intent_cid", "value"}
            or result["daemon_operation"] != operation
            or result["intent_cid"] != intent_cid
        ):
            raise EAAEFBootstrapRuntimeGatewayDiverged(
                "owner receipt does not bind the exact EAAEF intent"
            )
        return result["value"]

    def _find_receipt(self, envelope: AuthorizedStateCommand) -> Mapping[str, Any] | None:
        receipt = self._transport.lookup_receipt(envelope)
        if receipt is None:
            return None
        if not isinstance(receipt, Mapping):
            raise EAAEFBootstrapRuntimeGatewayDiverged(
                "owner receipt lookup returned a non-record"
            )
        if receipt.get("submission_id") != envelope.submission_id:
            raise EAAEFBootstrapRuntimeGatewayDiverged(
                "owner receipt lookup returned a different submission"
            )
        if receipt.get("envelope_cid") != envelope.envelope_cid:
            raise EAAEFBootstrapRuntimeGatewayDiverged(
                "ordinary submission_id collision has a different envelope"
            )
        return receipt

    def _finish_receipt(
        self,
        receipt: Mapping[str, Any],
        *,
        operation_key: str,
        operation: str,
        intent_cid: str,
        envelope: AuthorizedStateCommand,
    ) -> Any:
        # Validate before persistence so a malformed projection row cannot
        # permanently poison the exact-envelope continuation.
        self._decode_receipt(
            receipt,
            envelope=envelope,
            operation=operation,
            intent_cid=intent_cid,
        )
        durable = self._journal.commit_receipt(
            operation_key=operation_key,
            operation=operation,
            intent_cid=intent_cid,
            envelope=envelope,
            receipt=receipt,
        )
        value = self._decode_receipt(
            durable,
            envelope=envelope,
            operation=operation,
            intent_cid=intent_cid,
        )
        if quack_daemon_operation_command_vocabulary()[operation] == CommandKind.OBSERVE.value:
            self._journal.clear_observation(
                operation_key=operation_key,
                operation=operation,
                intent_cid=intent_cid,
            )
        self._register_identity(value)
        return value

    def _resolve_pending(
        self,
        checked: VerifiedEAAEFLaneRuntimeAdmission,
    ) -> None:
        """Adopt or replay the lane's sole exact envelope before new work."""

        pending = self._journal.pending()
        if pending is None:
            return
        operation_key, operation, intent_cid, envelope = pending
        visible = self._find_receipt(envelope)
        if visible is not None:
            self._finish_receipt(
                visible,
                operation_key=operation_key,
                operation=operation,
                intent_cid=intent_cid,
                envelope=envelope,
            )
            return
        try:
            intent = quack_daemon_operation_intent_from_envelope(envelope)
            verify_eaaef_bootstrap_operation_submission(
                envelope,
                intent,
                verified_capability=checked.operational_capability,
                authorization_policy=self._policy,
                now_ms=time.time_ns() // 1_000_000,
            )
        except Exception as exc:
            raise EAAEFBootstrapRuntimeGatewayAmbiguous(
                "unresolved exact lane envelope is no longer live"
            ) from exc
        append_error: Exception | None = None
        try:
            self._transport.append(envelope)
        except Exception as exc:
            append_error = exc
        authorized_wait_ms = min(
            self._transport._maximum_wait_ms,
            max(0, int(envelope.deadline_ms) - time.time_ns() // 1_000_000),
        )
        stop = time.monotonic_ns() // 1_000_000 + authorized_wait_ms
        while True:
            visible = self._find_receipt(envelope)
            if visible is not None:
                self._finish_receipt(
                    visible,
                    operation_key=operation_key,
                    operation=operation,
                    intent_cid=intent_cid,
                    envelope=envelope,
                )
                return
            remaining = stop - time.monotonic_ns() // 1_000_000
            if remaining <= 0:
                break
            time.sleep(min(self._transport._poll_interval_ms, remaining) / 1000)
        message = "unresolved exact lane envelope has no durable owner receipt"
        if append_error is not None:
            raise EAAEFBootstrapRuntimeGatewayAmbiguous(message) from append_error
        raise EAAEFBootstrapRuntimeGatewayAmbiguous(message)

    def dispatch(self, operation: str, arguments: Mapping[str, Any]) -> Any:
        with self._lock:
            if not self._attached:
                raise EAAEFBootstrapRuntimeGatewayError("EAAEF command gateway is not attached")
            checked = self._reverify_admission()
            self._resolve_pending(checked)
            authorized_arguments = self._authorized_arguments(operation, arguments)
            intent = quack_daemon_operation_intent(
                gateway_binding_cid=str(checked["gateway_binding_cid"]),
                operational_capability_cid=str(checked["operational_capability_cid"]),
                operation=operation,
                arguments=authorized_arguments,
            )
            intent_cid = _sha(intent["intent_cid"], "intent_cid")
            operation_key = _content_cid(
                {
                    "schema": "eaaef-exact-operation-key@1",
                    "lane_authority_cid": checked["lane_authority_cid"],
                    "operation": operation,
                    "arguments": authorized_arguments,
                },
                "exact operation key",
            )
            journaled = self._journal.lookup(
                operation_key=operation_key,
                operation=operation,
                intent_cid=intent_cid,
            )
            if journaled is None:
                envelope = self._authorization_client.authorize(intent)
                if (
                    type(envelope) is not AuthorizedStateCommand
                    or type(envelope.command) is not StateCommand
                ):
                    raise EAAEFBootstrapRuntimeGatewayError(
                        "command signer returned a non-exact envelope"
                    )
                now_ms = time.time_ns() // 1_000_000
                verify_eaaef_bootstrap_operation_submission(
                    envelope,
                    intent,
                    verified_capability=checked.operational_capability,
                    authorization_policy=self._policy,
                    now_ms=now_ms,
                )
                envelope = self._journal.prepare(
                    operation_key=operation_key,
                    operation=operation,
                    intent_cid=intent_cid,
                    envelope=envelope,
                )
                receipt = None
            else:
                envelope, receipt = journaled
            if receipt is not None:
                return self._finish_receipt(
                    receipt,
                    operation_key=operation_key,
                    envelope=envelope,
                    operation=operation,
                    intent_cid=intent_cid,
                )
            # Durable owner adoption precedes current freshness and lease
            # checks.  An exact receipt remains authoritative after the
            # response-lost envelope itself expires.
            visible = self._find_receipt(envelope)
            if visible is not None:
                return self._finish_receipt(
                    visible,
                    operation_key=operation_key,
                    operation=operation,
                    intent_cid=intent_cid,
                    envelope=envelope,
                )
            try:
                verify_eaaef_bootstrap_operation_submission(
                    envelope,
                    intent,
                    verified_capability=checked.operational_capability,
                    authorization_policy=self._policy,
                    now_ms=time.time_ns() // 1_000_000,
                )
            except Exception as exc:
                raise EAAEFBootstrapRuntimeGatewayAmbiguous(
                    "prepared exact envelope is no longer live; minting a replacement "
                    "would make owner application ambiguous"
                ) from exc
            append_error: BaseException | None = None
            try:
                self._transport.append(envelope)
            except BaseException as exc:
                append_error = exc
            authorized_wait_ms = min(
                self._transport._maximum_wait_ms,
                max(0, int(envelope.deadline_ms) - time.time_ns() // 1_000_000),
            )
            stop = time.monotonic_ns() // 1_000_000 + authorized_wait_ms
            while True:
                receipt = self._find_receipt(envelope)
                if receipt is not None:
                    return self._finish_receipt(
                        receipt,
                        operation_key=operation_key,
                        operation=operation,
                        intent_cid=intent_cid,
                        envelope=envelope,
                    )
                remaining = stop - time.monotonic_ns() // 1_000_000
                if remaining <= 0:
                    break
                time.sleep(min(self._transport._poll_interval_ms, remaining) / 1000)
            message = "exact EAAEF envelope has no durable owner receipt"
            if append_error is not None:
                raise EAAEFBootstrapRuntimeGatewayAmbiguous(message) from append_error
            raise EAAEFBootstrapRuntimeGatewayAmbiguous(message)

    def expired_running_arguments(self, *, limit: int, now_ms: int) -> Mapping[str, Any]:
        if not self._recovery_lane_bindings:
            raise EAAEFBootstrapRuntimeGatewayError(
                "dead-lane recovery requires prior typed lane admissions"
            )
        wall_now_ms = time.time_ns() // 1_000_000
        query_now_ms = _positive(now_ms, "recovery now_ms")
        if query_now_ms > wall_now_ms:
            raise EAAEFBootstrapRuntimeGatewayError(
                "dead-lane recovery cannot observe a future expiry frontier"
            )
        for artifact, expected_lane in zip(
            self._recovery_admissions,
            self._recovery_lane_bindings,
            strict=True,
        ):
            try:
                if type(artifact) is VerifiedEAAEFLaneRuntimeAdmission:
                    checked = artifact.reverify(now_ms=wall_now_ms)
                else:
                    checked = artifact.reverify_for_recovery(now_ms=wall_now_ms)
            except EAAEFLaneGatewayAdmissionError as exc:
                raise EAAEFBootstrapRuntimeGatewayError(
                    "dead-lane recovery admission failed source re-verification"
                ) from exc
            observed_lane = {
                "schema": EAAEF_DAEMON_LANE_BINDING_SCHEMA,
                "gateway_binding_cid": checked["gateway_binding_cid"],
                "owner_principal_did": checked["owner_principal_did"],
                "owner_session_id": checked["owner_session_id"],
                "owner_generation": checked["owner_generation"],
                "lane_session_id": checked["lane_session_id"],
                "lane_generation": checked["lane_generation"],
                "process_instance_id": checked["process_instance_id"],
                "fence_epoch": checked["fence_epoch"],
            }
            if observed_lane != expected_lane:
                raise EAAEFBootstrapRuntimeGatewayDiverged(
                    "dead-lane recovery binding changed after construction"
                )
        return eaaef_dead_lane_recovery_arguments(
            lane_bindings=self._recovery_lane_bindings,
            limit=limit,
            now_ms=query_now_ms,
            verified_capability=self._admission.operational_capability,
        )


class _EAAEFComponent:
    GATEWAY_COMPONENT_INTERFACE: ClassVar[str] = QUACK_DAEMON_GATEWAY_COMPONENT_INTERFACE
    __slots__ = ("_dispatcher", "gateway_binding_cid")

    def __init__(self, dispatcher: _EAAEFCommandDispatcher) -> None:
        self._dispatcher = dispatcher
        self.gateway_binding_cid = dispatcher.gateway_binding_cid

    def attach(self) -> None:
        # The composite gateway owns the one shared transport lifecycle.
        return None

    def close(self) -> None:
        return None


class EAAEFBootstrapTaskSourceProxy(_EAAEFComponent):
    """Closed EAAEF task surface; generic materialize/list stay excluded."""

    def materialize(self, *_args: Any, **_kwargs: Any) -> None:
        raise EAAEFBootstrapExcludedOperation("task.materialize is outside bootstrap R1")

    def list_tasks(self, *, limit: int) -> None:
        del limit
        raise EAAEFBootstrapExcludedOperation("task.list is outside bootstrap R1")

    def ready_tasks(self, *, limit: int) -> _GatewayTaskPage:
        value = self._dispatcher.dispatch("task.ready", {"limit": int(limit)})
        records = value.get("tasks", ()) if isinstance(value, Mapping) else value
        return _GatewayTaskPage(tuple(records or ()))

    def get(self, task_cid: str) -> _GatewayRecord | None:
        value = self._dispatcher.dispatch("task.get", {"task_cid": task_cid})
        return None if value is None else _GatewayRecord(value)

    def compare_and_set_status(self, task_cid: str, **kwargs: Any) -> _GatewayRecord:
        value = self._dispatcher.dispatch("task.cas_status", {"task_cid": task_cid, **kwargs})
        return _GatewayRecord(value)

    def record_validation_result(self, **kwargs: Any) -> Mapping[str, Any]:
        value = self._dispatcher.dispatch("task.record_validation", kwargs)
        return MappingProxyType(dict(value or {}))


class EAAEFBootstrapCoordinatorProxy(_EAAEFComponent):
    def register_task(self, **kwargs: Any) -> Any:
        return self._dispatcher.dispatch("coordination.register_task", kwargs)

    def claim_ready_task(self, **kwargs: Any) -> _GatewayRecord | None:
        value = self._dispatcher.dispatch("coordination.claim_ready", kwargs)
        return None if value is None else _GatewayRecord(value)

    def get_task_claim(self, claim_id: str) -> _GatewayRecord | None:
        value = self._dispatcher.dispatch("coordination.get_claim", {"claim_id": claim_id})
        return None if value is None else _GatewayRecord(value)

    def protect_task_claim(self, claim: Any, **kwargs: Any) -> _GatewayRecord:
        value = self._dispatcher.dispatch(
            "coordination.protect_claim",
            {"claim": _plain_mapping(claim, "claim"), **kwargs},
        )
        return _GatewayRecord(value)

    def renew(self, lease: Any, **kwargs: Any) -> _GatewayRecord:
        value = self._dispatcher.dispatch(
            "coordination.renew_lease",
            {"lease": _plain_mapping(lease, "lease"), **kwargs},
        )
        return _GatewayRecord(value)

    def prepare_task_completion(self, claim: Any, **kwargs: Any) -> Any:
        return self._dispatcher.dispatch(
            "coordination.prepare_completion",
            {"claim": _plain_mapping(claim, "claim"), **kwargs},
        )

    def get_prepared_task_completion(self, task_cid: str) -> Any:
        return self._dispatcher.dispatch(
            "coordination.get_prepared_completion", {"task_cid": task_cid}
        )

    def complete_task_claim(self, claim: Any, **kwargs: Any) -> Any:
        return self._dispatcher.dispatch(
            "coordination.complete_claim",
            {"claim": _plain_mapping(claim, "claim"), **kwargs},
        )

    def settle_task_claim(self, claim: Any, **kwargs: Any) -> Any:
        return self._dispatcher.dispatch(
            "coordination.settle_claim",
            {"claim": _plain_mapping(claim, "claim"), **kwargs},
        )

    def list_unsettled_task_completions(self, **kwargs: Any) -> Any:
        return self._dispatcher.dispatch("coordination.list_unsettled_completions", kwargs)

    def reconcile_promoted_task_completion(self, task_cid: str, **kwargs: Any) -> Any:
        return self._dispatcher.dispatch(
            "coordination.reconcile_promoted_completion",
            {"task_cid": task_cid, **kwargs},
        )

    def recover_prepared_task_completion(self, task_cid: str, **kwargs: Any) -> Any:
        return self._dispatcher.dispatch(
            "coordination.recover_prepared_completion",
            {"task_cid": task_cid, **kwargs},
        )

    def abort_prepared_task_completion(self, task_cid: str, **kwargs: Any) -> Any:
        return self._dispatcher.dispatch(
            "coordination.abort_prepared_completion",
            {"task_cid": task_cid, **kwargs},
        )

    def expire_task_claim(self, claim: Any, **kwargs: Any) -> Any:
        return self._dispatcher.dispatch(
            "coordination.expire_claim",
            {"claim": _plain_mapping(claim, "claim"), **kwargs},
        )


_EXECUTION_PROXY_FACTORY_TOKEN = object()


class EAAEFBootstrapExecutionRepositoryProxy(_EAAEFComponent):
    """Exact EAAEF execution surface sharing the composite R1 dispatcher."""

    INTERFACE: ClassVar[str] = EAAEF_BOOTSTRAP_EXECUTION_REPOSITORY_PROXY_INTERFACE
    EAAEF_INTERFACE: ClassVar[str] = EAAEF_BOOTSTRAP_EXECUTION_REPOSITORY_PROXY_INTERFACE
    SCHEMA: ClassVar[str] = EAAEF_BOOTSTRAP_EXECUTION_REPOSITORY_PROXY_SCHEMA
    QUALIFICATION_STATUS: ClassVar[str] = EAAEF_BOOTSTRAP_RUNTIME_GATEWAY_QUALIFICATION_STATUS

    __slots__ = ()

    def __init__(
        self,
        token: object | None = None,
        dispatcher: _EAAEFCommandDispatcher | None = None,
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        if token is not _EXECUTION_PROXY_FACTORY_TOKEN:
            raise EAAEFBootstrapRuntimeGatewayNoGo()
        if type(dispatcher) is not _EAAEFCommandDispatcher:
            raise EAAEFBootstrapRuntimeGatewayError(
                "execution proxy requires the exact shared dispatcher"
            )
        super().__init__(dispatcher)

    @classmethod
    def from_unqualified_runtime(cls, *_args: Any, **_kwargs: Any) -> None:
        """Return the typed no-go until all external runtime evidence exists."""

        raise EAAEFBootstrapRuntimeGatewayNoGo()

    def bind_daemon(self, metadata: Mapping[str, Any]) -> Any:
        return self._dispatcher.dispatch("execution.bind_daemon", {"metadata": metadata})

    def list_running_attempts(self, **kwargs: Any) -> Any:
        return self._dispatcher.dispatch("execution.list_running_attempts", kwargs)

    def list_expired_running_attempts(self, *, limit: int, now_ms: int) -> list[Any]:
        arguments = self._dispatcher.expired_running_arguments(limit=limit, now_ms=now_ms)
        value = self._dispatcher.dispatch("execution.list_running_attempts", arguments)
        return list(value or ())

    def record_event(self, **kwargs: Any) -> Any:
        return self._dispatcher.dispatch("execution.record_event", kwargs)

    def ensure_attempt(self, **kwargs: Any) -> Any:
        return self._dispatcher.dispatch("execution.ensure_attempt", kwargs)

    def get_attempt(self, attempt_id: str) -> Any:
        return self._dispatcher.dispatch("execution.get_attempt", {"attempt_id": attempt_id})

    def commit_phase(self, **kwargs: Any) -> Any:
        return self._dispatcher.dispatch("execution.commit_phase", kwargs)

    def commit_reconciled_attempt(self, **kwargs: Any) -> Any:
        return self._dispatcher.dispatch("execution.commit_reconciled_attempt", kwargs)

    def phase_history(self, attempt_id: str) -> Any:
        return self._dispatcher.dispatch("execution.phase_history", {"attempt_id": attempt_id})

    def get_idempotent_result(self, **kwargs: Any) -> Any:
        kind = str(kwargs.get("kind") or "")
        if kind not in {"provider", "effect"}:
            raise EAAEFBootstrapRuntimeGatewayError(
                "idempotent reservation kind is outside provider/effect"
            )
        return self._dispatcher.dispatch(f"{kind}.reserve", kwargs)

    def record_idempotent_result(self, **kwargs: Any) -> Any:
        kind = str(kwargs.get("kind") or "")
        if kind not in {"provider", "effect"}:
            raise EAAEFBootstrapRuntimeGatewayError(
                "idempotent commit kind is outside provider/effect"
            )
        return self._dispatcher.dispatch(f"{kind}.commit", kwargs)

    def reserve_provider(self, **kwargs: Any) -> Any:
        return self._dispatcher.dispatch("provider.reserve", kwargs)

    def commit_provider(self, **kwargs: Any) -> Any:
        return self._dispatcher.dispatch("provider.commit", kwargs)

    def reserve_effect(self, **kwargs: Any) -> Any:
        return self._dispatcher.dispatch("effect.reserve", kwargs)

    def commit_effect(self, **kwargs: Any) -> Any:
        return self._dispatcher.dispatch("effect.commit", kwargs)

    def record_validation(self, **kwargs: Any) -> Any:
        return self._dispatcher.dispatch("validation.record", kwargs)


class EAAEFBootstrapGatewayCapability:
    """Read-only capability projection consumed by DatabaseImplementationDaemon."""

    __slots__ = (
        "board_namespace",
        "shard_id",
        "store_id",
        "control_plane_schema_version",
        "state_schema_revision",
        "command_endpoint",
        "state_endpoint",
        "owner_principal_did",
        "owner_generation",
        "fence_epoch",
        "authorization_policy_cid",
        "command_fabric_qualification_cid",
        "operational_capability_cid",
        "operations",
        "production_admitted",
        "_gateway_binding_cid",
    )

    def __init__(self, admission: VerifiedEAAEFLaneRuntimeAdmission) -> None:
        capability = admission.operational_capability
        for name in (
            "board_namespace",
            "shard_id",
            "store_id",
            "control_plane_schema_version",
            "state_schema_revision",
            "command_endpoint",
            "state_endpoint",
            "owner_principal_did",
            "authorization_policy_cid",
            "command_fabric_qualification_cid",
        ):
            setattr(self, name, str(capability[name]))
        self.owner_generation = int(capability["owner_generation"])
        self.fence_epoch = int(capability["fence_epoch"])
        self.operational_capability_cid = str(capability["capability_cid"])
        self.operations = frozenset(capability["operations"])
        self.production_admitted = False
        self._gateway_binding_cid = str(capability["gateway_binding_cid"])

    @property
    def content_id(self) -> str:
        return self._gateway_binding_cid


_COMMAND_GATEWAY_FACTORY_TOKEN = object()


class EAAEFBootstrapCommandGateway(QuackDaemonCommandGateway):
    """Mutually exclusive R1 gateway satisfying the daemon's shared boundary."""

    INTERFACE: ClassVar[str] = EAAEF_BOOTSTRAP_COMMAND_GATEWAY_INTERFACE

    def __init__(
        self,
        token: object,
        *,
        admission: VerifiedEAAEFLaneRuntimeAdmission,
        dispatcher: _EAAEFCommandDispatcher,
    ) -> None:
        if token is not _COMMAND_GATEWAY_FACTORY_TOKEN:
            raise EAAEFBootstrapRuntimeGatewayNoGo()
        if (
            type(admission) is not VerifiedEAAEFLaneRuntimeAdmission
            or type(dispatcher) is not _EAAEFCommandDispatcher
            or dispatcher._admission is not admission
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "command gateway requires its exact factory-built dispatcher"
            )
        self._admission = admission
        self._dispatcher = dispatcher
        self.capability = EAAEFBootstrapGatewayCapability(admission)
        self.task_source = EAAEFBootstrapTaskSourceProxy(dispatcher)
        self.coordinator = EAAEFBootstrapCoordinatorProxy(dispatcher)
        self.execution_repository = EAAEFBootstrapExecutionRepositoryProxy(
            _EXECUTION_PROXY_FACTORY_TOKEN, dispatcher
        )
        self.merge_repository = None
        self.plan_repository = None
        self._attached = False
        self._runtime_dependencies = None
        self._container_dispatcher = None

    @property
    def attached(self) -> bool:
        return self._attached

    def _validate_components(self) -> None:
        expected = (
            (self.task_source, EAAEFBootstrapTaskSourceProxy),
            (self.coordinator, EAAEFBootstrapCoordinatorProxy),
            (self.execution_repository, EAAEFBootstrapExecutionRepositoryProxy),
        )
        for component, expected_type in expected:
            if (
                type(component) is not expected_type
                or component._dispatcher is not self._dispatcher
                or component.gateway_binding_cid != self.capability.content_id
                or component.GATEWAY_COMPONENT_INTERFACE != QUACK_DAEMON_GATEWAY_COMPONENT_INTERFACE
            ):
                raise QuackDaemonGatewayError(
                    "EAAEF gateway components do not share one exact dispatcher"
                )
        if self.merge_repository is not None or self.plan_repository is not None:
            raise QuackDaemonGatewayError(
                "EAAEF bootstrap R1 cannot carry merge or Plan-R2 components"
            )

    def require_production_admission(self) -> Mapping[str, Any]:
        if type(self._dispatcher._transport) is EAAEFTypedOwnerCommandTransport:
            raise QuackDaemonGatewayError(
                "EAAEF typed-owner transport remains production no-go: "
                + EAAEF_TYPED_OWNER_TRANSPORT_PRODUCTION_BLOCKER
            )
        try:
            checked = self._admission.reverify(now_ms=time.time_ns() // 1_000_000)
        except EAAEFLaneGatewayAdmissionError as exc:
            raise QuackDaemonGatewayError(
                "EAAEF lane production admission re-verification failed"
            ) from exc
        if checked["gateway_binding_cid"] != self.capability.content_id:
            raise QuackDaemonGatewayError("EAAEF lane gateway binding changed after construction")
        dependencies = self._runtime_dependencies
        if type(dependencies) is not EAAEFLaneRuntimeDependencyFactory:
            raise QuackDaemonGatewayError(
                "EAAEF bootstrap runtime remains production no-go: "
                + ",".join(EAAEF_BOOTSTRAP_RUNTIME_GATEWAY_PRODUCTION_BLOCKERS)
            )
        try:
            evidence = dependencies._reverify_built_gateway(self)
        except EAAEFBootstrapRuntimeGatewayError as exc:
            raise QuackDaemonGatewayError(
                "EAAEF composite runtime dependency re-verification failed"
            ) from exc
        return MappingProxyType(evidence)

    def attach(self) -> None:
        if self._attached:
            return
        self._validate_components()
        try:
            checked = self._admission.reverify(now_ms=time.time_ns() // 1_000_000)
            if checked["gateway_binding_cid"] != self.capability.content_id:
                raise EAAEFBootstrapRuntimeGatewayDiverged(
                    "EAAEF lane gateway binding changed before attach"
                )
            self._dispatcher.attach()
        except (EAAEFBootstrapRuntimeGatewayError, EAAEFLaneGatewayAdmissionError) as exc:
            raise QuackDaemonGatewayError("EAAEF gateway failed attach") from exc
        self._attached = True

    def close(self) -> None:
        try:
            self._dispatcher.close()
        except EAAEFBootstrapRuntimeGatewayError as exc:
            raise QuackDaemonGatewayError("EAAEF gateway failed close") from exc
        finally:
            self._attached = False

    def evidence(self) -> Mapping[str, Any]:
        typed_owner_transport = (
            type(self._dispatcher._transport)
            is EAAEFTypedOwnerCommandTransport
        )
        blockers = list(EAAEF_BOOTSTRAP_RUNTIME_GATEWAY_PRODUCTION_BLOCKERS)
        if typed_owner_transport:
            blockers.append(EAAEF_TYPED_OWNER_TRANSPORT_PRODUCTION_BLOCKER)
        return MappingProxyType(
            {
                "interface": self.INTERFACE,
                "qualification_status": (EAAEF_BOOTSTRAP_RUNTIME_GATEWAY_QUALIFICATION_STATUS),
                "gateway_binding_cid": self.capability.content_id,
                "operational_capability_cid": (self.capability.operational_capability_cid),
                "lane_authority_cid": self._admission["lane_authority_cid"],
                "verifier_receipt_cid": self._admission["verifier_receipt_cid"],
                "merge_admission_cid": self._admission["merge_admission_cid"],
                "operations": sorted(self.capability.operations),
                "excluded_operations": sorted(_EXCLUDED_OPERATIONS),
                "attached": self.attached,
                "direct_database_open": False,
                "arbitrary_sql_enabled": False,
                "transport": (
                    "typed_state_owner"
                    if typed_owner_transport
                    else "legacy_quack_clients"
                ),
                "production_blockers": (
                    []
                    if self.capability.production_admitted
                    else blockers
                ),
            }
        )


def create_eaaef_bootstrap_command_gateway(
    *,
    admission: VerifiedEAAEFLaneRuntimeAdmission,
    authorization_client: EAAEFCommandAuthorizationServiceClient,
    transport: EAAEFBootstrapCommandTransport | EAAEFTypedOwnerCommandTransport,
    journal: EAAEFExactEnvelopeJournal,
    recovery_admissions: Sequence[
        VerifiedEAAEFLaneRuntimeAdmission | VerifiedEAAEFExpiredLaneRecoveryAdmission
    ] = (),
) -> EAAEFBootstrapCommandGateway:
    """Construct R1 only from exact independently verified dependencies."""

    if (
        type(admission) is not VerifiedEAAEFLaneRuntimeAdmission
        or type(authorization_client) is not EAAEFCommandAuthorizationServiceClient
        or type(transport) not in _EAAEF_COMMAND_TRANSPORT_TYPES
        or type(journal) is not EAAEFExactEnvelopeJournal
    ):
        raise EAAEFBootstrapRuntimeGatewayError(
            "gateway factory rejects mappings, callbacks, tokens, and duck types"
        )
    if type(recovery_admissions) not in {tuple, list}:
        raise EAAEFBootstrapRuntimeGatewayError(
            "recovery admissions must be exact typed lane artifacts"
        )
    prior = tuple(recovery_admissions)
    if len(prior) > _MAX_RECOVERY_LANES:
        raise EAAEFBootstrapRuntimeGatewayError("recovery admissions exceed the five-lane frontier")
    recovery_types = (
        VerifiedEAAEFLaneRuntimeAdmission,
        VerifiedEAAEFExpiredLaneRecoveryAdmission,
    )
    if any(type(item) not in recovery_types for item in prior):
        raise EAAEFBootstrapRuntimeGatewayError("recovery admissions contain an untyped artifact")
    now_ms = time.time_ns() // 1_000_000
    try:
        checked = admission.reverify(now_ms=now_ms)
        authorization_verified, _policy = authorization_client._reverify()
    except Exception as exc:
        # Exact type checks above exclude arbitrary callback execution; the
        # broad catch normalizes transport/signer verification failures.
        raise EAAEFBootstrapRuntimeGatewayError(
            "gateway dependency re-verification failed"
        ) from exc
    if (
        authorization_verified["capability_cid"] != checked["operational_capability_cid"]
        or transport.admission_cid != checked["merge_admission_cid"]
        or journal._admission_cid != checked["merge_admission_cid"]
        or journal._lane_authority_cid != checked["lane_authority_cid"]
    ):
        raise EAAEFBootstrapRuntimeGatewayError(
            "gateway dependencies are not bound to one exact lane admission"
        )
    sessions: set[str] = {str(checked["lane_session_id"])}
    for item in prior:
        try:
            if type(item) is VerifiedEAAEFLaneRuntimeAdmission:
                verified_prior = item.reverify(now_ms=now_ms)
            else:
                verified_prior = item.reverify_for_recovery(now_ms=now_ms)
        except EAAEFLaneGatewayAdmissionError as exc:
            raise EAAEFBootstrapRuntimeGatewayError(
                "dead-lane recovery admission failed source re-verification"
            ) from exc
        if (
            verified_prior["gateway_binding_cid"] != checked["gateway_binding_cid"]
            or verified_prior["owner_principal_did"] != checked["owner_principal_did"]
            or verified_prior["owner_session_id"] != checked["owner_session_id"]
            or verified_prior["owner_generation"] != checked["owner_generation"]
            or verified_prior["fence_epoch"] != checked["fence_epoch"]
            or str(verified_prior["lane_session_id"]) in sessions
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "dead-lane recovery admission differs or repeats a lane session"
            )
        sessions.add(str(verified_prior["lane_session_id"]))
    dispatcher = _EAAEFCommandDispatcher(
        admission=checked,
        authorization_client=authorization_client,
        transport=transport,
        journal=journal,
        recovery_admissions=prior,
    )
    return EAAEFBootstrapCommandGateway(
        _COMMAND_GATEWAY_FACTORY_TOKEN,
        admission=checked,
        dispatcher=dispatcher,
    )


_RUNTIME_DEPENDENCY_FACTORY_TOKEN = object()
_RUNTIME_DEPENDENCY_BUNDLE_TOKEN = object()


class EAAEFLaneRuntimeDependencyBundle:
    """Constructed R1 gateway and its sole dynamic container dispatcher."""

    __slots__ = ("gateway", "container_dispatcher", "process_birth")

    def __init__(
        self,
        token: object,
        *,
        gateway: EAAEFBootstrapCommandGateway,
        container_dispatcher: ExternalAgentContainerWorkerDispatcher,
        process_birth: VerifiedEAAEFProcessBirth,
    ) -> None:
        if token is not _RUNTIME_DEPENDENCY_BUNDLE_TOKEN:
            raise TypeError("EAAEF runtime bundles come from the exact dependency factory")
        if (
            type(gateway) is not EAAEFBootstrapCommandGateway
            or type(container_dispatcher) is not ExternalAgentContainerWorkerDispatcher
            or type(process_birth) is not VerifiedEAAEFProcessBirth
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "runtime bundle rejects substitute dependencies"
            )
        self.gateway = gateway
        self.container_dispatcher = container_dispatcher
        self.process_birth = process_birth

    def close(self) -> None:
        """Release qualified client descriptors, even before daemon attach."""

        self.gateway.close()


class EAAEFLaneRuntimeDependencyFactory:
    """Lazy child-side composition; construction performs no Quack/service I/O."""

    __slots__ = (
        "_admission",
        "_process_birth",
        "_native_admission",
        "_native_launch",
        "_native_module",
        "_quack_qualification",
        "_sealed_descriptors",
        "_dispatcher_qualification",
        "_dispatcher_factory",
        "_authorization_client",
        "_journal_parent_directory",
        "_recovery_admissions",
        "_maximum_wait_ms",
        "_poll_interval_ms",
        "_built",
    )

    def __init__(
        self,
        token: object,
        *,
        admission: VerifiedEAAEFLaneRuntimeAdmission,
        process_birth: VerifiedEAAEFProcessBirth,
        native_admission: VerifiedAgentSupervisorNativeDependencyAdmission,
        native_launch: AgentSupervisorNativeDependencyLaunch,
        native_module: object,
        quack_qualification: VerifiedEAAEFQuackClientFactoryQualification,
        sealed_descriptors: EAAEFSealedQuackClientDescriptors,
        dispatcher_qualification: VerifiedEAAEFContainerDispatcherFactoryQualification,
        dispatcher_factory: EAAEFContainerDispatcherFactory,
        authorization_client: EAAEFCommandAuthorizationServiceClient,
        journal_parent_directory: str | Path,
        recovery_admissions: tuple[
            VerifiedEAAEFLaneRuntimeAdmission | VerifiedEAAEFExpiredLaneRecoveryAdmission,
            ...,
        ],
        maximum_wait_ms: int,
        poll_interval_ms: int,
    ) -> None:
        if token is not _RUNTIME_DEPENDENCY_FACTORY_TOKEN:
            raise TypeError("runtime dependency factories come from the exact binder")
        self._admission = admission
        self._process_birth = process_birth
        self._native_admission = native_admission
        self._native_launch = native_launch
        self._native_module = native_module
        self._quack_qualification = quack_qualification
        self._sealed_descriptors = sealed_descriptors
        self._dispatcher_qualification = dispatcher_qualification
        self._dispatcher_factory = dispatcher_factory
        self._authorization_client = authorization_client
        self._journal_parent_directory = Path(
            os.path.abspath(os.fspath(journal_parent_directory))
        )
        self._recovery_admissions = recovery_admissions
        self._maximum_wait_ms = maximum_wait_ms
        self._poll_interval_ms = poll_interval_ms
        self._built = False

    @property
    def journal_relative_path(self) -> Path:
        return eaaef_exact_envelope_journal_relative_path(self._admission)

    def _reverify_dependencies(self) -> dict[str, Any]:
        now_ms = time.time_ns() // 1_000_000
        try:
            lane = self._admission.reverify(now_ms=now_ms)
            native = self._native_admission.reverify(now_ms=now_ms)
            quack = self._quack_qualification.reverify(now_ms=now_ms)
            dispatcher = self._dispatcher_qualification.reverify(now_ms=now_ms)
            operational, _policy = self._authorization_client._reverify()
        except (
            AgentSupervisorNativeDependencyAdmissionError,
            EAAEFLaneGatewayAdmissionError,
            QuackCommandAuthorizationError,
        ) as exc:
            raise EAAEFBootstrapRuntimeGatewayError(
                "runtime dependencies failed source/signature re-verification"
            ) from exc
        try:
            native_path = verify_agent_supervisor_native_dependency_sealed_fd(
                self._native_launch
            )
        except ValueError as exc:
            raise EAAEFBootstrapRuntimeGatewayError(
                "runtime native launch descriptor is invalid"
            ) from exc
        native_pin = native.native_dependency_pin
        birth = self._process_birth
        if (
            birth.admission_cid != lane["merge_admission_cid"]
            or birth["pid"] != os.getpid()
            or birth["process_start_time_ticks"] != _process_start_time_ticks(os.getpid())
            or birth["lane_session_id"] != lane["lane_session_id"]
            or birth["lane_generation"] != lane["lane_generation"]
            or birth["process_instance_id"] != lane["process_instance_id"]
            or birth["process_birth_nonce"] != lane["process_birth_nonce"]
            or native["admission_cid"] != lane["native_dependency_admission_cid"]
            or quack.qualification_cid
            != lane["quack_client_factory_qualification_cid"]
            or dispatcher.qualification_cid
            != lane["container_dispatcher_factory_qualification_cid"]
            or operational["capability_cid"] != lane["operational_capability_cid"]
            or self._native_launch.accepted_authorization_id != native["admission_cid"]
            or self._native_launch.pin != native_pin
            or sys.modules.get(native_pin.module_name) is not self._native_module
            or sys.modules.get(native_pin.public_alias) is not self._native_module
            or getattr(self._native_module, "__name__", None) != native_pin.module_name
            or getattr(self._native_module, "__file__", None) != native_path
            or getattr(self._native_module, "__version__", None)
            != native_pin.distribution_version
            or not callable(getattr(self._native_module, "connect", None))
            or self._sealed_descriptors._admission_cid != lane["merge_admission_cid"]
            or self._sealed_descriptors._birth_cid != birth["birth_cid"]
            or self._sealed_descriptors._used
            or self._dispatcher_factory._created
            or self._dispatcher_factory._admission["merge_admission_cid"]
            != lane["merge_admission_cid"]
            or self._dispatcher_factory.qualification_cid != dispatcher.qualification_cid
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "runtime dependencies do not share one exact signed process birth"
            )
        return {
            "lane": lane,
            "native": native,
            "quack": quack,
            "dispatcher": dispatcher,
        }

    def _reverify_built_gateway(
        self, gateway: EAAEFBootstrapCommandGateway
    ) -> dict[str, Any]:
        if (
            not self._built
            or type(gateway) is not EAAEFBootstrapCommandGateway
            or gateway._runtime_dependencies is not self
            or type(gateway._container_dispatcher)
            is not ExternalAgentContainerWorkerDispatcher
            or not self._sealed_descriptors._used
            or not self._dispatcher_factory._created
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "runtime gateway is not the factory's completed exact bundle"
            )
        now_ms = time.time_ns() // 1_000_000
        try:
            lane = self._admission.reverify(now_ms=now_ms)
            native = self._native_admission.reverify(now_ms=now_ms)
            quack = self._quack_qualification.reverify(now_ms=now_ms)
            dispatcher = self._dispatcher_qualification.reverify(now_ms=now_ms)
            operational, _policy = self._authorization_client._reverify()
        except Exception as exc:
            raise EAAEFBootstrapRuntimeGatewayError(
                "built runtime dependencies failed live re-verification"
            ) from exc
        transport = gateway._dispatcher._transport
        if (
            gateway._admission["merge_admission_cid"] != lane["merge_admission_cid"]
            or native["admission_cid"] != lane["native_dependency_admission_cid"]
            or quack.qualification_cid
            != lane["quack_client_factory_qualification_cid"]
            or dispatcher.qualification_cid
            != lane["container_dispatcher_factory_qualification_cid"]
            or operational["capability_cid"] != lane["operational_capability_cid"]
            or transport._birth_cid != self._process_birth["birth_cid"]
            or transport._native_admission_cid != native["admission_cid"]
            or transport._qualification_cid != quack.qualification_cid
            or transport._closed
        ):
            raise EAAEFBootstrapRuntimeGatewayError(
                "built runtime gateway diverged from its exact dependencies"
            )
        return {
            "interface": "EAAEFLaneRuntimeProductionAdmission@1",
            "lane_authority_cid": lane["lane_authority_cid"],
            "lane_merge_admission_cid": lane["merge_admission_cid"],
            "process_birth_cid": self._process_birth["birth_cid"],
            "native_dependency_admission_cid": native["admission_cid"],
            "quack_client_factory_qualification_cid": quack.qualification_cid,
            "container_dispatcher_factory_qualification_cid": (
                dispatcher.qualification_cid
            ),
            "gateway_binding_cid": lane["gateway_binding_cid"],
            "direct_database_open": False,
            "raw_token_available": False,
            "plan_r2_enabled": False,
        }

    @staticmethod
    def _discard_qualified_clients(clients: EAAEFQualifiedQuackClients) -> None:
        for client in (clients._read_client, clients._command_client):
            try:
                client.close()
            except BaseException:
                pass
        try:
            os.close(clients._extension_descriptor)
        except OSError:
            pass
        clients._consumed = True

    def build(self) -> EAAEFLaneRuntimeDependencyBundle:
        """Build but do not attach the R1 gateway or invoke a dynamic service."""

        if self._built:
            raise EAAEFBootstrapRuntimeGatewayError(
                "runtime dependency factory is single-use"
            )
        checked = self._reverify_dependencies()
        clients: EAAEFQualifiedQuackClients | None = None
        transport: EAAEFBootstrapCommandTransport | None = None
        gateway: EAAEFBootstrapCommandGateway | None = None
        try:
            clients = create_eaaef_qualified_quack_clients(
                admission=checked["lane"],
                process_birth=self._process_birth,
                native_admission=checked["native"],
                native_launch=self._native_launch,
                native_module=self._native_module,
                qualification=checked["quack"],
                sealed_descriptors=self._sealed_descriptors,
            )
            transport = bind_eaaef_qualified_bootstrap_command_transport(
                clients=clients,
                admission=checked["lane"],
                process_birth=self._process_birth,
                maximum_wait_ms=self._maximum_wait_ms,
                poll_interval_ms=self._poll_interval_ms,
            )
            journal = open_eaaef_exact_envelope_journal(
                self._journal_parent_directory,
                admission=checked["lane"],
            )
            gateway = create_eaaef_bootstrap_command_gateway(
                admission=checked["lane"],
                authorization_client=self._authorization_client,
                transport=transport,
                journal=journal,
                recovery_admissions=self._recovery_admissions,
            )
            container_dispatcher = self._dispatcher_factory.create(
                execution_repository=gateway.execution_repository
            )
        except BaseException:
            if gateway is not None:
                try:
                    gateway._dispatcher.close()
                except BaseException:
                    pass
            elif transport is not None:
                try:
                    transport.close()
                except BaseException:
                    pass
            elif clients is not None:
                self._discard_qualified_clients(clients)
            raise
        gateway._runtime_dependencies = self
        gateway._container_dispatcher = container_dispatcher
        gateway.capability.production_admitted = True
        self._built = True
        return EAAEFLaneRuntimeDependencyBundle(
            _RUNTIME_DEPENDENCY_BUNDLE_TOKEN,
            gateway=gateway,
            container_dispatcher=container_dispatcher,
            process_birth=self._process_birth,
        )


def create_eaaef_lane_runtime_dependency_factory(
    *,
    admission: VerifiedEAAEFLaneRuntimeAdmission,
    process_birth: VerifiedEAAEFProcessBirth,
    native_admission: VerifiedAgentSupervisorNativeDependencyAdmission,
    native_launch: AgentSupervisorNativeDependencyLaunch,
    native_module: object,
    quack_qualification: VerifiedEAAEFQuackClientFactoryQualification,
    sealed_descriptors: EAAEFSealedQuackClientDescriptors,
    dispatcher_qualification: VerifiedEAAEFContainerDispatcherFactoryQualification,
    authorization_client: EAAEFCommandAuthorizationServiceClient,
    journal_parent_directory: str | Path,
    recovery_admissions: Sequence[
        VerifiedEAAEFLaneRuntimeAdmission | VerifiedEAAEFExpiredLaneRecoveryAdmission
    ] = (),
    maximum_wait_ms: int = 30_000,
    poll_interval_ms: int = 10,
) -> EAAEFLaneRuntimeDependencyFactory:
    """Freeze exact dependencies without constructing clients or opening services."""

    exact = (
        type(admission) is VerifiedEAAEFLaneRuntimeAdmission,
        type(process_birth) is VerifiedEAAEFProcessBirth,
        type(native_admission) is VerifiedAgentSupervisorNativeDependencyAdmission,
        type(native_launch) is AgentSupervisorNativeDependencyLaunch,
        type(quack_qualification) is VerifiedEAAEFQuackClientFactoryQualification,
        type(sealed_descriptors) is EAAEFSealedQuackClientDescriptors,
        type(dispatcher_qualification)
        is VerifiedEAAEFContainerDispatcherFactoryQualification,
        type(authorization_client) is EAAEFCommandAuthorizationServiceClient,
        type(recovery_admissions) in {tuple, list},
    )
    if not all(exact):
        raise EAAEFBootstrapRuntimeGatewayError(
            "runtime dependency factory rejects mappings, callbacks, tokens, and substitutes"
        )
    prior = tuple(recovery_admissions)
    if len(prior) > _MAX_RECOVERY_LANES or any(
        type(item)
        not in {VerifiedEAAEFLaneRuntimeAdmission, VerifiedEAAEFExpiredLaneRecoveryAdmission}
        for item in prior
    ):
        raise EAAEFBootstrapRuntimeGatewayError(
            "runtime dependency factory recovery frontier is invalid"
        )
    wait = _positive(maximum_wait_ms, "maximum receipt wait", maximum=60_000)
    poll = _positive(poll_interval_ms, "receipt poll interval", maximum=wait)
    dispatcher_factory = create_eaaef_container_dispatcher_factory(
        admission=admission,
        process_birth=process_birth,
        native_admission=native_admission,
        quack_qualification=quack_qualification,
        qualification=dispatcher_qualification,
    )
    factory = EAAEFLaneRuntimeDependencyFactory(
        _RUNTIME_DEPENDENCY_FACTORY_TOKEN,
        admission=admission,
        process_birth=process_birth,
        native_admission=native_admission,
        native_launch=native_launch,
        native_module=native_module,
        quack_qualification=quack_qualification,
        sealed_descriptors=sealed_descriptors,
        dispatcher_qualification=dispatcher_qualification,
        dispatcher_factory=dispatcher_factory,
        authorization_client=authorization_client,
        journal_parent_directory=journal_parent_directory,
        recovery_admissions=prior,
        maximum_wait_ms=wait,
        poll_interval_ms=poll,
    )
    factory._reverify_dependencies()
    return factory


def require_eaaef_bootstrap_command_gateway(
    value: object,
) -> EAAEFBootstrapCommandGateway:
    if type(value) is not EAAEFBootstrapCommandGateway:
        raise EAAEFBootstrapRuntimeGatewayError("exact EAAEF bootstrap command gateway is required")
    value._validate_components()
    return value


def require_eaaef_bootstrap_execution_repository_proxy(
    value: object,
) -> EAAEFBootstrapExecutionRepositoryProxy:
    """Require the exact EAAEF type and marker; reject duck-typed substitutes."""

    if (
        type(value) is not EAAEFBootstrapExecutionRepositoryProxy
        or value.EAAEF_INTERFACE != EAAEF_BOOTSTRAP_EXECUTION_REPOSITORY_PROXY_INTERFACE
        or value.INTERFACE != EAAEF_BOOTSTRAP_EXECUTION_REPOSITORY_PROXY_INTERFACE
    ):
        raise EAAEFBootstrapRuntimeGatewayError(
            "exact EAAEF execution repository proxy @2 is required"
        )
    return value


__all__ = (
    "EAAEF_BOOTSTRAP_COMMAND_GATEWAY_INTERFACE",
    "EAAEF_BOOTSTRAP_COMMAND_TRANSPORT_INTERFACE",
    "EAAEF_BOOTSTRAP_EXECUTION_REPOSITORY_PROXY_INTERFACE",
    "EAAEF_BOOTSTRAP_EXECUTION_REPOSITORY_PROXY_SCHEMA",
    "EAAEF_BOOTSTRAP_RUNTIME_GATEWAY_PRODUCTION_BLOCKERS",
    "EAAEF_BOOTSTRAP_RUNTIME_GATEWAY_QUALIFICATION_STATUS",
    "EAAEF_TYPED_OWNER_COMMAND_CLIENT_INTERFACE",
    "EAAEF_TYPED_OWNER_COMMAND_TRANSPORT_INTERFACE",
    "EAAEF_CONTAINER_DISPATCHER_FACTORY_INTERFACE",
    "EAAEF_CONTAINER_DYNAMIC_SERVICE_REQUEST_SCHEMA",
    "EAAEF_CONTAINER_DYNAMIC_SERVICE_RESPONSE_SCHEMA",
    "EAAEF_EXACT_ENVELOPE_JOURNAL_INTERFACE",
    "EAAEF_EXACT_ENVELOPE_JOURNAL_SCHEMA",
    "EAAEF_SEALED_QUACK_CLIENT_DESCRIPTORS_INTERFACE",
    "EAAEF_SEALED_QUACK_SECRET_INTERFACE",
    "EAAEF_SEALED_QUACK_SECRET_SCHEMA",
    "EAAEFBootstrapCommandGateway",
    "EAAEFBootstrapCommandTransport",
    "EAAEFBootstrapCoordinatorProxy",
    "EAAEFBootstrapExcludedOperation",
    "EAAEFBootstrapExecutionRepositoryProxy",
    "EAAEFBootstrapGatewayCapability",
    "EAAEFBootstrapRuntimeGatewayAmbiguous",
    "EAAEFBootstrapRuntimeGatewayDiverged",
    "EAAEFBootstrapRuntimeGatewayError",
    "EAAEFBootstrapRuntimeGatewayNoGo",
    "EAAEFBootstrapTaskSourceProxy",
    "EAAEFTypedOwnerCommandClient",
    "EAAEFTypedOwnerCommandTransport",
    "EAAEFExactEnvelopeJournal",
    "EAAEFContainerDispatcherFactory",
    "EAAEFLaneRuntimeDependencyBundle",
    "EAAEFLaneRuntimeDependencyFactory",
    "EAAEFQualifiedQuackClients",
    "EAAEFSealedQuackClientDescriptors",
    "EAAEFSealedQuackSecretDescriptor",
    "bind_eaaef_bootstrap_command_transport",
    "bind_eaaef_qualified_bootstrap_command_transport",
    "bind_eaaef_sealed_quack_client_descriptors",
    "bind_eaaef_typed_owner_command_client",
    "bind_eaaef_typed_owner_command_transport",
    "create_eaaef_bootstrap_command_gateway",
    "create_eaaef_container_dispatcher_factory",
    "create_eaaef_lane_runtime_dependency_factory",
    "create_eaaef_qualified_quack_clients",
    "create_eaaef_sealed_quack_secret_descriptor",
    "eaaef_daemon_lane_binding_projection",
    "eaaef_dead_lane_recovery_arguments",
    "eaaef_exact_envelope_journal_relative_path",
    "eaaef_task_operation_authority_projection",
    "open_eaaef_exact_envelope_journal",
    "require_eaaef_bootstrap_command_gateway",
    "require_eaaef_bootstrap_execution_repository_proxy",
)
