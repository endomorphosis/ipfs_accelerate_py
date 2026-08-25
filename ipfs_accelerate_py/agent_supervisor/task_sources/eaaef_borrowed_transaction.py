"""Closed EAAEF daemon operations over one borrowed owner transaction.

The adapter in this module never opens a database, begins or ends a
transaction, exposes SQL, performs a provider/container effect, or accepts an
operation callback.  It receives the already-active ``StateTransaction@1``
owned by :mod:`quack_command_fabric` and applies one operation to the sealed
EAAEF operational-profile-v2 relations.

Source implementation is not external qualification.  The exported evidence
therefore remains ``implemented_unqualified_fail_closed`` until an independent
signed qualification is verified by the process-remote factory.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .control_plane_contracts import canonical_json_bytes, content_identity
from .control_plane_transactions import StateTransaction
from .eaaef_bootstrap_daemon_gateway import (
    EAAEF_BOOTSTRAP_DAEMON_MISSING_OPERATIONS,
    EAAEF_BOOTSTRAP_DAEMON_OPERATIONS,
)
from .eaaef_execution_contracts import (
    EAAEF_CONTAINER_VALIDATION_EVIDENCE_SCHEMA,
    EAAEF_IDEMPOTENT_RESERVATION_SCHEMA,
)
from .eaaef_operational_schema import (
    EAAEF_BOARD_SCHEDULER_LEASE_KIND,
    EAAEF_BOARD_SCHEDULER_LEASE_MODE,
    eaaef_operation_vocabulary_cid,
    verify_eaaef_operational_connection,
)

EAAEF_BORROWED_TRANSACTION_INTERFACE: Final = (
    "EAAEFBootstrapBorrowedTransactionAdapter@1"
)
EAAEF_BORROWED_TRANSACTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-bootstrap-borrowed-transaction-adapter@1"
)
EAAEF_BORROWED_TRANSACTION_QUALIFICATION_STATUS: Final = (
    "implemented_unqualified_fail_closed"
)
EAAEF_BORROWED_TRANSACTION_HANDLER_INTERFACE: Final = (
    "EAAEFBootstrapBorrowedTransactionOperationHandler@1"
)
EAAEF_BORROWED_TRANSACTION_HANDLER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-bootstrap-borrowed-transaction-handler@1"
)
EAAEF_TASK_COMPLETION_PREPARATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-task-completion-preparation@1"
)
EAAEF_TASK_OPERATION_AUTHORITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-task-operation-authority@2"
)
EAAEF_DAEMON_LANE_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-daemon-lane-binding@1"
)
EAAEF_PREPARED_RECOVERY_SNAPSHOT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-prepared-recovery-snapshot@1"
)
EAAEF_RUNNING_RECOVERY_SNAPSHOT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-running-recovery-snapshot@1"
)
EAAEF_DEAD_LANE_RECOVERY_AUTHORITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-dead-lane-recovery-authority@1"
)
EAAEF_CONTAINER_DISPATCH_OPERATION_KIND: Final = (
    "external_agent_container_dispatch"
)

MAX_IDENTIFIER_BYTES: Final = 512
MAX_JSON_BYTES: Final = 48 * 1024
MAX_LIST_ITEMS: Final = 1_000
MAX_LEASE_MS: Final = 24 * 60 * 60 * 1000
MAX_CALLER_CLOCK_SKEW_MS: Final = 5 * 60 * 1000
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,511}\Z")
_SHA256_CID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_READY = frozenset(
    {"proposed", "admitted", "pending", "ready", "todo", "queued", "retrying"}
)
_COMPLETED = frozenset({"completed", "complete", "done", "skipped"})
_SUCCESSFUL_TASK_STATUSES = frozenset({"completed", "complete", "done"})
_TASK_STATUSES = _READY | _COMPLETED | frozenset(
    {"claimed", "in_progress", "running", "blocked", "cancelled", "failed", "quarantined", "rejected"}
)
_PHASES = (
    "claimed",
    "context",
    "provider",
    "effect",
    "validation",
    "complete",
)
_TERMINAL_PHASES = frozenset({"complete", "failed", "blocked"})
_COMPLETION_RECEIPT_FIELDS = {
    "operation",
    "attempt_id",
    "claim_id",
    "lease_id",
    "owner_session_id",
    "fencing_token",
    "fence_epoch",
    "evidence_digest",
    "coordination_preparation",
    "validation",
}
_BARRIER_FIELDS = {
    "schema",
    "task_cid",
    "claim_id",
    "attempt_id",
    "attempt_number",
    "lease_id",
    "owner_session_id",
    "fencing_token",
    "fence_epoch",
    "control_expected_revision",
    "control_expected_status",
    "evidence_digest",
    "preparation_digest",
    "prepared_at_ms",
    "status",
    "control_completion",
    "reconciliation",
    "body",
    "revision",
}
_RECONCILIATION_FIELDS = {
    "operation",
    "task_cid",
    "claim_id",
    "attempt_id",
    "status",
    "observed_at_ms",
    "lease_state",
    "replayed",
}
_EXPIRED_ATTEMPT_RECONCILIATION_FIELDS = {
    "task_cid",
    "claim_id",
    "attempt_id",
    "status",
    "lease_state",
    "retry_required",
    "provider_evidence_reused",
    "effect_evidence_reused",
    "reason",
}
_FULL_CLAIM_FIELDS = {
    "claim_id",
    "task_cid",
    "owner_session_id",
    "fencing_token",
    "fence_epoch",
    "attempt_id",
    "attempt_number",
    "lease_id",
    "claimed_at_ms",
    "expires_at_ms",
    "released_at_ms",
    "state",
    "revision",
    "worktree_id",
    "idempotency_key",
    "body",
}
_CONTAINER_DISPATCH_CLAIM_FIELDS = {
    "schema",
    "interface",
    "packet_cid",
    "task_id",
    "task_cid",
    "attempt_id",
    "attempt_number",
    "plan_revision_cid",
    "repository_tree",
    "semantic_state_root",
    "worktree_id",
    "planned_container_id",
    "container_profile_cid",
    "image_digest",
    "network_authorization_cid",
    "lease_id",
    "fencing_token",
    "fence_epoch",
    "idempotency_key",
    "effect_scope_cid",
    "gateway_binding_cid",
    "worker_principal_did",
    "provider_principal_did",
    "provider",
    "model_route_cid",
    "claim_cid",
}
_CONTAINER_ACCEPTED_RESULT_FIELDS = {
    "schema",
    "interface",
    "status",
    "accepted",
    "task_result_accepted",
    "merge_admitted",
    "task_cid",
    "attempt_id",
    "packet_cid",
    "claim_cid",
    "reservation_id",
    "proposal_receipt_cid",
    "verification_receipt_cid",
    "patch_artifact_cid",
    "artifact_cids",
    "test_receipt_cids",
    "proof_receipt_cids",
    "worker_principal_did",
    "independent_verifier_principal_did",
    "receipt_id",
}
_CONTAINER_EFFECT_RESULT_FIELDS = {
    "schema",
    "interface",
    "status",
    "effect",
    "effect_key",
    "task_cid",
    "attempt_id",
    "packet_cid",
    "claim_cid",
    "accepted_result_receipt_id",
    "patch_artifact_cid",
    "task_result_accepted",
    "merge_admitted",
    "host_mutation_performed",
    "receipt_cid",
}
_TASK_OPERATION_AUTHORITY_FIELDS = {
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
_DAEMON_LANE_BINDING_FIELDS = {
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
_LANE_BOUND_BOARD_OPERATIONS = frozenset(
    {
        "execution.bind_daemon",
        "coordination.claim_ready",
        "execution.list_running_attempts",
    }
)
_DEAD_LANE_RECOVERY_AUTHORITY_FIELDS = {
    "schema",
    "purpose",
    "lane_bindings",
    "limit",
    "now_ms",
}
_ATTEMPT_INPUT_FIELDS = {
    "schema",
    "interface",
    "attempt_id",
    "claim_id",
    "task_cid",
    "task_alias",
    "attempt_number",
    "owner_session_id",
    "fencing_token",
    "fence_epoch",
    "lease_id",
    "committed_phase",
    "status",
    "started_at_ms",
    "finished_at_ms",
    "revision",
    "body",
}
_CLAIMED_PHASE_FIELDS = {
    "phase",
    "committed_at_ms",
    "fencing_token",
    "fence_epoch",
    "revision",
    "body",
}
_DATABASE_TASK_ATTEMPT_INTERFACE = "DatabaseTaskAttempt@1"
_DATABASE_TASK_ATTEMPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/database-task-attempt@1"
)
EAAEF_BOARD_SCOPED_OPERATIONS: Final = frozenset(
    {
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


class EAAEFBorrowedTransactionError(RuntimeError):
    """An operation was malformed, stale, or outside its authority."""


class EAAEFBorrowedTransactionConflict(EAAEFBorrowedTransactionError):
    """A CAS, lease, phase, or idempotency identity was stale."""


class EAAEFBorrowedTransactionNotReady(EAAEFBorrowedTransactionError):
    """No task or completion state is currently claimable."""


def _identifier(value: Any, noun: str) -> str:
    text = str(value or "")
    if not _SAFE_ID.fullmatch(text) or len(text.encode("utf-8")) > MAX_IDENTIFIER_BYTES:
        raise EAAEFBorrowedTransactionError(f"{noun} is not a bounded identifier")
    return text


def _positive(value: Any, noun: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EAAEFBorrowedTransactionError(f"{noun} must be a positive integer")
    if maximum is not None and value > maximum:
        raise EAAEFBorrowedTransactionError(f"{noun} exceeds its bound")
    return value


def _nonnegative(value: Any, noun: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EAAEFBorrowedTransactionError(f"{noun} must be a non-negative integer")
    return value


def _trusted_now(value: Any) -> int:
    """Validate caller observation time but return only the owner clock."""

    supplied = _positive(value, "now_ms")
    observed = int(time.time_ns() // 1_000_000)
    if abs(supplied - observed) > MAX_CALLER_CLOCK_SKEW_MS:
        raise EAAEFBorrowedTransactionError(
            "caller now_ms is outside the bounded owner-clock skew"
        )
    return observed


def _json(value: Any, noun: str) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise EAAEFBorrowedTransactionError(f"{noun} is not canonical JSON") from exc
    if len(encoded.encode("utf-8")) > MAX_JSON_BYTES:
        raise EAAEFBorrowedTransactionError(f"{noun} exceeds its byte bound")
    if isinstance(value, float) or _has_float(value):
        raise EAAEFBorrowedTransactionError(f"{noun} cannot contain floats")
    return encoded


def _has_float(value: Any) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, Mapping):
        return any(_has_float(key) or _has_float(item) for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_has_float(item) for item in value)
    return False


def _object(value: Any, noun: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EAAEFBorrowedTransactionError(f"{noun} must be an object")
    result = {str(key): item for key, item in value.items()}
    _json(result, noun)
    return result


def _decode(value: Any, noun: str) -> dict[str, Any]:
    try:
        decoded = json.loads(str(value or "{}"))
    except (TypeError, ValueError) as exc:
        raise EAAEFBorrowedTransactionError(f"{noun} is corrupt") from exc
    return _object(decoded, noun)


def _exact(arguments: Mapping[str, Any], fields: set[str], operation: str) -> dict[str, Any]:
    result = _object(arguments, f"{operation} arguments")
    if set(result) != fields:
        missing = sorted(fields - set(result))
        extra = sorted(set(result) - fields)
        raise EAAEFBorrowedTransactionError(
            f"{operation} arguments are not exact: missing={missing};extra={extra}"
        )
    return result


def _exact_one_of(
    arguments: Mapping[str, Any], variants: Sequence[set[str]], operation: str
) -> dict[str, Any]:
    result = _object(arguments, f"{operation} arguments")
    if not any(set(result) == fields for fields in variants):
        raise EAAEFBorrowedTransactionError(
            f"{operation} arguments do not match a closed shape"
        )
    return result


def _id(namespace: str, value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_json(dict(value), namespace).encode("utf-8")).hexdigest()
    return f"{namespace}:sha256:{digest}"


def _sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_json(dict(value), "digest").encode("utf-8")).hexdigest()


def eaaef_reservation_id(
    *, kind: str, attempt_id: str, idempotency_key: str
) -> str:
    """Return the exact provider/effect reservation identity used by @1."""

    normalized_kind = str(kind or "")
    if normalized_kind not in {"provider", "effect"}:
        raise EAAEFBorrowedTransactionError("reservation kind is unsupported")
    return _id(
        f"{normalized_kind}-reservation",
        {
            "attempt_id": _identifier(attempt_id, "attempt_id"),
            "idempotency_key": _identifier(idempotency_key, "idempotency_key"),
        },
    )


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat().replace(
        "+00:00", "Z"
    )


def _row_values(row: Any) -> tuple[Any, ...]:
    if row is None:
        return ()
    if isinstance(row, Mapping):
        return tuple(row.values())
    return tuple(row)


def _closed_validation_payload(value: Any) -> dict[str, Any]:
    payload = _object(value or {}, "validation payload")
    if not payload:
        return payload
    allowed = {"outcome", "evidence_digest", "argv", "body", "run_id", "result_id"}
    required = {"outcome", "evidence_digest", "argv"}
    if not required.issubset(payload) or not set(payload).issubset(allowed):
        raise EAAEFBorrowedTransactionError(
            "validation payload does not use a closed shape"
        )
    if str(payload["outcome"]).strip().lower() != "passed":
        raise EAAEFBorrowedTransactionError(
            "completion validation payload is not passed"
        )
    _identifier(payload["evidence_digest"], "validation evidence_digest")
    argv = payload["argv"]
    if not isinstance(argv, list) or not argv or len(argv) > MAX_LIST_ITEMS or not all(
        isinstance(item, str) and item for item in argv
    ):
        raise EAAEFBorrowedTransactionError("validation argv is not a closed list")
    if "body" in payload:
        _object(payload["body"], "validation body")
    return payload


_HANDLER_RUNTIME_AUTHORITY_FIELDS: Final = (
    "board_namespace",
    "shard_id",
    "owner_principal_did",
    "command_principal_did",
    "owner_session_id",
    "owner_generation",
    "fence_epoch",
    "gateway_binding_cid",
    "control_plane_schema_version",
    "state_schema_revision",
)


def _adapter_source_evidence(
    *, board_namespace: str, shard_id: str
) -> Mapping[str, Any]:
    board = _identifier(board_namespace, "board_namespace")
    shard = _identifier(shard_id, "shard_id")
    payload = {
        "schema": EAAEF_BORROWED_TRANSACTION_SCHEMA,
        "interface": EAAEF_BORROWED_TRANSACTION_INTERFACE,
        "qualification_status": EAAEF_BORROWED_TRANSACTION_QUALIFICATION_STATUS,
        "board_namespace": board,
        "shard_id": shard,
        "board_scope": _identifier(f"board:{board}:{shard}", "board_scope"),
        "operation_count": len(EAAEF_BOOTSTRAP_DAEMON_MISSING_OPERATIONS),
        "runtime_authority_fields": list(_HANDLER_RUNTIME_AUTHORITY_FIELDS),
        "owns_transaction_lifecycle": False,
        "opens_database": False,
        "exposes_database_path": False,
        "performs_external_effects": False,
        "accepts_operation_callback": False,
        "production_admitted": False,
    }
    return MappingProxyType(
        {**payload, "source_evidence_cid": content_identity(payload)}
    )


def eaaef_bootstrap_handler_source_evidence(
    *, board_namespace: str, shard_id: str
) -> Mapping[str, Any]:
    """Return runtime-authority-free evidence for the exact 31-op handler.

    Materialization and signed capability construction can bind this source
    identity without fabricating a live Quack owner, command principal, lease,
    fence, or gateway instance.  Runtime construction separately supplies and
    verifies every field named by ``runtime_authority_fields``.
    """

    adapter = dict(
        _adapter_source_evidence(
            board_namespace=board_namespace,
            shard_id=shard_id,
        )
    )
    payload = {
        "schema": EAAEF_BORROWED_TRANSACTION_HANDLER_SCHEMA,
        "interface": EAAEF_BORROWED_TRANSACTION_HANDLER_INTERFACE,
        "qualification_status": EAAEF_BORROWED_TRANSACTION_QUALIFICATION_STATUS,
        "operation_count": len(EAAEF_BOOTSTRAP_DAEMON_OPERATIONS),
        "operations": sorted(EAAEF_BOOTSTRAP_DAEMON_OPERATIONS),
        "canonical_read_operations": ["task.get", "task.ready"],
        "borrowed_transaction_operations": sorted(
            EAAEF_BOOTSTRAP_DAEMON_MISSING_OPERATIONS
        ),
        "board_scope": adapter["board_scope"],
        "runtime_authority_fields": list(_HANDLER_RUNTIME_AUTHORITY_FIELDS),
        "adapter_source_evidence_cid": adapter["source_evidence_cid"],
        "opens_database": False,
        "owns_transaction_lifecycle": False,
        "performs_external_effects": False,
        "accepts_operation_callback": False,
        "production_admitted": False,
    }
    return MappingProxyType(
        {**payload, "handler_source_evidence_cid": content_identity(payload)}
    )


class EAAEFBorrowedTransactionAdapter:
    """Apply the closed 29-operation bootstrap vocabulary in one owner txn."""

    INTERFACE: ClassVar[str] = EAAEF_BORROWED_TRANSACTION_INTERFACE
    SCHEMA: ClassVar[str] = EAAEF_BORROWED_TRANSACTION_SCHEMA
    QUALIFICATION_STATUS: ClassVar[str] = (
        EAAEF_BORROWED_TRANSACTION_QUALIFICATION_STATUS
    )

    __slots__ = (
        "_board_namespace",
        "_shard_id",
        "_board_scope",
        "_owner_principal_did",
        "_command_principal_did",
        "_owner_session_id",
        "_owner_generation",
        "_fence_epoch",
        "_gateway_binding_cid",
        "_control_plane_schema_version",
        "_state_schema_revision",
    )

    def __init__(
        self,
        *,
        board_namespace: str,
        shard_id: str,
        owner_principal_did: str,
        command_principal_did: str,
        owner_session_id: str,
        owner_generation: int,
        fence_epoch: int,
        gateway_binding_cid: str,
        control_plane_schema_version: str,
        state_schema_revision: str,
    ) -> None:
        self._board_namespace = _identifier(board_namespace, "board_namespace")
        self._shard_id = _identifier(shard_id, "shard_id")
        self._board_scope = _identifier(
            f"board:{self._board_namespace}:{self._shard_id}", "board_scope"
        )
        self._owner_principal_did = _identifier(
            owner_principal_did, "owner_principal_did"
        )
        self._command_principal_did = _identifier(
            command_principal_did, "command_principal_did"
        )
        if self._command_principal_did == self._owner_principal_did:
            raise EAAEFBorrowedTransactionError(
                "command principal must be distinct from the Quack owner"
            )
        self._owner_session_id = _identifier(
            owner_session_id, "owner_session_id"
        )
        self._owner_generation = _positive(owner_generation, "owner_generation")
        self._fence_epoch = _positive(fence_epoch, "fence_epoch")
        self._gateway_binding_cid = _identifier(
            gateway_binding_cid, "gateway_binding_cid"
        )
        self._control_plane_schema_version = _identifier(
            control_plane_schema_version, "control_plane_schema_version"
        )
        self._state_schema_revision = _identifier(
            state_schema_revision, "state_schema_revision"
        )

    @property
    def board_scope(self) -> str:
        return self._board_scope

    @staticmethod
    def _active(transaction: Any) -> Any:
        if type(transaction) is not StateTransaction or transaction.active is not True:
            raise EAAEFBorrowedTransactionError(
                "EAAEF operation requires an active StateTransaction@1"
            )
        owned = getattr(transaction, "_connection", None)
        if owned is None or not callable(getattr(owned, "execute", None)):
            raise EAAEFBorrowedTransactionError(
                "EAAEF owner transaction lost its private store"
            )
        return owned

    @staticmethod
    def _verify_profile(owned: Any) -> None:
        try:
            verify_eaaef_operational_connection(
                owned,
                operation_vocabulary_cid=eaaef_operation_vocabulary_cid(
                    EAAEF_BOOTSTRAP_DAEMON_OPERATIONS
                ),
            )
        except Exception as exc:
            raise EAAEFBorrowedTransactionError(
                "sealed EAAEF operational profile v2 is not exact"
            ) from exc

    def _scope(self, command: Any) -> str:
        parameters = getattr(command, "parameters", None)
        if not isinstance(parameters, Mapping):
            raise EAAEFBorrowedTransactionError("authorized command parameters are absent")
        return _identifier(parameters.get("task_cid"), "authorized scope_id")

    def _require_board_scope(self, command: Any) -> None:
        if self._scope(command) != self._board_scope:
            raise EAAEFBorrowedTransactionError(
                "board operation is not bound to the exact board/shard claim lease"
            )

    def _require_task_scope(self, command: Any, task_cid: str) -> None:
        if self._scope(command) != _identifier(task_cid, "task_cid"):
            raise EAAEFBorrowedTransactionError(
                "task operation is not bound to its exact task claim lease"
            )

    def _assert_board_lease(
        self,
        owned: Any,
        *,
        command: Any,
        lease: Mapping[str, Any],
        now_ms: int,
    ) -> Mapping[str, Any]:
        """Bind claim selection/recovery to the provisioned board lease row."""

        self._require_board_scope(command)
        row = owned.execute(
            "SELECT claim_cid, claimant_did, fencing_token, expires_at_ms, state, "
            "fence_epoch, lease_kind, scope_id, mode, owner_session_id, logical_epoch "
            "FROM leases WHERE task_cid = ?",
            [self._board_scope],
        ).fetchone()
        if row is None:
            raise EAAEFBorrowedTransactionConflict(
                "board/shard scheduler lease is absent"
            )
        observed = {
            "lease_id": str(row[0]),
            "principal_did": str(row[1]),
            "fencing_token": int(row[2]),
            "expires_at_ms": int(row[3]),
            "state": str(row[4]),
            "fence_epoch": int(row[5]),
            "lease_kind": str(row[6]),
            "scope_id": str(row[7]),
            "mode": str(row[8]),
            "owner_session_id": str(row[9]),
            "owner_generation": int(row[10]),
        }
        supplied_principal = _identifier(
            lease.get("principal_did"), "board principal_did"
        )
        supplied_epoch = _positive(lease.get("fence_epoch"), "board fence_epoch")
        expected = {
            "lease_id": _identifier(lease.get("lease_id"), "board lease_id"),
            "principal_did": self._command_principal_did,
            "fencing_token": _positive(
                lease.get("fencing_token"), "board fencing_token"
            ),
            "fence_epoch": self._fence_epoch,
            "lease_kind": EAAEF_BOARD_SCHEDULER_LEASE_KIND,
            "scope_id": self._board_scope,
            "mode": EAAEF_BOARD_SCHEDULER_LEASE_MODE,
            "owner_session_id": self._owner_session_id,
            "owner_generation": self._owner_generation,
            "expires_at_ms": _positive(
                lease.get("expires_at_ms"), "board expires_at_ms"
            ),
        }
        mismatched = [name for name, value in expected.items() if observed[name] != value]
        if supplied_principal != self._command_principal_did:
            mismatched.append("authorized_principal_did")
        if supplied_epoch != self._fence_epoch:
            mismatched.append("authorized_fence_epoch")
        command_fence = _positive(
            getattr(command, "fence_epoch", 0), "command fence_epoch"
        )
        if observed["fence_epoch"] != command_fence:
            mismatched.append("command_fence_epoch")
        if command_fence != self._fence_epoch:
            mismatched.append("capability_fence_epoch")
        if observed["state"] != "accepted" or observed["expires_at_ms"] <= now_ms:
            mismatched.append("live_state")
        if mismatched:
            raise EAAEFBorrowedTransactionConflict(
                "board/shard scheduler lease differs: " + ",".join(sorted(mismatched))
            )
        return MappingProxyType(observed)

    def _delegating_board_expiry(
        self,
        owned: Any,
        *,
        principal_did: str,
        fence_epoch: int,
        now_ms: int,
    ) -> int:
        """Return the live parent board-lease expiry for a task delegation."""

        row = owned.execute(
            "SELECT claimant_did, fence_epoch, expires_at_ms, state, lease_kind, "
            "scope_id, mode, owner_session_id, logical_epoch FROM leases "
            "WHERE task_cid=?",
            [self._board_scope],
        ).fetchone()
        expected = (
            self._command_principal_did,
            self._fence_epoch,
            "accepted",
            EAAEF_BOARD_SCHEDULER_LEASE_KIND,
            self._board_scope,
            EAAEF_BOARD_SCHEDULER_LEASE_MODE,
            self._owner_session_id,
            self._owner_generation,
        )
        if (
            principal_did != self._command_principal_did
            or fence_epoch != self._fence_epoch
        ):
            raise EAAEFBorrowedTransactionConflict(
                "task delegation authorization differs from command principal"
            )
        if row is None or (
            str(row[0]),
            int(row[1]),
            str(row[3]),
            str(row[4]),
            str(row[5]),
            str(row[6]),
            str(row[7]),
            int(row[8]),
        ) != expected:
            raise EAAEFBorrowedTransactionConflict(
                "task delegation has no exact live parent board lease"
            )
        expires_at_ms = int(row[2])
        if expires_at_ms <= now_ms:
            raise EAAEFBorrowedTransactionConflict(
                "task delegation parent board lease expired"
            )
        return expires_at_ms

    @staticmethod
    def _task_record(owned: Any, task_key: str) -> dict[str, Any] | None:
        key = _identifier(task_key, "task_cid")
        rows = owned.execute(
            "SELECT task_cid, task_alias, goal_cid, plan_cid, objective_id, "
            "ordinal, status, revision, priority, body_json FROM tasks "
            "WHERE task_cid = ? OR task_alias = ? ORDER BY task_cid LIMIT 2",
            [key, key],
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise EAAEFBorrowedTransactionConflict("task lookup is ambiguous")
        row = rows[0]
        task_cid = str(row[0])
        dependencies = [
            str(item[0])
            for item in owned.execute(
                "SELECT DISTINCT dependency_task_cid FROM task_dependencies "
                "WHERE task_cid = ? ORDER BY dependency_task_cid",
                [task_cid],
            ).fetchall()
        ]
        outputs = [
            {
                "ordinal": int(item[0]),
                "path": str(item[1]),
                "effect": _decode(item[2], "task output effect"),
            }
            for item in owned.execute(
                "SELECT ordinal, path, effect_json FROM task_outputs "
                "WHERE task_cid = ? ORDER BY ordinal",
                [task_cid],
            ).fetchall()
        ]
        acceptance = [
            {
                "ordinal": int(item[0]),
                "criterion": str(item[1]),
                "evidence_policy": _decode(item[2], "task acceptance policy"),
            }
            for item in owned.execute(
                "SELECT ordinal, criterion, evidence_policy_json "
                "FROM task_acceptance WHERE task_cid = ? ORDER BY ordinal",
                [task_cid],
            ).fetchall()
        ]
        validations: list[dict[str, Any]] = []
        for item in owned.execute(
            "SELECT ordinal, argv_json, policy_json FROM task_validations "
            "WHERE task_cid = ? ORDER BY ordinal",
            [task_cid],
        ).fetchall():
            try:
                argv = json.loads(str(item[1] or "[]"))
            except (TypeError, ValueError) as exc:
                raise EAAEFBorrowedTransactionError(
                    "task validation argv is corrupt"
                ) from exc
            if not isinstance(argv, list) or not all(
                isinstance(part, str) for part in argv
            ):
                raise EAAEFBorrowedTransactionError(
                    "task validation argv is not a string list"
                )
            validations.append(
                {
                    "ordinal": int(item[0]),
                    "argv": argv,
                    "policy": _decode(item[2], "task validation policy"),
                }
            )
        return {
            "task_cid": task_cid,
            "task_alias": str(row[1]),
            "goal_cid": str(row[2]),
            "plan_cid": str(row[3] or ""),
            "objective_id": str(row[4] or ""),
            "ordinal": int(row[5]),
            "status": str(row[6]),
            "revision": int(row[7]),
            "priority": str(row[8] or ""),
            "body": _decode(row[9], "task body"),
            "dependencies": dependencies,
            "outputs": outputs,
            "acceptance": acceptance,
            "validations": validations,
        }

    @staticmethod
    def _event(
        owned: Any,
        *,
        event_id: str,
        event_type: str,
        task_cid: str,
        attempt_id: str,
        session_id: str,
        recorded_at_ms: int,
        body: Mapping[str, Any],
    ) -> dict[str, Any]:
        eid = _identifier(event_id, "event_id")
        stream = _identifier(
            f"eaaef-daemon:{attempt_id or task_cid or session_id}", "event stream"
        )
        existing = owned.execute(
            "SELECT stream_id, event_type, task_cid, attempt_id, session_id, "
            "recorded_at, body_json "
            "FROM domain_events WHERE event_id = ?",
            [eid],
        ).fetchone()
        body_json = _json(dict(body), "event body")
        if existing is not None:
            observed = tuple(str(existing[index]) for index in range(7))
            expected = (
                stream,
                event_type,
                task_cid,
                attempt_id,
                session_id,
                _iso(recorded_at_ms),
                body_json,
            )
            if observed != expected:
                raise EAAEFBorrowedTransactionConflict("event identity replay diverged")
            return {"event_id": eid, "replayed": True}
        sequence = int(
            owned.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM domain_events WHERE stream_id = ?",
                [stream],
            ).fetchone()[0]
        )
        global_sequence = int(
            owned.execute(
                "SELECT COALESCE(MAX(global_sequence), 0) + 1 FROM domain_events"
            ).fetchone()[0]
        )
        owned.execute(
            "INSERT INTO domain_events(event_id, stream_id, sequence, global_sequence, "
            "event_type, task_cid, attempt_id, session_id, recorded_at, body_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [eid, stream, sequence, global_sequence, event_type, task_cid, attempt_id, session_id, _iso(recorded_at_ms), body_json],
        )
        return {"event_id": eid, "sequence": sequence, "global_sequence": global_sequence, "replayed": False}

    @staticmethod
    def _claim_record(owned: Any, claim_id: str) -> dict[str, Any] | None:
        row = owned.execute(
            "SELECT claim_id, task_cid, owner_session_id, fencing_token, "
            "fence_epoch, claimed_at_ms, expires_at_ms, released_at_ms, state, "
            "revision, attempt_id, attempt_number, lease_id, worktree_id, "
            "idempotency_key, body_json FROM task_claims WHERE claim_id = ?",
            [_identifier(claim_id, "claim_id")],
        ).fetchone()
        if row is None:
            return None
        return {
            "claim_id": str(row[0]),
            "task_cid": str(row[1]),
            "owner_session_id": str(row[2]),
            "fencing_token": int(row[3]),
            "fence_epoch": int(row[4]),
            "claimed_at_ms": int(row[5]),
            "expires_at_ms": int(row[6]),
            "released_at_ms": None if row[7] is None else int(row[7]),
            "state": str(row[8]),
            "revision": int(row[9]),
            "attempt_id": str(row[10]),
            "attempt_number": int(row[11]),
            "lease_id": str(row[12]),
            "worktree_id": str(row[13] or ""),
            "idempotency_key": str(row[14] or ""),
            "body": _decode(row[15], "task claim body"),
        }

    @staticmethod
    def _claim_identity(value: Any) -> dict[str, Any]:
        record = _object(
            value.to_dict() if callable(getattr(value, "to_dict", None)) else value,
            "task claim",
        )
        required = {
            "claim_id",
            "task_cid",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
            "attempt_id",
            "attempt_number",
            "lease_id",
        }
        if set(record) not in (required, _FULL_CLAIM_FIELDS):
            raise EAAEFBorrowedTransactionError(
                "task claim identity does not use a closed shape"
            )
        return {
            "claim_id": _identifier(record["claim_id"], "claim_id"),
            "task_cid": _identifier(record["task_cid"], "task_cid"),
            "owner_session_id": _identifier(record["owner_session_id"], "owner_session_id"),
            "fencing_token": _positive(record["fencing_token"], "fencing_token"),
            "fence_epoch": _positive(record["fence_epoch"], "fence_epoch"),
            "attempt_id": _identifier(record["attempt_id"], "attempt_id"),
            "attempt_number": _positive(record["attempt_number"], "attempt_number"),
            "lease_id": _identifier(record["lease_id"], "lease_id"),
        }

    def _protect(
        self,
        owned: Any,
        identity: Mapping[str, Any],
        *,
        now_ms: int,
        authorized_lease: Mapping[str, Any],
        allow_logically_completed: bool = False,
    ) -> dict[str, Any]:
        current = self._claim_record(owned, str(identity["claim_id"]))
        if current is None:
            raise EAAEFBorrowedTransactionConflict("task claim is absent")
        exact = (
            "claim_id",
            "task_cid",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
            "attempt_id",
            "attempt_number",
            "lease_id",
        )
        mismatched = [name for name in exact if current[name] != identity[name]]
        if mismatched:
            raise EAAEFBorrowedTransactionConflict(
                "task claim fence identity differs: " + ",".join(mismatched)
            )
        allowed_states = {"accepted"}
        if allow_logically_completed:
            allowed_states.update({"completed", "released"})
        if current["state"] not in allowed_states:
            raise EAAEFBorrowedTransactionConflict("task claim is not live")
        if current["state"] == "accepted" and current["expires_at_ms"] <= now_ms:
            raise EAAEFBorrowedTransactionConflict("task claim lease expired")
        lease_row = owned.execute(
            "SELECT claim_cid, claim_id, attempt_id, attempt_number, claimant_did, "
            "fencing_token, fence_epoch, expires_at_ms, state, owner_session_id, "
            "lease_kind, scope_id, mode FROM leases WHERE task_cid = ?",
            [identity["task_cid"]],
        ).fetchone()
        if lease_row is None:
            raise EAAEFBorrowedTransactionConflict("task lease is absent")
        observed = {
            "lease_id": str(lease_row[0]),
            "claim_id": str(lease_row[1]),
            "attempt_id": str(lease_row[2]),
            "attempt_number": int(lease_row[3]),
            "fencing_token": int(lease_row[5]),
            "fence_epoch": int(lease_row[6]),
            "expires_at_ms": int(lease_row[7]),
            "state": str(lease_row[8]),
            "owner_session_id": str(lease_row[9]),
            "lease_kind": str(lease_row[10]),
            "scope_id": str(lease_row[11]),
            "mode": str(lease_row[12]),
            "principal_did": str(lease_row[4]),
        }
        for name in ("lease_id", "claim_id", "attempt_id", "attempt_number", "fencing_token", "fence_epoch"):
            if observed[name] != identity[name]:
                raise EAAEFBorrowedTransactionConflict("task claim and lease diverged")
        if observed["state"] not in allowed_states:
            raise EAAEFBorrowedTransactionConflict("task lease is not live")
        structural = {
            "owner_session_id": identity["owner_session_id"],
            "lease_kind": "task",
            "scope_id": identity["task_cid"],
            "mode": "exclusive",
            "principal_did": _identifier(
                authorized_lease.get("principal_did"),
                "authorized principal_did",
            ),
            "lease_id": _identifier(
                authorized_lease.get("lease_id"), "authorized lease_id"
            ),
            "fencing_token": _positive(
                authorized_lease.get("fencing_token"),
                "authorized fencing_token",
            ),
            "fence_epoch": _positive(
                authorized_lease.get("fence_epoch"), "authorized fence_epoch"
            ),
        }
        poisoned = [
            name for name, expected in structural.items() if observed[name] != expected
        ]
        if poisoned:
            raise EAAEFBorrowedTransactionConflict(
                "task lease physical binding differs: " + ",".join(poisoned)
            )
        return current

    def _protect_current_task(
        self,
        owned: Any,
        *,
        task_cid: str,
        authorized_lease: Mapping[str, Any],
        allow_logically_completed: bool = False,
    ) -> dict[str, Any]:
        """Resolve and protect the exact current claim for a task operation."""

        row = owned.execute(
            "SELECT claim_id FROM leases WHERE task_cid=? AND lease_kind='task' "
            "AND scope_id=? AND mode='exclusive'",
            [task_cid, task_cid],
        ).fetchone()
        if row is None or not str(row[0]):
            raise EAAEFBorrowedTransactionConflict(
                "task operation has no exact task claim lease"
            )
        claim = self._claim_record(owned, str(row[0]))
        if claim is None:
            raise EAAEFBorrowedTransactionConflict(
                "task operation claim history is absent"
            )
        identity = self._claim_identity(claim)
        now_ms = int(time.time_ns() // 1_000_000)
        return self._protect(
            owned,
            identity,
            now_ms=now_ms,
            authorized_lease=authorized_lease,
            allow_logically_completed=allow_logically_completed,
        )

    def _protect_current_attempt(
        self,
        owned: Any,
        *,
        attempt: Mapping[str, Any],
        authorized_lease: Mapping[str, Any],
        allow_logically_completed: bool = False,
    ) -> dict[str, Any]:
        """Require an attempt to be the exact current task-lease attempt."""

        claim = self._protect_current_task(
            owned,
            task_cid=str(attempt["task_cid"]),
            authorized_lease=authorized_lease,
            allow_logically_completed=allow_logically_completed,
        )
        fields = (
            "claim_id",
            "task_cid",
            "attempt_id",
            "attempt_number",
            "lease_id",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
        )
        mismatched = [name for name in fields if attempt[name] != claim[name]]
        if mismatched:
            raise EAAEFBorrowedTransactionConflict(
                "execution attempt is not the current task claim: "
                + ",".join(mismatched)
            )
        return claim

    def _unwrap_task_operation_authority(
        self,
        owned: Any,
        *,
        operation: str,
        arguments: Mapping[str, Any],
        command: Any,
        authorized_lease: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Cross-join and remove the signed EAAEF task authority projection."""

        outer = _object(arguments, f"{operation} authorized arguments")
        authority = _exact(
            _object(outer.pop("task_authority_binding", None), "task authority binding"),
            _TASK_OPERATION_AUTHORITY_FIELDS,
            "task authority binding",
        )
        if authority["schema"] != EAAEF_TASK_OPERATION_AUTHORITY_SCHEMA:
            raise EAAEFBorrowedTransactionError(
                "task authority binding schema is unsupported"
            )
        identity = {
            name: authority[name]
            for name in (
                "claim_id",
                "task_cid",
                "owner_session_id",
                "fencing_token",
                "fence_epoch",
                "attempt_id",
                "attempt_number",
                "lease_id",
            )
        }
        normalized = self._claim_identity(identity)
        lane_binding = self._daemon_lane_binding(
            authority["daemon_lane_binding"]
        )
        if lane_binding["lane_session_id"] != normalized["owner_session_id"]:
            raise EAAEFBorrowedTransactionConflict(
                "task authority lane differs from the task claim owner"
            )
        self._require_bound_lane(owned, lane_binding)
        self._require_task_scope(command, normalized["task_cid"])
        if (
            _identifier(
                authorized_lease.get("principal_did"),
                "authorized task principal_did",
            )
            != self._command_principal_did
            or _positive(
                authorized_lease.get("fence_epoch"),
                "authorized task fence_epoch",
            )
            != self._fence_epoch
        ):
            raise EAAEFBorrowedTransactionConflict(
                "task authorization differs from command principal/fence"
            )
        if any(authority[name] != normalized[name] for name in normalized):
            raise EAAEFBorrowedTransactionConflict(
                "task authority binding normalization differs"
            )
        self._protect(
            owned,
            normalized,
            now_ms=int(time.time_ns() // 1_000_000),
            authorized_lease=authorized_lease,
            allow_logically_completed=True,
        )
        return outer

    def _daemon_lane_binding(self, value: Any) -> dict[str, Any]:
        binding = _exact(
            _object(value, "daemon lane binding"),
            _DAEMON_LANE_BINDING_FIELDS,
            "daemon lane binding",
        )
        normalized = {
            "schema": str(binding["schema"]),
            "gateway_binding_cid": _identifier(
                binding["gateway_binding_cid"], "lane gateway_binding_cid"
            ),
            "owner_principal_did": _identifier(
                binding["owner_principal_did"], "lane owner_principal_did"
            ),
            "owner_session_id": _identifier(
                binding["owner_session_id"], "lane owner_session_id"
            ),
            "owner_generation": _positive(
                binding["owner_generation"], "lane owner_generation"
            ),
            "lane_session_id": _identifier(
                binding["lane_session_id"], "lane_session_id"
            ),
            "lane_generation": _positive(
                binding["lane_generation"], "lane_generation"
            ),
            "process_instance_id": _identifier(
                binding["process_instance_id"], "lane process_instance_id"
            ),
            "fence_epoch": _positive(binding["fence_epoch"], "lane fence_epoch"),
        }
        expected = {
            "schema": EAAEF_DAEMON_LANE_BINDING_SCHEMA,
            "gateway_binding_cid": self._gateway_binding_cid,
            "owner_principal_did": self._owner_principal_did,
            "owner_session_id": self._owner_session_id,
            "owner_generation": self._owner_generation,
            "fence_epoch": self._fence_epoch,
        }
        if any(normalized[name] != item for name, item in expected.items()) or (
            normalized["lane_session_id"] == self._owner_session_id
        ):
            raise EAAEFBorrowedTransactionConflict(
                "daemon lane binding differs from the verified Quack owner"
            )
        return normalized

    @staticmethod
    def _require_bound_lane(owned: Any, binding: Mapping[str, Any]) -> None:
        row = owned.execute(
            "SELECT daemon_id, fence_epoch, status, metadata_json FROM "
            "daemon_sessions WHERE session_id=?",
            [binding["lane_session_id"]],
        ).fetchone()
        if row is None:
            raise EAAEFBorrowedTransactionConflict(
                "daemon lane has not been bound by the owner"
            )
        metadata = _decode(row[3], "bound daemon lane metadata")
        if (
            str(row[0]) != binding["process_instance_id"]
            or int(row[1]) != binding["fence_epoch"]
            or str(row[2]) != "attached"
            or metadata.get("lane_binding") != dict(binding)
        ):
            raise EAAEFBorrowedTransactionConflict(
                "daemon lane binding differs from its persisted session"
            )

    def authorize_canonical_read(
        self,
        *,
        operation: str,
        arguments: Mapping[str, Any],
        transaction: Any,
        command: Any,
        lease: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Authorize and strip the two canonical reads used by EAAEF @2.

        ``task.ready`` is a board-scheduler query.  Ordinary ``task.get`` is
        tied to the exact live task claim and its bound daemon lane.  The only
        board-scoped ``task.get`` variant is a closed recovery observation
        joined to an existing completion barrier or an expired historical
        claim; it never grants normal task mutation authority.
        """

        owned = self._active(transaction)
        self._verify_profile(owned)
        name = str(operation or "")
        raw = _object(arguments, f"{name} arguments")
        if name == "task.ready":
            args = _exact(raw, {"limit"}, name)
            self._assert_board_lease(
                owned,
                command=command,
                lease=lease,
                now_ms=int(time.time_ns() // 1_000_000),
            )
            return MappingProxyType(args)
        if name != "task.get":
            raise EAAEFBorrowedTransactionError(
                "canonical read authorization is outside task.ready/task.get"
            )

        authority = _object(raw.get("task_authority_binding"), "task authority binding")
        authority_task_cid = _identifier(
            authority.get("task_cid"), "task authority task_cid"
        )
        stripped = self._unwrap_task_operation_authority(
            owned,
            operation=name,
            arguments=raw,
            command=command,
            authorized_lease=lease,
        )
        args = _exact(stripped, {"task_cid"}, name)
        if _identifier(args["task_cid"], "task_cid") != authority_task_cid:
            raise EAAEFBorrowedTransactionConflict(
                "task.get target differs from its exact task authority"
            )
        return MappingProxyType(args)

    @staticmethod
    def _historical_identity_matches(
        barrier: Mapping[str, Any],
        claim: Mapping[str, Any],
        attempt: Mapping[str, Any],
    ) -> bool:
        claim_fields = (
            "claim_id",
            "task_cid",
            "attempt_id",
            "attempt_number",
            "lease_id",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
        )
        attempt_fields = (
            "claim_id",
            "task_cid",
            "attempt_id",
            "attempt_number",
            "lease_id",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
        )
        return all(barrier[name] == claim[name] for name in claim_fields) and all(
            barrier[name] == attempt[name] for name in attempt_fields
        )

    @staticmethod
    def _attempt(owned: Any, attempt_id: str) -> dict[str, Any] | None:
        row = owned.execute(
            "SELECT attempt_id, claim_id, task_cid, task_alias, attempt_number, "
            "owner_session_id, fencing_token, fence_epoch, lease_id, committed_phase, "
            "status, started_at_ms, finished_at_ms, revision, body_json "
            "FROM task_attempts WHERE attempt_id = ?",
            [_identifier(attempt_id, "attempt_id")],
        ).fetchone()
        if row is None:
            return None
        return {
            "attempt_id": str(row[0]),
            "claim_id": str(row[1]),
            "task_cid": str(row[2]),
            "task_alias": str(row[3] or ""),
            "attempt_number": int(row[4]),
            "owner_session_id": str(row[5]),
            "fencing_token": int(row[6]),
            "fence_epoch": int(row[7]),
            "lease_id": str(row[8]),
            "committed_phase": str(row[9]),
            "status": str(row[10]),
            "started_at_ms": int(row[11]),
            "finished_at_ms": None if row[12] is None else int(row[12]),
            "revision": int(row[13]),
            "body": _decode(row[14], "attempt body"),
        }

    @staticmethod
    def _attempt_input(
        value: Any,
        claimed_phase: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Validate the exact daemon attempt/claimed-phase wire projection."""

        attempt = _object(value, "execution attempt")
        claimed = _object(claimed_phase, "claimed phase")
        if set(attempt) != _ATTEMPT_INPUT_FIELDS:
            raise EAAEFBorrowedTransactionError(
                "execution attempt does not use the closed DatabaseTaskAttempt@1 shape"
            )
        if set(claimed) != _CLAIMED_PHASE_FIELDS:
            raise EAAEFBorrowedTransactionError(
                "claimed phase does not use the closed phase shape"
            )
        if (
            attempt["schema"] != _DATABASE_TASK_ATTEMPT_SCHEMA
            or attempt["interface"] != _DATABASE_TASK_ATTEMPT_INTERFACE
            or str(attempt["committed_phase"]) != "claimed"
            or str(attempt["status"]) != "running"
            or attempt["finished_at_ms"] is not None
            or _positive(attempt["revision"], "attempt revision") != 1
            or str(claimed["phase"]) != "claimed"
            or _positive(claimed["revision"], "claimed phase revision") != 1
        ):
            raise EAAEFBorrowedTransactionError(
                "execution attempt is not the exact initial running/claimed state"
            )
        normalized = {
            "attempt_id": _identifier(attempt["attempt_id"], "attempt_id"),
            "claim_id": _identifier(attempt["claim_id"], "claim_id"),
            "task_cid": _identifier(attempt["task_cid"], "task_cid"),
            "task_alias": _identifier(attempt["task_alias"], "task_alias"),
            "attempt_number": _positive(
                attempt["attempt_number"], "attempt_number"
            ),
            "owner_session_id": _identifier(
                attempt["owner_session_id"], "owner_session_id"
            ),
            "fencing_token": _positive(
                attempt["fencing_token"], "fencing_token"
            ),
            "fence_epoch": _positive(attempt["fence_epoch"], "fence_epoch"),
            "lease_id": _identifier(attempt["lease_id"], "lease_id"),
            "committed_phase": "claimed",
            "status": "running",
            "started_at_ms": _positive(
                attempt["started_at_ms"], "attempt started_at_ms"
            ),
            "finished_at_ms": None,
            "revision": 1,
            "body": _object(attempt["body"], "attempt body"),
        }
        normalized_claimed = {
            "phase": "claimed",
            "committed_at_ms": _positive(
                claimed["committed_at_ms"], "claimed phase committed_at_ms"
            ),
            "fencing_token": _positive(
                claimed["fencing_token"], "claimed phase fencing_token"
            ),
            "fence_epoch": _positive(
                claimed["fence_epoch"], "claimed phase fence_epoch"
            ),
            "revision": 1,
            "body": _object(claimed["body"], "claimed phase body"),
        }
        if (
            normalized_claimed["fencing_token"] != normalized["fencing_token"]
            or normalized_claimed["fence_epoch"] != normalized["fence_epoch"]
            or normalized_claimed["committed_at_ms"] < normalized["started_at_ms"]
        ):
            raise EAAEFBorrowedTransactionConflict(
                "claimed phase differs from its exact attempt fence/time"
            )
        return normalized, normalized_claimed

    @staticmethod
    def _barrier(owned: Any, task_cid: str) -> dict[str, Any] | None:
        row = owned.execute(
            "SELECT task_cid, claim_id, attempt_id, attempt_number, lease_id, "
            "owner_session_id, fencing_token, fence_epoch, control_expected_revision, "
            "control_expected_status, evidence_digest, preparation_digest, prepared_at_ms, "
            "status, control_receipt_json, reconciliation_json, body_json, revision "
            "FROM eaaef_completion_barriers WHERE task_cid = ?",
            [_identifier(task_cid, "task_cid")],
        ).fetchone()
        if row is None:
            return None
        return {
            "schema": EAAEF_TASK_COMPLETION_PREPARATION_SCHEMA,
            "task_cid": str(row[0]),
            "claim_id": str(row[1]),
            "attempt_id": str(row[2]),
            "attempt_number": int(row[3]),
            "lease_id": str(row[4]),
            "owner_session_id": str(row[5]),
            "fencing_token": int(row[6]),
            "fence_epoch": int(row[7]),
            "control_expected_revision": int(row[8]),
            "control_expected_status": str(row[9]),
            "evidence_digest": str(row[10]),
            "preparation_digest": str(row[11]),
            "prepared_at_ms": int(row[12]),
            "status": str(row[13]),
            "control_completion": _decode(row[14], "control completion"),
            "reconciliation": _decode(row[15], "completion reconciliation"),
            "body": _decode(row[16], "completion body"),
            "revision": int(row[17]),
        }

    def _prepared_recovery_snapshot(
        self, owned: Any, *, task_cid: str, observed_at_ms: int
    ) -> dict[str, Any]:
        barrier = self._barrier(owned, task_cid)
        if barrier is None:
            raise EAAEFBorrowedTransactionConflict(
                "prepared recovery snapshot has no barrier"
            )
        claim = self._claim_record(owned, barrier["claim_id"])
        attempt = self._attempt(owned, barrier["attempt_id"])
        task = self._task_record(owned, barrier["task_cid"])
        if (
            claim is None
            or attempt is None
            or task is None
            or not self._historical_identity_matches(barrier, claim, attempt)
        ):
            raise EAAEFBorrowedTransactionConflict(
                "prepared recovery snapshot provenance differs"
            )
        payload = {
            "schema": EAAEF_PREPARED_RECOVERY_SNAPSHOT_SCHEMA,
            "kind": "prepared_completion",
            "preparation": barrier,
            "claim": claim,
            "attempt": attempt,
            "task": task,
            "control_completion": barrier["control_completion"],
            "observed_at_ms": observed_at_ms,
        }
        _json(payload, "prepared recovery snapshot")
        return {**payload, "snapshot_cid": content_identity(payload)}

    def _running_recovery_snapshot(
        self, owned: Any, *, attempt: Mapping[str, Any], observed_at_ms: int
    ) -> dict[str, Any]:
        claim = self._claim_record(owned, str(attempt["claim_id"]))
        if claim is None:
            raise EAAEFBorrowedTransactionConflict(
                "running recovery snapshot has no claim history"
            )
        identity = self._claim_identity(claim)
        if any(
            attempt[field] != identity[field]
            for field in (
                "claim_id",
                "task_cid",
                "attempt_id",
                "attempt_number",
                "lease_id",
                "owner_session_id",
                "fencing_token",
                "fence_epoch",
            )
        ):
            raise EAAEFBorrowedTransactionConflict(
                "running recovery snapshot attempt/claim differs"
            )
        barrier = self._barrier(owned, str(attempt["task_cid"]))
        if barrier is not None and not self._historical_identity_matches(
            barrier, claim, attempt
        ):
            raise EAAEFBorrowedTransactionConflict(
                "running recovery snapshot barrier differs"
            )
        payload = {
            "schema": EAAEF_RUNNING_RECOVERY_SNAPSHOT_SCHEMA,
            "kind": "running_attempt",
            "claim": claim,
            "preparation": barrier,
            "observed_at_ms": observed_at_ms,
        }
        _json(payload, "running recovery snapshot")
        return {**payload, "snapshot_cid": content_identity(payload)}

    @staticmethod
    def _current_evidence(
        owned: Any, task_cid: str
    ) -> tuple[set[str], set[str]]:
        rows = owned.execute(
            "SELECT evidence_kind, digest FROM evidence_nodes WHERE task_cid = ?",
            [task_cid],
        ).fetchall()
        return ({str(row[1]) for row in rows}, {str(row[0]) for row in rows})

    def _record_validation(
        self, owned: Any, arguments: Mapping[str, Any], *, attempt_default: str = ""
    ) -> dict[str, Any]:
        fields = {"task_cid", "outcome", "evidence_digest", "argv", "body"}
        with_attempt = fields | {"attempt_id"}
        args = _exact_one_of(arguments, (fields, with_attempt), "validation.record")
        task_cid = _identifier(args["task_cid"], "task_cid")
        if self._task_record(owned, task_cid) is None:
            raise EAAEFBorrowedTransactionConflict("validation task is absent")
        outcome = str(args["outcome"] or "").strip().lower()
        if outcome not in {"passed", "failed", "error", "skipped"}:
            raise EAAEFBorrowedTransactionError("validation outcome is unsupported")
        digest = _identifier(args["evidence_digest"], "evidence_digest")
        argv = args["argv"]
        if not isinstance(argv, list) or len(argv) > MAX_LIST_ITEMS or not all(
            isinstance(item, str) and item for item in argv
        ):
            raise EAAEFBorrowedTransactionError("validation argv is not bounded")
        body = _object(args["body"], "validation body")
        attempt_id = str(args.get("attempt_id") or attempt_default)
        identity = {
            "task_cid": task_cid,
            "attempt_id": attempt_id,
            "outcome": outcome,
            "evidence_digest": digest,
            "argv": argv,
        }
        run_id = _id("validation-run", identity)
        result_id = _id("validation-result", identity)
        now = int(time.time_ns() // 1_000_000)
        existing = owned.execute(
            "SELECT status, body_json FROM validation_runs WHERE run_id = ?", [run_id]
        ).fetchone()
        run_body = _json({"argv": argv, **body}, "validation run body")
        if existing is None:
            owned.execute(
                "INSERT INTO validation_runs(run_id, task_cid, attempt_id, started_at, "
                "finished_at, status, command_digest, body_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [run_id, task_cid, attempt_id, _iso(now), _iso(now), outcome, content_identity({"argv": argv}), run_body],
            )
            owned.execute(
                "INSERT INTO validation_results(result_id, run_id, task_cid, ordinal, "
                "outcome, evidence_digest, body_json) VALUES (?, ?, ?, 0, ?, ?, ?)",
                [result_id, run_id, task_cid, outcome, digest, _json(body, "validation result body")],
            )
            if outcome == "passed":
                evidence_id = _id(
                    "validation-evidence",
                    {"task_cid": task_cid, "digest": digest, "run_id": run_id},
                )
                owned.execute(
                    "INSERT INTO evidence_nodes(evidence_id, parent_evidence_id, task_cid, "
                    "evidence_kind, digest, created_at, body_json) VALUES (?, '', ?, 'validation', ?, ?, ?)",
                    [evidence_id, task_cid, digest, _iso(now), _json({"run_id": run_id, "result_id": result_id, "argv": argv, "outcome": outcome}, "evidence body")],
                )
            replayed = False
        else:
            if str(existing[0]) != outcome or str(existing[1]) != run_body:
                raise EAAEFBorrowedTransactionConflict(
                    "validation identity replay diverged"
                )
            replayed = True
        return {
            "run_id": run_id,
            "result_id": result_id,
            "task_cid": task_cid,
            "attempt_id": attempt_id,
            "outcome": outcome,
            "evidence_digest": digest,
            "replayed": replayed,
        }

    def _replay_committed_validation(
        self,
        owned: Any,
        arguments: Mapping[str, Any],
        *,
        attempt_default: str,
    ) -> dict[str, Any]:
        """Expose the two public validation operations as exact replays only."""

        fields = {"task_cid", "outcome", "evidence_digest", "argv", "body"}
        args = _exact_one_of(
            arguments,
            (fields, fields | {"attempt_id"}),
            "validation replay",
        )
        task_cid = _identifier(args["task_cid"], "task_cid")
        attempt_id = _identifier(
            args.get("attempt_id") or attempt_default, "attempt_id"
        )
        phase_row = owned.execute(
            "SELECT status, body_json FROM attempt_phases "
            "WHERE attempt_id=? AND phase_name='validation'",
            [attempt_id],
        ).fetchone()
        if phase_row is None or str(phase_row[0]) != "committed":
            raise EAAEFBorrowedTransactionNotReady(
                "validation record is created only by its atomic validation phase"
            )
        phase = _closed_validation_payload(
            _decode(phase_row[1], "committed validation phase")
        )
        expected = {
            "task_cid": task_cid,
            "attempt_id": attempt_id,
            "outcome": phase["outcome"],
            "evidence_digest": phase["evidence_digest"],
            "argv": phase["argv"],
            "body": phase["body"],
        }
        supplied = {
            "task_cid": task_cid,
            "attempt_id": attempt_id,
            "outcome": str(args["outcome"] or "").strip().lower(),
            "evidence_digest": _identifier(
                args["evidence_digest"], "evidence_digest"
            ),
            "argv": args["argv"],
            "body": _object(args["body"], "validation body"),
        }
        if supplied != expected:
            raise EAAEFBorrowedTransactionConflict(
                "validation replay differs from its committed phase"
            )
        run_id = _identifier(phase.get("run_id"), "validation run_id")
        result_id = _identifier(phase.get("result_id"), "validation result_id")
        durable = owned.execute(
            "SELECT runs.task_cid, runs.attempt_id, runs.status, "
            "results.task_cid, results.outcome, results.evidence_digest, "
            "results.body_json FROM validation_runs AS runs "
            "JOIN validation_results AS results ON results.run_id=runs.run_id "
            "WHERE runs.run_id=? AND results.result_id=?",
            [run_id, result_id],
        ).fetchall()
        if (
            len(durable) != 1
            or str(durable[0][0]) != task_cid
            or str(durable[0][1]) != attempt_id
            or str(durable[0][2]) != phase["outcome"]
            or str(durable[0][3]) != task_cid
            or str(durable[0][4]) != phase["outcome"]
            or str(durable[0][5]) != phase["evidence_digest"]
            or _decode(durable[0][6], "validation result body") != phase["body"]
        ):
            raise EAAEFBorrowedTransactionConflict(
                "validation replay lacks its exact durable result"
            )
        return {
            "run_id": run_id,
            "result_id": result_id,
            "task_cid": task_cid,
            "attempt_id": attempt_id,
            "outcome": phase["outcome"],
            "evidence_digest": phase["evidence_digest"],
            "replayed": True,
        }

    def _task_cas(
        self,
        owned: Any,
        arguments: Mapping[str, Any],
        *,
        authorized_lease: Mapping[str, Any],
    ) -> dict[str, Any]:
        args = _exact(
            arguments,
            {"task_cid", "expected_revision", "status", "receipt", "evidence_digests"},
            "task.cas_status",
        )
        task_cid = _identifier(args["task_cid"], "task_cid")
        expected = _positive(args["expected_revision"], "expected_revision")
        status = str(args["status"] or "").strip().lower()
        if status not in _TASK_STATUSES:
            raise EAAEFBorrowedTransactionError("task status is unsupported")
        receipt = _object(args["receipt"] or {}, "task status receipt")
        evidence_digests = args["evidence_digests"] or []
        if not isinstance(evidence_digests, list) or len(evidence_digests) > MAX_LIST_ITEMS:
            raise EAAEFBorrowedTransactionError("evidence_digests is not bounded")
        task = self._task_record(owned, task_cid)
        if task is None:
            raise EAAEFBorrowedTransactionConflict("task is absent")
        live_claim = self._protect_current_task(
            owned,
            task_cid=task_cid,
            authorized_lease=authorized_lease,
        )
        if task["revision"] != expected:
            raise EAAEFBorrowedTransactionConflict("task revision CAS is stale")
        prior = str(task["status"])
        same_status = prior == status
        if same_status and status not in _SUCCESSFUL_TASK_STATUSES:
            return {
                "schema": "ipfs_accelerate_py/agent-supervisor/database-task-cas@1",
                "task": task,
                "previous_status": prior,
                "revision": expected,
                "event_cursor": 0,
                "changed": False,
                "receipt_cid": "",
            }
        if status not in _SUCCESSFUL_TASK_STATUSES:
            claim_receipt_fields = {
                "operation",
                "claim_id",
                "attempt_id",
                "owner_session_id",
            }
            if (
                status != "in_progress"
                or prior not in _READY
                or set(receipt) != claim_receipt_fields
                or receipt.get("operation") != "database_claim"
                or any(
                    receipt.get(name) != live_claim[name]
                    for name in ("claim_id", "attempt_id", "owner_session_id")
                )
                or evidence_digests != []
            ):
                raise EAAEFBorrowedTransactionConflict(
                    "task status transition is outside the closed claim/completion table"
                )
        else:
            if set(receipt) != _COMPLETION_RECEIPT_FIELDS:
                raise EAAEFBorrowedTransactionError(
                    "completion receipt does not use the exact closed shape"
                )
            if receipt.get("operation") != "database_complete":
                raise EAAEFBorrowedTransactionError(
                    "completion receipt operation is unsupported"
                )
            validation = _closed_validation_payload(receipt.get("validation"))
            validation_phase_rows = owned.execute(
                "SELECT status, body_json FROM attempt_phases "
                "WHERE attempt_id=? AND phase_name='validation'",
                [live_claim["attempt_id"]],
            ).fetchall()
            if (
                len(validation_phase_rows) != 1
                or str(validation_phase_rows[0][0]) != "committed"
                or validation
                != _closed_validation_payload(
                    _decode(
                        validation_phase_rows[0][1],
                        "canonical validation phase",
                    )
                )
            ):
                raise EAAEFBorrowedTransactionConflict(
                    "completion validation differs from its committed phase"
                )
            barrier = self._barrier(owned, task_cid)
            allowed_barrier_states = (
                {"prepared", "succeeded"} if same_status else {"prepared"}
            )
            if barrier is None or barrier["status"] not in allowed_barrier_states:
                raise EAAEFBorrowedTransactionNotReady(
                    "terminal task CAS requires a live prepared completion barrier"
                )
            binding = _object(
                receipt.get("coordination_preparation"),
                "coordination preparation",
            )
            if set(binding) != _BARRIER_FIELDS:
                raise EAAEFBorrowedTransactionError(
                    "coordination preparation does not use the exact closed shape"
                )
            immutable = (
                "task_cid",
                "claim_id",
                "attempt_id",
                "attempt_number",
                "lease_id",
                "owner_session_id",
                "fencing_token",
                "fence_epoch",
                "control_expected_revision",
                "control_expected_status",
                "evidence_digest",
                "preparation_digest",
                "prepared_at_ms",
            )
            if any(binding.get(name) != barrier[name] for name in immutable):
                raise EAAEFBorrowedTransactionConflict(
                    "completion receipt preparation differs from the stored barrier"
                )
            receipt_identity = {
                "claim_id": live_claim["claim_id"],
                "attempt_id": live_claim["attempt_id"],
                "lease_id": live_claim["lease_id"],
                "owner_session_id": live_claim["owner_session_id"],
                "fencing_token": live_claim["fencing_token"],
                "fence_epoch": live_claim["fence_epoch"],
                "evidence_digest": barrier["evidence_digest"],
            }
            if any(receipt.get(name) != expected for name, expected in receipt_identity.items()):
                raise EAAEFBorrowedTransactionConflict(
                    "completion receipt differs from the live claim/barrier fence"
                )
            if validation and validation["evidence_digest"] != barrier["evidence_digest"]:
                raise EAAEFBorrowedTransactionConflict(
                    "completion validation differs from barrier evidence"
                )
            current_digests, current_kinds = self._current_evidence(owned, task_cid)
            supplied = {_identifier(value, "evidence_digest") for value in evidence_digests}
            if not supplied or not supplied.issubset(current_digests):
                raise EAAEFBorrowedTransactionNotReady(
                    "completion lacks current stored evidence"
                )
            if barrier["evidence_digest"] not in supplied:
                raise EAAEFBorrowedTransactionNotReady(
                    "completion evidence does not include the prepared digest"
                )
            acceptance = owned.execute(
                "SELECT criterion, evidence_policy_json FROM task_acceptance WHERE task_cid = ? ORDER BY ordinal",
                [task_cid],
            ).fetchall()
            for acceptance_row in acceptance:
                criterion, policy_json = acceptance_row[0], acceptance_row[1]
                policy = _decode(policy_json, "acceptance policy")
                required_digest = str(policy.get("required_digest") or policy.get("evidence_digest") or policy.get("digest") or "")
                required_kind = str(policy.get("evidence_kind") or policy.get("kind") or "")
                if required_digest and required_digest not in current_digests:
                    raise EAAEFBorrowedTransactionNotReady(
                        f"completion lacks digest:{required_digest}"
                    )
                if required_kind and required_kind not in current_kinds:
                    raise EAAEFBorrowedTransactionNotReady(
                        f"completion lacks kind:{required_kind}"
                    )
                if not required_digest and not required_kind and not current_digests:
                    raise EAAEFBorrowedTransactionNotReady(
                        f"completion lacks criterion:{criterion}"
                    )
            if same_status:
                durable = owned.execute(
                    "SELECT receipt_cid, body_json FROM completion_receipts "
                    "WHERE task_cid=? AND attempt_id=? AND claim_cid=? "
                    "AND fencing_token=? ORDER BY completed_at DESC LIMIT 2",
                    [
                        task_cid,
                        live_claim["attempt_id"],
                        live_claim["claim_id"],
                        live_claim["fencing_token"],
                    ],
                ).fetchall()
                if (
                    len(durable) != 1
                    or _decode(durable[0][1], "durable completion replay") != receipt
                    or task["body"].get("completion_receipt") != receipt
                ):
                    raise EAAEFBorrowedTransactionConflict(
                        "terminal task replay differs from its unique durable receipt"
                    )
                return {
                    "schema": "ipfs_accelerate_py/agent-supervisor/database-task-cas@1",
                    "task": task,
                    "previous_status": prior,
                    "revision": expected,
                    "event_cursor": 0,
                    "changed": False,
                    "receipt_cid": str(durable[0][0]),
                }
        now = int(time.time_ns() // 1_000_000)
        revision = expected + 1
        body = dict(task["body"])
        if receipt:
            body["completion_receipt"] = receipt
        changed = owned.execute(
            "UPDATE tasks SET status = ?, revision = ?, updated_at = ?, body_json = ? "
            "WHERE task_cid = ? AND revision = ? RETURNING task_cid",
            [status, revision, _iso(now), _json(body, "task body"), task_cid, expected],
        ).fetchone()
        if changed is None:
            raise EAAEFBorrowedTransactionConflict("task revision changed during CAS")
        owned.execute(
            "INSERT INTO task_revisions(task_cid, revision, status, body_json, recorded_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [task_cid, revision, status, _json(body, "task revision body"), _iso(now)],
        )
        receipt_cid = ""
        if status in _SUCCESSFUL_TASK_STATUSES:
            evidence_digest = content_identity(
                {"task_cid": task_cid, "revision": revision, "receipt": receipt, "evidence_digests": evidence_digests}
            )
            receipt_cid = content_identity(
                {"namespace": "completion-receipt", "task_cid": task_cid, "revision": revision, "evidence_digest": evidence_digest}
            )
            owned.execute(
                "INSERT INTO completion_receipts(receipt_cid, task_cid, goal_cid, "
                "attempt_id, claim_cid, fencing_token, completed_at, validation_run_id, "
                "evidence_digest, body_json) VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, ?)",
                [receipt_cid, task_cid, task["goal_cid"], str(receipt.get("attempt_id") or ""), str(receipt.get("claim_id") or ""), int(receipt.get("fencing_token") or 0), _iso(now), evidence_digest, _json(receipt, "completion receipt")],
            )
        updated = self._task_record(owned, task_cid)
        assert updated is not None
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/database-task-cas@1",
            "task": updated,
            "previous_status": prior,
            "revision": revision,
            "event_cursor": 0,
            "changed": True,
            "receipt_cid": receipt_cid,
        }

    def _register_task(self, owned: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
        args = _exact(
            arguments,
            {"task_cid", "task_id", "dependency_task_cids", "body"},
            "coordination.register_task",
        )
        task_cid = _identifier(args["task_cid"], "task_cid")
        task = self._task_record(owned, task_cid)
        if task is None:
            raise EAAEFBorrowedTransactionConflict("registered task is absent")
        dependencies = args["dependency_task_cids"]
        if not isinstance(dependencies, list) or len(dependencies) > MAX_LIST_ITEMS:
            raise EAAEFBorrowedTransactionError("task dependencies are not bounded")
        supplied = sorted({_identifier(item, "dependency_task_cid") for item in dependencies})
        if supplied != sorted(task["dependencies"]):
            raise EAAEFBorrowedTransactionConflict(
                "coordination registration differs from canonical task dependencies"
            )
        task_id = _identifier(args["task_id"], "task_id")
        registration_body = _object(args["body"], "coordination task projection")
        basic = {"task_alias", "status"}
        with_priority = basic | {"priority"}
        if (
            task_id != task["task_alias"]
            or set(registration_body) not in (basic, with_priority)
            or registration_body.get("task_alias") != task["task_alias"]
            or registration_body.get("status") != task["status"]
            or (
                "priority" in registration_body
                and registration_body["priority"] != task["priority"]
            )
        ):
            raise EAAEFBorrowedTransactionConflict(
                "coordination registration differs from canonical task projection"
            )
        return {
            "task_cid": task_cid,
            "task_id": task_id,
            "dependency_task_cids": supplied,
            "registered": True,
            "canonical_task_store_reused": True,
        }

    @staticmethod
    def _expire_due(owned: Any, now_ms: int) -> None:
        due = owned.execute(
            "SELECT task_cid, claim_id FROM leases WHERE state = 'accepted' "
            "AND expires_at_ms <= ? AND lease_kind = 'task'",
            [now_ms],
        ).fetchall()
        for due_row in due:
            task_cid, claim_id = due_row[0], due_row[1]
            owned.execute(
                "UPDATE leases SET state = 'expired', release_reason = 'expired', revision = revision + 1 "
                "WHERE task_cid = ? AND claim_id = ? AND state = 'accepted'",
                [task_cid, claim_id],
            )
            owned.execute(
                "UPDATE task_claims SET state = 'expired', released_at_ms = ?, revision = revision + 1 "
                "WHERE claim_id = ? AND state = 'accepted'",
                [now_ms, claim_id],
            )

    def _claim_ready(
        self,
        owned: Any,
        arguments: Mapping[str, Any],
        *,
        lease: Mapping[str, Any],
        idempotency_key: str,
        fence_epoch: int,
    ) -> dict[str, Any] | None:
        args = _exact(
            arguments,
            {"owner_session_id", "lease_ms", "exclude_task_cids", "now_ms"},
            "coordination.claim_ready",
        )
        owner_session_id = _identifier(args["owner_session_id"], "owner_session_id")
        duration = _positive(args["lease_ms"], "lease_ms", maximum=MAX_LEASE_MS)
        now = _trusted_now(args["now_ms"])
        excluded_raw = args["exclude_task_cids"]
        if not isinstance(excluded_raw, list) or len(excluded_raw) > MAX_LIST_ITEMS:
            raise EAAEFBorrowedTransactionError("claim exclusions are not bounded")
        excluded = {_identifier(item, "excluded task_cid") for item in excluded_raw}
        self._expire_due(owned, now)
        task_rows = owned.execute(
            "SELECT task_cid, ordinal, status FROM tasks ORDER BY ordinal, task_cid"
        ).fetchall()
        completed = {str(row[0]) for row in task_rows if str(row[2]) in _COMPLETED}
        for task_row in task_rows:
            task_cid_raw, status_raw = task_row[0], task_row[2]
            task_cid = str(task_cid_raw)
            if task_cid in excluded or str(status_raw) not in _READY:
                continue
            blocked = owned.execute(
                "SELECT 1 FROM task_blocks WHERE task_cid=? AND state='active' LIMIT 1",
                [task_cid],
            ).fetchone()
            if blocked is not None:
                continue
            cooldown = owned.execute(
                "SELECT retry_not_before_ms FROM leases WHERE task_cid=?",
                [task_cid],
            ).fetchone()
            if cooldown is not None and int(cooldown[0] or 0) > now:
                continue
            dependencies = {
                str(row[0])
                for row in owned.execute(
                    "SELECT DISTINCT dependency_task_cid FROM task_dependencies WHERE task_cid = ?",
                    [task_cid],
                ).fetchall()
            }
            if not dependencies.issubset(completed):
                continue
            active = owned.execute(
                "SELECT 1 FROM leases WHERE task_cid = ? AND state = 'accepted' AND expires_at_ms > ?",
                [task_cid, now],
            ).fetchone()
            if active is not None:
                continue
            unresolved_attempt = owned.execute(
                "SELECT 1 FROM task_attempts WHERE task_cid=? AND status='running' LIMIT 1",
                [task_cid],
            ).fetchone()
            if unresolved_attempt is not None:
                continue
            attempt_number = int(
                owned.execute(
                    "SELECT COALESCE(MAX(attempt_number), 0) + 1 FROM task_attempts WHERE task_cid = ?",
                    [task_cid],
                ).fetchone()[0]
            )
            fencing_token = int(
                owned.execute(
                    "SELECT COALESCE(MAX(fencing_token), 0) + 1 FROM token_history WHERE task_cid = ?",
                    [task_cid],
                ).fetchone()[0]
            )
            base = {
                "task_cid": task_cid,
                "owner_session_id": owner_session_id,
                "attempt_number": attempt_number,
                "idempotency_key": idempotency_key,
            }
            claim_id = _id("claim", base)
            attempt_id = _id("attempt", base)
            lease_id = claim_id
            board_expires = _positive(
                lease.get("expires_at_ms"), "board lease expires_at_ms"
            )
            expires = min(now + duration, board_expires)
            if expires <= now:
                raise EAAEFBorrowedTransactionConflict(
                    "board scheduler lease cannot delegate an expired task lease"
                )
            principal = _identifier(lease.get("principal_did"), "claim principal_did")
            owned.execute(
                "INSERT INTO token_history(task_cid, fencing_token, recorded_at_ms) VALUES (?, ?, ?)",
                [task_cid, fencing_token, now],
            )
            prior = owned.execute(
                "SELECT task_cid FROM leases WHERE task_cid = ?", [task_cid]
            ).fetchone()
            lease_values = [
                lease_id,
                "",
                principal,
                attempt_number,
                fencing_token,
                expires,
                attempt_number,
                "accepted",
                now,
                None,
                0,
                owner_session_id,
                fence_epoch,
                1,
                "eaaef-task-lease@1",
                "{}",
                claim_id,
                attempt_id,
                attempt_number,
                "task",
                task_cid,
                "exclusive",
            ]
            if prior is None:
                owned.execute(
                    "INSERT INTO leases(task_cid, claim_cid, resolution_cid, claimant_did, "
                    "logical_epoch, fencing_token, expires_at_ms, attempt, state, started_at_ms, "
                    "release_reason, retry_not_before_ms, owner_session_id, fence_epoch, revision, "
                    "extension_schema, extension_json, claim_id, attempt_id, attempt_number, "
                    "lease_kind, scope_id, mode) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [task_cid, *lease_values],
                )
            else:
                owned.execute(
                    "UPDATE leases SET claim_cid=?, resolution_cid=?, claimant_did=?, logical_epoch=?, "
                    "fencing_token=?, expires_at_ms=?, attempt=?, state=?, started_at_ms=?, release_reason=?, "
                    "retry_not_before_ms=?, owner_session_id=?, fence_epoch=?, revision=?, extension_schema=?, "
                    "extension_json=?, claim_id=?, attempt_id=?, attempt_number=?, lease_kind=?, scope_id=?, mode=? "
                    "WHERE task_cid=?",
                    [*lease_values, task_cid],
                )
            task = self._task_record(owned, task_cid)
            assert task is not None
            claim_body = {"task_alias": task["task_alias"], "priority": task["priority"]}
            owned.execute(
                "INSERT INTO task_claims(claim_id, task_cid, owner_session_id, fencing_token, "
                "fence_epoch, claimed_at, expires_at, released_at, state, revision, idempotency_key, "
                "attempt_id, attempt_number, lease_id, worktree_id, claimed_at_ms, expires_at_ms, "
                "released_at_ms, body_json) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 'accepted', 1, ?, ?, ?, ?, '', ?, ?, NULL, ?)",
                [claim_id, task_cid, owner_session_id, fencing_token, fence_epoch, _iso(now), _iso(expires), idempotency_key, attempt_id, attempt_number, lease_id, now, expires, _json(claim_body, "claim body")],
            )
            owned.execute(
                "INSERT INTO task_attempts(attempt_id, task_cid, attempt_number, owner_session_id, "
                "fencing_token, fence_epoch, started_at, finished_at, status, revision, claim_id, "
                "task_alias, lease_id, committed_phase, started_at_ms, finished_at_ms, body_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 'running', 1, ?, ?, ?, 'claimed', ?, NULL, '{}')",
                [attempt_id, task_cid, attempt_number, owner_session_id, fencing_token, fence_epoch, _iso(now), claim_id, task["task_alias"], lease_id, now],
            )
            return self._claim_record(owned, claim_id)
        return None

    def _protect_operation(
        self,
        owned: Any,
        arguments: Mapping[str, Any],
        *,
        authorized_lease: Mapping[str, Any],
    ) -> dict[str, Any]:
        base = {
            "claim",
            "expected_task_cid",
            "expected_attempt_id",
            "expected_owner_session_id",
            "expected_fencing_token",
            "expected_fence_epoch",
            "now_ms",
        }
        args = _exact_one_of(
            arguments,
            (base, base | {"allow_logically_completed"}),
            "coordination.protect_claim",
        )
        identity = self._claim_identity(args["claim"])
        expectations = {
            "task_cid": _identifier(args["expected_task_cid"], "expected_task_cid"),
            "attempt_id": _identifier(args["expected_attempt_id"], "expected_attempt_id"),
            "owner_session_id": _identifier(args["expected_owner_session_id"], "expected_owner_session_id"),
            "fencing_token": _positive(args["expected_fencing_token"], "expected_fencing_token"),
            "fence_epoch": _positive(args["expected_fence_epoch"], "expected_fence_epoch"),
        }
        if any(identity[name] != expected for name, expected in expectations.items()):
            raise EAAEFBorrowedTransactionConflict("protected claim expectation differs")
        return self._protect(
            owned,
            identity,
            now_ms=_trusted_now(args["now_ms"]),
            authorized_lease=authorized_lease,
            allow_logically_completed=args.get("allow_logically_completed") is True,
        )

    def _renew(
        self, owned: Any, arguments: Mapping[str, Any], *, authorized_lease: Mapping[str, Any]
    ) -> dict[str, Any]:
        args = _exact(
            arguments,
            {"lease", "lease_ms", "expected_fencing_token", "expected_fence_epoch", "now_ms"},
            "coordination.renew_lease",
        )
        identity = self._claim_identity(args["lease"])
        now = _trusted_now(args["now_ms"])
        duration = _positive(args["lease_ms"], "lease_ms", maximum=MAX_LEASE_MS)
        if identity["fencing_token"] != _positive(args["expected_fencing_token"], "expected_fencing_token") or identity["fence_epoch"] != _positive(args["expected_fence_epoch"], "expected_fence_epoch"):
            raise EAAEFBorrowedTransactionConflict("renew fence expectation differs")
        self._protect(owned, identity, now_ms=now, authorized_lease=authorized_lease)
        board_expires = self._delegating_board_expiry(
            owned,
            principal_did=_identifier(
                authorized_lease.get("principal_did"), "authorized principal_did"
            ),
            fence_epoch=identity["fence_epoch"],
            now_ms=now,
        )
        expires = min(now + duration, board_expires)
        owned.execute(
            "UPDATE leases SET expires_at_ms = ?, revision = revision + 1 WHERE task_cid = ? "
            "AND claim_id = ? AND fencing_token = ? AND fence_epoch = ? AND state = 'accepted'",
            [expires, identity["task_cid"], identity["claim_id"], identity["fencing_token"], identity["fence_epoch"]],
        )
        owned.execute(
            "UPDATE task_claims SET expires_at_ms = ?, expires_at = ?, revision = revision + 1 "
            "WHERE claim_id = ? AND state = 'accepted'",
            [expires, _iso(expires), identity["claim_id"]],
        )
        result = self._claim_record(owned, identity["claim_id"])
        assert result is not None
        return result

    def _prepare_completion(
        self, owned: Any, arguments: Mapping[str, Any], *, authorized_lease: Mapping[str, Any]
    ) -> dict[str, Any]:
        args = _exact(
            arguments,
            {"claim", "control_expected_revision", "control_expected_status", "evidence_digest", "body", "now_ms"},
            "coordination.prepare_completion",
        )
        identity = self._claim_identity(args["claim"])
        now = _trusted_now(args["now_ms"])
        self._protect(owned, identity, now_ms=now, authorized_lease=authorized_lease)
        task = self._task_record(owned, identity["task_cid"])
        if task is None or task["revision"] != _positive(args["control_expected_revision"], "control_expected_revision") or task["status"] != str(args["control_expected_status"]):
            raise EAAEFBorrowedTransactionConflict(
                "completion preparation differs from current control task"
            )
        attempt = self._attempt(owned, identity["attempt_id"])
        validation_phase = owned.execute(
            "SELECT status, committed_at_ms, fencing_token, fence_epoch, revision, "
            "body_json FROM attempt_phases WHERE attempt_id=? AND phase_name='validation'",
            [identity["attempt_id"]],
        ).fetchall()
        if attempt is None or any(
            attempt[field] != identity[field]
            for field in (
                "claim_id",
                "task_cid",
                "attempt_id",
                "attempt_number",
                "lease_id",
                "owner_session_id",
                "fencing_token",
                "fence_epoch",
            )
        ):
            raise EAAEFBorrowedTransactionConflict(
                "completion preparation attempt identity differs"
            )
        if (
            attempt["status"] != "running"
            or attempt["committed_phase"] != "validation"
            or len(validation_phase) != 1
        ):
            raise EAAEFBorrowedTransactionNotReady(
                "completion preparation requires the exact validation phase"
            )
        phase_row = validation_phase[0]
        validation_body = _closed_validation_payload(
            _decode(phase_row[5], "validation phase body")
        )
        evidence_digest = _identifier(
            args["evidence_digest"], "evidence_digest"
        )
        if (
            str(phase_row[0]) != "committed"
            or int(phase_row[2]) != identity["fencing_token"]
            or int(phase_row[3]) != identity["fence_epoch"]
            or int(phase_row[4]) != attempt["revision"]
            or int(phase_row[1]) < attempt["started_at_ms"]
            or validation_body.get("evidence_digest") != evidence_digest
        ):
            raise EAAEFBorrowedTransactionConflict(
                "completion preparation validation provenance differs"
            )
        validation_run_id = _identifier(
            validation_body.get("run_id"), "validation run_id"
        )
        validation_result_id = _identifier(
            validation_body.get("result_id"), "validation result_id"
        )
        validation_evidence_id = _id(
            "validation-evidence",
            {
                "task_cid": identity["task_cid"],
                "digest": evidence_digest,
                "run_id": validation_run_id,
            },
        )
        validation_record = owned.execute(
            "SELECT runs.task_cid, runs.attempt_id, runs.status, "
            "results.task_cid, results.outcome, results.evidence_digest, "
            "evidence.task_cid, evidence.evidence_kind, evidence.digest, "
            "evidence.body_json FROM validation_runs AS runs "
            "JOIN validation_results AS results ON results.run_id=runs.run_id "
            "JOIN evidence_nodes AS evidence ON evidence.evidence_id=? "
            "WHERE runs.run_id=? AND results.result_id=?",
            [validation_evidence_id, validation_run_id, validation_result_id],
        ).fetchall()
        expected_evidence_body = {
            "run_id": validation_run_id,
            "result_id": validation_result_id,
            "argv": validation_body["argv"],
            "outcome": "passed",
        }
        if len(validation_record) != 1:
            raise EAAEFBorrowedTransactionConflict(
                "completion validation has no unique durable evidence"
            )
        validation_row = validation_record[0]
        if (
            str(validation_row[0]) != identity["task_cid"]
            or str(validation_row[1]) != identity["attempt_id"]
            or str(validation_row[2]) != "passed"
            or str(validation_row[3]) != identity["task_cid"]
            or str(validation_row[4]) != "passed"
            or str(validation_row[5]) != evidence_digest
            or str(validation_row[6]) != identity["task_cid"]
            or str(validation_row[7]) != "validation"
            or str(validation_row[8]) != evidence_digest
            or _decode(validation_row[9], "validation evidence body")
            != expected_evidence_body
        ):
            raise EAAEFBorrowedTransactionConflict(
                "completion validation durable evidence differs"
            )
        completion_body = _object(args["body"], "completion body")
        existing = self._barrier(owned, identity["task_cid"])
        stable_fields = {
            **identity,
            "control_expected_revision": int(args["control_expected_revision"]),
            "control_expected_status": str(args["control_expected_status"]),
            "evidence_digest": evidence_digest,
            "body": completion_body,
        }
        if existing is not None and all(
            existing[name] == value for name, value in stable_fields.items()
        ):
            return existing
        prepared = {
            "schema": EAAEF_TASK_COMPLETION_PREPARATION_SCHEMA,
            **stable_fields,
            "prepared_at_ms": now,
        }
        prepared["preparation_digest"] = _sha(prepared)
        if existing is not None:
            replaceable = (
                existing["status"] == "aborted"
                and identity["attempt_number"] > existing["attempt_number"]
                and identity["fencing_token"] > existing["fencing_token"]
                and identity["fence_epoch"] >= existing["fence_epoch"]
            )
            if not replaceable:
                raise EAAEFBorrowedTransactionConflict(
                    "another completion barrier owns this task"
                )
            history_id = _id(
                "completion-barrier-history",
                {
                    "task_cid": existing["task_cid"],
                    "preparation_digest": existing["preparation_digest"],
                    "terminal_status": existing["status"],
                },
            )
            owned.execute(
                "INSERT INTO eaaef_completion_barrier_history(history_id, task_cid, "
                "claim_id, attempt_id, attempt_number, lease_id, fencing_token, "
                "fence_epoch, preparation_digest, terminal_status, archived_at_ms, "
                "body_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    history_id,
                    existing["task_cid"],
                    existing["claim_id"],
                    existing["attempt_id"],
                    existing["attempt_number"],
                    existing["lease_id"],
                    existing["fencing_token"],
                    existing["fence_epoch"],
                    existing["preparation_digest"],
                    existing["status"],
                    now,
                    _json(existing, "completion barrier history"),
                ],
            )
            owned.execute(
                "DELETE FROM eaaef_completion_barriers WHERE task_cid=? "
                "AND preparation_digest=? AND status='aborted'",
                [existing["task_cid"], existing["preparation_digest"]],
            )
        owned.execute(
            "INSERT INTO eaaef_completion_barriers(task_cid, claim_id, attempt_id, "
            "attempt_number, lease_id, owner_session_id, fencing_token, fence_epoch, "
            "control_expected_revision, control_expected_status, evidence_digest, "
            "preparation_digest, prepared_at_ms, status, control_receipt_json, "
            "reconciliation_json, body_json, revision) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared', '{}', '{}', ?, 1)",
            [identity["task_cid"], identity["claim_id"], identity["attempt_id"], identity["attempt_number"], identity["lease_id"], identity["owner_session_id"], identity["fencing_token"], identity["fence_epoch"], prepared["control_expected_revision"], prepared["control_expected_status"], prepared["evidence_digest"], prepared["preparation_digest"], now, _json(prepared["body"], "completion body")],
        )
        result = self._barrier(owned, identity["task_cid"])
        assert result is not None
        return result

    @staticmethod
    def _control_task_from_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
        raw = _object(receipt, "control completion receipt")
        task = raw.get("task") if isinstance(raw.get("task"), Mapping) else raw
        return _object(task, "control task projection")

    def _complete_claim(
        self, owned: Any, arguments: Mapping[str, Any], *, authorized_lease: Mapping[str, Any]
    ) -> dict[str, Any]:
        args = _exact(
            arguments,
            {"claim", "control_completion_receipt", "now_ms"},
            "coordination.complete_claim",
        )
        identity = self._claim_identity(args["claim"])
        now = _trusted_now(args["now_ms"])
        self._protect(
            owned,
            identity,
            now_ms=now,
            authorized_lease=authorized_lease,
            allow_logically_completed=True,
        )
        barrier = self._barrier(owned, identity["task_cid"])
        if barrier is None or any(barrier[name] != identity[name] for name in identity):
            raise EAAEFBorrowedTransactionConflict("completion barrier identity differs")
        receipt = _object(args["control_completion_receipt"], "control completion receipt")
        expected_receipt_fields = {
            "schema",
            "task",
            "previous_status",
            "revision",
            "event_cursor",
            "changed",
            "receipt_cid",
        }
        task = self._task_record(owned, identity["task_cid"])
        durable_rows = owned.execute(
            "SELECT receipt_cid, task_cid, attempt_id, claim_cid, fencing_token, body_json "
            "FROM completion_receipts WHERE task_cid=? AND attempt_id=? "
            "AND claim_cid=? AND fencing_token=? ORDER BY completed_at DESC LIMIT 2",
            [
                identity["task_cid"],
                identity["attempt_id"],
                identity["claim_id"],
                identity["fencing_token"],
            ],
        ).fetchall()
        if len(durable_rows) != 1:
            raise EAAEFBorrowedTransactionConflict(
                "control completion has no unique canonical durable receipt"
            )
        durable = durable_rows[0]
        receipt_cid = _identifier(durable[0], "receipt_cid")
        durable_body = _decode(durable[5], "durable completion receipt")
        if set(durable_body) != _COMPLETION_RECEIPT_FIELDS:
            raise EAAEFBorrowedTransactionConflict(
                "durable completion receipt shape differs"
            )
        if (
            task is None
            or task["task_cid"] != identity["task_cid"]
            or task["status"] not in _COMPLETED
            or task["revision"] != barrier["control_expected_revision"] + 1
            or str(durable[1]) != identity["task_cid"]
            or str(durable[2]) != identity["attempt_id"]
            or str(durable[3]) != identity["claim_id"]
            or int(durable[4]) != identity["fencing_token"]
        ):
            raise EAAEFBorrowedTransactionConflict(
                "control completion does not match its prepared CAS"
            )
        binding = durable_body.get("coordination_preparation")
        if (
            not isinstance(binding, Mapping)
            or str(binding.get("preparation_digest") or "")
            != barrier["preparation_digest"]
            or durable_body.get("evidence_digest") != barrier["evidence_digest"]
            or any(
                durable_body.get(name) != identity[name]
                for name in (
                    "claim_id",
                    "attempt_id",
                    "lease_id",
                    "owner_session_id",
                    "fencing_token",
                    "fence_epoch",
                )
            )
        ):
            raise EAAEFBorrowedTransactionConflict(
                "control completion is not bound to the preparation"
            )
        canonical_receipt = {
            "schema": "ipfs_accelerate_py/agent-supervisor/database-task-cas@1",
            "task": task,
            "previous_status": barrier["control_expected_status"],
            "revision": task["revision"],
            "event_cursor": 0,
            "changed": True,
            "receipt_cid": receipt_cid,
        }
        if set(receipt) == expected_receipt_fields:
            caller_matches = receipt == canonical_receipt
        else:
            caller_matches = receipt == task
        if not caller_matches:
            raise EAAEFBorrowedTransactionConflict(
                "caller control completion differs from canonical stored state"
            )
        if barrier["status"] == "succeeded":
            if barrier["control_completion"] != canonical_receipt:
                raise EAAEFBorrowedTransactionConflict("completion replay diverged")
            return barrier
        if barrier["status"] != "prepared":
            raise EAAEFBorrowedTransactionConflict("completion barrier is terminal")
        owned.execute(
            "UPDATE eaaef_completion_barriers SET status='succeeded', control_receipt_json=?, "
            "revision=revision+1 WHERE task_cid=? AND preparation_digest=? AND status='prepared'",
            [_json(canonical_receipt, "control completion receipt"), identity["task_cid"], barrier["preparation_digest"]],
        )
        result = self._barrier(owned, identity["task_cid"])
        assert result is not None
        return result

    def _settle(
        self, owned: Any, arguments: Mapping[str, Any], *, authorized_lease: Mapping[str, Any]
    ) -> dict[str, Any]:
        args = _exact(arguments, {"claim", "reason", "now_ms"}, "coordination.settle_claim")
        identity = self._claim_identity(args["claim"])
        now = _trusted_now(args["now_ms"])
        current = self._protect(
            owned,
            identity,
            now_ms=now,
            authorized_lease=authorized_lease,
            allow_logically_completed=True,
        )
        barrier = self._barrier(owned, identity["task_cid"])
        if barrier is None or barrier["status"] != "succeeded":
            raise EAAEFBorrowedTransactionNotReady("claim has no promoted completion")
        attempt = self._attempt(owned, identity["attempt_id"])
        terminal_phase = owned.execute(
            "SELECT revision, status FROM attempt_phases WHERE attempt_id=? "
            "AND phase_name='complete'",
            [identity["attempt_id"]],
        ).fetchone()
        if (
            attempt is None
            or not self._historical_identity_matches(barrier, current, attempt)
            or attempt["status"] != "succeeded"
            or attempt["committed_phase"] != "complete"
            or attempt["finished_at_ms"] is None
            or terminal_phase is None
            or int(terminal_phase[0]) != attempt["revision"]
            or str(terminal_phase[1]) != "committed"
        ):
            raise EAAEFBorrowedTransactionNotReady(
                "claim settlement requires the exact terminal execution attempt"
            )
        if current["state"] == "released":
            return current
        reason = str(args["reason"] or "attempt_complete")[:256]
        owned.execute(
            "UPDATE leases SET state='released', release_reason=?, revision=revision+1 "
            "WHERE task_cid=? AND claim_id=? AND state IN ('accepted','completed')",
            [reason, identity["task_cid"], identity["claim_id"]],
        )
        owned.execute(
            "UPDATE task_claims SET state='released', released_at=?, released_at_ms=?, "
            "revision=revision+1 WHERE claim_id=? AND state IN ('accepted','completed')",
            [_iso(now), now, identity["claim_id"]],
        )
        result = self._claim_record(owned, identity["claim_id"])
        assert result is not None
        return result

    def _expire_claim(self, owned: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
        args = _exact(arguments, {"claim", "now_ms"}, "coordination.expire_claim")
        identity = self._claim_identity(args["claim"])
        now = _trusted_now(args["now_ms"])
        current = self._claim_record(owned, identity["claim_id"])
        if current is None:
            raise EAAEFBorrowedTransactionConflict("task claim is absent")
        exact = (
            "claim_id",
            "task_cid",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
            "attempt_id",
            "attempt_number",
            "lease_id",
        )
        if any(current[name] != identity[name] for name in exact):
            raise EAAEFBorrowedTransactionConflict(
                "expired task claim historical identity differs"
            )
        if current["state"] == "accepted" and current["expires_at_ms"] > now:
            raise EAAEFBorrowedTransactionConflict("live task claim cannot be expired")
        self._expire_due(owned, now)
        ambiguous = int(
            owned.execute(
                "SELECT (SELECT COUNT(*) FROM provider_invocations WHERE attempt_id=?) + "
                "(SELECT COUNT(*) FROM effect_claims WHERE attempt_id=?)",
                [identity["attempt_id"], identity["attempt_id"]],
            ).fetchone()[0]
        )
        if ambiguous:
            task = self._task_record(owned, identity["task_cid"])
            if task is None:
                raise EAAEFBorrowedTransactionConflict(
                    "expired ambiguous attempt task is absent"
                )
            if task["status"] not in _SUCCESSFUL_TASK_STATUSES:
                revision = task["revision"] + 1
                recovery_no_go = {
                    "outcome": "mutation_not_admitted",
                    "reason": "expired_attempt_has_provider_or_effect_reservation",
                    "attempt_id": identity["attempt_id"],
                }
                body = {
                    **task["body"],
                    "recovery_no_go": recovery_no_go,
                }
                owned.execute(
                    "UPDATE tasks SET status='quarantined', revision=?, updated_at=?, "
                    "body_json=? WHERE task_cid=? AND revision=?",
                    [
                        revision,
                        _iso(now),
                        _json(body, "quarantined task body"),
                        identity["task_cid"],
                        task["revision"],
                    ],
                )
                owned.execute(
                    "INSERT INTO task_revisions(task_cid, revision, status, body_json, "
                    "recorded_at) VALUES (?, ?, 'quarantined', ?, ?)",
                    [
                        identity["task_cid"],
                        revision,
                        _json(body, "quarantined task revision"),
                        _iso(now),
                    ],
                )
                attempt = self._attempt(owned, identity["attempt_id"])
                if attempt is None or attempt["status"] != "running":
                    raise EAAEFBorrowedTransactionConflict(
                        "ambiguous expired attempt is not running"
                    )
                attempt_revision = attempt["revision"] + 1
                changed = owned.execute(
                    "UPDATE task_attempts SET status='blocked', committed_phase='blocked', "
                    "finished_at=?, finished_at_ms=?, revision=?, body_json=? "
                    "WHERE attempt_id=? AND revision=? AND status='running' "
                    "RETURNING attempt_id",
                    [
                        _iso(now),
                        now,
                        attempt_revision,
                        _json(recovery_no_go, "ambiguous attempt body"),
                        identity["attempt_id"],
                        attempt["revision"],
                    ],
                ).fetchone()
                if changed is None:
                    raise EAAEFBorrowedTransactionConflict(
                        "ambiguous attempt quarantine lost its CAS"
                    )
                owned.execute(
                    "INSERT INTO attempt_phases(attempt_id, phase_name, entered_at, "
                    "exited_at, status, committed_at_ms, fencing_token, fence_epoch, "
                    "revision, body_json) VALUES (?, 'blocked', ?, ?, 'committed', ?, ?, ?, ?, ?)",
                    [
                        identity["attempt_id"],
                        _iso(now),
                        _iso(now),
                        now,
                        identity["fencing_token"],
                        identity["fence_epoch"],
                        attempt_revision,
                        _json(recovery_no_go, "ambiguous blocked phase body"),
                    ],
                )
                claim_body = {**current["body"], "recovery_no_go": recovery_no_go}
                owned.execute(
                    "UPDATE task_claims SET body_json=? WHERE claim_id=?",
                    [_json(claim_body, "ambiguous claim body"), identity["claim_id"]],
                )
        result = self._claim_record(owned, identity["claim_id"])
        assert result is not None
        return result

    def _reconcile(
        self, owned: Any, operation: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        if operation == "coordination.reconcile_promoted_completion":
            args = _exact(arguments, {"task_cid", "control_completion_receipt", "now_ms"}, operation)
            receipt_key = "control_completion_receipt"
            target = "succeeded"
        elif operation == "coordination.recover_prepared_completion":
            args = _exact(arguments, {"task_cid", "control_completion_receipt", "now_ms"}, operation)
            receipt_key = "control_completion_receipt"
            target = "succeeded"
        else:
            args = _exact(arguments, {"task_cid", "control_task_observation", "reason", "now_ms"}, operation)
            receipt_key = "control_task_observation"
            target = "aborted"
        task_cid = _identifier(args["task_cid"], "task_cid")
        now = _trusted_now(args["now_ms"])
        self._expire_due(owned, now)
        barrier = self._barrier(owned, task_cid)
        if barrier is None:
            raise EAAEFBorrowedTransactionNotReady("completion barrier is absent")
        claim = self._claim_record(owned, barrier["claim_id"])
        if claim is None:
            raise EAAEFBorrowedTransactionConflict("completion claim history is absent")
        attempt = self._attempt(owned, barrier["attempt_id"])
        if attempt is None or not self._historical_identity_matches(
            barrier, claim, attempt
        ):
            raise EAAEFBorrowedTransactionConflict(
                "completion historical claim/attempt fence provenance differs"
            )
        token = owned.execute(
            "SELECT 1 FROM token_history WHERE task_cid=? AND fencing_token=?",
            [task_cid, barrier["fencing_token"]],
        ).fetchone()
        if token is None:
            raise EAAEFBorrowedTransactionConflict(
                "completion historical fencing token provenance is absent"
            )
        observation = _object(args[receipt_key], receipt_key)
        actual_task = self._task_record(owned, task_cid)
        if actual_task is None:
            raise EAAEFBorrowedTransactionConflict("recovery task is absent")
        completed = actual_task["status"] in _SUCCESSFUL_TASK_STATUSES
        if target == "succeeded" and not completed:
            raise EAAEFBorrowedTransactionNotReady("control completion is absent")
        if target == "aborted" and completed:
            raise EAAEFBorrowedTransactionConflict("completed task cannot abort its barrier")
        if completed:
            expected_fields = {
                "schema",
                "task",
                "previous_status",
                "revision",
                "event_cursor",
                "changed",
                "receipt_cid",
            }
            durable_rows = owned.execute(
                "SELECT receipt_cid, attempt_id, claim_cid, fencing_token, body_json "
                "FROM completion_receipts WHERE task_cid=? AND attempt_id=? "
                "AND claim_cid=? AND fencing_token=? ORDER BY completed_at DESC LIMIT 2",
                [
                    task_cid,
                    barrier["attempt_id"],
                    barrier["claim_id"],
                    barrier["fencing_token"],
                ],
            ).fetchall()
            if len(durable_rows) != 1:
                raise EAAEFBorrowedTransactionConflict(
                    "recovery completion has no unique canonical durable receipt"
                )
            durable = durable_rows[0]
            receipt_cid = _identifier(durable[0], "receipt_cid")
            durable_body = _decode(durable[4], "recovery durable completion")
            canonical = {
                "schema": "ipfs_accelerate_py/agent-supervisor/database-task-cas@1",
                "task": actual_task,
                "previous_status": barrier["control_expected_status"],
                "revision": actual_task["revision"],
                "event_cursor": 0,
                "changed": True,
                "receipt_cid": receipt_cid,
            }
            if set(observation) == expected_fields:
                caller_matches = observation == canonical
            else:
                caller_matches = observation == actual_task
            if not caller_matches:
                raise EAAEFBorrowedTransactionConflict(
                    "recovery caller receipt differs from canonical stored state"
                )
            canonical_observation = canonical
            binding = durable_body.get("coordination_preparation")
            if (
                str(durable[1]) != barrier["attempt_id"]
                or str(durable[2]) != barrier["claim_id"]
                or int(durable[3]) != barrier["fencing_token"]
                or not isinstance(binding, Mapping)
                or str(binding.get("preparation_digest") or "")
                != barrier["preparation_digest"]
                or durable_body.get("evidence_digest") != barrier["evidence_digest"]
            ):
                raise EAAEFBorrowedTransactionConflict(
                    "recovery durable completion differs from its barrier"
                )
            if actual_task["revision"] != barrier["control_expected_revision"] + 1:
                raise EAAEFBorrowedTransactionConflict(
                    "recovered completion revision differs from its preparation"
                )
        elif (
            observation != actual_task
            or actual_task["revision"] != barrier["control_expected_revision"]
            or actual_task["status"] != barrier["control_expected_status"]
        ):
            raise EAAEFBorrowedTransactionConflict(
                "aborted completion observation differs from its preparation"
            )
        else:
            canonical_observation = observation
        if barrier["status"] == target and barrier["reconciliation"]:
            durable_reconciliation = barrier["reconciliation"]
            successful_recovery_operations = {
                "coordination.reconcile_promoted_completion",
                "coordination.recover_prepared_completion",
            }
            operation_matches = durable_reconciliation.get("operation") == operation
            if target == "succeeded":
                operation_matches = (
                    durable_reconciliation.get("operation")
                    in successful_recovery_operations
                    and operation in successful_recovery_operations
                )
            if (
                set(durable_reconciliation) != _RECONCILIATION_FIELDS
                or not operation_matches
                or barrier["control_completion"] != canonical_observation
            ):
                raise EAAEFBorrowedTransactionConflict(
                    "terminal completion recovery replay diverged"
                )
            return {**durable_reconciliation, "replayed": True}
        if barrier["status"] not in {"prepared", target}:
            raise EAAEFBorrowedTransactionConflict(
                "completion barrier has a different terminal reconciliation"
            )
        if operation != "coordination.reconcile_promoted_completion" and claim["state"] == "accepted" and claim["expires_at_ms"] > now:
            raise EAAEFBorrowedTransactionConflict("live completion cannot be recovered")
        lease_state = "released" if target == "succeeded" else "expired"
        reconciliation = {
            "operation": operation,
            "task_cid": task_cid,
            "claim_id": barrier["claim_id"],
            "attempt_id": barrier["attempt_id"],
            "status": target,
            "observed_at_ms": now,
            "lease_state": lease_state,
            "replayed": False,
        }
        if target == "succeeded":
            owned.execute(
                "UPDATE eaaef_completion_barriers SET status=?, "
                "control_receipt_json=?, reconciliation_json=?, revision=revision+1 "
                "WHERE task_cid=? AND preparation_digest=?",
                [
                    target,
                    _json(canonical_observation, "canonical recovery completion"),
                    _json(reconciliation, "reconciliation"),
                    task_cid,
                    barrier["preparation_digest"],
                ],
            )
        else:
            owned.execute(
                "UPDATE eaaef_completion_barriers SET status=?, "
                "control_receipt_json=?, reconciliation_json=?, revision=revision+1 WHERE task_cid=? "
                "AND preparation_digest=?",
                [
                    target,
                    _json(canonical_observation, "canonical abort observation"),
                    _json(reconciliation, "reconciliation"),
                    task_cid,
                    barrier["preparation_digest"],
                ],
            )
        if target == "aborted" and actual_task["status"] != "ready":
            retry_revision = actual_task["revision"] + 1
            owned.execute(
                "UPDATE tasks SET status='ready', revision=?, updated_at=? "
                "WHERE task_cid=? AND revision=?",
                [retry_revision, _iso(now), task_cid, actual_task["revision"]],
            )
            owned.execute(
                "INSERT INTO task_revisions(task_cid, revision, status, body_json, "
                "recorded_at) VALUES (?, ?, 'ready', ?, ?)",
                [
                    task_cid,
                    retry_revision,
                    _json(actual_task["body"], "retry task body"),
                    _iso(now),
                ],
            )
        owned.execute(
            "UPDATE leases SET state=?, release_reason=?, revision=revision+1 WHERE task_cid=? AND claim_id=?",
            [lease_state, operation, task_cid, barrier["claim_id"]],
        )
        owned.execute(
            "UPDATE task_claims SET state=?, released_at=?, released_at_ms=?, revision=revision+1 WHERE claim_id=?",
            [lease_state, _iso(now), now, barrier["claim_id"]],
        )
        return reconciliation

    def _ensure_attempt(self, owned: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
        args = _exact(arguments, {"attempt", "claimed_phase"}, "execution.ensure_attempt")
        attempt, claimed = self._attempt_input(args["attempt"], args["claimed_phase"])
        identity = {
            name: attempt[name]
            for name in (
                "claim_id",
                "task_cid",
                "owner_session_id",
                "fencing_token",
                "fence_epoch",
                "attempt_id",
                "attempt_number",
                "lease_id",
            )
        }
        current = self._attempt(owned, identity["attempt_id"])
        if current is None:
            raise EAAEFBorrowedTransactionConflict(
                "execution attempt must be created by its exact task claim"
            )
        checks = {
            "claim_id": identity["claim_id"],
            "task_cid": identity["task_cid"],
            "owner_session_id": identity["owner_session_id"],
            "fencing_token": identity["fencing_token"],
            "fence_epoch": identity["fence_epoch"],
            "lease_id": identity["lease_id"],
        }
        if any(current[key] != value for key, value in checks.items()):
            raise EAAEFBorrowedTransactionConflict("execution attempt identity differs")
        alias = attempt["task_alias"]
        owned.execute(
            "UPDATE task_attempts SET task_alias=?, body_json=? WHERE attempt_id=?",
            [alias, _json(attempt["body"], "attempt body"), identity["attempt_id"]],
        )
        existing = owned.execute(
            "SELECT phase_name FROM attempt_phases WHERE attempt_id=? AND phase_name='claimed'",
            [identity["attempt_id"]],
        ).fetchone()
        if existing is None:
            owned.execute(
                "INSERT INTO attempt_phases(attempt_id, phase_name, entered_at, exited_at, status, "
                "committed_at_ms, fencing_token, fence_epoch, revision, body_json) "
                "VALUES (?, 'claimed', ?, ?, 'committed', ?, ?, ?, ?, ?)",
                [identity["attempt_id"], _iso(claimed["committed_at_ms"]), _iso(claimed["committed_at_ms"]), claimed["committed_at_ms"], identity["fencing_token"], identity["fence_epoch"], claimed["revision"], _json(claimed["body"], "claimed phase body")],
            )
        result = self._attempt(owned, identity["attempt_id"])
        assert result is not None
        return result

    def _phase_evidence_body(
        self,
        owned: Any,
        *,
        phase: str,
        attempt: Mapping[str, Any],
        value: Any,
    ) -> dict[str, Any]:
        """Join an execution phase to evidence already durable in this txn."""

        body = _object(value, f"{phase} phase body")
        if phase not in {"provider", "effect", "validation"}:
            return body
        if phase in {"provider", "effect"}:
            variants = (
                {"idempotency_key", "result"},
                {"replayed", "idempotency_key"},
            )
            body = _exact_one_of(body, variants, f"{phase} phase body")
            key = _identifier(body["idempotency_key"], f"{phase} idempotency_key")
            if "replayed" in body and body["replayed"] is not True:
                raise EAAEFBorrowedTransactionError(
                    f"{phase} replay marker must be true"
                )
            if phase == "provider":
                row = owned.execute(
                    "SELECT invocation_id, task_cid, owner_session_id, fencing_token, "
                    "fence_epoch, status, result_json FROM provider_invocations "
                    "WHERE attempt_id=? AND idempotency_key=?",
                    [attempt["attempt_id"], key],
                ).fetchone()
                operation_key = ""
            else:
                row = owned.execute(
                    "SELECT effect_id, task_cid, owner_session_id, fencing_token, "
                    "fence_epoch, state, result_json, operation_key FROM effect_claims "
                    "WHERE attempt_id=? AND idempotency_key=?",
                    [attempt["attempt_id"], key],
                ).fetchone()
                operation_key = "" if row is None else str(row[7] or "")
            durable_result = {} if row is None else _decode(
                row[6], f"{phase} durable result"
            )
            expected_record_id = eaaef_reservation_id(
                kind=phase,
                attempt_id=str(attempt["attempt_id"]),
                idempotency_key=key,
            )
            if (
                row is None
                or str(row[0]) != expected_record_id
                or str(row[1]) != attempt["task_cid"]
                or str(row[2]) != attempt["owner_session_id"]
                or int(row[3]) != attempt["fencing_token"]
                or int(row[4]) != attempt["fence_epoch"]
                or str(row[5]) != "committed"
                or (
                    "result" in body
                    and _object(body["result"], f"{phase} phase result")
                    != durable_result
                )
                or (
                    phase == "effect"
                    and (
                        not operation_key
                        or (
                            durable_result.get("effect_key") is not None
                            and durable_result.get("effect_key") != operation_key
                        )
                    )
                )
            ):
                raise EAAEFBorrowedTransactionNotReady(
                    f"{phase} phase lacks its exact committed reservation"
                )
            normalized = {
                "idempotency_key": key,
                "result": durable_result,
            }
            if phase == "effect":
                normalized["operation_key"] = operation_key
            if "replayed" in body:
                normalized["replayed"] = True
            return normalized

        validation = _exact(
            body,
            {"outcome", "evidence_digest", "argv", "body"},
            "validation phase body",
        )
        closed = _closed_validation_payload(validation)
        detail = _exact(
            _object(closed["body"], "validation authority evidence"),
            {
                "schema",
                "validator",
                "task_cid",
                "attempt_id",
                "control_claim_id",
                "dispatch_claim_cid",
                "owner_session_id",
                "fencing_token",
                "fence_epoch",
                "authority_cid",
                "admission_receipt",
                "delivery_mode",
                "merge_commit",
                "patch_artifact_cid",
            },
            "validation authority evidence",
        )
        admission = _exact(
            _object(detail["admission_receipt"], "host merge admission"),
            {
                "schema",
                "interface",
                "decision",
                "delivery_mode",
                "task_cid",
                "attempt_id",
                "claim_cid",
                "accepted_result_receipt_id",
                "patch_artifact_cid",
                "reviewer_principal_did",
                "effect_authority_cid",
                "merge_commit",
                "receipt_cid",
            },
            "host merge admission",
        )
        admission_body = {
            key: item for key, item in admission.items() if key != "receipt_cid"
        }
        admission_cid = str(admission.get("receipt_cid") or "")
        authority_cid = str(detail.get("authority_cid") or "")
        patch_artifact_cid = str(detail.get("patch_artifact_cid") or "")
        accepted_result_cid = str(
            admission.get("accepted_result_receipt_id") or ""
        )
        reviewer = str(admission.get("reviewer_principal_did") or "")
        delivery_mode = str(detail.get("delivery_mode") or "")
        merge_commit = str(detail.get("merge_commit") or "")
        if (
            detail.get("schema") != EAAEF_CONTAINER_VALIDATION_EVIDENCE_SCHEMA
            or detail.get("validator") != "ExternalAgentContainerWorkerDispatcher@1"
            or detail.get("task_cid") != attempt["task_cid"]
            or detail.get("attempt_id") != attempt["attempt_id"]
            or detail.get("control_claim_id") != attempt["claim_id"]
            or detail.get("owner_session_id") != attempt["owner_session_id"]
            or detail.get("fencing_token") != attempt["fencing_token"]
            or detail.get("fence_epoch") != attempt["fence_epoch"]
            or admission.get("schema")
            != "ipfs_accelerate_py/agent-supervisor/external-agent-host-merge-admission@1"
            or admission.get("interface")
            != "ExternalAgentContainerWorkerDispatcher@1"
            or admission.get("decision") != "accepted"
            or delivery_mode not in {"merge_accepted", "reviewed_patch"}
            or admission.get("delivery_mode") != delivery_mode
            or admission.get("task_cid") != attempt["task_cid"]
            or admission.get("attempt_id") != attempt["attempt_id"]
            or admission.get("claim_cid") != detail["dispatch_claim_cid"]
            or admission.get("effect_authority_cid") != authority_cid
            or admission.get("patch_artifact_cid") != patch_artifact_cid
            or admission.get("merge_commit") != merge_commit
            or closed["evidence_digest"] != admission_cid
            or admission_cid != _sha(admission_body)
            or _SHA256_CID.fullmatch(admission_cid) is None
            or _SHA256_CID.fullmatch(authority_cid) is None
            or _SHA256_CID.fullmatch(patch_artifact_cid) is None
            or _SHA256_CID.fullmatch(accepted_result_cid) is None
            or not reviewer.startswith("did:key:z")
            or (delivery_mode == "merge_accepted" and _GIT_COMMIT.fullmatch(merge_commit) is None)
            or (delivery_mode == "reviewed_patch" and merge_commit)
        ):
            raise EAAEFBorrowedTransactionConflict(
                "validation authority evidence is not exact or content-addressed"
            )
        dispatch_row = owned.execute(
            "SELECT state, result_json, body_json, effect_kind, operation_key, "
            "task_cid, owner_session_id, fencing_token, fence_epoch "
            "FROM effect_claims WHERE effect_id=? AND attempt_id=?",
            [detail["dispatch_claim_cid"], attempt["attempt_id"]],
        ).fetchone()
        provider_phase = owned.execute(
            "SELECT body_json FROM attempt_phases "
            "WHERE attempt_id=? AND phase_name='provider' AND status='committed'",
            [attempt["attempt_id"]],
        ).fetchone()
        effect_phase = owned.execute(
            "SELECT body_json FROM attempt_phases "
            "WHERE attempt_id=? AND phase_name='effect' AND status='committed'",
            [attempt["attempt_id"]],
        ).fetchone()
        dispatch_result = (
            {} if dispatch_row is None else _decode(
                dispatch_row[1], "container dispatch durable result"
            )
        )
        dispatch_reservation = (
            {} if dispatch_row is None else _decode(
                dispatch_row[2], "container dispatch durable reservation"
            )
        )
        durable_dispatch_claim = (
            {}
            if dispatch_row is None
            else self._container_dispatch_claim(
                dispatch_reservation.get("claim"), attempt=attempt
            )
        )
        provider_result = (
            {}
            if provider_phase is None
            else _decode(provider_phase[0], "provider phase evidence").get(
                "result", {}
            )
        )
        effect_result = (
            {}
            if effect_phase is None
            else _decode(effect_phase[0], "effect phase evidence").get(
                "result", {}
            )
        )
        effect_result = _object(effect_result, "container effect phase result")
        effect_body = {
            key: item for key, item in effect_result.items() if key != "receipt_cid"
        }
        if (
            dispatch_row is None
            or str(dispatch_row[0]) != "committed"
            or str(dispatch_row[3]) != EAAEF_CONTAINER_DISPATCH_OPERATION_KIND
            or str(dispatch_row[4]) != detail["dispatch_claim_cid"]
            or str(dispatch_row[5]) != attempt["task_cid"]
            or str(dispatch_row[6]) != attempt["owner_session_id"]
            or int(dispatch_row[7]) != attempt["fencing_token"]
            or int(dispatch_row[8]) != attempt["fence_epoch"]
            or durable_dispatch_claim.get("claim_cid")
            != detail["dispatch_claim_cid"]
            or reviewer
            in {
                durable_dispatch_claim.get("worker_principal_did"),
                durable_dispatch_claim.get("provider_principal_did"),
            }
            or dispatch_reservation.get("reservation_id")
            != _sha({"reservation": detail["dispatch_claim_cid"]})
            or dispatch_result.get("receipt_id")
            != admission["accepted_result_receipt_id"]
            or dispatch_result.get("patch_artifact_cid") != patch_artifact_cid
            or dispatch_result.get("task_cid") != attempt["task_cid"]
            or dispatch_result.get("attempt_id") != attempt["attempt_id"]
            or dispatch_result.get("claim_cid") != detail["dispatch_claim_cid"]
            or provider_phase is None
            or _object(provider_result, "container provider phase result")
            != dispatch_result
            or effect_phase is None
            or set(effect_result) != _CONTAINER_EFFECT_RESULT_FIELDS
            or effect_result.get("schema")
            != "ipfs_accelerate_py/agent-supervisor/external-agent-container-effect-receipt@1"
            or effect_result.get("interface")
            != "ExternalAgentContainerWorkerDispatcher@1"
            or effect_result.get("status") != "applied"
            or effect_result.get("task_cid") != attempt["task_cid"]
            or effect_result.get("attempt_id") != attempt["attempt_id"]
            or effect_result.get("claim_cid") != detail["dispatch_claim_cid"]
            or effect_result.get("accepted_result_receipt_id")
            != admission["accepted_result_receipt_id"]
            or effect_result.get("patch_artifact_cid") != patch_artifact_cid
            or effect_result.get("task_result_accepted") is not False
            or effect_result.get("merge_admitted") is not False
            or effect_result.get("host_mutation_performed") is not False
            or effect_result.get("receipt_cid") != _sha(effect_body)
        ):
            raise EAAEFBorrowedTransactionConflict(
                "validation is not joined to the exact durable container execution"
            )
        recorded = self._record_validation(
            owned,
            {
                "task_cid": attempt["task_cid"],
                "attempt_id": attempt["attempt_id"],
                "outcome": closed["outcome"],
                "evidence_digest": closed["evidence_digest"],
                "argv": closed["argv"],
                "body": _object(closed.get("body") or {}, "validation phase detail"),
            },
            attempt_default=str(attempt["attempt_id"]),
        )
        return {
            **closed,
            "run_id": recorded["run_id"],
            "result_id": recorded["result_id"],
        }

    def _commit_phase(self, owned: Any, arguments: Mapping[str, Any], *, reconciled: bool) -> dict[str, Any] | None:
        fields = {"attempt_id", "expected_revision", "expected_status", "committed_phase", "status", "finished_at_ms", "revision", "committed_at_ms", "fencing_token", "fence_epoch", "body"}
        if reconciled:
            fields |= {"preparation", "reconciliation"}
        operation = "execution.commit_reconciled_attempt" if reconciled else "execution.commit_phase"
        args = _exact(arguments, fields, operation)
        attempt_id = _identifier(args["attempt_id"], "attempt_id")
        preparation_input = (
            _object(args["preparation"], "reconciliation preparation")
            if reconciled
            else {}
        )
        expired_without_barrier = (
            reconciled and set(preparation_input) == _FULL_CLAIM_FIELDS
        )
        current = self._attempt(owned, attempt_id)
        if current is None:
            return None
        expected_status = str(args["expected_status"])
        allowed_current = {expected_status}
        if expired_without_barrier and expected_status == "running":
            allowed_current.add("expired")
        if current["revision"] != _positive(args["expected_revision"], "expected_revision") or current["status"] not in allowed_current:
            return None
        if current["fencing_token"] != _positive(args["fencing_token"], "fencing_token") or current["fence_epoch"] != _positive(args["fence_epoch"], "fence_epoch"):
            raise EAAEFBorrowedTransactionConflict("attempt phase fence is stale")
        phase = str(args["committed_phase"] or "").strip().lower()
        if phase not in set(_PHASES) | _TERMINAL_PHASES:
            raise EAAEFBorrowedTransactionError("attempt phase is unsupported")
        target_status = str(args["status"] or "").strip().lower()
        finished = args["finished_at_ms"]
        if not reconciled and phase in _PHASES:
            old_rank = _PHASES.index(current["committed_phase"])
            new_rank = _PHASES.index(phase)
            if new_rank != old_rank + 1:
                raise EAAEFBorrowedTransactionConflict("attempt phase skips a boundary")
        if not reconciled:
            if phase in {"context", "provider", "effect", "validation"}:
                if target_status != "running" or finished is not None:
                    raise EAAEFBorrowedTransactionConflict(
                        "nonterminal attempt phase has a terminal status/time"
                    )
            elif phase == "complete":
                if (
                    current["committed_phase"] != "validation"
                    or target_status != "succeeded"
                    or finished is None
                ):
                    raise EAAEFBorrowedTransactionConflict(
                        "complete phase is not the exact validation-to-success transition"
                    )
                barrier = self._barrier(owned, current["task_cid"])
                claim = self._claim_record(owned, current["claim_id"])
                if (
                    barrier is None
                    or claim is None
                    or barrier["status"] != "succeeded"
                    or not self._historical_identity_matches(barrier, claim, current)
                    or not barrier["control_completion"]
                ):
                    raise EAAEFBorrowedTransactionNotReady(
                        "successful attempt phase lacks its exact promoted completion"
                    )
            elif phase in {"failed", "blocked"}:
                if target_status != phase or finished is None:
                    raise EAAEFBorrowedTransactionConflict(
                        "terminal attempt phase/status/time differs"
                    )
            else:
                raise EAAEFBorrowedTransactionConflict(
                    "claimed phase cannot be recommitted"
                )
        revision = _positive(args["revision"], "revision")
        if revision != current["revision"] + 1:
            raise EAAEFBorrowedTransactionConflict("attempt revision is not consecutive")
        body = (
            self._phase_evidence_body(
                owned,
                phase=phase,
                attempt=current,
                value=args["body"],
            )
            if not reconciled
            else _object(args["body"], "attempt phase body")
        )
        if expired_without_barrier:
            preparation = preparation_input
            identity = self._claim_identity(preparation)
            reconciliation = _exact(
                _object(args["reconciliation"], "expired attempt reconciliation"),
                _EXPIRED_ATTEMPT_RECONCILIATION_FIELDS,
                "expired attempt reconciliation",
            )
            claim = self._claim_record(owned, identity["claim_id"])
            lease = owned.execute(
                "SELECT claim_id, attempt_id, attempt_number, claim_cid, "
                "owner_session_id, fencing_token, fence_epoch, state, lease_kind, "
                "scope_id, mode FROM leases WHERE task_cid=?",
                [identity["task_cid"]],
            ).fetchone()
            token = owned.execute(
                "SELECT 1 FROM token_history WHERE task_cid=? AND fencing_token=?",
                [identity["task_cid"], identity["fencing_token"]],
            ).fetchone()
            historical_fields = (
                "claim_id",
                "task_cid",
                "attempt_id",
                "attempt_number",
                "lease_id",
                "owner_session_id",
                "fencing_token",
                "fence_epoch",
            )
            lease_expected = (
                identity["claim_id"],
                identity["attempt_id"],
                identity["attempt_number"],
                identity["lease_id"],
                identity["owner_session_id"],
                identity["fencing_token"],
                identity["fence_epoch"],
                "expired",
                "task",
                identity["task_cid"],
                "exclusive",
            )
            reconciliation_expected = {
                "task_cid": identity["task_cid"],
                "claim_id": identity["claim_id"],
                "attempt_id": identity["attempt_id"],
                "status": "expired",
                "lease_state": "expired",
                "retry_required": True,
                "provider_evidence_reused": False,
                "effect_evidence_reused": False,
                "reason": "coordination_lease_expired_before_completion",
            }
            if (
                claim is None
                or any(claim[name] != identity[name] for name in historical_fields)
                or any(current[name] != identity[name] for name in historical_fields)
                or claim["state"] != "expired"
                or lease is None
                or _row_values(lease) != lease_expected
                or token is None
                or self._barrier(owned, identity["task_cid"]) is not None
                or reconciliation != reconciliation_expected
                or phase != "failed"
                or target_status != "failed"
                or finished is None
            ):
                raise EAAEFBorrowedTransactionConflict(
                    "expired attempt recovery differs from exact historical authority"
                )
            ambiguous = int(
                owned.execute(
                    "SELECT (SELECT COUNT(*) FROM provider_invocations WHERE attempt_id=?) + "
                    "(SELECT COUNT(*) FROM effect_claims WHERE attempt_id=?)",
                    [attempt_id, attempt_id],
                ).fetchone()[0]
            )
            if ambiguous:
                raise EAAEFBorrowedTransactionNotReady(
                    "expired attempt has ambiguous provider/effect reservations"
                )
            task = self._task_record(owned, identity["task_cid"])
            if task is None or task["status"] in _SUCCESSFUL_TASK_STATUSES | {
                "quarantined"
            }:
                raise EAAEFBorrowedTransactionConflict(
                    "expired attempt task is not safely retryable"
                )
            if task["status"] != "ready":
                task_revision = task["revision"] + 1
                owned.execute(
                    "UPDATE tasks SET status='ready', revision=?, updated_at=? "
                    "WHERE task_cid=? AND revision=?",
                    [
                        task_revision,
                        _iso(int(time.time_ns() // 1_000_000)),
                        identity["task_cid"],
                        task["revision"],
                    ],
                )
                owned.execute(
                    "INSERT INTO task_revisions(task_cid, revision, status, body_json, "
                    "recorded_at) VALUES (?, ?, 'ready', ?, ?)",
                    [
                        identity["task_cid"],
                        task_revision,
                        _json(task["body"], "expired retry task body"),
                        _iso(int(time.time_ns() // 1_000_000)),
                    ],
                )
            body = {
                **body,
                "preparation": preparation,
                "reconciliation": reconciliation,
            }
        elif reconciled:
            preparation = _exact(
                preparation_input,
                _BARRIER_FIELDS,
                "reconciliation preparation",
            )
            reconciliation = _exact(
                _object(args["reconciliation"], "reconciliation receipt"),
                _RECONCILIATION_FIELDS,
                "reconciliation receipt",
            )
            barrier = self._barrier(owned, current["task_cid"])
            claim = self._claim_record(owned, current["claim_id"])
            immutable = (
                "task_cid",
                "claim_id",
                "attempt_id",
                "attempt_number",
                "lease_id",
                "owner_session_id",
                "fencing_token",
                "fence_epoch",
                "control_expected_revision",
                "control_expected_status",
                "evidence_digest",
                "preparation_digest",
                "prepared_at_ms",
            )
            durable_reconciliation = (
                barrier["reconciliation"] if barrier is not None else {}
            )
            comparable_reconciliation = {
                **reconciliation,
                "replayed": durable_reconciliation.get("replayed"),
            }
            if (
                barrier is None
                or claim is None
                or any(preparation[name] != barrier[name] for name in immutable)
                or not self._historical_identity_matches(barrier, claim, current)
                or comparable_reconciliation != durable_reconciliation
                or reconciliation["task_cid"] != barrier["task_cid"]
                or reconciliation["claim_id"] != barrier["claim_id"]
                or reconciliation["attempt_id"] != barrier["attempt_id"]
            ):
                raise EAAEFBorrowedTransactionConflict("reconciled attempt lacks exact barrier")
            expected_terminal = (
                ("complete", "succeeded", "succeeded", "released")
                if barrier["status"] == "succeeded"
                else ("failed", "failed", "aborted", "expired")
            )
            if (
                phase,
                target_status,
                reconciliation["status"],
                reconciliation["lease_state"],
            ) != expected_terminal or finished is None:
                raise EAAEFBorrowedTransactionConflict(
                    "reconciled attempt terminal transition differs"
                )
            body = {
                **body,
                "preparation": preparation,
                "reconciliation": durable_reconciliation,
            }
        committed_at = _positive(args["committed_at_ms"], "committed_at_ms")
        if finished is not None:
            finished = _positive(finished, "finished_at_ms")
            if finished < committed_at:
                raise EAAEFBorrowedTransactionConflict(
                    "attempt finished before its committed phase"
                )
        prior_time = owned.execute(
            "SELECT COALESCE(MAX(committed_at_ms), 0) FROM attempt_phases "
            "WHERE attempt_id=?",
            [attempt_id],
        ).fetchone()
        if committed_at < max(current["started_at_ms"], int(prior_time[0] or 0)):
            raise EAAEFBorrowedTransactionConflict(
                "attempt phase time moves backwards"
            )
        changed = owned.execute(
            "UPDATE task_attempts SET committed_phase=?, status=?, finished_at=?, "
            "finished_at_ms=?, revision=?, body_json=? WHERE attempt_id=? AND revision=? "
            "AND status IN (" + ",".join("?" for _ in allowed_current) + ") RETURNING attempt_id",
            [phase, target_status, None if finished is None else _iso(finished), finished, revision, _json(body, "attempt body"), attempt_id, current["revision"], *sorted(allowed_current)],
        ).fetchone()
        if changed is None:
            return None
        existing = owned.execute(
            "SELECT revision, body_json FROM attempt_phases WHERE attempt_id=? AND phase_name=?",
            [attempt_id, phase],
        ).fetchone()
        phase_json = _json(body, "attempt phase body")
        if existing is None:
            owned.execute(
                "INSERT INTO attempt_phases(attempt_id, phase_name, entered_at, exited_at, status, "
                "committed_at_ms, fencing_token, fence_epoch, revision, body_json) "
                "VALUES (?, ?, ?, ?, 'committed', ?, ?, ?, ?, ?)",
                [attempt_id, phase, _iso(committed_at), _iso(committed_at), committed_at, current["fencing_token"], current["fence_epoch"], revision, phase_json],
            )
        elif int(existing[0]) != revision or str(existing[1]) != phase_json:
            raise EAAEFBorrowedTransactionConflict("attempt phase replay diverged")
        if reconciled:
            recovery_event_body = {
                "status": target_status,
                "phase": phase,
                "revision": revision,
                "reconciliation": body.get("reconciliation"),
            }
            self._event(
                owned,
                event_id=_id(
                    "event",
                    {
                        "type": "attempt_cross_store_reconciled",
                        "attempt_id": attempt_id,
                        "revision": revision,
                    },
                ),
                event_type="attempt_cross_store_reconciled",
                task_cid=current["task_cid"],
                attempt_id=attempt_id,
                session_id=self._owner_session_id,
                recorded_at_ms=committed_at,
                body=recovery_event_body,
            )
        if (
            not reconciled
            and phase == "failed"
            and body.get("portal_retryable_failure") is True
        ):
            if set(body) != {"reason", "portal_retryable_failure"} or not str(
                body.get("reason") or ""
            ):
                raise EAAEFBorrowedTransactionError(
                    "retryable Portal failure body is not exact"
                )
            task = self._task_record(owned, current["task_cid"])
            if task is None or task["status"] not in {"in_progress", "running", "claimed"}:
                raise EAAEFBorrowedTransactionConflict(
                    "retryable Portal failure task is not active"
                )
            task_revision = task["revision"] + 1
            task_body = {
                **task["body"],
                "last_retryable_failure": {
                    "attempt_id": current["attempt_id"],
                    "reason": str(body["reason"]),
                },
            }
            task_change = owned.execute(
                "UPDATE tasks SET status='ready', revision=?, updated_at=?, body_json=? "
                "WHERE task_cid=? AND revision=? RETURNING task_cid",
                [
                    task_revision,
                    _iso(committed_at),
                    _json(task_body, "retryable task body"),
                    current["task_cid"],
                    task["revision"],
                ],
            ).fetchone()
            if task_change is None:
                raise EAAEFBorrowedTransactionConflict(
                    "retryable Portal failure lost its task CAS"
                )
            owned.execute(
                "INSERT INTO task_revisions(task_cid, revision, status, body_json, "
                "recorded_at) VALUES (?, ?, 'ready', ?, ?)",
                [
                    current["task_cid"],
                    task_revision,
                    _json(task_body, "retryable task revision"),
                    _iso(committed_at),
                ],
            )
            owned.execute(
                "UPDATE leases SET state='expired', release_reason='retryable_portal_failure', "
                "revision=revision+1 WHERE task_cid=? AND claim_id=? "
                "AND attempt_id=? AND state='accepted'",
                [current["task_cid"], current["claim_id"], current["attempt_id"]],
            )
            owned.execute(
                "UPDATE task_claims SET state='expired', released_at=?, released_at_ms=?, "
                "revision=revision+1 WHERE claim_id=? AND state='accepted'",
                [_iso(committed_at), committed_at, current["claim_id"]],
            )
            self._event(
                owned,
                event_id=_id(
                    "event",
                    {
                        "type": "attempt_phase_committed",
                        "attempt_id": attempt_id,
                        "revision": revision,
                    },
                ),
                event_type="attempt_phase_committed",
                task_cid=current["task_cid"],
                attempt_id=attempt_id,
                session_id=current["owner_session_id"],
                recorded_at_ms=committed_at,
                body={"phase": phase, "revision": revision, **body},
            )
        return self._attempt(owned, attempt_id)

    def _phase_history(self, owned: Any, attempt_id: str) -> list[dict[str, Any]]:
        rows = owned.execute(
            "SELECT phase_name, committed_at_ms, fencing_token, fence_epoch, revision, body_json "
            "FROM attempt_phases WHERE attempt_id=? ORDER BY committed_at_ms, phase_name LIMIT ?",
            [_identifier(attempt_id, "attempt_id"), MAX_LIST_ITEMS + 1],
        ).fetchall()
        if len(rows) > MAX_LIST_ITEMS:
            raise EAAEFBorrowedTransactionError(
                "attempt phase history exceeds its closed bound"
            )
        return [
            {"phase": str(row[0]), "committed_at_ms": int(row[1]), "fencing_token": int(row[2]), "fence_epoch": int(row[3]), "revision": int(row[4]), "body": _decode(row[5], "attempt phase body")}
            for row in rows
        ]

    def _container_dispatch_claim(
        self,
        value: Any,
        *,
        attempt: Mapping[str, Any],
    ) -> dict[str, Any]:
        claim = _exact(
            _object(value, "container dispatch claim"),
            _CONTAINER_DISPATCH_CLAIM_FIELDS,
            "container dispatch claim",
        )
        body = {key: item for key, item in claim.items() if key != "claim_cid"}
        cid_fields = (
            "packet_cid",
            "plan_revision_cid",
            "semantic_state_root",
            "worktree_id",
            "planned_container_id",
            "container_profile_cid",
            "image_digest",
            "network_authorization_cid",
            "effect_scope_cid",
            "gateway_binding_cid",
            "model_route_cid",
            "claim_cid",
        )
        if (
            claim.get("schema")
            != "ipfs_accelerate_py/agent-supervisor/external-agent-container-dispatch-claim@1"
            or claim.get("interface")
            != "ExternalAgentContainerWorkerDispatcher@1"
            or claim.get("task_cid") != attempt["task_cid"]
            or claim.get("attempt_id") != attempt["attempt_id"]
            or claim.get("attempt_number") != attempt["attempt_number"]
            or claim.get("lease_id") != attempt["lease_id"]
            or claim.get("fencing_token") != attempt["fencing_token"]
            or claim.get("fence_epoch") != attempt["fence_epoch"]
            or claim.get("gateway_binding_cid") != self._gateway_binding_cid
            or claim.get("provider") not in {"codex", "grok"}
            or claim.get("worker_principal_did")
            != self._command_principal_did
            or not str(claim.get("provider_principal_did") or "").startswith(
                "did:key:z"
            )
            or claim.get("provider_principal_did")
            in {self._owner_principal_did, self._command_principal_did}
            or any(
                _SHA256_CID.fullmatch(str(claim.get(name) or "")) is None
                for name in cid_fields
            )
            or (
                _GIT_COMMIT.fullmatch(str(claim.get("repository_tree") or ""))
                is None
                and _SHA256_CID.fullmatch(
                    str(claim.get("repository_tree") or "")
                )
                is None
            )
            or claim["claim_cid"] != _sha(body)
        ):
            raise EAAEFBorrowedTransactionConflict(
                "container dispatch claim is not exact or fenced"
            )
        for name in ("task_id", "idempotency_key"):
            _identifier(claim[name], f"container dispatch {name}")
        return claim

    @staticmethod
    def _container_dispatch_reservation(
        *,
        claim_cid: str,
        reservation_id: str,
        outcome: str,
        accepted_result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = {
            "schema": "ipfs_accelerate_py/agent-supervisor/external-agent-container-dispatch-reservation@1",
            "claim_cid": claim_cid,
            "reservation_id": reservation_id,
            "outcome": outcome,
            "reason_codes": [],
            "accepted_result": (
                None if accepted_result is None else dict(accepted_result)
            ),
        }
        return {**body, "receipt_cid": _sha(body)}

    def _container_accepted_result(
        self,
        value: Any,
        *,
        attempt: Mapping[str, Any],
        claim: Mapping[str, Any],
        reservation_id: str,
    ) -> dict[str, Any]:
        result = _exact(
            _object(value, "container accepted result"),
            _CONTAINER_ACCEPTED_RESULT_FIELDS,
            "container accepted result",
        )
        body = {key: item for key, item in result.items() if key != "receipt_id"}
        cid_fields = (
            "packet_cid",
            "claim_cid",
            "reservation_id",
            "proposal_receipt_cid",
            "verification_receipt_cid",
            "patch_artifact_cid",
            "receipt_id",
        )
        verifier = str(result.get("independent_verifier_principal_did") or "")
        if (
            result.get("schema")
            != "ipfs_accelerate_py/agent-supervisor/external-agent-container-accepted-result@1"
            or result.get("interface")
            != "ExternalAgentContainerWorkerDispatcher@1"
            or result.get("status") != "succeeded"
            or result.get("accepted") is not True
            or result.get("task_result_accepted") is not False
            or result.get("merge_admitted") is not False
            or result.get("task_cid") != attempt["task_cid"]
            or result.get("attempt_id") != attempt["attempt_id"]
            or result.get("packet_cid") != claim["packet_cid"]
            or result.get("claim_cid") != claim["claim_cid"]
            or result.get("reservation_id") != reservation_id
            or result.get("worker_principal_did") != claim["worker_principal_did"]
            or not verifier.startswith("did:key:z")
            or verifier
            in {
                claim["worker_principal_did"],
                claim["provider_principal_did"],
            }
            or any(
                _SHA256_CID.fullmatch(str(result.get(name) or "")) is None
                for name in cid_fields
            )
            or any(
                not isinstance(result.get(name), list)
                or len(result[name]) > MAX_LIST_ITEMS
                or any(
                    _SHA256_CID.fullmatch(str(item or "")) is None
                    for item in result[name]
                )
                for name in (
                    "artifact_cids",
                    "test_receipt_cids",
                    "proof_receipt_cids",
                )
            )
            or result["receipt_id"] != _sha(body)
        ):
            raise EAAEFBorrowedTransactionConflict(
                "container accepted result is not exact or independently bound"
            )
        return result

    def _container_dispatch_idempotent(
        self,
        owned: Any,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        reserve_fields = {
            "kind",
            "record_id",
            "attempt_id",
            "task_cid",
            "operation_key",
            "idempotency_key",
            "owner_session_id",
            "recorded_at_ms",
            "fencing_token",
            "fence_epoch",
            "claim",
        }
        fields = reserve_fields if operation.endswith("reserve") else reserve_fields | {
            "reservation_id",
            "result",
        }
        args = _exact(arguments, fields, operation)
        if args["kind"] != EAAEF_CONTAINER_DISPATCH_OPERATION_KIND:
            raise EAAEFBorrowedTransactionError(
                "container dispatch operation kind differs"
            )
        attempt_id = _identifier(args["attempt_id"], "attempt_id")
        attempt = self._attempt(owned, attempt_id)
        if attempt is None or attempt["status"] != "running":
            raise EAAEFBorrowedTransactionConflict(
                "container dispatch attempt is absent or not running"
            )
        claim = self._container_dispatch_claim(args["claim"], attempt=attempt)
        claim_cid = str(claim["claim_cid"])
        key = _identifier(args["idempotency_key"], "idempotency_key")
        for name in (
            "task_cid",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
        ):
            if args[name] != attempt[name]:
                raise EAAEFBorrowedTransactionConflict(
                    "container dispatch authority differs"
                )
        if (
            args["record_id"] != claim_cid
            or args["operation_key"] != claim_cid
            or key != claim["idempotency_key"]
        ):
            raise EAAEFBorrowedTransactionConflict(
                "container dispatch reservation identity differs"
            )
        _trusted_now(args["recorded_at_ms"])
        reservation_id = _sha({"reservation": claim_cid})
        row = owned.execute(
            "SELECT effect_id, state, result_json, body_json, operation_key, "
            "task_cid, owner_session_id, fencing_token, fence_epoch "
            "FROM effect_claims WHERE attempt_id=? AND idempotency_key=?",
            [attempt_id, key],
        ).fetchone()
        if operation.endswith("reserve"):
            if row is None:
                now = int(time.time_ns() // 1_000_000)
                stored = {"claim": claim, "reservation_id": reservation_id}
                owned.execute(
                    "INSERT INTO effect_claims(effect_id, task_cid, attempt_id, effect_kind, target_path, "
                    "claimed_at, state, body_json, operation_key, idempotency_key, owner_session_id, "
                    "recorded_at_ms, result_json, fencing_token, fence_epoch) "
                    "VALUES (?, ?, ?, ?, '', ?, 'reserved', ?, ?, ?, ?, ?, '{}', ?, ?)",
                    [
                        claim_cid,
                        attempt["task_cid"],
                        attempt_id,
                        EAAEF_CONTAINER_DISPATCH_OPERATION_KIND,
                        _iso(now),
                        _json(stored, "container dispatch reservation"),
                        claim_cid,
                        key,
                        attempt["owner_session_id"],
                        now,
                        attempt["fencing_token"],
                        attempt["fence_epoch"],
                    ],
                )
                return self._container_dispatch_reservation(
                    claim_cid=claim_cid,
                    reservation_id=reservation_id,
                    outcome="reserved_new",
                )
            stored = _decode(row[3], "container dispatch reservation")
            if (
                str(row[0]) != claim_cid
                or str(row[4]) != claim_cid
                or str(row[5]) != attempt["task_cid"]
                or str(row[6]) != attempt["owner_session_id"]
                or int(row[7]) != attempt["fencing_token"]
                or int(row[8]) != attempt["fence_epoch"]
                or stored != {"claim": claim, "reservation_id": reservation_id}
                or str(row[1]) not in {"reserved", "committed"}
            ):
                raise EAAEFBorrowedTransactionConflict(
                    "container dispatch durable reservation differs"
                )
            accepted = (
                _decode(row[2], "container accepted result")
                if str(row[1]) == "committed"
                else None
            )
            return self._container_dispatch_reservation(
                claim_cid=claim_cid,
                reservation_id=reservation_id,
                outcome=(
                    "accepted_replay"
                    if accepted is not None
                    else "in_flight_ambiguous"
                ),
                accepted_result=accepted,
            )
        if row is None or str(row[1]) not in {"reserved", "committed"}:
            raise EAAEFBorrowedTransactionConflict(
                "container dispatch commit has no durable reservation"
            )
        if args["reservation_id"] != reservation_id:
            raise EAAEFBorrowedTransactionConflict(
                "container dispatch reservation_id differs"
            )
        result = self._container_accepted_result(
            args["result"],
            attempt=attempt,
            claim=claim,
            reservation_id=reservation_id,
        )
        if str(row[1]) == "committed":
            if _decode(row[2], "container accepted result") != result:
                raise EAAEFBorrowedTransactionConflict(
                    "container dispatch accepted replay diverged"
                )
            return result
        owned.execute(
            "UPDATE effect_claims SET state='committed', result_json=? "
            "WHERE attempt_id=? AND idempotency_key=? AND state='reserved'",
            [_json(result, "container accepted result"), attempt_id, key],
        )
        return result

    def _idempotent(
        self, owned: Any, operation: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        if (
            operation.startswith("effect.")
            and _object(arguments, operation).get("kind")
            == EAAEF_CONTAINER_DISPATCH_OPERATION_KIND
        ):
            return self._container_dispatch_idempotent(
                owned, operation, arguments
            )
        kind = "provider" if operation.startswith("provider.") else "effect"
        table = "provider_invocations" if kind == "provider" else "effect_claims"
        id_column = "invocation_id" if kind == "provider" else "effect_id"
        status_column = "status" if kind == "provider" else "state"
        if operation.endswith("reserve"):
            args = _exact(arguments, {"kind", "attempt_id", "idempotency_key"}, operation)
            if str(args["kind"]) != kind:
                raise EAAEFBorrowedTransactionError("reservation kind differs")
            attempt_id = _identifier(args["attempt_id"], "attempt_id")
            key = _identifier(args["idempotency_key"], "idempotency_key")
            row = owned.execute(
                f"SELECT {id_column}, result_json, {status_column}, task_cid FROM {table} "
                "WHERE attempt_id=? AND idempotency_key=?",
                [attempt_id, key],
            ).fetchone()
            if row is not None:
                state = str(row[2])
                if state not in {"reserved", "committed"}:
                    raise EAAEFBorrowedTransactionConflict(
                        f"{kind} reservation state is invalid"
                    )
                return {
                    "schema": EAAEF_IDEMPOTENT_RESERVATION_SCHEMA,
                    "kind": kind,
                    "state": (
                        "committed"
                        if state == "committed"
                        else "existing_reserved_ambiguous"
                    ),
                    "record_id": str(row[0]),
                    "attempt_id": attempt_id,
                    "task_cid": str(row[3]),
                    "idempotency_key": key,
                    "result": (
                        _decode(row[1], f"{kind} result")
                        if state == "committed"
                        else {}
                    ),
                }
            attempt = self._attempt(owned, attempt_id)
            if attempt is None or attempt["status"] != "running":
                raise EAAEFBorrowedTransactionConflict(f"{kind} attempt is not running")
            record_id = eaaef_reservation_id(
                kind=kind,
                attempt_id=attempt_id,
                idempotency_key=key,
            )
            now = int(time.time_ns() // 1_000_000)
            if kind == "provider":
                owned.execute(
                    "INSERT INTO provider_invocations(invocation_id, task_cid, attempt_id, provider_id, "
                    "started_at, finished_at, status, input_digest, output_digest, body_json, "
                    "idempotency_key, owner_session_id, recorded_at_ms, result_json, fencing_token, fence_epoch) "
                    "VALUES (?, ?, ?, 'reserved', ?, NULL, 'reserved', ?, '', '{}', ?, ?, ?, '{}', ?, ?)",
                    [record_id, attempt["task_cid"], attempt_id, _iso(now), _sha({"attempt_id": attempt_id, "key": key}), key, attempt["owner_session_id"], now, attempt["fencing_token"], attempt["fence_epoch"]],
                )
            else:
                owned.execute(
                    "INSERT INTO effect_claims(effect_id, task_cid, attempt_id, effect_kind, target_path, "
                    "claimed_at, state, body_json, operation_key, idempotency_key, owner_session_id, "
                    "recorded_at_ms, result_json, fencing_token, fence_epoch) "
                    "VALUES (?, ?, ?, 'reserved', '', ?, 'reserved', '{}', '', ?, ?, ?, '{}', ?, ?)",
                    [record_id, attempt["task_cid"], attempt_id, _iso(now), key, attempt["owner_session_id"], now, attempt["fencing_token"], attempt["fence_epoch"]],
                )
            return {
                "schema": EAAEF_IDEMPOTENT_RESERVATION_SCHEMA,
                "kind": kind,
                "state": "newly_reserved",
                "record_id": record_id,
                "attempt_id": attempt_id,
                "task_cid": attempt["task_cid"],
                "idempotency_key": key,
                "result": {},
            }
        args = _exact(
            arguments,
            {"kind", "record_id", "attempt_id", "task_cid", "operation_key", "idempotency_key", "owner_session_id", "recorded_at_ms", "result", "fencing_token", "fence_epoch"},
            operation,
        )
        if str(args["kind"]) != kind:
            raise EAAEFBorrowedTransactionError("commit kind differs")
        attempt_id = _identifier(args["attempt_id"], "attempt_id")
        key = _identifier(args["idempotency_key"], "idempotency_key")
        operation_key = str(args["operation_key"] or "")
        if kind == "provider":
            if operation_key:
                raise EAAEFBorrowedTransactionError(
                    "provider commit operation_key must be empty"
                )
        else:
            operation_key = _identifier(operation_key, "effect operation_key")
        attempt = self._attempt(owned, attempt_id)
        if attempt is None or attempt["status"] != "running":
            raise EAAEFBorrowedTransactionConflict(
                f"{kind} attempt is absent or not running"
            )
        for name in ("task_cid", "owner_session_id", "fencing_token", "fence_epoch"):
            if attempt[name] != args[name]:
                raise EAAEFBorrowedTransactionConflict(f"{kind} commit fence differs")
        result = _object(args["result"], f"{kind} result")
        operation_projection = (
            "'' AS operation_key" if kind == "provider" else "operation_key"
        )
        row = owned.execute(
            f"SELECT {id_column}, result_json, {status_column}, task_cid, "
            f"owner_session_id, fencing_token, fence_epoch, {operation_projection} "
            f"FROM {table} "
            "WHERE attempt_id=? AND idempotency_key=?",
            [attempt_id, key],
        ).fetchone()
        if row is None:
            raise EAAEFBorrowedTransactionConflict(f"{kind} commit has no reservation")
        if str(row[0]) != _identifier(args["record_id"], "record_id"):
            raise EAAEFBorrowedTransactionConflict(
                f"{kind} commit record_id differs from its reservation"
            )
        if (
            str(row[3]) != attempt["task_cid"]
            or str(row[4]) != attempt["owner_session_id"]
            or int(row[5]) != attempt["fencing_token"]
            or int(row[6]) != attempt["fence_epoch"]
            or (kind == "effect" and str(row[7] or "") not in {"", operation_key})
        ):
            raise EAAEFBorrowedTransactionConflict(
                f"{kind} reservation authority differs"
            )
        if str(row[2]) not in {"reserved", "committed"}:
            raise EAAEFBorrowedTransactionConflict(
                f"{kind} reservation state is not committable"
            )
        if str(row[2]) == "committed":
            prior = _decode(row[1], f"{kind} result")
            if prior != result:
                raise EAAEFBorrowedTransactionConflict(f"{kind} replay diverged")
            return prior
        now = _trusted_now(args["recorded_at_ms"])
        if kind == "provider":
            owned.execute(
                "UPDATE provider_invocations SET finished_at=?, status='committed', output_digest=?, "
                "result_json=?, body_json=? WHERE attempt_id=? AND idempotency_key=? AND status='reserved'",
                [_iso(now), _sha(result), _json(result, "provider result"), _json({"record_id": args["record_id"]}, "provider body"), attempt_id, key],
            )
        else:
            owned.execute(
                "UPDATE effect_claims SET state='committed', operation_key=?, result_json=?, body_json=? "
                "WHERE attempt_id=? AND idempotency_key=? AND state='reserved'",
                [operation_key, _json(result, "effect result"), _json({"record_id": args["record_id"]}, "effect body"), attempt_id, key],
            )
        return result

    def apply(
        self,
        *,
        operation: str,
        arguments: Mapping[str, Any],
        transaction: Any,
        command: Any,
        lease: Mapping[str, Any],
    ) -> Any:
        """Apply one exact operation without owning transaction lifecycle."""

        owned = self._active(transaction)
        self._verify_profile(owned)
        name = str(operation or "")
        idempotency_key = _identifier(
            getattr(command, "idempotency_key", ""), "command idempotency_key"
        )
        fence_epoch = _positive(getattr(command, "fence_epoch", 0), "command fence_epoch")
        _identifier(lease.get("principal_did"), "authorized principal_did")
        daemon_lane_binding: dict[str, Any] | None = None
        dead_lane_recovery: dict[str, Any] | None = None

        if name in EAAEF_BOARD_SCOPED_OPERATIONS:
            supplied_now = _object(arguments, f"{name} arguments").get("now_ms")
            now_hint = (
                _trusted_now(supplied_now)
                if supplied_now is not None
                else int(time.time_ns() // 1_000_000)
            )
            self._assert_board_lease(
                owned,
                command=command,
                lease=lease,
                now_ms=now_hint,
            )
            if name in _LANE_BOUND_BOARD_OPERATIONS:
                board_arguments = _object(arguments, f"{name} arguments")
                if (
                    name == "execution.list_running_attempts"
                    and set(board_arguments) == {"recovery_authority"}
                ):
                    dead_lane_recovery = _exact(
                        _object(
                            board_arguments.pop("recovery_authority"),
                            "dead lane recovery authority",
                        ),
                        _DEAD_LANE_RECOVERY_AUTHORITY_FIELDS,
                        "dead lane recovery authority",
                    )
                else:
                    daemon_lane_binding = self._daemon_lane_binding(
                        board_arguments.pop("daemon_lane_binding", None)
                    )
                    if name != "execution.bind_daemon":
                        self._require_bound_lane(owned, daemon_lane_binding)
                arguments = board_arguments
        else:
            raw_arguments = _object(arguments, f"{name} arguments")
            unscoped_event = (
                name == "execution.record_event"
                and not str(raw_arguments.get("task_cid") or "")
                and not str(raw_arguments.get("attempt_id") or "")
            )
            if not unscoped_event:
                arguments = self._unwrap_task_operation_authority(
                    owned,
                    operation=name,
                    arguments=raw_arguments,
                    command=command,
                    authorized_lease=lease,
                )

        if name == "task.cas_status":
            args = _object(arguments, "task.cas_status arguments")
            self._require_task_scope(command, str(args.get("task_cid") or ""))
            value = self._task_cas(owned, args, authorized_lease=lease)
        elif name in {"task.record_validation", "validation.record"}:
            args = _object(arguments, f"{name} arguments")
            task_cid = str(args.get("task_cid") or "")
            self._require_task_scope(command, task_cid)
            claim = self._protect_current_task(
                owned,
                task_cid=task_cid,
                authorized_lease=lease,
            )
            supplied_attempt_id = str(args.get("attempt_id") or "")
            if supplied_attempt_id and supplied_attempt_id != claim["attempt_id"]:
                raise EAAEFBorrowedTransactionConflict(
                    "validation attempt is not the current task claim"
                )
            value = self._replay_committed_validation(
                owned, args, attempt_default=claim["attempt_id"]
            )
        elif name == "coordination.register_task":
            value = self._register_task(owned, arguments)
        elif name == "coordination.claim_ready":
            if daemon_lane_binding is None:
                raise EAAEFBorrowedTransactionError(
                    "claim selection requires an exact bound daemon lane"
                )
            claim_arguments = _object(arguments, name)
            if (
                claim_arguments.get("owner_session_id")
                != daemon_lane_binding["lane_session_id"]
            ):
                raise EAAEFBorrowedTransactionConflict(
                    "claim owner_session_id differs from the bound daemon lane"
                )
            value = self._claim_ready(
                owned,
                claim_arguments,
                lease=lease,
                idempotency_key=idempotency_key,
                fence_epoch=fence_epoch,
            )
        elif name == "coordination.get_claim":
            args = _exact(arguments, {"claim_id"}, name)
            scope = self._scope(command)
            row = owned.execute(
                "SELECT task_claims.claim_id FROM task_claims JOIN leases "
                "ON leases.task_cid=task_claims.task_cid "
                "AND leases.claim_id=task_claims.claim_id "
                "WHERE task_claims.claim_id=? AND task_claims.task_cid=? "
                "AND leases.lease_kind='task' AND leases.scope_id=? "
                "AND leases.mode='exclusive'",
                [_identifier(args["claim_id"], "claim_id"), scope, scope],
            ).fetchone()
            value = None if row is None else self._claim_record(owned, str(row[0]))
            if value is not None:
                self._protect_current_task(
                    owned,
                    task_cid=scope,
                    authorized_lease=lease,
                    allow_logically_completed=True,
                )
        elif name == "coordination.protect_claim":
            identity = self._claim_identity(_object(arguments, name)["claim"])
            self._require_task_scope(command, identity["task_cid"])
            value = self._protect_operation(
                owned, arguments, authorized_lease=lease
            )
        elif name == "coordination.renew_lease":
            identity = self._claim_identity(_object(arguments, name)["lease"])
            self._require_task_scope(command, identity["task_cid"])
            value = self._renew(owned, arguments, authorized_lease=lease)
        elif name == "coordination.prepare_completion":
            identity = self._claim_identity(_object(arguments, name)["claim"])
            self._require_task_scope(command, identity["task_cid"])
            value = self._prepare_completion(
                owned, arguments, authorized_lease=lease
            )
        elif name == "coordination.get_prepared_completion":
            args = _exact(arguments, {"task_cid"}, name)
            task_cid = _identifier(args["task_cid"], "task_cid")
            self._require_task_scope(command, task_cid)
            self._protect_current_task(
                owned,
                task_cid=task_cid,
                authorized_lease=lease,
                allow_logically_completed=True,
            )
            value = self._barrier(owned, task_cid)
        elif name == "coordination.complete_claim":
            identity = self._claim_identity(_object(arguments, name)["claim"])
            self._require_task_scope(command, identity["task_cid"])
            value = self._complete_claim(
                owned, arguments, authorized_lease=lease
            )
        elif name == "coordination.settle_claim":
            identity = self._claim_identity(_object(arguments, name)["claim"])
            self._require_task_scope(command, identity["task_cid"])
            value = self._settle(owned, arguments, authorized_lease=lease)
        elif name == "coordination.list_unsettled_completions":
            args = _exact(arguments, {"limit", "now_ms"}, name)
            limit = _positive(args["limit"], "limit", maximum=MAX_LIST_ITEMS)
            now = _trusted_now(args["now_ms"])
            tasks = owned.execute(
                "SELECT barriers.task_cid FROM eaaef_completion_barriers AS barriers "
                "LEFT JOIN task_attempts AS attempts "
                "ON attempts.attempt_id=barriers.attempt_id "
                "LEFT JOIN task_claims AS claims "
                "ON claims.claim_id=barriers.claim_id "
                "LEFT JOIN leases ON leases.task_cid=barriers.task_cid "
                "AND leases.claim_id=barriers.claim_id "
                "AND leases.attempt_id=barriers.attempt_id "
                "WHERE barriers.status='prepared' OR "
                "(barriers.status='succeeded' AND (attempts.attempt_id IS NULL "
                "OR attempts.status<>'succeeded' OR attempts.committed_phase<>'complete' "
                "OR claims.claim_id IS NULL OR claims.state<>'released' "
                "OR leases.task_cid IS NULL OR leases.state<>'released')) OR "
                "(barriers.status='aborted' AND (attempts.attempt_id IS NULL "
                "OR attempts.status<>'failed' OR attempts.committed_phase<>'failed' "
                "OR claims.claim_id IS NULL OR claims.state<>'expired' "
                "OR leases.task_cid IS NULL OR leases.state<>'expired')) "
                "ORDER BY barriers.prepared_at_ms, barriers.task_cid LIMIT ?",
                [limit],
            ).fetchall()
            value = [
                self._prepared_recovery_snapshot(
                    owned,
                    task_cid=str(row[0]),
                    observed_at_ms=now,
                )
                for row in tasks
            ]
        elif name in {
            "coordination.reconcile_promoted_completion",
            "coordination.recover_prepared_completion",
            "coordination.abort_prepared_completion",
        }:
            args = _object(arguments, f"{name} arguments")
            self._require_board_scope(command)
            value = self._reconcile(owned, name, args)
        elif name == "coordination.expire_claim":
            identity = self._claim_identity(_object(arguments, name)["claim"])
            self._require_board_scope(command)
            value = self._expire_claim(owned, arguments)
        elif name == "execution.bind_daemon":
            if daemon_lane_binding is None:
                raise EAAEFBorrowedTransactionError(
                    "daemon binding requires an exact signed lane projection"
                )
            args = _exact(arguments, {"metadata"}, name)
            metadata = _exact(
                _object(args["metadata"], "daemon metadata"),
                {
                    "interface",
                    "schema",
                    "authority_mode",
                    "logical_owner_session_id",
                    "process_instance_id",
                    "state_schema_revision",
                    "gateway_binding_cid",
                    "gateway_owner_principal_did",
                    "gateway_owner_generation",
                    "gateway_fence_epoch",
                    "gateway_control_plane_schema_version",
                    "gateway_state_schema_revision",
                },
                "daemon metadata",
            )
            if (
                metadata["interface"] != "DatabaseImplementationDaemon@1"
                or metadata["schema"]
                != "ipfs_accelerate_py/agent-supervisor/database-implementation-daemon@1"
                or metadata["authority_mode"] != "quack"
                or metadata["logical_owner_session_id"]
                != daemon_lane_binding["lane_session_id"]
                or metadata["process_instance_id"]
                != daemon_lane_binding["process_instance_id"]
                or metadata["gateway_owner_principal_did"]
                != self._owner_principal_did
                or _positive(
                    metadata["gateway_owner_generation"],
                    "gateway owner_generation",
                )
                != self._owner_generation
                or _positive(metadata["gateway_fence_epoch"], "gateway fence_epoch")
                != self._fence_epoch
                or metadata["gateway_binding_cid"] != self._gateway_binding_cid
                or metadata["gateway_control_plane_schema_version"]
                != self._control_plane_schema_version
                or metadata["state_schema_revision"] != self._state_schema_revision
                or metadata["gateway_state_schema_revision"]
                != self._state_schema_revision
            ):
                raise EAAEFBorrowedTransactionConflict(
                    "daemon binding differs from the verified capability owner"
                )
            session_id = _identifier(metadata["logical_owner_session_id"], "session_id")
            daemon_id = _identifier(metadata["process_instance_id"], "daemon_id")
            for field in (
                "state_schema_revision",
                "gateway_binding_cid",
                "gateway_control_plane_schema_version",
                "gateway_state_schema_revision",
            ):
                _identifier(metadata[field], field)
            now = int(time.time_ns() // 1_000_000)
            existing = owned.execute(
                "SELECT daemon_id, fence_epoch, status, revision, metadata_json "
                "FROM daemon_sessions WHERE session_id=?",
                [session_id],
            ).fetchone()
            metadata_json = _json(
                {
                    "daemon_metadata": metadata,
                    "lane_binding": daemon_lane_binding,
                },
                "daemon lane metadata",
            )
            if existing is None:
                owned.execute(
                    "INSERT INTO daemon_sessions(session_id, daemon_id, process_birth_id, fence_epoch, "
                    "attached_at, last_heartbeat_at, status, revision, metadata_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'attached', 1, ?)",
                    [session_id, daemon_id, daemon_id, self._fence_epoch, _iso(now), _iso(now), metadata_json],
                )
                replayed = False
            else:
                prior_metadata_json = str(existing[4])
                if prior_metadata_json == metadata_json:
                    replayed = True
                else:
                    raise EAAEFBorrowedTransactionConflict(
                        "daemon lane binding replay diverged; a new process requires "
                        "a fresh signed lane session"
                    )
            value = {"session_id": session_id, "daemon_id": daemon_id, "replayed": replayed}
        elif name == "execution.record_event":
            args = _exact(arguments, {"event_id", "attempt_id", "task_cid", "event_type", "recorded_at_ms", "body"}, name)
            task_cid = str(args["task_cid"] or "")
            attempt_id = str(args["attempt_id"] or "")
            if attempt_id:
                attempt_key = _identifier(attempt_id, "attempt_id")
                scope = self._scope(command)
                attempt = self._attempt(owned, attempt_key)
                if attempt is None or attempt["task_cid"] != scope:
                    raise EAAEFBorrowedTransactionConflict(
                        "event attempt is absent from the exact authorized task scope"
                    )
                if task_cid and task_cid != attempt["task_cid"]:
                    raise EAAEFBorrowedTransactionConflict(
                        "event task and attempt identities differ"
                    )
                task_cid = attempt["task_cid"]
            if task_cid:
                self._require_task_scope(command, task_cid)
                if attempt_id:
                    assert attempt is not None
                    self._protect_current_attempt(
                        owned,
                        attempt=attempt,
                        authorized_lease=lease,
                        allow_logically_completed=True,
                    )
                else:
                    self._protect_current_task(
                        owned,
                        task_cid=task_cid,
                        authorized_lease=lease,
                        allow_logically_completed=True,
                    )
            else:
                self._assert_board_lease(
                    owned,
                    command=command,
                    lease=lease,
                    now_ms=_trusted_now(args["recorded_at_ms"]),
                )
            _trusted_now(args["recorded_at_ms"])
            recorded_at_ms = _positive(args["recorded_at_ms"], "recorded_at_ms")
            value = self._event(
                owned,
                event_id=str(args["event_id"]),
                event_type=_identifier(args["event_type"], "event_type"),
                task_cid=task_cid,
                attempt_id=attempt_id,
                session_id=str(lease.get("principal_did") or ""),
                recorded_at_ms=recorded_at_ms,
                body=_object(args["body"], "event body"),
            )
        elif name == "execution.ensure_attempt":
            ensure_args = _exact(
                arguments, {"attempt", "claimed_phase"}, "execution.ensure_attempt"
            )
            attempt_input, _ = self._attempt_input(
                ensure_args["attempt"], ensure_args["claimed_phase"]
            )
            identity = {
                key: attempt_input[key]
                for key in (
                    "claim_id",
                    "task_cid",
                    "owner_session_id",
                    "fencing_token",
                    "fence_epoch",
                    "attempt_id",
                    "attempt_number",
                    "lease_id",
                )
            }
            self._require_task_scope(command, identity["task_cid"])
            self._protect(
                owned,
                identity,
                now_ms=int(time.time_ns() // 1_000_000),
                authorized_lease=lease,
            )
            value = self._ensure_attempt(owned, arguments)
        elif name == "execution.get_attempt":
            args = _exact(arguments, {"attempt_id"}, name)
            scope = self._scope(command)
            attempt_id = _identifier(args["attempt_id"], "attempt_id")
            row = owned.execute(
                "SELECT task_attempts.attempt_id FROM task_attempts JOIN leases "
                "ON leases.task_cid=task_attempts.task_cid "
                "AND leases.attempt_id=task_attempts.attempt_id "
                "AND leases.claim_id=task_attempts.claim_id "
                "WHERE task_attempts.attempt_id=? AND task_attempts.task_cid=? "
                "AND leases.lease_kind='task' AND leases.scope_id=? "
                "AND leases.mode='exclusive'",
                [attempt_id, scope, scope],
            ).fetchone()
            value = None if row is None else self._attempt(owned, str(row[0]))
            if value is not None:
                self._protect_current_task(
                    owned,
                    task_cid=scope,
                    authorized_lease=lease,
                    allow_logically_completed=True,
                )
        elif name == "execution.list_running_attempts":
            if dead_lane_recovery is None:
                if daemon_lane_binding is None:
                    raise EAAEFBorrowedTransactionError(
                        "running-attempt listing requires an exact bound daemon lane"
                    )
                args = _exact(arguments, {"owner_session_id"}, name)
                owner = _identifier(args["owner_session_id"], "owner_session_id")
                if owner != daemon_lane_binding["lane_session_id"]:
                    raise EAAEFBorrowedTransactionConflict(
                        "running-attempt owner differs from the bound daemon lane"
                    )
                owners = [owner]
                query = (
                    "SELECT attempt_id FROM task_attempts WHERE owner_session_id=? "
                    "AND status='running' ORDER BY started_at_ms, attempt_id LIMIT ?"
                )
                parameters: list[Any] = [owner, MAX_LIST_ITEMS + 1]
            else:
                if (
                    dead_lane_recovery["schema"]
                    != EAAEF_DEAD_LANE_RECOVERY_AUTHORITY_SCHEMA
                    or dead_lane_recovery["purpose"] != "expired_lane_retirement"
                ):
                    raise EAAEFBorrowedTransactionError(
                        "dead lane recovery authority is unsupported"
                    )
                limit = _positive(
                    dead_lane_recovery["limit"], "dead lane limit", maximum=MAX_LIST_ITEMS
                )
                now = _trusted_now(dead_lane_recovery["now_ms"])
                raw_bindings = dead_lane_recovery["lane_bindings"]
                if (
                    not isinstance(raw_bindings, list)
                    or not 1 <= len(raw_bindings) <= 5
                ):
                    raise EAAEFBorrowedTransactionError(
                        "dead lane recovery is outside the admitted five-lane frontier"
                    )
                lane_bindings = [
                    self._daemon_lane_binding(item) for item in raw_bindings
                ]
                owners = [item["lane_session_id"] for item in lane_bindings]
                if len(set(owners)) != len(owners):
                    raise EAAEFBorrowedTransactionError(
                        "dead lane recovery contains duplicate lane sessions"
                    )
                for binding in lane_bindings:
                    self._require_bound_lane(owned, binding)
                placeholders = ",".join("?" for _ in owners)
                query = (
                    "SELECT attempts.attempt_id FROM task_attempts AS attempts "
                    "JOIN task_claims AS claims ON claims.claim_id=attempts.claim_id "
                    "AND claims.task_cid=attempts.task_cid "
                    "JOIN leases ON leases.claim_id=attempts.claim_id "
                    "AND leases.attempt_id=attempts.attempt_id "
                    "AND leases.task_cid=attempts.task_cid "
                    f"WHERE attempts.owner_session_id IN ({placeholders}) "
                    "AND attempts.status='running' "
                    "AND claims.state IN ('accepted', 'expired') "
                    "AND leases.state IN ('accepted', 'expired') "
                    "AND claims.expires_at_ms<=? AND leases.expires_at_ms<=? "
                    "AND leases.lease_kind='task' "
                    "AND leases.scope_id=attempts.task_cid AND leases.mode='exclusive' "
                    "ORDER BY attempts.started_at_ms, attempts.attempt_id LIMIT ?"
                )
                parameters = [*owners, now, now, limit + 1]
            rows = owned.execute(
                query,
                parameters,
            ).fetchall()
            result_limit = (
                MAX_LIST_ITEMS
                if dead_lane_recovery is None
                else int(dead_lane_recovery["limit"])
            )
            if len(rows) > result_limit:
                raise EAAEFBorrowedTransactionError(
                    "running attempt result exceeds its closed bound"
                )
            observed_at_ms = int(time.time_ns() // 1_000_000)
            value = []
            for row in rows:
                attempt = self._attempt(owned, str(row[0]))
                if attempt is None:  # pragma: no cover - owner txn is stable.
                    continue
                value.append(
                    {
                        **attempt,
                        "eaaef_recovery_snapshot": self._running_recovery_snapshot(
                            owned,
                            attempt=attempt,
                            observed_at_ms=observed_at_ms,
                        ),
                    }
                )
        elif name == "execution.commit_phase":
            attempt = self._attempt(owned, str(_object(arguments, name).get("attempt_id") or ""))
            if attempt is not None:
                self._require_task_scope(command, attempt["task_cid"])
                self._protect_current_attempt(
                    owned,
                    attempt=attempt,
                    authorized_lease=lease,
                    allow_logically_completed=(
                        str(_object(arguments, name).get("committed_phase") or "")
                        == "complete"
                    ),
                )
            value = self._commit_phase(owned, arguments, reconciled=False)
        elif name == "execution.commit_reconciled_attempt":
            attempt = self._attempt(owned, str(_object(arguments, name).get("attempt_id") or ""))
            if attempt is not None:
                self._require_board_scope(command)
            value = self._commit_phase(owned, arguments, reconciled=True)
        elif name == "execution.phase_history":
            args = _exact(arguments, {"attempt_id"}, name)
            scope = self._scope(command)
            attempt_id = _identifier(args["attempt_id"], "attempt_id")
            scoped = owned.execute(
                "SELECT task_attempts.attempt_id FROM task_attempts JOIN leases "
                "ON leases.task_cid=task_attempts.task_cid "
                "AND leases.attempt_id=task_attempts.attempt_id "
                "AND leases.claim_id=task_attempts.claim_id "
                "WHERE task_attempts.attempt_id=? AND task_attempts.task_cid=? "
                "AND leases.lease_kind='task' AND leases.scope_id=? "
                "AND leases.mode='exclusive'",
                [attempt_id, scope, scope],
            ).fetchone()
            attempt = None if scoped is None else self._attempt(owned, attempt_id)
            if attempt is not None:
                self._protect_current_task(
                    owned,
                    task_cid=scope,
                    authorized_lease=lease,
                    allow_logically_completed=True,
                )
            value = [] if attempt is None else self._phase_history(owned, attempt_id)
        elif name in {"provider.reserve", "provider.commit", "effect.reserve", "effect.commit"}:
            attempt_id = str(_object(arguments, name).get("attempt_id") or "")
            attempt = self._attempt(owned, attempt_id)
            if attempt is not None:
                self._require_task_scope(command, attempt["task_cid"])
                self._protect_current_attempt(
                    owned,
                    attempt=attempt,
                    authorized_lease=lease,
                )
            value = self._idempotent(owned, name, arguments)
        else:
            raise EAAEFBorrowedTransactionError(
                "operation is outside the implemented 29-operation vocabulary"
            )
        _json(value, f"{name} result")
        return value

    def evidence(self) -> Mapping[str, Any]:
        return _adapter_source_evidence(
            board_namespace=self._board_namespace,
            shard_id=self._shard_id,
        )


class EAAEFBootstrapBorrowedTransactionOperationHandler:
    """Closed 31-operation EAAEF handler composed from reviewed built-ins.

    ``task.ready`` and ``task.get`` retain their canonical v1 implementation;
    the other 29 operations use :class:`EAAEFBorrowedTransactionAdapter`.
    Construction accepts only the immutable board/shard binding.  Signed
    production admission remains the command fabric's responsibility.
    """

    INTERFACE: ClassVar[str] = EAAEF_BORROWED_TRANSACTION_HANDLER_INTERFACE
    SCHEMA: ClassVar[str] = EAAEF_BORROWED_TRANSACTION_HANDLER_SCHEMA
    QUALIFICATION_STATUS: ClassVar[str] = (
        EAAEF_BORROWED_TRANSACTION_QUALIFICATION_STATUS
    )

    __slots__ = ("_adapter", "_canonical")

    def __init__(
        self,
        *,
        board_namespace: str,
        shard_id: str,
        owner_principal_did: str,
        command_principal_did: str,
        owner_session_id: str,
        owner_generation: int,
        fence_epoch: int,
        gateway_binding_cid: str,
        control_plane_schema_version: str,
        state_schema_revision: str,
    ) -> None:
        from .quack_daemon_gateway import (
            QuackDaemonCanonicalOwnerOperationHandler,
        )

        self._adapter = EAAEFBorrowedTransactionAdapter(
            board_namespace=board_namespace,
            shard_id=shard_id,
            owner_principal_did=owner_principal_did,
            command_principal_did=command_principal_did,
            owner_session_id=owner_session_id,
            owner_generation=owner_generation,
            fence_epoch=fence_epoch,
            gateway_binding_cid=gateway_binding_cid,
            control_plane_schema_version=control_plane_schema_version,
            state_schema_revision=state_schema_revision,
        )
        self._canonical = QuackDaemonCanonicalOwnerOperationHandler()

    @property
    def board_scope(self) -> str:
        return self._adapter.board_scope

    @staticmethod
    def require_operation(operation: str) -> None:
        name = str(operation or "")
        if name not in EAAEF_BOOTSTRAP_DAEMON_OPERATIONS:
            raise EAAEFBorrowedTransactionError(
                "operation is outside the exact EAAEF 31-operation vocabulary"
            )

    def apply_authorized_daemon_operation(
        self,
        *,
        operation: str,
        arguments: Mapping[str, Any],
        transaction: Any,
        command: Any,
        lease: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Apply one operation without widening the generic daemon protocol."""

        name = str(operation or "")
        self.require_operation(name)
        from .quack_daemon_gateway import quack_daemon_operation_command_vocabulary

        expected_kind = quack_daemon_operation_command_vocabulary()[name]
        observed_kind = getattr(getattr(command, "command_kind", None), "value", None)
        if observed_kind != expected_kind:
            raise EAAEFBorrowedTransactionError(
                "EAAEF operation command kind changed after admission"
            )
        if name in {"task.ready", "task.get"}:
            authorized_arguments = self._adapter.authorize_canonical_read(
                operation=name,
                arguments=arguments,
                transaction=transaction,
                command=command,
                lease=lease,
            )
            result = self._canonical.apply_authorized_daemon_operation(
                operation=name,
                arguments=authorized_arguments,
                transaction=transaction,
                command=command,
                lease=lease,
            )
            return MappingProxyType(dict(result))
        if name not in EAAEF_BOOTSTRAP_DAEMON_MISSING_OPERATIONS:
            raise EAAEFBorrowedTransactionError(
                "EAAEF handler composition registry diverged"
            )
        value = self._adapter.apply(
            operation=name,
            arguments=arguments,
            transaction=transaction,
            command=command,
            lease=lease,
        )
        canonical_json_bytes(value)
        return MappingProxyType({"value": value})

    def evidence(self) -> Mapping[str, Any]:
        return eaaef_bootstrap_handler_source_evidence(
            board_namespace=self._adapter._board_namespace,
            shard_id=self._adapter._shard_id,
        )


__all__ = (
    "EAAEF_BORROWED_TRANSACTION_INTERFACE",
    "EAAEF_BORROWED_TRANSACTION_HANDLER_INTERFACE",
    "EAAEF_BORROWED_TRANSACTION_HANDLER_SCHEMA",
    "EAAEF_BORROWED_TRANSACTION_QUALIFICATION_STATUS",
    "EAAEF_BORROWED_TRANSACTION_SCHEMA",
    "EAAEF_CONTAINER_VALIDATION_EVIDENCE_SCHEMA",
    "EAAEF_CONTAINER_DISPATCH_OPERATION_KIND",
    "EAAEF_DAEMON_LANE_BINDING_SCHEMA",
    "EAAEF_IDEMPOTENT_RESERVATION_SCHEMA",
    "EAAEF_TASK_OPERATION_AUTHORITY_SCHEMA",
    "EAAEFBorrowedTransactionAdapter",
    "EAAEFBorrowedTransactionConflict",
    "EAAEFBorrowedTransactionError",
    "EAAEFBorrowedTransactionNotReady",
    "EAAEFBootstrapBorrowedTransactionOperationHandler",
    "eaaef_bootstrap_handler_source_evidence",
    "eaaef_reservation_id",
)
