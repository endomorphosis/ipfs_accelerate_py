"""Queryable, idempotent recovery and reconciliation transactions.

DQP-019 / RecoveryAction@1
==========================

:class:`DatabaseRecovery` records reconciliation replay, quarantine, retry
budget decisions, crash recovery, and rescue dispositions as durable database
rows. Actions are idempotent under an explicit idempotency key and always
queryable by task, entry, kind, or action id.

Authority rules (fail-closed)
-----------------------------
* Recovery mutates merge/validation state only through
  :class:`~ipfs_accelerate_py.agent_supervisor.merge.database_merge_queue.DatabaseMergeQueue`
  transactions that share task/attempt/worktree/fence coordinates.
* Replaying the same idempotency key returns the prior action without double
  applying side effects.
* JSON receipts and queue files cannot settle recovery or completion.
* Retry exhaustion is an explicit, queryable terminal disposition.

Cold import of this module performs no filesystem, database, network, provider,
or process action.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..merge.database_merge_queue import (
    DatabaseMergeError,
    DatabaseMergeQueue,
    MergeEntryStatus,
    duckdb_available as merge_duckdb_available,
    open_database_merge_queue,
)

# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_RECOVERY_INTERFACE: Final[str] = "DatabaseRecovery@1"
RECOVERY_ACTION_INTERFACE: Final[str] = "RecoveryAction@1"

DATABASE_RECOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-recovery@1"
)
RECOVERY_ACTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/recovery-action@1"
)

DEFAULT_RETRY_BUDGET: Final[int] = 3
MAX_PAYLOAD_BYTES: Final[int] = 262_144

ClockMs = Callable[[], int]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseRecoveryError(RuntimeError):
    """Base fail-closed error for database recovery authority."""

    code = "DQP_RECOVERY_ERROR"


class DatabaseRecoveryNotOpenError(DatabaseRecoveryError):
    code = "DQP_RECOVERY_NOT_OPEN"


class DatabaseRecoveryConflictError(DatabaseRecoveryError):
    code = "DQP_RECOVERY_CONFLICT"


class DatabaseRecoveryBoundsError(DatabaseRecoveryError, ValueError):
    code = "DQP_RECOVERY_BOUNDS"


class DuckDBUnavailableError(DatabaseRecoveryError):
    code = "DQP_DUCKDB_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class RecoveryActionKind(str, Enum):
    RECONCILE = "reconcile"
    QUARANTINE = "quarantine"
    RETRY = "retry"
    RETRY_EXHAUSTED = "retry_exhausted"
    RELEASE_STALE_CLAIM = "release_stale_claim"
    RESCUE = "rescue"
    REPLAY = "replay"


class RecoveryActionStatus(str, Enum):
    ACCEPTED = "accepted"
    APPLIED = "applied"
    NOOP = "noop"
    REJECTED = "rejected"
    EXHAUSTED = "exhausted"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def duckdb_available() -> bool:
    """Return whether DuckDB is available for recovery stores."""

    return merge_duckdb_available()


def _default_clock_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if "\x00" in text:
        raise DatabaseRecoveryError(f"{name} contains NUL")
    if required and not text:
        raise DatabaseRecoveryError(f"{name} is required")
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatabaseRecoveryBoundsError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DatabaseRecoveryBoundsError(f"{name} must be a positive integer")
    return value


def _bounded_mapping(
    body: Mapping[str, Any] | None,
    *,
    name: str,
    max_bytes: int = MAX_PAYLOAD_BYTES,
) -> dict[str, Any]:
    import json

    raw = dict(body or {})
    encoded = json.dumps(
        raw,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    if len(encoded) > max_bytes:
        raise DatabaseRecoveryBoundsError(
            f"{name} exceeds the {max_bytes}-byte bound"
        )
    return raw


def _enum_value(value: Any, enum_cls: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value).strip().lower())
    except Exception as exc:
        raise DatabaseRecoveryError(f"invalid {name}: {value!r}") from exc


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryAction:
    """One durable, queryable recovery or reconciliation decision."""

    INTERFACE: ClassVar[str] = RECOVERY_ACTION_INTERFACE
    SCHEMA: ClassVar[str] = RECOVERY_ACTION_SCHEMA

    action_id: str
    action_kind: RecoveryActionKind
    status: RecoveryActionStatus
    recorded_at_ms: int
    task_cid: str = ""
    entry_id: str = ""
    worktree_id: str = ""
    fencing_token: int = 0
    fence_epoch: int = 0
    idempotency_key: str = ""
    retry_budget: int = 0
    retry_count: int = 0
    reason: str = ""
    finished_at_ms: int | None = None
    revision: int = 1
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _text(self.action_id, "action_id"))
        object.__setattr__(
            self,
            "action_kind",
            _enum_value(self.action_kind, RecoveryActionKind, "action_kind"),
        )
        object.__setattr__(
            self,
            "status",
            _enum_value(self.status, RecoveryActionStatus, "status"),
        )
        object.__setattr__(
            self,
            "recorded_at_ms",
            _nonneg_int(int(self.recorded_at_ms), "recorded_at_ms"),
        )
        object.__setattr__(
            self, "task_cid", _text(self.task_cid, "task_cid", required=False)
        )
        object.__setattr__(
            self, "entry_id", _text(self.entry_id, "entry_id", required=False)
        )
        object.__setattr__(
            self,
            "worktree_id",
            _text(self.worktree_id, "worktree_id", required=False),
        )
        object.__setattr__(
            self,
            "fencing_token",
            _nonneg_int(int(self.fencing_token), "fencing_token"),
        )
        object.__setattr__(
            self, "fence_epoch", _nonneg_int(int(self.fence_epoch), "fence_epoch")
        )
        object.__setattr__(
            self,
            "idempotency_key",
            _text(self.idempotency_key, "idempotency_key", required=False),
        )
        object.__setattr__(
            self, "retry_budget", _nonneg_int(int(self.retry_budget), "retry_budget")
        )
        object.__setattr__(
            self, "retry_count", _nonneg_int(int(self.retry_count), "retry_count")
        )
        object.__setattr__(
            self, "reason", _text(self.reason, "reason", required=False)
        )
        if self.finished_at_ms is not None:
            object.__setattr__(
                self,
                "finished_at_ms",
                _nonneg_int(int(self.finished_at_ms), "finished_at_ms"),
            )
        object.__setattr__(
            self, "revision", _positive_int(int(self.revision), "revision")
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(_bounded_mapping(dict(self.body or {}), name="body")),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RecoveryAction":
        return cls(
            action_id=str(payload.get("action_id") or ""),
            action_kind=str(payload.get("action_kind") or ""),
            status=str(payload.get("status") or ""),
            recorded_at_ms=int(payload.get("recorded_at_ms") or 0),
            task_cid=str(payload.get("task_cid") or ""),
            entry_id=str(payload.get("entry_id") or ""),
            worktree_id=str(payload.get("worktree_id") or ""),
            fencing_token=int(payload.get("fencing_token") or 0),
            fence_epoch=int(payload.get("fence_epoch") or 0),
            idempotency_key=str(payload.get("idempotency_key") or ""),
            retry_budget=int(payload.get("retry_budget") or 0),
            retry_count=int(payload.get("retry_count") or 0),
            reason=str(payload.get("reason") or ""),
            finished_at_ms=(
                None
                if payload.get("finished_at_ms") is None
                else int(payload.get("finished_at_ms") or 0)
            ),
            revision=int(payload.get("revision") or 1),
            body=dict(payload.get("body") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "action_id": self.action_id,
            "action_kind": self.action_kind.value
            if isinstance(self.action_kind, RecoveryActionKind)
            else str(self.action_kind),
            "status": self.status.value
            if isinstance(self.status, RecoveryActionStatus)
            else str(self.status),
            "recorded_at_ms": int(self.recorded_at_ms),
            "finished_at_ms": self.finished_at_ms,
            "task_cid": self.task_cid,
            "entry_id": self.entry_id,
            "worktree_id": self.worktree_id,
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "idempotency_key": self.idempotency_key,
            "retry_budget": int(self.retry_budget),
            "retry_count": int(self.retry_count),
            "reason": self.reason,
            "revision": int(self.revision),
            "body": dict(self.body),
        }


# ---------------------------------------------------------------------------
# DatabaseRecovery
# ---------------------------------------------------------------------------


class DatabaseRecovery:
    """Recovery/reconciliation facade over the shared merge-recovery store.

    Uses the single :class:`DatabaseMergeQueue` writer connection so DuckDB
    exclusive-lock semantics hold. Recovery actions are durable rows queryable
    independently of JSON receipts.
    """

    INTERFACE: ClassVar[str] = DATABASE_RECOVERY_INTERFACE
    SCHEMA: ClassVar[str] = DATABASE_RECOVERY_SCHEMA

    def __init__(
        self,
        database_path: Path | str | None = None,
        *,
        merge_queue: DatabaseMergeQueue | None = None,
        clock_ms: ClockMs | None = None,
        default_retry_budget: int = DEFAULT_RETRY_BUDGET,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DatabaseRecovery; install the optional "
                "duckdb dependency"
            )
        if merge_queue is None and database_path is None:
            raise DatabaseRecoveryError(
                "database_path or merge_queue is required"
            )
        self._owned_queue = merge_queue is None
        if merge_queue is not None:
            self._queue = merge_queue
            self._path = merge_queue.database_path
        else:
            self._path = Path(database_path)  # type: ignore[arg-type]
            self._queue = DatabaseMergeQueue(
                self._path,
                clock_ms=clock_ms,
                max_attempts=max(1, int(default_retry_budget)),
            )
        self._clock_ms = clock_ms or getattr(
            self._queue, "_clock_ms", _default_clock_ms
        )
        self._default_retry_budget = _nonneg_int(
            int(default_retry_budget), "default_retry_budget"
        )
        self._closed = True

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def merge_queue(self) -> DatabaseMergeQueue:
        return self._queue

    @property
    def is_open(self) -> bool:
        return not self._closed and self._queue.is_open

    def open(self) -> "DatabaseRecovery":
        if self.is_open:
            return self
        if not self._queue.is_open:
            self._queue.open()
        self._closed = False
        return self

    def close(self) -> None:
        self._closed = True
        if self._owned_queue:
            self._queue.close()

    def __enter__(self) -> "DatabaseRecovery":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require(self) -> DatabaseMergeQueue:
        if not self.is_open or not self._queue.is_open:
            raise DatabaseRecoveryNotOpenError("DatabaseRecovery is not open")
        return self._queue

    def _record(
        self,
        *,
        action_kind: RecoveryActionKind,
        status: RecoveryActionStatus,
        task_cid: str = "",
        entry_id: str = "",
        worktree_id: str = "",
        fencing_token: int = 0,
        fence_epoch: int = 0,
        idempotency_key: str = "",
        retry_budget: int = 0,
        retry_count: int = 0,
        reason: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> RecoveryAction:
        queue = self._require()
        payload = queue.record_recovery_action(
            action_kind=action_kind.value,
            status=status.value,
            task_cid=task_cid,
            entry_id=entry_id,
            worktree_id=worktree_id,
            fencing_token=fencing_token,
            fence_epoch=fence_epoch,
            idempotency_key=idempotency_key,
            retry_budget=retry_budget,
            retry_count=retry_count,
            reason=reason,
            body=body,
        )
        return RecoveryAction.from_dict(payload)

    # -- query surface -------------------------------------------------------

    def get_action(self, action_id: str) -> RecoveryAction | None:
        queue = self._require()
        payload = queue.get_recovery_action(action_id)
        return None if payload is None else RecoveryAction.from_dict(payload)

    def get_by_idempotency_key(self, idempotency_key: str) -> RecoveryAction | None:
        queue = self._require()
        payload = queue.get_recovery_by_idempotency(idempotency_key)
        return None if payload is None else RecoveryAction.from_dict(payload)

    def list_actions(
        self,
        *,
        task_cid: str = "",
        entry_id: str = "",
        action_kind: RecoveryActionKind | str | None = None,
        status: RecoveryActionStatus | str | None = None,
        limit: int = 100,
    ) -> tuple[RecoveryAction, ...]:
        queue = self._require()
        kind = ""
        if action_kind is not None:
            kind = (
                action_kind.value
                if isinstance(action_kind, RecoveryActionKind)
                else str(action_kind).strip().lower()
            )
        status_text = ""
        if status is not None:
            status_text = (
                status.value
                if isinstance(status, RecoveryActionStatus)
                else str(status).strip().lower()
            )
        rows = queue.list_recovery_actions(
            task_cid=task_cid,
            entry_id=entry_id,
            action_kind=kind,
            status=status_text,
            limit=limit,
        )
        return tuple(RecoveryAction.from_dict(row) for row in rows)

    # -- recovery operations -------------------------------------------------

    def reconcile(
        self,
        *,
        entry_id: str,
        reason: str = "reconcile",
        idempotency_key: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> RecoveryAction:
        """Record a reconciliation observation for one merge entry (idempotent)."""

        eid = _text(entry_id, "entry_id")
        key = _text(idempotency_key, "idempotency_key", required=False)
        queue = self._require()
        if key:
            existing = queue.get_recovery_by_idempotency(key)
            if existing is not None:
                return RecoveryAction.from_dict(existing)
        entry = queue.get_entry(eid)
        if entry is None:
            raise DatabaseRecoveryError(f"unknown merge entry {eid}")
        validation = queue.current_validation(eid)
        completion = queue.get_completion(entry.task_cid)
        observed = {
            "entry": entry.to_dict(),
            "database_complete": queue.is_task_complete(entry.task_cid),
            "completion": None if completion is None else completion.to_dict(),
            "current_validation": (
                None if validation is None else validation.to_dict()
            ),
        }
        return self._record(
            action_kind=RecoveryActionKind.RECONCILE,
            status=RecoveryActionStatus.APPLIED,
            task_cid=entry.task_cid,
            entry_id=entry.entry_id,
            worktree_id=entry.worktree_id,
            fencing_token=entry.fencing_token,
            fence_epoch=entry.fence_epoch,
            idempotency_key=key,
            reason=_text(reason, "reason", required=False),
            body={
                **_bounded_mapping(body, name="body"),
                "observed": observed,
            },
        )

    def quarantine(
        self,
        *,
        entry_id: str,
        reason: str = "",
        claim_token: str = "",
        claim_generation: int = 0,
        force: bool = False,
        idempotency_key: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> RecoveryAction:
        """Quarantine a merge entry and record a durable recovery action."""

        eid = _text(entry_id, "entry_id")
        key = _text(idempotency_key, "idempotency_key", required=False)
        queue = self._require()
        if key:
            existing = queue.get_recovery_by_idempotency(key)
            if existing is not None:
                return RecoveryAction.from_dict(existing)
        try:
            entry = queue.quarantine_entry(
                entry_id=eid,
                reason=reason,
                claim_token=claim_token,
                claim_generation=claim_generation,
                force=force,
            )
        except DatabaseMergeError as exc:
            return self._record(
                action_kind=RecoveryActionKind.QUARANTINE,
                status=RecoveryActionStatus.REJECTED,
                entry_id=eid,
                idempotency_key=key,
                reason=str(exc),
                body=_bounded_mapping(body, name="body"),
            )
        return self._record(
            action_kind=RecoveryActionKind.QUARANTINE,
            status=RecoveryActionStatus.APPLIED,
            task_cid=entry.task_cid,
            entry_id=entry.entry_id,
            worktree_id=entry.worktree_id,
            fencing_token=entry.fencing_token,
            fence_epoch=entry.fence_epoch,
            idempotency_key=key,
            reason=_text(reason, "reason", required=False),
            body={
                **_bounded_mapping(body, name="body"),
                "entry_status": entry.status.value,
            },
        )

    def schedule_retry(
        self,
        *,
        entry_id: str,
        reason: str = "",
        claim_token: str = "",
        claim_generation: int = 0,
        force: bool = False,
        retry_budget: int | None = None,
        idempotency_key: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> RecoveryAction:
        """Retry within budget or record explicit exhaustion (idempotent)."""

        eid = _text(entry_id, "entry_id")
        key = _text(idempotency_key, "idempotency_key", required=False)
        budget = (
            self._default_retry_budget
            if retry_budget is None
            else _nonneg_int(int(retry_budget), "retry_budget")
        )
        queue = self._require()
        if key:
            existing = queue.get_recovery_by_idempotency(key)
            if existing is not None:
                return RecoveryAction.from_dict(existing)

        before = queue.get_entry(eid)
        if before is None:
            raise DatabaseRecoveryError(f"unknown merge entry {eid}")

        if int(before.failure_count) >= budget and before.status in {
            MergeEntryStatus.FAILED,
            MergeEntryStatus.CLAIMED,
            MergeEntryStatus.VALIDATING,
            MergeEntryStatus.MERGING,
            MergeEntryStatus.QUARANTINED,
        }:
            entry = before
            if before.status is not MergeEntryStatus.QUARANTINED:
                entry = queue.quarantine_entry(
                    entry_id=eid,
                    reason=reason or "retry_budget_exhausted",
                    claim_token=claim_token,
                    claim_generation=claim_generation,
                    force=force or not claim_token,
                )
            return self._record(
                action_kind=RecoveryActionKind.RETRY_EXHAUSTED,
                status=RecoveryActionStatus.EXHAUSTED,
                task_cid=entry.task_cid,
                entry_id=entry.entry_id,
                worktree_id=entry.worktree_id,
                fencing_token=entry.fencing_token,
                fence_epoch=entry.fence_epoch,
                idempotency_key=key,
                retry_budget=budget,
                retry_count=int(entry.failure_count),
                reason=_text(
                    reason or "retry_budget_exhausted",
                    "reason",
                    required=False,
                ),
                body=_bounded_mapping(body, name="body"),
            )

        entry = queue.requeue_for_retry(
            entry_id=eid,
            reason=reason,
            claim_token=claim_token,
            claim_generation=claim_generation,
            force=force or not claim_token,
        )
        exhausted = entry.status is MergeEntryStatus.QUARANTINED
        return self._record(
            action_kind=(
                RecoveryActionKind.RETRY_EXHAUSTED
                if exhausted
                else RecoveryActionKind.RETRY
            ),
            status=(
                RecoveryActionStatus.EXHAUSTED
                if exhausted
                else RecoveryActionStatus.APPLIED
            ),
            task_cid=entry.task_cid,
            entry_id=entry.entry_id,
            worktree_id=entry.worktree_id,
            fencing_token=entry.fencing_token,
            fence_epoch=entry.fence_epoch,
            idempotency_key=key,
            retry_budget=budget,
            retry_count=int(entry.failure_count),
            reason=_text(reason, "reason", required=False),
            body={
                **_bounded_mapping(body, name="body"),
                "entry_status": entry.status.value,
                "attempt_number": entry.attempt_number,
            },
        )

    def release_stale_claim(
        self,
        *,
        entry_id: str,
        expected_claim_token: str,
        expected_claim_generation: int,
        reason: str = "crash_recovery",
        idempotency_key: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> RecoveryAction:
        """CAS-release a crashed worker claim and record the recovery action."""

        eid = _text(entry_id, "entry_id")
        key = _text(idempotency_key, "idempotency_key", required=False)
        queue = self._require()
        if key:
            existing = queue.get_recovery_by_idempotency(key)
            if existing is not None:
                return RecoveryAction.from_dict(existing)
        try:
            entry = queue.release_stale_claim(
                entry_id=eid,
                expected_claim_token=expected_claim_token,
                expected_claim_generation=expected_claim_generation,
                reason=reason,
            )
            status = RecoveryActionStatus.APPLIED
            err_reason = reason
        except DatabaseMergeError as exc:
            entry = queue.get_entry(eid)
            status = RecoveryActionStatus.REJECTED
            err_reason = str(exc)
            if entry is None:
                raise DatabaseRecoveryError(str(exc)) from exc
        return self._record(
            action_kind=RecoveryActionKind.RELEASE_STALE_CLAIM,
            status=status,
            task_cid=entry.task_cid if entry else "",
            entry_id=eid,
            worktree_id=entry.worktree_id if entry else "",
            fencing_token=entry.fencing_token if entry else 0,
            fence_epoch=entry.fence_epoch if entry else 0,
            idempotency_key=key,
            reason=_text(err_reason, "reason", required=False),
            body={
                **_bounded_mapping(body, name="body"),
                "entry_status": entry.status.value if entry is not None else "",
            },
        )

    def rescue(
        self,
        *,
        entry_id: str,
        reason: str = "rescue",
        disposition: str = "manual_rescue",
        idempotency_key: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> RecoveryAction:
        """Record a rescue decision; does not forge completion authority."""

        eid = _text(entry_id, "entry_id")
        key = _text(idempotency_key, "idempotency_key", required=False)
        queue = self._require()
        if key:
            existing = queue.get_recovery_by_idempotency(key)
            if existing is not None:
                return RecoveryAction.from_dict(existing)
        entry = queue.get_entry(eid)
        if entry is None:
            raise DatabaseRecoveryError(f"unknown merge entry {eid}")
        if queue.is_task_complete(entry.task_cid):
            status = RecoveryActionStatus.NOOP
        else:
            status = RecoveryActionStatus.ACCEPTED
        return self._record(
            action_kind=RecoveryActionKind.RESCUE,
            status=status,
            task_cid=entry.task_cid,
            entry_id=entry.entry_id,
            worktree_id=entry.worktree_id,
            fencing_token=entry.fencing_token,
            fence_epoch=entry.fence_epoch,
            idempotency_key=key,
            reason=_text(reason, "reason", required=False),
            body={
                **_bounded_mapping(body, name="body"),
                "disposition": _text(disposition, "disposition"),
                "forges_completion": False,
                "database_complete": queue.is_task_complete(entry.task_cid),
            },
        )

    def replay(
        self,
        *,
        action_id: str,
        idempotency_key: str = "",
    ) -> RecoveryAction:
        """Replay a prior action by id; identical outcomes are no-ops."""

        aid = _text(action_id, "action_id")
        key = _text(idempotency_key, "idempotency_key", required=False)
        queue = self._require()
        if key:
            existing = queue.get_recovery_by_idempotency(key)
            if existing is not None:
                return RecoveryAction.from_dict(existing)
        prior_payload = queue.get_recovery_action(aid)
        if prior_payload is None:
            raise DatabaseRecoveryError(f"unknown recovery action {aid}")
        prior = RecoveryAction.from_dict(prior_payload)
        return self._record(
            action_kind=RecoveryActionKind.REPLAY,
            status=RecoveryActionStatus.NOOP,
            task_cid=prior.task_cid,
            entry_id=prior.entry_id,
            worktree_id=prior.worktree_id,
            fencing_token=prior.fencing_token,
            fence_epoch=prior.fence_epoch,
            idempotency_key=key or f"replay:{aid}",
            retry_budget=prior.retry_budget,
            retry_count=prior.retry_count,
            reason=f"replay of {aid}",
            body={
                "prior_action_id": prior.action_id,
                "prior_action_kind": prior.action_kind.value,
                "prior_status": prior.status.value,
                "idempotent": True,
            },
        )


def open_database_recovery(
    database_path: Path | str | None = None,
    *,
    merge_queue: DatabaseMergeQueue | None = None,
    clock_ms: ClockMs | None = None,
    default_retry_budget: int = DEFAULT_RETRY_BUDGET,
) -> DatabaseRecovery:
    """Open a :class:`DatabaseRecovery` store, optionally sharing a merge queue."""

    if merge_queue is None and database_path is not None:
        owned = open_database_merge_queue(
            database_path,
            clock_ms=clock_ms,
            max_attempts=max(1, int(default_retry_budget)),
        )
        recovery = DatabaseRecovery(
            merge_queue=owned,
            clock_ms=clock_ms,
            default_retry_budget=default_retry_budget,
        )
        recovery._owned_queue = True  # noqa: SLF001 — factory ownership handoff
        return recovery.open()
    return DatabaseRecovery(
        database_path,
        merge_queue=merge_queue,
        clock_ms=clock_ms,
        default_retry_budget=default_retry_budget,
    ).open()


__all__ = [
    "DATABASE_RECOVERY_INTERFACE",
    "RECOVERY_ACTION_INTERFACE",
    "DATABASE_RECOVERY_SCHEMA",
    "RECOVERY_ACTION_SCHEMA",
    "DEFAULT_RETRY_BUDGET",
    "RecoveryActionKind",
    "RecoveryActionStatus",
    "RecoveryAction",
    "DatabaseRecovery",
    "DatabaseRecoveryError",
    "DatabaseRecoveryNotOpenError",
    "DatabaseRecoveryConflictError",
    "DatabaseRecoveryBoundsError",
    "DuckDBUnavailableError",
    "duckdb_available",
    "open_database_recovery",
]
