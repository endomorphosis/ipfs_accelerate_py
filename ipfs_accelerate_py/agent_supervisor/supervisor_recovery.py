"""Bounded, fail-closed recovery for persistent supervisor state.

The module deliberately separates *repair evidence* from process liveness.
Recovery starts from a verified content-addressed checkpoint, retains an exact
event cursor, fences stale actors, and publishes an immutable receipt.  A
missing/corrupt proof or an exhausted resource bound produces quarantine, not
an inferred success.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .control_contracts import EventCursor
from .event_log import (
    latest_event_cursor,
    recover_jsonl_event_log_tail,
    utc_now,
)
from .prompt_workflow import (
    IncidentKind,
    ProgrammaticRecoveryExhaustionReceipt,
    PromptWorkflowBudget,
    RecordStatus,
    RecoveryAttempt,
    RecoveryAttemptOutcome,
    RescueOperation,
    SupervisorIncident,
    prompt_workflow_cid,
)
from .recovery_diagnostics import RecoveryDiagnosis
from .supervisor_v2_contracts import MAX_PROJECTION_BYTES, MAX_RECEIPT_BYTES


BOUNDED_RECOVERY_REQUIREMENT_ID: Final = (
    "asi-118:bounded-crash-recovery-and-repair-evidence"
)
RECOVERY_CHECKPOINT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/recovery-checkpoint@1"
)
REPAIR_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/repair-receipt@1"
)
RECOVERY_INCIDENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/recovery-incident@1"
)
PROGRAMMATIC_RECOVERY_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/programmatic-recovery-receipt@1"
)


class RecoveryError(RuntimeError):
    """Base class for bounded recovery failures."""


class RecoveryIntegrityError(RecoveryError):
    """Persistent recovery state failed its content-addressed contract."""


class RecoveryBoundExceeded(RecoveryError):
    """A retry, receipt, quarantine, or storage bound was exhausted."""


class RecoveryFault(str, Enum):
    PROCESS_CRASH = "process_crash"
    KILL_ESCALATION = "kill_escalation"
    PARTIAL_EVENT_WRITE = "partial_event_write"
    PARTIAL_CHECKPOINT_WRITE = "partial_checkpoint_write"
    STALE_LEASE = "stale_lease"
    CORRUPT_CACHE = "corrupt_cache"
    DUPLICATE_EVENT = "duplicate_event"
    PROVIDER_LOSS = "provider_loss"
    DISK_FULL = "disk_full"
    SLOW_DISK = "slow_disk"
    INTERRUPTED_VALIDATION = "interrupted_validation"
    INTERRUPTED_MERGE = "interrupted_merge"
    RESTART_DURING_REFILL = "restart_during_refill"


class RecoveryDisposition(str, Enum):
    RECOVERED = "recovered"
    NOOP = "noop"
    QUARANTINED = "quarantined"
    FAILED_CLOSED = "failed_closed"


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise RecoveryIntegrityError(
            "recovery artifacts require canonical JSON values"
        ) from exc


def _content_id(kind: str, value: Any) -> str:
    return f"{kind}:sha256:{hashlib.sha256(_canonical_bytes(value)).hexdigest()}"


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_json(member)
                for key, member in value.items()
            }
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return tuple(_freeze_json(member) for member in value)
    return value


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _plain_json(member)
            for key, member in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_plain_json(member) for member in value]
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _required_text(value: Any, name: str) -> str:
    selected = str(value or "").strip()
    if not selected:
        raise ValueError(f"{name} must be non-empty")
    return selected


def _serialize_incident(method: Callable[..., RepairReceipt]):
    """Collapse concurrent recovery of one incident to one persisted receipt."""

    @wraps(method)
    def wrapped(
        self: "SupervisorRecovery", *args: Any, **kwargs: Any
    ) -> RepairReceipt:
        incident_id = _required_text(
            kwargs.get("incident_id"), "incident_id"
        )
        digest = hashlib.sha256(incident_id.encode("utf-8")).hexdigest()
        lock_path = self.state_dir / "incident-locks" / f"{digest}.lock"
        with self._thread_lock:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+b") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    return method(self, *args, **kwargs)
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    return wrapped


@dataclass(frozen=True)
class RecoveryPolicy:
    """Hard bounds applied to one recovery store and incident."""

    max_attempts: int = 3
    max_checkpoints: int = 16
    max_receipts: int = 128
    max_checkpoint_bytes: int = MAX_PROJECTION_BYTES
    max_receipt_bytes: int = MAX_RECEIPT_BYTES
    max_quarantine_bytes: int = 4 * MAX_PROJECTION_BYTES
    max_storage_bytes: int = 64 * MAX_PROJECTION_BYTES
    slow_operation_seconds: float = 30.0

    def __post_init__(self) -> None:
        for name in (
            "max_attempts",
            "max_checkpoints",
            "max_receipts",
            "max_checkpoint_bytes",
            "max_receipt_bytes",
            "max_quarantine_bytes",
            "max_storage_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.slow_operation_seconds, bool)
            or not isinstance(self.slow_operation_seconds, (int, float))
            or self.slow_operation_seconds <= 0
        ):
            raise ValueError("slow_operation_seconds must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "max_checkpoints": self.max_checkpoints,
            "max_receipts": self.max_receipts,
            "max_checkpoint_bytes": self.max_checkpoint_bytes,
            "max_receipt_bytes": self.max_receipt_bytes,
            "max_quarantine_bytes": self.max_quarantine_bytes,
            "max_storage_bytes": self.max_storage_bytes,
            "slow_operation_seconds": self.slow_operation_seconds,
        }

    @property
    def policy_id(self) -> str:
        return _content_id("recovery-policy", self.to_dict())


@dataclass(frozen=True)
class RecoveryCheckpoint:
    """Canonical restart state bound to an event cursor and accepted evidence."""

    repository_id: str
    tree_id: str
    generation: int
    state: Mapping[str, Any]
    cursor: EventCursor
    accepted_merged_tree_evidence: tuple[str, ...] = ()
    semantic_roots: Mapping[str, str] = field(default_factory=dict)
    proof_index_id: str = ""
    cas_invalidation_id: str = ""
    fencing_epoch: int = 0
    created_at: str = field(default_factory=utc_now)
    checkpoint_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _required_text(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "tree_id", _required_text(self.tree_id, "tree_id"))
        object.__setattr__(
            self, "created_at", _required_text(self.created_at, "created_at")
        )
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 1
        ):
            raise ValueError("generation must be a positive integer")
        if not isinstance(self.state, Mapping):
            raise TypeError("state must be a mapping")
        canonical_state = json.loads(_canonical_bytes(self.state))
        if not isinstance(canonical_state, dict):
            raise TypeError("state must be a JSON object")
        object.__setattr__(self, "state", canonical_state)
        if not isinstance(self.cursor, EventCursor):
            raise TypeError("cursor must be an EventCursor")
        evidence = tuple(
            sorted(
                {
                    _required_text(item, "accepted_merged_tree_evidence item")
                    for item in self.accepted_merged_tree_evidence
                }
            )
        )
        object.__setattr__(self, "accepted_merged_tree_evidence", evidence)
        if not isinstance(self.semantic_roots, Mapping):
            raise TypeError("semantic_roots must be a mapping")
        roots = {
            _required_text(key, "semantic root kind"): _required_text(
                item, "semantic root identity"
            )
            for key, item in sorted(self.semantic_roots.items())
        }
        object.__setattr__(self, "semantic_roots", roots)
        for name in ("proof_index_id", "cas_invalidation_id"):
            value = str(getattr(self, name) or "").strip()
            object.__setattr__(self, name, value)
        if (
            isinstance(self.fencing_epoch, bool)
            or not isinstance(self.fencing_epoch, int)
            or self.fencing_epoch < 0
        ):
            raise ValueError("fencing_epoch must be a nonnegative integer")
        body = self.to_dict(include_id=False)
        expected = _content_id("recovery-checkpoint", body)
        if self.checkpoint_id and self.checkpoint_id != expected:
            raise RecoveryIntegrityError("recovery checkpoint identity mismatch")
        object.__setattr__(self, "checkpoint_id", expected)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": RECOVERY_CHECKPOINT_SCHEMA,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "generation": self.generation,
            "state": dict(self.state),
            "cursor": self.cursor.to_record(),
            "accepted_merged_tree_evidence": list(
                self.accepted_merged_tree_evidence
            ),
            "created_at": self.created_at,
        }
        # Empty additions are omitted so generation-2 @1 checkpoints retain
        # their historical content identities and remain readable.
        if self.semantic_roots:
            value["semantic_roots"] = dict(self.semantic_roots)
            value["semantic_roots_id"] = self.semantic_roots_id
        if self.proof_index_id:
            value["proof_index_id"] = self.proof_index_id
        if self.cas_invalidation_id:
            value["cas_invalidation_id"] = self.cas_invalidation_id
        if self.fencing_epoch:
            value["fencing_epoch"] = self.fencing_epoch
        if include_id:
            value["checkpoint_id"] = self.checkpoint_id
        return value

    @property
    def state_id(self) -> str:
        return _content_id("recovery-state", self.state)

    @property
    def semantic_roots_id(self) -> str:
        return _content_id("semantic-roots", dict(self.semantic_roots))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecoveryCheckpoint":
        if value.get("schema") != RECOVERY_CHECKPOINT_SCHEMA:
            raise RecoveryIntegrityError("unsupported recovery checkpoint schema")
        return cls(
            repository_id=str(value.get("repository_id") or ""),
            tree_id=str(value.get("tree_id") or ""),
            generation=value.get("generation"),  # type: ignore[arg-type]
            state=value.get("state") or {},
            cursor=EventCursor.from_dict(value.get("cursor") or {}),
            accepted_merged_tree_evidence=tuple(
                value.get("accepted_merged_tree_evidence") or ()
            ),
            semantic_roots=value.get("semantic_roots") or {},
            proof_index_id=str(value.get("proof_index_id") or ""),
            cas_invalidation_id=str(
                value.get("cas_invalidation_id") or ""
            ),
            fencing_epoch=value.get("fencing_epoch", 0),
            created_at=str(value.get("created_at") or ""),
            checkpoint_id=str(value.get("checkpoint_id") or ""),
        )


class RecoveryCheckpointStore:
    """Atomic head plus bounded immutable checkpoint history."""

    def __init__(
        self,
        root: Path | str,
        *,
        policy: RecoveryPolicy | None = None,
    ) -> None:
        self.root = Path(root)
        self.policy = policy or RecoveryPolicy()
        self.checkpoints = self.root / "checkpoints"
        self.head = self.root / "checkpoint-head.json"
        self.quarantine = self.root / "quarantine"
        self.lock_path = self.root / ".recovery.lock"
        self._thread_lock = threading.RLock()

    def _guard(self):
        store = self

        class Guard:
            handle: Any = None

            def __enter__(self) -> None:
                store._thread_lock.acquire()
                try:
                    store.root.mkdir(parents=True, exist_ok=True)
                    self.handle = store.lock_path.open("a+b")
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
                except BaseException:
                    store._thread_lock.release()
                    raise

            def __exit__(self, *_args: Any) -> None:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
                self.handle.close()
                store._thread_lock.release()

        return Guard()

    def _decode(self, path: Path) -> RecoveryCheckpoint:
        raw = path.read_bytes()
        if len(raw) > self.policy.max_checkpoint_bytes:
            raise RecoveryIntegrityError("recovery checkpoint exceeds byte bound")
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RecoveryIntegrityError("recovery checkpoint is malformed") from exc
        if not isinstance(value, Mapping):
            raise RecoveryIntegrityError("recovery checkpoint must be an object")
        checkpoint = RecoveryCheckpoint.from_dict(value)
        if path.parent == self.checkpoints and path != self._checkpoint_path(
            checkpoint
        ):
            raise RecoveryIntegrityError(
                "recovery checkpoint path does not match its identity"
            )
        return checkpoint

    def _checkpoint_path(self, checkpoint: RecoveryCheckpoint) -> Path:
        digest = checkpoint.checkpoint_id.rsplit(":", 1)[-1]
        return self.checkpoints / f"{checkpoint.generation:020d}-{digest}.json"

    def save(self, checkpoint: RecoveryCheckpoint) -> bool:
        payload = _canonical_bytes(checkpoint.to_dict()) + b"\n"
        if len(payload) > self.policy.max_checkpoint_bytes:
            raise RecoveryBoundExceeded("recovery checkpoint exceeds byte bound")
        with self._guard():
            current = self._load_last_valid_unlocked(quarantine_invalid=False)
            if current is not None:
                if checkpoint.generation < current.generation:
                    raise RecoveryIntegrityError(
                        "recovery checkpoint generation moved backwards"
                    )
                if (
                    checkpoint.generation == current.generation
                    and checkpoint.checkpoint_id != current.checkpoint_id
                ):
                    raise RecoveryIntegrityError(
                        "recovery checkpoint generation conflicts"
                    )
                if checkpoint.checkpoint_id == current.checkpoint_id:
                    return False
                if (
                    checkpoint.cursor.stream_id != current.cursor.stream_id
                    or checkpoint.cursor.snapshot_id
                    != current.cursor.snapshot_id
                    or checkpoint.cursor.position < current.cursor.position
                    or (
                        checkpoint.cursor.position == current.cursor.position
                        and checkpoint.cursor.last_event_id
                        != current.cursor.last_event_id
                    )
                ):
                    raise RecoveryIntegrityError(
                        "recovery checkpoint event cursor moved backwards or forked"
                    )
                if checkpoint.fencing_epoch < current.fencing_epoch:
                    raise RecoveryIntegrityError(
                        "recovery checkpoint fencing epoch moved backwards"
                    )
            path = self._checkpoint_path(checkpoint)
            _atomic_write(path, payload)
            _atomic_write(
                self.head,
                _canonical_bytes(
                    {
                        "schema": RECOVERY_CHECKPOINT_SCHEMA,
                        "checkpoint_id": checkpoint.checkpoint_id,
                        "generation": checkpoint.generation,
                        "path": path.name,
                    }
                )
                + b"\n",
            )
            self._prune_unlocked()
            return True

    checkpoint = save

    def _candidates(self) -> list[Path]:
        if not self.checkpoints.exists():
            return []
        return sorted(self.checkpoints.glob("*.json"), reverse=True)

    def _quarantine_invalid(self, path: Path) -> str:
        try:
            size = path.stat().st_size
        except OSError:
            return ""
        if size > self.policy.max_quarantine_bytes:
            return ""
        self.quarantine.mkdir(parents=True, exist_ok=True)
        target = self.quarantine / (
            f"{path.name}.invalid-{int(time.time_ns())}"
        )
        try:
            os.replace(path, target)
        except OSError:
            return ""
        return str(target)

    def _load_last_valid_unlocked(
        self, *, quarantine_invalid: bool
    ) -> RecoveryCheckpoint | None:
        for path in self._candidates()[: self.policy.max_checkpoints + 1]:
            try:
                return self._decode(path)
            except (OSError, RecoveryError, TypeError, ValueError):
                if quarantine_invalid:
                    self._quarantine_invalid(path)
        return None

    def load_last_valid(self) -> RecoveryCheckpoint | None:
        with self._guard():
            checkpoint = self._load_last_valid_unlocked(
                quarantine_invalid=True
            )
            if checkpoint is not None:
                expected_head = {
                    "schema": RECOVERY_CHECKPOINT_SCHEMA,
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "generation": checkpoint.generation,
                    "path": self._checkpoint_path(checkpoint).name,
                }
                valid_head = False
                try:
                    head_value = json.loads(self.head.read_bytes())
                    valid_head = head_value == expected_head
                except (
                    FileNotFoundError,
                    OSError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                ):
                    pass
                if not valid_head:
                    if self.head.exists():
                        self._quarantine_invalid(self.head)
                    _atomic_write(
                        self.head,
                        _canonical_bytes(expected_head) + b"\n",
                    )
            return checkpoint

    load = load_last_valid

    def _prune_unlocked(self) -> None:
        candidates = self._candidates()
        retained_bytes = 0
        for index, path in enumerate(candidates):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if (
                index >= self.policy.max_checkpoints
                or retained_bytes + size > self.policy.max_storage_bytes
            ):
                try:
                    path.unlink()
                except OSError:
                    pass
            else:
                retained_bytes += size


@dataclass(frozen=True)
class RepairReceipt:
    """Exact, bounded evidence for a single recovery decision."""

    incident_id: str
    fault: RecoveryFault
    disposition: RecoveryDisposition
    repository_id: str
    tree_id: str
    checkpoint_id: str
    state_id: str
    policy_id: str
    process_tree_id: str
    event_cursor: EventCursor
    attempts: int
    actions: tuple[str, ...]
    reason_code: str
    quarantined_paths: tuple[str, ...] = ()
    preserved_evidence_ids: tuple[str, ...] = ()
    resulting_projection_ids: tuple[str, ...] = ()
    stale_actor_fenced: bool = False
    replay_cursor: EventCursor | None = None
    checkpoint_semantic_roots: Mapping[str, str] = field(
        default_factory=dict
    )
    result_semantic_roots: Mapping[str, str] = field(default_factory=dict)
    precrash_permit_ids: tuple[str, ...] = ()
    invalidated_permit_ids: tuple[str, ...] = ()
    proof_index_id: str = ""
    cas_invalidation_id: str = ""
    fencing_epoch: int = 0
    observed_fencing_epoch: int = 0
    started_at: str = field(default_factory=utc_now)
    finished_at: str = field(default_factory=utc_now)
    requirement_id: str = BOUNDED_RECOVERY_REQUIREMENT_ID
    receipt_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "incident_id",
            "repository_id",
            "tree_id",
            "state_id",
            "policy_id",
            "process_tree_id",
            "reason_code",
            "started_at",
            "finished_at",
        ):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )
        if not isinstance(self.fault, RecoveryFault):
            object.__setattr__(self, "fault", RecoveryFault(str(self.fault)))
        if not isinstance(self.disposition, RecoveryDisposition):
            object.__setattr__(
                self,
                "disposition",
                RecoveryDisposition(str(self.disposition)),
            )
        if (
            isinstance(self.attempts, bool)
            or not isinstance(self.attempts, int)
            or self.attempts < 0
        ):
            raise ValueError("attempts must be a non-negative integer")
        if not isinstance(self.event_cursor, EventCursor):
            raise TypeError("event_cursor must be an EventCursor")
        for name in (
            "actions",
            "quarantined_paths",
            "preserved_evidence_ids",
            "resulting_projection_ids",
            "precrash_permit_ids",
            "invalidated_permit_ids",
        ):
            object.__setattr__(
                self,
                name,
                tuple(_required_text(item, f"{name} item") for item in getattr(self, name)),
            )
        for name in ("precrash_permit_ids", "invalidated_permit_ids"):
            object.__setattr__(
                self,
                name,
                tuple(sorted(set(getattr(self, name)))),
            )
        if self.replay_cursor is not None and not isinstance(
            self.replay_cursor, EventCursor
        ):
            raise TypeError("replay_cursor must be an EventCursor or None")
        for name in (
            "checkpoint_semantic_roots",
            "result_semantic_roots",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(
                self,
                name,
                {
                    _required_text(key, f"{name} key"): _required_text(
                        item, f"{name} identity"
                    )
                    for key, item in sorted(value.items())
                },
            )
        for name in ("proof_index_id", "cas_invalidation_id"):
            object.__setattr__(
                self, name, str(getattr(self, name) or "").strip()
            )
        for name in ("fencing_epoch", "observed_fencing_epoch"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"{name} must be a nonnegative integer"
                )
        body = self.to_dict(include_id=False)
        expected = _content_id("repair-receipt", body)
        if self.receipt_id and self.receipt_id != expected:
            raise RecoveryIntegrityError("repair receipt identity mismatch")
        object.__setattr__(self, "receipt_id", expected)

    @property
    def recovered(self) -> bool:
        return self.disposition in {
            RecoveryDisposition.RECOVERED,
            RecoveryDisposition.NOOP,
        }

    @property
    def failed_closed(self) -> bool:
        return self.disposition in {
            RecoveryDisposition.QUARANTINED,
            RecoveryDisposition.FAILED_CLOSED,
        }

    @property
    def evidence_claim_ids(self) -> tuple[str, ...]:
        return (self.requirement_id,) if self.recovered else ()

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": REPAIR_RECEIPT_SCHEMA,
            "requirement_id": self.requirement_id,
            "incident_id": self.incident_id,
            "fault": self.fault.value,
            "disposition": self.disposition.value,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "checkpoint_id": self.checkpoint_id,
            "state_id": self.state_id,
            "policy_id": self.policy_id,
            "process_tree_id": self.process_tree_id,
            "event_cursor": self.event_cursor.to_record(),
            "attempts": self.attempts,
            "actions": list(self.actions),
            "reason_code": self.reason_code,
            "quarantined_paths": list(self.quarantined_paths),
            "preserved_evidence_ids": list(self.preserved_evidence_ids),
            "resulting_projection_ids": list(
                self.resulting_projection_ids
            ),
            "stale_actor_fenced": self.stale_actor_fenced,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if self.replay_cursor is not None:
            value["replay_cursor"] = self.replay_cursor.to_record()
        if self.checkpoint_semantic_roots:
            value["checkpoint_semantic_roots"] = dict(
                self.checkpoint_semantic_roots
            )
        if self.result_semantic_roots:
            value["result_semantic_roots"] = dict(
                self.result_semantic_roots
            )
        if self.invalidated_permit_ids:
            value["invalidated_permit_ids"] = list(
                self.invalidated_permit_ids
            )
        if self.precrash_permit_ids:
            value["precrash_permit_ids"] = list(
                self.precrash_permit_ids
            )
        if self.proof_index_id:
            value["proof_index_id"] = self.proof_index_id
        if self.cas_invalidation_id:
            value["cas_invalidation_id"] = self.cas_invalidation_id
        if self.fencing_epoch:
            value["fencing_epoch"] = self.fencing_epoch
        if self.observed_fencing_epoch:
            value["observed_fencing_epoch"] = self.observed_fencing_epoch
        if include_id:
            value["receipt_id"] = self.receipt_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RepairReceipt":
        if value.get("schema") != REPAIR_RECEIPT_SCHEMA:
            raise RecoveryIntegrityError("unsupported repair receipt schema")
        return cls(
            incident_id=str(value.get("incident_id") or ""),
            fault=RecoveryFault(str(value.get("fault") or "")),
            disposition=RecoveryDisposition(
                str(value.get("disposition") or "")
            ),
            repository_id=str(value.get("repository_id") or ""),
            tree_id=str(value.get("tree_id") or ""),
            checkpoint_id=str(value.get("checkpoint_id") or ""),
            state_id=str(value.get("state_id") or ""),
            policy_id=str(value.get("policy_id") or ""),
            process_tree_id=str(value.get("process_tree_id") or ""),
            event_cursor=EventCursor.from_dict(value.get("event_cursor") or {}),
            attempts=value.get("attempts"),  # type: ignore[arg-type]
            actions=tuple(value.get("actions") or ()),
            reason_code=str(value.get("reason_code") or ""),
            quarantined_paths=tuple(value.get("quarantined_paths") or ()),
            preserved_evidence_ids=tuple(
                value.get("preserved_evidence_ids") or ()
            ),
            resulting_projection_ids=tuple(
                value.get("resulting_projection_ids") or ()
            ),
            stale_actor_fenced=bool(value.get("stale_actor_fenced", False)),
            replay_cursor=(
                EventCursor.from_dict(value.get("replay_cursor") or {})
                if value.get("replay_cursor") is not None
                else None
            ),
            checkpoint_semantic_roots=value.get(
                "checkpoint_semantic_roots"
            )
            or {},
            result_semantic_roots=value.get("result_semantic_roots") or {},
            precrash_permit_ids=tuple(
                value.get("precrash_permit_ids") or ()
            ),
            invalidated_permit_ids=tuple(
                value.get("invalidated_permit_ids") or ()
            ),
            proof_index_id=str(value.get("proof_index_id") or ""),
            cas_invalidation_id=str(
                value.get("cas_invalidation_id") or ""
            ),
            fencing_epoch=value.get("fencing_epoch", 0),
            observed_fencing_epoch=value.get(
                "observed_fencing_epoch", 0
            ),
            started_at=str(value.get("started_at") or ""),
            finished_at=str(value.get("finished_at") or ""),
            requirement_id=str(value.get("requirement_id") or ""),
            receipt_id=str(value.get("receipt_id") or ""),
        )


class FaultInjector:
    """Deterministic opt-in fault points for recovery fixtures.

    Production code is unaffected until a point is explicitly armed.  Values
    may be exception instances/types or callbacks; each arm has a finite hit
    count so fault injection itself cannot create an unbounded retry loop.
    """

    def __init__(self) -> None:
        self._points: dict[str, tuple[Any, int]] = {}
        self._lock = threading.Lock()

    def arm(self, point: str, effect: Any, *, times: int = 1) -> None:
        point = _required_text(point, "point")
        if isinstance(times, bool) or not isinstance(times, int) or times < 1:
            raise ValueError("times must be a positive integer")
        with self._lock:
            self._points[point] = (effect, times)

    def clear(self, point: str | None = None) -> None:
        with self._lock:
            if point is None:
                self._points.clear()
            else:
                self._points.pop(point, None)

    def inject(self, point: str, **context: Any) -> bool:
        with self._lock:
            selected = self._points.get(point)
            if selected is None:
                return False
            effect, remaining = selected
            if remaining == 1:
                self._points.pop(point, None)
            else:
                self._points[point] = (effect, remaining - 1)
        if isinstance(effect, BaseException):
            raise effect
        if isinstance(effect, type) and issubclass(effect, BaseException):
            raise effect(f"fault injected at {point}")
        if callable(effect):
            effect(point=point, **context)
        return True

    hit = inject


class SupervisorRecovery:
    """Coordinate bounded repair and persist one immutable exact receipt."""

    def __init__(
        self,
        state_dir: Path | str,
        *,
        policy: RecoveryPolicy | None = None,
        fault_injector: FaultInjector | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.policy = policy or RecoveryPolicy()
        self.checkpoints = RecoveryCheckpointStore(
            self.state_dir, policy=self.policy
        )
        self.receipts_dir = self.state_dir / "receipts"
        self.incidents_dir = self.state_dir / "incidents"
        self.quarantine_dir = self.state_dir / "quarantine"
        self.fault_injector = fault_injector or FaultInjector()
        self._thread_lock = threading.RLock()

    def checkpoint(
        self,
        *,
        repository_id: str,
        tree_id: str,
        generation: int,
        state: Mapping[str, Any],
        cursor: EventCursor,
        accepted_merged_tree_evidence: Sequence[str] = (),
        semantic_roots: Mapping[str, str] | None = None,
        proof_index_id: str = "",
        cas_invalidation_id: str = "",
        fencing_epoch: int = 0,
    ) -> RecoveryCheckpoint:
        checkpoint = RecoveryCheckpoint(
            repository_id=repository_id,
            tree_id=tree_id,
            generation=generation,
            state=state,
            cursor=cursor,
            accepted_merged_tree_evidence=tuple(
                accepted_merged_tree_evidence
            ),
            semantic_roots=semantic_roots or {},
            proof_index_id=proof_index_id,
            cas_invalidation_id=cas_invalidation_id,
            fencing_epoch=fencing_epoch,
        )
        self.checkpoints.save(checkpoint)
        return checkpoint

    def _receipt_path(self, incident_id: str) -> Path:
        digest = hashlib.sha256(incident_id.encode("utf-8")).hexdigest()
        return self.receipts_dir / f"{digest}.json"

    def receipt(self, incident_id: str) -> RepairReceipt | None:
        path = self._receipt_path(incident_id)
        try:
            raw = path.read_bytes()
            if len(raw) > self.policy.max_receipt_bytes:
                raise RecoveryIntegrityError("repair receipt exceeds byte bound")
            value = json.loads(raw)
            if not isinstance(value, Mapping):
                raise RecoveryIntegrityError("repair receipt must be an object")
            return RepairReceipt.from_dict(value)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RecoveryIntegrityError("repair receipt is malformed") from exc

    def _store_receipt(self, receipt: RepairReceipt) -> None:
        payload = _canonical_bytes(receipt.to_dict()) + b"\n"
        if len(payload) > self.policy.max_receipt_bytes:
            raise RecoveryBoundExceeded("repair receipt exceeds byte bound")
        _atomic_write(self._receipt_path(receipt.incident_id), payload)
        paths = sorted(
            self.receipts_dir.glob("*.json"),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
        for path in paths[self.policy.max_receipts :]:
            try:
                path.unlink()
            except OSError:
                pass

    def _quarantine_path(self, path: Path, incident_id: str) -> str:
        if not path.exists():
            return ""
        try:
            size = path.stat().st_size
        except OSError:
            return ""
        if size > self.policy.max_quarantine_bytes:
            return ""
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(
            f"{incident_id}:{path}".encode("utf-8")
        ).hexdigest()
        target = self.quarantine_dir / f"{digest}-{path.name}"
        try:
            if path.is_dir():
                shutil.move(str(path), str(target))
            else:
                os.replace(path, target)
        except OSError:
            return ""
        return str(target)

    @_serialize_incident
    def recover(
        self,
        *,
        incident_id: str,
        fault: RecoveryFault | str,
        repository_id: str,
        tree_id: str,
        event_log_path: Path | str | None = None,
        current_fencing_token: int | None = None,
        observed_fencing_token: int | None = None,
        repair: Callable[[RecoveryCheckpoint, int], bool | None] | None = None,
        verify: Callable[[RecoveryCheckpoint], bool] | None = None,
        quarantine_paths: Sequence[Path | str] = (),
        fence_actor: Callable[[int, int], bool] | None = None,
        process_tree_id: str = "process-tree:none",
        resulting_projection_ids: Sequence[str] = (),
        current_semantic_roots: Mapping[str, str] | None = None,
        current_event_cursor: EventCursor | Mapping[str, Any] | str | None = None,
        current_proof_index_id: str = "",
        current_cas_invalidation_id: str = "",
        precrash_permit_ids: Sequence[str] = (),
        fence_permits: Callable[[tuple[str, ...], int], bool] | None = None,
        replay_events: Callable[[RecoveryCheckpoint], Mapping[str, Any]] | None = None,
        root_reader: Callable[[], Mapping[str, str]] | None = None,
        runtime_cas: Any = None,
        proof_index: Any = None,
        canonical_artifacts: Sequence[Any] | None = None,
    ) -> RepairReceipt:
        """Recover one incident from the last valid checkpoint.

        ``repair`` may reconstruct external projections and is retried only up
        to ``policy.max_attempts``.  ``verify`` must positively attest the
        reconstructed state.  Omitted callbacks mean no external mutation is
        required; they never imply that an explicitly failing callback passed.
        """

        incident_id = _required_text(incident_id, "incident_id")
        repository_id = _required_text(repository_id, "repository_id")
        tree_id = _required_text(tree_id, "tree_id")
        process_tree_id = _required_text(process_tree_id, "process_tree_id")
        selected_roots = {
            _required_text(key, "semantic root kind"): _required_text(
                item, "semantic root identity"
            )
            for key, item in sorted((current_semantic_roots or {}).items())
        }
        permit_ids = tuple(
            sorted(
                {
                    _required_text(item, "precrash permit ID")
                    for item in precrash_permit_ids
                }
            )
        )
        if isinstance(current_event_cursor, str):
            expected_cursor = EventCursor.from_token(current_event_cursor)
        elif isinstance(current_event_cursor, Mapping):
            expected_cursor = EventCursor.from_dict(current_event_cursor)
        elif isinstance(current_event_cursor, EventCursor):
            expected_cursor = current_event_cursor
        elif current_event_cursor is None:
            expected_cursor = None
        else:
            raise TypeError(
                "current_event_cursor must be a cursor, record, token, or None"
            )
        selected_fault = (
            fault if isinstance(fault, RecoveryFault) else RecoveryFault(str(fault))
        )
        existing = self.receipt(incident_id)
        if existing is not None:
            if (
                existing.fault != selected_fault
                or existing.repository_id != repository_id
                or existing.tree_id != tree_id
                or existing.policy_id != self.policy.policy_id
                or existing.process_tree_id != process_tree_id
                or existing.resulting_projection_ids
                != tuple(resulting_projection_ids)
                or (
                    selected_roots
                    and dict(existing.result_semantic_roots)
                    != selected_roots
                )
                or (
                    expected_cursor is not None
                    and existing.replay_cursor != expected_cursor
                )
                or existing.precrash_permit_ids != permit_ids
                or (
                    current_proof_index_id
                    and existing.proof_index_id
                    != str(current_proof_index_id)
                )
                or (
                    current_cas_invalidation_id
                    and existing.cas_invalidation_id
                    != str(current_cas_invalidation_id)
                )
                or (
                    current_fencing_token is not None
                    and existing.fencing_epoch != current_fencing_token
                )
                or (
                    observed_fencing_token is not None
                    and existing.observed_fencing_epoch
                    != observed_fencing_token
                )
            ):
                raise RecoveryIntegrityError(
                    "incident identity was replayed with different bindings"
                )
            return existing

        started_at = utc_now()
        actions: list[str] = []
        quarantined: list[str] = []
        attempts = 0
        stale_actor_fenced = False
        permits_fenced = not permit_ids
        replay_cursor: EventCursor | None = None
        checkpoint_roots: dict[str, str] = {}
        result_roots = dict(selected_roots)
        result_proof_index_id = str(current_proof_index_id or "")
        result_cas_invalidation_id = str(
            current_cas_invalidation_id or ""
        )
        result_fencing_epoch = int(current_fencing_token or 0)
        root_reader_failed = False
        try:
            root_snapshot_before = (
                {
                    str(key): str(item)
                    for key, item in root_reader().items()
                }
                if root_reader is not None
                else None
            )
        except Exception:
            root_snapshot_before = None
            root_reader_failed = True
        checkpoint = self.checkpoints.load_last_valid()
        if checkpoint is None:
            disposition = RecoveryDisposition.FAILED_CLOSED
            reason = "no_valid_checkpoint"
            cursor = EventCursor.initial(
                f"recovery:{repository_id}", snapshot_id=f"tree:{tree_id}"
            )
            checkpoint_id = ""
            state_id = _content_id("recovery-state", {})
            evidence: tuple[str, ...] = ()
        elif (
            checkpoint.repository_id != repository_id
            or checkpoint.tree_id != tree_id
        ):
            disposition = RecoveryDisposition.FAILED_CLOSED
            reason = "checkpoint_binding_mismatch"
            cursor = checkpoint.cursor
            checkpoint_id = checkpoint.checkpoint_id
            state_id = checkpoint.state_id
            evidence = checkpoint.accepted_merged_tree_evidence
        else:
            checkpoint_roots = dict(checkpoint.semantic_roots)
            if not result_roots:
                result_roots = dict(checkpoint.semantic_roots)
            if not result_proof_index_id:
                result_proof_index_id = checkpoint.proof_index_id
            if not result_cas_invalidation_id:
                result_cas_invalidation_id = checkpoint.cas_invalidation_id
            result_fencing_epoch = max(
                result_fencing_epoch, checkpoint.fencing_epoch
            )
            cursor = checkpoint.cursor
            checkpoint_id = checkpoint.checkpoint_id
            state_id = checkpoint.state_id
            evidence = checkpoint.accepted_merged_tree_evidence
            disposition = RecoveryDisposition.NOOP
            reason = "checkpoint_verified"
            actions.append("load_last_valid_checkpoint")

            if root_reader_failed:
                disposition = RecoveryDisposition.FAILED_CLOSED
                reason = "root_reader_failed"
            elif selected_roots and not checkpoint.semantic_roots:
                disposition = RecoveryDisposition.FAILED_CLOSED
                reason = "checkpoint_semantic_roots_missing"
            elif (
                selected_roots
                and replay_events is None
                and dict(checkpoint.semantic_roots) != selected_roots
            ):
                disposition = RecoveryDisposition.FAILED_CLOSED
                reason = "checkpoint_semantic_roots_mismatch"
            elif (
                current_proof_index_id
                and checkpoint.proof_index_id
                and replay_events is None
                and checkpoint.proof_index_id != current_proof_index_id
            ):
                disposition = RecoveryDisposition.FAILED_CLOSED
                reason = "checkpoint_proof_index_mismatch"
            elif (
                current_cas_invalidation_id
                and checkpoint.cas_invalidation_id
                and replay_events is None
                and checkpoint.cas_invalidation_id
                != current_cas_invalidation_id
            ):
                disposition = RecoveryDisposition.FAILED_CLOSED
                reason = "checkpoint_cas_head_mismatch"

            if (
                disposition != RecoveryDisposition.FAILED_CLOSED
                and (
                current_fencing_token is not None
                or observed_fencing_token is not None
                )
            ):
                if (
                    isinstance(current_fencing_token, bool)
                    or isinstance(observed_fencing_token, bool)
                    or not isinstance(current_fencing_token, int)
                    or not isinstance(observed_fencing_token, int)
                    or current_fencing_token < 1
                    or observed_fencing_token < 1
                ):
                    disposition = RecoveryDisposition.FAILED_CLOSED
                    reason = "invalid_fencing_token"
                elif observed_fencing_token > current_fencing_token:
                    disposition = RecoveryDisposition.FAILED_CLOSED
                    reason = "fencing_epoch_race"
                elif observed_fencing_token < current_fencing_token:
                    if fence_actor is None:
                        # The accepted higher token is itself the persistent
                        # mutation fence; an optional callback additionally
                        # terminates an associated process tree.
                        stale_actor_fenced = True
                    else:
                        try:
                            stale_actor_fenced = (
                                fence_actor(
                                    observed_fencing_token,
                                    current_fencing_token,
                                )
                                is True
                            )
                        except Exception:
                            stale_actor_fenced = False
                    if stale_actor_fenced:
                        actions.append("fence_stale_actor")
                    else:
                        disposition = RecoveryDisposition.FAILED_CLOSED
                        reason = "stale_actor_fence_failed"

            if (
                disposition != RecoveryDisposition.FAILED_CLOSED
                and permit_ids
            ):
                if fence_permits is not None:
                    try:
                        permits_fenced = (
                            fence_permits(
                                permit_ids, int(current_fencing_token or 0)
                            )
                            is True
                        )
                    except Exception:
                        permits_fenced = False
                else:
                    permits_fenced = bool(
                        current_fencing_token is not None
                        and current_fencing_token > checkpoint.fencing_epoch
                    )
                if permits_fenced:
                    actions.append("fence_precrash_permits")
                else:
                    disposition = RecoveryDisposition.FAILED_CLOSED
                    reason = "precrash_permit_fence_failed"

            if (
                disposition != RecoveryDisposition.FAILED_CLOSED
                and event_log_path is not None
            ):
                log_result = recover_jsonl_event_log_tail(
                    event_log_path,
                    checkpoint=checkpoint.cursor,
                    max_quarantine_bytes=self.policy.max_quarantine_bytes,
                )
                if log_result.get("quarantine_path"):
                    quarantined.append(str(log_result["quarantine_path"]))
                if log_result.get("failed_closed"):
                    disposition = RecoveryDisposition.FAILED_CLOSED
                    reason = str(log_result.get("reason") or "event_log_recovery_failed")
                elif log_result.get("repaired"):
                    disposition = RecoveryDisposition.RECOVERED
                    reason = str(log_result.get("reason") or "event_log_repaired")
                    actions.append("repair_event_log_tail")

            if (
                disposition != RecoveryDisposition.FAILED_CLOSED
                and replay_events is not None
            ):
                try:
                    replay_result = replay_events(checkpoint)
                    if not isinstance(replay_result, Mapping):
                        raise TypeError(
                            "replay_events must return a mapping"
                        )
                    raw_cursor = replay_result.get("event_cursor")
                    if isinstance(raw_cursor, EventCursor):
                        replay_cursor = raw_cursor
                    elif isinstance(raw_cursor, str):
                        replay_cursor = EventCursor.from_token(raw_cursor)
                    elif isinstance(raw_cursor, Mapping):
                        replay_cursor = EventCursor.from_dict(raw_cursor)
                    else:
                        raise RecoveryIntegrityError(
                            "replay result is missing its event cursor"
                        )
                    replay_roots = replay_result.get("semantic_roots")
                    if not isinstance(replay_roots, Mapping):
                        raise RecoveryIntegrityError(
                            "replay result is missing semantic roots"
                        )
                    result_roots = {
                        _required_text(key, "replay semantic root kind"):
                        _required_text(item, "replay semantic root identity")
                        for key, item in sorted(replay_roots.items())
                    }
                    result_proof_index_id = str(
                        replay_result.get("proof_index_id")
                        or result_proof_index_id
                    )
                    result_cas_invalidation_id = str(
                        replay_result.get("cas_invalidation_id")
                        or result_cas_invalidation_id
                    )
                    replay_permits = tuple(
                        sorted(
                            {
                                str(item)
                                for item in replay_result.get(
                                    "invalidated_permit_ids", ()
                                )
                                if str(item)
                            }
                        )
                    )
                    if not set(permit_ids).issubset(replay_permits):
                        raise RecoveryIntegrityError(
                            "replay did not fence every pre-crash permit"
                        )
                    if selected_roots and result_roots != selected_roots:
                        raise RecoveryIntegrityError(
                            "replay semantic roots do not match current roots"
                        )
                    if (
                        expected_cursor is None
                        and event_log_path is not None
                    ):
                        expected_cursor = latest_event_cursor(
                            event_log_path
                        )
                    if (
                        expected_cursor is not None
                        and replay_cursor != expected_cursor
                    ):
                        raise RecoveryIntegrityError(
                            "replay cursor does not match the event head"
                        )
                    actions.append("replay_dependency_events")
                    disposition = RecoveryDisposition.RECOVERED
                    reason = "event_replay_verified"
                except Exception as exc:
                    disposition = RecoveryDisposition.FAILED_CLOSED
                    reason = f"event_replay_failed:{type(exc).__name__}"
            elif disposition != RecoveryDisposition.FAILED_CLOSED:
                replay_cursor = checkpoint.cursor
                head_cursor = expected_cursor
                if head_cursor is None and event_log_path is not None:
                    try:
                        head_cursor = latest_event_cursor(event_log_path)
                    except Exception:
                        disposition = RecoveryDisposition.FAILED_CLOSED
                        reason = "event_head_unreadable"
                if (
                    disposition != RecoveryDisposition.FAILED_CLOSED
                    and head_cursor is not None
                    and head_cursor != checkpoint.cursor
                ):
                    disposition = RecoveryDisposition.FAILED_CLOSED
                    reason = "unreplayed_events"
                elif head_cursor is not None:
                    replay_cursor = head_cursor

            if (
                disposition != RecoveryDisposition.FAILED_CLOSED
                and runtime_cas is not None
            ):
                try:
                    audit = runtime_cas.audit(rebuild=True)
                    if not audit.healthy:
                        disposition = RecoveryDisposition.QUARANTINED
                        reason = "corrupt_runtime_cas:" + ",".join(
                            audit.issue_codes
                        )
                    else:
                        actions.append("verify_runtime_cas")
                except Exception as exc:
                    disposition = RecoveryDisposition.QUARANTINED
                    reason = f"runtime_cas_audit_failed:{type(exc).__name__}"

            if (
                disposition
                not in {
                    RecoveryDisposition.FAILED_CLOSED,
                    RecoveryDisposition.QUARANTINED,
                }
                and proof_index is not None
            ):
                try:
                    if canonical_artifacts is None:
                        raise RecoveryIntegrityError(
                            "canonical artifacts are required to restore a proof index"
                        )
                    proof_index.revalidate_canonical_artifacts(
                        canonical_artifacts=canonical_artifacts
                    )
                    if (
                        result_proof_index_id
                        and proof_index.index_id != result_proof_index_id
                    ):
                        raise RecoveryIntegrityError(
                            "restored proof index identity is stale"
                        )
                    result_proof_index_id = proof_index.index_id
                    actions.append("verify_proof_scope_index")
                except Exception as exc:
                    disposition = RecoveryDisposition.QUARANTINED
                    reason = (
                        "stale_restored_artifact:"
                        + type(exc).__name__
                    )

            if (
                disposition
                not in {
                    RecoveryDisposition.FAILED_CLOSED,
                    RecoveryDisposition.QUARANTINED,
                }
                and repair
            ):
                last_error = ""
                succeeded = False
                repair_started = time.monotonic()
                for attempts in range(1, self.policy.max_attempts + 1):
                    try:
                        self.fault_injector.inject(
                            "before_repair",
                            incident_id=incident_id,
                            attempt=attempts,
                        )
                        result = repair(checkpoint, attempts)
                        if (
                            time.monotonic() - repair_started
                            > self.policy.slow_operation_seconds
                        ):
                            last_error = "slow_operation_bound_exceeded"
                            break
                        if result is not False:
                            succeeded = True
                            break
                        last_error = "repair_returned_false"
                    except Exception as exc:  # bounded and recorded, never trusted
                        last_error = type(exc).__name__
                actions.append("run_bounded_repair")
                if succeeded:
                    disposition = RecoveryDisposition.RECOVERED
                    reason = "repair_verified"
                else:
                    disposition = RecoveryDisposition.QUARANTINED
                    reason = f"repair_attempts_exhausted:{last_error}"

            if (
                disposition
                not in {
                    RecoveryDisposition.FAILED_CLOSED,
                    RecoveryDisposition.QUARANTINED,
                }
                and verify is not None
            ):
                try:
                    verified = verify(checkpoint) is True
                except Exception:
                    verified = False
                actions.append("verify_recovered_state")
                if not verified:
                    disposition = RecoveryDisposition.FAILED_CLOSED
                    reason = "recovery_verification_failed"

        if root_reader is not None:
            try:
                root_snapshot_after = {
                    str(key): str(item)
                    for key, item in root_reader().items()
                }
            except Exception:
                root_snapshot_after = None
            if root_snapshot_after is None:
                disposition = RecoveryDisposition.FAILED_CLOSED
                reason = "root_reader_failed"
            elif (
                root_snapshot_before is not None
                and root_snapshot_after != root_snapshot_before
            ):
                disposition = RecoveryDisposition.FAILED_CLOSED
                reason = "root_race"
            elif result_roots and root_snapshot_after != result_roots:
                disposition = RecoveryDisposition.FAILED_CLOSED
                reason = "restored_roots_stale"

        if disposition in {
            RecoveryDisposition.FAILED_CLOSED,
            RecoveryDisposition.QUARANTINED,
        }:
            actions.append("fail_closed")
            quarantined_bytes = 0
            for raw_path in quarantine_paths:
                source = Path(raw_path)
                try:
                    source_size = source.stat().st_size
                except OSError:
                    source_size = 0
                if (
                    quarantined_bytes + source_size
                    > self.policy.max_quarantine_bytes
                ):
                    reason = "quarantine_bound_exceeded"
                    disposition = RecoveryDisposition.FAILED_CLOSED
                    break
                target = self._quarantine_path(source, incident_id)
                if target:
                    quarantined.append(target)
                    quarantined_bytes += source_size

        receipt = RepairReceipt(
            incident_id=incident_id,
            fault=selected_fault,
            disposition=disposition,
            repository_id=repository_id,
            tree_id=tree_id,
            checkpoint_id=checkpoint_id,
            state_id=state_id,
            policy_id=self.policy.policy_id,
            process_tree_id=process_tree_id,
            event_cursor=cursor,
            attempts=attempts,
            actions=tuple(actions),
            reason_code=reason,
            quarantined_paths=tuple(quarantined),
            preserved_evidence_ids=evidence,
            resulting_projection_ids=tuple(resulting_projection_ids),
            stale_actor_fenced=stale_actor_fenced,
            replay_cursor=replay_cursor,
            checkpoint_semantic_roots=checkpoint_roots,
            result_semantic_roots=result_roots,
            precrash_permit_ids=permit_ids,
            invalidated_permit_ids=(
                permit_ids if permits_fenced else ()
            ),
            proof_index_id=result_proof_index_id,
            cas_invalidation_id=result_cas_invalidation_id,
            fencing_epoch=result_fencing_epoch,
            observed_fencing_epoch=int(observed_fencing_token or 0),
            started_at=started_at,
        )
        self._store_receipt(receipt)
        return receipt

    repair = recover


@dataclass(frozen=True)
class ProgrammaticRecoveryPolicy:
    """Policy bounds for the closed ASI-155 deterministic recovery ladder."""

    max_attempts_per_action: int = 2
    max_total_attempts: int = 8
    max_actions: int = 8
    cooldown_ms: int = 30_000
    deadline_ms: int = 120_000
    max_receipt_bytes: int = MAX_RECEIPT_BYTES
    quarantine_on_exhaustion: bool = False

    def __post_init__(self) -> None:
        for name in (
            "max_attempts_per_action",
            "max_total_attempts",
            "max_actions",
            "deadline_ms",
            "max_receipt_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_actions > 32:
            raise ValueError("max_actions exceeds the workflow absolute bound")
        if (
            isinstance(self.cooldown_ms, bool)
            or not isinstance(self.cooldown_ms, int)
            or self.cooldown_ms < 0
        ):
            raise ValueError("cooldown_ms must be a nonnegative integer")
        if not isinstance(self.quarantine_on_exhaustion, bool):
            raise ValueError("quarantine_on_exhaustion must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_attempts_per_action": self.max_attempts_per_action,
            "max_total_attempts": self.max_total_attempts,
            "max_actions": self.max_actions,
            "cooldown_ms": self.cooldown_ms,
            "deadline_ms": self.deadline_ms,
            "max_receipt_bytes": self.max_receipt_bytes,
            "quarantine_on_exhaustion": self.quarantine_on_exhaustion,
        }

    @property
    def policy_cid(self) -> str:
        return prompt_workflow_cid(
            {"schema": "programmatic-recovery-policy@1", **self.to_dict()}
        )


@dataclass(frozen=True)
class RecoveryActionSpec:
    """Complete binding for one operation in the closed recovery catalog."""

    operation: RescueOperation
    preconditions: tuple[str, ...]
    expected_effects: tuple[str, ...]
    compensation: RescueOperation | None
    max_attempts: int
    cooldown_ms: int
    deadline_ms: int
    post_health_required: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.operation, RescueOperation):
            object.__setattr__(
                self, "operation", RescueOperation(str(self.operation))
            )
        if self.compensation is not None and not isinstance(
            self.compensation, RescueOperation
        ):
            object.__setattr__(
                self,
                "compensation",
                RescueOperation(str(self.compensation)),
            )
        for name in ("preconditions", "expected_effects"):
            object.__setattr__(
                self,
                name,
                tuple(_required_text(item, f"{name} item") for item in getattr(self, name)),
            )
        for name in ("max_attempts", "deadline_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.cooldown_ms, bool)
            or not isinstance(self.cooldown_ms, int)
            or self.cooldown_ms < 0
        ):
            raise ValueError("cooldown_ms must be a nonnegative integer")
        if not isinstance(self.post_health_required, bool):
            raise ValueError("post_health_required must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "preconditions": list(self.preconditions),
            "expected_effects": list(self.expected_effects),
            "compensation": (
                self.compensation.value if self.compensation is not None else ""
            ),
            "max_attempts": self.max_attempts,
            "cooldown_ms": self.cooldown_ms,
            "deadline_ms": self.deadline_ms,
            "post_health_required": self.post_health_required,
        }


@dataclass(frozen=True)
class ProgrammaticRecoveryContext:
    """The only input passed to a registered deterministic action handler."""

    incident: SupervisorIncident
    action: RecoveryActionSpec
    attempt: int
    started_at_ms: int
    deadline_at_ms: int


@dataclass(frozen=True)
class RecoveryActionObservation:
    """Observed effects and health returned by a closed action handler."""

    succeeded: bool
    observed_effects: tuple[str, ...] = ()
    post_action_health: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""
    partial: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.succeeded, bool):
            raise ValueError("succeeded must be boolean")
        if not isinstance(self.partial, bool):
            raise ValueError("partial must be boolean")
        object.__setattr__(
            self,
            "observed_effects",
            tuple(
                _required_text(item, "observed effect")
                for item in self.observed_effects
            ),
        )
        if not isinstance(self.post_action_health, Mapping):
            raise TypeError("post_action_health must be a mapping")
        health = json.loads(
            _canonical_bytes(_plain_json(self.post_action_health))
        )
        if not isinstance(health, dict):
            raise TypeError("post_action_health must be a JSON object")
        object.__setattr__(
            self, "post_action_health", _freeze_json(health)
        )
        object.__setattr__(self, "reason", str(self.reason or "").strip())

    @property
    def healthy(self) -> bool:
        return (
            self.post_action_health.get("healthy") is True
            or str(self.post_action_health.get("status") or "").lower()
            in {"healthy", "ok", "recovered"}
        )


@dataclass(frozen=True)
class ProgrammaticRecoveryReceipt:
    """Content-addressed successful or quarantining deterministic action."""

    incident_cid: str
    repository_root_cid: str
    policy_root: str
    run_cid: str
    operation: RescueOperation
    target_id: str
    attempts: tuple[RecoveryAttempt, ...]
    action: RecoveryActionSpec
    disposition: RecoveryDisposition
    observed_effects: tuple[str, ...]
    post_action_health: Mapping[str, Any]
    compensations: tuple[RescueOperation, ...] = ()
    started_at_ms: int = 0
    finished_at_ms: int = 0
    receipt_cid: str = ""

    def __post_init__(self) -> None:
        for name in (
            "incident_cid",
            "repository_root_cid",
            "policy_root",
            "run_cid",
            "target_id",
        ):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )
        if not isinstance(self.operation, RescueOperation):
            object.__setattr__(
                self, "operation", RescueOperation(str(self.operation))
            )
        if not isinstance(self.action, RecoveryActionSpec):
            raise TypeError("action must be a RecoveryActionSpec")
        if self.action.operation is not self.operation:
            raise RecoveryIntegrityError(
                "programmatic recovery operation does not match its action"
            )
        if not self.attempts or not all(
            isinstance(item, RecoveryAttempt) for item in self.attempts
        ):
            raise TypeError("attempts must contain RecoveryAttempt records")
        if (
            self.attempts[-1].operation is not self.operation
            or self.attempts[-1].outcome
            is not RecoveryAttemptOutcome.SUCCEEDED
        ):
            raise RecoveryIntegrityError(
                "programmatic recovery receipt requires a successful final attempt"
            )
        if any(
            item.target_id != self.target_id for item in self.attempts
        ):
            raise RecoveryIntegrityError(
                "programmatic recovery attempts changed target"
            )
        if not isinstance(self.disposition, RecoveryDisposition):
            object.__setattr__(
                self,
                "disposition",
                RecoveryDisposition(str(self.disposition)),
            )
        expected_disposition = (
            RecoveryDisposition.QUARANTINED
            if self.operation is RescueOperation.QUARANTINE
            else RecoveryDisposition.RECOVERED
        )
        if self.disposition is not expected_disposition:
            raise RecoveryIntegrityError(
                "programmatic recovery disposition does not match operation"
            )
        object.__setattr__(
            self,
            "observed_effects",
            tuple(_required_text(item, "observed effect") for item in self.observed_effects),
        )
        if set(self.observed_effects) != set(self.action.expected_effects):
            raise RecoveryIntegrityError(
                "programmatic recovery observed effects do not exactly match"
            )
        if not isinstance(self.post_action_health, Mapping):
            raise TypeError("post_action_health must be a mapping")
        health = json.loads(
            _canonical_bytes(_plain_json(self.post_action_health))
        )
        if not isinstance(health, dict):
            raise TypeError("post_action_health must be a JSON object")
        object.__setattr__(
            self, "post_action_health", _freeze_json(health)
        )
        if not (
            health.get("healthy") is True
            or str(health.get("status") or "").lower()
            in {"healthy", "ok", "recovered"}
        ):
            raise RecoveryIntegrityError(
                "programmatic recovery receipt requires post-action health"
            )
        object.__setattr__(
            self,
            "compensations",
            tuple(
                item
                if isinstance(item, RescueOperation)
                else RescueOperation(str(item))
                for item in self.compensations
            ),
        )
        for name in ("started_at_ms", "finished_at_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        expected = prompt_workflow_cid(self.to_dict(include_id=False))
        if self.receipt_cid and self.receipt_cid != expected:
            raise RecoveryIntegrityError(
                "programmatic recovery receipt identity mismatch"
            )
        object.__setattr__(self, "receipt_cid", expected)

    @property
    def recovered(self) -> bool:
        return self.disposition is RecoveryDisposition.RECOVERED

    @property
    def quarantined(self) -> bool:
        return self.disposition is RecoveryDisposition.QUARANTINED

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result = {
            "schema": PROGRAMMATIC_RECOVERY_RECEIPT_SCHEMA,
            "incident_cid": self.incident_cid,
            "repository_root_cid": self.repository_root_cid,
            "policy_root": self.policy_root,
            "run_cid": self.run_cid,
            "operation": self.operation.value,
            "target_id": self.target_id,
            "attempts": [item.to_record() for item in self.attempts],
            "action": self.action.to_dict(),
            "disposition": self.disposition.value,
            "observed_effects": list(self.observed_effects),
            "post_action_health": _plain_json(self.post_action_health),
            "compensations": [item.value for item in self.compensations],
            "started_at_ms": self.started_at_ms,
            "finished_at_ms": self.finished_at_ms,
        }
        if include_id:
            result["receipt_cid"] = self.receipt_cid
        return result

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "ProgrammaticRecoveryReceipt":
        if value.get("schema") != PROGRAMMATIC_RECOVERY_RECEIPT_SCHEMA:
            raise RecoveryIntegrityError(
                "unsupported programmatic recovery receipt schema"
            )
        raw_action = value.get("action")
        if not isinstance(raw_action, Mapping):
            raise RecoveryIntegrityError(
                "programmatic recovery action binding is missing"
            )
        compensation = str(raw_action.get("compensation") or "")
        return cls(
            incident_cid=str(value.get("incident_cid") or ""),
            repository_root_cid=str(
                value.get("repository_root_cid") or ""
            ),
            policy_root=str(value.get("policy_root") or ""),
            run_cid=str(value.get("run_cid") or ""),
            operation=RescueOperation(str(value.get("operation") or "")),
            target_id=str(value.get("target_id") or ""),
            attempts=tuple(
                RecoveryAttempt.from_dict(item)
                for item in value.get("attempts") or ()
            ),
            action=RecoveryActionSpec(
                operation=RescueOperation(
                    str(raw_action.get("operation") or "")
                ),
                preconditions=tuple(raw_action.get("preconditions") or ()),
                expected_effects=tuple(
                    raw_action.get("expected_effects") or ()
                ),
                compensation=(
                    RescueOperation(compensation) if compensation else None
                ),
                max_attempts=raw_action.get("max_attempts"),  # type: ignore[arg-type]
                cooldown_ms=raw_action.get("cooldown_ms"),  # type: ignore[arg-type]
                deadline_ms=raw_action.get("deadline_ms"),  # type: ignore[arg-type]
                post_health_required=raw_action.get(
                    "post_health_required", True
                ),
            ),
            disposition=RecoveryDisposition(
                str(value.get("disposition") or "")
            ),
            observed_effects=tuple(value.get("observed_effects") or ()),
            post_action_health=value.get("post_action_health") or {},
            compensations=tuple(
                RescueOperation(str(item))
                for item in value.get("compensations") or ()
            ),
            started_at_ms=value.get("started_at_ms", 0),
            finished_at_ms=value.get("finished_at_ms", 0),
            receipt_cid=str(value.get("receipt_cid") or ""),
        )


@dataclass(frozen=True)
class ProgrammaticRecoveryResult:
    """One terminal result; successful recovery, quarantine, or exhaustion."""

    diagnosis: RecoveryDiagnosis
    attempts: tuple[RecoveryAttempt, ...]
    receipt: ProgrammaticRecoveryReceipt | None = None
    exhaustion_receipt: ProgrammaticRecoveryExhaustionReceipt | None = None
    deduplicated: bool = False

    def __post_init__(self) -> None:
        if (self.receipt is None) == (self.exhaustion_receipt is None):
            raise ValueError(
                "exactly one recovery or exhaustion receipt is required"
            )

    @property
    def recovered(self) -> bool:
        return bool(self.receipt is not None and self.receipt.recovered)

    @property
    def quarantined(self) -> bool:
        return bool(
            (self.receipt is not None and self.receipt.quarantined)
            or (
                self.exhaustion_receipt is not None
                and self.exhaustion_receipt.status
                is RecordStatus.QUARANTINED
            )
        )

    @property
    def terminal_cid(self) -> str:
        if self.receipt is not None:
            return self.receipt.receipt_cid
        assert self.exhaustion_receipt is not None
        return self.exhaustion_receipt.receipt_cid


def _catalog_spec(
    operation: RescueOperation,
    *,
    preconditions: Sequence[str],
    effects: Sequence[str],
    compensation: RescueOperation | None = None,
) -> tuple[
    RescueOperation,
    tuple[str, ...],
    tuple[str, ...],
    RescueOperation | None,
]:
    return (
        operation,
        tuple(preconditions),
        tuple(effects),
        compensation,
    )


_RECOVERY_LADDER: Final = MappingProxyType(
    {
        IncidentKind.STALE_PROJECTION: (
            _catalog_spec(
                RescueOperation.RECONCILE_PROJECTION,
                preconditions=("projection_stale", "live_state_healthy"),
                effects=("projection_reconciled",),
            ),
        ),
        IncidentKind.STALE_LIFECYCLE: (
            _catalog_spec(
                RescueOperation.REPAIR_LIFECYCLE_STATE,
                preconditions=("lifecycle_state_stale",),
                effects=("lifecycle_state_reconciled",),
            ),
        ),
        IncidentKind.STALE_LEASE: (
            _catalog_spec(
                RescueOperation.REPAIR_EXPIRED_LEASE,
                preconditions=("lease_expired", "fence_current"),
                effects=("lease_expired", "stale_owner_fenced"),
            ),
        ),
        IncidentKind.ORPHANED_LOCK: (
            _catalog_spec(
                RescueOperation.REPAIR_ORPHANED_LOCK,
                preconditions=("lock_orphaned", "owner_not_live"),
                effects=("lock_expired",),
            ),
        ),
        IncidentKind.CONSUMED_ATTEMPT: (
            _catalog_spec(
                RescueOperation.RETRY,
                preconditions=("attempt_consumed", "retry_permitted"),
                effects=("attempt_expired", "task_requeued"),
            ),
        ),
        IncidentKind.STALE_HEARTBEAT: (
            _catalog_spec(
                RescueOperation.RETRY,
                preconditions=("progress_signal_stale", "retry_permitted"),
                effects=("task_requeued",),
            ),
            _catalog_spec(
                RescueOperation.RESTART_LANE,
                preconditions=("progress_signal_stale", "lane_identity_current"),
                effects=("old_lane_fenced", "lane_restarted"),
                compensation=RescueOperation.STOP,
            ),
        ),
        IncidentKind.LANE_FAILURE: (
            _catalog_spec(
                RescueOperation.RETRY,
                preconditions=("task_retryable",),
                effects=("task_requeued",),
            ),
            _catalog_spec(
                RescueOperation.RESTART_LANE,
                preconditions=("lane_failed", "lane_identity_current"),
                effects=("old_lane_fenced", "lane_restarted"),
                compensation=RescueOperation.STOP,
            ),
        ),
        IncidentKind.DIRTY_WORKTREE: (
            _catalog_spec(
                RescueOperation.RESCUE_DIRTY_WORK,
                preconditions=("worktree_dirty", "worktree_owned"),
                effects=("dirty_work_preserved", "recovery_branch_created"),
            ),
            _catalog_spec(
                RescueOperation.RECONCILE_WORKTREE,
                preconditions=("dirty_work_preserved",),
                effects=("worktree_reconciled",),
            ),
        ),
        IncidentKind.VALIDATION_FAILURE: (
            _catalog_spec(
                RescueOperation.VALIDATION_REPLAY,
                preconditions=("validation_interrupted_or_retryable",),
                effects=("validation_replayed",),
            ),
        ),
        IncidentKind.MERGE_FAILURE: (
            _catalog_spec(
                RescueOperation.RECONCILE_WORKTREE,
                preconditions=("merge_interrupted_or_retryable",),
                effects=("merge_replayed",),
            ),
        ),
        IncidentKind.CORRUPT_TASK_SOURCE: (
            _catalog_spec(
                RescueOperation.QUARANTINE,
                preconditions=("task_source_corrupt", "scope_exact"),
                effects=("corrupt_scope_quarantined",),
            ),
        ),
        IncidentKind.RESOURCE_EXHAUSTION: (
            _catalog_spec(
                RescueOperation.QUARANTINE,
                preconditions=("resource_exhausted", "scope_exact"),
                effects=("affected_scope_quarantined",),
            ),
        ),
        IncidentKind.PROVIDER_UNAVAILABLE: (
            _catalog_spec(
                RescueOperation.REASSIGN_INDEPENDENT_WORK,
                preconditions=("provider_unavailable", "work_independent"),
                effects=("independent_work_reassigned",),
            ),
        ),
        IncidentKind.SPLIT_BRAIN: (
            _catalog_spec(
                RescueOperation.QUARANTINE,
                preconditions=("multiple_live_owners", "scope_exact"),
                effects=("split_brain_scope_quarantined",),
            ),
        ),
        IncidentKind.UNKNOWN: (
            _catalog_spec(
                RescueOperation.OBJECTIVE_RECONCILE,
                preconditions=("typed_objective_gap", "policy_permits"),
                effects=("objectives_reconciled",),
            ),
            _catalog_spec(
                RescueOperation.BACKLOG_REFILL,
                preconditions=("typed_backlog_empty", "policy_permits"),
                effects=("backlog_refilled",),
            ),
        ),
    }
)
PROGRAMMATIC_RECOVERY_ACTION_CATALOG: Final = _RECOVERY_LADDER
PROGRAMMATIC_RECOVERY_CATALOG_CID: Final = prompt_workflow_cid(
    {
        "schema": "programmatic-recovery-action-catalog@1",
        "ladder": {
            kind.value: [
                {
                    "operation": operation.value,
                    "preconditions": list(preconditions),
                    "expected_effects": list(effects),
                    "compensation": (
                        compensation.value
                        if compensation is not None
                        else ""
                    ),
                }
                for operation, preconditions, effects, compensation in specs
            ]
            for kind, specs in sorted(
                _RECOVERY_LADDER.items(), key=lambda item: item[0].value
            )
        },
    }
)


def _serialize_semantic_incident(
    method: Callable[..., ProgrammaticRecoveryResult],
):
    """Deduplicate one semantic incident across threads and processes."""

    @wraps(method)
    def wrapped(
        self: "ProgrammaticRecoveryController",
        diagnosis: RecoveryDiagnosis,
        *args: Any,
        **kwargs: Any,
    ) -> ProgrammaticRecoveryResult:
        if not isinstance(diagnosis, RecoveryDiagnosis):
            raise TypeError("diagnosis must be a RecoveryDiagnosis")
        digest = hashlib.sha256(
            diagnosis.incident_cid.encode("utf-8")
        ).hexdigest()
        lock_path = (
            self.state_dir / "programmatic-locks" / f"{digest}.lock"
        )
        with self._thread_lock:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+b") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    return method(
                        self, diagnosis, *args, **kwargs
                    )
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    return wrapped


class ProgrammaticRecoveryController:
    """Execute the least-invasive registered action without invoking a model."""

    def __init__(
        self,
        state_dir: Path | str,
        *,
        handlers: Mapping[
            RescueOperation | str,
            Callable[[ProgrammaticRecoveryContext], Any],
        ]
        | None = None,
        health_check: Callable[
            [ProgrammaticRecoveryContext, RecoveryActionObservation], Any
        ]
        | None = None,
        policy: ProgrammaticRecoveryPolicy | None = None,
        clock_ms: Callable[[], int] | None = None,
        fault_injector: FaultInjector | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.policy = policy or ProgrammaticRecoveryPolicy()
        self.handlers = {
            (
                key
                if isinstance(key, RescueOperation)
                else RescueOperation(str(key))
            ): handler
            for key, handler in (handlers or {}).items()
        }
        if not all(callable(item) for item in self.handlers.values()):
            raise TypeError("every recovery handler must be callable")
        self.health_check = health_check
        self.clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self.fault_injector = fault_injector or FaultInjector()
        self.results_dir = self.state_dir / "programmatic-recovery"
        self._thread_lock = threading.RLock()

    def _result_path(self, incident_cid: str) -> Path:
        digest = hashlib.sha256(
            (
                incident_cid
                + ":"
                + self.policy.policy_cid
                + ":"
                + PROGRAMMATIC_RECOVERY_CATALOG_CID
            ).encode("utf-8")
        ).hexdigest()
        return self.results_dir / f"{digest}.json"

    def _load(
        self, diagnosis: RecoveryDiagnosis
    ) -> ProgrammaticRecoveryResult | None:
        path = self._result_path(diagnosis.incident_cid)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        if len(raw) > self.policy.max_receipt_bytes:
            raise RecoveryIntegrityError(
                "programmatic recovery result exceeds byte bound"
            )
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RecoveryIntegrityError(
                "programmatic recovery result is malformed"
            ) from exc
        if not isinstance(value, Mapping):
            raise RecoveryIntegrityError(
                "programmatic recovery result must be an object"
            )
        if value.get("schema") != "programmatic-recovery-result@1":
            raise RecoveryIntegrityError(
                "unsupported programmatic recovery result schema"
            )
        if value.get("incident_cid") != diagnosis.incident_cid:
            raise RecoveryIntegrityError(
                "programmatic recovery result incident mismatch"
            )
        if value.get("recovery_policy_cid") != self.policy.policy_cid:
            raise RecoveryIntegrityError(
                "programmatic recovery result policy mismatch"
            )
        if (
            value.get("recovery_catalog_cid")
            != PROGRAMMATIC_RECOVERY_CATALOG_CID
        ):
            raise RecoveryIntegrityError(
                "programmatic recovery result catalog mismatch"
            )
        kind = value.get("terminal_kind")
        terminal = value.get("terminal")
        if not isinstance(terminal, Mapping):
            raise RecoveryIntegrityError(
                "programmatic recovery terminal receipt is missing"
            )
        if kind == "recovery":
            receipt = ProgrammaticRecoveryReceipt.from_dict(terminal)
            self._verify_terminal_bindings(
                diagnosis,
                incident_cid=receipt.incident_cid,
                repository_root_cid=receipt.repository_root_cid,
                policy_root=receipt.policy_root,
                run_cid=receipt.run_cid,
            )
            return ProgrammaticRecoveryResult(
                diagnosis=diagnosis,
                attempts=receipt.attempts,
                receipt=receipt,
                deduplicated=True,
            )
        if kind == "exhaustion":
            exhaustion = ProgrammaticRecoveryExhaustionReceipt.from_dict(
                terminal
            )
            self._verify_terminal_bindings(
                diagnosis,
                incident_cid=exhaustion.incident_cid,
                repository_root_cid=exhaustion.repository_root_cid,
                policy_root=exhaustion.policy_root,
                run_cid=exhaustion.run_cid,
            )
            if (
                exhaustion.exhaustion_reason
                == "action_cooldown_active"
                and not self._any_action_in_cooldown(
                    diagnosis, self.clock_ms()
                )
            ):
                return None
            return ProgrammaticRecoveryResult(
                diagnosis=diagnosis,
                attempts=exhaustion.attempts,
                exhaustion_receipt=exhaustion,
                deduplicated=True,
            )
        raise RecoveryIntegrityError(
            "programmatic recovery terminal kind is unsupported"
        )

    @staticmethod
    def _verify_terminal_bindings(
        diagnosis: RecoveryDiagnosis,
        *,
        incident_cid: str,
        repository_root_cid: str,
        policy_root: str,
        run_cid: str,
    ) -> None:
        incident = diagnosis.incident
        if (
            incident_cid != diagnosis.incident_cid
            or repository_root_cid != incident.repository_root_cid
            or policy_root != incident.policy_root
            or run_cid != incident.run_cid
        ):
            raise RecoveryIntegrityError(
                "programmatic recovery terminal binding mismatch"
            )

    def _store(
        self,
        result: ProgrammaticRecoveryResult,
    ) -> None:
        if result.receipt is not None:
            kind = "recovery"
            terminal = result.receipt.to_dict()
        else:
            assert result.exhaustion_receipt is not None
            kind = "exhaustion"
            terminal = result.exhaustion_receipt.to_record()
        payload = _canonical_bytes(
            {
                "schema": "programmatic-recovery-result@1",
                "incident_cid": result.diagnosis.incident_cid,
                "recovery_policy_cid": self.policy.policy_cid,
                "recovery_catalog_cid": PROGRAMMATIC_RECOVERY_CATALOG_CID,
                "terminal_kind": kind,
                "terminal": terminal,
            }
        ) + b"\n"
        if len(payload) > self.policy.max_receipt_bytes:
            raise RecoveryBoundExceeded(
                "programmatic recovery result exceeds byte bound"
            )
        _atomic_write(
            self._result_path(result.diagnosis.incident_cid), payload
        )

    def _specs(
        self, diagnosis: RecoveryDiagnosis
    ) -> tuple[RecoveryActionSpec, ...]:
        raw_specs = _RECOVERY_LADDER[diagnosis.kind][
            : self.policy.max_actions
        ]
        return tuple(
            RecoveryActionSpec(
                operation=operation,
                preconditions=preconditions,
                expected_effects=effects,
                compensation=compensation,
                max_attempts=self.policy.max_attempts_per_action,
                cooldown_ms=self.policy.cooldown_ms,
                deadline_ms=self.policy.deadline_ms,
            )
            for operation, preconditions, effects, compensation in raw_specs
        )

    @staticmethod
    def _observation(
        raw: Any, spec: RecoveryActionSpec
    ) -> RecoveryActionObservation:
        if isinstance(raw, RecoveryActionObservation):
            return raw
        if isinstance(raw, bool):
            if raw:
                raise RecoveryIntegrityError(
                    "successful recovery requires explicit effects and health"
                )
            return RecoveryActionObservation(
                succeeded=False,
                observed_effects=(),
                post_action_health={"healthy": False},
                reason="handler_attestation",
            )
        if isinstance(raw, Mapping):
            return RecoveryActionObservation(
                succeeded=raw.get("succeeded", raw.get("success", False)),
                observed_effects=tuple(
                    raw.get(
                        "observed_effects",
                        raw.get("effects", ()),
                    )
                    or ()
                ),
                post_action_health=raw.get(
                    "post_action_health", raw.get("health", {})
                )
                or {},
                reason=str(raw.get("reason") or ""),
                partial=raw.get("partial", False),
            )
        raise TypeError(
            "recovery handler must return bool, mapping, or observation"
        )

    def _in_cooldown(
        self,
        diagnosis: RecoveryDiagnosis,
        operation: RescueOperation,
        now_ms: int,
        cooldown_ms: int,
    ) -> bool:
        for item in diagnosis.evidence:
            if item.kind.value != "prior_action":
                continue
            if str(item.value.get("operation") or "") != operation.value:
                continue
            finished = item.value.get(
                "finished_at_ms", item.value.get("updated_at_ms", 0)
            )
            if (
                isinstance(finished, int)
                and not isinstance(finished, bool)
                and finished <= now_ms < finished + cooldown_ms
            ):
                return True
        return False

    def _any_action_in_cooldown(
        self,
        diagnosis: RecoveryDiagnosis,
        now_ms: int,
    ) -> bool:
        return any(
            self._in_cooldown(
                diagnosis,
                spec.operation,
                now_ms,
                spec.cooldown_ms,
            )
            for spec in self._specs(diagnosis)
        )

    @staticmethod
    def _preconditions_permit(
        diagnosis: RecoveryDiagnosis,
        spec: RecoveryActionSpec,
    ) -> bool:
        """Reject contradictory evidence before an effectful handler runs."""

        evidence = tuple(item.value for item in diagnosis.evidence)
        denial_keys = {
            "fence_current": ("fence_current",),
            "lane_identity_current": ("lane_identity_current",),
            "owner_not_live": ("owner_not_live",),
            "policy_permits": ("policy_permits",),
            "retry_permitted": ("retry_permitted",),
            "scope_exact": ("scope_exact",),
            "work_independent": ("work_independent",),
            "worktree_owned": ("worktree_owned",),
        }
        for precondition in spec.preconditions:
            keys = denial_keys.get(precondition, ())
            if any(
                key in record and record.get(key) is False
                for record in evidence
                for key in keys
            ):
                return False
            if precondition == "owner_not_live" and any(
                record.get("owner_live") is True for record in evidence
            ):
                return False
        if spec.operation is RescueOperation.OBJECTIVE_RECONCILE:
            return any(
                record.get("objective_gap") is True
                or record.get("objective_reconcile_required") is True
                for record in evidence
            ) and any(
                record.get("policy_permits") is True
                for record in evidence
            )
        if spec.operation is RescueOperation.BACKLOG_REFILL:
            return any(
                record.get("backlog_empty") is True
                or record.get("backlog_refill_required") is True
                for record in evidence
            ) and any(
                record.get("policy_permits") is True
                for record in evidence
            )
        return True

    def _attempt_receipt(
        self,
        *,
        diagnosis: RecoveryDiagnosis,
        spec: RecoveryActionSpec,
        attempt: int,
        outcome: RecoveryAttemptOutcome,
        started_at_ms: int,
        finished_at_ms: int,
        observation: RecoveryActionObservation | None,
        error: str = "",
    ) -> RecoveryAttempt:
        payload = {
            "schema": "programmatic-recovery-action-observation@1",
            "incident_cid": diagnosis.incident_cid,
            "operation": spec.operation.value,
            "target_id": diagnosis.target_ids[0],
            "attempt": attempt,
            "outcome": outcome.value,
            "observation": (
                {
                    "succeeded": observation.succeeded,
                    "observed_effects": list(
                        observation.observed_effects
                    ),
                    "post_action_health": _plain_json(
                        observation.post_action_health
                    ),
                    "reason": observation.reason,
                    "partial": observation.partial,
                }
                if observation is not None
                else {"error": error}
            ),
        }
        return RecoveryAttempt(
            operation=spec.operation,
            target_id=diagnosis.target_ids[0],
            attempt=attempt,
            outcome=outcome,
            receipt_cid=prompt_workflow_cid(payload),
            failure_fingerprint=(
                ""
                if outcome is RecoveryAttemptOutcome.SUCCEEDED
                else diagnosis.incident.failure_fingerprint
            ),
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
        )

    @_serialize_semantic_incident
    def recover(
        self,
        diagnosis: RecoveryDiagnosis,
        *,
        handlers: Mapping[
            RescueOperation | str,
            Callable[[ProgrammaticRecoveryContext], Any],
        ]
        | None = None,
    ) -> ProgrammaticRecoveryResult:
        """Run the finite ladder and persist one terminal incident-bound result."""

        if not isinstance(diagnosis, RecoveryDiagnosis):
            raise TypeError("diagnosis must be a RecoveryDiagnosis")
        selected_handlers = dict(self.handlers)
        for key, handler in (handlers or {}).items():
            operation = (
                key
                if isinstance(key, RescueOperation)
                else RescueOperation(str(key))
            )
            if not callable(handler):
                raise TypeError("every recovery handler must be callable")
            selected_handlers[operation] = handler

        with self._thread_lock:
            existing = self._load(diagnosis)
            if existing is not None:
                return existing
            started = self.clock_ms()
            if (
                isinstance(started, bool)
                or not isinstance(started, int)
                or started < 0
            ):
                raise RecoveryIntegrityError(
                    "recovery clock returned an invalid timestamp"
                )
            overall_deadline = started + self.policy.deadline_ms
            attempts: list[RecoveryAttempt] = []
            compensations: list[RescueOperation] = []
            inapplicable: set[RescueOperation] = set()
            last_reason = "no_applicable_registered_action"
            total_attempts = 0
            halt_recovery = False

            for spec in self._specs(diagnosis):
                handler = selected_handlers.get(spec.operation)
                now = self.clock_ms()
                if handler is None:
                    inapplicable.add(spec.operation)
                    continue
                if not self._preconditions_permit(diagnosis, spec):
                    inapplicable.add(spec.operation)
                    last_reason = "action_precondition_denied"
                    continue
                if self._in_cooldown(
                    diagnosis,
                    spec.operation,
                    now,
                    spec.cooldown_ms,
                ):
                    inapplicable.add(spec.operation)
                    last_reason = "action_cooldown_active"
                    continue
                action_deadline = min(
                    overall_deadline, now + spec.deadline_ms
                )
                for action_attempt in range(1, spec.max_attempts + 1):
                    if total_attempts >= self.policy.max_total_attempts:
                        last_reason = "total_attempt_bound_exhausted"
                        break
                    attempt_started = self.clock_ms()
                    if attempt_started >= action_deadline:
                        last_reason = "recovery_deadline_exhausted"
                        break
                    total_attempts += 1
                    context = ProgrammaticRecoveryContext(
                        incident=diagnosis.incident,
                        action=spec,
                        attempt=action_attempt,
                        started_at_ms=attempt_started,
                        deadline_at_ms=action_deadline,
                    )
                    observation: RecoveryActionObservation | None = None
                    error = ""
                    try:
                        self.fault_injector.inject(
                            "before_programmatic_recovery_action",
                            incident_cid=diagnosis.incident_cid,
                            operation=spec.operation.value,
                            attempt=action_attempt,
                        )
                        observation = self._observation(
                            handler(context), spec
                        )
                    except Exception as exc:
                        error = type(exc).__name__
                    finished = self.clock_ms()
                    timed_out = finished >= action_deadline
                    expected = set(spec.expected_effects)
                    observed = (
                        set(observation.observed_effects)
                        if observation is not None
                        else set()
                    )
                    effects_match = expected == observed
                    health_ok = bool(
                        observation is not None
                        and observation.healthy
                    )
                    if observation is not None and self.health_check is not None:
                        try:
                            checked = self.health_check(
                                context, observation
                            )
                            if isinstance(checked, Mapping):
                                observation = RecoveryActionObservation(
                                    succeeded=observation.succeeded,
                                    observed_effects=observation.observed_effects,
                                    post_action_health=checked,
                                    reason=observation.reason,
                                    partial=observation.partial,
                                )
                                health_ok = observation.healthy
                            else:
                                health_ok = checked is True
                        except Exception:
                            health_ok = False
                    succeeded = bool(
                        observation is not None
                        and observation.succeeded
                        and effects_match
                        and (health_ok or not spec.post_health_required)
                        and not timed_out
                    )
                    outcome = (
                        RecoveryAttemptOutcome.SUCCEEDED
                        if succeeded
                        else (
                            RecoveryAttemptOutcome.TIMED_OUT
                            if timed_out
                            else RecoveryAttemptOutcome.FAILED
                        )
                    )
                    attempt_record = self._attempt_receipt(
                        diagnosis=diagnosis,
                        spec=spec,
                        attempt=action_attempt,
                        outcome=outcome,
                        started_at_ms=attempt_started,
                        finished_at_ms=finished,
                        observation=observation,
                        error=error,
                    )
                    attempts.append(attempt_record)
                    if succeeded:
                        assert observation is not None
                        disposition = (
                            RecoveryDisposition.QUARANTINED
                            if spec.operation is RescueOperation.QUARANTINE
                            else RecoveryDisposition.RECOVERED
                        )
                        receipt = ProgrammaticRecoveryReceipt(
                            incident_cid=diagnosis.incident_cid,
                            repository_root_cid=(
                                diagnosis.incident.repository_root_cid
                            ),
                            policy_root=diagnosis.incident.policy_root,
                            run_cid=diagnosis.incident.run_cid,
                            operation=spec.operation,
                            target_id=diagnosis.target_ids[0],
                            attempts=tuple(attempts),
                            action=spec,
                            disposition=disposition,
                            observed_effects=(
                                observation.observed_effects
                            ),
                            post_action_health=(
                                observation.post_action_health
                            ),
                            compensations=tuple(compensations),
                            started_at_ms=started,
                            finished_at_ms=finished,
                        )
                        result = ProgrammaticRecoveryResult(
                            diagnosis=diagnosis,
                            attempts=tuple(attempts),
                            receipt=receipt,
                        )
                        self._store(result)
                        return result

                    last_reason = (
                        "action_timed_out"
                        if timed_out
                        else (
                            "post_action_health_failed"
                            if observation is not None
                            and observation.succeeded
                            and effects_match
                            else (
                                "expected_effects_missing"
                                if observation is not None
                                and observation.succeeded
                                else "action_failed:"
                                + (
                                    error
                                    or (
                                        observation.reason
                                        if observation
                                        else "unknown"
                                    )
                                )
                            )
                        )
                    )
                    if (
                        observation is not None
                        and spec.compensation is not None
                        and (
                            observation.partial
                            or bool(observation.observed_effects)
                        )
                    ):
                        compensation = selected_handlers.get(
                            spec.compensation
                        )
                        if compensation is None:
                            last_reason = "compensation_unavailable"
                            halt_recovery = True
                            break
                        else:
                            if (
                                total_attempts
                                >= self.policy.max_total_attempts
                                or finished >= action_deadline
                            ):
                                last_reason = (
                                    "compensation_bound_exhausted"
                                )
                                halt_recovery = True
                                break
                            compensation_spec = RecoveryActionSpec(
                                operation=spec.compensation,
                                preconditions=("partial_effect_observed",),
                                expected_effects=("partial_effect_compensated",),
                                compensation=None,
                                max_attempts=1,
                                cooldown_ms=0,
                                deadline_ms=max(
                                    1, action_deadline - finished
                                ),
                            )
                            total_attempts += 1
                            compensation_observation = None
                            compensation_error = ""
                            try:
                                compensation_context = (
                                    ProgrammaticRecoveryContext(
                                        incident=diagnosis.incident,
                                        action=compensation_spec,
                                        attempt=1,
                                        started_at_ms=finished,
                                        deadline_at_ms=action_deadline,
                                    )
                                )
                                compensation_observation = self._observation(
                                    compensation(compensation_context),
                                    compensation_spec,
                                )
                            except Exception as exc:
                                compensation_error = type(exc).__name__
                            compensation_finished = self.clock_ms()
                            compensation_succeeded = bool(
                                compensation_observation is not None
                                and compensation_observation.succeeded
                                and set(
                                    compensation_observation.observed_effects
                                )
                                == set(
                                    compensation_spec.expected_effects
                                )
                                and compensation_observation.healthy
                                and compensation_finished < action_deadline
                            )
                            attempts.append(
                                self._attempt_receipt(
                                    diagnosis=diagnosis,
                                    spec=compensation_spec,
                                    attempt=1,
                                    outcome=(
                                        RecoveryAttemptOutcome.SUCCEEDED
                                        if compensation_succeeded
                                        else (
                                            RecoveryAttemptOutcome.TIMED_OUT
                                            if compensation_finished
                                            >= action_deadline
                                            else RecoveryAttemptOutcome.FAILED
                                        )
                                    ),
                                    started_at_ms=finished,
                                    finished_at_ms=compensation_finished,
                                    observation=compensation_observation,
                                    error=compensation_error,
                                )
                            )
                            if compensation_succeeded:
                                compensations.append(spec.compensation)
                            else:
                                last_reason = "compensation_failed"
                                halt_recovery = True
                                break
                if halt_recovery:
                    break
                if total_attempts >= self.policy.max_total_attempts:
                    break
                if self.clock_ms() >= overall_deadline:
                    last_reason = "recovery_deadline_exhausted"
                    break

            finished = self.clock_ms()
            exhaustion = ProgrammaticRecoveryExhaustionReceipt(
                incident_cid=diagnosis.incident_cid,
                repository_root_cid=(
                    diagnosis.incident.repository_root_cid
                ),
                policy_root=diagnosis.incident.policy_root,
                run_cid=diagnosis.incident.run_cid,
                attempts=tuple(attempts),
                inapplicable_operations=tuple(inapplicable),
                exhaustion_reason=last_reason,
                budget=PromptWorkflowBudget(
                    max_latency_ms=min(
                        self.policy.deadline_ms, 3_600_000
                    ),
                    max_rescue_actions=self.policy.max_actions,
                ),
                circuit_open=(
                    total_attempts >= self.policy.max_total_attempts
                    or finished >= overall_deadline
                ),
                status=RecordStatus.QUARANTINED,
                created_at_ms=started,
                updated_at_ms=finished,
            )
            result = ProgrammaticRecoveryResult(
                diagnosis=diagnosis,
                attempts=tuple(attempts),
                exhaustion_receipt=exhaustion,
            )
            self._store(result)
            return result

    execute = recover


ProgrammaticRecovery = ProgrammaticRecoveryController
BoundedProgrammaticRecovery = ProgrammaticRecoveryController
ProgrammaticRecoveryEngine = ProgrammaticRecoveryController
ProgrammaticRecoveryExecutor = ProgrammaticRecoveryController
BoundRecoveryAction = RecoveryActionSpec
RecoveryAction = RecoveryActionSpec
RecoveryActionResult = RecoveryActionObservation


def recover_programmatically(
    diagnosis: RecoveryDiagnosis,
    *,
    state_dir: Path | str,
    handlers: Mapping[
        RescueOperation | str,
        Callable[[ProgrammaticRecoveryContext], Any],
    ],
    health_check: Callable[
        [ProgrammaticRecoveryContext, RecoveryActionObservation], Any
    ]
    | None = None,
    policy: ProgrammaticRecoveryPolicy | None = None,
    clock_ms: Callable[[], int] | None = None,
    fault_injector: FaultInjector | None = None,
) -> ProgrammaticRecoveryResult:
    """Convenience entry point for a single deterministic recovery pass."""

    return ProgrammaticRecoveryController(
        state_dir,
        handlers=handlers,
        health_check=health_check,
        policy=policy,
        clock_ms=clock_ms,
        fault_injector=fault_injector,
    ).recover(diagnosis)


def verify_repair_receipt(
    value: RepairReceipt | Mapping[str, Any],
    *,
    repository_id: str | None = None,
    tree_id: str | None = None,
) -> RepairReceipt:
    """Validate identity and optional current-tree bindings."""

    receipt = (
        value if isinstance(value, RepairReceipt) else RepairReceipt.from_dict(value)
    )
    if repository_id is not None and receipt.repository_id != repository_id:
        raise RecoveryIntegrityError("repair receipt repository binding mismatch")
    if tree_id is not None and receipt.tree_id != tree_id:
        raise RecoveryIntegrityError("repair receipt tree binding mismatch")
    return receipt


__all__ = [
    "BOUNDED_RECOVERY_REQUIREMENT_ID",
    "BoundRecoveryAction",
    "BoundedProgrammaticRecovery",
    "FaultInjector",
    "PROGRAMMATIC_RECOVERY_ACTION_CATALOG",
    "PROGRAMMATIC_RECOVERY_CATALOG_CID",
    "PROGRAMMATIC_RECOVERY_RECEIPT_SCHEMA",
    "RECOVERY_CHECKPOINT_SCHEMA",
    "RECOVERY_INCIDENT_SCHEMA",
    "REPAIR_RECEIPT_SCHEMA",
    "ProgrammaticRecovery",
    "ProgrammaticRecoveryContext",
    "ProgrammaticRecoveryController",
    "ProgrammaticRecoveryEngine",
    "ProgrammaticRecoveryExecutor",
    "ProgrammaticRecoveryPolicy",
    "ProgrammaticRecoveryReceipt",
    "ProgrammaticRecoveryResult",
    "RecoveryActionObservation",
    "RecoveryAction",
    "RecoveryActionResult",
    "RecoveryActionSpec",
    "RecoveryBoundExceeded",
    "RecoveryCheckpoint",
    "RecoveryCheckpointStore",
    "RecoveryDisposition",
    "RecoveryError",
    "RecoveryFault",
    "RecoveryIntegrityError",
    "RecoveryPolicy",
    "RepairReceipt",
    "SupervisorRecovery",
    "recover_programmatically",
    "verify_repair_receipt",
]
