"""Database-derived watchdog, stall diagnostics, and safe fenced repair.

DQP-032 / DatabaseWatchdog@1, StallDiagnosis@1, FencedRecoveryCommand@1
=====================================================================

:class:`DatabaseWatchdog` classifies control-plane health from durable
database facts (sessions, heartbeats, claims, attempts, events, worktrees,
merges, provider capacity, and server identity) plus exact process-birth
observations. It never treats file mtime/age alone as authority for action.

Authority rules (fail-closed)
-----------------------------
* Repair requires the current expected fence, process birth, and store
  generation; stale or mismatched fencing is rejected.
* Identical idempotency keys replay the durable result without side effects.
* File age / status-file mtime may appear as evidence but never alone authorize
  a repair, signal, lock deletion, or restart.
* Ready work without a valid owner, capacity, or dependency explanation is
  classified as actionable (``ready_unclaimable``).
* When ownership is unknown the doctor exposes evidence and abstains; no raw
  PID signal or lock deletion is performed.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
)
from ..runtime.daemon_registry import (
    process_birth_from_mapping,
    process_birth_id,
    process_births_match,
)
from ..task_sources.duckdb_state import open_duckdb_connection
from ..task_sources.task_identity import canonical_json_bytes

# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_WATCHDOG_INTERFACE: Final[str] = "DatabaseWatchdog@1"
STALL_DIAGNOSIS_INTERFACE: Final[str] = "StallDiagnosis@1"
FENCED_RECOVERY_COMMAND_INTERFACE: Final[str] = "FencedRecoveryCommand@1"

DATABASE_WATCHDOG_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-watchdog@1"
)
STALL_DIAGNOSIS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/stall-diagnosis@1"
)
FENCED_RECOVERY_COMMAND_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/fenced-recovery-command@1"
)
WATCHDOG_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/watchdog-evidence@1"
)
WATCHDOG_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/watchdog-observation@1"
)

DEFAULT_HEARTBEAT_STALE_MS: Final[int] = 30_000
DEFAULT_SESSION_EXPIRY_MS: Final[int] = 60_000
DEFAULT_PHASE_DEADLINE_GRACE_MS: Final[int] = 5_000
DEFAULT_LEASE_ORPHAN_GRACE_MS: Final[int] = 0
MAX_PAYLOAD_BYTES: Final[int] = 262_144
MAX_EVIDENCE_ITEMS: Final[int] = 256

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS watchdog_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS stall_detections (
    diagnosis_id VARCHAR PRIMARY KEY,
    classification VARCHAR NOT NULL,
    severity VARCHAR NOT NULL,
    actionable BOOLEAN NOT NULL,
    subject_kind VARCHAR NOT NULL DEFAULT '',
    subject_id VARCHAR NOT NULL DEFAULT '',
    task_cid VARCHAR NOT NULL DEFAULT '',
    session_id VARCHAR NOT NULL DEFAULT '',
    worktree_id VARCHAR NOT NULL DEFAULT '',
    lease_id VARCHAR NOT NULL DEFAULT '',
    observed_at_ms BIGINT NOT NULL,
    reason VARCHAR NOT NULL DEFAULT '',
    evidence_json VARCHAR NOT NULL DEFAULT '[]',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS stall_detections_class_idx
    ON stall_detections(classification, observed_at_ms);
CREATE INDEX IF NOT EXISTS stall_detections_subject_idx
    ON stall_detections(subject_id, observed_at_ms);

CREATE TABLE IF NOT EXISTS fenced_recovery_commands (
    command_id VARCHAR PRIMARY KEY,
    diagnosis_id VARCHAR NOT NULL DEFAULT '',
    action_kind VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    idempotency_key VARCHAR NOT NULL,
    expected_fence_epoch BIGINT NOT NULL,
    expected_fencing_token BIGINT NOT NULL,
    expected_process_birth_id VARCHAR NOT NULL DEFAULT '',
    expected_generation BIGINT NOT NULL,
    subject_kind VARCHAR NOT NULL DEFAULT '',
    subject_id VARCHAR NOT NULL DEFAULT '',
    task_cid VARCHAR NOT NULL DEFAULT '',
    session_id VARCHAR NOT NULL DEFAULT '',
    worktree_id VARCHAR NOT NULL DEFAULT '',
    decided_at_ms BIGINT NOT NULL,
    applied_at_ms BIGINT NOT NULL DEFAULT 0,
    result_digest VARCHAR NOT NULL DEFAULT '',
    reason VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS fenced_recovery_commands_idempotency_uidx
    ON fenced_recovery_commands(idempotency_key);
CREATE INDEX IF NOT EXISTS fenced_recovery_commands_subject_idx
    ON fenced_recovery_commands(subject_id, decided_at_ms);
CREATE INDEX IF NOT EXISTS fenced_recovery_commands_status_idx
    ON fenced_recovery_commands(status, decided_at_ms);

CREATE TABLE IF NOT EXISTS restart_decisions (
    decision_id VARCHAR PRIMARY KEY,
    command_id VARCHAR NOT NULL,
    diagnosis_id VARCHAR NOT NULL DEFAULT '',
    disposition VARCHAR NOT NULL,
    observed_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS restart_decisions_command_idx
    ON restart_decisions(command_id, observed_at_ms);

CREATE TABLE IF NOT EXISTS watchdog_events (
    event_id VARCHAR PRIMARY KEY,
    command_id VARCHAR NOT NULL DEFAULT '',
    diagnosis_id VARCHAR NOT NULL DEFAULT '',
    event_type VARCHAR NOT NULL,
    observed_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS watchdog_events_type_idx
    ON watchdog_events(event_type, observed_at_ms);
"""

ClockMs = Callable[[], int]
LivenessProbe = Callable[[ProcessBirthIdentity], OwnerLiveness]

# Closed classification vocabulary (effects surface).
CLASSIFICATION_VALUES: Final[tuple[str, ...]] = (
    "healthy_active",
    "quiescent_strict_shard",
    "provider_capacity_backoff",
    "expiring_session",
    "stale_session",
    "orphan_lease",
    "orphan_worktree",
    "ready_unclaimable",
    "phase_stall",
    "log_stall",
    "migration_fault",
    "server_fault",
    "backup_fault",
    "merge_blockage",
    "recovery_blockage",
    "terminal_drain",
    "ownership_unknown",
    "file_age_only",  # evidence class only; never actionable alone
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseWatchdogError(RuntimeError):
    """Base fail-closed error for database watchdog operations."""

    code = "DQP_WATCHDOG_ERROR"


class DatabaseWatchdogNotOpenError(DatabaseWatchdogError):
    """Operation requires an open watchdog store."""

    code = "DQP_WATCHDOG_NOT_OPEN"


class DatabaseWatchdogConflictError(DatabaseWatchdogError):
    """Identity or status conflict for a fenced recovery command."""

    code = "DQP_WATCHDOG_CONFLICT"


class DatabaseWatchdogFenceError(DatabaseWatchdogError):
    """Expected fence / process birth / generation does not match current state."""

    code = "DQP_WATCHDOG_FENCE"


class DatabaseWatchdogBoundsError(DatabaseWatchdogError, ValueError):
    """Payload or bound exceeded."""

    code = "DQP_WATCHDOG_BOUNDS"


class DatabaseWatchdogOwnershipError(DatabaseWatchdogError):
    """Ownership is unknown; repair is refused and doctor must abstain."""

    code = "DQP_WATCHDOG_OWNERSHIP_UNKNOWN"


class DatabaseWatchdogPolicyError(DatabaseWatchdogError):
    """Policy refused an unsafe action (e.g. file-age-only repair)."""

    code = "DQP_WATCHDOG_POLICY"


class DuckDBUnavailableError(DatabaseWatchdogError):
    """Optional DuckDB dependency is not installed."""

    code = "DQP_DUCKDB_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class StallClassification(str, Enum):
    """Closed set of watchdog health / stall classifications."""

    HEALTHY_ACTIVE = "healthy_active"
    QUIESCENT_STRICT_SHARD = "quiescent_strict_shard"
    PROVIDER_CAPACITY_BACKOFF = "provider_capacity_backoff"
    EXPIRING_SESSION = "expiring_session"
    STALE_SESSION = "stale_session"
    ORPHAN_LEASE = "orphan_lease"
    ORPHAN_WORKTREE = "orphan_worktree"
    READY_UNCLAIMABLE = "ready_unclaimable"
    PHASE_STALL = "phase_stall"
    LOG_STALL = "log_stall"
    MIGRATION_FAULT = "migration_fault"
    SERVER_FAULT = "server_fault"
    BACKUP_FAULT = "backup_fault"
    MERGE_BLOCKAGE = "merge_blockage"
    RECOVERY_BLOCKAGE = "recovery_blockage"
    TERMINAL_DRAIN = "terminal_drain"
    OWNERSHIP_UNKNOWN = "ownership_unknown"
    FILE_AGE_ONLY = "file_age_only"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CommandActionKind(str, Enum):
    """Closed set of safe fenced recovery actions (no raw PID / lock delete)."""

    NO_OP = "no_op"
    EXPIRE_SESSION = "expire_session"
    RECLAIM_ORPHAN_LEASE = "reclaim_orphan_lease"
    RECLAIM_ORPHAN_WORKTREE = "reclaim_orphan_worktree"
    MARK_READY_ACTIONABLE = "mark_ready_actionable"
    INTERRUPT_STALLED_PHASE = "interrupt_stalled_phase"
    REQUEST_SERVER_RESTART = "request_server_restart"
    REQUEST_BACKUP = "request_backup"
    UNBLOCK_MERGE = "unblock_merge"
    UNBLOCK_RECOVERY = "unblock_recovery"
    ABSTAIN = "abstain"


class CommandStatus(str, Enum):
    DECIDED = "decided"
    APPLIED = "applied"
    REPLAYED = "replayed"
    REJECTED = "rejected"
    ABSTAINED = "abstained"


class OwnershipState(str, Enum):
    """Observed exclusive state-owner condition."""

    ABSENT = "absent"
    DEAD = "dead"
    LIVE = "live"
    UNKNOWN = "unknown"


class DoctorDisposition(str, Enum):
    """Doctor report disposition."""

    REPORT = "report"
    ACTIONABLE = "actionable"
    ABSTAIN = "abstain"
    HEALTHY = "healthy"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def duckdb_available() -> bool:
    """Return whether the optional duckdb package can be imported."""

    try:
        import duckdb  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _default_clock_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _utc_iso_from_ms(epoch_ms: int) -> str:
    return (
        datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if "\x00" in text:
        raise DatabaseWatchdogError(f"{name} contains NUL")
    if required and not text:
        raise DatabaseWatchdogError(f"{name} is required")
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatabaseWatchdogBoundsError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DatabaseWatchdogBoundsError(f"{name} must be a positive integer")
    return value


def _sha256_hex(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> str:
    try:
        return canonical_json_bytes(value).decode("utf-8")
    except ValueError:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )


def _bounded_mapping(
    body: Mapping[str, Any] | None,
    *,
    name: str,
    max_bytes: int = MAX_PAYLOAD_BYTES,
) -> dict[str, Any]:
    raw = dict(body or {})
    encoded = _canonical_json(raw).encode("utf-8")
    if len(encoded) > max_bytes:
        raise DatabaseWatchdogBoundsError(
            f"{name} exceeds the {max_bytes}-byte bound"
        )
    return raw


def _row_mapping(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, Mapping):
        return {str(key): row[key] for key in row}
    try:
        keys = list(row.keys())  # type: ignore[attr-defined]
    except Exception:
        keys = []
    if keys:
        return {str(key): row[key] for key in keys}
    try:
        return {str(index): row[index] for index in range(len(row))}  # type: ignore[arg-type]
    except Exception:
        return {}


def _row_get(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
        upper = name.upper()
        if upper in mapping and mapping[upper] is not None:
            return mapping[upper]
        lower = name.lower()
        if lower in mapping and mapping[lower] is not None:
            return mapping[lower]
    wanted = {name.lower() for name in names}
    for key, value in mapping.items():
        if str(key).lower() in wanted and value is not None:
            return value
    return default


def _split_sql_statements(sql_text: str) -> list[str]:
    statements: list[str] = []
    for chunk in str(sql_text).split(";"):
        statement = chunk.strip()
        if not statement or statement.startswith("--"):
            continue
        lines = [
            line
            for line in statement.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        if lines:
            statements.append("\n".join(lines))
    return statements


def _new_id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4().hex}"


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        return []
    if isinstance(value, Sequence):
        return list(value)
    return []


def _default_idempotency_key(
    *,
    action_kind: str,
    subject_id: str,
    expected_fence_epoch: int,
    expected_fencing_token: int,
    expected_process_birth_id: str,
    expected_generation: int,
    reason: str,
    body: Mapping[str, Any],
) -> str:
    return _sha256_hex(
        _canonical_json(
            {
                "action_kind": action_kind,
                "subject_id": subject_id,
                "expected_fence_epoch": expected_fence_epoch,
                "expected_fencing_token": expected_fencing_token,
                "expected_process_birth_id": expected_process_birth_id,
                "expected_generation": expected_generation,
                "reason": reason,
                "body": dict(body),
            }
        ).encode("utf-8")
    )


def _parse_enum(value: Any, enum_cls: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    text = _text(value, name)
    try:
        return enum_cls(text)
    except ValueError as exc:
        raise DatabaseWatchdogError(f"unknown {name}: {text}") from exc


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatchdogEvidence:
    """One bounded evidence fact for diagnosis / doctor reports."""

    INTERFACE: ClassVar[str] = "WatchdogEvidence@1"
    SCHEMA: ClassVar[str] = WATCHDOG_EVIDENCE_SCHEMA

    kind: str
    subject_id: str = ""
    observed_at_ms: int = 0
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _text(self.kind, "kind"))
        object.__setattr__(
            self, "subject_id", _text(self.subject_id, "subject_id", required=False)
        )
        object.__setattr__(
            self,
            "observed_at_ms",
            _nonneg_int(int(self.observed_at_ms), "observed_at_ms"),
        )
        object.__setattr__(
            self,
            "detail",
            MappingProxyType(_bounded_mapping(self.detail, name="evidence detail")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "schema": self.SCHEMA,
            "kind": self.kind,
            "subject_id": self.subject_id,
            "observed_at_ms": int(self.observed_at_ms),
            "detail": dict(self.detail),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "WatchdogEvidence":
        data = dict(payload or {})
        return cls(
            kind=str(data.get("kind") or ""),
            subject_id=str(data.get("subject_id") or ""),
            observed_at_ms=int(data.get("observed_at_ms") or 0),
            detail=_mapping(data.get("detail")),
        )


@dataclass(frozen=True)
class StallDiagnosis:
    """One classified stall / health observation (StallDiagnosis@1)."""

    INTERFACE: ClassVar[str] = STALL_DIAGNOSIS_INTERFACE
    SCHEMA: ClassVar[str] = STALL_DIAGNOSIS_SCHEMA

    diagnosis_id: str
    classification: StallClassification
    severity: Severity
    actionable: bool
    observed_at_ms: int
    reason: str = ""
    subject_kind: str = ""
    subject_id: str = ""
    task_cid: str = ""
    session_id: str = ""
    worktree_id: str = ""
    lease_id: str = ""
    evidence: tuple[WatchdogEvidence, ...] = ()
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "diagnosis_id", _text(self.diagnosis_id, "diagnosis_id")
        )
        object.__setattr__(
            self,
            "classification",
            _parse_enum(self.classification, StallClassification, "classification"),
        )
        object.__setattr__(
            self, "severity", _parse_enum(self.severity, Severity, "severity")
        )
        object.__setattr__(self, "actionable", bool(self.actionable))
        object.__setattr__(
            self,
            "observed_at_ms",
            _nonneg_int(int(self.observed_at_ms), "observed_at_ms"),
        )
        object.__setattr__(
            self, "reason", _text(self.reason, "reason", required=False)
        )
        object.__setattr__(
            self,
            "subject_kind",
            _text(self.subject_kind, "subject_kind", required=False),
        )
        object.__setattr__(
            self, "subject_id", _text(self.subject_id, "subject_id", required=False)
        )
        object.__setattr__(
            self, "task_cid", _text(self.task_cid, "task_cid", required=False)
        )
        object.__setattr__(
            self, "session_id", _text(self.session_id, "session_id", required=False)
        )
        object.__setattr__(
            self,
            "worktree_id",
            _text(self.worktree_id, "worktree_id", required=False),
        )
        object.__setattr__(
            self, "lease_id", _text(self.lease_id, "lease_id", required=False)
        )
        evidence_items = tuple(self.evidence or ())
        if len(evidence_items) > MAX_EVIDENCE_ITEMS:
            raise DatabaseWatchdogBoundsError(
                f"evidence exceeds the {MAX_EVIDENCE_ITEMS}-item bound"
            )
        object.__setattr__(self, "evidence", evidence_items)
        object.__setattr__(
            self,
            "body",
            MappingProxyType(_bounded_mapping(self.body, name="diagnosis body")),
        )
        # File-age-only evidence is never actionable by itself.
        if (
            self.classification is StallClassification.FILE_AGE_ONLY
            and self.actionable
        ):
            object.__setattr__(self, "actionable", False)

    @property
    def abstain(self) -> bool:
        return self.classification is StallClassification.OWNERSHIP_UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "schema": self.SCHEMA,
            "diagnosis_id": self.diagnosis_id,
            "classification": self.classification.value,
            "severity": self.severity.value,
            "actionable": bool(self.actionable),
            "abstain": self.abstain,
            "observed_at_ms": int(self.observed_at_ms),
            "observed_at": _utc_iso_from_ms(int(self.observed_at_ms)),
            "reason": self.reason,
            "subject_kind": self.subject_kind,
            "subject_id": self.subject_id,
            "task_cid": self.task_cid,
            "session_id": self.session_id,
            "worktree_id": self.worktree_id,
            "lease_id": self.lease_id,
            "evidence": [item.to_dict() for item in self.evidence],
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class FencedRecoveryCommand:
    """One fenced, idempotent recovery command (FencedRecoveryCommand@1)."""

    INTERFACE: ClassVar[str] = FENCED_RECOVERY_COMMAND_INTERFACE
    SCHEMA: ClassVar[str] = FENCED_RECOVERY_COMMAND_SCHEMA

    command_id: str
    action_kind: CommandActionKind
    status: CommandStatus
    idempotency_key: str
    expected_fence_epoch: int
    expected_fencing_token: int
    expected_generation: int
    decided_at_ms: int
    expected_process_birth_id: str = ""
    diagnosis_id: str = ""
    subject_kind: str = ""
    subject_id: str = ""
    task_cid: str = ""
    session_id: str = ""
    worktree_id: str = ""
    applied_at_ms: int = 0
    result_digest: str = ""
    reason: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_id", _text(self.command_id, "command_id"))
        object.__setattr__(
            self,
            "action_kind",
            _parse_enum(self.action_kind, CommandActionKind, "action_kind"),
        )
        object.__setattr__(
            self, "status", _parse_enum(self.status, CommandStatus, "status")
        )
        object.__setattr__(
            self, "idempotency_key", _text(self.idempotency_key, "idempotency_key")
        )
        object.__setattr__(
            self,
            "expected_fence_epoch",
            _nonneg_int(int(self.expected_fence_epoch), "expected_fence_epoch"),
        )
        object.__setattr__(
            self,
            "expected_fencing_token",
            _nonneg_int(int(self.expected_fencing_token), "expected_fencing_token"),
        )
        object.__setattr__(
            self,
            "expected_generation",
            _nonneg_int(int(self.expected_generation), "expected_generation"),
        )
        object.__setattr__(
            self,
            "decided_at_ms",
            _nonneg_int(int(self.decided_at_ms), "decided_at_ms"),
        )
        object.__setattr__(
            self,
            "expected_process_birth_id",
            _text(
                self.expected_process_birth_id,
                "expected_process_birth_id",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "diagnosis_id",
            _text(self.diagnosis_id, "diagnosis_id", required=False),
        )
        object.__setattr__(
            self,
            "subject_kind",
            _text(self.subject_kind, "subject_kind", required=False),
        )
        object.__setattr__(
            self, "subject_id", _text(self.subject_id, "subject_id", required=False)
        )
        object.__setattr__(
            self, "task_cid", _text(self.task_cid, "task_cid", required=False)
        )
        object.__setattr__(
            self, "session_id", _text(self.session_id, "session_id", required=False)
        )
        object.__setattr__(
            self,
            "worktree_id",
            _text(self.worktree_id, "worktree_id", required=False),
        )
        object.__setattr__(
            self,
            "applied_at_ms",
            _nonneg_int(int(self.applied_at_ms), "applied_at_ms"),
        )
        object.__setattr__(
            self,
            "result_digest",
            _text(self.result_digest, "result_digest", required=False),
        )
        object.__setattr__(
            self, "reason", _text(self.reason, "reason", required=False)
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(_bounded_mapping(self.body, name="command body")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "schema": self.SCHEMA,
            "command_id": self.command_id,
            "action_kind": self.action_kind.value,
            "status": self.status.value,
            "idempotency_key": self.idempotency_key,
            "expected_fence_epoch": int(self.expected_fence_epoch),
            "expected_fencing_token": int(self.expected_fencing_token),
            "expected_process_birth_id": self.expected_process_birth_id,
            "expected_generation": int(self.expected_generation),
            "diagnosis_id": self.diagnosis_id,
            "subject_kind": self.subject_kind,
            "subject_id": self.subject_id,
            "task_cid": self.task_cid,
            "session_id": self.session_id,
            "worktree_id": self.worktree_id,
            "decided_at_ms": int(self.decided_at_ms),
            "applied_at_ms": int(self.applied_at_ms),
            "result_digest": self.result_digest,
            "reason": self.reason,
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class DoctorReport:
    """Doctor surface: evidence, diagnoses, and disposition."""

    disposition: DoctorDisposition
    ownership: OwnershipState
    diagnoses: tuple[StallDiagnosis, ...]
    evidence: tuple[WatchdogEvidence, ...]
    observed_at_ms: int
    reason: str = ""
    commands: tuple[FencedRecoveryCommand, ...] = ()
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _parse_enum(self.disposition, DoctorDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "ownership",
            _parse_enum(self.ownership, OwnershipState, "ownership"),
        )
        object.__setattr__(self, "diagnoses", tuple(self.diagnoses or ()))
        object.__setattr__(self, "evidence", tuple(self.evidence or ()))
        object.__setattr__(self, "commands", tuple(self.commands or ()))
        object.__setattr__(
            self,
            "observed_at_ms",
            _nonneg_int(int(self.observed_at_ms), "observed_at_ms"),
        )
        object.__setattr__(
            self, "reason", _text(self.reason, "reason", required=False)
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(_bounded_mapping(self.body, name="doctor body")),
        )

    @property
    def abstain(self) -> bool:
        return self.disposition is DoctorDisposition.ABSTAIN

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": "DuckDBQuackDoctorReport@1",
            "disposition": self.disposition.value,
            "ownership": self.ownership.value,
            "abstain": self.abstain,
            "observed_at_ms": int(self.observed_at_ms),
            "observed_at": _utc_iso_from_ms(int(self.observed_at_ms)),
            "reason": self.reason,
            "diagnoses": [item.to_dict() for item in self.diagnoses],
            "evidence": [item.to_dict() for item in self.evidence],
            "commands": [item.to_dict() for item in self.commands],
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class WatchdogObservation:
    """Normalized control-plane snapshot for one diagnostic pass."""

    INTERFACE: ClassVar[str] = "WatchdogObservation@1"
    SCHEMA: ClassVar[str] = WATCHDOG_OBSERVATION_SCHEMA

    now_ms: int
    generation: int = 1
    fence_epoch: int = 1
    fencing_token: int = 0
    server_process_birth: ProcessBirthIdentity | None = None
    server_process_birth_id: str = ""
    ownership: OwnershipState = OwnershipState.ABSENT
    sessions: tuple[Mapping[str, Any], ...] = ()
    heartbeats: tuple[Mapping[str, Any], ...] = ()
    tasks: tuple[Mapping[str, Any], ...] = ()
    claims: tuple[Mapping[str, Any], ...] = ()
    leases: tuple[Mapping[str, Any], ...] = ()
    attempts: tuple[Mapping[str, Any], ...] = ()
    worktrees: tuple[Mapping[str, Any], ...] = ()
    merges: tuple[Mapping[str, Any], ...] = ()
    recoveries: tuple[Mapping[str, Any], ...] = ()
    provider_capacity: Mapping[str, Any] = field(default_factory=dict)
    migrations: Mapping[str, Any] = field(default_factory=dict)
    backup: Mapping[str, Any] = field(default_factory=dict)
    server: Mapping[str, Any] = field(default_factory=dict)
    file_mirrors: tuple[Mapping[str, Any], ...] = ()
    shard_id: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "now_ms", _nonneg_int(int(self.now_ms), "now_ms"))
        object.__setattr__(
            self, "generation", _nonneg_int(int(self.generation), "generation")
        )
        object.__setattr__(
            self, "fence_epoch", _nonneg_int(int(self.fence_epoch), "fence_epoch")
        )
        object.__setattr__(
            self,
            "fencing_token",
            _nonneg_int(int(self.fencing_token), "fencing_token"),
        )
        birth = self.server_process_birth
        if birth is not None and not isinstance(birth, ProcessBirthIdentity):
            birth = process_birth_from_mapping(_mapping(birth))
            object.__setattr__(self, "server_process_birth", birth)
        birth_id = str(self.server_process_birth_id or "").strip()
        if not birth_id and birth is not None:
            birth_id = process_birth_id(birth)
        object.__setattr__(self, "server_process_birth_id", birth_id)
        object.__setattr__(
            self,
            "ownership",
            _parse_enum(self.ownership, OwnershipState, "ownership"),
        )
        for name in (
            "sessions",
            "heartbeats",
            "tasks",
            "claims",
            "leases",
            "attempts",
            "worktrees",
            "merges",
            "recoveries",
            "file_mirrors",
        ):
            items = tuple(
                MappingProxyType(_mapping(item)) for item in (getattr(self, name) or ())
            )
            object.__setattr__(self, name, items)
        object.__setattr__(
            self,
            "provider_capacity",
            MappingProxyType(
                _bounded_mapping(self.provider_capacity, name="provider_capacity")
            ),
        )
        object.__setattr__(
            self,
            "migrations",
            MappingProxyType(_bounded_mapping(self.migrations, name="migrations")),
        )
        object.__setattr__(
            self,
            "backup",
            MappingProxyType(_bounded_mapping(self.backup, name="backup")),
        )
        object.__setattr__(
            self,
            "server",
            MappingProxyType(_bounded_mapping(self.server, name="server")),
        )
        object.__setattr__(
            self, "shard_id", _text(self.shard_id, "shard_id", required=False)
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(_bounded_mapping(self.body, name="observation body")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "schema": self.SCHEMA,
            "now_ms": int(self.now_ms),
            "generation": int(self.generation),
            "fence_epoch": int(self.fence_epoch),
            "fencing_token": int(self.fencing_token),
            "server_process_birth": (
                self.server_process_birth.to_dict()
                if self.server_process_birth is not None
                else None
            ),
            "server_process_birth_id": self.server_process_birth_id,
            "ownership": self.ownership.value,
            "sessions": [dict(item) for item in self.sessions],
            "heartbeats": [dict(item) for item in self.heartbeats],
            "tasks": [dict(item) for item in self.tasks],
            "claims": [dict(item) for item in self.claims],
            "leases": [dict(item) for item in self.leases],
            "attempts": [dict(item) for item in self.attempts],
            "worktrees": [dict(item) for item in self.worktrees],
            "merges": [dict(item) for item in self.merges],
            "recoveries": [dict(item) for item in self.recoveries],
            "provider_capacity": dict(self.provider_capacity),
            "migrations": dict(self.migrations),
            "backup": dict(self.backup),
            "server": dict(self.server),
            "file_mirrors": [dict(item) for item in self.file_mirrors],
            "shard_id": self.shard_id,
            "body": dict(self.body),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "WatchdogObservation":
        data = dict(payload or {})
        birth_raw = data.get("server_process_birth")
        birth: ProcessBirthIdentity | None = None
        if isinstance(birth_raw, ProcessBirthIdentity):
            birth = birth_raw
        elif isinstance(birth_raw, Mapping):
            birth = process_birth_from_mapping(birth_raw)
        ownership_raw = data.get("ownership", OwnershipState.ABSENT)
        return cls(
            now_ms=int(data.get("now_ms") or 0),
            generation=int(data.get("generation") or 1),
            fence_epoch=int(data.get("fence_epoch") or 1),
            fencing_token=int(data.get("fencing_token") or 0),
            server_process_birth=birth,
            server_process_birth_id=str(data.get("server_process_birth_id") or ""),
            ownership=ownership_raw,  # type: ignore[arg-type]
            sessions=tuple(_sequence(data.get("sessions"))),
            heartbeats=tuple(_sequence(data.get("heartbeats"))),
            tasks=tuple(_sequence(data.get("tasks"))),
            claims=tuple(_sequence(data.get("claims"))),
            leases=tuple(_sequence(data.get("leases"))),
            attempts=tuple(_sequence(data.get("attempts"))),
            worktrees=tuple(_sequence(data.get("worktrees"))),
            merges=tuple(_sequence(data.get("merges"))),
            recoveries=tuple(_sequence(data.get("recoveries"))),
            provider_capacity=_mapping(data.get("provider_capacity")),
            migrations=_mapping(data.get("migrations")),
            backup=_mapping(data.get("backup")),
            server=_mapping(data.get("server")),
            file_mirrors=tuple(_sequence(data.get("file_mirrors"))),
            shard_id=str(data.get("shard_id") or ""),
            body=_mapping(data.get("body")),
        )


# ---------------------------------------------------------------------------
# Diagnosis engine (pure)
# ---------------------------------------------------------------------------


def _evidence(
    kind: str,
    *,
    subject_id: str = "",
    observed_at_ms: int = 0,
    **detail: Any,
) -> WatchdogEvidence:
    return WatchdogEvidence(
        kind=kind,
        subject_id=subject_id,
        observed_at_ms=observed_at_ms,
        detail=detail,
    )


def _diagnosis(
    classification: StallClassification,
    *,
    now_ms: int,
    reason: str,
    severity: Severity,
    actionable: bool,
    subject_kind: str = "",
    subject_id: str = "",
    task_cid: str = "",
    session_id: str = "",
    worktree_id: str = "",
    lease_id: str = "",
    evidence: Sequence[WatchdogEvidence] = (),
    body: Mapping[str, Any] | None = None,
) -> StallDiagnosis:
    return StallDiagnosis(
        diagnosis_id=_new_id("diagnosis"),
        classification=classification,
        severity=severity,
        actionable=actionable,
        observed_at_ms=now_ms,
        reason=reason,
        subject_kind=subject_kind,
        subject_id=subject_id,
        task_cid=task_cid,
        session_id=session_id,
        worktree_id=worktree_id,
        lease_id=lease_id,
        evidence=tuple(evidence),
        body=dict(body or {}),
    )


def _session_status(session: Mapping[str, Any]) -> str:
    return str(session.get("status") or session.get("session_status") or "").strip().lower()


def _task_status(task: Mapping[str, Any]) -> str:
    return str(task.get("status") or task.get("task_status") or "").strip().lower()


def _int_field(mapping: Mapping[str, Any], *names: str, default: int = 0) -> int:
    for name in names:
        if name in mapping and mapping[name] is not None:
            try:
                return int(mapping[name])
            except (TypeError, ValueError):
                continue
    return default


def _text_field(mapping: Mapping[str, Any], *names: str, default: str = "") -> str:
    for name in names:
        value = mapping.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def diagnose_observation(
    observation: WatchdogObservation | Mapping[str, Any],
    *,
    heartbeat_stale_ms: int = DEFAULT_HEARTBEAT_STALE_MS,
    session_expiry_ms: int = DEFAULT_SESSION_EXPIRY_MS,
    phase_deadline_grace_ms: int = DEFAULT_PHASE_DEADLINE_GRACE_MS,
    liveness: LivenessProbe | None = None,
) -> list[StallDiagnosis]:
    """Classify one control-plane observation into stall diagnoses.

    Pure function: does not open databases, signal processes, or delete locks.
    """

    obs = (
        observation
        if isinstance(observation, WatchdogObservation)
        else WatchdogObservation.from_mapping(observation)
    )
    now = int(obs.now_ms)
    diagnoses: list[StallDiagnosis] = []

    # Ownership unknown ⇒ expose evidence and abstain (no repair).
    if obs.ownership is OwnershipState.UNKNOWN:
        diagnoses.append(
            _diagnosis(
                StallClassification.OWNERSHIP_UNKNOWN,
                now_ms=now,
                reason="server ownership is unknown; doctor must abstain",
                severity=Severity.WARNING,
                actionable=False,
                subject_kind="server",
                subject_id=str(obs.server.get("server_id") or "server"),
                evidence=[
                    _evidence(
                        "ownership",
                        subject_id="server",
                        observed_at_ms=now,
                        ownership=obs.ownership.value,
                        generation=obs.generation,
                        process_birth_id=obs.server_process_birth_id,
                    )
                ],
            )
        )
        # Still collect non-mutating evidence classifications below, but mark
        # nothing as actionable when ownership is unknown.

    # Server / migration / backup faults.
    server_status = str(obs.server.get("status") or "").strip().lower()
    if server_status in {"failed", "crashed", "unhealthy", "fault"}:
        diagnoses.append(
            _diagnosis(
                StallClassification.SERVER_FAULT,
                now_ms=now,
                reason=f"server status is {server_status}",
                severity=Severity.CRITICAL,
                actionable=obs.ownership is not OwnershipState.UNKNOWN,
                subject_kind="server",
                subject_id=str(obs.server.get("server_id") or "server"),
                evidence=[
                    _evidence(
                        "server",
                        subject_id=str(obs.server.get("server_id") or "server"),
                        observed_at_ms=now,
                        status=server_status,
                        generation=obs.generation,
                    )
                ],
            )
        )

    migration_status = str(obs.migrations.get("status") or "").strip().lower()
    if migration_status in {"failed", "blocked", "fault"}:
        diagnoses.append(
            _diagnosis(
                StallClassification.MIGRATION_FAULT,
                now_ms=now,
                reason=f"migration status is {migration_status}",
                severity=Severity.CRITICAL,
                actionable=obs.ownership is not OwnershipState.UNKNOWN,
                subject_kind="migration",
                subject_id=str(obs.migrations.get("migration_id") or "migration"),
                evidence=[
                    _evidence(
                        "migration",
                        observed_at_ms=now,
                        status=migration_status,
                        detail=dict(obs.migrations),
                    )
                ],
            )
        )

    backup_status = str(obs.backup.get("status") or "").strip().lower()
    if backup_status in {"failed", "corrupt", "missing", "fault"}:
        diagnoses.append(
            _diagnosis(
                StallClassification.BACKUP_FAULT,
                now_ms=now,
                reason=f"backup status is {backup_status}",
                severity=Severity.ERROR,
                actionable=obs.ownership is not OwnershipState.UNKNOWN,
                subject_kind="backup",
                subject_id=str(obs.backup.get("backup_id") or "backup"),
                evidence=[
                    _evidence(
                        "backup",
                        observed_at_ms=now,
                        status=backup_status,
                        age_ms=_int_field(obs.backup, "age_ms", "backup_age_ms"),
                    )
                ],
            )
        )

    # Provider capacity backoff (typed temporary state, not failure).
    capacity = dict(obs.provider_capacity)
    backoff = _as_bool(capacity.get("backoff")) or _as_bool(
        capacity.get("capacity_exhausted")
    )
    available = capacity.get("available")
    if backoff or available is False or available == 0:
        diagnoses.append(
            _diagnosis(
                StallClassification.PROVIDER_CAPACITY_BACKOFF,
                now_ms=now,
                reason="provider capacity backoff is active",
                severity=Severity.INFO,
                actionable=False,
                subject_kind="provider",
                subject_id=str(capacity.get("provider_id") or "provider"),
                evidence=[
                    _evidence(
                        "provider_capacity",
                        observed_at_ms=now,
                        backoff=True,
                        capacity=capacity,
                    )
                ],
            )
        )

    # Terminal drain.
    if _as_bool(obs.body.get("terminal_drain")) or server_status in {
        "draining",
        "stopped",
        "stopping",
    }:
        if server_status in {"draining", "stopped", "stopping"} or _as_bool(
            obs.body.get("terminal_drain")
        ):
            diagnoses.append(
                _diagnosis(
                    StallClassification.TERMINAL_DRAIN,
                    now_ms=now,
                    reason="control plane is in terminal drain",
                    severity=Severity.INFO,
                    actionable=False,
                    subject_kind="server",
                    subject_id=str(obs.server.get("server_id") or "server"),
                    evidence=[
                        _evidence(
                            "terminal_drain",
                            observed_at_ms=now,
                            server_status=server_status,
                        )
                    ],
                )
            )

    # Sessions / heartbeats.
    for session in obs.sessions:
        session_id = _text_field(session, "session_id", "id")
        status = _session_status(session)
        last_hb = _int_field(
            session, "last_heartbeat_at_ms", "heartbeat_at_ms", "updated_at_ms"
        )
        deadline = _int_field(session, "deadline_ms", "expires_at_ms")
        process_birth = process_birth_from_mapping(
            session.get("process_birth")
            if isinstance(session.get("process_birth"), Mapping)
            else None
        )
        birth_id = _text_field(session, "process_birth_id") or (
            process_birth_id(process_birth) if process_birth.pid else ""
        )
        liveness_status = OwnerLiveness.UNKNOWN
        if liveness is not None and process_birth.pid:
            try:
                liveness_status = liveness(process_birth)
            except Exception:
                liveness_status = OwnerLiveness.UNKNOWN
        elif session.get("owner_liveness"):
            try:
                liveness_status = OwnerLiveness(
                    str(session.get("owner_liveness")).strip().lower()
                )
            except ValueError:
                liveness_status = OwnerLiveness.UNKNOWN

        age_ms = max(0, now - last_hb) if last_hb else 0
        expiring = bool(deadline and 0 < deadline - now <= session_expiry_ms)
        stale_by_heartbeat = bool(
            last_hb and age_ms > heartbeat_stale_ms and status in {"active", "open", "running", ""}
        )
        dead_owner = liveness_status is OwnerLiveness.DEAD

        if status in {"active", "open", "running", ""} and (stale_by_heartbeat or dead_owner):
            # Require more than file age: heartbeat/process birth facts.
            evidence = [
                _evidence(
                    "session",
                    subject_id=session_id,
                    observed_at_ms=now,
                    status=status or "active",
                    last_heartbeat_at_ms=last_hb,
                    heartbeat_age_ms=age_ms,
                    process_birth_id=birth_id,
                    owner_liveness=liveness_status.value,
                    fence_epoch=_int_field(session, "fence_epoch"),
                    fencing_token=_int_field(session, "fencing_token"),
                )
            ]
            diagnoses.append(
                _diagnosis(
                    StallClassification.STALE_SESSION,
                    now_ms=now,
                    reason=(
                        "session heartbeat stale with dead/reused owner"
                        if dead_owner
                        else "session heartbeat stale against database clock"
                    ),
                    severity=Severity.ERROR,
                    actionable=(
                        obs.ownership is not OwnershipState.UNKNOWN
                        and (
                            dead_owner
                            or bool(last_hb)
                        )
                    ),
                    subject_kind="session",
                    subject_id=session_id,
                    session_id=session_id,
                    evidence=evidence,
                    body={
                        "process_birth_id": birth_id,
                        "fence_epoch": _int_field(session, "fence_epoch"),
                        "fencing_token": _int_field(session, "fencing_token"),
                        "server_generation": _int_field(
                            session, "server_generation", "generation"
                        ),
                    },
                )
            )
        elif expiring and status in {"active", "open", "running", ""}:
            diagnoses.append(
                _diagnosis(
                    StallClassification.EXPIRING_SESSION,
                    now_ms=now,
                    reason="session deadline is within expiry window",
                    severity=Severity.WARNING,
                    actionable=False,
                    subject_kind="session",
                    subject_id=session_id,
                    session_id=session_id,
                    evidence=[
                        _evidence(
                            "session_expiry",
                            subject_id=session_id,
                            observed_at_ms=now,
                            deadline_ms=deadline,
                            remaining_ms=max(0, deadline - now),
                        )
                    ],
                )
            )

    # Orphan leases.
    for lease in obs.leases:
        lease_id = _text_field(lease, "lease_id", "claim_cid", "id")
        state = str(lease.get("state") or lease.get("status") or "").strip().lower()
        owner_session = _text_field(lease, "owner_session_id", "session_id")
        owner_alive = _as_bool(lease.get("owner_alive"), default=True)
        owner_liveness_raw = str(lease.get("owner_liveness") or "").strip().lower()
        expires_at = _int_field(lease, "expires_at_ms")
        expired = bool(expires_at and expires_at < now)
        if state in {"held", "active", "claimed", ""} and (
            not owner_alive
            or owner_liveness_raw in {"dead", "absent"}
            or (expired and not owner_session)
        ):
            diagnoses.append(
                _diagnosis(
                    StallClassification.ORPHAN_LEASE,
                    now_ms=now,
                    reason="lease held without live owner session",
                    severity=Severity.ERROR,
                    actionable=obs.ownership is not OwnershipState.UNKNOWN,
                    subject_kind="lease",
                    subject_id=lease_id,
                    lease_id=lease_id,
                    task_cid=_text_field(lease, "task_cid"),
                    session_id=owner_session,
                    evidence=[
                        _evidence(
                            "lease",
                            subject_id=lease_id,
                            observed_at_ms=now,
                            state=state or "held",
                            owner_session_id=owner_session,
                            owner_alive=owner_alive,
                            owner_liveness=owner_liveness_raw or "dead",
                            expires_at_ms=expires_at,
                            fencing_token=_int_field(lease, "fencing_token"),
                            fence_epoch=_int_field(lease, "fence_epoch"),
                        )
                    ],
                    body={
                        "fencing_token": _int_field(lease, "fencing_token"),
                        "fence_epoch": _int_field(lease, "fence_epoch"),
                        "process_birth_id": _text_field(lease, "process_birth_id"),
                        "generation": _int_field(lease, "generation", "server_generation"),
                    },
                )
            )

    # Orphan worktrees.
    for worktree in obs.worktrees:
        worktree_id = _text_field(worktree, "worktree_id", "id")
        state = str(worktree.get("state") or worktree.get("status") or "").strip().lower()
        owner_alive = _as_bool(worktree.get("owner_alive"), default=True)
        owner_liveness_raw = str(worktree.get("owner_liveness") or "").strip().lower()
        if state in {"active", "preparing", "settling", ""} and (
            not owner_alive or owner_liveness_raw in {"dead", "absent"}
        ):
            diagnoses.append(
                _diagnosis(
                    StallClassification.ORPHAN_WORKTREE,
                    now_ms=now,
                    reason="worktree active without live owner",
                    severity=Severity.ERROR,
                    actionable=obs.ownership is not OwnershipState.UNKNOWN,
                    subject_kind="worktree",
                    subject_id=worktree_id,
                    worktree_id=worktree_id,
                    task_cid=_text_field(worktree, "task_cid"),
                    session_id=_text_field(worktree, "owner_session_id", "session_id"),
                    evidence=[
                        _evidence(
                            "worktree",
                            subject_id=worktree_id,
                            observed_at_ms=now,
                            state=state or "active",
                            owner_alive=owner_alive,
                            owner_liveness=owner_liveness_raw or "dead",
                            fencing_token=_int_field(worktree, "fencing_token"),
                            fence_epoch=_int_field(worktree, "fence_epoch"),
                        )
                    ],
                    body={
                        "fencing_token": _int_field(worktree, "fencing_token"),
                        "fence_epoch": _int_field(worktree, "fence_epoch"),
                        "process_birth_id": _text_field(worktree, "process_birth_id"),
                        "generation": _int_field(
                            worktree, "generation", "server_generation"
                        ),
                    },
                )
            )

    # Ready tasks without valid owner / capacity / dependency reason.
    claimed_task_cids = {
        _text_field(claim, "task_cid")
        for claim in obs.claims
        if _text_field(claim, "task_cid")
        and str(claim.get("status") or claim.get("state") or "held").strip().lower()
        in {"held", "active", "claimed", ""}
    }
    claimed_task_cids |= {
        _text_field(lease, "task_cid")
        for lease in obs.leases
        if _text_field(lease, "task_cid")
        and str(lease.get("state") or lease.get("status") or "held").strip().lower()
        in {"held", "active", "claimed", ""}
    }

    ready_tasks: list[Mapping[str, Any]] = []
    ready_for_shard: list[Mapping[str, Any]] = []
    for task in obs.tasks:
        status = _task_status(task)
        if status not in {"ready", "selectable_ready", "open"}:
            continue
        ready_tasks.append(task)
        task_shard = _text_field(task, "shard_id", "shard")
        if not obs.shard_id or not task_shard or task_shard == obs.shard_id:
            ready_for_shard.append(task)

    capacity_blocks = bool(
        diagnoses
        and any(
            d.classification is StallClassification.PROVIDER_CAPACITY_BACKOFF
            for d in diagnoses
        )
    )

    for task in ready_tasks:
        task_cid = _text_field(task, "task_cid", "id")
        if task_cid in claimed_task_cids:
            continue
        dependency_reason = _text_field(
            task, "dependency_reason", "blocked_reason", "not_ready_reason"
        )
        has_unmet_dependency = _as_bool(task.get("has_unmet_dependency")) or bool(
            dependency_reason
            and dependency_reason
            not in {"", "none", "ready", "selectable", "no_reason"}
        )
        owner_session = _text_field(task, "owner_session_id", "session_id", "owner")
        has_valid_owner = bool(owner_session)
        task_shard = _text_field(task, "shard_id", "shard")
        owned_by_other_shard = bool(
            obs.shard_id and task_shard and task_shard != obs.shard_id
        )

        if owned_by_other_shard:
            continue  # quiescence handled below
        if has_unmet_dependency:
            continue
        if capacity_blocks:
            continue
        if has_valid_owner:
            # Owner present but work still ready/unclaimed is still actionable
            # only when the owner is not a valid claim holder.
            continue

        diagnoses.append(
            _diagnosis(
                StallClassification.READY_UNCLAIMABLE,
                now_ms=now,
                reason=(
                    "ready task has no valid owner, capacity backoff, or "
                    "dependency explanation"
                ),
                severity=Severity.ERROR,
                actionable=obs.ownership is not OwnershipState.UNKNOWN,
                subject_kind="task",
                subject_id=task_cid,
                task_cid=task_cid,
                evidence=[
                    _evidence(
                        "ready_task",
                        subject_id=task_cid,
                        observed_at_ms=now,
                        status=_task_status(task) or "ready",
                        owner_session_id=owner_session or None,
                        dependency_reason=dependency_reason or None,
                        capacity_backoff=capacity_blocks,
                        claimed=False,
                    )
                ],
            )
        )

    # Quiescent strict shard: no ready work for this shard while other shards own ready work.
    if obs.shard_id and not ready_for_shard and ready_tasks:
        other = [
            _text_field(task, "task_cid", "id")
            for task in ready_tasks
            if _text_field(task, "shard_id", "shard")
            and _text_field(task, "shard_id", "shard") != obs.shard_id
        ]
        if other:
            diagnoses.append(
                _diagnosis(
                    StallClassification.QUIESCENT_STRICT_SHARD,
                    now_ms=now,
                    reason="no selectable ready tasks for this shard (other shards own ready work)",
                    severity=Severity.INFO,
                    actionable=False,
                    subject_kind="shard",
                    subject_id=obs.shard_id,
                    evidence=[
                        _evidence(
                            "shard_quiescence",
                            subject_id=obs.shard_id,
                            observed_at_ms=now,
                            other_shard_ready_task_cids=other,
                            classification_note="no_shard_selectable_ready_tasks",
                        )
                    ],
                )
            )

    # Phase / log stalls on attempts.
    for attempt in obs.attempts:
        attempt_id = _text_field(attempt, "attempt_id", "id")
        phase = _text_field(attempt, "phase", "current_phase")
        phase_deadline = _int_field(attempt, "phase_deadline_ms", "deadline_ms")
        last_progress = _int_field(
            attempt, "last_progress_at_ms", "progress_at_ms", "updated_at_ms"
        )
        log_progress_at = _int_field(attempt, "log_progress_at_ms", "last_log_at_ms")
        status = str(attempt.get("status") or "").strip().lower()
        if status in {"completed", "failed", "cancelled", "terminal"}:
            continue
        if phase_deadline and now > phase_deadline + phase_deadline_grace_ms:
            diagnoses.append(
                _diagnosis(
                    StallClassification.PHASE_STALL,
                    now_ms=now,
                    reason=f"attempt phase {phase or 'unknown'} exceeded deadline",
                    severity=Severity.ERROR,
                    actionable=obs.ownership is not OwnershipState.UNKNOWN,
                    subject_kind="attempt",
                    subject_id=attempt_id,
                    task_cid=_text_field(attempt, "task_cid"),
                    session_id=_text_field(attempt, "session_id"),
                    evidence=[
                        _evidence(
                            "phase",
                            subject_id=attempt_id,
                            observed_at_ms=now,
                            phase=phase,
                            phase_deadline_ms=phase_deadline,
                            last_progress_at_ms=last_progress,
                            fencing_token=_int_field(attempt, "fencing_token"),
                            fence_epoch=_int_field(attempt, "fence_epoch"),
                        )
                    ],
                    body={
                        "fencing_token": _int_field(attempt, "fencing_token"),
                        "fence_epoch": _int_field(attempt, "fence_epoch"),
                        "process_birth_id": _text_field(attempt, "process_birth_id"),
                        "generation": _int_field(
                            attempt, "generation", "server_generation"
                        ),
                    },
                )
            )
        elif (
            log_progress_at
            and last_progress
            and now - log_progress_at > heartbeat_stale_ms
            and now - last_progress > heartbeat_stale_ms
        ):
            diagnoses.append(
                _diagnosis(
                    StallClassification.LOG_STALL,
                    now_ms=now,
                    reason="attempt log and progress cursors are both stale",
                    severity=Severity.WARNING,
                    actionable=obs.ownership is not OwnershipState.UNKNOWN,
                    subject_kind="attempt",
                    subject_id=attempt_id,
                    task_cid=_text_field(attempt, "task_cid"),
                    evidence=[
                        _evidence(
                            "log_progress",
                            subject_id=attempt_id,
                            observed_at_ms=now,
                            log_progress_at_ms=log_progress_at,
                            last_progress_at_ms=last_progress,
                        )
                    ],
                )
            )

    # Merge / recovery blockages.
    for merge in obs.merges:
        merge_id = _text_field(merge, "merge_id", "entry_id", "id")
        status = str(merge.get("status") or "").strip().lower()
        if status in {"blocked", "conflict", "stalled", "failed"}:
            diagnoses.append(
                _diagnosis(
                    StallClassification.MERGE_BLOCKAGE,
                    now_ms=now,
                    reason=f"merge is {status}",
                    severity=Severity.ERROR,
                    actionable=obs.ownership is not OwnershipState.UNKNOWN,
                    subject_kind="merge",
                    subject_id=merge_id,
                    task_cid=_text_field(merge, "task_cid"),
                    worktree_id=_text_field(merge, "worktree_id"),
                    evidence=[
                        _evidence(
                            "merge",
                            subject_id=merge_id,
                            observed_at_ms=now,
                            status=status,
                            fencing_token=_int_field(merge, "fencing_token"),
                            fence_epoch=_int_field(merge, "fence_epoch"),
                        )
                    ],
                    body={
                        "fencing_token": _int_field(merge, "fencing_token"),
                        "fence_epoch": _int_field(merge, "fence_epoch"),
                        "process_birth_id": _text_field(merge, "process_birth_id"),
                        "generation": _int_field(merge, "generation", "server_generation"),
                    },
                )
            )

    for recovery in obs.recoveries:
        recovery_id = _text_field(recovery, "recovery_id", "subject_id", "id")
        status = str(recovery.get("status") or "").strip().lower()
        if status in {"blocked", "exhausted", "failed", "stuck"}:
            diagnoses.append(
                _diagnosis(
                    StallClassification.RECOVERY_BLOCKAGE,
                    now_ms=now,
                    reason=f"recovery is {status}",
                    severity=Severity.ERROR,
                    actionable=obs.ownership is not OwnershipState.UNKNOWN,
                    subject_kind="recovery",
                    subject_id=recovery_id,
                    task_cid=_text_field(recovery, "task_cid"),
                    evidence=[
                        _evidence(
                            "recovery",
                            subject_id=recovery_id,
                            observed_at_ms=now,
                            status=status,
                        )
                    ],
                )
            )

    # File mirrors: may contribute evidence but NEVER alone authorize action.
    for mirror in obs.file_mirrors:
        path = _text_field(mirror, "path", "file_path", "name")
        mtime_ms = _int_field(mirror, "mtime_ms", "modified_at_ms", "age_basis_ms")
        age_ms = _int_field(mirror, "age_ms")
        if not age_ms and mtime_ms:
            age_ms = max(0, now - mtime_ms)
        stale = _as_bool(mirror.get("stale")) or (
            age_ms > heartbeat_stale_ms if age_ms else False
        )
        if stale:
            diagnoses.append(
                _diagnosis(
                    StallClassification.FILE_AGE_ONLY,
                    now_ms=now,
                    reason="file mirror age is stale; insufficient alone for action",
                    severity=Severity.INFO,
                    actionable=False,
                    subject_kind="file_mirror",
                    subject_id=path or "file_mirror",
                    evidence=[
                        _evidence(
                            "file_age",
                            subject_id=path,
                            observed_at_ms=now,
                            age_ms=age_ms,
                            mtime_ms=mtime_ms,
                            authority="none",
                            note="file age alone never authorizes repair",
                        )
                    ],
                )
            )

    # Healthy active when live sessions exist and no actionable/critical faults.
    actionable_present = any(item.actionable for item in diagnoses)
    critical_present = any(
        item.severity in {Severity.ERROR, Severity.CRITICAL}
        and item.classification
        not in {
            StallClassification.FILE_AGE_ONLY,
            StallClassification.OWNERSHIP_UNKNOWN,
        }
        for item in diagnoses
    )
    live_sessions = [
        s
        for s in obs.sessions
        if _session_status(s) in {"active", "open", "running", ""}
    ]
    if live_sessions and not actionable_present and not critical_present:
        # Avoid double-reporting healthy when only capacity/quiescence/info present.
        non_info = [
            d
            for d in diagnoses
            if d.classification
            not in {
                StallClassification.PROVIDER_CAPACITY_BACKOFF,
                StallClassification.QUIESCENT_STRICT_SHARD,
                StallClassification.TERMINAL_DRAIN,
                StallClassification.EXPIRING_SESSION,
                StallClassification.FILE_AGE_ONLY,
                StallClassification.OWNERSHIP_UNKNOWN,
            }
        ]
        if not non_info:
            diagnoses.append(
                _diagnosis(
                    StallClassification.HEALTHY_ACTIVE,
                    now_ms=now,
                    reason="live sessions with fresh database heartbeats",
                    severity=Severity.INFO,
                    actionable=False,
                    subject_kind="control_plane",
                    subject_id="control_plane",
                    evidence=[
                        _evidence(
                            "healthy",
                            observed_at_ms=now,
                            live_session_count=len(live_sessions),
                            generation=obs.generation,
                        )
                    ],
                )
            )

    # When ownership is unknown, force all diagnoses non-actionable.
    if obs.ownership is OwnershipState.UNKNOWN:
        forced: list[StallDiagnosis] = []
        for item in diagnoses:
            if item.actionable:
                forced.append(
                    StallDiagnosis(
                        diagnosis_id=item.diagnosis_id,
                        classification=item.classification,
                        severity=item.severity,
                        actionable=False,
                        observed_at_ms=item.observed_at_ms,
                        reason=item.reason,
                        subject_kind=item.subject_kind,
                        subject_id=item.subject_id,
                        task_cid=item.task_cid,
                        session_id=item.session_id,
                        worktree_id=item.worktree_id,
                        lease_id=item.lease_id,
                        evidence=item.evidence,
                        body=dict(item.body),
                    )
                )
            else:
                forced.append(item)
        return forced

    return diagnoses


def classification_to_action(classification: StallClassification) -> CommandActionKind:
    """Map a classification to a safe fenced action kind."""

    mapping = {
        StallClassification.STALE_SESSION: CommandActionKind.EXPIRE_SESSION,
        StallClassification.ORPHAN_LEASE: CommandActionKind.RECLAIM_ORPHAN_LEASE,
        StallClassification.ORPHAN_WORKTREE: CommandActionKind.RECLAIM_ORPHAN_WORKTREE,
        StallClassification.READY_UNCLAIMABLE: CommandActionKind.MARK_READY_ACTIONABLE,
        StallClassification.PHASE_STALL: CommandActionKind.INTERRUPT_STALLED_PHASE,
        StallClassification.LOG_STALL: CommandActionKind.INTERRUPT_STALLED_PHASE,
        StallClassification.SERVER_FAULT: CommandActionKind.REQUEST_SERVER_RESTART,
        StallClassification.MIGRATION_FAULT: CommandActionKind.REQUEST_SERVER_RESTART,
        StallClassification.BACKUP_FAULT: CommandActionKind.REQUEST_BACKUP,
        StallClassification.MERGE_BLOCKAGE: CommandActionKind.UNBLOCK_MERGE,
        StallClassification.RECOVERY_BLOCKAGE: CommandActionKind.UNBLOCK_RECOVERY,
        StallClassification.OWNERSHIP_UNKNOWN: CommandActionKind.ABSTAIN,
        StallClassification.FILE_AGE_ONLY: CommandActionKind.NO_OP,
        StallClassification.HEALTHY_ACTIVE: CommandActionKind.NO_OP,
        StallClassification.QUIESCENT_STRICT_SHARD: CommandActionKind.NO_OP,
        StallClassification.PROVIDER_CAPACITY_BACKOFF: CommandActionKind.NO_OP,
        StallClassification.EXPIRING_SESSION: CommandActionKind.NO_OP,
        StallClassification.TERMINAL_DRAIN: CommandActionKind.NO_OP,
    }
    return mapping.get(classification, CommandActionKind.NO_OP)


# ---------------------------------------------------------------------------
# DatabaseWatchdog
# ---------------------------------------------------------------------------


class DatabaseWatchdog:
    """DuckDB-backed watchdog authority for stall diagnosis and fenced repair.

    Interface: ``DatabaseWatchdog@1`` with projected records
    ``StallDiagnosis@1`` and ``FencedRecoveryCommand@1``.
    """

    INTERFACE: ClassVar[str] = DATABASE_WATCHDOG_INTERFACE
    SCHEMA: ClassVar[str] = DATABASE_WATCHDOG_SCHEMA

    def __init__(
        self,
        database_path: Path | str,
        *,
        clock_ms: ClockMs | None = None,
        liveness: LivenessProbe | None = None,
        heartbeat_stale_ms: int = DEFAULT_HEARTBEAT_STALE_MS,
        session_expiry_ms: int = DEFAULT_SESSION_EXPIRY_MS,
        phase_deadline_grace_ms: int = DEFAULT_PHASE_DEADLINE_GRACE_MS,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DatabaseWatchdog; install the optional "
                "duckdb dependency"
            )
        self._path = Path(database_path)
        self._clock_ms = clock_ms or _default_clock_ms
        self._liveness = liveness
        self._heartbeat_stale_ms = _positive_int(
            int(heartbeat_stale_ms), "heartbeat_stale_ms"
        )
        self._session_expiry_ms = _positive_int(
            int(session_expiry_ms), "session_expiry_ms"
        )
        self._phase_deadline_grace_ms = _nonneg_int(
            int(phase_deadline_grace_ms), "phase_deadline_grace_ms"
        )
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._closed = True

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    def open(self) -> "DatabaseWatchdog":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._path)
            try:
                for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                    connection.execute(statement)
                for key, value in (
                    ("interface", DATABASE_WATCHDOG_INTERFACE),
                    ("schema", DATABASE_WATCHDOG_SCHEMA),
                ):
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO watchdog_metadata(key, value)
                        VALUES (?, ?)
                        """,
                        [key, value],
                    )
                self._connection = connection
                self._closed = False
                self._commit_if_idle(connection)
                return self
            except Exception:
                try:
                    connection.close()
                except Exception:
                    pass
                raise

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            self._connection = None
            self._closed = True
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def __enter__(self) -> "DatabaseWatchdog":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def authority_policy(self) -> dict[str, str]:
        return {
            "semantic_authority": "database",
            "watchdog_authority": "database",
            "file_mtime_authority": "none",
            "raw_pid_authority": "none",
            "lock_deletion": "prohibited",
            "interface": self.INTERFACE,
            "schema": self.SCHEMA,
        }

    def _require(self) -> Any:
        if not self.is_open or self._connection is None:
            raise DatabaseWatchdogNotOpenError("DatabaseWatchdog is not open")
        return self._connection

    def _begin(self, connection: Any) -> None:
        if getattr(connection, "in_transaction", False):
            return
        try:
            connection.execute("BEGIN TRANSACTION")
        except Exception:
            pass

    def _rollback_if_open(self, connection: Any) -> None:
        try:
            rollback = getattr(connection, "rollback", None)
            if callable(rollback) and getattr(connection, "in_transaction", False):
                rollback()
                return
            raw = getattr(connection, "_connection", None)
            raw_rollback = getattr(raw, "rollback", None) if raw is not None else None
            if callable(raw_rollback):
                raw_rollback()
        except Exception:
            pass

    def _commit_if_idle(self, connection: Any) -> None:
        try:
            if getattr(connection, "in_transaction", False):
                commit = getattr(connection, "commit", None)
                if callable(commit):
                    commit()
                    return
            raw = getattr(connection, "_connection", None)
            raw_commit = getattr(raw, "commit", None) if raw is not None else None
            if callable(raw_commit):
                raw_commit()
                return
            commit = getattr(connection, "commit", None)
            if callable(commit):
                commit()
        except Exception:
            pass

    def _now_ms(self) -> int:
        return int(self._clock_ms())

    def _record_event(
        self,
        connection: Any,
        *,
        event_type: str,
        command_id: str = "",
        diagnosis_id: str = "",
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> None:
        now = self._now_ms() if now_ms is None else int(now_ms)
        connection.execute(
            """
            INSERT INTO watchdog_events(
                event_id, command_id, diagnosis_id, event_type, observed_at_ms, body_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                _new_id("watchdog-event"),
                command_id,
                diagnosis_id,
                event_type,
                now,
                _canonical_json(_bounded_mapping(body, name="event body")),
            ],
        )

    # -- diagnose ------------------------------------------------------------

    def diagnose(
        self,
        observation: WatchdogObservation | Mapping[str, Any],
        *,
        persist: bool = True,
    ) -> list[StallDiagnosis]:
        """Diagnose one observation and optionally persist stall detections."""

        obs = (
            observation
            if isinstance(observation, WatchdogObservation)
            else WatchdogObservation.from_mapping(observation)
        )
        diagnoses = diagnose_observation(
            obs,
            heartbeat_stale_ms=self._heartbeat_stale_ms,
            session_expiry_ms=self._session_expiry_ms,
            phase_deadline_grace_ms=self._phase_deadline_grace_ms,
            liveness=self._liveness,
        )
        if not persist:
            return diagnoses
        connection = self._require()
        with self._lock:
            self._begin(connection)
            try:
                for item in diagnoses:
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO stall_detections(
                            diagnosis_id, classification, severity, actionable,
                            subject_kind, subject_id, task_cid, session_id,
                            worktree_id, lease_id, observed_at_ms, reason,
                            evidence_json, body_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            item.diagnosis_id,
                            item.classification.value,
                            item.severity.value,
                            bool(item.actionable),
                            item.subject_kind,
                            item.subject_id,
                            item.task_cid,
                            item.session_id,
                            item.worktree_id,
                            item.lease_id,
                            int(item.observed_at_ms),
                            item.reason,
                            _canonical_json([e.to_dict() for e in item.evidence]),
                            _canonical_json(dict(item.body)),
                        ],
                    )
                    self._record_event(
                        connection,
                        event_type="stall_detected",
                        diagnosis_id=item.diagnosis_id,
                        body={
                            "classification": item.classification.value,
                            "actionable": item.actionable,
                        },
                        now_ms=item.observed_at_ms,
                    )
                self._commit_if_idle(connection)
            except Exception:
                self._rollback_if_open(connection)
                raise
        return diagnoses

    def list_diagnoses(
        self,
        *,
        classification: StallClassification | str | None = None,
        actionable_only: bool = False,
        limit: int = 100,
    ) -> list[StallDiagnosis]:
        connection = self._require()
        clauses: list[str] = []
        params: list[Any] = []
        if classification is not None:
            cls = (
                classification
                if isinstance(classification, StallClassification)
                else StallClassification(str(classification).strip().lower())
            )
            clauses.append("classification = ?")
            params.append(cls.value)
        if actionable_only:
            clauses.append("actionable = TRUE")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_n = _positive_int(int(limit), "limit")
        sql = (
            f"SELECT * FROM stall_detections {where} "
            f"ORDER BY observed_at_ms DESC LIMIT {limit_n}"
        )
        cursor = (
            connection.execute(sql, params)
            if params
            else connection.execute(sql)
        )
        rows = cursor.fetchall()
        return [self._row_to_diagnosis(_row_mapping(row)) for row in rows]

    # -- fenced repair -------------------------------------------------------

    def decide_repair(
        self,
        diagnosis: StallDiagnosis,
        *,
        expected_fence_epoch: int,
        expected_fencing_token: int,
        expected_generation: int,
        expected_process_birth_id: str = "",
        expected_process_birth: ProcessBirthIdentity | Mapping[str, Any] | None = None,
        idempotency_key: str = "",
        reason: str = "",
        body: Mapping[str, Any] | None = None,
        current_fence_epoch: int | None = None,
        current_fencing_token: int | None = None,
        current_generation: int | None = None,
        current_process_birth_id: str = "",
        current_process_birth: ProcessBirthIdentity | Mapping[str, Any] | None = None,
        ownership: OwnershipState | str = OwnershipState.ABSENT,
    ) -> FencedRecoveryCommand:
        """Decide a fenced recovery command for one diagnosis.

        Requires current expected fence / process birth / generation. Identical
        idempotency keys replay the prior durable result. File-age-only
        diagnoses and unknown ownership never produce an applied action.
        """

        ownership_state = (
            ownership
            if isinstance(ownership, OwnershipState)
            else OwnershipState(str(ownership).strip().lower())
        )
        exp_epoch = _nonneg_int(int(expected_fence_epoch), "expected_fence_epoch")
        exp_token = _nonneg_int(int(expected_fencing_token), "expected_fencing_token")
        exp_gen = _nonneg_int(int(expected_generation), "expected_generation")

        birth_id = _text(
            expected_process_birth_id, "expected_process_birth_id", required=False
        )
        if not birth_id and expected_process_birth is not None:
            birth = (
                expected_process_birth
                if isinstance(expected_process_birth, ProcessBirthIdentity)
                else process_birth_from_mapping(_mapping(expected_process_birth))
            )
            birth_id = process_birth_id(birth)

        action = classification_to_action(diagnosis.classification)
        cmd_reason = reason or diagnosis.reason
        cmd_body = _bounded_mapping(body, name="command body")
        if not cmd_body and diagnosis.body:
            cmd_body = dict(diagnosis.body)

        key = _text(idempotency_key, "idempotency_key", required=False)
        if not key:
            key = _default_idempotency_key(
                action_kind=action.value,
                subject_id=diagnosis.subject_id,
                expected_fence_epoch=exp_epoch,
                expected_fencing_token=exp_token,
                expected_process_birth_id=birth_id,
                expected_generation=exp_gen,
                reason=cmd_reason,
                body=cmd_body,
            )

        connection = self._require()
        with self._lock:
            self._begin(connection)
            try:
                existing = connection.execute(
                    "SELECT * FROM fenced_recovery_commands WHERE idempotency_key = ?",
                    [key],
                ).fetchone()
                if existing is not None:
                    command = self._row_to_command(_row_mapping(existing))
                    # Idempotent replay: mark as replayed if previously applied/decided.
                    if command.status in {
                        CommandStatus.APPLIED,
                        CommandStatus.DECIDED,
                        CommandStatus.REPLAYED,
                        CommandStatus.ABSTAINED,
                        CommandStatus.REJECTED,
                    }:
                        if command.status is CommandStatus.APPLIED:
                            # Keep APPLIED identity; emit replay event.
                            self._record_event(
                                connection,
                                event_type="command_replayed",
                                command_id=command.command_id,
                                diagnosis_id=command.diagnosis_id,
                                body={"idempotency_key": key},
                            )
                            # Persist a REPLAYED projection only when re-reading
                            # an already applied command: update status in place
                            # is intentionally avoided so the original applied
                            # result remains the durable authority. Return a
                            # REPLAYED view for the caller.
                            replayed = FencedRecoveryCommand(
                                command_id=command.command_id,
                                action_kind=command.action_kind,
                                status=CommandStatus.REPLAYED,
                                idempotency_key=command.idempotency_key,
                                expected_fence_epoch=command.expected_fence_epoch,
                                expected_fencing_token=command.expected_fencing_token,
                                expected_generation=command.expected_generation,
                                decided_at_ms=command.decided_at_ms,
                                expected_process_birth_id=command.expected_process_birth_id,
                                diagnosis_id=command.diagnosis_id,
                                subject_kind=command.subject_kind,
                                subject_id=command.subject_id,
                                task_cid=command.task_cid,
                                session_id=command.session_id,
                                worktree_id=command.worktree_id,
                                applied_at_ms=command.applied_at_ms,
                                result_digest=command.result_digest,
                                reason=command.reason,
                                body=dict(command.body),
                            )
                            self._commit_if_idle(connection)
                            return replayed
                        self._commit_if_idle(connection)
                        return command

                # Policy: file age alone never authorizes repair.
                if diagnosis.classification is StallClassification.FILE_AGE_ONLY:
                    command = self._insert_command(
                        connection,
                        action_kind=CommandActionKind.NO_OP,
                        status=CommandStatus.REJECTED,
                        idempotency_key=key,
                        expected_fence_epoch=exp_epoch,
                        expected_fencing_token=exp_token,
                        expected_generation=exp_gen,
                        expected_process_birth_id=birth_id,
                        diagnosis=diagnosis,
                        reason="no action follows file age alone",
                        body=cmd_body,
                    )
                    self._commit_if_idle(connection)
                    return command

                if (
                    ownership_state is OwnershipState.UNKNOWN
                    or diagnosis.classification is StallClassification.OWNERSHIP_UNKNOWN
                    or action is CommandActionKind.ABSTAIN
                ):
                    command = self._insert_command(
                        connection,
                        action_kind=CommandActionKind.ABSTAIN,
                        status=CommandStatus.ABSTAINED,
                        idempotency_key=key,
                        expected_fence_epoch=exp_epoch,
                        expected_fencing_token=exp_token,
                        expected_generation=exp_gen,
                        expected_process_birth_id=birth_id,
                        diagnosis=diagnosis,
                        reason="ownership unknown; doctor abstains",
                        body=cmd_body,
                    )
                    self._commit_if_idle(connection)
                    return command

                if not diagnosis.actionable or action is CommandActionKind.NO_OP:
                    command = self._insert_command(
                        connection,
                        action_kind=CommandActionKind.NO_OP,
                        status=CommandStatus.DECIDED,
                        idempotency_key=key,
                        expected_fence_epoch=exp_epoch,
                        expected_fencing_token=exp_token,
                        expected_generation=exp_gen,
                        expected_process_birth_id=birth_id,
                        diagnosis=diagnosis,
                        reason=cmd_reason or "no repair required",
                        body=cmd_body,
                    )
                    self._commit_if_idle(connection)
                    return command

                # Fence / generation / process-birth checks against *current* state.
                cur_epoch = (
                    exp_epoch
                    if current_fence_epoch is None
                    else _nonneg_int(int(current_fence_epoch), "current_fence_epoch")
                )
                cur_token = (
                    exp_token
                    if current_fencing_token is None
                    else _nonneg_int(int(current_fencing_token), "current_fencing_token")
                )
                cur_gen = (
                    exp_gen
                    if current_generation is None
                    else _nonneg_int(int(current_generation), "current_generation")
                )
                cur_birth_id = _text(
                    current_process_birth_id,
                    "current_process_birth_id",
                    required=False,
                )
                if not cur_birth_id and current_process_birth is not None:
                    cur_birth = (
                        current_process_birth
                        if isinstance(current_process_birth, ProcessBirthIdentity)
                        else process_birth_from_mapping(
                            _mapping(current_process_birth)
                        )
                    )
                    cur_birth_id = process_birth_id(cur_birth)
                if not cur_birth_id:
                    cur_birth_id = birth_id

                fence_ok = (
                    cur_epoch == exp_epoch
                    and cur_token == exp_token
                    and cur_gen == exp_gen
                )
                birth_ok = True
                if birth_id:
                    birth_ok = cur_birth_id == birth_id
                    if (
                        birth_ok
                        and expected_process_birth is not None
                        and current_process_birth is not None
                    ):
                        exp_birth = (
                            expected_process_birth
                            if isinstance(expected_process_birth, ProcessBirthIdentity)
                            else process_birth_from_mapping(
                                _mapping(expected_process_birth)
                            )
                        )
                        cur_birth = (
                            current_process_birth
                            if isinstance(current_process_birth, ProcessBirthIdentity)
                            else process_birth_from_mapping(
                                _mapping(current_process_birth)
                            )
                        )
                        birth_ok = process_births_match(exp_birth, cur_birth)

                if not fence_ok or not birth_ok:
                    command = self._insert_command(
                        connection,
                        action_kind=action,
                        status=CommandStatus.REJECTED,
                        idempotency_key=key,
                        expected_fence_epoch=exp_epoch,
                        expected_fencing_token=exp_token,
                        expected_generation=exp_gen,
                        expected_process_birth_id=birth_id,
                        diagnosis=diagnosis,
                        reason=(
                            "stale or mismatched fence/process-birth/generation"
                        ),
                        body={
                            **cmd_body,
                            "current_fence_epoch": cur_epoch,
                            "current_fencing_token": cur_token,
                            "current_generation": cur_gen,
                            "current_process_birth_id": cur_birth_id,
                        },
                    )
                    self._record_event(
                        connection,
                        event_type="fence_rejected",
                        command_id=command.command_id,
                        diagnosis_id=diagnosis.diagnosis_id,
                        body={
                            "expected_fence_epoch": exp_epoch,
                            "current_fence_epoch": cur_epoch,
                            "expected_generation": exp_gen,
                            "current_generation": cur_gen,
                            "expected_process_birth_id": birth_id,
                            "current_process_birth_id": cur_birth_id,
                        },
                    )
                    self._commit_if_idle(connection)
                    return command

                command = self._insert_command(
                    connection,
                    action_kind=action,
                    status=CommandStatus.DECIDED,
                    idempotency_key=key,
                    expected_fence_epoch=exp_epoch,
                    expected_fencing_token=exp_token,
                    expected_generation=exp_gen,
                    expected_process_birth_id=birth_id,
                    diagnosis=diagnosis,
                    reason=cmd_reason,
                    body=cmd_body,
                )
                self._commit_if_idle(connection)
                return command
            except Exception:
                self._rollback_if_open(connection)
                raise

    def apply_repair(
        self,
        command: FencedRecoveryCommand | str,
        *,
        current_fence_epoch: int | None = None,
        current_fencing_token: int | None = None,
        current_generation: int | None = None,
        current_process_birth_id: str = "",
        current_process_birth: ProcessBirthIdentity | Mapping[str, Any] | None = None,
        ownership: OwnershipState | str = OwnershipState.ABSENT,
        result_body: Mapping[str, Any] | None = None,
    ) -> FencedRecoveryCommand:
        """Apply a previously decided fenced recovery command (idempotent).

        Re-validates fence / process birth / generation at apply time. Never
        signals a raw PID or deletes a lock file.
        """

        ownership_state = (
            ownership
            if isinstance(ownership, OwnershipState)
            else OwnershipState(str(ownership).strip().lower())
        )
        connection = self._require()
        with self._lock:
            self._begin(connection)
            try:
                if isinstance(command, FencedRecoveryCommand):
                    command_id = command.command_id
                else:
                    command_id = _text(command, "command_id")
                row = connection.execute(
                    "SELECT * FROM fenced_recovery_commands WHERE command_id = ?",
                    [command_id],
                ).fetchone()
                if row is None:
                    raise DatabaseWatchdogError(f"unknown command_id: {command_id}")
                current = self._row_to_command(_row_mapping(row))

                if current.status is CommandStatus.APPLIED:
                    # Idempotent: already applied.
                    self._record_event(
                        connection,
                        event_type="command_replayed",
                        command_id=current.command_id,
                        diagnosis_id=current.diagnosis_id,
                        body={"idempotency_key": current.idempotency_key},
                    )
                    self._commit_if_idle(connection)
                    return FencedRecoveryCommand(
                        command_id=current.command_id,
                        action_kind=current.action_kind,
                        status=CommandStatus.REPLAYED,
                        idempotency_key=current.idempotency_key,
                        expected_fence_epoch=current.expected_fence_epoch,
                        expected_fencing_token=current.expected_fencing_token,
                        expected_generation=current.expected_generation,
                        decided_at_ms=current.decided_at_ms,
                        expected_process_birth_id=current.expected_process_birth_id,
                        diagnosis_id=current.diagnosis_id,
                        subject_kind=current.subject_kind,
                        subject_id=current.subject_id,
                        task_cid=current.task_cid,
                        session_id=current.session_id,
                        worktree_id=current.worktree_id,
                        applied_at_ms=current.applied_at_ms,
                        result_digest=current.result_digest,
                        reason=current.reason,
                        body=dict(current.body),
                    )

                if current.status in {
                    CommandStatus.REJECTED,
                    CommandStatus.ABSTAINED,
                }:
                    self._commit_if_idle(connection)
                    return current

                if current.action_kind in {
                    CommandActionKind.NO_OP,
                    CommandActionKind.ABSTAIN,
                }:
                    # Nothing to apply.
                    now = self._now_ms()
                    digest = _sha256_hex(
                        _canonical_json(
                            {
                                "command_id": current.command_id,
                                "action_kind": current.action_kind.value,
                                "status": current.status.value,
                            }
                        ).encode("utf-8")
                    )
                    connection.execute(
                        """
                        UPDATE fenced_recovery_commands
                        SET status = ?, applied_at_ms = ?, result_digest = ?
                        WHERE command_id = ?
                        """,
                        [
                            CommandStatus.APPLIED.value
                            if current.action_kind is CommandActionKind.NO_OP
                            else CommandStatus.ABSTAINED.value,
                            now,
                            digest,
                            current.command_id,
                        ],
                    )
                    self._commit_if_idle(connection)
                    return self.get_command(current.command_id)  # type: ignore[return-value]

                if ownership_state is OwnershipState.UNKNOWN:
                    raise DatabaseWatchdogOwnershipError(
                        "cannot apply repair while ownership is unknown"
                    )

                # Re-check fence at apply time.
                cur_epoch = (
                    current.expected_fence_epoch
                    if current_fence_epoch is None
                    else _nonneg_int(int(current_fence_epoch), "current_fence_epoch")
                )
                cur_token = (
                    current.expected_fencing_token
                    if current_fencing_token is None
                    else _nonneg_int(
                        int(current_fencing_token), "current_fencing_token"
                    )
                )
                cur_gen = (
                    current.expected_generation
                    if current_generation is None
                    else _nonneg_int(int(current_generation), "current_generation")
                )
                cur_birth_id = _text(
                    current_process_birth_id,
                    "current_process_birth_id",
                    required=False,
                )
                if not cur_birth_id and current_process_birth is not None:
                    cur_birth = (
                        current_process_birth
                        if isinstance(current_process_birth, ProcessBirthIdentity)
                        else process_birth_from_mapping(
                            _mapping(current_process_birth)
                        )
                    )
                    cur_birth_id = process_birth_id(cur_birth)
                if not cur_birth_id:
                    cur_birth_id = current.expected_process_birth_id

                if (
                    cur_epoch != current.expected_fence_epoch
                    or cur_token != current.expected_fencing_token
                    or cur_gen != current.expected_generation
                    or (
                        current.expected_process_birth_id
                        and cur_birth_id != current.expected_process_birth_id
                    )
                ):
                    connection.execute(
                        """
                        UPDATE fenced_recovery_commands
                        SET status = ?, reason = ?
                        WHERE command_id = ?
                        """,
                        [
                            CommandStatus.REJECTED.value,
                            "stale fence/process-birth/generation at apply",
                            current.command_id,
                        ],
                    )
                    self._record_event(
                        connection,
                        event_type="fence_rejected_at_apply",
                        command_id=current.command_id,
                        diagnosis_id=current.diagnosis_id,
                        body={
                            "expected_fence_epoch": current.expected_fence_epoch,
                            "current_fence_epoch": cur_epoch,
                            "expected_generation": current.expected_generation,
                            "current_generation": cur_gen,
                            "expected_process_birth_id": current.expected_process_birth_id,
                            "current_process_birth_id": cur_birth_id,
                        },
                    )
                    self._commit_if_idle(connection)
                    updated = self.get_command(current.command_id)
                    assert updated is not None
                    return updated

                # Safe apply: record decision only (no PID signal / lock delete).
                now = self._now_ms()
                result = {
                    "action_kind": current.action_kind.value,
                    "subject_id": current.subject_id,
                    "applied_at_ms": now,
                    "effects": "database_projection_only",
                    "raw_pid_signal": False,
                    "lock_deletion": False,
                    "result_body": dict(result_body or {}),
                }
                digest = _sha256_hex(_canonical_json(result).encode("utf-8"))
                connection.execute(
                    """
                    UPDATE fenced_recovery_commands
                    SET status = ?, applied_at_ms = ?, result_digest = ?,
                        body_json = ?
                    WHERE command_id = ?
                    """,
                    [
                        CommandStatus.APPLIED.value,
                        now,
                        digest,
                        _canonical_json({**dict(current.body), "apply_result": result}),
                        current.command_id,
                    ],
                )
                connection.execute(
                    """
                    INSERT INTO restart_decisions(
                        decision_id, command_id, diagnosis_id, disposition,
                        observed_at_ms, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        _new_id("restart-decision"),
                        current.command_id,
                        current.diagnosis_id,
                        current.action_kind.value,
                        now,
                        _canonical_json(result),
                    ],
                )
                self._record_event(
                    connection,
                    event_type="command_applied",
                    command_id=current.command_id,
                    diagnosis_id=current.diagnosis_id,
                    body=result,
                    now_ms=now,
                )
                self._commit_if_idle(connection)
                updated = self.get_command(current.command_id)
                assert updated is not None
                return updated
            except Exception:
                self._rollback_if_open(connection)
                raise

    def decide_and_apply(
        self,
        diagnosis: StallDiagnosis,
        *,
        expected_fence_epoch: int,
        expected_fencing_token: int,
        expected_generation: int,
        expected_process_birth_id: str = "",
        expected_process_birth: ProcessBirthIdentity | Mapping[str, Any] | None = None,
        idempotency_key: str = "",
        reason: str = "",
        body: Mapping[str, Any] | None = None,
        current_fence_epoch: int | None = None,
        current_fencing_token: int | None = None,
        current_generation: int | None = None,
        current_process_birth_id: str = "",
        current_process_birth: ProcessBirthIdentity | Mapping[str, Any] | None = None,
        ownership: OwnershipState | str = OwnershipState.ABSENT,
    ) -> FencedRecoveryCommand:
        """Decide then apply a fenced repair in one call (still idempotent)."""

        decided = self.decide_repair(
            diagnosis,
            expected_fence_epoch=expected_fence_epoch,
            expected_fencing_token=expected_fencing_token,
            expected_generation=expected_generation,
            expected_process_birth_id=expected_process_birth_id,
            expected_process_birth=expected_process_birth,
            idempotency_key=idempotency_key,
            reason=reason,
            body=body,
            current_fence_epoch=current_fence_epoch,
            current_fencing_token=current_fencing_token,
            current_generation=current_generation,
            current_process_birth_id=current_process_birth_id,
            current_process_birth=current_process_birth,
            ownership=ownership,
        )
        if decided.status in {
            CommandStatus.REJECTED,
            CommandStatus.ABSTAINED,
            CommandStatus.REPLAYED,
        }:
            return decided
        return self.apply_repair(
            decided,
            current_fence_epoch=current_fence_epoch
            if current_fence_epoch is not None
            else expected_fence_epoch,
            current_fencing_token=current_fencing_token
            if current_fencing_token is not None
            else expected_fencing_token,
            current_generation=current_generation
            if current_generation is not None
            else expected_generation,
            current_process_birth_id=current_process_birth_id
            or expected_process_birth_id,
            current_process_birth=current_process_birth or expected_process_birth,
            ownership=ownership,
        )

    def get_command(self, command_id: str) -> FencedRecoveryCommand | None:
        connection = self._require()
        row = connection.execute(
            "SELECT * FROM fenced_recovery_commands WHERE command_id = ?",
            [_text(command_id, "command_id")],
        ).fetchone()
        if row is None:
            return None
        return self._row_to_command(_row_mapping(row))

    def get_command_by_idempotency_key(
        self, idempotency_key: str
    ) -> FencedRecoveryCommand | None:
        connection = self._require()
        row = connection.execute(
            "SELECT * FROM fenced_recovery_commands WHERE idempotency_key = ?",
            [_text(idempotency_key, "idempotency_key")],
        ).fetchone()
        if row is None:
            return None
        return self._row_to_command(_row_mapping(row))

    # -- doctor surface ------------------------------------------------------

    def doctor(
        self,
        observation: WatchdogObservation | Mapping[str, Any],
        *,
        persist: bool = True,
        propose_repairs: bool = False,
    ) -> DoctorReport:
        """Expose evidence and diagnoses; abstain when ownership is unknown."""

        obs = (
            observation
            if isinstance(observation, WatchdogObservation)
            else WatchdogObservation.from_mapping(observation)
        )
        diagnoses = self.diagnose(obs, persist=persist)
        evidence: list[WatchdogEvidence] = []
        for item in diagnoses:
            evidence.extend(item.evidence)
        evidence.append(
            _evidence(
                "ownership",
                subject_id="server",
                observed_at_ms=obs.now_ms,
                ownership=obs.ownership.value,
                generation=obs.generation,
                fence_epoch=obs.fence_epoch,
                process_birth_id=obs.server_process_birth_id,
            )
        )

        commands: list[FencedRecoveryCommand] = []
        if obs.ownership is OwnershipState.UNKNOWN:
            return DoctorReport(
                disposition=DoctorDisposition.ABSTAIN,
                ownership=obs.ownership,
                diagnoses=tuple(diagnoses),
                evidence=tuple(evidence),
                observed_at_ms=obs.now_ms,
                reason="ownership unknown; exposing evidence and abstaining",
                commands=(),
                body={
                    "policy": self.authority_policy(),
                    "file_age_alone_action": False,
                },
            )

        actionable = [d for d in diagnoses if d.actionable]
        if propose_repairs and actionable:
            for item in actionable:
                cmd = self.decide_repair(
                    item,
                    expected_fence_epoch=obs.fence_epoch,
                    expected_fencing_token=obs.fencing_token
                    or _int_field(item.body, "fencing_token"),
                    expected_generation=obs.generation
                    or _int_field(item.body, "generation", "server_generation", default=1),
                    expected_process_birth_id=(
                        obs.server_process_birth_id
                        or _text_field(item.body, "process_birth_id")
                    ),
                    expected_process_birth=obs.server_process_birth,
                    current_fence_epoch=obs.fence_epoch,
                    current_fencing_token=obs.fencing_token
                    or _int_field(item.body, "fencing_token"),
                    current_generation=obs.generation,
                    current_process_birth_id=obs.server_process_birth_id,
                    current_process_birth=obs.server_process_birth,
                    ownership=obs.ownership,
                )
                commands.append(cmd)

        if actionable:
            disposition = DoctorDisposition.ACTIONABLE
            reason = f"{len(actionable)} actionable diagnosis(es)"
        elif any(
            d.classification is StallClassification.HEALTHY_ACTIVE for d in diagnoses
        ):
            disposition = DoctorDisposition.HEALTHY
            reason = "healthy active control plane"
        else:
            disposition = DoctorDisposition.REPORT
            reason = "diagnostic report only; no actionable repair"

        return DoctorReport(
            disposition=disposition,
            ownership=obs.ownership,
            diagnoses=tuple(diagnoses),
            evidence=tuple(evidence),
            observed_at_ms=obs.now_ms,
            reason=reason,
            commands=tuple(commands),
            body={
                "policy": self.authority_policy(),
                "file_age_alone_action": False,
                "actionable_count": len(actionable),
            },
        )

    # -- row mappers ---------------------------------------------------------

    def _insert_command(
        self,
        connection: Any,
        *,
        action_kind: CommandActionKind,
        status: CommandStatus,
        idempotency_key: str,
        expected_fence_epoch: int,
        expected_fencing_token: int,
        expected_generation: int,
        expected_process_birth_id: str,
        diagnosis: StallDiagnosis,
        reason: str,
        body: Mapping[str, Any],
    ) -> FencedRecoveryCommand:
        now = self._now_ms()
        command_id = _new_id("command")
        connection.execute(
            """
            INSERT INTO fenced_recovery_commands(
                command_id, diagnosis_id, action_kind, status, idempotency_key,
                expected_fence_epoch, expected_fencing_token,
                expected_process_birth_id, expected_generation,
                subject_kind, subject_id, task_cid, session_id, worktree_id,
                decided_at_ms, reason, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                command_id,
                diagnosis.diagnosis_id,
                action_kind.value,
                status.value,
                idempotency_key,
                int(expected_fence_epoch),
                int(expected_fencing_token),
                expected_process_birth_id,
                int(expected_generation),
                diagnosis.subject_kind,
                diagnosis.subject_id,
                diagnosis.task_cid,
                diagnosis.session_id,
                diagnosis.worktree_id,
                now,
                reason,
                _canonical_json(dict(body)),
            ],
        )
        self._record_event(
            connection,
            event_type="command_decided",
            command_id=command_id,
            diagnosis_id=diagnosis.diagnosis_id,
            body={
                "action_kind": action_kind.value,
                "status": status.value,
                "idempotency_key": idempotency_key,
            },
            now_ms=now,
        )
        return FencedRecoveryCommand(
            command_id=command_id,
            action_kind=action_kind,
            status=status,
            idempotency_key=idempotency_key,
            expected_fence_epoch=expected_fence_epoch,
            expected_fencing_token=expected_fencing_token,
            expected_generation=expected_generation,
            decided_at_ms=now,
            expected_process_birth_id=expected_process_birth_id,
            diagnosis_id=diagnosis.diagnosis_id,
            subject_kind=diagnosis.subject_kind,
            subject_id=diagnosis.subject_id,
            task_cid=diagnosis.task_cid,
            session_id=diagnosis.session_id,
            worktree_id=diagnosis.worktree_id,
            reason=reason,
            body=dict(body),
        )

    def _row_to_diagnosis(self, mapping: Mapping[str, Any]) -> StallDiagnosis:
        evidence_raw = _row_get(mapping, "evidence_json", default="[]")
        try:
            evidence_list = json.loads(evidence_raw) if evidence_raw else []
        except (TypeError, ValueError, json.JSONDecodeError):
            evidence_list = []
        body_raw = _row_get(mapping, "body_json", default="{}")
        try:
            body = json.loads(body_raw) if body_raw else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            body = {}
        return StallDiagnosis(
            diagnosis_id=str(_row_get(mapping, "diagnosis_id") or ""),
            classification=str(_row_get(mapping, "classification") or ""),
            severity=str(_row_get(mapping, "severity") or "info"),
            actionable=_as_bool(_row_get(mapping, "actionable")),
            observed_at_ms=int(_row_get(mapping, "observed_at_ms", default=0) or 0),
            reason=str(_row_get(mapping, "reason") or ""),
            subject_kind=str(_row_get(mapping, "subject_kind") or ""),
            subject_id=str(_row_get(mapping, "subject_id") or ""),
            task_cid=str(_row_get(mapping, "task_cid") or ""),
            session_id=str(_row_get(mapping, "session_id") or ""),
            worktree_id=str(_row_get(mapping, "worktree_id") or ""),
            lease_id=str(_row_get(mapping, "lease_id") or ""),
            evidence=tuple(
                WatchdogEvidence.from_mapping(item)
                for item in evidence_list
                if isinstance(item, Mapping)
            ),
            body=body if isinstance(body, Mapping) else {},
        )

    def _row_to_command(self, mapping: Mapping[str, Any]) -> FencedRecoveryCommand:
        body_raw = _row_get(mapping, "body_json", default="{}")
        try:
            body = json.loads(body_raw) if body_raw else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            body = {}
        return FencedRecoveryCommand(
            command_id=str(_row_get(mapping, "command_id") or ""),
            action_kind=str(_row_get(mapping, "action_kind") or "no_op"),
            status=str(_row_get(mapping, "status") or "decided"),
            idempotency_key=str(_row_get(mapping, "idempotency_key") or ""),
            expected_fence_epoch=int(
                _row_get(mapping, "expected_fence_epoch", default=0) or 0
            ),
            expected_fencing_token=int(
                _row_get(mapping, "expected_fencing_token", default=0) or 0
            ),
            expected_generation=int(
                _row_get(mapping, "expected_generation", default=0) or 0
            ),
            decided_at_ms=int(_row_get(mapping, "decided_at_ms", default=0) or 0),
            expected_process_birth_id=str(
                _row_get(mapping, "expected_process_birth_id") or ""
            ),
            diagnosis_id=str(_row_get(mapping, "diagnosis_id") or ""),
            subject_kind=str(_row_get(mapping, "subject_kind") or ""),
            subject_id=str(_row_get(mapping, "subject_id") or ""),
            task_cid=str(_row_get(mapping, "task_cid") or ""),
            session_id=str(_row_get(mapping, "session_id") or ""),
            worktree_id=str(_row_get(mapping, "worktree_id") or ""),
            applied_at_ms=int(_row_get(mapping, "applied_at_ms", default=0) or 0),
            result_digest=str(_row_get(mapping, "result_digest") or ""),
            reason=str(_row_get(mapping, "reason") or ""),
            body=body if isinstance(body, Mapping) else {},
        )


def open_database_watchdog(
    database_path: Path | str,
    *,
    clock_ms: ClockMs | None = None,
    liveness: LivenessProbe | None = None,
    heartbeat_stale_ms: int = DEFAULT_HEARTBEAT_STALE_MS,
    session_expiry_ms: int = DEFAULT_SESSION_EXPIRY_MS,
    phase_deadline_grace_ms: int = DEFAULT_PHASE_DEADLINE_GRACE_MS,
) -> DatabaseWatchdog:
    """Open (or create) a DatabaseWatchdog store."""

    watchdog = DatabaseWatchdog(
        database_path,
        clock_ms=clock_ms,
        liveness=liveness,
        heartbeat_stale_ms=heartbeat_stale_ms,
        session_expiry_ms=session_expiry_ms,
        phase_deadline_grace_ms=phase_deadline_grace_ms,
    )
    return watchdog.open()


def build_database_watchdog(
    database_path: Path | str,
    **kwargs: Any,
) -> DatabaseWatchdog:
    """Alias for :func:`open_database_watchdog` (factory naming parity)."""

    return open_database_watchdog(database_path, **kwargs)


__all__ = [
    "DATABASE_WATCHDOG_INTERFACE",
    "STALL_DIAGNOSIS_INTERFACE",
    "FENCED_RECOVERY_COMMAND_INTERFACE",
    "DATABASE_WATCHDOG_SCHEMA",
    "STALL_DIAGNOSIS_SCHEMA",
    "FENCED_RECOVERY_COMMAND_SCHEMA",
    "CLASSIFICATION_VALUES",
    "DEFAULT_HEARTBEAT_STALE_MS",
    "DEFAULT_SESSION_EXPIRY_MS",
    "CommandActionKind",
    "CommandStatus",
    "DatabaseWatchdog",
    "DatabaseWatchdogConflictError",
    "DatabaseWatchdogError",
    "DatabaseWatchdogFenceError",
    "DatabaseWatchdogNotOpenError",
    "DatabaseWatchdogOwnershipError",
    "DatabaseWatchdogPolicyError",
    "DoctorDisposition",
    "DoctorReport",
    "DuckDBUnavailableError",
    "FencedRecoveryCommand",
    "OwnershipState",
    "Severity",
    "StallClassification",
    "StallDiagnosis",
    "WatchdogEvidence",
    "WatchdogObservation",
    "build_database_watchdog",
    "classification_to_action",
    "diagnose_observation",
    "duckdb_available",
    "open_database_watchdog",
    "process_birth_id",
    "process_births_match",
]
