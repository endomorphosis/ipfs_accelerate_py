"""Backfill legacy state and prove shadow decision parity (DQP-037).

Interfaces: ``DatabaseShadowRollout@1``, ``ShadowParityReport@1``

This module reconciles reviewed legacy programs into the control-plane
database, dual-observes production (legacy) reads and lifecycle decisions
against non-authoritative shadow database projections, and emits a
content-addressed parity report.

Safety invariants (non-waivable):

* Shadow writes never control production effect; only the production
  authority channel may mutate schedule/lifecycle outcomes.
* Dual observation is temporary evidence collection with bounded duration
  and retention — never a permanent dual-authority architecture.
* Every authority-relevant drift requires an explicit reviewed disposition;
  unexplained drift fails closed and blocks parity pass.
* Rollback changes the authority/read route without deleting history.
* Exact re-run of the same backfill + observation inputs yields the same
  parity decision.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..task_sources.legacy_state_import import (
    ConflictPolicy,
    ImportDomain,
    ImportManifest,
    ImportMediaType,
    ImportMode,
    ImportReceipt,
    ImportSourceSpec,
    LegacyStateImport,
    OUTCOME_APPLIED,
    OUTCOME_REPLAYED,
    build_import_manifest,
    duckdb_available,
)
from ..task_sources.task_identity import canonical_content_cid, canonical_json_bytes
from ..task_sources.task_source import (
    StateAuthorityMode,
    state_authority_mode_policy,
)


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_SHADOW_ROLLOUT_INTERFACE: Final[str] = "DatabaseShadowRollout@1"
SHADOW_PARITY_REPORT_INTERFACE: Final[str] = "ShadowParityReport@1"

DATABASE_SHADOW_ROLLOUT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-shadow-rollout@1"
)
SHADOW_PARITY_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/shadow-parity-report@1"
)
SHADOW_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/shadow-observation@1"
)
DRIFT_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/shadow-drift-record@1"
)
DUAL_OBSERVATION_WINDOW_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dual-observation-window@1"
)
SHADOW_ROLLBACK_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/shadow-rollback-receipt@1"
)
SHADOW_HISTORY_ENTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/shadow-history-entry@1"
)

DATABASE_SHADOW_ROLLOUT_VERSION: Final[int] = 1
TASK_ID: Final[str] = "DQP-037"
GOAL_ID: Final[str] = "DQP-G080"
EVIDENCE: Final[str] = "dqp/database-shadow-rollout@1"

# Authority-relevant comparison surfaces (evidence subset for DQP-037).
AUTHORITY_SURFACES: Final[tuple[str, ...]] = (
    "counts_digests",
    "duplicate_conflict",
    "task_cid",
    "readiness",
    "lease_fence",
    "status",
    "event_cursor",
    "completion",
    "restart",
    "export",
)

# Dual-observation bounds (fail-closed ceilings; operators may tighten).
DEFAULT_MAX_DURATION_SECONDS: Final[int] = 7 * 24 * 60 * 60  # 7 days
DEFAULT_MAX_RETENTION_RECORDS: Final[int] = 10_000
DEFAULT_MAX_RETENTION_BYTES: Final[int] = 16 * 1024 * 1024
MAX_REASON_CODES: Final[int] = 128
MAX_TEXT_BYTES: Final[int] = 512
MAX_ID_BYTES: Final[int] = 512
MAX_DISPOSITIONS: Final[int] = 4_096
MAX_OBSERVATIONS: Final[int] = 50_000
MAX_HISTORY_ENTRIES: Final[int] = 50_000

# Production effect marker: shadow channel is never allowed to set this.
PRODUCTION_EFFECT_CHANNEL: Final[str] = "production"
SHADOW_EFFECT_CHANNEL: Final[str] = "shadow"
EXPORT_NON_AUTHORITY_MARKER: Final[str] = "EXPORT_NON_AUTHORITATIVE"

Clock = Callable[[], float]


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class ShadowRolloutStage(str, Enum):
    """Staged dual-observation path prior to canary/cutover (DQP-038)."""

    OFF = "off"
    BACKFILL = "backfill"
    OBSERVE = "observe"
    SHADOW = "shadow"


class ParityDecision(str, Enum):
    """Closed parity outcomes for one shadow evaluation."""

    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"


class DriftDispositionKind(str, Enum):
    """Reviewed dispositions for authority-relevant drift.

    ``match`` is automatic when sides agree. Every non-match requires an
    explicit reviewed disposition; ``unexplained`` fails closed.
    """

    MATCH = "match"
    REVIEWED_ACCEPT = "reviewed_accept"
    REVIEWED_IGNORE = "reviewed_ignore"
    REVIEWED_QUARANTINE = "reviewed_quarantine"
    UNEXPLAINED = "unexplained"


class HistoryKind(str, Enum):
    BACKFILL = "backfill"
    OBSERVATION = "observation"
    PARITY = "parity"
    ROLLBACK = "rollback"
    EXPIRE = "expire"
    EFFECT = "effect"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseShadowRolloutError(RuntimeError):
    """Base fail-closed error for shadow rollout."""


class ShadowParityError(DatabaseShadowRolloutError):
    """Parity evaluation failed or is blocked by unexplained drift."""


class ShadowAuthorityError(DatabaseShadowRolloutError):
    """Shadow channel attempted a production-controlling action."""


class DualObservationBoundError(DatabaseShadowRolloutError):
    """Dual observation exceeded duration or retention bounds."""


class ShadowDispositionError(DatabaseShadowRolloutError, ValueError):
    """A drift disposition is missing, unknown, or malformed."""


class ShadowStageError(DatabaseShadowRolloutError, ValueError):
    """An illegal stage transition was requested."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _utc_iso(value: datetime | float | None = None) -> str:
    if value is None:
        moment = _utc_now()
    elif isinstance(value, (int, float)):
        moment = datetime.fromtimestamp(float(value), tz=timezone.utc).replace(
            microsecond=0
        )
    else:
        moment = value
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (
        moment.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return canonical_json_bytes(_plain(value))
    except (TypeError, ValueError) as exc:
        raise DatabaseShadowRolloutError(
            "shadow rollout data must be canonical JSON"
        ) from exc


def _identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _content_cid(value: Any) -> str:
    return canonical_content_cid(_plain(value))


def _text(value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DatabaseShadowRolloutError(
            f"{name} must be non-empty canonical text"
        )
    if "\x00" in value or len(value.encode("utf-8")) > maximum:
        raise DatabaseShadowRolloutError(f"{name} is unsafe or too large")
    return value


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DatabaseShadowRolloutError("expected a mapping")
    return {str(key): value[key] for key in value}


def _digest_of_surface(payload: Mapping[str, Any] | None) -> str:
    if not payload:
        return _identity({})
    return _identity(dict(payload))


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DualObservationWindow:
    """Bounded dual-observation duration and retention policy.

    Interface projection for dual-observation bounds on DatabaseShadowRollout@1.
    """

    SCHEMA: ClassVar[str] = DUAL_OBSERVATION_WINDOW_SCHEMA

    max_duration_seconds: int = DEFAULT_MAX_DURATION_SECONDS
    max_retention_records: int = DEFAULT_MAX_RETENTION_RECORDS
    max_retention_bytes: int = DEFAULT_MAX_RETENTION_BYTES
    opened_at: str = ""
    expires_at: str = ""

    def __post_init__(self) -> None:
        if int(self.max_duration_seconds) <= 0:
            raise DualObservationBoundError(
                "max_duration_seconds must be positive"
            )
        if int(self.max_duration_seconds) > DEFAULT_MAX_DURATION_SECONDS:
            raise DualObservationBoundError(
                "max_duration_seconds exceeds hard ceiling "
                f"of {DEFAULT_MAX_DURATION_SECONDS}s"
            )
        if int(self.max_retention_records) <= 0:
            raise DualObservationBoundError(
                "max_retention_records must be positive"
            )
        if int(self.max_retention_records) > DEFAULT_MAX_RETENTION_RECORDS:
            raise DualObservationBoundError(
                "max_retention_records exceeds hard ceiling "
                f"of {DEFAULT_MAX_RETENTION_RECORDS}"
            )
        if int(self.max_retention_bytes) <= 0:
            raise DualObservationBoundError(
                "max_retention_bytes must be positive"
            )
        if int(self.max_retention_bytes) > DEFAULT_MAX_RETENTION_BYTES:
            raise DualObservationBoundError(
                "max_retention_bytes exceeds hard ceiling "
                f"of {DEFAULT_MAX_RETENTION_BYTES}"
            )
        object.__setattr__(
            self, "max_duration_seconds", int(self.max_duration_seconds)
        )
        object.__setattr__(
            self, "max_retention_records", int(self.max_retention_records)
        )
        object.__setattr__(
            self, "max_retention_bytes", int(self.max_retention_bytes)
        )

    def open(self, *, now: float | None = None) -> "DualObservationWindow":
        opened = float(now if now is not None else time.time())
        expires = opened + float(self.max_duration_seconds)
        return DualObservationWindow(
            max_duration_seconds=self.max_duration_seconds,
            max_retention_records=self.max_retention_records,
            max_retention_bytes=self.max_retention_bytes,
            opened_at=_utc_iso(opened),
            expires_at=_utc_iso(expires),
        )

    def is_expired(self, *, now: float | None = None) -> bool:
        if not self.expires_at:
            return False
        current = float(now if now is not None else time.time())
        expires = datetime.fromisoformat(
            self.expires_at.replace("Z", "+00:00")
        ).timestamp()
        return current > expires

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "max_duration_seconds": self.max_duration_seconds,
            "max_retention_records": self.max_retention_records,
            "max_retention_bytes": self.max_retention_bytes,
            "opened_at": self.opened_at,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class DriftDisposition:
    """Operator-reviewed disposition for one authority-relevant drift."""

    surface: str
    kind: DriftDispositionKind
    reason: str
    reviewer: str = "operator"
    record_id: str = ""

    def __post_init__(self) -> None:
        surface = _text(self.surface, "surface", maximum=MAX_ID_BYTES)
        if surface not in AUTHORITY_SURFACES:
            raise ShadowDispositionError(
                f"unknown authority surface {surface!r}; closed set is "
                f"{', '.join(AUTHORITY_SURFACES)}"
            )
        kind = self.kind
        if not isinstance(kind, DriftDispositionKind):
            try:
                kind = DriftDispositionKind(str(kind))
            except ValueError as exc:
                raise ShadowDispositionError(
                    f"unknown disposition kind {self.kind!r}"
                ) from exc
        if kind is DriftDispositionKind.MATCH:
            raise ShadowDispositionError(
                "match is automatic; do not register match dispositions"
            )
        if kind is DriftDispositionKind.UNEXPLAINED:
            raise ShadowDispositionError(
                "unexplained is not a valid reviewed disposition"
            )
        reason = _text(self.reason, "reason")
        reviewer = _text(self.reviewer, "reviewer", maximum=MAX_ID_BYTES)
        record_id = str(self.record_id or "").strip()
        if record_id and (
            "\x00" in record_id or len(record_id.encode("utf-8")) > MAX_ID_BYTES
        ):
            raise ShadowDispositionError("record_id is unsafe or too large")
        object.__setattr__(self, "surface", surface)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "reviewer", reviewer)
        object.__setattr__(self, "record_id", record_id)

    @property
    def disposition_key(self) -> str:
        if self.record_id:
            return f"{self.surface}:{self.record_id}"
        return self.surface

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "kind": self.kind.value,
            "reason": self.reason,
            "reviewer": self.reviewer,
            "record_id": self.record_id,
            "disposition_key": self.disposition_key,
        }


@dataclass(frozen=True)
class DriftRecord:
    """One surface comparison between production and shadow."""

    SCHEMA: ClassVar[str] = DRIFT_RECORD_SCHEMA

    surface: str
    production_digest: str
    shadow_digest: str
    matched: bool
    disposition: DriftDispositionKind
    reason: str = ""
    record_id: str = ""
    production_value: Mapping[str, Any] = field(default_factory=dict)
    shadow_value: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "surface": self.surface,
            "record_id": self.record_id,
            "production_digest": self.production_digest,
            "shadow_digest": self.shadow_digest,
            "matched": bool(self.matched),
            "disposition": self.disposition.value,
            "reason": self.reason,
            "authority_relevant": self.surface in AUTHORITY_SURFACES,
            "production_value": dict(self.production_value or {}),
            "shadow_value": dict(self.shadow_value or {}),
        }


@dataclass(frozen=True)
class ShadowObservation:
    """One dual-observed read or lifecycle decision pair."""

    SCHEMA: ClassVar[str] = SHADOW_OBSERVATION_SCHEMA

    observation_id: str
    kind: str
    production: Mapping[str, Any]
    shadow: Mapping[str, Any]
    observed_at: str
    production_digest: str
    shadow_digest: str
    matched: bool
    surface: str = "status"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "observation_id": self.observation_id,
            "kind": self.kind,
            "surface": self.surface,
            "production": dict(self.production),
            "shadow": dict(self.shadow),
            "observed_at": self.observed_at,
            "production_digest": self.production_digest,
            "shadow_digest": self.shadow_digest,
            "matched": bool(self.matched),
        }


@dataclass(frozen=True)
class ShadowParityReport:
    """Content-addressed parity decision for one shadow evaluation.

    Interface: ShadowParityReport@1
    """

    SCHEMA: ClassVar[str] = SHADOW_PARITY_REPORT_SCHEMA
    INTERFACE: ClassVar[str] = SHADOW_PARITY_REPORT_INTERFACE

    report_id: str
    decision: ParityDecision
    stage: ShadowRolloutStage
    import_reconciled: bool
    import_receipt_cid: str
    production_snapshot_digest: str
    shadow_snapshot_digest: str
    drifts: tuple[DriftRecord, ...]
    unexplained_drift_count: int
    reviewed_disposition_count: int
    matched_surface_count: int
    dual_observation: DualObservationWindow
    shadow_controls_production: bool
    history_preserved: bool
    reason_codes: tuple[str, ...]
    evidence_subset: Mapping[str, Any]
    created_at: str
    parity_digest: str = ""
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID

    def __post_init__(self) -> None:
        if not self.parity_digest:
            object.__setattr__(self, "parity_digest", self._compute_parity_digest())

    def _compute_parity_digest(self) -> str:
        body = {
            "interface": self.INTERFACE,
            "schema": self.SCHEMA,
            "decision": self.decision.value,
            "stage": self.stage.value,
            "import_reconciled": self.import_reconciled,
            "import_receipt_cid": self.import_receipt_cid,
            "production_snapshot_digest": self.production_snapshot_digest,
            "shadow_snapshot_digest": self.shadow_snapshot_digest,
            "drifts": [item.to_dict() for item in self.drifts],
            "unexplained_drift_count": self.unexplained_drift_count,
            "reviewed_disposition_count": self.reviewed_disposition_count,
            "matched_surface_count": self.matched_surface_count,
            "dual_observation": self.dual_observation.to_dict(),
            "shadow_controls_production": self.shadow_controls_production,
            "history_preserved": self.history_preserved,
            "reason_codes": list(self.reason_codes),
            "evidence_subset": dict(self.evidence_subset),
            "task_id": self.task_id,
            "goal_id": self.goal_id,
        }
        return _identity(body)

    @property
    def passed(self) -> bool:
        return (
            self.decision is ParityDecision.PASS
            and self.import_reconciled
            and self.unexplained_drift_count == 0
            and not self.shadow_controls_production
        )

    def require_passed(self) -> "ShadowParityReport":
        if not self.passed:
            raise ShadowParityError(
                "shadow parity did not pass: "
                + ", ".join(self.reason_codes or ("parity_failed",))
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "report_id": self.report_id,
            "decision": self.decision.value,
            "stage": self.stage.value,
            "passed": self.passed,
            "import_reconciled": self.import_reconciled,
            "import_receipt_cid": self.import_receipt_cid,
            "production_snapshot_digest": self.production_snapshot_digest,
            "shadow_snapshot_digest": self.shadow_snapshot_digest,
            "drifts": [item.to_dict() for item in self.drifts],
            "unexplained_drift_count": int(self.unexplained_drift_count),
            "reviewed_disposition_count": int(self.reviewed_disposition_count),
            "matched_surface_count": int(self.matched_surface_count),
            "dual_observation": self.dual_observation.to_dict(),
            "shadow_controls_production": bool(self.shadow_controls_production),
            "history_preserved": bool(self.history_preserved),
            "reason_codes": list(self.reason_codes),
            "evidence_subset": dict(self.evidence_subset),
            "created_at": self.created_at,
            "parity_digest": self.parity_digest,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "evidence": EVIDENCE,
            "version": DATABASE_SHADOW_ROLLOUT_VERSION,
        }


@dataclass(frozen=True)
class ShadowRollbackReceipt:
    """Receipt for a kill-switch rollback that preserves history."""

    SCHEMA: ClassVar[str] = SHADOW_ROLLBACK_RECEIPT_SCHEMA

    receipt_id: str
    from_stage: str
    to_stage: str
    reason: str
    history_preserved: bool
    history_entry_count: int
    prior_parity_digest: str
    rolled_back_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "receipt_id": self.receipt_id,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "reason": self.reason,
            "history_preserved": bool(self.history_preserved),
            "history_entry_count": int(self.history_entry_count),
            "prior_parity_digest": self.prior_parity_digest,
            "rolled_back_at": self.rolled_back_at,
            "reason_codes": ["rollback", "history_preserved"],
        }


@dataclass(frozen=True)
class HistoryEntry:
    """Append-only history record for backfill/observe/parity/rollback."""

    SCHEMA: ClassVar[str] = SHADOW_HISTORY_ENTRY_SCHEMA

    entry_id: str
    kind: HistoryKind
    recorded_at: str
    payload: Mapping[str, Any]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "entry_id": self.entry_id,
            "kind": self.kind.value,
            "recorded_at": self.recorded_at,
            "payload": dict(self.payload),
            "digest": self.digest,
        }


@dataclass(frozen=True)
class ProductionEffect:
    """An effect applied exclusively through the production channel."""

    effect_id: str
    channel: str
    action: str
    payload: Mapping[str, Any]
    applied_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "effect_id": self.effect_id,
            "channel": self.channel,
            "action": self.action,
            "payload": dict(self.payload),
            "applied_at": self.applied_at,
        }


# ---------------------------------------------------------------------------
# Snapshot projection for parity
# ---------------------------------------------------------------------------


def project_authority_snapshot(
    *,
    tasks: Sequence[Mapping[str, Any]] = (),
    readiness: Sequence[str] = (),
    events: Sequence[Mapping[str, Any]] = (),
    revisions: Mapping[str, int] | None = None,
    leases: Sequence[Mapping[str, Any]] = (),
    statuses: Mapping[str, str] | None = None,
    exports: Sequence[Mapping[str, Any]] = (),
    completions: Sequence[Mapping[str, Any]] = (),
    restarts: Sequence[Mapping[str, Any]] = (),
    counts: Mapping[str, int] | None = None,
    digests: Mapping[str, str] | None = None,
    conflicts: Sequence[Mapping[str, Any]] = (),
    duplicates: Sequence[Mapping[str, Any]] = (),
    event_cursor: str = "",
) -> dict[str, Any]:
    """Build a closed authority-relevant snapshot for parity comparison."""

    task_list = [dict(item) for item in tasks]
    task_cids = tuple(
        sorted(
            str(item.get("task_cid") or item.get("id") or "")
            for item in task_list
            if str(item.get("task_cid") or item.get("id") or "")
        )
    )
    ready = tuple(sorted(str(item) for item in readiness if str(item)))
    event_list = [dict(item) for item in events]
    lease_list = [dict(item) for item in leases]
    status_map = {
        str(key): str(value) for key, value in dict(statuses or {}).items()
    }
    revision_map = {
        str(key): int(value) for key, value in dict(revisions or {}).items()
    }
    export_list = [dict(item) for item in exports]
    completion_list = [dict(item) for item in completions]
    restart_list = [dict(item) for item in restarts]
    count_map = {str(key): int(value) for key, value in dict(counts or {}).items()}
    digest_map = {
        str(key): str(value) for key, value in dict(digests or {}).items()
    }
    conflict_list = [dict(item) for item in conflicts]
    duplicate_list = [dict(item) for item in duplicates]

    if not count_map:
        count_map = {
            "tasks": len(task_list),
            "events": len(event_list),
            "leases": len(lease_list),
            "ready": len(ready),
            "exports": len(export_list),
            "completions": len(completion_list),
        }
    if not digest_map:
        digest_map = {
            "tasks": _identity(task_list),
            "events": _identity(event_list),
            "leases": _identity(lease_list),
            "statuses": _identity(status_map),
        }

    return {
        "counts_digests": {
            "counts": count_map,
            "digests": digest_map,
        },
        "duplicate_conflict": {
            "duplicates": duplicate_list,
            "conflicts": conflict_list,
        },
        "task_cid": {"task_cids": list(task_cids), "tasks": task_list},
        "readiness": {"ready_task_cids": list(ready)},
        "lease_fence": {"leases": lease_list},
        "status": {"statuses": status_map, "revisions": revision_map},
        "event_cursor": {
            "cursor": str(event_cursor or ""),
            "events": event_list,
        },
        "completion": {"completions": completion_list},
        "restart": {"restarts": restart_list},
        "export": {
            "exports": export_list,
            "non_authority_marker": EXPORT_NON_AUTHORITY_MARKER,
        },
    }


def _surface_payload(
    snapshot: Mapping[str, Any], surface: str
) -> dict[str, Any]:
    value = snapshot.get(surface)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


# ---------------------------------------------------------------------------
# DatabaseShadowRollout
# ---------------------------------------------------------------------------


_ALLOWED_TRANSITIONS: Final[Mapping[ShadowRolloutStage, frozenset[ShadowRolloutStage]]] = (
    MappingProxyType(
        {
            ShadowRolloutStage.OFF: frozenset(
                {ShadowRolloutStage.BACKFILL, ShadowRolloutStage.OFF}
            ),
            ShadowRolloutStage.BACKFILL: frozenset(
                {
                    ShadowRolloutStage.OBSERVE,
                    ShadowRolloutStage.SHADOW,
                    ShadowRolloutStage.OFF,
                }
            ),
            ShadowRolloutStage.OBSERVE: frozenset(
                {
                    ShadowRolloutStage.SHADOW,
                    ShadowRolloutStage.OFF,
                    ShadowRolloutStage.BACKFILL,
                }
            ),
            ShadowRolloutStage.SHADOW: frozenset(
                {
                    ShadowRolloutStage.OBSERVE,
                    ShadowRolloutStage.OFF,
                    ShadowRolloutStage.BACKFILL,
                }
            ),
        }
    )
)


class DatabaseShadowRollout:
    """Backfill legacy state and prove non-authoritative shadow parity.

    Interface: DatabaseShadowRollout@1

    Production effects always flow through the production channel. Shadow
    projections may be dual-observed and retained under explicit bounds, but
    never become schedule or lifecycle authority until a later canary decision
    (DQP-038).
    """

    INTERFACE: ClassVar[str] = DATABASE_SHADOW_ROLLOUT_INTERFACE
    SCHEMA: ClassVar[str] = DATABASE_SHADOW_ROLLOUT_SCHEMA

    def __init__(
        self,
        *,
        work_dir: Path | str | None = None,
        source_root: Path | str | None = None,
        target_database: Path | str | None = None,
        window: DualObservationWindow | None = None,
        dispositions: Sequence[DriftDisposition] = (),
        clock: Clock | None = None,
        importer: LegacyStateImport | None = None,
        authority_mode: StateAuthorityMode | str = StateAuthorityMode.QUACK_SHADOW,
    ) -> None:
        self.work_dir = Path(work_dir) if work_dir is not None else None
        self.source_root = Path(source_root) if source_root is not None else None
        self.target_database = (
            Path(target_database) if target_database is not None else None
        )
        self._window_policy = window or DualObservationWindow()
        self._window: DualObservationWindow | None = None
        self._clock: Clock = clock or time.time
        self._lock = threading.RLock()
        self._stage = ShadowRolloutStage.OFF
        self._dispositions: dict[str, DriftDisposition] = {}
        for item in dispositions:
            self.register_disposition(item)
        self._history: list[HistoryEntry] = []
        self._observations: list[ShadowObservation] = []
        self._production_effects: list[ProductionEffect] = []
        self._shadow_records: list[dict[str, Any]] = []
        self._production_snapshot: dict[str, Any] = project_authority_snapshot()
        self._shadow_snapshot: dict[str, Any] = project_authority_snapshot()
        self._import_receipt: ImportReceipt | None = None
        self._last_parity: ShadowParityReport | None = None
        self._authority_mode = (
            authority_mode
            if isinstance(authority_mode, StateAuthorityMode)
            else StateAuthorityMode(str(authority_mode))
        )
        policy = state_authority_mode_policy(self._authority_mode)
        self._dual_observation_enabled = bool(policy.dual_observation)
        if importer is not None:
            self._importer = importer
        else:
            self._importer = LegacyStateImport(
                target_database=self.target_database,
                source_root=self.source_root,
            )

    # -- identity -----------------------------------------------------------

    @property
    def interface(self) -> str:
        return self.INTERFACE

    @property
    def stage(self) -> ShadowRolloutStage:
        return self._stage

    @property
    def authority_mode(self) -> StateAuthorityMode:
        return self._authority_mode

    @property
    def dual_observation_enabled(self) -> bool:
        return self._dual_observation_enabled

    @property
    def window(self) -> DualObservationWindow | None:
        return self._window

    @property
    def import_receipt(self) -> ImportReceipt | None:
        return self._import_receipt

    @property
    def last_parity_report(self) -> ShadowParityReport | None:
        return self._last_parity

    @property
    def history(self) -> tuple[HistoryEntry, ...]:
        return tuple(self._history)

    @property
    def observations(self) -> tuple[ShadowObservation, ...]:
        return tuple(self._observations)

    @property
    def production_effects(self) -> tuple[ProductionEffect, ...]:
        return tuple(self._production_effects)

    @property
    def shadow_controls_production(self) -> bool:
        """Shadow never controls production under this interface."""

        return False

    # -- dispositions -------------------------------------------------------

    def register_disposition(self, disposition: DriftDisposition) -> None:
        with self._lock:
            if len(self._dispositions) >= MAX_DISPOSITIONS:
                raise ShadowDispositionError("disposition registry is full")
            self._dispositions[disposition.disposition_key] = disposition

    def clear_dispositions(self) -> None:
        with self._lock:
            self._dispositions.clear()

    # -- stage control ------------------------------------------------------

    def transition(
        self,
        to_stage: ShadowRolloutStage | str,
        *,
        reason: str = "operator",
    ) -> ShadowRolloutStage:
        with self._lock:
            target = (
                to_stage
                if isinstance(to_stage, ShadowRolloutStage)
                else ShadowRolloutStage(str(to_stage))
            )
            allowed = _ALLOWED_TRANSITIONS[self._stage]
            if target not in allowed:
                raise ShadowStageError(
                    f"transition {self._stage.value} -> {target.value} "
                    "is not allowed"
                )
            prior = self._stage
            self._stage = target
            if target in (
                ShadowRolloutStage.OBSERVE,
                ShadowRolloutStage.SHADOW,
            ) and self._window is None:
                self._window = self._window_policy.open(now=self._clock())
            if target is ShadowRolloutStage.OFF:
                # Closing dual observation expires the window but keeps history.
                if self._window is not None:
                    self._append_history(
                        HistoryKind.EXPIRE,
                        {
                            "reason": reason,
                            "from_stage": prior.value,
                            "window": self._window.to_dict(),
                        },
                    )
            self._append_history(
                HistoryKind.OBSERVATION
                if target in (ShadowRolloutStage.OBSERVE, ShadowRolloutStage.SHADOW)
                else HistoryKind.BACKFILL
                if target is ShadowRolloutStage.BACKFILL
                else HistoryKind.EXPIRE,
                {
                    "transition": {
                        "from": prior.value,
                        "to": target.value,
                        "reason": reason,
                    }
                },
            )
            return self._stage

    # -- backfill -----------------------------------------------------------

    def backfill(
        self,
        manifest: ImportManifest | Mapping[str, Any],
        *,
        production_snapshot: Mapping[str, Any] | None = None,
    ) -> ImportReceipt:
        """Apply exact legacy import and seed dual-observation snapshots.

        Exact import must reconcile (accepted rows with digests; no silent
        last-write-wins). Replay of an identical import is a no-op with the
        same receipt identity semantics as LegacyStateImport@1.
        """

        with self._lock:
            if self._stage is ShadowRolloutStage.OFF:
                self._stage = ShadowRolloutStage.BACKFILL
            elif self._stage not in (
                ShadowRolloutStage.BACKFILL,
                ShadowRolloutStage.OBSERVE,
                ShadowRolloutStage.SHADOW,
            ):
                raise ShadowStageError(
                    f"backfill refused in stage {self._stage.value}"
                )

            resolved = (
                manifest
                if isinstance(manifest, ImportManifest)
                else ImportManifest.from_dict(manifest)
            )
            # Force apply mode for backfill of reviewed programs.
            working = ImportManifest(
                import_id=resolved.import_id,
                sources=resolved.sources,
                mode=ImportMode.APPLY,
                strict=resolved.strict,
                default_conflict_policy=resolved.default_conflict_policy,
                conflict_resolutions=resolved.conflict_resolutions,
                target_database=resolved.target_database
                or (
                    str(self.target_database)
                    if self.target_database is not None
                    else ""
                ),
                metadata=dict(resolved.metadata),
            )
            receipt = self._importer.apply(working)
            if receipt.outcome not in (OUTCOME_APPLIED, OUTCOME_REPLAYED):
                raise DatabaseShadowRolloutError(
                    f"import did not reconcile: outcome={receipt.outcome}"
                )
            if receipt.strict and receipt.rejected_rows:
                raise DatabaseShadowRolloutError(
                    "strict import left rejected rows; not reconciled"
                )

            self._import_receipt = receipt
            shadow_rows = [dict(row) for row in receipt.accepted_rows]
            self._shadow_records = shadow_rows
            shadow_snapshot = self._snapshot_from_import(receipt)
            if production_snapshot is not None:
                production = self._normalize_snapshot(production_snapshot)
            else:
                # Default: production matches the reconciled import projection.
                production = dict(shadow_snapshot)
            self._production_snapshot = production
            self._shadow_snapshot = shadow_snapshot

            self._append_history(
                HistoryKind.BACKFILL,
                {
                    "import_id": receipt.import_id,
                    "receipt_cid": receipt.receipt_cid,
                    "manifest_cid": receipt.manifest_cid,
                    "outcome": receipt.outcome,
                    "accepted_count": len(receipt.accepted_rows),
                    "rejected_count": len(receipt.rejected_rows),
                    "conflict_count": len(receipt.conflicts),
                    "replayed": bool(receipt.replayed),
                    "production_snapshot_digest": _identity(production),
                    "shadow_snapshot_digest": _identity(shadow_snapshot),
                },
            )
            return receipt

    def _snapshot_from_import(self, receipt: ImportReceipt) -> dict[str, Any]:
        tasks: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        leases: list[dict[str, Any]] = []
        statuses: dict[str, str] = {}
        revisions: dict[str, int] = {}
        readiness: list[str] = []
        completions: list[dict[str, Any]] = []
        for row in receipt.accepted_rows:
            domain = str(row.get("domain") or "")
            record_id = str(row.get("record_id") or "")
            payload = dict(row.get("payload") or {})
            task_cid = str(
                payload.get("task_cid")
                or row.get("row_cid")
                or record_id
            )
            if domain in (
                ImportDomain.TASKBOARDS.value,
                ImportDomain.QUEUES.value,
                ImportDomain.OBJECTIVES.value,
                ImportDomain.GENERIC.value,
                ImportDomain.PLAN_REVISIONS.value,
            ):
                status = str(payload.get("status") or "todo")
                tasks.append(
                    {
                        "id": record_id,
                        "task_cid": task_cid,
                        "status": status,
                        "title": str(payload.get("title") or ""),
                        "priority": str(payload.get("priority") or ""),
                    }
                )
                statuses[task_cid] = status
                revisions[task_cid] = int(payload.get("revision") or 0)
                if status in ("todo", "ready", "queued"):
                    readiness.append(task_cid)
                if status in ("completed", "complete", "done"):
                    completions.append(
                        {"task_cid": task_cid, "status": status}
                    )
            elif domain == ImportDomain.EVENTS.value:
                events.append(
                    {
                        "event_id": record_id,
                        "task_id": str(payload.get("task_id") or ""),
                        "kind": str(payload.get("kind") or ""),
                    }
                )
            elif domain == ImportDomain.LEASES.value:
                leases.append(
                    {
                        "lease_id": record_id,
                        "owner": str(payload.get("owner") or ""),
                        "task_id": str(payload.get("task_id") or ""),
                        "fence": int(payload.get("fence") or 0),
                    }
                )
            elif domain == ImportDomain.STATUSES.value:
                statuses[task_cid] = str(payload.get("status") or "todo")
                revisions[task_cid] = int(payload.get("revision") or 0)

        conflicts = [item.to_dict() for item in receipt.conflicts]
        duplicates = [
            item.to_dict()
            for item in receipt.conflicts
            if item.decision == "deduplicated"
        ]
        digests = {
            observation.source_id: observation.source_digest
            for observation in receipt.source_observations
        }
        return project_authority_snapshot(
            tasks=tasks,
            readiness=readiness,
            events=events,
            revisions=revisions,
            leases=leases,
            statuses=statuses,
            completions=completions,
            counts={
                "tasks": len(tasks),
                "events": len(events),
                "leases": len(leases),
                "ready": len(readiness),
                "accepted": len(receipt.accepted_rows),
                "rejected": len(receipt.rejected_rows),
                "conflicts": len(receipt.conflicts),
            },
            digests=digests,
            conflicts=conflicts,
            duplicates=duplicates,
            event_cursor=str(len(events)),
            exports=[
                {
                    "marker": EXPORT_NON_AUTHORITY_MARKER,
                    "import_receipt_cid": receipt.receipt_cid,
                }
            ],
            restarts=[],
        )

    def _normalize_snapshot(
        self, snapshot: Mapping[str, Any]
    ) -> dict[str, Any]:
        # Accept either a full multi-surface snapshot or a flat projection.
        if any(surface in snapshot for surface in AUTHORITY_SURFACES):
            normalized: dict[str, Any] = {}
            for surface in AUTHORITY_SURFACES:
                normalized[surface] = dict(snapshot.get(surface) or {})
            return normalized
        return project_authority_snapshot(
            tasks=list(snapshot.get("tasks") or ()),
            readiness=list(snapshot.get("readiness") or snapshot.get("ready") or ()),
            events=list(snapshot.get("events") or ()),
            revisions=dict(snapshot.get("revisions") or {}),
            leases=list(snapshot.get("leases") or ()),
            statuses=dict(snapshot.get("statuses") or {}),
            exports=list(snapshot.get("exports") or ()),
            completions=list(snapshot.get("completions") or ()),
            restarts=list(snapshot.get("restarts") or ()),
            counts=dict(snapshot.get("counts") or {}),
            digests=dict(snapshot.get("digests") or {}),
            conflicts=list(snapshot.get("conflicts") or ()),
            duplicates=list(snapshot.get("duplicates") or ()),
            event_cursor=str(snapshot.get("event_cursor") or ""),
        )

    # -- dual observation ---------------------------------------------------

    def _ensure_window_open(self) -> DualObservationWindow:
        if self._window is None:
            self._window = self._window_policy.open(now=self._clock())
        if self._window.is_expired(now=self._clock()):
            raise DualObservationBoundError(
                "dual observation window has expired; "
                f"expires_at={self._window.expires_at}"
            )
        if len(self._observations) >= self._window.max_retention_records:
            raise DualObservationBoundError(
                "dual observation retention record bound exceeded "
                f"({self._window.max_retention_records})"
            )
        # Approximate byte bound from canonical observation history size.
        body = _canonical_bytes([item.to_dict() for item in self._observations])
        if len(body) >= self._window.max_retention_bytes:
            raise DualObservationBoundError(
                "dual observation retention byte bound exceeded "
                f"({self._window.max_retention_bytes})"
            )
        return self._window

    def observe_decision(
        self,
        *,
        kind: str,
        production: Mapping[str, Any],
        shadow: Mapping[str, Any] | None = None,
        surface: str = "status",
    ) -> ShadowObservation:
        """Record one dual-observed decision without applying production effect.

        The shadow side is non-authoritative. Callers that need a production
        effect must use :meth:`apply_production_effect`.
        """

        with self._lock:
            if self._stage not in (
                ShadowRolloutStage.OBSERVE,
                ShadowRolloutStage.SHADOW,
                ShadowRolloutStage.BACKFILL,
            ):
                # Auto-enter observe when backfill already completed.
                if self._import_receipt is not None:
                    self._stage = ShadowRolloutStage.OBSERVE
                else:
                    raise ShadowStageError(
                        "observe_decision requires backfill or observe/shadow stage"
                    )
            window = self._ensure_window_open()
            if len(self._observations) >= MAX_OBSERVATIONS:
                raise DualObservationBoundError(
                    f"observation cap {MAX_OBSERVATIONS} exceeded"
                )

            surface_name = _text(surface, "surface", maximum=MAX_ID_BYTES)
            if surface_name not in AUTHORITY_SURFACES:
                raise DatabaseShadowRolloutError(
                    f"unknown observation surface {surface_name!r}"
                )
            kind_text = _text(kind, "kind", maximum=MAX_ID_BYTES)
            production_map = dict(production)
            if shadow is None:
                # Default shadow mirrors production for exact parity path.
                shadow_map = dict(production_map)
            else:
                shadow_map = dict(shadow)
            production_digest = _identity(production_map)
            shadow_digest = _identity(shadow_map)
            matched = production_digest == shadow_digest
            observed_at = _utc_iso(self._clock())
            observation_id = _identity(
                {
                    "kind": kind_text,
                    "surface": surface_name,
                    "production_digest": production_digest,
                    "shadow_digest": shadow_digest,
                    "observed_at": observed_at,
                    "ordinal": len(self._observations),
                }
            )
            observation = ShadowObservation(
                observation_id=observation_id,
                kind=kind_text,
                production=MappingProxyType(production_map),
                shadow=MappingProxyType(shadow_map),
                observed_at=observed_at,
                production_digest=production_digest,
                shadow_digest=shadow_digest,
                matched=matched,
                surface=surface_name,
            )
            self._observations.append(observation)

            # Update live snapshots for the observed surface.
            prod_surface = _surface_payload(self._production_snapshot, surface_name)
            shadow_surface = _surface_payload(self._shadow_snapshot, surface_name)
            prod_surface = {**prod_surface, **production_map}
            shadow_surface = {**shadow_surface, **shadow_map}
            self._production_snapshot = {
                **self._production_snapshot,
                surface_name: prod_surface,
            }
            self._shadow_snapshot = {
                **self._shadow_snapshot,
                surface_name: shadow_surface,
            }

            self._append_history(
                HistoryKind.OBSERVATION,
                {
                    "observation_id": observation_id,
                    "kind": kind_text,
                    "surface": surface_name,
                    "matched": matched,
                    "window": window.to_dict(),
                },
            )
            return observation

    def apply_production_effect(
        self,
        *,
        action: str,
        payload: Mapping[str, Any],
        channel: str = PRODUCTION_EFFECT_CHANNEL,
    ) -> ProductionEffect:
        """Apply an effect exclusively through the production channel.

        Attempts to route effects through the shadow channel are refused.
        """

        with self._lock:
            channel_name = str(channel or "").strip().lower()
            if channel_name in ("shadow", SHADOW_EFFECT_CHANNEL):
                raise ShadowAuthorityError(
                    "shadow channel cannot control production effect"
                )
            if channel_name not in ("", PRODUCTION_EFFECT_CHANNEL, "production", "legacy"):
                # Unknown channels are refused rather than silently accepted.
                if channel_name != PRODUCTION_EFFECT_CHANNEL:
                    raise ShadowAuthorityError(
                        f"unknown effect channel {channel!r}; "
                        f"only {PRODUCTION_EFFECT_CHANNEL!r} is authoritative"
                    )
            action_text = _text(action, "action", maximum=MAX_ID_BYTES)
            body = dict(payload)
            applied_at = _utc_iso(self._clock())
            effect_id = _identity(
                {
                    "action": action_text,
                    "payload": body,
                    "applied_at": applied_at,
                    "ordinal": len(self._production_effects),
                }
            )
            effect = ProductionEffect(
                effect_id=effect_id,
                channel=PRODUCTION_EFFECT_CHANNEL,
                action=action_text,
                payload=MappingProxyType(body),
                applied_at=applied_at,
            )
            self._production_effects.append(effect)
            self._append_history(
                HistoryKind.EFFECT,
                {
                    "effect_id": effect_id,
                    "channel": PRODUCTION_EFFECT_CHANNEL,
                    "action": action_text,
                    "shadow_controls_production": False,
                },
            )
            return effect

    def shadow_write(
        self,
        *,
        action: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Record a non-authoritative shadow write (never a production effect)."""

        with self._lock:
            if self._stage not in (
                ShadowRolloutStage.OBSERVE,
                ShadowRolloutStage.SHADOW,
                ShadowRolloutStage.BACKFILL,
            ):
                raise ShadowStageError(
                    "shadow_write requires observe/shadow/backfill stage"
                )
            self._ensure_window_open()
            action_text = _text(action, "action", maximum=MAX_ID_BYTES)
            body = dict(payload)
            record = {
                "channel": SHADOW_EFFECT_CHANNEL,
                "action": action_text,
                "payload": body,
                "authoritative": False,
                "controls_production": False,
                "recorded_at": _utc_iso(self._clock()),
            }
            record["record_id"] = _identity(record)
            self._shadow_records.append(dict(record))
            self._append_history(
                HistoryKind.OBSERVATION,
                {
                    "shadow_write": record["record_id"],
                    "authoritative": False,
                    "controls_production": False,
                },
            )
            return dict(record)

    # -- parity -------------------------------------------------------------

    def evaluate_parity(
        self,
        *,
        production_snapshot: Mapping[str, Any] | None = None,
        shadow_snapshot: Mapping[str, Any] | None = None,
    ) -> ShadowParityReport:
        """Compare production vs shadow and require reviewed drift dispositions."""

        with self._lock:
            if self._import_receipt is None:
                raise ShadowParityError(
                    "parity requires a reconciled backfill import first"
                )
            if self._stage is ShadowRolloutStage.OFF:
                raise ShadowStageError(
                    "parity evaluation refused while stage is off"
                )

            # Duration bound still applies during active dual observation.
            if self._window is not None and self._window.is_expired(
                now=self._clock()
            ):
                raise DualObservationBoundError(
                    "dual observation window expired before parity evaluation"
                )

            production = (
                self._normalize_snapshot(production_snapshot)
                if production_snapshot is not None
                else dict(self._production_snapshot)
            )
            shadow = (
                self._normalize_snapshot(shadow_snapshot)
                if shadow_snapshot is not None
                else dict(self._shadow_snapshot)
            )
            self._production_snapshot = production
            self._shadow_snapshot = shadow

            drifts: list[DriftRecord] = []
            unexplained = 0
            reviewed = 0
            matched = 0
            reason_codes: list[str] = []

            for surface in AUTHORITY_SURFACES:
                prod_value = _surface_payload(production, surface)
                shadow_value = _surface_payload(shadow, surface)
                prod_digest = _digest_of_surface(prod_value)
                shadow_digest = _digest_of_surface(shadow_value)
                is_match = prod_digest == shadow_digest
                if is_match:
                    matched += 1
                    drifts.append(
                        DriftRecord(
                            surface=surface,
                            production_digest=prod_digest,
                            shadow_digest=shadow_digest,
                            matched=True,
                            disposition=DriftDispositionKind.MATCH,
                            reason="exact match",
                            production_value=MappingProxyType(prod_value),
                            shadow_value=MappingProxyType(shadow_value),
                        )
                    )
                    continue

                disposition = self._resolve_disposition(surface)
                if disposition is None:
                    unexplained += 1
                    drifts.append(
                        DriftRecord(
                            surface=surface,
                            production_digest=prod_digest,
                            shadow_digest=shadow_digest,
                            matched=False,
                            disposition=DriftDispositionKind.UNEXPLAINED,
                            reason="authority-relevant drift lacks reviewed disposition",
                            production_value=MappingProxyType(prod_value),
                            shadow_value=MappingProxyType(shadow_value),
                        )
                    )
                    reason_codes.append(f"unexplained_drift:{surface}")
                else:
                    reviewed += 1
                    drifts.append(
                        DriftRecord(
                            surface=surface,
                            production_digest=prod_digest,
                            shadow_digest=shadow_digest,
                            matched=False,
                            disposition=disposition.kind,
                            reason=disposition.reason,
                            record_id=disposition.record_id,
                            production_value=MappingProxyType(prod_value),
                            shadow_value=MappingProxyType(shadow_value),
                        )
                    )
                    reason_codes.append(
                        f"reviewed_drift:{surface}:{disposition.kind.value}"
                    )

            import_reconciled = self._import_is_reconciled(self._import_receipt)
            if not import_reconciled:
                reason_codes.append("import_not_reconciled")

            if self.shadow_controls_production:
                reason_codes.append("shadow_controls_production")

            if self._dual_observation_enabled:
                reason_codes.append("dual_observation_non_authoritative")

            # Determine decision.
            if not import_reconciled or self.shadow_controls_production:
                decision = ParityDecision.BLOCKED
                reason_codes.append("parity_blocked")
            elif unexplained > 0:
                decision = ParityDecision.FAIL
                reason_codes.append("unexplained_authority_relevant_drift")
            else:
                decision = ParityDecision.PASS
                reason_codes.append("parity_pass")

            # Stable, content-addressed reason code order (dedupe, preserve).
            stable_reasons = tuple(dict.fromkeys(reason_codes))[:MAX_REASON_CODES]

            window = self._window or self._window_policy
            evidence = self._evidence_subset(
                production=production,
                shadow=shadow,
                drifts=drifts,
                receipt=self._import_receipt,
            )
            created_at = _utc_iso(self._clock())
            report_body = {
                "decision": decision.value,
                "stage": self._stage.value,
                "import_receipt_cid": self._import_receipt.receipt_cid,
                "production_snapshot_digest": _identity(production),
                "shadow_snapshot_digest": _identity(shadow),
                "drifts": [item.to_dict() for item in drifts],
                "unexplained_drift_count": unexplained,
                "reviewed_disposition_count": reviewed,
                "matched_surface_count": matched,
                "reason_codes": list(stable_reasons),
                "evidence_subset": evidence,
            }
            report_id = _identity(report_body)
            report = ShadowParityReport(
                report_id=report_id,
                decision=decision,
                stage=self._stage,
                import_reconciled=import_reconciled,
                import_receipt_cid=self._import_receipt.receipt_cid,
                production_snapshot_digest=_identity(production),
                shadow_snapshot_digest=_identity(shadow),
                drifts=tuple(drifts),
                unexplained_drift_count=unexplained,
                reviewed_disposition_count=reviewed,
                matched_surface_count=matched,
                dual_observation=window,
                shadow_controls_production=self.shadow_controls_production,
                history_preserved=True,
                reason_codes=stable_reasons,
                evidence_subset=MappingProxyType(evidence),
                created_at=created_at,
            )
            self._last_parity = report
            self._append_history(
                HistoryKind.PARITY,
                {
                    "report_id": report.report_id,
                    "decision": report.decision.value,
                    "parity_digest": report.parity_digest,
                    "unexplained_drift_count": unexplained,
                    "passed": report.passed,
                },
            )
            return report

    def _resolve_disposition(self, surface: str) -> DriftDisposition | None:
        if surface in self._dispositions:
            return self._dispositions[surface]
        # Allow record-scoped dispositions to cover a surface when unique.
        matches = [
            item
            for key, item in self._dispositions.items()
            if key.startswith(f"{surface}:")
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def _import_is_reconciled(self, receipt: ImportReceipt) -> bool:
        if receipt.outcome not in (OUTCOME_APPLIED, OUTCOME_REPLAYED):
            return False
        if receipt.strict and receipt.rejected_rows:
            return False
        # Every accepted row must carry source digest provenance.
        for row in receipt.accepted_rows:
            if not str(row.get("source_digest") or ""):
                return False
            if not str(row.get("source_id") or ""):
                return False
        # Source observations must have digests.
        for observation in receipt.source_observations:
            if not observation.source_digest.startswith("sha256:"):
                return False
        return True

    def _evidence_subset(
        self,
        *,
        production: Mapping[str, Any],
        shadow: Mapping[str, Any],
        drifts: Sequence[DriftRecord],
        receipt: ImportReceipt,
    ) -> dict[str, Any]:
        return {
            "counts_digests": {
                "production": _surface_payload(production, "counts_digests"),
                "shadow": _surface_payload(shadow, "counts_digests"),
            },
            "duplicate_conflict": {
                "production": _surface_payload(production, "duplicate_conflict"),
                "shadow": _surface_payload(shadow, "duplicate_conflict"),
            },
            "task_cid": {
                "production": _surface_payload(production, "task_cid"),
                "shadow": _surface_payload(shadow, "task_cid"),
            },
            "readiness": {
                "production": _surface_payload(production, "readiness"),
                "shadow": _surface_payload(shadow, "readiness"),
            },
            "lease_fence": {
                "production": _surface_payload(production, "lease_fence"),
                "shadow": _surface_payload(shadow, "lease_fence"),
            },
            "status": {
                "production": _surface_payload(production, "status"),
                "shadow": _surface_payload(shadow, "status"),
            },
            "event_cursor": {
                "production": _surface_payload(production, "event_cursor"),
                "shadow": _surface_payload(shadow, "event_cursor"),
            },
            "completion": {
                "production": _surface_payload(production, "completion"),
                "shadow": _surface_payload(shadow, "completion"),
            },
            "restart": {
                "production": _surface_payload(production, "restart"),
                "shadow": _surface_payload(shadow, "restart"),
            },
            "export": {
                "production": _surface_payload(production, "export"),
                "shadow": _surface_payload(shadow, "export"),
            },
            "import": {
                "receipt_cid": receipt.receipt_cid,
                "accepted_count": len(receipt.accepted_rows),
                "rejected_count": len(receipt.rejected_rows),
                "conflict_count": len(receipt.conflicts),
                "outcome": receipt.outcome,
            },
            "drift_summary": [
                {
                    "surface": item.surface,
                    "matched": item.matched,
                    "disposition": item.disposition.value,
                }
                for item in drifts
            ],
        }

    # -- rollback / re-run --------------------------------------------------

    def rollback(
        self,
        *,
        reason: str = "kill_switch",
        to_stage: ShadowRolloutStage | str = ShadowRolloutStage.OFF,
    ) -> ShadowRollbackReceipt:
        """Roll back authority/read route without deleting history."""

        with self._lock:
            target = (
                to_stage
                if isinstance(to_stage, ShadowRolloutStage)
                else ShadowRolloutStage(str(to_stage))
            )
            if target not in (
                ShadowRolloutStage.OFF,
                ShadowRolloutStage.BACKFILL,
                ShadowRolloutStage.OBSERVE,
            ):
                raise ShadowStageError(
                    f"rollback target {target.value} is not a safe prior stage"
                )
            prior = self._stage
            prior_parity = (
                self._last_parity.parity_digest if self._last_parity else ""
            )
            history_count_before = len(self._history)
            self._stage = target
            # Window closes on full off, but history is retained.
            if target is ShadowRolloutStage.OFF:
                # Intentionally keep _window metadata for audit; do not clear history.
                pass
            rolled_back_at = _utc_iso(self._clock())
            receipt_body = {
                "from_stage": prior.value,
                "to_stage": target.value,
                "reason": _text(reason, "reason"),
                "history_entry_count": history_count_before,
                "prior_parity_digest": prior_parity,
                "rolled_back_at": rolled_back_at,
            }
            receipt = ShadowRollbackReceipt(
                receipt_id=_identity(receipt_body),
                from_stage=prior.value,
                to_stage=target.value,
                reason=str(reason),
                history_preserved=True,
                history_entry_count=history_count_before,
                prior_parity_digest=prior_parity,
                rolled_back_at=rolled_back_at,
            )
            self._append_history(
                HistoryKind.ROLLBACK,
                receipt.to_dict(),
            )
            # History must still be present and longer after rollback.
            if len(self._history) <= history_count_before:
                raise DatabaseShadowRolloutError(
                    "rollback failed to preserve append-only history"
                )
            return receipt

    def rerun_parity(
        self,
        *,
        production_snapshot: Mapping[str, Any] | None = None,
        shadow_snapshot: Mapping[str, Any] | None = None,
    ) -> ShadowParityReport:
        """Re-evaluate parity; identical inputs yield the same decision digest."""

        prior = self._last_parity
        report = self.evaluate_parity(
            production_snapshot=production_snapshot,
            shadow_snapshot=shadow_snapshot,
        )
        if prior is not None:
            # Decision and parity digest must be stable for identical snapshots.
            same_inputs = (
                (production_snapshot is None and shadow_snapshot is None)
                or (
                    production_snapshot is not None
                    and shadow_snapshot is not None
                    and _identity(self._normalize_snapshot(production_snapshot))
                    == prior.production_snapshot_digest
                    and _identity(self._normalize_snapshot(shadow_snapshot))
                    == prior.shadow_snapshot_digest
                )
            )
            if same_inputs and production_snapshot is None and shadow_snapshot is None:
                if report.decision != prior.decision:
                    raise ShadowParityError(
                        "re-run produced a different parity decision"
                    )
                if report.parity_digest != prior.parity_digest:
                    raise ShadowParityError(
                        "re-run produced a different parity digest"
                    )
            elif (
                production_snapshot is not None
                and shadow_snapshot is not None
                and _identity(self._normalize_snapshot(production_snapshot))
                == prior.production_snapshot_digest
                and _identity(self._normalize_snapshot(shadow_snapshot))
                == prior.shadow_snapshot_digest
            ):
                if report.decision != prior.decision:
                    raise ShadowParityError(
                        "re-run produced a different parity decision"
                    )
                if report.parity_digest != prior.parity_digest:
                    raise ShadowParityError(
                        "re-run produced a different parity digest"
                    )
        return report

    # -- history ------------------------------------------------------------

    def _append_history(
        self, kind: HistoryKind, payload: Mapping[str, Any]
    ) -> HistoryEntry:
        if len(self._history) >= MAX_HISTORY_ENTRIES:
            raise DualObservationBoundError(
                f"history cap {MAX_HISTORY_ENTRIES} exceeded"
            )
        recorded_at = _utc_iso(self._clock())
        body = dict(payload)
        digest = _identity({"kind": kind.value, "payload": body, "at": recorded_at})
        entry = HistoryEntry(
            entry_id=digest,
            kind=kind,
            recorded_at=recorded_at,
            payload=MappingProxyType(body),
            digest=digest,
        )
        self._history.append(entry)
        return entry

    def history_digests(self) -> tuple[str, ...]:
        return tuple(entry.digest for entry in self._history)

    def status(self) -> dict[str, Any]:
        """Bounded operator status projection."""

        with self._lock:
            return {
                "schema": self.SCHEMA,
                "interface": self.INTERFACE,
                "stage": self._stage.value,
                "authority_mode": self._authority_mode.value,
                "dual_observation_enabled": self._dual_observation_enabled,
                "shadow_controls_production": self.shadow_controls_production,
                "import_receipt_cid": (
                    self._import_receipt.receipt_cid
                    if self._import_receipt is not None
                    else ""
                ),
                "observation_count": len(self._observations),
                "history_count": len(self._history),
                "production_effect_count": len(self._production_effects),
                "disposition_count": len(self._dispositions),
                "window": (
                    self._window.to_dict() if self._window is not None else None
                ),
                "last_parity_digest": (
                    self._last_parity.parity_digest
                    if self._last_parity is not None
                    else ""
                ),
                "last_parity_decision": (
                    self._last_parity.decision.value
                    if self._last_parity is not None
                    else ""
                ),
                "task_id": TASK_ID,
                "goal_id": GOAL_ID,
                "evidence": EVIDENCE,
            }


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def build_shadow_manifest(
    import_id: str,
    sources: Sequence[Mapping[str, Any] | ImportSourceSpec],
    **kwargs: Any,
) -> ImportManifest:
    """Build an apply-oriented import manifest for shadow backfill."""

    return build_import_manifest(import_id, sources, mode=ImportMode.APPLY, **kwargs)


def run_shadow_parity(
    *,
    work_dir: Path | str,
    sources: Sequence[Mapping[str, Any] | ImportSourceSpec],
    import_id: str = "dqp-037-shadow-backfill",
    dispositions: Sequence[DriftDisposition] = (),
    production_snapshot: Mapping[str, Any] | None = None,
    window: DualObservationWindow | None = None,
    clock: Clock | None = None,
    durable: bool = True,
) -> ShadowParityReport:
    """One-shot helper: backfill reviewed sources and evaluate shadow parity.

    When ``durable`` is true and DuckDB is available, the backfill commits to
    ``shadow_control.duckdb`` under ``work_dir``. Otherwise the import is kept
    in hermetic memory while still producing a content-addressed parity report.
    """

    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    use_durable = bool(durable) and duckdb_available()
    target_db = root / "shadow_control.duckdb" if use_durable else None
    rollout = DatabaseShadowRollout(
        work_dir=root,
        source_root=root,
        target_database=target_db,
        window=window,
        dispositions=dispositions,
        clock=clock,
    )
    manifest = build_shadow_manifest(
        import_id,
        sources,
        target_database=str(target_db) if target_db is not None else "",
        strict=True,
        default_conflict_policy=ConflictPolicy.REJECT,
    )
    rollout.backfill(manifest, production_snapshot=production_snapshot)
    rollout.transition(ShadowRolloutStage.SHADOW, reason="dual_observation")
    return rollout.evaluate_parity()


__all__ = [
    "DATABASE_SHADOW_ROLLOUT_INTERFACE",
    "SHADOW_PARITY_REPORT_INTERFACE",
    "DATABASE_SHADOW_ROLLOUT_SCHEMA",
    "SHADOW_PARITY_REPORT_SCHEMA",
    "DATABASE_SHADOW_ROLLOUT_VERSION",
    "AUTHORITY_SURFACES",
    "DEFAULT_MAX_DURATION_SECONDS",
    "DEFAULT_MAX_RETENTION_RECORDS",
    "DEFAULT_MAX_RETENTION_BYTES",
    "PRODUCTION_EFFECT_CHANNEL",
    "SHADOW_EFFECT_CHANNEL",
    "EXPORT_NON_AUTHORITY_MARKER",
    "TASK_ID",
    "GOAL_ID",
    "EVIDENCE",
    "DatabaseShadowRolloutError",
    "ShadowParityError",
    "ShadowAuthorityError",
    "DualObservationBoundError",
    "ShadowDispositionError",
    "ShadowStageError",
    "ShadowRolloutStage",
    "ParityDecision",
    "DriftDispositionKind",
    "HistoryKind",
    "DualObservationWindow",
    "DriftDisposition",
    "DriftRecord",
    "ShadowObservation",
    "ShadowParityReport",
    "ShadowRollbackReceipt",
    "HistoryEntry",
    "ProductionEffect",
    "DatabaseShadowRollout",
    "project_authority_snapshot",
    "build_shadow_manifest",
    "run_shadow_parity",
    "ImportMediaType",
    "ImportDomain",
    "ImportSourceSpec",
    "ConflictPolicy",
]
