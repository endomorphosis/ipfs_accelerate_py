"""Bounded, semantic diagnostics for deterministic supervisor recovery.

The diagnostic surface intentionally accepts data, not commands.  It reduces
the supervisor's many mutable projections to a stable failure fingerprint and
an existing :class:`~prompt_workflow.SupervisorIncident`.  Observation times
and continuously increasing ages are excluded from identity, while threshold
crossings, identities, health states, and prior recovery outcomes remain
identity-bearing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..prompt.prompt_workflow import (
    IncidentKind,
    RecordStatus,
    SupervisorIncident,
    prompt_workflow_cid,
)


RECOVERY_DIAGNOSTIC_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/recovery-diagnostic@1"
)
RECOVERY_DIAGNOSTIC_REQUIREMENT_ID: Final = (
    "asi-155:unified-incident-diagnosis-and-programmatic-recovery"
)


class RecoveryDiagnosticError(ValueError):
    """Diagnostic evidence is malformed, ambiguous, or over its bound."""


class RecoveryEvidenceKind(str, Enum):
    STATUS = "status"
    HEALTH = "health"
    PROCESS = "process"
    HEARTBEAT = "heartbeat"
    EVENT = "event"
    LEASE = "lease"
    LOCK = "lock"
    TASK = "task"
    ATTEMPT = "attempt"
    TASK_SOURCE = "task_source"
    WORKTREE = "worktree"
    MERGE = "merge"
    PROVIDER = "provider"
    VALIDATION = "validation"
    DISK = "disk"
    PRIOR_ACTION = "prior_action"


_VOLATILE_KEYS: Final = frozenset(
    {
        "age",
        "age_ms",
        "checked_at",
        "checked_at_ms",
        "created_at",
        "created_at_ms",
        "elapsed_ms",
        "finished_at",
        "finished_at_ms",
        "observed_at",
        "observed_at_ms",
        "started_at",
        "started_at_ms",
        "timestamp",
        "timestamp_ms",
        "updated_at",
        "updated_at_ms",
    }
)
_SECRET_MARKERS: Final = (
    "access_token",
    "api_key",
    "credential",
    "password",
    "private_key",
    "prompt_body",
    "prompt_text",
    "raw_log",
    "secret",
    "source_body",
    "source_text",
)
_IDENTITY_KEYS: Final = (
    "artifact_id",
    "attempt_id",
    "lane_id",
    "lease_id",
    "lock_id",
    "merge_id",
    "process_tree_id",
    "provider_id",
    "run_id",
    "task_id",
    "validation_id",
    "worktree_id",
)


@dataclass(frozen=True)
class RecoveryDiagnosticLimits:
    """Hard input bounds for one diagnostic pass."""

    max_evidence_items: int = 64
    max_mapping_items: int = 256
    max_depth: int = 8
    max_serialized_bytes: int = 128 * 1024
    max_targets: int = 256
    max_prior_actions: int = 32

    def __post_init__(self) -> None:
        for name in (
            "max_evidence_items",
            "max_mapping_items",
            "max_depth",
            "max_serialized_bytes",
            "max_targets",
            "max_prior_actions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RecoveryDiagnosticError(
                    f"{name} must be a positive integer"
                )


def _bounded_json(
    value: Any,
    *,
    limits: RecoveryDiagnosticLimits,
    name: str,
) -> Any:
    seen = 0

    def visit(item: Any, depth: int, key_name: str = "") -> Any:
        nonlocal seen
        seen += 1
        if seen > limits.max_mapping_items:
            raise RecoveryDiagnosticError(f"{name} exceeds item-count bound")
        if depth > limits.max_depth:
            raise RecoveryDiagnosticError(f"{name} exceeds depth bound")
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, int) and not isinstance(item, bool):
            return item
        if isinstance(item, float):
            if not (item == item and abs(item) != float("inf")):
                raise RecoveryDiagnosticError(
                    f"{name} contains a non-finite number"
                )
            return item
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, str):
            if "\x00" in item:
                raise RecoveryDiagnosticError(f"{name} contains NUL")
            return item
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for raw_key in sorted(item, key=str):
                key = str(raw_key).strip()
                if not key:
                    raise RecoveryDiagnosticError(f"{name} contains an empty key")
                normalized = key.lower().replace("-", "_")
                if any(marker in normalized for marker in _SECRET_MARKERS):
                    # Diagnostic inputs are commonly assembled from broad
                    # status records.  Omit secret-bearing fields instead of
                    # allowing them into an incident or failure fingerprint.
                    continue
                result[key] = visit(item[raw_key], depth + 1, key)
            return result
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray, memoryview)
        ):
            return [visit(member, depth + 1, key_name) for member in item]
        raise RecoveryDiagnosticError(
            f"{name} contains unsupported type {type(item).__name__}"
        )

    result = visit(value, 0)
    try:
        payload = json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise RecoveryDiagnosticError(f"{name} is not canonical JSON") from exc
    if len(payload) > limits.max_serialized_bytes:
        raise RecoveryDiagnosticError(f"{name} exceeds serialized byte bound")
    return result


def _semantic(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _semantic(member)
            for key, member in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).lower().replace("-", "_") not in _VOLATILE_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_semantic(member) for member in value]
    return value


def _deep_freeze(value: Any) -> Any:
    """Make CID-bearing diagnostic values recursively immutable."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _deep_freeze(member)
                for key, member in value.items()
            }
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return tuple(_deep_freeze(member) for member in value)
    return value


def _truthy(record: Mapping[str, Any], *names: str) -> bool:
    return any(record.get(name) is True for name in names)


def _state(record: Mapping[str, Any]) -> str:
    return str(
        record.get("state")
        or record.get("status")
        or record.get("outcome")
        or ""
    ).strip().lower()


def _failed(record: Mapping[str, Any]) -> bool:
    state = _state(record)
    return (
        _truthy(record, "failed", "fault", "unhealthy", "unavailable", "corrupt")
        or state
        in {
            "blocked",
            "corrupt",
            "dead",
            "error",
            "expired",
            "failed",
            "faulted",
            "interrupted",
            "lost",
            "missing",
            "offline",
            "stalled",
            "timed_out",
            "unavailable",
            "unhealthy",
        }
    )


def _explicitly_false(record: Mapping[str, Any], *names: str) -> bool:
    return any(name in record and record.get(name) is False for name in names)


def _healthy(record: Mapping[str, Any]) -> bool:
    return (
        _truthy(record, "alive", "healthy", "ok", "ready", "running")
        or _state(record) in {"alive", "healthy", "ok", "ready", "running"}
    )


def _stale(record: Mapping[str, Any]) -> bool:
    return _truthy(record, "stale", "expired") or _state(record) in {
        "expired",
        "stale",
    }


def _more_than_one(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 1
    )


@dataclass(frozen=True)
class RecoveryEvidence:
    """One typed and content-addressed diagnostic observation."""

    kind: RecoveryEvidenceKind
    value: Mapping[str, Any]
    target_id: str = ""
    observed_at_ms: int = 0
    evidence_cid: str = ""
    _limits: RecoveryDiagnosticLimits = field(
        default_factory=RecoveryDiagnosticLimits,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RecoveryEvidenceKind):
            object.__setattr__(
                self, "kind", RecoveryEvidenceKind(str(self.kind))
            )
        if not isinstance(self.value, Mapping):
            raise RecoveryDiagnosticError("evidence value must be a mapping")
        frozen = _bounded_json(
            self.value, limits=self._limits, name=f"{self.kind.value} evidence"
        )
        object.__setattr__(
            self, "value", _deep_freeze(frozen)
        )
        target = str(self.target_id or "").strip()
        if "\x00" in target:
            raise RecoveryDiagnosticError("target_id contains NUL")
        object.__setattr__(self, "target_id", target)
        if (
            isinstance(self.observed_at_ms, bool)
            or not isinstance(self.observed_at_ms, int)
            or self.observed_at_ms < 0
        ):
            raise RecoveryDiagnosticError(
                "observed_at_ms must be a nonnegative integer"
            )
        expected = prompt_workflow_cid(self.semantic_record())
        if self.evidence_cid and self.evidence_cid != expected:
            raise RecoveryDiagnosticError("evidence CID does not match content")
        object.__setattr__(self, "evidence_cid", expected)

    def semantic_record(self) -> dict[str, Any]:
        result = {
            "schema": RECOVERY_DIAGNOSTIC_SCHEMA,
            "kind": self.kind.value,
            "value": _semantic(self.value),
        }
        if self.target_id:
            result["target_id"] = self.target_id
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.semantic_record(),
            "observed_at_ms": self.observed_at_ms,
            "evidence_cid": self.evidence_cid,
        }


@dataclass(frozen=True)
class RecoveryDiagnosis:
    """Semantic incident plus classification facts used by recovery policy."""

    incident: SupervisorIncident
    evidence: tuple[RecoveryEvidence, ...]
    reason_codes: tuple[str, ...]
    live_fault: bool
    stale_projection: bool

    @property
    def incident_cid(self) -> str:
        return self.incident.incident_cid

    @property
    def kind(self) -> IncidentKind:
        return self.incident.kind

    @property
    def target_ids(self) -> tuple[str, ...]:
        return self.incident.target_ids

    @property
    def health(self) -> Mapping[str, Any]:
        return self.incident.health

    def evidence_for(
        self, kind: RecoveryEvidenceKind | str
    ) -> tuple[RecoveryEvidence, ...]:
        selected = (
            kind
            if isinstance(kind, RecoveryEvidenceKind)
            else RecoveryEvidenceKind(str(kind))
        )
        return tuple(item for item in self.evidence if item.kind is selected)


def _classify(
    records: Mapping[RecoveryEvidenceKind, tuple[RecoveryEvidence, ...]]
) -> tuple[IncidentKind, tuple[str, ...], bool, bool]:
    def values(kind: RecoveryEvidenceKind) -> tuple[Mapping[str, Any], ...]:
        return tuple(item.value for item in records.get(kind, ()))

    process = values(RecoveryEvidenceKind.PROCESS)
    status = values(RecoveryEvidenceKind.STATUS)
    health = values(RecoveryEvidenceKind.HEALTH)
    heartbeat = values(RecoveryEvidenceKind.HEARTBEAT)
    event = values(RecoveryEvidenceKind.EVENT)
    lease = values(RecoveryEvidenceKind.LEASE)
    lock = values(RecoveryEvidenceKind.LOCK)
    task = values(RecoveryEvidenceKind.TASK)
    attempt = values(RecoveryEvidenceKind.ATTEMPT)
    source = values(RecoveryEvidenceKind.TASK_SOURCE)
    worktree = values(RecoveryEvidenceKind.WORKTREE)
    merge = values(RecoveryEvidenceKind.MERGE)
    provider = values(RecoveryEvidenceKind.PROVIDER)
    validation = values(RecoveryEvidenceKind.VALIDATION)
    disk = values(RecoveryEvidenceKind.DISK)

    candidates: list[tuple[IncidentKind, str]] = []
    if any(
        _truthy(item, "split_brain", "multiple_live_owners")
        or _more_than_one(item.get("live_owner_count"))
        for item in process
    ):
        candidates.append((IncidentKind.SPLIT_BRAIN, "multiple_live_owners"))
    if any(
        _truthy(item, "corrupt", "digest_mismatch", "integrity_failed")
        or _failed(item)
        for item in source
    ):
        candidates.append(
            (IncidentKind.CORRUPT_TASK_SOURCE, "task_source_integrity_failed")
        )
    if any(
        _truthy(
            item,
            "full",
            "exhausted",
            "insufficient_space",
            "read_only",
        )
        or _state(item)
        in {"full", "exhausted", "insufficient_space", "read_only"}
        for item in disk
    ):
        candidates.append(
            (IncidentKind.RESOURCE_EXHAUSTION, "disk_resource_exhausted")
        )
    if any(
        _truthy(item, "dirty", "conflicted")
        or _state(item) in {"dirty", "conflicted"}
        for item in worktree
    ):
        candidates.append((IncidentKind.DIRTY_WORKTREE, "worktree_dirty"))
    if any(_failed(item) for item in merge):
        candidates.append((IncidentKind.MERGE_FAILURE, "merge_failed"))
    if any(_failed(item) for item in validation):
        candidates.append(
            (IncidentKind.VALIDATION_FAILURE, "validation_failed")
        )
    if any(_stale(item) for item in lease):
        candidates.append((IncidentKind.STALE_LEASE, "lease_expired"))
    if any(
        _truthy(item, "orphaned")
        or (_stale(item) and not _truthy(item, "owner_live"))
        for item in lock
    ):
        candidates.append((IncidentKind.ORPHANED_LOCK, "lock_orphaned"))
    if any(
        _truthy(item, "consumed", "abandoned", "expired")
        or _state(item) in {"consumed", "abandoned", "expired"}
        for item in attempt
    ):
        candidates.append((IncidentKind.CONSUMED_ATTEMPT, "attempt_consumed"))
    if any(_stale(item) for item in heartbeat) or any(
        _truthy(item, "cursor_stale") or _stale(item) for item in event
    ):
        candidates.append((IncidentKind.STALE_HEARTBEAT, "progress_signal_stale"))
    if any(_failed(item) for item in provider):
        candidates.append(
            (IncidentKind.PROVIDER_UNAVAILABLE, "provider_unavailable")
        )
    if any(
        _truthy(
            item,
            "lifecycle_stale",
            "lifecycle_state_stale",
            "stale_lifecycle",
        )
        or _state(item) == "stale_lifecycle"
        for item in status
    ):
        candidates.append(
            (IncidentKind.STALE_LIFECYCLE, "lifecycle_projection_stale")
        )
    if any(
        _failed(item)
        or _explicitly_false(item, "alive", "healthy", "ok", "ready")
        for item in process + health + task
    ):
        candidates.append((IncidentKind.LANE_FAILURE, "live_lane_fault"))

    explicitly_stale_projection = any(
        _truthy(item, "projection_stale", "stale_projection")
        or _state(item) == "stale_projection"
        for item in status + health
    )
    live_state_healthy = (
        any(_healthy(item) for item in process)
        and any(_healthy(item) for item in health)
        and (
            not heartbeat
            or any(
                _healthy(item)
                or _state(item) in {"current", "fresh"}
                for item in heartbeat
            )
        )
    )
    projection_stale = explicitly_stale_projection or (
        any(_failed(item) for item in status) and live_state_healthy
    )
    if projection_stale and live_state_healthy:
        candidates = [
            candidate
            for candidate in candidates
            if candidate[0] is not IncidentKind.LANE_FAILURE
        ]
    live_signal_fault = bool(candidates)
    if projection_stale and not live_signal_fault:
        return (
            IncidentKind.STALE_PROJECTION,
            ("projection_stale_live_state_healthy",),
            False,
            True,
        )
    if candidates:
        selected = candidates[0][0]
        reasons = tuple(
            sorted({reason for kind, reason in candidates if kind is selected})
        )
        return selected, reasons, True, False
    return IncidentKind.UNKNOWN, ("no_supported_fault_signal",), False, False


def _coerce_items(
    kind: RecoveryEvidenceKind,
    raw: Any,
    *,
    observed_at_ms: int,
    limits: RecoveryDiagnosticLimits,
) -> tuple[RecoveryEvidence, ...]:
    if raw is None:
        return ()
    if isinstance(raw, RecoveryEvidence):
        if raw.kind is not kind:
            raise RecoveryDiagnosticError("evidence kind does not match its slot")
        return (
            RecoveryEvidence(
                kind=raw.kind,
                value=raw.value,
                target_id=raw.target_id,
                observed_at_ms=raw.observed_at_ms,
                evidence_cid=raw.evidence_cid,
                _limits=limits,
            ),
        )
    if isinstance(raw, Mapping):
        raw_items: Sequence[Any] = (raw,)
    elif isinstance(raw, Sequence) and not isinstance(
        raw, (str, bytes, bytearray, memoryview)
    ):
        raw_items = raw
    else:
        raise RecoveryDiagnosticError(f"{kind.value} evidence must be a mapping")
    result: list[RecoveryEvidence] = []
    for raw_item in raw_items:
        if isinstance(raw_item, RecoveryEvidence):
            if raw_item.kind is not kind:
                raise RecoveryDiagnosticError(
                    "evidence kind does not match its slot"
                )
            item = RecoveryEvidence(
                kind=raw_item.kind,
                value=raw_item.value,
                target_id=raw_item.target_id,
                observed_at_ms=raw_item.observed_at_ms,
                evidence_cid=raw_item.evidence_cid,
                _limits=limits,
            )
        else:
            if not isinstance(raw_item, Mapping):
                raise RecoveryDiagnosticError(
                    f"{kind.value} evidence item must be a mapping"
                )
            target = ""
            for key in _IDENTITY_KEYS:
                if raw_item.get(key):
                    target = str(raw_item[key])
                    break
            item = RecoveryEvidence(
                kind=kind,
                value=raw_item,
                target_id=target,
                observed_at_ms=observed_at_ms,
                _limits=limits,
            )
        result.append(item)
    return tuple(result)


def diagnose_supervisor_incident(
    *,
    repository_root: str,
    state_root: str,
    repository_root_cid: str,
    policy_root: str,
    run_cid: str,
    status: Any = None,
    health: Any = None,
    process: Any = None,
    heartbeat: Any = None,
    event: Any = None,
    lease: Any = None,
    lock: Any = None,
    task: Any = None,
    attempt: Any = None,
    task_source: Any = None,
    worktree: Any = None,
    merge: Any = None,
    provider: Any = None,
    validation: Any = None,
    disk: Any = None,
    prior_actions: Sequence[Any] = (),
    observed_at_ms: int = 0,
    limits: RecoveryDiagnosticLimits | None = None,
) -> RecoveryDiagnosis:
    """Derive one stable incident from bounded cross-subsystem evidence."""

    selected_limits = limits or RecoveryDiagnosticLimits()
    if (
        isinstance(observed_at_ms, bool)
        or not isinstance(observed_at_ms, int)
        or observed_at_ms < 0
    ):
        raise RecoveryDiagnosticError(
            "observed_at_ms must be a nonnegative integer"
        )
    if len(prior_actions) > selected_limits.max_prior_actions:
        raise RecoveryDiagnosticError("prior actions exceed bound")
    inputs = {
        RecoveryEvidenceKind.STATUS: status,
        RecoveryEvidenceKind.HEALTH: health,
        RecoveryEvidenceKind.PROCESS: process,
        RecoveryEvidenceKind.HEARTBEAT: heartbeat,
        RecoveryEvidenceKind.EVENT: event,
        RecoveryEvidenceKind.LEASE: lease,
        RecoveryEvidenceKind.LOCK: lock,
        RecoveryEvidenceKind.TASK: task,
        RecoveryEvidenceKind.ATTEMPT: attempt,
        RecoveryEvidenceKind.TASK_SOURCE: task_source,
        RecoveryEvidenceKind.WORKTREE: worktree,
        RecoveryEvidenceKind.MERGE: merge,
        RecoveryEvidenceKind.PROVIDER: provider,
        RecoveryEvidenceKind.VALIDATION: validation,
        RecoveryEvidenceKind.DISK: disk,
        RecoveryEvidenceKind.PRIOR_ACTION: prior_actions,
    }
    grouped = {
        kind: _coerce_items(
            kind,
            raw,
            observed_at_ms=observed_at_ms,
            limits=selected_limits,
        )
        for kind, raw in inputs.items()
    }
    evidence = tuple(
        sorted(
            (item for items in grouped.values() for item in items),
            key=lambda item: (
                item.kind.value,
                item.target_id,
                item.evidence_cid,
            ),
        )
    )
    if not evidence:
        raise RecoveryDiagnosticError("at least one evidence item is required")
    if len(evidence) > selected_limits.max_evidence_items:
        raise RecoveryDiagnosticError("diagnostic evidence exceeds bound")
    aggregate_payload = json.dumps(
        [item.semantic_record() for item in evidence],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(aggregate_payload) > selected_limits.max_serialized_bytes:
        raise RecoveryDiagnosticError(
            "aggregate diagnostic evidence exceeds serialized byte bound"
        )

    kind, reasons, live_fault, stale_projection = _classify(grouped)
    target_kinds = {
        IncidentKind.STALE_PROJECTION: {
            RecoveryEvidenceKind.STATUS,
            RecoveryEvidenceKind.HEALTH,
        },
        IncidentKind.STALE_LIFECYCLE: {RecoveryEvidenceKind.STATUS},
        IncidentKind.STALE_HEARTBEAT: {
            RecoveryEvidenceKind.HEARTBEAT,
            RecoveryEvidenceKind.EVENT,
            RecoveryEvidenceKind.PROCESS,
        },
        IncidentKind.STALE_LEASE: {RecoveryEvidenceKind.LEASE},
        IncidentKind.ORPHANED_LOCK: {RecoveryEvidenceKind.LOCK},
        IncidentKind.CONSUMED_ATTEMPT: {
            RecoveryEvidenceKind.ATTEMPT,
            RecoveryEvidenceKind.TASK,
        },
        IncidentKind.LANE_FAILURE: {
            RecoveryEvidenceKind.PROCESS,
            RecoveryEvidenceKind.HEALTH,
            RecoveryEvidenceKind.TASK,
        },
        IncidentKind.DIRTY_WORKTREE: {RecoveryEvidenceKind.WORKTREE},
        IncidentKind.VALIDATION_FAILURE: {
            RecoveryEvidenceKind.VALIDATION
        },
        IncidentKind.MERGE_FAILURE: {RecoveryEvidenceKind.MERGE},
        IncidentKind.CORRUPT_TASK_SOURCE: {
            RecoveryEvidenceKind.TASK_SOURCE
        },
        IncidentKind.RESOURCE_EXHAUSTION: {RecoveryEvidenceKind.DISK},
        IncidentKind.PROVIDER_UNAVAILABLE: {
            RecoveryEvidenceKind.PROVIDER
        },
        IncidentKind.SPLIT_BRAIN: {RecoveryEvidenceKind.PROCESS},
        IncidentKind.UNKNOWN: {
            RecoveryEvidenceKind.STATUS,
            RecoveryEvidenceKind.TASK,
        },
    }[kind]
    target_ids = tuple(
        sorted(
            {
                item.target_id
                for item in evidence
                if item.target_id and item.kind in target_kinds
            }
        )
    )
    if not target_ids:
        target_ids = (str(run_cid),)
    if len(target_ids) > selected_limits.max_targets:
        raise RecoveryDiagnosticError("diagnostic targets exceed bound")
    prior_cids = tuple(
        item.evidence_cid
        for item in evidence
        if item.kind is RecoveryEvidenceKind.PRIOR_ACTION
    )
    semantic_health = {
        "classification": kind.value,
        "live_fault": live_fault,
        "reason_codes": list(reasons),
        "stale_projection": stale_projection,
    }
    fingerprint_payload = json.dumps(
        {
            "schema": RECOVERY_DIAGNOSTIC_SCHEMA,
            "kind": kind.value,
            "targets": list(target_ids),
            "evidence": [item.evidence_cid for item in evidence],
            "prior_actions": list(prior_cids),
            "reason_codes": list(reasons),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    fingerprint = "sha256:" + hashlib.sha256(fingerprint_payload).hexdigest()

    incident = SupervisorIncident(
        repository_root=repository_root,
        state_root=state_root,
        repository_root_cid=repository_root_cid,
        policy_root=policy_root,
        run_cid=run_cid,
        kind=kind,
        failure_fingerprint=fingerprint,
        target_ids=target_ids,
        evidence_cids=tuple(item.evidence_cid for item in evidence),
        health=semantic_health,
        prior_recovery_cids=prior_cids,
        cooldown_key=f"{kind.value}:{target_ids[0]}",
        status=(
            RecordStatus.FAILED
            if live_fault
            else RecordStatus.BLOCKED
        ),
        observed_at_ms=observed_at_ms,
        updated_at_ms=observed_at_ms,
    )
    return RecoveryDiagnosis(
        incident=incident,
        evidence=evidence,
        reason_codes=reasons,
        live_fault=live_fault,
        stale_projection=stale_projection,
    )


class RecoveryDiagnostics:
    """Reusable bounded diagnostic facade for supervisor integrations."""

    def __init__(
        self, limits: RecoveryDiagnosticLimits | None = None
    ) -> None:
        self.limits = limits or RecoveryDiagnosticLimits()

    def diagnose(self, **evidence: Any) -> RecoveryDiagnosis:
        if "limits" in evidence:
            raise RecoveryDiagnosticError(
                "diagnostic facade limits are fixed at construction"
            )
        return diagnose_supervisor_incident(
            **evidence, limits=self.limits
        )

    derive = diagnose


derive_incident = diagnose_supervisor_incident
diagnose_incident = diagnose_supervisor_incident
DiagnosticEvidence = RecoveryEvidence
IncidentDiagnosis = RecoveryDiagnosis


__all__ = [
    "DiagnosticEvidence",
    "IncidentDiagnosis",
    "RECOVERY_DIAGNOSTIC_REQUIREMENT_ID",
    "RECOVERY_DIAGNOSTIC_SCHEMA",
    "RecoveryDiagnosis",
    "RecoveryDiagnostics",
    "RecoveryDiagnosticError",
    "RecoveryDiagnosticLimits",
    "RecoveryEvidence",
    "RecoveryEvidenceKind",
    "derive_incident",
    "diagnose_incident",
    "diagnose_supervisor_incident",
]
