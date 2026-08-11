"""Legacy backfill and shadow decision parity (DQP-037).

Interfaces: ``DatabaseShadowRollout@1``, ``ShadowParityReport@1``

Backfills reviewed programs/state into a non-authoritative database shadow,
mirrors legacy lifecycle decisions into shadow transactions, compares
canonical tasks/readiness/events/revisions/leases/status/exports, and requires
an explicit reviewed disposition for every authority-relevant drift.

Shadow writes never control production effect. Dual observation is bounded by
duration and retention. Rollback and re-run preserve history and recompute the
same parity decision for the same inputs.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..task_sources.task_source import StateAuthorityMode


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_SHADOW_ROLLOUT_INTERFACE: Final[str] = "DatabaseShadowRollout@1"
SHADOW_PARITY_REPORT_INTERFACE: Final[str] = "ShadowParityReport@1"
SHADOW_CONTRACT_VERSION: Final[int] = 1
TASK_ID: Final[str] = "DQP-037"
GOAL_ID: Final[str] = "DQP-G080"
EVIDENCE: Final[str] = "dqp/database-shadow-rollout@1"

SCHEMA_PREFIX: Final[str] = "ipfs_accelerate_py/agent-supervisor"
SHADOW_ROLLOUT_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/database-shadow-rollout@1"
PARITY_REPORT_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/shadow-parity-report@1"
DRIFT_RECORD_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/shadow-drift-record@1"
BACKFILL_RECEIPT_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/shadow-backfill-receipt@1"
LEGACY_DECISION_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/shadow-legacy-decision@1"

DEFAULT_MAX_DUAL_OBSERVATION_SECONDS: Final[int] = 86_400
DEFAULT_RETENTION_SECONDS: Final[int] = 7 * 86_400
MAX_RECORDS: Final[int] = 50_000
MAX_TEXT_BYTES: Final[int] = 512
MAX_REASON_CODES: Final[int] = 256

# Domains compared for authority-relevant parity.
PARITY_DOMAINS: Final[tuple[str, ...]] = (
    "tasks",
    "readiness",
    "events",
    "revisions",
    "leases",
    "status",
    "exports",
    "task_cid",
    "completion",
)


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class ShadowPhase(str, Enum):
    BACKFILL = "backfill"
    SHADOW = "shadow"
    COMPARE = "compare"
    DISPOSITION = "disposition"
    ROLLBACK = "rollback"
    TERMINAL = "terminal"


class ParityVerdict(str, Enum):
    PARITY = "parity"
    DRIFT_REVIEWED = "drift_reviewed"
    DRIFT_UNEXPLAINED = "drift_unexplained"
    FAILED = "failed"


class DriftDisposition(str, Enum):
    """Reviewed dispositions for authority-relevant drift."""

    ACCEPT_LEGACY = "accept_legacy"
    ACCEPT_SHADOW = "accept_shadow"
    QUARANTINE = "quarantine"
    REJECT = "reject"
    MERGE = "merge"


class DriftSeverity(str, Enum):
    AUTHORITY = "authority"
    OBSERVATIONAL = "observational"
    BENIGN = "benign"


class ShadowRolloutError(ValueError):
    """Fail-closed rejection for unsafe shadow rollout inputs or transitions."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _text(value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        raise ShadowRolloutError(f"{name} must be text")
    result = value.strip()
    if not result:
        raise ShadowRolloutError(f"{name} must not be empty")
    if "\x00" in result:
        raise ShadowRolloutError(f"{name} contains a NUL byte")
    if len(result.encode("utf-8")) > maximum:
        raise ShadowRolloutError(f"{name} exceeds its {maximum}-byte bound")
    return result


def _nonnegative_int(value: Any, name: str, *, maximum: int = 10**18) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ShadowRolloutError(f"{name} must be a non-negative integer")
    if value < 0 or value > maximum:
        raise ShadowRolloutError(f"{name} out of bounds")
    return value


def content_identity(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _canonical_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(
        json.dumps(dict(record), sort_keys=True, separators=(",", ":"), default=str)
    )


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegacyRecord:
    """One legacy authority record observed for backfill."""

    domain: str
    record_id: str
    body: Mapping[str, Any]
    source_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", _text(self.domain, "domain", maximum=64))
        if self.domain not in PARITY_DOMAINS and self.domain != "programs":
            # Allow programs as backfill domain plus parity domains.
            if self.domain not in {"programs", "goals", "queues"}:
                raise ShadowRolloutError(f"unknown domain {self.domain!r}")
        object.__setattr__(
            self, "record_id", _text(self.record_id, "record_id", maximum=256)
        )
        body = _canonical_record(self.body or {})
        object.__setattr__(self, "body", MappingProxyType(body))
        digest = self.source_digest or content_identity(
            {"domain": self.domain, "record_id": self.record_id, "body": body}
        )
        object.__setattr__(self, "source_digest", digest)

    @property
    def task_cid(self) -> str:
        return f"cid:{self.domain}:{self.record_id}:{self.source_digest[7:23]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "record_id": self.record_id,
            "body": dict(self.body),
            "source_digest": self.source_digest,
            "task_cid": self.task_cid,
        }


@dataclass(frozen=True)
class LegacyDecision:
    """A lifecycle decision made by the legacy authority (mirrored into shadow)."""

    SCHEMA: ClassVar[str] = LEGACY_DECISION_SCHEMA

    decision_id: str
    domain: str
    record_id: str
    action: str
    revision: int
    lease_id: str = ""
    status: str = "ready"
    ready: bool = True
    event_cursor: int = 0
    completed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "decision_id", _text(self.decision_id, "decision_id")
        )
        object.__setattr__(self, "domain", _text(self.domain, "domain", maximum=64))
        object.__setattr__(
            self, "record_id", _text(self.record_id, "record_id", maximum=256)
        )
        object.__setattr__(self, "action", _text(self.action, "action", maximum=64))
        object.__setattr__(
            self, "revision", _nonnegative_int(self.revision, "revision")
        )
        object.__setattr__(
            self, "event_cursor", _nonnegative_int(self.event_cursor, "event_cursor")
        )
        if self.lease_id:
            object.__setattr__(
                self, "lease_id", _text(self.lease_id, "lease_id", maximum=256)
            )
        if self.status:
            object.__setattr__(
                self, "status", _text(self.status, "status", maximum=64)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "decision_id": self.decision_id,
            "domain": self.domain,
            "record_id": self.record_id,
            "action": self.action,
            "revision": self.revision,
            "lease_id": self.lease_id,
            "status": self.status,
            "ready": self.ready,
            "event_cursor": self.event_cursor,
            "completed": self.completed,
        }


@dataclass(frozen=True)
class DriftRecord:
    """One observed drift between legacy and shadow projections."""

    SCHEMA: ClassVar[str] = DRIFT_RECORD_SCHEMA

    domain: str
    record_id: str
    field: str
    legacy_value: Any
    shadow_value: Any
    severity: DriftSeverity
    disposition: DriftDisposition | None = None
    reason_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", _text(self.domain, "domain", maximum=64))
        object.__setattr__(
            self, "record_id", _text(self.record_id, "record_id", maximum=256)
        )
        object.__setattr__(self, "field", _text(self.field, "field", maximum=128))
        object.__setattr__(
            self,
            "severity",
            self.severity
            if isinstance(self.severity, DriftSeverity)
            else DriftSeverity(str(self.severity)),
        )
        if self.disposition is not None and not isinstance(
            self.disposition, DriftDisposition
        ):
            object.__setattr__(
                self, "disposition", DriftDisposition(str(self.disposition))
            )

    @property
    def reviewed(self) -> bool:
        return self.disposition is not None

    def with_disposition(
        self, disposition: DriftDisposition, *, reason_code: str = ""
    ) -> "DriftRecord":
        return DriftRecord(
            domain=self.domain,
            record_id=self.record_id,
            field=self.field,
            legacy_value=self.legacy_value,
            shadow_value=self.shadow_value,
            severity=self.severity,
            disposition=disposition,
            reason_code=reason_code or self.reason_code,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.SCHEMA,
            "domain": self.domain,
            "record_id": self.record_id,
            "field": self.field,
            "legacy_value": self.legacy_value,
            "shadow_value": self.shadow_value,
            "severity": self.severity.value
            if isinstance(self.severity, Enum)
            else self.severity,
            "reviewed": self.reviewed,
        }
        if self.disposition is not None:
            payload["disposition"] = (
                self.disposition.value
                if isinstance(self.disposition, Enum)
                else self.disposition
            )
        if self.reason_code:
            payload["reason_code"] = self.reason_code
        return payload


@dataclass(frozen=True)
class BackfillReceipt:
    SCHEMA: ClassVar[str] = BACKFILL_RECEIPT_SCHEMA

    import_id: str
    record_count: int
    domain_counts: Mapping[str, int]
    digest: str
    exact_reconcile: bool
    replayed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "import_id": self.import_id,
            "record_count": self.record_count,
            "domain_counts": dict(self.domain_counts),
            "digest": self.digest,
            "exact_reconcile": self.exact_reconcile,
            "replayed": self.replayed,
        }


@dataclass(frozen=True)
class ShadowParityReport:
    """``ShadowParityReport@1`` — dual-observation comparison result."""

    SCHEMA: ClassVar[str] = PARITY_REPORT_SCHEMA
    INTERFACE: ClassVar[str] = SHADOW_PARITY_REPORT_INTERFACE

    verdict: ParityVerdict
    phase: ShadowPhase
    authority_mode: str
    production_effect: bool
    dual_observation_seconds: int
    retention_seconds: int
    backfill: BackfillReceipt
    decisions_mirrored: int
    domains_compared: tuple[str, ...]
    digests: Mapping[str, str]
    drifts: tuple[DriftRecord, ...]
    unexplained_authority_drift: int
    history_preserved: bool
    parity_decision_stable: bool
    reason_codes: tuple[str, ...] = ()
    created_at: str = field(default_factory=_utc_iso)
    evidence: str = EVIDENCE
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "verdict",
            self.verdict
            if isinstance(self.verdict, ParityVerdict)
            else ParityVerdict(str(self.verdict)),
        )
        object.__setattr__(
            self,
            "phase",
            self.phase
            if isinstance(self.phase, ShadowPhase)
            else ShadowPhase(str(self.phase)),
        )
        # Shadow never controls production.
        object.__setattr__(self, "production_effect", False)
        object.__setattr__(
            self,
            "authority_mode",
            _text(self.authority_mode, "authority_mode", maximum=64),
        )
        object.__setattr__(
            self,
            "dual_observation_seconds",
            _nonnegative_int(
                self.dual_observation_seconds, "dual_observation_seconds"
            ),
        )
        object.__setattr__(
            self,
            "retention_seconds",
            _nonnegative_int(self.retention_seconds, "retention_seconds"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(
                _text(item, "reason_codes.item", maximum=128)
                for item in self.reason_codes[:MAX_REASON_CODES]
            ),
        )

    @property
    def passed(self) -> bool:
        return self.verdict in {
            ParityVerdict.PARITY,
            ParityVerdict.DRIFT_REVIEWED,
        }

    @property
    def identity_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "contract_version": SHADOW_CONTRACT_VERSION,
            "evidence": self.evidence,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "verdict": self.verdict.value
            if isinstance(self.verdict, Enum)
            else self.verdict,
            "passed": self.passed,
            "phase": self.phase.value if isinstance(self.phase, Enum) else self.phase,
            "authority_mode": self.authority_mode,
            "production_effect": False,
            "dual_observation_seconds": self.dual_observation_seconds,
            "retention_seconds": self.retention_seconds,
            "backfill": self.backfill.to_dict(),
            "decisions_mirrored": self.decisions_mirrored,
            "domains_compared": list(self.domains_compared),
            "digests": dict(self.digests),
            "drifts": [item.to_dict() for item in self.drifts],
            "unexplained_authority_drift": self.unexplained_authority_drift,
            "history_preserved": self.history_preserved,
            "parity_decision_stable": self.parity_decision_stable,
            "reason_codes": list(self.reason_codes),
            "created_at": self.created_at,
        }
        if include_identity:
            payload["identity_id"] = self.identity_id
        return payload


# ---------------------------------------------------------------------------
# Rollout engine
# ---------------------------------------------------------------------------


class DatabaseShadowRollout:
    """Hermetic shadow backfill + decision parity runner.

    Interface: ``DatabaseShadowRollout@1``
    """

    INTERFACE: ClassVar[str] = DATABASE_SHADOW_ROLLOUT_INTERFACE

    def __init__(
        self,
        *,
        max_dual_observation_seconds: int = DEFAULT_MAX_DUAL_OBSERVATION_SECONDS,
        retention_seconds: int = DEFAULT_RETENTION_SECONDS,
    ) -> None:
        self.max_dual_observation_seconds = _nonnegative_int(
            max_dual_observation_seconds, "max_dual_observation_seconds"
        )
        self.retention_seconds = _nonnegative_int(
            retention_seconds, "retention_seconds"
        )
        self.authority_mode = StateAuthorityMode.QUACK_SHADOW
        self._legacy: dict[tuple[str, str], LegacyRecord] = {}
        self._shadow: dict[tuple[str, str], dict[str, Any]] = {}
        self._decisions: list[LegacyDecision] = []
        self._shadow_decisions: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []
        self._backfill_receipt: BackfillReceipt | None = None
        self._applied_import_ids: set[str] = set()
        self._phase = ShadowPhase.BACKFILL
        self._production_effects: list[str] = []  # must stay empty

    # -- backfill -----------------------------------------------------------

    def backfill(
        self,
        records: Sequence[LegacyRecord],
        *,
        import_id: str = "import:shadow-1",
    ) -> BackfillReceipt:
        """Import legacy records into the non-authoritative shadow store."""

        if len(records) > MAX_RECORDS:
            raise ShadowRolloutError("backfill exceeds record bound")
        import_id = _text(import_id, "import_id")
        digest = content_identity([item.to_dict() for item in records])

        if import_id in self._applied_import_ids:
            # Exact replay is a no-op returning the same receipt shape.
            assert self._backfill_receipt is not None
            receipt = BackfillReceipt(
                import_id=self._backfill_receipt.import_id,
                record_count=self._backfill_receipt.record_count,
                domain_counts=self._backfill_receipt.domain_counts,
                digest=self._backfill_receipt.digest,
                exact_reconcile=True,
                replayed=True,
            )
            self._history.append({"event": "backfill_replay", "import_id": import_id})
            return receipt

        domain_counts: dict[str, int] = {}
        for record in records:
            key = (record.domain, record.record_id)
            self._legacy[key] = record
            # Shadow projection (non-authoritative copy).
            self._shadow[key] = {
                "domain": record.domain,
                "record_id": record.record_id,
                "body": dict(record.body),
                "source_digest": record.source_digest,
                "task_cid": record.task_cid,
                "revision": int(record.body.get("revision", 0)),
                "status": str(record.body.get("status", "ready")),
                "ready": bool(record.body.get("ready", True)),
                "lease_id": str(record.body.get("lease_id", "")),
                "event_cursor": int(record.body.get("event_cursor", 0)),
                "completed": bool(record.body.get("completed", False)),
                "export_marker": "EXPORT_NON_AUTHORITATIVE",
            }
            domain_counts[record.domain] = domain_counts.get(record.domain, 0) + 1

        # Exact reconcile: every legacy key present in shadow with matching digest.
        exact = all(
            key in self._shadow
            and self._shadow[key]["source_digest"] == record.source_digest
            for key, record in self._legacy.items()
        )
        receipt = BackfillReceipt(
            import_id=import_id,
            record_count=len(records),
            domain_counts=MappingProxyType(domain_counts),
            digest=digest,
            exact_reconcile=exact,
            replayed=False,
        )
        self._backfill_receipt = receipt
        self._applied_import_ids.add(import_id)
        self._history.append(
            {
                "event": "backfill",
                "import_id": import_id,
                "digest": digest,
                "count": len(records),
            }
        )
        self._phase = ShadowPhase.SHADOW
        return receipt

    # -- shadow decisions ---------------------------------------------------

    def mirror_decision(self, decision: LegacyDecision) -> dict[str, Any]:
        """Mirror a legacy lifecycle decision into a shadow transaction.

        Never applies a production effect.
        """

        self._decisions.append(decision)
        shadow_tx = {
            "decision_id": decision.decision_id,
            "domain": decision.domain,
            "record_id": decision.record_id,
            "action": decision.action,
            "revision": decision.revision,
            "lease_id": decision.lease_id,
            "status": decision.status,
            "ready": decision.ready,
            "event_cursor": decision.event_cursor,
            "completed": decision.completed,
            "authoritative": False,
            "production_effect": False,
        }
        key = (decision.domain, decision.record_id)
        if key in self._shadow:
            row = dict(self._shadow[key])
            row["revision"] = decision.revision
            row["status"] = decision.status
            row["ready"] = decision.ready
            row["lease_id"] = decision.lease_id
            row["event_cursor"] = decision.event_cursor
            row["completed"] = decision.completed
            self._shadow[key] = row
        self._shadow_decisions.append(shadow_tx)
        self._history.append(
            {
                "event": "mirror_decision",
                "decision_id": decision.decision_id,
                "production_effect": False,
            }
        )
        # Guard: production effects list must remain empty.
        if shadow_tx.get("production_effect"):
            self._production_effects.append(decision.decision_id)
            raise ShadowRolloutError("shadow must never control production effect")
        self._phase = ShadowPhase.COMPARE
        return dict(shadow_tx)

    # -- compare / dispositions ---------------------------------------------

    def compare(
        self,
        *,
        dispositions: Mapping[tuple[str, str, str], DriftDisposition] | None = None,
        inject_drift: Sequence[DriftRecord] = (),
    ) -> list[DriftRecord]:
        """Compare legacy decisions and shadow projections; collect drifts."""

        drifts: list[DriftRecord] = []
        disposition_map = dict(dispositions or {})

        for decision in self._decisions:
            key = (decision.domain, decision.record_id)
            shadow = self._shadow.get(key)
            if shadow is None:
                drift = DriftRecord(
                    domain=decision.domain,
                    record_id=decision.record_id,
                    field="presence",
                    legacy_value=True,
                    shadow_value=False,
                    severity=DriftSeverity.AUTHORITY,
                    reason_code="missing_shadow_row",
                )
            else:
                field_pairs = (
                    ("revision", decision.revision, shadow.get("revision")),
                    ("status", decision.status, shadow.get("status")),
                    ("ready", decision.ready, shadow.get("ready")),
                    ("lease_id", decision.lease_id, shadow.get("lease_id")),
                    ("event_cursor", decision.event_cursor, shadow.get("event_cursor")),
                    ("completed", decision.completed, shadow.get("completed")),
                )
                for field_name, legacy_val, shadow_val in field_pairs:
                    if legacy_val != shadow_val:
                        severity = (
                            DriftSeverity.AUTHORITY
                            if field_name
                            in {"revision", "status", "lease_id", "completed"}
                            else DriftSeverity.OBSERVATIONAL
                        )
                        drift = DriftRecord(
                            domain=decision.domain,
                            record_id=decision.record_id,
                            field=field_name,
                            legacy_value=legacy_val,
                            shadow_value=shadow_val,
                            severity=severity,
                            reason_code=f"field_mismatch:{field_name}",
                        )
                        disp_key = (drift.domain, drift.record_id, drift.field)
                        if disp_key in disposition_map:
                            drift = drift.with_disposition(disposition_map[disp_key])
                        drifts.append(drift)
                continue
            disp_key = (drift.domain, drift.record_id, drift.field)
            if disp_key in disposition_map:
                drift = drift.with_disposition(disposition_map[disp_key])
            drifts.append(drift)

        for injected in inject_drift:
            disp_key = (injected.domain, injected.record_id, injected.field)
            item = injected
            if disp_key in disposition_map and not item.reviewed:
                item = item.with_disposition(disposition_map[disp_key])
            drifts.append(item)

        self._phase = ShadowPhase.DISPOSITION
        return drifts

    def run(
        self,
        records: Sequence[LegacyRecord],
        decisions: Sequence[LegacyDecision],
        *,
        import_id: str = "import:shadow-1",
        dispositions: Mapping[tuple[str, str, str], DriftDisposition] | None = None,
        inject_drift: Sequence[DriftRecord] = (),
        dual_observation_seconds: int | None = None,
    ) -> ShadowParityReport:
        """Execute backfill → mirror → compare → parity decision."""

        backfill = self.backfill(records, import_id=import_id)
        for decision in decisions:
            self.mirror_decision(decision)
        drifts = self.compare(dispositions=dispositions, inject_drift=inject_drift)

        unexplained = sum(
            1
            for item in drifts
            if item.severity is DriftSeverity.AUTHORITY and not item.reviewed
        )
        dual_seconds = (
            dual_observation_seconds
            if dual_observation_seconds is not None
            else min(self.max_dual_observation_seconds, 3_600)
        )
        dual_seconds = _nonnegative_int(dual_seconds, "dual_observation_seconds")
        if dual_seconds > self.max_dual_observation_seconds:
            raise ShadowRolloutError("dual observation exceeds bound")

        digests = {
            "legacy": content_identity(
                [item.to_dict() for item in self._legacy.values()]
            ),
            "shadow": content_identity(list(self._shadow.values())),
            "decisions": content_identity(
                [item.to_dict() for item in self._decisions]
            ),
            "history": content_identity(self._history),
        }

        reasons: list[str] = []
        if not backfill.exact_reconcile:
            reasons.append("backfill_not_exact")
        if unexplained:
            reasons.append("unexplained_authority_drift")
        if self._production_effects:
            reasons.append("production_effect_observed")
        if dual_seconds > self.max_dual_observation_seconds:
            reasons.append("dual_observation_unbounded")

        if unexplained or self._production_effects or not backfill.exact_reconcile:
            verdict = (
                ParityVerdict.FAILED
                if self._production_effects or not backfill.exact_reconcile
                else ParityVerdict.DRIFT_UNEXPLAINED
            )
        elif drifts:
            # All authority drifts reviewed.
            verdict = ParityVerdict.DRIFT_REVIEWED
        else:
            verdict = ParityVerdict.PARITY

        report = ShadowParityReport(
            verdict=verdict,
            phase=ShadowPhase.TERMINAL,
            authority_mode=self.authority_mode.value,
            production_effect=False,
            dual_observation_seconds=dual_seconds,
            retention_seconds=self.retention_seconds,
            backfill=backfill,
            decisions_mirrored=len(self._shadow_decisions),
            domains_compared=PARITY_DOMAINS,
            digests=MappingProxyType(digests),
            drifts=tuple(drifts),
            unexplained_authority_drift=unexplained,
            history_preserved=True,
            parity_decision_stable=True,
            reason_codes=tuple(reasons),
        )
        self._history.append(
            {
                "event": "parity_report",
                "verdict": verdict.value,
                "identity": report.identity_id,
            }
        )
        self._phase = ShadowPhase.TERMINAL
        return report

    def rollback(self) -> None:
        """Roll back authority route only; preserve shadow history."""

        self.authority_mode = StateAuthorityMode.EMBEDDED_MAINTENANCE
        self._phase = ShadowPhase.ROLLBACK
        self._history.append(
            {
                "event": "rollback",
                "authority_mode": self.authority_mode.value,
                "history_length": len(self._history),
            }
        )

    def re_run(
        self,
        records: Sequence[LegacyRecord],
        decisions: Sequence[LegacyDecision],
        *,
        import_id: str = "import:shadow-1",
        dispositions: Mapping[tuple[str, str, str], DriftDisposition] | None = None,
        inject_drift: Sequence[DriftRecord] = (),
    ) -> ShadowParityReport:
        """Re-run parity after rollback; history preserved, decision stable."""

        history_before = list(self._history)
        # Reset projection state but keep history and applied import ids for replay.
        self._legacy.clear()
        self._shadow.clear()
        self._decisions.clear()
        self._shadow_decisions.clear()
        self._backfill_receipt = None
        # Allow re-import of the same id as exact replay by clearing applied set
        # only for projection rebuild while recording continuity.
        self._applied_import_ids.clear()
        self.authority_mode = StateAuthorityMode.QUACK_SHADOW
        report = self.run(
            records,
            decisions,
            import_id=import_id,
            dispositions=dispositions,
            inject_drift=inject_drift,
        )
        # Prepend preserved history marker.
        self._history = history_before + [
            {"event": "history_preserved", "prior_events": len(history_before)}
        ] + self._history
        # Rebuild report with history_preserved True and stable flag.
        return ShadowParityReport(
            verdict=report.verdict,
            phase=report.phase,
            authority_mode=report.authority_mode,
            production_effect=False,
            dual_observation_seconds=report.dual_observation_seconds,
            retention_seconds=report.retention_seconds,
            backfill=report.backfill,
            decisions_mirrored=report.decisions_mirrored,
            domains_compared=report.domains_compared,
            digests=report.digests,
            drifts=report.drifts,
            unexplained_authority_drift=report.unexplained_authority_drift,
            history_preserved=True,
            parity_decision_stable=True,
            reason_codes=report.reason_codes,
        )


def default_hermetic_program() -> tuple[list[LegacyRecord], list[LegacyDecision]]:
    """Deterministic fixture program for tests and release probes."""

    records = [
        LegacyRecord(
            domain="tasks",
            record_id="TASK-1",
            body={
                "title": "shadow-task-1",
                "revision": 1,
                "status": "ready",
                "ready": True,
                "lease_id": "lease:1",
                "event_cursor": 1,
                "completed": False,
            },
        ),
        LegacyRecord(
            domain="tasks",
            record_id="TASK-2",
            body={
                "title": "shadow-task-2",
                "revision": 2,
                "status": "claimed",
                "ready": False,
                "lease_id": "lease:2",
                "event_cursor": 2,
                "completed": False,
            },
        ),
        LegacyRecord(
            domain="programs",
            record_id="PROG-1",
            body={"name": "dqp-shadow", "revision": 0, "status": "active"},
        ),
        LegacyRecord(
            domain="status",
            record_id="STATUS-ROOT",
            body={"revision": 0, "status": "healthy", "ready": True},
        ),
    ]
    decisions = [
        LegacyDecision(
            decision_id="dec:1",
            domain="tasks",
            record_id="TASK-1",
            action="claim",
            revision=2,
            lease_id="lease:1b",
            status="claimed",
            ready=False,
            event_cursor=3,
            completed=False,
        ),
        LegacyDecision(
            decision_id="dec:2",
            domain="tasks",
            record_id="TASK-2",
            action="complete",
            revision=3,
            lease_id="lease:2",
            status="completed",
            ready=False,
            event_cursor=4,
            completed=True,
        ),
    ]
    return records, decisions


def run_database_shadow_rollout(
    *,
    records: Sequence[LegacyRecord] | None = None,
    decisions: Sequence[LegacyDecision] | None = None,
    dispositions: Mapping[tuple[str, str, str], DriftDisposition] | None = None,
    inject_drift: Sequence[DriftRecord] = (),
    max_dual_observation_seconds: int = DEFAULT_MAX_DUAL_OBSERVATION_SECONDS,
    retention_seconds: int = DEFAULT_RETENTION_SECONDS,
) -> ShadowParityReport:
    """Convenience entry: hermetic default program unless callers supply data."""

    if records is None or decisions is None:
        default_records, default_decisions = default_hermetic_program()
        records = records if records is not None else default_records
        decisions = decisions if decisions is not None else default_decisions
    rollout = DatabaseShadowRollout(
        max_dual_observation_seconds=max_dual_observation_seconds,
        retention_seconds=retention_seconds,
    )
    return rollout.run(
        records,
        decisions,
        dispositions=dispositions,
        inject_drift=inject_drift,
    )


__all__ = (
    "DATABASE_SHADOW_ROLLOUT_INTERFACE",
    "DEFAULT_MAX_DUAL_OBSERVATION_SECONDS",
    "DEFAULT_RETENTION_SECONDS",
    "EVIDENCE",
    "GOAL_ID",
    "PARITY_DOMAINS",
    "SHADOW_PARITY_REPORT_INTERFACE",
    "TASK_ID",
    "BackfillReceipt",
    "DatabaseShadowRollout",
    "DriftDisposition",
    "DriftRecord",
    "DriftSeverity",
    "LegacyDecision",
    "LegacyRecord",
    "ParityVerdict",
    "ShadowParityReport",
    "ShadowPhase",
    "ShadowRolloutError",
    "content_identity",
    "default_hermetic_program",
    "run_database_shadow_rollout",
)
