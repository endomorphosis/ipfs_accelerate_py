"""Exact, read-only architecture and event drift monitoring for CASF.

The monitor compares an admitted root set with one current observation.  It is
deliberately a pure observer: it accepts no database path or state client and
cannot schedule, complete, admit, or mutate work.  Findings therefore block a
later promotion decision without manufacturing that decision themselves.

Interface: ``FederationDriftMonitor@1``
Evidence: ``casf/drift-report@1``
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar

from ..task_sources.control_plane_contracts import content_identity
from .contracts import (
    FederationBinding,
    FederationBoundsError,
    FederationContractError,
    UnknownNormativeFieldError,
    _identifier,
    _integer,
    _text,
    _timestamp,
)
from .events import DomainEvent, EventClass

FEDERATION_DRIFT_MONITOR_INTERFACE = "FederationDriftMonitor@1"
DRIFT_ROOT_SET_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/drift-root-set@1"
)
DRIFT_FINDING_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/drift-finding@1"
)
DRIFT_REPORT_SCHEMA = "casf/drift-report@1"

MAX_DRIFT_EVENTS = 4_096
MAX_DRIFT_FINDINGS = 16_384
MAX_KNOWN_CAUSAL_PARENTS = 65_536


class DriftMonitorError(FederationContractError):
    """A drift observation is malformed or cannot be bound exactly."""


class StaleDriftReportError(DriftMonitorError):
    """A report is not evidence for the requested current tree/generation."""


class DriftKind(str, Enum):  # noqa: UP042 - Python 3.8 compatibility
    """Closed drift dimensions understood by the monitor."""

    BINDING = "binding"
    SCHEMA = "schema"
    OPERATION = "operation"
    EVENT = "event"
    CAUSAL = "causal"


# A descriptive alias for callers that use "dimension" terminology.
DriftDimension = DriftKind


def _closed_mapping(
    payload: Mapping[str, Any],
    name: str,
    fields: tuple[str, ...],
    *,
    schema: str | None = None,
) -> Mapping[str, Any]:
    """Reject an incomplete or extensible wire payload before decoding it."""

    if not isinstance(payload, Mapping):
        raise DriftMonitorError(f"{name} payload must be an object")
    allowed = set(fields)
    if schema is not None:
        allowed.add("schema")
        if payload.get("schema") != schema:
            raise DriftMonitorError(f"{name}.schema must equal {schema!r}")
    unknown = set(payload) - allowed
    if unknown:
        raise UnknownNormativeFieldError(
            f"{name} has unknown fields: {sorted(unknown)}"
        )
    missing = set(fields) - set(payload)
    if missing:
        raise DriftMonitorError(f"{name} is missing fields: {sorted(missing)}")
    return payload


def _closed_sequence(value: Any, name: str) -> Sequence[Any]:
    """Accept only a bounded wire-array shape, never a text or mapping value."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DriftMonitorError(f"{name} must be an array")
    return value


def closed_event_catalog_root() -> str:
    """Return the content identity of the exact closed CASF event catalog."""

    return content_identity(
        {
            "schema": "casf/closed-event-catalog@1",
            "event_classes": sorted(item.value for item in EventClass),
        }
    )


def _bounded_identifiers(
    values: Sequence[str],
    name: str,
    *,
    maximum: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise DriftMonitorError(f"{name} must be an array")
    result = tuple(values)
    if len(result) > maximum:
        raise FederationBoundsError(f"{name} exceeds its bound")
    for value in result:
        _identifier(value, name)
    if len(set(result)) != len(result):
        raise DriftMonitorError(f"{name} contains duplicate identities")
    return result


@dataclass(frozen=True)
class FederationDriftRoots:
    """Exact roots and monotonic positions for one federation observation."""

    SCHEMA: ClassVar[str] = DRIFT_ROOT_SET_SCHEMA

    tenant_id: str
    federation_id: str
    repository_id: str
    repository_tree_id: str
    control_plane_generation: int
    schema_root: str
    operation_catalog_root: str
    event_catalog_root: str
    causal_graph_root: str
    causal_graph_revision: int
    event_watermark: int

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "federation_id",
            "repository_id",
            "repository_tree_id",
            "schema_root",
            "operation_catalog_root",
            "event_catalog_root",
            "causal_graph_root",
        ):
            _identifier(getattr(self, name), name)
        _integer(
            self.control_plane_generation,
            "control_plane_generation",
            minimum=1,
        )
        _integer(self.causal_graph_revision, "causal_graph_revision")
        _integer(self.event_watermark, "event_watermark")

    @classmethod
    def from_binding(
        cls,
        binding: FederationBinding,
        *,
        federation_id: str,
        schema_root: str,
        event_catalog_root: str,
        causal_graph_root: str,
        event_watermark: int,
        repository_id: str | None = None,
        repository_tree_id: str | None = None,
        operation_catalog_root: str | None = None,
    ) -> FederationDriftRoots:
        """Project a closed binding into an exact monitor root set."""

        if not isinstance(binding, FederationBinding):
            raise DriftMonitorError("binding must be a FederationBinding")
        selected_repository = (
            binding.repository_ids[0]
            if repository_id is None
            else repository_id
        )
        try:
            repository_index = binding.repository_ids.index(selected_repository)
        except ValueError as exc:
            raise DriftMonitorError(
                "repository_id is absent from the federation binding"
            ) from exc
        bound_tree = binding.repository_tree_ids[repository_index]
        selected_tree = bound_tree if repository_tree_id is None else repository_tree_id
        if selected_tree != bound_tree:
            raise DriftMonitorError(
                "repository_tree_id disagrees with the federation binding"
            )
        return cls(
            tenant_id=binding.tenant_id,
            federation_id=federation_id,
            repository_id=selected_repository,
            repository_tree_id=selected_tree,
            control_plane_generation=binding.control_plane_generation,
            schema_root=schema_root,
            operation_catalog_root=(
                binding.operation_catalog_ref
                if operation_catalog_root is None
                else operation_catalog_root
            ),
            event_catalog_root=event_catalog_root,
            causal_graph_root=causal_graph_root,
            causal_graph_revision=binding.causal_graph_revision,
            event_watermark=event_watermark,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "tenant_id": self.tenant_id,
            "federation_id": self.federation_id,
            "repository_id": self.repository_id,
            "repository_tree_id": self.repository_tree_id,
            "control_plane_generation": self.control_plane_generation,
            "schema_root": self.schema_root,
            "operation_catalog_root": self.operation_catalog_root,
            "event_catalog_root": self.event_catalog_root,
            "causal_graph_root": self.causal_graph_root,
            "causal_graph_revision": self.causal_graph_revision,
            "event_watermark": self.event_watermark,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FederationDriftRoots:
        """Decode an exact, versioned root observation without extensions."""

        fields = (
            "tenant_id",
            "federation_id",
            "repository_id",
            "repository_tree_id",
            "control_plane_generation",
            "schema_root",
            "operation_catalog_root",
            "event_catalog_root",
            "causal_graph_root",
            "causal_graph_revision",
            "event_watermark",
        )
        values = _closed_mapping(payload, cls.__name__, fields, schema=cls.SCHEMA)
        return cls(**{field: values[field] for field in fields})

    @property
    def cid(self) -> str:
        return content_identity(self.to_dict())


@dataclass(frozen=True)
class DriftFinding:
    """One precise mismatch; it is evidence, never corrective authority."""

    SCHEMA: ClassVar[str] = DRIFT_FINDING_SCHEMA

    kind: DriftKind
    code: str
    subject_ref: str
    expected: str
    observed: str
    blocks_promotion: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.kind, DriftKind):
            raise DriftMonitorError("finding kind is not closed")
        _identifier(self.code, "code")
        _identifier(self.subject_ref, "subject_ref")
        _text(self.expected, "expected")
        _text(self.observed, "observed")
        if self.blocks_promotion is not True:
            raise DriftMonitorError("drift findings must block promotion")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "kind": self.kind.value,
            "code": self.code,
            "subject_ref": self.subject_ref,
            "expected": self.expected,
            "observed": self.observed,
            "blocks_promotion": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DriftFinding:
        """Decode one closed, promotion-blocking drift finding."""

        fields = (
            "kind",
            "code",
            "subject_ref",
            "expected",
            "observed",
            "blocks_promotion",
        )
        values = _closed_mapping(payload, cls.__name__, fields, schema=cls.SCHEMA)
        try:
            kind = DriftKind(values["kind"])
        except (TypeError, ValueError) as exc:
            raise DriftMonitorError("finding kind is not closed") from exc
        return cls(
            kind=kind,
            code=values["code"],
            subject_ref=values["subject_ref"],
            expected=values["expected"],
            observed=values["observed"],
            blocks_promotion=values["blocks_promotion"],
        )


@dataclass(frozen=True)
class DriftReport:
    """Content-addressed, non-authoritative result of one exact comparison."""

    SCHEMA: ClassVar[str] = DRIFT_REPORT_SCHEMA

    report_id: str
    baseline: FederationDriftRoots
    observed: FederationDriftRoots
    observed_at: str
    event_range_start: int
    event_range_end: int
    observed_event_count: int
    findings: tuple[DriftFinding, ...]

    def __post_init__(self) -> None:
        _identifier(self.report_id, "report_id")
        if not isinstance(self.baseline, FederationDriftRoots):
            raise DriftMonitorError("baseline roots are required")
        if not isinstance(self.observed, FederationDriftRoots):
            raise DriftMonitorError("observed roots are required")
        _timestamp(self.observed_at, "observed_at")
        _integer(self.event_range_start, "event_range_start")
        _integer(self.event_range_end, "event_range_end")
        _integer(
            self.observed_event_count,
            "observed_event_count",
            maximum=MAX_DRIFT_EVENTS,
        )
        if self.event_range_end < self.event_range_start:
            if self.observed_event_count != 0:
                raise DriftMonitorError("empty event range has a nonzero count")
        elif (
            self.event_range_end - self.event_range_start + 1
            < self.observed_event_count
        ):
            raise DriftMonitorError("event count exceeds the reported range")
        if not isinstance(self.findings, tuple) or any(
            not isinstance(item, DriftFinding) for item in self.findings
        ):
            raise DriftMonitorError("findings must be DriftFinding records")
        if len(self.findings) > MAX_DRIFT_FINDINGS:
            raise FederationBoundsError("drift finding bound exceeded")
        if self.findings != tuple(
            sorted(
                self.findings,
                key=lambda item: (
                    item.kind.value,
                    item.code,
                    item.subject_ref,
                    item.expected,
                    item.observed,
                ),
            )
        ):
            raise DriftMonitorError("drift findings are not canonical")
        if self.report_id != content_identity(self._identity_payload()):
            raise DriftMonitorError("drift report content identity mismatches")

    @property
    def drifted(self) -> bool:
        return bool(self.findings)

    @property
    def current(self) -> bool:
        return not self.findings

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "baseline": self.baseline.to_dict(),
            "observed": self.observed.to_dict(),
            "observed_at": self.observed_at,
            "event_range": {
                "start": self.event_range_start,
                "end": self.event_range_end,
                "count": self.observed_event_count,
            },
            "findings": [item.to_dict() for item in self.findings],
            "authority": False,
            "production_state_changed": False,
            "ducklake_authoritative": False,
            "model_calls": 0,
            "provider_calls": 0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "report_id": self.report_id,
            "status": "drifted" if self.drifted else "current",
            "blocks_promotion": self.drifted,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DriftReport:
        """Decode canonical evidence and reject forged status or capability flags."""

        fields = (
            "report_id",
            "baseline",
            "observed",
            "observed_at",
            "event_range",
            "findings",
            "authority",
            "production_state_changed",
            "ducklake_authoritative",
            "model_calls",
            "provider_calls",
            "status",
            "blocks_promotion",
        )
        values = _closed_mapping(payload, cls.__name__, fields, schema=cls.SCHEMA)
        event_range = _closed_mapping(
            values["event_range"],
            "DriftReport.event_range",
            ("start", "end", "count"),
        )
        findings = tuple(
            DriftFinding.from_dict(item)
            for item in _closed_sequence(values["findings"], "findings")
        )
        for field in (
            "authority",
            "production_state_changed",
            "ducklake_authoritative",
        ):
            if values[field] is not False:
                raise DriftMonitorError(f"DriftReport.{field} must be false")
        for field in ("model_calls", "provider_calls"):
            if (
                isinstance(values[field], bool)
                or not isinstance(values[field], int)
                or values[field] != 0
            ):
                raise DriftMonitorError(f"DriftReport.{field} must be zero")
        report = cls(
            report_id=values["report_id"],
            baseline=FederationDriftRoots.from_dict(values["baseline"]),
            observed=FederationDriftRoots.from_dict(values["observed"]),
            observed_at=values["observed_at"],
            event_range_start=event_range["start"],
            event_range_end=event_range["end"],
            observed_event_count=event_range["count"],
            findings=findings,
        )
        if values["status"] != ("drifted" if report.drifted else "current"):
            raise DriftMonitorError("DriftReport.status disagrees with findings")
        if values["blocks_promotion"] is not report.drifted:
            raise DriftMonitorError(
                "DriftReport.blocks_promotion disagrees with findings"
            )
        return report


def _finding(
    kind: DriftKind,
    code: str,
    subject_ref: str,
    expected: Any,
    observed: Any,
) -> DriftFinding:
    return DriftFinding(
        kind=kind,
        code=code,
        subject_ref=subject_ref,
        expected=str(expected),
        observed=str(observed),
    )


def _append_finding(
    findings: list[DriftFinding],
    finding: DriftFinding,
) -> None:
    """Append bounded evidence instead of constructing an oversized report.

    A hostile but individually valid event batch can create several findings
    per event (notably one per causal parent).  Do not let that turn a
    read-only observation into unbounded memory use.  A typed bounds failure
    is fail-closed and leaves no partially trustworthy report behind.
    """

    if len(findings) >= MAX_DRIFT_FINDINGS:
        raise FederationBoundsError("drift finding bound exceeded")
    findings.append(finding)


class FederationDriftMonitor:
    """Compare exact roots and event windows without any stateful effect."""

    INTERFACE: ClassVar[str] = FEDERATION_DRIFT_MONITOR_INTERFACE

    def __init__(self, baseline: FederationDriftRoots) -> None:
        if not isinstance(baseline, FederationDriftRoots):
            raise DriftMonitorError(
                "drift monitor requires exact roots, never a database path"
            )
        self._baseline = baseline

    @property
    def baseline(self) -> FederationDriftRoots:
        return self._baseline

    def observe(
        self,
        observed: FederationDriftRoots,
        *,
        events: Sequence[DomainEvent] = (),
        known_causal_parent_ids: Sequence[str] = (),
        observed_at: str,
    ) -> DriftReport:
        """Produce one deterministic current-root drift report."""

        if not isinstance(observed, FederationDriftRoots):
            raise DriftMonitorError("observed roots are required")
        if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
            raise DriftMonitorError("events must be an array")
        if len(events) > MAX_DRIFT_EVENTS:
            raise FederationBoundsError("event observation bound exceeded")
        if any(not isinstance(item, DomainEvent) for item in events):
            raise DriftMonitorError("events must be closed DomainEvent records")
        known_parents = _bounded_identifiers(
            known_causal_parent_ids,
            "known_causal_parent_ids",
            maximum=MAX_KNOWN_CAUSAL_PARENTS,
        )
        _timestamp(observed_at, "observed_at")

        findings: list[DriftFinding] = []
        self._compare_roots(observed, findings)
        self._compare_events(observed, tuple(events), known_parents, findings)
        canonical_findings = tuple(
            sorted(
                findings,
                key=lambda item: (
                    item.kind.value,
                    item.code,
                    item.subject_ref,
                    item.expected,
                    item.observed,
                ),
            )
        )
        sequences = tuple(item.global_sequence for item in events)
        range_start = min(sequences) if sequences else observed.event_watermark + 1
        range_end = max(sequences) if sequences else observed.event_watermark
        payload = {
            "schema": DRIFT_REPORT_SCHEMA,
            "baseline": self._baseline.to_dict(),
            "observed": observed.to_dict(),
            "observed_at": observed_at,
            "event_range": {
                "start": range_start,
                "end": range_end,
                "count": len(events),
            },
            "findings": [item.to_dict() for item in canonical_findings],
            "authority": False,
            "production_state_changed": False,
            "ducklake_authoritative": False,
            "model_calls": 0,
            "provider_calls": 0,
        }
        return DriftReport(
            report_id=content_identity(payload),
            baseline=self._baseline,
            observed=observed,
            observed_at=observed_at,
            event_range_start=range_start,
            event_range_end=range_end,
            observed_event_count=len(events),
            findings=canonical_findings,
        )

    scan = observe

    def _compare_roots(
        self,
        observed: FederationDriftRoots,
        findings: list[DriftFinding],
    ) -> None:
        expected = self._baseline
        comparisons = (
            (
                DriftKind.BINDING,
                "tenant_changed",
                "tenant",
                expected.tenant_id,
                observed.tenant_id,
            ),
            (
                DriftKind.BINDING,
                "federation_changed",
                "federation",
                expected.federation_id,
                observed.federation_id,
            ),
            (
                DriftKind.BINDING,
                "repository_changed",
                "repository",
                expected.repository_id,
                observed.repository_id,
            ),
            (
                DriftKind.BINDING,
                "repository_tree_changed",
                "repository_tree",
                expected.repository_tree_id,
                observed.repository_tree_id,
            ),
            (
                DriftKind.BINDING,
                "control_plane_generation_changed",
                "control_plane_generation",
                expected.control_plane_generation,
                observed.control_plane_generation,
            ),
            (
                DriftKind.SCHEMA,
                "schema_root_changed",
                "schema_root",
                expected.schema_root,
                observed.schema_root,
            ),
            (
                DriftKind.OPERATION,
                "operation_catalog_root_changed",
                "operation_catalog_root",
                expected.operation_catalog_root,
                observed.operation_catalog_root,
            ),
            (
                DriftKind.EVENT,
                "event_catalog_root_changed",
                "event_catalog_root",
                expected.event_catalog_root,
                observed.event_catalog_root,
            ),
            (
                DriftKind.CAUSAL,
                "causal_graph_root_changed",
                "causal_graph_root",
                expected.causal_graph_root,
                observed.causal_graph_root,
            ),
            (
                DriftKind.CAUSAL,
                "causal_graph_revision_changed",
                "causal_graph_revision",
                expected.causal_graph_revision,
                observed.causal_graph_revision,
            ),
        )
        for kind, code, subject, prior, current in comparisons:
            if prior != current:
                _append_finding(
                    findings, _finding(kind, code, subject, prior, current)
                )
        if observed.event_watermark < expected.event_watermark:
            _append_finding(
                findings,
                _finding(
                    DriftKind.EVENT,
                    "event_watermark_regressed",
                    "event_watermark",
                    expected.event_watermark,
                    observed.event_watermark,
                ),
            )

    def _compare_events(
        self,
        observed: FederationDriftRoots,
        events: tuple[DomainEvent, ...],
        known_parents: tuple[str, ...],
        findings: list[DriftFinding],
    ) -> None:
        expected = self._baseline
        watermark_span = observed.event_watermark - expected.event_watermark
        if not events:
            if observed.event_watermark > expected.event_watermark:
                _append_finding(
                    findings,
                    _finding(
                        DriftKind.EVENT,
                        "event_watermark_advance_unobserved",
                        "event_watermark",
                        expected.event_watermark,
                        observed.event_watermark,
                    ),
                )
            return

        ordered = tuple(
            sorted(events, key=lambda item: (item.global_sequence, item.event_id))
        )
        sequences = tuple(item.global_sequence for item in ordered)
        event_ids = tuple(item.event_id for item in ordered)
        duplicate_ids = sorted(
            event_id for event_id, count in Counter(event_ids).items() if count > 1
        )
        for event_id in duplicate_ids:
            _append_finding(
                findings,
                _finding(
                    DriftKind.EVENT,
                    "duplicate_event_id",
                    event_id,
                    "unique",
                    "duplicate",
                ),
            )
        duplicate_sequences = sorted(
            sequence
            for sequence, count in Counter(sequences).items()
            if count > 1
        )
        for sequence in duplicate_sequences:
            _append_finding(
                findings,
                _finding(
                    DriftKind.EVENT,
                    "duplicate_global_sequence",
                    f"global_sequence:{sequence}",
                    "unique",
                    "duplicate",
                ),
            )

        if watermark_span > MAX_DRIFT_EVENTS:
            # Never materialize an attacker-controlled integer range.  The
            # bounded input contract makes an exact observation impossible,
            # so this window is conclusively drifted rather than truncated.
            _append_finding(
                findings,
                _finding(
                    DriftKind.EVENT,
                    "event_window_exceeds_observation_bound",
                    "event_range",
                    f"at_most:{MAX_DRIFT_EVENTS}",
                    watermark_span,
                ),
            )
        else:
            expected_sequences = tuple(
                range(expected.event_watermark + 1, observed.event_watermark + 1)
            )
            if sequences != expected_sequences:
                _append_finding(
                    findings,
                    _finding(
                        DriftKind.EVENT,
                        "event_range_not_exact",
                        "event_range",
                        _compact_sequence(expected_sequences),
                        _compact_sequence(sequences),
                    ),
                )

        available_parents = set(known_parents)
        for event in ordered:
            for field_name, expected_value, observed_value in (
                ("tenant_id", observed.tenant_id, event.tenant_id),
                ("federation_id", observed.federation_id, event.federation_id),
                ("repository_id", observed.repository_id, event.repository_id),
                ("tree_id", observed.repository_tree_id, event.tree_id),
            ):
                if expected_value != observed_value:
                    _append_finding(
                        findings,
                        _finding(
                            DriftKind.EVENT,
                            f"event_{field_name}_changed",
                            event.event_id,
                            expected_value,
                            observed_value,
                        ),
                    )
            missing_parents = tuple(
                parent
                for parent in event.causal_parent_ids
                if parent not in available_parents
            )
            for parent in missing_parents:
                _append_finding(
                    findings,
                    _finding(
                        DriftKind.CAUSAL,
                        "event_causal_parent_missing",
                        event.event_id,
                        parent,
                        "missing",
                    ),
                )
            available_parents.add(event.event_id)


def _compact_sequence(values: Sequence[int]) -> str:
    if not values:
        return "empty"
    if len(values) == 1:
        return str(values[0])
    if tuple(values) == tuple(range(values[0], values[-1] + 1)):
        return f"{values[0]}..{values[-1]}"
    return ",".join(str(item) for item in values)


def produce_drift_report(
    baseline: FederationDriftRoots,
    observed: FederationDriftRoots,
    *,
    events: Sequence[DomainEvent] = (),
    known_causal_parent_ids: Sequence[str] = (),
    observed_at: str,
) -> DriftReport:
    """Functional evidence producer for callers that need no monitor object."""

    return FederationDriftMonitor(baseline).observe(
        observed,
        events=events,
        known_causal_parent_ids=known_causal_parent_ids,
        observed_at=observed_at,
    )


def validate_current_drift_report(
    report: DriftReport,
    *,
    current_repository_tree_id: str,
    current_control_plane_generation: int,
    require_drift_free: bool = False,
) -> Mapping[str, Any]:
    """Validate exact-current-tree evidence without promoting it to authority."""

    if not isinstance(report, DriftReport):
        raise StaleDriftReportError("report must be a DriftReport")
    _identifier(current_repository_tree_id, "current_repository_tree_id")
    _integer(
        current_control_plane_generation,
        "current_control_plane_generation",
        minimum=1,
    )
    if report.observed.repository_tree_id != current_repository_tree_id:
        raise StaleDriftReportError("drift report is bound to a stale tree")
    if (
        report.observed.control_plane_generation
        != current_control_plane_generation
    ):
        raise StaleDriftReportError("drift report is bound to a stale generation")
    if require_drift_free and report.drifted:
        raise StaleDriftReportError("drift report contains promotion blockers")
    payload = report.to_dict()
    if payload["report_id"] != content_identity(report._identity_payload()):
        raise StaleDriftReportError("drift report identity is invalid")
    return MappingProxyType(
        {
            "schema": "casf/drift-report-validation@1",
            "report_id": report.report_id,
            "current_tree_bound": True,
            "current_generation_bound": True,
            "drift_free": report.current,
            "authority": False,
            "production_state_changed": False,
        }
    )


# Compatibility names for integrations that describe the same pure service by
# its evidence rather than its federation scope.
ArchitectureEventDriftMonitor = FederationDriftMonitor
DriftRootSet = FederationDriftRoots


__all__ = [
    "ArchitectureEventDriftMonitor",
    "DRIFT_FINDING_SCHEMA",
    "DRIFT_REPORT_SCHEMA",
    "DRIFT_ROOT_SET_SCHEMA",
    "DriftDimension",
    "DriftFinding",
    "DriftKind",
    "DriftMonitorError",
    "DriftReport",
    "DriftRootSet",
    "FEDERATION_DRIFT_MONITOR_INTERFACE",
    "FederationDriftMonitor",
    "FederationDriftRoots",
    "StaleDriftReportError",
    "closed_event_catalog_root",
    "produce_drift_report",
    "validate_current_drift_report",
]
