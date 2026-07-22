"""Fingerprint-independent codebase audits and exhaustion evidence.

Normal refill scans deliberately remember fingerprints they have already
materialized.  That optimization is not trustworthy evidence of exhaustion:
a saturated or damaged seen set can hide every candidate.  This module runs
the same analyzer with an empty seen set, compares complete finding catalogs,
and contributes a distinct audit channel to the exhaustion quorum.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .analyzer_health import AnalyzerHealthStatus, AnalyzerHealthThresholds, classify_analyzer_health
from .backlog_refinery import (
    CODEBASE_SCAN_ANALYZER_VERSION,
    CODEBASE_AUDIT_SCANNER_VERSION,
    CODEBASE_SCAN_SKIP_PREFIXES,
    CodebaseFinding,
    CodebaseScanInventory,
    codebase_exhaustion_configuration,
    codebase_source_tree_identity,
    run_codebase_analyzer_canaries,
    scan_codebase_findings,
)
from .dataset_store import DatasetAuditSnapshotArtifact, ObjectiveDatasetStore
from .scan_receipts import (
    DEFAULT_EXHAUSTION_QUORUM_SIZE,
    ExhaustionBinding,
    ExhaustionQuorumResult,
    RefillScanResult,
    RepositoryTreeIdentity,
    ScanMode,
    ScanTerminalReason,
    build_scan_result,
    canonical_revision,
    evaluate_exhaustion_quorum,
    objective_revision as canonical_objective_revision,
    scan_configuration_revision,
    scan_identity,
)


AUDIT_SCANNER_VERSION = CODEBASE_AUDIT_SCANNER_VERSION


class AuditFindingStatus(str, Enum):
    KNOWN = "known"
    STALE = "stale"
    CHANGED = "changed"
    NOVEL = "novel"


# Alternate wording retained for status/CLI consumers.
AuditFindingDisposition = AuditFindingStatus


def _finding_mapping(value: CodebaseFinding | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, CodebaseFinding):
        return value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError("audit findings must be CodebaseFinding or mapping values")
    fields = CodebaseFinding.__dataclass_fields__
    payload = {name: value.get(name) for name in fields}
    payload["line_number"] = int(payload.get("line_number") or 0)
    for name in fields:
        if name != "line_number":
            payload[name] = str(payload.get(name) or "")
    return payload


def _finding(value: CodebaseFinding | Mapping[str, Any]) -> CodebaseFinding:
    if isinstance(value, CodebaseFinding):
        return value
    return CodebaseFinding(**_finding_mapping(value))


def audit_finding_key(value: CodebaseFinding | Mapping[str, Any]) -> str:
    """Stable logical location used to distinguish changed from stale/novel."""

    finding = _finding_mapping(value)
    return canonical_revision(
        {
            "kind": finding["kind"],
            "path": Path(finding["root_relative_path"]).as_posix(),
            "line": finding["line_number"],
        },
        namespace="audit-finding-key",
    )


def audit_finding_content_revision(value: CodebaseFinding | Mapping[str, Any]) -> str:
    """Independent full-content identity; the normal fingerprint is ignored."""

    finding = _finding_mapping(value)
    finding.pop("fingerprint", None)
    return canonical_revision(finding, namespace="audit-finding-content")


def audit_snapshot_record(value: CodebaseFinding | Mapping[str, Any]) -> dict[str, Any]:
    payload = _finding_mapping(value)
    payload["audit_key"] = audit_finding_key(payload)
    payload["content_revision"] = audit_finding_content_revision(payload)
    return payload


@dataclass(frozen=True)
class AuditFindingRecord:
    status: AuditFindingStatus
    audit_key: str
    current: CodebaseFinding | None = None
    prior: CodebaseFinding | None = None
    current_content_revision: str = ""
    prior_content_revision: str = ""

    def __post_init__(self) -> None:
        status = self.status if isinstance(self.status, AuditFindingStatus) else AuditFindingStatus(str(self.status))
        object.__setattr__(self, "status", status)
        if self.current is not None and not isinstance(self.current, CodebaseFinding):
            object.__setattr__(self, "current", _finding(self.current))
        if self.prior is not None and not isinstance(self.prior, CodebaseFinding):
            object.__setattr__(self, "prior", _finding(self.prior))
        if not self.audit_key:
            basis = self.current or self.prior
            if basis is None:
                raise ValueError("audit record requires current or prior finding")
            object.__setattr__(self, "audit_key", audit_finding_key(basis))
        if self.current is not None and not self.current_content_revision:
            object.__setattr__(self, "current_content_revision", audit_finding_content_revision(self.current))
        if self.prior is not None and not self.prior_content_revision:
            object.__setattr__(self, "prior_content_revision", audit_finding_content_revision(self.prior))

    @property
    def finding(self) -> CodebaseFinding:
        finding = self.current or self.prior
        if finding is None:  # pragma: no cover - invariant
            raise RuntimeError("audit record has no finding")
        return finding

    @property
    def fingerprint(self) -> str:
        return self.finding.fingerprint

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "audit_key": self.audit_key,
            "current": self.current.to_dict() if self.current else None,
            "prior": self.prior.to_dict() if self.prior else None,
            "current_content_revision": self.current_content_revision,
            "prior_content_revision": self.prior_content_revision,
        }


AuditFindingChange = AuditFindingRecord


def classify_audit_findings(
    current_findings: Iterable[CodebaseFinding | Mapping[str, Any]],
    baseline_findings: Iterable[CodebaseFinding | Mapping[str, Any]],
) -> tuple[AuditFindingRecord, ...]:
    """Return a deterministic, disjoint known/stale/changed/novel partition."""

    current = [_finding(item) for item in current_findings]
    prior = [_finding(item) for item in baseline_findings]
    current.sort(key=lambda item: (item.root_relative_path, item.line_number, item.kind, audit_finding_content_revision(item)))
    prior.sort(key=lambda item: (item.root_relative_path, item.line_number, item.kind, audit_finding_content_revision(item)))
    current_by_key: dict[str, list[CodebaseFinding]] = {}
    prior_by_key: dict[str, list[CodebaseFinding]] = {}
    for finding in current:
        current_by_key.setdefault(audit_finding_key(finding), []).append(finding)
    for finding in prior:
        prior_by_key.setdefault(audit_finding_key(finding), []).append(finding)

    records: list[AuditFindingRecord] = []
    unmatched_current: list[CodebaseFinding] = []
    unmatched_prior: list[CodebaseFinding] = []
    for key in sorted(set(current_by_key) | set(prior_by_key)):
        current_group = list(current_by_key.get(key, ()))
        prior_group = list(prior_by_key.get(key, ()))
        while current_group and prior_group:
            current_item = current_group.pop(0)
            current_revision = audit_finding_content_revision(current_item)
            exact_index = next(
                (index for index, item in enumerate(prior_group) if audit_finding_content_revision(item) == current_revision),
                None,
            )
            if exact_index is not None:
                prior_item = prior_group.pop(exact_index)
                status = AuditFindingStatus.KNOWN
            else:
                prior_item = prior_group.pop(0)
                status = AuditFindingStatus.CHANGED
            records.append(AuditFindingRecord(status, key, current_item, prior_item))
        unmatched_current.extend(current_group)
        unmatched_prior.extend(prior_group)

    # A single finding moving within a file is a change. Ambiguous multi-item
    # groups remain stale+novel rather than being paired arbitrarily.
    def path_key(item: CodebaseFinding) -> tuple[str, str]:
        return item.kind, Path(item.root_relative_path).as_posix()

    current_groups: dict[tuple[str, str], list[CodebaseFinding]] = {}
    prior_groups: dict[tuple[str, str], list[CodebaseFinding]] = {}
    for item in unmatched_current:
        current_groups.setdefault(path_key(item), []).append(item)
    for item in unmatched_prior:
        prior_groups.setdefault(path_key(item), []).append(item)
    consumed_current: set[int] = set()
    consumed_prior: set[int] = set()
    for key in sorted(set(current_groups) & set(prior_groups)):
        if len(current_groups[key]) == len(prior_groups[key]) == 1:
            current_item = current_groups[key][0]
            prior_item = prior_groups[key][0]
            consumed_current.add(id(current_item))
            consumed_prior.add(id(prior_item))
            records.append(
                AuditFindingRecord(
                    AuditFindingStatus.CHANGED,
                    audit_finding_key(current_item),
                    current_item,
                    prior_item,
                )
            )
    for item in unmatched_current:
        if id(item) not in consumed_current:
            records.append(AuditFindingRecord(AuditFindingStatus.NOVEL, audit_finding_key(item), item, None))
    for item in unmatched_prior:
        if id(item) not in consumed_prior:
            records.append(AuditFindingRecord(AuditFindingStatus.STALE, audit_finding_key(item), None, item))
    return tuple(sorted(records, key=lambda item: (item.status.value, item.audit_key, item.fingerprint)))


@dataclass(frozen=True)
class AuditScanResult:
    receipt: RefillScanResult[Any]
    records: tuple[AuditFindingRecord, ...]
    inventory: CodebaseScanInventory
    binding: ExhaustionBinding
    quorum: ExhaustionQuorumResult
    snapshot_artifact: DatasetAuditSnapshotArtifact | None = None

    @property
    def known(self) -> tuple[AuditFindingRecord, ...]:
        return tuple(item for item in self.records if item.status is AuditFindingStatus.KNOWN)

    @property
    def stale(self) -> tuple[AuditFindingRecord, ...]:
        return tuple(item for item in self.records if item.status is AuditFindingStatus.STALE)

    @property
    def changed(self) -> tuple[AuditFindingRecord, ...]:
        return tuple(item for item in self.records if item.status is AuditFindingStatus.CHANGED)

    @property
    def novel(self) -> tuple[AuditFindingRecord, ...]:
        return tuple(item for item in self.records if item.status is AuditFindingStatus.NOVEL)

    @property
    def counts(self) -> dict[str, int]:
        return {status.value: sum(item.status is status for item in self.records) for status in AuditFindingStatus}

    known_count = property(lambda self: len(self.known))
    stale_count = property(lambda self: len(self.stale))
    changed_count = property(lambda self: len(self.changed))
    novel_count = property(lambda self: len(self.novel))
    findings = property(lambda self: self.records)

    @property
    def exhausted(self) -> bool:
        return self.receipt.terminal_reason is ScanTerminalReason.EXHAUSTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt": self.receipt.to_dict(),
            "binding": self.binding.to_dict(),
            "counts": self.counts,
            "known": [item.to_dict() for item in self.known],
            "stale": [item.to_dict() for item in self.stale],
            "changed": [item.to_dict() for item in self.changed],
            "novel": [item.to_dict() for item in self.novel],
            "inventory": self.inventory.details_dict(),
            "quorum": self.quorum.to_dict(),
            "snapshot_artifact": self.snapshot_artifact.to_dict() if self.snapshot_artifact else None,
        }


AuditScanReport = AuditScanResult


def _audit_scope_id(repository_id: str) -> str:
    return canonical_revision(
        {"repository_id": repository_id, "analyzer": CODEBASE_SCAN_ANALYZER_VERSION},
        namespace="codebase-audit-scope",
    )


def _configuration(
    *,
    skip_prefixes: Sequence[str],
    health_thresholds: AnalyzerHealthThresholds,
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return codebase_exhaustion_configuration(
        skip_prefixes=skip_prefixes,
        health_thresholds=health_thresholds,
        extra=extra,
    )


def run_audit_scan(
    repo_root: Path,
    *,
    dataset_store: ObjectiveDatasetStore | None = None,
    dataset_dir: Path | None = None,
    baseline_findings: Iterable[CodebaseFinding | Mapping[str, Any]] | None = None,
    known_findings: Iterable[CodebaseFinding | Mapping[str, Any]] | None = None,
    normal_seen_fingerprints: Iterable[str] = (),
    seen_fingerprints: Iterable[str] | None = None,
    strategy_path: Path | None = None,
    max_findings: int | None = None,
    skip_prefixes: Sequence[str] = CODEBASE_SCAN_SKIP_PREFIXES,
    health_thresholds: AnalyzerHealthThresholds | Mapping[str, Any] | None = None,
    configuration: Mapping[str, Any] | None = None,
    objective: Any = "",
    objective_path: Path | None = None,
    objective_revision: str = "",
    prior_receipts: Iterable[RefillScanResult[Any] | Mapping[str, Any]] = (),
    required_quorum: int = DEFAULT_EXHAUSTION_QUORUM_SIZE,
    quorum_size: int | None = None,
    persist: bool = True,
    analyzer_version: str = CODEBASE_SCAN_ANALYZER_VERSION,
) -> AuditScanResult:
    """Run an exhaustive audit without reading or mutating the normal seen set."""

    # Deliberately materialize no values from normal_seen_fingerprints.  This
    # argument documents and tests the trust boundary without even iterating a
    # caller-owned lazy/mutable collection.
    del normal_seen_fingerprints, seen_fingerprints, strategy_path, max_findings
    started_at = datetime.now(timezone.utc)
    root = Path(repo_root).resolve()
    generic_identity = scan_identity(root)
    store = dataset_store or (ObjectiveDatasetStore(dataset_dir) if dataset_dir is not None else None)
    scope_id = _audit_scope_id(generic_identity.repository_id)
    if baseline_findings is not None and known_findings is not None:
        raise ValueError("provide baseline_findings or known_findings, not both")
    explicit_baseline = baseline_findings if baseline_findings is not None else known_findings
    if explicit_baseline is None:
        baseline_rows = store.load_audit_snapshot(scope_id) if store is not None else []
    else:
        baseline_rows = list(explicit_baseline)

    if objective_path is not None and not objective_revision:
        objective = objective_path
    policy = AnalyzerHealthThresholds.from_value(health_thresholds)
    canaries = run_codebase_analyzer_canaries()
    try:
        inventory = scan_codebase_findings(
            root,
            max_findings=None,
            seen_fingerprints=(),
            exhaustive=True,
            skip_prefixes=skip_prefixes,
            return_inventory=True,
        )
        if not isinstance(inventory, CodebaseScanInventory):  # pragma: no cover
            raise TypeError("instrumented audit did not return inventory")
    except TimeoutError as exc:
        empty = CodebaseScanInventory(complete=False)
        binding = ExhaustionBinding(
            generic_identity.repository_id,
            generic_identity.tree_id,
            analyzer_version,
            scan_configuration_revision(configuration or {}),
            objective_revision or canonical_objective_revision(objective),
        )
        receipt = build_scan_result(
            ScanTerminalReason.TIMED_OUT, ScanMode.AUDIT, analyzer_version, root, started_at,
            error=str(exc) or type(exc).__name__, identity=generic_identity,
        )
        quorum = evaluate_exhaustion_quorum([*prior_receipts, receipt], binding=binding, required_members=quorum_size or required_quorum)
        return AuditScanResult(receipt, (), empty, binding, quorum)
    except Exception as exc:
        empty = CodebaseScanInventory(complete=False)
        binding = ExhaustionBinding(
            generic_identity.repository_id,
            generic_identity.tree_id,
            analyzer_version,
            scan_configuration_revision(configuration or {}),
            objective_revision or canonical_objective_revision(objective),
        )
        receipt = build_scan_result(
            ScanTerminalReason.FAILED, ScanMode.AUDIT, analyzer_version, root, started_at,
            error=f"{type(exc).__name__}: {exc}", identity=generic_identity,
        )
        quorum = evaluate_exhaustion_quorum([*prior_receipts, receipt], binding=binding, required_members=quorum_size or required_quorum)
        return AuditScanResult(receipt, (), empty, binding, quorum)

    records = classify_audit_findings(inventory.findings, baseline_rows)
    counts = {status.value: sum(item.status is status for item in records) for status in AuditFindingStatus}
    health_inventory = inventory.health_inventory_dict(appended_tasks=0)
    # Audited observations are intentionally not materialized. Account for
    # every raw unique finding as policy-rejected for health-funnel purposes.
    health_inventory["seen_candidates"] = 0
    health_inventory["rejected_candidates"] = max(
        0, inventory.raw_candidate_count - inventory.deduplicated_candidate_count
    )
    health = classify_analyzer_health(
        health_inventory,
        canaries=canaries,
        thresholds=policy,
    )
    source_tree_id = codebase_source_tree_identity(root, inventory)
    config_material = _configuration(
        skip_prefixes=skip_prefixes,
        health_thresholds=policy,
        extra=configuration,
    )
    binding = ExhaustionBinding(
        repository_id=generic_identity.repository_id,
        tree_id=source_tree_id,
        analyzer_version=analyzer_version,
        configuration_revision=scan_configuration_revision(config_material),
        objective_revision=objective_revision or canonical_objective_revision(objective),
    )
    actionable = counts[AuditFindingStatus.NOVEL.value] + counts[AuditFindingStatus.CHANGED.value]
    if health.status is AnalyzerHealthStatus.UNHEALTHY:
        terminal_reason = ScanTerminalReason.FAILED
        error = "unhealthy audit scan: " + ", ".join(health.reasons)
    elif health.status is AnalyzerHealthStatus.PARTIAL or not inventory.complete:
        terminal_reason = ScanTerminalReason.PARTIAL
        error = None
    elif actionable:
        terminal_reason = ScanTerminalReason.PARTIAL
        error = None
    else:
        terminal_reason = ScanTerminalReason.EXHAUSTED
        error = None

    metadata = {
        "audit_scanner_version": AUDIT_SCANNER_VERSION,
        "audit_summary": counts,
        "audit_baseline_count": len(baseline_rows),
        "audit_current_count": len(inventory.findings),
        "audit_scope_id": scope_id,
        "analyzer_health": health.to_dict(),
        "analyzer_canaries": canaries.to_dict(),
        "health_thresholds": policy.to_dict(),
        "audit_coverage": inventory.coverage_dict(),
        "coverage_complete": inventory.complete,
        "exhaustive": True,
        "evidence_channel": "codebase:audit",
        "configuration_revision": binding.configuration_revision,
        "objective_revision": binding.objective_revision,
        "exhaustion_binding": binding.to_dict(),
    }
    captured = RepositoryTreeIdentity(binding.repository_id, binding.tree_id)
    receipt = build_scan_result(
        terminal_reason,
        ScanMode.AUDIT,
        analyzer_version,
        root,
        started_at,
        safe_for_completion_reasoning=False,
        error=error,
        metadata=metadata,
        identity=captured,
    )

    previous_members: list[Mapping[str, Any]] = []
    if store is not None:
        stored = store.load_exhaustion_quorum(binding.repository_id)
        previous_members.extend(stored.get("members", ()) if isinstance(stored, Mapping) else ())
    evidence = [*previous_members, *prior_receipts, receipt]
    required = quorum_size or required_quorum
    quorum = evaluate_exhaustion_quorum(evidence, binding=binding, required_members=required)
    if terminal_reason is ScanTerminalReason.EXHAUSTED and quorum.satisfied:
        metadata = {**metadata, "exhaustion_quorum": quorum.to_dict()}
        receipt = build_scan_result(
            terminal_reason,
            ScanMode.AUDIT,
            analyzer_version,
            root,
            started_at,
            finished_at=receipt.finished_at,
            safe_for_completion_reasoning=True,
            metadata=metadata,
            identity=captured,
        )
        quorum = evaluate_exhaustion_quorum(
            [*previous_members, *prior_receipts, receipt],
            binding=binding,
            required_members=required,
        )

    snapshot_artifact: DatasetAuditSnapshotArtifact | None = None
    if store is not None and persist:
        snapshot_artifact = store.persist_audit_snapshot(
            scope_id=scope_id,
            findings=(audit_snapshot_record(item) for item in inventory.findings),
            metadata={
                "binding": binding.to_dict(),
                "counts": counts,
                "coverage": inventory.coverage_dict(),
                "health": health.to_dict(),
            },
        )
        store.persist_exhaustion_quorum(quorum)
    return AuditScanResult(receipt, records, inventory, binding, quorum, snapshot_artifact)


audit_codebase_findings = run_audit_scan
scan_codebase_audit = run_audit_scan


__all__ = [
    "AUDIT_SCANNER_VERSION",
    "AuditFindingChange",
    "AuditFindingDisposition",
    "AuditFindingRecord",
    "AuditFindingStatus",
    "AuditScanReport",
    "AuditScanResult",
    "audit_codebase_findings",
    "audit_finding_content_revision",
    "audit_finding_key",
    "audit_snapshot_record",
    "classify_audit_findings",
    "run_audit_scan",
    "scan_codebase_audit",
]
