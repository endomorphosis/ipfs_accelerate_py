"""Fingerprint-independent codebase audits and exhaustion evidence.

Normal refill scans deliberately remember fingerprints they have already
materialized.  That optimization is not trustworthy evidence of exhaustion:
a saturated or damaged seen set can hide every candidate.  This module runs
the same analyzer with an empty seen set, compares complete finding catalogs,
and contributes a distinct audit channel to the exhaustion quorum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .analyzer_health import (
    ANALYSIS_ESCALATION_SCHEMA,
    AnalysisEscalationPolicy,
    AnalysisEscalationRecord,
    AnalysisEscalationStage,
    AnalysisEscalationStatus,
    AnalyzerHealthStatus,
    AnalyzerHealthThresholds,
    classify_analyzer_health,
)
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
    DEFAULT_SCAN_FRESHNESS_SECONDS,
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
AST_COVERAGE_ANALYZER_VERSION = "objective-ast-coverage/v1"


class AuditFindingStatus(str, Enum):
    KNOWN = "known"
    STALE = "stale"
    CHANGED = "changed"
    NOVEL = "novel"


# Alternate wording retained for status/CLI consumers.
AuditFindingDisposition = AuditFindingStatus


@dataclass(frozen=True)
class AstCoverageReport:
    """Complete AST/symbol coverage evidence for one repository snapshot."""

    analyzer_version: str
    records: tuple[Mapping[str, Any], ...]
    objective_terms: tuple[str, ...]
    term_evidence: Mapping[str, tuple[str, ...]]
    expected_file_count: int
    scanned_file_count: int
    parsed_file_count: int
    reused_file_count: int
    parse_failures: tuple[Mapping[str, str], ...]
    source_bytes: int
    ast_truncated_count: int
    complete: bool
    limit_reason: str = ""
    elapsed_seconds: float = 0.0

    @property
    def healthy(self) -> bool:
        return self.complete and not self.parse_failures

    @property
    def covered_terms(self) -> tuple[str, ...]:
        return tuple(term for term in self.objective_terms if self.term_evidence.get(term))

    @property
    def uncovered_terms(self) -> tuple[str, ...]:
        return tuple(term for term in self.objective_terms if not self.term_evidence.get(term))

    @property
    def confidence(self) -> float:
        if not self.complete or self.expected_file_count < 1:
            return 0.0
        return max(
            0.0,
            min(1.0, (self.scanned_file_count - len(self.parse_failures)) / self.expected_file_count),
        )

    @property
    def novelty(self) -> float:
        if not self.objective_terms:
            return 0.0
        return len(self.uncovered_terms) / len(self.objective_terms)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "analyzer_version": self.analyzer_version,
            "healthy": self.healthy,
            "complete": self.complete,
            "expected_file_count": self.expected_file_count,
            "scanned_file_count": self.scanned_file_count,
            "parsed_file_count": self.parsed_file_count,
            "reused_file_count": self.reused_file_count,
            "parse_failure_count": len(self.parse_failures),
            "source_bytes": self.source_bytes,
            "ast_truncated_count": self.ast_truncated_count,
            "covered_terms": list(self.covered_terms),
            "uncovered_terms": list(self.uncovered_terms),
            "confidence": self.confidence,
            "novelty": self.novelty,
            "limit_reason": self.limit_reason,
            "elapsed_seconds": self.elapsed_seconds,
        }

    def to_dict(self, *, include_records: bool = False) -> dict[str, Any]:
        payload = self.summary_dict()
        payload["objective_terms"] = list(self.objective_terms)
        payload["term_evidence"] = {
            key: list(value) for key, value in self.term_evidence.items()
        }
        payload["parse_failures"] = [dict(item) for item in self.parse_failures]
        if include_records:
            payload["records"] = [dict(item) for item in self.records]
        return payload


def run_exhaustive_ast_coverage(
    repo_root: Path,
    *,
    objective_path: Path | None = None,
    objective_terms: Sequence[str] = (),
    previous_records: Sequence[Mapping[str, Any]] = (),
    max_records: int = 50000,
    max_source_bytes: int = 67108864,
    max_ast_chars: int = 1000000,
    excluded_roots: Iterable[Path] = (),
) -> AstCoverageReport:
    """Run the repository's real AST dataset scanner with explicit coverage limits."""

    from .objective_graph import (
        collect_ast_dataset_records,
        evidence_index,
        objective_candidate_files,
        repo_relative_path,
    )

    record_limit = int(max_records)
    byte_limit = int(max_source_bytes)
    if record_limit < 1 or byte_limit < 1:
        raise ValueError("AST record and source-byte limits must be at least 1")
    root = Path(repo_root).resolve()
    objective = Path(objective_path or (root / ".agent-supervisor-objective")).resolve()
    terms = tuple(dict.fromkeys(str(item).strip() for item in objective_terms if str(item).strip()))
    expected_paths = objective_candidate_files(root, objective_path=objective)
    stats: dict[str, Any] = {}
    rows = collect_ast_dataset_records(
        root,
        objective_path=objective,
        max_ast_chars=max(1, int(max_ast_chars)),
        previous_records=previous_records,
        scan_stats=stats,
        excluded_roots=excluded_roots,
    )
    retained: list[Mapping[str, Any]] = []
    retained_bytes = 0
    limit_reason = ""
    for row in rows:
        row_bytes = max(0, int(row.get("source_bytes") or 0))
        if len(retained) >= record_limit:
            limit_reason = "ast_record_limit_reached"
            break
        if retained_bytes + row_bytes > byte_limit:
            limit_reason = "ast_source_byte_limit_reached"
            break
        retained.append(dict(row))
        retained_bytes += row_bytes
    retained_paths = {str(row.get("root_relative_path") or "") for row in retained}
    expected_relative = {
        repo_relative_path(root, path)
        for path in expected_paths
        if path.is_file()
    }
    complete = not limit_reason and retained_paths == expected_relative
    failures = tuple(
        {
            "path": str(row.get("root_relative_path") or ""),
            "error": str(row.get("parse_error") or ""),
        }
        for row in retained
        if str(row.get("parse_error") or "")
    )
    evidence = evidence_index(
        root,
        objective_path=objective,
        terms=terms,
        records=retained,
    )
    return AstCoverageReport(
        analyzer_version=AST_COVERAGE_ANALYZER_VERSION,
        records=tuple(retained),
        objective_terms=terms,
        term_evidence={key: tuple(value) for key, value in evidence.items()},
        expected_file_count=len(expected_relative),
        scanned_file_count=len(retained_paths),
        parsed_file_count=int(stats.get("parsed_record_count") or 0),
        reused_file_count=int(stats.get("reused_record_count") or 0),
        parse_failures=failures,
        source_bytes=retained_bytes,
        ast_truncated_count=sum(bool(row.get("ast_truncated")) for row in retained),
        complete=complete,
        limit_reason=limit_reason or ("ast_inventory_incomplete" if not complete else ""),
        elapsed_seconds=max(0.0, float(stats.get("scan_elapsed_seconds") or 0.0)),
    )


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

    def completion_gate_evidence(
        self,
        *,
        evaluated_at: datetime | str | None = None,
        max_age_seconds: float = DEFAULT_SCAN_FRESHNESS_SECONDS,
    ) -> dict[str, Any]:
        """Return the exact, fail-closed audit proof consumed by completion.

        A receipt's safety bit is necessary but deliberately insufficient.
        This projection rechecks the terminal state, analyzer health, complete
        exhaustive coverage, actionable findings, binding, quorum, and the age
        of both the receipt and every quorum member.  It never infers success
        from an empty or duplicate-only result.
        """

        if max_age_seconds < 0:
            raise ValueError("max_age_seconds must be non-negative")

        def utc(value: datetime | str | None) -> datetime | None:
            if isinstance(value, datetime):
                parsed = value
            else:
                text = str(value or "").strip()
                if not text:
                    return None
                if text.endswith("Z"):
                    text = f"{text[:-1]}+00:00"
                try:
                    parsed = datetime.fromisoformat(text)
                except ValueError:
                    return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)

        instant = utc(evaluated_at) or datetime.now(timezone.utc)
        receipt = self.receipt.to_dict()
        metadata = receipt.get("metadata") if isinstance(receipt.get("metadata"), Mapping) else {}
        health = dict(metadata.get("analyzer_health") or {})
        quorum = self.quorum.to_dict()
        binding = self.binding.to_dict()
        terminal_reason = str(receipt.get("terminal_reason") or "").strip().lower()
        health_status = str(health.get("status") or "").strip().lower()
        coverage_complete = metadata.get("coverage_complete") is True
        exhaustive = metadata.get("exhaustive") is True
        summary = metadata.get("audit_summary")
        summary = summary if isinstance(summary, Mapping) else {}
        actionable_count = int(summary.get("novel", 0) or 0) + int(summary.get("changed", 0) or 0)

        finished_at = utc(receipt.get("finished_at"))
        receipt_age = (instant - finished_at).total_seconds() if finished_at is not None else None
        receipt_fresh = bool(
            finished_at is not None
            and receipt_age is not None
            and -1.0 <= receipt_age <= float(max_age_seconds)
        )
        stale_member_ids: list[str] = []
        for member in quorum.get("members", ()):
            if not isinstance(member, Mapping):
                stale_member_ids.append("[invalid-member]")
                continue
            member_finished = utc(member.get("finished_at"))
            member_age = (
                (instant - member_finished).total_seconds()
                if member_finished is not None
                else None
            )
            if member_age is None or member_age < -1.0 or member_age > float(max_age_seconds):
                stale_member_ids.append(str(member.get("member_id") or "[missing-member-id]"))
        quorum_fresh = bool(quorum.get("members")) and not stale_member_ids

        reason_codes: list[str] = []
        if terminal_reason != ScanTerminalReason.EXHAUSTED.value:
            reason_codes.append(f"analysis_{terminal_reason or 'terminal_state_missing'}")
        if self.receipt.safe_for_completion_reasoning is not True:
            reason_codes.append("analysis_not_completion_safe")
        if health_status != AnalyzerHealthStatus.HEALTHY.value:
            reason_codes.append("analyzer_health_missing" if not health_status else "analyzer_unhealthy")
        if not coverage_complete:
            reason_codes.append("coverage_incomplete")
        if not exhaustive:
            reason_codes.append("analysis_not_exhaustive")
        if actionable_count:
            reason_codes.append("actionable_audit_findings")
        if self.quorum.satisfied is not True:
            reason_codes.append("exhaustion_quorum_unsatisfied")
        if self.quorum.binding != self.binding:
            reason_codes.append("exhaustion_quorum_binding_mismatch")
        if not receipt_fresh:
            reason_codes.append("audit_evidence_stale")
        if not quorum_fresh:
            reason_codes.append("exhaustion_quorum_evidence_stale")
        if receipt.get("error"):
            reason_codes.append("analysis_error")
        escalation = metadata.get("analysis_escalation")
        if isinstance(escalation, Mapping) and escalation.get("exhaustion_eligible") is not True:
            reason_codes.append("analysis_escalation_ineligible")
        reason_codes = list(dict.fromkeys(reason_codes))
        passed = not reason_codes

        # The health safety flag describes this exact health observation in
        # context; it is not a replacement for the receipt or quorum verdict.
        health["safe_for_completion_reasoning"] = bool(
            health_status == AnalyzerHealthStatus.HEALTHY.value
            and coverage_complete
            and exhaustive
        )
        freshness = {
            "evaluated_at": instant.isoformat(),
            "max_age_seconds": float(max_age_seconds),
            "receipt_finished_at": finished_at.isoformat() if finished_at else "",
            "receipt_age_seconds": receipt_age,
            "receipt_fresh": receipt_fresh,
            "quorum_fresh": quorum_fresh,
            "stale_member_ids": stale_member_ids,
        }
        evaluated_evidence = {
            "receipt": receipt,
            "analyzer_health": health,
            "exhaustion_quorum": quorum,
            "binding": binding,
            "freshness": freshness,
            "audit_records": [item.to_dict() for item in self.records],
            "inventory": self.inventory.details_dict(),
        }
        return {
            "schema": "ipfs_accelerate_py.agent_supervisor.audit_completion_gate.v1",
            "passed": passed,
            "pass_reasons": ["healthy_fresh_exhaustive_quorum_satisfied"] if passed else [],
            "fail_reasons": [
                {"code": code, "message": _audit_gate_reason(code)} for code in reason_codes
            ],
            "reason_codes": reason_codes,
            "terminal_reason": terminal_reason,
            "status": terminal_reason,
            "receipt": receipt,
            "analysis_result": receipt,
            "analyzer_health": health,
            "exhaustion_quorum": quorum,
            "binding": binding,
            "repository_id": binding.get("repository_id", ""),
            "repository_tree": binding.get("tree_id", ""),
            "coverage_complete": coverage_complete,
            "exhaustive": exhaustive,
            "actionable_finding_count": actionable_count,
            "freshness": freshness,
            "safe_for_completion_reasoning": passed,
            "evaluated_evidence": evaluated_evidence,
        }


AuditScanReport = AuditScanResult


_AUDIT_GATE_REASON_MESSAGES: Mapping[str, str] = {
    "analysis_not_completion_safe": "The receipt was not explicitly marked safe for completion reasoning.",
    "analyzer_health_missing": "The receipt does not contain an analyzer health verdict.",
    "analyzer_unhealthy": "The analyzer health verdict is not healthy.",
    "coverage_incomplete": "The analyzer did not cover its complete configured inventory.",
    "analysis_not_exhaustive": "The scan was not exhaustive.",
    "actionable_audit_findings": "The audit contains novel or changed actionable findings.",
    "exhaustion_quorum_unsatisfied": "The configured independent exhaustion quorum is not satisfied.",
    "exhaustion_quorum_binding_mismatch": "The quorum is not bound to this audit repository snapshot.",
    "audit_evidence_stale": "The audit receipt is missing a current completion timestamp or is stale.",
    "exhaustion_quorum_evidence_stale": "At least one required quorum member is missing a current timestamp or is stale.",
    "analysis_error": "The audit receipt contains an analysis error.",
    "analysis_escalation_ineligible": "The bounded escalation result is inconclusive or ineligible for exhaustion proof.",
}


def _audit_gate_reason(code: str) -> str:
    if code.startswith("analysis_") and code not in _AUDIT_GATE_REASON_MESSAGES:
        return f"The audit terminal state {code.removeprefix('analysis_')!r} cannot prove completion."
    return _AUDIT_GATE_REASON_MESSAGES.get(code, "The audit evidence cannot prove completion.")


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


@dataclass(frozen=True)
class AnalysisEscalationResult:
    """Finite low-backlog analysis outcome; never implicit completion proof."""

    status: AnalysisEscalationStatus
    backlog_before: int
    backlog_after: int
    backlog_target: int
    records: tuple[AnalysisEscalationRecord, ...]
    proposals: tuple[Mapping[str, Any], ...] = ()
    analysis_inconclusive: bool = False
    deterministic_fallback: bool = False
    reason: str = ""
    pipeline_result: Mapping[str, Any] = field(default_factory=dict)

    @property
    def backlog_satisfied(self) -> bool:
        return self.backlog_after >= self.backlog_target

    @property
    def exhausted(self) -> bool:
        # Generating work means the repository is not exhausted. Failing to
        # generate work is inconclusive and must be handled by a completion
        # gate with independent quorum evidence.
        return False

    @property
    def safe_for_completion_reasoning(self) -> bool:
        return False

    @property
    def exhaustion_eligible(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_ESCALATION_SCHEMA,
            "status": self.status.value,
            "backlog_before": self.backlog_before,
            "backlog_after": self.backlog_after,
            "backlog_target": self.backlog_target,
            "backlog_satisfied": self.backlog_satisfied,
            "records": [item.to_dict() for item in self.records],
            "proposals": [dict(item) for item in self.proposals],
            "analysis_inconclusive": self.analysis_inconclusive,
            "deterministic_fallback": self.deterministic_fallback,
            "reason": self.reason,
            "analysis_pipeline": dict(self.pipeline_result),
            "exhausted": False,
            "safe_for_completion_reasoning": False,
            "exhaustion_eligible": False,
        }


def _analysis_candidate_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return {str(key): item for key, item in payload.items()}
    return {"candidate": str(value)}


def _stage_values(value: Any) -> tuple[list[Any], bool, float, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Normalize injectable scanner results used by the policy and tests."""

    if isinstance(value, Mapping):
        raw_candidates = value.get("candidates", value.get("findings", ())) or ()
        candidates = list(raw_candidates) if not isinstance(raw_candidates, (str, bytes)) else []
        healthy = bool(value.get("healthy", value.get("complete", True)))
        confidence = float(value.get("confidence", 1.0 if healthy else 0.0) or 0.0)
        scope = dict(value.get("scope") or {})
        cost = dict(value.get("cost") or {})
        rejected = [
            _analysis_candidate_dict(item)
            for item in (value.get("rejected_candidates") or ())
        ]
        return candidates, healthy, confidence, scope, cost, rejected
    if isinstance(value, CodebaseScanInventory):
        canaries = run_codebase_analyzer_canaries()
        health_inventory = value.health_inventory_dict(appended_tasks=len(value.findings))
        health = classify_analyzer_health(health_inventory, canaries=canaries)
        rejected = [
            {"reason": "previously_seen", "count": value.seen_candidate_count},
            {"reason": "deduplicated", "count": value.deduplicated_candidate_count},
            {"reason": "scan_limit", "count": value.rejected_candidate_count},
        ]
        rejected = [item for item in rejected if item["count"]]
        confidence = 1.0 if health.status is AnalyzerHealthStatus.HEALTHY else (
            0.5 if health.status is AnalyzerHealthStatus.PARTIAL else 0.0
        )
        return (
            list(value.findings),
            health.status is not AnalyzerHealthStatus.UNHEALTHY,
            confidence,
            value.coverage_dict(),
            {"files_parsed": value.parsed_file_count, "candidates_examined": value.raw_candidate_count},
            rejected,
        )
    if isinstance(value, AstCoverageReport):
        return (
            [],
            value.healthy,
            value.confidence,
            value.summary_dict(),
            {
                "files_scanned": value.scanned_file_count,
                "source_bytes": value.source_bytes,
                "elapsed_seconds": value.elapsed_seconds,
            },
            [dict(item) for item in value.parse_failures],
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value), True, 1.0, {"candidate_count": len(value)}, {"candidate_count": len(value)}, []
    raise TypeError("analysis scanner must return a sequence, mapping, or coverage inventory")


def run_low_backlog_analysis(
    repo_root: Path,
    *,
    healthy_backlog_count: int,
    objective_terms: Sequence[str],
    objective_path: Path | None = None,
    policy: AnalysisEscalationPolicy | Mapping[str, Any] | None = None,
    seen_fingerprints: Iterable[str] = (),
    known_proposal_ids: Iterable[str] = (),
    router_calls_in_window: int | Iterable[Any] = 0,
    incremental_scanner: Callable[[], Any] | None = None,
    ast_scanner: Callable[[], Any] | None = None,
    router: Callable[[str], str] | None = None,
    router_config: Any = None,
    fallback_planner: Callable[[object, int], Sequence[Any]] | None = None,
    analysis_pipeline: Any = None,
    analysis_pipeline_request: Any = None,
) -> AnalysisEscalationResult:
    """Escalate static -> exhaustive AST -> bounded LLM planning.

    Scanner callbacks are intentionally nullary: callers can bind repository
    state in closures, and tests can prove ordering without filesystem setup.
    This function only returns evidence and candidates; taskboard mutation is
    owned by the objective daemon.
    """

    from .task_proposal_router import (
        StructuredPlanRouterConfig,
        generate_analysis_proposals,
    )

    limits = AnalysisEscalationPolicy.from_value(policy)
    before = max(0, int(healthy_backlog_count))
    projected = before
    terms = tuple(dict.fromkeys(str(item).strip() for item in objective_terms if str(item).strip()))
    effective_terms = terms or ("unresolved objective",)
    records: list[AnalysisEscalationRecord] = []
    proposals: list[dict[str, Any]] = []
    pipeline_payload: dict[str, Any] = {}
    ast_value: object | None = None
    if projected >= limits.backlog_target:
        return AnalysisEscalationResult(
            AnalysisEscalationStatus.SATISFIED,
            before,
            projected,
            limits.backlog_target,
            (),
            reason="healthy backlog already meets target",
        )

    if incremental_scanner is None:
        incremental_scanner = lambda: scan_codebase_findings(
            Path(repo_root).resolve(),
            max_findings=limits.max_incremental_candidates,
            seen_fingerprints=seen_fingerprints,
            exhaustive=False,
            return_inventory=True,
        )
    try:
        static_value = incremental_scanner()
        candidates, healthy, confidence, scope, cost, rejected = _stage_values(static_value)
        remaining = max(0, limits.backlog_target - projected)
        accepted = candidates[:remaining]
        overflow = candidates[remaining:]
        rejected.extend(
            {**_analysis_candidate_dict(item), "reason": "backlog_target_reached"}
            for item in overflow
        )
        accepted_payloads = [_analysis_candidate_dict(item) for item in accepted]
        proposals.extend(accepted_payloads)
        projected += len(accepted_payloads)
        novelty = len(accepted_payloads) / max(1, len(candidates) + len(rejected))
        records.append(
            AnalysisEscalationRecord(
                AnalysisEscalationStage.INCREMENTAL_STATIC,
                AnalysisEscalationStatus.COMPLETED if healthy else AnalysisEscalationStatus.LIMITED,
                cost=cost,
                scope=scope,
                novelty=novelty,
                confidence=max(0.0, min(1.0, confidence)),
                rejected_candidates=tuple(rejected[: limits.max_rejected_candidates]),
                objective_terms=effective_terms,
                accepted_candidates=tuple(accepted_payloads),
                reason="" if healthy else "incremental scan was partial; escalating to AST coverage",
            )
        )
    except Exception as exc:
        records.append(
            AnalysisEscalationRecord(
                AnalysisEscalationStage.INCREMENTAL_STATIC,
                AnalysisEscalationStatus.FAILED,
                objective_terms=effective_terms,
                reason=f"{type(exc).__name__}: {exc}",
            )
        )

    if projected >= limits.backlog_target:
        return AnalysisEscalationResult(
            AnalysisEscalationStatus.SATISFIED,
            before,
            projected,
            limits.backlog_target,
            tuple(records),
            tuple(proposals),
            reason="incremental static analysis restored the backlog target",
        )

    if ast_scanner is None:
        ast_scanner = lambda: run_exhaustive_ast_coverage(
            Path(repo_root).resolve(),
            objective_path=objective_path,
            objective_terms=effective_terms,
            max_records=limits.max_ast_records,
            max_source_bytes=limits.max_ast_bytes,
        )
    ast_summary: dict[str, Any] = {}
    ast_healthy = False
    try:
        ast_value = ast_scanner()
        candidates, ast_healthy, confidence, scope, cost, rejected = _stage_values(ast_value)
        if isinstance(ast_value, AstCoverageReport):
            ast_summary = ast_value.summary_dict()
        else:
            ast_summary = dict(scope)
        remaining = max(0, limits.backlog_target - projected)
        accepted = candidates[:remaining]
        overflow = candidates[remaining:]
        rejected.extend(
            {**_analysis_candidate_dict(item), "reason": "backlog_target_reached"}
            for item in overflow
        )
        accepted_payloads = [_analysis_candidate_dict(item) for item in accepted]
        proposals.extend(accepted_payloads)
        projected += len(accepted_payloads)
        records.append(
            AnalysisEscalationRecord(
                AnalysisEscalationStage.EXHAUSTIVE_AST,
                AnalysisEscalationStatus.COMPLETED if ast_healthy else AnalysisEscalationStatus.LIMITED,
                cost=cost,
                scope=scope,
                novelty=len(accepted_payloads) / max(1, len(candidates) + len(rejected)),
                confidence=max(0.0, min(1.0, confidence)),
                rejected_candidates=tuple(rejected[: limits.max_rejected_candidates]),
                objective_terms=effective_terms,
                accepted_candidates=tuple(accepted_payloads),
                reason="" if ast_healthy else "AST coverage was incomplete or unhealthy",
            )
        )
    except Exception as exc:
        records.append(
            AnalysisEscalationRecord(
                AnalysisEscalationStage.EXHAUSTIVE_AST,
                AnalysisEscalationStatus.FAILED,
                objective_terms=effective_terms,
                reason=f"{type(exc).__name__}: {exc}",
            )
        )

    if analysis_pipeline is not None:
        try:
            from .analysis_pipeline import AnalysisPipelineRequest
            from .conflict_graph import build_python_ast_blob_record

            if analysis_pipeline_request is None:
                identity = scan_identity(Path(repo_root).resolve())
                ast_records: list[tuple[str, Any]] = []
                retrieval_records: list[dict[str, Any]] = []
                if isinstance(ast_value, AstCoverageReport):
                    for row in ast_value.records:
                        path = str(row.get("root_relative_path") or "")
                        compact = {
                            str(key): value
                            for key, value in row.items()
                            if key not in {"evidence_text", "ast_text"}
                        }
                        compact.setdefault("record_id", path)
                        compact.setdefault("path", path)
                        retrieval_records.append(compact)
                        if str(row.get("suffix") or "").lower() != ".py":
                            continue
                        source = row.get("evidence_text")
                        if not isinstance(source, str):
                            continue
                        ast_records.append(
                            (
                                path,
                                build_python_ast_blob_record(
                                    source,
                                    blob_identity=str(
                                        row.get("blob_hash")
                                        or row.get("source_sha1")
                                        or ""
                                    ),
                                ),
                            )
                        )
                analysis_pipeline_request = AnalysisPipelineRequest(
                    repository_id=identity.repository_id,
                    tree_id=canonical_revision(
                        {
                            "repository_id": identity.repository_id,
                            "objective_revision": canonical_objective_revision(
                                objective_path or effective_terms
                            ),
                            "records": [
                                {
                                    "path": str(
                                        row.get("root_relative_path") or ""
                                    ),
                                    "blob": str(
                                        row.get("blob_hash")
                                        or row.get("source_sha1")
                                        or ""
                                    ),
                                }
                                for row in (
                                    ast_value.records
                                    if isinstance(ast_value, AstCoverageReport)
                                    else ()
                                )
                            ],
                        },
                        namespace="low-backlog-analysis-tree",
                    ),
                    objective_revision=canonical_objective_revision(
                        objective_path or effective_terms
                    ),
                    query={
                        "text": " ".join(effective_terms),
                        "objective_terms": list(effective_terms),
                    },
                    analyzer_id="objective.low_backlog_analysis",
                    analyzer_version=AST_COVERAGE_ANALYZER_VERSION,
                    configuration={
                        "escalation_policy": limits.to_dict(),
                        "objective_path": str(objective_path or ""),
                    },
                    policy={"purpose": "low_backlog_proposal_analysis"},
                    ast_records=tuple(ast_records),
                    retrieval_inputs={"records": tuple(retrieval_records)},
                )
            pipeline_value = analysis_pipeline.analyze(
                analysis_pipeline_request
            )
            converter = getattr(pipeline_value, "to_dict", None)
            pipeline_projection = (
                converter() if callable(converter) else pipeline_value
            )
            if not isinstance(pipeline_projection, Mapping):
                raise TypeError("analysis pipeline returned no result mapping")
            # Pipeline output is ranked proposal context only.  Even a
            # conclusive local packet cannot become exhaustion/completion
            # evidence through this escalation channel.
            pipeline_payload = {
                "result_id": pipeline_projection.get("result_id", ""),
                "cache_status": pipeline_projection.get("cache_status", ""),
                "cache_lookup_status": pipeline_projection.get(
                    "cache_lookup_status", ""
                ),
                "cache_reason_codes": list(
                    pipeline_projection.get("cache_reason_codes") or ()
                ),
                "retrieval_response_id": pipeline_projection.get(
                    "retrieval_response_id", ""
                ),
                "ranked_evidence_references": list(
                    pipeline_projection.get("ranked_evidence_references") or ()
                ),
                "retrieval_backend_health": dict(
                    pipeline_projection.get("retrieval_backend_health") or {}
                ),
                "retrieval_truncation": dict(
                    pipeline_projection.get("retrieval_truncation") or {}
                ),
                "safe_for_completion_reasoning": False,
                "nomination_only": True,
            }
            ast_summary["analysis_pipeline"] = dict(pipeline_payload)
        except Exception as exc:
            pipeline_payload = {
                "status": "failed",
                "reason_code": "integrated_analysis_pipeline_failed",
                "error_type": type(exc).__name__,
                "detail": str(exc)[:500],
                "safe_for_completion_reasoning": False,
                "nomination_only": True,
            }
            ast_summary["analysis_pipeline"] = dict(pipeline_payload)

    if projected >= limits.backlog_target and ast_healthy:
        return AnalysisEscalationResult(
            AnalysisEscalationStatus.SATISFIED,
            before,
            projected,
            limits.backlog_target,
            tuple(records),
            tuple(proposals),
            reason="exhaustive AST analysis restored the backlog target",
            pipeline_result=pipeline_payload,
        )

    shortage = max(1, limits.backlog_target - projected)
    if router_config is None:
        router_config = StructuredPlanRouterConfig(
            repo_root=Path(repo_root).resolve(),
            branch_count=max(1, min(shortage, limits.max_novel_proposals or 1)),
            max_new_tokens=limits.max_router_tokens,
        )
    context = {
        "task_id": "analysis-escalation",
        "title": "Cover unresolved objective terms",
        "missing_evidence": list(effective_terms),
        "predicted_files": [],
        "predicted_symbols": [],
    }
    try:
        routed = generate_analysis_proposals(
            context,
            objective_terms=effective_terms,
            ast_evidence=ast_summary,
            router=router,
            config=router_config,
            policy=limits,
            known_proposal_ids=known_proposal_ids,
            router_calls_in_window=router_calls_in_window,
            fallback_planner=fallback_planner,
        )
        accepted = list(routed.accepted)
        accepted_payloads = [item.to_dict() for item in accepted[:shortage]]
        proposals.extend(accepted_payloads)
        projected += len(accepted_payloads)
        rejected = [item.to_dict() for item in routed.rejected]
        router_evaluation = routed.router_evaluation or routed.evaluation
        records.append(
            AnalysisEscalationRecord(
                AnalysisEscalationStage.LLM_ROUTER,
                (
                    AnalysisEscalationStatus.ANALYSIS_INCONCLUSIVE
                    if routed.analysis_inconclusive
                    else AnalysisEscalationStatus.COMPLETED
                ),
                cost={
                    "router_calls": routed.router_calls,
                    "router_retries": routed.router_retries,
                    "reserved_tokens": routed.reserved_tokens,
                    "router_call_timestamps": list(routed.router_call_timestamps),
                },
                scope={
                    "proposal_count": len(routed.proposals),
                    "accepted_count": len(accepted_payloads),
                    "limit_reason": routed.limit_reason,
                },
                novelty=router_evaluation.novelty,
                confidence=router_evaluation.confidence,
                rejected_candidates=tuple(rejected[: limits.max_rejected_candidates]),
                objective_terms=effective_terms,
                accepted_candidates=tuple(accepted_payloads),
                reason=str(routed.router_error or ""),
            )
        )
        if routed.used_fallback:
            fallback_payloads = [item.to_dict() for item in routed.proposals]
            records.append(
                AnalysisEscalationRecord(
                    AnalysisEscalationStage.DETERMINISTIC_FALLBACK,
                    AnalysisEscalationStatus.COMPLETED,
                    cost={"generated_candidates": len(fallback_payloads)},
                    scope={"candidate_limit": limits.max_novel_proposals},
                    novelty=routed.evaluation.novelty,
                    confidence=routed.evaluation.confidence,
                    rejected_candidates=tuple(rejected[: limits.max_rejected_candidates]),
                    objective_terms=effective_terms,
                    accepted_candidates=tuple(accepted_payloads),
                    reason="fallback work is not exhaustion evidence",
                )
            )
        inconclusive = bool(routed.analysis_inconclusive or projected < limits.backlog_target or not ast_healthy)
        return AnalysisEscalationResult(
            (
                AnalysisEscalationStatus.ANALYSIS_INCONCLUSIVE
                if inconclusive
                else AnalysisEscalationStatus.SATISFIED
            ),
            before,
            projected,
            limits.backlog_target,
            tuple(records),
            tuple(proposals),
            analysis_inconclusive=inconclusive,
            deterministic_fallback=routed.used_fallback,
            reason=(
                "bounded analysis ended without completion evidence"
                if inconclusive
                else "schema-constrained proposals restored the backlog target"
            ),
            pipeline_result=pipeline_payload,
        )
    except Exception as exc:
        records.append(
            AnalysisEscalationRecord(
                AnalysisEscalationStage.LLM_ROUTER,
                AnalysisEscalationStatus.FAILED,
                objective_terms=effective_terms,
                reason=f"{type(exc).__name__}: {exc}",
            )
        )
        return AnalysisEscalationResult(
            AnalysisEscalationStatus.ANALYSIS_INCONCLUSIVE,
            before,
            projected,
            limits.backlog_target,
            tuple(records),
            tuple(proposals),
            analysis_inconclusive=True,
            reason="router and fallback planning failed; completion is forbidden",
            pipeline_result=pipeline_payload,
        )


# Public aliases using the terminology from the supervisor policy document.
run_analysis_escalation = run_low_backlog_analysis
escalate_low_backlog_analysis = run_low_backlog_analysis


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
    analysis_escalation: AnalysisEscalationResult | Mapping[str, Any] | None = None,
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

    if isinstance(analysis_escalation, AnalysisEscalationResult):
        escalation_payload: dict[str, Any] | None = analysis_escalation.to_dict()
    elif isinstance(analysis_escalation, Mapping):
        escalation_payload = dict(analysis_escalation)
    else:
        escalation_payload = None
    escalation_eligible = bool(
        escalation_payload and escalation_payload.get("exhaustion_eligible", False)
    )
    if terminal_reason is ScanTerminalReason.EXHAUSTED and escalation_payload is not None and not escalation_eligible:
        terminal_reason = ScanTerminalReason.PARTIAL

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
        "analysis_escalation": escalation_payload,
        "analysis_escalation_required": escalation_payload is not None,
        "analysis_escalation_exhaustion_eligible": escalation_eligible,
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
    if terminal_reason is ScanTerminalReason.EXHAUSTED and quorum.satisfied and (
        escalation_payload is None or escalation_eligible
    ):
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
    "AST_COVERAGE_ANALYZER_VERSION",
    "AnalysisEscalationResult",
    "AstCoverageReport",
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
    "escalate_low_backlog_analysis",
    "run_analysis_escalation",
    "run_audit_scan",
    "run_exhaustive_ast_coverage",
    "run_low_backlog_analysis",
    "scan_codebase_audit",
]
