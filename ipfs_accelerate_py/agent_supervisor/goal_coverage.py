"""Deterministic, explainable objective coverage maps.

The supervisor has several independently useful views of work: objective goals,
todo records, predicted conflict surfaces, repository observations, and
validation receipts.  This module joins those views without treating a task's
terminal status as proof.  Every acceptance criterion gets an explicit row for
each required proof surface and missing rows are retained as ``uncovered``
edges instead of disappearing from the graph.

The public builders accept dataclasses or ordinary mappings so persisted JSON
artifacts can be audited with the same code that handles live supervisor
objects.  All identifiers, ordering, scores, and explanations are derived from
canonical input values; callers can provide ``evaluated_at`` when freshness is
part of the calculation to make replay byte-for-byte deterministic.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence

from .objective_graph import (
    CoverageStatus,
    CoverageSurfaceKind,
    ObjectiveCoverageEdge,
    ObjectiveCoverageGraph,
    materialize_objective_coverage_graph,
)


GOAL_COVERAGE_SCHEMA_VERSION = 1
UNMAPPED_GOAL_ID = "__unmapped__"
DEFAULT_FINDING_MIN_SCORE = 0.20
DEFAULT_EVIDENCE_MAX_AGE_SECONDS = 24 * 60 * 60
MISSING_ACCEPTANCE_CRITERION = "[missing acceptance criterion]"


# The domain assembler and generic objective graph intentionally share one
# enum vocabulary.  Keep this descriptive alias for callers of this module.
CoverageSurface = CoverageSurfaceKind


REQUIRED_SURFACE_GROUPS: tuple[tuple[str, tuple[CoverageSurface, ...]], ...] = (
    ("tasks", (CoverageSurface.TASK,)),
    ("predicted_files", (CoverageSurface.PREDICTED_FILE,)),
    ("changed_files", (CoverageSurface.CHANGED_FILE,)),
    ("symbols_or_interfaces", (CoverageSurface.AST_SYMBOL, CoverageSurface.INTERFACE)),
    ("validation_commands", (CoverageSurface.VALIDATION_COMMAND,)),
    ("validation_receipts", (CoverageSurface.VALIDATION_RECEIPT,)),
    ("receipt_provenance", (CoverageSurface.VALIDATION_RECEIPT,)),
)


def _payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        result = asdict(value)
        return dict(result) if isinstance(result, dict) else {}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        return dict(result) if isinstance(result, Mapping) else {}
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _nested_sources(value: Any) -> list[dict[str, Any]]:
    root = _payload(value)
    sources = [root]
    for name in ("fields", "finding", "metadata", "payload", "coverage_inputs"):
        nested = root.get(name)
        if isinstance(nested, Mapping):
            sources.append(dict(nested))
    return sources


def _items(value: Any, *, split_sentences: bool = False) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "n/a", "null"}:
            return []
        if text[:1] in "[{":
            try:
                return _items(json.loads(text), split_sentences=split_sentences)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        pattern = r"[;\n]+" if split_sentences else r"[,;\n]+"
        return [" ".join(item.split()) for item in re.split(pattern, text) if item.strip()]
    if isinstance(value, Mapping):
        return [json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)]
    if isinstance(value, Iterable):
        result: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                result.append(json.dumps(item, sort_keys=True, separators=(",", ":"), default=str))
            else:
                text = " ".join(str(item or "").split())
                if text:
                    result.append(text)
        return result
    text = " ".join(str(value).split())
    return [text] if text else []


def _field_items(sources: Sequence[Mapping[str, Any]], names: Sequence[str], *, sentences: bool = False) -> list[str]:
    found: set[str] = set()
    for source in sources:
        for name in names:
            found.update(_items(source.get(name), split_sentences=sentences))
    return sorted(found, key=lambda item: (item.casefold(), item))


def _first(sources: Sequence[Mapping[str, Any]], names: Sequence[str]) -> Any:
    for source in sources:
        for name in names:
            value = source.get(name)
            if value is not None and str(value).strip():
                return value
    return ""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _stable_id(prefix: str, value: Any) -> str:
    return f"{prefix}-{sha256(_canonical(value).encode('utf-8')).hexdigest()[:24]}"


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _normalized(value))
        if len(token) > 1
    }


def _similarity(left: Any, right: Any) -> float:
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _utc(value: Any) -> datetime | None:
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


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = _normalized(value)
    if text in {"true", "yes", "passed", "pass", "success", "succeeded", "complete", "completed", "ok", "fresh", "valid"}:
        return True
    if text in {"false", "no", "failed", "fail", "error", "cancelled", "canceled", "timed_out", "timeout", "stale", "invalid", "expired"}:
        return False
    return None


def _freshness_bool(value: Any) -> bool | None:
    if isinstance(value, Mapping):
        for key in ("fresh", "is_fresh", "valid", "current"):
            if key in value:
                return _bool(value.get(key))
        return None
    return _bool(value)


def _status_value(value: CoverageStatus | str) -> str:
    return value.value if isinstance(value, CoverageStatus) else str(value)


@dataclass(frozen=True)
class CoverageEdge:
    """One explainable relationship between an acceptance criterion and a surface."""

    goal_id: str
    criterion_id: str
    surface: CoverageSurface | str
    value: str
    relation: str
    status: CoverageStatus | str
    confidence: float
    explanation: str
    task_id: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    provenance_cid: str = ""
    edge_id: str = ""

    def __post_init__(self) -> None:
        surface = self.surface if isinstance(self.surface, CoverageSurface) else CoverageSurface(str(self.surface))
        status = self.status if isinstance(self.status, CoverageStatus) else CoverageStatus(str(self.status))
        object.__setattr__(self, "surface", surface)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "confidence", round(max(0.0, min(1.0, float(self.confidence))), 6))
        object.__setattr__(self, "evidence", dict(self.evidence or {}))
        if not self.edge_id:
            identity = {
                "goal_id": self.goal_id,
                "criterion_id": self.criterion_id,
                "surface": surface.value,
                "value": self.value,
                "relation": self.relation,
                "task_id": self.task_id,
                "provenance_cid": self.provenance_cid,
                "evidence": self.evidence,
            }
            object.__setattr__(self, "edge_id", _stable_id("edge", identity))

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "goal_id": self.goal_id,
            "criterion_id": self.criterion_id,
            "surface": self.surface.value,
            "value": self.value,
            "relation": self.relation,
            "status": self.status.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "task_id": self.task_id,
            "evidence": dict(self.evidence),
            "provenance_cid": self.provenance_cid,
        }


@dataclass(frozen=True)
class ValidationReceiptCoverage:
    """Normalized validation proof and its fail-closed conclusion."""

    receipt_id: str
    task_id: str
    criterion: str
    command: str
    status: CoverageStatus
    passed: bool | None
    repository_tree: str
    observed_at: str
    provenance_cid: str
    explanation: str
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True)
class AcceptanceCoverage:
    """All implementation and proof surfaces for one mandatory criterion."""

    criterion_id: str
    goal_id: str
    criterion: str
    status: CoverageStatus
    task_ids: list[str] = field(default_factory=list)
    predicted_files: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    ast_symbols: list[str] = field(default_factory=list)
    interfaces: list[str] = field(default_factory=list)
    validation_commands: list[str] = field(default_factory=list)
    validation_receipt_ids: list[str] = field(default_factory=list)
    provenance_cids: list[str] = field(default_factory=list)
    missing_surfaces: list[str] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)
    explanation: str = ""

    @property
    def verified(self) -> bool:
        return self.status is CoverageStatus.VERIFIED

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["verified"] = self.verified
        return payload


@dataclass(frozen=True)
class FindingAssignment:
    """Auditable attachment of a dynamic finding to a registered goal or bucket."""

    finding_id: str
    goal_id: str
    confidence: float
    inferred: bool
    explanation: str
    source_goal_id: str = ""
    finding: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GoalCoverageMap:
    """Serializable output of the coverage join."""

    criteria: list[AcceptanceCoverage]
    edges: list[CoverageEdge]
    receipts: list[ValidationReceiptCoverage]
    finding_assignments: list[FindingAssignment]
    registered_goal_ids: list[str]
    evaluated_at: str
    repository_tree: str = ""
    schema_version: int = GOAL_COVERAGE_SCHEMA_VERSION
    graph_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "criteria",
            sorted(self.criteria, key=lambda item: (item.goal_id, _normalized(item.criterion), item.criterion_id)),
        )
        object.__setattr__(self, "edges", sorted(self.edges, key=lambda item: item.edge_id))
        object.__setattr__(self, "receipts", sorted(self.receipts, key=lambda item: item.receipt_id))
        object.__setattr__(
            self,
            "finding_assignments",
            sorted(self.finding_assignments, key=lambda item: (item.goal_id, item.finding_id)),
        )
        object.__setattr__(self, "registered_goal_ids", sorted(set(self.registered_goal_ids)))
        if not self.graph_id:
            identity = {
                "schema_version": self.schema_version,
                "criteria": [item.to_dict() for item in self.criteria],
                "edges": [item.to_dict() for item in self.edges],
                "receipts": [item.to_dict() for item in self.receipts],
                "findings": [item.to_dict() for item in self.finding_assignments],
                "registered_goal_ids": self.registered_goal_ids,
                "repository_tree": self.repository_tree,
            }
            object.__setattr__(self, "graph_id", _stable_id("coverage", identity))

    def edges_with_status(self, status: CoverageStatus | str) -> list[CoverageEdge]:
        value = _status_value(status)
        return [edge for edge in self.edges if edge.status.value == value]

    @property
    def uncovered(self) -> list[CoverageEdge]:
        return self.edges_with_status(CoverageStatus.UNCOVERED)

    @property
    def weakly_inferred(self) -> list[CoverageEdge]:
        return self.edges_with_status(CoverageStatus.WEAKLY_INFERRED)

    @property
    def stale(self) -> list[CoverageEdge]:
        return self.edges_with_status(CoverageStatus.STALE)

    @property
    def contradicted(self) -> list[CoverageEdge]:
        return self.edges_with_status(CoverageStatus.CONTRADICTED)

    @property
    def verified(self) -> list[CoverageEdge]:
        return self.edges_with_status(CoverageStatus.VERIFIED)

    @property
    def unmapped_findings(self) -> list[FindingAssignment]:
        return [item for item in self.finding_assignments if item.goal_id == UNMAPPED_GOAL_ID]

    def to_objective_graph(self) -> ObjectiveCoverageGraph:
        """Project this domain map into the repository's generic graph type."""

        nodes: list[dict[str, Any]] = []
        generic_edges: list[ObjectiveCoverageEdge] = []
        for goal_id in self.registered_goal_ids:
            nodes.append({"node_id": f"goal:{goal_id}", "kind": "goal", "goal_id": goal_id})
        nodes.append(
            {
                "node_id": f"goal:{UNMAPPED_GOAL_ID}",
                "kind": "unmapped_bucket",
                "goal_id": UNMAPPED_GOAL_ID,
                "label": "Unmapped dynamic findings",
            }
        )
        for criterion in self.criteria:
            nodes.append(
                {
                    "node_id": criterion.criterion_id,
                    "surface_kind": CoverageSurface.ACCEPTANCE_CRITERION.value,
                    "goal_id": criterion.goal_id,
                    "value": criterion.criterion,
                    "status": criterion.status.value,
                }
            )
            generic_edges.append(
                ObjectiveCoverageEdge(
                    source=f"goal:{criterion.goal_id}",
                    target=criterion.criterion_id,
                    kind="requires_acceptance_criterion",
                    status=criterion.status,
                    confidence=1.0,
                    explanation=criterion.explanation,
                    evidence=({"missing_surfaces": criterion.missing_surfaces},),
                )
            )
        for edge in self.edges:
            surface_node_id = _stable_id(
                "surface",
                {"surface": edge.surface.value, "value": edge.value, "task_id": edge.task_id},
            )
            nodes.append(
                {
                    "node_id": surface_node_id,
                    "surface_kind": edge.surface.value,
                    "value": edge.value,
                    "task_id": edge.task_id,
                }
            )
            generic_edges.append(
                ObjectiveCoverageEdge(
                    source=edge.criterion_id,
                    target=surface_node_id,
                    kind=edge.relation,
                    status=edge.status,
                    confidence=edge.confidence,
                    explanation=edge.explanation,
                    evidence=(edge.evidence,),
                    provenance_cid=edge.provenance_cid,
                )
            )
        for assignment in self.finding_assignments:
            finding_node_id = f"finding:{assignment.finding_id}"
            nodes.append(
                {
                    "node_id": finding_node_id,
                    "surface_kind": CoverageSurface.FINDING.value,
                    "value": assignment.finding_id,
                    "finding": assignment.finding,
                }
            )
            generic_edges.append(
                ObjectiveCoverageEdge(
                    source=f"goal:{assignment.goal_id}",
                    target=finding_node_id,
                    kind="has_dynamic_finding",
                    status=(
                        CoverageStatus.UNCOVERED
                        if assignment.goal_id == UNMAPPED_GOAL_ID
                        else CoverageStatus.WEAKLY_INFERRED
                        if assignment.inferred
                        else CoverageStatus.VERIFIED
                    ),
                    confidence=assignment.confidence,
                    explanation=assignment.explanation,
                    evidence=(assignment.finding,),
                )
            )
        return materialize_objective_coverage_graph(nodes, generic_edges)

    def to_dict(self) -> dict[str, Any]:
        by_status = {
            status.value: [edge.to_dict() for edge in self.edges_with_status(status)]
            for status in CoverageStatus
        }
        return {
            "schema_version": self.schema_version,
            "graph_id": self.graph_id,
            "evaluated_at": self.evaluated_at,
            "repository_tree": self.repository_tree,
            "registered_goal_ids": self.registered_goal_ids,
            "criteria": [item.to_dict() for item in self.criteria],
            "edges": [item.to_dict() for item in self.edges],
            "receipts": [item.to_dict() for item in self.receipts],
            "finding_assignments": [item.to_dict() for item in self.finding_assignments],
            "unmapped_bucket": {
                "goal_id": UNMAPPED_GOAL_ID,
                "label": "Unmapped dynamic findings",
                "findings": [item.to_dict() for item in self.unmapped_findings],
            },
            "surfaces_by_status": by_status,
            "status_counts": {key: len(value) for key, value in by_status.items()},
            "criterion_status_counts": {
                status.value: sum(item.status is status for item in self.criteria)
                for status in CoverageStatus
            },
            "objective_graph": self.to_objective_graph().to_dict(),
        }


def acceptance_criteria_for_goal(goal: Any) -> list[str]:
    """Return explicit mandatory criteria, falling back to required evidence."""

    sources = _nested_sources(goal)
    criteria = _field_items(
        sources,
        ("acceptance_criteria", "acceptance_criterion", "acceptance"),
        sentences=True,
    )
    if not criteria:
        criteria = _field_items(sources, ("required_evidence", "evidence"), sentences=True)
    return criteria


def _goal_id(goal: Any) -> str:
    sources = _nested_sources(goal)
    return str(_first(sources, ("goal_id", "id")) or "").strip()


def _goal_search_text(goal: Any) -> str:
    sources = _nested_sources(goal)
    values = [
        str(_first(sources, ("goal_id", "id")) or ""),
        str(_first(sources, ("title", "goal", "summary", "description")) or ""),
        *_field_items(sources, ("acceptance", "acceptance_criteria", "evidence", "required_evidence"), sentences=True),
        *_field_items(sources, ("outputs", "predicted_files", "ast_symbols", "interfaces")),
    ]
    return " ".join(values)


def _task_record(task: Any) -> dict[str, Any]:
    sources = _nested_sources(task)
    task_id = str(_first(sources, ("task_id", "id", "canonical_task_id", "task_cid")) or "").strip()
    criteria = _field_items(sources, ("acceptance_criteria", "acceptance_criterion", "acceptance"), sentences=True)
    return {
        "raw": _payload(task),
        "task_id": task_id or _stable_id("task", _payload(task)),
        "goal_id": str(_first(sources, ("goal_id", "objective_id")) or "").strip(),
        "criteria": criteria,
        "predicted_files": _field_items(sources, ("predicted_files", "files", "outputs", "requested_outputs")),
        "changed_files": _field_items(sources, ("changed_files", "changed_paths", "actual_changed_paths", "branch_diff_paths")),
        "ast_symbols": _field_items(sources, ("ast_symbols", "predicted_symbols", "symbols")),
        "interfaces": _field_items(sources, ("interfaces", "interface_contracts", "public_interfaces")),
        "validation_commands": _field_items(sources, ("validation_commands", "validation", "commands"), sentences=True),
    }


def _task_matches_criterion(task: Mapping[str, Any], criterion: str, criterion_count: int) -> tuple[bool, float, str]:
    declared = list(task.get("criteria") or [])
    if not declared:
        return True, 0.75, "task explicitly names the goal and has no narrower acceptance criterion"
    exact = {_normalized(item) for item in declared}
    if _normalized(criterion) in exact:
        return True, 1.0, "task explicitly declares this acceptance criterion"
    score = max((_similarity(item, criterion) for item in declared), default=0.0)
    if score >= 0.20:
        return True, round(score, 6), f"task acceptance text overlaps criterion (Jaccard={score:.6f})"
    if criterion_count == 1:
        return True, max(0.5, score), "single-criterion goal makes the goal-scoped task relevant"
    return False, score, f"task acceptance overlap {score:.6f} is below 0.200000"


def _receipt_dicts(receipts: Sequence[Any], tasks: Sequence[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in receipts:
        payload = _payload(value)
        if payload:
            rows.append(payload)
        elif str(value or "").strip():
            rows.append({"receipt_id": str(value).strip()})
    for task in tasks:
        sources = _nested_sources(task)
        contextual_task_id = str(_first(sources, ("task_id", "id", "producing_task_or_scan", "producer_id")) or "").strip()
        for source in sources:
            for name in ("validation_receipts", "receipts", "validation_receipt"):
                value = source.get(name)
                if isinstance(value, Mapping):
                    rows.append(dict(value))
                elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    rows.extend(dict(item) for item in value if isinstance(item, Mapping))
                elif isinstance(value, str) and value.strip().startswith(("{", "[")):
                    try:
                        decoded = json.loads(value)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(decoded, Mapping):
                        rows.append(dict(decoded))
                    elif isinstance(decoded, list):
                        rows.extend(dict(item) for item in decoded if isinstance(item, Mapping))
                elif isinstance(value, str) and value.strip():
                    rows.append({"receipt_id": value.strip(), "task_id": contextual_task_id})
    unique = {_canonical(row): row for row in rows if row}
    return [unique[key] for key in sorted(unique)]


def normalize_validation_receipt(
    receipt: Any,
    *,
    evaluated_at: datetime,
    repository_tree: str = "",
    max_age_seconds: int = DEFAULT_EVIDENCE_MAX_AGE_SECONDS,
) -> ValidationReceiptCoverage:
    """Normalize receipt aliases and classify proof using fail-closed rules."""

    raw = _payload(receipt)
    sources = _nested_sources(raw)
    task_id = str(_first(sources, ("task_id", "producing_task_or_scan", "producer_id", "producer")) or "").strip()
    criterion = str(_first(sources, ("acceptance_criterion", "criterion")) or "").strip()
    command = str(_first(sources, ("command", "validation_command", "validation")) or "").strip()
    provenance = str(_first(sources, ("provenance_cid", "receipt_cid", "cid", "evidence_cid")) or "").strip()
    tree = str(_first(sources, ("repository_tree", "tree_id", "tree_identity")) or "").strip()
    observed = _utc(_first(sources, ("observed_at", "finished_at", "generated_at", "created_at", "timestamp")))
    passed = _bool(_first(sources, ("validation_passed", "passed", "success", "status")))
    contradictory = _bool(_first(sources, ("contradictory", "contradicted"))) is True
    contradiction_text = str(_first(sources, ("contradiction", "failure_reason", "error")) or "").strip()
    freshness = _freshness_bool(_first(sources, ("freshness", "fresh", "is_fresh")))
    fresh_until = _utc(_first(sources, ("fresh_until", "expires_at")))

    status: CoverageStatus | None = None
    explanation = ""
    if contradictory or contradiction_text or passed is False:
        status = CoverageStatus.CONTRADICTED
        explanation = contradiction_text or "validation receipt explicitly contradicts the claimed coverage"
    elif repository_tree and tree and tree != repository_tree:
        status = CoverageStatus.CONTRADICTED
        explanation = f"receipt repository tree {tree!r} does not match current tree {repository_tree!r}"
    elif freshness is False or (fresh_until is not None and fresh_until < evaluated_at):
        status = CoverageStatus.STALE
        explanation = "validation receipt is explicitly stale or expired"
    elif observed is not None and max_age_seconds >= 0 and observed + timedelta(seconds=max_age_seconds) < evaluated_at:
        status = CoverageStatus.STALE
        explanation = f"validation receipt is older than {max_age_seconds} seconds"
    current_enough = bool(
        freshness is True
        or (fresh_until is not None and fresh_until >= evaluated_at)
        or (
            observed is not None
            and (max_age_seconds < 0 or observed + timedelta(seconds=max_age_seconds) >= evaluated_at)
        )
    )
    if status is None and passed is True and provenance and (not repository_tree or tree == repository_tree) and current_enough:
        status = CoverageStatus.VERIFIED
        explanation = "successful, current validation receipt has provenance"
    elif status is None:
        status = CoverageStatus.WEAKLY_INFERRED
        missing = []
        if passed is not True:
            missing.append("an explicit passing result")
        if not provenance:
            missing.append("a provenance CID")
        if repository_tree and not tree:
            missing.append("a repository tree identity")
        if not current_enough:
            missing.append("freshness evidence")
        explanation = "validation receipt lacks " + ", ".join(missing or ["complete verification metadata"])

    identity = raw or {
        "task_id": task_id,
        "criterion": criterion,
        "command": command,
        "provenance_cid": provenance,
    }
    return ValidationReceiptCoverage(
        receipt_id=str(_first(sources, ("receipt_id",)) or "").strip() or _stable_id("receipt", identity),
        task_id=task_id,
        criterion=criterion,
        command=command,
        status=status,
        passed=passed,
        repository_tree=tree,
        observed_at=observed.isoformat() if observed else "",
        provenance_cid=provenance,
        explanation=explanation,
        raw=raw,
    )


def attach_findings_to_goals(
    goals: Sequence[Any],
    findings: Sequence[Any],
    *,
    min_score: float = DEFAULT_FINDING_MIN_SCORE,
) -> list[FindingAssignment]:
    """Attach each dynamic finding once, retaining a labeled unmapped bucket."""

    goal_rows = [(_goal_id(goal), _goal_search_text(goal)) for goal in goals if _goal_id(goal)]
    goal_text = {goal_id: text for goal_id, text in goal_rows}
    assignments: list[FindingAssignment] = []
    for finding in findings:
        sources = _nested_sources(finding)
        raw = _payload(finding)
        finding_id = str(_first(sources, ("finding_id", "fingerprint", "id")) or "").strip() or _stable_id("finding", raw)
        source_goal_id = str(_first(sources, ("goal_id", "objective_id")) or "").strip()
        search_text = " ".join(
            [
                str(_first(sources, ("title", "summary", "description", "goal", "gap_task")) or ""),
                *_field_items(sources, ("missing_evidence", "outputs", "predicted_files", "ast_symbols", "interfaces")),
            ]
        )
        if source_goal_id in goal_text:
            goal_id, score, inferred = source_goal_id, 1.0, False
            explanation = "finding explicitly names a registered goal"
        else:
            ranked = sorted(
                ((_similarity(search_text, text), goal_id) for goal_id, text in goal_rows),
                key=lambda item: (-item[0], item[1]),
            )
            score, candidate = ranked[0] if ranked else (0.0, "")
            if candidate and score >= min_score:
                goal_id, inferred = candidate, True
                explanation = f"most relevant registered goal by deterministic token Jaccard score {score:.6f}"
            else:
                goal_id, inferred = UNMAPPED_GOAL_ID, True
                explanation = f"no registered goal met the relevance threshold {min_score:.6f}; best score was {score:.6f}"
        assignments.append(
            FindingAssignment(
                finding_id=finding_id,
                goal_id=goal_id,
                confidence=round(score, 6),
                inferred=inferred,
                explanation=explanation,
                source_goal_id=source_goal_id,
                finding=raw,
            )
        )
    return sorted(assignments, key=lambda item: (item.goal_id, item.finding_id))


def _receipt_relevant(receipt: ValidationReceiptCoverage, criterion: str, task_ids: set[str]) -> bool:
    if receipt.task_id and receipt.task_id in task_ids:
        return True
    if receipt.criterion and _normalized(receipt.criterion) == _normalized(criterion):
        return True
    return False


def _criterion_status(
    *,
    missing: Sequence[str],
    receipts: Sequence[ValidationReceiptCoverage],
) -> tuple[CoverageStatus, str]:
    if any(receipt.status is CoverageStatus.CONTRADICTED for receipt in receipts):
        return CoverageStatus.CONTRADICTED, "one or more validation receipts contradict the claimed criterion coverage"
    if not missing and any(receipt.status is CoverageStatus.VERIFIED for receipt in receipts):
        return CoverageStatus.VERIFIED, "every required implementation surface has current provenance-backed validation proof"
    if any(receipt.status is CoverageStatus.STALE for receipt in receipts):
        return CoverageStatus.STALE, "no current proof supersedes the stale validation evidence"
    if "tasks" in missing:
        return CoverageStatus.UNCOVERED, "no task covers this acceptance criterion"
    return CoverageStatus.WEAKLY_INFERRED, "coverage is partial or lacks current provenance-backed validation proof"


def build_goal_coverage_map(
    goals: Sequence[Any],
    tasks: Sequence[Any] = (),
    *,
    findings: Sequence[Any] = (),
    validation_receipts: Sequence[Any] = (),
    repository_tree: str = "",
    evaluated_at: datetime | str | None = None,
    evidence_max_age_seconds: int = DEFAULT_EVIDENCE_MAX_AGE_SECONDS,
    finding_min_score: float = DEFAULT_FINDING_MIN_SCORE,
) -> GoalCoverageMap:
    """Build the complete goal-to-task/code/AST/acceptance/validation map."""

    instant = _utc(evaluated_at) or datetime.now(timezone.utc)
    goal_rows = [
        (
            goal,
            _goal_id(goal),
            acceptance_criteria_for_goal(goal) or [MISSING_ACCEPTANCE_CRITERION],
        )
        for goal in goals
    ]
    registered = sorted({goal_id for _goal, goal_id, _criteria in goal_rows if goal_id})
    task_rows = [_task_record(task) for task in tasks]
    normalized_receipts = [
        normalize_validation_receipt(
            receipt,
            evaluated_at=instant,
            repository_tree=repository_tree,
            max_age_seconds=evidence_max_age_seconds,
        )
        for receipt in _receipt_dicts(validation_receipts, tasks)
    ]
    edges: list[CoverageEdge] = []
    criterion_rows: list[AcceptanceCoverage] = []

    for _goal, goal_id, criteria in sorted(goal_rows, key=lambda item: item[1]):
        if not goal_id:
            continue
        for criterion in criteria:
            criterion_id = _stable_id("criterion", {"goal_id": goal_id, "criterion": _normalized(criterion)})
            scoped: list[tuple[dict[str, Any], float, str]] = []
            for task in task_rows:
                if criterion == MISSING_ACCEPTANCE_CRITERION:
                    continue
                if task["goal_id"] != goal_id:
                    continue
                matched, confidence, reason = _task_matches_criterion(task, criterion, len(criteria))
                if matched:
                    scoped.append((task, confidence, reason))
            scoped.sort(key=lambda item: item[0]["task_id"])
            task_ids = sorted({task["task_id"] for task, _score, _reason in scoped})
            predicted = sorted({item for task, _score, _reason in scoped for item in task["predicted_files"]})
            changed = sorted({item for task, _score, _reason in scoped for item in task["changed_files"]})
            symbols = sorted({item for task, _score, _reason in scoped for item in task["ast_symbols"]})
            interfaces = sorted({item for task, _score, _reason in scoped for item in task["interfaces"]})
            commands = sorted({item for task, _score, _reason in scoped for item in task["validation_commands"]})
            relevant_receipts = [
                receipt for receipt in normalized_receipts if _receipt_relevant(receipt, criterion, set(task_ids))
            ]
            provenance = sorted({receipt.provenance_cid for receipt in relevant_receipts if receipt.provenance_cid})
            dimension_values: dict[CoverageSurface, list[str]] = {
                CoverageSurface.TASK: task_ids,
                CoverageSurface.PREDICTED_FILE: predicted,
                CoverageSurface.CHANGED_FILE: changed,
                CoverageSurface.AST_SYMBOL: symbols,
                CoverageSurface.INTERFACE: interfaces,
                CoverageSurface.VALIDATION_COMMAND: commands,
                CoverageSurface.VALIDATION_RECEIPT: [receipt.receipt_id for receipt in relevant_receipts],
            }
            missing: list[str] = []
            if not task_ids:
                missing.append("tasks")
            if not predicted:
                missing.append("predicted_files")
            if not changed:
                missing.append("changed_files")
            if not symbols and not interfaces:
                missing.append("symbols_or_interfaces")
            if not commands:
                missing.append("validation_commands")
            if not relevant_receipts:
                missing.append("validation_receipts")
            if not provenance:
                missing.append("receipt_provenance")
            status, explanation = _criterion_status(missing=missing, receipts=relevant_receipts)

            criterion_edge_ids: list[str] = []
            criterion_edge = CoverageEdge(
                goal_id=goal_id,
                criterion_id=criterion_id,
                surface=CoverageSurface.ACCEPTANCE_CRITERION,
                value=criterion,
                relation="criterion_coverage",
                status=status,
                confidence=1.0 if status in {CoverageStatus.UNCOVERED, CoverageStatus.VERIFIED} else 0.75,
                explanation=explanation,
                evidence={
                    "missing_surfaces": missing,
                    "task_ids": task_ids,
                    "validation_receipt_ids": [item.receipt_id for item in relevant_receipts],
                },
            )
            edges.append(criterion_edge)
            criterion_edge_ids.append(criterion_edge.edge_id)
            for task, confidence, reason in scoped:
                edge = CoverageEdge(
                    goal_id=goal_id,
                    criterion_id=criterion_id,
                    surface=CoverageSurface.TASK,
                    value=task["task_id"],
                    relation="implemented_by",
                    status=CoverageStatus.VERIFIED if confidence == 1.0 else CoverageStatus.WEAKLY_INFERRED,
                    confidence=confidence,
                    explanation=reason,
                    task_id=task["task_id"],
                    evidence={"declared_goal_id": task["goal_id"], "declared_criteria": task["criteria"]},
                )
                edges.append(edge)
                criterion_edge_ids.append(edge.edge_id)
            for surface, values in dimension_values.items():
                if surface is CoverageSurface.TASK:
                    continue
                for value in values:
                    owner = next(
                        (task["task_id"] for task, _score, _reason in scoped if value in task.get({
                            CoverageSurface.PREDICTED_FILE: "predicted_files",
                            CoverageSurface.CHANGED_FILE: "changed_files",
                            CoverageSurface.AST_SYMBOL: "ast_symbols",
                            CoverageSurface.INTERFACE: "interfaces",
                            CoverageSurface.VALIDATION_COMMAND: "validation_commands",
                        }.get(surface, ""), [])),
                        "",
                    )
                    receipt = next((item for item in relevant_receipts if item.receipt_id == value), None)
                    edge_status = receipt.status if receipt else (CoverageStatus.VERIFIED if owner else CoverageStatus.WEAKLY_INFERRED)
                    provenance_cid = receipt.provenance_cid if receipt else ""
                    relation = "proved_by" if receipt else "maps_to"
                    edge_explanation = receipt.explanation if receipt else f"{surface.value} is explicitly declared by task {owner}"
                    edge = CoverageEdge(
                        goal_id=goal_id,
                        criterion_id=criterion_id,
                        surface=surface,
                        value=value,
                        relation=relation,
                        status=edge_status,
                        confidence=1.0 if owner or receipt else 0.5,
                        explanation=edge_explanation,
                        task_id=owner or (receipt.task_id if receipt else ""),
                        evidence=receipt.raw if receipt else {"source": "task_metadata"},
                        provenance_cid=provenance_cid,
                    )
                    edges.append(edge)
                    criterion_edge_ids.append(edge.edge_id)
            for missing_name in missing:
                missing_surface = next(group[1][0] for group in REQUIRED_SURFACE_GROUPS if group[0] == missing_name)
                edge = CoverageEdge(
                    goal_id=goal_id,
                    criterion_id=criterion_id,
                    surface=missing_surface,
                    value=f"uncovered:{missing_name}",
                    relation="missing_surface",
                    status=CoverageStatus.UNCOVERED,
                    confidence=1.0,
                    explanation=f"acceptance criterion has no mapped {missing_name.replace('_', ' ')} evidence",
                    evidence={"required_surface_group": missing_name},
                )
                edges.append(edge)
                criterion_edge_ids.append(edge.edge_id)
            criterion_rows.append(
                AcceptanceCoverage(
                    criterion_id=criterion_id,
                    goal_id=goal_id,
                    criterion=criterion,
                    status=status,
                    task_ids=task_ids,
                    predicted_files=predicted,
                    changed_files=changed,
                    ast_symbols=symbols,
                    interfaces=interfaces,
                    validation_commands=commands,
                    validation_receipt_ids=sorted(receipt.receipt_id for receipt in relevant_receipts),
                    provenance_cids=provenance,
                    missing_surfaces=missing,
                    edge_ids=sorted(set(criterion_edge_ids)),
                    explanation=explanation,
                )
            )

    assignments = attach_findings_to_goals(goals, findings, min_score=finding_min_score)
    for assignment in assignments:
        if assignment.goal_id == UNMAPPED_GOAL_ID:
            continue
        criterion_ids = [item.criterion_id for item in criterion_rows if item.goal_id == assignment.goal_id]
        for criterion_id in criterion_ids:
            edges.append(
                CoverageEdge(
                    goal_id=assignment.goal_id,
                    criterion_id=criterion_id,
                    surface=CoverageSurface.FINDING,
                    value=assignment.finding_id,
                    relation="finding_for_goal",
                    status=CoverageStatus.WEAKLY_INFERRED if assignment.inferred else CoverageStatus.VERIFIED,
                    confidence=assignment.confidence,
                    explanation=assignment.explanation,
                    evidence=assignment.finding,
                )
            )

    # Deduplicate identical relationships produced by aliased metadata.
    edge_by_id = {edge.edge_id: edge for edge in edges}
    return GoalCoverageMap(
        criteria=criterion_rows,
        edges=list(edge_by_id.values()),
        receipts=normalized_receipts,
        finding_assignments=assignments,
        registered_goal_ids=registered,
        evaluated_at=instant.isoformat(),
        repository_tree=repository_tree,
    )


def goal_coverage_graph(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility convenience returning a plain JSON-safe graph."""

    return build_goal_coverage_map(*args, **kwargs).to_dict()


# Descriptive compatibility aliases for graph-oriented consumers.
GoalCoverageGraph = GoalCoverageMap
GoalCoverageEdge = CoverageEdge
build_goal_coverage = build_goal_coverage_map


def write_goal_coverage_map(path: Any, coverage: GoalCoverageMap | Mapping[str, Any]) -> Any:
    """Atomically persist a coverage artifact with stable JSON formatting."""

    from pathlib import Path
    import os
    import tempfile

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = coverage.to_dict() if isinstance(coverage, GoalCoverageMap) else dict(coverage)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return target


__all__ = [
    "AcceptanceCoverage",
    "CoverageEdge",
    "CoverageStatus",
    "CoverageSurface",
    "DEFAULT_EVIDENCE_MAX_AGE_SECONDS",
    "DEFAULT_FINDING_MIN_SCORE",
    "FindingAssignment",
    "GOAL_COVERAGE_SCHEMA_VERSION",
    "GoalCoverageMap",
    "GoalCoverageGraph",
    "GoalCoverageEdge",
    "MISSING_ACCEPTANCE_CRITERION",
    "REQUIRED_SURFACE_GROUPS",
    "UNMAPPED_GOAL_ID",
    "ValidationReceiptCoverage",
    "acceptance_criteria_for_goal",
    "attach_findings_to_goals",
    "build_goal_coverage_map",
    "build_goal_coverage",
    "goal_coverage_graph",
    "normalize_validation_receipt",
    "write_goal_coverage_map",
]
