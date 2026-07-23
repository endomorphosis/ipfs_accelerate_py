"""Compile canonical supervisor records into formal work plans.

The compiler is a deliberately deterministic boundary between operational
records (objective/taskboard/AST/policy data) and the reviewed planning
vocabulary.  It never asks a model to infer a formula from prose.  Text which
cannot be interpreted by a reviewed rule is retained as an auditable
abstraction, while missing or ambiguous *semantic* input produces an explicit
unsupported or invalid result.

JSON and DuckDB readers feed the same normalized record bundle.  Source
ordering, database row ordering, and storage-specific metadata therefore do
not participate in plan or graph identity.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

from .formal_logic_vocabulary import (
    LOGIC_VOCABULARY_VERSION,
    Formula,
    ReviewedPredicate,
    TDFOL,
    TermSort,
    atom,
    constant,
)
from .formal_planning_contracts import (
    Actor,
    ActorKind,
    Effect,
    EffectOperation,
    EvidenceRequirement,
    EvidenceRequirementKind,
    EventKind,
    Fluent,
    FluentValueType,
    FormalWorkPlan,
    Goal,
    Norm,
    NormKind,
    PlanEvent,
    PlanTask,
    Precondition,
    TemporalConstraint,
    TemporalConstraintKind,
)
from .formal_verification_contracts import (
    AssuranceLevel,
    ContractValidationError,
    canonical_json,
    content_identity,
)


FORMAL_PLAN_COMPILER_VERSION: Final = 1
FORMAL_PLAN_COMPILATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-plan-compilation@1"
)
FORMAL_PLAN_GRAPH_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-plan-graph@1"
)
FORMAL_PLAN_INPUT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-plan-input@1"
)
DEFAULT_VOCABULARY_PROFILE_ID: Final = "supervisor-reviewed"


class CompilationStatus(str, Enum):
    """Terminal outcome of compilation."""

    COMPILED = "compiled"
    SUCCESS = "compiled"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"


class CompilationIssueSeverity(str, Enum):
    ABSTRACTION = "abstraction"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


class CompilationIssueCode(str, Enum):
    MALFORMED_JSON = "malformed_json"
    DUCKDB_UNAVAILABLE = "duckdb_unavailable"
    DUCKDB_READ_ERROR = "duckdb_read_error"
    INVALID_SOURCE = "invalid_source"
    INVALID_RECORD = "invalid_record"
    MISSING_SEMANTICS = "missing_semantics"
    UNKNOWN_SEMANTIC = "unknown_semantic"
    UNKNOWN_FIELD = "unknown_field"
    ABSTRACTED_FIELD = "abstracted_field"
    CYCLE = "cycle"
    UNKNOWN_DEPENDENCY = "unknown_dependency"
    AMBIGUOUS_EFFECT = "ambiguous_effect"
    MULTIPLE_REPOSITORY_TREES = "multiple_repository_trees"
    CONTRACT_VIOLATION = "contract_violation"


@dataclass(frozen=True)
class CompilationIssue:
    """One stable diagnostic or abstraction decision."""

    code: CompilationIssueCode
    severity: CompilationIssueSeverity
    path: str
    message: str
    source_id: str = ""
    field_name: str = ""
    value: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", CompilationIssueCode(self.code))
        object.__setattr__(self, "severity", CompilationIssueSeverity(self.severity))
        for name in ("path", "message", "source_id", "field_name"):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())

    @property
    def issue_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "path": self.path,
            "message": self.message,
            "source_id": self.source_id,
            "field_name": self.field_name,
            "value": _canonical_safe(self.value),
        }


@dataclass(frozen=True)
class PlanGraphProjection:
    """Small canonical graph projection used by planning and context queries."""

    nodes: tuple[Mapping[str, Any], ...] = ()
    edges: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        nodes = _unique_records(self.nodes, "node_id")
        edges = _unique_records(self.edges, "edge_id")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)

    @property
    def graph_id(self) -> str:
        return content_identity(self.canonical_records())

    def canonical_records(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "nodes": [dict(item) for item in self.nodes],
            "edges": [dict(item) for item in self.edges],
        }

    def to_dict(self) -> dict[str, Any]:
        records = self.canonical_records()
        return {
            "schema": FORMAL_PLAN_GRAPH_SCHEMA,
            "graph_id": self.graph_id,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            **records,
        }


@dataclass(frozen=True)
class PlanCompilationResult:
    """Compilation output, including failures which intentionally have no plan."""

    status: CompilationStatus
    plan: FormalWorkPlan | None = None
    formulas: tuple[Formula, ...] = ()
    graph_projection: PlanGraphProjection = field(default_factory=PlanGraphProjection)
    issues: tuple[CompilationIssue, ...] = ()
    source_identity: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", CompilationStatus(self.status))
        formulas = {formula.formula_id: formula for formula in self.formulas}
        object.__setattr__(
            self, "formulas", tuple(formulas[key] for key in sorted(formulas))
        )
        object.__setattr__(
            self,
            "issues",
            tuple(
                sorted(
                    self.issues,
                    key=lambda item: (
                        item.severity.value,
                        item.path,
                        item.code.value,
                        item.issue_id,
                    ),
                )
            ),
        )
        if self.status is CompilationStatus.COMPILED and self.plan is None:
            raise ValueError("compiled results require a formal plan")
        if self.status is not CompilationStatus.COMPILED and self.plan is not None:
            raise ValueError("failed compilation results cannot carry a formal plan")

    @property
    def plan_id(self) -> str:
        return self.plan.plan_id if self.plan is not None else ""

    @property
    def graph(self) -> PlanGraphProjection:
        return self.graph_projection

    @property
    def valid(self) -> bool:
        return self.status is CompilationStatus.COMPILED

    @property
    def supported(self) -> bool:
        return self.status is not CompilationStatus.UNSUPPORTED

    @property
    def succeeded(self) -> bool:
        return self.valid

    @property
    def unsupported_fields(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    item.path
                    for item in self.issues
                    if item.severity is CompilationIssueSeverity.UNSUPPORTED
                }
            )
        )

    @property
    def abstraction_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                item.issue_id
                for item in self.issues
                if item.severity is CompilationIssueSeverity.ABSTRACTION
            )
        )

    def formula_by_id(self, formula_id: str) -> Formula | None:
        return next(
            (item for item in self.formulas if item.formula_id == formula_id), None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FORMAL_PLAN_COMPILATION_SCHEMA,
            "compiler_version": FORMAL_PLAN_COMPILER_VERSION,
            "status": self.status.value,
            "source_identity": self.source_identity,
            "plan_id": self.plan_id,
            "plan": self.plan.to_record() if self.plan is not None else None,
            "formulas": [item.to_record() for item in self.formulas],
            "graph_projection": self.graph_projection.to_dict(),
            "issues": [
                {**item.to_dict(), "issue_id": item.issue_id} for item in self.issues
            ],
            "unsupported_fields": list(self.unsupported_fields),
            "abstraction_ids": list(self.abstraction_ids),
        }


_SECTION_ALIASES: Final = {
    "objective": "objectives",
    "objective_record": "objectives",
    "objective_records": "objectives",
    "goal": "objectives",
    "goals": "objectives",
    "objectives": "objectives",
    "task": "tasks",
    "task_record": "tasks",
    "task_records": "tasks",
    "taskboard": "tasks",
    "taskboard_records": "tasks",
    "tasks": "tasks",
    "ast": "ast",
    "ast_record": "ast",
    "ast_records": "ast",
    "ast_scope": "ast",
    "ast_scopes": "ast",
    "symbols": "ast",
    "policy": "policies",
    "policies": "policies",
    "policy_record": "policies",
    "policy_records": "policies",
    "proof_policy": "policies",
    "proof_policies": "policies",
    "lease": "leases",
    "leases": "leases",
    "lease_record": "leases",
    "lease_records": "leases",
    "evidence": "evidence",
    "evidence_record": "evidence",
    "evidence_records": "evidence",
}

_KNOWN_FIELDS: Final = {
    "objectives": frozenset(
        {
            "id", "goal_id", "goal_cid", "canonical_goal_id", "cid",
            "owner_actor_id", "owner_id", "actor_id", "title", "description",
            "terminal_states", "source_ids", "subgoals", "acceptance_criteria",
            "trace_bound", "deadline", "metadata", "schema", "content_id",
        }
    ),
    "tasks": frozenset(
        {
            "id", "task_id", "task_cid", "canonical_task_id",
            "canonical_task_cid", "cid", "goal_id", "goal_cid", "subgoal_id",
            "subgoal_cid", "actor_id", "actor_ids", "assigned_to", "assignee",
            "depends_on", "dependencies", "dependency_task_cids",
            "blocking_task_cids", "lease", "lease_id", "lease_cid", "lease_holder",
            "holder_id", "fencing_token", "resource_needs", "resources",
            "required_resources", "changed_ast_scopes", "ast_scope_ids",
            "symbol_cids", "tree_cid", "repository_tree_id",
            "acceptance_criteria", "validation_commands", "effects",
            "preconditions", "events", "evidence_cids", "evidence_ids",
            "status", "title", "description", "terminal_states", "deadline",
            "metadata", "schema", "content_id",
        }
    ),
    "ast": frozenset(
        {
            "id", "cid", "content_id", "ast_cid", "scope_cid", "symbol_cid",
            "tree_cid", "repository_tree_id", "task_id", "task_cid", "symbol",
            "symbol_name", "qualified_name", "path", "file_path", "changed",
            "change_kind", "kind", "metadata", "schema",
        }
    ),
    "policies": frozenset(
        {
            "id", "cid", "content_id", "policy_id", "policy_cid", "name",
            "required_evidence", "evidence_requirements", "minimum_assurance",
            "minimum_code_assurance", "freshness_seconds", "fallback_checks",
            "fallback_check_ids", "obligations", "permissions", "prohibitions",
            "trace_bound", "metadata", "schema",
        }
    ),
    "leases": frozenset(
        {
            "id", "cid", "content_id", "lease_id", "lease_cid", "task_id",
            "task_cid", "actor_id", "holder_id", "lease_holder", "owner_id",
            "fencing_token", "valid_from", "valid_until", "expires_at",
            "resource_needs", "resources", "metadata", "schema",
        }
    ),
    "evidence": frozenset(
        {
            "id", "cid", "content_id", "evidence_id", "evidence_cid",
            "task_id", "task_cid", "goal_id", "goal_cid", "kind", "freshness",
            "scope_ids", "source_scope_ids", "metadata", "schema",
        }
    ),
}

_ABSTRACTED_FIELDS: Final = frozenset(
    {
        "title", "description", "status", "path", "file_path", "symbol",
        "symbol_name", "qualified_name", "change_kind", "expires_at", "name",
        "metadata", "schema", "changed", "kind", "freshness",
    }
)

_UNSUPPORTED_SEMANTIC_FIELDS: Final = {
    "objectives": frozenset({"subgoals"}),
    "tasks": frozenset({"preconditions", "events"}),
    "ast": frozenset(),
    "policies": frozenset({"obligations", "permissions", "prohibitions"}),
    "leases": frozenset(),
    "evidence": frozenset(),
}


def _canonical_safe(value: Any) -> Any:
    try:
        return json.loads(canonical_json(value))
    except (ContractValidationError, TypeError, ValueError):
        return repr(value)


def _text(record: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _values(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Mapping)):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(value)
    return (value,)


def _strings(value: Any) -> tuple[str, ...]:
    return tuple(
        sorted({str(item).strip() for item in _values(value) if str(item).strip()})
    )


def _resource_values(value: Any) -> tuple[str, ...]:
    """Normalize list- and mapping-shaped resource declarations."""

    if isinstance(value, Mapping):
        return tuple(
            str(key).strip()
            for key, _ in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).strip()
        )
    return _strings(value)


def _record(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        normalized = json.loads(canonical_json(value))
        if not isinstance(normalized, dict):
            raise TypeError("record canonicalization did not produce an object")
        return normalized
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return _record(result)
    to_record = getattr(value, "to_record", None)
    if callable(to_record):
        result = to_record()
        if isinstance(result, Mapping):
            return _record(result)
    raise TypeError("records must be mappings or expose to_dict()/to_record()")


def _records(value: Any) -> tuple[dict[str, Any], ...]:
    return tuple(_record(item) for item in _values(value))


def _source_id(record: Mapping[str, Any], kind: str) -> str:
    preferred = {
        "objectives": ("goal_cid", "content_id", "cid", "goal_id", "id"),
        "tasks": (
            "task_cid", "canonical_task_cid", "content_id", "cid", "task_id", "id"
        ),
        "ast": (
            "symbol_cid", "scope_cid", "ast_cid", "content_id", "cid", "id"
        ),
        "policies": ("policy_cid", "content_id", "cid", "policy_id", "id"),
        "leases": ("lease_cid", "content_id", "cid", "lease_id", "id"),
        "evidence": ("evidence_cid", "content_id", "cid", "evidence_id", "id"),
    }[kind]
    return _text(record, *preferred) or content_identity(record)


def _unique_records(
    values: Iterable[Mapping[str, Any]], key: str
) -> tuple[Mapping[str, Any], ...]:
    result: dict[str, dict[str, Any]] = {}
    for raw in values:
        item = _record(raw)
        identity = str(item.get(key) or "").strip()
        if not identity:
            raise ValueError(f"{key} is required")
        previous = result.get(identity)
        if previous is not None and previous != item:
            raise ValueError(f"conflicting graph record {identity}")
        result[identity] = item
    return tuple(result[item] for item in sorted(result))


def _graph_node(kind: str, record_id: str, **attributes: Any) -> dict[str, Any]:
    material = {
        "kind": kind,
        "record_id": record_id,
        "attributes": {
            key: _canonical_safe(value)
            for key, value in sorted(attributes.items())
            if value not in (None, "", (), [], {})
        },
    }
    return {"node_id": content_identity(material), **material}


def _graph_edge(
    kind: str, source: str, target: str, *, source_id: str = ""
) -> dict[str, Any]:
    material = {
        "kind": kind,
        "source": source,
        "target": target,
        "source_id": source_id,
    }
    return {"edge_id": content_identity(material), **material}


def _normalize_bundle(payload: Mapping[str, Any]) -> dict[str, Any]:
    bundle: dict[str, Any] = {
        "objectives": [],
        "tasks": [],
        "ast": [],
        "policies": [],
        "leases": [],
        "evidence": [],
        "repository_tree_id": _text(
            payload, "repository_tree_id", "tree_cid", "tree_id"
        ),
    }
    records_value = payload.get("records")
    if isinstance(records_value, Sequence) and not isinstance(
        records_value, (str, bytes, bytearray)
    ):
        for item in records_value:
            record = _record(item)
            section = _SECTION_ALIASES.get(
                _text(record, "record_type", "section", "kind").lower()
            )
            if section:
                bundle[section].append(record.get("record", record.get("payload", record)))
    for key, value in payload.items():
        section = _SECTION_ALIASES.get(str(key).lower())
        if section:
            bundle[section].extend(_records(value))
    graph_payloads = []
    if isinstance(payload.get("evidence_graph"), Mapping):
        graph_payloads.append(payload["evidence_graph"])
    if isinstance(payload.get("code_evidence_graph"), Mapping):
        graph_payloads.append(payload["code_evidence_graph"])
    if isinstance(payload.get("nodes"), Sequence) and not isinstance(
        payload.get("nodes"), (str, bytes, bytearray)
    ):
        graph_payloads.append(payload)
    for graph in graph_payloads:
        for raw_node in graph.get("nodes", ()):
            if not isinstance(raw_node, Mapping):
                continue
            kind = _text(raw_node, "kind", "node_kind")
            record = raw_node.get("record")
            if not isinstance(record, Mapping):
                payload_value = raw_node.get("payload")
                record = payload_value if isinstance(payload_value, Mapping) else None
            if not isinstance(record, Mapping):
                continue
            section = {
                "goal": "objectives",
                "objective": "objectives",
                "task": "tasks",
                "tree": "ast",
                "symbol": "ast",
                "ast_scope": "ast",
                "policy": "policies",
                "obligation": "evidence",
                "proof": "evidence",
                "validation": "evidence",
                "evidence": "evidence",
            }.get(kind)
            if section:
                bundle[section].append(_record(record))
    # A bare objective/task/etc. record is accepted when record_type identifies it.
    record_type = _SECTION_ALIASES.get(
        _text(payload, "record_type", "section").lower()
    )
    if record_type and not bundle[record_type]:
        bundle[record_type].append(_record(payload))
    for section in _KNOWN_FIELDS:
        unique: dict[str, dict[str, Any]] = {}
        for item in bundle[section]:
            record = _record(item)
            unique[canonical_json(record)] = record
        bundle[section] = [
            unique[key] for key in sorted(unique)
        ]
    return bundle


def _failure_result(
    status: CompilationStatus,
    issues: Iterable[CompilationIssue],
    *,
    source_identity: str = "",
    graph: PlanGraphProjection | None = None,
) -> PlanCompilationResult:
    return PlanCompilationResult(
        status=status,
        issues=tuple(issues),
        source_identity=source_identity,
        graph_projection=graph or PlanGraphProjection(),
    )


class FormalPlanCompiler:
    """Deterministic compiler for objective/taskboard/AST/policy snapshots."""

    def __init__(
        self,
        *,
        strict_unknown_semantics: bool = True,
        default_trace_bound: int = 16,
    ) -> None:
        if isinstance(default_trace_bound, bool) or default_trace_bound <= 0:
            raise ValueError("default_trace_bound must be a positive integer")
        self.strict_unknown_semantics = bool(strict_unknown_semantics)
        self.default_trace_bound = int(default_trace_bound)

    def compile(
        self,
        source: Mapping[str, Any] | None = None,
        *,
        objective_records: Iterable[Any] = (),
        task_records: Iterable[Any] = (),
        ast_records: Iterable[Any] = (),
        policy_records: Iterable[Any] = (),
        lease_records: Iterable[Any] = (),
        evidence_records: Iterable[Any] = (),
        repository_tree_id: str = "",
    ) -> PlanCompilationResult:
        """Compile mappings or explicit record channels into a formal plan."""

        try:
            payload = _record(source or {})
            additions = {
                "objective_records": tuple(objective_records),
                "task_records": tuple(task_records),
                "ast_records": tuple(ast_records),
                "policy_records": tuple(policy_records),
                "lease_records": tuple(lease_records),
                "evidence_records": tuple(evidence_records),
            }
            for key, value in additions.items():
                if value:
                    payload[key] = [_record(item) for item in value]
            if repository_tree_id:
                payload["repository_tree_id"] = str(repository_tree_id)
            bundle = _normalize_bundle(payload)
        except (ContractValidationError, TypeError, ValueError) as exc:
            return _failure_result(
                CompilationStatus.INVALID,
                (
                    CompilationIssue(
                        CompilationIssueCode.INVALID_SOURCE,
                        CompilationIssueSeverity.ERROR,
                        "$",
                        str(exc),
                    ),
                ),
            )
        return self._compile_bundle(bundle)

    def compile_json(self, source: str | bytes | Path) -> PlanCompilationResult:
        """Compile a JSON object, JSON text, or path without leaking parse errors."""

        try:
            if isinstance(source, Path):
                text = source.read_text(encoding="utf-8")
            elif isinstance(source, bytes):
                text = source.decode("utf-8")
            elif isinstance(source, str):
                candidate = Path(source)
                if (
                    not source.lstrip().startswith(("{", "["))
                    and len(source) < 4096
                    and candidate.is_file()
                ):
                    text = candidate.read_text(encoding="utf-8")
                else:
                    text = source
            else:
                raise TypeError("JSON source must be text, bytes, or a path")
            value = json.loads(text)
            if not isinstance(value, Mapping):
                raise ValueError("formal-plan JSON must contain an object")
        except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return _failure_result(
                CompilationStatus.INVALID,
                (
                    CompilationIssue(
                        CompilationIssueCode.MALFORMED_JSON,
                        CompilationIssueSeverity.ERROR,
                        "$",
                        f"could not parse formal-plan input: {exc}",
                    ),
                ),
            )
        return self.compile(value)

    def compile_duckdb(
        self,
        source: Any,
        *,
        objective_records: Iterable[Any] = (),
        task_records: Iterable[Any] = (),
        ast_records: Iterable[Any] = (),
        policy_records: Iterable[Any] = (),
        lease_records: Iterable[Any] = (),
        evidence_records: Iterable[Any] = (),
        repository_tree_id: str = "",
    ) -> PlanCompilationResult:
        """Compile canonical rows from a DuckDB path or open connection."""

        try:
            import duckdb  # type: ignore
        except ImportError as exc:
            return _failure_result(
                CompilationStatus.UNSUPPORTED,
                (
                    CompilationIssue(
                        CompilationIssueCode.DUCKDB_UNAVAILABLE,
                        CompilationIssueSeverity.UNSUPPORTED,
                        "$",
                        f"DuckDB input is unavailable: {exc}",
                    ),
                ),
            )

        connection = None
        owned = False
        try:
            if hasattr(source, "execute") and callable(source.execute):
                connection = source
            else:
                connection = duckdb.connect(str(Path(source)), read_only=True)
                owned = True
            bundle = self._read_duckdb(connection)
            supplements = {
                "objectives": objective_records,
                "tasks": task_records,
                "ast": ast_records,
                "policies": policy_records,
                "leases": lease_records,
                "evidence": evidence_records,
            }
            combined = dict(bundle)
            for section, values in supplements.items():
                combined[section] = [
                    *bundle.get(section, ()),
                    *(_record(item) for item in values),
                ]
            if repository_tree_id:
                combined["repository_tree_id"] = str(repository_tree_id)
            bundle = _normalize_bundle(combined)
        except Exception as exc:
            return _failure_result(
                CompilationStatus.INVALID,
                (
                    CompilationIssue(
                        CompilationIssueCode.DUCKDB_READ_ERROR,
                        CompilationIssueSeverity.ERROR,
                        "$",
                        f"could not read DuckDB formal-plan input: {exc}",
                    ),
                ),
            )
        finally:
            if owned and connection is not None:
                connection.close()
        return self._compile_bundle(bundle)

    def compile_source(self, source: Any) -> PlanCompilationResult:
        """Detect mapping, JSON, or DuckDB input and compile it."""

        if isinstance(source, Mapping):
            return self.compile(source)
        if hasattr(source, "execute") and callable(source.execute):
            return self.compile_duckdb(source)
        if isinstance(source, (str, Path)):
            suffix = (
                Path(source).suffix.lower()
                if not isinstance(source, str) or len(source) < 4096
                else ""
            )
            if suffix in {".duckdb", ".db"}:
                return self.compile_duckdb(source)
        return self.compile_json(source)

    def _read_duckdb(self, connection: Any) -> dict[str, Any]:
        tables = {
            str(row[0])
            for row in connection.execute("SHOW TABLES").fetchall()
        }
        bundle: dict[str, Any] = {
            "objectives": [],
            "tasks": [],
            "ast": [],
            "policies": [],
            "leases": [],
            "evidence": [],
            "repository_tree_id": "",
        }
        # Lossless generic representation written by
        # write_formal_plan_compiler_input_duckdb.
        if "formal_plan_input_records" in tables:
            rows = connection.execute(
                "SELECT section, payload_json FROM formal_plan_input_records "
                "ORDER BY section, record_id"
            ).fetchall()
            for section, payload in rows:
                target = _SECTION_ALIASES.get(str(section).lower())
                if target:
                    value = json.loads(str(payload))
                    bundle[target].append(_record(value))
            if "formal_plan_input_metadata" in tables:
                metadata = dict(
                    connection.execute(
                        "SELECT field_name, field_value FROM formal_plan_input_metadata"
                    ).fetchall()
                )
                bundle["repository_tree_id"] = str(
                    metadata.get("repository_tree_id") or ""
                )
            return _normalize_bundle(bundle)

        aliases = {
            "objectives": ("objective_records", "objectives", "goals"),
            "tasks": ("task_records", "taskboard", "tasks"),
            "ast": ("ast_records", "ast_scopes", "symbols"),
            "policies": ("policy_records", "proof_policies", "policies"),
            "leases": ("lease_records", "leases"),
            "evidence": ("evidence_records",),
        }
        for section, candidates in aliases.items():
            table = next((name for name in candidates if name in tables), None)
            if table is not None:
                bundle[section].extend(self._duckdb_table_records(connection, table))

        # Evidence-graph artifacts are also accepted.  Their payloads preserve
        # the original authoritative task/AST/evidence records.
        if "evidence_nodes" in tables:
            rows = connection.execute(
                "SELECT node_kind, payload_json FROM evidence_nodes "
                "ORDER BY node_id"
            ).fetchall()
            for node_kind, payload in rows:
                value = json.loads(str(payload))
                record = value.get("record") if isinstance(value, Mapping) else None
                if not isinstance(record, Mapping):
                    continue
                section = {
                    "task": "tasks",
                    "tree": "ast",
                    "symbol": "ast",
                    "ast_scope": "ast",
                    "obligation": "evidence",
                    "proof": "evidence",
                    "validation": "evidence",
                }.get(str(node_kind))
                if section:
                    bundle[section].append(_record(record))

        if not any(bundle[name] for name in _KNOWN_FIELDS):
            raise ValueError("no recognized formal-plan source tables")
        return _normalize_bundle(bundle)

    @staticmethod
    def _duckdb_table_records(connection: Any, table: str) -> list[dict[str, Any]]:
        cursor = connection.execute(f'SELECT * FROM "{table}"')
        columns = [str(item[0]) for item in cursor.description]
        result: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            record = dict(zip(columns, row))
            json_field = next(
                (
                    name for name in ("payload_json", "record_json", "json")
                    if name in record and record[name] not in (None, "")
                ),
                None,
            )
            if json_field:
                value = json.loads(str(record[json_field]))
                if not isinstance(value, Mapping):
                    raise ValueError(f"{table}.{json_field} must contain an object")
                result.append(_record(value))
            else:
                result.append(
                    {
                        key: _decode_duckdb_value(value)
                        for key, value in record.items()
                        if value is not None
                    }
                )
        return result

    def _compile_bundle(self, bundle: Mapping[str, Any]) -> PlanCompilationResult:
        source_identity = content_identity(
            {
                "schema": FORMAL_PLAN_INPUT_SCHEMA,
                **{
                    name: bundle.get(name, [])
                    for name in ("objectives", "tasks", "ast", "policies", "leases", "evidence")
                },
                "repository_tree_id": bundle.get("repository_tree_id", ""),
            }
        )
        issues: list[CompilationIssue] = []
        for section in _KNOWN_FIELDS:
            for index, record in enumerate(bundle.get(section, ())):
                sid = _source_id(record, section)
                for field_name in sorted(set(record) - _KNOWN_FIELDS[section]):
                    severity = (
                        CompilationIssueSeverity.UNSUPPORTED
                        if self.strict_unknown_semantics
                        and any(
                            token in field_name.lower()
                            for token in ("effect", "semantic", "operator", "predicate")
                        )
                        else CompilationIssueSeverity.ABSTRACTION
                    )
                    issues.append(
                        CompilationIssue(
                            CompilationIssueCode.UNKNOWN_SEMANTIC
                            if severity is CompilationIssueSeverity.UNSUPPORTED
                            else CompilationIssueCode.UNKNOWN_FIELD,
                            severity,
                            f"$.{section}[{index}].{field_name}",
                            "field is not part of the reviewed compiler mapping",
                            sid,
                            field_name,
                            record[field_name],
                        )
                    )
                for field_name in sorted(set(record) & _ABSTRACTED_FIELDS):
                    if record.get(field_name) not in (None, "", {}, []):
                        issues.append(
                            CompilationIssue(
                                CompilationIssueCode.ABSTRACTED_FIELD,
                                CompilationIssueSeverity.ABSTRACTION,
                                f"$.{section}[{index}].{field_name}",
                                "descriptive field was retained as provenance, "
                                "not parsed into logic",
                                sid,
                                field_name,
                                record[field_name],
                            )
                        )
                for field_name in sorted(
                    set(record) & _UNSUPPORTED_SEMANTIC_FIELDS[section]
                ):
                    if record.get(field_name) not in (None, "", {}, []):
                        issues.append(
                            CompilationIssue(
                                CompilationIssueCode.UNKNOWN_SEMANTIC,
                                CompilationIssueSeverity.UNSUPPORTED,
                                f"$.{section}[{index}].{field_name}",
                                "nested semantic field has no reviewed compiler rule",
                                sid,
                                field_name,
                                record[field_name],
                            )
                        )

        objectives = list(bundle.get("objectives", ()))
        tasks = list(bundle.get("tasks", ()))
        policies = list(bundle.get("policies", ()))
        ast_records = list(bundle.get("ast", ()))
        leases = list(bundle.get("leases", ()))
        evidence = list(bundle.get("evidence", ()))
        if not objectives:
            issues.append(
                CompilationIssue(
                    CompilationIssueCode.MISSING_SEMANTICS,
                    CompilationIssueSeverity.UNSUPPORTED,
                    "$.objectives",
                    "at least one objective with a stable identity is required",
                )
            )
        if not tasks:
            issues.append(
                CompilationIssue(
                    CompilationIssueCode.MISSING_SEMANTICS,
                    CompilationIssueSeverity.UNSUPPORTED,
                    "$.tasks",
                    "at least one task with acceptance or effect semantics is required",
                )
            )
        if not policies:
            issues.append(
                CompilationIssue(
                    CompilationIssueCode.MISSING_SEMANTICS,
                    CompilationIssueSeverity.UNSUPPORTED,
                    "$.policies",
                    "a proof/evidence policy is required",
                )
            )
        if any(
            item.severity is CompilationIssueSeverity.UNSUPPORTED for item in issues
        ):
            graph = self._source_graph(bundle)
            return _failure_result(
                CompilationStatus.UNSUPPORTED,
                issues,
                source_identity=source_identity,
                graph=graph,
            )

        try:
            return self._build_plan(
                objectives, tasks, ast_records, policies, leases, evidence,
                str(bundle.get("repository_tree_id") or ""),
                issues, source_identity,
            )
        except ContractValidationError as exc:
            text = str(exc)
            if "acyclic" in text:
                code = CompilationIssueCode.CYCLE
            elif "unknown dependency" in text:
                code = CompilationIssueCode.UNKNOWN_DEPENDENCY
            elif "conflicting effects" in text:
                code = CompilationIssueCode.AMBIGUOUS_EFFECT
            else:
                code = CompilationIssueCode.CONTRACT_VIOLATION
            issues.append(
                CompilationIssue(
                    code,
                    CompilationIssueSeverity.ERROR,
                    "$",
                    text,
                )
            )
        except (TypeError, ValueError) as exc:
            issues.append(
                CompilationIssue(
                    CompilationIssueCode.INVALID_RECORD,
                    CompilationIssueSeverity.ERROR,
                    "$",
                    str(exc),
                )
            )
        return _failure_result(
            CompilationStatus.INVALID,
            issues,
            source_identity=source_identity,
            graph=self._source_graph(bundle),
        )

    def _build_plan(
        self,
        objectives: list[Mapping[str, Any]],
        tasks: list[Mapping[str, Any]],
        ast_records: list[Mapping[str, Any]],
        policies: list[Mapping[str, Any]],
        leases: list[Mapping[str, Any]],
        evidence_records: list[Mapping[str, Any]],
        repository_tree_id: str,
        issues: list[CompilationIssue],
        source_identity: str,
    ) -> PlanCompilationResult:
        objective_ids: dict[str, str] = {}
        for record in objectives:
            canonical = _text(
                record, "goal_cid", "content_id", "cid", "canonical_goal_id",
                "goal_id", "id"
            )
            if not canonical:
                raise ValueError("objective identity is required")
            for alias in (
                canonical,
                _text(record, "goal_id"),
                _text(record, "id"),
                _text(record, "canonical_goal_id"),
            ):
                if alias:
                    objective_ids[alias] = canonical

        task_ids: dict[str, str] = {}
        task_by_canonical: dict[str, Mapping[str, Any]] = {}
        for record in tasks:
            canonical = _text(
                record, "task_cid", "canonical_task_cid", "content_id", "cid",
                "canonical_task_id", "task_id", "id"
            )
            if not canonical:
                raise ValueError("task identity is required")
            if canonical in task_by_canonical and task_by_canonical[canonical] != record:
                raise ValueError(f"conflicting task records for {canonical}")
            task_by_canonical[canonical] = record
            for alias in (
                canonical,
                _text(record, "task_id"),
                _text(record, "id"),
                _text(record, "canonical_task_id"),
                _text(record, "canonical_task_cid"),
            ):
                if alias:
                    task_ids[alias] = canonical

        lease_by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for lease in leases:
            reference = _text(lease, "task_cid", "task_id")
            target = task_ids.get(reference, reference)
            if target:
                lease_by_task[target].append(lease)
        for task_id, record in task_by_canonical.items():
            if isinstance(record.get("lease"), Mapping):
                lease_by_task[task_id].append(_record(record["lease"]))

        actors: dict[str, Actor] = {}
        supervisor = Actor(
            actor_id="supervisor",
            kind=ActorKind.SUPERVISOR,
            capabilities=("assign", "delegate", "enforce_policy"),
        )
        actors[supervisor.actor_id] = supervisor

        formulae: dict[str, Formula] = {}
        plan_goals: list[Goal] = []
        trace_candidates = [self.default_trace_bound, len(tasks) * 3 + 2]
        for record in objectives:
            value = record.get("trace_bound")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                trace_candidates.append(value)
        for policy in policies:
            value = policy.get("trace_bound")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                trace_candidates.append(value)
        for record in tasks:
            value = record.get("deadline")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                trace_candidates.append(value)
        trace_bound = max(trace_candidates)

        goal_requirement_ids: dict[str, list[str]] = defaultdict(list)
        task_requirement_ids: dict[str, list[str]] = defaultdict(list)
        requirements: dict[str, EvidenceRequirement] = {}

        # Changed AST scopes are authoritative evidence scopes, never formulas.
        ast_by_task: dict[str, set[str]] = defaultdict(set)
        tree_ids: set[str] = set()
        if repository_tree_id:
            tree_ids.add(repository_tree_id)
        for ast_record in ast_records:
            tree = _text(ast_record, "tree_cid", "repository_tree_id")
            if tree:
                tree_ids.add(tree)
            scope_id = _text(
                ast_record, "symbol_cid", "scope_cid", "ast_cid", "content_id", "cid", "id"
            )
            task_ref = _text(ast_record, "task_cid", "task_id")
            target = task_ids.get(task_ref, task_ref)
            if scope_id and target:
                ast_by_task[target].add(scope_id)
        for task_id, record in task_by_canonical.items():
            ast_by_task[task_id].update(
                _strings(
                    record.get("changed_ast_scopes")
                    or record.get("ast_scope_ids")
                    or record.get("symbol_cids")
                )
            )
            tree = _text(record, "tree_cid", "repository_tree_id")
            if tree:
                tree_ids.add(tree)
        if len(tree_ids) > 1:
            issues.append(
                CompilationIssue(
                    CompilationIssueCode.MULTIPLE_REPOSITORY_TREES,
                    CompilationIssueSeverity.ERROR,
                    "$.ast",
                    "records refer to multiple repository trees: "
                    + ", ".join(sorted(tree_ids)),
                )
            )
            return _failure_result(
                CompilationStatus.INVALID,
                issues,
                source_identity=source_identity,
                graph=self._source_graph(
                    {
                        "objectives": objectives, "tasks": tasks, "ast": ast_records,
                        "policies": policies, "leases": leases,
                        "evidence": evidence_records,
                        "repository_tree_id": repository_tree_id,
                    }
                ),
            )
        tree_id = next(iter(tree_ids), repository_tree_id)

        for record in objectives:
            goal_id = objective_ids[
                _text(
                    record, "goal_cid", "content_id", "cid", "canonical_goal_id",
                    "goal_id", "id"
                )
            ]
            owner = _text(record, "owner_actor_id", "owner_id", "actor_id") or "supervisor"
            actors.setdefault(owner, Actor(actor_id=owner, kind=ActorKind.HUMAN))
            formula = TDFOL.goal_satisfaction(goal_id, trace_bound)
            formulae[formula.formula_id] = formula
            plan_goals.append(
                Goal(
                    goal_id=goal_id,
                    owner_actor_id=owner,
                    satisfaction_formula_id=formula.formula_id,
                    terminal_states=_strings(record.get("terminal_states")) or ("satisfied",),
                    source_ids=(
                        _source_id(record, "objectives"),
                        *tuple(_strings(record.get("source_ids"))),
                    ),
                    metadata=_descriptive_metadata(record),
                )
            )

        effects: list[Effect] = []
        fluents: list[Fluent] = []
        events: list[PlanEvent] = []
        preconditions: list[Precondition] = []
        constraints: list[TemporalConstraint] = []
        norms: list[Norm] = []
        plan_tasks: list[PlanTask] = []
        topo_times = _topological_times(task_by_canonical, task_ids)

        # Policy records become evidence requirements and reviewed deontic norms.
        policy_ids = tuple(sorted(_source_id(item, "policies") for item in policies))
        policy_defaults = _policy_defaults(policies)
        for task_id, record in sorted(task_by_canonical.items()):
            raw_goal = _text(record, "goal_cid", "goal_id")
            goal_id = objective_ids.get(raw_goal, raw_goal)
            if not goal_id and len(plan_goals) == 1:
                goal_id = plan_goals[0].goal_id
                issues.append(
                    CompilationIssue(
                        CompilationIssueCode.ABSTRACTED_FIELD,
                        CompilationIssueSeverity.ABSTRACTION,
                        f"$.tasks[{task_id}].goal_id",
                        "single objective identity was used for an omitted task goal",
                        _source_id(record, "tasks"),
                        "goal_id",
                    )
                )
            if goal_id not in {item.goal_id for item in plan_goals}:
                raise ValueError(f"task {task_id} has unknown goal {goal_id!r}")

            dependencies = tuple(
                sorted(
                    {
                        task_ids.get(value, value)
                        for value in _strings(
                            record.get("depends_on")
                            or record.get("dependencies")
                            or record.get("dependency_task_cids")
                            or record.get("blocking_task_cids")
                        )
                    }
                )
            )
            actor_values = set(
                _strings(record.get("actor_ids"))
                or _strings(
                    _text(record, "actor_id", "assigned_to", "assignee")
                )
            )
            task_leases = lease_by_task.get(task_id, [])
            for lease in task_leases:
                holder = _text(
                    lease, "actor_id", "holder_id", "lease_holder", "owner_id"
                )
                if holder:
                    actor_values.add(holder)
            if len(task_leases) > 1:
                active_holders = {
                    _text(item, "actor_id", "holder_id", "lease_holder", "owner_id")
                    for item in task_leases
                } - {""}
                if len(active_holders) > 1:
                    raise ValueError(
                        f"task {task_id} has ambiguous active lease holders"
                    )
            if not actor_values:
                actor_values.add("supervisor")
            for actor_id in actor_values:
                resource_spec = (
                    record.get("resource_needs")
                    or record.get("resources")
                    or record.get("required_resources")
                )
                resources = _resource_values(resource_spec)
                previous_actor = actors.get(actor_id)
                actors[actor_id] = Actor(
                    actor_id=actor_id,
                    kind=previous_actor.kind if previous_actor else ActorKind.AGENT,
                    principal_id=previous_actor.principal_id if previous_actor else "",
                    capabilities=tuple(
                        sorted(
                            {
                                *(previous_actor.capabilities if previous_actor else ()),
                                *resources,
                            }
                        )
                    ),
                    authority_ids=(
                        previous_actor.authority_ids if previous_actor else ()
                    ),
                    metadata=previous_actor.metadata if previous_actor else {},
                )

            base_time = topo_times[task_id] * 3
            event_records: list[PlanEvent] = []
            for ordinal, (kind, offset) in enumerate(
                (
                    (EventKind.ASSIGNED, 0),
                    (EventKind.STARTED, 1),
                    (EventKind.COMPLETED, 2),
                )
            ):
                event_id = f"{task_id}:event:{kind.value}"
                provenance = {
                    _source_id(record, "tasks"),
                    *(_source_id(item, "leases") for item in task_leases),
                }
                event = PlanEvent(
                    event_id=event_id,
                    kind=kind,
                    actor_id=sorted(actor_values)[0],
                    task_id=task_id,
                    logical_time=base_time + offset,
                    provenance_ids=tuple(sorted(provenance)),
                    metadata={
                        "lease_ids": sorted(
                            _source_id(item, "leases") for item in task_leases
                        ),
                        "fencing_tokens": sorted(
                            {
                                str(item.get("fencing_token"))
                                for item in task_leases
                                if item.get("fencing_token") not in (None, "")
                            }
                        ),
                    },
                )
                event_records.append(event)
                events.append(event)

            state_id = f"{task_id}:state"
            fluents.append(
                Fluent(
                    fluent_id=state_id,
                    value_type=FluentValueType.SYMBOL,
                    initial_value="pending",
                    metadata={"task_id": task_id},
                )
            )
            task_effects = self._compile_effects(
                task_id, record, state_id, event_records[-1].event_id
            )
            effects.extend(task_effects)
            known_fluent_ids = {item.fluent_id for item in fluents}
            for effect in task_effects:
                if not effect.fluent_id or effect.fluent_id in known_fluent_ids:
                    continue
                value = effect.value
                if isinstance(value, bool):
                    value_type = FluentValueType.BOOLEAN
                elif isinstance(value, int):
                    value_type = FluentValueType.INTEGER
                elif isinstance(value, str):
                    value_type = FluentValueType.SYMBOL
                else:
                    value_type = FluentValueType.SYMBOL
                fluents.append(
                    Fluent(
                        fluent_id=effect.fluent_id,
                        value_type=value_type,
                        initial_value=None,
                        metadata={
                            "task_id": task_id,
                            "compiler_rule": "effect_fluent_type_inference",
                        },
                    )
                )
                known_fluent_ids.add(effect.fluent_id)

            task_preconditions: list[Precondition] = []
            for actor_id in sorted(actor_values):
                authorization_formula = atom(
                    ReviewedPredicate.AUTHORIZED,
                    constant(TermSort.ACTOR, actor_id),
                    constant(TermSort.TASK, task_id),
                )
                formulae[authorization_formula.formula_id] = authorization_formula
                authorization = Precondition(
                    precondition_id=content_identity(
                        {
                            "kind": "authorization",
                            "actor_id": actor_id,
                            "task_id": task_id,
                            "lease_ids": sorted(
                                _source_id(item, "leases") for item in task_leases
                            ),
                            "resource_needs": sorted(resources),
                            "resource_spec": _canonical_safe(resource_spec),
                        }
                    ),
                    formula_id=authorization_formula.formula_id,
                    task_id=task_id,
                    event_id=event_records[1].event_id,
                    metadata={
                        "actor_id": actor_id,
                        "lease_ids": sorted(
                            _source_id(item, "leases") for item in task_leases
                        ),
                        "resource_needs": sorted(resources),
                        "resource_spec": _canonical_safe(resource_spec),
                    },
                )
                task_preconditions.append(authorization)
                preconditions.append(authorization)
            for dependency in dependencies:
                dependency_formula = atom(
                    ReviewedPredicate.DEPENDENCY_SATISFIED,
                    constant(TermSort.TASK, dependency),
                    constant(TermSort.TASK, task_id),
                )
                formulae[dependency_formula.formula_id] = dependency_formula
                precondition = Precondition(
                    precondition_id=content_identity(
                        {
                            "kind": "dependency",
                            "predecessor": dependency,
                            "successor": task_id,
                        }
                    ),
                    formula_id=dependency_formula.formula_id,
                    task_id=task_id,
                    event_id=event_records[1].event_id,
                    metadata={"dependency_task_id": dependency},
                )
                task_preconditions.append(precondition)
                preconditions.append(precondition)
                temporal_formula = TDFOL.dependency_order(dependency, task_id)
                formulae[temporal_formula.formula_id] = temporal_formula
                constraints.append(
                    TemporalConstraint(
                        constraint_id=content_identity(
                            {
                                "kind": "dependency_order",
                                "predecessor": dependency,
                                "successor": task_id,
                            }
                        ),
                        kind=TemporalConstraintKind.DEPENDENCY_ORDER,
                        subject_ids=(dependency, task_id),
                        formula_id=temporal_formula.formula_id,
                        upper_bound=trace_bound,
                    )
                )
            deadline = record.get("deadline")
            if deadline not in (None, ""):
                if (
                    isinstance(deadline, bool)
                    or not isinstance(deadline, int)
                    or deadline < 0
                ):
                    raise ValueError(
                        f"task {task_id} deadline must be a non-negative integer"
                    )
                deadline_formula = TDFOL.deadline(task_id, deadline)
                formulae[deadline_formula.formula_id] = deadline_formula
                constraints.append(
                    TemporalConstraint(
                        constraint_id=content_identity(
                            {
                                "kind": "deadline",
                                "task_id": task_id,
                                "deadline": deadline,
                            }
                        ),
                        kind=TemporalConstraintKind.DEADLINE,
                        subject_ids=(task_id,),
                        formula_id=deadline_formula.formula_id,
                        upper_bound=deadline,
                    )
                )

            acceptance = _values(record.get("acceptance_criteria"))
            validation_commands = _strings(record.get("validation_commands"))
            if not acceptance and not validation_commands and not record.get("effects"):
                issues.append(
                    CompilationIssue(
                        CompilationIssueCode.MISSING_SEMANTICS,
                        CompilationIssueSeverity.UNSUPPORTED,
                        f"$.tasks[{task_id}]",
                        "task has no acceptance criteria, validation commands, or explicit effects",
                        _source_id(record, "tasks"),
                    )
                )
                continue
            task_scopes = tuple(sorted(ast_by_task.get(task_id, set())))
            for ordinal, criterion in enumerate(acceptance):
                if isinstance(criterion, Mapping):
                    criterion_record = _record(criterion)
                    criterion_source = _source_id(record, "tasks")
                    criterion_known = {
                        "id", "requirement_id", "kind", "check_ids",
                        "validation_commands", "source_scope_ids",
                        "minimum_code_assurance", "freshness_seconds",
                    }
                    for field_name in sorted(set(criterion_record) - criterion_known):
                        issues.append(
                            CompilationIssue(
                                CompilationIssueCode.ABSTRACTED_FIELD,
                                CompilationIssueSeverity.ABSTRACTION,
                                f"$.tasks[{task_id}].acceptance_criteria"
                                f"[{ordinal}].{field_name}",
                                "criterion field is retained as evidence metadata, "
                                "not parsed into logic",
                                criterion_source,
                                field_name,
                                criterion_record[field_name],
                            )
                        )
                    kind = _evidence_kind(criterion_record.get("kind", "test"))
                    check_ids = _strings(
                        criterion_record.get("check_ids")
                        or criterion_record.get("validation_commands")
                    )
                    criterion_value: Any = criterion_record
                else:
                    kind = EvidenceRequirementKind.TEST
                    check_ids = ()
                    criterion_value = str(criterion)
                requirement_id = content_identity(
                    {
                        "task_id": task_id,
                        "criterion": criterion_value,
                        "ordinal": ordinal,
                    }
                )
                requirements[requirement_id] = EvidenceRequirement(
                    requirement_id=requirement_id,
                    kind=kind,
                    subject_ids=(task_id,),
                    source_scope_ids=task_scopes,
                    minimum_code_assurance=policy_defaults["minimum_assurance"],
                    freshness_seconds=policy_defaults["freshness_seconds"],
                    fallback_check_ids=tuple(
                        sorted(
                            {
                                *check_ids,
                                *validation_commands,
                                *policy_defaults["fallback_checks"],
                            }
                        )
                    ),
                    metadata={
                        "criterion": criterion_value,
                        "policy_ids": list(policy_ids),
                    },
                )
                task_requirement_ids[task_id].append(requirement_id)
            if not acceptance and validation_commands:
                requirement_id = content_identity(
                    {
                        "task_id": task_id,
                        "validation_commands": validation_commands,
                    }
                )
                requirements[requirement_id] = EvidenceRequirement(
                    requirement_id=requirement_id,
                    kind=EvidenceRequirementKind.TEST,
                    subject_ids=(task_id,),
                    source_scope_ids=task_scopes,
                    minimum_code_assurance=policy_defaults["minimum_assurance"],
                    freshness_seconds=policy_defaults["freshness_seconds"],
                    fallback_check_ids=validation_commands,
                    metadata={"policy_ids": list(policy_ids)},
                )
                task_requirement_ids[task_id].append(requirement_id)

            for actor_id in sorted(actor_values):
                norms.append(
                    Norm(
                        norm_id=content_identity(
                            {
                                "kind": "obligation",
                                "actor_id": actor_id,
                                "task_id": task_id,
                                "policy_ids": policy_ids,
                            }
                        ),
                        kind=NormKind.OBLIGATION,
                        bearer_actor_id=actor_id,
                        issuer_actor_id="supervisor",
                        action_id=task_id,
                        valid_from=base_time,
                        valid_until=trace_bound,
                        metadata={
                            "policy_ids": list(policy_ids),
                            "lease_ids": [
                                _source_id(item, "leases") for item in task_leases
                            ],
                        },
                    )
                )
            plan_tasks.append(
                PlanTask(
                    task_id=task_id,
                    goal_id=goal_id,
                    subgoal_id="",
                    actor_ids=tuple(sorted(actor_values)),
                    depends_on=dependencies,
                    precondition_ids=tuple(
                        item.precondition_id for item in task_preconditions
                    ),
                    effect_ids=tuple(item.effect_id for item in task_effects),
                    event_ids=tuple(item.event_id for item in event_records),
                    evidence_requirement_ids=tuple(
                        sorted(task_requirement_ids[task_id])
                    ),
                    terminal_states=_strings(record.get("terminal_states"))
                    or ("completed", "failed", "cancelled"),
                    metadata={
                        **_descriptive_metadata(record),
                        "source_cids": sorted(
                            {
                                _source_id(record, "tasks"),
                                *(_source_id(item, "leases") for item in task_leases),
                                *_strings(
                                    record.get("evidence_cids")
                                    or record.get("evidence_ids")
                                ),
                            }
                        ),
                        "changed_ast_scope_ids": list(task_scopes),
                        "resource_needs": list(
                            _resource_values(
                                record.get("resource_needs")
                                or record.get("resources")
                                or record.get("required_resources")
                            )
                        ),
                        "resource_spec": _canonical_safe(
                            record.get("resource_needs")
                            or record.get("resources")
                            or record.get("required_resources")
                        ),
                        "policy_ids": list(policy_ids),
                    },
                )
            )

        if any(
            item.severity is CompilationIssueSeverity.UNSUPPORTED for item in issues
        ):
            return _failure_result(
                CompilationStatus.UNSUPPORTED,
                issues,
                source_identity=source_identity,
                graph=self._source_graph(
                    {
                        "objectives": objectives, "tasks": tasks, "ast": ast_records,
                        "policies": policies, "leases": leases,
                        "evidence": evidence_records,
                        "repository_tree_id": repository_tree_id,
                    }
                ),
            )

        # Explicit policy evidence applies to all goals unless scoped by task.
        for record in objectives:
            goal_id = objective_ids[
                _text(
                    record, "goal_cid", "content_id", "cid", "canonical_goal_id",
                    "goal_id", "id"
                )
            ]
            for ordinal, criterion in enumerate(
                _values(record.get("acceptance_criteria"))
            ):
                requirement_id = content_identity(
                    {
                        "goal_id": goal_id,
                        "criterion": criterion,
                        "ordinal": ordinal,
                    }
                )
                requirements[requirement_id] = EvidenceRequirement(
                    requirement_id=requirement_id,
                    kind=EvidenceRequirementKind.REVIEW,
                    subject_ids=(goal_id,),
                    minimum_code_assurance=policy_defaults["minimum_assurance"],
                    freshness_seconds=policy_defaults["freshness_seconds"],
                    fallback_check_ids=policy_defaults["fallback_checks"],
                    metadata={
                        "criterion": _canonical_safe(criterion),
                        "policy_ids": list(policy_ids),
                    },
                )
                goal_requirement_ids[goal_id].append(requirement_id)

        for policy in policies:
            for ordinal, spec in enumerate(
                _values(
                    policy.get("required_evidence")
                    or policy.get("evidence_requirements")
                )
            ):
                item = _record(spec) if isinstance(spec, Mapping) else {"kind": spec}
                policy_requirement_known = {
                    "id", "requirement_id", "kind", "subject_ids",
                    "source_scope_ids", "minimum_code_assurance",
                    "freshness_seconds", "fallback_check_ids",
                }
                for field_name in sorted(set(item) - policy_requirement_known):
                    issues.append(
                        CompilationIssue(
                            CompilationIssueCode.ABSTRACTED_FIELD,
                            CompilationIssueSeverity.ABSTRACTION,
                            f"$.policies[{_source_id(policy, 'policies')}]."
                            f"required_evidence[{ordinal}].{field_name}",
                            "policy evidence field is retained as provenance, "
                            "not parsed into logic",
                            _source_id(policy, "policies"),
                            field_name,
                            item[field_name],
                        )
                    )
                subjects = tuple(
                    sorted(
                        task_ids.get(value, objective_ids.get(value, value))
                        for value in _strings(item.get("subject_ids"))
                    )
                )
                if not subjects:
                    subjects = tuple(item.goal_id for item in plan_goals)
                requirement_id = _text(item, "requirement_id", "id") or content_identity(
                    {
                        "policy_id": _source_id(policy, "policies"),
                        "ordinal": ordinal,
                        "spec": item,
                        "subjects": subjects,
                    }
                )
                requirements[requirement_id] = EvidenceRequirement(
                    requirement_id=requirement_id,
                    kind=_evidence_kind(item.get("kind", "artifact")),
                    subject_ids=subjects,
                    source_scope_ids=_strings(item.get("source_scope_ids")),
                    minimum_code_assurance=_assurance(
                        item.get(
                            "minimum_code_assurance",
                            policy_defaults["minimum_assurance"],
                        )
                    ),
                    freshness_seconds=_optional_nonnegative(
                        item.get("freshness_seconds", policy_defaults["freshness_seconds"])
                    ),
                    fallback_check_ids=_strings(
                        item.get("fallback_check_ids")
                        or policy_defaults["fallback_checks"]
                    ),
                    metadata={"policy_id": _source_id(policy, "policies")},
                )
                for subject in subjects:
                    if subject in {item.goal_id for item in plan_goals}:
                        goal_requirement_ids[subject].append(requirement_id)

        plan_goals = [
            Goal(
                goal_id=item.goal_id,
                owner_actor_id=item.owner_actor_id,
                satisfaction_formula_id=item.satisfaction_formula_id,
                evidence_requirement_ids=tuple(
                    sorted(goal_requirement_ids[item.goal_id])
                ),
                terminal_states=item.terminal_states,
                source_ids=item.source_ids,
                metadata=item.metadata,
            )
            for item in plan_goals
        ]
        source_ids = sorted(
            {
                _source_id(record, section)
                for section, records in (
                    ("objectives", objectives),
                    ("tasks", tasks),
                    ("ast", ast_records),
                    ("policies", policies),
                    ("leases", leases),
                    ("evidence", evidence_records),
                )
                for record in records
            }
        )
        source_ids.extend(
            _source_id(lease, "leases")
            for task_id in sorted(lease_by_task)
            for lease in lease_by_task[task_id]
            if _source_id(lease, "leases") not in source_ids
        )
        source_ids.sort()
        abstraction_ids = tuple(
            sorted(
                item.issue_id
                for item in issues
                if item.severity is CompilationIssueSeverity.ABSTRACTION
            )
        )
        plan = FormalWorkPlan(
            vocabulary_profile_id=DEFAULT_VOCABULARY_PROFILE_ID,
            vocabulary_version=LOGIC_VOCABULARY_VERSION,
            actors=tuple(actors.values()),
            goals=tuple(plan_goals),
            subgoals=(),
            tasks=tuple(plan_tasks),
            events=tuple(events),
            fluents=tuple(fluents),
            preconditions=tuple(preconditions),
            effects=tuple(effects),
            norms=tuple(norms),
            temporal_constraints=tuple(constraints),
            evidence_requirements=tuple(requirements.values()),
            source_ids=tuple(source_ids),
            repository_tree_id=tree_id,
            trace_bound=trace_bound,
            abstraction_ids=abstraction_ids,
            metadata={
                "compiler_version": FORMAL_PLAN_COMPILER_VERSION,
                "formula_records": [
                    formulae[key].to_record() for key in sorted(formulae)
                ],
                "policy_ids": list(policy_ids),
                "evidence_cids": sorted(
                    _source_id(item, "evidence") for item in evidence_records
                ),
            },
        )
        graph = self._plan_graph(
            plan,
            ast_records=ast_records,
            policies=policies,
            leases=tuple(
                {
                    canonical_json(item): item
                    for item in (
                        *leases,
                        *(
                            lease
                            for task_id in sorted(lease_by_task)
                            for lease in lease_by_task[task_id]
                        ),
                    )
                }.values()
            ),
            evidence_records=evidence_records,
        )
        return PlanCompilationResult(
            status=CompilationStatus.COMPILED,
            plan=plan,
            formulas=tuple(formulae.values()),
            graph_projection=graph,
            issues=tuple(issues),
            source_identity=source_identity,
        )

    @staticmethod
    def _compile_effects(
        task_id: str,
        record: Mapping[str, Any],
        state_id: str,
        completed_event_id: str,
    ) -> list[Effect]:
        raw = _values(record.get("effects"))
        if not raw:
            return [
                Effect(
                    effect_id=content_identity(
                        {
                            "task_id": task_id,
                            "operation": "assign",
                            "fluent_id": state_id,
                            "value": "completed",
                        }
                    ),
                    operation=EffectOperation.ASSIGN,
                    task_id=task_id,
                    fluent_id=state_id,
                    event_id=completed_event_id,
                    value="completed",
                    metadata={"compiler_rule": "task_completion_lifecycle"},
                )
            ]
        result: list[Effect] = []
        assignments: dict[tuple[str, str], Any] = {}
        for ordinal, value in enumerate(raw):
            if not isinstance(value, Mapping):
                raise ValueError(f"task {task_id} effect {ordinal} must be an object")
            item = _record(value)
            operation_text = _text(item, "operation", "op")
            if not operation_text:
                raise ValueError(f"task {task_id} effect {ordinal} has missing semantics")
            try:
                operation = EffectOperation(operation_text)
            except ValueError as exc:
                raise ValueError(
                    f"task {task_id} effect {ordinal} uses unsupported operation "
                    f"{operation_text!r}"
                ) from exc
            fluent_id = _text(item, "fluent_id") or (
                state_id if operation is not EffectOperation.EMIT else ""
            )
            event_id = _text(item, "event_id") or completed_event_id
            effect = Effect(
                effect_id=_text(item, "effect_id", "id")
                or content_identity(
                    {
                        "task_id": task_id,
                        "ordinal": ordinal,
                        "effect": item,
                    }
                ),
                operation=operation,
                task_id=task_id,
                fluent_id=fluent_id,
                event_id=event_id,
                value=item.get("value"),
                metadata={"source_effect": item},
            )
            if operation is EffectOperation.ASSIGN:
                key = (event_id, fluent_id)
                if key in assignments and assignments[key] != effect.value:
                    raise ContractValidationError(
                        f"conflicting effects assign {fluent_id} at {event_id}"
                    )
                assignments[key] = effect.value
            result.append(effect)
        return result

    @staticmethod
    def _source_graph(bundle: Mapping[str, Any]) -> PlanGraphProjection:
        nodes: list[dict[str, Any]] = []
        node_by_alias: dict[str, str] = {}
        section_kind = {
            "objectives": "goal",
            "tasks": "task",
            "ast": "ast_scope",
            "policies": "policy",
            "leases": "lease",
            "evidence": "evidence",
        }
        for section, kind in section_kind.items():
            for record in bundle.get(section, ()):
                sid = _source_id(record, section)
                node = _graph_node(kind, sid, source_cid=sid)
                nodes.append(node)
                node_by_alias[sid] = node["node_id"]
                for alias in (
                    _text(record, "id", "task_id", "goal_id", "policy_id", "lease_id"),
                    _text(record, "task_cid", "goal_cid", "policy_cid", "lease_cid"),
                ):
                    if alias:
                        node_by_alias[alias] = node["node_id"]
        edges: list[dict[str, Any]] = []
        for record in bundle.get("tasks", ()):
            task_node = node_by_alias.get(_source_id(record, "tasks"))
            if not task_node:
                continue
            goal_node = node_by_alias.get(_text(record, "goal_cid", "goal_id"))
            if goal_node:
                edges.append(_graph_edge("contributes_to", task_node, goal_node))
            for dependency in _strings(
                record.get("depends_on")
                or record.get("dependencies")
                or record.get("dependency_task_cids")
            ):
                target = node_by_alias.get(dependency)
                if target:
                    edges.append(_graph_edge("depends_on", task_node, target))
        return PlanGraphProjection(tuple(nodes), tuple(edges))

    @staticmethod
    def _plan_graph(
        plan: FormalWorkPlan,
        *,
        ast_records: Iterable[Mapping[str, Any]],
        policies: Iterable[Mapping[str, Any]],
        leases: Iterable[Mapping[str, Any]],
        evidence_records: Iterable[Mapping[str, Any]],
    ) -> PlanGraphProjection:
        nodes: list[dict[str, Any]] = []
        by_record: dict[str, str] = {}
        for kind, records, identity_name in (
            ("actor", plan.actors, "actor_id"),
            ("goal", plan.goals, "goal_id"),
            ("task", plan.tasks, "task_id"),
            ("event", plan.events, "event_id"),
            ("fluent", plan.fluents, "fluent_id"),
            ("precondition", plan.preconditions, "precondition_id"),
            ("effect", plan.effects, "effect_id"),
            ("norm", plan.norms, "norm_id"),
            ("temporal_constraint", plan.temporal_constraints, "constraint_id"),
            ("evidence_requirement", plan.evidence_requirements, "requirement_id"),
        ):
            for record in records:
                record_id = str(getattr(record, identity_name))
                node = _graph_node(
                    kind, record_id, content_id=record.content_id
                )
                nodes.append(node)
                by_record[record_id] = node["node_id"]
        formula_records = plan.metadata.get("formula_records", ())
        if isinstance(formula_records, Sequence) and not isinstance(
            formula_records, (str, bytes, bytearray)
        ):
            for raw in formula_records:
                if not isinstance(raw, Mapping):
                    continue
                formula_id = _text(raw, "formula_id", "content_id")
                if not formula_id:
                    continue
                node = _graph_node(
                    "predicate",
                    formula_id,
                    content_id=formula_id,
                    operator=raw.get("operator"),
                    predicate=raw.get("predicate"),
                )
                nodes.append(node)
                by_record[formula_id] = node["node_id"]
        for record, section, kind in (
            *((item, "ast", "ast_scope") for item in ast_records),
            *((item, "policies", "policy") for item in policies),
            *((item, "leases", "lease") for item in leases),
            *((item, "evidence", "evidence") for item in evidence_records),
        ):
            record_id = _source_id(record, section)
            node = _graph_node(kind, record_id, source_cid=record_id)
            nodes.append(node)
            by_record[record_id] = node["node_id"]

        edges: list[dict[str, Any]] = []
        for goal in plan.goals:
            edges.append(
                _graph_edge(
                    "owned_by",
                    by_record[goal.goal_id],
                    by_record[goal.owner_actor_id],
                )
            )
        for task in plan.tasks:
            edges.append(
                _graph_edge(
                    "contributes_to",
                    by_record[task.task_id],
                    by_record[task.goal_id],
                )
            )
            for actor_id in task.actor_ids:
                edges.append(
                    _graph_edge(
                        "assigned_to",
                        by_record[task.task_id],
                        by_record[actor_id],
                    )
                )
            for dependency in task.depends_on:
                edges.append(
                    _graph_edge(
                        "depends_on",
                        by_record[task.task_id],
                        by_record[dependency],
                    )
                )
            for requirement_id in task.evidence_requirement_ids:
                edges.append(
                    _graph_edge(
                        "requires_evidence",
                        by_record[task.task_id],
                        by_record[requirement_id],
                    )
                )
            for event_id in task.event_ids:
                edges.append(
                    _graph_edge(
                        "has_event",
                        by_record[task.task_id],
                        by_record[event_id],
                    )
                )
            for precondition_id in task.precondition_ids:
                edges.append(
                    _graph_edge(
                        "requires_precondition",
                        by_record[task.task_id],
                        by_record[precondition_id],
                    )
                )
            for effect_id in task.effect_ids:
                edges.append(
                    _graph_edge(
                        "has_effect",
                        by_record[task.task_id],
                        by_record[effect_id],
                    )
                )
            for scope_id in _strings(task.metadata.get("changed_ast_scope_ids")):
                if scope_id in by_record:
                    edges.append(
                        _graph_edge(
                            "changes_scope",
                            by_record[task.task_id],
                            by_record[scope_id],
                        )
                    )
        for precondition in plan.preconditions:
            if precondition.formula_id in by_record:
                edges.append(
                    _graph_edge(
                        "uses_predicate",
                        by_record[precondition.precondition_id],
                        by_record[precondition.formula_id],
                    )
                )
        for constraint in plan.temporal_constraints:
            if constraint.formula_id in by_record:
                edges.append(
                    _graph_edge(
                        "uses_predicate",
                        by_record[constraint.constraint_id],
                        by_record[constraint.formula_id],
                    )
                )
        policy_nodes = [
            by_record[_source_id(item, "policies")]
            for item in policies
            if _source_id(item, "policies") in by_record
        ]
        for task in plan.tasks:
            for policy_node in policy_nodes:
                edges.append(
                    _graph_edge(
                        "governed_by", by_record[task.task_id], policy_node
                    )
                )
        for lease in leases:
            task_ref = _text(lease, "task_cid", "task_id")
            lease_id = _source_id(lease, "leases")
            if task_ref in by_record and lease_id in by_record:
                edges.append(
                    _graph_edge(
                        "leased_by", by_record[task_ref], by_record[lease_id]
                    )
                )
        return PlanGraphProjection(tuple(nodes), tuple(edges))


def _decode_duckdb_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    if isinstance(value, tuple):
        return list(value)
    return value


def _optional_nonnegative(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("freshness_seconds must be a non-negative integer")
    return value


def _assurance(value: Any) -> AssuranceLevel:
    if isinstance(value, AssuranceLevel):
        return value
    try:
        return AssuranceLevel(str(value or AssuranceLevel.UNVERIFIED.value))
    except ValueError as exc:
        raise ValueError(f"unsupported assurance level {value!r}") from exc


def _evidence_kind(value: Any) -> EvidenceRequirementKind:
    if isinstance(value, EvidenceRequirementKind):
        return value
    aliases = {
        "pytest": "test",
        "validation": "test",
        "proof": "code_proof",
        "receipt": "artifact",
    }
    text = aliases.get(str(value).strip().lower(), str(value).strip().lower())
    try:
        return EvidenceRequirementKind(text)
    except ValueError as exc:
        raise ValueError(f"unsupported evidence requirement kind {value!r}") from exc


def _policy_defaults(policies: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    assurance = AssuranceLevel.UNVERIFIED
    freshness: int | None = None
    fallbacks: set[str] = set()
    for policy in policies:
        raw_assurance = policy.get(
            "minimum_code_assurance", policy.get("minimum_assurance")
        )
        if raw_assurance not in (None, ""):
            candidate = _assurance(raw_assurance)
            if candidate.rank > assurance.rank:
                assurance = candidate
        raw_freshness = policy.get("freshness_seconds")
        if raw_freshness not in (None, ""):
            candidate_freshness = _optional_nonnegative(raw_freshness)
            if freshness is None or (
                candidate_freshness is not None
                and candidate_freshness < freshness
            ):
                freshness = candidate_freshness
        fallbacks.update(
            _strings(
                policy.get("fallback_check_ids") or policy.get("fallback_checks")
            )
        )
    return {
        "minimum_assurance": assurance,
        "freshness_seconds": freshness,
        "fallback_checks": tuple(sorted(fallbacks)),
    }


def _descriptive_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in sorted(_ABSTRACTED_FIELDS):
        value = record.get(name)
        if value not in (None, "", {}, []):
            result[name] = _canonical_safe(value)
    return result


def _topological_times(
    tasks: Mapping[str, Mapping[str, Any]], aliases: Mapping[str, str]
) -> dict[str, int]:
    dependencies: dict[str, tuple[str, ...]] = {}
    for task_id, record in tasks.items():
        dependencies[task_id] = tuple(
            sorted(
                {
                    aliases.get(value, value)
                    for value in _strings(
                        record.get("depends_on")
                        or record.get("dependencies")
                        or record.get("dependency_task_cids")
                        or record.get("blocking_task_cids")
                    )
                }
            )
        )
    visiting: set[str] = set()
    levels: dict[str, int] = {}

    def level(task_id: str) -> int:
        if task_id in levels:
            return levels[task_id]
        if task_id in visiting:
            raise ContractValidationError("task dependencies must be acyclic")
        visiting.add(task_id)
        current = 0
        for dependency in dependencies[task_id]:
            if dependency not in tasks:
                raise ContractValidationError(
                    f"task {task_id} has unknown dependency {dependency}"
                )
            current = max(current, level(dependency) + 1)
        visiting.remove(task_id)
        levels[task_id] = current
        return current

    for task_id in sorted(tasks):
        level(task_id)
    # Tasks at a shared DAG level are ordered deterministically to prevent
    # lifecycle events from sharing an event identifier/time pair.
    result: dict[str, int] = {}
    ordinal = 0
    for _, task_id in sorted((levels[item], item) for item in levels):
        result[task_id] = ordinal
        ordinal += 1
    return result


def compile_formal_plan(
    source: Mapping[str, Any] | None = None, **records: Any
) -> PlanCompilationResult:
    """Functional entry point for :class:`FormalPlanCompiler`."""

    return FormalPlanCompiler().compile(source, **records)


def compile_formal_plan_json(source: str | bytes | Path) -> PlanCompilationResult:
    return FormalPlanCompiler().compile_json(source)


def compile_formal_plan_duckdb(
    source: Any, **records: Any
) -> PlanCompilationResult:
    return FormalPlanCompiler().compile_duckdb(source, **records)


def write_formal_plan_compiler_input_duckdb(
    path: str | Path, source: Mapping[str, Any]
) -> Path:
    """Write the lossless normalized DuckDB input used by equivalence tests/tools."""

    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise RuntimeError("duckdb is required to write compiler input") from exc
    bundle = _normalize_bundle(_record(source))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(target))
    try:
        connection.execute("DROP TABLE IF EXISTS formal_plan_input_records")
        connection.execute("DROP TABLE IF EXISTS formal_plan_input_metadata")
        connection.execute(
            "CREATE TABLE formal_plan_input_records "
            "(section VARCHAR, record_id VARCHAR, payload_json VARCHAR, "
            "PRIMARY KEY(section, record_id))"
        )
        connection.execute(
            "CREATE TABLE formal_plan_input_metadata "
            "(field_name VARCHAR PRIMARY KEY, field_value VARCHAR)"
        )
        rows = []
        for section in _KNOWN_FIELDS:
            for record in bundle[section]:
                rows.append(
                    (
                        section,
                        content_identity(record),
                        canonical_json(record),
                    )
                )
        if rows:
            connection.executemany(
                "INSERT INTO formal_plan_input_records VALUES (?, ?, ?)", rows
            )
        connection.execute(
            "INSERT INTO formal_plan_input_metadata VALUES (?, ?)",
            ("repository_tree_id", bundle["repository_tree_id"]),
        )
    finally:
        connection.close()
    return target


# Compatibility names make the outcome vocabulary discoverable to API callers.
FormalPlanCompilationResult = PlanCompilationResult
CompilerResult = PlanCompilationResult
CompilationResult = PlanCompilationResult
FormalWorkPlanCompiler = FormalPlanCompiler


__all__ = [
    "CompilationIssue",
    "CompilationIssueCode",
    "CompilationIssueSeverity",
    "CompilationResult",
    "CompilationStatus",
    "CompilerResult",
    "DEFAULT_VOCABULARY_PROFILE_ID",
    "FORMAL_PLAN_COMPILATION_SCHEMA",
    "FORMAL_PLAN_COMPILER_VERSION",
    "FORMAL_PLAN_GRAPH_SCHEMA",
    "FORMAL_PLAN_INPUT_SCHEMA",
    "FormalPlanCompilationResult",
    "FormalPlanCompiler",
    "FormalWorkPlanCompiler",
    "PlanCompilationResult",
    "PlanGraphProjection",
    "compile_formal_plan",
    "compile_formal_plan_duckdb",
    "compile_formal_plan_json",
    "write_formal_plan_compiler_input_duckdb",
]
