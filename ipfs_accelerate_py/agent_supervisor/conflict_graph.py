"""Conflict-surface prediction and explainable lane coloring.

The dependency graph answers *when* a task is ready.  This module answers the
orthogonal question of which ready tasks are safe to execute together.  It
models every declared work surface (not merely the first output path), augments
predictions with branch observations and merge-conflict receipts, and produces
a deterministic coloring with an explanation for every pair of tasks.

The data types deliberately serialize to ordinary JSON objects.  Supervisor
manifests can therefore retain the plan and its evidence without importing this
module when they are inspected by another process.
"""

from __future__ import annotations

import ast
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field, is_dataclass
from hashlib import sha1
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_SURFACE_WEIGHTS: dict[str, float] = {
    "files": 8.0,
    "changed_paths": 10.0,
    "ast_symbols": 2.0,
    "interfaces": 7.0,
    "submodules": 12.0,
    "generated_artifacts": 9.0,
}
CONFLICT_RECEIPT_STATUSES = frozenset(
    {"conflict", "conflicted", "failed", "merge_conflict", "quarantined", "rejected", "resolved"}
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


def _sources(value: Any) -> list[dict[str, Any]]:
    """Return top-level and common nested task metadata mappings."""

    root = _payload(value)
    found = [root]
    for key in ("finding", "metadata", "conflict_surface", "profile_g", "payload"):
        nested = root.get(key)
        if isinstance(nested, Mapping):
            found.append(dict(nested))
    return found


def _items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        # Symbols and paths may be supplied by markdown fields.
        return [item.strip() for item in re.split(r"[,;\n]", value) if item.strip()]
    if isinstance(value, Mapping):
        return [str(key).strip() for key, enabled in value.items() if enabled and str(key).strip()]
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _field_items(sources: Sequence[Mapping[str, Any]], names: Sequence[str]) -> list[str]:
    values: list[str] = []
    for source in sources:
        for name in names:
            values.extend(_items(source.get(name)))
    return values


def normalize_repo_path(value: str, *, repo_root: Path | None = None) -> str:
    """Normalize a repository path without requiring that it already exists."""

    text = str(value or "").strip().replace("\\", "/")
    if not text or "\0" in text:
        return ""
    path = Path(text)
    if path.is_absolute() and repo_root is not None:
        try:
            text = path.resolve().relative_to(Path(repo_root).resolve()).as_posix()
        except (OSError, ValueError):
            return ""
    elif path.is_absolute():
        return ""
    while text.startswith("./"):
        text = text[2:]
    parts: list[str] = []
    for part in PurePosixPath(text).parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                return ""
            parts.pop()
        else:
            parts.append(part)
    return "/".join(parts).rstrip("/")


def _normalized_paths(values: Iterable[str], repo_root: Path | None) -> list[str]:
    return sorted({path for value in values if (path := normalize_repo_path(value, repo_root=repo_root))})


def _normalized_terms(values: Iterable[str]) -> list[str]:
    return sorted({" ".join(str(value).strip().split()) for value in values if str(value).strip()})


def _gitmodule_paths(repo_root: Path | None) -> list[str]:
    if repo_root is None:
        return []
    path = Path(repo_root) / ".gitmodules"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    return _normalized_paths(
        (match.group(1).strip() for match in re.finditer(r"^\s*path\s*=\s*(.+?)\s*$", text, re.MULTILINE)),
        repo_root,
    )


def _under(path: str, parent: str) -> bool:
    return bool(path and parent and (path == parent or path.startswith(parent.rstrip("/") + "/")))


def _looks_generated(path: str) -> bool:
    parts = set(PurePosixPath(path).parts)
    name = PurePosixPath(path).name.lower()
    return bool(
        parts & {"build", "dist", "generated", "artifacts", "coverage", "playwright-report", "test-results"}
        or ".generated." in name
        or name.endswith((".min.js", ".lock", ".manifest.json"))
    )


@dataclass(frozen=True)
class ConflictSurface:
    """All predicted and observed mutation surfaces for one task."""

    task_id: str
    task_cid: str = ""
    files: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    ast_symbols: list[str] = field(default_factory=list)
    global_ast_symbols: list[str] | None = None
    interfaces: list[str] = field(default_factory=list)
    submodules: list[str] = field(default_factory=list)
    generated_artifacts: list[str] = field(default_factory=list)
    allow_concurrent_with: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_cid:
            object.__setattr__(self, "task_cid", self.task_id)
        if self.global_ast_symbols is None:
            object.__setattr__(self, "global_ast_symbols", list(self.ast_symbols))

    @property
    def all_paths(self) -> list[str]:
        return sorted(set(self.files) | set(self.changed_paths) | set(self.submodules) | set(self.generated_artifacts))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["all_paths"] = self.all_paths
        return payload


def _python_symbols(path: Path) -> set[str]:
    """Collect qualified Python definitions from a predicted existing file."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return set()
    symbols: set[str] = set()
    scope: list[str] = []

    class Visitor(ast.NodeVisitor):
        def _definition(self, node: ast.AST, name: str) -> None:
            symbols.add(name)
            if scope:
                symbols.add(".".join([*scope, name]))
            scope.append(name)
            self.generic_visit(node)
            scope.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._definition(node, node.name)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._definition(node, node.name)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._definition(node, node.name)

    Visitor().visit(tree)
    return symbols


def build_conflict_surface(
    task: Any,
    *,
    repo_root: Path | None = None,
    changed_paths: Sequence[str] | None = None,
) -> ConflictSurface:
    """Build a complete normalized conflict surface from a task-like object.

    The accepted aliases cover objective findings, todo-vector rows, proposal
    branches, and bundle payloads.  Unknown metadata is retained for audit but
    does not silently influence the coloring.
    """

    if isinstance(task, ConflictSurface):
        # Re-normalizing makes surfaces loaded from JSON as safe as live ones.
        task = task.to_dict()
    sources = _sources(task)
    root = sources[0]
    task_id = str(root.get("task_id") or root.get("id") or root.get("canonical_task_id") or "").strip()
    task_cid = str(
        root.get("task_cid")
        or root.get("canonical_task_cid")
        or next((source.get("task_cid") for source in sources[1:] if source.get("task_cid")), "")
        or task_id
    ).strip()
    if not task_id:
        task_id = task_cid
    if not task_id:
        digest = sha1(json.dumps(root, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
        task_id = task_cid = f"task-{digest}"

    files = _normalized_paths(
        _field_items(
            sources,
            ("files", "predicted_files", "predicted_paths", "outputs", "requested_outputs", "affected_files"),
        ),
        repo_root,
    )
    normalized_changed_paths = _normalized_paths(
        [
            *_field_items(sources, ("changed_paths", "actual_paths", "branch_diff", "diff_paths")),
            *(changed_paths or []),
        ],
        repo_root,
    )
    declared_ast_symbols = _normalized_terms(
        _field_items(sources, ("ast_symbols", "predicted_symbols", "symbols", "ast_query"))
    )
    has_explicit_global_symbols = any("global_ast_symbols" in source for source in sources)
    global_ast_symbols = (
        _normalized_terms(_field_items(sources, ("global_ast_symbols",)))
        if has_explicit_global_symbols
        else list(declared_ast_symbols)
    )
    ast_symbols = list(declared_ast_symbols)
    if repo_root is not None:
        discovered: set[str] = set(ast_symbols)
        for relative in files:
            if relative.endswith(".py"):
                discovered.update(_python_symbols(Path(repo_root) / relative))
        ast_symbols = sorted(discovered)
    interfaces = _normalized_terms(
        _field_items(
            sources,
            (
                "interfaces",
                "provides_interfaces",
                "requires_interfaces",
                "required_interfaces",
                "interface_dependencies",
                "public_interfaces",
            ),
        )
    )
    explicit_submodules = _normalized_paths(
        _field_items(sources, ("submodules", "submodule_paths", "gitlinks")), repo_root
    )
    known_submodules = _gitmodule_paths(repo_root)
    submodules = sorted(
        set(explicit_submodules)
        | {
            module
            for module in known_submodules
            if any(_under(path, module) for path in files + normalized_changed_paths)
        }
    )
    generated = _normalized_paths(
        _field_items(
            sources,
            ("generated_artifacts", "generated_paths", "artifacts", "generated_outputs", "derived_outputs"),
        ),
        repo_root,
    )
    generated = sorted(set(generated) | {path for path in files if _looks_generated(path)})
    allowed = _normalized_terms(
        _field_items(sources, ("allow_concurrent_with", "concurrency_overrides", "allowed_concurrent_tasks"))
    )
    return ConflictSurface(
        task_id=task_id,
        task_cid=task_cid,
        files=files,
        changed_paths=normalized_changed_paths,
        ast_symbols=ast_symbols,
        global_ast_symbols=global_ast_symbols,
        interfaces=interfaces,
        submodules=submodules,
        generated_artifacts=generated,
        allow_concurrent_with=allowed,
        metadata={
            key: value
            for key, value in root.items()
            if key
            not in {
                "files", "predicted_files", "outputs", "changed_paths", "ast_symbols",
                "global_ast_symbols", "interfaces",
                "submodules", "generated_artifacts", "allow_concurrent_with",
            }
        },
    )


def _merge_duplicate_surfaces(
    left: ConflictSurface,
    right: ConflictSurface,
) -> ConflictSurface:
    """Coalesce aliases that resolve to the same canonical task identity."""

    if left.task_cid != right.task_cid:
        raise ValueError("cannot merge conflict surfaces with different task CIDs")

    ordered = sorted(
        (left, right),
        key=lambda surface: (
            surface.task_id,
            json.dumps(surface.metadata, sort_keys=True, default=str),
        ),
    )
    representative = ordered[0]
    aliases = {
        left.task_id,
        right.task_id,
        *[str(value) for value in left.metadata.get("task_id_aliases", [])],
        *[str(value) for value in right.metadata.get("task_id_aliases", [])],
    }
    metadata = dict(representative.metadata)
    metadata["task_id_aliases"] = sorted(alias for alias in aliases if alias)
    return ConflictSurface(
        task_id=representative.task_id,
        task_cid=representative.task_cid,
        files=sorted(set(left.files) | set(right.files)),
        changed_paths=sorted(set(left.changed_paths) | set(right.changed_paths)),
        ast_symbols=sorted(set(left.ast_symbols) | set(right.ast_symbols)),
        global_ast_symbols=sorted(
            set(left.global_ast_symbols or []) | set(right.global_ast_symbols or [])
        ),
        interfaces=sorted(set(left.interfaces) | set(right.interfaces)),
        submodules=sorted(set(left.submodules) | set(right.submodules)),
        generated_artifacts=sorted(
            set(left.generated_artifacts) | set(right.generated_artifacts)
        ),
        allow_concurrent_with=sorted(
            set(left.allow_concurrent_with) | set(right.allow_concurrent_with)
        ),
        metadata=metadata,
    )


def _pair_key(left: str, right: str) -> str:
    return "\0".join(sorted((str(left), str(right))))


@dataclass
class ConflictWeightHistory:
    """Learned conflict weights accumulated from diffs and receipts."""

    path_weights: dict[str, float] = field(default_factory=dict)
    symbol_weights: dict[str, float] = field(default_factory=dict)
    interface_weights: dict[str, float] = field(default_factory=dict)
    submodule_weights: dict[str, float] = field(default_factory=dict)
    artifact_weights: dict[str, float] = field(default_factory=dict)
    pair_weights: dict[str, float] = field(default_factory=dict)
    observation_count: int = 0

    def observe_diff(self, task_cid: str, paths: Iterable[str], *, repo_root: Path | None = None) -> None:
        observed = _normalized_paths(paths, repo_root)
        for path in observed:
            self.path_weights[path] = self.path_weights.get(path, 0.0) + 1.0
        if observed:
            self.observation_count += 1

    def observe_receipt(self, receipt: Mapping[str, Any], *, repo_root: Path | None = None) -> None:
        left, right = _receipt_pair(receipt)
        severity = _receipt_severity(receipt)
        if left and right and severity:
            key = _pair_key(left, right)
            self.pair_weights[key] = self.pair_weights.get(key, 0.0) + severity
        for path in _receipt_paths(receipt, repo_root=repo_root):
            self.path_weights[path] = self.path_weights.get(path, 0.0) + max(1.0, severity)
        for symbol in _field_items([receipt], ("ast_symbols", "symbols", "conflicting_symbols")):
            self.symbol_weights[symbol] = self.symbol_weights.get(symbol, 0.0) + max(1.0, severity)
        if left or right or _receipt_paths(receipt, repo_root=repo_root):
            self.observation_count += 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ConflictWeightHistory":
        source = dict(payload or {})
        kwargs: dict[str, Any] = {}
        for name in ("path_weights", "symbol_weights", "interface_weights", "submodule_weights", "artifact_weights", "pair_weights"):
            value = source.get(name)
            kwargs[name] = {str(key): float(weight) for key, weight in value.items()} if isinstance(value, Mapping) else {}
        kwargs["observation_count"] = int(source.get("observation_count") or 0)
        return cls(**kwargs)

    @classmethod
    def load(cls, path: Path) -> "ConflictWeightHistory":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return cls()
        return cls.from_dict(payload if isinstance(payload, Mapping) else {})

    def write(self, path: Path) -> Path:
        """Atomically persist history so a killed planner cannot truncate it."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        return target


@dataclass(frozen=True)
class ConflictEdge:
    left_task_cid: str
    right_task_cid: str
    weight: float
    reasons: list[str]
    overlaps: dict[str, list[str]] = field(default_factory=dict)
    predicted_weight: float = 0.0
    observed_weight: float = 0.0
    explicitly_allowed: bool = False

    @property
    def left(self) -> str:
        return self.left_task_cid

    @property
    def right(self) -> str:
        return self.right_task_cid

    @property
    def blocks_concurrency(self) -> bool:
        return self.weight > 0 and not self.explicitly_allowed

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["blocks_concurrency"] = self.blocks_concurrency
        return payload


@dataclass(frozen=True)
class LaneAssignment:
    task_cid: str
    task_id: str
    lane: int
    color: int
    explanation: str

    @property
    def lane_color(self) -> int:
        """Compatibility name used by serialized lane planners."""

        return self.color

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lane_color"] = self.lane_color
        return payload


@dataclass(frozen=True)
class LaneDecision:
    left_task_cid: str
    right_task_cid: str
    action: str
    explanation: str
    weight: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskConflictGraph:
    surfaces: dict[str, ConflictSurface]
    edges: list[ConflictEdge]
    assignments: list[LaneAssignment]
    decisions: list[LaneDecision]
    lanes: dict[int, list[str]]
    history: ConflictWeightHistory = field(default_factory=ConflictWeightHistory)

    @property
    def colors(self) -> dict[str, int]:
        return {assignment.task_cid: assignment.color for assignment in self.assignments}

    def edge_for(self, left: str, right: str) -> ConflictEdge | None:
        pair = _pair_key(left, right)
        return next(
            (edge for edge in self.edges if _pair_key(edge.left_task_cid, edge.right_task_cid) == pair), None
        )

    def conflicts_for(self, task_cid: str) -> list[ConflictEdge]:
        return [
            edge for edge in self.edges
            if edge.blocks_concurrency and task_cid in {edge.left_task_cid, edge.right_task_cid}
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ipfs_accelerate_py.agent_supervisor.conflict_graph@1",
            "surfaces": {key: value.to_dict() for key, value in sorted(self.surfaces.items())},
            "edges": [edge.to_dict() for edge in self.edges],
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "lanes": {str(key): list(value) for key, value in sorted(self.lanes.items())},
            "history": self.history.to_dict(),
        }


# A concise alias used by callers that do not need the task qualifier.
ConflictGraph = TaskConflictGraph


@dataclass(frozen=True)
class SurfaceEvidenceEdge:
    """One explainable predicted-to-observed surface relationship.

    Coverage maps use these edges independently of conflict weights.  Keeping
    the original predicted and observed values is important: a directory
    prediction can cover a more specific changed path without pretending the
    two strings were an exact match.
    """

    dimension: str
    predicted: str
    observed: str
    relationship: str
    explanation: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SurfaceEvidenceComparison:
    """Deterministic comparison of planned and observed implementation work."""

    matched_paths: list[str] = field(default_factory=list)
    missing_paths: list[str] = field(default_factory=list)
    unexpected_paths: list[str] = field(default_factory=list)
    matched_symbols: list[str] = field(default_factory=list)
    missing_symbols: list[str] = field(default_factory=list)
    unexpected_symbols: list[str] = field(default_factory=list)
    matched_interfaces: list[str] = field(default_factory=list)
    missing_interfaces: list[str] = field(default_factory=list)
    unexpected_interfaces: list[str] = field(default_factory=list)
    evidence_edges: list[SurfaceEvidenceEdge] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)

    @property
    def matched(self) -> dict[str, list[str]]:
        return {
            "paths": list(self.matched_paths),
            "symbols": list(self.matched_symbols),
            "interfaces": list(self.matched_interfaces),
        }

    @property
    def missing(self) -> dict[str, list[str]]:
        return {
            "paths": list(self.missing_paths),
            "symbols": list(self.missing_symbols),
            "interfaces": list(self.missing_interfaces),
        }

    @property
    def unexpected(self) -> dict[str, list[str]]:
        return {
            "paths": list(self.unexpected_paths),
            "symbols": list(self.unexpected_symbols),
            "interfaces": list(self.unexpected_interfaces),
        }

    @property
    def predicted_count(self) -> int:
        matched_predictions = {
            (edge.dimension, edge.predicted) for edge in self.evidence_edges
        }
        return len(matched_predictions) + sum(len(values) for values in self.missing.values())

    @property
    def matched_count(self) -> int:
        return len({(edge.dimension, edge.predicted) for edge in self.evidence_edges})

    @property
    def coverage_ratio(self) -> float:
        """Return an order-independent ratio, with an empty plan fully covered."""

        return self.matched_count / self.predicted_count if self.predicted_count else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "missing": self.missing,
            "unexpected": self.unexpected,
            # Flat aliases make the record convenient for tabular/vector
            # indexes while the grouped fields remain the canonical shape.
            "matched_paths": list(self.matched_paths),
            "missing_paths": list(self.missing_paths),
            "unexpected_paths": list(self.unexpected_paths),
            "matched_symbols": list(self.matched_symbols),
            "missing_symbols": list(self.missing_symbols),
            "unexpected_symbols": list(self.unexpected_symbols),
            "matched_interfaces": list(self.matched_interfaces),
            "missing_interfaces": list(self.missing_interfaces),
            "unexpected_interfaces": list(self.unexpected_interfaces),
            "predicted_count": self.predicted_count,
            "matched_count": self.matched_count,
            "coverage_ratio": self.coverage_ratio,
            "evidence_edges": [edge.to_dict() for edge in self.evidence_edges],
            "explanations": list(self.explanations),
        }


@dataclass(frozen=True)
class SurfaceContradiction:
    """Strong evidence that a planned surface and observed evidence disagree."""

    dimension: str
    kind: str
    expected: str = ""
    observed: str = ""
    source: str = ""
    provenance_cid: str = ""
    explanation: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SurfaceContradictionReport:
    """Serializable contradiction result retaining the underlying comparison."""

    comparison: SurfaceEvidenceComparison
    contradictions: list[SurfaceContradiction] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)

    @property
    def contradicted(self) -> bool:
        return bool(self.contradictions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contradicted": self.contradicted,
            "contradictions": [item.to_dict() for item in self.contradictions],
            "explanations": list(self.explanations),
            "comparison": self.comparison.to_dict(),
        }


def _comparison_values(
    value: Any,
    *,
    observed: bool,
    repo_root: Path | None,
) -> dict[str, list[str]]:
    """Extract comparison dimensions without conflating plans with diffs."""

    if isinstance(value, ConflictSurface):
        if observed:
            paths = list(value.changed_paths)
            symbols = list(value.ast_symbols)
            interfaces = list(value.interfaces)
        else:
            paths = [*value.files, *value.submodules, *value.generated_artifacts]
            symbols = list(value.global_ast_symbols or value.ast_symbols)
            interfaces = list(value.interfaces)
        return {
            "paths": _normalized_paths(paths, repo_root),
            "symbols": _normalized_terms(symbols),
            "interfaces": _normalized_terms(interfaces),
        }

    sources = _sources(value)
    if observed:
        paths = _field_items(
            sources,
            (
                "changed_paths", "actual_paths", "observed_paths", "branch_diff", "diff_paths",
            ),
        )
        # ``files`` is the neutral spelling used by scan receipts, but on task
        # rows it commonly means a prediction.  Only use it when no
        # observation-specific path field exists.
        if not paths:
            paths = _field_items(sources, ("files",))
        symbols = _field_items(
            sources,
            (
                "changed_ast_symbols", "actual_ast_symbols", "observed_ast_symbols",
                "changed_symbols", "actual_symbols", "observed_symbols",
            ),
        )
        if not symbols:
            symbols = _field_items(sources, ("ast_symbols", "symbols"))
        interfaces = _field_items(
            sources,
            (
                "changed_interfaces", "actual_interfaces", "observed_interfaces",
            ),
        )
        if not interfaces:
            interfaces = _field_items(sources, ("interfaces", "provides_interfaces", "public_interfaces"))
    else:
        paths = _field_items(
            sources,
            (
                "files", "predicted_files", "predicted_paths", "outputs", "requested_outputs",
                "affected_files", "submodules", "submodule_paths", "gitlinks",
                "generated_artifacts", "generated_paths", "generated_outputs", "derived_outputs",
            ),
        )
        symbols = _field_items(
            sources,
            ("global_ast_symbols", "ast_symbols", "predicted_symbols", "symbols", "ast_query"),
        )
        interfaces = _field_items(
            sources,
            (
                "interfaces", "provides_interfaces", "requires_interfaces", "required_interfaces",
                "interface_dependencies", "public_interfaces",
            ),
        )
    return {
        "paths": _normalized_paths(paths, repo_root),
        "symbols": _normalized_terms(symbols),
        "interfaces": _normalized_terms(interfaces),
    }


def compare_surface_evidence(
    predicted: Any,
    observed: Any,
    *,
    repo_root: Path | None = None,
) -> SurfaceEvidenceComparison:
    """Compare predicted files/symbols/interfaces with observed evidence.

    Paths use the same containment semantics as conflict planning, so a
    predicted directory covers a changed descendant.  Symbols and interfaces
    require exact normalized names.  The returned order depends only on the
    values, never on mapping or input iteration order.
    """

    planned = _comparison_values(predicted, observed=False, repo_root=repo_root)
    actual = _comparison_values(observed, observed=True, repo_root=repo_root)
    edges: list[SurfaceEvidenceEdge] = []

    matched_paths: set[str] = set()
    covered_planned_paths: set[str] = set()
    for planned_path in planned["paths"]:
        for actual_path in actual["paths"]:
            if not (_under(planned_path, actual_path) or _under(actual_path, planned_path)):
                continue
            relationship = "exact" if planned_path == actual_path else "path_contains"
            matched_paths.add(actual_path)
            covered_planned_paths.add(planned_path)
            edges.append(
                SurfaceEvidenceEdge(
                    dimension="paths",
                    predicted=planned_path,
                    observed=actual_path,
                    relationship=relationship,
                    explanation=(
                        f"Observed path {actual_path!r} exactly matches predicted path {planned_path!r}."
                        if relationship == "exact"
                        else f"Observed path {actual_path!r} overlaps predicted path {planned_path!r} by containment."
                    ),
                )
            )

    dimension_results: dict[str, tuple[list[str], list[str], list[str]]] = {}
    for dimension in ("symbols", "interfaces"):
        matched = sorted(set(planned[dimension]) & set(actual[dimension]))
        missing = sorted(set(planned[dimension]) - set(actual[dimension]))
        unexpected = sorted(set(actual[dimension]) - set(planned[dimension]))
        dimension_results[dimension] = (matched, missing, unexpected)
        for term in matched:
            edges.append(
                SurfaceEvidenceEdge(
                    dimension=dimension,
                    predicted=term,
                    observed=term,
                    relationship="exact",
                    explanation=f"Observed {dimension[:-1]} {term!r} exactly matches the prediction.",
                )
            )

    missing_paths = sorted(set(planned["paths"]) - covered_planned_paths)
    unexpected_paths = sorted(
        path
        for path in actual["paths"]
        if not any(_under(path, planned_path) or _under(planned_path, path) for planned_path in planned["paths"])
    )
    edges.sort(key=lambda edge: (edge.dimension, edge.predicted, edge.observed, edge.relationship))

    results = {
        "paths": (sorted(matched_paths), missing_paths, unexpected_paths),
        **dimension_results,
    }
    explanations = [
        (
            f"{dimension}: {len(matched)} observed value(s) match, "
            f"{len(missing)} predicted value(s) lack evidence, and "
            f"{len(unexpected)} observed value(s) were not predicted."
        )
        for dimension, (matched, missing, unexpected) in results.items()
    ]
    return SurfaceEvidenceComparison(
        matched_paths=results["paths"][0],
        missing_paths=results["paths"][1],
        unexpected_paths=results["paths"][2],
        matched_symbols=results["symbols"][0],
        missing_symbols=results["symbols"][1],
        unexpected_symbols=results["symbols"][2],
        matched_interfaces=results["interfaces"][0],
        missing_interfaces=results["interfaces"][1],
        unexpected_interfaces=results["interfaces"][2],
        evidence_edges=edges,
        explanations=explanations,
    )


def _strong_evidence_records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, bool):
        # ``True`` is a concise assertion that the supplied observation is an
        # exhaustive inventory.  False contributes no evidence.
        return [{"coverage_complete": True, "source": "strong_evidence"}] if value else []
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def detect_surface_contradictions(
    predicted: Any,
    observed: Any | None = None,
    *,
    repo_root: Path | None = None,
    unexpected_is_contradiction: bool = False,
    missing_is_contradiction: bool = False,
    strong_evidence: Any = None,
    receipts: Any = None,
) -> SurfaceContradictionReport:
    """Promote comparison discrepancies only when policy or evidence warrants it.

    By default, missing predictions are coverage gaps and extra changed files
    are useful dynamic findings, not contradictions.  Callers may request an
    exact contract with the two boolean flags.  Strong records/receipts can
    also assert ``contradictory``/``contradiction`` or an exhaustive inventory
    (``coverage_complete``/``exhaustive``), in which case missing planned
    surfaces become contradictory.  Receipt provenance is retained verbatim.
    """

    if isinstance(predicted, SurfaceEvidenceComparison):
        if observed is not None:
            raise ValueError("observed must be omitted when predicted is already a comparison")
        comparison = predicted
    else:
        comparison = compare_surface_evidence(predicted, observed or {}, repo_root=repo_root)

    contradictions: list[SurfaceContradiction] = []

    def add(
        dimension: str,
        kind: str,
        *,
        expected: str = "",
        actual: str = "",
        source: str = "policy",
        provenance_cid: str = "",
        explanation: str,
    ) -> None:
        contradictions.append(
            SurfaceContradiction(
                dimension=dimension,
                kind=kind,
                expected=expected,
                observed=actual,
                source=source,
                provenance_cid=provenance_cid,
                explanation=explanation,
            )
        )

    if missing_is_contradiction:
        for dimension, values in comparison.missing.items():
            for value in values:
                add(
                    dimension,
                    "missing_expected_surface",
                    expected=value,
                    explanation=f"Exact-surface policy requires predicted {dimension[:-1]} {value!r}, but it was not observed.",
                )
    if unexpected_is_contradiction:
        for dimension, values in comparison.unexpected.items():
            for value in values:
                add(
                    dimension,
                    "unexpected_observed_surface",
                    actual=value,
                    explanation=f"Exact-surface policy rejects unpredicted observed {dimension[:-1]} {value!r}.",
                )

    records = [*_strong_evidence_records(strong_evidence), *_receipts(receipts)]
    records.sort(key=lambda item: json.dumps(item, sort_keys=True, default=str))
    for record in records:
        provenance = str(
            record.get("provenance_cid") or record.get("receipt_cid") or record.get("cid") or ""
        ).strip()
        source = str(record.get("source") or record.get("producer_id") or "evidence").strip()
        exhaustive = record.get("coverage_complete") is True or record.get("exhaustive") is True
        if exhaustive:
            for dimension, values in comparison.missing.items():
                for value in values:
                    add(
                        dimension,
                        "missing_from_exhaustive_evidence",
                        expected=value,
                        source=source,
                        provenance_cid=provenance,
                        explanation=(
                            f"Exhaustive evidence {provenance or source!r} does not contain predicted "
                            f"{dimension[:-1]} {value!r}."
                        ),
                    )
        exact_surface = (
            record.get("exact_surface") is True
            or record.get("unexpected_is_contradiction") is True
            or record.get("reject_unexpected") is True
        )
        if exact_surface:
            for dimension, values in comparison.unexpected.items():
                for value in values:
                    add(
                        dimension,
                        "unexpected_in_exact_evidence",
                        actual=value,
                        source=source,
                        provenance_cid=provenance,
                        explanation=(
                            f"Exact-surface evidence {provenance or source!r} rejects unpredicted "
                            f"observed {dimension[:-1]} {value!r}."
                        ),
                    )

        reason = str(record.get("contradiction") or record.get("reason") or "").strip()
        status = str(record.get("status") or record.get("outcome") or "").strip().lower()
        explicit = record.get("contradictory") is True or bool(record.get("contradiction"))
        explicit = explicit or status in {
            "contradicted", "contradictory", "conflict", "conflicted", "failed", "failure", "invalid", "rejected",
        }
        specific: list[tuple[str, str]] = []
        for dimension in ("paths", "symbols", "interfaces"):
            singular = dimension[:-1]
            values = _items(
                record.get(f"contradicted_{dimension}")
                or record.get(f"conflicting_{dimension}")
                or record.get(f"invalid_{dimension}")
                or record.get(f"contradicted_{singular}")
            )
            specific.extend((dimension, value) for value in values)
        if explicit and specific:
            for dimension, value in sorted(set(specific)):
                add(
                    dimension,
                    "explicit_evidence_contradiction",
                    actual=value,
                    source=source,
                    provenance_cid=provenance,
                    explanation=reason or f"Evidence explicitly contradicts {dimension[:-1]} {value!r}.",
                )
        elif explicit:
            add(
                "evidence",
                "explicit_evidence_contradiction",
                source=source,
                provenance_cid=provenance,
                explanation=reason or f"Evidence status {status!r} explicitly reports a contradiction.",
            )

    # Multiple strong records may report the same fact.  Collapse them without
    # losing different provenance, and make serialization independent of input
    # order.
    unique = {
        (
            item.dimension, item.kind, item.expected, item.observed,
            item.source, item.provenance_cid, item.explanation,
        ): item
        for item in contradictions
    }
    ordered = [unique[key] for key in sorted(unique)]
    explanations = (
        [item.explanation for item in ordered]
        if ordered
        else [
            "No contradiction was established; missing predictions remain coverage gaps and unexpected observations remain dynamic findings."
        ]
    )
    return SurfaceContradictionReport(
        comparison=comparison,
        contradictions=ordered,
        explanations=explanations,
    )


def _path_overlaps(left: Iterable[str], right: Iterable[str]) -> list[str]:
    overlaps: set[str] = set()
    for left_path in left:
        for right_path in right:
            if _under(left_path, right_path) or _under(right_path, left_path):
                # Retain the more specific path; it is the actionable surface.
                overlaps.add(left_path if len(left_path) >= len(right_path) else right_path)
    return sorted(overlaps)


def _history_path_weight(paths: Iterable[str], history: ConflictWeightHistory) -> float:
    total = 0.0
    for path in paths:
        total += sum(weight for known, weight in history.path_weights.items() if _under(path, known) or _under(known, path))
    return total


def _is_allowed(left: ConflictSurface, right: ConflictSurface, overrides: set[str]) -> bool:
    candidates = {left.task_cid, left.task_id}
    others = {right.task_cid, right.task_id}
    if any(_pair_key(left_key, right_key) in overrides for left_key in candidates for right_key in others):
        return True
    return bool(candidates & set(right.allow_concurrent_with) or others & set(left.allow_concurrent_with))


def _override_pairs(value: Any) -> set[str]:
    pairs: set[str] = set()
    if isinstance(value, Mapping):
        iterable: Iterable[Any] = value.items()
    else:
        iterable = value or []
    for item in iterable:
        left = right = ""
        enabled = True
        if isinstance(item, Mapping):
            left = str(item.get("left") or item.get("left_task_cid") or item.get("task") or "")
            right = str(item.get("right") or item.get("right_task_cid") or item.get("with") or "")
            enabled = bool(item.get("allowed", item.get("allow", True)))
        elif isinstance(item, tuple) and len(item) == 2:
            if isinstance(item[1], bool):
                # Mapping.items() with {"A\0B": True} or {(A, B): True}.
                key = item[0]
                if isinstance(key, (tuple, list, set, frozenset)) and len(key) == 2:
                    left, right = map(str, key)
                else:
                    parts = re.split(r"\s*(?:<->|::|,|\||\x00)\s*", str(key), maxsplit=1)
                    if len(parts) == 2:
                        left, right = parts
                enabled = item[1]
            else:
                left, right = str(item[0]), str(item[1])
        elif isinstance(item, (list, set, frozenset)) and len(item) == 2:
            left, right = map(str, item)
        elif isinstance(item, str):
            parts = re.split(r"\s*(?:<->|::|,|\|)\s*", item, maxsplit=1)
            if len(parts) == 2:
                left, right = parts
        if left and right and enabled:
            pairs.add(_pair_key(left, right))
    return pairs


def _receipt_pair(receipt: Mapping[str, Any]) -> tuple[str, str]:
    left = str(
        receipt.get("left_task_cid") or receipt.get("source_task_cid") or receipt.get("task_cid")
        or receipt.get("left_task_id") or ""
    ).strip()
    right = str(
        receipt.get("right_task_cid") or receipt.get("target_task_cid") or receipt.get("other_task_cid")
        or receipt.get("right_task_id") or ""
    ).strip()
    pair = receipt.get("task_cids") or receipt.get("tasks")
    if (not left or not right) and isinstance(pair, Sequence) and not isinstance(pair, str) and len(pair) >= 2:
        left, right = str(pair[0]), str(pair[1])
    return left, right


def _receipt_severity(receipt: Mapping[str, Any]) -> float:
    explicit = receipt.get("weight") or receipt.get("conflict_weight") or receipt.get("severity_weight")
    if explicit is not None:
        try:
            base = max(0.0, float(explicit))
            try:
                return base * max(1, int(receipt.get("count") or 1))
            except (TypeError, ValueError):
                return base
        except (TypeError, ValueError):
            pass
    status = " ".join(
        str(receipt.get(key) or "").lower()
        for key in ("status", "reason", "result", "outcome", "stderr")
    )
    if any(marker in status for marker in CONFLICT_RECEIPT_STATUSES):
        try:
            return 5.0 * max(1, int(receipt.get("count") or 1))
        except (TypeError, ValueError):
            return 5.0
    return 0.0


def _receipt_paths(receipt: Mapping[str, Any], *, repo_root: Path | None) -> list[str]:
    return _normalized_paths(
        _field_items(
            [receipt],
            ("paths", "conflicting_paths", "changed_paths", "files", "overlapping_paths", "generated_artifacts"),
        ),
        repo_root,
    )


def _receipts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        # A single receipt has characteristic fields; otherwise values are a receipt map.
        if set(value) & {
            "left_task_cid", "source_task_cid", "task_cids", "status", "outcome",
            "conflicting_paths", "contradictory", "contradiction", "receipt_cid",
            "coverage_complete", "exhaustive", "exact_surface", "reject_unexpected",
        }:
            return [dict(value)]
        receipts: list[dict[str, Any]] = []
        for key, item in value.items():
            if not isinstance(item, Mapping):
                continue
            receipt = dict(item)
            if not any(_receipt_pair(receipt)):
                parts = re.split(r"\s*(?:<->|::|,|\||\x00)\s*", str(key), maxsplit=1)
                if len(parts) == 2:
                    receipt["task_cids"] = parts
            receipts.append(receipt)
        return receipts
    return [dict(item) for item in (value or []) if isinstance(item, Mapping)]


def _branch_diff_map(value: Any) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if isinstance(value, Mapping):
        for key, paths in value.items():
            if isinstance(paths, Mapping):
                paths = paths.get("changed_paths") or paths.get("paths") or paths.get("files") or []
            result[str(key)] = _items(paths)
        return result
    for record in value or []:
        if not isinstance(record, Mapping):
            continue
        task = str(record.get("task_cid") or record.get("task_id") or "")
        if task:
            result.setdefault(task, []).extend(
                _items(record.get("changed_paths") or record.get("paths") or record.get("files"))
            )
    return result


def _make_edge(
    left: ConflictSurface,
    right: ConflictSurface,
    *,
    history: ConflictWeightHistory,
    allowed: bool,
    weights: Mapping[str, float],
) -> ConflictEdge | None:
    overlaps: dict[str, list[str]] = {}
    predicted = 0.0
    observed = 0.0

    # All path categories are compared cross-category; a generated artifact
    # declared by one task still conflicts with the same path declared as a file
    # by another task.
    path_groups = {
        "files": (left.files, right.files),
        "changed_paths": (left.changed_paths, right.changed_paths),
        "submodules": (left.submodules, right.submodules),
        "generated_artifacts": (left.generated_artifacts, right.generated_artifacts),
    }
    all_path_overlap = _path_overlaps(left.all_paths, right.all_paths)
    if all_path_overlap:
        overlaps["paths"] = all_path_overlap
    for surface, (left_values, right_values) in path_groups.items():
        shared = _path_overlaps(left_values, right_values)
        if shared:
            overlaps[surface] = shared
            contribution = float(weights[surface]) * len(shared)
            if surface == "changed_paths":
                observed += contribution
            else:
                predicted += contribution
    # Cross-category path overlaps not counted above remain a real conflict.
    counted_paths = {item for key, values in overlaps.items() if key != "paths" for item in values}
    cross_paths = sorted(set(all_path_overlap) - counted_paths)
    if cross_paths:
        overlaps["cross_surface_paths"] = cross_paths
        predicted += float(weights["files"]) * len(cross_paths)

    ast_symbol_values = (
        (left.ast_symbols, right.ast_symbols)
        if all_path_overlap
        else (left.global_ast_symbols or [], right.global_ast_symbols or [])
    )
    for surface, left_values, right_values in (
        ("ast_symbols", *ast_symbol_values),
        ("interfaces", left.interfaces, right.interfaces),
    ):
        shared = sorted(set(left_values) & set(right_values))
        if shared:
            overlaps[surface] = shared
            predicted += float(weights[surface]) * len(shared)

    history_paths = all_path_overlap or _path_overlaps(left.files + left.generated_artifacts, right.files + right.generated_artifacts)
    path_history = _history_path_weight(history_paths, history)
    symbol_history = sum(history.symbol_weights.get(symbol, 0.0) for symbol in overlaps.get("ast_symbols", []))
    interface_history = sum(history.interface_weights.get(name, 0.0) for name in overlaps.get("interfaces", []))
    pair_history = history.pair_weights.get(_pair_key(left.task_cid, right.task_cid), 0.0)
    observed += path_history + symbol_history + interface_history + pair_history
    if pair_history:
        overlaps["historical_task_pair"] = [f"{left.task_cid}<->{right.task_cid}"]

    weight = predicted + observed
    if weight <= 0:
        return None
    reason_names = {
        "files": "file",
        "changed_paths": "changed_path",
        "ast_symbols": "ast_symbol",
        "interfaces": "interface",
        "submodules": "submodule",
        "generated_artifacts": "generated_artifact",
        "cross_surface_paths": "file",
        "historical_task_pair": "conflict_receipt",
        "paths": "path",
    }
    reasons: list[str] = []
    for surface in sorted(overlaps):
        code = reason_names.get(surface, surface)
        if code not in reasons:
            reasons.append(code)
    # Keep concrete evidence alongside stable reason codes.  Codes make policy
    # filtering reliable; details make receipts and manifests self-explanatory.
    reasons.extend(f"{surface}: {', '.join(values)}" for surface, values in sorted(overlaps.items()))
    if observed:
        reasons.append(f"observed conflict evidence: +{observed:g}")
    if allowed:
        reasons.append("explicit concurrency override")
    return ConflictEdge(
        left_task_cid=left.task_cid,
        right_task_cid=right.task_cid,
        weight=weight,
        reasons=reasons,
        overlaps=overlaps,
        predicted_weight=predicted,
        observed_weight=observed,
        explicitly_allowed=allowed,
    )


def _color(
    surfaces: Mapping[str, ConflictSurface], edges: Sequence[ConflictEdge], *, max_lanes: int | None
) -> tuple[list[LaneAssignment], dict[int, list[str]]]:
    adjacency: dict[str, set[str]] = {task_cid: set() for task_cid in surfaces}
    weighted_degree: dict[str, float] = {task_cid: 0.0 for task_cid in surfaces}
    for edge in edges:
        if not edge.blocks_concurrency:
            continue
        adjacency[edge.left_task_cid].add(edge.right_task_cid)
        adjacency[edge.right_task_cid].add(edge.left_task_cid)
        weighted_degree[edge.left_task_cid] += edge.weight
        weighted_degree[edge.right_task_cid] += edge.weight

    # Deterministic DSATUR: most constrained first, then weighted degree and CID.
    colors: dict[str, int] = {}
    capacity = int(max_lanes) if max_lanes is not None and int(max_lanes) > 0 else None
    while len(colors) < len(surfaces):
        uncolored = [node for node in surfaces if node not in colors]
        node = min(
            uncolored,
            key=lambda item: (
                -len({colors[neighbor] for neighbor in adjacency[item] if neighbor in colors}),
                -len(adjacency[item]),
                -weighted_degree[item],
                item,
            ),
        )
        forbidden = {colors[neighbor] for neighbor in adjacency[node] if neighbor in colors}
        color = 0
        while color in forbidden or (capacity is not None and sum(value == color for value in colors.values()) >= capacity):
            color += 1
        colors[node] = color

    lanes: dict[int, list[str]] = {}
    for task_cid, color in sorted(colors.items(), key=lambda item: (item[1], item[0])):
        lanes.setdefault(color, []).append(task_cid)
    assignments = [
        LaneAssignment(
            task_cid=task_cid,
            task_id=surfaces[task_cid].task_id,
            lane=color,
            color=color,
            explanation=(
                f"color {color} avoids {len(adjacency[task_cid])} blocking conflict(s)"
                if adjacency[task_cid]
                else f"color {color} co-locates this task with non-overlapping work"
            ),
        )
        for task_cid, color in sorted(colors.items(), key=lambda item: (item[1], item[0]))
    ]
    return assignments, lanes


def _explain_decisions(
    surfaces: Mapping[str, ConflictSurface],
    edges: Sequence[ConflictEdge],
    assignments: Sequence[LaneAssignment],
) -> list[LaneDecision]:
    """Explain every pair against the supplied (possibly recolored) plan."""

    task_cids = sorted(surfaces)
    colors = {assignment.task_cid: assignment.color for assignment in assignments}
    edge_by_pair = {_pair_key(edge.left_task_cid, edge.right_task_cid): edge for edge in edges}
    decisions: list[LaneDecision] = []
    for index, left in enumerate(task_cids):
        for right in task_cids[index + 1 :]:
            edge = edge_by_pair.get(_pair_key(left, right))
            if edge is not None and edge.explicitly_allowed:
                action = "concurrent_override"
                explanation = (
                    f"Explicit override permits concurrency despite weight {edge.weight:g}: "
                    + "; ".join(edge.reasons)
                )
            elif edge is not None:
                action = "separate"
                explanation = (
                    f"Separated into colors {colors[left]} and {colors[right]} for conflict weight {edge.weight:g}: "
                    + "; ".join(edge.reasons)
                )
            elif colors[left] == colors[right]:
                action = "co_locate"
                explanation = f"Co-located in color {colors[left]} because no conflict surface overlaps."
            else:
                action = "separate"
                explanation = (
                    f"No conflict surface overlaps; colors {colors[left]} and {colors[right]} differ only because "
                    "of capacity or their conflicts with other tasks."
                )
            decisions.append(
                LaneDecision(
                    left_task_cid=left,
                    right_task_cid=right,
                    action=action,
                    explanation=explanation,
                    weight=edge.weight if edge else 0.0,
                    reasons=list(edge.reasons) if edge else (["lane_capacity"] if colors[left] != colors[right] else []),
                )
            )
    return decisions


def materialize_task_conflict_graph(
    tasks: Sequence[Any],
    *,
    repo_root: Path | None = None,
    branch_diffs: Any = None,
    conflict_receipts: Any = None,
    concurrency_overrides: Any = None,
    history: ConflictWeightHistory | Mapping[str, Any] | None = None,
    max_lanes: int | None = None,
    surface_weights: Mapping[str, float] | None = None,
) -> TaskConflictGraph:
    """Materialize conflict edges, deterministic colors, and pair decisions.

    ``max_lanes`` is treated as concurrent capacity per color, never as
    permission to place conflicting work together.  Additional colors are
    created when either conflicts or capacity require them.
    """

    learned = history if isinstance(history, ConflictWeightHistory) else ConflictWeightHistory.from_dict(history)
    diffs = _branch_diff_map(branch_diffs)
    surfaces: dict[str, ConflictSurface] = {}
    observed_paths_by_cid: dict[str, set[str]] = {}
    for task in tasks:
        surface = build_conflict_surface(task, repo_root=repo_root)
        observed_paths = diffs.get(surface.task_cid, []) + diffs.get(surface.task_id, [])
        if observed_paths:
            normalized = _normalized_paths([*surface.changed_paths, *observed_paths], repo_root)
            surface = ConflictSurface(**{**asdict(surface), "changed_paths": normalized})
            observed_paths_by_cid.setdefault(surface.task_cid, set()).update(
                _normalized_paths(observed_paths, repo_root)
            )
        if surface.task_cid in surfaces:
            surface = _merge_duplicate_surfaces(surfaces[surface.task_cid], surface)
        surfaces[surface.task_cid] = surface

    for task_cid, observed_paths in observed_paths_by_cid.items():
        learned.observe_diff(task_cid, observed_paths, repo_root=repo_root)

    receipts = _receipts(conflict_receipts)
    identity_aliases = {
        identity: surface.task_cid
        for surface in surfaces.values()
        for identity in {
            surface.task_id,
            surface.task_cid,
            *[str(value) for value in surface.metadata.get("task_id_aliases", [])],
        }
    }
    for receipt in receipts:
        left, right = _receipt_pair(receipt)
        normalized_receipt = dict(receipt)
        if left and right:
            normalized_receipt["task_cids"] = [
                identity_aliases.get(left, left),
                identity_aliases.get(right, right),
            ]
            # Ensure the canonical pair wins over any alias-specific fields.
            for key in (
                "left_task_cid", "source_task_cid", "task_cid", "left_task_id",
                "right_task_cid", "target_task_cid", "other_task_cid", "right_task_id",
                "tasks",
            ):
                normalized_receipt.pop(key, None)
        learned.observe_receipt(normalized_receipt, repo_root=repo_root)
    overrides = _override_pairs(concurrency_overrides)
    weights = {**DEFAULT_SURFACE_WEIGHTS, **dict(surface_weights or {})}

    task_cids = sorted(surfaces)
    edges: list[ConflictEdge] = []
    for index, left_cid in enumerate(task_cids):
        for right_cid in task_cids[index + 1 :]:
            left, right = surfaces[left_cid], surfaces[right_cid]
            edge = _make_edge(
                left,
                right,
                history=learned,
                allowed=_is_allowed(left, right, overrides),
                weights=weights,
            )
            if edge is not None:
                edges.append(edge)
    edges.sort(key=lambda edge: (-edge.weight, edge.left_task_cid, edge.right_task_cid))
    assignments, lanes = _color(surfaces, edges, max_lanes=max_lanes)
    decisions = _explain_decisions(surfaces, edges, assignments)
    return TaskConflictGraph(
        surfaces=surfaces,
        edges=edges,
        assignments=assignments,
        decisions=decisions,
        lanes=lanes,
        history=learned,
    )


def build_conflict_graph(tasks: Sequence[Any], **kwargs: Any) -> TaskConflictGraph:
    """Compatibility alias for :func:`materialize_task_conflict_graph`."""

    return materialize_task_conflict_graph(tasks, **kwargs)


def color_conflict_graph(
    graph: TaskConflictGraph | Sequence[Any], *, max_lanes: int | None = None, **kwargs: Any
) -> TaskConflictGraph:
    """Color an existing graph or materialize and color task-like inputs."""

    if not isinstance(graph, TaskConflictGraph):
        return materialize_task_conflict_graph(graph, max_lanes=max_lanes, **kwargs)
    assignments, lanes = _color(graph.surfaces, graph.edges, max_lanes=max_lanes)
    return TaskConflictGraph(
        surfaces=graph.surfaces,
        edges=graph.edges,
        assignments=assignments,
        decisions=_explain_decisions(graph.surfaces, graph.edges, assignments),
        lanes=lanes,
        history=graph.history,
    )


def update_conflict_weights(
    history: ConflictWeightHistory,
    *,
    branch_diffs: Any = None,
    conflict_receipts: Any = None,
    repo_root: Path | None = None,
) -> ConflictWeightHistory:
    """Apply observations to a reusable history object and return it."""

    for task_cid, paths in _branch_diff_map(branch_diffs).items():
        history.observe_diff(task_cid, paths, repo_root=repo_root)
    for receipt in _receipts(conflict_receipts):
        history.observe_receipt(receipt, repo_root=repo_root)
    return history


__all__ = [
    "ConflictEdge",
    "ConflictGraph",
    "ConflictSurface",
    "ConflictWeightHistory",
    "LaneAssignment",
    "LaneDecision",
    "SurfaceContradiction",
    "SurfaceContradictionReport",
    "SurfaceEvidenceComparison",
    "SurfaceEvidenceEdge",
    "TaskConflictGraph",
    "build_conflict_graph",
    "build_conflict_surface",
    "color_conflict_graph",
    "compare_surface_evidence",
    "detect_surface_contradictions",
    "materialize_task_conflict_graph",
    "normalize_repo_path",
    "update_conflict_weights",
]
