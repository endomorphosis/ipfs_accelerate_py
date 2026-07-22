"""Objective-graph scanner and bundle planner for autonomous agent todos.

This module ports the objective-driven backlog generation that was previously
implemented as repository scripts into ``ipfs_accelerate_py``.  It is designed
for acceleration-oriented agent systems: one objective heap can generate many
bundle-local todo shards, and those shards can be submitted to the existing P2P
task queue so multiple Codex workers can drain independent lanes.
"""

from __future__ import annotations

import ast
import heapq
import json
import math
import os
import re
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .dataset_store import DatasetArtifact, ObjectiveDatasetStore
from .scan_receipts import RefillScanResult, ScanTerminalReason, build_scan_result
from .task_identity import TaskIdentity, canonical_bundle_identity, canonical_task_identity
from .taskboard_store import (
    locked_taskboard,
    replace_locked_taskboard,
    task_ids_from_artifact_names,
)


DEFAULT_EMBEDDING_DIMENSIONS = int(os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_EMBEDDING_DIMENSIONS", "64"))
DEFAULT_EMBEDDING_MIN_SCORE = float(os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_EMBEDDING_MIN_SCORE", "0.62"))
DEFAULT_BUNDLE_CLUSTER_MIN_SCORE = float(os.environ.get("IPFS_ACCELERATE_AGENT_BUNDLE_CLUSTER_MIN_SCORE", "0.42"))
DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX = os.environ.get(
    "IPFS_ACCELERATE_AGENT_OBJECTIVE_TASK_SUMMARY_PREFIX",
    "Close objective gap",
)


def parse_python_ast_quietly(text: str) -> ast.AST:
    """Parse Python source without surfacing scanner-only syntax warnings."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        return ast.parse(text)


DEFAULT_DISCOVERY_OUTPUT_PATH = os.environ.get(
    "IPFS_ACCELERATE_AGENT_DISCOVERY_OUTPUT_PATH",
    "data/agent_supervisor/discovery",
)
DEFAULT_SURPLUS_FINDINGS_PER_GOAL = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_SURPLUS_FINDINGS_PER_GOAL", "3")
)
DEFAULT_SURPLUS_MIN_TERMS_PER_TODO = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_SURPLUS_MIN_TERMS_PER_TODO", "3")
)
DEFAULT_SCAN_OVERSAMPLE_MULTIPLIER = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_SCAN_OVERSAMPLE_MULTIPLIER", "2")
)
DEFAULT_TASK_PREFIX = "AUTO-"
OBJECTIVE_SCAN_ANALYZER_VERSION = "objective-gap-analyzer/v1"
DEFAULT_AST_DATASET_MAX_CHARS = int(os.environ.get("IPFS_ACCELERATE_AGENT_AST_DATASET_MAX_CHARS", "1000000"))
AST_DATASET_RECORD_SCHEMA_VERSION = 2
LAUNCH_PLAYWRIGHT_VALIDATION_COMMAND = (
    "(test ! -f swissknife/package.json || npm --prefix swissknife run test:e2e:meta-glasses) && "
    "(test ! -f hallucinate_app/package.json || "
    "npm --prefix hallucinate_app run test:e2e -- multimodal-control-surface.spec.ts)"
)
LAUNCH_PLAYWRIGHT_VALIDATION_MARKERS = (
    "test:e2e:meta-glasses",
    "meta-glasses-virtual-os.spec.ts",
    "multimodal-control-surface.spec.ts",
)
LAUNCH_PLAYWRIGHT_VALIDATION_GATE_EVIDENCE = "launch Playwright validation gate"
SCAN_SUFFIXES = {
    ".cjs",
    ".css",
    ".html",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".mjs",
    ".py",
    ".rs",
    ".sh",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "playwright-report",
    "test-results",
}


@dataclass(frozen=True)
class ObjectiveGoal:
    """One markdown objective-heap node."""

    goal_id: str
    title: str
    fields: dict[str, str] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return str(self.fields.get("status") or "active").strip().lower()

    @property
    def lifecycle_state(self) -> Any:
        """Return this goal's canonical :class:`GoalState`.

        The import is intentionally local.  ``goal_completion`` consumes the
        objective parser defined in this module, so a module-level import would
        create a cycle.  Keeping normalization at this boundary lets older
        objective heaps continue to use labels such as ``open`` or
        ``completed`` while every graph consumer sees the six-state lifecycle.
        """

        from .goal_completion import normalize_goal_state

        return normalize_goal_state(self.status)

    @property
    def lifecycle_state_value(self) -> str:
        """Return the serializable value of :attr:`lifecycle_state`."""

        state = self.lifecycle_state
        return str(getattr(state, "value", state))

    @property
    def is_schedulable(self) -> bool:
        """Whether new implementation work may be generated for this goal."""

        from .goal_completion import is_schedulable_goal_state

        return is_schedulable_goal_state(self.lifecycle_state)

    @property
    def is_terminal(self) -> bool:
        """Whether the goal has reached the sole terminal lifecycle state."""

        from .goal_completion import is_terminal_goal_state

        return is_terminal_goal_state(self.lifecycle_state)

    @property
    def completion_evidence_metadata(self) -> dict[str, Any]:
        """Return completion-proof references recorded on the goal.

        Objective heaps predate the typed evidence ledger and consequently use
        a few field spellings in the wild.  This projection is deliberately
        lossless (values are not split or otherwise interpreted) and gives
        graph artifacts stable names for the proof dimensions required by the
        completion gate.
        """

        aliases = {
            "acceptance_criterion": (
                "acceptance_criterion",
                "acceptance_criteria",
                "acceptance",
            ),
            "producer": (
                "producing_task_or_scan",
                "producing_task",
                "producing_scan",
                "produced_by",
            ),
            "validation_receipt": (
                "validation_receipt",
                "validation_receipt_cid",
            ),
            "repository_tree": (
                "repository_tree",
                "repository_tree_cid",
                "repo_tree",
                "tree",
            ),
            "freshness": (
                "freshness",
                "evidence_freshness",
                "fresh_at",
            ),
            "provenance_cid": (
                "provenance_cid",
                "evidence_provenance_cid",
                "provenance",
            ),
        }
        metadata: dict[str, Any] = {}
        for canonical_name, field_names in aliases.items():
            for field_name in field_names:
                value = str(self.fields.get(field_name) or "").strip()
                if value:
                    metadata[canonical_name] = value
                    break
        records = self.completion_evidence_records
        if records:
            record = records[0]
            metadata.setdefault("acceptance_criterion", record.get("acceptance_criterion", ""))
            metadata.setdefault(
                "producer",
                record.get("producing_task_or_scan", record.get("producer_id", "")),
            )
            metadata.setdefault("validation_receipt", record.get("validation_receipt", ""))
            metadata.setdefault(
                "repository_tree", record.get("repository_tree", record.get("tree_id", ""))
            )
            metadata.setdefault("freshness", record.get("freshness", ""))
            metadata.setdefault(
                "provenance_cid", record.get("provenance_cid", record.get("receipt_cid", ""))
            )
        return metadata

    @property
    def completion_evidence_records(self) -> list[dict[str, Any]]:
        """Return typed completion records persisted by the tracker."""

        raw = str(
            self.fields.get("completion_evidence_records")
            or self.fields.get("completion_evidence_json")
            or ""
        ).strip()
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        if isinstance(payload, Mapping):
            payload = [payload]
        if not isinstance(payload, list):
            return []
        return [dict(item) for item in payload if isinstance(item, Mapping)]

    @property
    def priority(self) -> tuple[int, str]:
        try:
            fib_priority = int(str(self.fields.get("fib_priority") or "999999").strip())
        except ValueError:
            fib_priority = 999999
        return fib_priority, self.goal_id

    @property
    def required_evidence(self) -> list[str]:
        return split_terms(str(self.fields.get("evidence") or self.fields.get("required_evidence") or ""))

    @property
    def parent_goal_ids(self) -> list[str]:
        return split_terms(str(self.fields.get("parents") or self.fields.get("parent") or ""))

    def bundle_key(self, missing_terms: Sequence[str]) -> str:
        explicit = str(self.fields.get("bundle") or "").strip()
        if explicit:
            return explicit.strip("/ ")
        track = str(self.fields.get("track") or "ops").strip().lower() or "ops"
        roots = split_terms(str(self.fields.get("outputs") or ""))
        root = "general"
        for candidate in roots:
            first = candidate.split("/", 1)[0].strip()
            if first and first not in {"data", "tests"}:
                root = first
                break
        fingerprint = sha1("|".join([self.goal_id, *missing_terms]).encode("utf-8")).hexdigest()[:8]
        return f"objective/{track}/{root}/{fingerprint}"


@dataclass(frozen=True)
class ObjectiveFinding:
    """A missing objective proof that can become a todo task."""

    fingerprint: str
    goal_id: str
    title: str
    summary: str
    priority: str
    track: str
    missing_evidence: list[str]
    present_evidence: dict[str, list[str]]
    evidence_methods: list[str]
    objective_path: str
    outputs: list[str]
    validation: str
    goal: str = ""
    refinement: str = ""
    gap_task: str = ""
    parent_goal_ids: list[str] = field(default_factory=list)
    graph_depth: int = 0
    bundle_key: str = "objective/general"
    parallel_lane: str = "objective/general"
    bundle_explicit: bool = False
    bundle_strategy: str = "semantic_ast"
    embedding_query: str = ""
    ast_query: str = ""
    conflict_policy: str = "prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts"
    refinement_depth: str = "0"
    candidate_kind: str = "aggregate"
    surplus_group: str = ""
    merge_key: str = ""
    merge_family: str = ""
    merge_role: str = ""
    work_item_count: int = 0
    work_scope: str = ""
    todo_vector_key: str = ""
    goal_packet_key: str = ""
    goal_packet_role: str = ""
    goal_packet_goal_ids: list[str] = field(default_factory=list)
    goal_packet_task_count: int = 0
    goal_packet_work_item_count: int = 0
    predicted_files: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    ast_symbols: list[str] = field(default_factory=list)
    interfaces: list[str] = field(default_factory=list)
    submodules: list[str] = field(default_factory=list)
    generated_artifacts: list[str] = field(default_factory=list)
    allow_concurrent_with: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectiveTaskRecord:
    """A generated todo task and its bundle metadata."""

    task_id: str
    task_block: str
    finding: ObjectiveFinding
    discovery_path: Path


@dataclass(frozen=True)
class ObjectiveHeapRecord:
    """One scheduled objective-heap entry."""

    heap_index: int
    goal_id: str
    title: str
    status: str
    fib_priority: int
    priority: str
    priority_rank: int
    track: str
    graph_depth: int
    work_surface_score: int
    required_evidence_count: int
    output_count: int
    parents: list[str]
    sort_key: list[Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEPENDENCY_EDGE_KINDS = frozenset(
    {"goal", "import", "interface", "output_input", "migration", "validation"}
)
SUCCESSFUL_MERGE_RECEIPT_STATUSES = frozenset(
    {"complete", "completed", "merged", "passed", "success", "succeeded"}
)


@dataclass(frozen=True)
class DependencyEdge:
    """A prerequisite relationship directed from producer to consumer.

    ``provenance`` intentionally remains structured instead of being collapsed
    into a reason string.  Persisted DAGs can consequently explain which task
    field and value produced an edge, even after task aliases are reconciled.
    """

    source_task_cid: str
    target_task_cid: str
    kind: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskDependencyNode:
    """One canonical task in a materialized dependency graph."""

    task_cid: str
    task_id: str
    goal_id: str
    status: str
    objective_priority: int
    created_at_ms: int
    estimated_duration: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DependencyRepairEvidence:
    """Bounded, actionable evidence for malformed dependency metadata."""

    kind: str
    task_cid: str
    task_id: str
    reference: str
    message: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskScheduleRecord:
    """Critical-path scheduling metrics for one canonical task."""

    task_cid: str
    task_id: str
    claimable: bool
    blocking_task_cids: list[str]
    critical_path_length: int
    slack: int
    downstream_unlock_value: int
    age_seconds: int
    objective_priority: int
    score: int
    sort_key: list[Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskDependencyGraph:
    """A task DAG, its schedule, and finite repair evidence.

    The type retains cyclic or incomplete nodes for diagnosis.  Such nodes are
    simply made unclaimable; they never prevent independent acyclic components
    from being scheduled.
    """

    nodes: dict[str, TaskDependencyNode]
    edges: list[DependencyEdge]
    schedule: list[TaskScheduleRecord] = field(default_factory=list)
    repair_evidence: list[DependencyRepairEvidence] = field(default_factory=list)
    invalid_task_cids: list[str] = field(default_factory=list)

    @property
    def claimable_task_cids(self) -> list[str]:
        return [record.task_cid for record in self.schedule if record.claimable]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {key: value.to_dict() for key, value in sorted(self.nodes.items())},
            "edges": [edge.to_dict() for edge in self.edges],
            "schedule": [record.to_dict() for record in self.schedule],
            "repair_evidence": [item.to_dict() for item in self.repair_evidence],
            "invalid_task_cids": sorted(self.invalid_task_cids),
            "claimable_task_cids": self.claimable_task_cids,
        }


# A concise alias for callers that use the backlog task's DAG terminology.
TaskDependencyDAG = TaskDependencyGraph


@dataclass(frozen=True)
class TaskPlanningGraph:
    """Combined dependency schedule and conflict-colored execution plan.

    Keeping the two graph types together prevents callers from accidentally
    treating a dependency-ready task as concurrency-safe.  The conflict graph
    remains the owner of lane assignments and their explanations, while the
    dependency graph remains the owner of prerequisite claimability.
    """

    dependency_graph: TaskDependencyGraph
    conflict_graph: Any

    @property
    def claimable_task_cids(self) -> list[str]:
        return self.dependency_graph.claimable_task_cids

    def to_dict(self) -> dict[str, Any]:
        dependency = self.dependency_graph.to_dict()
        conflict = self.conflict_graph.to_dict()
        return {
            "task_dependency_graph": dependency,
            "dependency_dag": dependency,
            "task_conflict_graph": conflict,
            "conflict_graph": conflict,
            "claimable_task_cids": self.claimable_task_cids,
            "lanes": conflict.get("lanes", {}),
            "lane_assignments": conflict.get("assignments", []),
            "planning_decisions": conflict.get("decisions", []),
        }


@dataclass(frozen=True)
class BundleWriteResult:
    """Files produced when bundle shards and the index are written."""

    generated_paths: list[Path]
    index_path: Path
    bundle_paths: dict[str, Path]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def split_terms(value: str) -> list[str]:
    terms: list[str] = []
    for raw in re.split(r"[,;]", value):
        term = " ".join(raw.strip().split())
        if term:
            terms.append(term)
    return terms


def objective_tokens(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", value.lower()) if len(token) > 1]


def text_embedding(value: str, *, dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS) -> list[float]:
    vector = [0.0] * max(1, int(dimensions))
    for token in objective_tokens(value):
        digest = sha1(token.encode("utf-8")).digest()
        vector[int.from_bytes(digest[:4], "big") % len(vector)] += 1.0
    norm = math.sqrt(sum(item * item for item in vector))
    if norm == 0:
        return vector
    return [item / norm for item in vector]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def normalize_field_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def parse_goal_heap(text: str) -> list[ObjectiveGoal]:
    """Parse flat markdown objective records.

    Records use ``## GOAL-ID Title`` headers and ``- Field: value`` rows.  The
    goal id is intentionally not tied to a project prefix so other packages can
    provide their own objective heaps.
    """

    goals: list[ObjectiveGoal] = []
    current_id = ""
    current_title = ""
    current_fields: dict[str, str] = {}
    header_pattern = re.compile(r"^##\s+(\S+)\s+(.+?)\s*$")

    def flush() -> None:
        if current_id and current_fields:
            goals.append(ObjectiveGoal(goal_id=current_id, title=current_title.strip(), fields=dict(current_fields)))

    for line in text.splitlines():
        header = header_pattern.match(line)
        if header:
            flush()
            current_id = header.group(1)
            current_title = header.group(2)
            current_fields = {}
            continue
        if not current_id or not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        current_fields[normalize_field_key(key)] = value.strip()
    flush()
    return goals


def safe_bundle_key(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip("/ ").lower()).strip("-")
    return safe or "objective-general"


def repo_relative_path(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def repo_relative_path_safe(relative: str) -> bool:
    if not relative or relative.startswith("/") or "\0" in relative:
        return False
    return ".." not in Path(relative).parts


def symbol_terms(path: Path, text: str) -> set[str]:
    """Extract AST/schema-ish terms from code and structured files."""

    suffix = path.suffix.lower()
    symbols: set[str] = set()
    if suffix == ".py":
        try:
            tree = parse_python_ast_quietly(text)
        except SyntaxError:
            tree = None
        if tree is not None:
            class_stack: list[str] = []

            class Visitor(ast.NodeVisitor):
                def visit_ClassDef(self, node: ast.ClassDef) -> Any:
                    symbols.add(node.name)
                    class_stack.append(node.name)
                    self.generic_visit(node)
                    class_stack.pop()

                def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
                    symbols.add(node.name)
                    if class_stack:
                        symbols.add(f"{class_stack[-1]}.{node.name}")
                    self.generic_visit(node)

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
                    self.visit_FunctionDef(node)  # type: ignore[arg-type]

                def visit_Import(self, node: ast.Import) -> Any:
                    for alias in node.names:
                        symbols.add(alias.name)

                def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
                    if node.module:
                        symbols.add(node.module)
                        for alias in node.names:
                            symbols.add(f"{node.module}.{alias.name}")

            Visitor().visit(tree)
    elif suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        for match in re.finditer(
            r"\b(?:class|function|interface|type|const|let|var)\s+([A-Za-z_$][\w$]*)",
            text,
        ):
            symbols.add(match.group(1))
        for match in re.finditer(r"\bexport\s+\{([^}]+)\}", text):
            for raw in match.group(1).split(","):
                symbol = raw.strip().split(" as ", 1)[0].strip()
                if symbol:
                    symbols.add(symbol)
    elif suffix in {".md", ".rst"}:
        for line in text.splitlines():
            stripped = line.strip("#= ")
            if line.startswith("#") and stripped:
                symbols.add(stripped)
    elif suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None

        def collect(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    symbols.add(str(key))
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(payload)

    expanded: set[str] = set()
    for symbol in symbols:
        expanded.add(symbol)
        expanded.add(" ".join(objective_tokens(symbol)))
    return {item.lower() for item in expanded if item.strip()}


def ast_dataset_payload(path: Path, text: str, *, max_chars: int = DEFAULT_AST_DATASET_MAX_CHARS) -> dict[str, Any]:
    """Return a serializable AST/symbol payload suitable for dataset storage."""

    suffix = path.suffix.lower()
    ast_kind = "symbols"
    ast_text = ""
    parse_error = ""
    if suffix == ".py":
        ast_kind = "python_ast"
        try:
            tree = parse_python_ast_quietly(text)
            ast_text = ast.dump(tree, include_attributes=True)
        except SyntaxError as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    elif suffix == ".json":
        ast_kind = "json_keys"
    elif suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        ast_kind = "js_ts_symbols"
    elif suffix in {".md", ".rst"}:
        ast_kind = "markdown_headings"

    truncated = False
    if len(ast_text) > max_chars:
        ast_text = ast_text[:max_chars]
        truncated = True
    return {
        "ast_kind": ast_kind,
        "ast_text": ast_text,
        "ast_truncated": truncated,
        "parse_error": parse_error,
    }


def collect_ast_dataset_records(
    repo_root: Path,
    *,
    objective_path: Path,
    max_ast_chars: int = DEFAULT_AST_DATASET_MAX_CHARS,
    previous_records: Sequence[Mapping[str, Any]] = (),
    scan_stats: dict[str, Any] | None = None,
    excluded_roots: Iterable[Path] = (),
) -> list[dict[str, Any]]:
    """Collect a complete snapshot while reusing unchanged source blobs."""

    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    prior_rows = [dict(row) for row in previous_records if isinstance(row, Mapping)]
    prior_by_blob: dict[str, list[dict[str, Any]]] = {}
    prior_by_source: dict[str, list[dict[str, Any]]] = {}
    for row in prior_rows:
        if int(row.get("record_schema_version") or 0) != AST_DATASET_RECORD_SCHEMA_VERSION:
            continue
        blob_hash = str(row.get("blob_hash") or "")
        source_hash = str(row.get("source_sha1") or "")
        if blob_hash:
            prior_by_blob.setdefault(blob_hash, []).append(row)
        if source_hash:
            prior_by_source.setdefault(source_hash, []).append(row)
    for candidates in [*prior_by_blob.values(), *prior_by_source.values()]:
        candidates.sort(key=lambda item: str(item.get("root_relative_path") or ""))

    blob_hashes = tracked_blob_hashes(repo_root)
    current_paths: set[str] = set()
    current_path_blobs: dict[str, str] = {}
    parsed_count = 0
    reused_count = 0
    parse_elapsed = 0.0
    saved_parse_seconds = 0.0
    excluded = tuple(root.resolve() for root in excluded_roots)
    for path in objective_candidate_files(repo_root, objective_path=objective_path):
        resolved_path = path.resolve()
        if any(resolved_path == root or root in resolved_path.parents for root in excluded):
            continue
        root_relative = repo_relative_path(repo_root, path)
        current_paths.add(root_relative)
        blob_hash = blob_hashes.get(resolved_path, "")
        prior = prior_by_blob.get(blob_hash, [None])[0] if blob_hash else None
        source_bytes: bytes | None = None
        text: str | None = None
        source_hash = ""
        if prior is None:
            try:
                source_bytes = path.read_bytes()
            except OSError:
                continue
            text = source_bytes.decode("utf-8", errors="replace")
            source_hash = sha1(source_bytes).hexdigest()
            prior = prior_by_source.get(source_hash, [None])[0]

        if prior is not None:
            row = dict(prior)
            if str(row.get("root_relative_path") or "") != root_relative:
                row.update(
                    _ast_evidence_fields(
                        root_relative,
                        str(row.get("evidence_text") or ""),
                        _record_symbols(row),
                    )
                )
            row.update(
                {
                    "root_relative_path": root_relative,
                    "suffix": path.suffix.lower(),
                    "blob_hash": blob_hash or str(row.get("blob_hash") or source_hash),
                }
            )
            rows.append(row)
            reused_count += 1
            saved_parse_seconds += _nonnegative_seconds(row.get("parse_elapsed_seconds"))
            current_path_blobs[root_relative] = str(row.get("blob_hash") or row.get("source_sha1") or "")
            continue

        assert text is not None and source_bytes is not None
        parse_started = time.monotonic()
        symbols = sorted(symbol_terms(path, text))
        payload = ast_dataset_payload(path, text, max_chars=max_ast_chars)
        row_parse_elapsed = max(0.0, time.monotonic() - parse_started)
        parsed_count += 1
        parse_elapsed += row_parse_elapsed
        effective_blob_hash = blob_hash or source_hash
        rows.append(
            {
                "record_schema_version": AST_DATASET_RECORD_SCHEMA_VERSION,
                "root_relative_path": root_relative,
                "suffix": path.suffix.lower(),
                "blob_hash": effective_blob_hash,
                "source_sha1": source_hash,
                "source_bytes": len(source_bytes),
                "symbols_json": json.dumps(symbols, sort_keys=True),
                "token_count": len(objective_tokens(text)),
                "parse_elapsed_seconds": row_parse_elapsed,
                **_ast_evidence_fields(root_relative, text, symbols),
                **payload,
            }
        )
        current_path_blobs[root_relative] = effective_blob_hash

    rows.sort(key=lambda row: str(row.get("root_relative_path") or ""))
    prior_paths = {
        str(row.get("root_relative_path") or "")
        for row in prior_rows
        if str(row.get("root_relative_path") or "")
    }
    deleted_paths = sorted(prior_paths - current_paths)
    prior_blob_by_path = {
        str(row.get("root_relative_path") or ""): str(row.get("blob_hash") or row.get("source_sha1") or "")
        for row in prior_rows
    }
    deleted_blob_counts: dict[str, int] = {}
    added_blob_counts: dict[str, int] = {}
    for path in deleted_paths:
        blob = prior_blob_by_path.get(path, "")
        if blob:
            deleted_blob_counts[blob] = deleted_blob_counts.get(blob, 0) + 1
    for path in current_paths - prior_paths:
        blob = current_path_blobs.get(path, "")
        if blob:
            added_blob_counts[blob] = added_blob_counts.get(blob, 0) + 1
    renamed_count = sum(
        min(count, added_blob_counts.get(blob, 0))
        for blob, count in deleted_blob_counts.items()
    )
    if scan_stats is not None:
        scan_stats.clear()
        scan_stats.update(
            {
                "scanned_record_count": len(rows),
                "parsed_record_count": parsed_count,
                "reused_record_count": reused_count,
                "deleted_record_count": len(deleted_paths),
                "renamed_record_count": renamed_count,
                "invalidated_record_count": len(deleted_paths),
                "scan_elapsed_seconds": max(0.0, time.monotonic() - started),
                "parse_elapsed_seconds": parse_elapsed,
                "saved_parse_seconds": saved_parse_seconds,
                "deleted_paths": deleted_paths,
            }
        )
    return rows


def _record_symbols(row: Mapping[str, Any]) -> list[str]:
    try:
        value = json.loads(str(row.get("symbols_json") or "[]"))
    except (TypeError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    return sorted({str(item).strip().lower() for item in value if str(item).strip()})


def _ast_evidence_fields(root_relative: str, text: str, symbols: Sequence[str]) -> dict[str, Any]:
    document_text = f"{root_relative}\n{' '.join(sorted(symbols))}\n{text[:12000]}"
    return {
        "evidence_text": text,
        "document_tokens_json": json.dumps(sorted(set(objective_tokens(document_text)))),
        "document_embedding_json": json.dumps(text_embedding(document_text)),
    }


def _nonnegative_seconds(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def persist_objective_ast_dataset(
    *,
    repo_root: Path,
    objective_path: Path,
    dataset_dir: Path,
    dataset_id: str = "objective-ast",
) -> DatasetArtifact:
    """Persist scan AST/symbol records with the optional ipfs_datasets backend."""

    store = ObjectiveDatasetStore(dataset_dir)
    stats: dict[str, Any] = {}
    rows = collect_ast_dataset_records(
        repo_root,
        objective_path=objective_path,
        previous_records=store.load_records(dataset_id),
        scan_stats=stats,
        excluded_roots=(dataset_dir,),
    )
    return store.persist_records(
        dataset_id=dataset_id,
        records=rows,
        scan_stats=stats,
    )


def discover_git_worktrees(repo_root: Path) -> list[Path]:
    """Return repo_root plus nested tracked git worktrees."""

    roots = [repo_root]
    result = subprocess.run(
        ["git", "submodule", "foreach", "--quiet", "printf '%s\\n' \"$sm_path\""],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                roots.append(repo_root / line)
    return list(dict.fromkeys(path.resolve() for path in roots if path.exists()))


def tracked_blob_hashes(repo_root: Path) -> dict[Path, str]:
    """Map clean tracked paths to their Git blob ids.

    Unstaged paths are omitted because the index blob does not describe their
    current source.  Those files fall back to a content read/hash in the
    incremental collector, while unchanged files need no source read at all.
    """

    hashes: dict[Path, str] = {}
    for git_root in discover_git_worktrees(repo_root):
        listed = subprocess.run(
            ["git", "ls-files", "-s", "-z"],
            cwd=git_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if listed.returncode != 0:
            continue
        dirty_result = subprocess.run(
            ["git", "diff-files", "--name-only", "-z"],
            cwd=git_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        dirty = (
            {
                raw.decode("utf-8", errors="surrogateescape")
                for raw in dirty_result.stdout.split(b"\0")
                if raw
            }
            if dirty_result.returncode == 0
            else set()
        )
        for entry in listed.stdout.split(b"\0"):
            if not entry or b"\t" not in entry:
                continue
            metadata, raw_relative = entry.split(b"\t", 1)
            fields = metadata.split()
            if len(fields) < 3 or fields[2] != b"0":
                continue
            relative = raw_relative.decode("utf-8", errors="surrogateescape")
            if relative in dirty or not repo_relative_path_safe(relative):
                continue
            hashes[(git_root / relative).resolve()] = fields[1].decode("ascii", errors="ignore")
    return hashes


def tracked_files(git_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=git_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return [path for path in git_root.rglob("*") if path.is_file()]
    files: list[Path] = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        if repo_relative_path_safe(relative):
            path = git_root / relative
            if path.is_file():
                files.append(path)
    return files


def scan_candidate(path: Path, *, repo_root: Path, objective_path: Path) -> bool:
    if path.resolve() == objective_path.resolve():
        return False
    root_relative = repo_relative_path(repo_root, path)
    parts = set(Path(root_relative).parts)
    if parts & SKIP_DIRS:
        return False
    if path.suffix.lower() not in SCAN_SUFFIXES:
        return False
    if "-objective-gap-" in path.name or path.name == "index.json":
        return False
    if {"discovery", "objective_bundles"} & parts:
        return False
    if root_relative.startswith("data/"):
        return False
    try:
        return path.stat().st_size <= 262144
    except OSError:
        return False


def objective_candidate_files(repo_root: Path, *, objective_path: Path) -> list[Path]:
    files: list[Path] = []
    for git_root in discover_git_worktrees(repo_root):
        for path in tracked_files(git_root):
            if scan_candidate(path, repo_root=repo_root, objective_path=objective_path):
                files.append(path)
    return sorted(dict.fromkeys(files), key=lambda path: repo_relative_path(repo_root, path))


def evidence_methods(present_evidence: Mapping[str, Any]) -> list[str]:
    methods: set[str] = set()
    for paths in present_evidence.values():
        values = paths if isinstance(paths, list) else [paths]
        for value in values:
            match = re.search(r"\((path|exact|ast|embedding)(?::[^)]*)?\)\s*$", str(value))
            if match:
                methods.add(match.group(1))
    return sorted(methods)


def evidence_index(
    repo_root: Path,
    *,
    objective_path: Path,
    terms: Sequence[str],
    embedding_min_score: float = DEFAULT_EMBEDDING_MIN_SCORE,
    records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, list[str]]:
    normalized_terms = [term for term in dict.fromkeys(str(term).strip() for term in terms) if term]
    evidence = {term: [] for term in normalized_terms}
    if not normalized_terms:
        return evidence

    term_tokens = {term: set(objective_tokens(term)) for term in normalized_terms}
    term_embeddings = {term: text_embedding(term) for term in normalized_terms}

    for term in normalized_terms:
        if not repo_relative_path_safe(term):
            continue
        candidate = repo_root / term
        if candidate.exists():
            evidence[term].append(f"{Path(term).as_posix()} (path)")

    lowered_terms = {term: term.lower() for term in normalized_terms}
    cached_records = list(records) if records is not None else None
    candidates: list[tuple[str, str, set[str], set[str], list[float]]] = []
    if cached_records is not None:
        for row in sorted(cached_records, key=lambda item: str(item.get("root_relative_path") or "")):
            root_relative = str(row.get("root_relative_path") or "")
            if not root_relative:
                continue
            text = str(row.get("evidence_text") or "")
            symbols = set(_record_symbols(row))
            try:
                raw_tokens = json.loads(str(row.get("document_tokens_json") or "[]"))
            except (TypeError, ValueError):
                raw_tokens = []
            document_tokens = {str(item) for item in raw_tokens} if isinstance(raw_tokens, list) else set()
            try:
                raw_embedding = json.loads(str(row.get("document_embedding_json") or "[]"))
                document_embedding = [float(item) for item in raw_embedding] if isinstance(raw_embedding, list) else []
            except (TypeError, ValueError):
                document_embedding = []
            # Legacy/incomplete rows are never silently treated as negative
            # evidence.  Rebuild their cheap derived fields from cached text.
            if not document_embedding or not document_tokens:
                document_text = f"{root_relative}\n{' '.join(sorted(symbols))}\n{text[:12000]}"
                document_embedding = text_embedding(document_text)
                document_tokens = set(objective_tokens(document_text))
            candidates.append((root_relative, text, symbols, document_tokens, document_embedding))
    else:
        for path in objective_candidate_files(repo_root, objective_path=objective_path):
            root_relative = repo_relative_path(repo_root, path)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            symbols = symbol_terms(path, text)
            document_text = f"{root_relative}\n{' '.join(sorted(symbols))}\n{text[:12000]}"
            candidates.append(
                (
                    root_relative,
                    text,
                    symbols,
                    set(objective_tokens(document_text)),
                    text_embedding(document_text),
                )
            )

    for root_relative, text, symbols, document_tokens, document_embedding in candidates:
        haystack = f"{root_relative}\n{text}".lower()

        for term, lowered in lowered_terms.items():
            if len(evidence[term]) >= 3:
                continue
            if lowered in haystack:
                evidence[term].append(f"{root_relative} (exact)")
                continue
            normalized_symbol = " ".join(objective_tokens(term))
            if normalized_symbol and (
                normalized_symbol in symbols
                or any(normalized_symbol in symbol or symbol in normalized_symbol for symbol in symbols)
            ):
                evidence[term].append(f"{root_relative} (ast)")
                continue
            overlap = term_tokens[term] & document_tokens
            required_overlap = max(1, min(2, len(term_tokens[term])))
            score = cosine(term_embeddings[term], document_embedding)
            overlap_ratio = len(overlap) / max(1, len(term_tokens[term]))
            threshold = embedding_min_score
            if overlap_ratio >= 0.75:
                threshold = min(threshold, 0.30)
            if len(overlap) >= required_overlap and score >= threshold:
                evidence[term].append(f"{root_relative} (embedding:{score:.2f})")
    return evidence


def goal_graph(goals: Sequence[ObjectiveGoal]) -> dict[str, Any]:
    """Materialize objective hierarchy, lifecycle, and proof requirements.

    ``nodes``, ``edges``, ``children``, ``roots``, and ``depths`` retain their
    historical shapes.  The additional projections make the graph safe for
    lifecycle-aware callers: no consumer needs to infer completion from a lack
    of scheduled tasks, and proof references remain attached to their goal.
    """

    nodes = {goal.goal_id: goal for goal in goals if goal.goal_id}
    edges: list[dict[str, str]] = []
    children: dict[str, list[str]] = {goal_id: [] for goal_id in nodes}
    roots: list[str] = []
    for goal_id, goal in nodes.items():
        parents = [parent for parent in goal.parent_goal_ids if parent]
        if not parents:
            roots.append(goal_id)
        for parent in parents:
            edges.append({"from": parent, "to": goal_id, "kind": "refines"})
            children.setdefault(parent, []).append(goal_id)

    depths: dict[str, int] = {}

    def depth_for(goal_id: str, seen: set[str] | None = None) -> int:
        if goal_id in depths:
            return depths[goal_id]
        seen = set(seen or set())
        if goal_id in seen:
            depths[goal_id] = 0
            return 0
        seen.add(goal_id)
        parents = nodes.get(goal_id, ObjectiveGoal(goal_id, "", {})).parent_goal_ids
        if not parents:
            depths[goal_id] = 0
            return 0
        known_parents = [parent for parent in parents if parent in nodes]
        if not known_parents:
            depths[goal_id] = 1
            return 1
        depths[goal_id] = 1 + max(depth_for(parent, seen) for parent in known_parents)
        return depths[goal_id]

    for goal_id in nodes:
        depth_for(goal_id)

    node_details: dict[str, dict[str, Any]] = {}
    state_counts: dict[str, int] = {}
    schedulable_goal_ids: list[str] = []
    terminal_goal_ids: list[str] = []
    evidence_nodes: list[dict[str, Any]] = []
    evidence_edges: list[dict[str, str]] = []
    for goal_id, goal in sorted(nodes.items()):
        state = goal.lifecycle_state_value
        state_counts[state] = state_counts.get(state, 0) + 1
        if goal.is_schedulable:
            schedulable_goal_ids.append(goal_id)
        if goal.is_terminal:
            terminal_goal_ids.append(goal_id)
        required_evidence = goal.required_evidence
        node_details[goal_id] = {
            "goal_id": goal_id,
            "title": goal.title,
            "status": goal.status,
            "lifecycle_state": state,
            "schedulable": goal.is_schedulable,
            "terminal": goal.is_terminal,
            "parents": goal.parent_goal_ids,
            "required_evidence": required_evidence,
            "completion_evidence": goal.completion_evidence_metadata,
        }
        for term in required_evidence:
            evidence_id = "evidence:" + sha1(
                f"{goal_id}\0{term}".encode("utf-8")
            ).hexdigest()[:16]
            evidence_nodes.append(
                {
                    "id": evidence_id,
                    "goal_id": goal_id,
                    "acceptance_criterion": term,
                }
            )
            evidence_edges.append(
                {
                    "from": goal_id,
                    "to": evidence_id,
                    "kind": "requires_evidence",
                }
            )
        for record in goal.completion_evidence_records:
            criterion = str(record.get("acceptance_criterion") or "").strip()
            provenance_cid = str(
                record.get("provenance_cid") or record.get("receipt_cid") or ""
            ).strip()
            proof_id = "completion-evidence:" + sha1(
                f"{goal_id}\0{criterion}\0{provenance_cid}".encode("utf-8")
            ).hexdigest()[:16]
            evidence_nodes.append(
                {
                    "id": proof_id,
                    "goal_id": goal_id,
                    "acceptance_criterion": criterion,
                    "kind": "completion_evidence",
                    "producing_task_or_scan": str(
                        record.get("producing_task_or_scan")
                        or record.get("producer_id")
                        or ""
                    ),
                    "validation_receipt": record.get("validation_receipt"),
                    "repository_tree": str(
                        record.get("repository_tree") or record.get("tree_id") or ""
                    ),
                    "freshness": record.get("freshness"),
                    "provenance_cid": provenance_cid,
                }
            )
            evidence_edges.append(
                {"from": goal_id, "to": proof_id, "kind": "supported_by"}
            )

    lifecycle = {
        "state_counts": dict(sorted(state_counts.items())),
        "schedulable_goal_ids": schedulable_goal_ids,
        "terminal_goal_ids": terminal_goal_ids,
        "nonterminal_goal_ids": sorted(set(nodes) - set(terminal_goal_ids)),
    }

    return {
        "nodes": sorted(nodes),
        "node_details": node_details,
        "edges": edges,
        "children": {key: sorted(value) for key, value in children.items() if value},
        "roots": sorted(roots),
        "depths": depths,
        "lifecycle": lifecycle,
        "state_counts": lifecycle["state_counts"],
        "schedulable_goal_ids": lifecycle["schedulable_goal_ids"],
        "terminal_goal_ids": lifecycle["terminal_goal_ids"],
        "evidence_nodes": evidence_nodes,
        "evidence_edges": evidence_edges,
    }


def priority_rank(value: str) -> int:
    normalized = str(value or "").strip().upper()
    if normalized.startswith("P"):
        try:
            return max(0, int(normalized[1:]))
        except ValueError:
            pass
    return 9


def _task_record_mapping(task: Any) -> dict[str, Any]:
    if isinstance(task, Mapping):
        return dict(task)
    if hasattr(task, "to_dict"):
        value = task.to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    if hasattr(task, "__dataclass_fields__"):
        return asdict(task)
    return {
        name: getattr(task, name)
        for name in dir(task)
        if not name.startswith("_") and not callable(getattr(task, name, None))
    }


def _dependency_values(task: Mapping[str, Any], *names: str) -> list[str]:
    normalized = {
        re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_"): value
        for key, value in task.items()
    }
    values: list[str] = []
    for name in names:
        value = normalized.get(name)
        if value in (None, "", [], ()):
            continue
        if isinstance(value, str):
            candidates = re.split(r"[,;]", value)
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            candidates = value
        else:
            candidates = [value]
        for candidate in candidates:
            text = " ".join(str(candidate).strip().split())
            if text and text.lower() not in {"none", "n/a"} and text not in values:
                values.append(text)
    return values


def _task_created_at_ms(task: Mapping[str, Any]) -> int:
    for name in ("created_at_ms", "queued_at_ms", "submitted_at_ms"):
        value = task.get(name)
        try:
            if value not in (None, ""):
                return max(0, int(value))
        except (TypeError, ValueError):
            continue
    for name in ("created_at", "queued_at", "submitted_at"):
        value = str(task.get(name) or "").strip()
        if not value:
            continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0, int(parsed.timestamp() * 1000))
        except ValueError:
            continue
    return 0


def _task_duration(task: Mapping[str, Any]) -> int:
    for name in ("estimated_duration", "duration", "work_item_count", "weight"):
        try:
            value = int(task.get(name) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 1


def _successful_merge_receipt_cids(
    merge_receipts: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    aliases: Mapping[str, str],
) -> set[str]:
    succeeded: set[str] = set()
    if isinstance(merge_receipts, Mapping):
        receipt_items: list[tuple[str, Any]] = list(merge_receipts.items())
    else:
        receipt_items = [("", receipt) for receipt in merge_receipts]
    for key, raw_receipt in receipt_items:
        receipt = dict(raw_receipt) if isinstance(raw_receipt, Mapping) else {"status": raw_receipt}
        reference = str(
            receipt.get("canonical_task_cid")
            or receipt.get("task_cid")
            or receipt.get("task_id")
            or key
            or ""
        ).strip()
        status = str(
            receipt.get("merge_status")
            or receipt.get("status")
            or receipt.get("outcome")
            or receipt.get("result")
            or ""
        ).strip().lower()
        successful = receipt.get("succeeded") is True or receipt.get("success") is True
        if successful or status in SUCCESSFUL_MERGE_RECEIPT_STATUSES:
            cid = aliases.get(reference, reference)
            if cid:
                succeeded.add(cid)
    return succeeded


def _cycle_components(nodes: Iterable[str], adjacency: Mapping[str, set[str]]) -> list[list[str]]:
    """Return cyclic strongly-connected components in deterministic order."""

    node_list = sorted(set(nodes))
    visited: set[str] = set()
    finish_order: list[str] = []
    # Iterative Kosaraju traversal avoids recursion failures on malformed,
    # machine-generated graphs with thousands of nodes.
    for root in node_list:
        if root in visited:
            continue
        visited.add(root)
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            node, exiting = stack.pop()
            if exiting:
                finish_order.append(node)
                continue
            stack.append((node, True))
            for child in sorted(adjacency.get(node, set()), reverse=True):
                if child not in visited:
                    visited.add(child)
                    stack.append((child, False))

    reverse: dict[str, set[str]] = {node: set() for node in node_list}
    for source in node_list:
        for target in adjacency.get(source, set()):
            if target in reverse:
                reverse[target].add(source)
    assigned: set[str] = set()
    components: list[list[str]] = []
    for root in reversed(finish_order):
        if root in assigned:
            continue
        assigned.add(root)
        component: list[str] = []
        stack = [(root, False)]
        while stack:
            node, _unused = stack.pop()
            component.append(node)
            for parent in sorted(reverse[node], reverse=True):
                if parent not in assigned:
                    assigned.add(parent)
                    stack.append((parent, False))
        component.sort()
        if len(component) > 1 or (component and component[0] in adjacency.get(component[0], set())):
            components.append(component)
    return sorted(components, key=lambda item: tuple(item))


def critical_path_schedule(
    graph: TaskDependencyGraph,
    *,
    merge_receipts: Mapping[str, Any] | Iterable[Mapping[str, Any]] = (),
    now: datetime | int | None = None,
) -> list[TaskScheduleRecord]:
    """Schedule all tasks by critical path, slack, unlock value, age, and priority.

    Prerequisite completion is deliberately defined by a successful merge
    receipt, not by mutable todo status.  Invalid components remain visible at
    the end of the schedule but cannot be claimed.
    """

    aliases: dict[str, str] = {}
    for cid, node in graph.nodes.items():
        aliases[cid] = cid
        if node.task_id:
            aliases[node.task_id] = cid
    succeeded = _successful_merge_receipt_cids(merge_receipts, aliases)
    incoming: dict[str, set[str]] = {cid: set() for cid in graph.nodes}
    outgoing: dict[str, set[str]] = {cid: set() for cid in graph.nodes}
    for edge in graph.edges:
        if edge.source_task_cid in graph.nodes and edge.target_task_cid in graph.nodes:
            incoming[edge.target_task_cid].add(edge.source_task_cid)
            outgoing[edge.source_task_cid].add(edge.target_task_cid)

    invalid = set(graph.invalid_task_cids)
    invalid.update(item.task_cid for item in graph.repair_evidence if item.task_cid in graph.nodes)
    unresolved_nodes = set(graph.nodes) - succeeded
    unresolved_adjacency = {
        cid: {child for child in outgoing[cid] if child in unresolved_nodes}
        for cid in unresolved_nodes
    }
    cycle_nodes = {
        cid
        for group in _cycle_components(unresolved_nodes, unresolved_adjacency)
        for cid in group
    }
    invalid.update(cycle_nodes)

    indegree = {
        cid: len([parent for parent in incoming[cid] if parent not in invalid])
        for cid in graph.nodes
        if cid not in invalid
    }
    ready = sorted(cid for cid, count in indegree.items() if count == 0)
    topo: list[str] = []
    while ready:
        cid = ready.pop(0)
        topo.append(cid)
        for child in sorted(outgoing[cid]):
            if child not in indegree:
                continue
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()

    longest_to_finish: dict[str, int] = {}
    descendants: dict[str, set[str]] = {}
    for cid in reversed(topo):
        valid_children = [child for child in outgoing[cid] if child not in invalid]
        longest_to_finish[cid] = graph.nodes[cid].estimated_duration + max(
            (longest_to_finish.get(child, 0) for child in valid_children), default=0
        )
        unlocked: set[str] = set(valid_children)
        for child in valid_children:
            unlocked.update(descendants.get(child, set()))
        descendants[cid] = unlocked

    earliest_start: dict[str, int] = {}
    for cid in topo:
        earliest_start[cid] = max(
            (
                earliest_start.get(parent, 0) + graph.nodes[parent].estimated_duration
                for parent in incoming[cid]
                if parent not in invalid
            ),
            default=0,
        )
    project_duration = max(
        (earliest_start.get(cid, 0) + graph.nodes[cid].estimated_duration for cid in topo),
        default=0,
    )
    if isinstance(now, datetime):
        current_ms = int(now.timestamp() * 1000)
    elif now is None:
        current_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    else:
        current_ms = int(now)

    records: list[TaskScheduleRecord] = []
    for cid, node in graph.nodes.items():
        blockers = sorted(parent for parent in incoming[cid] if parent not in succeeded)
        claimable = cid not in invalid and node.status not in SUCCESSFUL_MERGE_RECEIPT_STATUSES and not blockers
        critical_length = longest_to_finish.get(cid, 0)
        slack = max(0, project_duration - earliest_start.get(cid, 0) - critical_length)
        unlock_value = len(descendants.get(cid, set()))
        age_seconds = max(0, (current_ms - node.created_at_ms) // 1000) if node.created_at_ms else 0
        # The score is informational and stable.  Ordering uses the full tuple
        # below so no weighting can hide a critical-path distinction.
        score = (
            critical_length * 1_000_000
            + unlock_value * 10_000
            + min(age_seconds, 999_999)
            + node.objective_priority * 100
            - slack * 1_000
        )
        sort_key: list[Any] = [
            0 if claimable else 1,
            -critical_length,
            slack,
            -unlock_value,
            -age_seconds,
            -node.objective_priority,
            node.task_id,
            cid,
        ]
        records.append(
            TaskScheduleRecord(
                task_cid=cid,
                task_id=node.task_id,
                claimable=claimable,
                blocking_task_cids=blockers,
                critical_path_length=critical_length,
                slack=slack,
                downstream_unlock_value=unlock_value,
                age_seconds=age_seconds,
                objective_priority=node.objective_priority,
                score=score,
                sort_key=sort_key,
            )
        )
    records.sort(key=lambda item: tuple(item.sort_key))
    return records


def materialize_task_dependency_dag(
    tasks: Sequence[Any],
    *,
    merge_receipts: Mapping[str, Any] | Iterable[Mapping[str, Any]] = (),
    now: datetime | int | None = None,
    max_repair_evidence: int = 64,
) -> TaskDependencyGraph:
    """Materialize typed task prerequisites and their critical-path schedule.

    Supported metadata is intentionally permissive because records arrive from
    markdown, vector indexes, and Profile G adapters.  Direct task ids/CIDs and
    producer/consumer artifact declarations are both accepted.
    """

    limit = max(0, int(max_repair_evidence))
    raw_tasks = [_task_record_mapping(task) for task in tasks]
    nodes: dict[str, TaskDependencyNode] = {}
    records_by_cid: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    repair: list[DependencyRepairEvidence] = []
    invalid_task_cids: set[str] = set()

    def add_repair(kind: str, cid: str, reference: str, message: str, provenance: Mapping[str, Any]) -> None:
        if cid:
            invalid_task_cids.add(cid)
        if len(repair) >= limit:
            return
        node = nodes.get(cid)
        repair.append(
            DependencyRepairEvidence(
                kind=kind,
                task_cid=cid,
                task_id=node.task_id if node else "",
                reference=reference,
                message=message,
                provenance=dict(provenance),
            )
        )

    for index, task in enumerate(raw_tasks):
        task_id = str(task.get("task_id") or task.get("id") or "").strip()
        task_status = str(task.get("status") or "todo").strip().lower()
        provided_cid = str(task.get("canonical_task_cid") or task.get("task_cid") or "").strip()
        if provided_cid:
            cid = provided_cid
        else:
            identity_input = dict(task)
            if not any(
                identity_input.get(name)
                for name in (
                    "canonical_task_key",
                    "dedupe_key",
                    "title",
                    "summary",
                    "outputs",
                    "acceptance",
                    "missing_evidence",
                    "goal_id",
                    "semantic_key",
                    "bundle_key",
                    "work_scope",
                    "fingerprint",
                )
            ):
                # Legacy bundle indexes sometimes contain only a display id.
                # Preserve support without allowing aliases from different
                # boards to collapse onto one execution identity.
                identity_input["dedupe_key"] = ":".join(
                    [
                        "legacy-task",
                        str(task.get("board_namespace") or "objective-graph"),
                        task_id or str(index),
                    ]
                )
            identity = canonical_task_identity(
                identity_input,
                board_namespace=str(task.get("board_namespace") or "objective-graph"),
                source_path=str(task.get("source_path") or ""),
            )
            cid = identity.canonical_task_cid
        if cid in nodes:
            aliases[task_id] = cid
            existing = nodes[cid]
            aliases_metadata = dict(existing.metadata)
            aliases_metadata["task_id_aliases"] = sorted(
                {
                    existing.task_id,
                    task_id,
                    *[str(value) for value in aliases_metadata.get("task_id_aliases", [])],
                }
            )
            nodes[cid] = replace(existing, metadata=aliases_metadata)
            if (
                existing.status in SUCCESSFUL_MERGE_RECEIPT_STATUSES
                or task_status in SUCCESSFUL_MERGE_RECEIPT_STATUSES
            ):
                continue
            add_repair(
                "duplicate_task",
                cid,
                task_id,
                f"multiple task records resolve to canonical task CID {cid}",
                {"record_index": index},
            )
            continue
        rank = priority_rank(str(task.get("priority") or task.get("objective_priority") or "P2"))
        configured_priority = max(0, 9 - rank)
        try:
            explicit_priority = int(task.get("objective_priority"))
            configured_priority = max(0, explicit_priority)
        except (TypeError, ValueError):
            pass
        node = TaskDependencyNode(
            task_cid=cid,
            task_id=task_id,
            goal_id=str(task.get("goal_id") or "").strip(),
            status=task_status,
            objective_priority=configured_priority,
            created_at_ms=_task_created_at_ms(task),
            estimated_duration=_task_duration(task),
            metadata=dict(task),
        )
        nodes[cid] = node
        records_by_cid[cid] = task
        aliases[cid] = cid
        if task_id:
            if task_id in aliases and aliases[task_id] != cid:
                add_repair(
                    "duplicate_alias",
                    cid,
                    task_id,
                    f"task alias {task_id} resolves to more than one canonical task",
                    {"field": "task_id"},
                )
            else:
                aliases[task_id] = cid
        canonical_key = str(task.get("canonical_task_key") or "").strip()
        if canonical_key:
            aliases[canonical_key] = cid

    goals: dict[str, list[str]] = {}
    for cid, node in nodes.items():
        if node.goal_id:
            goals.setdefault(node.goal_id, []).append(cid)

    provider_fields = {
        "import": ("provides_imports", "provided_imports", "exports", "modules"),
        "interface": ("provides_interfaces", "provided_interfaces", "interfaces"),
        "migration": ("provides_migrations", "provided_migrations", "migrations"),
        "validation": ("provides_validations", "validation_outputs", "validation_receipts"),
        "output_input": ("outputs", "output_paths", "produces"),
    }
    requirement_fields = {
        "import": ("import_dependencies", "required_imports", "imports"),
        "interface": ("interface_dependencies", "required_interfaces"),
        "migration": ("migration_dependencies", "required_migrations", "migrations_after"),
        "validation": ("validation_dependencies", "validation_prerequisites"),
        "output_input": ("input_dependencies", "inputs", "input_paths", "consumes"),
    }
    providers: dict[str, dict[str, set[str]]] = {kind: {} for kind in provider_fields}
    for cid, task in records_by_cid.items():
        for kind, fields_for_kind in provider_fields.items():
            for value in _dependency_values(task, *fields_for_kind):
                normalized = value.strip().replace("\\", "/")
                providers[kind].setdefault(normalized, set()).add(cid)

    edges: list[DependencyEdge] = []
    edge_keys: set[tuple[str, str, str, str, str]] = set()

    def add_edge(source: str, target: str, kind: str, *, field_name: str, value: str, resolution: str) -> None:
        provenance = {
            "field": field_name,
            "value": value,
            "resolution": resolution,
            "source_task_id": nodes[source].task_id,
            "target_task_id": nodes[target].task_id,
        }
        key = (source, target, kind, field_name, value)
        if key not in edge_keys:
            edge_keys.add(key)
            edges.append(DependencyEdge(source, target, kind, provenance))

    for target, task in records_by_cid.items():
        parent_goals = _dependency_values(task, "parent_goal_ids", "graph_parents", "goal_parents")
        for parent_goal in parent_goals:
            sources = goals.get(parent_goal, [])
            # Parent goals primarily describe objective hierarchy. They become
            # executable prerequisites only when this planning snapshot also
            # materializes a task for that goal. Explicit dependency fields
            # below remain strict and still produce repair evidence.
            for source in sources:
                add_edge(source, target, "goal", field_name="parent_goal_ids", value=parent_goal, resolution="goal_id")

        direct_fields = ("dependency_task_cids", "depends_on", "dependency_task_ids", "prerequisite_task_cids")
        dependency_kinds = task.get("dependency_kinds") if isinstance(task.get("dependency_kinds"), Mapping) else {}
        for field_name in direct_fields:
            for reference in _dependency_values(task, field_name):
                source = aliases.get(reference)
                configured_kind = str(dependency_kinds.get(reference) or "goal").strip().lower().replace("-", "_")
                kind = configured_kind if configured_kind in DEPENDENCY_EDGE_KINDS else "goal"
                if source is None:
                    add_repair(
                        "missing_dependency",
                        target,
                        reference,
                        f"{field_name} references an unknown prerequisite task",
                        {"edge_kind": kind, "field": field_name},
                    )
                    continue
                add_edge(source, target, kind, field_name=field_name, value=reference, resolution="task_alias")

        for kind, fields_for_kind in requirement_fields.items():
            for field_name in fields_for_kind:
                for requirement in _dependency_values(task, field_name):
                    direct_source = aliases.get(requirement)
                    matched_sources = set(providers[kind].get(requirement.replace("\\", "/"), set()))
                    if direct_source:
                        matched_sources.add(direct_source)
                    for source in sorted(matched_sources):
                        if source != target:
                            add_edge(
                                source,
                                target,
                                kind,
                                field_name=field_name,
                                value=requirement,
                                resolution="task_alias" if source == direct_source else "producer_consumer_match",
                            )
                    # Explicit dependency/prerequisite fields promise an
                    # in-graph producer. Plain imports/inputs may be external.
                    if not matched_sources and ("dependencies" in field_name or "prerequisites" in field_name):
                        add_repair(
                            "missing_dependency",
                            target,
                            requirement,
                            f"{kind} prerequisite {requirement} has no materialized producer",
                            {"edge_kind": kind, "field": field_name},
                        )

    edges.sort(key=lambda edge: (edge.source_task_cid, edge.target_task_cid, edge.kind, json.dumps(edge.provenance, sort_keys=True)))
    adjacency: dict[str, set[str]] = {cid: set() for cid in nodes}
    for edge in edges:
        adjacency[edge.source_task_cid].add(edge.target_task_cid)
    succeeded_task_cids = _successful_merge_receipt_cids(merge_receipts, aliases)
    unresolved_nodes = set(nodes) - succeeded_task_cids
    unresolved_adjacency = {
        cid: {target for target in adjacency[cid] if target in unresolved_nodes}
        for cid in unresolved_nodes
    }
    for component in _cycle_components(unresolved_nodes, unresolved_adjacency):
        cycle = " -> ".join([*(nodes[cid].task_id or cid for cid in component), nodes[component[0]].task_id or component[0]])
        for cid in component:
            add_repair(
                "dependency_cycle",
                cid,
                cycle,
                f"task participates in dependency cycle: {cycle}",
                {"component_task_cids": component},
            )

    graph = TaskDependencyGraph(
        nodes=nodes,
        edges=edges,
        repair_evidence=repair[:limit],
        invalid_task_cids=sorted(invalid_task_cids),
    )
    schedule = critical_path_schedule(graph, merge_receipts=merge_receipts, now=now)
    return replace(graph, schedule=schedule)


# Verbose aliases make the API discoverable to callers that say graph rather
# than DAG, without creating separate implementations.
materialize_task_dependency_graph = materialize_task_dependency_dag
schedule_critical_path = critical_path_schedule


def materialize_task_planning_graph(
    tasks: Sequence[Any],
    *,
    repo_root: Path | None = None,
    merge_receipts: Mapping[str, Any] | Iterable[Mapping[str, Any]] = (),
    branch_diffs: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    conflict_receipts: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    concurrency_overrides: Mapping[str, Any] | Iterable[Any] | None = None,
    conflict_history: Any = None,
    max_lanes: int | None = None,
    now: datetime | int | None = None,
    max_repair_evidence: int = 64,
) -> TaskPlanningGraph:
    """Materialize dependency readiness and conflict-colored lanes together.

    ``tasks`` is copied before either graph consumes it, which is important for
    callers that supply generators.  Historical branch diffs and merge-conflict
    receipts are deliberately routed only to the conflict model; successful
    merge receipts independently control dependency claimability.
    """

    from .conflict_graph import materialize_task_conflict_graph

    task_records = list(tasks)
    dependency_graph = materialize_task_dependency_dag(
        task_records,
        merge_receipts=merge_receipts,
        now=now,
        max_repair_evidence=max_repair_evidence,
    )
    conflict_graph = materialize_task_conflict_graph(
        task_records,
        repo_root=repo_root,
        branch_diffs=branch_diffs,
        conflict_receipts=conflict_receipts,
        concurrency_overrides=concurrency_overrides,
        history=conflict_history,
        max_lanes=max_lanes,
    )
    return TaskPlanningGraph(
        dependency_graph=dependency_graph,
        conflict_graph=conflict_graph,
    )


# Discoverable aliases for scheduler and planner callers.
materialize_task_execution_graph = materialize_task_planning_graph
plan_task_lanes = materialize_task_planning_graph


def objective_goal_work_surface(goal: ObjectiveGoal) -> int:
    """Estimate how much coherent work one goal can support."""

    evidence_count = len(goal.required_evidence)
    output_count = len(split_terms(str(goal.fields.get("outputs") or "")))
    ast_count = len(split_terms(str(goal.fields.get("ast_query") or "")))
    interface_count = len(split_terms(str(goal.fields.get("interfaces") or goal.fields.get("interface_contracts") or "")))
    submodule_count = len(split_terms(str(goal.fields.get("submodules") or goal.fields.get("interoperability_pair") or "")))
    return evidence_count * 4 + output_count * 2 + ast_count + interface_count * 3 + submodule_count * 3


def objective_goal_requires_launch_playwright_validation(goal: ObjectiveGoal) -> bool:
    """Return whether a goal represents the launch slice that needs browser proof."""

    fields = goal.fields
    track = str(fields.get("track") or "").strip().lower()
    bundle = str(fields.get("bundle") or "").strip().lower()
    haystack = " ".join([goal.goal_id, goal.title, *fields.values()]).lower()
    if track == "launch" or bundle.startswith("objective/launch/"):
        return True
    return all(term in haystack for term in ("phone", "desktop", "swissknife", "meta glasses"))


def objective_goal_validation(goal: ObjectiveGoal, fallback_validation: str) -> str:
    """Return validation for generated work from an objective goal."""

    validation = str(goal.fields.get("validation") or fallback_validation).strip()
    if not objective_goal_requires_launch_playwright_validation(goal):
        return validation
    lowered = validation.lower()
    if any(marker in lowered for marker in LAUNCH_PLAYWRIGHT_VALIDATION_MARKERS):
        return validation
    if not validation:
        return LAUNCH_PLAYWRIGHT_VALIDATION_COMMAND
    return f"{validation} && {LAUNCH_PLAYWRIGHT_VALIDATION_COMMAND}"


def objective_goal_validation_gap_terms(goal: ObjectiveGoal) -> list[str]:
    """Return synthetic evidence terms for forced validation-gate work."""

    if not objective_goal_requires_launch_playwright_validation(goal):
        validation = str(goal.fields.get("validation") or "").strip()
        if validation:
            return ["objective validation repair"]
        return []
    return [LAUNCH_PLAYWRIGHT_VALIDATION_GATE_EVIDENCE]


def canonical_interoperability_component(value: str) -> str:
    """Return a stable component key for interoperability dedupe."""

    normalized = " ".join(str(value or "").strip().replace("\\", "/").split()).lower()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    leaf = normalized.rsplit("/", 1)[-1]
    key = re.sub(r"[^a-z0-9]+", "_", leaf).strip("_")
    aliases = {
        "ipfs_accelerate_py": "ipfs_accelerate",
        "ipfs_datasets_py": "ipfs_datasets",
        "ipfs_kit_py": "ipfs_kit",
        "mcpplusplus": "mcp_plus_plus",
    }
    return aliases.get(key, key)


def interoperability_pair_schedule_key(goal: ObjectiveGoal) -> str:
    """Return a stable key for duplicate interoperability pair goals."""

    pair_value = str(goal.fields.get("interoperability_pair") or "").strip()
    if not pair_value:
        return ""
    pair_terms = sorted(
        key
        for term in split_terms(pair_value)
        for key in [canonical_interoperability_component(term)]
        if key
    )
    if not pair_terms:
        return ""
    return "\0".join(pair_terms)


def objective_heap_schedule(goals: Sequence[ObjectiveGoal]) -> list[ObjectiveHeapRecord]:
    """Return schedulable goals in Fibonacci-priority heap order.

    The persisted objective heap stores Fibonacci priority buckets in markdown.
    This function materializes those buckets as a deterministic heap schedule and
    adds work-surface tie breakers so larger integration goals win within the
    same Fibonacci band.
    """

    graph = goal_graph(goals)
    heap: list[tuple[tuple[Any, ...], ObjectiveGoal]] = []
    seen_interoperability_pairs: set[str] = set()
    for goal in goals:
        if not goal.is_schedulable:
            continue
        interoperability_key = interoperability_pair_schedule_key(goal)
        if interoperability_key:
            if interoperability_key in seen_interoperability_pairs:
                continue
            seen_interoperability_pairs.add(interoperability_key)
        fib_priority = goal.priority[0]
        rank = priority_rank(str(goal.fields.get("priority") or "P2"))
        work_surface = objective_goal_work_surface(goal)
        graph_depth = int(graph["depths"].get(goal.goal_id, 0))
        sort_key = (
            fib_priority,
            rank,
            -work_surface,
            graph_depth,
            goal.goal_id,
        )
        heapq.heappush(heap, (sort_key, goal))

    records: list[ObjectiveHeapRecord] = []
    while heap:
        sort_key, goal = heapq.heappop(heap)
        records.append(
            ObjectiveHeapRecord(
                heap_index=len(records),
                goal_id=goal.goal_id,
                title=goal.title,
                status=goal.status,
                fib_priority=goal.priority[0],
                priority=str(goal.fields.get("priority") or "P2"),
                priority_rank=priority_rank(str(goal.fields.get("priority") or "P2")),
                track=str(goal.fields.get("track") or "ops"),
                graph_depth=int(graph["depths"].get(goal.goal_id, 0)),
                work_surface_score=objective_goal_work_surface(goal),
                required_evidence_count=len(goal.required_evidence),
                output_count=len(split_terms(str(goal.fields.get("outputs") or ""))),
                parents=goal.parent_goal_ids,
                sort_key=list(sort_key),
            )
        )
    return records


def objective_fingerprint(goal: ObjectiveGoal, missing_terms: Sequence[str]) -> str:
    payload = "\0".join(
        [
            "objective_goal_gap",
            goal.goal_id,
            goal.title,
            *[" ".join(str(term).lower().split()) for term in missing_terms],
        ]
    )
    return sha1(payload.encode("utf-8")).hexdigest()


def objective_merge_key(goal: ObjectiveGoal, missing_terms: Sequence[str], *, candidate_kind: str = "aggregate") -> str:
    payload = {
        "goal_id": goal.goal_id,
        "candidate_kind": candidate_kind,
        "missing_terms": sorted(" ".join(str(term).split()).lower() for term in missing_terms),
        "outputs": sorted(split_terms(str(goal.fields.get("outputs") or ""))),
        "ast_query": str(goal.fields.get("ast_query") or ""),
    }
    return sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def objective_surplus_group(goal: ObjectiveGoal) -> str:
    return f"objective/{goal.goal_id}"


def objective_todo_vector_key(goal: ObjectiveGoal, missing_terms: Sequence[str], *, candidate_kind: str) -> str:
    payload = "\0".join([goal.goal_id, candidate_kind, *sorted(str(term) for term in missing_terms)])
    return sha1(payload.encode("utf-8")).hexdigest()[:16]


def objective_goal_packet_aggregate_fingerprint(
    packet_key: str,
    findings: Sequence[ObjectiveFinding],
    missing_terms: Sequence[str],
) -> str:
    payload = {
        "kind": "objective_goal_packet_aggregate",
        "packet_key": packet_key,
        "finding_fingerprints": sorted(finding.fingerprint for finding in findings),
        "missing_terms": sorted(" ".join(str(term).lower().split()) for term in missing_terms),
    }
    return sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def objective_goal_packet_aggregate_key(packet_key: str, missing_terms: Sequence[str]) -> str:
    payload = {
        "packet_key": packet_key,
        "missing_terms": sorted(" ".join(str(term).lower().split()) for term in missing_terms),
    }
    return sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def objective_goal_packet_key(findings: Sequence[ObjectiveFinding]) -> str:
    """Return a stable packet key for related goal/subgoal findings."""

    goals = sorted({finding.goal_id for finding in findings if finding.goal_id})
    parents = sorted({parent for finding in findings for parent in finding.parent_goal_ids})
    tracks = sorted({finding.track for finding in findings if finding.track})
    roots = sorted({finding_conflict_root(finding) for finding in findings})
    ast_queries = sorted({finding.ast_query for finding in findings if finding.ast_query})
    seed = {
        "goals": goals,
        "parents": parents or goals,
        "tracks": tracks,
        "roots": roots,
        "ast_queries": ast_queries,
    }
    digest = sha1(json.dumps(seed, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    track = safe_bundle_key(tracks[0] if tracks else "ops").replace("-", "_") or "ops"
    root = safe_bundle_key(roots[0] if roots else "general").replace("-", "_") or "general"
    return f"goal_packet/{track}/{root}/{digest}"


def _goal_packet_group_key(finding: ObjectiveFinding) -> tuple[str, tuple[str, ...], str]:
    family = tuple(sorted(finding.parent_goal_ids or [finding.goal_id]))
    return (finding.track, family, finding_conflict_root(finding))


def assign_goal_subgoal_packets(findings: Sequence[ObjectiveFinding]) -> list[ObjectiveFinding]:
    """Annotate findings with packet metadata for sibling goal/subgoal work."""

    groups: dict[tuple[str, tuple[str, ...], str], list[ObjectiveFinding]] = {}
    for finding in findings:
        groups.setdefault(_goal_packet_group_key(finding), []).append(finding)

    packet_by_fingerprint: dict[str, ObjectiveFinding] = {}
    for group_findings in groups.values():
        if len(group_findings) < 2:
            continue
        packet_key = objective_goal_packet_key(group_findings)
        goal_ids = sorted({finding.goal_id for finding in group_findings if finding.goal_id})
        task_count = len(group_findings)
        packet_work_items = sum(finding.work_item_count or len(finding.missing_evidence) for finding in group_findings)
        multi_goal_packet = len(goal_ids) > 1
        for index, finding in enumerate(
            sorted(group_findings, key=lambda item: (item.priority, item.goal_id, item.candidate_kind, item.fingerprint))
        ):
            role = "packet_anchor" if index == 0 else "packet_member"
            merge_family = packet_key if multi_goal_packet else finding.merge_family
            work_scope = finding.work_scope or "goal_subgoal_multi_evidence_batch"
            if "goal_subgoal_packet" not in work_scope:
                work_scope = f"{work_scope}; goal_subgoal_packet"
            packet_by_fingerprint[finding.fingerprint] = replace(
                finding,
                merge_family=merge_family,
                goal_packet_key=packet_key,
                goal_packet_role=role,
                goal_packet_goal_ids=goal_ids,
                goal_packet_task_count=task_count,
                goal_packet_work_item_count=packet_work_items,
                work_scope=work_scope,
            )

    return [packet_by_fingerprint.get(finding.fingerprint, finding) for finding in findings]


def _unique_strings(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _goal_conflict_terms(goal: ObjectiveGoal, *field_names: str) -> list[str]:
    """Collect normalized conflict-surface declarations from goal fields."""

    return _unique_strings(
        term
        for field_name in field_names
        for term in split_terms(str(goal.fields.get(field_name) or ""))
    )


def _merge_present_evidence(findings: Sequence[ObjectiveFinding]) -> dict[str, list[str]]:
    present: dict[str, list[str]] = {}
    for finding in findings:
        for term, paths in finding.present_evidence.items():
            bucket = present.setdefault(str(term), [])
            for path in paths:
                path_text = str(path)
                if path_text and path_text not in bucket:
                    bucket.append(path_text)
    return present


def _first_non_empty(values: Iterable[str], default: str = "") -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return default


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def add_goal_packet_aggregate_findings(
    findings: Sequence[ObjectiveFinding],
    *,
    max_findings: int,
    seen_fingerprints: Iterable[str] = (),
    summary_prefix: str = DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX,
) -> list[ObjectiveFinding]:
    """Add larger packet-level todos for related goal/subgoal findings when capacity allows."""

    planned = list(findings)
    if len(planned) >= max_findings:
        return planned[:max_findings]

    seen = {str(item) for item in seen_fingerprints if str(item).strip()}
    seen.update(finding.fingerprint for finding in planned)
    groups: dict[str, list[ObjectiveFinding]] = {}
    for finding in planned:
        if finding.goal_packet_key:
            groups.setdefault(finding.goal_packet_key, []).append(finding)

    for packet_key, group_findings in sorted(groups.items()):
        if len(planned) >= max_findings:
            break
        if len(group_findings) < 2:
            continue
        sorted_group = sorted(
            group_findings,
            key=lambda item: (item.priority, item.goal_id, item.candidate_kind, item.fingerprint),
        )
        goal_ids = _unique_strings(finding.goal_id for finding in sorted_group)
        if len(goal_ids) < 2:
            continue
        missing_terms = _unique_strings(term for finding in sorted_group for term in finding.missing_evidence)
        if len(missing_terms) < 2:
            continue
        fingerprint = objective_goal_packet_aggregate_fingerprint(packet_key, sorted_group, missing_terms)
        if fingerprint in seen:
            continue

        anchor = sorted_group[0]
        parent_goal_ids = _unique_strings(parent for finding in sorted_group for parent in finding.parent_goal_ids)
        outputs = _unique_strings(output for finding in sorted_group for output in finding.outputs)
        evidence_methods = _unique_strings(method for finding in sorted_group for method in finding.evidence_methods)
        goal_lines = [
            f"{finding.goal_id}: {finding.goal or finding.title}"
            for finding in sorted_group
            if finding.goal_id or finding.goal or finding.title
        ]
        merge_key = objective_goal_packet_aggregate_key(packet_key, missing_terms)
        title = f"Goal packet aggregate for {', '.join(goal_ids)}"
        summary = f"{summary_prefix} packet: {', '.join(goal_ids)}"
        ast_terms = _unique_strings(term for finding in sorted_group for term in split_terms(finding.ast_query))
        if not ast_terms:
            ast_terms = missing_terms
        aggregate = ObjectiveFinding(
            fingerprint=fingerprint,
            goal_id=anchor.goal_id,
            title=title,
            summary=summary,
            priority=anchor.priority,
            track=anchor.track,
            missing_evidence=missing_terms,
            present_evidence=_merge_present_evidence(sorted_group),
            evidence_methods=evidence_methods,
            objective_path=anchor.objective_path,
            outputs=outputs,
            validation=_first_non_empty((finding.validation for finding in sorted_group), anchor.validation),
            goal="Close packet goals:\n" + "\n".join(f"- {line}" for line in goal_lines),
            refinement=_first_non_empty((finding.refinement for finding in sorted_group)),
            gap_task=(
                "Close the packet-level missing evidence across these related goals/subgoals in one cohesive "
                "change when the output paths overlap."
            ),
            parent_goal_ids=parent_goal_ids,
            graph_depth=min(finding.graph_depth for finding in sorted_group),
            bundle_key=anchor.bundle_key,
            parallel_lane=anchor.parallel_lane,
            bundle_explicit=anchor.bundle_explicit,
            bundle_strategy=anchor.bundle_strategy,
            embedding_query="; ".join(
                _unique_strings(
                    [
                        f"goal packet {packet_key}",
                        *(finding.embedding_query for finding in sorted_group),
                        *(finding.title for finding in sorted_group),
                    ]
                )
            ),
            ast_query=", ".join(ast_terms),
            conflict_policy=anchor.conflict_policy,
            refinement_depth=str(min(_parse_int(finding.refinement_depth, 0) for finding in sorted_group)),
            candidate_kind="goal_packet_aggregate",
            surplus_group=packet_key,
            merge_key=merge_key,
            merge_family=packet_key,
            merge_role="packet_aggregate",
            work_item_count=len(missing_terms),
            work_scope="goal_subgoal_packet_aggregate; vector_ast_bundle",
            todo_vector_key=sha1(f"{packet_key}\0goal_packet_aggregate\0{merge_key}".encode("utf-8")).hexdigest()[:16],
            goal_packet_key=packet_key,
            goal_packet_role="packet_aggregate",
            goal_packet_goal_ids=goal_ids,
            goal_packet_task_count=len(sorted_group) + 1,
            goal_packet_work_item_count=sum(
                finding.work_item_count or len(finding.missing_evidence) for finding in sorted_group
            ),
            predicted_files=_unique_strings(
                path for finding in sorted_group for path in (finding.predicted_files or finding.outputs)
            ),
            changed_paths=_unique_strings(
                path for finding in sorted_group for path in finding.changed_paths
            ),
            ast_symbols=_unique_strings(symbol for finding in sorted_group for symbol in finding.ast_symbols),
            interfaces=_unique_strings(interface for finding in sorted_group for interface in finding.interfaces),
            submodules=_unique_strings(submodule for finding in sorted_group for submodule in finding.submodules),
            generated_artifacts=_unique_strings(
                artifact for finding in sorted_group for artifact in finding.generated_artifacts
            ),
            allow_concurrent_with=_unique_strings(
                task for finding in sorted_group for task in finding.allow_concurrent_with
            ),
        )
        planned.append(aggregate)
        seen.add(fingerprint)
    return planned[:max_findings]


def objective_scan_candidate_limit(*, max_findings: int, surplus_findings_per_goal: int) -> int:
    """Return the internal candidate pool size used before final todo selection."""

    try:
        surplus_count = max(1, int(surplus_findings_per_goal))
    except (TypeError, ValueError):
        surplus_count = DEFAULT_SURPLUS_FINDINGS_PER_GOAL
    try:
        oversample_multiplier = max(1, int(DEFAULT_SCAN_OVERSAMPLE_MULTIPLIER))
    except (TypeError, ValueError):
        oversample_multiplier = 2
    return max_findings * max(2, surplus_count, oversample_multiplier)


def prioritize_larger_work_surface_findings(
    findings: Sequence[ObjectiveFinding],
    *,
    max_findings: int,
) -> list[ObjectiveFinding]:
    """Select findings that give each Codex invocation a larger coherent work surface."""

    if max_findings <= 0:
        return []

    def candidate_rank(finding: ObjectiveFinding) -> int:
        if finding.candidate_kind == "goal_packet_aggregate":
            return 0
        if finding.candidate_kind == "aggregate":
            return 1
        if finding.candidate_kind == "evidence_cluster":
            return 2
        return 3

    def sort_key(finding: ObjectiveFinding) -> tuple[Any, ...]:
        packet_goal_count = len(finding.goal_packet_goal_ids)
        packet_work_items = finding.goal_packet_work_item_count or 0
        work_items = finding.work_item_count or len(finding.missing_evidence)
        return (
            candidate_rank(finding),
            -packet_goal_count,
            -packet_work_items,
            -work_items,
            finding.priority,
            finding.graph_depth,
            finding.goal_id,
            finding.candidate_kind,
            finding.fingerprint,
        )

    return sorted(findings, key=sort_key)[:max_findings]


def surplus_missing_term_groups(
    missing_terms: Sequence[str],
    *,
    surplus_findings_per_goal: int = DEFAULT_SURPLUS_FINDINGS_PER_GOAL,
    min_terms_per_todo: int = DEFAULT_SURPLUS_MIN_TERMS_PER_TODO,
) -> list[tuple[str, list[str]]]:
    """Return mergeable objective-gap candidate groups for one goal.

    The first group preserves the historical behavior: one aggregate task that
    covers all missing terms.  Additional groups are multi-evidence batches by
    default so each generated todo has enough implementation work to justify a
    Codex invocation while still carrying merge-family metadata for bundling.
    """

    terms = [term for term in dict.fromkeys(str(item).strip() for item in missing_terms) if term]
    if not terms:
        return []
    try:
        surplus_count = max(1, int(surplus_findings_per_goal))
    except (TypeError, ValueError):
        surplus_count = 1
    groups: list[tuple[str, list[str]]] = [("aggregate", terms)]
    if surplus_count <= 1 or len(terms) <= 1:
        return groups

    try:
        minimum_terms = max(1, int(min_terms_per_todo))
    except (TypeError, ValueError):
        minimum_terms = 2
    minimum_terms = min(max(1, minimum_terms), len(terms))
    extra_count = surplus_count - 1
    group_size = min(len(terms), max(minimum_terms, math.ceil(len(terms) / max(1, extra_count))))
    seen_term_sets = {tuple(terms)}

    def append_group(candidate_terms: Sequence[str]) -> None:
        normalized = [term for term in dict.fromkeys(str(item).strip() for item in candidate_terms) if term]
        if len(normalized) < minimum_terms:
            return
        key = tuple(normalized)
        if key in seen_term_sets:
            return
        seen_term_sets.add(key)
        groups.append(("evidence_cluster" if len(normalized) > 1 else "evidence_term", normalized))

    for start in range(0, len(terms), group_size):
        if len(groups) >= surplus_count:
            break
        append_group(terms[start : start + group_size])
    start = 1
    while len(groups) < surplus_count and start < len(terms):
        if start + group_size <= len(terms):
            append_group(terms[start : start + group_size])
        else:
            append_group([*terms[start:], *terms[: max(0, group_size - (len(terms) - start))]])
        start += 1
    return groups


def _path_root(value: str) -> str:
    value = str(value or "").strip()
    if not value or not repo_relative_path_safe(value):
        return ""
    first = Path(value).parts[0] if Path(value).parts else ""
    if first in {"data", "tests", "docs", "."}:
        return ""
    return first


def finding_conflict_root(finding: ObjectiveFinding) -> str:
    """Return the path-root conflict domain used for implicit bundle planning."""

    for output in finding.outputs:
        root = _path_root(output)
        if root:
            return root
    for paths in finding.present_evidence.values():
        for raw_path in paths:
            root = _path_root(str(raw_path).split(" (", 1)[0])
            if root:
                return root
    return "general"


def finding_semantic_bundle_text(finding: ObjectiveFinding) -> str:
    present_paths: list[str] = []
    for paths in finding.present_evidence.values():
        present_paths.extend(str(path) for path in paths)
    return "\n".join(
        [
            finding.title,
            finding.goal,
            finding.embedding_query,
            finding.ast_query,
            " ".join(finding.missing_evidence),
            " ".join(present_paths),
        ]
    )


def _bundle_cluster_key(*, finding: ObjectiveFinding, root: str, cluster_text: str) -> str:
    digest = sha1(cluster_text.encode("utf-8")).hexdigest()[:8]
    track = safe_bundle_key(finding.track).replace("-", "_") or "ops"
    safe_root = safe_bundle_key(root).replace("-", "_") or "general"
    return f"objective/{track}/{safe_root}/semantic-{digest}"


def plan_semantic_ast_bundles(
    findings: Sequence[ObjectiveFinding],
    *,
    min_score: float = DEFAULT_BUNDLE_CLUSTER_MIN_SCORE,
    preserve_explicit: bool = True,
) -> list[ObjectiveFinding]:
    """Assign implicit objective findings to AST/embedding-aware bundle lanes.

    Explicit ``Bundle:`` fields are left unchanged.  Remaining findings are
    clustered inside a conflict-domain path root using deterministic token
    embeddings built from the goal text, missing evidence, AST query, and present
    evidence paths.  This keeps likely-overlapping work in the same lane while
    letting unrelated roots drain in parallel.
    """

    planned: list[ObjectiveFinding] = []
    clusters: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for finding in findings:
        if preserve_explicit and finding.bundle_explicit:
            planned.append(finding)
            continue
        root = finding_conflict_root(finding)
        group_key = (finding.track, root)
        text = finding_semantic_bundle_text(finding)
        vector = text_embedding(text)
        selected: dict[str, Any] | None = None
        best_score = -1.0
        for cluster in clusters.get(group_key, []):
            score = cosine(vector, cluster["vector"])
            if score > best_score:
                best_score = score
                selected = cluster
        if selected is None or best_score < min_score:
            bundle_key = _bundle_cluster_key(finding=finding, root=root, cluster_text=text)
            selected = {"bundle_key": bundle_key, "vector": vector, "texts": [text]}
            clusters.setdefault(group_key, []).append(selected)
        else:
            selected["texts"].append(text)
            vectors = [text_embedding(item) for item in selected["texts"]]
            averaged = [sum(values) / len(vectors) for values in zip(*vectors)]
            norm = math.sqrt(sum(value * value for value in averaged))
            selected["vector"] = [value / norm for value in averaged] if norm else averaged
        bundle_key = str(selected["bundle_key"])
        planned.append(
            replace(
                finding,
                bundle_key=bundle_key,
                parallel_lane=bundle_key,
                bundle_strategy="semantic_ast",
            )
        )
    return planned


def scan_objective_gaps(
    repo_root: Path,
    *,
    objective_path: Path,
    max_findings: int = 10,
    seen_fingerprints: Iterable[str] = (),
    force_goal_ids: Iterable[str] = (),
    embedding_min_score: float = DEFAULT_EMBEDDING_MIN_SCORE,
    summary_prefix: str = DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX,
    surplus_findings_per_goal: int = DEFAULT_SURPLUS_FINDINGS_PER_GOAL,
    surplus_min_terms_per_todo: int = DEFAULT_SURPLUS_MIN_TERMS_PER_TODO,
    dataset_dir: Path | None = None,
    dataset_id: str = "objective-ast",
    scan_stats: dict[str, Any] | None = None,
) -> list[ObjectiveFinding]:
    if max_findings <= 0 or not objective_path.exists():
        return []
    goals = [
        goal
        for goal in parse_goal_heap(objective_path.read_text(encoding="utf-8"))
        if goal.is_schedulable
    ]
    if not goals:
        return []
    graph = goal_graph(goals)
    required_terms: list[str] = []
    for goal in goals:
        required_terms.extend(goal.required_evidence)
    cached_records: list[dict[str, Any]] | None = None
    if dataset_dir is not None:
        artifact = persist_objective_ast_dataset(
            repo_root=repo_root,
            objective_path=objective_path,
            dataset_dir=dataset_dir,
            dataset_id=dataset_id,
        )
        cached_records = ObjectiveDatasetStore(dataset_dir).load_records(dataset_id)
        if scan_stats is not None:
            scan_stats.clear()
            scan_stats.update(artifact.to_dict())
    evidence = evidence_index(
        repo_root,
        objective_path=objective_path,
        terms=required_terms,
        embedding_min_score=embedding_min_score,
        records=cached_records,
    )
    seen = {str(item) for item in seen_fingerprints if str(item).strip()}
    forced_goal_ids = {str(item) for item in force_goal_ids if str(item).strip()}
    findings: list[ObjectiveFinding] = []
    candidate_limit = objective_scan_candidate_limit(
        max_findings=max_findings,
        surplus_findings_per_goal=surplus_findings_per_goal,
    )

    goals_by_id = {goal.goal_id: goal for goal in goals}
    scheduled_goals = [goals_by_id[record.goal_id] for record in objective_heap_schedule(goals) if record.goal_id in goals_by_id]
    for goal in scheduled_goals:
        terms = goal.required_evidence
        missing_terms = [term for term in terms if not evidence.get(term)]
        forced_goal = goal.goal_id in forced_goal_ids
        validation_gap = False
        if not missing_terms:
            if not forced_goal:
                continue
            missing_terms = objective_goal_validation_gap_terms(goal)
            if not missing_terms:
                continue
            validation_gap = True
        launch_validation_gap = validation_gap and objective_goal_requires_launch_playwright_validation(goal)
        fields = goal.fields
        present = {term: evidence.get(term, []) for term in terms if evidence.get(term)}
        explicit_bundle = bool(str(fields.get("bundle") or "").strip())
        for candidate_kind, candidate_missing_terms in surplus_missing_term_groups(
            missing_terms,
            surplus_findings_per_goal=surplus_findings_per_goal,
            min_terms_per_todo=surplus_min_terms_per_todo,
        ):
            if validation_gap:
                candidate_kind = "validation_gate"
            fingerprint = objective_fingerprint(goal, candidate_missing_terms)
            if fingerprint in seen and not forced_goal:
                continue
            bundle_key = goal.bundle_key(candidate_missing_terms)
            finding = ObjectiveFinding(
                fingerprint=fingerprint,
                goal_id=goal.goal_id,
                title=goal.title,
                summary=f"{summary_prefix}: {goal.title}",
                priority=str(fields.get("priority") or "P2"),
                track=str(fields.get("track") or "ops"),
                missing_evidence=candidate_missing_terms,
                present_evidence=present,
                evidence_methods=evidence_methods(present),
                objective_path=repo_relative_path(repo_root, objective_path),
                outputs=split_terms(str(fields.get("outputs") or "")),
                validation=objective_goal_validation(
                    goal,
                    f"test -f {repo_relative_path(repo_root, objective_path)}",
                ),
                goal=str(fields.get("goal") or ""),
                refinement=str(fields.get("refinement") or ""),
                gap_task=(
                    "Run and repair the launch readiness validation gate until the phone, desktop, "
                    "Swissknife, Hallucinate App, and Meta glasses Playwright checks pass."
                    if launch_validation_gap
                    else "Run and repair the objective validation command until it passes, then record the evidence."
                    if validation_gap
                    else str(fields.get("gap_task") or "")
                ),
                parent_goal_ids=goal.parent_goal_ids,
                graph_depth=int(graph["depths"].get(goal.goal_id, 0)),
                bundle_key=bundle_key,
                parallel_lane=str(fields.get("parallel_lane") or bundle_key),
                bundle_explicit=explicit_bundle,
                bundle_strategy="explicit" if explicit_bundle else "semantic_ast",
                embedding_query=str(fields.get("embedding_query") or fields.get("goal") or goal.title),
                ast_query=str(fields.get("ast_query") or ", ".join(terms)),
                conflict_policy=str(
                    fields.get("conflict_policy")
                    or "prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts"
                ),
                refinement_depth=str(fields.get("refinement_depth") or graph["depths"].get(goal.goal_id, 0)),
                candidate_kind=candidate_kind,
                surplus_group=objective_surplus_group(goal),
                merge_key=objective_merge_key(goal, candidate_missing_terms, candidate_kind=candidate_kind),
                merge_family=objective_surplus_group(goal),
                merge_role=candidate_kind,
                work_item_count=len(candidate_missing_terms),
                work_scope=(
                    "launch_validation_gate"
                    if launch_validation_gap
                    else "objective_validation_repair"
                    if validation_gap
                    else "goal_subgoal_multi_evidence_batch"
                ),
                todo_vector_key=objective_todo_vector_key(
                    goal,
                    candidate_missing_terms,
                    candidate_kind=candidate_kind,
                ),
                predicted_files=_unique_strings(
                    [
                        *split_terms(str(fields.get("outputs") or "")),
                        *_goal_conflict_terms(goal, "predicted_files", "files"),
                    ]
                ),
                changed_paths=_goal_conflict_terms(
                    goal,
                    "changed_paths",
                    "actual_changed_paths",
                    "branch_diff_paths",
                ),
                ast_symbols=_unique_strings(
                    [
                        *_goal_conflict_terms(goal, "ast_symbols"),
                        *split_terms(str(fields.get("ast_query") or ", ".join(terms))),
                    ]
                ),
                interfaces=_goal_conflict_terms(
                    goal,
                    "interfaces",
                    "interface_contracts",
                    "provides_interfaces",
                    "requires_interfaces",
                    "required_interfaces",
                    "interface_dependencies",
                    "public_interfaces",
                ),
                submodules=_goal_conflict_terms(
                    goal,
                    "submodules",
                    "submodule_paths",
                    "interoperability_pair",
                    "gitlinks",
                ),
                generated_artifacts=_goal_conflict_terms(
                    goal,
                    "generated_artifacts",
                    "generated_outputs",
                    "generated_paths",
                    "artifacts",
                ),
                allow_concurrent_with=_goal_conflict_terms(
                    goal,
                    "allow_concurrent_with",
                    "concurrency_overrides",
                ),
            )
            findings.append(finding)
            if not forced_goal:
                seen.add(fingerprint)
            if len(findings) >= candidate_limit:
                break
        if len(findings) >= candidate_limit:
            break
    packeted_findings = assign_goal_subgoal_packets(plan_semantic_ast_bundles(findings))
    expanded_findings = add_goal_packet_aggregate_findings(
        packeted_findings,
        max_findings=candidate_limit,
        seen_fingerprints=seen_fingerprints,
        summary_prefix=summary_prefix,
    )
    return prioritize_larger_work_surface_findings(expanded_findings, max_findings=max_findings)


def task_ids_from_todo(todo_text: str, *, task_prefix: str = DEFAULT_TASK_PREFIX) -> list[str]:
    ids: list[str] = []
    header_prefix = f"## {task_prefix}"
    for line in todo_text.splitlines():
        if line.startswith(header_prefix):
            parts = line[3:].strip().split(" ", 1)
            if parts:
                ids.append(parts[0])
    return ids


def next_task_id(
    todo_text: str,
    *,
    task_prefix: str = DEFAULT_TASK_PREFIX,
    reserved_task_ids: Iterable[str] = (),
) -> str:
    highest = 0
    normalized = task_prefix.rstrip("-")
    task_ids = [
        *task_ids_from_todo(todo_text, task_prefix=f"{normalized}-"),
        *(str(item) for item in reserved_task_ids),
    ]
    for task_id in task_ids:
        try:
            prefix, number = task_id.rsplit("-", 1)
            if prefix != normalized:
                continue
            highest = max(highest, int(number))
        except (IndexError, ValueError):
            continue
    return f"{normalized}-{highest + 1:03d}"


def bundle_path(bundle_dir: Path, bundle_key: str) -> Path:
    return bundle_dir / f"{safe_bundle_key(bundle_key)}.todo.md"


def write_discovery(
    *,
    discovery_dir: Path,
    task_id: str,
    finding: ObjectiveFinding,
) -> Path:
    date = datetime.now(timezone.utc).date().isoformat()
    path = discovery_dir / f"{date}-{task_id.lower()}-objective-gap-{finding.fingerprint[:12]}.md"
    discovery_dir.mkdir(parents=True, exist_ok=True)
    missing = "\n".join(f"- {term}" for term in finding.missing_evidence) or "- none"
    present_items: list[str] = []
    for term, paths in finding.present_evidence.items():
        present_items.append(f"- {term}: {', '.join(str(path) for path in paths)}")
    present = "\n".join(present_items) if present_items else "- none found for this goal"
    parents = ", ".join(finding.parent_goal_ids) or "none"
    packet_goals = ", ".join(finding.goal_packet_goal_ids) or "none"
    content = f"""# {task_id} Objective Goal Gap

Date: {date}
Fingerprint: {finding.fingerprint}
Goal id: {finding.goal_id}
Goal title: {finding.title}
Objective heap: {finding.objective_path}
Priority: {finding.priority}
Track: {finding.track}
Parent goals: {parents}
Graph depth: {finding.graph_depth}
Bundle: {finding.bundle_key}
Parallel lane: {finding.parallel_lane}
Bundle strategy: {finding.bundle_strategy}
Goal packet: {finding.goal_packet_key or "none"}
Goal packet role: {finding.goal_packet_role or "none"}
Goal packet goals: {packet_goals}
Goal packet task count: {finding.goal_packet_task_count}
Goal packet work item count: {finding.goal_packet_work_item_count}
Evidence methods: {", ".join(finding.evidence_methods) or "none"}
Embedding query: {finding.embedding_query}
AST query: {finding.ast_query}
Conflict policy: {finding.conflict_policy}
Predicted files: {", ".join(finding.predicted_files or finding.outputs) or "none"}
AST symbols: {", ".join(finding.ast_symbols) or "none"}
Interfaces: {", ".join(finding.interfaces) or "none"}
Submodules: {", ".join(finding.submodules) or "none"}
Generated artifacts: {", ".join(finding.generated_artifacts) or "none"}
Allow concurrent with: {", ".join(finding.allow_concurrent_with) or "none"}

## Goal

{finding.goal or finding.title}

## Missing Evidence

{missing}

## Present Evidence

{present}

## Suggested Handling

{finding.gap_task or "Close the missing evidence with focused code, tests, or documentation."}
"""
    path.write_text(content, encoding="utf-8")
    return path


def objective_finding_task_identity(task_id: str, finding: ObjectiveFinding) -> TaskIdentity:
    """Return the stable work identity for an objective finding."""

    return canonical_task_identity(
        {
            "task_id": task_id,
            "dedupe_key": f"objective-finding:{finding.fingerprint}",
        },
        board_namespace="objective-graph",
        source_path=finding.objective_path,
    )


def objective_finding_conflict_record(task_id: str, finding: ObjectiveFinding) -> dict[str, Any]:
    """Return the canonical conflict-surface fields for a generated finding."""

    identity = objective_finding_task_identity(task_id, finding)
    predicted_files = _unique_strings([*(finding.predicted_files or finding.outputs), *finding.outputs])
    return {
        "task_id": task_id,
        "canonical_task_cid": identity.canonical_task_cid,
        "task_cid": identity.canonical_task_cid,
        "predicted_files": predicted_files,
        "files": predicted_files,
        "changed_paths": _unique_strings(finding.changed_paths),
        "outputs": _unique_strings(finding.outputs),
        "ast_symbols": _unique_strings(finding.ast_symbols or split_terms(finding.ast_query)),
        "interfaces": _unique_strings(finding.interfaces),
        "submodules": _unique_strings(finding.submodules),
        "generated_artifacts": _unique_strings(finding.generated_artifacts),
        "allow_concurrent_with": _unique_strings(finding.allow_concurrent_with),
        "conflict_policy": finding.conflict_policy,
    }


def render_task_block(
    *,
    task_id: str,
    finding: ObjectiveFinding,
    discovery_path: Path,
    depends_on: Sequence[str] = (),
    bundle_shard: str = "",
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
) -> str:
    outputs = [discovery_output_path, finding.objective_path]
    outputs.extend(str(item) for item in finding.outputs if str(item).strip())
    unique_outputs = list(dict.fromkeys(outputs))
    missing = ", ".join(finding.missing_evidence)
    refinement = finding.refinement or "Refine the objective heap if the gap needs smaller child goals."
    parents = ", ".join(finding.parent_goal_ids) or "none"
    packet_goals = ", ".join(finding.goal_packet_goal_ids)
    packet_acceptance = (
        f"This task is part of {finding.goal_packet_key}; implement a complete, cohesive change that fully advances "
        f"the packet goals ({packet_goals}) and covers all the shared packet evidence in one comprehensive pass."
        if finding.goal_packet_key and packet_goals
        else ""
    )
    identity = objective_finding_task_identity(task_id, finding)
    bundle_shard = bundle_shard or f"data/agent_supervisor/objective_bundles/{safe_bundle_key(finding.bundle_key)}.todo.md"
    return f"""## {task_id} {finding.summary}

- Status: todo
- Completion: manual
- Priority: {finding.priority}
- Track: {finding.track}
- Depends on: {", ".join(depends_on)}
- Outputs: {", ".join(unique_outputs)}
- Validation: {finding.validation}
- Bundle: {finding.bundle_key}
- Bundle shard: {bundle_shard}
- Bundle strategy: {finding.bundle_strategy}
- Graph parents: {parents}
- Graph depth: {finding.graph_depth}
- Parallel lane: {finding.parallel_lane}
- Conflict policy: {finding.conflict_policy}
- Predicted files: {", ".join(finding.predicted_files or finding.outputs)}
- Changed paths: {", ".join(finding.changed_paths)}
- AST symbols: {", ".join(finding.ast_symbols)}
- Interfaces: {", ".join(finding.interfaces)}
- Submodules: {", ".join(finding.submodules)}
- Generated artifacts: {", ".join(finding.generated_artifacts)}
- Allow concurrent with: {", ".join(finding.allow_concurrent_with)}
- Goal id: {finding.goal_id}
- Canonical task key: {identity.canonical_task_key}
- Canonical task CID: {identity.canonical_task_cid}
- Missing evidence: {missing}
- Embedding query: {finding.embedding_query}
- AST query: {finding.ast_query}
- Surplus group: {finding.surplus_group}
- Merge key: {finding.merge_key}
- Merge family: {finding.merge_family or finding.surplus_group}
- Merge role: {finding.merge_role or finding.candidate_kind}
- Work item count: {finding.work_item_count or len(finding.missing_evidence)}
- Work scope: {finding.work_scope or "goal_subgoal_multi_evidence_batch"}
- Goal packet: {finding.goal_packet_key}
- Goal packet role: {finding.goal_packet_role}
- Goal packet goals: {packet_goals}
- Goal packet task count: {finding.goal_packet_task_count}
- Goal packet work item count: {finding.goal_packet_work_item_count}
- Candidate kind: {finding.candidate_kind}
- Todo vector key: {finding.todo_vector_key}
- Acceptance: Objective scan filed this gap for {finding.goal_id}. Use evidence in {discovery_path}, add code/tests/docs or child goals that prove the missing evidence terms are covered ({missing}), and keep the supervisor-fed backlog aligned with the objective heap. {packet_acceptance} {refinement}
"""


def write_bundle_shards(
    *,
    bundle_dir: Path,
    repo_root: Path,
    todo_path: Path,
    records: Sequence[ObjectiveTaskRecord],
) -> BundleWriteResult:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []
    bundle_paths: dict[str, Path] = {}
    source_todo = repo_relative_path(repo_root, todo_path)
    groups: dict[str, list[ObjectiveTaskRecord]] = {}
    for record in records:
        groups.setdefault(record.finding.bundle_key, []).append(record)

    generated_planning_graph = materialize_task_planning_graph(
        [
            {
                **objective_finding_conflict_record(record.task_id, record.finding),
                "task_id": record.task_id,
                "canonical_task_cid": objective_finding_task_identity(record.task_id, record.finding).canonical_task_cid,
                "goal_id": record.finding.goal_id,
                "parent_goal_ids": record.finding.parent_goal_ids,
                "priority": record.finding.priority,
                "outputs": record.finding.outputs,
                "work_item_count": record.finding.work_item_count or len(record.finding.missing_evidence),
                "status": "todo",
            }
            for record in records
        ]
    )
    generated_graph = generated_planning_graph.dependency_graph
    generated_incoming: dict[str, set[str]] = {cid: set() for cid in generated_graph.nodes}
    for edge in generated_graph.edges:
        generated_incoming.setdefault(edge.target_task_cid, set()).add(edge.source_task_cid)
    generated_schedule = {item.task_cid: item for item in generated_graph.schedule}

    for key, bundle_records in sorted(groups.items()):
        shard_path = bundle_path(bundle_dir, key)
        bundle_paths[key] = shard_path
        if shard_path.exists():
            shard_text = shard_path.read_text(encoding="utf-8")
        else:
            shard_text = (
                f"# Objective Bundle: {key}\n\n"
                f"Source todo: {source_todo}\n"
                "Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.\n"
                "Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.\n"
            )

        changed = False
        for record in bundle_records:
            if f"## {record.task_id} " in shard_text:
                continue
            shard_text = shard_text.rstrip() + "\n\n" + record.task_block.strip() + "\n"
            changed = True
        if changed or not shard_path.exists():
            shard_path.write_text(shard_text, encoding="utf-8")
            generated_paths.append(shard_path)

    index_path = bundle_dir / "index.json"
    if index_path.exists():
        try:
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            index_payload = {}
    else:
        index_payload = {}
    if not isinstance(index_payload, dict):
        index_payload = {}
    bundles = index_payload.get("bundles")
    if not isinstance(bundles, dict):
        bundles = {}

    for key, bundle_records in sorted(groups.items()):
        task_map: dict[str, dict[str, Any]] = {}
        existing = bundles.get(key, {})
        if isinstance(existing, Mapping):
            for item in existing.get("tasks", []) if isinstance(existing.get("tasks"), list) else []:
                if isinstance(item, Mapping) and str(item.get("task_id") or ""):
                    task_map[str(item["task_id"])] = dict(item)
        for record in bundle_records:
            identity = objective_finding_task_identity(record.task_id, record.finding)
            schedule_record = generated_schedule.get(identity.canonical_task_cid)
            task_map[record.task_id] = {
                **objective_finding_conflict_record(record.task_id, record.finding),
                "task_id": record.task_id,
                "canonical_task_key": identity.canonical_task_key,
                "canonical_task_cid": identity.canonical_task_cid,
                "goal_id": record.finding.goal_id,
                "graph_depth": record.finding.graph_depth,
                "parent_goal_ids": record.finding.parent_goal_ids,
                "missing_evidence": record.finding.missing_evidence,
                "discovery_path": repo_relative_path(repo_root, record.discovery_path),
                "bundle_strategy": record.finding.bundle_strategy,
                "surplus_group": record.finding.surplus_group,
                "merge_key": record.finding.merge_key,
                "merge_family": record.finding.merge_family or record.finding.surplus_group,
                "merge_role": record.finding.merge_role or record.finding.candidate_kind,
                "work_item_count": record.finding.work_item_count or len(record.finding.missing_evidence),
                "work_scope": record.finding.work_scope or "goal_subgoal_multi_evidence_batch",
                "goal_packet_key": record.finding.goal_packet_key,
                "goal_packet_role": record.finding.goal_packet_role,
                "goal_packet_goal_ids": record.finding.goal_packet_goal_ids,
                "goal_packet_task_count": record.finding.goal_packet_task_count,
                "goal_packet_work_item_count": record.finding.goal_packet_work_item_count,
                "candidate_kind": record.finding.candidate_kind,
                "todo_vector_key": record.finding.todo_vector_key,
                "dependency_task_cids": sorted(generated_incoming.get(identity.canonical_task_cid, set())),
                "critical_path_length": schedule_record.critical_path_length if schedule_record else 1,
                "slack": schedule_record.slack if schedule_record else 0,
                "downstream_unlock_value": schedule_record.downstream_unlock_value if schedule_record else 0,
                "objective_priority": schedule_record.objective_priority if schedule_record else 0,
            }
        bundles[key] = {
            "bundle_key": key,
            "shard_path": repo_relative_path(repo_root, bundle_path(bundle_dir, key)),
            "parallel_lane": bundle_records[0].finding.parallel_lane,
            "bundle_strategy": bundle_records[0].finding.bundle_strategy,
            "conflict_policy": bundle_records[0].finding.conflict_policy,
            "tasks": [task_map[task_id] for task_id in sorted(task_map)],
        }

    all_index_tasks = [
        dict(item)
        for info in bundles.values()
        if isinstance(info, Mapping)
        for item in (info.get("tasks") or [])
        if isinstance(item, Mapping)
    ]
    index_planning_graph = materialize_task_planning_graph(all_index_tasks, repo_root=repo_root)
    index_graph = index_planning_graph.dependency_graph
    index_incoming: dict[str, set[str]] = {cid: set() for cid in index_graph.nodes}
    for edge in index_graph.edges:
        index_incoming.setdefault(edge.target_task_cid, set()).add(edge.source_task_cid)
    index_schedule = {item.task_cid: item for item in index_graph.schedule}
    for info in bundles.values():
        if not isinstance(info, dict):
            continue
        annotated: list[dict[str, Any]] = []
        for raw_task in info.get("tasks") or []:
            if not isinstance(raw_task, Mapping):
                continue
            item = dict(raw_task)
            cid = str(item.get("canonical_task_cid") or item.get("task_cid") or "")
            scheduled = index_schedule.get(cid)
            item["dependency_task_cids"] = sorted(index_incoming.get(cid, set()))
            if scheduled:
                item.update(
                    {
                        "critical_path_length": scheduled.critical_path_length,
                        "slack": scheduled.slack,
                        "downstream_unlock_value": scheduled.downstream_unlock_value,
                        "age_seconds": scheduled.age_seconds,
                        "objective_priority": scheduled.objective_priority,
                        "schedule_score": scheduled.score,
                    }
                )
            annotated.append(item)
        info["tasks"] = annotated

    index_payload["generated_at"] = utc_now()
    index_payload["source_todo"] = source_todo
    index_payload["bundles"] = bundles
    index_payload["task_dependency_graph"] = index_graph.to_dict()
    index_payload["dependency_dag"] = index_graph.to_dict()
    index_payload["task_conflict_graph"] = index_planning_graph.conflict_graph.to_dict()
    index_payload["conflict_graph"] = index_planning_graph.conflict_graph.to_dict()
    index_payload["task_planning_graph"] = index_planning_graph.to_dict()
    index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    generated_paths.append(index_path)
    return BundleWriteResult(generated_paths=generated_paths, index_path=index_path, bundle_paths=bundle_paths)


def generate_objective_todos(
    *,
    repo_root: Path,
    objective_path: Path,
    todo_path: Path,
    discovery_dir: Path,
    bundle_dir: Path,
    dataset_dir: Path | None = None,
    task_prefix: str = DEFAULT_TASK_PREFIX,
    depends_on: Sequence[str] = (),
    max_findings: int = 10,
    seen_fingerprints: Iterable[str] = (),
    force_goal_ids: Iterable[str] = (),
    persist_ast_dataset: bool = True,
    write_todo_vector_index: bool = True,
    todo_vector_index_path: Path | None = None,
    surplus_findings_per_goal: int = DEFAULT_SURPLUS_FINDINGS_PER_GOAL,
    surplus_min_terms_per_todo: int = DEFAULT_SURPLUS_MIN_TERMS_PER_TODO,
    summary_prefix: str = DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX,
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
) -> list[ObjectiveTaskRecord]:
    """Append generated objective gap tasks and write bundle shards."""

    records: list[ObjectiveTaskRecord] = []
    findings = scan_objective_gaps(
        repo_root,
        objective_path=objective_path,
        max_findings=max_findings,
        seen_fingerprints=seen_fingerprints,
        force_goal_ids=force_goal_ids,
        summary_prefix=summary_prefix,
        surplus_findings_per_goal=surplus_findings_per_goal,
        surplus_min_terms_per_todo=surplus_min_terms_per_todo,
        dataset_dir=(dataset_dir or bundle_dir.parent / "objective_datasets") if persist_ast_dataset else None,
        dataset_id=f"{task_prefix.rstrip('-').lower()}-objective-ast",
    )
    with locked_taskboard(todo_path) as taskboard:
        todo_text = taskboard.read() or "# Objective Todo\n"
        reserved_task_ids = task_ids_from_artifact_names(
            discovery_dir,
            task_prefix=task_prefix,
        )
        for finding in findings:
            task_id = next_task_id(
                todo_text,
                task_prefix=task_prefix,
                reserved_task_ids=reserved_task_ids,
            )
            reserved_task_ids.add(task_id)
            shard_relative = repo_relative_path(
                repo_root, bundle_path(bundle_dir, finding.bundle_key)
            )
            discovery_path = write_discovery(
                discovery_dir=discovery_dir,
                task_id=task_id,
                finding=finding,
            )
            task_block = render_task_block(
                task_id=task_id,
                finding=finding,
                discovery_path=discovery_path,
                depends_on=depends_on,
                bundle_shard=shard_relative,
                discovery_output_path=discovery_output_path,
            )
            todo_text = todo_text.rstrip() + "\n\n" + task_block.strip() + "\n"
            records.append(
                ObjectiveTaskRecord(
                    task_id=task_id,
                    task_block=task_block,
                    finding=finding,
                    discovery_path=discovery_path,
                )
            )

        if records:
            replace_locked_taskboard(taskboard, todo_text)

    if not records:
        return []
    bundle_result = write_bundle_shards(bundle_dir=bundle_dir, repo_root=repo_root, todo_path=todo_path, records=records)
    if write_todo_vector_index:
        from .todo_vector_index import write_todo_vector_index as write_index

        task_header_prefix = task_prefix.strip()
        if not task_header_prefix.startswith("## "):
            task_header_prefix = f"## {task_header_prefix.rstrip('-')}-"
        write_index(
            repo_root=repo_root,
            todo_path=todo_path,
            index_path=todo_vector_index_path or bundle_dir / "todo_vector_index.json",
            task_header_prefix=task_header_prefix,
            objective_path=objective_path,
            bundle_index_path=bundle_result.index_path,
            dataset_dir=(dataset_dir or bundle_dir.parent / "objective_datasets") if persist_ast_dataset else None,
            dataset_id=f"{task_prefix.rstrip('-').lower()}-todo-vector-index",
            persist_dataset=persist_ast_dataset,
        )
    return records


def generate_objective_todos_result(
    *,
    scan_mode: str = "direct",
    **kwargs: Any,
) -> RefillScanResult[ObjectiveTaskRecord]:
    """Generate objective todos and describe the terminal scan outcome.

    ``generate_objective_todos`` remains the list-returning primitive for
    callers that are explicitly operating on task records.  Refill and goal
    completion code should use this typed boundary so an empty collection can
    never be mistaken for successful exhaustion without a terminal reason.
    """

    started_at = datetime.now(timezone.utc)
    repo_root = Path(kwargs.get("repo_root") or ".").resolve()
    try:
        max_findings = int(kwargs.get("max_findings", 10))
    except (TypeError, ValueError):
        max_findings = 0
    if max_findings <= 0:
        return build_scan_result(
            ScanTerminalReason.DISABLED,
            scan_mode,
            OBJECTIVE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            metadata={"cause": "non_positive_max_findings"},
        )

    objective_path = Path(kwargs.get("objective_path") or "")
    todo_path = Path(kwargs.get("todo_path") or "")
    missing_inputs = [
        name
        for name, path in (("objective_path", objective_path), ("todo_path", todo_path))
        if not path.exists()
    ]
    if missing_inputs:
        return build_scan_result(
            ScanTerminalReason.FAILED,
            scan_mode,
            OBJECTIVE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            error=f"missing required scan input: {', '.join(missing_inputs)}",
            metadata={"missing_inputs": missing_inputs},
        )

    try:
        records = generate_objective_todos(**kwargs)
    except TimeoutError as exc:
        return build_scan_result(
            ScanTerminalReason.TIMED_OUT,
            scan_mode,
            OBJECTIVE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            error=str(exc) or type(exc).__name__,
        )
    except Exception as exc:
        return build_scan_result(
            ScanTerminalReason.FAILED,
            scan_mode,
            OBJECTIVE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            error=f"{type(exc).__name__}: {exc}",
        )

    terminal_reason = ScanTerminalReason.GENERATED
    duplicate_candidate_count = 0
    if not records:
        terminal_reason = ScanTerminalReason.EXHAUSTED
        seen_fingerprints = tuple(kwargs.get("seen_fingerprints") or ())
        if seen_fingerprints:
            try:
                duplicate_candidates = scan_objective_gaps(
                    repo_root,
                    objective_path=objective_path,
                    max_findings=max_findings,
                    seen_fingerprints=(),
                    force_goal_ids=kwargs.get("force_goal_ids") or (),
                    summary_prefix=str(
                        kwargs.get("summary_prefix") or DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX
                    ),
                    surplus_findings_per_goal=int(
                        kwargs.get("surplus_findings_per_goal", DEFAULT_SURPLUS_FINDINGS_PER_GOAL)
                    ),
                    surplus_min_terms_per_todo=int(
                        kwargs.get("surplus_min_terms_per_todo", DEFAULT_SURPLUS_MIN_TERMS_PER_TODO)
                    ),
                )
            except TimeoutError as exc:
                return build_scan_result(
                    ScanTerminalReason.TIMED_OUT,
                    scan_mode,
                    OBJECTIVE_SCAN_ANALYZER_VERSION,
                    repo_root,
                    started_at,
                    error=str(exc) or type(exc).__name__,
                )
            except Exception as exc:
                return build_scan_result(
                    ScanTerminalReason.FAILED,
                    scan_mode,
                    OBJECTIVE_SCAN_ANALYZER_VERSION,
                    repo_root,
                    started_at,
                    error=f"{type(exc).__name__}: {exc}",
                )
            duplicate_candidate_count = len(duplicate_candidates)
            if duplicate_candidates:
                terminal_reason = ScanTerminalReason.DUPLICATE_ONLY
    return build_scan_result(
        terminal_reason,
        scan_mode,
        OBJECTIVE_SCAN_ANALYZER_VERSION,
        repo_root,
        started_at,
        records,
        safe_for_completion_reasoning=(
            terminal_reason is ScanTerminalReason.EXHAUSTED
            and str(scan_mode).endswith("exhaustive")
        ),
        metadata={
            "candidate_count": len(records),
            "duplicate_candidate_count": duplicate_candidate_count,
        },
    )


def _profile_g_safe_planning_value(value: Any) -> Any:
    """Encode graph weights without violating Profile G's no-float codec."""

    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return format(value, ".12g")
    if isinstance(value, Mapping):
        return {str(key): _profile_g_safe_planning_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_profile_g_safe_planning_value(item) for item in value]
    return value


def _project_task_conflict_graph(
    graph: Mapping[str, Any], member_task_cids: set[str]
) -> dict[str, Any]:
    """Return the incident conflict evidence needed by one bundle payload.

    The complete O(n²) decision matrix belongs once in the bundle index.  Queue
    payloads carry only their member surfaces, assignments, and incident
    decisions/edges; otherwise serializing and hashing n full graphs creates
    cubic planning work for large objective boards.
    """

    def incident(item: Any) -> bool:
        if not isinstance(item, Mapping):
            return False
        left = str(item.get("left_task_cid") or item.get("left") or "")
        right = str(item.get("right_task_cid") or item.get("right") or "")
        return bool({left, right} & member_task_cids)

    surfaces = graph.get("surfaces") if isinstance(graph.get("surfaces"), Mapping) else {}
    assignments = graph.get("assignments") if isinstance(graph.get("assignments"), list) else []
    lanes = graph.get("lanes") if isinstance(graph.get("lanes"), Mapping) else {}
    projected_lanes: dict[str, list[str]] = {}
    for color, raw_task_cids in lanes.items():
        if not isinstance(raw_task_cids, list):
            continue
        selected = [str(cid) for cid in raw_task_cids if str(cid) in member_task_cids]
        if selected:
            projected_lanes[str(color)] = selected
    return {
        "schema": graph.get("schema", "ipfs_accelerate_py.agent_supervisor.conflict_graph@1"),
        "projection": "bundle_incident",
        "surfaces": {
            cid: dict(surface)
            for cid, surface in surfaces.items()
            if str(cid) in member_task_cids and isinstance(surface, Mapping)
        },
        "edges": [dict(item) for item in graph.get("edges", []) if incident(item)],
        "assignments": [
            dict(item)
            for item in assignments
            if isinstance(item, Mapping) and str(item.get("task_cid") or "") in member_task_cids
        ],
        "decisions": [dict(item) for item in graph.get("decisions", []) if incident(item)],
        "lanes": projected_lanes,
    }


def build_bundle_task_payloads(bundle_index_path: Path) -> list[dict[str, Any]]:
    """Build dependency-aware task-queue payloads and Profile G adapters."""

    # Local import avoids making objective scanning depend on coordination
    # initialization while ensuring queue consumers receive immutable links.
    from .lease_coordination import adapt_goal_bundle

    payload = json.loads(bundle_index_path.read_text(encoding="utf-8"))
    bundles = payload.get("bundles") if isinstance(payload, Mapping) else {}
    if not isinstance(bundles, Mapping):
        return []
    task_payloads: list[dict[str, Any]] = []
    for key, info in sorted(bundles.items()):
        if not isinstance(info, Mapping):
            continue
        task_payload = {
            "bundle_key": str(key),
            "todo_path": info.get("shard_path", ""),
            "parallel_lane": info.get("parallel_lane", key),
            "conflict_policy": info.get("conflict_policy", ""),
            "tasks": info.get("tasks", []),
            "source_todo": payload.get("source_todo", ""),
            "objective_bundle_index": str(bundle_index_path),
        }
        task_payloads.append(task_payload)

    flat_tasks = [
        {**dict(item), "bundle_key": str(bundle_payload.get("bundle_key") or "objective/general")}
        for bundle_payload in task_payloads
        for item in bundle_payload.get("tasks", [])
        if isinstance(item, Mapping)
    ]
    terminal_receipts = {
        str(task.get("canonical_task_cid") or task.get("task_cid") or task.get("task_id") or ""): {
            "status": "succeeded"
        }
        for task in flat_tasks
        if str(task.get("status") or "").strip().lower() in SUCCESSFUL_MERGE_RECEIPT_STATUSES
        and str(task.get("canonical_task_cid") or task.get("task_cid") or task.get("task_id") or "")
    }
    planning_graph = materialize_task_planning_graph(flat_tasks, merge_receipts=terminal_receipts)
    graph = planning_graph.dependency_graph
    graph_dict = graph.to_dict()
    conflict_graph_dict = planning_graph.conflict_graph.to_dict()
    invalid_task_cids = set(graph.invalid_task_cids)
    schedule_by_cid = {item.task_cid: item for item in graph.schedule}
    member_cids_by_bundle_and_id = {
        (str(node.metadata.get("bundle_key") or ""), node.task_id): cid
        for cid, node in graph.nodes.items()
    }
    incoming: dict[str, set[str]] = {cid: set() for cid in graph.nodes}
    for edge in graph.edges:
        incoming.setdefault(edge.target_task_cid, set()).add(edge.source_task_cid)
    unresolved_incoming = {
        cid: (
            set()
            if node.status in SUCCESSFUL_MERGE_RECEIPT_STATUSES
            else set(schedule_by_cid.get(cid).blocking_task_cids if cid in schedule_by_cid else incoming.get(cid, set()))
        )
        for cid, node in graph.nodes.items()
    }

    member_bundle: dict[str, str] = {}
    bundle_identity_cids: dict[str, str] = {}
    for bundle_payload in task_payloads:
        bundle_key = str(bundle_payload["bundle_key"])
        bundle_identity_cids[bundle_key] = canonical_bundle_identity(bundle_payload).canonical_task_cid
        annotated_tasks: list[dict[str, Any]] = []
        for raw_task in bundle_payload.get("tasks", []):
            if not isinstance(raw_task, Mapping):
                continue
            item = dict(raw_task)
            cid = str(item.get("canonical_task_cid") or item.get("task_cid") or "")
            if not cid:
                cid = member_cids_by_bundle_and_id.get((bundle_key, str(item.get("task_id") or "")), "")
            if not cid:
                cid = canonical_task_identity(
                    {
                        **item,
                        "semantic_key": f"{bundle_key}:{item.get('task_id') or len(annotated_tasks)}",
                    }
                ).canonical_task_cid
            item["canonical_task_cid"] = cid
            member_bundle[cid] = bundle_key
            item["dependency_task_cids"] = sorted(incoming.get(cid, set()))
            scheduled = schedule_by_cid.get(cid)
            if scheduled:
                item.update(
                    {
                        "claimable": scheduled.claimable,
                        "blocking_task_cids": scheduled.blocking_task_cids,
                        "critical_path_length": scheduled.critical_path_length,
                        "slack": scheduled.slack,
                        "downstream_unlock_value": scheduled.downstream_unlock_value,
                        "age_seconds": scheduled.age_seconds,
                        "objective_priority": scheduled.objective_priority,
                        "schedule_score": scheduled.score,
                    }
                )
            annotated_tasks.append(item)
        bundle_payload["tasks"] = annotated_tasks

    for bundle_payload in task_payloads:
        bundle_key = str(bundle_payload["bundle_key"])
        member_cids = {
            str(item.get("canonical_task_cid") or item.get("task_cid") or "")
            for item in bundle_payload.get("tasks", [])
            if isinstance(item, Mapping)
        }
        dependency_bundle_keys = {
            member_bundle[source]
            for target in member_cids
            for source in unresolved_incoming.get(target, set())
            if source in member_bundle and member_bundle[source] != bundle_key
        }
        dependency_task_cids = sorted(bundle_identity_cids[key] for key in dependency_bundle_keys)
        member_schedule = [schedule_by_cid[cid] for cid in member_cids if cid in schedule_by_cid]
        repair_evidence: list[dict[str, Any]] = []
        bundle_resolved_cycle_cids: set[str] = set()
        for item in graph.repair_evidence:
            if item.task_cid not in member_cids:
                continue
            component = {
                str(cid)
                for cid in item.provenance.get("component_task_cids", [])
                if str(cid)
            }
            if item.kind == "dependency_cycle" and component and component <= member_cids:
                # A lane owns every member of this strongly connected
                # component, so the cycle is an internal implementation order
                # concern rather than an impossible cross-lane prerequisite.
                bundle_resolved_cycle_cids.update(component)
                continue
            repair_evidence.append(item.to_dict())
        blocking_repair_cids = {
            str(item.get("task_cid") or "")
            for item in repair_evidence
            if str(item.get("task_cid") or "")
        }
        invalid_member_cids = sorted(
            blocking_repair_cids
            | ((member_cids & invalid_task_cids) - bundle_resolved_cycle_cids)
        )
        if invalid_member_cids and not repair_evidence:
            # The graph keeps the complete invalid-CID set even when detailed
            # repair evidence reaches its global bound. Preserve one compact
            # bundle-local marker so truncation can never make invalid work
            # appear claimable at the lease boundary.
            repair_evidence.append(
                {
                    "kind": "missing_dependency",
                    "task_cid": invalid_member_cids[0],
                    "task_id": "",
                    "reference": "bounded_dependency_repair_evidence",
                    "message": "dependency repair details were truncated; regenerate or repair the task DAG",
                    "provenance": {
                        "invalid_task_cids": invalid_member_cids[:16],
                        "evidence_truncated": True,
                    },
                }
            )
        external_blockers = sorted(
            {
                bundle_identity_cids[member_bundle[source]]
                for target in member_cids
                for source in unresolved_incoming.get(target, set())
                if source in member_bundle and member_bundle[source] != bundle_key
            }
        )
        projected_conflicts = _project_task_conflict_graph(conflict_graph_dict, member_cids)
        bundle_payload.update(
            {
                "canonical_task_cid": bundle_identity_cids[bundle_key],
                "dependency_task_cids": dependency_task_cids,
                "blocking_task_cids": external_blockers,
                "claimable": not external_blockers and not invalid_member_cids,
                "critical_path_length": max((item.critical_path_length for item in member_schedule), default=1),
                "slack": min((item.slack for item in member_schedule), default=0),
                "downstream_unlock_value": max(
                    (item.downstream_unlock_value for item in member_schedule), default=0
                ),
                "age_seconds": max((item.age_seconds for item in member_schedule), default=0),
                "objective_priority": max((item.objective_priority for item in member_schedule), default=0),
                "schedule_score": max((item.score for item in member_schedule), default=0),
                "dependency_repair_evidence": repair_evidence,
                "task_dependency_graph": graph_dict,
                "dependency_dag": graph_dict,
                "task_conflict_graph": projected_conflicts,
                "conflict_graph": projected_conflicts,
                "conflict_planning_decisions": projected_conflicts["decisions"],
            }
        )

    task_payloads.sort(
        key=lambda item: (
            0 if item["claimable"] else 1,
            -int(item["critical_path_length"]),
            int(item["slack"]),
            -int(item["downstream_unlock_value"]),
            -int(item["age_seconds"]),
            -int(item["objective_priority"]),
            str(item["bundle_key"]),
        )
    )
    for rank, task_payload in enumerate(task_payloads):
        task_payload["schedule_rank"] = rank
        task_payload["profile_g"] = adapt_goal_bundle(_profile_g_safe_planning_value(task_payload))
    return task_payloads


def submit_bundle_tasks(
    bundle_index_path: Path,
    *,
    queue: Any = None,
    queue_path: str | None = None,
    task_type: str = "codex.todo_bundle",
    model_name: str = "codex",
) -> list[str]:
    """Submit bundle shards to the ipfs_accelerate task queue.

    The queue parameter is injectable for tests.  When omitted, the local
    ``TaskQueue`` is used without importing ipfs_datasets_py.
    """

    if queue is None:
        from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue

        queue = TaskQueue(path=queue_path)
    task_ids: list[str] = []
    for payload in build_bundle_task_payloads(bundle_index_path):
        task_ids.append(
            queue.submit(
                task_type=task_type,
                model_name=model_name,
                payload=payload,
            )
        )
    return task_ids
