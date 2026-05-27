"""Objective-graph scanner and bundle planner for autonomous agent todos.

This module ports the objective-driven backlog generation that was previously
implemented as repository scripts into ``ipfs_accelerate_py``.  It is designed
for acceleration-oriented agent systems: one objective heap can generate many
bundle-local todo shards, and those shards can be submitted to the existing P2P
task queue so multiple Codex workers can drain independent lanes.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .dataset_store import DatasetArtifact, ObjectiveDatasetStore


DEFAULT_EMBEDDING_DIMENSIONS = int(os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_EMBEDDING_DIMENSIONS", "64"))
DEFAULT_EMBEDDING_MIN_SCORE = float(os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_EMBEDDING_MIN_SCORE", "0.62"))
DEFAULT_BUNDLE_CLUSTER_MIN_SCORE = float(os.environ.get("IPFS_ACCELERATE_AGENT_BUNDLE_CLUSTER_MIN_SCORE", "0.42"))
DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX = os.environ.get(
    "IPFS_ACCELERATE_AGENT_OBJECTIVE_TASK_SUMMARY_PREFIX",
    "Close objective gap",
)
DEFAULT_DISCOVERY_OUTPUT_PATH = os.environ.get(
    "IPFS_ACCELERATE_AGENT_DISCOVERY_OUTPUT_PATH",
    "data/agent_supervisor/discovery",
)
DEFAULT_TASK_PREFIX = "AUTO-"
DEFAULT_AST_DATASET_MAX_CHARS = int(os.environ.get("IPFS_ACCELERATE_AGENT_AST_DATASET_MAX_CHARS", "1000000"))
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
            tree = ast.parse(text)
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
            tree = ast.parse(text)
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
) -> list[dict[str, Any]]:
    """Collect AST/symbol records for dataset-backed objective scans."""

    rows: list[dict[str, Any]] = []
    for path in objective_candidate_files(repo_root, objective_path=objective_path):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        symbols = sorted(symbol_terms(path, text))
        payload = ast_dataset_payload(path, text, max_chars=max_ast_chars)
        rows.append(
            {
                "root_relative_path": repo_relative_path(repo_root, path),
                "suffix": path.suffix.lower(),
                "source_sha1": sha1(text.encode("utf-8", errors="replace")).hexdigest(),
                "source_bytes": len(text.encode("utf-8", errors="replace")),
                "symbols_json": json.dumps(symbols, sort_keys=True),
                "token_count": len(objective_tokens(text)),
                **payload,
            }
        )
    return rows


def persist_objective_ast_dataset(
    *,
    repo_root: Path,
    objective_path: Path,
    dataset_dir: Path,
    dataset_id: str = "objective-ast",
) -> DatasetArtifact:
    """Persist scan AST/symbol records with the optional ipfs_datasets backend."""

    store = ObjectiveDatasetStore(dataset_dir)
    return store.persist_records(
        dataset_id=dataset_id,
        records=collect_ast_dataset_records(repo_root, objective_path=objective_path),
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
    return files


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
    for path in objective_candidate_files(repo_root, objective_path=objective_path):
        root_relative = repo_relative_path(repo_root, path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        symbols = symbol_terms(path, text)
        symbol_text = " ".join(sorted(symbols))
        document_text = f"{root_relative}\n{symbol_text}\n{text[:12000]}"
        document_embedding = text_embedding(document_text)
        document_tokens = set(objective_tokens(document_text))
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

    return {
        "nodes": sorted(nodes),
        "edges": edges,
        "children": {key: sorted(value) for key, value in children.items() if value},
        "roots": sorted(roots),
        "depths": depths,
    }


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


def surplus_missing_term_groups(
    missing_terms: Sequence[str],
    *,
    surplus_findings_per_goal: int = 1,
    min_terms_per_todo: int = 2,
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
    embedding_min_score: float = DEFAULT_EMBEDDING_MIN_SCORE,
    summary_prefix: str = DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX,
    surplus_findings_per_goal: int = 1,
    surplus_min_terms_per_todo: int = 2,
) -> list[ObjectiveFinding]:
    if max_findings <= 0 or not objective_path.exists():
        return []
    goals = [
        goal
        for goal in parse_goal_heap(objective_path.read_text(encoding="utf-8"))
        if goal.status in {"active", "todo", "open"}
    ]
    if not goals:
        return []
    graph = goal_graph(goals)
    required_terms: list[str] = []
    for goal in goals:
        required_terms.extend(goal.required_evidence)
    evidence = evidence_index(
        repo_root,
        objective_path=objective_path,
        terms=required_terms,
        embedding_min_score=embedding_min_score,
    )
    seen = {str(item) for item in seen_fingerprints if str(item).strip()}
    findings: list[ObjectiveFinding] = []

    for goal in sorted(goals, key=lambda item: item.priority):
        terms = goal.required_evidence
        missing_terms = [term for term in terms if not evidence.get(term)]
        if not missing_terms:
            continue
        fields = goal.fields
        present = {term: evidence.get(term, []) for term in terms if evidence.get(term)}
        explicit_bundle = bool(str(fields.get("bundle") or "").strip())
        for candidate_kind, candidate_missing_terms in surplus_missing_term_groups(
            missing_terms,
            surplus_findings_per_goal=surplus_findings_per_goal,
            min_terms_per_todo=surplus_min_terms_per_todo,
        ):
            fingerprint = objective_fingerprint(goal, candidate_missing_terms)
            if fingerprint in seen:
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
                validation=str(fields.get("validation") or f"test -f {repo_relative_path(repo_root, objective_path)}"),
                goal=str(fields.get("goal") or ""),
                refinement=str(fields.get("refinement") or ""),
                gap_task=str(fields.get("gap_task") or ""),
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
                work_scope="goal_subgoal_multi_evidence_batch",
                todo_vector_key=objective_todo_vector_key(
                    goal,
                    candidate_missing_terms,
                    candidate_kind=candidate_kind,
                ),
            )
            findings.append(finding)
            seen.add(fingerprint)
            if len(findings) >= max_findings:
                break
        if len(findings) >= max_findings:
            break
    return plan_semantic_ast_bundles(findings)


def task_ids_from_todo(todo_text: str, *, task_prefix: str = DEFAULT_TASK_PREFIX) -> list[str]:
    ids: list[str] = []
    header_prefix = f"## {task_prefix}"
    for line in todo_text.splitlines():
        if line.startswith(header_prefix):
            parts = line[3:].strip().split(" ", 1)
            if parts:
                ids.append(parts[0])
    return ids


def next_task_id(todo_text: str, *, task_prefix: str = DEFAULT_TASK_PREFIX) -> str:
    highest = 0
    normalized = task_prefix.rstrip("-")
    for task_id in task_ids_from_todo(todo_text, task_prefix=f"{normalized}-"):
        try:
            highest = max(highest, int(task_id.split("-", 1)[1]))
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
Evidence methods: {", ".join(finding.evidence_methods) or "none"}
Embedding query: {finding.embedding_query}
AST query: {finding.ast_query}
Conflict policy: {finding.conflict_policy}

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
- Goal id: {finding.goal_id}
- Missing evidence: {missing}
- Embedding query: {finding.embedding_query}
- AST query: {finding.ast_query}
- Surplus group: {finding.surplus_group}
- Merge key: {finding.merge_key}
- Merge family: {finding.merge_family or finding.surplus_group}
- Merge role: {finding.merge_role or finding.candidate_kind}
- Work item count: {finding.work_item_count or len(finding.missing_evidence)}
- Work scope: {finding.work_scope or "goal_subgoal_multi_evidence_batch"}
- Candidate kind: {finding.candidate_kind}
- Todo vector key: {finding.todo_vector_key}
- Acceptance: Objective scan filed this gap for {finding.goal_id}. Use evidence in {discovery_path}, add code/tests/docs or child goals that prove the missing evidence terms are covered ({missing}), and keep the supervisor-fed backlog aligned with the objective heap. {refinement}
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
            task_map[record.task_id] = {
                "task_id": record.task_id,
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
                "candidate_kind": record.finding.candidate_kind,
                "todo_vector_key": record.finding.todo_vector_key,
            }
        bundles[key] = {
            "bundle_key": key,
            "shard_path": repo_relative_path(repo_root, bundle_path(bundle_dir, key)),
            "parallel_lane": bundle_records[0].finding.parallel_lane,
            "bundle_strategy": bundle_records[0].finding.bundle_strategy,
            "conflict_policy": bundle_records[0].finding.conflict_policy,
            "tasks": [task_map[task_id] for task_id in sorted(task_map)],
        }

    index_payload["generated_at"] = utc_now()
    index_payload["source_todo"] = source_todo
    index_payload["bundles"] = bundles
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
    persist_ast_dataset: bool = True,
    write_todo_vector_index: bool = True,
    todo_vector_index_path: Path | None = None,
    surplus_findings_per_goal: int = 1,
    surplus_min_terms_per_todo: int = 2,
    summary_prefix: str = DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX,
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
) -> list[ObjectiveTaskRecord]:
    """Append generated objective gap tasks and write bundle shards."""

    todo_text = todo_path.read_text(encoding="utf-8") if todo_path.exists() else "# Objective Todo\n"
    records: list[ObjectiveTaskRecord] = []
    for finding in scan_objective_gaps(
        repo_root,
        objective_path=objective_path,
        max_findings=max_findings,
        seen_fingerprints=seen_fingerprints,
        summary_prefix=summary_prefix,
        surplus_findings_per_goal=surplus_findings_per_goal,
        surplus_min_terms_per_todo=surplus_min_terms_per_todo,
    ):
        task_id = next_task_id(todo_text, task_prefix=task_prefix)
        shard_relative = repo_relative_path(repo_root, bundle_path(bundle_dir, finding.bundle_key))
        discovery_path = write_discovery(discovery_dir=discovery_dir, task_id=task_id, finding=finding)
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

    if not records:
        return []
    todo_path.parent.mkdir(parents=True, exist_ok=True)
    todo_path.write_text(todo_text, encoding="utf-8")
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
    if persist_ast_dataset:
        persist_objective_ast_dataset(
            repo_root=repo_root,
            objective_path=objective_path,
            dataset_dir=dataset_dir or bundle_dir.parent / "objective_datasets",
            dataset_id=f"{task_prefix.rstrip('-').lower()}-objective-ast",
        )
    return records


def build_bundle_task_payloads(bundle_index_path: Path) -> list[dict[str, Any]]:
    """Build task-queue payloads for each bundle shard in an index."""

    payload = json.loads(bundle_index_path.read_text(encoding="utf-8"))
    bundles = payload.get("bundles") if isinstance(payload, Mapping) else {}
    if not isinstance(bundles, Mapping):
        return []
    tasks: list[dict[str, Any]] = []
    for key, info in sorted(bundles.items()):
        if not isinstance(info, Mapping):
            continue
        tasks.append(
            {
                "bundle_key": str(key),
                "todo_path": info.get("shard_path", ""),
                "parallel_lane": info.get("parallel_lane", key),
                "conflict_policy": info.get("conflict_policy", ""),
                "tasks": info.get("tasks", []),
                "source_todo": payload.get("source_todo", ""),
                "objective_bundle_index": str(bundle_index_path),
            }
        )
    return tasks


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
