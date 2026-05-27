"""Vector and AST indexes for autonomous-agent todo boards.

The objective scanner can intentionally create more candidate todos than a
single daemon should read at once.  This module keeps those candidates compact:
it parses the todo board into structured rows, builds deterministic text
embeddings and AST/symbol hints, and writes a small JSON index that bundle
supervisors can use to keep related work together.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dataset_store import DatasetArtifact, ObjectiveDatasetStore
from .objective_graph import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_BUNDLE_CLUSTER_MIN_SCORE,
    cosine,
    objective_tokens,
    repo_relative_path,
    repo_relative_path_safe,
    safe_bundle_key,
    symbol_terms,
    text_embedding,
)


DEFAULT_TODO_VECTOR_INDEX_SCHEMA = "ipfs_accelerate_py.agent_supervisor.todo_vector_index"


@dataclass(frozen=True)
class TodoIndexRecord:
    """A compact semantic representation of one markdown todo task."""

    task_id: str
    title: str
    status: str
    priority: str
    track: str
    source_line: int
    bundle_key: str = ""
    bundle_shard: str = ""
    bundle_strategy: str = ""
    goal_id: str = ""
    graph_parents: list[str] = field(default_factory=list)
    graph_depth: int = 0
    missing_evidence: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    validation: list[str] = field(default_factory=list)
    acceptance: str = ""
    embedding_query: str = ""
    ast_query: str = ""
    conflict_policy: str = ""
    surplus_group: str = ""
    merge_key: str = ""
    candidate_kind: str = ""
    vector_key: str = ""
    token_count: int = 0
    embedding: list[float] = field(default_factory=list)
    ast_symbols: list[str] = field(default_factory=list)
    related_task_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def split_csv(value: str) -> list[str]:
    items: list[str] = []
    for raw in str(value or "").split(","):
        item = " ".join(raw.strip().split())
        if item and item.lower() not in {"none", "n/a"} and item not in items:
            items.append(item)
    return items


def normalize_metadata_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def parse_todo_blocks(todo_text: str, *, task_header_prefix: str) -> list[tuple[str, str, int, dict[str, str]]]:
    """Parse markdown todo blocks and keep all metadata fields."""

    prefix = task_header_prefix.strip()
    if not prefix.startswith("## "):
        prefix = f"## {prefix}"
    blocks: list[tuple[str, str, int, dict[str, str]]] = []
    current_id = ""
    current_title = ""
    current_line = 0
    current_fields: dict[str, str] = {}

    def flush() -> None:
        nonlocal current_id, current_title, current_line, current_fields
        if current_id:
            blocks.append((current_id, current_title, current_line, dict(current_fields)))
        current_id = ""
        current_title = ""
        current_line = 0
        current_fields = {}

    for line_number, line in enumerate(todo_text.splitlines(), start=1):
        if line.startswith(prefix):
            flush()
            header = line[3:].strip()
            parts = header.split(" ", 1)
            current_id = parts[0] if parts else ""
            current_title = parts[1].strip() if len(parts) > 1 else ""
            current_line = line_number
            continue
        if not current_id:
            continue
        stripped = line.strip()
        if not stripped.startswith("- ") or ":" not in stripped:
            continue
        key, value = stripped[2:].split(":", 1)
        current_fields[normalize_metadata_key(key)] = value.strip()
    flush()
    return blocks


def infer_goal_id(fields: Mapping[str, str], acceptance: str) -> str:
    direct = str(fields.get("goal_id") or "").strip()
    if direct:
        return direct
    match = re.search(r"\bfor\s+([A-Z][A-Z0-9_]*-G\d+|[A-Z][A-Z0-9_]*-\d+)\b", acceptance)
    return match.group(1) if match else ""


def infer_missing_evidence(fields: Mapping[str, str], acceptance: str) -> list[str]:
    direct = split_csv(str(fields.get("missing_evidence") or ""))
    if direct:
        return direct
    match = re.search(r"missing evidence terms are covered\s+\(([^)]+)\)", acceptance, flags=re.IGNORECASE)
    return split_csv(match.group(1)) if match else []


def infer_merge_key(
    *,
    task_id: str,
    goal_id: str,
    surplus_group: str,
    missing_evidence: Sequence[str],
    outputs: Sequence[str],
    ast_query: str,
) -> str:
    payload = {
        "goal_id": goal_id,
        "surplus_group": surplus_group,
        "missing_evidence": sorted(str(item) for item in missing_evidence),
        "outputs": sorted(str(item) for item in outputs),
        "ast_query": ast_query,
    }
    if not goal_id and not surplus_group and not missing_evidence:
        payload["task_id"] = task_id
    return sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def record_embedding_text(record: TodoIndexRecord) -> str:
    return "\n".join(
        [
            record.task_id,
            record.title,
            record.priority,
            record.track,
            record.bundle_key,
            record.goal_id,
            record.surplus_group,
            record.embedding_query,
            record.ast_query,
            " ".join(record.graph_parents),
            " ".join(record.missing_evidence),
            " ".join(record.outputs),
            record.acceptance,
        ]
    )


def collect_output_symbols(repo_root: Path, outputs: Sequence[str], *, max_file_bytes: int = 262144) -> list[str]:
    symbols: set[str] = set()
    for output in outputs:
        relative = str(output).strip()
        if not repo_relative_path_safe(relative):
            continue
        path = repo_root / relative
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > max_file_bytes:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        symbols.update(symbol_terms(path, text))
    return sorted(symbols)


def parse_todo_vector_records(
    *,
    repo_root: Path,
    todo_path: Path,
    task_header_prefix: str,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> list[TodoIndexRecord]:
    """Return vector-index records for every task in a todo markdown file."""

    if not todo_path.exists():
        return []
    todo_text = todo_path.read_text(encoding="utf-8")
    records: list[TodoIndexRecord] = []
    for task_id, title, source_line, fields in parse_todo_blocks(todo_text, task_header_prefix=task_header_prefix):
        outputs = split_csv(fields.get("outputs", ""))
        acceptance = str(fields.get("acceptance") or "")
        missing_evidence = infer_missing_evidence(fields, acceptance)
        goal_id = infer_goal_id(fields, acceptance)
        surplus_group = str(fields.get("surplus_group") or goal_id or "").strip()
        ast_query = str(fields.get("ast_query") or "").strip()
        merge_key = str(fields.get("merge_key") or "").strip() or infer_merge_key(
            task_id=task_id,
            goal_id=goal_id,
            surplus_group=surplus_group,
            missing_evidence=missing_evidence,
            outputs=outputs,
            ast_query=ast_query,
        )
        vector_key = str(fields.get("todo_vector_key") or "").strip() or sha1(
            f"{task_id}\0{merge_key}".encode("utf-8")
        ).hexdigest()[:16]
        base_record = TodoIndexRecord(
            task_id=task_id,
            title=title,
            status=str(fields.get("status") or "todo").strip().lower(),
            priority=str(fields.get("priority") or "P2").strip().upper(),
            track=str(fields.get("track") or "ops").strip().lower(),
            source_line=source_line,
            bundle_key=str(fields.get("bundle") or "").strip(),
            bundle_shard=str(fields.get("bundle_shard") or "").strip(),
            bundle_strategy=str(fields.get("bundle_strategy") or "").strip(),
            goal_id=goal_id,
            graph_parents=split_csv(fields.get("graph_parents", "")),
            graph_depth=parse_int(fields.get("graph_depth"), 0),
            missing_evidence=missing_evidence,
            outputs=outputs,
            validation=[item.strip() for item in str(fields.get("validation") or "").split(";") if item.strip()],
            acceptance=acceptance,
            embedding_query=str(fields.get("embedding_query") or "").strip(),
            ast_query=ast_query,
            conflict_policy=str(fields.get("conflict_policy") or "").strip(),
            surplus_group=surplus_group,
            merge_key=merge_key,
            candidate_kind=str(fields.get("candidate_kind") or "").strip(),
            vector_key=vector_key,
            ast_symbols=collect_output_symbols(repo_root, outputs),
        )
        text = record_embedding_text(base_record)
        records.append(
            TodoIndexRecord(
                **{
                    **base_record.to_dict(),
                    "token_count": len(objective_tokens(text)),
                    "embedding": text_embedding(text, dimensions=dimensions),
                }
            )
        )
    return attach_related_task_ids(records)


def attach_related_task_ids(records: Sequence[TodoIndexRecord], *, max_related: int = 5) -> list[TodoIndexRecord]:
    """Annotate records with nearest related tasks for compact prompt context."""

    related: list[TodoIndexRecord] = []
    for record in records:
        scored: list[tuple[float, str]] = []
        record_symbols = set(record.ast_symbols)
        for other in records:
            if other.task_id == record.task_id:
                continue
            score = cosine(record.embedding, other.embedding)
            if record.merge_key and record.merge_key == other.merge_key:
                score += 1.0
            elif record.surplus_group and record.surplus_group == other.surplus_group:
                score += 0.50
            elif record.bundle_key and record.bundle_key == other.bundle_key:
                score += 0.20
            other_symbols = set(other.ast_symbols)
            if record_symbols and other_symbols:
                score += min(0.25, len(record_symbols & other_symbols) / max(1, len(record_symbols | other_symbols)))
            if score > 0:
                scored.append((score, other.task_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        related.append(replace_record(record, related_task_ids=[task_id for _score, task_id in scored[:max_related]]))
    return related


def replace_record(record: TodoIndexRecord, **changes: Any) -> TodoIndexRecord:
    payload = record.to_dict()
    payload.update(changes)
    return TodoIndexRecord(**payload)


def cluster_records(
    records: Sequence[TodoIndexRecord],
    *,
    min_score: float = DEFAULT_BUNDLE_CLUSTER_MIN_SCORE,
) -> list[dict[str, Any]]:
    """Cluster todo records by explicit bundle, merge key, AST overlap, and vectors."""

    clusters: list[dict[str, Any]] = []
    for record in records:
        selected: dict[str, Any] | None = None
        best_score = -1.0
        for cluster in clusters:
            if record.bundle_key and cluster.get("bundle_key") == record.bundle_key:
                selected = cluster
                best_score = 1.0
                break
            if record.merge_key and record.merge_key in cluster.get("merge_keys", []):
                selected = cluster
                best_score = 1.0
                break
            score = cosine(record.embedding, cluster.get("centroid", []))
            if record.surplus_group and record.surplus_group in cluster.get("surplus_groups", []):
                score += 0.35
            if score > best_score:
                best_score = score
                selected = cluster
        if selected is None or best_score < min_score:
            key_source = record.bundle_key or record.surplus_group or record.merge_key or record.task_id
            selected = {
                "cluster_key": f"todo/{safe_bundle_key(record.track or 'ops')}/{sha1(key_source.encode('utf-8')).hexdigest()[:8]}",
                "bundle_key": record.bundle_key,
                "task_ids": [],
                "merge_keys": [],
                "surplus_groups": [],
                "ast_symbols": [],
                "centroid": record.embedding,
                "estimated_prompt_tokens": 0,
            }
            clusters.append(selected)
        selected["task_ids"].append(record.task_id)
        if record.merge_key and record.merge_key not in selected["merge_keys"]:
            selected["merge_keys"].append(record.merge_key)
        if record.surplus_group and record.surplus_group not in selected["surplus_groups"]:
            selected["surplus_groups"].append(record.surplus_group)
        selected_symbols = set(selected.get("ast_symbols") or [])
        selected_symbols.update(record.ast_symbols)
        selected["ast_symbols"] = sorted(selected_symbols)[:200]
        selected["estimated_prompt_tokens"] = int(selected.get("estimated_prompt_tokens") or 0) + record.token_count
        vectors = [item.embedding for item in records if item.task_id in selected["task_ids"]]
        if vectors:
            averaged = [sum(values) / len(vectors) for values in zip(*vectors)]
            norm = math.sqrt(sum(value * value for value in averaged))
            selected["centroid"] = [value / norm for value in averaged] if norm else averaged

    for cluster in clusters:
        cluster["task_ids"] = sorted(cluster["task_ids"])
        cluster["merge_keys"] = sorted(cluster["merge_keys"])
        cluster["surplus_groups"] = sorted(cluster["surplus_groups"])
        cluster["centroid_sha1"] = sha1(
            json.dumps(cluster.pop("centroid", []), sort_keys=True).encode("utf-8")
        ).hexdigest()
    return sorted(clusters, key=lambda item: (str(item.get("bundle_key") or ""), str(item.get("cluster_key") or "")))


def write_todo_vector_index(
    *,
    repo_root: Path,
    todo_path: Path,
    index_path: Path,
    task_header_prefix: str,
    objective_path: Path | None = None,
    bundle_index_path: Path | None = None,
    dataset_dir: Path | None = None,
    dataset_id: str = "todo-vector-index",
    persist_dataset: bool = False,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> dict[str, Any]:
    """Build and persist a vector/AST index for a todo board."""

    records = parse_todo_vector_records(
        repo_root=repo_root,
        todo_path=todo_path,
        task_header_prefix=task_header_prefix,
        dimensions=dimensions,
    )
    clusters = cluster_records(records)
    payload: dict[str, Any] = {
        "schema": DEFAULT_TODO_VECTOR_INDEX_SCHEMA,
        "generated_at": utc_now(),
        "repo_root": str(repo_root),
        "todo_path": repo_relative_path(repo_root, todo_path),
        "objective_path": repo_relative_path(repo_root, objective_path) if objective_path else "",
        "task_header_prefix": task_header_prefix,
        "embedding_dimensions": dimensions,
        "task_count": len(records),
        "active_task_count": sum(1 for record in records if record.status not in {"completed", "blocked"}),
        "records": [record.to_dict() for record in records],
        "clusters": clusters,
    }
    if bundle_index_path is not None:
        payload["bundle_index_path"] = repo_relative_path(repo_root, bundle_index_path)
    if persist_dataset and dataset_dir is not None:
        artifact = persist_todo_vector_dataset(
            dataset_dir=dataset_dir,
            dataset_id=dataset_id,
            records=records,
        )
        payload["dataset_artifact"] = artifact.to_dict()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if bundle_index_path is not None and bundle_index_path.exists():
        update_bundle_index_with_todo_vectors(bundle_index_path=bundle_index_path, records=records, clusters=clusters)
    return payload


def persist_todo_vector_dataset(
    *,
    dataset_dir: Path,
    dataset_id: str,
    records: Sequence[TodoIndexRecord],
) -> DatasetArtifact:
    store = ObjectiveDatasetStore(dataset_dir)
    return store.persist_records(
        dataset_id=dataset_id,
        records=[record.to_dict() for record in records],
    )


def update_bundle_index_with_todo_vectors(
    *,
    bundle_index_path: Path,
    records: Sequence[TodoIndexRecord],
    clusters: Sequence[Mapping[str, Any]],
) -> None:
    try:
        payload = json.loads(bundle_index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    bundles = payload.get("bundles")
    if not isinstance(bundles, dict):
        return
    by_task = {record.task_id: record for record in records}
    by_bundle: dict[str, list[TodoIndexRecord]] = {}
    for record in records:
        if record.bundle_key:
            by_bundle.setdefault(record.bundle_key, []).append(record)
    cluster_by_task: dict[str, str] = {}
    for cluster in clusters:
        cluster_key = str(cluster.get("cluster_key") or "")
        for task_id in cluster.get("task_ids", []) if isinstance(cluster.get("task_ids"), list) else []:
            cluster_by_task[str(task_id)] = cluster_key
    for bundle_key, bundle_payload in bundles.items():
        if not isinstance(bundle_payload, dict):
            continue
        bundle_records = by_bundle.get(str(bundle_key), [])
        bundle_payload["todo_vector_summary"] = {
            "task_count": len(bundle_records),
            "merge_keys": sorted({record.merge_key for record in bundle_records if record.merge_key}),
            "surplus_groups": sorted({record.surplus_group for record in bundle_records if record.surplus_group}),
            "estimated_prompt_tokens": sum(record.token_count for record in bundle_records),
        }
        tasks = bundle_payload.get("tasks")
        if not isinstance(tasks, list):
            continue
        for task in tasks:
            if not isinstance(task, dict):
                continue
            record = by_task.get(str(task.get("task_id") or ""))
            if record is None:
                continue
            task["merge_key"] = record.merge_key
            task["surplus_group"] = record.surplus_group
            task["todo_vector_key"] = record.vector_key
            task["todo_cluster_key"] = cluster_by_task.get(record.task_id, "")
            task["related_task_ids"] = record.related_task_ids
    bundle_index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
