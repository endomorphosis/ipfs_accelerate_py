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
DEFAULT_EXECUTION_PACKET_MAX_TASKS = 6


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
    merge_family: str = ""
    merge_role: str = ""
    work_item_count: int = 0
    work_scope: str = ""
    goal_packet_key: str = ""
    goal_packet_role: str = ""
    goal_packet_goal_ids: list[str] = field(default_factory=list)
    goal_packet_task_count: int = 0
    goal_packet_work_item_count: int = 0
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
            record.merge_family,
            record.merge_role,
            str(record.work_item_count),
            record.work_scope,
            record.goal_packet_key,
            record.goal_packet_role,
            " ".join(record.goal_packet_goal_ids),
            str(record.goal_packet_task_count),
            str(record.goal_packet_work_item_count),
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
        candidate_kind = str(fields.get("candidate_kind") or "").strip()
        merge_family = str(fields.get("merge_family") or surplus_group or goal_id or merge_key).strip()
        merge_role = str(fields.get("merge_role") or candidate_kind or "candidate").strip()
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
            merge_family=merge_family,
            merge_role=merge_role,
            work_item_count=parse_int(fields.get("work_item_count"), len(missing_evidence)),
            work_scope=str(fields.get("work_scope") or "").strip(),
            goal_packet_key=str(fields.get("goal_packet") or fields.get("goal_packet_key") or "").strip(),
            goal_packet_role=str(fields.get("goal_packet_role") or "").strip(),
            goal_packet_goal_ids=split_csv(fields.get("goal_packet_goals") or fields.get("goal_packet_goal_ids") or ""),
            goal_packet_task_count=parse_int(fields.get("goal_packet_task_count"), 0),
            goal_packet_work_item_count=parse_int(fields.get("goal_packet_work_item_count"), 0),
            candidate_kind=candidate_kind,
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
            elif record.merge_family and record.merge_family == other.merge_family:
                score += 0.70
            elif record.goal_packet_key and record.goal_packet_key == other.goal_packet_key:
                score += 0.65
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
            if record.merge_family and record.merge_family in cluster.get("merge_families", []):
                selected = cluster
                best_score = 0.9
                break
            if record.goal_packet_key and record.goal_packet_key in cluster.get("goal_packet_keys", []):
                selected = cluster
                best_score = 0.85
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
                "merge_families": [],
                "goal_packet_keys": [],
                "surplus_groups": [],
                "ast_symbols": [],
                "centroid": record.embedding,
                "estimated_prompt_tokens": 0,
            }
            clusters.append(selected)
        selected["task_ids"].append(record.task_id)
        if record.merge_key and record.merge_key not in selected["merge_keys"]:
            selected["merge_keys"].append(record.merge_key)
        if record.merge_family and record.merge_family not in selected["merge_families"]:
            selected["merge_families"].append(record.merge_family)
        if record.goal_packet_key and record.goal_packet_key not in selected["goal_packet_keys"]:
            selected["goal_packet_keys"].append(record.goal_packet_key)
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
        cluster["merge_families"] = sorted(cluster["merge_families"])
        cluster["goal_packet_keys"] = sorted(cluster["goal_packet_keys"])
        cluster["surplus_groups"] = sorted(cluster["surplus_groups"])
        cluster["centroid_sha1"] = sha1(
            json.dumps(cluster.pop("centroid", []), sort_keys=True).encode("utf-8")
        ).hexdigest()
    return sorted(clusters, key=lambda item: (str(item.get("bundle_key") or ""), str(item.get("cluster_key") or "")))


def active_record(record: TodoIndexRecord) -> bool:
    return record.status not in {"blocked", "completed"}


def sorted_unique(values: Sequence[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def build_merge_candidate(
    *,
    group_type: str,
    group_value: str,
    records: Sequence[TodoIndexRecord],
    cluster_by_task: Mapping[str, str],
) -> dict[str, Any] | None:
    task_ids = sorted_unique([record.task_id for record in records])
    if len(task_ids) < 2:
        return None
    active_task_ids = sorted_unique([record.task_id for record in records if active_record(record)])
    if not active_task_ids:
        return None
    all_outputs = sorted_unique([output for record in records for output in record.outputs])
    output_sets = [set(record.outputs) for record in records if record.outputs]
    shared_outputs = sorted(output_sets[0].intersection(*output_sets[1:])) if output_sets else []
    ast_symbols = sorted_unique([symbol for record in records for symbol in record.ast_symbols])[:80]
    missing_evidence = sorted_unique([item for record in records for item in record.missing_evidence])
    work_counts = [record.work_item_count for record in records if record.work_item_count > 0]
    packet_work_counts = [record.goal_packet_work_item_count for record in records if record.goal_packet_work_item_count > 0]
    graph_depths = [record.graph_depth for record in records if record.graph_depth >= 0]
    candidate_seed = json.dumps({"group_type": group_type, "group_value": group_value, "task_ids": task_ids}, sort_keys=True)
    exact_merge_key_count = len({record.merge_key for record in records if record.merge_key})
    if group_type == "merge_key":
        confidence = "high"
    elif group_type == "goal_packet_key":
        confidence = "high" if len({record.goal_id for record in records if record.goal_id}) > 1 else "medium"
    elif group_type == "merge_family" and shared_outputs:
        confidence = "high"
    elif group_type == "merge_family":
        confidence = "medium"
    elif group_type == "surplus_group" and exact_merge_key_count <= max(1, len(records) // 2):
        confidence = "medium"
    else:
        confidence = "low"
    merge_ready_task_ids = (
        active_task_ids
        if len(active_task_ids) > 1
        and (group_type in {"merge_key", "goal_packet_key", "merge_family", "surplus_group"} or bool(shared_outputs))
        else []
    )
    return {
        "candidate_key": f"{group_type}/{sha1(candidate_seed.encode('utf-8')).hexdigest()[:12]}",
        "group_type": group_type,
        "group_value": group_value,
        "confidence": confidence,
        "task_ids": task_ids,
        "active_task_ids": active_task_ids,
        "completed_task_ids": sorted_unique([record.task_id for record in records if record.status == "completed"]),
        "blocked_task_ids": sorted_unique([record.task_id for record in records if record.status == "blocked"]),
        "goal_ids": sorted_unique([record.goal_id for record in records]),
        "graph_parent_ids": sorted_unique([parent for record in records for parent in record.graph_parents]),
        "graph_depth_min": min(graph_depths) if graph_depths else 0,
        "graph_depth_max": max(graph_depths) if graph_depths else 0,
        "bundle_keys": sorted_unique([record.bundle_key for record in records]),
        "merge_keys": sorted_unique([record.merge_key for record in records]),
        "merge_families": sorted_unique([record.merge_family for record in records]),
        "merge_roles": sorted_unique([record.merge_role for record in records]),
        "goal_packet_keys": sorted_unique([record.goal_packet_key for record in records]),
        "goal_packet_roles": sorted_unique([record.goal_packet_role for record in records]),
        "goal_packet_goal_ids": sorted_unique([goal_id for record in records for goal_id in record.goal_packet_goal_ids]),
        "goal_packet_task_count_max": max([record.goal_packet_task_count for record in records], default=0),
        "goal_packet_work_item_count_max": max(packet_work_counts) if packet_work_counts else 0,
        "surplus_groups": sorted_unique([record.surplus_group for record in records]),
        "cluster_keys": sorted_unique([cluster_by_task.get(record.task_id, "") for record in records]),
        "shared_outputs": shared_outputs,
        "all_outputs": all_outputs,
        "missing_evidence": missing_evidence,
        "ast_symbols": ast_symbols,
        "work_item_count_min": min(work_counts) if work_counts else 0,
        "work_item_count_max": max(work_counts) if work_counts else 0,
        "work_item_count_total": sum(work_counts),
        "merge_ready_task_ids": merge_ready_task_ids,
        "estimated_prompt_tokens": sum(record.token_count for record in records),
    }


def build_merge_candidates(
    records: Sequence[TodoIndexRecord],
    clusters: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return compact groups of todos that can be reasoned about together."""

    cluster_by_task: dict[str, str] = {}
    for cluster in clusters:
        cluster_key = str(cluster.get("cluster_key") or "")
        if not cluster_key:
            continue
        task_ids = cluster.get("task_ids")
        if not isinstance(task_ids, list):
            continue
        for task_id in task_ids:
            cluster_by_task[str(task_id)] = cluster_key

    groups: list[tuple[str, str, list[TodoIndexRecord]]] = []
    for group_type, getter in (
        ("merge_key", lambda record: record.merge_key),
        ("goal_packet_key", lambda record: record.goal_packet_key),
        ("merge_family", lambda record: record.merge_family),
        ("surplus_group", lambda record: record.surplus_group),
    ):
        by_value: dict[str, list[TodoIndexRecord]] = {}
        for record in records:
            value = str(getter(record) or "")
            if value:
                by_value.setdefault(value, []).append(record)
        groups.extend((group_type, value, group_records) for value, group_records in by_value.items())

    records_by_task = {record.task_id: record for record in records}
    for cluster in clusters:
        cluster_key = str(cluster.get("cluster_key") or "")
        task_ids = cluster.get("task_ids")
        if not cluster_key or not isinstance(task_ids, list):
            continue
        cluster_records_for_key = [records_by_task[task_id] for task_id in map(str, task_ids) if task_id in records_by_task]
        groups.append(("vector_cluster", cluster_key, cluster_records_for_key))

    candidates: list[dict[str, Any]] = []
    seen_task_sets: set[tuple[str, ...]] = set()
    for group_type, group_value, group_records in groups:
        candidate = build_merge_candidate(
            group_type=group_type,
            group_value=group_value,
            records=group_records,
            cluster_by_task=cluster_by_task,
        )
        if candidate is None:
            continue
        task_set = tuple(candidate["task_ids"])
        if task_set in seen_task_sets:
            continue
        seen_task_sets.add(task_set)
        candidates.append(candidate)

    confidence_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        candidates,
        key=lambda candidate: (
            confidence_order.get(str(candidate.get("confidence") or ""), 9),
            -len(candidate.get("active_task_ids") or []),
            int(candidate.get("estimated_prompt_tokens") or 0),
            str(candidate.get("candidate_key") or ""),
        ),
    )


def _compact_context_text(context: Mapping[str, Any]) -> str:
    parts = [
        str(context.get("context_key") or ""),
        f"merge_ready={str(bool(context.get('merge_ready'))).lower()}",
        f"active={', '.join(context.get('active_task_ids') or [])}",
        f"goals={', '.join(context.get('goal_ids') or [])}",
        f"parents={', '.join(context.get('graph_parent_ids') or [])}",
        f"merge_family={', '.join(context.get('merge_families') or [])}",
        f"goal_packet={', '.join(context.get('goal_packet_keys') or [])}",
        f"work_items={context.get('work_item_count_min')}-{context.get('work_item_count_max')}",
        f"packet_work={context.get('goal_packet_work_item_count_max') or 0}",
        f"missing={', '.join(context.get('missing_evidence') or [])}",
        f"outputs={', '.join((context.get('shared_outputs') or context.get('all_outputs') or [])[:4])}",
        f"ast={', '.join((context.get('ast_symbols') or [])[:12])}",
    ]
    return "; ".join(part for part in parts if not part.endswith("=") and part.strip())


def build_bundle_context(
    *,
    source_type: str,
    source_key: str,
    confidence: str,
    records: Sequence[TodoIndexRecord],
) -> dict[str, Any] | None:
    """Build one compact prompt context from goal/subgoal-related todos."""

    if not records:
        return None
    task_ids = sorted_unique([record.task_id for record in records])
    if len(task_ids) < 2:
        return None
    active_task_ids = sorted_unique([record.task_id for record in records if active_record(record)])
    if not active_task_ids:
        return None
    all_outputs = sorted_unique([output for record in records for output in record.outputs])
    output_sets = [set(record.outputs) for record in records if record.outputs]
    shared_outputs = sorted(output_sets[0].intersection(*output_sets[1:])) if output_sets else []
    graph_depths = [record.graph_depth for record in records if record.graph_depth >= 0]
    work_counts = [record.work_item_count for record in records if record.work_item_count > 0]
    merge_families = sorted_unique([record.merge_family for record in records])
    context_seed = json.dumps(
        {"source_type": source_type, "source_key": source_key, "task_ids": task_ids},
        sort_keys=True,
    )
    merge_ready = len(active_task_ids) > 1 and (
        bool(shared_outputs)
        or source_type in {"merge_candidate", "merge_key", "goal_packet_key", "merge_family", "surplus_group"}
        or bool(merge_families)
    )
    packet_work_counts = [record.goal_packet_work_item_count for record in records if record.goal_packet_work_item_count > 0]
    representative_task_id = active_task_ids[0]
    context: dict[str, Any] = {
        "context_key": f"bundle_context/{sha1(context_seed.encode('utf-8')).hexdigest()[:12]}",
        "source_type": source_type,
        "source_key": source_key,
        "confidence": confidence,
        "task_ids": task_ids,
        "active_task_ids": active_task_ids,
        "representative_task_id": representative_task_id,
        "merge_ready": merge_ready,
        "merge_ready_task_ids": active_task_ids if merge_ready else [],
        "goal_ids": sorted_unique([record.goal_id for record in records]),
        "graph_parent_ids": sorted_unique([parent for record in records for parent in record.graph_parents]),
        "graph_depth_min": min(graph_depths) if graph_depths else 0,
        "graph_depth_max": max(graph_depths) if graph_depths else 0,
        "bundle_keys": sorted_unique([record.bundle_key for record in records]),
        "merge_keys": sorted_unique([record.merge_key for record in records]),
        "merge_families": merge_families,
        "merge_roles": sorted_unique([record.merge_role for record in records]),
        "goal_packet_keys": sorted_unique([record.goal_packet_key for record in records]),
        "goal_packet_roles": sorted_unique([record.goal_packet_role for record in records]),
        "goal_packet_goal_ids": sorted_unique([goal_id for record in records for goal_id in record.goal_packet_goal_ids]),
        "goal_packet_task_count_max": max([record.goal_packet_task_count for record in records], default=0),
        "goal_packet_work_item_count_max": max(packet_work_counts) if packet_work_counts else 0,
        "work_scopes": sorted_unique([record.work_scope for record in records]),
        "work_item_count_min": min(work_counts) if work_counts else 0,
        "work_item_count_max": max(work_counts) if work_counts else 0,
        "work_item_count_total": sum(work_counts),
        "surplus_groups": sorted_unique([record.surplus_group for record in records]),
        "candidate_kinds": sorted_unique([record.candidate_kind for record in records]),
        "shared_outputs": shared_outputs,
        "all_outputs": all_outputs,
        "validation": sorted_unique([command for record in records for command in record.validation])[:8],
        "missing_evidence": sorted_unique([item for record in records for item in record.missing_evidence]),
        "ast_symbols": sorted_unique([symbol for record in records for symbol in record.ast_symbols])[:80],
        "raw_prompt_tokens": sum(record.token_count for record in records),
    }
    compact_context = _compact_context_text(context)
    context["compact_context"] = compact_context
    context["compact_context_tokens"] = len(objective_tokens(compact_context))
    return context


def build_bundle_contexts(
    records: Sequence[TodoIndexRecord],
    clusters: Sequence[Mapping[str, Any]],
    merge_candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build token-efficient contexts that bundle related goal/subgoal todos."""

    records_by_task = {record.task_id: record for record in records}
    contexts: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    def add_context(source_type: str, source_key: str, confidence: str, task_ids: Sequence[str]) -> None:
        selected = [records_by_task[task_id] for task_id in map(str, task_ids) if task_id in records_by_task]
        task_set = tuple(sorted(record.task_id for record in selected))
        if len(task_set) < 2 or task_set in seen:
            return
        context = build_bundle_context(
            source_type=source_type,
            source_key=source_key,
            confidence=confidence,
            records=selected,
        )
        if context is None:
            return
        seen.add(task_set)
        contexts.append(context)

    for candidate in merge_candidates:
        task_ids = candidate.get("task_ids") if isinstance(candidate, Mapping) else None
        if not isinstance(task_ids, list):
            continue
        add_context(
            str(candidate.get("group_type") or "merge_candidate"),
            str(candidate.get("candidate_key") or ""),
            str(candidate.get("confidence") or "low"),
            [str(task_id) for task_id in task_ids],
        )

    for cluster in clusters:
        task_ids = cluster.get("task_ids") if isinstance(cluster, Mapping) else None
        if not isinstance(task_ids, list):
            continue
        add_context(
            "vector_cluster",
            str(cluster.get("cluster_key") or ""),
            "low",
            [str(task_id) for task_id in task_ids],
        )

    confidence_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        contexts,
        key=lambda context: (
            0 if context.get("merge_ready") else 1,
            confidence_order.get(str(context.get("confidence") or ""), 9),
            -len(context.get("active_task_ids") or []),
            int(context.get("compact_context_tokens") or 0),
            str(context.get("context_key") or ""),
        ),
    )


def _compact_record_summary(record: TodoIndexRecord) -> str:
    details: list[str] = []
    if record.work_item_count:
        details.append(f"w{record.work_item_count}")
    if record.missing_evidence:
        details.append(f"m={','.join(record.missing_evidence[:3])}")
    if record.outputs:
        details.append(f"o={','.join(record.outputs[:2])}")
    return f"{record.task_id}[{';'.join(details)}]" if details else record.task_id


def _compact_execution_packet_text(packet: Mapping[str, Any]) -> str:
    parts = [
        str(packet.get("packet_key") or ""),
        f"primary={packet.get('primary_task_id') or ''}",
        f"ids={','.join(packet.get('active_task_ids') or [])}",
        f"mf={','.join(packet.get('merge_families') or [])}",
        f"gp={','.join(packet.get('goal_packet_keys') or [])}",
        f"w={packet.get('work_item_count_total') or 0}",
        f"pw={packet.get('goal_packet_work_item_count_max') or 0}",
        f"miss={','.join((packet.get('missing_evidence') or [])[:10])}",
        f"out={','.join((packet.get('shared_outputs') or packet.get('all_outputs') or [])[:5])}",
        f"ast={','.join((packet.get('ast_symbols') or [])[:12])}",
        f"todo={'|'.join(packet.get('task_summaries') or [])}",
    ]
    return ";".join(part for part in parts if not part.endswith("=") and part.strip())


def execution_packet_record_rank(record: TodoIndexRecord) -> tuple[int, int, int, int, str]:
    """Prefer larger aggregate packet tasks as the prompt entry point."""

    candidate_kind = record.candidate_kind.strip().lower()
    packet_role = record.goal_packet_role.strip().lower()
    merge_role = record.merge_role.strip().lower()
    if candidate_kind == "goal_packet_aggregate" or packet_role == "packet_aggregate" or merge_role == "packet_aggregate":
        role_rank = 0
    elif packet_role == "packet_anchor":
        role_rank = 1
    elif candidate_kind == "aggregate":
        role_rank = 2
    elif candidate_kind == "evidence_cluster":
        role_rank = 3
    elif packet_role == "packet_member":
        role_rank = 4
    else:
        role_rank = 5
    return (
        role_rank,
        -(record.work_item_count or 0),
        -(record.goal_packet_work_item_count or 0),
        record.token_count,
        record.task_id,
    )


def ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def build_execution_packet(
    *,
    context: Mapping[str, Any],
    records: Sequence[TodoIndexRecord],
    max_tasks: int = DEFAULT_EXECUTION_PACKET_MAX_TASKS,
) -> dict[str, Any] | None:
    """Build a compact multi-todo work packet from one bundle context."""

    if not records:
        return None
    active_records = [record for record in records if active_record(record)]
    if len(active_records) < 2:
        return None
    selected_records = sorted(active_records, key=execution_packet_record_rank)[: max(2, max_tasks)]
    task_ids = sorted_unique([record.task_id for record in selected_records])
    active_task_ids = ordered_unique([record.task_id for record in selected_records if active_record(record)])
    if len(active_task_ids) < 2:
        return None
    all_outputs = sorted_unique([output for record in selected_records for output in record.outputs])
    output_sets = [set(record.outputs) for record in selected_records if record.outputs]
    shared_outputs = sorted(output_sets[0].intersection(*output_sets[1:])) if output_sets else []
    work_counts = [record.work_item_count for record in selected_records if record.work_item_count > 0]
    packet_work_counts = [
        record.goal_packet_work_item_count for record in selected_records if record.goal_packet_work_item_count > 0
    ]
    packet_seed = json.dumps(
        {
            "context_key": context.get("context_key"),
            "active_task_ids": active_task_ids,
            "merge_families": sorted_unique([record.merge_family for record in selected_records]),
        },
        sort_keys=True,
    )
    packet: dict[str, Any] = {
        "packet_key": f"execution_packet/{sha1(packet_seed.encode('utf-8')).hexdigest()[:12]}",
        "source_context_key": str(context.get("context_key") or ""),
        "source_type": str(context.get("source_type") or ""),
        "source_key": str(context.get("source_key") or ""),
        "confidence": str(context.get("confidence") or "low"),
        "merge_ready": bool(context.get("merge_ready")),
        "task_ids": task_ids,
        "active_task_ids": active_task_ids,
        "primary_task_id": selected_records[0].task_id,
        "goal_ids": sorted_unique([record.goal_id for record in selected_records]),
        "graph_parent_ids": sorted_unique([parent for record in selected_records for parent in record.graph_parents]),
        "bundle_keys": sorted_unique([record.bundle_key for record in selected_records]),
        "merge_keys": sorted_unique([record.merge_key for record in selected_records]),
        "merge_families": sorted_unique([record.merge_family for record in selected_records]),
        "merge_roles": sorted_unique([record.merge_role for record in selected_records]),
        "goal_packet_keys": sorted_unique([record.goal_packet_key for record in selected_records]),
        "goal_packet_roles": sorted_unique([record.goal_packet_role for record in selected_records]),
        "goal_packet_goal_ids": sorted_unique(
            [goal_id for record in selected_records for goal_id in record.goal_packet_goal_ids]
        ),
        "goal_packet_task_count_max": max([record.goal_packet_task_count for record in selected_records], default=0),
        "goal_packet_work_item_count_max": max(packet_work_counts) if packet_work_counts else 0,
        "surplus_groups": sorted_unique([record.surplus_group for record in selected_records]),
        "candidate_kinds": sorted_unique([record.candidate_kind for record in selected_records]),
        "work_scopes": sorted_unique([record.work_scope for record in selected_records]),
        "work_item_count_min": min(work_counts) if work_counts else 0,
        "work_item_count_max": max(work_counts) if work_counts else 0,
        "work_item_count_total": sum(work_counts),
        "shared_outputs": shared_outputs,
        "all_outputs": all_outputs,
        "validation": sorted_unique([command for record in selected_records for command in record.validation])[:8],
        "missing_evidence": sorted_unique([item for record in selected_records for item in record.missing_evidence]),
        "ast_symbols": sorted_unique([symbol for record in selected_records for symbol in record.ast_symbols])[:80],
        "task_summaries": [_compact_record_summary(record) for record in selected_records],
        "raw_prompt_tokens": sum(record.token_count for record in selected_records),
    }
    compact_packet = _compact_execution_packet_text(packet)
    packet["compact_packet"] = compact_packet
    packet["compact_packet_tokens"] = len(objective_tokens(compact_packet))
    packet["estimated_token_savings"] = max(0, int(packet["raw_prompt_tokens"]) - int(packet["compact_packet_tokens"]))
    return packet


def build_execution_packets(
    records: Sequence[TodoIndexRecord],
    bundle_contexts: Sequence[Mapping[str, Any]],
    *,
    max_tasks: int = DEFAULT_EXECUTION_PACKET_MAX_TASKS,
) -> list[dict[str, Any]]:
    """Return compact execution packets for related goal/subgoal todo groups."""

    records_by_task = {record.task_id: record for record in records}
    packets: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for context in bundle_contexts:
        task_ids = context.get("active_task_ids") or context.get("task_ids")
        if not isinstance(task_ids, list):
            continue
        selected = [records_by_task[task_id] for task_id in map(str, task_ids) if task_id in records_by_task]
        task_set = tuple(sorted(record.task_id for record in selected if active_record(record)))
        if len(task_set) < 2 or task_set in seen:
            continue
        packet = build_execution_packet(context=context, records=selected, max_tasks=max_tasks)
        if packet is None:
            continue
        seen.add(tuple(packet["active_task_ids"]))
        packets.append(packet)

    confidence_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        packets,
        key=lambda packet: (
            0 if packet.get("merge_ready") else 1,
            confidence_order.get(str(packet.get("confidence") or ""), 9),
            -int(packet.get("work_item_count_total") or 0),
            -len(packet.get("active_task_ids") or []),
            int(packet.get("compact_packet_tokens") or 0),
            str(packet.get("packet_key") or ""),
        ),
    )


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
    merge_candidates = build_merge_candidates(records, clusters)
    bundle_contexts = build_bundle_contexts(records, clusters, merge_candidates)
    execution_packets = build_execution_packets(records, bundle_contexts)
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
        "estimated_raw_prompt_tokens": sum(record.token_count for record in records if active_record(record)),
        "estimated_compact_context_tokens": sum(
            int(context.get("compact_context_tokens") or 0) for context in bundle_contexts
        ),
        "estimated_execution_packet_tokens": sum(
            int(packet.get("compact_packet_tokens") or 0) for packet in execution_packets
        ),
        "records": [record.to_dict() for record in records],
        "clusters": clusters,
        "merge_candidates": merge_candidates,
        "bundle_contexts": bundle_contexts,
        "execution_packets": execution_packets,
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
        update_bundle_index_with_todo_vectors(
            bundle_index_path=bundle_index_path,
            records=records,
            clusters=clusters,
            merge_candidates=merge_candidates,
            bundle_contexts=bundle_contexts,
            execution_packets=execution_packets,
        )
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
    merge_candidates: Sequence[Mapping[str, Any]] = (),
    bundle_contexts: Sequence[Mapping[str, Any]] = (),
    execution_packets: Sequence[Mapping[str, Any]] = (),
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
    context_keys_by_task: dict[str, list[str]] = {}
    merge_ready_by_task: dict[str, list[str]] = {}
    for context in bundle_contexts:
        if not isinstance(context, Mapping):
            continue
        context_key = str(context.get("context_key") or "")
        if not context_key:
            continue
        task_ids = context.get("task_ids")
        if isinstance(task_ids, list):
            for task_id in task_ids:
                normalized = str(task_id)
                if normalized:
                    context_keys_by_task.setdefault(normalized, []).append(context_key)
        merge_ready_task_ids = context.get("merge_ready_task_ids")
        if isinstance(merge_ready_task_ids, list):
            for task_id in merge_ready_task_ids:
                normalized = str(task_id)
                if normalized:
                    merge_ready_by_task.setdefault(normalized, []).append(context_key)
    packet_keys_by_task: dict[str, list[str]] = {}
    for packet in execution_packets:
        if not isinstance(packet, Mapping):
            continue
        packet_key = str(packet.get("packet_key") or "")
        if not packet_key:
            continue
        task_ids = packet.get("active_task_ids") or packet.get("task_ids")
        if isinstance(task_ids, list):
            for task_id in task_ids:
                normalized = str(task_id)
                if normalized:
                    packet_keys_by_task.setdefault(normalized, []).append(packet_key)
    for bundle_key, bundle_payload in bundles.items():
        if not isinstance(bundle_payload, dict):
            continue
        bundle_records = by_bundle.get(str(bundle_key), [])
        bundle_payload["todo_vector_summary"] = {
            "task_count": len(bundle_records),
            "merge_keys": sorted({record.merge_key for record in bundle_records if record.merge_key}),
            "merge_families": sorted({record.merge_family for record in bundle_records if record.merge_family}),
            "goal_packet_keys": sorted({record.goal_packet_key for record in bundle_records if record.goal_packet_key}),
            "goal_packet_goal_ids": sorted(
                {goal_id for record in bundle_records for goal_id in record.goal_packet_goal_ids}
            ),
            "goal_packet_work_item_count_max": max(
                [record.goal_packet_work_item_count for record in bundle_records if record.goal_packet_work_item_count],
                default=0,
            ),
            "surplus_groups": sorted({record.surplus_group for record in bundle_records if record.surplus_group}),
            "estimated_prompt_tokens": sum(record.token_count for record in bundle_records),
            "compact_context_tokens": sum(
                int(context.get("compact_context_tokens") or 0)
                for context in bundle_contexts
                if set(context.get("task_ids") or []) & {record.task_id for record in bundle_records}
            ),
            "execution_packet_tokens": sum(
                int(packet.get("compact_packet_tokens") or 0)
                for packet in execution_packets
                if set(packet.get("active_task_ids") or packet.get("task_ids") or [])
                & {record.task_id for record in bundle_records}
            ),
            "merge_candidate_keys": [
                str(candidate.get("candidate_key") or "")
                for candidate in merge_candidates
                if set(candidate.get("task_ids") or []) & {record.task_id for record in bundle_records}
            ],
            "bundle_context_keys": sorted(
                {
                    context_key
                    for record in bundle_records
                    for context_key in context_keys_by_task.get(record.task_id, [])
                }
            ),
            "execution_packet_keys": sorted(
                {
                    packet_key
                    for record in bundle_records
                    for packet_key in packet_keys_by_task.get(record.task_id, [])
                }
            ),
            "merge_ready_task_ids": sorted(
                {
                    record.task_id
                    for record in bundle_records
                    if merge_ready_by_task.get(record.task_id)
                }
            ),
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
            task["merge_family"] = record.merge_family
            task["merge_role"] = record.merge_role
            task["work_item_count"] = record.work_item_count
            task["work_scope"] = record.work_scope
            task["goal_packet_key"] = record.goal_packet_key
            task["goal_packet_role"] = record.goal_packet_role
            task["goal_packet_goal_ids"] = record.goal_packet_goal_ids
            task["goal_packet_task_count"] = record.goal_packet_task_count
            task["goal_packet_work_item_count"] = record.goal_packet_work_item_count
            task["surplus_group"] = record.surplus_group
            task["todo_vector_key"] = record.vector_key
            task["todo_cluster_key"] = cluster_by_task.get(record.task_id, "")
            task["todo_bundle_context_keys"] = context_keys_by_task.get(record.task_id, [])[:5]
            task["todo_execution_packet_keys"] = packet_keys_by_task.get(record.task_id, [])[:5]
            task["related_task_ids"] = record.related_task_ids
    bundle_index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
