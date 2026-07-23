"""Dual JSON/DuckDB storage and bounded queries for supervisor artifacts.

JSON remains the portable interchange format.  Each write also materializes a
normalized DuckDB sidecar so schedulers and operators can inspect a small,
typed projection without loading a complete planning graph into a prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

BUNDLE_INDEX_KIND = "bundle_planning_index"
SCHEDULER_MANIFEST_KIND = "scheduler_manifest"
CODE_EVIDENCE_GRAPH_KIND = "code_evidence_graph"
EVIDENCE_GRAPH_KIND = CODE_EVIDENCE_GRAPH_KIND
QUERY_SCHEMA = "ipfs_accelerate_py.agent_supervisor.queryable_artifact@2"
MAX_QUERY_ROWS = 1_000
MAX_GRAPH_QUERY_HOPS = 8

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_READ_ONLY_SQL = re.compile(r"^(?:select|with|describe|show)\b", re.IGNORECASE)


@dataclass(frozen=True)
class QueryArtifactPaths:
    """The portable and queryable representations of one artifact."""

    json_path: Path
    duckdb_path: Path


def query_artifact_paths(path: Path | str) -> QueryArtifactPaths:
    """Resolve either a JSON or DuckDB path to both artifact representations."""

    resolved = Path(path).resolve()
    suffix = resolved.suffix.lower()
    if suffix == ".duckdb":
        return QueryArtifactPaths(
            json_path=resolved.with_suffix(".json"), duckdb_path=resolved
        )
    if suffix == ".json":
        return QueryArtifactPaths(
            json_path=resolved, duckdb_path=resolved.with_suffix(".duckdb")
        )
    raise ValueError(f"queryable artifacts require a .json or .duckdb path: {resolved}")


def _duckdb_module() -> Any:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - declared runtime dependency
        raise RuntimeError(
            "DuckDB is required for queryable supervisor artifacts"
        ) from exc
    return duckdb


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_value(value: str) -> Any:
    return json.loads(value)


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return bool(value)


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_kind(payload: Mapping[str, Any]) -> str:
    schema = str(payload.get("schema") or "")
    if schema == "ipfs_accelerate_py.agent_supervisor.code-evidence-graph@1" or (
        isinstance(payload.get("nodes"), list)
        and isinstance(payload.get("edges"), list)
        and str(payload.get("graph_id") or "").startswith("graph-")
    ):
        return CODE_EVIDENCE_GRAPH_KIND
    if isinstance(payload.get("bundles"), Mapping):
        return BUNDLE_INDEX_KIND
    if any(key in payload for key in ("lanes", "tasks", "scheduler_state", "counts")):
        return SCHEDULER_MANIFEST_KIND
    raise ValueError("could not infer supervisor artifact kind")


def _query_descriptor(kind: str, paths: QueryArtifactPaths) -> dict[str, Any]:
    return {
        "schema": QUERY_SCHEMA,
        "artifact_kind": kind,
        "duckdb_path": paths.duckdb_path.name,
        "catalog_table": "artifact_catalog",
    }


def _common_schema(connection: Any) -> None:
    connection.execute("""
        CREATE TABLE artifact_catalog (
            artifact_kind VARCHAR NOT NULL,
            schema_version VARCHAR NOT NULL,
            source_path VARCHAR NOT NULL,
            generated_at VARCHAR,
            source_sha256 VARCHAR NOT NULL,
            database_payload_sha256 VARCHAR NOT NULL,
            source_size BIGINT NOT NULL,
            source_mtime_ns BIGINT NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE artifact_fields (
            field_name VARCHAR PRIMARY KEY,
            value_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE artifact_tables (
            table_name VARCHAR PRIMARY KEY,
            description VARCHAR NOT NULL
        )
        """)


def _bundle_schema(connection: Any) -> None:
    connection.execute("""
        CREATE TABLE bundles (
            bundle_key VARCHAR PRIMARY KEY,
            shard_path VARCHAR,
            parallel_lane VARCHAR,
            bundle_strategy VARCHAR,
            conflict_policy VARCHAR,
            task_count BIGINT NOT NULL,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE bundle_tasks (
            bundle_key VARCHAR NOT NULL,
            task_ordinal BIGINT NOT NULL,
            task_id VARCHAR,
            canonical_task_cid VARCHAR,
            goal_id VARCHAR,
            parent_goal_id VARCHAR,
            subgoal_id VARCHAR,
            status VARCHAR,
            priority VARCHAR,
            title VARCHAR,
            payload_json VARCHAR NOT NULL,
            PRIMARY KEY (bundle_key, task_ordinal)
        )
        """)
    connection.execute("""
        CREATE TABLE bundle_task_dependencies (
            bundle_key VARCHAR NOT NULL,
            task_id VARCHAR,
            dependency_kind VARCHAR NOT NULL,
            dependency_id VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE dependency_edges (
            edge_ordinal BIGINT NOT NULL,
            source_task_cid VARCHAR,
            target_task_cid VARCHAR,
            edge_kind VARCHAR,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE conflict_edges (
            edge_ordinal BIGINT NOT NULL,
            left_task_cid VARCHAR,
            right_task_cid VARCHAR,
            reason VARCHAR,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE planning_decisions (
            decision_ordinal BIGINT NOT NULL,
            left_task_cid VARCHAR,
            right_task_cid VARCHAR,
            decision VARCHAR,
            reason VARCHAR,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("CREATE INDEX bundle_tasks_task_id_idx ON bundle_tasks(task_id)")
    connection.execute(
        "CREATE INDEX bundle_tasks_cid_idx ON bundle_tasks(canonical_task_cid)"
    )
    connection.execute("""
        CREATE VIEW open_bundle_tasks AS
        SELECT * FROM bundle_tasks
        WHERE lower(coalesce(status, 'todo')) NOT IN
              ('complete', 'completed', 'done', 'succeeded', 'blocked')
        """)


def _manifest_schema(connection: Any) -> None:
    connection.execute("""
        CREATE TABLE manifest_tasks (
            task_ordinal BIGINT NOT NULL,
            task_cid VARCHAR,
            task_id VARCHAR,
            bundle_key VARCHAR,
            state VARCHAR,
            lease_state VARCHAR,
            attempt BIGINT,
            claimant_did VARCHAR,
            updated_at_ms BIGINT,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE manifest_lanes (
            lane_ordinal BIGINT NOT NULL,
            bundle_key VARCHAR,
            parallel_lane VARCHAR,
            task_cid VARCHAR,
            state VARCHAR,
            pid BIGINT,
            claimable BOOLEAN,
            conflict_color BIGINT,
            schedule_rank BIGINT,
            log_path VARCHAR,
            task_ids_json VARCHAR NOT NULL,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE manifest_decisions (
            decision_ordinal BIGINT NOT NULL,
            task_cid VARCHAR,
            bundle_key VARCHAR,
            decision VARCHAR,
            reason VARCHAR,
            snapshot_id VARCHAR,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE manifest_conflict_edges (
            edge_ordinal BIGINT NOT NULL,
            left_task_id VARCHAR,
            right_task_id VARCHAR,
            blocks_concurrency BOOLEAN,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE manifest_conflict_decisions (
            decision_ordinal BIGINT NOT NULL,
            left_task_cid VARCHAR,
            right_task_cid VARCHAR,
            action VARCHAR,
            weight DOUBLE,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE scheduler_task_states (
            state_ordinal BIGINT NOT NULL,
            task_cid VARCHAR,
            task_id VARCHAR,
            goal_cid VARCHAR,
            subgoal_cid VARCHAR,
            lane_id VARCHAR,
            provider_id VARCHAR,
            phase VARCHAR,
            status VARCHAR,
            last_event_at VARCHAR,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE scheduler_metrics (
            metric_ordinal BIGINT NOT NULL,
            task_cid VARCHAR,
            goal_cid VARCHAR,
            subgoal_cid VARCHAR,
            lane_id VARCHAR,
            provider_id VARCHAR,
            queue_wait_seconds DOUBLE,
            implementation_duration_seconds DOUBLE,
            validation_duration_seconds DOUBLE,
            merge_wait_seconds DOUBLE,
            retries BIGINT,
            conflicts BIGINT,
            completions BIGINT,
            total_tokens BIGINT,
            total_cost_usd DOUBLE,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE scheduler_phases (
            phase VARCHAR PRIMARY KEY,
            task_count BIGINT NOT NULL
        )
        """)
    connection.execute(
        "CREATE INDEX manifest_tasks_cid_idx ON manifest_tasks(task_cid)"
    )
    connection.execute("CREATE INDEX manifest_tasks_state_idx ON manifest_tasks(state)")
    connection.execute(
        "CREATE INDEX manifest_lanes_key_idx ON manifest_lanes(bundle_key)"
    )
    connection.execute(
        "CREATE INDEX scheduler_task_states_cid_idx ON scheduler_task_states(task_cid)"
    )
    connection.execute(
        "CREATE VIEW ready_tasks AS SELECT * FROM manifest_tasks WHERE state = 'ready'"
    )
    connection.execute("""
        CREATE VIEW blocked_tasks AS
        SELECT * FROM manifest_tasks AS task
        WHERE task.state = 'blocked'
           OR (task.state = 'accepted' AND NOT EXISTS (
               SELECT 1 FROM manifest_lanes AS lane WHERE lane.task_cid = task.task_cid
           ))
        """)
    connection.execute(
        "CREATE VIEW completed_tasks AS SELECT * FROM manifest_tasks WHERE state = 'completed'"
    )
    connection.execute(
        "CREATE VIEW active_lanes AS SELECT * FROM manifest_lanes WHERE state = 'running'"
    )


def _code_evidence_graph_schema(connection: Any) -> None:
    connection.execute("""
        CREATE TABLE evidence_nodes (
            node_id VARCHAR PRIMARY KEY,
            node_kind VARCHAR NOT NULL,
            record_key VARCHAR NOT NULL,
            provenance VARCHAR NOT NULL,
            authoritative BOOLEAN NOT NULL,
            task_id VARCHAR,
            tree_id VARCHAR,
            symbol VARCHAR,
            obligation_id VARCHAR,
            assurance VARCHAR,
            freshness VARCHAR,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE evidence_edges (
            edge_id VARCHAR PRIMARY KEY,
            source_node_id VARCHAR NOT NULL,
            target_node_id VARCHAR NOT NULL,
            edge_kind VARCHAR NOT NULL,
            provenance VARCHAR NOT NULL,
            provenance_record_id VARCHAR NOT NULL,
            authoritative BOOLEAN NOT NULL,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE graph_records (
            record_type VARCHAR NOT NULL,
            record_id VARCHAR NOT NULL,
            record_ordinal BIGINT NOT NULL,
            payload_json VARCHAR NOT NULL,
            PRIMARY KEY (record_type, record_id)
        )
        """)
    for statement in (
        "CREATE INDEX evidence_nodes_kind_idx ON evidence_nodes(node_kind)",
        "CREATE INDEX evidence_nodes_task_idx ON evidence_nodes(task_id)",
        "CREATE INDEX evidence_nodes_tree_idx ON evidence_nodes(tree_id)",
        "CREATE INDEX evidence_nodes_symbol_idx ON evidence_nodes(symbol)",
        "CREATE INDEX evidence_nodes_obligation_idx ON evidence_nodes(obligation_id)",
        "CREATE INDEX evidence_nodes_assurance_idx ON evidence_nodes(assurance)",
        "CREATE INDEX evidence_nodes_freshness_idx ON evidence_nodes(freshness)",
        "CREATE INDEX evidence_edges_source_idx ON evidence_edges(source_node_id)",
        "CREATE INDEX evidence_edges_target_idx ON evidence_edges(target_node_id)",
        "CREATE INDEX evidence_edges_kind_idx ON evidence_edges(edge_kind)",
    ):
        connection.execute(statement)

    # Stable, narrow query surfaces.  Both the concise ``*_index`` spellings
    # and explicit ``code_evidence_*`` aliases are kept for callers composing
    # SQL across multiple supervisor artifact kinds.
    connection.execute(
        "CREATE VIEW task_index AS SELECT * FROM evidence_nodes "
        "WHERE node_kind = 'task'"
    )
    connection.execute(
        "CREATE VIEW tree_index AS SELECT * FROM evidence_nodes "
        "WHERE node_kind = 'tree'"
    )
    connection.execute(
        "CREATE VIEW symbol_index AS SELECT * FROM evidence_nodes "
        "WHERE node_kind = 'symbol'"
    )
    connection.execute(
        "CREATE VIEW obligation_index AS SELECT * FROM evidence_nodes "
        "WHERE node_kind = 'obligation'"
    )
    connection.execute(
        "CREATE VIEW assurance_index AS SELECT node_id, node_kind, task_id, "
        "obligation_id, assurance, authoritative, payload_json FROM evidence_nodes "
        "WHERE assurance <> ''"
    )
    connection.execute(
        "CREATE VIEW freshness_index AS SELECT node_id, node_kind, task_id, "
        "obligation_id, freshness, authoritative, payload_json FROM evidence_nodes "
        "WHERE freshness <> ''"
    )
    connection.execute(
        "CREATE VIEW dependency_index AS SELECT * FROM evidence_edges "
        "WHERE edge_kind = 'depends_on'"
    )
    connection.execute(
        "CREATE VIEW authoritative_evidence_edges AS SELECT * FROM evidence_edges "
        "WHERE authoritative"
    )
    for alias, source in (
        ("graph_nodes", "evidence_nodes"),
        ("graph_edges", "evidence_edges"),
        ("tasks", "task_index"),
        ("trees", "tree_index"),
        ("symbols", "symbol_index"),
        ("obligations", "obligation_index"),
        ("assurances", "assurance_index"),
        ("freshness", "freshness_index"),
        ("dependencies", "dependency_index"),
        ("graph_tasks", "task_index"),
        ("graph_trees", "tree_index"),
        ("graph_symbols", "symbol_index"),
        ("graph_obligations", "obligation_index"),
        ("graph_assurance", "assurance_index"),
        ("graph_freshness", "freshness_index"),
        ("graph_dependencies", "dependency_index"),
        ("code_evidence_tasks", "task_index"),
        ("code_evidence_trees", "tree_index"),
        ("code_evidence_symbols", "symbol_index"),
        ("code_evidence_obligations", "obligation_index"),
        ("code_evidence_assurance", "assurance_index"),
        ("code_evidence_freshness", "freshness_index"),
        ("code_evidence_dependencies", "dependency_index"),
    ):
        connection.execute(f"CREATE VIEW {alias} AS SELECT * FROM {source}")


def _top_level_fields(
    payload: Mapping[str, Any], kind: str
) -> Iterable[tuple[str, str]]:
    excluded = (
        {"bundles"}
        if kind == BUNDLE_INDEX_KIND
        else {"nodes", "edges"}
        if kind == CODE_EVIDENCE_GRAPH_KIND
        else {
            "blocked",
            "completed",
            "lanes",
            "ready",
            "scheduler_decisions",
            "tasks",
        }
    )
    for key, value in payload.items():
        if key not in excluded:
            yield str(key), _json_text(value)


def _graph_mapping(payload: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _mapping_items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _populate_bundle_tables(connection: Any, payload: Mapping[str, Any]) -> None:
    bundles = (
        payload.get("bundles") if isinstance(payload.get("bundles"), Mapping) else {}
    )
    bundle_rows: list[tuple[Any, ...]] = []
    task_rows: list[tuple[Any, ...]] = []
    dependency_rows: list[tuple[Any, ...]] = []
    for bundle_key, raw_bundle in sorted(bundles.items()):
        if not isinstance(raw_bundle, Mapping):
            continue
        tasks = _mapping_items(raw_bundle.get("tasks"))
        bundle_payload = {
            key: value for key, value in raw_bundle.items() if key != "tasks"
        }
        bundle_rows.append(
            (
                str(bundle_key),
                str(raw_bundle.get("shard_path") or ""),
                str(raw_bundle.get("parallel_lane") or ""),
                str(raw_bundle.get("bundle_strategy") or ""),
                str(raw_bundle.get("conflict_policy") or ""),
                len(tasks),
                _json_text(bundle_payload),
            )
        )
        for ordinal, task in enumerate(tasks):
            task_id = str(task.get("task_id") or "")
            task_rows.append(
                (
                    str(bundle_key),
                    ordinal,
                    task_id,
                    str(task.get("canonical_task_cid") or task.get("task_cid") or ""),
                    str(task.get("goal_id") or ""),
                    str(task.get("parent_goal_id") or ""),
                    str(task.get("subgoal_id") or ""),
                    str(task.get("status") or ""),
                    str(task.get("priority") or ""),
                    str(task.get("title") or ""),
                    _json_text(task),
                )
            )
            for dependency_kind in (
                "depends_on",
                "dependency_task_cids",
                "blocking_task_cids",
            ):
                dependency_rows.extend(
                    (str(bundle_key), task_id, dependency_kind, dependency_id)
                    for dependency_id in _string_values(task.get(dependency_kind))
                )
    if bundle_rows:
        connection.executemany(
            "INSERT INTO bundles VALUES (?, ?, ?, ?, ?, ?, ?)", bundle_rows
        )
    if task_rows:
        connection.executemany(
            "INSERT INTO bundle_tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            task_rows,
        )
    if dependency_rows:
        connection.executemany(
            "INSERT INTO bundle_task_dependencies VALUES (?, ?, ?, ?)", dependency_rows
        )

    dependency_graph = _graph_mapping(
        payload, "task_dependency_graph", "dependency_dag"
    )
    dependency_edges = _mapping_items(dependency_graph.get("edges"))
    if dependency_edges:
        connection.executemany(
            "INSERT INTO dependency_edges VALUES (?, ?, ?, ?, ?)",
            [
                (
                    ordinal,
                    str(edge.get("source_task_cid") or edge.get("source") or ""),
                    str(edge.get("target_task_cid") or edge.get("target") or ""),
                    str(edge.get("kind") or edge.get("edge_kind") or ""),
                    _json_text(edge),
                )
                for ordinal, edge in enumerate(dependency_edges)
            ],
        )

    conflict_graph = _graph_mapping(payload, "task_conflict_graph", "conflict_graph")
    conflict_edges = _mapping_items(conflict_graph.get("edges"))
    if conflict_edges:
        connection.executemany(
            "INSERT INTO conflict_edges VALUES (?, ?, ?, ?, ?)",
            [
                (
                    ordinal,
                    str(edge.get("left_task_cid") or edge.get("left") or ""),
                    str(edge.get("right_task_cid") or edge.get("right") or ""),
                    str(edge.get("reason") or edge.get("conflict_reason") or ""),
                    _json_text(edge),
                )
                for ordinal, edge in enumerate(conflict_edges)
            ],
        )
    decisions = _mapping_items(
        payload.get("conflict_planning_decisions") or conflict_graph.get("decisions")
    )
    if decisions:
        connection.executemany(
            "INSERT INTO planning_decisions VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    ordinal,
                    str(item.get("left_task_cid") or item.get("left") or ""),
                    str(item.get("right_task_cid") or item.get("right") or ""),
                    str(item.get("decision") or ""),
                    str(item.get("reason") or ""),
                    _json_text(item),
                )
                for ordinal, item in enumerate(decisions)
            ],
        )


def _populate_manifest_tables(connection: Any, payload: Mapping[str, Any]) -> None:
    tasks = _mapping_items(payload.get("tasks"))
    if tasks:
        connection.executemany(
            "INSERT INTO manifest_tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    ordinal,
                    str(item.get("task_cid") or ""),
                    str(item.get("task_id") or ""),
                    str(item.get("bundle_key") or ""),
                    str(item.get("state") or ""),
                    str(item.get("lease_state") or ""),
                    _as_int(item.get("attempt")),
                    str(item.get("claimant_did") or ""),
                    _as_int(item.get("updated_at_ms")),
                    _json_text(item),
                )
                for ordinal, item in enumerate(tasks)
            ],
        )
    lanes = _mapping_items(payload.get("lanes"))
    if lanes:
        connection.executemany(
            "INSERT INTO manifest_lanes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    ordinal,
                    str(item.get("bundle_key") or ""),
                    str(item.get("parallel_lane") or ""),
                    str(item.get("task_cid") or ""),
                    str(item.get("state") or ""),
                    _as_int(item.get("pid")),
                    _as_bool(item.get("claimable")),
                    _as_int(item.get("conflict_color")),
                    _as_int(item.get("schedule_rank")),
                    str(item.get("log_path") or ""),
                    _json_text(_string_values(item.get("task_ids"))),
                    _json_text(item),
                )
                for ordinal, item in enumerate(lanes)
            ],
        )
    decisions = _mapping_items(payload.get("scheduler_decisions"))
    if decisions:
        connection.executemany(
            "INSERT INTO manifest_decisions VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    ordinal,
                    str(item.get("task_cid") or ""),
                    str(item.get("bundle_key") or ""),
                    str(item.get("decision") or ""),
                    str(item.get("reason") or ""),
                    str(item.get("snapshot_id") or ""),
                    _json_text(item),
                )
                for ordinal, item in enumerate(decisions)
            ],
        )
    conflict_graph = payload.get("conflict_graph")
    if not isinstance(conflict_graph, Mapping):
        conflict_graph = {}
    conflict_edges = _mapping_items(conflict_graph.get("edges"))
    if conflict_edges:
        connection.executemany(
            "INSERT INTO manifest_conflict_edges VALUES (?, ?, ?, ?, ?)",
            [
                (
                    ordinal,
                    str(item.get("left_task_id") or item.get("left_task_cid") or ""),
                    str(item.get("right_task_id") or item.get("right_task_cid") or ""),
                    _as_bool(item.get("blocks_concurrency")),
                    _json_text(item),
                )
                for ordinal, item in enumerate(conflict_edges)
            ],
        )
    conflict_decisions = _mapping_items(conflict_graph.get("decisions"))
    if conflict_decisions:
        connection.executemany(
            "INSERT INTO manifest_conflict_decisions VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    ordinal,
                    str(item.get("left_task_cid") or ""),
                    str(item.get("right_task_cid") or ""),
                    str(item.get("action") or item.get("decision") or ""),
                    _as_float(item.get("weight")),
                    _json_text(item),
                )
                for ordinal, item in enumerate(conflict_decisions)
            ],
        )

    scheduler_snapshot = payload.get("scheduler_snapshot")
    if not isinstance(scheduler_snapshot, Mapping):
        scheduler_snapshot = {}
    task_states = _mapping_items(scheduler_snapshot.get("task_states"))
    if task_states:
        connection.executemany(
            "INSERT INTO scheduler_task_states VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    ordinal,
                    str(item.get("task_cid") or item.get("canonical_task_cid") or ""),
                    str(item.get("task_id") or ""),
                    str(item.get("goal_cid") or item.get("canonical_goal_id") or ""),
                    str(
                        item.get("subgoal_cid")
                        or item.get("canonical_subgoal_id")
                        or ""
                    ),
                    str(item.get("lane_id") or item.get("canonical_lane_id") or ""),
                    str(
                        item.get("provider_id")
                        or item.get("canonical_provider_id")
                        or ""
                    ),
                    str(item.get("phase") or ""),
                    str(item.get("status") or ""),
                    str(item.get("last_event_at") or ""),
                    _json_text(item),
                )
                for ordinal, item in enumerate(task_states)
            ],
        )
    metrics = _mapping_items(scheduler_snapshot.get("metrics"))
    if metrics:
        connection.executemany(
            "INSERT INTO scheduler_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    ordinal,
                    str(item.get("task_cid") or item.get("canonical_task_cid") or ""),
                    str(item.get("goal_cid") or item.get("canonical_goal_id") or ""),
                    str(
                        item.get("subgoal_cid")
                        or item.get("canonical_subgoal_id")
                        or ""
                    ),
                    str(item.get("lane_id") or item.get("canonical_lane_id") or ""),
                    str(
                        item.get("provider_id")
                        or item.get("canonical_provider_id")
                        or ""
                    ),
                    _as_float(item.get("queue_wait_seconds")),
                    _as_float(item.get("implementation_duration_seconds")),
                    _as_float(item.get("validation_duration_seconds")),
                    _as_float(item.get("merge_wait_seconds")),
                    _as_int(item.get("retries")),
                    _as_int(item.get("conflicts")),
                    _as_int(item.get("completions")),
                    _as_int(item.get("total_tokens", item.get("tokens"))),
                    _as_float(item.get("total_cost_usd", item.get("cost_usd"))),
                    _json_text(item),
                )
                for ordinal, item in enumerate(metrics)
            ],
        )
    phases = scheduler_snapshot.get("phases")
    if isinstance(phases, Mapping):
        phase_rows = [
            (
                str(phase),
                _as_int(value.get("count")) or 0 if isinstance(value, Mapping) else 0,
            )
            for phase, value in sorted(phases.items())
        ]
        if phase_rows:
            connection.executemany(
                "INSERT INTO scheduler_phases VALUES (?, ?)", phase_rows
            )


def _populate_code_evidence_graph_tables(
    connection: Any, payload: Mapping[str, Any]
) -> None:
    # Decode through the graph contract before persistence.  This rejects
    # forged identities and any enrichment-originated authoritative edge.
    from .code_evidence_graph import CodeEvidenceGraph

    graph = CodeEvidenceGraph.from_dict(payload)
    node_rows: list[tuple[Any, ...]] = []
    graph_rows: list[tuple[Any, ...]] = []
    for ordinal, node in enumerate(graph.nodes):
        item = node.to_dict()
        text = _json_text(item)
        node_rows.append(
            (
                node.node_id,
                node.kind.value,
                node.record_key,
                node.provenance.value,
                node.authoritative,
                node.task_id,
                node.tree_id,
                node.symbol,
                node.obligation_id,
                node.assurance,
                node.freshness,
                text,
            )
        )
        graph_rows.append(("node", node.node_id, ordinal, text))
    edge_rows: list[tuple[Any, ...]] = []
    for ordinal, edge in enumerate(graph.edges):
        item = edge.to_dict()
        text = _json_text(item)
        edge_rows.append(
            (
                edge.edge_id,
                edge.source,
                edge.target,
                edge.kind.value,
                edge.provenance.value,
                edge.provenance_record_id,
                edge.authoritative,
                text,
            )
        )
        graph_rows.append(("edge", edge.edge_id, ordinal, text))
    if node_rows:
        connection.executemany(
            "INSERT INTO evidence_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            node_rows,
        )
    if edge_rows:
        connection.executemany(
            "INSERT INTO evidence_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            edge_rows,
        )
    if graph_rows:
        connection.executemany(
            "INSERT INTO graph_records VALUES (?, ?, ?, ?)", graph_rows
        )


def _table_descriptions(kind: str) -> dict[str, str]:
    common = {
        "artifact_catalog": "Artifact identity, source freshness, and schema version.",
        "artifact_fields": "Individually queryable top-level JSON fields.",
        "artifact_tables": "Descriptions of the normalized query tables.",
    }
    if kind == BUNDLE_INDEX_KIND:
        return {
            **common,
            "bundles": "One row per objective bundle without embedded member tasks.",
            "bundle_tasks": "One typed row per bundle member with full JSON in payload_json.",
            "bundle_task_dependencies": "Normalized declared and computed task dependencies.",
            "dependency_edges": "Normalized dependency graph edges.",
            "conflict_edges": "Normalized conflict graph edges.",
            "planning_decisions": "Normalized conflict-planning decisions.",
            "open_bundle_tasks": "View of bundle tasks that are not terminal or blocked.",
        }
    if kind == CODE_EVIDENCE_GRAPH_KIND:
        return {
            **common,
            "evidence_nodes": "Canonical graph nodes with indexed task, tree, symbol, obligation, assurance, and freshness fields.",
            "evidence_edges": "Canonical provenance edges and their derived authority.",
            "graph_records": "Lossless canonical node and edge records used for projection round trips.",
            "task_index": "Task nodes.",
            "tree_index": "Repository tree nodes.",
            "symbol_index": "Qualified AST symbol nodes.",
            "obligation_index": "Code proof obligation nodes.",
            "assurance_index": "Evidence records with an assurance projection.",
            "freshness_index": "Evidence records with a freshness projection.",
            "dependency_index": "Task and proof dependency edges.",
            "authoritative_evidence_edges": "Gate-relevant edges derived from trusted record boundaries.",
            "graph_nodes": "Compatibility alias for evidence_nodes.",
            "graph_edges": "Compatibility alias for evidence_edges.",
            "tasks": "Compatibility alias for task_index.",
            "trees": "Compatibility alias for tree_index.",
            "symbols": "Compatibility alias for symbol_index.",
            "obligations": "Compatibility alias for obligation_index.",
            "assurances": "Compatibility alias for assurance_index.",
            "freshness": "Compatibility alias for freshness_index.",
            "dependencies": "Compatibility alias for dependency_index.",
            "graph_tasks": "Compatibility alias for task_index.",
            "graph_trees": "Compatibility alias for tree_index.",
            "graph_symbols": "Compatibility alias for symbol_index.",
            "graph_obligations": "Compatibility alias for obligation_index.",
            "graph_assurance": "Compatibility alias for assurance_index.",
            "graph_freshness": "Compatibility alias for freshness_index.",
            "graph_dependencies": "Compatibility alias for dependency_index.",
            "code_evidence_tasks": "Compatibility alias for task_index.",
            "code_evidence_trees": "Compatibility alias for tree_index.",
            "code_evidence_symbols": "Compatibility alias for symbol_index.",
            "code_evidence_obligations": "Compatibility alias for obligation_index.",
            "code_evidence_assurance": "Compatibility alias for assurance_index.",
            "code_evidence_freshness": "Compatibility alias for freshness_index.",
            "code_evidence_dependencies": "Compatibility alias for dependency_index.",
        }
    return {
        **common,
        "manifest_tasks": "Current scheduler task projection, one task per row.",
        "manifest_lanes": "Current worker lane projection, one lane per row.",
        "manifest_decisions": "Scheduler admission and deferral decisions.",
        "manifest_conflict_edges": "Normalized scheduler bundle-conflict edges.",
        "manifest_conflict_decisions": "Normalized scheduler conflict-coloring decisions.",
        "scheduler_task_states": "Per-task lifecycle states from the authoritative scheduler snapshot.",
        "scheduler_metrics": "Per-task timing, retry, conflict, token, and cost metrics.",
        "scheduler_phases": "Task counts grouped by scheduler lifecycle phase.",
        "ready_tasks": "View of tasks currently ready for a lease.",
        "blocked_tasks": "View of tasks currently blocked.",
        "completed_tasks": "View of completed tasks.",
        "active_lanes": "View of currently running lanes.",
    }


def _write_duckdb(
    path: Path,
    payload: Mapping[str, Any],
    *,
    kind: str,
    source_path: Path,
    source_sha256: str,
    source_size: int,
    source_mtime_ns: int,
) -> None:
    duckdb = _duckdb_module()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.unlink(missing_ok=True)
    Path(f"{temporary}.wal").unlink(missing_ok=True)
    payload_text = _json_text(payload)
    try:
        connection = duckdb.connect(str(temporary))
        try:
            _common_schema(connection)
            if kind == BUNDLE_INDEX_KIND:
                _bundle_schema(connection)
                _populate_bundle_tables(connection, payload)
            elif kind == SCHEDULER_MANIFEST_KIND:
                _manifest_schema(connection)
                _populate_manifest_tables(connection, payload)
            elif kind == CODE_EVIDENCE_GRAPH_KIND:
                _code_evidence_graph_schema(connection)
                _populate_code_evidence_graph_tables(connection, payload)
            else:
                raise ValueError(f"unsupported query artifact kind: {kind}")
            connection.execute(
                "INSERT INTO artifact_catalog VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    kind,
                    QUERY_SCHEMA,
                    str(source_path),
                    str(payload.get("generated_at") or ""),
                    source_sha256,
                    hashlib.sha256(payload_text.encode("utf-8")).hexdigest(),
                    source_size,
                    source_mtime_ns,
                ),
            )
            fields = list(_top_level_fields(payload, kind))
            if fields:
                connection.executemany(
                    "INSERT INTO artifact_fields VALUES (?, ?)", fields
                )
            connection.executemany(
                "INSERT INTO artifact_tables VALUES (?, ?)",
                sorted(_table_descriptions(kind).items()),
            )
            connection.execute("CHECKPOINT")
        finally:
            connection.close()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        Path(f"{temporary}.wal").unlink(missing_ok=True)


def write_queryable_artifact(
    path: Path | str,
    payload: Mapping[str, Any],
    *,
    kind: str | None = None,
    database_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically write equivalent JSON and normalized DuckDB artifacts."""

    paths = query_artifact_paths(path)
    resolved_kind = kind or _artifact_kind(payload)
    rendered = dict(payload)
    rendered["query_store"] = _query_descriptor(resolved_kind, paths)
    database_rendered = dict(database_payload or rendered)
    database_rendered["query_store"] = dict(rendered["query_store"])
    source_text = json.dumps(rendered, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(paths.json_path, source_text)
    source_stat = paths.json_path.stat()
    _write_duckdb(
        paths.duckdb_path,
        database_rendered,
        kind=resolved_kind,
        source_path=paths.json_path,
        source_sha256=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        source_size=source_stat.st_size,
        source_mtime_ns=source_stat.st_mtime_ns,
    )
    return rendered


def write_bundle_index_artifact(
    path: Path | str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    return write_queryable_artifact(path, payload, kind=BUNDLE_INDEX_KIND)


def write_scheduler_manifest_artifact(
    path: Path | str,
    payload: Mapping[str, Any],
    *,
    database_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return write_queryable_artifact(
        path,
        payload,
        kind=SCHEDULER_MANIFEST_KIND,
        database_payload=database_payload,
    )


def write_code_evidence_graph_artifact(
    path: Path | str, payload: Any
) -> dict[str, Any]:
    """Write a validated code-evidence graph to paired JSON and DuckDB files."""

    from .code_evidence_graph import CodeEvidenceGraph

    graph = (
        payload
        if isinstance(payload, CodeEvidenceGraph)
        else CodeEvidenceGraph.from_dict(payload)
    )
    return write_queryable_artifact(
        path, graph.to_dict(), kind=CODE_EVIDENCE_GRAPH_KIND
    )


def _database_fresh(database_path: Path, source_path: Path, kind: str | None) -> bool:
    if not database_path.exists() or not source_path.exists():
        return False
    duckdb = _duckdb_module()
    try:
        connection = duckdb.connect(str(database_path), read_only=True)
        try:
            row = connection.execute(
                "SELECT artifact_kind, schema_version, source_size, source_mtime_ns "
                "FROM artifact_catalog LIMIT 1"
            ).fetchone()
        finally:
            connection.close()
    except Exception:
        return False
    stat = source_path.stat()
    return bool(
        row
        and (kind is None or str(row[0]) == kind)
        and str(row[1]) == QUERY_SCHEMA
        and int(row[2]) == stat.st_size
        and int(row[3]) == stat.st_mtime_ns
    )


def _read_stable_json(
    path: Path,
) -> tuple[Mapping[str, Any], os.stat_result, str]:
    """Read one atomic JSON generation even when another process is replacing it."""

    for _attempt in range(3):
        before = path.stat()
        text = path.read_text(encoding="utf-8")
        after = path.stat()
        if (
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            continue
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError(f"artifact JSON must contain an object: {path}")
        source_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return payload, after, source_sha256
    raise RuntimeError(f"artifact changed repeatedly while being read: {path}")


def ensure_query_database(path: Path | str, *, kind: str | None = None) -> Path:
    """Return a current DuckDB representation for a JSON or DuckDB artifact."""

    requested = Path(path).resolve()
    paths = query_artifact_paths(requested)
    if requested.suffix.lower() == ".duckdb":
        if not requested.exists():
            raise FileNotFoundError(requested)
        duckdb = _duckdb_module()
        connection = duckdb.connect(str(requested), read_only=True)
        try:
            row = connection.execute(
                "SELECT artifact_kind, schema_version FROM artifact_catalog LIMIT 1"
            ).fetchone()
        finally:
            connection.close()
        if (
            row
            and str(row[1]) == QUERY_SCHEMA
            and (kind is None or str(row[0]) == kind)
        ):
            return requested
        if paths.json_path.exists():
            return ensure_query_database(paths.json_path, kind=kind)
        actual_kind = str(row[0]) if row else "unknown"
        actual_schema = str(row[1]) if row else "unknown"
        raise ValueError(
            f"expected {kind or actual_kind} {QUERY_SCHEMA} DuckDB artifact, "
            f"found {actual_kind} {actual_schema}: {requested}"
        )
    if _database_fresh(paths.duckdb_path, paths.json_path, kind):
        return paths.duckdb_path
    payload, source_stat, source_sha256 = _read_stable_json(paths.json_path)
    resolved_kind = kind or _artifact_kind(payload)
    _write_duckdb(
        paths.duckdb_path,
        payload,
        kind=resolved_kind,
        source_path=paths.json_path,
        source_sha256=source_sha256,
        source_size=source_stat.st_size,
        source_mtime_ns=source_stat.st_mtime_ns,
    )
    return paths.duckdb_path


def read_artifact_fields(
    path: Path | str,
    field_names: Sequence[str],
    *,
    kind: str | None = None,
) -> dict[str, Any]:
    """Read selected top-level fields without decoding the full artifact."""

    if not field_names:
        return {}
    database_path = ensure_query_database(path, kind=kind)
    duckdb = _duckdb_module()
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        placeholders = ", ".join("?" for _ in field_names)
        rows = connection.execute(
            f"SELECT field_name, value_json FROM artifact_fields WHERE field_name IN ({placeholders})",
            list(field_names),
        ).fetchall()
    finally:
        connection.close()
    return {str(name): _json_value(str(value)) for name, value in rows}


def read_bundle_index_projection(
    path: Path | str,
    *,
    field_names: Sequence[str] = ("source_todo",),
) -> dict[str, Any]:
    """Read bundle rows plus only the requested top-level planning fields."""

    database_path = ensure_query_database(path, kind=BUNDLE_INDEX_KIND)
    duckdb = _duckdb_module()
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        bundle_rows = connection.execute(
            "SELECT bundle_key, payload_json FROM bundles ORDER BY bundle_key"
        ).fetchall()
        task_rows = connection.execute(
            "SELECT bundle_key, payload_json FROM bundle_tasks ORDER BY bundle_key, task_ordinal"
        ).fetchall()
        fields: dict[str, Any] = {}
        if field_names:
            placeholders = ", ".join("?" for _ in field_names)
            for name, value in connection.execute(
                f"SELECT field_name, value_json FROM artifact_fields WHERE field_name IN ({placeholders})",
                list(field_names),
            ).fetchall():
                fields[str(name)] = _json_value(str(value))
    finally:
        connection.close()
    bundles = {str(key): _json_value(str(value)) for key, value in bundle_rows}
    for bundle_key, value in task_rows:
        bundles.setdefault(str(bundle_key), {})
        bundles[str(bundle_key)].setdefault("tasks", []).append(_json_value(str(value)))
    return {**fields, "bundles": bundles}


def read_bundle_index_artifact(path: Path | str) -> dict[str, Any]:
    """Reconstruct a complete bundle index from either representation."""

    database_path = ensure_query_database(path, kind=BUNDLE_INDEX_KIND)
    duckdb = _duckdb_module()
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        field_names = [
            str(row[0])
            for row in connection.execute(
                "SELECT field_name FROM artifact_fields ORDER BY field_name"
            ).fetchall()
        ]
    finally:
        connection.close()
    return read_bundle_index_projection(path, field_names=field_names)


def read_code_evidence_graph_projection(path: Path | str) -> dict[str, Any]:
    """Reconstruct canonical graph records from either artifact representation."""

    database_path = ensure_query_database(path, kind=CODE_EVIDENCE_GRAPH_KIND)
    duckdb = _duckdb_module()
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        rows = connection.execute(
            "SELECT record_type, payload_json FROM graph_records "
            "ORDER BY record_type DESC, record_ordinal, record_id"
        ).fetchall()
        fields = {
            str(name): _json_value(str(value))
            for name, value in connection.execute(
                "SELECT field_name, value_json FROM artifact_fields"
            ).fetchall()
        }
    finally:
        connection.close()
    nodes: list[Any] = []
    edges: list[Any] = []
    for record_type, value in rows:
        (nodes if str(record_type) == "node" else edges).append(_json_value(str(value)))
    from .code_evidence_graph import CodeEvidenceGraph

    graph = CodeEvidenceGraph.from_dict({**fields, "nodes": nodes, "edges": edges})
    return graph.to_dict()


def read_code_evidence_graph_artifact(path: Path | str) -> dict[str, Any]:
    """Compatibility spelling for the lossless graph projection reader."""

    return read_code_evidence_graph_projection(path)


def read_code_evidence_graph(path: Path | str) -> Any:
    """Return a typed graph reconstructed from JSON or DuckDB."""

    from .code_evidence_graph import CodeEvidenceGraph

    return CodeEvidenceGraph.from_dict(read_code_evidence_graph_projection(path))


def canonical_code_evidence_graph_records(
    path: Path | str,
) -> dict[str, list[dict[str, Any]]]:
    """Read only the canonical records used to compare graph projections."""

    return read_code_evidence_graph(path).canonical_records()


# Concise compatibility spellings for callers whose artifact type is already
# clear from context.
write_evidence_graph_artifact = write_code_evidence_graph_artifact
read_evidence_graph_artifact = read_code_evidence_graph_artifact
read_evidence_graph_projection = read_code_evidence_graph_projection
canonical_evidence_graph_records = canonical_code_evidence_graph_records


def _jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    return value


def _validated_identifier(value: str, *, label: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def query_artifact(
    path: Path | str,
    *,
    table: str | None = None,
    columns: Sequence[str] = ("*",),
    where: str = "",
    sql: str = "",
    limit: int = 50,
    kind: str | None = None,
) -> dict[str, Any]:
    """Execute one read-only, row-bounded query against either artifact format."""

    row_limit = max(1, min(int(limit), MAX_QUERY_ROWS))
    database_path = ensure_query_database(path, kind=kind)
    if sql:
        statement = sql.strip().rstrip(";").strip()
        if ";" in statement or not _READ_ONLY_SQL.match(statement):
            raise ValueError(
                "only one read-only SELECT/WITH/DESCRIBE/SHOW query is allowed"
            )
        if re.match(r"^(?:select|with)\b", statement, re.IGNORECASE):
            statement = f"SELECT * FROM ({statement}) AS bounded_artifact_query LIMIT {row_limit + 1}"
    else:
        selected_table = _validated_identifier(
            table or "artifact_catalog", label="table name"
        )
        if columns == ("*",) or list(columns) == ["*"]:
            selected_columns = "*"
        else:
            selected_columns = ", ".join(
                _validated_identifier(column, label="column name") for column in columns
            )
        if ";" in where:
            raise ValueError("where clauses may not contain statement separators")
        statement = f"SELECT {selected_columns} FROM {selected_table}"
        if where.strip():
            statement += f" WHERE {where.strip()}"
        statement += f" LIMIT {row_limit + 1}"

    duckdb = _duckdb_module()
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        cursor = connection.execute(statement)
        names = [str(item[0]) for item in cursor.description or ()]
        values = cursor.fetchmany(row_limit + 1)
    finally:
        connection.close()
    truncated = len(values) > row_limit
    rows = [
        {name: _jsonable(value) for name, value in zip(names, row)}
        for row in values[:row_limit]
    ]
    return {
        "schema": QUERY_SCHEMA,
        "duckdb_path": str(database_path),
        "columns": names,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
        "limit": row_limit,
    }


def query_code_evidence_graph(
    path: Path | str,
    **query: Any,
) -> dict[str, Any]:
    """Execute one bounded query against a code-evidence graph artifact."""

    supplied_kind = query.pop("kind", CODE_EVIDENCE_GRAPH_KIND)
    if supplied_kind != CODE_EVIDENCE_GRAPH_KIND:
        raise ValueError("code evidence graph queries require code_evidence_graph kind")
    return query_artifact(path, kind=CODE_EVIDENCE_GRAPH_KIND, **query)


def _exact_strings(values: Sequence[str] | str, *, label: str) -> tuple[str, ...]:
    """Normalize exact-match selectors without accepting wildcard spellings."""

    raw_values: Sequence[str]
    if isinstance(values, str):
        raw_values = (values,)
    else:
        raw_values = values
    result = tuple(
        sorted({str(value).strip() for value in raw_values if str(value).strip()})
    )
    if any(value in {"*", "%"} for value in result):
        raise ValueError(f"{label} selectors must be exact identifiers")
    return result


def query_code_evidence_neighborhood(
    path: Path | str,
    *,
    task_id: str,
    symbols: Sequence[str] | str = (),
    dependency_task_ids: Sequence[str] | str = (),
    obligation_ids: Sequence[str] | str = (),
    receipt_ids: Sequence[str] | str = (),
    contradiction_ids: Sequence[str] | str = (),
    max_hops: int = 2,
    limit: int = 100,
) -> dict[str, Any]:
    """Return a deterministic, exact, row-bounded proof neighborhood.

    This is intentionally narrower than arbitrary graph traversal.  Repository
    tree and AST-blob nodes are never traversed, receipt/transcript children are
    terminal, and only proof-relevant edge directions may expand a seed.  The
    resulting query is therefore safe to feed into a context reducer without
    first materializing the complete evidence graph.
    """

    exact_task = str(task_id or "").strip()
    if not exact_task or exact_task in {"*", "%"}:
        raise ValueError("task_id must be one exact identifier")
    exact_symbols = _exact_strings(symbols, label="symbol")
    exact_dependencies = _exact_strings(dependency_task_ids, label="dependency task")
    exact_obligations = _exact_strings(obligation_ids, label="obligation")
    exact_receipts = _exact_strings(receipt_ids, label="receipt")
    exact_contradictions = _exact_strings(contradiction_ids, label="contradiction")
    hop_limit = int(max_hops)
    if hop_limit < 0 or hop_limit > MAX_GRAPH_QUERY_HOPS:
        raise ValueError(f"max_hops must be between 0 and {MAX_GRAPH_QUERY_HOPS}")
    row_limit = max(1, min(int(limit), MAX_QUERY_ROWS))
    database_path = ensure_query_database(path, kind=CODE_EVIDENCE_GRAPH_KIND)
    duckdb = _duckdb_module()

    seed_clauses = [
        "(node_kind = 'task' AND task_id = ?)",
        # Enrichments deliberately carry no authoritative task index.  Select
        # them by their exact declared target, not by graph alias inference.
        "(node_kind = 'enrichment' AND ("
        "json_extract_string(payload_json, '$.record.target') = ? OR "
        "json_extract_string(payload_json, '$.record.target_id') = ? OR "
        "json_contains(json_extract(payload_json, '$.record.targets'), ?) OR "
        "json_contains(json_extract(payload_json, '$.record.target_ids'), ?)"
        "))",
    ]
    parameters: list[Any] = [
        exact_task,
        exact_task,
        exact_task,
        json.dumps(exact_task),
        json.dumps(exact_task),
    ]

    def add_in(clause: str, values: tuple[str, ...]) -> None:
        if not values:
            return
        placeholders = ", ".join("?" for _ in values)
        seed_clauses.append(clause.format(placeholders=placeholders))
        parameters.extend(values)

    add_in("(node_kind = 'symbol' AND symbol IN ({placeholders}))", exact_symbols)
    add_in(
        "(node_kind = 'task' AND task_id IN ({placeholders}))",
        exact_dependencies,
    )
    add_in(
        "(node_kind = 'obligation' AND obligation_id IN ({placeholders}))",
        exact_obligations,
    )
    add_in(
        "(node_kind IN ('proof', 'validation', 'merge') "
        "AND record_key IN ({placeholders}))",
        exact_receipts,
    )
    if exact_contradictions:
        placeholders = ", ".join("?" for _ in exact_contradictions)
        seed_clauses.append(
            "("
            f"record_key IN ({placeholders}) OR "
            f"json_extract_string(payload_json, '$.record.contradiction_id') IN ({placeholders}) OR "
            f"json_extract_string(payload_json, '$.record.source_receipt_id') IN ({placeholders})"
            ")"
        )
        parameters.extend(exact_contradictions)
        parameters.extend(exact_contradictions)
        parameters.extend(exact_contradictions)

    node_columns = (
        "node_id, node_kind, record_key, provenance, authoritative, task_id, "
        "tree_id, symbol, obligation_id, assurance, freshness, payload_json"
    )

    def node_dict(row: Sequence[Any]) -> dict[str, Any]:
        names = (
            "node_id",
            "node_kind",
            "record_key",
            "provenance",
            "authoritative",
            "task_id",
            "tree_id",
            "symbol",
            "obligation_id",
            "assurance",
            "freshness",
            "payload_json",
        )
        value = {name: _jsonable(item) for name, item in zip(names, row)}
        value["payload"] = _json_value(str(value.pop("payload_json")))
        return value

    def edge_dict(row: Sequence[Any]) -> dict[str, Any]:
        names = (
            "edge_id",
            "source_node_id",
            "target_node_id",
            "edge_kind",
            "provenance",
            "provenance_record_id",
            "authoritative",
            "payload_json",
        )
        value = {name: _jsonable(item) for name, item in zip(names, row)}
        value["payload"] = _json_value(str(value.pop("payload_json")))
        return value

    connection = duckdb.connect(str(database_path), read_only=True)
    truncated = False
    try:
        seed_rows = connection.execute(
            f"SELECT {node_columns} FROM evidence_nodes WHERE "
            + " OR ".join(seed_clauses)
            + " ORDER BY node_kind, record_key, node_id "
            + f"LIMIT {row_limit + 1}",
            parameters,
        ).fetchall()
        if len(seed_rows) > row_limit:
            truncated = True
            seed_rows = seed_rows[:row_limit]
        selected = {str(row[0]): node_dict(row) for row in seed_rows}
        seed_ids = frozenset(selected)
        frontier = set(seed_ids)

        # Legal directional expansions by current node kind.  In particular,
        # TARGETS_TREE/CONTAINS/DEFINES_SYMBOL never expand a context query.
        allowed: dict[tuple[str, str, str], frozenset[str]] = {
            ("task", "out", "depends_on"): frozenset({"task"}),
            ("task", "out", "has_obligation"): frozenset({"obligation"}),
            ("task", "in", "validates"): frozenset({"validation"}),
            ("task", "in", "merged"): frozenset({"merge"}),
            ("task", "in", "completes"): frozenset({"merge"}),
            ("task", "in", "mentions"): frozenset({"enrichment"}),
            ("task", "in", "suggests"): frozenset({"enrichment"}),
            ("task", "in", "related_to"): frozenset({"enrichment"}),
            ("obligation", "out", "depends_on"): frozenset({"obligation"}),
            ("obligation", "out", "covers"): frozenset({"symbol"}),
            ("obligation", "in", "proves"): frozenset({"proof"}),
            ("obligation", "in", "derived_from"): frozenset({"proof"}),
            ("obligation", "in", "covers"): frozenset({"validation"}),
            ("symbol", "in", "covers"): frozenset({"obligation"}),
        }
        # An explicitly selected receipt may lead back to its exact subject,
        # but receipts discovered during traversal remain terminal.
        receipt_seed_expansions = {
            ("proof", "out", "proves"): frozenset({"obligation"}),
            ("proof", "out", "derived_from"): frozenset({"obligation"}),
            ("validation", "out", "covers"): frozenset({"obligation"}),
            ("validation", "out", "validates"): frozenset({"task"}),
            ("merge", "out", "merged"): frozenset({"task"}),
            ("merge", "out", "completes"): frozenset({"task"}),
            ("enrichment", "out", "mentions"): frozenset(
                {"task", "obligation", "symbol"}
            ),
            ("enrichment", "out", "suggests"): frozenset(
                {"task", "obligation", "symbol"}
            ),
            ("enrichment", "out", "related_to"): frozenset(
                {"task", "obligation", "symbol"}
            ),
        }
        candidate_edges: dict[str, dict[str, Any]] = {}
        for _hop in range(hop_limit):
            if not frontier or len(selected) >= row_limit:
                truncated = truncated or bool(frontier)
                break
            placeholders = ", ".join("?" for _ in frontier)
            edge_rows = connection.execute(
                "SELECT edge_id, source_node_id, target_node_id, edge_kind, "
                "provenance, provenance_record_id, authoritative, payload_json "
                "FROM evidence_edges "
                f"WHERE source_node_id IN ({placeholders}) "
                f"OR target_node_id IN ({placeholders}) "
                "ORDER BY edge_id "
                f"LIMIT {MAX_QUERY_ROWS + 1}",
                [*sorted(frontier), *sorted(frontier)],
            ).fetchall()
            if len(edge_rows) > MAX_QUERY_ROWS:
                truncated = True
                edge_rows = edge_rows[:MAX_QUERY_ROWS]
            neighbor_ids = sorted(
                {
                    str(row[index])
                    for row in edge_rows
                    for index in (1, 2)
                    if str(row[index]) not in selected
                }
            )
            if not neighbor_ids:
                break
            node_placeholders = ", ".join("?" for _ in neighbor_ids)
            neighbor_rows = connection.execute(
                f"SELECT {node_columns} FROM evidence_nodes "
                f"WHERE node_id IN ({node_placeholders}) ORDER BY node_id "
                f"LIMIT {MAX_QUERY_ROWS + 1}",
                neighbor_ids,
            ).fetchall()
            neighbor_map = {
                str(row[0]): node_dict(row) for row in neighbor_rows[:MAX_QUERY_ROWS]
            }
            accepted: set[str] = set()
            for row in edge_rows:
                edge = edge_dict(row)
                source_id = str(row[1])
                target_id = str(row[2])
                edge_kind = str(row[3])
                if source_id in selected and target_id in selected:
                    candidate_edges[str(row[0])] = edge
                for current_id, other_id, direction in (
                    (source_id, target_id, "out"),
                    (target_id, source_id, "in"),
                ):
                    if current_id not in frontier or other_id not in neighbor_map:
                        continue
                    current_kind = str(selected[current_id]["node_kind"])
                    next_kind = str(neighbor_map[other_id]["node_kind"])
                    permitted = allowed.get((current_kind, direction, edge_kind))
                    if current_id in seed_ids:
                        permitted = permitted or receipt_seed_expansions.get(
                            (current_kind, direction, edge_kind)
                        )
                    if permitted and next_kind in permitted:
                        accepted.add(other_id)
                        candidate_edges[str(row[0])] = edge
            frontier = set()
            for node_id in sorted(accepted):
                if len(selected) >= row_limit:
                    truncated = True
                    break
                selected[node_id] = neighbor_map[node_id]
                frontier.add(node_id)

        remaining = max(0, row_limit - len(selected))
        edges = [
            edge
            for _, edge in sorted(candidate_edges.items())
            if edge["source_node_id"] in selected and edge["target_node_id"] in selected
        ]
        if len(edges) > remaining:
            truncated = True
            edges = edges[:remaining]
    finally:
        connection.close()

    nodes = [selected[node_id] for node_id in sorted(selected)]
    return {
        "schema": QUERY_SCHEMA,
        "artifact_kind": CODE_EVIDENCE_GRAPH_KIND,
        "duckdb_path": str(database_path),
        "query": {
            "task_id": exact_task,
            "symbols": list(exact_symbols),
            "dependency_task_ids": list(exact_dependencies),
            "obligation_ids": list(exact_obligations),
            "receipt_ids": list(exact_receipts),
            "contradiction_ids": list(exact_contradictions),
        },
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "row_count": len(nodes) + len(edges),
        "truncated": truncated,
        "limit": row_limit,
        "max_hops": hop_limit,
    }


# Compatibility spelling for callers where the graph kind is implicit.
query_evidence_neighborhood = query_code_evidence_neighborhood


def artifact_schema(path: Path | str) -> dict[str, Any]:
    """Return typed table/column metadata without returning artifact rows."""

    return query_artifact(
        path,
        sql=(
            "SELECT table_name, column_name, data_type, is_nullable "
            "FROM information_schema.columns "
            "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position"
        ),
        limit=MAX_QUERY_ROWS,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bounded queries against supervisor JSON or DuckDB artifacts."
    )
    parser.add_argument("artifact_path", type=Path)
    parser.add_argument("--table", default="artifact_catalog")
    parser.add_argument("--columns", default="*")
    parser.add_argument("--where", default="")
    parser.add_argument("--sql", default="")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--schema", action="store_true", help="Return table and column metadata"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.schema:
        result = artifact_schema(args.artifact_path)
    else:
        columns = tuple(
            item.strip() for item in args.columns.split(",") if item.strip()
        ) or ("*",)
        result = query_artifact(
            args.artifact_path,
            table=args.table,
            columns=columns,
            where=args.where,
            sql=args.sql,
            limit=args.limit,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
