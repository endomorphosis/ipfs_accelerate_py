from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.artifact_store import (
    BUNDLE_INDEX_KIND,
    QUERY_SCHEMA,
    SCHEDULER_MANIFEST_KIND,
    artifact_schema,
    query_artifact,
    read_artifact_fields,
    read_bundle_index_planning_projection,
    read_bundle_index_projection,
    write_bundle_index_artifact,
    write_scheduler_manifest_artifact,
)
from ipfs_accelerate_py.agent_supervisor.objective_graph import (
    build_bundle_task_payloads,
)


def _bundle_index() -> dict[str, object]:
    return {
        "schema": "test.bundle-index@1",
        "generated_at": "2026-07-22T00:00:00Z",
        "source_todo": "tasks.todo.md",
        "excluded_bundle_keys": ["objective/test/excluded"],
        "bundles": {
            "objective/test/one": {
                "bundle_key": "objective/test/one",
                "shard_path": "bundles/one.todo.md",
                "parallel_lane": "lane-one",
                "conflict_policy": "bundle-local",
                "tasks": [
                    {
                        "task_id": "T-1",
                        "canonical_task_cid": "cid-t-1",
                        "status": "todo",
                        "priority": "P0",
                        "title": "First task",
                        "depends_on": [],
                    },
                    {
                        "task_id": "T-2",
                        "canonical_task_cid": "cid-t-2",
                        "status": "completed",
                        "priority": "P1",
                        "title": "Second task",
                        "depends_on": ["T-1"],
                    },
                ],
            }
        },
    }


def test_bundle_index_has_equivalent_json_and_duckdb_queries(tmp_path: Path) -> None:
    json_path = tmp_path / "index.json"

    rendered = write_bundle_index_artifact(json_path, _bundle_index())

    duckdb_path = tmp_path / "index.duckdb"
    assert json_path.exists()
    assert duckdb_path.exists()
    assert rendered["query_store"] == {
        "schema": QUERY_SCHEMA,
        "artifact_kind": BUNDLE_INDEX_KIND,
        "duckdb_path": "index.duckdb",
        "catalog_table": "artifact_catalog",
    }
    json_query = query_artifact(
        json_path,
        table="bundle_tasks",
        columns=("task_id", "status", "title"),
        where="status = 'todo'",
        limit=5,
    )
    duckdb_query = query_artifact(
        duckdb_path,
        table="bundle_tasks",
        columns=("task_id", "status", "title"),
        where="status = 'todo'",
        limit=5,
    )
    assert (
        json_query["rows"]
        == duckdb_query["rows"]
        == [{"task_id": "T-1", "status": "todo", "title": "First task"}]
    )
    assert read_artifact_fields(json_path, ("source_todo", "schema")) == {
        "source_todo": "tasks.todo.md",
        "schema": "test.bundle-index@1",
    }

    projection = read_bundle_index_projection(duckdb_path)
    assert projection["bundles"]["objective/test/one"]["tasks"][1]["task_id"] == "T-2"
    assert (
        build_bundle_task_payloads(duckdb_path)[0]["bundle_key"] == "objective/test/one"
    )
    schema_rows = artifact_schema(json_path)["rows"]
    assert any(
        row["table_name"] == "bundle_tasks"
        and row["column_name"] == "canonical_task_cid"
        for row in schema_rows
    )


def test_bundle_planning_projection_omits_repeated_evidence_blobs(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "index.json"
    payload = _bundle_index()
    bundle = payload["bundles"]["objective/test/one"]
    bundle["todo_vector_summary"] = {"conflict_decisions": ["large"]}
    task = bundle["tasks"][0]
    task.update(
        {
            "ast_symbols": ["module.symbol"],
            "conflict_surface": {"ast_records": ["large"]},
            "conflict_edges": [{"left": "T-1", "right": "T-2"}],
            "conflict_decisions": [{"left": "T-1", "right": "T-2"}],
            "coverage_inputs": {"records": ["large"]},
        }
    )
    write_bundle_index_artifact(json_path, payload)

    complete = read_bundle_index_projection(json_path)
    projected = read_bundle_index_planning_projection(json_path)
    complete_bundle = complete["bundles"]["objective/test/one"]
    complete_task = complete_bundle["tasks"][0]
    projected_bundle = projected["bundles"]["objective/test/one"]
    projected_task = projected_bundle["tasks"][0]

    assert "todo_vector_summary" in complete_bundle
    assert complete_bundle["todo_vector_summary"]["conflict_decision_count"] == 1
    assert "conflict_decisions" not in complete_bundle["todo_vector_summary"]
    assert complete_task["conflict_decision_count"] == 1
    assert complete_task["conflict_edge_count"] == 1
    assert "ast_records" not in complete_task["conflict_surface"]
    assert complete_task["conflict_surface"]["ast_record_count"] == 1
    assert "coverage_inputs" not in complete_task
    assert "todo_vector_summary" not in projected_bundle
    assert "conflict_surface" not in projected_task
    assert "conflict_edges" not in projected_task
    assert "conflict_decisions" not in projected_task
    assert "coverage_inputs" not in projected_task
    assert projected_task["ast_symbols"] == ["module.symbol"]
    assert projected_task["depends_on"] == []

    write_bundle_index_artifact(json_path, complete)
    rewritten = read_bundle_index_projection(json_path)
    rewritten_task = rewritten["bundles"]["objective/test/one"]["tasks"][0]
    assert rewritten_task["conflict_surface"]["ast_record_count"] == 1


def test_scheduler_manifest_normalizes_rows_and_bounds_query_output(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "bundle_lanes.json"
    payload = {
        "schema": "ipfs_accelerate_py.agent_supervisor.dynamic_bundle_scheduler@1",
        "generated_at": "2026-07-22T00:00:00Z",
        "scheduler_state": "running",
        "cycle": 4,
        "counts": {"active": 1, "ready": 2, "blocked": 0, "completed": 0},
        "tasks": [
            {
                "task_cid": f"cid-{ordinal}",
                "task_id": f"T-{ordinal}",
                "bundle_key": f"objective/test/{ordinal}",
                "state": "ready",
            }
            for ordinal in range(3)
        ],
        "lanes": [
            {
                "bundle_key": "objective/test/active",
                "parallel_lane": "lane-active",
                "task_cid": "cid-active",
                "task_ids": ["T-ACTIVE"],
                "state": "running",
                "pid": 123,
                "claimable": True,
            }
        ],
        "scheduler_decisions": [
            {
                "task_cid": "cid-active",
                "bundle_key": "objective/test/active",
                "decision": "launched",
                "reason": "capacity_available",
                "snapshot_id": "snapshot-1",
            }
        ],
        "conflict_graph": {
            "edges": [
                {
                    "left_task_id": "objective/test/active",
                    "right_task_id": "objective/test/2",
                    "blocks_concurrency": True,
                }
            ],
            "decisions": [
                {
                    "left_task_cid": "cid-active",
                    "right_task_cid": "cid-2",
                    "action": "serialize",
                    "weight": 1.0,
                }
            ],
        },
        "scheduler_snapshot": {
            "task_states": [
                {
                    "task_cid": "cid-active",
                    "task_id": "T-ACTIVE",
                    "phase": "active",
                    "status": "active",
                }
            ],
            "metrics": [
                {
                    "task_cid": "cid-active",
                    "queue_wait_seconds": 2.5,
                    "retries": 1,
                    "total_tokens": 120,
                }
            ],
            "phases": {"active": {"count": 1, "items": []}},
        },
    }

    rendered = write_scheduler_manifest_artifact(manifest_path, payload)

    assert rendered["query_store"]["artifact_kind"] == SCHEDULER_MANIFEST_KIND
    assert (tmp_path / "bundle_lanes.duckdb").exists()
    ready = query_artifact(manifest_path, table="ready_tasks", limit=2)
    assert ready["row_count"] == 2
    assert ready["truncated"] is True
    active = query_artifact(
        tmp_path / "bundle_lanes.duckdb",
        table="active_lanes",
        columns=("bundle_key", "pid", "task_ids_json"),
        limit=5,
    )
    assert active["rows"] == [
        {
            "bundle_key": "objective/test/active",
            "pid": 123,
            "task_ids_json": '["T-ACTIVE"]',
        }
    ]
    decision = query_artifact(
        manifest_path,
        sql="SELECT decision, reason FROM manifest_decisions WHERE task_cid = 'cid-active'",
        limit=5,
    )
    assert decision["rows"] == [
        {"decision": "launched", "reason": "capacity_available"}
    ]
    lifecycle = query_artifact(
        manifest_path,
        table="scheduler_task_states",
        columns=("task_id", "phase", "status"),
    )
    assert lifecycle["rows"] == [
        {"task_id": "T-ACTIVE", "phase": "active", "status": "active"}
    ]
    conflict = query_artifact(
        manifest_path,
        table="manifest_conflict_decisions",
        columns=("action", "weight"),
    )
    assert conflict["rows"] == [{"action": "serialize", "weight": 1.0}]
    with pytest.raises(ValueError, match="read-only"):
        query_artifact(manifest_path, sql="DELETE FROM manifest_tasks")


def test_json_changes_refresh_the_query_sidecar(tmp_path: Path) -> None:
    json_path = tmp_path / "index.json"
    write_bundle_index_artifact(json_path, _bundle_index())
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["bundles"]["objective/test/one"]["tasks"].append(
        {"task_id": "T-3", "status": "todo", "title": "Refilled task"}
    )
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    refreshed = query_artifact(
        json_path,
        table="bundle_tasks",
        columns=("task_id",),
        where="task_id = 'T-3'",
    )

    assert refreshed["rows"] == [{"task_id": "T-3"}]
