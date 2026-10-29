"""Canonical taskboard ingestion retains declared goal/subgoal hierarchy."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import taskboard_ingest as ingest
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import default_control_plane_schema
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


pytestmark = pytest.mark.proof_reuse_disabled(reason="Exercise hierarchy validation and native goal persistence")


def _board() -> dict:
    return {
        "board_namespace": "goal-hierarchy-regression-v1", "plan_revision": "R1",
        # Forward references intentionally put a grandchild first.
        "goals": [
            {"goal_id": "G-LEAF", "title": "Specific result", "parent": "G-CHILD"},
            {"goal_id": "G-ROOT", "goal_cid": "goal:root", "title": "Root", "status": "active"},
            {"goal_id": "G-CHILD", "goal_alias": "child-name", "title": "Child", "parent_goal_cid": "goal:root"},
        ],
        "tasks": [
            {"task_id": "T-1", "status": "ready", "subgoal_id": "G-LEAF", "dependencies": []},
            {"task_id": "T-2", "status": "blocked", "goal_id": "child-name", "dependencies": ["T-1"]},
        ],
    }


def test_population_resolves_forward_parent_aliases_and_exact_task_goal_cids() -> None:
    population = ingest.board_to_population(_board())
    goals = {goal["goal_id"]: goal for goal in population["goals"]}
    assert goals["G-ROOT"]["goal_cid"] == "goal:root"
    assert goals["G-CHILD"]["parent_goal_cid"] == "goal:root"
    assert goals["G-LEAF"]["parent_goal_cid"] == goals["G-CHILD"]["goal_cid"]
    assert population["plans"][0]["goal_cid"] == "goal:root"
    assert population["tasks"][0]["goal_cid"] == goals["G-LEAF"]["goal_cid"]
    assert population["tasks"][1]["goal_cid"] == goals["G-CHILD"]["goal_cid"]
    assert population["tasks"][1]["dependencies"] == [population["tasks"][0]["task_cid"]]
    assert {edge["edge_kind"] for edge in population["goal_edges"]} == {"subgoal"}
    assert len(population["goal_edges"]) == 2


def test_hierarchy_does_not_change_existing_plan_or_goal_identity_cids() -> None:
    board = _board()
    flat = copy.deepcopy(board)
    for goal in flat["goals"]:
        for key in ("parent", "parent_goal_id", "parent_goal_cid"):
            goal.pop(key, None)
    old = ingest.board_to_population(flat)
    new = ingest.board_to_population(board)
    for key in ("repository_tree_id", "plan_root_cid"):
        assert old[key] == new[key]
    assert [goal["goal_cid"] for goal in old["goals"]] == [goal["goal_cid"] for goal in new["goals"]]
    assert [task["task_cid"] for task in old["tasks"]] == [task["task_cid"] for task in new["tasks"]]
    assert old["goals"][1]["status"] == new["goals"][1]["status"] == "active"


@pytest.mark.parametrize("change,match", [
    (lambda b: b["goals"][0].update(parent="missing-parent"), "unknown parent"),
    (lambda b: b["goals"][0].update(parent="G-LEAF"), "cycle"),
    (lambda b: b["goals"][1].update(parent="G-LEAF"), "cycle"),
    (lambda b: b["goals"].append(dict(b["goals"][0])), "duplicate"),
    (lambda b: b["goals"][0].update(goal_cid="goal:root"), "duplicate"),
    (lambda b: b["goals"][0].update(parent_goal_id="G-ROOT"), "conflicting"),
    (lambda b: b["goals"][0].update(ordinal="not-an-integer"), "ordinal"),
    (lambda b: b["goals"][0].update(ordinal=[]), "ordinal"),
    (lambda b: b["tasks"][0].update(subgoal_id="unknown-goal"), "unknown goal reference"),
])
def test_invalid_hierarchy_is_rejected_before_database_materialization(change, match: str, tmp_path: Path, monkeypatch) -> None:
    board = _board()
    change(board)
    assert any(match in error for error in ingest.validate_taskboard(board))
    with pytest.raises(ingest.TaskboardIngestError, match=match):
        ingest.board_to_population(board)
    board_path = tmp_path / "invalid.json"
    board_path.write_text(json.dumps(board))
    monkeypatch.setattr(ingest, "probe_ingest_quack", lambda **_kwargs: {})

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid hierarchy reached persistent storage")

    monkeypatch.setattr(ingest, "DatabaseTaskSource", forbidden)
    with pytest.raises(ingest.TaskboardIngestError, match=match):
        ingest.ingest_taskboard(board_path=board_path, store_path=tmp_path / "never-created.duckdb", require_quack=False)
    assert not (tmp_path / "never-created.duckdb").exists()


def test_matching_parent_alias_and_cid_references_are_not_conflicting() -> None:
    board = _board()
    board["goals"][2].update(parent="G-ROOT", parent_goal_id="G-ROOT")
    assert ingest.validate_taskboard(board) == []
    assert len(ingest.board_to_population(board)["goal_edges"]) == 2


def test_task_only_legacy_boards_keep_default_root_and_do_not_invent_hierarchy() -> None:
    board = {"tasks": [{"task_id": "T-1", "status": "ready", "subgoal_id": "external-legacy-goal"}]}
    assert ingest.validate_taskboard(board) == []
    population = ingest.board_to_population(board)
    assert population["goals"][0]["goal_alias"] == "G-ROOT"
    assert population["tasks"][0]["goal_cid"] == population["goals"][0]["goal_cid"]
    assert population["tasks"][0]["subgoal_id"] == "external-legacy-goal"
    assert population["goal_edges"] == []


def test_real_native_memory_database_retains_parent_columns_and_subgoal_edges() -> None:
    duckdb = pytest.importorskip("duckdb")
    connection = duckdb.connect(":memory:", config={"threads": 1})
    try:
        connection.execute(default_control_plane_schema().sql_text())
        repository = IntentRepository(bound_connection=connection, owner_id="hierarchy-test-owner")
        population = ingest.board_to_population(_board())
        connection.execute("BEGIN TRANSACTION")
        try:
            for goal in population["goals"]:
                repository.upsert_goal(goal_cid=goal["goal_cid"], goal_alias=goal["goal_alias"], title=goal["title"],
                                       parent_goal_cid=goal.get("parent_goal_cid", ""), ordinal=goal["ordinal"], status=goal["status"])
            for edge in population["goal_edges"]:
                repository.link_goal_edge(**edge)
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        assert connection.execute("SELECT count(*) FROM goals").fetchone()[0] == 3
        assert connection.execute("SELECT count(*) FROM goals WHERE parent_goal_cid <> ''").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM goal_edges WHERE edge_kind='subgoal'").fetchone()[0] == 2
        leaf = repository.get_goal("G-LEAF")
        child = repository.get_goal("G-CHILD")
        assert leaf["parent_goal_cid"] == child["goal_cid"]
        assert child["parent_goal_cid"] == "goal:root"
    finally:
        connection.close()


def test_repeat_ingest_backfills_a_legacy_flat_hierarchy_without_rewriting_tasks(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    board = _board()
    flat = copy.deepcopy(board)
    for goal in flat["goals"]:
        for key in ("parent", "parent_goal_id", "parent_goal_cid"):
            goal.pop(key, None)
    source = tmp_path / "board.json"
    store = tmp_path / "plan.duckdb"
    source.write_text(json.dumps(flat))
    ingest.ingest_taskboard(board_path=source, store_path=store, require_quack=False)

    def rows(sql: str):
        connection = duckdb.connect(str(store), read_only=True)
        try:
            return connection.execute(sql).fetchall()
        finally:
            connection.close()

    tasks_before = rows("SELECT * FROM tasks ORDER BY task_cid")
    assert rows("SELECT count(*) FROM goal_edges") == [(0,)]
    source.write_text(json.dumps(board))
    repaired = ingest.ingest_taskboard(board_path=source, store_path=store, require_quack=False)
    assert repaired["inserted_count"] == 0
    assert repaired["hierarchy_backfilled"] is True
    assert rows("SELECT count(*) FROM goal_edges WHERE edge_kind='subgoal'") == [(2,)]
    assert rows("SELECT count(*) FROM goals WHERE parent_goal_cid <> ''") == [(2,)]
    assert rows("SELECT * FROM tasks ORDER BY task_cid") == tasks_before
    revisions = rows("SELECT goal_cid, revision FROM goals ORDER BY goal_cid")
    repeat = ingest.ingest_taskboard(board_path=source, store_path=store, require_quack=False)
    assert repeat["hierarchy_backfilled"] is False
    assert rows("SELECT goal_cid, revision FROM goals ORDER BY goal_cid") == revisions
    assert rows("SELECT * FROM tasks ORDER BY task_cid") == tasks_before
