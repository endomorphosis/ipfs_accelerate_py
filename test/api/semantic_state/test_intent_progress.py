"""Native observed graph counts are never promoted to goal completion."""
from copy import deepcopy

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state.intent_progress import project_intent_progress
from ipfs_accelerate_py.agent_supervisor.semantic_state.intent_world_snapshot import capture_intent_world_snapshot
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


def populate(intent):
    intent.upsert_goal(goal_cid="root", goal_alias="ROOT", title="Root")
    intent.upsert_goal(goal_cid="child", goal_alias="CHILD", title="Child", parent_goal_cid="root")
    intent.upsert_task(task_cid="first", task_alias="FIRST", goal_cid="child", body={"title": "First"})
    intent.upsert_task(task_cid="second", task_alias="SECOND", goal_cid="child", body={"title": "Second"})


def test_native_selected_population_retains_parent_links_without_global_closure(tmp_path):
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        populate(intent)
        before = intent.event_watermark()
        selected = capture_intent_world_snapshot(intent, repository_id="actual-local-repository", task_cids=["first"])
        progress = project_intent_progress(selected)
        assert intent.event_watermark() == before
        assert progress["status"] == "observed"
        assert progress["population_scope"] == "selected-tasks"
        assert progress["task_population_complete"] is False
        assert progress["roots"] == ["root"]
        assert progress["parent_edges"] == [{"parent_goal_cid": "root", "child_goal_cid": "child",
                                               "source_field": "goals.parent_goal_cid"}]
        goals = {item["goal_cid"]: item for item in progress["goals"]}
        assert goals["root"]["child_goal_cids"] == ["child"]
        assert goals["root"]["direct_task_cids"] == []
        assert goals["child"]["direct_task_cids"] == ["first"]
        assert all(goal["status"] == "open" for goal in goals.values())
        assert all(goal["observed_counts"] == {
            "observed_descendant_tasks": 1, "observed_status_counts": {"ready": 1},
            "current_completion_receipts": 0, "completed_with_current_receipt": 0,
        } for goal in goals.values())
        assert progress["tasks"][0]["current_completion_receipt"] is None
        assert progress["goal_completion_authority"] == "unresolved"
        assert progress["goal_contracts_evaluated"] is False
        assert progress["completion_authority"] is False
        assert progress["canonical_state_mutated"] is False
        entire = project_intent_progress(capture_intent_world_snapshot(intent, repository_id="actual-local-repository"))
        assert entire["task_population_complete"] is True
        assert entire["population_scope"] == "all-intent-tasks"
        assert {task["task_cid"] for task in entire["tasks"]} == {"first", "second"}
        assert next(goal for goal in entire["goals"] if goal["goal_cid"] == "root")["observed_counts"]["observed_descendant_tasks"] == 2
        assert entire["completion_authority"] is False
        assert intent.event_watermark() == before


@pytest.mark.parametrize("parent, reason", [("child", "cyclic_goal_hierarchy"), ("absent", "missing_parent_goal")])
def test_real_native_incomplete_hierarchy_stays_unavailable(tmp_path, parent, reason):
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        populate(intent)
        intent.upsert_goal(goal_cid="root", goal_alias="ROOT", title="Root", parent_goal_cid=parent, expected_revision=1)
        before = intent.event_watermark()
        result = project_intent_progress(capture_intent_world_snapshot(intent, repository_id="actual-local-repository"))
        assert result["status"] == "unavailable" and reason in result["unavailable_reasons"]
        assert all(goal["observed_counts"] is None for goal in result["goals"])
        assert all(goal["status"] == "open" for goal in result["goals"])
        assert result["goal_contracts_evaluated"] is False
        assert intent.event_watermark() == before


def test_capture_tampering_cannot_turn_an_observed_task_into_completion(tmp_path):
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        populate(intent)
        capture = capture_intent_world_snapshot(intent, repository_id="actual-local-repository")
        mutated = deepcopy(capture)
        mutated["plan_projection"]["tasks"][0]["status"] = "completed"
        with pytest.raises(ValueError, match="intact native world capture"):
            project_intent_progress(mutated)
        assert all(task["status"] == "ready" for task in intent.plan_projection()["tasks"])
