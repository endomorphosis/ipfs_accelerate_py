"""Native admission of worldwide goals and tasks. Markdown is bootstrap only."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.objectives.goal_quality import GoalQualityError, ObjectiveTypedGoals
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search import (
    ADMITTED_ROOT_BREADTH, ADMISSION_SCHEMA, NATIVE_DEFAULT_BREADTH, ROOT_GOAL_ID,
    WorldwideAdmissionError, admit_goals_and_tasks,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource


PACKAGE = Path("/home/barberb/worldwidelaws.github.io/implementation_plan/worldwide_legal_search")
RECEIPT = Path("artifacts/worldwide-legal-search/receipts/WWL-002.json")
FOREST = {
    "site": {"path": "/site", "head": "1" * 40, "tree": "2" * 40},
    "accelerate": {"path": "/framework", "head": "3" * 40, "tree": "4" * 40},
}


def campaign(*, children: int = ADMITTED_ROOT_BREADTH, namespace: str = "worldwide-legal-search-v1", acceptance: str = "Native current-tree receipts bind this contract"):
    tasks, goals = [], []
    for index in range(children):
        task_id = f"WWL-{index:03d}"
        goal_id = f"WWL-G{(index + 1) * 10:03d}"
        tasks.append({
            "id": task_id, "title": "Fixture task " + task_id, "status": "done", "depends_on": [] if index == 0 else [f"WWL-{index - 1:03d}"],
            "repository": "site" if index % 2 == 0 else "accelerate", "goal_id": goal_id, "subgoal_id": goal_id,
            "outputs": [f"artifacts/worldwide-legal-search/receipts/{task_id}.json"],
            "evidence_output": f"artifacts/worldwide-legal-search/receipts/{task_id}.json",
            "acceptance": acceptance, "validation_argv": [["python3", "-c", "print('validated fixture')"]],
            "bounds": {"max_attempts": 3, "wall_seconds_per_attempt": 5400},
        })
        goals.append({"id": goal_id, "title": "Fixture child " + goal_id, "parent_ids": [ROOT_GOAL_ID], "task_ids": [task_id], "acceptance": acceptance})
    plan = {
        "schema": "worldwide-legal-search.plan@1", "namespace": namespace,
        "goals": [{"id": ROOT_GOAL_ID, "title": "Fixture root", "parent_ids": [], "task_ids": [task["id"] for task in tasks], "acceptance": acceptance}, *goals],
        "tasks": tasks,
    }
    heap = ["# Worldwide legal search goal bootstrap", ""]
    for goal in plan["goals"]:
        heap.extend([
            f"## {goal['id']} {goal['title']}", "- Status: active",
            "- Parent goal ids json: " + json.dumps(goal["parent_ids"]),
            "- Required task ids json: " + json.dumps(goal["task_ids"]),
            "- Acceptance: " + goal["acceptance"], "",
        ])
    markdown_tasks = [f"## {task['id']} {task['title']}\n- Status: done\n" for task in tasks]
    return plan, "\n".join(heap), "\n".join(markdown_tasks)


def admit(tmp_path: Path, *, name: str = "control.duckdb", forest=None, **overrides):
    plan, heap, tasks = campaign(**overrides)
    root = tmp_path / name
    board = tmp_path / (name + ".tasks.md")
    board.write_text(tasks, encoding="utf-8")
    before = board.read_bytes()
    receipt = admit_goals_and_tasks(
        plan, heap, database_path=root, source_forest=FOREST if forest is None else forest,
        config_path=tmp_path / "runtime.json", python_executable="python3", task_markdown=tasks,
    )
    assert board.read_bytes() == before
    return plan, receipt, root


def test_admission_materializes_stable_dependencies_and_replays_in_place(tmp_path):
    plan, receipt, database = admit(tmp_path)
    assert receipt["schema"] == ADMISSION_SCHEMA
    assert receipt["idempotent_replay"] is False
    assert receipt["execution_completed"] is False
    assert receipt["markdown_is_execution_authority"] is False
    assert receipt["markdown_bootstrap_only"] is True
    assert receipt["ready_task_ids"] == ["WWL-000"]
    assert receipt["task_count"] == 10
    assert receipt["goal_count"] == 11
    assert receipt["dependency_count"] == 9
    assert set(receipt["task_statuses"].values()) == {"todo"}
    assert receipt["markdown_completed_task_ids"] == [f"WWL-{index:03d}" for index in range(10)]
    document = ObjectiveTypedGoals.from_dict(receipt["typed_goals"])
    assert document.objective_heap_id == receipt["objective_heap_id"]
    assert len(document.goals) == 11
    root = document.goal_map()[ROOT_GOAL_ID]
    assert list(root.scope.include) == receipt["root_child_ids"] == [f"WWL-G{(index + 1) * 10:03d}" for index in range(10)]
    assert root.refinement_budget.max_children == ADMITTED_ROOT_BREADTH == 10
    assert root.resources.max_scope_items >= 10
    for goal in document.goals:
        assert goal.assumptions and goal.non_goals and goal.evidence_producers and goal.validation_rules
        assert goal.freshness.max_age_seconds == 3600
        assert goal.resources.bounded and goal.refinement_budget.bounded and goal.uncertainties
        assert goal.uncertainties[0].resolution
        assert all("*" not in item for item in goal.scope.include)
        assert "validate-goal" in goal.validation_rules[0].command
    with DatabaseTaskSource(database, owner_id="worldwide-legal-search-v1:wwl-admission") as source:
        first, second = source.get_task("WWL-000"), source.get_task("WWL-001")
        assert second.dependencies == (first.task_cid,)
        assert list(first.validations[0]["argv"]) == plan["tasks"][0]["validation_argv"][0]
        assert first.body["base_revision"] == FOREST["site"]["head"]
        assert second.body["base_revision"] == FOREST["accelerate"]["head"]
        assert first.body["source_plan_task"]["status"] == "done"
        assert first.status == "todo"
        progressed = source.compare_and_set_status(first.task_cid, first.revision, "in_progress", {"operation": "fixture-progress"})
    replay = admit_goals_and_tasks(
        plan, campaign()[1], database_path=database, source_forest=FOREST,
        config_path=tmp_path / "runtime.json", python_executable="python3", task_markdown=campaign()[2],
    )
    assert replay["idempotent_replay"] is True
    assert replay["task_cids_by_alias"] == receipt["task_cids_by_alias"]
    assert replay["plan_root_cid"] == receipt["plan_root_cid"]
    assert replay["task_revisions"]["WWL-000"] == progressed.task.revision
    assert replay["task_statuses"]["WWL-000"] == "in_progress"
    assert replay["ready_task_ids"] == []
    drift = copy.deepcopy(FOREST)
    drift["site"]["head"] = "f" * 40
    drifted = admit_goals_and_tasks(
        plan, campaign()[1], database_path=database, source_forest=drift,
        config_path=tmp_path / "moved.json", python_executable="/usr/bin/python3.12", task_markdown=campaign()[2],
    )
    assert drifted["idempotent_replay"] is True
    assert drifted["caller_source_forest_ignored"] is True
    assert drifted["task_cids_by_alias"] == receipt["task_cids_by_alias"]
    assert drifted["task_statuses"]["WWL-000"] == "in_progress"
    assert drifted["repository_tree_id"] == receipt["repository_tree_id"]
    changed = copy.deepcopy(plan)
    changed["tasks"][1]["depends_on"] = []
    with pytest.raises(WorldwideAdmissionError, match="plan definition"):
        admit_goals_and_tasks(
            changed, campaign()[1], database_path=database, source_forest=FOREST,
            config_path=tmp_path / "runtime.json", python_executable="python3", task_markdown=campaign()[2],
        )
    with DatabaseTaskSource(database, owner_id="worldwide-legal-search-v1:wwl-admission") as source:
        preserved = source.get_task("WWL-001")
    assert preserved is not None
    assert preserved.revision == receipt["task_revisions"]["WWL-001"]
    assert preserved.status == "todo"
    assert drifted["task_revisions"]["WWL-001"] == receipt["task_revisions"]["WWL-001"]


def test_namespaces_stay_isolated(tmp_path):
    plan, receipt, database = admit(tmp_path)
    other_root = tmp_path / "other.duckdb"
    other_plan, other_heap, other_tasks = campaign(namespace="worldwide-legal-search-other")
    other = admit_goals_and_tasks(
        other_plan, other_heap, database_path=other_root, source_forest=FOREST,
        config_path=tmp_path / "runtime.json", python_executable="python3", task_markdown=other_tasks,
    )
    assert other["namespace"] == "worldwide-legal-search-other"
    assert other["task_cids_by_alias"]["WWL-000"] != receipt["task_cids_by_alias"]["WWL-000"]
    assert other["task_statuses"]["WWL-000"] == "todo"
    with pytest.raises(WorldwideAdmissionError, match="isolated namespace"):
        admit_goals_and_tasks(
            other_plan, other_heap, database_path=database, source_forest=FOREST,
            config_path=tmp_path / "runtime.json", python_executable="python3", task_markdown=other_tasks,
        )
    with DatabaseTaskSource(database, owner_id="worldwide-legal-search-v1:wwl-admission") as source:
        assert source.get_task("WWL-000").status == "todo"
        assert source.get_task("WWL-000").task_cid == receipt["task_cids_by_alias"]["WWL-000"]


def test_rejected_contracts_do_not_create_a_namespace(tmp_path):
    narrow, narrow_heap, narrow_tasks = campaign(children=NATIVE_DEFAULT_BREADTH)
    narrow_path = tmp_path / "narrow" / "control.duckdb"
    with pytest.raises(WorldwideAdmissionError, match="ten-child"):
        admit_goals_and_tasks(
            narrow, narrow_heap, database_path=narrow_path, source_forest=FOREST,
            config_path=tmp_path / "runtime.json", python_executable="python3", task_markdown=narrow_tasks,
        )
    assert not narrow_path.parent.exists()
    vague, vague_heap, vague_tasks = campaign(acceptance="Ship a reasonable milestone")
    vague_path = tmp_path / "vague" / "control.duckdb"
    with pytest.raises(GoalQualityError, match="goal quality"):
        admit_goals_and_tasks(
            vague, vague_heap, database_path=vague_path, source_forest=FOREST,
            config_path=tmp_path / "runtime.json", python_executable="python3", task_markdown=vague_tasks,
        )
    assert not vague_path.parent.exists()


def test_receipt_matches_the_admission_contract():
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["task_id"] == "WWL-002"
    assert payload["admitted_root_breadth"] == ADMITTED_ROOT_BREADTH == 10
    assert payload["native_default_breadth"] == NATIVE_DEFAULT_BREADTH == 8
    assert payload["markdown_bootstrap_only"] is True
    assert payload["markdown_is_execution_authority"] is False
    assert payload["execution_completed"] is False
    assert payload["completion_authority"] is False
    assert payload["validation_argv"] == ["python3", "-m", "pytest", "test/api/test_worldwide_legal_search_admission.py", "-q"]
    assert payload["task_source"] == "DatabaseTaskSource"
    assert "worldwide_legal_search.py" in "".join(payload["outputs"])


def test_real_campaign_admits_the_ten_child_root(tmp_path):
    plan = json.loads((PACKAGE / "plan.json").read_text(encoding="utf-8"))
    heap = (PACKAGE / "goals.objectives.md").read_text(encoding="utf-8")
    tasks = (PACKAGE / "tasks.todo.md").read_text(encoding="utf-8")
    database = tmp_path / "campaign.duckdb"
    receipt = admit_goals_and_tasks(
        plan, heap, database_path=database, source_forest=FOREST,
        config_path=tmp_path / "runtime.json", python_executable="python3", task_markdown=tasks,
    )
    assert receipt["typed_goals_accepted"] is True
    assert receipt["root_child_ids"] == [f"WWL-G{number:03d}" for number in range(10, 110, 10)]
    assert receipt["admitted_root_breadth"] == 10
    assert receipt["native_default_breadth"] == 8
    assert receipt["default_breadth_rejects_root"] is True
    assert receipt["task_count"] == len(plan["tasks"])
    assert receipt["goal_count"] == len(plan["goals"]) == 31
    assert "WWL-000" in receipt["ready_task_ids"]
    assert "WWL-002" not in receipt["ready_task_ids"]
    assert set(receipt["task_statuses"].values()) == {"todo"}
    replay = admit_goals_and_tasks(
        plan, heap, database_path=database, source_forest=FOREST,
        config_path=tmp_path / "runtime.json", python_executable="python3", task_markdown=tasks,
    )
    assert replay["idempotent_replay"] is True
    assert replay["task_cids_by_alias"] == receipt["task_cids_by_alias"]
    assert replay["task_revisions"] == receipt["task_revisions"]
