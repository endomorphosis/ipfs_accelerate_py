"""Fail-closed invariants for the operator-authored qualification plan."""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor import goal_graph, parse_goal_heap
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    external_authority_goal_fence,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OBJECTIVE_PATH = (
    REPO_ROOT / "docs/architecture/self_hosting_qualification.objectives.md"
)


def _goals():
    return parse_goal_heap(OBJECTIVE_PATH.read_text(encoding="utf-8"))


def _descendant_closure(goals, seed_goal_ids: set[str]) -> set[str]:
    closure = set(seed_goal_ids)
    while True:
        expanded = closure | {
            goal.goal_id
            for goal in goals
            if closure.intersection(goal.parent_goal_ids)
        }
        if expanded == closure:
            return closure
        closure = expanded


def test_plan_has_one_closed_combined_goal_dag() -> None:
    goals = _goals()
    goal_ids = [goal.goal_id for goal in goals]
    goal_id_set = set(goal_ids)
    hierarchy = goal_graph(goals)

    assert len(goals) == 47
    assert len(goal_id_set) == len(goal_ids)
    assert hierarchy["roots"] == ["SHQ-G000"]
    assert all(
        edge["from"] in goal_id_set and edge["to"] in goal_id_set
        for edge in hierarchy["edges"]
    )

    # Parent and explicit dependency relations are both prerequisites.  Check
    # the union so a cycle split across the two native fields cannot hide from
    # the hierarchy-only projection.
    prerequisites = {
        goal.goal_id: set(goal.parent_goal_ids) | set(goal.dependencies)
        for goal in goals
    }
    assert all(
        prerequisite in goal_id_set
        for values in prerequisites.values()
        for prerequisite in values
    )
    assert all(goal_id not in values for goal_id, values in prerequisites.items())

    dependents: dict[str, set[str]] = {goal_id: set() for goal_id in goal_ids}
    remaining = {
        goal_id: set(values) for goal_id, values in prerequisites.items()
    }
    for goal_id, values in remaining.items():
        for prerequisite in values:
            dependents[prerequisite].add(goal_id)
    ready = sorted(goal_id for goal_id, values in remaining.items() if not values)
    visited: list[str] = []
    while ready:
        goal_id = ready.pop(0)
        visited.append(goal_id)
        for dependent in sorted(dependents[goal_id]):
            remaining[dependent].discard(goal_id)
            if not remaining[dependent] and dependent not in visited and dependent not in ready:
                ready.append(dependent)
        ready.sort()

    assert set(visited) == goal_id_set, {
        goal_id: sorted(values)
        for goal_id, values in remaining.items()
        if values
    }


def test_external_admission_and_preregistration_gates_fail_closed() -> None:
    goals = _goals()

    declared_external, blocked = external_authority_goal_fence(
        goals,
        trust_recorded_completion=False,
    )
    expected_external = {"SHQ-G010", "SHQ-G072"}
    expected_blocked = _descendant_closure(goals, expected_external)

    assert declared_external == expected_external
    assert blocked == expected_blocked
    assert external_authority_goal_fence(goals)[1] == expected_blocked
    assert "SHQ-G006" not in blocked
    assert _descendant_closure(goals, {"SHQ-G072"}) == {
        "SHQ-G072",
        "SHQ-G073",
        "SHQ-G074",
        "SHQ-G075",
        "SHQ-G076",
    }


def test_work_units_target_and_repository_ownership_are_explicit() -> None:
    source = OBJECTIVE_PATH.read_text(encoding="utf-8")
    normalized_source = " ".join(source.split())
    goals = _goals()
    by_id = {goal.goal_id: goal for goal in goals}
    work_goals = [goal for goal in goals if goal.required_evidence]
    external_work_goals = [
        goal for goal in work_goals if goal.requires_external_completion
    ]
    local_work_goals = [
        goal for goal in work_goals if not goal.requires_external_completion
    ]

    assert "endomorphosis/ipfs_kit_py:ipfs_kit_py/core/wal" in source
    assert "`core.operation_contracts` is a read-only dependency" in normalized_source
    assert len(work_goals) == 39
    assert len(local_work_goals) == 37
    assert {goal.goal_id for goal in external_work_goals} == {
        "SHQ-G010",
        "SHQ-G072",
    }
    assert all(goal.fields.get("gap_task") for goal in work_goals)
    assert all(goal.predicted_files for goal in work_goals)
    assert all(goal.validation_commands for goal in work_goals)

    # These interfaces are the ownership seams: datasets owns task meaning and
    # evaluation, kit owns durable evidence/release state, MCP++ owns only the
    # wire schema, and accelerate owns orchestration/runtime behavior.
    assert "SelfHostingTaskCorpus" in by_id["SHQ-G032"].fields["interfaces"]
    assert "compare_task_outcomes" in by_id["SHQ-G036"].fields["interfaces"]
    assert "QualificationArtifactStore" in by_id["SHQ-G041"].fields["interfaces"]
    assert "create_qualification_manifest" in by_id["SHQ-G044"].fields["interfaces"]
    assert "QualificationRuntimePort@1" in by_id["SHQ-G031"].fields["interfaces"]
    assert "GovernedCodingAgentRuntime" in by_id["SHQ-G052"].fields["interfaces"]
    assert "SelfHostingQualificationHarness" in by_id["SHQ-G053"].fields["interfaces"]

    datasets_goal_ids = {
        "SHQ-G032",
        "SHQ-G033",
        "SHQ-G034",
        "SHQ-G035",
        "SHQ-G036",
        "SHQ-G037",
        "SHQ-G038",
    }
    assert {
        by_id[goal_id].fields["bundle"] for goal_id in datasets_goal_ids
    } == {"datasets/self-hosting/corpus"}
    assert all(
        by_id[goal_id].fields.get("submodules") == "ipfs_datasets_py"
        for goal_id in datasets_goal_ids
    )
    assert all(
        output.startswith(
            (
                "ipfs_datasets_py/",
                "artifacts/agent_supervisor/self_hosting_qualification/",
            )
        )
        for goal_id in datasets_goal_ids
        for output in by_id[goal_id].predicted_files
    )

    kit_goal_ids = {"SHQ-G041", "SHQ-G042", "SHQ-G043", "SHQ-G044"}
    assert all(
        output.startswith("ipfs_kit_py/")
        for goal_id in kit_goal_ids
        for output in by_id[goal_id].predicted_files
    )
