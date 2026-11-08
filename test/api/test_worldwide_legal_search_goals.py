from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_goals import build_typed_goals, lint_typed_goals
from ipfs_accelerate_py.agent_supervisor.objectives.goal_quality import GoalQualityError, ObjectiveTypedGoals, validate_objective_typed_goals


PACKAGE = Path("/home/barberb/worldwidelaws.github.io/implementation_plan/worldwide_legal_search")


@pytest.fixture
def campaign():
    return json.loads((PACKAGE / "plan.json").read_text()), (PACKAGE / "goals.objectives.md").read_text()


def test_actual_campaign_goal_contracts_are_heap_bound_and_lint_clean(campaign, tmp_path):
    plan, markdown = campaign
    document = build_typed_goals(plan, markdown, config_path=tmp_path / "runtime.json", python_executable="python3")
    reports = lint_typed_goals(document)
    assert len(document.goals) == 31
    assert all(report.accepted and not report.debt for report in reports)
    assert ObjectiveTypedGoals.from_dict(document.to_dict()) == document
    root = document.goal_map()["WWL-G000"]
    assert len(root.scope.include) == 10
    assert root.refinement_budget.max_children == 10
    assert not root.authorized_completion_producer_ids
    assert all(goal.root == root.root for goal in document.goals)
    assert "validate-goal" in root.validation_rules[0].command


def test_typed_contracts_reject_stale_or_missing_heap_goals(campaign, tmp_path):
    plan, markdown = campaign
    document = build_typed_goals(plan, markdown, config_path=tmp_path / "runtime.json", python_executable="python3")
    with pytest.raises(GoalQualityError, match="stale"):
        validate_objective_typed_goals(markdown + "\n", document)
    missing = copy.deepcopy(plan)
    missing["goals"].pop()
    with pytest.raises(GoalQualityError, match="identities differ"):
        build_typed_goals(missing, markdown, config_path=tmp_path / "runtime.json", python_executable="python3")


def test_resource_and_scope_bounds_are_not_unbounded_defaults(campaign, tmp_path):
    plan, markdown = campaign
    document = build_typed_goals(plan, markdown, config_path=tmp_path / "runtime.json", python_executable="python3")
    for goal in document.goals:
        assert goal.resources.bounded and goal.refinement_budget.bounded
        assert goal.resources.max_parallelism == 1
        assert goal.freshness.max_age_seconds == 3600
        assert all("*" not in value for value in goal.scope.include)
    with pytest.raises(GoalQualityError, match="freshness"):
        build_typed_goals(plan, markdown, config_path=tmp_path / "runtime.json", python_executable="python3", freshness_seconds=0)
