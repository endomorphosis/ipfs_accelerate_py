"""Focused goal/task completion contract checks."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.objectives.completion_contracts import (
    TASK_GATES,
    CompletionContractError,
    completion_truth_table,
    evaluate_goal_completion,
    evaluate_task_completion,
)


def test_task_requires_every_gate_and_rejects_self_approval() -> None:
    complete = {gate: True for gate in TASK_GATES}
    assert evaluate_task_completion(complete)["accepted"] is True
    incomplete = dict(complete)
    incomplete["supervisor_acceptance"] = False
    result = evaluate_task_completion(incomplete)
    assert result["accepted"] is False
    assert "supervisor_acceptance" in result["missing_gates"]
    with pytest.raises(CompletionContractError, match="self-approval"):
        evaluate_task_completion({**complete, "self_approved": True})


def test_goal_conjunction_rejects_task_count_and_stale_evidence() -> None:
    complete = {
        "observable_state": True,
        "semantic_properties": True,
        "accepted_children": True,
        "tests_proofs": True,
        "resolved_counterexamples": True,
        "resolved_gaps": True,
        "accepted_tree_root": True,
        "human_approval": True,
    }
    assert evaluate_goal_completion(complete)["accepted"] is True
    missing_child = dict(complete)
    missing_child["accepted_children"] = False
    assert evaluate_goal_completion(missing_child)["accepted"] is False
    with pytest.raises(CompletionContractError, match="task-count"):
        evaluate_goal_completion({**complete, "task_count_completion": True})
    with pytest.raises(CompletionContractError, match="stale"):
        evaluate_goal_completion({**complete, "stale_evidence": True})
    table = completion_truth_table()
    assert table[0]["self_approval"] is False
    assert "human_approval" in table[1]["conjunction"]
