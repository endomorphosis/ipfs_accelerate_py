from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.goal_completion import (
    GoalState,
    contradictions_from_proof_invalidation,
    reconcile_goal_reopenings,
)
from ipfs_accelerate_py.agent_supervisor.objective_graph import ObjectiveGoal
from ipfs_accelerate_py.agent_supervisor.objective_task_janitor import (
    reconcile_objective_task_strategy,
)
from ipfs_accelerate_py.agent_supervisor.proof_scope_index import (
    ProofInvalidationEvent,
    ProofInvalidationResult,
    build_proof_scope_index,
    invalidate_proof_evidence,
)


SOURCE_TREE = "tree:semantic-change"
CHANGED_INPUT = {"kind": "qualified_symbol", "value": "pkg.api.Service.run"}
DEPENDENCY_EDGE = {
    "kind": "proof_dependency",
    "source": "obligation:consumer",
    "target": "obligation:api",
    "metadata": {"required": True},
}
CONFLICT_EDGE = {
    "kind": "semantic_conflict",
    "source": "obligation:api",
    "target": "qualified_symbol:pkg.api.Service.run",
    "metadata": {"scope": "src/api.py"},
}


def _index():
    return build_proof_scope_index(
        scope_blobs=[
            {
                "path": "src/api.py",
                "blob_id": "blob:api",
                "scopes": [
                    {
                        "scope_id": "scope:api",
                        "path": "src/api.py",
                        "qualified_symbol": "pkg.api.Service.run",
                    }
                ],
            },
            {
                "path": "src/consumer.py",
                "blob_id": "blob:consumer",
                "scopes": [
                    {
                        "scope_id": "scope:consumer",
                        "path": "src/consumer.py",
                        "qualified_symbol": "pkg.consumer.consume",
                    }
                ],
            },
            {
                "path": "src/unrelated.py",
                "blob_id": "blob:unrelated",
                "scopes": [
                    {
                        "scope_id": "scope:unrelated",
                        "path": "src/unrelated.py",
                        "qualified_symbol": "pkg.unrelated.stable",
                    }
                ],
            },
        ],
        obligations=[
            {
                "obligation_id": "obligation:api",
                "ast_scope_ids": ["scope:api"],
                "premise_ids": ["premise:lease"],
                "contradiction_ids": ["contradiction:counterexample"],
            },
            {
                "obligation_id": "obligation:consumer",
                "ast_scope_ids": ["scope:consumer"],
                "depends_on": ["obligation:api"],
            },
            {
                "obligation_id": "obligation:unrelated",
                "ast_scope_ids": ["scope:unrelated"],
            },
        ],
        receipts=[
            {
                "receipt_id": "receipt:api",
                "obligation_id": "obligation:api",
                "ast_scope_ids": ["scope:api"],
                "repository_tree_id": "tree:proved",
            },
            {
                "receipt_id": "receipt:consumer",
                "obligation_id": "obligation:consumer",
                "ast_scope_ids": ["scope:consumer"],
                "repository_tree_id": "tree:proved",
            },
            {
                "receipt_id": "receipt:unrelated",
                "obligation_id": "obligation:unrelated",
                "ast_scope_ids": ["scope:unrelated"],
                "repository_tree_id": "tree:proved",
            },
        ],
    )


def _completion_evidence():
    return {
        "G1": [
            {
                "criterion_id": "criterion:api",
                "acceptance_criterion": "The API invariant is proved.",
                "obligation_id": "obligation:api",
                "proof_receipt_id": "receipt:api",
                "repository_tree": "tree:proved",
                "producing_task_or_scan": "REF-API",
            }
        ],
        "G2": [
            {
                "criterion_id": "criterion:consumer",
                "acceptance_criterion": "The consumer preserves the API invariant.",
                "obligation_id": "obligation:consumer",
                "proof_receipt_id": "receipt:consumer",
                "repository_tree": "tree:proved",
                "producing_task_or_scan": "REF-CONSUMER",
            }
        ],
        "G3": [
            {
                "criterion_id": "criterion:unrelated",
                "acceptance_criterion": "The unrelated surface remains valid.",
                "obligation_id": "obligation:unrelated",
                "proof_receipt_id": "receipt:unrelated",
                "repository_tree": "tree:proved",
                "producing_task_or_scan": "REF-UNRELATED",
            }
        ],
    }


def _goals():
    return [
        ObjectiveGoal(
            "G1",
            "Verified API",
            {
                "status": "verified_complete",
                "acceptance": "The API invariant is proved.",
            },
        ),
        ObjectiveGoal(
            "G2",
            "Provisional consumer",
            {
                "status": "provisionally_complete",
                "acceptance": "The consumer preserves the API invariant.",
            },
        ),
        ObjectiveGoal(
            "G3",
            "Unrelated verified goal",
            {
                "status": "verified_complete",
                "acceptance": "The unrelated surface remains valid.",
            },
        ),
    ]


def _invalidate(index=None):
    return invalidate_proof_evidence(
        index or _index(),
        [CHANGED_INPUT],
        source_tree=SOURCE_TREE,
        completion_evidence_by_goal=_completion_evidence(),
        dependency_edges=[DEPENDENCY_EDGE],
        conflict_edges=[CONFLICT_EDGE],
    )


def test_transitive_invalidation_records_complete_semantic_impact() -> None:
    result = _invalidate()
    event = result.event

    assert event.changed_inputs[0].to_dict() == CHANGED_INPUT
    assert event.affected_obligation_ids == (
        "obligation:api",
        "obligation:consumer",
    )
    assert event.affected_receipt_ids == ("receipt:api", "receipt:consumer")
    assert event.affected_criterion_ids == (
        "criterion:api",
        "criterion:consumer",
    )
    assert event.affected_goal_ids == ("G1", "G2")
    assert event.source_tree == SOURCE_TREE
    assert {
        record.subject_id for record in event.invalidation_records
    } == {
        "obligation:api",
        "obligation:consumer",
        "receipt:api",
        "receipt:consumer",
    }

    assert result.index.active_obligation_ids == ("obligation:unrelated",)
    assert result.index.active_receipt_ids == ("receipt:unrelated",)
    # Invalidated proof receipts are retained in both the append-only index
    # and the semantic-change event for audit.
    assert {receipt.receipt_id for receipt in result.index.receipts} == {
        "receipt:api",
        "receipt:consumer",
        "receipt:unrelated",
    }
    assert {receipt.receipt_id for receipt in event.historical_receipts} == {
        "receipt:api",
        "receipt:consumer",
    }

    assert ProofInvalidationEvent.from_json(event.to_json()) == event
    assert ProofInvalidationResult.from_json(result.to_json()) == result


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        ("premise", "premise:lease"),
        ("contradiction", "contradiction:counterexample"),
        ("symbol", "pkg.api.Service.run"),
    ],
)
def test_each_semantic_change_kind_can_seed_transitive_invalidation(
    kind: str,
    value: str,
) -> None:
    result = invalidate_proof_evidence(
        _index(),
        [{"kind": kind, "value": value}],
        source_tree=SOURCE_TREE,
    )

    assert result.event.affected_obligation_ids == (
        "obligation:api",
        "obligation:consumer",
    )
    assert result.event.affected_receipt_ids == ("receipt:api", "receipt:consumer")


def test_affected_completed_goals_reopen_without_churning_unrelated_goal() -> None:
    result = _invalidate()
    contradictions = contradictions_from_proof_invalidation(result)

    assert [item.goal_id for item in contradictions] == ["G1", "G2"]
    assert all(item.invalidation_event_id == result.event.event_id for item in contradictions)
    assert all(item.source_tree == SOURCE_TREE for item in contradictions)

    decisions = reconcile_goal_reopenings(
        _goals(),
        contradictions,
        historical_completion_receipts=[
            {"goal_id": "G1", "receipt_id": "completion:G1"},
            {"goal_id": "G2", "receipt_id": "completion:G2"},
            {"goal_id": "G3", "receipt_id": "completion:G3"},
        ],
        now="2026-07-23T00:00:00+00:00",
    )

    assert decisions["G1"].state is GoalState.REOPENED
    assert decisions["G2"].state is GoalState.REOPENED
    assert "G3" not in decisions
    assert decisions["G1"].historical_completion_receipts == (
        {"goal_id": "G1", "receipt_id": "completion:G1"},
    )
    assert decisions["G2"].historical_completion_receipts == (
        {"goal_id": "G2", "receipt_id": "completion:G2"},
    )


def test_replay_is_idempotent_and_replacement_edges_remain_auditable() -> None:
    first = _invalidate()
    replayed = _invalidate(first.index)

    assert replayed.index == first.index
    assert replayed.event == first.event
    assert replayed.event.event_id == first.event.event_id

    contradictions = contradictions_from_proof_invalidation(first)
    replayed_contradictions = contradictions_from_proof_invalidation(
        replayed,
        detected_at="2026-07-23T01:00:00+00:00",
    )
    assert [item.contradiction_id for item in replayed_contradictions] == [
        item.contradiction_id for item in contradictions
    ]

    completion_history = {
        "G1": {
            "receipt_id": "completion:G1",
            "state": "verified_complete",
            "verified": True,
        },
        "G2": {
            "receipt_id": "completion:G2",
            "state": "provisionally_complete",
            "verified": False,
        },
        "G3": {
            "receipt_id": "completion:G3",
            "state": "verified_complete",
            "verified": True,
        },
    }
    first_run = reconcile_objective_task_strategy(
        goals=_goals(),
        tasks=[],
        strategy={"objective_completion_decisions": completion_history},
        contradictions=contradictions,
        now="2026-07-23T00:00:00+00:00",
    )
    second_run = reconcile_objective_task_strategy(
        goals=_goals(),
        tasks=[],
        strategy=first_run["strategy"],
        contradictions=replayed_contradictions,
        now="2026-07-23T01:00:00+00:00",
    )

    assert first_run["contradiction_reopened_goal_ids"] == ["G1", "G2"]
    assert second_run["contradiction_reopened_goal_ids"] == []
    assert second_run["effective_reopened_goal_ids"] == ["G1", "G2"]
    assert second_run["replacement_tasks"] == []
    assert second_run["persisted_replacement_tasks"] == first_run[
        "persisted_replacement_tasks"
    ]
    assert len(first_run["goal_reopening_receipts"]) == 2
    assert second_run["goal_reopening_receipts"] == first_run[
        "goal_reopening_receipts"
    ]
    assert second_run["strategy"]["objective_completion_decisions"] == completion_history

    replacement_tasks = first_run["persisted_replacement_tasks"]
    assert {task["goal_id"] for task in replacement_tasks} == {"G1", "G2"}
    for task in replacement_tasks:
        assert {
            "edge_kind": "proof_dependency",
            "source_id": "obligation:consumer",
            "target_id": "obligation:api",
            "metadata": {"required": True},
        } in task["dependency_edges"]
        assert {
            "edge_kind": "semantic_conflict",
            "source_id": "obligation:api",
            "target_id": "qualified_symbol:pkg.api.Service.run",
            "metadata": {"scope": "src/api.py"},
        } in task["conflict_edges"]
        assert task["changed_inputs"] == [CHANGED_INPUT]
        assert task["source_tree"] == SOURCE_TREE
        assert set(task["affected_obligation_ids"]) == {
            "obligation:api",
            "obligation:consumer",
        }
        assert set(task["affected_receipt_ids"]) == {
            "receipt:api",
            "receipt:consumer",
        }
