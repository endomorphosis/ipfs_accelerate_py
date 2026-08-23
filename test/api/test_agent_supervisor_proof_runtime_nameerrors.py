"""Regressions for proof capabilities that must import and execute directly."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.proof.proof_directed_retrieval import (
    ABSOLUTE_MAX_BRANCH_FACTOR,
    ABSOLUTE_MAX_PREMISE_TOP_K,
    BOUNDED_EXPANSION_SCHEMA,
    DEFAULT_MAX_BRANCH_FACTOR,
    DEFAULT_PREMISE_TOP_K,
    PREMISE_RANKING_SCHEMA,
    RANKING_BASIS_POINTS,
    expand_bounded_branches,
    rank_proof_premises,
)
from ipfs_accelerate_py.agent_supervisor.proof.security_contract_analysis import (
    evaluate_fixed_point_security,
)


def test_proof_retrieval_closed_contracts_import_and_execute() -> None:
    assert (
        PREMISE_RANKING_SCHEMA
        == "ipfs_accelerate_py/agent-supervisor/proof-premise-ranking@1"
    )
    assert (
        BOUNDED_EXPANSION_SCHEMA
        == "ipfs_accelerate_py/agent-supervisor/proof-bounded-branch-expansion@1"
    )
    assert DEFAULT_PREMISE_TOP_K == 8
    assert ABSOLUTE_MAX_PREMISE_TOP_K == 64
    assert DEFAULT_MAX_BRANCH_FACTOR == 8
    assert ABSOLUTE_MAX_BRANCH_FACTOR == 32
    assert RANKING_BASIS_POINTS == 10_000

    ranking = rank_proof_premises(
        (
            {
                "premise_id": "premise:b",
                "score_millionths": 500_000,
                "predicted_cost_ms": 3,
            },
            {
                "premise_id": "premise:a",
                "score_millionths": 900_000,
                "predicted_cost_ms": 2,
            },
        ),
        k=1,
        relevant_ids=("premise:a",),
    )
    assert ranking.schema == PREMISE_RANKING_SCHEMA
    assert ranking.ranked_ids == ("premise:a",)
    assert ranking.recall_at_k_bps == RANKING_BASIS_POINTS
    assert ranking.proof_authority is False
    assert ranking.to_dict()["completion_authority"] is False

    expansion = expand_bounded_branches(
        {"root": ("a", "b"), "a": ("c",)},
        ("root",),
        max_branch_factor=1,
        max_depth=2,
    )
    assert expansion.schema == BOUNDED_EXPANSION_SCHEMA
    assert expansion.included_ids == ("root", "a", "c")
    assert expansion.omitted_ids == ("b",)
    assert expansion.truncated is True
    assert expansion.to_dict()["proof_authority"] is False


def test_fixed_point_security_derives_facts_from_mapping_effects() -> None:
    receipt = evaluate_fixed_point_security(
        candidate_tree_id="tree:fixed-point",
        intent_effects=("effect:read",),
        code_effects=({"effect_id": "effect:read"},),
        covered_effect_ids=("effect:read",),
        run_flow_analysis=False,
    )

    assert receipt.all_passed is True
    assert tuple(fact.effect_id for fact in receipt.code_facts) == ("effect:read",)
    assert receipt.forbidden.code_effect_ids == ("effect:read",)
    assert receipt.to_dict()["all_passed"] is True
