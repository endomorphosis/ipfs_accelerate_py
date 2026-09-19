"""SAWM-036 guarded program-world influence."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_guarded import (
    evaluate_guarded_program_world_influence,
)


def test_exact_and_proof_backed_may_influence_planning() -> None:
    exact = evaluate_guarded_program_world_influence(
        {"kind": "exact_hit", "current": True}
    )
    assert exact.influences_planning is True
    assert exact.completion_authority is False
    proof = evaluate_guarded_program_world_influence(
        {"kind": "proof_backed", "independent_proof": True}
    )
    assert proof.influences_planning is True


def test_neural_candidates_are_context_only() -> None:
    neural = evaluate_guarded_program_world_influence({"kind": "neural"})
    assert neural.allowed is True
    assert neural.influences_planning is False
    assert neural.neural_only_context is True
    unqualified = evaluate_guarded_program_world_influence(
        {"kind": "exact_hit", "shadow_unqualified": True}
    )
    assert unqualified.allowed is False
