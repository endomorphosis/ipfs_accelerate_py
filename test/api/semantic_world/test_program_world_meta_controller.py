"""SAWM-032 Autonomous Meta-Controller cascade."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.planning.program_world_meta_controller import (
    CognitiveAction,
    explain_program_world_escalation,
    select_program_world_cognitive_action,
)


def test_exact_reuse_precedes_models() -> None:
    receipt = select_program_world_cognitive_action(
        {
            "residual_question": "can this call be reused?",
            "available_actions": ["exact_reuse", "local_model", "remote_model"],
            "model_calls_remaining": 3,
            "remaining_validation_reserve": 0.3,
        }
    )
    assert receipt.action is CognitiveAction.EXACT_REUSE
    assert receipt.proposal_only is True
    assert receipt.completion_authority is False


def test_models_cannot_spend_empty_validation_reserve() -> None:
    receipt = select_program_world_cognitive_action(
        {
            "exhausted_actions": ["exact_reuse", "proof_receipt", "procedure", "static_symbolic", "retrieval", "specialist"],
            "model_calls_remaining": 2,
            "remaining_validation_reserve": 0,
            "residual_question": "need a model?",
        }
    )
    assert receipt.action is CognitiveAction.HUMAN_REVIEW
    explanation = explain_program_world_escalation(
        {"exhausted_actions": ["exact_reuse"], "residual_question": "next?"}
    )
    assert explanation["cascade"][0] == "exact_reuse"
    assert explanation["completion_authority"] is False
