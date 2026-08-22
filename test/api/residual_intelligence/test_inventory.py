from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    EvidenceAnswer,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.inventory import (
    ModelInvocationObservation,
    ResidualFamilyBoundary,
    ResidualReasoningInventory,
    TrajectoryOutcome,
)


def boundary() -> ResidualFamilyBoundary:
    return ResidualFamilyBoundary(
        task_family=ResidualTaskFamily.FAILURE_ATTRIBUTION,
        input_semantics="validated failure signature plus bounded dependency references",
        output_semantics="one failure class and one bounded action candidate",
        risk_class=RiskClass.R2,
        authority_class="candidate_only",
        validation_contract="failure-attribution-validator@1",
        error_behavior="invalid output or failed validation escalates",
        abstention_behavior="unknown signatures abstain",
    )


def observation(boundary_id: str, *, invocation_id: str = "invoke:1") -> ModelInvocationObservation:
    return ModelInvocationObservation(
        invocation_id=invocation_id,
        trajectory_id="trajectory:1",
        repository_state_cid="tree:1",
        stage="validation-recovery",
        task_family=ResidualTaskFamily.FAILURE_ATTRIBUTION,
        question_type="classify_failure",
        input_contract="failure-signature@1",
        output_contract="failure-attribution@1",
        context_size_bytes=512,
        provider="fixture-provider",
        model="fixture-model",
        input_tokens=64,
        output_tokens=8,
        latency_ms=5,
        cost_microunits=0,
        validation_references=("validator:1",),
        terminal_outcome=TrajectoryOutcome.ACCEPTED,
        deterministic_answer_possible=EvidenceAnswer.UNKNOWN,
        verified_procedure_answer_possible=EvidenceAnswer.NO,
        smaller_model_answer_possible=EvidenceAnswer.YES,
        affected_decision=EvidenceAnswer.YES,
        authoritative=False,
        task_risk=RiskClass.R2,
        family_boundary_id=boundary_id,
    )


def test_inventory_binds_every_observation_to_semantic_boundary() -> None:
    family = boundary()
    inventory = ResidualReasoningInventory(
        repository_revision="git:abc",
        environment_id="env:fixture",
        boundaries=(family,),
        observations=(observation(family.boundary_id),),
    )
    assert inventory.summary()["observation_count"] == 1
    assert inventory.summary()["authoritative_invocation_count"] == 0


def test_prompt_similarity_cannot_override_family_boundary() -> None:
    family = boundary()
    with pytest.raises(ResidualIntelligenceError, match="boundary"):
        ResidualReasoningInventory(
            repository_revision="git:abc",
            environment_id="env:fixture",
            boundaries=(family,),
            observations=(observation("boundary:embedding-only"),),
        )


def test_duplicate_invocation_is_rejected() -> None:
    family = boundary()
    item = observation(family.boundary_id)
    with pytest.raises(ResidualIntelligenceError, match="duplicate invocation"):
        ResidualReasoningInventory(
            repository_revision="git:abc",
            environment_id="env:fixture",
            boundaries=(family,),
            observations=(item, item),
        )
