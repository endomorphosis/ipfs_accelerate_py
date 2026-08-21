from __future__ import annotations

import copy

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    EpisodeKind,
    ExecutionTrajectory,
    HoleType,
    StepOperation,
    TraceEventStatus,
    TrajectoryOutcome,
    TrajectoryStep,
    TrajectoryTerminalStatus,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.trajectory import (
    TrajectoryContractError,
    parse_execution_trajectory,
    validate_execution_trajectory_contract,
)


def trajectory() -> ExecutionTrajectory:
    bindings = ArtifactBindings(
        "repo",
        "commit",
        "tree",
        "PCPC-G000",
        "PCPC-002",
        "contract-v1",
        "policy-v1",
        "env-v1",
    )
    steps = (
        TrajectoryStep(
            sequence=0,
            operation=StepOperation.REQUEST_TYPED_MODEL_HOLE,
            operation_contract="typed-hole-service@1",
            initial_state_cid="state-0",
            terminal_state_cid="state-1",
            observation_cids=("hole-observation",),
            effect_ids=("model-request",),
            validation_receipt_cids=("hole-validation",),
            hole_type=HoleType.CLASSIFY_FAILURE.value,
            model_calls=1,
            input_tokens=10,
            output_tokens=2,
            latency_ms=20,
            status=TraceEventStatus.SUCCEEDED,
        ),
        TrajectoryStep(
            sequence=1,
            operation=StepOperation.RUN_SELECTED_TESTS,
            operation_contract="test-runner@1",
            initial_state_cid="state-1",
            terminal_state_cid="state-2",
            observation_cids=("test-observation",),
            effect_ids=("validation",),
            validation_receipt_cids=("test-receipt",),
            latency_ms=30,
            status=TraceEventStatus.SUCCEEDED,
        ),
    )
    return ExecutionTrajectory(
        bindings=bindings,
        source_episode_cid="accepted-receipt",
        source_episode_kind=EpisodeKind.ACCEPTED_TASK_RECEIPT,
        initial_abstract_state_cid="state-0",
        terminal_abstract_state_cid="state-2",
        objective_criterion_ids=("criterion-a", "criterion-b"),
        task_family_hint="ERROR_BRANCH_COMPLETION",
        steps=steps,
        outcome=TrajectoryOutcome(
            status=TrajectoryTerminalStatus.ACCEPTED,
            accepted_criterion_ids=("criterion-a",),
            validation_receipt_cids=("hole-validation", "test-receipt"),
            proof_receipt_cids=(),
        ),
        total_cost_units=3,
        total_tokens=12,
        total_latency_ms=55,
        human_interventions=0,
    )


def test_trajectory_wire_parser_round_trip() -> None:
    value = trajectory()
    assert parse_execution_trajectory(value.to_dict()) == value
    assert parse_execution_trajectory(value.to_json()) == value
    assert parse_execution_trajectory(value.canonical_bytes()).content_id == value.content_id


def test_trajectory_rejects_discontinuous_state_chain() -> None:
    payload = copy.deepcopy(trajectory().to_dict())
    payload["steps"][1]["initial_state_cid"] = "different-state"
    with pytest.raises(TrajectoryContractError, match="discontinuous"):
        parse_execution_trajectory(payload)


def test_trajectory_preserves_token_and_validation_denominators() -> None:
    payload = copy.deepcopy(trajectory().to_dict())
    payload["total_tokens"] = 10
    with pytest.raises(TrajectoryContractError, match="denominator"):
        parse_execution_trajectory(payload)

    payload = copy.deepcopy(trajectory().to_dict())
    payload["outcome"]["validation_receipt_cids"] = ["test-receipt"]
    with pytest.raises(TrajectoryContractError, match="omits"):
        parse_execution_trajectory(payload)


def test_model_cost_requires_a_closed_typed_hole() -> None:
    payload = copy.deepcopy(trajectory().to_dict())
    payload["steps"][0]["hole_type"] = ""
    with pytest.raises(TrajectoryContractError, match="typed hole"):
        parse_execution_trajectory(payload)
    payload = copy.deepcopy(trajectory().to_dict())
    payload["steps"][0]["hole_type"] = "AUTHORITY_DECISION"
    with pytest.raises(TrajectoryContractError, match="unknown hole"):
        parse_execution_trajectory(payload)


def test_trajectory_p0_contract_helpers_remain_available() -> None:
    """P0 ships contract validation. G020 may add a normalizer in this module."""

    assert validate_execution_trajectory_contract(trajectory()) == trajectory()
    assert parse_execution_trajectory(trajectory().to_dict()) == trajectory()
