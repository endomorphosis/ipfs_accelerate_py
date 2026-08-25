from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    EpisodeKind,
    ExecutionTrajectory,
    HoleType,
    StepOperation,
    TraceEventStatus,
    TrajectoryNormalizationReceipt,
    TrajectoryOutcome,
    TrajectoryStep,
    TrajectoryTerminalStatus,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.trajectory import (
    SOURCE_EPISODE_SCHEMA,
    TRAJECTORY_NORMALIZER_REVISION,
    RemovedFieldClass,
    TrajectoryAdmissionError,
    TrajectoryAdmissionPolicy,
    TrajectoryAdmissionReason,
    TrajectoryContractError,
    TrajectoryNormalizer,
    normalize_source_episode,
    parse_execution_trajectory,
    validate_execution_trajectory_contract,
)


def bindings() -> ArtifactBindings:
    return ArtifactBindings(
        "repo",
        "commit",
        "tree",
        "PCPC-G000",
        "PCPC-002",
        "contract-v1",
        "policy-v1",
        "env-v1",
    )


def trajectory() -> ExecutionTrajectory:
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
        bindings=bindings(),
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


def policy(**changes: Any) -> TrajectoryAdmissionPolicy:
    values: dict[str, Any] = {"current_bindings": bindings(), "now_ms": 1_000}
    values.update(changes)
    return TrajectoryAdmissionPolicy(**values)


def _step(**changes: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sequence": 0,
        "operation": StepOperation.RUN_SELECTED_TESTS.value,
        "operation_contract": "test-runner@1",
        "initial_state_cid": "state-0",
        "terminal_state_cid": "state-1",
        "observation_cids": ["observation-0"],
        "effect_ids": ["validation"],
        "validation_receipt_cids": ["validation-0"],
        "latency_ms": 10,
        "status": TraceEventStatus.SUCCEEDED.value,
    }
    payload.update(changes)
    return payload


def _episode(**changes: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SOURCE_EPISODE_SCHEMA,
        "episode_cid": "accepted-receipt",
        "episode_kind": EpisodeKind.ACCEPTED_TASK_RECEIPT.value,
        "bindings": bindings().to_dict(),
        "signature": "sig-accepted-receipt",
        "signed": True,
        "unsigned": False,
        "current": True,
        "simulated": False,
        "pre_merge_only": False,
        "stale": False,
        "production_mode": "live",
        "issued_at_ms": 1_000,
        "expires_at_ms": 10_000,
        "initial_abstract_state_cid": "state-0",
        "terminal_abstract_state_cid": "state-1",
        "objective_criterion_ids": ["criterion-a"],
        "task_family_hint": "ERROR_BRANCH_COMPLETION",
        "accepted_criterion_ids": ["criterion-a"],
        "validation_receipt_cids": ["validation-0"],
        "steps": [_step()],
        "outcome_status": TrajectoryTerminalStatus.ACCEPTED.value,
        "total_cost_units": 1,
        "total_tokens": 0,
        "total_latency_ms": 10,
        "human_interventions": 0,
    }
    payload.update(changes)
    return payload


def episode_for(kind: EpisodeKind) -> dict[str, Any]:
    if kind is EpisodeKind.ACCEPTED_TASK_RECEIPT:
        return _episode(
            episode_cid="accepted-task",
            episode_kind=kind.value,
            signature="sig-accepted-task",
            terminal_abstract_state_cid="state-2",
            objective_criterion_ids=["criterion-a", "criterion-b"],
            accepted_criterion_ids=["criterion-a"],
            validation_receipt_cids=["hole-validation", "test-receipt"],
            steps=[
                _step(
                    sequence=0,
                    operation=StepOperation.REQUEST_TYPED_MODEL_HOLE.value,
                    operation_contract="typed-hole-service@1",
                    terminal_state_cid="state-1",
                    observation_cids=["hole-observation"],
                    effect_ids=["model-request"],
                    validation_receipt_cids=["hole-validation"],
                    hole_type=HoleType.CLASSIFY_FAILURE.value,
                    model_calls=1,
                    input_tokens=10,
                    output_tokens=2,
                    latency_ms=20,
                ),
                _step(
                    sequence=1,
                    operation=StepOperation.RUN_SELECTED_TESTS.value,
                    initial_state_cid="state-1",
                    terminal_state_cid="state-2",
                    observation_cids=["test-observation"],
                    effect_ids=["validation"],
                    validation_receipt_cids=["test-receipt"],
                    latency_ms=30,
                ),
            ],
            total_cost_units=3,
            total_tokens=12,
            total_latency_ms=55,
        )
    if kind is EpisodeKind.CURRENT_TREE_POST_MERGE_RECEIPT:
        return _episode(
            episode_cid="post-merge-receipt",
            episode_kind=kind.value,
            signature="sig-post-merge",
            terminal_abstract_state_cid="state-2",
            validation_receipt_cids=["merge-validation", "verify-validation"],
            steps=[
                _step(
                    sequence=0,
                    operation=StepOperation.MERGE_IN_ISOLATED_TRAIN.value,
                    operation_contract="merge-train@1",
                    terminal_state_cid="state-1",
                    observation_cids=["merge-observation"],
                    effect_ids=["merge"],
                    validation_receipt_cids=["merge-validation"],
                    latency_ms=15,
                ),
                _step(
                    sequence=1,
                    operation=StepOperation.VERIFY_MERGED_TREE.value,
                    operation_contract="tree-verifier@1",
                    initial_state_cid="state-1",
                    terminal_state_cid="state-2",
                    observation_cids=["verify-observation"],
                    effect_ids=["validation"],
                    validation_receipt_cids=["verify-validation"],
                    latency_ms=12,
                ),
            ],
            total_cost_units=2,
            total_latency_ms=27,
        )
    if kind is EpisodeKind.VERIFIED_PROOF_RECEIPT:
        return _episode(
            episode_cid="verified-proof",
            episode_kind=kind.value,
            signature="sig-verified-proof",
            validation_receipt_cids=["proof-validation"],
            proof_receipt_cids=["proof-receipt"],
            steps=[
                _step(
                    operation=StepOperation.RUN_PROOF.value,
                    operation_contract="proof-runner@1",
                    observation_cids=["proof-observation"],
                    effect_ids=["proof"],
                    validation_receipt_cids=["proof-validation"],
                    latency_ms=40,
                )
            ],
            total_cost_units=1,
            total_latency_ms=40,
        )
    if kind is EpisodeKind.ADMITTED_TEST_RECEIPT:
        return _episode(
            episode_cid="admitted-test",
            episode_kind=kind.value,
            signature="sig-admitted-test",
        )
    if kind is EpisodeKind.SUCCESSFUL_ROLLBACK_RECEIPT:
        return _episode(
            episode_cid="successful-rollback",
            episode_kind=kind.value,
            signature="sig-successful-rollback",
            accepted_criterion_ids=[],
            outcome_status=TrajectoryTerminalStatus.ROLLED_BACK.value,
            steps=[
                _step(
                    operation=StepOperation.ROLLBACK.value,
                    operation_contract="rollback-service@1",
                    observation_cids=["rollback-observation"],
                    effect_ids=["rollback"],
                    validation_receipt_cids=["rollback-validation"],
                    latency_ms=8,
                )
            ],
            validation_receipt_cids=["rollback-validation"],
            total_latency_ms=8,
        )
    if kind is EpisodeKind.AUTHORIZED_HUMAN_DECISION_RECEIPT:
        return _episode(
            episode_cid="human-decision",
            episode_kind=kind.value,
            signature="sig-human-decision",
            steps=[
                _step(
                    operation=StepOperation.ESCALATE.value,
                    operation_contract="human-review@1",
                    observation_cids=["human-observation"],
                    effect_ids=["escalation"],
                    validation_receipt_cids=["human-validation"],
                    human_interventions=1,
                    latency_ms=50,
                )
            ],
            validation_receipt_cids=["human-validation"],
            total_cost_units=1,
            total_latency_ms=50,
            human_interventions=1,
        )
    if kind is EpisodeKind.REJECTED_TASK_RECORD:
        return _episode(
            episode_cid="rejected-task",
            episode_kind=kind.value,
            signature="sig-rejected-task",
            accepted_criterion_ids=[],
            outcome_status=TrajectoryTerminalStatus.REJECTED.value,
            rejection_reason_code="scope_escape",
            steps=[
                _step(
                    operation=StepOperation.CHECK_POLICY.value,
                    operation_contract="policy-checker@1",
                    observation_cids=["policy-observation"],
                    effect_ids=["observe"],
                    validation_receipt_cids=["rejection-validation"],
                    status=TraceEventStatus.FAILED.value,
                    latency_ms=5,
                )
            ],
            validation_receipt_cids=["rejection-validation"],
            total_latency_ms=5,
        )
    return _episode(
        episode_cid="failed-recovered",
        episode_kind=EpisodeKind.FAILED_RECOVERED_EXECUTION.value,
        signature="sig-failed-recovered",
        accepted_criterion_ids=[],
        outcome_status=TrajectoryTerminalStatus.FAILED_RECOVERED.value,
        terminal_abstract_state_cid="state-2",
        validation_receipt_cids=["failure-validation", "recovery-validation"],
        steps=[
            _step(
                sequence=0,
                operation=StepOperation.APPLY_APPROVED_PATCH_TEMPLATE.value,
                operation_contract="patch-template@1",
                terminal_state_cid="state-1",
                observation_cids=["failure-observation"],
                effect_ids=["repository_write"],
                validation_receipt_cids=["failure-validation"],
                status=TraceEventStatus.FAILED.value,
                latency_ms=18,
            ),
            _step(
                sequence=1,
                operation=StepOperation.ROLLBACK.value,
                operation_contract="rollback-service@1",
                initial_state_cid="state-1",
                terminal_state_cid="state-2",
                observation_cids=["recovery-observation"],
                effect_ids=["rollback"],
                validation_receipt_cids=["recovery-validation"],
                status=TraceEventStatus.ROLLED_BACK.value,
                latency_ms=9,
            ),
        ],
        total_cost_units=2,
        total_latency_ms=27,
    )


def _assert_normalized_fields(result: Any) -> None:
    value = result.trajectory
    assert value.initial_abstract_state_cid == value.steps[0].initial_state_cid
    assert value.terminal_abstract_state_cid == value.steps[-1].terminal_state_cid
    assert tuple(step.sequence for step in value.steps) == tuple(range(len(value.steps)))
    assert all(step.operation_contract for step in value.steps)
    assert all(step.observation_cids for step in value.steps)
    assert all(step.effect_ids for step in value.steps)
    assert value.outcome.status in TrajectoryTerminalStatus
    assert value.outcome.validation_receipt_cids
    assert type(value.total_cost_units) is int
    assert type(value.total_tokens) is int
    assert type(value.total_latency_ms) is int
    assert type(value.human_interventions) is int
    assert value.total_tokens == sum(step.input_tokens + step.output_tokens for step in value.steps)
    assert value.total_latency_ms >= sum(step.latency_ms for step in value.steps)
    assert value.human_interventions == sum(step.human_interventions for step in value.steps)
    assert result.receipt.trajectory_cid == value.content_id
    assert result.receipt.source_episode_cid == value.source_episode_cid
    assert result.receipt.normalizer_revision == TRAJECTORY_NORMALIZER_REVISION
    assert result.receipt.admitted_evidence_cids
    public = json.dumps(value.to_dict())
    assert "private prompt text" not in public
    assert "chain of thought" not in public


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
    """P0 contract validation stays available after the G020 normalizer lands."""

    import ipfs_accelerate_py.agent_supervisor.procedure_compiler.trajectory as module

    assert not hasattr(module, "normalize_episode")
    assert module.TrajectoryNormalizer is TrajectoryNormalizer
    assert validate_execution_trajectory_contract(trajectory()) == trajectory()
    assert parse_execution_trajectory(trajectory().to_dict()) == trajectory()
    assert TrajectoryNormalizer(policy()).revision == TRAJECTORY_NORMALIZER_REVISION


@pytest.mark.parametrize("kind", list(EpisodeKind))
def test_every_admissible_source_category_normalizes_complete_fields(kind: EpisodeKind) -> None:
    result = TrajectoryNormalizer(policy()).normalize(episode_for(kind))
    assert result.trajectory.source_episode_kind is kind
    _assert_normalized_fields(result)
    assert parse_execution_trajectory(result.trajectory.to_dict()) == result.trajectory
    assert TrajectoryNormalizationReceipt.from_dict(result.receipt.to_dict()) == result.receipt


def test_normalizer_is_deterministic_for_the_same_admitted_episode() -> None:
    episode = episode_for(EpisodeKind.ACCEPTED_TASK_RECEIPT)
    first = normalize_source_episode(episode, policy=policy())
    second = normalize_source_episode(json.dumps(episode), policy=policy())
    assert first.trajectory == second.trajectory
    assert first.trajectory.content_id == second.trajectory.content_id
    assert first.receipt.trajectory_cid == second.receipt.trajectory_cid


def test_normalizer_completes_missing_cost_and_intervention_totals() -> None:
    episode = episode_for(EpisodeKind.AUTHORIZED_HUMAN_DECISION_RECEIPT)
    del episode["total_cost_units"]
    del episode["total_tokens"]
    del episode["total_latency_ms"]
    del episode["human_interventions"]
    result = TrajectoryNormalizer(policy()).normalize(episode)
    _assert_normalized_fields(result)
    assert result.trajectory.human_interventions == 1
    assert result.trajectory.total_latency_ms == 50
    assert result.trajectory.total_tokens == 0


def test_normalizer_redacts_prompts_secrets_bodies_logs_and_chain_of_thought() -> None:
    episode = episode_for(EpisodeKind.ACCEPTED_TASK_RECEIPT)
    episode["prompt"] = "private prompt text"
    episode["chain_of_thought"] = "hidden reasoning"
    episode["api_key"] = "secret-key"
    episode["credential"] = "user:pass"
    episode["source_body"] = "unbounded source"
    episode["unbounded_log"] = "log line " * 50
    episode["steps"][0]["model_transcript"] = "token stream"
    episode["steps"][0]["private_prompt"] = "hole prompt"
    result = TrajectoryNormalizer(policy()).normalize(episode)
    _assert_normalized_fields(result)
    assert set(result.receipt.removed_field_classes) == {
        RemovedFieldClass.PROMPT.value,
        RemovedFieldClass.CHAIN_OF_THOUGHT.value,
        RemovedFieldClass.SECRET.value,
        RemovedFieldClass.CREDENTIAL.value,
        RemovedFieldClass.BODY.value,
        RemovedFieldClass.LOG.value,
    }
    serialized = result.trajectory.to_json() + result.receipt.to_json()
    assert "private prompt text" not in serialized
    assert "hidden reasoning" not in serialized
    assert "secret-key" not in serialized
    assert "unbounded source" not in serialized


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"prose": "the task looks done"}, TrajectoryAdmissionReason.PROSE_EVIDENCE),
        ({"board_status": "done"}, TrajectoryAdmissionReason.BOARD_STATUS_EVIDENCE),
        ({"model_confidence": "high"}, TrajectoryAdmissionReason.MODEL_CONFIDENCE_EVIDENCE),
        ({"source_class": "prose"}, TrajectoryAdmissionReason.PROSE_EVIDENCE),
        ({"evidence_class": "board_status"}, TrajectoryAdmissionReason.BOARD_STATUS_EVIDENCE),
        ({"signature": "", "signed": False, "unsigned": True}, TrajectoryAdmissionReason.UNSIGNED_RECEIPT),
        ({"unsigned": True}, TrajectoryAdmissionReason.UNSIGNED_RECEIPT),
        ({"stale": True}, TrajectoryAdmissionReason.STALE_RECEIPT),
        ({"current": False}, TrajectoryAdmissionReason.STALE_RECEIPT),
        ({"simulated": True}, TrajectoryAdmissionReason.SIMULATED_PRODUCTION),
        ({"production_mode": "simulated"}, TrajectoryAdmissionReason.SIMULATED_PRODUCTION),
        ({"pre_merge_only": True}, TrajectoryAdmissionReason.PRE_MERGE_ONLY_VALIDATION),
        ({"issued_at_ms": 1}, TrajectoryAdmissionReason.STALE_RECEIPT),
    ],
)
def test_admission_rejects_non_demonstration_sources(
    changes: dict[str, Any],
    reason: TrajectoryAdmissionReason,
) -> None:
    episode = episode_for(EpisodeKind.ACCEPTED_TASK_RECEIPT)
    episode.update(changes)
    if reason is TrajectoryAdmissionReason.STALE_RECEIPT and "issued_at_ms" in changes:
        active = policy(now_ms=50_000, max_receipt_age_ms=10)
    else:
        active = policy()
    decision = active.decide(episode)
    assert decision.admitted is False
    assert decision.reason_code == reason.value
    with pytest.raises(TrajectoryAdmissionError) as rejected:
        TrajectoryNormalizer(active).normalize(episode)
    assert rejected.value.reason_code == reason.value


def test_admission_rejects_binding_mismatch_as_stale() -> None:
    episode = episode_for(EpisodeKind.CURRENT_TREE_POST_MERGE_RECEIPT)
    stale_bindings = ArtifactBindings(
        "repo",
        "other-commit",
        "other-tree",
        "PCPC-G000",
        "PCPC-002",
        "contract-v1",
        "policy-v1",
        "env-v1",
    )
    episode["bindings"] = stale_bindings.to_dict()
    with pytest.raises(TrajectoryAdmissionError) as rejected:
        TrajectoryNormalizer(policy()).normalize(episode)
    assert rejected.value.reason_code == TrajectoryAdmissionReason.STALE_RECEIPT.value


def test_admission_rejects_expired_and_future_receipts() -> None:
    expired = episode_for(EpisodeKind.ADMITTED_TEST_RECEIPT)
    expired["expires_at_ms"] = 500
    with pytest.raises(TrajectoryAdmissionError) as rejected:
        TrajectoryNormalizer(policy(now_ms=1_000)).normalize(expired)
    assert rejected.value.reason_code == TrajectoryAdmissionReason.STALE_RECEIPT.value

    future = episode_for(EpisodeKind.ADMITTED_TEST_RECEIPT)
    future["issued_at_ms"] = 5_000
    with pytest.raises(TrajectoryAdmissionError) as rejected_future:
        TrajectoryNormalizer(policy(now_ms=1_000)).normalize(future)
    assert rejected_future.value.reason_code == TrajectoryAdmissionReason.STALE_RECEIPT.value


def test_rejected_and_failed_sources_cannot_claim_accepted_success() -> None:
    rejected = episode_for(EpisodeKind.REJECTED_TASK_RECORD)
    rejected["outcome_status"] = TrajectoryTerminalStatus.ACCEPTED.value
    rejected["accepted_criterion_ids"] = ["criterion-a"]
    with pytest.raises(TrajectoryAdmissionError) as rejected_error:
        TrajectoryNormalizer(policy()).normalize(rejected)
    assert rejected_error.value.reason_code == TrajectoryAdmissionReason.KIND_SHAPE_MISMATCH.value

    recovered = episode_for(EpisodeKind.FAILED_RECOVERED_EXECUTION)
    recovered["outcome_status"] = TrajectoryTerminalStatus.ACCEPTED.value
    recovered["accepted_criterion_ids"] = ["criterion-a"]
    with pytest.raises(TrajectoryAdmissionError) as recovered_error:
        TrajectoryNormalizer(policy()).normalize(recovered)
    assert recovered_error.value.reason_code == TrajectoryAdmissionReason.KIND_SHAPE_MISMATCH.value


def test_inconsistent_declared_costs_fail_closed() -> None:
    episode = episode_for(EpisodeKind.ACCEPTED_TASK_RECEIPT)
    episode["total_tokens"] = 1
    with pytest.raises(TrajectoryAdmissionError) as tokens:
        TrajectoryNormalizer(policy()).normalize(episode)
    assert tokens.value.reason_code == TrajectoryAdmissionReason.INCONSISTENT_COST.value

    episode = episode_for(EpisodeKind.ACCEPTED_TASK_RECEIPT)
    episode["human_interventions"] = 4
    with pytest.raises(TrajectoryAdmissionError) as humans:
        TrajectoryNormalizer(policy()).normalize(episode)
    assert humans.value.reason_code == TrajectoryAdmissionReason.INCONSISTENT_COST.value


def test_steps_without_observations_or_effects_are_rejected() -> None:
    episode = episode_for(EpisodeKind.ADMITTED_TEST_RECEIPT)
    episode["steps"][0]["observation_cids"] = []
    with pytest.raises(TrajectoryAdmissionError) as observations:
        TrajectoryNormalizer(policy()).normalize(episode)
    assert observations.value.reason_code == TrajectoryAdmissionReason.MISSING_OBSERVATION.value

    episode = episode_for(EpisodeKind.ADMITTED_TEST_RECEIPT)
    episode["steps"][0]["effect_ids"] = []
    with pytest.raises(TrajectoryAdmissionError) as effects:
        TrajectoryNormalizer(policy()).normalize(episode)
    assert effects.value.reason_code == TrajectoryAdmissionReason.MISSING_EFFECT.value


def test_kind_requires_matching_operations_and_human_intervention() -> None:
    proof = episode_for(EpisodeKind.VERIFIED_PROOF_RECEIPT)
    proof["steps"][0]["operation"] = StepOperation.READ_STATE.value
    proof["steps"][0]["effect_ids"] = ["observe"]
    with pytest.raises(TrajectoryAdmissionError) as proof_error:
        TrajectoryNormalizer(policy()).normalize(proof)
    assert proof_error.value.reason_code == TrajectoryAdmissionReason.KIND_SHAPE_MISMATCH.value

    human = episode_for(EpisodeKind.AUTHORIZED_HUMAN_DECISION_RECEIPT)
    human["steps"][0]["human_interventions"] = 0
    human["human_interventions"] = 0
    with pytest.raises(TrajectoryAdmissionError) as human_error:
        TrajectoryNormalizer(policy()).normalize(human)
    assert human_error.value.reason_code == TrajectoryAdmissionReason.KIND_SHAPE_MISMATCH.value


def test_already_normalized_trajectory_is_not_a_source_episode() -> None:
    with pytest.raises(TrajectoryAdmissionError) as rejected:
        TrajectoryNormalizer(policy()).normalize(trajectory())
    assert rejected.value.reason_code == TrajectoryAdmissionReason.MALFORMED_EPISODE.value


def test_normalization_receipt_binds_current_identities() -> None:
    result = TrajectoryNormalizer(policy(now_ms=1_000)).normalize(
        episode_for(EpisodeKind.VERIFIED_PROOF_RECEIPT),
        emitted_at_ms=1_500,
    )
    assert result.receipt.emitted_at_ms == 1_500
    assert result.receipt.bindings == bindings()
    assert "proof-receipt" in result.receipt.admitted_evidence_cids
    assert result.trajectory.outcome.proof_receipt_cids == ("proof-receipt",)
    assert result.trajectory.steps[0].operation is StepOperation.RUN_PROOF
