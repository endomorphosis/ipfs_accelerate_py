from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping, Sequence

logging.disable(logging.WARNING)

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    ArtifactState,
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
    ADMITTED_EVIDENCE_CLASS,
    NORMALIZER_REVISION,
    SOURCE_EPISODE_SCHEMA,
    RedactedFieldClass,
    TrajectoryAdmissionError,
    TrajectoryAdmissionPolicy,
    TrajectoryAdmissionReason,
    TrajectoryContractError,
    TrajectoryNormalizer,
    normalize_trajectory,
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
    """P0 contract helpers stay available after G020 normalizer admission."""

    assert validate_execution_trajectory_contract(trajectory()) == trajectory()
    assert parse_execution_trajectory(trajectory().to_dict()) == trajectory()


def _bindings() -> ArtifactBindings:
    return ArtifactBindings(
        repository_id="repo",
        repository_commit="commit",
        tree_id="tree",
        objective_id="PCPC-G000",
        task_id="PCPC-009",
        contract_revision="contract-v1",
        policy_revision="policy-v1",
        environment_id="env-v1",
    )


def _step(
    sequence: int,
    operation: StepOperation,
    *,
    contract: str,
    initial: str,
    terminal: str,
    observations: tuple[str, ...] = (),
    effects: tuple[str, ...] = (),
    validation: tuple[str, ...] = (),
    hole_type: str = "",
    model_calls: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: int = 0,
    human_interventions: int = 0,
    status: TraceEventStatus = TraceEventStatus.SUCCEEDED,
) -> dict[str, object]:
    return {
        "sequence": sequence,
        "operation": operation.value,
        "operation_contract": contract,
        "initial_state_cid": initial,
        "terminal_state_cid": terminal,
        "observation_cids": list(observations),
        "effect_ids": list(effects),
        "validation_receipt_cids": list(validation),
        "hole_type": hole_type,
        "model_calls": model_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "human_interventions": human_interventions,
        "status": status.value,
    }


def _as_episode_kind(kind: EpisodeKind | str) -> EpisodeKind:
    if isinstance(kind, EpisodeKind):
        return kind
    text = str(kind)
    try:
        return EpisodeKind(text)
    except ValueError:
        name = text.split(".")[-1].upper()
        try:
            return EpisodeKind[name]
        except KeyError as exc:
            raise ValueError(f"unsupported episode kind {kind!r}") from exc


def _kind_recipe(kind: EpisodeKind | str) -> dict[str, object]:
    kind = _as_episode_kind(kind)
    if kind == EpisodeKind.ACCEPTED_TASK_RECEIPT:
        steps = [
            _step(
                0,
                StepOperation.REQUEST_TYPED_MODEL_HOLE,
                contract="typed-hole-service@1",
                initial="state-0",
                terminal="state-1",
                observations=("hole-observation",),
                effects=("model-request",),
                validation=("hole-validation",),
                hole_type=HoleType.CLASSIFY_FAILURE.value,
                model_calls=1,
                input_tokens=10,
                output_tokens=2,
                latency_ms=20,
            ),
            _step(
                1,
                StepOperation.RUN_SELECTED_TESTS,
                contract="test-runner@1",
                initial="state-1",
                terminal="state-2",
                observations=("test-observation",),
                effects=("validation",),
                validation=("test-receipt",),
                latency_ms=30,
            ),
        ]
        outcome: dict[str, object] = {
            "status": TrajectoryTerminalStatus.ACCEPTED.value,
            "accepted_criterion_ids": ["criterion-a"],
            "validation_receipt_cids": ["hole-validation", "test-receipt"],
            "proof_receipt_cids": [],
            "rejection_reason_code": "",
        }
        terminal = "state-2"
        extra: dict[str, object] = {"total_cost_units": 3, "total_latency_ms": 55}
    elif kind == EpisodeKind.CURRENT_TREE_POST_MERGE_RECEIPT:
        steps = [
            _step(
                0,
                StepOperation.MERGE_IN_ISOLATED_TRAIN,
                contract="merge-train@1",
                initial="state-0",
                terminal="state-1",
                observations=("merge-observation",),
                effects=("merge",),
                validation=("merge-validation",),
                latency_ms=40,
            ),
            _step(
                1,
                StepOperation.VERIFY_MERGED_TREE,
                contract="tree-verifier@1",
                initial="state-1",
                terminal="state-2",
                observations=("tree-observation",),
                effects=("validation",),
                validation=("post-merge-receipt",),
                latency_ms=15,
            ),
        ]
        outcome = {
            "status": TrajectoryTerminalStatus.ACCEPTED.value,
            "accepted_criterion_ids": ["criterion-a"],
            "validation_receipt_cids": ["merge-validation", "post-merge-receipt"],
            "proof_receipt_cids": [],
            "rejection_reason_code": "",
        }
        terminal = "state-2"
        extra = {"post_merge": True, "total_cost_units": 4}
    elif kind == EpisodeKind.VERIFIED_PROOF_RECEIPT:
        steps = [
            _step(
                0,
                StepOperation.RUN_PROOF,
                contract="proof-runner@1",
                initial="state-0",
                terminal="state-1",
                observations=("proof-observation",),
                effects=("proof",),
                validation=("proof-validation",),
                latency_ms=80,
            )
        ]
        outcome = {
            "status": TrajectoryTerminalStatus.ACCEPTED.value,
            "accepted_criterion_ids": ["criterion-a"],
            "validation_receipt_cids": ["proof-validation"],
            "proof_receipt_cids": ["verified-proof"],
            "rejection_reason_code": "",
        }
        terminal = "state-1"
        extra = {"proof_receipt_cids": ["verified-proof"], "total_cost_units": 5}
    elif kind == EpisodeKind.ADMITTED_TEST_RECEIPT:
        steps = [
            _step(
                0,
                StepOperation.RUN_SELECTED_TESTS,
                contract="test-runner@1",
                initial="state-0",
                terminal="state-1",
                observations=("test-observation",),
                effects=("validation",),
                validation=("admitted-test",),
                latency_ms=25,
            )
        ]
        outcome = {
            "status": TrajectoryTerminalStatus.ACCEPTED.value,
            "accepted_criterion_ids": ["criterion-a"],
            "validation_receipt_cids": ["admitted-test"],
            "proof_receipt_cids": [],
            "rejection_reason_code": "",
        }
        terminal = "state-1"
        extra = {"total_cost_units": 2}
    elif kind == EpisodeKind.SUCCESSFUL_ROLLBACK_RECEIPT:
        steps = [
            _step(
                0,
                StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,
                contract="patch-template@1",
                initial="state-0",
                terminal="state-1",
                observations=("patch-observation",),
                effects=("repository-write",),
                validation=("patch-validation",),
                latency_ms=10,
                status=TraceEventStatus.FAILED,
            ),
            _step(
                1,
                StepOperation.ROLLBACK,
                contract="rollback-service@1",
                initial="state-1",
                terminal="state-0b",
                observations=("rollback-observation",),
                effects=("rollback",),
                validation=("rollback-receipt",),
                latency_ms=12,
            ),
        ]
        outcome = {
            "status": TrajectoryTerminalStatus.ROLLED_BACK.value,
            "accepted_criterion_ids": [],
            "validation_receipt_cids": ["patch-validation", "rollback-receipt"],
            "proof_receipt_cids": [],
            "rejection_reason_code": "",
        }
        terminal = "state-0b"
        extra = {"total_cost_units": 2}
    elif kind == EpisodeKind.AUTHORIZED_HUMAN_DECISION_RECEIPT:
        steps = [
            _step(
                0,
                StepOperation.CHECK_AUTHORITY,
                contract="authority-gate@1",
                initial="state-0",
                terminal="state-1",
                observations=("authority-observation",),
                effects=("escalation",),
                validation=("human-decision",),
                latency_ms=5,
                human_interventions=1,
            )
        ]
        outcome = {
            "status": TrajectoryTerminalStatus.ACCEPTED.value,
            "accepted_criterion_ids": ["criterion-a"],
            "validation_receipt_cids": ["human-decision"],
            "proof_receipt_cids": [],
            "rejection_reason_code": "",
        }
        terminal = "state-1"
        extra = {"total_cost_units": 1}
    elif kind == EpisodeKind.REJECTED_TASK_RECORD:
        steps = [
            _step(
                0,
                StepOperation.CHECK_SCOPE,
                contract="scope-gate@1",
                initial="state-0",
                terminal="state-1",
                observations=("scope-observation",),
                effects=("observe",),
                validation=("rejection-validation",),
                latency_ms=8,
                status=TraceEventStatus.FAILED,
            )
        ]
        outcome = {
            "status": TrajectoryTerminalStatus.REJECTED.value,
            "accepted_criterion_ids": [],
            "validation_receipt_cids": ["rejection-validation"],
            "proof_receipt_cids": [],
            "rejection_reason_code": "scope_escape",
        }
        terminal = "state-1"
        extra = {"total_cost_units": 1}
    elif kind != EpisodeKind.FAILED_RECOVERED_EXECUTION:
        raise AssertionError(f"untested episode kind {kind}")
    else:
        steps = [
            _step(
                0,
                StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,
                contract="patch-template@1",
                initial="state-0",
                terminal="state-1",
                observations=("failure-observation",),
                effects=("repository-write",),
                validation=("failure-validation",),
                latency_ms=9,
                status=TraceEventStatus.FAILED,
            ),
            _step(
                1,
                StepOperation.ROLLBACK,
                contract="rollback-service@1",
                initial="state-1",
                terminal="state-2",
                observations=("recovery-observation",),
                effects=("rollback",),
                validation=("recovery-receipt",),
                latency_ms=11,
            ),
        ]
        outcome = {
            "status": TrajectoryTerminalStatus.FAILED_RECOVERED.value,
            "accepted_criterion_ids": [],
            "validation_receipt_cids": ["failure-validation", "recovery-receipt"],
            "proof_receipt_cids": [],
            "rejection_reason_code": "",
        }
        terminal = "state-2"
        extra = {"total_cost_units": 2}
    episode: dict[str, object] = {
        "schema": SOURCE_EPISODE_SCHEMA,
        "episode_cid": f"{kind.value}-episode",
        "episode_kind": kind.value,
        "bindings": _bindings().to_dict(),
        "signed": True,
        "signature_cid": f"{kind.value}-signature",
        "current": True,
        "simulated": False,
        "pre_merge_only": False,
        "evidence_class": ADMITTED_EVIDENCE_CLASS,
        "initial_abstract_state_cid": "state-0",
        "terminal_abstract_state_cid": terminal,
        "objective_criterion_ids": ["criterion-a", "criterion-b"],
        "task_family_hint": "ERROR_BRANCH_COMPLETION",
        "steps": steps,
        "outcome": outcome,
        "admitted_evidence_cids": [f"{kind.value}-admission"],
        "emitted_at_ms": 1000,
    }
    episode.update(extra)
    return episode


def _mapping_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_mapping_keys(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            keys.update(_mapping_keys(item))
    return keys


_ADMISSIBLE_SOURCE_KINDS = tuple(EpisodeKind)


@pytest.mark.parametrize(
    "kind",
    _ADMISSIBLE_SOURCE_KINDS,
    ids=[f"source-category-{index:02d}" for index, _ in enumerate(_ADMISSIBLE_SOURCE_KINDS)],
)
def test_every_admissible_source_category_normalizes_complete_fields(
    kind: EpisodeKind,
) -> None:
    kind = _as_episode_kind(kind)
    policy = TrajectoryAdmissionPolicy(current_tree_id="tree", current_repository_commit="commit")
    normalizer = TrajectoryNormalizer(policy, emitted_at_ms=1000)
    result = normalizer.normalize(_kind_recipe(kind))
    trajectory = result.trajectory
    receipt = result.receipt
    assert result.artifact_state == ArtifactState.CANDIDATE
    assert trajectory.source_episode_kind == kind
    assert trajectory.initial_abstract_state_cid == "state-0"
    assert trajectory.terminal_abstract_state_cid == trajectory.steps[-1].terminal_state_cid
    assert tuple(step.sequence for step in trajectory.steps) == tuple(range(len(trajectory.steps)))
    assert all(step.operation_contract for step in trajectory.steps)
    assert any(step.observation_cids for step in trajectory.steps)
    assert any(step.effect_ids for step in trajectory.steps)
    assert any(step.validation_receipt_cids for step in trajectory.steps)
    assert all(isinstance(step.hole_type, str) for step in trajectory.steps)
    assert trajectory.outcome.status is not None
    assert all(step.initial_state_cid and step.terminal_state_cid for step in trajectory.steps)
    assert all(
        (not step.hole_type) or step.hole_type in {item.value for item in HoleType}
        for step in trajectory.steps
    )
    assert trajectory.total_tokens == sum(
        step.input_tokens + step.output_tokens for step in trajectory.steps
    )
    assert trajectory.total_latency_ms >= sum(step.latency_ms for step in trajectory.steps)
    assert trajectory.human_interventions == sum(
        step.human_interventions for step in trajectory.steps
    )
    assert type(trajectory.total_cost_units) is int
    assert type(trajectory.total_tokens) is int
    assert type(trajectory.total_latency_ms) is int
    assert type(trajectory.human_interventions) is int
    assert receipt.trajectory_cid == trajectory.content_id
    assert receipt.source_episode_cid == f"{kind.value}-episode"
    assert receipt.normalizer_revision == NORMALIZER_REVISION
    assert receipt.admitted_evidence_cids
    assert normalizer.get(trajectory.content_id) == trajectory
    assert normalizer.get(receipt.content_id) == receipt
    assert TrajectoryNormalizationReceipt.from_dict(receipt.to_dict()) == receipt
    validate_execution_trajectory_contract(trajectory)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            {"evidence_class": "prose", "prose": "the agent succeeded"},
            TrajectoryAdmissionReason.PROSE_EPISODE,
        ),
        ({"board_status": "done"}, TrajectoryAdmissionReason.BOARD_STATUS),
        ({"model_confidence": 1}, TrajectoryAdmissionReason.MODEL_CONFIDENCE),
        ({"simulated": True}, TrajectoryAdmissionReason.SIMULATED_PRODUCTION),
        ({"production_mode": "simulated"}, TrajectoryAdmissionReason.SIMULATED_PRODUCTION),
        ({"pre_merge_only": True}, TrajectoryAdmissionReason.PRE_MERGE_ONLY),
        ({"current": False}, TrajectoryAdmissionReason.STALE_EPISODE),
        ({"stale": True}, TrajectoryAdmissionReason.STALE_EPISODE),
        ({"signed": False}, TrajectoryAdmissionReason.UNSIGNED_EPISODE),
        ({"unsigned": True}, TrajectoryAdmissionReason.UNSIGNED_EPISODE),
        ({"signature_cid": ""}, TrajectoryAdmissionReason.UNSIGNED_EPISODE),
    ],
)
def test_admission_rejects_untrusted_source_categories(
    mutation: dict[str, object], reason: TrajectoryAdmissionReason
) -> None:
    episode = _kind_recipe(EpisodeKind.ACCEPTED_TASK_RECEIPT)
    episode.update(mutation)
    with pytest.raises(TrajectoryAdmissionError) as raised:
        TrajectoryNormalizer().normalize(episode)
    assert raised.value.reason_code == reason.value


def test_stale_tree_or_missing_current_declaration_is_rejected() -> None:
    policy = TrajectoryAdmissionPolicy(current_tree_id="other-tree")
    with pytest.raises(TrajectoryAdmissionError) as raised:
        TrajectoryNormalizer(policy).normalize(_kind_recipe(EpisodeKind.ADMITTED_TEST_RECEIPT))
    assert raised.value.reason_code == TrajectoryAdmissionReason.STALE_EPISODE.value

    episode = _kind_recipe(EpisodeKind.ADMITTED_TEST_RECEIPT)
    del episode["current"]
    with pytest.raises(TrajectoryAdmissionError) as raised:
        TrajectoryNormalizer().normalize(episode)
    assert raised.value.reason_code == TrajectoryAdmissionReason.STALE_EPISODE.value


def test_rejected_and_recovered_kinds_cannot_demonstrate_accepted_success() -> None:
    rejected = _kind_recipe(EpisodeKind.REJECTED_TASK_RECORD)
    rejected["outcome"]["status"] = TrajectoryTerminalStatus.ACCEPTED.value
    rejected["outcome"]["accepted_criterion_ids"] = ["criterion-a"]
    rejected["outcome"]["rejection_reason_code"] = ""
    with pytest.raises(TrajectoryAdmissionError) as raised:
        TrajectoryNormalizer().normalize(rejected)
    assert raised.value.reason_code == TrajectoryAdmissionReason.SUCCESS_KIND_MISMATCH.value

    recovered = _kind_recipe(EpisodeKind.FAILED_RECOVERED_EXECUTION)
    recovered["outcome"]["status"] = TrajectoryTerminalStatus.ACCEPTED.value
    recovered["outcome"]["accepted_criterion_ids"] = ["criterion-a"]
    with pytest.raises(TrajectoryAdmissionError) as raised:
        TrajectoryNormalizer().normalize(recovered)
    assert raised.value.reason_code == TrajectoryAdmissionReason.SUCCESS_KIND_MISMATCH.value


def test_normalizer_redacts_private_and_unbounded_fields() -> None:
    episode = _kind_recipe(EpisodeKind.ACCEPTED_TASK_RECEIPT)
    episode["prompt"] = "secret system prompt"
    episode["chain_of_thought"] = "stepwise private reasoning"
    episode["api_key"] = "not-a-real-key"
    episode["credentials"] = {"password": "hidden"}
    episode["source_body"] = "unbounded file body"
    episode["unbounded_logs"] = ["line-1", "line-2"]
    steps = episode["steps"]
    assert isinstance(steps, list)
    first_step = steps[0]
    assert isinstance(first_step, dict)
    first_step["private_prompt"] = "hole prompt"
    first_step["model_transcript"] = "raw model text"
    original = copy.deepcopy(episode)

    result = normalize_trajectory(episode, emitted_at_ms=42, persist=True)
    payload_keys = _mapping_keys(result.trajectory.to_dict())
    receipt_keys = _mapping_keys(result.receipt.to_dict())
    forbidden = {
        "prompt",
        "private_prompt",
        "chain_of_thought",
        "model_transcript",
        "api_key",
        "credentials",
        "password",
        "source_body",
        "unbounded_logs",
    }
    assert forbidden.isdisjoint(payload_keys)
    assert forbidden.isdisjoint(receipt_keys)
    assert result.receipt.removed_field_classes == (
        RedactedFieldClass.PROMPT.value,
        RedactedFieldClass.CHAIN_OF_THOUGHT.value,
        RedactedFieldClass.SECRET.value,
        RedactedFieldClass.CREDENTIAL.value,
        RedactedFieldClass.REDUNDANT_BODY.value,
        RedactedFieldClass.UNBOUNDED_LOG.value,
    )
    assert result.receipt.emitted_at_ms == 42
    assert episode == original
    assert result.trajectory.total_tokens == 12
    assert result.trajectory.human_interventions == 0


def test_normalizer_orders_contracts_and_completes_cost_fields() -> None:
    episode = _kind_recipe(EpisodeKind.ACCEPTED_TASK_RECEIPT)
    episode["steps"] = [episode["steps"][1], episode["steps"][0]]
    episode.pop("total_tokens", None)
    episode.pop("total_latency_ms", None)
    result = TrajectoryNormalizer().normalize(episode)
    assert tuple(step.operation_contract for step in result.trajectory.steps) == (
        "typed-hole-service@1",
        "test-runner@1",
    )
    assert tuple(step.sequence for step in result.trajectory.steps) == (0, 1)
    assert result.trajectory.total_tokens == 12
    assert result.trajectory.total_latency_ms == 50
    assert result.trajectory.steps[0].hole_type == HoleType.CLASSIFY_FAILURE.value


def test_human_decision_completes_intervention_fields() -> None:
    episode = _kind_recipe(EpisodeKind.AUTHORIZED_HUMAN_DECISION_RECEIPT)
    steps = episode["steps"]
    assert isinstance(steps, list)
    first_step = steps[0]
    assert isinstance(first_step, dict)
    first_step["human_interventions"] = 0
    result = TrajectoryNormalizer().normalize(episode)
    assert result.trajectory.human_interventions == 1
    assert result.trajectory.steps[0].human_interventions == 1


def test_forbidden_operation_and_hole_types_are_rejected() -> None:
    episode = _kind_recipe(EpisodeKind.ACCEPTED_TASK_RECEIPT)
    episode["steps"][0]["operation"] = "ARBITRARY_SHELL"
    with pytest.raises(TrajectoryAdmissionError) as raised:
        TrajectoryNormalizer().normalize(episode)
    assert raised.value.reason_code == TrajectoryAdmissionReason.FORBIDDEN_OPERATION.value

    episode = _kind_recipe(EpisodeKind.ACCEPTED_TASK_RECEIPT)
    episode["steps"][0]["hole_type"] = "AUTHORITY_DECISION"
    with pytest.raises(TrajectoryAdmissionError) as raised:
        TrajectoryNormalizer().normalize(episode)
    assert raised.value.reason_code == TrajectoryAdmissionReason.FORBIDDEN_HOLE.value


def test_already_normalized_trajectory_is_not_an_admitted_source() -> None:
    with pytest.raises(TrajectoryAdmissionError):
        TrajectoryNormalizer().normalize(trajectory().to_dict())


def test_float_confidence_cannot_enter_normalization() -> None:
    episode = _kind_recipe(EpisodeKind.ACCEPTED_TASK_RECEIPT)
    episode["priority"] = 0.5
    with pytest.raises(TrajectoryAdmissionError) as raised:
        TrajectoryNormalizer().normalize(episode)
    assert raised.value.reason_code == TrajectoryAdmissionReason.FLOATING_POINT.value


def test_source_episode_json_normalizes_through_the_same_admission_path() -> None:
    episode = _kind_recipe(EpisodeKind.ADMITTED_TEST_RECEIPT)
    encoded = json.dumps(episode, separators=(",", ":"), sort_keys=True)
    result = TrajectoryNormalizer().normalize(encoded)
    assert result.trajectory.source_episode_kind == EpisodeKind.ADMITTED_TEST_RECEIPT
    assert result.receipt.source_episode_cid == "admitted_test_receipt-episode"
    assert parse_execution_trajectory(result.trajectory.to_json()) == result.trajectory
    from_bytes = TrajectoryNormalizer().normalize(encoded.encode("utf-8"))
    assert from_bytes.trajectory.content_id == result.trajectory.content_id


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item) for item in value)
    return ()


def _step_from_recipe(raw: Mapping[str, object]) -> TrajectoryStep:
    return TrajectoryStep(
        sequence=raw["sequence"],
        operation=StepOperation(str(raw["operation"])),
        operation_contract=str(raw["operation_contract"]),
        initial_state_cid=str(raw["initial_state_cid"]),
        terminal_state_cid=str(raw["terminal_state_cid"]),
        observation_cids=_string_tuple(raw.get("observation_cids", ())),
        effect_ids=_string_tuple(raw.get("effect_ids", ())),
        validation_receipt_cids=_string_tuple(raw.get("validation_receipt_cids", ())),
        hole_type=str(raw.get("hole_type") or ""),
        model_calls=raw.get("model_calls", 0),
        input_tokens=raw.get("input_tokens", 0),
        output_tokens=raw.get("output_tokens", 0),
        latency_ms=raw.get("latency_ms", 0),
        human_interventions=raw.get("human_interventions", 0),
        status=TraceEventStatus(str(raw.get("status") or TraceEventStatus.SUCCEEDED.value)),
    )


def test_typed_and_serialized_nested_contracts_normalize() -> None:
    episode = _kind_recipe(EpisodeKind.ADMITTED_TEST_RECEIPT)
    raw_steps = episode["steps"]
    assert isinstance(raw_steps, list)
    assert isinstance(raw_steps[0], dict)
    step = _step_from_recipe(raw_steps[0])
    episode["steps"] = [step]
    episode["bindings"] = _bindings()
    episode["content_id"] = "ignored-episode-identity"
    episode["cid"] = "ignored-episode-identity"
    typed = TrajectoryNormalizer().normalize(episode)
    assert typed.trajectory.steps[0].operation is StepOperation.RUN_SELECTED_TESTS
    assert typed.trajectory.terminal_abstract_state_cid == "state-1"

    serialized = _kind_recipe(EpisodeKind.ADMITTED_TEST_RECEIPT)
    serialized["steps"] = [step.to_dict()]
    serialized["outcome"] = TrajectoryOutcome(
        status=TrajectoryTerminalStatus.ACCEPTED,
        accepted_criterion_ids=("criterion-a",),
        validation_receipt_cids=("admitted-test",),
        proof_receipt_cids=(),
    ).to_dict()
    serialized["content_id"] = "ignored-serialized-identity"
    result = TrajectoryNormalizer().normalize(serialized)
    assert result.trajectory.steps[0].validation_receipt_cids == ("admitted-test",)
    assert result.artifact_state == ArtifactState.CANDIDATE
    validate_execution_trajectory_contract(result.trajectory)


def test_omitted_cost_fields_are_completed_from_steps() -> None:
    episode = _kind_recipe(EpisodeKind.ADMITTED_TEST_RECEIPT)
    episode.pop("total_cost_units", None)
    episode.pop("total_tokens", None)
    episode.pop("total_latency_ms", None)
    episode.pop("human_interventions", None)
    result = TrajectoryNormalizer().normalize(episode)
    assert result.trajectory.total_tokens == 0
    assert result.trajectory.total_latency_ms == 25
    assert result.trajectory.total_cost_units == 0
    assert result.trajectory.human_interventions == 0


def test_human_decision_completes_zero_episode_intervention_total() -> None:
    episode = _kind_recipe(EpisodeKind.AUTHORIZED_HUMAN_DECISION_RECEIPT)
    episode["human_interventions"] = 0
    steps = episode["steps"]
    assert isinstance(steps, list)
    first_step = steps[0]
    assert isinstance(first_step, dict)
    first_step["human_interventions"] = 0
    result = TrajectoryNormalizer().normalize(episode)
    assert result.trajectory.human_interventions == 1
    assert result.trajectory.steps[0].human_interventions == 1


def test_unknown_hole_type_is_forbidden() -> None:
    episode = _kind_recipe(EpisodeKind.ACCEPTED_TASK_RECEIPT)
    episode["steps"][0]["hole_type"] = "NOT_A_TYPED_HOLE"
    with pytest.raises(TrajectoryAdmissionError) as raised:
        TrajectoryNormalizer().normalize(episode)
    assert raised.value.reason_code == TrajectoryAdmissionReason.FORBIDDEN_HOLE.value
