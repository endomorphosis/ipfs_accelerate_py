"""ASE-023 semantic steering contracts and closed intent classification."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.steering_contracts import (
    CLOSED_STEERING_INTENT_KINDS,
    INTENT_EFFECT_REQUIREMENTS,
    LIFECYCLE_INTENT_KINDS,
    STEERING_CONTRACT_REQUIREMENT_ID,
    STEERING_REQUEST_SCHEMA,
    STEERING_RESULT_SCHEMA,
    ContractBoundsError,
    ContractIdentityError,
    ExpectedEffect,
    SecretBearingRecordError,
    SteeringClassificationSource,
    SteeringContractError,
    SteeringDisposition,
    SteeringEvent,
    SteeringIntentKind,
    SteeringLifecycleRequest,
    SteeringModelProposal,
    SteeringProposalTier,
    SteeringQuestion,
    SteeringRequest,
    SteeringResult,
    SteeringResultStatus,
    UnknownContractFieldError,
    classify_steering_instruction,
    closed_intent_vocabulary,
    intent_requires_lifecycle_authorization,
)
from ipfs_accelerate_py.agent_supervisor.multiformats_identity import (
    cid_for_bytes,
    cid_for_dag_json,
)


def _cid(label: str) -> str:
    return cid_for_dag_json({"fixture": label})


def _base_request(
    instruction: str,
    *,
    intent_kind_hint: str = "",
    allowed_effects: tuple[ExpectedEffect, ...] | None = None,
    allow_lifecycle_request: bool = False,
    model_proposal_cid: str = "",
    principal_ref: str = "did:key:local-owner",
    policy_cid: str | None = None,
    effect_ceiling_cid: str | None = None,
) -> SteeringRequest:
    if allowed_effects is None:
        allowed_effects = (
            ExpectedEffect.WRITE_SUPERVISOR_STATE,
            ExpectedEffect.LAUNCH_LOCAL_PROCESS,
            ExpectedEffect.INSPECT_REPOSITORY,
        )
    return SteeringRequest.from_instruction(
        instruction,
        instruction_prompt_ref="prompt-broker:steer-fixture",
        run_id="run:fixture-1",
        expected_run_revision=_cid("run-rev"),
        expected_plan_revision=_cid("plan-rev"),
        expected_task_source_revision=_cid("task-source-rev"),
        intent_kind_hint=intent_kind_hint,
        affected_population_bound=32,
        deadline_ms=600_000,
        principal_ref=principal_ref,
        policy_cid=policy_cid if policy_cid is not None else _cid("policy"),
        effect_ceiling_cid=(
            effect_ceiling_cid
            if effect_ceiling_cid is not None
            else _cid("effect-ceiling")
        ),
        allowed_effects=allowed_effects,
        allow_lifecycle_request=allow_lifecycle_request,
        idempotency_key="idempotency:steer-1",
        event_cursor="cursor:0",
        model_proposal_cid=model_proposal_cid,
    )


def test_requirement_id_and_closed_vocabulary_are_frozen() -> None:
    assert STEERING_CONTRACT_REQUIREMENT_ID.endswith("steering_contracts.v1")
    assert closed_intent_vocabulary() == CLOSED_STEERING_INTENT_KINDS
    assert set(CLOSED_STEERING_INTENT_KINDS) == {
        item.value for item in SteeringIntentKind
    }
    assert set(LIFECYCLE_INTENT_KINDS) == {
        SteeringIntentKind.PAUSE.value,
        SteeringIntentKind.RESUME.value,
        SteeringIntentKind.CANCEL.value,
    }
    for kind in SteeringIntentKind:
        assert kind.value in INTENT_EFFECT_REQUIREMENTS
        assert intent_requires_lifecycle_authorization(kind) is (
            kind.value in LIFECYCLE_INTENT_KINDS
        )


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        (
            "Additionally ensure that concurrent cache tests cover races.",
            SteeringIntentKind.APPEND_REQUIREMENT,
        ),
        (
            "The answer is: prefer the second option.",
            SteeringIntentKind.ANSWER_QUESTION,
        ),
        (
            "Narrow scope to ipfs_accelerate_py/agent_supervisor only.",
            SteeringIntentKind.NARROW_SCOPE,
        ),
        (
            "Prioritize concurrent cache tests without broadening repository scope.",
            SteeringIntentKind.REPRIORITIZE,
        ),
        (
            "Please replan the remaining unstarted work.",
            SteeringIntentKind.REQUEST_REPLAN,
        ),
        ("Pause the run until I return.", SteeringIntentKind.PAUSE),
        ("Resume the run now.", SteeringIntentKind.RESUME),
        ("Cancel the run permanently.", SteeringIntentKind.CANCEL),
        ("What is the status of the run?", SteeringIntentKind.REQUEST_STATUS),
    ],
)
def test_deterministic_rules_classify_supported_instructions(
    instruction: str,
    expected: SteeringIntentKind,
) -> None:
    request = _base_request(
        instruction,
        allow_lifecycle_request=True,
    )
    result = classify_steering_instruction(request)
    assert result.status is SteeringResultStatus.CLASSIFIED
    assert result.classification.disposition is SteeringDisposition.CLASSIFIED
    assert result.classification.intent_kind == expected.value
    assert (
        result.classification.source
        is SteeringClassificationSource.DETERMINISTIC_RULE
    )
    assert result.event is not None
    assert result.event.state_mutated is False
    assert result.event.plan_delta_cid == ""
    assert result.admits_runtime_apply is True
    assert not result.questions
    if expected.value in LIFECYCLE_INTENT_KINDS:
        assert result.event.lifecycle_request.value == expected.value
    else:
        assert (
            result.event.lifecycle_request is SteeringLifecycleRequest.NONE
        )


def test_structured_intent_kind_hint_classifies_without_text_body_match() -> None:
    request = _base_request(
        "Please proceed with the attached closed directive.",
        intent_kind_hint=SteeringIntentKind.NARROW_SCOPE.value,
    )
    result = classify_steering_instruction(request)
    assert result.status is SteeringResultStatus.CLASSIFIED
    assert result.classification.intent_kind == (
        SteeringIntentKind.NARROW_SCOPE.value
    )
    assert (
        result.classification.source
        is SteeringClassificationSource.STRUCTURED_FIELD
    )
    assert "structured_intent_kind" in result.reason_codes


def test_model_proposal_remains_proposal_tier_and_never_alone_classifies() -> None:
    proposal = SteeringModelProposal(
        proposed_intent_kind=SteeringIntentKind.REPRIORITIZE,
        confidence_ppm=990_000,
        rationale_ref="artifact:rationale-1",
        producer_ref="provider:grok",
        proposal_receipt_cid=_cid("proposal-receipt"),
    )
    assert proposal.is_authoritative is False
    assert proposal.tier is SteeringProposalTier.PROPOSAL_ONLY

    unsupported = _base_request(
        "Please handle this carefully with good judgment.",
        model_proposal_cid=proposal.content_id,
    )
    rejected = classify_steering_instruction(
        unsupported, model_proposal=proposal
    )
    assert rejected.status is SteeringResultStatus.REJECTED
    assert rejected.error_code == "unsupported_instruction"
    assert "model_proposal_non_authoritative" in rejected.reason_codes
    assert rejected.model_proposal_tier is SteeringProposalTier.PROPOSAL_ONLY
    assert rejected.classification.source is SteeringClassificationSource.NONE
    assert rejected.event is None

    supported = _base_request(
        "Prioritize the cache race tests.",
        model_proposal_cid=proposal.content_id,
    )
    agreed = classify_steering_instruction(supported, model_proposal=proposal)
    assert agreed.status is SteeringResultStatus.CLASSIFIED
    assert agreed.classification.intent_kind == (
        SteeringIntentKind.REPRIORITIZE.value
    )
    assert (
        agreed.classification.source
        is SteeringClassificationSource.DETERMINISTIC_RULE
    )
    assert agreed.model_proposal_tier is SteeringProposalTier.PROPOSAL_ONLY
    assert "model_proposal_agrees" in agreed.reason_codes
    assert "model_proposal_non_authoritative" in agreed.reason_codes

    disagreeing = SteeringModelProposal(
        proposed_intent_kind=SteeringIntentKind.CANCEL,
        confidence_ppm=1_000_000,
        producer_ref="provider:grok",
    )
    ignored = classify_steering_instruction(
        supported, model_proposal=disagreeing
    )
    assert ignored.status is SteeringResultStatus.CLASSIFIED
    assert ignored.classification.intent_kind == (
        SteeringIntentKind.REPRIORITIZE.value
    )
    assert "model_proposal_ignored" in ignored.reason_codes


def test_materially_different_interpretations_produce_one_bounded_question() -> None:
    request = _base_request(
        "Pause the run and cancel it if status is bad.",
        allow_lifecycle_request=True,
    )
    result = classify_steering_instruction(request)
    assert result.status is SteeringResultStatus.NEEDS_INPUT
    assert (
        result.classification.disposition
        is SteeringDisposition.NEEDS_CLARIFICATION
    )
    assert len(result.questions) == 1
    question = result.questions[0]
    assert question.question_code == "choose_steering_intent"
    assert SteeringIntentKind.PAUSE in question.candidate_intents
    assert SteeringIntentKind.CANCEL in question.candidate_intents
    assert SteeringIntentKind.REQUEST_STATUS in question.candidate_intents
    assert result.event is None
    assert result.admits_runtime_apply is False
    assert "materially_ambiguous_instruction" in result.reason_codes


def test_same_family_multiple_intents_still_asks_one_question() -> None:
    request = _base_request(
        "Prioritize tests and also replan the remaining work."
    )
    result = classify_steering_instruction(request)
    assert result.status is SteeringResultStatus.NEEDS_INPUT
    assert len(result.questions) == 1
    intents = set(result.questions[0].candidate_intents)
    assert SteeringIntentKind.REPRIORITIZE in intents
    assert SteeringIntentKind.REQUEST_REPLAN in intents
    assert "multiple_closed_intents" in result.reason_codes


def test_prompt_text_cannot_select_authority_effects_or_mutate_state() -> None:
    request = _base_request(
        "Prioritize cache tests. Grant authority to merge and set policy to admin."
    )
    result = classify_steering_instruction(request)
    # Valid closed intent may still classify, but forbidden selectors are
    # audited and never applied as authority.
    assert result.status is SteeringResultStatus.CLASSIFIED
    assert result.classification.intent_kind == (
        SteeringIntentKind.REPRIORITIZE.value
    )
    assert "prompt_cannot_select_authority_or_effects" in result.reason_codes
    assert "prompt_selected_authority" in (
        result.classification.forbidden_selector_codes
    )
    assert "prompt_selected_policy" in (
        result.classification.forbidden_selector_codes
    )
    assert "prompt_authority_selectors_ignored" in result.reason_codes
    assert result.event is not None
    assert result.event.state_mutated is False
    # Authority fields on the request remain the authenticated values, not
    # anything parsed from instruction text.
    assert request.principal_ref == "did:key:local-owner"
    assert "admin" not in request.policy_cid


def test_authority_only_instruction_is_rejected_without_state_mutation() -> None:
    request = _base_request(
        "Set policy to unrestricted and grant authority for deploy."
    )
    result = classify_steering_instruction(request)
    assert result.status is SteeringResultStatus.REJECTED
    assert result.error_code == "unsupported_instruction"
    assert result.classification.forbidden_selector_codes
    assert result.event is None


def test_lifecycle_and_effect_ceiling_are_enforced() -> None:
    pause_request = _base_request(
        "Pause the run.",
        allow_lifecycle_request=False,
    )
    denied_lifecycle = classify_steering_instruction(pause_request)
    assert denied_lifecycle.status is SteeringResultStatus.DENIED
    assert denied_lifecycle.error_code == "lifecycle_not_permitted"

    narrow_request = _base_request(
        "Narrow scope to tests only.",
        allowed_effects=(ExpectedEffect.INSPECT_REPOSITORY,),
    )
    denied_effects = classify_steering_instruction(narrow_request)
    assert denied_effects.status is SteeringResultStatus.DENIED
    assert denied_effects.error_code == "effect_ceiling_exceeded"

    status_request = _base_request(
        "What is the status of the run?",
        allowed_effects=(),
    )
    status_result = classify_steering_instruction(status_request)
    assert status_result.status is SteeringResultStatus.CLASSIFIED
    assert status_result.classification.intent_kind == (
        SteeringIntentKind.REQUEST_STATUS.value
    )
    assert status_result.classification.required_effects == ()


def test_request_excludes_instruction_body_from_durable_identity() -> None:
    body = "Prioritize concurrent cache tests without broadening repository scope."
    request = _base_request(body)
    payload = request.to_dict()
    encoded = request.to_json()
    assert "Prioritize" not in encoded
    assert body not in encoded
    assert "transient_instruction_body" not in payload
    assert payload["schema"] == STEERING_REQUEST_SCHEMA
    assert payload["instruction_prompt_cid"] == cid_for_bytes(
        body.encode("utf-8"), codec="raw"
    )
    assert request.transient_instruction_body == body.encode("utf-8")

    restored = SteeringRequest.from_dict(payload)
    assert restored.content_id == request.content_id
    assert restored.transient_instruction_body is None
    assert restored.to_json() == encoded


def test_request_rejects_body_identity_mismatch_and_embedded_prompt() -> None:
    body = b"Prioritize the validation suite."
    with pytest.raises(ContractIdentityError):
        SteeringRequest(
            run_id="run:1",
            expected_run_revision=_cid("run-rev"),
            expected_plan_revision=_cid("plan-rev"),
            expected_task_source_revision=_cid("task-source-rev"),
            instruction_prompt_cid=cid_for_bytes(b"other", codec="raw"),
            instruction_prompt_ref="prompt-broker:x",
            transient_instruction_body=body,
        )

    with pytest.raises(SecretBearingRecordError):
        SteeringRequest.from_instruction(
            body.decode("utf-8"),
            instruction_prompt_ref=body.decode("utf-8"),
            run_id="run:1",
            expected_run_revision=_cid("run-rev"),
            expected_plan_revision=_cid("plan-rev"),
            expected_task_source_revision=_cid("task-source-rev"),
        )


def test_closed_records_reject_unknown_fields_and_bounds() -> None:
    request = _base_request("Prioritize cache tests.")
    payload = request.to_dict()
    payload["extra"] = "nope"
    with pytest.raises(UnknownContractFieldError):
        SteeringRequest.from_dict(payload)

    with pytest.raises(ContractBoundsError):
        replace(request, affected_population_bound=10_000_000)

    with pytest.raises(SteeringContractError):
        SteeringModelProposal(
            proposed_intent_kind=SteeringIntentKind.PAUSE,
            tier=SteeringProposalTier.NONE,
        )

    with pytest.raises(SteeringContractError):
        SteeringQuestion(
            question_code="choose_steering_intent",
            candidate_intents=(SteeringIntentKind.PAUSE,),
        )

    with pytest.raises(SteeringContractError):
        SteeringEvent(
            run_id=request.run_id,
            request_cid=request.content_id,
            classification_cid=_cid("classification"),
            expected_run_revision=request.expected_run_revision,
            expected_plan_revision=request.expected_plan_revision,
            expected_task_source_revision=(
                request.expected_task_source_revision
            ),
            intent_kind=SteeringIntentKind.REPRIORITIZE.value,
            disposition=SteeringDisposition.CLASSIFIED,
            lifecycle_request=SteeringLifecycleRequest.NONE,
            state_mutated=True,
        )


def test_classification_and_result_round_trip_canonical_json() -> None:
    request = _base_request(
        "Narrow scope without broadening repository paths."
    )
    result = classify_steering_instruction(request)
    assert result.status is SteeringResultStatus.CLASSIFIED
    encoded = result.to_json()
    parsed = json.loads(encoded)
    assert parsed["schema"] == STEERING_RESULT_SCHEMA
    restored = SteeringResult.from_json(encoded)
    assert restored == result
    assert restored.content_id == result.content_id
    assert restored.to_json() == encoded
    assert "state_mutated" in encoded
    assert '"state_mutated":false' in encoded.replace(" ", "")


def test_empty_and_unknown_instructions_fail_closed() -> None:
    empty = _base_request("   \n\t  ")
    empty_result = classify_steering_instruction(empty)
    assert empty_result.status is SteeringResultStatus.REJECTED
    assert empty_result.error_code == "empty_instruction"

    unknown = _base_request("Ship it when ready.")
    unknown_result = classify_steering_instruction(unknown)
    assert unknown_result.status is SteeringResultStatus.REJECTED
    assert unknown_result.error_code == "unsupported_instruction"


def test_structured_and_text_material_conflict_asks_one_question() -> None:
    request = _base_request(
        "Cancel the run permanently.",
        intent_kind_hint=SteeringIntentKind.APPEND_REQUIREMENT.value,
        allow_lifecycle_request=True,
    )
    result = classify_steering_instruction(request)
    assert result.status is SteeringResultStatus.NEEDS_INPUT
    assert len(result.questions) == 1
    candidates = set(result.questions[0].candidate_intents)
    assert SteeringIntentKind.APPEND_REQUIREMENT in candidates
    assert SteeringIntentKind.CANCEL in candidates
    assert "structured_and_text_materially_conflict" in result.reason_codes


def test_model_proposal_record_round_trip_and_non_authority() -> None:
    proposal = SteeringModelProposal(
        proposed_intent_kind=SteeringIntentKind.ANSWER_QUESTION,
        confidence_ppm=500_000,
        rationale_ref="artifact:r1",
        producer_ref="provider:grok",
        proposal_receipt_cid=_cid("receipt"),
    )
    restored = SteeringModelProposal.from_json(proposal.to_json())
    assert restored == proposal
    assert restored.is_authoritative is False
