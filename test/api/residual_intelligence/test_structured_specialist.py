from __future__ import annotations

import json

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.abstention import (
    SelectivePredictionPolicy,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.calibration import (
    CalibrationEvidence,
    CalibrationGroup,
    CalibrationThresholdBinding,
    ThresholdChangeOrigin,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ExpertDisposition,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    TrainingAvailability,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.expert_specs import (
    ExpertClass,
    expert_spec_for,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.local_experts import (
    IndependentValidationReceipt,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.residual_ir import ResidualTaskInput
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.structured_decoding import (
    DecodeStatus,
    grammar_for,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.structured_specialist import (
    ADMITTED_STRUCTURED_FAMILIES,
    CONSTRAINED_DECODER_FORM,
    ConstrainedStructuredExpert,
    StructuredDecodeRequest,
    StructuredDecodeResult,
    StructuredSpecialistForm,
    StructuredSpecialistPrediction,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.task_families import family_spec_for

from .helpers import admission

ADMISSION_ID = "admission:fixture-current"
SPLIT_ROOT = "split:fixture-current"
HOLDOUT_ROOT = "holdout:fixture"
EVALUATION_ID = "evaluation:fixture-current"
THRESHOLD = 800_000


def group(
    *,
    family: ResidualTaskFamily = ResidualTaskFamily.PROCEDURE_HOLE_FILLING,
    risk: RiskClass = RiskClass.R3,
    model: str = "fixture-constrained@1",
) -> CalibrationGroup:
    return CalibrationGroup(
        family=family,
        repository="ipfs_accelerate_py",
        language="python",
        framework="pytest",
        risk=risk,
        model=model,
        quantization="none",
        hardware="cpu-standard",
        context_tier="evidence",
    )


def evidence(target: CalibrationGroup | None = None) -> CalibrationEvidence:
    subject = target or group()
    return CalibrationEvidence(
        group=subject,
        admission_id=ADMISSION_ID,
        admission_decision=TrainingAvailability.ADMITTED,
        split_root=SPLIT_ROOT,
        holdout_root=HOLDOUT_ROOT,
        evaluation_identity=EVALUATION_ID,
        example_identities=tuple(f"example:{index}" for index in range(10)),
        adversarial_example_identities=("adversarial:1",),
        evaluated_threshold_candidates=(THRESHOLD,),
        accept_count=8,
        abstain_count=2,
        reject_input_count=0,
        ood_count=0,
        capability_unavailable_count=0,
        validation_required_count=0,
        false_accept_count=0,
        critical_false_accept_count=0,
        precision_ppm=1_000_000,
        abstention_rate_ppm=200_000,
    )


def policy(target: CalibrationGroup | None = None) -> SelectivePredictionPolicy:
    subject = target or group()
    record = evidence(subject)
    return SelectivePredictionPolicy(
        current_admission_id=ADMISSION_ID,
        current_split_root=SPLIT_ROOT,
        current_holdout_root=HOLDOUT_ROOT,
        current_evaluation_identity=EVALUATION_ID,
        evidence=(record,),
        bindings=(
            CalibrationThresholdBinding(
                group_key=subject.group_key,
                accept_threshold_ppm=THRESHOLD,
                evidence_id=record.evidence_id,
                cas_identity="cas:operator:fixture",
                origin=ThresholdChangeOrigin.OPERATOR_CAS,
                rollback_threshold_ppm=THRESHOLD,
            ),
        ),
    )


def hole_features(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "hole_id": "hole:bind-arg-1",
        "procedure_root": "procedure:root:1",
        "procedure_preconditions_satisfied": True,
        "symbol_ids": ["arg:0"],
        "operation": "bind_argument",
        "context_complete": True,
    }
    payload.update(overrides)
    return payload


def patch_features(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "allowed_paths": ["ipfs_accelerate_py/module.py"],
        "symbol_ids": ["module.symbol"],
        "maximum_changed_lines": 200,
        "validation_ids": ["pytest:focused"],
        "operation": "replace_function",
        "context_complete": True,
    }
    payload.update(overrides)
    return payload


def task_input(
    *,
    family: ResidualTaskFamily = ResidualTaskFamily.PROCEDURE_HOLE_FILLING,
    risk: RiskClass = RiskClass.R3,
    features: dict[str, object] | None = None,
    allowed: tuple[str, ...] | None = None,
    token_budget: int = 256,
    question_id: str = "question:structured:1",
) -> ResidualTaskInput:
    output_class = (
        "PROCEDURE_HOLE_RESOLUTION"
        if family is ResidualTaskFamily.PROCEDURE_HOLE_FILLING
        else "PATCH_SKETCH"
    )
    return ResidualTaskInput(
        task_family=family,
        question_id=question_id,
        repository_state_cid="repo:tree:abc",
        objective_cid="objective:vrif",
        task_cid="task:VRIF-015",
        policy_cid="policy:residual-v1",
        context_capsule_cid="capsule:bounded:1",
        compact_features=features if features is not None else hole_features(),
        allowed_outputs=allowed or (output_class, "ABSTAIN"),
        risk_class=risk,
        validation_policy=family_spec_for(family).validator_identity,
        token_budget=token_budget,
    )


def validator(
    family: ResidualTaskFamily = ResidualTaskFamily.PROCEDURE_HOLE_FILLING,
    *,
    accepted: bool = True,
) -> IndependentValidationReceipt:
    return IndependentValidationReceipt(
        validator_identity=family_spec_for(family).validator_identity,
        accepted=accepted,
        evidence_references=("validator:current-tree",),
    )


def expert(
    *,
    family: ResidualTaskFamily = ResidualTaskFamily.PROCEDURE_HOLE_FILLING,
    risk: RiskClass | None = None,
    bind_policy: bool = True,
    compiler_available: bool = True,
    decoder_available: bool = True,
    maximum_changed_lines: int = 200,
) -> ConstrainedStructuredExpert:
    resolved_risk = risk if risk is not None else (
        RiskClass.R3 if family is ResidualTaskFamily.PROCEDURE_HOLE_FILLING else RiskClass.R4
    )
    subject = group(family=family, risk=resolved_risk)
    return ConstrainedStructuredExpert(
        task_family=family,
        calibration_group=subject,
        compiler_available=compiler_available,
        decoder_available=decoder_available,
        maximum_changed_lines=maximum_changed_lines,
        selective_policy=policy(subject) if bind_policy else None,
    )


def request_for(
    item: ResidualTaskInput,
    *,
    raw_output: str = "",
    accepted: bool = True,
) -> StructuredDecodeRequest:
    return StructuredDecodeRequest(
        task_input=item,
        raw_output=raw_output,
        independent_validation=validator(item.task_family, accepted=accepted),
        candidate_only=True,
    )


def hole_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "hole_id": "hole:bind-arg-1",
        "operator_id": "bind_argument",
        "argument_reference_ids": ["arg:0"],
        "precondition_reference_ids": ["procedure:root:1"],
    }
    payload.update(overrides)
    return payload


def encoded_hole(
    *,
    output_class: str = "PROCEDURE_HOLE_RESOLUTION",
    payload: dict[str, object] | None = None,
    score: int = 900_000,
    abstained: bool = False,
    reason_codes: list[str] | None = None,
    extra: dict[str, object] | None = None,
    calibration_group: str | None = None,
) -> str:
    body: dict[str, object] = {
        "output_class": output_class,
        "structured_payload": hole_payload() if payload is None else payload,
        "confidence_or_score": score,
        "calibration_group": calibration_group or group().group_key,
        "abstained": abstained,
        "reason_codes": [] if reason_codes is None else reason_codes,
        "evidence_references": ["hole:bind-arg-1"],
        "candidate_only": True,
    }
    if extra:
        body.update(extra)
    return json.dumps(body)


def test_specialist_is_family_bounded_class_e_grammar() -> None:
    hole = expert()
    patch = expert(family=ResidualTaskFamily.PATCH_SKETCH_GENERATION, risk=RiskClass.R4)
    assert ADMITTED_STRUCTURED_FAMILIES == {
        ResidualTaskFamily.PROCEDURE_HOLE_FILLING,
        ResidualTaskFamily.PATCH_SKETCH_GENERATION,
    }
    assert hole.expert_class is ExpertClass.E
    assert hole.form == CONSTRAINED_DECODER_FORM
    assert hole.grammar.grammar_id == grammar_for(hole.task_family).grammar_id
    assert hole.expert_spec.grammar_id == hole.grammar.grammar_id
    assert hole.expert_spec.forms == (CONSTRAINED_DECODER_FORM,)
    assert hole.family_spec.emit_prose_by_default is False
    assert hole.candidate_only is True
    assert patch.grammar.task_family is ResidualTaskFamily.PATCH_SKETCH_GENERATION
    with pytest.raises(ResidualIntelligenceError, match="unsupported_structured_family"):
        ConstrainedStructuredExpert(
            task_family=ResidualTaskFamily.FAILURE_ATTRIBUTION,
            calibration_group=group(family=ResidualTaskFamily.FAILURE_ATTRIBUTION, risk=RiskClass.R2),
        )
    with pytest.raises(ResidualIntelligenceError, match="class_e"):
        ConstrainedStructuredExpert(
            task_family=ResidualTaskFamily.PROCEDURE_HOLE_FILLING,
            calibration_group=group(),
            expert_class=ExpertClass.A,
        )


def test_grammar_constrained_emit_and_accept_procedure_hole() -> None:
    specialist = expert()
    item = task_input()
    emitted = specialist.emit_constrained(item)
    assert emitted is not None
    decoded = specialist.decode(
        StructuredDecodeRequest(
            task_input=item,
            raw_output=emitted,
            independent_validation=validator(),
        )
    )
    assert decoded.status is DecodeStatus.VALID
    assert decoded.output is not None
    assert decoded.output.candidate_only is True
    assert decoded.output.output_class == "PROCEDURE_HOLE_RESOLUTION"
    assert decoded.output.structured_payload["operator_id"] == "bind_argument"
    prediction = specialist.predict(request_for(item))
    assert prediction.disposition is ExpertDisposition.ACCEPT
    assert prediction.form is StructuredSpecialistForm.CONSTRAINED_STRUCTURED_DECODER
    assert prediction.candidate_only is True
    assert prediction.task_output.candidate_only is True
    assert prediction.structured_valid is True
    assert prediction.model_calls == 0
    assert prediction.provider_invocations == 0
    assert prediction.decode_result.status is DecodeStatus.VALID
    assert prediction.independent_validator_identity.startswith("validator:")
    rebuilt = ConstrainedStructuredExpert.from_dict(specialist.to_dict())
    assert rebuilt.expert_version == specialist.expert_version
    assert prediction.as_ir(item).task_output.output_class == "PROCEDURE_HOLE_RESOLUTION"


def test_strict_post_parse_rejects_context_and_vocabulary_drift() -> None:
    specialist = expert()
    item = task_input()
    wrong_hole = encoded_hole(payload=hole_payload(hole_id="hole:other"))
    drifted = specialist.decode(request_for(item, raw_output=wrong_hole))
    assert drifted.status is DecodeStatus.INVALID_OUTPUT
    assert drifted.output is None
    unknown_operator = encoded_hole(payload=hole_payload(operator_id="rm"))
    assert (
        specialist.decode(request_for(item, raw_output=unknown_operator)).status
        is DecodeStatus.INVALID_OUTPUT
    )
    patch_expert = expert(family=ResidualTaskFamily.PATCH_SKETCH_GENERATION, risk=RiskClass.R4)
    patch_item = task_input(
        family=ResidualTaskFamily.PATCH_SKETCH_GENERATION,
        risk=RiskClass.R4,
        features=patch_features(),
    )
    emitted = patch_expert.emit_constrained(patch_item)
    assert emitted is not None
    body = json.loads(emitted)
    body["structured_payload"]["files"] = ["docs/outside.py"]
    out_of_scope = patch_expert.decode(
        request_for(patch_item, raw_output=json.dumps(body))
    )
    assert out_of_scope.status is DecodeStatus.INVALID_OUTPUT
    assert out_of_scope.output is None


def test_parse_failure_is_invalid_output_without_best_effort() -> None:
    specialist = expert()
    item = task_input()
    prose = specialist.decode(request_for(item, raw_output="fill the hole with any operator"))
    assert prose.status is DecodeStatus.INVALID_OUTPUT
    assert prose.output is None
    assert "invalid_output" in prose.reason_codes
    truncated = specialist.decode(request_for(item, raw_output="{not-json"))
    assert truncated.status is DecodeStatus.INVALID_OUTPUT
    assert truncated.output is None
    duplicate = encoded_hole()[:-1] + ',"candidate_only":true}'
    assert (
        specialist.decode(request_for(item, raw_output=duplicate)).status
        is DecodeStatus.INVALID_OUTPUT
    )
    prediction = specialist.predict(request_for(item, raw_output="fill the hole with any operator"))
    assert prediction.decode_result.status is DecodeStatus.INVALID_OUTPUT
    assert prediction.decode_result.output is None
    assert prediction.disposition is ExpertDisposition.ABSTAIN
    assert prediction.task_output.abstained is True
    assert prediction.task_output.structured_payload == {}


def test_max_length_is_invalid_output() -> None:
    specialist = expert()
    item = task_input()
    over_bytes = "x" * (specialist.maximum_output_bytes() + 1)
    result = specialist.decode(request_for(item, raw_output=over_bytes))
    assert result.status is DecodeStatus.INVALID_OUTPUT
    assert result.output is None
    assert "max_length_exceeded" in result.reason_codes
    over_tokens = "token " * (specialist.maximum_output_tokens() + 1)
    token_result = specialist.decode(request_for(item, raw_output=over_tokens))
    assert token_result.status is DecodeStatus.INVALID_OUTPUT
    assert token_result.output is None
    oversized_input = task_input(
        token_budget=family_spec_for(item.task_family).maximum_input_tokens + 1
    )
    with pytest.raises(ResidualIntelligenceError, match="family_token_limit_exceeded"):
        family_spec_for(item.task_family).validate_task_input(oversized_input)
    rejected = specialist.predict(request_for(oversized_input))
    assert rejected.disposition is ExpertDisposition.REJECT_INPUT
    assert "family_token_limit_exceeded" in rejected.task_output.reason_codes


def test_candidate_only_is_immutable() -> None:
    specialist = expert()
    item = task_input()
    prediction = specialist.predict(request_for(item))
    assert prediction.candidate_only is True
    with pytest.raises(ResidualIntelligenceError, match="candidate_only"):
        StructuredDecodeRequest(task_input=item, candidate_only=False)
    with pytest.raises(ResidualIntelligenceError, match="candidate_only"):
        StructuredSpecialistPrediction(
            decode_result=prediction.decode_result,
            task_output=prediction.task_output,
            form=prediction.form,
            disposition=prediction.disposition,
            feature_identity=prediction.feature_identity,
            abstention=prediction.abstention,
            independent_validator_identity=prediction.independent_validator_identity,
            structured_valid=True,
            candidate_only=False,
        )


def test_bounded_context_rejects_private_bodies_and_unknown_features() -> None:
    specialist = expert()
    with pytest.raises(ResidualIntelligenceError, match="unknown_compact_feature"):
        specialist.family_spec.validate_task_input(
            task_input(features=hole_features(prompt_text="secret prompt"))
        )
    with pytest.raises(ResidualIntelligenceError, match="private body"):
        StructuredDecodeRequest(
            task_input=task_input(features=hole_features(**{"source_text": "body"}))
        )
    incomplete = specialist.predict(
        request_for(task_input(features=hole_features(context_complete=False)))
    )
    assert incomplete.disposition is ExpertDisposition.ABSTAIN
    assert "bounded_context_insufficient" in incomplete.task_output.reason_codes
    assert "abstain_escalate" in incomplete.task_output.reason_codes


def test_abstain_escalate_on_missing_compiler_and_failed_preconditions() -> None:
    blocked = expert(compiler_available=False)
    unavailable = blocked.predict(request_for(task_input()))
    assert unavailable.disposition is ExpertDisposition.CAPABILITY_UNAVAILABLE
    assert "compiler_capability_unavailable" in unavailable.task_output.reason_codes
    assert unavailable.task_output.abstained is True
    failed = expert().predict(
        request_for(task_input(features=hole_features(procedure_preconditions_satisfied=False)))
    )
    assert failed.disposition is ExpertDisposition.ABSTAIN
    assert "procedure_preconditions_unsatisfied" in failed.task_output.reason_codes
    assert "abstain_escalate" in failed.task_output.reason_codes
    mismatched = expert().predict(
        request_for(
            task_input(
                family=ResidualTaskFamily.PATCH_SKETCH_GENERATION,
                risk=RiskClass.R4,
                features=patch_features(),
            )
        )
    )
    assert mismatched.disposition is ExpertDisposition.REJECT_INPUT
    decoder_down = expert(decoder_available=False).predict(request_for(task_input()))
    assert decoder_down.disposition is ExpertDisposition.CAPABILITY_UNAVAILABLE
    pending = expert().predict(
        StructuredDecodeRequest(task_input=task_input(), raw_output="", candidate_only=True)
    )
    assert pending.disposition is ExpertDisposition.VALIDATION_REQUIRED
    assert pending.task_output.abstained is False
    rejected = expert().predict(request_for(task_input(), accepted=False))
    assert rejected.disposition is ExpertDisposition.VALIDATION_REQUIRED


def test_no_freeform_authority_or_shell_fields() -> None:
    specialist = expert()
    item = task_input()
    for extra in (
        {"explanation": "long prose about the hole"},
        {"policy_permission": True},
        {"completed": True},
    ):
        raw = encoded_hole(extra=extra)
        result = specialist.decode(request_for(item, raw_output=raw))
        assert result.status is DecodeStatus.INVALID_OUTPUT
        assert result.output is None
    body = json.loads(encoded_hole())
    body["structured_payload"]["completed"] = True
    authority = specialist.decode(request_for(item, raw_output=json.dumps(body)))
    assert authority.status is DecodeStatus.INVALID_OUTPUT
    patch_expert = expert(family=ResidualTaskFamily.PATCH_SKETCH_GENERATION, risk=RiskClass.R4)
    patch_item = task_input(
        family=ResidualTaskFamily.PATCH_SKETCH_GENERATION,
        risk=RiskClass.R4,
        features=patch_features(),
    )
    emitted = patch_expert.emit_constrained(patch_item)
    assert emitted is not None
    sketch = json.loads(emitted)
    sketch["structured_payload"]["operations"] = ["delete_test"]
    deleted = patch_expert.decode(request_for(patch_item, raw_output=json.dumps(sketch)))
    assert deleted.status is DecodeStatus.INVALID_OUTPUT
    sketch["structured_payload"]["operations"] = ["weaken_validation"]
    weakened = patch_expert.decode(request_for(patch_item, raw_output=json.dumps(sketch)))
    assert weakened.status is DecodeStatus.INVALID_OUTPUT
    sketch["structured_payload"]["operations"] = ["shell"]
    shell = patch_expert.decode(request_for(patch_item, raw_output=json.dumps(sketch)))
    assert shell.status is DecodeStatus.INVALID_OUTPUT


def test_patch_sketch_remains_proposal_tier_after_independent_validation() -> None:
    specialist = expert(family=ResidualTaskFamily.PATCH_SKETCH_GENERATION, risk=RiskClass.R4)
    item = task_input(
        family=ResidualTaskFamily.PATCH_SKETCH_GENERATION,
        risk=RiskClass.R4,
        features=patch_features(),
    )
    prediction = specialist.predict(request_for(item))
    assert prediction.disposition is ExpertDisposition.VALIDATION_REQUIRED
    assert prediction.task_output.abstained is False
    assert prediction.task_output.output_class == "PATCH_SKETCH"
    assert prediction.task_output.candidate_only is True
    assert "validation_required" in prediction.task_output.reason_codes
    assert "VALIDATION_REQUIRED" in prediction.task_output.reason_codes
    assert "r4_r5_proposal_tier" in prediction.task_output.reason_codes
    assert prediction.structured_valid is True
    rebuilt = prediction.as_ir(item)
    assert rebuilt.task_output.output_class == "PATCH_SKETCH"
    assert rebuilt.task_output.abstained is False
    assert "VALIDATION_REQUIRED" in rebuilt.task_output.reason_codes
    spec = expert_spec_for(ResidualTaskFamily.PATCH_SKETCH_GENERATION, ExpertClass.E)
    assert spec.independent_validator_required is True


def test_training_unavailable_blocks_fit_without_model_download() -> None:
    blocked, _examples = admission(admitted=False)
    specialist = expert(bind_policy=False)
    with pytest.raises(ResidualIntelligenceError, match="training_unavailable"):
        specialist.fit(admission=blocked)
    record, _examples = admission()
    fitted = specialist.fit(admission=record, examples=1, steps=1)
    assert fitted.fitted is True
    assert fitted.admission_id == record.admission_id
    assert fitted.checkpoint_count == 1
    assert specialist.fitted is False


def test_request_and_result_round_trip_and_predicted_symbols() -> None:
    item = task_input()
    req = request_for(item, raw_output=encoded_hole())
    rebuilt = StructuredDecodeRequest.from_dict(req.to_dict())
    assert rebuilt.request_id == req.request_id
    assert isinstance(StructuredDecodeResult, type)
    specialist = expert()
    prediction = specialist.predict(req)
    cloned = StructuredSpecialistPrediction.from_dict(prediction.to_dict())
    assert cloned.prediction_id == prediction.prediction_id
    assert cloned.decode_result.status is DecodeStatus.VALID
