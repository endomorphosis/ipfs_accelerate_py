from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.abstention import (
    SelectivePredictionPolicy,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.baselines import (
    DeclarativeRule,
    ExactLookupEntry,
    LinearForm,
    RankingItem,
    RulePredicate,
    RulePredicateKind,
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
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.distillation import (
    DistillationBudget,
    DistillationResult,
    distill_classification_expert,
    distill_ranking_expert,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.expert_specs import (
    MIN_ROUTING_CHANGING_DELTA_PPM,
    ExpertClass,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.local_experts import (
    MAX_LOCAL_EXAMPLES,
    MAX_LOCAL_GPU_SECONDS,
    BatchedExpertRequest,
    ExpertEvaluation,
    ExpertEvaluationCase,
    IndependentValidationReceipt,
    LocalClassificationExpert,
    LocalExpertForm,
    LocalExpertPrediction,
    LocalRankingExpert,
    SmallRanker,
    admit_local_expert_class,
    classification_payload,
    local_feature_identity,
    ranking_payload,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.ood import (
    CANDIDATE_ONLY_AUTHORITY,
    BoundaryContract,
    FeatureRange,
    OODSignalKind,
    ReferenceDistribution,
    assess_out_of_distribution,
    observation_from_task_input,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.residual_ir import ResidualTaskInput
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.splits import SplitPartition
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.task_families import family_spec_for

from .helpers import admission

ADMISSION_ID = "admission:fixture-current"
SPLIT_ROOT = "split:fixture-current"
HOLDOUT_ROOT = "holdout:fixture"
EVALUATION_ID = "evaluation:fixture-current"
THRESHOLD = 800_000
REPOSITORY = "ipfs_accelerate_py"
SCHEMA = "failure-signature@1"
OPERATION = "classify_failure"
EFFECT = "candidate_failure_class"
CAPABILITY = "cpu-standard"
CONTEXT_FIELD = "context:evidence"
FAILURE_PAYLOAD = {
    "failure_class": "missing_dependency_edge",
    "recommended_action": "expand_context_reference",
    "reference_ids": ["dependency:1"],
}
TIMEOUT_PAYLOAD = {
    "failure_class": "provider_timeout",
    "recommended_action": "retry_local_validator",
    "reference_ids": ["timeout:1"],
}
FEATURE_NAMES = ("exit_code", "failure_signature")


def group(
    *,
    family: ResidualTaskFamily = ResidualTaskFamily.FAILURE_ATTRIBUTION,
    risk: RiskClass = RiskClass.R2,
    model: str = "fixture-linear@1",
) -> CalibrationGroup:
    return CalibrationGroup(
        family=family,
        repository=REPOSITORY,
        language="python",
        framework="pytest",
        risk=risk,
        model=model,
        quantization="none",
        hardware="cpu-standard",
        context_tier="evidence",
    )


def evidence(
    target: CalibrationGroup | None = None,
    *,
    accept_count: int = 8,
    abstain_count: int = 2,
    false_accept_count: int = 0,
    critical_false_accept_count: int = 0,
    thresholds: tuple[int, ...] = (THRESHOLD,),
) -> CalibrationEvidence:
    n_examples = accept_count + abstain_count
    precision = (
        0
        if accept_count == 0
        else ((accept_count - false_accept_count) * 1_000_000) // accept_count
    )
    return CalibrationEvidence(
        group=target or group(),
        admission_id=ADMISSION_ID,
        admission_decision=TrainingAvailability.ADMITTED,
        split_root=SPLIT_ROOT,
        holdout_root=HOLDOUT_ROOT,
        evaluation_identity=EVALUATION_ID,
        example_identities=tuple(f"example:{index}" for index in range(n_examples)),
        adversarial_example_identities=("adversarial:1",),
        evaluated_threshold_candidates=thresholds,
        accept_count=accept_count,
        abstain_count=abstain_count,
        reject_input_count=0,
        ood_count=0,
        capability_unavailable_count=0,
        validation_required_count=0,
        false_accept_count=false_accept_count,
        critical_false_accept_count=critical_false_accept_count,
        precision_ppm=precision,
        abstention_rate_ppm=(abstain_count * 1_000_000) // n_examples,
    )


def binding(
    target: CalibrationGroup | None = None,
    *,
    record: CalibrationEvidence | None = None,
    accept_threshold_ppm: int = THRESHOLD,
) -> CalibrationThresholdBinding:
    item = record or evidence(target)
    subject = target or item.group
    return CalibrationThresholdBinding(
        group_key=subject.group_key,
        accept_threshold_ppm=accept_threshold_ppm,
        evidence_id=item.evidence_id,
        cas_identity="cas:operator:fixture",
        origin=ThresholdChangeOrigin.OPERATOR_CAS,
        rollback_threshold_ppm=accept_threshold_ppm,
    )


def policy(
    target: CalibrationGroup | None = None,
    *,
    ood_signals_binding: bool = False,
    accept_threshold_ppm: int = THRESHOLD,
) -> SelectivePredictionPolicy:
    subject = target or group()
    record = evidence(subject)
    return SelectivePredictionPolicy(
        current_admission_id=ADMISSION_ID,
        current_split_root=SPLIT_ROOT,
        current_holdout_root=HOLDOUT_ROOT,
        current_evaluation_identity=EVALUATION_ID,
        evidence=(record,),
        bindings=(binding(subject, record=record, accept_threshold_ppm=accept_threshold_ppm),),
        ood_signals_binding=ood_signals_binding,
    )


def compact(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {"exit_code": 1, "failure_signature": "missing-edge"}
    payload.update(overrides)
    return payload


def task_input(
    *,
    family: ResidualTaskFamily = ResidualTaskFamily.FAILURE_ATTRIBUTION,
    risk: RiskClass = RiskClass.R2,
    features: dict[str, object] | None = None,
    allowed: tuple[str, ...] | None = None,
    question_id: str = "question:local:1",
) -> ResidualTaskInput:
    return ResidualTaskInput(
        task_family=family,
        question_id=question_id,
        repository_state_cid="repo:tree:abc",
        objective_cid="objective:vrif",
        task_cid="task:VRIF-014",
        policy_cid="policy:residual-v1",
        context_capsule_cid="capsule:bounded:1",
        compact_features=features if features is not None else compact(),
        allowed_outputs=allowed or (family.value, "ABSTAIN"),
        risk_class=risk,
        validation_policy="validator:local-expert@1",
        token_budget=256,
    )


def validator(
    family: ResidualTaskFamily = ResidualTaskFamily.FAILURE_ATTRIBUTION,
    *,
    accepted: bool = True,
) -> IndependentValidationReceipt:
    spec = family_spec_for(family)
    return IndependentValidationReceipt(
        validator_identity=spec.validator_identity,
        accepted=accepted,
        evidence_references=("validator:current-tree",),
    )


def lookup_entry(features: dict[str, object] | None = None) -> ExactLookupEntry:
    item = features if features is not None else compact()
    return ExactLookupEntry(
        feature_identity=local_feature_identity(item, FEATURE_NAMES),
        output_class="FAILURE_ATTRIBUTION",
        structured_payload=FAILURE_PAYLOAD,
        score_ppm=990_000,
        evidence_references=("cache:failure:1",),
    )


def timeout_rule(*, priority: int = 10) -> DeclarativeRule:
    return DeclarativeRule(
        rule_id="timeout-exit-code",
        priority=priority,
        predicates=(
            RulePredicate(feature="exit_code", kind=RulePredicateKind.INT_EQUALS, value=1),
            RulePredicate(
                feature="failure_signature",
                kind=RulePredicateKind.EQUALS,
                value="timeout",
            ),
        ),
        output_class="FAILURE_ATTRIBUTION",
        structured_payload=TIMEOUT_PAYLOAD,
        score_ppm=880_000,
        evidence_references=("rule:timeout",),
    )


def ood_reference(target: CalibrationGroup) -> ReferenceDistribution:
    record, _examples = admission()
    return ReferenceDistribution(
        group=target,
        admission_id=record.admission_id,
        admission_decision=TrainingAvailability.ADMITTED,
        allowed_families=(target.family,),
        allowed_schemas=(SCHEMA,),
        allowed_operations=(OPERATION,),
        allowed_repositories=(REPOSITORY,),
        allowed_effects=(EFFECT,),
        allowed_authorities=(CANDIDATE_ONLY_AUTHORITY,),
        allowed_capabilities=(CAPABILITY,),
        required_context_fields=(CONTEXT_FIELD,),
        feature_ranges=(
            FeatureRange(name="exit_code", minimum=0, maximum=2, observed_count=10),
            FeatureRange(
                name="failure_signature",
                allowed_values=("missing-edge", "timeout", "linear-only"),
                observed_count=10,
            ),
        ),
        example_identities=("example:1", "example:2"),
        statistic_identities=("stat:exit_code", "stat:failure_signature"),
        compact_statistics={"n_examples": 2, "feature_count": 2},
        family_distance_threshold_ppm=0,
    )


def ood_boundary(target: CalibrationGroup) -> BoundaryContract:
    return BoundaryContract(
        family=target.family,
        schema=SCHEMA,
        effects=(EFFECT,),
        authority_class=CANDIDATE_ONLY_AUTHORITY,
        repository=REPOSITORY,
        calibration_group_key=target.group_key,
        capabilities=(CAPABILITY,),
        required_context_fields=(CONTEXT_FIELD,),
        risk_ceiling=target.risk,
    )


def classification_expert(
    *,
    expert_class: ExpertClass = ExpertClass.C,
    lookup: tuple[ExactLookupEntry, ...] = (),
    rules: tuple[DeclarativeRule, ...] = (),
    coefficients: tuple[tuple[int, ...], ...] = ((8_000, 0),),
    intercepts: tuple[int, ...] = (8_000,),
    class_labels: tuple[str, ...] = ("missing_dependency_edge",),
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    bind_policy: bool = True,
    bind_ood: bool = False,
    policy_admits_ood: bool = False,
    risk: RiskClass = RiskClass.R2,
) -> LocalClassificationExpert:
    subject = group(risk=risk)
    return LocalClassificationExpert(
        task_family=ResidualTaskFamily.FAILURE_ATTRIBUTION,
        expert_class=expert_class,
        calibration_group=subject,
        feature_names=feature_names,
        lookup=lookup,
        rules=rules,
        class_labels=class_labels,
        linear_form=LinearForm.LOGISTIC,
        coefficients=coefficients,
        intercepts=intercepts,
        linear_threshold_ppm=1,
        selective_policy=(
            policy(subject, ood_signals_binding=policy_admits_ood) if bind_policy else None
        ),
        ood_reference=ood_reference(subject) if bind_ood else None,
        ood_boundary=ood_boundary(subject) if bind_ood else None,
        policy_admits_ood=policy_admits_ood,
        ood_schema=SCHEMA,
        ood_operation=OPERATION,
        ood_effects=(EFFECT,),
        ood_capabilities=(CAPABILITY,),
        ood_context_fields=(CONTEXT_FIELD,),
    )


def ranking_features(
    candidates: list[str],
    signals: dict[str, int],
) -> dict[str, object]:
    return {"ranking_candidates": candidates, "ranking_signals": signals}


def ranking_expert(
    *,
    expert_class: ExpertClass = ExpertClass.B,
    family: ResidualTaskFamily = ResidualTaskFamily.EVIDENCE_RANKING,
    risk: RiskClass = RiskClass.R2,
    lookup: tuple[ExactLookupEntry, ...] = (),
    ranking_weights: tuple[tuple[str, int], ...] = (),
    ranker: SmallRanker | None = None,
    bind_policy: bool = True,
) -> LocalRankingExpert:
    subject = group(family=family, risk=risk)
    return LocalRankingExpert(
        task_family=family,
        expert_class=expert_class,
        calibration_group=subject,
        feature_names=("ranking_candidates",),
        lookup=lookup,
        ranking_weights=ranking_weights,
        small_ranker=ranker or SmallRanker(),
        selective_policy=policy(subject) if bind_policy else None,
    )


def eval_case(
    item: ResidualTaskInput,
    expected: str,
    *,
    payload: dict[str, object] | None = None,
    critical: bool = False,
    adversarial: bool = False,
    partition: SplitPartition = SplitPartition.HELD_OUT,
    accepted: bool = True,
    identity: str = "",
) -> ExpertEvaluationCase:
    return ExpertEvaluationCase(
        task_input=item,
        expected_output_class=expected,
        expected_payload=payload or {},
        critical=critical,
        adversarial=adversarial,
        partition=partition,
        example_identity=identity or item.question_id,
        independent_validation=validator(item.task_family, accepted=accepted),
    )


def test_smallest_form_uses_exact_lookup_before_linear() -> None:
    features = compact()
    expert = classification_expert(
        lookup=(lookup_entry(features),),
        coefficients=((9_000, 0),),
        intercepts=(9_000,),
    )
    first = expert.predict(task_input(features=features), independent_validation=validator())
    second = expert.predict(task_input(features=features), independent_validation=validator())
    assert first.form is LocalExpertForm.EXACT_LOOKUP
    assert first.disposition is ExpertDisposition.ACCEPT
    assert first.candidate_only is True
    assert first.task_output.candidate_only is True
    assert first.task_output.structured_payload == FAILURE_PAYLOAD
    assert first.prediction_id == second.prediction_id
    assert first.cost.model_calls == 0
    assert first.cost.provider_invocations == 0
    assert first.structured_valid is True
    assert first.as_ir(task_input(features=features)).task_output.output_class == (
        "FAILURE_ATTRIBUTION"
    )
    class_a = classification_expert(
        expert_class=ExpertClass.A,
        lookup=(lookup_entry(features),),
        coefficients=((9_000, 0),),
        intercepts=(9_000,),
    )
    miss = class_a.predict(
        task_input(features=compact(failure_signature="linear-only")),
        independent_validation=validator(),
    )
    assert miss.form is LocalExpertForm.ABSTAIN
    assert miss.disposition is ExpertDisposition.ABSTAIN


def test_class_a_cannot_skip_to_linear_without_held_out_delta() -> None:
    with pytest.raises(ResidualIntelligenceError, match="routing-changing"):
        admit_local_expert_class(
            ResidualTaskFamily.FAILURE_ATTRIBUTION,
            ExpertClass.C,
            risk=RiskClass.R2,
            compared_class=ExpertClass.B,
            evidence_current=True,
        )
    with pytest.raises(ResidualIntelligenceError, match="smallest-form-order"):
        admit_local_expert_class(
            ResidualTaskFamily.FAILURE_ATTRIBUTION,
            ExpertClass.C,
            risk=RiskClass.R2,
            quality_delta_ppm=MIN_ROUTING_CHANGING_DELTA_PPM,
            routing_changing=True,
            evidence_current=True,
        )
    record, _examples = admission()
    admitted = admit_local_expert_class(
        ResidualTaskFamily.FAILURE_ATTRIBUTION,
        ExpertClass.B,
        risk=RiskClass.R2,
        quality_delta_ppm=MIN_ROUTING_CHANGING_DELTA_PPM,
        routing_changing=True,
        evidence_current=True,
        admission=record,
    )
    assert admitted.expert_class is ExpertClass.B


def test_classification_linear_after_lookup_and_rules() -> None:
    expert = classification_expert(
        lookup=(lookup_entry(),),
        rules=(timeout_rule(),),
        feature_names=FEATURE_NAMES,
        coefficients=((8_000, 0),),
        intercepts=(8_000,),
    )
    timeout = expert.predict(
        task_input(features=compact(failure_signature="timeout")),
        independent_validation=validator(),
    )
    assert timeout.form is LocalExpertForm.DECLARATIVE_RULE
    assert timeout.task_output.structured_payload == TIMEOUT_PAYLOAD
    linear = expert.predict(
        task_input(features=compact(failure_signature="linear-only")),
        independent_validation=validator(),
    )
    assert linear.form is LocalExpertForm.LINEAR_LOGISTIC
    assert linear.disposition is ExpertDisposition.ACCEPT
    assert linear.task_output.structured_payload["failure_class"] == "missing_dependency_edge"
    rebuilt = LocalClassificationExpert.from_dict(expert.to_dict())
    assert rebuilt.expert_version == expert.expert_version
    assert rebuilt.predict(
        task_input(features=compact(failure_signature="linear-only")),
        independent_validation=validator(),
    ).prediction_id == linear.prediction_id


def test_ranking_is_stable_and_small_ranker_breaks_ties() -> None:
    features = ranking_features(
        ["ev:b", "ev:a", "ev:c"], {"ev:a": 900_000, "ev:b": 900_000, "ev:c": 100_000}
    )
    tied = ranking_expert(expert_class=ExpertClass.B)
    abstained = tied.predict(
        task_input(family=ResidualTaskFamily.EVIDENCE_RANKING, features=features),
        independent_validation=validator(ResidualTaskFamily.EVIDENCE_RANKING),
    )
    assert abstained.form is LocalExpertForm.ABSTAIN
    assert "ranking_score_tie" in abstained.task_output.reason_codes
    distinct = ranking_features(
        ["ev:b", "ev:a", "ev:c"], {"ev:a": 900_000, "ev:b": 500_000, "ev:c": 100_000}
    )
    ranked = tied.predict(
        task_input(family=ResidualTaskFamily.EVIDENCE_RANKING, features=distinct),
        independent_validation=validator(ResidualTaskFamily.EVIDENCE_RANKING),
    )
    assert ranked.form is LocalExpertForm.DETERMINISTIC_RANKING
    assert [item.reference_id for item in ranked.ranking] == ["ev:a", "ev:b", "ev:c"]
    assert ranked.task_output.structured_payload["scores_ppm"] == sorted(
        ranked.task_output.structured_payload["scores_ppm"],
        reverse=True,
    )
    small = ranking_expert(
        expert_class=ExpertClass.D,
        ranker=SmallRanker(candidate_weights=(("ev:a", 50_000), ("ev:b", 10_000))),
    )
    broken = small.predict(
        task_input(family=ResidualTaskFamily.EVIDENCE_RANKING, features=features),
        independent_validation=validator(ResidualTaskFamily.EVIDENCE_RANKING),
    )
    assert broken.form is LocalExpertForm.SMALL_RANKER
    assert [item.reference_id for item in broken.ranking] == ["ev:a", "ev:b", "ev:c"]
    assert broken.disposition is ExpertDisposition.ACCEPT
    assert broken.candidate_only is True


def test_batch_preserves_order_and_avoids_model_calls() -> None:
    expert = classification_expert(lookup=(lookup_entry(),), rules=(timeout_rule(),))
    request = BatchedExpertRequest(
        task_inputs=(
            task_input(question_id="question:local:batch-1"),
            task_input(
                features=compact(failure_signature="timeout"),
                question_id="question:local:batch-2",
            ),
            task_input(
                features=compact(failure_signature="linear-only"),
                question_id="question:local:batch-3",
            ),
        ),
        independent_validations=(validator(), validator(), validator()),
    )
    predictions = expert.predict_batch(request)
    assert [item.form for item in predictions] == [
        LocalExpertForm.EXACT_LOOKUP,
        LocalExpertForm.DECLARATIVE_RULE,
        LocalExpertForm.LINEAR_LOGISTIC,
    ]
    assert all(item.cost.model_calls == 0 for item in predictions)
    assert all(item.cost.provider_invocations == 0 for item in predictions)
    assert all(item.candidate_only is True for item in predictions)
    rebuilt = BatchedExpertRequest.from_dict(request.to_dict())
    assert rebuilt.request_id == request.request_id


def test_calibrated_abstention_is_group_keyed() -> None:
    expert = classification_expert(lookup=(lookup_entry(),))
    accepted = expert.predict(task_input(), independent_validation=validator())
    assert accepted.disposition is ExpertDisposition.ACCEPT
    assert accepted.abstention is not None
    assert accepted.abstention.group_key == expert.calibration_group.group_key
    low = classification_expert(
        lookup=(
            ExactLookupEntry(
                feature_identity=local_feature_identity(compact(), FEATURE_NAMES),
                output_class="FAILURE_ATTRIBUTION",
                structured_payload=FAILURE_PAYLOAD,
                score_ppm=100_000,
                evidence_references=("cache:low",),
            ),
        )
    )
    abstained = low.predict(task_input(), independent_validation=validator())
    assert abstained.disposition is ExpertDisposition.ABSTAIN
    assert "below_group_threshold" in abstained.abstention.reason_codes
    unbound = classification_expert(lookup=(lookup_entry(),), bind_policy=False)
    pending = unbound.predict(task_input(), independent_validation=validator())
    assert pending.disposition is ExpertDisposition.ABSTAIN
    assert "current_evidence_required" in pending.task_output.reason_codes


def test_independent_validator_decides_acceptance() -> None:
    expert = classification_expert(lookup=(lookup_entry(),))
    required = expert.predict(task_input())
    assert required.disposition is ExpertDisposition.VALIDATION_REQUIRED
    rejected = expert.predict(task_input(), independent_validation=validator(accepted=False))
    assert rejected.disposition is ExpertDisposition.VALIDATION_REQUIRED
    accepted = expert.predict(task_input(), independent_validation=validator(accepted=True))
    assert accepted.disposition is ExpertDisposition.ACCEPT
    assert accepted.independent_validator_identity.startswith("validator:")


def test_ood_bound_signal_abstains_and_in_boundary_remains_eligible() -> None:
    subject = group()
    expert = classification_expert(
        lookup=(lookup_entry(),),
        bind_ood=True,
        policy_admits_ood=True,
    )
    observation = observation_from_task_input(
        task_input(),
        schema=SCHEMA,
        operation=OPERATION,
        repository=REPOSITORY,
        effects=(EFFECT,),
        authority_class=CANDIDATE_ONLY_AUTHORITY,
        calibration_group_key=subject.group_key,
        context_fields=(CONTEXT_FIELD,),
        capabilities=(CAPABILITY,),
        detection_available=True,
        context_complete=True,
    )
    in_boundary = assess_out_of_distribution(
        observation,
        reference=expert.ood_reference,
        boundary=expert.ood_boundary,
        policy_admits_ood=True,
    )
    assert in_boundary.in_boundary_eligible is True
    assert in_boundary.safety_established is False
    accepted = expert.predict(task_input(), independent_validation=validator())
    assert accepted.disposition is ExpertDisposition.ACCEPT
    assert accepted.ood_assessment is not None
    assert accepted.ood_assessment.in_boundary_eligible is True
    ood = expert.predict(
        task_input(features=compact(exit_code=99)),
        independent_validation=validator(),
    )
    assert ood.disposition is ExpertDisposition.OUT_OF_DISTRIBUTION
    assert ood.task_output.abstained is True
    assert ood.ood_assessment is not None
    assert ood.ood_assessment.bound_ood is True
    assert OODSignalKind.FEATURE_RANGE in {item.kind for item in ood.ood_assessment.signals}
    advisory = classification_expert(
        lookup=(lookup_entry(),),
        bind_ood=True,
        policy_admits_ood=False,
    )
    still_local = advisory.predict(
        task_input(features=compact(exit_code=99, failure_signature="missing-edge")),
        independent_validation=validator(),
    )
    assert still_local.ood_assessment is not None
    assert still_local.ood_assessment.advisory_ood is True
    assert still_local.ood_assessment.bound_ood is False
    assert still_local.disposition is not ExpertDisposition.OUT_OF_DISTRIBUTION


def test_held_out_evaluation_preserves_denominators() -> None:
    expert = classification_expert(lookup=(lookup_entry(),), rules=(timeout_rule(),))
    cases = (
        eval_case(task_input(), "FAILURE_ATTRIBUTION", payload=FAILURE_PAYLOAD),
        eval_case(
            task_input(features=compact(failure_signature="timeout")),
            "FAILURE_ATTRIBUTION",
            payload=TIMEOUT_PAYLOAD,
        ),
        eval_case(
            task_input(features=compact(failure_signature="linear-only")),
            "FAILURE_ATTRIBUTION",
        ),
        eval_case(
            task_input(family=ResidualTaskFamily.TASK_CLASSIFICATION, features={"label_candidates": ["x"]}),
            "ABSTAIN",
        ),
        eval_case(
            task_input(features=compact(failure_signature="missing-edge")),
            "FAILURE_ATTRIBUTION",
            critical=True,
            adversarial=True,
            partition=SplitPartition.ADVERSARIAL,
        ),
    )
    report = expert.evaluate(cases)
    assert report.example_count == 5
    assert report.exact_lookup_count == 2
    assert report.rule_count == 1
    assert report.linear_count == 1
    assert report.reject_form_count == 1
    assert report.reject_input_count == 1
    assert report.held_out_count == 4
    assert report.adversarial_count == 1
    assert report.model_calls == 0
    assert report.provider_invocations == 0
    assert report.avoided_model_calls == report.example_count
    assert report.avoided_remote_calls == report.example_count
    assert (
        report.exact_lookup_count
        + report.rule_count
        + report.ranking_count
        + report.linear_count
        + report.small_ranker_count
        + report.cascade_abstain_count
        + report.reject_form_count
        == report.example_count
    )
    assert (
        report.accept_count
        + report.abstain_count
        + report.reject_input_count
        + report.ood_count
        + report.capability_unavailable_count
        + report.validation_required_count
        == report.example_count
    )
    assert report.structured_valid_count + report.structured_invalid_count == report.example_count
    assert report.coverage_ppm == (report.accept_count * 1_000_000) // report.example_count
    assert report.group_key == expert.calibration_group.group_key
    rebuilt = ExpertEvaluation.from_dict(report.to_dict())
    assert rebuilt == report


def test_candidate_only_and_r4_remain_proposal_tier() -> None:
    expert = classification_expert(lookup=(lookup_entry(),))
    accepted = expert.predict(task_input(), independent_validation=validator())
    with pytest.raises(ResidualIntelligenceError, match="candidate_only"):
        LocalExpertPrediction(
            task_output=accepted.task_output,
            form=accepted.form,
            feature_identity=accepted.feature_identity,
            cost=accepted.cost,
            disposition=accepted.disposition,
            abstention=accepted.abstention,
            structured_valid=True,
            independent_validator_identity=accepted.independent_validator_identity,
            candidate_only=False,
        )
    with pytest.raises(ResidualIntelligenceError, match="unsupported_family_risk"):
        classification_expert(lookup=(lookup_entry(),), risk=RiskClass.R4)
    ranking = ranking_expert(
        family=ResidualTaskFamily.PLAN_BRANCH_RANKING,
        risk=RiskClass.R4,
    )
    proposal = ranking.predict(
        task_input(
            family=ResidualTaskFamily.PLAN_BRANCH_RANKING,
            risk=RiskClass.R4,
            features=ranking_features(["ev:a", "ev:b"], {"ev:a": 900_000, "ev:b": 100_000}),
        ),
        independent_validation=validator(ResidualTaskFamily.PLAN_BRANCH_RANKING),
    )
    assert proposal.disposition is ExpertDisposition.VALIDATION_REQUIRED
    assert proposal.task_output.abstained is False
    assert proposal.candidate_only is True


def test_training_unavailable_blocks_fit_and_distillation() -> None:
    blocked, _examples = admission(admitted=False)
    expert = classification_expert(bind_policy=False)
    train = eval_case(
        task_input(features=compact(exit_code=1, failure_signature="linear-only")),
        "FAILURE_ATTRIBUTION",
        partition=SplitPartition.TRAIN,
    )
    with pytest.raises(ResidualIntelligenceError, match="training_unavailable"):
        expert.fit(admission=blocked, cases=(train,))
    with pytest.raises(ResidualIntelligenceError, match="training_unavailable"):
        distill_classification_expert(expert, admission=blocked, cases=(train,))
    with pytest.raises(ResidualIntelligenceError, match="gpu_seconds"):
        DistillationBudget(gpu_seconds=1)
    assert MAX_LOCAL_GPU_SECONDS == 0
    record, _examples = admission()
    too_many = (train,) * (MAX_LOCAL_EXAMPLES + 1)
    with pytest.raises(ResidualIntelligenceError, match="50000"):
        expert.fit(admission=record, cases=too_many)


def test_distillation_selects_smallest_reliable_form() -> None:
    record, _examples = admission()
    seed = classification_expert(
        lookup=(),
        rules=(timeout_rule(),),
        bind_policy=True,
    )
    train = (
        eval_case(
            task_input(question_id="question:train-lookup"),
            "FAILURE_ATTRIBUTION",
            payload=FAILURE_PAYLOAD,
            partition=SplitPartition.TRAIN,
            identity="example:train-lookup",
        ),
        eval_case(
            task_input(
                features=compact(failure_signature="timeout"),
                question_id="question:train-rule",
            ),
            "FAILURE_ATTRIBUTION",
            payload=TIMEOUT_PAYLOAD,
            partition=SplitPartition.TRAIN,
            identity="example:train-rule",
        ),
        eval_case(
            task_input(
                features=compact(failure_signature="linear-only"),
                question_id="question:train-linear",
            ),
            "FAILURE_ATTRIBUTION",
            partition=SplitPartition.TRAIN,
            identity="example:train-linear",
        ),
    )
    held = (
        eval_case(
            task_input(question_id="question:hold-lookup"),
            "FAILURE_ATTRIBUTION",
            payload=FAILURE_PAYLOAD,
            identity="example:hold-lookup",
        ),
        eval_case(
            task_input(
                features=compact(failure_signature="timeout"),
                question_id="question:hold-rule",
            ),
            "FAILURE_ATTRIBUTION",
            payload=TIMEOUT_PAYLOAD,
            identity="example:hold-rule",
        ),
        eval_case(
            task_input(
                features=compact(failure_signature="linear-only"),
                question_id="question:hold-linear",
            ),
            "FAILURE_ATTRIBUTION",
            adversarial=True,
            partition=SplitPartition.ADVERSARIAL,
            identity="example:hold-linear",
        ),
    )
    result = distill_classification_expert(
        seed,
        admission=record,
        cases=train + held,
        requested_class=ExpertClass.C,
    )
    assert result.candidate_only is True
    assert result.training_unavailable is False
    assert result.classification_expert is not None
    assert result.evaluation.zero_critical_false_accepts is True
    assert result.evaluation.adversarial_count >= 1
    assert result.selected_class in {ExpertClass.A, ExpertClass.B, ExpertClass.C}
    if result.selected_class is ExpertClass.A:
        assert result.classification_expert.coefficients == ()
    rebuilt = DistillationResult.from_dict(result.to_dict())
    assert rebuilt.result_id == result.result_id
    rank_seed = ranking_expert(expert_class=ExpertClass.D)
    rank_train = eval_case(
        task_input(
            family=ResidualTaskFamily.EVIDENCE_RANKING,
            features=ranking_features(["ev:a", "ev:b"], {"ev:a": 9, "ev:b": 1}),
            question_id="question:rank-train",
        ),
        "EVIDENCE_RANKING",
        payload=ranking_payload(
            ResidualTaskFamily.EVIDENCE_RANKING,
            (
                RankingItem(reference_id="ev:a", score_ppm=9),
                RankingItem(reference_id="ev:b", score_ppm=1),
            ),
        ),
        partition=SplitPartition.TRAIN,
        identity="example:rank-train",
    )
    rank_hold = eval_case(
        task_input(
            family=ResidualTaskFamily.EVIDENCE_RANKING,
            features=ranking_features(["ev:a", "ev:b"], {"ev:a": 9, "ev:b": 1}),
            question_id="question:rank-hold",
        ),
        "EVIDENCE_RANKING",
        payload=ranking_payload(
            ResidualTaskFamily.EVIDENCE_RANKING,
            (
                RankingItem(reference_id="ev:a", score_ppm=9),
                RankingItem(reference_id="ev:b", score_ppm=1),
            ),
        ),
        identity="example:rank-hold",
    )
    ranked = distill_ranking_expert(
        rank_seed,
        admission=record,
        cases=(rank_train, rank_hold),
        requested_class=ExpertClass.B,
    )
    assert ranked.ranking_expert is not None
    assert ranked.selected_class in {ExpertClass.A, ExpertClass.B}
    assert ranked.evaluation.zero_critical_false_accepts is True


def test_classification_payload_and_family_gate() -> None:
    payload = classification_payload(
        ResidualTaskFamily.TASK_CLASSIFICATION, "FAILURE_ATTRIBUTION"
    )
    assert payload["label"] == "FAILURE_ATTRIBUTION"
    with pytest.raises(ResidualIntelligenceError, match="unsupported_semantic_kind"):
        LocalRankingExpert(
            task_family=ResidualTaskFamily.FAILURE_ATTRIBUTION,
            expert_class=ExpertClass.B,
            calibration_group=group(),
            feature_names=("exit_code",),
        )
    with pytest.raises(ResidualIntelligenceError, match="unsupported_semantic_kind"):
        LocalClassificationExpert(
            task_family=ResidualTaskFamily.EVIDENCE_RANKING,
            expert_class=ExpertClass.A,
            calibration_group=group(family=ResidualTaskFamily.EVIDENCE_RANKING),
            feature_names=("ranking_candidates",),
        )
