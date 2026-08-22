from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.baselines import (
    BASELINE_CASCADE_ORDER,
    LOGIT_CLAMP,
    MAX_COEFFICIENT,
    MAX_LINEAR_EXAMPLES,
    BaselineCostReceipt,
    BaselineEvaluation,
    BaselineEvaluationCase,
    BaselinePrediction,
    BaselineRoute,
    DeclarativeRule,
    DeterministicResidualExpert,
    ExactLookupEntry,
    LinearForm,
    LinearResidualExpert,
    ProcedureBinding,
    RankingItem,
    RulePredicate,
    RulePredicateKind,
    extract_stable_features,
    logistic_ppm,
    stable_feature_identity,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    AuthorityViolationError,
    ExpertDisposition,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    UnknownFieldError,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.residual_ir import ResidualTaskInput

from .helpers import admission

FEATURE_NAMES = ("exit_code", "failure_signature")
CALIBRATION_GROUP = "failure:python:R2:fixture"
FAILURE_PAYLOAD = {
    "failure_class": "missing_dependency_edge",
    "recommended_action": "expand_context_reference",
    "reference_ids": ["dependency:1"],
}
ALT_PAYLOAD = {
    "failure_class": "provider_timeout",
    "recommended_action": "retry_local_validator",
    "reference_ids": ["timeout:1"],
}


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
) -> ResidualTaskInput:
    output_class = family.value
    return ResidualTaskInput(
        task_family=family,
        question_id="question:baseline:1",
        repository_state_cid="repo:tree:abc",
        objective_cid="objective:vrif",
        task_cid="task:VRIF-009",
        policy_cid="policy:residual-v1",
        context_capsule_cid="capsule:bounded:1",
        compact_features=features if features is not None else compact(),
        allowed_outputs=allowed or (output_class, "ABSTAIN"),
        risk_class=risk,
        validation_policy="validator:baseline@1",
        token_budget=256,
    )


def lookup_entry(features: dict[str, object] | None = None) -> ExactLookupEntry:
    item = features if features is not None else compact()
    return ExactLookupEntry(
        feature_identity=stable_feature_identity(item, FEATURE_NAMES),
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
        structured_payload=ALT_PAYLOAD,
        score_ppm=880_000,
        evidence_references=("rule:timeout",),
    )


def missing_edge_rule(*, priority: int = 5) -> DeclarativeRule:
    return DeclarativeRule(
        rule_id="missing-edge",
        priority=priority,
        predicates=(
            RulePredicate(
                feature="failure_signature",
                kind=RulePredicateKind.EQUALS,
                value="missing-edge",
            ),
        ),
        output_class="FAILURE_ATTRIBUTION",
        structured_payload=FAILURE_PAYLOAD,
        score_ppm=870_000,
        evidence_references=("rule:missing-edge",),
    )


def procedure_binding() -> ProcedureBinding:
    return ProcedureBinding(
        procedure_root="procedure:failure-attribution@1",
        output_class="FAILURE_ATTRIBUTION",
        structured_payload=FAILURE_PAYLOAD,
        evidence_references=("procedure:root:1",),
        score_ppm=1_000_000,
    )


def deterministic(
    *,
    lookup: tuple[ExactLookupEntry, ...] = (),
    rules: tuple[DeclarativeRule, ...] = (),
    procedures: tuple[ProcedureBinding, ...] = (),
    ranking_weights: tuple[tuple[str, int], ...] = (),
    family: ResidualTaskFamily = ResidualTaskFamily.FAILURE_ATTRIBUTION,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> DeterministicResidualExpert:
    return DeterministicResidualExpert(
        task_family=family,
        calibration_group=CALIBRATION_GROUP,
        feature_names=feature_names,
        lookup=lookup,
        rules=rules,
        procedures=procedures,
        ranking_weights=ranking_weights,
    )


def linear_expert(
    expert: DeterministicResidualExpert | None = None,
    *,
    form: LinearForm = LinearForm.LOGISTIC,
    feature_names: tuple[str, ...] = ("exit_code", "flag"),
    class_labels: tuple[str, ...] = ("missing_dependency_edge",),
    coefficients: tuple[tuple[int, ...], ...] = ((2_000, 2_000),),
    intercepts: tuple[int, ...] = (0,),
    threshold_ppm: int = 500_000,
) -> LinearResidualExpert:
    return LinearResidualExpert(
        deterministic=expert or deterministic(),
        form=form,
        feature_names=feature_names,
        class_labels=class_labels,
        coefficients=coefficients,
        intercepts=intercepts,
        threshold_ppm=threshold_ppm,
    )


def test_cascade_order_is_exact_first() -> None:
    assert BASELINE_CASCADE_ORDER == (
        BaselineRoute.EXACT_LOOKUP,
        BaselineRoute.VERIFIED_PROCEDURE,
        BaselineRoute.DETERMINISTIC_RULE,
        BaselineRoute.DETERMINISTIC_RANKING,
        BaselineRoute.LINEAR_LOGISTIC,
    )


def test_exact_lookup_precedes_rules_and_linear() -> None:
    features = compact()
    expert = linear_expert(
        deterministic(
            lookup=(lookup_entry(features),),
            rules=(missing_edge_rule(priority=1_000_000),),
        ),
        feature_names=FEATURE_NAMES,
        class_labels=("missing_dependency_edge",),
        coefficients=((9_000, 9_000),),
    )
    first = expert.predict(task_input(features=features))
    second = expert.predict(task_input(features=features))
    assert first.route is BaselineRoute.EXACT_LOOKUP
    assert first.disposition is ExpertDisposition.ACCEPT
    assert first.candidate_only is True
    assert first.task_output.candidate_only is True
    assert first.task_output.structured_payload == FAILURE_PAYLOAD
    assert first.prediction_id == second.prediction_id
    assert first.feature_identity == stable_feature_identity(features, FEATURE_NAMES)
    assert first.cost.model_calls == 0
    assert first.cost.provider_invocations == 0
    assert first.as_ir(task_input(features=features)).task_output.output_class == (
        "FAILURE_ATTRIBUTION"
    )


def test_procedure_route_when_preconditions_satisfied() -> None:
    features = compact(
        procedure_root="procedure:failure-attribution@1",
        procedure_answer_available=True,
        procedure_preconditions_satisfied=True,
    )
    expert = deterministic(procedures=(procedure_binding(),), rules=(missing_edge_rule(),))
    prediction = expert.predict(task_input(features=features))
    assert prediction.route is BaselineRoute.VERIFIED_PROCEDURE
    assert prediction.disposition is ExpertDisposition.ACCEPT
    assert "verified_procedure" in prediction.task_output.reason_codes
    assert prediction.cost.invoked_model_or_provider is False


def test_procedure_precondition_failure_does_not_invoke_model_or_linear() -> None:
    features = compact(
        exit_code=1,
        flag=1,
        procedure_root="procedure:failure-attribution@1",
        procedure_answer_available=True,
        procedure_preconditions_satisfied=False,
    )
    expert = linear_expert(
        deterministic(procedures=(procedure_binding(),), rules=(missing_edge_rule(),)),
        feature_names=("exit_code", "flag"),
        coefficients=((8_000, 8_000),),
        intercepts=(8_000,),
        threshold_ppm=1,
    )
    prediction = expert.predict(task_input(features=features))
    assert prediction.route is BaselineRoute.ABSTAIN
    assert prediction.disposition is ExpertDisposition.ABSTAIN
    assert prediction.task_output.reason_codes == ("procedure_precondition_failure",)
    assert prediction.cost.model_calls == 0
    assert prediction.cost.provider_invocations == 0
    assert prediction.cost.remote_input_tokens == 0
    assert prediction.cost.avoided_remote_calls == 1
    assert prediction.cost.avoided_strong_calls == 1


def test_declarative_rules_are_deterministic() -> None:
    features = compact(failure_signature="timeout")
    broad = DeclarativeRule(
        rule_id="any-nonzero-exit",
        priority=1,
        predicates=(
            RulePredicate(feature="exit_code", kind=RulePredicateKind.INT_EQUALS, value=1),
        ),
        output_class="FAILURE_ATTRIBUTION",
        structured_payload=FAILURE_PAYLOAD,
        score_ppm=100_000,
        evidence_references=("rule:broad",),
    )
    expert = deterministic(rules=(timeout_rule(priority=20), broad))
    first = expert.predict(task_input(features=features))
    second = expert.predict(task_input(features=features))
    assert first.route is BaselineRoute.DETERMINISTIC_RULE
    assert first.task_output.structured_payload == ALT_PAYLOAD
    assert first.prediction_id == second.prediction_id
    rebuilt = DeterministicResidualExpert.from_dict(expert.to_dict())
    assert rebuilt == expert
    assert rebuilt.predict(task_input(features=features)).prediction_id == first.prediction_id


def test_deterministic_ranking_is_stable() -> None:
    features = compact(
        ranking_candidates=["ev:b", "ev:a", "ev:c"],
        ranking_signals={"ev:a": 50, "ev:b": 50, "ev:c": 10},
    )
    expert = deterministic(
        family=ResidualTaskFamily.EVIDENCE_RANKING,
        ranking_weights=(("ev:c", 1),),
    )
    prediction = expert.predict(
        task_input(
            family=ResidualTaskFamily.EVIDENCE_RANKING,
            features=features,
        )
    )
    assert prediction.route is BaselineRoute.DETERMINISTIC_RANKING
    assert [item.reference_id for item in prediction.ranking] == ["ev:a", "ev:b", "ev:c"]
    assert prediction.task_output.structured_payload["scores_ppm"] == [
        item.score_ppm for item in prediction.ranking
    ]
    assert prediction.task_output.structured_payload["scores_ppm"] == sorted(
        prediction.task_output.structured_payload["scores_ppm"],
        reverse=True,
    )
    again = expert.predict(
        task_input(family=ResidualTaskFamily.EVIDENCE_RANKING, features=features)
    )
    assert again.prediction_id == prediction.prediction_id


def test_ranking_precedes_linear_when_candidates_are_present() -> None:
    features = compact(
        exit_code=1,
        flag=1,
        ranking_candidates=["ev:1", "ev:2"],
        ranking_signals={"ev:1": 9, "ev:2": 1},
    )
    expert = linear_expert(
        deterministic(family=ResidualTaskFamily.EVIDENCE_RANKING),
        feature_names=("exit_code", "flag"),
        coefficients=((9_000, 9_000),),
    )
    prediction = expert.predict(
        task_input(family=ResidualTaskFamily.EVIDENCE_RANKING, features=features)
    )
    assert prediction.route is BaselineRoute.DETERMINISTIC_RANKING


def test_bounded_linear_and_logistic_integer_coefficients() -> None:
    assert logistic_ppm(0) == 500_000
    assert logistic_ppm(-2_000) < logistic_ppm(0) < logistic_ppm(2_000)
    assert logistic_ppm(LOGIT_CLAMP) == 1_000_000
    assert logistic_ppm(-LOGIT_CLAMP) == 0
    expert = linear_expert(form=LinearForm.LINEAR, threshold_ppm=1)
    vector = (1, 1)
    assert expert.score_vector(vector) == expert.score_vector(vector)
    prediction = expert.predict(task_input(features=compact(exit_code=1, flag=1)))
    assert prediction.route is BaselineRoute.LINEAR_LOGISTIC
    assert prediction.task_output.reason_codes == ("linear_logistic",)
    logistic = linear_expert(form=LinearForm.LOGISTIC, threshold_ppm=1)
    logistic_prediction = logistic.predict(task_input(features=compact(exit_code=1, flag=1)))
    assert logistic_prediction.route is BaselineRoute.LINEAR_LOGISTIC
    assert logistic_prediction.task_output.confidence_or_score == logistic_ppm(4_000)
    assert logistic.score_vector(vector) == (logistic_ppm(4_000),)
    with pytest.raises(ResidualIntelligenceError, match="coefficient"):
        linear_expert(coefficients=((MAX_COEFFICIENT + 1, 0),))


def test_stable_feature_identity_is_canonical() -> None:
    left = compact()
    right = compact(procedure_root="procedure:other")
    assert extract_stable_features(left, FEATURE_NAMES) == extract_stable_features(
        right, FEATURE_NAMES
    )
    assert stable_feature_identity(left, FEATURE_NAMES) == stable_feature_identity(
        right, FEATURE_NAMES
    )
    changed = compact(exit_code=2)
    assert stable_feature_identity(left, FEATURE_NAMES) != stable_feature_identity(
        changed, FEATURE_NAMES
    )
    with pytest.raises(ResidualIntelligenceError, match="private body"):
        stable_feature_identity({"source_text": "def secret(): pass", "exit_code": 1}, FEATURE_NAMES)


def test_linear_fit_requires_admitted_corpus() -> None:
    blocked, _examples = admission(admitted=False)
    unfitted = LinearResidualExpert(
        deterministic=deterministic(),
        form=LinearForm.LOGISTIC,
        feature_names=("exit_code", "flag"),
        class_labels=("missing_dependency_edge",),
    )
    train_case = BaselineEvaluationCase(
        task_input=task_input(features=compact(exit_code=1, flag=1)),
        expected_output_class="FAILURE_ATTRIBUTION",
    )
    with pytest.raises(ResidualIntelligenceError, match="training_unavailable"):
        unfitted.fit(admission=blocked, cases=(train_case,))
    record, _examples = admission()
    fitted = unfitted.fit(
        admission=record,
        cases=(
            train_case,
            BaselineEvaluationCase(
                task_input=task_input(features=compact(exit_code=0, flag=0)),
                expected_output_class="ABSTAIN",
            ),
        ),
    )
    assert fitted.fitted is True
    assert fitted.checkpoint_count == 1
    assert fitted.admission_id == record.admission_id
    assert fitted.coefficients
    too_many = (train_case,) * (MAX_LINEAR_EXAMPLES + 1)
    with pytest.raises(ResidualIntelligenceError, match="10000"):
        unfitted.fit(admission=record, cases=too_many)
    with pytest.raises(ResidualIntelligenceError, match="cpu_seconds"):
        unfitted.fit(admission=record, cases=(train_case,), cpu_seconds=1_801)


def test_evaluation_preserves_denominators() -> None:
    features = compact()
    expert = linear_expert(
        deterministic(
            lookup=(lookup_entry(features),),
            rules=(timeout_rule(),),
            procedures=(procedure_binding(),),
        ),
        feature_names=("exit_code", "flag"),
        coefficients=((4_000, 4_000),),
        intercepts=(0,),
        form=LinearForm.LINEAR,
        threshold_ppm=1,
    )
    cases = (
        BaselineEvaluationCase(task_input=task_input(features=features), expected_output_class="FAILURE_ATTRIBUTION"),
        BaselineEvaluationCase(
            task_input=task_input(features=compact(failure_signature="timeout")),
            expected_output_class="FAILURE_ATTRIBUTION",
        ),
        BaselineEvaluationCase(
            task_input=task_input(
                features=compact(
                    procedure_root="procedure:failure-attribution@1",
                    procedure_answer_available=True,
                    procedure_preconditions_satisfied=True,
                    failure_signature="other",
                )
            ),
            expected_output_class="FAILURE_ATTRIBUTION",
        ),
        BaselineEvaluationCase(
            task_input=task_input(features=compact(exit_code=1, flag=1, failure_signature="linear-only")),
            expected_output_class="FAILURE_ATTRIBUTION",
        ),
        BaselineEvaluationCase(
            task_input=task_input(features=compact(input_valid=False, failure_signature="invalid")),
            expected_output_class="ABSTAIN",
        ),
        BaselineEvaluationCase(
            task_input=task_input(features=compact(critical_boundary=True)),
            expected_output_class="ABSTAIN",
            critical=True,
        ),
    )
    report = expert.evaluate(cases)
    assert report.example_count == 6
    assert report.exact_lookup_count == 1
    assert report.rule_count == 1
    assert report.procedure_count == 1
    assert report.linear_count == 1
    assert report.ranking_count == 0
    assert report.reject_input_count == 1
    assert report.abstain_count == 1
    assert report.false_accept_count == 0
    assert report.critical_false_accept_count == 0
    assert (
        report.exact_lookup_count
        + report.procedure_count
        + report.rule_count
        + report.ranking_count
        + report.linear_count
        + report.abstain_route_count
        + report.reject_route_count
        == report.example_count
    )
    assert (
        report.accept_count
        + report.abstain_count
        + report.reject_input_count
        + report.validation_required_count
        == report.example_count
    )
    assert report.model_calls == 0
    assert report.provider_invocations == 0
    assert report.avoided_model_calls == report.example_count
    assert report.avoided_remote_calls == report.example_count
    assert report.coverage_ppm == (report.accept_count * 1_000_000) // report.example_count
    assert report.precision_ppm == (
        ((report.accept_count - report.false_accept_count) * 1_000_000) // report.accept_count
    )
    assert report.abstention_rate_ppm == (
        (report.abstain_count * 1_000_000) // report.example_count
    )
    rebuilt = BaselineEvaluation.from_dict(report.to_dict())
    assert rebuilt == report


def test_candidate_only_and_critical_boundary_abstention() -> None:
    features = compact(critical_boundary=True)
    expert = deterministic(lookup=(lookup_entry(compact()),))
    prediction = expert.predict(task_input(features=features))
    assert prediction.route is BaselineRoute.ABSTAIN
    assert prediction.disposition is ExpertDisposition.ABSTAIN
    assert "critical_boundary_abstention" in prediction.task_output.reason_codes
    assert prediction.candidate_only is True
    with pytest.raises(ResidualIntelligenceError, match="candidate_only"):
        BaselinePrediction(
            task_output=prediction.task_output,
            route=prediction.route,
            feature_identity=prediction.feature_identity,
            cost=prediction.cost,
            disposition=prediction.disposition,
            candidate_only=False,
        )


def test_r4_r5_remain_validation_required() -> None:
    features = compact()
    expert = deterministic(lookup=(lookup_entry(features),))
    for risk in (RiskClass.R4, RiskClass.R5):
        prediction = expert.predict(task_input(features=features, risk=risk))
        assert prediction.route is BaselineRoute.EXACT_LOOKUP
        assert prediction.disposition is ExpertDisposition.VALIDATION_REQUIRED
        assert "VALIDATION_REQUIRED" in prediction.task_output.reason_codes
        assert prediction.task_output.abstained is False
        assert prediction.as_ir(task_input(features=features, risk=risk)).ir_id


def test_rules_and_lookup_reject_private_bodies() -> None:
    with pytest.raises(ResidualIntelligenceError, match="private body"):
        RulePredicate(feature="source_text", kind=RulePredicateKind.PRESENT)
    with pytest.raises(ResidualIntelligenceError, match="private body"):
        ExactLookupEntry(
            feature_identity="features:1",
            output_class="FAILURE_ATTRIBUTION",
            structured_payload={"chain_of_thought": "hidden"},
            score_ppm=1,
        )
    with pytest.raises(AuthorityViolationError):
        ExactLookupEntry(
            feature_identity="features:1",
            output_class="FAILURE_ATTRIBUTION",
            structured_payload={"completed": True},
            score_ppm=1,
        )


def test_early_exit_cost_receipt_has_zero_provider_invocations() -> None:
    prediction = deterministic(lookup=(lookup_entry(),)).predict(task_input())
    assert prediction.route is BaselineRoute.EXACT_LOOKUP
    assert prediction.cost.model_calls == 0
    assert prediction.cost.provider_invocations == 0
    with pytest.raises(ResidualIntelligenceError, match="model or provider"):
        BaselineCostReceipt(
            route=BaselineRoute.EXACT_LOOKUP,
            feature_ops=1,
            avoided_remote_calls=1,
            avoided_strong_calls=1,
            model_calls=1,
        )


def test_family_mismatch_rejects_input_without_a_model_call() -> None:
    expert = deterministic()
    prediction = expert.predict(
        task_input(family=ResidualTaskFamily.TEST_SELECTION, allowed=("TEST_SELECTION", "ABSTAIN"))
    )
    assert prediction.route is BaselineRoute.REJECT_INPUT
    assert prediction.disposition is ExpertDisposition.REJECT_INPUT
    assert prediction.cost.provider_invocations == 0


def test_unfitted_linear_abstains_after_deterministic_miss() -> None:
    expert = LinearResidualExpert(
        deterministic=deterministic(),
        form=LinearForm.LOGISTIC,
        feature_names=("exit_code", "flag"),
        class_labels=("missing_dependency_edge",),
    )
    prediction = expert.predict(task_input(features=compact(failure_signature="unseen")))
    assert prediction.route is BaselineRoute.ABSTAIN
    assert "linear_coefficients_unavailable" in prediction.task_output.reason_codes
    assert prediction.cost.model_calls == 0


def test_round_trip_and_unknown_fields() -> None:
    expert = deterministic(
        lookup=(lookup_entry(),),
        rules=(missing_edge_rule(),),
        procedures=(procedure_binding(),),
    )
    assert DeterministicResidualExpert.from_dict(expert.to_dict()) == expert
    prediction = expert.predict(task_input())
    assert BaselinePrediction.from_dict(prediction.to_dict()) == prediction
    payload = prediction.to_dict()
    payload["global_threshold_ppm"] = 1
    with pytest.raises(UnknownFieldError):
        BaselinePrediction.from_dict(payload)
    linear = linear_expert(expert)
    assert LinearResidualExpert.from_dict(linear.to_dict()) == linear
    with pytest.raises(ResidualIntelligenceError, match="stably ordered"):
        BaselinePrediction(
            task_output=prediction.task_output,
            route=prediction.route,
            feature_identity=prediction.feature_identity,
            cost=prediction.cost,
            disposition=prediction.disposition,
            ranking=(
                RankingItem(reference_id="ev:b", score_ppm=1),
                RankingItem(reference_id="ev:a", score_ppm=2),
            ),
        )
