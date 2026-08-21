from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
    AuthorityClass,
    AutonomyEnvelope,
    AutonomyLevel,
    AutonomyPolicy,
    CognitiveBudget,
    DecisionQuestion,
    DecisionQuestionType,
    MetaAction,
    MetaDecision,
    MetaDecisionDisposition,
    PrivacyClass,
    QuestionDisposition,
    RiskAssessment,
    RiskClass,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.human_escalation import (
    HUMAN_ESCALATION_COMPILER_INTERFACE,
    HumanEscalationCompiler,
    HumanEscalationContext,
    HumanEscalationDisposition,
    HumanEscalationError,
    classify_irreducibility,
    requests_full_history_review,
)


def _budget(**overrides: int) -> CognitiveBudget:
    values = {
        "max_total_model_calls": 4,
        "max_strong_model_calls": 1,
        "max_input_tokens": 8_000,
        "max_output_tokens": 2_000,
        "max_provider_spend_micros": 20_000,
        "max_proof_time_ms": 10_000,
        "max_validation_time_ms": 10_000,
        "max_human_questions": 1,
        "max_repair_rounds": 1,
        "max_plan_branches": 1,
        "max_context_expansions": 2,
        "max_wall_time_ms": 30_000,
        "validation_reserve_ms": 1_000,
    }
    values.update(overrides)
    return CognitiveBudget(**values)


def _policy(**overrides: object) -> AutonomyPolicy:
    values: dict[str, object] = {
        "policy_revision": "policy-rev-1",
        "authority_id": "operator-policy-authority",
        "human_escalation_policy_id": "human-policy-1",
        "default_level": AutonomyLevel.RECOMMEND,
    }
    values.update(overrides)
    return AutonomyPolicy(**values)


def _envelope(*, policy: AutonomyPolicy, **overrides: object) -> AutonomyEnvelope:
    values: dict[str, object] = {
        "repository_id": "repo-1",
        "tree_id": "tree-1",
        "objective_id": "APMC-G000",
        "objective_revision": "objective-rev-1",
        "task_id": "APMC-014",
        "acceptance_criterion_ids": ("AC-operator-choice",),
        "risk_assessment": RiskAssessment(
            risk_class=RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL,
            reversible=False,
            irreversible_external_effect=True,
            legal_or_financial_effect=True,
            evidence_ids=("evidence-risk",),
            reason_codes=("external_release",),
        ),
        "autonomy_level": AutonomyLevel.RECOMMEND,
        "cognitive_budget": _budget(),
        "allowed_paths": ("ipfs_accelerate_py/agent_supervisor/autonomy",),
        "allowed_symbols": ("HumanEscalationCompiler",),
        "required_test_ids": ("test-human-escalation",),
        "required_proof_ids": (),
        "authority_id": "operator-policy-authority",
        "policy_id": policy.policy_id,
        "provider_usage_envelope_id": "provider-envelope-1",
        "resource_budget_id": "resource-budget-1",
        "human_escalation_policy_id": "human-policy-1",
        "expiry_ms": 12_000,
        "reversible": False,
    }
    values.update(overrides)
    return AutonomyEnvelope(**values)


def _question(**overrides: object) -> DecisionQuestion:
    values: dict[str, object] = {
        "objective_id": "APMC-G000",
        "acceptance_criterion_ids": ("AC-operator-choice",),
        "question_type": DecisionQuestionType.WHETHER_HUMAN_CHOICE_IS_IRREDUCIBLE,
        "current_alternatives": ("keep_shadow", "request_review"),
        "required_evidence_ids": ("held-out-route-1",),
        "known_evidence_ids": ("held-out-route-1",),
        "contradictory_evidence_ids": (),
        "residual_uncertainty_bp": 8_000,
        "decision_deadline_ms": 10_000,
        "risk_if_incorrect": RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL,
        "risk_if_left_unresolved": RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL,
        "possible_resolution_action_ids": ("action:human",),
        "dependency_question_ids": (),
        "terminal_decision_rule": "operator authority is required for the external effect",
        "mandatory": True,
        "disposition": QuestionDisposition.UNRESOLVED,
    }
    values.update(overrides)
    return DecisionQuestion(**values)


def _software_question(**overrides: object) -> DecisionQuestion:
    values: dict[str, object] = {
        "objective_id": "APMC-G000",
        "acceptance_criterion_ids": ("AC-type-check",),
        "question_type": DecisionQuestionType.WHICH_TEST_IS_REQUIRED,
        "current_alternatives": ("run_type_check", "skip_type_check"),
        "required_evidence_ids": ("typed-api-delta",),
        "known_evidence_ids": ("typed-api-delta",),
        "contradictory_evidence_ids": (),
        "residual_uncertainty_bp": 1_000,
        "decision_deadline_ms": 10_000,
        "risk_if_incorrect": RiskClass.R1_READ_ONLY,
        "risk_if_left_unresolved": RiskClass.R1_READ_ONLY,
        "possible_resolution_action_ids": ("action:type-check",),
        "dependency_question_ids": (),
        "terminal_decision_rule": "current typed selector resolves the question",
        "mandatory": False,
        "disposition": QuestionDisposition.UNRESOLVED,
    }
    values.update(overrides)
    return DecisionQuestion(**values)


def _human_decision(question: DecisionQuestion) -> MetaDecision:
    return MetaDecision(
        question_id=question.question_id,
        selected_candidate_id="candidate:human",
        selected_action=MetaAction.REQUEST_HUMAN_DECISION,
        considered_candidate_ids=("candidate:human",),
        rejected_candidate_ids=(),
        evidence_ids=question.known_evidence_ids,
        reservation_id="",
        policy_id="policy-v1",
        disposition=MetaDecisionDisposition.SELECTED,
        reason_codes=("closed_route_precedence",),
    )


def _static_decision(question: DecisionQuestion) -> MetaDecision:
    return MetaDecision(
        question_id=question.question_id,
        selected_candidate_id="candidate:static",
        selected_action=MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
        considered_candidate_ids=("candidate:static",),
        rejected_candidate_ids=(),
        evidence_ids=question.known_evidence_ids,
        reservation_id="",
        policy_id="policy-v1",
        disposition=MetaDecisionDisposition.SELECTED,
        reason_codes=("hard_constraints_passed",),
    )


def test_interface_identity_is_versioned() -> None:
    assert HUMAN_ESCALATION_COMPILER_INTERFACE == "HumanEscalationCompiler@1"
    assert HumanEscalationCompiler.interface == HUMAN_ESCALATION_COMPILER_INTERFACE


def test_r5_operator_choice_compiles_one_bounded_packet() -> None:
    question = _question()
    result = HumanEscalationCompiler().compile(
        question=question,
        context=HumanEscalationContext(
            no_admitted_non_human_route=True,
            required_authority_class=AuthorityClass.OPERATOR_REQUIRED,
            meta_decision=_human_decision(question),
        ),
    )
    assert result.escalated is True
    assert result.disposition is HumanEscalationDisposition.PACKET_COMPILED
    packet = result.packet
    assert packet is not None
    assert packet.objective_id == "APMC-G000"
    assert packet.blocked_criterion_ids == ("AC-operator-choice",)
    assert 2 <= len(packet.options) <= 4
    assert packet.recommended_option == "keep_shadow"
    assert packet.recommended_option in packet.options
    assert set(packet.predicted_consequences) == set(packet.options)
    assert set(packet.cost_and_risk) == set(packet.options)
    assert set(packet.continuation_by_option) == set(packet.options)
    assert packet.continuation_by_option["keep_shadow"] == "record_non_promotion"
    assert packet.evidence_ids == ("held-out-route-1",)
    assert packet.expires_at_ms == 10_000
    assert "irreversible_or_legal_effect" in result.reason_codes
    assert "operator_only_authority" in result.reason_codes
    assert result.metrics.packets_emitted == 1
    assert result.metrics.options_emitted == 2
    assert result.metrics.mandatory_decisions_preserved == 1
    assert requests_full_history_review(packet.question) is False


def test_packets_never_ask_for_full_history_review() -> None:
    question = _question(
        current_alternatives=(
            "keep_shadow",
            "request_review",
            "review the full history",
            "dump logs",
        ),
        terminal_decision_rule="review full history before choosing an external release",
    )
    result = HumanEscalationCompiler().compile(
        question=question,
        context=HumanEscalationContext(no_admitted_non_human_route=True),
    )
    packet = result.packet
    assert packet is not None
    assert len(packet.options) == 2
    assert "review the full history" not in packet.options
    assert "dump logs" not in packet.options
    assert requests_full_history_review(packet.question) is False
    for option in packet.options:
        assert requests_full_history_review(option) is False
    for value in packet.predicted_consequences.values():
        assert requests_full_history_review(str(value)) is False
    assert result.metrics.full_history_requests_rejected == 1


def test_full_history_only_non_mandatory_question_is_not_escalated() -> None:
    question = _software_question(
        current_alternatives=("review full history", "dump the log"),
        terminal_decision_rule="operator should inspect the entire history",
        mandatory=False,
    )
    result = HumanEscalationCompiler().compile(
        question=question,
        context=HumanEscalationContext(suppress_unnecessary=False),
    )
    assert result.escalated is False
    assert result.packet is None
    assert result.reason_codes == ("full_history_review_forbidden",)
    assert result.metrics.non_escalations == 1
    assert result.metrics.packets_emitted == 0
    assert result.metrics.full_history_requests_rejected == 1


def test_required_human_decision_cannot_be_bypassed_by_software_route() -> None:
    question = _question()
    result = HumanEscalationCompiler().compile(
        question=question,
        context=HumanEscalationContext(
            admitted_non_human_actions=frozenset({MetaAction.RUN_LOCAL_STATIC_ANALYSIS}),
            meta_decision=_static_decision(question),
            suppress_unnecessary=True,
            required_authority_class=AuthorityClass.OPERATOR_REQUIRED,
        ),
    )
    assert result.escalated is True
    assert result.packet is not None
    assert "irreversible_or_legal_effect" in result.reason_codes
    assert result.metrics.mandatory_decisions_preserved == 1


def test_required_human_decision_cannot_be_bypassed_by_suppression_flag() -> None:
    question = _question(mandatory=True)
    irreducible, codes = classify_irreducibility(
        question,
        HumanEscalationContext(
            suppress_unnecessary=True,
            no_admitted_non_human_route=True,
            required_authority_class=AuthorityClass.OPERATOR_REQUIRED,
        ),
    )
    assert irreducible is True
    assert "mandatory_human_decision" in codes
    result = HumanEscalationCompiler().compile(
        question=question,
        context=HumanEscalationContext(
            suppress_unnecessary=True,
            no_admitted_non_human_route=True,
        ),
    )
    assert result.packet is not None


def test_non_irreducible_question_returns_deterministic_non_escalation_reason() -> None:
    question = _software_question()
    context = HumanEscalationContext(
        admitted_non_human_actions=frozenset({MetaAction.RUN_TYPE_CHECK}),
        meta_decision=_static_decision(question),
    )
    first = HumanEscalationCompiler().compile(question=question, context=context)
    second = HumanEscalationCompiler().compile(question=question, context=context)
    assert first.escalated is False
    assert first.packet is None
    assert first.reason_codes == ("admitted_non_human_route",)
    assert first.reason_codes == second.reason_codes
    assert first.metrics.to_dict() == second.metrics.to_dict()


def test_unnecessary_low_risk_question_is_suppressed() -> None:
    question = _software_question(mandatory=False)
    result = HumanEscalationCompiler().compile(
        question=question,
        context=HumanEscalationContext(suppress_unnecessary=True),
    )
    assert result.escalated is False
    assert result.reason_codes[0] == "unnecessary_question_suppressed"
    assert result.suppressed_question_ids == (question.question_id,)
    assert result.metrics.questions_suppressed == 1


def test_authority_privacy_budget_and_ambiguity_cases() -> None:
    compiler = HumanEscalationCompiler()

    authority = compiler.compile(
        question=_question(
            risk_if_incorrect=RiskClass.R4_SECURITY_OR_PROTOCOL_SENSITIVE,
            risk_if_left_unresolved=RiskClass.R4_SECURITY_OR_PROTOCOL_SENSITIVE,
            residual_uncertainty_bp=0,
        ),
        context=HumanEscalationContext(
            no_admitted_non_human_route=True,
            required_authority_class=AuthorityClass.OPERATOR_REQUIRED,
        ),
    )
    assert authority.packet is not None
    assert "operator_only_authority" in authority.reason_codes

    privacy = compiler.compile(
        question=_question(
            risk_if_incorrect=RiskClass.R2_REVERSIBLE_LOCAL,
            risk_if_left_unresolved=RiskClass.R2_REVERSIBLE_LOCAL,
            residual_uncertainty_bp=0,
            question_type=DecisionQuestionType.WHETHER_CACHE_IS_REUSABLE,
            current_alternatives=("keep_local", "disclose_remote"),
            terminal_decision_rule="privacy policy forbids unattended disclosure",
        ),
        context=HumanEscalationContext(
            no_admitted_non_human_route=True,
            privacy_class=PrivacyClass.FORBIDDEN_EXTERNAL,
            privacy_choice_required=True,
        ),
    )
    assert privacy.packet is not None
    assert "privacy_policy_choice" in privacy.reason_codes

    budget = compiler.compile(
        question=_question(
            risk_if_incorrect=RiskClass.R2_REVERSIBLE_LOCAL,
            risk_if_left_unresolved=RiskClass.R2_REVERSIBLE_LOCAL,
            residual_uncertainty_bp=0,
            current_alternatives=("wait_for_budget", "spend_last_human_question"),
            terminal_decision_rule="policy requires a human budget choice",
        ),
        context=HumanEscalationContext(
            no_admitted_non_human_route=True,
            budget_choice_required=True,
            model_budget_remaining=0,
            human_budget_remaining=1,
        ),
    )
    assert budget.packet is not None
    assert "policy_required_budget_choice" in budget.reason_codes

    ambiguity = compiler.compile(
        question=_question(
            risk_if_incorrect=RiskClass.R3_BOUNDED_REPOSITORY_MUTATION,
            risk_if_left_unresolved=RiskClass.R2_REVERSIBLE_LOCAL,
            contradictory_evidence_ids=("counter-1",),
            known_evidence_ids=("held-out-route-1",),
            required_evidence_ids=("held-out-route-1",),
            residual_uncertainty_bp=9_000,
        ),
        context=HumanEscalationContext(no_admitted_non_human_route=True),
    )
    assert ambiguity.packet is not None
    assert "irresolvable_contradiction" in ambiguity.reason_codes
    assert "irreducible_ambiguity" in ambiguity.reason_codes


def test_batch_rule_collapses_equivalent_questions_into_one_packet() -> None:
    first = _question(acceptance_criterion_ids=("AC-release",))
    second = _question(
        acceptance_criterion_ids=("AC-legal",),
        known_evidence_ids=("held-out-route-1", "counsel-memo"),
        required_evidence_ids=("held-out-route-1", "counsel-memo"),
    )
    unrelated = _question(
        acceptance_criterion_ids=("AC-other",),
        question_type=DecisionQuestionType.WHETHER_HUMAN_CHOICE_IS_IRREDUCIBLE,
        current_alternatives=("defer_merge", "authorize_merge"),
        terminal_decision_rule="choose the merge authorization",
        known_evidence_ids=("merge-witness",),
        required_evidence_ids=("merge-witness",),
    )
    result = HumanEscalationCompiler().compile(
        questions=(first, second, unrelated),
        context=HumanEscalationContext(no_admitted_non_human_route=True),
    )
    packet = result.packet
    assert packet is not None
    assert packet.blocked_criterion_ids == ("AC-legal", "AC-release")
    assert "counsel-memo" in packet.evidence_ids
    assert result.metrics.packets_emitted == 1
    assert result.metrics.questions_considered == 3
    assert result.metrics.questions_batched == 2
    assert set(result.batched_question_ids) == {first.question_id, second.question_id}
    assert "batched_equivalent_questions" in result.reason_codes
    assert requests_full_history_review(packet.question) is False
    assert "AC-other" not in packet.blocked_criterion_ids


def test_more_than_four_alternatives_are_compacted_to_the_safest_four() -> None:
    question = _question(
        current_alternatives=(
            "request_review",
            "keep_shadow",
            "authorize_release",
            "promote_now",
            "wait_for_counsel",
        )
    )
    result = HumanEscalationCompiler().compile(
        question=question,
        context=HumanEscalationContext(no_admitted_non_human_route=True),
    )
    packet = result.packet
    assert packet is not None
    assert len(packet.options) == 4
    assert packet.recommended_option == packet.options[0]
    assert packet.recommended_option in {"keep_shadow", "wait_for_counsel"}
    assert "promote_now" not in packet.options or packet.recommended_option != "promote_now"


def test_two_to_four_option_contract_is_preserved_for_sparse_mandatory_questions() -> None:
    question = _question(current_alternatives=("review full history",))
    result = HumanEscalationCompiler().compile(
        question=question,
        context=HumanEscalationContext(no_admitted_non_human_route=True),
    )
    packet = result.packet
    assert packet is not None
    assert packet.options == ("keep_current_safest", "request_authorized_review")
    assert packet.recommended_option == "keep_current_safest"
    assert result.metrics.options_emitted == 2


def test_terminal_and_empty_inputs_do_not_escalate() -> None:
    compiler = HumanEscalationCompiler()
    empty = compiler.compile()
    assert empty.reason_codes == ("no_named_unresolved_question",)
    assert empty.metrics.questions_considered == 0

    resolved = _question(
        residual_uncertainty_bp=0,
        disposition=QuestionDisposition.RESOLVED,
        terminal_answer="keep_shadow",
        contradictory_evidence_ids=(),
    )
    terminal = compiler.compile(
        question=resolved,
        context=HumanEscalationContext(no_admitted_non_human_route=True),
    )
    assert terminal.reason_codes == ("question_already_terminal",)

    invalidated = compiler.compile(
        question=_software_question(disposition=QuestionDisposition.INVALIDATED),
        context=HumanEscalationContext(no_admitted_non_human_route=True),
    )
    assert invalidated.reason_codes == ("question_invalidated",)


def test_compile_is_order_stable_and_content_addressed() -> None:
    first = _question(acceptance_criterion_ids=("AC-release",))
    second = _question(
        acceptance_criterion_ids=("AC-legal",),
        known_evidence_ids=("held-out-route-1", "counsel-memo"),
        required_evidence_ids=("held-out-route-1", "counsel-memo"),
    )
    context = HumanEscalationContext(no_admitted_non_human_route=True)
    compiler = HumanEscalationCompiler()
    one = compiler.compile(questions=(first, second), context=context)
    two = compiler.compile(questions=(second, first), context=context)
    assert one.packet is not None and two.packet is not None
    assert one.packet.packet_id == two.packet.packet_id
    assert one.reason_codes == two.reason_codes
    assert one.metrics.to_dict() == two.metrics.to_dict()


def test_expiry_uses_the_tightest_positive_bound() -> None:
    policy = _policy()
    question = _question(decision_deadline_ms=20_000)
    result = HumanEscalationCompiler().compile(
        question=question,
        context=HumanEscalationContext(
            no_admitted_non_human_route=True,
            policy=policy,
            envelope=_envelope(policy=policy, expiry_ms=7_500),
            now_ms=1_000,
            default_ttl_ms=50_000,
        ),
    )
    assert result.packet is not None
    assert result.packet.expires_at_ms == 7_500


def test_malformed_context_and_mixed_objectives_fail_closed() -> None:
    with pytest.raises(HumanEscalationError):
        HumanEscalationContext(no_admitted_non_human_route="yes")  # type: ignore[arg-type]
    with pytest.raises(HumanEscalationError):
        HumanEscalationContext(
            admitted_non_human_actions=frozenset({MetaAction.REQUEST_HUMAN_DECISION})
        )
    with pytest.raises(HumanEscalationError):
        HumanEscalationCompiler().compile(
            questions=(
                _question(),
                _software_question(objective_id="APMC-G070"),
            )
        )


def test_requests_full_history_review_detects_closed_markers() -> None:
    assert requests_full_history_review("Please review the full-history dump") is True
    assert requests_full_history_review("keep_shadow") is False
    assert requests_full_history_review("") is False


def test_outcome_metrics_are_integer_and_closed() -> None:
    question = _question()
    extra = _software_question(mandatory=False)
    result = HumanEscalationCompiler().compile(
        questions=(question, extra),
        context=HumanEscalationContext(no_admitted_non_human_route=True),
    )
    metrics = result.metrics.to_dict()
    assert set(metrics) == {
        "questions_considered",
        "packets_emitted",
        "questions_batched",
        "questions_suppressed",
        "mandatory_decisions_preserved",
        "options_emitted",
        "full_history_requests_rejected",
        "non_escalations",
    }
    assert all(isinstance(value, int) and not isinstance(value, bool) for value in metrics.values())
    assert metrics["questions_considered"] == 2
    assert metrics["packets_emitted"] == 1
    assert metrics["questions_suppressed"] == 1
    assert metrics["non_escalations"] == 0
