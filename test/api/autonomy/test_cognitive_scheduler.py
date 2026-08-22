from __future__ import annotations

from dataclasses import replace

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomy.cognitive_scheduler import (
    CognitiveScheduler,
    CognitiveSchedulingContext,
    CognitiveSchedulingError,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
    AuthorityClass,
    BudgetLedger,
    CancellationBehavior,
    CognitiveBudget,
    DecisionQuestion,
    DecisionQuestionType,
    MetaAction,
    MetaDecisionDisposition,
    PrivacyClass,
    QuestionDisposition,
    ResolutionAction,
    ResolutionCandidate,
    ResolutionEvidenceKind,
    RiskClass,
)


def _budget(**overrides: int) -> CognitiveBudget:
    values = {
        "max_total_model_calls": 5,
        "max_strong_model_calls": 2,
        "max_input_tokens": 10_000,
        "max_output_tokens": 2_000,
        "max_provider_spend_micros": 100_000,
        "max_proof_time_ms": 30_000,
        "max_validation_time_ms": 30_000,
        "max_human_questions": 1,
        "max_repair_rounds": 2,
        "max_plan_branches": 2,
        "max_context_expansions": 2,
        "max_wall_time_ms": 60_000,
        "validation_reserve_ms": 5_000,
    }
    values.update(overrides)
    return CognitiveBudget(**values)


def _ledger(**budget_overrides: int) -> BudgetLedger:
    return BudgetLedger(budget=_budget(**budget_overrides), epoch=1)


def _action(kind: MetaAction, **overrides: object) -> ResolutionAction:
    remote = kind in {MetaAction.CALL_REMOTE_STANDARD_MODEL, MetaAction.CALL_REMOTE_STRONG_MODEL}
    model = remote or kind is MetaAction.CALL_LOCAL_SMALL_MODEL
    values: dict[str, object] = {
        "action": kind,
        "precondition_ids": ("tree-current",),
        "expected_evidence_kind": ResolutionEvidenceKind.MODEL_ADVICE
        if model
        else ResolutionEvidenceKind.STATIC_ANALYSIS,
        "expected_uncertainty_reduction_bp": 8_000,
        "token_cost": 1_000 if model else 0,
        "latency_cost_ms": 2_000 if model else 100,
        "provider_cost_micros": 1_000 if remote else 0,
        "resource_cost_units": 1,
        "invalidation_cost_units": 1,
        "privacy_cost_units": 1 if remote else 0,
        "privacy_class": PrivacyClass.PUBLIC if remote else PrivacyClass.LOCAL_ONLY,
        "risk_class": RiskClass.R1_READ_ONLY,
        "cancellation_behavior": CancellationBehavior.COOPERATIVE,
        "cacheable": True,
        "authority_class": AuthorityClass.VERIFIED,
        "can_change_decision": True,
        "accepted_as_authority": True,
    }
    values.update(overrides)
    return ResolutionAction(**values)


def _question(actions: tuple[ResolutionAction, ...]) -> DecisionQuestion:
    return DecisionQuestion(
        objective_id="APMC-G000",
        acceptance_criterion_ids=("AC-1",),
        question_type=DecisionQuestionType.WHETHER_CONTEXT_IS_SUFFICIENT,
        current_alternatives=("sufficient", "expand"),
        required_evidence_ids=("context-witness",),
        known_evidence_ids=(),
        contradictory_evidence_ids=(),
        residual_uncertainty_bp=8_000,
        decision_deadline_ms=10_000,
        risk_if_incorrect=RiskClass.R2_REVERSIBLE_LOCAL,
        risk_if_left_unresolved=RiskClass.R1_READ_ONLY,
        possible_resolution_action_ids=tuple(item.action_id for item in actions),
        dependency_question_ids=(),
        terminal_decision_rule="current completeness witness resolves the question",
        disposition=QuestionDisposition.UNRESOLVED,
    )


def _candidate(
    question: DecisionQuestion, action: ResolutionAction, *, value: int = 100
) -> ResolutionCandidate:
    return ResolutionCandidate(
        question_id=question.question_id,
        resolution_action=action,
        expected_decision_value=value,
        admissible=True,
        policy_id="policy-v1",
    )


def _context(**overrides: object) -> CognitiveSchedulingContext:
    values: dict[str, object] = {
        "policy_id": "policy-v1",
        "satisfied_precondition_ids": frozenset({"tree-current"}),
        "local_small_model_available": True,
        "remote_standard_model_available": True,
        "remote_strong_model_available": True,
        "remote_disclosure_permitted": True,
        "protected_validation_input_tokens": 1_000,
        "required_authority_class": AuthorityClass.DERIVED,
    }
    values.update(overrides)
    return CognitiveSchedulingContext(**values)


def test_no_named_question_abstains_without_a_model() -> None:
    result = CognitiveScheduler().select(
        question=None, candidates=(), budget_ledger=_ledger(), context=_context()
    )
    assert result.selected_action is MetaAction.NO_OP
    assert result.disposition is MetaDecisionDisposition.NO_OP
    assert result.reason_codes == ("no_named_unresolved_question",)


def test_deterministic_authority_precedes_even_higher_value_strong_model() -> None:
    deterministic = _action(MetaAction.RUN_LOCAL_STATIC_ANALYSIS)
    strong = _action(MetaAction.CALL_REMOTE_STRONG_MODEL)
    question = _question((deterministic, strong))
    result = CognitiveScheduler().select(
        question=question,
        candidates=(
            _candidate(question, strong, value=100_000),
            _candidate(question, deterministic, value=1),
        ),
        budget_ledger=_ledger(),
        context=_context(),
    )
    assert result.selected_action is MetaAction.RUN_LOCAL_STATIC_ANALYSIS
    assert result.disposition is MetaDecisionDisposition.SELECTED


@pytest.mark.parametrize(
    "deterministic_candidate",
    ("stale_policy", "over_budget"),
)
def test_ineligible_deterministic_route_does_not_suppress_model(
    deterministic_candidate: str,
) -> None:
    deterministic = _action(
        MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
        token_cost=20_000 if deterministic_candidate == "over_budget" else 0,
    )
    local = _action(MetaAction.CALL_LOCAL_SMALL_MODEL)
    question = _question((deterministic, local))
    deterministic_route = _candidate(question, deterministic)
    if deterministic_candidate == "stale_policy":
        deterministic_route = replace(deterministic_route, policy_id="policy-stale")

    result = CognitiveScheduler().select(
        question=question,
        candidates=(deterministic_route, _candidate(question, local)),
        budget_ledger=_ledger(),
        context=_context(),
    )
    assert result.selected_action is MetaAction.CALL_LOCAL_SMALL_MODEL


def test_no_op_and_quarantine_do_not_preempt_resolution_work() -> None:
    no_op = _action(MetaAction.NO_OP)
    quarantine = _action(MetaAction.QUARANTINE_TASK)
    deterministic = _action(MetaAction.RUN_LOCAL_STATIC_ANALYSIS)
    question = _question((no_op, quarantine, deterministic))
    result = CognitiveScheduler().select(
        question=question,
        candidates=tuple(
            _candidate(question, action, value=100_000) for action in (no_op, quarantine)
        )
        + (_candidate(question, deterministic, value=1),),
        budget_ledger=_ledger(),
        context=_context(),
    )
    assert result.selected_action is MetaAction.RUN_LOCAL_STATIC_ANALYSIS


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("remote_disclosure_permitted", "false"),
        ("local_small_model_available", 1),
        ("protected_validation_input_tokens", True),
    ),
)
def test_scheduling_context_rejects_truthy_non_boolean_policy_facts(
    field: str,
    value: object,
) -> None:
    with pytest.raises(CognitiveSchedulingError):
        _context(**{field: value})


def test_current_authoritative_cached_receipt_precedes_deterministic_analysis() -> None:
    cached = _action(
        MetaAction.READ_CACHED_RECEIPT,
        expected_evidence_kind=ResolutionEvidenceKind.CACHED_RECEIPT,
        authority_class=AuthorityClass.AUTHORITATIVE,
    )
    deterministic = _action(MetaAction.RUN_LOCAL_STATIC_ANALYSIS)
    question = _question((cached, deterministic))
    result = CognitiveScheduler().select(
        question=question,
        candidates=(
            _candidate(question, deterministic, value=100_000),
            _candidate(question, cached, value=1),
        ),
        budget_ledger=_ledger(),
        context=_context(),
    )
    assert result.selected_action is MetaAction.READ_CACHED_RECEIPT


def test_derived_cached_analysis_follows_deterministic_software() -> None:
    cached = _action(
        MetaAction.READ_CACHED_RECEIPT,
        expected_evidence_kind=ResolutionEvidenceKind.CACHED_RECEIPT,
        authority_class=AuthorityClass.DERIVED,
    )
    deterministic = _action(MetaAction.RUN_LOCAL_STATIC_ANALYSIS)
    question = _question((cached, deterministic))
    result = CognitiveScheduler().select(
        question=question,
        candidates=(
            _candidate(question, cached, value=100_000),
            _candidate(question, deterministic, value=1),
        ),
        budget_ledger=_ledger(),
        context=_context(required_authority_class=AuthorityClass.DERIVED),
    )
    assert result.selected_action is MetaAction.RUN_LOCAL_STATIC_ANALYSIS


def test_integer_value_per_cost_orders_within_route_deterministically() -> None:
    first = _action(MetaAction.RUN_GRAPH_RETRIEVAL, latency_cost_ms=200)
    second = _action(MetaAction.RUN_LOCAL_STATIC_ANALYSIS, latency_cost_ms=100)
    question = _question((first, second))
    candidates = (_candidate(question, first, value=100), _candidate(question, second, value=75))
    scheduler = CognitiveScheduler()
    one = scheduler.select(
        question=question, candidates=candidates, budget_ledger=_ledger(), context=_context()
    )
    two = scheduler.select(
        question=question,
        candidates=tuple(reversed(candidates)),
        budget_ledger=_ledger(),
        context=_context(),
    )
    assert one.decision_id == two.decision_id
    assert one.selected_candidate_id == candidates[1].candidate_id


@pytest.mark.parametrize(
    ("context_update", "expected_reason"),
    [
        ({"remote_disclosure_permitted": False}, "privacy_policy_forbids_disclosure"),
        ({"remote_standard_model_available": False}, "remote_standard_model_unavailable"),
    ],
)
def test_remote_model_abstention(context_update: dict[str, object], expected_reason: str) -> None:
    remote = _action(MetaAction.CALL_REMOTE_STANDARD_MODEL)
    question = _question((remote,))
    result = CognitiveScheduler().select(
        question=question,
        candidates=(_candidate(question, remote),),
        budget_ledger=_ledger(),
        context=_context(**context_update),
    )
    assert result.selected_action is MetaAction.NO_OP
    assert expected_reason in result.reason_codes


def test_same_current_result_and_no_value_both_abstain() -> None:
    current = _action(MetaAction.CALL_LOCAL_SMALL_MODEL)
    no_value = _action(MetaAction.CALL_REMOTE_STANDARD_MODEL, can_change_decision=False)
    question = _question((current, no_value))
    result = CognitiveScheduler().select(
        question=question,
        candidates=(_candidate(question, current), _candidate(question, no_value)),
        budget_ledger=_ledger(),
        context=_context(current_result_action_ids=frozenset({current.action_id})),
    )
    assert result.disposition is MetaDecisionDisposition.BLOCKED
    assert set(result.reason_codes) == {
        "answer_cannot_change_decision",
        "current_identical_result_exists",
    }


def test_repeated_failure_without_new_evidence_quarantines() -> None:
    local = _action(MetaAction.CALL_LOCAL_SMALL_MODEL)
    question = _question((local,))
    result = CognitiveScheduler().select(
        question=question,
        candidates=(_candidate(question, local),),
        budget_ledger=_ledger(),
        context=_context(repeated_failure_action_ids=frozenset({local.action_id})),
    )
    assert result.selected_action is MetaAction.QUARANTINE_TASK
    assert result.disposition is MetaDecisionDisposition.QUARANTINE


def test_validation_token_reserve_forbids_model_call() -> None:
    local = _action(MetaAction.CALL_LOCAL_SMALL_MODEL, token_cost=9_500)
    question = _question((local,))
    result = CognitiveScheduler().select(
        question=question,
        candidates=(_candidate(question, local),),
        budget_ledger=_ledger(max_input_tokens=10_000),
        context=_context(protected_validation_input_tokens=1_000),
    )
    assert result.selected_action is MetaAction.NO_OP
    assert "protected_validation_token_reserve" in result.reason_codes


def test_model_call_abstains_when_conservative_output_bound_cannot_be_reserved() -> None:
    local = _action(MetaAction.CALL_LOCAL_SMALL_MODEL, token_cost=1_000)
    question = _question((local,))
    result = CognitiveScheduler().select(
        question=question,
        candidates=(_candidate(question, local),),
        budget_ledger=_ledger(max_output_tokens=999),
        context=_context(),
    )
    assert result.selected_action is MetaAction.QUARANTINE_TASK
    assert result.disposition is MetaDecisionDisposition.QUARANTINE
    assert "output_token_budget_exhausted" in result.reason_codes


def test_unaccepted_advisory_result_cannot_resolve_required_authority() -> None:
    advice = _action(
        MetaAction.CALL_LOCAL_SMALL_MODEL,
        authority_class=AuthorityClass.ADVISORY,
        accepted_as_authority=False,
    )
    question = _question((advice,))
    result = CognitiveScheduler().select(
        question=question,
        candidates=(_candidate(question, advice),),
        budget_ledger=_ledger(),
        context=_context(required_authority_class=AuthorityClass.VERIFIED),
    )
    assert result.selected_action is MetaAction.NO_OP
    assert "result_not_accepted_as_authority" in result.reason_codes


def test_unresolved_question_escalates_local_then_remote_by_closed_order() -> None:
    local = _action(MetaAction.CALL_LOCAL_SMALL_MODEL)
    standard = _action(MetaAction.CALL_REMOTE_STANDARD_MODEL)
    strong = _action(MetaAction.CALL_REMOTE_STRONG_MODEL)
    question = _question((local, standard, strong))
    candidates = tuple(_candidate(question, item) for item in (local, standard, strong))
    scheduler = CognitiveScheduler()
    selected_local = scheduler.select(
        question=question, candidates=candidates, budget_ledger=_ledger(), context=_context()
    )
    assert selected_local.selected_action is MetaAction.CALL_LOCAL_SMALL_MODEL

    selected_standard = scheduler.select(
        question=question,
        candidates=candidates,
        budget_ledger=_ledger(),
        context=_context(local_small_model_available=False),
    )
    assert selected_standard.selected_action is MetaAction.CALL_REMOTE_STANDARD_MODEL

    selected_strong = scheduler.select(
        question=question,
        candidates=candidates,
        budget_ledger=_ledger(),
        context=_context(local_small_model_available=False, remote_standard_model_available=False),
    )
    assert selected_strong.selected_action is MetaAction.CALL_REMOTE_STRONG_MODEL


def test_resolved_question_never_schedules_work() -> None:
    action = _action(MetaAction.CALL_REMOTE_STRONG_MODEL)
    question = replace(
        _question((action,)),
        required_evidence_ids=(),
        residual_uncertainty_bp=0,
        disposition=QuestionDisposition.RESOLVED,
        terminal_answer="sufficient",
    )
    result = CognitiveScheduler().select(
        question=question,
        candidates=(_candidate(question, action),),
        budget_ledger=_ledger(),
        context=_context(),
    )
    assert result.selected_action is MetaAction.NO_OP
    assert result.reason_codes == ("question_already_terminal",)


def test_inadmissible_terminal_claim_is_blocked_not_treated_as_complete() -> None:
    action = _action(MetaAction.CALL_REMOTE_STRONG_MODEL)
    question = replace(
        _question((action,)),
        disposition=QuestionDisposition.RESOLVED,
        terminal_answer="sufficient",
    )
    result = CognitiveScheduler().select(
        question=question,
        candidates=(_candidate(question, action),),
        budget_ledger=_ledger(),
        context=_context(),
    )
    assert result.disposition is MetaDecisionDisposition.BLOCKED
    assert result.reason_codes == ("inadmissible_terminal_claim",)
