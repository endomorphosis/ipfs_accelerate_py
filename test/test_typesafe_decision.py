from __future__ import annotations

from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.autonomy.cognitive_budget import (
    ObjectiveCognitiveBudgetLedger,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.cognitive_scheduler import (
    CognitiveSchedulingContext,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
    AuthorityClass,
    CancellationBehavior,
    CognitiveBudget,
    DecisionQuestion,
    DecisionQuestionType,
    MetaAction,
    PrivacyClass,
    QuestionDisposition,
    ResolutionAction,
    ResolutionCandidate,
    ResolutionEvidenceKind,
    RiskClass,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.decision_graph import (
    DecisionGraphController,
    question_is_admissibly_terminal,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.runtime import (
    AutonomousMetaController,
    MetaControllerStepStatus,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.typesafe_decision import (
    apply_typesafe_question_advice,
    prefer_escalation_candidates,
    prepare_step_candidates,
)


def _question() -> DecisionQuestion:
    return DecisionQuestion(
        objective_id="APMC-G000",
        acceptance_criterion_ids=("AC-1",),
        question_type=DecisionQuestionType.WHICH_PROOF_OBLIGATION_APPLIES,
        current_alternatives=("obl-1", "obl-2"),
        required_evidence_ids=(),
        known_evidence_ids=(),
        contradictory_evidence_ids=(),
        residual_uncertainty_bp=5_000,
        decision_deadline_ms=1_000,
        risk_if_incorrect=RiskClass.R1_READ_ONLY,
        risk_if_left_unresolved=RiskClass.R1_READ_ONLY,
        possible_resolution_action_ids=("action:smt",),
        dependency_question_ids=(),
        terminal_decision_rule="select only from current alternatives",
        mandatory=True,
        disposition=QuestionDisposition.UNRESOLVED,
        terminal_answer="",
    )


def test_apply_typesafe_advice_records_evidence_without_resolving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor.typesafe_permitted",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.autonomy.typesafe_decision.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {"answer": SimpleNamespace(choice="obl-1", confidence=0.33)}
        scores = {}
        nouls = {}

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_args, **_kwargs: _Result(),
    )
    question = _question()
    controller = DecisionGraphController.compile(
        repository_id="repo:ipfs-accelerate",
        tree_id="tree:one",
        objective_id="APMC-G000",
        objective_revision="revision:one",
        questions=(question,),
    )
    compiled = controller.graph.questions[0]
    advice = apply_typesafe_question_advice(
        controller,
        compiled,
        state={"board_item": "TASK-1"},
    )
    updated = next(
        item
        for item in controller.graph.questions
        if advice.evidence_id in item.known_evidence_ids
    )
    assert advice.nominated_answer == "obl-1"
    assert advice.next_action == "RUN_SMT_OR_PROVER"
    assert advice.can_resolve is False
    assert updated.terminal_answer == ""
    assert updated.disposition is QuestionDisposition.UNRESOLVED
    assert updated.residual_uncertainty_bp >= 1
    assert not question_is_admissibly_terminal(updated)
    assert not controller.complete


def _budget() -> CognitiveBudget:
    return CognitiveBudget(
        max_total_model_calls=4,
        max_strong_model_calls=2,
        max_input_tokens=8_000,
        max_output_tokens=2_000,
        max_provider_spend_micros=20_000,
        max_proof_time_ms=10_000,
        max_validation_time_ms=10_000,
        max_human_questions=1,
        max_repair_rounds=1,
        max_plan_branches=1,
        max_context_expansions=2,
        max_wall_time_ms=30_000,
        validation_reserve_ms=1_000,
    )


def _action(kind: MetaAction) -> ResolutionAction:
    is_model = kind in {
        MetaAction.CALL_REMOTE_STANDARD_MODEL,
        MetaAction.CALL_REMOTE_STRONG_MODEL,
    }
    evidence = ResolutionEvidenceKind.MODEL_ADVICE
    if kind is MetaAction.RUN_SMT_OR_PROVER:
        evidence = ResolutionEvidenceKind.PROOF_RESULT
    elif kind is MetaAction.RUN_LOCAL_STATIC_ANALYSIS:
        evidence = ResolutionEvidenceKind.STATIC_ANALYSIS
    return ResolutionAction(
        action=kind,
        precondition_ids=("tree-current",),
        expected_evidence_kind=evidence,
        expected_uncertainty_reduction_bp=8_000,
        token_cost=500 if is_model else 0,
        latency_cost_ms=100,
        provider_cost_micros=100 if is_model else 0,
        resource_cost_units=1,
        invalidation_cost_units=0,
        privacy_cost_units=0,
        privacy_class=PrivacyClass.PUBLIC if is_model else PrivacyClass.LOCAL_ONLY,
        risk_class=RiskClass.R1_READ_ONLY,
        cancellation_behavior=CancellationBehavior.COOPERATIVE,
        cacheable=True,
        authority_class=AuthorityClass.VERIFIED,
        accepted_as_authority=True,
    )


def _context() -> CognitiveSchedulingContext:
    return CognitiveSchedulingContext(
        policy_id="policy:one",
        satisfied_precondition_ids=frozenset({"tree-current"}),
        local_small_model_available=True,
        remote_standard_model_available=True,
        remote_strong_model_available=True,
        remote_disclosure_permitted=True,
        required_authority_class=AuthorityClass.DERIVED,
    )


def test_prepare_step_candidates_prefers_smt_over_cheaper_static_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor.typesafe_permitted",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.autonomy.typesafe_decision.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {"answer": SimpleNamespace(choice="obl-1", confidence=0.91)}
        scores = {}
        nouls = {}

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_args, **_kwargs: _Result(),
    )
    static = _action(MetaAction.RUN_LOCAL_STATIC_ANALYSIS)
    smt = _action(MetaAction.RUN_SMT_OR_PROVER)
    strong = _action(MetaAction.CALL_REMOTE_STRONG_MODEL)
    question = DecisionQuestion(
        objective_id="APMC-G000",
        acceptance_criterion_ids=("AC-1",),
        question_type=DecisionQuestionType.WHICH_PROOF_OBLIGATION_APPLIES,
        current_alternatives=("obl-1", "obl-2"),
        required_evidence_ids=(),
        known_evidence_ids=(),
        contradictory_evidence_ids=(),
        residual_uncertainty_bp=5_000,
        decision_deadline_ms=1_000,
        risk_if_incorrect=RiskClass.R1_READ_ONLY,
        risk_if_left_unresolved=RiskClass.R1_READ_ONLY,
        possible_resolution_action_ids=(static.action_id, smt.action_id, strong.action_id),
        dependency_question_ids=(),
        terminal_decision_rule="select only from current alternatives",
        mandatory=True,
        disposition=QuestionDisposition.UNRESOLVED,
        terminal_answer="",
    )
    controller = DecisionGraphController.compile(
        repository_id="repo:ipfs-accelerate",
        tree_id="tree:one",
        objective_id="APMC-G000",
        objective_revision="revision:one",
        questions=(question,),
    )
    compiled = controller.graph.questions[0]
    candidates = (
        ResolutionCandidate(
            question_id=compiled.question_id,
            resolution_action=static,
            expected_decision_value=100,
            admissible=True,
            policy_id="policy:one",
        ),
        ResolutionCandidate(
            question_id=compiled.question_id,
            resolution_action=smt,
            expected_decision_value=100,
            admissible=True,
            policy_id="policy:one",
        ),
        ResolutionCandidate(
            question_id=compiled.question_id,
            resolution_action=strong,
            expected_decision_value=100,
            admissible=True,
            policy_id="policy:one",
        ),
    )
    meta = AutonomousMetaController(
        decision_graph=controller,
        budget_controller=ObjectiveCognitiveBudgetLedger(_budget(), epoch=1),
    )
    unfiltered = meta.step(candidates=candidates, context=_context())
    assert unfiltered.candidate is not None
    assert unfiltered.candidate.resolution_action.action is MetaAction.RUN_LOCAL_STATIC_ANALYSIS

    prepared, updated, advice = prepare_step_candidates(
        controller,
        compiled,
        candidates,
        state={"board_item": "TASK-1"},
    )
    assert advice is not None
    assert advice.next_action == "RUN_SMT_OR_PROVER"
    assert {item.resolution_action.action for item in prepared} == {MetaAction.RUN_SMT_OR_PROVER}
    assert all(item.question_id == updated.question_id for item in prepared)

    step = meta.step(candidates=prepared, context=_context())
    assert step.status is MetaControllerStepStatus.ACTION_ADMITTED
    assert step.candidate is not None
    assert step.candidate.resolution_action.action is MetaAction.RUN_SMT_OR_PROVER
    assert not question_is_admissibly_terminal(step.question)
    assert not step.authorizes_effect


def test_privacy_skip_leaves_candidate_set_unchanged() -> None:
    static = _action(MetaAction.RUN_LOCAL_STATIC_ANALYSIS)
    question = DecisionQuestion(
        objective_id="APMC-G000",
        acceptance_criterion_ids=("AC-1",),
        question_type=DecisionQuestionType.WHICH_PROOF_OBLIGATION_APPLIES,
        current_alternatives=("obl-1", "obl-2"),
        required_evidence_ids=(),
        known_evidence_ids=(),
        contradictory_evidence_ids=(),
        residual_uncertainty_bp=5_000,
        decision_deadline_ms=1_000,
        risk_if_incorrect=RiskClass.R1_READ_ONLY,
        risk_if_left_unresolved=RiskClass.R1_READ_ONLY,
        possible_resolution_action_ids=(static.action_id,),
        dependency_question_ids=(),
        terminal_decision_rule="select only from current alternatives",
        mandatory=True,
        disposition=QuestionDisposition.UNRESOLVED,
        terminal_answer="",
    )
    controller = DecisionGraphController.compile(
        repository_id="repo:ipfs-accelerate",
        tree_id="tree:one",
        objective_id="APMC-G000",
        objective_revision="revision:one",
        questions=(question,),
    )
    compiled = controller.graph.questions[0]
    candidates = (
        ResolutionCandidate(
            question_id=compiled.question_id,
            resolution_action=static,
            expected_decision_value=100,
            admissible=True,
            policy_id="policy:one",
        ),
    )
    prepared, updated, advice = prepare_step_candidates(
        controller,
        compiled,
        candidates,
        state={},
        privacy_class="local_only",
        remote_disclosure_permitted=True,
    )
    assert advice is None
    assert prepared == candidates
    assert updated.question_id == compiled.question_id


def test_prefer_escalation_fails_open_when_no_match() -> None:
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
        AdvisoryReceipt,
        DecisionQuestionAdvice,
    )

    static = _action(MetaAction.RUN_LOCAL_STATIC_ANALYSIS)
    candidate = ResolutionCandidate(
        question_id="q1",
        resolution_action=static,
        expected_decision_value=100,
        admissible=True,
        policy_id="policy:one",
    )
    advice = DecisionQuestionAdvice(
        receipt=AdvisoryReceipt(action="answered", choice="obl-1", confidence=0.9),
        nominated_answer="obl-1",
        next_action="RUN_SMT_OR_PROVER",
        residual_uncertainty_bp=1_000,
        evidence_id="typesafe-advice-deadbeef",
    )
    assert prefer_escalation_candidates((candidate,), advice) == (candidate,)
