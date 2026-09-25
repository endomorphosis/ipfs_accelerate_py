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
    AutonomyRuntime,
    AutonomyRuntimeStatus,
    AutonomyWakeEvent,
    AutonomyWakeKind,
    AutonomousMetaController,
    MetaControllerStepStatus,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.typesafe_runtime_adapter import (
    TypesafeAuthorityError,
    TypesafeSupervisorAdapter,
    assert_typesafe_not_execution_authority,
    dispatch_autonomy_wake,
)


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


def test_assert_rejects_typesafe_as_sole_or_authoritative_evidence() -> None:
    with pytest.raises(TypesafeAuthorityError):
        assert_typesafe_not_execution_authority(
            ({"receipt_id": "typesafe-advice-abc", "authority": "advisory"},)
        )
    with pytest.raises(TypesafeAuthorityError):
        assert_typesafe_not_execution_authority(
            (
                {
                    "receipt_id": "typesafe-advice-abc",
                    "authority": "authoritative",
                },
                {"receipt_id": "kernel-1", "authority": "authoritative"},
            )
        )
    assert_typesafe_not_execution_authority(
        (
            {"receipt_id": "typesafe-advice-abc", "authority": "advisory"},
            {"receipt_id": "kernel-1", "authority": "authoritative"},
        )
    )


def test_adapter_prepare_and_step_prefers_smt_and_does_not_authorize_effect(
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
        possible_resolution_action_ids=(static.action_id, smt.action_id),
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
    )
    meta = AutonomousMetaController(
        decision_graph=controller,
        budget_controller=ObjectiveCognitiveBudgetLedger(_budget(), epoch=1),
    )
    adapter = TypesafeSupervisorAdapter(meta)
    handoff = adapter.prepare_and_step(
        candidates=candidates,
        context=_context(),
        state={"board_item": "TASK-1"},
    )
    assert handoff.admitted
    assert handoff.authorizes_effect is False
    step_dict = handoff.to_dict()
    assert step_dict["authorizes_effect"] is False
    assert step_dict["completion_authority"] is False
    assert step_dict["recovery_plan_delta_outstanding"] is False
    assert handoff.requires_decision_runtime
    assert handoff.step.status is MetaControllerStepStatus.ACTION_ADMITTED
    assert handoff.step.candidate is not None
    assert handoff.step.candidate.resolution_action.action is MetaAction.RUN_SMT_OR_PROVER
    assert handoff.advice is not None
    assert handoff.advice.can_resolve is False
    assert not question_is_admissibly_terminal(handoff.step.question)

    with pytest.raises(TypesafeAuthorityError, match="non-advisory"):
        adapter.admit_with_decision_runtime(handoff.step)

    class _Runtime:
        def decide(self, value: object) -> str:
            return "ok"

    with pytest.raises(TypesafeAuthorityError, match="sole execution evidence"):
        adapter.admit_with_decision_runtime(
            handoff.step,
            runtime=_Runtime(),
            runtime_input=SimpleNamespace(
                evidence_receipts=(
                    {"receipt_id": "typesafe-advice-abc", "authority": "advisory"},
                )
            ),
        )
    assert (
        adapter.admit_with_decision_runtime(
            handoff.step,
            runtime=_Runtime(),
            runtime_input=SimpleNamespace(
                evidence_receipts=(
                    {"receipt_id": "typesafe-advice-abc", "authority": "advisory"},
                    {"receipt_id": "kernel-1", "authority": "authoritative"},
                )
            ),
        )
        == "ok"
    )


def test_adapter_handle_wake_prefers_smt_and_runtime_stays_provider_free(
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
        possible_resolution_action_ids=(static.action_id, smt.action_id),
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
    )
    meta = AutonomousMetaController(
        decision_graph=controller,
        budget_controller=ObjectiveCognitiveBudgetLedger(_budget(), epoch=1),
    )
    runtime = AutonomyRuntime(controller=meta)
    adapter = TypesafeSupervisorAdapter(meta)
    event = AutonomyWakeEvent(kind=AutonomyWakeKind.PROOF, cursor_id="cursor:proof:1", sequence=1)
    handoff = adapter.handle_wake(
        runtime,
        event,
        candidates=candidates,
        context=_context(),
        state={"board_item": "TASK-1"},
    )
    assert handoff.model_called is False
    assert handoff.authorizes_effect is False
    assert handoff.result.model_called is False
    assert handoff.result.status is AutonomyRuntimeStatus.PROGRESSING
    assert handoff.result.step is not None
    assert handoff.result.step.candidate is not None
    assert (
        handoff.result.step.candidate.resolution_action.action
        is MetaAction.RUN_SMT_OR_PROVER
    )
    assert not handoff.result.authorizes_effect
    assert not handoff.result.authorizes_completion
    payload = handoff.to_dict()
    assert payload["authorizes_effect"] is False
    assert payload["completion_authority"] is False
    assert payload["recovery_delta_outstanding"] is False
    assert payload["similarity_not_resolution"] is False
    assert payload["cold_execution_required"] is False
    assert payload["qualification_incomplete"] is False
    assert payload["observations_not_preserved"] is False
    assert payload["negative_memory_blocks"] is False
    assert payload["world_root_cas_not_completion"] is False
    assert payload["boundary_contract_required"] is False
    assert payload["incompatible_identity"] is False


def _proof_wake_fixture():
    static = _action(MetaAction.RUN_LOCAL_STATIC_ANALYSIS)
    smt = _action(MetaAction.RUN_SMT_OR_PROVER)
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
        possible_resolution_action_ids=(static.action_id, smt.action_id),
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
    )
    meta = AutonomousMetaController(
        decision_graph=controller,
        budget_controller=ObjectiveCognitiveBudgetLedger(_budget(), epoch=1),
    )
    runtime = AutonomyRuntime(controller=meta)
    event = AutonomyWakeEvent(kind=AutonomyWakeKind.PROOF, cursor_id="cursor:proof:1", sequence=1)
    return runtime, event, candidates


def test_dispatch_autonomy_wake_without_key_keeps_original_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    runtime, event, candidates = _proof_wake_fixture()
    handoff = dispatch_autonomy_wake(
        runtime,
        event,
        candidates=candidates,
        context=_context(),
        state={"board_item": "TASK-1"},
    )
    assert handoff.model_called is False
    assert handoff.result.model_called is False
    assert handoff.advice is None
    assert handoff.result.step is not None
    assert handoff.result.step.candidate is not None
    assert (
        handoff.result.step.candidate.resolution_action.action
        is MetaAction.RUN_LOCAL_STATIC_ANALYSIS
    )
    assert not handoff.result.authorizes_effect


def test_dispatch_autonomy_wake_forwards_stall_reason_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    runtime, event, candidates = _proof_wake_fixture()
    handoff = dispatch_autonomy_wake(
        runtime,
        event,
        candidates=candidates,
        context=_context(),
        state={"reason": "provider_down"},
    )
    assert handoff.model_called is False
    assert handoff.result.status is AutonomyRuntimeStatus.IDLE
    assert handoff.result.reason_codes == ("retry_provider",)
    assert handoff.result.step is None


def test_dispatch_autonomy_wake_with_advice_prefers_smt(
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
    runtime, event, candidates = _proof_wake_fixture()
    handoff = dispatch_autonomy_wake(
        runtime,
        event,
        candidates=candidates,
        context=_context(),
        state={"board_item": "TASK-1"},
    )
    assert handoff.model_called is False
    assert handoff.result.step.candidate.resolution_action.action is MetaAction.RUN_SMT_OR_PROVER
    assert not handoff.result.authorizes_effect
