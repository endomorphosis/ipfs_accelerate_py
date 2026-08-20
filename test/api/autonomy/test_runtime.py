from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomy.cognitive_budget import (
    ObjectiveCognitiveBudgetLedger,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.cognitive_scheduler import (
    CognitiveSchedulingContext,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
    AuthorityClass,
    BudgetLedger,
    BudgetPurpose,
    BudgetReservation,
    BudgetReservationStatus,
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
)
from ipfs_accelerate_py.agent_supervisor.autonomy.runtime import (
    MAX_AUTONOMY_RUNTIME_SNAPSHOT_BYTES,
    AutonomousMetaController,
    AutonomousMetaControllerError,
    BudgetAdmission,
    BudgetAdmissionStatus,
    MetaControllerStepStatus,
    ObjectiveBudgetControllerAdapter,
)


def _budget() -> CognitiveBudget:
    return CognitiveBudget(
        max_total_model_calls=4,
        max_strong_model_calls=1,
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


class _FakeBudgetController:
    """Protocol fake: it models reservations but owns no persistence."""

    def __init__(
        self,
        ledger: BudgetLedger | None = None,
        *,
        outcome: BudgetAdmissionStatus = BudgetAdmissionStatus.RESERVED,
    ) -> None:
        self._ledger = ledger or BudgetLedger(budget=_budget(), epoch=1)
        self.outcome = outcome
        self.reserve_calls = 0
        self.question_ids: list[str] = []

    @property
    def ledger(self) -> BudgetLedger:
        return self._ledger

    def reserve_for_candidate(
        self,
        *,
        question: DecisionQuestion,
        candidate: ResolutionCandidate,
        idempotency_key: str,
    ) -> BudgetAdmission:
        self.reserve_calls += 1
        self.question_ids.append(question.question_id)
        if self.outcome is not BudgetAdmissionStatus.RESERVED:
            return BudgetAdmission(
                status=self.outcome,
                ledger=self._ledger,
                reason_codes=(
                    "budget_exhausted"
                    if self.outcome is BudgetAdmissionStatus.EXHAUSTED
                    else "budget_store_unavailable",
                ),
            )
        for item in self._ledger.reservations:
            if item.idempotency_key == idempotency_key:
                return BudgetAdmission(
                    status=BudgetAdmissionStatus.RESERVED,
                    ledger=self._ledger,
                    reservation=item,
                    reason_codes=("idempotent_replay",),
                )
        action = candidate.resolution_action
        is_model = action.action in {
            MetaAction.CALL_LOCAL_SMALL_MODEL,
            MetaAction.CALL_REMOTE_STANDARD_MODEL,
            MetaAction.CALL_REMOTE_STRONG_MODEL,
        }
        reservation = BudgetReservation(
            budget_id=self._ledger.budget.budget_id,
            idempotency_key=idempotency_key,
            question_id=question.question_id,
            action_id=action.action_id,
            purpose=BudgetPurpose.MODEL if is_model else BudgetPurpose.ANALYSIS,
            status=BudgetReservationStatus.RESERVED,
            max_total_model_calls=1 if is_model else 0,
            max_strong_model_calls=(
                1 if action.action is MetaAction.CALL_REMOTE_STRONG_MODEL else 0
            ),
            max_input_tokens=action.token_cost,
            max_output_tokens=action.token_cost if is_model else 0,
            max_provider_spend_micros=action.provider_cost_micros,
            max_wall_time_ms=action.latency_cost_ms,
        )
        self._ledger = replace(
            self._ledger,
            reservations=self._ledger.reservations + (reservation,),
        )
        return BudgetAdmission(
            status=BudgetAdmissionStatus.RESERVED,
            ledger=self._ledger,
            reservation=reservation,
        )

    def snapshot(self) -> dict[str, Any]:
        return {"ledger": self._ledger.to_record()}

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> _FakeBudgetController:
        assert set(snapshot) == {"ledger"}
        return cls(BudgetLedger.from_dict(snapshot["ledger"]))


def _action(kind: MetaAction = MetaAction.RUN_LOCAL_STATIC_ANALYSIS) -> ResolutionAction:
    is_model = kind in {
        MetaAction.CALL_LOCAL_SMALL_MODEL,
        MetaAction.CALL_REMOTE_STANDARD_MODEL,
        MetaAction.CALL_REMOTE_STRONG_MODEL,
    }
    return ResolutionAction(
        action=kind,
        precondition_ids=("tree-current",),
        expected_evidence_kind=(
            ResolutionEvidenceKind.MODEL_ADVICE
            if is_model
            else ResolutionEvidenceKind.STATIC_ANALYSIS
        ),
        expected_uncertainty_reduction_bp=8_000,
        token_cost=500 if is_model else 0,
        latency_cost_ms=100,
        provider_cost_micros=(
            100
            if kind
            in {
                MetaAction.CALL_REMOTE_STANDARD_MODEL,
                MetaAction.CALL_REMOTE_STRONG_MODEL,
            }
            else 0
        ),
        resource_cost_units=1,
        invalidation_cost_units=0,
        privacy_cost_units=0,
        privacy_class=(PrivacyClass.PUBLIC if is_model else PrivacyClass.LOCAL_ONLY),
        risk_class=RiskClass.R1_READ_ONLY,
        cancellation_behavior=CancellationBehavior.COOPERATIVE,
        cacheable=True,
        authority_class=AuthorityClass.VERIFIED,
        accepted_as_authority=True,
    )


def _question(
    *,
    criterion: str,
    action: ResolutionAction,
    deadline: int = 1_000,
    mandatory: bool = True,
    dependencies: tuple[str, ...] = (),
    question_type: DecisionQuestionType = DecisionQuestionType.WHICH_TEST_IS_REQUIRED,
) -> DecisionQuestion:
    return DecisionQuestion(
        objective_id="APMC-G000",
        acceptance_criterion_ids=(criterion,),
        question_type=question_type,
        current_alternatives=("not_required", "selected"),
        required_evidence_ids=(),
        known_evidence_ids=(),
        contradictory_evidence_ids=(),
        residual_uncertainty_bp=8_000,
        decision_deadline_ms=deadline,
        risk_if_incorrect=RiskClass.R1_READ_ONLY,
        risk_if_left_unresolved=RiskClass.R1_READ_ONLY,
        possible_resolution_action_ids=(action.action_id,),
        dependency_question_ids=dependencies,
        terminal_decision_rule="select a declared alternative from current evidence",
        mandatory=mandatory,
        disposition=QuestionDisposition.UNRESOLVED,
    )


def _graph(*questions: DecisionQuestion) -> DecisionGraphController:
    return DecisionGraphController.compile(
        repository_id="repo:ipfs-accelerate",
        tree_id="tree:current",
        objective_id="APMC-G000",
        objective_revision="revision:one",
        questions=questions,
    )


def _candidate(
    question: DecisionQuestion, action: ResolutionAction, *, value: int = 100
) -> ResolutionCandidate:
    return ResolutionCandidate(
        question_id=question.question_id,
        resolution_action=action,
        expected_decision_value=value,
        admissible=True,
        policy_id="policy:one",
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


def test_selects_exactly_one_named_question_and_reserves_before_admission() -> None:
    action = _action()
    later = _question(criterion="AC-later", action=action, deadline=5_000)
    first = _question(
        criterion="AC-first",
        action=action,
        deadline=100,
        question_type=DecisionQuestionType.WHETHER_CACHE_IS_REUSABLE,
    )
    graph = _graph(later, first)
    current = {item.acceptance_criterion_ids[0]: item for item in graph.graph.questions}
    candidates = tuple(_candidate(question, action) for question in current.values())
    budget = _FakeBudgetController()
    runtime = AutonomousMetaController(decision_graph=graph, budget_controller=budget)

    result = runtime.step(candidates=tuple(reversed(candidates)), context=_context())

    assert result.status is MetaControllerStepStatus.ACTION_ADMITTED
    assert result.question is not None
    assert result.question.acceptance_criterion_ids == ("AC-first",)
    assert budget.question_ids == [result.question.question_id]
    assert result.decision.reservation_id == result.reservation.reservation_id
    assert result.admitted
    assert result.requires_decision_runtime
    assert not result.authorizes_effect
    # Scheduling and budget reservation cannot resolve the graph or dispatch an
    # action; effect and completion authority remain untouched.
    assert runtime.decision_graph.graph.graph_id == graph.graph.graph_id
    assert result.question.disposition is QuestionDisposition.UNRESOLVED


def test_unchanged_and_no_mandatory_question_idle_without_budget_writes() -> None:
    action = _action()
    unresolved = _question(criterion="AC-1", action=action)
    budget = _FakeBudgetController()
    runtime = AutonomousMetaController(decision_graph=_graph(unresolved), budget_controller=budget)
    before = runtime.snapshot_json()

    unchanged = runtime.step(
        candidates=(_candidate(unresolved, action),),
        context=_context(),
        meaningful_change=False,
    )

    assert unchanged.status is MetaControllerStepStatus.IDLE
    assert unchanged.decision.selected_action is MetaAction.NO_OP
    assert unchanged.reason_codes == ("unchanged_state",)
    assert budget.reserve_calls == 0
    assert runtime.snapshot_json() == before

    optional = _question(criterion="AC-optional", action=action, mandatory=False)
    optional_budget = _FakeBudgetController()
    optional_runtime = AutonomousMetaController(
        decision_graph=_graph(optional), budget_controller=optional_budget
    )
    healthy_idle = optional_runtime.step(candidates=(), context=_context())
    assert healthy_idle.status is MetaControllerStepStatus.IDLE
    assert healthy_idle.reason_codes == ("no_unresolved_mandatory_question",)
    assert optional_budget.reserve_calls == 0


@pytest.mark.parametrize(
    ("budget_status", "step_status"),
    [
        (BudgetAdmissionStatus.EXHAUSTED, MetaControllerStepStatus.BUDGET_EXHAUSTED),
        (BudgetAdmissionStatus.UNAVAILABLE, MetaControllerStepStatus.UNAVAILABLE),
    ],
)
def test_budget_failure_is_typed_and_never_admits_action(
    budget_status: BudgetAdmissionStatus,
    step_status: MetaControllerStepStatus,
) -> None:
    action = _action()
    question = _question(criterion="AC-1", action=action)
    graph = _graph(question)
    current = graph.graph.questions[0]
    budget = _FakeBudgetController(outcome=budget_status)
    runtime = AutonomousMetaController(decision_graph=graph, budget_controller=budget)

    result = runtime.step(candidates=(_candidate(current, action),), context=_context())

    assert result.status is step_status
    assert not result.admitted
    assert not result.authorizes_effect
    assert result.reservation is None
    assert result.decision.reservation_id == ""


def test_missing_route_is_typed_unavailable_without_reservation() -> None:
    action = _action()
    question = _question(criterion="AC-1", action=action)
    budget = _FakeBudgetController()
    runtime = AutonomousMetaController(decision_graph=_graph(question), budget_controller=budget)

    result = runtime.step(candidates=(), context=_context())

    assert result.status is MetaControllerStepStatus.UNAVAILABLE
    assert result.reason_codes == ("no_resolution_candidate",)
    assert budget.reserve_calls == 0


def test_inadmissible_resolved_claim_is_blocked_instead_of_reported_idle() -> None:
    action = _action()
    question = replace(
        _question(criterion="AC-1", action=action),
        required_evidence_ids=("required-evidence",),
        residual_uncertainty_bp=0,
        disposition=QuestionDisposition.RESOLVED,
        terminal_answer="selected",
    )
    budget = _FakeBudgetController()
    runtime = AutonomousMetaController(
        decision_graph=_graph(question),
        budget_controller=budget,
    )

    result = runtime.step(candidates=(), context=_context())

    assert result.status is MetaControllerStepStatus.BLOCKED
    assert result.reason_codes == ("inadmissible_terminal_claim",)
    assert budget.reserve_calls == 0


def test_canonical_objective_budget_ledger_is_composed_without_a_second_authority() -> None:
    action = _action(MetaAction.CALL_LOCAL_SMALL_MODEL)
    question = _question(criterion="AC-1", action=action)
    graph = _graph(question)
    current = graph.graph.questions[0]
    objective_ledger = ObjectiveCognitiveBudgetLedger(_budget(), epoch=7)
    runtime = AutonomousMetaController(
        decision_graph=graph,
        budget_controller=objective_ledger,
    )

    result = runtime.step(candidates=(_candidate(current, action),), context=_context())

    assert result.status is MetaControllerStepStatus.ACTION_ADMITTED
    assert result.reservation is not None
    assert result.reservation.max_total_model_calls == 1
    assert result.reservation.max_input_tokens == action.token_cost
    assert result.reservation.max_output_tokens == action.token_cost
    assert result.reservation.max_wall_time_ms == action.latency_cost_ms
    assert objective_ledger.snapshot().ledger_id == result.ledger_id
    assert isinstance(runtime.budget_controller, ObjectiveBudgetControllerAdapter)

    recovered = AutonomousMetaController.from_snapshot(
        runtime.snapshot_json(),
        budget_loader=ObjectiveBudgetControllerAdapter.from_snapshot,
    )
    assert recovered.snapshot_json() == runtime.snapshot_json()


def test_released_idempotent_reservation_cannot_be_re_admitted() -> None:
    action = _action()
    question = _question(criterion="AC-1", action=action)
    graph = _graph(question)
    current = graph.graph.questions[0]
    objective_ledger = ObjectiveCognitiveBudgetLedger(_budget(), epoch=7)
    runtime = AutonomousMetaController(
        decision_graph=graph,
        budget_controller=objective_ledger,
    )
    candidate = _candidate(current, action)

    first = runtime.step(candidates=(candidate,), context=_context())
    assert first.status is MetaControllerStepStatus.ACTION_ADMITTED
    assert first.reservation is not None
    objective_ledger.release(first.reservation.reservation_id)

    replay = runtime.step(candidates=(candidate,), context=_context())
    assert replay.status is MetaControllerStepStatus.UNAVAILABLE
    assert replay.reservation is None
    assert replay.reason_codes == ("reservation_not_active", "released")


def test_restart_snapshot_is_deterministic_content_bound_and_provider_free() -> None:
    action = _action()
    question = _question(criterion="AC-1", action=action)
    runtime = AutonomousMetaController(
        decision_graph=_graph(question),
        budget_controller=_FakeBudgetController(),
    )
    snapshot = runtime.snapshot_json()

    recovered = AutonomousMetaController.from_snapshot(
        snapshot,
        budget_loader=lambda value: _FakeBudgetController.from_snapshot(dict(value)),
    )

    assert recovered.snapshot_json() == snapshot
    assert recovered.decision_graph.graph.graph_id == runtime.decision_graph.graph.graph_id
    assert (
        recovered.budget_controller.ledger.ledger_id == runtime.budget_controller.ledger.ledger_id
    )

    forged = json.loads(snapshot)
    forged["graph_id"] = "tree:forged"
    with pytest.raises(AutonomousMetaControllerError, match="identity mismatch"):
        AutonomousMetaController.from_snapshot(
            forged,
            budget_loader=lambda value: _FakeBudgetController.from_snapshot(dict(value)),
        )

    unknown = json.loads(snapshot)
    unknown["provider_client"] = "forbidden"
    with pytest.raises(AutonomousMetaControllerError, match="unknown fields"):
        AutonomousMetaController.from_snapshot(
            unknown,
            budget_loader=lambda value: _FakeBudgetController.from_snapshot(dict(value)),
        )


def test_restart_snapshot_rejects_duplicate_malformed_and_unbounded_json() -> None:
    action = _action()
    question = _question(criterion="AC-1", action=action)
    runtime = AutonomousMetaController(
        decision_graph=_graph(question),
        budget_controller=_FakeBudgetController(),
    )
    snapshot = runtime.snapshot_json()
    duplicate = snapshot[:-1] + ',"schema":"duplicate"}'

    for payload, reason in (
        (duplicate, "duplicate"),
        ("{", "malformed"),
        (snapshot + " " * MAX_AUTONOMY_RUNTIME_SNAPSHOT_BYTES, "bounded size"),
    ):
        with pytest.raises(AutonomousMetaControllerError, match=reason):
            AutonomousMetaController.from_snapshot(
                payload,
                budget_loader=lambda value: _FakeBudgetController.from_snapshot(dict(value)),
            )
