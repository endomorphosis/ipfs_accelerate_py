from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

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
from ipfs_accelerate_py.agent_supervisor.autonomy.metrics import AutonomyMetrics
from ipfs_accelerate_py.agent_supervisor.autonomy.runtime import (
    DEFAULT_SAFETY_INTERVAL_MS,
    AutonomyRuntime,
    AutonomyRuntimeStatus,
    AutonomyWakeEvent,
    AutonomyWakeKind,
    AutonomousMetaController,
    BudgetAdmission,
    BudgetAdmissionStatus,
    InMemoryAutonomyCheckpointSink,
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
    def __init__(
        self,
        ledger: BudgetLedger | None = None,
        *,
        outcome: BudgetAdmissionStatus = BudgetAdmissionStatus.RESERVED,
    ) -> None:
        self._ledger = ledger or BudgetLedger(budget=_budget(), epoch=1)
        self.outcome = outcome
        self.reserve_calls = 0

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
        reservation = BudgetReservation(
            budget_id=self._ledger.budget.budget_id,
            idempotency_key=idempotency_key,
            question_id=question.question_id,
            action_id=candidate.resolution_action.action_id,
            purpose=BudgetPurpose.ANALYSIS,
            status=BudgetReservationStatus.RESERVED,
            max_wall_time_ms=candidate.resolution_action.latency_cost_ms,
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
        provider_cost_micros=0,
        resource_cost_units=1,
        invalidation_cost_units=0,
        privacy_cost_units=0,
        privacy_class=PrivacyClass.LOCAL_ONLY,
        risk_class=RiskClass.R1_READ_ONLY,
        cancellation_behavior=CancellationBehavior.COOPERATIVE,
        cacheable=True,
        authority_class=AuthorityClass.VERIFIED,
        accepted_as_authority=True,
    )


def _question(
    *,
    action: ResolutionAction,
    resolved: bool = False,
) -> DecisionQuestion:
    return DecisionQuestion(
        objective_id="APMC-G000",
        acceptance_criterion_ids=("AC-idle",),
        question_type=DecisionQuestionType.WHICH_TEST_IS_REQUIRED,
        current_alternatives=("not_required", "selected"),
        required_evidence_ids=("evidence-done",) if resolved else (),
        known_evidence_ids=("evidence-done",) if resolved else (),
        contradictory_evidence_ids=(),
        residual_uncertainty_bp=0 if resolved else 8_000,
        decision_deadline_ms=1_000,
        risk_if_incorrect=RiskClass.R1_READ_ONLY,
        risk_if_left_unresolved=RiskClass.R1_READ_ONLY,
        possible_resolution_action_ids=(action.action_id,),
        dependency_question_ids=(),
        terminal_decision_rule="select a declared alternative from current evidence",
        mandatory=True,
        disposition=(
            QuestionDisposition.RESOLVED if resolved else QuestionDisposition.UNRESOLVED
        ),
        terminal_answer="selected" if resolved else "",
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


def _complete_runtime(
    *,
    sink: InMemoryAutonomyCheckpointSink | None = None,
    interval_ms: int = DEFAULT_SAFETY_INTERVAL_MS,
) -> AutonomyRuntime:
    action = _action()
    controller = AutonomousMetaController(
        decision_graph=DecisionGraphController.compile(
            repository_id="repo:ipfs-accelerate",
            tree_id="tree:current",
            objective_id="APMC-G000",
            objective_revision="revision:one",
            questions=(_question(action=action, resolved=True),),
        ),
        budget_controller=_FakeBudgetController(),
    )
    assert not controller.has_work()
    return AutonomyRuntime(
        controller=controller,
        checkpoint_sink=sink,
        safety_interval_ms=interval_ms,
        now_ms=0,
    )


def test_unchanged_complete_board_window_ticks_do_not_call_write_scan_or_refill() -> None:
    sink = InMemoryAutonomyCheckpointSink()
    runtime = _complete_runtime(sink=sink, interval_ms=1_000)
    before = runtime.snapshot_json()
    assert runtime.healthy_idle

    for ordinal in range(1, 6):
        event = runtime.safety_timer_event(now_ms=ordinal * 1_000)
        assert event is not None
        result = runtime.handle_wake(event, candidates=(), context=_context())
        assert result.status is AutonomyRuntimeStatus.IDLE
        assert result.reason_codes == ("unchanged_complete_board",)
        assert result.scanned is False
        assert result.wrote_state is False
        assert result.model_called is False
        assert result.refilled is False
        assert result.safety_timer is True

    assert sink.write_count == 0
    assert runtime.snapshot_json() == before
    assert runtime.metrics.model_calls == 0
    assert runtime.metrics.writes == 0
    assert runtime.metrics.scans == 0
    assert runtime.metrics.refills == 0
    assert runtime.metrics.admitted_actions == 0
    assert runtime.metrics.unchanged_complete_idle
    assert runtime.metrics.safety_timer_wakes == 5
    assert runtime.metrics.idle_cycles == 5


def test_meaningful_wakes_on_a_complete_board_confirm_idle_without_writes_or_models() -> None:
    sink = InMemoryAutonomyCheckpointSink()
    runtime = _complete_runtime(sink=sink)
    action = _action()
    compiled = runtime.controller.decision_graph.graph.questions[0]
    candidate = ResolutionCandidate(
        question_id=compiled.question_id,
        resolution_action=action,
        expected_decision_value=100,
        admissible=True,
        policy_id="policy:one",
    )
    kinds = (
        AutonomyWakeKind.REPOSITORY,
        AutonomyWakeKind.OBJECTIVE,
        AutonomyWakeKind.TASK,
        AutonomyWakeKind.VALIDATION,
        AutonomyWakeKind.PROOF,
        AutonomyWakeKind.PROVIDER,
        AutonomyWakeKind.LEASE,
        AutonomyWakeKind.HUMAN,
        AutonomyWakeKind.BUDGET,
        AutonomyWakeKind.COUNTEREXAMPLE,
        AutonomyWakeKind.FRESHNESS,
    )
    for kind in kinds:
        result = runtime.handle_wake(
            AutonomyWakeEvent(kind=kind, cursor_id=f"cursor:{kind.value}", sequence=1),
            candidates=(candidate,),
            context=_context(),
        )
        assert result.status is AutonomyRuntimeStatus.IDLE
        assert result.wrote_state is False
        assert result.model_called is False
        assert result.refilled is False
        assert result.scanned is True

    assert sink.write_count == 0
    assert runtime.metrics.model_calls == 0
    assert runtime.metrics.writes == 0
    assert runtime.metrics.refills == 0
    assert runtime.metrics.admitted_actions == 0


def test_healthy_exhaustion_does_not_refill_or_rescan_on_the_safety_timer() -> None:
    action = _action()
    question = _question(action=action, resolved=False)
    graph = DecisionGraphController.compile(
        repository_id="repo:ipfs-accelerate",
        tree_id="tree:current",
        objective_id="APMC-G000",
        objective_revision="revision:one",
        questions=(question,),
    )
    compiled = graph.graph.questions[0]
    sink = InMemoryAutonomyCheckpointSink()
    runtime = AutonomyRuntime(
        controller=AutonomousMetaController(
            decision_graph=graph,
            budget_controller=_FakeBudgetController(outcome=BudgetAdmissionStatus.EXHAUSTED),
        ),
        checkpoint_sink=sink,
        safety_interval_ms=5_000,
        now_ms=0,
    )
    candidate = ResolutionCandidate(
        question_id=compiled.question_id,
        resolution_action=action,
        expected_decision_value=100,
        admissible=True,
        policy_id="policy:one",
    )
    first = runtime.handle_wake(
        AutonomyWakeEvent(kind=AutonomyWakeKind.BUDGET, cursor_id="cursor:budget", sequence=1),
        candidates=(candidate,),
        context=_context(),
    )
    assert first.status is AutonomyRuntimeStatus.EXHAUSTED
    assert first.scanned is True
    assert runtime.healthy_exhausted
    writes_after_stop = sink.write_count
    scans_after_stop = runtime.metrics.scans

    for ordinal in range(1, 4):
        event = runtime.safety_timer_event(now_ms=ordinal * 5_000)
        assert event is not None
        idle = runtime.handle_wake(event, candidates=(candidate,), context=_context())
        assert idle.status is AutonomyRuntimeStatus.EXHAUSTED
        assert idle.reason_codes == ("healthy_exhaustion",)
        assert idle.scanned is False
        assert idle.wrote_state is False
        assert idle.refilled is False
        assert idle.model_called is False

    assert sink.write_count == writes_after_stop
    assert runtime.metrics.scans == scans_after_stop
    assert runtime.metrics.refills == 0
    assert runtime.metrics.model_calls == 0


def test_complete_board_idle_loop_uses_near_zero_cpu() -> None:
    runtime = _complete_runtime(interval_ms=1)
    started = time.process_time()
    cycles = 400
    for ordinal in range(1, cycles + 1):
        event = runtime.safety_timer_event(now_ms=ordinal)
        assert event is not None
        result = runtime.handle_wake(event, context=_context())
        assert result.scanned is False
        assert result.wrote_state is False
        assert result.model_called is False
        assert result.refilled is False
    elapsed = time.process_time() - started
    assert elapsed < 1.0
    assert runtime.metrics.scans == 0
    assert runtime.metrics.writes == 0
    assert runtime.metrics.model_calls == 0
    assert runtime.metrics.refills == 0
    assert runtime.metrics.idle_cycles == cycles


def test_objective_ledger_is_not_refilled_while_idle() -> None:
    action = _action()
    ledger = ObjectiveCognitiveBudgetLedger(_budget(), epoch=3)
    controller = AutonomousMetaController(
        decision_graph=DecisionGraphController.compile(
            repository_id="repo:ipfs-accelerate",
            tree_id="tree:current",
            objective_id="APMC-G000",
            objective_revision="revision:one",
            questions=(_question(action=action, resolved=True),),
        ),
        budget_controller=ledger,
    )
    runtime = AutonomyRuntime(controller=controller, safety_interval_ms=10, now_ms=0)
    before = ledger.snapshot().ledger_id
    for ordinal in range(1, 4):
        event = runtime.safety_timer_event(now_ms=ordinal * 10)
        runtime.handle_wake(event, context=_context())
    assert ledger.snapshot().ledger_id == before
    assert runtime.metrics.refills == 0
    assert runtime.metrics.model_calls == 0


def test_idle_metrics_exclude_window_ticks_from_durable_identity() -> None:
    metrics = AutonomyMetrics()
    first = metrics.durable_identity()
    metrics.record_wake(AutonomyWakeKind.WINDOW, safety_timer=True)
    metrics.record_idle(reason_codes=("unchanged_complete_board",))
    assert metrics.durable_identity() == first
    assert metrics.idle_cycles == 1
    assert metrics.safety_timer_wakes == 1
    metrics.record_scan()
    metrics.record_write()
    assert metrics.durable_identity() == first
    metrics.record_model_action(MetaAction.CALL_LOCAL_SMALL_MODEL)
    assert metrics.durable_identity() != first
    assert metrics.model_calls == 1
