from __future__ import annotations

import json
import time
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
from ipfs_accelerate_py.agent_supervisor.autonomy.metrics import AutonomyMetrics
from ipfs_accelerate_py.agent_supervisor.autonomy.runtime import (
    DEFAULT_SAFETY_INTERVAL_MS,
    AutonomyRuntime,
    AutonomyRuntimeStatus,
    AutonomyWakeEvent,
    AutonomyWakeKind,
    AutonomousMetaController,
    AutonomousMetaControllerError,
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


def test_empty_queue_stale_wake_is_not_board_success() -> None:
    runtime = _complete_runtime(interval_ms=1_000)
    assert runtime.healthy_idle
    stale = runtime.handle_wake(
        AutonomyWakeEvent(
            kind=AutonomyWakeKind.FRESHNESS,
            cursor_id="cursor:empty-stale",
            sequence=1,
            stale=True,
        ),
        candidates=(),
        context=_context(),
    )
    assert stale.status is AutonomyRuntimeStatus.IDLE
    assert stale.reason_codes[0] == "stale_invalidated"
    assert "recovery_plan_delta" in stale.reason_codes
    assert runtime.healthy_idle is False
    assert runtime.recovery_delta_outstanding is True
    stale_record = stale.to_record()
    assert stale_record["recovery_delta_outstanding"] is True
    assert stale_record["completion_authority"] is False

    tick = runtime.safety_timer_event(now_ms=1_000)
    assert tick is not None
    held = runtime.handle_wake(tick, candidates=(), context=_context())
    assert held.status is AutonomyRuntimeStatus.IDLE
    assert held.reason_codes == ("recovery_plan_delta_outstanding",)
    assert held.scanned is False
    assert runtime.healthy_idle is False
    assert runtime.recovery_delta_outstanding is True

    fresh = runtime.handle_wake(
        AutonomyWakeEvent(
            kind=AutonomyWakeKind.TASK,
            cursor_id="cursor:empty-fresh",
            sequence=2,
        ),
        candidates=(),
        context=_context(),
    )
    assert fresh.status is AutonomyRuntimeStatus.IDLE
    assert fresh.reason_codes == ("no_unresolved_mandatory_question",)
    assert runtime.recovery_delta_outstanding is False
    assert runtime.healthy_idle is True
    assert fresh.to_record()["recovery_delta_outstanding"] is False
    assert stale.receipt_id != fresh.receipt_id


def test_similarity_nomination_cannot_complete_an_empty_board() -> None:
    runtime = _complete_runtime(interval_ms=1_000)
    assert runtime.healthy_idle
    similar = runtime.handle_wake(
        AutonomyWakeEvent(
            kind=AutonomyWakeKind.TASK,
            cursor_id="cursor:similar-empty",
            sequence=1,
        ),
        candidates=(),
        context=_context(),
        typesafe_state={
            "similarity_candidates": [{"score": 0.99, "nearest": True}],
        },
    )
    assert similar.status is AutonomyRuntimeStatus.IDLE
    assert "similarity_not_resolution" in similar.reason_codes
    assert similar.similarity_not_resolution is True
    assert similar.to_record()["similarity_not_resolution"] is True
    assert similar.to_record()["completion_authority"] is False
    assert runtime.healthy_idle is False
    assert runtime.similarity_not_resolution is True

    tick = runtime.safety_timer_event(now_ms=1_000)
    assert tick is not None
    held = runtime.handle_wake(tick, candidates=(), context=_context())
    assert held.reason_codes == ("similarity_not_resolution",)
    assert held.scanned is False
    assert runtime.healthy_idle is False

    exact = runtime.handle_wake(
        AutonomyWakeEvent(
            kind=AutonomyWakeKind.TASK,
            cursor_id="cursor:exact-empty",
            sequence=2,
        ),
        candidates=(),
        context=_context(),
        typesafe_state={
            "reuse_decision": {
                "decision": "reuse",
                "exact_match": True,
                "reason_code": "exact_identity_match",
            },
            "similarity_candidates": [{"score": 0.91}],
        },
    )
    assert exact.status is AutonomyRuntimeStatus.IDLE
    assert exact.reason_codes == ("no_unresolved_mandatory_question",)
    assert runtime.similarity_not_resolution is False
    assert runtime.healthy_idle is True
    assert exact.similarity_not_resolution is False


def test_mutated_wave_observations_cannot_complete_an_empty_board() -> None:
    runtime = _complete_runtime(interval_ms=1_000)
    mutated = runtime.handle_wake(
        AutonomyWakeEvent(
            kind=AutonomyWakeKind.TASK,
            cursor_id="cursor:wave-mutated",
            sequence=1,
        ),
        candidates=(),
        context=_context(),
        typesafe_state={
            "refactor_wave": True,
            "declared_profile": "profile:spar-w4",
            "observation_cids": ("obs:1", "obs:2"),
            "wave_observation_cids": ("obs:1", "obs:changed"),
        },
    )
    assert mutated.status is AutonomyRuntimeStatus.IDLE
    assert "observations_not_preserved" in mutated.reason_codes
    assert mutated.observations_not_preserved is True
    assert mutated.to_record()["completion_authority"] is False
    assert runtime.healthy_idle is False

    preserved = runtime.handle_wake(
        AutonomyWakeEvent(
            kind=AutonomyWakeKind.TASK,
            cursor_id="cursor:wave-preserved",
            sequence=2,
        ),
        candidates=(),
        context=_context(),
        typesafe_state={
            "refactor_wave": True,
            "declared_profile": "profile:spar-w4",
            "observation_cids": ("obs:1", "obs:2"),
            "current_observation_cids": ("obs:1", "obs:2"),
        },
    )
    assert preserved.status is AutonomyRuntimeStatus.IDLE
    assert preserved.reason_codes == ("no_unresolved_mandatory_question",)
    assert runtime.observations_not_preserved is False
    assert runtime.healthy_idle is True


def test_negative_episode_cannot_complete_an_empty_board() -> None:
    runtime = _complete_runtime(interval_ms=1_000)
    blocked = runtime.handle_wake(
        AutonomyWakeEvent(
            kind=AutonomyWakeKind.TASK,
            cursor_id="cursor:neg-mem",
            sequence=1,
        ),
        candidates=(),
        context=_context(),
        typesafe_state={
            "negative_episode": True,
            "key_cid": "key:failed",
            "accepted_transition": {
                "pre_cid": "pre:1",
                "action_cid": "act:1",
                "post_cid": "post:1",
                "accepted": True,
                "transition_cid": "key:failed",
            },
        },
    )
    assert blocked.status is AutonomyRuntimeStatus.IDLE
    assert "negative_memory_blocks" in blocked.reason_codes
    assert blocked.negative_memory_blocks is True
    assert blocked.to_record()["completion_authority"] is False
    assert runtime.healthy_idle is False

    cleared = runtime.handle_wake(
        AutonomyWakeEvent(
            kind=AutonomyWakeKind.TASK,
            cursor_id="cursor:neg-clear",
            sequence=2,
        ),
        candidates=(),
        context=_context(),
        typesafe_state={
            "accepted_transition": {
                "pre_cid": "pre:1",
                "action_cid": "act:1",
                "post_cid": "post:1",
                "accepted": True,
                "transition_cid": "tr:ok",
            }
        },
    )
    assert cleared.status is AutonomyRuntimeStatus.IDLE
    assert cleared.reason_codes == ("no_unresolved_mandatory_question",)
    assert runtime.negative_memory_blocks is False
    assert runtime.healthy_idle is True


def test_world_root_cas_cannot_complete_an_empty_board() -> None:
    runtime = _complete_runtime(interval_ms=1_000)
    published = runtime.handle_wake(
        AutonomyWakeEvent(
            kind=AutonomyWakeKind.TASK,
            cursor_id="cursor:world-cas",
            sequence=1,
        ),
        candidates=(),
        context=_context(),
        typesafe_state={
            "world_root_publication": {
                "semantic_world_root_cid": "root:1",
                "cas_completed": True,
            }
        },
    )
    assert published.status is AutonomyRuntimeStatus.IDLE
    assert "world_root_cas_not_completion" in published.reason_codes
    assert published.world_root_cas_not_completion is True
    assert published.to_record()["completion_authority"] is False
    assert runtime.healthy_idle is False

    request = runtime.handle_wake(
        AutonomyWakeEvent(
            kind=AutonomyWakeKind.TASK,
            cursor_id="cursor:world-request",
            sequence=2,
        ),
        candidates=(),
        context=_context(),
        typesafe_state={
            "world_root_publication": {
                "semantic_world_root_cid": "root:1",
                "proposal_only": True,
                "cas_completed": False,
                "completion_authority": False,
            }
        },
    )
    assert request.status is AutonomyRuntimeStatus.IDLE
    assert request.reason_codes == ("no_unresolved_mandatory_question",)
    assert runtime.world_root_cas_not_completion is False
    assert runtime.healthy_idle is True


def test_cross_scc_wave_cannot_complete_an_empty_board() -> None:
    runtime = _complete_runtime(interval_ms=1_000)
    crossed = runtime.handle_wake(
        AutonomyWakeEvent(
            kind=AutonomyWakeKind.TASK,
            cursor_id="cursor:scc-cross",
            sequence=1,
        ),
        candidates=(),
        context=_context(),
        typesafe_state={
            "refactor_wave": True,
            "declared_profile": "profile:spar-w4",
            "observation_cids": ("obs:1",),
            "current_observation_cids": ("obs:1",),
            "write_sccs": ("scc:a", "scc:b"),
        },
    )
    assert crossed.status is AutonomyRuntimeStatus.IDLE
    assert "boundary_contract_required" in crossed.reason_codes
    assert crossed.boundary_contract_required is True
    assert crossed.to_record()["completion_authority"] is False
    assert runtime.healthy_idle is False

    inside = runtime.handle_wake(
        AutonomyWakeEvent(
            kind=AutonomyWakeKind.TASK,
            cursor_id="cursor:scc-in",
            sequence=2,
        ),
        candidates=(),
        context=_context(),
        typesafe_state={
            "refactor_wave": True,
            "declared_profile": "profile:spar-w4",
            "observation_cids": ("obs:1",),
            "current_observation_cids": ("obs:1",),
            "write_sccs": ("scc:a",),
        },
    )
    assert inside.status is AutonomyRuntimeStatus.IDLE
    assert inside.reason_codes == ("no_unresolved_mandatory_question",)
    assert runtime.boundary_contract_required is False
    assert runtime.healthy_idle is True


def test_similar_transition_cannot_complete_an_empty_board() -> None:
    runtime = _complete_runtime(interval_ms=1_000)
    similar = runtime.handle_wake(
        AutonomyWakeEvent(
            kind=AutonomyWakeKind.TASK,
            cursor_id="cursor:transition-similar",
            sequence=1,
        ),
        candidates=(),
        context=_context(),
        typesafe_state={
            "accepted_transition": {
                "pre_cid": "pre:1",
                "action_cid": "act:1",
                "post_cid": "post:1",
                "accepted": True,
            },
            "query_transition": {
                "pre_cid": "pre:1",
                "action_cid": "act:1",
                "post_cid": "post:near",
            },
            "similar_transition": True,
        },
    )
    assert similar.status is AutonomyRuntimeStatus.IDLE
    assert "similarity_not_resolution" in similar.reason_codes
    assert similar.similarity_not_resolution is True
    assert similar.to_record()["completion_authority"] is False
    assert runtime.healthy_idle is False

    exact = runtime.handle_wake(
        AutonomyWakeEvent(
            kind=AutonomyWakeKind.TASK,
            cursor_id="cursor:transition-exact",
            sequence=2,
        ),
        candidates=(),
        context=_context(),
        typesafe_state={
            "accepted_transition": {
                "pre_cid": "pre:1",
                "action_cid": "act:1",
                "post_cid": "post:1",
                "accepted": True,
            }
        },
    )
    assert exact.status is AutonomyRuntimeStatus.IDLE
    assert exact.reason_codes == ("no_unresolved_mandatory_question",)
    assert runtime.similarity_not_resolution is False
    assert runtime.healthy_idle is True


def test_hash_memo_cannot_complete_an_empty_board() -> None:
    runtime = _complete_runtime(interval_ms=1_000)
    assert runtime.healthy_idle
    memo = runtime.handle_wake(
        AutonomyWakeEvent(
            kind=AutonomyWakeKind.TASK,
            cursor_id="cursor:memo-empty",
            sequence=1,
        ),
        candidates=(),
        context=_context(),
        typesafe_state={
            "proof_reuse_decision": {
                "action": "SKIP",
                "reason_code": "proof_cache_hit",
            }
        },
    )
    assert memo.status is AutonomyRuntimeStatus.IDLE
    assert "cold_execution_required" in memo.reason_codes
    assert memo.cold_execution_required is True
    assert memo.to_record()["cold_execution_required"] is True
    assert memo.to_record()["completion_authority"] is False
    assert runtime.healthy_idle is False
    assert runtime.cold_execution_required is True

    tick = runtime.safety_timer_event(now_ms=1_000)
    assert tick is not None
    held = runtime.handle_wake(tick, candidates=(), context=_context())
    assert held.reason_codes == ("cold_execution_required",)
    assert held.scanned is False
    assert runtime.healthy_idle is False

    cold = runtime.handle_wake(
        AutonomyWakeEvent(
            kind=AutonomyWakeKind.TASK,
            cursor_id="cursor:cold-empty",
            sequence=2,
        ),
        candidates=(),
        context=_context(),
        typesafe_state={"cold_execution": True, "action": "RUN"},
    )
    assert cold.status is AutonomyRuntimeStatus.IDLE
    assert cold.reason_codes == ("no_unresolved_mandatory_question",)
    assert runtime.cold_execution_required is False
    assert runtime.healthy_idle is True
    assert cold.cold_execution_required is False


def test_collection_seed_cannot_complete_an_empty_board() -> None:
    runtime = _complete_runtime(interval_ms=1_000)
    seed = runtime.handle_wake(
        AutonomyWakeEvent(
            kind=AutonomyWakeKind.TASK,
            cursor_id="cursor:seed-empty",
            sequence=1,
        ),
        candidates=(),
        context=_context(),
        typesafe_state={
            "pctdd_stage": "collection_seed",
            "hash_memo": True,
            "current_inputs": True,
            "current_effects": True,
            "current_evidence": True,
            "publication_checked": True,
            "proof_reuse_decision": {
                "action": "SKIP",
                "reason_code": "proof_cache_hit",
            },
        },
    )
    assert seed.status is AutonomyRuntimeStatus.IDLE
    assert "cold_execution_required" in seed.reason_codes
    assert seed.cold_execution_required is True
    assert seed.to_record()["completion_authority"] is False
    assert runtime.healthy_idle is False


def test_collapsed_qualification_cannot_complete_an_empty_board() -> None:
    runtime = _complete_runtime(interval_ms=1_000)
    assert runtime.healthy_idle
    collapsed = runtime.handle_wake(
        AutonomyWakeEvent(
            kind=AutonomyWakeKind.TASK,
            cursor_id="cursor:qualify-empty",
            sequence=1,
        ),
        candidates=(),
        context=_context(),
        typesafe_state={
            "qualification": {
                "location": "specification_only",
                "implementation": "present",
                "integration": "adapter_only",
                "validation": "reported",
                "rollout": "shadow",
                "landing_date": "2026-10-01",
            }
        },
    )
    assert collapsed.status is AutonomyRuntimeStatus.IDLE
    assert "qualification_incomplete" in collapsed.reason_codes
    assert collapsed.qualification_incomplete is True
    assert collapsed.to_record()["qualification_incomplete"] is True
    assert collapsed.to_record()["completion_authority"] is False
    assert runtime.healthy_idle is False

    tick = runtime.safety_timer_event(now_ms=1_000)
    assert tick is not None
    held = runtime.handle_wake(tick, candidates=(), context=_context())
    assert held.reason_codes == ("qualification_incomplete",)
    assert held.scanned is False
    assert runtime.healthy_idle is False

    sufficient = runtime.handle_wake(
        AutonomyWakeEvent(
            kind=AutonomyWakeKind.TASK,
            cursor_id="cursor:qualify-sufficient",
            sequence=2,
        ),
        candidates=(),
        context=_context(),
        typesafe_state={
            "qualification": {
                "location": "main",
                "implementation": "present",
                "integration": "called",
                "validation": "reproduced",
                "rollout": "guarded",
                "paper_claim": {
                    "mechanism": "H.2 independent dimensions",
                    "population": "autonomy idle path",
                    "scope": "existing board owner",
                    "limitations": "not a new campaign",
                },
            }
        },
    )
    assert sufficient.status is AutonomyRuntimeStatus.IDLE
    assert sufficient.reason_codes == ("no_unresolved_mandatory_question",)
    assert runtime.qualification_incomplete is False
    assert runtime.healthy_idle is True
    assert sufficient.qualification_incomplete is False


def test_program_catalog_cannot_complete_an_empty_board() -> None:
    runtime = _complete_runtime(interval_ms=1_000)
    assert runtime.healthy_idle
    catalog = runtime.handle_wake(
        AutonomyWakeEvent(
            kind=AutonomyWakeKind.TASK,
            cursor_id="cursor:programs-empty",
            sequence=1,
        ),
        candidates=(),
        context=_context(),
        typesafe_state={"program_catalog": True},
    )
    assert catalog.status is AutonomyRuntimeStatus.IDLE
    assert "qualification_incomplete" in catalog.reason_codes
    assert catalog.qualification_incomplete is True
    assert catalog.to_record()["completion_authority"] is False
    assert runtime.healthy_idle is False

    tick = runtime.safety_timer_event(now_ms=1_000)
    assert tick is not None
    held = runtime.handle_wake(tick, candidates=(), context=_context())
    assert held.reason_codes == ("qualification_incomplete",)
    assert runtime.healthy_idle is False


def test_restart_preserves_outstanding_recovery_plan_delta() -> None:
    runtime = _complete_runtime(interval_ms=1_000)
    runtime.handle_wake(
        AutonomyWakeEvent(
            kind=AutonomyWakeKind.FRESHNESS,
            cursor_id="cursor:restart-stale",
            sequence=1,
            stale=True,
        ),
        candidates=(),
        context=_context(),
    )
    assert runtime.recovery_delta_outstanding is True
    snapshot = runtime.snapshot_json()
    recovered = AutonomyRuntime.from_snapshot(
        snapshot,
        budget_loader=lambda value: _FakeBudgetController.from_snapshot(dict(value)),
    )
    assert recovered.snapshot_json() == snapshot
    assert recovered.recovery_delta_outstanding is True
    assert recovered.healthy_idle is False
    tick = recovered.safety_timer_event(now_ms=1_000)
    assert tick is not None
    held = recovered.handle_wake(tick, candidates=(), context=_context())
    assert held.reason_codes == ("recovery_plan_delta_outstanding",)
    assert recovered.healthy_idle is False

    forged = json.loads(snapshot)
    forged["recovery_delta_outstanding"] = False
    with pytest.raises(AutonomousMetaControllerError, match="identity mismatch"):
        AutonomyRuntime.from_snapshot(
            forged,
            budget_loader=lambda value: _FakeBudgetController.from_snapshot(dict(value)),
        )


def test_wake_state_coordinator_consume_clears_board_owner(tmp_path) -> None:
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        duckdb_available,
        open_database_coordinator,
    )

    if not duckdb_available():
        pytest.skip("DuckDB is required for board-owner consume")

    coordinator = open_database_coordinator(tmp_path / "coordination.duckdb")
    try:
        runtime = _complete_runtime(interval_ms=1_000)
        stale = runtime.handle_wake(
            AutonomyWakeEvent(
                kind=AutonomyWakeKind.FRESHNESS,
                cursor_id="cursor:coord-stale",
                sequence=1,
                stale=True,
            ),
            candidates=(),
            context=_context(),
            typesafe_state={"claim_coordinator": coordinator},
        )
        assert "recovery_plan_delta" in stale.reason_codes
        assert coordinator.outstanding_recovery_plan_delta_count() == 1
        fresh = runtime.handle_wake(
            AutonomyWakeEvent(
                kind=AutonomyWakeKind.TASK,
                cursor_id="cursor:coord-fresh",
                sequence=2,
            ),
            candidates=(),
            context=_context(),
            typesafe_state={"claim_coordinator": coordinator},
        )
        assert fresh.reason_codes == ("no_unresolved_mandatory_question",)
        assert runtime.recovery_delta_outstanding is False
        assert coordinator.outstanding_recovery_plan_delta_count() == 0
    finally:
        coordinator.close()


def test_restore_reconciles_outstanding_from_board_owner(tmp_path) -> None:
    from ipfs_accelerate_py.agent_supervisor.autonomy.recovery_board_fence import (
        build_recovery_delta_items,
    )
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        duckdb_available,
        open_database_coordinator,
    )
    from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
        LifecycleState,
    )

    if not duckdb_available():
        pytest.skip("DuckDB is required for board-owner restore")

    coordinator = open_database_coordinator(tmp_path / "coordination.duckdb")
    try:
        items = build_recovery_delta_items(
            {
                "action": "invalidate_stale_evidence",
                "admit": False,
                "stall_reason": "stale_evidence",
            },
            target_cid="task:restore",
            lifecycle=LifecycleState.CLAIMED,
        )
        recorded = coordinator.record_recovery_plan_delta(
            task_cid="task:restore",
            items=tuple(item.to_dict() for item in items),
            now_ms=3,
        )
        idle = _complete_runtime(interval_ms=1_000)
        assert idle.recovery_delta_outstanding is False
        recovered = AutonomyRuntime.from_snapshot(
            idle.snapshot_json(),
            budget_loader=lambda value: _FakeBudgetController.from_snapshot(dict(value)),
            claim_coordinator=coordinator,
        )
        assert recovered.recovery_delta_outstanding is True
        assert recovered.healthy_idle is False
        coordinator.consume_recovery_plan_delta(
            delta_event_id=recorded["event_id"],
            task_cid="task:restore",
            now_ms=4,
        )
        recovered.bind_claim_coordinator(coordinator)
        assert recovered.recovery_delta_outstanding is False
    finally:
        coordinator.close()


def test_dispatch_wake_uses_bound_claim_coordinator(tmp_path) -> None:
    from ipfs_accelerate_py.agent_supervisor.autonomy.typesafe_runtime_adapter import (
        dispatch_autonomy_wake,
    )
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        duckdb_available,
        open_database_coordinator,
    )

    if not duckdb_available():
        pytest.skip("DuckDB is required for board-owner consume")

    coordinator = open_database_coordinator(tmp_path / "coordination.duckdb")
    try:
        runtime = _complete_runtime(interval_ms=1_000)
        runtime.bind_claim_coordinator(coordinator)
        stale = dispatch_autonomy_wake(
            runtime,
            AutonomyWakeEvent(
                kind=AutonomyWakeKind.FRESHNESS,
                cursor_id="cursor:dispatch-stale",
                sequence=1,
                stale=True,
            ),
            candidates=(),
            context=_context(),
        )
        assert stale.model_called is False
        assert stale.authorizes_effect is False
        stale_dict = stale.to_dict()
        assert stale_dict["authorizes_effect"] is False
        assert stale_dict["completion_authority"] is False
        assert stale_dict["recovery_delta_outstanding"] is True
        assert coordinator.outstanding_recovery_plan_delta_count() == 1
        fresh = dispatch_autonomy_wake(
            runtime,
            AutonomyWakeEvent(
                kind=AutonomyWakeKind.TASK,
                cursor_id="cursor:dispatch-fresh",
                sequence=2,
            ),
            candidates=(),
            context=_context(),
        )
        assert fresh.result.reason_codes == ("no_unresolved_mandatory_question",)
        assert fresh.to_dict()["recovery_delta_outstanding"] is False
        assert coordinator.outstanding_recovery_plan_delta_count() == 0
    finally:
        coordinator.close()


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
