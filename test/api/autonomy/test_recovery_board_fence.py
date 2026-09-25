from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomy.closed_recovery import (
    apply_recovery_plan,
    last_closed_recovery,
    plan_closed_recovery,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
    AuthorityClass,
    CancellationBehavior,
    MetaAction,
    PrivacyClass,
    ResolutionAction,
    ResolutionCandidate,
    ResolutionEvidenceKind,
    RiskClass,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.recovery_board_fence import (
    apply_recovery_board_fence,
    build_recovery_delta_items,
    recovery_needs_plan_delta,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
    DeltaEffectClass,
    LifecycleState,
    PlanDeltaOperation,
)


class _LiveClaimError(Exception):
    """Mirrors DuckDB refuse-to-mutate a still-live claim."""


class _FakeCoordinator:
    def __init__(self, *, live: bool = False) -> None:
        self.live = live
        self.expired: list[object] = []
        self.recorded: list[tuple[str, tuple[object, ...]]] = []

    def expire_task_claim(self, claim: object, now_ms: int | None = None) -> dict[str, str]:
        if self.live:
            raise _LiveClaimError("still-live claim fails without mutation")
        self.expired.append({"claim": claim, "now_ms": now_ms})
        return {"state": "expired"}

    def record_recovery_plan_delta(
        self, *, task_cid: str, items: tuple[object, ...], **_kwargs: object
    ) -> None:
        self.recorded.append((task_cid, tuple(items)))


def _candidate(action: MetaAction) -> ResolutionCandidate:
    resolution = ResolutionAction(
        action=action,
        precondition_ids=("tree-current",),
        expected_evidence_kind=ResolutionEvidenceKind.STATIC_ANALYSIS,
        expected_uncertainty_reduction_bp=1,
        token_cost=0,
        latency_cost_ms=1,
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
    return ResolutionCandidate(
        question_id="q1",
        resolution_action=resolution,
        expected_decision_value=1,
        admissible=True,
        policy_id="policy:one",
    )


def test_preserve_and_retry_do_not_need_a_plan_delta(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    preserve = plan_closed_recovery(
        reason="false_missing",
        wake_kind="task",
        candidates=(_candidate(MetaAction.REPLAN_AFFECTED_SUFFIX),),
        typesafe_state={"watchdog_label": "false_missing"},
        call_typesafe=False,
    )
    retry = plan_closed_recovery(
        reason="provider_down",
        wake_kind="provider",
        candidates=(_candidate(MetaAction.RUN_LOCAL_STATIC_ANALYSIS),),
        call_typesafe=False,
    )
    assert recovery_needs_plan_delta(preserve) is False
    assert recovery_needs_plan_delta(retry) is False
    assert build_recovery_delta_items(preserve) == ()
    assert build_recovery_delta_items(retry) == ()


def test_invalidate_emits_claimed_safe_uncertainty_and_lifecycle_request() -> None:
    recovery = {
        "action": "invalidate_stale_evidence",
        "admit": False,
        "stall_reason": "stale_evidence",
        "accepted_as_authority": False,
    }
    items = build_recovery_delta_items(
        recovery,
        target_cid="task:stale",
        lifecycle=LifecycleState.CLAIMED,
    )
    assert [item.operation for item in items] == [
        PlanDeltaOperation.RECORD_UNCERTAINTY,
        PlanDeltaOperation.REQUEST_LIFECYCLE_ACTION,
    ]
    assert {item.effect_class for item in items} == {
        DeltaEffectClass.EVIDENCE_ONLY,
        DeltaEffectClass.LIFECYCLE_REQUEST,
    }
    assert all(item.expected_target_lifecycle is LifecycleState.CLAIMED for item in items)
    assert all(item.target_cid == "task:stale" for item in items)
    assert all(item.provenance["accepted_as_authority"] is False for item in items)
    assert all(item.provenance["completes_task"] is False for item in items)


def test_admitted_replan_requests_suffix_without_mutating_history() -> None:
    recovery = {
        "action": "replan_suffix",
        "admit": True,
        "stall_reason": "stale_evidence",
        "accepted_as_authority": False,
    }
    items = build_recovery_delta_items(
        recovery,
        target_cid="task:suffix",
        lifecycle=LifecycleState.RUNNING,
    )
    ops = [item.operation for item in items]
    assert PlanDeltaOperation.RECORD_UNCERTAINTY in ops
    assert PlanDeltaOperation.REQUEST_LIFECYCLE_ACTION in ops
    assert PlanDeltaOperation.AMEND_UNSTARTED_TASK not in ops
    assert PlanDeltaOperation.SUPERSEDE_UNSTARTED_TASK not in ops
    replan = next(
        item for item in items if "replan-suffix" in item.expected_effects
    )
    assert replan.effect_class is DeltaEffectClass.LIFECYCLE_REQUEST
    assert replan.expected_target_lifecycle is LifecycleState.RUNNING


def test_completed_history_still_allows_claimed_safe_ops() -> None:
    recovery = {
        "action": "invalidate_stale_evidence",
        "admit": False,
        "stall_reason": "stale_evidence",
    }
    items = build_recovery_delta_items(
        recovery,
        target_cid="task:done",
        lifecycle=LifecycleState.COMPLETED,
    )
    assert items
    assert all(
        item.operation
        in {
            PlanDeltaOperation.RECORD_UNCERTAINTY,
            PlanDeltaOperation.REQUEST_LIFECYCLE_ACTION,
        }
        for item in items
    )


def test_invalidate_expires_wall_clock_claim_and_records_delta() -> None:
    recovery = {
        "action": "invalidate_stale_evidence",
        "admit": False,
        "stall_reason": "stale_evidence",
    }
    coordinator = _FakeCoordinator()
    claim = {"claim_id": "claim:1", "task_cid": "task:stale"}
    fence = apply_recovery_board_fence(
        recovery,
        target_cid="task:stale",
        coordinator=coordinator,
        claim=claim,
        now_ms=12,
    )
    assert fence["accepted_as_authority"] is False
    assert fence["completes_task"] is False
    assert fence["claim_fenced"] is True
    assert fence["claim_fence_reason"] == "expired"
    assert "recovery_plan_delta" in fence["reason_codes"]
    assert "claim_fenced" in fence["reason_codes"]
    assert "recovery_delta_recorded" in fence["reason_codes"]
    assert coordinator.expired == [{"claim": claim, "now_ms": 12}]
    assert coordinator.recorded[0][0] == "task:stale"
    assert coordinator.recorded[0][1]


def test_live_claim_is_not_mutated() -> None:
    recovery = {
        "action": "invalidate_stale_evidence",
        "admit": False,
        "stall_reason": "stale_evidence",
    }
    coordinator = _FakeCoordinator(live=True)
    fence = apply_recovery_board_fence(
        recovery,
        target_cid="task:live",
        coordinator=coordinator,
        claim={"claim_id": "claim:live"},
    )
    assert fence["claim_fenced"] is False
    assert fence["claim_fence_reason"] == "claim_still_live"
    assert "claim_still_live" in fence["reason_codes"]
    assert fence["delta_item_cids"]
    assert coordinator.expired == []


def test_admitted_replan_does_not_expire_the_live_claim(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    view = plan_closed_recovery(
        reason="stale_evidence",
        wake_kind="freshness",
        candidates=(_candidate(MetaAction.REPLAN_AFFECTED_SUFFIX),),
        call_typesafe=False,
    )
    coordinator = _FakeCoordinator()
    claim = {"claim_id": "claim:replan"}
    disposition, matching, reasons = apply_recovery_plan(
        view,
        (_candidate(MetaAction.REPLAN_AFFECTED_SUFFIX),),
        stale=True,
        target_cid="task:replan",
        coordinator=coordinator,
        claim=claim,
    )
    assert disposition == "admit"
    assert matching
    assert "recovery_plan_delta" in reasons
    assert last_closed_recovery()["delta_operations"]
    assert last_closed_recovery()["accepted_as_authority"] is False
    assert coordinator.expired == []


def test_apply_recovery_plan_appends_delta_without_changing_idle_head(
    monkeypatch,
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    view = plan_closed_recovery(
        reason="stale_evidence",
        wake_kind="freshness",
        candidates=(_candidate(MetaAction.RUN_LOCAL_STATIC_ANALYSIS),),
        call_typesafe=False,
    )
    disposition, _, reasons = apply_recovery_plan(view, (), stale=True)
    assert disposition == "idle"
    assert reasons[0] == "stale_invalidated"
    assert "recovery_plan_delta" in reasons
    assert "claim_fence_deferred" in reasons
    recorded = last_closed_recovery()
    assert recorded["claim_fenced"] is False
    assert recorded["delta_item_cids"]
    assert recorded["accepted_as_authority"] is False
    assert recorded["completes_task"] is False


def test_invalidate_persists_expired_claim_in_duckdb(tmp_path) -> None:
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        LeaseState,
        duckdb_available,
        open_database_coordinator,
    )

    if not duckdb_available():
        pytest.skip("DuckDB is required for the board-owner claim fence")

    class _Clock:
        def __init__(self) -> None:
            self.now = 1_000_000

        def __call__(self) -> int:
            return int(self.now)

        def advance(self, ms: int) -> None:
            self.now += int(ms)

    clock = _Clock()
    coordinator = open_database_coordinator(
        tmp_path / "coordination.duckdb",
        clock_ms=clock,
        default_lease_ms=10_000,
    )
    try:
        coordinator.register_task(task_cid="task:board-fence", task_id="FENCE")
        claim = coordinator.claim_task(
            task_cid="task:board-fence",
            owner_session_id="session:recovery",
            idempotency_key="recovery-fence",
        )
        recovery = {
            "action": "invalidate_stale_evidence",
            "admit": False,
            "stall_reason": "stale_evidence",
        }
        live = apply_recovery_board_fence(
            recovery,
            target_cid="task:board-fence",
            coordinator=coordinator,
            claim=claim,
            now_ms=clock.now,
        )
        assert live["claim_fenced"] is False
        assert live["claim_fence_reason"] == "claim_still_live"
        assert coordinator.get_task_claim(claim.claim_id).state is LeaseState.ACCEPTED

        clock.advance(10_000)
        expired = apply_recovery_board_fence(
            recovery,
            target_cid="task:board-fence",
            coordinator=coordinator,
            claim=claim,
            now_ms=clock.now,
        )
        assert expired["claim_fenced"] is True
        assert expired["delta_item_cids"]
        assert coordinator.get_task_claim(claim.claim_id).state is LeaseState.EXPIRED
        replay = apply_recovery_board_fence(
            recovery,
            target_cid="task:board-fence",
            coordinator=coordinator,
            claim=claim,
            now_ms=clock.now,
        )
        assert replay["claim_fenced"] is True
        replacement = coordinator.claim_task(
            task_cid="task:board-fence",
            owner_session_id="session:next",
            idempotency_key="recovery-fence-next",
        )
        assert replacement.attempt_number == claim.attempt_number + 1
        assert coordinator.get_task_claim(replacement.claim_id).state is LeaseState.ACCEPTED
        from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
            LeaseKind,
            RECOVERY_PLAN_DELTA_EVENT,
            exclusive_scope_key,
        )

        scope_key = exclusive_scope_key(
            lease_kind=LeaseKind.TASK,
            scope="task:board-fence",
            task_cid="task:board-fence",
        )
        events = [
            event
            for event in coordinator.lease_events(scope_key=scope_key)
            if event["event_type"] == RECOVERY_PLAN_DELTA_EVENT
        ]
        assert len(events) == 1
        assert events[0]["body"]["accepted_as_authority"] is False
        assert events[0]["body"]["completes_task"] is False
        assert events[0]["body"]["task_cid"] == "task:board-fence"
        assert "record_uncertainty" in events[0]["body"]["operations"]
    finally:
        coordinator.close()


def test_record_recovery_plan_delta_is_idempotent_and_claimed_safe(tmp_path) -> None:
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        DatabaseCoordinationError,
        RECOVERY_PLAN_DELTA_EVENT,
        duckdb_available,
        open_database_coordinator,
    )

    if not duckdb_available():
        pytest.skip("DuckDB is required for the board-owner claim fence")

    items = build_recovery_delta_items(
        {
            "action": "invalidate_stale_evidence",
            "admit": False,
            "stall_reason": "stale_evidence",
        },
        target_cid="task:delta",
        lifecycle=LifecycleState.CLAIMED,
    )
    coordinator = open_database_coordinator(tmp_path / "coordination.duckdb")
    try:
        first = coordinator.record_recovery_plan_delta(
            task_cid="task:delta",
            items=tuple(item.to_dict() for item in items),
            now_ms=5,
        )
        second = coordinator.record_recovery_plan_delta(
            task_cid="task:delta",
            items=tuple(item.to_dict() for item in items),
            now_ms=9,
        )
        assert first["recorded"] is True
        assert first["accepted_as_authority"] is False
        assert first["completes_task"] is False
        assert second["idempotent"] is True
        assert second["event_id"] == first["event_id"]
        events = [
            event
            for event in coordinator.lease_events()
            if event["event_type"] == RECOVERY_PLAN_DELTA_EVENT
        ]
        assert len(events) == 1
        with pytest.raises(DatabaseCoordinationError, match="claimed-safe"):
            coordinator.record_recovery_plan_delta(
                task_cid="task:delta",
                items=({"operation": "amend_unstarted_task", "item_key": "bad"},),
            )
        empty = coordinator.record_recovery_plan_delta(task_cid="task:delta", items=())
        assert empty["recorded"] is False
        assert coordinator.outstanding_recovery_plan_delta_count() == 1
        consumed = coordinator.consume_recovery_plan_delta(
            delta_event_id=first["event_id"],
            task_cid="task:delta",
            now_ms=11,
        )
        replay_consume = coordinator.consume_recovery_plan_delta(
            delta_event_id=first["event_id"],
            task_cid="task:delta",
            now_ms=12,
        )
        assert consumed["recorded"] is True
        assert consumed["completes_task"] is False
        assert replay_consume["idempotent"] is True
        assert coordinator.outstanding_recovery_plan_delta_count() == 0
    finally:
        coordinator.close()
