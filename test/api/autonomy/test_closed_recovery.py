from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.autonomy.closed_recovery import (
    apply_recovery_plan,
    plan_closed_recovery,
    should_recover,
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


def test_stale_without_replan_invalidates_without_typesafe(
    monkeypatch,
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    view = plan_closed_recovery(
        reason="stale_evidence",
        wake_kind="freshness",
        candidates=(_candidate(MetaAction.RUN_LOCAL_STATIC_ANALYSIS),),
        call_typesafe=True,
    )
    assert view["action"] == "invalidate_stale_evidence"
    assert view["admit"] is False
    assert view["accepted_as_authority"] is False
    assert view["typesafe_required"] is False
    assert view["invents_meta_action"] is False


def test_stale_with_declared_replan_admits_replan_without_typesafe(
    monkeypatch,
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    view = plan_closed_recovery(
        reason="stale_evidence",
        wake_kind="freshness",
        candidates=(_candidate(MetaAction.REPLAN_AFFECTED_SUFFIX),),
        call_typesafe=False,
    )
    assert view["action"] == "replan_suffix"
    assert view["admit"] is True
    assert view["meta_action"] == MetaAction.REPLAN_AFFECTED_SUFFIX.value
    assert view["typesafe_used"] is False


def test_undeclared_human_recovery_is_not_invented(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    view = plan_closed_recovery(
        reason="needs_human",
        wake_kind="human",
        candidates=(_candidate(MetaAction.RUN_LOCAL_STATIC_ANALYSIS),),
        call_typesafe=False,
    )
    assert view["action"] == "preserve"
    assert view["admit"] is False
    assert view["invents_meta_action"] is False


def test_provider_down_retries_without_typesafe(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    view = plan_closed_recovery(
        reason="provider_down",
        wake_kind="provider",
        candidates=(_candidate(MetaAction.RUN_LOCAL_STATIC_ANALYSIS),),
        call_typesafe=False,
    )
    assert view["action"] == "retry_provider"
    assert view["admit"] is False
    disposition, _, reasons = apply_recovery_plan(view, (), stale=False)
    assert disposition == "idle"
    assert reasons == ("retry_provider",)


def test_false_missing_preserves_and_does_not_replan(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    view = plan_closed_recovery(
        reason="notification",
        wake_kind="task",
        candidates=(_candidate(MetaAction.REPLAN_AFFECTED_SUFFIX),),
        typesafe_state={"watchdog_label": "false_missing"},
        call_typesafe=False,
    )
    assert view["action"] == "preserve"
    assert view["admit"] is False
    assert should_recover(
        stale=False,
        reason="notification",
        state={"watchdog_label": "false_missing"},
    )
    disposition, _, reasons = apply_recovery_plan(view, (), stale=False)
    assert disposition == "continue"
    assert "closed_recovery_preserve" in reasons


def test_notification_task_wake_does_not_recover() -> None:
    assert should_recover(stale=False, wake_kind="task", reason="notification") is False
    assert should_recover(stale=True, wake_kind="task", reason="notification") is True
