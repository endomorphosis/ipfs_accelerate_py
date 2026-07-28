"""ASI-170: staged supervisor usage rollout modes and promotion gates."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.supervisor_usage_rollout import (
    DEFAULT_LIVE_BUDGET_MICROS,
    LIVE_BUDGET_ENV,
    LIVE_ENV,
    SUPERVISOR_USAGE_BEHAVIOR_ID,
    SUPERVISOR_USAGE_ROLLOUT_REQUIREMENT_ID,
    SupervisorUsagePairedReport,
    SupervisorUsageRolloutEvaluation,
    SupervisorUsageRolloutMode,
    build_default_binding,
    build_default_policy,
    build_paired_report,
    discover_schemas,
    evaluate_supervisor_usage_rollout,
    live_budget_micros,
    live_smoke_enabled,
    mode_alters_execution,
    mode_is_non_selecting,
    verify_supervisor_usage_rollout,
)


def _evaluations(
    *,
    tree_id: str = "tree:supervisor-usage-rollout",
) -> tuple[
    SupervisorUsageRolloutEvaluation,
    SupervisorUsageRolloutEvaluation,
]:
    qualification = SupervisorUsageRolloutEvaluation(
        "evaluation:qualification@1",
        "2026-07-28T12:00:00Z",
        build_paired_report(
            observation_label="qualification",
            tree_id=tree_id,
            observed_at="2026-07-28T12:00:00Z",
        ),
    )
    current = SupervisorUsageRolloutEvaluation(
        "evaluation:current@1",
        "2026-07-29T12:00:00Z",
        build_paired_report(
            observation_label="current",
            tree_id=tree_id,
            observed_at="2026-07-29T12:00:00Z",
        ),
    )
    return qualification, current


def test_requirement_and_mode_vocabulary() -> None:
    catalog = discover_schemas()
    assert catalog["requirement_id"] == SUPERVISOR_USAGE_ROLLOUT_REQUIREMENT_ID
    assert catalog["is_completion_evidence"] == "false"
    assert catalog["authorizes_usage"] == "false"
    modes = {m.value for m in SupervisorUsageRolloutMode}
    assert modes == {"off", "observe", "shadow", "assist", "enforce"}
    assert mode_is_non_selecting("off")
    assert mode_is_non_selecting("observe")
    assert mode_is_non_selecting("shadow")
    assert mode_alters_execution("assist")
    assert mode_alters_execution("enforce")


def test_off_observe_shadow_assist_and_enforce_modes() -> None:
    qualification, current = _evaluations()
    binding = build_default_binding()
    policy = build_default_policy(approve_enforce=True, approve_assist=True)

    off = evaluate_supervisor_usage_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=SupervisorUsageRolloutMode.OFF,
    )
    assert off.effective_mode is SupervisorUsageRolloutMode.OFF
    assert not off.rollback_applied
    assert off.observed_usage_retained

    observe = evaluate_supervisor_usage_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode="observe",
    )
    assert observe.effective_mode is SupervisorUsageRolloutMode.OBSERVE

    shadow = evaluate_supervisor_usage_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode="shadow",
    )
    assert shadow.effective_mode is SupervisorUsageRolloutMode.SHADOW

    assist = evaluate_supervisor_usage_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode="assist",
        operator_authority_granted=True,
    )
    assert assist.effective_mode is SupervisorUsageRolloutMode.ASSIST
    assert assist.qualification_passed

    enforce = evaluate_supervisor_usage_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode="enforce",
        current_evaluation=current,
        operator_authority_granted=True,
    )
    assert enforce.effective_mode is SupervisorUsageRolloutMode.ENFORCE
    assert enforce.enforce_ready
    assert not enforce.authoritative
    assert not enforce.completion_authoritative
    assert verify_supervisor_usage_rollout(
        enforce,
        qualification,
        binding=binding,
        policy=policy,
        current_evaluation=current,
        operator_authority_granted=True,
    )


def test_assist_requires_operator_authority() -> None:
    qualification, _current = _evaluations()
    binding = build_default_binding()
    policy = build_default_policy(approve_assist=True)

    denied = evaluate_supervisor_usage_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode="assist",
        operator_authority_granted=False,
    )
    assert denied.effective_mode is SupervisorUsageRolloutMode.SHADOW
    assert denied.rollback_applied
    assert "operator-authority-required" in denied.reason_codes


def test_enforce_requires_later_distinct_passing_paired_report() -> None:
    qualification, current = _evaluations()
    binding = build_default_binding()
    policy = build_default_policy(approve_enforce=True)

    missing = evaluate_supervisor_usage_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode="enforce",
    )
    assert missing.effective_mode is SupervisorUsageRolloutMode.SHADOW
    assert "current-evaluation-required" in missing.reason_codes

    unapproved = evaluate_supervisor_usage_rollout(
        qualification,
        binding=binding,
        policy=build_default_policy(approve_enforce=False),
        desired_mode="enforce",
        current_evaluation=current,
    )
    assert unapproved.effective_mode is SupervisorUsageRolloutMode.SHADOW
    assert "mode-not-policy-approved" in unapproved.reason_codes

    # Same evaluation identity is not a later distinct observation.
    same = evaluate_supervisor_usage_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode="enforce",
        current_evaluation=qualification,
    )
    assert same.effective_mode is SupervisorUsageRolloutMode.SHADOW
    assert "current-evaluation-not-distinct" in same.reason_codes


def test_binding_or_safety_regression_returns_to_shadow_retaining_usage() -> None:
    qualification, current = _evaluations()
    binding = build_default_binding()
    policy = build_default_policy(approve_enforce=True)
    stale = replace(binding, tree_id="tree:different-current-root")

    decision = evaluate_supervisor_usage_rollout(
        qualification,
        binding=stale,
        policy=policy,
        desired_mode="enforce",
        current_evaluation=current,
    )
    assert decision.effective_mode is SupervisorUsageRolloutMode.SHADOW
    assert decision.rollback_applied
    assert decision.observed_usage_retained
    assert "stale-binding:tree_id" in decision.reason_codes
    assert decision.affected_behavior_ids == (SUPERVISOR_USAGE_BEHAVIOR_ID,)


def test_metric_and_population_regression_blocks_enforce() -> None:
    qualification, current = _evaluations()
    binding = build_default_binding()
    policy = build_default_policy(approve_enforce=True)

    # Narrow the chaos population on the current observation.
    narrowed_report = SupervisorUsagePairedReport(
        observation_label=current.report.observation_label,
        e2e_receipts=current.report.e2e_receipts,
        chaos_receipts=current.report.chaos_receipts[:-1],
        observed_at=current.report.observed_at,
        tree_id=current.report.tree_id,
        max_cost_micros=current.report.max_cost_micros,
        max_latency_ms=current.report.max_latency_ms,
        min_quality_bps=current.report.min_quality_bps,
        max_wait_ms=current.report.max_wait_ms,
    )
    narrowed = SupervisorUsageRolloutEvaluation(
        "evaluation:narrowed@1",
        "2026-07-30T12:00:00Z",
        narrowed_report,
    )
    blocked = evaluate_supervisor_usage_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode="enforce",
        current_evaluation=narrowed,
    )
    assert blocked.effective_mode is SupervisorUsageRolloutMode.SHADOW
    assert blocked.rollback_applied
    assert any(
        code.startswith("current:missing-chaos-boundary:")
        or code == "benchmark-population-narrowed"
        for code in blocked.reason_codes
    )


def test_distributed_enforcement_fails_closed_without_fenced_coordinator() -> None:
    qualification, current = _evaluations()
    binding = build_default_binding()
    policy = build_default_policy(approve_enforce=True)

    decision = evaluate_supervisor_usage_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode="enforce",
        current_evaluation=current,
        fenced_coordinator_available=False,
        distributed_enforcement_requested=True,
    )
    assert decision.distributed_fail_closed
    assert decision.effective_mode is SupervisorUsageRolloutMode.OFF
    assert "distributed-enforcement-fail-closed" in decision.reason_codes
    assert decision.observed_usage_retained


def test_cost_latency_quality_limits_gate_enforce() -> None:
    qualification, current = _evaluations()
    binding = build_default_binding()
    tight = build_default_policy(approve_enforce=True)
    tight = replace(tight, max_cost_micros=0, max_latency_ms=0, min_quality_bps=10_000)

    decision = evaluate_supervisor_usage_rollout(
        qualification,
        binding=binding,
        policy=tight,
        desired_mode="enforce",
        current_evaluation=current,
    )
    assert decision.effective_mode is SupervisorUsageRolloutMode.SHADOW
    assert any(
        code in decision.reason_codes
        for code in ("cost_limit", "latency_limit", "quality_limit")
    )


def test_live_smoke_is_opt_in_and_budget_capped() -> None:
    assert live_smoke_enabled() is False
    assert live_budget_micros() == DEFAULT_LIVE_BUDGET_MICROS
    previous = os.environ.get(LIVE_ENV)
    previous_budget = os.environ.get(LIVE_BUDGET_ENV)
    try:
        os.environ[LIVE_ENV] = "1"
        os.environ[LIVE_BUDGET_ENV] = "999999"
        assert live_smoke_enabled() is True
        # Hard-capped at default reviewed budget.
        assert live_budget_micros() == DEFAULT_LIVE_BUDGET_MICROS
    finally:
        if previous is None:
            os.environ.pop(LIVE_ENV, None)
        else:
            os.environ[LIVE_ENV] = previous
        if previous_budget is None:
            os.environ.pop(LIVE_BUDGET_ENV, None)
        else:
            os.environ[LIVE_BUDGET_ENV] = previous_budget


@pytest.mark.skipif(
    not live_smoke_enabled(),
    reason="opt-in live supervisor usage smoke disabled",
)
def test_opt_in_live_budget_cap_marker() -> None:
    """Environment-gated live marker; never runs in default CI."""

    assert live_budget_micros() <= DEFAULT_LIVE_BUDGET_MICROS
