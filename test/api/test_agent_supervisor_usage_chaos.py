"""ASI-170: chaos injection across supervisor usage boundaries."""

from __future__ import annotations

from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.supervisor_usage_rollout import (
    REQUIRED_CHAOS_BOUNDARIES,
    ChaosBoundary,
    FaultOutcome,
    SupervisorUsagePairedReport,
    build_paired_report,
    run_chaos_population,
)


def test_every_required_chaos_boundary_is_exercised() -> None:
    receipts = run_chaos_population(observation_label="chaos")
    covered = {r.boundary for r in receipts}
    assert covered == set(REQUIRED_CHAOS_BOUNDARIES)
    assert len(receipts) == len(REQUIRED_CHAOS_BOUNDARIES)


def test_chaos_outcomes_are_typed_and_pass_safety() -> None:
    receipts = run_chaos_population(observation_label="typed")
    outcomes = {r.outcome for r in receipts}
    assert outcomes <= set(FaultOutcome)
    for receipt in receipts:
        assert receipt.passed, (
            receipt.boundary,
            receipt.outcome,
            receipt.reason_codes,
            receipt.overshoot,
            receipt.double_charge,
            receipt.hygiene_failure,
        )
        assert receipt.endpoint_scope_id
        assert receipt.task_id
        assert receipt.wait_ms >= 0
        assert not receipt.overshoot
        assert not receipt.double_charge
        assert not receipt.hygiene_failure
        assert not receipt.authority_escape
        assert not receipt.completion_escape


def test_reservation_race_cannot_overshoot_hard_limit() -> None:
    receipts = run_chaos_population(observation_label="race")
    race = next(
        r
        for r in receipts
        if r.boundary is ChaosBoundary.CONCURRENT_RESERVATION_RACE
    )
    assert race.overshoot is False
    assert race.outcome in {FaultOutcome.RECOVERED, FaultOutcome.BACKPRESSURE}
    assert race.charged_requests <= 2


def test_cancel_timeout_before_and_after_dispatch() -> None:
    receipts = {
        r.boundary: r
        for r in run_chaos_population(observation_label="cancel")
    }
    before = receipts[ChaosBoundary.CANCEL_BEFORE_DISPATCH]
    after = receipts[ChaosBoundary.CANCEL_AFTER_DISPATCH]
    t_before = receipts[ChaosBoundary.TIMEOUT_BEFORE_DISPATCH]
    t_after = receipts[ChaosBoundary.TIMEOUT_AFTER_DISPATCH]
    assert before.charged_requests == 0
    assert t_before.charged_requests == 0
    # Post-dispatch conservatively retains charge.
    assert after.charged_requests >= 1
    assert t_after.charged_requests >= 1
    assert before.passed and after.passed and t_before.passed and t_after.passed


def test_distributed_partition_and_split_brain_fail_closed() -> None:
    receipts = {
        r.boundary: r
        for r in run_chaos_population(observation_label="partition")
    }
    for boundary in (
        ChaosBoundary.COORDINATOR_PARTITION,
        ChaosBoundary.SPLIT_BRAIN,
        ChaosBoundary.LEDGER_OUTAGE,
        ChaosBoundary.STALE_LEASE_FENCE,
    ):
        item = receipts[boundary]
        assert item.outcome is FaultOutcome.DENIED
        assert item.charged_requests == 0
        assert item.overshoot is False
        assert item.passed


def test_callsite_bypass_is_quarantined_without_authority_escape() -> None:
    receipts = run_chaos_population(observation_label="bypass")
    bypass = next(r for r in receipts if r.boundary is ChaosBoundary.CALLSITE_BYPASS)
    assert bypass.outcome is FaultOutcome.QUARANTINED
    assert bypass.authority_escape is False
    assert bypass.completion_escape is False
    assert bypass.charged_requests == 0


def test_fair_queue_and_reset_herd_are_bounded() -> None:
    receipts = {
        r.boundary: r for r in run_chaos_population(observation_label="fair")
    }
    fair = receipts[ChaosBoundary.UNFAIR_QUEUE_PRESSURE]
    herd = receipts[ChaosBoundary.RESET_HERD]
    assert fair.passed
    assert "starvation" not in " ".join(fair.reason_codes)
    assert herd.passed
    assert herd.wait_ms <= 5_000


def test_chaos_escape_fails_paired_report() -> None:
    report = build_paired_report(observation_label="escape-base")
    assert report.passed
    target = next(
        r
        for r in report.chaos_receipts
        if r.boundary is ChaosBoundary.REPLAY
    )
    escaped = replace(
        target,
        overshoot=True,
        authority_escape=True,
        reason_codes=target.reason_codes + ("injected-escape",),
    )
    mutated = SupervisorUsagePairedReport(
        observation_label=report.observation_label,
        e2e_receipts=report.e2e_receipts,
        chaos_receipts=tuple(
            escaped if r.receipt_id == target.receipt_id else r
            for r in report.chaos_receipts
        ),
        observed_at=report.observed_at,
        tree_id=report.tree_id,
        max_cost_micros=report.max_cost_micros,
        max_latency_ms=report.max_latency_ms,
        min_quality_bps=report.min_quality_bps,
        max_wait_ms=report.max_wait_ms,
    )
    assert not mutated.passed
    codes = mutated.failure_codes()
    assert "hard_limit_overshoot" in codes
    assert "authority_escape" in codes
