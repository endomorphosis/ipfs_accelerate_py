from __future__ import annotations

import time
from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.resource_scheduler import (
    ADAPTIVE_SCHEDULING_THROUGHPUT_REQUIREMENT_ID,
    AdaptiveThroughputBenchmarkReceipt,
    AdaptiveThroughputRun,
    HostResourceSnapshot,
    LaneResourceRequirements,
    ResourcePolicy,
    ResourceScheduler,
    benchmark_adaptive_execution,
    evaluate_adaptive_throughput_benchmark,
)


def _host(**overrides: object) -> HostResourceSnapshot:
    values: dict[str, object] = {
        "observed_at_ms": 1_000,
        "cpu_percent": 20,
        "memory_percent": 20,
        "disk_percent": 20,
        "memory_available_bytes": 1_000_000,
        "disk_available_bytes": 1_000_000,
        "active_workers": 0,
        "worker_limit": 4,
        "available_worker_capacity": 4,
        "capabilities": ("cpu",),
        "resource_classes": ("cpu-small",),
    }
    values.update(overrides)
    return HostResourceSnapshot(**values)  # type: ignore[arg-type]


def _policy(**overrides: object) -> ResourcePolicy:
    values: dict[str, object] = {
        "max_lanes": 4,
        "adaptive_enabled": True,
        "adaptive_target_utilization_percent": 60,
        "stage_concurrency_limits": {
            "analysis": 4,
            "validation": 2,
        },
        "stage_min_concurrency": {
            "analysis": 1,
            "validation": 1,
        },
    }
    values.update(overrides)
    return ResourcePolicy(**values)  # type: ignore[arg-type]


def test_adaptive_stage_capacity_contracts_before_the_hard_host_gate() -> None:
    scheduler = ResourceScheduler(_policy())

    headroom = scheduler.adaptive_stage_capacity(
        "analysis", host=_host(cpu_percent=40), queued=8
    )
    pressure = scheduler.adaptive_stage_capacity(
        "analysis", host=_host(cpu_percent=80), queued=8
    )

    assert headroom.effective_limit == 4
    assert headroom.reason == "live_headroom"
    assert pressure.effective_limit == 2
    assert pressure.reason == "live_pressure_backoff"
    assert pressure.to_dict()["pressure_percent"] == 80

    schedule = scheduler.schedule(
        [
            LaneResourceRequirements(lane_id=f"analysis-{index}", stage="analysis")
            for index in range(4)
        ],
        host=_host(cpu_percent=80),
    )
    assert schedule.admitted_lane_ids == ("analysis-0", "analysis-1")
    assert schedule.decisions[2].reason == "stage_concurrency"
    assert schedule.stage_capacities[0].effective_limit == 2
    assert schedule.stage_capacities[0].active == 2


def test_adaptive_admission_round_robins_stages_and_exports_lane_metrics() -> None:
    scheduler = ResourceScheduler(_policy(max_lanes=2))
    lanes = [
        LaneResourceRequirements(lane_id="analysis-1", stage="analysis"),
        LaneResourceRequirements(lane_id="analysis-2", stage="analysis"),
        LaneResourceRequirements(lane_id="analysis-3", stage="analysis"),
        LaneResourceRequirements(lane_id="validation-1", stage="validation"),
    ]

    schedule = scheduler.schedule(
        lanes,
        host=_host(worker_limit=2, available_worker_capacity=2),
    )

    # A validation candidate cannot starve behind a deep analysis queue.
    assert schedule.admitted_lane_ids == ("analysis-1", "validation-1")
    assert [item.stage for item in schedule.decisions[:2]] == [
        "analysis",
        "validation",
    ]
    assert schedule.adaptive_metrics is not None
    by_stage = schedule.adaptive_metrics.by_stage
    assert by_stage["analysis"].scheduled == 3
    assert by_stage["analysis"].admitted == 1
    assert by_stage["analysis"].backpressured == 2
    assert by_stage["validation"].admitted == 1

    metric = scheduler.record_stage_completion(
        "validation", duration_ms=25, accepted=True
    )
    scheduler.record_stage_completion(
        "analysis", duration_ms=50, accepted=False, cancelled=True
    )
    assert metric.acceptance_throughput_per_million_ms == 40_000
    snapshot = scheduler.metrics_snapshot(observed_at_ms=2_000)
    assert snapshot.by_stage["analysis"].cancelled == 1
    assert snapshot.by_stage["validation"].accepted == 1
    assert snapshot.to_dict()["observed_at_ms"] == 2_000


def test_content_addressed_benchmark_receipt_is_fail_closed_and_rebinds() -> None:
    policy = _policy()
    fixtures = ("fixture-a", "fixture-b", "fixture-c", "fixture-d")
    baseline = AdaptiveThroughputRun(
        fixture_ids=fixtures,
        executed_fixture_ids=fixtures,
        accepted_fixture_ids=fixtures,
        duration_ms=400,
        peak_concurrency=1,
    )
    adaptive = AdaptiveThroughputRun(
        fixture_ids=fixtures,
        executed_fixture_ids=fixtures,
        accepted_fixture_ids=fixtures,
        duration_ms=100,
        peak_concurrency=4,
    )

    receipt = evaluate_adaptive_throughput_benchmark(
        baseline,
        adaptive,
        policy=policy,
        repository_tree_id="tree:current",
    )

    assert receipt.passed
    assert receipt.failure_codes == ()
    assert receipt.content_id.startswith("sha256:")
    assert receipt.proved_requirement_ids_for(
        policy=policy, repository_tree_id="tree:current"
    ) == (ADAPTIVE_SCHEDULING_THROUGHPUT_REQUIREMENT_ID,)
    assert AdaptiveThroughputBenchmarkReceipt.from_mapping(
        receipt.to_dict()
    ).proved_requirement_ids_for(
        policy=policy, repository_tree_id="tree:current"
    ) == (ADAPTIVE_SCHEDULING_THROUGHPUT_REQUIREMENT_ID,)

    assert receipt.proved_requirement_ids_for(
        policy=replace(policy, max_lanes=3),
        repository_tree_id="tree:current",
    ) == ()
    assert receipt.proved_requirement_ids_for(
        policy=policy, repository_tree_id="tree:stale"
    ) == ()
    forged = replace(receipt, adaptive=replace(adaptive, duration_ms=300))
    assert forged.proved_requirement_ids_for(
        policy=policy, repository_tree_id="tree:current"
    ) == ()


def test_benchmark_runner_proves_two_x_without_duplicates_or_overcommit() -> None:
    policy = _policy()

    def independent_fixture() -> bool:
        time.sleep(0.03)
        return True

    receipt = benchmark_adaptive_execution(
        {
            f"independent-{index}": independent_fixture
            for index in range(4)
        },
        policy=policy,
        repository_tree_id="tree:benchmark",
    )

    assert receipt.passed, receipt.failure_codes
    assert receipt.adaptive.peak_concurrency == 4
    assert len(set(receipt.adaptive.executed_fixture_ids)) == 4
    assert (
        receipt.adaptive.accepted_count * receipt.baseline.duration_ms
        >= 2
        * receipt.baseline.accepted_count
        * receipt.adaptive.duration_ms
    )
    assert receipt.proved_requirement_ids_for(
        policy=policy, repository_tree_id="tree:benchmark"
    ) == (ADAPTIVE_SCHEDULING_THROUGHPUT_REQUIREMENT_ID,)


def test_benchmark_rejects_duplicates_incomplete_acceptance_and_overcommit() -> None:
    policy = _policy(max_lanes=2)
    baseline = AdaptiveThroughputRun(
        fixture_ids=("a", "b"),
        executed_fixture_ids=("a", "b"),
        accepted_fixture_ids=("a", "b"),
        duration_ms=20,
        peak_concurrency=1,
    )
    invalid = AdaptiveThroughputRun(
        fixture_ids=("a", "b"),
        executed_fixture_ids=("a", "a"),
        accepted_fixture_ids=("a",),
        duration_ms=5,
        peak_concurrency=3,
    )

    receipt = evaluate_adaptive_throughput_benchmark(
        baseline,
        invalid,
        policy=policy,
        repository_tree_id="tree:current",
    )

    assert not receipt.passed
    assert {
        "adaptive_duplicate_execution",
        "adaptive_execution_incomplete",
        "adaptive_acceptance_incomplete",
        "adaptive_resource_overcommit",
    }.issubset(receipt.failure_codes)
    assert receipt.proved_requirement_ids_for(
        policy=policy, repository_tree_id="tree:current"
    ) == ()
