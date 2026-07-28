from __future__ import annotations

import time
from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    ADAPTIVE_SCHEDULING_THROUGHPUT_REQUIREMENT_ID,
    ADAPTIVE_STAGE_PROFILES,
    AdaptiveThroughputBenchmarkReceipt,
    AdaptiveThroughputRun,
    HostResourceSnapshot,
    LaneResourceRequirements,
    ProviderCapacity,
    ResourcePolicy,
    ResourceScheduler,
    adaptive_stage_profile,
    benchmark_adaptive_execution,
    evaluate_adaptive_throughput_benchmark,
    normalize_adaptive_stage,
)
from ipfs_accelerate_py.agent_supervisor.runtime.scheduler_metrics import (
    RESOURCE_ADMISSION_METRICS_SCHEMA,
    build_scheduler_snapshot,
    project_resource_admission_metrics,
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
        "gpu_memory_percent": 0,
        "gpu_memory_total_bytes": 0,
        "gpu_memory_available_bytes": 0,
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


def _provider(**overrides: object) -> ProviderCapacity:
    values: dict[str, object] = {
        "provider_id": "provider-a",
        "healthy": True,
        "quota_remaining": 100,
        "latency_ms": 10,
        "context_window_tokens": 32_000,
        "token_budget_remaining": 50_000,
        "max_concurrency": 4,
        "active_requests": 0,
        "capabilities": ("json",),
        "observed_at_ms": 1_000,
    }
    values.update(overrides)
    return ProviderCapacity(**values)  # type: ignore[arg-type]


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


def test_resource_pools_expose_fair_order_and_backpressure() -> None:
    scheduler = ResourceScheduler(_policy(max_lanes=2))
    schedule = scheduler.schedule(
        [
            LaneResourceRequirements(
                lane_id="analysis-a",
                stage="analysis",
                fairness_key="analysis",
            ),
            LaneResourceRequirements(
                lane_id="analysis-b",
                stage="analysis",
                fairness_key="analysis",
            ),
            LaneResourceRequirements(
                lane_id="validation-a",
                stage="validation",
                fairness_key="validation",
            ),
            LaneResourceRequirements(
                lane_id="validation-b",
                stage="validation",
                fairness_key="validation",
            ),
        ],
        host=_host(worker_limit=2, available_worker_capacity=2),
    )

    # Both stages use the same physical cpu-proof pool. Its durable projection
    # demonstrates round-robin evaluation and explains the rejected remainder.
    assert len(schedule.pool_admissions) == 1
    pool = schedule.pool_admissions[0]
    assert pool.resource_pool == "cpu-proof"
    assert pool.fairness_order == (
        "analysis-a",
        "validation-a",
        "analysis-b",
        "validation-b",
    )
    assert pool.fairness_keys == (
        "analysis",
        "validation",
        "analysis",
        "validation",
    )
    assert pool.admitted_lane_ids == ("analysis-a", "validation-a")
    assert pool.scheduled_count == 4
    assert pool.admitted_count == 2
    assert pool.backpressured_count == 2
    assert pool.backpressure_counts == {
        "cpu_proof_concurrency": 2,
        "host_worker_capacity": 2,
    }
    assert [decision.admission_rank for decision in schedule.decisions] == [
        1,
        2,
        3,
        4,
    ]
    assert [decision.fairness_key for decision in schedule.decisions] == [
        "analysis",
        "validation",
        "analysis",
        "validation",
    ]
    assert schedule.to_dict()["pool_admissions"] == [pool.to_dict()]


def test_six_stage_profiles_and_git_aliases_are_explicit_and_deterministic() -> None:
    assert [profile.stage for profile in ADAPTIVE_STAGE_PROFILES] == [
        "analysis",
        "inference",
        "proof",
        "validation",
        "merge",
        "persistence",
    ]
    assert len({profile.pool for profile in ADAPTIVE_STAGE_PROFILES}) == 6
    assert adaptive_stage_profile("llm").requires_provider is True
    assert adaptive_stage_profile("llm").gpu_memory_sensitive is True
    assert adaptive_stage_profile("git_merge").disk_sensitive is True
    assert adaptive_stage_profile("artifact").stage == "persistence"
    assert normalize_adaptive_stage("git") == "merge"
    assert normalize_adaptive_stage("git_merge") == "merge"

    # Profile and lane artifacts remain integer-only, stable mappings.
    inference = adaptive_stage_profile("provider").to_dict()
    assert inference == {
        "stage": "inference",
        "pool": "inference",
        "resource_class": "llm-proof-draft",
        "requires_provider": True,
        "cpu_sensitive": True,
        "memory_sensitive": True,
        "gpu_memory_sensitive": True,
        "disk_sensitive": False,
    }


def test_gpu_provider_and_disk_pressure_only_block_relevant_stages() -> None:
    scheduler = ResourceScheduler(_policy(adaptive_recovery_samples=1))
    pressured = _host(
        disk_percent=99,
        disk_available_bytes=100,
        gpu_memory_percent=99,
        gpu_memory_total_bytes=1_000,
        gpu_memory_available_bytes=10,
    )
    unavailable_provider = _provider(max_concurrency=0)

    analysis = scheduler.evaluate(
        LaneResourceRequirements(lane_id="analysis", stage="analysis"),
        host=pressured,
        providers=(unavailable_provider,),
    )
    validation = scheduler.evaluate(
        LaneResourceRequirements(lane_id="validation", stage="validation"),
        host=pressured,
        providers=(unavailable_provider,),
    )
    inference = scheduler.evaluate(
        LaneResourceRequirements(
            lane_id="inference",
            stage="inference",
            requires_provider=True,
            gpu_memory_bytes=100,
        ),
        host=pressured,
        providers=(unavailable_provider,),
    )
    merge = scheduler.evaluate(
        LaneResourceRequirements(
            lane_id="merge",
            stage="git",
            disk_bytes=100,
        ),
        host=pressured,
        providers=(unavailable_provider,),
    )

    assert analysis.admitted and validation.admitted
    assert not inference.admitted
    assert {
        "host_gpu_memory_high_watermark",
        "host_gpu_memory_headroom",
        "provider_concurrency",
    }.intersection(inference.reasons)
    assert not merge.admitted
    assert "host_disk_high_watermark" in merge.reasons
    assert not set(analysis.reasons).intersection(
        {"host_disk_high_watermark", "host_gpu_memory_high_watermark"}
    )


def test_queue_depth_merge_age_and_active_leases_are_capacity_signals() -> None:
    policy = _policy(
        adaptive_queue_depth_per_slot=2,
        adaptive_merge_age_ms=500,
    )
    shallow = ResourceScheduler(policy).adaptive_stage_capacity(
        "analysis",
        host=_host(observed_at_ms=1_000),
        queued=1,
    )
    deep = ResourceScheduler(policy).adaptive_stage_capacity(
        "analysis",
        host=_host(observed_at_ms=1_000),
        queued=8,
    )
    overdue_merge = ResourceScheduler(policy).adaptive_stage_capacity(
        "git",
        host=_host(observed_at_ms=1_000),
        active=1,
        queued=6,
        merge_age_ms=500,
        active_leases=1,
    )

    assert shallow.effective_limit == 1
    assert deep.effective_limit == 4
    assert overdue_merge.stage == "merge"
    assert overdue_merge.reason == "merge_age_priority"
    assert overdue_merge.queue_depth == 6
    assert overdue_merge.merge_age_ms == 500
    assert overdue_merge.active_leases == 1
    assert overdue_merge.available == overdue_merge.effective_limit - 1

    scheduler = ResourceScheduler(_policy(max_lanes=2))
    first_decision, first_lease = scheduler.acquire(
        LaneResourceRequirements(lane_id="active", stage="analysis"),
        host=_host(worker_limit=2, available_worker_capacity=2),
    )
    assert first_decision.admitted and first_lease is not None
    snapshot = scheduler.schedule(
        [
            LaneResourceRequirements(lane_id="queued-a", stage="analysis"),
            LaneResourceRequirements(lane_id="queued-b", stage="analysis"),
        ],
        host=_host(
            active_workers=1,
            worker_limit=2,
            available_worker_capacity=1,
        ),
        signals={"queue_depth": {"analysis": 2}},
    )
    assert snapshot.active_lease_count == 1
    assert snapshot.signals["active_lease_count"] == 1
    assert snapshot.signals["queue_depth_by_stage"] == {"analysis": 2}
    # The existing lease is always accounted. Hysteresis may conservatively
    # retain its one-slot limit until a later fresh recovery sample.
    assert snapshot.admitted_count <= 1
    assert all(decision.active_leases >= 1 for decision in snapshot.decisions)


def test_hysteresis_and_resource_loss_recovery_are_sample_deterministic() -> None:
    scheduler = ResourceScheduler(
        _policy(
            adaptive_hysteresis_percent=10,
            adaptive_recovery_samples=2,
        )
    )

    baseline = scheduler.adaptive_stage_capacity(
        "analysis",
        host=_host(observed_at_ms=1_000, cpu_percent=20),
        queued=8,
    )
    contracted = scheduler.adaptive_stage_capacity(
        "analysis",
        host=_host(observed_at_ms=2_000, cpu_percent=80),
        queued=8,
    )
    deadband = scheduler.adaptive_stage_capacity(
        "analysis",
        host=_host(observed_at_ms=3_000, cpu_percent=55),
        queued=8,
    )
    first_recovery = scheduler.adaptive_stage_capacity(
        "analysis",
        host=_host(observed_at_ms=4_000, cpu_percent=40),
        queued=8,
    )
    duplicate_sample = scheduler.adaptive_stage_capacity(
        "analysis",
        host=_host(observed_at_ms=4_000, cpu_percent=40),
        queued=8,
    )
    recovered = scheduler.adaptive_stage_capacity(
        "analysis",
        host=_host(observed_at_ms=5_000, cpu_percent=40),
        queued=8,
    )

    assert baseline.effective_limit == 4
    assert contracted.effective_limit == 2
    assert contracted.hysteresis_state == "contracted"
    assert deadband.effective_limit == contracted.effective_limit
    assert first_recovery.recovery_samples == 1
    assert duplicate_sample.recovery_samples == 1
    assert duplicate_sample.effective_limit == contracted.effective_limit
    assert recovered.effective_limit == baseline.effective_limit
    assert recovered.hysteresis_state == "recovered"
    metrics = scheduler.metrics_snapshot(observed_at_ms=5_000).by_stage["analysis"]
    assert metrics.contraction_events == 1
    assert metrics.recovery_events == 1


def test_critical_path_wins_but_aged_merge_work_cannot_starve() -> None:
    policy = _policy(
        max_lanes=1,
        adaptive_starvation_age_ms=100,
        adaptive_merge_age_ms=100,
        stage_concurrency_limits={"analysis": 1, "merge": 1},
    )
    host = _host(worker_limit=1, available_worker_capacity=1)

    critical_schedule = ResourceScheduler(policy).schedule(
        [
            LaneResourceRequirements(
                lane_id="normal",
                stage="analysis",
                fairness_key="analysis",
                critical_path_length=1,
            ),
            LaneResourceRequirements(
                lane_id="critical",
                stage="analysis",
                fairness_key="analysis",
                critical_path_length=5,
            ),
        ],
        host=host,
    )
    assert critical_schedule.admitted_lane_ids == ("critical",)

    starvation_schedule = ResourceScheduler(policy).schedule(
        [
            LaneResourceRequirements(
                lane_id="critical",
                stage="analysis",
                critical_path_length=50,
            ),
            LaneResourceRequirements(
                lane_id="old-merge",
                stage="merge",
                queue_age_ms=100,
                merge_age_ms=100,
            ),
        ],
        host=host,
    )
    assert starvation_schedule.admitted_lane_ids == ("old-merge",)
    assert starvation_schedule.decision_for("critical") is not None
    assert starvation_schedule.decision_for("critical").admitted is False


def test_cancellation_releases_lease_and_capacity_for_the_next_lane() -> None:
    scheduler = ResourceScheduler(_policy(max_lanes=1))
    host = _host(worker_limit=1, available_worker_capacity=1)
    admitted, lease = scheduler.acquire(
        LaneResourceRequirements(lane_id="first", stage="analysis"),
        host=host,
    )
    assert admitted.admitted and lease is not None

    blocked, blocked_lease = scheduler.acquire(
        LaneResourceRequirements(lane_id="second", stage="analysis"),
        host=host,
    )
    assert not blocked.admitted and blocked_lease is None
    assert scheduler.cancel(lease, reason="operator_cancelled")
    assert scheduler.active_leases == ()
    assert not scheduler.cancel(lease, reason="duplicate_cancel")

    retried, replacement = scheduler.acquire(
        LaneResourceRequirements(lane_id="second", stage="analysis"),
        host=host,
    )
    assert retried.admitted and replacement is not None
    metrics = scheduler.metrics_snapshot(observed_at_ms=2_000)
    assert metrics.active_lease_count == 1
    assert metrics.by_stage["analysis"].cancelled == 1
    assert metrics.by_stage["analysis"].leases_released == 1


def test_aggregate_gpu_memory_and_active_slots_never_overadmit() -> None:
    scheduler = ResourceScheduler(_policy(max_lanes=4))
    host = _host(
        worker_limit=4,
        available_worker_capacity=4,
        memory_available_bytes=800,
        disk_available_bytes=800,
        gpu_memory_percent=20,
        gpu_memory_total_bytes=1_000,
        gpu_memory_available_bytes=800,
    )
    requirements = [
        LaneResourceRequirements(
            lane_id=f"inference-{index}",
            stage="inference",
            requires_provider=True,
            memory_bytes=200,
            gpu_memory_bytes=400,
            disk_bytes=100,
        )
        for index in range(3)
    ]
    schedule = scheduler.schedule(
        requirements,
        host=host,
        providers=(_provider(max_concurrency=10),),
    )
    admitted = [
        requirement
        for requirement in requirements
        if schedule.decision_for(requirement.lane_id).admitted
    ]

    assert len(schedule.admitted_lane_ids) == len(set(schedule.admitted_lane_ids))
    assert sum(item.process_slots for item in admitted) <= host.available_worker_capacity
    assert sum(item.memory_bytes for item in admitted) <= host.memory_available_bytes
    assert (
        sum(item.gpu_memory_bytes for item in admitted)
        <= host.gpu_memory_available_bytes
    )
    assert sum(item.disk_bytes for item in admitted) <= host.disk_available_bytes
    assert any(not decision.admitted for decision in schedule.decisions)


def test_resource_backpressure_event_projection_is_latest_and_taskless() -> None:
    old = {
        "type": "resource_schedule_observed",
        "timestamp": "2026-01-01T00:00:00Z",
        "resource_schedule": {
            "observed_at_ms": 1_000,
            "configured_max_lanes": 4,
            "effective_slots": 4,
            "available_slots": 4,
            "admitted_count": 0,
            "stage_capacities": [
                {
                    "stage": "analysis",
                    "configured_limit": 4,
                    "effective_limit": 4,
                    "queued": 1,
                    "available": 4,
                }
            ],
        },
    }
    latest = {
        "type": "adaptive_resources_observed",
        "timestamp": "2026-01-01T00:00:01Z",
        "adaptive_resources": {
            "observed_at_ms": 2_000,
            "configured_max_lanes": 4,
            "effective_slots": 1,
            "available_slots": 0,
            "admitted_count": 1,
            "active_lease_count": 1,
            "signals": {
                "cpu_percent": 82,
                "queue_depth": 6,
                "merge_age_ms": 700,
                "active_lease_count": 1,
            },
            "decisions": [
                {"lane_id": "accepted", "stage": "analysis", "admitted": True},
                {
                    "lane_id": "deferred",
                    "stage": "git",
                    "admitted": False,
                    "reasons": ["host_disk_high_watermark"],
                },
            ],
            "stage_capacities": [
                {
                    "stage": "git",
                    "configured_limit": 2,
                    "effective_limit": 1,
                    "active": 1,
                    "queued": 6,
                    "available": 0,
                    "pressure_percent": 82,
                    "reason": "merge_age_priority",
                }
            ],
            "adaptive_metrics": {
                "stages": [
                    {
                        "stage": "git",
                        "scheduled": 2,
                        "admitted": 1,
                        "backpressured": 1,
                    }
                ]
            },
        },
    }

    projection = project_resource_admission_metrics((latest, old))
    snapshot = build_scheduler_snapshot((latest, old))

    assert projection is not None
    assert projection["schema"] == RESOURCE_ADMISSION_METRICS_SCHEMA
    assert projection["observed_at_ms"] == 2_000
    assert projection["effective_slots"] == 1
    assert projection["queue_depth"] == 6
    assert projection["merge_age_ms"] == 700
    assert projection["active_lease_count"] == 1
    assert projection["backpressure_reason_counts"] == {
        "host_disk_high_watermark": 1
    }
    assert projection["by_stage"]["merge"]["reason"] == "merge_age_priority"
    assert projection["by_stage"]["merge"]["backpressured"] == 1
    assert snapshot.resource_admission == projection
    assert snapshot.adaptive_resources == projection
    assert snapshot["task_states"] == []
    assert snapshot["metrics"] == []


def test_generic_stage_events_are_not_resource_admission_observations() -> None:
    event = {
        "type": "validation_completed",
        "timestamp": "2026-01-01T00:00:00Z",
        "task_cid": "task:validation",
        "stages": [
            {"stage": "unit", "passed": True},
            {"stage": "integration", "passed": True},
        ],
    }

    assert project_resource_admission_metrics((event,)) is None
    snapshot = build_scheduler_snapshot((event,))
    assert snapshot.resource_admission is None
    assert snapshot["task_states"][0]["task_cid"] == "task:validation"


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
