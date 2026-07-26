from __future__ import annotations

import time
from collections.abc import Sequence

from ipfs_accelerate_py.agent_supervisor.provider_batch_scheduler import (
    ProviderBatchRequest,
    ProviderBatchScheduler,
    ProviderBatchSchedulerConfig,
)
from ipfs_accelerate_py.agent_supervisor.resource_scheduler import (
    ADAPTIVE_SCHEDULING_THROUGHPUT_REQUIREMENT_ID,
    ADAPTIVE_STAGE_PROFILES,
    AdaptiveThroughputRun,
    HostResourceSnapshot,
    LaneResourceRequirements,
    ProviderCapacity,
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
        "memory_total_bytes": 16_000,
        "memory_available_bytes": 12_000,
        "disk_total_bytes": 32_000,
        "disk_available_bytes": 24_000,
        "gpu_memory_percent": 20,
        "gpu_memory_total_bytes": 8_000,
        "gpu_memory_available_bytes": 6_000,
        "active_workers": 0,
        "worker_limit": 6,
        "available_worker_capacity": 6,
        "capabilities": ("cpu",),
        "resource_classes": ("cpu-small",),
    }
    values.update(overrides)
    return HostResourceSnapshot(**values)  # type: ignore[arg-type]


def _provider(**overrides: object) -> ProviderCapacity:
    values: dict[str, object] = {
        "provider_id": "shared",
        "healthy": True,
        "quota_remaining": 100,
        "latency_ms": 10,
        "context_window_tokens": 32_000,
        "token_budget_remaining": 50_000,
        "max_concurrency": 6,
        "active_requests": 0,
        "capabilities": (),
        "observed_at_ms": 1_000,
    }
    values.update(overrides)
    return ProviderCapacity(**values)  # type: ignore[arg-type]


def _policy(**overrides: object) -> ResourcePolicy:
    values: dict[str, object] = {
        "max_lanes": 6,
        "adaptive_enabled": True,
        "adaptive_target_utilization_percent": 60,
        "adaptive_hysteresis_percent": 10,
        "adaptive_recovery_samples": 2,
        "adaptive_queue_depth_per_slot": 1,
        "adaptive_starvation_age_ms": 100,
        "adaptive_max_pending_tasks": 16,
        "adaptive_max_merge_debt": 4,
        "adaptive_artifact_pressure_high_watermark_percent": 80,
        "stage_concurrency_limits": {
            stage: 1
            for stage in (
                "analysis",
                "inference",
                "proof",
                "validation",
                "merge",
                "persistence",
            )
        },
    }
    values.update(overrides)
    return ResourcePolicy(**values)  # type: ignore[arg-type]


def test_six_stage_pools_have_independent_admission_ceilings() -> None:
    scheduler = ResourceScheduler(_policy())
    stages = tuple(profile.stage for profile in ADAPTIVE_STAGE_PROFILES)
    lanes = [
        LaneResourceRequirements(
            lane_id=f"{stage}-{index}",
            stage=stage,
            resource_class=(
                "llm-proof-draft"
                if stage == "inference"
                else "io-artifact"
                if stage == "persistence"
                else "cpu-small"
            ),
            requires_provider=stage == "inference",
            fairness_key=stage,
        )
        for stage in stages
        for index in range(2)
    ]

    snapshot = scheduler.schedule(
        lanes,
        host=_host(),
        providers=(_provider(),),
    )

    assert len({profile.pool for profile in ADAPTIVE_STAGE_PROFILES}) == 6
    by_stage = {item.stage: item for item in snapshot.stage_capacities}
    assert set(by_stage) == set(stages)
    assert all(item.configured_limit == 1 for item in by_stage.values())
    assert all(item.effective_limit == 1 for item in by_stage.values())
    assert {
        decision.stage
        for decision in snapshot.decisions
        if decision.admitted
    } == set(stages)
    assert all(
        sum(
            decision.admitted and decision.stage == stage
            for decision in snapshot.decisions
        )
        == 1
        for stage in stages
    )


def test_work_stealing_prefers_critical_paths_but_starvation_is_bounded() -> None:
    scheduler = ResourceScheduler(_policy())
    critical_home = LaneResourceRequirements(
        lane_id="critical-analysis",
        stage="analysis",
        critical_path_length=100,
        queue_age_ms=1,
    )
    starved_foreign = LaneResourceRequirements(
        lane_id="starved-proof",
        stage="proof",
        critical_path_length=1,
        queue_age_ms=100,
    )

    decision = scheduler.select_stealable_work(
        (critical_home, starved_foreign),
        worker_stage="analysis",
    )
    assert decision.selected_lane_id == "starved-proof"
    assert decision.stolen
    assert decision.starvation_override
    assert decision.considered_lane_ids[0] == "starved-proof"

    critical_order = scheduler.fair_work_order(
        (
            LaneResourceRequirements(
                lane_id="short", stage="validation", critical_path_length=1
            ),
            LaneResourceRequirements(
                lane_id="long", stage="validation", critical_path_length=10
            ),
        )
    )
    assert [item.lane_id for item in critical_order] == ["long", "short"]


def test_artifact_pressure_and_merge_debt_stop_upstream_but_preserve_drain() -> None:
    scheduler = ResourceScheduler(
        _policy(
            max_lanes=4,
            stage_concurrency_limits={
                "analysis": 4,
                "proof": 4,
                "merge": 2,
                "persistence": 2,
            },
        )
    )
    snapshot = scheduler.schedule(
        (
            LaneResourceRequirements(lane_id="analysis", stage="analysis"),
            LaneResourceRequirements(lane_id="proof", stage="proof"),
            LaneResourceRequirements(lane_id="merge", stage="merge"),
            LaneResourceRequirements(
                lane_id="persistence",
                stage="persistence",
                resource_class="io-artifact",
            ),
        ),
        host=_host(worker_limit=4, available_worker_capacity=4),
        signals={
            "artifact_pressure_percent": 80,
            "merge_debt": 4,
            "pending_tasks": 16,
        },
    )

    assert not snapshot.decision_for("analysis").admitted
    assert not snapshot.decision_for("proof").admitted
    assert snapshot.decision_for("merge").admitted
    assert snapshot.decision_for("persistence").admitted
    assert snapshot.task_generation is not None
    assert not snapshot.task_generation.admitted
    assert set(snapshot.task_generation.reasons) == {
        "pending_task_capacity",
        "merge_debt",
        "artifact_pressure",
    }
    assert snapshot.signals["task_generation_admitted"] is False


def test_task_generation_reopens_only_after_fresh_low_watermark_samples() -> None:
    scheduler = ResourceScheduler(_policy())
    blocked = scheduler.task_generation_backpressure(
        host=_host(observed_at_ms=1_000),
        pending_tasks=16,
        artifact_pressure_percent=80,
        merge_debt=4,
    )
    recovering = scheduler.task_generation_backpressure(
        host=_host(observed_at_ms=2_000),
        pending_tasks=2,
        artifact_pressure_percent=20,
        merge_debt=0,
    )
    duplicate_sample = scheduler.task_generation_backpressure(
        host=_host(observed_at_ms=2_000),
        pending_tasks=2,
        artifact_pressure_percent=20,
        merge_debt=0,
    )
    recovered = scheduler.task_generation_backpressure(
        host=_host(observed_at_ms=3_000),
        pending_tasks=2,
        artifact_pressure_percent=20,
        merge_debt=0,
    )

    assert not blocked.admitted
    assert recovering.hysteresis_state == "recovering"
    assert recovering.recovery_samples == 1
    assert duplicate_sample.recovery_samples == 1
    assert recovered.admitted
    assert recovered.hysteresis_state == "recovered"


def test_shared_model_and_prover_batching_preserves_member_accounting() -> None:
    calls: list[tuple[str, tuple[int, ...]]] = []

    def dispatch(
        requests: Sequence[ProviderBatchRequest],
    ) -> list[dict[str, object]]:
        calls.append(
            (
                requests[0].route,
                tuple(request.token_budget for request in requests),
            )
        )
        return [
            {"request_id": request.request_id, "route": request.route}
            for request in requests
        ]

    config = ProviderBatchSchedulerConfig(
        max_batch_size=8,
        batch_window_ms=20,
        max_parallel_batches=2,
        provider_limits={"shared": 2},
        admission_retry_ms=1,
    )
    with ProviderBatchScheduler(dispatch, config=config) as scheduler:
        futures = [
            scheduler.submit(
                ProviderBatchRequest(
                    request_id=f"{route}-{index}",
                    payload={"member": index},
                    provider_id="shared",
                    route=route,
                    model="shared-service",
                    operation="generate" if route == "model" else "prove",
                    token_budget=100 + index,
                    timeout_ms=2_000,
                    provenance={"stage": route, "member": index},
                )
            )
            for route in ("model", "prover")
            for index in range(2)
        ]
        results = tuple(future.result(timeout=2) for future in futures)
        metrics = scheduler.metrics()
        receipts = scheduler.evidence_receipts()

    assert sorted(calls) == [
        ("model", (100, 101)),
        ("prover", (100, 101)),
    ]
    assert {result.request_id for result in results} == {
        "model-0",
        "model-1",
        "prover-0",
        "prover-1",
    }
    assert all(result.receipt_id for result in results)
    assert {member.token_budget for receipt in receipts for member in receipt.members} == {
        100,
        101,
    }
    assert metrics.provider_calls == 2
    assert metrics.physical_executions == 4
    assert metrics.duplicate_executions == 0
    assert metrics.duplicate_compute_percent_millionths == 0
    assert metrics.peak_active_batches <= config.max_parallel_batches


def test_three_x_gate_rejects_the_old_two_x_threshold() -> None:
    fixtures = ("a", "b", "c")
    baseline = AdaptiveThroughputRun(
        fixture_ids=fixtures,
        executed_fixture_ids=fixtures,
        accepted_fixture_ids=fixtures,
        duration_ms=300,
        peak_concurrency=1,
    )
    only_two_x = AdaptiveThroughputRun(
        fixture_ids=fixtures,
        executed_fixture_ids=fixtures,
        accepted_fixture_ids=fixtures,
        duration_ms=150,
        peak_concurrency=3,
    )
    policy = _policy(max_lanes=3)

    failed = evaluate_adaptive_throughput_benchmark(
        baseline,
        only_two_x,
        policy=policy,
        repository_tree_id="tree:asi-112",
    )
    passed = evaluate_adaptive_throughput_benchmark(
        baseline,
        AdaptiveThroughputRun(
            fixture_ids=fixtures,
            executed_fixture_ids=fixtures,
            accepted_fixture_ids=fixtures,
            duration_ms=100,
            peak_concurrency=3,
        ),
        policy=policy,
        repository_tree_id="tree:asi-112",
    )

    assert not failed.passed
    assert "throughput_below_three_x" in failed.failure_codes
    assert passed.passed
    assert passed.adaptive.duplicate_compute_percent_millionths == 0
    assert passed.proved_requirement_ids_for(
        policy=policy,
        repository_tree_id="tree:asi-112",
    ) == (ADAPTIVE_SCHEDULING_THROUGHPUT_REQUIREMENT_ID,)


def test_measured_adaptive_run_reaches_three_x_with_stable_resources() -> None:
    policy = _policy(max_lanes=4)

    def fixture() -> bool:
        time.sleep(0.02)
        return True

    receipt = benchmark_adaptive_execution(
        {f"fixture-{index}": fixture for index in range(8)},
        policy=policy,
        repository_tree_id="tree:asi-112-measured",
    )

    assert receipt.passed, receipt.failure_codes
    assert receipt.adaptive.peak_concurrency <= policy.max_lanes
    assert receipt.adaptive.duplicate_compute_percent_millionths < 5_000_000
    assert (
        receipt.adaptive.accepted_count * receipt.baseline.duration_ms
        >= 3
        * receipt.baseline.accepted_count
        * receipt.adaptive.duration_ms
    )
