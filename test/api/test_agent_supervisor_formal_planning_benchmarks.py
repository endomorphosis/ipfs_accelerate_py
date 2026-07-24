from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_planning_metrics import (
    FORMAL_PLANNING_METRIC_DIMENSIONS,
    BenchmarkMode,
    FormalPlanningBenchmarkReport,
    FormalPlanningBenchmarkSample,
    FormalPlanningMetricDimensions,
    FormalPlanningMetricsCollector,
    FormalPlanningMetricsError,
    build_formal_planning_benchmark_report,
)
from ipfs_accelerate_py.agent_supervisor.formal_planning_rollout import (
    DEFAULT_ROLLOUT_THRESHOLDS,
    FormalPlanningOverrideStore,
    FormalPlanningRolloutError,
    FormalPlanningRolloutGate,
    FormalPlanningRolloutOverride,
    RolloutDisposition,
    build_formal_planning_operator_projection,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_policy import (
    RiskLevel,
    RolloutMode,
)


NOW = datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc)


def _dimensions(
    *,
    property_class: str = "state_machine",
    translator_profile: str = "tla-exact-v1",
    prover: str = "apalache",
    kernel: str = "lean",
    bound: int | None = 32,
    rollout_mode: RolloutMode = RolloutMode.ENFORCEMENT,
    risk: RiskLevel = RiskLevel.HIGH,
    assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED,
) -> FormalPlanningMetricDimensions:
    return FormalPlanningMetricDimensions(
        property_class=property_class,
        translator_profile=translator_profile,
        prover=prover,
        kernel=kernel,
        finite_bound=bound,
        rollout_mode=rollout_mode,
        task_risk=risk,
        authoritative_assurance=assurance,
    )


def _sample(
    mode: BenchmarkMode,
    *,
    sample_id: str | None = None,
    dimensions: FormalPlanningMetricDimensions | None = None,
    available: bool = True,
    low_value: bool = False,
    context_tokens: int = 450,
    defects: int = 2,
    supported: int = 9,
    cache_hits: int | None = None,
    queue_latency_ms: float = 100,
    cpu: float = 60,
    memory: int = 512 * 1024**2,
    accepted: int = 10,
    elapsed: float = 10,
    baseline_throughput: float = 1,
) -> FormalPlanningBenchmarkSample:
    if cache_hits is None:
        cache_hits = 8 if mode is BenchmarkMode.WARM else 1
    return FormalPlanningBenchmarkSample(
        sample_id=sample_id or f"{mode.value}-sample",
        benchmark_mode=mode,
        dimensions=dimensions or _dimensions(),
        baseline_context_tokens=1_000,
        formal_context_tokens=context_tokens,
        plans_evaluated=10,
        defects_found_before_dispatch=defects,
        obligations=10,
        proof_supported_obligations=supported,
        counterexamples=2,
        actionable_counterexamples=2,
        minimized_counterexamples=2,
        cache_lookups=10,
        cache_hits=cache_hits,
        queue_latency_ms=queue_latency_ms,
        cpu_saturation_percent=cpu,
        memory_peak_bytes=memory,
        accepted_tasks=accepted,
        elapsed_seconds=elapsed,
        baseline_accepted_tasks_per_second=baseline_throughput,
        available=available,
        low_value=low_value,
        advisory_reason_codes=(
            ("provider_unavailable",) if not available else ()
        ),
    )


def _report(
    *,
    dimensions: FormalPlanningMetricDimensions | None = None,
    **changes,
) -> FormalPlanningBenchmarkReport:
    dimensions = dimensions or _dimensions()
    return build_formal_planning_benchmark_report(
        (
            _sample(
                BenchmarkMode.COLD,
                sample_id=f"{dimensions.lane_id}-cold",
                dimensions=dimensions,
                **changes,
            ),
            _sample(
                BenchmarkMode.WARM,
                sample_id=f"{dimensions.lane_id}-warm",
                dimensions=dimensions,
                **changes,
            ),
        ),
        generated_at=NOW,
    )


def test_cold_and_warm_report_measures_every_acceptance_metric() -> None:
    report = _report()

    assert report["benchmark_modes"] == ["cold", "warm"]
    assert report["metric_dimensions"] == list(FORMAL_PLANNING_METRIC_DIMENSIONS)
    assert report["summary"]["context_token_reduction"] == pytest.approx(0.55)
    assert report["summary"]["defects_found_before_dispatch"] == 4
    lane = report.matrix[0]
    cold = lane["benchmark_modes"]["cold"]
    warm = lane["benchmark_modes"]["warm"]
    assert cold["proof_support_rate"] == 0.9
    assert cold["counterexample_quality"] == 1.0
    assert warm["cache_reuse_rate"] == 0.8
    assert cold["queue_latency_ms_max"] == 100
    assert cold["cpu_saturation_percent_max"] == 60
    assert cold["memory_peak_bytes"] == 512 * 1024**2
    assert cold["accepted_tasks_per_second"] == 1
    assert cold["throughput_ratio"] == 1
    assert lane["cold_to_warm"]["cache_reuse_improvement"] == 0.7


def test_matrix_never_combines_distinct_assurance_or_translator_lanes() -> None:
    first = _dimensions()
    second = _dimensions(
        translator_profile="smt-conservative-v2",
        prover="z3",
        kernel="none",
        assurance=AssuranceLevel.SOLVER_CHECKED,
    )
    report = build_formal_planning_benchmark_report(
        (
            _sample(BenchmarkMode.COLD, sample_id="a-cold", dimensions=first),
            _sample(BenchmarkMode.WARM, sample_id="a-warm", dimensions=first),
            _sample(BenchmarkMode.COLD, sample_id="b-cold", dimensions=second),
            _sample(BenchmarkMode.WARM, sample_id="b-warm", dimensions=second),
        ),
        generated_at=NOW,
    )

    assert report["lane_count"] == 2
    keys = {
        (
            lane["dimensions"]["translator_profile"],
            lane["dimensions"]["authoritative_assurance"],
        )
        for lane in report.matrix
    }
    assert keys == {
        ("tla-exact-v1", "kernel_verified"),
        ("smt-conservative-v2", "solver_checked"),
    }


def test_enforcement_promotes_only_when_assurance_and_throughput_pass() -> None:
    decision = FormalPlanningRolloutGate().evaluate(
        _report(), target_mode=RolloutMode.ENFORCEMENT, now=NOW
    )

    assert decision.promotion_allowed
    assert decision["lane_decisions"][0]["disposition"] == "promote"
    assert decision["thresholds"] == DEFAULT_ROLLOUT_THRESHOLDS[
        RolloutMode.ENFORCEMENT
    ].to_dict()
    assert set(decision["all_thresholds"]) == {"shadow", "canary", "enforcement"}


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"cpu": 95}, "cpu_saturation_above_threshold"),
        (
            {"accepted": 5, "elapsed": 10, "baseline_throughput": 1},
            "accepted_task_throughput_below_threshold",
        ),
        ({"queue_latency_ms": 500}, "queue_latency_above_threshold"),
        ({"supported": 2}, "proof_support_below_threshold"),
    ],
)
def test_resource_or_assurance_value_regression_holds_rollout(
    changes: dict[str, object], reason: str
) -> None:
    decision = FormalPlanningRolloutGate().evaluate(
        _report(**changes), target_mode=RolloutMode.ENFORCEMENT, now=NOW
    )

    assert not decision.promotion_allowed
    assert decision["lane_decisions"][0]["disposition"] == "hold"
    assert reason in decision["lane_decisions"][0]["remaining_reason_codes"]


def test_unavailable_and_low_value_lanes_remain_advisory() -> None:
    healthy = _dimensions()
    unavailable = _dimensions(
        property_class="protocol",
        translator_profile="proverif-v1",
        prover="proverif",
        kernel="none",
        assurance=AssuranceLevel.SOLVER_CHECKED,
    )
    samples = [
        _sample(BenchmarkMode.COLD, sample_id="healthy-c", dimensions=healthy),
        _sample(BenchmarkMode.WARM, sample_id="healthy-w", dimensions=healthy),
        _sample(
            BenchmarkMode.COLD,
            sample_id="missing-c",
            dimensions=unavailable,
            available=False,
            low_value=True,
            supported=0,
        ),
        _sample(
            BenchmarkMode.WARM,
            sample_id="missing-w",
            dimensions=unavailable,
            available=False,
            low_value=True,
            supported=0,
        ),
    ]
    report = build_formal_planning_benchmark_report(samples, generated_at=NOW)
    decision = FormalPlanningRolloutGate().evaluate(
        report, target_mode=RolloutMode.ENFORCEMENT, now=NOW
    )

    assert decision.promotion_allowed
    advisory = next(
        item
        for item in decision["lane_decisions"]
        if item["lane_id"] == unavailable.lane_id
    )
    assert advisory["disposition"] == RolloutDisposition.ADVISORY.value
    assert not advisory["eligible_for_blocking"]
    assert {"lane_unavailable", "lane_low_value"} <= set(
        advisory["remaining_reason_codes"]
    )


def test_all_advisory_matrix_cannot_claim_rollout_success() -> None:
    decision = FormalPlanningRolloutGate().evaluate(
        _report(available=False, low_value=True, supported=0),
        target_mode=RolloutMode.ENFORCEMENT,
        now=NOW,
    )
    assert not decision.promotion_allowed
    assert decision["executable_lane_count"] == 0


def test_override_is_durable_expiring_and_exactly_lane_scoped(tmp_path) -> None:
    dimensions = _dimensions()
    report = _report(dimensions=dimensions, cpu=90)
    first = FormalPlanningRolloutGate().evaluate(
        report, target_mode=RolloutMode.ENFORCEMENT, now=NOW
    )
    assert not first.promotion_allowed

    override = FormalPlanningRolloutOverride.create(
        policy_id=first["policy_id"],
        dimensions=dimensions,
        target_mode=RolloutMode.ENFORCEMENT,
        waived_reason_codes=("cpu_saturation_above_threshold",),
        actor="operator:alice",
        reason="temporary isolated benchmark host contention",
        ttl_seconds=600,
        now=NOW,
        ticket_id="OPS-294",
    )
    store = FormalPlanningOverrideStore(tmp_path / "overrides")
    path = store.persist(override)
    loaded = store.load(override.receipt_id)
    assert loaded == override
    assert path.stat().st_mode & 0o777 == 0o600

    overridden = FormalPlanningRolloutGate().evaluate(
        report,
        target_mode=RolloutMode.ENFORCEMENT,
        overrides=(loaded,),
        now=NOW,
    )
    assert overridden.promotion_allowed
    assert overridden["lane_decisions"][0]["waived_reason_codes"] == [
        "cpu_saturation_above_threshold"
    ]

    other = _report(
        dimensions=_dimensions(property_class="authorization"), cpu=90
    )
    not_cross_lane = FormalPlanningRolloutGate().evaluate(
        other,
        target_mode=RolloutMode.ENFORCEMENT,
        overrides=(loaded,),
        now=NOW,
    )
    assert not not_cross_lane.promotion_allowed


def test_override_cannot_promote_an_unavailable_lane() -> None:
    dimensions = _dimensions()
    report = _report(
        dimensions=dimensions, available=False, low_value=True, supported=0
    )
    override = FormalPlanningRolloutOverride.create(
        policy_id="formal-planning-rollout-v1",
        dimensions=dimensions,
        target_mode=RolloutMode.ENFORCEMENT,
        waived_reason_codes=(
            "proof_support_below_threshold",
            "authoritative_assurance_below_threshold",
        ),
        actor="operator:alice",
        reason="test unavailable boundary",
        ttl_seconds=60,
        now=NOW,
    )
    decision = FormalPlanningRolloutGate().evaluate(
        report,
        target_mode=RolloutMode.ENFORCEMENT,
        overrides=(override,),
        now=NOW,
    )
    assert not decision.promotion_allowed
    assert decision["lane_decisions"][0]["disposition"] == "advisory"


def test_operator_projection_is_complete_and_drops_raw_payloads() -> None:
    report = _report()
    decision = FormalPlanningRolloutGate().evaluate(
        report, target_mode=RolloutMode.ENFORCEMENT, now=NOW
    )
    projection = build_formal_planning_operator_projection(
        report,
        decision,
        active_formal_plans=(
            {
                "plan_id": "plan-1",
                "status": "running",
                "task_count": 3,
                "prompt": "must never appear",
            },
        ),
        unmet_obligations=(
            {
                "obligation_id": "obligation-1",
                "plan_id": "plan-1",
                "required_assurance": "kernel_verified",
                "reason_code": "kernel_pending",
                "raw_context": "must never appear",
            },
        ),
        trace_violations=(
            {
                "violation_id": "violation-1",
                "plan_id": "plan-1",
                "event_count": 2,
                "reason_code": "transition_reordered",
                "events": [{"secret": "must never appear"}],
            },
        ),
        generated_at=NOW,
    )

    assert projection["executable_matrix"]
    assert projection["rollout_decisions"]
    assert projection["active_formal_plans"][0]["plan_id"] == "plan-1"
    assert projection["unmet_obligations"][0]["obligation_id"] == "obligation-1"
    assert projection["trace_violations"][0]["violation_id"] == "violation-1"
    encoded = json.dumps(projection, sort_keys=True)
    for forbidden in ("must never appear", '"prompt"', '"raw_context"', '"events"'):
        assert forbidden not in encoded
    assert projection["contains_raw_context"] is False
    assert projection["contains_trace_payloads"] is False


def test_collector_is_initialized_bounded_and_rejects_duplicate_records() -> None:
    collector = FormalPlanningMetricsCollector(max_samples=2)
    collector.record(_sample(BenchmarkMode.COLD))
    collector.record(_sample(BenchmarkMode.WARM))
    assert collector.report(generated_at=NOW)["sample_count"] == 2
    with pytest.raises(FormalPlanningMetricsError, match="bound"):
        collector.record(
            _sample(BenchmarkMode.COLD, sample_id="third")
        )


def test_report_identity_and_private_field_boundaries_fail_closed() -> None:
    payload = _report().to_dict()
    payload["summary"]["prompt"] = "private"
    with pytest.raises(FormalPlanningMetricsError):
        FormalPlanningBenchmarkReport(payload)

    with pytest.raises(FormalPlanningMetricsError, match="defects"):
        _sample(BenchmarkMode.COLD, defects=11)


def test_operator_projection_rejects_mismatched_decision() -> None:
    first = _report()
    second = _report(dimensions=_dimensions(property_class="authorization"))
    decision = FormalPlanningRolloutGate().evaluate(
        first, target_mode=RolloutMode.ENFORCEMENT, now=NOW
    )
    with pytest.raises(FormalPlanningRolloutError, match="do not match"):
        build_formal_planning_operator_projection(second, decision)
