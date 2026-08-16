"""Tests for DuckDBQuackBenchmarkReport@1 (DQP-036)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_baseline import (
    SAFETY_FLOOR_KEYS,
    BaselineCriteria,
    BaselineEnvironment,
    BaselineStratum,
    BaselineVerdict,
    MetricSample,
    TelemetryStatus,
    UnavailableReason,
    default_workload,
    establish_duckdb_quack_baselines,
)
from ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_benchmark import (
    BENCHMARK_CONTRACT_VERSION,
    DUCKDB_QUACK_BENCHMARK_INTERFACE,
    EVIDENCE,
    GOAL_ID,
    TASK_ID,
    BenchmarkVerdict,
    ComparisonStatus,
    DuckDBQuackBenchmarkError,
    DuckDBQuackBenchmarkReport,
    SafetyFloorSnapshot,
    compare_baseline_to_candidate,
    establish_candidate_observations,
    run_duckdb_quack_benchmark,
)


TREE_ID = "tree:sha256:dqp036-fixture-tree"


def _fixed_environment() -> BaselineEnvironment:
    return BaselineEnvironment(
        python_version="3.12.0",
        platform_name="Linux-fixed",
        implementation="CPython",
        path_fingerprint="sha256:" + ("ef" * 32),
        duckdb_version="1.5.2",
        extra={"machine": "x86_64", "system": "Linux"},
    )


def _fixed_workload(*, sample_count: int = 8):
    return default_workload(seed=0xD036, sample_count=sample_count)


# ---------------------------------------------------------------------------
# Interface identity
# ---------------------------------------------------------------------------


def test_interface_identity() -> None:
    assert DUCKDB_QUACK_BENCHMARK_INTERFACE == "DuckDBQuackBenchmarkReport@1"
    assert DuckDBQuackBenchmarkReport.INTERFACE == DUCKDB_QUACK_BENCHMARK_INTERFACE
    assert BENCHMARK_CONTRACT_VERSION == 1
    assert TASK_ID == "DQP-036"
    assert GOAL_ID == "DQP-G050"
    assert EVIDENCE == "dqp/duckdb-quack-benchmark@1"


# ---------------------------------------------------------------------------
# Happy path: warm reuse, no floor regression, duplicates gone
# ---------------------------------------------------------------------------


def test_hermetic_benchmark_passes_warm_reuse_and_floors() -> None:
    report = run_duckdb_quack_benchmark(
        tree_id=TREE_ID,
        environment=_fixed_environment(),
        workload=_fixed_workload(),
        repository_id="repository:dqp-036",
    )
    assert report.verdict is BenchmarkVerdict.PASSED
    assert report.passed is True
    assert report.warm_reuse_improved is True
    assert report.duplicates_eliminated is True
    assert report.quality_non_inferior is True
    assert report.safety_floors_zero is True
    assert report.latency_within_bounds is True
    assert report.promotion_allowed is False
    assert report.environment_equivalent is True
    assert report.missing_telemetry == ()
    assert report.sample_count > 0
    assert report.confidence_bps > 0

    payload = report.to_dict()
    assert payload["interface"] == DUCKDB_QUACK_BENCHMARK_INTERFACE
    assert payload["passed"] is True
    assert payload["task_id"] == "DQP-036"
    assert payload["causality_inferred"] is False
    assert payload["promotion_allowed"] is False
    assert "identity_id" in payload

    # All four strata compared.
    strata = {item.stratum for item in report.stratum_comparisons}
    assert strata == {s.value for s in BaselineStratum}

    warm = next(
        item
        for item in report.stratum_comparisons
        if item.stratum == BaselineStratum.WARM.value
    )
    assert warm.warm_reuse_improved is True
    assert warm.duplicates_eliminated is True
    reuse = next(d for d in warm.deltas if d.metric_name == "cache_reuse_hits")
    assert reuse.candidate_value is not None
    assert reuse.baseline_value is not None
    assert reuse.candidate_value > reuse.baseline_value
    dups = next(d for d in warm.deltas if d.metric_name == "duplicate_semantic_inputs")
    assert dups.candidate_value == 0


def test_safety_floors_cover_sealed_keys() -> None:
    snap = SafetyFloorSnapshot.zeros()
    assert set(snap.floors) == set(SAFETY_FLOOR_KEYS)
    assert snap.all_zero is True
    report = run_duckdb_quack_benchmark(
        tree_id=TREE_ID,
        environment=_fixed_environment(),
        workload=_fixed_workload(),
        candidate_safety=snap,
    )
    assert report.candidate_safety.all_zero is True
    assert all(v == 0 for v in report.baseline_safety.to_dict().values())


# ---------------------------------------------------------------------------
# Missing telemetry honesty
# ---------------------------------------------------------------------------


def test_missing_telemetry_reported_honestly_not_as_zero() -> None:
    report = run_duckdb_quack_benchmark(
        tree_id=TREE_ID,
        environment=_fixed_environment(),
        workload=_fixed_workload(),
        missing_metrics=("cache_reuse_hits",),
    )
    assert report.verdict in {
        BenchmarkVerdict.INSUFFICIENT,
        BenchmarkVerdict.FAILED,
    }
    assert report.passed is False
    assert report.missing_telemetry
    assert any("cache_reuse_hits" in item for item in report.missing_telemetry)

    # No fabricated measured zero for the missing warm reuse metric.
    warm = next(
        item
        for item in report.stratum_comparisons
        if item.stratum == BaselineStratum.WARM.value
    )
    reuse = next(d for d in warm.deltas if d.metric_name == "cache_reuse_hits")
    assert reuse.status is ComparisonStatus.UNAVAILABLE
    assert reuse.candidate_value is None
    assert "value" not in reuse.to_dict() or reuse.to_dict().get("candidate_value") is None
    assert reuse.reason_code


def test_metric_delta_omits_numeric_when_unavailable() -> None:
    report = run_duckdb_quack_benchmark(
        tree_id=TREE_ID,
        environment=_fixed_environment(),
        workload=_fixed_workload(),
        missing_metrics=("provider_calls",),
    )
    unavailable = [
        delta
        for comparison in report.stratum_comparisons
        for delta in comparison.deltas
        if delta.metric_name == "provider_calls"
        and delta.status is ComparisonStatus.UNAVAILABLE
    ]
    assert unavailable
    for delta in unavailable:
        payload = delta.to_dict()
        assert payload["candidate_status"] == TelemetryStatus.UNAVAILABLE.value
        assert "candidate_value" not in payload
        assert "delta" not in payload


# ---------------------------------------------------------------------------
# Floor regressions
# ---------------------------------------------------------------------------


def test_nonzero_candidate_safety_fails() -> None:
    bad = SafetyFloorSnapshot(
        floors={
            **{key: 0 for key in SAFETY_FLOOR_KEYS},
            "unauthorized_sql": 1,
        }
    )
    report = run_duckdb_quack_benchmark(
        tree_id=TREE_ID,
        environment=_fixed_environment(),
        workload=_fixed_workload(),
        candidate_safety=bad,
    )
    assert report.verdict is BenchmarkVerdict.FAILED
    assert report.safety_floors_zero is False
    assert "safety_floor_nonzero" in report.reason_codes


def test_baseline_must_be_established() -> None:
    env = _fixed_environment()
    work = _fixed_workload(sample_count=1)  # may still establish; force reject path
    # Build a real established baseline then mark it rejected via criteria path:
    # use compare with a handcrafted insufficient baseline by lowering samples
    # below min through a synthetic object is hard; instead raise via empty
    # criteria misuse: establish then compare requires ESTABLISHED.
    baseline = establish_duckdb_quack_baselines(
        tree_id=TREE_ID,
        environment=env,
        workload=default_workload(seed=1, sample_count=8),
        criteria=BaselineCriteria.sealed_defaults(),
    )
    assert baseline.verdict is BaselineVerdict.ESTABLISHED

    # Corrupt path: construct candidate and pass a baseline with rejected verdict
    # by replacing via object.__setattr__ is frozen — use a shallow clone dict path.
    # Instead, feed compare with a non-established baseline by re-establishing
    # with completion_authoritative which fails at assert; so we only test the
    # explicit error path by subclassing is not needed — force via replace of
    # verdict on a rebuilt report is not allowed.  Raise by calling compare
    # after swapping verdict through object on a non-frozen intermediate is
    # awkward; use DuckDBQuackBaselineReport reconstruction.
    from ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_baseline import (
        DuckDBQuackBaselineReport,
    )

    rejected = DuckDBQuackBaselineReport(
        state_baseline=baseline.state_baseline,
        llm_churn_baseline=baseline.llm_churn_baseline,
        criteria=baseline.criteria,
        verdict=BaselineVerdict.REJECTED,
        reason_codes=("forced",),
    )
    cand_state, cand_llm, *_ = establish_candidate_observations(
        tree_id=f"{TREE_ID}:c",
        environment=env,
        workload=default_workload(seed=1, sample_count=8),
    )
    with pytest.raises(DuckDBQuackBenchmarkError, match="established"):
        compare_baseline_to_candidate(
            baseline=rejected,
            candidate_state_strata=cand_state,
            candidate_llm_strata=cand_llm,
            candidate_tree_id=f"{TREE_ID}:c",
        )


# ---------------------------------------------------------------------------
# Criteria / environment binding
# ---------------------------------------------------------------------------


def test_report_binds_criteria_identity() -> None:
    criteria = BaselineCriteria.sealed_defaults()
    report = run_duckdb_quack_benchmark(
        tree_id=TREE_ID,
        environment=_fixed_environment(),
        workload=_fixed_workload(),
        criteria=criteria,
    )
    assert report.criteria_identity == criteria.identity_id


def test_unknown_safety_floor_key_rejected() -> None:
    with pytest.raises(DuckDBQuackBenchmarkError, match="unknown safety floor"):
        SafetyFloorSnapshot(floors={"not_a_real_floor": 0})


def test_cannot_infer_causality_flag_is_sealed() -> None:
    report = run_duckdb_quack_benchmark(
        tree_id=TREE_ID,
        environment=_fixed_environment(),
        workload=_fixed_workload(),
    )
    assert report.to_dict()["causality_inferred"] is False


def test_round_trip_dict_shape() -> None:
    report = run_duckdb_quack_benchmark(
        tree_id=TREE_ID,
        environment=_fixed_environment(),
        workload=_fixed_workload(),
    )
    payload = report.to_dict()
    for key in (
        "schema",
        "interface",
        "verdict",
        "warm_reuse_improved",
        "duplicates_eliminated",
        "quality_non_inferior",
        "safety_floors_zero",
        "latency_within_bounds",
        "missing_telemetry",
        "stratum_comparisons",
        "aggregate_deltas",
        "reason_codes",
    ):
        assert key in payload
