"""IPS-037: proving, aggregation, verification, storage, and savings metrics."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.metrics import (
    BASIS_POINTS,
    COMPARISON_SCHEMA,
    COST_SCHEMA,
    DEFAULT_AGGREGATE_CPU_MS,
    DEFAULT_LEAF_CPU_MS,
    DEFAULT_VERIFY_CPU_MS,
    EVIDENCE_SUBSET,
    MetricProvenance,
    MetricsError,
    ProofCostComparison,
    ProofCostRecord,
    ProofMetricsCollector,
    ProvenancedValue,
    ResourceTimings,
    RunVisibility,
    UnitCounts,
    compare_costs,
    compare_full_and_incremental,
    estimate_path_cost,
    estimated,
    measured,
    unavailable,
    unit_counts_from_sets,
)


def _counts(
    *,
    required: int = 10,
    reused: int = 7,
    invalidated: int = 2,
    added: int = 1,
    removed: int = 0,
    simulated: int = 0,
) -> UnitCounts:
    newly = invalidated + added
    return UnitCounts(
        required=required,
        reused=reused,
        invalidated=invalidated,
        added=added,
        removed=removed,
        newly_proved=newly,
        simulated=simulated,
    )


def test_evidence_subset_and_schemas() -> None:
    assert EVIDENCE_SUBSET == "ips/proof-cost@1"
    assert COST_SCHEMA.endswith("proof-cost-record@1")
    assert COMPARISON_SCHEMA.endswith("proof-cost-comparison@1")
    assert BASIS_POINTS == 10_000


def test_unit_count_arithmetic_invariants() -> None:
    counts = _counts()
    assert counts.required == counts.reused + counts.newly_proved
    assert counts.newly_proved == counts.invalidated + counts.added
    assert counts.cache_hit_basis_points == (7 * BASIS_POINTS) // 10

    with pytest.raises(MetricsError, match="required must equal"):
        UnitCounts(
            required=5,
            reused=2,
            invalidated=1,
            added=1,
            removed=0,
            newly_proved=2,
        )

    with pytest.raises(MetricsError, match="newly_proved must equal"):
        UnitCounts(
            required=4,
            reused=2,
            invalidated=1,
            added=0,
            removed=0,
            newly_proved=2,
        )


def test_unit_counts_from_sets_partition() -> None:
    counts = unit_counts_from_sets(
        required=("a", "b", "c", "d"),
        reused=("a", "b"),
        invalidated=("c",),
        added=("d",),
        removed=("e",),
    )
    assert counts.required == 4
    assert counts.reused == 2
    assert counts.newly_proved == 2
    assert counts.removed == 1
    assert counts.cache_hit_basis_points == BASIS_POINTS // 2

    with pytest.raises(MetricsError, match="partition"):
        unit_counts_from_sets(
            required=("a", "b"),
            reused=("a",),
            newly_proved=(),
        )


def test_provenanced_value_rejects_estimated_as_measured_and_unknown_zero() -> None:
    assert measured(12, "ms").is_measured
    assert estimated(12, "ms").is_estimated
    assert unavailable("ms").value is None
    assert unavailable("ms").provenance is MetricProvenance.UNAVAILABLE

    with pytest.raises(MetricsError, match="unavailable"):
        ProvenancedValue(0, MetricProvenance.UNAVAILABLE, "ms")

    with pytest.raises(MetricsError, match="numeric value"):
        ProvenancedValue(None, MetricProvenance.MEASURED, "ms")

    with pytest.raises(MetricsError, match="int"):
        ProvenancedValue(1.5, MetricProvenance.MEASURED, "ms")  # type: ignore[arg-type]


def test_estimate_path_cost_is_deterministic_for_equivalent_work() -> None:
    counts = _counts(required=10, reused=7, invalidated=2, added=1)
    first = estimate_path_cost(counts, full=True, aggregate_nodes=2)
    second = estimate_path_cost(counts, full=True, aggregate_nodes=2)
    assert first.to_canonical() == second.to_canonical()
    assert first.cpu_ms.provenance is MetricProvenance.ESTIMATED
    assert first.cpu_ms.value == 10 * DEFAULT_LEAF_CPU_MS + 2 * DEFAULT_AGGREGATE_CPU_MS + 10 * DEFAULT_VERIFY_CPU_MS

    incremental = estimate_path_cost(counts, full=False, aggregate_nodes=2)
    assert incremental.leaf_proving_ms.value == 3 * DEFAULT_LEAF_CPU_MS
    assert incremental.cpu_ms.is_estimated
    assert (first.cpu_ms.value or 0) > (incremental.cpu_ms.value or 0)


def test_compare_costs_compute_saved_uses_equivalent_required_work() -> None:
    counts = _counts(required=10, reused=7, invalidated=2, added=1)
    comparison = compare_costs(counts, aggregate_nodes=2)
    assert isinstance(comparison, ProofCostComparison)
    assert comparison.counts == counts
    assert comparison.full.counts == comparison.incremental.counts
    assert comparison.full.path == "full"
    assert comparison.incremental.path == "incremental"

    full_ms = comparison.full.cost_ms.value
    inc_ms = comparison.incremental.cost_ms.value
    assert full_ms is not None and inc_ms is not None
    expected_saved = full_ms - inc_ms
    assert comparison.compute_saved_ms.value == expected_saved
    assert comparison.compute_saved_basis_points.value == (
        expected_saved * BASIS_POINTS
    ) // full_ms
    # Cost-model comparison is estimated, never measured.
    assert comparison.compute_saved_ms.provenance is MetricProvenance.ESTIMATED
    assert comparison.full.cost_ms.provenance is MetricProvenance.ESTIMATED
    assert comparison.incremental.cost_ms.provenance is MetricProvenance.ESTIMATED


def test_measured_savings_require_measured_both_sides() -> None:
    counts = _counts(required=4, reused=2, invalidated=1, added=1)
    full = ResourceTimings(
        leaf_proving_ms=measured(4000, "ms"),
        aggregation_ms=measured(50, "ms"),
        verification_ms=measured(80, "ms"),
        wall_ms=measured(4200, "ms"),
        cpu_ms=measured(4130, "ms"),
        gpu_ms=measured(0, "ms"),
        peak_memory_bytes=measured(64 * 1024 * 1024, "bytes"),
        proof_size_bytes=measured(16_384, "bytes"),
        seal_size_bytes=measured(20_000, "bytes"),
        storage_growth_bytes=measured(20_000, "bytes"),
    )
    incremental = ResourceTimings(
        leaf_proving_ms=measured(2000, "ms"),
        aggregation_ms=measured(50, "ms"),
        verification_ms=measured(80, "ms"),
        wall_ms=measured(2200, "ms"),
        cpu_ms=measured(2130, "ms"),
        gpu_ms=measured(0, "ms"),
        peak_memory_bytes=measured(32 * 1024 * 1024, "bytes"),
        proof_size_bytes=measured(8_192, "bytes"),
        seal_size_bytes=measured(10_000, "bytes"),
        storage_growth_bytes=measured(10_000, "bytes"),
    )
    comparison = compare_costs(
        counts,
        full_timings=full,
        incremental_timings=incremental,
    )
    assert comparison.compute_saved_ms.provenance is MetricProvenance.MEASURED
    assert comparison.compute_saved_ms.value == 4130 - 2130
    assert comparison.storage_saved_bytes.provenance is MetricProvenance.MEASURED
    assert comparison.storage_saved_bytes.value == 10_000

    mixed = compare_costs(
        counts,
        full_timings=full,
        incremental_timings=estimate_path_cost(counts, full=False),
    )
    assert mixed.compute_saved_ms.provenance is MetricProvenance.ESTIMATED


def test_estimates_never_reported_as_measurements() -> None:
    counts = _counts()
    comparison = compare_full_and_incremental(counts)
    for record in (comparison.full, comparison.incremental):
        assert record.cost_ms.provenance is not MetricProvenance.MEASURED
        for field in record.timings.to_canonical().values():
            assert field["provenance"] != MetricProvenance.MEASURED.value

    with pytest.raises(MetricsError, match="measured compute savings"):
        ProofCostComparison(
            schema=COMPARISON_SCHEMA,
            evidence_subset=EVIDENCE_SUBSET,
            counts=counts,
            full=comparison.full,
            incremental=comparison.incremental,
            compute_saved_ms=measured(1, "ms"),
            compute_saved_basis_points=measured(1, "basis_points"),
            storage_saved_bytes=unavailable("bytes"),
            visibility=RunVisibility.SUCCESS,
        )


def test_failed_and_fallback_runs_remain_visible() -> None:
    collector = ProofMetricsCollector()
    counts = _counts(required=5, reused=0, invalidated=5, added=0)

    failed = collector.record(
        counts,
        visibility=RunVisibility.FAILED,
    )
    fallback = collector.record(
        counts,
        visibility=RunVisibility.FALLBACK_FULL,
        fallback_reason="schema_change",
    )
    timeout = collector.record(
        counts,
        visibility=RunVisibility.TIMEOUT,
    )
    success = collector.record(
        _counts(required=5, reused=4, invalidated=1, added=0),
        visibility=RunVisibility.SUCCESS,
    )

    assert len(collector.history) == 4
    visible = collector.visible_failures()
    assert failed in visible
    assert fallback in visible
    assert timeout in visible
    assert success not in visible
    assert fallback.fallback_reason == "schema_change"
    assert fallback.visibility is RunVisibility.FALLBACK_FULL
    assert failed.is_visible_failure
    assert collector.to_canonical()["visible_failure_count"] == 3

    with pytest.raises(MetricsError, match="fallback_reason"):
        compare_costs(counts, visibility=RunVisibility.FALLBACK_FULL)


def test_absent_counters_remain_unavailable_not_zero() -> None:
    timings = ResourceTimings()
    assert timings.cpu_ms.provenance is MetricProvenance.UNAVAILABLE
    assert timings.cpu_ms.value is None
    assert timings.peak_memory_bytes.value is None
    assert timings.gpu_ms.value is None

    counts = _counts(required=2, reused=1, invalidated=1, added=0)
    comparison = compare_costs(
        counts,
        full_timings=ResourceTimings(cpu_ms=unavailable("ms"), wall_ms=unavailable("ms")),
        incremental_timings=ResourceTimings(
            cpu_ms=unavailable("ms"), wall_ms=unavailable("ms")
        ),
    )
    assert comparison.compute_saved_ms.provenance is MetricProvenance.UNAVAILABLE
    assert comparison.compute_saved_ms.value is None
    assert comparison.storage_saved_bytes.provenance is MetricProvenance.UNAVAILABLE


def test_wall_cpu_gpu_remain_distinct() -> None:
    timings = ResourceTimings(
        wall_ms=measured(5000, "ms"),
        cpu_ms=measured(4200, "ms"),
        gpu_ms=measured(800, "ms"),
        leaf_proving_ms=measured(4000, "ms"),
        aggregation_ms=measured(100, "ms"),
        verification_ms=measured(100, "ms"),
    )
    canonical = timings.to_canonical()
    assert canonical["wall_ms"]["value"] == 5000
    assert canonical["cpu_ms"]["value"] == 4200
    assert canonical["gpu_ms"]["value"] == 800
    # Cost prefers CPU over wall when both measured.
    record = ProofCostRecord(
        schema=COST_SCHEMA,
        evidence_subset=EVIDENCE_SUBSET,
        path="incremental",
        counts=_counts(required=2, reused=1, invalidated=1, added=0),
        timings=timings,
        cost_ms=measured(4200, "ms"),
        visibility=RunVisibility.SUCCESS,
    )
    assert record.cost_ms.value == 4200

    with pytest.raises(MetricsError, match="unit must be 'ms'"):
        ResourceTimings(cpu_ms=measured(1, "bytes"))


def test_collector_record_measured_incremental_keeps_full_estimated() -> None:
    collector = ProofMetricsCollector()
    counts = _counts(required=6, reused=4, invalidated=1, added=1)
    comparison = collector.record_measured_incremental(
        counts,
        leaf_proving_ms=2000,
        aggregation_ms=50,
        verification_ms=120,
        wall_ms=2500,
        cpu_ms=2170,
        peak_memory_bytes=10_000_000,
        proof_size_bytes=8192,
        seal_size_bytes=9000,
        storage_growth_bytes=9000,
        aggregate_nodes=2,
        chain_depth=3,
    )
    assert comparison.incremental.cost_ms.is_measured
    assert comparison.full.cost_ms.is_estimated
    assert comparison.compute_saved_ms.is_estimated
    assert comparison.incremental.timings.peak_memory_bytes.value == 10_000_000
    assert comparison.incremental.timings.gpu_ms.provenance is MetricProvenance.UNAVAILABLE
    assert comparison.incremental.chain_depth == 3
    assert comparison.comparison_cid().startswith("sha256:")
    # Deterministic CID for identical inputs.
    again = collector.record_measured_incremental(
        counts,
        leaf_proving_ms=2000,
        aggregation_ms=50,
        verification_ms=120,
        wall_ms=2500,
        cpu_ms=2170,
        peak_memory_bytes=10_000_000,
        proof_size_bytes=8192,
        seal_size_bytes=9000,
        storage_growth_bytes=9000,
        aggregate_nodes=2,
        chain_depth=3,
    )
    assert again.comparison_cid() == comparison.comparison_cid()


def test_simulated_units_excluded_from_production_measured_claims() -> None:
    counts = _counts(
        required=5,
        reused=2,
        invalidated=2,
        added=1,
        simulated=1,
    )
    full = ResourceTimings(
        cpu_ms=measured(5000, "ms"),
        wall_ms=measured(5000, "ms"),
        leaf_proving_ms=measured(5000, "ms"),
        aggregation_ms=measured(0, "ms"),
        verification_ms=measured(0, "ms"),
        storage_growth_bytes=measured(20_000, "bytes"),
    )
    incremental = ResourceTimings(
        cpu_ms=measured(2000, "ms"),
        wall_ms=measured(2000, "ms"),
        leaf_proving_ms=measured(2000, "ms"),
        aggregation_ms=measured(0, "ms"),
        verification_ms=measured(0, "ms"),
        storage_growth_bytes=measured(8_000, "bytes"),
    )
    comparison = compare_costs(
        counts,
        full_timings=full,
        incremental_timings=incremental,
    )
    assert comparison.production_claims_excluded is True
    assert comparison.compute_saved_ms.provenance is MetricProvenance.ESTIMATED
    assert "simulated_units_excluded_from_production_compute_claims" in (
        comparison.full.notes
    )


def test_compare_full_and_incremental_mapping_inputs() -> None:
    comparison = compare_full_and_incremental(
        {
            "required": 3,
            "reused": 2,
            "invalidated": 1,
            "added": 0,
            "removed": 0,
        },
        visibility="success",
    )
    assert comparison.counts.required == 3
    assert comparison.counts.newly_proved == 1
    assert comparison.visibility is RunVisibility.SUCCESS


def test_canonical_round_trip_is_stable() -> None:
    counts = _counts()
    a = compare_costs(counts, aggregate_nodes=3, chain_depth=2)
    b = compare_costs(counts, aggregate_nodes=3, chain_depth=2)
    assert a.to_canonical() == b.to_canonical()
    assert a.comparison_cid() == b.comparison_cid()
    assert a.full.record_cid() == b.full.record_cid()
