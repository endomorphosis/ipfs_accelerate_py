"""IPS-037: measured versus estimated proof-cost accounting."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.metrics import (
    COST_EVIDENCE,
    UNKNOWN,
    CostProvenance,
    ProofMetricsCollector,
    RunDisposition,
    compare_costs,
)


def _measured_run(*, cpu: int, wall: int, storage: int, failed: bool = False) -> object:
    collector = ProofMetricsCollector()
    collector.record_units(required=4, reused=2, invalidated=1, proved=2, cache_hits=2)
    collector.observe_leaf_ms(10)
    collector.observe_aggregate_ms(4)
    collector.observe_verify_ms(3)
    collector.observe_wall_ms(wall)
    collector.observe_cpu_ms(cpu)
    collector.observe_storage_growth_bytes(storage)
    if failed:
        collector.mark_failed("proof_failed")
    return collector.snapshot()


def test_evidence_subset() -> None:
    assert COST_EVIDENCE == "ips/proof-cost@1"


def test_absent_counters_are_unknown_not_zero() -> None:
    record = ProofMetricsCollector().snapshot()
    assert record.gpu_time_ms.provenance is CostProvenance.UNKNOWN
    assert record.gpu_time_ms.value is None
    assert record.gpu_time_ms.to_canonical()["value"] == UNKNOWN
    assert record.peak_memory_bytes.provenance is CostProvenance.UNKNOWN
    assert record.estimated is False


def test_observed_zero_is_measured_zero() -> None:
    collector = ProofMetricsCollector()
    collector.observe_gpu_ms(0)
    record = collector.snapshot()
    assert record.gpu_time_ms.value == 0
    assert record.gpu_time_ms.provenance is CostProvenance.MEASURED


def test_estimates_never_reported_as_measurements() -> None:
    collector = ProofMetricsCollector(estimated=True)
    collector.observe_cpu_ms(900)
    record = collector.snapshot()
    assert record.estimated is True
    assert record.cpu_time_ms.provenance is CostProvenance.ESTIMATED
    assert record.cpu_time_ms.to_canonical()["provenance"] == "estimated"


def test_compare_costs_uses_equivalent_required_work() -> None:
    full = _measured_run(cpu=1000, wall=1200, storage=8000)
    incremental = _measured_run(cpu=400, wall=500, storage=3000)
    comparison = compare_costs(full, incremental)
    assert comparison.compute_saved_cpu_ms == 600
    assert comparison.compute_saved_wall_ms == 700
    assert comparison.storage_saved_bytes == 5000
    assert comparison.savings_provenance is CostProvenance.MEASURED
    assert comparison.visible_failure is False
    assert comparison.to_canonical()["estimated_as_measured"] is False


def test_failed_or_estimated_runs_keep_savings_unknown() -> None:
    full = _measured_run(cpu=1000, wall=1200, storage=8000)
    failed = _measured_run(cpu=400, wall=500, storage=3000, failed=True)
    comparison = compare_costs(full, failed)
    assert comparison.visible_failure is True
    assert comparison.compute_saved_cpu_ms is None
    assert comparison.to_canonical()["compute_saved_cpu_ms"] == UNKNOWN
    assert comparison.savings_provenance is CostProvenance.UNKNOWN
    assert failed.disposition is RunDisposition.FAILED
    assert failed.fallback_reason == "proof_failed"

    estimated = ProofMetricsCollector(estimated=True)
    estimated.observe_cpu_ms(100)
    estimated.observe_wall_ms(100)
    estimated.observe_storage_growth_bytes(100)
    blocked = compare_costs(full, estimated.snapshot())
    assert blocked.visible_failure is True
    assert blocked.compute_saved_cpu_ms is None
