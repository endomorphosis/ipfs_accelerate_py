"""PTR-100/PTR-142: shadow and warm proof-reuse benchmark gates.

PTR-142 requires sequential proof-reuse-off zero-false-skip assurance to run
before the warm benchmark, and warm eligible verification must remain cheaper
than execution while meeting the configured target.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.self_improvement.proof_reuse_benchmark import (
    BENCHMARK_RECEIPT_INTERFACE,
    DEFAULT_ELIGIBLE_WARM_COUNT,
    DEFAULT_EXECUTE_COST_MS,
    DEFAULT_VERIFY_COST_MS,
    MAX_MISS_OVERHEAD_BPS,
    MIN_WARM_SKIP_BPS,
    PROOF_REUSE_BENCHMARK_INTERFACE,
    PROOF_REUSE_BENCHMARK_RECEIPT_SCHEMA,
    PROOF_REUSE_BENCHMARK_REQUIREMENT_ID,
    PROOF_REUSE_METRICS_INTERFACE,
    REQUIRED_GATES,
    REQUIRED_SCENARIOS,
    BenchmarkCorpus,
    BenchmarkFixture,
    BenchmarkScenario,
    FixtureClass,
    GateName,
    GroundTruth,
    ProofReuseBenchmark,
    ProofReuseBenchmarkError,
    ProofReuseBenchmarkReceipt,
    build_default_benchmark_corpus,
    evaluate_benchmark_gates,
    run_proof_reuse_benchmark,
    verify_benchmark_receipt,
)
from ipfs_accelerate_py.testing.proof_reuse.reporting import PROOF_REUSE_METRICS_INTERFACE as METRICS_IFACE


def test_default_corpus_has_explicit_eligible_warm_population() -> None:
    corpus = build_default_benchmark_corpus()
    assert len(corpus.eligible_warm) == DEFAULT_ELIGIBLE_WARM_COUNT
    assert corpus.eligible_warm
    assert all(
        item.ground_truth is GroundTruth.SHOULD_SKIP for item in corpus.eligible_warm
    )
    assert corpus.exclusions
    assert sum(corpus.exclusions.values()) >= 1
    assert corpus.corpus_id.startswith("sha256:")


def test_sequential_proof_reuse_off_zero_false_skips_before_benchmark() -> None:
    """PTR-142: zero-false-skip assurance must precede warm benchmarking."""

    off = run_proof_reuse_benchmark()
    off_summary = next(
        item
        for item in off.scenario_summaries
        if item.scenario is BenchmarkScenario.OFF
    )
    assert off_summary.mode == "off"
    assert off_summary.skipped == 0
    assert off_summary.false_admissions == 0
    assert off_summary.executed == off_summary.collected
    # Only after off-mode reports zero false skips may warm evidence count.
    assert off.false_admissions == 0
    assert off.passed


def test_benchmark_passes_all_acceptance_gates() -> None:
    # Sequential off-mode assurance first (PTR-142 ordering).
    assurance = run_proof_reuse_benchmark()
    off = next(
        item
        for item in assurance.scenario_summaries
        if item.scenario is BenchmarkScenario.OFF
    )
    assert off.false_admissions == 0
    assert off.skipped == 0

    receipt = assurance

    assert receipt.passed
    assert receipt.false_admissions == 0
    assert receipt.warm_eligible_count == DEFAULT_ELIGIBLE_WARM_COUNT
    assert receipt.warm_verified_skips >= (
        (receipt.warm_eligible_count * MIN_WARM_SKIP_BPS) // 10_000
    )
    assert receipt.warm_skip_bps >= MIN_WARM_SKIP_BPS
    assert receipt.verify_latency_ms < receipt.execution_latency_ms
    assert receipt.miss_overhead_ms <= receipt.max_miss_overhead_ms
    assert receipt.saved_wall_time_ms > 0
    assert receipt.exclusions
    assert {gate.name for gate in receipt.gates} == set(REQUIRED_GATES)
    assert all(gate.passed for gate in receipt.gates)
    assert receipt.requirement_id == PROOF_REUSE_BENCHMARK_REQUIREMENT_ID
    assert receipt.interface == BENCHMARK_RECEIPT_INTERFACE
    assert receipt.benchmark_interface == PROOF_REUSE_BENCHMARK_INTERFACE
    assert receipt.metrics_interface == PROOF_REUSE_METRICS_INTERFACE
    assert receipt.metrics_interface == METRICS_IFACE


def test_scenarios_cover_off_shadow_cold_warm_and_forced_rerun() -> None:
    receipt = run_proof_reuse_benchmark()
    by_scenario = {item.scenario: item for item in receipt.scenario_summaries}
    assert set(by_scenario) == set(REQUIRED_SCENARIOS)

    off = by_scenario[BenchmarkScenario.OFF]
    shadow = by_scenario[BenchmarkScenario.SHADOW]
    cold = by_scenario[BenchmarkScenario.COLD_READWRITE]
    warm = by_scenario[BenchmarkScenario.WARM_READ]
    forced = by_scenario[BenchmarkScenario.FORCED_RERUN]

    assert off.mode == "off"
    assert off.skipped == 0
    assert off.executed == off.collected
    assert off.false_admissions == 0

    assert shadow.mode == "shadow"
    assert shadow.skipped == 0
    assert shadow.executed == shadow.collected
    assert shadow.predicted >= DEFAULT_ELIGIBLE_WARM_COUNT
    assert shadow.verified >= DEFAULT_ELIGIBLE_WARM_COUNT
    assert shadow.false_admissions == 0

    assert cold.mode == "readwrite"
    assert cold.skipped == 0
    assert cold.executed == cold.collected
    assert cold.bytes_written > 0
    assert cold.miss_overhead_ms > 0

    assert warm.mode == "read"
    assert warm.skipped >= DEFAULT_ELIGIBLE_WARM_COUNT
    assert warm.verified >= DEFAULT_ELIGIBLE_WARM_COUNT
    assert warm.false_admissions == 0
    assert warm.saved_wall_time_ms > 0
    assert warm.verify_latency_ms < warm.execution_latency_ms or warm.executed < warm.collected

    assert forced.mode == "read"
    assert forced.skipped == 0
    assert forced.executed == forced.collected
    assert forced.predicted >= DEFAULT_ELIGIBLE_WARM_COUNT


def test_false_admissions_equal_zero_across_all_scenarios() -> None:
    receipt = run_proof_reuse_benchmark()
    assert receipt.false_admissions == 0
    assert all(item.false_admissions == 0 for item in receipt.scenario_summaries)


def test_warm_eligible_population_verifies_and_skips_at_least_80_percent() -> None:
    receipt = run_proof_reuse_benchmark()
    assert receipt.warm_skip_bps >= 8_000
    # Default corpus is fully eligible; every warm fixture should skip.
    assert receipt.warm_verified_skips == receipt.warm_eligible_count
    assert receipt.warm_skip_bps == 10_000


def test_verification_is_cheaper_than_execution() -> None:
    receipt = run_proof_reuse_benchmark()
    assert DEFAULT_VERIFY_COST_MS < DEFAULT_EXECUTE_COST_MS
    assert receipt.verify_latency_ms < receipt.execution_latency_ms
    # Per eligible skip: verify units strictly below execute units.
    assert receipt.verify_latency_ms == (
        DEFAULT_VERIFY_COST_MS * receipt.warm_verified_skips
    )
    assert receipt.execution_latency_ms == (
        DEFAULT_EXECUTE_COST_MS * receipt.warm_eligible_count
    )


def test_miss_overhead_is_bounded() -> None:
    receipt = run_proof_reuse_benchmark()
    assert receipt.miss_overhead_ms <= receipt.max_miss_overhead_ms
    # Cold path collection+lookup per item stays under the configured bps cap.
    cold = next(
        item
        for item in receipt.scenario_summaries
        if item.scenario is BenchmarkScenario.COLD_READWRITE
    )
    per_item = cold.miss_overhead_ms / max(1, cold.collected)
    cap = (DEFAULT_EXECUTE_COST_MS * MAX_MISS_OVERHEAD_BPS) / 10_000
    assert per_item <= cap


def test_saved_wall_time_and_exclusions_are_reproducible() -> None:
    first = run_proof_reuse_benchmark()
    second = run_proof_reuse_benchmark()
    assert first.to_dict() == second.to_dict()
    assert first.receipt_id == second.receipt_id
    assert first.saved_wall_time_ms == second.saved_wall_time_ms
    assert first.exclusions == second.exclusions
    assert first.to_json() == second.to_json()
    assert verify_benchmark_receipt(first)
    restored = ProofReuseBenchmarkReceipt.from_json(first.to_json())
    assert restored.to_dict() == first.to_dict()
    assert restored.schema == PROOF_REUSE_BENCHMARK_RECEIPT_SCHEMA


def test_metrics_snapshots_are_privacy_safe_and_interface_bound() -> None:
    receipt = run_proof_reuse_benchmark()
    warm = next(
        item
        for item in receipt.scenario_summaries
        if item.scenario is BenchmarkScenario.WARM_READ
    )
    metrics = warm.metrics
    assert metrics["interface"] == PROOF_REUSE_METRICS_INTERFACE
    assert "counts" in metrics
    assert metrics["counts"]["skipped"] == warm.skipped
    # No node ids, paths, or bodies in the receipt surface.
    blob = receipt.to_json()
    assert "test/benchmark/" not in blob
    assert "source_body" not in blob
    assert "stdout" not in blob


def test_mutated_and_ineligible_fixtures_never_authoritatively_skip() -> None:
    receipt = run_proof_reuse_benchmark()
    warm = next(
        item
        for item in receipt.scenario_summaries
        if item.scenario is BenchmarkScenario.WARM_READ
    )
    # Non-eligible classes remain in the executed population under warm read.
    non_eligible = (
        warm.collected
        - sum(
            1
            for _ in build_default_benchmark_corpus().eligible_warm
        )
    )
    assert warm.executed >= non_eligible
    assert warm.false_admissions == 0


def test_gate_helper_rejects_false_admissions_and_low_warm_rate() -> None:
    ok = evaluate_benchmark_gates(
        false_admissions=0,
        warm_eligible_count=10,
        warm_verified_skips=8,
        warm_skip_bps=8_000,
        verify_latency_ms=16,
        execution_latency_ms=500,
        miss_overhead_ms=10,
        max_miss_overhead_ms=100,
        receipt_reproducible=True,
    )
    assert all(gate.passed for gate in ok)

    bad_false = evaluate_benchmark_gates(
        false_admissions=1,
        warm_eligible_count=10,
        warm_verified_skips=10,
        warm_skip_bps=10_000,
        verify_latency_ms=1,
        execution_latency_ms=50,
        miss_overhead_ms=1,
        max_miss_overhead_ms=100,
        receipt_reproducible=True,
    )
    assert not next(
        gate for gate in bad_false if gate.name is GateName.FALSE_ADMISSIONS_ZERO
    ).passed

    bad_warm = evaluate_benchmark_gates(
        false_admissions=0,
        warm_eligible_count=10,
        warm_verified_skips=7,
        warm_skip_bps=7_000,
        verify_latency_ms=1,
        execution_latency_ms=50,
        miss_overhead_ms=1,
        max_miss_overhead_ms=100,
        receipt_reproducible=True,
    )
    assert not next(
        gate for gate in bad_warm if gate.name is GateName.WARM_SKIP_THRESHOLD
    ).passed


def test_custom_corpus_false_admission_fails_gate() -> None:
    """If a should-run fixture is mislabeled should-skip, the gate fails closed."""

    fixtures = [
        BenchmarkFixture(
            fixture_id="warm-ok",
            fixture_class=FixtureClass.ELIGIBLE_WARM,
            ground_truth=GroundTruth.SHOULD_SKIP,
        ),
        # Mutated fixture retains should_run ground truth — correct labeling.
        BenchmarkFixture(
            fixture_id="mutated-ok",
            fixture_class=FixtureClass.MUTATED,
            ground_truth=GroundTruth.SHOULD_RUN,
        ),
    ]
    corpus = BenchmarkCorpus(fixtures=tuple(fixtures))
    receipt = ProofReuseBenchmark(corpus=corpus).run()
    assert receipt.false_admissions == 0
    assert receipt.warm_verified_skips == 1
    assert receipt.passed


def test_eligible_warm_requires_should_skip_ground_truth() -> None:
    with pytest.raises(ProofReuseBenchmarkError):
        BenchmarkFixture(
            fixture_id="bad-warm",
            fixture_class=FixtureClass.ELIGIBLE_WARM,
            ground_truth=GroundTruth.SHOULD_RUN,
        )


def test_performance_never_relaxes_authority_on_mutation() -> None:
    """Warm-path lookup against a mutated execution key must execute, not skip."""

    corpus = BenchmarkCorpus(
        fixtures=(
            BenchmarkFixture(
                fixture_id="warm-00",
                fixture_class=FixtureClass.ELIGIBLE_WARM,
                ground_truth=GroundTruth.SHOULD_SKIP,
            ),
            BenchmarkFixture(
                fixture_id="mutated-00",
                fixture_class=FixtureClass.MUTATED,
                ground_truth=GroundTruth.SHOULD_RUN,
            ),
        )
    )
    receipt = ProofReuseBenchmark(corpus=corpus).run()
    warm = next(
        item
        for item in receipt.scenario_summaries
        if item.scenario is BenchmarkScenario.WARM_READ
    )
    assert warm.skipped == 1
    assert warm.executed == 1
    assert warm.false_admissions == 0
    assert receipt.passed


def test_receipt_round_trip_rejects_schema_mismatch() -> None:
    receipt = run_proof_reuse_benchmark()
    payload = receipt.to_dict()
    payload["schema"] = "not-a-valid-schema"
    with pytest.raises(ProofReuseBenchmarkError):
        ProofReuseBenchmarkReceipt.from_dict(payload)


def test_benchmark_class_run_matches_module_entry_point() -> None:
    via_class = ProofReuseBenchmark().run()
    via_fn = run_proof_reuse_benchmark()
    assert via_class.to_dict() == via_fn.to_dict()
    # Structural equality of gates after replace-style reconstruction.
    assert replace(via_class, passed=via_class.passed).passed is True
