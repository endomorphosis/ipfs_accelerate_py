from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.vfs_symbolic_benchmark import (
    BenchmarkConclusion,
    BenchmarkProfile,
    CacheMeasurement,
    DETERMINISTIC_STAGE_NAMES,
    FindingMeasurement,
    FindingTruth,
    FixtureIdentity,
    GateStatus,
    InvalidationMeasurement,
    InventoryMeasurement,
    ProviderPacketMeasurement,
    REQUIRED_CACHE_STAGES,
    REQUIRED_SCAN_MODES,
    ResourceMeasurement,
    ScanMode,
    SymbolicBenchmarkError,
    SymbolicBenchmarkObservation,
    SymbolicBenchmarkPopulation,
    SymbolicEfficiencyBenchmarkReport,
    TaskMeasurement,
    ToolchainIdentity,
    build_symbolic_efficiency_report,
    evaluate_symbolic_efficiency,
    verify_symbolic_efficiency_report,
)


PROFILE = BenchmarkProfile(
    profile_id="profile:vfs-035",
    profile_revision="revision:1",
)
FIXTURE = FixtureIdentity(
    fixture_id="fixture:symbolic-efficiency",
    fixture_revision="fixture-revision:1",
    repository_id="repository:test",
    forest_id="forest:test",
    dirty_overlay_id="overlay:clean",
    inventory_policy_id="inventory-policy:test",
    inventory_policy_revision="inventory-policy-revision:1",
    seeded_findings_id="seed-set:true-false-unknown",
)
TOOLCHAIN = ToolchainIdentity(
    scanner_id="scanner:1",
    parser_id="parser:1",
    analyzer_id="analyzer:1",
    graph_schema_id="graph-schema:1",
    resolver_id="resolver:1",
    contract_schema_id="contract-schema:1",
    prover_id="prover:1",
    proof_circuit_id="proof-circuit:1",
    cache_schema_id="cache-schema:1",
    packet_schema_id="packet-schema:1",
    provider_id="provider:test",
    provider_revision="provider-revision:1",
    tokenizer_id="tokenizer:test",
)


def _cache_measurements(mode: ScanMode) -> tuple[CacheMeasurement, ...]:
    hits = {
        ScanMode.COLD: 0,
        ScanMode.WARM: 3,
        ScanMode.EXACT: 4,
        ScanMode.DELTA: 3,
    }[mode]
    return tuple(
        CacheMeasurement(
            stage=stage,
            lookups=4,
            hits=hits,
            reused_artifacts=hits,
            reused_bytes=hits * 100,
            produced_artifacts=4 - hits,
            produced_bytes=(4 - hits) * 100,
        )
        for stage in REQUIRED_CACHE_STAGES
    )


def _findings(sample_index: int) -> tuple[FindingMeasurement, ...]:
    return (
        FindingMeasurement(
            seed_id="seed:true",
            expected_truth=FindingTruth.TRUE,
            observed_truth=FindingTruth.TRUE,
            evidence_ids=("evidence:true",),
            time_to_counterexample_ns=1_000_000 + sample_index,
            counterexample_id=f"counterexample:{sample_index}",
        ),
        FindingMeasurement(
            seed_id="seed:false",
            expected_truth=FindingTruth.FALSE,
            observed_truth=FindingTruth.FALSE,
            evidence_ids=("evidence:false",),
        ),
        FindingMeasurement(
            seed_id="seed:unknown",
            expected_truth=FindingTruth.UNKNOWN,
            observed_truth=FindingTruth.UNKNOWN,
            evidence_ids=("evidence:unknown",),
        ),
    )


def _observation(
    mode: ScanMode,
    sample_index: int,
) -> SymbolicBenchmarkObservation:
    evidence_ids = (
        "evidence:false",
        "evidence:true",
        "evidence:unknown",
    )
    seed_ids = ("seed:false", "seed:true", "seed:unknown")
    return SymbolicBenchmarkObservation(
        mode=mode,
        sample_index=sample_index,
        fixture=FIXTURE,
        toolchain=TOOLCHAIN,
        profile_id=PROFILE.profile_id,
        profile_revision=PROFILE.profile_revision,
        inventory=InventoryMeasurement(
            observed_paths=10,
            emitted_paths=10,
            included_paths=8,
            excluded_paths=2,
            omitted_paths=0,
            exhaustive=True,
        ),
        caches=_cache_measurements(mode),
        deterministic_stage_llm_calls=tuple(
            (stage, 0) for stage in DETERMINISTIC_STAGE_NAMES
        ),
        findings=_findings(sample_index),
        invalidation=(
            InvalidationMeasurement(
                changed_source_ids=("source:a",),
                expected_invalidated_ids=("node:a", "node:b"),
                actual_invalidated_ids=("node:a", "node:b"),
            )
            if mode is ScanMode.DELTA
            else None
        ),
        tasks=TaskMeasurement(
            candidate_findings=4,
            eligible_findings=3,
            emitted_tasks=2,
            deduplicated_findings=1,
            duplicate_group_ids=("duplicate:root",),
        ),
        packet=ProviderPacketMeasurement(
            pair_id=f"pair:{mode.value}:{sample_index}",
            baseline_context_bound_bytes=120_000,
            baseline_input_bytes=100_000 + sample_index,
            baseline_input_tokens=20_000 + sample_index,
            packet_input_bytes=10_000,
            packet_input_tokens=1_500,
            baseline_required_evidence_ids=evidence_ids,
            packet_evidence_ids=evidence_ids,
            baseline_seed_coverage_ids=seed_ids,
            packet_seed_coverage_ids=seed_ids,
        ),
        resources=ResourceMeasurement(
            wall_time_ns=1_000_000_000 + sample_index,
            cpu_time_ns=500_000_000 + sample_index,
            peak_rss_bytes=64 * 1024 * 1024,
            peak_process_count=2,
            disk_bytes_before=1_000_000,
            disk_bytes_after=1_004_096,
            artifact_bytes=2_048,
            idle_observation_ns=2_000_000_000,
            idle_cpu_time_ns=10_000_000,
            idle_write_operations=0,
            idle_write_bytes=0,
        ),
        source_receipt_ids=("receipt:inventory", "receipt:scan"),
    )


def _population(samples_per_mode: int = 3) -> SymbolicBenchmarkPopulation:
    return SymbolicBenchmarkPopulation(
        profile=PROFILE,
        observations=tuple(
            _observation(ScanMode(mode), sample_index)
            for mode in REQUIRED_SCAN_MODES
            for sample_index in range(1, samples_per_mode + 1)
        ),
    )


def test_report_measures_full_symbolic_efficiency_population() -> None:
    population = _population()
    report = evaluate_symbolic_efficiency(population)

    assert report.conclusion is BenchmarkConclusion.PASSED
    assert report.passed
    assert not report.authoritative
    assert not report.completion_authoritative
    assert not report.promotion_authoritative
    assert report.failure_codes == ()
    assert all(gate.status is GateStatus.PASSED for gate in report.gates)

    assert dict(report.sample_counts_by_mode) == {
        mode: 3 for mode in REQUIRED_SCAN_MODES
    }
    assert report.observation_count == 12
    assert report.inventory_observed_paths == 120
    assert report.inventory_emitted_paths == 120
    assert report.inventory_included_paths == 96
    assert report.inventory_excluded_paths == 24
    assert report.inventory_omitted_paths == 0
    assert report.inventory_complete_observations == 12
    assert dict(report.cache_lookups_by_stage) == {
        stage: 48 for stage in REQUIRED_CACHE_STAGES
    }
    assert dict(report.cache_hits_by_stage) == {
        stage: 30 for stage in REQUIRED_CACHE_STAGES
    }
    assert dict(report.cache_reused_artifacts_by_stage) == {
        stage: 30 for stage in REQUIRED_CACHE_STAGES
    }
    assert dict(report.cache_reused_bytes_by_stage) == {
        stage: 3_000 for stage in REQUIRED_CACHE_STAGES
    }
    assert report.invalidation_expected_count == 6
    assert report.invalidation_actual_count == 6
    assert report.invalidation_false_positive_count == 0
    assert report.invalidation_false_negative_count == 0
    assert dict(report.seeded_expected_by_truth) == {
        truth.value: 12 for truth in FindingTruth
    }
    assert dict(report.seeded_covered_by_truth) == {
        truth.value: 12 for truth in FindingTruth
    }
    assert report.counterexample_count == 12
    assert report.median_counterexample_time_ns == {
        "numerator": 1_000_002,
        "denominator": 1,
    }
    assert report.artifact_bytes == 24_576
    assert report.wall_time_ns == 12_000_000_024
    assert report.cpu_time_ns == 6_000_000_024
    assert dict(report.scan_wall_time_ns_by_mode) == {
        mode: 3_000_000_006 for mode in REQUIRED_SCAN_MODES
    }
    assert dict(report.scan_cpu_time_ns_by_mode) == {
        mode: 1_500_000_006 for mode in REQUIRED_SCAN_MODES
    }
    assert report.peak_rss_bytes == 64 * 1024 * 1024
    assert report.peak_process_count == 2
    assert report.disk_growth_bytes == 49_152
    assert report.idle_cpu_time_ns == 120_000_000
    assert report.idle_write_operations == 0
    assert report.idle_write_bytes == 0
    assert (
        report.candidate_findings,
        report.eligible_findings,
        report.emitted_tasks,
        report.deduplicated_findings,
    ) == (48, 36, 24, 12)
    assert report.provider_pair_count == 12
    assert report.provider_byte_reduction_basis_points >= 8_000
    assert report.provider_token_reduction_basis_points >= 8_000
    assert report.deterministic_llm_calls == 0
    assert len(report.fixture_identity_ids) == 1
    assert len(report.toolchain_identity_ids) == 1
    assert report.profile_identity_id == PROFILE.identity_id
    assert verify_symbolic_efficiency_report(report, population)
    assert build_symbolic_efficiency_report(population) == report


def test_observation_population_and_report_are_canonical_replay_artifacts() -> None:
    observation = _observation(ScanMode.COLD, 1)
    replayed_observation = SymbolicBenchmarkObservation.from_json(
        observation.to_json()
    )
    assert replayed_observation == observation
    assert replayed_observation.observation_id == observation.observation_id

    population = _population()
    replayed_population = SymbolicBenchmarkPopulation.from_json(
        population.to_json()
    )
    assert replayed_population == population
    assert replayed_population.population_id == population.population_id

    report = evaluate_symbolic_efficiency(population)
    replayed_report = SymbolicEfficiencyBenchmarkReport.from_json(
        report.to_json(),
        population=population,
    )
    assert replayed_report == report
    tampered = report.to_dict()
    tampered["provider_pair_count"] += 1
    with pytest.raises(SymbolicBenchmarkError, match="complete population"):
        SymbolicEfficiencyBenchmarkReport.from_dict(
            tampered,
            population=population,
        )
    with pytest.raises(SymbolicBenchmarkError, match="duplicate keys"):
        SymbolicBenchmarkObservation.from_json('{"schema":1,"schema":2}')


def test_insufficient_samples_never_make_a_promotion_claim() -> None:
    population = _population(samples_per_mode=1)
    report = evaluate_symbolic_efficiency(population)

    assert report.conclusion is BenchmarkConclusion.INSUFFICIENT_SAMPLES
    assert not report.passed
    assert not report.authoritative
    assert not report.completion_authoritative
    assert not report.promotion_authoritative
    assert report.failure_codes == ()
    assert (
        report.gate("sample-sufficiency").status
        is GateStatus.INSUFFICIENT_SAMPLES
    )
    assert (
        report.gate("provider-byte-reduction").status
        is GateStatus.INSUFFICIENT_SAMPLES
    )
    assert (
        report.gate("provider-token-reduction").status
        is GateStatus.INSUFFICIENT_SAMPLES
    )


def test_failed_observations_fail_closed_across_required_gates() -> None:
    changed: list[SymbolicBenchmarkObservation] = []
    for observation in _population().observations:
        packet = replace(
            observation.packet,
            baseline_input_bytes=60_000,
            baseline_input_tokens=8_000,
            packet_input_bytes=15_000,
            packet_input_tokens=3_000,
        )
        inventory = observation.inventory
        calls = observation.deterministic_stage_llm_calls
        caches = observation.caches
        findings = observation.findings
        invalidation = observation.invalidation
        resources = observation.resources
        if observation.mode is ScanMode.COLD and observation.sample_index == 1:
            inventory = InventoryMeasurement(
                observed_paths=10,
                emitted_paths=9,
                included_paths=8,
                excluded_paths=1,
                omitted_paths=1,
                exhaustive=False,
                unexplained_gap_codes=("missing:path",),
            )
            calls = ((DETERMINISTIC_STAGE_NAMES[0], 1),) + calls[1:]
            packet = replace(
                packet,
                packet_evidence_ids=("evidence:true",),
            )
            resources = replace(
                resources,
                wall_time_ns=PROFILE.max_wall_time_ns + 1,
                idle_write_operations=1,
                idle_write_bytes=1,
            )
            findings = tuple(
                replace(
                    finding,
                    observed_truth=FindingTruth.UNKNOWN,
                )
                if finding.expected_truth is FindingTruth.FALSE
                else finding
                for finding in findings
            )
        if observation.mode is ScanMode.EXACT and observation.sample_index == 1:
            caches = tuple(
                replace(
                    cache,
                    hits=3,
                    reused_artifacts=3,
                    reused_bytes=300,
                    produced_artifacts=1,
                    produced_bytes=100,
                )
                for cache in caches
            )
        if observation.mode is ScanMode.DELTA and observation.sample_index == 1:
            assert invalidation is not None
            invalidation = replace(
                invalidation,
                actual_invalidated_ids=("node:a",),
            )
        changed.append(replace(
            observation,
            inventory=inventory,
            caches=caches,
            deterministic_stage_llm_calls=calls,
            findings=findings,
            invalidation=invalidation,
            packet=packet,
            resources=resources,
        ))

    report = evaluate_symbolic_efficiency(SymbolicBenchmarkPopulation(
        profile=PROFILE,
        observations=tuple(changed),
    ))

    assert report.conclusion is BenchmarkConclusion.FAILED
    assert set(report.failure_codes) == {
        "cache-reuse",
        "deterministic-zero-llm",
        "idle-quiescence",
        "inventory-completeness",
        "invalidation-precision",
        "packet-evidence-parity",
        "provider-byte-reduction",
        "provider-token-reduction",
        "resource-ceilings",
        "seeded-finding-coverage",
    }
    assert all(
        report.gate(name).status is GateStatus.FAILED
        for name in report.failure_codes
    )
    assert not report.passed
    assert not report.promotion_authoritative


def test_observation_contract_rejects_incomplete_or_unbounded_input() -> None:
    observation = _observation(ScanMode.COLD, 1)
    with pytest.raises(SymbolicBenchmarkError, match="cache measurements"):
        replace(observation, caches=observation.caches[:-1])
    with pytest.raises(SymbolicBenchmarkError, match="all deterministic stages"):
        replace(
            observation,
            deterministic_stage_llm_calls=(
                observation.deterministic_stage_llm_calls[:-1]
            ),
        )

    oversized_packet = replace(
        observation.packet,
        packet_input_bytes=PROFILE.packet_input_budget_bytes + 1,
    )
    with pytest.raises(SymbolicBenchmarkError, match="profile input budget"):
        SymbolicBenchmarkPopulation(
            profile=PROFILE,
            observations=(replace(observation, packet=oversized_packet),),
        )

    population_json = json.loads(_population().to_json())
    population_json["population_id"] = "sha256:" + "0" * 64
    with pytest.raises(SymbolicBenchmarkError, match="population ID mismatch"):
        SymbolicBenchmarkPopulation.from_dict(population_json)
