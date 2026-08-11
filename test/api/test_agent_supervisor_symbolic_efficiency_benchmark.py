"""Tests for the generalized symbolic-efficiency benchmark."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.symbolic_efficiency_benchmark import (
    BenchmarkConclusion,
    BenchmarkProfile,
    CacheMeasurement,
    FindingMeasurement,
    FindingTruth,
    FixtureIdentity,
    GateStatus,
    InvalidationMeasurement,
    InventoryMeasurement,
    ProviderPacketMeasurement,
    ResourceMeasurement,
    ScanMode,
    SymbolicBenchmarkError,
    SymbolicBenchmarkObservation,
    SymbolicBenchmarkPolicy,
    SymbolicBenchmarkPopulation,
    SymbolicEfficiencyBenchmarkReport,
    TaskMeasurement,
    ToolchainIdentity,
    build_symbolic_efficiency_report,
    evaluate_symbolic_efficiency,
    verify_symbolic_efficiency_report,
)

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "validation"
    / "symbolic_efficiency_benchmark.py"
)

_DOMAIN_LITERALS = re.compile(
    r"(?i)\\b(?:vfs|ipfs(?!_accelerate_py)|fsspec|swissknife|swiss[_-]?knife)\\b"
)


def build_vfs_equivalent_policy() -> SymbolicBenchmarkPolicy:
    """Inject locked VFS schemas/stages/modes rather than renaming in place."""
    return SymbolicBenchmarkPolicy(
        policy_id="policy:vfs-symbolic-efficiency@1",
        policy_revision="revision:1",
        evidence_schema="vfs/symbolic-efficiency-benchmark@1",
        observation_schema="vfs/symbolic-efficiency-observation@1",
        population_schema="vfs/symbolic-efficiency-population@1",
        version=1,
        deterministic_stage_names=(
            "inventory",
            "scan",
            "parse",
            "identity",
            "graph",
            "contract",
            "cache",
            "proof",
        ),
        required_cache_stages=("ast", "graph", "contract", "proof"),
        required_scan_modes=("cold", "warm", "exact", "delta"),
        required_finding_truths=("true", "false", "unknown"),
        invalidation_mode="delta",
        exact_reuse_mode="exact",
        partial_reuse_modes=("warm", "delta"),
    )


def build_widget_policy() -> SymbolicBenchmarkPolicy:
    """Unrelated profile with custom stages, modes, and bounds."""
    return SymbolicBenchmarkPolicy(
        policy_id="policy:widget-efficiency@1",
        policy_revision="revision:widget-1",
        evidence_schema="widget/efficiency-benchmark@1",
        observation_schema="widget/efficiency-observation@1",
        population_schema="widget/efficiency-population@1",
        version=1,
        deterministic_stage_names=("discover", "normalize", "index"),
        required_cache_stages=("index", "manifest"),
        required_scan_modes=("baseline", "incremental"),
        required_finding_truths=("true", "false"),
        invalidation_mode="incremental",
        exact_reuse_mode="baseline",
        partial_reuse_modes=("incremental",),
        max_observations=100,
        max_observation_bytes=256 * 1024,
        max_population_bytes=2 * 1024 * 1024,
        max_report_bytes=256 * 1024,
    )


VFS_POLICY = build_vfs_equivalent_policy()
VFS_PROFILE = BenchmarkProfile(
    profile_id="profile:vfs-035",
    profile_revision="revision:1",
    policy=VFS_POLICY,
)
WIDGET_POLICY = build_widget_policy()
WIDGET_PROFILE = BenchmarkProfile(
    profile_id="profile:widget-efficiency",
    profile_revision="revision:widget-1",
    policy=WIDGET_POLICY,
    minimum_samples_per_mode=2,
    minimum_packet_pairs=2,
    minimum_provider_reduction_basis_points=5_000,
    packet_input_budget_bytes=8_192,
    max_wall_time_ns=30_000_000_000,
    max_cpu_time_ns=30_000_000_000,
    max_peak_rss_bytes=512 * 1024 * 1024,
    max_process_count=8,
    max_disk_growth_bytes=64 * 1024 * 1024,
    max_artifact_bytes=64 * 1024 * 1024,
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


def _cache_measurements(
    mode: str,
    stages: tuple[str, ...],
    *,
    hits_by_mode: dict[str, int] | None = None,
    lookups: int = 4,
) -> tuple[CacheMeasurement, ...]:
    hits_map = hits_by_mode or {
        "cold": 0,
        "warm": 3,
        "exact": 4,
        "delta": 3,
        "baseline": 4,
        "incremental": 2,
    }
    hits = hits_map.get(mode, 0)
    return tuple(
        CacheMeasurement(
            stage=stage,
            lookups=lookups,
            hits=hits,
            reused_artifacts=hits,
            reused_bytes=hits * 100,
            produced_artifacts=lookups - hits,
            produced_bytes=(lookups - hits) * 100,
        )
        for stage in stages
    )


def _findings(
    sample_index: int,
    truths: tuple[str, ...] = ("true", "false", "unknown"),
) -> tuple[FindingMeasurement, ...]:
    items: list[FindingMeasurement] = []
    for truth in truths:
        if truth == "true":
            items.append(
                FindingMeasurement(
                    seed_id="seed:true",
                    expected_truth=FindingTruth.TRUE,
                    observed_truth=FindingTruth.TRUE,
                    evidence_ids=("evidence:true",),
                    time_to_counterexample_ns=1_000_000 + sample_index,
                    counterexample_id=f"counterexample:{sample_index}",
                )
            )
        elif truth == "false":
            items.append(
                FindingMeasurement(
                    seed_id="seed:false",
                    expected_truth=FindingTruth.FALSE,
                    observed_truth=FindingTruth.FALSE,
                    evidence_ids=("evidence:false",),
                )
            )
        else:
            items.append(
                FindingMeasurement(
                    seed_id=f"seed:{truth}",
                    expected_truth=FindingTruth(truth),
                    observed_truth=FindingTruth(truth),
                    evidence_ids=(f"evidence:{truth}",),
                )
            )
    return tuple(items)


def _observation(
    mode: str,
    sample_index: int,
    *,
    profile: BenchmarkProfile = VFS_PROFILE,
) -> SymbolicBenchmarkObservation:
    policy = profile.policy
    evidence_ids = tuple(f"evidence:{t}" for t in policy.required_finding_truths)
    seed_ids = tuple(f"seed:{t}" for t in policy.required_finding_truths)
    # Pair evidence labels for false/true/unknown from _findings
    if set(policy.required_finding_truths) == {"true", "false", "unknown"}:
        evidence_ids = ("evidence:false", "evidence:true", "evidence:unknown")
        seed_ids = ("seed:false", "seed:true", "seed:unknown")
    return SymbolicBenchmarkObservation(
        mode=mode,
        sample_index=sample_index,
        fixture=FIXTURE,
        toolchain=TOOLCHAIN,
        profile_id=profile.profile_id,
        profile_revision=profile.profile_revision,
        inventory=InventoryMeasurement(
            observed_paths=10,
            emitted_paths=10,
            included_paths=8,
            excluded_paths=2,
            omitted_paths=0,
            exhaustive=True,
        ),
        caches=_cache_measurements(mode, policy.required_cache_stages),
        deterministic_stage_llm_calls=tuple(
            (stage, 0) for stage in policy.deterministic_stage_names
        ),
        findings=_findings(sample_index, policy.required_finding_truths),
        invalidation=(
            InvalidationMeasurement(
                changed_source_ids=("source:a",),
                expected_invalidated_ids=("node:a", "node:b"),
                actual_invalidated_ids=("node:a", "node:b"),
            )
            if mode == policy.invalidation_mode
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
            pair_id=f"pair:{mode}:{sample_index}",
            baseline_context_bound_bytes=120_000,
            baseline_input_bytes=100_000 + sample_index,
            baseline_input_tokens=20_000 + sample_index,
            packet_input_bytes=min(10_000, profile.packet_input_budget_bytes - 1),
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


def _population(
    samples_per_mode: int = 3,
    *,
    profile: BenchmarkProfile = VFS_PROFILE,
) -> SymbolicBenchmarkPopulation:
    policy = profile.policy
    return SymbolicBenchmarkPopulation(
        profile=profile,
        observations=tuple(
            _observation(mode, sample_index, profile=profile)
            for mode in policy.required_scan_modes
            for sample_index in range(1, samples_per_mode + 1)
        ),
    )


def test_generic_module_contains_no_product_domain_literals() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    # Strip the module docstring and comments before scanning bodies.
    matches = _DOMAIN_LITERALS.findall(source)
    assert matches == [], f"domain literals leaked into generic module: {matches}"


def test_vfs_equivalent_population_passes_all_gates() -> None:
    population = _population()
    report = evaluate_symbolic_efficiency(population)

    assert report.conclusion is BenchmarkConclusion.PASSED
    assert report.passed
    assert not report.authoritative
    assert not report.completion_authoritative
    assert not report.promotion_authoritative
    assert report.failure_codes == ()
    assert all(gate.status is GateStatus.PASSED for gate in report.gates)
    assert report.evidence_schema == "vfs/symbolic-efficiency-benchmark@1"
    assert report.policy_identity_id == VFS_POLICY.identity_id
    assert dict(report.sample_counts_by_mode) == {
        mode: 3 for mode in VFS_POLICY.required_scan_modes
    }
    assert report.observation_count == 12
    assert report.inventory_observed_paths == 120
    assert report.inventory_complete_observations == 12
    assert dict(report.cache_lookups_by_stage) == {
        stage: 48 for stage in VFS_POLICY.required_cache_stages
    }
    assert dict(report.cache_hits_by_stage) == {
        stage: 30 for stage in VFS_POLICY.required_cache_stages
    }
    assert report.invalidation_expected_count == 6
    assert report.invalidation_false_positive_count == 0
    assert report.invalidation_false_negative_count == 0
    assert dict(report.seeded_expected_by_truth) == {
        truth: 12 for truth in VFS_POLICY.required_finding_truths
    }
    assert report.counterexample_count == 12
    assert report.median_counterexample_time_ns == {
        "numerator": 1_000_002,
        "denominator": 1,
    }
    assert report.provider_byte_reduction_basis_points >= 8_000
    assert report.provider_token_reduction_basis_points >= 8_000
    assert report.deterministic_llm_calls == 0
    assert report.profile_identity_id == VFS_PROFILE.identity_id
    assert verify_symbolic_efficiency_report(report, population)
    assert build_symbolic_efficiency_report(population) == report


def test_byte_equivalent_locked_vfs_observations_yield_equivalent_decisions() -> None:
    """Two identical VFS-injected observation payloads must decide identically."""
    first = _population()
    # Rebuild from canonical JSON — byte-equivalent locked observations.
    second = SymbolicBenchmarkPopulation.from_json(first.to_json())
    assert first.to_json() == second.to_json()
    assert first.population_id == second.population_id

    report_a = evaluate_symbolic_efficiency(first)
    report_b = evaluate_symbolic_efficiency(second)
    assert report_a.to_dict() == report_b.to_dict()
    assert report_a.conclusion is report_b.conclusion
    assert [g.to_dict() for g in report_a.gates] == [
        g.to_dict() for g in report_b.gates
    ]
    # Observation schema is the injected VFS identity, not the generic default.
    obs_payload = json.loads(first.observations[0].to_json(policy=VFS_POLICY))
    assert obs_payload["schema"] == "vfs/symbolic-efficiency-observation@1"
    pop_payload = json.loads(first.to_json())
    assert pop_payload["schema"] == "vfs/symbolic-efficiency-population@1"


def test_observation_population_and_report_are_canonical_replay_artifacts() -> None:
    observation = _observation(ScanMode.COLD, 1).validate(VFS_POLICY)
    replayed_observation = SymbolicBenchmarkObservation.from_json(
        observation.to_json(policy=VFS_POLICY),
        policy=VFS_POLICY,
    )
    assert replayed_observation == observation
    assert replayed_observation.observation_id_for(VFS_POLICY) == (
        observation.observation_id_for(VFS_POLICY)
    )

    population = _population()
    replayed_population = SymbolicBenchmarkPopulation.from_json(population.to_json())
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
        SymbolicBenchmarkObservation.from_json(
            '{"schema":1,"schema":2}',
            policy=VFS_POLICY,
        )


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
        if observation.mode == "cold" and observation.sample_index == 1:
            inventory = InventoryMeasurement(
                observed_paths=10,
                emitted_paths=9,
                included_paths=8,
                excluded_paths=1,
                omitted_paths=1,
                exhaustive=False,
                unexplained_gap_codes=("missing:path",),
            )
            calls = ((VFS_POLICY.deterministic_stage_names[0], 1),) + calls[1:]
            packet = replace(packet, packet_evidence_ids=("evidence:true",))
            resources = replace(
                resources,
                wall_time_ns=VFS_PROFILE.max_wall_time_ns + 1,
                idle_write_operations=1,
                idle_write_bytes=1,
            )
            findings = tuple(
                replace(finding, observed_truth=FindingTruth.UNKNOWN)
                if finding.expected_truth is FindingTruth.FALSE
                else finding
                for finding in findings
            )
        if observation.mode == "exact" and observation.sample_index == 1:
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
        if observation.mode == "delta" and observation.sample_index == 1:
            assert invalidation is not None
            invalidation = replace(
                invalidation,
                actual_invalidated_ids=("node:a",),
            )
        changed.append(
            replace(
                observation,
                inventory=inventory,
                caches=caches,
                deterministic_stage_llm_calls=calls,
                findings=findings,
                invalidation=invalidation,
                packet=packet,
                resources=resources,
            )
        )

    report = evaluate_symbolic_efficiency(
        SymbolicBenchmarkPopulation(profile=VFS_PROFILE, observations=tuple(changed))
    )

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
    observation = _observation("cold", 1)
    incomplete = replace(observation, caches=observation.caches[:-1])
    with pytest.raises(SymbolicBenchmarkError, match="cache stage"):
        incomplete.validate(VFS_POLICY)
    with pytest.raises(SymbolicBenchmarkError, match="deterministic stages"):
        replace(
            observation,
            deterministic_stage_llm_calls=observation.deterministic_stage_llm_calls[:-1],
        ).validate(VFS_POLICY)

    oversized_packet = replace(
        observation.packet,
        packet_input_bytes=VFS_PROFILE.packet_input_budget_bytes + 1,
    )
    with pytest.raises(SymbolicBenchmarkError, match="profile input budget"):
        SymbolicBenchmarkPopulation(
            profile=VFS_PROFILE,
            observations=(replace(observation, packet=oversized_packet),),
        )

    population_json = json.loads(_population().to_json())
    population_json["population_id"] = "sha256:" + "0" * 64
    with pytest.raises(SymbolicBenchmarkError, match="population ID mismatch"):
        SymbolicBenchmarkPopulation.from_dict(population_json)


def test_malformed_stale_mixed_profile_and_forged_report_fail_closed() -> None:
    population = _population()

    # Malformed: unsupported schema.
    obs = json.loads(population.observations[0].to_json(policy=VFS_POLICY))
    obs["schema"] = "stale/symbolic-efficiency-observation@0"
    with pytest.raises(SymbolicBenchmarkError, match="unsupported observation schema"):
        SymbolicBenchmarkObservation.from_dict(obs, policy=VFS_POLICY)

    # Stale: version skew.
    obs = json.loads(population.observations[0].to_json(policy=VFS_POLICY))
    obs["version"] = 99
    # Rebuild observation_id would still fail schema/version check first.
    with pytest.raises(SymbolicBenchmarkError, match="unsupported observation schema"):
        SymbolicBenchmarkObservation.from_dict(obs, policy=VFS_POLICY)

    # Mixed-profile: observation detached from profile.
    mixed = replace(
        population.observations[0],
        profile_id="profile:other",
    )
    with pytest.raises(SymbolicBenchmarkError, match="detached from benchmark profile"):
        SymbolicBenchmarkPopulation(
            profile=VFS_PROFILE,
            observations=(mixed,) + population.observations[1:],
        )

    # Under-sampled is a soft gate (not hard fail) — already covered, assert no promotion.
    under = evaluate_symbolic_efficiency(_population(samples_per_mode=1))
    assert under.conclusion is BenchmarkConclusion.INSUFFICIENT_SAMPLES
    assert not under.promotion_authoritative

    # Resource-exceeding: hard fail on resource ceilings.
    heavy = []
    for item in _population().observations:
        heavy.append(
            replace(
                item,
                resources=replace(
                    item.resources,
                    peak_rss_bytes=VFS_PROFILE.max_peak_rss_bytes + 1,
                ),
            )
        )
    heavy_report = evaluate_symbolic_efficiency(
        SymbolicBenchmarkPopulation(profile=VFS_PROFILE, observations=tuple(heavy))
    )
    assert heavy_report.conclusion is BenchmarkConclusion.FAILED
    assert "resource-ceilings" in heavy_report.failure_codes

    # Forged report: mutate a gate after evaluation.
    good = evaluate_symbolic_efficiency(population)
    forged = good.to_dict()
    forged["conclusion"] = "passed"
    forged["failure_codes"] = []
    # Even if conclusion already passed, change a metric.
    forged["deterministic_llm_calls"] = 99
    with pytest.raises(SymbolicBenchmarkError, match="complete population"):
        SymbolicEfficiencyBenchmarkReport.from_dict(forged, population=population)
    forged_report = SymbolicEfficiencyBenchmarkReport(
        **{
            **{name: getattr(good, name) for name in good.__dataclass_fields__},
            "deterministic_llm_calls": 99,
        }
    )
    assert not verify_symbolic_efficiency_report(forged_report, population)


def test_widget_profile_exercises_custom_stages_and_bounds() -> None:
    population = _population(samples_per_mode=2, profile=WIDGET_PROFILE)
    report = evaluate_symbolic_efficiency(population)

    assert report.conclusion is BenchmarkConclusion.PASSED
    assert report.evidence_schema == "widget/efficiency-benchmark@1"
    assert dict(report.sample_counts_by_mode) == {"baseline": 2, "incremental": 2}
    assert dict(report.cache_lookups_by_stage) == {"index": 16, "manifest": 16}
    assert set(dict(report.seeded_expected_by_truth)) == {"true", "false"}
    assert report.observation_count == 4
    assert report.deterministic_llm_calls == 0
    assert verify_symbolic_efficiency_report(report, population)

    # Widget stages are not the VFS stage vocabulary.
    assert "inventory" not in dict(report.cache_lookups_by_stage)
    assert "cold" not in dict(report.sample_counts_by_mode)

    # Widget bound rejects oversized packet more tightly.
    obs = _observation("baseline", 1, profile=WIDGET_PROFILE)
    with pytest.raises(SymbolicBenchmarkError, match="profile input budget"):
        SymbolicBenchmarkPopulation(
            profile=WIDGET_PROFILE,
            observations=(
                replace(
                    obs,
                    packet=replace(
                        obs.packet,
                        packet_input_bytes=WIDGET_PROFILE.packet_input_budget_bytes + 1,
                    ),
                ),
            ),
        )


def test_scan_mode_enum_accepted_for_default_vocabulary() -> None:
    obs = _observation(ScanMode.WARM, 1)
    assert obs.mode == "warm"
    closed = obs.validate(VFS_POLICY)
    assert closed.mode == "warm"


def test_rational_integer_arithmetic_and_finite_canonical_records() -> None:
    population = _population()
    report = evaluate_symbolic_efficiency(population)
    # Medians are exact rational dictionaries, not floats.
    for field in (
        "median_counterexample_time_ns",
        "median_baseline_input_bytes",
        "median_packet_input_bytes",
        "median_baseline_input_tokens",
        "median_packet_input_tokens",
    ):
        value = getattr(report, field)
        assert set(value) == {"numerator", "denominator"}
        assert isinstance(value["numerator"], int)
        assert isinstance(value["denominator"], int)
        assert value["denominator"] > 0
    # Canonical JSON is finite (no NaN/Infinity) and stable under reload.
    raw = report.to_json()
    assert "NaN" not in raw and "Infinity" not in raw
    reloaded = json.loads(raw)
    assert reloaded["authoritative"] is False
    assert reloaded["completion_authoritative"] is False
    assert reloaded["promotion_authoritative"] is False

