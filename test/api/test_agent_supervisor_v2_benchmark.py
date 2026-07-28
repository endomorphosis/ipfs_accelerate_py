from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.self_improvement.supervisor_efficiency_metrics import (
    build_efficiency_baseline_fixtures,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.supervisor_v2_benchmark import (
    REQUIRED_V2_FIXTURE_KINDS,
    V2_CAUSAL_BASELINE_REPORT_ID,
    V2_FROZEN_CAPABILITY_ID,
    V2_FROZEN_CAPABILITY_REVISION,
    V2_FROZEN_OBJECTIVE_ID,
    V2_FROZEN_OBJECTIVE_REVISION,
    V2_FROZEN_POLICY_ID,
    V2_FROZEN_POLICY_REVISION,
    V2_FROZEN_PROVIDER_ID,
    V2_FROZEN_PROVIDER_REVISION,
    V2_FROZEN_REPOSITORY_ID,
    V2_FROZEN_TREE_ID,
    V2_NON_COMPENSABLE_SAFETY_GATES,
    V2_PAIRED_BASELINE_REQUIREMENT_ID,
    V2BenchmarkArm,
    V2BenchmarkReport,
    V2BenchmarkValidationError,
    V2CausalReceipt,
    V2FixtureKind,
    V2FrozenIdentity,
    V2PairedBenchmarkCase,
    V2PairedBenchmarkCorpus,
    adapt_v1_efficiency_receipt,
    build_frozen_v2_paired_corpus,
    build_v2_benchmark_report,
    replace_v2_candidate_metrics,
    verify_v2_benchmark_report,
)


def _corpus() -> V2PairedBenchmarkCorpus:
    return build_frozen_v2_paired_corpus()


def test_frozen_corpus_covers_exact_generation_2_fixture_population() -> None:
    corpus = _corpus()

    assert tuple(item.fixture_kind for item in corpus.cases) == (
        REQUIRED_V2_FIXTURE_KINDS
    )
    assert {item.fixture_kind.value for item in corpus.cases} == {
        "cold",
        "warm",
        "broad-goal",
        "malformed-output",
        "contradictory-input",
        "stale-cache",
        "unavailable-provider",
        "independent-lane",
        "conflicting-lane",
        "failed-validation",
        "restart",
        "drained-board",
        "artifact-pressure",
        "untrusted-repository",
    }
    assert len(corpus.fixture_population_ids) == 14
    assert len(set(corpus.fixture_population_ids)) == 14
    assert corpus.requirement_id == V2_PAIRED_BASELINE_REQUIREMENT_ID


def test_every_pair_freezes_all_semantic_and_observation_identities() -> None:
    corpus = _corpus()

    for case in corpus.cases:
        baseline = case.baseline
        candidate = case.candidate
        assert baseline.input_id == candidate.input_id
        assert baseline.identity.pairing_identity == (
            candidate.identity.pairing_identity
        )
        assert baseline.identity.observation_id != (
            candidate.identity.observation_id
        )
        assert candidate.causal_parent_ids == (baseline.receipt_id,)
        assert baseline.arm is V2BenchmarkArm.BASELINE
        assert candidate.arm is V2BenchmarkArm.CANDIDATE
        for name in (
            "repository_id",
            "tree_id",
            "objective_id",
            "objective_revision",
            "provider_id",
            "provider_revision",
            "capability_id",
            "capability_revision",
            "policy_id",
            "policy_revision",
            "fault_id",
            "observation_id",
        ):
            assert getattr(baseline.identity, name)
            assert getattr(candidate.identity, name)
        assert baseline.identity.repository_id == V2_FROZEN_REPOSITORY_ID
        assert baseline.identity.tree_id == V2_FROZEN_TREE_ID
        assert baseline.identity.objective_id == V2_FROZEN_OBJECTIVE_ID
        assert (
            baseline.identity.objective_revision
            == V2_FROZEN_OBJECTIVE_REVISION
        )
        assert baseline.identity.provider_id == V2_FROZEN_PROVIDER_ID
        assert (
            baseline.identity.provider_revision
            == V2_FROZEN_PROVIDER_REVISION
        )
        assert baseline.identity.capability_id == V2_FROZEN_CAPABILITY_ID
        assert (
            baseline.identity.capability_revision
            == V2_FROZEN_CAPABILITY_REVISION
        )
        assert baseline.identity.policy_id == V2_FROZEN_POLICY_ID
        assert baseline.identity.policy_revision == V2_FROZEN_POLICY_REVISION

    report = build_v2_benchmark_report(corpus)
    assert report.report_id == V2_CAUSAL_BASELINE_REPORT_ID


def test_causal_receipts_join_all_required_measurement_dimensions() -> None:
    metrics = _corpus().cases[0].candidate.metrics

    assert metrics.stage_latencies
    assert metrics.stage_latency_ms > 0
    assert metrics.queue_delay_ms >= 0
    assert metrics.provider_input_tokens > 0
    assert metrics.provider_output_tokens > 0
    assert metrics.provider_reused_input_tokens >= 0
    assert metrics.cache_lookup_count >= metrics.cache_hit_count
    assert metrics.retry_count >= 0
    assert metrics.validation_status
    assert metrics.proof_status
    assert metrics.merge_status
    assert metrics.persistence_status
    assert metrics.idle_cpu_milli_percent >= 0
    assert metrics.required_criterion_ids
    assert metrics.terminal_criteria_complete

    report = build_v2_benchmark_report(_corpus())
    assert set(report.candidate_minus_baseline) == {
        "stage_latency_ms",
        "elapsed_ms",
        "queue_delay_ms",
        "provider_input_tokens",
        "provider_output_tokens",
        "provider_reused_input_tokens",
        "cache_lookup_count",
        "cache_hit_count",
        "cache_reused_bytes",
        "retry_count",
        "retry_input_tokens",
        "retry_output_tokens",
        "validation_latency_ms",
        "proof_latency_ms",
        "merge_latency_ms",
        "persistence_latency_ms",
        "persistence_write_count",
        "persistence_bytes",
        "idle_cpu_milli_percent",
        "artifact_count",
        "artifact_bytes",
        "terminal_required_criteria",
        "terminal_accepted_criteria",
    }


def test_receipts_are_compact_and_never_embed_sensitive_or_nested_bodies() -> None:
    corpus = _corpus()
    encoded = corpus.to_json()
    forbidden = (
        '"prompt"',
        '"prompts"',
        '"source_body"',
        '"source_bodies"',
        '"decoded_output"',
        '"patch"',
        '"patches"',
        '"artifact_graph"',
        '"nested_artifact_graph"',
    )

    assert len(encoded.encode("utf-8")) < 2 * 1024 * 1024
    assert all(value not in encoded for value in forbidden)
    assert all(
        len(case.baseline.canonical_bytes()) < 65_536
        and len(case.candidate.canonical_bytes()) < 65_536
        for case in corpus.cases
    )

    payload = corpus.cases[0].baseline.to_dict()
    payload["prompt"] = "do not persist me"
    with pytest.raises(V2BenchmarkValidationError, match="cannot contain"):
        V2CausalReceipt.from_dict(payload)


def test_deterministic_replay_round_trip_rejects_tampering() -> None:
    corpus = _corpus()
    restored = V2PairedBenchmarkCorpus.from_json(corpus.to_json())
    first = build_v2_benchmark_report(corpus)
    second = build_v2_benchmark_report(restored)

    assert restored == corpus
    assert restored.corpus_id == corpus.corpus_id
    assert second == first
    assert second.report_id == first.report_id
    assert verify_v2_benchmark_report(first, restored) == first
    assert V2BenchmarkReport.from_json(
        json.dumps(first.to_dict(include_report_id=True)),
        corpus=restored,
    ) == first

    altered = first.to_dict(include_report_id=True)
    altered["candidate_minus_baseline"]["provider_input_tokens"] += 1
    with pytest.raises(
        V2BenchmarkValidationError, match="does not match deterministic"
    ):
        V2BenchmarkReport.from_dict(altered, corpus=corpus)

    duplicated_key = '{"schema":"x","schema":"y"}'
    with pytest.raises(V2BenchmarkValidationError, match="duplicate"):
        V2CausalReceipt.from_json(duplicated_key)


def test_closed_population_cannot_be_narrowed_widened_or_duplicated() -> None:
    corpus = _corpus()

    with pytest.raises(V2BenchmarkValidationError, match="cannot be narrowed"):
        V2PairedBenchmarkCorpus(cases=corpus.cases[:-1])
    with pytest.raises(V2BenchmarkValidationError, match="cannot be narrowed"):
        V2PairedBenchmarkCorpus(cases=corpus.cases + (corpus.cases[0],))
    with pytest.raises(V2BenchmarkValidationError, match="cannot be narrowed"):
        V2PairedBenchmarkCorpus(
            cases=(corpus.cases[0],) + corpus.cases[1:-1]
        )

    payload = corpus.to_dict(include_corpus_id=True)
    payload["fixture_population_ids"] = payload["fixture_population_ids"][:-1]
    with pytest.raises(
        V2BenchmarkValidationError, match="population identity"
    ):
        V2PairedBenchmarkCorpus.from_dict(payload)


def test_non_compensable_safety_failure_cannot_be_offset_by_efficiency() -> None:
    corpus = replace_v2_candidate_metrics(
        _corpus(),
        V2FixtureKind.COLD,
        elapsed_ms=1,
        queue_delay_ms=0,
        stage_latencies=(),
        provider_input_tokens=0,
        provider_output_tokens=0,
        provider_reused_input_tokens=0,
        false_completion_count=1,
    )
    report = build_v2_benchmark_report(corpus)
    cold_id = next(
        item.fixture_id
        for item in corpus.cases
        if item.fixture_kind is V2FixtureKind.COLD
    )

    assert report.candidate_minus_baseline["provider_input_tokens"] < 0
    assert not report.non_compensable_safety_passed
    assert not report.passed
    assert report.evidence_claim_ids == ()
    assert report.gate_failures["authority"] == (cold_id,)
    assert set(report.gate_failures) == set(
        V2_NON_COMPENSABLE_SAFETY_GATES
    )


@pytest.mark.parametrize(
    ("kind", "changes", "failed_gate"),
    (
        (
            V2FixtureKind.STALE_CACHE,
            {"stale_authoritative_cache_hit_count": 1},
            "cache-authority",
        ),
        (
            V2FixtureKind.FAILED_VALIDATION,
            {"escaped_validation_failure_count": 1},
            "validation",
        ),
        (
            V2FixtureKind.RESTART,
            {"restart_inconsistency_count": 1},
            "restart-recovery",
        ),
        (
            V2FixtureKind.DRAINED_BOARD,
            {"persistence_write_count": 1},
            "idle-board",
        ),
        (
            V2FixtureKind.ARTIFACT_PRESSURE,
            {"unbounded_artifact_count": 1},
            "artifact-bounds",
        ),
        (
            V2FixtureKind.UNTRUSTED_REPOSITORY,
            {"untrusted_repository_mutation_count": 1},
            "repository-trust",
        ),
    ),
)
def test_fault_specific_safety_gates_fail_closed(
    kind: V2FixtureKind,
    changes: dict[str, int],
    failed_gate: str,
) -> None:
    corpus = replace_v2_candidate_metrics(_corpus(), kind, **changes)
    report = build_v2_benchmark_report(corpus)

    assert not report.passed
    assert report.gate_failures[failed_gate]


def test_baseline_candidate_pairing_rejects_detached_identity_and_parent() -> None:
    case = _corpus().cases[0]
    detached_identity = replace(
        case.candidate.identity,
        provider_revision="sha256:" + "f" * 64,
    )
    detached_candidate = replace(case.candidate, identity=detached_identity)
    with pytest.raises(V2BenchmarkValidationError, match="semantic identities"):
        replace(case, candidate=detached_candidate)

    detached_candidate = replace(
        case.candidate,
        causal_parent_ids=("sha256:" + "e" * 64,),
    )
    with pytest.raises(V2BenchmarkValidationError, match="causally reference"):
        replace(case, candidate=detached_candidate)

    wrong_input = replace(
        case.candidate,
        input_id="sha256:" + "d" * 64,
    )
    with pytest.raises(V2BenchmarkValidationError, match="same input"):
        V2PairedBenchmarkCase(
            fixture_id=case.fixture_id,
            fixture_kind=case.fixture_kind,
            fixture_revision=case.fixture_revision,
            baseline=case.baseline,
            candidate=wrong_input,
        )


def test_v1_receipt_adapter_preserves_compact_measurements_and_source_id() -> None:
    source = build_efficiency_baseline_fixtures()["cold"]
    canonical = _corpus().cases[0].baseline
    identity = V2FrozenIdentity.from_dict(canonical.identity.to_dict())
    adapted = adapt_v1_efficiency_receipt(
        source,
        fixture_kind=V2FixtureKind.COLD,
        arm=V2BenchmarkArm.BASELINE,
        identity=identity,
        input_id=canonical.input_id,
    )

    assert adapted.source_receipt_ids == (source.receipt_id,)
    assert adapted.metrics.provider_input_tokens == source.tokens.input_tokens
    assert adapted.metrics.provider_output_tokens == source.tokens.output_tokens
    assert adapted.metrics.queue_delay_ms == source.queue_delay_ms
    assert adapted.metrics.validation_latency_ms == (
        source.validation.duration_ms
    )
    assert adapted.metrics.proof_latency_ms == source.proof.duration_ms
    assert adapted.metrics.artifact_count == len(source.artifacts)
    assert adapted.metrics.artifact_bytes == sum(
        item.byte_count for item in source.artifacts
    )
    encoded = json.dumps(adapted.to_dict())
    assert "text/x-diff" not in encoded
    assert "src/cold.py" not in encoded


def test_default_causal_baseline_passes_all_non_compensable_gates() -> None:
    report = build_v2_benchmark_report(_corpus())

    assert report.population_complete
    assert report.baseline_candidate_paired
    assert report.non_compensable_safety_passed
    assert report.passed
    assert report.evidence_claim_ids == (V2_PAIRED_BASELINE_REQUIREMENT_ID,)
    assert all(not failures for failures in report.gate_failures.values())
    assert report.candidate.provider_input_tokens < (
        report.baseline.provider_input_tokens
    )
    assert report.candidate.retry_input_tokens < (
        report.baseline.retry_input_tokens
    )
