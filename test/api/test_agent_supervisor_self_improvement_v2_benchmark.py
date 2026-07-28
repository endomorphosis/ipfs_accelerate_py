from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.self_improvement.self_improvement_v2 import (
    MAX_V2_ABLATIONS,
    REQUIRED_V2_OBJECTIVE_DIMENSIONS,
    REWARD_RESISTANT_EVALUATION_REQUIREMENT_ID,
    V2AblationReceipt,
    V2CacheState,
    V2EvaluationDecision,
    V2ObjectiveDimension,
    V2ProducerReceipt,
    V2SelfEvaluationError,
    V2SelfEvaluationReport,
    V2SelfImprovementEvaluator,
    build_frozen_v2_ablation_receipts,
    build_frozen_v2_producer_receipts,
    build_frozen_v2_self_evaluation_inputs,
    evaluate_v2_self_improvement,
    verify_v2_self_evaluation_report,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.self_improvement.supervisor_v2_benchmark import (
    REQUIRED_V2_FIXTURE_KINDS,
    V2BenchmarkArm,
    V2FixtureKind,
    build_frozen_v2_paired_corpus,
    replace_v2_candidate_metrics,
)


def _inputs():
    corpus = build_frozen_v2_paired_corpus()
    producers, ablations = build_frozen_v2_self_evaluation_inputs(corpus)
    return corpus, producers, ablations


def _evaluate(producers, *, corpus=None, ablations=None):
    if corpus is None:
        corpus = build_frozen_v2_paired_corpus()
    if ablations is None:
        ablations = build_frozen_v2_ablation_receipts(corpus, producers)
    return evaluate_v2_self_improvement(corpus, producers, ablations)


def _replace_candidate(producers, dimension, **changes):
    result = []
    for receipt in producers:
        if (
            receipt.dimension is dimension
            and receipt.arm is V2BenchmarkArm.CANDIDATE
        ):
            receipt = replace(receipt, **changes)
        result.append(receipt)
    return tuple(result)


def _pair(producers, dimension):
    baseline = next(
        item
        for item in producers
        if item.dimension is dimension
        and item.arm is V2BenchmarkArm.BASELINE
    )
    candidate = next(
        item
        for item in producers
        if item.dimension is dimension
        and item.arm is V2BenchmarkArm.CANDIDATE
    )
    return baseline, candidate


def test_complete_frozen_population_produces_non_compensating_pareto_vector():
    corpus, producers, ablations = _inputs()
    report = evaluate_v2_self_improvement(corpus, producers, ablations)

    assert tuple(report.pareto_vector) == REQUIRED_V2_OBJECTIVE_DIMENSIONS
    assert {dimension.value for dimension in report.pareto_vector} == {
        "safety",
        "tokens",
        "context-reuse",
        "planning",
        "analysis",
        "cache",
        "validation",
        "task-quality",
        "throughput",
        "persistence",
        "idle-reliability",
        "control",
        "refill",
    }
    assert len(producers) == 2 * len(REQUIRED_V2_OBJECTIVE_DIMENSIONS)
    assert len(ablations) == len(REQUIRED_V2_OBJECTIVE_DIMENSIONS)
    assert report.population_complete
    assert report.pareto_passed
    assert report.passed
    assert report.decision is V2EvaluationDecision.PROVISIONAL
    assert report.evidence_claim_ids == (
        REWARD_RESISTANT_EVALUATION_REQUIREMENT_ID,
    )
    assert all(item.passed for item in report.pareto_vector.values())
    assert all(not item.regressed for item in report.pareto_vector.values())
    assert not any(report.anti_gaming_failures.values())
    assert not report.non_compensable_failures


def test_every_metric_is_recomputed_from_integer_producer_counts():
    corpus, producers, ablations = _inputs()
    report = _evaluate(producers, corpus=corpus, ablations=ablations)
    baseline, candidate = _pair(producers, V2ObjectiveDimension.TOKENS)
    component = report.pareto_vector[V2ObjectiveDimension.TOKENS]

    for name, sample in baseline.metric_samples.items():
        assert component.baseline_values_millionths[name] == (
            sample.numerator * 1_000_000 // sample.denominator
        )
    for name, sample in candidate.metric_samples.items():
        assert component.candidate_values_millionths[name] == (
            sample.numerator * 1_000_000 // sample.denominator
        )
    assert (
        candidate.metric_samples[
            "input-tokens-per-criterion"
        ].numerator
        == sum(
            case.candidate.metrics.provider_input_tokens
            for case in corpus.cases
        )
    )


def test_receipts_and_report_are_compact_replayable_and_body_free():
    corpus, producers, ablations = _inputs()
    report = _evaluate(producers, corpus=corpus, ablations=ablations)

    assert V2ProducerReceipt.from_json(producers[0].to_json()) == producers[0]
    assert V2AblationReceipt.from_json(ablations[0].to_json()) == ablations[0]
    restored = V2SelfEvaluationReport.from_json(
        report.to_json(),
        corpus=corpus,
        producer_receipts=producers,
        ablation_receipts=ablations,
    )
    assert restored == report
    assert verify_v2_self_evaluation_report(
        report, corpus, producers, ablations
    ) == report
    encoded = report.to_json()
    assert len(encoded.encode()) < 1_048_576
    assert all(
        forbidden not in encoded
        for forbidden in (
            '"prompt"',
            '"source_body"',
            '"decoded_output"',
            '"patch"',
            '"artifact_graph"',
            '"reasoning"',
        )
    )

    forged = report.to_dict(include_report_id=True)
    forged["pareto_passed"] = False
    with pytest.raises(V2SelfEvaluationError, match="deterministic"):
        V2SelfEvaluationReport.from_dict(
            forged,
            corpus=corpus,
            producer_receipts=producers,
            ablation_receipts=ablations,
        )
    with pytest.raises(V2SelfEvaluationError, match="duplicate object key"):
        V2ProducerReceipt.from_json('{"schema":"a","schema":"b"}')


def test_bounded_ablations_identify_each_causal_component():
    corpus, producers, ablations = _inputs()
    report = _evaluate(producers, corpus=corpus, ablations=ablations)

    assert len(report.ablations) <= MAX_V2_ABLATIONS
    assert len(report.ablations) == len(REQUIRED_V2_OBJECTIVE_DIMENSIONS)
    assert all(item.causal for item in report.ablations)
    assert all(item.affected_metric_ids for item in report.ablations)
    assert all(report.causal_contributors[dimension.value] for dimension in (
        REQUIRED_V2_OBJECTIVE_DIMENSIONS
    ))

    evaluator = V2SelfImprovementEvaluator()
    with pytest.raises(V2SelfEvaluationError, match="budget"):
        evaluator.evaluate(
            corpus,
            producers,
            tuple(ablations)
            + tuple(
                replace(
                    ablations[0],
                    contributor_id=f"component:extra-{index}@1",
                )
                for index in range(MAX_V2_ABLATIONS)
            ),
        )


@pytest.mark.parametrize(
    ("check", "mutate"),
    (
        (
            "denominator-shift",
            lambda baseline, candidate: {
                "metric_samples": {
                    **candidate.metric_samples,
                    "input-tokens-per-criterion": replace(
                        candidate.metric_samples[
                            "input-tokens-per-criterion"
                        ],
                        denominator=candidate.metric_samples[
                            "input-tokens-per-criterion"
                        ].denominator
                        + 1,
                    ),
                }
            },
        ),
        (
            "omitted-hard-fixture",
            lambda baseline, candidate: {
                "hard_fixture_ids": candidate.hard_fixture_ids[:-1]
            },
        ),
        (
            "metric-substitution",
            lambda baseline, candidate: {
                "metric_samples": {
                    key: value
                    for key, value in candidate.metric_samples.items()
                    if key != "retry-input-tokens-per-task"
                }
            },
        ),
        (
            "duplicated-evidence",
            lambda baseline, candidate: {
                "evidence_ids": (
                    baseline.evidence_ids[0],
                    *candidate.evidence_ids[1:],
                )
            },
        ),
        (
            "cherry-picked-task",
            lambda baseline, candidate: {
                "measured_task_ids": candidate.measured_task_ids[:-1]
            },
        ),
        (
            "cache-warming-leakage",
            lambda baseline, candidate: {
                "cache_states": {
                    **candidate.cache_states,
                    "fixture:supervisor-v2:cold@1": V2CacheState.WARM,
                }
            },
        ),
        (
            "work-outside-window",
            lambda baseline, candidate: {
                "work_started_ms": candidate.window_started_ms - 1
            },
        ),
    ),
)
def test_each_reward_hacking_strategy_is_detected_and_forces_shadow(
    check, mutate
):
    corpus, producers, original_ablations = _inputs()
    baseline, candidate = _pair(producers, V2ObjectiveDimension.TOKENS)
    altered = _replace_candidate(
        producers,
        V2ObjectiveDimension.TOKENS,
        **mutate(baseline, candidate),
    )
    try:
        ablations = build_frozen_v2_ablation_receipts(corpus, altered)
    except KeyError:
        ablations = original_ablations
    report = _evaluate(altered, corpus=corpus, ablations=ablations)

    assert report.anti_gaming_failures[check] == ("tokens",)
    assert not report.passed
    assert report.decision is V2EvaluationDecision.SHADOW
    assert report.evidence_claim_ids == ()


def test_warmup_or_work_before_the_window_is_not_counted_as_an_improvement():
    corpus, producers, _ = _inputs()
    altered = _replace_candidate(
        producers,
        V2ObjectiveDimension.CACHE,
        warmup_started_ms=999,
        warmup_ended_ms=1_100,
        work_started_ms=999,
    )
    report = _evaluate(altered, corpus=corpus)

    assert report.anti_gaming_failures["cache-warming-leakage"] == ("cache",)
    assert report.anti_gaming_failures["work-outside-window"] == ("cache",)
    assert report.decision is V2EvaluationDecision.SHADOW


def test_duplicated_or_missing_producer_and_ablation_populations_fail_closed():
    corpus, producers, ablations = _inputs()

    for bad_producers, bad_ablations in (
        (producers[:-1], ablations),
        (producers + (producers[0],), ablations),
        (producers, ablations[:-1]),
        (producers, ablations + (ablations[0],)),
    ):
        report = _evaluate(
            bad_producers,
            corpus=corpus,
            ablations=bad_ablations,
        )
        assert not report.population_complete
        assert report.decision is V2EvaluationDecision.SHADOW
        assert any(
            value in report.non_compensable_failures
            for value in ("producer-population", "ablation-population")
        )


def test_non_compensable_benchmark_or_producer_failure_cannot_be_offset():
    corpus, producers, _ = _inputs()
    unsafe_corpus = replace_v2_candidate_metrics(
        corpus,
        V2FixtureKind.COLD,
        elapsed_ms=1,
        queue_delay_ms=0,
        stage_latencies=(),
        provider_input_tokens=0,
        provider_output_tokens=0,
        provider_reused_input_tokens=0,
        false_completion_count=1,
    )
    unsafe_producers = build_frozen_v2_producer_receipts(unsafe_corpus)
    unsafe_ablations = build_frozen_v2_ablation_receipts(
        unsafe_corpus, unsafe_producers
    )
    report = _evaluate(
        unsafe_producers,
        corpus=unsafe_corpus,
        ablations=unsafe_ablations,
    )
    assert any(
        failure.startswith("benchmark:authority:")
        for failure in report.non_compensable_failures
    )
    assert report.decision is V2EvaluationDecision.SHADOW

    _, token_candidate = _pair(producers, V2ObjectiveDimension.TOKENS)
    altered = _replace_candidate(
        producers,
        V2ObjectiveDimension.TOKENS,
        non_compensable_failures=("escaped-defect",),
        metric_samples={
            **token_candidate.metric_samples,
            "input-tokens-per-criterion": replace(
                token_candidate.metric_samples["input-tokens-per-criterion"],
                numerator=0,
            ),
        },
    )
    producer_report = _evaluate(altered, corpus=corpus)
    assert (
        "producer:tokens:candidate:escaped-defect"
        in producer_report.non_compensable_failures
    )
    assert producer_report.decision is V2EvaluationDecision.SHADOW


_REGRESSION_METRIC = {
    V2ObjectiveDimension.SAFETY: (
        "unsafe-fixture-rate",
        lambda sample, baseline: replace(sample, numerator=1),
    ),
    V2ObjectiveDimension.TOKENS: (
        "input-tokens-per-criterion",
        lambda sample, baseline: baseline,
    ),
    V2ObjectiveDimension.CONTEXT_REUSE: (
        "stable-prefix-reuse",
        lambda sample, baseline: baseline,
    ),
    V2ObjectiveDimension.PLANNING: (
        "hard-constraint-violation-rate",
        lambda sample, baseline: replace(sample, numerator=1),
    ),
    V2ObjectiveDimension.ANALYSIS: (
        "reuse-or-offload-rate",
        lambda sample, baseline: baseline,
    ),
    V2ObjectiveDimension.CACHE: (
        "warm-exact-reuse-rate",
        lambda sample, baseline: baseline,
    ),
    V2ObjectiveDimension.VALIDATION: (
        "escaped-seeded-defect-rate",
        lambda sample, baseline: replace(sample, numerator=1),
    ),
    V2ObjectiveDimension.TASK_QUALITY: (
        "acceptance-coverage-rate",
        lambda sample, baseline: replace(sample, numerator=0),
    ),
    V2ObjectiveDimension.THROUGHPUT: (
        "accepted-throughput",
        lambda sample, baseline: baseline,
    ),
    V2ObjectiveDimension.PERSISTENCE: (
        "maximum-receipt-bytes",
        lambda sample, baseline: replace(sample, numerator=262_145),
    ),
    V2ObjectiveDimension.IDLE_RELIABILITY: (
        "idle-cpu-milli-percent",
        lambda sample, baseline: replace(sample, numerator=2_001),
    ),
    V2ObjectiveDimension.CONTROL: (
        "surface-conformance-rate",
        lambda sample, baseline: replace(sample, numerator=0),
    ),
    V2ObjectiveDimension.REFILL: (
        "exact-replay-noop-rate",
        lambda sample, baseline: replace(sample, numerator=0),
    ),
}


@pytest.mark.parametrize("dimension", REQUIRED_V2_OBJECTIVE_DIMENSIONS)
def test_each_pareto_component_regression_independently_forces_shadow(
    dimension,
):
    corpus, producers, _ = _inputs()
    baseline, candidate = _pair(producers, dimension)
    metric_name, mutate = _REGRESSION_METRIC[dimension]
    samples = dict(candidate.metric_samples)
    samples[metric_name] = mutate(
        samples[metric_name], baseline.metric_samples[metric_name]
    )
    altered = _replace_candidate(
        producers, dimension, metric_samples=samples
    )
    report = _evaluate(altered, corpus=corpus)

    component = report.pareto_vector[dimension]
    assert not component.passed
    assert metric_name in component.gate_failures
    assert not report.pareto_passed
    assert report.decision is V2EvaluationDecision.SHADOW
    assert report.evidence_claim_ids == ()


def test_exact_fixture_and_task_population_are_bound_to_each_receipt():
    corpus, producers, _ = _inputs()
    expected_fixtures = corpus.fixture_population_ids
    expected_tasks = tuple(
        f"task:supervisor-v2:{kind.value}@1"
        for kind in REQUIRED_V2_FIXTURE_KINDS
    )

    for receipt in producers:
        assert receipt.fixture_population_ids == expected_fixtures
        assert receipt.eligible_task_ids == expected_tasks
        assert receipt.measured_task_ids == expected_tasks
        assert len(receipt.source_receipt_ids) == len(expected_fixtures)
        assert len(receipt.evidence_ids) >= len(expected_fixtures)


def test_metric_claim_cannot_replace_raw_counts_during_decode():
    _, producers, _ = _inputs()
    payload = producers[0].to_dict(include_receipt_id=True)
    metric = next(iter(payload["metric_samples"].values()))
    metric["value_millionths"] += 1

    with pytest.raises(V2SelfEvaluationError, match="producer-count replay"):
        V2ProducerReceipt.from_dict(payload)

    payload = json.loads(producers[0].to_json())
    payload["prompt"] = "hidden work"
    with pytest.raises(V2SelfEvaluationError, match="sensitive bodies"):
        V2ProducerReceipt.from_dict(payload)
