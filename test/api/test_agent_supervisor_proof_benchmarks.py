from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.proof_metrics import (
    PROOF_BENCHMARK_PHASES,
    ProofBenchmarkThresholds,
    build_proof_benchmark_report,
)


def _sample(mode: str, **changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "sample_id": f"sample:{mode}",
        "mode": mode,
        "raw_context_bytes": 20_000,
        "capsule_context_bytes": 5_000,
        "raw_context_tokens": 5_000,
        "capsule_context_tokens": 1_200,
        "retrieved_items": 10,
        "relevant_retrieved_items": 9,
        "attempted_tasks": 10,
        "accepted_tasks": 9,
        "model_cost": 4.5,
        "wall_time_ms": 9_000,
        "baseline_accepted_tasks_per_second": 1.0,
        "cpu_percent": 75,
        "cpu_time_ms": 6_000,
        "memory_peak_bytes": 256 * 1024 * 1024,
        "requested_workers": 4,
        "worker_limit": 4,
        "peak_workers": 4,
        "nested_workers": 0,
        "cache_lookups": 10,
        "cache_hits": 7,
        "cancelled_work_baseline_ms": 1_000,
        "cancelled_work_actual_ms": 400,
        "single_flight_requests": 10,
        "single_flight_executions": 4,
        "template_count": 10,
        "unsupported_template_count": 1,
        "phase_latencies_ms": {
            "translation": 500,
            "solver": 2_000,
            "kernel": 1_500,
            "cache": 100,
            "model": 1_000,
            "validation": 2_000,
            "merge": 500,
        },
    }
    values.update(changes)
    return values


def test_benchmark_compares_raw_and_capsule_context_and_accepted_task_cost() -> None:
    report = build_proof_benchmark_report(
        (_sample("cold"), _sample("warm"), _sample("parallel")),
        generated_at="2026-07-23T12:00:00Z",
    )
    sample = report["samples"][0]

    assert sample["context_byte_reduction"] == 0.75
    assert sample["context_token_reduction"] == 0.76
    assert sample["retrieval_precision"] == 0.9
    assert sample["accepted_task_cost"] == 0.5
    assert sample["raw_context_bytes"] > sample["capsule_context_bytes"]
    assert report.rollout_expansion_allowed is True
    # Reports are safe telemetry, not retained prompt or proof material.
    rendered = json.dumps(report.to_dict())
    assert report["contains_prompts"] is False
    assert report["contains_proof_transcripts"] is False
    assert "raw_prompt" not in rendered

    incomplete = build_proof_benchmark_report((_sample("cold"),))
    assert incomplete.rollout_expansion_allowed is False
    assert {
        item["reason_code"] for item in incomplete["failures"]
    } >= {"warm_sample_missing", "parallel_sample_missing"}


def test_cold_and_warm_runs_report_every_phase_cpu_memory_and_cache_reuse() -> None:
    report = build_proof_benchmark_report(
        (
            _sample("cold", cache_hits=0),
            _sample(
                "warm",
                wall_time_ms=6_000,
                cache_hits=9,
                phase_latencies_ms={
                    "translation": 300,
                    "solver": 500,
                    "kernel": 500,
                    "cache": 80,
                    "model": 800,
                    "validation": 1_500,
                    "merge": 400,
                },
            ),
            _sample("parallel"),
        )
    )
    cold, warm = report["samples"][:2]

    assert set(cold["phase_latencies_ms"]) == set(PROOF_BENCHMARK_PHASES)
    assert set(warm["phase_latencies_ms"]) == set(PROOF_BENCHMARK_PHASES)
    assert warm["cache_hit_rate"] == 0.9
    assert warm["accepted_tasks_per_second"] > cold["accepted_tasks_per_second"]
    assert warm["cpu_time_ms"] > 0
    assert warm["memory_peak_bytes"] > 0
    assert report.rollout_expansion_allowed is True


def test_parallel_run_detects_oversubscription_and_quantifies_saved_work() -> None:
    passing = build_proof_benchmark_report(
        (_sample("cold"), _sample("warm"), _sample("parallel"))
    )
    sample = passing["samples"][2]

    assert sample["nested_oversubscription"] == 0
    assert sample["cancellation_savings"] == 0.6
    assert sample["single_flight_savings"] == 0.6
    assert passing.rollout_expansion_allowed is True

    failing = build_proof_benchmark_report(
        (
            _sample("cold"),
            _sample("warm"),
            _sample(
                "parallel",
                peak_workers=4,
                nested_workers=2,
                cancelled_work_actual_ms=1_000,
                single_flight_executions=10,
            ),
        )
    )
    reasons = {item["reason_code"] for item in failing["failures"]}
    assert failing.rollout_expansion_allowed is False
    assert "nested_oversubscription_detected" in reasons
    assert "cancellation_savings_below_threshold" in reasons
    assert "single_flight_savings_below_threshold" in reasons


def test_rollout_thresholds_reject_low_value_or_unsupported_templates() -> None:
    strict = ProofBenchmarkThresholds(
        min_context_byte_reduction=0.70,
        min_context_token_reduction=0.70,
        min_retrieval_precision=0.90,
        max_accepted_task_cost=0.60,
        max_unsupported_template_rate=0.10,
    )
    report = build_proof_benchmark_report(
        (
            _sample(
                "cold",
                capsule_context_bytes=10_000,
                capsule_context_tokens=3_000,
                relevant_retrieved_items=7,
                model_cost=9.0,
                unsupported_template_count=4,
            ),
            _sample("warm"),
            _sample("parallel"),
        ),
        thresholds=strict,
    )
    reasons = {item["reason_code"] for item in report["failures"]}

    assert report.rollout_expansion_allowed is False
    assert reasons == {
        "accepted_task_cost_above_threshold",
        "context_byte_reduction_below_threshold",
        "context_token_reduction_below_threshold",
        "retrieval_precision_below_threshold",
        "unsupported_template_rate_above_threshold",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("capsule_context_bytes", 20_001, "bounded context"),
        ("relevant_retrieved_items", 11, "retrieval counts"),
        ("cache_hits", 11, "cache hits"),
        ("single_flight_executions", 11, "single-flight executions"),
    ),
)
def test_invalid_benchmark_measurements_fail_closed(
    field: str, value: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_proof_benchmark_report((_sample("cold", **{field: value}),))
