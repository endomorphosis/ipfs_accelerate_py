"""SAWM-024 next-call benchmark and deterministic baselines."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.agent_supervisor.semantic_addressed_world_model.next_call_benchmark import (
    NextCallBenchmarkError,
    VectorRetrievalBaseline,
    run_next_call_benchmark,
    validate_next_call_benchmark_result,
)


FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "semantic_world" / "next_call_cases.json"


def test_metrics_reproduce_from_fixture_with_separate_static_recall() -> None:
    result = run_next_call_benchmark(FIXTURE)
    checked = validate_next_call_benchmark_result(result)
    assert checked["runtime_authority"] is False
    assert 0.0 <= checked["static_recall"] <= 1.0
    ranking = checked["ranking"]
    assert ranking["n"] == 12
    assert "top1" in ranking and "mrr" in ranking
    assert checked["vector_backend"] == "vector_backend_unavailable"
    assert checked["ood_rejection"] > 0


def test_missing_vector_backend_is_typed() -> None:
    with pytest.raises(NextCallBenchmarkError, match="vector_backend_unavailable"):
        VectorRetrievalBaseline(available=False).rank(("mod.resolved.a",))


def test_benchmark_cannot_grant_runtime_authority() -> None:
    result = run_next_call_benchmark(FIXTURE)
    forged = dict(result)
    forged["runtime_authority"] = True
    with pytest.raises(NextCallBenchmarkError, match="no runtime authority"):
        validate_next_call_benchmark_result(forged)
