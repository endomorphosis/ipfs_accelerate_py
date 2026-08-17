"""SCH-017: semantic compression benchmark runner and published results.

Validates SemanticStateBenchmark@1:

* exactly 40 measured rows with production_eligible=false;
* median context reduction >= 30 percent without coverage omissions;
* zero stale/simulated admissions and zero controlled false negatives;
* results expose task-type reductions, precision, recall, failures, uncertainty;
* --check recomputes identical deterministic fields excluding wall-clock observations.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state.benchmark import (
    BENCHMARK_INTERFACE,
    EXPECTED_TASK_COUNT,
    MIN_MEDIAN_REDUCTION,
    OBSERVATIONAL_FIELD_NAMES,
    BenchmarkError,
    BenchmarkRunner,
    BenchmarkSummary,
    check_report,
    compare_context_modes,
    deterministic_report_digest,
    load_report,
    measure_selection,
    measure_task,
    render_markdown,
    run_benchmark,
    strip_observational_fields,
    summarize_results,
    write_report,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_JSON = (
    REPO_ROOT / "docs" / "benchmarks" / "semantic_compression_harness_results.json"
)
RESULTS_MD = (
    REPO_ROOT / "docs" / "benchmarks" / "semantic_compression_harness_results.md"
)
RUN_BENCHMARK = REPO_ROOT / "benchmarks" / "semantic_state" / "run_benchmark.py"


@pytest.fixture(scope="module")
def runner() -> BenchmarkRunner:
    return BenchmarkRunner().load()


@pytest.fixture(scope="module")
def report(runner: BenchmarkRunner) -> dict[str, Any]:
    _results, _summary, built = runner.run()
    return built


@pytest.fixture(scope="module")
def results(runner: BenchmarkRunner) -> tuple[Any, ...]:
    rows, _summary, _report = runner.run()
    return rows


def test_interface_and_task_count(report: dict[str, Any]) -> None:
    assert report["interface"] == BENCHMARK_INTERFACE
    assert report["task_count"] == EXPECTED_TASK_COUNT
    assert len(report["results"]) == EXPECTED_TASK_COUNT
    assert report["summary"]["task_count"] == EXPECTED_TASK_COUNT


def test_every_oracle_replay_row_production_ineligible(results: tuple[Any, ...]) -> None:
    assert len(results) == EXPECTED_TASK_COUNT
    for row in results:
        assert row.production_eligible is False
        assert row.model_receipt_emitted is False
        assert row.production_root_advanced is False
        body = row.to_dict()
        assert body["production_eligible"] is False
        assert body["candidate_source"] in {
            "oracle_replay_fixture",
            "controlled_fixture_mutation",
        }
        assert body["production_acceptance"] in {
            "not_applicable",
            "rejected",
            "blocked",
        }


def test_median_reduction_at_least_30_percent_without_coverage_omissions(
    report: dict[str, Any],
) -> None:
    summary = report["summary"]
    assert summary["median_reduction_ratio"] >= MIN_MEDIAN_REDUCTION
    assert summary["coverage_omission_count"] == 0
    assert summary["gates"]["median_reduction_at_least_30_percent"] is True
    assert summary["gates"]["zero_coverage_omissions"] is True
    for row in report["results"]:
        assert row["coverage_satisfied"] is True
        assert row["context"]["coverage_omissions"] == []
        assert row["baseline_tokens"] > row["semantic_tokens"] or row[
            "reduction_ratio"
        ] >= 0


def test_zero_stale_simulated_and_controlled_false_negatives(
    report: dict[str, Any],
) -> None:
    summary = report["summary"]
    assert summary["total_stale_admissions"] == 0
    assert summary["total_simulated_admissions"] == 0
    assert summary["total_false_negatives"] == 0
    assert summary["gates"]["zero_stale_admissions"] is True
    assert summary["gates"]["zero_simulated_admissions"] is True
    assert summary["gates"]["zero_controlled_false_negatives"] is True
    assert summary["overall_recall_bp"] == 10_000
    for row in report["results"]:
        assert row["stale_admissions"] == 0
        assert row["simulated_admissions"] == 0
        assert row["false_negatives"] == []
        assert row["recall_bp"] == 10_000


def test_results_report_reductions_precision_recall_failures_uncertainty(
    report: dict[str, Any],
) -> None:
    summary = report["summary"]
    # Task-type reductions present for all six categories.
    expected_categories = {
        "small_bug_fix",
        "test_repair",
        "api_adapter",
        "schema_migration",
        "multi_file_refactor",
        "rejection_or_escalation",
    }
    assert set(summary["category_counts"]) == expected_categories
    assert set(summary["category_median_reduction"]) == expected_categories
    assert sum(summary["category_counts"].values()) == EXPECTED_TASK_COUNT

    # Precision / recall not hidden.
    assert summary["overall_precision_bp"] is not None
    assert summary["overall_recall_bp"] is not None
    assert "total_false_positives" in summary

    # Failures remain visible (rejection/escalation cohort).
    assert summary["failure_counts"]
    assert summary["verification_outcome_counts"]
    assert "reject" in summary["verification_outcome_counts"] or "escalate" in summary[
        "verification_outcome_counts"
    ]

    # Uncertainty represented rather than dropped.
    assert summary["uncertainty_task_count"] >= 1
    uncertain_rows = [
        row
        for row in report["results"]
        if row["uncertainty"] and row["uncertainty"] != ["none_declared"]
    ]
    assert uncertain_rows

    md = render_markdown(report)
    assert "Reduction by task type" in md
    assert "precision" in md.lower()
    assert "recall" in md.lower()
    assert "Failures" in md
    assert "Uncertainty" in md or "uncertainty" in md


def test_same_tokenizer_estimator_raw_and_semantic(
    runner: BenchmarkRunner, results: tuple[Any, ...]
) -> None:
    for row in results:
        assert row.context.tokenizer_id
        assert row.context.estimator_version
        assert row.context.tokenizer_id == results[0].context.tokenizer_id
        assert row.context.estimator_version == results[0].context.estimator_version


def test_compare_context_modes_hard_coverage(runner: BenchmarkRunner) -> None:
    task = runner.corpus.get_task("sch-bench-01-core-add-body-fix")
    mutation = runner.fixture.get_mutation(task.base_mutation_case_id)
    tree = runner.fixture.mutated_tree(task.base_mutation_case_id)
    comparison = compare_context_modes(task, tree_files=tree, mutation=mutation)
    assert comparison.coverage_satisfied is True
    assert comparison.baseline_tokens > 0
    assert comparison.semantic_tokens > 0
    assert comparison.reduction_ratio >= 0.0
    for path in task.target_paths:
        assert path in comparison.required_exact_paths


def test_measure_selection_zero_false_negatives(runner: BenchmarkRunner) -> None:
    for task in runner.corpus.tasks:
        mutation = runner.fixture.get_mutation(task.base_mutation_case_id)
        metrics = measure_selection(task, mutation)
        assert metrics.false_negatives == ()
        assert metrics.recall_bp == 10_000
        assert set(metrics.oracle_test_node_ids).issubset(
            set(metrics.selected_test_node_ids)
        )


def test_strip_observational_fields_excludes_wall_clock() -> None:
    payload = {
        "task_id": "x",
        "observational_latency_ms": 12.5,
        "stage_latencies_ms": {"a": 1.0},
        "run_wall_clock_ms": 99.0,
        "generated_at_unix_ms": 123,
        "nested": {"latency_ms": 3, "keep": True},
        "results": [{"wall_clock_ms": 1, "value": 2}],
    }
    cleaned = strip_observational_fields(payload)
    assert "observational_latency_ms" not in cleaned
    assert "stage_latencies_ms" not in cleaned
    assert "run_wall_clock_ms" not in cleaned
    assert "generated_at_unix_ms" not in cleaned
    assert cleaned["nested"] == {"keep": True}
    assert cleaned["results"] == [{"value": 2}]
    assert cleaned["task_id"] == "x"
    for name in OBSERVATIONAL_FIELD_NAMES:
        assert name not in json.dumps(cleaned)


def test_deterministic_digest_stable_across_observational_noise(
    report: dict[str, Any],
) -> None:
    mutated = json.loads(json.dumps(report))
    mutated["run_wall_clock_ms"] = report["run_wall_clock_ms"] + 12345.0
    mutated["generated_at_unix_ms"] = report["generated_at_unix_ms"] + 999
    for row in mutated["results"]:
        row["observational_latency_ms"] = row["observational_latency_ms"] + 50.0
        row["stage_latencies_ms"] = {"noise": 1.0}
    assert deterministic_report_digest(report) == deterministic_report_digest(mutated)


def test_run_benchmark_gates_all_pass(report: dict[str, Any]) -> None:
    gates = report["summary"]["gates"]
    assert all(gates.values()), gates


def test_summarize_requires_exactly_40(results: tuple[Any, ...]) -> None:
    with pytest.raises(BenchmarkError):
        summarize_results(results[:5])


def test_write_and_check_roundtrip(tmp_path: Path, report: dict[str, Any]) -> None:
    json_path = tmp_path / "results.json"
    md_path = tmp_path / "results.md"
    write_report(report, json_path=json_path, markdown_path=md_path)
    assert json_path.is_file()
    assert md_path.is_file()
    loaded = load_report(json_path)
    assert loaded["task_count"] == EXPECTED_TASK_COUNT
    envelope = check_report(loaded)
    assert envelope["deterministic_equal"] is True
    assert envelope["gates_ok"] is True


def test_check_detects_semantic_drift(tmp_path: Path, report: dict[str, Any]) -> None:
    drifted = json.loads(json.dumps(report))
    drifted["results"][0]["baseline_tokens"] = (
        int(drifted["results"][0]["baseline_tokens"]) + 1000
    )
    drifted["results"][0]["context"]["baseline_tokens"] = drifted["results"][0][
        "baseline_tokens"
    ]
    json_path = tmp_path / "drifted.json"
    json_path.write_text(
        json.dumps(drifted, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(BenchmarkError, match="deterministic semantic fields differ"):
        check_report(json_path=json_path)


def test_published_results_exist_and_pass_check() -> None:
    assert RESULTS_JSON.is_file(), "published results JSON missing; run with --write"
    assert RESULTS_MD.is_file(), "published results Markdown missing"
    published = load_report(RESULTS_JSON)
    assert published["interface"] == BENCHMARK_INTERFACE
    assert published["task_count"] == EXPECTED_TASK_COUNT
    assert published["summary"]["median_reduction_ratio"] >= MIN_MEDIAN_REDUCTION
    for row in published["results"]:
        assert row["production_eligible"] is False
    envelope = check_report(published)
    assert envelope["deterministic_equal"] is True
    assert envelope["gates_ok"] is True
    md = RESULTS_MD.read_text(encoding="utf-8")
    assert "Reduction by task type" in md
    assert "precision" in md.lower()
    assert "recall" in md.lower()


def test_cli_check_subprocess() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUN_BENCHMARK), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "deterministic fields match" in completed.stdout


def test_measure_task_and_run_benchmark_entrypoints(runner: BenchmarkRunner) -> None:
    task = runner.corpus.tasks[0]
    row = measure_task(task, fixture_repo=runner.fixture)
    assert isinstance(row.task_id, str)
    assert row.production_eligible is False
    rows, summary, _report = runner.run()
    assert isinstance(summary, BenchmarkSummary)
    assert len(rows) == EXPECTED_TASK_COUNT
    built = run_benchmark()
    assert built["task_count"] == EXPECTED_TASK_COUNT
