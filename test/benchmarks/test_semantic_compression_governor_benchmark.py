"""SCG-045: semantic compression governor benchmark artifact and harness.

Validates that:

* checked-in artifacts report outcome distribution, detection, critical
  acceptance, expansion, reduction, routes, quality, regressions, overhead,
  cost, proposals, and rejections;
* missing evidence is explicit (especially empty live / unavailable sensors);
* simulated and live cohorts are labeled and separated;
* targets are thresholds (not output constants) and never assert production
  authority for controlled fixtures;
* ``--check`` recomputes deterministic fields successfully.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_MODULE = (
    REPO_ROOT / "benchmarks" / "agent_supervisor" / "semantic_compression_governor.py"
)


def _load_benchmark_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "scg_semantic_compression_governor_benchmark",
        BENCHMARK_MODULE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_BENCH = _load_benchmark_module()
BENCHMARK_EVIDENCE = _BENCH.BENCHMARK_EVIDENCE
BENCHMARK_INTERFACE = _BENCH.BENCHMARK_INTERFACE
BENCHMARK_SCHEMA = _BENCH.BENCHMARK_SCHEMA
BENCHMARK_SUMMARY_SCHEMA = _BENCH.BENCHMARK_SUMMARY_SCHEMA
DEFAULT_BENCHMARK_RELPATH = _BENCH.DEFAULT_BENCHMARK_RELPATH
DEFAULT_SUMMARY_RELPATH = _BENCH.DEFAULT_SUMMARY_RELPATH
GOAL_ID = _BENCH.GOAL_ID
TASK_ID = _BENCH.TASK_ID
TOKENIZER_ID = _BENCH.TOKENIZER_ID
artifacts_structurally_equivalent = _BENCH.artifacts_structurally_equivalent
check_benchmark_artifacts = _BENCH.check_benchmark_artifacts
run_semantic_compression_governor_benchmark = (
    _BENCH.run_semantic_compression_governor_benchmark
)
write_benchmark_artifacts = _BENCH.write_benchmark_artifacts

BENCHMARK_PATH = REPO_ROOT / DEFAULT_BENCHMARK_RELPATH
SUMMARY_PATH = REPO_ROOT / DEFAULT_SUMMARY_RELPATH

REQUIRED_SUMMARY_FIELDS = {
    "outcome_distribution",
    "detection",
    "critical_acceptance",
    "expansion",
    "reduction",
    "routes",
    "quality",
    "regressions",
    "overhead",
    "cost",
    "proposals",
    "rejections",
    "missing_evidence",
}

REQUIRED_BENCHMARK_TOP = {
    "schema",
    "interface",
    "evidence",
    "task_id",
    "goal_id",
    "authoritative",
    "status",
    "corpus",
    "metrics",
    "targets",
    "target_misses",
    "cases",
    "missing_evidence",
    "cohorts",
    "content_id",
    "policy",
    "effective_environment",
    "commands",
    "measurement_schema",
}


def _load_json(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"missing artifact: {path}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "ipfs_kit_py:ipfs_datasets_py:."
    # Hermetic launch path: user-site packages (e.g. multiformats) must not be
    # required for --check. Matches agent-supervisor validation environments.
    env["PYTHONNOUSERSITE"] = "1"
    return subprocess.run(
        [sys.executable, str(BENCHMARK_MODULE), *args],
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_benchmark_module_exists_and_exports_runner() -> None:
    assert BENCHMARK_MODULE.is_file()
    assert callable(run_semantic_compression_governor_benchmark)
    assert callable(check_benchmark_artifacts)
    assert BENCHMARK_INTERFACE == "SemanticGovernorBenchmark@1"
    assert BENCHMARK_EVIDENCE == "scg/benchmark-results@1"
    assert TASK_ID == "SCG-045"
    assert GOAL_ID == "SCG-G080"
    assert BENCHMARK_SCHEMA.endswith("semantic-governor-benchmark@1")


# ---------------------------------------------------------------------------
# Checked-in artifacts
# ---------------------------------------------------------------------------


def test_artifacts_exist_and_bind_identity() -> None:
    bench = _load_json(BENCHMARK_PATH)
    summary = _load_json(SUMMARY_PATH)

    missing = REQUIRED_BENCHMARK_TOP - set(bench)
    assert not missing, f"benchmark missing keys: {sorted(missing)}"

    assert bench["schema"] == BENCHMARK_SCHEMA
    assert bench["interface"] == BENCHMARK_INTERFACE
    assert bench["evidence"] == BENCHMARK_EVIDENCE
    assert bench["task_id"] == TASK_ID
    assert bench["goal_id"] == GOAL_ID
    assert bench["authoritative"] is False
    assert bench["target_success_asserted"] is False
    assert bench["production_eligible"] is False
    assert bench["status"] in {"green", "red", "yellow", "not_measured"}
    assert bench["content_id"].startswith("sha256:")

    assert summary["schema"] == BENCHMARK_SUMMARY_SCHEMA
    assert summary["interface"] == BENCHMARK_INTERFACE
    assert summary["task_id"] == TASK_ID
    assert summary["authoritative"] is False
    assert summary["production_eligible"] is False


def test_summary_reports_all_required_measurement_fields() -> None:
    summary = _load_json(SUMMARY_PATH)
    missing = REQUIRED_SUMMARY_FIELDS - set(summary)
    assert not missing, f"summary missing fields: {sorted(missing)}"

    outcomes = summary["outcome_distribution"]
    assert outcomes["total_cases"] == summary["case_count"]
    assert isinstance(outcomes["measured_outcome_counts"], dict)
    assert isinstance(outcomes["comparative_outcome_counts"], dict)
    assert outcomes["total_cases"] > 0

    detection = summary["detection"]
    assert "detected_before_execution_count" in detection
    assert "detection_before_rate_bp" in detection

    critical = summary["critical_acceptance"]
    assert "critical_omission_count" in critical
    assert "critical_omissions_accepted_count" in critical
    assert "critical_acceptance_rate_bp" in critical

    expansion = summary["expansion"]
    assert "expansion_count" in expansion
    assert "expansion_precision_bp" in expansion or expansion.get(
        "expansion_true_positive_count"
    ) is not None

    reduction = summary["reduction"]
    assert "median_context_reduction_bp" in reduction
    assert "raw_tokens_total" in reduction

    routes = summary["routes"]
    assert isinstance(routes.get("route_share_counts"), dict)
    assert "escalation_count" in routes

    quality = summary["quality"]
    assert "accepted_patch_count" in quality
    assert "outcome_counts" in quality

    regressions = summary["regressions"]
    assert "regression_count" in regressions

    overhead = summary["overhead"]
    assert "total_audit_overhead_micros" in overhead
    assert overhead.get("cohort") == "simulated"

    cost = summary["cost"]
    assert "model_spend_micros_total" in cost
    assert "net_savings_micros" in cost
    assert cost.get("cohort") == "simulated"
    assert cost.get("live_cost_evidence") == "missing"

    proposals = summary["proposals"]
    assert "proposed_count" in proposals
    assert "accepted_count" in proposals
    assert "rejected_count" in proposals

    rejections = summary["rejections"]
    assert "stale_present_count" in rejections or "policy_verdict" in rejections


def test_missing_evidence_is_explicit() -> None:
    summary = _load_json(SUMMARY_PATH)
    bench = _load_json(BENCHMARK_PATH)

    missing = summary["missing_evidence"]
    assert isinstance(missing, list)
    assert missing, "missing_evidence must not be empty"
    assert any("live" in str(item) for item in missing)

    assert isinstance(bench["missing_evidence"], list)
    assert bench["missing_evidence"]
    live = bench["cohorts"]["live"]
    assert live["observation_count"] == 0
    assert live.get("quality_claims") is False


def test_metrics_mirror_summary_dimensions() -> None:
    bench = _load_json(BENCHMARK_PATH)
    metrics = bench["metrics"]
    for key in (
        "outcome_distribution",
        "detection",
        "critical_acceptance",
        "expansion",
        "reduction",
        "routes",
        "quality",
        "regressions",
        "overhead",
        "cost",
        "proposals",
        "rejections",
    ):
        assert key in metrics, f"metrics missing {key}"

    collector = metrics["collector_report"]
    assert "live" in collector
    assert "simulated" in collector
    assert collector["live"]["observation_count"] == 0
    assert collector["simulated"]["observation_count"] == len(bench["cases"])
    # Cohort separation: live quality counters must not absorb simulated.
    assert collector["live"]["quality"]["observation_count"] == 0


def test_cases_are_controlled_and_non_production() -> None:
    bench = _load_json(BENCHMARK_PATH)
    cases = bench["cases"]
    assert len(cases) >= 1
    assert all(case.get("production_eligible") is False for case in cases)
    assert all(case.get("cohort") == "simulated" for case in cases)
    assert all(case.get("measurement_status") == "measured" for case in cases)
    partitions = {case["partition"] for case in cases}
    assert "held_out" in partitions
    assert "calibration" in partitions or "development" in partitions


def test_targets_are_thresholds_not_output_constants() -> None:
    bench = _load_json(BENCHMARK_PATH)
    summary = _load_json(SUMMARY_PATH)
    targets = bench["targets"]
    assert targets
    for name, target in targets.items():
        assert "threshold" in target, name
        assert "value" in target, name
        assert "comparator" in target, name
        assert "met" in target, name
        assert target["status"] in {"met", "red", "yellow"}
    assert bench["policy"]["targets_are_thresholds"] is True
    assert bench["policy"]["live_model_quality_claimed"] is False
    # Summary mirrors target evaluation without asserting success constants.
    assert "targets" in summary
    assert summary["status"] == bench["status"]


def test_policy_environment_and_commands_present() -> None:
    bench = _load_json(BENCHMARK_PATH)
    policy = bench["policy"]
    assert policy["policy_id"]
    assert policy["zero_stale_simulated_acceptance_hard"] is True
    assert policy["cohort_separation_required"] is True

    env = bench["effective_environment"]
    assert env.get("python_version")
    assert env.get("platform")
    assert env.get("tokenizer_id") == TOKENIZER_ID

    commands = bench["commands"]
    assert "generate_artifact" in commands
    assert "check" in commands
    assert "validate" in commands

    measurement = bench["measurement_schema"]
    assert measurement["version"]
    assert measurement["tokenizer_id"] == TOKENIZER_ID
    for field in REQUIRED_SUMMARY_FIELDS:
        assert field in measurement["fields"]


def test_zero_stale_simulated_acceptance_flag() -> None:
    bench = _load_json(BENCHMARK_PATH)
    assert bench["zero_stale_simulated_accepted"] is True
    stale_target = bench["targets"]["zero_stale_admissions"]
    assert stale_target["hard"] is True


# ---------------------------------------------------------------------------
# Fresh run / check path
# ---------------------------------------------------------------------------


def test_runner_produces_structurally_equivalent_results() -> None:
    recomputed_bench, recomputed_sum = run_semantic_compression_governor_benchmark(
        repo_root_path=REPO_ROOT
    )
    published_bench = _load_json(BENCHMARK_PATH)
    published_sum = _load_json(SUMMARY_PATH)

    assert artifacts_structurally_equivalent(published_bench, recomputed_bench)
    assert artifacts_structurally_equivalent(published_sum, recomputed_sum)
    assert recomputed_bench["interface"] == BENCHMARK_INTERFACE
    assert recomputed_sum["case_count"] == len(recomputed_bench["cases"])


def test_check_cli_passes_against_published_artifacts() -> None:
    completed = _run_cli("--check")
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "OK:" in completed.stdout
    payload_line = None
    for line in completed.stdout.splitlines():
        if line.strip().startswith("{"):
            payload_line = line
            # Full JSON may be multi-line; parse entire stdout JSON block.
            break
    # Prefer parsing the first JSON object from stdout.
    text = completed.stdout
    start = text.find("{")
    end = text.rfind("}")
    assert start != -1 and end != -1
    envelope = json.loads(text[start : end + 1])
    assert envelope["ok"] is True
    assert envelope["benchmark_match"] is True
    assert envelope["summary_match"] is True
    assert envelope["case_count"] >= 1


def test_check_function_matches_cli() -> None:
    envelope = check_benchmark_artifacts(repo_root_path=REPO_ROOT)
    assert envelope["ok"] is True
    assert envelope["interface"] == BENCHMARK_INTERFACE
    assert isinstance(envelope["missing_evidence"], list)
    assert envelope["missing_evidence"]


def test_critical_acceptance_and_detection_are_honest() -> None:
    summary = _load_json(SUMMARY_PATH)
    critical = summary["critical_acceptance"]
    detection = summary["detection"]
    # Values may be zero / None only when denominators empty; otherwise integers.
    for key in (
        "critical_omission_count",
        "critical_omissions_accepted_count",
    ):
        assert isinstance(critical[key], int)
        assert critical[key] >= 0
    assert isinstance(detection["detected_before_execution_count"], int)
    # Hard gate from plan: zero critical controlled omissions accepted.
    assert critical["critical_omissions_accepted_count"] == 0


def test_proposals_and_rejections_measured() -> None:
    bench = _load_json(BENCHMARK_PATH)
    block = bench["proposals_and_rejections"]
    assert block.get("measurement_status") in {"measured", "not_measured"}
    if block.get("measurement_status") == "measured":
        proposals = block["proposals"]
        assert proposals["proposed_count"] >= 1
        assert proposals["promotion_authorized"] is False
        assert proposals["accepted_count"] + proposals["rejected_count"] == (
            proposals["proposed_count"]
        )
        rejections = block["rejections"]
        assert "stale_present_count" in rejections
        assert rejections.get("policy_verdict")


def test_cost_and_overhead_use_simulated_estimator() -> None:
    summary = _load_json(SUMMARY_PATH)
    assert summary["overhead"]["estimator_id"]
    assert summary["cost"]["estimator_id"]
    assert summary["cost"]["cohort"] == "simulated"
    # Net savings may be non-null for simulated paired estimator.
    net = summary["cost"]["net_savings_micros"]
    assert net is None or isinstance(net, int)


def test_import_is_side_effect_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the module must not write artifacts or touch the network."""

    monkeypatch.chdir(tmp_path)
    # Fresh import under a unique name.
    name = "scg_bench_import_hygiene"
    spec = importlib.util.spec_from_file_location(name, BENCHMARK_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    assert not (tmp_path / "artifacts").exists()
    assert module.BENCHMARK_INTERFACE == BENCHMARK_INTERFACE
