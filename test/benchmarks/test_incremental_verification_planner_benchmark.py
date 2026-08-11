"""IVP-017: incremental-verification benchmark artifact and harness.

Validates that:

* a freshly generated artifact binds current tree, corpus CID/evaluated count,
  policy, effective environment, commands, measurement schema, and status;
* metrics cover cache hit rate, tests selected/full, ground-truth FN/FP,
  outcome discrepancies, static/proof execution, wall samples, paired or
  estimated reused time, route, frontier escalation, counterexample context,
  and estimator-bound token savings;
* zero stale/simulated acceptance is hard while target misses are recorded
  rather than blocking artifact creation;
* deterministic commitments and old-key historical preservation hold;
* incompatible cross-tree unaffected reuse is explicitly unmet;
* small route appears in at least one and 20% of measured localized fixtures
  or the target is red;
* missing canonical fixtures or real provers are typed unavailable/not_measured.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.verification.evaluation import (
    MeasurementStatus,
    default_fixture_root,
)
from ipfs_accelerate_py.agent_supervisor.verification.model_route import (
    ModelRoute,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_MODULE = (
    REPO_ROOT / "benchmarks" / "agent_supervisor" / "incremental_verification.py"
)


def _load_benchmark_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "ivp_incremental_verification_benchmark",
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
DEFAULT_OUTPUT_RELPATH = _BENCH.DEFAULT_OUTPUT_RELPATH
GOAL_ID = _BENCH.GOAL_ID
SMALL_ROUTE_MIN_FRACTION = _BENCH.SMALL_ROUTE_MIN_FRACTION
TASK_ID = _BENCH.TASK_ID
TOKENIZER_ID = _BENCH.TOKENIZER_ID
artifacts_structurally_equivalent = _BENCH.artifacts_structurally_equivalent
ensure_corpus_manifest = _BENCH.ensure_corpus_manifest
estimate_tokens = _BENCH.estimate_tokens
run_incremental_verification_benchmark = (
    _BENCH.run_incremental_verification_benchmark
)
write_stable_benchmark_artifact = _BENCH.write_stable_benchmark_artifact

ARTIFACT_PATH = REPO_ROOT / DEFAULT_OUTPUT_RELPATH
FIXTURE_ROOT = default_fixture_root(REPO_ROOT)

REQUIRED_TOP_LEVEL = {
    "schema",
    "interface",
    "evidence",
    "task_id",
    "goal_id",
    "authoritative",
    "status",
    "tree_id",
    "corpus",
    "policy",
    "effective_environment",
    "commands",
    "measurement_schema",
    "metrics",
    "targets",
    "target_misses",
    "cases",
    "provers",
    "commitments",
    "historical_preservation",
    "cross_tree_unaffected_reuse",
    "zero_stale_simulated_accepted",
    "content_id",
}

REQUIRED_METRICS = {
    "cache",
    "tests",
    "false_negatives",
    "false_positives",
    "outcome_discrepancies",
    "static_proof_execution",
    "wall_samples",
    "reused_time",
    "routes",
    "frontier_escalation",
    "counterexample_context",
    "token_savings",
}


def _load_artifact() -> dict[str, Any]:
    assert ARTIFACT_PATH.is_file(), f"missing benchmark artifact: {ARTIFACT_PATH}"
    payload = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


# ---------------------------------------------------------------------------
# Module / entry points
# ---------------------------------------------------------------------------


def test_benchmark_module_exists_and_exports_runner() -> None:
    assert BENCHMARK_MODULE.is_file()
    assert callable(run_incremental_verification_benchmark)
    assert BENCHMARK_SCHEMA.endswith("incremental-verification-benchmark@1")
    assert BENCHMARK_INTERFACE == "IncrementalVerificationBenchmark@1"
    assert BENCHMARK_EVIDENCE == "ivp/benchmark@1"
    assert TASK_ID == "IVP-017"
    assert GOAL_ID == "IVP-G090"


def test_artifact_exists_and_binds_identity_surfaces() -> None:
    doc = _load_artifact()
    missing = REQUIRED_TOP_LEVEL - set(doc)
    assert not missing, f"artifact missing keys: {sorted(missing)}"
    assert doc["schema"] == BENCHMARK_SCHEMA
    assert doc["interface"] == BENCHMARK_INTERFACE
    assert doc["evidence"] == BENCHMARK_EVIDENCE
    assert doc["task_id"] == TASK_ID
    assert doc["goal_id"] == GOAL_ID
    assert doc["authoritative"] is False
    assert doc["target_success_asserted"] is False
    assert doc["tree_id"] == _git_head()
    assert doc["status"] in {"green", "red", "yellow", "not_measured"}

    corpus = doc["corpus"]
    assert "corpus_id" in corpus
    assert "evaluated_count" in corpus
    assert "corpus_cid" in corpus or corpus.get("present") is False

    policy = doc["policy"]
    assert policy.get("policy_id")
    assert policy.get("zero_stale_simulated_acceptance_hard") is True

    env = doc["effective_environment"]
    assert env.get("python_version")
    assert env.get("platform")

    commands = doc["commands"]
    assert "generate_artifact" in commands
    assert "validate" in commands

    measurement = doc["measurement_schema"]
    assert measurement.get("version")
    assert TOKENIZER_ID in {
        measurement.get("tokenizer_id"),
        (doc["metrics"]["token_savings"].get("tokenizer_id")),
    }
    for field in (
        "cache_hit_rate",
        "tests_selected_full",
        "ground_truth_false_negatives",
        "ground_truth_false_positives",
        "outcome_discrepancies",
        "static_proof_execution",
        "wall_samples",
        "paired_estimated_reused_time",
        "route",
        "frontier_escalation",
        "counterexample_context",
        "estimator_bound_token_savings",
    ):
        assert field in measurement["fields"]


def test_metrics_cover_required_dimensions() -> None:
    doc = _load_artifact()
    metrics = doc["metrics"]
    missing = REQUIRED_METRICS - set(metrics)
    assert not missing, f"metrics missing: {sorted(missing)}"

    cache = metrics["cache"]
    assert "hit_rate" in cache
    assert 0.0 <= float(cache["hit_rate"]) <= 1.0
    assert cache["zero_stale_simulated_accepted"] is True

    tests = metrics["tests"]
    assert "selected_total" in tests
    assert "full_total" in tests

    assert "ground_truth_total" in metrics["false_negatives"]
    assert "ground_truth_total" in metrics["false_positives"]
    assert "case_count" in metrics["outcome_discrepancies"]

    static_proof = metrics["static_proof_execution"]
    assert "static_checks_executed" in static_proof
    assert "proof_obligations_executed" in static_proof
    assert static_proof["status"] in {
        MeasurementStatus.MEASURED.value,
        MeasurementStatus.NOT_MEASURED.value,
    }

    wall = metrics["wall_samples"]
    assert int(wall["sample_count"]) >= 1 or wall.get("status") == (
        MeasurementStatus.NOT_MEASURED.value
    )
    if int(wall["sample_count"]) >= 1:
        assert "tolerance_ms" in wall
        assert wall["role"] == "observational"
        assert len(wall["samples_ms"]) == int(wall["sample_count"])

    reused = metrics["reused_time"]
    assert reused.get("label") in {"paired", "estimated"}
    paired = reused.get("paired_cache") or {}
    assert paired.get("label") == "paired" or reused.get("label") == "estimated"

    routes = metrics["routes"]
    assert isinstance(routes.get("counts"), dict)
    assert "frontier_escalation_rate" in routes
    assert "rate" in metrics["frontier_escalation"]

    cx = metrics["counterexample_context"]
    assert "total_bytes" in cx
    assert "total_tokens" in cx

    tokens = metrics["token_savings"]
    assert tokens["estimator_bound"] is True
    assert tokens["tokenizer_id"] == TOKENIZER_ID
    assert tokens["tokenizer_version"]
    assert "tokens_saved_total" in tokens
    assert "compared_artifact_bounds" in tokens


def test_zero_stale_simulated_is_hard_and_target_misses_do_not_block() -> None:
    doc = _load_artifact()
    assert doc["zero_stale_simulated_accepted"] is True
    hard = doc["targets"]["zero_stale_simulated_accepted"]
    assert hard["hard"] is True
    assert hard["status"] == "met"
    assert hard["value"] is True

    # Seeded corpus FN makes the soft release target red; artifact still lands.
    assert isinstance(doc["target_misses"], list)
    assert ARTIFACT_PATH.is_file()
    # Creation is never blocked: status is one of the closed vocabulary.
    assert doc["status"] in {"green", "red", "yellow", "not_measured"}
    # Soft FN miss must be recorded when corpus measured a nonzero FN total.
    fn_total = doc["metrics"]["false_negatives"].get("corpus_total")
    if isinstance(fn_total, int) and fn_total > 0:
        assert any(
            item.get("target") == "zero_controlled_false_negatives"
            for item in doc["target_misses"]
        )
        assert doc["targets"]["zero_controlled_false_negatives"]["status"] == "red"


def test_deterministic_commitments_and_historical_preservation() -> None:
    doc = _load_artifact()
    commitments = doc["commitments"]
    assert commitments["deterministic"] is True
    assert commitments.get("commitment_cid")
    assert commitments.get("body", {}).get("tree_id") == doc["tree_id"]

    hist = doc["historical_preservation"]
    assert hist["holds"] is True
    assert hist["old_key_reusable"] is True
    assert hist["historical_present"] is True
    assert doc["targets"]["old_key_historical_preservation"]["status"] == "met"
    assert doc["targets"]["deterministic_commitments"]["status"] == "met"


def test_cross_tree_unaffected_reuse_is_explicitly_unmet() -> None:
    doc = _load_artifact()
    cross = doc["cross_tree_unaffected_reuse"]
    assert cross["status"] == "unmet"
    assert cross["explicitly_unmet"] is True
    assert cross["new_tree_reusable"] is False
    assert "exact_full_tree" in str(cross.get("reason") or "")
    target = doc["targets"]["incompatible_cross_tree_unaffected_reuse"]
    assert target["status"] == "unmet"
    assert target["explicitly_unmet"] is True


def test_small_route_distribution_or_red() -> None:
    doc = _load_artifact()
    cases = doc["cases"]
    measured_localized = [
        case
        for case in cases
        if case.get("localized")
        and case.get("measurement_status") == MeasurementStatus.MEASURED.value
    ]
    small = [
        case
        for case in measured_localized
        if (case.get("route") or {}).get("route")
        == ModelRoute.SMALL_LOCAL_MODEL.value
    ]
    target = doc["targets"]["small_route_localized_distribution"]
    if not measured_localized:
        assert target["status"] == "not_measured"
        return
    fraction = len(small) / len(measured_localized)
    if len(small) >= 1 and fraction >= SMALL_ROUTE_MIN_FRACTION:
        assert target["status"] == "met"
    else:
        assert target["status"] == "red"
        assert any(
            item.get("target") == "small_route_localized_distribution"
            for item in doc["target_misses"]
        )


def test_cases_report_per_fixture_metrics() -> None:
    doc = _load_artifact()
    cases = doc["cases"]
    assert isinstance(cases, list)
    if not cases:
        # Corpus absent path — provers / fixtures typed unavailable.
        assert doc["corpus"].get("present") is False or doc["status"] == (
            "not_measured"
        )
        return

    assert len(cases) == int(doc["corpus"].get("evaluated_count") or len(cases))
    for case in cases:
        assert case.get("fixture_id")
        assert "tests" in case
        assert "selected_count" in case["tests"]
        assert "full_count" in case["tests"]
        assert "false_negatives" in case
        assert "false_positives" in case
        assert "outcome_discrepancies" in case
        assert "static_proof_execution" in case
        assert "wall" in case
        assert case["reused_time"]["label"] in {"paired", "estimated"}
        assert case["route"]["route"]
        assert "frontier_escalation" in case["route"]
        assert "counterexample_context" in case
        tokens = case["token_savings"]
        assert tokens["estimator_bound"] is True
        assert tokens["tokenizer_id"] == TOKENIZER_ID


def test_provers_typed_available_or_unavailable() -> None:
    doc = _load_artifact()
    provers = doc["provers"]
    assert "probes" in provers
    for name, probe in provers["probes"].items():
        assert probe["status"] in {"available", "unavailable"}
        if probe["status"] == "unavailable":
            assert probe["measurement_status"] == MeasurementStatus.NOT_MEASURED.value
        else:
            assert probe["measurement_status"] == MeasurementStatus.MEASURED.value
            assert probe.get("path")
    # Missing provers must appear in unavailable list, never fabricated as green wins.
    for name in provers.get("unavailable") or ():
        assert provers["probes"][name]["status"] == "unavailable"


def test_fresh_run_binds_current_tree_and_is_schema_stable(
    tmp_path: Path,
) -> None:
    out = tmp_path / "benchmark.json"
    artifact = run_incremental_verification_benchmark(
        repo_root_path=REPO_ROOT,
        wall_samples=2,
        output_path=out,
    )
    assert artifact["tree_id"] == _git_head()
    assert artifact["schema"] == BENCHMARK_SCHEMA
    assert artifact["authoritative"] is False
    assert "content_id" in artifact
    # Deterministic commitment body includes tree and corpus.
    body = artifact["commitments"]["body"]
    assert body["tree_id"] == artifact["tree_id"]
    # Re-run yields same commitment for same tree/corpus (timing fields excluded).
    again = run_incremental_verification_benchmark(
        repo_root_path=REPO_ROOT,
        wall_samples=2,
        output_path=out,
    )
    assert again["commitments"]["commitment_cid"] == artifact["commitments"][
        "commitment_cid"
    ]
    assert again["commitments"]["body"]["case_fixture_ids"] == artifact[
        "commitments"
    ]["body"]["case_fixture_ids"]


def test_absent_corpus_is_not_measured_never_zero_fn(tmp_path: Path) -> None:
    empty = tmp_path / "empty_fixtures"
    empty.mkdir()
    artifact = run_incremental_verification_benchmark(
        repo_root_path=REPO_ROOT,
        fixture_root=empty,
        wall_samples=1,
        output_path=tmp_path / "absent.json",
    )
    assert artifact["corpus"]["present"] is False or artifact["corpus"][
        "evaluated_count"
    ] == 0
    # FN totals must not be fabricated as zero when not measured.
    summary = artifact["selection_summary"]
    assert summary["measurement_status"] == MeasurementStatus.NOT_MEASURED.value
    assert summary["total_false_negatives"] is None
    assert summary["total_false_positives"] is None
    assert artifact["authoritative"] is False


def test_estimator_token_savings_bound_to_tokenizer_version() -> None:
    doc = _load_artifact()
    tokens = doc["metrics"]["token_savings"]
    assert tokens["tokenizer_id"] == TOKENIZER_ID
    assert tokens["tokenizer_version"]
    assert tokens["estimator_bound"] is True
    # Estimator is deterministic.
    assert estimate_tokens("abcd") == estimate_tokens("abcd")
    assert estimate_tokens("abcd") > 0


def test_cli_writes_output(tmp_path: Path) -> None:
    out = tmp_path / "cli-benchmark.json"
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    prefix = "ipfs_kit_py:ipfs_datasets_py:."
    env["PYTHONPATH"] = f"{prefix}:{existing}" if existing else prefix
    completed = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK_MODULE),
            "--output",
            str(out),
            "--wall-samples",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == BENCHMARK_SCHEMA
    assert payload["tree_id"] == _git_head()
    # Ephemeral process noise must not appear in the stable artifact.
    assert "pid" not in (payload.get("effective_environment") or {})
    assert "generated_at_unix_ms" not in payload


def test_stable_write_is_fixed_point_across_measured_reruns(tmp_path: Path) -> None:
    """Re-running the generator must not rewrite when only wall samples change.

    Candidate stabilization re-validates once; nonconvergent wall-sample churn
    previously failed post-validation with candidate_stabilization_nonconvergent.
    """

    out = tmp_path / "fixed-point.json"
    first = run_incremental_verification_benchmark(
        repo_root_path=REPO_ROOT,
        wall_samples=2,
        output_path=out,
    )
    written_first, preserved_first = write_stable_benchmark_artifact(out, first)
    assert preserved_first is False
    assert out.is_file()
    first_bytes = out.read_bytes()

    second = run_incremental_verification_benchmark(
        repo_root_path=REPO_ROOT,
        wall_samples=2,
        output_path=out,
    )
    # Measured timings may differ; structural identity must still hold.
    assert artifacts_structurally_equivalent(written_first, second)
    written_second, preserved_second = write_stable_benchmark_artifact(out, second)
    assert preserved_second is True
    assert out.read_bytes() == first_bytes
    assert written_second["content_id"] == written_first["content_id"]


def test_checked_in_artifact_matches_runner_structural_contract() -> None:
    """Checked-in artifact must remain a valid structural projection."""

    doc = _load_artifact()
    # Ensure corpus was evaluable when artifact was generated, or honestly not.
    corpus = doc["corpus"]
    if corpus.get("present"):
        assert int(corpus.get("evaluated_count") or 0) >= 1
        assert corpus.get("corpus_cid")
        assert len(doc["cases"]) == int(corpus["evaluated_count"])
    else:
        assert doc["status"] in {"not_measured", "red", "yellow"}
        assert corpus.get("measurement_status") == (
            MeasurementStatus.NOT_MEASURED.value
        )


def test_ensure_corpus_manifest_reports_status() -> None:
    info = ensure_corpus_manifest(FIXTURE_ROOT)
    assert "present" in info
    assert "corpus_id" in info
    if info["present"]:
        assert info["corpus_cid"]
        assert int(info["case_count"]) >= 1
