"""RPR-045: measure adversarial transitive-change propagation safety.

Deterministic benchmark over the hermetic change-propagation fixture corpus.
Asserts:

* every fixture family is evaluated with exact code/graph/index/model/
  translator/toolchain/policy roots;
* outcome vocabulary distinguishes delta miss, graph miss, open frontier,
  missed consumer, retrieval miss, proof abstention, wrong value,
  behavior/placement error, plan omission, implementation error, rollback
  error, and false completion;
* propagation and legacy safety floors are absolute zero;
* metrics record impact recall, consumer precision, proof-eligible value
  recall, unique-source precision, abstention, analytical coverage, LLM
  rate/scope escape, plan completeness, SCC rollback, iterations, closure
  success, latency/cache/tokens/context;
* repeated clean runs are identity-equivalent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Load the benchmark module from scripts/ without requiring package install.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "benchmark_change_propagation.py"


def _load_benchmark_module():
    name = "benchmark_change_propagation_rpr045"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_benchmark_module()

REQUIRED_ROOT_FIELDS = (
    "code_root",
    "graph_root",
    "index_root",
    "model_root",
    "translator_root",
    "toolchain_root",
    "policy_root",
)

REQUIRED_OUTCOME_VALUES = {
    "success",
    "delta_miss",
    "graph_miss",
    "open_frontier",
    "missed_consumer",
    "retrieval_miss",
    "proof_abstention",
    "wrong_value",
    "behavior_placement_error",
    "plan_omission",
    "implementation_error",
    "rollback_error",
    "false_completion",
}


@pytest.fixture(scope="module")
def report() -> dict:
    return bench.run_benchmark()


@pytest.fixture(scope="module")
def metrics(report: dict) -> dict:
    return report["metrics"]


def test_script_and_data_placeholders_exist() -> None:
    assert _SCRIPT_PATH.is_file()
    gitkeep = (
        _REPO_ROOT
        / "data"
        / "agent_supervisor"
        / "proof_gated_change_propagation"
        / "benchmark"
        / ".gitkeep"
    )
    assert gitkeep.is_file()


def test_benchmark_interface_and_schema(report: dict) -> None:
    assert report["schema"] == bench.BENCHMARK_SCHEMA
    assert report["interface"] == bench.BENCHMARK_INTERFACE
    assert report["task_id"] == "RPR-045"
    assert report["goal_id"] == "RPR-G220"
    assert report["corpus_id"] == bench.CORPUS_VERSION
    assert report["authoritative"] is False
    assert report["completion_authoritative"] is False
    assert report["mutation_authorized"] is False
    assert bench.verify_report(report)


def test_runs_all_fixture_families(report: dict) -> None:
    families = set(report["fixture_families"])
    assert families == set(bench.REQUIRED_FIXTURE_FAMILIES)
    corpus_cases = [
        case
        for case in report["cases"]
        if not str(case["fixture_id"]).startswith("probe:")
    ]
    seen = {case["family"] for case in corpus_cases}
    assert seen == set(bench.REQUIRED_FIXTURE_FAMILIES)
    scenarios = {case["scenario"] for case in corpus_cases}
    expected_scenarios = set().union(*bench.FIXTURE_FAMILIES.values())
    assert scenarios == expected_scenarios
    assert len(corpus_cases) == len(expected_scenarios)


def test_records_exact_authority_roots(report: dict) -> None:
    for case in report["cases"]:
        if str(case["fixture_id"]).startswith("probe:"):
            for field in REQUIRED_ROOT_FIELDS:
                assert case[field]
            continue
        for field in REQUIRED_ROOT_FIELDS:
            value = case[field]
            assert isinstance(value, str) and value, (case["fixture_id"], field)
        roots = case["roots"]
        for key in (
            "repository_id",
            "forest_id",
            "tree_id",
            "graph_id",
            "index_id",
            "model_id",
            "config_id",
            "translator_id",
            "toolchain_id",
            "policy_id",
            "code_root",
            "graph_root",
            "index_root",
            "proof_root",
        ):
            assert roots[key], (case["fixture_id"], key)
        assert case["code_root"].startswith("sha256:")
        assert case["graph_root"].startswith("sha256:")
        assert case["index_root"].startswith("sha256:")
        assert case["code_root"] == roots["code_root"]
        assert case["graph_root"] == roots["graph_root"]
        assert case["index_root"] == roots["index_root"]
        assert case["model_root"] == roots["model_id"]
        assert case["translator_root"] == roots["translator_id"]
        assert case["toolchain_root"] == roots["toolchain_id"]
        assert case["policy_root"] == roots["policy_id"]


def test_safety_floors_are_absolute_zero(metrics: dict) -> None:
    floors = metrics["safety_floors"]
    absolute = metrics["safety_absolute"]
    for key in bench.SAFETY_FLOOR_KEYS:
        assert key in floors
        assert floors[key] == 0, key
    for key in bench.SAFETY_ABSOLUTE_KEYS:
        assert absolute[key] == 0, key
    rebuilt = bench.ChangePropagationBenchmarkMetrics.from_cases(
        _cases_from_fresh_run()
    )
    assert rebuilt.floors_hold()


def _cases_from_fresh_run():
    """Rebuild CaseResult objects from a fresh benchmark run for floor checks."""

    report = bench.run_benchmark()
    results = []
    for case in report["cases"]:
        safety_raw = case["safety"]
        results.append(
            bench.CaseResult(
                fixture_id=case["fixture_id"],
                scenario=case["scenario"],
                family=case["family"],
                roots=case["roots"],
                code_root=case["code_root"],
                graph_root=case["graph_root"],
                index_root=case["index_root"],
                model_root=case["model_root"],
                translator_root=case["translator_root"],
                toolchain_root=case["toolchain_root"],
                policy_root=case["policy_root"],
                outcome_kind=bench.OutcomeKind(case["outcome_kind"]),
                impact_hit=case["impact_hit"],
                consumer_precise=case["consumer_precise"],
                proof_eligible_value=case["proof_eligible_value"],
                unique_source_precise=case["unique_source_precise"],
                analytical_path=case["analytical_path"],
                llm_invoked=case["llm_invoked"],
                llm_scope_escape=case["llm_scope_escape"],
                plan_complete=case["plan_complete"],
                scc_rollback=case["scc_rollback"],
                fixed_point_iterations=case["fixed_point_iterations"],
                closure_success=case["closure_success"],
                admitted=case["admitted"],
                automated_write=case["automated_write"],
                completion_success=case["completion_success"],
                cost_units=case["cost_units"],
                token_units=case["token_units"],
                context_bytes=case["context_bytes"],
                latency_units=case["latency_units"],
                cache_hits=case["cache_hits"],
                cache_lookups=case["cache_lookups"],
                reason_codes=tuple(case["reason_codes"]),
                safety=bench.SafetyCounters(
                    **{key: safety_raw[key] for key in bench.SAFETY_ABSOLUTE_KEYS},
                    admission_attempts=1,
                    consumer_resolution_attempts=1,
                    value_source_admission_attempts=1,
                    completion_attempts=1,
                    plan_admission_attempts=1,
                ),
                plan_admission=case["plan_admission"],
                value_mapping=case["value_mapping"],
                impact_disposition=case["impact_disposition"],
                completion=case["completion"],
            )
        )
    return results


def test_report_distinguishes_all_outcome_kinds(report: dict) -> None:
    kinds = {case["outcome_kind"] for case in report["cases"]}
    assert REQUIRED_OUTCOME_VALUES.issubset(kinds), (
        f"missing outcome kinds: {REQUIRED_OUTCOME_VALUES - kinds}"
    )
    assert set(report["outcome_kinds"]) == REQUIRED_OUTCOME_VALUES
    corpus_kinds = {
        case["outcome_kind"]
        for case in report["cases"]
        if not str(case["fixture_id"]).startswith("probe:")
    }
    assert "success" in corpus_kinds
    assert "proof_abstention" in corpus_kinds or "wrong_value" in corpus_kinds
    assert "rollback_error" in corpus_kinds or "open_frontier" in corpus_kinds


def test_adversarial_fixtures_never_admit_or_write(report: dict) -> None:
    by_scenario = {
        case["scenario"]: case
        for case in report["cases"]
        if not str(case["fixture_id"]).startswith("probe:")
    }
    for scenario in bench.FAIL_CLOSED_SCENARIOS:
        case = by_scenario[scenario]
        assert case["admitted"] is False, scenario
        assert case["automated_write"] is False, scenario
        assert case["completion_success"] is False, scenario
        assert case["outcome_kind"] != "success", (scenario, case["outcome_kind"])


def test_admissible_fixtures_can_succeed_without_mutation(report: dict) -> None:
    recoverable = {
        "two_to_three_argument_callers",
        "unique_in_scope_value",
        "parameter_threading",
        "config_di_factory_construction",
        "schema_serializer_generated_client",
        "new_class_method_data_structure",
        "stateful_service",
        "dependency_cycle_scc",
    }
    by_scenario = {
        case["scenario"]: case
        for case in report["cases"]
        if not str(case["fixture_id"]).startswith("probe:")
    }
    successes = 0
    for scenario in recoverable:
        case = by_scenario[scenario]
        if case["admitted"]:
            assert case["outcome_kind"] == "success", scenario
            assert case["automated_write"] is False
            successes += 1
    assert successes >= 1

    second = by_scenario["second_order_breaking_delta"]
    assert second["admitted"] is True
    assert second["completion_success"] is False
    assert second["fixed_point_iterations"] >= 1
    assert second["safety"]["false_fixed_point_completion"] == 0


def test_stale_poison_frontier_and_rollback_rejected(report: dict) -> None:
    by_scenario = {
        case["scenario"]: case
        for case in report["cases"]
        if not str(case["fixture_id"]).startswith("probe:")
    }

    stale = by_scenario["stale_graph_vector_proof"]
    assert stale["admitted"] is False
    assert stale["outcome_kind"] == "graph_miss"
    assert stale["safety"]["stale_graph_index_plan_admission"] == 0

    poison = by_scenario["poisoned_retrieval"]
    assert poison["admitted"] is False
    assert poison["outcome_kind"] == "retrieval_miss"

    frontier = by_scenario["reflection_plugin_registry_ffi_frontier"]
    assert frontier["admitted"] is False
    assert frontier["outcome_kind"] == "open_frontier"
    assert frontier["closure_success"] is False

    wrong = by_scenario["same_typed_wrong_information"]
    assert wrong["admitted"] is False
    assert wrong["outcome_kind"] == "wrong_value"
    assert wrong["safety"]["unproved_or_wrong_value_source_admission"] == 0

    partial = by_scenario["partial_transaction"]
    assert partial["admitted"] is False
    assert partial["scc_rollback"] is True
    assert partial["outcome_kind"] == "rollback_error"
    assert partial["safety"]["partial_propagation_completion"] == 0

    llm = by_scenario["llm_scope_escape"]
    assert llm["admitted"] is False
    assert llm["llm_scope_escape"] is False
    assert llm["outcome_kind"] == "behavior_placement_error"


def test_metrics_include_release_dimensions(metrics: dict) -> None:
    for key in (
        "impact_recall",
        "consumer_precision",
        "proof_eligible_value_recall",
        "unique_source_precision",
        "abstention_count",
        "analytical_coverage",
        "llm_rate",
        "llm_scope_escape_rate",
        "plan_completeness",
        "scc_rollback_count",
        "fixed_point_iterations_total",
        "closure_success_rate",
        "completion_success_rate",
        "total_cost_units",
        "total_token_units",
        "total_context_bytes",
        "total_latency_units",
        "cache_hit_rate",
        "safety_floors",
        "safety_absolute",
        "metrics_id",
        "case_count",
    ):
        assert key in metrics
    assert metrics["case_count"] >= 20
    assert metrics["abstention_count"] >= 1
    assert metrics["total_cost_units"] > 0
    assert metrics["total_token_units"] > 0
    assert metrics["total_latency_units"] > 0
    assert metrics["llm_scope_escape_rate"] == 0
    assert metrics["metrics_id"].startswith("sha256:")


def test_repeated_clean_runs_are_equivalent() -> None:
    first = bench.run_benchmark()
    second = bench.run_benchmark()
    assert first["report_id"] == second["report_id"]
    assert first["metrics"]["metrics_id"] == second["metrics"]["metrics_id"]
    assert first["metrics"]["safety_floors"] == second["metrics"]["safety_floors"]
    assert first["metrics"]["safety_absolute"] == second["metrics"]["safety_absolute"]
    first_case_ids = [case["case_id"] for case in first["cases"]]
    second_case_ids = [case["case_id"] for case in second["cases"]]
    assert first_case_ids == second_case_ids
    assert bench.verify_report(first)
    assert bench.verify_report(second)


def test_report_tamper_is_detected(report: dict) -> None:
    tampered = json.loads(json.dumps(report))
    tampered["metrics"]["impact_recall"] = report["metrics"]["impact_recall"] + 1
    assert not bench.verify_report(tampered)
    forged = dict(report)
    forged["report_id"] = "sha256:" + ("0" * 64)
    assert not bench.verify_report(forged)


def test_cli_main_writes_sealed_report(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    rc = bench.main(["--output", str(output), "--recall-k", "5"])
    assert rc == 0
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert bench.verify_report(payload)
    assert all(v == 0 for v in payload["metrics"]["safety_floors"].values())


def test_evaluate_fixture_records_roots_and_outcome() -> None:
    manifest = bench.load_fixture_manifest()
    arity = next(
        case
        for case in manifest["cases"]
        if case["scenario"] == "two_to_three_argument_callers"
    )
    result = bench.evaluate_fixture(arity)
    assert result.family == "arity_and_threading"
    assert result.code_root.startswith("sha256:")
    assert result.graph_root.startswith("sha256:")
    assert result.index_root.startswith("sha256:")
    assert result.model_root
    assert result.translator_root
    assert result.toolchain_root
    assert result.policy_root
    assert result.case_id.startswith("sha256:")
    assert result.safety.missed_resolved_impacted_consumer == 0
    assert result.admitted is True
    assert result.outcome_kind is bench.OutcomeKind.SUCCESS
    assert result.automated_write is False

    wrong = next(
        case
        for case in manifest["cases"]
        if case["scenario"] == "same_typed_wrong_information"
    )
    wrong_result = bench.evaluate_fixture(wrong, probe_unsafe=True)
    assert wrong_result.admitted is False
    assert wrong_result.outcome_kind is bench.OutcomeKind.WRONG_VALUE


def test_forged_artifact_content_id_is_rejected() -> None:
    manifest = bench.load_fixture_manifest()
    pure = json.loads(
        json.dumps(
            next(
                case
                for case in manifest["cases"]
                if case["scenario"] == "unique_in_scope_value"
            )
        )
    )
    pure["artifacts"]["delta"]["content_id"] = "sha256:" + ("a" * 64)
    with pytest.raises(
        bench.ChangePropagationBenchmarkError, match="forged or stale"
    ):
        bench.evaluate_fixture(pure)


def test_benchmark_metrics_floors_hold_helper(metrics: dict) -> None:
    rebuilt = bench.ChangePropagationBenchmarkMetrics(
        case_count=metrics["case_count"],
        family_counts=metrics["family_counts"],
        outcome_counts=metrics["outcome_counts"],
        impact_recall=metrics["impact_recall"],
        consumer_precision=metrics["consumer_precision"],
        proof_eligible_value_recall=metrics["proof_eligible_value_recall"],
        unique_source_precision=metrics["unique_source_precision"],
        abstention_count=metrics["abstention_count"],
        analytical_coverage=metrics["analytical_coverage"],
        llm_rate=metrics["llm_rate"],
        llm_scope_escape_rate=metrics["llm_scope_escape_rate"],
        plan_completeness=metrics["plan_completeness"],
        scc_rollback_count=metrics["scc_rollback_count"],
        fixed_point_iterations_total=metrics["fixed_point_iterations_total"],
        closure_success_rate=metrics["closure_success_rate"],
        completion_success_rate=metrics["completion_success_rate"],
        total_cost_units=metrics["total_cost_units"],
        total_token_units=metrics["total_token_units"],
        total_context_bytes=metrics["total_context_bytes"],
        total_latency_units=metrics["total_latency_units"],
        cache_hit_rate=metrics["cache_hit_rate"],
        safety_floors=metrics["safety_floors"],
        safety_absolute=metrics["safety_absolute"],
        recall_k=metrics["recall_k"],
    )
    assert rebuilt.floors_hold()
    broken = bench.ChangePropagationBenchmarkMetrics(
        case_count=1,
        family_counts={"arity_and_threading": 1},
        outcome_counts={"success": 1},
        impact_recall=0,
        consumer_precision=0,
        proof_eligible_value_recall=0,
        unique_source_precision=0,
        abstention_count=0,
        analytical_coverage=0,
        llm_rate=0,
        llm_scope_escape_rate=0,
        plan_completeness=0,
        scc_rollback_count=0,
        fixed_point_iterations_total=0,
        closure_success_rate=0,
        completion_success_rate=0,
        total_cost_units=1,
        total_token_units=1,
        total_context_bytes=1,
        total_latency_units=1,
        cache_hit_rate=0,
        safety_floors={
            key: (1 if key == "missed_resolved_impacted_consumer_rate" else 0)
            for key in bench.SAFETY_FLOOR_KEYS
        },
        safety_absolute={
            key: (1 if key == "missed_resolved_impacted_consumer" else 0)
            for key in bench.SAFETY_ABSOLUTE_KEYS
        },
    )
    assert not broken.floors_hold()


def test_change_propagation_benchmark_class_run_matches_helper() -> None:
    via_class = bench.ChangePropagationBenchmark().run()
    via_helper = bench.run_benchmark()
    assert via_class["report_id"] == via_helper["report_id"]
    assert via_class["metrics"]["metrics_id"] == via_helper["metrics"]["metrics_id"]
