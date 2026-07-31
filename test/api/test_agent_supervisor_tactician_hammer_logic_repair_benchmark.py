"""LPR-019: benchmark adversarial logic prediction and all-caller repair.

Deterministic measurement over the hermetic tactician/hammer logic-repair
fixture corpus.  Asserts:

* every fixture is evaluated twice with exact roots and identity-equivalent
  receipts;
* outcome vocabulary distinguishes static / impact / goal / corpus /
  Tactician / retrieval / lowering / solver / raw-countermodel /
  countermodel-validation / native-goal / reconstruction / admission /
  analytical / provider / transaction / fixed-point failures plus success;
* safety floors are absolute zero for missed caller, unreconstructed or
  raw-countermodel admission, unauthorized axiom, invented behavior, wrong
  value/source/placement, stale root/corpus/receipt, failed-obligation
  override, LLM scope/semantic escape, partial transaction, and false
  completion;
* ordinary generic-provider signature-change overlay and explicit LPR cases
  are included;
* metrics report goal/subgoal and hypothesis precision/recall, premise
  recall@k, first-plan closure, lowering/reconstruction/validated-
  countermodel/abstention/analytical/model/all-caller rates, platform
  enforcement, iterations, p50/p95 time/CPU/memory/context/tokens, and
  cache/invalidation accuracy without making metrics authority;
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
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "benchmark_tactician_hammer_logic_repair.py"


def _load_benchmark_module():
    name = "benchmark_tactician_hammer_logic_repair_lpr019"
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
    "index_root",
    "corpus_root",
    "goal_root",
    "model_root",
    "translator_root",
    "toolchain_root",
    "policy_root",
)

REQUIRED_FAILURE_STAGES = {
    "success",
    "static",
    "impact",
    "goal",
    "corpus",
    "tactician",
    "retrieval",
    "lowering",
    "solver",
    "raw_countermodel",
    "countermodel_validation",
    "native_goal",
    "reconstruction",
    "admission",
    "analytical",
    "provider",
    "transaction",
    "fixed_point",
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
        / "tactician_hammer_logic_repair"
        / "benchmark"
        / ".gitkeep"
    )
    assert gitkeep.is_file()


def test_benchmark_interface_and_schema(report: dict) -> None:
    assert report["schema"] == bench.BENCHMARK_SCHEMA
    assert report["interface"] == bench.BENCHMARK_INTERFACE
    assert report["task_id"] == "LPR-019"
    assert report["goal_id"] == "LPR-G060"
    assert report["corpus_id"] == bench.CORPUS_VERSION
    assert report["authoritative"] is False
    assert report["completion_authoritative"] is False
    assert report["mutation_authorized"] is False
    assert report["metrics_authoritative"] is False
    assert report["metrics"]["metrics_authoritative"] is False
    assert (
        report["fixture_manifest_interface"]
        == bench.LOGIC_REPAIR_FIXTURE_MANIFEST_INTERFACE
    )
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


def test_dual_run_exact_roots_and_identity_equivalent_receipts(report: dict) -> None:
    dual = report["dual_run"]
    assert dual["pass_count"] == bench.DUAL_RUN_PASSES
    assert dual["identity_equivalent"] is True
    assert report["metrics"]["dual_run_identity_equivalent"] is True
    assert len(dual["receipts"]) == 25  # full LPR-004 corpus
    for receipt in dual["receipts"]:
        assert receipt["identity_equivalent"] is True
        assert receipt["pass_count"] == bench.DUAL_RUN_PASSES
        assert receipt["code_root"].startswith("sha256:")
        assert receipt["index_root"].startswith("sha256:")
        assert receipt["corpus_root"].startswith("sha256:")
        assert receipt["goal_root"].startswith("sha256:")
        assert receipt["prediction_receipt_id"].startswith("sha256:")
        assert receipt["countermodel_receipt_id"].startswith("sha256:")
        assert receipt["completion_receipt_id"].startswith("sha256:")
        assert receipt["fixed_point_attachment_id"].startswith("sha256:")


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
            "corpus_id",
            "goal_id",
            "model_id",
            "config_id",
            "translator_id",
            "toolchain_id",
            "policy_id",
            "code_root",
            "index_root",
            "corpus_root",
            "goal_root",
            "proof_root",
            "fixed_point_root",
        ):
            assert roots[key], (case["fixture_id"], key)
        assert case["code_root"].startswith("sha256:")
        assert case["index_root"].startswith("sha256:")
        assert case["corpus_root"].startswith("sha256:")
        assert case["goal_root"].startswith("sha256:")
        assert case["code_root"] == roots["code_root"]
        assert case["index_root"] == roots["index_root"]
        assert case["corpus_root"] == roots["corpus_root"]
        assert case["goal_root"] == roots["goal_root"]
        assert case["model_root"] == roots["model_id"]
        assert case["translator_root"] == roots["translator_id"]
        assert case["toolchain_root"] == roots["toolchain_id"]
        assert case["policy_root"] == roots["policy_id"]
        assert case["interfaces"]["prediction"] == bench.LOGIC_PREDICTION_RECEIPT_INTERFACE
        assert (
            case["interfaces"]["countermodel"]
            == bench.COUNTERMODEL_VALIDATION_RECEIPT_INTERFACE
        )
        assert (
            case["interfaces"]["completion"]
            == bench.PROPAGATION_COMPLETION_RECEIPT_INTERFACE
        )
        assert (
            case["interfaces"]["fixed_point_attachment"]
            == bench.LOGIC_FIXED_POINT_EVIDENCE_ATTACHMENT_INTERFACE
        )


def test_safety_floors_are_absolute_zero(metrics: dict) -> None:
    floors = metrics["safety_floors"]
    absolute = metrics["safety_absolute"]
    for key in bench.SAFETY_FLOOR_KEYS:
        assert key in floors
        assert floors[key] == 0, key
    for key in bench.SAFETY_ABSOLUTE_KEYS:
        assert absolute[key] == 0, key
    rebuilt = bench.LogicRepairBenchmarkMetrics.from_cases(_cases_from_fresh_run())
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
                index_root=case["index_root"],
                corpus_root=case["corpus_root"],
                goal_root=case["goal_root"],
                model_root=case["model_root"],
                translator_root=case["translator_root"],
                toolchain_root=case["toolchain_root"],
                policy_root=case["policy_root"],
                outcome_kind=bench.LogicRepairFailureStage(case["outcome_kind"]),
                failure_stage=bench.LogicRepairFailureStage(case["failure_stage"]),
                goal_hit=case["goal_hit"],
                subgoal_hit=case["subgoal_hit"],
                hypothesis_hit=case["hypothesis_hit"],
                premise_hit_at_k=case["premise_hit_at_k"],
                first_plan_closure=case["first_plan_closure"],
                lowering_ok=case["lowering_ok"],
                reconstruction_ok=case["reconstruction_ok"],
                validated_countermodel=case["validated_countermodel"],
                abstention=case["abstention"],
                analytical_path=case["analytical_path"],
                model_path=case["model_path"],
                all_caller_closure=case["all_caller_closure"],
                platform_enforced=case["platform_enforced"],
                fixed_point_iterations=case["fixed_point_iterations"],
                admitted=case["admitted"],
                automated_write=case["automated_write"],
                completion_success=case["completion_success"],
                cost_units=case["cost_units"],
                token_units=case["token_units"],
                context_bytes=case["context_bytes"],
                latency_units=case["latency_units"],
                cpu_units=case["cpu_units"],
                memory_units=case["memory_units"],
                cache_hits=case["cache_hits"],
                cache_lookups=case["cache_lookups"],
                invalidation_correct=case["invalidation_correct"],
                reason_codes=tuple(case["reason_codes"]),
                safety=bench.SafetyCounters(
                    **{key: safety_raw[key] for key in bench.SAFETY_ABSOLUTE_KEYS},
                    admission_attempts=1,
                    caller_resolution_attempts=1,
                    reconstruction_attempts=1,
                    axiom_admission_attempts=1,
                    behavior_authority_claims=1,
                    value_source_placement_attempts=1,
                    root_receipt_admission_attempts=1,
                    obligation_gate_attempts=1,
                ),
                repair_disposition=case["repair_disposition"],
                proof_disposition=case["proof_disposition"],
                plan_admission=case["plan_admission"],
                completion=case["completion"],
                prediction_receipt_id=case["prediction_receipt_id"],
                countermodel_receipt_id=case["countermodel_receipt_id"],
                completion_receipt_id=case["completion_receipt_id"],
                fixed_point_attachment_id=case["fixed_point_attachment_id"],
                dual_pass_index=case.get("dual_pass_index", 0),
            )
        )
    return results


def test_report_distinguishes_all_failure_stages(report: dict) -> None:
    stages = {case["failure_stage"] for case in report["cases"]}
    assert REQUIRED_FAILURE_STAGES.issubset(stages), (
        f"missing failure stages: {REQUIRED_FAILURE_STAGES - stages}"
    )
    assert set(report["failure_stages"]) == REQUIRED_FAILURE_STAGES
    assert set(report["outcome_kinds"]) == REQUIRED_FAILURE_STAGES
    corpus_stages = {
        case["failure_stage"]
        for case in report["cases"]
        if not str(case["fixture_id"]).startswith("probe:")
    }
    assert "success" in corpus_stages
    assert "provider" in corpus_stages
    assert "raw_countermodel" in corpus_stages or "countermodel_validation" in corpus_stages
    assert "transaction" in corpus_stages
    assert "fixed_point" in corpus_stages


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
        for key in bench.SAFETY_ABSOLUTE_KEYS:
            assert case["safety"][key] == 0, (scenario, key)


def test_admissible_analytical_and_explicit_lpr_cases(report: dict) -> None:
    by_scenario = {
        case["scenario"]: case
        for case in report["cases"]
        if not str(case["fixture_id"]).startswith("probe:")
    }
    successes = 0
    for scenario in bench.ADMITTABLE_ANALYTICAL_SCENARIOS:
        case = by_scenario[scenario]
        assert case["admitted"] is True, scenario
        assert case["outcome_kind"] == "success", scenario
        assert case["analytical_path"] is True, scenario
        assert case["automated_write"] is False
        assert case["completion_success"] is True, scenario
        assert case["all_caller_closure"] is True, scenario
        successes += 1
    assert successes == len(bench.ADMITTABLE_ANALYTICAL_SCENARIOS)

    second = by_scenario["second_order_logic_gap"]
    assert second["admitted"] is True
    assert second["completion_success"] is False
    assert second["fixed_point_iterations"] >= 1
    assert second["failure_stage"] == "fixed_point"
    assert second["safety"]["false_fixed_point_completion"] == 0

    model = by_scenario["model_required_path"]
    assert model["admitted"] is False
    assert model["model_path"] is True
    assert model["failure_stage"] == "analytical"
    assert model["completion"] == "approval_required"


def test_ordinary_generic_provider_overlay_and_adversarial_stages(
    report: dict,
) -> None:
    by_scenario = {
        case["scenario"]: case
        for case in report["cases"]
        if not str(case["fixture_id"]).startswith("probe:")
    }

    overlay = by_scenario["ordinary_generic_provider_overlay"]
    assert overlay["admitted"] is False
    assert overlay["failure_stage"] == "provider"
    assert overlay["outcome_kind"] == "provider"

    raw = by_scenario["raw_malformed_countermodel"]
    assert raw["admitted"] is False
    assert raw["failure_stage"] == "raw_countermodel"
    assert raw["safety"]["unreconstructed_or_raw_countermodel_admission"] == 0

    stale = by_scenario["stale_forged_proof"]
    assert stale["admitted"] is False
    assert stale["failure_stage"] == "reconstruction"
    assert stale["safety"]["stale_root_corpus_receipt_admission"] == 0
    assert stale["invalidation_correct"] is True

    wrong = by_scenario["same_typed_wrong_value"]
    assert wrong["admitted"] is False
    assert wrong["failure_stage"] == "goal"
    assert wrong["validated_countermodel"] is True
    assert wrong["safety"]["wrong_value_source_placement_admission"] == 0

    partial = by_scenario["partial_scc_rollback"]
    assert partial["admitted"] is False
    assert partial["failure_stage"] == "transaction"
    assert partial["safety"]["partial_transaction_completion"] == 0

    missed = by_scenario["passing_tests_missed_caller"]
    assert missed["admitted"] is False
    assert missed["failure_stage"] == "fixed_point"
    assert missed["safety"]["missed_resolved_caller"] == 0

    llm = by_scenario["path_prompt_escape"]
    assert llm["admitted"] is False
    assert llm["failure_stage"] == "admission"
    assert llm["safety"]["llm_scope_semantic_escape"] == 0

    frontier = by_scenario["dynamic_reflection_generated_ffi_lifetime_concurrency"]
    assert frontier["admitted"] is False
    assert frontier["failure_stage"] == "impact"

    timeout = by_scenario["timeout_cancellation"]
    assert timeout["admitted"] is False
    assert timeout["failure_stage"] == "solver"

    poison = by_scenario["vector_kg_comment_poisoning"]
    assert poison["admitted"] is False
    assert poison["failure_stage"] == "retrieval"

    circular = by_scenario["contradictory_circular_premises"]
    assert circular["admitted"] is False
    assert circular["failure_stage"] == "corpus"

    native = by_scenario["wrong_theorem_native_statement_drift"]
    assert native["admitted"] is False
    assert native["failure_stage"] == "native_goal"


def test_metrics_include_release_dimensions(metrics: dict) -> None:
    for key in (
        "goal_precision",
        "goal_recall",
        "subgoal_precision",
        "subgoal_recall",
        "hypothesis_precision",
        "hypothesis_recall",
        "premise_recall_at_k",
        "first_plan_closure_rate",
        "lowering_rate",
        "reconstruction_rate",
        "validated_countermodel_rate",
        "abstention_rate",
        "analytical_rate",
        "model_rate",
        "all_caller_rate",
        "platform_enforcement_rate",
        "fixed_point_iterations_total",
        "completion_success_rate",
        "total_cost_units",
        "total_token_units",
        "total_context_bytes",
        "total_latency_units",
        "total_cpu_units",
        "total_memory_units",
        "p50_latency_units",
        "p95_latency_units",
        "p50_cpu_units",
        "p95_cpu_units",
        "p50_memory_units",
        "p95_memory_units",
        "p50_context_bytes",
        "p95_context_bytes",
        "p50_token_units",
        "p95_token_units",
        "cache_hit_rate",
        "invalidation_accuracy",
        "dual_run_identity_equivalent",
        "safety_floors",
        "safety_absolute",
        "metrics_id",
        "case_count",
        "metrics_authoritative",
        "failure_stage_counts",
    ):
        assert key in metrics
    assert metrics["case_count"] >= 25
    assert metrics["metrics_authoritative"] is False
    assert metrics["total_cost_units"] > 0
    assert metrics["total_token_units"] > 0
    assert metrics["total_latency_units"] > 0
    assert metrics["p50_latency_units"] > 0
    assert metrics["p95_latency_units"] >= metrics["p50_latency_units"]
    assert metrics["p50_cpu_units"] > 0
    assert metrics["p95_cpu_units"] >= metrics["p50_cpu_units"]
    assert metrics["p50_memory_units"] > 0
    assert metrics["p95_memory_units"] >= metrics["p50_memory_units"]
    assert metrics["p50_context_bytes"] > 0
    assert metrics["p95_context_bytes"] >= metrics["p50_context_bytes"]
    assert metrics["p50_token_units"] > 0
    assert metrics["p95_token_units"] >= metrics["p50_token_units"]
    assert metrics["platform_enforcement_rate"] == 1_000_000
    assert metrics["safety_floors"]["llm_scope_semantic_escape_rate"] == 0
    assert metrics["metrics_id"].startswith("sha256:")
    assert metrics["analytical_rate"] > 0
    assert metrics["abstention_rate"] > 0
    assert metrics["all_caller_rate"] > 0
    assert metrics["invalidation_accuracy"] == 1_000_000


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
    assert first["dual_run"]["identity_equivalent"] is True
    assert second["dual_run"]["identity_equivalent"] is True
    assert bench.verify_report(first)
    assert bench.verify_report(second)


def test_report_tamper_is_detected(report: dict) -> None:
    tampered = json.loads(json.dumps(report))
    tampered["metrics"]["goal_recall"] = report["metrics"]["goal_recall"] + 1
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
    assert payload["dual_run"]["identity_equivalent"] is True


def test_evaluate_fixture_records_roots_and_outcome() -> None:
    manifest = bench.load_fixture_manifest()
    local = next(
        case
        for case in manifest["cases"]
        if case["scenario"] == "unique_local_value"
    )
    result = bench.evaluate_fixture(local)
    assert result.family == "arity_and_values"
    assert result.code_root.startswith("sha256:")
    assert result.index_root.startswith("sha256:")
    assert result.corpus_root.startswith("sha256:")
    assert result.goal_root.startswith("sha256:")
    assert result.model_root
    assert result.translator_root
    assert result.toolchain_root
    assert result.policy_root
    assert result.case_id.startswith("sha256:")
    assert result.safety.missed_resolved_caller == 0
    assert result.admitted is True
    assert result.outcome_kind is bench.LogicRepairFailureStage.SUCCESS
    assert result.failure_stage is bench.LogicRepairFailureStage.SUCCESS
    assert result.automated_write is False
    assert result.prediction_receipt_id.startswith("sha256:")
    assert result.countermodel_receipt_id.startswith("sha256:")

    wrong = next(
        case
        for case in manifest["cases"]
        if case["scenario"] == "same_typed_wrong_value"
    )
    wrong_result = bench.evaluate_fixture(wrong, probe_unsafe=True)
    assert wrong_result.admitted is False
    assert wrong_result.outcome_kind is bench.LogicRepairFailureStage.GOAL

    overlay = next(
        case
        for case in manifest["cases"]
        if case["scenario"] == "ordinary_generic_provider_overlay"
    )
    overlay_result = bench.evaluate_fixture(overlay)
    assert overlay_result.admitted is False
    assert overlay_result.failure_stage is bench.LogicRepairFailureStage.PROVIDER


def test_forged_artifact_content_id_is_rejected() -> None:
    manifest = bench.load_fixture_manifest()
    pure = json.loads(
        json.dumps(
            next(
                case
                for case in manifest["cases"]
                if case["scenario"] == "unique_local_value"
            )
        )
    )
    pure["artifacts"]["delta"]["content_id"] = "sha256:" + ("a" * 64)
    with pytest.raises(
        bench.LogicRepairBenchmarkError, match="forged or stale"
    ):
        bench.evaluate_fixture(pure)


def test_benchmark_metrics_floors_hold_helper(metrics: dict) -> None:
    rebuilt = bench.LogicRepairBenchmarkMetrics(
        case_count=metrics["case_count"],
        family_counts=metrics["family_counts"],
        outcome_counts=metrics["outcome_counts"],
        failure_stage_counts=metrics["failure_stage_counts"],
        goal_precision=metrics["goal_precision"],
        goal_recall=metrics["goal_recall"],
        subgoal_precision=metrics["subgoal_precision"],
        subgoal_recall=metrics["subgoal_recall"],
        hypothesis_precision=metrics["hypothesis_precision"],
        hypothesis_recall=metrics["hypothesis_recall"],
        premise_recall_at_k=metrics["premise_recall_at_k"],
        first_plan_closure_rate=metrics["first_plan_closure_rate"],
        lowering_rate=metrics["lowering_rate"],
        reconstruction_rate=metrics["reconstruction_rate"],
        validated_countermodel_rate=metrics["validated_countermodel_rate"],
        abstention_rate=metrics["abstention_rate"],
        analytical_rate=metrics["analytical_rate"],
        model_rate=metrics["model_rate"],
        all_caller_rate=metrics["all_caller_rate"],
        platform_enforcement_rate=metrics["platform_enforcement_rate"],
        fixed_point_iterations_total=metrics["fixed_point_iterations_total"],
        completion_success_rate=metrics["completion_success_rate"],
        total_cost_units=metrics["total_cost_units"],
        total_token_units=metrics["total_token_units"],
        total_context_bytes=metrics["total_context_bytes"],
        total_latency_units=metrics["total_latency_units"],
        total_cpu_units=metrics["total_cpu_units"],
        total_memory_units=metrics["total_memory_units"],
        p50_latency_units=metrics["p50_latency_units"],
        p95_latency_units=metrics["p95_latency_units"],
        p50_cpu_units=metrics["p50_cpu_units"],
        p95_cpu_units=metrics["p95_cpu_units"],
        p50_memory_units=metrics["p50_memory_units"],
        p95_memory_units=metrics["p95_memory_units"],
        p50_context_bytes=metrics["p50_context_bytes"],
        p95_context_bytes=metrics["p95_context_bytes"],
        p50_token_units=metrics["p50_token_units"],
        p95_token_units=metrics["p95_token_units"],
        cache_hit_rate=metrics["cache_hit_rate"],
        invalidation_accuracy=metrics["invalidation_accuracy"],
        dual_run_identity_equivalent=metrics["dual_run_identity_equivalent"],
        safety_floors=metrics["safety_floors"],
        safety_absolute=metrics["safety_absolute"],
        recall_k=metrics["recall_k"],
    )
    assert rebuilt.floors_hold()
    broken = bench.LogicRepairBenchmarkMetrics(
        case_count=1,
        family_counts={"arity_and_values": 1},
        outcome_counts={"success": 1},
        failure_stage_counts={"success": 1},
        goal_precision=0,
        goal_recall=0,
        subgoal_precision=0,
        subgoal_recall=0,
        hypothesis_precision=0,
        hypothesis_recall=0,
        premise_recall_at_k=0,
        first_plan_closure_rate=0,
        lowering_rate=0,
        reconstruction_rate=0,
        validated_countermodel_rate=0,
        abstention_rate=0,
        analytical_rate=0,
        model_rate=0,
        all_caller_rate=0,
        platform_enforcement_rate=0,
        fixed_point_iterations_total=0,
        completion_success_rate=0,
        total_cost_units=1,
        total_token_units=1,
        total_context_bytes=1,
        total_latency_units=1,
        total_cpu_units=1,
        total_memory_units=1,
        p50_latency_units=1,
        p95_latency_units=1,
        p50_cpu_units=1,
        p95_cpu_units=1,
        p50_memory_units=1,
        p95_memory_units=1,
        p50_context_bytes=1,
        p95_context_bytes=1,
        p50_token_units=1,
        p95_token_units=1,
        cache_hit_rate=0,
        invalidation_accuracy=0,
        dual_run_identity_equivalent=True,
        safety_floors={
            key: (1 if key == "missed_resolved_caller_rate" else 0)
            for key in bench.SAFETY_FLOOR_KEYS
        },
        safety_absolute={
            key: (1 if key == "missed_resolved_caller" else 0)
            for key in bench.SAFETY_ABSOLUTE_KEYS
        },
    )
    assert not broken.floors_hold()


def test_logic_repair_benchmark_class_run_matches_helper() -> None:
    via_class = bench.LogicRepairBenchmark().run()
    via_helper = bench.run_benchmark()
    assert via_class["report_id"] == via_helper["report_id"]
    assert via_class["metrics"]["metrics_id"] == via_helper["metrics"]["metrics_id"]


def test_ast_symbols_exported() -> None:
    assert hasattr(bench, "LogicRepairBenchmark")
    assert hasattr(bench, "LogicRepairBenchmarkMetrics")
    assert hasattr(bench, "LogicRepairFailureStage")
    assert bench.BenchmarkMetrics is bench.LogicRepairBenchmarkMetrics
    assert bench.OutcomeKind is bench.LogicRepairFailureStage
