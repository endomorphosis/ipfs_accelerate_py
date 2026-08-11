"""RPR-019: measure proof-gated retrieval and repair safety.

Deterministic benchmark over the hermetic adversarial fixture corpus.  Asserts:

* every fixture family is evaluated;
* exact code/index/model/translator/toolchain/policy roots are recorded;
* the four release safety floors are absolute zero;
* reports distinguish nomination failure, proof abstention, target error,
  implementation error, and validation failure;
* repeated clean runs produce equivalent metrics and identities.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Load the benchmark module from scripts/ without requiring package install.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "benchmark_contract_repair.py"


def _load_benchmark_module():
    name = "benchmark_contract_repair_rpr019"
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
    "model_root",
    "translator_root",
    "toolchain_root",
    "policy_root",
)

SAFETY_ABSOLUTE_KEYS = (
    "wrong_path_automated_mutation",
    "failed_obligation_override",
    "stale_forged_or_poisoned_authoritative_admission",
    "unsupported_memory_safety_promotion",
)


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
        / "proof_gated_contract_repair"
        / "benchmark"
        / ".gitkeep"
    )
    assert gitkeep.is_file()


def test_benchmark_interface_and_schema(report: dict) -> None:
    assert report["schema"] == bench.BENCHMARK_SCHEMA
    assert report["interface"] == bench.BENCHMARK_INTERFACE
    assert report["task_id"] == "RPR-019"
    assert report["goal_id"] == "RPR-G090"
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
            # Probes inherit sealed roots from the corpus template.
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
            "index_root",
            "proof_root",
        ):
            assert roots[key], (case["fixture_id"], key)
        # Code and index roots must be content identities from the fixture.
        assert case["code_root"].startswith("sha256:")
        assert case["index_root"].startswith("sha256:")
        assert case["code_root"] == roots["code_root"]
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
    for key in SAFETY_ABSOLUTE_KEYS:
        assert absolute[key] == 0, key
    rebuilt = bench.BenchmarkMetrics.from_cases(_cases_from_report_metrics_source())
    assert rebuilt.floors_hold()


def _cases_from_report_metrics_source():
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
                model_root=case["model_root"],
                translator_root=case["translator_root"],
                toolchain_root=case["toolchain_root"],
                policy_root=case["policy_root"],
                outcome_kind=bench.OutcomeKind(case["outcome_kind"]),
                nomination_disposition=case["nomination_disposition"],
                proof_eligible=case["proof_eligible"],
                admitted=case["admitted"],
                automated_write=case["automated_write"],
                nomination_hit_at_k=case["nomination_hit_at_k"],
                proof_eligible_hit_at_k=case["proof_eligible_hit_at_k"],
                target_precise=case["target_precise"],
                rename_equivalence_precise=case["rename_equivalence_precise"],
                repair_success=case["repair_success"],
                stale_poison_rejected=case["stale_poison_rejected"],
                cost_units=case["cost_units"],
                token_units=case["token_units"],
                context_bytes=case["context_bytes"],
                cache_hits=case["cache_hits"],
                cache_lookups=case["cache_lookups"],
                reason_codes=tuple(case["reason_codes"]),
                safety=bench.SafetyCounters(
                    wrong_path_automated_mutation=safety_raw[
                        "wrong_path_automated_mutation"
                    ],
                    failed_obligation_override=safety_raw[
                        "failed_obligation_override"
                    ],
                    stale_forged_or_poisoned_authoritative_admission=safety_raw[
                        "stale_forged_or_poisoned_authoritative_admission"
                    ],
                    unsupported_memory_safety_promotion=safety_raw[
                        "unsupported_memory_safety_promotion"
                    ],
                    admission_attempts=1,
                ),
                nomination_receipt_id=case["nomination_receipt_id"],
            )
        )
    return results


def test_report_distinguishes_all_outcome_kinds(report: dict) -> None:
    kinds = {case["outcome_kind"] for case in report["cases"]}
    required = {kind.value for kind in bench.OutcomeKind}
    assert required.issubset(kinds), f"missing outcome kinds: {required - kinds}"
    assert set(report["outcome_kinds"]) == required
    # Corpus must produce both success and at least one abstention class.
    corpus_kinds = {
        case["outcome_kind"]
        for case in report["cases"]
        if not str(case["fixture_id"]).startswith("probe:")
    }
    assert "success" in corpus_kinds
    assert "nomination_failure" in corpus_kinds or "proof_abstention" in corpus_kinds


def test_adversarial_fixtures_never_admit_or_write(report: dict) -> None:
    abstain_scenarios = {
        "signature_drift",
        "vector_nearest_incompatible_decoy",
        "declaration_without_implementation",
        "multiple_site_abstention",
        "dynamic_dispatch",
        "reflection",
        "ffi",
        "ownership_lifetime_unsupported",
        "stale_roots",
        "read_only_target",
        "dependency_cycle",
        "tombstone",
    }
    by_scenario = {
        case["scenario"]: case
        for case in report["cases"]
        if not str(case["fixture_id"]).startswith("probe:")
    }
    for scenario in abstain_scenarios:
        case = by_scenario[scenario]
        assert case["admitted"] is False, scenario
        assert case["automated_write"] is False, scenario
        assert case["outcome_kind"] in {
            "nomination_failure",
            "proof_abstention",
            "target_error",
        }, (scenario, case["outcome_kind"])


def test_recoverable_fixtures_can_succeed(report: dict) -> None:
    recoverable = {
        "pure_rename",
        "module_move",
        "alias",
        "re_export",
        "registration",
        "adapter_required",
        "unique_new_site",
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
            assert case["outcome_kind"] == "success"
            assert case["automated_write"] is False  # measurement never mutates
            successes += 1
    assert successes >= 1


def test_stale_poison_and_memory_safety_rejected(report: dict) -> None:
    by_scenario = {
        case["scenario"]: case
        for case in report["cases"]
        if not str(case["fixture_id"]).startswith("probe:")
    }
    for scenario in (
        "stale_roots",
        "vector_nearest_incompatible_decoy",
        "tombstone",
        "read_only_target",
    ):
        assert by_scenario[scenario]["stale_poison_rejected"] is True
        assert by_scenario[scenario]["admitted"] is False
    memory = by_scenario["ownership_lifetime_unsupported"]
    assert memory["admitted"] is False
    assert memory["safety"]["unsupported_memory_safety_promotion"] == 0


def test_metrics_include_release_dimensions(metrics: dict) -> None:
    for key in (
        "recall_at_k",
        "proof_eligible_recall_at_k",
        "admitted_target_precision",
        "rename_equivalence_precision",
        "repair_success_rate",
        "stale_poison_rejection_rate",
        "abstention_count",
        "total_cost_units",
        "total_token_units",
        "total_context_bytes",
        "cache_hit_rate",
        "safety_floors",
        "safety_absolute",
        "metrics_id",
        "case_count",
    ):
        assert key in metrics
    assert metrics["case_count"] >= 19  # full corpus
    assert metrics["abstention_count"] >= 1
    assert metrics["total_cost_units"] > 0
    assert metrics["total_token_units"] > 0
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
    # Sealed verification must hold for both.
    assert bench.verify_report(first)
    assert bench.verify_report(second)


def test_report_tamper_is_detected(report: dict) -> None:
    tampered = json.loads(json.dumps(report))
    tampered["metrics"]["recall_at_k"] = report["metrics"]["recall_at_k"] + 1
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
    pure = next(case for case in manifest["cases"] if case["scenario"] == "pure_rename")
    result = bench.evaluate_fixture(pure)
    assert result.family == "rename_and_move"
    assert result.code_root.startswith("sha256:")
    assert result.index_root.startswith("sha256:")
    assert result.model_root
    assert result.translator_root
    assert result.toolchain_root
    assert result.policy_root
    assert result.case_id.startswith("sha256:")
    assert result.safety.wrong_path_automated_mutation == 0

    decoy = next(
        case
        for case in manifest["cases"]
        if case["scenario"] == "vector_nearest_incompatible_decoy"
    )
    decoy_result = bench.evaluate_fixture(decoy, probe_unsafe=True)
    assert decoy_result.admitted is False
    assert decoy_result.stale_poison_rejected is True
    assert decoy_result.outcome_kind in {
        bench.OutcomeKind.NOMINATION_FAILURE,
        bench.OutcomeKind.PROOF_ABSTENTION,
    }


def test_forged_artifact_content_id_is_rejected() -> None:
    manifest = bench.load_fixture_manifest()
    pure = json.loads(json.dumps(next(
        case for case in manifest["cases"] if case["scenario"] == "pure_rename"
    )))
    pure["artifacts"]["source"]["content_id"] = "sha256:" + ("a" * 64)
    with pytest.raises(bench.ContractRepairBenchmarkError, match="forged or stale"):
        bench.evaluate_fixture(pure)


def test_benchmark_metrics_floors_hold_helper(metrics: dict) -> None:
    rebuilt = bench.BenchmarkMetrics(
        case_count=metrics["case_count"],
        family_counts=metrics["family_counts"],
        outcome_counts=metrics["outcome_counts"],
        recall_at_k=metrics["recall_at_k"],
        proof_eligible_recall_at_k=metrics["proof_eligible_recall_at_k"],
        admitted_target_precision=metrics["admitted_target_precision"],
        rename_equivalence_precision=metrics["rename_equivalence_precision"],
        repair_success_rate=metrics["repair_success_rate"],
        stale_poison_rejection_rate=metrics["stale_poison_rejection_rate"],
        abstention_count=metrics["abstention_count"],
        total_cost_units=metrics["total_cost_units"],
        total_token_units=metrics["total_token_units"],
        total_context_bytes=metrics["total_context_bytes"],
        cache_hit_rate=metrics["cache_hit_rate"],
        safety_floors=metrics["safety_floors"],
        safety_absolute=metrics["safety_absolute"],
        recall_k=metrics["recall_k"],
    )
    assert rebuilt.floors_hold()
    broken = bench.BenchmarkMetrics(
        case_count=1,
        family_counts={"rename_and_move": 1},
        outcome_counts={"success": 1},
        recall_at_k=0,
        proof_eligible_recall_at_k=0,
        admitted_target_precision=0,
        rename_equivalence_precision=0,
        repair_success_rate=0,
        stale_poison_rejection_rate=0,
        abstention_count=0,
        total_cost_units=1,
        total_token_units=1,
        total_context_bytes=1,
        cache_hit_rate=0,
        safety_floors={
            key: (1 if key == "wrong_path_automated_mutation_rate" else 0)
            for key in bench.SAFETY_FLOOR_KEYS
        },
        safety_absolute={
            "wrong_path_automated_mutation": 1,
            "failed_obligation_override": 0,
            "stale_forged_or_poisoned_authoritative_admission": 0,
            "unsupported_memory_safety_promotion": 0,
        },
    )
    assert not broken.floors_hold()


def test_contract_repair_benchmark_class_run_matches_helper() -> None:
    via_class = bench.ContractRepairBenchmark().run()
    via_helper = bench.run_benchmark()
    assert via_class["report_id"] == via_helper["report_id"]
    assert via_class["metrics"]["metrics_id"] == via_helper["metrics"]["metrics_id"]
