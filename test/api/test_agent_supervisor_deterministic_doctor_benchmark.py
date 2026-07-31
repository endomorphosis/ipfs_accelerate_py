"""LPR-040: benchmark adversarial no-LLM diagnosis and repair.

Deterministic measurement over the hermetic deterministic-doctor fixture
corpus. Asserts:

* every required positive and adversarial scenario is present;
* every fixture is evaluated twice with identity-equivalent receipts;
* positive cases repair correctly; adversarial cases abstain or roll back;
* safety floors are absolute zero for missed mandatory caller, authority
  promotion, stale proof/CID, out-of-scope/sandbox write, partial
  transaction, rollback failure, nondeterministic render, false fixed point,
  and llm_router/LLM/model-provider calls;
* stage receipts cover diagnose/retrieve/prove/transform/impact/transaction/
  rollback/fixed_point;
* metrics are non-authoritative and reports seal/reverify identically.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.validation import (
    deterministic_doctor_benchmark as bench,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "validation"
    / "deterministic_doctor_benchmark.py"
)
FIXTURE_DIR = (
    REPO_ROOT / "test" / "fixtures" / "agent_supervisor" / "deterministic_doctor"
)
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"
BUILD_MANIFEST = FIXTURE_DIR / "build_manifest.py"

REQUIRED_AST_SYMBOLS = {
    "DeterministicDoctorFixture",
    "DeterministicDoctorBenchmarkPolicy",
    "DeterministicDoctorMetrics",
    "DeterministicDoctorBenchmarkReport",
}

REQUIRED_POSITIVE = {
    "renamed_moved_symbol",
    "import_export_registration",
    "two_to_three_argument_callers",
    "constructor_factory_context_threading",
    "adapter_schema_serializer_manifest_artifact",
}

REQUIRED_ADVERSARIAL = {
    "same_type_wrong_value",
    "vector_collision",
    "kg_omission",
    "constant_embedding_fallback",
    "stale_corrupt_forged_cid_cache",
    "solver_lie_countermodel",
    "incomplete_ast_impact_scc",
    "dynamic_generated_native_ffi_public_schema_cross_root",
    "sandbox_escape",
    "crash_rollback",
    "oscillation",
}

REQUIRED_ROOT_FIELDS = (
    "code_root",
    "graph_root",
    "index_root",
    "model_root",
    "translator_root",
    "toolchain_root",
    "policy_root",
    "cache_root",
)


@pytest.fixture(scope="module")
def report() -> dict:
    return bench.run_benchmark()


@pytest.fixture(scope="module")
def metrics(report: dict) -> dict:
    return report["metrics"]


def test_declared_outputs_exist() -> None:
    assert MODULE_PATH.is_file()
    assert FIXTURE_DIR.is_dir()
    assert MANIFEST_PATH.is_file()
    assert BUILD_MANIFEST.is_file()
    assert Path(__file__).is_file()


def test_ast_symbols_present() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = REQUIRED_AST_SYMBOLS - names
    assert not missing, f"missing AST symbols: {sorted(missing)}"


def test_benchmark_interface_and_schema(report: dict) -> None:
    assert report["schema"] == bench.BENCHMARK_SCHEMA
    assert report["interface"] == bench.BENCHMARK_INTERFACE
    assert report["task_id"] == "LPR-040"
    assert report["goal_id"] == "LPR-G110"
    assert report["corpus_id"] == bench.CORPUS_VERSION
    assert report["authoritative"] is False
    assert report["completion_authoritative"] is False
    assert report["mutation_authorized"] is False
    assert report["metrics_authoritative"] is False
    assert report["metrics"]["metrics_authoritative"] is False
    assert report["service_interface"] == bench.DETERMINISTIC_DOCTOR_SERVICE_INTERFACE
    assert report["snapshot_interface"] == bench.DOCTOR_EVIDENCE_SNAPSHOT_INTERFACE
    assert report["receipt_interface"] == bench.DETERMINISTIC_DOCTOR_RUN_RECEIPT_INTERFACE
    assert report["resource_interface"] == bench.RESOURCE_MEASUREMENT_INTERFACE
    assert bench.verify_report(report)


def test_fixture_manifest_covers_required_scenarios() -> None:
    manifest = bench.load_fixture_manifest()
    scenarios = {case["scenario"] for case in manifest["cases"]}
    assert scenarios == REQUIRED_POSITIVE | REQUIRED_ADVERSARIAL
    assert scenarios == set().union(*bench.FIXTURE_FAMILIES.values())
    for case in manifest["cases"]:
        for role in bench.ARTIFACT_ROLES:
            art = case["artifacts"][role]
            assert art["content_id"].startswith("sha256:")
            recomputed = bench._fixture_content_id(art["content"])
            assert recomputed == art["content_id"], case["id"]


def test_runs_all_fixture_families(report: dict) -> None:
    families = set(report["fixture_families"])
    assert families == set(bench.REQUIRED_FIXTURE_FAMILIES)
    corpus_cases = report["cases"]
    seen = {case["family"] for case in corpus_cases}
    assert seen == set(bench.REQUIRED_FIXTURE_FAMILIES)
    scenarios = {case["scenario"] for case in corpus_cases}
    expected_scenarios = set().union(*bench.FIXTURE_FAMILIES.values())
    assert scenarios == expected_scenarios
    assert len(corpus_cases) == len(expected_scenarios)


def test_positive_and_adversarial_outcomes(report: dict) -> None:
    by_scenario = {case["scenario"]: case for case in report["cases"]}

    for scenario in REQUIRED_POSITIVE:
        case = by_scenario[scenario]
        assert case["repair_success"] is True, scenario
        assert case["outcome_kind"] == "success", scenario
        assert case["disposition"] == "supported", scenario
        assert case["completion"] == "success", scenario
        assert case["abstained"] is False, scenario
        assert case["callers_repaired"] == case["mandatory_callers"], scenario
        assert case["mandatory_callers"] >= 1, scenario
        assert case["analytical_path"] is True, scenario

    arity = by_scenario["two_to_three_argument_callers"]
    assert arity["mandatory_callers"] == 4
    assert set(arity["reason_codes"]) >= {
        "each_two_arg_caller_gets_obligation",
        "compatible_default_does_not_discharge_others",
    }

    for scenario in REQUIRED_ADVERSARIAL:
        case = by_scenario[scenario]
        assert case["repair_success"] is False, scenario
        assert case["completion"] != "success" or case["outcome_kind"] != "success", scenario
        assert case["callers_repaired"] == 0, scenario
        assert case["llm_invoked"] is False, scenario
        assert case["model_provider_called"] is False, scenario

    assert by_scenario["same_type_wrong_value"]["outcome_kind"] == "wrong_value"
    assert by_scenario["vector_collision"]["outcome_kind"] == "retrieval_degraded"
    assert by_scenario["kg_omission"]["outcome_kind"] == "retrieval_degraded"
    assert by_scenario["constant_embedding_fallback"]["outcome_kind"] == "retrieval_degraded"
    assert by_scenario["stale_corrupt_forged_cid_cache"]["outcome_kind"] == "stale_cache"
    assert by_scenario["solver_lie_countermodel"]["outcome_kind"] == "solver_lie"
    assert by_scenario["incomplete_ast_impact_scc"]["outcome_kind"] == "incomplete_impact"
    assert (
        by_scenario["dynamic_generated_native_ffi_public_schema_cross_root"]["outcome_kind"]
        == "open_frontier"
    )
    assert by_scenario["sandbox_escape"]["outcome_kind"] == "sandbox_escape"
    assert by_scenario["crash_rollback"]["outcome_kind"] == "rollback"
    assert by_scenario["crash_rollback"]["completion"] == "rollback"
    assert by_scenario["oscillation"]["outcome_kind"] == "oscillation"


def test_dual_run_exact_roots_and_identity_equivalent_receipts(report: dict) -> None:
    dual = report["dual_run"]
    assert dual["pass_count"] == bench.DUAL_RUN_PASSES
    assert dual["identity_equivalent"] is True
    assert report["metrics"]["dual_run_identity_equivalent"] is True
    assert len(dual["receipts"]) == bench.DUAL_RUN_PASSES
    first = dual["receipts"][0]
    second = dual["receipts"][1]
    assert first["case_ids"] == second["case_ids"]
    assert first["receipt_ids"] == second["receipt_ids"]
    assert len(first["receipt_ids"]) == len(report["cases"])
    assert all(rid.startswith("sha256:") for rid in first["receipt_ids"])


def test_records_exact_authority_roots(report: dict) -> None:
    for case in report["cases"]:
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
            "cache_id",
            "translator_id",
            "toolchain_id",
            "policy_id",
            "code_root",
            "graph_root",
            "index_root",
            "proof_root",
            "cache_root",
        ):
            assert roots[key], (case["fixture_id"], key)
        assert case["code_root"].startswith("sha256:")
        assert case["graph_root"].startswith("sha256:")
        assert case["index_root"].startswith("sha256:")
        assert case["cache_root"].startswith("sha256:")
        assert case["code_root"] == roots["code_root"]
        assert case["graph_root"] == roots["graph_root"]
        assert case["index_root"] == roots["index_root"]
        assert case["cache_root"] == roots["cache_root"]
        assert case["snapshot"]["interface"] == bench.DOCTOR_EVIDENCE_SNAPSHOT_INTERFACE
        assert case["receipt"]["interface"] == bench.DETERMINISTIC_DOCTOR_RUN_RECEIPT_INTERFACE
        assert case["receipt"]["service_interface"] == bench.DETERMINISTIC_DOCTOR_SERVICE_INTERFACE
        assert case["receipt"]["receipt_id"] == case["receipt_id"]
        assert case["receipt"]["llm_invocations"] == 0
        assert case["receipt"]["model_provider_calls"] == 0
        assert case["receipt"]["llm_router_calls"] == 0


def test_stage_receipts_cover_required_stages(report: dict) -> None:
    for case in report["cases"]:
        stages = set(case["stage_receipts"])
        assert stages == set(bench.REQUIRED_STAGES), case["fixture_id"]
        for stage, receipt_id in case["stage_receipts"].items():
            assert receipt_id.startswith("sha256:"), (case["fixture_id"], stage)


def test_safety_floors_are_absolute_zero(metrics: dict) -> None:
    floors = metrics["safety_floors"]
    absolute = metrics["safety_absolute"]
    for key in bench.SAFETY_FLOOR_KEYS:
        assert key in floors
        assert floors[key] == 0, key
    for key in bench.SAFETY_ABSOLUTE_KEYS:
        assert absolute[key] == 0, key
    assert metrics["llm_invocation_count"] == 0
    assert metrics["model_provider_call_count"] == 0
    rebuilt = bench.DeterministicDoctorMetrics.from_cases(
        [bench.evaluate_fixture(case) for case in bench.load_fixture_manifest()["cases"]]
    )
    assert rebuilt.floors_hold()


def test_metrics_cover_efficacy_without_authority(metrics: dict, report: dict) -> None:
    assert metrics["case_count"] == len(REQUIRED_POSITIVE | REQUIRED_ADVERSARIAL)
    assert metrics["repair_success_count"] == len(REQUIRED_POSITIVE)
    assert metrics["abstention_count"] >= len(REQUIRED_ADVERSARIAL)
    assert metrics["analytical_coverage"] == 1_000_000
    assert metrics["all_caller_closure_rate"] == 1_000_000
    assert metrics["diagnosis_hit_rate"] == 1_000_000
    assert metrics["stage_receipt_coverage"] == 1_000_000
    assert metrics["total_stage_cost_units"] > 0
    assert metrics["total_token_units"] > 0
    assert metrics["total_context_bytes"] > 0
    assert metrics["metrics_authoritative"] is False
    # Resource measurements present and finite.
    for case in report["cases"]:
        resources = case["resources"]
        assert resources["interface"] == bench.RESOURCE_MEASUREMENT_INTERFACE
        assert resources["stage_cost_units"] == len(bench.REQUIRED_STAGES)
        assert resources["token_units"] > 0
        assert resources["context_bytes"] > 0
        assert resources["disk_growth_bytes"] == 0


def test_policy_forbids_model_and_mutation() -> None:
    policy = bench.DeterministicDoctorBenchmarkPolicy()
    assert policy.model_invocation_forbidden is True
    assert policy.metrics_authoritative is False
    assert policy.completion_authoritative is False
    assert policy.mutation_authorized is False
    assert policy.dual_run_passes >= 2
    with pytest.raises(bench.DeterministicDoctorBenchmarkError):
        bench.DeterministicDoctorBenchmarkPolicy(model_invocation_forbidden=False)
    with pytest.raises(bench.DeterministicDoctorBenchmarkError):
        bench.DeterministicDoctorBenchmarkPolicy(mutation_authorized=True)
    with pytest.raises(bench.DeterministicDoctorBenchmarkError):
        bench.DeterministicDoctorBenchmarkPolicy(metrics_authoritative=True)


def test_model_route_guards_raise() -> None:
    previous = bench.install_model_route_guards()
    try:
        import llm_router  # type: ignore

        with pytest.raises(Exception):
            llm_router.complete("anything")  # type: ignore[attr-defined]
        with pytest.raises(Exception):
            getattr(llm_router, "chat")()
    finally:
        bench.restore_model_route_guards(previous)


def test_forged_artifact_content_id_rejected() -> None:
    manifest = bench.load_fixture_manifest()
    case = json.loads(json.dumps(manifest["cases"][0]))
    case["artifacts"]["delta"]["content_id"] = (
        "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    )
    with pytest.raises(bench.DeterministicDoctorBenchmarkError):
        bench.evaluate_fixture(case)


def test_repeated_runs_are_identity_equivalent() -> None:
    first = bench.run_benchmark()
    second = bench.run_benchmark()
    assert first["report_id"] == second["report_id"]
    assert first["metrics"]["metrics_id"] == second["metrics"]["metrics_id"]
    assert [c["case_id"] for c in first["cases"]] == [
        c["case_id"] for c in second["cases"]
    ]
    assert [c["receipt_id"] for c in first["cases"]] == [
        c["receipt_id"] for c in second["cases"]
    ]


def test_fixture_dataclass_round_trip() -> None:
    manifest = bench.load_fixture_manifest()
    for raw in manifest["cases"]:
        fixture = bench.DeterministicDoctorFixture.from_manifest_case(raw)
        assert fixture.fixture_id == raw["id"]
        assert fixture.content_id.startswith("sha256:")
        assert fixture.to_dict()["schema"] == bench.FIXTURE_SCHEMA


def test_build_manifest_recipes_match_required_scenarios() -> None:
    # Import the builder without package install.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "dd_build_manifest", BUILD_MANIFEST
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.REQUIRED_SCENARIOS) == REQUIRED_POSITIVE | REQUIRED_ADVERSARIAL
    rebuilt = module.build_manifest()
    on_disk = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert rebuilt["corpus_id"] == on_disk["corpus_id"]
    assert len(rebuilt["cases"]) == len(on_disk["cases"])
    for left, right in zip(rebuilt["cases"], on_disk["cases"]):
        assert left["id"] == right["id"]
        assert left["scenario"] == right["scenario"]
        for role in module.ARTIFACT_ROLES:
            assert (
                left["artifacts"][role]["content_id"]
                == right["artifacts"][role]["content_id"]
            )
