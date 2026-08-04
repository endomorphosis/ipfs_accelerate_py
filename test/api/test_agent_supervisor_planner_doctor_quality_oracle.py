"""PDR-072: protected hidden quality oracles, adversarial cases, and ablations.

Covers:
* independent oracle binding to benchmark manifest / exact case population
* defect/localization precision-recall, repair success / correct abstention
* acceptance coverage, hidden tests, mutation score
* property / fuzz / differential / metamorphic outcomes
* proof coverage / kernel reconstruction / counterexample validity
* SecurityIR / IntentIR conformance, API/schema compatibility
* blast radius / minimality, flake / post-merge recurrence, exact rollback
* candidate-generated tests/proofs cannot define truth
* adversarial families and one-factor subsystem ablations
* fail-closed mount / incomplete / promotion rules
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.planner_doctor_quality_oracle import (
    ALL_QUALITY_METRICS,
    DOCTOR_QUALITY_METRICS,
    ORACLE_HANDLE,
    PLANNER_DOCTOR_ABLATION_INTERFACE,
    PLANNER_DOCTOR_QUALITY_ORACLE_INTERFACE,
    PLANNER_QUALITY_METRICS,
    PRODUCER_TASK_ID,
    QUALITY_ORACLE_MANIFEST_SCHEMA,
    SOLUTION_QUALITY_METRICS,
    AblationSubsystem,
    AdversarialFamily,
    CandidateArmObservation,
    ExpectedDisposition,
    ObservationDisposition,
    OracleEvaluationDisposition,
    OracleTruthRecipe,
    PlannerDoctorAblation,
    PlannerDoctorQualityOracle,
    QualityOracleError,
    QualityOracleManifest,
    QualityOracleReceipt,
    assert_independent_truth_source,
    build_default_oracle_manifest,
    build_quality_oracle_manifest,
    coverage_millionths,
    create_planner_doctor_quality_oracle,
    default_ablations,
    default_adversarial_cases,
    is_forbidden_truth_source,
    perfect_observation_for_slot,
    ratio_millionths,
    set_precision_recall_millionths,
)

ROOT = Path(__file__).resolve().parents[2]
ORACLE_PATH = (
    ROOT / "test/fixtures/agent_supervisor/planner_doctor_holdout/oracle.manifest.json"
)
BENCHMARK_MANIFEST_PATH = (
    ROOT / "test/fixtures/agent_supervisor/planner_doctor_holdout/manifest.json"
)
BENCHMARK_POLICY_PATH = (
    ROOT / "config/agent_supervisor_planner_doctor_benchmark.json"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        assert key not in result, f"duplicate JSON key: {key}"
        result[key] = value
    return result


@pytest.fixture(scope="module")
def benchmark_manifest() -> dict[str, Any]:
    return json.loads(
        BENCHMARK_MANIFEST_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


@pytest.fixture(scope="module")
def benchmark_policy() -> dict[str, Any]:
    return json.loads(
        BENCHMARK_POLICY_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


@pytest.fixture(scope="module")
def oracle_document() -> dict[str, Any]:
    return json.loads(
        ORACLE_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


@pytest.fixture(scope="module")
def oracle_manifest(oracle_document: dict[str, Any]) -> QualityOracleManifest:
    return QualityOracleManifest.from_dict(oracle_document)


@pytest.fixture(scope="module")
def oracle(oracle_manifest: QualityOracleManifest) -> PlannerDoctorQualityOracle:
    return PlannerDoctorQualityOracle(oracle_manifest)


# ---------------------------------------------------------------------------
# Manifest contract
# ---------------------------------------------------------------------------


def test_oracle_manifest_exists_and_matches_schema(
    oracle_document: dict[str, Any],
    oracle_manifest: QualityOracleManifest,
) -> None:
    assert ORACLE_PATH.is_file()
    assert oracle_document["schema"] == QUALITY_ORACLE_MANIFEST_SCHEMA
    assert oracle_document["interface"] == PLANNER_DOCTOR_QUALITY_ORACLE_INTERFACE
    assert oracle_document["oracle_handle"] == ORACLE_HANDLE
    assert oracle_document["task_id"] == PRODUCER_TASK_ID
    assert oracle_document["oracle_manifest_cid"] == oracle_manifest.content_id
    assert oracle_manifest.interface == PLANNER_DOCTOR_QUALITY_ORACLE_INTERFACE


def test_oracle_binds_benchmark_manifest_and_exact_case_population(
    oracle: PlannerDoctorQualityOracle,
    benchmark_manifest: dict[str, Any],
    benchmark_policy: dict[str, Any],
) -> None:
    oracle.require_benchmark_binding(
        benchmark_manifest_cid=benchmark_manifest["manifest_cid"],
        benchmark_policy_cid=benchmark_policy["policy_cid"],
    )
    case_ids = [case["case_id"] for case in benchmark_manifest["cases"]]
    oracle.require_exact_case_population(case_ids)
    assert len(oracle.manifest.slots) == 12
    assert sum(s.partition == "development" for s in oracle.manifest.slots) == 6
    assert sum(s.partition == "heldout" for s in oracle.manifest.slots) == 6

    # Slot IDs must match the public holdout oracle slots exactly.
    public_slots = {
        case["case_id"]: case["oracle_slot_id"]
        for case in benchmark_manifest["cases"]
    }
    for slot in oracle.manifest.slots:
        assert slot.oracle_slot_id == public_slots[slot.case_id]
        assert slot.case_cid
        assert slot.input_commitment_cid


def test_oracle_binds_implementation_toolchain_and_property_catalog(
    oracle_document: dict[str, Any],
) -> None:
    impl = oracle_document["implementation_binding"]
    toolchain = oracle_document["toolchain_binding"]
    assert impl["implementation_id"]
    assert impl["module"].endswith("planner_doctor_quality_oracle")
    assert impl["interface"] == PLANNER_DOCTOR_QUALITY_ORACLE_INTERFACE
    assert impl["ablation_interface"] == PLANNER_DOCTOR_ABLATION_INTERFACE
    assert impl["producer_task_id"] == PRODUCER_TASK_ID
    assert toolchain["toolchain_manifest_id"]
    assert toolchain["property_catalog_id"]
    assert toolchain["python_target"] == "/usr/bin/python3.12"


def test_oracle_protection_and_mount_contract(
    oracle_document: dict[str, Any],
    benchmark_policy: dict[str, Any],
) -> None:
    protection = oracle_document["protection"]
    policy_oracle = benchmark_policy["quality_oracle"]
    assert protection["operator_owned"] is True
    assert protection["candidate_may_not_read_or_write"] is True
    assert protection["planner_may_not_read_or_write"] is True
    assert protection["candidate_generated_tests_are_not_truth"] is True
    assert protection["candidate_generated_proofs_are_not_truth"] is True
    assert protection["fixture_expected_fields_are_not_oracle_evidence"] is True
    assert protection["missing_unsealed_or_incomplete_disposition"] == (
        "reject-promotion"
    )
    assert protection["mount"] == policy_oracle["mount"]
    assert "termination" in protection["mount_phase"]
    assert policy_oracle["manifest_path"] == (
        "test/fixtures/agent_supervisor/planner_doctor_holdout/oracle.manifest.json"
    )
    assert policy_oracle["oracle_handle"] == ORACLE_HANDLE
    assert policy_oracle["producer_task_id"] == PRODUCER_TASK_ID


def test_forged_oracle_manifest_cid_is_rejected(
    oracle_document: dict[str, Any],
) -> None:
    tampered = copy.deepcopy(oracle_document)
    tampered["oracle_manifest_cid"] = "baguqeera" + "a" * 52
    with pytest.raises(QualityOracleError, match="oracle_manifest_cid"):
        QualityOracleManifest.from_dict(tampered)


# ---------------------------------------------------------------------------
# Metric helpers and independence
# ---------------------------------------------------------------------------


def test_ratio_and_set_metrics() -> None:
    assert ratio_millionths(1, 2) == 500_000
    assert ratio_millionths(0, 0) == 0
    p, r = set_precision_recall_millionths(["a", "b"], ["a", "c"])
    assert p == 500_000
    assert r == 500_000
    assert coverage_millionths(["a", "b"], ["a", "b", "c"]) == 666_666
    p2, r2 = set_precision_recall_millionths([], [])
    assert p2 == 1_000_000 and r2 == 1_000_000


def test_forbidden_truth_sources_fail_closed() -> None:
    for source in (
        "candidate",
        "candidate_generated",
        "self_authored",
        "model",
        "llm",
        "task_status",
        "fixture_expected",
        "retrieval_score",
    ):
        assert is_forbidden_truth_source(source)
        with pytest.raises(QualityOracleError, match="not independent"):
            assert_independent_truth_source(source)
    assert assert_independent_truth_source("operator-sealed-holdout")


def test_truth_recipe_rejects_candidate_source() -> None:
    with pytest.raises(QualityOracleError, match="not independent"):
        OracleTruthRecipe(
            expected_disposition=ExpectedDisposition.SUCCEED,
            truth_source="candidate_generated",
        )


# ---------------------------------------------------------------------------
# Evaluation: perfect and imperfect arms
# ---------------------------------------------------------------------------


def test_perfect_observation_passes_all_slots(
    oracle: PlannerDoctorQualityOracle,
) -> None:
    for slot in oracle.manifest.slots:
        observation = perfect_observation_for_slot(slot)
        receipt = oracle.evaluate(observation)
        assert receipt.oracle_handle == ORACLE_HANDLE
        assert receipt.oracle_manifest_cid == oracle.oracle_manifest_cid
        assert receipt.case_id == slot.case_id
        assert receipt.oracle_slot_id == slot.oracle_slot_id
        assert receipt.promotion_eligible is False
        assert receipt.candidate_tests_used_as_truth is False
        assert receipt.candidate_proofs_used_as_truth is False
        metric_names = {m.metric_name for m in receipt.metrics}
        assert set(ALL_QUALITY_METRICS).issubset(metric_names) or metric_names == set(
            ALL_QUALITY_METRICS
        )
        assert set(PLANNER_QUALITY_METRICS) <= metric_names
        assert set(DOCTOR_QUALITY_METRICS) <= metric_names
        assert set(SOLUTION_QUALITY_METRICS) <= metric_names

        if slot.truth.expected_disposition in {
            ExpectedDisposition.ABSTAIN,
            ExpectedDisposition.DEGRADE,
        }:
            assert receipt.disposition is OracleEvaluationDisposition.ABSTAIN_CORRECT
        else:
            assert receipt.disposition is OracleEvaluationDisposition.PASS

        metrics = receipt.metric_map()
        if slot.truth.hidden_test_ids:
            assert metrics["independent_test_pass_millionths"] == 1_000_000
        if slot.truth.seeded_defect_ids:
            assert metrics["seeded_defect_precision_millionths"] == 1_000_000
            assert metrics["seeded_defect_recall_millionths"] == 1_000_000
            assert metrics["causal_localization_millionths"] == 1_000_000
        if slot.truth.require_exact_rollback:
            assert metrics["rollback_integrity_millionths"] == 1_000_000
        if slot.truth.security_ir_constraint_ids:
            assert metrics["security_ir_conformance_millionths"] == 1_000_000
        if slot.truth.intent_ir_constraint_ids:
            assert metrics["intent_ir_conformance_millionths"] == 1_000_000
        if slot.truth.counterexample_ids:
            assert metrics["counterexample_validity_millionths"] == 1_000_000
        if slot.truth.mutation_operator_ids:
            assert metrics["mutation_score_millionths"] == 1_000_000
        if slot.truth.property_ids:
            assert metrics["property_check_pass_millionths"] == 1_000_000
        if slot.truth.proof_obligation_ids:
            assert metrics["proof_obligation_coverage_millionths"] == 1_000_000
        if slot.truth.kernel_fragment_ids:
            assert metrics["kernel_reconstructed_fraction_millionths"] == 1_000_000
        if slot.truth.api_schema_ids:
            assert metrics["api_schema_compatibility_millionths"] == 1_000_000
        assert metrics["acceptance_coverage_millionths"] == 1_000_000


def test_defect_localization_precision_recall(
    oracle: PlannerDoctorQualityOracle,
) -> None:
    slot = next(
        s for s in oracle.manifest.slots if s.pair_family == "doctor-diagnosis"
        and s.truth.allow_repair
    )
    base = perfect_observation_for_slot(slot)
    # Extra false positive localization + missing one gold defect.
    imperfect = CandidateArmObservation.from_dict(
        {
            **base.to_dict(),
            "predicted_defect_ids": list(slot.truth.seeded_defect_ids)
            + ["defect:spurious"],
            "predicted_localization_targets": list(slot.truth.localization_targets)[
                :1
            ]
            + ["loc:wrong"],
            "schema": base.SCHEMA,
        }
    )
    # Rebuild without schema identity claim issues
    imperfect = CandidateArmObservation(
        case_id=base.case_id,
        arm_id=base.arm_id,
        output_root_cid=base.output_root_cid,
        disposition=base.disposition,
        predicted_defect_ids=tuple(slot.truth.seeded_defect_ids) + ("defect:spurious",),
        predicted_localization_targets=tuple(slot.truth.localization_targets[:1])
        + ("loc:wrong",),
        repaired_defect_ids=base.repaired_defect_ids,
        satisfied_acceptance_ids=base.satisfied_acceptance_ids,
        passed_hidden_test_ids=base.passed_hidden_test_ids,
        killed_mutation_ids=base.killed_mutation_ids,
        passed_property_ids=base.passed_property_ids,
        passed_fuzz_ids=base.passed_fuzz_ids,
        passed_differential_ids=base.passed_differential_ids,
        passed_metamorphic_ids=base.passed_metamorphic_ids,
        discharged_proof_obligation_ids=base.discharged_proof_obligation_ids,
        reconstructed_kernel_fragment_ids=base.reconstructed_kernel_fragment_ids,
        valid_counterexample_ids=base.valid_counterexample_ids,
        satisfied_security_ir_ids=base.satisfied_security_ir_ids,
        satisfied_intent_ir_ids=base.satisfied_intent_ir_ids,
        compatible_api_schema_ids=base.compatible_api_schema_ids,
        predicted_dependency_ids=base.predicted_dependency_ids,
        gold_dependency_ids=base.gold_dependency_ids,
        first_valid_plan=base.first_valid_plan,
        goal_ids_covered=base.goal_ids_covered,
        gold_goal_ids=base.gold_goal_ids,
        blast_radius_changed_lines=base.blast_radius_changed_lines,
        exact_rollback=base.exact_rollback,
        typed_abstention=base.typed_abstention,
        process_tree_terminated=True,
        capabilities_revoked=True,
        output_root_sealed=True,
    )
    receipt = oracle.evaluate(imperfect)
    metrics = receipt.metric_map()
    assert metrics["seeded_defect_precision_millionths"] == 500_000
    assert metrics["seeded_defect_recall_millionths"] == 1_000_000
    assert metrics["causal_localization_millionths"] < 1_000_000


def test_correct_and_incorrect_abstention(
    oracle: PlannerDoctorQualityOracle,
) -> None:
    slot = next(
        s
        for s in oracle.manifest.slots
        if s.truth.require_typed_abstention
        and s.truth.expected_disposition is ExpectedDisposition.ABSTAIN
    )
    good = perfect_observation_for_slot(slot)
    good_receipt = oracle.evaluate(good)
    assert good_receipt.disposition is OracleEvaluationDisposition.ABSTAIN_CORRECT
    assert good_receipt.metric_map()["correct_abstention_millionths"] == 1_000_000

    bad = CandidateArmObservation(
        case_id=slot.case_id,
        arm_id="hybrid-residual-only",
        output_root_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        disposition=ObservationDisposition.SUCCEED,
        predicted_defect_ids=slot.truth.seeded_defect_ids,
        predicted_localization_targets=slot.truth.localization_targets,
        satisfied_acceptance_ids=slot.truth.acceptance_criterion_ids,
        passed_hidden_test_ids=slot.truth.hidden_test_ids,
        process_tree_terminated=True,
        capabilities_revoked=True,
        output_root_sealed=True,
        typed_abstention=False,
    )
    bad_receipt = oracle.evaluate(bad)
    assert bad_receipt.disposition is OracleEvaluationDisposition.ABSTAIN_INCORRECT
    assert bad_receipt.metric_map()["correct_abstention_millionths"] == 0


def test_exact_rollback_required(
    oracle: PlannerDoctorQualityOracle,
) -> None:
    slot = next(
        s for s in oracle.manifest.slots if s.truth.require_exact_rollback
    )
    good = perfect_observation_for_slot(slot)
    assert oracle.evaluate(good).disposition is OracleEvaluationDisposition.PASS

    incomplete = CandidateArmObservation(
        case_id=slot.case_id,
        arm_id="deterministic-symbolic",
        output_root_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        disposition=ObservationDisposition.ROLLBACK,
        predicted_defect_ids=slot.truth.seeded_defect_ids,
        predicted_localization_targets=slot.truth.localization_targets,
        satisfied_acceptance_ids=slot.truth.acceptance_criterion_ids,
        passed_hidden_test_ids=slot.truth.hidden_test_ids,
        exact_rollback=False,
        process_tree_terminated=True,
        capabilities_revoked=True,
        output_root_sealed=True,
    )
    receipt = oracle.evaluate(incomplete)
    assert receipt.disposition is OracleEvaluationDisposition.FAIL
    assert "rollback_incomplete" in receipt.reason_codes
    assert receipt.metric_map()["rollback_integrity_millionths"] == 0


def test_candidate_authored_tests_and_proofs_cannot_define_truth(
    oracle: PlannerDoctorQualityOracle,
) -> None:
    slot = oracle.manifest.slots[0]
    observation = CandidateArmObservation(
        case_id=slot.case_id,
        arm_id="hybrid-residual-only",
        output_root_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        disposition=ObservationDisposition.SUCCEED,
        satisfied_acceptance_ids=slot.truth.acceptance_criterion_ids,
        # Claims to pass gold hidden tests via candidate-authored suite.
        passed_hidden_test_ids=slot.truth.hidden_test_ids,
        discharged_proof_obligation_ids=slot.truth.proof_obligation_ids,
        candidate_authored_test_ids=("test_candidate_self_score",),
        candidate_authored_proof_ids=("proof_candidate_self_score",),
        compatible_api_schema_ids=slot.truth.api_schema_ids,
        passed_property_ids=slot.truth.property_ids,
        process_tree_terminated=True,
        capabilities_revoked=True,
        output_root_sealed=True,
        first_valid_plan=True,
        goal_ids_covered=("goal:primary",),
        gold_goal_ids=("goal:primary",),
    )
    receipt = oracle.evaluate(observation)
    metrics = receipt.metric_map()
    assert metrics["independent_test_pass_millionths"] == 0
    assert metrics["proof_obligation_coverage_millionths"] == 0
    assert "candidate_authored_tests_ignored_as_truth" in receipt.reason_codes
    assert "candidate_authored_proofs_ignored_as_truth" in receipt.reason_codes
    # Receipt itself forbids encoding candidate truth as authority.
    assert receipt.candidate_tests_used_as_truth is False
    assert receipt.candidate_proofs_used_as_truth is False


def test_candidate_supplied_gold_ids_cannot_define_planner_truth(
    oracle: PlannerDoctorQualityOracle,
) -> None:
    slot = oracle.manifest.slots[0]
    observation = CandidateArmObservation(
        case_id=slot.case_id,
        arm_id="adversarial-self-scoring-arm",
        output_root_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        disposition=ObservationDisposition.SUCCEED,
        predicted_dependency_ids=("dep:candidate-chosen",),
        gold_dependency_ids=("dep:candidate-chosen",),
        goal_ids_covered=("goal:candidate-chosen",),
        gold_goal_ids=("goal:candidate-chosen",),
    )

    receipt = oracle.evaluate(observation)
    metrics = receipt.metric_map()

    assert metrics["dependency_precision_millionths"] == 0
    assert metrics["dependency_recall_millionths"] == 0
    assert metrics["goal_coverage_millionths"] == 0
    assert "candidate_gold_dependency_ids_ignored_as_truth" in receipt.reason_codes
    assert "candidate_gold_goal_ids_ignored_as_truth" in receipt.reason_codes
    assert "independent_dependency_gold_unavailable" in receipt.reason_codes
    assert "independent_goal_gold_unavailable" in receipt.reason_codes


def test_absent_gold_never_falls_back_to_candidate_predictions(
    oracle: PlannerDoctorQualityOracle,
) -> None:
    slot = oracle.manifest.slots[0]
    observation = CandidateArmObservation(
        case_id=slot.case_id,
        arm_id="adversarial-missing-gold-arm",
        output_root_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        disposition=ObservationDisposition.SUCCEED,
        predicted_dependency_ids=("dep:prediction",),
        goal_ids_covered=("goal:prediction",),
    )

    receipt = oracle.evaluate(observation)
    metrics = receipt.metric_map()

    assert metrics["dependency_precision_millionths"] == 0
    assert metrics["dependency_recall_millionths"] == 0
    assert metrics["goal_coverage_millionths"] == 0
    assert "independent_dependency_gold_unavailable" in receipt.reason_codes
    assert "independent_goal_gold_unavailable" in receipt.reason_codes


def test_candidate_prediction_error_claims_cannot_score_as_measurements(
    oracle: PlannerDoctorQualityOracle,
) -> None:
    slot = oracle.manifest.slots[0]
    observation = CandidateArmObservation(
        case_id=slot.case_id,
        arm_id="adversarial-zero-error-arm",
        output_root_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        disposition=ObservationDisposition.SUCCEED,
        prediction_error_millionths={
            "critical_path": 0,
            "path": 0,
            "symbol": 0,
            "resource": 0,
            "ready_width": 0,
        },
    )

    receipt = oracle.evaluate(observation)
    metrics = receipt.metric_map()

    for metric_name in (
        "critical_path_prediction_error_millionths",
        "path_prediction_error_millionths",
        "symbol_prediction_error_millionths",
        "resource_prediction_error_millionths",
        "ready_width_error_millionths",
    ):
        assert metrics[metric_name] == 1_000_000
    assert "candidate_prediction_errors_ignored_as_truth" in receipt.reason_codes
    assert "independent_prediction_measurements_unavailable" in receipt.reason_codes

    absent = CandidateArmObservation(
        case_id=slot.case_id,
        arm_id="adversarial-absent-error-arm",
        output_root_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        disposition=ObservationDisposition.SUCCEED,
    )
    absent_metrics = oracle.evaluate(absent).metric_map()
    for metric_name in (
        "critical_path_prediction_error_millionths",
        "path_prediction_error_millionths",
        "symbol_prediction_error_millionths",
        "resource_prediction_error_millionths",
        "ready_width_error_millionths",
    ):
        assert absent_metrics[metric_name] == 1_000_000


def test_receipt_rejects_candidate_truth_flags() -> None:
    with pytest.raises(QualityOracleError, match="cannot define truth"):
        QualityOracleReceipt(
            oracle_handle=ORACLE_HANDLE,
            oracle_manifest_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            case_id="case",
            oracle_slot_id="slot",
            arm_id="arm",
            observation_cid="obs",
            disposition=OracleEvaluationDisposition.PASS,
            metrics=(),
            candidate_tests_used_as_truth=True,
        )


def test_judge_mount_not_ready_rejects_promotion(
    oracle: PlannerDoctorQualityOracle,
) -> None:
    slot = oracle.manifest.slots[0]
    observation = perfect_observation_for_slot(slot)
    unready = CandidateArmObservation(
        case_id=observation.case_id,
        arm_id=observation.arm_id,
        output_root_cid=observation.output_root_cid,
        disposition=observation.disposition,
        process_tree_terminated=False,
        capabilities_revoked=False,
        output_root_sealed=False,
    )
    receipt = oracle.evaluate(unready)
    assert receipt.disposition is OracleEvaluationDisposition.REJECT_PROMOTION
    assert "judge_mount_not_ready" in receipt.reason_codes
    assert receipt.promotion_eligible is False


def test_solution_quality_lanes_property_fuzz_diff_meta(
    oracle: PlannerDoctorQualityOracle,
) -> None:
    slot = next(
        s
        for s in oracle.manifest.slots
        if s.truth.property_ids
        and s.truth.fuzz_check_ids
        and s.truth.differential_check_ids
        and s.truth.metamorphic_check_ids
    )
    perfect = perfect_observation_for_slot(slot)
    perfect_metrics = oracle.evaluate(perfect).metric_map()
    assert perfect_metrics["property_check_pass_millionths"] == 1_000_000
    assert perfect_metrics["fuzz_check_pass_millionths"] == 1_000_000
    assert perfect_metrics["differential_check_pass_millionths"] == 1_000_000
    assert perfect_metrics["metamorphic_check_pass_millionths"] == 1_000_000

    partial = CandidateArmObservation(
        case_id=slot.case_id,
        arm_id="current-mainline-baseline",
        output_root_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        disposition=ObservationDisposition.SUCCEED,
        satisfied_acceptance_ids=slot.truth.acceptance_criterion_ids,
        passed_hidden_test_ids=slot.truth.hidden_test_ids,
        passed_property_ids=slot.truth.property_ids[:1],
        passed_fuzz_ids=(),
        passed_differential_ids=slot.truth.differential_check_ids,
        passed_metamorphic_ids=(),
        discharged_proof_obligation_ids=slot.truth.proof_obligation_ids,
        reconstructed_kernel_fragment_ids=slot.truth.kernel_fragment_ids,
        compatible_api_schema_ids=slot.truth.api_schema_ids,
        predicted_defect_ids=slot.truth.seeded_defect_ids,
        predicted_localization_targets=slot.truth.localization_targets,
        repaired_defect_ids=slot.truth.seeded_defect_ids if slot.truth.allow_repair else (),
        first_valid_plan=True,
        process_tree_terminated=True,
        capabilities_revoked=True,
        output_root_sealed=True,
    )
    metrics = oracle.evaluate(partial).metric_map()
    assert metrics["property_check_pass_millionths"] == 1_000_000
    assert metrics["fuzz_check_pass_millionths"] == 0
    assert metrics["differential_check_pass_millionths"] == 1_000_000
    assert metrics["metamorphic_check_pass_millionths"] == 0


def test_blast_radius_minimality_and_recurrence_flake(
    oracle: PlannerDoctorQualityOracle,
) -> None:
    slot = next(
        s
        for s in oracle.manifest.slots
        if s.truth.allow_repair and s.truth.max_blast_radius_lines > 0
    )
    tight = perfect_observation_for_slot(slot)
    wide = CandidateArmObservation(
        case_id=slot.case_id,
        arm_id="hybrid-residual-only",
        output_root_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        disposition=ObservationDisposition.SUCCEED,
        predicted_defect_ids=slot.truth.seeded_defect_ids,
        predicted_localization_targets=slot.truth.localization_targets,
        repaired_defect_ids=slot.truth.seeded_defect_ids,
        satisfied_acceptance_ids=slot.truth.acceptance_criterion_ids,
        passed_hidden_test_ids=slot.truth.hidden_test_ids,
        killed_mutation_ids=slot.truth.mutation_operator_ids,
        passed_property_ids=slot.truth.property_ids,
        passed_fuzz_ids=slot.truth.fuzz_check_ids,
        passed_differential_ids=slot.truth.differential_check_ids,
        passed_metamorphic_ids=slot.truth.metamorphic_check_ids,
        discharged_proof_obligation_ids=slot.truth.proof_obligation_ids,
        reconstructed_kernel_fragment_ids=slot.truth.kernel_fragment_ids,
        compatible_api_schema_ids=slot.truth.api_schema_ids,
        blast_radius_changed_lines=slot.truth.max_blast_radius_lines + 10,
        recurrence_count=2,
        post_merge_regression_count=1,
        flake_failures=1,
        flake_trials=4,
        process_tree_terminated=True,
        capabilities_revoked=True,
        output_root_sealed=True,
        first_valid_plan=True,
    )
    tight_m = oracle.evaluate(tight).metric_map()
    wide_m = oracle.evaluate(wide).metric_map()
    assert tight_m["patch_minimality_millionths"] > wide_m["patch_minimality_millionths"]
    assert wide_m["patch_minimality_millionths"] == 0
    assert wide_m["blast_radius_changed_lines"] == slot.truth.max_blast_radius_lines + 10
    assert wide_m["recurrence_count"] == 2
    assert wide_m["post_merge_regression_count"] == 1
    assert wide_m["flake_rate_millionths"] == 250_000


def test_security_and_intent_ir_conformance(
    oracle: PlannerDoctorQualityOracle,
) -> None:
    slot = next(s for s in oracle.manifest.slots if s.pair_family == "security-ir")
    perfect = perfect_observation_for_slot(slot)
    assert oracle.evaluate(perfect).metric_map()[
        "security_ir_conformance_millionths"
    ] == 1_000_000
    missing = CandidateArmObservation(
        case_id=slot.case_id,
        arm_id="deterministic-symbolic",
        output_root_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        disposition=ObservationDisposition.REJECT,
        predicted_defect_ids=slot.truth.seeded_defect_ids,
        predicted_localization_targets=slot.truth.localization_targets,
        satisfied_acceptance_ids=slot.truth.acceptance_criterion_ids,
        passed_hidden_test_ids=slot.truth.hidden_test_ids,
        passed_property_ids=slot.truth.property_ids,
        satisfied_security_ir_ids=(),
        satisfied_intent_ir_ids=(),
        process_tree_terminated=True,
        capabilities_revoked=True,
        output_root_sealed=True,
    )
    receipt = oracle.evaluate(missing)
    assert receipt.disposition is OracleEvaluationDisposition.FAIL
    assert "security_ir_incomplete" in receipt.reason_codes or (
        receipt.metric_map()["security_ir_conformance_millionths"] == 0
    )


# ---------------------------------------------------------------------------
# Adversarial population
# ---------------------------------------------------------------------------


def test_adversarial_population_covers_required_families(
    oracle: PlannerDoctorQualityOracle,
) -> None:
    families = {item.family for item in oracle.adversarial_cases()}
    assert families == set(AdversarialFamily)
    assert len(oracle.adversarial_cases()) >= 15
    for family in AdversarialFamily:
        assert any(item.family is family for item in oracle.adversarial_cases())


def test_adversarial_evaluation_fail_closed(
    oracle: PlannerDoctorQualityOracle,
) -> None:
    injection = next(
        item
        for item in oracle.adversarial_cases()
        if item.family is AdversarialFamily.INJECTION
    )
    ok = oracle.evaluate_adversarial(
        injection.adversarial_id,
        observed_disposition=ObservationDisposition.REJECT,
        safety_floor_counts={key: 0 for key in injection.non_compensable_floor_keys},
    )
    assert ok["passed"] is True
    assert ok["promotion_eligible"] is False

    leaked = oracle.evaluate_adversarial(
        injection.adversarial_id,
        observed_disposition=ObservationDisposition.SUCCEED,
        safety_floor_counts={"hidden_oracle_access_or_mutation_count": 1},
    )
    assert leaked["passed"] is False
    assert "hidden_oracle_access_or_mutation_count" in leaked["floor_violations"]

    reward = next(
        item
        for item in oracle.adversarial_cases()
        if item.family is AdversarialFamily.REWARD_HACKING
    )
    gamed = oracle.evaluate_adversarial(
        reward.adversarial_id,
        observed_disposition=ObservationDisposition.SUCCEED,
        safety_floor_counts={"benchmark_or_denominator_mutation_count": 1},
    )
    assert gamed["passed"] is False


def test_adversarial_families_include_poison_forged_frontiers_and_resource_loss(
    oracle: PlannerDoctorQualityOracle,
) -> None:
    required = {
        AdversarialFamily.POISONED_INDEX,
        AdversarialFamily.POISONED_CACHE,
        AdversarialFamily.FORGED_RECEIPT,
        AdversarialFamily.MISSING_CALLER,
        AdversarialFamily.DYNAMIC_FRONTIER,
        AdversarialFamily.NATIVE_FRONTIER,
        AdversarialFamily.CONCURRENCY_FRONTIER,
        AdversarialFamily.SANDBOX_FAULT,
        AdversarialFamily.TRANSACTION_FAULT,
        AdversarialFamily.ROLLBACK_FAULT,
        AdversarialFamily.FIXED_POINT_FAULT,
        AdversarialFamily.RESOURCE_LOSS,
        AdversarialFamily.TELEMETRY_LOSS,
        AdversarialFamily.REWARD_HACKING,
        AdversarialFamily.INJECTION,
    }
    assert required <= {item.family for item in oracle.adversarial_cases()}


# ---------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------


def test_ablations_isolate_required_subsystems(
    oracle: PlannerDoctorQualityOracle,
    benchmark_policy: dict[str, Any],
) -> None:
    ablations = oracle.ablations()
    by_id = {item.ablation_id: item for item in ablations}
    # Benchmark diagnostic set must be present.
    for arm in benchmark_policy["diagnostic_ablations"]["arms"]:
        assert arm["ablation_id"] in by_id
        assert by_id[arm["ablation_id"]].disabled_subsystem.value == (
            arm["disabled_subsystem"]
        )
        assert by_id[arm["ablation_id"]].promotion_authority is False
    # Acceptance also requires LLM and parallel isolations.
    assert "without-llm" in by_id
    assert by_id["without-llm"].disabled_subsystem is AblationSubsystem.LLM
    assert "without-parallel" in by_id
    assert by_id["without-parallel"].disabled_subsystem is AblationSubsystem.PARALLEL
    assert all(item.interface == PLANNER_DOCTOR_ABLATION_INTERFACE for item in ablations)
    assert all(item.one_factor_at_a_time for item in ablations)


def test_ablation_rejects_promotion_authority() -> None:
    with pytest.raises(QualityOracleError, match="promotion"):
        PlannerDoctorAblation(
            ablation_id="without-llm",
            disabled_subsystem=AblationSubsystem.LLM,
            promotion_authority=True,
        )


def test_ablation_delta_explains_metric_change(
    oracle: PlannerDoctorQualityOracle,
) -> None:
    slot = oracle.manifest.slots[0]
    reference_obs = perfect_observation_for_slot(slot)
    reference = oracle.evaluate(reference_obs)
    ablated_obs = CandidateArmObservation(
        case_id=slot.case_id,
        arm_id="hybrid-residual-only",
        output_root_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        disposition=ObservationDisposition.SUCCEED,
        satisfied_acceptance_ids=slot.truth.acceptance_criterion_ids[:1],
        passed_hidden_test_ids=(),
        first_valid_plan=False,
        process_tree_terminated=True,
        capabilities_revoked=True,
        output_root_sealed=True,
    )
    ablated = oracle.evaluate(ablated_obs, ablation_id="without-ast-program-graph")
    delta = oracle.evaluate_ablation_delta(
        ablation_id="without-ast-program-graph",
        reference=reference,
        ablated=ablated,
    )
    assert delta["promotion_authority"] is False
    assert delta["interface"] == PLANNER_DOCTOR_ABLATION_INTERFACE
    assert delta["ablation_id"] == "without-ast-program-graph"
    assert "acceptance_coverage_millionths" in delta["metric_deltas"]
    assert delta["metric_deltas"]["acceptance_coverage_millionths"] < 0


def test_default_ablations_and_adversarial_builders() -> None:
    ablations = default_ablations()
    assert len(ablations) == 9
    adversarial = default_adversarial_cases()
    assert {item.family for item in adversarial} == set(AdversarialFamily)


# ---------------------------------------------------------------------------
# Rebuild identity / loaders
# ---------------------------------------------------------------------------


def test_rebuild_matches_sealed_fixture(
    benchmark_manifest: dict[str, Any],
    benchmark_policy: dict[str, Any],
    oracle_manifest: QualityOracleManifest,
) -> None:
    rebuilt = build_quality_oracle_manifest(
        benchmark_manifest=benchmark_manifest,
        benchmark_policy_cid=benchmark_policy["policy_cid"],
        benchmark_manifest_cid=benchmark_manifest["manifest_cid"],
    )
    assert rebuilt.content_id == oracle_manifest.content_id
    again = build_default_oracle_manifest(repo_root=ROOT)
    assert again.content_id == oracle_manifest.content_id


def test_create_and_load_default_helpers() -> None:
    engine = create_planner_doctor_quality_oracle(repo_root=ROOT)
    assert engine.interface == PLANNER_DOCTOR_QUALITY_ORACLE_INTERFACE
    loaded = PlannerDoctorQualityOracle.load_default(repo_root=ROOT)
    assert loaded.oracle_manifest_cid == engine.oracle_manifest_cid


def test_population_mismatch_is_rejected(
    oracle: PlannerDoctorQualityOracle,
) -> None:
    with pytest.raises(QualityOracleError, match="population"):
        oracle.require_exact_case_population(["only-one-case"])


def test_incomplete_adversarial_manifest_rejected(
    oracle_manifest: QualityOracleManifest,
) -> None:
    with pytest.raises(QualityOracleError, match="adversarial"):
        QualityOracleManifest(
            oracle_handle=ORACLE_HANDLE,
            benchmark_manifest_cid=oracle_manifest.benchmark_manifest_cid,
            benchmark_policy_cid=oracle_manifest.benchmark_policy_cid,
            slots=oracle_manifest.slots,
            adversarial_cases=oracle_manifest.adversarial_cases[:3],
            ablations=oracle_manifest.ablations,
        )


def test_incomplete_ablation_manifest_rejected(
    oracle_manifest: QualityOracleManifest,
) -> None:
    with pytest.raises(QualityOracleError, match="ablation"):
        QualityOracleManifest(
            oracle_handle=ORACLE_HANDLE,
            benchmark_manifest_cid=oracle_manifest.benchmark_manifest_cid,
            benchmark_policy_cid=oracle_manifest.benchmark_policy_cid,
            slots=oracle_manifest.slots,
            adversarial_cases=oracle_manifest.adversarial_cases,
            ablations=oracle_manifest.ablations[:2],
        )


def test_receipt_round_trip(oracle: PlannerDoctorQualityOracle) -> None:
    slot = oracle.manifest.slots[2]
    receipt = oracle.evaluate(perfect_observation_for_slot(slot))
    restored = QualityOracleReceipt.from_dict(receipt.to_dict())
    assert restored.content_id == receipt.content_id
    assert restored.disposition is receipt.disposition
    assert restored.metric_map() == receipt.metric_map()


def test_benchmark_policy_metric_registry_alignment(
    benchmark_policy: dict[str, Any],
) -> None:
    registry = benchmark_policy["metric_registry"]
    # Oracle solution metrics include counterexample validity as an extension;
    # every preregistered planner/doctor/solution metric must be measurable.
    for name in registry["planner_quality"]:
        assert name in PLANNER_QUALITY_METRICS
    for name in registry["doctor_quality"]:
        assert name in DOCTOR_QUALITY_METRICS
    for name in registry["solution_quality"]:
        assert name in SOLUTION_QUALITY_METRICS
    assert "counterexample_validity_millionths" in SOLUTION_QUALITY_METRICS
