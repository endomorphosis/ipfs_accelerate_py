"""PDR-092: independently replayed terminal Planner/Doctor release receipt.

Acceptance:

* validator reloads every required source receipt/artifact;
* recomputes CIDs/preimages/current roots and child-goal coverage;
* rejects stale/synthetic/skipped/forged/self-authored/incomplete evidence;
* proves zero safety floors and exact rollback;
* distinguishes task completion from objective completion;
* replaying the same current inputs is identity-equivalent;
* unavailable optional capabilities are documented without converting to pass;
* automatic promotion stays subject to a later held-out current-tree decision.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.validation import (
    planner_doctor_release as release,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = (
    _REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "validation"
    / "planner_doctor_release.py"
)
_DOC_PATH = (
    _REPO_ROOT / "docs" / "architecture" / "PROOF_DIRECTED_PLANNER_DOCTOR_RELEASE.md"
)

REQUIRED_AST_SYMBOLS = {
    "PlannerDoctorReleasePolicy",
    "PlannerDoctorReleaseReceipt",
    "PlannerDoctorReleaseValidator",
    "validate_planner_doctor_release",
    "replay_release_receipt",
    "classify_evidence_disposition",
    "check_child_goal_coverage",
    "check_reject_bad_evidence",
    "check_zero_safety_floors",
    "check_exact_rollback",
}


@pytest.fixture(scope="module")
def receipt() -> release.PlannerDoctorReleaseReceipt:
    return release.validate_planner_doctor_release(_REPO_ROOT)


@pytest.fixture(scope="module")
def report(receipt: release.PlannerDoctorReleaseReceipt) -> dict:
    return receipt.to_dict()


@pytest.fixture(scope="module")
def second_receipt() -> release.PlannerDoctorReleaseReceipt:
    return release.validate_planner_doctor_release(_REPO_ROOT)


# ---------------------------------------------------------------------------
# Declared outputs / interfaces
# ---------------------------------------------------------------------------


def test_declared_outputs_exist() -> None:
    assert _MODULE_PATH.is_file()
    assert _DOC_PATH.is_file()
    assert Path(__file__).is_file()


def test_ast_symbols_present() -> None:
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = REQUIRED_AST_SYMBOLS - names
    assert not missing, f"missing AST symbols: {sorted(missing)}"


def test_interfaces_and_policy_defaults() -> None:
    policy = release.default_release_policy()
    assert policy.task_id == "PDR-092"
    assert policy.goal_id == "PDR-G100"
    assert policy.board_namespace == (
        "agent-supervisor-proof-directed-planner-doctor-v1"
    )
    assert policy.default_mode == "report_only"
    assert policy.mutation_authorized is False
    assert policy.completion_authoritative is False
    assert policy.automatic_promotion_enabled is False
    assert policy.doctor_mutation_authorized is False
    assert policy.refill_enabled is False
    assert policy.llm_invocations_allowed is False
    assert policy.remote_model_provider_calls_allowed is False
    assert release.RELEASE_POLICY_INTERFACE == "PlannerDoctorReleasePolicy@1"
    assert release.RELEASE_RECEIPT_INTERFACE == "PlannerDoctorReleaseReceipt@1"
    assert release.RELEASE_VALIDATOR_INTERFACE == "PlannerDoctorReleaseValidator@1"
    assert release.CANONICAL_TASK_COUNT == 43
    assert release.CANONICAL_GOAL_COUNT == 11
    assert release.TERMINAL_TASK_ID == "PDR-092"
    assert release.LANE_COUNT == 6
    assert len(release.SAFETY_FLOOR_KEYS) == 20
    assert len(release.EXPECTED_TASK_IDS) == 43
    assert len(release.EXPECTED_GOAL_IDS) == 11
    assert release.EXPECTED_TASK_IDS[-1] == "PDR-092"
    assert "PDR-G100" in release.EXPECTED_GOAL_IDS


def test_policy_rejects_unsafe_defaults() -> None:
    with pytest.raises(release.PlannerDoctorReleaseError):
        release.PlannerDoctorReleasePolicy(default_mode="automatic")
    with pytest.raises(release.PlannerDoctorReleaseError):
        release.PlannerDoctorReleasePolicy(mutation_authorized=True)
    with pytest.raises(release.PlannerDoctorReleaseError):
        release.PlannerDoctorReleasePolicy(completion_authoritative=True)
    with pytest.raises(release.PlannerDoctorReleaseError):
        release.PlannerDoctorReleasePolicy(automatic_promotion_enabled=True)
    with pytest.raises(release.PlannerDoctorReleaseError):
        release.PlannerDoctorReleasePolicy(
            safety_floors={**release._zero_floors(), "false_completion_count": 1}
        )


# ---------------------------------------------------------------------------
# Board / artifacts / coverage
# ---------------------------------------------------------------------------


def test_declared_artifacts_and_protected_anchors() -> None:
    artifacts = release.check_declared_artifacts(_REPO_ROOT)
    assert artifacts.ok, artifacts.detail
    assert artifacts.status is release.CheckStatus.PASS
    assert all(artifacts.evidence["artifacts"].values())

    protected = release.check_protected_anchors(_REPO_ROOT)
    assert protected.ok, protected.detail
    assert protected.evidence["release_may_rewrite_protected"] is False
    assert all(protected.evidence["protected_present"].values())


def test_canonical_board_terminal_and_preimages() -> None:
    result = release.check_canonical_board(_REPO_ROOT)
    assert result.ok, result.detail
    assert result.status is release.CheckStatus.PASS
    assert result.evidence["canonical_task_count"] == 43
    assert result.evidence["goal_count"] == 11
    assert result.evidence["sinks"] == ["PDR-092"]
    assert result.evidence["terminal_goal"] == "PDR-G100"
    assert result.evidence["max_lanes"] == 6
    assert result.evidence["terminal_depends_on"] == list(release.TERMINAL_DEPENDENCIES)
    assert str(result.evidence["task_preimage_root"]).startswith("sha256:")
    assert len(result.evidence["task_preimages"]) == 43


def test_source_artifact_reload_recomputes_forest_root(
    receipt: release.PlannerDoctorReleaseReceipt,
) -> None:
    result = release.check_source_artifact_reload(_REPO_ROOT)
    assert result.ok, result.detail
    assert result.evidence["missing"] == []
    assert result.evidence["forged"] == []
    assert result.evidence["reloaded_from_current_tree"] is True
    assert result.evidence["self_authored_release_module"] is False
    assert str(result.evidence["forest_root"]).startswith("sha256:")
    assert receipt.forest_root == result.evidence["forest_root"]
    assert receipt.forest_root


def test_child_goal_coverage_independent_of_task_counts(
    receipt: release.PlannerDoctorReleaseReceipt,
) -> None:
    result = release.check_child_goal_coverage(_REPO_ROOT)
    assert result.ok, result.detail
    assert result.evidence["covered_count"] == result.evidence["required_count"]
    assert result.evidence["objective_completion_from_task_counts"] is False
    for goal_id in release.CHILD_GOAL_IDS:
        body = result.evidence["coverage"][goal_id]
        assert body["covered"] is True, goal_id
        assert body["uses_task_count_as_authority"] is False
        assert body["missing_artifacts"] == []
        assert str(body["evidence_root"]).startswith("sha256:")
    root = result.evidence["coverage"][release.ROOT_GOAL_ID]
    assert root["covered"] is True
    assert root["children_covered"] is True
    assert receipt.child_goal_coverage_root.startswith("sha256:")


def test_task_vs_objective_completion_distinction() -> None:
    result = release.check_task_vs_objective_completion(_REPO_ROOT)
    assert result.ok, result.detail
    assert result.evidence["distinct"] is True
    assert result.evidence["completion_authoritative"] is False
    assert result.evidence["task_completion"]["unique_terminal"] is True
    assert result.evidence["objective_completion"]["from_task_counts"] is False
    assert result.evidence["objective_completion"][
        "requires_independent_evidence_roots"
    ] is True
    assert result.evidence["objective_completion"]["child_goals_covered"] is True


# ---------------------------------------------------------------------------
# Evidence rejection / floors / rollback
# ---------------------------------------------------------------------------


def test_reject_bad_evidence_classes() -> None:
    result = release.check_reject_bad_evidence()
    assert result.ok, result.detail
    seen = set(result.evidence["rejected_classes_seen"])
    for class_name in release.REJECTED_EVIDENCE_CLASSES:
        assert class_name in seen, class_name
    assert result.evidence["wrongly_admitted"] == []

    # Direct classifier probes.
    assert (
        release.classify_evidence_disposition(
            present=True, required=True, synthetic=True
        )
        is release.EvidenceDisposition.SYNTHETIC
    )
    assert (
        release.classify_evidence_disposition(
            present=True, required=True, skipped=True
        )
        is release.EvidenceDisposition.SKIPPED
    )
    assert (
        release.classify_evidence_disposition(
            present=True,
            required=True,
            content_id="sha256:" + ("a" * 64),
            claimed_content_id="sha256:" + ("b" * 64),
        )
        is release.EvidenceDisposition.FORGED
    )
    assert (
        release.classify_evidence_disposition(
            present=True, required=True, self_authored=True
        )
        is release.EvidenceDisposition.SELF_AUTHORED
    )
    assert (
        release.classify_evidence_disposition(
            present=False, required=True
        )
        is release.EvidenceDisposition.UNAVAILABLE_REQUIRED
    )
    assert release.evidence_is_admissible(
        release.EvidenceDisposition.CURRENT, required=True
    )
    assert not release.evidence_is_admissible(
        release.EvidenceDisposition.SYNTHETIC, required=True
    )
    assert release.evidence_is_admissible(
        release.EvidenceDisposition.UNAVAILABLE_OPTIONAL, required=False
    )


def test_zero_safety_floors_and_nonzero_fail(
    receipt: release.PlannerDoctorReleaseReceipt,
) -> None:
    floors = receipt.checks["zero_safety_floors"]
    assert floors["status"] == "pass", floors
    assert floors["evidence"]["nonzero"] == {}
    for key in release.SAFETY_FLOOR_KEYS:
        assert floors["evidence"]["floors"][key] == 0

    bad = release.check_zero_safety_floors(
        floor_projection={**release._zero_floors(), "false_completion_count": 1}
    )
    assert bad.ok is False
    assert bad.status is release.CheckStatus.FAIL
    assert "false_completion_count" in bad.evidence["nonzero"]


def test_exact_rollback(
    receipt: release.PlannerDoctorReleaseReceipt,
) -> None:
    result = receipt.checks["exact_rollback"]
    assert result["status"] == "pass", result
    assert result["evidence"]["roots_match"] is True
    assert result["evidence"]["rollback_failure_count"] == 0
    assert result["evidence"]["before_root"] == result["evidence"]["restored_root"]
    assert result["evidence"]["before_root"] != result["evidence"]["tampered_root"]


# ---------------------------------------------------------------------------
# Optional capabilities / automatic / drain / cold
# ---------------------------------------------------------------------------


def test_optional_capabilities_documented_without_auto_pass(
    receipt: release.PlannerDoctorReleaseReceipt,
) -> None:
    result = receipt.checks["optional_capabilities"]
    assert result["status"] == "pass", result
    assert result["evidence"]["absence_converted_to_pass"] is False
    assert result["evidence"]["required_gates_independent_of_optional_presence"] is True
    for body in result["evidence"]["capabilities"].values():
        assert body["required"] is False
        assert body["counts_as_release_pass"] is False


def test_automatic_promotion_gated_to_later_holdout(
    receipt: release.PlannerDoctorReleaseReceipt,
) -> None:
    result = receipt.checks["automatic_promotion_gated"]
    assert result["status"] == "pass", result
    assert result["evidence"]["automatic_enabled"] is False
    assert result["evidence"]["release_grants_automatic"] is False
    assert result["evidence"]["holdout_operator_decision_required"] is True
    assert result["evidence"]["holdout_manifest_present"] is True
    assert result["evidence"]["authority_automatic_promotion_enabled"] is not True
    assert result["evidence"]["rollout_automatic_in_allowed_modes"] is False


def test_six_lane_drain_and_cold_imports_and_report_only(
    receipt: release.PlannerDoctorReleaseReceipt,
) -> None:
    drain = receipt.checks["six_lane_supervisor_drain"]
    assert drain["status"] == "pass", drain
    assert drain["evidence"]["lanes"] == 6
    assert drain["evidence"]["dependency_blockage"] is False
    assert drain["evidence"]["provider_blockage"] is False
    assert drain["evidence"]["protected_path_blockage"] is False
    assert drain["evidence"]["merge_blockage"] is False
    assert drain["evidence"]["lifecycle_blockage"] is False
    assert drain["evidence"]["terminal_task_id"] == "PDR-092"

    cold = receipt.checks["cold_imports"]
    assert cold["status"] == "pass", cold
    assert cold["evidence"]["failed"] == []
    assert cold["evidence"]["optional_providers_not_required"] is True

    report_only = receipt.checks["report_only_no_write"]
    assert report_only["status"] == "pass", report_only
    assert report_only["evidence"]["mode"] == "report_only"
    assert report_only["evidence"]["mutation_authorized"] is False
    assert report_only["evidence"]["tree_unchanged"] is True


# ---------------------------------------------------------------------------
# Receipt / identity / forgery
# ---------------------------------------------------------------------------


def test_terminal_release_receipt_valid(
    receipt: release.PlannerDoctorReleaseReceipt,
    report: dict,
) -> None:
    assert receipt.valid is True
    assert report["valid"] is True
    assert report["task_id"] == "PDR-092"
    assert report["goal_id"] == "PDR-G100"
    assert report["board_terminal"] == "PDR-092"
    assert report["mutation_authorized"] is False
    assert report["completion_authoritative"] is False
    assert report["automatic_promotion_enabled"] is False
    assert report["interface"] == "PlannerDoctorReleaseReceipt@1"
    assert str(report["receipt_id"]).startswith("sha256:")
    assert release.verify_sealed(report)
    assert report["forest_root"]
    assert report["task_preimage_root"]
    assert report["child_goal_coverage_root"]

    consumed = report["consumed_interfaces"]
    assert consumed["authority_policy"] == "PlannerDoctorAuthorityPolicy@1"
    assert consumed["rollout_policy"] == "PlannerDoctorRolloutPolicy@1"
    assert consumed["live_benchmark"] == "PlannerDoctorLiveBenchmark@1"
    assert consumed["operations"] == "PlannerDoctorOperations@1"
    assert "stale" in report["rejected_evidence_classes"]
    assert "synthetic" in report["rejected_evidence_classes"]
    assert "forged" in report["rejected_evidence_classes"]


def test_dual_full_release_identity_equivalent(
    receipt: release.PlannerDoctorReleaseReceipt,
    second_receipt: release.PlannerDoctorReleaseReceipt,
) -> None:
    assert receipt.valid is True
    assert second_receipt.valid is True
    assert receipt.receipt_id == second_receipt.receipt_id
    assert receipt.forest_root == second_receipt.forest_root
    assert receipt.task_preimage_root == second_receipt.task_preimage_root
    assert receipt.child_goal_coverage_root == second_receipt.child_goal_coverage_root

    first = receipt.to_dict()
    second = second_receipt.to_dict()
    assert set(first["checks"]) == set(second["checks"])
    for name, item in first["checks"].items():
        assert item["status"] == second["checks"][name]["status"], name


def test_replay_release_receipt_round_trip(
    receipt: release.PlannerDoctorReleaseReceipt,
) -> None:
    payload = receipt.to_dict()
    replay = release.replay_release_receipt(payload)
    assert replay["valid"] is True
    assert replay["identity_ok"] is True
    assert replay["claimed_receipt_id"] == payload["receipt_id"]
    assert replay["recomputed_receipt_id"] == payload["receipt_id"]
    assert replay["mutation_authorized"] is False
    assert replay["completion_authoritative"] is False
    assert replay["automatic_promotion_enabled"] is False
    assert replay["interface"] == "PlannerDoctorReleaseReplay@1"

    typed_replay = release.replay_release_receipt(receipt)
    assert typed_replay["identity_ok"] is True
    assert typed_replay["claimed_receipt_id"] == receipt.receipt_id


def test_forged_receipt_fails_closed(
    receipt: release.PlannerDoctorReleaseReceipt,
) -> None:
    payload = receipt.to_dict()
    forged = dict(payload)
    forged["forest_root"] = "sha256:" + ("0" * 64)
    assert release.verify_sealed(forged) is False
    replay = release.replay_release_receipt(forged)
    assert replay["identity_ok"] is False
    assert replay["valid"] is False


def test_seal_payload_is_deterministic() -> None:
    body = {
        "task_id": "PDR-092",
        "goal_id": "PDR-G100",
        "valid": True,
        "nested": {"b": 2, "a": 1},
        "list": [3, 1, 2],
    }
    first = release.seal_payload(body)
    second = release.seal_payload(body)
    assert first["receipt_id"] == second["receipt_id"]
    assert release.verify_sealed(first) is True


def test_policy_binding_stable() -> None:
    first = release.default_release_policy().policy_binding_id
    second = release.default_release_policy().policy_binding_id
    assert first == second
    assert first.startswith("sha256:")


def test_validator_facade() -> None:
    validator = release.PlannerDoctorReleaseValidator(_REPO_ROOT)
    payload = validator.run_all()
    assert payload["valid"] is True
    assert payload["validator_interface"] == "PlannerDoctorReleaseValidator@1"
    assert payload["mutation_authorized"] is False
    assert payload["automatic_promotion_enabled"] is False
    assert validator.to_dict()["task_id"] == "PDR-092"
    typed = validator.validate()
    assert typed.valid is True
    assert typed.receipt_id == payload["receipt_id"]


def test_release_doc_documents_terminal_gate() -> None:
    text = _DOC_PATH.read_text(encoding="utf-8").casefold()
    for topic in (
        "pdr-092",
        "report-only",
        "child-goal",
        "rollback",
        "safety floor",
        "automatic",
        "holdout",
        "optional",
        "task completion",
        "objective completion",
        "identity",
        "no automatic",
    ):
        assert topic in text, topic


def test_all_required_checks_present_and_passing(
    receipt: release.PlannerDoctorReleaseReceipt,
) -> None:
    required = {
        "declared_artifacts",
        "protected_anchors",
        "canonical_board",
        "source_artifact_reload",
        "child_goal_coverage",
        "task_vs_objective_completion",
        "reject_bad_evidence",
        "zero_safety_floors",
        "exact_rollback",
        "optional_capabilities",
        "automatic_promotion_gated",
        "six_lane_supervisor_drain",
        "cold_imports",
        "report_only_no_write",
    }
    assert required <= set(receipt.checks)
    for name in required:
        status = receipt.checks[name]["status"]
        assert status in {"pass", "skip", "warn"}, f"{name}={status}"
