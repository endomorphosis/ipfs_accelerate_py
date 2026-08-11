"""LPR-042: end-to-end joined VFS + deterministic-doctor release.

Acceptance:

* 43 canonical tasks, 12 goals, LPR-042 unique terminal;
* semantic CIDs of LPR-000 through LPR-028 preserved;
* VFS and non-VFS profiles plus doctor fixtures dual-run with identical CIDs;
* optional provider absence and cold imports are safe;
* report-only makes no write; eligible no-model repair reaches all-caller fixed point;
* ambiguous/unsupported cases abstain with clean trees; rollback restores exact roots;
* zero LLM/model-provider, authority promotion, stale CID, missed caller, escape,
  partial transaction, rollback failure, nondeterminism, and false completion floors;
* healthy four-lane supervisor can drain the joined DAG without blockage.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.validation import (
    deterministic_doctor_release as release,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = (
    _REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "validation"
    / "deterministic_doctor_release.py"
)
_DOC_PATH = _REPO_ROOT / "docs" / "architecture" / "DETERMINISTIC_DOCTOR_RELEASE.md"
_REPLAY_TEST = (
    _REPO_ROOT / "test" / "api" / "test_agent_supervisor_deterministic_doctor_replay.py"
)

REQUIRED_AST_SYMBOLS = {
    "DeterministicDoctorReleasePolicy",
    "DeterministicDoctorReleaseReceipt",
    "validate_deterministic_doctor_release",
}


@pytest.fixture(scope="module")
def receipt() -> release.DeterministicDoctorReleaseReceipt:
    return release.validate_deterministic_doctor_release(_REPO_ROOT)


@pytest.fixture(scope="module")
def report(receipt: release.DeterministicDoctorReleaseReceipt) -> dict:
    return receipt.to_dict()


def test_declared_outputs_exist() -> None:
    assert _MODULE_PATH.is_file()
    assert _DOC_PATH.is_file()
    assert Path(__file__).is_file()
    assert _REPLAY_TEST.is_file()


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
    assert policy.task_id == "LPR-042"
    assert policy.goal_id == "LPR-G110"
    assert policy.default_mode == "report_only"
    assert policy.mutation_authorized is False
    assert policy.completion_authoritative is False
    assert policy.llm_invocations_allowed is False
    assert policy.remote_model_provider_calls_allowed is False
    assert release.RELEASE_POLICY_INTERFACE == "DeterministicDoctorReleasePolicy@1"
    assert release.RELEASE_RECEIPT_INTERFACE == "DeterministicDoctorReleaseReceipt@1"
    assert release.CANONICAL_TASK_COUNT == 43
    assert release.CANONICAL_GOAL_COUNT == 12
    assert release.TERMINAL_TASK_ID == "LPR-042"
    assert len(release.SEALED_TASK_CIDS_LPR_000_028) == 29
    assert len(release.SAFETY_FLOOR_KEYS) == 9


def test_canonical_board_and_preserved_cids() -> None:
    result = release.check_canonical_board(_REPO_ROOT)
    assert result.ok, result.detail
    assert result.status is release.CheckStatus.PASS
    assert result.evidence["canonical_task_count"] == 43
    assert result.evidence["goal_count"] == 12
    assert result.evidence["sinks"] == ["LPR-042"]
    assert result.evidence["cid_mismatches"] == []
    assert result.evidence["preserved_cid_count"] == 29


def test_four_lane_supervisor_drain() -> None:
    result = release.check_four_lane_supervisor_drain(_REPO_ROOT)
    assert result.ok, result.detail
    assert result.evidence["lanes"] == 4
    assert result.evidence["dependency_blockage"] is False
    assert result.evidence["provider_blockage"] is False
    assert result.evidence["protected_path_blockage"] is False
    assert result.evidence["merge_blockage"] is False
    assert result.evidence["lifecycle_blockage"] is False


def test_report_only_no_write() -> None:
    result = release.check_report_only_no_write(_REPO_ROOT)
    assert result.ok, result.detail
    assert result.evidence["mode"] == "report_only"
    assert result.evidence["mutation_authorized"] is False
    assert result.evidence["tree_unchanged"] is True


def test_optional_provider_absence_and_cold_imports() -> None:
    providers = release.check_optional_provider_absence_safe()
    assert providers.ok, providers.detail
    assert providers.evidence["report_only_startup_ok"] is True
    assert providers.evidence["absence_blocks_report_only_startup"] is False

    cold = release.check_cold_imports()
    assert cold.ok, cold.detail
    assert cold.evidence["failed"] == []


def test_doctor_fixture_dual_run_and_fixed_point(
    receipt: release.DeterministicDoctorReleaseReceipt,
) -> None:
    checks = receipt.checks
    doctor = checks["doctor_fixture_dual_run"]
    assert doctor["status"] == "pass", doctor
    assert doctor["evidence"]["identity_equivalent"] is True
    assert doctor["evidence"]["llm_zero"] is True
    assert doctor["evidence"]["floors_hold"] is True
    assert doctor["evidence"]["case_count"] >= 16

    fixed = checks["eligible_fixed_point"]
    assert fixed["status"] == "pass", fixed
    assert fixed["evidence"]["all_caller_atomic_fixed_point"] is True
    assert fixed["evidence"]["failures"] == []
    assert fixed["evidence"]["positive_count"] == len(release.ADMITTABLE_SCENARIOS)


def test_abstention_rollback_and_zero_floors(
    receipt: release.DeterministicDoctorReleaseReceipt,
) -> None:
    abstain = receipt.checks["abstention_and_rollback"]
    assert abstain["status"] == "pass", abstain
    assert abstain["evidence"]["clean_tree"] is True
    assert abstain["evidence"]["rollback_restores_exact_roots"] is True
    assert abstain["evidence"]["abstention_failures"] == []

    floors = receipt.checks["zero_safety_floors"]
    assert floors["status"] == "pass", floors
    assert floors["evidence"]["nonzero"] == {}
    for key in release.SAFETY_FLOOR_KEYS:
        assert floors["evidence"]["floors"][key] == 0


def test_vfs_and_non_vfs_profiles_dual_run(
    receipt: release.DeterministicDoctorReleaseReceipt,
) -> None:
    vfs = receipt.checks["vfs_profiles_dual_run"]
    assert vfs["status"] == "pass", vfs
    assert vfs["evidence"]["equivalence_passed"] is True
    assert vfs["evidence"]["conformance_passed"] is True
    assert vfs["evidence"]["vfs_ok"] is True
    assert vfs["evidence"]["non_vfs_ok"] is True
    assert vfs["evidence"]["equivalence_identity_equivalent"] is True
    assert vfs["evidence"]["vfs_identity_equivalent"] is True
    assert vfs["evidence"]["non_vfs_identity_equivalent"] is True
    assert vfs["evidence"]["vfs_profile_id"] != vfs["evidence"]["non_vfs_profile_id"]


def test_joined_release_receipt_valid(
    receipt: release.DeterministicDoctorReleaseReceipt,
    report: dict,
) -> None:
    assert receipt.valid is True
    assert report["valid"] is True
    assert report["task_id"] == "LPR-042"
    assert report["goal_id"] == "LPR-G110"
    assert report["board_terminal"] == "LPR-042"
    assert report["mutation_authorized"] is False
    assert report["completion_authoritative"] is False
    assert report["interface"] == "DeterministicDoctorReleaseReceipt@1"
    assert str(report["receipt_id"]).startswith("sha256:")
    assert release.verify_sealed(report)
    assert report["doctor_report_id"]
    assert report["vfs_equivalence_content_id"]
    assert report["two_profile_content_id"]

    consumed = report["consumed_interfaces"]
    assert consumed["vfs_equivalence"] == "VfsGeneralizationEquivalenceReceipt@1"
    assert consumed["two_profile"] == "AssuranceTwoProfileConformance@1"
    assert consumed["doctor_run_receipt"] == "DeterministicDoctorRunReceipt@1"
    assert consumed["doctor_metrics"] == "DeterministicDoctorMetrics@1"
    assert consumed["propagation_completion"] == "PropagationCompletionReceipt@1"
    assert consumed["logic_fixed_point"] == "LogicFixedPointEvidenceAttachment@1"


def test_validator_facade() -> None:
    validator = release.DeterministicDoctorReleaseValidator(_REPO_ROOT)
    payload = validator.run_all()
    assert payload["valid"] is True
    assert payload["validator_interface"] == "DeterministicDoctorReleaseValidator@1"
    assert payload["mutation_authorized"] is False
    assert validator.to_dict()["task_id"] == "LPR-042"


def test_release_doc_documents_joined_gate() -> None:
    text = _DOC_PATH.read_text(encoding="utf-8").casefold()
    for topic in (
        "lpr-042",
        "report-only",
        "vfs",
        "fixed point",
        "rollback",
        "cold import",
        "safety floor",
        "four-lane",
        "no llm",
    ):
        assert topic in text, topic
