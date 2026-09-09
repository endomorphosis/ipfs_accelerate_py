"""PCPR-036 fail-closed Accelerate Python compatibility metadata."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.legacy_mock_coordinator_quarantine import (
    discover_accelerate_root,
)
from ipfs_accelerate_py.agent_supervisor.validation.python_compatibility_metadata import (
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    HERMETIC_CANDIDATE_SUITES,
    PCPR_036_GOAL_ID,
    PCPR_036_TASK_ID,
    COMPATIBILITY_EVALUATOR_INTERFACE,
    PythonCompatibilityMetadataError,
    current_head_compatibility_probes,
    current_head_pcpr_036_current_tree_binding,
    current_head_pcpr_036_receipt_promotion,
    current_head_pcpr_036_receipt_sections,
    qualify_current_head_compatibility,
    qualify_python_compatibility_metadata,
    validate_pcpr_036_outer_receipt,
)
from ipfs_accelerate_py.assurance.python_compatibility import (
    COMPATIBILITY_COMMAND,
    DECLARED_PYTHON_VERSIONS,
    FORBIDDEN_CLASSIFIERS,
    PYTHON_FLOOR,
    PYPROJECT_REQUIRES_PYTHON_LINE,
    REQUIRES_PYTHON,
    SETUP_PYTHON_REQUIRES,
    STABLE_COMMAND,
    source_has_forbidden_classifiers,
)


def test_closed_vocabularies_match_pcpr_036_requirements() -> None:
    assert PCPR_036_TASK_ID == "PCPR-036"
    assert PCPR_036_GOAL_ID == "PCPR-G420"
    assert COMPATIBILITY_EVALUATOR_INTERFACE == "PythonCompatibilityMetadata@1"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert PYTHON_FLOOR == "3.12"
    assert REQUIRES_PYTHON == ">=3.12"
    assert DECLARED_PYTHON_VERSIONS == ("3.12",)
    assert STABLE_COMMAND == "ipfs-accelerate"
    assert COMPATIBILITY_COMMAND == "ipfs_accelerate"
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_python_compatibility_metadata.py"
    )


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_compatibility()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.declared_python_floor == "3.12"
    assert verdict.requires_python == ">=3.12"
    assert verdict.declared_python_versions == ("3.12",)
    assert verdict.metadata_agrees_on_floor is True
    assert verdict.ci_declares_same_floor is True
    assert verdict.commands_retain_explicit_migration is True
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.live_ci_matrix_executed is False
    assert verdict.live_ci_matrix_evidence_kind == "unavailable"
    assert verdict.live_older_python_qualified is False
    assert verdict.live_older_python_evidence_kind == "unavailable"
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.verdict_cid.startswith("baguqeera")
    section = current_head_pcpr_036_receipt_promotion()
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["live_ci_matrix_executed"] is False
    assert section["live_older_python_qualified"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_pyproject_and_setup_agree_on_python_312_floor() -> None:
    root = discover_accelerate_root()
    assert root is not None
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    setup = (root / "setup.py").read_text(encoding="utf-8")
    assert PYPROJECT_REQUIRES_PYTHON_LINE in pyproject
    assert SETUP_PYTHON_REQUIRES in setup
    assert 'requires-python = ">=3.8"' not in pyproject
    assert 'python_requires=">=3.8"' not in setup
    assert not source_has_forbidden_classifiers(pyproject)
    assert not source_has_forbidden_classifiers(setup)
    for classifier in FORBIDDEN_CLASSIFIERS:
        assert classifier not in pyproject
        assert classifier not in setup
    assert STABLE_COMMAND in pyproject
    assert COMPATIBILITY_COMMAND in pyproject


def test_current_head_probes_are_measured_not_live() -> None:
    probes = {item.probe_id: item for item in current_head_compatibility_probes()}
    assert probes["canonical_floor_table"].present is True
    assert probes["pyproject_requires_python"].present is True
    assert probes["pyproject_classifiers"].present is True
    assert probes["setup_python_requires"].present is True
    assert probes["setup_classifiers"].present is True
    assert probes["ci_workflows_declare_python_312"].present is True
    assert probes["command_hierarchy_retains_compatibility_migration"].present is True
    assert probes["declared_versions_only_3_12"].present is True
    assert probes["live_ci_matrix_execution"].evidence_kind == "unavailable"
    assert probes["live_ci_matrix_execution"].present is None
    assert probes["live_older_python_qualification"].evidence_kind == "unavailable"
    for item in probes.values():
        assert item.live is False
        assert item.simulated_represented_as_live is False
        assert item.evidence_kind != "measured_live"
        assert item.evidence_kind != "simulated"


def test_current_head_receipt_sections_are_rnd_non_promoted() -> None:
    sections = current_head_pcpr_036_receipt_sections()
    assert sections["promotion_status"] == "rnd_non_promoted"
    assert sections["closed_release_outcome"] is None
    assert sections["release_claim"] is False
    assert sections["contracts_frozen"] is False
    assert sections["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert sections["qualification_prerequisite"]["task_id"] == "PCPR-035"
    assert sections["compatibility"]["declared_python_floor"] == "3.12"
    assert sections["compatibility"]["commands_retain_explicit_migration"] is True
    assert sections["negative_results"]["closed_release_outcome_not_emitted"] is True
    assert sections["negative_results"]["direct_database_bypass_not_used"] is True
    assert sections["negative_results"]["live_ci_matrix_not_claimed"] is True
    assert sections["negative_results"]["untested_python_version_not_declared"] is True


def test_outer_receipt_validator_accepts_generated_non_promotion_receipt() -> None:
    sections = current_head_pcpr_036_receipt_sections()
    payload = {
        "task_id": PCPR_036_TASK_ID,
        "status": "implemented",
        "completion_authoritative": False,
        "release_claim": False,
        "qualification_verdict": sections["qualification_verdict"],
        "qualification_prerequisite": sections["qualification_prerequisite"],
        "current_tree_binding": current_head_pcpr_036_current_tree_binding(),
        "acceptance": {
            "named_receipt_exists": True,
            "promotion_status": "rnd_non_promoted",
            "closed_release_outcome": None,
            "release_claim": False,
            "contracts_frozen": False,
        },
    }
    checked = validate_pcpr_036_outer_receipt(payload)
    assert checked["valid"] is True
    assert checked["promotion_status"] == "rnd_non_promoted"
    assert checked["closed_release_outcome"] is None
    assert checked["release_claim"] is False
    assert checked["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_outer_receipt_validator_rejects_closed_release_outcome() -> None:
    sections = current_head_pcpr_036_receipt_sections()
    forged = {
        "task_id": PCPR_036_TASK_ID,
        "status": "implemented",
        "qualification_verdict": dict(sections["qualification_verdict"]),
        "acceptance": {
            "promotion_status": "release_candidate_qualified",
            "closed_release_outcome": "release_candidate_qualified",
            "release_claim": True,
        },
    }
    with pytest.raises(PythonCompatibilityMetadataError, match="closed PCPR release"):
        validate_pcpr_036_outer_receipt(forged)


def test_qualify_rejects_duckdb_write() -> None:
    probes = current_head_compatibility_probes()
    with pytest.raises(PythonCompatibilityMetadataError, match="DuckDB"):
        qualify_python_compatibility_metadata(
            probes=probes,
            duckdb_or_quack_state_written=True,
        )


def test_qualify_rejects_live_ci_claim() -> None:
    probes = current_head_compatibility_probes()
    with pytest.raises(PythonCompatibilityMetadataError, match="live"):
        qualify_python_compatibility_metadata(
            probes=probes,
            live_ci_matrix_executed=True,
        )


def test_current_tree_binding_is_measured_and_not_a_release() -> None:
    binding = current_head_pcpr_036_current_tree_binding()
    assert binding["evidence_kind"] == "measured"
    assert binding["origin_main_is_ancestor"] is True
    assert binding["accelerator_origin_main_is_ancestor"] is True
    assert binding["accelerator_post_change_commit"].startswith("pending")
    assert binding["outer_repository"] == "endomorphosis/lift_coding"
