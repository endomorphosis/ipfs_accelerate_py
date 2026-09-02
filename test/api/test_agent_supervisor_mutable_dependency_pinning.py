"""PCPR-035 fail-closed Accelerate mutable-dependency pinning."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.legacy_mock_coordinator_quarantine import (
    discover_accelerate_root,
)
from ipfs_accelerate_py.agent_supervisor.validation.mutable_dependency_pinning import (
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    HERMETIC_CANDIDATE_SUITES,
    PCPR_035_GOAL_ID,
    PCPR_035_TASK_ID,
    PINNING_EVALUATOR_INTERFACE,
    MutableDependencyPinningError,
    current_head_pcpr_035_current_tree_binding,
    current_head_pcpr_035_receipt_promotion,
    current_head_pcpr_035_receipt_sections,
    current_head_pinning_probes,
    qualify_current_head_pinning,
    qualify_mutable_dependency_pinning,
    validate_pcpr_035_outer_receipt,
)
from ipfs_accelerate_py.assurance.mutable_dependency_pins import (
    COMMIT_RE,
    PINS,
    mutable_git_references,
    pep508_spec,
)
from ipfs_accelerate_py.mcplusplus_module.p2p.libp2p_runtime import (
    PY_LIBP2P_MAIN_SPEC,
    PY_LIBP2P_PINNED_SPEC,
)


def test_closed_vocabularies_match_pcpr_035_requirements() -> None:
    assert PCPR_035_TASK_ID == "PCPR-035"
    assert PCPR_035_GOAL_ID == "PCPR-G420"
    assert PINNING_EVALUATOR_INTERFACE == "MutableDependencyPinning@1"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_mutable_dependency_pinning.py"
    )
    for pin in PINS.values():
        assert COMMIT_RE.fullmatch(pin.commit)
        assert pin.live is False
        assert "@main" not in pin.pep508()
        assert "@master" not in pin.pep508()


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_pinning()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.qualified_profiles_contain_mutable_branch is False
    assert verdict.vcs_refs_are_immutable_commits is True
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.live_vcs_install_qualified is False
    assert verdict.live_vcs_install_evidence_kind == "unavailable"
    assert verdict.live_libp2p_qualified is False
    assert verdict.live_libp2p_evidence_kind == "unavailable"
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.verdict_cid.startswith("baguqeera")
    section = current_head_pcpr_035_receipt_promotion()
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["live_vcs_install_qualified"] is False
    assert section["live_libp2p_qualified"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_parser_treats_branch_refs_as_mutable_and_commits_as_pins() -> None:
    mutable = mutable_git_references(
        'demo @ git+https://example.invalid/demo.git@main\n'
        'kit @ git+https://github.com/endomorphosis/ipfs_kit_py.git@'
        'a4fd25d944e6aaf25fb51e054a3f62bbcbea544d\n'
    )
    assert len(mutable) == 1
    assert mutable[0]["ref"] == "main"
    assert pep508_spec("libp2p") == PY_LIBP2P_PINNED_SPEC
    assert PY_LIBP2P_MAIN_SPEC == PY_LIBP2P_PINNED_SPEC
    assert "@main" not in PY_LIBP2P_PINNED_SPEC
    assert COMMIT_RE.fullmatch(PINS["libp2p"].commit)


def test_pyproject_and_hf_server_have_no_mutable_branch() -> None:
    root = discover_accelerate_root()
    assert root is not None
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    hf = (root / "requirements-hf-server.txt").read_text(encoding="utf-8")
    assert not mutable_git_references(pyproject)
    assert not mutable_git_references(hf)
    assert PINS["ipfs_transformers_py"].commit in pyproject
    assert PINS["ipfs_model_manager_py"].commit in pyproject
    assert PINS["libp2p"].commit in pyproject
    assert PINS["ipfs_kit_py"].commit in hf
    assert PINS["ipfs_datasets_py"].commit in hf
    assert "@main" not in pyproject
    assert "@main" not in hf


def test_current_head_probes_are_measured_not_live() -> None:
    probes = {item.probe_id: item for item in current_head_pinning_probes()}
    assert probes["canonical_pin_table"].present is True
    assert probes["pyproject_qualified_extras_pinned"].present is True
    assert probes["hf_server_requirements_pinned"].present is True
    assert probes["qualified_profiles_no_mutable_branch"].present is True
    assert probes["runtime_libp2p_spec_pinned"].present is True
    assert probes["ipfs_kit_py_gitlink"].present is True
    assert probes["live_vcs_install_qualification"].evidence_kind == "unavailable"
    assert probes["live_vcs_install_qualification"].present is None
    assert probes["live_libp2p_qualification"].evidence_kind == "unavailable"
    for item in probes.values():
        assert item.live is False
        assert item.simulated_represented_as_live is False
        assert item.evidence_kind != "measured_live"
        assert item.evidence_kind != "simulated"


def test_current_head_receipt_sections_are_rnd_non_promoted() -> None:
    sections = current_head_pcpr_035_receipt_sections()
    assert sections["promotion_status"] == "rnd_non_promoted"
    assert sections["closed_release_outcome"] is None
    assert sections["release_claim"] is False
    assert sections["contracts_frozen"] is False
    assert sections["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert sections["qualification_prerequisite"]["task_id"] == "PCPR-034"
    assert sections["pinning"]["qualified_profiles_contain_mutable_branch"] is False
    assert sections["pinning"]["vcs_refs_are_immutable_commits"] is True
    assert sections["negative_results"]["closed_release_outcome_not_emitted"] is True
    assert sections["negative_results"]["direct_database_bypass_not_used"] is True
    assert sections["negative_results"]["live_vcs_install_not_claimed"] is True
    assert sections["negative_results"]["mutable_branch_not_retained_in_qualified_profiles"] is True


def test_outer_receipt_validator_accepts_generated_non_promotion_receipt() -> None:
    sections = current_head_pcpr_035_receipt_sections()
    payload = {
        "task_id": PCPR_035_TASK_ID,
        "status": "implemented",
        "completion_authoritative": False,
        "release_claim": False,
        "qualification_verdict": sections["qualification_verdict"],
        "qualification_prerequisite": sections["qualification_prerequisite"],
        "current_tree_binding": current_head_pcpr_035_current_tree_binding(),
        "acceptance": {
            "named_receipt_exists": True,
            "promotion_status": "rnd_non_promoted",
            "closed_release_outcome": None,
            "release_claim": False,
            "contracts_frozen": False,
        },
    }
    checked = validate_pcpr_035_outer_receipt(payload)
    assert checked["valid"] is True
    assert checked["promotion_status"] == "rnd_non_promoted"
    assert checked["closed_release_outcome"] is None
    assert checked["release_claim"] is False
    assert checked["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_outer_receipt_validator_rejects_closed_release_outcome() -> None:
    sections = current_head_pcpr_035_receipt_sections()
    forged = {
        "task_id": PCPR_035_TASK_ID,
        "status": "implemented",
        "qualification_verdict": dict(sections["qualification_verdict"]),
        "acceptance": {
            "promotion_status": "release_candidate_qualified",
            "closed_release_outcome": "release_candidate_qualified",
            "release_claim": True,
        },
    }
    with pytest.raises(MutableDependencyPinningError, match="closed PCPR release"):
        validate_pcpr_035_outer_receipt(forged)


def test_qualify_rejects_duckdb_write() -> None:
    probes = current_head_pinning_probes()
    with pytest.raises(MutableDependencyPinningError, match="DuckDB"):
        qualify_mutable_dependency_pinning(
            probes=probes,
            duckdb_or_quack_state_written=True,
        )


def test_qualify_rejects_live_libp2p_claim() -> None:
    probes = current_head_pinning_probes()
    with pytest.raises(MutableDependencyPinningError, match="live"):
        qualify_mutable_dependency_pinning(
            probes=probes,
            live_libp2p_qualified=True,
        )
