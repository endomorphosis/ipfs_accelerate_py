"""PCPR-043 fail-closed cross-repository compatibility checks."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.cross_repository_compatibility import (
    CLOSED_RELEASE_OUTCOMES,
    COMPATIBILITY_INTERFACE,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    HERMETIC_CANDIDATE_SUITES,
    PCPR_043_GOAL_ID,
    PCPR_043_TASK_ID,
    CrossRepositoryCompatibilityQualificationError,
    current_head_compatibility_probes,
    current_head_pcpr_043_current_tree_binding,
    current_head_pcpr_043_receipt_promotion,
    current_head_pcpr_043_receipt_sections,
    qualify_cross_repository_compatibility,
    qualify_current_head_compatibility,
    validate_pcpr_043_outer_receipt,
)
from ipfs_accelerate_py.assurance.cross_repository_compatibility import (
    INCOMPATIBLE_CATEGORIES,
    INTERFACE,
    PINNED_CATALOG_CID,
    PINNED_COMPATIBILITY_DOCUMENT_CID,
    SCHEMA,
    CrossRepositoryCompatibilityError,
    admit_combination,
    compatibility_document,
    compatibility_document_cid,
    evaluate_incompatible_combinations,
    refuse_compatibility_remint,
    supported_snapshot,
)
from ipfs_accelerate_py.assurance.shared_contracts import catalog_cid


def test_closed_vocabularies_match_pcpr_043_requirements() -> None:
    assert PCPR_043_TASK_ID == "PCPR-043"
    assert PCPR_043_GOAL_ID == "PCPR-G520"
    assert COMPATIBILITY_INTERFACE == "CrossRepositoryCompatibilityQualification@1"
    assert INTERFACE == "CrossRepositoryCompatibility@1"
    assert SCHEMA == "ipfs_accelerate_py/assurance/cross-repository-compatibility@1"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert INCOMPATIBLE_CATEGORIES == (
        "missing_repository",
        "reminted_identity",
        "sibling_import",
        "cross_version",
        "mixed_catalog_vectors",
        "partial_publication",
        "unsupported_python",
        "authority_boundary",
        "claimed_early",
    )
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_cross_repository_compatibility.py"
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
    assert verdict.categories_complete is True
    assert verdict.supported_combination_count == 1
    assert verdict.incompatible_count == 18
    assert verdict.datasets_binding_aligned is True
    assert verdict.kit_binding_aligned is True
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.live_supervisor_qualified is False
    assert verdict.live_supervisor_evidence_kind == "unavailable"
    assert verdict.live_cuda_qualified is False
    assert verdict.live_cuda_evidence_kind == "unavailable"
    assert verdict.production_authorized is False
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.verdict_cid.startswith("baguqeera")
    assert verdict.catalog_cid == catalog_cid() == PINNED_CATALOG_CID
    assert verdict.compatibility_document_cid == PINNED_COMPATIBILITY_DOCUMENT_CID
    section = current_head_pcpr_043_receipt_promotion()
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["contracts_frozen"] is False


def test_current_head_probes_are_measured_not_live() -> None:
    probes = {item.probe_id: item for item in current_head_compatibility_probes()}
    assert probes["compatibility_module"].present is True
    assert probes["one_supported_combination"].present is True
    assert probes["nine_incompatible_categories"].present is True
    assert probes["eighteen_incompatible_combinations_reject"].present is True
    assert probes["fail_identically_under_reorder"].present is True
    assert probes["catalog_identity_not_reminted"].present is True
    assert probes["vector_cids_not_reminted"].present is True
    assert probes["negative_document_not_reminted"].present is True
    assert probes["datasets_binding_does_not_remint"].present is True
    assert probes["kit_binding_does_not_remint"].present is True
    assert probes["ownership_boundaries_held"].present is True
    assert probes["lock_not_claimed"].present is True
    assert probes["live_supervisor_qualification"].evidence_kind == "unavailable"
    assert probes["live_supervisor_qualification"].present is None
    assert probes["live_cuda_qualification"].evidence_kind == "unavailable"
    for item in probes.values():
        assert item.live is False
        assert item.simulated_represented_as_live is False
        assert item.evidence_kind != "measured_live"
        assert item.evidence_kind != "simulated"


def test_incompatibles_reject_and_supported_combination_admits() -> None:
    document = compatibility_document()
    assert document["incompatible_count"] == 18
    assert document["supported_combination_count"] == 1
    assert document["catalog_cid"] == PINNED_CATALOG_CID
    assert compatibility_document_cid() == PINNED_COMPATIBILITY_DOCUMENT_CID
    assert (
        refuse_compatibility_remint(PINNED_COMPATIBILITY_DOCUMENT_CID)
        == PINNED_COMPATIBILITY_DOCUMENT_CID
    )
    with pytest.raises(CrossRepositoryCompatibilityError, match="remints"):
        refuse_compatibility_remint(
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
    admit_combination(supported_snapshot())
    rows = evaluate_incompatible_combinations()
    assert {item["category"] for item in rows} == set(INCOMPATIBLE_CATEGORIES)
    assert all(item["rejected"] is True for item in rows)


def test_current_head_receipt_sections_are_rnd_non_promoted() -> None:
    sections = current_head_pcpr_043_receipt_sections()
    assert sections["promotion_status"] == "rnd_non_promoted"
    assert sections["closed_release_outcome"] is None
    assert sections["release_claim"] is False
    assert sections["contracts_frozen"] is False
    assert sections["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert sections["qualification_prerequisite"]["task_id"] == "PCPR-042"
    assert sections["cross_repository_compatibility"]["frozen"] is False
    assert sections["cross_repository_compatibility"]["lock"] is False
    assert sections["cross_repository_compatibility"]["incompatible_count"] == 18
    assert sections["cross_repository_compatibility"]["supported_combination_count"] == 1
    assert sections["cross_repository_compatibility"]["typescript_compiler"] == (
        "unavailable"
    )
    assert sections["negative_results"]["closed_release_outcome_not_emitted"] is True
    assert sections["negative_results"]["catalog_not_reminted"] is True
    assert sections["negative_results"]["lock_not_claimed"] is True
    binding = current_head_pcpr_043_current_tree_binding()
    assert binding["origin_main_is_ancestor"] is True
    assert binding["accelerator_post_change_commit"].startswith("pending")


def test_outer_receipt_validator_accepts_generated_non_promotion_receipt() -> None:
    sections = current_head_pcpr_043_receipt_sections()
    payload = {
        "task_id": PCPR_043_TASK_ID,
        "status": "implemented",
        "completion_authoritative": False,
        "release_claim": False,
        "qualification_verdict": sections["qualification_verdict"],
        "qualification_prerequisite": sections["qualification_prerequisite"],
        "current_tree_binding": current_head_pcpr_043_current_tree_binding(),
        "acceptance": {
            "named_receipt_exists": True,
            "promotion_status": "rnd_non_promoted",
            "closed_release_outcome": None,
            "release_claim": False,
            "contracts_frozen": False,
        },
    }
    checked = validate_pcpr_043_outer_receipt(payload)
    assert checked["valid"] is True
    assert checked["promotion_status"] == "rnd_non_promoted"
    assert checked["closed_release_outcome"] is None
    assert checked["release_claim"] is False
    assert checked["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_outer_receipt_validator_rejects_closed_release_outcome() -> None:
    sections = current_head_pcpr_043_receipt_sections()
    forged = {
        "task_id": PCPR_043_TASK_ID,
        "status": "implemented",
        "qualification_verdict": dict(sections["qualification_verdict"]),
        "acceptance": {
            "promotion_status": "release_candidate_qualified",
            "closed_release_outcome": "release_candidate_qualified",
            "release_claim": True,
        },
    }
    with pytest.raises(
        CrossRepositoryCompatibilityQualificationError, match="closed PCPR release"
    ):
        validate_pcpr_043_outer_receipt(forged)


def test_qualify_rejects_duckdb_write() -> None:
    probes = current_head_compatibility_probes()
    with pytest.raises(CrossRepositoryCompatibilityQualificationError, match="DuckDB"):
        qualify_cross_repository_compatibility(
            probes=probes,
            duckdb_or_quack_state_written=True,
        )


def test_qualify_rejects_freeze_and_live_claims() -> None:
    probes = current_head_compatibility_probes()
    with pytest.raises(CrossRepositoryCompatibilityQualificationError, match="freeze"):
        qualify_cross_repository_compatibility(probes=probes, contracts_frozen=True)
    with pytest.raises(CrossRepositoryCompatibilityQualificationError, match="live"):
        qualify_cross_repository_compatibility(
            probes=probes, live_cuda_qualified=True
        )
