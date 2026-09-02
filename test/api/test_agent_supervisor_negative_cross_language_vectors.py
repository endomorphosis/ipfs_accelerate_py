"""PCPR-042 fail-closed negative and cross-language vectors."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.negative_cross_language_vectors import (
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    HERMETIC_CANDIDATE_SUITES,
    PCPR_042_GOAL_ID,
    PCPR_042_TASK_ID,
    VECTOR_INTERFACE,
    NegativeCrossLanguageVectorQualificationError,
    current_head_pcpr_042_current_tree_binding,
    current_head_pcpr_042_receipt_promotion,
    current_head_pcpr_042_receipt_sections,
    current_head_vector_probes,
    qualify_current_head_vectors,
    qualify_negative_cross_language_vectors,
    validate_pcpr_042_outer_receipt,
)
from ipfs_accelerate_py.assurance.negative_cross_language_vectors import (
    INTERFACE,
    NEGATIVE_CATEGORIES,
    PINNED_CATALOG_CID,
    PINNED_VECTOR_DOCUMENT_CID,
    SCHEMA,
    NegativeCrossLanguageVectorError,
    evaluate_negative_vectors,
    javascript_agrees_with_python,
    negative_cross_language_vector_document,
    refuse_negative_remint,
    run_javascript_vectors,
    vector_document_cid,
)
from ipfs_accelerate_py.assurance.shared_contracts import catalog_cid


def test_closed_vocabularies_match_pcpr_042_requirements() -> None:
    assert PCPR_042_TASK_ID == "PCPR-042"
    assert PCPR_042_GOAL_ID == "PCPR-G520"
    assert VECTOR_INTERFACE == "NegativeCrossLanguageVectorQualification@1"
    assert INTERFACE == "NegativeCrossLanguageVectors@1"
    assert SCHEMA == "ipfs_accelerate_py/assurance/negative-cross-language-vectors@1"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert NEGATIVE_CATEGORIES == (
        "invalid",
        "stale",
        "unknown",
        "out_of_bound",
        "reordered",
        "reminted",
        "cross_version",
    )
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_negative_cross_language_vectors.py"
    )


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_vectors()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.categories_complete is True
    assert verdict.javascript_agreed is True
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
    assert verdict.vector_document_cid == PINNED_VECTOR_DOCUMENT_CID
    section = current_head_pcpr_042_receipt_promotion()
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["contracts_frozen"] is False


def test_current_head_probes_are_measured_not_live() -> None:
    probes = {item.probe_id: item for item in current_head_vector_probes()}
    assert probes["negative_vector_module"].present is True
    assert probes["seven_negative_categories"].present is True
    assert probes["eighteen_negative_vectors_reject"].present is True
    assert probes["fail_identically_under_reorder"].present is True
    assert probes["catalog_identity_not_reminted"].present is True
    assert probes["positive_cid_vectors_not_reminted"].present is True
    assert probes["javascript_encoder_agrees"].present is True
    assert probes["typescript_source_present"].present is True
    assert probes["typescript_compiler"].evidence_kind == "unavailable"
    assert probes["typescript_compiler"].present is None
    assert probes["datasets_binding_does_not_remint"].present is True
    assert probes["kit_binding_does_not_remint"].present is True
    assert probes["live_supervisor_qualification"].evidence_kind == "unavailable"
    assert probes["live_supervisor_qualification"].present is None
    assert probes["live_cuda_qualification"].evidence_kind == "unavailable"
    for item in probes.values():
        assert item.live is False
        assert item.simulated_represented_as_live is False
        assert item.evidence_kind != "measured_live"
        assert item.evidence_kind != "simulated"


def test_negatives_reject_and_javascript_agrees() -> None:
    document = negative_cross_language_vector_document()
    assert document["negative_count"] == 18
    assert document["cross_language_count"] == 7
    assert document["catalog_cid"] == PINNED_CATALOG_CID
    assert vector_document_cid() == PINNED_VECTOR_DOCUMENT_CID
    assert refuse_negative_remint(PINNED_VECTOR_DOCUMENT_CID) == PINNED_VECTOR_DOCUMENT_CID
    with pytest.raises(NegativeCrossLanguageVectorError, match="remints"):
        refuse_negative_remint(
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
    negatives = evaluate_negative_vectors()
    assert {item["category"] for item in negatives} == set(NEGATIVE_CATEGORIES)
    report = run_javascript_vectors()
    javascript_agrees_with_python(report)
    assert report["language"] == "JavaScript"
    assert report["simulated"] is False


def test_current_head_receipt_sections_are_rnd_non_promoted() -> None:
    sections = current_head_pcpr_042_receipt_sections()
    assert sections["promotion_status"] == "rnd_non_promoted"
    assert sections["closed_release_outcome"] is None
    assert sections["release_claim"] is False
    assert sections["contracts_frozen"] is False
    assert sections["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert sections["qualification_prerequisite"]["task_id"] == "PCPR-041"
    assert sections["negative_and_cross_language_vectors"]["frozen"] is False
    assert sections["negative_and_cross_language_vectors"]["negative_count"] == 18
    assert sections["negative_and_cross_language_vectors"]["typescript_compiler"] == (
        "unavailable"
    )
    assert sections["negative_results"]["closed_release_outcome_not_emitted"] is True
    assert sections["negative_results"]["catalog_not_reminted"] is True
    binding = current_head_pcpr_042_current_tree_binding()
    assert binding["origin_main_is_ancestor"] is True
    assert binding["accelerator_post_change_commit"].startswith("pending")


def test_outer_receipt_validator_accepts_generated_non_promotion_receipt() -> None:
    sections = current_head_pcpr_042_receipt_sections()
    payload = {
        "task_id": PCPR_042_TASK_ID,
        "status": "implemented",
        "completion_authoritative": False,
        "release_claim": False,
        "qualification_verdict": sections["qualification_verdict"],
        "qualification_prerequisite": sections["qualification_prerequisite"],
        "current_tree_binding": current_head_pcpr_042_current_tree_binding(),
        "acceptance": {
            "named_receipt_exists": True,
            "promotion_status": "rnd_non_promoted",
            "closed_release_outcome": None,
            "release_claim": False,
            "contracts_frozen": False,
        },
    }
    checked = validate_pcpr_042_outer_receipt(payload)
    assert checked["valid"] is True
    assert checked["promotion_status"] == "rnd_non_promoted"
    assert checked["closed_release_outcome"] is None
    assert checked["release_claim"] is False
    assert checked["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_outer_receipt_validator_rejects_closed_release_outcome() -> None:
    sections = current_head_pcpr_042_receipt_sections()
    forged = {
        "task_id": PCPR_042_TASK_ID,
        "status": "implemented",
        "qualification_verdict": dict(sections["qualification_verdict"]),
        "acceptance": {
            "promotion_status": "release_candidate_qualified",
            "closed_release_outcome": "release_candidate_qualified",
            "release_claim": True,
        },
    }
    with pytest.raises(
        NegativeCrossLanguageVectorQualificationError, match="closed PCPR release"
    ):
        validate_pcpr_042_outer_receipt(forged)


def test_qualify_rejects_duckdb_write() -> None:
    probes = current_head_vector_probes()
    with pytest.raises(NegativeCrossLanguageVectorQualificationError, match="DuckDB"):
        qualify_negative_cross_language_vectors(
            probes=probes,
            duckdb_or_quack_state_written=True,
        )


def test_qualify_rejects_freeze_and_live_claims() -> None:
    probes = current_head_vector_probes()
    with pytest.raises(NegativeCrossLanguageVectorQualificationError, match="freeze"):
        qualify_negative_cross_language_vectors(probes=probes, contracts_frozen=True)
    with pytest.raises(NegativeCrossLanguageVectorQualificationError, match="live"):
        qualify_negative_cross_language_vectors(
            probes=probes, live_cuda_qualified=True
        )
