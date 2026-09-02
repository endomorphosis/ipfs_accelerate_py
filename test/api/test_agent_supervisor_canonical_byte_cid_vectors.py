"""PCPR-041 fail-closed canonical-byte and CID vectors."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.canonical_byte_cid_vectors import (
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    HERMETIC_CANDIDATE_SUITES,
    PCPR_041_GOAL_ID,
    PCPR_041_TASK_ID,
    VECTOR_INTERFACE,
    CanonicalByteCidVectorQualificationError,
    current_head_pcpr_041_current_tree_binding,
    current_head_pcpr_041_receipt_promotion,
    current_head_pcpr_041_receipt_sections,
    current_head_vector_probes,
    qualify_canonical_byte_cid_vectors,
    qualify_current_head_vectors,
    validate_pcpr_041_outer_receipt,
)
from ipfs_accelerate_py.assurance.canonical_byte_cid_vectors import (
    INTERFACE,
    PINNED_CATALOG_CID,
    PINNED_CONTRACT_VECTORS,
    PINNED_VECTOR_DOCUMENT_CID,
    SCHEMA,
    CanonicalByteCidVectorError,
    canonical_byte_cid_vector_document,
    catalog_identity_vector,
    contract_vector,
    decode_and_recompute,
    refuse_vector_remint,
    vector_document_cid,
)
from ipfs_accelerate_py.assurance.shared_contracts import (
    REQUIRED_CONTRACT_NAMES,
    catalog_cid,
)


def test_closed_vocabularies_match_pcpr_041_requirements() -> None:
    assert PCPR_041_TASK_ID == "PCPR-041"
    assert PCPR_041_GOAL_ID == "PCPR-G510"
    assert VECTOR_INTERFACE == "CanonicalByteCidVectorQualification@1"
    assert INTERFACE == "CanonicalByteCidVectors@1"
    assert SCHEMA == "ipfs_accelerate_py/assurance/canonical-byte-cid-vectors@1"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert len(REQUIRED_CONTRACT_NAMES) == 14
    assert len(PINNED_CONTRACT_VECTORS) == 14
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_canonical_byte_cid_vectors.py"
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
    assert verdict.identities_unique is True
    assert verdict.canonical_bytes_pinned is True
    assert verdict.cid_recomputes is True
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
    section = current_head_pcpr_041_receipt_promotion()
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["contracts_frozen"] is False


def test_current_head_probes_are_measured_not_live() -> None:
    probes = {item.probe_id: item for item in current_head_vector_probes()}
    assert probes["canonical_vector_module"].present is True
    assert probes["fourteen_contract_vectors"].present is True
    assert probes["canonical_bytes_pinned"].present is True
    assert probes["cid_recomputes_from_bytes"].present is True
    assert probes["key_order_independent"].present is True
    assert probes["unicode_nfc_same_cid"].present is True
    assert probes["catalog_identity_not_reminted"].present is True
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


def test_vectors_recompute_and_refuse_remint() -> None:
    document = canonical_byte_cid_vector_document()
    assert document["vector_count"] == 17
    assert document["catalog_cid"] == PINNED_CATALOG_CID
    assert vector_document_cid() == PINNED_VECTOR_DOCUMENT_CID
    for item in document["vectors"]:
        if item.get("canonical_hex"):
            assert decode_and_recompute(item) == item["cid"]
    primary = contract_vector("SupervisorObjectiveIntent")
    assert refuse_vector_remint(
        "SupervisorObjectiveIntent", primary["cid"]
    ) == primary["cid"]
    with pytest.raises(CanonicalByteCidVectorError, match="remints"):
        refuse_vector_remint(
            "SupervisorObjectiveIntent",
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
    catalog = catalog_identity_vector()
    assert catalog["cid"] == PINNED_CATALOG_CID


def test_current_head_receipt_sections_are_rnd_non_promoted() -> None:
    sections = current_head_pcpr_041_receipt_sections()
    assert sections["promotion_status"] == "rnd_non_promoted"
    assert sections["closed_release_outcome"] is None
    assert sections["release_claim"] is False
    assert sections["contracts_frozen"] is False
    assert sections["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert sections["qualification_prerequisite"]["task_id"] == "PCPR-040"
    assert sections["canonical_byte_and_cid_vectors"]["frozen"] is False
    assert sections["canonical_byte_and_cid_vectors"]["vector_count"] == 17
    assert len(sections["canonical_byte_and_cid_vectors"]["vector_cids"]) == 14
    assert sections["negative_results"]["closed_release_outcome_not_emitted"] is True
    assert sections["negative_results"]["catalog_not_reminted"] is True
    binding = current_head_pcpr_041_current_tree_binding()
    assert binding["origin_main_is_ancestor"] is True
    assert binding["accelerator_post_change_commit"].startswith("pending")


def test_outer_receipt_validator_accepts_generated_non_promotion_receipt() -> None:
    sections = current_head_pcpr_041_receipt_sections()
    payload = {
        "task_id": PCPR_041_TASK_ID,
        "status": "implemented",
        "completion_authoritative": False,
        "release_claim": False,
        "qualification_verdict": sections["qualification_verdict"],
        "qualification_prerequisite": sections["qualification_prerequisite"],
        "current_tree_binding": current_head_pcpr_041_current_tree_binding(),
        "acceptance": {
            "named_receipt_exists": True,
            "promotion_status": "rnd_non_promoted",
            "closed_release_outcome": None,
            "release_claim": False,
            "contracts_frozen": False,
        },
    }
    checked = validate_pcpr_041_outer_receipt(payload)
    assert checked["valid"] is True
    assert checked["promotion_status"] == "rnd_non_promoted"
    assert checked["closed_release_outcome"] is None
    assert checked["release_claim"] is False
    assert checked["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_outer_receipt_validator_rejects_closed_release_outcome() -> None:
    sections = current_head_pcpr_041_receipt_sections()
    forged = {
        "task_id": PCPR_041_TASK_ID,
        "status": "implemented",
        "qualification_verdict": dict(sections["qualification_verdict"]),
        "acceptance": {
            "promotion_status": "release_candidate_qualified",
            "closed_release_outcome": "release_candidate_qualified",
            "release_claim": True,
        },
    }
    with pytest.raises(
        CanonicalByteCidVectorQualificationError, match="closed PCPR release"
    ):
        validate_pcpr_041_outer_receipt(forged)


def test_qualify_rejects_duckdb_write() -> None:
    probes = current_head_vector_probes()
    with pytest.raises(CanonicalByteCidVectorQualificationError, match="DuckDB"):
        qualify_canonical_byte_cid_vectors(
            probes=probes,
            duckdb_or_quack_state_written=True,
        )


def test_qualify_rejects_freeze_and_live_claims() -> None:
    probes = current_head_vector_probes()
    with pytest.raises(CanonicalByteCidVectorQualificationError, match="freeze"):
        qualify_canonical_byte_cid_vectors(probes=probes, contracts_frozen=True)
    with pytest.raises(CanonicalByteCidVectorQualificationError, match="live"):
        qualify_canonical_byte_cid_vectors(
            probes=probes, live_cuda_qualified=True
        )
