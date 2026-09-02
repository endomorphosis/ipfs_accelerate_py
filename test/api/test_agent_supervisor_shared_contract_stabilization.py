"""PCPR-040 fail-closed shared-contract stabilization."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.legacy_mock_coordinator_quarantine import (
    discover_accelerate_root,
)
from ipfs_accelerate_py.agent_supervisor.validation.shared_contract_stabilization import (
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    HERMETIC_CANDIDATE_SUITES,
    PCPR_040_GOAL_ID,
    PCPR_040_TASK_ID,
    STABILIZATION_INTERFACE,
    SharedContractStabilizationError,
    current_head_pcpr_040_current_tree_binding,
    current_head_pcpr_040_receipt_promotion,
    current_head_pcpr_040_receipt_sections,
    current_head_stabilization_probes,
    qualify_current_head_stabilization,
    qualify_shared_contract_stabilization,
    validate_pcpr_040_outer_receipt,
)
from ipfs_accelerate_py.assurance.shared_contracts import (
    CONTRACT_SCHEMA_IDS,
    INTERFACE,
    REQUIRED_CONTRACT_NAMES,
    SCHEMA,
    SharedContractAdmissionError,
    admit_shared_contract,
    catalog_cid,
    identity_fixture,
    refuse_remint,
)


def test_closed_vocabularies_match_pcpr_040_requirements() -> None:
    assert PCPR_040_TASK_ID == "PCPR-040"
    assert PCPR_040_GOAL_ID == "PCPR-G500"
    assert STABILIZATION_INTERFACE == "SharedContractStabilization@1"
    assert INTERFACE == "SharedContractCatalog@1"
    assert SCHEMA == "ipfs_accelerate_py/assurance/shared-contracts-catalog@1"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert len(REQUIRED_CONTRACT_NAMES) == 14
    assert REQUIRED_CONTRACT_NAMES[0] == "SupervisorObjectiveIntent"
    assert REQUIRED_CONTRACT_NAMES[-1] == "PortfolioCompatibilityManifest"
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_shared_contract_stabilization.py"
    )


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_stabilization()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.identities_unique is True
    assert verdict.unknown_fields_rejected is True
    assert verdict.canonicalization_deterministic is True
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
    assert verdict.catalog_cid == catalog_cid()
    section = current_head_pcpr_040_receipt_promotion()
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["contracts_frozen"] is False


def test_current_head_probes_are_measured_not_live() -> None:
    probes = {item.probe_id: item for item in current_head_stabilization_probes()}
    assert probes["canonical_catalog_module"].present is True
    assert probes["fourteen_unique_identities"].present is True
    assert probes["unknown_fields_rejected"].present is True
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


def test_unknown_fields_and_remint_are_rejected() -> None:
    fixture = identity_fixture("ExecutionReceipt")
    with pytest.raises(SharedContractAdmissionError, match="unknown fields"):
        admit_shared_contract(
            "ExecutionReceipt", {**fixture, "release_claim": True}
        )
    with pytest.raises(SharedContractAdmissionError, match="remints"):
        refuse_remint("ProofObligation", "not_yet_normative")
    with pytest.raises(SharedContractAdmissionError, match="raw SHA-256"):
        admit_shared_contract(
            "DurableArtifactReceipt",
            {**identity_fixture("DurableArtifactReceipt"), "cid": "ab" * 32},
        )


def test_catalog_identities_are_unique() -> None:
    assert len(set(CONTRACT_SCHEMA_IDS.values())) == 14
    assert all(
        schema.startswith("pcpr/shared-contracts/")
        for schema in CONTRACT_SCHEMA_IDS.values()
    )
    root = discover_accelerate_root()
    assert root is not None
    source = (
        root / "ipfs_accelerate_py/assurance/shared_contracts.py"
    ).read_text(encoding="utf-8")
    for name in REQUIRED_CONTRACT_NAMES:
        assert name in source


def test_current_head_receipt_sections_are_rnd_non_promoted() -> None:
    sections = current_head_pcpr_040_receipt_sections()
    assert sections["promotion_status"] == "rnd_non_promoted"
    assert sections["closed_release_outcome"] is None
    assert sections["release_claim"] is False
    assert sections["contracts_frozen"] is False
    assert sections["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert sections["qualification_prerequisite"]["task_id"] == "PCPR-039"
    assert sections["shared_contracts"]["frozen"] is False
    assert len(sections["shared_contracts"]["contracts"]) == 14
    assert sections["negative_results"]["closed_release_outcome_not_emitted"] is True
    assert sections["negative_results"]["identities_not_reminted"] is True
    binding = current_head_pcpr_040_current_tree_binding()
    assert binding["origin_main_is_ancestor"] is True
    assert binding["accelerator_post_change_commit"].startswith("pending")


def test_outer_receipt_validator_accepts_generated_non_promotion_receipt() -> None:
    sections = current_head_pcpr_040_receipt_sections()
    payload = {
        "task_id": PCPR_040_TASK_ID,
        "status": "implemented",
        "completion_authoritative": False,
        "release_claim": False,
        "qualification_verdict": sections["qualification_verdict"],
        "qualification_prerequisite": sections["qualification_prerequisite"],
        "current_tree_binding": current_head_pcpr_040_current_tree_binding(),
        "acceptance": {
            "named_receipt_exists": True,
            "promotion_status": "rnd_non_promoted",
            "closed_release_outcome": None,
            "release_claim": False,
            "contracts_frozen": False,
        },
    }
    checked = validate_pcpr_040_outer_receipt(payload)
    assert checked["valid"] is True
    assert checked["promotion_status"] == "rnd_non_promoted"
    assert checked["closed_release_outcome"] is None
    assert checked["release_claim"] is False
    assert checked["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_outer_receipt_validator_rejects_closed_release_outcome() -> None:
    sections = current_head_pcpr_040_receipt_sections()
    forged = {
        "task_id": PCPR_040_TASK_ID,
        "status": "implemented",
        "qualification_verdict": dict(sections["qualification_verdict"]),
        "acceptance": {
            "promotion_status": "release_candidate_qualified",
            "closed_release_outcome": "release_candidate_qualified",
            "release_claim": True,
        },
    }
    with pytest.raises(SharedContractStabilizationError, match="closed PCPR release"):
        validate_pcpr_040_outer_receipt(forged)


def test_qualify_rejects_duckdb_write() -> None:
    probes = current_head_stabilization_probes()
    with pytest.raises(SharedContractStabilizationError, match="DuckDB"):
        qualify_shared_contract_stabilization(
            probes=probes,
            duckdb_or_quack_state_written=True,
        )


def test_qualify_rejects_freeze_and_live_claims() -> None:
    probes = current_head_stabilization_probes()
    with pytest.raises(SharedContractStabilizationError, match="freeze"):
        qualify_shared_contract_stabilization(probes=probes, contracts_frozen=True)
    with pytest.raises(SharedContractStabilizationError, match="live"):
        qualify_shared_contract_stabilization(
            probes=probes, live_cuda_qualified=True
        )
