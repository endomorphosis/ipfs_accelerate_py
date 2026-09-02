"""PCPR-032 fail-closed Accelerate pseudo-CID identity removal."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.pseudo_cid_identity_removal import (
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    HERMETIC_CANDIDATE_SUITES,
    PCPR_032_GOAL_ID,
    PCPR_032_TASK_ID,
    REMOVAL_INTERFACE,
    PseudoCidIdentityRemovalError,
    RemovalProbe,
    current_head_pcpr_032_current_tree_binding,
    current_head_pcpr_032_receipt_promotion,
    current_head_pcpr_032_receipt_sections,
    current_head_removal_probes,
    qualify_current_head_removal,
    qualify_pseudo_cid_identity_removal,
    validate_pcpr_032_outer_receipt,
)
from ipfs_accelerate_py.assurance.content_identity import (
    CanonicalIPFSMultiformats,
    is_qm_like,
    is_raw_sha256_hex,
    legacy_pseudo_cid,
)
from ipfs_accelerate_py.compatibility.simulation.pseudo_cid import (
    MockIPFSClient,
    PseudoCidIdentityError,
    UnavailableIpfsClient,
    instantiate_mock_ipfs_client,
    load_ordinary_ipfs_client,
    load_ordinary_multiformats,
    mint_canonical_cid,
    ordinary_store_to_ipfs,
    random_cid,
    simulate_store_to_ipfs,
)


def test_closed_vocabularies_match_pcpr_032_requirements() -> None:
    assert PCPR_032_TASK_ID == "PCPR-032"
    assert PCPR_032_GOAL_ID == "PCPR-G410"
    assert REMOVAL_INTERFACE == "PseudoCidIdentityRemoval@1"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_pseudo_cid_identity_removal.py"
    )


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_removal()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.ordinary_runtime_uses_pseudo_cid is False
    assert verdict.mock_ipfs_quarantined is True
    assert verdict.random_cid_requires_canonical_bytes is True
    assert verdict.canonical_bytes_produce_real_cid is True
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.live_ipfs_qualified is False
    assert verdict.live_ipfs_evidence_kind == "unavailable"
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.verdict_cid.startswith("baguqeera")
    section = current_head_pcpr_032_receipt_promotion()
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["live_ipfs_qualified"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_ordinary_multiformats_mints_canonical_cid_not_hex() -> None:
    adapter = load_ordinary_multiformats()
    assert isinstance(adapter, CanonicalIPFSMultiformats)
    payload = b"pcpr-032-ordinary"
    cid = adapter.get_cid(payload)
    assert not is_raw_sha256_hex(cid)
    assert not is_qm_like(cid)
    assert cid.startswith("b")
    assert cid == mint_canonical_cid(payload)
    hex_form = legacy_pseudo_cid(payload)
    assert is_raw_sha256_hex(hex_form)
    assert hex_form != cid


def test_ordinary_ipfs_client_is_typed_unavailable_not_mock() -> None:
    client = load_ordinary_ipfs_client()
    assert isinstance(client, UnavailableIpfsClient)
    assert not isinstance(client, MockIPFSClient)
    assert client.live is False
    result = client.add_file("missing")
    assert result["cid"] is None
    assert result["outcome"] == "Unavailable"
    with pytest.raises(PseudoCidIdentityError, match="MockIPFSClient"):
        instantiate_mock_ipfs_client()


def test_random_cid_cannot_mint_qm_even_under_simulation() -> None:
    with pytest.raises(PseudoCidIdentityError, match="random_cid"):
        random_cid()
    with pytest.raises(PseudoCidIdentityError, match="canonical bytes"):
        random_cid(explicit_simulation=True)


def test_mock_ipfs_requires_explicit_simulation_and_is_not_live(
    tmp_path,
) -> None:
    simulated = instantiate_mock_ipfs_client(explicit_simulation=True)
    assert isinstance(simulated, MockIPFSClient)
    assert simulated.live is False
    assert simulated.origin == "simulated"
    path = tmp_path / "payload.txt"
    path.write_bytes(b"pcpr-032-simulated-bytes")
    added = simulated.add_file(str(path))
    assert added["live"] is False
    assert added["outcome"] == "Simulated"
    assert not str(added["Hash"]).startswith("Qm")
    assert added["Hash"].startswith("b")
    retrieved = simulated.cat(added["Hash"])
    assert retrieved == b"pcpr-032-simulated-bytes"


def test_ordinary_store_is_unavailable_and_simulation_is_not_live() -> None:
    unavailable = ordinary_store_to_ipfs(b"pcpr-032-store")
    assert unavailable["cid"] is None
    assert unavailable["outcome"] == "Unavailable"
    assert unavailable["live"] is False
    simulated = simulate_store_to_ipfs(b"pcpr-032-store", explicit_simulation=True)
    assert simulated["outcome"] == "Simulated"
    assert simulated["live"] is False
    assert simulated["cid"]
    assert not str(simulated["cid"]).startswith("Qm")


def test_current_head_probes_are_measured_not_live() -> None:
    probes = {item.probe_id: item for item in current_head_removal_probes()}
    assert probes["ordinary_runtime_multiformats"].present is True
    assert probes["ordinary_runtime_multiformats"].evidence_kind == "measured"
    assert probes["simulation_namespace_pseudo_cid"].present is True
    assert probes["random_cid_requires_canonical_bytes"].present is True
    assert probes["live_ipfs_qualification"].evidence_kind == "unavailable"
    assert probes["live_ipfs_qualification"].present is None
    for item in probes.values():
        assert item.live is False
        assert item.simulated_represented_as_live is False
        assert item.evidence_kind != "measured_live"
        assert item.evidence_kind != "simulated"


def test_current_head_receipt_sections_are_rnd_non_promoted() -> None:
    sections = current_head_pcpr_032_receipt_sections()
    assert sections["promotion_status"] == "rnd_non_promoted"
    assert sections["closed_release_outcome"] is None
    assert sections["release_claim"] is False
    assert sections["contracts_frozen"] is False
    assert sections["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert sections["qualification_prerequisite"]["task_id"] == "PCPR-031"
    assert sections["removal"]["mock_ipfs_quarantined"] is True
    assert sections["removal"]["ordinary_runtime_uses_pseudo_cid"] is False
    assert sections["negative_results"]["closed_release_outcome_not_emitted"] is True
    assert sections["negative_results"]["direct_database_bypass_not_used"] is True
    assert sections["negative_results"]["live_ipfs_not_claimed"] is True


def test_outer_receipt_validator_accepts_generated_non_promotion_receipt() -> None:
    sections = current_head_pcpr_032_receipt_sections()
    payload = {
        "task_id": PCPR_032_TASK_ID,
        "status": "implemented",
        "completion_authoritative": False,
        "release_claim": False,
        "qualification_verdict": sections["qualification_verdict"],
        "qualification_prerequisite": sections["qualification_prerequisite"],
        "current_tree_binding": current_head_pcpr_032_current_tree_binding(),
        "acceptance": {
            "named_receipt_exists": True,
            "promotion_status": "rnd_non_promoted",
            "closed_release_outcome": None,
            "release_claim": False,
            "contracts_frozen": False,
        },
    }
    checked = validate_pcpr_032_outer_receipt(payload)
    assert checked["valid"] is True
    assert checked["promotion_status"] == "rnd_non_promoted"
    assert checked["closed_release_outcome"] is None
    assert checked["release_claim"] is False
    assert checked["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_outer_receipt_validator_rejects_closed_release_outcome() -> None:
    sections = current_head_pcpr_032_receipt_sections()
    forged = {
        "task_id": PCPR_032_TASK_ID,
        "status": "implemented",
        "qualification_verdict": dict(sections["qualification_verdict"]),
        "acceptance": {
            "promotion_status": "release_candidate_qualified",
            "closed_release_outcome": "release_candidate_qualified",
            "release_claim": True,
        },
    }
    with pytest.raises(PseudoCidIdentityRemovalError, match="closed PCPR release"):
        validate_pcpr_032_outer_receipt(forged)
    forged_verdict = {
        "task_id": PCPR_032_TASK_ID,
        "status": "implemented",
        "qualification_verdict": {
            **sections["qualification_verdict"],
            "closed_release_outcome": "non_promoted_live_compute_gap",
        },
        "acceptance": {
            "promotion_status": "rnd_non_promoted",
            "closed_release_outcome": None,
            "release_claim": False,
        },
    }
    with pytest.raises(PseudoCidIdentityRemovalError, match="closed PCPR release"):
        validate_pcpr_032_outer_receipt(forged_verdict)


def test_unavailable_probes_do_not_count_as_passing() -> None:
    probes = (
        RemovalProbe(
            probe_id="accelerate_source_tree",
            present=None,
            evidence_kind="unavailable",
            live=False,
            simulated_represented_as_live=False,
            reason="fixture unavailable",
        ),
    )
    verdict = qualify_pseudo_cid_identity_removal(probes=probes)
    assert verdict.promotion_status == "typed_blocked"
    assert verdict.closed_release_outcome is None
    assert verdict.live_ipfs_qualified is False
