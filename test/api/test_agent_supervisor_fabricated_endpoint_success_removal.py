"""PCPR-033 fail-closed Accelerate fabricated endpoint-success removal."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.fabricated_endpoint_success_removal import (
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    HERMETIC_CANDIDATE_SUITES,
    PCPR_033_GOAL_ID,
    PCPR_033_TASK_ID,
    REMOVAL_INTERFACE,
    FabricatedEndpointSuccessRemovalError,
    RemovalProbe,
    current_head_pcpr_033_current_tree_binding,
    current_head_pcpr_033_receipt_promotion,
    current_head_pcpr_033_receipt_sections,
    current_head_removal_probes,
    qualify_current_head_removal,
    qualify_fabricated_endpoint_success_removal,
    validate_pcpr_033_outer_receipt,
)
from ipfs_accelerate_py.compatibility.simulation.fabricated_endpoint_success import (
    FabricatedEndpointSuccessError,
    SimulatedEndpointHandler,
    UnavailableEndpointHandler,
    create_simulated_endpoint_handler,
    create_unavailable_endpoint_handler,
    ordinary_mock_inference,
    simulate_inference,
    unavailable_hardware_test,
)


def test_closed_vocabularies_match_pcpr_033_requirements() -> None:
    assert PCPR_033_TASK_ID == "PCPR-033"
    assert PCPR_033_GOAL_ID == "PCPR-G410"
    assert REMOVAL_INTERFACE == "FabricatedEndpointSuccessRemoval@1"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_fabricated_endpoint_success_removal.py"
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
    assert verdict.ordinary_runtime_uses_mock_endpoint_success is False
    assert verdict.mock_endpoint_handler_quarantined is True
    assert verdict.mock_inference_quarantined is True
    assert verdict.hardware_test_fallback_not_overall_passed is True
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.live_endpoint_qualified is False
    assert verdict.live_endpoint_evidence_kind == "unavailable"
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.verdict_cid.startswith("baguqeera")
    section = current_head_pcpr_033_receipt_promotion()
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["live_endpoint_qualified"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_ordinary_endpoint_handler_is_typed_unavailable_not_mock() -> None:
    handler = create_unavailable_endpoint_handler("bert-base", "cpu:0")
    assert isinstance(handler, UnavailableEndpointHandler)
    assert not isinstance(handler, SimulatedEndpointHandler)
    assert handler.live is False
    result = handler("hello")
    assert result["outcome"] == "Unavailable"
    assert result["status"] != "success"
    assert result["implementation_type"] != "REAL"
    assert result["live"] is False
    with pytest.raises(FabricatedEndpointSuccessError, match="Mock endpoint handler"):
        create_simulated_endpoint_handler("bert-base", "cpu:0")


def test_mock_handler_requires_explicit_simulation_and_is_not_live() -> None:
    simulated = create_simulated_endpoint_handler(
        "gpt2", "cpu:0", explicit_simulation=True
    )
    assert isinstance(simulated, SimulatedEndpointHandler)
    assert simulated.live is False
    assert simulated.origin == "simulated"
    labeled = simulated("pcpr-033")
    assert labeled["live"] is False
    assert labeled["origin"] == "simulated"
    assert labeled["outcome"] == "Simulated"
    assert labeled["implementation_type"] == "MOCK"
    assert labeled["status"] != "success"


def test_ordinary_mock_inference_is_unavailable_and_simulation_is_not_live() -> None:
    unavailable = ordinary_mock_inference(
        "causal_language_modeling", "gpt2", {"prompt": "pcpr-033"}
    )
    assert unavailable["outcome"] == "Unavailable"
    assert unavailable["live"] is False
    assert "generated_text" not in unavailable
    simulated = simulate_inference(
        "causal_language_modeling",
        "gpt2",
        {"prompt": "pcpr-033"},
        explicit_simulation=True,
    )
    assert simulated["outcome"] == "Simulated"
    assert simulated["live"] is False
    assert simulated["status"] != "success"


def test_hardware_test_fallback_is_not_overall_passed() -> None:
    result = unavailable_hardware_test("cuda", "basic")
    assert result["outcome"] == "Unavailable"
    assert result["overall_passed"] is None
    assert result["status"] != "success"
    assert result["live"] is False


def test_current_head_probes_are_measured_not_live() -> None:
    probes = {item.probe_id: item for item in current_head_removal_probes()}
    assert probes["ordinary_runtime_no_real_mock_label"].present is True
    assert probes["ordinary_runtime_no_real_mock_label"].evidence_kind == "measured"
    assert probes["simulation_namespace_fabricated_endpoint_success"].present is True
    assert probes["hardware_fallback_not_overall_passed"].present is True
    assert probes["live_endpoint_qualification"].evidence_kind == "unavailable"
    assert probes["live_endpoint_qualification"].present is None
    for item in probes.values():
        assert item.live is False
        assert item.simulated_represented_as_live is False
        assert item.evidence_kind != "measured_live"
        assert item.evidence_kind != "simulated"


def test_current_head_receipt_sections_are_rnd_non_promoted() -> None:
    sections = current_head_pcpr_033_receipt_sections()
    assert sections["promotion_status"] == "rnd_non_promoted"
    assert sections["closed_release_outcome"] is None
    assert sections["release_claim"] is False
    assert sections["contracts_frozen"] is False
    assert sections["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert sections["qualification_prerequisite"]["task_id"] == "PCPR-032"
    assert sections["removal"]["mock_endpoint_handler_quarantined"] is True
    assert sections["removal"]["ordinary_runtime_uses_mock_endpoint_success"] is False
    assert sections["negative_results"]["closed_release_outcome_not_emitted"] is True
    assert sections["negative_results"]["direct_database_bypass_not_used"] is True
    assert sections["negative_results"]["live_endpoint_not_claimed"] is True


def test_outer_receipt_validator_accepts_generated_non_promotion_receipt() -> None:
    sections = current_head_pcpr_033_receipt_sections()
    payload = {
        "task_id": PCPR_033_TASK_ID,
        "status": "implemented",
        "completion_authoritative": False,
        "release_claim": False,
        "qualification_verdict": sections["qualification_verdict"],
        "qualification_prerequisite": sections["qualification_prerequisite"],
        "current_tree_binding": current_head_pcpr_033_current_tree_binding(),
        "acceptance": {
            "named_receipt_exists": True,
            "promotion_status": "rnd_non_promoted",
            "closed_release_outcome": None,
            "release_claim": False,
            "contracts_frozen": False,
        },
    }
    checked = validate_pcpr_033_outer_receipt(payload)
    assert checked["valid"] is True
    assert checked["promotion_status"] == "rnd_non_promoted"
    assert checked["closed_release_outcome"] is None
    assert checked["release_claim"] is False
    assert checked["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_outer_receipt_validator_rejects_closed_release_outcome() -> None:
    sections = current_head_pcpr_033_receipt_sections()
    forged = {
        "task_id": PCPR_033_TASK_ID,
        "status": "implemented",
        "qualification_verdict": dict(sections["qualification_verdict"]),
        "acceptance": {
            "promotion_status": "release_candidate_qualified",
            "closed_release_outcome": "release_candidate_qualified",
            "release_claim": True,
        },
    }
    with pytest.raises(FabricatedEndpointSuccessRemovalError, match="closed PCPR release"):
        validate_pcpr_033_outer_receipt(forged)
    forged_verdict = {
        "task_id": PCPR_033_TASK_ID,
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
    with pytest.raises(FabricatedEndpointSuccessRemovalError, match="closed PCPR release"):
        validate_pcpr_033_outer_receipt(forged_verdict)


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
    verdict = qualify_fabricated_endpoint_success_removal(probes=probes)
    assert verdict.promotion_status == "typed_blocked"
    assert verdict.closed_release_outcome is None
    assert verdict.live_endpoint_qualified is False
