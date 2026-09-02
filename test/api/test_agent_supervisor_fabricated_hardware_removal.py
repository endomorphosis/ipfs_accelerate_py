"""PCPR-031 fail-closed Accelerate fabricated hardware removal."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.fabricated_hardware_removal import (
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    HERMETIC_CANDIDATE_SUITES,
    PCPR_031_GOAL_ID,
    PCPR_031_TASK_ID,
    REMOVAL_INTERFACE,
    FabricatedHardwareRemovalError,
    RemovalProbe,
    current_head_pcpr_031_current_tree_binding,
    current_head_pcpr_031_receipt_promotion,
    current_head_pcpr_031_receipt_sections,
    current_head_removal_probes,
    qualify_current_head_removal,
    qualify_fabricated_hardware_removal,
    validate_pcpr_031_outer_receipt,
)
from ipfs_accelerate_py.assurance.hardware_capability_ladder import (
    LADDER_RUNGS,
    from_device_visibility,
    from_package_import,
    from_platform_presence,
    production_authorized,
)
from ipfs_accelerate_py.compatibility.simulation.fabricated_hardware import (
    FabricatedHardwareError,
    MockHardwareDetection,
    UnavailableHardwareDetection,
    instantiate_mock_hardware_detection,
    load_ordinary_hardware_detection,
    create_simulated_cuda_implementation,
)
from ipfs_accelerate_py.kit.hardware_kit import HardwareKit


def test_closed_vocabularies_match_pcpr_031_requirements() -> None:
    assert PCPR_031_TASK_ID == "PCPR-031"
    assert PCPR_031_GOAL_ID == "PCPR-G410"
    assert REMOVAL_INTERFACE == "FabricatedHardwareRemoval@1"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert LADDER_RUNGS[-1] == "production_authorized"
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_fabricated_hardware_removal.py"
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
    assert verdict.ordinary_runtime_uses_mock_hardware is False
    assert verdict.mock_hardware_quarantined is True
    assert verdict.cuda_mock_requires_explicit_simulation is True
    assert verdict.device_visibility_implies_production_authorized is False
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.live_cuda_qualified is False
    assert verdict.live_cuda_evidence_kind == "unavailable"
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.verdict_cid.startswith("baguqeera")
    section = current_head_pcpr_031_receipt_promotion()
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["live_cuda_qualified"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_ordinary_hardware_detection_is_typed_unavailable_not_mock() -> None:
    detector = load_ordinary_hardware_detection()
    assert isinstance(detector, UnavailableHardwareDetection)
    assert not isinstance(detector, MockHardwareDetection)
    assert detector.live is False
    assert detector.simulated is False
    hardware = detector.detect_all_hardware()
    assert hardware["cuda"]["available"] is None
    assert hardware["cuda"]["origin"] == "absent"
    assert hardware["cuda"]["outcome"] == "Unavailable"
    assert hardware["cuda"]["production_authorized"] is False
    assert hardware["cuda"]["live"] is False
    with pytest.raises(FabricatedHardwareError, match="Mock hardware"):
        instantiate_mock_hardware_detection()


def test_mock_hardware_requires_explicit_simulation_and_is_not_live() -> None:
    simulated = instantiate_mock_hardware_detection(explicit_simulation=True)
    assert isinstance(simulated, MockHardwareDetection)
    assert simulated.live is False
    assert simulated.origin == "simulated"
    hardware = simulated.detect_all_hardware()
    assert hardware["cuda"]["available"] is not True
    assert hardware["cuda"]["outcome"] == "Simulated"
    assert hardware["cuda"]["live"] is False
    assert hardware["cuda"]["production_authorized"] is False


def test_cuda_mock_requires_explicit_simulation_and_is_not_live() -> None:
    with pytest.raises(FabricatedHardwareError):
        create_simulated_cuda_implementation("lm")
    _endpoint, _model, handler, _queue, _batch = create_simulated_cuda_implementation(
        "lm", explicit_simulation=True
    )
    labeled = handler("pcpr-031")
    assert labeled["live"] is False
    assert labeled["origin"] == "simulated"
    assert labeled["outcome"] == "Simulated"
    assert labeled["implementation_type"] == "MOCK"
    assert labeled["production_authorized"] is False
    from ipfs_accelerate_py.agent_supervisor.validation.fabricated_hardware_removal import (
        CUDA_UTILS_RELPATH,
        _load_source_module,
        discover_accelerate_root,
    )

    root = discover_accelerate_root()
    assert root is not None
    cuda_mod = _load_source_module(root, CUDA_UTILS_RELPATH, "pcpr031_test_cuda_utils")
    assert cuda_mod is not None
    utils = cuda_mod.cuda_utils(resources={})
    with pytest.raises(FabricatedHardwareError):
        utils.create_cuda_mock_implementation("lm")


def test_ladder_rejects_visibility_import_and_platform() -> None:
    visibility = from_device_visibility("cuda", devices=[{"name": "probe"}])
    imported = from_package_import("openvino", package="openvino")
    platform = from_platform_presence("metal", platform_name="Darwin")
    assert production_authorized(visibility) is False
    assert production_authorized(imported) is False
    assert production_authorized(platform) is False
    assert visibility["available"] is True
    assert imported["installed"] is True
    assert imported["available"] is False
    assert platform["available"] is None
    assert platform["declared"] is True


def test_detector_and_kit_never_authorize_production() -> None:
    from ipfs_accelerate_py.agent_supervisor.validation.fabricated_hardware_removal import (
        DETECTOR_RELPATH,
        _load_source_module,
        discover_accelerate_root,
    )

    root = discover_accelerate_root()
    assert root is not None
    detector_mod = _load_source_module(
        root, DETECTOR_RELPATH, "pcpr031_test_hardware_detector"
    )
    assert detector_mod is not None
    detector = detector_mod.HardwareDetector()
    assert detector.is_production_authorized("cuda") is False
    assert detector.is_production_authorized("cpu") is False
    cuda_cap = detector.get_capability("cuda")
    assert cuda_cap is not None
    assert cuda_cap.production_authorized is False
    assert cuda_cap.live is False
    kit = HardwareKit()
    cuda_info = kit.detect_cuda()
    assert cuda_info["production_authorized"] is False
    assert cuda_info["live"] is False
    metal = kit.detect_metal()
    assert metal["production_authorized"] is False
    assert metal.get("available") is not True


def test_current_head_probes_are_measured_not_live() -> None:
    probes = {item.probe_id: item for item in current_head_removal_probes()}
    assert probes["ordinary_legacy_hardware_setup"].present is True
    assert probes["ordinary_legacy_hardware_setup"].evidence_kind == "measured"
    assert probes["simulation_namespace_mock_hardware"].present is True
    assert probes["cuda_mock_ordinary_instantiation"].present is True
    assert probes["live_cuda_qualification"].evidence_kind == "unavailable"
    assert probes["live_cuda_qualification"].present is None
    for item in probes.values():
        assert item.live is False
        assert item.simulated_represented_as_live is False
        assert item.evidence_kind != "measured_live"
        assert item.evidence_kind != "simulated"


def test_current_head_receipt_sections_are_rnd_non_promoted() -> None:
    sections = current_head_pcpr_031_receipt_sections()
    assert sections["promotion_status"] == "rnd_non_promoted"
    assert sections["closed_release_outcome"] is None
    assert sections["release_claim"] is False
    assert sections["contracts_frozen"] is False
    assert sections["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert sections["qualification_prerequisite"]["task_id"] == "PCPR-030"
    assert sections["removal"]["mock_hardware_quarantined"] is True
    assert sections["removal"]["ordinary_runtime_uses_mock_hardware"] is False
    assert sections["negative_results"]["closed_release_outcome_not_emitted"] is True
    assert sections["negative_results"]["direct_database_bypass_not_used"] is True
    assert sections["negative_results"]["live_cuda_not_claimed"] is True


def test_outer_receipt_validator_accepts_generated_non_promotion_receipt() -> None:
    sections = current_head_pcpr_031_receipt_sections()
    payload = {
        "task_id": PCPR_031_TASK_ID,
        "status": "implemented",
        "completion_authoritative": False,
        "release_claim": False,
        "qualification_verdict": sections["qualification_verdict"],
        "qualification_prerequisite": sections["qualification_prerequisite"],
        "current_tree_binding": current_head_pcpr_031_current_tree_binding(),
        "acceptance": {
            "named_receipt_exists": True,
            "promotion_status": "rnd_non_promoted",
            "closed_release_outcome": None,
            "release_claim": False,
            "contracts_frozen": False,
        },
    }
    checked = validate_pcpr_031_outer_receipt(payload)
    assert checked["valid"] is True
    assert checked["promotion_status"] == "rnd_non_promoted"
    assert checked["closed_release_outcome"] is None
    assert checked["release_claim"] is False
    assert checked["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_outer_receipt_validator_rejects_closed_release_outcome() -> None:
    sections = current_head_pcpr_031_receipt_sections()
    forged = {
        "task_id": PCPR_031_TASK_ID,
        "status": "implemented",
        "qualification_verdict": dict(sections["qualification_verdict"]),
        "acceptance": {
            "promotion_status": "release_candidate_qualified",
            "closed_release_outcome": "release_candidate_qualified",
            "release_claim": True,
        },
    }
    with pytest.raises(FabricatedHardwareRemovalError, match="closed PCPR release"):
        validate_pcpr_031_outer_receipt(forged)
    forged_verdict = {
        "task_id": PCPR_031_TASK_ID,
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
    with pytest.raises(FabricatedHardwareRemovalError, match="closed PCPR release"):
        validate_pcpr_031_outer_receipt(forged_verdict)


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
    verdict = qualify_fabricated_hardware_removal(probes=probes)
    assert verdict.promotion_status == "typed_blocked"
    assert verdict.closed_release_outcome is None
    assert verdict.live_cuda_qualified is False
