"""PCPR-034 fail-closed Accelerate capability-ladder consolidation."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.capability_ladder_consolidation import (
    CLOSED_RELEASE_OUTCOMES,
    CONSOLIDATION_EVALUATOR_INTERFACE,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    DETECTOR_RELPATH,
    HERMETIC_CANDIDATE_SUITES,
    PCPR_034_GOAL_ID,
    PCPR_034_TASK_ID,
    CapabilityLadderConsolidationError,
    _load_source_module,
    current_head_consolidation_probes,
    current_head_pcpr_034_current_tree_binding,
    current_head_pcpr_034_receipt_promotion,
    current_head_pcpr_034_receipt_sections,
    discover_accelerate_root,
    qualify_capability_ladder_consolidation,
    qualify_current_head_consolidation,
    validate_pcpr_034_outer_receipt,
)
from ipfs_accelerate_py.assurance.hardware_capability_ladder import (
    LADDER_RUNGS,
    PRODUCTION_EXECUTION_CODE,
    admit_production_execution,
    from_canary,
    from_device_visibility,
    from_package_import,
    from_platform_presence,
    production_authorized,
)
from ipfs_accelerate_py.kit.hardware_kit import HardwareKit


def test_closed_vocabularies_match_pcpr_034_requirements() -> None:
    assert PCPR_034_TASK_ID == "PCPR-034"
    assert PCPR_034_GOAL_ID == "PCPR-G410"
    assert CONSOLIDATION_EVALUATOR_INTERFACE == "CapabilityLadderConsolidation@1"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert LADDER_RUNGS[-1] == "production_authorized"
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_capability_ladder_consolidation.py"
    )


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_consolidation()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.remaining_detectors_use_canonical_ladder is True
    assert verdict.production_execution_requires_production_authorized is True
    assert verdict.detection_implies_production_authorized is False
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.live_cuda_qualified is False
    assert verdict.live_cuda_evidence_kind == "unavailable"
    assert verdict.live_cpu_qualified is False
    assert verdict.live_cpu_evidence_kind == "unavailable"
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.verdict_cid.startswith("baguqeera")
    section = current_head_pcpr_034_receipt_promotion()
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["live_cuda_qualified"] is False
    assert section["live_cpu_qualified"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_detector_production_selection_is_fail_closed() -> None:
    root = discover_accelerate_root()
    assert root is not None
    detector_mod = _load_source_module(
        root, DETECTOR_RELPATH, "pcpr034_test_hardware_detector"
    )
    assert detector_mod is not None
    detector = detector_mod.HardwareDetector()
    selector = detector_mod.HardwareSelector(detector)
    hardware, reason = selector.select_hardware(
        {"supported_hardware": ["cuda", "rocm", "cpu"]},
        purpose="production",
    )
    assert hardware is None
    assert "production_authorized" in reason
    assert detector.is_production_authorized("cuda") is False
    assert detector.is_production_authorized("cpu") is False
    assert detector.get_best_production_hardware(["cuda", "cpu"]) is None
    admission = selector.admit_production_execution("cuda")
    assert admission["admitted"] is False
    assert admission["code"] == PRODUCTION_EXECUTION_CODE
    assert admission["live"] is False
    cuda_cap = detector.get_capability("cuda")
    assert cuda_cap is not None
    assert cuda_cap.production_authorized is False
    assert cuda_cap.to_ladder()["production_authorized"] is False
    assert cuda_cap.to_ladder()["consolidation_task_id"] == "PCPR-034"


def test_hardware_kit_remaining_detectors_are_ladder_consolidated() -> None:
    kit = HardwareKit()
    webgpu = kit.detect_webgpu()
    webnn = kit.detect_webnn()
    rocm = kit.detect_rocm()
    metal = kit.detect_metal()
    assert webgpu["available"] is None
    assert webnn["available"] is None
    assert webgpu["production_authorized"] is False
    assert webnn["production_authorized"] is False
    assert rocm["production_authorized"] is False
    assert metal["production_authorized"] is False
    assert metal.get("available") is not True
    info = kit.get_hardware_info(include_detailed=True)
    assert "cuda" in info.accelerators
    assert "webgpu" in info.accelerators
    assert info.accelerators["webgpu"]["available"] is None
    rec = kit.recommend_hardware("bert-base")
    assert rec["production_authorized"] is False
    assert rec["advisory"] is True
    for item in rec["recommendations"]:
        assert item.get("production_authorized") is False


def test_ladder_rejects_visibility_import_platform_and_canary() -> None:
    visibility = from_device_visibility("cuda", devices=[{"name": "probe"}])
    imported = from_package_import("openvino", package="openvino")
    platform = from_platform_presence("metal", platform_name="Darwin")
    canary = from_canary("cuda", passed=True)
    assert production_authorized(visibility) is False
    assert production_authorized(imported) is False
    assert production_authorized(platform) is False
    assert production_authorized(canary) is False
    assert canary["canary_passed"] is True
    assert canary["qualified"] is False
    assert admit_production_execution(canary)["admitted"] is False
    assert visibility["available"] is True
    assert imported["installed"] is True
    assert imported["available"] is False
    assert platform["available"] is None


def test_current_head_probes_are_measured_not_live() -> None:
    probes = {item.probe_id: item for item in current_head_consolidation_probes()}
    assert probes["canonical_ladder_module"].present is True
    assert probes["detector_uses_canonical_ladder"].present is True
    assert probes["hardware_kit_remaining_detectors"].present is True
    assert probes["mcp_basic_info_no_fabricated_false"].present is True
    assert probes["model_loader_requires_production_authorized"].present is True
    assert probes["production_selection_fail_closed"].present is True
    assert probes["live_cuda_qualification"].evidence_kind == "unavailable"
    assert probes["live_cuda_qualification"].present is None
    assert probes["live_cpu_qualification"].evidence_kind == "unavailable"
    for item in probes.values():
        assert item.live is False
        assert item.simulated_represented_as_live is False
        assert item.evidence_kind != "measured_live"
        assert item.evidence_kind != "simulated"


def test_current_head_receipt_sections_are_rnd_non_promoted() -> None:
    sections = current_head_pcpr_034_receipt_sections()
    assert sections["promotion_status"] == "rnd_non_promoted"
    assert sections["closed_release_outcome"] is None
    assert sections["release_claim"] is False
    assert sections["contracts_frozen"] is False
    assert sections["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert sections["qualification_prerequisite"]["task_id"] == "PCPR-033"
    assert sections["consolidation"]["remaining_detectors_use_canonical_ladder"] is True
    assert sections["consolidation"][
        "production_execution_requires_production_authorized"
    ] is True
    assert sections["negative_results"]["closed_release_outcome_not_emitted"] is True
    assert sections["negative_results"]["direct_database_bypass_not_used"] is True
    assert sections["negative_results"]["live_cuda_not_claimed"] is True
    assert sections["negative_results"]["detection_not_production_authorized"] is True


def test_outer_receipt_validator_accepts_generated_non_promotion_receipt() -> None:
    sections = current_head_pcpr_034_receipt_sections()
    payload = {
        "task_id": PCPR_034_TASK_ID,
        "status": "implemented",
        "completion_authoritative": False,
        "release_claim": False,
        "qualification_verdict": sections["qualification_verdict"],
        "qualification_prerequisite": sections["qualification_prerequisite"],
        "current_tree_binding": current_head_pcpr_034_current_tree_binding(),
        "acceptance": {
            "named_receipt_exists": True,
            "promotion_status": "rnd_non_promoted",
            "closed_release_outcome": None,
            "release_claim": False,
            "contracts_frozen": False,
        },
    }
    checked = validate_pcpr_034_outer_receipt(payload)
    assert checked["valid"] is True
    assert checked["promotion_status"] == "rnd_non_promoted"
    assert checked["closed_release_outcome"] is None
    assert checked["release_claim"] is False
    assert checked["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_outer_receipt_validator_rejects_closed_release_outcome() -> None:
    sections = current_head_pcpr_034_receipt_sections()
    forged = {
        "task_id": PCPR_034_TASK_ID,
        "status": "implemented",
        "qualification_verdict": dict(sections["qualification_verdict"]),
        "acceptance": {
            "promotion_status": "release_candidate_qualified",
            "closed_release_outcome": "release_candidate_qualified",
            "release_claim": True,
        },
    }
    with pytest.raises(CapabilityLadderConsolidationError, match="closed PCPR release"):
        validate_pcpr_034_outer_receipt(forged)


def test_qualify_rejects_duckdb_write() -> None:
    probes = current_head_consolidation_probes()
    with pytest.raises(CapabilityLadderConsolidationError, match="DuckDB"):
        qualify_capability_ladder_consolidation(
            probes=probes,
            duckdb_or_quack_state_written=True,
        )
