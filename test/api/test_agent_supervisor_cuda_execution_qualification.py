"""PCPR-038 fail-closed live CUDA execution qualification."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.cuda_execution_qualification import (
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    HERMETIC_CANDIDATE_SUITES,
    PCPR_038_GOAL_ID,
    PCPR_038_TASK_ID,
    CUDA_EVALUATOR_INTERFACE,
    CudaExecutionQualificationError,
    current_head_cuda_probes,
    current_head_pcpr_038_current_tree_binding,
    current_head_pcpr_038_receipt_promotion,
    current_head_pcpr_038_receipt_sections,
    qualify_cuda_execution,
    qualify_current_head_cuda,
    validate_pcpr_038_outer_receipt,
)
from ipfs_accelerate_py.assurance.cuda_execution import (
    CANARY_EXPECTED,
    CANARY_N,
    FIXTURE_EXPECTED,
    FIXTURE_N,
    cuda_integer_kernel,
    qualify_live_cuda_execution,
)
from ipfs_accelerate_py.assurance.hardware_capability_ladder import (
    from_live_cuda_execution,
    production_authorized,
)
from ipfs_accelerate_py.kit.hardware_kit import HardwareKit


def test_closed_vocabularies_match_pcpr_038_requirements() -> None:
    assert PCPR_038_TASK_ID == "PCPR-038"
    assert PCPR_038_GOAL_ID == "PCPR-G430"
    assert CUDA_EVALUATOR_INTERFACE == "CudaExecutionQualification@1"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_cuda_execution_qualification.py"
    )
    assert cuda_integer_kernel(FIXTURE_N) == FIXTURE_EXPECTED
    assert cuda_integer_kernel(CANARY_N) == CANARY_EXPECTED


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_cuda()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.live_cuda_execution_qualified is True
    assert verdict.production_authorized is False
    assert verdict.hardware_ladder_qualified is False
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.live_cuda_qualified is True
    assert verdict.live_cuda_evidence_kind == "measured"
    assert verdict.live_model_provider_qualified is False
    assert verdict.live_model_provider_evidence_kind == "unavailable"
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.verdict_cid.startswith("baguqeera")
    section = current_head_pcpr_038_receipt_promotion()
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["live_cuda_execution_qualified"] is True
    assert section["production_authorized"] is False
    assert section["live_model_provider_qualified"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_live_cuda_canary_is_not_production_authorized() -> None:
    report = qualify_live_cuda_execution(test_level="comprehensive")
    assert report["cuda_execution_qualified"] is True
    assert report["tests_passed"] is True
    assert report["live"] is True
    assert report["production_authorized"] is False
    assert report["qualified"] is False
    assert report["model_compatible"] is None
    assert report["simulated"] is False
    assert report["nvidia_smi_is_not_qualification"] is True
    assert report["nvcc_not_used"] is True
    assert report["torch_not_used"] is True
    ladder = from_live_cuda_execution(canary_passed=True)
    assert production_authorized(ladder) is False
    assert ladder["live"] is True
    assert ladder["cuda_execution_qualified"] is True
    assert ladder["qualified"] is False
    kit = HardwareKit()
    basic = kit._test_cuda("basic")
    comprehensive = kit._test_cuda("comprehensive")
    assert basic["production_authorized"] is False
    assert comprehensive["production_authorized"] is False
    assert basic["cuda_execution_qualified"] is True
    assert comprehensive["cuda_execution_qualified"] is True
    assert comprehensive["live"] is True
    visibility = kit.detect_cuda()
    assert visibility["live"] is False
    assert visibility["production_authorized"] is False
    assert visibility["qualified"] is False


def test_current_head_live_probes_are_measured() -> None:
    probes = {item.probe_id: item for item in current_head_cuda_probes()}
    assert probes["canonical_cuda_execution_module"].present is True
    assert probes["canonical_cuda_ptx_kernel"].present is True
    assert probes["ladder_exposes_live_cuda_execution"].present is True
    assert probes["hardware_kit_uses_live_cuda_canary"].present is True
    assert probes["cuda_identity"].live is True
    assert probes["cuda_compute_kernel"].live is True
    assert probes["cuda_output_validation"].present is True
    assert probes["cuda_repetition"].present is True
    assert probes["cuda_cancellation"].present is True
    assert probes["cuda_timeout"].present is True
    assert probes["cuda_cleanup"].present is True
    assert probes["cuda_resource_admission_fail_closed"].present is True
    assert probes["production_authorized_not_granted"].present is True
    assert probes["nvidia_smi_is_not_qualification"].present is True
    assert probes["live_model_provider_qualification"].evidence_kind == "unavailable"
    assert probes["live_model_provider_qualification"].present is None
    for item in probes.values():
        assert item.simulated_represented_as_live is False
        assert item.evidence_kind != "simulated"


def test_current_head_receipt_sections_are_rnd_non_promoted() -> None:
    sections = current_head_pcpr_038_receipt_sections()
    assert sections["promotion_status"] == "rnd_non_promoted"
    assert sections["closed_release_outcome"] is None
    assert sections["release_claim"] is False
    assert sections["contracts_frozen"] is False
    assert sections["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert sections["qualification_prerequisite"]["task_id"] == "PCPR-037"
    assert sections["cuda_execution"]["live_cuda_execution_qualified"] is True
    assert sections["cuda_execution"]["production_authorized"] is False
    assert sections["negative_results"]["closed_release_outcome_not_emitted"] is True
    assert sections["negative_results"]["direct_database_bypass_not_used"] is True
    assert sections["negative_results"]["nvidia_smi_is_not_cuda_qualification"] is True
    assert sections["negative_results"]["production_authorized_not_granted"] is True


def test_outer_receipt_validator_accepts_generated_non_promotion_receipt() -> None:
    sections = current_head_pcpr_038_receipt_sections()
    payload = {
        "task_id": PCPR_038_TASK_ID,
        "status": "implemented",
        "completion_authoritative": False,
        "release_claim": False,
        "qualification_verdict": sections["qualification_verdict"],
        "qualification_prerequisite": sections["qualification_prerequisite"],
        "current_tree_binding": current_head_pcpr_038_current_tree_binding(),
        "acceptance": {
            "named_receipt_exists": True,
            "promotion_status": "rnd_non_promoted",
            "closed_release_outcome": None,
            "release_claim": False,
            "contracts_frozen": False,
        },
    }
    checked = validate_pcpr_038_outer_receipt(payload)
    assert checked["valid"] is True
    assert checked["promotion_status"] == "rnd_non_promoted"
    assert checked["closed_release_outcome"] is None
    assert checked["release_claim"] is False
    assert checked["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_outer_receipt_validator_rejects_closed_release_outcome() -> None:
    sections = current_head_pcpr_038_receipt_sections()
    forged = {
        "task_id": PCPR_038_TASK_ID,
        "status": "implemented",
        "qualification_verdict": dict(sections["qualification_verdict"]),
        "acceptance": {
            "promotion_status": "release_candidate_qualified",
            "closed_release_outcome": "release_candidate_qualified",
            "release_claim": True,
        },
    }
    with pytest.raises(CudaExecutionQualificationError, match="closed PCPR release"):
        validate_pcpr_038_outer_receipt(forged)


def test_qualify_rejects_duckdb_write() -> None:
    probes = current_head_cuda_probes()
    with pytest.raises(CudaExecutionQualificationError, match="DuckDB"):
        qualify_cuda_execution(
            probes=probes,
            duckdb_or_quack_state_written=True,
        )


def test_qualify_rejects_model_and_production_claims() -> None:
    probes = current_head_cuda_probes()
    with pytest.raises(CudaExecutionQualificationError, match="model/provider"):
        qualify_cuda_execution(probes=probes, live_model_provider_qualified=True)
    with pytest.raises(CudaExecutionQualificationError, match="production_authorized"):
        qualify_cuda_execution(probes=probes, production_authorized_claim=True)


def test_current_tree_binding_is_measured_and_not_a_release() -> None:
    binding = current_head_pcpr_038_current_tree_binding()
    assert binding["evidence_kind"] == "measured"
    assert binding["origin_main_is_ancestor"] is True
    assert binding["accelerator_origin_main_is_ancestor"] is True
    assert binding["accelerator_post_change_commit"].startswith("pending")
    assert binding["outer_repository"] == "endomorphosis/lift_coding"
