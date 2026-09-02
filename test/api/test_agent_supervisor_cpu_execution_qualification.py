"""PCPR-037 fail-closed live CPU execution qualification."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.cpu_execution_qualification import (
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    HERMETIC_CANDIDATE_SUITES,
    PCPR_037_GOAL_ID,
    PCPR_037_TASK_ID,
    CPU_EVALUATOR_INTERFACE,
    CpuExecutionQualificationError,
    current_head_cpu_probes,
    current_head_pcpr_037_current_tree_binding,
    current_head_pcpr_037_receipt_promotion,
    current_head_pcpr_037_receipt_sections,
    qualify_cpu_execution,
    qualify_current_head_cpu,
    validate_pcpr_037_outer_receipt,
)
from ipfs_accelerate_py.assurance.cpu_execution import (
    CANARY_EXPECTED,
    CANARY_N,
    FIXTURE_EXPECTED,
    FIXTURE_N,
    cpu_integer_kernel,
    qualify_live_cpu_execution,
)
from ipfs_accelerate_py.assurance.hardware_capability_ladder import (
    from_live_cpu_execution,
    production_authorized,
)
from ipfs_accelerate_py.kit.hardware_kit import HardwareKit


def test_closed_vocabularies_match_pcpr_037_requirements() -> None:
    assert PCPR_037_TASK_ID == "PCPR-037"
    assert PCPR_037_GOAL_ID == "PCPR-G430"
    assert CPU_EVALUATOR_INTERFACE == "CpuExecutionQualification@1"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_cpu_execution_qualification.py"
    )
    assert cpu_integer_kernel(FIXTURE_N) == FIXTURE_EXPECTED
    assert cpu_integer_kernel(CANARY_N) == CANARY_EXPECTED


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_cpu()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.live_cpu_execution_qualified is True
    assert verdict.production_authorized is False
    assert verdict.hardware_ladder_qualified is False
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.live_cuda_qualified is False
    assert verdict.live_cuda_evidence_kind == "unavailable"
    assert verdict.live_model_provider_qualified is False
    assert verdict.live_model_provider_evidence_kind == "unavailable"
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.verdict_cid.startswith("baguqeera")
    section = current_head_pcpr_037_receipt_promotion()
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["live_cpu_execution_qualified"] is True
    assert section["production_authorized"] is False
    assert section["live_cuda_qualified"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_live_cpu_canary_is_not_production_authorized() -> None:
    report = qualify_live_cpu_execution(test_level="comprehensive")
    assert report["cpu_execution_qualified"] is True
    assert report["tests_passed"] is True
    assert report["live"] is True
    assert report["production_authorized"] is False
    assert report["qualified"] is False
    assert report["model_compatible"] is None
    assert report["simulated"] is False
    ladder = from_live_cpu_execution(canary_passed=True)
    assert production_authorized(ladder) is False
    assert ladder["live"] is True
    assert ladder["cpu_execution_qualified"] is True
    assert ladder["qualified"] is False
    kit = HardwareKit()
    basic = kit._test_cpu("basic")
    comprehensive = kit._test_cpu("comprehensive")
    assert basic["production_authorized"] is False
    assert comprehensive["production_authorized"] is False
    assert basic["cpu_execution_qualified"] is True
    assert comprehensive["cpu_execution_qualified"] is True
    assert comprehensive["live"] is True


def test_current_head_live_probes_are_measured() -> None:
    probes = {item.probe_id: item for item in current_head_cpu_probes()}
    assert probes["canonical_cpu_execution_module"].present is True
    assert probes["ladder_exposes_live_cpu_execution"].present is True
    assert probes["hardware_kit_uses_live_cpu_canary"].present is True
    assert probes["cpu_identity"].live is True
    assert probes["cpu_compute_kernel"].live is True
    assert probes["cpu_output_validation"].present is True
    assert probes["cpu_repetition"].present is True
    assert probes["cpu_cancellation"].present is True
    assert probes["cpu_timeout"].present is True
    assert probes["cpu_cleanup"].present is True
    assert probes["cpu_resource_admission_fail_closed"].present is True
    assert probes["production_authorized_not_granted"].present is True
    assert probes["live_cuda_qualification"].evidence_kind == "unavailable"
    assert probes["live_cuda_qualification"].present is None
    assert probes["live_model_provider_qualification"].evidence_kind == "unavailable"
    for item in probes.values():
        assert item.simulated_represented_as_live is False
        assert item.evidence_kind != "simulated"


def test_current_head_receipt_sections_are_rnd_non_promoted() -> None:
    sections = current_head_pcpr_037_receipt_sections()
    assert sections["promotion_status"] == "rnd_non_promoted"
    assert sections["closed_release_outcome"] is None
    assert sections["release_claim"] is False
    assert sections["contracts_frozen"] is False
    assert sections["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert sections["qualification_prerequisite"]["task_id"] == "PCPR-036"
    assert sections["cpu_execution"]["live_cpu_execution_qualified"] is True
    assert sections["cpu_execution"]["production_authorized"] is False
    assert sections["negative_results"]["closed_release_outcome_not_emitted"] is True
    assert sections["negative_results"]["direct_database_bypass_not_used"] is True
    assert sections["negative_results"]["live_cuda_not_claimed"] is True
    assert sections["negative_results"]["production_authorized_not_granted"] is True


def test_outer_receipt_validator_accepts_generated_non_promotion_receipt() -> None:
    sections = current_head_pcpr_037_receipt_sections()
    payload = {
        "task_id": PCPR_037_TASK_ID,
        "status": "implemented",
        "completion_authoritative": False,
        "release_claim": False,
        "qualification_verdict": sections["qualification_verdict"],
        "qualification_prerequisite": sections["qualification_prerequisite"],
        "current_tree_binding": current_head_pcpr_037_current_tree_binding(),
        "acceptance": {
            "named_receipt_exists": True,
            "promotion_status": "rnd_non_promoted",
            "closed_release_outcome": None,
            "release_claim": False,
            "contracts_frozen": False,
        },
    }
    checked = validate_pcpr_037_outer_receipt(payload)
    assert checked["valid"] is True
    assert checked["promotion_status"] == "rnd_non_promoted"
    assert checked["closed_release_outcome"] is None
    assert checked["release_claim"] is False
    assert checked["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_outer_receipt_validator_rejects_closed_release_outcome() -> None:
    sections = current_head_pcpr_037_receipt_sections()
    forged = {
        "task_id": PCPR_037_TASK_ID,
        "status": "implemented",
        "qualification_verdict": dict(sections["qualification_verdict"]),
        "acceptance": {
            "promotion_status": "release_candidate_qualified",
            "closed_release_outcome": "release_candidate_qualified",
            "release_claim": True,
        },
    }
    with pytest.raises(CpuExecutionQualificationError, match="closed PCPR release"):
        validate_pcpr_037_outer_receipt(forged)


def test_qualify_rejects_duckdb_write() -> None:
    probes = current_head_cpu_probes()
    with pytest.raises(CpuExecutionQualificationError, match="DuckDB"):
        qualify_cpu_execution(
            probes=probes,
            duckdb_or_quack_state_written=True,
        )


def test_qualify_rejects_cuda_and_production_claims() -> None:
    probes = current_head_cpu_probes()
    with pytest.raises(CpuExecutionQualificationError, match="CUDA"):
        qualify_cpu_execution(probes=probes, live_cuda_qualified=True)
    with pytest.raises(CpuExecutionQualificationError, match="model/provider"):
        qualify_cpu_execution(probes=probes, live_model_provider_qualified=True)
    with pytest.raises(CpuExecutionQualificationError, match="production_authorized"):
        qualify_cpu_execution(probes=probes, production_authorized_claim=True)


def test_current_tree_binding_is_measured_and_not_a_release() -> None:
    binding = current_head_pcpr_037_current_tree_binding()
    assert binding["evidence_kind"] == "measured"
    assert binding["origin_main_is_ancestor"] is True
    assert binding["accelerator_origin_main_is_ancestor"] is True
    assert binding["accelerator_post_change_commit"].startswith("pending")
    assert binding["outer_repository"] == "endomorphosis/lift_coding"
