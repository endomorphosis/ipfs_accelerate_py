"""PCPR-039 fail-closed live model/provider execution qualification."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.model_provider_execution_qualification import (
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    HERMETIC_CANDIDATE_SUITES,
    PCPR_039_GOAL_ID,
    PCPR_039_TASK_ID,
    MODEL_EVALUATOR_INTERFACE,
    ModelProviderExecutionQualificationError,
    current_head_model_provider_probes,
    current_head_pcpr_039_current_tree_binding,
    current_head_pcpr_039_receipt_promotion,
    current_head_pcpr_039_receipt_sections,
    qualify_current_head_model_provider,
    qualify_model_provider_execution,
    validate_pcpr_039_outer_receipt,
)
from ipfs_accelerate_py.assurance.model_provider_execution import (
    CANARY_EXPECTED,
    CANARY_PROMPT,
    FIXTURE_EMBEDDING,
    FIXTURE_EXPECTED,
    FIXTURE_PROMPT,
    canary_digest,
    canary_embedding,
    qualify_live_model_provider_execution,
)
from ipfs_accelerate_py.assurance.hardware_capability_ladder import (
    from_live_model_provider_execution,
    production_authorized,
)
from ipfs_accelerate_py.kit.hardware_kit import HardwareKit


def test_closed_vocabularies_match_pcpr_039_requirements() -> None:
    assert PCPR_039_TASK_ID == "PCPR-039"
    assert PCPR_039_GOAL_ID == "PCPR-G430"
    assert MODEL_EVALUATOR_INTERFACE == "ModelProviderExecutionQualification@1"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_model_provider_execution_qualification.py"
    )
    assert canary_embedding(FIXTURE_PROMPT) == FIXTURE_EMBEDDING
    assert canary_digest(FIXTURE_PROMPT) == FIXTURE_EXPECTED
    assert canary_digest(CANARY_PROMPT) == CANARY_EXPECTED


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_model_provider()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.live_model_provider_execution_qualified is True
    assert verdict.production_authorized is False
    assert verdict.hardware_ladder_qualified is False
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.live_cuda_qualified is False
    assert verdict.live_cuda_evidence_kind == "unavailable"
    assert verdict.live_model_provider_qualified is True
    assert verdict.live_model_provider_evidence_kind == "measured"
    assert verdict.live_llamacpp_qualified is True
    assert verdict.live_llamacpp_evidence_kind == "measured"
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.verdict_cid.startswith("baguqeera")
    section = current_head_pcpr_039_receipt_promotion()
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["live_model_provider_execution_qualified"] is True
    assert section["production_authorized"] is False
    assert section["live_cuda_qualified"] is False
    assert section["live_llamacpp_qualified"] is True
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_live_model_provider_canary_is_not_production_authorized() -> None:
    report = qualify_live_model_provider_execution(test_level="comprehensive")
    assert report["model_provider_execution_qualified"] is True
    assert report["tests_passed"] is True
    assert report["live"] is True
    assert report["production_authorized"] is False
    assert report["qualified"] is False
    assert report["model_compatible"] is True
    assert report["simulated"] is False
    assert report["torch_not_used"] is True
    assert report["transformers_not_used"] is True
    assert report["live_llamacpp_qualified"] is True
    ladder = from_live_model_provider_execution(canary_passed=True)
    assert production_authorized(ladder) is False
    assert ladder["live"] is True
    assert ladder["model_provider_execution_qualified"] is True
    assert ladder["model_compatible"] is True
    assert ladder["qualified"] is False
    kit = HardwareKit()
    basic = kit._test_model_provider("basic")
    comprehensive = kit._test_model_provider("comprehensive")
    assert basic["production_authorized"] is False
    assert comprehensive["production_authorized"] is False
    assert basic["model_provider_execution_qualified"] is True
    assert comprehensive["model_provider_execution_qualified"] is True
    assert comprehensive["live"] is True


def test_current_head_live_probes_are_measured() -> None:
    probes = {item.probe_id: item for item in current_head_model_provider_probes()}
    assert probes["canonical_model_provider_execution_module"].present is True
    assert probes["ladder_exposes_live_model_provider_execution"].present is True
    assert probes["hardware_kit_uses_live_model_provider_canary"].present is True
    assert probes["model_provider_identity"].live is True
    assert probes["model_inference"].live is True
    assert probes["model_output_validation"].present is True
    assert probes["model_repetition"].present is True
    assert probes["model_cancellation"].present is True
    assert probes["model_timeout"].present is True
    assert probes["model_cleanup"].present is True
    assert probes["model_resource_admission_fail_closed"].present is True
    assert probes["llamacpp_identity"].live is True
    assert probes["llamacpp_inference"].live is True
    assert probes["llamacpp_repetition"].present is True
    assert probes["production_authorized_not_granted"].present is True
    assert probes["torch_is_not_qualification"].present is True
    assert probes["live_cuda_qualification"].evidence_kind == "unavailable"
    assert probes["live_cuda_qualification"].present is None
    for item in probes.values():
        assert item.simulated_represented_as_live is False
        assert item.evidence_kind != "simulated"


def test_current_head_receipt_sections_are_rnd_non_promoted() -> None:
    sections = current_head_pcpr_039_receipt_sections()
    assert sections["promotion_status"] == "rnd_non_promoted"
    assert sections["closed_release_outcome"] is None
    assert sections["release_claim"] is False
    assert sections["contracts_frozen"] is False
    assert sections["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert sections["qualification_prerequisite"]["task_id"] == "PCPR-038"
    assert sections["model_provider_execution"][
        "live_model_provider_execution_qualified"
    ] is True
    assert sections["model_provider_execution"]["production_authorized"] is False
    assert sections["negative_results"]["closed_release_outcome_not_emitted"] is True
    assert sections["negative_results"]["direct_database_bypass_not_used"] is True
    assert sections["negative_results"]["torch_is_not_model_qualification"] is True
    assert sections["negative_results"]["production_authorized_not_granted"] is True


def test_outer_receipt_validator_accepts_generated_non_promotion_receipt() -> None:
    sections = current_head_pcpr_039_receipt_sections()
    payload = {
        "task_id": PCPR_039_TASK_ID,
        "status": "implemented",
        "completion_authoritative": False,
        "release_claim": False,
        "qualification_verdict": sections["qualification_verdict"],
        "qualification_prerequisite": sections["qualification_prerequisite"],
        "current_tree_binding": current_head_pcpr_039_current_tree_binding(),
        "acceptance": {
            "named_receipt_exists": True,
            "promotion_status": "rnd_non_promoted",
            "closed_release_outcome": None,
            "release_claim": False,
            "contracts_frozen": False,
        },
    }
    checked = validate_pcpr_039_outer_receipt(payload)
    assert checked["valid"] is True
    assert checked["promotion_status"] == "rnd_non_promoted"
    assert checked["closed_release_outcome"] is None
    assert checked["release_claim"] is False
    assert checked["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_outer_receipt_validator_rejects_closed_release_outcome() -> None:
    sections = current_head_pcpr_039_receipt_sections()
    forged = {
        "task_id": PCPR_039_TASK_ID,
        "status": "implemented",
        "qualification_verdict": dict(sections["qualification_verdict"]),
        "acceptance": {
            "promotion_status": "release_candidate_qualified",
            "closed_release_outcome": "release_candidate_qualified",
            "release_claim": True,
        },
    }
    with pytest.raises(
        ModelProviderExecutionQualificationError, match="closed PCPR release"
    ):
        validate_pcpr_039_outer_receipt(forged)


def test_qualify_rejects_duckdb_write() -> None:
    probes = current_head_model_provider_probes()
    with pytest.raises(ModelProviderExecutionQualificationError, match="DuckDB"):
        qualify_model_provider_execution(
            probes=probes,
            duckdb_or_quack_state_written=True,
        )


def test_qualify_rejects_cuda_and_production_claims() -> None:
    probes = current_head_model_provider_probes()
    with pytest.raises(ModelProviderExecutionQualificationError, match="CUDA"):
        qualify_model_provider_execution(probes=probes, live_cuda_qualified=True)
    with pytest.raises(
        ModelProviderExecutionQualificationError, match="production_authorized"
    ):
        qualify_model_provider_execution(
            probes=probes, production_authorized_claim=True
        )


def test_current_tree_binding_is_measured_and_not_a_release() -> None:
    binding = current_head_pcpr_039_current_tree_binding()
    assert binding["evidence_kind"] == "measured"
    assert binding["origin_main_is_ancestor"] is True
    assert binding["accelerator_origin_main_is_ancestor"] is True
    assert binding["accelerator_post_change_commit"].startswith("pending")
    assert binding["outer_repository"] == "endomorphosis/lift_coding"
