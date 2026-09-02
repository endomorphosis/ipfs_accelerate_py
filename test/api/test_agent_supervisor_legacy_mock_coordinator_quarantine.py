"""PCPR-030 fail-closed Accelerate legacy mock-coordinator quarantine."""

from __future__ import annotations

import tempfile

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.legacy_mock_coordinator_quarantine import (
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    HERMETIC_CANDIDATE_SUITES,
    PCPR_030_GOAL_ID,
    PCPR_030_TASK_ID,
    QUARANTINE_INTERFACE,
    LegacyMockCoordinatorQuarantineError,
    QuarantineProbe,
    current_head_pcpr_030_current_tree_binding,
    current_head_pcpr_030_receipt_promotion,
    current_head_pcpr_030_receipt_sections,
    current_head_quarantine_probes,
    qualify_current_head_quarantine,
    qualify_legacy_mock_coordinator_quarantine,
    validate_pcpr_030_outer_receipt,
)
from ipfs_accelerate_py.compatibility.simulation.legacy_mock_coordinator import (
    LegacyMockCoordinatorError,
    MockWorker,
    UnavailableWorker,
    instantiate_mock_worker,
    load_ordinary_runtime_worker,
)
from ipfs_accelerate_py.datasets_integration.workflow import WorkflowCoordinator
from ipfs_accelerate_py.mcp_server.tools.workflow.native_workflow_tools import (
    workflow_coordinator_submit_task,
)


def _unavailable_probes() -> tuple[QuarantineProbe, ...]:
    return (
        QuarantineProbe(
            probe_id="accelerate_source_tree",
            present=None,
            evidence_kind="unavailable",
            live=False,
            simulated_represented_as_live=False,
            reason="fixture unavailable",
        ),
    )


def test_closed_vocabularies_match_pcpr_030_requirements() -> None:
    assert PCPR_030_TASK_ID == "PCPR-030"
    assert PCPR_030_GOAL_ID == "PCPR-G410"
    assert QUARANTINE_INTERFACE == "LegacyMockCoordinatorQuarantine@1"
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_legacy_mock_coordinator_quarantine.py"
    )


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_quarantine()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.ordinary_runtime_instantiates_mock_worker is False
    assert verdict.mock_worker_quarantined is True
    assert verdict.workflow_coordinator_requires_explicit_simulation is True
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.live_cuda_qualified is False
    assert verdict.live_cuda_evidence_kind == "unavailable"
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.verdict_cid.startswith("baguqeera")
    section = current_head_pcpr_030_receipt_promotion()
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["live_cuda_qualified"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_ordinary_runtime_worker_is_typed_unavailable_not_mock() -> None:
    worker = load_ordinary_runtime_worker()
    assert isinstance(worker, UnavailableWorker)
    assert not isinstance(worker, MockWorker)
    assert worker.live is False
    assert worker.simulated is False
    hardware = worker.test_hardware()
    assert hardware["cuda"] is None
    assert hardware["origin"] == "unavailable"
    assert hardware["outcome"] == "Unavailable"
    assert hardware["live"] is False
    with pytest.raises(LegacyMockCoordinatorError, match="MockWorker"):
        load_ordinary_runtime_worker({"worker": MockWorker()})


def test_mock_worker_requires_explicit_simulation_and_is_not_live() -> None:
    with pytest.raises(LegacyMockCoordinatorError):
        instantiate_mock_worker()
    simulated = instantiate_mock_worker(explicit_simulation=True)
    assert isinstance(simulated, MockWorker)
    assert simulated.live is False
    assert simulated.origin == "simulated"
    hardware = simulated.test_hardware()
    assert hardware["cuda"] is not True
    assert hardware["outcome"] == "Simulated"
    assert hardware["live"] is False
    assert hardware["production_authorized"] is False


def test_workflow_coordinator_requires_explicit_simulation() -> None:
    with pytest.raises(LegacyMockCoordinatorError):
        WorkflowCoordinator()
    with tempfile.TemporaryDirectory(prefix="pcpr-030-test-") as tmp:
        coordinator = WorkflowCoordinator(
            {"cache_dir": tmp, "explicit_simulation": True}
        )
        assert coordinator.live is False
        assert coordinator.origin == "simulated"
        status = coordinator.get_status()
        assert status["live"] is False
        assert status["origin"] == "simulated"
        assert status["outcome"] == "Simulated"


def test_mcp_ordinary_path_is_typed_unavailable() -> None:
    result = workflow_coordinator_submit_task(
        task_id="pcpr-030-test",
        task_type="quarantine",
        data={},
    )
    assert result["status"] == "unavailable"
    assert result["success"] is False
    assert result["outcome"] == "Unavailable"
    assert result["live"] is False
    assert result["code"] == "legacy_mock_coordinator_quarantined"


def test_current_head_probes_are_measured_not_live() -> None:
    probes = {item.probe_id: item for item in current_head_quarantine_probes()}
    assert probes["ordinary_runtime_constructor"].present is False
    assert probes["ordinary_runtime_constructor"].evidence_kind == "measured"
    assert probes["simulation_namespace_mock_worker"].present is True
    assert probes["workflow_coordinator_gate"].present is True
    assert probes["mcp_submit_ordinary_path"].present is True
    assert probes["live_cuda_qualification"].evidence_kind == "unavailable"
    assert probes["live_cuda_qualification"].present is None
    for item in probes.values():
        assert item.live is False
        assert item.simulated_represented_as_live is False
        assert item.evidence_kind != "measured_live"
        assert item.evidence_kind != "simulated"


def test_current_head_receipt_sections_are_rnd_non_promoted() -> None:
    sections = current_head_pcpr_030_receipt_sections()
    assert sections["promotion_status"] == "rnd_non_promoted"
    assert sections["closed_release_outcome"] is None
    assert sections["release_claim"] is False
    assert sections["contracts_frozen"] is False
    assert sections["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert sections["qualification_prerequisite"]["task_id"] == "PCPR-003"
    assert sections["quarantine"]["mock_worker_quarantined"] is True
    assert sections["quarantine"]["ordinary_runtime_instantiates_mock_worker"] is False
    assert sections["negative_results"]["closed_release_outcome_not_emitted"] is True
    assert sections["negative_results"]["direct_database_bypass_not_used"] is True
    assert sections["negative_results"]["live_cuda_not_claimed"] is True


def test_outer_receipt_validator_accepts_generated_non_promotion_receipt() -> None:
    sections = current_head_pcpr_030_receipt_sections()
    payload = {
        "task_id": PCPR_030_TASK_ID,
        "status": "implemented",
        "completion_authoritative": False,
        "release_claim": False,
        "qualification_verdict": sections["qualification_verdict"],
        "qualification_prerequisite": sections["qualification_prerequisite"],
        "current_tree_binding": current_head_pcpr_030_current_tree_binding(),
        "acceptance": {
            "named_receipt_exists": True,
            "promotion_status": "rnd_non_promoted",
            "closed_release_outcome": None,
            "release_claim": False,
            "contracts_frozen": False,
        },
    }
    checked = validate_pcpr_030_outer_receipt(payload)
    assert checked["valid"] is True
    assert checked["promotion_status"] == "rnd_non_promoted"
    assert checked["closed_release_outcome"] is None
    assert checked["release_claim"] is False
    assert checked["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_outer_receipt_validator_rejects_closed_release_outcome() -> None:
    sections = current_head_pcpr_030_receipt_sections()
    forged = {
        "task_id": PCPR_030_TASK_ID,
        "status": "implemented",
        "qualification_verdict": dict(sections["qualification_verdict"]),
        "acceptance": {
            "promotion_status": "release_candidate_qualified",
            "closed_release_outcome": "release_candidate_qualified",
            "release_claim": True,
        },
    }
    with pytest.raises(LegacyMockCoordinatorQuarantineError, match="closed PCPR release"):
        validate_pcpr_030_outer_receipt(forged)
    forged_verdict = {
        "task_id": PCPR_030_TASK_ID,
        "status": "implemented",
        "qualification_verdict": {
            **sections["qualification_verdict"],
            "closed_release_outcome": "non_promoted_live_compute_gap",
        },
    }
    with pytest.raises(LegacyMockCoordinatorQuarantineError, match="must be null"):
        validate_pcpr_030_outer_receipt(forged_verdict)
    forged_cuda = {
        "task_id": PCPR_030_TASK_ID,
        "status": "implemented",
        "qualification_verdict": {
            **sections["qualification_verdict"],
            "live_cuda_qualified": True,
        },
    }
    with pytest.raises(LegacyMockCoordinatorQuarantineError, match="live CUDA"):
        validate_pcpr_030_outer_receipt(forged_cuda)


def test_duckdb_or_quack_write_is_rejected() -> None:
    with pytest.raises(LegacyMockCoordinatorQuarantineError, match="DuckDB or Quack"):
        qualify_legacy_mock_coordinator_quarantine(
            probes=_unavailable_probes(),
            duckdb_or_quack_state_written=True,
        )


def test_unavailable_probes_are_typed_unavailable_not_a_release() -> None:
    verdict = qualify_legacy_mock_coordinator_quarantine(probes=_unavailable_probes())
    assert verdict.promotion_status == "typed_blocked" or verdict.promotion_status == "typed_unavailable"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.live_cuda_qualified is False


def test_current_tree_binding_is_measured_and_not_a_release() -> None:
    binding = current_head_pcpr_030_current_tree_binding()
    assert binding["evidence_kind"] == "measured"
    assert binding["origin_main_is_ancestor"] is True
    assert binding["accelerator_origin_main_is_ancestor"] is True
    assert binding["accelerator_post_change_commit"].startswith("pending")
    assert "release_candidate_qualified" not in binding["outer_subject"]
