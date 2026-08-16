"""ENABLE-RPR admission unit tests."""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.sca_rpr_admission import (
    AdmissionRejection,
    AdmittedTargetPacket,
    admit_implement_task,
    assert_llm_implement_allowed,
    write_readiness_receipt,
)


def test_unbound_implement_is_rejected() -> None:
    result = admit_implement_task(
        {"task_id": "SCA-X", "snapshot_id": "snap:1"},
        current_snapshot_id="snap:1",
    )
    assert isinstance(result, AdmissionRejection)
    assert "missing_counterexample" in result.reason_codes
    assert result.model_write_authority == "reject"


def test_snapshot_mismatch_is_rejected() -> None:
    result = admit_implement_task(
        {
            "task_id": "SCA-X",
            "snapshot_id": "snap:old",
            "counterexample_id": "cex:1",
            "reproof_command": "pytest -q",
        },
        current_snapshot_id="snap:new",
    )
    assert isinstance(result, AdmissionRejection)
    assert "snapshot_mismatch" in result.reason_codes


def test_bound_packet_is_admitted() -> None:
    result = admit_implement_task(
        {
            "task_id": "SCA-X",
            "snapshot_id": "snap:1",
            "counterexample_id": "cex:1",
            "reproof_command": "pytest -q",
            "write_paths": ["external/ipfs_accelerate/x.py"],
        },
        current_snapshot_id="snap:1",
    )
    assert isinstance(result, AdmittedTargetPacket)
    assert result.llm_output == "proposal_only"
    assert_llm_implement_allowed(result)


def test_write_readiness_receipt(tmp_path: Path) -> None:
    path = tmp_path / "rpr_admission_ready.json"
    receipt = write_readiness_receipt(path, doctor_bridge_ok=True, ready=True)
    assert path.is_file()
    assert receipt["ready"] is True
    assert receipt["policy"]["unbound_implement"] == "reject"
