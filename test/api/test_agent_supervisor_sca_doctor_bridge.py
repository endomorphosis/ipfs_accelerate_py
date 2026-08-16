"""ENABLE-DOCTOR bridge unit tests."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.sca_doctor_bridge import (
    AnalyticalAbstention,
    DoctorDisposition,
    TransformReceipt,
    map_finding,
    map_findings,
)


def test_refuted_finding_maps_to_transform_receipt() -> None:
    receipt = map_finding(
        {
            "finding_id": "f1",
            "kind": "parity_refuted",
            "snapshot_id": "snap:1",
            "path": "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/runtime.py",
        }
    )
    assert isinstance(receipt, TransformReceipt)
    assert receipt.disposition == DoctorDisposition.TRANSFORM_RECEIPT.value
    assert receipt.model_call_count == 0
    assert receipt.operator


def test_unknown_finding_maps_to_analytical_abstention() -> None:
    abstention = map_finding(
        {"finding_id": "f2", "kind": "unsupported_observation"}
    )
    assert isinstance(abstention, AnalyticalAbstention)
    assert abstention.disposition == DoctorDisposition.ANALYTICAL_ABSTENTION.value
    assert abstention.model_call_count == 0


def test_map_findings_batch_serializable() -> None:
    rows = map_findings(
        [
            {"finding_id": "a", "kind": "direct_dispatch"},
            {"finding_id": "b", "kind": "noise"},
        ]
    )
    assert len(rows) == 2
    assert rows[0]["disposition"] == DoctorDisposition.TRANSFORM_RECEIPT.value
    assert rows[1]["disposition"] == DoctorDisposition.ANALYTICAL_ABSTENTION.value
    assert all(r["model_call_count"] == 0 for r in rows)
