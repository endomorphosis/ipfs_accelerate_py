"""DCR-083: single content-addressed authority projection for statuses."""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.objectives.repair_authority_projection import (
    GOAL_COMPLETION_INTERFACE,
    REPAIR_AUTHORITY_PROJECTION_INTERFACE,
    GoalAuthorityStatus,
    TaskAuthorityStatus,
    build_repair_authority_projection,
    derive_goal_status,
    derive_task_status,
    materialize_authority_projection,
)


def test_interfaces() -> None:
    assert REPAIR_AUTHORITY_PROJECTION_INTERFACE == "RepairAuthorityProjection@1"
    assert GOAL_COMPLETION_INTERFACE == "GoalCompletion@1"


def test_board_completed_without_evidence_reopens() -> None:
    status = derive_task_status(
        {"task_id": "DCR-081", "status": "completed"},
        evidence={
            "require_evidence_for_completion": True,
            "required_receipts": ("admission", "validation"),
            "present_receipts": (),
        },
    )
    assert status == TaskAuthorityStatus.REOPENED.value


def test_completed_with_evidence_stays_completed() -> None:
    status = derive_task_status(
        {"task_id": "DCR-080", "status": "completed"},
        evidence={
            "admission_ok": True,
            "validation_ok": True,
            "publication_ok": True,
        },
    )
    assert status == TaskAuthorityStatus.COMPLETED.value


def test_ready_and_blocked_projections() -> None:
    assert (
        derive_task_status(
            {"task_id": "DCR-083", "status": "todo", "dependencies_satisfied": True}
        )
        == TaskAuthorityStatus.READY.value
    )
    assert (
        derive_task_status({"task_id": "DCR-X", "status": "todo", "blocked_by": ["DCR-Y"]})
        == TaskAuthorityStatus.BLOCKED.value
    )


def test_goal_derivation_and_reopen() -> None:
    assert (
        derive_goal_status(
            {"goal_id": "G1", "task_ids": ("T1", "T2")},
            task_statuses={"T1": "completed", "T2": "completed"},
        )
        == GoalAuthorityStatus.COMPLETED.value
    )
    assert (
        derive_goal_status(
            {"goal_id": "G1", "task_ids": ("T1", "T2")},
            task_statuses={"T1": "completed", "T2": "reopened"},
        )
        == GoalAuthorityStatus.REOPENED.value
    )


def test_build_projection_marks_reopened_and_disables_board_authority() -> None:
    projection = build_repair_authority_projection(
        tasks=(
            {
                "task_id": "DCR-080",
                "status": "completed",
                "evidence": {
                    "admission_ok": True,
                    "validation_ok": True,
                    "publication_ok": True,
                },
            },
            {
                "task_id": "DCR-081",
                "status": "completed",
                "evidence": {
                    "require_evidence_for_completion": True,
                    "required_receipts": ("admission",),
                    "present_receipts": (),
                },
            },
            {"task_id": "DCR-083", "status": "ready"},
        ),
        goals=(
            {
                "goal_id": "DCR-G090",
                "task_ids": ("DCR-080", "DCR-081", "DCR-083"),
            },
        ),
    )
    assert projection.task_statuses["DCR-080"] == "completed"
    assert projection.task_statuses["DCR-081"] == "reopened"
    assert "DCR-081" in projection.reopened_task_ids
    assert projection.independent_board_authority is False
    assert projection.runtime_model_calls == 0
    assert "board_not_independent" in projection.reason_codes


def test_materialize_authority_projection(tmp_path: Path) -> None:
    dest = tmp_path / "authority-projection.json"
    payload = materialize_authority_projection(destination=dest)
    assert dest.is_file()
    assert payload["runtime_model_calls"] == 0
    assert payload["independent_board_authority"] is False
    assert payload["interface"] == REPAIR_AUTHORITY_PROJECTION_INTERFACE
