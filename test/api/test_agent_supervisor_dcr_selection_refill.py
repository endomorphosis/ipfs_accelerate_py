"""DCR-081: selection and refill consume typed repair dispositions."""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.objectives.deterministic_repair_selection import (
    DETERMINISTIC_REPAIR_SELECTION_INTERFACE,
    SelectionDisposition,
    materialize_selection_refill,
    project_repair_disposition,
    refill_from_dispositions,
    select_deterministic_repair_task,
)


def test_interfaces() -> None:
    assert DETERMINISTIC_REPAIR_SELECTION_INTERFACE == "DeterministicRepairSelection@1"


def test_project_repair_disposition_closed_vocabulary() -> None:
    assert project_repair_disposition({"status": "ready"}) == "ready"
    assert project_repair_disposition({"disposition": "abstain_review"}) == "abstain_review"
    assert (
        project_repair_disposition(
            {"status": "todo"}, evidence={"defer_capability": True}
        )
        == "defer_capability"
    )


def test_select_skips_terminal_and_orders_deterministically() -> None:
    receipt = select_deterministic_repair_task(
        (
            {"task_id": "DCR-B", "status": "ready", "priority": 2},
            {"task_id": "DCR-A", "status": "ready", "priority": 1},
            {"task_id": "DCR-DONE", "status": "completed"},
            {"task_id": "DCR-RES", "disposition": "abstain_review"},
            {"task_id": "DCR-DEF", "disposition": "defer_capability"},
        )
    )
    assert receipt.disposition is SelectionDisposition.SELECTED
    assert receipt.selected_task_id == "DCR-A"
    assert receipt.selectable == ("DCR-A", "DCR-B")
    assert "DCR-RES" in receipt.residual
    assert "DCR-DEF" in receipt.deferred
    assert "DCR-DONE" in receipt.terminal
    assert receipt.runtime_model_calls == 0
    assert receipt.grants_provider_dispatch is False


def test_empty_and_residual_only() -> None:
    empty = select_deterministic_repair_task(())
    assert empty.disposition is SelectionDisposition.EMPTY
    residual = select_deterministic_repair_task(
        ({"task_id": "DCR-R", "disposition": "abstain_review"},)
    )
    assert residual.disposition is SelectionDisposition.ABSTAIN
    deferred = select_deterministic_repair_task(
        ({"task_id": "DCR-D", "disposition": "defer_capability"},)
    )
    assert deferred.disposition is SelectionDisposition.DEFERRED


def test_refill_bounds_ready_without_provider_dispatch() -> None:
    refill = refill_from_dispositions(
        (
            {"task_id": "DCR-1", "status": "ready", "priority": 0},
            {"task_id": "DCR-2", "status": "ready", "priority": 1},
            {"task_id": "DCR-3", "status": "ready", "priority": 2},
            {"task_id": "DCR-R", "disposition": "abstain_review"},
        ),
        max_refill=2,
    )
    assert refill["ready_task_ids"] == ["DCR-1", "DCR-2"]
    assert refill["residual_task_ids"] == ["DCR-R"]
    assert refill["runtime_model_calls"] == 0
    assert refill["grants_provider_dispatch"] is False


def test_materialize_selection_refill(tmp_path: Path) -> None:
    dest = tmp_path / "selection-refill.json"
    payload = materialize_selection_refill(destination=dest)
    assert dest.is_file()
    assert payload["runtime_model_calls"] == 0
    assert payload["selection"]["disposition"] == "selected"
