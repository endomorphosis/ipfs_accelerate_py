"""WPD-040: Supervisor selection uses doctor/planner dispositions.

Acceptance (from the sealed WPD board):

* ``selection_idle_reason`` includes doctor/planner disposition classes
* ``provider_capacity_backoff`` remains distinct from disposition idle codes
* Scheduler priority hints prefer ``closed_deterministic`` over residual LLM

Interface: ``SelectionDispositionProjection@1``
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_disposition import (
    ImplementationDisposition,
    closed_disposition_values,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    IMPLEMENTATION_RETRY_DEFERRED_IDLE_PREFIX,
    PROVIDER_CAPACITY_BACKOFF_IDLE_REASON,
    SELECTION_DISPOSITION_IDLE_REASON_PREFIX,
    SELECTION_DISPOSITION_PROJECTION_EVIDENCE,
    SELECTION_DISPOSITION_PROJECTION_INTERFACE,
    SELECTION_DISPOSITION_PROJECTION_VERSION,
    _projection_is_quiescent_for_heartbeat_fallback,
    closed_disposition_selection_idle_reasons,
    compare_disposition_selection_priority,
    disposition_selection_idle_reason,
    disposition_selection_priority_hint,
    is_disposition_selection_idle_reason,
    is_provider_capacity_backoff_idle_reason,
    project_selection_disposition,
    rank_tasks_by_disposition_priority,
)


# ---------------------------------------------------------------------------
# Interface / closed vocabulary
# ---------------------------------------------------------------------------


def test_selection_disposition_projection_interface_identity() -> None:
    assert (
        SELECTION_DISPOSITION_PROJECTION_INTERFACE
        == "SelectionDispositionProjection@1"
    )
    assert SELECTION_DISPOSITION_PROJECTION_VERSION == 1
    assert SELECTION_DISPOSITION_PROJECTION_EVIDENCE == (
        "wpd/selection-disposition@1"
    )


def test_selection_idle_reason_includes_every_disposition_class() -> None:
    expected = frozenset(
        f"{SELECTION_DISPOSITION_IDLE_REASON_PREFIX}{value}"
        for value in closed_disposition_values()
    )
    assert closed_disposition_selection_idle_reasons() == expected
    for disposition in ImplementationDisposition:
        reason = disposition_selection_idle_reason(disposition)
        assert reason in expected
        assert is_disposition_selection_idle_reason(reason)
        assert not is_provider_capacity_backoff_idle_reason(reason)


@pytest.mark.parametrize(
    "disposition",
    list(ImplementationDisposition),
)
def test_disposition_idle_reason_wire_format(
    disposition: ImplementationDisposition,
) -> None:
    reason = disposition_selection_idle_reason(disposition)
    assert reason == f"disposition_idle:{disposition.value}"
    assert reason.startswith(SELECTION_DISPOSITION_IDLE_REASON_PREFIX)
    assert is_disposition_selection_idle_reason(reason)


def test_unknown_disposition_idle_reason_rejected() -> None:
    assert not is_disposition_selection_idle_reason("disposition_idle:")
    assert not is_disposition_selection_idle_reason(
        "disposition_idle:not_a_real_class"
    )
    assert not is_disposition_selection_idle_reason("closed_deterministic")
    assert not is_disposition_selection_idle_reason(
        PROVIDER_CAPACITY_BACKOFF_IDLE_REASON
    )
    with pytest.raises((TypeError, ValueError)):
        disposition_selection_idle_reason("not_a_real_class")


# ---------------------------------------------------------------------------
# provider_capacity_backoff remains distinct
# ---------------------------------------------------------------------------


def test_provider_capacity_backoff_remains_distinct_from_dispositions() -> None:
    assert PROVIDER_CAPACITY_BACKOFF_IDLE_REASON == "provider_capacity_backoff"
    assert is_provider_capacity_backoff_idle_reason(
        PROVIDER_CAPACITY_BACKOFF_IDLE_REASON
    )
    deferred = (
        f"{IMPLEMENTATION_RETRY_DEFERRED_IDLE_PREFIX}"
        f"{PROVIDER_CAPACITY_BACKOFF_IDLE_REASON}"
    )
    assert is_provider_capacity_backoff_idle_reason(deferred)

    for disposition in ImplementationDisposition:
        reason = disposition_selection_idle_reason(disposition)
        assert reason != PROVIDER_CAPACITY_BACKOFF_IDLE_REASON
        assert not is_provider_capacity_backoff_idle_reason(reason)
        assert not is_disposition_selection_idle_reason(
            PROVIDER_CAPACITY_BACKOFF_IDLE_REASON
        )


def test_projection_capacity_backoff_defers_residual_not_closed() -> None:
    # Mixed ready set: closed_deterministic still runs during residual capacity
    # backoff so planner/doctor work is preferred over model retries.
    mixed = project_selection_disposition(
        {
            "active_task_id": "",
            "ready_count": 2,
            "selection_idle_reason": "",
        },
        ready_task_dispositions={
            "TASK-A": ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED,
            "TASK-B": ImplementationDisposition.CLOSED_DETERMINISTIC,
        },
        provider_capacity_backoff=True,
    )
    assert mixed["selection_idle_reason"] == ""
    meta = mixed["selection_disposition_projection"]
    assert meta["interface"] == SELECTION_DISPOSITION_PROJECTION_INTERFACE
    assert meta["provider_capacity_backoff"] is True
    assert meta["residual_deferred_by_provider_capacity"] == 1
    assert meta["preferred_task_id"] == "TASK-B"
    assert meta["preferred_disposition"] == "closed_deterministic"
    hints = mixed["selection_disposition_priority_hints"]
    assert [item["task_id"] for item in hints] == ["TASK-B", "TASK-A"]
    assert hints[0]["disposition"] == "closed_deterministic"
    assert hints[1]["disposition"] == "residual_llm_authorized"

    # Residual-only ready set: capacity backoff is distinct from disposition.
    residual_only = project_selection_disposition(
        {
            "active_task_id": "",
            "ready_count": 1,
            "selection_idle_reason": "",
        },
        ready_task_dispositions={
            "TASK-A": ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED,
        },
        provider_capacity_backoff=True,
    )
    assert residual_only["selection_idle_reason"] == (
        PROVIDER_CAPACITY_BACKOFF_IDLE_REASON
    )
    assert not is_disposition_selection_idle_reason(
        residual_only["selection_idle_reason"]
    )
    assert is_provider_capacity_backoff_idle_reason(
        residual_only["selection_idle_reason"]
    )
    assert residual_only["selection_idle_reason"] != (
        disposition_selection_idle_reason("residual_llm_authorized")
    )
    assert residual_only["selection_disposition_projection"][
        "residual_deferred_by_provider_capacity"
    ] == 1


# ---------------------------------------------------------------------------
# Prefer closed_deterministic over residual LLM
# ---------------------------------------------------------------------------


def test_priority_hint_prefers_closed_deterministic_over_residual() -> None:
    closed = disposition_selection_priority_hint(
        ImplementationDisposition.CLOSED_DETERMINISTIC
    )
    residual = disposition_selection_priority_hint(
        ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED
    )
    abstain = disposition_selection_priority_hint(
        ImplementationDisposition.ABSTAIN_REVIEW
    )
    defer = disposition_selection_priority_hint(
        ImplementationDisposition.DEFER_CAPABILITY
    )
    assert closed < residual < abstain <= defer
    assert (
        compare_disposition_selection_priority(
            ImplementationDisposition.CLOSED_DETERMINISTIC,
            ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED,
        )
        == -1
    )
    assert (
        compare_disposition_selection_priority(
            ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED,
            ImplementationDisposition.CLOSED_DETERMINISTIC,
        )
        == 1
    )


def test_rank_tasks_prefers_closed_deterministic_readiness() -> None:
    ordered = rank_tasks_by_disposition_priority(
        {
            "TASK-RESIDUAL": "residual_llm_authorized",
            "TASK-CLOSED": "closed_deterministic",
            "TASK-ABSTAIN": "abstain_review",
            "TASK-DEFER": ImplementationDisposition.DEFER_CAPABILITY,
        }
    )
    assert ordered[0] == "TASK-CLOSED"
    assert ordered[1] == "TASK-RESIDUAL"
    assert set(ordered[2:]) == {"TASK-ABSTAIN", "TASK-DEFER"}


def test_rank_tasks_stable_by_task_id_on_tie() -> None:
    ordered = rank_tasks_by_disposition_priority(
        {
            "TASK-B": ImplementationDisposition.CLOSED_DETERMINISTIC,
            "TASK-A": ImplementationDisposition.CLOSED_DETERMINISTIC,
            "TASK-C": ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED,
        }
    )
    assert ordered == ["TASK-A", "TASK-B", "TASK-C"]


def test_policy_may_disable_closed_deterministic_preference() -> None:
    closed = disposition_selection_priority_hint(
        "closed_deterministic",
        prefer_closed_deterministic=False,
    )
    residual = disposition_selection_priority_hint(
        "residual_llm_authorized",
        prefer_closed_deterministic=False,
    )
    assert closed == residual
    ordered = rank_tasks_by_disposition_priority(
        {
            "TASK-RESIDUAL": "residual_llm_authorized",
            "TASK-CLOSED": "closed_deterministic",
        },
        prefer_closed_deterministic=False,
    )
    # Tie-break by task id when preference is disabled.
    assert ordered == ["TASK-CLOSED", "TASK-RESIDUAL"]


# ---------------------------------------------------------------------------
# Status projection
# ---------------------------------------------------------------------------


def test_projection_emits_disposition_idle_when_only_idle_classes_remain() -> None:
    projected = project_selection_disposition(
        {
            "active_task_id": "",
            "implementation_in_progress": False,
            "ready_count": 2,
            "selectable_ready_count": 2,
            "eligible_ready_count": 2,
            "blocked_count": 0,
            "selection_idle_reason": "",
        },
        ready_task_dispositions={
            "TASK-1": ImplementationDisposition.ABSTAIN_REVIEW,
            "TASK-2": ImplementationDisposition.DEFER_CAPABILITY,
        },
    )
    idle = projected["selection_idle_reason"]
    assert is_disposition_selection_idle_reason(idle)
    assert idle in {
        disposition_selection_idle_reason("abstain_review"),
        disposition_selection_idle_reason("defer_capability"),
    }
    # Dominant is the better (lower) priority among idle classes.
    assert idle == disposition_selection_idle_reason("abstain_review")
    meta = projected["selection_disposition_projection"]
    assert meta["ready_disposition_counts"]["abstain_review"] == 1
    assert meta["ready_disposition_counts"]["defer_capability"] == 1
    assert meta["ready_disposition_counts"]["closed_deterministic"] == 0
    assert meta["ready_disposition_counts"]["residual_llm_authorized"] == 0


def test_projection_clears_idle_reason_when_task_selected() -> None:
    projected = project_selection_disposition(
        {
            "active_task_id": "TASK-CLOSED",
            "selection_idle_reason": disposition_selection_idle_reason(
                "abstain_review"
            ),
        },
        ready_task_dispositions={
            "TASK-CLOSED": "closed_deterministic",
            "TASK-RESIDUAL": "residual_llm_authorized",
        },
    )
    assert projected["selection_idle_reason"] == ""
    assert projected["selection_disposition_projection"][
        "preferred_task_id"
    ] == "TASK-CLOSED"
    assert projected["selection_disposition_projection"][
        "preferred_disposition"
    ] == "closed_deterministic"


def test_projection_prefers_closed_deterministic_in_hints_when_mixed() -> None:
    projected = project_selection_disposition(
        {"active_task_id": "", "selection_idle_reason": ""},
        ready_task_dispositions={
            "TASK-RESIDUAL": ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED,
            "TASK-CLOSED": ImplementationDisposition.CLOSED_DETERMINISTIC,
        },
        provider_capacity_backoff=False,
    )
    # Runnable closed work present → not disposition-idle and not capacity backoff.
    assert projected["selection_idle_reason"] == ""
    hints = projected["selection_disposition_priority_hints"]
    assert hints[0]["task_id"] == "TASK-CLOSED"
    assert hints[0]["priority_hint"] < hints[1]["priority_hint"]
    meta = projected["selection_disposition_projection"]
    assert meta["preferred_task_id"] == "TASK-CLOSED"
    assert meta["preferred_disposition"] == "closed_deterministic"
    assert meta["prefer_closed_deterministic"] is True


def test_projection_accepts_disposition_mapping_payloads() -> None:
    projected = project_selection_disposition(
        ready_task_dispositions={
            "TASK-1": {"disposition": "closed_deterministic"},
            "TASK-2": {"disposition": ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED},
        }
    )
    assert [item["disposition"] for item in projected[
        "selection_disposition_priority_hints"
    ]] == ["closed_deterministic", "residual_llm_authorized"]


# ---------------------------------------------------------------------------
# Heartbeat fallback treats disposition idle as quiescent (and capacity distinct)
# ---------------------------------------------------------------------------


def _idle_projection(**overrides):
    projection = {
        "active_task_id": "",
        "implementation_in_progress": False,
        "ready_count": 0,
        "selectable_ready_count": 0,
        "eligible_ready_count": 0,
        "blocked_count": 0,
        "selection_idle_reason": "no_shard_selectable_ready_tasks",
    }
    projection.update(overrides)
    return projection


def test_heartbeat_fallback_accepts_disposition_idle_reasons() -> None:
    for disposition in ImplementationDisposition:
        reason = disposition_selection_idle_reason(disposition)
        assert _projection_is_quiescent_for_heartbeat_fallback(
            _idle_projection(
                ready_count=1,
                selectable_ready_count=1,
                eligible_ready_count=1,
                selection_idle_reason=reason,
            )
        )


def test_heartbeat_fallback_accepts_provider_capacity_backoff_distinctly() -> None:
    assert _projection_is_quiescent_for_heartbeat_fallback(
        _idle_projection(
            ready_count=1,
            selectable_ready_count=1,
            eligible_ready_count=1,
            selection_idle_reason=PROVIDER_CAPACITY_BACKOFF_IDLE_REASON,
        )
    )
    assert _projection_is_quiescent_for_heartbeat_fallback(
        _idle_projection(
            ready_count=1,
            selectable_ready_count=1,
            eligible_ready_count=1,
            selection_idle_reason=(
                f"{IMPLEMENTATION_RETRY_DEFERRED_IDLE_PREFIX}"
                f"{PROVIDER_CAPACITY_BACKOFF_IDLE_REASON}"
            ),
        )
    )


def test_heartbeat_fallback_rejects_active_work_even_with_disposition_idle() -> None:
    reason = disposition_selection_idle_reason("abstain_review")
    assert not _projection_is_quiescent_for_heartbeat_fallback(
        _idle_projection(
            active_task_id="TASK-001",
            ready_count=1,
            selection_idle_reason=reason,
        )
    )
    assert not _projection_is_quiescent_for_heartbeat_fallback(
        _idle_projection(
            implementation_in_progress=True,
            ready_count=1,
            selection_idle_reason=PROVIDER_CAPACITY_BACKOFF_IDLE_REASON,
        )
    )


def test_heartbeat_fallback_rejects_unsafe_and_empty_deferred_reasons() -> None:
    assert not _projection_is_quiescent_for_heartbeat_fallback(
        _idle_projection(selection_idle_reason="todo_read_failed")
    )
    assert not _projection_is_quiescent_for_heartbeat_fallback(
        _idle_projection(selection_idle_reason="implementation_retry_deferred:")
    )
    assert not _projection_is_quiescent_for_heartbeat_fallback(
        _idle_projection(selection_idle_reason="disposition_idle:forged")
    )


def test_end_to_end_projection_is_quiescent_for_disposition_idle() -> None:
    projected = project_selection_disposition(
        _idle_projection(
            ready_count=1,
            selectable_ready_count=1,
            eligible_ready_count=1,
            selection_idle_reason="",
        ),
        ready_task_dispositions={
            "TASK-1": ImplementationDisposition.DEFER_CAPABILITY,
        },
    )
    assert projected["selection_idle_reason"] == disposition_selection_idle_reason(
        "defer_capability"
    )
    assert _projection_is_quiescent_for_heartbeat_fallback(projected)

    capacity = project_selection_disposition(
        _idle_projection(
            ready_count=1,
            selectable_ready_count=1,
            eligible_ready_count=1,
        ),
        ready_task_dispositions={
            "TASK-1": ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED,
        },
        provider_capacity_backoff=True,
    )
    assert capacity["selection_idle_reason"] == PROVIDER_CAPACITY_BACKOFF_IDLE_REASON
    assert capacity["selection_idle_reason"] != disposition_selection_idle_reason(
        "residual_llm_authorized"
    )
    assert _projection_is_quiescent_for_heartbeat_fallback(capacity)
