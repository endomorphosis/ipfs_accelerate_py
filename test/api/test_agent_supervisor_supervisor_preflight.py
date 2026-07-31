from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.validation.supervisor_preflight import (
    SupervisorPreflightError,
    summarize_supervisor_preflight,
)


def _state(**overrides):
    payload = {
        "task_count": 29,
        "completed_count": 12,
        "ready_count": 1,
        "ready_task_ids": ["LPR-012"],
        "eligible_ready_count": 1,
        "eligible_ready_task_ids": ["LPR-012"],
        "blocked_count": 0,
        "blocked_task_ids": [],
    }
    payload.update(overrides)
    return payload


def test_runnable_preflight_is_accepted():
    summary = summarize_supervisor_preflight(_state(), expected_task_count=29)

    assert summary["progress_state"] == "runnable"
    assert summary["active_claimed_task_ids"] == []
    assert summary["drained"] is False


def test_matching_live_claim_is_progress_when_task_is_not_dispatchable():
    payload = _state(eligible_ready_count=0, eligible_ready_task_ids=[])
    lanes = [
        {
            "active_task_id": "LPR-012",
            "active_phase": "implementing",
            "implementation_in_progress": True,
        }
    ]

    summary = summarize_supervisor_preflight(
        payload,
        expected_task_count=29,
        live_lane_states=lanes,
    )

    assert summary["progress_state"] == "active_claimed"
    assert summary["active_claimed_task_ids"] == ["LPR-012"]


@pytest.mark.parametrize(
    "lanes",
    [
        [],
        [{"active_task_id": "LPR-999", "implementation_in_progress": True}],
        [{"active_task_id": "LPR-012", "implementation_in_progress": False}],
    ],
)
def test_ready_count_alone_cannot_mask_an_unselectable_board(lanes):
    payload = _state(eligible_ready_count=0, eligible_ready_task_ids=[])

    with pytest.raises(SupervisorPreflightError, match="neither eligible work"):
        summarize_supervisor_preflight(
            payload,
            expected_task_count=29,
            live_lane_states=lanes,
        )


def test_blocked_tasks_fail_closed_even_with_a_live_claim():
    payload = _state(blocked_count=1, blocked_task_ids=["LPR-012"])

    with pytest.raises(SupervisorPreflightError, match="blocked tasks"):
        summarize_supervisor_preflight(
            payload,
            expected_task_count=29,
            live_lane_states=[
                {
                    "active_task_id": "LPR-012",
                    "implementation_in_progress": True,
                }
            ],
        )


def test_complete_board_is_drained_without_ready_work():
    payload = _state(
        completed_count=29,
        ready_count=0,
        ready_task_ids=[],
        eligible_ready_count=0,
        eligible_ready_task_ids=[],
    )

    summary = summarize_supervisor_preflight(payload, expected_task_count=29)

    assert summary["progress_state"] == "drained"
    assert summary["drained"] is True


def test_validated_operational_appendix_does_not_change_canonical_count():
    payload = _state(
        task_count=31,
        completed_count=14,
        task_statuses={
            **{f"LPR-{number:03d}": "todo" for number in range(29)},
            "LPR-029": "completed",
            "LPR-030": "completed",
        },
    )

    summary = summarize_supervisor_preflight(
        payload,
        expected_task_count=29,
        operational_task_ids=["LPR-029", "LPR-030"],
    )

    assert summary["task_count"] == 31
    assert summary["canonical_task_count"] == 29
    assert summary["operational_task_count"] == 2
    assert summary["operational_task_ids"] == ["LPR-029", "LPR-030"]
    assert summary["progress_state"] == "runnable"


def test_unvalidated_extra_tasks_still_fail_closed():
    payload = _state(task_count=31)

    with pytest.raises(SupervisorPreflightError, match="unexpected task count"):
        summarize_supervisor_preflight(
            payload,
            expected_task_count=29,
        )


def test_operational_task_must_exist_in_status_projection():
    payload = _state(
        task_count=30,
        task_statuses={f"LPR-{number:03d}": "todo" for number in range(29)},
    )

    with pytest.raises(SupervisorPreflightError, match="absent from task_statuses"):
        summarize_supervisor_preflight(
            payload,
            expected_task_count=29,
            operational_task_ids=["LPR-029"],
        )
