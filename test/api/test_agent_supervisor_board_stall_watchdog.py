"""Board-stall watchdog: a subagent may plan or repair structure, never complete."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.rescue.board_stall_watchdog import (
    RuntimeSnapshot,
    accept_receipt,
    append_blocked_successor_tasks,
    apply_repair_packet,
    classify_board_stall,
    decide_spawn,
    parse_task_board,
    repair_missing_task_status,
    stall_signature,
)


_PREFIX = "## APP-"


def _board(*sections: str) -> str:
    return "\n".join(sections) + "\n"


def _alive(**overrides: object) -> RuntimeSnapshot:
    payload = dict(supervisor_alive=True, phase="no_selectable_pending", active_task_id="",
                   heartbeat_age_s=1.0, idle_for_s=0.0)
    payload.update(overrides)
    return RuntimeSnapshot(**payload)  # type: ignore[arg-type]


def test_todo_and_none_dependency_stay_open_without_a_fake_task() -> None:
    parsed = parse_task_board(_board(
        "## LCM-000 Baseline",
        "- Status: todo",
        "- Parent goal: LCM-G010",
        "- Depends on: none",
        "## LCM-010 Register",
        "- Status: todo",
        "- Parent goal: LCM-G020",
        "- Depends on: LCM-000",
    ), "## LCM-")
    assert parsed.unknown_status_ids == ()
    assert parsed.tasks[0].depends_on == ()
    assert parsed.tasks[0].goal_id == "LCM-G010"
    assert parsed.tasks[1].depends_on == ("LCM-000",)
    verdict = classify_board_stall(parsed, _alive())
    assert verdict.stalled is False
    assert verdict.reason == "selectable_grace"
    assert verdict.task_ids == ("LCM-000",)


def test_blocked_frontier_requests_successor_subgoals_without_completion() -> None:
    parsed = parse_task_board(_board(
        "## APP-1 Blocked parent",
        "- Status: blocked",
        "- Depends on: APP-0",
        "- Goal id: GOAL-1",
        "## APP-2 Waiting",
        "- Status: ready",
        "- Depends on: APP-1",
        "- Goal id: GOAL-2",
        "## APP-0 Done",
        "- Status: completed",
        "- Goal id: GOAL-0",
    ), _PREFIX)
    verdict = classify_board_stall(parsed, _alive(work_complete=True, completion_authority=True))
    assert verdict.stalled is True
    assert verdict.reason == "idle_incomplete"
    assert verdict.repair_kind == "successor_tasks"
    assert verdict.task_ids == ("APP-1", "APP-2")
    assert verdict.work_complete is False
    assert verdict.completion_authority is False
    assert verdict.to_dict()["work_complete"] is False


def test_selectable_task_is_not_a_stall_during_grace() -> None:
    parsed = parse_task_board(_board(
        "## APP-2 Ready",
        "- Status: ready",
        "- Depends on: APP-1",
        "## APP-1 Done",
        "- Status: completed",
    ), _PREFIX)
    verdict = classify_board_stall(parsed, _alive(idle_for_s=10))
    assert verdict.stalled is False
    assert verdict.reason == "selectable_grace"
    decision = decide_spawn({}, verdict, stall_signature(parsed, verdict), child_alive=False)
    assert decision["spawn"] is False
    assert decision["action"] == "observe"


def test_selectable_idle_does_not_restart_or_add_a_ready_task() -> None:
    parsed = parse_task_board(_board(
        "## APP-2 Ready",
        "- Status: ready",
        "- Depends on: APP-1",
        "## APP-1 Done",
        "- Status: completed",
    ), _PREFIX)
    verdict = classify_board_stall(parsed, _alive(idle_for_s=600))
    assert verdict.reason == "idle_with_open_work"
    assert verdict.repair_kind == "attention"
    decision = decide_spawn({}, verdict, "sig", child_alive=False)
    assert decision["spawn"] is True
    assert decision["repair_kind"] == "record_only"


def test_later_idle_frontier_recovers_after_an_earlier_repair() -> None:
    first = parse_task_board(_board(
        "## LCM-082 Non-admission check",
        "- Status: todo",
        "- Depends on: none",
        "## LCM-083 Later check",
        "- Status: todo",
        "- Depends on: LCM-082",
    ), "## LCM-")
    later = parse_task_board(_board(
        "## LCM-082 Non-admission check",
        "- Status: completed",
        "- Depends on: none",
        "## LCM-083 Later check",
        "- Status: todo",
        "- Depends on: LCM-082",
    ), "## LCM-")
    first_verdict = classify_board_stall(first, _alive(idle_for_s=600))
    later_verdict = classify_board_stall(later, _alive(idle_for_s=600))
    assert first_verdict.reason == later_verdict.reason == "idle_with_open_work"
    assert first_verdict.task_ids == ("LCM-082",)
    assert later_verdict.task_ids == ("LCM-083",)
    first_signature = stall_signature(first, first_verdict)
    later_signature = stall_signature(later, later_verdict)
    assert first_signature != later_signature
    for state in (
        {"last_success_signature": first_signature},
        {"attempt_signature": first_signature, "attempts": 3},
    ):
        held = decide_spawn(state, first_verdict, first_signature, child_alive=False)
        assert held["spawn"] is False
        recovered = decide_spawn(state, later_verdict, later_signature, child_alive=False)
        assert recovered["spawn"] is True
        assert recovered["attempts"] == 1
        assert recovered["task_ids"] == ["LCM-083"]
        assert recovered["work_complete"] is False
        assert recovered["completion_authority"] is False


def test_missing_status_inserts_blocked_and_keeps_dependencies() -> None:
    original = _board(
        "## APP-9 Needs a status",
        "",
        "- Depends on: APP-1",
        "- Goal id: GOAL-9",
        "## APP-1 Done",
        "- Status: completed",
        "- Depends on: APP-0",
    )
    updated, inserted = repair_missing_task_status(original, _PREFIX)
    assert inserted == ("APP-9",)
    assert "- Status: blocked" in updated
    assert "- Depends on: APP-1" in updated
    assert "- Depends on: APP-0" in updated
    assert updated.count("- Status: completed") == 1
    parsed = parse_task_board(updated, _PREFIX)
    assert parsed.missing_status_ids == ()
    verdict = classify_board_stall(parsed, _alive())
    assert verdict.reason == "idle_incomplete"
    assert verdict.repair_kind == "successor_tasks"


def test_duplicate_cycle_and_unknown_status_do_not_rewrite_the_board() -> None:
    original = _board(
        "## APP-1 First",
        "- Status: blocked",
        "- Depends on: APP-2",
        "## APP-2 Second",
        "- Status: blocked",
        "- Depends on: APP-1",
        "## APP-1 Duplicate",
        "- Status: ready",
        "## APP-3 Odd",
        "- Status: finished",
    )
    parsed = parse_task_board(original, _PREFIX)
    assert parsed.duplicate_ids == ("APP-1",)
    verdict = classify_board_stall(parsed, _alive())
    assert verdict.repair_kind == "record_only"
    assert verdict.reason == "duplicate_task_heading"
    packet = {"repair_kind": "record_only", "reason": verdict.reason, "signature": "sig",
              "requested_effects": []}
    applied = apply_repair_packet(packet, board_text=original)
    assert applied["board_text"] == original
    assert applied["work_complete"] is False

    no_dup = _board(
        "## APP-1 First",
        "- Status: blocked",
        "- Depends on: APP-2",
        "## APP-2 Second",
        "- Status: blocked",
        "- Depends on: APP-1",
    )
    cycled = parse_task_board(no_dup, _PREFIX)
    assert cycled.cycles
    cycle_verdict = classify_board_stall(cycled, _alive())
    assert cycle_verdict.reason == "dependency_cycle"
    assert cycle_verdict.repair_kind == "record_only"
    kept, inserted = repair_missing_task_status(no_dup, _PREFIX)
    assert inserted == ()
    assert kept == no_dup
    unknown = parse_task_board(_board(
        "## APP-3 Odd",
        "- Status: finished",
        "- Depends on: APP-1",
    ), _PREFIX)
    unknown_verdict = classify_board_stall(unknown, _alive())
    assert unknown_verdict.reason == "unknown_task_status"
    assert unknown_verdict.repair_kind == "record_only"
    untouched, ids = repair_missing_task_status(_board(
        "## APP-3 Odd",
        "- Status: finished",
        "- Depends on: APP-1",
    ), _PREFIX)
    assert ids == ()
    assert "- Depends on: APP-1" in untouched
    assert "- Status: finished" in untouched


def test_completion_and_forbidden_effects_are_refused() -> None:
    board = _board("## APP-1 Held", "- Status: blocked", "- Depends on: APP-0")
    refused = apply_repair_packet(
        {"repair_kind": "repair_board", "prefix": _PREFIX, "work_complete": True,
         "requested_effects": []},
        board_text=board,
    )
    assert refused["reason"] == "completion_refused"
    assert refused["board_text"] == board
    dropped = apply_repair_packet(
        {"repair_kind": "successor_subgoals", "requested_effects": ["drop_dependency"],
         "signature": "sig"},
        board_text=board,
        successor=lambda: {"admitted": True, "work_complete": False, "completion_authority": False},
    )
    assert dropped["reason"] == "forbidden_effect"
    assert dropped["applied"] is False
    assert "APP-0" in dropped["board_text"]


def test_successor_receipt_cannot_grant_completion_or_change_the_board() -> None:
    board = _board("## APP-1 Held", "- Status: blocked")
    applied = apply_repair_packet(
        {"repair_kind": "successor_subgoals", "signature": "sig", "requested_effects": []},
        board_text=board,
        successor=lambda: {
            "admitted": True,
            "content_id": "abc",
            "subgoal_count": 1,
            "task_ids": ["APP-1"],
            "work_complete": False,
            "completion_authority": False,
        },
    )
    assert applied["applied"] is True
    assert applied["board_text"] == board
    assert applied["work_complete"] is False
    assert applied["admitted"] is True
    lying = apply_repair_packet(
        {"repair_kind": "successor_subgoals", "signature": "sig", "requested_effects": []},
        board_text=board,
        successor=lambda: {"work_complete": True, "completion_authority": False},
    )
    assert lying["reason"] == "completion_refused"
    assert lying["board_text"] == board


def test_spawn_is_once_per_signature_and_a_live_child_blocks_another() -> None:
    parsed = parse_task_board(_board(
        "## APP-1 Held",
        "- Status: blocked",
    ), _PREFIX)
    verdict = classify_board_stall(parsed, _alive())
    signature = stall_signature(parsed, verdict)
    first = decide_spawn({}, verdict, signature, child_alive=False)
    assert first["action"] == "subagent_started"
    assert first["attempts"] == 1
    running = decide_spawn(
        {"attempt_signature": signature, "attempts": 1},
        verdict, signature, child_alive=True,
    )
    assert running["action"] == "subagent_running"
    assert running["spawn"] is False
    accepted = accept_receipt(
        {"signature": signature, "work_complete": False, "completion_authority": False,
         "reason": "successor_subgoals_recorded"},
        signature,
    )
    assert accepted["accepted"] is True
    again = decide_spawn(
        {"last_success_signature": signature, "attempt_signature": signature, "attempts": 1},
        verdict, signature, child_alive=False,
    )
    assert again["action"] == "subagent_started"
    assert again["repair_kind"] == "successor_tasks"
    bad = accept_receipt({"signature": signature, "work_complete": True}, signature)
    assert bad["accepted"] is False
    assert bad["reason"] == "completion_refused"


def test_dead_supervisor_is_attention_and_does_not_request_a_restart() -> None:
    parsed = parse_task_board(_board("## APP-1 Held", "- Status: blocked"), _PREFIX)
    verdict = classify_board_stall(parsed, RuntimeSnapshot(supervisor_alive=False))
    assert verdict.reason == "supervisor_dead"
    assert verdict.repair_kind == "attention"
    decision = decide_spawn({}, verdict, "sig", child_alive=False)
    assert decision["repair_kind"] == "record_only"
    assert "restart_supervisor" not in decision["repair_kind"]


def _proposal() -> dict[str, object]:
    return {
        "task_id": "APP-ST1",
        "title": "Record the observed owner result",
        "status": "blocked",
        "depends_on": ["APP-1"],
        "goal_id": "GOAL-1",
        "command": "python3 worker.py --task-id APP-ST1",
        "validation": "python3 worker.py --validate --task-id APP-ST1",
        "predicted_files": "state/plan.json",
        "acceptance": "The parent task is not completed by this task.",
    }


def test_idle_incomplete_records_subgoals_before_blocked_tasks() -> None:
    board = _board(
        "## APP-1 Held",
        "- Status: blocked",
        "- Depends on: APP-0",
        "## APP-0 Done",
        "- Status: completed",
    )
    parsed = parse_task_board(board, _PREFIX)
    verdict = classify_board_stall(parsed, _alive())
    signature = stall_signature(parsed, verdict)
    first = decide_spawn({}, verdict, signature, child_alive=False)
    assert first["repair_kind"] == "successor_subgoals"
    assert first["spawn"] is True
    second = decide_spawn(
        {
            "repair_signature": signature,
            "completed_kinds": ["successor_subgoals"],
            "attempt_signature": signature,
            "attempts": 1,
        },
        verdict,
        signature,
        child_alive=False,
    )
    assert second["repair_kind"] == "successor_tasks"
    updated, inserted = append_blocked_successor_tasks(board, _PREFIX, (_proposal(),))
    assert inserted == ("APP-ST1",)
    assert "- Depends on: APP-0" in updated
    assert updated.count("- Status: completed") == 1
    assert "- Status: blocked" in updated.split("## APP-ST1", 1)[1]
    repeated, more = append_blocked_successor_tasks(updated, _PREFIX, (_proposal(),))
    assert more == ()
    assert repeated == updated
    ready = dict(_proposal())
    ready["status"] = "ready"
    refused = apply_repair_packet(
        {"repair_kind": "successor_tasks", "prefix": _PREFIX, "requested_effects": [], "signature": "sig"},
        board_text=board,
        proposals=(ready,),
    )
    assert refused["reason"] == "add_ready_task"
    assert refused["board_text"] == board
    done = decide_spawn(
        {
            "repair_signature": signature,
            "completed_kinds": ["successor_subgoals", "successor_tasks"],
        },
        verdict,
        signature,
        child_alive=False,
    )
    assert done["action"] == "already_repaired"
    assert done["spawn"] is False


def test_active_task_is_not_a_stall_even_when_the_snapshot_claims_completion() -> None:
    parsed = parse_task_board(_board("## APP-4 Running", "- Status: in_progress"), _PREFIX)
    verdict = classify_board_stall(
        parsed,
        _alive(active_task_id="APP-4", work_complete=True, completion_authority=True),
    )
    assert verdict.stalled is False
    assert verdict.reason == "active_task"
    assert verdict.work_complete is False
