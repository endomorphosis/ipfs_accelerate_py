"""Open-frontier recovery names idle work and does not complete it."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.rescue.board_stall_watchdog import parse_task_board
from ipfs_accelerate_py.agent_supervisor.rescue.open_frontier_recovery import (
    classify_open_work,
    frontier_latch_action,
    grok_usage_balance_exhausted,
    select_route,
)


def _board(*sections: str) -> str:
    return "\n".join(sections) + "\n"


def test_selectable_todo_is_idle_until_a_worker_is_alive() -> None:
    parsed = parse_task_board(_board(
        "## LCM-000 Baseline",
        "- Status: todo",
        "- Parent goal: LCM-G010",
        "- Depends on: none",
        "## LCM-010 Register",
        "- Status: todo",
        "- Depends on: LCM-000",
    ), "## LCM-")
    idle = classify_open_work(parsed, worker_alive=False)
    assert idle["stalled"] is True
    assert idle["reason"] == "idle_with_open_work"
    assert idle["repair"] == "route_model"
    assert idle["task_id"] == "LCM-000"
    assert idle["work_complete"] is False
    active = classify_open_work(parsed, worker_alive=True)
    assert active["stalled"] is False
    assert active["reason"] == "worker_active"


def test_dead_worker_recovers_in_progress_selectable_work() -> None:
    parsed = parse_task_board(_board(
        "## LCM-000 Baseline",
        "- Status: completed",
        "- Depends on: none",
        "## LCM-001 Inventory",
        "- Status: in_progress",
        "- Depends on: LCM-000",
    ), "## LCM-")
    idle = classify_open_work(parsed, worker_alive=False)
    assert idle["stalled"] is True
    assert idle["reason"] == "idle_with_open_work"
    assert idle["repair"] == "route_model"
    assert idle["task_id"] == "LCM-001"
    assert idle["work_complete"] is False
    live = classify_open_work(parsed, worker_alive=True)
    assert live["stalled"] is False
    assert live["reason"] == "worker_active"
    assert live["task_id"] == "LCM-001"


def test_reason_latch_holds_one_frontier_and_releases_the_next() -> None:
    same = frontier_latch_action(
        "idle_with_open_work\n",
        reason="idle_with_open_work",
        task_id="LCM-000",
        recorded_task_id="LCM-000",
    )
    assert same["hold"] is True
    assert same["release"] is False
    assert same["completion_authority"] is False
    unseen = frontier_latch_action(
        "idle_with_open_work\n",
        reason="idle_with_open_work",
        task_id="LCM-000",
        recorded_task_id="",
    )
    assert unseen["hold"] is True
    assert unseen["release"] is False
    assert unseen["record_task_id"] == "LCM-000"
    later = frontier_latch_action(
        "idle_with_open_work\n",
        reason="idle_with_open_work",
        task_id="LCM-001",
        recorded_task_id="LCM-000",
    )
    assert later["release"] is True
    assert later["hold"] is False
    assert later["record_task_id"] == "LCM-000"
    assert later["work_complete"] is False
    clear = frontier_latch_action(
        "",
        reason="idle_with_open_work",
        task_id="LCM-001",
        recorded_task_id="LCM-000",
    )
    assert clear["release"] is False
    assert clear["record_task_id"] == "LCM-001"
    structural = frontier_latch_action(
        "dependency_cycle\n",
        reason="dependency_cycle",
        task_id="LCM-001",
        recorded_task_id="LCM-000",
    )
    assert structural["release"] is False
    assert structural["record_task_id"] == "LCM-000"


def test_later_frontier_release_is_retried_until_the_latch_is_removed() -> None:
    recorded = "LCM-074"
    for _ in range(2):
        # The health check persists record_task_id before unlinking the latch.
        # A failed unlink or interrupted poll must not consume the transition.
        action = frontier_latch_action(
            "idle_with_open_work\n",
            reason="idle_with_open_work",
            task_id="LCM-075",
            recorded_task_id=recorded,
        )
        assert action["release"] is True
        assert action["hold"] is False
        recorded = action["record_task_id"]
    cleared = frontier_latch_action(
        "",
        reason="idle_with_open_work",
        task_id="LCM-075",
        recorded_task_id=recorded,
    )
    assert cleared["record_task_id"] == "LCM-075"
    held = frontier_latch_action(
        "idle_with_open_work\n",
        reason="idle_with_open_work",
        task_id="LCM-075",
        recorded_task_id=cleared["record_task_id"],
    )
    assert held["hold"] is True
    assert held["release"] is False


def test_route_keeps_the_injected_provider_and_refuses_other_names() -> None:
    routed = select_route(True, False, decider=lambda _grok, _codex: {
        "selected_provider": "grok",
        "authorized": True,
        "reason_codes": ["preferred_provider_ready"],
    })
    assert routed["authorized"] is True
    assert routed["selected_provider"] == "grok"
    refused = select_route(False, False, decider=lambda _grok, _codex: {
        "selected_provider": "claude",
        "authorized": True,
        "reason_codes": ["secondary"],
    })
    assert refused["authorized"] is False
    assert refused["selected_provider"] == ""
    assert refused["completion_authority"] is False


def test_known_402_balance_text_is_grok_quota_evidence() -> None:
    text = (
        '{"type":"error","message":"Internal error: {\\n  \\"message\\": '
        '\\"API error (status 402 Payment Required): '
        'Grok Build usage balance exhausted\\",\\n  \\"http_status\\": 402'
        '\\n}","cwd":"/tmp"}'
    )
    assert grok_usage_balance_exhausted(text) is True
    assert grok_usage_balance_exhausted("Error: max turns reached") is False


def test_grok_usage_balance_quota_selects_codex() -> None:
    routed = select_route(True, True, grok_hard_quota=True)
    assert routed["authorized"] is True
    assert routed["selected_provider"] == "codex"
    assert routed["work_complete"] is False
    assert routed["completion_authority"] is False
