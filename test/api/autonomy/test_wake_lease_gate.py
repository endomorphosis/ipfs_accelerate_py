from __future__ import annotations

import time

from ipfs_accelerate_py.agent_supervisor.autonomy.recovery_leases import (
    RecoveryLeaseTable,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.runtime import AutonomyWakeEvent
from ipfs_accelerate_py.agent_supervisor.autonomy.wake_lease_gate import (
    WakeLeaseSession,
    apply_lease_held_backoff,
    arm_heartbeat,
    begin_wake_leases,
    safe_begin_wake_leases,
    session_for_task,
)


def test_second_owner_is_gated_on_shared_task() -> None:
    table = RecoveryLeaseTable()
    first_event = AutonomyWakeEvent(
        kind="task",
        cursor_id="cursor:a",
        subject_id="TASK-9",
        lane_id="lane-0",
    )
    second_event = AutonomyWakeEvent(
        kind="task",
        cursor_id="cursor:b",
        subject_id="TASK-9",
        lane_id="lane-1",
    )
    first = begin_wake_leases([first_event], owner_id="lane-0", table=table)
    assert first.blocked is False
    second = begin_wake_leases([second_event], owner_id="lane-1", table=table)
    assert second.blocked is True
    assert second.reason_codes == ("lease_held",)
    first.release()
    retry = begin_wake_leases([second_event], owner_id="lane-1", table=table)
    assert retry.blocked is False
    retry.release()


def test_window_ticks_do_not_take_a_lease() -> None:
    table = RecoveryLeaseTable()
    event = AutonomyWakeEvent(
        kind="window",
        cursor_id="cursor:window",
        safety_timer=True,
    )
    session = begin_wake_leases([event], owner_id="lane-0", table=table)
    assert session.blocked is False
    assert session.held == ()
    assert session.reason == "no_lease_required"


def test_duplicate_task_events_acquire_once() -> None:
    table = RecoveryLeaseTable()
    first = AutonomyWakeEvent(
        kind="task", cursor_id="cursor:a", subject_id="TASK-1", lane_id="lane-0"
    )
    second = AutonomyWakeEvent(
        kind="task", cursor_id="cursor:b", subject_id="TASK-1", lane_id="lane-0"
    )
    session = begin_wake_leases([first, second], owner_id="lane-0", table=table)
    assert session.blocked is False
    assert session.held == (("task", "TASK-1"), ("lane", "lane-0"))
    session.release()


def test_blocked_session_reports_retry_after() -> None:
    table = RecoveryLeaseTable()
    table.try_acquire("task", "TASK-1", "lane-0", ttl_seconds=3600)
    event = AutonomyWakeEvent(kind="task", cursor_id="c", subject_id="TASK-1")
    blocked = begin_wake_leases([event], owner_id="lane-1", table=table)
    assert blocked.blocked is True
    assert blocked.retry_after_seconds is not None
    assert blocked.retry_after_seconds > 0
    assert blocked.expires_at is not None


def test_window_tick_still_leases_active_task() -> None:
    table = RecoveryLeaseTable()
    first = begin_wake_leases(
        [],
        owner_id="lane-0",
        table=table,
        extra_resources=(("task", "TASK-77"),),
    )
    assert first.blocked is False
    assert ("task", "TASK-77") in first.held
    window = AutonomyWakeEvent(
        kind="window", cursor_id="cursor:window", safety_timer=True
    )
    second = begin_wake_leases(
        [window],
        owner_id="lane-1",
        table=table,
        extra_resources=(("task", "TASK-77"),),
    )
    assert second.blocked is True
    first.release()


def test_renew_extends_ttl_so_another_owner_cannot_steal() -> None:
    table = RecoveryLeaseTable()
    session = begin_wake_leases(
        [],
        owner_id="lane-0",
        table=table,
        extra_resources=(("task", "TASK-ttl"),),
        ttl_seconds=2,
    )
    assert session.renew(ttl_seconds=3600) is True
    denied, _ = table.try_acquire("task", "TASK-ttl", "lane-1", ttl_seconds=10)
    assert denied is False
    session.release()


def test_heartbeat_keeps_lease_alive_past_original_ttl() -> None:
    table = RecoveryLeaseTable()
    session = begin_wake_leases(
        [],
        owner_id="lane-0",
        table=table,
        extra_resources=(("task", "TASK-beat"),),
        ttl_seconds=0.4,
    )
    session.start_heartbeat(interval_seconds=0.05)
    time.sleep(0.55)
    denied, _ = table.try_acquire("task", "TASK-beat", "lane-1", ttl_seconds=10)
    assert denied is False
    session.release()
    ok, _ = table.try_acquire("task", "TASK-beat", "lane-1", ttl_seconds=10)
    assert ok is True


def test_release_prevents_renew_from_reacquiring() -> None:
    table = RecoveryLeaseTable()
    session = begin_wake_leases(
        [],
        owner_id="lane-0",
        table=table,
        extra_resources=(("task", "TASK-gone"),),
        ttl_seconds=60,
    )
    session.release()
    assert session.renew() is False
    assert table.inspect("task", "TASK-gone") is None
    session.start_heartbeat(interval_seconds=0.05)
    assert session._thread is None


def test_invalid_extra_resource_kind_is_skipped() -> None:
    table = RecoveryLeaseTable()
    session = begin_wake_leases(
        [],
        owner_id="lane-0",
        table=table,
        extra_resources=(("secret", "nope"), ("task", "TASK-ok")),
    )
    assert session.blocked is False
    assert session.held == (("task", "TASK-ok"),)
    session.release()


def test_arm_heartbeat_releases_keys_if_start_fails() -> None:
    table = RecoveryLeaseTable()
    session = begin_wake_leases(
        [],
        owner_id="lane-0",
        table=table,
        extra_resources=(("task", "TASK-arm"),),
        ttl_seconds=60,
    )

    def _boom(*, interval_seconds=None):
        raise RuntimeError("thread start failed")

    session.start_heartbeat = _boom  # type: ignore[method-assign]
    try:
        arm_heartbeat(session)
        raise AssertionError("arm_heartbeat should propagate")
    except RuntimeError:
        pass
    assert table.inspect("task", "TASK-arm") is None
    assert session.renew() is False


def test_safe_begin_fail_opens_without_leaking(monkeypatch) -> None:
    table = RecoveryLeaseTable()

    def _boom(*_a, **_k):
        raise RuntimeError("gate exploded")

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.autonomy.wake_lease_gate.begin_wake_leases",
        _boom,
    )
    session = safe_begin_wake_leases(
        [],
        owner_id="lane-0",
        table=table,
        extra_resources=(("task", "TASK-safe"),),
    )
    assert session.blocked is False
    assert session.reason == "lease_gate_fail_open"
    assert table.inspect("task", "TASK-safe") is None


def test_session_for_task_contends_on_cid() -> None:
    table = RecoveryLeaseTable()
    first = session_for_task("cid-1", owner_id="owner-a", table=table, ttl_seconds=60)
    assert first.blocked is False
    second = session_for_task("cid-1", owner_id="owner-b", table=table, ttl_seconds=60)
    assert second.blocked is True
    first.release()
    retry = session_for_task("cid-1", owner_id="owner-b", table=table, ttl_seconds=60)
    assert retry.blocked is False
    retry.release()


def test_detach_keeps_table_keys_for_the_next_session() -> None:
    table = RecoveryLeaseTable()
    first = session_for_task("cid-d", owner_id="owner-a", table=table, ttl_seconds=60)
    assert first.blocked is False
    first.detach()
    assert first.renew() is False
    successor = session_for_task(
        "cid-d", owner_id="owner-a", table=table, ttl_seconds=60
    )
    assert successor.blocked is False
    peer = session_for_task("cid-d", owner_id="owner-b", table=table, ttl_seconds=60)
    assert peer.blocked is True
    successor.release()


def test_apply_lease_held_backoff_sets_runner_wait() -> None:
    session = WakeLeaseSession(
        blocked=True,
        reason="recovery_lease_held",
        reason_codes=("lease_held",),
        retry_after_seconds=12.5,
        expires_at=100.0,
    )
    result = apply_lease_held_backoff(
        {"unchanged": True, "next_wake_after_seconds": 30.0},
        session,
    )
    assert result["blocked"] is True
    assert result["reason"] == "lease_held"
    assert result["next_wake_after_seconds"] == 12.5
    assert result["wake_lease"]["retry_after_seconds"] == 12.5
