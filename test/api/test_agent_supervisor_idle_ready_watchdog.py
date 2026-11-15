"""Polling activity cannot indefinitely conceal a stalled ready backlog."""

from __future__ import annotations

import shlex
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_supervisor as module
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTaskState
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    DATABASE_IDLE_DAEMON_STALL_REASON,
    PortalImplementationSupervisor,
    PortalSupervisorConfig,
)


def _case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, implement: bool = True):
    # Keep these tests independent of board registration and database services.
    outer = object.__new__(PortalImplementationSupervisor)
    outer.config = PortalSupervisorConfig(
        todo_path=tmp_path / "tasks.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        state_dir=tmp_path / "state",
        repo_root=tmp_path,
        implement=implement,
        check_interval=1,
        daemon_interval=5,
        implementation_log_stall_seconds=30,
        watchdog_startup_grace_seconds=0,
    )
    outer._idle_ready_observation = None
    outer.board_namespace = "idle-ready-watchdog-test"
    outer._last_supervisor_maintenance_at = 0.0
    outer._active_agent_worker_processes = lambda: []
    outer._active_validation_subprocess_exists = lambda: False
    outer._current_supervisor_checkout_lease = lambda: None
    outer._repo_merge_lock_path = lambda: tmp_path / "board-checkout.lock"
    monkeypatch.setattr(module, "read_process_birth", lambda _pid: "exact-birth")
    monkeypatch.setattr(module, "owner_liveness", lambda _birth: module.OwnerLiveness.ALIVE)
    monkeypatch.setattr(module, "descendant_processes", lambda _pid: [])
    child = SimpleNamespace(
        pid=12345, identity_record_id="child-a", identity_process_birth="exact-birth"
    )
    state = PortalTaskState(
        ready_count=139, selectable_ready_count=139,
        selectable_ready_task_ids=["TASK-001"],
    )
    return outer, child, state


def test_fresh_polling_heartbeat_does_not_renew_dispatch_progress(tmp_path, monkeypatch):
    outer, child, state = _case(tmp_path, monkeypatch)
    first = outer._idle_ready_child_watchdog_status(state, child, now_monotonic=100)
    state.heartbeat_at = "2099-01-01T00:00:00+00:00"
    state.last_progress_at = state.heartbeat_at
    result = outer._idle_ready_child_watchdog_status(state, child, now_monotonic=130)

    assert first["implementation_progress_status"] == "awaiting_dispatch"
    assert result["implementation_progress_status"] == "ready_work_stalled"
    assert result["implementation_progress_healthy"] is False
    assert result["idle_ready_recycle_safe"] is True
    assert result["attempt_budget_consumed"] is False
    assert result["provider_invocation_consumed"] is False


@pytest.mark.parametrize("change", ["completed", "attempt", "child"])
def test_semantic_progress_or_new_child_starts_a_new_window(tmp_path, monkeypatch, change):
    outer, child, state = _case(tmp_path, monkeypatch)
    outer._idle_ready_child_watchdog_status(state, child, now_monotonic=100)
    if change == "completed":
        state.completed_count = 1
        state.completed_task_ids = ["TASK-000"]
    elif change == "attempt":
        state.last_implementation_started_at = "2026-10-20T12:00:00+00:00"
    else:
        child.identity_record_id = "child-b"
    result = outer._idle_ready_child_watchdog_status(state, child, now_monotonic=130)
    assert result["implementation_progress_status"] == "awaiting_dispatch"
    assert result["idle_ready_seconds"] == 0
    assert result["idle_ready_recycle_safe"] is False


@pytest.mark.parametrize("reason", [
    "provider_capacity_backoff", "implementation_retry_deferred:quota",
    "resource_claim_deferred:repository", "disposition_idle:approval_required",
    "all_selectable_ready_tasks_reached_max_task_attempts",
])
def test_retry_and_policy_backpressure_are_reported_without_restarting(
    tmp_path, monkeypatch, reason,
):
    outer, child, state = _case(tmp_path, monkeypatch)
    state.selection_idle_reason = reason
    result = outer._idle_ready_child_watchdog_status(state, child, now_monotonic=1000)
    assert result["implementation_progress_status"] == "backpressure"
    assert result["implementation_progress_healthy"] is False
    assert result["implementation_backpressure_reason"] == reason
    assert result["idle_ready_recycle_safe"] is False


def test_monitor_only_backlog_is_explicit_and_cannot_trigger_implementation(tmp_path, monkeypatch):
    outer, child, state = _case(tmp_path, monkeypatch, implement=False)
    result = outer._idle_ready_child_watchdog_status(state, child, now_monotonic=1000)
    assert result["implementation_mode"] == "monitor_only"
    assert result["implementation_enabled"] is False
    assert result["implementation_progress_status"] == "implementation_disabled"
    assert result["implementation_progress_healthy"] is False
    assert result["idle_ready_recycle_safe"] is False


@pytest.mark.parametrize("busy", ["worker", "validation"])
def test_live_worker_prevents_duplicate_dispatch(tmp_path, monkeypatch, busy):
    outer, child, state = _case(tmp_path, monkeypatch)
    outer._idle_ready_child_watchdog_status(state, child, now_monotonic=100)
    if busy == "worker":
        outer._active_agent_worker_processes = lambda: [{"pid": 12346}]
    else:
        outer._active_validation_subprocess_exists = lambda: True
    result = outer._idle_ready_child_watchdog_status(state, child, now_monotonic=130)
    assert result["implementation_progress_status"] == "worker_active"
    assert result["idle_ready_recycle_safe"] is False
    assert outer._idle_ready_observation is None


@pytest.mark.parametrize("fence", ["descendant", "checkout", "birth", "lease"])
def test_stall_recycling_preserves_live_processes_and_checkout_fences(tmp_path, monkeypatch, fence):
    outer, child, state = _case(tmp_path, monkeypatch)
    outer._idle_ready_child_watchdog_status(state, child, now_monotonic=100)
    if fence == "descendant":
        monkeypatch.setattr(module, "descendant_processes", lambda _pid: [{"pid": 12346}])
    elif fence == "checkout":
        outer._repo_merge_lock_path().write_text("retained transaction")
    elif fence == "birth":
        child.identity_process_birth = "prior-birth"
    else:
        outer._current_supervisor_checkout_lease = lambda: object()
    result = outer._idle_ready_child_watchdog_status(state, child, now_monotonic=130)
    assert result["implementation_progress_status"] == "ready_work_stalled"
    assert result["implementation_progress_healthy"] is False
    assert result["idle_ready_recycle_safe"] is False
    assert result["idle_ready_recycle_blockers"]


def test_idle_deadline_respects_daemon_interval_and_startup_grace(tmp_path, monkeypatch):
    outer, child, state = _case(tmp_path, monkeypatch)
    outer.config = replace(outer.config, daemon_interval=100, watchdog_startup_grace_seconds=200)
    outer._idle_ready_child_watchdog_status(state, child, now_monotonic=100)
    result = outer._idle_ready_child_watchdog_status(state, child, now_monotonic=299)
    assert result["idle_ready_stall_after_seconds"] == 200
    assert result["idle_ready_recycle_safe"] is False


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("source_changed", [False, True])
def test_idle_ready_hook_recycles_through_existing_owned_child_lifecycle(
    tmp_path, monkeypatch, selected, source_changed,
):
    outer, child, state = _case(tmp_path, monkeypatch)
    if selected:
        state.active_task_id = "TASK-001"
    outer._idle_ready_child_watchdog_status(state, child, now_monotonic=100)
    monkeypatch.setattr(module.time, "monotonic", lambda: 130)
    monkeypatch.setattr(module.PortalTaskState, "load", lambda _path: state)
    outer._refresh_native_dispatch_child = lambda _child: None
    outer._refresh_loop_proof_rollout_status = lambda _loop: None
    outer._control_plane_status_projection = lambda: {
        "control_plane_update_pending": source_changed,
    }
    outer.is_stuck = lambda *_args, **_kwargs: (False, "")
    outer._database_authority_enabled = lambda: False
    events = []
    outer._record_event = lambda *args: events.append(args)
    loop = SimpleNamespace(config=SimpleNamespace(status_extra_fields={}))

    decision = outer._supervisor_loop_watchdog_decision(loop, child, {})

    assert decision.action == "recycle"
    assert decision.reason == DATABASE_IDLE_DAEMON_STALL_REASON
    assert loop.config.status_extra_fields["implementation_progress_healthy"] is False
    assert events[0][0] == "ready_work_idle_daemon_stall_detected"


def test_inflight_attempt_retains_implementation_watchdog_authority(tmp_path, monkeypatch):
    outer, child, state = _case(tmp_path, monkeypatch)
    state.active_task_id = "TASK-001"
    state.active_phase = "implementing"
    state.implementation_in_progress = True
    result = outer._idle_ready_child_watchdog_status(state, child, now_monotonic=1000)
    assert result["implementation_progress_status"] == "active"
    assert result["idle_ready_recycle_safe"] is False


@pytest.mark.parametrize("enabled", [False, True])
def test_managed_command_always_forwards_explicit_implementation_policy(tmp_path, monkeypatch, enabled):
    outer, _child, _state = _case(tmp_path, monkeypatch, implement=enabled)
    command = outer._build_daemon_command()
    selected = "--implement" if enabled else "--no-implement"
    other = "--no-implement" if enabled else "--implement"
    assert command.count(selected) == 1
    assert other not in command
    assert outer._managed_daemon_matches_command_line(shlex.join(command)) is True
    assert outer._managed_daemon_matches_command_line(
        shlex.join([token for token in command if token != selected])
    ) is False


@pytest.mark.parametrize("reason,blocked", [
    ("all_selectable_ready_tasks_blocked_on_attempt_accounting_evidence", True),
    ("unknown_selection_contract_reason", False),
])
def test_accounting_evidence_idle_reason_keeps_native_recycle_fence(tmp_path, monkeypatch, reason, blocked):
    outer, child, state = _case(tmp_path, monkeypatch)
    state.selection_idle_reason = reason
    state.implementation_attempts = {"TASK-001": 1}
    state.implementation_attempts_by_cid = {"cid-unit": 1}
    outer._idle_ready_child_watchdog_status(state, child, now_monotonic=100)
    monkeypatch.setattr(module.time, "monotonic", lambda: 1000)
    monkeypatch.setattr(module.PortalTaskState, "load", lambda _path: state)
    outer._refresh_native_dispatch_child = lambda _child: None
    outer._refresh_loop_proof_rollout_status = lambda _loop: None
    outer._control_plane_status_projection = lambda: {"control_plane_update_pending": False}
    outer.is_stuck = lambda *_args, **_kwargs: (False, "")
    outer._database_authority_enabled = lambda: False
    events = []
    outer._record_event = lambda *args: events.append(args)
    loop = SimpleNamespace(config=SimpleNamespace(status_extra_fields={}))
    decision = outer._supervisor_loop_watchdog_decision(loop, child, {})
    if blocked:
        assert decision.action == module.SupervisorLoopDecision.keep_running().action
        assert loop.config.status_extra_fields["implementation_progress_status"] == "backpressure"
        assert loop.config.status_extra_fields["implementation_backpressure_reason"] == reason
        assert loop.config.status_extra_fields["idle_ready_recycle_safe"] is False
        assert events == []
    else:
        assert decision.action == "recycle"
        assert decision.reason == DATABASE_IDLE_DAEMON_STALL_REASON
        assert events[0][0] == "ready_work_idle_daemon_stall_detected"
    assert state.implementation_attempts == {"TASK-001": 1}
    assert state.implementation_attempts_by_cid == {"cid-unit": 1}
