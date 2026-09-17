"""Fault coverage for bounded, durable multi-board recovery."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import fleet_watchdog as fleet
from ipfs_accelerate_py.agent_supervisor.rescue import live_board_probe


def test_supervisor_pythonpath_precedes_stale_probe_release():
    stale = "/home/barberb/.local/lib/ipfs-taskboard-watchdog/releases/2c3572c47c2749d81b2e"
    path = fleet.supervisor_pythonpath(stale)
    root = str(Path(fleet.__file__).resolve().parents[3])
    assert path.split(":")[0] == root
    assert stale in path.split(":")


def test_fifo_state_does_not_stall_other_board_observations(tmp_path):
    directory = tmp_path / "fifo"
    directory.mkdir()
    os.mkfifo(directory / "state.json")
    script = """
import importlib.util, json, pathlib, sys, types
package = types.ModuleType('isolated_fleet')
package.__path__ = []
sys.modules[package.__name__] = package
for name in ('live_board_probe', 'fleet_watchdog'):
    spec = importlib.util.spec_from_file_location(
        'isolated_fleet.' + name, pathlib.Path(sys.argv[1]) / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
fleet = sys.modules['isolated_fleet.fleet_watchdog']
boards = []
for name in ('fifo', 'healthy'):
    observation = dict(board_id=name, health='healthy', complete=False,
                       busy=False, progress_token='task-1', reason_codes=[])
    boards.append(dict(id=name, cwd=sys.argv[2], probe=dict(
        argv=[sys.executable, '-c', 'print(' + repr(json.dumps(observation)) + ')'])))
report = fleet.run_cycle(dict(state_dir=sys.argv[2], boards=boards), apply=False)
assert report['boards']['fifo']['health'] == 'watchdog_error', report
assert report['boards']['healthy']['health'] == 'healthy', report
assert (pathlib.Path(sys.argv[2]) / 'fifo/state.json').is_fifo()
"""
    subprocess.run([sys.executable, "-c", script,
                    str(Path(live_board_probe.__file__).parent), str(tmp_path)],
                   check=True, timeout=5, capture_output=True)


@pytest.mark.parametrize("raw", [b"{", b"[]", b" "])
def test_bad_persisted_state_does_not_reset_backoff(tmp_path, raw):
    directory = tmp_path / "spar"
    directory.mkdir()
    state = directory / "state.json"
    state.write_bytes(raw)
    with pytest.raises(ValueError):
        fleet.tick_board(_board(tmp_path), tmp_path, apply=True,
                         runner=lambda *a, **k: pytest.fail("invalid prior state must stop this board"))
    assert state.read_bytes() == raw


def test_persisted_state_missing_and_symlink_are_distinct(tmp_path):
    path = tmp_path / "state.json"
    assert fleet.read_json(path) == {}
    target = tmp_path / "target.json"
    target.write_text('{"attempts":7}')
    path.symlink_to(target)
    with pytest.raises(OSError):
        fleet.read_json(path)
    assert json.loads(target.read_text()) == {"attempts": 7}


def _board(tmp_path, **changes):
    return {
        "id": "spar", "cwd": str(tmp_path),
        "probe": {"argv": ["probe"]}, "ensure": {"argv": ["ensure"]},
        "repair": {"argv": ["repair"]}, "failure_grace_seconds": 10,
        "blocked_grace_seconds": 20, "cooldown_seconds": 30,
        "stall_seconds": 60, **changes,
    }


def _observation(**changes):
    return {"board_id": "spar", "health": "healthy", "complete": False,
            "busy": False, "progress_token": "task-1", "reason_codes": [], **changes}


def _result(observation, **changes):
    return {"returncode": 0, "stdout": json.dumps(observation),
            "stderr": "", "timed_out": False, **changes}


def test_control_events_cannot_postpone_task_stall_repair(tmp_path):
    board = _board(tmp_path)
    authority = {"task_count": 2, "task_statuses": {"T-001": "completed", "T-002": "in_progress"}}
    runner = Runner(_observation(progress_token=live_board_probe._progress({**authority, "event_cursor": 1})))
    state_root = tmp_path / "watch"
    first = fleet.tick_board(board, state_root, apply=True, runner=runner, now=100)
    assert first["health"] == "healthy"
    for now, cursor in ((130, 100), (160, 200), (181, 300)):
        runner.observation = _observation(progress_token=live_board_probe._progress(
            {**authority, "event_cursor": cursor}))
        latest = fleet.tick_board(board, state_root, apply=True, runner=runner, now=now)
    assert latest["last_progress_at"] == 100
    assert latest["health"] == "stalled"
    assert latest["last_action"] == "repair"
    repairs = [call for call in runner.calls if call["argv"][0] == "repair"]
    assert len(repairs) == 1
    incident = json.loads(Path(repairs[0]["argv"][2]).read_text())
    assert incident["action"] == "repair"
    assert "no_task_progress" in incident["observation"]["reason_codes"]


def _native_observation(status="in_progress", *, completed=32, receipts=32,
                        goals=7, authenticated=True, **changes):
    return _observation(progress_token=f"{status}:{completed}:{receipts}:{goals}",
        details={"authenticated_task_observation": authenticated,
                 "task_counts": {"completed": completed, status: 1},
                 "completion_receipt_count": receipts, "unsettled_goal_count": goals},
        **changes)


def test_native_retry_cycles_cannot_postpone_task_stall_repair(tmp_path):
    board = _board(tmp_path)
    runner = Runner(_native_observation())
    root = tmp_path / "watch"
    fleet.tick_board(board, root, apply=True, runner=runner, now=100)
    for now, status in ((130, "retrying"), (160, "in_progress"), (181, "retrying")):
        runner.observation = _native_observation(status)
        latest = fleet.tick_board(board, root, apply=True, runner=runner, now=now)
    assert latest["last_progress_at"] == 100
    assert latest["health"] == "stalled"
    assert latest["last_action"] == "supervisor_heal"
    assert not [call for call in runner.calls if call["argv"][0] == "repair"]


def test_native_progress_regression_and_restoration_do_not_extend_deadline(tmp_path):
    board = _board(tmp_path)
    state = fleet.assess(_native_observation(), {}, board, 100)
    state = fleet.assess(_native_observation(completed=31, receipts=31, goals=8), state, board, 130)
    state = fleet.assess(_native_observation(), state, board, 161)
    assert state["last_progress_at"] == 100
    assert state["health"] == "stalled"


def test_native_authentication_gap_cannot_reestablish_progress(tmp_path):
    board = _board(tmp_path)
    state = fleet.assess(_native_observation(), {}, board, 100)
    state = fleet.assess(_native_observation(completed=33, authenticated=False), state, board, 130)
    state = fleet.assess(_observation(progress_token="unavailable", health="unknown"), state, board, 140)
    state = fleet.assess(_native_observation(), state, board, 161)
    assert state["last_progress_at"] == 100
    assert state["health"] == "stalled"


def test_first_probe_failure_after_upgrade_preserves_native_progress_baseline(tmp_path):
    board = _board(tmp_path)
    state = {"observation": _native_observation(), "last_progress_at": 100,
             "progress_token": "old-native-token", "health": "healthy"}
    state = fleet.assess(_observation(progress_token="unavailable", health="unknown"), state, board, 130)
    state = fleet.assess(_native_observation(completed=31), state, board, 140)
    state = fleet.assess(_native_observation(), state, board, 170)
    assert state["last_progress_at"] == 100
    assert state["health"] == "stalled"


@pytest.mark.parametrize("alias", sorted(live_board_probe.COMPLETED))
def test_native_completion_aliases_count_work_without_counting_relabels(tmp_path, alias):
    board = _board(tmp_path)
    state = fleet.assess(_native_observation(), {}, board, 100)
    observed = _native_observation()
    observed["details"]["task_counts"] = {alias: 32, "in_progress": 1}
    state = fleet.assess(observed, state, board, 170)
    assert state["health"] == "stalled"
    observed["details"]["task_counts"][alias] = 33
    state = fleet.assess(observed, state, board, 180)
    assert state["last_progress_at"] == 180
    assert state["health"] == "healthy"


@pytest.mark.parametrize("advancement", [{"completed": 33}, {"receipts": 33}, {"goals": 6}])
def test_native_accepted_progress_renews_task_stall_deadline(tmp_path, advancement):
    board = _board(tmp_path)
    state = fleet.assess(_native_observation(), {}, board, 100)
    state = fleet.assess(_native_observation(**advancement), state, board, 170)
    assert state["last_progress_at"] == 170
    assert state["health"] == "healthy"


def test_live_native_worker_does_not_erase_idle_progress_deadline(tmp_path):
    board = _board(tmp_path)
    state = fleet.assess(_native_observation(), {}, board, 100)
    state = fleet.assess(_native_observation("retrying", busy=True), state, board, 161)
    assert state["health"] == "healthy"
    state = fleet.assess(_native_observation(), state, board, 200)
    assert state["health"] == "stalled"
    assert state["last_progress_at"] == 100


class Runner:
    def __init__(self, observation, action=None):
        self.observation = observation
        self.action = action
        self.calls = []

    def __call__(self, spec, **kwargs):
        self.calls.append(spec)
        if spec["argv"][0] == "probe":
            return _result(self.observation)
        if self.action:
            return self.action(spec, **kwargs)
        return {"returncode": 0, "stdout": "private output", "stderr": ""}


def _blocked_with_ready_owner(**changes):
    observation = _native_observation(health="blocked", reason_codes=["unsettled_goals"])
    observation["details"].update(
        owner_ready=True,
        owner={"pid": 42, "start_time_ticks": 12345, "boot_id": "current-boot"},
        owner_writer_custody={"configured": True, "verified": True, "held": True},
    )
    observation["details"].update(changes)
    return observation


def test_blocked_work_does_not_spend_a_ready_owners_later_restart_budget(tmp_path):
    board = _board(tmp_path)
    prior = {"attempts": 56, "last_action": "repair", "health": "blocked",
             "last_progress_at": 10, "next_action_at": 500, "incident_since": 10}
    observed = fleet.assess(_blocked_with_ready_owner(), prior, board, 100)
    assert observed["attempts"] == 56
    assert observed["next_action_at"] == 500
    assert observed["last_progress_at"] == 10
    stopped = fleet.assess(_observation(health="stopped", recovery_action="ensure"), observed, board, 501)
    assert fleet.select_action(stopped, board, 501) == "ensure"


def test_owner_missing_does_not_spend_llm_router_after_ensure_budget(tmp_path):
    board = _board(tmp_path, max_ensure_attempts=2)
    state = {
        "health": "stopped",
        "stall_class": "owner_missing",
        "observation": _observation(
            health="stopped",
            recovery_action="ensure",
            reason_codes=["owner_process_missing_or_birth_mismatch"],
        ),
        "incident_since": 0,
        "ensure_attempts": 2,
        "attempts": 80,
        "last_action": "repair",
        "last_action_result": {
            "supervisor_heal": {
                "recipe": "clear_overlay_copies_for_owner_start",
                "status": "skip",
                "reason": "no_overlay_copies_to_clear",
            },
        },
        "next_action_at": 10_000,
    }
    assert fleet.select_action(state, board, 20_000) == "ensure"
    state["last_action"] = "ensure"
    state["last_action_result"] = {"returncode": 1}
    assert fleet.select_action(state, board, 20_000) == "supervisor_heal"


def test_sawm_in_progress_sealed_package_selects_supervisor_heal(tmp_path):
    state = {
        "health": "degraded",
        "stall_class": "in_progress_awaiting_effect",
        "observation": {
            "board_id": "sawm",
            "health": "degraded",
            "complete": False,
            "reason_codes": ["extra_gate_recursion_sealed_package"],
            "details": {"task_counts": {"in_progress": 2, "todo": 23, "completed": 20}},
        },
        "incident_since": 0,
        "next_action_at": 0,
    }
    assert fleet.select_action(state, _board(tmp_path, id="sawm"), 100) == "supervisor_heal"


def test_repair_enqueue_does_not_consume_ensure_attempts(tmp_path):
    board = _board(tmp_path)
    root = tmp_path / "watch"
    directory = root / "spar"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "attempts": 56, "ensure_attempts": 0, "health": "blocked",
        "last_progress_at": 10, "incident_since": 10,
    }))
    runner = Runner(_observation(health="blocked", reason_codes=["blocked_task"]))
    state = fleet.tick_board(board, root, apply=True, runner=runner, now=100)
    assert state["last_action"] == "repair" and state["ensure_attempts"] == 0
    runner.observation = _observation(health="stopped", recovery_action="ensure")
    state = fleet.tick_board(board, root, apply=True, runner=runner, now=state["next_action_at"] + 1)
    assert state["last_action"] == "ensure" and state["ensure_attempts"] == 1


@pytest.mark.parametrize("change", [
    {"authenticated_task_observation": False}, {"owner_ready": False},
    {"owner": {}}, {"owner": {"pid": True, "start_time_ticks": 5, "boot_id": "b"}},
    {"owner_writer_custody": {"configured": True, "verified": False, "held": True}},
    {"owner_writer_custody": {"configured": True, "verified": True, "held": False}},
])
def test_unverified_readiness_cannot_refund_restart_budget(tmp_path, change):
    board = _board(tmp_path)
    state = fleet.assess(_blocked_with_ready_owner(**change),
                         {"attempts": 10, "ensure_attempts": 2, "health": "blocked"}, board, 100)
    state = fleet.assess(_observation(health="stopped", recovery_action="ensure"), state, board, 200)
    assert fleet.select_action(state, board, 200) == "repair"


def test_same_ready_owner_does_not_refund_an_inflight_ensure(tmp_path):
    board = _board(tmp_path)
    state = fleet.assess(_blocked_with_ready_owner(), {}, board, 100)
    state.update(ensure_attempts=2, attempts=12, pending_action="ensure", next_action_at=500)
    state = fleet.assess(_blocked_with_ready_owner(), state, board, 200)
    assert state["ensure_attempts"] == 2 and state["pending_action"] == "ensure"
    assert state["next_action_at"] == 500


def test_interrupted_ensure_persists_only_its_own_budget_before_side_effect(tmp_path):
    board = _board(tmp_path)
    root = tmp_path / "watch"
    directory = root / "spar"; directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "attempts": 56, "ensure_attempts": 1, "health": "stopped", "incident_since": 1,
    }))
    def interrupted(*args, **kwargs):
        saved = json.loads((directory / "state.json").read_text())
        assert saved["pending_action"] == "ensure" and saved["ensure_attempts"] == 2
        assert saved["attempts"] == 57
        raise KeyboardInterrupt()
    runner = Runner(_observation(health="stopped", recovery_action="ensure"), action=interrupted)
    with pytest.raises(KeyboardInterrupt):fleet.tick_board(board, root, apply=True, runner=runner, now=100)
    saved = json.loads((directory / "state.json").read_text())
    assert fleet.select_action(saved, board, saved["next_action_at"] + 1) == "repair"


def test_transient_kernel_wait_does_not_trigger_repair_but_persistent_wait_does(tmp_path):
    board = _board(tmp_path)
    runner = Runner(_observation(health="degraded", busy=True,
        reason_codes=["lane_0_daemon_process_uninterruptible"]))
    root = tmp_path / "watch"
    first = fleet.tick_board(board, root, apply=True, runner=runner, now=100)
    assert first["planned_action"] == ""
    runner.observation = _observation(busy=True)
    recovered = fleet.tick_board(board, root, apply=True, runner=runner, now=105)
    assert recovered["health"] == "healthy"
    runner.observation = _observation(health="degraded", busy=True,
        reason_codes=["lane_0_daemon_process_uninterruptible"])
    for now in (110, 129):
        assert fleet.tick_board(board, root, apply=True, runner=runner, now=now)["planned_action"] == ""
    persistent = fleet.tick_board(board, root, apply=True, runner=runner, now=131)
    # D-state I/O is not a coding stall. Persistent uninterruptible wait stays
    # native; an LLM cannot unstick __flush_work or rewrite live receipts.
    # Supervisor_heal rearms false-terminal peers and waits; it does not signal.
    assert persistent["stall_class"] == "kernel_uninterruptible_wait"
    assert persistent.get("last_action") != "repair"
    assert persistent["last_action"] == "supervisor_heal"
    assert persistent["observation"]["busy"] is True
    assert [call["argv"][0] for call in runner.calls].count("repair") == 0
    assert not any(call["argv"][0] == "ensure" for call in runner.calls)


def test_stopped_board_uses_ensure_only_after_grace(tmp_path):
    board = _board(tmp_path)
    runner = Runner(_observation(health="stopped", recovery_action="ensure"))
    root = tmp_path / "watch"
    first = fleet.tick_board(board, root, apply=True, runner=runner, now=100)
    assert first["planned_action"] == ""
    second = fleet.tick_board(board, root, apply=True, runner=runner, now=111)
    assert second["last_action"] == "ensure"
    assert [x["argv"][0] for x in runner.calls] == ["probe", "probe", "ensure"]


@pytest.mark.parametrize("bad", [
    _result(_observation(), returncode=2),
    _result(_observation(), timed_out=True),
    _result(_observation(board_id="sawm")),
    _result(_observation(health="invented")),
    _result(_observation(), stdout="not JSON"),
])
def test_failed_probe_cannot_authorize_ensure(tmp_path, bad):
    observation = fleet.normalize_probe("spar", bad)
    assert observation["health"] == "unknown"
    board = _board(tmp_path)
    state = fleet.assess(observation, {}, board, 100)
    assert fleet.select_action(state, board, 111) == "repair"


def test_missing_probe_executable_reaches_durable_repair(tmp_path):
    board = _board(tmp_path)
    calls = []

    def runner(spec, **kwargs):
        calls.append(spec)
        if spec["argv"][0] == "probe":
            raise FileNotFoundError("missing adapter")
        return {"returncode": 0}

    root = tmp_path / "watch"
    assert fleet.tick_board(board, root, apply=True, runner=runner, now=100)["health"] == "unknown"
    state = fleet.tick_board(board, root, apply=True, runner=runner, now=111)
    assert state["last_action"] == "repair"
    assert "--incident" in calls[-1]["argv"]


def test_stall_cooldown_survives_restart_and_reason_changes(tmp_path):
    board = _board(tmp_path, failure_grace_seconds=0)
    root = tmp_path / "watch"
    runner = Runner(_observation(health="stopped", recovery_action="ensure", reason_codes=["dead"]))
    initial = fleet.tick_board(board, root, apply=True, runner=runner, now=100)
    assert initial["next_action_at"] == 130
    runner.observation["reason_codes"] = ["owner_missing"]
    cooling = fleet.tick_board(board, root, apply=True, runner=runner, now=101)
    assert cooling["planned_action"] == ""
    assert cooling["attempts"] == 1
    second = fleet.tick_board(board, root, apply=True, runner=runner, now=131)
    assert second["attempts"] == 2
    assert second["next_action_at"] == 191
    third = fleet.tick_board(board, root, apply=True, runner=runner, now=192)
    assert third["last_action"] == "repair"


def test_changing_failure_reasons_do_not_postpone_grace_forever(tmp_path):
    board = _board(tmp_path)
    first = fleet.assess(_observation(health="blocked", reason_codes=["a"]), {}, board, 100)
    second = fleet.assess(_observation(health="degraded", reason_codes=["b"]), first, board, 121)
    assert second["incident_since"] == 100
    assert fleet.select_action(second, board, 121) == "repair"


def test_crash_after_intent_preserves_pending_action_and_cooldown(tmp_path):
    board = _board(tmp_path, failure_grace_seconds=0)
    root = tmp_path / "watch"

    def crash(*args, **kwargs):
        raise KeyboardInterrupt("watchdog crash")

    runner = Runner(_observation(health="stopped", recovery_action="ensure"), action=crash)
    with pytest.raises(KeyboardInterrupt):
        fleet.tick_board(board, root, apply=True, runner=runner, now=100)
    durable = fleet.read_json(root / "spar/state.json")
    assert durable["pending_action"] == "ensure"
    assert durable["next_action_at"] == 130
    runner.action = None
    resumed = fleet.tick_board(board, root, apply=True, runner=runner, now=101)
    assert resumed["planned_action"] == ""
    assert resumed["pending_action"] == "ensure"
    assert sum(call["argv"][0] == "ensure" for call in runner.calls) == 1


def test_another_board_lock_prevents_even_probe(tmp_path):
    board = _board(tmp_path)
    runner = Runner(_observation())
    root = tmp_path / "watch"
    with fleet.lock(root / "spar/watchdog.lock") as acquired:
        assert acquired
        state = fleet.tick_board(board, root, apply=True, runner=runner)
    assert state["health"] == "owned_by_another_watchdog"
    assert runner.calls == []


def test_aseh_source_hold_releases_stop_marker_without_forging_admission(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_holds
    hold = tmp_path / "HOLD"
    operation = tmp_path / "operation"
    operation.mkdir()
    (operation / "0004-source_operation_consumed.json").write_text("{}\n")
    hold.write_text(json.dumps({
        "schema": fleet_holds.ASEH_SOURCE_HOLD_SCHEMA,
        "source_admission_verified": False,
        "callback_settlement_authority": False,
        "source_head": "aa5e43ec",
        "target_head": "5205bd1d",
        "operation": str(operation),
    }))
    board = _board(tmp_path, hold_files=[str(hold)],
                   launch_only_hold_files=[str(tmp_path / "watchdog.hold")])
    (tmp_path / "watchdog.hold").write_text("cron owns launch\n")
    observation = _observation(
        health="healthy",
        details={"owner_ready": True, "authenticated_task_observation": True,
                 "native_completion_authority": False},
    )
    runner = Runner(observation)
    state = fleet.tick_board(board, tmp_path / "watch", apply=True, runner=runner, now=100)
    assert not hold.exists()
    archive = tmp_path / "HOLD.unverified-source-admission.json"
    record = json.loads(archive.read_text())
    assert record["source_admission_verified"] is False
    assert record["callback_settlement_authority"] is False
    assert record["stop_marker_release"] == "native_owner_live_operation_consumed"
    assert state["last_action"] == "hold_review"
    assert state["health"] != "operator_hold"
    assert (tmp_path / "watchdog.hold").exists()


def test_pcpr_deletion_hold_is_retained(tmp_path):
    hold = tmp_path / "watchdog.hold"
    hold.write_text("Original PCPR authority was deleted. Do not rematerialize.\n")
    board = _board(tmp_path, hold_files=[str(hold)])
    runner = Runner(_observation(health="unknown", reason_codes=["probe_failed"]))
    state = fleet.tick_board(board, tmp_path / "watch", apply=True, runner=runner, now=100)
    assert hold.exists()
    assert state["health"] == "operator_hold"
    assert state["last_action"] == "hold_review"
    assert state["last_action_result"]["status"] == "wait"
    assert state["last_action_result"]["retained"][0]["reason"] == (
        "deleted_authority_requires_original_or_retirement"
    )


def test_operator_hold_keeps_probe_failure_as_hold_stall(tmp_path):
    hold = tmp_path / "watchdog.hold"
    hold.touch()
    board = _board(tmp_path, hold_files=[str(hold)])

    class FailedProbe(Runner):
        def __call__(self, spec, **kwargs):
            self.calls.append(spec)
            if spec["argv"][0] == "probe":
                return {"returncode": 1, "stdout": "", "stderr": "timeout", "timed_out": False}
            return super().__call__(spec, **kwargs)

    runner = FailedProbe(_observation())
    state = fleet.tick_board(board, tmp_path / "watch", apply=True, runner=runner, now=100)
    assert state["health"] == "operator_hold"
    assert state["stall_class"] == "operator_hold"
    assert state["observed_health"] == "unknown"
    assert "probe_failed" in state["observation"]["reason_codes"]
    assert state["planned_action"] == ""
    assert [call["argv"] for call in runner.calls] == [["probe"]]


def test_operator_hold_keeps_observation_fresh_without_recovery(tmp_path):
    hold = tmp_path / "pause"
    hold.touch()
    board = _board(tmp_path, hold_files=[str(hold)])
    runner = Runner(_observation(health="stopped", recovery_action="ensure"))
    state = fleet.tick_board(board, tmp_path / "watch", apply=True, runner=runner, now=100)
    assert state["health"] == "operator_hold"
    assert state["stall_class"] == "operator_hold"
    assert state["observed_health"] == "stopped"
    runner.observation = _observation(progress_token="task-2")
    fresh = fleet.tick_board(board, tmp_path / "watch", apply=True, runner=runner, now=200)
    assert fresh["health"] == "operator_hold"
    assert fresh["observed_health"] == "healthy"
    assert fresh["last_progress_at"] == 200
    assert fresh["planned_action"] == ""
    assert hold.exists()
    assert [call["argv"] for call in runner.calls] == [["probe"], ["probe"]]


def test_launch_custody_queues_repair_without_starting_owner(tmp_path):
    hold = tmp_path / "watchdog.hold"
    hold.write_text("Existing cron owns relaunch")
    board = _board(tmp_path, hold_files=[str(hold)], launch_only_hold_files=[str(hold)],
                   failure_grace_seconds=0)
    runner = Runner(_observation(health="stopped", recovery_action="ensure"))
    state = fleet.tick_board(board, tmp_path / "watch", apply=True, runner=runner, now=100)
    assert state["last_action"] == "repair"
    assert state["launch_only_holds"] == [str(hold)]
    assert [call["argv"][0] for call in runner.calls] == ["probe", "repair"]
    assert hold.read_text() == "Existing cron owns relaunch"


@pytest.mark.parametrize("name", ["HOLD", "OPERATOR_STOP", "watchdog.disabled"])
def test_full_stop_cannot_be_scoped_to_launch_only(tmp_path, name):
    hold = tmp_path / name
    hold.touch()
    board = _board(tmp_path, hold_files=[str(hold)], launch_only_hold_files=[str(hold)],
                   failure_grace_seconds=0)
    runner = Runner(_observation(health="stopped", recovery_action="ensure"))
    state = fleet.tick_board(board, tmp_path / "watch", apply=True, runner=runner, now=100)
    assert state["health"] == "operator_hold"
    assert [call["argv"][0] for call in runner.calls] == ["probe"]


def test_symlink_launch_marker_remains_a_full_hold(tmp_path):
    marker = tmp_path / "launch.hold"
    marker.symlink_to(tmp_path / "absent")
    board = _board(tmp_path, hold_files=[str(marker)], launch_only_hold_files=[str(marker)])
    assert fleet.repair_hold_paths(board) == [str(marker)]


@pytest.mark.parametrize("token", ["unchanged", None])
def test_idle_heartbeat_without_work_progress_eventually_stalls(tmp_path, token):
    board = _board(tmp_path)
    observation = _observation(progress_token=token)
    first = fleet.assess(observation, {}, board, 100)
    later = fleet.assess(dict(observation, observed_at=161), first, board, 161)
    assert later["health"] == "stalled"
    assert "no_task_progress" in later["observation"]["reason_codes"]


def test_real_progress_and_live_busy_protect_healthy_board(tmp_path):
    board = _board(tmp_path)
    first = fleet.assess(_observation(), {}, board, 100)
    busy = fleet.assess(_observation(busy=True), first, board, 200)
    assert busy["health"] == "healthy"
    progress = fleet.assess(_observation(progress_token="task-2"), busy, board, 220)
    assert progress["health"] == "healthy"
    assert progress["last_progress_at"] == 220


def test_native_complete_uses_publication_gate_with_cooldown(tmp_path):
    board = _board(tmp_path, publication={"approved": True})
    state = fleet.assess(_observation(health="complete", complete=True), {}, board, 100)
    assert fleet.select_action(state, board, 100) == "publish"
    state["next_action_at"] = 130
    assert fleet.select_action(state, board, 110) == ""
    assert fleet.select_action(state, board, 131) == "publish"
    board.pop("publication")
    assert fleet.select_action(state, board, 131) == "completion_review"


def test_unsettled_goals_are_a_wait_stall_not_llm_closeout():
    observation = {
        "health": "degraded",
        "complete": False,
        "completion_candidate": False,
        "board_id": "aseh",
        "reason_codes": ["board_has_unsettled_goals"],
        "details": {"native_completion_authority": False, "unsettled_goal_count": 9,
                    "task_counts": {"completed": 40}},
    }
    assert fleet.classify_stall(observation) == "closeout_waiting_on_unsettled_goals"
    assert "closeout_waiting_on_unsettled_goals" in fleet.WAIT_STALLS
    state = {
        "health": "degraded", "observation": observation,
        "stall_class": "closeout_waiting_on_unsettled_goals",
        "incident_since": 0, "next_action_at": 0, "attempts": 0,
    }
    assert fleet.select_action(
        state, {"publication": {"approved": True}, "repair": {"argv": ["llm"]},
                "failure_grace_seconds": 0, "blocked_grace_seconds": 0}, 100
    ) == ""


def test_disabled_extra_gate_closeout_selects_provisional_heal():
    observation = {
        "health": "degraded",
        "complete": False,
        "reason_codes": ["board_has_unsettled_goals", "goal_closeout_disabled_on_launch"],
        "details": {"unsettled_goal_count": 9, "task_counts": {"completed": 40}},
    }
    state = {
        "health": "degraded", "observation": observation,
        "stall_class": "closeout_waiting_on_unsettled_goals",
        "incident_since": 0, "next_action_at": 0, "attempts": 0,
    }
    assert fleet.select_action(
        state, {"failure_grace_seconds": 0, "blocked_grace_seconds": 0, "repair": {"argv": ["llm"]}}, 100
    ) == "supervisor_heal"
    state["last_action_result"] = {"recipe": "provisionally_complete_terminal_goals"}
    assert fleet.select_action(
        state, {"failure_grace_seconds": 0, "blocked_grace_seconds": 0, "repair": {"argv": ["llm"]}}, 100
    ) == ""


def test_owner_cas_failed_heal_retries_on_cooldown_not_max_backoff(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_heals

    board = _board(tmp_path, failure_grace_seconds=0, blocked_grace_seconds=0,
                   cooldown_seconds=180, max_backoff_seconds=3600)
    observation = {
        "health": "degraded", "complete": False, "board_id": board["id"],
        "busy": False, "progress_token": "g",
        "reason_codes": ["board_has_unsettled_goals", "goal_closeout_disabled_on_launch"],
        "details": {"unsettled_goal_count": 9, "task_counts": {"completed": 40}},
    }
    monkeypatch.setattr(
        fleet_heals, "apply_supervisor_heal",
        lambda *a, **k: {
            "status": "wait", "recipe": "native_goals_still_active",
            "completion_authority": False,
            "reason": "owner_cas_failed:ImportError", "changed_goal_ids": [],
        },
    )
    runner = Runner(observation)
    root = tmp_path / "watch"
    prior = {
        "health": "degraded", "observation": observation,
        "stall_class": "closeout_waiting_on_unsettled_goals",
        "incident_since": 0, "next_action_at": 0, "attempts": 12,
    }
    fleet.write_json(root / board["id"] / "state.json", prior)
    state = fleet.tick_board(board, root, apply=True, runner=runner, now=100)
    assert state["last_action"] == "supervisor_heal"
    assert state["last_action_result"]["reason"] == "owner_cas_failed:ImportError"
    assert state["next_action_at"] == 280
    cooling = fleet.tick_board(board, root, apply=True, runner=runner, now=101)
    assert cooling["planned_action"] == ""


def test_ready_owner_after_ensure_does_not_block_receipt_heal():
    observation = {
        "health": "blocked", "complete": False, "board_id": "doep",
        "reason_codes": ["board_has_blocked_or_quarantined_tasks", "no_ready_independent_tasks"],
        "details": {
            "owner_ready": True,
            "task_counts": {"todo": 23, "blocked": 2},
            "blocked_task_ids": ["DOEP-044", "DOEP-063"],
        },
    }
    state = {
        "health": "blocked", "observation": observation,
        "stall_class": "blocked_without_independent_work",
        "incident_since": 0, "next_action_at": 10_000, "attempts": 12,
        "last_action": "ensure",
    }
    board = {
        "failure_grace_seconds": 0, "blocked_grace_seconds": 0,
        "repair": {"argv": ["llm"]}, "ensure": {"argv": ["ensure"]},
        "max_ensure_attempts": 2,
    }
    assert fleet.select_action(state, board, 100) == "supervisor_heal"
    state["last_action"] = "supervisor_heal"
    assert fleet.select_action(state, board, 100) == ""


def test_local_validation_heal_retries_unstall_on_cooldown(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_heals

    board = _board(tmp_path, failure_grace_seconds=0, blocked_grace_seconds=0,
                   cooldown_seconds=180, max_backoff_seconds=3600)
    observation = {
        "health": "blocked", "complete": False, "board_id": board["id"],
        "busy": False, "progress_token": "d",
        "reason_codes": ["board_has_blocked_or_quarantined_tasks", "no_ready_independent_tasks"],
        "details": {
            "task_counts": {"todo": 23, "blocked": 2},
            "ready_count": 0, "blocked_task_ids": ["DOEP-044", "DOEP-063"],
        },
    }
    monkeypatch.setattr(
        fleet_heals, "apply_supervisor_heal",
        lambda *a, **k: {
            "status": "wait", "recipe": "local_validation_pending_native_admission",
            "completion_authoritative": False,
            "reason": "local checks already recorded; native fenced admission still required",
        },
    )
    runner = Runner(observation)
    root = tmp_path / "watch"
    prior = {
        "health": "blocked", "observation": observation,
        "stall_class": "blocked_without_independent_work",
        "incident_since": 0, "next_action_at": 0, "attempts": 120,
    }
    fleet.write_json(root / board["id"] / "state.json", prior)
    state = fleet.tick_board(board, root, apply=True, runner=runner, now=100)
    assert state["last_action"] == "supervisor_heal"
    assert state["last_action_result"]["recipe"] == "local_validation_pending_native_admission"
    assert state["next_action_at"] == 280
    cooling = fleet.tick_board(board, root, apply=True, runner=runner, now=101)
    assert cooling["planned_action"] == ""


def test_live_worker_wait_retries_unstall_on_cooldown(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_heals

    board = _board(tmp_path, failure_grace_seconds=0, blocked_grace_seconds=0,
                   cooldown_seconds=180, max_backoff_seconds=3600)
    observation = {
        "health": "blocked", "complete": False, "board_id": board["id"],
        "busy": False, "progress_token": "p",
        "reason_codes": ["board_has_blocked_or_quarantined_tasks"],
        "details": {
            "task_counts": {"todo": 28, "blocked": 2, "in_progress": 1},
            "blocked_task_ids": ["PCTDD-035", "PCTDD-038"],
            "lanes": [{"daemon": {"pid": 1}}],
        },
    }
    monkeypatch.setattr(
        fleet_heals, "apply_supervisor_heal",
        lambda *a, **k: {
            "status": "wait", "recipe": "independent_work_has_live_workers",
            "reason": "blocked peers stay blocked; live lanes own independent todos",
        },
    )
    runner = Runner(observation)
    root = tmp_path / "watch"
    prior = {
        "health": "blocked", "observation": observation,
        "stall_class": "independent_work_beside_blocked_peer",
        "incident_since": 0, "next_action_at": 0, "attempts": 102,
    }
    fleet.write_json(root / board["id"] / "state.json", prior)
    state = fleet.tick_board(board, root, apply=True, runner=runner, now=100)
    assert state["last_action"] == "supervisor_heal"
    assert state["last_action_result"]["recipe"] == "independent_work_has_live_workers"
    assert state["next_action_at"] == 280
    cooling = fleet.tick_board(board, root, apply=True, runner=runner, now=101)
    assert cooling["planned_action"] == ""


def test_configured_board_owner_duration_lets_cron_relaunch():
    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        CONFIGURED_BOARD_OWNER_DURATION_SECONDS,
        bound_configured_board_owner_duration,
    )

    assert CONFIGURED_BOARD_OWNER_DURATION_SECONDS == 28800.0
    assert bound_configured_board_owner_duration(
        float("inf"), closeout_disabled=True,
    ) == CONFIGURED_BOARD_OWNER_DURATION_SECONDS
    assert math.isinf(bound_configured_board_owner_duration(
        float("inf"), closeout_disabled=False,
    ))
    assert bound_configured_board_owner_duration(60.0, closeout_disabled=True) == 60.0
    source = (
        Path(__file__).resolve().parents[2]
        / "scripts/run_agent_supervisor_efficiency_state_hardening.py"
    ).read_text(encoding="utf-8")
    assert 'run.add_argument("--duration-seconds", type=float, default=28800.0)' in source
    sawm = (
        Path(__file__).resolve().parents[2]
        / "scripts/ops/agent_supervisor/semantic_addressed_world_model.py"
    ).read_text(encoding="utf-8")
    assert 'launch.add_argument("--duration-seconds", type=float, default=28800.0)' in sawm


def test_aseh_closeout_is_not_spar_clause_stall():
    observation = _observation(health="healthy", complete=False, completion_candidate=True,
                               board_id="aseh")
    observation["details"] = {"native_completion_authority": False}
    assert fleet.classify_stall(observation) == "closeout_requires_native_authority"


def test_missing_clause_evidence_is_not_publication_or_llm_repair():
    observation = _observation(health="healthy", complete=False, completion_candidate=True)
    observation["details"] = {"native_completion_authority": False}
    assert fleet.classify_stall(observation) == "missing_independent_clause_evidence"
    state = {
        "health": "healthy", "observation": observation,
        "stall_class": "missing_independent_clause_evidence",
        "incident_since": 0, "next_action_at": 0, "attempts": 0,
    }
    assert fleet.select_action(
        state, {"publication": {"approved": True}, "repair": {"argv": ["llm"]}, "failure_grace_seconds": 0}, 100
    ) == ""


def test_dirty_control_plane_selects_supervisor_heal(tmp_path):
    observation = {
        "health": "degraded", "complete": False, "reason_codes": ["source_integrity_not_verified"],
        "details": {"owner_ready": True},
    }
    assert fleet.classify_stall(observation) == "configured_control_plane_dirty"
    waiting = {
        "health": "degraded", "observation": observation,
        "stall_class": "configured_control_plane_dirty",
        "incident_since": 0, "next_action_at": 0, "attempts": 0,
    }
    assert fleet.select_action(
        waiting, {"failure_grace_seconds": 0, "blocked_grace_seconds": 0, "repair": {"argv": ["r"]}}, 100
    ) == "supervisor_heal"


def test_independent_work_with_live_lanes_does_not_enqueue_llm(tmp_path):
    board = _board(tmp_path, failure_grace_seconds=0, blocked_grace_seconds=0)
    observation = _observation(
        health="blocked",
        reason_codes=["board_has_blocked_or_quarantined_tasks"],
        details={"owner_ready": True, "task_counts": {"todo": 23, "blocked": 2},
                 "lanes": [{"lane": 0, "daemon": {"pid": 9}}]},
    )
    runner = Runner(observation)
    state = fleet.tick_board(board, tmp_path / "watch", apply=True, runner=runner, now=100)
    assert state["stall_class"] == "independent_todos_unclaimed"
    assert [spec["argv"][0] for spec in runner.calls] == ["probe"]
    assert state.get("last_action") != "repair"


def test_observational_candidate_is_not_publication_authority():
    observation = _observation(health="healthy", complete=False, completion_candidate=True)
    assert fleet.classify_stall(observation) == "closeout_requires_native_authority"
    state = {
        "health": "healthy", "observation": observation,
        "stall_class": "closeout_requires_native_authority",
        "incident_since": 0, "next_action_at": 0, "attempts": 0,
    }
    board = {"publication": {"approved": True}, "failure_grace_seconds": 0}
    assert fleet.select_action(state, board, 100) == ""
    assert fleet.select_action(state, {"failure_grace_seconds": 0, "repair": {"argv": ["r"]}}, 100) == ""


@pytest.mark.parametrize("status", ["held", "failed"])
def test_unsuccessful_publication_queues_repair_with_typed_evidence(tmp_path, monkeypatch, status):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_completion

    board = _board(tmp_path, publication={"approved": True})
    calls = []

    def publisher(manifest, state_dir):
        calls.append(manifest)
        return {"status": status, "reason": "repo: merge conflict", "repositories": [{"id": "repo", "status": "held"}]}

    def repair(spec, **kwargs):
        incident = fleet.read_json(Path(spec["argv"][spec["argv"].index("--incident") + 1]))
        assert incident["recovery_action"] == "repair"
        assert incident["publication_failure"]["reason_code"] == f"publication_{status}"
        assert incident["publication_failure"]["reason"] == "repo: merge conflict"
        assert f"publication_{status}" in incident["observation"]["reason_codes"]
        return {"returncode": 0, "stdout": "private repair output", "stderr": ""}

    monkeypatch.setattr(fleet_completion, "publish_completed_board", publisher)
    runner = Runner(_observation(health="complete", complete=True), action=repair)
    root = tmp_path / "watch"
    state = fleet.tick_board(board, root, apply=True, runner=runner, now=100)
    assert [spec["argv"][0] for spec in runner.calls] == ["probe", "repair"]
    assert state["publication_failure"]["reason_code"] == f"publication_{status}"
    assert state["last_action_result"]["repair_result"] == {"returncode": 0}
    assert state["next_action_at"] == 130
    cooling = fleet.tick_board(board, root, apply=True, runner=runner, now=101)
    assert cooling["planned_action"] == ""
    assert len(calls) == 1
    assert sum(spec["argv"][0] == "repair" for spec in runner.calls) == 1


def test_publisher_exception_is_diagnosed_and_queued_for_repair(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_completion

    def publisher(*args):
        raise RuntimeError("private command output")

    monkeypatch.setattr(fleet_completion, "publish_completed_board", publisher)
    runner = Runner(_observation(health="complete", complete=True))
    state = fleet.tick_board(_board(tmp_path, publication={"approved": True}), tmp_path / "watch",
                             apply=True, runner=runner, now=100)
    assert state["publication_failure"]["reason_code"] == "publication_failed"
    assert state["publication_failure"]["reason"] == "publisher raised RuntimeError"
    assert [spec["argv"][0] for spec in runner.calls] == ["probe", "repair"]


def test_classify_publication_hold_encodes_publisher_logic_stalls():
    assert fleet.classify_publication_hold({
        "reason": "datasets: changed gitlinks lack declared repository dependencies",
    }) == "nested_leaf_gitlinks_travel_with_source"
    assert fleet.classify_publication_hold({
        "reason": "datasets: source_ref must match its clean integration checkout HEAD",
    }) == "nested_source_head_from_parent_gitlink"
    assert fleet.classify_publication_hold({
        "reason": "current_rollout_mode_is_not_required:bootstrap",
    }) == "bootstrap_mode_after_native_authority"
    assert fleet.classify_publication_hold({
        "reason": "publication pull request created; awaiting required GitHub checks and reviews",
    }) == "publication_awaiting_github_review"
    assert fleet.classify_publication_hold({
        "reason": "another publisher holds this board's lock",
    }) == "publication_lock_busy"
    assert fleet.classify_publication_hold({
        "reason": "GitHub Actions account is locked due to a billing issue; retain the PR for retry",
    }) == "publication_github_actions_billing_locked"
    assert fleet.classify_publication_hold({
        "reason": "local required checks failed: command failed with exit code 1",
    }) == "publication_local_required_checks_failed"
    assert fleet.classify_publication_hold({
        "reason": "command timed out after 300s",
    }) == "publication_command_timeout"
    assert fleet.classify_publication_hold({
        "reason": "publication validation failed with exit code 1: ImportError: cannot import name 'open_quack_state_owner_connection'",
    }) == "publication_integration_diverged_from_accepted_source"
    assert fleet.classify_publication_hold({"reason": "repo: merge conflict"}) == "publication_held"


def test_integration_divergence_retries_publish_without_llm():
    observation = _observation(health="complete", complete=True)
    state = {
        "health": "complete", "observation": observation,
        "publication_failure": {
            "stall_class": "publication_integration_diverged_from_accepted_source",
        },
        "incident_since": 0, "next_action_at": 0, "attempts": 0,
    }
    assert fleet.select_action(
        state, {"publication": {"approved": True}, "repair": {"argv": ["llm"]}}, 100
    ) == "publish"


def test_integration_divergence_hold_retries_without_llm(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_completion

    board = _board(tmp_path, publication={"approved": True})

    def publisher(manifest, state_dir):
        return {
            "status": "held",
            "reason": "publication validation failed with exit code 1: AttributeError: module has no attribute 'open_quack_state_owner_connection'",
        }

    monkeypatch.setattr(fleet_completion, "publish_completed_board", publisher)
    runner = Runner(_observation(health="complete", complete=True))
    state = fleet.tick_board(board, tmp_path / "watch", apply=True, runner=runner, now=100)
    assert [spec["argv"][0] for spec in runner.calls] == ["probe"]
    assert state["publication_failure"]["stall_class"] == "publication_integration_diverged_from_accepted_source"
    assert state["last_action_result"]["repair_result"]["status"] == "autoheal_retry"


def test_github_billing_lock_retries_publish_without_llm(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_completion

    board = _board(tmp_path, publication={"approved": True})

    def publisher(manifest, state_dir):
        return {
            "status": "held",
            "reason": "GitHub Actions account is locked due to a billing issue; retain the PR for retry",
        }

    monkeypatch.setattr(fleet_completion, "publish_completed_board", publisher)
    runner = Runner(_observation(health="complete", complete=True))
    state = fleet.tick_board(board, tmp_path / "watch", apply=True, runner=runner, now=100)
    assert [spec["argv"][0] for spec in runner.calls] == ["probe"]
    assert state["publication_failure"]["stall_class"] == "publication_github_actions_billing_locked"
    assert state["last_action_result"]["repair_result"]["status"] == "autoheal_retry"
    assert state["next_action_at"] == 130


def test_nested_leaf_publication_hold_retries_without_llm(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_completion

    board = _board(tmp_path, publication={"approved": True})
    calls = []

    def publisher(manifest, state_dir):
        calls.append(manifest)
        return {
            "status": "held",
            "reason": "datasets: changed gitlinks lack declared repository dependencies",
            "repositories": [{"id": "datasets", "status": "held"}],
        }

    monkeypatch.setattr(fleet_completion, "publish_completed_board", publisher)
    runner = Runner(_observation(health="complete", complete=True))
    state = fleet.tick_board(board, tmp_path / "watch", apply=True, runner=runner, now=100)
    assert [spec["argv"][0] for spec in runner.calls] == ["probe"]
    assert state["publication_failure"]["stall_class"] == "nested_leaf_gitlinks_travel_with_source"
    assert state["last_action_result"]["repair_result"]["status"] == "autoheal_retry"
    assert state["next_action_at"] == 130
    assert len(calls) == 1


def test_successful_publication_clears_prior_failure_without_repair(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_completion

    monkeypatch.setattr(fleet_completion, "publish_completed_board", lambda *args: {"status": "published"})
    root = tmp_path / "watch"
    fleet.write_json(root / "spar/state.json", {"publication_failure": {"reason_code": "publication_held"}})
    runner = Runner(_observation(health="complete", complete=True))
    state = fleet.tick_board(_board(tmp_path, publication={"approved": True}), root,
                             apply=True, runner=runner, now=100)
    assert "publication_failure" not in state
    assert [spec["argv"][0] for spec in runner.calls] == ["probe"]


def test_dry_run_does_not_consume_attempt_budget(tmp_path):
    board = _board(tmp_path, failure_grace_seconds=0)
    runner = Runner(_observation(health="stopped", recovery_action="ensure"))
    state = fleet.tick_board(board, tmp_path / "watch", runner=runner, now=100)
    assert state["planned_action"] == "ensure"
    assert state["attempts"] == 0
    assert len(runner.calls) == 1


def test_hold_created_during_probe_prevents_action(tmp_path):
    hold = tmp_path / "HOLD"
    board = _board(tmp_path, failure_grace_seconds=0, hold_files=[str(hold)])
    delegate = Runner(_observation(health="stopped", recovery_action="ensure"))

    def runner(spec, **kwargs):
        hold.touch()
        return delegate(spec, **kwargs)

    state = fleet.tick_board(board, tmp_path / "watch", apply=True, runner=runner, now=100)
    assert state["health"] == "operator_hold"
    assert state["attempts"] == 0
    assert len(delegate.calls) == 1


def test_owner_missing_selects_ensure_even_without_probe_recovery_action():
    observation = {"health": "stopped", "details": {"owner_ready": False}, "reason_codes": []}
    state = {"health": "stopped", "observation": observation, "stall_class": "owner_missing",
             "incident_since": 0, "next_action_at": 0, "ensure_attempts": 0, "attempts": 0}
    board = {"ensure": {"argv": ["ensure"]}, "failure_grace_seconds": 0}
    assert fleet.select_action(state, board, 100) == "ensure"
    assert fleet.select_action(state, {"failure_grace_seconds": 0}, 100) == "repair"


def test_independent_work_with_supervisor_dirt_still_selects_restore():
    observation = {
        "health": "blocked",
        "reason_codes": ["board_has_blocked_or_quarantined_tasks", "source_integrity_not_verified"],
        "details": {"task_counts": {"todo": 23, "blocked": 2, "in_progress": 2},
                    "lanes": [{"daemon": {"pid": 1}}]},
    }
    stall = fleet.classify_stall(observation)
    state = {"health": "blocked", "observation": observation, "stall_class": stall,
             "incident_since": 0, "next_action_at": 0, "attempts": 0}
    assert fleet.select_action(
        state, {"failure_grace_seconds": 0, "blocked_grace_seconds": 0, "repair": {"argv": ["r"]}}, 100
    ) == "supervisor_heal"


def test_todos_waiting_on_blocked_peers_are_not_independent():
    observation = {
        "health": "blocked",
        "reason_codes": ["board_has_blocked_or_quarantined_tasks", "no_ready_independent_tasks"],
        "details": {
            "task_counts": {"todo": 23, "blocked": 2, "in_progress": 0},
            "ready_count": 0,
            "eligible_ready_count": 0,
            "selection_idle_reason": "no_ready_tasks",
            "blocked_task_ids": ["DOEP-044", "DOEP-063"],
            "lanes": [{"daemon": {"pid": 1}}],
        },
    }
    assert fleet.classify_stall(observation) == "blocked_without_independent_work"
    assert "blocked_without_independent_work" in fleet.WAIT_STALLS
    state = {
        "health": "blocked", "observation": observation,
        "stall_class": "blocked_without_independent_work",
        "incident_since": 0, "next_action_at": 0, "attempts": 0,
    }
    assert fleet.select_action(
        state, {"failure_grace_seconds": 0, "blocked_grace_seconds": 0, "repair": {"argv": ["r"]}}, 100
    ) == "supervisor_heal"
    state["last_action_result"] = {
        "recipe": "local_validation_pending_native_admission",
        "results": [
            {"task_id": "DOEP-044", "status": "passed"},
            {"task_id": "DOEP-063", "status": "receipt_missing"},
        ],
    }
    assert fleet.select_action(
        state, {"failure_grace_seconds": 0, "blocked_grace_seconds": 0, "repair": {"argv": ["r"]}}, 100
    ) == "supervisor_heal"


def test_unclaimed_independent_todos_are_a_wait_stall():
    observation = {
        "health": "blocked",
        "reason_codes": ["board_has_blocked_or_quarantined_tasks"],
        "details": {"task_counts": {"todo": 23, "blocked": 2, "in_progress": 0},
                    "lanes": [{"daemon": {"pid": 1}}]},
    }
    assert fleet.classify_stall(observation) == "independent_todos_unclaimed"
    assert "independent_todos_unclaimed" in fleet.WAIT_STALLS
    state = {"health": "blocked", "observation": observation,
             "stall_class": "independent_todos_unclaimed",
             "incident_since": 0, "next_action_at": 0, "attempts": 0}
    assert fleet.select_action(
        state, {"failure_grace_seconds": 0, "blocked_grace_seconds": 0, "repair": {"argv": ["r"]}}, 100
    ) == "supervisor_heal"


def test_board_checkout_missing_is_a_wait_stall():
    observation = {"health": "unknown", "reason_codes": ["board_checkout_missing"], "complete": False}
    assert fleet.classify_stall(observation) == "board_checkout_missing"
    assert "board_checkout_missing" in fleet.WAIT_STALLS


def test_missing_checkout_is_typed_before_probe_subprocess(tmp_path):
    gone = tmp_path / "deleted-authority"
    board = _board(gone, failure_grace_seconds=0, blocked_grace_seconds=0)
    runner = Runner(_observation(health="unknown", reason_codes=["probe_failed"]))
    state = fleet.tick_board(board, tmp_path / "watch", apply=True, runner=runner, now=100)
    assert runner.calls == []
    assert state["stall_class"] == "board_checkout_missing"
    assert state["observation"]["reason_codes"] == ["board_checkout_missing"]
    assert state["observation"]["details"]["missing"] == "cwd"
    assert state.get("last_action") != "repair"
    assert state["planned_action"] == ""
    assert fleet.select_action(state, board, 100) == ""


def test_missing_checkout_under_deletion_hold_is_not_rematerialized(tmp_path):
    gone = tmp_path / "deleted-pcpr"
    hold = tmp_path / "watchdog.hold"
    hold.write_text(
        "Original PCPR authority and source were deleted. Preserve evidence; "
        "do not rematerialize authoritative state from projections.\n"
    )
    board = _board(gone, hold_files=[str(hold)], failure_grace_seconds=0)
    runner = Runner(_observation(health="unknown", reason_codes=["probe_failed"]))
    state = fleet.tick_board(board, tmp_path / "watch", apply=True, runner=runner, now=100)
    assert runner.calls == []
    assert hold.exists()
    assert state["health"] == "operator_hold"
    assert state["stall_class"] == "board_checkout_missing"
    assert state["last_action"] == "hold_review"
    assert state["last_action_result"]["retained"][0]["reason"] == (
        "deleted_authority_requires_original_or_retirement"
    )
    assert state["planned_action"] == ""


def test_nonzero_native_status_with_live_lanes_is_not_llm_repair():
    observation = {
        "health": "degraded",
        "complete": False,
        "reason_codes": ["native_status_nonzero"],
        "details": {
            "task_counts": {},
            "lanes": [{"lane": 0, "daemon": {"pid": 1}, "status": "running"}],
        },
    }
    assert fleet.classify_stall(observation) == "native_status_unavailable_with_live_workers"
    assert "native_status_unavailable_with_live_workers" in fleet.WAIT_STALLS
    state = {
        "health": "degraded", "observation": observation,
        "stall_class": "native_status_unavailable_with_live_workers",
        "incident_since": 0, "next_action_at": 0, "attempts": 0,
    }
    assert fleet.select_action(
        state, {"failure_grace_seconds": 0, "blocked_grace_seconds": 0, "repair": {"argv": ["r"]}}, 100
    ) == "supervisor_heal"


def test_kernel_uninterruptible_in_progress_is_not_llm_repair():
    observation = {
        "health": "degraded",
        "complete": False,
        "reason_codes": [
            "lane_0_supervisor_process_uninterruptible",
            "lane_2_daemon_process_uninterruptible",
        ],
        "details": {
            "owner_ready": True,
            "task_counts": {"completed": 20, "in_progress": 2, "todo": 23},
            "lanes": [{"daemon": {"pid": 1, "process_state": "D"}}],
        },
    }
    assert fleet.classify_stall(observation) == "in_progress_awaiting_effect"
    waiting = {
        "health": "degraded", "observation": observation,
        "stall_class": "in_progress_awaiting_effect",
        "incident_since": 0, "next_action_at": 0, "attempts": 0,
    }
    assert fleet.select_action(
        waiting, {"failure_grace_seconds": 0, "blocked_grace_seconds": 0, "repair": {"argv": ["r"]}}, 100
    ) == "supervisor_heal"
    dstate = {
        "health": "degraded",
        "reason_codes": ["lane_0_daemon_process_uninterruptible"],
        "details": {"task_counts": {"todo": 23, "in_progress": 0}},
    }
    assert fleet.classify_stall(dstate) == "kernel_uninterruptible_wait"
    assert "kernel_uninterruptible_wait" in fleet.WAIT_STALLS
    dstate_state = {
        "health": "degraded", "observation": dstate,
        "stall_class": "kernel_uninterruptible_wait",
        "incident_since": 0, "next_action_at": 0, "attempts": 0,
    }
    assert fleet.select_action(
        dstate_state, {"failure_grace_seconds": 0, "blocked_grace_seconds": 0, "repair": {"argv": ["r"]}}, 100
    ) == "supervisor_heal"


def test_flush_work_dstate_in_lane_identity_selects_supervisor_heal():
    observation = {
        "health": "degraded",
        "reason_codes": ["no_task_progress"],
        "details": {
            "task_counts": {"completed": 20, "in_progress": 2, "todo": 23},
            "lanes": [
                {"lane": 0, "daemon": {"pid": 1, "process_state": "D", "wait_channel": "__flush_work"}},
                {"lane": 3, "daemon": {"pid": 4, "process_state": "R"}},
            ],
        },
    }
    assert fleet.classify_stall(observation) == "in_progress_awaiting_effect"
    state = {
        "health": "degraded", "observation": observation,
        "stall_class": "in_progress_awaiting_effect",
        "incident_since": 0, "next_action_at": 0, "attempts": 0,
    }
    assert fleet.select_action(
        state, {"failure_grace_seconds": 0, "blocked_grace_seconds": 0, "repair": {"argv": ["llm"]}}, 100
    ) == "supervisor_heal"


def test_native_unhealthy_empty_counts_with_dstate_is_not_llm():
    observation = {
        "health": "degraded",
        "reason_codes": [
            "goal_closeout_disabled_on_launch",
            "native_operator_reports_unhealthy",
            "task_observation_not_completion_authority",
        ],
        "details": {
            "owner_ready": True,
            "task_counts": {},
            "lanes": [{
                "lane": 0,
                "daemon": {"pid": 1, "process_state": "D", "wait_channel": "jbd2_log_wait_commit"},
            }],
        },
    }
    assert fleet.classify_stall(observation) == "kernel_uninterruptible_wait"
    state = {
        "health": "degraded", "observation": observation,
        "stall_class": "kernel_uninterruptible_wait",
        "incident_since": 0, "next_action_at": 0, "attempts": 0,
    }
    assert fleet.select_action(
        state, {"failure_grace_seconds": 0, "blocked_grace_seconds": 0, "repair": {"argv": ["llm"]}}, 100
    ) == "supervisor_heal"


def test_carried_task_counts_keep_independent_work_classification():
    previous = {
        "observation": {
            "details": {
                "owner_ready": True,
                "task_counts": {"completed": 22, "in_progress": 3, "todo": 28, "blocked": 2},
                "blocked_task_ids": ["PCTDD-035", "PCTDD-038"],
            },
        },
    }
    observation = {
        "health": "degraded",
        "reason_codes": ["native_operator_reports_unhealthy", "task_observation_not_completion_authority"],
        "details": {
            "owner_ready": True,
            "task_counts": {},
            "lanes": [{"lane": 0, "daemon": {"pid": 1, "process_state": "S"}}],
        },
    }
    carried = fleet._carry_observational_task_counts(observation, previous)
    assert carried["details"]["task_counts"]["in_progress"] == 3
    assert carried["details"]["blocked_task_ids"] == ["PCTDD-035", "PCTDD-038"]
    assert carried["details"]["task_counts_source"] == "carried_last_native_projection"
    assert fleet.classify_stall(carried) == "independent_work_beside_blocked_peer"


def test_native_unhealthy_with_live_owner_is_not_llm():
    observation = {
        "health": "degraded",
        "reason_codes": [
            "native_operator_reports_unhealthy",
            "task_observation_not_completion_authority",
        ],
        "details": {
            "owner_ready": True,
            "task_counts": {},
            "lanes": [{"lane": 0, "daemon": {"pid": 1, "process_state": "S"}}],
        },
    }
    assert fleet.classify_stall(observation) == "native_status_unavailable_with_live_workers"
    state = {
        "health": "degraded", "observation": observation,
        "stall_class": "native_status_unavailable_with_live_workers",
        "incident_since": 0, "next_action_at": 0, "attempts": 0,
    }
    assert fleet.select_action(
        state, {"failure_grace_seconds": 0, "blocked_grace_seconds": 0, "repair": {"argv": ["llm"]}}, 100
    ) == "supervisor_heal"


def test_blocked_independent_work_outranks_remaining_board_doc_dirt():
    observation = {
        "health": "blocked",
        "reason_codes": ["board_has_blocked_or_quarantined_tasks", "source_integrity_not_verified"],
        "details": {"task_counts": {"todo": 23, "blocked": 2},
                    "lanes": [{"daemon": {"pid": 1}}]},
    }
    assert fleet.classify_stall(observation) == "independent_todos_unclaimed"


def test_classify_stall_distinguishes_independent_work_beside_blocked_peers():
    blocked = {"health": "blocked", "reason_codes": ["board_has_blocked_or_quarantined_tasks"],
               "details": {"owner_ready": True, "task_counts": {"todo": 23, "blocked": 1}}}
    assert fleet.classify_stall(blocked) == "independent_work_beside_blocked_peer"
    blocked["details"]["task_counts"] = {"todo": 0, "in_progress": 0, "blocked": 2}
    assert fleet.classify_stall(blocked) == "blocked_without_independent_work"
    assert fleet.classify_stall({"health": "stopped", "details": {"owner_ready": False}}) == "owner_missing"
    stalled = {"health": "healthy", "busy": False, "complete": False, "reason_codes": ["no_task_progress"]}
    assert fleet.classify_stall(stalled) == "stalled_no_progress"
    stalled["details"] = {"task_counts": {"in_progress": 2, "todo": 23}}
    assert fleet.classify_stall(stalled) == "in_progress_awaiting_effect"
    rearmed = {
        "health": "stalled", "reason_codes": ["no_task_progress"],
        "details": {
            "task_counts": {"completed": 22, "retrying": 2, "todo": 28, "in_progress": 0},
            "lanes": [{"daemon": {"pid": 1}}],
        },
    }
    assert fleet.classify_stall(rearmed) == "in_progress_awaiting_effect"
    rearmed_state = {
        "health": "stalled", "observation": rearmed, "stall_class": "in_progress_awaiting_effect",
        "incident_since": 0, "next_action_at": 0, "attempts": 0,
    }
    assert fleet.select_action(
        rearmed_state,
        {"failure_grace_seconds": 0, "blocked_grace_seconds": 0, "repair": {"argv": ["llm"]}},
        100,
    ) == "supervisor_heal"
    todos = {
        "health": "stalled", "reason_codes": ["no_task_progress"],
        "details": {
            "task_counts": {"todo": 28, "in_progress": 0, "retrying": 0},
            "lanes": [{"daemon": {"pid": 1}}],
        },
    }
    assert fleet.classify_stall(todos) == "independent_todos_unclaimed"
    waiting = {
        "health": "stalled", "observation": stalled, "stall_class": "in_progress_awaiting_effect",
        "incident_since": 0, "next_action_at": 0, "attempts": 0,
    }
    assert fleet.select_action(
        waiting, {"failure_grace_seconds": 0, "blocked_grace_seconds": 0, "repair": {"argv": ["r"]}}, 100
    ) == "supervisor_heal"
    live_status = {
        "health": "stopped",
        "reason_codes": ["owner_status_identity_missing", "owner_not_ready"],
        "details": {"owner_ready": False, "owner": {"pid": 7136}},
    }
    assert fleet.classify_stall(live_status) == "owner_live_status_unreadable"
    waiting_live = {
        "health": "stopped", "observation": live_status,
        "stall_class": "owner_live_status_unreadable",
        "incident_since": 0, "next_action_at": 0, "attempts": 0, "ensure_attempts": 0,
    }
    assert fleet.select_action(
        waiting_live,
        {"failure_grace_seconds": 0, "ensure": {"argv": ["ensure"]}, "repair": {"argv": ["r"]}},
        100,
    ) == ""


def test_run_cycle_writes_ducklake_fleet_health_without_completion_authority(tmp_path):
    observation = dict(board_id="sawm", health="stopped", complete=False, busy=False,
                       progress_token="t1", reason_codes=["owner_process_missing"],
                       details={"owner_ready": False, "task_counts": {}})
    board = dict(id="sawm", cwd=str(tmp_path), probe=dict(
        argv=[sys.executable, "-c", "print(" + repr(json.dumps(observation)) + ")"]))
    report = fleet.run_cycle(dict(state_dir=str(tmp_path / "watch"), boards=[board]), apply=False)
    payload = json.loads((tmp_path / "watch" / "ducklake_fleet_health.json").read_text())
    assert payload["schema"] == fleet.FLEET_HEALTH_SCHEMA
    assert payload["completion_authority"] is False
    assert payload["boards"]["sawm"]["stall_class"] == "owner_missing"
    assert payload["boards"]["sawm"]["owner_ready"] is False
    assert report["boards"]["sawm"]["stall_class"] == "owner_missing"


def test_command_timeout_kills_descendant_after_launcher_exits(tmp_path):
    pidfile = tmp_path / "child.pid"
    child = (
        "import os, signal, time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(30)"
    )
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(30)"
    )
    result = fleet.command({
        "argv": [sys.executable, "-c", parent], "timeout_seconds": 0.4,
        "termination_grace_seconds": 0.1,
    }, cwd=str(tmp_path))
    assert result["timed_out"] is True
    pid = int(pidfile.read_text())
    proc_stat = Path(f"/proc/{pid}/stat")
    deadline = time.monotonic() + 2
    try:
        while proc_stat.exists() and time.monotonic() < deadline:
            if proc_stat.read_text().rsplit(")", 1)[-1].split()[0] == "Z":
                break
            time.sleep(0.01)
        assert not proc_stat.exists() or proc_stat.read_text().rsplit(")", 1)[-1].split()[0] == "Z"
    finally:
        # A failing regression must not leak the deliberately stubborn child.
        if proc_stat.exists():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_detached_launcher_stdout_cannot_hold_command_open(tmp_path):
    # The successful launcher intentionally leaves a detached process behind;
    # temporary output files avoid waiting for it to close inherited pipes.
    code = (
        "import subprocess, sys; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(0.5)'], "
        "start_new_session=True); print('launched')"
    )
    result = fleet.command({"argv": [sys.executable, "-c", code]}, cwd=str(tmp_path), timeout=0.3)
    assert result["returncode"] == 0
    assert result["timed_out"] is False
    assert "launched" in result["stdout"]


def test_live_unready_owner_outranks_extra_gate_recursion():
    observation = {
        "health": "degraded",
        "complete": False,
        "reason_codes": [
            "extra_gate_recursion_sealed_package",
            "owner_not_ready",
            "owner_status_identity_missing",
        ],
        "details": {
            "owner_ready": False,
            "owner": {"pid": 1107722, "process_state": "R"},
            "task_counts": {},
        },
    }
    assert fleet.classify_stall(observation) == "owner_live_status_unreadable"
    state = {
        "health": "degraded", "observation": observation,
        "stall_class": "owner_live_status_unreadable",
        "incident_since": 0, "next_action_at": 0, "attempts": 0,
        "ensure_attempts": 0,
    }
    assert fleet.select_action(
        state, {"failure_grace_seconds": 0, "blocked_grace_seconds": 0,
                "repair": {"argv": ["llm"]}, "ensure": {"argv": ["ensure"]},
                "max_ensure_attempts": 2}, 100
    ) == ""


def test_unsettled_goals_outrank_extra_gate_recursion():
    observation = {
        "health": "degraded",
        "complete": False,
        "reason_codes": [
            "board_has_unsettled_goals",
            "extra_gate_recursion_sealed_package",
            "goal_closeout_disabled_on_launch",
        ],
        "details": {"task_counts": {"completed": 40}, "unsettled_goal_count": 9},
    }
    assert fleet.classify_stall(observation) == "closeout_waiting_on_unsettled_goals"


def test_native_in_progress_outranks_extra_gate_recursion():
    observation = {
        "health": "stalled",
        "complete": False,
        "reason_codes": [
            "extra_gate_recursion_sealed_package",
            "extra_gate_recursion_competing_unit",
            "no_task_progress",
        ],
        "details": {
            "owner_ready": True,
            "task_counts": {"in_progress": 3, "todo": 28},
            "lanes": [{"daemon": {"pid": 1}}],
        },
    }
    assert fleet.classify_stall(observation) == "in_progress_awaiting_effect"
    blocked = {
        "health": "blocked",
        "complete": False,
        "reason_codes": [
            "board_has_blocked_or_quarantined_tasks",
            "extra_gate_recursion_sealed_package",
            "no_ready_independent_tasks",
        ],
        "details": {
            "owner_ready": True,
            "task_counts": {"blocked": 2, "completed": 60, "todo": 23},
            "blocked_task_ids": ["DOEP-044", "DOEP-063"],
            "selection_idle_reason": "no_ready_tasks",
            "ready_count": 0,
        },
    }
    assert fleet.classify_stall(blocked) == "blocked_without_independent_work"


def test_extra_gate_recursion_selects_supervisor_heal_not_ensure_or_llm():
    observation = {
        "health": "stalled",
        "complete": False,
        "reason_codes": [
            "extra_gate_recursion_sealed_package",
            "extra_gate_recursion_competing_unit",
            "no_task_progress",
        ],
        "details": {
            "owner_ready": True,
            "task_counts": {"completed": 20},
            "extra_gate": {
                "live_owner_unit": "pctdd-g9-quack-owner.service",
                "inventory_owner_unit": "ipfs-accelerate-pctdd-g9-watchdog.service",
                "heal_overlay": False,
            },
            "lanes": [{"daemon": {"pid": 1}}],
        },
    }
    assert fleet.classify_stall(observation) == "extra_gate_recursion"
    assert "extra_gate_recursion" not in fleet.WAIT_STALLS
    state = {
        "health": "stalled", "observation": observation,
        "stall_class": "extra_gate_recursion",
        "incident_since": 0, "next_action_at": 0, "attempts": 0,
        "ensure_attempts": 0,
    }
    board = {
        "failure_grace_seconds": 0, "blocked_grace_seconds": 0,
        "repair": {"argv": ["llm"]}, "ensure": {"argv": ["ensure"]},
        "max_ensure_attempts": 2,
    }
    assert fleet.select_action(state, board, 100) == "supervisor_heal"
