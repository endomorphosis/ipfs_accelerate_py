"""Fault coverage for bounded, durable multi-board recovery."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import fleet_watchdog as fleet
from ipfs_accelerate_py.agent_supervisor.rescue import live_board_probe


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
    assert persistent["last_action"] == "repair"
    assert persistent["observation"]["busy"] is True
    assert [call["argv"][0] for call in runner.calls].count("repair") == 1
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


def test_operator_hold_keeps_observation_fresh_without_recovery(tmp_path):
    hold = tmp_path / "pause"
    hold.touch()
    board = _board(tmp_path, hold_files=[str(hold)])
    runner = Runner(_observation(health="stopped", recovery_action="ensure"))
    state = fleet.tick_board(board, tmp_path / "watch", apply=True, runner=runner, now=100)
    assert state["health"] == "operator_hold"
    assert state["observed_health"] == "stopped"
    runner.observation = _observation(progress_token="task-2")
    fresh = fleet.tick_board(board, tmp_path / "watch", apply=True, runner=runner, now=200)
    assert fresh["health"] == "operator_hold"
    assert fresh["observed_health"] == "healthy"
    assert fresh["last_progress_at"] == 200
    assert fresh["planned_action"] == ""
    assert hold.exists()
    assert [call["argv"] for call in runner.calls] == [["probe"], ["probe"]]


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


@pytest.mark.parametrize("health,candidate", [("complete", False), ("healthy", True), ("stopped", True)])
def test_completion_and_candidates_use_publication_gate_with_cooldown(tmp_path, health, candidate):
    board = _board(tmp_path, publication={"approved": True})
    state = fleet.assess(_observation(health=health, completion_candidate=candidate), {}, board, 100)
    assert fleet.select_action(state, board, 100) == "publish"
    state["next_action_at"] = 130
    assert fleet.select_action(state, board, 110) == ""
    assert fleet.select_action(state, board, 131) == "publish"
    board.pop("publication")
    assert fleet.select_action(state, board, 131) == "completion_review"


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
    runner = Runner(_observation(completion_candidate=True), action=repair)
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
    runner = Runner(_observation(completion_candidate=True))
    state = fleet.tick_board(_board(tmp_path, publication={"approved": True}), tmp_path / "watch",
                             apply=True, runner=runner, now=100)
    assert state["publication_failure"]["reason_code"] == "publication_failed"
    assert state["publication_failure"]["reason"] == "publisher raised RuntimeError"
    assert [spec["argv"][0] for spec in runner.calls] == ["probe", "repair"]


def test_successful_publication_clears_prior_failure_without_repair(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_completion

    monkeypatch.setattr(fleet_completion, "publish_completed_board", lambda *args: {"status": "published"})
    root = tmp_path / "watch"
    fleet.write_json(root / "spar/state.json", {"publication_failure": {"reason_code": "publication_held"}})
    runner = Runner(_observation(completion_candidate=True))
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
