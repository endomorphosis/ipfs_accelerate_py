"""Spent paid attempts must produce a finite, truthful native stop."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
import os
import subprocess
import sys
import time
import shlex

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as daemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_supervisor as portal
from ipfs_accelerate_py.agent_supervisor.todo_daemon import portal_task_state_control_plane as plane
from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_runtime as runtime
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop import SupervisorLoop
from ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery import mark_task_statuses_in_todo_text


@pytest.fixture
def scope(tmp_path, monkeypatch):
    monkeypatch.setenv(plane.ENV_DUCKDB, str(tmp_path / "control-state.duckdb"))
    monkeypatch.setenv(plane.ENV_DUCKLAKE, str(tmp_path / "history"))
    monkeypatch.delenv(plane.ENV_QUACK, raising=False)
    for key in ("IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET",
                "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD"):
        monkeypatch.delenv(key, raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    board = repo / "tasks.md"
    board.write_text("# Tasks\n\n" + "\n".join(
        f"## ACCEL-00{i} Task {i}\n\n- Status: todo\n- Priority: P1\n- Track: ops\n"
        f"- Outputs: result{i}.json\n- Acceptance: real result {i}\n"
        for i in (1, 2)))
    subprocess.run(["git", "add", "tasks.md"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "synthetic task source"], cwd=repo, check=True)
    script = repo / "benign_daemon.py"
    script.write_text("import time\ntime.sleep(60)\n")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    config = portal.PortalSupervisorConfig(
        todo_path=board, state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json", events_path=state_dir / "events.jsonl",
        state_dir=state_dir, repo_root=repo, task_prefix="## ACCEL-", max_task_attempts=3,
        check_interval=.05, daemon_script_path=script, implementation_protected_paths=(),
        merge_target_branch="main",
        worktree_root=tmp_path / "worktrees", objective_task_janitor_enabled=False,
        objective_goal_migration_enabled=False, retry_budget_guardrail_enabled=False,
        dependency_guardrail_enabled=False, reconciliation_guardrail_enabled=False)
    supervisor = portal.PortalImplementationSupervisor(config)
    native = daemon.PortalImplementationDaemon(
        todo_path=board, state_path=config.state_path, strategy_path=config.strategy_path,
        events_path=state_dir / "daemon_events.jsonl", repo_root=repo,
        task_header_prefix="## ACCEL-", max_task_attempts=3, merge_target_branch="main")
    tasks = daemon.parse_task_file(board, "## ACCEL-")
    identities = {task.task_id: native._identity_for_task(task).to_dict() for task in tasks}
    state = daemon.PortalTaskState(
        ready_task_ids=list(identities), ready_count=2, task_count=2,
        task_statuses={key: "ready" for key in identities}, task_identities=identities,
        implementation_attempts={key: 3 for key in identities},
        implementation_attempts_by_cid={value["canonical_task_cid"]: 3 for value in identities.values()},
        selection_idle_reason="all_selectable_ready_tasks_reached_max_task_attempts")
    state.save(config.state_path)
    return supervisor, state


@contextmanager
def child_for(supervisor):
    loop = SupervisorLoop(supervisor.build_supervisor_loop_config())
    spec = loop._child_spec("exhausted-frontier-fixture")
    child = runtime.launch_supervised_child(spec)
    birth = runtime.read_process_birth(child.pid)
    try:
        deadline = time.monotonic() + 5
        while runtime.read_process_command_argv(child.pid) != tuple(spec.command):
            assert time.monotonic() < deadline
            time.sleep(.01)
        yield loop, child
    finally:
        if runtime.read_process_birth(child.pid) == birth:
            assert loop._terminate_child(child)
        try:
            os.waitpid(child.pid, 0)
        except ChildProcessError:
            pass


def test_real_maintenance_reports_spent_frontier_without_rearming(scope):
    supervisor, state = scope
    before = asdict(daemon.PortalTaskState.load(supervisor.config.state_path))
    result = supervisor.run_once(include_refill=False)
    assert result["stuck"] is True
    assert result["reason"] == "task_attempt_budget_exhausted"
    blocker = result["selection_blocker"]
    assert [row["attempt_count"] for row in blocker["limited_tasks"]] == [3, 3]
    assert blocker["attempt_budget_consumed"] is False
    assert blocker["provider_invocation_consumed"] is False
    after = asdict(daemon.PortalTaskState.load(supervisor.config.state_path))
    assert after["implementation_attempts"] == before["implementation_attempts"]
    assert after["implementation_attempts_by_cid"] == before["implementation_attempts_by_cid"]
    assert after["completed_task_ids"] == before["completed_task_ids"] == []


def test_ordinary_watchdog_stops_sealed_child_once_without_restart(scope):
    supervisor, state = scope
    with child_for(supervisor) as (loop, child):
        decision = supervisor._supervisor_loop_watchdog_decision(loop, child, {})
        assert decision.action == "stop"
        assert decision.status == "task_attempt_budget_exhausted"
        assert decision.reason == "task_attempt_budget_exhausted"
        assert loop._terminate_child(child) is True
        assert runtime._supervised_child_identity_session_quiesced(
            runtime.load_supervised_child_identity(child.identity_path))
        assert supervisor.restart_count == 0
    persisted = daemon.PortalTaskState.load(supervisor.config.state_path)
    assert persisted.implementation_attempts == state.implementation_attempts
    assert persisted.implementation_attempts_by_cid == state.implementation_attempts_by_cid


def test_completion_status_change_during_reconciliation_wins_over_stale_projection(scope, monkeypatch):
    supervisor, state = scope
    original = supervisor.reconcile_backlogged_worktrees
    seen = []
    def reconcile(**kwargs):
        result = original(**kwargs)
        # Exercise the observer race against a real source edit. This fixture
        # does not claim that a candidate has passed Native validation.
        text, changed = mark_task_statuses_in_todo_text(
            supervisor.config.todo_path.read_text(), ["ACCEL-001"],
            task_prefix="ACCEL-", status="completed")
        assert changed == ["ACCEL-001"]
        supervisor.config.todo_path.write_text(text)
        seen.append("completed")
        return result
    monkeypatch.setattr(supervisor, "reconcile_backlogged_worktrees", reconcile)
    with child_for(supervisor) as (loop, child):
        decision = supervisor._supervisor_loop_watchdog_decision(loop, child, {})
        assert decision.action == "continue"
        assert runtime.read_process_birth(child.pid) == child.identity_process_birth
    assert seen == ["completed"]
    assert daemon.PortalTaskState.load(supervisor.config.state_path).implementation_attempts == state.implementation_attempts


@pytest.mark.parametrize("transition", ["active", "lower_canonical_count", "awaiting_user", "runnable"])
def test_live_or_intentional_wait_is_not_exhaustion(scope, transition):
    supervisor, state = scope
    if transition == "active":
        state.active_task_id = "ACCEL-001"
        state.implementation_in_progress = True
    elif transition == "lower_canonical_count":
        cid = state.task_identities["ACCEL-001"]["canonical_task_cid"]
        state.implementation_attempts_by_cid[cid] = 2
    elif transition == "awaiting_user":
        state.selection_idle_reason = "awaiting_user"
    else:
        state.selectable_ready_count = 1
        state.selectable_ready_task_ids = ["ACCEL-001"]
    assert supervisor._attempt_budget_exhaustion_status(state) == {}


def test_real_validation_child_defers_terminal_stop(scope):
    supervisor, state = scope
    marker = supervisor.config.repo_root / "validator-pid.txt"
    validator_log = supervisor.config.repo_root / "validator.log"
    test_file = supervisor.config.repo_root / "test_wait.py"
    test_file.write_text("def test_wait():\n    import time\n    time.sleep(30)\n")
    supervisor.config.daemon_script_path.write_text(
        "import os, subprocess, sys, time\nfrom pathlib import Path\n"
        f"out = Path({str(validator_log)!r}).open('w')\n"
        f"p = subprocess.Popen([{sys.executable!r}, '-m', 'pytest', '-q', '-c', '/dev/null', {str(test_file)!r}], "
        "stdout=out, stderr=subprocess.STDOUT, "
        "env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'))\n"
        f"Path({str(marker)!r}).write_text(str(p.pid))\n"
        "p.wait()\ntime.sleep(60)\n")
    with child_for(supervisor) as (loop, child):
        deadline = time.monotonic() + 5
        while not marker.exists() or not supervisor._active_validation_subprocess_exists():
            assert time.monotonic() < deadline, {
                "managed": supervisor._read_managed_daemon_pid(),
                "descendants": portal.descendant_processes(child.pid),
                "validator_log": validator_log.read_text() if validator_log.exists() else "missing"}
            time.sleep(.01)
        validator_birth = runtime.read_process_birth(int(marker.read_text()))
        assert validator_birth is not None
        decision = supervisor._supervisor_loop_watchdog_decision(loop, child, {})
        assert decision.action == "continue"
        assert runtime.read_process_birth(validator_birth.pid) == validator_birth
        fields = loop.config.status_extra_fields
        assert fields["attempt_budget_terminal_pending"] is True
        assert "validation_worker_active" in fields["attempt_budget_terminal_guard"]["blockers"]


def test_restart_budget_is_shared_across_real_typed_and_clean_child_exits(scope, monkeypatch):
    supervisor, state = scope
    supervisor.config.max_restarts = 5
    supervisor.config.watchdog_startup_grace_seconds = .01
    # This test concerns restart accounting, with no attempt-limited frontier.
    state.ready_count = 0
    state.ready_task_ids = []
    state.selection_idle_reason = "no_ready_tasks"
    state.save(supervisor.config.state_path)
    counter = supervisor.config.repo_root / "child-starts.txt"
    supervisor.config.daemon_script_path.write_text(
        "from pathlib import Path\nimport time\n"
        f"p = Path({str(counter)!r})\n"
        "n = int(p.read_text()) + 1 if p.exists() else 1\n"
        "p.write_text(str(n))\ntime.sleep(.2)\n"
        "raise SystemExit(78 if n <= 2 else 0)\n")
    monkeypatch.setattr(supervisor, "_supervisor_loop_recovery_delay_seconds", lambda: 0)
    original = supervisor._supervisor_loop_watchdog_decision
    def bounded_fixture_watchdog(loop, child, status):
        # Keep an unfixed implementation finite: its sixth genuine child
        # demonstrates the renewed budget, and the assertion below rejects it.
        if counter.exists() and int(counter.read_text()) > 5:
            return portal.SupervisorLoopDecision.stop("fixture_restart_overrun")
        return original(loop, child, status)
    monkeypatch.setattr(supervisor, "_supervisor_loop_watchdog_decision", bounded_fixture_watchdog)
    supervisor._run_forever_loop()
    assert counter.read_text() == "5"
    assert supervisor.restart_count == 5
    persisted = daemon.PortalTaskState.load(supervisor.config.state_path)
    assert persisted.implementation_attempts == state.implementation_attempts
    assert persisted.implementation_attempts_by_cid == state.implementation_attempts_by_cid
    status = json.loads(supervisor._supervisor_status_path().read_text())
    assert status["status"] == "max_restarts_reached"
    assert status["restart_count"] == 5
    assert status["restart_budget_exhausted"] is True


@pytest.mark.parametrize("producer_mode", ["empty_evidence", "forged_pass", "failure", "board_trivia", "maintenance_pulse", "missing_objective"])
def test_declared_release_producer_runs_once_then_independent_gate_preserves_tasks(scope, producer_mode):
    supervisor, state = scope
    repo = supervisor.config.repo_root
    producer = repo / "produce_evidence.py"
    counter = repo / "producer-starts.txt"
    gate = repo / "gate.json"
    objective = repo / "goals.md"
    objective.write_text("# Objectives\n")
    if producer_mode == "missing_objective":
        objective.unlink()
    body = {"goals": {}} if producer_mode in {"empty_evidence", "board_trivia", "maintenance_pulse"} else {"goals": {"GOAL-001": {"passed": True}}}
    producer.write_text(
        "from pathlib import Path\n"
        f"p = Path({str(counter)!r})\n"
        "p.write_text(str(int(p.read_text()) + 1 if p.exists() else 1))\n"
        f"Path({str(gate)!r}).write_text({json.dumps(body)!r})\n"
        + (f"b = Path({str(supervisor.config.todo_path)!r})\nb.write_text(b.read_text() + '\\n# Producer note\\n')\n"
           if producer_mode == "board_trivia" else "")
        + ("import json, time\n"
           f"s = Path({str(supervisor._supervisor_status_path())!r})\n"
           "a = json.loads(s.read_text())\ntime.sleep(6)\nb = json.loads(s.read_text())\n"
           f"Path({str(repo / 'pulse.json')!r}).write_text(json.dumps([a,b]))\n"
           if producer_mode == "maintenance_pulse" else "")
        +
        f"raise SystemExit({7 if producer_mode == 'failure' else 0})\n")
    supervisor.config.objective_path = objective
    supervisor.config.objective_reconcile_goal_completion = True
    supervisor.config.objective_goal_completion_gate_path = gate
    supervisor.config.objective_goal_completion_artifact_refresh_command = shlex.join([sys.executable, str(producer)])
    supervisor.config.objective_goal_completion_artifact_refresh_timeout_seconds = 10
    with child_for(supervisor) as (loop, child):
        first = supervisor._supervisor_loop_watchdog_decision(loop, child, {})
        assert first.action == ("continue" if producer_mode == "board_trivia" else "stop")
        outcome = next(iter(supervisor._exhausted_frontier_release_outcomes.values()))
        assert outcome["attempted"] is True
        assert outcome["completion_authority"] is False
        assert outcome["status"] == ("failed" if producer_mode == "failure" else
                                      "unavailable" if producer_mode == "missing_objective" else "reconciled")
        if producer_mode not in {"failure", "missing_objective"}:
            assert outcome["reconciliation"]["attempted"] is True
            assert outcome["reconciliation"]["completed_goal_ids"] == []
        if producer_mode == "missing_objective":
            assert outcome["reconciliation"] == {"attempted": False, "reason": "objective_path_missing"}
        supervisor._last_supervisor_maintenance_at = 0
        second = supervisor._supervisor_loop_watchdog_decision(loop, child, {})
        assert second.action == "stop"
        assert loop.config.status_extra_fields["release_requirements"]["replayed"] is True
    assert counter.read_text() == "1"
    persisted = daemon.PortalTaskState.load(supervisor.config.state_path)
    assert persisted.implementation_attempts == state.implementation_attempts
    assert persisted.implementation_attempts_by_cid == state.implementation_attempts_by_cid
    assert persisted.completed_task_ids == []
    if producer_mode == "maintenance_pulse":
        first, later = json.loads((repo / "pulse.json").read_text())
        assert first["last_agentic_maintenance_phase"] == later["last_agentic_maintenance_phase"] == "release_requirements"
        assert first["updated_at"] != later["updated_at"]
        assert first["supervisor_maintenance"]["deadline_at"] == later["supervisor_maintenance"]["deadline_at"]
        assert first["daemon_process_birth"] == later["daemon_process_birth"]
        assert first["daemon_pid_alive"] is True


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), 0, -1])
def test_unbounded_release_timeout_never_starts_producer(scope, timeout):
    supervisor, state = scope
    supervisor.config.objective_goal_completion_artifact_refresh_command = "must-not-run"
    supervisor.config.objective_goal_completion_gate_path = supervisor.config.repo_root / "gate.json"
    supervisor.config.objective_goal_completion_artifact_refresh_timeout_seconds = timeout
    outcome = supervisor._refresh_exhausted_frontier_release_requirements(
        supervisor._attempt_budget_exhaustion_status(state))
    assert outcome == {"attempted": False, "status": "failed", "reason": "invalid_refresh_timeout"}
    assert not getattr(supervisor, "_exhausted_frontier_release_outcomes", {})


def test_semantically_revised_board_task_does_not_inherit_old_spent_cid(scope):
    supervisor, state = scope
    assert supervisor._attempt_budget_exhaustion_status(state)
    board = supervisor.config.todo_path
    board.write_text(board.read_text().replace("real result 1", "a revised required result 1"))
    assert supervisor._attempt_budget_exhaustion_status(state) == {}
    assert daemon.PortalTaskState.load(supervisor.config.state_path).implementation_attempts_by_cid == state.implementation_attempts_by_cid


@pytest.mark.parametrize("change", ["manual_hold", "dependency", "new_ready_task"])
def test_current_board_frontier_change_defers_without_consuming_recovery(scope, change):
    supervisor, state = scope
    board = supervisor.config.todo_path
    text = board.read_text()
    if change == "manual_hold":
        text = text.replace("- Status: todo", "- Status: todo\n- Manual hold: true", 1)
    elif change == "dependency":
        text = text.replace("- Status: todo", "- Status: todo\n- Depends on: ACCEL-002", 1)
    else:
        text += "\n## ACCEL-003 New work\n\n- Status: todo\n- Outputs: third.json\n- Acceptance: actual new work\n"
    board.write_text(text)
    assert supervisor._attempt_budget_exhaustion_status(state) == {}
    assert not getattr(supervisor, "_exhausted_frontier_release_outcomes", {})
    assert daemon.PortalTaskState.load(supervisor.config.state_path).implementation_attempts_by_cid == state.implementation_attempts_by_cid
