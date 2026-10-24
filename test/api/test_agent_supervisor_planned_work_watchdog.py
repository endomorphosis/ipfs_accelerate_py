"""Planned work is maintenance evidence, never an automatic status change."""
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue.planned_work_watchdog import observe_planned_work
from ipfs_accelerate_py.agent_supervisor.rescue import repair_dispatcher as dispatcher


@pytest.fixture
def plan(tmp_path):
    tasks = {"T-001": {"status": "completed", "depends": []},
             "T-002": {"status": "planned", "depends": ["T-001"]}}
    goals = {"tasks": [{"id": k, "goal_id": "G-001", "depends_on": t["depends"],
                        "work_package": {"activation_requires": ["real producer and protected validator"]}}
                       for k, t in tasks.items()]}
    (tmp_path / "goals.json").write_text(json.dumps(goals))
    (tmp_path / "research.md").write_text("Reviewed actual external release date.\n")
    policy = {"enabled": True, "grace_seconds": 60, "goals_path": "goals.json"}
    return tmp_path, tasks, goals, policy


def watch(plan, state, now=1000, **changes):
    root, tasks, _goals, policy = plan
    return observe_planned_work(repo_root=root, tasks=tasks, policy=policy, lanes=changes.pop("lanes", []),
                                state=state, now=now, **changes)


def test_opt_out_preserves_existing_idle_behavior(plan):
    root, tasks, _goals, _policy = plan
    state = {}
    result = observe_planned_work(policy=None, repo_root=root, tasks=tasks,
                                  lanes=[], state=state, now=1000)
    assert result == {"enabled": False, "state": "disabled", "incidents": []}
    assert not state


@pytest.mark.parametrize('invalid', [{'enabled': 'yes'}, {'enabled': 1}, []])
def test_malformed_explicit_policy_reports_an_incident_instead_of_healthy_idle(plan, invalid):
    root, tasks, _, _ = plan
    result = observe_planned_work(policy=invalid, repo_root=root, tasks=tasks,
                                  lanes=[], state={}, now=1000)
    assert result['enabled'] is False
    assert result['state'] == 'configuration_error'
    assert result['configuration_errors']
    assert result['incidents'][0]['kind'] == 'planned_work_policy_invalid'


def test_real_dependency_frontier_is_actionable_only_after_persisted_grace(plan):
    root, tasks, goals, _policy = plan
    tasks["T-003"] = {"status": "planned", "depends": ["T-002"]}
    goals["tasks"].append({"id": "T-003", "goal_id": "G-002", "depends_on": ["T-002"]})
    (root / "goals.json").write_text(json.dumps(goals))
    original = copy.deepcopy(tasks)
    state = {}
    first = watch(plan, state)
    assert first["planned_task_ids"] == ["T-002", "T-003"]
    assert first["dependency_ready_planned_task_ids"] == ["T-002"]
    assert not first["incidents"] and not watch(plan, state, 1059)["incidents"]
    restarted = json.loads(json.dumps(state))
    due = watch(plan, restarted, 1060)
    assert due["incidents"][0]["kind"] == "planned_work_unmaterialized"
    assert due["first_seen"] == 1000 and due["materialization_task_ids"] == ["T-002"]
    assert due["work_packages"][0]["activation_requires"] == ["real producer and protected validator"]
    assert tasks == original and not due["completion_authority"]


def test_byte_changes_rearm_but_mtime_and_poll_clocks_do_not(plan):
    root, _tasks, goals, _policy = plan
    state = {}
    watch(plan, state)
    os.utime(root / "goals.json", (2000, 2000))
    assert watch(plan, state, 1060)["first_seen"] == 1000
    goals["tasks"][1]["work_package"]["experiment"] = "A genuinely changed producer contract"
    (root / "goals.json").write_text(json.dumps(goals))
    changed = watch(plan, state, 1070)
    assert changed["first_seen"] == 1070 and not changed["incidents"]


@pytest.mark.parametrize("gate", ["pending", "running", "live_active", "hold"])
def test_runnable_or_inflight_or_held_work_blocks_materialization(plan, gate):
    _root, tasks, _goals, _policy = plan
    state = {}
    watch(plan, state)
    if gate in {"pending", "running"}:
        tasks["T-004"] = {"status": gate, "depends": []}
    result = watch(plan, state, 1200, held=gate == "hold",
                   **({"lanes": [{"active_task_id": "T-004"}]} if gate == "live_active" else {}))
    assert not result["incidents"] and not state.get("planned_work_observation")


def test_reviewed_future_date_waits_then_becomes_actionable_at_exact_boundary(plan):
    _root, _tasks, _goals, policy = plan
    boundary = datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp()
    policy["deferred_tasks"] = {"T-002": {"kind": "not_before", "not_before_utc": "2026-11-01T00:00:00Z",
        "reason": "Actual dated external release", "evidence_path": "research.md"}}
    state = {}
    future = watch(plan, state, boundary - 1)
    assert future["state"] == "waiting_external_gate" and not future["incidents"]
    assert future["deferred_tasks"]["T-002"]["state"] == "waiting_external_gate"
    assert not watch(plan, state, boundary)["incidents"]
    assert watch(plan, state, boundary + 60)["incidents"]


@pytest.mark.parametrize("bad", ["unknown", "naive", "bad-date", "missing-evidence", "generic-external"])
def test_bad_deferral_never_hides_actionable_work(plan, bad):
    _root, _tasks, _goals, policy = plan
    declaration = {"kind": "not_before", "not_before_utc": "2026-11-01T00:00:00Z",
                   "reason": "Concrete external release", "evidence_path": "research.md"}
    task_id = "T-002"
    if bad == "unknown": task_id = "UNKNOWN"
    if bad == "naive": declaration["not_before_utc"] = "2026-11-01T00:00:00"
    if bad == "bad-date": declaration["not_before_utc"] = "someday"
    if bad == "missing-evidence": declaration["evidence_path"] = "missing.md"
    if bad == "generic-external": declaration["kind"] = "external_gate"
    policy["deferred_tasks"] = {task_id: declaration}
    state = {}
    watch(plan, state)
    due = watch(plan, state, 1060)
    assert due["configuration_errors"] and due["incidents"]
    assert due["materialization_task_ids"] == ["T-002"]


def test_missing_goals_is_actionable_configuration_evidence(plan):
    root, _tasks, _goals, _policy = plan
    (root / "goals.json").unlink()
    state = {}
    watch(plan, state)
    due = watch(plan, state, 1060)
    assert due["configuration_errors"] and due["incidents"]


def test_malformed_goal_dependency_is_actionable_instead_of_crashing_observation(plan):
    root, _tasks, goals, _policy = plan
    goals["tasks"][1]["depends_on"] = None
    (root / "goals.json").write_text(json.dumps(goals))
    state = {}
    watch(plan, state)
    due = watch(plan, state, 1060)
    assert due["configuration_errors"] == ["planned task dependency mismatch: T-002"]
    assert due["incidents"]


def test_dispatcher_runs_one_real_bounded_maintenance_child_without_activating_board(plan, monkeypatch):
    root, _tasks, _goals, policy = plan
    runtime = root / "runtime"
    lane = runtime / "lane"
    lane.mkdir(parents=True)
    manifest = runtime / "manifest.json"
    manifest.write_text(json.dumps({"todo_path": "todo.md", "task_header_prefix": "## T-",
        "report_path": "runtime/report.json", "planned_work_policy": policy,
        "lanes": [{"bundle_key": "test", "state_dir": "runtime/lane", "state_prefix": "test"}]}))
    board = root / "todo.md"
    board.write_text("## T-001 Done\n- Status: completed\n- Depends on:\n"
                     "## T-002 Need a real implementation\n- Status: planned\n- Depends on: T-001\n")
    original = board.read_bytes()
    prompt = root / "maintenance.prompt"
    prompt.write_text("Prepare a real producer and protected validation; never blindly activate a task.")
    (runtime / "watchdog.pid").write_text("100")
    (runtime / "report.json").write_text(json.dumps({"status": {"pid": 100}, "reports": []}))
    config = dispatcher.Config(manifest, root, runtime / "dispatcher", prompt_path=prompt,
        worker_argv=[sys.executable, "-c", "import pathlib,sys;pathlib.Path('maintenance-observed.txt').write_text(sys.stdin.read())"],
        worker_timeout=3, cooldown_seconds=10, max_cooldown_seconds=20, max_attempts=1)
    monkeypatch.setattr(dispatcher, "discover_watchdogs", lambda *_: ([], []))
    def probe(pid):
        return {"pid": pid, "alive": pid == 100, "age_seconds": 5000,
                "start_ticks": 1000, "cwd": str(root), "argv": ["python", "-m",
                "ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog",
                "--manifest-path", str(manifest), "--repo-root", str(root)]}
    now = (runtime / "report.json").stat().st_mtime
    state = {}
    assert dispatcher.tick(config, state, apply=True, now=now, probe=probe)["action"] == "healthy"
    result = dispatcher.tick(config, state, apply=True, now=now + 60, probe=probe)
    assert result["action"] == "worker_finished"
    assert result["worker"]["exit_code"] == 0
    assert (root / "maintenance-observed.txt").read_text() == prompt.read_text()
    assert board.read_bytes() == original and not result["completion_authority"]
    again = dispatcher.tick(config, state, apply=True, now=now + 90, probe=probe)
    assert again["action"] == "attention_required" and again["attempt_count"] == 1
