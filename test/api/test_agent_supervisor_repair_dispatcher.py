"""Behavioral tests for bounded repair dispatch, separate from lane tests."""

import json
import os
from pathlib import Path
import sys
import time
import signal
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue.repair_dispatcher import (
    AlreadyRunning, Config, main, observe, probe_process, run_worker, singleton_lock, tick,
    repair_lock,
)
from ipfs_accelerate_py.agent_supervisor.rescue import repair_dispatcher


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr(repair_dispatcher, "discover_watchdogs", lambda *_: ([], []))
    now = time.time()
    runtime = tmp_path / "runtime"
    lane = runtime / "lane"
    lane.mkdir(parents=True)
    manifest = runtime / "bundle_lanes.json"
    board = tmp_path / "todo.md"
    board.write_text("## VOG-001 Work\n- Status: pending\n- Depends on:\n")
    write_json(manifest, {"todo_path": "todo.md", "task_header_prefix": "## VOG-", "report_path": "runtime/report.json",
                          "lanes": [{"bundle_key": "vog", "state_dir": "runtime/lane", "state_prefix": "vog"}]})
    (runtime / "watchdog.pid").write_text("100")
    (lane / "vog_bundle_supervisor.pid").write_text("200")
    write_json(lane / "vog_status.json", {"pid": 200, "state": "healthy", "phase": "deterministic_command",
                                          "active_task_id": "VOG-001", "heartbeat_at_ms": now * 1000,
                                          "last_task_transition_at_ms": (now - 2000) * 1000})
    (lane / "supervisor.log").write_text("live-click 1\n")
    write_json(runtime / "report.json", {"reports": [{"bundle_key": "vog", "action": "none"}]})
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Repair only the observed supervisor problem.\n")
    source = tmp_path / "source.py"
    source.write_text("version = 1\n")
    config = Config(manifest, tmp_path, runtime / "dispatcher", prompt_path=prompt,
                    evidence_paths=(source,), cooldown_seconds=10, max_cooldown_seconds=20)
    live = {100: True, 200: True}

    def probe(pid):
        return {"pid": pid, "alive": live.get(pid, False), "age_seconds": 5000, "start_ticks": pid * 10,
                "argv": ["python", "-m", "ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog",
                         "--manifest-path", str(manifest), "--repo-root", str(tmp_path)], "cwd": str(tmp_path)}

    return config, now, lane, live, probe


def status(lane, **changes):
    path = lane / "vog_status.json"
    data = json.loads(path.read_text())
    data.update(changes)
    write_json(path, data)


def kinds(observation):
    return {item["kind"] for item in observation["incidents"]}


def test_live_click_and_old_task_transition_are_healthy(rig):
    config, now, lane, live, probe = rig
    result = observe(config, {}, now=now, probe=probe)
    assert not result["incidents"]
    assert result["lanes"][0]["transition_age_seconds"] == pytest.approx(2000, abs=0.001)


def test_no_progress_requires_both_log_and_task_transition_evidence(rig):
    config, now, lane, live, probe = rig
    log = lane / "supervisor.log"
    os.utime(log, (now - 2000, now - 2000))
    assert "lane_no_progress" in kinds(observe(config, {}, now=now, probe=probe))
    status(lane, last_task_transition_at_ms=None)
    assert "lane_no_progress" not in kinds(observe(config, {}, now=now, probe=probe))


def test_dead_owner_and_lane_with_pending_work_are_concrete(rig):
    config, now, lane, live, probe = rig
    live.update({100: False, 200: False})
    assert {"owner_dead", "lane_dead_pending"} <= kinds(observe(config, {}, now=now, probe=probe))


def test_operator_stop_and_hold_marker_prevent_dispatch(rig):
    config, now, lane, live, probe = rig
    live[200] = False
    status(lane, terminal_reason="resume_stopped", state="stopped")
    result = tick(config, {}, apply=True, now=now, probe=probe, worker=lambda *_: pytest.fail("worker ran"))
    assert result["action"] == "held"
    status(lane, terminal_reason="", state="blocked")
    config.state_dir.mkdir(exist_ok=True)
    (config.state_dir / "hold").touch()
    result = tick(config, {}, apply=True, now=now, probe=probe, worker=lambda *_: pytest.fail("worker ran"))
    assert result["action"] == "held"


def test_new_owner_has_report_startup_grace(rig):
    config, now, lane, live, probe = rig
    (config.repo_root / "runtime/report.json").unlink()
    live[200] = False

    def starting(pid):
        return {**probe(pid), "age_seconds": 5}

    result = observe(config, {}, now=now, probe=starting)
    assert "owner_report_stale" not in kinds(result)
    assert "lane_dead_pending" in kinds(result)


def test_stale_owner_report_is_actionable_but_old_lane_failure_is_suppressed(rig):
    config, now, lane, live, probe = rig
    path = config.repo_root / "runtime/report.json"
    write_json(path, {"reports": [{"bundle_key": "vog", "action": "repeated_stopped_pending_repair"}]})
    os.utime(path, (now - 1000, now - 1000))
    assert kinds(observe(config, {}, now=now, probe=probe)) == {"owner_report_stale"}
    status(lane, state="blocked", phase="deterministic_resume_unavailable")
    assert {"lane_blocked", "repeated_repair", "owner_report_stale"} <= kinds(observe(config, {}, now=now, probe=probe))


def test_failed_same_signature_retries_with_persisted_cooldown_then_caps(rig):
    config, now, lane, live, probe = rig
    config.report_seconds = 100000
    live[200] = False
    calls = []

    def fail(*args):
        calls.append(args)
        return {"outcome": "failed", "exit_code": 1}

    state = {}
    first = tick(config, state, apply=True, now=now, probe=probe, worker=fail)
    assert first["action"] == "worker_finished"
    state = json.loads((config.state_dir / "state.json").read_text())
    second = tick(config, state, apply=True, now=now + 9, probe=probe, worker=fail)
    assert second["action"] == "cooldown"
    retry = tick(config, state, apply=True, now=now + 10, probe=probe, worker=fail)
    assert retry["signature"] == first["signature"]
    assert retry["action"] == "worker_finished"
    assert state["next_dispatch_at"] == now + 30
    tick(config, state, apply=True, now=now + 30, probe=probe, worker=fail)
    cap = tick(config, state, apply=True, now=now + 10000, probe=probe, worker=fail)
    assert cap["action"] == "attention_required"
    assert len(calls) == 3
    assert cap["work_complete"] is False and cap["completion_authority"] is False
    config.evidence_paths[0].write_text("version = 2\n")
    assert tick(config, state, apply=True, now=now + 10001, probe=probe, worker=fail)["action"] == "worker_finished"


def test_success_requires_fresh_observation_and_does_not_complete_work(rig):
    config, now, lane, live, probe = rig
    live[200] = False
    state = {}
    result = tick(config, state, apply=True, now=now, probe=probe,
                  worker=lambda *_: {"outcome": "awaiting_observation", "exit_code": 0})
    assert result["action"] == "worker_finished"
    assert state["unresolved_signature"]
    assert "recovery_observed_at" not in state
    live[200] = True
    recovered = tick(config, state, now=now + 1, probe=probe)
    assert recovered["action"] == "recovery_observed"
    assert not state["unresolved_signature"]
    assert json.loads((lane / "vog_status.json").read_text())["active_task_id"] == "VOG-001"


def test_gate_fixed_point_requires_persistent_sole_ready_task_and_completed_deps(rig):
    config, now, lane, live, probe = rig
    config.quiet_seconds = 30
    board = config.repo_root / "todo.md"
    board.write_text("## VOG-000 Upstream\n- Status: completed\n- Depends on:\n"
                     "## VOG-001 Closeout\n- Status: pending\n- Depends on: VOG-000\n")
    status(lane, phase="gate_withheld")
    state = {}
    assert not observe(config, state, now=now, probe=probe)["incidents"]
    # Heartbeats and log progress can continue while an impossible gate repeats.
    status(lane, heartbeat_at_ms=(now + 31) * 1000, last_task_transition_at_ms=(now + 31) * 1000)
    assert "gate_fixed_point" in kinds(observe(config, state, now=now + 31, probe=probe))
    board.write_text(board.read_text() + "## VOG-002 Other work\n- Status: pending\n- Depends on:\n")
    assert "gate_fixed_point" not in kinds(observe(config, state, now=now + 100, probe=probe))
    assert not state["gate_observations"]


def test_observational_tick_never_starts_worker(rig):
    config, now, lane, live, probe = rig
    live[200] = False
    result = tick(config, {}, now=now, probe=probe, worker=lambda *_: pytest.fail("worker ran"))
    assert result["action"] == "would_dispatch"


def test_singleton_lock_excludes_second_owner(tmp_path):
    path = tmp_path / "lock"
    with singleton_lock(path):
        with pytest.raises(AlreadyRunning):
            with singleton_lock(path):
                pytest.fail("acquired duplicate lock")
    with singleton_lock(path):
        assert path.read_text().strip() == str(os.getpid())


def test_worker_uses_stdin_and_preserves_literal_argv(rig):
    config, now, lane, live, probe = rig
    config.state_dir.mkdir(parents=True)
    output = config.state_dir / "captured.json"
    literal = "$(touch should-not-exist); `echo nope`"
    config.worker_argv = [sys.executable, "-c",
                          "import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps([sys.stdin.read(),sys.argv[2]]))",
                          str(output), literal]
    result = run_worker(config, config.state_dir / "worker.log", lambda _: None)
    assert result == {"outcome": "awaiting_observation", "exit_code": 0}
    assert json.loads(output.read_text()) == [config.prompt_path.read_text(), literal]


def test_worker_timeout_only_stops_its_owned_child(rig):
    config, now, lane, live, probe = rig
    config.state_dir.mkdir(parents=True)
    config.worker_timeout = 0.1
    config.worker_argv = [sys.executable, "-c", "import time; time.sleep(10)"]
    seen = []
    result = run_worker(config, config.state_dir / "worker.log", seen.append)
    assert result["outcome"] == "timed_out"
    assert not probe_process(seen[0]["pid"])["alive"]
    assert probe_process(os.getpid())["alive"]


def test_malformed_persisted_state_fails_closed(rig):
    config, now, lane, live, probe = rig
    config.state_dir.mkdir(parents=True)
    (config.state_dir / "state.json").write_text("bad JSON")
    assert main(["--manifest-path", str(config.manifest_path), "--repo-root", str(config.repo_root),
                 "--state-dir", str(config.state_dir), "--once"]) == 2


def test_orphaned_owned_worker_prevents_duplicate_dispatch(rig):
    config, now, lane, live, probe = rig
    live[200] = False
    live[300] = True
    state = {"active_worker": {"pid": 300, "start_ticks": 3000, "deadline": now + 100}}
    result = tick(config, state, apply=True, now=now, probe=probe, worker=lambda *_: pytest.fail("worker ran"))
    assert result["action"] == "worker_already_running"


@pytest.mark.parametrize("name", ["HOLD", "OPERATOR_STOP", "watchdog.disabled"])
@pytest.mark.parametrize("location", ["repo", "manifest"])
def test_standard_operator_holds_prevent_worker(rig, name, location):
    config, now, lane, live, probe = rig
    live[200] = False
    directory = config.repo_root if location == "repo" else config.manifest_path.parent
    (directory / name).touch()
    result = tick(config, {}, apply=True, now=now, probe=probe, worker=lambda *_: pytest.fail("worker ran"))
    assert result["action"] == "held"


def test_unrelated_reused_owner_pid_is_not_accepted_or_exposed(rig):
    config, now, lane, live, probe = rig

    def reused(pid):
        result = probe(pid)
        if pid == 100:
            result["argv"] = ["unrelated", "secret-argument"]
        return result

    result = observe(config, {}, now=now, probe=reused)
    assert "owner_dead" in kinds(result)
    assert result["owner"]["identity_verified"] is False
    assert "secret-argument" not in json.dumps(result)


def test_shared_repair_marker_is_atomic_and_excludes_legacy_fixer(rig, monkeypatch):
    config, now, lane, live, probe = rig
    path = config.manifest_path.parent / "fixer.lock"
    path.write_text("legacy fixer")
    config.repair_lock_path = path
    live[200] = False
    state = {}
    result = tick(config, state, apply=True, now=now, probe=probe, worker=lambda *_: pytest.fail("worker ran"))
    assert result["action"] == "repair_already_running"
    assert next(iter(state["attempts"].values()))["count"] == 0
    assert path.read_text() == "legacy fixer"
    path.unlink()
    original = os.link
    published = []

    def checked_link(source, target):
        assert json.loads(Path(source).read_text())["token"]
        original(source, target)
        published.append(json.loads(Path(target).read_text()))

    monkeypatch.setattr(os, "link", checked_link)
    with repair_lock(path):
        assert published[0]["pid"] == os.getpid()
        with pytest.raises(AlreadyRunning):
            with repair_lock(path):
                pytest.fail("duplicate repair lock")
    assert not path.exists()


def test_shared_marker_cleanup_does_not_remove_replacement(tmp_path):
    path = tmp_path / "fixer.lock"
    with repair_lock(path):
        path.unlink()
        path.write_text("replacement owned by another agent")
    assert path.read_text() == "replacement owned by another agent"


def test_changed_gate_evidence_restarts_quiet_period(rig):
    config, now, lane, live, probe = rig
    config.quiet_seconds = 30
    status(lane, phase="gate_withheld")
    state = {}
    observe(config, state, now=now, probe=probe)
    config.evidence_paths[0].write_text("version = 2\n")
    assert "gate_fixed_point" not in kinds(observe(config, state, now=now + 31, probe=probe))
    assert state["gate_observations"]["vog"]["first_seen"] == now + 31


def test_timeout_cleans_detached_descendant_and_preserves_unrelated_sentinel(rig):
    config, now, lane, live, probe = rig
    config.state_dir.mkdir(parents=True)
    child_pid_path = config.state_dir / "child.pid"
    config.worker_timeout = 0.8
    config.worker_argv = [sys.executable, "-c",
        "import subprocess,sys,time; from pathlib import Path; "
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True); "
        "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(30)", str(child_pid_path)]
    sentinel = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True)
    try:
        result = run_worker(config, config.state_dir / "worker.log", lambda _: None)
        assert result["outcome"] == "timed_out"
        child = int(child_pid_path.read_text())
        assert not probe_process(child)["alive"]
        assert sentinel.poll() is None
    finally:
        sentinel.terminate()
        sentinel.wait()


def test_adopted_hung_worker_deadline_is_enforced_after_restart(rig):
    config, now, lane, live, probe = rig
    live[200] = False
    child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True)
    identity = probe_process(child.pid)
    identity.update({"owned_session": True, "started_at": now - 20, "deadline": now - 1})
    state = {"active_worker": identity}

    def real_worker_probe(pid):
        return probe_process(pid) if pid == child.pid else probe(pid)

    try:
        result = tick(config, state, apply=True, now=now, probe=real_worker_probe,
                      worker=lambda *_: pytest.fail("duplicate worker ran"))
        assert result["action"] == "adopted_worker_timed_out"
        assert not state.get("active_worker")
        assert child.wait(timeout=5) == -signal.SIGTERM
        assert state["next_dispatch_at"] == now + config.cooldown_seconds
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


def test_uncertain_spawn_is_explicit_and_does_not_duplicate(rig):
    config, now, lane, live, probe = rig
    live[200] = False
    state = {"spawn_intent": {"started_at": now - 100}}
    result = tick(config, state, apply=True, now=now, probe=probe, worker=lambda *_: pytest.fail("worker ran"))
    assert result["action"] == "attention_required"
    assert result["reason"] == "spawn_uncertain_after_dispatcher_interruption"


@pytest.mark.parametrize("held", [False, True])
def test_observational_or_held_adoption_never_signals_expired_worker(rig, monkeypatch, held):
    config, now, lane, live, probe = rig
    live[300] = True
    if held:
        (config.repo_root / "HOLD").touch()
    state = {"active_worker": {"pid": 300, "start_ticks": 3000, "deadline": now - 10}}
    monkeypatch.setattr(repair_dispatcher, "stop_adopted_worker", lambda *_: pytest.fail("signalled worker"))
    result = tick(config, state, apply=held, now=now, probe=probe)
    assert result["action"] == ("held" if held else "would_timeout_adopted_worker")
    assert state["active_worker"]["pid"] == 300


def test_ready_task_and_completed_dependency_synonyms_are_selectable(rig):
    config, now, lane, live, probe = rig
    (config.repo_root / "todo.md").write_text("## VOG-000 Done\n- Status: done\n- Depends on:\n"
                                             "## VOG-001 Ready\n- Status: ready\n- Depends on: VOG-000\n")
    live[200] = False
    result = observe(config, {}, now=now, probe=probe)
    assert result["selectable_task_ids"] == ["VOG-001"]
    assert "lane_dead_pending" in kinds(result)


def test_stale_pid_file_reconciles_verified_live_owner_from_report(rig):
    config, now, lane, live, probe = rig
    live[100] = False
    live[101] = True
    write_json(config.repo_root / "runtime/report.json", {"status": {"pid": 101}, "reports": []})
    result = observe(config, {}, now=now, probe=probe)
    assert result["owner"]["pid"] == 101
    assert result["owner"]["identity_verified"] is True
    assert not result["incidents"]


def test_unrelated_report_pid_cannot_replace_dead_owner(rig):
    config, now, lane, live, probe = rig
    live[100] = False
    live[101] = True
    write_json(config.repo_root / "runtime/report.json", {"status": {"pid": 101}, "reports": []})

    def unrelated(pid):
        result = probe(pid)
        if pid == 101:
            result["argv"] = ["python", "unrelated.py", "secret-argument"]
        return result

    result = observe(config, {}, now=now, probe=unrelated)
    assert "owner_dead" in kinds(result)
    assert result["owner"]["alive"] is False
    assert "secret-argument" not in json.dumps(result)


def test_exact_proc_fallback_recovers_unpublished_owner(rig, monkeypatch):
    config, now, lane, live, probe = rig
    live[100] = False
    live[102] = True
    monkeypatch.setattr(repair_dispatcher, "discover_watchdogs", lambda *_: ([{"pid": 102}], []))
    result = observe(config, {}, now=now, probe=probe)
    assert result["owner"]["pid"] == 102
    assert not result["incidents"]


def test_multiple_verified_owners_are_uncertain_and_do_not_dispatch(rig):
    config, now, lane, live, probe = rig
    live[101] = True
    write_json(config.repo_root / "runtime/report.json", {"status": {"pid": 101}, "reports": []})
    result = tick(config, {}, apply=True, now=now, probe=probe, worker=lambda *_: pytest.fail("worker ran"))
    assert result["action"] == "owner_uncertain"
    assert result["owner_discovery_errors"] == ["multiple_verified_watchdog_owners"]


def test_lane_gate_recovery_packet_dispatches_without_another_quiet_delay(rig):
    config, now, lane, live, probe = rig
    status(lane, state="degraded", phase="gate_recovery_needed",
           gate_evidence_fingerprint="same-no-go", repair_packet="runtime/repair.json",
           next_retry_at_ms=(now + 300) * 1000, gate_recovery_needed=True)
    state = {}
    result = tick(config, state, now=now, probe=probe)
    assert result["action"] == "would_dispatch"
    assert kinds(result) == {"gate_fixed_point"}
    incident = result["incidents"][0]
    assert incident["evidence"]["repair_packet"] == "runtime/repair.json"
    assert incident["evidence"]["gate_evidence_fingerprint"] == "same-no-go"
    assert result["work_complete"] is result["completion_authority"] is False
    signature = result["signature"]
    status(lane, state="healthy", phase="gate_retry_wait")
    assert tick(config, state, now=now + 1, probe=probe)["signature"] == signature


def test_declared_gate_retry_wait_is_not_quiet_log_stall(rig):
    config, now, lane, live, probe = rig
    status(lane, phase="gate_retry_wait")
    os.utime(lane / "supervisor.log", (now - 2000, now - 2000))
    result = observe(config, {}, now=now, probe=probe)
    assert not result["incidents"]


def test_operator_hold_preserves_gate_recovery_packet_without_dispatch(rig):
    config, now, lane, live, probe = rig
    status(lane, state="degraded", phase="gate_recovery_needed", gate_recovery_needed=True)
    (config.manifest_path.parent / "OPERATOR_STOP").touch()
    result = tick(config, {}, apply=True, now=now, probe=probe,
                  worker=lambda *_: pytest.fail("worker ran under operator stop"))
    assert result["action"] == "held"


def test_regenerating_the_same_no_go_report_does_not_reset_gate_clock_or_budget(rig):
    config, now, lane, live, probe = rig
    evidence = config.repo_root / "coverage.json"
    write_json(evidence, {"decision": "no_go", "reasons": ["missing native call"], "generated_at": "first"})
    config.evidence_paths = (evidence,)
    config.quiet_seconds = 30
    status(lane, phase="gate_withheld")
    state = {}
    observe(config, state, now=now, probe=probe)
    write_json(evidence, {"decision": "no_go", "reasons": ["missing native call"], "generated_at": "regenerated"})
    first = tick(config, state, now=now + 31, probe=probe)
    assert first["action"] == "would_dispatch"
    assert state["gate_observations"]["vog"]["first_seen"] == now
    write_json(evidence, {"decision": "no_go", "reasons": ["missing native call"], "generated_at": "regenerated again"})
    assert tick(config, state, now=now + 32, probe=probe)["signature"] == first["signature"]
    write_json(evidence, {"decision": "no_go", "reasons": ["different missing proof"], "generated_at": "next"})
    assert not observe(config, state, now=now + 33, probe=probe)["incidents"]


def exhausted_wait(rig, *, attempts=3, exit_code=0, failures=3, deadline_seconds=900):
    """Write a real-shaped invocation receipt, rather than trusting a phase."""
    import hashlib
    from ipfs_accelerate_py.agent_supervisor.rescue.lane_resume import (
        _command_contract, parse_deterministic_command,
    )
    config, now, lane, live, probe = rig
    board = config.repo_root / "todo.md"
    fields = {"status": "pending", "depends_on": "", "deterministic_command": "python3 source.py",
              "validation": "python3 -c 'assert True'", "predicted_files": "source.py, coverage.json"}
    board.write_text("## VOG-001 Work\n" + "\n".join([
        "- Status: pending", "- Depends on:", "- Deterministic command: " + fields["deterministic_command"],
        "- Validation: " + fields["validation"], "- Predicted files: " + fields["predicted_files"],
    ]) + "\n")
    write_json(config.repo_root / "coverage.json", {"decision": "no_go", "generated_at": "original"})
    hashes = {name: hashlib.sha256((config.repo_root / name).read_bytes()).hexdigest()
              for name in ["source.py", "coverage.json"]}
    command = parse_deterministic_command(fields["deterministic_command"], repo_root=config.repo_root)
    validation = parse_deterministic_command(fields["validation"], repo_root=config.repo_root)
    ledger = {"VOG-001": attempts, "_retry_after_ms": {"VOG-001": int((now + deadline_seconds) * 1000)},
              "_command_evidence": {"VOG-001": {
                  "contract": _command_contract(fields, command, validation, repo_root=config.repo_root),
                  "before": hashes, "after": hashes, "exit_code": exit_code,
              }}, "_validation_fixed_points": {"VOG-001": {"failures": failures, "digest": "a" * 64}}}
    write_json(lane / "lane_resume_attempts.json", ledger)
    status(lane, phase="attempt_retry_wait", active_task_id="",
           backpressure_reasons=["VOG-001:attempts_exhausted"])
    return ledger


@pytest.mark.parametrize("exit_code", [0, 1])
def test_unchanged_exhausted_hard_failure_dispatches_after_bounded_wait(rig, exit_code):
    config, now, lane, live, probe = rig
    config.quiet_seconds = 30
    exhausted_wait(rig, exit_code=exit_code)
    original_board = (config.repo_root / "todo.md").read_text()
    state = {}
    assert not observe(config, state, now=now, probe=probe)["incidents"]
    state = json.loads(json.dumps(state))  # The escalation clock survives restart.
    status(lane, heartbeat_at_ms=(now + 31) * 1000)
    result = tick(config, state, apply=True, now=now + 31, probe=probe,
                  worker=lambda *_: {"outcome": "awaiting_observation", "exit_code": 0})
    assert result["action"] == "worker_finished"
    assert kinds(result) == {"exhausted_hard_failure"}
    assert result["incidents"][0]["task"] == "VOG-001"
    assert result["incidents"][0]["evidence"]["failure_stage"] == ("validation" if exit_code == 0 else "command")
    assert result["work_complete"] is result["completion_authority"] is False
    assert (config.repo_root / "todo.md").read_text() == original_board


@pytest.mark.parametrize("change", ["first_retry", "second_retry", "missing_receipt", "no_deadline", "gate_exit", "validator_not_exhausted", "changed_contract"])
def test_benign_or_unproven_retry_wait_does_not_escalate(rig, change):
    config, now, lane, live, probe = rig
    config.quiet_seconds = 1
    ledger = exhausted_wait(rig)
    if change == "first_retry": ledger["VOG-001"] = 1
    if change == "second_retry": ledger["VOG-001"] = 2
    if change == "missing_receipt": ledger.pop("_command_evidence")
    if change == "no_deadline": ledger["_retry_after_ms"].clear()
    if change == "gate_exit": ledger["_command_evidence"]["VOG-001"]["exit_code"] = 75
    if change == "validator_not_exhausted": ledger["_validation_fixed_points"]["VOG-001"]["failures"] = 1
    if change == "changed_contract": ledger["_command_evidence"]["VOG-001"]["contract"] = "b" * 64
    write_json(lane / "lane_resume_attempts.json", ledger)
    state = {}
    observe(config, state, now=now, probe=probe)
    assert not observe(config, state, now=now + 2, probe=probe)["incidents"]
    assert not state["exhausted_wait_observations"]


def test_exhausted_wait_signature_ignores_clocks_logs_counts_and_extended_deadline(rig):
    config, now, lane, live, probe = rig
    config.quiet_seconds = 30
    ledger = exhausted_wait(rig)
    state = {}
    observe(config, state, now=now, probe=probe)
    first = observe(config, state, now=now + 31, probe=probe)
    assert kinds(first) == {"exhausted_hard_failure"}
    write_json(config.repo_root / "coverage.json", {"decision": "no_go", "generated_at": "new clock"})
    (lane / "supervisor.log").write_text("fresh retry waiting\n")
    status(lane, heartbeat_at_ms=(now + 32) * 1000)
    ledger["VOG-001"] += 1
    ledger["_retry_after_ms"]["VOG-001"] += 900000
    write_json(lane / "lane_resume_attempts.json", ledger)
    second = observe(config, state, now=now + 32, probe=probe)
    assert second["signature"] == first["signature"]
    assert state["exhausted_wait_observations"]["vog:VOG-001"]["first_seen"] == now


@pytest.mark.parametrize("change", ["semantic_artifact", "task_transition", "source"])
def test_changed_evidence_or_real_progress_resets_exhausted_wait(rig, change):
    config, now, lane, live, probe = rig
    config.quiet_seconds = 30
    exhausted_wait(rig)
    state = {}
    observe(config, state, now=now, probe=probe)
    assert "exhausted_hard_failure" in kinds(observe(config, state, now=now + 31, probe=probe))
    if change == "semantic_artifact":
        write_json(config.repo_root / "coverage.json", {"decision": "no_go", "reasons": ["a different real gap"]})
    elif change == "task_transition":
        status(lane, last_task_transition_at_ms=(now + 32) * 1000)
    else:
        config.evidence_paths[0].write_text("new implementation\n")
    assert "exhausted_hard_failure" not in kinds(observe(config, state, now=now + 32, probe=probe))
    if change != "source":
        assert state["exhausted_wait_observations"]["vog:VOG-001"]["first_seen"] == now + 32
    else:
        assert not state["exhausted_wait_observations"], "old invocation cannot authorize repair against changed code"


def test_original_retry_deadline_bounds_escalation_even_when_extended(rig):
    config, now, lane, live, probe = rig
    ledger = exhausted_wait(rig, deadline_seconds=20)
    state = {}
    assert not observe(config, state, now=now, probe=probe)["incidents"]
    assert not observe(config, state, now=now + 19, probe=probe)["incidents"]
    ledger["_retry_after_ms"]["VOG-001"] = int((now + 900) * 1000)
    write_json(lane / "lane_resume_attempts.json", ledger)
    assert "exhausted_hard_failure" in kinds(observe(config, state, now=now + 21, probe=probe))


@pytest.mark.parametrize("location", ["global", "lane"])
def test_operator_holds_block_and_reset_exhausted_failure_dispatch(rig, location):
    config, now, lane, live, probe = rig
    config.quiet_seconds = 30
    exhausted_wait(rig)
    state = {}
    observe(config, state, now=now, probe=probe)
    hold = config.manifest_path.parent / "OPERATOR_STOP" if location == "global" else lane / "operator.hold"
    hold.touch()
    result = tick(config, state, apply=True, now=now + 31, probe=probe,
                  worker=lambda *_: pytest.fail("worker ran under intentional hold"))
    assert result["action"] == "held"
    assert not state["exhausted_wait_observations"]
    hold.unlink()
    assert not observe(config, state, now=now + 32, probe=probe)["incidents"]
