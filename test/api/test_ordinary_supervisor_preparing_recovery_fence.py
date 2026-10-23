"""Real native claim shapes and harmless children for stale-setup recovery."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as daemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_supervisor as portal
from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_runtime as runtime
from ipfs_accelerate_py.agent_supervisor.todo_daemon import portal_task_state_control_plane as plane
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop import SupervisorLoop


def _stamp(age=0):
    return datetime.fromtimestamp(time.time() - age, tz=timezone.utc).isoformat()


@pytest.fixture(autouse=True)
def _isolated_authority(tmp_path, monkeypatch):
    monkeypatch.setenv(plane.ENV_DUCKDB, str(tmp_path / "control-state.duckdb"))
    monkeypatch.setenv(plane.ENV_DUCKLAKE, str(tmp_path / "history"))
    monkeypatch.delenv(plane.ENV_QUACK, raising=False)
    for name in ("IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET", "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD"):
        monkeypatch.delenv(name, raising=False)


def _scope(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    board = repo / "tasks.todo.md"
    board.write_text("# Tasks\n\n## ACCEL-001 Setup test\n\n- Status: todo\n- Priority: P1\n- Track: ops\n- Outputs: result.json\n- Acceptance: a real result\n")
    script = repo / "benign_daemon.py"
    script.write_text("import time\ntime.sleep(60)\n")
    args = portal.parse_args(["--todo-path", str(board), "--state-dir", str(tmp_path / "state"), "--max-task-attempts", "3", "--task-prefix", "## ACCEL-"])
    config = replace(portal.supervisor_config_from_args(args, repo_root=repo),
        daemon_script_path=script, implementation_protected_paths=(),
        stale_seconds=120, implementation_timeout=900, implementation_log_stall_seconds=180,
        worktree_root=tmp_path / "worktrees", check_interval=1)
    supervisor = portal.PortalImplementationSupervisor(config)
    native = daemon.PortalImplementationDaemon(todo_path=board, state_path=config.state_path,
        strategy_path=config.strategy_path, events_path=tmp_path / "daemon_events.jsonl",
        repo_root=repo, task_header_prefix="## ACCEL-", implementation_protected_paths=())
    task = daemon.parse_task_file(board, task_header_prefix="## ACCEL-")[0]
    identity = native._identity_for_task(task)
    state = daemon.PortalTaskState(active_task_id=task.task_id,
        active_task_key=identity.canonical_task_key, active_task_cid=identity.canonical_task_cid,
        active_task_started_at=_stamp(160), heartbeat_at=_stamp(160), last_progress_at=_stamp(160),
        implementation_attempts={task.task_id: 2}, implementation_attempts_by_cid={identity.canonical_task_cid: 2},
        ready_count=1, task_statuses={task.task_id: "todo"})
    state.save(config.state_path)
    return supervisor, native, task, state


@contextmanager
def _child(supervisor):
    configured = supervisor.build_supervisor_loop_config()
    spec = SupervisorLoop(configured)._child_spec("synthetic-preparing-proof")
    child = runtime.launch_supervised_child(spec)
    birth = runtime.read_process_birth(child.pid)
    assert birth is not None
    try:
        deadline = time.monotonic() + 5
        while runtime.read_process_command_argv(child.pid) != tuple(spec.command):
            assert time.monotonic() < deadline
            time.sleep(.01)
        yield child
    finally:
        if runtime.read_process_birth(child.pid) == birth:
            os.kill(child.pid, signal.SIGTERM)
        try:
            os.waitpid(child.pid, 0)
        except ChildProcessError:
            pass


def _claims(supervisor, native, task, state, child, age=100):
    if age > 160:
        state.active_task_started_at = _stamp(age + 30)
        state.save(supervisor.config.state_path)
    started = _stamp(age)
    # Use the native builders, preserving all canonical/source metadata. The
    # harmless child's real birth is the synthetic fixture's explicit owner.
    native._implementation_dispatch_process_birth = child.identity_process_birth
    claim = native._build_implementation_task_claim_metadata(task, 3, started)
    claim["pid"] = child.pid
    claim["dispatch_authority_cid"] = native._implementation_dispatch_authority_cid(claim)
    lease = native._build_implementation_lock_metadata(task, 3, started)
    lease["pid"] = child.pid
    for path, body in ((supervisor._task_dispatch_claim_path(state), claim),
                       (supervisor._implementation_maintenance_lock_path(), lease)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body))
        path.chmod(0o600)
    return claim, lease


def test_owned_native_preparing_claim_is_not_a_stale_selected_heartbeat(tmp_path):
    supervisor, native, task, state = _scope(tmp_path)
    with _child(supervisor) as child:
        claim, lease = _claims(supervisor, native, task, state, child)
        proof = supervisor._owned_setup_status(state, now_ts=time.time())
        assert proof["owned"] and proof["active"]
        assert proof["deadline_seconds"] == 180
        assert proof["dispatch_authority_cid"] == claim["dispatch_authority_cid"]
        assert supervisor.is_stuck(state, now_ts=time.time()) == (False, "")
        assert asdict(daemon.PortalTaskState.load(supervisor.config.state_path)) == asdict(state)
        # Repeated probes cannot slide the fixed setup deadline.
        later = supervisor.is_stuck(state, now_ts=time.time() + 100)
        assert later[0] and later[1].startswith("implementation setup deadline exceeded")


@pytest.mark.parametrize("corruption", ["missing_claim", "foreign_birth", "foreign_source", "foreign_lease", "changed_authority", "future_started"])
def test_setup_grace_requires_exact_native_owner_source_and_time(tmp_path, corruption):
    supervisor, native, task, state = _scope(tmp_path)
    with _child(supervisor) as child:
        claim, lease = _claims(supervisor, native, task, state, child)
        if corruption == "missing_claim":
            supervisor._task_dispatch_claim_path(state).unlink()
        else:
            if corruption == "foreign_birth": claim["owner_process_birth"]["start_time_ticks"] += 1
            if corruption == "foreign_source": claim["canonical_task_cid"] = "foreign"
            if corruption == "foreign_lease": lease["lease_id"] = ""
            if corruption == "changed_authority": claim["dispatch_authority_cid"] = "untrusted"
            if corruption == "future_started": claim["started_at"] = lease["started_at"] = _stamp(-60)
            supervisor._task_dispatch_claim_path(state).write_text(json.dumps(claim))
            supervisor._implementation_maintenance_lock_path().write_text(json.dumps(lease))
        assert not supervisor._owned_setup_status(state, now_ts=time.time())["owned"]
        assert supervisor.is_stuck(state, now_ts=time.time())[0]


def test_empty_protected_paths_still_fence_live_implementation_lease(tmp_path, monkeypatch):
    supervisor, native, task, state = _scope(tmp_path)
    assert supervisor.config.implementation_protected_paths == ()
    with _child(supervisor) as child:
        _claims(supervisor, native, task, state, child)
        before = plane.read_portal_task_state_snapshot(supervisor.config.state_path)
        monkeypatch.setattr(supervisor, "_run_once_with_maintenance_under_lease", lambda *a, **k: pytest.fail("maintenance entered live setup"))
        result = supervisor._run_once_with_maintenance(lambda _: None)
        assert result["maintenance_blocked"]
        assert plane.read_portal_task_state_snapshot(supervisor.config.state_path) == before


def test_watchdog_quiesces_owned_child_before_any_stale_maintenance(tmp_path, monkeypatch):
    supervisor, native, task, state = _scope(tmp_path)
    with _child(supervisor) as child:
        _claims(supervisor, native, task, state, child, age=190)
        seen = []
        monkeypatch.setattr(supervisor, "_watchdog_container_quiescence", lambda _: {"quiesced": True})
        def maintain(_update):
            identity = runtime.load_supervised_child_identity(child.identity_path)
            assert runtime._supervised_child_identity_session_quiesced(identity)
            seen.append("maintenance after quiescence")
            return {"stuck": True, "reason": "bounded setup expired"}
        monkeypatch.setattr(supervisor, "_run_once_with_maintenance", maintain)
        decision = supervisor._supervisor_loop_watchdog_decision(None, child, {})
        assert decision.action == "recycle" and seen
        # The reusable loop must accept exact already-quiesced custody, without
        # attempting another signal to a now-reusable numeric PID.
        loop = SupervisorLoop(supervisor.build_supervisor_loop_config())
        assert loop._terminate_child(child)
        assert daemon.PortalTaskState.load(supervisor.config.state_path).implementation_attempts == state.implementation_attempts


@pytest.mark.parametrize("blocked", ["identity", "fence", "container_before", "container_after"])
def test_unproven_quiescence_never_enters_recovery(tmp_path, monkeypatch, blocked):
    supervisor, native, task, state = _scope(tmp_path)
    with _child(supervisor) as child:
        _claims(supervisor, native, task, state, child, age=190)
        before = plane.read_portal_task_state_snapshot(supervisor.config.state_path)
        monkeypatch.setattr(supervisor, "_run_once_with_maintenance", lambda *a: pytest.fail("unproven child entered maintenance"))
        probes = iter([{"quiesced": blocked != "container_before"}, {"quiesced": blocked != "container_after"}])
        monkeypatch.setattr(supervisor, "_watchdog_container_quiescence", lambda _: next(probes))
        if blocked == "identity": child.identity_path.unlink()
        if blocked == "fence": monkeypatch.setattr(supervisor, "_fence_recorded_managed_daemon", lambda **_: {"fenced": False})
        decision = supervisor._supervisor_loop_watchdog_decision(None, child, {})
        assert decision.action == "block"
        assert plane.read_portal_task_state_snapshot(supervisor.config.state_path) == before
        assert not supervisor.config.strategy_path.exists() or not (json.loads(supervisor.config.strategy_path.read_text()).get("autonomous_unstall_quarantines"))


def test_start_committed_during_fence_preserves_new_state_counts_and_strategy(tmp_path, monkeypatch):
    supervisor, native, task, state = _scope(tmp_path)
    with _child(supervisor) as child:
        _claims(supervisor, native, task, state, child, age=190)
        original = supervisor._fence_recorded_managed_daemon
        started = replace(state, implementation_in_progress=True, active_phase="implementing", active_attempt=3,
            last_implementation_started_at=_stamp(), heartbeat_at=_stamp(),
            implementation_attempts={task.task_id: 3}, implementation_attempts_by_cid={state.active_task_cid: 3})
        def fence(**kwargs):
            started.save(supervisor.config.state_path)
            return original(**kwargs)
        monkeypatch.setattr(supervisor, "_fence_recorded_managed_daemon", fence)
        monkeypatch.setattr(supervisor, "_watchdog_container_quiescence", lambda _: {"quiesced": True})
        monkeypatch.setattr(supervisor, "_run_once_with_maintenance", lambda *a: pytest.fail("new start was overwritten by old diagnosis"))
        decision = supervisor._supervisor_loop_watchdog_decision(None, child, {})
        assert decision.reason == "state_changed_before_stale_recovery"
        assert asdict(daemon.PortalTaskState.load(supervisor.config.state_path)) == asdict(started)


@pytest.mark.parametrize("board_store", [False, True])
def test_atomic_state_cas_keeps_a_concurrent_dispatch_and_returns_current(tmp_path, board_store):
    path = tmp_path / "state" / "lane.json"
    target = tmp_path / "board-control-plane" / "control.duckdb" if board_store else tmp_path / "lane.duckdb"
    plane.bind_task_state_control_plane(path, str(target))
    initial = {"active_task_id": "A", "active_task_cid": "cid-A", "active_attempt": 0, "source_revision": "source-1"}
    plane.save_portal_task_state(path.stem, initial, state_path=path)
    stale = plane.read_portal_task_state_snapshot(path)
    newer = {**initial, "active_attempt": 3, "implementation_in_progress": True, "source_revision": "source-2"}
    plane.save_portal_task_state(path.stem, newer, state_path=path)
    result = plane.compare_and_save_portal_task_state(path.stem, {**initial, "active_task_id": ""},
        expected=stale["body"], expected_revision=stale["revision"], state_path=path)
    assert not result["matched"] and result["current"] == {**newer, "completion_authority": False}
    assert plane.read_portal_task_state_snapshot(path)["body"] == result["current"]
    fresh = plane.read_portal_task_state_snapshot(path)
    # The same snapshot succeeds once, and its old revision cannot replay.
    saved = plane.compare_and_save_portal_task_state(path.stem, {**newer, "ack": True},
        expected=fresh["body"], expected_revision=fresh["revision"], state_path=path)
    assert saved["matched"] and saved["revision"] > fresh["revision"]
    assert not plane.compare_and_save_portal_task_state(path.stem, initial,
        expected=fresh["body"], expected_revision=fresh["revision"], state_path=path)["matched"]


def test_repair_compare_miss_preserves_state_and_never_rewrites_strategy(tmp_path, monkeypatch):
    supervisor, native, task, stale = _scope(tmp_path)
    newer = replace(stale, active_attempt=3, implementation_in_progress=True, active_phase="implementing",
        heartbeat_at=_stamp(), last_implementation_started_at=_stamp(),
        implementation_attempts={task.task_id: 3}, implementation_attempts_by_cid={stale.active_task_cid: 3})
    real = plane.compare_and_save_portal_task_state
    def concurrent(*args, **kwargs):
        newer.save(supervisor.config.state_path)
        return real(*args, **kwargs)
    monkeypatch.setattr(plane, "compare_and_save_portal_task_state", concurrent)
    strategy_before = supervisor.config.strategy_path.read_bytes() if supervisor.config.strategy_path.exists() else None
    result = supervisor.repair_blocked_progress_state(stale, "heartbeat stale", now_ts=time.time())
    assert not result["repaired"]
    assert asdict(daemon.PortalTaskState.load(supervisor.config.state_path)) == asdict(newer)
    assert (supervisor.config.strategy_path.read_bytes() if supervisor.config.strategy_path.exists() else None) == strategy_before


@pytest.mark.parametrize("hold", ["waiting", "blocked", "reserved", "unknown_status", "strategy"])
def test_automatic_stale_recovery_preserves_holds_and_spent_attempts(tmp_path, hold):
    supervisor, native, task, state = _scope(tmp_path)
    if hold == "waiting": state.waiting_task_ids = [task.task_id]
    if hold == "blocked": state.blocked_task_ids = [task.task_id]
    if hold == "reserved": state.external_reserved_task_ids = [task.task_id]
    if hold == "unknown_status": state.task_statuses[task.task_id] = "awaiting_user"
    if hold == "strategy":
        supervisor.config.strategy_path.parent.mkdir(parents=True, exist_ok=True)
        supervisor.config.strategy_path.write_text(json.dumps({"blocked_tasks": [task.task_id], "autonomous_unstall_quarantines": [{"incident_cid": "retained", "task_id": task.task_id}]}))
    state.save(supervisor.config.state_path)
    before = plane.read_portal_task_state_snapshot(supervisor.config.state_path)
    result = supervisor._run_autonomous_unstall(state, "heartbeat stale")
    assert result["status"] == "deferred" and not result["quarantined"]
    assert plane.read_portal_task_state_snapshot(supervisor.config.state_path) == before


def test_missing_cas_authority_never_inserts_a_lane(tmp_path):
    path = tmp_path / "state.json"
    target = tmp_path / "empty.duckdb"
    plane.bind_task_state_control_plane(path, str(target))
    plane.save_portal_task_state("other", {"exists": True}, state_path=path)
    result = plane.compare_and_save_portal_task_state(path.stem, {"x": 2}, expected={"x": 1}, expected_revision=1, state_path=path)
    assert not result["matched"] and result["current"] is None
    assert plane.load_portal_task_state(path.stem, state_path=path, create=False) == {}
    with pytest.raises(plane.PortalTaskStateControlPlaneError):
        plane.compare_and_save_portal_task_state(path.stem, {"x": 2}, expected={}, expected_revision=1, state_path=path)


@pytest.mark.parametrize("bad", ["malformed", "foreign_scope", "missing_pid", "symlink"])
def test_unknown_implementation_lease_is_preserved_with_no_maintenance(tmp_path, monkeypatch, bad):
    supervisor, native, task, state = _scope(tmp_path)
    path = supervisor._implementation_maintenance_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    value = native._build_implementation_lock_metadata(task, 3, _stamp(300))
    if bad == "foreign_scope": value["state_dir"] = str(tmp_path / "foreign")
    if bad == "missing_pid": value.pop("pid")
    payload = b"{" if bad == "malformed" else json.dumps(value).encode()
    if bad == "symlink":
        target = tmp_path / "foreign-lease"
        target.write_bytes(payload)
        path.symlink_to(target)
    else:
        path.write_bytes(payload)
        path.chmod(0o600)
    monkeypatch.setattr(supervisor, "_run_once_with_maintenance_under_lease", lambda *a, **k: pytest.fail("unknown lease was replaced"))
    result = supervisor._run_once_with_maintenance(lambda _: None)
    assert result["maintenance_blocked"] and path.read_bytes() == payload
    if bad == "symlink": assert path.is_symlink()


def test_successful_stale_state_cas_preserves_lifetime_consumption(tmp_path):
    supervisor, native, task, state = _scope(tmp_path)
    state.active_attempt = 1  # Lifetime count may exceed the old local ordinal.
    state.implementation_in_progress = True
    state.last_implementation_started_at = _stamp(1000)
    state.heartbeat_at = _stamp(1000)
    state.implementation_attempts = {task.task_id: 4}
    state.implementation_attempts_by_cid = {state.active_task_cid: 4}
    state.save(supervisor.config.state_path)
    result = supervisor.repair_blocked_progress_state(state, "heartbeat stale", now_ts=time.time())
    assert result["repaired"]
    fresh = daemon.PortalTaskState.load(supervisor.config.state_path)
    assert not fresh.active_task_id and not fresh.implementation_in_progress
    assert fresh.implementation_attempts[task.task_id] == fresh.implementation_attempts_by_cid[state.active_task_cid] == 4
    assert not result["attempt_recovery"]["consumed"]


def test_generic_loop_stale_heartbeat_uses_native_setup_exception(tmp_path):
    supervisor, native, task, state = _scope(tmp_path)
    state.active_task_started_at = _stamp(1100)
    state.heartbeat_at = state.last_progress_at = _stamp(1000)
    state.save(supervisor.config.state_path)
    with _child(supervisor) as child:
        _claims(supervisor, native, task, state, child, age=100)
        loop = SupervisorLoop(supervisor.build_supervisor_loop_config(),
            watchdog_hook=supervisor._supervisor_loop_watchdog_decision,
            stale_heartbeat_hook=supervisor._supervisor_loop_watchdog_decision)
        old = time.time() - 1200
        os.utime(child.log_path, (old, old))
        current_status = {"heartbeat_at": _stamp(1200)}
        portal.write_json_atomic(loop.config.spec.resolve(loop.config.spec.status_path), current_status)
        # Exercise the default watchdog before the custom override; merely
        # calling is_stuck would not cover the generic stale-heartbeat path.
        assert loop.default_watchdog(child, current_status).reason == "stale_heartbeat"
        assert loop.watchdog_decision(child).action == "continue"
        assert runtime.supervised_child_identity_liveness(runtime.load_supervised_child_identity(child.identity_path)) is runtime.OwnerLiveness.ALIVE


@pytest.mark.parametrize("inventory", ["empty", "foreign", "owned", "malformed", "unavailable"])
def test_container_quiescence_observation_is_bounded_and_never_removes(tmp_path, monkeypatch, inventory):
    supervisor, native, task, state = _scope(tmp_path)
    from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner
    monkeypatch.setattr(grok_cli_runner, "_docker_isolation_binary", lambda: "/usr/bin/docker")
    calls = []
    identifier = "a" * 64
    def run(command, **kwargs):
        calls.append(command)
        assert command[:2] == ["/usr/bin/docker", "--host=unix:///var/run/docker.sock"]
        assert kwargs["timeout"] <= 3 and kwargs["stdin"] == subprocess.DEVNULL
        assert not any(word in command for word in ("kill", "stop", "rm", "remove"))
        if inventory == "unavailable": raise subprocess.TimeoutExpired(command, 3)
        if "ps" in command:
            output = "" if inventory == "empty" else ("not-an-id\n" if inventory == "malformed" else identifier + "\n")
        else:
            root = supervisor.config.worktree_root if inventory == "owned" else tmp_path / "another-campaign"
            output = json.dumps([{"Type": "bind", "Source": str(root / "workspace"), "Destination": "/workspace"}])
        return subprocess.CompletedProcess(command, 0, output, "")
    monkeypatch.setattr(portal.subprocess, "run", run)
    observed = supervisor._watchdog_container_quiescence(state)
    assert observed["quiesced"] is (inventory in {"empty", "foreign"})
    assert len(calls) <= 3


def test_cas_revision_rejects_aba_with_identical_current_body(tmp_path):
    path = tmp_path / "aba.json"
    plane.save_portal_task_state(path.stem, {"active_task_id": "A"}, state_path=path)
    old = plane.read_portal_task_state_snapshot(path)
    plane.save_portal_task_state(path.stem, {"active_task_id": "B"}, state_path=path)
    plane.save_portal_task_state(path.stem, {"active_task_id": "A"}, state_path=path)
    result = plane.compare_and_save_portal_task_state(path.stem, {"active_task_id": ""},
        expected=old["body"], expected_revision=old["revision"], state_path=path)
    assert not result["matched"] and result["revision"] > old["revision"]


@pytest.mark.parametrize("hold", ["raw_status", "manual_metadata", "protected_snapshot", "protected_incident", "source_revision"])
def test_source_and_protected_holds_cannot_be_cleared_by_stale_repair(tmp_path, hold):
    supervisor, native, task, state = _scope(tmp_path)
    if hold == "raw_status":
        supervisor.config.todo_path.write_text(supervisor.config.todo_path.read_text().replace("Status: todo", "Status: awaiting_user"))
    if hold == "manual_metadata":
        supervisor.config.todo_path.write_text(supervisor.config.todo_path.read_text() + "\n- Manual hold: true\n")
    if hold == "source_revision":
        supervisor.config.todo_path.write_text(supervisor.config.todo_path.read_text().replace("a real result", "a different source revision"))
    if hold.startswith("protected_"):
        supervisor.config.state_path.parent.mkdir(parents=True, exist_ok=True)
        name = portal.IMPLEMENTATION_PROTECTED_ACTIVE_SNAPSHOT_FILENAME if hold == "protected_snapshot" else portal.IMPLEMENTATION_PROTECTED_INCIDENT_FILENAME
        (supervisor.config.state_path.parent / name).write_text("{\"retained\":true}\n")
    before = plane.read_portal_task_state_snapshot(supervisor.config.state_path)
    source = supervisor.config.todo_path.read_bytes()
    result = supervisor.repair_blocked_progress_state(state, "heartbeat stale", now_ts=time.time())
    assert not result["repaired"]
    assert plane.read_portal_task_state_snapshot(supervisor.config.state_path) == before
    assert supervisor.config.todo_path.read_bytes() == source


def test_retained_dispatch_intent_does_not_slide_setup_deadline(tmp_path):
    supervisor, native, task, state = _scope(tmp_path)
    with _child(supervisor) as child:
        claim, lease = _claims(supervisor, native, task, state, child, age=100)
        lease["started_at"] = _stamp(10)
        supervisor._implementation_maintenance_lock_path().write_text(json.dumps(lease))
        proof = supervisor._owned_setup_status(state, now_ts=time.time())
        assert proof["owned"] and proof["active"] and proof["age_seconds"] >= 100
        assert supervisor.is_stuck(state, now_ts=time.time() + 90)[0]


def test_changed_dispatch_claim_cannot_be_cleared_with_an_old_snapshot(tmp_path, monkeypatch):
    supervisor, native, task, state = _scope(tmp_path)
    with _child(supervisor) as child:
        claim, lease = _claims(supervisor, native, task, state, child, age=190)
        assert supervisor._fence_recorded_managed_daemon(pid=child.pid,
            expected_identity_record_id=child.identity_record_id)["fenced"]
        before = plane.read_portal_task_state_snapshot(supervisor.config.state_path)
        original = supervisor._compare_and_save_stale_recovery
        changed = {**claim, "lease_id": "a-new-claim-is-not-the-old-authority"}
        def change_claim(*args, **kwargs):
            supervisor._task_dispatch_claim_path(state).write_text(json.dumps(changed))
            return original(*args, **kwargs)
        monkeypatch.setattr(supervisor, "_compare_and_save_stale_recovery", change_claim)
        result = supervisor.repair_blocked_progress_state(state, "heartbeat stale", now_ts=time.time())
        assert not result["repaired"]
        assert plane.read_portal_task_state_snapshot(supervisor.config.state_path) == before
        assert json.loads(supervisor._task_dispatch_claim_path(state).read_text()) == changed


def test_unreadable_strategy_cannot_be_replaced_by_stale_recovery(tmp_path):
    supervisor, native, task, state = _scope(tmp_path)
    supervisor.config.strategy_path.parent.mkdir(parents=True, exist_ok=True)
    supervisor.config.strategy_path.write_bytes(b"{\"blocked_tasks\":")
    before = plane.read_portal_task_state_snapshot(supervisor.config.state_path)
    result = supervisor._run_autonomous_unstall(state, "heartbeat stale")
    assert result["status"] == "deferred" and not result["quarantined"]
    assert supervisor.config.strategy_path.read_bytes() == b"{\"blocked_tasks\":"
    assert plane.read_portal_task_state_snapshot(supervisor.config.state_path) == before


def test_real_loop_retains_live_owner_and_markers_when_custody_is_blocked(tmp_path):
    supervisor, native, task, state = _scope(tmp_path)
    config = replace(supervisor.build_supervisor_loop_config(),
        watchdog_startup_grace_seconds=0, heartbeat_seconds=.01, poll_seconds=.01)
    retained = []
    def blocked(_loop, child, _status):
        retained.append(child)
        return portal.SupervisorLoopDecision.block("workspace_container_present")
    loop = SupervisorLoop(config, watchdog_hook=blocked, stale_heartbeat_hook=blocked)
    try:
        result = loop.run()
        assert result.status == "termination_blocked" and result.restart_count == 0
        assert len(retained) == 1
        child = retained[0]
        assert runtime.read_process_birth(child.pid) == child.identity_process_birth
        assert child.identity_path.exists() and child.child_pid_path.exists()
        assert asdict(daemon.PortalTaskState.load(supervisor.config.state_path)) == asdict(state)
    finally:
        for child in retained:
            assert runtime.terminate_supervised_child(child, grace_seconds=1)
            runtime.wait_for_child_exit(child)


def test_real_coordinator_cas_race_defers_without_false_isolation_or_quarantine(tmp_path, monkeypatch):
    supervisor, native, task, stale = _scope(tmp_path)
    newer = replace(stale, implementation_in_progress=True, active_phase="implementing", active_attempt=3,
        last_implementation_started_at=_stamp(), heartbeat_at=_stamp(),
        implementation_attempts={task.task_id: 3}, implementation_attempts_by_cid={stale.active_task_cid: 3})
    supervisor.config.strategy_path.parent.mkdir(parents=True, exist_ok=True)
    supervisor.config.strategy_path.write_text(json.dumps({"autonomous_unstall_policy": {
        "deterministic_max_total_attempts": 1, "deterministic_max_actions": 1,
        "cooldown_ms": 0, "deadline_ms": 1000}}))
    strategy_before = supervisor.config.strategy_path.read_bytes()
    real = plane.compare_and_save_portal_task_state
    def competing_start(*args, **kwargs):
        newer.save(supervisor.config.state_path)
        return real(*args, **kwargs)
    monkeypatch.setattr(plane, "compare_and_save_portal_task_state", competing_start)
    result = supervisor._run_autonomous_unstall(stale, "heartbeat stale for active task ACCEL-001")
    assert result["status"] == "deferred" and not result["recovered"] and not result["quarantined"]
    assert not result["completion_authority"]
    assert asdict(daemon.PortalTaskState.load(supervisor.config.state_path)) == asdict(newer)
    assert supervisor.config.strategy_path.read_bytes() == strategy_before
    incident_before = supervisor._autonomous_unstall_state_path().read_bytes()
    again = supervisor._run_autonomous_unstall(stale, "heartbeat stale for active task ACCEL-001")
    assert again["status"] == "deferred" and not again["recovered"] and not again["quarantined"]
    assert supervisor._autonomous_unstall_state_path().read_bytes() == incident_before
    assert supervisor._autonomous_unstall_status()["latest"]["phase"] == "deferred"


def test_explicit_scope_refusal_retains_finite_coordinator_action_budget(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import RescueOperation, prompt_workflow_cid
    from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog import AutonomousUnstallCoordinator, AutonomousUnstallPolicy
    calls = []
    cid = lambda value: prompt_workflow_cid({"synthetic-refusal-fixture": value})
    def retry(_context):
        calls.append("retry")
        return {"succeeded": False, "reason": "task_state_compare_failed"}
    coordinator = AutonomousUnstallCoordinator(state_dir=tmp_path / "incident",
        repository_root=tmp_path, repository_root_cid=cid("repo"), policy_root=cid("policy"), run_cid=cid("run"),
        policy=AutonomousUnstallPolicy(deterministic_max_total_attempts=1, deterministic_max_actions=1, cooldown_ms=0),
        recovery_handlers={RescueOperation.RETRY: retry}, health_probe=lambda: {"healthy": False},
        quarantine_scope=lambda *args: {"quarantined": False, "reason": "stale_recovery_authority_changed"},
        clock_ms=lambda: 10000)
    evidence = {"heartbeat": {"lane_id": "synthetic", "stale": True}}
    first = coordinator.unstall(evidence=evidence)
    second = coordinator.unstall(evidence=evidence)
    assert calls == ["retry"]
    for result in (first, second):
        assert result["status"] == "deferred" and not result["recovered"] and not result["quarantined"]
        assert not result["completion_authority"] and not result["work_complete"]


def test_quarantine_action_compare_miss_is_not_successful_isolation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import RescueOperation
    supervisor, native, task, state = _scope(tmp_path)
    # An independent writer has deliberately removed the old selected task.
    # That cannot be reported as the failed quarantine handler's own effect.
    newer = replace(state, active_task_id="", active_task_key="", active_task_cid="")
    real = plane.compare_and_save_portal_task_state
    def competing_write(*args, **kwargs):
        newer.save(supervisor.config.state_path)
        return real(*args, **kwargs)
    monkeypatch.setattr(plane, "compare_and_save_portal_task_state", competing_write)
    class QuarantineActionProbe:
        def __init__(self, *, recovery_handlers, health_probe, **kwargs):
            self.handler, self.health = recovery_handlers[RescueOperation.QUARANTINE], health_probe
        def unstall(self, **kwargs):
            observation = self.handler(SimpleNamespace(
                incident=SimpleNamespace(target_ids=(task.task_id,), incident_cid="synthetic-unit-action"),
                action=SimpleNamespace(expected_effects=("scope_fenced",))))
            health = self.health()
            assert not observation["succeeded"] and observation["observed_effects"] == ()
            assert not health["healthy"] and not health["affected_task_isolated"]
            return {"status": "deferred", "recovered": False, "quarantined": False, "health": health}
    supervisor.autonomous_unstall_coordinator_class = QuarantineActionProbe
    result = supervisor._run_autonomous_unstall(state, "heartbeat stale")
    assert not result["state_repair"]["repaired"] and not result["state_repair"]["quarantined"]
    assert asdict(daemon.PortalTaskState.load(supervisor.config.state_path)) == asdict(newer)
