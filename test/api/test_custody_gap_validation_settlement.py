"""A stored custody gap may settle one sealed validation failure, never an unknown callback."""
import subprocess
import sys
import time

from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalImplementationDaemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import (
    _BOUND_STATE_TARGETS,
    resolve_control_plane,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_cli_subreaper import (
    _direct_child_pids,
    _process_is_zombie,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_provider_custody_observation import (
    append_native_provider_custody_check,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_validation_failure_retry import _proof
from test.api.test_native_validation_failure_retry import _chain


def _diagnostic(checks, *, phase="unknown_callback", reason="portal_provider_failed", present=False, custody=True):
    observation = None
    if custody:
        for stage, reason_code in checks:
            observation = append_native_provider_custody_check(observation, stage, reason_code)
    return {
        "phase": phase,
        "reason_code": reason,
        "completion_authority": False,
        "retry_authority": False,
        "settlement_authority": False,
        "callback": {"state": "started_outcome_unknown", "native_exit": {"present": present}},
        "native_provider_custody_observation": observation,
    }


_GAP = (
    ("issuance", "eligible"),
    ("native_exit", "custody_not_established"),
    ("bridge_join", "native_exit_unavailable"),
    ("capability_take", "not_issued"),
)


def test_only_the_stored_custody_gap_admits_replay():
    admit = DatabasePortalExecutionBridge.custody_gap_diagnostics_admit_replay
    gap = _diagnostic(_GAP)
    assert admit([gap]) is True
    later = _diagnostic((), reason="other", custody=False)
    assert admit([gap, later]) is True
    assert admit([]) is False
    assert admit([_diagnostic((("native_exit", "not_issued"),))]) is False
    assert admit([_diagnostic(_GAP, phase="terminal_failure")]) is False
    assert admit([_diagnostic(_GAP, present=True)]) is False
    verified = _diagnostic((("issuance", "eligible"), ("native_exit", "verified")))
    assert admit([verified]) is False
    unstable = (
        ("issuance", "eligible"),
        ("native_exit", "child_census_not_empty_or_unstable"),
        ("bridge_join", "native_exit_unavailable"),
        ("capability_take", "not_issued"),
    )
    assert admit([_diagnostic(unstable)]) is True
    assert admit([_diagnostic((("issuance", "eligible"), ("native_exit", "subreaper_disabled")))]) is False
    assert admit([_diagnostic((("issuance", "eligible"), ("native_exit", "child_census_unavailable")))]) is False


def test_zombie_child_is_reaped_and_a_live_child_still_blocks_custody():
    zombie = subprocess.Popen([sys.executable, "-c", "raise SystemExit(0)"], start_new_session=True)
    live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        deadline = time.time() + 5
        while time.time() < deadline and not _process_is_zombie(zombie.pid):
            time.sleep(0.01)
        assert _process_is_zombie(zombie.pid)
        assert PortalImplementationDaemon._begin_database_provider_child_custody() is None
        assert live.poll() is None
        assert zombie.pid not in _direct_child_pids()
        live.kill()
        deadline = time.time() + 5
        while time.time() < deadline and live.pid in _direct_child_pids() and not _process_is_zombie(live.pid):
            time.sleep(0.01)
        assert PortalImplementationDaemon._begin_database_provider_child_custody() is not None
        assert live.pid not in _direct_child_pids()
    finally:
        if live.poll() is None:
            live.kill()
            try:
                live.wait(timeout=5)
            except ChildProcessError:
                pass
        try:
            zombie.wait(timeout=1)
        except ChildProcessError:
            pass


def test_custody_gap_known_validation_failure_settles_and_can_rearm(tmp_path, monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(tmp_path / "orchestration"))
    owner, _bridge, portals = _chain(tmp_path)
    script = tmp_path / "router_timeout.py"
    script.write_text("from pathlib import Path\nPath('result.py').write_text('VALUE = 1\\n')\n")
    owner.max_task_attempts = 3
    monkeypatch.setattr(
        PortalImplementationDaemon,
        "_begin_database_provider_child_custody",
        staticmethod(lambda: None),
    )
    try:
        first = owner.run_once()
        attempt = owner.get_attempt(first["attempt_id"])
        callback = owner.provider_invocation_recorded(
            attempt.attempt_id, idempotency_key="provider:" + attempt.attempt_id)
        assert attempt.status == "running"
        assert callback["callback_state"] == "started_outcome_unknown"
        assert _proof(owner, attempt) is None
        original_empty = PortalImplementationDaemon._database_provider_children_empty
        monkeypatch.setattr(
            PortalImplementationDaemon,
            "_database_provider_children_empty",
            staticmethod(lambda: False),
        )
        blocked = owner.run_once()
        assert blocked["implementation_result"]["reason"] == "provider_callback_outcome_unknown"
        assert owner.get_attempt(attempt.attempt_id).status == "running"
        monkeypatch.setattr(PortalImplementationDaemon, "_database_provider_children_empty", original_empty)
        # A restarted supervisor does not keep the portal factory's process-local
        # binding. The sealed attempt database has to be enough on its own.
        saved_targets = dict(_BOUND_STATE_TARGETS)
        _BOUND_STATE_TARGETS.clear()
        attempt_paths = _bridge._paths(attempt)
        assert (attempt_paths.root / "portal-state.duckdb").is_file()
        assert resolve_control_plane(attempt_paths.state)[1] != "bound"
        try:
            settled = owner.run_once()
        finally:
            _BOUND_STATE_TARGETS.clear()
            _BOUND_STATE_TARGETS.update(saved_targets)
        assert settled["implementation_result"]["portal_terminal_failure"] is True
        failed = owner.get_attempt(attempt.attempt_id)
        assert failed.status == "failed"
        receipt = owner.provider_invocation_recorded(
            failed.attempt_id, idempotency_key="provider:" + failed.attempt_id)
        assert receipt["callback_state"] == "failed_outcome_settled"
        assert receipt["native_exit"]["schema"] == "database-native-provider-exit@2"
        assert receipt["native_exit"]["provider_returncode"] == 0
        assert receipt["native_exit"]["returncode"] != 0
        task = owner.task_source.get_task(failed.task_cid)
        assert task.status == "blocked"
        assert getattr(owner, "native_validation_failure_retry_enabled", False) is not True
        assert _proof(owner, failed) is None
        owner.native_validation_failure_retry_enabled = True
        assert _proof(owner, failed) is not None
        rearmed = owner.reconcile_recoverable_portal_failure_rearms()
        assert rearmed and rearmed[0]["rearmed"] is True
        assert rearmed[0]["provider_dispatched"] is False
        assert owner.task_source.get_task(failed.task_cid).status == "retrying"
        assert owner.phase_history(failed.attempt_id)[-1]["phase"] == "failed"
    finally:
        for portal in portals:
            portal.close_event_runtime()
        owner.close()
