"""Retained driver boundaries; disposable records and real advisory locks."""
from __future__ import annotations

import fcntl
import json
import time
from types import SimpleNamespace

import pytest

from scripts.ops.agent_supervisor import spar_retained_capture_driver as driver
from scripts.ops.agent_supervisor import spar_legacy_capture as capture
from test.api.semantic_refactoring.test_spar_legacy_capture import armed as armed_fixture

armed = armed_fixture


def fleet_fixture(tmp_path, monkeypatch, repo, hold):
    state = tmp_path / "fleet"
    (state / "spar").mkdir(parents=True)
    (state / "repairs/spar").mkdir(parents=True)
    for path in (state / "spar/watchdog.lock", state / "repairs/spar/queue.lock"):
        path.touch()
    job = {"board_id": "spar", "status": "queued", "attempts": 1,
           "last_started_at": time.time() - 20, "finished_at": time.time() - 10}
    job_path = state / "repairs/spar/job.json"
    job_path.write_text(json.dumps(job))
    worker_path = state / "repair-worker.json"
    worker_path.write_text(json.dumps({"status": "waiting", "observed_at": time.time()}))
    config = {"schema": "agent-supervisor/fleet-watchdog-config@1", "state_dir": str(state),
        "boards": [{"id": "spar", "cwd": str(repo),
            "hold_files": [str(hold.parent / name) for name in driver.STOP_NAMES],
            "ensure": {"argv": ["systemctl", "--user", "start", capture.UNIT]}}]}
    config_path = tmp_path / "fleet.json"
    config_path.write_text(json.dumps(config))
    unit = {"Id": driver.REPAIR_UNIT, "LoadState": "loaded", "MainPID": "0",
            "ActiveState": "inactive", "SubState": "dead"}
    monkeypatch.setattr(driver, "_job_unit", lambda: dict(unit))
    return SimpleNamespace(state=state, config=config_path, job=job_path, worker=worker_path, unit=unit)


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    return fleet_fixture(tmp_path, monkeypatch, tmp_path / "native", tmp_path / "runtime/HOLD")


def exclude(fleet, tmp_path):
    return driver.FleetCaptureExclusion(fleet.config, tmp_path / "native", tmp_path / "runtime/HOLD")


def test_existing_fleet_locks_are_retained_and_never_recreated(fleet, tmp_path):
    exclusion = exclude(fleet, tmp_path)
    try:
        for path in (fleet.state / "spar/watchdog.lock", fleet.state / "repairs/spar/queue.lock"):
            with path.open("a") as other:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        exclusion.require_current()
    finally:
        exclusion.close()
    with (fleet.state / "spar/watchdog.lock").open("a") as other:
        fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("change", ["running-claim", "unfinished-attempt", "unknown-status", "stale-worker",
                                  "selected-before-claim", "live-job", "unknown-job", "busy-lock", "missing-lock"])
def test_job_or_claim_launch_gap_refuses_exclusion(fleet, tmp_path, change):
    job = json.loads(fleet.job.read_text())
    worker = json.loads(fleet.worker.read_text())
    lock = None
    if change == "running-claim":
        job["status"] = "running"
    elif change == "unfinished-attempt":
        del job["finished_at"]
    elif change == "unknown-status":
        job["status"] = "probably-done"
    elif change == "stale-worker":
        worker["observed_at"] -= 1000
    elif change == "selected-before-claim":
        worker.update(status="running", board_id="spar")
    elif change == "live-job":
        fleet.unit.update(MainPID="345", ActiveState="active", SubState="running")
    elif change == "unknown-job":
        fleet.unit.update(ActiveState="unknown", SubState="unknown")
    elif change == "busy-lock":
        lock = (fleet.state / "repairs/spar/queue.lock").open("a")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    else:
        (fleet.state / "repairs/spar/queue.lock").unlink()
    fleet.job.write_text(json.dumps(job))
    fleet.worker.write_text(json.dumps(worker))
    try:
        with pytest.raises((driver.CaptureDriverDenied, driver.role.SparMergeOwnerError)):
            exclude(fleet, tmp_path)
        if change == "missing-lock":
            assert not (fleet.state / "repairs/spar/queue.lock").exists()
        # Failure after the first acquisition releases only this inspector's FD.
        with (fleet.state / "spar/watchdog.lock").open("a") as other:
            fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if lock:
            lock.close()


@pytest.mark.parametrize("change", ["lock-replacement", "config-change", "job-change"])
def test_retained_fleet_boundary_detects_substitution(fleet, tmp_path, change):
    exclusion = exclude(fleet, tmp_path)
    try:
        if change == "lock-replacement":
            path = fleet.state / "repairs/spar/queue.lock"
            path.rename(path.with_suffix(".retained"))
            path.touch()
        elif change == "config-change":
            fleet.config.write_text(fleet.config.read_text() + "\n")
        else:
            fleet.job.write_text(fleet.job.read_text() + "\n")
        with pytest.raises(driver.CaptureDriverDenied):
            exclusion.require_current()
    finally:
        exclusion.close()


@pytest.mark.parametrize("name", driver.STOP_NAMES)
def test_every_native_stop_marker_blocks_prearm_even_dangling(fleet, tmp_path, name):
    hold = tmp_path / "runtime" / name
    hold.parent.mkdir()
    hold.symlink_to(tmp_path / "missing")
    exclusion = exclude(fleet, tmp_path)
    try:
        with pytest.raises(driver.CaptureDriverDenied, match="already_exists"):
            exclusion.require_no_stops()
    finally:
        exclusion.close()
    assert hold.is_symlink()


@pytest.fixture
def retained(armed, tmp_path, monkeypatch):
    fleet = fleet_fixture(tmp_path, monkeypatch, tmp_path, tmp_path / "HOLD")
    events = []
    class Inhibitor:
        def __init__(self, unit, hold):
            self.before = dict(armed.unit)
            self.files_before = {}
        def apply(self):
            events.append("inhibited")
            assert not (tmp_path / "HOLD").exists()
            armed.unit.update(Restart="no", SendSIGKILL="no", TimeoutStopUSec="infinity",
                RefuseManualStart="yes", Delegate="yes", ExitType="cgroup",
                Conditions=[["ConditionPathExists", False, True, str(tmp_path / "HOLD")]])
        def require_current(self, **kwargs):
            assert "inhibited" in events
    monkeypatch.setattr(driver, "UnitCaptureInhibitor", Inhibitor)
    monkeypatch.setattr(capture, "RetainedNativeLegacySession", lambda **kwargs: armed.session)
    def sentinel():
        assert events == ["inhibited"]
        assert not (tmp_path / "HOLD").exists()
        events.append("sentinel")
        armed.session._sentinel = SimpleNamespace(pid=99999)
        armed.session._sentinel_pidfd = armed.session.pidfd
        (armed.cgroup / "cgroup.procs").write_text(f"{armed.process.pid}\n99999\n")
        return {"fixture_only": True}
    monkeypatch.setattr(armed.session, "retain_workflow_sentinel", sentinel)
    output = tmp_path.parent / (tmp_path.name + "-output")
    value = driver.RetainedCaptureDriver(repository_root=tmp_path, config_path=tmp_path / "config.json",
        fleet_config=fleet.config, operation_root=output,
        expected_source_commit=armed.source["head"], expected_source_tree=armed.source["tree"])
    try:
        yield SimpleNamespace(driver=value, armed=armed, fleet=fleet, events=events, output=output)
    finally:
        armed.session._sentinel = None
        armed.session._sentinel_pidfd = None
        if value.exclusion:
            value.exclusion.close()


def test_inspection_is_read_only_and_does_not_create_operation_or_hold(retained):
    value = retained.driver
    inspected = value.inspect()
    assert inspected["pre_stop_namespaces"]["mnt"]["state"] == "unknown"
    assert not retained.output.exists() and not (retained.armed.tmp / "HOLD").exists()
    assert retained.events == []
    assert value.close_inspection()["stage"] == "inspection-closed"


def test_exported_inspection_cannot_mutate_retained_owner_source_or_review_cid(retained):
    value = retained.driver
    exported = value.inspect()
    original = json.loads(json.dumps(value.inspection))
    exported["source"]["head"] = "0" * 40
    exported["owner_identity"]["generation"] = 999
    exported["inspection_cid"] = "sha256:" + "0" * 64
    assert value.inspection == original
    assert value.session._source["head"] == original["source"]["head"]
    assert value.session.identity["generation"] == original["owner_identity"]["generation"]


def test_native_hold_can_only_follow_inhibition_and_retained_sentinel(retained):
    value = retained.driver
    inspected = value.inspect()
    with pytest.raises(driver.CaptureDriverDenied):
        value.request_closure()
    value.arm(inspected["inspection_cid"])
    assert retained.events == ["inhibited", "sentinel"]
    assert not value.session.hold_path.exists()
    value.request_closure()
    assert value.session.hold_path.read_bytes() == value.hold_bytes
    assert not capture._exited(value.session.pidfd)
    assert value.poll()["closure"] is None
    for action in (value.capture, value.prepare, value.install, value.finish, value.close_inspection):
        with pytest.raises(driver.CaptureDriverDenied):
            action()
    assert not value.session._closed


@pytest.mark.parametrize("change", ["stale-review", "owner-exit", "new-claim", "new-stop"])
def test_changed_admission_cannot_arm_or_write_hold(retained, change):
    value = retained.driver
    inspected = value.inspect()
    cid = inspected["inspection_cid"]
    if change == "stale-review":
        cid = "sha256:" + "0" * 64
    elif change == "owner-exit":
        retained.armed.process.terminate()
        retained.armed.process.wait(timeout=10)
    elif change == "new-claim":
        retained.fleet.job.write_text(retained.fleet.job.read_text() + "\n")
    else:
        (retained.armed.tmp / "OPERATOR_STOP").write_text("other actor")
    with pytest.raises(driver.CaptureDriverDenied):
        value.arm(cid)
    assert retained.events == []
    assert not value.session.hold_path.exists() and not retained.output.exists()


def test_failed_sentinel_attachment_keeps_inhibition_and_never_requests_stop(retained, monkeypatch):
    value = retained.driver
    inspected = value.inspect()
    def failed():
        raise driver.CaptureDriverDenied("sentinel_attach_denied")
    monkeypatch.setattr(value.session, "retain_workflow_sentinel", failed)
    with pytest.raises(driver.CaptureDriverDenied, match="sentinel_attach_denied"):
        value.arm(inspected["inspection_cid"])
    assert value.stage == "arming" and not value.session.hold_path.exists()
    assert retained.events == ["inhibited"] and not value.session._closed
    with pytest.raises(driver.CaptureDriverDenied):
        value.close_inspection()


def test_exclusive_artifacts_never_overwrite_or_follow_symlink(tmp_path):
    target = tmp_path / "original"
    target.write_bytes(b"other operation")
    link = tmp_path / "output"
    link.symlink_to(target)
    with pytest.raises(FileExistsError):
        driver._exclusive_write(link, b"replacement")
    assert target.read_bytes() == b"other operation"


def _close_fixture_driver(retained, monkeypatch):
    value = retained.driver
    inspected = value.inspect()
    monkeypatch.setattr(capture, "RetainedNativeLegacySession", type(value.session))
    value.arm(inspected["inspection_cid"])
    value.request_closure()
    retained.armed.close_native()
    # This fixture uses no real sentinel; the separate host suite does.
    value.session._sentinel = value.session._sentinel_pidfd = None
    value.session.hold_path.write_bytes(value.hold_bytes)
    retained.closure_report = value.poll()
    assert retained.closure_report["stage"] == "closed"
    return value


def _tree_bytes(path):
    return {name: (path / name).read_bytes() for name in driver.role.file_inventory(path)}


def test_failed_capture_retries_same_native_handles_into_fresh_output(retained, monkeypatch):
    value = _close_fixture_driver(retained, monkeypatch)
    original = capture.producer.produce_offline_import_plan
    calls = []

    def fail_once(**kwargs):
        result = original(**kwargs)
        calls.append(kwargs)
        if len(calls) == 1:
            raise ModuleNotFoundError("private diagnostic must not be exported", name="duckdb")
        return result

    monkeypatch.setattr(capture.producer, "produce_offline_import_plan", fail_once)
    canonical = _tree_bytes(value.session.queue_root)
    session, pidfd, cgroup = value.session, value.session.pidfd, value.session.cgroup_fd
    with pytest.raises(ModuleNotFoundError):
        value.capture()
    assert value.stage == "capture-failed"
    descriptors = tuple(item[1] for item in session._retained_queue_locks)
    raw = _tree_bytes(value.output / "raw-capture")
    inspection = _tree_bytes(value.output / "inspection-copy")
    diagnostic = value.status()["diagnostics"][0]["diagnostic"]
    assert diagnostic["missing_module"] == "duckdb" and diagnostic["traceback"]
    assert "private diagnostic" not in json.dumps(diagnostic)
    assert value.retry_capture()["stage"] == "captured"
    assert value.session is session and session.pidfd == pidfd and session.cgroup_fd == cgroup
    assert tuple(item[1] for item in session._retained_queue_locks) == descriptors
    assert value.captured is session._capture
    assert value.captured.path == value.output / "raw-capture-002"
    assert _tree_bytes(value.output / "raw-capture") == raw == canonical
    assert _tree_bytes(value.output / "inspection-copy") == inspection
    assert _tree_bytes(session.queue_root) == canonical
    assert value.status()["retry_available"] is None
    with pytest.raises(driver.CaptureDriverDenied):
        value.retry_capture()


@pytest.mark.parametrize("change", ["released-lock", "changed-input"])
def test_capture_retry_revalidates_original_custody_and_input(retained, monkeypatch, change):
    value = _close_fixture_driver(retained, monkeypatch)
    original = capture.producer.produce_offline_import_plan

    def fail(**kwargs):
        raise OSError("injected pre-install failure")

    monkeypatch.setattr(capture.producer, "produce_offline_import_plan", fail)
    with pytest.raises(OSError):
        value.capture()
    monkeypatch.setattr(capture.producer, "produce_offline_import_plan", original)
    if change == "released-lock":
        descriptor = value.session._retained_queue_locks[0][1]
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    else:
        (value.session.queue_root / "private/callback-signing-material").write_bytes(b"changed")
    with pytest.raises(driver.role.SparMergeOwnerError):
        value.retry_capture()
    assert value.captured is None and value.installed is None
    assert not (value.output / "raw-capture-002").exists()
    assert not value.session._closed


def test_failed_preparation_preserves_capture_and_retries_new_clone(retained, monkeypatch):
    value = _close_fixture_driver(retained, monkeypatch)
    value.capture()
    captured = value.captured
    raw = _tree_bytes(captured.path)
    original = driver.role.prepare_offline_clone
    calls = []

    def fail_once(**kwargs):
        result = original(**kwargs)
        calls.append(kwargs)
        if len(calls) == 1:
            raise OSError("injected after disposable clone migration")
        return result

    monkeypatch.setattr(driver.role, "prepare_offline_clone", fail_once)
    with pytest.raises(OSError):
        value.prepare()
    assert value.stage == "prepare-failed" and value.prepared is None
    first = _tree_bytes(value.output / "prepared-clone")
    assert value.retry_prepare()["stage"] == "prepared"
    assert value.captured is captured and value.captured is value.session._capture
    assert value.prepared.database_path.parent == value.output / "prepared-clone-002"
    assert _tree_bytes(value.output / "prepared-clone") == first
    assert _tree_bytes(captured.path) == raw
    assert value.installed is None and not value.session._closed
    with pytest.raises(driver.CaptureDriverDenied):
        value.retry_prepare()


def test_runtime_change_blocks_arm_before_output_or_inhibition(retained, monkeypatch):
    value = retained.driver
    inspected = value.inspect()

    def changed():
        raise driver.CaptureRuntimeDenied("runtime_dependency_hash_changed")

    monkeypatch.setattr(value.runtime, "require_current", changed)
    with pytest.raises(driver.CaptureRuntimeDenied):
        value.arm(inspected["inspection_cid"])
    assert retained.events == [] and not retained.output.exists()
    assert not value.session.hold_path.exists()


def test_closed_poll_export_cannot_rewrite_retained_retry_gate(retained, monkeypatch):
    value = _close_fixture_driver(retained, monkeypatch)
    original = json.loads(json.dumps(value._closure))
    retained.closure_report["closure"]["native_actors_closed"] = False
    retained.closure_report["closure"]["unit"]["MainPID"] = "99999"
    assert value._closure == original
