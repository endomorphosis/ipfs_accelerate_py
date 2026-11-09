"""Real private canonical-native integration, without injected host authority.

The productive test invocation has its own outer host grant. These fixtures
exercise real file-backed native schedulers and ledgers, never the shared host
state. The launcher must bind the canonical datasets package before loading
accelerate's conftest; a different source tree fails rather than retargeting an
already loaded package. Isolated process-group accounting is cooperative and
does not claim containment of arbitrary descendants creating new sessions.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_daemon_resources as native_disk
from ipfs_datasets_py.optimizers.logic_theorem_optimizer import existing_resource_scheduler as existing
from ipfs_datasets_py.optimizers.logic_theorem_optimizer import resource_scheduler as native
from ipfs_datasets_py.optimizers.logic_theorem_optimizer import parented_phase_resources as parented
from ipfs_accelerate_py.agent_supervisor.runtime import repository_pipeline_resources as pipeline
from ipfs_accelerate_py.agent_supervisor.runtime import repository_resource_bridge as bridge
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import ResourcePolicy, ResourceScheduler


CANONICAL_DATASETS = Path(native.__file__).resolve().parents[3]
CANONICAL_ACCELERATE = Path(bridge.__file__).resolve().parents[3]


def _json(path):
    return json.loads(Path(path).read_bytes())


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _wait(predicate, message, seconds=5):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        result = predicate()
        if result:
            return result
        time.sleep(.01)
    raise AssertionError(message)


def _stop_group(process):
    """Reap only a known direct child's original isolated process group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()


@pytest.fixture
def private_native(tmp_path):
    expected = {
        native_disk: CANONICAL_DATASETS / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_daemon_resources.py",
        existing: CANONICAL_DATASETS / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/existing_resource_scheduler.py",
        native: CANONICAL_DATASETS / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py",
        parented: CANONICAL_DATASETS / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/parented_phase_resources.py",
        pipeline: CANONICAL_ACCELERATE / "ipfs_accelerate_py/agent_supervisor/runtime/repository_pipeline_resources.py",
        bridge: CANONICAL_ACCELERATE / "ipfs_accelerate_py/agent_supervisor/runtime/repository_resource_bridge.py",
    }
    for module, path in expected.items():
        assert Path(module.__file__).resolve() == path, "different producer tree was loaded"
    for name, root in (("IPFS_DATASETS_EXPECTED_SOURCE_ROOT", CANONICAL_DATASETS),
                       ("IPFS_ACCELERATE_EXPECTED_SOURCE_ROOT", CANONICAL_ACCELERATE)):
        if os.environ.get(name):
            assert Path(os.environ[name]).resolve() == root, "explicit producer tree differs"
    root = tmp_path / "outputs"
    root.mkdir(mode=0o700)
    scheduler = native.GlobalResourceScheduler(native.ResourceSchedulerConfig(
        total_cpu_slots=8, total_memory_mb=16384, total_child_process_slots=8,
        total_gpu_memory_mb=None, total_unified_memory_mb=None,
        lane_reservations={"validation": {"cpu_slots": 1, "memory_mb": 0}},
        state_path=tmp_path / "existing-native.json", proof_safety_enabled=True,
        proof_memory_stall_percent=100.0, proof_cpu_stall_percent=100.0,
        proof_io_stall_percent=100.0, max_waiting_requests=32,
        auto_renew_leases=False,
    ))
    policy = deepcopy(_json(scheduler.state_path)["config"])
    policy_sha256 = existing.policy_sha256(policy)
    ledger = tmp_path / "existing-disk.json"
    foreign = {"reservation_id": "unrelated-retained", "status": "retained",
        "owner_pid": os.getpid(), "storage_bytes": 1234, "external_charges": {},
        "reason": "native integration must preserve unrelated retained evidence"}
    _write_json(ledger, {"schema": native_disk.SCHEMA,
        "roots": [native_disk._root_identity(root)], "limit_bytes": 145_000_000_000,
        "reservations": {"unrelated-retained": foreign}})
    supervisor = ResourceScheduler(ResourcePolicy(max_lanes=8))
    yield dict(root=root, ledger=ledger, scheduler=scheduler, policy=policy,
        policy_sha256=policy_sha256, foreign=foreign, supervisor=supervisor,
        producer_sources={str(path): _sha(path) for path in expected.values()})
    assert not supervisor.active_leases
    assert scheduler.snapshot()["allocated"]["cpu_slots"] == 0
    assert _json(scheduler.state_path)["config"] == policy
    saved = _json(ledger)
    assert saved["limit_bytes"] == 145_000_000_000
    assert saved["reservations"]["unrelated-retained"] == foreign


def _attempt(control, name):
    result = control["root"] / name
    result.mkdir(mode=0o700)
    return result


@contextmanager
def _hierarchy(control, lane=native.ResourceLane.VALIDATION):
    selected = existing.open_existing_scheduler(
        control["scheduler"].state_path, control["policy_sha256"])
    with selected.acquire(native.ResourceLane.ORCHESTRATION, cpu_slots=4, memory_mb=8192,
                          child_process_slots=4, timeout=0, request_id="private-native-root") as root:
        with root.acquire_child(lane=lane, cpu_slots=2, memory_mb=4096,
                                child_process_slots=2, timeout=0,
                                request_id="private-native-phase") as phase:
            yield root, phase


def _disk_owner(control, phase, attempt, *, storage_bytes=1_000_000, child_process_slots=1):
    return parented.ParentedPhaseResourceReservation(
        control["ledger"], roots=[control["root"]], attempt_directory=attempt,
        parent_lease=phase, storage_bytes=storage_bytes, memory_mb=1024,
        cpu_slots=1, child_process_slots=child_process_slots,
        scheduler_state_path=control["scheduler"].state_path,
        expected_scheduler_policy_sha256=control["policy_sha256"], timeout_seconds=10)


def _blocked_child(attempt, mode="finish"):
    source = (
        "import json,os,pathlib,sys,time\n"
        "assert sys.stdin.readline().strip()=='attached'\n"
        "path=pathlib.Path(sys.argv[1]);mode=sys.argv[2]\n"
        "(path/'actual-result.json').write_text(json.dumps({'sum':sum((1,2,3)),'pid':os.getpid()}))\n"
        "if mode=='descendant':\n"
        " child=os.fork()\n"
        " if child:\n"
        "  (path/'descendant.pid').write_text(str(child));sys.exit(0)\n"
        " time.sleep(20);os._exit(0)\n"
        "if mode=='wait': time.sleep(20)\n")
    return subprocess.Popen([sys.executable, "-I", "-B", "-c", source, str(attempt), mode],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True)


def _unblock(process):
    process.stdin.write("attached\n")
    process.stdin.flush()


@pytest.mark.parametrize("lane", [native.ResourceLane.VALIDATION, native.ResourceLane.TRAINER,
                                  native.ResourceLane.PERSISTENCE, native.ResourceLane.HAMMER_LEAN])
def test_real_child_lineage_lane_and_completed_process_preserve_one_root(private_native, lane):
    c = private_native
    attempt = _attempt(c, "actual-" + lane.value)
    with _hierarchy(c, lane) as (root, phase):
        with _disk_owner(c, phase, attempt) as owned:
            actual = owned.native_lease
            assert actual.parent_lease_id == phase.lease_id
            assert actual.lane == phase.lane == lane.value
            records = {row["lease_id"]: row for row in c["scheduler"].active_leases()}
            assert records[actual.lease_id]["request_id"] == "daemon:" + owned.reservation_id
            assert c["scheduler"].snapshot()["active_root_lease_count"] == 1
            assert c["scheduler"].snapshot()["allocated"]["cpu_slots"] == 4
            with actual.acquire_child(lane=lane, cpu_slots=1, memory_mb=128,
                                      child_process_slots=1, timeout=0) as consumer:
                assert consumer.parent_lease_id == actual.lease_id
                with pytest.raises(native_disk.DaemonResourceError, match="consumer|descendant|native"):
                    owned.finalize(attempt, artifacts_durable=True)
                assert c["scheduler"].snapshot()["allocated"]["cpu_slots"] == 4
                assert _json(c["ledger"])["reservations"][owned.reservation_id]["status"] == "active"
            child = _blocked_child(attempt)
            try:
                owned.check_usage(attempt, child_pid=child.pid)
                recorded = owned.to_dict()["record"]["child"]
                assert recorded == {"pid": child.pid, "birth": native_disk._process(child.pid)["birth"]}
                _unblock(child)
                child.wait(timeout=5)
                assert child.returncode == 0, child.stderr.read()
            finally:
                _stop_group(child)
            assert _json(attempt / "actual-result.json")["sum"] == 6
            released = owned.finalize(attempt, artifacts_durable=True)
            assert released["status"] == "released" and released["resource_lease"]["released"]
            assert released["record"]["finalization_profile"]["global_inventory_count"] == 1
            assert "lease_key" not in json.dumps(released)
            remaining = {row["lease_id"] for row in c["scheduler"].active_leases()}
            assert root.lease_id in remaining and phase.lease_id in remaining and actual.lease_id not in remaining
            assert c["scheduler"].snapshot()["allocated"]["cpu_slots"] == 4


@pytest.mark.parametrize("revoke", ["cancel", "release_request"])
def test_real_revocation_retains_consumers_and_cleanup_releases_only_owned_child(private_native, revoke):
    c = private_native
    attempt = _attempt(c, "revoked-" + revoke)
    with _hierarchy(c) as (root, phase):
        owned = _disk_owner(c, phase, attempt)
        owned.__enter__()
        owned_child_id = owned.native_lease.lease_id
        consumer = owned.native_lease.acquire_child(cpu_slots=1, memory_mb=128,
                                                  child_process_slots=1, timeout=0)
        try:
            if revoke == "cancel": root.cancel()
            else: root.release()
            with pytest.raises(native_disk.DaemonResourceError, match="parent|active|cancel|admission|custody"):
                _ = owned.native_lease
            assert consumer.cancelled
            owned.__exit__(RuntimeError, RuntimeError("actual parent revoked"), None)
            row = _json(c["ledger"])["reservations"][owned.reservation_id]
            assert row["status"] == "retained" and row["storage_bytes"] == 1_000_000
            active = {r["lease_id"]: r for r in c["scheduler"].active_leases()}
            assert {root.lease_id, phase.lease_id, owned_child_id, consumer.lease_id} <= set(active)
            assert c["scheduler"].snapshot()["allocated"]["cpu_slots"] == 4
            if revoke == "release_request":
                assert root.released is True and active[root.lease_id]["release_requested"] is True
            with pytest.raises(native_disk.DaemonResourceError, match="consumer|descendant|native"):
                owned.release(artifacts_durable=True)
            consumer.release()
            assert c["scheduler"].snapshot()["allocated"]["cpu_slots"] == 4
            cleaned = owned.release(artifacts_durable=True)
            assert cleaned["status"] == "released"
            active = {r["lease_id"] for r in c["scheduler"].active_leases()}
            assert phase.lease_id in active and root.lease_id in active and owned_child_id not in active
        finally:
            consumer.release()
            if owned.to_dict()["status"] != "released":
                owned.release(artifacts_durable=True)


def test_revocation_during_attachment_keeps_actual_pid_for_stopped_cleanup(private_native):
    c = private_native
    attempt = _attempt(c, "attach-revoked")
    with _hierarchy(c) as (root, phase):
        with _disk_owner(c, phase, attempt) as owned:
            child = _blocked_child(attempt, "wait")
            try:
                actual_birth = native_disk._process(child.pid)["birth"]
                root.cancel()
                with pytest.raises(native_disk.DaemonResourceError):
                    owned.check_usage(attempt, child_pid=child.pid)
                assert owned.to_dict()["record"]["child"] == {"pid": child.pid, "birth": actual_birth}
                assert not (attempt / "actual-result.json").exists()
            finally:
                _stop_group(child)
            assert owned.release(artifacts_durable=True)["status"] == "released"


def test_live_same_group_descendant_prevents_finalize_and_stopped_cleanup(private_native):
    c = private_native
    attempt = _attempt(c, "cooperative-descendant")
    with _hierarchy(c) as (_, phase):
        with _disk_owner(c, phase, attempt, child_process_slots=2) as owned:
            child = _blocked_child(attempt, "descendant")
            try:
                owned.check_usage(attempt, child_pid=child.pid)
                _unblock(child)
                child.wait(timeout=5)
                assert child.returncode == 0
                recorded = owned.to_dict()["record"]["child"]
                assert _wait(lambda: native_disk._group_usage(recorded)["live_processes"], "descendant not observed")
                with pytest.raises(native_disk.DaemonResourceError, match="alive"):
                    owned.finalize(attempt, artifacts_durable=True)
                with pytest.raises(native_disk.DaemonResourceError, match="alive"):
                    owned.release(artifacts_durable=True)
            finally:
                _stop_group(child)
            assert owned.release(artifacts_durable=True)["status"] == "released"


@pytest.mark.parametrize("kind", ["many_small_files", "sqlite_sidecars_and_exports"])
def test_aggregate_storage_includes_real_files_sidecars_and_export_overlap(private_native, kind):
    c = private_native
    attempt = _attempt(c, kind)
    allowance = 4096 if kind == "many_small_files" else 250_000
    connection = None
    with _hierarchy(c) as (_, phase):
        try:
            with _disk_owner(c, phase, attempt, storage_bytes=allowance) as owned:
                key = owned.reservation_id
                if kind == "many_small_files":
                    for index in range(50): (attempt / str(index)).write_bytes(b"x" * 100)
                else:
                    database = attempt / "private.sqlite"
                    connection = sqlite3.connect(database)
                    assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
                    connection.execute("PRAGMA wal_autocheckpoint=0")
                    connection.execute("CREATE TABLE evidence(payload BLOB NOT NULL)")
                    connection.execute("INSERT INTO evidence VALUES (?)", (b"e" * 65536,))
                    connection.commit()
                    assert Path(str(database) + "-wal").is_file() and Path(str(database) + "-shm").is_file()
                    owned.check_usage(attempt)
                    export = attempt / "snapshot.sqlite"
                    with sqlite3.connect(export) as destination: connection.backup(destination)
                    (attempt / "atomic-export.tmp").write_bytes(export.read_bytes())
                files = [p for p in attempt.iterdir() if p.is_file()]
                assert all(p.stat().st_size < allowance for p in files)
                assert sum(p.stat().st_size for p in files) > allowance
                with pytest.raises(native_disk.DaemonResourceError, match="storage"):
                    owned.check_usage(attempt)
            row = _json(c["ledger"])["reservations"][key]
            assert row["status"] == "retained" and row["storage_bytes"] == allowance
        finally:
            if connection is not None: connection.close()


@pytest.mark.parametrize("fault", ["cpu_budget", "parent_binding", "child_birth"])
def test_exact_owned_ledger_changes_cannot_be_overwritten_as_new_authority(private_native, fault):
    c = private_native
    attempt = _attempt(c, "changed-" + fault)
    with _hierarchy(c) as (_, phase):
        owned = _disk_owner(c, phase, attempt)
        owned.__enter__()
        child = _blocked_child(attempt)
        try:
            owned.check_usage(attempt, child_pid=child.pid)
            _unblock(child); child.wait(timeout=5)
            assert child.returncode == 0
            state = _json(c["ledger"]); row = state["reservations"][owned.reservation_id]
            if fault == "cpu_budget": row["cpu_slots"] += 1
            elif fault == "parent_binding": row["parent_lease_id"] = "not-the-original-parent"
            else: row["child"]["birth"] = "0"
            _write_json(c["ledger"], state)
            tampered = c["ledger"].read_bytes()
            with pytest.raises(native_disk.DaemonResourceError, match="record|own|changed|differ|custody"):
                owned.check_usage(attempt, full=True)
        finally:
            _stop_group(child)
            owned.__exit__(RuntimeError, RuntimeError("private test changed own durable row"), None)
        assert c["ledger"].read_bytes() == tampered
        assert owned.to_dict()["cleanup_error"] is not None
        assert _json(c["ledger"])["reservations"][owned.reservation_id]["status"] != "released"


@pytest.mark.parametrize("fault", ["wrong_policy", "missing_state", "foreign_actual_parent"])
def test_unbound_or_foreign_real_parent_refused_without_disk_write(private_native, fault):
    c = private_native
    attempt = _attempt(c, "refused-" + fault)
    with _hierarchy(c) as (_, phase):
        other_root = other_phase = None
        if fault == "foreign_actual_parent":
            config = native.ResourceSchedulerConfig(state_path=attempt.parent.parent / "foreign-state.json", **c["policy"])
            other = native.GlobalResourceScheduler(config)
            other_root = other.acquire(native.ResourceLane.ORCHESTRATION,
                cpu_slots=1, memory_mb=128, child_process_slots=1, timeout=0)
            other_phase = other_root.acquire_child(lane=native.ResourceLane.VALIDATION,
                cpu_slots=1, memory_mb=128, child_process_slots=1, timeout=0)
            selected_parent = other_phase
        else: selected_parent = phase
        raw = c["ledger"].read_bytes()
        try:
            kwargs = dict(roots=[c["root"]], attempt_directory=attempt, parent_lease=selected_parent,
                storage_bytes=1000, memory_mb=128, cpu_slots=1, child_process_slots=1,
                scheduler_state_path=(attempt.parent.parent / "missing-native.json") if fault == "missing_state" else c["scheduler"].state_path,
                expected_scheduler_policy_sha256="0" * 64 if fault == "wrong_policy" else c["policy_sha256"], timeout_seconds=10)
            with pytest.raises((native_disk.DaemonResourceError, existing.ExistingResourceSchedulerError, FileNotFoundError)):
                with parented.ParentedPhaseResourceReservation(c["ledger"], **kwargs):
                    pytest.fail("unbound parent must not admit productive work")
            assert c["ledger"].read_bytes() == raw and not tuple(attempt.iterdir())
        finally:
            if other_phase is not None: other_phase.release()
            if other_root is not None: other_root.release()


@contextmanager
def _pipeline(control, *, wall_time_ms=15000):
    limits = bridge.RepositoryResourceBudget(cpu_slots=3, memory_mb=4096, process_slots=3,
        disk_bytes=4 * bridge.MIB, wall_time_ms=wall_time_ms)
    policy = pipeline.PipelineResourcePolicy(protected_cpu_slots=1, protected_memory_mb=1024,
        protected_process_slots=1, protected_disk_bytes=bridge.MIB)
    with pipeline.RepositoryPipelineResources(control["supervisor"]).reserve(
            repository_id="real-native-private-integration", workspace=control["root"], budget=limits,
            policy=policy, ledger_path=control["ledger"], roots=[control["root"]],
            scheduler_state_path=control["scheduler"].state_path,
            expected_scheduler_policy_sha256=control["policy_sha256"]) as parent:
        yield parent


def _validation_demand():
    return bridge.RepositoryPhaseDemand("validation", cpu_slots=1, memory_mb=1024,
        process_slots=1, disk_bytes=500_000)


def test_strict_pipeline_runs_actual_work_and_preserves_unrelated_native_root(private_native):
    c = private_native
    foreign = c["scheduler"].acquire(native.ResourceLane.CODEX, cpu_slots=1, memory_mb=128,
        child_process_slots=1, timeout=0, request_id="unrelated-native-owner")
    try:
        foreign_record = deepcopy(_json(c["scheduler"].state_path)["leases"][foreign.lease_id])
        with _pipeline(c) as parent:
            with parent.phase(_validation_demand(), payload=b"forty-two", attempt_directory=_attempt(c, "pipeline-work")) as phase:
                owned = phase.daemon.native_lease
                assert owned.parent_lease_id == phase.native.native.lease_id
                assert phase.native.native.parent_lease_id == parent.parent.native.lease_id
                with phase.native_options()["parent_lease"].acquire_child(cpu_slots=1, memory_mb=128,
                        child_process_slots=1, timeout=0) as consumer:
                    assert consumer.parent_lease_id == owned.lease_id
                    assert c["scheduler"].snapshot()["active_root_lease_count"] == 2
                    assert c["scheduler"].snapshot()["allocated"]["cpu_slots"] == 4
                result = phase.run([str(Path(sys.executable).resolve()), "-I", "-B", "-c",
                    "import sys;assert 6*7==42;print(sys.stdin.buffer.read().decode())"], timeout_seconds=3)
                assert result.returncode == 0 and result.stdout.strip() == "forty-two"
                phase.finalize(artifacts_durable=True)
            receipt = parent.receipt()
            assert receipt["events"][0]["disk"]["status"] == "released"
            assert "lease_key" not in json.dumps(receipt)
        assert _json(c["scheduler"].state_path)["leases"][foreign.lease_id] == foreign_record
    finally:
        foreign.release()


def test_actual_parallel_ordinary_work_leaves_native_protected_validation_capacity(private_native):
    c = private_native
    entered, finish = threading.Event(), threading.Event()
    errors = []
    training_attempt = _attempt(c, "ordinary-work")
    with _pipeline(c) as parent:
        def ordinary():
            try:
                demand = bridge.RepositoryPhaseDemand("training", cpu_slots=2, memory_mb=2048,
                    process_slots=2, disk_bytes=500_000)
                with parent.phase(demand, attempt_directory=training_attempt) as phase:
                    (training_attempt / "actual-work.txt").write_text("private resource integration; no model training")
                    entered.set()
                    assert finish.wait(5)
                    phase.finalize(artifacts_durable=True)
            except BaseException as error: errors.append(error)
        worker = threading.Thread(target=ordinary)
        worker.start()
        try:
            assert entered.wait(5), errors
            with parent.phase(_validation_demand(), attempt_directory=_attempt(c, "protected-work")) as phase:
                assert parent.receipt()["active_phases"] == 2
                assert c["scheduler"].snapshot()["active_root_lease_count"] == 1
                assert c["scheduler"].snapshot()["allocated"]["cpu_slots"] == 3
                result = phase.run([str(Path(sys.executable).resolve()), "-I", "-B", "-c", "assert 6*7==42;print(42)"], timeout_seconds=3)
                assert result.returncode == 0 and result.stdout.strip() == "42"
                phase.finalize(artifacts_durable=True)
        finally:
            finish.set(); worker.join(5)
        assert not worker.is_alive() and not errors


def test_actual_pipeline_cancellation_reaps_child_and_retained_cleanup_adds_no_grant(private_native):
    c = private_native
    attempt = _attempt(c, "pipeline-cancel")
    marker = attempt / "actual-child-ready"
    errors, results = [], []
    parent = None
    with pytest.raises(native.LeaseCancelledError):
        with _pipeline(c) as parent:
            def run():
                try:
                    with parent.phase(_validation_demand(), attempt_directory=attempt) as phase:
                        try:
                            phase.run([str(Path(sys.executable).resolve()), "-I", "-B", "-c",
                                "import pathlib,sys,time;pathlib.Path(sys.argv[1]).write_text('actual');time.sleep(20)",
                                str(marker)], timeout_seconds=3)
                        finally:
                            results.append(phase.last_process_result)
                except BaseException as error: errors.append(error)
            worker = threading.Thread(target=run)
            worker.start()
            assert _wait(marker.exists, "actual child never became ready")
            parent.parent.cancel()
            worker.join(5)
            assert not worker.is_alive() and errors
    assert parent is not None
    retained = parent.receipt()["retained_disk_reservations"]
    assert retained and results and results[0].cancelled and results[0].process_tree_terminated
    before = _json(c["scheduler"].state_path)["metrics"]["acquisitions_total"]
    for key in retained:
        row = _json(c["ledger"])["reservations"][key]
        assert row["status"] == "retained" and not native_disk._group_usage(row["child"])["live_processes"]
        parent.reap_retained(key, artifacts_durable=True)
        assert _json(c["ledger"])["reservations"][key]["status"] == "released"
    assert _json(c["scheduler"].state_path)["metrics"]["acquisitions_total"] == before
    assert marker.read_text() == "actual"


def test_native_renewal_cannot_extend_the_pipeline_productive_deadline(private_native):
    c = private_native
    attempt = _attempt(c, "fixed-deadline")
    with pytest.raises(native.LeaseTimeoutError):
        with _pipeline(c, wall_time_ms=1000) as parent:
            with parent.phase(_validation_demand(), attempt_directory=attempt) as phase:
                assert phase.daemon.native_lease.renew()
                time.sleep(1.05)
                parent.parent.remaining()
    assert parent.receipt()["retained_disk_reservations"]
    for key in tuple(parent.receipt()["retained_disk_reservations"]):
        parent.reap_retained(key, artifacts_durable=True)

