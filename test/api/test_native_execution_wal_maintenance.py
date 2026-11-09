"""Real crash-WAL fixtures for physical maintenance; no original-store authority.

Each case creates a new private database and launches only the pinned disposable
producer. The source-unit owner must explicitly admit that finite child; no
provider, Docker, supervisor, original database or native grant is used here.
Positive tests use the genuine sealed policy and installed DuckDB. Two clearly
named fault tests wrap genuine handles only to inject native close failures.
"""
from __future__ import annotations

from dataclasses import replace
import fcntl
import importlib
import importlib.util
import os
from pathlib import Path
import stat
import time

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state
from ipfs_accelerate_py.agent_supervisor.todo_daemon import native_execution_wal_maintenance as maintenance
from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_execution_status_owner import (
    NativeExecutionStatusBinding, NativeExecutionStatusOwner,
)
from test.api import native_execution_crash_wal_fixture as fixture_producer


@pytest.fixture
def crash_store(tmp_path, request):
    producer = Path(importlib.util.find_spec(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
    ).origin).absolute()
    created = fixture_producer.create_crash_fixture(tmp_path / "fresh-lane", producer, getattr(request, "param", "baseline"))
    path = created["path"]
    writer = path.with_name("." + path.name + ".writer.lock")
    policy = path.with_name("." + path.name + ".lock")
    reader = Path(importlib.import_module(NativeExecutionStatusOwner.__module__).__file__).absolute()
    sources = (producer, reader, Path(duckdb_state.__file__).absolute(), Path(fixture_producer.__file__).absolute())
    metadata = created["result"]["metadata"]
    known = {key: value for key, value in metadata.items() if key not in {
        "process_instance_id", "state_schema_revision", "control_schema_profile_id", "control_schema_fingerprint",
    }}
    selected = NativeExecutionStatusBinding(
        backend="duckdb", execution_path=str(path),
        execution_file_identity=fixture_producer.identity(path),
        writer_lock_identity=fixture_producer.identity(writer), expected_metadata=known,
        source_bindings={str(source): fixture_producer.digest(source) for source in sources},
        control_store_id="fixture-primary-identity", control_schema_version=5,
        task_cid="task:crash-fixture", control_revision=12, expected_control_status="in_progress",
        attempt_id="attempt:crash-fixture2", attempt_number=2, claim_id="claim:crash-fixture2",
        lease_id="lease:crash-fixture2", owner_session_id="embedded-store:crash-fixture",
        fencing_token=2, fence_epoch=2,
    )
    backups = tmp_path / "retained-maintenance-backups"
    backups.mkdir(mode=0o700)  # A sibling of the execution lane, never inside it.
    bound = maintenance.NativeExecutionWalMaintenanceBinding.from_dict({
        "schema": maintenance.BINDING_SCHEMA, "status_binding": selected.to_dict(),
        "wal_file_identity": fixture_producer.identity(created["wal"]),
        "policy_lock_identity": fixture_producer.identity(policy), "backup_directory": str(backups),
        "maintenance_source_sha256": fixture_producer.digest(Path(maintenance.__file__).absolute()),
    })
    return {**created, "writer": writer, "policy": policy, "backups": backups,
            "status_binding": selected, "binding": bound, "fixture_guards": []}


class FixtureCustodyGuard:
    """Explicit nonadmitting fixture callback; no native grant or ACK exists."""

    def __init__(self):
        self.calls = []
        self.fail_force_when = None

    def __call__(self, *, force_ack=False, check_only=False):
        self.calls.append((force_ack, check_only))
        if force_ack and self.fail_force_when is not None and self.fail_force_when():
            raise RuntimeError("explicit fixture custody loss")


def owner(store, binding=None, custody_guard=None):
    guard = custody_guard or FixtureCustodyGuard()
    store["fixture_guards"].append(guard)
    return maintenance.NativeExecutionWalMaintenanceOwner(
        binding or store["binding"], deadline_monotonic=time.monotonic() + 60,
        custody_guard=guard,
    )


def original_state(store):
    return {key: (fixture_producer.identity(store[key]), fixture_producer.digest(store[key]))
            for key in ("path", "wal", "writer", "policy")}


def assert_released(instance, store):
    assert instance._connection is None and instance._fds == {} and instance._backup_fd is None
    assert instance.failure_state["connection_close_verified"] is True
    assert instance.failure_state["original_locks_released"] is True
    for key in ("writer", "policy"):
        descriptor = os.open(store[key], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)


def test_real_committed_crash_wal_replays_without_logical_or_authority_changes(crash_store):
    before = original_state(crash_store)
    assert crash_store["child_reaped"] is True
    assert crash_store["owned_process_group"] == os.getpgrp()
    assert crash_store["result"]["committed_before_os_exit"] is True
    assert before["wal"][0]["size"] > 0
    instance = owner(crash_store)
    report = instance.run()
    assert_released(instance, crash_store)
    assert sum(force is True for force, _check in crash_store["fixture_guards"][-1].calls) >= 2
    assert report["physical_maintenance_only"] is report["checkpoint_executed"] is True
    assert report["logical_rows_preserved"] is report["all_metadata_preserved"] is True
    assert report["connection_close_verified"] is report["original_locks_released"] is True
    assert report["observed_process_instance_id"] == "process:fixture-wal-committed"
    assert report["wal_after"] == "absent" and not crash_store["wal"].exists()
    assert report["table_row_counts"] == {
        "daemon_execution_metadata": 13, "database_task_attempts": 1, "attempt_phases": 2,
        "provider_invocations": 1, "effect_claims": 0, "daemon_execution_events": 1,
    }
    assert report["provider_effect_state"] == "unknown" and report["unknown_not_settled"] is True
    for key in ("completion_authority", "automatic_retry_admitted", "semantic_admission",
                "coordination_authority_observed", "control_join_verified_by_component",
                "original_process_impersonated", "historical_inode_attestation"):
        assert report[key] is False
    assert report["native_store_uuid"] is None
    for original, name in (("path", "original.db"), ("wal", "original.db.wal")):
        backup = crash_store["backups"] / name
        assert fixture_producer.digest(backup) == before[original][1]
        assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    for key in ("writer", "policy"):
        assert (fixture_producer.identity(crash_store[key]), fixture_producer.digest(crash_store[key])) == before[key]
    # Checkpoint changes the current physical identity; explicitly bind the new
    # identity. Reusing the pre-maintenance selector must never be automatic.
    refreshed = replace(crash_store["status_binding"], execution_file_identity=report["execution_file_identity"])
    with NativeExecutionStatusOwner(refreshed, deadline_monotonic=time.monotonic() + 30) as reader:
        observation = reader.observe_attempt()
        assert observation["attempt"]["status"] == "running"
        assert observation["attempt"]["attempt_number"] == 2
        assert observation["attempt"]["body"]["feature_score"] == 0.75
        assert observation["provider_records"][0]["result"]["callback_state"] == "started_outcome_unknown"
        assert observation["observed_process_instance_id"] == "process:fixture-wal-committed"
        assert observation["preknown_original_process_instance_id"] is False
        assert observation["provider_effect_state"] == "unknown"
        assert observation["completion_authority"] is observation["automatic_retry_admitted"] is False
    assert_released(instance, crash_store)


@pytest.mark.parametrize("lock_key", ["writer", "policy"])
def test_existing_original_lock_contention_refuses_before_any_physical_write(crash_store, lock_key):
    before = original_state(crash_store)
    descriptor = os.open(crash_store[lock_key], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    instance = owner(crash_store)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        with pytest.raises(maintenance.NativeExecutionWalMaintenanceError, match="maintenance_original_lock_contended"):
            instance.run()
        assert instance.failure_state["physical_mutation_possible"] is False
        assert list(crash_store["backups"].iterdir()) == []
        assert original_state(crash_store) == before
    finally:
        os.close(descriptor)
        instance.close()
    assert_released(instance, crash_store)


def test_current_wal_identity_drift_refuses_without_touching_originals(crash_store):
    before = original_state(crash_store)
    drift = dict(crash_store["binding"].wal_file_identity)
    drift["mtime_ns"] += 1
    instance = owner(crash_store, replace(crash_store["binding"], wal_file_identity=drift))
    with pytest.raises(maintenance.NativeExecutionWalMaintenanceError, match="maintenance_original_file_drift"):
        instance.run()
    assert instance.failure_state["physical_mutation_possible"] is False
    assert original_state(crash_store) == before and list(crash_store["backups"].iterdir()) == []
    assert_released(instance, crash_store)


def test_wrong_loaded_maintenance_source_pin_refuses_before_write(crash_store):
    before = original_state(crash_store)
    instance = owner(crash_store, replace(crash_store["binding"], maintenance_source_sha256="0" * 64))
    with pytest.raises(maintenance.NativeExecutionWalMaintenanceError, match="maintenance_source_changed"):
        instance.run()
    assert instance.failure_state["physical_mutation_possible"] is False
    assert original_state(crash_store) == before and list(crash_store["backups"].iterdir()) == []
    assert_released(instance, crash_store)


def test_nonprivate_backup_directory_refuses_before_replay(crash_store):
    before = original_state(crash_store)
    crash_store["backups"].chmod(0o755)  # Only this new fixture directory changes.
    instance = owner(crash_store)
    with pytest.raises(maintenance.NativeExecutionWalMaintenanceError, match="maintenance_backup_directory_not_private"):
        instance.run()
    assert instance.failure_state["physical_mutation_possible"] is False
    assert original_state(crash_store) == before and list(crash_store["backups"].iterdir()) == []
    assert_released(instance, crash_store)


def test_owner_selector_and_deadline_cannot_be_rebound(crash_store):
    before = original_state(crash_store)
    instance = owner(crash_store)
    with pytest.raises(AttributeError):
        instance.binding = crash_store["binding"]
    with pytest.raises(AttributeError):
        instance.deadline_monotonic = time.monotonic() + 600
    assert instance.binding is crash_store["binding"]
    instance.close()
    assert original_state(crash_store) == before
    assert_released(instance, crash_store)


class _ExplicitCloseFault:
    """Fixture fault only; all actual queries use the genuine native handle."""

    def __init__(self, raw):
        self.raw = raw
        self.close_failures_remaining = 2

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def close(self):
        if self.close_failures_remaining:
            self.close_failures_remaining -= 1
            raise RuntimeError("explicit disposable fixture native close fault")
        return self.raw.close()


def _inject_real_handle_close_fault(monkeypatch):
    genuine = duckdb.connect
    captured = []

    def capture(*args, **kwargs):
        proxy = _ExplicitCloseFault(genuine(*args, **kwargs))
        captured.append((Path(args[0]), proxy))
        return proxy

    monkeypatch.setattr(duckdb, "connect", capture)
    return captured


def _assert_failed_close_custody_then_retry(instance, store, captured, before):
    assert len(captured) == 1 and captured[0][0] != store["path"]
    assert captured[0][0] == store["backups"] / "shadow" / "shadow.db"
    proxy = captured[0][1]
    assert instance._connection is proxy and proxy.raw.execute("SELECT 1").fetchone() == (1,)
    assert instance.failure_state["physical_mutation_possible"] is False
    assert instance.failure_state["connection_close_verified"] is False
    assert instance.failure_state["original_locks_released"] is False
    assert instance.failure_state["close_retry_required"] is True
    assert original_state(store) == before
    for key in ("writer", "policy"):
        descriptor = os.open(store[key], os.O_RDONLY | os.O_CLOEXEC)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)
    instance.close()  # Explicit retry closes the same native handle and fences.
    assert_released(instance, store)
    with pytest.raises(duckdb.Error):
        proxy.raw.execute("SELECT 1")
    assert original_state(store) == before


def test_explicit_shadow_close_fault_retains_native_custody_and_blocks_original_write(crash_store, monkeypatch):
    before = original_state(crash_store)
    captured = _inject_real_handle_close_fault(monkeypatch)
    instance = owner(crash_store)
    with pytest.raises(maintenance.NativeExecutionWalMaintenanceError, match="maintenance_native_close_failed_custody_retained"):
        instance.run()
    _assert_failed_close_custody_then_retry(instance, crash_store, captured, before)


def test_explicit_policy_verify_and_close_fault_retains_native_custody_before_original_write(crash_store, monkeypatch):
    before = original_state(crash_store)
    captured = _inject_real_handle_close_fault(monkeypatch)
    genuine_verify = duckdb_state._verify_duckdb_connection_policy

    def explicit_verify_fault(connection, **kwargs):
        genuine_verify(connection, **kwargs)  # Actual sealed settings first.
        raise duckdb_state.DuckDBConnectionPolicyError("explicit fixture post-verification fault")

    monkeypatch.setattr(duckdb_state, "_verify_duckdb_connection_policy", explicit_verify_fault)
    instance = owner(crash_store)
    with pytest.raises(maintenance.NativeExecutionWalMaintenanceError, match="maintenance_native_close_failed_custody_retained"):
        instance.run()
    _assert_failed_close_custody_then_retry(instance, crash_store, captured, before)


@pytest.mark.parametrize("crash_store", ["extra_relation", "oversized_scalar"],
                         indirect=True, ids=["extra-user-relation", "oversized-json-scalar"])
def test_genuine_wal_shadow_preflight_refuses_unsupported_stored_material_before_original_write(crash_store):
    before = original_state(crash_store)
    instance = owner(crash_store)
    with pytest.raises(maintenance.NativeExecutionWalMaintenanceError):
        instance.run()
    assert instance.failure_state["physical_mutation_possible"] is False
    assert instance.failure_state["backup_verified"] is True
    assert original_state(crash_store) == before
    for original, name in (("path", "original.db"), ("wal", "original.db.wal")):
        assert fixture_producer.digest(crash_store["backups"] / name) == before[original][1]
    assert_released(instance, crash_store)


def test_explicit_custody_guard_loss_after_shadow_preflight_blocks_original_write(crash_store):
    before = original_state(crash_store)
    guard = FixtureCustodyGuard()
    instance = owner(crash_store, custody_guard=guard)
    guard.fail_force_when = lambda: instance.failure_state.get("shadow_preflight_verified") is True
    with pytest.raises(maintenance.NativeExecutionWalMaintenanceError):
        instance.run()
    assert instance.failure_state["shadow_preflight_verified"] is True
    assert instance.failure_state["physical_mutation_possible"] is False
    assert instance.failure_state["backup_verified"] is True
    assert any(force is True for force, _check in guard.calls)
    assert original_state(crash_store) == before
    for original, name in (("path", "original.db"), ("wal", "original.db.wal")):
        assert fixture_producer.digest(crash_store["backups"] / name) == before[original][1]
    assert_released(instance, crash_store)


def test_explicit_failure_after_actual_original_checkpoint_retains_backups_and_reports_possible_mutation(crash_store, monkeypatch):
    before = original_state(crash_store)
    genuine = duckdb.connect
    original_writes = []

    class AfterCheckpointFault:
        def __init__(self, raw):
            self.raw = raw

        def __getattr__(self, name):
            return getattr(self.raw, name)

        def execute(self, sql, *args, **kwargs):
            result = self.raw.execute(sql, *args, **kwargs)
            if sql.strip().upper() == "CHECKPOINT":
                original_writes.append("actual_checkpoint_returned")
                raise RuntimeError("explicit fixture failure after actual CHECKPOINT")
            return result

    def capture(*args, **kwargs):
        raw = genuine(*args, **kwargs)
        if Path(args[0]) == crash_store["path"] and kwargs.get("read_only") is False:
            return AfterCheckpointFault(raw)
        return raw

    monkeypatch.setattr(duckdb, "connect", capture)
    instance = owner(crash_store)
    with pytest.raises(maintenance.NativeExecutionWalMaintenanceError):
        instance.run()
    state = instance.failure_state
    assert original_writes == ["actual_checkpoint_returned"]
    assert state["physical_mutation_possible"] is True
    assert state["original_checkpoint_verified"] is False
    assert state["backup_verified"] is True
    for key in ("completion_authority", "automatic_retry_admitted", "semantic_admission"):
        assert state[key] is False
    for original, name in (("path", "original.db"), ("wal", "original.db.wal")):
        assert fixture_producer.digest(crash_store["backups"] / name) == before[original][1]
    assert_released(instance, crash_store)
    assert not crash_store["wal"].exists()
    refreshed = replace(crash_store["status_binding"], execution_file_identity=fixture_producer.identity(crash_store["path"]))
    with NativeExecutionStatusOwner(refreshed, deadline_monotonic=time.monotonic() + 30) as reader:
        observation = reader.observe_attempt()
        assert observation["attempt"]["status"] == "running"
        assert observation["provider_effect_state"] == "unknown"
        assert observation["observed_process_instance_id"] == "process:fixture-wal-committed"


def test_explicit_fixture_watchdog_custody_loss_interrupts_genuine_shadow_query_before_original_write(crash_store, monkeypatch):
    """Actual C++ query and interrupt; fixture callback supplies no native ACK."""
    import threading

    before = original_state(crash_store)
    genuine = duckdb.connect
    query_started = threading.Event()
    emergency_fired = threading.Event()
    watcher_guard_failed = threading.Event()
    query_interrupted = threading.Event()
    query_started_at = [None]
    timers = []
    captured = []
    interrupt_threads = []
    main_thread = threading.get_ident()

    class WatchdogFixtureGuard(FixtureCustodyGuard):
        def __call__(self, *, force_ack=False, check_only=False):
            super().__call__(force_ack=force_ack, check_only=check_only)
            # The short floor allows the real execute call to enter DuckDB
            # after publishing the event, rather than interrupting before it.
            if (check_only and query_started.is_set() and
                    time.monotonic() - query_started_at[0] >= 0.05):
                watcher_guard_failed.set()
                raise RuntimeError("explicit nonadmitting fixture watchdog custody loss")

    class ShadowQueryProxy:
        def __init__(self, raw):
            self.raw = raw

        def __getattr__(self, name):
            return getattr(self.raw, name)

        def interrupt(self):
            interrupt_threads.append(threading.get_ident())
            return self.raw.interrupt()

        def execute(self, sql, *args, **kwargs):
            result = self.raw.execute(sql, *args, **kwargs)
            if sql.strip().upper() == "CHECKPOINT":
                def emergency_interrupt():
                    emergency_fired.set()
                    self.raw.interrupt()

                timer = threading.Timer(3.0, emergency_interrupt)
                timer.daemon = True
                timers.append(timer)
                query_started_at[0] = time.monotonic()
                query_started.set()
                timer.start()
                try:
                    # Streaming, constant-memory work that cannot finish before
                    # the genuine 0.25s custody watcher interrupts the handle.
                    self.raw.execute("SELECT sum(hash(i)) FROM range(1000000000000) AS work(i)").fetchone()
                except duckdb.InterruptException:
                    query_interrupted.set()
                    raise
                raise AssertionError("genuine fixture query unexpectedly completed without interruption")
            return result

    def capture(*args, **kwargs):
        raw = genuine(*args, **kwargs)
        path = Path(args[0])
        captured.append(path)
        assert path != crash_store["path"], "watchdog must block every original connector"
        return ShadowQueryProxy(raw)

    monkeypatch.setattr(duckdb, "connect", capture)
    guard = WatchdogFixtureGuard()
    instance = owner(crash_store, custody_guard=guard)
    try:
        with pytest.raises(maintenance.NativeExecutionWalMaintenanceError):
            instance.run()
    finally:
        for timer in timers:
            timer.cancel()
            timer.join(timeout=0.5)
            assert not timer.is_alive()
        instance.close()
    assert query_started.is_set() and query_interrupted.is_set()
    assert watcher_guard_failed.is_set() and not emergency_fired.is_set()
    assert any(thread != main_thread for thread in interrupt_threads)
    assert instance._custody_lost.is_set() and instance._cancelled.is_set()
    assert captured == [crash_store["backups"] / "shadow" / "shadow.db"]
    assert instance.failure_state["physical_mutation_possible"] is False
    assert instance.failure_state["backup_verified"] is True
    assert original_state(crash_store) == before
    for original, name in (("path", "original.db"), ("wal", "original.db.wal")):
        assert fixture_producer.digest(crash_store["backups"] / name) == before[original][1]
    assert_released(instance, crash_store)
    assert instance._monitor is None
