"""Actual peer sockets and kernel writer locks authenticate bounded observations."""
from __future__ import annotations

import fcntl
import errno
import json
import multiprocessing
import os
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import owner_status_observation as observation
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import DuckDBConnection

CONFIG = {"board_namespace": "test-board", "database_program": {"store_generation": 1}}
EXPECTED = {"store_id": "control.duckdb", "repository_id": "repository:test-board", "generation": 1}
SECRET = "private-owner-token-must-never-appear"


def _owner(directory, control):
    import duckdb

    base = Path(directory)
    database = base / "control.duckdb"
    lock_fds = []
    connection = None
    listener = None
    try:
        for lock in observation._locks(database):
            fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            lock_fds.append(fd)
        connection = duckdb.connect(str(database), config={"threads": 1})
        connection.execute("CREATE TABLE store_generations (generation BIGINT, database_uuid VARCHAR, birth_id VARCHAR, fence_epoch BIGINT)")
        connection.execute("INSERT INTO store_generations VALUES (1, 'database:test', 'birth:test', 1)")
        connection.execute("CREATE TABLE tasks (task_cid VARCHAR, task_alias VARCHAR, status VARCHAR, revision BIGINT)")
        connection.execute("INSERT INTO tasks VALUES ('task:a', 'TEST-001', 'completed', 2), ('task:b', 'TEST-002', 'in_progress', 3)")
        connection.execute("CREATE TABLE domain_events (global_sequence BIGINT)")
        connection.execute("INSERT INTO domain_events VALUES (7)")
        identity = {**EXPECTED, "server_id": "server:test", "database_uuid": "database:test",
                    "fence_epoch": 1, "process_birth_id": "birth:test", "secret_handle": SECRET}

        class Server:
            config = SimpleNamespace(database_path=database, state_dir=base)
            lifecycle = SimpleNamespace(value="ready")
            _connection = DuckDBConnection.wrap(connection)
            _lock = threading.RLock()

            @property
            def _vault(self):
                raise AssertionError("observation accessed credentials")

        server = Server()
        server.identity = SimpleNamespace(to_dict=lambda: dict(identity))
        listener = observation.OwnerStatusObservation(server, program_id="test-board", configuration=CONFIG,
            source_head="a" * 40, source_tree="b" * 40, task_registry={"task:a": "TEST-001", "task:b": "TEST-002"})
        control.send({"ready": True, "scope": listener.scope})
        while True:
            if control.poll():
                command = control.recv()
                if command == "stop":
                    break
                if command == "release":
                    fcntl.flock(lock_fds[0], fcntl.LOCK_UN)
                elif command == "stale":
                    clock = time.time
                    observation.time.time = lambda: clock() - 100
                elif command == "foreign_task":
                    connection.execute("UPDATE tasks SET task_cid='task:foreign' WHERE task_cid='task:b'")
                elif command == "wrong_generation":
                    connection.execute("UPDATE store_generations SET fence_epoch=2")
                elif command == "accept_error":
                    original = listener.listener
                    class FailOnce:
                        failed = False
                        def fileno(self):
                            return original.fileno()
                        def accept(self):
                            if not self.failed:
                                self.failed = True
                                raise OSError(errno.EMFILE, SECRET)
                            return original.accept()
                        def close(self):
                            original.close()
                    listener.listener = FailOnce()
                elif command == "query_error":
                    original_snapshot = listener._snapshot
                    failed = False
                    def fail_query_once():
                        nonlocal failed
                        if not failed:
                            failed = True
                            raise RuntimeError(SECRET)
                        return original_snapshot()
                    listener._snapshot = fail_query_once
                elif command == "slow_query":
                    original_snapshot = listener._snapshot
                    def slow_snapshot():
                        time.sleep(0.15)
                        return original_snapshot()
                    listener._snapshot = slow_snapshot
                elif command == "long_query_once":
                    original_bounded_snapshot = listener._bounded_snapshot
                    long_query_pending = True
                    def long_query_once(read_connection):
                        nonlocal long_query_pending
                        if long_query_pending:
                            long_query_pending = False
                            connection.execute("SELECT SUM(a.i * b.i) FROM range(10000000) a(i), range(10000000) b(i)")
                        return original_bounded_snapshot(read_connection)
                    listener._bounded_snapshot = long_query_once
                elif command == "forbid_parameter_conversion":
                    original_snapshot = listener._bounded_snapshot
                    def without_conversion(connection):
                        class ReadConnection:
                            def execute(self, query, *args, **kwargs):
                                if args or kwargs:
                                    raise AssertionError("optional parameter conversion entered")
                                return connection.execute(query)
                        return original_snapshot(ReadConnection())
                    listener._bounded_snapshot = without_conversion
                elif command == "normal_mutation":
                    connection.execute("UPDATE tasks SET revision=revision+1 WHERE task_cid='task:b'")
                elif command == "state":
                    control.send({"tasks": connection.execute("SELECT * FROM tasks ORDER BY task_cid").fetchall(),
                                  "events": connection.execute("SELECT * FROM domain_events").fetchall()})
                    continue
                listener.next_sample = 0
                control.send({"ready": True})
            listener.poll()
            time.sleep(0.005)
    except BaseException as exc:
        control.send({"failed_type": type(exc).__name__})
    finally:
        if listener is not None:
            listener.close()
        if connection is not None:
            connection.close()
        for fd in lock_fds:
            os.close(fd)


@pytest.fixture
def native_owner(tmp_path):
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe()
    process = context.Process(target=_owner, args=(str(tmp_path), child))
    process.start()
    child.close()
    assert parent.poll(20), "isolated owner startup exceeded test budget"
    ready = parent.recv()
    assert ready.get("ready") is True, ready
    try:
        yield tmp_path, parent, process, ready["scope"]
    finally:
        if process.is_alive():
            parent.send("stop")
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()
        assert not process.is_alive()


def _read(base, **changes):
    return observation.read_owner_status(database=base / "control.duckdb", state_dir=base,
        program_id=changes.get("program_id", "test-board"), configuration=changes.get("configuration", CONFIG),
        expected_owner=changes.get("expected_owner", EXPECTED),
        expected_task_registry=changes.get("expected_task_registry", {"task:a": "TEST-001", "task:b": "TEST-002"}))


def test_real_owner_reply_is_fresh_bound_and_read_only(native_owner):
    base, control, process, scope = native_owner
    control.send("state")
    before = control.recv()
    result = _read(base)
    assert result["peer_authenticated_observation"] is True
    assert result["owner_identity"] == scope["owner_identity"]
    facts = result["task_authority"]
    assert facts["task_counts"] == {"completed": 1, "in_progress": 1}
    assert facts["event_cursor"] == 7
    assert facts["task_revisions"] == {"TEST-001": 2, "TEST-002": 3}
    assert result["completion_authority"] is False
    assert result["source_transition_authority"] is False
    assert result["source_context_only"] is True and result["source_verified"] is False
    assert SECRET not in json.dumps(result)
    control.send("state")
    assert control.recv() == before
    assert process.is_alive()


@pytest.mark.parametrize("mismatch", ["program", "configuration", "registry", "owner", "owner_type", "birth", "source", "uid"])
def test_locator_does_not_authenticate_a_foreign_binding(native_owner, mismatch):
    base, _control, _process, scope = native_owner
    options = {}
    if mismatch == "program":
        options["program_id"] = "foreign"
    elif mismatch == "configuration":
        options["configuration"] = {"foreign": True}
    elif mismatch == "registry":
        options["expected_task_registry"] = {"task:a": "FOREIGN-001", "task:b": "TEST-002"}
    elif mismatch == "owner":
        options["expected_owner"] = {**EXPECTED, "generation": 2}
    elif mismatch == "owner_type":
        options["expected_owner"] = {**EXPECTED, "generation": True}
    else:
        if mismatch == "birth":
            scope["owner_birth"]["start_time_ticks"] += 1
        elif mismatch == "source":
            scope["source_head"] = "f" * 40
        else:
            scope["uid"] += 1
        (base / observation.DESCRIPTOR).write_text(json.dumps(scope))
    with pytest.raises(observation.OwnerObservationUnavailable):
        _read(base, **options)


@pytest.mark.parametrize("command", ["release", "stale", "foreign_task", "wrong_generation"])
def test_no_snapshot_from_lost_custody_stale_or_foreign_rows(native_owner, command):
    base, control, process, _scope = native_owner
    control.send(command)
    assert control.recv() == {"ready": True}
    with pytest.raises(observation.OwnerObservationUnavailable):
        _read(base)
    assert process.is_alive()


@pytest.mark.parametrize("packet", [b'{"x":' + b'[' * 1200 + b'0' + b']' * 1200 + b'}',
                                    b'{"schema":"one","schema":"two"}', b'x' * 70000],
                         ids=["nested", "duplicate", "oversized"])
def test_malformed_request_cannot_stop_owner_or_grant_observation(native_owner, packet):
    base, _control, process, scope = native_owner
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
        connection.settimeout(2)
        connection.connect(observation._address(scope))
        connection.sendall(packet)
        reply = json.loads(connection.recv(observation.MAX_PACKET))
    assert reply["task_authority"] is None
    assert reply["completion_authority"] is False
    assert SECRET not in json.dumps(reply)
    assert _read(base)["peer_authenticated_observation"] is True
    assert process.is_alive()


def test_fifo_descriptor_is_bounded_and_never_authenticates(native_owner):
    base, _control, process, _scope = native_owner
    path = base / observation.DESCRIPTOR
    path.unlink()
    os.mkfifo(path)
    started = time.monotonic()
    with pytest.raises(observation.OwnerObservationUnavailable):
        _read(base)
    assert time.monotonic() - started < 1
    assert process.is_alive()


def test_read_only_holder_with_both_conventional_locks_is_not_native_writer(tmp_path):
    import duckdb
    database = tmp_path / "control.duckdb"
    writer = duckdb.connect(str(database))
    writer.execute("CREATE TABLE preserved (value INTEGER)")
    writer.execute("INSERT INTO preserved VALUES (7)")
    writer.close()
    descriptors = []
    reader = duckdb.connect(str(database), read_only=True)
    try:
        for path in observation._locks(database):
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            descriptors.append(fd)
        identity = {**EXPECTED, "server_id": "server:test", "database_uuid": "database:test",
                    "fence_epoch": 1, "process_birth_id": "birth:test"}
        server = SimpleNamespace(config=SimpleNamespace(database_path=database, state_dir=tmp_path),
            identity=SimpleNamespace(to_dict=lambda: identity), _connection=DuckDBConnection.wrap(reader),
            lifecycle=SimpleNamespace(value="ready"), _lock=threading.RLock())
        with pytest.raises(observation.OwnerObservationUnavailable):
            observation.OwnerStatusObservation(server, program_id="test-board", configuration=CONFIG,
                source_head="a" * 40, source_tree="b" * 40, task_registry={"task:a": "TEST-001"})
        assert reader.execute("SELECT value FROM preserved").fetchone()[0] == 7
        assert not (tmp_path / observation.DESCRIPTOR).exists()
    finally:
        reader.close()
        for fd in descriptors:
            os.close(fd)


def test_client_rejects_wrong_kernel_peer(native_owner, monkeypatch):
    base, _control, process, _scope = native_owner
    monkeypatch.setattr(observation, "_peer", lambda _connection: (os.getpid(), os.getuid()))
    with pytest.raises(observation.OwnerObservationUnavailable):
        _read(base)
    assert process.is_alive()


def test_slow_snapshot_cannot_receive_a_fresh_timestamp_after_query(native_owner, monkeypatch):
    base, control, process, _scope = native_owner
    control.send("slow_query")
    assert control.recv() == {"ready": True}
    monkeypatch.setattr(observation, "MAX_AGE_SECONDS", 0.1)
    with pytest.raises(observation.OwnerObservationUnavailable):
        _read(base)
    assert process.is_alive()


@pytest.mark.parametrize("delay_before_query", [0.0, 0.1])
def test_native_query_budget_cancels_long_query_and_preserves_next_mutation(monkeypatch, delay_before_query):
    import duckdb
    connection = DuckDBConnection.wrap(duckdb.connect(":memory:", config={"threads": 1}))
    monkeypatch.setattr(observation, "QUERY_BUDGET_SECONDS", 0.05)
    connection.execute("CREATE TABLE progress (value BIGINT)")
    started = time.monotonic()
    try:
        with pytest.raises(duckdb.InterruptException):
            with observation._query_budget(connection) as read_connection:
                # The budget may expire between custody checks/read statements;
                # cancellation must remain armed until the observation returns.
                time.sleep(delay_before_query)
                read_connection.execute("SELECT SUM(a.i * b.i) FROM range(10000000) a(i), range(10000000) b(i)")
        assert time.monotonic() - started < 2
        assert not any(t.name == "native-owner-observation-deadline" for t in threading.enumerate())
        connection.execute("INSERT INTO progress VALUES (1)")
        time.sleep(0.1)
        assert connection.execute("SELECT SUM(value) FROM progress").fetchone()[0] == 1
        with observation._query_budget(connection):
            assert connection.execute("SELECT 42").fetchone()[0] == 42
        time.sleep(0.1)
        connection.execute("INSERT INTO progress VALUES (2)")
        assert connection.execute("SELECT SUM(value) FROM progress").fetchone()[0] == 3
    finally:
        connection.close()


def test_query_budget_rejects_active_transaction_without_interrupting_it():
    import duckdb
    connection = DuckDBConnection.wrap(duckdb.connect(":memory:"))
    try:
        connection.execute("CREATE TABLE progress (value BIGINT)")
        connection.execute("BEGIN TRANSACTION")
        connection.execute("INSERT INTO progress VALUES (1)")
        with pytest.raises(observation.OwnerObservationUnavailable):
            with observation._query_budget(connection):
                pytest.fail("observer entered an existing transaction")
        connection.commit()
        assert connection.execute("SELECT SUM(value) FROM progress").fetchone()[0] == 1
    finally:
        connection.close()


def test_blocked_observation_query_returns_to_native_work_loop(native_owner):
    base, control, process, _scope = native_owner
    control.send("long_query_once")
    assert control.recv() == {"ready": True}
    started = time.monotonic()
    with pytest.raises(observation.OwnerObservationUnavailable):
        _read(base)
    assert time.monotonic() - started < 2
    control.send("normal_mutation")
    assert control.poll(2), "observation prevented the retained owner's next mutation pass"
    assert control.recv() == {"ready": True}
    assert _read(base)["task_authority"]["task_revisions"]["TEST-002"] == 4
    assert process.is_alive()


@pytest.mark.parametrize("command", ["accept_error", "query_error"])
def test_transient_observer_failure_preserves_owner_and_recovers(native_owner, command):
    base, control, process, _scope = native_owner
    control.send(command)
    assert control.recv() == {"ready": True}
    if command == "query_error":
        with pytest.raises(observation.OwnerObservationUnavailable) as failure:
            _read(base)
        assert SECRET not in str(failure.value)
        control.send("reset")
        assert control.recv() == {"ready": True}
    assert _read(base)["peer_authenticated_observation"] is True
    assert process.is_alive()




@pytest.mark.parametrize("mismatch", ["scope", "birth", "extra"])
def test_well_formed_request_requires_exact_requester_binding(native_owner, mismatch):
    base, _control, process, scope = native_owner
    packet = {"schema": observation.SCHEMA, "scope_cid": observation._digest(scope),
              "requester_birth": observation._birth(os.getpid()), "nonce": "a" * 32}
    if mismatch == "scope":
        packet["scope_cid"] = "0" * 64
    elif mismatch == "birth":
        packet["requester_birth"]["start_time_ticks"] += 1
    else:
        packet["sql"] = "SELECT private_credentials"
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
        connection.settimeout(2)
        connection.connect(observation._address(scope))
        connection.sendall(json.dumps(packet).encode())
        reply = json.loads(connection.recv(observation.MAX_PACKET))
    assert reply["task_authority"] is None
    assert reply["completion_authority"] is False
    assert _read(base)["peer_authenticated_observation"] is True
    assert process.is_alive()


def test_cold_owner_observation_does_not_require_optional_parameter_conversion(native_owner):
    base, control, process, _scope = native_owner
    control.send("forbid_parameter_conversion")
    assert control.recv() == {"ready": True}
    result = _read(base)
    assert result["peer_authenticated_observation"] is True
    assert result["task_authority"]["task_revisions"] == {"TEST-001": 2, "TEST-002": 3}
    assert result["completion_authority"] is False
    control.send("normal_mutation")
    assert control.recv() == {"ready": True}
    assert _read(base)["task_authority"]["task_revisions"]["TEST-002"] == 4
    assert process.is_alive()


def test_observation_retires_interrupt_before_queued_typed_mutation(monkeypatch):
    import duckdb
    connection = DuckDBConnection.wrap(duckdb.connect(":memory:", config={"threads": 1}))
    connection.execute("CREATE TABLE progress (value BIGINT)")
    monkeypatch.setattr(observation, "QUERY_BUDGET_SECONDS", 0.05)
    attempting, completed = threading.Event(), threading.Event()
    errors = []
    def mutate():
        attempting.set()
        try:
            connection.execute("INSERT INTO progress VALUES (1)")
        except Exception as exc:
            errors.append(type(exc).__name__)
        finally:
            completed.set()
    worker = threading.Thread(target=mutate)
    try:
        with pytest.raises(duckdb.InterruptException):
            with observation._query_budget(connection) as read_connection:
                worker.start()
                assert attempting.wait(1)
                assert not completed.wait(0.01)
                read_connection.execute("SELECT SUM(a.i * b.i) FROM range(10000000) a(i), range(10000000) b(i)")
        worker.join(2)
        assert completed.is_set() and not errors
        assert [row[0] for row in connection.execute("SELECT * FROM progress").fetchall()] == [1]
        assert not connection._poisoned
        assert not any(t.name == "native-owner-observation-deadline" for t in threading.enumerate())
    finally:
        if worker.ident is not None:
            worker.join(2)
        connection.close()
