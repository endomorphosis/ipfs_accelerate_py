"""Complete quarantine guards survive only same-owner replica withdrawals."""
from __future__ import annotations

import contextlib
import json
import shutil
import threading
import time

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as state
from ipfs_accelerate_py.agent_supervisor.task_sources import owner_task_quarantine as quarantine
from ipfs_accelerate_py.agent_supervisor.task_sources import quack_read_continuity as continuity
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    IntentRepository, IntentRepositoryReadUnavailableError,
)
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import DatabaseCoordinationExpiredError
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationProviderDispatchError
from test.api.test_pctdd_task_read_continuity import real_owner as base_owner
from test.api.test_owner_task_quarantine import apply, quarantine_request
from test.api.test_agent_supervisor_database_implementation_daemon import _open_daemon, _population

LOST = "IO Error: Failed to send message: IO Error: Could not connect to server"


@pytest.fixture
def owned(tmp_path, monkeypatch):
    fixture = base_owner.__wrapped__(tmp_path, monkeypatch)
    try:
        server, repo = next(fixture)
    except BaseException:
        fixture.close()
        shutil.rmtree(tmp_path)
        raise
    daemons = []

    def daemon():
        path = tmp_path / "daemon"
        path.mkdir()
        clock = {"now": 1000}
        value = _open_daemon(path, clock_ms=lambda: clock["now"])
        entry = [value, None]
        daemons.append(entry)
        population = _population(1)
        population["tasks"][0]["task_cid"] = "task:test"
        value.materialize_population(population)
        attempt = value.claim_next()
        assert attempt is not None
        original_intent = value.task_source.intent
        entry[1] = original_intent
        value.task_source._intent = repo
        return value, attempt, clock

    try:
        yield server, repo, daemon
    finally:
        for value, original in daemons:
            if original is not None:
                value.task_source._intent = original
            value.close()
        fixture.close()
        assert not any(t.name == "quack-task-read-deadline" for t in threading.enumerate())
        assert not list(tmp_path.rglob(".git"))
        details = tmp_path.stat()
        with (tmp_path.parent / "closed-fixtures.jsonl").open("a") as ledger:
            ledger.write(json.dumps({"path": str(tmp_path), "device": details.st_dev,
                                    "inode": details.st_ino, "owner_closed": True}) + "\n")
        shutil.rmtree(tmp_path)


def withdraw(owned, monkeypatch, select, *, after_loss=None, drift=None):
    server, _, _ = owned
    original = state._open_quack_transport_connection_once
    calls, handles, closed, deadlines = [], [], [], []

    def open_connection(*args, **kwargs):
        deadlines.append(kwargs.get("deadline_monotonic"))
        connection = original(*args, **kwargs)
        handles.append(connection)
        index = len(handles)
        execute, close = connection.execute, connection.close
        queries = []
        calls.append(queries)

        def closed_connection():
            assert not any(t.name == "quack-task-read-deadline" for t in threading.enumerate())
            closed.append(connection)
            close()

        def query(sql, *a, **kw):
            queries.append(sql)
            if index == 1 and len(queries) == select:
                server._stop_transport_connection(observe_closed=True)
                try:
                    return execute(sql, *a, **kw)
                finally:
                    if after_loss is not None:
                        after_loss()
                    server._refresh_read_replica()
            return execute(sql, *a, **kw)

        connection.execute, connection.close = query, closed_connection
        if index > 1:
            assert handles[index-2] in closed
            if drift:
                binding = dict(connection._quack_mutation_binding)
                value = binding[drift]
                binding[drift] = value + 1 if type(value) is int else value + ":foreign"
                connection._quack_mutation_binding = binding
        return connection

    monkeypatch.setattr(state, "_open_quack_transport_connection_once", open_connection)
    return calls, handles, closed, deadlines


def begin(daemon, attempt, callbacks):
    daemon._begin_callback_dispatch(attempt, dispatch_kind="provider",
                                    idempotency_key=f"provider:{attempt.attempt_id}")
    callbacks.append(attempt.attempt_id)


def journal(daemon):
    return daemon._require_connection().execute("SELECT * FROM attempt_dispatch_journal").fetchall()


@pytest.mark.parametrize("select", [1, 2, 3])
def test_real_replica_loss_restarts_entire_guard_without_replaying_admission(owned, monkeypatch, select):
    _, _, make_daemon = owned
    daemon, attempt, _ = make_daemon()
    callbacks = []
    calls, handles, closed, deadlines = withdraw(owned, monkeypatch, select)
    begin(daemon, attempt, callbacks)
    assert callbacks == [attempt.attempt_id]
    assert len(journal(daemon)) == 1
    assert len(calls) == 2 and len(calls[0]) == select and len(calls[1]) == 3
    assert "count(*)" in calls[1][0] and "ORDER BY global_sequence" in calls[1][1]
    assert "control_plane_metadata" in calls[1][2]
    assert handles == closed and len(set(deadlines)) == 1 and deadlines[0] is not None
    # A later read is not permission to replay an already entered callback.
    before = journal(daemon)
    with pytest.raises(DatabaseImplementationProviderDispatchError, match="non-replayable"):
        begin(daemon, attempt, callbacks)
    assert callbacks == [attempt.attempt_id] and journal(daemon) == before


@pytest.mark.parametrize("existing_marker", [False, True])
def test_new_quarantine_on_retry_blocks_callback_and_preserves_unknown_journal(owned, monkeypatch, existing_marker):
    server, _, make_daemon = owned
    daemon, attempt, _ = make_daemon()
    callbacks = []
    token = server._vault.resolve(server.identity.secret_handle)
    if existing_marker:
        daemon._begin_callback_dispatch(attempt, dispatch_kind="provider",
                                        idempotency_key=f"provider:{attempt.attempt_id}")
    before = journal(daemon)

    def append_quarantine():
        apply(server, quarantine_request(server, token))

    withdraw(owned, monkeypatch, 2, after_loss=append_quarantine)
    with pytest.raises(quarantine.QuarantineDenied, match="task_custody_quarantined"):
        begin(daemon, attempt, callbacks)
    assert callbacks == [] and journal(daemon) == before


def test_expired_claim_is_rechecked_after_successful_guard_retry(owned, monkeypatch):
    _, _, make_daemon = owned
    daemon, attempt, clock = make_daemon()
    claim = daemon.coordinator.get_task_claim(attempt.claim_id)
    callbacks = []
    withdraw(owned, monkeypatch, 3,
             after_loss=lambda: clock.update(now=claim.expires_at_ms + 1))
    with pytest.raises(DatabaseCoordinationExpiredError):
        begin(daemon, attempt, callbacks)
    assert callbacks == [] and journal(daemon) == []


@pytest.mark.parametrize("field", sorted(continuity._BINDING_FIELDS))
def test_quarantine_read_never_adopts_a_changed_owner_field(owned, monkeypatch, field):
    _, repo, _ = owned
    _, handles, closed, _ = withdraw(owned, monkeypatch, 2, drift=field)
    with pytest.raises(IntentRepositoryReadUnavailableError, match="binding"):
        repo.owner_task_quarantines()
    assert len(handles) == 2 and closed == handles


@pytest.mark.parametrize("kind", ["quarantine", "policy", "unknown", "other_io", "stop"])
def test_non_replica_errors_and_process_stops_keep_exact_identity(owned, monkeypatch, kind):
    _, repo, _ = owned
    original = state._open_quack_transport_connection_once
    error = {"quarantine": quarantine.QuarantineDenied("retained_history_invalid"),
             "policy": state.DuckDBConnectionPolicyError("denied"),
             "unknown": RuntimeError("read_replica_refresh_unknown_outcome canonical_effects_present=True"),
             "other_io": duckdb.IOException("IO Error: different I/O failure"),
             "stop": SystemExit(143)}[kind]
    handles = []

    def open_connection(*args, **kwargs):
        connection = original(*args, **kwargs)
        handles.append(connection)
        def fail(*a, **kw):
            raise error
        connection.execute = fail
        return connection

    monkeypatch.setattr(state, "_open_quack_transport_connection_once", open_connection)
    with pytest.raises(type(error)) as caught:
        repo.owner_task_quarantines()
    assert caught.value is error and len(handles) == 1


def test_one_deadline_closes_all_handles_and_admits_no_callback(owned, monkeypatch):
    _, _, make_daemon = owned
    daemon, attempt, _ = make_daemon()
    original = state._open_quack_transport_connection_once
    opened, closed, deadlines = [], [], []
    def open_connection(*args, **kwargs):
        connection = original(*args, **kwargs)
        deadlines.append(kwargs.get("deadline_monotonic"))
        opened.append(connection)
        close = connection.close
        def fail(*a, **kw):
            time.sleep(.03)
            raise duckdb.IOException(LOST)
        def close_connection():
            close()
            closed.append(connection)
        connection.execute, connection.close = fail, close_connection
        return connection
    monkeypatch.setattr(state, "_open_quack_transport_connection_once", open_connection)
    monkeypatch.setattr(continuity, "READ_RETRY_SECONDS", .25)
    callbacks = []
    with pytest.raises(IntentRepositoryReadUnavailableError, match="deadline"):
        begin(daemon, attempt, callbacks)
    assert callbacks == [] and journal(daemon) == []
    assert len(opened) >= 1 and opened == closed and len(set(deadlines)) == 1


@pytest.mark.parametrize("mode", ["borrowed", "subclass"])
def test_borrowed_and_subclass_quarantine_reads_do_not_reopen(owned, monkeypatch, mode):
    _, repo, _ = owned
    raw = duckdb.connect(":memory:")
    raw.execute("CREATE TABLE domain_events(event_id VARCHAR, stream_id VARCHAR, sequence BIGINT, global_sequence BIGINT, event_type VARCHAR, task_cid VARCHAR, attempt_id VARCHAR, session_id VARCHAR, recorded_at VARCHAR, body_json VARCHAR)")
    raw.execute("CREATE TABLE control_plane_metadata(key VARCHAR, value VARCHAR)")
    wrapped = state.DuckDBConnection.wrap(raw)
    @contextlib.contextmanager
    def borrowed(*, write=False):
        assert write is False
        yield wrapped
    if mode == "borrowed":
        repo._connection = borrowed
        reader = repo
    else:
        class Reader(IntentRepository):
            def _connection(self, *, write=False):
                return borrowed(write=write)
        reader = Reader(repo._open_target, install_schema=False)
    def forbidden(**kwargs):
        pytest.fail("caller-owned/subclass handles cannot enter continuity")
    monkeypatch.setattr(continuity, "read_owner_quarantine_heads", forbidden)
    monkeypatch.setattr(continuity, "read_owner_quarantine_observation", forbidden)
    try:
        raw.execute("BEGIN")
        assert reader.owner_task_quarantines() == {}
        assert raw.execute("SELECT count(*) FROM domain_events").fetchone() == (0,)
        raw.execute("ROLLBACK")
    finally:
        raw.close()


def test_late_quarantine_denial_is_not_masked_as_deadline_unavailable(owned, monkeypatch):
    _, repo, _ = owned
    original = state._open_quack_transport_connection_once
    error = quarantine.QuarantineDenied("late_complete_history_denial")
    handles = []
    def open_connection(*args, **kwargs):
        connection = original(*args, **kwargs)
        handles.append(connection)
        def fail(*a, **kw):
            monkeypatch.setattr(continuity.time, "monotonic", lambda: kwargs["deadline_monotonic"] + 1)
            raise error
        connection.execute = fail
        return connection
    monkeypatch.setattr(state, "_open_quack_transport_connection_once", open_connection)
    with pytest.raises(quarantine.QuarantineDenied) as caught:
        repo.owner_task_quarantines()
    assert caught.value is error and len(handles) == 1


def test_retry_revalidates_anchor_and_denies_corrupted_history(owned, monkeypatch):
    server, _, make_daemon = owned
    daemon, attempt, _ = make_daemon()
    def corrupt_anchor():
        server._connection.execute("INSERT INTO control_plane_metadata VALUES (?, ?, ?)",
                                   [quarantine.ANCHOR_KEY, "{}", "now"])
    calls, _, _, _ = withdraw(owned, monkeypatch, 3, after_loss=corrupt_anchor)
    callbacks = []
    with pytest.raises(quarantine.QuarantineDenied, match="anchor_mismatch"):
        begin(daemon, attempt, callbacks)
    assert callbacks == [] and journal(daemon) == [] and len(calls) == 2
