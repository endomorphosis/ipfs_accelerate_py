"""A replica withdrawal after ATTACH must not break a pure get_task read."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    IntentRepository,
)
from test.api.test_agent_supervisor_quack_owner_mutation import (
    _admitted_observation,
    _isolation_receipt,
    _isolation_server_kwargs,
    _seed,
    build_server,
    probe_quack_capabilities,
)


@pytest.fixture
def real_owner(tmp_path, monkeypatch):
    database = tmp_path / "control" / "control.duckdb"
    _seed(database)
    receipt_path, receipt = _isolation_receipt(tmp_path)
    server = build_server(
        database_path=database,
        state_dir=receipt_path.parent,
        **_isolation_server_kwargs(receipt),
        store_id=str(database),
        repository_id="repository:test",
        isolation_receipt_path=receipt_path,
        isolation_observer=_admitted_observation,
        capability_probe=lambda **_: probe_quack_capabilities(),
    )
    identity = server.start()
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", raising=False)
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE", identity.secret_handle
    )
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(database))
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION", str(identity.generation)
    )
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_LIVE_SCHEMA_REVISION",
        str(identity.schema_revision),
    )
    monkeypatch.setenv("IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT", str(tmp_path))
    repo = IntentRepository(identity.listen_uri, install_schema=False)
    try:
        yield server, repo
    finally:
        repo.close()
        server.stop()


def test_real_endpoint_refresh_after_attach_replays_only_task_read(
    real_owner, monkeypatch
):
    server, repo = real_owner
    original = duckdb_state._open_quack_transport_connection_once
    attempts, closed, failures = [], [], []

    def interrupted_open(*args, **kwargs):
        connection = original(*args, **kwargs)
        attempts.append(connection)
        close = connection.close

        def track_close():
            closed.append(connection)
            close()

        connection.close = track_close
        if len(attempts) == 1:
            execute = connection.execute
            server._stop_transport_connection(observe_closed=True)

            def first_query(*a, **kw):
                try:
                    return execute(*a, **kw)
                except Exception as exc:
                    failures.append(type(exc).__name__)
                    raise
                finally:
                    server._refresh_read_replica()

            connection.execute = first_query
        else:
            assert attempts[0] in closed, "failed handle must close before reattach"
        return connection

    monkeypatch.setattr(
        duckdb_state, "_open_quack_transport_connection_once", interrupted_open
    )
    result = repo.get_task("task:test")
    assert (
        result is not None and result["revision"] == 1 and result["status"] == "ready"
    )
    assert failures == ["IOException"]
    assert len(attempts) == 2 and closed == attempts
    assert (
        server._connection.execute(
            "SELECT COUNT(*) FROM completion_receipts"
        ).fetchone()[0]
        == 0
    )
    row = server._connection.execute(
        "SELECT status,revision FROM tasks WHERE task_cid='task:test'"
    ).fetchone()
    assert (row[0], row[1]) == ("ready", 1)


@pytest.mark.parametrize("failure", ["policy", "other_io", "query_error"])
def test_non_transport_errors_are_not_replayed(real_owner, monkeypatch, failure):
    import duckdb

    server, repo = real_owner
    original = duckdb_state._open_quack_transport_connection_once
    attempts, closed = [], []
    error = {
        "policy": duckdb_state.DuckDBConnectionPolicyError("denied"),
        "other_io": duckdb.IOException("IO Error: permission denied"),
        "query_error": ValueError("invalid row"),
    }[failure]

    def broken(*args, **kwargs):
        connection = original(*args, **kwargs)
        attempts.append(connection)
        old_close = connection.close

        def close():
            closed.append(connection)
            old_close()

        def execute(*_args, **_kwargs):
            raise error

        connection.close = close
        connection.execute = execute
        return connection

    monkeypatch.setattr(duckdb_state, "_open_quack_transport_connection_once", broken)
    with pytest.raises(type(error)) as observed:
        repo.get_task("task:test")
    assert observed.value is error
    assert len(attempts) == 1 and closed == attempts
    assert (
        server._connection.execute(
            "SELECT COUNT(*) FROM completion_receipts"
        ).fetchone()[0]
        == 0
    )


@pytest.mark.parametrize(
    "field",
    [
        "server_id",
        "store_id",
        "database_uuid",
        "schema_revision",
        "generation",
        "process_birth_id",
        "listen_uri",
        "extension_fingerprint",
        "schema_fingerprint",
        "generation_type",
    ],
)
def test_retry_never_adopts_a_different_owner_binding(real_owner, monkeypatch, field):
    import duckdb

    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
        IntentRepositoryReadUnavailableError,
    )

    _server, repo = real_owner
    original = duckdb_state._open_quack_transport_connection_once
    attempts, closed = [], []

    def changed(*args, **kwargs):
        connection = original(*args, **kwargs)
        attempts.append(connection)
        old_close = connection.close

        def close():
            closed.append(connection)
            old_close()

        connection.close = close
        if len(attempts) == 1:

            def lost(*_args, **_kwargs):
                raise duckdb.IOException(
                    "IO Error: Failed to send message: IO Error: Could not connect to server"
                )

            connection.execute = lost
        else:
            assert attempts[0] in closed
            binding = dict(connection._quack_mutation_binding)
            if field == "generation_type":
                binding["generation"] = True
            else:
                value = binding[field]
                binding[field] = value + 1 if type(value) is int else value + ":foreign"
            connection._quack_mutation_binding = binding
        return connection

    monkeypatch.setattr(duckdb_state, "_open_quack_transport_connection_once", changed)
    with pytest.raises(IntentRepositoryReadUnavailableError, match="binding"):
        repo.get_task("task:test")
    assert len(attempts) == 2 and closed == attempts


def test_query_timeout_interrupts_only_owned_read_and_retires_timer(
    real_owner, monkeypatch
):
    import threading
    import time

    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        quack_read_continuity as continuity,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
        IntentRepositoryReadUnavailableError,
    )

    server, repo = real_owner
    original = duckdb_state._open_quack_transport_connection_once
    closed, attempts = [], []

    def slow_read(*args, **kwargs):
        connection = original(*args, **kwargs)
        attempts.append(connection)
        old_close = connection.close

        def close():
            closed.append(connection)
            old_close()

        def execute(*_args, **_kwargs):
            return connection._connection.execute(
                "SELECT sum(sin(i)) FROM range(10000000000) t(i)"
            )

        connection.close = close
        connection.execute = execute
        return connection

    monkeypatch.setattr(continuity, "READ_RETRY_SECONDS", 0.25)
    monkeypatch.setattr(
        duckdb_state, "_open_quack_transport_connection_once", slow_read
    )
    started = time.monotonic()
    with pytest.raises(IntentRepositoryReadUnavailableError, match="deadline"):
        repo.get_task("task:test")
    assert time.monotonic() - started < 2
    assert len(attempts) == 1 and closed == attempts
    assert not any(t.name == "quack-task-read-deadline" for t in threading.enumerate())
    assert server._connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1


def test_repeated_reattach_and_queries_share_one_deadline(real_owner, monkeypatch):
    import time

    import duckdb

    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        quack_read_continuity as continuity,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
        IntentRepositoryReadUnavailableError,
    )

    _server, repo = real_owner
    original = duckdb_state._open_quack_transport_connection_once
    deadlines, attempts, closed = [], [], []

    def lose(*args, **kwargs):
        deadlines.append(kwargs["deadline_monotonic"])
        connection = original(*args, **kwargs)
        attempts.append(connection)
        old_close = connection.close

        def close():
            closed.append(connection)
            old_close()

        def execute(*_args, **_kwargs):
            time.sleep(0.03)
            raise duckdb.IOException(
                "IO Error: Failed to send message: IO Error: Could not connect to server"
            )

        connection.close = close
        connection.execute = execute
        return connection

    monkeypatch.setattr(continuity, "READ_RETRY_SECONDS", 0.25)
    monkeypatch.setattr(duckdb_state, "_open_quack_transport_connection_once", lose)
    started = time.monotonic()
    with pytest.raises(IntentRepositoryReadUnavailableError, match="deadline"):
        repo.get_task("task:test")
    assert time.monotonic() - started < 2
    assert len(set(deadlines)) == 1 and len(deadlines) >= 2
    assert closed == attempts


def test_local_and_borrowed_reads_never_enter_reconnect_loop(tmp_path, monkeypatch):
    import contextlib

    import duckdb

    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        quack_read_continuity as continuity,
    )

    path = tmp_path / "local.duckdb"
    _seed(path)
    repo = IntentRepository(path, install_schema=False)

    def forbidden(**_kwargs):
        pytest.fail("local or borrowed handles cannot be replayed")

    monkeypatch.setattr(continuity, "read_task_projection", forbidden)
    assert repo.get_task("task:test")["revision"] == 1
    raw = duckdb.connect(str(path))
    raw.execute("BEGIN")
    raw.execute(
        "UPDATE tasks SET priority='own-uncommitted' WHERE task_cid='task:test'"
    )
    wrapped = duckdb_state.DuckDBConnection.wrap(raw)

    @contextlib.contextmanager
    def borrowed(*, write=False):
        assert write is False
        yield wrapped

    repo._quack_transport = True
    repo._connection = borrowed
    assert repo.get_task("task:test")["priority"] == "own-uncommitted"
    raw.execute("COMMIT")
    assert raw.execute("SELECT priority FROM tasks").fetchone()[0] == "own-uncommitted"
    raw.close()
    repo.close()


def test_actual_attach_work_uses_the_same_interruptible_deadline(
    real_owner, monkeypatch
):
    import threading
    import time

    import duckdb

    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        quack_read_continuity as continuity,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
        IntentRepositoryReadUnavailableError,
    )

    server, repo = real_owner
    original_connect = duckdb.connect
    opened, closed = [], []

    class SlowAttach:
        def __init__(self, raw):
            self.raw = raw

        def __getattr__(self, name):
            return getattr(self.raw, name)

        def execute(self, sql, *args, **kwargs):
            if sql.startswith("ATTACH "):
                return self.raw.execute(
                    "SELECT sum(sin(i)) FROM range(10000000000) t(i)"
                )
            return self.raw.execute(sql, *args, **kwargs)

        def close(self):
            assert not any(
                t.name == "quack-task-read-deadline" for t in threading.enumerate()
            )
            self.raw.close()
            closed.append(self)

    def connect(*args, **kwargs):
        assert args == (":memory:",), "task reads cannot open a canonical database"
        connection = SlowAttach(original_connect(*args, **kwargs))
        opened.append(connection)
        return connection

    monkeypatch.setattr(duckdb, "connect", connect)
    monkeypatch.setattr(continuity, "READ_RETRY_SECONDS", 0.25)
    started = time.monotonic()
    with pytest.raises(IntentRepositoryReadUnavailableError, match="deadline"):
        repo.get_task("task:test")
    assert time.monotonic() - started < 2
    assert len(opened) == 1 and closed == opened
    assert server._connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1


def test_real_endpoint_unavailable_before_attach_recovers_with_original_deadline(
    real_owner, monkeypatch
):
    server, repo = real_owner
    original = duckdb_state._open_quack_transport_connection_once
    attempts, failures = [], []
    server._stop_transport_connection(observe_closed=True)

    def restore_after_unavailable(*args, **kwargs):
        attempts.append(kwargs["deadline_monotonic"])
        try:
            return original(*args, **kwargs)
        except Exception as error:
            failures.append(type(error).__name__)
            server._refresh_read_replica()
            raise

    monkeypatch.setattr(
        duckdb_state, "_open_quack_transport_connection_once", restore_after_unavailable
    )
    result = repo.get_task("task:test")
    assert result["status"] == "ready" and result["revision"] == 1
    assert failures == ["IOException"]
    assert len(attempts) == 2 and len(set(attempts)) == 1
