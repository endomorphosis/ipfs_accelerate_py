"""Scoped legacy reads reuse one native adapter without gaining write authority."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import intent_repository as module
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)


class CountedConnection:
    def __init__(self, connection):
        self.connection = connection
        self.thread = threading.get_ident()
        self.closes = 0
        self.statements = []
        self.close_error = False

    def execute(self, sql, *args):
        assert threading.get_ident() == self.thread, "connection crossed threads"
        self.statements.append(sql)
        return self.connection.execute(sql, *args)

    def close(self):
        assert threading.get_ident() == self.thread, (
            "connection closed by foreign thread"
        )
        self.closes += 1
        self.connection.close()
        if self.close_error:
            raise RuntimeError("injected close error")


@pytest.fixture
def counted_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_PREFER", "0")
    repo = module.open_intent_repository(tmp_path / "control.duckdb")
    repo.upsert_goal(goal_cid="goal:a", goal_alias="G", title="Goal")
    repo.upsert_plan(plan_cid="plan:a", goal_cid="goal:a", plan_alias="P")
    repo.upsert_task(
        task_cid="task:a", task_alias="A", goal_cid="goal:a", plan_cid="plan:a"
    )
    original = module.open_duckdb_connection
    connections = []
    guard = threading.Lock()

    def counted(target):
        connection = CountedConnection(original(target))
        with guard:
            connections.append(connection)
        return connection

    monkeypatch.setattr(module, "open_duckdb_connection", counted)
    try:
        yield repo, connections
    finally:
        repo.close()


def test_source_reads_and_nested_sessions_open_and_close_once(counted_repo):
    repo, connections = counted_repo
    source = DatabaseTaskSource(intent=repo, plan_root_cid="plan:a")
    with repo.read_session() as borrowed:
        assert borrowed is repo
        before = source.snapshot()
        for _ in range(3):
            assert source.get_task("A").status == "ready"
            assert source.get_plan("plan:a") is not None
            with repo.read_session():
                assert source.snapshot().to_dict() == before.to_dict()
                assert len(connections) == 1 and connections[0].closes == 0
        assert connections[0].closes == 0
    assert len(connections) == 1 and connections[0].closes == 1
    assert not any(
        sql.strip().upper().startswith(("BEGIN", "COMMIT", "ROLLBACK"))
        for sql in connections[0].statements
    )
    assert repo.is_open
    repo.get_task("A")
    assert len(connections) == 2 and connections[1].closes == 1


def test_writes_in_read_session_are_refused_before_sql(counted_repo):
    repo, connections = counted_repo
    before = repo.snapshot()
    connections.clear()
    with repo.read_session():
        count = len(connections[0].statements)
        with pytest.raises(module.IntentRepositoryError, match="writes are forbidden"):
            repo.cas_task_status(
                task_cid="A", expected_revision=1, new_status="in_progress"
            )
        assert len(connections[0].statements) == count
        assert repo.snapshot().projection_cid == before.projection_cid
    assert len(connections) == 1


class Interrupted(BaseException):
    pass


@pytest.mark.parametrize(
    "error", [RuntimeError("body error"), Interrupted("body interrupted")]
)
def test_nested_exception_closes_once_and_next_session_is_fresh(counted_repo, error):
    repo, connections = counted_repo
    with pytest.raises(type(error)):
        with repo.read_session():
            with repo.read_session():
                repo.get_task("A")
                raise error
    assert len(connections) == 1 and connections[0].closes == 1
    with repo.read_session():
        assert repo.get_task("A")["revision"] == 1
    assert len(connections) == 2 and all(c.closes == 1 for c in connections)


def test_inner_exception_can_be_caught_without_closing_outer_session(counted_repo):
    repo, connections = counted_repo
    with repo.read_session():
        with pytest.raises(RuntimeError):
            with repo.read_session():
                raise RuntimeError("inner")
        assert connections[0].closes == 0
        assert repo.get_task("A")["status"] == "ready"
    assert connections[0].closes == 1


def test_repository_close_during_session_defers_physical_close_to_owner(counted_repo):
    repo, connections = counted_repo
    with repo.read_session():
        repo.close()
        assert connections[0].closes == 0
        with pytest.raises(module.IntentRepositoryNotOpenError):
            repo.get_task("A")
        with pytest.raises(module.IntentRepositoryNotOpenError):
            with repo.read_session():
                pytest.fail("closed repository admitted a session")
    assert connections[0].closes == 1
    repo.close()
    assert connections[0].closes == 1


def test_exceptional_close_clears_session_before_next_entry(counted_repo):
    repo, connections = counted_repo
    with pytest.raises(RuntimeError, match="injected close"):
        with repo.read_session():
            connections[0].close_error = True
            repo.get_task("A")
    with repo.read_session():
        assert repo.get_task("A")["status"] == "ready"
    assert len(connections) == 2 and all(c.closes == 1 for c in connections)


def use_admitted_native_connections(repo, connections, monkeypatch):
    """Exercise independent owner-admitted adapters over actual DuckDB.

    The legacy file opener takes an exclusive lock for its whole lifetime and
    intentionally serializes clients. Quack admissions provide independent
    client connections. This controlled fixture supplies real native wrappers
    without altering either production opener's authority or fallback policy.
    """
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        DuckDBConnection,
    )

    guard = threading.Lock()

    def admitted(target):
        assert target == repo._open_target
        connection = CountedConnection(
            DuckDBConnection.wrap(duckdb.connect(str(target)))
        )
        with guard:
            connections.append(connection)
        return connection

    monkeypatch.setattr(module, "open_duckdb_connection", admitted)


def test_simultaneous_thread_sessions_never_share_connections(
    counted_repo, monkeypatch
):
    repo, connections = counted_repo
    use_admitted_native_connections(repo, connections, monkeypatch)
    together = threading.Barrier(2)

    def reader():
        with repo.read_session():
            together.wait(timeout=5)
            with repo.read_session():
                assert repo.get_task("A")["status"] == "ready"
            together.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(reader) for _ in range(2)]
        for future in futures:
            future.result(timeout=10)
    assert len(connections) == 2
    assert len({c.thread for c in connections}) == 2
    assert all(c.closes == 1 for c in connections)


def test_independent_writer_is_unchanged_and_session_does_not_freeze_snapshot(
    counted_repo, monkeypatch
):
    repo, connections = counted_repo
    use_admitted_native_connections(repo, connections, monkeypatch)
    opened, updated = threading.Event(), threading.Event()

    def reader():
        with repo.read_session():
            before = repo.get_task("A")
            opened.set()
            assert updated.wait(timeout=5)
            after = repo.get_task("A")
        return before["status"], after["status"]

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(reader)
        assert opened.wait(timeout=5)
        try:
            repo.cas_task_status(
                task_cid="A", expected_revision=1, new_status="in_progress"
            )
        finally:
            updated.set()
        assert future.result(timeout=10) == ("ready", "in_progress")
    assert len(connections) == 2 and all(c.closes == 1 for c in connections)


def test_failed_quack_admission_never_retries_or_changes_target(monkeypatch):
    target = "quack://127.0.0.1:24444"
    repo = module.IntentRepository(target, install_schema=False)
    assert repo._quack_transport is True
    attempts = []

    def denied(observed):
        attempts.append(observed)
        raise RuntimeError("native admission unavailable")

    monkeypatch.setattr(module, "open_duckdb_connection", denied)
    try:
        with pytest.raises(RuntimeError, match="native admission unavailable"):
            with repo.read_session():
                pytest.fail("denied admission entered read session")
        assert attempts == [repo._open_target]
        assert getattr(repo._read_session_state, "connection", None) is None
    finally:
        repo.close()
