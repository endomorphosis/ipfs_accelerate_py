from __future__ import annotations

import gc
import hashlib
import sqlite3
import threading
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.merge.merge_resolver import MergeResolverRegistry
from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    DUCKDB_ONLY_ENV,
    DuckDBConnection,
    DuckDBConnectionPolicyError,
    initialize_duckdb_database,
    is_sqlite_database,
    open_duckdb_connection,
    resolve_duckdb_path,
)


def test_connection_serializes_peer_reads_across_an_explicit_transaction(
    tmp_path: Path,
) -> None:
    """A peer thread cannot reuse one native result or enter its transaction."""

    transaction_started = threading.Event()
    allow_commit = threading.Event()
    peer_returned = threading.Event()
    failures: list[BaseException] = []
    observed_counts: list[int] = []

    with open_duckdb_connection(tmp_path / "threaded.duckdb") as connection:
        connection.execute("CREATE TABLE items (value INTEGER NOT NULL)")

        def transaction_owner() -> None:
            try:
                connection.execute("BEGIN TRANSACTION")
                connection.execute("INSERT INTO items VALUES (1)")
                transaction_started.set()
                if not allow_commit.wait(timeout=3.0):
                    raise AssertionError("transaction release was not signalled")
                connection.execute("COMMIT")
            except BaseException as exc:
                failures.append(exc)

        def peer_reader() -> None:
            try:
                if not transaction_started.wait(timeout=3.0):
                    raise AssertionError("transaction did not start")
                row = connection.execute("SELECT COUNT(*) FROM items").fetchone()
                if row is None:
                    raise AssertionError("peer query returned no count")
                observed_counts.append(int(row[0]))
                peer_returned.set()
            except BaseException as exc:
                failures.append(exc)

        owner = threading.Thread(target=transaction_owner)
        peer = threading.Thread(target=peer_reader)
        owner.start()
        assert transaction_started.wait(timeout=3.0)
        peer.start()
        try:
            assert not peer_returned.wait(timeout=0.1)
        finally:
            allow_commit.set()
        owner.join(timeout=3.0)
        peer.join(timeout=3.0)

        assert not owner.is_alive()
        assert not peer.is_alive()
        assert failures == []
        assert observed_counts == [1]


def test_nested_begin_preserves_outer_transaction_and_peer_exclusion(
    tmp_path: Path,
) -> None:
    peer_returned = threading.Event()
    peer_failures: list[BaseException] = []
    peer_counts: list[int] = []

    with open_duckdb_connection(tmp_path / "nested.duckdb") as connection:
        connection.execute("CREATE TABLE items (value INTEGER NOT NULL)")
        connection.execute("BEGIN TRANSACTION")
        connection.execute("INSERT INTO items VALUES (1)")

        with pytest.raises(DuckDBConnectionPolicyError, match="already active"):
            connection.execute("BEGIN TRANSACTION")
        assert connection.in_transaction is True
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1

        observed_state: list[bool] = []
        observer = threading.Thread(
            target=lambda: observed_state.append(connection.in_transaction)
        )
        observer.start()
        observer.join(timeout=1.0)
        assert not observer.is_alive()
        assert observed_state == [True]

        def peer_reader() -> None:
            try:
                row = connection.execute("SELECT COUNT(*) FROM items").fetchone()
                if row is None:
                    raise AssertionError("peer query returned no count")
                peer_counts.append(int(row[0]))
                peer_returned.set()
            except BaseException as exc:
                peer_failures.append(exc)

        peer = threading.Thread(target=peer_reader)
        peer.start()
        try:
            assert not peer_returned.wait(timeout=0.1)
        finally:
            connection.rollback()
        peer.join(timeout=3.0)

        assert not peer.is_alive()
        assert peer_failures == []
        assert peer_counts == [0]


def test_peer_transaction_termination_is_rejected_without_waiting(
    tmp_path: Path,
) -> None:
    with open_duckdb_connection(tmp_path / "peer-terminal.duckdb") as connection:
        connection.execute("CREATE TABLE items (value INTEGER NOT NULL)")
        connection.execute("BEGIN TRANSACTION")
        connection.execute("INSERT INTO items VALUES (1)")
        failures: list[BaseException] = []
        operations = (
            lambda: connection.execute("COMMIT"),
            lambda: connection.execute("ROLLBACK"),
            connection.commit,
            connection.rollback,
        )

        for operation in operations:
            peer = threading.Thread(
                target=lambda action=operation: _capture_failure(action, failures)
            )
            peer.start()
            peer.join(timeout=1.0)
            assert not peer.is_alive()

        assert len(failures) == 4
        assert all(
            isinstance(exc, DuckDBConnectionPolicyError) for exc in failures
        )
        assert connection.in_transaction is True
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1

        observed_counts: list[int] = []
        reader = threading.Thread(
            target=lambda: observed_counts.append(
                int(connection.execute("SELECT COUNT(*) FROM items").fetchone()[0])
            )
        )
        reader.start()
        assert reader.is_alive()
        connection.rollback()
        reader.join(timeout=3.0)

        assert not reader.is_alive()
        assert observed_counts == [0]


def _capture_failure(action, failures: list[BaseException]) -> None:
    try:
        action()
    except BaseException as exc:
        failures.append(exc)


def test_transaction_control_normalizes_boundaries_and_rejects_scripts(
    tmp_path: Path,
) -> None:
    with open_duckdb_connection(tmp_path / "boundaries.duckdb") as connection:
        connection.execute("CREATE TABLE items (value INTEGER NOT NULL)")
        connection.execute("/* lead */ BEGIN IMMEDIATE TRANSACTION; -- tail")
        connection.execute("INSERT INTO items VALUES (1)")
        connection.execute("COMMIT TRANSACTION; /* tail */")

        assert connection.in_transaction is False
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1
        connection.execute("/* outer /* inner */ still */ BEGIN;")
        connection.execute("INSERT INTO items VALUES (2)")
        connection.execute("-- é\rEND WORK;")
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 2
        connection.execute("/* 💩 */ BEGIN TRANSACTION;")
        connection.execute("INSERT INTO items VALUES (3)")
        connection.execute("ABORT TRANSACTION;")
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 2
        with pytest.raises(DuckDBConnectionPolicyError, match="standalone"):
            connection.execute("BEGIN TRANSACTION; SELECT 1")
        with pytest.raises(DuckDBConnectionPolicyError, match="standalone"):
            connection.execute("BEGIN; -- x\r COMMIT;")
        with pytest.raises(DuckDBConnectionPolicyError, match="transaction control"):
            connection.executescript("BEGIN; INSERT INTO items VALUES (2); COMMIT;")
        with pytest.raises(DuckDBConnectionPolicyError, match="transaction control"):
            connection.executemany("BEGIN", [[]])
        assert connection.execute("SELECT $$BEGIN; COMMIT$$").fetchone()[0] == (
            "BEGIN; COMMIT"
        )
        assert connection.in_transaction is False


def test_cross_thread_context_entry_is_rejected_without_poisoning_owner() -> None:
    import duckdb

    raw = duckdb.connect(":memory:")
    connection = DuckDBConnection.wrap(raw, transaction_on_context=True)
    connection.execute("CREATE TABLE items (value INTEGER NOT NULL)")
    entered = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []

    def context_owner() -> None:
        try:
            with connection:
                connection.execute("INSERT INTO items VALUES (1)")
                entered.set()
                if not release.wait(timeout=3.0):
                    raise AssertionError("context release was not signalled")
        except BaseException as exc:
            failures.append(exc)

    owner = threading.Thread(target=context_owner)
    owner.start()
    assert entered.wait(timeout=3.0)
    with pytest.raises(DuckDBConnectionPolicyError, match="another thread"):
        connection.__enter__()
    release.set()
    owner.join(timeout=3.0)

    assert not owner.is_alive()
    assert failures == []
    with connection:
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1
    connection.close()


def test_context_owner_remains_reserved_during_native_close() -> None:
    close_started = threading.Event()
    allow_close = threading.Event()
    peer_started = threading.Event()
    peer_returned = threading.Event()

    class BlockingClose:
        description = ()

        def execute(self, _sql, _parameters=None):
            return self

        def fetchall(self):
            return []

        def rollback(self) -> None:
            return None

        def close(self) -> None:
            close_started.set()
            if not allow_close.wait(timeout=3.0):
                raise AssertionError("native close release was not signalled")

    class FakeLockContext:
        def __exit__(self, _exc_type, _exc, _traceback) -> None:
            return None

    connection = DuckDBConnection.wrap(
        BlockingClose(),
        transaction_on_context=True,
    )
    connection._lock_context = FakeLockContext()
    failures: list[BaseException] = []
    peer_failures: list[BaseException] = []

    def context_owner() -> None:
        try:
            with connection:
                pass
        except BaseException as exc:
            failures.append(exc)

    owner = threading.Thread(target=context_owner)
    owner.start()
    assert close_started.wait(timeout=3.0)
    with pytest.raises(DuckDBConnectionPolicyError, match="unusable|another thread"):
        connection.__enter__()

    def peer_query() -> None:
        peer_started.set()
        try:
            connection.execute("SELECT 1")
        except BaseException as exc:
            peer_failures.append(exc)
        finally:
            peer_returned.set()

    peer = threading.Thread(target=peer_query)
    peer.start()
    assert peer_started.wait(timeout=3.0)
    assert not peer_returned.wait(timeout=0.1)
    connection._discard_pooled_connection()
    assert peer_returned.wait(timeout=1.0)
    allow_close.set()
    owner.join(timeout=3.0)
    peer.join(timeout=3.0)

    assert not owner.is_alive()
    assert not peer.is_alive()
    assert failures == []
    assert len(peer_failures) == 1
    assert isinstance(peer_failures[0], DuckDBConnectionPolicyError)


def test_same_raw_connection_cannot_receive_independent_wrapper_locks() -> None:
    import duckdb

    raw = duckdb.connect(":memory:")
    connection = DuckDBConnection.wrap(raw)
    try:
        with pytest.raises(DuckDBConnectionPolicyError, match="already owned"):
            DuckDBConnection.wrap(raw)
    finally:
        connection.close()


def test_abandoned_wrapper_closes_raw_before_registry_release() -> None:
    class ClosableRaw:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    raw = ClosableRaw()
    connection = DuckDBConnection.wrap(raw)
    key = connection._raw_wrapper_key
    del connection
    gc.collect()

    assert raw.closed is True
    assert key not in duckdb_state._RAW_WRAPPERS


def test_close_finishes_native_cleanup_after_rollback_failure() -> None:
    class FailingRollback:
        description = ()

        def __init__(self) -> None:
            self.closed = False

        def execute(self, _sql, _parameters=None):
            return self

        def fetchall(self):
            return []

        def rollback(self) -> None:
            raise RuntimeError("rollback failed")

        def close(self) -> None:
            self.closed = True

    raw = FailingRollback()
    connection = DuckDBConnection.wrap(raw)
    connection.execute("BEGIN TRANSACTION")
    connection.close()

    assert raw.closed is True
    assert connection.in_transaction is False
    with pytest.raises(DuckDBConnectionPolicyError, match="unusable"):
        connection.execute("SELECT 1")


def test_exclusive_file_connection_recovers_after_uncertain_transaction(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recover.duckdb"
    with open_duckdb_connection(path) as connection:
        connection.execute("CREATE TABLE items (value INTEGER NOT NULL)")
        connection.execute("INSERT INTO items VALUES (1)")
        with connection._execution_condition:
            connection._poison_locked()
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1
        connection.execute("INSERT INTO items VALUES (2)")
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 2


def test_art_index_delete_fatal_is_detected() -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        is_art_index_delete_fatal,
    )

    class FatalException(Exception):
        pass

    assert is_art_index_delete_fatal(
        FatalException(
            "FATAL Error: Invalid Input Error: Failed to delete all rows "
            "from index. Only deleted 0 out of 1 rows."
        )
    )
    assert not is_art_index_delete_fatal(FatalException("native handle died"))
    assert not is_art_index_delete_fatal(
        RuntimeError("Failed to delete all rows from index")
    )


def test_rebuild_task_status_indexes_recreates_status_indexes(tmp_path: Path) -> None:
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        rebuild_task_status_indexes,
    )

    connection = duckdb.connect(str(tmp_path / "control.duckdb"))
    connection.execute(
        """
        CREATE TABLE tasks (
            task_cid VARCHAR PRIMARY KEY,
            task_alias VARCHAR NOT NULL UNIQUE,
            goal_cid VARCHAR NOT NULL,
            ordinal BIGINT NOT NULL,
            status VARCHAR NOT NULL,
            revision BIGINT NOT NULL,
            updated_at VARCHAR NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX tasks_goal_idx ON tasks(goal_cid, status)")
    connection.execute("CREATE INDEX tasks_status_idx ON tasks(status, ordinal)")
    connection.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
        ["cid-010", "PCSM-010", "goal:root", 6, "retrying", 9, "2026-08-27T05:08:42Z"],
    )
    rebuilt = rebuild_task_status_indexes(connection)
    assert any("tasks_status_idx" in sql for sql in rebuilt)
    assert any("tasks_goal_idx" in sql for sql in rebuilt)
    names = {
        str(row[0])
        for row in connection.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'tasks'"
        ).fetchall()
    }
    assert "tasks_status_idx" in names
    assert "tasks_goal_idx" in names
    connection.execute(
        "UPDATE tasks SET status = ?, revision = ? WHERE task_cid = ? AND revision = ?",
        ["in_progress", 10, "cid-010", 9],
    )
    row = connection.execute(
        "SELECT status, revision FROM tasks WHERE task_cid = 'cid-010'"
    ).fetchone()
    assert tuple(row) == ("in_progress", 10)


def test_exclusive_file_connection_recovers_after_native_fatal_dml(
    tmp_path: Path,
) -> None:
    class FatalException(Exception):
        pass

    class FatalOnce:
        def __init__(self, inner: object) -> None:
            self._inner = inner
            self._fatal_inserts = 1

        def execute(self, sql, parameters=None):
            if self._fatal_inserts and "INSERT" in str(sql).upper():
                self._fatal_inserts -= 1
                raise FatalException("native handle died")
            if parameters is None:
                return self._inner.execute(sql)
            return self._inner.execute(sql, parameters)

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    path = tmp_path / "fatal-dml.duckdb"
    with open_duckdb_connection(path) as connection:
        connection.execute("CREATE TABLE items (value INTEGER NOT NULL)")
        connection._connection = FatalOnce(connection._connection)
        with pytest.raises(FatalException):
            connection.execute("INSERT INTO items VALUES (1)")
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 0
        connection.execute("INSERT INTO items VALUES (1)")
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1


def test_wrapped_memory_connection_stays_unusable_after_poison() -> None:
    import duckdb

    raw = duckdb.connect(":memory:")
    connection = DuckDBConnection.wrap(raw)
    try:
        connection.execute("CREATE TABLE items (value INTEGER NOT NULL)")
        with connection._execution_condition:
            connection._poison_locked()
        with pytest.raises(DuckDBConnectionPolicyError, match="unusable"):
            connection.execute("SELECT 1")
    finally:
        connection.close()


def test_legacy_sqlite_tables_are_migrated_once_without_mutating_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.sqlite3"
    legacy = sqlite3.connect(source)
    legacy.execute("CREATE TABLE items (item_id TEXT PRIMARY KEY, value TEXT NOT NULL)")
    legacy.execute("INSERT INTO items VALUES ('item-1', 'preserved')")
    legacy.commit()
    legacy.close()

    target = tmp_path / "state.duckdb"
    for _ in range(2):
        initialize_duckdb_database(
            target,
            schema_sql=(
                "CREATE TABLE IF NOT EXISTS items (item_id TEXT PRIMARY KEY, value TEXT NOT NULL);"
            ),
            table_names=("items",),
            legacy_sqlite_path=source,
        )

    assert is_sqlite_database(source)
    assert not is_sqlite_database(target)
    with open_duckdb_connection(target) as connection:
        rows = connection.execute("SELECT item_id, value FROM items").fetchall()
        migrations = connection.execute(
            """
            SELECT COUNT(*) FROM agent_supervisor_store_metadata
            WHERE key LIKE 'sqlite_migration:%'
            """
        ).fetchone()
    assert [tuple(row[index] for index in range(2)) for row in rows] == [("item-1", "preserved")]
    assert migrations is not None and migrations[0] == 1


def test_strict_duckdb_only_mode_never_probes_or_migrates_sqlite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    legacy = tmp_path / "state.sqlite3"
    legacy.write_bytes(b"SQLite format 3\0legacy bytes must remain unread")
    monkeypatch.setenv(DUCKDB_ONLY_ENV, "true")

    original_probe = duckdb_state.is_sqlite_database

    def reject_sqlite_probe(path: Path | str) -> bool:
        if Path(path) == legacy:
            raise AssertionError("strict DuckDB-only mode probed a legacy SQLite file")
        return original_probe(path)

    monkeypatch.setattr(duckdb_state, "is_sqlite_database", reject_sqlite_probe)

    target, source = resolve_duckdb_path(
        legacy,
        default_filename="state.duckdb",
        temporary_prefix="strict-duckdb-",
    )
    assert target == legacy.with_suffix(".duckdb")
    assert source is None

    initialize_duckdb_database(
        target,
        schema_sql=(
            "CREATE TABLE IF NOT EXISTS items (item_id TEXT PRIMARY KEY, value TEXT NOT NULL);"
        ),
        table_names=("items",),
        legacy_sqlite_path=legacy,
    )

    assert legacy.read_bytes() == b"SQLite format 3\0legacy bytes must remain unread"
    with open_duckdb_connection(target) as connection:
        rows = connection.execute("SELECT item_id, value FROM items").fetchall()
        migrations = connection.execute(
            """
            SELECT COUNT(*) FROM agent_supervisor_store_metadata
            WHERE key LIKE 'sqlite_migration:%'
            """
        ).fetchone()
    assert rows == []
    assert migrations is not None and migrations[0] == 0


def test_merge_queue_inherits_strict_duckdb_only_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    queue_dir = tmp_path / "merge-queue"
    queue_dir.mkdir()
    legacy = queue_dir / "merge_queue.sqlite3"
    source = sqlite3.connect(legacy)
    source.execute("CREATE TABLE legacy_sentinel (value TEXT NOT NULL)")
    source.execute("INSERT INTO legacy_sentinel VALUES ('do-not-read')")
    source.commit()
    source.close()
    monkeypatch.setenv(DUCKDB_ONLY_ENV, "1")

    original_probe = duckdb_state.is_sqlite_database

    def reject_legacy_probe(path: Path | str) -> bool:
        if Path(path) == legacy:
            raise AssertionError("MergeQueue probed its legacy SQLite sibling")
        return original_probe(path)

    monkeypatch.setattr(duckdb_state, "is_sqlite_database", reject_legacy_probe)

    queue = MergeQueue(queue_dir)

    assert queue.database_path == queue_dir / "merge_queue.duckdb"
    assert queue.pending_count() == 0
    assert queue.processing_count() == 0


def test_merge_queue_migrates_legacy_sqlite_and_keeps_deduplication(
    tmp_path: Path,
) -> None:
    queue_dir = tmp_path / "merge-queue"
    queue_dir.mkdir()
    source = queue_dir / "merge_queue.sqlite3"
    dedupe_key = hashlib.sha256(b"task-key-1\0abc123").hexdigest()
    enqueued_at = 1_784_690_172.5146759
    finished_at = 1_784_690_175.7506707
    legacy = sqlite3.connect(source)
    legacy.executescript(
        """
        CREATE TABLE merge_requests (
            request_id TEXT PRIMARY KEY,
            branch_name TEXT NOT NULL,
            task_id TEXT NOT NULL,
            priority TEXT NOT NULL,
            lane_id TEXT NOT NULL,
            enqueued_at REAL NOT NULL,
            attempt INTEGER NOT NULL,
            metadata_json TEXT NOT NULL,
            commit_sha TEXT NOT NULL,
            canonical_task_id TEXT NOT NULL,
            canonical_task_key TEXT NOT NULL,
            dedupe_key TEXT NOT NULL,
            status TEXT NOT NULL,
            claimed_at REAL NOT NULL DEFAULT 0,
            consumer_id TEXT NOT NULL DEFAULT '',
            failure_count INTEGER NOT NULL DEFAULT 0,
            failure_reason TEXT NOT NULL DEFAULT '',
            finished_at REAL NOT NULL DEFAULT 0,
            updated_at REAL NOT NULL
        );
        """
    )
    legacy.execute(
        """
        INSERT INTO merge_requests VALUES (
            'legacy-request', 'implementation/ref-1', 'REF-1', 'P1', 'lane-a',
            ?, 1, '{}', 'abc123', 'task-cid-1', 'task-key-1',
            ?, 'completed', 0, '', 0, '', ?, ?
        )
        """,
        (enqueued_at, dedupe_key, finished_at, finished_at),
    )
    legacy.commit()
    legacy.close()

    queue = MergeQueue(queue_dir)
    migrated = queue.get("legacy-request")
    duplicate = queue.enqueue(
        branch_name="implementation/ref-1-retry",
        task_id="REF-1",
        priority="P0",
        commit_sha="abc123",
        canonical_task_key="task-key-1",
    )

    assert queue.database_path.name == "merge_queue.duckdb"
    assert is_sqlite_database(source)
    assert migrated is not None and migrated.status == "completed"
    assert migrated.enqueued_at == enqueued_at
    assert duplicate.request_id == "legacy-request"


def test_merge_resolver_migration_preserves_epoch_precision(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "merge-resolver"
    state_dir.mkdir()
    source = state_dir / "merge_resolver.sqlite3"
    acquired_at = 1_784_698_207.9501562
    lease_expires_at = 1_784_699_107.1251562
    updated_at = 1_784_698_426.730546
    legacy = sqlite3.connect(source)
    legacy.executescript(
        """
        CREATE TABLE conflict_resolutions (
            fingerprint TEXT PRIMARY KEY,
            state TEXT NOT NULL,
            owner_id TEXT NOT NULL DEFAULT '',
            token TEXT NOT NULL DEFAULT '',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            acquired_at REAL NOT NULL DEFAULT 0,
            lease_expires_at REAL NOT NULL DEFAULT 0,
            updated_at REAL NOT NULL,
            last_error TEXT NOT NULL DEFAULT '',
            event_json TEXT NOT NULL DEFAULT '{}',
            outcome_json TEXT NOT NULL DEFAULT '{}',
            receipt_path TEXT NOT NULL DEFAULT ''
        );
        """
    )
    legacy.execute(
        """
        INSERT INTO conflict_resolutions VALUES (
            'conflict-1', 'failed', 'resolver-1', 'token-1', 1,
            ?, ?, ?, 'merge failed', '{}', '{}', ''
        )
        """,
        (acquired_at, lease_expires_at, updated_at),
    )
    legacy.commit()
    legacy.close()

    registry = MergeResolverRegistry(state_dir)
    migrated = registry.status("conflict-1")

    assert registry.database_path.name == "merge_resolver.duckdb"
    assert is_sqlite_database(source)
    assert migrated["acquired_at"] == acquired_at
    assert migrated["lease_expires_at"] == lease_expires_at
