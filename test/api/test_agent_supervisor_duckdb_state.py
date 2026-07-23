from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.duckdb_state import (
    initialize_duckdb_database,
    is_sqlite_database,
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.merge_queue import MergeQueue


def test_legacy_sqlite_tables_are_migrated_once_without_mutating_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.sqlite3"
    legacy = sqlite3.connect(source)
    legacy.execute(
        "CREATE TABLE items (item_id TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    legacy.execute("INSERT INTO items VALUES ('item-1', 'preserved')")
    legacy.commit()
    legacy.close()

    target = tmp_path / "state.duckdb"
    for _ in range(2):
        initialize_duckdb_database(
            target,
            schema_sql=(
                "CREATE TABLE IF NOT EXISTS items "
                "(item_id TEXT PRIMARY KEY, value TEXT NOT NULL);"
            ),
            table_names=("items",),
            legacy_sqlite_path=source,
        )

    assert is_sqlite_database(source)
    assert not is_sqlite_database(target)
    with open_duckdb_connection(target) as connection:
        rows = connection.execute(
            "SELECT item_id, value FROM items"
        ).fetchall()
        migrations = connection.execute(
            """
            SELECT COUNT(*) FROM agent_supervisor_store_metadata
            WHERE key LIKE 'sqlite_migration:%'
            """
        ).fetchone()
    assert [tuple(row[index] for index in range(2)) for row in rows] == [
        ("item-1", "preserved")
    ]
    assert migrations is not None and migrations[0] == 1


def test_merge_queue_migrates_legacy_sqlite_and_keeps_deduplication(
    tmp_path: Path,
) -> None:
    queue_dir = tmp_path / "merge-queue"
    queue_dir.mkdir()
    source = queue_dir / "merge_queue.sqlite3"
    dedupe_key = hashlib.sha256(b"task-key-1\0abc123").hexdigest()
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
            10.0, 1, '{}', 'abc123', 'task-cid-1', 'task-key-1',
            ?, 'completed', 0, '', 0, '', 11.0, 11.0
        )
        """,
        (dedupe_key,),
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
    assert duplicate.request_id == "legacy-request"
