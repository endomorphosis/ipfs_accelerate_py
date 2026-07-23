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
from ipfs_accelerate_py.agent_supervisor.merge_resolver import MergeResolverRegistry


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
