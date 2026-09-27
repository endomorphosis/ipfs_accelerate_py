"""Upgrade an existing intent catalog without changing task or migration history."""
from pathlib import Path

import duckdb

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    ControlPlaneMigrationRunner,
    load_default_catalog,
)


def _runner(database: Path, owner: str):
    return ControlPlaneMigrationRunner.for_database(
        database,
        catalog=load_default_catalog(),
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id=owner,
    )


def _records(database: Path):
    with duckdb.connect(str(database), read_only=True) as connection:
        return {
            "migrations": connection.execute(
                "SELECT * FROM schema_migrations ORDER BY version"
            ).fetchall(),
            "tasks": connection.execute(
                "SELECT * FROM tasks ORDER BY task_cid"
            ).fetchall(),
            "attempts": connection.execute(
                "SELECT * FROM schema_migration_attempts ORDER BY version, attempt_id"
            ).fetchall(),
        }


def test_v4_upgrade_preserves_rows_and_receipts_and_reapplies_without_writes(tmp_path):
    database = tmp_path / "existing-control.duckdb"
    original = _runner(database, "index-upgrade-before")
    before_report = original.apply(target_version=4)
    assert before_report.to_version == 4
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            "INSERT INTO tasks (task_cid, task_alias, goal_cid, plan_cid, "
            "objective_id, ordinal, status, revision, priority, created_at, "
            "updated_at, identity_json, body_json, extension_schema, extension_json) "
            "VALUES ('task:preserve', 'AFTD-preserve', 'goal:preserve', '', '', "
            "7, 'blocked', 3, 'P1', 'before', 'before', '{}', "
            "'{\"completion_receipt\":{\"admitted\":false,\"formalized\":false}}', '', '{}')"
        )
        before_indexes = dict(connection.execute(
            "SELECT index_name, expressions FROM duckdb_indexes() "
            "WHERE table_name = 'tasks'"
        ).fetchall())
    assert "tasks_status_idx" in before_indexes
    before = _records(database)

    upgraded = _runner(database, "index-upgrade-after").apply(target_version=5)
    assert upgraded.changed is True
    assert upgraded.from_version == 4 and upgraded.to_version == 5
    assert [receipt.version for receipt in upgraded.receipts] == [5]
    after = _records(database)
    assert after["tasks"] == before["tasks"]
    assert after["migrations"][:4] == before["migrations"]
    assert after["attempts"][:4] == before["attempts"]
    assert len(after["migrations"]) == len(after["attempts"]) == 5

    with duckdb.connect(str(database)) as connection:
        after_indexes = dict(connection.execute(
            "SELECT index_name, expressions FROM duckdb_indexes() "
            "WHERE table_name = 'tasks'"
        ).fetchall())
        assert "tasks_status_idx" not in after_indexes
        assert "tasks_goal_idx" not in after_indexes
        assert "status" not in after_indexes["tasks_goal_replacement_idx"]
        assert "goal_cid" in after_indexes["tasks_goal_replacement_idx"]
        assert "ordinal" in after_indexes["tasks_ordinal_idx"]
        connection.execute(
            "UPDATE tasks SET status = 'ready', revision = revision + 1 "
            "WHERE task_cid = 'task:preserve' AND status = 'blocked' AND revision = 3"
        )
        assert connection.execute(
            "SELECT status, revision, body_json FROM tasks WHERE task_cid = 'task:preserve'"
        ).fetchone() == (
            "ready", 4,
            '{"completion_receipt":{"admitted":false,"formalized":false}}',
        )

    settled = _records(database)
    replay = _runner(database, "index-upgrade-reopen").apply(target_version=5)
    assert replay.changed is False
    assert replay.from_version == replay.to_version == 5
    assert tuple(receipt.version for receipt in replay.receipts) == (1, 2, 3, 4, 5)
    assert tuple(receipt.receipt_cid for receipt in replay.receipts) == tuple(
        receipt.receipt_cid
        for receipt in (*before_report.receipts, *upgraded.receipts)
    )
    assert _records(database) == settled
