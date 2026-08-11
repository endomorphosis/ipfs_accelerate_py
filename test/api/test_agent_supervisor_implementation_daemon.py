"""Regression coverage for implementation daemon authority selection (DQP-018).

Implied validation target for the database cutover: legacy Markdown mode still
constructs PortalImplementationDaemon, while duckdb/embedded authority selects
DatabaseImplementationDaemon without requiring JSON projections.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationDaemon,
    PortalImplementationDaemon,
    is_database_authority_mode,
    parse_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    build_portal_implementation_daemon_from_args,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)


def test_legacy_markdown_default_builds_portal_daemon(tmp_path: Path) -> None:
    todo = tmp_path / "todo.md"
    todo.write_text("# Todos\n\n## ACCEL-001 Work\n\n- Status: todo\n", encoding="utf-8")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    args = parse_args(
        [
            "--todo-path",
            str(todo),
            "--task-source-kind",
            "legacy-markdown",
            "--explicit-legacy-task-source",
            "--state-dir",
            str(state_dir),
            "--state-prefix",
            "portal",
            "--once",
        ]
    )
    # Without database path/authority, builder keeps the legacy portal daemon.
    assert not is_database_authority_mode(
        authority_mode=str(getattr(args, "authority_mode", "") or ""),
        task_source_kind=str(args.task_source_kind or ""),
    )
    daemon, context = build_portal_implementation_daemon_from_args(
        args,
        repo_root=tmp_path,
    )
    try:
        assert isinstance(daemon, PortalImplementationDaemon)
        assert context.state_path == state_dir / "portal_task_state.json"
        assert context.events_path == state_dir / "portal_events.jsonl"
    finally:
        close = getattr(daemon, "close_event_runtime", None)
        if callable(close):
            close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_database_authority_builds_database_daemon_without_projections(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "control.duckdb"
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--database-path",
            str(database_path),
            "--todo-path",
            str(tmp_path / "board.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--state-prefix",
            "dqp",
            "--once",
        ]
    )
    assert is_database_authority_mode(
        authority_mode=args.authority_mode,
        task_source_kind=args.task_source_kind,
    )
    daemon, _context = build_portal_implementation_daemon_from_args(
        args,
        repo_root=tmp_path,
    )
    try:
        assert isinstance(daemon, DatabaseImplementationDaemon)
        assert daemon.projections_required() is False
        assert daemon.state_path is None
        assert daemon.events_path is None
        assert daemon.queue_path is None
        assert daemon.pid_path is None
    finally:
        daemon.close()


def test_parse_args_database_flags_round_trip() -> None:
    args = parse_args(
        [
            "--authority-mode",
            "quack",
            "--task-source-kind",
            "duckdb",
            "--endpoint-secret-handle",
            "env://QUACK_TOKEN",
            "--state-store-id",
            "control.duckdb",
            "--state-store-generation",
            "gen-1",
            "--state-schema-revision",
            "schema-v1",
            "--state-failover-policy",
            "fail_closed",
            "--once",
        ]
    )
    assert args.authority_mode == "quack"
    assert args.task_source_kind == "duckdb"
    assert args.endpoint_secret_handle == "env://QUACK_TOKEN"
    assert args.state_store_id == "control.duckdb"
    assert args.state_store_generation == "gen-1"
    assert args.state_schema_revision == "schema-v1"
    assert args.state_failover_policy == "fail_closed"
