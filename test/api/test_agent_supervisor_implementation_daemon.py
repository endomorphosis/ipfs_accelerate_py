"""Regression coverage for implementation daemon authority selection (DQP-018).

Implied validation target for the database cutover: legacy Markdown mode still
constructs PortalImplementationDaemon, while duckdb/embedded authority selects
DatabaseImplementationDaemon without requiring JSON projections.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
    GROK_CODEX_EXECUTION_MODE,
    TaskExecutionRouteBinding,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationDaemon,
    PortalImplementationDaemon,
    PortalTask,
    is_database_authority_mode,
    parse_args,
    parse_task_text,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    build_portal_implementation_daemon_from_args,
)


def _route_binding(*, task_alias: str, task_cid: str) -> dict[str, object]:
    return TaskExecutionRouteBinding(
        policy_id="policy:test-portal-projection-route",
        plan_root_cid="plan:test-portal-projection-route",
        repository_tree_id="tree:test-portal-projection-route",
        source_revision=1,
        task_cid=task_cid,
        task_alias=task_alias,
        task_revision=1,
        task_contract_cid="contract:test-portal-projection-route",
        execution_mode=GROK_CODEX_EXECUTION_MODE,
    ).to_dict()


def _database_projection_task(
    tmp_path: Path,
    *,
    database_task_cid: str,
    canonical_task_cid: str | None = None,
    projection_authority: str = "false",
) -> PortalTask:
    tasks = parse_task_text(
        "\n".join(
            (
                "# Database attempt projection (non-authoritative)",
                "",
                "## ROUTE-001 Execute the admitted task",
                "",
                "- Status: ready",
                "- Completion: auto",
                "- Projection authority: " + projection_authority,
                "- Database task CID: " + database_task_cid,
                "- Canonical task CID: "
                + (canonical_task_cid or database_task_cid),
                "",
            )
        ),
        path=tmp_path / "database-attempt-projection.md",
        task_header_prefix="## ROUTE-",
    )
    assert len(tasks) == 1
    return tasks[0]


def test_launch_route_uses_authoritative_cid_from_database_projection(
    tmp_path: Path,
) -> None:
    database_task_cid = "task:canonical-database-task"
    task = _database_projection_task(
        tmp_path,
        database_task_cid=database_task_cid,
    )
    assert task.canonical_task_cid != database_task_cid
    daemon = object.__new__(PortalImplementationDaemon)
    daemon._launch_task_execution_route_binding = None

    daemon.bind_launch_task_execution_route(
        _route_binding(task_alias=task.task_id, task_cid=database_task_cid)
    )

    assert (
        daemon._launch_execution_mode_for_task(task)
        == GROK_CODEX_EXECUTION_MODE
    )


@pytest.mark.parametrize(
    ("canonical_task_cid", "projection_authority"),
    (
        ("task:forged-canonical-task", "false"),
        (None, "true"),
    ),
)
def test_launch_route_rejects_invalid_database_projection_authority(
    tmp_path: Path,
    canonical_task_cid: str | None,
    projection_authority: str,
) -> None:
    database_task_cid = "task:canonical-database-task"
    task = _database_projection_task(
        tmp_path,
        database_task_cid=database_task_cid,
        canonical_task_cid=canonical_task_cid,
        projection_authority=projection_authority,
    )
    daemon = object.__new__(PortalImplementationDaemon)
    daemon._launch_task_execution_route_binding = None
    daemon.bind_launch_task_execution_route(
        _route_binding(task_alias=task.task_id, task_cid=database_task_cid)
    )

    with pytest.raises(RuntimeError, match="invalid launch task authority"):
        daemon._launch_execution_mode_for_task(task)


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
