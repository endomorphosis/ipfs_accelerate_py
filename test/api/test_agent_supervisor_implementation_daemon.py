"""Regression coverage for implementation daemon database cutover (DQP-018).

Implied validation target: ensure PortalImplementationDaemon and its runner
preserve legacy defaults while accepting DatabaseProgramConfig@1 authority
fields without demoting Quack or writing Markdown under database authority.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    AUTHORITY_MODE_EMBEDDED,
    AUTHORITY_MODE_QUACK,
    DatabaseProgramConfig,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    database_authority_mode_active,
    database_program_from_daemon_namespace,
    open_database_implementation_daemon,
    parse_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    apply_database_program_defaults,
    implementation_state_paths,
)


def test_parse_args_default_remains_legacy_markdown_without_authority() -> None:
    args = parse_args(["--once"])
    assert args.task_source_kind == "legacy-markdown"
    assert args.authority_mode == ""
    assert database_program_from_daemon_namespace(args) is None


def test_parse_args_accepts_full_database_program_cli() -> None:
    program = DatabaseProgramConfig(
        authority_mode=AUTHORITY_MODE_QUACK,
        task_source_kind="duckdb",
        endpoint_secret_handle="env://QUACK_TOKEN",
        store_id="control.duckdb",
        store_generation="gen-9",
        schema_revision="schema-v2",
        event_store_path="state/events",
        runtime_registry_path="state/registry",
        export_profile="operator-export",
        failover_policy="fail_closed",
    )
    args = parse_args(
        [
            "--todo-path",
            "state/control.duckdb",
            *program.cli_args(),
        ]
    )
    assert args.authority_mode == AUTHORITY_MODE_QUACK
    assert args.task_source_kind == "duckdb"
    assert args.endpoint_secret_handle == "env://QUACK_TOKEN"
    assert args.state_store_id == "control.duckdb"
    assert args.state_store_generation == "gen-9"
    assert args.state_schema_revision == "schema-v2"
    assert args.event_store_path == "state/events"
    assert args.runtime_registry_path == "state/registry"
    assert args.export_profile == "operator-export"
    assert args.state_failover_policy == "fail_closed"
    restored = database_program_from_daemon_namespace(args)
    assert restored is not None
    assert restored.to_dict() == program.to_dict()


def test_apply_database_program_defaults_is_idempotent() -> None:
    program = DatabaseProgramConfig(
        authority_mode=AUTHORITY_MODE_EMBEDDED,
        task_source_kind="duckdb",
        store_id="control.duckdb",
        store_generation="gen-1",
        schema_revision="schema-v1",
    )
    first = apply_database_program_defaults([], database_program=program)
    second = apply_database_program_defaults(first, database_program=program)
    assert first.count("--authority-mode") == 1
    assert second.count("--authority-mode") == 1
    assert second.count("--task-source-kind") == 1


@pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for database-authority portal daemon tests",
)
def test_database_authority_portal_daemon_skips_markdown_checkout_writes(
    tmp_path: Path,
) -> None:
    board = tmp_path / "board.md"
    original = "## DQP-018 Cutover\n\n- Status: todo\n"
    board.write_text(original, encoding="utf-8")
    program = DatabaseProgramConfig(
        authority_mode=AUTHORITY_MODE_EMBEDDED,
        task_source_kind="duckdb",
        store_id="control.duckdb",
        store_generation="gen-1",
        schema_revision="schema-v1",
    )
    db = open_database_implementation_daemon(
        tmp_path / "coordination.duckdb",
        database_program=program,
        markdown_board_path=board,
    )
    try:
        state_dir = tmp_path / "state"
        daemon = PortalImplementationDaemon(
            todo_path=board,
            state_path=state_dir / "portal_task_state.json",
            strategy_path=state_dir / "portal_strategy.json",
            events_path=state_dir / "portal_events.jsonl",
            repo_root=tmp_path,
            database_program=program,
            database_implementation=db,
        )
        assert database_authority_mode_active(daemon.database_program.authority_mode)
        assert daemon.database_authority_active is True
        assert daemon._task_source_writes_markdown_checkout() is False
        # Missing projections are acceptable under database authority.
        paths = implementation_state_paths(
            parse_args(
                [
                    "--state-dir",
                    str(state_dir),
                    "--state-prefix",
                    "portal",
                ]
            )
        )
        assert not paths["state_path"].exists()
        assert not paths["events_path"].exists()
        result = daemon._mark_tasks_completed_in_todo(
            ["DQP-018"],
            primary_task_id="DQP-018",
            completion_reason="database_cutover",
        )
        assert result["updated"] is True
        assert result.get("writes_markdown_checkout") is False
        assert board.read_text(encoding="utf-8") == original
    finally:
        db.close()
