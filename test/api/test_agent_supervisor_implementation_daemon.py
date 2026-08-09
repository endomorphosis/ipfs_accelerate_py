"""Regression coverage for implementation daemon authority cutover (DQP-018).

Implied validation target for the database-authoritative daemon cutover.
Keeps legacy Markdown construction working while asserting database-authority
guards and CLI/runner propagation.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    AUTHORITY_MODE_EMBEDDED,
    AUTHORITY_MODE_LEGACY_MARKDOWN,
    AUTHORITY_MODE_QUACK,
    DATABASE_PROGRAM_JSON_ENV,
    DatabaseProgramConfig,
    DatabaseProgramConfigError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATABASE_IMPLEMENTATION_DAEMON_INTERFACE,
    PortalImplementationDaemon,
    database_program_from_cli_namespace,
    is_database_authority_mode,
    parse_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    build_portal_implementation_daemon_from_args,
    implementation_state_paths,
)


def _git_init(repo: Path) -> None:
    import subprocess

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "checkout", "-b", "main"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _git_commit_all(repo: Path, message: str = "seed") -> None:
    import subprocess

    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", message, "--allow-empty"],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_legacy_markdown_daemon_constructs_without_database_program(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_init(repo)
    todo = repo / "todo.md"
    todo.write_text(
        """# Todos

## PORTAL-001 Example

- Status: todo
- Priority: P1
- Track: ops
""",
        encoding="utf-8",
    )
    _git_commit_all(repo)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    daemon = PortalImplementationDaemon(
        todo_path=todo,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## PORTAL-",
    )
    assert daemon.database_program is None
    assert daemon.database_authority_active is False
    assert daemon.database_implementation_daemon() is None
    tasks = daemon._load_tasks()
    assert any(task.task_id == "PORTAL-001" for task in tasks)


def test_database_authority_active_when_program_is_database(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_init(repo)
    todo = repo / "todo.md"
    todo.write_text("# unused\n", encoding="utf-8")
    _git_commit_all(repo)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    program = DatabaseProgramConfig.from_mapping(
        {
            "authority_mode": AUTHORITY_MODE_EMBEDDED,
            "task_source_kind": "duckdb",
            "store_id": "control.duckdb",
            "failover_policy": "fail_closed",
        }
    )
    daemon = PortalImplementationDaemon(
        todo_path=todo,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        database_program=program,
    )
    assert daemon.database_authority_active is True
    assert daemon._task_source_writes_markdown_checkout() is False


def test_database_authority_blocks_markdown_completion_mutation(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_init(repo)
    todo = repo / "todo.md"
    original = """# Todos

## PORTAL-010 Example

- Status: todo
- Priority: P0
"""
    todo.write_text(original, encoding="utf-8")
    _git_commit_all(repo)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    program = DatabaseProgramConfig.from_mapping(
        {
            "authority_mode": AUTHORITY_MODE_EMBEDDED,
            "task_source_kind": "duckdb",
            "store_id": "control.duckdb",
            "failover_policy": "fail_closed",
        }
    )
    daemon = PortalImplementationDaemon(
        todo_path=todo,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        database_program=program,
    )
    result = daemon._mark_tasks_completed_in_todo_unchecked(
        ["PORTAL-010"],
        primary_task_id="PORTAL-010",
        completion_reason="unit-test",
    )
    assert result["updated"] is False
    assert result["reason"] == (
        "database_authority_forbids_markdown_status_update"
    )
    assert todo.read_text(encoding="utf-8") == original


def test_parse_args_defaults_remain_legacy_markdown() -> None:
    args = parse_args(["--once"])
    assert args.task_source_kind == "legacy-markdown"
    assert args.authority_mode == ""
    program = database_program_from_cli_namespace(args)
    # Implicit legacy default without explicit authority flags remains None
    # so construction stays backward compatible.
    assert program is None


def test_database_program_from_env_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = DatabaseProgramConfig.from_mapping(
        {
            "authority_mode": AUTHORITY_MODE_QUACK,
            "task_source_kind": "duckdb",
            "endpoint_secret_handle": "env://QUACK_TOKEN",
            "store_id": "control.duckdb",
            "store_generation": "gen-9",
            "schema_revision": "schema-v1",
            "failover_policy": "fail_closed",
        }
    )
    monkeypatch.setenv(
        DATABASE_PROGRAM_JSON_ENV,
        json.dumps(program.to_dict(), sort_keys=True),
    )
    args = SimpleNamespace(
        authority_mode="",
        task_source_kind="",
        explicit_legacy_task_source=False,
        endpoint_secret_handle="",
        state_store_id="",
        state_store_generation="",
        state_schema_revision="",
        event_store_path="",
        runtime_registry_path="",
        worktree_root="",
        export_profile="",
        state_failover_policy="",
    )
    restored = database_program_from_cli_namespace(args)
    assert restored == program
    assert is_database_authority_mode(restored.authority_mode)


def test_database_program_from_cli_marks_legacy_explicit() -> None:
    args = SimpleNamespace(
        authority_mode=AUTHORITY_MODE_LEGACY_MARKDOWN,
        task_source_kind="legacy-markdown",
        explicit_legacy_task_source=False,
        endpoint_secret_handle="",
        state_store_id="",
        state_store_generation="",
        state_schema_revision="",
        event_store_path="",
        runtime_registry_path="",
        worktree_root="",
        export_profile="",
        state_failover_policy="",
    )
    # Parser auto-sets explicit_legacy when authority_mode is legacy_markdown so
    # DatabaseProgramConfig@1 construction remains fail-closed but usable.
    program = database_program_from_cli_namespace(args)
    assert program is not None
    assert program.authority_mode == AUTHORITY_MODE_LEGACY_MARKDOWN
    assert program.explicit_legacy is True
    assert not is_database_authority_mode(program.authority_mode)


def test_database_program_from_cli_rejects_quack_without_secret_handle() -> None:
    args = SimpleNamespace(
        authority_mode=AUTHORITY_MODE_QUACK,
        task_source_kind="duckdb",
        explicit_legacy_task_source=False,
        endpoint_secret_handle="",
        state_store_id="control.duckdb",
        state_store_generation="gen-1",
        state_schema_revision="schema-v1",
        event_store_path="",
        runtime_registry_path="",
        worktree_root="",
        export_profile="",
        state_failover_policy="fail_closed",
    )
    with pytest.raises(DatabaseProgramConfigError):
        database_program_from_cli_namespace(args)


def test_build_portal_implementation_daemon_from_args_propagates_program(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_init(repo)
    todo = repo / "todo.md"
    todo.write_text("# board\n", encoding="utf-8")
    _git_commit_all(repo)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    args = parse_args(
        [
            "--todo-path",
            str(todo),
            "--state-dir",
            str(state_dir),
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--state-store-id",
            "control.duckdb",
            "--database-session-id",
            "session:runner",
            "--once",
        ]
    )
    # Avoid opening a duckdb task source against a markdown path: use
    # legacy path construction with only the database program for authority.
    args.task_source_kind = "legacy-markdown"
    args.explicit_legacy_task_source = True
    # Force embedded authority while keeping a path-backed todo for construction.
    monkeypatch.setenv(
        DATABASE_PROGRAM_JSON_ENV,
        json.dumps(
            {
                "authority_mode": AUTHORITY_MODE_EMBEDDED,
                "task_source_kind": "duckdb",
                "store_id": "control.duckdb",
                "failover_policy": "fail_closed",
            },
            sort_keys=True,
        ),
    )
    # Re-parse program with env while keeping todo as markdown path for open.
    args.authority_mode = ""
    args.task_source_kind = ""
    daemon, context = build_portal_implementation_daemon_from_args(
        args,
        repo_root=repo,
    )
    assert isinstance(daemon, PortalImplementationDaemon)
    assert daemon.database_program is not None
    assert daemon.database_authority_active is True
    assert context.state_path == implementation_state_paths(args)["state_path"]
    assert DATABASE_IMPLEMENTATION_DAEMON_INTERFACE == (
        "DatabaseImplementationDaemon@1"
    )


def test_daemon_pass_helpers_remain_exported() -> None:
    # Runner helpers stay available for configured wrappers.
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
        compact_daemon_pass_result,
        daemon_pass_is_idle,
    )

    idle = {
        "unchanged": True,
        "write_count": 0,
        "task_count": 0,
        "ready_count": 0,
    }
    assert daemon_pass_is_idle(idle) is True
    assert "task_count" in compact_daemon_pass_result(idle)


def test_module_exports_database_cutover_symbols() -> None:
    assert hasattr(daemon_module, "DatabaseImplementationDaemon")
    assert hasattr(daemon_module, "DatabaseTaskAttempt")
    assert hasattr(daemon_module, "database_program_from_cli_namespace")
    assert hasattr(daemon_module, "is_database_authority_mode")
    assert hasattr(daemon_module, "open_database_implementation_daemon")
