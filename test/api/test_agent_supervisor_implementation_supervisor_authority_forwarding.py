from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    DATABASE_PROGRAM_JSON_ENV,
    STATE_AUTHORITY_MODE_ENV,
    STATE_FAILOVER_POLICY_ENV,
    TASK_SOURCE_KIND_ENV,
    DatabaseProgramConfigError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor as supervisor_module,
)

_LEGACY_AUTHORITY_ARGS = (
    "--task-source-kind",
    "legacy-markdown",
    "--authority-mode",
    "legacy_markdown",
    "--state-failover-policy",
    "fail_closed",
    "--explicit-legacy-task-source",
)


def test_supervisor_accepts_and_forwards_explicit_legacy_authority(
    tmp_path: Path,
) -> None:
    todo_path = tmp_path / "tasks.md"
    todo_path.write_text("# Tasks\n", encoding="utf-8")
    state_dir = tmp_path / "state"
    args = supervisor_module.parse_args(
        [
            "--once",
            "--implement",
            "--todo-path",
            str(todo_path),
            "--state-dir",
            str(state_dir),
            "--worktree-root",
            str(tmp_path / "worktrees"),
            *_LEGACY_AUTHORITY_ARGS,
        ]
    )

    program = supervisor_module.database_program_from_cli_namespace(
        args,
        environ={},
    )
    assert program is not None
    assert program.task_source_kind == "legacy-markdown"
    assert program.authority_mode == "legacy_markdown"
    assert program.failover_policy == "fail_closed"
    assert program.explicit_legacy is True
    assert program.worktree_root == ""

    config = supervisor_module.supervisor_config_from_args(
        args,
        repo_root=tmp_path,
    )
    assert config.database_program == program
    command = supervisor_module.PortalImplementationSupervisor(
        config
    )._build_daemon_command()
    assert command.count("--task-source-kind") == 1
    source_index = command.index("--task-source-kind")
    assert command[source_index + 1] == "legacy-markdown"

    child_env = supervisor_module._managed_daemon_child_environment(
        database_program=config.database_program,
    )
    assert child_env[TASK_SOURCE_KIND_ENV] == "legacy-markdown"
    assert child_env[STATE_AUTHORITY_MODE_ENV] == "legacy_markdown"
    assert child_env[STATE_FAILOVER_POLICY_ENV] == "fail_closed"
    assert DATABASE_PROGRAM_JSON_ENV in child_env

    daemon_entrypoint = (
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
    )
    daemon_argv = command[command.index(daemon_entrypoint) + 1 :]
    daemon_args = daemon_module.parse_args(daemon_argv)
    assert daemon_args.worktree_root == tmp_path / "worktrees"
    daemon_program = daemon_module.database_program_from_daemon_namespace(
        daemon_args,
        environ=child_env,
    )
    assert daemon_program == program


def test_supervisor_round_trips_full_quack_authority_without_raw_credentials(
    tmp_path: Path,
) -> None:
    args = supervisor_module.parse_args(
        [
            "--todo-path",
            str(tmp_path / "tasks.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--worktree-root",
            str(tmp_path / "worktrees"),
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "quack",
            "--endpoint-secret-handle",
            "env://QUACK_TOKEN",
            "--state-store-id",
            "control.duckdb",
            "--state-store-generation",
            "gen-7",
            "--state-schema-revision",
            "schema-v1",
            "--event-store-path",
            "state/events",
            "--runtime-registry-path",
            "state/registry",
            "--export-profile",
            "operator-export",
            "--state-failover-policy",
            "fail_closed",
        ]
    )
    program = supervisor_module.database_program_from_cli_namespace(
        args,
        environ={},
    )
    assert program is not None
    assert program.authority_mode == "quack"
    assert program.task_source_kind == "duckdb"
    assert program.endpoint_secret_handle == "env://QUACK_TOKEN"
    assert program.store_id == "control.duckdb"
    assert program.store_generation == "gen-7"
    assert program.schema_revision == "schema-v1"
    assert program.worktree_root == ""

    child_env = supervisor_module._managed_daemon_child_environment(
        database_program=program,
    )
    assert child_env[STATE_AUTHORITY_MODE_ENV] == "quack"
    assert child_env[TASK_SOURCE_KIND_ENV] == "duckdb"
    assert "QUACK_TOKEN" not in child_env


def test_supervisor_rejects_inconsistent_authority_selection(
    tmp_path: Path,
) -> None:
    args = supervisor_module.parse_args(
        [
            "--todo-path",
            str(tmp_path / "tasks.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "legacy_markdown",
            "--explicit-legacy-task-source",
        ]
    )
    with pytest.raises(DatabaseProgramConfigError, match="legacy_markdown"):
        supervisor_module.database_program_from_cli_namespace(
            args,
            environ={},
        )
