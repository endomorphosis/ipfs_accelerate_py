from __future__ import annotations

import json
import os
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
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop import (
    SupervisorLoop,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    SUPERVISED_CHILD_IDENTITY_PATH_ENV,
    SUPERVISED_CHILD_OWNER_SCOPE_ENV,
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QUACK_TOKEN", raising=False)
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", raising=False)
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
            "--quack-endpoint",
            "quack:127.0.0.1:45123",
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
    assert program.quack_endpoint == "quack:127.0.0.1:45123"
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


def test_managed_daemon_forwards_env_secret_handle_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUACK_TOKEN", "admitted-parent-token")
    (tmp_path / "tasks.md").write_text("# Tasks\n", encoding="utf-8")
    (tmp_path / "state").mkdir()
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
            "--quack-endpoint",
            "quack:127.0.0.1:45123",
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
    child_env = supervisor_module._managed_daemon_child_environment(
        database_program=program,
    )
    assert child_env["QUACK_TOKEN"] == "admitted-parent-token"


def test_direct_supervisor_round_trips_embedded_one_writer_authority(
    tmp_path: Path,
) -> None:
    todo_path = tmp_path / "tasks.md"
    todo_path.write_text("# Tasks\n", encoding="utf-8")
    state_dir = tmp_path / "state"
    store_id = "data/agent_supervisor/lgcvf-bootstrap/control.duckdb"
    args = supervisor_module.parse_args(
        [
            "--todo-path",
            str(todo_path),
            "--state-dir",
            str(state_dir),
            "--worktree-root",
            str(tmp_path / "worktrees"),
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--state-store-id",
            store_id,
            "--state-store-generation",
            "lgcvf-bootstrap-v1",
            "--state-schema-revision",
            "datasets-authoritative-operational-v1",
            "--state-failover-policy",
            "fail_closed",
        ]
    )

    config = supervisor_module.supervisor_config_from_args(
        args,
        repo_root=tmp_path,
    )
    program = config.database_program
    assert program is not None
    assert program.authority_mode == "embedded"
    assert program.task_source_kind == "duckdb"
    assert program.store_id == store_id
    assert not program.endpoint_secret_handle
    assert not program.quack_endpoint

    supervisor = supervisor_module.PortalImplementationSupervisor(config)
    command = supervisor._build_daemon_command()
    assert command.count("--task-source-kind") == 1
    assert command[command.index("--task-source-kind") + 1] == "duckdb"
    assert command.count("--authority-mode") == 1
    assert command[command.index("--authority-mode") + 1] == "embedded"
    assert command.count("--state-store-id") == 1
    assert command[command.index("--state-store-id") + 1] == store_id
    assert command.count("--state-store-generation") == 1
    assert (
        command[command.index("--state-store-generation") + 1]
        == "lgcvf-bootstrap-v1"
    )
    assert command.count("--state-schema-revision") == 1
    assert (
        command[command.index("--state-schema-revision") + 1]
        == "datasets-authoritative-operational-v1"
    )
    assert supervisor._managed_daemon_matches_command_line(" ".join(command))

    for option, stale_value in (
        ("--task-source-kind", "markdown"),
        ("--authority-mode", "embedded_exclusive"),
        ("--state-store-id", "different-control.duckdb"),
        ("--state-store-generation", "stale-generation"),
        ("--state-schema-revision", "stale-schema"),
        ("--state-failover-policy", "require_explicit_operator"),
    ):
        stale_command = list(command)
        stale_command[stale_command.index(option) + 1] = stale_value
        assert not supervisor._managed_daemon_matches_command_line(
            " ".join(stale_command)
        )

    injected_command = [
        *command,
        "--runtime-registry-path",
        "different/registry",
    ]
    assert not supervisor._managed_daemon_matches_command_line(
        " ".join(injected_command)
    )

    authority_env = supervisor_module._managed_daemon_child_environment(
        database_program=program,
    )
    loop_config = supervisor.build_supervisor_loop_config()
    child_env = dict(loop_config.child_env)
    assert {
        key: value
        for key, value in child_env.items()
        if key
        not in {
            SUPERVISED_CHILD_IDENTITY_PATH_ENV,
            SUPERVISED_CHILD_OWNER_SCOPE_ENV,
        }
    } == authority_env
    assert child_env[SUPERVISED_CHILD_IDENTITY_PATH_ENV] == str(
        supervisor._managed_daemon_identity_path()
    )
    assert json.loads(child_env[SUPERVISED_CHILD_OWNER_SCOPE_ENV]) == (
        supervisor._managed_daemon_owner_scope()
    )
    assert loop_config.spec.launch_env == child_env
    loop = SupervisorLoop(loop_config)
    assert loop._child_spec("initial").env == child_env
    assert loop._child_spec("restart").env == child_env
    assert str(supervisor_module.REPO_ROOT) in child_env["PYTHONPATH"].split(os.pathsep)
    daemon_entrypoint = (
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
    )
    daemon_argv = command[command.index(daemon_entrypoint) + 1 :]
    daemon_args = daemon_module.parse_args(daemon_argv)
    assert daemon_module.database_program_from_daemon_namespace(
        daemon_args,
        environ=authority_env,
    ) == program
    # The immutable non-secret store authority is also reconstructable from
    # argv alone.  A missing environment binding must never reinterpret the
    # Markdown projection as the DuckDB store.
    assert daemon_module.database_program_from_daemon_namespace(
        daemon_args,
        environ={},
    ) == program

    provider_env = supervisor.provider_subprocess_environment(
        {"QUACK_TOKEN": "must-not-cross", **child_env}
    )
    assert "QUACK_TOKEN" not in provider_env
    assert DATABASE_PROGRAM_JSON_ENV not in provider_env
    assert STATE_AUTHORITY_MODE_ENV not in provider_env
    assert TASK_SOURCE_KIND_ENV not in provider_env


def test_source_change_reload_preserves_embedded_programmatic_launch_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An embedded ``main(argv)`` reload must never fall back to CLI defaults."""

    repo = tmp_path.resolve()
    todo_path = repo / "docs" / "lgcvf.todo.md"
    todo_path.parent.mkdir()
    todo_path.write_text("# LGCVF tasks\n", encoding="utf-8")
    for relative in ("policy/operator-seal.json", "policy/benchmark.json"):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    original_argv = [
        "--todo-path",
        str(todo_path),
        "--state-dir",
        str(repo / "run-v16" / "state"),
        "--state-prefix",
        "portal",
        "--task-prefix",
        "## LGCVF-",
        "--stale-seconds",
        "1800",
        "--check-interval",
        "17",
        "--max-restarts",
        "10",
        "--max-task-attempts",
        "3",
        "--daemon-interval",
        "23",
        "--implement",
        "--implementation-command",
        "python -m deterministic_provider",
        "--implementation-timeout",
        "901",
        "--implementation-max-timeout",
        "1201",
        "--validation-max-workers",
        "4",
        "--worktree-root",
        str(repo / "run-v16" / "worktrees"),
        "--worktree-submodule-path",
        "ipfs_datasets_py",
        "--implementation-protected-path",
        "policy/operator-seal.json",
        "--implementation-protected-path",
        "policy/benchmark.json",
        "--task-source-kind",
        "duckdb",
        "--authority-mode",
        "embedded",
        "--state-store-id",
        "run-v16/control.duckdb",
        "--state-store-generation",
        "lgcvf-run-v16",
        "--state-schema-revision",
        "datasets-authoritative-operational-v1",
        "--state-failover-policy",
        "fail_closed",
        "--task-shard-count",
        "1",
        "--task-shard-index",
        "0",
        "--strict-task-sharding",
    ]
    parsed = supervisor_module.parse_args(original_argv)
    config = supervisor_module.supervisor_config_from_args(
        parsed,
        repo_root=repo,
    )
    supervisor = supervisor_module.PortalImplementationSupervisor(config)
    original_child_command = supervisor._build_daemon_command()

    class ExecRequested(Exception):
        pass

    calls: list[tuple[str, list[str]]] = []

    def fake_execv(executable: str, arguments: list[str]) -> None:
        calls.append((executable, arguments))
        raise ExecRequested

    # Reproduce the failed production shape: the accepted arguments were
    # passed to main(argv), while sys.argv belonged to an embedding launcher.
    monkeypatch.setattr(supervisor_module.sys, "argv", ["embedded-launcher"])
    monkeypatch.setattr(supervisor_module.os, "execv", fake_execv)

    with pytest.raises(ExecRequested):
        supervisor._reload_for_control_plane_update()

    assert len(calls) == 1
    executable, reload_command = calls[0]
    assert executable == supervisor_module.sys.executable
    assert reload_command[:3] == [
        supervisor_module.sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor",
    ]
    assert reload_command[3:] == original_argv

    reloaded_args = supervisor_module.parse_args(reload_command[3:])
    reloaded_config = supervisor_module.supervisor_config_from_args(
        reloaded_args,
        repo_root=repo,
    )
    reloaded_child_command = supervisor_module.PortalImplementationSupervisor(
        reloaded_config
    )._build_daemon_command()

    # Exact command equality covers every forwarded policy field and repeated
    # protected path.  The focused assertions make the safety-critical values
    # visible if this regression ever fails.
    assert reloaded_child_command == original_child_command
    assert "--implement" in reloaded_child_command
    assert reloaded_child_command[
        reloaded_child_command.index("--task-prefix") + 1
    ] == "## LGCVF-"
    assert reloaded_child_command[
        reloaded_child_command.index("--max-task-attempts") + 1
    ] == "3"
    assert reloaded_child_command[
        reloaded_child_command.index("--validation-max-workers") + 1
    ] == "4"
    protected = [
        reloaded_child_command[index + 1]
        for index, token in enumerate(reloaded_child_command[:-1])
        if token == "--implementation-protected-path"
    ]
    assert protected == [
        "policy/operator-seal.json",
        "policy/benchmark.json",
    ]
    assert str(todo_path) in reloaded_child_command
    assert "docs/211_SERVICE_NAVIGATION_PORTAL_TODO.md" not in (
        reloaded_child_command
    )


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
