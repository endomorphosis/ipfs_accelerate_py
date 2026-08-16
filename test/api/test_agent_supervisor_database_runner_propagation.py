"""Tests for DatabaseProgramConfig@1 runner propagation (DQP-017).

Evidence subset: parser round trip, argv/env redaction, defaults, explicit
legacy mode, child environment, lane isolation, restart/adoption.

Acceptance: No database selection is lost between configured-board,
multi-runner, implementation supervisor and daemon; current implicit
legacy-Markdown default is deprecated; Quack authority never silently becomes
local DuckDB or file authority; provider subprocess lacks state credentials.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
    ConfiguredBoardError,
    configured_board_common_args,
    configured_board_launch_plan,
    load_configured_board,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    AUTHORITY_MODE_QUACK,
    DATABASE_IMPLEMENTATION_TRACK_INTERFACE,
    DATABASE_PROGRAM_CONFIG_INTERFACE,
    DATABASE_PROGRAM_JSON_ENV,
    DatabaseImplementationTrack,
    DatabaseProgramConfig,
    DatabaseProgramConfigError,
    STATE_AUTHORITY_MODE_ENV,
    STATE_CREDENTIAL_ENV_NAMES,
    STATE_ENDPOINT_SECRET_HANDLE_ENV,
    TASK_SOURCE_KIND_ENV,
    expand_database_implementation_track_lanes,
    parse_database_program_config,
    provider_subprocess_environment,
    redact_database_program_argv,
    scrub_state_credentials_from_environment,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    PortalSupervisorConfig,
    database_program_from_cli_namespace,
    provider_environment_without_state_credentials,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor as supervisor_module,
)


def _quack_program(**overrides: object) -> DatabaseProgramConfig:
    payload = {
        "authority_mode": "quack",
        "task_source_kind": "duckdb",
        "endpoint_secret_handle": "env://QUACK_TOKEN",
        "store_id": "control.duckdb",
        "store_generation": "gen-7",
        "schema_revision": "schema-v1",
        "event_store_path": "state/events",
        "runtime_registry_path": "state/registry",
        "worktree_root": "state/worktrees",
        "export_profile": "operator-export",
        "failover_policy": "fail_closed",
        "explicit_legacy": False,
    }
    payload.update(overrides)
    return DatabaseProgramConfig.from_mapping(payload)


def test_interface_identities() -> None:
    assert DATABASE_PROGRAM_CONFIG_INTERFACE == "DatabaseProgramConfig@1"
    assert DATABASE_IMPLEMENTATION_TRACK_INTERFACE == (
        "DatabaseImplementationTrack@1"
    )
    assert DatabaseProgramConfig.INTERFACE == DATABASE_PROGRAM_CONFIG_INTERFACE
    assert (
        DatabaseImplementationTrack.INTERFACE
        == DATABASE_IMPLEMENTATION_TRACK_INTERFACE
    )


def test_parser_round_trip_preserves_all_fields() -> None:
    program = _quack_program()
    again = DatabaseProgramConfig.from_mapping(program.to_dict())
    assert again == program
    assert again.to_dict() == program.to_dict()
    assert parse_database_program_config(program.to_dict()) == program


def test_explicit_legacy_mode_is_required() -> None:
    with pytest.raises(DatabaseProgramConfigError, match="explicit_legacy"):
        DatabaseProgramConfig(
            authority_mode="legacy_markdown",
            task_source_kind="legacy-markdown",
            explicit_legacy=False,
        )
    legacy = DatabaseProgramConfig.explicit_legacy_markdown()
    assert legacy.authority_mode == "legacy_markdown"
    assert legacy.task_source_kind == "legacy-markdown"
    assert legacy.explicit_legacy is True
    assert "--task-source-kind" in legacy.cli_args()
    assert "legacy-markdown" in legacy.cli_args()
    assert "--explicit-legacy-task-source" in legacy.cli_args()


def test_quack_authority_rejects_silent_local_failover() -> None:
    with pytest.raises(DatabaseProgramConfigError, match="fail_closed"):
        _quack_program(failover_policy="require_explicit_operator")
    with pytest.raises(DatabaseProgramConfigError, match="endpoint_secret_handle"):
        _quack_program(endpoint_secret_handle="")
    with pytest.raises(DatabaseProgramConfigError, match="raw credentials"):
        _quack_program(endpoint_secret_handle="super-secret-token-value")
    program = _quack_program()
    with pytest.raises(DatabaseProgramConfigError, match="silently become"):
        program.assert_quack_not_demoted(candidate_mode="embedded")
    with pytest.raises(DatabaseProgramConfigError, match="silently become"):
        program.assert_quack_not_demoted(candidate_mode="legacy_markdown")
    with pytest.raises(DatabaseProgramConfigError, match="silently become"):
        program.assert_quack_not_demoted(candidate_mode="file")
    program.assert_quack_not_demoted(candidate_mode=AUTHORITY_MODE_QUACK)


def test_argv_and_env_redaction_keeps_handles_not_tokens() -> None:
    program = _quack_program()
    argv = program.cli_args() + ["--state-token", "raw-token-bytes"]
    redacted = redact_database_program_argv(argv)
    assert "env://QUACK_TOKEN" in redacted
    assert "raw-token-bytes" not in redacted
    assert "secret_material" in redacted

    env = {
        "QUACK_TOKEN": "raw-token-bytes",
        "PATH": "/usr/bin",
        STATE_ENDPOINT_SECRET_HANDLE_ENV: "env://QUACK_TOKEN",
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN": "also-secret",
    }
    cleaned = scrub_state_credentials_from_environment(
        env,
        secret_handle=program.endpoint_secret_handle,
    )
    assert "QUACK_TOKEN" not in cleaned
    assert "IPFS_ACCELERATE_AGENT_QUACK_TOKEN" not in cleaned
    assert cleaned["PATH"] == "/usr/bin"
    assert cleaned[STATE_ENDPOINT_SECRET_HANDLE_ENV] == "env://QUACK_TOKEN"
    for name in STATE_CREDENTIAL_ENV_NAMES:
        assert name not in cleaned


def test_provider_subprocess_lacks_state_credentials() -> None:
    program = _quack_program()
    ambient = {
        "QUACK_TOKEN": "raw-token-bytes",
        "HOME": "/tmp/home",
        STATE_AUTHORITY_MODE_ENV: "quack",
        TASK_SOURCE_KIND_ENV: "duckdb",
        DATABASE_PROGRAM_JSON_ENV: json.dumps(program.to_dict()),
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER": "grok_cli",
    }
    provider_env = provider_subprocess_environment(
        ambient,
        program=program,
    )
    assert "QUACK_TOKEN" not in provider_env
    assert STATE_AUTHORITY_MODE_ENV not in provider_env
    assert TASK_SOURCE_KIND_ENV not in provider_env
    assert DATABASE_PROGRAM_JSON_ENV not in provider_env
    assert provider_env["IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER"] == (
        "grok_cli"
    )
    assert provider_env["HOME"] == "/tmp/home"


def test_lane_isolation_preserves_identical_program_selection() -> None:
    program = _quack_program()
    track = DatabaseImplementationTrack(
        name="dqp-main",
        script_path="scripts/ops/agent_supervisor/implementation_supervisor_entry.py",
        state_dir="state/agent_supervisor/state",
        state_prefix="dqp",
        database_program=program,
    )
    lanes = expand_database_implementation_track_lanes(
        track,
        stamp="20260809T000000Z",
        lanes_per_track=4,
    )
    assert len(lanes) == 4
    for index, lane in enumerate(lanes):
        assert lane.database_program == program
        assert f"lane-{index}" in str(lane.supervisor_pid_path)
        assert "--task-source-kind" in lane.extra_args
        assert "duckdb" in lane.extra_args
        assert "--authority-mode" in lane.extra_args
        assert "quack" in lane.extra_args
        assert "--endpoint-secret-handle" in lane.extra_args
        assert "env://QUACK_TOKEN" in lane.extra_args
        assert "--task-shard-count" in lane.extra_args
        assert lane.extra_args[lane.extra_args.index("--task-shard-index") + 1] == (
            str(index)
        )


def test_supervisor_propagates_program_to_daemon_command_and_child_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = _quack_program(worktree_root="")
    todo = tmp_path / "tasks.md"
    todo.write_text("# Tasks\n", encoding="utf-8")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    config = PortalSupervisorConfig(
        todo_path=todo,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        state_dir=state_dir,
        implement=True,
        database_program=program,
        repo_root=tmp_path,
    )
    supervisor = PortalImplementationSupervisor(config)
    command = supervisor._build_daemon_command()
    assert command.count("--task-source-kind") == 1
    kind_index = command.index("--task-source-kind")
    assert command[kind_index + 1] == "duckdb"

    child_env = supervisor_module._managed_daemon_child_environment(
        database_program=program,
    )
    assert child_env[STATE_AUTHORITY_MODE_ENV] == "quack"
    assert child_env[TASK_SOURCE_KIND_ENV] == "duckdb"
    assert child_env[STATE_ENDPOINT_SECRET_HANDLE_ENV] == "env://QUACK_TOKEN"
    assert DATABASE_PROGRAM_JSON_ENV in child_env
    restored = DatabaseProgramConfig.from_mapping(
        json.loads(child_env[DATABASE_PROGRAM_JSON_ENV])
    )
    assert restored == program

    provider_env = supervisor.provider_subprocess_environment(
        {
            "QUACK_TOKEN": "raw-token-bytes",
            "PATH": "/usr/bin",
            **program.environment(),
        }
    )
    assert "QUACK_TOKEN" not in provider_env
    assert STATE_AUTHORITY_MODE_ENV not in provider_env


def test_supervisor_cli_round_trip_from_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = _quack_program(worktree_root="")
    monkeypatch.setenv(
        DATABASE_PROGRAM_JSON_ENV,
        json.dumps(program.to_dict(), sort_keys=True),
    )

    class _Args:
        authority_mode = ""
        task_source_kind = ""
        endpoint_secret_handle = ""
        state_store_id = ""
        state_store_generation = ""
        state_schema_revision = ""
        event_store_path = ""
        runtime_registry_path = ""
        export_profile = ""
        state_failover_policy = ""
        explicit_legacy_task_source = False
        worktree_root = None

    loaded = database_program_from_cli_namespace(_Args())
    assert loaded == program

    class _CliArgs:
        authority_mode = "quack"
        task_source_kind = "duckdb"
        endpoint_secret_handle = "handle:quack-token-1"
        state_store_id = "control.duckdb"
        state_store_generation = "gen-9"
        state_schema_revision = "schema-v2"
        event_store_path = "state/events"
        runtime_registry_path = "state/registry"
        export_profile = "ops"
        state_failover_policy = "fail_closed"
        explicit_legacy_task_source = False
        worktree_root = None

    monkeypatch.delenv(DATABASE_PROGRAM_JSON_ENV, raising=False)
    from_cli = database_program_from_cli_namespace(_CliArgs())
    assert from_cli is not None
    assert from_cli.authority_mode == "quack"
    assert from_cli.store_generation == "gen-9"
    assert from_cli.endpoint_secret_handle == "handle:quack-token-1"


def test_configured_board_propagates_database_program(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "config").mkdir()
    (repo / "scripts" / "ops" / "agent_supervisor").mkdir(parents=True)
    (repo / "docs" / "tasks.md").write_text("# Tasks\n", encoding="utf-8")
    (repo / "docs" / "objectives.md").write_text("# Objectives\n", encoding="utf-8")
    (repo / "docs" / "plan.md").write_text("plan\n", encoding="utf-8")
    (repo / "scripts" / "validate_board.py").write_text(
        "print('ok')\n",
        encoding="utf-8",
    )
    (
        repo
        / "scripts"
        / "ops"
        / "agent_supervisor"
        / "implementation_supervisor_entry.py"
    ).write_text("raise SystemExit(0)\n", encoding="utf-8")

    import subprocess

    def _git(*args: str) -> None:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    _git("init", "-b", "main")
    _git("config", "user.name", "DQP Propagation")
    _git("config", "user.email", "dqp@example.invalid")
    for path in (
        "docs/tasks.md",
        "docs/objectives.md",
        "docs/plan.md",
        "scripts/validate_board.py",
        "scripts/ops/agent_supervisor/implementation_supervisor_entry.py",
    ):
        _git("add", path)
    _git("commit", "-m", "seed")
    ancestor = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()

    config_path = repo / "config" / "scheduler.json"
    payload = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "configured_board_test.scheduler_config@1"
        ),
        "taskboard_path": "docs/tasks.md",
        "objectives_path": "docs/objectives.md",
        "plan_path": "docs/plan.md",
        "validator_path": "scripts/validate_board.py",
        "task_prefix": "TEST-",
        "goal_prefix": "TEST-G",
        "board_namespace": "configured-board-db",
        "merge_target_branch": "main",
        "source_binding": {
            "accelerator_required_ancestor": ancestor,
            "accelerator_required_branch": "main",
        },
        "max_lanes": 2,
        "strict_task_sharding": True,
        "exit_when_all_tracks_terminal": True,
        "objective_refill_enabled": False,
        "codebase_refill_enabled": False,
        "poll_interval_seconds": 5,
        "daemon_interval_seconds": 60,
        "check_interval_seconds": 30,
        "stale_seconds": 1800,
        "watchdog_startup_grace_seconds": 300,
        "max_restarts": 3,
        "max_task_attempts": 3,
        "implementation_retry_budget": 3,
        "validation_retry_budget": 3,
        "merge_retry_budget": 3,
        "implementation_timeout_seconds": 7200,
        "implementation_max_timeout_seconds": 21600,
        "implementation_log_stall_seconds": 1200,
        "worktree_submodule_paths": [],
        "protected_paths": [
            "config/scheduler.json",
            "docs/plan.md",
            "docs/objectives.md",
            "docs/tasks.md",
            "scripts/validate_board.py",
        ],
        "runtime_paths": {
            "root": "data/configured-board",
            "state": "data/configured-board/state",
            "worktrees": "data/configured-board/worktrees",
            "merge_queue": "data/configured-board/merge-queue",
            "logs": "data/configured-board/logs",
        },
        "lanes": [
            {"index": 0, "name": "lane-0", "strict_shard_remainder": 0},
            {"index": 1, "name": "lane-1", "strict_shard_remainder": 1},
        ],
        "provider": {
            "provider_id": "codex",
            "model_id": "test-model",
            "max_concurrency": 2,
        },
        "database_program": {
            "authority_mode": "quack",
            "task_source_kind": "duckdb",
            "endpoint_secret_handle": "env://QUACK_TOKEN",
            "store_id": "control.duckdb",
            "store_generation": "gen-1",
            "schema_revision": "schema-v1",
            "event_store_path": "data/configured-board/events",
            "runtime_registry_path": "data/configured-board/registry",
            "export_profile": "board-export",
            "failover_policy": "fail_closed",
        },
    }
    config_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _git("add", "config/scheduler.json")
    _git("commit", "-m", "add database program")

    board = load_configured_board(config_path, repo_root=repo)
    assert board.database_program is not None
    assert board.database_program.authority_mode == "quack"
    assert board.database_program.store_generation == "gen-1"
    assert board.database_program.worktree_root == (
        "data/configured-board/worktrees"
    )

    common = configured_board_common_args(board, implement=True)
    assert common.count("--task-source-kind") == 1
    assert common[common.index("--task-source-kind") + 1] == "duckdb"
    assert common[common.index("--authority-mode") + 1] == "quack"
    assert common[common.index("--endpoint-secret-handle") + 1] == (
        "env://QUACK_TOKEN"
    )
    assert common[common.index("--state-store-generation") + 1] == "gen-1"

    plan = configured_board_launch_plan(
        board,
        implement=True,
        detach=True,
        stamp="20260809T120000Z",
    )
    assert plan["database_program"]["authority_mode"] == "quack"
    assert plan["environment"][STATE_AUTHORITY_MODE_ENV] == "quack"
    assert plan["environment"][TASK_SOURCE_KIND_ENV] == "duckdb"
    assert plan["environment"][STATE_ENDPOINT_SECRET_HANDLE_ENV] == (
        "env://QUACK_TOKEN"
    )
    # Raw credentials never enter the launch environment.
    assert "QUACK_TOKEN" not in plan["environment"]
    assert "raw" not in json.dumps(plan["database_program"])

    # Common-arg propagation preserves the selection through multi-runner argv.
    common_from_plan = [
        item[len("--common-arg=") :]
        for item in plan["argv"]
        if isinstance(item, str) and item.startswith("--common-arg=")
    ]
    assert "--task-source-kind" in common_from_plan
    assert "duckdb" in common_from_plan
    assert "--authority-mode" in common_from_plan
    assert "quack" in common_from_plan


def test_configured_board_rejects_quack_without_handle(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "config").mkdir()
    (repo / "scripts").mkdir()
    (repo / "docs" / "tasks.md").write_text("# Tasks\n", encoding="utf-8")
    (repo / "docs" / "objectives.md").write_text("# Objectives\n", encoding="utf-8")
    (repo / "docs" / "plan.md").write_text("plan\n", encoding="utf-8")
    (repo / "scripts" / "validate_board.py").write_text("print('ok')\n", encoding="utf-8")
    config_path = repo / "config" / "scheduler.json"
    payload = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "configured_board_test.scheduler_config@1"
        ),
        "taskboard_path": "docs/tasks.md",
        "objectives_path": "docs/objectives.md",
        "plan_path": "docs/plan.md",
        "validator_path": "scripts/validate_board.py",
        "task_prefix": "TEST-",
        "board_namespace": "configured-board-db",
        "merge_target_branch": "main",
        "max_lanes": 1,
        "strict_task_sharding": True,
        "exit_when_all_tracks_terminal": True,
        "objective_refill_enabled": False,
        "codebase_refill_enabled": False,
        "poll_interval_seconds": 5,
        "daemon_interval_seconds": 60,
        "check_interval_seconds": 30,
        "stale_seconds": 1800,
        "watchdog_startup_grace_seconds": 300,
        "max_restarts": 3,
        "max_task_attempts": 3,
        "implementation_retry_budget": 3,
        "validation_retry_budget": 3,
        "merge_retry_budget": 3,
        "implementation_timeout_seconds": 7200,
        "implementation_max_timeout_seconds": 21600,
        "implementation_log_stall_seconds": 1200,
        "worktree_submodule_paths": [],
        "protected_paths": ["config/scheduler.json"],
        "runtime_paths": {
            "root": "data/root",
            "state": "data/root/state",
            "worktrees": "data/root/worktrees",
            "merge_queue": "data/root/merge-queue",
            "logs": "data/root/logs",
        },
        "lanes": [
            {"index": 0, "name": "lane-0", "strict_shard_remainder": 0},
        ],
        "provider": {"max_concurrency": 1},
        "database_program": {
            "authority_mode": "quack",
            "task_source_kind": "duckdb",
            "store_id": "control.duckdb",
            "store_generation": "1",
            "schema_revision": "schema-v1",
        },
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfiguredBoardError, match="endpoint_secret_handle"):
        load_configured_board(config_path, repo_root=repo)


def test_provider_environment_helper_on_supervisor_module() -> None:
    cleaned = provider_environment_without_state_credentials(
        {
            "QUACK_TOKEN": "x",
            "PATH": "/usr/bin",
        },
        database_program=_quack_program(),
    )
    assert "QUACK_TOKEN" not in cleaned
    assert cleaned["PATH"] == "/usr/bin"
