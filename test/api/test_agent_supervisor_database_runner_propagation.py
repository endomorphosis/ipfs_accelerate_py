"""DQP-017: database task-source / Quack option propagation through runners."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
    ConfiguredBoardError,
    configured_board_common_args,
    configured_board_launch_plan,
    load_configured_board,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    DATABASE_IMPLEMENTATION_TRACK_INTERFACE,
    DATABASE_PROGRAM_CONFIG_INTERFACE,
    DatabaseProgramConfig,
    DatabaseProgramConfigError,
    ImplementationSupervisorTrackConfig,
    expand_implementation_track_config_lanes,
    expand_implementation_track_lanes,
    redact_database_program_argv,
    scrub_state_credentials_from_environment,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    parse_args as parse_supervisor_args,
    supervisor_config_from_args,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(repo: Path, *args: str):
    import subprocess

    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _seed_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "DQP-017")
    _git(repo, "config", "user.email", "dqp017@example.invalid")
    _write(repo / "README.md", "seed\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "seed")
    ancestor = _git(repo, "rev-parse", "HEAD").stdout.strip()

    _write(repo / "docs/plan.md", "plan\n")
    _write(repo / "docs/objectives.md", "# Objectives\n")
    _write(repo / "docs/tasks.md", "# Tasks\n")
    _write(
        repo / "scripts/validate_board.py",
        (
            "import json\n"
            "print(json.dumps({'valid': True, 'errors': []}, sort_keys=True))\n"
        ),
    )
    _write(
        repo
        / "scripts/ops/agent_supervisor/implementation_supervisor_entry.py",
        "raise SystemExit(0)\n",
    )
    config_path = repo / "config/scheduler.json"
    payload = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "database_runner_propagation.scheduler_config@1"
        ),
        "taskboard_path": "docs/tasks.md",
        "objectives_path": "docs/objectives.md",
        "plan_path": "docs/plan.md",
        "validator_path": "scripts/validate_board.py",
        "task_prefix": "TEST-",
        "goal_prefix": "TEST-G",
        "board_namespace": "database-runner-propagation",
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
            {
                "index": 0,
                "name": "lane-0",
                "strict_shard_remainder": 0,
            },
            {
                "index": 1,
                "name": "lane-1",
                "strict_shard_remainder": 1,
            },
        ],
        "provider": {
            "provider_id": "codex",
            "model_id": "test-model",
            "max_concurrency": 2,
        },
    }
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _git(
        repo,
        "add",
        "config/scheduler.json",
        "docs/plan.md",
        "docs/objectives.md",
        "docs/tasks.md",
        "scripts/validate_board.py",
        "scripts/ops/agent_supervisor/implementation_supervisor_entry.py",
    )
    _git(repo, "commit", "-m", "add board")
    return repo, config_path


def _common_args(plan: dict[str, object]) -> list[str]:
    prefix = "--common-arg="
    return [
        item[len(prefix) :]
        for item in plan["argv"]  # type: ignore[index]
        if isinstance(item, str) and item.startswith(prefix)
    ]


def test_database_program_config_interface_and_round_trip() -> None:
    program = DatabaseProgramConfig(
        authority_mode="quack",
        task_source_kind="quack",
        endpoint="quack:127.0.0.1:18080",
        secret_handle="handle:quack-token",
        store_id="store:control-plane",
        generation=3,
        schema_revision=7,
        expected_task_source_root="root:plan",
        expected_task_source_repository_root="repo:tree",
        event_store="events:control",
        runtime_registry="registry:runtime",
        worktree_root="state/worktrees",
        export_profile="json-status",
        failover_policy="fail-closed",
    )
    assert program.INTERFACE == DATABASE_PROGRAM_CONFIG_INTERFACE
    payload = program.to_dict()
    restored = DatabaseProgramConfig.from_mapping(payload)
    assert restored == program
    assert restored.redacted_dict()["secret_handle"] == "handle:quack-token"
    assert "--state-authority-mode" in restored.supervisor_cli_args()
    assert restored.daemon_cli_args()[0:2] == ("--task-source-kind", "duckdb")


def test_quack_authority_never_silently_becomes_local_or_file() -> None:
    with pytest.raises(DatabaseProgramConfigError, match="quack:host:port"):
        DatabaseProgramConfig(
            authority_mode="quack",
            task_source_kind="quack",
            endpoint="/tmp/control.duckdb",
            secret_handle="handle:token",
        )
    with pytest.raises(DatabaseProgramConfigError, match="opaque"):
        DatabaseProgramConfig(
            authority_mode="quack",
            task_source_kind="quack",
            endpoint="quack:127.0.0.1:18080",
            secret_handle="raw-token-value-not-a-handle",
        )
    with pytest.raises(DatabaseProgramConfigError, match="silently demote"):
        DatabaseProgramConfig(
            authority_mode="quack",
            task_source_kind="quack",
            endpoint="quack:127.0.0.1:18080",
            secret_handle="handle:token",
            failover_policy="local-duckdb",
        )
    with pytest.raises(DatabaseProgramConfigError, match="markdown"):
        DatabaseProgramConfig(
            authority_mode="quack",
            task_source_kind="legacy-markdown",
            endpoint="quack:127.0.0.1:18080",
            secret_handle="handle:token",
        )


def test_implicit_legacy_markdown_is_deprecated_but_explicit() -> None:
    program = DatabaseProgramConfig.from_mapping(None)
    assert program.authority_mode == "legacy-markdown"
    assert program.task_source_kind == "legacy-markdown"
    assert program.deprecated_implicit_legacy is True
    assert program.explicit is True
    args = program.supervisor_cli_args()
    assert args[args.index("--state-authority-mode") + 1] == "legacy-markdown"
    assert args[args.index("--task-source-kind") + 1] == "legacy-markdown"


def test_configured_board_propagates_quack_selection_to_launch_plan(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["database_program"] = {
        "authority_mode": "quack",
        "task_source_kind": "quack",
        "endpoint": "quack:127.0.0.1:19090",
        "secret_handle": "env://QUACK_TOKEN",
        "store_id": "store:dqp",
        "generation": 2,
        "schema_revision": 5,
        "event_store": "events:dqp",
        "runtime_registry": "registry:dqp",
        "export_profile": "jsonl-audit",
        "failover_policy": "require-recovery",
        "expected_task_source_root": "root:dqp",
        "expected_task_source_repository_root": "repo:dqp",
    }
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _git(repo, "add", "config/scheduler.json")
    _git(repo, "commit", "-m", "configure quack program")

    board = load_configured_board(config_path, repo_root=repo)
    assert board.database_program.authority_mode == "quack"
    assert board.database_program.secret_handle == "env://QUACK_TOKEN"

    plan = configured_board_launch_plan(
        board,
        implement=True,
        detach=True,
        stamp="20260809T000000Z",
    )
    common = _common_args(plan)
    assert common[common.index("--state-authority-mode") + 1] == "quack"
    assert common[common.index("--task-source-kind") + 1] == "quack"
    assert (
        common[common.index("--state-endpoint") + 1] == "quack:127.0.0.1:19090"
    )
    assert common[common.index("--state-secret-handle") + 1] == "env://QUACK_TOKEN"
    assert common[common.index("--state-store-id") + 1] == "store:dqp"
    assert common[common.index("--state-generation") + 1] == "2"
    assert common[common.index("--state-schema-revision") + 1] == "5"
    assert common[common.index("--event-store") + 1] == "events:dqp"
    assert common[common.index("--runtime-registry") + 1] == "registry:dqp"
    assert common[common.index("--export-profile") + 1] == "jsonl-audit"
    assert common[common.index("--failover-policy") + 1] == "require-recovery"
    assert plan["state_environment"][
        "IPFS_ACCELERATE_AGENT_STATE_AUTHORITY_MODE"
    ] == "quack"
    assert plan["state_environment"][
        "IPFS_ACCELERATE_AGENT_STATE_SECRET_HANDLE"
    ] == "env://QUACK_TOKEN"
    # Provider-safe env must never carry state credentials.
    provider_env = plan["provider_safe_environment"]
    assert "IPFS_ACCELERATE_AGENT_STATE_SECRET_HANDLE" not in provider_env
    assert "IPFS_ACCELERATE_AGENT_STATE_ENDPOINT" not in provider_env
    assert "IPFS_ACCELERATE_AGENT_STATE_AUTHORITY_MODE" not in provider_env
    assert "raw-token" not in json.dumps(plan["database_program"])


def test_configured_board_rejects_quack_with_file_endpoint(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["database_program"] = {
        "authority_mode": "quack",
        "task_source_kind": "quack",
        "endpoint": "data/control.duckdb",
        "secret_handle": "handle:token",
    }
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ConfiguredBoardError, match="quack:host:port"):
        load_configured_board(config_path, repo_root=repo)


def test_configured_board_default_emits_explicit_legacy_selection(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_repo(tmp_path)
    board = load_configured_board(config_path, repo_root=repo)
    assert board.database_program.deprecated_implicit_legacy is True
    common = configured_board_common_args(board, implement=False)
    assert "--state-authority-mode" in common
    assert common[common.index("--state-authority-mode") + 1] == (
        "legacy-markdown"
    )
    assert common[common.index("--task-source-kind") + 1] == "legacy-markdown"


def test_multi_runner_lane_isolation_preserves_database_program() -> None:
    program = DatabaseProgramConfig(
        authority_mode="embedded",
        task_source_kind="duckdb",
        endpoint="state/control.duckdb",
        store_id="store:lanes",
        generation=1,
        schema_revision=1,
        event_store="events:lanes",
        runtime_registry="registry:lanes",
        failover_policy="fail-closed",
    )
    config = ImplementationSupervisorTrackConfig(
        name="prog",
        script_path="scripts/ops/agent_supervisor/implementation_supervisor_entry.py",
        state_dir="state/prog",
        state_prefix="prog",
        database_program=program,
    )
    assert config.INTERFACE == DATABASE_IMPLEMENTATION_TRACK_INTERFACE
    lanes = expand_implementation_track_config_lanes(
        config,
        stamp="20260809T000000Z",
        lanes_per_track=3,
    )
    assert len(lanes) == 3
    for index, lane in enumerate(lanes):
        args = list(lane.extra_args)
        assert args[args.index("--state-authority-mode") + 1] == "embedded"
        assert args[args.index("--task-source-kind") + 1] == "duckdb"
        assert args[args.index("--state-endpoint") + 1] == "state/control.duckdb"
        assert args[args.index("--task-shard-count") + 1] == "3"
        assert args[args.index("--task-shard-index") + 1] == str(index)
        assert f"lane-{index}" in str(lane.supervisor_pid_path)


def test_multi_runner_expand_lanes_string_path_propagates_program() -> None:
    program = DatabaseProgramConfig.explicit_legacy_markdown()
    lanes = expand_implementation_track_lanes(
        "name|script.py|state/dir|prefix",
        stamp="stamp",
        lanes_per_track=2,
        database_program=program,
    )
    assert len(lanes) == 2
    for lane in lanes:
        assert "--state-authority-mode" in lane.extra_args
        assert "legacy-markdown" in lane.extra_args


def test_argv_and_env_redaction_strips_state_credentials() -> None:
    argv = [
        "--state-authority-mode",
        "quack",
        "--state-secret-handle",
        "handle:super-secret",
        "--state-endpoint",
        "quack:127.0.0.1:1",
    ]
    redacted = redact_database_program_argv(argv)
    assert "handle:super-secret" not in redacted
    assert "<redacted-secret-handle>" in redacted
    env = {
        "PATH": "/usr/bin",
        "IPFS_ACCELERATE_AGENT_STATE_SECRET_HANDLE": "handle:x",
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN": "raw-token-value",
        "QUACK_TOKEN": "also-raw",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER": "grok_cli",
    }
    safe = scrub_state_credentials_from_environment(env)
    assert safe == {
        "PATH": "/usr/bin",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER": "grok_cli",
    }


def test_implementation_supervisor_propagates_database_selection_to_daemon(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.todo.md"
    board.write_text("# Tasks\n", encoding="utf-8")
    parsed = parse_supervisor_args(
        [
            "--todo-path",
            str(board),
            "--state-dir",
            str(tmp_path / "state"),
            "--implement",
            "--worktree-root",
            str(tmp_path / "worktrees"),
            "--state-authority-mode",
            "quack",
            "--task-source-kind",
            "quack",
            "--state-endpoint",
            "quack:127.0.0.1:19191",
            "--state-secret-handle",
            "handle:quack-token",
            "--state-store-id",
            "store:sup",
            "--state-generation",
            "4",
            "--state-schema-revision",
            "9",
            "--expected-task-source-root",
            "root:sup",
            "--expected-task-source-repository-root",
            "repo:sup",
            "--event-store",
            "events:sup",
            "--runtime-registry",
            "registry:sup",
            "--export-profile",
            "release-bundle",
            "--failover-policy",
            "fail-closed",
        ]
    )
    config = supervisor_config_from_args(parsed, repo_root=tmp_path)
    program = config.database_program_config()
    assert program.authority_mode == "quack"
    assert program.endpoint == "quack:127.0.0.1:19191"
    assert program.secret_handle == "handle:quack-token"
    assert program.store_id == "store:sup"
    assert program.generation == 4
    assert program.schema_revision == 9
    assert program.export_profile == "release-bundle"

    supervisor = PortalImplementationSupervisor(config)
    command = supervisor._build_daemon_command()
    assert command[:4] == [
        sys.executable,
        "-P",
        "-m",
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon",
    ]
    # Daemon currently accepts duckdb storage kind for Quack transport.
    assert command[command.index("--task-source-kind") + 1] == "duckdb"
    assert command[command.index("--expected-task-source-root") + 1] == "root:sup"
    assert (
        command[command.index("--expected-task-source-repository-root") + 1]
        == "repo:sup"
    )

    child_env = program.child_environment()
    assert child_env["IPFS_ACCELERATE_AGENT_STATE_AUTHORITY_MODE"] == "quack"
    assert (
        child_env["IPFS_ACCELERATE_AGENT_STATE_SECRET_HANDLE"]
        == "handle:quack-token"
    )
    # Provider subprocess environment must lack state credentials.
    provider_env = scrub_state_credentials_from_environment(
        {
            **child_env,
            "PATH": "/usr/bin",
            "IPFS_ACCELERATE_AGENT_QUACK_TOKEN": "must-not-leak",
        }
    )
    assert "IPFS_ACCELERATE_AGENT_STATE_SECRET_HANDLE" not in provider_env
    assert "IPFS_ACCELERATE_AGENT_QUACK_TOKEN" not in provider_env
    assert "must-not-leak" not in provider_env.values()


def test_implementation_supervisor_explicit_legacy_mode(tmp_path: Path) -> None:
    board = tmp_path / "tasks.todo.md"
    board.write_text("# Tasks\n", encoding="utf-8")
    parsed = parse_supervisor_args(
        [
            "--todo-path",
            str(board),
            "--state-dir",
            str(tmp_path / "state"),
            "--state-authority-mode",
            "legacy-markdown",
            "--task-source-kind",
            "legacy-markdown",
        ]
    )
    config = supervisor_config_from_args(parsed, repo_root=tmp_path)
    program = config.database_program_config()
    assert program.authority_mode == "legacy-markdown"
    assert program.deprecated_implicit_legacy is False
    command = PortalImplementationSupervisor(config)._build_daemon_command()
    assert command[command.index("--task-source-kind") + 1] == "legacy-markdown"


def test_implementation_supervisor_rejects_raw_secret_handle(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.todo.md"
    board.write_text("# Tasks\n", encoding="utf-8")
    parsed = parse_supervisor_args(
        [
            "--todo-path",
            str(board),
            "--state-dir",
            str(tmp_path / "state"),
            "--state-authority-mode",
            "quack",
            "--task-source-kind",
            "quack",
            "--state-endpoint",
            "quack:127.0.0.1:1",
            "--state-secret-handle",
            "not-a-handle-token-value",
        ]
    )
    config = supervisor_config_from_args(parsed, repo_root=tmp_path)
    with pytest.raises(ValueError, match="opaque handle"):
        config.database_program_config()


def test_end_to_end_selection_not_lost_configured_board_to_supervisor(
    tmp_path: Path,
) -> None:
    """Common-args from the board are accepted by the supervisor parser."""

    repo, config_path = _seed_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["database_program"] = {
        "authority_mode": "embedded",
        "task_source_kind": "duckdb",
        "endpoint": "state/control.duckdb",
        "store_id": "store:e2e",
        "generation": 1,
        "schema_revision": 2,
        "event_store": "events:e2e",
        "runtime_registry": "registry:e2e",
        "export_profile": "markdown-taskboard",
        "failover_policy": "fail-closed",
    }
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _git(repo, "add", "config/scheduler.json")
    _git(repo, "commit", "-m", "embedded program")

    board = load_configured_board(config_path, repo_root=repo)
    common = list(configured_board_common_args(board, implement=False))
    # Feed the board common args into the supervisor parser.
    todo = repo / board.taskboard_path
    argv = [
        "--todo-path",
        str(todo),
        "--state-dir",
        str(tmp_path / "sup-state"),
        *common,
    ]
    parsed = parse_supervisor_args(argv)
    config = supervisor_config_from_args(parsed, repo_root=repo)
    program = config.database_program_config()
    assert program.authority_mode == "embedded"
    assert program.task_source_kind == "duckdb"
    assert program.endpoint.endswith("state/control.duckdb") or (
        program.endpoint == "state/control.duckdb"
    )
    assert program.store_id == "store:e2e"
    assert program.generation == 1
    assert program.schema_revision == 2
    assert program.event_store == "events:e2e"
    assert program.runtime_registry == "registry:e2e"
    assert program.export_profile == "markdown-taskboard"

    daemon_command = PortalImplementationSupervisor(config)._build_daemon_command()
    assert daemon_command[daemon_command.index("--task-source-kind") + 1] == (
        "duckdb"
    )
