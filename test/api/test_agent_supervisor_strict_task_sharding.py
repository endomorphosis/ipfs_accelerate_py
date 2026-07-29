from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    build_arg_parser as build_multi_supervisor_arg_parser,
    common_args_from_parsed_args,
    implementation_supervisor_common_args,
    tracks_from_parsed_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTaskState,
    parse_args as parse_daemon_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    build_portal_implementation_daemon_from_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    parse_args as parse_supervisor_args,
    supervisor_config_from_args,
)


def _write_cross_shard_board(path: Path) -> None:
    path.write_text(
        """# Agent Todos

## ACCEL-000 Blocked task in shard zero

- Status: blocked
- Completion: manual
- Priority: P1
- Track: ops

## ACCEL-001 Ready task in shard one

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
""",
        encoding="utf-8",
    )


def _daemon(
    repo: Path,
    board: Path,
    state_dir: Path,
    *,
    strict_task_sharding: bool = False,
) -> PortalImplementationDaemon:
    return PortalImplementationDaemon(
        todo_path=board,
        state_path=state_dir / "state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=strict_task_sharding,
    )


def test_strict_task_sharding_disables_cross_lane_ready_fallback(tmp_path: Path):
    board = tmp_path / "todo.md"
    _write_cross_shard_board(board)

    legacy_dir = tmp_path / "legacy"
    strict_dir = tmp_path / "strict"
    legacy = _daemon(tmp_path, board, legacy_dir)
    strict = _daemon(
        tmp_path,
        board,
        strict_dir,
        strict_task_sharding=True,
    )

    legacy_result = legacy.run_once()
    strict_result = strict.run_once()
    strict_state = PortalTaskState.load(strict_dir / "state.json")

    assert legacy.strict_task_sharding is False
    assert legacy_result["active_task_id"] == "ACCEL-001"
    assert "task_shard_ready_fallback" in (
        legacy_dir / "events.jsonl"
    ).read_text(encoding="utf-8")

    assert strict.strict_task_sharding is True
    assert strict_result["active_task_id"] == ""
    assert strict_result["selection_idle_reason"] == "no_shard_selectable_ready_tasks"
    assert strict_state.ready_task_ids == ["ACCEL-001"]
    assert strict_state.selectable_ready_task_ids == []
    assert "task_shard_ready_fallback" not in (
        strict_dir / "events.jsonl"
    ).read_text(encoding="utf-8")


def test_daemon_cli_and_builder_propagate_strict_task_sharding(tmp_path: Path):
    board = tmp_path / "todo.md"
    _write_cross_shard_board(board)

    default_args = parse_daemon_args(["--todo-path", str(board)])
    strict_args = parse_daemon_args(
        [
            "--todo-path",
            str(board),
            "--state-dir",
            str(tmp_path / "state"),
            "--task-shard-count",
            "2",
            "--task-shard-index",
            "0",
            "--strict-task-sharding",
        ]
    )
    daemon, _context = build_portal_implementation_daemon_from_args(
        strict_args,
        repo_root=tmp_path,
    )

    assert default_args.strict_task_sharding is False
    assert strict_args.strict_task_sharding is True
    assert daemon.strict_task_sharding is True


def test_supervisor_propagates_strict_task_sharding_to_managed_daemon(
    tmp_path: Path,
):
    board = tmp_path / "todo.md"
    _write_cross_shard_board(board)
    parsed = parse_supervisor_args(
        [
            "--todo-path",
            str(board),
            "--state-dir",
            str(tmp_path / "state"),
            "--task-shard-count",
            "2",
            "--task-shard-index",
            "0",
            "--strict-task-sharding",
        ]
    )

    config = supervisor_config_from_args(parsed, repo_root=tmp_path)
    supervisor = PortalImplementationSupervisor(config)
    command = supervisor._build_daemon_command()

    assert config.strict_task_sharding is True
    assert command.count("--strict-task-sharding") == 1
    assert supervisor._managed_daemon_matches_command_line(" ".join(command))
    assert not supervisor._managed_daemon_matches_command_line(
        " ".join(part for part in command if part != "--strict-task-sharding")
    )

    default_config = supervisor_config_from_args(
        parse_supervisor_args(
            [
                "--todo-path",
                str(board),
                "--state-dir",
                str(tmp_path / "default-state"),
            ]
        ),
        repo_root=tmp_path,
    )
    assert default_config.strict_task_sharding is False
    default_supervisor = PortalImplementationSupervisor(default_config)
    default_command = default_supervisor._build_daemon_command()
    assert "--strict-task-sharding" not in default_command
    assert default_supervisor._managed_daemon_matches_command_line(
        " ".join(default_command)
    )


def test_multi_supervisor_wrapper_propagates_strict_task_sharding():
    parsed = build_multi_supervisor_arg_parser().parse_args(
        [
            "--implementation-track",
            "T|worker.py|state|agent",
            "--implementation-supervisor-lanes-per-track",
            "2",
            "--implementation-supervisor-strict-task-sharding",
        ]
    )

    common_args = common_args_from_parsed_args(parsed)
    tracks = tracks_from_parsed_args(parsed)

    assert common_args == ["--strict-task-sharding"]
    assert len(tracks) == 2
    assert all("--task-shard-count" in track.extra_args for track in tracks)
    assert implementation_supervisor_common_args(
        strict_task_sharding=True
    ).count("--strict-task-sharding") == 1

    defaults = build_multi_supervisor_arg_parser().parse_args(
        ["--implementation-track", "T|worker.py|state|agent"]
    )
    assert "--strict-task-sharding" not in common_args_from_parsed_args(defaults)
