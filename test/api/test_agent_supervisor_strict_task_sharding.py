from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    build_arg_parser as build_multi_supervisor_arg_parser,
    common_args_from_parsed_args,
    implementation_supervisor_common_args,
    tracks_from_parsed_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
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
- Outputs: modules/alpha/src/ready.py
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


def test_strict_task_sharding_disables_cross_lane_ready_fallback(
    tmp_path: Path,
) -> None:
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


def test_strict_shard_idle_reason_ignores_cross_shard_resource_claim(
    tmp_path: Path,
    monkeypatch,
) -> None:
    board = tmp_path / "todo.md"
    _write_cross_shard_board(board)
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon."
        "implementation_daemon.process_command_line",
        lambda _pid: f"python -m pytest {Path(sys.argv[0]).name}",
    )
    common = {
        "todo_path": board,
        "repo_root": tmp_path,
        "task_header_prefix": "## ACCEL-",
        "task_shard_count": 2,
        "strict_task_sharding": True,
        "worktree_submodule_paths": ("modules/alpha",),
    }
    holder = PortalImplementationDaemon(
        **common,
        task_shard_index=1,
        state_path=tmp_path / "holder" / "state.json",
        strategy_path=tmp_path / "holder" / "strategy.json",
        events_path=tmp_path / "holder" / "events.jsonl",
    )
    observer = PortalImplementationDaemon(
        **common,
        task_shard_index=0,
        state_path=tmp_path / "observer" / "state.json",
        strategy_path=tmp_path / "observer" / "strategy.json",
        events_path=tmp_path / "observer" / "events.jsonl",
    )
    borrower = PortalImplementationDaemon(
        **{**common, "strict_task_sharding": False},
        task_shard_index=0,
        state_path=tmp_path / "borrower" / "state.json",
        strategy_path=tmp_path / "borrower" / "strategy.json",
        events_path=tmp_path / "borrower" / "events.jsonl",
    )
    task = PortalTask(
        task_id="ACCEL-001",
        title="Ready task in shard one",
        status="todo",
        completion="manual",
        priority="P1",
        track="ops",
        outputs=["modules/alpha/src/ready.py"],
    )
    claims, unavailable, reason, _existing = (
        holder._acquire_implementation_resource_claims(
            task,
            attempt=1,
            started_at="2026-08-03T00:00:00+00:00",
        )
    )
    assert unavailable == ""
    assert reason == "acquired"

    try:
        result = observer.run_once()
        borrower_result = borrower.run_once()
    finally:
        assert holder._release_implementation_resource_claims(claims)

    assert result["resource_reserved_task_ids"] == ["ACCEL-001"]
    assert result["active_task_id"] == ""
    assert result["selection_idle_reason"] == (
        "no_shard_selectable_ready_tasks"
    )
    assert borrower_result["selection_idle_reason"] == (
        "all_selectable_ready_tasks_deferred_by_resource_claim"
    )


def test_resource_claim_ignores_sibling_worktree_with_same_repository_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    sibling = tmp_path / "sibling"
    repo.mkdir()
    sibling.mkdir()
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon."
        "implementation_daemon.process_command_line",
        lambda _pid: (
            "python -m ipfs_accelerate_py.agent_supervisor.todo_daemon."
            "implementation_daemon"
        ),
    )
    daemon = PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "lane" / "state.json",
        strategy_path=repo / "lane" / "strategy.json",
        events_path=repo / "lane" / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## IPS-",
        implement=True,
        worktree_submodule_paths=("ipfs_datasets_py",),
    )
    daemon.merge_target_repository_id = "repository:shared-accelerate"
    task = PortalTask(
        task_id="IPS-007",
        title="Identities",
        status="todo",
        completion="auto",
        priority="P0",
        track="datasets-identity",
        outputs=[
            "ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/identity.py"
        ],
    )
    claim_path = daemon._implementation_resource_claim_path("ipfs_datasets_py")
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "implementation_resource_claim",
        "lease_id": "foreign-scg",
        "pid": os.getpid(),
        "owner_script": "implementation_daemon.py",
        "repo_root": "",
        "worktree_root": str(sibling.resolve()),
        "repository_id": "repository:shared-accelerate",
        "state_dir": str((sibling / "lane").resolve()),
        "task_id": "SCG-016",
        "board_namespace": "semantic-compression-governor-v1",
        "resource_kind": "submodule",
        "resource_path": "ipfs_datasets_py",
    }
    claim_path.write_text(json.dumps(payload), encoding="utf-8")

    assert (
        daemon._implementation_resource_claim_owner_is_active(payload) is False
    )
    assert daemon._active_implementation_resource_claims([task]) == {}

    local = dict(payload)
    local["worktree_root"] = str(repo.resolve())
    local["task_id"] = "IPS-006"
    claim_path.write_text(json.dumps(local), encoding="utf-8")
    assert daemon._implementation_resource_claim_owner_is_active(local) is True
    assert "ipfs_datasets_py" in daemon._active_implementation_resource_claims(
        [task]
    )


def test_daemon_cli_and_builder_propagate_strict_task_sharding(
    tmp_path: Path,
) -> None:
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
) -> None:
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


def test_multi_supervisor_wrapper_propagates_strict_task_sharding() -> None:
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
    assert (
        implementation_supervisor_common_args(
            strict_task_sharding=True
        ).count("--strict-task-sharding")
        == 1
    )

    defaults = build_multi_supervisor_arg_parser().parse_args(
        ["--implementation-track", "T|worker.py|state|agent"]
    )
    assert "--strict-task-sharding" not in common_args_from_parsed_args(defaults)
