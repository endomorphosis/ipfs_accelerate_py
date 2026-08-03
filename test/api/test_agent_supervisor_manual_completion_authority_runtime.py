from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor as supervisor_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    build_portal_implementation_daemon_from_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    load_supervisor_scheduler_config,
    supervisor_config_from_args,
)


def _write_scheduler_profile(root: Path) -> Path:
    (root / "config").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "module-a").mkdir()
    (root / "docs" / "objectives.md").write_text(
        "# Objectives\n",
        encoding="utf-8",
    )
    (root / "docs" / "staged.txt").write_text(
        "candidate\n",
        encoding="utf-8",
    )
    (root / "docs" / "tasks.md").write_text(
        """# Tasks

## TEST-001 Staged operator-reviewed task

- Status: todo
- Completion: manual
- Outputs: docs/staged.txt

## TEST-002 Ordinary legacy manual task

- Status: todo
- Completion: manual
""",
        encoding="utf-8",
    )
    profile = root / "config" / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema": (
                    "ipfs_accelerate_py.agent_supervisor."
                    "manual_authority_test.scheduler_config@1"
                ),
                "taskboard_path": "docs/tasks.md",
                "objectives_path": "docs/objectives.md",
                "task_prefix": "## TEST-",
                "board_namespace": "manual-authority-test-v1",
                "merge_target_branch": "main",
                "max_lanes": 2,
                "poll_interval_seconds": 1,
                "daemon_interval_seconds": 1,
                "check_interval_seconds": 1,
                "stale_seconds": 60,
                "max_restarts": 1,
                "max_task_attempts": 1,
                "implementation_timeout_seconds": 60,
                "validation_max_workers": 1,
                "worktree_submodule_paths": ["module-a"],
                "protected_paths": [
                    "docs/tasks.md",
                    "docs/objectives.md",
                    "config/profile.json",
                ],
                "protected_after_manual_completion": {
                    "TEST-001": ["docs/staged.txt"]
                },
                "manual_completion_seals": {},
                "derived_refill": {"enabled_at_bootstrap": False},
                "doctor": {"mutation_authorized": False},
                "rollout": {"automatic_enabled": False},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return profile


def _write_runtime_board(path: Path, *, gated_status: str) -> None:
    path.write_text(
        f"""# Tasks

## TEST-001 Staged operator-reviewed task

- Status: {gated_status}
- Completion: manual
- Priority: P0
- Track: gated

## TEST-002 Dependent task

- Status: todo
- Completion: auto
- Priority: P0
- Track: dependent
- Depends on: TEST-001

## TEST-003 Ordinary legacy manual task

- Status: todo
- Completion: manual
- Priority: P1
- Track: legacy
""",
        encoding="utf-8",
    )


def _daemon(
    root: Path,
    board: Path,
    *,
    suffix: str,
) -> daemon_module.PortalImplementationDaemon:
    state_dir = root / f"state-{suffix}"
    return daemon_module.PortalImplementationDaemon(
        todo_path=board,
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=root,
        task_header_prefix="## TEST-",
        implement=False,
        assumed_completed_task_ids=("TEST-001",),
        manual_completion_authority_required_task_ids=("TEST-001",),
    )


def test_scheduler_passes_only_unverified_staged_manual_tasks_to_daemon(
    tmp_path: Path,
    monkeypatch,
) -> None:
    profile_path = _write_scheduler_profile(tmp_path)
    profile = load_supervisor_scheduler_config(
        profile_path,
        repo_root=tmp_path,
    )

    assert profile["manual_completion_authority_required_task_ids"] == (
        "TEST-001",
    )
    assert "TEST-002" not in profile[
        "manual_completion_authority_required_task_ids"
    ]

    monkeypatch.setattr(supervisor_module, "REPO_ROOT", tmp_path)
    parsed = supervisor_module.parse_args(
        ["--scheduler-config", str(profile_path), "--once"]
    )
    config = supervisor_config_from_args(parsed, repo_root=tmp_path)
    supervisor = PortalImplementationSupervisor(config)
    command = supervisor._build_daemon_command()

    assert config.manual_completion_authority_required_task_ids == (
        "TEST-001",
    )
    option = "--manual-completion-authority-required-task-id"
    assert [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == option
    ] == ["TEST-001"]
    assert supervisor._managed_daemon_matches_command_line(" ".join(command))
    daemon_args = daemon_module.parse_args(command[4:])
    monkeypatch.setattr(
        daemon_module.PortalImplementationDaemon,
        "_git_ref_exists",
        lambda _self, _ref: True,
    )
    daemon, _context = build_portal_implementation_daemon_from_args(
        daemon_args,
        repo_root=tmp_path,
    )
    assert daemon.manual_completion_authority_required_task_ids == frozenset(
        {"TEST-001"}
    )


def test_unverified_manual_status_cannot_complete_or_unlock_dependencies(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.md"
    _write_runtime_board(board, gated_status="completed")
    daemon = _daemon(tmp_path, board, suffix="completed-claim")

    result = daemon.run_once()
    state = daemon_module.PortalTaskState.load(Path(result["state_path"]))

    assert result["completed_count"] == 0
    assert result["quarantined_manual_completion_status_task_ids"] == [
        "TEST-001"
    ]
    assert state.task_statuses["TEST-001"] == "blocked"
    assert state.task_statuses["TEST-002"] == "waiting"
    assert state.task_statuses["TEST-003"] == "ready"
    assert result["active_task_id"] == "TEST-003"


def test_unverified_staged_task_is_not_selected_or_autonomously_completed(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.md"
    _write_runtime_board(board, gated_status="todo")
    daemon = _daemon(tmp_path, board, suffix="pending")

    result = daemon.run_once()
    update = daemon._mark_task_completed_in_todo("TEST-001")

    assert result["manual_completion_authority_required_task_ids"] == [
        "TEST-001"
    ]
    assert result["active_task_id"] == "TEST-003"
    assert update["updated"] is False
    assert update["durable"] is False
    assert update["reason"] == "manual_completion_authority_required"
    assert "- Status: todo" in board.read_text(encoding="utf-8")
