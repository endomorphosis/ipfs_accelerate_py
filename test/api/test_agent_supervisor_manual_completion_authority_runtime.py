from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def _write_revalidation_board(
    path: Path,
    *,
    descendant_status: str,
    descendant_title: str = "Authority descendant",
    validation: str = "python -c 'raise SystemExit(0)'",
    depends_on: str = "TEST-001",
) -> None:
    path.write_text(
        f"""# Tasks

## TEST-001 Activated operator-reviewed task

- Status: completed
- Completion: manual
- Priority: P0

## TEST-002 {descendant_title}

- Status: {descendant_status}
- Completion: artifact
- Priority: P0
- Depends on: {depends_on}
- Validation: {validation}
""",
        encoding="utf-8",
    )


def _revalidation_daemon(
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
        manual_completion_authority_task_ids=("TEST-001",),
        manual_completion_authority_epoch_id="seal-epoch-current",
    )


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _git_revalidation_repo(
    tmp_path: Path,
    *,
    descendants: list[tuple[str, str, str]],
) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Authority Test")
    _git(repo, "config", "user.email", "authority@example.invalid")
    sections = [
        """## TEST-001 Activated root

- Status: completed
- Completion: manual
- Priority: P0
"""
    ]
    for task_id, status, depends_on in descendants:
        sections.append(
            f"""## {task_id} Authority descendant {task_id}

- Status: {status}
- Completion: manual
- Priority: P0
- Depends on: {depends_on}
- Validation: python -c 'raise SystemExit(0)'
"""
        )
    board = repo / "tasks.md"
    board.write_text("# Tasks\n\n" + "\n".join(sections), encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    return repo, board


def _implementation_revalidation_daemon(
    tmp_path: Path,
    repo: Path,
    board: Path,
    *,
    suffix: str,
    shard_count: int = 1,
    shard_index: int = 0,
    max_task_attempts: int = 1,
    revalidation_only: bool = False,
) -> daemon_module.PortalImplementationDaemon:
    state_dir = tmp_path / f"state-{suffix}"
    return daemon_module.PortalImplementationDaemon(
        todo_path=board,
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## TEST-",
        implement=True,
        use_ephemeral_worktree=False,
        worktree_root=tmp_path / "worktrees",
        merge_queue_dir=tmp_path / "merge-queue",
        validation_cache_dir=tmp_path / "validation-cache",
        worktree_pool_enabled=True,
        max_task_attempts=max_task_attempts,
        task_shard_count=shard_count,
        task_shard_index=shard_index,
        strict_task_sharding=shard_count > 1,
        manual_completion_authority_task_ids=("TEST-001",),
        manual_completion_authority_epoch_id="seal-epoch-current",
        manual_completion_authority_revalidation_only=revalidation_only,
    )


def _forbid_revalidation_provider_and_seeding(
    daemon: daemon_module.PortalImplementationDaemon,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        pytest.fail("authority revalidation consulted provider/context/seeding")

    for name in (
        "_build_implementation_prompt",
        "_build_implementation_command",
        "_compile_implementation_context",
        "_task_declared_implementation_provider",
        "_task_context_token_limit",
        "_task_uses_typed_local_execution",
        "_active_provider_capacity_backoff_for_task",
        "_prior_attempt_seed_plan",
        "_apply_prior_attempt_seed",
        "_link_shared_worktree_paths",
        "_seed_untracked_worktree_context",
        "_seed_operator_prepared_outputs",
    ):
        monkeypatch.setattr(daemon, name, forbidden)
    monkeypatch.setattr(
        daemon_module,
        "run_process_group_stream",
        forbidden,
    )


def _fresh_revalidation_evidence(
    daemon: daemon_module.PortalImplementationDaemon,
    root: Path,
    *,
    log_name: str,
) -> tuple[str, dict[str, object]]:
    guard = daemon._refresh_manual_completion_authority_guard()
    assert guard["available"] is True
    context_id = daemon._manual_completion_authority_policy_id()
    task = {
        item.task_id: item for item in daemon._load_tasks()
    }["TEST-002"]
    evidence = daemon._run_validation_commands(
        root,
        task,
        root / log_name,
    )
    assert evidence["passed"] is True
    return context_id, evidence


def _complete_with_fresh_revalidation(
    daemon: daemon_module.PortalImplementationDaemon,
    root: Path,
    *,
    log_name: str,
) -> tuple[str, dict[str, object], dict[str, object]]:
    context_id, evidence = _fresh_revalidation_evidence(
        daemon,
        root,
        log_name=log_name,
    )
    update = daemon._mark_tasks_completed_in_todo(
        ["TEST-002"],
        primary_task_id="TEST-002",
        completion_reason="fresh_authority_revalidation",
        manual_completion_authority_context_id=context_id,
        manual_completion_authority_evidence=evidence,
    )
    assert update["updated"] is True, update
    return context_id, evidence, update


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
    assert profile["manual_completion_authority_task_ids"] == (
        "TEST-001",
    )
    assert profile["manual_completion_authority_epoch_id"].startswith("b")
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
    assert config.manual_completion_authority_task_ids == (
        "TEST-001",
    )
    assert (
        config.manual_completion_authority_epoch_id
        == profile["manual_completion_authority_epoch_id"]
    )
    scope_option = "--manual-completion-authority-task-id"
    assert [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == scope_option
    ] == ["TEST-001"]
    epoch_option = "--manual-completion-authority-epoch-id"
    assert command[command.index(epoch_option) + 1] == profile[
        "manual_completion_authority_epoch_id"
    ]
    option = "--manual-completion-authority-required-task-id"
    assert [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == option
    ] == ["TEST-001"]
    assert supervisor._managed_daemon_matches_command_line(" ".join(command))
    stale_epoch_command = list(command)
    stale_epoch_command[stale_epoch_command.index(epoch_option) + 1] = (
        "stale-seal-epoch"
    )
    assert not supervisor._managed_daemon_matches_command_line(
        " ".join(stale_epoch_command)
    )
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
    assert daemon.manual_completion_authority_task_ids == frozenset(
        {"TEST-001"}
    )
    assert (
        daemon.manual_completion_authority_epoch_id
        == profile["manual_completion_authority_epoch_id"]
    )


def test_live_status_revocation_requarantines_activated_root(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.md"
    _write_runtime_board(board, gated_status="completed")
    state_dir = tmp_path / "state-live-revocation"
    daemon = daemon_module.PortalImplementationDaemon(
        todo_path=board,
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=tmp_path,
        task_header_prefix="## TEST-",
        implement=False,
        manual_completion_authority_task_ids=("TEST-001",),
        manual_completion_authority_epoch_id="seal-epoch-a",
    )
    activated_context_id = daemon._manual_completion_authority_policy_id()

    _write_runtime_board(board, gated_status="todo")
    result = daemon.run_once()
    revoked_context_id = daemon._manual_completion_authority_policy_id()
    update = daemon._mark_task_completed_in_todo("TEST-001")

    # Configured required roots (and thus durable receipt policy context) stay
    # stable when a previously-activated root flips live status.  Hard-block
    # and revalidation sets still expand from live revocation so dependents
    # cannot complete without a fresh authority path.
    assert activated_context_id == revoked_context_id
    assert result["manual_completion_authority_required_task_ids"] == [
        "TEST-001"
    ]
    assert result["manual_completion_authority_dependency_task_ids"] == [
        "TEST-002"
    ]
    assert result["active_task_id"] == "TEST-003"
    assert update["updated"] is False
    assert update["reason"] == "manual_completion_authority_required"


def test_completed_descendant_without_current_epoch_receipt_is_revalidated(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.md"
    board.write_text(
        """# Tasks

## TEST-001 Activated operator-reviewed task

- Status: completed
- Completion: manual
- Priority: P0

## TEST-002 Pre-seal completed descendant

- Status: completed
- Completion: artifact
- Priority: P0
- Depends on: TEST-001
- Validation: python -c 'raise SystemExit(0)'
""",
        encoding="utf-8",
    )
    state_dir = tmp_path / "state-preseal-completed-descendant"
    daemon = daemon_module.PortalImplementationDaemon(
        todo_path=board,
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=tmp_path,
        task_header_prefix="## TEST-",
        implement=False,
        manual_completion_authority_task_ids=("TEST-001",),
        manual_completion_authority_epoch_id="seal-epoch-current",
    )

    result = daemon.run_once()
    state = daemon_module.PortalTaskState.load(Path(result["state_path"]))

    assert result["completed_count"] == 1
    assert result["manual_completion_revalidation_task_ids"] == [
        "TEST-002"
    ]
    assert result["quarantined_manual_completion_status_task_ids"] == [
        "TEST-002"
    ]
    assert result["active_task_id"] == "TEST-002"
    assert state.task_statuses["TEST-002"] == "ready"


def test_cross_board_candidate_requires_explicit_authority_metadata(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.md"
    _write_runtime_board(board, gated_status="completed")
    state_dir = tmp_path / "state-cross-board-metadata"
    daemon = daemon_module.PortalImplementationDaemon(
        todo_path=board,
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=tmp_path,
        task_header_prefix="## TEST-",
        implement=False,
        manual_completion_authority_task_ids=("TEST-001",),
        manual_completion_authority_epoch_id="seal-epoch-current",
    )
    request = SimpleNamespace(
        metadata={
            "target_binding_schema": daemon_module.MERGE_TARGET_BINDING_SCHEMA,
            "target_repository_id": daemon.merge_target_repository_id,
            "target_branch": daemon.resolved_merge_target_branch,
            "todo_path": str(tmp_path / "other-tasks.md"),
            "task": {
                "task_id": "OTHER-001",
                "title": "legacy cross-board task",
                "status": "todo",
            },
        },
        target_repository_id=daemon.merge_target_repository_id,
        target_branch=daemon.resolved_merge_target_branch,
        task_id="OTHER-001",
        priority="P0",
        branch_name="agent/other-001",
        commit_sha="deadbeef",
        attempt=1,
    )

    result = daemon._merge_train_callback(request)

    assert result["attempted"] is False
    assert result["reason"] == (
        "cross_board_manual_completion_authority_metadata_missing"
    )
    assert result["missing_metadata_fields"] == [
        "manual_completion_authority_context_id",
        "manual_completion_authority_epoch_id",
        "manual_completion_authority_required_task_ids",
        "manual_completion_authority_revocation_generation",
        "manual_completion_authority_task_ids",
    ]


def test_one_descendant_validation_cannot_authorize_sibling_completion(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.md"
    board.write_text(
        """# Tasks

## TEST-001 Activated operator-reviewed task

- Status: completed
- Completion: manual
- Priority: P0

## TEST-002 First descendant

- Status: todo
- Completion: artifact
- Priority: P0
- Depends on: TEST-001
- Validation: python -c 'raise SystemExit(0)'

## TEST-003 Second descendant

- Status: todo
- Completion: artifact
- Priority: P0
- Depends on: TEST-001
- Validation: python -c 'raise SystemExit(0)'
""",
        encoding="utf-8",
    )
    state_dir = tmp_path / "state-sibling-revalidation"
    daemon = daemon_module.PortalImplementationDaemon(
        todo_path=board,
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=tmp_path,
        task_header_prefix="## TEST-",
        implement=False,
        manual_completion_authority_task_ids=("TEST-001",),
        manual_completion_authority_epoch_id="seal-epoch-current",
    )
    daemon._refresh_manual_completion_authority_guard()
    context_id = daemon._manual_completion_authority_policy_id()
    rejection = daemon._manual_completion_authority_rejection(
        ["TEST-002", "TEST-003"],
        authority_context_id=context_id,
        authority_evidence={
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "manual_completion_authority_context_id": context_id,
            "manual_completion_authority_revalidation": True,
            "manual_completion_authority_force_uncached": True,
            "manual_completion_authority_task_id": "TEST-002",
            "manual_completion_authority_validation_result_count": 1,
            "results": [
                {
                    "command": "python -c 'raise SystemExit(0)'",
                    "returncode": 0,
                    "cache_hit": False,
                    "timed_out": False,
                    "validation_result_digest": "a" * 64,
                }
            ],
        },
    )

    assert rejection is not None
    assert rejection["reason"] == (
        "manual_completion_authority_revalidation_required"
    )
    assert rejection["manual_completion_authority_evidence_valid"] is False
    assert rejection[
        "manual_completion_authority_revalidation_task_ids"
    ] == ["TEST-002", "TEST-003"]


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


def test_stale_completion_sources_cannot_bypass_manual_dependency_closure(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.md"
    board.write_text(
        """# Tasks

## TEST-001 Staged operator-reviewed task

- Status: completed
- Completion: manual
- Priority: P0
- Track: gated

## TEST-002 Direct dependent task

- Status: todo
- Completion: artifact
- Priority: P0
- Track: dependent
- Depends on: TEST-001
- Outputs: direct.txt

## TEST-003 Transitive dependent task

- Status: todo
- Completion: artifact
- Priority: P0
- Track: dependent
- Depends on: TEST-002
- Outputs: transitive.txt
""",
        encoding="utf-8",
    )
    (tmp_path / "direct.txt").write_text("old\n", encoding="utf-8")
    (tmp_path / "transitive.txt").write_text("old\n", encoding="utf-8")
    daemon = _daemon(tmp_path, board, suffix="stale-descendants")
    tasks = {task.task_id: task for task in daemon._load_tasks()}
    stale_ids = {"TEST-002", "TEST-003"}
    daemon._shared_completed_task_cid_bindings = lambda: {  # type: ignore[method-assign]
        task_id: {daemon._canonical_ref(tasks[task_id])}
        for task_id in stale_ids
    }
    daemon._successfully_merged_task_ids = lambda: set(stale_ids)  # type: ignore[method-assign]

    result = daemon.run_once()
    state = daemon_module.PortalTaskState.load(Path(result["state_path"]))

    assert result["completed_count"] == 0
    assert result["shared_completed_task_ids"] == []
    assert result["merged_status_repair"] == {}
    assert result["manual_completion_authority_dependency_task_ids"] == [
        "TEST-002",
        "TEST-003",
    ]
    assert result["manual_completion_revalidation_task_ids"] == [
        "TEST-002",
        "TEST-003",
    ]
    assert state.task_statuses == {
        "TEST-001": "blocked",
        "TEST-002": "waiting",
        "TEST-003": "waiting",
    }
    direct_update = daemon._mark_tasks_completed_in_todo(
        ["TEST-002"],
        primary_task_id="TEST-002",
        completion_reason="stale_callback",
    )
    assert direct_update["updated"] is False
    assert direct_update["durable"] is False
    assert direct_update["reason"] == (
        "manual_completion_authority_dependency_required"
    )
    assert direct_update["manual_completion_authority_blocked_task_ids"] == [
        "TEST-002"
    ]
    assert "- Status: todo" in board.read_text(encoding="utf-8")


def test_merge_callback_rejects_gated_descendant_before_git_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    board = tmp_path / "tasks.md"
    _write_runtime_board(board, gated_status="todo")
    daemon = _daemon(tmp_path, board, suffix="merge-descendant")
    merge_called = False

    def unexpected_merge(*_args, **_kwargs):
        nonlocal merge_called
        merge_called = True
        raise AssertionError("gated candidate reached Git mutation")

    monkeypatch.setattr(daemon, "_merge_branch_to_main", unexpected_merge)
    request = SimpleNamespace(
        metadata={
            "target_binding_schema": daemon_module.MERGE_TARGET_BINDING_SCHEMA,
            "target_repository_id": daemon.merge_target_repository_id,
            "target_branch": daemon.resolved_merge_target_branch,
            "task": {
                "task_id": "TEST-002",
                "title": "stale dependent candidate",
                "status": "todo",
                "completion": "auto",
                "priority": "P0",
                "track": "dependent",
                "depends_on": ["TEST-001"],
            },
        },
        target_repository_id=daemon.merge_target_repository_id,
        target_branch=daemon.resolved_merge_target_branch,
        task_id="TEST-002",
        priority="P0",
        branch_name="agent/test-002",
        commit_sha="deadbeef",
        attempt=1,
    )

    result = daemon._merge_train_callback(request)

    assert result["attempted"] is False
    assert result["merged"] is False
    assert result["reason"] == (
        "manual_completion_authority_dependency_required"
    )
    assert merge_called is False


def test_assumed_completed_goal_cannot_unlock_authority_affected_task(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.md"
    board.write_text(
        """# Tasks

## TEST-001 Staged operator-reviewed task

- Status: todo
- Completion: manual
- Priority: P0
- Track: gated
- Goal ID: GOAL-GATED

## TEST-002 Goal-dependent task

- Status: todo
- Completion: auto
- Priority: P0
- Track: dependent
- Depends on: GOAL-GATED
""",
        encoding="utf-8",
    )
    state_dir = tmp_path / "state-goal-reference"
    daemon = daemon_module.PortalImplementationDaemon(
        todo_path=board,
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=tmp_path,
        task_header_prefix="## TEST-",
        implement=False,
        assumed_completed_task_ids=("GOAL-GATED",),
        manual_completion_authority_task_ids=("TEST-001",),
        manual_completion_authority_required_task_ids=("TEST-001",),
    )

    result = daemon.run_once()
    state = daemon_module.PortalTaskState.load(Path(result["state_path"]))

    assert result["manual_completion_authority_affected_goal_ids"] == [
        "GOAL-GATED"
    ]
    assert state.assumed_completed_task_ids == []
    assert state.task_statuses == {
        "TEST-001": "blocked",
        "TEST-002": "waiting",
    }


def test_pending_descendant_requires_fresh_revalidation_after_manual_activation(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.md"
    board.write_text(
        """# Tasks

## TEST-001 Activated operator-reviewed task

- Status: completed
- Completion: manual
- Priority: P0
- Track: gated

## TEST-002 Pending dependent task

- Status: todo
- Completion: artifact
- Priority: P0
- Track: dependent
- Depends on: TEST-001
- Outputs: dependent.txt
- Validation: python -c 'raise SystemExit(0)'
""",
        encoding="utf-8",
    )
    (tmp_path / "dependent.txt").write_text("old\n", encoding="utf-8")
    state_dir = tmp_path / "state-activated"
    daemon = daemon_module.PortalImplementationDaemon(
        todo_path=board,
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=tmp_path,
        task_header_prefix="## TEST-",
        implement=False,
        manual_completion_authority_task_ids=("TEST-001",),
        manual_completion_authority_epoch_id="seal-epoch-a",
    )
    tasks = {task.task_id: task for task in daemon._load_tasks()}
    stale_cid = daemon._canonical_ref(tasks["TEST-002"])
    daemon._shared_completed_task_cid_bindings = lambda: {  # type: ignore[method-assign]
        "TEST-002": {stale_cid}
    }
    daemon._successfully_merged_task_ids = lambda: {"TEST-002"}  # type: ignore[method-assign]

    result = daemon.run_once()
    state = daemon_module.PortalTaskState.load(Path(result["state_path"]))

    assert result["completed_count"] == 1
    assert result["active_task_id"] == "TEST-002"
    assert result["shared_completed_task_ids"] == []
    assert result["merged_status_repair"] == {}
    assert result["manual_completion_authority_dependency_task_ids"] == []
    assert result["manual_completion_revalidation_task_ids"] == ["TEST-002"]
    assert state.task_statuses == {
        "TEST-001": "completed",
        "TEST-002": "ready",
    }
    assert "- Status: todo" in board.read_text(encoding="utf-8")

    preseal_state_dir = tmp_path / "state-preseal-context"
    preseal_daemon = daemon_module.PortalImplementationDaemon(
        todo_path=board,
        state_path=preseal_state_dir / "task-state.json",
        strategy_path=preseal_state_dir / "strategy.json",
        events_path=preseal_state_dir / "events.jsonl",
        repo_root=tmp_path,
        task_header_prefix="## TEST-",
        implement=False,
        manual_completion_authority_task_ids=("TEST-001",),
        manual_completion_authority_required_task_ids=("TEST-001",),
        manual_completion_authority_epoch_id="seal-epoch-a",
    )
    stale_context_id = (
        preseal_daemon._manual_completion_authority_policy_id()
    )
    current_context_id = daemon._manual_completion_authority_policy_id()
    assert stale_context_id != current_context_id

    stale_update = daemon._mark_tasks_completed_in_todo(
        ["TEST-002"],
        primary_task_id="TEST-002",
        completion_reason="preseal_candidate",
        manual_completion_authority_context_id=stale_context_id,
    )
    assert stale_update["updated"] is False
    assert stale_update["reason"] == (
        "manual_completion_authority_revalidation_required"
    )
    assert "- Status: todo" in board.read_text(encoding="utf-8")

    hash_only_update = daemon._mark_tasks_completed_in_todo(
        ["TEST-002"],
        primary_task_id="TEST-002",
        completion_reason="hash_only_not_validation",
        manual_completion_authority_context_id=current_context_id,
    )
    assert hash_only_update["updated"] is False
    assert hash_only_update[
        "manual_completion_authority_evidence_valid"
    ] is False

    structurally_incomplete_update = daemon._mark_tasks_completed_in_todo(
        ["TEST-002"],
        primary_task_id="TEST-002",
        completion_reason="incomplete_validation_record",
        manual_completion_authority_context_id=current_context_id,
        manual_completion_authority_evidence={
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "manual_completion_authority_context_id": current_context_id,
            "manual_completion_authority_revalidation": True,
            "manual_completion_authority_force_uncached": True,
            "manual_completion_authority_task_id": "TEST-002",
            "manual_completion_authority_validation_result_count": 1,
            "results": [{}],
        },
    )
    assert structurally_incomplete_update["updated"] is False
    assert structurally_incomplete_update[
        "manual_completion_authority_evidence_valid"
    ] is False

    current_task = {
        task.task_id: task for task in daemon._load_tasks()
    }["TEST-002"]
    validation_result = daemon._run_validation_commands(
        tmp_path,
        current_task,
        tmp_path / "fresh-revalidation.log",
    )
    assert validation_result["attempted"] is True
    assert validation_result["passed"] is True
    assert validation_result[
        "manual_completion_authority_force_uncached"
    ] is True
    assert validation_result[
        "manual_completion_authority_context_id"
    ] == current_context_id
    assert validation_result["results"]
    assert all(
        item.get("cache_hit") is not True
        for item in validation_result["results"]
    )
    assert validation_result[
        "manual_completion_authority_validation_result_count"
    ] == len(validation_result["results"])

    evidence_without_context = dict(validation_result)
    evidence_without_context.pop("manual_completion_authority_context_id")
    substituted_context_update = daemon._mark_tasks_completed_in_todo(
        ["TEST-002"],
        primary_task_id="TEST-002",
        completion_reason="top_level_context_substitution",
        manual_completion_authority_context_id=current_context_id,
        manual_completion_authority_evidence=evidence_without_context,
    )
    assert substituted_context_update["updated"] is False
    assert substituted_context_update[
        "actual_manual_completion_authority_context_id"
    ] == ""

    resealed_state_dir = tmp_path / "state-resealed-context"
    resealed_daemon = daemon_module.PortalImplementationDaemon(
        todo_path=board,
        state_path=resealed_state_dir / "task-state.json",
        strategy_path=resealed_state_dir / "strategy.json",
        events_path=resealed_state_dir / "events.jsonl",
        repo_root=tmp_path,
        task_header_prefix="## TEST-",
        implement=False,
        manual_completion_authority_task_ids=("TEST-001",),
        manual_completion_authority_epoch_id="seal-epoch-b",
    )
    resealed_context_id = (
        resealed_daemon._manual_completion_authority_policy_id()
    )
    stale_reseal_update = resealed_daemon._mark_tasks_completed_in_todo(
        ["TEST-002"],
        primary_task_id="TEST-002",
        completion_reason="stale_pre_reseal_validation",
        manual_completion_authority_context_id=current_context_id,
        manual_completion_authority_evidence=validation_result,
    )
    assert resealed_context_id != current_context_id
    assert stale_reseal_update["updated"] is False
    assert stale_reseal_update[
        "expected_manual_completion_authority_context_id"
    ] == resealed_context_id

    fresh_update = daemon._mark_tasks_completed_in_todo(
        ["TEST-002"],
        primary_task_id="TEST-002",
        completion_reason="fresh_postseal_validation",
        manual_completion_authority_context_id=current_context_id,
        manual_completion_authority_evidence=validation_result,
    )
    assert fresh_update["updated"] is True, fresh_update
    assert daemon._todo_completion_is_durable(fresh_update)
    receipt_result = fresh_update[
        "manual_completion_authority_revalidation_receipt"
    ]
    assert receipt_result["persisted"] is True
    assert receipt_result["task_ids"] == ["TEST-002"]
    assert "- Status: completed" in board.read_text(encoding="utf-8")

    post_revalidation = daemon.run_once()

    assert post_revalidation["completed_count"] == 2
    assert post_revalidation[
        "manual_completion_revalidation_task_ids"
    ] == []
    receipt_guard = daemon._refresh_manual_completion_authority_guard()
    assert receipt_guard[
        "revalidation_receipt_task_ids"
    ] == ["TEST-002"]
    assert post_revalidation[
        "quarantined_manual_completion_status_task_ids"
    ] == []


def test_revocation_generation_blocks_completed_todo_completed_aba_replay(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.md"
    _write_revalidation_board(board, descendant_status="todo")
    daemon = _revalidation_daemon(
        tmp_path,
        board,
        suffix="aba-generation",
    )
    original_context, _evidence, original_update = (
        _complete_with_fresh_revalidation(
            daemon,
            tmp_path,
            log_name="aba-validation.log",
        )
    )
    original_generation = (
        daemon._manual_completion_authority_revocation_generation
    )
    original_receipt_id = original_update[
        "manual_completion_authority_revalidation_receipt"
    ]["receipt_ids"]["TEST-002"]
    store_path = daemon._manual_completion_revalidation_store_path()
    original_store_bytes = store_path.read_bytes()

    _write_revalidation_board(board, descendant_status="todo")
    revoked_guard = daemon._refresh_manual_completion_authority_guard()
    revoked_context = daemon._manual_completion_authority_policy_id()

    assert revoked_guard["available"] is True
    assert revoked_guard["revocation_guard"]["revoked_task_ids"] == [
        "TEST-002"
    ]
    assert (
        daemon._manual_completion_authority_revocation_generation
        == original_generation + 1
    )
    assert revoked_context != original_context

    _write_revalidation_board(board, descendant_status="completed")
    replay_guard = daemon._refresh_manual_completion_authority_guard()

    assert replay_guard["available"] is True
    assert replay_guard["revalidation_receipt_task_ids"] == []
    assert replay_guard["revalidation_task_ids"] == ["TEST-002"]
    assert replay_guard["revocation_generation"] == original_generation + 1

    # A same-UID writer may restore a byte-for-byte valid old store.  The
    # in-process generation floor must treat that as rollback, not reactivate
    # the still-known receipt ID.
    store_path.write_bytes(original_store_bytes)
    rollback_guard = daemon._refresh_manual_completion_authority_guard()
    rollback_state = rollback_guard["revocation_guard"]

    assert rollback_state["rollback_detected"] is True
    assert rollback_state["revocation_generation"] >= original_generation + 2
    assert original_receipt_id not in (
        daemon._trusted_manual_completion_revalidation_receipt_ids
    )
    assert rollback_guard["revalidation_receipt_task_ids"] == []

    restarted = _revalidation_daemon(
        tmp_path,
        board,
        suffix="aba-generation-restart",
    )
    restarted_result = restarted.run_once()

    assert (
        restarted._manual_completion_authority_revocation_generation
        == rollback_state["revocation_generation"]
    )
    assert restarted_result["manual_completion_revalidation_task_ids"] == [
        "TEST-002"
    ]
    assert restarted_result[
        "quarantined_manual_completion_status_task_ids"
    ] == ["TEST-002"]


def test_restart_and_same_uid_disk_forgery_lose_receipt_trust(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.md"
    _write_revalidation_board(board, descendant_status="todo")
    daemon = _revalidation_daemon(
        tmp_path,
        board,
        suffix="receipt-producer",
    )
    _context, _evidence, update = _complete_with_fresh_revalidation(
        daemon,
        tmp_path,
        log_name="receipt-producer-validation.log",
    )
    receipt_id = update[
        "manual_completion_authority_revalidation_receipt"
    ]["receipt_ids"]["TEST-002"]

    restarted = _revalidation_daemon(
        tmp_path,
        board,
        suffix="receipt-consumer-restart",
    )
    restart_guard = restarted._refresh_manual_completion_authority_guard()

    # Durable self-consistent receipts are cold-start re-admitted so a
    # supervisor restart does not rewalk the full revalidation DAG.
    assert receipt_id in (
        restarted._trusted_manual_completion_revalidation_receipt_ids
    )
    assert restart_guard["revalidation_receipt_task_ids"] == ["TEST-002"]
    assert "TEST-002" not in restart_guard["revalidation_task_ids"]

    store_path = daemon._manual_completion_revalidation_store_path()
    store = json.loads(store_path.read_text(encoding="utf-8"))
    forged_receipt = dict(store["records"]["TEST-002"])
    forged_receipt.pop("receipt_id")
    forged_receipt["validation_evidence_id"] = daemon_module.content_identity(
        {"forged_by_same_uid_disk_writer": True}
    )
    forged_receipt_id = daemon_module.content_identity(forged_receipt)
    store_body = dict(store)
    store_body.pop("store_id")
    store_body["records"] = {
        **dict(store_body["records"]),
        "TEST-002": {
            **forged_receipt,
            "receipt_id": forged_receipt_id,
        },
    }
    forged_store = {
        **store_body,
        "store_id": daemon_module.content_identity(store_body),
    }
    store_path.write_text(
        json.dumps(forged_store, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    forged_guard = daemon._refresh_manual_completion_authority_guard()

    assert forged_receipt_id != receipt_id
    assert forged_receipt_id not in (
        daemon._trusted_manual_completion_revalidation_receipt_ids
    )
    assert forged_guard["revalidation_receipt_task_ids"] == []
    assert forged_guard["revalidation_receipt_guard"][
        "invalid_task_ids"
    ] == ["TEST-002"]

    # Deleting a generation-zero store must still invalidate live producer
    # trust; otherwise restoring the original valid bytes could revive it.
    store_path.unlink()
    missing_guard = daemon._refresh_manual_completion_authority_guard()
    assert missing_guard["revocation_guard"]["rollback_detected"] is True
    assert receipt_id not in (
        daemon._trusted_manual_completion_revalidation_receipt_ids
    )
    store_path.write_text(
        json.dumps(store, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    restored_guard = daemon._refresh_manual_completion_authority_guard()
    assert restored_guard["revocation_guard"]["rollback_detected"] is True
    assert restored_guard["revalidation_receipt_task_ids"] == []


def test_revalidation_evidence_rejects_task_plan_tree_and_strict_field_rebinds(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.md"
    _write_revalidation_board(board, descendant_status="todo")
    daemon = _revalidation_daemon(
        tmp_path,
        board,
        suffix="evidence-bindings",
    )
    context_id, evidence = _fresh_revalidation_evidence(
        daemon,
        tmp_path,
        log_name="evidence-binding-validation.log",
    )

    def rejection_for(candidate: dict[str, object]) -> dict[str, object]:
        rejection = daemon._manual_completion_authority_rejection(
            ["TEST-002"],
            authority_context_id=context_id,
            authority_evidence=candidate,
        )
        assert rejection is not None
        assert rejection["manual_completion_authority_evidence_valid"] is False
        return rejection

    malformed_evidence: list[dict[str, object]] = []
    missing_timeout = json.loads(json.dumps(evidence))
    missing_timeout["results"][0].pop("timed_out")
    malformed_evidence.append(missing_timeout)
    integer_timeout = json.loads(json.dumps(evidence))
    integer_timeout["results"][0]["timed_out"] = 0
    malformed_evidence.append(integer_timeout)
    integer_cache_hit = json.loads(json.dumps(evidence))
    integer_cache_hit["results"][0]["cache_hit"] = 0
    malformed_evidence.append(integer_cache_hit)
    prefixed_digest = json.loads(json.dumps(evidence))
    prefixed_digest["results"][0]["validation_result_digest"] = (
        "sha256:" + prefixed_digest["results"][0]["validation_result_digest"]
    )
    malformed_evidence.append(prefixed_digest)
    rebound_tree = json.loads(json.dumps(evidence))
    rebound_tree["manual_completion_authority_validated_tree_identity"][
        "target_commit"
    ] = "different-tree"
    rebound_tree["manual_completion_authority_validated_tree_id"] = (
        daemon_module.content_identity(
            rebound_tree[
                "manual_completion_authority_validated_tree_identity"
            ]
        )
    )
    malformed_evidence.append(rebound_tree)

    for malformed in malformed_evidence:
        rejection_for(malformed)

    _write_revalidation_board(
        board,
        descendant_status="todo",
        depends_on="TEST-001, TEST-999",
    )
    rejection_for(evidence)

    _write_revalidation_board(
        board,
        descendant_status="todo",
        validation="python -c 'raise SystemExit(1)'",
    )
    rejection_for(evidence)

    _write_revalidation_board(
        board,
        descendant_status="todo",
        descendant_title="Rebound task revision",
    )
    rejection_for(evidence)


def test_invalid_revalidation_store_is_archived_and_rebuilt_fail_closed(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.md"
    _write_revalidation_board(board, descendant_status="completed")
    daemon = _revalidation_daemon(
        tmp_path,
        board,
        suffix="invalid-store-recovery",
    )
    store_path = daemon._manual_completion_revalidation_store_path()
    store_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_bytes = b"{ definitely-not-json\n"
    store_path.write_bytes(invalid_bytes)

    guard = daemon._refresh_manual_completion_authority_guard()
    recovery = guard["revocation_guard"]
    archive_path = Path(recovery["invalid_store_archive_path"])
    rebuilt = json.loads(store_path.read_text(encoding="utf-8"))

    assert guard["available"] is True
    assert recovery["recovered_invalid_store"] is True
    assert recovery["revocation_generation"] == 1
    assert archive_path.is_file()
    assert archive_path.read_bytes() == invalid_bytes
    assert rebuilt["records"] == {}
    assert rebuilt["revocation_generation"] == 1
    assert daemon._validated_manual_completion_revalidation_store(rebuilt)
    assert guard["revalidation_receipt_task_ids"] == []
    assert guard["revalidation_task_ids"] == ["TEST-002"]


def test_cross_board_authority_metadata_must_be_nonempty_and_consistent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    board = tmp_path / "tasks.md"
    _write_runtime_board(board, gated_status="completed")
    daemon = _revalidation_daemon(
        tmp_path,
        board,
        suffix="cross-board-strict-metadata",
    )
    construction_called = False

    def unexpected_construction(_metadata):
        nonlocal construction_called
        construction_called = True
        raise AssertionError("cross-board completion daemon was constructed")

    monkeypatch.setattr(
        daemon,
        "_completion_daemon_for_merge_request",
        unexpected_construction,
    )
    other_board = tmp_path / "other-tasks.md"
    base_metadata = {
        "target_binding_schema": daemon_module.MERGE_TARGET_BINDING_SCHEMA,
        "target_repository_id": daemon.merge_target_repository_id,
        "target_branch": daemon.resolved_merge_target_branch,
        "todo_path": str(other_board),
        "task": {
            "task_id": "OTHER-001",
            "title": "cross-board task",
            "status": "todo",
        },
    }
    invalid_authority_metadata = (
        {
            "manual_completion_authority_context_id": None,
            "manual_completion_authority_task_ids": None,
            "manual_completion_authority_required_task_ids": None,
            "manual_completion_authority_epoch_id": None,
            "manual_completion_authority_revocation_generation": None,
        },
        {
            "manual_completion_authority_context_id": "forged-context",
            "manual_completion_authority_task_ids": [],
            "manual_completion_authority_required_task_ids": [],
            "manual_completion_authority_epoch_id": "epoch",
            "manual_completion_authority_revocation_generation": 0,
        },
        {
            "manual_completion_authority_context_id": "forged-context",
            "manual_completion_authority_task_ids": ["OTHER-ROOT"],
            "manual_completion_authority_required_task_ids": ["OUTSIDE"],
            "manual_completion_authority_epoch_id": "epoch",
            "manual_completion_authority_revocation_generation": 0,
        },
        {
            "manual_completion_authority_context_id": "wrong-context",
            "manual_completion_authority_task_ids": ["OTHER-ROOT"],
            "manual_completion_authority_required_task_ids": ["OTHER-ROOT"],
            "manual_completion_authority_epoch_id": "epoch",
            "manual_completion_authority_revocation_generation": 0,
        },
    )

    for authority_metadata in invalid_authority_metadata:
        request = SimpleNamespace(
            metadata={**base_metadata, **authority_metadata},
            target_repository_id=daemon.merge_target_repository_id,
            target_branch=daemon.resolved_merge_target_branch,
            task_id="OTHER-001",
            priority="P0",
            branch_name="agent/other-001",
            commit_sha="deadbeef",
            attempt=1,
        )
        result = daemon._merge_train_callback(request)
        assert result["reason"] == (
            "cross_board_manual_completion_authority_metadata_invalid"
        )

    valid_body = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "manual-completion-authority-context@3"
        ),
        "todo_path": str(other_board.resolve(strict=False)),
        "task_ids": ["OTHER-ROOT"],
        "required_task_ids": ["OTHER-ROOT"],
        "scheduler_epoch_id": "epoch",
        "revocation_generation": 0,
    }
    valid_request = SimpleNamespace(
        metadata={
            **base_metadata,
            "manual_completion_authority_context_id": (
                daemon_module.content_identity(valid_body)
            ),
            "manual_completion_authority_task_ids": ["OTHER-ROOT"],
            "manual_completion_authority_required_task_ids": ["OTHER-ROOT"],
            "manual_completion_authority_epoch_id": "epoch",
            "manual_completion_authority_revocation_generation": 0,
        },
        target_repository_id=daemon.merge_target_repository_id,
        target_branch=daemon.resolved_merge_target_branch,
        task_id="OTHER-001",
        priority="P0",
        branch_name="agent/other-001",
        commit_sha="deadbeef",
        attempt=1,
    )
    valid_result = daemon._merge_train_callback(valid_request)

    assert valid_result["reason"] == (
        "cross_board_manual_completion_authority_unavailable"
    )
    assert construction_called is False


def test_enqueue_transfers_producer_trust_to_candidate_tree_proof_only_in_process(
    tmp_path: Path,
    monkeypatch,
) -> None:
    board = tmp_path / "tasks.md"
    _write_revalidation_board(board, descendant_status="todo")
    daemon = _revalidation_daemon(
        tmp_path,
        board,
        suffix="candidate-proof-producer",
    )
    _install_successful_authority_validation_runner(daemon, monkeypatch)
    _context, evidence = _fresh_revalidation_evidence(
        daemon,
        tmp_path,
        log_name="candidate-proof-validation.log",
    )
    task = {item.task_id: item for item in daemon._load_tasks()}["TEST-002"]
    captured_metadata: dict[str, object] = {}
    captured_enqueue: dict[str, object] = {}

    def capture_enqueue(**kwargs):
        captured_enqueue.update(kwargs)
        captured_metadata.update(kwargs["metadata"])
        return SimpleNamespace(request_id="candidate-proof-request")

    monkeypatch.setattr(daemon.merge_queue, "enqueue", capture_enqueue)
    monkeypatch.setattr(
        daemon,
        "_reject_protected_merge_candidate",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_candidate_repository_tree",
        lambda _commit: "c" * 40,
    )
    monkeypatch.setattr(
        daemon,
        "_proof_changed_scopes",
        lambda **_kwargs: ([], True),
    )

    daemon._enqueue_merge_candidate(
        branch_name="agent/test-002",
        implementation_commit="a" * 40,
        baseline_ref="b" * 40,
        worktree_path=None,
        task=task,
        attempt=1,
        validation_result=evidence,
    )

    candidate_proof = captured_metadata["validation_proof"]
    candidate_tree_identity = candidate_proof[
        "manual_completion_authority_validated_tree_identity"
    ]
    candidate_proof_id = (
        daemon._manual_completion_revalidation_evidence_id(candidate_proof)
    )
    assert candidate_proof_id in (
        daemon._trusted_manual_completion_revalidation_evidence_ids
    )
    assert daemon._manual_completion_authority_rejection(
        ["TEST-002"],
        authority_context_id=captured_metadata[
            "manual_completion_authority_context_id"
        ],
        authority_evidence=candidate_proof,
        expected_validated_tree_identity=candidate_tree_identity,
    ) is None
    # Without the exact pre-merge expectation, the token cannot be consumed
    # until its candidate commit is actually integrated.
    preintegration_rejection = daemon._manual_completion_authority_rejection(
        ["TEST-002"],
        authority_context_id=captured_metadata[
            "manual_completion_authority_context_id"
        ],
        authority_evidence=candidate_proof,
    )
    assert preintegration_rejection is not None
    assert preintegration_rejection[
        "manual_completion_authority_evidence_valid"
    ] is False
    monkeypatch.setattr(
        daemon,
        "_manual_completion_validated_tree_is_current",
        lambda _identity: True,
    )
    assert daemon._manual_completion_authority_rejection(
        ["TEST-002"],
        authority_context_id=captured_metadata[
            "manual_completion_authority_context_id"
        ],
        authority_evidence=candidate_proof,
    ) is None

    restarted = _revalidation_daemon(
        tmp_path,
        board,
        suffix="candidate-proof-restart",
    )
    restart_rejection = restarted._manual_completion_authority_rejection(
        ["TEST-002"],
        authority_context_id=captured_metadata[
            "manual_completion_authority_context_id"
        ],
        authority_evidence=candidate_proof,
        expected_validated_tree_identity=candidate_tree_identity,
    )
    assert restart_rejection is not None
    assert restart_rejection[
        "manual_completion_authority_evidence_valid"
    ] is False

    merge_call: dict[str, object] = {}

    def capture_merge(*args, **kwargs):
        merge_call["args"] = args
        merge_call["kwargs"] = kwargs
        return {
            "attempted": False,
            "merged": False,
            "returncode": 2,
            "reason": "seeded_pre_merge_stop",
            "submodule_merge_results": [],
        }

    def fake_git_run(command, **_kwargs):
        if command[:2] == ["git", "rev-parse"]:
            ref = str(command[-1])
            resolved = (
                "a" * 40
                if ref.startswith("a" * 40)
                else "b" * 40
            )
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=f"{resolved}\n",
                stderr="",
            )
        if command[:3] == ["git", "merge-base", "--is-ancestor"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=1,
                stdout="",
                stderr="",
            )
        raise AssertionError(f"unexpected Git command: {command}")

    monkeypatch.setattr(daemon_module.subprocess, "run", fake_git_run)
    monkeypatch.setattr(
        daemon,
        "_scope_adjudication_merge_binding_error",
        lambda _proof: "",
    )
    monkeypatch.setattr(
        daemon,
        "_rehydrate_merge_request_branch",
        lambda **_kwargs: {"ready": True, "rehydrated": False},
    )
    monkeypatch.setattr(
        daemon,
        "_changed_submodule_durability_preflight",
        lambda **_kwargs: {"attempted": False, "verified": True},
    )
    monkeypatch.setattr(daemon, "_merge_branch_to_main", capture_merge)
    monkeypatch.setattr(daemon, "_git_ref_is_ancestor", lambda *_args: False)
    request = SimpleNamespace(
        metadata=dict(captured_metadata),
        target_repository_id=captured_enqueue["target_repository_id"],
        target_branch=captured_enqueue["target_branch"],
        task_id="TEST-002",
        canonical_task_id=captured_enqueue["canonical_task_id"],
        canonical_task_key=captured_enqueue["canonical_task_key"],
        priority="P0",
        branch_name="agent/test-002",
        commit_sha="a" * 40,
        attempt=1,
    )

    callback_result = daemon._merge_train_callback(request)

    assert callback_result["reason"] == "seeded_pre_merge_stop"
    assert merge_call["kwargs"]["expected_candidate_commit"] == "a" * 40
    assert merge_call["kwargs"]["expected_candidate_tree"] == "c" * 40


def test_completion_rechecks_generation_inside_mutation_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    board = tmp_path / "tasks.md"
    _write_revalidation_board(board, descendant_status="todo")
    daemon = _revalidation_daemon(
        tmp_path,
        board,
        suffix="completion-generation-cas",
    )
    context_id, evidence = _fresh_revalidation_evidence(
        daemon,
        tmp_path,
        log_name="completion-generation-cas.log",
    )
    original_expectation = daemon._completion_callback_expectation

    def revoke_after_initial_guard(*args, **kwargs):
        expectation = original_expectation(*args, **kwargs)
        current = board.read_text(encoding="utf-8")
        board.write_text(
            current.replace("- Status: completed", "- Status: todo", 1),
            encoding="utf-8",
        )
        return expectation

    monkeypatch.setattr(
        daemon,
        "_completion_callback_expectation",
        revoke_after_initial_guard,
    )

    result = daemon._mark_tasks_completed_in_todo(
        ["TEST-002"],
        primary_task_id="TEST-002",
        completion_reason="raced_completion",
        manual_completion_authority_context_id=context_id,
        manual_completion_authority_evidence=evidence,
    )

    assert result["updated"] is False
    assert result["durable"] is False
    assert result["reason"] == (
        "manual_completion_authority_dependency_required"
    )
    assert daemon._manual_completion_authority_revocation_generation == 1
    assert "## TEST-002" in board.read_text(encoding="utf-8")
    assert board.read_text(encoding="utf-8").count("- Status: todo") == 2


def test_merge_generation_cas_rejects_before_first_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    board = tmp_path / "tasks.md"
    _write_revalidation_board(board, descendant_status="todo")
    daemon = _revalidation_daemon(
        tmp_path,
        board,
        suffix="merge-generation-cas",
    )
    daemon._refresh_manual_completion_authority_guard()
    expected_generation = (
        daemon._manual_completion_authority_revocation_generation
    )
    task = {item.task_id: item for item in daemon._load_tasks()}["TEST-002"]
    mutation_called = False

    def racing_authority_check(*_args, **_kwargs):
        daemon._manual_completion_authority_revocation_generation += 1
        return None

    def unexpected_mutation():
        nonlocal mutation_called
        mutation_called = True
        raise AssertionError("merge mutation ran after authority generation race")

    monkeypatch.setattr(
        daemon,
        "_manual_completion_authority_rejection",
        racing_authority_check,
    )
    monkeypatch.setattr(
        daemon,
        "_preserve_generated_nested_worktree_directories",
        unexpected_mutation,
    )

    result = daemon._merge_branch_to_main_locked(
        "agent/test-002",
        task,
        1,
        manual_completion_authority_task_ids=("TEST-002",),
        manual_completion_authority_context_id="context",
        manual_completion_authority_evidence={},
        manual_completion_authority_expected_generation=expected_generation,
        manual_completion_authority_expected_tree_identity={},
    )

    assert result["merged"] is False
    assert result["reason"] == (
        "manual_completion_authority_generation_changed"
    )
    assert result["expected_manual_completion_authority_generation"] == (
        expected_generation
    )
    assert result["actual_manual_completion_authority_generation"] == (
        expected_generation + 1
    )
    assert mutation_called is False


def test_merge_rechecks_generation_after_preparation_before_rebase(
    tmp_path: Path,
    monkeypatch,
) -> None:
    board = tmp_path / "tasks.md"
    _write_revalidation_board(board, descendant_status="todo")
    daemon = _revalidation_daemon(
        tmp_path,
        board,
        suffix="merge-preparation-cas",
    )
    daemon._refresh_manual_completion_authority_guard()
    expected_generation = (
        daemon._manual_completion_authority_revocation_generation
    )
    task = {item.task_id: item for item in daemon._load_tasks()}["TEST-002"]
    authority_checks = 0
    preparation_steps: list[str] = []
    rebase_called = False

    def racing_second_authority_check(*_args, **_kwargs):
        nonlocal authority_checks
        authority_checks += 1
        if authority_checks == 2:
            daemon._manual_completion_authority_revocation_generation += 1
        return None

    def preserve():
        preparation_steps.append("preserve")

    def repair(_root):
        preparation_steps.append("repair")
        return {"repairs": []}

    def unexpected_rebase(*_args, **_kwargs):
        nonlocal rebase_called
        rebase_called = True
        raise AssertionError("rebase ran after authority generation race")

    monkeypatch.setattr(
        daemon,
        "_manual_completion_authority_rejection",
        racing_second_authority_check,
    )
    monkeypatch.setattr(
        daemon,
        "_preserve_generated_nested_worktree_directories",
        preserve,
    )
    monkeypatch.setattr(
        daemon,
        "_repair_stale_submodule_worktree_configs",
        repair,
    )
    monkeypatch.setattr(
        daemon,
        "_resolve_git_commit_in_repo",
        lambda *_args, **_kwargs: "a" * 40,
    )
    monkeypatch.setattr(
        daemon,
        "_rebase_stale_submodule_pointers",
        unexpected_rebase,
    )

    result = daemon._merge_branch_to_main_locked(
        "agent/test-002",
        task,
        1,
        manual_completion_authority_task_ids=("TEST-002",),
        manual_completion_authority_context_id="context",
        manual_completion_authority_evidence={},
        manual_completion_authority_expected_generation=expected_generation,
        manual_completion_authority_expected_tree_identity={},
    )

    assert authority_checks == 2
    assert preparation_steps == ["preserve", "repair"]
    assert rebase_called is False
    assert result["reason"] == (
        "manual_completion_authority_generation_changed"
    )


def test_queue_bound_merge_rejects_candidate_moved_before_locked_callback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    board = tmp_path / "tasks.md"
    _write_revalidation_board(board, descendant_status="todo")
    daemon = _revalidation_daemon(
        tmp_path,
        board,
        suffix="candidate-moved-before-lock",
    )
    task = {item.task_id: item for item in daemon._load_tasks()}["TEST-002"]
    expected_candidate = "a" * 40
    moved_candidate = "b" * 40
    expected_tree = "c" * 40
    events: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(daemon, "_main_branch_name", lambda: "main")
    monkeypatch.setattr(
        daemon,
        "_resolve_git_commit_in_repo",
        lambda _root, ref: (
            moved_candidate
            if ref == "implementation/test-002"
            else expected_candidate
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_candidate_repository_tree",
        lambda commit: expected_tree if commit == expected_candidate else "",
    )
    monkeypatch.setattr(
        daemon,
        "_preserve_generated_nested_worktree_directories",
        lambda: pytest.fail("candidate mismatch reached workspace mutation"),
    )
    monkeypatch.setattr(
        daemon,
        "_repair_stale_submodule_worktree_configs",
        lambda _root: pytest.fail("candidate mismatch reached repository repair"),
    )
    monkeypatch.setattr(
        daemon,
        "_record_event",
        lambda event_type, payload: events.append((event_type, dict(payload))),
    )

    result = daemon._merge_branch_to_main_locked(
        "implementation/test-002",
        task,
        1,
        expected_candidate_commit=expected_candidate,
        expected_candidate_tree=expected_tree,
    )

    assert result["merged"] is False
    assert result["returncode"] == 2
    assert result["reason"] == "merge_branch_candidate_mismatch"
    assert result["expected_candidate_commit"] == expected_candidate
    assert result["candidate_commit"] == expected_candidate
    assert result["branch_commit"] == moved_candidate
    assert [event_type for event_type, _payload in events] == [
        "merge_finished"
    ]


def test_mutated_merge_candidate_is_rejected_at_final_mutation_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    board = tmp_path / "tasks.md"
    _write_revalidation_board(board, descendant_status="todo")
    daemon = _revalidation_daemon(
        tmp_path,
        board,
        suffix="rebased-candidate-binding",
    )
    task = {item.task_id: item for item in daemon._load_tasks()}["TEST-002"]
    original_candidate = "a" * 40
    rebased_candidate = "b" * 40
    branch_commits = iter((original_candidate, rebased_candidate))
    candidate_tree = "c" * 40
    events: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        daemon,
        "_preserve_generated_nested_worktree_directories",
        lambda: None,
    )
    monkeypatch.setattr(
        daemon,
        "_repair_stale_submodule_worktree_configs",
        lambda _root: {"repairs": []},
    )
    monkeypatch.setattr(daemon, "_main_branch_name", lambda: "main")
    monkeypatch.setattr(
        daemon,
        "_resolve_git_commit_in_repo",
        lambda _root, ref: (
            next(branch_commits)
            if ref == "implementation/test-002"
            else original_candidate
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_candidate_repository_tree",
        lambda commit: candidate_tree if commit == original_candidate else "",
    )
    monkeypatch.setattr(
        daemon,
        "_rebase_stale_submodule_pointers",
        lambda *_args, **_kwargs: pytest.fail(
            "an immutable validated candidate was rebased in place"
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_record_event",
        lambda event_type, payload: events.append((event_type, dict(payload))),
    )
    monkeypatch.setattr(
        daemon,
        "_merge_candidate_completion_recheck",
        lambda *_args, **_kwargs: {
            "terminal": False,
            "candidate_ancestor": False,
            "branch_ancestor": False,
        },
    )
    monkeypatch.setattr(
        daemon,
        "_prepare_main_merge_workspace",
        lambda *_args, **_kwargs: {
            "available": True,
            "path": str(tmp_path),
            "ephemeral": False,
        },
    )
    monkeypatch.setattr(
        daemon,
        "_restore_incidental_main_gitlink_checkouts",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_resolve_generated_add_add_conflicts",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        daemon,
        "_identical_untracked_merge_paths",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        daemon,
        "_restore_generated_dirty_merge_overlap",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_dirty_merge_conflict_paths",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        daemon,
        "_reconcile_generated_dirty_submodule_overlap",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        daemon,
        "_remove_untracked_paths_for_merge",
        lambda *_args, **_kwargs: pytest.fail(
            "a mutated candidate reached target mutation"
        ),
    )

    result = daemon._merge_branch_to_main_locked(
        "implementation/test-002",
        task,
        1,
        baseline_ref="",
        changed_submodule_paths={"external/dependency"},
        expected_candidate_commit=original_candidate,
        expected_candidate_tree=candidate_tree,
    )

    assert result["merged"] is False
    assert result["returncode"] == 2
    assert result["reason"] == "merge_branch_candidate_mismatch"
    assert result["candidate_commit"] == original_candidate
    assert result["branch_commit"] == rebased_candidate
    assert [event_type for event_type, _payload in events] == [
        "submodule_pointer_rebase_skipped",
        "merge_finished",
    ]


def test_exact_candidate_merge_survives_concurrent_target_advance(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo-exact-candidate"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Authority Test")
    _git(repo, "config", "user.email", "authority@example.invalid")
    board = repo / "tasks.md"
    _write_revalidation_board(board, descendant_status="todo")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")

    branch = "implementation/test-002"
    _git(repo, "checkout", "-b", branch)
    (repo / "candidate.txt").write_text("candidate\n", encoding="utf-8")
    _git(repo, "add", "candidate.txt")
    _git(repo, "commit", "-m", "candidate")
    candidate_commit = _git(repo, "rev-parse", "HEAD")
    candidate_tree = _git(repo, "show", "-s", "--format=%T", "HEAD")

    _git(repo, "checkout", "main")
    (repo / "unrelated.txt").write_text("target advance\n", encoding="utf-8")
    _git(repo, "add", "unrelated.txt")
    _git(repo, "commit", "-m", "advance target independently")
    target_before = _git(repo, "rev-parse", "HEAD")

    daemon = _implementation_revalidation_daemon(
        tmp_path,
        repo,
        board,
        suffix="exact-candidate-concurrent-target",
    )
    task = {item.task_id: item for item in daemon._load_tasks()}["TEST-002"]
    result = daemon._merge_branch_to_main_locked(
        branch,
        task,
        1,
        expected_candidate_commit=candidate_commit,
        expected_candidate_tree=candidate_tree,
    )

    assert result["merged"] is True
    assert result["returncode"] == 0
    assert _git(repo, "rev-parse", branch) == candidate_commit
    integration_commit = _git(repo, "rev-parse", "main")
    parents = _git(repo, "show", "-s", "--format=%P", integration_commit).split()
    assert parents == [target_before, candidate_commit]
    assert _git(
        repo,
        "merge-base",
        "--is-ancestor",
        candidate_commit,
        integration_commit,
    ) == ""
    assert (repo / "candidate.txt").read_text(encoding="utf-8") == "candidate\n"
    assert (repo / "unrelated.txt").read_text(encoding="utf-8") == "target advance\n"


def test_enqueue_refreshes_authority_before_context_and_rejects_any_denial(
    tmp_path: Path,
    monkeypatch,
) -> None:
    board = tmp_path / "tasks.md"
    _write_runtime_board(board, gated_status="completed")
    daemon = _revalidation_daemon(
        tmp_path,
        board,
        suffix="enqueue-authority-refresh",
    )
    initial_guard = daemon._refresh_manual_completion_authority_guard()
    initial_context = daemon._manual_completion_authority_policy_id()
    assert initial_guard["required_task_ids"] == []
    pre_revocation_tasks = {
        task.task_id: task for task in daemon._load_tasks()
    }

    _write_runtime_board(board, gated_status="todo")
    captured_metadata: list[dict[str, object]] = []

    def capture_enqueue(**kwargs):
        captured_metadata.append(dict(kwargs["metadata"]))
        return SimpleNamespace(request_id="request-1")

    monkeypatch.setattr(daemon.merge_queue, "enqueue", capture_enqueue)
    monkeypatch.setattr(
        daemon,
        "_reject_protected_merge_candidate",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_candidate_repository_tree",
        lambda _commit: "c" * 40,
    )
    monkeypatch.setattr(
        daemon,
        "_proof_changed_scopes",
        lambda **_kwargs: ([], True),
    )

    daemon._enqueue_merge_candidate(
        branch_name="agent/test-003",
        implementation_commit="a" * 40,
        baseline_ref="b" * 40,
        worktree_path=None,
        task=pre_revocation_tasks["TEST-003"],
        attempt=1,
    )

    assert len(captured_metadata) == 1
    metadata = captured_metadata[0]
    current_context = daemon._manual_completion_authority_policy_id()
    assert current_context != initial_context
    assert metadata["manual_completion_authority_context_id"] == (
        current_context
    )
    assert metadata["manual_completion_authority_required_task_ids"] == [
        "TEST-001"
    ]
    assert metadata[
        "manual_completion_authority_revocation_generation"
    ] == daemon._manual_completion_authority_revocation_generation

    current_root = {
        task.task_id: task for task in daemon._load_tasks()
    }["TEST-001"]
    try:
        daemon._enqueue_merge_candidate(
            branch_name="agent/test-001",
            implementation_commit="d" * 40,
            baseline_ref="b" * 40,
            worktree_path=None,
            task=current_root,
            attempt=1,
        )
    except RuntimeError as exc:
        assert "manual_completion_authority_required" in str(exc)
    else:
        raise AssertionError("authority-denied root reached merge enqueue")
    assert len(captured_metadata) == 1


def test_missing_configured_authority_root_fails_closed(tmp_path: Path) -> None:
    board = tmp_path / "tasks.md"
    board.write_text(
        """# Tasks

## TEST-002 Ordinary task

- Status: todo
- Completion: auto
- Priority: P0
""",
        encoding="utf-8",
    )
    state_dir = tmp_path / "state-missing-root"
    daemon = daemon_module.PortalImplementationDaemon(
        todo_path=board,
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=tmp_path,
        task_header_prefix="## TEST-",
        implement=False,
        manual_completion_authority_task_ids=("TEST-001",),
        manual_completion_authority_required_task_ids=("TEST-001",),
    )

    result = daemon.run_once()

    assert result["blocked"] is True
    assert result["reason"] == "manual_completion_authority_guard_unavailable"
    assert result["manual_completion_authority_guard"]["reason"] == (
        "manual_completion_authority_guard_roots_invalid"
    )
    assert result["manual_completion_authority_guard"][
        "missing_root_task_ids"
    ] == ["TEST-001"]


def test_completed_claim_renews_isolated_despite_stale_scheduler_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, board = _git_revalidation_repo(
        tmp_path,
        descendants=[("TEST-002", "completed", "TEST-001")],
    )
    daemon = _implementation_revalidation_daemon(
        tmp_path, repo, board, suffix="stale-gates"
    )
    _forbid_revalidation_provider_and_seeding(daemon, monkeypatch)
    tasks = {task.task_id: task for task in daemon._load_tasks()}
    daemon._register_task_identities(list(tasks.values()))
    state = daemon_module.PortalTaskState()
    state.implementation_attempts["TEST-002"] = 1
    state.implementation_attempts_by_cid[
        daemon._canonical_ref(tasks["TEST-002"])
    ] = 1
    state.save(daemon.state_path)
    monkeypatch.setattr(
        daemon,
        "_task_has_recent_no_change_outcome",
        lambda task_id, *_args, **_kwargs: task_id == "TEST-002",
    )
    monkeypatch.setattr(
        daemon,
        "_pending_queued_merge_task_ids",
        lambda *_args, **_kwargs: {"TEST-002"},
    )
    monkeypatch.setattr(
        daemon,
        "_quarantined_queued_merge_task_ids",
        lambda *_args, **_kwargs: {"TEST-002"},
    )
    monkeypatch.setattr(
        daemon,
        "_unresolved_merge_failures_by_task",
        lambda *_args, **_kwargs: {"TEST-002": {"branch": "stale"}},
    )
    monkeypatch.setattr(
        daemon,
        "_transient_merge_deferrals_by_task",
        lambda *_args, **_kwargs: {"TEST-002": {"reason": "stale"}},
    )
    monkeypatch.setattr(
        daemon_module,
        "task_implementation_protected_path_conflicts",
        lambda task, _paths: ("tasks.md",)
        if task.task_id == "TEST-002"
        else (),
    )
    monkeypatch.setattr(
        daemon.task_queue,
        "is_cooled_down",
        lambda _task_id: True,
    )
    if daemon.worktree_pool is not None:
        monkeypatch.setattr(
            daemon.worktree_pool,
            "acquire",
            lambda **_kwargs: pytest.fail("authority path used worktree pool"),
        )
    head_before = _git(repo, "rev-parse", "HEAD")
    status_before = _git(repo, "status", "--porcelain", "--untracked-files=all")

    result = daemon.run_once()
    implementation = result["implementation_result"]

    assert implementation["returncode"] == 0, implementation
    assert implementation["authority_revalidation_only"] is True
    assert implementation["forced_isolation"] is True
    assert implementation["attempt_consumed"] is False
    assert implementation["provider_dispatched"] is False
    assert implementation["cleanup_result"]["cleaned"] is True
    assert not Path(implementation["worktree_path"]).exists()
    assert implementation["validation_result"]["passed"] is True
    assert all(
        item["cache_hit"] is False
        for item in implementation["validation_result"]["results"]
    )
    receipt = json.loads(
        Path(implementation["task_execution_receipt_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert receipt["isolation_audit"] == {
        "llm_call_count": 0,
        "model_call_count": 0,
        "provider_call_count": 0,
    }
    persisted = implementation["todo_update_result"][
        "manual_completion_authority_revalidation_receipt"
    ]
    assert persisted["task_ids"] == ["TEST-002"]
    assert _git(repo, "rev-parse", "HEAD") == head_before
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == status_before


def test_completed_claim_rejects_validation_mutation_without_source_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, board = _git_revalidation_repo(
        tmp_path,
        descendants=[("TEST-002", "completed", "TEST-001")],
    )
    board.write_text(
        board.read_text(encoding="utf-8").replace(
            "python -c 'raise SystemExit(0)'",
            "python -c \"from pathlib import Path; Path('rogue.txt').write_text('x')\"",
        ),
        encoding="utf-8",
    )
    _git(repo, "add", "tasks.md")
    _git(repo, "commit", "-m", "mutation validation")
    daemon = _implementation_revalidation_daemon(
        tmp_path, repo, board, suffix="mutation"
    )
    _forbid_revalidation_provider_and_seeding(daemon, monkeypatch)
    head_before = _git(repo, "rev-parse", "HEAD")

    implementation = daemon.run_once()["implementation_result"]

    assert implementation["returncode"] == 1
    assert implementation["validation_result"]["reason"] == (
        "declared_validation_failed"
    )
    assert implementation["validation_result"]["results"][0][
        "returncode"
    ] != 0
    assert "manual_completion_authority_revalidation_receipt" not in (
        implementation.get("todo_update_result") or {}
    )
    assert implementation["cleanup_result"]["cleaned"] is True
    assert not (repo / "rogue.txt").exists()
    assert _git(repo, "rev-parse", "HEAD") == head_before
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""


def test_todo_descendant_remains_on_provider_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, board = _git_revalidation_repo(
        tmp_path,
        descendants=[("TEST-002", "todo", "TEST-001")],
    )
    daemon = _implementation_revalidation_daemon(
        tmp_path, repo, board, suffix="todo-provider"
    )
    calls: list[str] = []
    monkeypatch.setattr(
        daemon,
        "_build_implementation_prompt",
        lambda *_args, **_kwargs: calls.append("prompt") or "",
    )
    monkeypatch.setattr(
        daemon,
        "_build_implementation_command",
        lambda *_args, **_kwargs: ["provider-test-double"],
    )
    monkeypatch.setattr(
        daemon,
        "_persist_implementation_context_receipt",
        lambda *_args, **_kwargs: tmp_path / "provider-context.json",
    )
    monkeypatch.setattr(
        daemon_module,
        "run_process_group_stream",
        lambda *_args, **_kwargs: calls.append("provider")
        or subprocess.CompletedProcess(args=(), returncode=1),
    )

    implementation = daemon.run_once()["implementation_result"]

    assert implementation["returncode"] == 1
    assert calls == ["prompt", "provider"]
    assert implementation.get("authority_revalidation_only") is not True


def test_restart_freshly_renews_completed_claim_without_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, board = _git_revalidation_repo(
        tmp_path,
        descendants=[("TEST-002", "completed", "TEST-001")],
    )
    first = _implementation_revalidation_daemon(
        tmp_path, repo, board, suffix="restart"
    )
    _forbid_revalidation_provider_and_seeding(first, monkeypatch)
    first_result = first.run_once()["implementation_result"]
    assert first_result is not None
    assert first_result["returncode"] == 0
    assert first_result["authority_revalidation_only"] is True
    assert first_result["validation_result"]["results"]
    assert first_result["task_execution_receipt_id"]
    monkeypatch.undo()
    second_patch = pytest.MonkeyPatch()
    try:
        second = _implementation_revalidation_daemon(
            tmp_path, repo, board, suffix="restart"
        )
        _forbid_revalidation_provider_and_seeding(second, second_patch)
        second_pass = second.run_once()
        second_result = second_pass.get("implementation_result")
        second_guard = second._refresh_manual_completion_authority_guard()
    finally:
        second_patch.undo()

    # After the first no-provider revalidation, a cold restart re-admits the
    # durable receipt and does not schedule another implement/revalidate pass
    # (and still never consults the provider).
    assert second_result is None
    assert "TEST-002" in second_guard["revalidation_receipt_task_ids"]
    assert "TEST-002" not in second_guard["revalidation_task_ids"]
    assert second_pass.get("ready_count", 0) == 0 or not second_pass.get(
        "eligible_ready_task_ids"
    )

def _install_successful_authority_validation_runner(
    daemon: daemon_module.PortalImplementationDaemon,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep control-flow regressions independent of the host Docker image."""

    contract = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "authority-validation-isolation@2"
        ),
        "available": True,
        "backend": "docker-local-cuda",
        "contract_id": daemon_module.content_identity(
            {"test_authority_validation_contract": 2}
        ),
        "docker_endpoint": "unix:///var/run/docker.sock",
        "image_id": "sha256:" + ("a" * 64),
        "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
        "gpu_requested": True,
        "network_mode": "none",
        "host_filesystem": "workspace_only_read_only",
        "workspace_mode": "read_only",
        "writable_filesystems": ["private_tmpfs", "private_shm"],
        "pid_namespace": "private",
        "capabilities": "none",
        "no_new_privileges": True,
        "container_auto_remove": True,
        "container_root": "read_only",
        "image_pull_allowed": False,
        "container_log_driver": "none",
        "output_limit_bytes": (
            daemon_module.AUTHORITY_VALIDATION_OUTPUT_LIMIT_BYTES
        ),
        "memory_limit_bytes": (
            daemon_module.AUTHORITY_VALIDATION_MEMORY_LIMIT_BYTES
        ),
        "tmpfs_limit_bytes": (
            daemon_module.AUTHORITY_VALIDATION_TMPFS_LIMIT_BYTES
        ),
        "cpu_limit": daemon_module.AUTHORITY_VALIDATION_CPU_LIMIT,
        "pids_limit": daemon_module.AUTHORITY_VALIDATION_PIDS_LIMIT,
    }

    def successful_runner(
        *,
        spec,
        workspace_path: Path,
        timeout_seconds: float,
        environment: dict[str, str],
    ) -> dict[str, object]:
        del timeout_seconds, environment
        receipt_body = {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "authority-validation-isolation-receipt@2"
            ),
            "contract_id": contract["contract_id"],
            "backend": "docker-local-cuda",
            "docker_endpoint": contract["docker_endpoint"],
            "image_id": contract["image_id"],
            "gpu_uuid": contract["gpu_uuid"],
            "gpu_requested": True,
            "network_mode": "none",
            "host_filesystem": "workspace_only_read_only",
            "workspace_path": str(workspace_path.resolve()),
            "workspace_read_only": True,
            "private_pid_namespace": True,
            "cgroup_process_limit": (
                daemon_module.AUTHORITY_VALIDATION_PIDS_LIMIT
            ),
            "memory_limit_bytes": (
                daemon_module.AUTHORITY_VALIDATION_MEMORY_LIMIT_BYTES
            ),
            "tmpfs_limit_bytes": (
                daemon_module.AUTHORITY_VALIDATION_TMPFS_LIMIT_BYTES
            ),
            "cpu_limit": daemon_module.AUTHORITY_VALIDATION_CPU_LIMIT,
            "capabilities_dropped": "all",
            "no_new_privileges": True,
            "container_root_read_only": True,
            "container_log_driver": "none",
            "output_limit_bytes": (
                daemon_module.AUTHORITY_VALIDATION_OUTPUT_LIMIT_BYTES
            ),
            "output_limit_exceeded": False,
            "output_bounded": True,
            "storage_bounded": True,
            "cpu_bounded": True,
            "container_removed": True,
            "process_tree_quiesced": True,
        }
        return {
            "command": str(spec.command),
            "raw_command": str(spec.raw_command or spec.command),
            "started_at": "2026-08-03T00:00:00+00:00",
            "finished_at": "2026-08-03T00:00:01+00:00",
            "returncode": 0,
            "output": "",
            "timed_out": False,
            "infrastructure_failure": False,
            "error": "",
            "reason": "",
            "authority_validation_isolation": dict(contract),
            "authority_validation_isolation_receipt": {
                **receipt_body,
                "receipt_id": daemon_module.content_identity(receipt_body),
            },
        }

    monkeypatch.setattr(
        daemon,
        "_authority_validation_isolation_contract",
        lambda: dict(contract),
    )
    monkeypatch.setattr(
        daemon,
        "_authority_validation_command_runner",
        successful_runner,
    )


def test_completed_to_todo_race_at_receipt_boundary_leaves_no_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, board = _git_revalidation_repo(
        tmp_path,
        descendants=[("TEST-002", "completed", "TEST-001")],
    )
    daemon = _implementation_revalidation_daemon(
        tmp_path,
        repo,
        board,
        suffix="completed-to-todo-race",
    )
    _forbid_revalidation_provider_and_seeding(daemon, monkeypatch)
    _install_successful_authority_validation_runner(daemon, monkeypatch)
    publish = (
        daemon._publish_manual_completion_authority_revalidation_receipt_only
    )
    trusted_evidence_at_boundary: set[str] = set()

    def reopen_descendant_before_receipt_publication(*args, **kwargs):
        trusted_evidence_at_boundary.update(
            daemon._trusted_manual_completion_revalidation_evidence_ids
        )
        board_text = board.read_text(encoding="utf-8")
        prefix, descendant = board_text.split("## TEST-002", 1)
        board.write_text(
            prefix
            + "## TEST-002"
            + descendant.replace(
                "- Status: completed",
                "- Status: todo",
                1,
            ),
            encoding="utf-8",
        )
        return publish(*args, **kwargs)

    monkeypatch.setattr(
        daemon,
        "_publish_manual_completion_authority_revalidation_receipt_only",
        reopen_descendant_before_receipt_publication,
    )

    implementation = daemon.run_once()["implementation_result"]
    todo_update = implementation["todo_update_result"]

    assert trusted_evidence_at_boundary
    assert implementation["returncode"] == 1, implementation
    assert implementation["validation_result"]["passed"] is False
    assert todo_update["updated"] is False
    assert todo_update["durable"] is False
    assert todo_update["reason"] == (
        "manual_completion_authority_revalidation_required"
    )
    assert "completion_callback_evidence" not in todo_update
    assert (
        "manual_completion_authority_revalidation_receipt"
        not in todo_update
    )
    assert trusted_evidence_at_boundary.isdisjoint(
        daemon._trusted_manual_completion_revalidation_evidence_ids
    )
    valid_receipt_task_ids, receipt_guard = (
        daemon._current_manual_completion_revalidation_receipts(
            daemon._load_tasks(),
            authority_context_id=(
                daemon._manual_completion_authority_policy_id()
            ),
        )
    )
    assert valid_receipt_task_ids == set()
    assert receipt_guard["valid_task_ids"] == []
    assert "- Status: todo" in board.read_text(encoding="utf-8")


def test_authority_renewal_does_not_consult_indirect_proof_provider_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, board = _git_revalidation_repo(
        tmp_path,
        descendants=[("TEST-002", "completed", "TEST-001")],
    )
    daemon = _implementation_revalidation_daemon(
        tmp_path,
        repo,
        board,
        suffix="proof-provider-options",
    )
    _forbid_revalidation_provider_and_seeding(daemon, monkeypatch)
    _install_successful_authority_validation_runner(daemon, monkeypatch)

    def forbidden_proof_options(*_args, **_kwargs):
        pytest.fail("authority renewal consulted indirect proof provider options")

    monkeypatch.setattr(
        daemon,
        "_proof_workflow_options",
        forbidden_proof_options,
    )

    implementation = daemon.run_once()["implementation_result"]

    assert implementation["returncode"] == 0, implementation
    assert implementation["authority_revalidation_only"] is True
    assert implementation["provider_dispatched"] is False
    assert implementation["provider_metadata_consulted"] is False
    assert implementation["validation_result"]["passed"] is True
    assert implementation["validation_result"]["results"]


def test_completed_dependency_chain_revalidates_in_topological_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, board = _git_revalidation_repo(
        tmp_path,
        descendants=[
            ("TEST-002", "completed", "TEST-001"),
            ("TEST-003", "completed", "TEST-002"),
        ],
    )
    daemon = _implementation_revalidation_daemon(
        tmp_path,
        repo,
        board,
        suffix="completed-chain",
    )
    _forbid_revalidation_provider_and_seeding(daemon, monkeypatch)
    _install_successful_authority_validation_runner(daemon, monkeypatch)

    first = daemon.run_once()
    first_implementation = first["implementation_result"]
    after_first = daemon._refresh_manual_completion_authority_guard()
    second = daemon.run_once()
    second_implementation = second["implementation_result"]
    after_second = daemon._refresh_manual_completion_authority_guard()

    assert first_implementation["task_id"] == "TEST-002"
    assert first_implementation["returncode"] == 0, first_implementation
    assert after_first["revalidation_receipt_task_ids"] == ["TEST-002"]
    assert after_first["revalidation_task_ids"] == ["TEST-003"]
    assert second_implementation["task_id"] == "TEST-003"
    assert second_implementation["returncode"] == 0, second_implementation
    assert after_second["revalidation_receipt_task_ids"] == [
        "TEST-002",
        "TEST-003",
    ]
    assert after_second["revalidation_task_ids"] == []


def test_revalidation_only_supervisor_bypasses_all_ordinary_maintenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = tmp_path / "tasks.md"
    board.write_text("# Tasks\n", encoding="utf-8")
    parsed = supervisor_module.parse_args(
        [
            "--todo-path",
            str(board),
            "--state-dir",
            str(tmp_path / "supervisor-state"),
            "--task-prefix",
            "## TEST-",
            "--implement",
            "--manual-completion-authority-task-id",
            "TEST-001",
            "--manual-completion-authority-revalidation-only",
            "--objective-refill-scan",
            "--codebase-refill-scan",
        ]
    )
    supervisor = PortalImplementationSupervisor(
        supervisor_config_from_args(parsed, repo_root=tmp_path)
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("revalidation-only supervisor entered ordinary maintenance")

    for name in (
        "_run_once_with_maintenance_under_lease",
        "_acquire_implementation_maintenance_lease",
        "repair_main_checkout_merge_state",
        "refill_objective_backlog",
        "refill_codebase_backlog",
        "_build_worktree_reconciliation_daemon",
    ):
        monkeypatch.setattr(supervisor, name, forbidden)

    result = supervisor.run_once()

    assert result == {
        "stuck": False,
        "maintenance_blocked": False,
        "reason": "manual_completion_authority_revalidation_only",
        "manual_completion_authority_revalidation_only": True,
        "ordinary_provider_dispatch_allowed": False,
    }


def test_revalidation_only_mode_selects_renewal_and_fences_every_provider_seam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, board = _git_revalidation_repo(
        tmp_path,
        descendants=[
            ("TEST-002", "completed", "TEST-001"),
            ("TEST-003", "todo", "TEST-001"),
        ],
    )
    daemon = _implementation_revalidation_daemon(
        tmp_path,
        repo,
        board,
        suffix="revalidation-only",
        revalidation_only=True,
    )
    _forbid_revalidation_provider_and_seeding(daemon, monkeypatch)
    _install_successful_authority_validation_runner(daemon, monkeypatch)

    def forbidden_ordinary_path(*_args, **_kwargs):
        pytest.fail("revalidation-only daemon entered an ordinary provider path")

    for name in (
        "_consume_one_merge_candidate",
        "_provider_capacity_backoff_schedule",
        "_recover_protected_checkout_mutation",
        "_recover_pending_external_completion_callbacks",
        "_reconcile_implementation_protected_path_fence",
        "_reconcile_failed_merges",
        "_cleanup_already_merged_worktrees",
        "_periodic_maintenance",
        "_reset_attempt_budgets_for_completed_retry_repairs",
        "_mark_tasks_ready_in_todo",
        "_task_completion_receipt_bindings",
        "load_strategy",
        "_release_completed_retry_budget_strategy_blocks",
        "_mark_tasks_completed_in_todo",
        "_mark_tasks_completed_in_todo_unchecked",
        "_commit_generated_file_update",
        "_commit_generated_file_update_locked",
        "_commit_parent_gitlink_updates",
    ):
        monkeypatch.setattr(daemon, name, forbidden_ordinary_path)

    first = daemon.run_once()
    implementation = first["implementation_result"]
    state = daemon_module.PortalTaskState.load(daemon.state_path)

    assert first["manual_completion_authority_revalidation_only"] is True
    assert first["ordinary_provider_dispatch_allowed"] is False
    assert first["merge_reconciliation"] == []
    assert implementation["task_id"] == "TEST-002"
    assert implementation["returncode"] == 0, implementation
    assert implementation["authority_revalidation_only"] is True
    assert implementation["provider_dispatched"] is False
    assert state.selectable_ready_task_ids == ["TEST-002"]
    assert "TEST-003" not in state.selectable_ready_task_ids

    tasks = {task.task_id: task for task in daemon._load_tasks()}
    ordinary = tasks["TEST-003"]
    direct = daemon._run_implementation(ordinary, state)
    assert direct == {
        "skipped": True,
        "reason": "manual_completion_authority_revalidation_only",
        "task_id": "TEST-003",
        "attempt": 1,
        "attempt_consumed": False,
        "provider_dispatched": False,
    }

    ephemeral = daemon._run_implementation_in_ephemeral_worktree(
        task=ordinary,
        state=state,
        attempt=1,
        started_at="2026-08-03T00:00:00+00:00",
        log_path=tmp_path / "forbidden-ephemeral.log",
        prompt="must not be dispatched",
    )
    assert ephemeral == {
        "skipped": True,
        "reason": "manual_completion_authority_revalidation_only",
        "task_id": "TEST-003",
        "attempt": 1,
        "attempt_consumed": False,
        "provider_dispatched": False,
    }

    merge_resolver = daemon._invoke_llm_merge_resolver_for_failed_merge(
        workspace=repo,
        task=ordinary,
        attempt=1,
        branch_name="implementation/test-003",
        target_branch="main",
        merge_command=["git", "merge", "implementation/test-003"],
        merge_stdout="",
        merge_stderr="conflict",
    )
    assert merge_resolver == {
        "attempted": False,
        "applied": False,
        "reason": "manual_completion_authority_revalidation_only",
        "provider_dispatched": False,
    }

    with pytest.raises(RuntimeError, match="model-assisted routing is forbidden"):
        daemon.route_model_assisted_contract_packet(
            None,
            current_snapshot_id="unused",
            task=ordinary,
            grok_provider=forbidden_ordinary_path,
            codex_provider=forbidden_ordinary_path,
            deterministic_provider=forbidden_ordinary_path,
            admission_gate=forbidden_ordinary_path,
            writer=forbidden_ordinary_path,
        )

    second = daemon.run_once()
    second_state = daemon_module.PortalTaskState.load(daemon.state_path)
    assert second["implementation_result"] is None
    assert second["active_task_id"] == ""
    assert second_state.selectable_ready_task_ids == []
    assert "TEST-003" in second_state.ready_task_ids


def test_revalidation_only_retained_checkout_lease_blocks_without_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, board = _git_revalidation_repo(
        tmp_path,
        descendants=[("TEST-002", "completed", "TEST-001")],
    )
    daemon = _implementation_revalidation_daemon(
        tmp_path,
        repo,
        board,
        suffix="revalidation-only-retained-lease",
        revalidation_only=True,
    )
    lease_path = daemon_module.checkout_mutation_lock_path(repo)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease_payload = {
        "lease_id": "retained-authority-test-lease",
        "protected_recovery_required": True,
        "protected_recovery_owner": "implementation_daemon",
        "protected_recovery_intent": {
            "task_id": "TEST-002",
            "completion_intent": {"status": "completed"},
        },
    }
    lease_bytes = (json.dumps(lease_payload, sort_keys=True) + "\n").encode()
    lease_path.write_bytes(lease_bytes)
    board_bytes = board.read_bytes()
    head = _git(repo, "rev-parse", "HEAD")

    def forbidden(*_args, **_kwargs):
        pytest.fail("revalidation-only daemon replayed retained recovery")

    for name in (
        "_recover_protected_checkout_mutation",
        "_adopt_protected_checkout_recovery",
        "_replay_completion_callback_expectation",
        "_commit_generated_file_update_locked",
        "_finish_retained_checkout_mutation_recovery",
        "_publish_guarded_completion_if_ready",
        "_release_checkout_mutation_lease",
        "_record_event",
    ):
        monkeypatch.setattr(daemon, name, forbidden)

    result = daemon.run_once()

    assert result["blocked"] is True
    assert result["reason"] == "protected_checkout_recovery_forbidden"
    assert result["unchanged"] is True
    assert result["write_count"] == 0
    assert result["implementation_result"] is None
    assert lease_path.read_bytes() == lease_bytes
    assert board.read_bytes() == board_bytes
    assert _git(repo, "rev-parse", "HEAD") == head
    assert not daemon.state_path.exists()
    assert not daemon.strategy_path.exists()
    assert not daemon.events_path.exists()


def test_revalidation_receipt_only_preserves_dirty_board_index_and_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, board = _git_revalidation_repo(
        tmp_path,
        descendants=[("TEST-002", "completed", "TEST-001")],
    )
    staged_text = board.read_text(encoding="utf-8").replace(
        "# Tasks\n\n",
        "# Tasks\n\nStaged unrelated operator note.\n\n",
        1,
    )
    board.write_text(staged_text, encoding="utf-8")
    _git(repo, "add", "tasks.md")
    board.write_text(
        staged_text.replace(
            "# Tasks\n\n",
            "# Tasks\n\nUnstaged unrelated operator note.\n\n",
            1,
        ),
        encoding="utf-8",
    )
    daemon = _implementation_revalidation_daemon(
        tmp_path,
        repo,
        board,
        suffix="revalidation-receipt-only-dirty-board",
        revalidation_only=True,
    )
    _forbid_revalidation_provider_and_seeding(daemon, monkeypatch)
    _install_successful_authority_validation_runner(daemon, monkeypatch)
    before = {
        "board": board.read_bytes(),
        "head": _git(repo, "rev-parse", "HEAD"),
        "index": _git(repo, "write-tree"),
        "status": _git(repo, "status", "--porcelain=v1"),
    }

    def forbidden(*_args, **_kwargs):
        pytest.fail("authority receipt publication entered a board commit path")

    for name in (
        "_mark_tasks_completed_in_todo",
        "_mark_tasks_completed_in_todo_unchecked",
        "_commit_generated_file_update",
        "_commit_generated_file_update_locked",
        "_commit_parent_gitlink_updates",
    ):
        monkeypatch.setattr(daemon, name, forbidden)

    implementation = daemon.run_once()["implementation_result"]
    publication = implementation["todo_update_result"]

    assert implementation["returncode"] == 0, implementation
    assert publication["authority_receipt_only"] is True
    assert publication["updated"] is False
    assert publication["reason"] == "already_completed"
    assert publication["durable"] is True
    assert publication["protected_board_postcondition"]["trusted"] is True
    assert publication[
        "manual_completion_authority_revalidation_receipt"
    ]["persisted"] is True
    assert board.read_bytes() == before["board"]
    assert _git(repo, "rev-parse", "HEAD") == before["head"]
    assert _git(repo, "write-tree") == before["index"]
    assert _git(repo, "status", "--porcelain=v1") == before["status"]


def test_revalidation_only_stale_implementation_state_blocks_without_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, board = _git_revalidation_repo(
        tmp_path,
        descendants=[("TEST-002", "completed", "TEST-001")],
    )
    daemon = _implementation_revalidation_daemon(
        tmp_path,
        repo,
        board,
        suffix="revalidation-only-stale-state",
        revalidation_only=True,
    )
    stale_state = daemon_module.PortalTaskState(
        active_task_id="TEST-099",
        active_attempt=3,
        active_worktree_path=str(tmp_path / "stale-worktree"),
        implementation_in_progress=True,
    )
    stale_state.save(daemon.state_path)
    state_bytes = daemon.state_path.read_bytes()
    board_bytes = board.read_bytes()
    head = _git(repo, "rev-parse", "HEAD")

    def forbidden(*_args, **_kwargs):
        pytest.fail("revalidation-only daemon mutated stale implementation state")

    monkeypatch.setattr(
        daemon,
        "_find_live_inflight_implementation",
        lambda: None,
    )
    monkeypatch.setattr(
        daemon_module,
        "consume_stale_active_attempt",
        forbidden,
    )
    monkeypatch.setattr(daemon, "_clear_active_execution_state", forbidden)
    monkeypatch.setattr(daemon, "_record_event", forbidden)
    monkeypatch.setattr(daemon_module.PortalTaskState, "save", forbidden)

    result = daemon.run_once()

    assert result["blocked"] is True
    assert result["reason"] == (
        "ordinary_implementation_state_recovery_forbidden"
    )
    assert result["active_task_id"] == "TEST-099"
    assert result["active_attempt"] == 3
    assert result["unchanged"] is True
    assert result["write_count"] == 0
    assert result["implementation_result"] is None
    assert daemon.state_path.read_bytes() == state_bytes
    assert board.read_bytes() == board_bytes
    assert _git(repo, "rev-parse", "HEAD") == head
    assert not daemon.strategy_path.exists()
    assert not daemon.events_path.exists()


@pytest.mark.parametrize(
    ("runtime_field", "runtime_value"),
    (
        ("decision_runtime", object()),
        ("decision_runtime_config", {}),
    ),
)
def test_revalidation_only_constructor_rejects_custom_decision_runtime(
    tmp_path: Path,
    runtime_field: str,
    runtime_value: object,
) -> None:
    board = tmp_path / "tasks.md"
    board.write_text("# Tasks\n", encoding="utf-8")
    state_dir = tmp_path / "state"

    with pytest.raises(ValueError, match="forbids a custom decision runtime"):
        daemon_module.PortalImplementationDaemon(
            todo_path=board,
            state_path=state_dir / "task-state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            repo_root=tmp_path,
            implement=True,
            manual_completion_authority_task_ids=("TEST-001",),
            manual_completion_authority_revalidation_only=True,
            **{runtime_field: runtime_value},
        )


def test_validated_tree_identity_accepts_ancestor_of_current_head(
    tmp_path: Path,
) -> None:
    """Ordinary forward commits must not invalidate durable revalidation receipts."""

    repo, board = _git_revalidation_repo(
        tmp_path,
        descendants=[("TEST-002", "completed", "TEST-001")],
    )
    daemon = _implementation_revalidation_daemon(
        tmp_path, repo, board, suffix="tree-ancestry"
    )
    base = _git(repo, "rev-parse", "HEAD")
    base_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    identity = {
        "target_commit": base,
        "repository_tree_id": f"git-tree:{base_tree}",
    }
    assert daemon._manual_completion_validated_tree_is_current(identity) is True

    (repo / "forward.txt").write_text("forward\n", encoding="utf-8")
    _git(repo, "add", "forward.txt")
    _git(repo, "commit", "-m", "forward progress")
    assert _git(repo, "rev-parse", "HEAD") != base
    assert daemon._manual_completion_validated_tree_is_current(identity) is True

    # Rewrite the merge-target branch onto an unrelated root so the validated
    # commit is no longer an ancestor of current HEAD.
    _git(repo, "checkout", "--orphan", "divergent")
    (repo / "other.txt").write_text("other\n", encoding="utf-8")
    _git(repo, "add", "other.txt")
    _git(repo, "commit", "-m", "divergent root")
    main_branch = daemon._main_branch_name()
    _git(repo, "branch", "-M", main_branch)
    assert daemon._manual_completion_validated_tree_is_current(identity) is False
