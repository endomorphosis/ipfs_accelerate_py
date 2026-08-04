from __future__ import annotations

import json

from ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor import (
    build_arg_parser as build_bundle_arg_parser,
    implementation_supervisor_command,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTaskState,
    parse_task_file,
    parse_args as parse_daemon_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    PortalSupervisorConfig,
    parse_args as parse_supervisor_args,
    supervisor_config_from_args,
)


def _write_single_task_board(path) -> None:
    path.write_text(
        """# Tasks

## TASK-001 Keep a failed task fenced

- Status: todo
- Priority: P0
- Track: agent
- Outputs: src/retry_fence.py
- Acceptance: A failed first attempt must not launch a second model invocation.
""",
        encoding="utf-8",
    )


def test_canonical_attempt_limit_blocks_cooldown_fallback_retry(
    tmp_path,
    monkeypatch,
) -> None:
    todo_path = tmp_path / "tasks.todo.md"
    _write_single_task_board(todo_path)
    state_dir = tmp_path / "state"
    state_path = state_dir / "task_state.json"
    events_path = state_dir / "events.jsonl"
    strategy_path = state_dir / "strategy.json"
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=state_path,
        strategy_path=strategy_path,
        events_path=events_path,
        repo_root=tmp_path,
        task_header_prefix="## TASK-",
        implement=True,
        max_task_attempts=1,
        merge_queue_dir=tmp_path / "merge-queue",
        validation_cache_dir=tmp_path / "validation-cache",
        worktree_pool_enabled=False,
    )
    model_attempts: list[int] = []

    def fake_model_invocation(task, state):
        attempt = daemon._task_attempt(state, task)
        model_attempts.append(attempt)
        daemon._record_task_attempt(state, task, attempt)
        state.last_implementation_returncode = 1
        state.save(state_path)
        daemon._record_task_queue_outcome(task, 1, reason="test_failure")
        result = {"task_id": task.task_id, "attempt": attempt, "returncode": 1}
        daemon._record_event("implementation_started", result)
        daemon._record_event("implementation_finished", result)
        return result

    monkeypatch.setattr(daemon, "_run_implementation", fake_model_invocation)

    first = daemon.run_once()
    first_state = PortalTaskState.load(state_path)
    canonical_task_cid = first_state.task_identities["TASK-001"]["canonical_task_cid"]

    assert first["implementation_result"]["attempt"] == 1
    assert first_state.implementation_attempts_by_cid[canonical_task_cid] == 1
    assert daemon.task_queue.is_cooled_down(canonical_task_cid) is True

    second = daemon.run_once()
    second_state = PortalTaskState.load(state_path)

    assert model_attempts == [1]
    assert second["implementation_result"] is None
    assert second["active_task_id"] == ""
    assert second["ready_count"] == 1
    assert second["selectable_ready_count"] == 0
    assert second["attempt_limited_task_ids"] == ["TASK-001"]
    assert second["selection_idle_reason"] == (
        "all_selectable_ready_tasks_reached_max_task_attempts"
    )
    assert second_state.implementation_attempts_by_cid[canonical_task_cid] == 1

    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    backpressure = [
        event
        for event in events
        if event["type"] == "task_attempt_limit_backpressure"
    ]
    assert len(backpressure) == 1
    assert backpressure[0]["reason"] == "max_task_attempts_reached"
    assert backpressure[0]["max_task_attempts"] == 1
    assert backpressure[0]["limited_tasks"] == [
        {
            "task_id": "TASK-001",
            "canonical_task_key": first_state.task_identities["TASK-001"][
                "canonical_task_key"
            ],
            "canonical_task_cid": canonical_task_cid,
            "attempt_count": 1,
        }
    ]
    daemon_pass = events[-1]
    assert daemon_pass["type"] == "daemon_pass"
    assert daemon_pass["execution_slice_task_statuses"] == {
        "TASK-001": "ready",
    }
    assert daemon_pass["execution_slice_task_cids_by_id"] == {
        "TASK-001": canonical_task_cid,
    }


def test_completed_retry_repair_restores_attempt_budget_and_queue_eligibility(
    tmp_path,
    monkeypatch,
) -> None:
    todo_path = tmp_path / "tasks.todo.md"
    _write_single_task_board(todo_path)
    state_dir = tmp_path / "state"
    state_path = state_dir / "task_state.json"
    events_path = state_dir / "events.jsonl"
    strategy_path = state_dir / "strategy.json"
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=state_path,
        strategy_path=strategy_path,
        events_path=events_path,
        repo_root=tmp_path,
        task_header_prefix="## TASK-",
        implement=True,
        max_task_attempts=1,
        merge_queue_dir=tmp_path / "merge-queue",
        validation_cache_dir=tmp_path / "validation-cache",
        worktree_pool_enabled=False,
    )
    source_task = parse_task_file(todo_path, "## TASK-")[0]
    daemon._register_task_identities([source_task])
    source_identity = daemon._identity_for_task(source_task)
    exhausted_state = PortalTaskState(
        task_identities={source_task.task_id: source_identity.to_dict()},
        implementation_attempts={source_task.task_id: 1},
        implementation_attempts_by_cid={
            source_identity.canonical_task_cid: 1
        },
        last_implementation_task_id=source_task.task_id,
        last_implementation_task_key=source_identity.canonical_task_key,
        last_implementation_task_cid=source_identity.canonical_task_cid,
    )
    exhausted_state.save(state_path)
    daemon.task_queue.register_task(source_identity).record_failure("validation")
    daemon.task_queue.save()
    todo_path.write_text(
        todo_path.read_text(encoding="utf-8")
        + """

## TASK-002 Resolve validation retry-budget failure for TASK-001

- Status: completed
- Priority: P0
- Track: agent
- Outputs: test/retry_repair.py
- Acceptance: Repair the validation contract, then release TASK-001 from strategy blocked_tasks.
""",
        encoding="utf-8",
    )
    strategy_path.write_text(
        json.dumps(
            {
                "blocked_tasks": ["TASK-001", "TASK-999"],
                "focus_tracks": ["agent"],
            }
        ),
        encoding="utf-8",
    )
    launched_attempts: list[int] = []

    def record_launch(task, current_state):
        attempt = daemon._task_attempt(current_state, task)
        launched_attempts.append(attempt)
        daemon._record_task_attempt(current_state, task, attempt)
        current_state.save(state_path)
        return {
            "task_id": task.task_id,
            "attempt": attempt,
            "returncode": 0,
        }

    monkeypatch.setattr(daemon, "_run_implementation", record_launch)

    first = daemon.run_once()
    reset_state = PortalTaskState.load(state_path)
    second = daemon.run_once()

    assert launched_attempts == [1]
    assert first["retry_budget_resets"][0]["source_task_id"] == "TASK-001"
    assert first["retry_budget_resets"][0][
        "previous_display_attempt_count"
    ] == 1
    assert first["released_retry_budget_strategy_blocks"] == [
        {
            "source_task_id": "TASK-001",
            "repair_task_id": "TASK-002",
            "failure_kind": "validation",
        }
    ]
    assert daemon.load_strategy()["blocked_tasks"] == ["TASK-999"]
    assert first["attempt_limited_task_ids"] == []
    assert reset_state.retry_budget_repair_receipts == {
        "TASK-001": "TASK-002"
    }
    assert reset_state.implementation_attempts["TASK-001"] == 1
    assert reset_state.implementation_attempts_by_cid[
        source_identity.canonical_task_cid
    ] == 1
    assert daemon.task_queue.is_cooled_down(
        source_identity.canonical_task_cid
    ) is False
    assert second["retry_budget_resets"] == []
    assert second["attempt_limited_task_ids"] == ["TASK-001"]


def test_max_task_attempts_threads_from_bundle_to_daemon_command(tmp_path) -> None:
    bundle_args = build_bundle_arg_parser().parse_args(
        [
            "--bundle-index-path",
            str(tmp_path / "bundles.json"),
            "--max-task-attempts",
            "1",
        ]
    )
    assert bundle_args.max_task_attempts == 1

    supervisor_command = implementation_supervisor_command(
        todo_path=tmp_path / "tasks.todo.md",
        state_dir=tmp_path / "state",
        worktree_root=tmp_path / "worktrees",
        state_prefix="task",
        task_prefix="## TASK-",
        implement=True,
        daemon_interval=1.0,
        stale_seconds=2.0,
        check_interval=3.0,
        watchdog_startup_grace_seconds=None,
        max_restarts=0,
        implementation_timeout=4.0,
        max_task_attempts=bundle_args.max_task_attempts,
    )
    supervisor_flag = supervisor_command.index("--max-task-attempts")
    assert supervisor_command[supervisor_flag + 1] == "1"

    supervisor_args = parse_supervisor_args(
        [
            "--todo-path",
            str(tmp_path / "tasks.todo.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--max-task-attempts",
            supervisor_command[supervisor_flag + 1],
        ]
    )
    config = supervisor_config_from_args(supervisor_args, repo_root=tmp_path)
    assert config.max_task_attempts == 1

    daemon_command = PortalImplementationSupervisor(config)._build_daemon_command()
    daemon_flag = daemon_command.index("--max-task-attempts")
    assert daemon_command[daemon_flag + 1] == "1"
    assert parse_daemon_args(
        ["--max-task-attempts", daemon_command[daemon_flag + 1]]
    ).max_task_attempts == 1


def test_max_task_attempts_defaults_to_unlimited() -> None:
    assert build_bundle_arg_parser().parse_args(
        ["--bundle-index-path", "bundles.json"]
    ).max_task_attempts == 0
    assert parse_supervisor_args([]).max_task_attempts == 0
    assert parse_daemon_args([]).max_task_attempts == 0


def test_merge_target_branch_threads_from_bundle_to_daemon_command(tmp_path) -> None:
    bundle_args = build_bundle_arg_parser().parse_args(
        [
            "--bundle-index-path",
            str(tmp_path / "bundles.json"),
            "--merge-target-branch",
            "world-aid-duckdb-supervisor",
        ]
    )
    supervisor_command = implementation_supervisor_command(
        todo_path=tmp_path / "tasks.todo.md",
        state_dir=tmp_path / "state",
        worktree_root=tmp_path / "worktrees",
        state_prefix="task",
        task_prefix="## TASK-",
        implement=True,
        daemon_interval=1.0,
        stale_seconds=2.0,
        check_interval=3.0,
        watchdog_startup_grace_seconds=None,
        max_restarts=0,
        implementation_timeout=4.0,
        merge_target_branch=bundle_args.merge_target_branch,
    )
    supervisor_flag = supervisor_command.index("--merge-target-branch")
    assert supervisor_command[supervisor_flag + 1] == "world-aid-duckdb-supervisor"

    supervisor_args = parse_supervisor_args(
        [
            "--todo-path",
            str(tmp_path / "tasks.todo.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--implement",
            "--merge-target-branch",
            supervisor_command[supervisor_flag + 1],
        ]
    )
    config = supervisor_config_from_args(supervisor_args, repo_root=tmp_path)
    daemon_command = PortalImplementationSupervisor(config)._build_daemon_command()
    daemon_flag = daemon_command.index("--merge-target-branch")
    assert daemon_command[daemon_flag + 1] == "world-aid-duckdb-supervisor"
    assert parse_daemon_args(
        ["--merge-target-branch", daemon_command[daemon_flag + 1]]
    ).merge_target_branch == "world-aid-duckdb-supervisor"


def test_started_attempt_survives_abrupt_daemon_death(
    tmp_path,
    monkeypatch,
) -> None:
    todo_path = tmp_path / "tasks.todo.md"
    _write_single_task_board(todo_path)
    state_dir = tmp_path / "state"
    state_path = state_dir / "task_state.json"
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=state_path,
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=tmp_path,
        task_header_prefix="## TASK-",
        implement=True,
        max_task_attempts=1,
        worktree_pool_enabled=False,
    )
    task = parse_task_file(todo_path, "## TASK-")[0]
    daemon._register_task_identities([task])
    identity = daemon._identity_for_task(task)
    state = PortalTaskState(
        task_identities={task.task_id: identity.to_dict()},
    )
    daemon._mark_implementation_started(
        state,
        task=task,
        attempt=1,
        started_at="2026-07-24T00:00:00+00:00",
        log_path=state_dir / "attempt-1.log",
    )
    launched: list[str] = []
    monkeypatch.setattr(
        daemon,
        "_run_implementation",
        lambda selected, _state: launched.append(selected.task_id),
    )

    result = daemon.run_once()
    recovered = PortalTaskState.load(state_path)

    assert launched == []
    assert result["implementation_result"] is None
    assert result["attempt_limited_task_ids"] == ["TASK-001"]
    assert recovered.implementation_attempts["TASK-001"] == 1
    assert recovered.implementation_attempts_by_cid[
        identity.canonical_task_cid
    ] == 1
    assert recovered.implementation_in_progress is False


def test_supervisor_stale_repair_migrates_legacy_active_attempt(tmp_path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state_path = state_dir / "task_state.json"
    task_cid = "baguqeerastaleattempt"
    PortalTaskState(
        active_task_id="TASK-001",
        active_task_key="task/v1/stale",
        active_task_cid=task_cid,
        active_attempt=1,
        active_worktree_path=str(tmp_path / "dead-worktree"),
        active_branch="implementation/dead-attempt",
        implementation_in_progress=True,
    ).save(state_path)
    supervisor = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=tmp_path / "tasks.todo.md",
            state_path=state_path,
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=tmp_path,
        )
    )
    supervisor._read_managed_daemon_pid = lambda: None
    supervisor._list_process_commands = lambda: []

    result = supervisor.repair_stale_active_execution_state()
    recovered = PortalTaskState.load(state_path)

    assert result["repaired"] is True
    assert result["attempt_recovery"]["consumed"] is True
    assert recovered.implementation_attempts["TASK-001"] == 1
    assert recovered.implementation_attempts_by_cid[task_cid] == 1
    assert recovered.active_attempt == 0


def test_classify_provider_capacity_detects_grok_402_balance_exhausted() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        classify_provider_capacity_failure,
    )

    text = (
        'Internal error: {\n'
        '  "message": "API error (status 402 Payment Required): '
        'Grok Build usage balance exhausted",\n'
        '  "http_status": 402\n'
        "}\n"
    )
    classified = classify_provider_capacity_failure(text)
    assert classified["exhausted"] is True
    assert "grok" in classified["providers"] or "provider" in classified["providers"]
    assert classified["reason"] == "provider_capacity_exhausted"


def test_provider_capacity_deferral_rolls_back_start_charge(tmp_path) -> None:
    todo_path = tmp_path / "tasks.todo.md"
    _write_single_task_board(todo_path)
    state_dir = tmp_path / "state"
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=tmp_path,
        task_header_prefix="## TASK-",
        implement=True,
        max_task_attempts=1,
        worktree_pool_enabled=False,
    )
    task = parse_task_file(todo_path, "## TASK-")[0]
    daemon._register_task_identities([task])
    identity = daemon._identity_for_task(task)
    state = PortalTaskState(
        task_identities={task.task_id: identity.to_dict()},
    )
    log_path = state_dir / "attempt-1.log"
    daemon._mark_implementation_started(
        state,
        task=task,
        attempt=1,
        started_at="2026-07-24T00:00:00+00:00",
        log_path=log_path,
    )

    result = daemon._record_provider_capacity_deferral(
        task=task,
        state=state,
        attempt=1,
        started_at="2026-07-24T00:00:00+00:00",
        returncode=1,
        log_path=log_path,
        failure={"providers": ["codex"], "evidence": ["rate limited"]},
    )
    recovered = PortalTaskState.load(daemon.state_path)

    assert result["attempt_consumed"] is False
    assert task.task_id not in recovered.implementation_attempts
    assert (
        identity.canonical_task_cid
        not in recovered.implementation_attempts_by_cid
    )
    assert daemon._task_attempt(recovered, task) == 1


def test_provider_review_only_acceptance_preserves_retry_budget(tmp_path) -> None:
    todo_path = tmp_path / "tasks.todo.md"
    _write_single_task_board(todo_path)
    state_dir = tmp_path / "state"
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=tmp_path,
        task_header_prefix="## TASK-",
        implement=True,
        max_task_attempts=1,
        worktree_pool_enabled=False,
    )
    task = parse_task_file(todo_path, "## TASK-")[0]
    daemon._register_task_identities([task])
    identity = daemon._identity_for_task(task)
    state = PortalTaskState(
        task_identities={task.task_id: identity.to_dict()},
    )
    daemon._mark_implementation_started(
        state,
        task=task,
        attempt=1,
        started_at="2026-08-03T00:00:00+00:00",
        log_path=state_dir / "attempt-1.log",
    )
    acceptance = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "authoritative-acceptance-status@1"
        ),
        "task_id": task.task_id,
        "merge_commit": "a" * 40,
        "acceptance_state": "implemented_merged_but_pending",
        "admitted": False,
        "authoritatively_completed": False,
        "completion_authoritative": False,
        "pending_gates": ["provider_review"],
        "gate": {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "authoritative-completion-gate@1"
            ),
            "task_id": task.task_id,
            "admitted": False,
            "completion_authoritative": False,
            "acceptance_state": "implemented_merged_but_pending",
            "merge_commit": "a" * 40,
            "repository_tree_id": f"git-tree:{'b' * 40}",
            "pending_gates": ["provider_review"],
            "satisfied_gates": [
                "merge",
                "freshness",
                "semantic",
                "proof",
                "deterministic_only",
            ],
        },
        "receipt": {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "implementation-receipt@1"
            ),
            "task_id": task.task_id,
            "merged": True,
            "completion_authoritative": False,
            "acceptance_state": "implemented_merged_but_pending",
            "pending_gates": ["provider_review"],
            "validation_passed": True,
            "validation_stale": False,
            "merge_commit": "a" * 40,
            "repository_tree_id": f"git-tree:{'b' * 40}",
        },
    }

    deferred = daemon._defer_provider_review_only_acceptance(
        task=task,
        state=state,
        attempt=1,
        acceptance_result=acceptance,
    )
    state.save(daemon.state_path)
    recovered = PortalTaskState.load(daemon.state_path)
    selectable, limited = daemon._partition_tasks_at_attempt_limit(
        [task],
        {task.task_id: "ready"},
        recovered,
    )

    assert deferred["deferred"] is True
    assert deferred["resumable"] is True
    assert deferred["attempt_consumed"] is False
    assert recovered.implementation_attempts == {}
    assert recovered.implementation_attempts_by_cid == {}
    assert selectable == [task]
    assert limited == []
    assert daemon.task_queue.is_cooled_down(identity.canonical_task_cid) is True

    missing_semantic = {
        **acceptance,
        "gate": {
            **acceptance["gate"],
            "satisfied_gates": [
                "merge",
                "freshness",
                "proof",
                "deterministic_only",
            ],
        },
    }
    assert daemon._provider_review_only_acceptance_pending(missing_semantic) is False

    missing_merge = json.loads(json.dumps(acceptance))
    missing_merge.pop("merge_commit")
    assert daemon._provider_review_only_acceptance_pending(missing_merge) is False

    mismatched_receipt_merge = json.loads(json.dumps(acceptance))
    mismatched_receipt_merge["receipt"]["merge_commit"] = "c" * 40
    assert (
        daemon._provider_review_only_acceptance_pending(mismatched_receipt_merge)
        is False
    )

    missing_tree = json.loads(json.dumps(acceptance))
    missing_tree["gate"].pop("repository_tree_id")
    assert daemon._provider_review_only_acceptance_pending(missing_tree) is False

    mismatched_receipt_tree = json.loads(json.dumps(acceptance))
    mismatched_receipt_tree["receipt"]["repository_tree_id"] = (
        f"git-tree:{'d' * 40}"
    )
    assert (
        daemon._provider_review_only_acceptance_pending(mismatched_receipt_tree)
        is False
    )

    missing_admission = json.loads(json.dumps(acceptance))
    missing_admission.pop("admitted")
    assert daemon._provider_review_only_acceptance_pending(missing_admission) is False


def test_new_canonical_revision_gets_fresh_attempt_budget(
    tmp_path,
    monkeypatch,
) -> None:
    todo_path = tmp_path / "tasks.todo.md"
    _write_single_task_board(todo_path)
    state_dir = tmp_path / "state"
    state_path = state_dir / "task_state.json"
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=state_path,
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=tmp_path,
        task_header_prefix="## TASK-",
        implement=True,
        max_task_attempts=1,
        worktree_pool_enabled=False,
    )
    revision_a = parse_task_file(todo_path, "## TASK-")[0]
    daemon._register_task_identities([revision_a])
    identity_a = daemon._identity_for_task(revision_a)
    state = PortalTaskState(
        task_identities={revision_a.task_id: identity_a.to_dict()},
    )
    daemon._record_task_attempt(state, revision_a, 1)
    state.save(state_path)
    todo_path.write_text(
        todo_path.read_text(encoding="utf-8").replace(
            "A failed first attempt must not launch a second model invocation.",
            "A DuckDB-backed revised task must receive a fresh canonical budget.",
        ),
        encoding="utf-8",
    )
    revision_b = parse_task_file(todo_path, "## TASK-")[0]
    identity_b = daemon._identity_for_task(revision_b)
    launched_attempts: list[int] = []

    def record_launch(task, current_state):
        launched_attempts.append(daemon._task_attempt(current_state, task))
        return {
            "task_id": task.task_id,
            "attempt": launched_attempts[-1],
            "returncode": 0,
        }

    monkeypatch.setattr(daemon, "_run_implementation", record_launch)

    result = daemon.run_once()
    updated = PortalTaskState.load(state_path)

    assert identity_b.canonical_task_cid != identity_a.canonical_task_cid
    assert launched_attempts == [1]
    assert result["attempt_limited_task_ids"] == []
    assert updated.implementation_attempts_by_cid[
        identity_a.canonical_task_cid
    ] == 1
