from __future__ import annotations

import hashlib
import json

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.lease_coordination import (
    LeaseCoordinator,
    adapt_goal_bundle,
    profile_g_cid,
)
from ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor import (
    build_arg_parser as build_bundle_arg_parser,
)
from ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor import (
    implementation_supervisor_command,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTaskState,
    parse_task_file,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    parse_args as parse_daemon_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    PortalSupervisorConfig,
    supervisor_config_from_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    parse_args as parse_supervisor_args,
)
from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
from ipfs_datasets_py.logic.profile_g import validate_profile_g_artifact


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


def _idle_heartbeat_projection(**overrides):
    projection = {
        "active_task_id": "",
        "implementation_in_progress": False,
        "ready_count": 0,
        "selectable_ready_count": 0,
        "eligible_ready_count": 0,
        "blocked_count": 0,
        "selection_idle_reason": "no_shard_selectable_ready_tasks",
    }
    projection.update(overrides)
    return projection


def _framed_grok_quota_stderr(
    raw_error: str,
    *,
    kind: str,
    http_status: int | None,
) -> str:
    from ipfs_accelerate_py.agent_supervisor.grok_cli_runner import (
        GROK_QUOTA_RECEIPT_SCHEMA,
    )

    raw_bytes = raw_error.encode("utf-8")
    receipt = {
        "schema": GROK_QUOTA_RECEIPT_SCHEMA,
        "provider": "grok_cli",
        "model": "grok-4.5",
        "failure_kind": "quota_or_balance_exhausted",
        "message": "Grok Build usage balance exhausted",
        "raw_error_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "raw_error_size": len(raw_bytes),
        "kind": kind,
        "http_status": http_status,
    }
    separator = "" if raw_error.endswith("\n") else "\n"
    return (
        raw_error
        + separator
        + json.dumps(receipt, sort_keys=True, separators=(",", ":"))
        + "\n"
    )


def test_heartbeat_fallback_accepts_strict_shard_with_global_ready_work() -> None:
    assert _projection_is_quiescent_for_heartbeat_fallback(
        _idle_heartbeat_projection(
            ready_count=3,
            blocked_count=2,
        )
    )


def test_heartbeat_fallback_accepts_resource_claim_deferral() -> None:
    assert _projection_is_quiescent_for_heartbeat_fallback(
        _idle_heartbeat_projection(
            ready_count=1,
            selection_idle_reason=(
                "all_selectable_ready_tasks_deferred_by_resource_claim"
            ),
        )
    )


def test_heartbeat_fallback_accepts_attempt_limit_backpressure() -> None:
    assert _projection_is_quiescent_for_heartbeat_fallback(
        _idle_heartbeat_projection(
            ready_count=1,
            selection_idle_reason=(
                "all_selectable_ready_tasks_reached_max_task_attempts"
            ),
        )
    )


def test_heartbeat_fallback_accepts_implementation_retry_deferral() -> None:
    assert _projection_is_quiescent_for_heartbeat_fallback(
        _idle_heartbeat_projection(
            ready_count=1,
            selection_idle_reason=(
                "implementation_retry_deferred:provider_capacity_backoff"
            ),
        )
    )
    assert not _projection_is_quiescent_for_heartbeat_fallback(
        _idle_heartbeat_projection(
            ready_count=1,
            selection_idle_reason="implementation_retry_deferred:",
        )
    )


def test_heartbeat_fallback_accepts_only_valid_empty_backlog_projection() -> None:
    empty_projection = _idle_heartbeat_projection(
        selection_idle_reason="no_tasks_found",
    )
    assert _projection_is_quiescent_for_heartbeat_fallback(empty_projection)

    for field_name in (
        "ready_count",
        "selectable_ready_count",
        "eligible_ready_count",
        "blocked_count",
    ):
        assert not _projection_is_quiescent_for_heartbeat_fallback(
            {
                **empty_projection,
                field_name: 1,
            }
        )

    for unsafe_reason in ("task_source_invalid", "todo_read_failed"):
        assert not _projection_is_quiescent_for_heartbeat_fallback(
            {
                **empty_projection,
                "selection_idle_reason": unsafe_reason,
            }
        )


@pytest.mark.parametrize(
    ("idle_reason", "selectable_ready_count", "eligible_ready_count"),
    (
        ("all_selectable_ready_tasks_deprioritized_as_off_mission", 2, 0),
        ("no_eligible_ready_tasks_after_selection_filters", 2, 0),
        ("provider_capacity_backoff", 1, 1),
        ("resource_claim_deferred:ipfs_kit_py", 1, 1),
    ),
)
def test_heartbeat_fallback_accepts_other_explicit_idle_policies(
    idle_reason,
    selectable_ready_count,
    eligible_ready_count,
) -> None:
    assert _projection_is_quiescent_for_heartbeat_fallback(
        _idle_heartbeat_projection(
            ready_count=2,
            selectable_ready_count=selectable_ready_count,
            eligible_ready_count=eligible_ready_count,
            selection_idle_reason=idle_reason,
        )
    )


@pytest.mark.parametrize(
    ("active_task_id", "implementation_in_progress"),
    (
        ("TASK-001", False),
        ("", True),
        ("TASK-001", True),
    ),
)
def test_heartbeat_fallback_rejects_active_or_implementing_projection(
    active_task_id,
    implementation_in_progress,
) -> None:
    assert not _projection_is_quiescent_for_heartbeat_fallback(
        _idle_heartbeat_projection(
            active_task_id=active_task_id,
            implementation_in_progress=implementation_in_progress,
            ready_count=1,
            selection_idle_reason=(
                "implementation_retry_deferred:provider_capacity_backoff"
            ),
        )
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
    assert _projection_is_quiescent_for_heartbeat_fallback(
        {
            "active_task_id": second["active_task_id"],
            "implementation_in_progress": (
                second_state.implementation_in_progress
            ),
            "ready_count": second["ready_count"],
            "selectable_ready_count": second["selectable_ready_count"],
            "eligible_ready_count": second["eligible_ready_count"],
            "blocked_count": second["blocked_count"],
            "selection_idle_reason": second["selection_idle_reason"],
        }
    )

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


def test_unlimited_attempts_translate_only_at_profile_g_task_spec_boundary() -> None:
    bundle = {
        "bundle_key": "objective/runtime",
        "source_todo": "docs/tasks.todo.md",
        "tasks": [{"task_id": "TASK-001"}],
        "max_attempts": 0,
    }

    adapted = adapt_goal_bundle(bundle, created_at_ms=1_783_872_000_000)

    assert bundle["max_attempts"] == 0
    assert adapted["task"]["max_attempts"] == 100
    assert (
        validate_profile_g_artifact("TaskSpec", adapted["task"])
        == adapted["task_cid"]
    )

    finite = adapt_goal_bundle(
        {**bundle, "max_attempts": 4},
        created_at_ms=1_783_872_000_000,
    )
    assert finite["task"]["max_attempts"] == 4


def test_default_planned_lane_is_unlimited_in_worker_and_coordinator(
    tmp_path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    bundle_dir = repo / "bundles"
    bundle_dir.mkdir()
    shard_path = bundle_dir / "runtime.todo.md"
    shard_path.write_text(
        """## TASK-001 Unlimited task

- Status: todo
""",
        encoding="utf-8",
    )
    index_path = bundle_dir / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "source_todo": "docs/tasks.todo.md",
                "bundles": {
                    "objective/runtime": {
                        "shard_path": "runtime.todo.md",
                        "parallel_lane": "objective/runtime",
                        "tasks": [{"task_id": "TASK-001"}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    [lane] = plan_bundle_lanes(
        bundle_index_path=index_path,
        repo_root=repo,
        state_root=repo / "state",
        worktree_root=repo / "worktrees",
        log_dir=repo / "logs",
        task_prefix="TASK-",
        optimize_bundles=False,
    )

    worker_flag = lane.command.index("--max-task-attempts")
    assert lane.command[worker_flag + 1] == "0"
    assert lane.queue_payload["max_attempts"] == 0
    profile_g = lane.queue_payload["profile_g"]
    assert profile_g["task"]["max_attempts"] == 100
    assert (
        validate_profile_g_artifact("TaskSpec", profile_g["task"])
        == profile_g["task_cid"]
    )
    with LeaseCoordinator(repo / "coordination.duckdb") as coordinator:
        registered = coordinator.register_bundle(lane.queue_payload)
        for expected_attempt in range(1, 5):
            grant = coordinator.claim(
                registered["task_cid"],
                "did:web:lane.example",
            )
            assert grant.attempt == expected_attempt
            coordinator.release(grant, reason="retry")

    task_queue = TaskQueue(str(repo / "task-queue.duckdb"))
    try:
        [submitted_id] = submit_bundle_tasks(index_path, queue=task_queue)
        assert task_queue.get(submitted_id)["max_attempts"] == 0
        for expected_attempt in range(1, 5):
            claimed = task_queue.claim_next(worker_id="worker-a")
            assert claimed is not None
            assert claimed.attempt == expected_attempt
            assert claimed.max_attempts == 0
            assert task_queue.retry(
                task_id=submitted_id,
                worker_id="worker-a",
                error="retryable",
            )
        expiring = task_queue.claim_next(
            worker_id="worker-a",
            lease_seconds=1,
        )
        assert expiring is not None
        assert expiring.attempt == 5
        assert expiring.lease_until is not None
        assert task_queue.recover_expired_leases(
            now=expiring.lease_until + 1,
        ) == 1
        recovered = task_queue.claim_next(worker_id="worker-b")
        assert recovered is not None
        assert recovered.attempt == 6
    finally:
        task_queue.close()


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
    classified = classify_provider_capacity_failure(
        _framed_grok_quota_stderr(
            text,
            kind="usage_balance_exhausted",
            http_status=402,
        ),
        provider_labels=("grok",),
        provider_returncode=86,
    )
    assert classified["exhausted"] is True
    assert classified["providers"] == ["grok"]
    assert classified["reason"] == "provider_capacity_exhausted"
    assert classified["capacity_failure_kind"] == "quota_or_balance_exhausted"
    assert classified["provider_attribution"] == "implementation_command"
    assert classified["fallback_eligible"] is True
    assert classified["fallback_trigger"] == "primary_quota_exhausted"


def test_classify_generic_usage_limit_uses_dispatched_grok_attribution() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        classify_provider_capacity_failure,
    )

    classified = classify_provider_capacity_failure(
        _framed_grok_quota_stderr(
            "You've hit your usage limit.",
            kind="usage_limit",
            http_status=None,
        ),
        provider_labels=("grok",),
        provider_returncode=86,
    )

    assert classified["providers"] == ["grok"]
    assert classified["capacity_failure_kind"] == "quota_or_balance_exhausted"
    assert classified["provider_attribution"] == "implementation_command"
    assert classified["fallback_eligible"] is True


def test_framed_quota_receipt_requires_trusted_runner_exit_code() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        classify_provider_capacity_failure,
    )

    framed = _framed_grok_quota_stderr(
        "You've hit your usage limit.",
        kind="usage_limit",
        http_status=None,
    )
    classified = classify_provider_capacity_failure(
        framed,
        provider_labels=("grok",),
        provider_returncode=1,
    )

    assert classified["fallback_eligible"] is False
    assert classified["fallback_trigger"] == ""


@pytest.mark.parametrize(
    "text",
    (
        "GitHub API quota exceeded while fetching PR",
        "Hugging Face usage balance exhausted",
        "Test fixture: quota exhausted",
        "nested test says xAI usage balance exhausted",
    ),
)
def test_nested_service_quota_text_does_not_impersonate_grok(
    text: str,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        classify_provider_capacity_failure,
    )

    classified = classify_provider_capacity_failure(
        text,
        provider_labels=("grok",),
    )

    assert classified["exhausted"] is False
    assert classified["providers"] == []
    assert classified["fallback_eligible"] is False
    assert classified["fallback_trigger"] == ""


def test_unstructured_grok_quota_prose_cannot_authorize_fallback() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        classify_provider_capacity_failure,
    )

    classified = classify_provider_capacity_failure(
        "unit fixture status 402 then Grok Build usage balance exhausted",
        provider_labels=("grok",),
        provider_returncode=86,
    )

    assert classified["fallback_eligible"] is False
    assert classified["fallback_trigger"] == ""


@pytest.mark.parametrize(
    "text",
    (
        "xAI HTTP 429: too many requests",
        "Grok is temporarily overloaded: resource exhausted",
        "Grok authentication failed; login required",
        "Grok service unavailable",
    ),
)
def test_classify_grok_nonquota_failures_do_not_authorize_codex_fallback(
    text: str,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        classify_provider_capacity_failure,
    )

    classified = classify_provider_capacity_failure(
        text,
        provider_labels=("grok",),
    )

    assert classified["fallback_eligible"] is False
    assert classified["fallback_trigger"] == ""
    assert classified["capacity_failure_kind"] != "quota_or_balance_exhausted"


@pytest.mark.parametrize(
    "diagnostic",
    (
        "Grok CLI failed without a terminal-correlated native quota record; "
        "Codex fallback is forbidden",
        "Independent pinned Grok-4.5 verifier did not confirm quota; "
        "Codex fallback is forbidden",
        "The workspace changed while Grok quota was being verified; "
        "Codex fallback is forbidden",
    ),
)
def test_classify_provider_capacity_ignores_grok_policy_diagnostic(
    diagnostic: str,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        classify_provider_capacity_failure,
    )

    text = (
        "PermissionError: [Errno 13] Permission denied: "
        "'/run/ipfs-accelerate/prompt.md'\n"
        f"{diagnostic}\n"
    )

    classified = classify_provider_capacity_failure(text)

    assert classified == {"exhausted": False, "providers": [], "reason": ""}


def test_classify_codex_quota_does_not_poison_grok_capacity() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        classify_provider_capacity_failure,
    )

    classified = classify_provider_capacity_failure(
        "You've hit your usage limit. Try again later."
    )

    assert classified == {
        "exhausted": True,
        "providers": ["codex"],
        "reason": "provider_capacity_exhausted",
        "capacity_failure_kind": "provider_capacity_exhausted",
        "provider_attribution": "log_text",
        "fallback_eligible": False,
        "fallback_trigger": "",
    }


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


def test_semantic_key_revision_resets_projection_without_inheriting_backpressure(
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
        max_task_attempts=5,
        worktree_pool_enabled=False,
    )
    old_task = parse_task_file(todo_path, "## TASK-")[0]
    daemon._register_task_identities([old_task])
    old_identity = daemon._identity_for_task(old_task)
    latch_body = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "protected-implementation-attempt-latch@1"
        ),
        "task_id": old_task.task_id,
        "attempt": 5,
        "task_revision_cid": old_identity.canonical_task_cid,
        "board_namespace": old_task.board_namespace,
        "route_id": "route:old-revision",
        "invocation_id": "invocation:old-revision",
        "logical_attempt_id": "logical:old-revision",
        "worktree_id": "worktree:old-revision",
        "provider_attempt_store": str(state_dir / "provider-attempts"),
        "provider_attempt_store_identity": "sha256:" + "a" * 64,
    }
    latch = {
        **latch_body,
        "latch_id": content_identity(latch_body),
    }
    latch_key = daemon._protected_attempt_latch_key(
        old_task.task_id,
        5,
        old_identity.canonical_task_cid,
    )
    old_state = PortalTaskState(
        task_identities={old_task.task_id: old_identity.to_dict()},
        implementation_attempts={old_task.task_id: 5},
        implementation_attempts_by_cid={old_identity.canonical_task_cid: 5},
        protected_implementation_attempts={latch_key: latch},
    )
    old_state.save(state_path)
    daemon.task_queue.record_failure(
        old_identity.canonical_task_cid,
        reason="old revision infrastructure failure",
    )
    daemon.task_queue.save()
    assert daemon.task_queue.is_cooled_down(old_identity.canonical_task_cid)
    assert daemon._durable_protected_recovery_attempt(old_state, old_task) == 5

    todo_path.write_text(
        todo_path.read_text(encoding="utf-8")
        + "- Semantic key: provider-effect-retry-revision@1\n",
        encoding="utf-8",
    )
    new_task = parse_task_file(todo_path, "## TASK-")[0]
    new_identity = daemon._identity_for_task(new_task)
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

    result = daemon.run_once()
    updated = PortalTaskState.load(state_path)

    assert new_task.acceptance == old_task.acceptance
    assert new_identity.canonical_task_cid != old_identity.canonical_task_cid
    assert launched_attempts == [1]
    assert result["attempt_limited_task_ids"] == []
    assert updated.implementation_attempts == {old_task.task_id: 1}
    assert updated.implementation_attempts_by_cid == {
        old_identity.canonical_task_cid: 5,
        new_identity.canonical_task_cid: 1,
    }
    assert updated.task_identities[old_task.task_id][
        "canonical_task_cid"
    ] == new_identity.canonical_task_cid
    assert updated.protected_implementation_attempts[latch_key] == latch
    assert daemon._durable_protected_recovery_attempt(updated, old_task) == 5
    assert daemon._durable_protected_recovery_attempt(updated, new_task) is None
    assert (
        daemon.task_queue.is_cooled_down(new_identity.canonical_task_cid)
        is False
    )
