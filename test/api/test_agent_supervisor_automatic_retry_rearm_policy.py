"""Strict campaign attempt caps survive automatic recovery and restart."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    AUTOMATIC_RETRY_REARM_ENV,
    PortalImplementationDaemon,
    PortalTaskState,
    automatic_retry_rearm_enabled,
    parse_task_file,
)
from test.api.test_agent_supervisor_retry_unstall_regression import (
    _source_and_repair_board,
)
from test.api.test_agent_supervisor_stale_replay_unstall import (
    _exhausted_ordinary_task,
)
from test.api.test_agent_supervisor_zero_provider_rearm_evidence import (
    _native_callback,
)


@pytest.mark.parametrize("environment,expected", [({}, True),
    ({AUTOMATIC_RETRY_REARM_ENV: "0"}, False),
    ({AUTOMATIC_RETRY_REARM_ENV: "1"}, True)])
def test_campaign_policy_has_closed_explicit_settings(environment, expected):
    assert automatic_retry_rearm_enabled(environment) is expected


@pytest.mark.parametrize("value", ["", "false", "true", "off", "2", " 0", "0\n"])
def test_invalid_campaign_policy_is_rejected_before_constructor_mutation(
    tmp_path, monkeypatch, value,
):
    monkeypatch.setenv(AUTOMATIC_RETRY_REARM_ENV, value)
    with pytest.raises(ValueError, match=AUTOMATIC_RETRY_REARM_ENV):
        PortalImplementationDaemon(
            todo_path=tmp_path / "missing.todo.md",
            state_path=tmp_path / "state" / "state.json",
            strategy_path=tmp_path / "state" / "strategy.json",
            events_path=tmp_path / "state" / "events.jsonl",
            repo_root=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def _daemon(tmp_path: Path, *, repair=False):
    todo = tmp_path / "tasks.todo.md"
    if repair:
        _source_and_repair_board(todo)
    else:
        todo.write_text("""# Tasks

## TEST-001 Keep the lifetime attempt cap

- Status: todo
- Priority: P0
- Outputs: src/capped.py
- Acceptance: Three consumed attempts stay spent across source repairs.
""")
    return PortalImplementationDaemon(
        todo_path=todo, state_path=tmp_path / "state" / "task_state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl", repo_root=tmp_path,
        task_header_prefix="## TEST-", implement=True, max_task_attempts=3,
        worktree_pool_enabled=False,
    )


def _exhaust(daemon, task):
    daemon._register_task_identities((task,))
    identity = daemon._identity_for_task(task)
    cid = identity.canonical_task_cid
    state = PortalTaskState(
        task_identities={task.task_id: identity.to_dict()},
        implementation_attempts={task.task_id: 3},
        implementation_attempts_by_cid={cid: 3},
        last_implementation_task_id=task.task_id,
        last_implementation_task_cid=cid,
        last_implementation_returncode=1,
        retry_budget_repair_rearm_receipts={cid: ["historical-repair-rearm"]},
        stale_proposal_replay_rearm_receipts={cid: ["historical-stale-rearm"]},
        validation_obsolescence_rearm_receipts={cid: ["historical-validation-rearm"]},
    )
    state.save(daemon.state_path)
    daemon.task_queue.register_task(
        identity, priority=task.priority, track=task.track,
    ).record_failure("provider_max_turns")
    daemon.task_queue.save()
    daemon._record_event("implementation_finished", {
        "task_id": task.task_id, "canonical_task_cid": cid,
        "attempt": 3, "returncode": 1, "attempt_consumed": True,
        "provider_dispatched": True,
        "validation_result": {"attempted": False, "passed": False,
            "reason": "implementation_provider_failed"},
    })
    return state, cid


def _queue(daemon):
    return {key: entry.to_dict() for key, entry in daemon.task_queue.entries.items()}


def test_three_spent_attempts_stay_blocked_across_source_revisions_and_restart(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv(AUTOMATIC_RETRY_REARM_ENV, "0")
    daemon = _daemon(tmp_path)
    task = parse_task_file(daemon.todo_path, "## TEST-")[0]
    state, cid = _exhaust(daemon, task)
    histories = [deepcopy(getattr(state, field)) for field in (
        "retry_budget_repair_rearm_receipts", "stale_proposal_replay_rearm_receipts",
        "validation_obsolescence_rearm_receipts")]
    queue_before = _queue(daemon)

    for revision in ("source-before-fix", "source-after-fix"):
        monkeypatch.setattr(daemon, "_retry_budget_repair_runtime_revision", lambda: revision)
        def forbidden_provider(*args, **kwargs):
            pytest.fail("strict cap must prevent a fourth provider attempt")
        monkeypatch.setattr(daemon, "_run_implementation", forbidden_provider)
        result = daemon.run_once()
        persisted = PortalTaskState.load(daemon.state_path)
        assert result["implementation_result"] is None
        assert result["attempt_limited_task_ids"] == [task.task_id]
        assert result["selection_idle_reason"] == "all_selectable_ready_tasks_reached_max_task_attempts"
        assert persisted.implementation_attempts[task.task_id] == 3
        assert persisted.implementation_attempts_by_cid[cid] == 3
        assert [getattr(persisted, field) for field in (
            "retry_budget_repair_rearm_receipts", "stale_proposal_replay_rearm_receipts",
            "validation_obsolescence_rearm_receipts")] == histories
        assert _queue(daemon) == queue_before
        # Reconstruct the actual native daemon and reload its durable queue.
        daemon = PortalImplementationDaemon(
            todo_path=daemon.todo_path, state_path=daemon.state_path,
            strategy_path=daemon.strategy_path, events_path=daemon.events_path,
            repo_root=tmp_path, task_header_prefix="## TEST-", implement=True,
            max_task_attempts=3, worktree_pool_enabled=False,
        )
    assert not any(event.get("type", "").endswith("_rearmed") for event in daemon._iter_events())


@pytest.mark.parametrize("kind", ["validation", "repair", "stale"])
def test_all_automatic_rearm_entrypoints_preserve_state_queue_and_history(
    tmp_path, monkeypatch, kind,
):
    monkeypatch.setenv(AUTOMATIC_RETRY_REARM_ENV, "0")
    if kind == "stale":
        daemon, task, state, _, cid = _exhausted_ordinary_task(tmp_path)
        method = daemon._rearm_attempt_limited_stale_proposal_tasks
    else:
        daemon = _daemon(tmp_path, repair=kind == "repair")
        task = parse_task_file(daemon.todo_path, "## TEST-")[-1]
        state, cid = _exhaust(daemon, task)
        method = (daemon._rearm_attempt_limited_retry_repairs if kind == "repair"
            else daemon._rearm_attempt_limited_obsoleted_validation_tasks)
    before = deepcopy(asdict(state))
    queue_before = _queue(daemon)
    events_before = daemon.events_path.read_bytes()
    for revision in ("source-revision-A", "source-revision-B"):
        assert method(state, (task,), {task.task_id: "ready"}, recovery_revision=revision) == []
    assert asdict(state) == before
    assert asdict(PortalTaskState.load(daemon.state_path)) == before
    assert _queue(daemon) == queue_before
    assert daemon.events_path.read_bytes() == events_before


def test_completed_generated_repair_cannot_reset_source_budget_when_disabled(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv(AUTOMATIC_RETRY_REARM_ENV, "0")
    daemon = _daemon(tmp_path, repair=True)
    daemon.todo_path.write_text(daemon.todo_path.read_text().replace(
        "- Status: todo\n- Completion: manual", "- Status: completed\n- Completion: manual"))
    tasks = parse_task_file(daemon.todo_path, "## TEST-")
    state, cid = _exhaust(daemon, tasks[0])
    before = deepcopy(asdict(state))
    queue_before = _queue(daemon)
    assert daemon._reset_attempt_budgets_for_completed_retry_repairs(state, tasks) == ([], [])
    assert asdict(state) == before
    assert asdict(PortalTaskState.load(daemon.state_path)) == before
    assert _queue(daemon) == queue_before
    assert state.implementation_attempts_by_cid[cid] == 3


def test_unfinished_dispatched_attempt_remains_consumed_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv(AUTOMATIC_RETRY_REARM_ENV, "0")
    daemon = _daemon(tmp_path)
    task = parse_task_file(daemon.todo_path, "## TEST-")[0]
    state, cid = _exhaust(daemon, task)
    state.active_task_id = task.task_id
    state.active_task_cid = cid
    state.implementation_in_progress = True
    before = deepcopy(asdict(state))
    result = daemon._release_unfinished_active_attempt(state, task_id=task.task_id, attempt=3)
    assert result["consumed"] is True
    assert result["released"] is False
    assert result["reason"] == "automatic_retry_rearm_disabled"
    assert asdict(state) == before


def test_invalid_policy_after_construction_prevents_run_mutation(tmp_path, monkeypatch):
    monkeypatch.delenv(AUTOMATIC_RETRY_REARM_ENV, raising=False)
    daemon = _daemon(tmp_path)
    task = parse_task_file(daemon.todo_path, "## TEST-")[0]
    state, _ = _exhaust(daemon, task)
    state_before = asdict(state)
    queue_before = _queue(daemon)
    event_bytes = daemon.events_path.read_bytes()
    monkeypatch.setenv(AUTOMATIC_RETRY_REARM_ENV, "false")
    calls = [daemon.run_once, daemon._run_once, lambda:
        daemon._reset_attempt_budgets_for_completed_retry_repairs(state, (task,)),
        lambda: daemon._release_unfinished_active_attempt(
            state, task_id=task.task_id, attempt=3)]
    for method in (daemon._rearm_attempt_limited_obsoleted_validation_tasks,
        daemon._rearm_attempt_limited_retry_repairs,
        daemon._rearm_attempt_limited_stale_proposal_tasks):
        calls.append(lambda method=method: method(
            state, (task,), {task.task_id: "ready"}, recovery_revision="new-source"))
    for invoke in calls:
        with pytest.raises(ValueError, match=AUTOMATIC_RETRY_REARM_ENV):
            invoke()
    assert asdict(state) == state_before
    assert asdict(PortalTaskState.load(daemon.state_path)) == state_before
    assert _queue(daemon) == queue_before
    assert daemon.events_path.read_bytes() == event_bytes


def test_verified_undispatched_callback_keeps_separate_recovery_authority(tmp_path, monkeypatch):
    monkeypatch.setenv(AUTOMATIC_RETRY_REARM_ENV, "0")
    bridge, attempt, paths = _native_callback(tmp_path)
    before = {p.name: p.read_bytes() for p in paths.root.iterdir() if p.is_file()}
    assert bridge.zero_provider_failure_rearm_ready(attempt)
    assert before == {p.name: p.read_bytes() for p in paths.root.iterdir() if p.is_file()}


@pytest.mark.parametrize("setting", [None, "1"])
def test_enabled_legacy_rearm_remains_once_per_source_revision(tmp_path, monkeypatch, setting):
    if setting is None:
        monkeypatch.delenv(AUTOMATIC_RETRY_REARM_ENV, raising=False)
    else:
        monkeypatch.setenv(AUTOMATIC_RETRY_REARM_ENV, setting)
    daemon = _daemon(tmp_path)
    task = parse_task_file(daemon.todo_path, "## TEST-")[0]
    state, cid = _exhaust(daemon, task)
    rearm = daemon._rearm_attempt_limited_obsoleted_validation_tasks
    assert len(rearm(state, (task,), {task.task_id: "ready"}, recovery_revision="revision-A")) == 1
    persisted = PortalTaskState.load(daemon.state_path)
    assert task.task_id not in persisted.implementation_attempts
    assert cid not in persisted.implementation_attempts_by_cid
    assert len(persisted.validation_obsolescence_rearm_receipts[cid]) == 2
    assert not daemon.task_queue.is_cooled_down(cid)
    persisted.implementation_attempts[task.task_id] = 3
    persisted.implementation_attempts_by_cid[cid] = 3
    persisted.save(daemon.state_path)
    assert rearm(persisted, (task,), {task.task_id: "ready"}, recovery_revision="revision-A") == []
    assert persisted.implementation_attempts_by_cid[cid] == 3
    assert len(rearm(persisted, (task,), {task.task_id: "ready"}, recovery_revision="revision-B")) == 1
    assert len(PortalTaskState.load(daemon.state_path).validation_obsolescence_rearm_receipts[cid]) == 3


def test_new_canonical_revision_keeps_old_spent_count_and_fresh_budget(tmp_path, monkeypatch):
    monkeypatch.setenv(AUTOMATIC_RETRY_REARM_ENV, "0")
    daemon = _daemon(tmp_path)
    previous = parse_task_file(daemon.todo_path, "## TEST-")[0]
    state, old_cid = _exhaust(daemon, previous)
    daemon.todo_path.write_text(daemon.todo_path.read_text().replace(
        "Three consumed attempts stay spent across source repairs.",
        "A materially revised canonical task has a separate budget."))
    revised = parse_task_file(daemon.todo_path, "## TEST-")[0]
    new_cid = daemon._identity_for_task(revised).canonical_task_cid
    assert old_cid != new_cid
    assert daemon._task_attempt(state, revised) == 1
    assert state.implementation_attempts_by_cid[old_cid] == 3
