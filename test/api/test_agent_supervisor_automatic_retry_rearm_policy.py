"""Strict campaign attempt caps survive automatic recovery and restart."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
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
        validation_obsolescence_rearm_receipts={},
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
@pytest.mark.parametrize("returncode,reason", [
    (124, "validation_command_failed"),
    (1, "validation_command_failed"),
    (78, "proposal_gate_failed"),
])
def test_source_revision_does_not_rearm_ordinary_validation_failures(
    tmp_path, monkeypatch, setting, returncode, reason,
):
    if setting is None:
        monkeypatch.delenv(AUTOMATIC_RETRY_REARM_ENV, raising=False)
    else:
        monkeypatch.setenv(AUTOMATIC_RETRY_REARM_ENV, setting)
    assert automatic_retry_rearm_enabled() is True
    daemon = _daemon(tmp_path)
    daemon.max_task_attempts = 4
    task = parse_task_file(daemon.todo_path, "## TEST-")[0]
    state, cid = _exhaust(daemon, task)
    state.implementation_attempts[task.task_id] = 4
    state.implementation_attempts_by_cid[cid] = 4
    state.last_implementation_returncode = returncode
    state.save(daemon.state_path)
    daemon._record_event("implementation_finished", {
        "task_id": task.task_id, "canonical_task_cid": cid,
        "canonical_task_key": daemon._identity_for_task(task).canonical_task_key,
        "attempt": 4, "returncode": returncode, "attempt_consumed": True,
        "provider_dispatched": True,
        "validation_result": {"attempted": True, "passed": False,
            "reason": reason, "timed_out": returncode == 124,
            "elapsed_seconds": 1800.0 if returncode == 124 else 1.0},
    })
    histories = deepcopy(state.validation_obsolescence_rearm_receipts)
    queue_before = _queue(daemon)

    # A control-plane replacement creates a new native daemon, then polling
    # repeats under the same replacement. Neither transition grants budget.
    for revision in ("old-control-plane", "new-utc-head-tail-control-plane",
                     "new-utc-head-tail-control-plane"):
        monkeypatch.setattr(daemon, "_retry_budget_repair_runtime_revision",
                            lambda revision=revision: revision)
        def forbidden_provider(*args, **kwargs):
            pytest.fail("source-only revision must not dispatch a fifth attempt")
        monkeypatch.setattr(daemon, "_run_implementation", forbidden_provider)
        result = daemon.run_once()
        persisted = PortalTaskState.load(daemon.state_path)
        assert result["implementation_result"] is None
        assert result["attempt_limited_task_ids"] == [task.task_id]
        assert persisted.implementation_attempts[task.task_id] == 4
        assert persisted.implementation_attempts_by_cid[cid] == 4
        assert persisted.validation_obsolescence_rearm_receipts == histories
        assert _queue(daemon) == queue_before
        daemon = PortalImplementationDaemon(
            todo_path=daemon.todo_path, state_path=daemon.state_path,
            strategy_path=daemon.strategy_path, events_path=daemon.events_path,
            repo_root=tmp_path, task_header_prefix="## TEST-", implement=True,
            max_task_attempts=4, worktree_pool_enabled=False,
        )
        assert daemon._task_attempt_count(persisted, task) == 4
    assert not any(event.get("type") == "ordinary_task_validation_obsolescence_rearmed"
                   for event in daemon._iter_events())


def test_completed_board_bound_repair_keeps_separate_once_only_reset(
    tmp_path, monkeypatch,
):
    monkeypatch.delenv(AUTOMATIC_RETRY_REARM_ENV, raising=False)
    daemon = _daemon(tmp_path, repair=True)
    tasks = parse_task_file(daemon.todo_path, "## TEST-")
    state, cid = _exhaust(daemon, tasks[0])
    daemon.todo_path.write_text(daemon.todo_path.read_text().replace(
        "- Status: todo\n- Completion: manual", "- Status: completed\n- Completion: manual"))
    tasks = parse_task_file(daemon.todo_path, "## TEST-")
    resets, deferred = daemon._reset_attempt_budgets_for_completed_retry_repairs(state, tasks)
    assert deferred == [] and len(resets) == 1
    assert resets[0]["source_task_id"] == tasks[0].task_id
    assert resets[0]["repair_task_id"] == tasks[1].task_id
    assert resets[0]["previous_canonical_attempt_counts"][cid] == 3
    assert state.retry_budget_repair_receipts[tasks[0].task_id] == tasks[1].task_id
    assert tasks[0].task_id not in state.implementation_attempts
    assert cid not in state.implementation_attempts_by_cid

    # The same completed repair remains consumed after another generation.
    state.implementation_attempts[tasks[0].task_id] = 3
    state.implementation_attempts_by_cid[cid] = 3
    state.save(daemon.state_path)
    restarted = PortalImplementationDaemon(
        todo_path=daemon.todo_path, state_path=daemon.state_path,
        strategy_path=daemon.strategy_path, events_path=daemon.events_path,
        repo_root=tmp_path, task_header_prefix="## TEST-", implement=True,
        max_task_attempts=3, worktree_pool_enabled=False,
    )
    restored = PortalTaskState.load(restarted.state_path)
    assert restarted._reset_attempt_budgets_for_completed_retry_repairs(restored, tasks) == ([], [])
    assert restored.implementation_attempts_by_cid[cid] == 3


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


def _post_obsolescence_rearm(daemon, task, state, *, previous=4, revision="old-unrelated-control-plane-revision"):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as owner
    identity = daemon._identity_for_task(task)
    cid = identity.canonical_task_cid
    fingerprint = "sha256:" + hashlib.sha256(json.dumps({
        "schema": owner.VALIDATION_OBSOLESCENCE_REARM_POLICY_REVISION,
        "recovery_revision": revision, "canonical_task_cid": cid,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    daemon._record_event("ordinary_task_validation_obsolescence_rearmed", {
        "schema": owner.VALIDATION_OBSOLESCENCE_REARM_POLICY_REVISION,
        "recovery_revision": revision,
        "rearmed": [{"task_id": task.task_id, "canonical_task_cid": cid,
            "previous_attempt_count": previous,
            "previous_canonical_attempt_count": previous,
            "previous_display_attempt_counts": {task.task_id: previous},
            "recovery_revision": revision, "rearm_fingerprint": fingerprint}],
    })
    state.validation_obsolescence_rearm_receipts[cid] = [fingerprint]
    state.implementation_attempts[task.task_id] = 1
    state.implementation_attempts_by_cid[cid] = 1
    state.save(daemon.state_path)
    daemon._record_event("implementation_finished", {
        "task_id": task.task_id, "canonical_task_cid": cid,
        "canonical_task_key": identity.canonical_task_key,
        "attempt": 1, "returncode": 124, "attempt_consumed": True,
        "provider_dispatched": True,
        "validation_result": {"attempted": True, "passed": False,
            "reason": "validation_command_failed", "timed_out": True},
    })
    return cid, fingerprint


def test_historical_reset_floor_waits_for_quiescence_and_survives_restart(tmp_path, monkeypatch):
    daemon = _daemon(tmp_path)
    daemon.max_task_attempts = 4
    task = parse_task_file(daemon.todo_path, "## TEST-")[0]
    state, _ = _exhaust(daemon, task)
    cid, fingerprint = _post_obsolescence_rearm(daemon, task, state)
    original_history = daemon.events_path.read_bytes()
    state.active_task_id, state.active_task_cid = task.task_id, cid
    state.implementation_in_progress = True
    selectable, limited = daemon._partition_tasks_at_attempt_limit((task,), {task.task_id: "ready"}, state)
    assert selectable == [task] and limited == []
    assert state.implementation_attempts_by_cid[cid] == 1
    assert daemon.events_path.read_bytes() == original_history

    state.active_task_id = state.active_task_cid = ""
    state.implementation_in_progress = False
    state.save(daemon.state_path)
    selectable, limited = daemon._partition_tasks_at_attempt_limit((task,), {task.task_id: "ready"}, state)
    assert selectable == [] and limited[0]["attempt_count"] == 4
    assert state.implementation_attempts_by_cid[cid] == 4
    assert state.implementation_attempts[task.task_id] == 4
    assert state.validation_obsolescence_rearm_receipts[cid] == [fingerprint]
    assert daemon.events_path.read_bytes().startswith(original_history)
    assert [event["attempt"] for event in daemon._iter_events()
            if event.get("type") == "implementation_finished"] == [3, 1]

    restarted = PortalImplementationDaemon(
        todo_path=daemon.todo_path, state_path=daemon.state_path,
        strategy_path=daemon.strategy_path, events_path=daemon.events_path,
        repo_root=tmp_path, task_header_prefix="## TEST-", implement=True,
        max_task_attempts=4, worktree_pool_enabled=False,
    )
    def forbidden_provider(*args, **kwargs):
        pytest.fail("authenticated historical floor must block another provider attempt")
    monkeypatch.setattr(restarted, "_run_implementation", forbidden_provider)
    result = restarted.run_once()
    assert result["implementation_result"] is None
    assert result["attempt_limited_task_ids"] == [task.task_id]
    assert PortalTaskState.load(restarted.state_path).implementation_attempts_by_cid[cid] == 4
    assert len([event for event in restarted._iter_events()
                if event.get("type") == "ordinary_task_validation_attempt_floor_restored"]) == 1


@pytest.mark.parametrize("failure", ["too_many_files", "too_many_bytes", "malformed", "fingerprint"])
def test_unknown_historical_spent_budget_fails_closed_before_dispatch(tmp_path, monkeypatch, failure):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as owner
    daemon = _daemon(tmp_path)
    daemon.max_task_attempts = 4
    task = parse_task_file(daemon.todo_path, "## TEST-")[0]
    state, _ = _exhaust(daemon, task)
    cid, _ = _post_obsolescence_rearm(daemon, task, state)
    if failure == "too_many_files":
        monkeypatch.setattr(owner, "event_log_sources", lambda *a, **k: [tmp_path / str(i) for i in range(65)])
    elif failure == "too_many_bytes":
        large = daemon.events_path.parent / "oversized-history.jsonl"
        with large.open("wb") as handle:
            handle.truncate(owner.VALIDATION_REARM_FLOOR_MAX_JOURNAL_BYTES + 1)
        monkeypatch.setattr(owner, "event_log_sources", lambda *a, **k: [large])
    elif failure == "malformed":
        with daemon.events_path.open("ab") as handle:
            handle.write(b"not-json\n")
    else:
        state.validation_obsolescence_rearm_receipts[cid] = ["sha256:" + "0" * 64]
    before = daemon.events_path.read_bytes()
    selected, limited = daemon._partition_tasks_at_attempt_limit((task,), {task.task_id: "ready"}, state)
    assert selected == [] and limited[0]["reason"] == "validation_rearm_floor_evidence_unverified"
    assert limited[0]["new_attempt_authority"] is False
    assert state.implementation_attempts_by_cid[cid] == 1
    assert PortalTaskState.load(daemon.state_path).implementation_attempts_by_cid[cid] == 1
    assert daemon.events_path.read_bytes() == before


def test_completed_repair_supersedes_an_older_authenticated_floor(tmp_path):
    daemon = _daemon(tmp_path, repair=True)
    tasks = parse_task_file(daemon.todo_path, "## TEST-")
    state, _ = _exhaust(daemon, tasks[0])
    cid, fingerprint = _post_obsolescence_rearm(daemon, tasks[0], state)
    daemon.todo_path.write_text(daemon.todo_path.read_text().replace(
        "- Status: todo\n- Completion: manual", "- Status: completed\n- Completion: manual"))
    tasks = parse_task_file(daemon.todo_path, "## TEST-")
    resets, _ = daemon._reset_attempt_budgets_for_completed_retry_repairs(state, tasks)
    assert len(resets) == 1
    selectable, limited = daemon._partition_tasks_at_attempt_limit((tasks[0],), {tasks[0].task_id: "ready"}, state)
    assert selectable == [tasks[0]] and limited == []
    assert cid not in state.implementation_attempts_by_cid
    assert cid not in state.validation_obsolescence_rearm_receipts
    assert any(event.get("type") == "ordinary_task_validation_obsolescence_rearmed"
               for event in daemon._iter_events())


def test_new_receipt_free_identity_does_not_acquire_legacy_history(tmp_path, monkeypatch):
    daemon = _daemon(tmp_path)
    daemon.max_task_attempts = 4
    task = parse_task_file(daemon.todo_path, "## TEST-")[0]
    state, cid = _exhaust(daemon, task)
    state.implementation_attempts[task.task_id] = 1
    state.implementation_attempts_by_cid[cid] = 1
    state.save(daemon.state_path)
    monkeypatch.setattr(daemon, "_iter_merge_lifecycle_events",
                        lambda **kwargs: pytest.fail("new identity must not scan legacy history"))
    selected, limited = daemon._partition_tasks_at_attempt_limit((task,), {task.task_id: "ready"}, state)
    assert selected == [task] and limited == []
    assert state.validation_obsolescence_rearm_receipts == {}
    assert state.implementation_attempts_by_cid[cid] == 1


@pytest.mark.parametrize("kind,allowed", [("files", True), ("files", False), ("bytes", True), ("bytes", False)])
def test_named_caller_evidence_budget_exact_boundaries_precede_manifest_reads(tmp_path, monkeypatch, kind, allowed):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as owner
    daemon = _daemon(tmp_path)
    daemon.events_path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "files":
        sources = []
        for index in range(64 if allowed else 65):
            source = daemon.events_path.parent / ("segment-" + str(index))
            source.write_bytes(b"")
            sources.append(source)
    else:
        source = daemon.events_path.parent / "sparse-boundary"
        with source.open("wb") as handle:
            handle.truncate(owner.VALIDATION_REARM_FLOOR_MAX_JOURNAL_BYTES + (0 if allowed else 1))
        sources = [source]
    monkeypatch.setattr(owner, "event_log_sources", lambda *a, **k: sources)
    class ReplayReached(Exception):
        pass
    def manifest(*args):
        raise ReplayReached("caller preflight allowed exact physical boundary")
    monkeypatch.setattr(owner, "event_log_manifest", manifest)
    expected = ReplayReached if allowed else owner.CursorReplayError
    with pytest.raises(expected):
        daemon._iter_merge_lifecycle_events(
            maximum_evidence_files=64,
            maximum_evidence_bytes=owner.VALIDATION_REARM_FLOOR_MAX_JOURNAL_BYTES)


@pytest.mark.parametrize("alias", ["symlink", "hardlink", "duplicate"])
def test_named_caller_evidence_budget_refuses_physical_alias_before_manifest(tmp_path, monkeypatch, alias):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as owner
    daemon = _daemon(tmp_path)
    daemon.events_path.parent.mkdir(parents=True, exist_ok=True)
    source = daemon.events_path.parent / "source"
    source.write_bytes(b"bounded fixture\n")
    other = daemon.events_path.parent / "alias"
    if alias == "symlink":
        other.symlink_to(source)
        sources = [other]
    elif alias == "hardlink":
        other.hardlink_to(source)
        sources = [source, other]
    else:
        sources = [source, source]
    monkeypatch.setattr(owner, "event_log_sources", lambda *a, **k: sources)
    monkeypatch.setattr(owner, "event_log_manifest", lambda *a: pytest.fail("unsafe evidence reached replay"))
    with pytest.raises(owner.CursorReplayError, match="physical"):
        daemon._iter_merge_lifecycle_events(maximum_evidence_files=64,
            maximum_evidence_bytes=owner.VALIDATION_REARM_FLOOR_MAX_JOURNAL_BYTES)


def test_warm_verified_replay_cache_cannot_bypass_later_evidence_growth(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as owner
    daemon = _daemon(tmp_path)
    daemon.max_task_attempts = 4
    task = parse_task_file(daemon.todo_path, "## TEST-")[0]
    state, _ = _exhaust(daemon, task)
    cid, _ = _post_obsolescence_rearm(daemon, task, state)
    daemon._iter_merge_lifecycle_events(maximum_evidence_files=64,
        maximum_evidence_bytes=owner.VALIDATION_REARM_FLOOR_MAX_JOURNAL_BYTES)
    cached_key = daemon._merge_lifecycle_events_cache_key
    with daemon.events_path.open("ab") as handle:
        handle.truncate(owner.VALIDATION_REARM_FLOOR_MAX_JOURNAL_BYTES + 1)
    before = daemon.events_path.stat()
    selected, limited = daemon._partition_tasks_at_attempt_limit((task,), {task.task_id: "ready"}, state)
    assert selected == [] and "byte budget" in limited[0]["evidence_error"]
    after = daemon.events_path.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    assert daemon._merge_lifecycle_events_cache_key == cached_key
    assert state.implementation_attempts_by_cid[cid] == 1
    assert PortalTaskState.load(daemon.state_path).implementation_attempts_by_cid[cid] == 1


@pytest.mark.parametrize("invalid_count", [True, 4.5, "4", -1])
def test_authenticated_rearm_requires_exact_positive_count_fields(tmp_path, invalid_count):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as owner
    daemon = _daemon(tmp_path)
    daemon.max_task_attempts = 4
    task = parse_task_file(daemon.todo_path, "## TEST-")[0]
    state, _ = _exhaust(daemon, task)
    cid, _ = _post_obsolescence_rearm(daemon, task, state, previous=invalid_count)
    before = daemon.events_path.read_bytes()
    selected, limited = daemon._partition_tasks_at_attempt_limit((task,), {task.task_id: "ready"}, state)
    assert selected == [] and "witness" in limited[0]["evidence_error"]
    assert daemon.events_path.read_bytes() == before
    assert state.implementation_attempts_by_cid[cid] == 1


def test_unknown_legacy_budget_blocks_only_affected_identity(tmp_path, monkeypatch):
    daemon = _daemon(tmp_path)
    daemon.max_task_attempts = 4
    daemon.todo_path.write_text(daemon.todo_path.read_text() + """
## TEST-002 A new post-policy contract

- Status: todo
- Outputs: src/new.py
- Acceptance: New identity keeps its independent budget.
""")
    tasks = parse_task_file(daemon.todo_path, "## TEST-")
    daemon._register_task_identities(tasks)
    state, cid = _exhaust(daemon, tasks[0])
    _post_obsolescence_rearm(daemon, tasks[0], state)
    state.validation_obsolescence_rearm_receipts[cid] = ["sha256:" + "0" * 64]
    before = daemon.events_path.read_bytes()
    selected, limited = daemon._partition_tasks_at_attempt_limit(tasks, {task.task_id: "ready" for task in tasks}, state)
    assert selected == [tasks[1]]
    assert limited[0]["task_id"] == tasks[0].task_id
    assert limited[0]["reason"] == "validation_rearm_floor_evidence_unverified"
    assert state.implementation_attempts_by_cid[cid] == 1
    assert daemon.events_path.read_bytes() == before


@pytest.mark.parametrize("active_field,active_value", [("active_task_id", "closeout-task"),
    ("active_task_cid", "closeout-cid"), ("active_phase", "validation")])
def test_floor_acquisition_waits_for_active_identity_or_phase(tmp_path, monkeypatch, active_field, active_value):
    daemon = _daemon(tmp_path)
    daemon.max_task_attempts = 4
    task = parse_task_file(daemon.todo_path, "## TEST-")[0]
    state, _ = _exhaust(daemon, task)
    cid, _ = _post_obsolescence_rearm(daemon, task, state)
    setattr(state, active_field, active_value)
    state.implementation_in_progress = False
    monkeypatch.setattr(daemon, "_iter_merge_lifecycle_events",
                        lambda **kwargs: pytest.fail("closeout must not acquire history"))
    before = daemon.events_path.read_bytes()
    daemon._partition_tasks_at_attempt_limit((task,), {task.task_id: "ready"}, state)
    assert state.implementation_attempts_by_cid[cid] == 1
    assert daemon.events_path.read_bytes() == before


def test_unpersisted_same_cid_rearm_cannot_raise_authenticated_floor(tmp_path):
    daemon = _daemon(tmp_path)
    daemon.max_task_attempts = 4
    task = parse_task_file(daemon.todo_path, "## TEST-")[0]
    state, _ = _exhaust(daemon, task)
    cid, persisted = _post_obsolescence_rearm(daemon, task, state, previous=4)
    _post_obsolescence_rearm(daemon, task, state, previous=9, revision="unpersisted-revision")
    state.validation_obsolescence_rearm_receipts[cid] = [persisted]
    state.save(daemon.state_path)
    selected, limited = daemon._partition_tasks_at_attempt_limit((task,), {task.task_id: "ready"}, state)
    assert selected == [] and limited[0]["attempt_count"] == 4
    assert state.implementation_attempts_by_cid[cid] == 4


def test_rearm_witness_cannot_be_borrowed_by_different_canonical_revision(tmp_path):
    daemon = _daemon(tmp_path)
    daemon.max_task_attempts = 4
    task = parse_task_file(daemon.todo_path, "## TEST-")[0]
    state, old_cid = _exhaust(daemon, task)
    _, fingerprint = _post_obsolescence_rearm(daemon, task, state)
    daemon.todo_path.write_text(daemon.todo_path.read_text().replace(
        "Three consumed attempts stay spent across source repairs.", "A different canonical contract."))
    revised = parse_task_file(daemon.todo_path, "## TEST-")[0]
    daemon._register_task_identities((revised,))
    new_cid = daemon._canonical_ref(revised)
    assert new_cid != old_cid
    state.implementation_attempts_by_cid[new_cid] = 1
    state.validation_obsolescence_rearm_receipts[new_cid] = [fingerprint]
    state.save(daemon.state_path)
    before = daemon.events_path.read_bytes()
    selected, limited = daemon._partition_tasks_at_attempt_limit((revised,), {revised.task_id: "ready"}, state)
    assert selected == [] and limited[0]["reason"] == "validation_rearm_floor_evidence_unverified"
    assert state.implementation_attempts_by_cid[new_cid] == 1
    assert daemon.events_path.read_bytes() == before


@pytest.mark.parametrize("invalid", [True, "1", -1])
def test_malformed_completed_repair_counts_cannot_suppress_legacy_floor(tmp_path, invalid):
    daemon = _daemon(tmp_path, repair=True)
    tasks = parse_task_file(daemon.todo_path, "## TEST-")
    state, _ = _exhaust(daemon, tasks[0])
    cid, fingerprint = _post_obsolescence_rearm(daemon, tasks[0], state)
    daemon.todo_path.write_text(daemon.todo_path.read_text().replace(
        "- Status: todo\n- Completion: manual", "- Status: completed\n- Completion: manual"))
    tasks = parse_task_file(daemon.todo_path, "## TEST-")
    daemon._reset_attempt_budgets_for_completed_retry_repairs(state, tasks)
    state.validation_obsolescence_rearm_receipts[cid] = [fingerprint]
    state.implementation_attempts[tasks[0].task_id] = 1
    state.implementation_attempts_by_cid[cid] = 1
    state.save(daemon.state_path)
    daemon._record_event("task_retry_budget_reset", {"resets": [{
        "source_task_id": tasks[0].task_id, "repair_task_id": tasks[1].task_id,
        "failure_kind": "validation", "previous_display_attempt_count": invalid,
        "previous_canonical_attempt_counts": {cid: invalid}}]})
    before = daemon.events_path.read_bytes()
    selected, limited = daemon._partition_tasks_at_attempt_limit((tasks[0],), {tasks[0].task_id: "ready"}, state)
    assert selected == [] and "completed repair witness" in limited[0]["evidence_error"]
    assert state.implementation_attempts_by_cid[cid] == 1
    assert daemon.events_path.read_bytes() == before


@pytest.mark.parametrize("invalid", [None, {}, True, "fingerprint"])
def test_malformed_stored_rearm_receipts_cannot_fall_through_partition(tmp_path, invalid):
    daemon = _daemon(tmp_path)
    daemon.max_task_attempts = 4
    task = parse_task_file(daemon.todo_path, "## TEST-")[0]
    state, cid = _exhaust(daemon, task)
    state.implementation_attempts[task.task_id] = 1
    state.implementation_attempts_by_cid[cid] = 1
    state.validation_obsolescence_rearm_receipts[cid] = invalid
    selected, limited = daemon._partition_tasks_at_attempt_limit((task,), {task.task_id: "ready"}, state)
    assert selected == [] and limited[0]["reason"] == "validation_rearm_floor_evidence_unverified"
    assert state.implementation_attempts_by_cid[cid] == 1


def test_existing_manifest_counts_toward_total_caller_evidence_budget(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as owner
    daemon = _daemon(tmp_path)
    daemon.events_path.parent.mkdir(parents=True, exist_ok=True)
    daemon.events_path.write_bytes(b"log")
    manifest = daemon.events_path.with_name(daemon.events_path.name + ".manifest.json")
    manifest.write_bytes(b"manifest")
    monkeypatch.setattr(owner, "event_log_sources", lambda *a, **k: [daemon.events_path])
    monkeypatch.setattr(owner, "event_log_manifest", lambda *a: pytest.fail("total budget must refuse before manifest read"))
    with pytest.raises(owner.CursorReplayError, match="byte budget"):
        daemon._iter_merge_lifecycle_events(maximum_evidence_files=64, maximum_evidence_bytes=10)
    with pytest.raises(owner.CursorReplayError, match="file budget"):
        daemon._iter_merge_lifecycle_events(maximum_evidence_files=1, maximum_evidence_bytes=100)


def test_protected_same_attempt_recovery_is_exempt_before_floor_mutation(tmp_path, monkeypatch):
    daemon = _daemon(tmp_path)
    daemon.max_task_attempts = 4
    task = parse_task_file(daemon.todo_path, "## TEST-")[0]
    state, _ = _exhaust(daemon, task)
    cid, _ = _post_obsolescence_rearm(daemon, task, state)
    checked = []
    def verified_protected_owner(current, selected, **kwargs):
        assert current is state and selected is task
        rows = kwargs["verified_lifecycle_events"]
        assert rows and any(row.get("type") == "ordinary_task_validation_obsolescence_rearmed" for row in rows)
        checked.append(True)
        return 1  # Original owner confirms reuse of this consumed ordinal.
    monkeypatch.setattr(daemon, "_durable_protected_recovery_attempt", verified_protected_owner)
    monkeypatch.setattr(daemon, "_iter_events", lambda: pytest.fail("verified legacy lookup must not repair history"))
    before = daemon.events_path.read_bytes()
    selected, limited = daemon._partition_tasks_at_attempt_limit((task,), {task.task_id: "ready"}, state)
    assert selected == [task] and limited == [] and checked == [True]
    assert state.implementation_attempts_by_cid[cid] == 1
    assert daemon.events_path.read_bytes() == before


def test_completed_repair_invalidates_only_its_legacy_cache_before_archive(tmp_path, monkeypatch):
    daemon = _daemon(tmp_path, repair=True)
    tasks = parse_task_file(daemon.todo_path, "## TEST-")
    state, _ = _exhaust(daemon, tasks[0])
    cid, fingerprint = _post_obsolescence_rearm(daemon, tasks[0], state)
    state.validation_obsolescence_rearm_receipts["unrelated-cid"] = ["unrelated-fingerprint"]
    unchanged = deepcopy(state.validation_obsolescence_rearm_receipts)
    assert daemon._reset_attempt_budgets_for_completed_retry_repairs(state, tasks) == ([], [])
    assert state.validation_obsolescence_rearm_receipts == unchanged
    daemon.todo_path.write_text(daemon.todo_path.read_text().replace(
        "- Status: todo\n- Completion: manual", "- Status: completed\n- Completion: manual"))
    tasks = parse_task_file(daemon.todo_path, "## TEST-")
    resets, _ = daemon._reset_attempt_budgets_for_completed_retry_repairs(state, tasks)
    assert resets[0]["superseded_validation_obsolescence_rearm_receipts"] == {cid: [fingerprint]}
    assert cid not in state.validation_obsolescence_rearm_receipts
    assert state.validation_obsolescence_rearm_receipts["unrelated-cid"] == ["unrelated-fingerprint"]
    archive = daemon.events_path.with_name("pre-repair-history.retained.jsonl")
    daemon.events_path.rename(archive)
    state.implementation_attempts[tasks[0].task_id] = 1
    state.implementation_attempts_by_cid[cid] = 1
    state.save(daemon.state_path)
    monkeypatch.setattr(daemon, "_iter_merge_lifecycle_events",
                        lambda **kwargs: pytest.fail("completed repaired retry must not scan evicted legacy history"))
    selected, limited = daemon._partition_tasks_at_attempt_limit((tasks[0],), {tasks[0].task_id: "ready"}, state)
    assert selected == [tasks[0]] and limited == [] and archive.is_file()


@pytest.mark.parametrize("invalid_task_id", [[], {}])
def test_malformed_rearm_task_id_is_a_typed_partition_refusal(tmp_path, monkeypatch, invalid_task_id):
    daemon = _daemon(tmp_path)
    daemon.max_task_attempts = 4
    task = parse_task_file(daemon.todo_path, "## TEST-")[0]
    state, _ = _exhaust(daemon, task)
    original = daemon._record_event
    def record(kind, payload):
        if kind == "ordinary_task_validation_obsolescence_rearmed":
            payload = deepcopy(payload)
            payload["rearmed"][0]["task_id"] = invalid_task_id
        return original(kind, payload)
    monkeypatch.setattr(daemon, "_record_event", record)
    cid, _ = _post_obsolescence_rearm(daemon, task, state)
    before = daemon.events_path.read_bytes()
    selected, limited = daemon._partition_tasks_at_attempt_limit((task,), {task.task_id: "ready"}, state)
    assert selected == [] and limited[0]["reason"] == "validation_rearm_floor_evidence_unverified"
    assert state.implementation_attempts_by_cid[cid] == 1
    assert daemon.events_path.read_bytes() == before


def test_verified_rows_reuse_original_protected_latch_and_finish_predicates(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as owner
    daemon = _daemon(tmp_path)
    task = parse_task_file(daemon.todo_path, "## TEST-")[0]
    state, cid = _exhaust(daemon, task)
    state.implementation_attempts[task.task_id] = 1
    state.implementation_attempts_by_cid[cid] = 1
    namespace = task.board_namespace or daemon.todo_path.name
    latch = {"schema": "ipfs_accelerate_py.agent_supervisor.protected-implementation-attempt-latch@1",
        "task_id": task.task_id, "attempt": 1, "task_revision_cid": cid,
        "board_namespace": namespace, "route_id": "unit-route", "invocation_id": "unit-invocation",
        "logical_attempt_id": "unit-attempt", "worktree_id": "unit-worktree",
        "provider_attempt_store": "unit-store", "provider_attempt_store_identity": "unit-store-pin"}
    latch["latch_id"] = owner.content_identity(latch)
    key = daemon._protected_attempt_latch_key(task.task_id, 1, cid)
    state.protected_implementation_attempts[key] = dict(latch)
    rows = [{"type": "protected_implementation_attempt_latched", **latch}]
    monkeypatch.setattr(daemon, "_iter_events", lambda: pytest.fail("verified rows must not use repairing history"))
    assert daemon._durable_protected_recovery_attempt(state, task, verified_lifecycle_events=rows) == 1
    rows.append({"type": "implementation_finished", "task_id": task.task_id, "attempt": 1,
                 "canonical_task_cid": cid, "board_namespace": namespace})
    assert daemon._durable_protected_recovery_attempt(state, task, verified_lifecycle_events=rows) is None
