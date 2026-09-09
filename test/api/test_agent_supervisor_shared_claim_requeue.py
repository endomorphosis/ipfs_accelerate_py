"""A lane's empty sidecar must never reopen another lane's shared claim."""

from datetime import UTC

import pytest
from test.api.test_agent_supervisor_database_implementation_daemon import _open_daemon, _population


@pytest.fixture
def lanes(tmp_path, monkeypatch):
    clock = {"ms": 1000}
    owner = _open_daemon(
        tmp_path,
        session="owner",
        lane="owner",
        repo_root=tmp_path,
        clock_ms=lambda: clock["ms"],
        lease_ms=5000,
    )
    observer = _open_daemon(
        tmp_path,
        session="observer",
        lane="observer",
        repo_root=tmp_path,
        clock_ms=lambda: clock["ms"],
        lease_ms=5000,
    )
    owner.materialize_population(_population(1))
    attempt = owner.claim_next()
    assert attempt is not None
    for daemon in (owner, observer):
        monkeypatch.setattr(daemon, "_task_declared_output_paths", lambda task: ("missing.py",))
        monkeypatch.setattr(daemon, "_task_outputs_landed_on_target", lambda task: False)
    try:
        yield owner, observer, attempt, clock
    finally:
        observer.close()
        owner.close()


def test_foreign_live_claim_survives_empty_observer_execution_store(lanes):
    owner, observer, attempt, _clock = lanes
    before = owner.task_source.get(attempt.task_cid)
    assert before.status == "in_progress"
    assert observer.list_running_attempts() == []
    assert observer._requeue_unimplemented_control_task(before) is None
    after = owner.task_source.get(attempt.task_cid)
    assert (after.status, after.revision, after.body) == (
        before.status,
        before.revision,
        before.body,
    )
    assert owner.get_attempt(attempt.attempt_id).status == "running"


def test_local_live_attempt_is_not_an_orphan(lanes):
    owner, _observer, attempt, _clock = lanes
    task = owner.task_source.get(attempt.task_cid)
    assert owner._requeue_unimplemented_control_task(task) is None
    assert owner.task_source.get(attempt.task_cid).revision == task.revision


def test_local_failure_still_requires_expired_exact_coordination(lanes):
    owner, observer, attempt, clock = lanes
    attempt = owner.commit_phase(attempt, "context")
    attempt = owner.commit_phase(attempt, "failed", body={"reason": "test failure"})
    task = owner.task_source.get(attempt.task_cid)
    assert owner._requeue_unimplemented_control_task(task) is None
    assert owner.task_source.get(attempt.task_cid).revision == task.revision
    clock["ms"] = 7000
    owner._reconcile_failed_attempt_coordination(attempt)
    # Even an expired foreign claim cannot be inferred from this lane's empty store.
    assert observer._requeue_unimplemented_control_task(task) is None
    result = owner._requeue_unimplemented_control_task(task)
    assert result["requeued"] is True
    recovered = owner.task_source.get(attempt.task_cid)
    assert recovered.status == "todo"
    assert recovered.revision == task.revision + 1
    assert owner.get_attempt(attempt.attempt_id).status == "failed"


def test_orphan_scan_preserves_foreign_live_claim_without_lifecycle(lanes, monkeypatch):
    from datetime import datetime, timedelta
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as module

    owner, observer, attempt, _clock = lanes
    before = owner.task_source.get(attempt.task_cid)
    monkeypatch.setattr(
        module,
        "WorktreeLifecycleStore",
        lambda **kwargs: SimpleNamespace(find_nonterminal_for_task=lambda **kwargs: None),
    )
    monkeypatch.setattr(
        observer,
        "_parse_control_task_updated_at",
        lambda value: datetime.now(UTC) - timedelta(hours=1),
    )
    assert observer._unstall_orphan_in_progress_gates() == []
    after = owner.task_source.get(attempt.task_cid)
    assert (after.status, after.revision, after.body) == (
        before.status,
        before.revision,
        before.body,
    )
    assert owner.get_attempt(attempt.attempt_id).status == "running"


def test_orphan_scan_recovers_only_exact_expired_local_failure(lanes, monkeypatch):
    from datetime import datetime, timedelta
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as module

    owner, observer, attempt, clock = lanes
    monkeypatch.setattr(
        module,
        "WorktreeLifecycleStore",
        lambda **kwargs: SimpleNamespace(find_nonterminal_for_task=lambda **kwargs: None),
    )
    for daemon in (owner, observer):
        monkeypatch.setattr(
            daemon,
            "_parse_control_task_updated_at",
            lambda value: datetime.now(UTC) - timedelta(hours=1),
        )
    attempt = owner.commit_phase(attempt, "context")
    attempt = owner.commit_phase(attempt, "failed", body={"reason": "test failure"})
    assert owner._unstall_orphan_in_progress_gates() == []
    clock["ms"] = 7000
    owner._reconcile_failed_attempt_coordination(attempt)
    assert observer._unstall_orphan_in_progress_gates() == []
    assert owner._unstall_orphan_in_progress_gates()[0]["unstalled"] is True
    assert owner.task_source.get(attempt.task_cid).status == "retrying"
