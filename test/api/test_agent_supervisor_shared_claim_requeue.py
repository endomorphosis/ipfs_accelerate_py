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


def test_legacy_global_orphan_scan_cannot_complete_foreign_work(lanes, monkeypatch):
    owner, observer, attempt, _clock = lanes
    before = owner.task_source.get(attempt.task_cid)
    calls = []
    monkeypatch.setattr(observer, "_task_outputs_landed_on_target", lambda task: True)
    monkeypatch.setattr(observer, "_complete_landed_quarantined_task", lambda task: calls.append(task))
    assert observer.reconcile_orphaned_in_progress_gates() == []
    assert calls == []
    assert owner.task_source.get(attempt.task_cid).revision == before.revision


@pytest.mark.parametrize("operation", ["database_claim", "database_attempt_admitted"])
def test_legacy_recovery_rejects_typed_claim_without_typed_transport(lanes, operation, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as module

    owner, _observer, attempt, clock = lanes
    attempt = owner.commit_phase(attempt, "context")
    attempt = owner.commit_phase(attempt, "failed", body={"reason": "test failure"})
    clock["ms"] = 7000
    owner._reconcile_failed_attempt_coordination(attempt)
    task = owner.task_source.get(attempt.task_cid)
    assert owner._stale_control_claim_is_recoverable(task)
    assert not callable(getattr(owner.task_source, "claim_process_attestation", None))
    receipt = dict(task.body["completion_receipt"])
    receipt.update(
        operation=operation,
        claim_phase_schema=(
            module.TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA
            if operation == "database_claim"
            else module.TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA
        ),
        claim_process_attestation={"present": True},
    )
    typed_task = replace(task, body={**task.body, "completion_receipt": receipt})
    assert not owner._stale_control_claim_is_recoverable(typed_task)
    with monkeypatch.context() as patch:
        patch.setattr(owner, "_task_source", SimpleNamespace(claim_process_attestation=lambda: None))
        assert owner._shared_claim_binding_for_this_owner(typed_task) is not None
    after = owner.task_source.get(attempt.task_cid)
    assert (after.status, after.revision, after.body) == (task.status, task.revision, task.body)


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
