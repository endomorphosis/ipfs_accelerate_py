"""A missing local attempt cannot authorize rearm of a shared live claim."""

from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationDaemon,
)
from test.api.test_agent_supervisor_database_implementation_daemon import _open_daemon, _population


def test_live_extra_gate_claim_survives_foreign_observer(tmp_path, monkeypatch):
    owner = _open_daemon(tmp_path, session="owner", lease_ms=5000, clock_ms=lambda: 1000)
    observer = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "observer-coordination.duckdb",
        execution_path=tmp_path / "observer-execution.duckdb",
        owner_session_id="observer",
        authority_mode="embedded",
        lease_ms=5000,
        clock_ms=lambda: 1000,
    )
    try:
        population = _population(1)
        owner.materialize_population(population)
        attempt = owner.claim_next()
        assert attempt is not None
        task = owner.task_source.get(attempt.task_cid)
        monkeypatch.setattr(owner, "_task_alias_is_extra_gate", lambda task: True)
        monkeypatch.setattr(observer, "_task_alias_is_extra_gate", lambda task: True)
        assert not observer._extra_gate_stale_in_progress_opens_generic_rearm(
            task, running_cids=set()
        )
        assert observer.reconcile_blocked_unknown_outcome_tasks() == []
        after = owner.task_source.get(attempt.task_cid)
        assert (after.status, after.revision, after.body) == (task.status, task.revision, task.body)
        assert owner.get_attempt(attempt.attempt_id).status == "running"
    finally:
        observer.close()
        owner.close()


def test_extra_gate_rearm_requires_exact_expired_local_failure(tmp_path, monkeypatch):
    clock = {"ms": 1000}
    owner = _open_daemon(tmp_path, session="owner", lease_ms=5000, clock_ms=lambda: clock["ms"])
    try:
        population = _population(1)
        owner.materialize_population(population)
        attempt = owner.claim_next()
        assert attempt is not None
        attempt = owner.commit_phase(attempt, "context")
        attempt = owner.commit_phase(attempt, "failed", body={"reason": "test failure"})
        task = owner.task_source.get(attempt.task_cid)
        monkeypatch.setattr(owner, "_task_alias_is_extra_gate", lambda task: True)
        assert not owner._extra_gate_stale_in_progress_opens_generic_rearm(task, running_cids=set())
        clock["ms"] = 7000
        owner.coordinator.expire_task_claim(
            owner.coordinator.get_task_claim(attempt.claim_id), now_ms=7000
        )
        assert owner._extra_gate_stale_in_progress_opens_generic_rearm(task, running_cids=set())
        receipt = dict(task.body["completion_receipt"])
        receipt["fence_epoch"] += 1
        mismatch = replace(task, body={**task.body, "completion_receipt": receipt})
        assert not owner._extra_gate_stale_in_progress_opens_generic_rearm(
            mismatch, running_cids=set()
        )
        assert not owner._extra_gate_stale_in_progress_opens_generic_rearm(
            task, running_cids={task.task_cid}
        )
        with monkeypatch.context() as patch:
            patch.setattr(
                owner.coordinator, "get_prepared_task_completion", lambda task_cid: object()
            )
            assert not owner._extra_gate_stale_in_progress_opens_generic_rearm(
                task, running_cids=set()
            )
        with monkeypatch.context() as patch:
            patch.setattr(
                owner.coordinator,
                "get_task_claim_successor_projection",
                lambda **kwargs: {"successor": "present"},
            )
            assert not owner._extra_gate_stale_in_progress_opens_generic_rearm(
                task, running_cids=set()
            )
        with monkeypatch.context() as patch:
            patch.setattr(owner.coordinator, "get_task_claim_successor_projection", None)
            assert not owner._extra_gate_stale_in_progress_opens_generic_rearm(
                task, running_cids=set()
            )
        owner.process_instance_id = "process:restarted-owner"
        result = owner.reconcile_blocked_unknown_outcome_tasks()
        assert len(result) == 1 and result[0]["rearmed"] is True
        recovered = owner.task_source.get(task.task_cid)
        assert recovered.status == "retrying"
        assert recovered.body["completion_receipt"]["previous_claim_id"] == attempt.claim_id
        assert recovered.body["completion_receipt"]["previous_attempt_id"] == attempt.attempt_id
    finally:
        owner.close()
