"""Missing logical completion must preserve claims and retry ordinary settlement."""

import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinationNotReadyError,
    LeaseState,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.completion_deferral import (
    missing_completion_deferral,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationDaemon,
)
from test.api.test_agent_supervisor_database_coordination import (
    _completed_control_task,
    _open,
)


def test_missing_completion_keeps_claim_and_resumes_only_after_real_receipt(tmp_path):
    coordinator, _ = _open(tmp_path)
    daemon = DatabaseImplementationDaemon.__new__(DatabaseImplementationDaemon)
    daemon._lock = threading.RLock()
    daemon._embedded_writer_lock_handles = {}
    daemon._require_typed_quack_authority_binding = lambda: None
    coordinator.register_task(task_cid="task:missing", task_id="MISSING")
    claim = coordinator.claim_task(task_cid="task:missing", owner_session_id="session:one")
    before_claim = coordinator.get_task_claim(claim.claim_id)
    before_attempt = coordinator.get_task_attempt(claim.attempt_id)
    calls = []

    def settle():
        calls.append("settle")
        daemon._idle_recovery_prefix = {"post_merge_recovery": {"write_count": 1}}
        coordinator.settle_task_claim(claim)
        return {"settled": True}

    daemon._run_once_impl = settle
    try:
        for _ in range(2):
            result = daemon.run_once()
            assert result["selection_idle_reason"] == "completion_evidence_unavailable"
            assert result["completion_authority"] is False
            assert result["unchanged"] is None
            assert result["provider_dispatched"] == "unknown"
            assert result["recovery_provider_dispatched"] is False
            assert result["recovery_prefix"]["post_merge_recovery"]["write_count"] == 1
            assert result["completion_evidence"]["attempt_id"] == claim.attempt_id
            assert daemon._idle_recovery_prefix is None
            assert coordinator.get_task_claim(claim.claim_id) == before_claim
            assert coordinator.get_task_attempt(claim.attempt_id) == before_attempt
            assert coordinator.get_prepared_task_completion(claim.task_cid) is None
        assert calls == ["settle", "settle"]
        prepared = coordinator.prepare_task_completion(
            claim, control_expected_revision=2, evidence_digest="sha256:evidence"
        )
        coordinator.complete_task_claim(
            claim, control_completion_receipt=_completed_control_task(prepared)
        )
        assert daemon.run_once() == {"settled": True}
        assert calls == ["settle", "settle", "settle"]
        assert coordinator.get_task_claim(claim.claim_id).state is LeaseState.RELEASED
    finally:
        coordinator.close()


@pytest.mark.parametrize("change", [
    {"reason": "other"}, {"attempt_id": ""}, {"claim_id": None},
    {"task_cid": "x" * 513},
])
def test_unknown_or_unbound_missing_completion_remains_an_error(change):
    evidence = dict(reason="completion_missing", task_cid="task:a", claim_id="claim:a", attempt_id="attempt:a")
    evidence.update(change)
    assert missing_completion_deferral(DatabaseCoordinationNotReadyError("failure", evidence=evidence)) is None
    assert missing_completion_deferral(RuntimeError("completion_missing")) is None


def test_unknown_fields_are_not_promoted_into_deferral_authority():
    evidence = dict(reason="completion_missing", task_cid="task:a", claim_id="claim:a", attempt_id="attempt:a", token="private", completed=True)
    result = missing_completion_deferral(DatabaseCoordinationNotReadyError("private", evidence=evidence))
    assert result["completion_evidence"] == {key: evidence[key] for key in ("task_cid", "claim_id", "attempt_id")}
    assert "private" not in repr(result)
    assert result["completion_authority"] is False


def test_tick_does_not_swallow_other_coordination_failures():
    daemon = DatabaseImplementationDaemon.__new__(DatabaseImplementationDaemon)
    daemon._embedded_writer_lock_handles = {}
    daemon._require_typed_quack_authority_binding = lambda: None
    daemon._reopen_unusable_embedded_sidecars = lambda error: None
    daemon._is_quack_attach_contention = lambda error: False
    error = DatabaseCoordinationNotReadyError("other", evidence={"reason": "already_completed"})

    def fail():
        raise error

    daemon._run_once_impl = fail
    with pytest.raises(DatabaseCoordinationNotReadyError) as caught:
        daemon.run_once()
    assert caught.value is error
    assert daemon._idle_recovery_prefix is None
