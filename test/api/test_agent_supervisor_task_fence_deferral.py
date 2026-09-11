"""A retained fence mismatch ends the tick without resetting unknown effects."""

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinationStaleFenceError,
    DatabaseCoordinationTaskFenceMismatchError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.completion_deferral import (
    task_fence_mismatch_deferral,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationDaemon,
)
from test.api.test_agent_supervisor_database_coordination import _open


def evidence(**changes):
    return dict({
        "reason": "task_claim_latest_fence_mismatch",
        "task_cid": "task:a", "claim_id": "claim:a", "attempt_id": "attempt:a",
        "expected_fencing_token": 1, "expected_fence_epoch": 1,
        "observed_fencing_token": 2, "observed_fence_epoch": 2,
    }, **changes)


def make_error(**changes):
    return DatabaseCoordinationTaskFenceMismatchError(
        "private diagnostic", evidence=evidence(**changes),
    )


@pytest.mark.parametrize("token_delta,epoch_delta", [(1, 0), (0, 1), (1, 1), (-1, -1)])
def test_real_fence_guard_preserves_claim_and_lease_at_every_deferred_tick(
    tmp_path, token_delta, epoch_delta,
):
    coordinator, _ = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:fence", task_id="FENCE")
        claim = coordinator.claim_task(task_cid="task:fence", owner_session_id="session:one")
        connection = coordinator._require()
        # Model retained history drift without issuing a replacement claim.
        # The diagnostic must not claim that any observed token is authority.
        connection.execute(
            "UPDATE token_history SET fencing_token = fencing_token + ?, fence_epoch = fence_epoch + ?",
            [token_delta, epoch_delta],
        )
        before = (
            coordinator.get_task_claim(claim.claim_id),
            coordinator.get_task_attempt(claim.attempt_id),
            coordinator.get_lease(claim.lease_id),
        )
        daemon = DatabaseImplementationDaemon.__new__(DatabaseImplementationDaemon)
        daemon._embedded_writer_lock_handles = {}
        daemon._require_typed_quack_authority_binding = lambda: None
        calls = []

        def resume():
            daemon._idle_recovery_prefix = {"earlier_reconciliation": {"write_count": 1}}
            calls.append("guard")
            coordinator.protect_task_claim(claim)
            pytest.fail("the retained mismatched claim must not enter a callback")

        daemon._run_once_impl = resume
        for _ in range(2):
            result = daemon.run_once()
            assert result["selection_idle_reason"] == "task_fence_evidence_unavailable"
            assert result["task_fence_evidence"] == {
                "task_cid": claim.task_cid, "claim_id": claim.claim_id,
                "attempt_id": claim.attempt_id,
                "expected_fencing_token": claim.fencing_token,
                "expected_fence_epoch": claim.fence_epoch,
                "observed_fencing_token": claim.fencing_token + token_delta,
                "observed_fence_epoch": claim.fence_epoch + epoch_delta,
            }
            assert result["unchanged"] is None
            assert result["provider_dispatched"] == "unknown"
            assert result["recovery_provider_dispatched"] is False
            assert result["completion_authority"] is False
            assert result["coordination_mutation_authority"] is False
            assert result["recovery_prefix"]["earlier_reconciliation"]["write_count"] == 1
            assert daemon._idle_recovery_prefix is None
            assert (
                coordinator.get_task_claim(claim.claim_id),
                coordinator.get_task_attempt(claim.attempt_id),
                coordinator.get_lease(claim.lease_id),
            ) == before
            assert coordinator.get_prepared_task_completion(claim.task_cid) is None
        assert calls == ["guard", "guard"]
        # Deferral itself cannot bypass a future authority check.
        with pytest.raises(DatabaseCoordinationTaskFenceMismatchError):
            coordinator.protect_task_claim(claim)
    finally:
        coordinator.close()


@pytest.mark.parametrize("stage", [
    "reconcile_prepared_task_completions", "reconcile_expired_running_attempts",
    "reconcile_landed_merged_tasks", "reconcile_terminal_portal_failures",
])
def test_fence_mismatch_at_any_reconciliation_boundary_prevents_dispatch(stage):
    daemon = DatabaseImplementationDaemon.__new__(DatabaseImplementationDaemon)
    daemon._embedded_writer_lock_handles = {}
    daemon._require_typed_quack_authority_binding = lambda: None
    daemon.require_real_execution = True
    daemon._recover_lost_typed_claim_reservations = lambda: []
    daemon._quack_transport_preflight = lambda: None
    for prefix in (
        "_rearm_blocked_tasks_with_outputs_on_head",
        "_settle_invalid_metadata_portal_quarantines",
        "_consume_pending_same_board_merges", "_run_post_merge_recovery",
        "_consume_bound_pending_merge_train",
    ):
        setattr(daemon, prefix, lambda: {})
    daemon._reopen_unusable_embedded_sidecars = lambda error: pytest.fail(
        "a task fence diagnostic must not enter sidecar recovery")
    calls = []
    stages = [
        "reconcile_prepared_task_completions", "reconcile_expired_running_attempts",
        "reconcile_landed_merged_tasks", "reconcile_blocked_merge_queue_completions",
        "reconcile_unimplemented_unknown_callback_quarantines",
        "reconcile_consumed_no_progress_without_effect_quarantines",
        "reconcile_sandbox_host_failure_quarantines", "reconcile_terminal_portal_failures",
    ]
    for name in stages:
        def reconcile(name=name):
            calls.append(name)
            if name == stage:
                raise make_error()
            if stages.index(name) > stages.index(stage):
                pytest.fail("a later reconciliation ran past a fence blocker")
            return []
        setattr(daemon, name, reconcile)
    daemon.list_running_attempts = lambda: pytest.fail("retained work selected after fence blocker")
    daemon.claim_next = lambda: pytest.fail("new work selected after fence blocker")
    for _ in range(2):
        result = daemon.run_once()
        assert result["deferred"] is True
        assert result["recovery_attempt_consumed"] is False
    assert calls == stages[:stages.index(stage) + 1] * 2


@pytest.mark.parametrize("change", [
    {"reason": "other"}, {"task_cid": ""}, {"claim_id": None},
    {"attempt_id": "x" * 513}, {"expected_fencing_token": True},
    {"expected_fence_epoch": 0}, {"observed_fencing_token": -1},
    {"observed_fence_epoch": 2**63}, {"observed_fence_epoch": "2"},
    {"observed_fencing_token": 1, "observed_fence_epoch": 1},
])
def test_unbound_fence_mismatch_is_not_reclassified(change):
    assert task_fence_mismatch_deferral(make_error(**change)) is None


def test_plain_stale_fence_still_raises_and_diagnostic_extras_do_not_escape():
    error = DatabaseCoordinationStaleFenceError("caller fence mismatch")
    assert task_fence_mismatch_deferral(error) is None
    daemon = DatabaseImplementationDaemon.__new__(DatabaseImplementationDaemon)
    daemon._embedded_writer_lock_handles = {}
    daemon._require_typed_quack_authority_binding = lambda: None
    daemon._reopen_unusable_embedded_sidecars = lambda error: None
    daemon._is_quack_attach_contention = lambda error: False
    daemon._is_quack_transport_unavailable = lambda error: False
    def fail():
        raise error
    daemon._run_once_impl = fail
    with pytest.raises(DatabaseCoordinationStaleFenceError) as caught:
        daemon.run_once()
    assert caught.value is error
    assert daemon._idle_recovery_prefix is None
    result = task_fence_mismatch_deferral(make_error(token="private", retry_authorized=True))
    assert "private" not in repr(result)
    assert "retry_authorized" not in repr(result)
    assert result["coordination_mutation_authority"] is False
