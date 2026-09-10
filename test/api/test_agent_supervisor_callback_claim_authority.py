from pathlib import Path
import pytest
from test.api.test_agent_supervisor_database_implementation_daemon import (
    _open_daemon,
    _population,
)


@pytest.mark.parametrize("budget", [2, 0])
def test_prior_no_source_alias_cannot_claim_or_dispatch(tmp_path: Path, budget: int):
    calls = []
    daemon = _open_daemon(
        tmp_path / "lane", max_task_attempts=budget, provider_calls=calls
    )
    try:
        daemon.materialize_population(_population(2))
        task = daemon.task_source.get("task:cid:001")
        receipt = {
            "operation": "database_portal_callback_no_effect_recovery",
            "schema": "ipfs_accelerate_py/agent-supervisor/database-portal-unknown-callback-no-merge-recovery@1",
            "reason": "unknown_callback_no_merge_source_requeued",
            "attempt_id": "attempt:unknown",
            "attempt_number": 1,
            "claim_id": "claim:unknown",
            "lease_id": "lease:unknown",
            "owner_session_id": "session:unknown",
            "fencing_token": 1,
            "fence_epoch": 1,
            "queue_reason": "database_portal_unknown_callback_no_merge_recovery:unverified",
            "backoff_ms": 0,
            "retry_not_before_ms": 0,
            "provider_dispatched": True,
            "attempt_consumed": True,
        }
        row = daemon.task_source.compare_and_set_status(
            task.task_cid, task.revision, "retrying", receipt=receipt
        ).task
        assert not daemon._callback_no_effect_retry_claim_is_within_budget(row)
        assert daemon.claim_next(exclude_task_cids=("task:cid:002",)) is None
        current = daemon.task_source.get(task.task_cid)
        assert current.status == "retrying" and current.revision == row.revision
        assert current.body["completion_receipt"] == receipt
        assert calls == []
        assert daemon.list_running_attempts() == []
        other = daemon.claim_next()
        assert other is not None and other.task_cid == "task:cid:002"
    finally:
        daemon.close()


@pytest.mark.parametrize("budget", [2, 0])
def test_verified_callback_retry_requires_its_existing_bounded_successor_policy(
    tmp_path: Path, budget: int
):
    from test.api.test_agent_supervisor_database_implementation_daemon import (
        _exact_callback_no_effect_receipt,
    )

    class ProviderCrash(BaseException):
        pass

    now = {"ms": 1000}
    calls = []

    def provider(attempt):
        calls.append(attempt.attempt_id)
        raise ProviderCrash("callback outcome unavailable")

    settings = {
        "max_task_attempts": budget,
        "lease_ms": 5000,
        "clock_ms": lambda: now["ms"],
        "session": "session:closed-callback",
        "strict_task_sharding": True,
        "provider_fn": provider,
    }
    first = _open_daemon(tmp_path / "lane", **settings)
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        with pytest.raises(ProviderCrash):
            first._resume_attempt_without_process_crash(attempt)
    finally:
        first.close()

    now["ms"] = 7000
    restarted = _open_daemon(
        tmp_path / "lane",
        repo_root=tmp_path,
        post_commit_candidate_recovery_fn=lambda source: (
            _exact_callback_no_effect_receipt(
                source, workspace=tmp_path / "retained", max_task_attempts=budget
            )
        ),
        **settings,
    )
    try:
        assert any(
            item.get("disposition") == "quarantined"
            for item in restarted.reconcile_expired_running_attempts()
        )
        [recovery] = restarted.reconcile_unimplemented_unknown_callback_quarantines()
        assert recovery["reason"] == "exact_no_effect_callback_rearmed"
        successor = restarted.claim_next()
        if budget > 0:
            assert successor is not None
            assert successor.task_cid == attempt.task_cid
            assert successor.attempt_number == attempt.attempt_number + 1
        else:
            # Existing retry-consumption authority requires a bounded successor.
            # Reject before claiming instead of entering an unsupported fence.
            assert successor is None
            assert restarted.list_running_attempts() == []
        assert calls == [attempt.attempt_id]
    finally:
        restarted.close()
