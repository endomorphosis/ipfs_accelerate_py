"""A lost production callback result is not permission to retry its effects."""

from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinationError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalExecutionBridge,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.expired_attempt_custody import (
    ExpiredExecutionCustodyPending,
    expired_execution_deferral,
    guard_generic_retirement,
)
from test.api.test_agent_supervisor_database_implementation_daemon import (
    _open_daemon,
    _population,
)


def all_rows(connection):
    names = connection.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name"
    ).fetchall()
    return {
        row[0]: [
            tuple(item[index] for index in range(len(item)))
            for item in connection.execute(
                'SELECT * FROM "' + row[0] + '" ORDER BY ALL'
            ).fetchall()
        ]
        for row in names
    }


def retained_rows(daemon):
    return {
        "execution": all_rows(daemon._require_connection()),
        "coordination": all_rows(daemon.coordinator._require()),
        "task": daemon.task_source.get("task:cid:001").to_dict(),
    }


@pytest.mark.parametrize("phase", ["claimed", "context", "provider", "effect"])
@pytest.mark.parametrize("already_expired", [False, True])
def test_production_expiry_preserves_native_rows_at_every_unsettled_phase(
    tmp_path, phase, already_expired
):
    now = {"ms": 1000}
    daemon = _open_daemon(
        tmp_path,
        lease_ms=5000,
        clock_ms=lambda: now["ms"],
        provider_fn=lambda _: {"status": "succeeded", "accepted": True},
    )
    daemon.require_real_execution = True
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        if phase != "claimed":
            attempt = daemon.commit_phase(attempt, "context")
        provider = {}
        if phase in {"provider", "effect"}:
            attempt, provider, _ = daemon.run_provider(attempt)
        if phase == "effect":
            attempt, _, _ = daemon.run_effect(attempt, provider)
        now["ms"] = 7000
        if already_expired:
            daemon.coordinator.expire_task_claim(
                daemon.coordinator.get_task_claim(attempt.claim_id), now_ms=now["ms"]
            )
        before = retained_rows(daemon)
        for _ in range(2):
            result = daemon.run_once()
            assert (
                result["selection_idle_reason"]
                == "expired_attempt_settlement_unavailable"
            )
            assert result["provider_dispatched"] == "unknown"
            assert result["recovery_provider_dispatched"] is False
            assert result["retry_authorized"] is False
            assert result["coordination_mutation_authority"] is False
            assert (
                result["retained_attempt_evidence"]["attempt_id"] == attempt.attempt_id
            )
            assert retained_rows(daemon) == before
    finally:
        daemon.close()


def test_callback_effect_can_exist_without_result_and_is_not_repeated_after_expiry(
    tmp_path,
):
    now = {"ms": 1000}
    calls = []
    effect_path = tmp_path / "actual-callback-effect"

    def callback(attempt):
        calls.append(attempt.attempt_id)
        effect_path.write_text("effect already happened")
        now["ms"] = 7000
        return {"status": "succeeded", "accepted": True}

    daemon = _open_daemon(
        tmp_path, provider_fn=callback, lease_ms=5000, clock_ms=lambda: now["ms"]
    )
    daemon.require_real_execution = True
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.commit_phase(daemon.claim_next(), "context")
        with pytest.raises(DatabaseCoordinationError):
            daemon.run_provider(attempt)
        assert (
            daemon.provider_invocation_recorded(
                attempt.attempt_id, idempotency_key=f"provider:{attempt.attempt_id}"
            )
            is None
        )
        assert (
            daemon.effect_claim_recorded(
                attempt.attempt_id, idempotency_key=f"effect:{attempt.attempt_id}"
            )
            is None
        )
        before = retained_rows(daemon)
        for _ in range(2):
            assert daemon.run_once()["retry_authorized"] is False
            assert retained_rows(daemon) == before
        assert calls == [attempt.attempt_id]
        assert effect_path.read_text() == "effect already happened"
    finally:
        daemon.close()


def test_missing_claim_history_does_not_terminalize_a_production_attempt(
    tmp_path, monkeypatch
):
    daemon = _open_daemon(tmp_path)
    daemon.require_real_execution = True
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        before = retained_rows(daemon)
        monkeypatch.setattr(daemon.coordinator, "get_task_claim", lambda _: None)
        result = daemon.run_once()
        assert result["retained_attempt_evidence"]["reason"] == "claim_history_missing"
        assert daemon.get_attempt(attempt.attempt_id).status == "running"
        assert retained_rows(daemon) == before
    finally:
        daemon.close()


def test_native_portal_callback_binding_is_guarded_without_production_flag():
    bridge = object.__new__(DatabasePortalExecutionBridge)
    daemon = SimpleNamespace(
        require_real_execution=False, _provider_fn=bridge.run_provider
    )
    attempt = SimpleNamespace(
        task_cid="task:one",
        claim_id="claim:one",
        attempt_id="attempt:one",
        lease_id="lease:one",
        owner_session_id="session:one",
        attempt_number=1,
        fencing_token=1,
        fence_epoch=1,
    )
    with pytest.raises(ExpiredExecutionCustodyPending) as caught:
        guard_generic_retirement(daemon, attempt, reason="claim_authority_expired")
    assert expired_execution_deferral(caught.value)["retry_authorized"] is False


@pytest.mark.parametrize(
    "change",
    [
        {"task_cid": ""},
        {"claim_id": None},
        {"attempt_id": "x" * 513},
        {"fencing_token": True},
        {"attempt_number": 0},
        {"fence_epoch": 2**63},
        {"reason": "safe to retry"},
        {"extra": "private diagnostic"},
    ],
)
def test_malformed_custody_diagnostic_cannot_be_promoted_to_a_deferral(change):
    evidence = dict(
        task_cid="task:one",
        claim_id="claim:one",
        attempt_id="attempt:one",
        lease_id="lease:one",
        owner_session_id="session:one",
        attempt_number=1,
        fencing_token=1,
        fence_epoch=1,
        reason="claim_authority_expired",
    )
    evidence.update(change)
    assert expired_execution_deferral(ExpiredExecutionCustodyPending(evidence)) is None


def test_completed_reconciliation_prefix_remains_visible_when_expiry_defers(
    tmp_path, monkeypatch
):
    now = {"ms": 1000}
    daemon = _open_daemon(tmp_path, lease_ms=5000, clock_ms=lambda: now["ms"])
    daemon.require_real_execution = True
    try:
        daemon.materialize_population(_population(1))
        daemon.claim_next()
        now["ms"] = 7000
        earlier = [{"reason": "earlier_native_terminal_settlement", "write_count": 1}]
        monkeypatch.setattr(
            daemon, "reconcile_terminal_portal_failures", lambda: earlier
        )
        monkeypatch.setattr(
            daemon,
            "reconcile_recoverable_portal_failure_rearms",
            lambda: pytest.fail("later rearm ran past unresolved custody"),
        )
        monkeypatch.setattr(
            daemon, "claim_next", lambda: pytest.fail("new claim ran past custody")
        )
        result = daemon.run_once()
        assert result["unchanged"] is None
        assert result["recovery_prefix"]["portal_failure_reconciliations"] == earlier
        assert daemon._idle_recovery_prefix is None
    finally:
        daemon.close()
