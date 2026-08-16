"""Tests for DatabaseCoordinator (DQP-015).

Evidence subset: acquire, renew, release, expiry, takeover, fairness,
dependency readiness, epoch monotonicity, stale fence, response loss.

Acceptance: Four processes never own the same exclusive scope; expired session
cannot renew or mutate; append/fair scheduling remains concurrent; stale
fencing epoch is rejected in every protected write; claim and task-attempt
creation are one transaction.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DATABASE_COORDINATOR_INTERFACE,
    FENCED_LEASE_INTERFACE,
    MAINTENANCE_LEASE_INTERFACE,
    RESOURCE_CLAIM_INTERFACE,
    TASK_CLAIM_INTERFACE,
    AttemptStatus,
    DatabaseCoordinationConflictError,
    DatabaseCoordinationExpiredError,
    DatabaseCoordinationNotReadyError,
    DatabaseCoordinationStaleFenceError,
    DatabaseCoordinator,
    LeaseKind,
    LeaseMode,
    LeaseState,
    duckdb_available,
    exclusive_scope_key,
    open_database_coordinator,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for DatabaseCoordinator hermetic tests",
)


class FakeClock:
    def __init__(self, start_ms: int = 1_000_000) -> None:
        self.now = int(start_ms)

    def __call__(self) -> int:
        return int(self.now)

    def advance(self, ms: int) -> None:
        self.now += int(ms)


def _open(
    tmp_path: Path,
    *,
    clock: FakeClock | None = None,
    default_lease_ms: int = 60_000,
) -> tuple[DatabaseCoordinator, FakeClock]:
    clock = clock or FakeClock()
    coordinator = open_database_coordinator(
        tmp_path / "coordination.duckdb",
        clock_ms=clock,
        default_lease_ms=default_lease_ms,
    )
    return coordinator, clock


def _completed_control_task(
    prepared: dict[str, object],
    *,
    nested_cas_result: bool = False,
) -> dict[str, object]:
    task = {
        "task_cid": prepared["task_cid"],
        "status": "completed",
        "revision": int(prepared["control_expected_revision"]) + 1,
        "body": {
            "completion_receipt": {
                "operation": "database_complete",
                "coordination_preparation": dict(prepared),
            }
        },
    }
    if not nested_cas_result:
        return task
    return {
        "task": task,
        "previous_status": prepared["control_expected_status"],
        "revision": task["revision"],
        "event_cursor": 7,
        "changed": True,
        "receipt_cid": "cid:control-completion",
    }


def _incomplete_control_task(prepared: dict[str, object]) -> dict[str, object]:
    return {
        "task_cid": prepared["task_cid"],
        "status": prepared["control_expected_status"],
        "revision": prepared["control_expected_revision"],
        "body": {},
    }


# ---------------------------------------------------------------------------
# Interface identities
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert DATABASE_COORDINATOR_INTERFACE == "DatabaseCoordinator@1"
    assert FENCED_LEASE_INTERFACE == "FencedLease@1"
    assert TASK_CLAIM_INTERFACE == "TaskClaim@1"
    assert RESOURCE_CLAIM_INTERFACE == "ResourceClaim@1"
    assert MAINTENANCE_LEASE_INTERFACE == "MaintenanceLease@1"
    assert DatabaseCoordinator.INTERFACE == DATABASE_COORDINATOR_INTERFACE


def test_exclusive_scope_key_is_stable() -> None:
    first = exclusive_scope_key(lease_kind=LeaseKind.TASK, scope="task:a")
    second = exclusive_scope_key(
        lease_kind="task", scope="task:a", task_cid="task:a"
    )
    assert first == second
    assert exclusive_scope_key(
        lease_kind=LeaseKind.PATH,
        scope="src/main.py",
        repository_id="repository:demo",
        path="src/main.py",
    ).startswith("path:repository:demo:")


# ---------------------------------------------------------------------------
# Acquire / renew / release
# ---------------------------------------------------------------------------


def test_acquire_renew_release_round_trip(tmp_path: Path) -> None:
    coordinator, clock = _open(tmp_path)
    try:
        lease = coordinator.acquire(
            lease_kind=LeaseKind.MERGE,
            scope="merge:main",
            owner_session_id="session:a",
            lease_ms=30_000,
        )
        assert lease.state is LeaseState.ACCEPTED
        assert lease.fencing_token >= 1
        assert lease.fence_epoch >= 1
        assert lease.owner_session_id == "session:a"

        clock.advance(5_000)
        renewed = coordinator.renew(lease, lease_ms=30_000)
        assert renewed.lease_id == lease.lease_id
        assert renewed.expires_at_ms > lease.expires_at_ms
        assert renewed.fencing_token == lease.fencing_token
        assert renewed.fence_epoch == lease.fence_epoch

        released = coordinator.release(renewed, reason="done")
        assert released.state is LeaseState.RELEASED
        events = coordinator.lease_events(lease_id=lease.lease_id)
        types = {item["event_type"] for item in events}
        assert "acquired" in types
        assert "renewed" in types
        assert "released" in types
    finally:
        coordinator.close()


def test_four_processes_never_own_same_exclusive_scope(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        first = coordinator.acquire(
            lease_kind=LeaseKind.RESOURCE,
            scope="gpu:0",
            owner_session_id="session:1",
            resource_kind="gpu",
            resource_id="gpu:0",
        )
        assert first.active
        for session in ("session:2", "session:3", "session:4"):
            with pytest.raises(DatabaseCoordinationConflictError, match="owned by"):
                coordinator.acquire(
                    lease_kind=LeaseKind.RESOURCE,
                    scope="gpu:0",
                    owner_session_id=session,
                    resource_kind="gpu",
                    resource_id="gpu:0",
                )
        active = coordinator.list_active_leases(lease_kind=LeaseKind.RESOURCE)
        owners = {item.owner_session_id for item in active}
        assert owners == {"session:1"}
    finally:
        coordinator.close()


def test_shared_mode_allows_concurrent_append_owners(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        first = coordinator.acquire(
            lease_kind=LeaseKind.PROVIDER_CAPACITY,
            scope="provider:shared",
            owner_session_id="session:a",
            mode=LeaseMode.SHARED,
            resource_id="provider:shared",
        )
        second = coordinator.acquire(
            lease_kind=LeaseKind.PROVIDER_CAPACITY,
            scope="provider:shared",
            owner_session_id="session:b",
            mode=LeaseMode.SHARED,
            resource_id="provider:shared",
        )
        assert first.scope_key == second.scope_key
        assert first.owner_session_id != second.owner_session_id
        active = coordinator.list_active_leases(
            lease_kind=LeaseKind.PROVIDER_CAPACITY
        )
        assert len(active) == 2
    finally:
        coordinator.close()


# ---------------------------------------------------------------------------
# Expiry / takeover / epoch monotonicity / stale fence
# ---------------------------------------------------------------------------


def test_expired_session_cannot_renew_or_mutate(tmp_path: Path) -> None:
    coordinator, clock = _open(tmp_path, default_lease_ms=10_000)
    try:
        lease = coordinator.acquire(
            lease_kind=LeaseKind.MAINTENANCE,
            scope="control-plane",
            owner_session_id="session:old",
            lease_ms=10_000,
        )
        clock.advance(10_001)
        with pytest.raises(DatabaseCoordinationExpiredError):
            coordinator.renew(lease, lease_ms=10_000)
        with pytest.raises(DatabaseCoordinationExpiredError):
            coordinator.protect_write(lease)
    finally:
        coordinator.close()


def test_takeover_after_expiry_advances_epoch_monotonically(tmp_path: Path) -> None:
    coordinator, clock = _open(tmp_path, default_lease_ms=10_000)
    try:
        original = coordinator.acquire(
            lease_kind=LeaseKind.PATH,
            scope="src/a.py",
            owner_session_id="session:a",
            resource_kind="path",
            resource_id="src/a.py",
            repository_id="repository:demo",
            path="src/a.py",
            lease_ms=10_000,
        )
        clock.advance(10_001)
        takeover = coordinator.takeover(
            lease_kind=LeaseKind.PATH,
            scope="src/a.py",
            owner_session_id="session:b",
            resource_kind="path",
            resource_id="src/a.py",
            repository_id="repository:demo",
            path="src/a.py",
            lease_ms=10_000,
        )
        assert takeover.owner_session_id == "session:b"
        assert takeover.fencing_token > original.fencing_token
        assert takeover.fence_epoch > original.fence_epoch
        with pytest.raises(DatabaseCoordinationConflictError):
            coordinator.takeover(
                lease_kind=LeaseKind.PATH,
                scope="src/a.py",
                owner_session_id="session:c",
                resource_kind="path",
                resource_id="src/a.py",
                repository_id="repository:demo",
                path="src/a.py",
            )
    finally:
        coordinator.close()


def test_stale_fencing_epoch_rejected_on_protected_writes(tmp_path: Path) -> None:
    coordinator, clock = _open(tmp_path, default_lease_ms=10_000)
    try:
        first = coordinator.acquire(
            lease_kind=LeaseKind.MERGE,
            scope="merge:train",
            owner_session_id="session:a",
            lease_ms=10_000,
        )
        clock.advance(10_001)
        second = coordinator.takeover(
            lease_kind=LeaseKind.MERGE,
            scope="merge:train",
            owner_session_id="session:b",
            lease_ms=10_000,
        )
        assert second.fence_epoch > first.fence_epoch
        with pytest.raises(DatabaseCoordinationStaleFenceError, match="stale"):
            coordinator.protect_write(
                second,
                expected_fencing_token=first.fencing_token,
                expected_fence_epoch=first.fence_epoch,
            )
        with pytest.raises(DatabaseCoordinationStaleFenceError):
            coordinator.renew(
                second,
                expected_fencing_token=first.fencing_token,
                expected_fence_epoch=first.fence_epoch,
            )
        with pytest.raises(DatabaseCoordinationStaleFenceError):
            coordinator.release(
                second,
                expected_fencing_token=first.fencing_token,
                expected_fence_epoch=first.fence_epoch,
            )
        # Current fence is accepted.
        assert coordinator.protect_write(second).lease_id == second.lease_id
    finally:
        coordinator.close()


# ---------------------------------------------------------------------------
# Task claims, attempts, fairness, dependency readiness, response loss
# ---------------------------------------------------------------------------


def test_claim_and_task_attempt_are_one_transaction(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:alpha", task_id="ALPHA")
        claim = coordinator.claim_task(
            task_cid="task:alpha",
            owner_session_id="session:worker",
            worktree_id="worktree:1",
        )
        assert claim.state is LeaseState.ACCEPTED
        assert claim.attempt_id
        assert claim.attempt_number == 1
        attempt = coordinator.get_task_attempt(claim.attempt_id)
        assert attempt is not None
        assert attempt.task_cid == "task:alpha"
        assert attempt.fencing_token == claim.fencing_token
        assert attempt.fence_epoch == claim.fence_epoch
        assert attempt.owner_session_id == "session:worker"
        lease = coordinator.get_lease(claim.lease_id)
        assert lease is not None
        assert lease.claim_id == claim.claim_id
        assert lease.attempt_id == claim.attempt_id
    finally:
        coordinator.close()


def test_dependency_readiness_blocks_claims(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:dep", task_id="DEP")
        coordinator.register_task(
            task_cid="task:child",
            task_id="CHILD",
            dependency_task_cids=["task:dep"],
        )
        readiness = coordinator.claimability("task:child")
        assert readiness["claimable"] is False
        assert readiness["blocked_dependency_task_cids"] == ["task:dep"]
        with pytest.raises(DatabaseCoordinationNotReadyError) as excinfo:
            coordinator.claim_task(
                task_cid="task:child",
                owner_session_id="session:worker",
            )
        assert excinfo.value.evidence["blocked_dependency_task_cids"] == ["task:dep"]

        coordinator.mark_task_complete("task:dep", status="succeeded")
        ready = coordinator.claimability("task:child")
        assert ready["claimable"] is True
        claim = coordinator.claim_task(
            task_cid="task:child",
            owner_session_id="session:worker",
        )
        assert claim.task_cid == "task:child"
    finally:
        coordinator.close()


def test_logically_completed_task_cannot_be_claimed_again(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:complete", task_id="COMPLETE")
        coordinator.mark_task_complete("task:complete", status="succeeded")

        readiness = coordinator.claimability("task:complete")
        assert readiness["claimable"] is False
        assert readiness["completion_status"] == "succeeded"
        assert readiness["repair_evidence"][0]["kind"] == "already_completed"
        assert coordinator.claim_ready_task(owner_session_id="session:next") is None
        with pytest.raises(DatabaseCoordinationNotReadyError) as excinfo:
            coordinator.claim_task(
                task_cid="task:complete",
                owner_session_id="session:direct",
            )
        assert excinfo.value.evidence["reason"] == "already_completed"
    finally:
        coordinator.close()


def test_fair_claim_ready_selects_oldest_registered_task(tmp_path: Path) -> None:
    coordinator, clock = _open(tmp_path)
    try:
        coordinator.register_task(
            task_cid="task:second", task_id="SECOND", now_ms=clock.now + 100
        )
        coordinator.register_task(
            task_cid="task:first", task_id="FIRST", now_ms=clock.now
        )
        coordinator.register_task(
            task_cid="task:third", task_id="THIRD", now_ms=clock.now + 200
        )
        first = coordinator.claim_ready_task(owner_session_id="session:a")
        assert first is not None
        assert first.task_cid == "task:first"
        second = coordinator.claim_ready_task(owner_session_id="session:b")
        assert second is not None
        assert second.task_cid == "task:second"
        third = coordinator.claim_ready_task(owner_session_id="session:c")
        assert third is not None
        assert third.task_cid == "task:third"
        assert coordinator.claim_ready_task(owner_session_id="session:d") is None
    finally:
        coordinator.close()


def test_response_loss_idempotency_replays_same_claim(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:idem", task_id="IDEM")
        first = coordinator.claim_task(
            task_cid="task:idem",
            owner_session_id="session:worker",
            idempotency_key="idem-1",
        )
        # Simulated response loss: client retries with the same key.
        second = coordinator.claim_task(
            task_cid="task:idem",
            owner_session_id="session:worker",
            idempotency_key="idem-1",
        )
        assert second.claim_id == first.claim_id
        assert second.attempt_id == first.attempt_id
        assert second.fencing_token == first.fencing_token
        assert second.fence_epoch == first.fence_epoch

        lease = coordinator.acquire(
            lease_kind=LeaseKind.RESOURCE,
            scope="disk:cache",
            owner_session_id="session:worker",
            resource_kind="disk",
            resource_id="disk:cache",
            idempotency_key="lease-idem-1",
        )
        replay = coordinator.acquire(
            lease_kind=LeaseKind.RESOURCE,
            scope="disk:cache",
            owner_session_id="session:worker",
            resource_kind="disk",
            resource_id="disk:cache",
            idempotency_key="lease-idem-1",
        )
        assert replay.lease_id == lease.lease_id
        assert replay.fencing_token == lease.fencing_token
    finally:
        coordinator.close()


def test_completed_task_guard_precedes_same_key_idempotency_replay(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:completed-idem", task_id="IDEM")
        claim = coordinator.claim_task(
            task_cid="task:completed-idem",
            owner_session_id="session:worker",
            idempotency_key="idem-completed",
        )
        coordinator.mark_task_complete(
            claim.task_cid,
            status="succeeded",
            body={"attempt_id": claim.attempt_id},
        )

        with pytest.raises(DatabaseCoordinationNotReadyError) as excinfo:
            coordinator.claim_task(
                task_cid=claim.task_cid,
                owner_session_id=claim.owner_session_id,
                idempotency_key=claim.idempotency_key,
            )
        assert excinfo.value.evidence["reason"] == "already_completed"
    finally:
        coordinator.close()


def test_task_idempotency_key_is_scoped_to_the_requested_task(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:idem-a", task_id="IDEM-A")
        coordinator.register_task(task_cid="task:idem-b", task_id="IDEM-B")
        first = coordinator.claim_task(
            task_cid="task:idem-a",
            owner_session_id="session:worker",
            idempotency_key="same-key",
        )
        second = coordinator.claim_task(
            task_cid="task:idem-b",
            owner_session_id="session:worker",
            idempotency_key="same-key",
        )

        assert first.task_cid == "task:idem-a"
        assert second.task_cid == "task:idem-b"
        assert second.claim_id != first.claim_id
        assert second.attempt_id != first.attempt_id
    finally:
        coordinator.close()


def test_same_owner_task_reacquire_without_key_replays_exact_live_claim(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:implicit-replay", task_id="REPLAY")
        first = coordinator.claim_task(
            task_cid="task:implicit-replay",
            owner_session_id="session:worker",
        )
        replay = coordinator.claim_task(
            task_cid=first.task_cid,
            owner_session_id=first.owner_session_id,
        )
        assert replay.claim_id == first.claim_id
        assert replay.attempt_id == first.attempt_id
        assert replay.lease_id == first.lease_id
        assert replay.fencing_token == first.fencing_token
        assert replay.fence_epoch == first.fence_epoch
    finally:
        coordinator.close()


def test_expired_same_key_retry_creates_new_claim_and_never_replays_old_attempt(
    tmp_path: Path,
) -> None:
    coordinator, clock = _open(tmp_path, default_lease_ms=10_000)
    try:
        coordinator.register_task(task_cid="task:expired-idem", task_id="EXPIRED")
        original = coordinator.claim_task(
            task_cid="task:expired-idem",
            owner_session_id="session:old",
            idempotency_key="same-response",
        )
        clock.advance(10_000)

        replacement = coordinator.claim_task(
            task_cid=original.task_cid,
            owner_session_id=original.owner_session_id,
            idempotency_key=original.idempotency_key,
        )
        assert replacement.claim_id != original.claim_id
        assert replacement.attempt_id != original.attempt_id
        assert replacement.attempt_number == original.attempt_number + 1
        assert replacement.fencing_token > original.fencing_token
        assert replacement.fence_epoch > original.fence_epoch
        with pytest.raises(DatabaseCoordinationExpiredError):
            coordinator.protect_task_claim(original)
        assert coordinator.protect_task_claim(replacement).lease_id == replacement.lease_id
        replay = coordinator.claim_task(
            task_cid=replacement.task_cid,
            owner_session_id=replacement.owner_session_id,
            idempotency_key=replacement.idempotency_key,
        )
        assert replay.claim_id == replacement.claim_id
        assert replay.attempt_id == replacement.attempt_id
    finally:
        coordinator.close()


def test_exact_task_claim_expiry_persists_without_prior_scope_sweep(
    tmp_path: Path,
) -> None:
    coordinator, clock = _open(tmp_path, default_lease_ms=10_000)
    try:
        coordinator.register_task(task_cid="task:explicit-expiry", task_id="EXPIRY")
        claim = coordinator.claim_task(
            task_cid="task:explicit-expiry",
            owner_session_id="session:old",
            idempotency_key="old-attempt",
        )

        with pytest.raises(DatabaseCoordinationExpiredError):
            coordinator.expire_task_claim(claim)
        assert coordinator.get_task_claim(claim.claim_id).state is LeaseState.ACCEPTED

        clock.advance(10_000)
        # No other coordinator mutation has swept this task scope.
        assert coordinator.get_task_claim(claim.claim_id).state is LeaseState.ACCEPTED
        expired = coordinator.expire_task_claim(claim)
        assert expired.state is LeaseState.EXPIRED
        assert coordinator.get_task_claim(claim.claim_id).state is LeaseState.EXPIRED
        assert coordinator.get_lease(claim.lease_id).state is LeaseState.EXPIRED
        assert (
            coordinator.get_task_attempt(claim.attempt_id).status
            is AttemptStatus.EXPIRED
        )
        assert coordinator.expire_task_claim(claim).state is LeaseState.EXPIRED

        replacement = coordinator.claim_task(
            task_cid=claim.task_cid,
            owner_session_id="session:new",
            idempotency_key="new-attempt",
        )
        assert replacement.attempt_number == claim.attempt_number + 1
        assert replacement.fencing_token > claim.fencing_token
        with pytest.raises(DatabaseCoordinationStaleFenceError):
            coordinator.expire_task_claim(claim)
        assert coordinator.get_task_claim(replacement.claim_id).state is LeaseState.ACCEPTED
    finally:
        coordinator.close()


def test_released_same_key_retry_creates_new_claim(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:released-idem", task_id="RELEASED")
        claim = coordinator.claim_task(
            task_cid="task:released-idem",
            owner_session_id="session:worker",
            idempotency_key="released-response",
        )
        coordinator.release(claim.as_fenced_lease(), reason="abandoned")

        replacement = coordinator.claim_task(
            task_cid=claim.task_cid,
            owner_session_id=claim.owner_session_id,
            idempotency_key=claim.idempotency_key,
        )
        assert replacement.claim_id != claim.claim_id
        assert replacement.attempt_id != claim.attempt_id
        assert replacement.attempt_number == claim.attempt_number + 1
        assert replacement.fencing_token > claim.fencing_token
    finally:
        coordinator.close()


def test_exact_task_claim_protection_rejects_identity_mismatch(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:protected", task_id="PROTECTED")
        claim = coordinator.claim_task(
            task_cid="task:protected",
            owner_session_id="session:worker",
        )

        protected = coordinator.protect_task_claim(
            claim,
            expected_task_cid=claim.task_cid,
            expected_attempt_id=claim.attempt_id,
            expected_owner_session_id=claim.owner_session_id,
            expected_fencing_token=claim.fencing_token,
            expected_fence_epoch=claim.fence_epoch,
        )
        assert protected.lease_id == claim.lease_id

        mismatched = claim.to_dict()
        mismatched["attempt_id"] = "attempt:not-authoritative"
        with pytest.raises(DatabaseCoordinationStaleFenceError):
            coordinator.protect_task_claim(mismatched)
        mismatched = claim.to_dict()
        mismatched["attempt_number"] += 1
        with pytest.raises(DatabaseCoordinationStaleFenceError):
            coordinator.protect_task_claim(mismatched)
        with pytest.raises(DatabaseCoordinationStaleFenceError):
            coordinator.protect_task_claim(
                claim,
                expected_owner_session_id="session:not-authoritative",
            )
    finally:
        coordinator.close()


def test_claim_aware_completion_and_successful_settlement_are_ordered(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:settle", task_id="SETTLE")
        claim = coordinator.claim_task(
            task_cid="task:settle",
            owner_session_id="session:worker",
        )

        prepared = coordinator.prepare_task_completion(
            claim,
            control_expected_revision=2,
            control_expected_status="in_progress",
            evidence_digest="sha256:test",
        )
        assert prepared["status"] == "prepared"
        assert prepared["replayed"] is False
        assert coordinator.claimability(claim.task_cid)["claimable"] is False
        assert coordinator.get_prepared_task_completion(claim.task_cid) is not None
        assert [
            item["task_cid"]
            for item in coordinator.list_prepared_task_completions(limit=10)
        ] == [claim.task_cid]

        control_cas = _completed_control_task(
            prepared,
            nested_cas_result=True,
        )
        completion = coordinator.complete_task_claim(
            claim,
            control_completion_receipt=control_cas,
        )
        assert completion["replayed"] is False
        assert coordinator.claimability(claim.task_cid)["claimable"] is False
        promoted_preparation = coordinator.get_prepared_task_completion(
            claim.task_cid
        )
        assert promoted_preparation is not None
        assert promoted_preparation["status"] == AttemptStatus.SUCCEEDED.value
        assert coordinator.list_prepared_task_completions(limit=10) == []
        assert coordinator.get_task_claim(claim.claim_id).state is LeaseState.ACCEPTED
        assert coordinator.get_lease(claim.lease_id).state is LeaseState.ACCEPTED
        assert (
            coordinator.get_task_attempt(claim.attempt_id).status
            is AttemptStatus.RUNNING
        )
        coordinator.protect_task_claim(claim, allow_logically_completed=True)

        replay = coordinator.complete_task_claim(
            claim,
            control_completion_receipt=control_cas["task"],
        )
        assert replay["replayed"] is True
        settled = coordinator.settle_task_claim(claim)
        assert settled.state is LeaseState.RELEASED
        assert coordinator.get_task_claim(claim.claim_id).state is LeaseState.RELEASED
        assert (
            coordinator.get_task_attempt(claim.attempt_id).status
            is AttemptStatus.SUCCEEDED
        )
        settlement_replay = coordinator.settle_task_claim(claim)
        assert settlement_replay.lease_id == settled.lease_id
        assert settlement_replay.state is LeaseState.RELEASED
        with pytest.raises(DatabaseCoordinationExpiredError):
            coordinator.protect_task_claim(
                claim,
                allow_logically_completed=True,
            )
    finally:
        coordinator.close()


def test_expired_task_claim_cannot_complete_or_settle(tmp_path: Path) -> None:
    coordinator, clock = _open(tmp_path, default_lease_ms=10_000)
    try:
        coordinator.register_task(task_cid="task:late", task_id="LATE")
        claim = coordinator.claim_task(
            task_cid="task:late",
            owner_session_id="session:late",
        )
        prepared = coordinator.prepare_task_completion(
            claim,
            control_expected_revision=2,
            evidence_digest="sha256:late",
        )
        control_task = _completed_control_task(prepared)
        clock.advance(10_000)

        with pytest.raises(DatabaseCoordinationExpiredError):
            coordinator.complete_task_claim(
                claim,
                control_completion_receipt=control_task,
            )
        assert coordinator.claimability(claim.task_cid)["claimable"] is False
        with pytest.raises(DatabaseCoordinationExpiredError):
            coordinator.settle_task_claim(claim)
    finally:
        coordinator.close()


def test_prepared_completion_does_not_satisfy_dependents_and_rejects_forgery(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:prepared", task_id="PREPARED")
        coordinator.register_task(
            task_cid="task:dependent",
            task_id="DEPENDENT",
            dependency_task_cids=("task:prepared",),
        )
        claim = coordinator.claim_task(
            task_cid="task:prepared",
            owner_session_id="session:worker",
        )
        prepared = coordinator.prepare_task_completion(
            claim,
            control_expected_revision=2,
            evidence_digest="sha256:prepared",
        )

        assert coordinator.claimability(claim.task_cid)["claimable"] is False
        dependent = coordinator.claimability("task:dependent")
        assert dependent["claimable"] is False
        assert dependent["blocked_dependency_task_cids"] == [claim.task_cid]

        forged = _completed_control_task(prepared)
        forged_binding = forged["body"]["completion_receipt"][
            "coordination_preparation"
        ]
        forged_binding["preparation_digest"] = "sha256:forged"
        with pytest.raises(DatabaseCoordinationStaleFenceError):
            coordinator.complete_task_claim(
                claim,
                control_completion_receipt=forged,
            )
        pending = coordinator.get_prepared_task_completion(claim.task_cid)
        assert pending is not None
        assert pending["preparation_digest"] == prepared["preparation_digest"]
    finally:
        coordinator.close()


def test_expired_prepared_completion_recovers_from_bound_control_task(
    tmp_path: Path,
) -> None:
    coordinator, clock = _open(tmp_path, default_lease_ms=10_000)
    try:
        coordinator.register_task(task_cid="task:recover", task_id="RECOVER")
        coordinator.register_task(
            task_cid="task:after-recover",
            task_id="AFTER",
            dependency_task_cids=("task:recover",),
        )
        claim = coordinator.claim_task(
            task_cid="task:recover",
            owner_session_id="session:old",
        )
        prepared = coordinator.prepare_task_completion(
            claim,
            control_expected_revision=2,
            evidence_digest="sha256:recover",
        )
        control_task = _completed_control_task(prepared)
        clock.advance(10_000)

        recovered = coordinator.recover_prepared_task_completion(
            claim.task_cid,
            control_completion_receipt=control_task,
        )
        assert recovered["recovered"] is True
        assert recovered["lease_state"] == LeaseState.COMPLETED.value
        promoted_preparation = coordinator.get_prepared_task_completion(
            claim.task_cid
        )
        assert promoted_preparation is not None
        assert promoted_preparation["status"] == AttemptStatus.SUCCEEDED.value
        assert coordinator.get_lease(claim.lease_id).state is LeaseState.COMPLETED
        assert coordinator.get_task_claim(claim.claim_id).state is LeaseState.COMPLETED
        assert (
            coordinator.get_task_attempt(claim.attempt_id).status
            is AttemptStatus.SUCCEEDED
        )
        assert coordinator.claimability("task:after-recover")["claimable"] is True
    finally:
        coordinator.close()


def test_expired_prepared_completion_aborts_only_with_unchanged_control_truth(
    tmp_path: Path,
) -> None:
    coordinator, clock = _open(tmp_path, default_lease_ms=10_000)
    try:
        coordinator.register_task(task_cid="task:abort", task_id="ABORT")
        claim = coordinator.claim_task(
            task_cid="task:abort",
            owner_session_id="session:old",
            idempotency_key="old-attempt",
        )
        prepared = coordinator.prepare_task_completion(
            claim,
            control_expected_revision=2,
            evidence_digest="sha256:abort",
        )
        clock.advance(10_000)

        with pytest.raises(DatabaseCoordinationStaleFenceError):
            coordinator.abort_prepared_task_completion(
                claim.task_cid,
                control_task_observation=_completed_control_task(prepared),
            )
        assert coordinator.get_prepared_task_completion(claim.task_cid) is not None

        aborted = coordinator.abort_prepared_task_completion(
            claim.task_cid,
            control_task_observation=_incomplete_control_task(prepared),
        )
        assert aborted["status"] == "aborted"
        assert aborted["ready"] is True
        assert coordinator.get_prepared_task_completion(claim.task_cid) is None
        replacement = coordinator.claim_task(
            task_cid=claim.task_cid,
            owner_session_id="session:new",
            idempotency_key="new-attempt",
        )
        assert replacement.attempt_number == claim.attempt_number + 1
        assert replacement.fencing_token > claim.fencing_token
    finally:
        coordinator.close()


def test_prepared_enumeration_atomically_expires_without_prior_sweep(
    tmp_path: Path,
) -> None:
    coordinator, clock = _open(tmp_path, default_lease_ms=10_000)
    try:
        coordinator.register_task(task_cid="task:lazy-expiry", task_id="LAZY")
        coordinator.register_task(
            task_cid="task:lazy-dependent",
            task_id="LAZY-DEPENDENT",
            dependency_task_cids=("task:lazy-expiry",),
        )
        claim = coordinator.claim_task(
            task_cid="task:lazy-expiry",
            owner_session_id="session:lazy",
        )
        prepared = coordinator.prepare_task_completion(
            claim,
            control_expected_revision=2,
            evidence_digest="sha256:lazy-expiry",
        )
        clock.advance(10_000)

        # Merely advancing the clock does not mutate stored projections.
        assert coordinator.get_task_claim(claim.claim_id).state is LeaseState.ACCEPTED
        assert coordinator.get_lease(claim.lease_id).state is LeaseState.ACCEPTED
        assert (
            coordinator.get_task_attempt(claim.attempt_id).status
            is AttemptStatus.RUNNING
        )

        pending = coordinator.list_prepared_task_completions(limit=10)
        assert len(pending) == 1
        assert pending[0]["preparation_digest"] == prepared["preparation_digest"]
        assert pending[0]["claim_state"] == LeaseState.EXPIRED.value
        assert pending[0]["lease_state"] == LeaseState.EXPIRED.value
        assert pending[0]["attempt_status"] == AttemptStatus.EXPIRED.value
        assert coordinator.get_task_claim(claim.claim_id).state is LeaseState.EXPIRED
        assert coordinator.get_lease(claim.lease_id).state is LeaseState.EXPIRED
        assert (
            coordinator.get_task_attempt(claim.attempt_id).status
            is AttemptStatus.EXPIRED
        )
        assert coordinator.claimability("task:lazy-dependent")["claimable"] is False

        aborted = coordinator.abort_prepared_task_completion(
            claim.task_cid,
            control_task_observation=_incomplete_control_task(prepared),
        )
        assert aborted["ready"] is True
        assert coordinator.claimability("task:lazy-expiry")["claimable"] is True
    finally:
        coordinator.close()


def test_promoted_completion_is_enumerated_and_reconciled_while_live(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:pending-live", task_id="PENDING")
        pending_claim = coordinator.claim_task(
            task_cid="task:pending-live",
            owner_session_id="session:pending",
        )
        coordinator.prepare_task_completion(
            pending_claim,
            control_expected_revision=2,
            evidence_digest="sha256:pending-live",
        )
        coordinator.register_task(task_cid="task:promoted-live", task_id="LIVE")
        claim = coordinator.claim_task(
            task_cid="task:promoted-live",
            owner_session_id="session:live",
        )
        prepared = coordinator.prepare_task_completion(
            claim,
            control_expected_revision=2,
            evidence_digest="sha256:promoted-live",
        )
        control_task = _completed_control_task(prepared)
        coordinator.complete_task_claim(
            claim,
            control_completion_receipt=control_task,
        )

        # A live pending preparation cannot starve an actionable promoted row
        # from a bounded reconciliation query.
        unsettled = coordinator.list_unsettled_task_completions(limit=1)
        assert len(unsettled) == 1
        assert unsettled[0]["task_cid"] == claim.task_cid
        assert unsettled[0]["status"] == AttemptStatus.SUCCEEDED.value
        assert unsettled[0]["lease_state"] == LeaseState.ACCEPTED.value
        assert unsettled[0]["attempt_status"] == AttemptStatus.RUNNING.value

        reconciled = coordinator.reconcile_promoted_task_completion(
            claim.task_cid,
            control_completion_receipt=control_task,
        )
        assert reconciled["lease_state"] == LeaseState.RELEASED.value
        assert reconciled["replayed"] is False
        assert coordinator.get_task_claim(claim.claim_id).state is LeaseState.RELEASED
        assert coordinator.get_lease(claim.lease_id).state is LeaseState.RELEASED
        assert (
            coordinator.get_task_attempt(claim.attempt_id).status
            is AttemptStatus.SUCCEEDED
        )
        remaining = coordinator.list_unsettled_task_completions(limit=10)
        assert [item["task_cid"] for item in remaining] == [pending_claim.task_cid]

        replay = coordinator.reconcile_promoted_task_completion(
            claim.task_cid,
            control_completion_receipt=control_task,
        )
        assert replay["lease_state"] == LeaseState.RELEASED.value
        assert replay["replayed"] is True

        # Settled history ahead of a bounded query cannot starve a later
        # promoted-but-unsettled barrier.
        coordinator.register_task(
            task_cid="task:promoted-live-next",
            task_id="LIVE-NEXT",
        )
        next_claim = coordinator.claim_task(
            task_cid="task:promoted-live-next",
            owner_session_id="session:live",
        )
        next_prepared = coordinator.prepare_task_completion(
            next_claim,
            control_expected_revision=2,
            evidence_digest="sha256:promoted-live-next",
        )
        coordinator.complete_task_claim(
            next_claim,
            control_completion_receipt=_completed_control_task(next_prepared),
        )
        bounded = coordinator.list_unsettled_task_completions(limit=1)
        assert [item["task_cid"] for item in bounded] == [next_claim.task_cid]
    finally:
        coordinator.close()


def test_promoted_completion_reconciliation_expires_and_recovers_atomically(
    tmp_path: Path,
) -> None:
    coordinator, clock = _open(tmp_path, default_lease_ms=10_000)
    try:
        coordinator.register_task(
            task_cid="task:promoted-expired",
            task_id="EXPIRED",
        )
        claim = coordinator.claim_task(
            task_cid="task:promoted-expired",
            owner_session_id="session:expired",
        )
        prepared = coordinator.prepare_task_completion(
            claim,
            control_expected_revision=2,
            evidence_digest="sha256:promoted-expired",
        )
        control_task = _completed_control_task(prepared)
        coordinator.complete_task_claim(
            claim,
            control_completion_receipt=control_task,
        )
        clock.advance(10_000)

        # No explicit lease sweep occurs before this atomic reconciliation.
        assert coordinator.get_task_claim(claim.claim_id).state is LeaseState.ACCEPTED
        forged_control_task = _completed_control_task(prepared)
        forged_control_task["body"]["completion_receipt"][
            "coordination_preparation"
        ]["claim_id"] = "claim:forged"
        with pytest.raises(DatabaseCoordinationStaleFenceError):
            coordinator.reconcile_promoted_task_completion(
                claim.task_cid,
                control_completion_receipt=forged_control_task,
            )
        assert coordinator.get_task_claim(claim.claim_id).state is LeaseState.ACCEPTED

        reconciled = coordinator.reconcile_promoted_task_completion(
            claim.task_cid,
            control_completion_receipt=control_task,
        )
        assert reconciled["lease_state"] == LeaseState.COMPLETED.value
        assert reconciled["replayed"] is False
        assert coordinator.get_task_claim(claim.claim_id).state is LeaseState.COMPLETED
        assert coordinator.get_lease(claim.lease_id).state is LeaseState.COMPLETED
        assert (
            coordinator.get_task_attempt(claim.attempt_id).status
            is AttemptStatus.SUCCEEDED
        )
        assert coordinator.list_unsettled_task_completions(limit=10) == []

        replay = coordinator.reconcile_promoted_task_completion(
            claim.task_cid,
            control_completion_receipt=control_task,
        )
        assert replay["lease_state"] == LeaseState.COMPLETED.value
        assert replay["replayed"] is True
    finally:
        coordinator.close()


def test_resource_and_maintenance_lease_projections(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        resource = coordinator.claim_resource(
            resource_kind="provider",
            resource_id="provider:openai",
            owner_session_id="session:scheduler",
            task_cid="task:x",
        )
        assert resource.resource_kind == "provider"
        assert resource.fencing_token >= 1
        fenced = resource.as_fenced_lease()
        assert fenced.lease_kind is LeaseKind.PROVIDER_CAPACITY

        maintenance = coordinator.acquire_maintenance_lease(
            owner_session_id="session:ops",
            scope="control-plane",
            process_birth_id="birth:abc",
        )
        assert maintenance.active
        loaded = coordinator.get_maintenance_lease(maintenance.lease_id)
        assert loaded is not None
        assert loaded.process_birth_id == "birth:abc"
        with pytest.raises(DatabaseCoordinationConflictError):
            coordinator.acquire_maintenance_lease(
                owner_session_id="session:other",
                scope="control-plane",
            )
        coordinator.release(maintenance.as_fenced_lease())
        second = coordinator.acquire_maintenance_lease(
            owner_session_id="session:other",
            scope="control-plane",
        )
        assert second.fence_epoch > maintenance.fence_epoch
    finally:
        coordinator.close()


def test_same_owner_reacquire_is_idempotent_without_idempotency_key(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        first = coordinator.acquire(
            lease_kind=LeaseKind.MERGE,
            scope="merge:lane-a",
            owner_session_id="session:same",
        )
        second = coordinator.acquire(
            lease_kind=LeaseKind.MERGE,
            scope="merge:lane-a",
            owner_session_id="session:same",
        )
        assert second.lease_id == first.lease_id
        assert second.fencing_token == first.fencing_token
    finally:
        coordinator.close()
