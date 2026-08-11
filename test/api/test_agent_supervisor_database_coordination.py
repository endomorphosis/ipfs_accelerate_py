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
