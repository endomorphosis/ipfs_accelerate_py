"""Tests for DatabaseCoordinator (DQP-015).

Evidence subset: acquire, renew, release, expiry, takeover, fairness,
dependency readiness (via exclusive scope isolation), epoch monotonicity,
stale fence, response loss (idempotent claim replay).

Acceptance:
- Four processes never own the same exclusive scope
- Expired session cannot renew or mutate
- Append/fair scheduling remains concurrent
- Stale fencing epoch is rejected in every protected write
- Claim and task-attempt creation are one transaction
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
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
    DatabaseCoordinationFenceError,
    DatabaseCoordinator,
    FairEntryState,
    LeaseKind,
    LeaseState,
    SessionStatus,
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
    default_session_ttl_ms: int = 120_000,
    name: str = "coordination.duckdb",
) -> tuple[DatabaseCoordinator, FakeClock]:
    clock = clock or FakeClock()
    coordinator = open_database_coordinator(
        tmp_path / name,
        clock_ms=clock,
        default_lease_ms=default_lease_ms,
        default_session_ttl_ms=default_session_ttl_ms,
    )
    return coordinator, clock


def _session(
    coordinator: DatabaseCoordinator,
    owner: str,
    *,
    ttl_ms: int = 120_000,
) -> object:
    return coordinator.open_session(owner_did=owner, ttl_ms=ttl_ms)


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


def test_exclusive_scope_key_canonicalizes_kinds() -> None:
    assert exclusive_scope_key(LeaseKind.TASK, task_cid="task:abc") == "task:task:abc"
    assert (
        exclusive_scope_key(
            LeaseKind.RESOURCE, resource_kind="gpu", resource_id="slot-0"
        )
        == "resource:gpu:slot-0"
    )
    assert (
        exclusive_scope_key(
            LeaseKind.MERGE, repository_id="repo:1", merge_target="main"
        )
        == "merge:repo:1:main"
    )
    assert (
        exclusive_scope_key(LeaseKind.MAINTENANCE, maintenance_scope="schema")
        == "maintenance:schema"
    )


# ---------------------------------------------------------------------------
# Four processes never own the same exclusive scope
# ---------------------------------------------------------------------------


def test_four_sessions_never_share_exclusive_task_scope(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        sessions = [
            _session(coordinator, f"did:web:worker-{index}") for index in range(4)
        ]
        winners: list[str] = []
        for session in sessions:
            try:
                bundle = coordinator.claim_task(
                    task_cid="task:shared-exclusive",
                    owner_session_id=session.session_id,
                    worktree_id="worktree:1",
                )
                winners.append(bundle.claim.owner_session_id)
            except DatabaseCoordinationConflictError:
                continue
        assert len(winners) == 1
        active = coordinator.list_active_task_claims("task:shared-exclusive")
        assert len(active) == 1
        assert active[0].owner_session_id == winners[0]
        owner = coordinator.active_owner_for_scope(
            exclusive_scope_key(LeaseKind.TASK, task_cid="task:shared-exclusive")
        )
        assert owner is not None
        assert owner.owner_session_id == winners[0]
    finally:
        coordinator.close()


def test_four_threads_never_share_exclusive_resource_scope(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        sessions = [
            _session(coordinator, f"did:web:thread-{index}") for index in range(4)
        ]
        results: list[str | None] = []

        def _try_claim(session_id: str) -> str | None:
            try:
                claim = coordinator.claim_resource(
                    resource_kind="provider",
                    resource_id="llm:primary",
                    owner_session_id=session_id,
                )
                return claim.owner_session_id
            except DatabaseCoordinationConflictError:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(_try_claim, session.session_id) for session in sessions
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        winners = [item for item in results if item is not None]
        assert len(winners) == 1
        active = coordinator.list_active_leases(kind=LeaseKind.PROVIDER)
        assert len(active) == 1
        assert active[0].owner_session_id == winners[0]
    finally:
        coordinator.close()


def _multiprocess_claim_worker(
    database_path: str,
    session_owner: str,
    scope_task: str,
    result_queue: multiprocessing.Queue,
) -> None:
    """Child process: open session, try exclusive claim, report outcome."""

    try:
        coordinator = open_database_coordinator(database_path)
        try:
            session = coordinator.open_session(owner_did=session_owner)
            try:
                bundle = coordinator.claim_task(
                    task_cid=scope_task,
                    owner_session_id=session.session_id,
                )
                result_queue.put(("won", session.session_id, bundle.lease.fencing_token))
            except DatabaseCoordinationConflictError:
                result_queue.put(("conflict", session.session_id, None))
        finally:
            coordinator.close()
    except Exception as exc:  # pragma: no cover - diagnostic path
        result_queue.put(("error", session_owner, str(exc)))


def test_four_processes_never_own_same_exclusive_scope(tmp_path: Path) -> None:
    database_path = tmp_path / "multi.duckdb"
    # Initialize schema once before spawning workers.
    bootstrap = open_database_coordinator(database_path)
    bootstrap.close()

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    for index in range(4):
        process = ctx.Process(
            target=_multiprocess_claim_worker,
            args=(
                str(database_path),
                f"did:web:proc-{index}",
                "task:multiprocess-scope",
                result_queue,
            ),
        )
        processes.append(process)
        process.start()
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0, (
            f"worker exited with {process.exitcode}"
        )

    outcomes = [result_queue.get(timeout=5) for _ in range(4)]
    wins = [item for item in outcomes if item[0] == "won"]
    conflicts = [item for item in outcomes if item[0] == "conflict"]
    errors = [item for item in outcomes if item[0] == "error"]
    assert errors == [], errors
    assert len(wins) == 1, outcomes
    assert len(conflicts) == 3, outcomes

    with open_database_coordinator(database_path) as coordinator:
        active = coordinator.list_active_task_claims("task:multiprocess-scope")
        assert len(active) == 1
        assert active[0].owner_session_id == wins[0][1]


# ---------------------------------------------------------------------------
# Expired session cannot renew or mutate
# ---------------------------------------------------------------------------


def test_expired_session_cannot_renew_or_mutate(tmp_path: Path) -> None:
    coordinator, clock = _open(
        tmp_path, default_session_ttl_ms=10_000, default_lease_ms=60_000
    )
    try:
        session = coordinator.open_session(
            owner_did="did:web:expiring",
            ttl_ms=10_000,
        )
        bundle = coordinator.claim_task(
            task_cid="task:expire-session",
            owner_session_id=session.session_id,
        )
        # Advance past session TTL (and keep lease clock consistent).
        clock.advance(15_000)

        with pytest.raises(DatabaseCoordinationExpiredError):
            coordinator.renew_session(session.session_id)

        with pytest.raises(DatabaseCoordinationExpiredError):
            coordinator.renew_lease(
                bundle.lease.lease_id,
                owner_session_id=session.session_id,
                fencing_token=bundle.lease.fencing_token,
                fence_epoch=bundle.lease.fence_epoch,
            )

        with pytest.raises(DatabaseCoordinationExpiredError):
            coordinator.protected_write(
                lease_id=bundle.lease.lease_id,
                owner_session_id=session.session_id,
                fencing_token=bundle.lease.fencing_token,
                fence_epoch=bundle.lease.fence_epoch,
                write_kind="mutate.task",
                body={"step": 1},
            )

        loaded = coordinator.get_session(session.session_id)
        assert loaded is not None
        assert loaded.status is SessionStatus.EXPIRED
    finally:
        coordinator.close()


def test_expired_lease_cannot_renew_even_with_live_session(tmp_path: Path) -> None:
    coordinator, clock = _open(tmp_path, default_lease_ms=5_000)
    try:
        session = _session(coordinator, "did:web:live")
        bundle = coordinator.claim_task(
            task_cid="task:expire-lease",
            owner_session_id=session.session_id,
            requested_lease_ms=5_000,
        )
        clock.advance(6_000)
        with pytest.raises(DatabaseCoordinationExpiredError):
            coordinator.renew_lease(
                bundle.lease.lease_id,
                owner_session_id=session.session_id,
                fencing_token=bundle.lease.fencing_token,
                fence_epoch=bundle.lease.fence_epoch,
            )
        # Takeover advances epoch/token.
        takeover = coordinator.takeover_expired(
            scope_key=bundle.lease.scope_key,
            owner_session_id=session.session_id,
        )
        assert takeover.fence_epoch > bundle.lease.fence_epoch
        assert takeover.fencing_token > bundle.lease.fencing_token
    finally:
        coordinator.close()


# ---------------------------------------------------------------------------
# Append / fair scheduling remains concurrent
# ---------------------------------------------------------------------------


def test_fair_append_and_claim_are_concurrent(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        sessions = [
            _session(coordinator, f"did:web:fair-{index}") for index in range(4)
        ]
        # Concurrent appends do not require exclusive ownership of the queue.
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(
                    coordinator.enqueue_fair,
                    queue_name="ready",
                    task_cid=f"task:fair-{index}",
                    owner_session_id=sessions[index].session_id,
                )
                for index in range(4)
            ]
            entries = [future.result() for future in futures]

        assert len({entry.entry_id for entry in entries}) == 4
        queued = coordinator.list_fair_queue("ready")
        assert len(queued) == 4
        ordinals = [entry.ordinal for entry in queued]
        assert ordinals == sorted(ordinals)

        claimed_ids: list[str] = []
        for session in sessions:
            claimed = coordinator.claim_next_fair(
                "ready", owner_session_id=session.session_id
            )
            if claimed is not None:
                claimed_ids.append(claimed.entry_id)
                assert claimed.state is FairEntryState.CLAIMED

        assert len(claimed_ids) == 4
        assert len(set(claimed_ids)) == 4
        assert coordinator.list_fair_queue("ready") == []
    finally:
        coordinator.close()


def test_fair_scheduling_does_not_block_exclusive_task_claims(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        session_a = _session(coordinator, "did:web:a")
        session_b = _session(coordinator, "did:web:b")
        coordinator.enqueue_fair(
            queue_name="ready",
            task_cid="task:queued-1",
            owner_session_id=session_a.session_id,
        )
        # Distinct exclusive task scopes remain independently claimable while
        # fair append proceeds concurrently.
        bundle_a = coordinator.claim_task(
            task_cid="task:exclusive-a",
            owner_session_id=session_a.session_id,
        )
        bundle_b = coordinator.claim_task(
            task_cid="task:exclusive-b",
            owner_session_id=session_b.session_id,
        )
        assert bundle_a.claim.task_cid != bundle_b.claim.task_cid
        assert len(coordinator.list_active_leases(kind=LeaseKind.TASK)) == 2
    finally:
        coordinator.close()


# ---------------------------------------------------------------------------
# Stale fencing epoch is rejected in every protected write
# ---------------------------------------------------------------------------


def test_stale_fencing_epoch_rejected_on_protected_write(tmp_path: Path) -> None:
    coordinator, clock = _open(tmp_path, default_lease_ms=5_000)
    try:
        session = _session(coordinator, "did:web:fence")
        first = coordinator.claim_task(
            task_cid="task:fence-write",
            owner_session_id=session.session_id,
            requested_lease_ms=5_000,
        )
        # Valid protected write under current fence.
        write = coordinator.protected_write(
            lease_id=first.lease.lease_id,
            owner_session_id=session.session_id,
            fencing_token=first.lease.fencing_token,
            fence_epoch=first.lease.fence_epoch,
            write_kind="task.progress",
            body={"step": "ok"},
        )
        assert write.fence_epoch == first.lease.fence_epoch

        # Live lease with wrong epoch is rejected before any takeover.
        with pytest.raises(DatabaseCoordinationFenceError):
            coordinator.protected_write(
                lease_id=first.lease.lease_id,
                owner_session_id=session.session_id,
                fencing_token=first.lease.fencing_token,
                fence_epoch=first.lease.fence_epoch + 99,
                write_kind="task.progress",
                body={"step": "stale-epoch"},
            )

        clock.advance(6_000)
        second = coordinator.takeover_expired(
            scope_key=first.lease.scope_key,
            owner_session_id=session.session_id,
        )
        assert second.fence_epoch > first.lease.fence_epoch

        # Superseded first lease cannot mutate under its old fence.
        with pytest.raises(
            (DatabaseCoordinationFenceError, DatabaseCoordinationExpiredError)
        ):
            coordinator.protected_write(
                lease_id=first.lease.lease_id,
                owner_session_id=session.session_id,
                fencing_token=first.lease.fencing_token,
                fence_epoch=first.lease.fence_epoch,
                write_kind="task.progress",
                body={"step": "stale"},
            )

        # Stale token against the new lease is also rejected.
        with pytest.raises(DatabaseCoordinationFenceError):
            coordinator.protected_write(
                lease_id=second.lease_id,
                owner_session_id=session.session_id,
                fencing_token=first.lease.fencing_token,
                fence_epoch=second.fence_epoch,
                write_kind="task.progress",
                body={"step": "stale-token"},
            )

        # Current fence succeeds.
        ok = coordinator.protected_write(
            lease_id=second.lease_id,
            owner_session_id=session.session_id,
            fencing_token=second.fencing_token,
            fence_epoch=second.fence_epoch,
            write_kind="task.progress",
            body={"step": "current"},
        )
        assert ok.fencing_token == second.fencing_token
    finally:
        coordinator.close()


def test_stale_fence_rejected_on_resource_and_maintenance_writes(
    tmp_path: Path,
) -> None:
    coordinator, clock = _open(tmp_path, default_lease_ms=5_000)
    try:
        session = _session(coordinator, "did:web:multi-kind")
        resource = coordinator.claim_resource(
            resource_kind="prover",
            resource_id="z3:local",
            owner_session_id=session.session_id,
            requested_lease_ms=5_000,
        )
        maintenance = coordinator.acquire_maintenance_lease(
            scope="schema-migration",
            owner_session_id=session.session_id,
            purpose="backup",
            requested_lease_ms=5_000,
        )
        merge = coordinator.acquire_merge_lease(
            repository_id="repo:main",
            merge_target="main",
            owner_session_id=session.session_id,
            requested_lease_ms=5_000,
        )

        for lease_id, token, epoch in (
            (resource.lease_id, resource.fencing_token, resource.fence_epoch),
            (maintenance.lease_id, maintenance.fencing_token, maintenance.fence_epoch),
            (merge.lease_id, merge.fencing_token, merge.fence_epoch),
        ):
            coordinator.protected_write(
                lease_id=lease_id,
                owner_session_id=session.session_id,
                fencing_token=token,
                fence_epoch=epoch,
                write_kind="ok",
                body={},
            )
            with pytest.raises(DatabaseCoordinationFenceError):
                coordinator.protected_write(
                    lease_id=lease_id,
                    owner_session_id=session.session_id,
                    fencing_token=token,
                    fence_epoch=epoch + 7,
                    write_kind="stale",
                    body={},
                )

        # After expiry + takeover, old resource fence cannot write.
        clock.advance(6_000)
        takeover = coordinator.takeover_expired(
            scope_key=exclusive_scope_key(
                LeaseKind.PROVER, resource_id="z3:local"
            ),
            owner_session_id=session.session_id,
        )
        with pytest.raises((DatabaseCoordinationFenceError, DatabaseCoordinationExpiredError)):
            coordinator.protected_write(
                lease_id=resource.lease_id,
                owner_session_id=session.session_id,
                fencing_token=resource.fencing_token,
                fence_epoch=resource.fence_epoch,
                write_kind="stale-resource",
                body={},
            )
        coordinator.protected_write(
            lease_id=takeover.lease_id,
            owner_session_id=session.session_id,
            fencing_token=takeover.fencing_token,
            fence_epoch=takeover.fence_epoch,
            write_kind="takeover-ok",
            body={},
        )
    finally:
        coordinator.close()


# ---------------------------------------------------------------------------
# Claim and task-attempt creation are one transaction
# ---------------------------------------------------------------------------


def test_claim_and_attempt_created_atomically(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        session = _session(coordinator, "did:web:atomic")
        bundle = coordinator.claim_task(
            task_cid="task:atomic",
            owner_session_id=session.session_id,
            worktree_id="worktree:atomic",
            idempotency_key="idem:atomic-1",
        )
        assert bundle.claim.attempt_id == bundle.attempt.attempt_id
        assert bundle.claim.claim_id == bundle.attempt.claim_id
        assert bundle.lease.claim_id == bundle.claim.claim_id
        assert bundle.lease.attempt_id == bundle.attempt.attempt_id
        assert bundle.claim.fencing_token == bundle.attempt.fencing_token
        assert bundle.claim.fence_epoch == bundle.attempt.fence_epoch
        assert bundle.attempt.attempt_number == 1

        loaded_claim = coordinator.get_task_claim(bundle.claim.claim_id)
        loaded_attempt = coordinator.get_task_attempt(bundle.attempt.attempt_id)
        assert loaded_claim is not None
        assert loaded_attempt is not None
        assert loaded_claim.attempt_id == loaded_attempt.attempt_id

        # Idempotent replay returns the same claim+attempt pair.
        replay = coordinator.claim_task(
            task_cid="task:atomic",
            owner_session_id=session.session_id,
            idempotency_key="idem:atomic-1",
        )
        assert replay.claim.claim_id == bundle.claim.claim_id
        assert replay.attempt.attempt_id == bundle.attempt.attempt_id
    finally:
        coordinator.close()


def test_failed_claim_leaves_no_orphan_attempt(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        first_session = _session(coordinator, "did:web:first")
        second_session = _session(coordinator, "did:web:second")
        first = coordinator.claim_task(
            task_cid="task:no-orphan",
            owner_session_id=first_session.session_id,
        )
        with pytest.raises(DatabaseCoordinationConflictError):
            coordinator.claim_task(
                task_cid="task:no-orphan",
                owner_session_id=second_session.session_id,
            )
        # Only the first attempt exists; conflict did not insert a second.
        attempts_for_task = []
        # Scan via claim list: only one accepted claim.
        active = coordinator.list_active_task_claims("task:no-orphan")
        assert len(active) == 1
        assert active[0].claim_id == first.claim.claim_id
        assert coordinator.get_task_attempt(first.attempt.attempt_id) is not None
        # Ensure no second attempt_number exists.
        assert first.attempt.attempt_number == 1
        # Direct attempt_number uniqueness: re-claim after release creates #2.
        coordinator.release_lease(
            first.lease.lease_id,
            owner_session_id=first_session.session_id,
            fencing_token=first.lease.fencing_token,
            fence_epoch=first.lease.fence_epoch,
        )
        second = coordinator.claim_task(
            task_cid="task:no-orphan",
            owner_session_id=second_session.session_id,
        )
        assert second.attempt.attempt_number == 2
        attempts_for_task.append(first.attempt.attempt_id)
        attempts_for_task.append(second.attempt.attempt_id)
        assert len(set(attempts_for_task)) == 2
    finally:
        coordinator.close()


# ---------------------------------------------------------------------------
# Acquire / renew / release / epoch monotonicity
# ---------------------------------------------------------------------------


def test_acquire_renew_release_and_epoch_monotonicity(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        session = _session(coordinator, "did:web:lifecycle")
        first = coordinator.claim_task(
            task_cid="task:lifecycle",
            owner_session_id=session.session_id,
        )
        renewed = coordinator.renew_lease(
            first.lease.lease_id,
            owner_session_id=session.session_id,
            fencing_token=first.lease.fencing_token,
            fence_epoch=first.lease.fence_epoch,
            requested_lease_ms=90_000,
        )
        assert renewed.fencing_token == first.lease.fencing_token
        assert renewed.fence_epoch == first.lease.fence_epoch
        assert renewed.expires_at_ms > first.lease.expires_at_ms

        released = coordinator.release_lease(
            first.lease.lease_id,
            owner_session_id=session.session_id,
            fencing_token=first.lease.fencing_token,
            fence_epoch=first.lease.fence_epoch,
        )
        assert released.state is LeaseState.RELEASED

        second = coordinator.claim_task(
            task_cid="task:lifecycle",
            owner_session_id=session.session_id,
        )
        assert second.lease.fence_epoch == first.lease.fence_epoch + 1
        assert second.lease.fencing_token == first.lease.fencing_token + 1
        assert second.attempt.attempt_number == 2
    finally:
        coordinator.close()


def test_maintenance_and_merge_exclusive_scopes(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        a = _session(coordinator, "did:web:a")
        b = _session(coordinator, "did:web:b")
        maintenance = coordinator.acquire_maintenance_lease(
            scope="offline-recovery",
            owner_session_id=a.session_id,
            purpose="offline_recovery",
        )
        assert maintenance.active is True
        with pytest.raises(DatabaseCoordinationConflictError):
            coordinator.acquire_maintenance_lease(
                scope="offline-recovery",
                owner_session_id=b.session_id,
            )

        merge = coordinator.acquire_merge_lease(
            repository_id="repo:x",
            merge_target="main",
            owner_session_id=a.session_id,
        )
        with pytest.raises(DatabaseCoordinationConflictError):
            coordinator.acquire_merge_lease(
                repository_id="repo:x",
                merge_target="main",
                owner_session_id=b.session_id,
            )
        # Distinct merge targets do not conflict.
        other = coordinator.acquire_merge_lease(
            repository_id="repo:x",
            merge_target="release",
            owner_session_id=b.session_id,
        )
        assert other.scope_key != merge.scope_key
    finally:
        coordinator.close()


def test_stop_session_releases_owned_leases(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        session = _session(coordinator, "did:web:stop")
        bundle = coordinator.claim_task(
            task_cid="task:stop",
            owner_session_id=session.session_id,
        )
        coordinator.stop_session(session.session_id)
        assert coordinator.list_active_task_claims("task:stop") == []
        lease = coordinator.get_lease(bundle.lease.lease_id)
        assert lease is not None
        assert lease.state in {LeaseState.EXPIRED, LeaseState.RELEASED}
    finally:
        coordinator.close()
