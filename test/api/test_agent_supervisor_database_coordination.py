"""Tests for DatabaseCoordinator (DQP-015).

Evidence subset: acquire, renew, release, expiry, takeover, fairness,
dependency readiness, epoch monotonicity, stale fence, response loss.

Acceptance: Four processes never own the same exclusive scope; expired session
cannot renew or mutate; append/fair scheduling remains concurrent; stale
fencing epoch is rejected in every protected write; claim and task-attempt
creation are one transaction.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    CONTROL_READY_FRONTIER_RECONCILIATION_EVENT,
    CONTROL_READY_FRONTIER_RECONCILIATION_SCHEMA,
    COORDINATION_HISTORY_PROJECTION_SCHEMA,
    COORDINATION_REGISTRY_PROJECTION_SCHEMA,
    DATABASE_COORDINATOR_INTERFACE,
    FENCED_LEASE_INTERFACE,
    MAINTENANCE_LEASE_INTERFACE,
    RESOURCE_CLAIM_INTERFACE,
    TASK_CLAIM_INTERFACE,
    TASK_COMPLETION_REARM_SCHEMA,
    TASK_DEPENDENCY_AMENDMENT_SCHEMA,
    TYPED_STRICT_REQUEUE_ATTEMPT_FLOOR_SOURCE,
    AttemptStatus,
    DatabaseCoordinationBoundsError,
    DatabaseCoordinationConflictError,
    DatabaseCoordinationError,
    DatabaseCoordinationExpiredError,
    DatabaseCoordinationNotReadyError,
    DatabaseCoordinationStaleFenceError,
    DatabaseCoordinator,
    LeaseKind,
    LeaseMode,
    LeaseState,
    ResourceClaim,
    TaskClaim,
    duckdb_available,
    exclusive_scope_key,
    open_database_coordinator,
    read_coordination_history_projection,
    read_coordination_registry_projection,
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


def _settled_control_completion(
    coordinator: DatabaseCoordinator,
    task_cid: str,
    *,
    expected_revision: int = 2,
) -> tuple[TaskClaim, dict[str, object]]:
    claim = coordinator.claim_task(
        task_cid=task_cid,
        owner_session_id=f"session:{task_cid}",
    )
    prepared = coordinator.prepare_task_completion(
        claim,
        control_expected_revision=expected_revision,
        evidence_digest=f"sha256:{task_cid}",
    )
    control_task = _completed_control_task(prepared)
    coordinator.complete_task_claim(
        claim,
        control_completion_receipt=control_task,
    )
    coordinator.settle_task_claim(claim)
    return claim, control_task


def _control_rearm_observation(
    *,
    task_cid: str,
    task_alias: str,
    revision: int = 4,
    status: str = "retrying",
    nested: bool = True,
) -> dict[str, object]:
    task: dict[str, object] = {
        "task_cid": task_cid,
        "task_alias": task_alias,
        "status": status,
        "revision": revision,
        "body": {},
    }
    if not nested:
        return task
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/database-task-cas@1",
        "task": task,
        "previous_status": "completed",
        "revision": revision,
        "event_cursor": 9,
        "changed": True,
        "receipt_cid": "cid:control-rearm",
    }


def _control_task_projection(
    *,
    task_cid: str,
    task_alias: str,
    status: str,
    revision: int,
) -> dict[str, object]:
    return {
        "task_cid": task_cid,
        "task_alias": task_alias,
        "goal_cid": "goal:control",
        "plan_cid": "plan:control",
        "objective_id": "objective:control",
        "ordinal": 1,
        "status": status,
        "revision": revision,
        "priority": "P0",
        "body": {"authority": "canonical-control"},
        "dependencies": [],
        "outputs": [],
        "acceptance": [],
        "validations": [],
    }


def _control_ready_frontier(
    *tasks: dict[str, object],
    revision: int = 50,
) -> dict[str, object]:
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/database-task-page@1",
        "tasks": list(tasks),
        "revision": revision,
        "next_cursor": "",
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


def _claim_task_and_writer(
    coordinator: DatabaseCoordinator,
    *,
    task_lease_ms: int = 30_000,
    writer_lease_ms: int = 30_000,
) -> tuple[TaskClaim, ResourceClaim]:
    coordinator.register_task(task_cid="task:guarded", task_id="GUARDED")
    claim = coordinator.claim_task(
        task_cid="task:guarded",
        owner_session_id="session:guarded",
        lease_ms=task_lease_ms,
    )
    writer = coordinator.claim_resource(
        resource_kind="database_writer",
        resource_id="control-store:guarded",
        owner_session_id="session:guarded",
        task_cid="task:guarded",
        repository_id="repository:guarded",
        lease_ms=writer_lease_ms,
        body={"purpose": "control_cas"},
    )
    return claim, writer


def test_cross_store_callback_runs_under_exact_task_and_writer_fences(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        claim, writer = _claim_task_and_writer(coordinator)
        prepared = coordinator.prepare_task_completion(
            claim,
            control_expected_revision=1,
            control_expected_status="todo",
            evidence_digest="sha256:evidence",
            body={"requires_cross_store_fence_guard": True},
        )
        control_receipt = _completed_control_task(
            prepared,
            nested_cas_result=True,
        )
        calls: list[str] = []

        def control_cas() -> dict[str, object]:
            calls.append("called")
            return control_receipt

        result = coordinator.execute_with_task_and_resource_fences(
            claim,
            writer,
            control_cas,
            allow_logically_completed=True,
        )

        assert result == control_receipt
        assert calls == ["called"]
        event_types = {item["event_type"] for item in coordinator.lease_events()}
        assert "protected_task_write" in event_types
        assert "protected_resource_write" in event_types
        assert "cross_store_fence_guard_succeeded" in event_types
        promoted = coordinator.complete_task_claim(
            claim,
            control_completion_receipt=control_receipt,
        )
        assert promoted["status"] == "succeeded"
        assert coordinator.get_task_claim(claim.claim_id) is not None
        assert coordinator.get_lease(writer.lease_id) is not None
    finally:
        coordinator.close()


def test_cross_store_callback_failure_rolls_back_coordinator_guard(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        claim, writer = _claim_task_and_writer(coordinator)
        before = coordinator.lease_events()

        def failing_control_cas() -> None:
            raise RuntimeError("control CAS failed")

        with pytest.raises(RuntimeError, match="control CAS failed"):
            coordinator.execute_with_task_and_resource_fences(
                claim,
                writer,
                failing_control_cas,
            )

        assert coordinator.lease_events() == before
        assert coordinator.get_task_claim(claim.claim_id).state is LeaseState.ACCEPTED
        assert coordinator.get_lease(writer.lease_id).state is LeaseState.ACCEPTED
    finally:
        coordinator.close()


def test_guarded_completion_cannot_promote_without_guard_success_receipt(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        claim, _writer = _claim_task_and_writer(coordinator)
        prepared = coordinator.prepare_task_completion(
            claim,
            control_expected_revision=1,
            control_expected_status="todo",
            evidence_digest="sha256:unguarded-control-cas",
            body={"requires_cross_store_fence_guard": True},
        )
        control_receipt = _completed_control_task(
            prepared,
            nested_cas_result=True,
        )

        with pytest.raises(DatabaseCoordinationNotReadyError) as excinfo:
            coordinator.complete_task_claim(
                claim,
                control_completion_receipt=control_receipt,
            )
        assert excinfo.value.evidence["reason"] == "cross_store_fence_guard_missing"
        pending = coordinator.get_prepared_task_completion(claim.task_cid)
        assert pending is not None
        assert pending["status"] == "prepared"
    finally:
        coordinator.close()


def test_cross_store_callback_rejects_coordinator_reentry_even_when_swallowed(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        claim, writer = _claim_task_and_writer(coordinator)
        before = coordinator.lease_events()

        def reentrant_callback() -> str:
            with pytest.raises(
                DatabaseCoordinationConflictError,
                match="must not re-enter",
            ):
                coordinator.release(writer.as_fenced_lease())
            return "swallowed"

        with pytest.raises(
            DatabaseCoordinationConflictError,
            match="attempted to re-enter",
        ):
            coordinator.execute_with_task_and_resource_fences(
                claim,
                writer,
                reentrant_callback,
            )

        assert coordinator.lease_events() == before
        assert coordinator.get_lease(writer.lease_id).state is LeaseState.ACCEPTED
    finally:
        coordinator.close()


def test_cross_store_callback_fails_postcheck_when_fences_expire(
    tmp_path: Path,
) -> None:
    coordinator, clock = _open(tmp_path, default_lease_ms=10_000)
    try:
        claim, writer = _claim_task_and_writer(
            coordinator,
            task_lease_ms=10_000,
            writer_lease_ms=10_000,
        )
        prepared = coordinator.prepare_task_completion(
            claim,
            control_expected_revision=1,
            control_expected_status="todo",
            evidence_digest="sha256:expiry-evidence",
            body={"requires_cross_store_fence_guard": True},
        )
        control_receipt = _completed_control_task(
            prepared,
            nested_cas_result=True,
        )
        before = coordinator.lease_events()
        external_effects: list[dict[str, object]] = []

        def slow_control_cas() -> dict[str, object]:
            external_effects.append(control_receipt)
            clock.advance(10_001)
            return control_receipt

        with pytest.raises(DatabaseCoordinationExpiredError):
            coordinator.execute_with_task_and_resource_fences(
                claim,
                writer,
                slow_control_cas,
                allow_logically_completed=True,
            )

        # The coordinator transaction rolls back, but an external callback's
        # effect cannot be undone and must be reconciled by its receipt.
        assert external_effects == [control_receipt]
        assert coordinator.lease_events() == before
        with pytest.raises(DatabaseCoordinationNotReadyError) as recovery_error:
            coordinator.recover_prepared_task_completion(
                claim.task_cid,
                control_completion_receipt=control_receipt,
            )
        assert (
            recovery_error.value.evidence["reason"]
            == "cross_store_fence_guard_missing"
        )
        assert coordinator.list_active_leases() == []
    finally:
        coordinator.close()


def test_cross_store_callback_postcheck_rejects_injected_later_writer_fence(
    tmp_path: Path,
) -> None:
    coordinator, clock = _open(tmp_path)
    try:
        claim, writer = _claim_task_and_writer(coordinator)
        connection = coordinator._require()
        before = coordinator.lease_events()
        scope_key = writer.as_fenced_lease().scope_key

        def inject_illicit_takeover() -> str:
            # Deliberately bypass the public coordinator API to exercise the
            # post-callback latest-fence check. Supported re-entry is rejected
            # separately above.
            connection.execute(
                """
                INSERT INTO token_history(
                    scope_key, fencing_token, fence_epoch, recorded_at_ms
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    scope_key,
                    writer.fencing_token + 1,
                    writer.fence_epoch + 1,
                    clock.now,
                ],
            )
            return "forged-result"

        with pytest.raises(DatabaseCoordinationStaleFenceError, match="latest"):
            coordinator.execute_with_task_and_resource_fences(
                claim,
                writer,
                inject_illicit_takeover,
            )

        assert coordinator.lease_events() == before
        latest = connection.execute(
            """
            SELECT MAX(fencing_token), MAX(fence_epoch)
            FROM token_history WHERE scope_key = ?
            """,
            [scope_key],
        ).fetchone()
        assert latest is not None
        assert (latest[0], latest[1]) == (
            writer.fencing_token,
            writer.fence_epoch,
        )
    finally:
        coordinator.close()


def test_cross_store_callback_rejects_superseded_writer_before_execution(
    tmp_path: Path,
) -> None:
    coordinator, clock = _open(tmp_path)
    try:
        claim, writer = _claim_task_and_writer(
            coordinator,
            task_lease_ms=30_000,
            writer_lease_ms=10_000,
        )
        clock.advance(10_001)
        successor = coordinator.claim_resource(
            resource_kind="database_writer",
            resource_id=writer.resource_id,
            owner_session_id="session:successor",
            task_cid="task:successor",
            repository_id=writer.repository_id,
            lease_ms=10_000,
            body={"purpose": "successor_control_cas"},
        )
        assert successor.fencing_token > writer.fencing_token
        calls: list[str] = []

        with pytest.raises(DatabaseCoordinationStaleFenceError, match="latest"):
            coordinator.execute_with_task_and_resource_fences(
                claim,
                writer,
                lambda: calls.append("called"),
            )
        assert calls == []
    finally:
        coordinator.close()


def test_coordination_registry_projection_is_exact_and_timestamp_free(
    tmp_path: Path,
) -> None:
    first, _first_clock = _open(tmp_path / "first", clock=FakeClock(1_000_000))
    second, _second_clock = _open(tmp_path / "second", clock=FakeClock(9_000_000))
    try:
        for coordinator, timestamp in ((first, 1_000_000), (second, 9_000_000)):
            coordinator.register_task(
                task_cid="task:dep",
                task_id="DEP",
                body={"kind": "analysis", "nested": {"ordinal": 1}},
                now_ms=timestamp,
            )
            coordinator.register_task(
                task_cid="task:child",
                task_id="CHILD",
                worktree_id="worktree:child",
                dependency_task_cids=("task:dep",),
                body={"kind": "implementation"},
                now_ms=timestamp + 50,
            )
            coordinator.mark_task_complete(
                "task:dep",
                status="succeeded",
                body={"receipt_cid": "sha256:dep"},
                now_ms=timestamp + 100,
            )

        projection = first.coordination_registry_projection()
        assert projection["schema"] == COORDINATION_REGISTRY_PROJECTION_SCHEMA
        assert projection["tasks"] == [
            {
                "task_cid": "task:child",
                "task_id": "CHILD",
                "worktree_id": "worktree:child",
                "ready": True,
                "body": {"kind": "implementation"},
            },
            {
                "task_cid": "task:dep",
                "task_id": "DEP",
                "worktree_id": "",
                "ready": False,
                "body": {"kind": "analysis", "nested": {"ordinal": 1}},
            },
        ]
        assert projection["dependency_edges"] == [
            {
                "task_cid": "task:child",
                "dependency_task_cid": "task:dep",
            }
        ]
        assert projection["logical_completions"] == [
            {
                "task_cid": "task:dep",
                "status": "succeeded",
                "body": {"receipt_cid": "sha256:dep"},
            }
        ]
        assert projection["counts"] == {
            "registered_tasks": 2,
            "dependency_edges": 1,
            "logical_completions": 1,
            "task_claims": 0,
            "active_task_claims": 0,
            "resource_claims": 0,
            "active_resource_claims": 0,
            "task_attempts": 0,
            "active_task_attempts": 0,
            "fenced_leases": 0,
            "active_fenced_leases": 0,
            "maintenance_leases": 0,
            "active_maintenance_leases": 0,
        }
        assert projection["projection_root"].startswith("sha256:")
        assert len(projection["projection_root"]) == 71

        # Registration and completion wall-clock values are deliberately not
        # logical registry identity.
        assert second.coordination_registry_projection() == projection
        assert first.coordination_registry_projection() == projection
    finally:
        first.close()
        second.close()


def test_authoritative_task_sync_is_idempotent_fail_closed_and_preserves_prepared(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        completed = coordinator.synchronize_authoritative_task(
            task_cid="task:dep",
            task_id="DEP",
            authoritative_status="completed",
            authoritative_revision=1,
            authoritative_ready=False,
            authoritative_completed=True,
            now_ms=1_000_000,
        )
        assert completed["changed"] is True
        replay = coordinator.synchronize_authoritative_task(
            task_cid="task:dep",
            task_id="DEP",
            authoritative_status="completed",
            authoritative_revision=1,
            authoritative_ready=False,
            authoritative_completed=True,
            now_ms=1_000_100,
        )
        assert replay["changed"] is False
        advanced = coordinator.synchronize_authoritative_task(
            task_cid="task:dep",
            task_id="DEP",
            authoritative_status="completed",
            authoritative_revision=2,
            authoritative_ready=False,
            authoritative_completed=True,
            now_ms=1_000_150,
        )
        assert advanced["changed"] is True
        dependency_completion = next(
            completion
            for completion in coordinator.coordination_registry_projection()[
                "logical_completions"
            ]
            if completion["task_cid"] == "task:dep"
        )
        assert dependency_completion["body"]["authoritative_revision"] == 2
        for invalid_revision in (True, 2.0, "2"):
            with pytest.raises(DatabaseCoordinationBoundsError, match="positive integer"):
                coordinator.synchronize_authoritative_task(
                    task_cid="task:invalid-revision",
                    task_id="INVALID-REVISION",
                    authoritative_status="ready",
                    authoritative_revision=invalid_revision,  # type: ignore[arg-type]
                    authoritative_ready=True,
                    authoritative_completed=False,
                )

        with pytest.raises(
            DatabaseCoordinationConflictError,
            match="typed strict-requeue ready status",
        ):
            coordinator.synchronize_authoritative_task(
                task_cid="task:ready-floor-unsealed",
                task_id="READY-FLOOR-UNSEALED",
                authoritative_status="ready",
                authoritative_revision=2,
                authoritative_ready=True,
                authoritative_completed=False,
                authoritative_attempt_floor=3,
            )
        with pytest.raises(
            DatabaseCoordinationConflictError,
            match="source requires its exact ready floor",
        ):
            coordinator.synchronize_authoritative_task(
                task_cid="task:ready-floor-empty",
                task_id="READY-FLOOR-EMPTY",
                authoritative_status="ready",
                authoritative_revision=2,
                authoritative_ready=True,
                authoritative_completed=False,
                authoritative_attempt_floor_source=(
                    TYPED_STRICT_REQUEUE_ATTEMPT_FLOOR_SOURCE
                ),
            )
        ready_floor = coordinator.synchronize_authoritative_task(
            task_cid="task:ready-floor",
            task_id="READY-FLOOR",
            authoritative_status="ready",
            authoritative_revision=2,
            authoritative_ready=True,
            authoritative_completed=False,
            authoritative_attempt_floor=3,
            authoritative_attempt_floor_source=(
                TYPED_STRICT_REQUEUE_ATTEMPT_FLOOR_SOURCE
            ),
            now_ms=1_000_175,
        )
        assert ready_floor["authoritative_attempt_floor"] == 3
        floor_claim = coordinator.claim_task(
            task_cid="task:ready-floor",
            owner_session_id="session:ready-floor",
            now_ms=1_000_176,
        )
        assert floor_claim.attempt_number == 4

        coordinator.synchronize_authoritative_task(
            task_cid="task:child",
            task_id="CHILD",
            dependency_task_cids=("task:dep",),
            authoritative_status="ready",
            authoritative_revision=1,
            authoritative_ready=True,
            authoritative_completed=False,
            now_ms=1_000_200,
        )
        assert coordinator.claimability("task:child")["claimable"] is True

        with pytest.raises(DatabaseCoordinationConflictError, match="dependency drift"):
            coordinator.synchronize_authoritative_task(
                task_cid="task:child",
                task_id="CHILD",
                dependency_task_cids=(),
                authoritative_status="ready",
                authoritative_revision=1,
                authoritative_ready=True,
                authoritative_completed=False,
            )
        with pytest.raises(DatabaseCoordinationConflictError, match="contradicts"):
            coordinator.synchronize_authoritative_task(
                task_cid="task:dep",
                task_id="DEP",
                authoritative_status="completed",
                authoritative_revision=2,
                authoritative_ready=False,
                authoritative_completed=False,
            )
        with pytest.raises(DatabaseCoordinationConflictError, match="same-revision"):
            coordinator.synchronize_authoritative_task(
                task_cid="task:dep",
                task_id="DEP",
                authoritative_status="ready",
                authoritative_revision=2,
                authoritative_ready=True,
                authoritative_completed=False,
            )
        with pytest.raises(DatabaseCoordinationConflictError, match="revision regression"):
            coordinator.synchronize_authoritative_task(
                task_cid="task:dep",
                task_id="DEP",
                authoritative_status="completed",
                authoritative_revision=1,
                authoritative_ready=False,
                authoritative_completed=True,
            )
        with pytest.raises(DatabaseCoordinationConflictError, match="no prior local claim"):
            coordinator.synchronize_authoritative_task(
                task_cid="task:fresh-retry",
                task_id="FRESH-RETRY",
                authoritative_status="in_progress",
                authoritative_revision=2,
                authoritative_ready=False,
                authoritative_completed=False,
                restart_recovery_ready=True,
                restart_recovery_owner_session_id="session:forged-retry",
                restart_recovery_binding={
                    "claim_id": "claim:forged",
                    "attempt_id": "attempt:forged",
                    "lease_id": "lease:forged",
                    "owner_session_id": "session:forged-retry",
                    "attempt_number": 1,
                    "fencing_token": 1,
                    "fence_epoch": 1,
                },
            )

        coordinator.synchronize_authoritative_task(
            task_cid="task:recovery",
            task_id="RECOVERY",
            authoritative_status="ready",
            authoritative_revision=1,
            authoritative_ready=True,
            authoritative_completed=False,
            now_ms=1_000_000,
        )
        recovery_claim = coordinator.claim_task(
            task_cid="task:recovery",
            owner_session_id="session:recovery",
            lease_ms=5_000,
            now_ms=1_000_000,
        )
        recovery_binding = {
            "claim_id": recovery_claim.claim_id,
            "attempt_id": recovery_claim.attempt_id,
            "lease_id": recovery_claim.lease_id,
            "owner_session_id": recovery_claim.owner_session_id,
            "attempt_number": recovery_claim.attempt_number,
            "fencing_token": recovery_claim.fencing_token,
            "fence_epoch": recovery_claim.fence_epoch,
        }
        expired = coordinator.synchronize_authoritative_task(
            task_cid="task:recovery",
            task_id="RECOVERY",
            authoritative_status="in_progress",
            authoritative_revision=2,
            authoritative_ready=False,
            authoritative_completed=False,
            restart_recovery_ready=True,
            restart_recovery_owner_session_id="session:recovery",
            restart_recovery_binding=recovery_binding,
            now_ms=1_006_000,
        )
        assert expired["changed"] is True
        assert expired["ready"] is True
        assert expired["active_claim_preserved"] is False
        expired_replay = coordinator.synchronize_authoritative_task(
            task_cid="task:recovery",
            task_id="RECOVERY",
            authoritative_status="in_progress",
            authoritative_revision=2,
            authoritative_ready=False,
            authoritative_completed=False,
            restart_recovery_ready=True,
            restart_recovery_owner_session_id="session:recovery",
            restart_recovery_binding=recovery_binding,
            now_ms=1_006_100,
        )
        assert expired_replay["changed"] is False
        assert (
            coordinator.claim_ready_task(
                owner_session_id="session:foreign-recovery",
                exclude_task_cids=("task:child",),
                now_ms=1_006_200,
            )
            is None
        )
        later_active = coordinator.claim_ready_task(
            owner_session_id="session:recovery",
            exclude_task_cids=("task:child",),
            now_ms=1_006_200,
        )
        assert later_active is not None
        assert later_active.attempt_number == 2
        with pytest.raises(DatabaseCoordinationConflictError, match="later active"):
            coordinator.synchronize_authoritative_task(
                task_cid="task:recovery",
                task_id="RECOVERY",
                authoritative_status="in_progress",
                authoritative_revision=2,
                authoritative_ready=False,
                authoritative_completed=False,
                restart_recovery_ready=True,
                restart_recovery_owner_session_id="session:recovery",
                restart_recovery_binding=recovery_binding,
                now_ms=1_006_300,
            )

        reopened = coordinator.synchronize_authoritative_task(
            task_cid="task:dep",
            task_id="DEP",
            authoritative_status="ready",
            authoritative_revision=3,
            authoritative_ready=True,
            authoritative_completed=False,
            now_ms=1_000_300,
        )
        assert reopened["changed"] is True
        coordinator.synchronize_authoritative_task(
            task_cid="task:child",
            task_id="CHILD",
            dependency_task_cids=("task:dep",),
            authoritative_status="ready",
            authoritative_revision=1,
            authoritative_ready=False,
            authoritative_completed=False,
            now_ms=1_000_300,
        )
        assert coordinator.claimability("task:child")["claimable"] is False

        coordinator.synchronize_authoritative_task(
            task_cid="task:prepared",
            task_id="PREPARED",
            authoritative_status="ready",
            authoritative_revision=1,
            authoritative_ready=True,
            authoritative_completed=False,
            now_ms=1_000_400,
        )
        claim = coordinator.claim_task(
            task_cid="task:prepared",
            owner_session_id="session:prepared-sync",
            now_ms=1_000_400,
        )
        coordinator.prepare_task_completion(
            claim,
            control_expected_revision=2,
            evidence_digest="sha256:" + "a" * 64,
            now_ms=1_000_500,
        )
        preserved = coordinator.synchronize_authoritative_task(
            task_cid="task:prepared",
            task_id="PREPARED",
            authoritative_status="completed",
            authoritative_revision=2,
            authoritative_ready=False,
            authoritative_completed=True,
            now_ms=1_000_600,
        )
        assert preserved["prepared_completion_preserved"] is True
        assert preserved["ready"] is False
        assert coordinator.get_prepared_task_completion("task:prepared")["status"] == (
            "prepared"
        )
    finally:
        coordinator.close()


def test_authoritative_task_sync_rebuilds_checkpointed_registry_without_update(
    tmp_path: Path,
) -> None:
    import duckdb  # type: ignore

    coordinator, clock = _open(tmp_path)
    database_path = coordinator.database_path
    try:
        connection = coordinator._require()  # noqa: SLF001
        connection.execute(
            """
            INSERT INTO coordination_tasks(
                task_cid, task_id, worktree_id, registered_at_ms,
                ready, body_json
            )
            SELECT 'task:sync-bulk-' || LPAD(CAST(i AS VARCHAR), 4, '0'),
                   'SYNC-BULK-' || LPAD(CAST(i AS VARCHAR), 4, '0'),
                   '', 900000 + i, TRUE, '{}'
            FROM range(256) AS generated(i)
            """
        )
        coordinator._commit_if_idle(connection)  # noqa: SLF001
        coordinator.synchronize_authoritative_task(
            task_cid="task:sync-persisted",
            task_id="SYNC-PERSISTED",
            authoritative_status="pending",
            authoritative_revision=1,
            authoritative_ready=False,
            authoritative_completed=False,
        )
    finally:
        coordinator.close()

    checkpoint = duckdb.connect(str(database_path))
    try:
        checkpoint.execute("CHECKPOINT")
        assert checkpoint.execute(
            "SELECT COUNT(*) FROM coordination_tasks"
        ).fetchone()[0] == 257
    finally:
        checkpoint.close()

    coordinator = open_database_coordinator(database_path, clock_ms=clock)
    raw_connection = coordinator._require()  # noqa: SLF001

    class RejectCoordinationTaskUpdate:
        def __init__(self, raw: object) -> None:
            self.raw = raw

        @property
        def in_transaction(self) -> bool:
            return bool(self.raw.in_transaction)  # type: ignore[attr-defined]

        def execute(
            self,
            statement: str,
            parameters: object | None = None,
        ) -> object:
            normalized = " ".join(str(statement).upper().split())
            if normalized.startswith("UPDATE COORDINATION_TASKS"):
                raise AssertionError(
                    "persisted coordination_tasks rows must not be updated"
                )
            if parameters is None:
                return self.raw.execute(statement)  # type: ignore[attr-defined]
            return self.raw.execute(  # type: ignore[attr-defined]
                statement,
                parameters,
            )

        def __getattr__(self, name: str) -> object:
            return getattr(self.raw, name)

    coordinator._connection = RejectCoordinationTaskUpdate(  # noqa: SLF001
        raw_connection
    )
    try:
        synchronized = coordinator.synchronize_authoritative_task(
            task_cid="task:sync-persisted",
            task_id="SYNC-PERSISTED",
            authoritative_status="pending",
            authoritative_revision=2,
            authoritative_ready=False,
            authoritative_completed=False,
        )
        assert synchronized["changed"] is True
        assert synchronized["ready"] is False
    finally:
        coordinator.close()

    landed = duckdb.connect(str(database_path))
    try:
        landed.execute("CHECKPOINT")
        assert landed.execute(
            "SELECT COUNT(*), COUNT(DISTINCT task_cid) FROM coordination_tasks"
        ).fetchone() == (257, 257)
        ready, body_json = landed.execute(
            "SELECT ready, body_json FROM coordination_tasks WHERE task_cid = ?",
            ["task:sync-persisted"],
        ).fetchone()
        assert ready is False
        assert json.loads(body_json) == {
            "authority": "task_source",
            "authoritative_attempt_floor": 0,
            "authoritative_attempt_floor_source": "",
            "authoritative_revision": 2,
            "authoritative_status": "pending",
            "restart_recovery_binding": {},
            "restart_recovery_owner_session_id": "",
            "restart_recovery_ready": False,
        }
        assert landed.execute(
            """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_name = 'coordination_tasks_rearm_staging'
            """
        ).fetchone()[0] == 0
        assert landed.execute(
            """
            SELECT COUNT(*) FROM duckdb_indexes()
            WHERE schema_name = 'main'
              AND index_name = 'coordination_tasks_ready_idx'
              AND table_name = 'coordination_tasks'
            """
        ).fetchone()[0] == 1
    finally:
        landed.close()


@pytest.mark.parametrize("failure_point", ["begin", "commit"])
def test_authoritative_task_sync_transaction_boundaries_fail_closed(
    tmp_path: Path,
    failure_point: str,
) -> None:
    coordinator, _clock = _open(tmp_path)
    coordinator.synchronize_authoritative_task(
        task_cid="task:sync-transaction",
        task_id="SYNC-TRANSACTION",
        authoritative_status="pending",
        authoritative_revision=1,
        authoritative_ready=False,
        authoritative_completed=False,
    )
    raw_connection = coordinator._require()  # noqa: SLF001

    class TransactionFault:
        def __init__(self, raw: object) -> None:
            self.raw = raw

        @property
        def in_transaction(self) -> bool:
            return bool(self.raw.in_transaction)  # type: ignore[attr-defined]

        def execute(
            self,
            statement: str,
            parameters: object | None = None,
        ) -> object:
            normalized = " ".join(str(statement).upper().split())
            if failure_point == "begin" and normalized == "BEGIN TRANSACTION":
                return self.raw.execute("SELECT 1")  # type: ignore[attr-defined]
            if failure_point == "commit" and normalized == "COMMIT":
                raise RuntimeError("injected authoritative-sync commit failure")
            if parameters is None:
                return self.raw.execute(statement)  # type: ignore[attr-defined]
            return self.raw.execute(  # type: ignore[attr-defined]
                statement,
                parameters,
            )

        def __getattr__(self, name: str) -> object:
            return getattr(self.raw, name)

    coordinator._connection = TransactionFault(raw_connection)  # noqa: SLF001
    expected_error = (
        DatabaseCoordinationError if failure_point == "begin" else RuntimeError
    )
    try:
        with pytest.raises(expected_error):
            coordinator.synchronize_authoritative_task(
                task_cid="task:sync-transaction",
                task_id="SYNC-TRANSACTION",
                authoritative_status="pending",
                authoritative_revision=2,
                authoritative_ready=False,
                authoritative_completed=False,
            )
        assert coordinator._require().in_transaction is False  # noqa: SLF001
        task_row = coordinator._require().execute(  # noqa: SLF001
            "SELECT ready, body_json FROM coordination_tasks WHERE task_cid = ?",
            ["task:sync-transaction"],
        ).fetchone()
        assert task_row is not None
        assert task_row[0] is False
        assert json.loads(task_row[1])["authoritative_revision"] == 1
        assert coordinator._require().execute(  # noqa: SLF001
            """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_name = 'coordination_tasks_rearm_staging'
            """
        ).fetchone()[0] == 0
    finally:
        coordinator.close()


def test_coordination_registry_projection_exposes_exact_claim_and_lease_counts(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:work", task_id="WORK")
        coordinator.claim_task(
            task_cid="task:work",
            owner_session_id="session:worker",
        )
        coordinator.claim_resource(
            resource_kind="gpu",
            resource_id="gpu:0",
            owner_session_id="session:worker",
            task_cid="task:work",
        )

        projection = coordinator.coordination_registry_projection()
        assert projection["counts"] == {
            "registered_tasks": 1,
            "dependency_edges": 0,
            "logical_completions": 0,
            "task_claims": 1,
            "active_task_claims": 1,
            "resource_claims": 1,
            "active_resource_claims": 1,
            "task_attempts": 1,
            "active_task_attempts": 1,
            "fenced_leases": 2,
            "active_fenced_leases": 2,
            "maintenance_leases": 0,
            "active_maintenance_leases": 0,
        }
        assert projection["task_claim_state_counts"] == [
            {"state": "accepted", "count": 1}
        ]
        assert projection["resource_claim_state_counts"] == [
            {"state": "accepted", "count": 1}
        ]
        assert projection["task_attempt_status_counts"] == [
            {"status": "running", "count": 1}
        ]
        assert projection["fenced_lease_kind_state_counts"] == [
            {"lease_kind": "resource", "state": "accepted", "count": 1},
            {"lease_kind": "task", "state": "accepted", "count": 1},
        ]
    finally:
        coordinator.close()


def test_coordination_registry_projection_makes_dependency_tamper_visible(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:dep", task_id="DEP")
        coordinator.register_task(
            task_cid="task:child",
            task_id="CHILD",
            dependency_task_cids=("task:dep",),
        )
        before = coordinator.coordination_registry_projection()

        # Simulate an out-of-band database writer changing the registry edge.
        connection = coordinator._require()
        connection.execute(
            """
            UPDATE task_dependencies
            SET dependency_task_cid = 'task:forged'
            WHERE task_cid = 'task:child'
            """
        )
        coordinator._commit_if_idle(connection)

        after = coordinator.coordination_registry_projection()
        assert after["dependency_edges"] == [
            {
                "task_cid": "task:child",
                "dependency_task_cid": "task:forged",
            }
        ]
        assert after["projection_root"] != before["projection_root"]
    finally:
        coordinator.close()


def test_add_unstarted_task_dependency_is_exact_and_identity_preserving(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:old-dep", task_id="OLD-DEP")
        coordinator.register_task(task_cid="task:new-dep", task_id="NEW-DEP")
        coordinator.register_task(
            task_cid="task:future",
            task_id="FUTURE",
            worktree_id="worktree:future",
            dependency_task_cids=("task:old-dep",),
            body={"logical_task_cid": "task:future", "ordinal": 120},
        )
        before = coordinator.coordination_registry_projection()
        before_task = next(
            item for item in before["tasks"] if item["task_cid"] == "task:future"
        )

        receipt = coordinator.add_unstarted_task_dependency(
            task_cid="task:future",
            dependency_task_cid="task:new-dep",
            expected_dependency_task_cids=("task:old-dep",),
            operation_id="plan-revision:r2:120-requires-113",
        )

        assert receipt["schema"] == TASK_DEPENDENCY_AMENDMENT_SCHEMA
        assert receipt["changed"] is True
        assert receipt["before_dependency_task_cids"] == ["task:old-dep"]
        assert receipt["after_dependency_task_cids"] == [
            "task:new-dep",
            "task:old-dep",
        ]
        assert receipt["receipt_cid"].startswith("sha256:")
        after = coordinator.coordination_registry_projection()
        assert next(
            item for item in after["tasks"] if item["task_cid"] == "task:future"
        ) == before_task
        assert after["logical_completions"] == before["logical_completions"]
        assert {
            (item["task_cid"], item["dependency_task_cid"])
            for item in after["dependency_edges"]
        } - {
            (item["task_cid"], item["dependency_task_cid"])
            for item in before["dependency_edges"]
        } == {("task:future", "task:new-dep")}

        replay = coordinator.add_unstarted_task_dependency(
            task_cid="task:future",
            dependency_task_cid="task:new-dep",
            expected_dependency_task_cids=("task:old-dep",),
            operation_id="plan-revision:r2:120-requires-113",
        )
        assert replay["changed"] is False
        assert coordinator.coordination_registry_projection() == after
    finally:
        coordinator.close()


def test_add_unstarted_task_dependency_rejects_stale_cas_and_missing_target(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:old-dep", task_id="OLD-DEP")
        coordinator.register_task(task_cid="task:new-dep", task_id="NEW-DEP")
        coordinator.register_task(
            task_cid="task:future",
            task_id="FUTURE",
            dependency_task_cids=("task:old-dep",),
        )
        before = coordinator.coordination_registry_projection()

        with pytest.raises(
            DatabaseCoordinationConflictError,
            match="compare-and-swap failed",
        ):
            coordinator.add_unstarted_task_dependency(
                task_cid="task:future",
                dependency_task_cid="task:new-dep",
                expected_dependency_task_cids=(),
                operation_id="stale-plan-revision",
            )
        with pytest.raises(
            DatabaseCoordinationConflictError,
            match="dependency task is absent",
        ):
            coordinator.add_unstarted_task_dependency(
                task_cid="task:future",
                dependency_task_cid="task:not-registered",
                expected_dependency_task_cids=("task:old-dep",),
                operation_id="missing-dependency",
            )
        assert coordinator.coordination_registry_projection() == before
    finally:
        coordinator.close()


@pytest.mark.parametrize("history_kind", ["completion", "claim"])
def test_add_unstarted_task_dependency_rejects_any_execution_history(
    tmp_path: Path,
    history_kind: str,
) -> None:
    coordinator, _clock = _open(tmp_path / history_kind)
    try:
        coordinator.register_task(task_cid="task:new-dep", task_id="NEW-DEP")
        coordinator.register_task(task_cid="task:started", task_id="STARTED")
        if history_kind == "completion":
            coordinator.mark_task_complete("task:started")
        else:
            coordinator.claim_task(
                task_cid="task:started",
                owner_session_id="session:worker",
            )
        before = coordinator.coordination_registry_projection()

        with pytest.raises(
            DatabaseCoordinationConflictError,
            match="requires an unstarted task",
        ):
            coordinator.add_unstarted_task_dependency(
                task_cid="task:started",
                dependency_task_cid="task:new-dep",
                expected_dependency_task_cids=(),
                operation_id=f"reject-{history_kind}",
            )
        assert coordinator.coordination_registry_projection() == before
    finally:
        coordinator.close()


def test_read_only_projection_preserves_database_bytes_and_exposes_histories(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    database_path = coordinator.database_path
    try:
        coordinator.register_task(task_cid="task:expected", task_id="EXPECTED")
        coordinator.register_task(task_cid="task:foreign", task_id="FOREIGN")
        task_claim = coordinator.claim_task(
            task_cid="task:foreign",
            owner_session_id="session:foreign",
            worktree_id="worktree:foreign",
            idempotency_key="foreign-task-attempt",
            body={"result_identity": "sha256:foreign"},
        )
        resource_claim = coordinator.claim_resource(
            resource_kind="database_writer",
            resource_id="writer:foreign",
            owner_session_id="session:foreign",
            task_cid="task:foreign",
            repository_id="repo:foreign",
            body={"permit": "foreign"},
        )
        maintenance = coordinator.acquire_maintenance_lease(
            owner_session_id="session:maintenance",
            scope="foreign-maintenance",
            process_birth_id="process:foreign",
            body={"reason": "foreign"},
        )
        coordinator.release(task_claim.as_fenced_lease(), reason="terminal foreign")
        coordinator.release(resource_claim.as_fenced_lease(), reason="terminal foreign")
        coordinator.release(maintenance.as_fenced_lease(), reason="terminal foreign")
    finally:
        coordinator.close()

    before_bytes = database_path.read_bytes()
    before_digest = hashlib.sha256(before_bytes).hexdigest()
    before_entries = sorted(path.name for path in database_path.parent.iterdir())

    projection = read_coordination_registry_projection(database_path)

    assert hashlib.sha256(database_path.read_bytes()).hexdigest() == before_digest
    assert database_path.read_bytes() == before_bytes
    assert sorted(path.name for path in database_path.parent.iterdir()) == before_entries
    assert projection["tasks"][1]["task_id"] == "FOREIGN"
    assert projection["task_claims"] == [
        {
            "claim_id": task_claim.claim_id,
            "task_cid": "task:foreign",
            "owner_session_id": "session:foreign",
            "fencing_token": task_claim.fencing_token,
            "fence_epoch": task_claim.fence_epoch,
            "state": "released",
            "revision": 2,
            "attempt_id": task_claim.attempt_id,
            "attempt_number": 1,
            "lease_id": task_claim.lease_id,
            "worktree_id": "worktree:foreign",
            "idempotency_key": "foreign-task-attempt",
            "body": {"result_identity": "sha256:foreign"},
        }
    ]
    assert projection["task_attempts"] == [
        {
            "attempt_id": task_claim.attempt_id,
            "task_cid": "task:foreign",
            "attempt_number": 1,
            "owner_session_id": "session:foreign",
            "fencing_token": task_claim.fencing_token,
            "fence_epoch": task_claim.fence_epoch,
            "status": "released",
            "revision": 2,
        }
    ]
    assert {item["lease_id"] for item in projection["fenced_leases"]} == {
        task_claim.lease_id,
        resource_claim.lease_id,
        maintenance.lease_id,
    }
    assert projection["resource_claims"] == [
        {
            "claim_id": resource_claim.claim_id,
            "resource_kind": "database_writer",
            "resource_id": "writer:foreign",
            "owner_session_id": "session:foreign",
            "fencing_token": resource_claim.fencing_token,
            "fence_epoch": resource_claim.fence_epoch,
            "state": "released",
            "revision": 2,
            "lease_id": resource_claim.lease_id,
            "task_cid": "task:foreign",
            "repository_id": "repo:foreign",
            "path": "",
            "worktree_id": "",
            "mode": "exclusive",
            "body": {"permit": "foreign"},
        }
    ]
    assert projection["maintenance_leases"] == [
        {
            "lease_id": maintenance.lease_id,
            "scope": "foreign-maintenance",
            "owner_session_id": "session:maintenance",
            "process_birth_id": "process:foreign",
            "fencing_token": maintenance.fencing_token,
            "fence_epoch": maintenance.fence_epoch,
            "state": "released",
            "revision": 2,
            "body": {"reason": "foreign"},
        }
    ]


def test_coordination_history_projection_is_closed_deterministic_and_read_only(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    database_path = coordinator.database_path
    try:
        lease = coordinator.acquire(
            lease_kind="merge",
            scope="history:exact",
            owner_session_id="session:history",
            lease_ms=30_000,
            body={"reason": "history-projection"},
        )
        coordinator.release(lease, reason="history-complete")
    finally:
        coordinator.close()

    before = (
        database_path.stat().st_size,
        database_path.stat().st_mtime_ns,
        hashlib.sha256(database_path.read_bytes()).hexdigest(),
    )
    first = read_coordination_history_projection(database_path)
    second = read_coordination_history_projection(database_path)

    assert first == second
    assert first["schema"] == COORDINATION_HISTORY_PROJECTION_SCHEMA
    assert first["counts"] == {"token_history": 1, "lease_events": 2}
    assert set(first) == {
        "schema",
        "authority_schema",
        "schema_inventory",
        "token_history",
        "lease_events",
        "counts",
        "projection_root",
    }
    material = dict(first)
    claimed = material.pop("projection_root")
    encoded = json.dumps(
        material, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert claimed == "sha256:" + hashlib.sha256(encoded).hexdigest()
    assert before == (
        database_path.stat().st_size,
        database_path.stat().st_mtime_ns,
        hashlib.sha256(database_path.read_bytes()).hexdigest(),
    )

    import duckdb  # type: ignore

    connection = duckdb.connect(str(database_path))
    try:
        connection.execute("CREATE TABLE forged_hidden_authority(value VARCHAR)")
    finally:
        connection.close()
    forged_before = hashlib.sha256(database_path.read_bytes()).hexdigest()
    with pytest.raises(
        DatabaseCoordinationStaleFenceError,
        match="table inventory differs",
    ):
        read_coordination_history_projection(database_path)
    assert hashlib.sha256(database_path.read_bytes()).hexdigest() == forged_before


def test_coordination_projection_rejects_duplicate_json_authority(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    database_path = coordinator.database_path
    try:
        coordinator.register_task(
            task_cid="task:duplicate-json",
            task_id="DUP-001",
            body={"task_alias": "DUP-001", "status": "todo"},
        )
        coordinator._require().execute(  # noqa: SLF001
            "UPDATE coordination_tasks SET body_json = ? WHERE task_cid = ?",
            [
                '{"task_alias":"EVIL","task_alias":"DUP-001","status":"todo"}',
                "task:duplicate-json",
            ],
        )
    finally:
        coordinator.close()

    with pytest.raises(
        DatabaseCoordinationStaleFenceError,
        match="unambiguous JSON",
    ):
        read_coordination_registry_projection(database_path)

    import duckdb  # type: ignore

    connection = duckdb.connect(str(database_path))
    try:
        connection.execute(
            "UPDATE coordination_tasks SET body_json = '' WHERE task_cid = ?",
            ["task:duplicate-json"],
        )
    finally:
        connection.close()
    with pytest.raises(
        DatabaseCoordinationStaleFenceError,
        match="unambiguous JSON",
    ):
        read_coordination_registry_projection(database_path)


@pytest.mark.parametrize("tamper", ["metadata", "schema"])
def test_read_only_projection_fails_closed_without_repairing_authority(
    tmp_path: Path,
    tamper: str,
) -> None:
    coordinator, _clock = _open(tmp_path)
    database_path = coordinator.database_path
    connection = coordinator._require()
    if tamper == "metadata":
        connection.execute(
            "UPDATE coordination_metadata SET value = 'forged' WHERE key = 'schema'"
        )
    else:
        connection.execute("DROP INDEX task_claims_task_idx")
    coordinator._commit_if_idle(connection)
    coordinator.close()

    before_bytes = database_path.read_bytes()
    before_digest = hashlib.sha256(before_bytes).hexdigest()
    with pytest.raises(DatabaseCoordinationStaleFenceError, match="coordination authority"):
        read_coordination_registry_projection(database_path)
    assert hashlib.sha256(database_path.read_bytes()).hexdigest() == before_digest
    assert database_path.read_bytes() == before_bytes


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


def test_claim_ready_honors_exact_eligible_order_and_boundary(tmp_path: Path) -> None:
    coordinator, clock = _open(tmp_path)
    try:
        coordinator.register_task(
            task_cid="task:oldest", task_id="OLDEST", now_ms=clock.now
        )
        coordinator.register_task(
            task_cid="task:preferred", task_id="PREFERRED", now_ms=clock.now + 10
        )
        coordinator.register_task(
            task_cid="task:excluded-by-boundary",
            task_id="EXCLUDED",
            now_ms=clock.now - 10,
        )

        claim = coordinator.claim_ready_task(
            owner_session_id="session:ordered",
            eligible_task_cids=("task:preferred", "task:oldest"),
        )
        assert claim is not None
        assert claim.task_cid == "task:preferred"

        # An explicit empty eligibility projection is authoritative.
        assert (
            coordinator.claim_ready_task(
                owner_session_id="session:empty", eligible_task_cids=()
            )
            is None
        )
        with pytest.raises(DatabaseCoordinationError, match="absent"):
            coordinator.claim_ready_task(
                owner_session_id="session:unknown",
                eligible_task_cids=("task:not-registered",),
            )
    finally:
        coordinator.close()


def test_claim_ready_respects_accept_task_cid_before_taking_scope(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:skip", task_id="SKIP")
        coordinator.register_task(task_cid="task:keep", task_id="KEEP")
        claim = coordinator.claim_ready_task(
            owner_session_id="session:shard",
            accept_task_cid=lambda cid: cid == "task:keep",
        )
        assert claim is not None
        assert claim.task_cid == "task:keep"
        skipped = coordinator.claimability("task:skip")
        assert skipped["claimable"] is True
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


def test_control_rearm_removes_exact_completion_and_is_replay_safe(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:rearm", task_id="REARM")
        first_claim, _control_task = _settled_control_completion(
            coordinator,
            "task:rearm",
        )
        observation = _control_rearm_observation(
            task_cid="task:rearm",
            task_alias="REARM",
            nested=True,
        )

        rearmed = coordinator.rearm_task_from_control(
            "task:rearm",
            control_task_observation=observation,
        )
        assert rearmed == {
            "schema": TASK_COMPLETION_REARM_SCHEMA,
            "task_cid": "task:rearm",
            "previous_control_revision": 3,
            "control_revision": 4,
            "control_status": "retrying",
            "ready": True,
            "replayed": False,
        }
        assert coordinator.claimability("task:rearm")["claimable"] is True
        projection = coordinator.coordination_registry_projection()
        assert projection["logical_completions"] == []
        assert projection["tasks"][0]["ready"] is True

        # Response-loss replay requires the same exact typed CAS receipt.
        replay = coordinator.rearm_task_from_control(
            "task:rearm",
            control_task_observation=observation,
        )
        assert replay == {**rearmed, "replayed": True}

        replacement = coordinator.claim_task(
            task_cid="task:rearm",
            owner_session_id="session:replacement",
        )
        assert replacement.attempt_number == first_claim.attempt_number + 1
        assert replacement.fencing_token > first_claim.fencing_token
    finally:
        coordinator.close()


def test_control_ready_frontier_reconciliation_demotes_with_art_safe_replay(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    coordinator.register_task(
        task_cid="task:frontier-stale",
        task_id="FRONTIER-STALE",
        body={"preserve": "exactly"},
    )
    raw_connection = coordinator._require()  # noqa: SLF001

    class RejectCoordinationTaskUpdate:
        def __init__(self, raw: object) -> None:
            self.raw = raw

        @property
        def in_transaction(self) -> bool:
            return bool(self.raw.in_transaction)  # type: ignore[attr-defined]

        def execute(
            self,
            statement: str,
            parameters: object | None = None,
        ) -> object:
            normalized = " ".join(str(statement).upper().split())
            if normalized.startswith("UPDATE COORDINATION_TASKS"):
                raise AssertionError(
                    "ready-frontier reconciliation must rebuild the ART table"
                )
            if parameters is None:
                return self.raw.execute(statement)  # type: ignore[attr-defined]
            return self.raw.execute(  # type: ignore[attr-defined]
                statement,
                parameters,
            )

        def __getattr__(self, name: str) -> object:
            return getattr(self.raw, name)

    coordinator._connection = RejectCoordinationTaskUpdate(  # noqa: SLF001
        raw_connection
    )
    terminal = _control_task_projection(
        task_cid="task:frontier-stale",
        task_alias="FRONTIER-STALE",
        status="completed",
        revision=14,
    )
    inventory = _control_ready_frontier(terminal, revision=77)
    frontier = _control_ready_frontier(revision=77)
    try:
        reconciled = coordinator.reconcile_task_from_control_ready_frontier(
            "task:frontier-stale",
            control_task_inventory_observation=inventory,
            control_ready_frontier_observation=frontier,
            owner_session_id="session:frontier-reconciler",
        )
        assert reconciled == {
            "schema": CONTROL_READY_FRONTIER_RECONCILIATION_SCHEMA,
            "task_cid": "task:frontier-stale",
            "task_alias": "FRONTIER-STALE",
            "control_task_status": "completed",
            "control_task_revision": 14,
            "control_snapshot_revision": 77,
            "direction": "demote",
            "ready_before": True,
            "ready_after": False,
            "changed": True,
            "receipt_cid": reconciled["receipt_cid"],
            "replayed": False,
        }
        assert str(reconciled["receipt_cid"]).startswith("sha256:")
        projection = coordinator.coordination_registry_projection()
        assert projection["tasks"] == [
            {
                "task_cid": "task:frontier-stale",
                "task_id": "FRONTIER-STALE",
                "worktree_id": "",
                "ready": False,
                "body": {"preserve": "exactly"},
            }
        ]
        events = coordinator.lease_events(limit=20)
        reconciliation_events = [
            event
            for event in events
            if event["event_type"]
            == CONTROL_READY_FRONTIER_RECONCILIATION_EVENT
        ]
        assert len(reconciliation_events) == 1
        assert reconciliation_events[0]["body"]["receipt_cid"] == reconciled[
            "receipt_cid"
        ]
        assert {
            event["event_type"] for event in events
        } == {
            "acquired",
            "released",
            CONTROL_READY_FRONTIER_RECONCILIATION_EVENT,
        }

        replay = coordinator.reconcile_task_from_control_ready_frontier(
            "task:frontier-stale",
            control_task_inventory_observation=inventory,
            control_ready_frontier_observation=frontier,
            owner_session_id="session:frontier-reconciler",
        )
        assert replay == {**reconciled, "replayed": True}
        assert len(
            [
                event
                for event in coordinator.lease_events(limit=20)
                if event["event_type"]
                == CONTROL_READY_FRONTIER_RECONCILIATION_EVENT
            ]
        ) == 1

        reopened = {
            **terminal,
            "status": "retrying",
            "revision": 15,
        }
        reopened_inventory = _control_ready_frontier(reopened, revision=78)
        reopened_frontier = _control_ready_frontier(reopened, revision=78)
        promoted = coordinator.reconcile_task_from_control_ready_frontier(
            "task:frontier-stale",
            control_task_inventory_observation=reopened_inventory,
            control_ready_frontier_observation=reopened_frontier,
            owner_session_id="session:frontier-reconciler",
        )
        assert promoted["direction"] == "promote"
        assert promoted["ready_before"] is False
        assert promoted["ready_after"] is True
        assert promoted["changed"] is True
        assert coordinator.claimability("task:frontier-stale")["claimable"] is True
    finally:
        coordinator.close()


@pytest.mark.parametrize(
    "tampered_projection",
    ("fenced_lease_body", "maintenance_process_birth"),
)
def test_control_ready_frontier_replay_rejects_tampered_maintenance_projection(
    tmp_path: Path,
    tampered_projection: str,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        task_cid = "task:frontier-tampered-replay"
        task_alias = "FRONTIER-TAMPERED-REPLAY"
        coordinator.register_task(task_cid=task_cid, task_id=task_alias)
        terminal = _control_task_projection(
            task_cid=task_cid,
            task_alias=task_alias,
            status="completed",
            revision=14,
        )
        inventory = _control_ready_frontier(terminal, revision=77)
        frontier = _control_ready_frontier(revision=77)
        reconciled = coordinator.reconcile_task_from_control_ready_frontier(
            task_cid,
            control_task_inventory_observation=inventory,
            control_ready_frontier_observation=frontier,
            owner_session_id="session:frontier-tampered-replay",
        )
        event = next(
            item
            for item in coordinator.lease_events(limit=20)
            if item["event_type"]
            == CONTROL_READY_FRONTIER_RECONCILIATION_EVENT
        )
        lease_id = str(event["lease_id"])
        connection = coordinator._require()  # noqa: SLF001
        if tampered_projection == "fenced_lease_body":
            connection.execute(
                "UPDATE fenced_leases SET body_json = ? WHERE lease_id = ?",
                [json.dumps({"tampered": True}), lease_id],
            )
        else:
            connection.execute(
                "UPDATE maintenance_leases SET process_birth_id = ? "
                "WHERE lease_id = ?",
                ["birth:forged", lease_id],
            )

        with pytest.raises(
            DatabaseCoordinationStaleFenceError,
            match="durable receipt: maintenance_lease",
        ):
            coordinator.reconcile_task_from_control_ready_frontier(
                task_cid,
                control_task_inventory_observation=inventory,
                control_ready_frontier_observation=frontier,
                owner_session_id="session:frontier-tampered-replay",
            )
        ready_row = connection.execute(
            "SELECT ready FROM coordination_tasks WHERE task_cid = ?",
            [task_cid],
        ).fetchone()
        assert ready_row is not None and bool(ready_row[0]) is False
        assert reconciled["ready_after"] is False
    finally:
        coordinator.close()


def test_control_ready_frontier_reconciliation_preserves_promoted_history(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(
            task_cid="task:frontier-completed",
            task_id="FRONTIER-COMPLETED",
        )
        _settled_control_completion(coordinator, "task:frontier-completed")
        connection = coordinator._require()  # noqa: SLF001

        def task_history() -> tuple[list[object], list[object], list[object]]:
            return (
                list(
                    connection.execute(
                        "SELECT * FROM task_completions WHERE task_cid = ?",
                        ["task:frontier-completed"],
                    ).fetchall()
                ),
                list(
                    connection.execute(
                        "SELECT * FROM task_claims WHERE task_cid = ?",
                        ["task:frontier-completed"],
                    ).fetchall()
                ),
                list(
                    connection.execute(
                        "SELECT * FROM task_attempts WHERE task_cid = ?",
                        ["task:frontier-completed"],
                    ).fetchall()
                ),
            )

        before = task_history()
        reopened = _control_task_projection(
            task_cid="task:frontier-completed",
            task_alias="FRONTIER-COMPLETED",
            status="retrying",
            revision=14,
        )
        inventory = _control_ready_frontier(reopened, revision=81)
        frontier = _control_ready_frontier(reopened, revision=81)
        receipt = coordinator.reconcile_task_from_control_ready_frontier(
            "task:frontier-completed",
            control_task_inventory_observation=inventory,
            control_ready_frontier_observation=frontier,
            owner_session_id="session:frontier-history",
        )
        assert receipt["changed"] is False
        assert receipt["direction"] == ""
        assert receipt["ready_before"] is False
        assert receipt["ready_after"] is False
        assert receipt["receipt_cid"] == ""
        assert task_history() == before
        assert coordinator.claimability("task:frontier-completed")[
            "completion_status"
        ] == AttemptStatus.SUCCEEDED.value
        assert coordinator.reconcile_task_from_control_ready_frontier(
            "task:frontier-completed",
            control_task_inventory_observation=inventory,
            control_ready_frontier_observation=frontier,
            owner_session_id="session:frontier-history",
        ) == receipt
        assert task_history() == before
        assert all(
            event["event_type"] != CONTROL_READY_FRONTIER_RECONCILIATION_EVENT
            for event in coordinator.lease_events(limit=100)
        )
    finally:
        coordinator.close()


def test_control_ready_frontier_reconciliation_fails_closed(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(
            task_cid="task:frontier-closed",
            task_id="FRONTIER-CLOSED",
        )
        terminal = _control_task_projection(
            task_cid="task:frontier-closed",
            task_alias="FRONTIER-CLOSED",
            status="completed",
            revision=4,
        )
        ready_version = {
            **terminal,
            "status": "retrying",
        }
        invalid_pairs = [
            (
                _control_ready_frontier(
                    {**terminal, "task_alias": "WRONG"}, revision=50
                ),
                _control_ready_frontier(revision=50),
            ),
            (
                _control_ready_frontier(
                    {**terminal, "revision": "4"}, revision=50
                ),
                _control_ready_frontier(revision=50),
            ),
            (
                _control_ready_frontier(terminal, revision=50),
                _control_ready_frontier(ready_version, revision=50),
            ),
            (
                _control_ready_frontier(terminal, revision=50),
                _control_ready_frontier(
                    _control_task_projection(
                        task_cid="task:frontier-not-in-inventory",
                        task_alias="FRONTIER-NOT-IN-INVENTORY",
                        status="retrying",
                        revision=4,
                    ),
                    revision=50,
                ),
            ),
            (
                _control_ready_frontier(terminal, revision=50),
                _control_ready_frontier(revision=51),
            ),
            (
                _control_ready_frontier(terminal, revision=50),
                {
                    **_control_ready_frontier(revision=50),
                    "next_cursor": "cursor:not-complete",
                },
            ),
        ]
        for inventory_observation, frontier_observation in invalid_pairs:
            with pytest.raises(
                (DatabaseCoordinationError, DatabaseCoordinationNotReadyError)
            ):
                coordinator.reconcile_task_from_control_ready_frontier(
                    "task:frontier-closed",
                    control_task_inventory_observation=inventory_observation,
                    control_ready_frontier_observation=frontier_observation,
                    owner_session_id="session:frontier-closed",
                )
            assert coordinator.claimability("task:frontier-closed")[
                "claimable"
            ] is True

        claim = coordinator.claim_task(
            task_cid="task:frontier-closed",
            owner_session_id="session:active-frontier",
        )
        with pytest.raises(
            DatabaseCoordinationConflictError,
            match="requires a quiescent sidecar",
        ):
            coordinator.reconcile_task_from_control_ready_frontier(
                "task:frontier-closed",
                control_task_inventory_observation=_control_ready_frontier(
                    terminal, revision=50
                ),
                control_ready_frontier_observation=_control_ready_frontier(
                    revision=50
                ),
                owner_session_id="session:frontier-closed",
            )
        assert coordinator.get_task_claim(claim.claim_id).state is LeaseState.ACCEPTED
    finally:
        coordinator.close()


def test_control_rearm_recomputes_blocked_dependency_readiness(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:rearm-dep", task_id="REARM-DEP")
        coordinator.register_task(
            task_cid="task:rearm-child",
            task_id="REARM-CHILD",
            dependency_task_cids=("task:rearm-dep",),
        )
        _settled_control_completion(coordinator, "task:rearm-dep")
        _settled_control_completion(coordinator, "task:rearm-child")

        dependency = coordinator.rearm_task_from_control(
            "task:rearm-dep",
            control_task_observation=_control_rearm_observation(
                task_cid="task:rearm-dep",
                task_alias="REARM-DEP",
            ),
        )
        assert dependency["ready"] is True
        child = coordinator.rearm_task_from_control(
            "task:rearm-child",
            control_task_observation=_control_rearm_observation(
                task_cid="task:rearm-child",
                task_alias="REARM-CHILD",
            ),
        )
        assert child["ready"] is False
        readiness = coordinator.claimability("task:rearm-child")
        assert readiness["claimable"] is False
        assert readiness["blocked_dependency_task_cids"] == ["task:rearm-dep"]

        # Projection validation also proves the exact required ready index was
        # recreated after both indexed boolean updates.
        projection = coordinator.coordination_registry_projection()
        tasks = {item["task_cid"]: item for item in projection["tasks"]}
        assert tasks["task:rearm-dep"]["ready"] is True
        assert tasks["task:rearm-child"]["ready"] is False
    finally:
        coordinator.close()


def test_control_rearm_rebuilds_checkpointed_many_row_registry_without_update(
    tmp_path: Path,
) -> None:
    import duckdb  # type: ignore

    def task_schema_inventory(connection: object) -> dict[str, object]:
        columns = connection.execute(  # type: ignore[attr-defined]
            """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'main' AND table_name = 'coordination_tasks'
            ORDER BY ordinal_position
            """
        ).fetchall()
        constraints = connection.execute(  # type: ignore[attr-defined]
            """
            SELECT constraint_type, constraint_column_names, constraint_text
            FROM duckdb_constraints()
            WHERE schema_name = 'main' AND table_name = 'coordination_tasks'
            ORDER BY constraint_index
            """
        ).fetchall()
        indexes = connection.execute(  # type: ignore[attr-defined]
            """
            SELECT index_name, is_unique, is_primary, expressions
            FROM duckdb_indexes()
            WHERE schema_name = 'main' AND table_name = 'coordination_tasks'
            ORDER BY index_name
            """
        ).fetchall()
        return {
            "columns": tuple(tuple(row) for row in columns),
            "constraints": tuple(
                (row[0], tuple(row[1]), row[2]) for row in constraints
            ),
            "indexes": tuple(tuple(row) for row in indexes),
        }

    coordinator, clock = _open(tmp_path)
    database_path = coordinator.database_path
    try:
        connection = coordinator._require()  # noqa: SLF001
        connection.execute(
            """
            INSERT INTO coordination_tasks(
                task_cid, task_id, worktree_id, registered_at_ms,
                ready, body_json
            )
            SELECT 'task:bulk-' || LPAD(CAST(i AS VARCHAR), 4, '0'),
                   'BULK-' || LPAD(CAST(i AS VARCHAR), 4, '0'),
                   '', 900000 + i, TRUE, '{}'
            FROM range(512) AS generated(i)
            """
        )
        coordinator._commit_if_idle(connection)  # noqa: SLF001
        coordinator.register_task(
            task_cid="task:rearm-persisted",
            task_id="REARM-PERSISTED",
            body={"payload": "preserve-exactly"},
        )
        _settled_control_completion(coordinator, "task:rearm-persisted")
    finally:
        coordinator.close()

    checkpoint = duckdb.connect(str(database_path))
    try:
        checkpoint.execute("CHECKPOINT")
        assert checkpoint.execute(
            "SELECT COUNT(*) FROM coordination_tasks"
        ).fetchone()[0] == 513
        before_inventory = task_schema_inventory(checkpoint)
    finally:
        checkpoint.close()

    coordinator = open_database_coordinator(database_path, clock_ms=clock)
    raw_connection = coordinator._require()  # noqa: SLF001

    class RejectCoordinationTaskUpdate:
        def __init__(self, raw: object) -> None:
            self.raw = raw

        @property
        def in_transaction(self) -> bool:
            return bool(self.raw.in_transaction)  # type: ignore[attr-defined]

        def execute(
            self,
            statement: str,
            parameters: object | None = None,
        ) -> object:
            normalized = " ".join(str(statement).upper().split())
            if normalized.startswith("UPDATE COORDINATION_TASKS"):
                raise AssertionError(
                    "persisted coordination_tasks rows must not be updated"
                )
            if parameters is None:
                return self.raw.execute(statement)  # type: ignore[attr-defined]
            return self.raw.execute(  # type: ignore[attr-defined]
                statement,
                parameters,
            )

        def __getattr__(self, name: str) -> object:
            return getattr(self.raw, name)

    coordinator._connection = RejectCoordinationTaskUpdate(  # noqa: SLF001
        raw_connection
    )
    try:
        rearmed = coordinator.rearm_task_from_control(
            "task:rearm-persisted",
            control_task_observation=_control_rearm_observation(
                task_cid="task:rearm-persisted",
                task_alias="REARM-PERSISTED",
            ),
        )
        assert rearmed["ready"] is True
        assert rearmed["replayed"] is False
    finally:
        coordinator.close()

    landed = duckdb.connect(str(database_path))
    try:
        landed.execute("CHECKPOINT")
        assert task_schema_inventory(landed) == before_inventory
        assert landed.execute(
            "SELECT COUNT(*), COUNT(DISTINCT task_cid) FROM coordination_tasks"
        ).fetchone() == (513, 513)
        assert landed.execute(
            "SELECT ready, body_json FROM coordination_tasks WHERE task_cid = ?",
            ["task:rearm-persisted"],
        ).fetchone() == (True, '{"payload":"preserve-exactly"}')
        assert landed.execute(
            "SELECT COUNT(*) FROM task_completions WHERE task_cid = ?",
            ["task:rearm-persisted"],
        ).fetchone()[0] == 0
        assert landed.execute(
            """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_name = 'coordination_tasks_rearm_staging'
            """
        ).fetchone()[0] == 0
        assert landed.execute(
            """
            SELECT COUNT(*) FROM duckdb_indexes()
            WHERE schema_name = 'main'
              AND index_name = 'coordination_tasks_ready_idx'
              AND table_name = 'coordination_tasks'
            """
        ).fetchone()[0] == 1
    finally:
        landed.close()

    reopened = open_database_coordinator(database_path, clock_ms=clock)
    try:
        replay = reopened.rearm_task_from_control(
            "task:rearm-persisted",
            control_task_observation=_control_rearm_observation(
                task_cid="task:rearm-persisted",
                task_alias="REARM-PERSISTED",
            ),
        )
        assert replay == {**rearmed, "replayed": True}
        projection = reopened.coordination_registry_projection()
        assert projection["counts"]["registered_tasks"] == 513
        assert projection["logical_completions"] == []
    finally:
        reopened.close()


@pytest.mark.parametrize("failure_point", ["begin", "commit"])
def test_control_rearm_transaction_boundaries_fail_closed(
    tmp_path: Path,
    failure_point: str,
) -> None:
    coordinator, _clock = _open(tmp_path)
    coordinator.register_task(
        task_cid="task:rearm-transaction",
        task_id="REARM-TRANSACTION",
    )
    _settled_control_completion(coordinator, "task:rearm-transaction")
    raw_connection = coordinator._require()  # noqa: SLF001

    class TransactionFault:
        def __init__(self, raw: object) -> None:
            self.raw = raw

        @property
        def in_transaction(self) -> bool:
            return bool(self.raw.in_transaction)  # type: ignore[attr-defined]

        def execute(
            self,
            statement: str,
            parameters: object | None = None,
        ) -> object:
            normalized = " ".join(str(statement).upper().split())
            if failure_point == "begin" and normalized == "BEGIN TRANSACTION":
                return self.raw.execute("SELECT 1")  # type: ignore[attr-defined]
            if failure_point == "commit" and normalized == "COMMIT":
                raise RuntimeError("injected task-rearm commit failure")
            if parameters is None:
                return self.raw.execute(statement)  # type: ignore[attr-defined]
            return self.raw.execute(  # type: ignore[attr-defined]
                statement,
                parameters,
            )

        def __getattr__(self, name: str) -> object:
            return getattr(self.raw, name)

    coordinator._connection = TransactionFault(raw_connection)  # noqa: SLF001
    expected_error = (
        DatabaseCoordinationError if failure_point == "begin" else RuntimeError
    )
    try:
        with pytest.raises(expected_error):
            coordinator.rearm_task_from_control(
                "task:rearm-transaction",
                control_task_observation=_control_rearm_observation(
                    task_cid="task:rearm-transaction",
                    task_alias="REARM-TRANSACTION",
                ),
            )
        assert coordinator._require().in_transaction is False  # noqa: SLF001
        readiness = coordinator.claimability("task:rearm-transaction")
        assert readiness["completion_status"] == AttemptStatus.SUCCEEDED.value
        assert readiness["claimable"] is False
        assert coordinator._require().execute(  # noqa: SLF001
            "SELECT ready FROM coordination_tasks WHERE task_cid = ?",
            ["task:rearm-transaction"],
        ).fetchone()[0] is False
        assert coordinator._require().execute(  # noqa: SLF001
            """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_name = 'coordination_tasks_rearm_staging'
            """
        ).fetchone()[0] == 0
    finally:
        coordinator.close()


def test_control_rearm_rejects_active_completion_authority(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:rearm-live", task_id="REARM-LIVE")
        claim = coordinator.claim_task(
            task_cid="task:rearm-live",
            owner_session_id="session:live-rearm",
        )
        prepared = coordinator.prepare_task_completion(
            claim,
            control_expected_revision=2,
            evidence_digest="sha256:live-rearm",
        )
        coordinator.complete_task_claim(
            claim,
            control_completion_receipt=_completed_control_task(prepared),
        )
        observation = _control_rearm_observation(
            task_cid="task:rearm-live",
            task_alias="REARM-LIVE",
            nested=True,
        )

        with pytest.raises(
            DatabaseCoordinationConflictError,
            match="requires a quiescent sidecar",
        ):
            coordinator.rearm_task_from_control(
                "task:rearm-live",
                control_task_observation=observation,
            )
        assert coordinator.claimability("task:rearm-live")[
            "completion_status"
        ] == AttemptStatus.SUCCEEDED.value

        coordinator.settle_task_claim(claim)
        maintenance = coordinator.acquire_maintenance_lease(
            owner_session_id="session:rearm-maintenance",
            scope="rearm-maintenance",
        )
        with pytest.raises(
            DatabaseCoordinationConflictError,
            match="requires a quiescent sidecar",
        ):
            coordinator.rearm_task_from_control(
                "task:rearm-live",
                control_task_observation=observation,
            )
        coordinator.release(maintenance.as_fenced_lease())
        assert coordinator.rearm_task_from_control(
            "task:rearm-live",
            control_task_observation=observation,
        )["ready"] is True
    finally:
        coordinator.close()


def test_control_rearm_rejects_malformed_stale_or_mismatched_observations(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:rearm-closed", task_id="REARM-CLOSED")
        _settled_control_completion(coordinator, "task:rearm-closed")
        invalid_observations: list[dict[str, object]] = [
            {},
            {"task_cid": "task:rearm-closed", "status": "retrying"},
            _control_rearm_observation(
                task_cid="task:rearm-closed",
                task_alias="REARM-CLOSED",
                nested=False,
            ),
            _control_rearm_observation(
                task_cid="task:not-the-task",
                task_alias="REARM-CLOSED",
            ),
            _control_rearm_observation(
                task_cid="task:rearm-closed",
                task_alias="NOT-THE-REGISTERED-TASK",
            ),
            _control_rearm_observation(
                task_cid="task:rearm-closed",
                task_alias="REARM-CLOSED",
                revision=3,
            ),
            _control_rearm_observation(
                task_cid="task:rearm-closed",
                task_alias="REARM-CLOSED",
                status="completed",
            ),
            _control_rearm_observation(
                task_cid="task:rearm-closed",
                task_alias="REARM-CLOSED",
                status="blocked",
            ),
            _control_rearm_observation(
                task_cid="task:rearm-closed",
                task_alias="REARM-CLOSED",
                status="in_progress",
            ),
            {
                "task_cid": "task:rearm-closed",
                "task_alias": "REARM-CLOSED",
                "status": "retrying",
                "revision": "4",
            },
            {
                **_control_rearm_observation(
                    task_cid="task:rearm-closed",
                    task_alias="REARM-CLOSED",
                    nested=True,
                ),
                "changed": False,
            },
            {
                **_control_rearm_observation(
                    task_cid="task:rearm-closed",
                    task_alias="REARM-CLOSED",
                ),
                "receipt_cid": "",
            },
        ]
        for observation in invalid_observations:
            with pytest.raises(
                (
                    DatabaseCoordinationStaleFenceError,
                    DatabaseCoordinationNotReadyError,
                )
            ):
                coordinator.rearm_task_from_control(
                    "task:rearm-closed",
                    control_task_observation=observation,
                )
            assert coordinator.claimability("task:rearm-closed")[
                "completion_status"
            ] == AttemptStatus.SUCCEEDED.value

        valid = _control_rearm_observation(
            task_cid="task:rearm-closed",
            task_alias="REARM-CLOSED",
        )
        coordinator.rearm_task_from_control(
            "task:rearm-closed",
            control_task_observation=valid,
        )
        with pytest.raises(DatabaseCoordinationStaleFenceError):
            coordinator.rearm_task_from_control(
                "task:rearm-closed",
                control_task_observation={
                    **valid,
                    "receipt_cid": "cid:forged-control-rearm",
                },
            )
        with pytest.raises(DatabaseCoordinationStaleFenceError):
            coordinator.rearm_task_from_control(
                "task:rearm-closed",
                control_task_observation=_control_rearm_observation(
                    task_cid="task:rearm-closed",
                    task_alias="REARM-CLOSED",
                    revision=5,
                ),
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
