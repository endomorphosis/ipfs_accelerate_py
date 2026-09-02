"""Tests for DatabaseCoordinator (DQP-015).

Evidence subset: acquire, renew, release, expiry, takeover, fairness,
dependency readiness, epoch monotonicity, stale fence, response loss.

Acceptance: Four processes never own the same exclusive scope; expired session
cannot renew or mutate; append/fair scheduling remains concurrent; stale
fencing epoch is rejected in every protected write; claim and task-attempt
creation are one transaction.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.merge import (
    database_coordination as coordination_module,
)
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    COORDINATION_REGISTRY_PROJECTION_SCHEMA,
    COORDINATION_STORAGE_REPAIR_SCHEMA,
    DATABASE_COORDINATOR_INTERFACE,
    FENCED_TASK_AUTHORITY_POPULATION_RECEIPT_SCHEMA,
    FENCED_LEASE_INTERFACE,
    MAINTENANCE_LEASE_INTERFACE,
    RESOURCE_CLAIM_INTERFACE,
    TASK_CLAIM_INTERFACE,
    AttemptStatus,
    DatabaseCoordinationConflictError,
    DatabaseCoordinationExpiredError,
    DatabaseCoordinationNotReadyError,
    DatabaseCoordinationStaleFenceError,
    DatabaseCoordinationStorageRepairedError,
    DatabaseCoordinator,
    LeaseKind,
    LeaseMode,
    LeaseState,
    ResourceClaim,
    TaskClaim,
    duckdb_available,
    exclusive_scope_key,
    fenced_task_authority_population_receipt_valid,
    open_database_coordinator,
    read_coordination_registry_projection,
    repair_coordination_art_index_storage,
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


def test_fenced_task_authority_population_receipt_is_closed_and_replay_stable(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(
            task_cid="task:receipt",
            task_id="RECEIPT",
            body={"private_value": "not-emitted"},
        )
        claim = coordinator.claim_task(
            task_cid="task:receipt",
            owner_session_id="session:receipt",
            idempotency_key="receipt:claim",
            body={"private_value": "not-emitted"},
        )
        kwargs = {
            "task_cid": claim.task_cid,
            "attempt_id": claim.attempt_id,
            "claim_id": claim.claim_id,
            "lease_id": claim.lease_id,
            "owner_session_id": claim.owner_session_id,
            "fencing_token": claim.fencing_token,
            "fence_epoch": claim.fence_epoch,
            "receipt_nonce": "receipt-nonce:one",
            "receipt_epoch": 1,
        }
        first = dict(
            coordinator.fenced_task_authority_population_receipt(**kwargs)
        )
        second = dict(
            coordinator.fenced_task_authority_population_receipt(**kwargs)
        )
    finally:
        coordinator.close()

    assert first == second
    assert first["schema"] == FENCED_TASK_AUTHORITY_POPULATION_RECEIPT_SCHEMA
    assert fenced_task_authority_population_receipt_valid(first)
    assert first["groups"]["coordination_tasks"]["count"] == 1
    assert first["groups"]["task_claims"]["count"] == 1
    assert first["groups"]["task_attempts"]["count"] == 1
    assert first["groups"]["fenced_leases"]["count"] == 1
    assert "not-emitted" not in str(first)


def test_fenced_task_authority_population_receipt_rejects_tamper_and_unknowns(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:tamper", task_id="TAMPER")
        claim = coordinator.claim_task(
            task_cid="task:tamper",
            owner_session_id="session:tamper",
        )
        receipt = dict(
            coordinator.fenced_task_authority_population_receipt(
                task_cid=claim.task_cid,
                attempt_id=claim.attempt_id,
                claim_id=claim.claim_id,
                lease_id=claim.lease_id,
                owner_session_id=claim.owner_session_id,
                fencing_token=claim.fencing_token,
                fence_epoch=claim.fence_epoch,
                receipt_nonce="receipt-nonce:tamper",
                receipt_epoch=1,
            )
        )
    finally:
        coordinator.close()

    unknown = {**receipt, "unreviewed": True}
    assert not fenced_task_authority_population_receipt_valid(unknown)
    missing_groups = dict(receipt)
    missing_groups["groups"] = dict(receipt["groups"])
    missing_groups["groups"].pop("lease_events")
    assert not fenced_task_authority_population_receipt_valid(missing_groups)
    tampered = dict(receipt)
    tampered["groups"] = dict(receipt["groups"])
    tampered["groups"]["task_claims"] = dict(
        receipt["groups"]["task_claims"]
    )
    tampered["groups"]["task_claims"]["count"] = 2
    assert not fenced_task_authority_population_receipt_valid(tampered)

    def rehash(candidate: dict[str, object]) -> dict[str, object]:
        unsigned = copy.deepcopy(candidate)
        unsigned.pop("receipt_cid", None)
        unsigned["receipt_cid"] = coordination_module._sha256_hex(
            coordination_module.canonical_json_bytes(unsigned)
        )
        return unsigned

    subject_splice = copy.deepcopy(receipt)
    subject_splice["subject"]["task_cid"] = "task:forged"
    assert not fenced_task_authority_population_receipt_valid(
        rehash(subject_splice)
    )

    deep_unknown = copy.deepcopy(receipt)
    claim_group = deep_unknown["groups"]["task_claims"]
    claim_group["rows"][0]["unreviewed"] = True
    claim_group["rows_digest"] = coordination_module._sha256_hex(
        coordination_module.canonical_json_bytes(claim_group["rows"])
    )
    deep_unknown["task_population_root"] = coordination_module._sha256_hex(
        coordination_module.canonical_json_bytes(deep_unknown["groups"])
    )
    assert not fenced_task_authority_population_receipt_valid(
        rehash(deep_unknown)
    )

    oversized = copy.deepcopy(receipt)
    body_commitment = oversized["groups"]["task_claims"]["rows"][0][
        "body_json"
    ]
    body_commitment["canonical_byte_length"] = (
        coordination_module.MAX_FENCED_TASK_AUTHORITY_FIELD_BYTES + 1
    )
    oversized_group = oversized["groups"]["task_claims"]
    oversized_group["rows_digest"] = coordination_module._sha256_hex(
        coordination_module.canonical_json_bytes(oversized_group["rows"])
    )
    oversized["task_population_root"] = coordination_module._sha256_hex(
        coordination_module.canonical_json_bytes(oversized["groups"])
    )
    assert not fenced_task_authority_population_receipt_valid(rehash(oversized))

    zero_length = copy.deepcopy(receipt)
    zero_group = zero_length["groups"]["task_claims"]
    zero_group["rows"][0]["body_json"]["canonical_byte_length"] = 0
    zero_group["rows_digest"] = coordination_module._sha256_hex(
        coordination_module.canonical_json_bytes(zero_group["rows"])
    )
    zero_length["task_population_root"] = coordination_module._sha256_hex(
        coordination_module.canonical_json_bytes(zero_length["groups"])
    )
    assert not fenced_task_authority_population_receipt_valid(
        rehash(zero_length)
    )

    out_of_range = copy.deepcopy(receipt)
    task_group = out_of_range["groups"]["coordination_tasks"]
    task_group["rows"][0]["registered_at_ms"] = 2**63
    task_group["rows_digest"] = coordination_module._sha256_hex(
        coordination_module.canonical_json_bytes(task_group["rows"])
    )
    out_of_range["task_population_root"] = coordination_module._sha256_hex(
        coordination_module.canonical_json_bytes(out_of_range["groups"])
    )
    assert not fenced_task_authority_population_receipt_valid(
        rehash(out_of_range)
    )

    duplicate = copy.deepcopy(receipt)
    token_group = duplicate["groups"]["token_history"]
    assert token_group["rows"]
    token_group["rows"].append(copy.deepcopy(token_group["rows"][0]))
    token_group["count"] = len(token_group["rows"])
    token_group["rows_digest"] = coordination_module._sha256_hex(
        coordination_module.canonical_json_bytes(token_group["rows"])
    )
    duplicate["task_population_root"] = coordination_module._sha256_hex(
        coordination_module.canonical_json_bytes(duplicate["groups"])
    )
    assert not fenced_task_authority_population_receipt_valid(rehash(duplicate))

    same_primary_key = copy.deepcopy(receipt)
    event_group = same_primary_key["groups"]["lease_events"]
    assert event_group["rows"]
    forged_event = copy.deepcopy(event_group["rows"][0])
    forged_event["observed_at_ms"] += 1
    event_group["rows"].append(forged_event)
    event_group["rows"].sort(
        key=lambda row: (row["observed_at_ms"], row["event_id"])
    )
    event_group["count"] = len(event_group["rows"])
    event_group["rows_digest"] = coordination_module._sha256_hex(
        coordination_module.canonical_json_bytes(event_group["rows"])
    )
    same_primary_key["task_population_root"] = coordination_module._sha256_hex(
        coordination_module.canonical_json_bytes(same_primary_key["groups"])
    )
    assert not fenced_task_authority_population_receipt_valid(
        rehash(same_primary_key)
    )

    cross_row_splice = copy.deepcopy(receipt)
    claim_row = cross_row_splice["groups"]["task_claims"]["rows"][0]
    lease_row = cross_row_splice["groups"]["fenced_leases"]["rows"][0]
    claim_row["attempt_number"] += 100
    lease_row["attempt_number"] += 100
    for group_name in ("task_claims", "fenced_leases"):
        group = cross_row_splice["groups"][group_name]
        group["rows_digest"] = coordination_module._sha256_hex(
            coordination_module.canonical_json_bytes(group["rows"])
        )
    cross_row_splice["task_population_root"] = coordination_module._sha256_hex(
        coordination_module.canonical_json_bytes(cross_row_splice["groups"])
    )
    assert not fenced_task_authority_population_receipt_valid(
        rehash(cross_row_splice)
    )

    assert receipt["privacy_boundary"] == (
        coordination_module.FENCED_TASK_AUTHORITY_PRIVACY_BOUNDARY
    )
    assert receipt["nonclaims"] == list(
        coordination_module.FENCED_TASK_AUTHORITY_NONCLAIMS
    )


def test_fenced_task_authority_cross_store_callback_blocks_reentry(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:callback", task_id="CALLBACK")
        claim = coordinator.claim_task(
            task_cid="task:callback",
            owner_session_id="session:callback",
        )
        kwargs = {
            "task_cid": claim.task_cid,
            "attempt_id": claim.attempt_id,
            "claim_id": claim.claim_id,
            "lease_id": claim.lease_id,
            "owner_session_id": claim.owner_session_id,
            "fencing_token": claim.fencing_token,
            "fence_epoch": claim.fence_epoch,
            "receipt_nonce": "receipt-nonce:callback",
            "receipt_epoch": 1,
        }
        result = coordinator.execute_with_fenced_task_authority_population(
            **kwargs,
            callback=lambda receipt: {
                "transaction_boundary": "cross_store_stable_read",
                "coordinator_receipt_cid": receipt["receipt_cid"],
            },
        )
        assert result["transaction_boundary"] == "cross_store_stable_read"

        def reenter(_receipt: object) -> dict[str, object]:
            coordinator.get_task_attempt(claim.attempt_id)
            return {}

        with pytest.raises(DatabaseCoordinationConflictError, match="re-enter"):
            coordinator.execute_with_fenced_task_authority_population(
                **kwargs,
                callback=reenter,
            )

        def cancel(_receipt: object) -> dict[str, object]:
            raise KeyboardInterrupt("cancelled receipt capture")

        with pytest.raises(KeyboardInterrupt, match="cancelled receipt"):
            coordinator.execute_with_fenced_task_authority_population(
                **kwargs,
                callback=cancel,
            )
        assert not coordinator._connection.in_transaction
        recovered = coordinator.execute_with_fenced_task_authority_population(
            **kwargs,
            callback=lambda receipt: {
                "coordinator_receipt_cid": receipt["receipt_cid"]
            },
        )
        assert recovered["coordinator_receipt_cid"]
    finally:
        coordinator.close()


def test_fenced_task_authority_cross_store_callback_rejects_quack_transport(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator._quack_transport = True
        with pytest.raises(
            DatabaseCoordinationConflictError,
            match="unavailable through Quack",
        ):
            coordinator.execute_with_fenced_task_authority_population(
                task_cid="task:not-read",
                attempt_id="attempt:not-read",
                claim_id="claim:not-read",
                lease_id="lease:not-read",
                owner_session_id="session:not-read",
                fencing_token=1,
                fence_epoch=1,
                receipt_nonce="receipt:not-read",
                receipt_epoch=1,
                callback=lambda _receipt: {},
            )
    finally:
        coordinator._quack_transport = False
        coordinator.close()


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


def test_commit_failure_is_never_reported_as_success(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)

    class FailingCommitConnection:
        in_transaction = True

        @staticmethod
        def commit() -> None:
            raise RuntimeError("injected DuckDB commit failure")

    try:
        with pytest.raises(RuntimeError, match="injected DuckDB commit failure"):
            coordinator._commit_if_idle(FailingCommitConnection())
    finally:
        coordinator.close()


def test_exact_art_commit_failure_repairs_storage_but_requires_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, _clock = _open(tmp_path)
    actual = coordinator._require()
    actual.close()

    class FatalException(RuntimeError):
        pass

    class FailedConnection:
        in_transaction = True
        closed = False

        def commit(self) -> None:
            raise FatalException(
                "FATAL Error: Invalid Input Error: Failed to delete all rows "
                "from index. Only deleted 0 out of 1 rows."
            )

        def rollback(self) -> None:
            self.in_transaction = False

        def close(self) -> None:
            self.closed = True

    failed = FailedConnection()
    coordinator._connection = failed
    coordinator._closed = False
    expected_receipt = {
        "schema": COORDINATION_STORAGE_REPAIR_SCHEMA,
        "receipt_cid": "sha256:repair",
        "logical_projection_equal": True,
        "retry_required": True,
    }
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.merge.database_coordination."
        "repair_coordination_art_index_storage",
        lambda _path: expected_receipt,
    )

    with pytest.raises(DatabaseCoordinationStorageRepairedError) as captured:
        coordinator._commit_if_idle(failed)

    assert dict(captured.value.receipt) == expected_receipt
    assert failed.closed is True
    assert coordinator.is_open is False


def test_art_storage_rebuild_preserves_exact_logical_projection_and_source(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    database_path = coordinator.database_path
    try:
        coordinator.register_task(task_cid="task:done", task_id="DONE")
        coordinator.register_task(task_cid="task:ready", task_id="READY")
        before = coordinator.coordination_registry_projection()
    finally:
        coordinator.close()
    database_path.with_name(database_path.name + ".wal").touch()

    receipt = repair_coordination_art_index_storage(database_path)

    assert receipt["schema"] == COORDINATION_STORAGE_REPAIR_SCHEMA
    assert receipt["pre_projection_root"] == receipt["post_projection_root"]
    assert receipt["pre_registry_projection_root"] == before["projection_root"]
    assert receipt["post_registry_projection_root"] == before["projection_root"]
    assert receipt["logical_projection_equal"] is True
    assert receipt["interrupted_transaction_outcome"] == (
        "not_inferred_reconcile_exact_operation"
    )
    assert receipt["repair_accepted_interrupted_transaction"] is False
    assert receipt["reconciliation_required"] is True
    assert receipt["retry_required"] is True
    assert Path(receipt["quarantined_source_path"]).is_file()
    assert Path(receipt["quarantined_empty_wal_path"]).is_file()
    assert receipt["source_sha256"].startswith("sha256:")
    assert receipt["replacement_sha256"].startswith("sha256:")

    with open_database_coordinator(database_path) as rebuilt:
        assert rebuilt.coordination_registry_projection() == before
        claim = rebuilt.claim_ready_task(
            owner_session_id="session:after-repair",
        )
        assert claim is not None
        assert claim.task_cid in {"task:done", "task:ready"}


def test_storage_projection_includes_metadata_timestamps_and_histories(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(task_cid="task:history", task_id="HISTORY")
        coordinator.acquire(
            lease_kind=LeaseKind.RESOURCE,
            scope="resource:history",
            owner_session_id="session:history",
            resource_kind="file",
            resource_id="history",
        )
        connection = coordinator._require()
        registry_before = coordinator.coordination_registry_projection()
        storage_before = (
            coordination_module._coordination_storage_projection_from_connection(
                connection,
                validate_authority=True,
            )
        )
        coordinator._begin(connection)
        connection.execute(
            "INSERT INTO coordination_metadata(key, value) VALUES (?, ?)",
            ["repair-projection-test", "changed"],
        )
        connection.execute(
            "UPDATE lease_events SET observed_at_ms = observed_at_ms + 777"
        )
        connection.execute(
            "UPDATE token_history SET recorded_at_ms = recorded_at_ms + 888"
        )
        coordinator._commit_if_idle(connection)
        registry_after = coordinator.coordination_registry_projection()
        storage_after = (
            coordination_module._coordination_storage_projection_from_connection(
                connection,
                validate_authority=True,
            )
        )
    finally:
        coordinator.close()

    # The scheduling identity deliberately omits these values. Physical repair
    # evidence must not.
    assert registry_after["projection_root"] == registry_before["projection_root"]
    assert storage_after["projection_root"] != storage_before["projection_root"]
    before_tables = {
        item["table"]: item for item in storage_before["tables"]
    }
    after_tables = {item["table"]: item for item in storage_after["tables"]}
    assert (
        before_tables["coordination_metadata"]["rows_root"]
        != after_tables["coordination_metadata"]["rows_root"]
    )
    assert (
        before_tables["lease_events"]["rows_root"]
        != after_tables["lease_events"]["rows_root"]
    )
    assert (
        before_tables["token_history"]["rows_root"]
        != after_tables["token_history"]["rows_root"]
    )


def _seed_art_repair_failure_case(tmp_path: Path) -> tuple[Path, dict[str, object], str]:
    coordinator, _clock = _open(tmp_path)
    database_path = coordinator.database_path
    try:
        coordinator.register_task(task_cid="task:preserve", task_id="PRESERVE")
        before = coordinator.coordination_registry_projection()
    finally:
        coordinator.close()
    digest = "sha256:" + hashlib.sha256(database_path.read_bytes()).hexdigest()
    return database_path, before, digest


def _assert_repair_failure_preserved_authority(
    database_path: Path,
    before: dict[str, object],
    source_digest: str,
) -> None:
    assert database_path.is_file()
    with open_database_coordinator(database_path) as coordinator:
        assert coordinator.coordination_registry_projection() == before
    quarantine = database_path.parent / ".coordination-art-repair-quarantine"
    assert tuple(quarantine.glob(f"{database_path.name}.{source_digest[7:]}.*.duckdb"))


def test_art_repair_install_rename_failure_keeps_original_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path, before, source_digest = _seed_art_repair_failure_case(tmp_path)
    real_replace = coordination_module.os.replace

    def fail_candidate_install(source: object, target: object) -> None:
        source_path = Path(source)
        if (
            Path(target) == database_path
            and ".art-repair-" in source_path.name
            and ".art-repair-rollback-" not in source_path.name
        ):
            raise OSError("injected candidate install rename failure")
        real_replace(source, target)

    monkeypatch.setattr(coordination_module.os, "replace", fail_candidate_install)
    with pytest.raises(
        coordination_module.DatabaseCoordinationStorageRepairError,
        match="candidate was not installed",
    ):
        repair_coordination_art_index_storage(database_path)

    assert (
        "sha256:" + hashlib.sha256(database_path.read_bytes()).hexdigest()
        == source_digest
    )
    _assert_repair_failure_preserved_authority(
        database_path, before, source_digest
    )


def test_art_repair_post_install_fsync_failure_rolls_back_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path, before, source_digest = _seed_art_repair_failure_case(tmp_path)
    real_fsync_directory = coordination_module._fsync_coordination_directory
    parent_calls = 0

    def fail_first_post_install_fsync(path: Path) -> None:
        nonlocal parent_calls
        if path == database_path.parent:
            parent_calls += 1
            if parent_calls == 2:
                raise OSError("injected post-install directory fsync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(
        coordination_module,
        "_fsync_coordination_directory",
        fail_first_post_install_fsync,
    )
    with pytest.raises(
        coordination_module.DatabaseCoordinationStorageRepairError,
        match="candidate was not installed",
    ):
        repair_coordination_art_index_storage(database_path)

    assert (
        "sha256:" + hashlib.sha256(database_path.read_bytes()).hexdigest()
        == source_digest
    )
    _assert_repair_failure_preserved_authority(
        database_path, before, source_digest
    )
    assert tuple(
        (database_path.parent / ".coordination-art-repair-quarantine").glob(
            "*.failed-replacement.duckdb"
        )
    )


def test_art_repair_verifier_failure_rolls_back_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path, before, source_digest = _seed_art_repair_failure_case(tmp_path)
    real_connect = coordination_module.connect_duckdb_with_policy
    source_read_count = 0

    def fail_independent_verifier(
        duckdb_module: object,
        database: Path | str,
        *,
        read_only: bool = False,
        configuration: dict[str, object] | None = None,
    ) -> object:
        nonlocal source_read_count
        if Path(database) == database_path and read_only:
            source_read_count += 1
            if source_read_count == 2:
                raise RuntimeError("injected independent verifier failure")
        return real_connect(
            duckdb_module,
            database,
            read_only=read_only,
            configuration=configuration,
        )

    monkeypatch.setattr(
        coordination_module,
        "connect_duckdb_with_policy",
        fail_independent_verifier,
    )
    with pytest.raises(
        coordination_module.DatabaseCoordinationStorageRepairError,
        match="candidate was not installed",
    ):
        repair_coordination_art_index_storage(database_path)

    assert (
        "sha256:" + hashlib.sha256(database_path.read_bytes()).hexdigest()
        == source_digest
    )
    _assert_repair_failure_preserved_authority(
        database_path, before, source_digest
    )


def test_art_repair_refuses_nonempty_wal_without_touching_authority(
    tmp_path: Path,
) -> None:
    database_path, before, source_digest = _seed_art_repair_failure_case(tmp_path)
    wal_path = database_path.with_name(database_path.name + ".wal")
    wal_path.write_bytes(b"uncheckpointed-authority")

    with pytest.raises(
        coordination_module.DatabaseCoordinationStorageRepairError,
        match="refuses an uncheckpointed WAL",
    ):
        repair_coordination_art_index_storage(database_path)

    assert wal_path.read_bytes() == b"uncheckpointed-authority"
    assert (
        "sha256:" + hashlib.sha256(database_path.read_bytes()).hexdigest()
        == source_digest
    )
    with open_database_coordinator(database_path) as coordinator:
        assert coordinator.coordination_registry_projection() == before


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


def test_terminal_claim_lease_barrier_rejects_successor_fence(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(
            task_cid="task:terminal-barrier-successor",
            task_id="TERMINAL-BARRIER-SUCCESSOR",
        )
        old = coordinator.claim_task(
            task_cid="task:terminal-barrier-successor",
            owner_session_id="session:old",
        )
        coordinator.release(old.as_fenced_lease(), reason="old-terminal")
        old = coordinator.get_task_claim(old.claim_id)
        assert old is not None and old.state is LeaseState.RELEASED
        old_lease = coordinator.get_lease(old.lease_id)
        assert old_lease is not None and old_lease.state is LeaseState.RELEASED

        successor = coordinator.claim_task(
            task_cid=old.task_cid,
            owner_session_id="session:successor",
        )
        called: list[str] = []
        with pytest.raises(
            DatabaseCoordinationStaleFenceError,
            match="latest fence|successor task authority",
        ):
            coordinator.execute_with_terminal_task_claim_barrier(
                old,
                lambda: called.append("control-cas"),
                lease=old_lease,
            )
        assert called == []
        current = coordinator.get_task_claim(successor.claim_id)
        assert current is not None and current.state is LeaseState.ACCEPTED
    finally:
        coordinator.close()


def test_terminal_claim_lease_barrier_admits_exact_latest_terminal(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(
            task_cid="task:terminal-barrier-exact",
            task_id="TERMINAL-BARRIER-EXACT",
        )
        claim = coordinator.claim_task(
            task_cid="task:terminal-barrier-exact",
            owner_session_id="session:old",
        )
        coordinator.release(claim.as_fenced_lease(), reason="exact-terminal")
        claim = coordinator.get_task_claim(claim.claim_id)
        assert claim is not None and claim.state is LeaseState.RELEASED
        lease = coordinator.get_lease(claim.lease_id)
        assert lease is not None and lease.state is LeaseState.RELEASED
        called: list[str] = []

        result = coordinator.execute_with_terminal_task_claim_barrier(
            claim,
            lambda: called.append("control-cas") or "committed",
            lease=lease,
        )

        assert result == "committed"
        assert called == ["control-cas"]
    finally:
        coordinator.close()


def test_unprepared_terminalization_never_releases_prepared_claim(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(
            task_cid="task:prepared-terminalization",
            task_id="PREPARED-TERMINALIZATION",
        )
        claim = coordinator.claim_task(
            task_cid="task:prepared-terminalization",
            owner_session_id="session:worker",
        )
        coordinator.prepare_task_completion(
            claim,
            control_expected_revision=7,
            evidence_digest="sha256:" + "a" * 64,
        )

        with pytest.raises(
            DatabaseCoordinationNotReadyError,
            match="preparation owns settlement",
        ):
            coordinator.terminalize_unprepared_task_claim(
                claim,
                lease=claim.as_fenced_lease(),
                reason="must-not-release",
            )

        claim_readback = coordinator.get_task_claim(claim.claim_id)
        lease_readback = coordinator.get_lease(claim.lease_id)
        assert claim_readback is not None
        assert claim_readback.state is LeaseState.ACCEPTED
        assert lease_readback is not None
        assert lease_readback.state is LeaseState.ACCEPTED
        assert coordinator.get_prepared_task_completion(claim.task_cid) is not None
    finally:
        coordinator.close()


def test_renewed_claim_terminalizes_and_cross_store_barrier_remains_exact(
    tmp_path: Path,
) -> None:
    coordinator, clock = _open(tmp_path)
    try:
        coordinator.register_task(
            task_cid="task:renewed-terminal-barrier",
            task_id="RENEWED-TERMINAL-BARRIER",
        )
        original = coordinator.claim_task(
            task_cid="task:renewed-terminal-barrier",
            owner_session_id="session:worker",
        )
        clock.advance(1_000)
        renewed_lease = coordinator.renew(
            original.as_fenced_lease(),
            lease_ms=90_000,
            now_ms=clock(),
        )
        renewed_claim = coordinator.get_task_claim(original.claim_id)
        assert renewed_claim is not None
        assert renewed_claim.revision > original.revision
        assert renewed_claim.as_fenced_lease().to_dict() == renewed_lease.to_dict()

        terminal_claim, terminal_lease = (
            coordinator.terminalize_unprepared_task_claim(
                renewed_claim,
                lease=renewed_lease,
                reason="renewed-orphan-terminal",
                now_ms=clock(),
            )
        )
        called: list[str] = []
        result = coordinator.execute_with_terminal_task_claim_barrier(
            terminal_claim,
            lambda: called.append("control-cas") or "committed",
            lease=terminal_lease,
        )

        assert result == "committed"
        assert called == ["control-cas"]
        assert terminal_claim.state is LeaseState.RELEASED
        assert terminal_lease.state is LeaseState.RELEASED
    finally:
        coordinator.close()


@pytest.mark.parametrize(
    "corruption",
    (
        "attempt_revision",
        "attempt_started_at",
        "attempt_finished_at",
        "claim_body_malformed",
        "lease_body_malformed",
    ),
)
def test_unprepared_terminalization_rejects_corrupt_running_authority_without_mutation(
    tmp_path: Path,
    corruption: str,
) -> None:
    coordinator, clock = _open(tmp_path)
    try:
        coordinator.register_task(
            task_cid=f"task:terminalization-corrupt:{corruption}",
            task_id=f"TERMINALIZATION-CORRUPT-{corruption}",
        )
        claim = coordinator.claim_task(
            task_cid=f"task:terminalization-corrupt:{corruption}",
            owner_session_id="session:worker",
        )
        lease = claim.as_fenced_lease()
        connection = coordinator._require()
        coordinator._begin(connection)
        if corruption == "attempt_revision":
            connection.execute(
                "UPDATE task_attempts SET revision = 999 WHERE attempt_id = ?",
                [claim.attempt_id],
            )
        elif corruption == "attempt_started_at":
            connection.execute(
                "UPDATE task_attempts SET started_at_ms = started_at_ms + 1 "
                "WHERE attempt_id = ?",
                [claim.attempt_id],
            )
        elif corruption == "attempt_finished_at":
            connection.execute(
                "UPDATE task_attempts SET finished_at_ms = ? WHERE attempt_id = ?",
                [clock(), claim.attempt_id],
            )
        elif corruption == "claim_body_malformed":
            connection.execute(
                "UPDATE task_claims SET body_json = 'xx' WHERE claim_id = ?",
                [claim.claim_id],
            )
        else:
            connection.execute(
                "UPDATE fenced_leases SET body_json = 'xx' WHERE lease_id = ?",
                [claim.lease_id],
            )
        coordinator._commit_if_idle(connection)

        with pytest.raises(DatabaseCoordinationStaleFenceError):
            coordinator.terminalize_unprepared_task_claim(
                claim,
                lease=lease,
                reason="must-not-terminalize",
                now_ms=clock(),
            )

        observed_claim = coordinator.get_task_claim(claim.claim_id)
        observed_lease = coordinator.get_lease(claim.lease_id)
        observed_attempt = coordinator.get_task_attempt(claim.attempt_id)
        assert observed_claim is not None
        assert observed_claim.state is LeaseState.ACCEPTED
        assert observed_lease is not None
        assert observed_lease.state is LeaseState.ACCEPTED
        assert observed_attempt is not None
        assert observed_attempt.status is AttemptStatus.RUNNING
        terminal_event_count = connection.execute(
            "SELECT COUNT(*) FROM lease_events "
            "WHERE lease_id = ? AND event_type IN ('released', 'expired')",
            [claim.lease_id],
        ).fetchone()
        assert terminal_event_count is not None
        assert int(terminal_event_count[0]) == 0
    finally:
        coordinator.close()


def test_unprepared_terminalization_rejects_regressed_clock_without_mutation(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(
            task_cid="task:terminalization-regressed-clock",
            task_id="TERMINALIZATION-REGRESSED-CLOCK",
        )
        claim = coordinator.claim_task(
            task_cid="task:terminalization-regressed-clock",
            owner_session_id="session:worker",
        )

        with pytest.raises(DatabaseCoordinationStaleFenceError, match="clock"):
            coordinator.terminalize_unprepared_task_claim(
                claim,
                lease=claim.as_fenced_lease(),
                now_ms=claim.claimed_at_ms - 1,
            )

        observed_claim = coordinator.get_task_claim(claim.claim_id)
        observed_attempt = coordinator.get_task_attempt(claim.attempt_id)
        assert observed_claim is not None
        assert observed_claim.state is LeaseState.ACCEPTED
        assert observed_attempt is not None
        assert observed_attempt.status is AttemptStatus.RUNNING
    finally:
        coordinator.close()


@pytest.mark.parametrize(
    "corruption",
    (
        "attempt_revision",
        "attempt_finished_at",
        "claim_released_at",
        "terminal_event_timestamp",
        "claim_body_malformed",
        "lease_body_malformed",
        "terminal_event_body_malformed",
        "terminal_event_empty_reason",
        "attempt_status_oversized",
    ),
)
def test_terminal_claim_barrier_rejects_corrupt_durable_relation_before_callback(
    tmp_path: Path,
    corruption: str,
) -> None:
    coordinator, clock = _open(tmp_path)
    try:
        coordinator.register_task(
            task_cid=f"task:terminal-barrier-corrupt:{corruption}",
            task_id=f"TERMINAL-BARRIER-CORRUPT-{corruption}",
        )
        original = coordinator.claim_task(
            task_cid=f"task:terminal-barrier-corrupt:{corruption}",
            owner_session_id="session:worker",
        )
        claim, lease = coordinator.terminalize_unprepared_task_claim(
            original,
            lease=original.as_fenced_lease(),
            reason="canonical-release",
            now_ms=clock(),
        )
        connection = coordinator._require()
        coordinator._begin(connection)
        if corruption == "attempt_revision":
            connection.execute(
                "UPDATE task_attempts SET revision = 999 WHERE attempt_id = ?",
                [claim.attempt_id],
            )
        elif corruption == "attempt_finished_at":
            connection.execute(
                "UPDATE task_attempts SET finished_at_ms = finished_at_ms + 1 "
                "WHERE attempt_id = ?",
                [claim.attempt_id],
            )
        elif corruption == "claim_released_at":
            connection.execute(
                "UPDATE task_claims SET released_at_ms = released_at_ms + 1 "
                "WHERE claim_id = ?",
                [claim.claim_id],
            )
        elif corruption == "terminal_event_timestamp":
            connection.execute(
                "UPDATE lease_events SET observed_at_ms = observed_at_ms + 1 "
                "WHERE lease_id = ? AND event_type = 'released'",
                [claim.lease_id],
            )
        elif corruption == "claim_body_malformed":
            connection.execute(
                "UPDATE task_claims SET body_json = 'xx' WHERE claim_id = ?",
                [claim.claim_id],
            )
        elif corruption == "lease_body_malformed":
            connection.execute(
                "UPDATE fenced_leases SET body_json = 'xx' WHERE lease_id = ?",
                [claim.lease_id],
            )
        elif corruption == "terminal_event_body_malformed":
            connection.execute(
                "UPDATE lease_events SET body_json = repeat('x', "
                "octet_length(encode(body_json))) "
                "WHERE lease_id = ? AND event_type = 'released'",
                [claim.lease_id],
            )
        elif corruption == "terminal_event_empty_reason":
            connection.execute(
                "UPDATE lease_events SET body_json = '{\"reason\":\"\"}' "
                "WHERE lease_id = ? AND event_type = 'released'",
                [claim.lease_id],
            )
        else:
            connection.execute(
                "UPDATE task_attempts SET status = repeat('x', 300000) "
                "WHERE attempt_id = ?",
                [claim.attempt_id],
            )
        coordinator._commit_if_idle(connection)
        called: list[str] = []

        with pytest.raises(DatabaseCoordinationStaleFenceError):
            coordinator.execute_with_terminal_task_claim_barrier(
                claim,
                lambda: called.append("control-cas"),
                lease=lease,
            )

        assert called == []
    finally:
        coordinator.close()


def test_terminal_claim_barrier_admits_exact_expired_relation(
    tmp_path: Path,
) -> None:
    coordinator, clock = _open(tmp_path)
    try:
        coordinator.register_task(
            task_cid="task:terminal-barrier-expired",
            task_id="TERMINAL-BARRIER-EXPIRED",
        )
        original = coordinator.claim_task(
            task_cid="task:terminal-barrier-expired",
            owner_session_id="session:worker",
        )
        clock.advance(60_001)
        claim, lease = coordinator.terminalize_unprepared_task_claim(
            original,
            lease=original.as_fenced_lease(),
            now_ms=clock(),
        )
        called: list[str] = []

        result = coordinator.execute_with_terminal_task_claim_barrier(
            claim,
            lambda: called.append("control-cas") or "committed",
            lease=lease,
        )

        assert claim.state is LeaseState.EXPIRED
        assert lease.state is LeaseState.EXPIRED
        assert result == "committed"
        assert called == ["control-cas"]
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


def test_ordinary_succeeded_completion_is_not_reinterpreted_as_preparation(
    tmp_path: Path,
) -> None:
    coordinator, clock = _open(tmp_path, default_lease_ms=10_000)
    try:
        coordinator.register_task(
            task_cid="task:controller-completed",
            task_id="CONTROLLER-COMPLETED",
        )
        claim = coordinator.claim_task(
            task_cid="task:controller-completed",
            owner_session_id="session:historical-worker",
        )
        clock.advance(10_000)
        coordinator.expire_task_claim(claim, now_ms=clock())
        coordinator.mark_task_complete(
            claim.task_cid,
            status=AttemptStatus.SUCCEEDED.value,
            body={
                "schema": "pctdd/orphan-terminal-coordination-completion@1",
                "task_cid": claim.task_cid,
                "claim_id": claim.claim_id,
                "operator_owned": True,
            },
        )

        # Controller-owned completion records are dependency authority, not
        # two-phase preparation barriers.  A historical matching claim must
        # not make the daemon reinterpret their closed schema and crash-loop.
        assert coordinator.list_unsettled_task_completions(limit=10) == []
        assert coordinator.get_task_claim(claim.claim_id).state is LeaseState.EXPIRED
        assert coordinator.claimability(claim.task_cid)["completion_status"] == (
            AttemptStatus.SUCCEEDED.value
        )

        # A non-preparation logical completion that sorts first must not
        # consume a bounded recovery slot or starve a real promoted barrier.
        coordinator.register_task(
            task_cid="task:authoritative-promoted",
            task_id="AUTHORITATIVE-PROMOTED",
        )
        promoted_claim = coordinator.claim_task(
            task_cid="task:authoritative-promoted",
            owner_session_id="session:current-worker",
        )
        prepared = coordinator.prepare_task_completion(
            promoted_claim,
            control_expected_revision=2,
            evidence_digest="sha256:authoritative-promoted",
        )
        coordinator.complete_task_claim(
            promoted_claim,
            control_completion_receipt=_completed_control_task(prepared),
        )
        bounded = coordinator.list_unsettled_task_completions(limit=1)
        assert [item["task_cid"] for item in bounded] == [promoted_claim.task_cid]
    finally:
        coordinator.close()


def test_non_authoritative_literal_prepared_completion_fails_closed(
    tmp_path: Path,
) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(
            task_cid="task:malformed-prepared",
            task_id="MALFORMED-PREPARED",
        )
        claim = coordinator.claim_task(
            task_cid="task:malformed-prepared",
            owner_session_id="session:malformed",
        )
        prepared = coordinator.prepare_task_completion(
            claim,
            control_expected_revision=2,
            evidence_digest="sha256:malformed-prepared",
        )
        malformed = dict(prepared)
        malformed["schema"] = "pctdd/orphan-terminal-coordination-completion@1"
        coordinator.mark_task_complete(
            claim.task_cid,
            status="prepared",
            body=malformed,
        )

        with pytest.raises(
            DatabaseCoordinationStaleFenceError,
            match="prepared completion schema is not authoritative",
        ):
            coordinator.list_unsettled_task_completions(limit=10)
    finally:
        coordinator.close()


def test_forged_promoted_preparation_digest_fails_closed(tmp_path: Path) -> None:
    coordinator, _clock = _open(tmp_path)
    try:
        coordinator.register_task(
            task_cid="task:forged-promoted",
            task_id="FORGED-PROMOTED",
        )
        claim = coordinator.claim_task(
            task_cid="task:forged-promoted",
            owner_session_id="session:forged",
        )
        prepared = coordinator.prepare_task_completion(
            claim,
            control_expected_revision=2,
            evidence_digest="sha256:forged-promoted",
        )
        forged = dict(prepared)
        forged["preparation_digest"] = "sha256:forged"
        coordinator.mark_task_complete(
            claim.task_cid,
            status=AttemptStatus.SUCCEEDED.value,
            body=forged,
        )

        with pytest.raises(
            DatabaseCoordinationStaleFenceError,
            match="prepared completion digest does not match its bound body",
        ):
            coordinator.list_unsettled_task_completions(limit=10)
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
