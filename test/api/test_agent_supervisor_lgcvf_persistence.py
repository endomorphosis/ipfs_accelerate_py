"""LGCVF-101: typed operational references, CAS, leases, fences, restart.

Required evidence: restart, stale-worker, duplicate completion, fence,
single-writer, and outbox tests. Quack is not treated as qualified.
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_repository import (
    TYPED_OPERATIONAL_STORE_INTERFACE,
    TypedOperationalReferenceStore,
    TypedOperationalStoreError,
)


def test_append_only_cas_and_single_writer() -> None:
    store = TypedOperationalReferenceStore(writer_id="writer:a")
    assert store.INTERFACE == TYPED_OPERATIONAL_STORE_INTERFACE
    assert store.quack_qualified is False
    assert store.single_writer is True
    first = store.append_reference(
        "ref:semantic",
        "baguqeeraexamplecid0001",
        operation_id="op:1",
    )
    second = store.append_reference(
        "ref:semantic",
        "baguqeeraexamplecid0002",
        operation_id="op:2",
        expected_cas=first.cas_token,
    )
    assert store.get("ref:semantic") == second
    with pytest.raises(TypedOperationalStoreError, match="CAS"):
        store.append_reference(
            "ref:semantic",
            "baguqeeraexamplecid0003",
            operation_id="op:3",
            expected_cas=first.cas_token,
        )


def test_restart_reconciles_heads_and_outbox() -> None:
    store = TypedOperationalReferenceStore()
    store.append_reference("ref:a", "cid:a", operation_id="op:a")
    store.append_reference("ref:b", "cid:b", operation_id="op:b")
    restored = store.restart()
    assert restored.get("ref:a") is not None
    assert restored.get("ref:b") is not None
    assert restored.outbox_cursor == store.outbox_cursor
    assert [item.operation_id for item in restored.outbox()] == ["op:a", "op:b"]
    assert restored.single_writer is True
    assert restored.quack_qualified is False


def test_stale_worker_is_rejected() -> None:
    store = TypedOperationalReferenceStore()
    store.acquire_lease("writer:a")
    with pytest.raises(TypedOperationalStoreError, match="stale-worker"):
        store.append_reference(
            "ref:x",
            "cid:x",
            operation_id="op:x",
            writer_id="writer:b",
        )


def test_duplicate_completion_is_rejected() -> None:
    store = TypedOperationalReferenceStore()
    store.append_reference("ref:x", "cid:1", operation_id="op:dup")
    with pytest.raises(TypedOperationalStoreError, match="duplicate"):
        store.append_reference(
            "ref:y",
            "cid:2",
            operation_id="op:dup",
        )


def test_fence_mismatch_is_rejected() -> None:
    store = TypedOperationalReferenceStore()
    fence = store.acquire_lease("writer:a")
    with pytest.raises(TypedOperationalStoreError, match="fence"):
        store.append_reference(
            "ref:x",
            "cid:x",
            operation_id="op:x",
            writer_id="writer:a",
            fence=fence + 1,
        )


def test_single_writer_lease_excludes_second_holder() -> None:
    store = TypedOperationalReferenceStore()
    store.acquire_lease("writer:a")
    with pytest.raises(TypedOperationalStoreError, match="single-writer"):
        store.acquire_lease("writer:b")
    store.release_lease("writer:a")
    store.acquire_lease("writer:b")
    assert store.outbox(after=0) == ()
