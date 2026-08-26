"""LGCVF-101: typed operational references, CAS, leases, fences, restart.

Required evidence: restart, stale-worker, duplicate completion, fence,
single-writer, and outbox tests. Quack is not treated as qualified.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_repository import (
    TYPED_OPERATIONAL_STORE_INTERFACE,
    TYPED_OPERATIONAL_STORE_SCHEMA,
    TypedOperationalReferenceStore,
    TypedOperationalStoreError,
)


def test_append_only_cas_and_single_writer() -> None:
    store = TypedOperationalReferenceStore(writer_id="writer:a")
    assert store.INTERFACE == TYPED_OPERATIONAL_STORE_INTERFACE
    assert store.SCHEMA == TYPED_OPERATIONAL_STORE_SCHEMA
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
    assert restored.lease_holder is None


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


def test_outbox_cursor_pages_append_only_log() -> None:
    store = TypedOperationalReferenceStore(writer_id="writer:a")
    first = store.append_reference("ref:a", "cid:a", operation_id="op:a")
    second = store.append_reference("ref:b", "cid:b", operation_id="op:b")
    assert store.outbox_cursor == 2
    assert store.outbox() == (first, second)
    assert store.outbox(after=first.sequence) == (second,)
    assert store.outbox(after=store.outbox_cursor) == ()


def test_restart_expires_lease_and_preserves_fence_cas() -> None:
    store = TypedOperationalReferenceStore(writer_id="writer:a")
    fence = store.acquire_lease("writer:a")
    first = store.append_reference(
        "ref:head",
        "cid:1",
        operation_id="op:1",
        writer_id="writer:a",
        fence=fence,
    )
    restored = store.restart()
    assert restored.lease_holder is None
    assert restored.fence == fence
    assert restored.get("ref:head") == first
    advanced = restored.acquire_lease("writer:a")
    assert advanced == fence + 1
    with pytest.raises(TypedOperationalStoreError, match="duplicate"):
        restored.append_reference("ref:other", "cid:2", operation_id="op:1")
    with pytest.raises(TypedOperationalStoreError, match="CAS"):
        restored.append_reference(
            "ref:head",
            "cid:3",
            operation_id="op:2",
            expected_cas="stale-cas",
        )
    with pytest.raises(TypedOperationalStoreError, match="fence"):
        restored.append_reference(
            "ref:head",
            "cid:3",
            operation_id="op:3",
            expected_cas=first.cas_token,
            writer_id="writer:a",
            fence=fence,
        )
    second = restored.append_reference(
        "ref:head",
        "cid:3",
        operation_id="op:3",
        expected_cas=first.cas_token,
        writer_id="writer:a",
        fence=advanced,
    )
    assert restored.get("ref:head") == second


def test_durable_journal_survives_process_reopen(tmp_path: Path) -> None:
    directory = tmp_path / "operational-store"
    with TypedOperationalReferenceStore(
        writer_id="writer:a",
        directory=directory,
    ) as store:
        assert store.quack_qualified is False
        first = store.append_reference("ref:a", "cid:a", operation_id="op:a")
        store.append_reference(
            "ref:a",
            "cid:a2",
            operation_id="op:a2",
            expected_cas=first.cas_token,
        )
        store.append_reference("ref:b", "cid:b", operation_id="op:b")
        cursor = store.outbox_cursor
        fence = store.fence
        log_bytes = (directory / "operational-references.jsonl").read_bytes()
        assert log_bytes.count(b"\n") == 3

    with TypedOperationalReferenceStore(
        writer_id="writer:a",
        directory=directory,
    ) as restored:
        assert restored.quack_qualified is False
        assert restored.single_writer is True
        assert restored.lease_holder is None
        assert restored.outbox_cursor == cursor
        assert restored.fence == fence
        assert restored.get("ref:a") is not None
        assert restored.get("ref:a").cid == "cid:a2"
        assert [item.operation_id for item in restored.outbox()] == [
            "op:a",
            "op:a2",
            "op:b",
        ]
        with pytest.raises(TypedOperationalStoreError, match="duplicate"):
            restored.append_reference("ref:c", "cid:c", operation_id="op:a")
        with pytest.raises(TypedOperationalStoreError, match="CAS"):
            restored.append_reference("ref:a", "cid:a3", operation_id="op:a3")


def test_durable_restart_replays_journal_without_second_writer(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "operational-store"
    store = TypedOperationalReferenceStore(
        writer_id="writer:a",
        directory=directory,
    )
    store.append_reference("ref:a", "cid:a", operation_id="op:a")
    restored = store.restart()
    assert restored.directory == directory
    assert restored.get("ref:a") is not None
    assert restored.outbox_cursor == store.outbox_cursor
    assert restored.lease_holder is None
    with pytest.raises(TypedOperationalStoreError, match="single-writer"):
        TypedOperationalReferenceStore(writer_id="writer:b", directory=directory)
    store.close()
    with TypedOperationalReferenceStore(
        writer_id="writer:b",
        directory=directory,
    ) as successor:
        successor.append_reference("ref:b", "cid:b", operation_id="op:b")
        assert successor.get("ref:a") is not None
        assert successor.get("ref:b") is not None


def test_durable_single_writer_claim_excludes_second_holder(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "operational-store"
    holder = TypedOperationalReferenceStore(
        writer_id="writer:a",
        directory=directory,
    )
    with pytest.raises(TypedOperationalStoreError, match="single-writer"):
        TypedOperationalReferenceStore(writer_id="writer:b", directory=directory)
    holder.acquire_lease("writer:a")
    holder.release_lease("writer:a")
    holder.close()
    with TypedOperationalReferenceStore(
        writer_id="writer:b",
        directory=directory,
    ) as successor:
        successor.acquire_lease("writer:b")
        assert successor.outbox(after=0) == ()
        assert successor.quack_qualified is False
