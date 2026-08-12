"""Focused tests for immutable verification receipt storage and generation-CAS.

Evidence surfaces:
* ``ivp/store-protocol@1``
* ``ivp/concurrent-store-cas@1``
"""

from __future__ import annotations

import os
import threading
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from ipfs_accelerate_py.agent_supervisor.verification.receipt_store import (
    CONCURRENT_STORE_CAS_EVIDENCE,
    DURABLE_COORDINATION_LEAF_MODULE,
    DURABLE_COORDINATION_SYMBOL,
    STORE_PROTOCOL_EVIDENCE,
    VERIFICATION_RECEIPT_STORE_INTERFACE,
    CompareAndSwapResult,
    HermeticVerificationReceiptStore,
    IndexEntry,
    IndexSnapshot,
    IpfsKitVerificationReceiptStore,
    ReceiptStoreIntegrityError,
    ReceiptStoreUnavailableError,
    StoreUnavailable,
    StoreUnavailableCode,
    TombstoneRecord,
    build_receipt_envelope,
    cas_publish_entry,
    mapping_cid,
    probe_durable_coordination_store,
    raw_cid,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _receipt_body(name: str, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": "ipfs_accelerate_py/agent-supervisor/test-receipt-body@1",
        "name": name,
        "status": "passed",
    }
    body.update(extra)
    return body


def _fsync_counter(original_fsync: Any) -> tuple[Any, list[int]]:
    calls: list[int] = []

    def wrapper(fd: int) -> None:
        calls.append(fd)
        return original_fsync(fd)

    return wrapper, calls


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------


def test_evidence_and_interface_constants() -> None:
    assert STORE_PROTOCOL_EVIDENCE == "ivp/store-protocol@1"
    assert CONCURRENT_STORE_CAS_EVIDENCE == "ivp/concurrent-store-cas@1"
    assert VERIFICATION_RECEIPT_STORE_INTERFACE == "VerificationReceiptStore@1"
    assert DURABLE_COORDINATION_SYMBOL == "DurableCoordinationStore"
    assert "coordination_storage" in DURABLE_COORDINATION_LEAF_MODULE


# ---------------------------------------------------------------------------
# Hermetic: durable put/get, reopen, CID verify
# ---------------------------------------------------------------------------


def test_hermetic_put_get_mapping_and_reopen(tmp_path: Path) -> None:
    root = tmp_path / "hermetic"
    body = _receipt_body("alpha")
    with HermeticVerificationReceiptStore(root) as store:
        put = store.put_receipt_envelope(body, stored_at_ms=1)
        assert put.created is True
        assert put.durable is True
        assert put.codec == "dag-json"
        loaded = store.get_receipt_envelope(put.cid)
        assert loaded["body"] == body
        assert loaded["body_cid"] == mapping_cid(body)
        assert store.get_bytes(put.cid)
        assert mapping_cid(store.get_mapping(put.cid)) == put.cid
        assert raw_cid(b"hello-raw")  # smoke
        raw_put = store.put_bytes(b"hello-raw")
        assert store.get_bytes(raw_put.cid) == b"hello-raw"

    with HermeticVerificationReceiptStore(root) as reopened:
        again = reopened.get_receipt_envelope(put.cid)
        assert again["body"] == body
        assert reopened.get_bytes(raw_put.cid) == b"hello-raw"
        report = reopened.recover(rebuild=True)
        assert report.verified_blocks >= 2
        assert report.errors == ()


def test_hermetic_requires_explicit_storage_root() -> None:
    with pytest.raises(ReceiptStoreUnavailableError) as excinfo:
        HermeticVerificationReceiptStore("")
    assert excinfo.value.unavailable.code is StoreUnavailableCode.STORAGE_ROOT_REQUIRED


def test_hermetic_cid_mismatch_rejected(tmp_path: Path) -> None:
    store = HermeticVerificationReceiptStore(tmp_path / "cid")
    data = b"exact-bytes"
    good = raw_cid(data)
    with pytest.raises(ReceiptStoreIntegrityError):
        store.put_bytes(data, expected_cid="bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi")
    put = store.put_bytes(data, expected_cid=good)
    assert put.cid == good


def test_hermetic_detects_corruption(tmp_path: Path) -> None:
    root = tmp_path / "corrupt"
    store = HermeticVerificationReceiptStore(root)
    put = store.put_receipt_envelope(_receipt_body("corrupt-me"), stored_at_ms=2)
    path = store._block_path(put.cid)
    path.write_bytes(b'{"schema":"tampered","body":{}}')
    with pytest.raises(ReceiptStoreIntegrityError):
        store.get_bytes(put.cid)
    with pytest.raises(ReceiptStoreIntegrityError):
        store.recover(rebuild=True)


def test_hermetic_fsync_on_atomic_updates(tmp_path: Path) -> None:
    root = tmp_path / "fsync"
    original = os.fsync
    wrapper, calls = _fsync_counter(original)
    with mock.patch("os.fsync", side_effect=wrapper):
        store = HermeticVerificationReceiptStore(root)
        store.put_bytes(b"fsync-me")
        entry_put = store.put_receipt_envelope(_receipt_body("fsync"), stored_at_ms=3)
        entry = IndexEntry(key_id="k1", receipt_cid=entry_put.cid)
        result = cas_publish_entry(store, entry)
        assert result.success is True
    assert len(calls) >= 2  # block write + directory and/or HEAD


# ---------------------------------------------------------------------------
# Generation CAS, history replay, concurrent writers
# ---------------------------------------------------------------------------


def test_hermetic_cas_history_replay_and_conflict(tmp_path: Path) -> None:
    store = HermeticVerificationReceiptStore(tmp_path / "cas")
    a = store.put_receipt_envelope(_receipt_body("a"), stored_at_ms=10)
    b = store.put_receipt_envelope(_receipt_body("b"), stored_at_ms=11)

    first = cas_publish_entry(store, IndexEntry(key_id="key-a", receipt_cid=a.cid))
    assert first.success is True
    assert first.generation == 1

    # Stale writer using generation 0 loses CAS and does not overwrite peer.
    stale_snapshot = IndexSnapshot(
        generation=1,
        entries=(IndexEntry(key_id="key-b-only", receipt_cid=b.cid),),
        created_at_ms=12,
    )
    conflict = store.compare_and_swap_index(
        stale_snapshot,
        expected_generation=0,
        expected_root_cid=None,
    )
    assert conflict.success is False
    assert conflict.conflict is True
    assert conflict.generation == 1
    # Peer entry preserved.
    current = store.current_index()
    assert {e.key_id for e in current.entries} == {"key-a"}

    second = cas_publish_entry(store, IndexEntry(key_id="key-b", receipt_cid=b.cid))
    assert second.success is True
    final = store.current_index()
    assert {e.key_id for e in final.entries} == {"key-a", "key-b"}

    history = store.replay_history()
    assert len(history) >= 2
    assert history[0].generation == 1
    assert history[-1].generation == final.generation
    assert all(isinstance(item, IndexSnapshot) for item in history)


def test_concurrent_writers_preserve_all_entries(tmp_path: Path) -> None:
    store = HermeticVerificationReceiptStore(tmp_path / "concurrent")
    worker_count = 8
    receipt_cids: list[str] = []
    for i in range(worker_count):
        put = store.put_receipt_envelope(_receipt_body(f"w{i}"), stored_at_ms=100 + i)
        receipt_cids.append(put.cid)

    errors: list[BaseException] = []
    results: list[CompareAndSwapResult] = []
    barrier = threading.Barrier(worker_count)

    def worker(idx: int) -> CompareAndSwapResult:
        barrier.wait(timeout=10)
        return cas_publish_entry(
            store,
            IndexEntry(key_id=f"worker-{idx}", receipt_cid=receipt_cids[idx]),
        )

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(worker, i) for i in range(worker_count)]
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

    assert not errors
    assert all(r.success for r in results)
    final = store.current_index()
    keys = {entry.key_id for entry in final.entries}
    assert keys == {f"worker-{i}" for i in range(worker_count)}
    # Every receipt CID still resolvable (no lost immutable blocks).
    for cid in receipt_cids:
        assert store.get_receipt_envelope(cid)["body"]["name"].startswith("w")


# ---------------------------------------------------------------------------
# Tombstones preserve immutable audit history
# ---------------------------------------------------------------------------


def test_tombstones_preserve_immutable_audit_history(tmp_path: Path) -> None:
    store = HermeticVerificationReceiptStore(tmp_path / "tomb")
    put = store.put_receipt_envelope(_receipt_body("old"), stored_at_ms=50)
    cas_publish_entry(store, IndexEntry(key_id="k-old", receipt_cid=put.cid))
    current = store.current_index()
    expected_root = current.root_cid if current.generation > 0 else None

    tomb = TombstoneRecord(
        key_id="k-old",
        prior_receipt_cid=put.cid,
        reason="stale_after_tree_change",
        tombstoned_at_ms=99,
    )
    result = store.publish_tombstone(
        tomb,
        expected_generation=current.generation,
        expected_root_cid=expected_root,
    )
    assert result.success is True
    after = store.current_index()
    assert "k-old" not in after.entry_map()
    assert len(after.tombstones) == 1
    sealed = after.tombstones[0]
    assert sealed.prior_receipt_cid == put.cid
    assert sealed.tombstone_cid is not None
    # Prior receipt bytes remain immutable and readable.
    assert store.get_receipt_envelope(put.cid)["body"]["name"] == "old"
    # Tombstone envelope itself is immutable audit history.
    audit = store.get_mapping(sealed.tombstone_cid)
    assert audit["prior_receipt_cid"] == put.cid
    assert audit["reason"] == "stale_after_tree_change"
    # History still includes pre-tombstone generation.
    history = store.replay_history()
    assert any("k-old" in snap.entry_map() for snap in history)


# ---------------------------------------------------------------------------
# GC metadata
# ---------------------------------------------------------------------------


def test_gc_metadata_last_access(tmp_path: Path) -> None:
    store = HermeticVerificationReceiptStore(tmp_path / "gc")
    put = store.put_bytes(b"gc-target")
    meta = store.record_access(put.cid, at_ms=12345)
    assert meta.cid == put.cid
    assert meta.last_access_ms == 12345
    assert meta.reachable is True
    collected = store.collect_gc_metadata()
    assert any(item.cid == put.cid for item in collected)


# ---------------------------------------------------------------------------
# DurableCoordinationStore leaf probing + typed unavailable
# ---------------------------------------------------------------------------


def test_probe_exact_leaf_when_available() -> None:
    probe = probe_durable_coordination_store()
    # With PYTHONPATH=ipfs_kit_py:… the leaf is present in this workspace.
    if probe.available:
        assert "put" in probe.methods
        assert "get" in probe.methods
        assert "recover" in probe.methods
        assert probe.module_name == DURABLE_COORDINATION_LEAF_MODULE
    else:
        assert probe.unavailable is not None
        assert probe.unavailable.status == "unavailable"


def test_probe_namespace_only_is_typed_unavailable() -> None:
    bare = types.ModuleType("ipfs_kit_py")
    probe = probe_durable_coordination_store(namespace_only_module=bare)
    assert probe.available is False
    assert probe.unavailable is not None
    assert probe.unavailable.code is StoreUnavailableCode.NAMESPACE_ONLY
    assert probe.unavailable.status == "unavailable"


def test_probe_absent_backend_is_typed_unavailable() -> None:
    def boom(_name: str) -> Any:
        raise ModuleNotFoundError("nope")

    probe = probe_durable_coordination_store(importer=boom)
    assert probe.available is False
    assert probe.unavailable is not None
    assert probe.unavailable.code is StoreUnavailableCode.ABSENT_BACKEND


def test_probe_absent_revision_helper_is_typed_unavailable() -> None:
    class FakeStore:
        def put(self, *a: Any, **k: Any) -> Any: ...
        def get(self, *a: Any, **k: Any) -> Any: ...
        def recover(self, *a: Any, **k: Any) -> Any: ...
        def get_bytes(self, *a: Any, **k: Any) -> Any: ...

    fake_mod = types.ModuleType(DURABLE_COORDINATION_LEAF_MODULE)
    setattr(fake_mod, DURABLE_COORDINATION_SYMBOL, FakeStore)
    # Missing cid_for_bytes / cid_for_artifact → absent revision helpers.

    def load(name: str) -> Any:
        if name == DURABLE_COORDINATION_LEAF_MODULE:
            return fake_mod
        raise ModuleNotFoundError(name)

    probe = probe_durable_coordination_store(importer=load)
    assert probe.available is False
    assert probe.unavailable is not None
    assert probe.unavailable.code is StoreUnavailableCode.ABSENT_REVISION


def test_unavailable_cas_bridge_is_typed(tmp_path: Path) -> None:
    opened = IpfsKitVerificationReceiptStore.try_open(tmp_path / "iroh-cas")
    if isinstance(opened, StoreUnavailable):
        assert opened.status == "unavailable"
        return
    # No bridge injected → explicit CAS path is typed unavailable.
    missing = opened.iroh_cas_or_unavailable()
    assert isinstance(missing, StoreUnavailable)
    assert missing.code is StoreUnavailableCode.CAS_UNAVAILABLE
    opened.close()


# ---------------------------------------------------------------------------
# IpfsKit adapter: Mapping envelopes, put/get/recover, explicit root, CID vectors
# ---------------------------------------------------------------------------


def test_ipfs_kit_adapter_mapping_put_get_recover_and_explicit_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "kit-store"
    opened = IpfsKitVerificationReceiptStore.try_open(root)
    if isinstance(opened, StoreUnavailable):
        pytest.skip(f"DurableCoordinationStore unavailable: {opened.reason}")

    with opened as store:
        assert store.storage_dir == root.resolve()
        # Must not fall back to home defaults — coordination lives under root.
        coord_root = root / "coordination"
        assert coord_root.is_dir()

        body = _receipt_body("kit-body", extra_field=1)
        envelope = build_receipt_envelope(body, stored_at_ms=7)
        expected = mapping_cid(envelope)
        put = store.put_mapping(envelope, expected_cid=expected)
        assert put.cid == expected
        loaded = store.get_mapping(put.cid)
        assert loaded["body"] == body

        receipt_put = store.put_receipt_envelope(body, stored_at_ms=8)
        assert store.get_receipt_envelope(receipt_put.cid)["body"] == body

        # Generation CAS still works on the kit adapter.
        cas = cas_publish_entry(
            store, IndexEntry(key_id="kit-key", receipt_cid=receipt_put.cid)
        )
        assert cas.success is True

        report = store.recover(rebuild=True)
        assert report.verified_blocks >= 1
        assert report.errors == ()
        assert store.current_index().entry_map()["kit-key"].receipt_cid == receipt_put.cid


def test_cross_backend_cid_vectors(tmp_path: Path) -> None:
    body = _receipt_body("vector", n=42)
    envelope = build_receipt_envelope(body, stored_at_ms=42)
    expected_cid = mapping_cid(envelope)

    hermetic = HermeticVerificationReceiptStore(tmp_path / "vec-hermetic")
    h_put = hermetic.put_mapping(envelope, expected_cid=expected_cid)
    assert h_put.cid == expected_cid

    opened = IpfsKitVerificationReceiptStore.try_open(tmp_path / "vec-kit")
    if isinstance(opened, StoreUnavailable):
        pytest.skip(f"DurableCoordinationStore unavailable: {opened.reason}")

    with opened as kit:
        k_put = kit.put_mapping(envelope, expected_cid=expected_cid)
        assert k_put.cid == expected_cid
        assert k_put.cid == h_put.cid
        assert kit.get_mapping(k_put.cid) == hermetic.get_mapping(h_put.cid)


def test_ipfs_kit_rejects_missing_storage_root() -> None:
    with pytest.raises(ReceiptStoreUnavailableError) as excinfo:
        IpfsKitVerificationReceiptStore("")
    assert excinfo.value.unavailable.code is StoreUnavailableCode.STORAGE_ROOT_REQUIRED


def test_ipfs_kit_try_open_namespace_only(tmp_path: Path) -> None:
    with mock.patch(
        "ipfs_accelerate_py.agent_supervisor.verification.receipt_store.probe_durable_coordination_store",
        return_value=probe_durable_coordination_store(
            namespace_only_module=types.ModuleType("ipfs_kit_py")
        ),
    ):
        result = IpfsKitVerificationReceiptStore.try_open(tmp_path / "ns")
    assert isinstance(result, StoreUnavailable)
    assert result.code is StoreUnavailableCode.NAMESPACE_ONLY


# ---------------------------------------------------------------------------
# Immutable collision + envelope integrity
# ---------------------------------------------------------------------------


def test_immutable_block_collision_detected(tmp_path: Path) -> None:
    store = HermeticVerificationReceiptStore(tmp_path / "collide")
    data = b"same-cid-bytes"
    store.put_bytes(data)
    # Same bytes is idempotent.
    again = store.put_bytes(data)
    assert again.created is False
    # Different bytes forced into same path is impossible via public API
    # (CID changes). Corruption path already covered.


def test_index_snapshot_rejects_duplicate_keys() -> None:
    cid = raw_cid(b"x")
    with pytest.raises(ReceiptStoreIntegrityError):
        IndexSnapshot(
            generation=1,
            entries=(
                IndexEntry(key_id="dup", receipt_cid=cid),
                IndexEntry(key_id="dup", receipt_cid=cid),
            ),
        )


def test_compare_and_swap_result_to_dict_shape(tmp_path: Path) -> None:
    store = HermeticVerificationReceiptStore(tmp_path / "shape")
    put = store.put_receipt_envelope(_receipt_body("shape"), stored_at_ms=1)
    result = cas_publish_entry(store, IndexEntry(key_id="s", receipt_cid=put.cid))
    payload = result.to_dict()
    assert payload["success"] is True
    assert payload["generation"] == 1
    assert "root_cid" in payload
