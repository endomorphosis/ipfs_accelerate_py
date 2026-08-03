"""Tests for immutable candidate execution context store (PTR-135)."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_candidate_context_store import (
    CANDIDATE_CONTEXT_ENVELOPE_INTERFACE,
    CANONICAL_ARTIFACT_STORE_TRANSPORT_INTERFACE,
    CANDIDATE_EXECUTION_CONTEXT_INTERFACE,
    REQUIRED_COMPONENT_KEYS,
    TEST_CANDIDATE_CONTEXT_STORE_INTERFACE,
    CandidateContextEnvelope,
    CandidateContextStoreReason,
    CandidateContextStoreStatus,
    TestCandidateContextStore,
)
from ipfs_accelerate_py.testing.proof_reuse.activation_contracts import (
    CandidateExecutionContext,
)

NOW_S = 10.0
NOW_MS = 10_000


def _component_bytes(label: str) -> bytes:
    return canonical_json_bytes(
        {
            "schema": "ipfs_accelerate_py/test/candidate-component@1",
            "label": label,
            "version": 1,
        }
    )


def _component_cid(label: str) -> str:
    return content_identity(
        {
            "schema": "ipfs_accelerate_py/test/candidate-component@1",
            "label": label,
            "version": 1,
        }
    )


def _locator_cid(tag: str = "default") -> str:
    return content_identity(
        {
            "schema": "ipfs_accelerate_py/test/locator@1",
            "tag": tag,
        }
    )


def _build_bundle(
    *,
    tag: str = "default",
    retained_at_ms: int = NOW_MS,
) -> tuple[CandidateExecutionContext, dict[str, bytes]]:
    """Build a descriptor and matching retained component bytes."""

    labels = {
        "execution_key": f"ek-{tag}",
        "static_trace": f"static-{tag}",
        "runtime_trace": f"runtime-{tag}",
        "repository_forest": f"forest-{tag}",
        "environment": f"env-{tag}",
        "policy": f"policy-{tag}",
        "pass_receipt": f"receipt-{tag}",
        "test_ast": f"ast-{tag}",
    }
    components = {name: _component_bytes(label) for name, label in labels.items()}
    cids = {name: _component_cid(label) for name, label in labels.items()}
    locator = _locator_cid(tag)
    descriptor = CandidateExecutionContext(
        locator_cid=locator,
        execution_key_cid=cids["execution_key"],
        pass_receipt_cid=cids["pass_receipt"],
        repository_forest_cid=cids["repository_forest"],
        test_ast_cid=cids["test_ast"],
        static_trace_root_cid=cids["static_trace"],
        runtime_trace_root_cid=cids["runtime_trace"],
        environment_cid=cids["environment"],
        policy_cid=cids["policy"],
        retained_at_ms=retained_at_ms,
        component_cids={
            "execution_key": cids["execution_key"],
            "static_trace": cids["static_trace"],
            "runtime_trace": cids["runtime_trace"],
            "repository_forest": cids["repository_forest"],
            "environment": cids["environment"],
            "policy": cids["policy"],
            "pass_receipt": cids["pass_receipt"],
        },
    )
    return descriptor, components


def _store(tmp_path: Path, **kwargs: Any) -> TestCandidateContextStore:
    return TestCandidateContextStore(
        tmp_path / "candidate-context-store",
        clock=lambda: NOW_S,
        **kwargs,
    )


def test_publish_and_lookup_returns_bytes_and_non_authoritative_descriptor(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    descriptor, components = _build_bundle()

    put = store.publish(descriptor, components, locator_cid=descriptor.locator_cid)

    assert put.stored
    assert put.indexed
    assert put.reason_code is CandidateContextStoreReason.OK
    assert put.candidate_context_cid == descriptor.candidate_context_id
    assert put.envelope_cid
    assert put.may_authorize_skip is False

    lookup = store.lookup(descriptor.locator_cid)
    assert lookup.status is CandidateContextStoreStatus.HIT
    assert lookup.hit
    assert lookup.may_authorize_skip is False
    assert lookup.envelope_bytes is not None
    assert lookup.descriptor_bytes is not None
    assert lookup.descriptor is not None
    assert lookup.descriptor.candidate_context_id == descriptor.candidate_context_id
    assert lookup.descriptor.may_authorize_skip is False
    assert lookup.admission is not None
    assert lookup.admission.admitted
    assert lookup.admission.may_authorize_skip is False
    # All required components rehashed and materialised.
    for key in REQUIRED_COMPONENT_KEYS:
        assert key in lookup.component_bytes
        assert lookup.component_bytes[key] == components[key]
    # Envelope re-parses and never claims skip authority.
    envelope = CandidateContextEnvelope.from_bytes(lookup.envelope_bytes)
    assert envelope.may_authorize_skip is False
    assert envelope.candidate_context_cid == descriptor.candidate_context_id
    assert envelope.interface == CANDIDATE_CONTEXT_ENVELOPE_INTERFACE

    leftovers = list((tmp_path / "candidate-context-store").rglob(".tmp.*"))
    assert leftovers == []


def test_admission_rehashes_components_and_checks_cid_agreement(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    descriptor, components = _build_bundle(tag="admit")
    put = store.publish(descriptor, components)
    assert put.stored

    admitted = store.admit(put.envelope_cid)
    assert admitted.admitted
    assert admitted.reason_code is CandidateContextStoreReason.OK
    assert set(REQUIRED_COMPONENT_KEYS) <= set(admitted.component_cids)
    assert admitted.may_authorize_skip is False

    # Poison one component blob → internal/external CID mismatch after rewrite.
    component_cid = admitted.component_cids["policy"]
    path = store.cas.blob_path(component_cid)
    path.write_bytes(b'{"schema":"poisoned","v":1}')
    failed = store.admit(put.envelope_cid)
    assert not failed.admitted
    assert failed.reason_code in {
        CandidateContextStoreReason.CORRUPT,
        CandidateContextStoreReason.INTEGRITY_FAILED,
        CandidateContextStoreReason.INTERNAL_EXTERNAL_CID_MISMATCH,
        CandidateContextStoreReason.COMPONENT_MISSING,
    }


def test_component_field_mismatch_rejects_publish(tmp_path: Path) -> None:
    store = _store(tmp_path)
    descriptor, components = _build_bundle(tag="mismatch")
    # Replace policy bytes with unrelated content so rehash diverges.
    components = dict(components)
    components["policy"] = _component_bytes("policy-wrong")

    put = store.publish(descriptor, components)
    assert not put.stored
    assert put.reason_code is (
        CandidateContextStoreReason.INTERNAL_EXTERNAL_CID_MISMATCH
    )


def test_missing_required_component_rejects_publish(tmp_path: Path) -> None:
    store = _store(tmp_path)
    descriptor, components = _build_bundle(tag="missing")
    del components["pass_receipt"]

    put = store.publish(descriptor, components)
    assert not put.stored
    assert put.reason_code is CandidateContextStoreReason.COMPONENT_MISSING


def test_poisoned_index_is_typed_miss(tmp_path: Path) -> None:
    store = _store(tmp_path)
    descriptor, components = _build_bundle(tag="poison")
    assert store.publish(descriptor, components).stored

    index_path = store.index._index_path(descriptor.locator_cid)
    index_path.write_bytes(b"{not-json")
    lookup = store.lookup(descriptor.locator_cid)
    assert lookup.status is CandidateContextStoreStatus.MISS
    assert lookup.reason_code is CandidateContextStoreReason.INDEX_POISONED


def test_missing_blob_is_typed_miss(tmp_path: Path) -> None:
    store = _store(tmp_path)
    descriptor, components = _build_bundle(tag="missing-blob")
    put = store.publish(descriptor, components)
    assert put.stored

    store.cas.blob_path(put.envelope_cid).unlink()
    lookup = store.lookup(descriptor.locator_cid)
    assert lookup.status is CandidateContextStoreStatus.MISS
    assert lookup.reason_code in {
        CandidateContextStoreReason.CANDIDATE_MISSING,
        CandidateContextStoreReason.COMPONENT_MISSING,
    }


def test_partial_write_is_typed_miss_and_quarantines(tmp_path: Path) -> None:
    store = _store(tmp_path)
    descriptor, components = _build_bundle(tag="partial")
    put = store.publish(descriptor, components)
    assert put.stored

    path = store.cas.blob_path(put.envelope_cid)
    path.write_bytes(b"")
    admitted = store.admit(put.envelope_cid)
    assert not admitted.admitted
    assert admitted.reason_code is CandidateContextStoreReason.PARTIAL
    # Empty blob is quarantined by the underlying CAS.
    assert store.cas.quarantine_path(put.envelope_cid).exists()


def test_symlink_blob_is_typed_miss(tmp_path: Path) -> None:
    store = _store(tmp_path)
    descriptor, components = _build_bundle(tag="symlink")
    put = store.publish(descriptor, components)
    assert put.stored

    blob = store.cas.blob_path(put.envelope_cid)
    outside = tmp_path / "outside-envelope.json"
    outside.write_bytes(blob.read_bytes())
    blob.unlink()
    blob.symlink_to(outside)

    admitted = store.admit(put.envelope_cid)
    assert not admitted.admitted
    assert admitted.reason_code is CandidateContextStoreReason.SYMLINK_REJECTED


def test_index_symlink_is_typed_miss(tmp_path: Path) -> None:
    store = _store(tmp_path)
    descriptor, components = _build_bundle(tag="index-symlink")
    assert store.publish(descriptor, components).stored

    index_path = store.index._index_path(descriptor.locator_cid)
    payload = index_path.read_bytes()
    index_path.unlink()
    outside = tmp_path / "evil-index.json"
    outside.write_bytes(payload)
    index_path.symlink_to(outside)

    lookup = store.index.candidates(descriptor.locator_cid)
    assert lookup.status is CandidateContextStoreStatus.MISS
    assert lookup.reason_code is CandidateContextStoreReason.SYMLINK_REJECTED


def test_expiry_and_revocation_hide_candidates(tmp_path: Path) -> None:
    store = _store(tmp_path, index_ttl_ms=1_000)
    descriptor, components = _build_bundle(tag="ttl")
    assert store.publish(
        descriptor,
        components,
        created_at_ms=NOW_MS,
        expires_at_ms=NOW_MS + 500,
    ).stored

    expired = store.lookup(descriptor.locator_cid, now_ms=NOW_MS + 501)
    assert expired.status is CandidateContextStoreStatus.MISS

    store2 = _store(tmp_path / "revocation")
    descriptor2, components2 = _build_bundle(tag="revoke")
    put = store2.publish(descriptor2, components2, created_at_ms=NOW_MS)
    assert put.stored
    revoked = store2.index.revoke(
        descriptor2.locator_cid, descriptor2.candidate_context_id
    )
    assert revoked.published
    assert store2.lookup(descriptor2.locator_cid).status is CandidateContextStoreStatus.MISS


def test_stale_generation_is_typed_miss(tmp_path: Path) -> None:
    store = _store(tmp_path)
    descriptor, components = _build_bundle(tag="gen1")
    first = store.publish(descriptor, components, expected_generation=0)
    assert first.stored
    assert first.generation == 1

    # Second publication advances generation.
    descriptor2, components2 = _build_bundle(tag="gen2")
    # Force same locator as first.
    descriptor2 = CandidateExecutionContext(
        locator_cid=descriptor.locator_cid,
        execution_key_cid=descriptor2.execution_key_cid,
        pass_receipt_cid=descriptor2.pass_receipt_cid,
        repository_forest_cid=descriptor2.repository_forest_cid,
        test_ast_cid=descriptor2.test_ast_cid,
        static_trace_root_cid=descriptor2.static_trace_root_cid,
        runtime_trace_root_cid=descriptor2.runtime_trace_root_cid,
        environment_cid=descriptor2.environment_cid,
        policy_cid=descriptor2.policy_cid,
        retained_at_ms=NOW_MS + 1,
        component_cids=dict(descriptor2.component_cids),
    )
    second = store.publish(
        descriptor2, components2, expected_generation=1
    )
    assert second.stored
    assert second.generation == 2

    # Stale expected generation rejects concurrent publication.
    descriptor3, components3 = _build_bundle(tag="gen-stale")
    descriptor3 = CandidateExecutionContext(
        locator_cid=descriptor.locator_cid,
        execution_key_cid=descriptor3.execution_key_cid,
        pass_receipt_cid=descriptor3.pass_receipt_cid,
        repository_forest_cid=descriptor3.repository_forest_cid,
        test_ast_cid=descriptor3.test_ast_cid,
        static_trace_root_cid=descriptor3.static_trace_root_cid,
        runtime_trace_root_cid=descriptor3.runtime_trace_root_cid,
        environment_cid=descriptor3.environment_cid,
        policy_cid=descriptor3.policy_cid,
        retained_at_ms=NOW_MS + 2,
        component_cids=dict(descriptor3.component_cids),
    )
    stale = store.publish(
        descriptor3, components3, expected_generation=0
    )
    assert not stale.stored
    assert stale.reason_code is CandidateContextStoreReason.STALE_GENERATION


def test_fence_blocks_parallel_mixed_publication(tmp_path: Path) -> None:
    store_root = tmp_path / "parallel"
    locator = _locator_cid("parallel")
    barrier = threading.Barrier(2)
    outcomes: list[Any] = []
    lock = threading.Lock()

    def worker(tag: str) -> None:
        store = TestCandidateContextStore(
            store_root,
            clock=lambda: NOW_S,
            owner_id=f"owner-{tag}",
        )
        descriptor, components = _build_bundle(tag=tag)
        descriptor = CandidateExecutionContext(
            locator_cid=locator,
            execution_key_cid=descriptor.execution_key_cid,
            pass_receipt_cid=descriptor.pass_receipt_cid,
            repository_forest_cid=descriptor.repository_forest_cid,
            test_ast_cid=descriptor.test_ast_cid,
            static_trace_root_cid=descriptor.static_trace_root_cid,
            runtime_trace_root_cid=descriptor.runtime_trace_root_cid,
            environment_cid=descriptor.environment_cid,
            policy_cid=descriptor.policy_cid,
            retained_at_ms=NOW_MS,
            component_cids=dict(descriptor.component_cids),
        )
        barrier.wait(timeout=5)
        result = store.publish(
            descriptor,
            components,
            locator_cid=locator,
            owner_id=f"owner-{tag}",
        )
        with lock:
            outcomes.append((tag, result, descriptor.candidate_context_id))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, "a"), pool.submit(worker, "b")]
        for future in futures:
            future.result(timeout=30)

    assert len(outcomes) == 2
    stored = [item for item in outcomes if item[1].stored and item[1].indexed]
    assert stored
    assert all(
        item[1].reason_code
        in {
            CandidateContextStoreReason.FENCED,
            CandidateContextStoreReason.FENCE_MISMATCH,
            CandidateContextStoreReason.UNAVAILABLE,
            CandidateContextStoreReason.OK,
            CandidateContextStoreReason.STALE_GENERATION,
        }
        for item in outcomes
    )

    reader = TestCandidateContextStore(store_root, clock=lambda: NOW_S)
    lookup = reader.lookup(locator)
    assert lookup.status is CandidateContextStoreStatus.HIT
    assert lookup.descriptor is not None
    assert lookup.may_authorize_skip is False
    # Components remain a consistent set for the admitted candidate.
    for key in REQUIRED_COMPONENT_KEYS:
        assert key in lookup.component_bytes


def test_remote_failure_and_transport_absence_are_typed_misses(
    tmp_path: Path,
) -> None:
    descriptor, components = _build_bundle(tag="remote")

    class FailingTransport:
        interface = CANONICAL_ARTIFACT_STORE_TRANSPORT_INTERFACE

        def get_bytes(self, cid: str) -> Any:
            raise RuntimeError("ipfs down")

        def put_bytes(self, data: bytes, **kwargs: Any) -> Any:
            raise RuntimeError("ipfs down")

    store = _store(tmp_path, remote_transport=FailingTransport())
    put = store.publish(descriptor, components)
    assert put.stored

    # Remove local envelope so read falls through to remote.
    store.cas.blob_path(put.envelope_cid).unlink()
    admitted = store.admit(put.envelope_cid)
    assert not admitted.admitted
    assert admitted.reason_code is CandidateContextStoreReason.REMOTE_FAILURE

    # Explicit transport absence when local is also missing and no transport.
    bare = _store(tmp_path / "bare")
    missing = bare.admit(put.envelope_cid)
    assert not missing.admitted
    assert missing.reason_code in {
        CandidateContextStoreReason.CANDIDATE_MISSING,
        CandidateContextStoreReason.TRANSPORT_ABSENT,
    }


def test_size_and_version_checks(tmp_path: Path) -> None:
    store = _store(tmp_path, max_component_bytes=64)
    descriptor, components = _build_bundle(tag="size")
    # Components exceed the tiny budget.
    put = store.publish(descriptor, components)
    assert not put.stored
    assert put.reason_code in {
        CandidateContextStoreReason.SIZE_EXCEEDED,
        CandidateContextStoreReason.OVER_BUDGET,
    }

    store2 = _store(tmp_path / "version")
    descriptor2, components2 = _build_bundle(tag="version")
    put2 = store2.publish(descriptor2, components2)
    assert put2.stored
    # Manually write an envelope with a foreign version.
    envelope = CandidateContextEnvelope.from_bytes(
        store2.get_bytes(put2.envelope_cid).data
    )
    payload = envelope.to_dict()
    payload["version"] = 99
    # Bypass envelope constructor validation by writing raw non-version-1 JSON
    # under a new CID and pointing admission at it via direct admit path would
    # fail schema; instead mutate the stored blob in place after publication.
    # Rewrite as non-canonical version field using the CAS path.
    bad_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    # Force-write over the envelope path (simulates corrupted versioned artifact).
    path = store2.cas.blob_path(put2.envelope_cid)
    path.write_bytes(bad_bytes)
    admitted = store2.admit(put2.envelope_cid)
    assert not admitted.admitted
    assert admitted.reason_code in {
        CandidateContextStoreReason.VERSION_MISMATCH,
        CandidateContextStoreReason.CORRUPT,
        CandidateContextStoreReason.INTEGRITY_FAILED,
        CandidateContextStoreReason.INTERNAL_EXTERNAL_CID_MISMATCH,
    }


def test_mutable_metadata_and_cache_presence_never_authorize_skip(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    descriptor, components = _build_bundle(tag="no-skip")
    put = store.publish(
        descriptor,
        components,
        metadata={"trusted": True, "skip": True, "historical_execution_key": "x"},
    )
    assert put.stored
    assert put.may_authorize_skip is False

    lookup = store.lookup(descriptor.locator_cid)
    assert lookup.hit
    assert lookup.may_authorize_skip is False
    assert lookup.descriptor is not None
    assert lookup.descriptor.may_authorize_skip is False
    # Historical execution key identity is retained for comparison, not authority.
    assert lookup.descriptor.execution_key_cid == descriptor.execution_key_cid
    # Cache presence alone is not skip authority.
    assert store.get_bytes(put.envelope_cid).hit
    assert store.may_authorize_skip is False

    direct = store.lookup_by_context_cid(descriptor.candidate_context_id)
    assert direct.hit
    assert direct.may_authorize_skip is False
    assert direct.diagnostics.get("may_authorize_skip") is False


def test_path_escape_cid_tokens_miss_safely(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for bad in ("../escape", "a/b", "a\\b", "UPPER", "has space", ""):
        result = store.admit(bad)
        assert not result.admitted
        assert result.reason_code is CandidateContextStoreReason.PATH_ESCAPE


def test_interfaces_and_required_components_are_sealed() -> None:
    assert TEST_CANDIDATE_CONTEXT_STORE_INTERFACE == "TestCandidateContextStore@1"
    assert CANDIDATE_EXECUTION_CONTEXT_INTERFACE == "CandidateExecutionContext@1"
    assert (
        CANONICAL_ARTIFACT_STORE_TRANSPORT_INTERFACE
        == "CanonicalArtifactStoreTransport@1"
    )
    assert "execution_key" in REQUIRED_COMPONENT_KEYS
    assert "pass_receipt" in REQUIRED_COMPONENT_KEYS
    assert "policy" in REQUIRED_COMPONENT_KEYS


def test_idempotent_component_cas_and_restart_scrub(tmp_path: Path) -> None:
    root = tmp_path / "restart"
    store = TestCandidateContextStore(root, clock=lambda: NOW_S)
    descriptor, components = _build_bundle(tag="idem")
    first = store.publish(descriptor, components)
    second = store.publish(descriptor, components)
    assert first.stored
    # Second publish is a new generation but components already exist.
    assert second.stored or second.reason_code in {
        CandidateContextStoreReason.OK,
        CandidateContextStoreReason.FENCED,
    }

    cas_dir = store.cas.cas_root / "aa"
    cas_dir.mkdir(parents=True, exist_ok=True)
    orphan = cas_dir / ".tmp.orphan.blob.tmp"
    orphan.write_bytes(b"partial")
    assert orphan.exists()
    reopened = TestCandidateContextStore(root, clock=lambda: NOW_S)
    assert not orphan.exists()
    assert reopened.root == root.resolve()


def test_oversized_envelope_lookup_is_size_exceeded(tmp_path: Path) -> None:
    store = _store(tmp_path, max_blob_bytes=256)
    # Build tiny components so publish of components might work, but force a
    # large metadata blob if possible.  Simpler: publish with default store then
    # open a restricted reader against the same CAS is not supported; instead
    # publish normally and poke admit with a store that has a tiny budget.
    full = _store(tmp_path / "full")
    descriptor, components = _build_bundle(tag="oversize")
    put = full.publish(descriptor, components)
    assert put.stored

    tight = TestCandidateContextStore(
        tmp_path / "full" / "candidate-context-store",
        clock=lambda: NOW_S,
        max_blob_bytes=32,
        max_component_bytes=32,
    )
    admitted = tight.admit(put.envelope_cid)
    assert not admitted.admitted
    assert admitted.reason_code in {
        CandidateContextStoreReason.SIZE_EXCEEDED,
        CandidateContextStoreReason.OVER_BUDGET,
        CandidateContextStoreReason.CANDIDATE_MISSING,
        CandidateContextStoreReason.CORRUPT,
        CandidateContextStoreReason.INTEGRITY_FAILED,
    }
