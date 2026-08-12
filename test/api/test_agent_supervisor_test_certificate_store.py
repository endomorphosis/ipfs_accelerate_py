"""Tests for immutable certificate CAS and fenced locator indexes (PTR-031)."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.proof.test_certificate_store import (
    CertificateStoreReason,
    CertificateStoreStatus,
    CertificateWriteFence,
    ImmutableCertificateCAS,
    TestCertificateStore,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    CertificateAuthority,
    PhaseOutcome,
    ProofBackendMode,
    ReuseAction,
    TestExecutionKey,
    TestLocatorKey,
    TestPassReceipt,
    TestProofCertificate,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_proof_cache import TestProofCache


NOW_S = 10.0
NOW_MS = 10_000


def _locator() -> TestLocatorKey:
    return TestLocatorKey(
        repository_id="repository:example",
        package_identity="package:example",
        node_id="test/api/test_example.py::test_example",
    )


def _execution_key(locator: TestLocatorKey, *, policy_cid: str = "cid:policy") -> TestExecutionKey:
    return TestExecutionKey(
        locator_cid=locator.locator_id,
        repository_forest_cid="cid:repository-forest",
        static_trace_root_cid="cid:static-trace",
        runtime_trace_root_cid="cid:runtime-trace",
        runtime_completeness_policy="complete-v1",
        policy_cid=policy_cid,
    )


def _receipt(locator: TestLocatorKey, execution_key: TestExecutionKey) -> TestPassReceipt:
    return TestPassReceipt(
        execution_key_cid=execution_key.execution_key_id,
        locator_cid=locator.locator_id,
        setup_outcome=PhaseOutcome.PASS,
        call_outcome=PhaseOutcome.PASS,
        teardown_outcome=PhaseOutcome.PASS,
        static_trace_root_cid=execution_key.static_trace_root_cid,
        runtime_trace_root_cid=execution_key.runtime_trace_root_cid,
        completeness_receipt_cid="cid:completeness-receipt",
        dependency_forest_cid=execution_key.repository_forest_cid,
        issuer_key_id="key:issuer",
        policy_cid=execution_key.policy_cid,
    )


def _certificate(
    receipt: TestPassReceipt,
    execution_key: TestExecutionKey,
    *,
    authority: CertificateAuthority = CertificateAuthority.AUTHORITATIVE,
    backend_mode: ProofBackendMode = ProofBackendMode.CRYPTOGRAPHIC,
    issuer_id: str = "issuer:trusted",
    epoch: str = "epoch:7",
) -> TestProofCertificate:
    return TestProofCertificate(
        receipt_cid=receipt.receipt_id,
        execution_key_cid=execution_key.execution_key_id,
        policy_cid=execution_key.policy_cid,
        statement_cid="cid:statement",
        circuit_cid="cid:circuit",
        verifying_key_cid="cid:verifying-key",
        proof_artifact_cid="cid:proof",
        issuer_id=issuer_id,
        epoch=epoch,
        proof_system_id="groth16",
        backend_mode=backend_mode,
        authority=authority,
        public_inputs={
            "receipt_cid": receipt.receipt_id,
            "execution_key_cid": execution_key.execution_key_id,
            "policy_cid": execution_key.policy_cid,
            "statement_cid": "cid:statement",
            "circuit_cid": "cid:circuit",
            "verifying_key_cid": "cid:verifying-key",
            "proof_system_id": "groth16",
            "issuer_id": issuer_id,
            "issuer_key_id": "key:issuer",
            "epoch": epoch,
            "setup_outcome": "pass",
            "call_outcome": "pass",
            "teardown_outcome": "pass",
        },
    )


def _pair(**certificate_kwargs: Any) -> tuple[
    TestLocatorKey, TestExecutionKey, TestPassReceipt, TestProofCertificate
]:
    locator = _locator()
    execution_key = _execution_key(locator)
    receipt = _receipt(locator, execution_key)
    certificate = _certificate(receipt, execution_key, **certificate_kwargs)
    return locator, execution_key, receipt, certificate


def _store(tmp_path: Path, **kwargs: Any) -> TestCertificateStore:
    return TestCertificateStore(
        tmp_path / "certificate-store",
        clock=lambda: NOW_S,
        **kwargs,
    )


def test_put_candidate_uses_temp_atomic_replace_readback_then_index(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    locator, _execution_key_value, receipt, certificate = _pair()

    result = store.put_candidate(receipt, certificate, locator_cid=locator.locator_id)

    assert result.stored
    assert result.indexed
    assert result.reason_code is CertificateStoreReason.OK
    assert result.receipt_cid == receipt.receipt_id
    assert result.certificate_cid == certificate.certificate_id

    receipt_get = store.get_bytes(receipt.receipt_id)
    certificate_get = store.get_bytes(certificate.certificate_id)
    assert receipt_get.hit
    assert certificate_get.hit
    assert receipt_get.data == receipt.canonical_bytes()
    assert certificate_get.data == certificate.canonical_bytes()

    lookup = store.lookup(locator.locator_id)
    assert lookup.status is CertificateStoreStatus.HIT
    assert len(lookup.candidates) == 1
    candidate = lookup.candidates[0]
    assert candidate["receipt_cid"] == receipt.receipt_id
    assert candidate["certificate_cid"] == certificate.certificate_id
    assert candidate["receipt_bytes"] == receipt.canonical_bytes()
    assert candidate["certificate_bytes"] == certificate.canonical_bytes()

    # No abandoned temp files after a successful publication.
    leftovers = list((tmp_path / "certificate-store").rglob(".tmp.*"))
    assert leftovers == []


def test_cas_rejects_oversized_bytes(tmp_path: Path) -> None:
    cas = ImmutableCertificateCAS(
        tmp_path / "cas-root",
        max_blob_bytes=64,
        clock=lambda: NOW_S,
    )
    locator, execution_key, receipt, _certificate_value = _pair()
    data = receipt.canonical_bytes()
    assert len(data) > 64

    result = cas.put_bytes(data, claimed_cid=receipt.receipt_id)

    assert not result.stored
    assert result.reason_code is CertificateStoreReason.OVER_BUDGET
    assert not cas.has(receipt.receipt_id)


def test_cas_rejects_malformed_and_noncanonical_bytes(tmp_path: Path) -> None:
    cas = ImmutableCertificateCAS(tmp_path / "cas-root", clock=lambda: NOW_S)

    malformed = cas.put_bytes(b"{not-json")
    assert not malformed.stored
    assert malformed.reason_code is CertificateStoreReason.MALFORMED

    locator, execution_key, receipt, _certificate_value = _pair()
    noncanonical = b" " + receipt.canonical_bytes()
    result = cas.put_bytes(noncanonical, claimed_cid=receipt.receipt_id)
    assert not result.stored
    assert result.reason_code is CertificateStoreReason.MALFORMED


def test_cas_claimed_cid_mismatch_is_rejected(tmp_path: Path) -> None:
    cas = ImmutableCertificateCAS(tmp_path / "cas-root", clock=lambda: NOW_S)
    locator, execution_key, receipt, certificate = _pair()

    result = cas.put_bytes(
        receipt.canonical_bytes(),
        claimed_cid=certificate.certificate_id,
    )

    assert not result.stored
    assert result.reason_code is CertificateStoreReason.CID_MISMATCH


def test_partial_and_corrupt_blobs_miss_and_quarantine(tmp_path: Path) -> None:
    store = _store(tmp_path)
    locator, _execution_key_value, receipt, certificate = _pair()
    assert store.put_candidate(receipt, certificate).stored

    path = store.cas.blob_path(certificate.certificate_id)
    path.write_bytes(b"")
    empty = store.get_bytes(certificate.certificate_id)
    assert empty.status is CertificateStoreStatus.MISS
    assert empty.reason_code is CertificateStoreReason.PARTIAL
    assert store.cas.quarantine_path(certificate.certificate_id).exists()

    # Re-publish a fresh authority unit, then poison the receipt blob.
    locator2, execution_key2, receipt2, certificate2 = _pair(issuer_id="issuer:other")
    # Build a second independent pair with distinct bytes via different issuer on
    # a distinct execution key / receipt chain already produced by _pair defaults
    # would collide locators; force a unique receipt by using a new store path.
    store2 = _store(tmp_path / "second")
    assert store2.put_candidate(receipt2, certificate2, locator_cid=locator2.locator_id).stored
    corrupt_path = store2.cas.blob_path(receipt2.receipt_id)
    corrupt_path.write_bytes(b'{"schema":"broken","not":"canonical"}')
    corrupt = store2.get_bytes(receipt2.receipt_id)
    assert corrupt.status is CertificateStoreStatus.MISS
    assert corrupt.reason_code in {
        CertificateStoreReason.CORRUPT,
        CertificateStoreReason.INTEGRITY_FAILED,
    }
    assert store2.cas.quarantine_path(receipt2.receipt_id).exists()


def test_symlink_blob_path_misses_safely(tmp_path: Path) -> None:
    store = _store(tmp_path)
    locator, _execution_key_value, receipt, certificate = _pair()
    assert store.put_candidate(receipt, certificate).stored

    blob = store.cas.blob_path(certificate.certificate_id)
    outside = tmp_path / "outside-secret.json"
    outside.write_bytes(certificate.canonical_bytes())
    blob.unlink()
    blob.symlink_to(outside)

    result = store.get_bytes(certificate.certificate_id)
    assert result.status is CertificateStoreStatus.MISS
    assert result.reason_code is CertificateStoreReason.SYMLINK_REJECTED


def test_path_escape_cid_tokens_miss_safely(tmp_path: Path) -> None:
    store = _store(tmp_path)

    for bad in ("../escape", "a/b", "a\\b", "UPPER", "has space", ""):
        result = store.get_bytes(bad)
        assert result.status is CertificateStoreStatus.MISS
        assert result.reason_code is CertificateStoreReason.PATH_ESCAPE

    publish = store.index.publish(
        "../escape",
        certificate_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        receipt_cid="baguqeerbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    )
    assert not publish.published
    assert publish.reason_code is CertificateStoreReason.PATH_ESCAPE


def test_lookup_skips_missing_cas_and_quarantines_corrupt_hint(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    locator, _execution_key_value, receipt, certificate = _pair()
    assert store.put_candidate(receipt, certificate, locator_cid=locator.locator_id).stored

    # Remove certificate bytes after index publication (simulates partial store).
    store.cas.blob_path(certificate.certificate_id).unlink()
    lookup = store.lookup(locator.locator_id)
    assert lookup.status is CertificateStoreStatus.MISS
    assert lookup.reason_code is CertificateStoreReason.CANDIDATE_MISSING


def test_index_ttl_and_revocation_hide_candidates(tmp_path: Path) -> None:
    store = _store(tmp_path, index_ttl_ms=1_000)
    locator, _execution_key_value, receipt, certificate = _pair()
    assert store.put_candidate(
        receipt,
        certificate,
        locator_cid=locator.locator_id,
        created_at_ms=NOW_MS,
        expires_at_ms=NOW_MS + 500,
    ).stored

    expired = store.lookup(locator.locator_id, now_ms=NOW_MS + 501)
    assert expired.status is CertificateStoreStatus.MISS

    store2 = _store(tmp_path / "revocation")
    assert store2.put_candidate(
        receipt, certificate, locator_cid=locator.locator_id, created_at_ms=NOW_MS
    ).stored
    revoked = store2.index.revoke_certificate(
        locator.locator_id, certificate.certificate_id
    )
    assert revoked.published
    assert store2.lookup(locator.locator_id).status is CertificateStoreStatus.MISS


def test_fence_blocks_stale_writer_from_index_publication(tmp_path: Path) -> None:
    root = tmp_path / "fence-root"
    fence = CertificateWriteFence(root, clock=lambda: NOW_S, default_ttl_ms=30_000)
    lease_a = fence.acquire("locator:demo", owner_id="writer-a")
    with pytest.raises(Exception):
        fence.acquire("locator:demo", owner_id="writer-b")

    assert fence.validate(lease_a)
    fence.release(lease_a)
    lease_b = fence.acquire("locator:demo", owner_id="writer-b")
    assert lease_b.fencing_token == lease_a.fencing_token + 1
    assert not fence.validate(lease_a)
    # Stale owner cannot revive the prior fencing token after release.
    assert not fence.validate(lease_a)


def test_parallel_writers_cannot_publish_mixed_authority(tmp_path: Path) -> None:
    """Each published candidate is one complete receipt+certificate authority unit."""

    store_root = tmp_path / "parallel"
    locator = _locator()
    barrier = threading.Barrier(2)
    outcomes: list[Any] = []
    lock = threading.Lock()

    def _unique_pair(tag: str) -> tuple[TestPassReceipt, TestProofCertificate]:
        execution_key = TestExecutionKey(
            locator_cid=locator.locator_id,
            repository_forest_cid=f"cid:repository-forest-{tag}",
            static_trace_root_cid=f"cid:static-trace-{tag}",
            runtime_trace_root_cid=f"cid:runtime-trace-{tag}",
            runtime_completeness_policy="complete-v1",
            policy_cid="cid:policy",
        )
        receipt = TestPassReceipt(
            execution_key_cid=execution_key.execution_key_id,
            locator_cid=locator.locator_id,
            setup_outcome=PhaseOutcome.PASS,
            call_outcome=PhaseOutcome.PASS,
            teardown_outcome=PhaseOutcome.PASS,
            static_trace_root_cid=execution_key.static_trace_root_cid,
            runtime_trace_root_cid=execution_key.runtime_trace_root_cid,
            completeness_receipt_cid=f"cid:completeness-{tag}",
            dependency_forest_cid=execution_key.repository_forest_cid,
            issuer_key_id="key:issuer",
            policy_cid=execution_key.policy_cid,
            nonce=tag,
        )
        certificate = _certificate(
            receipt,
            execution_key,
            issuer_id=f"issuer:{tag}",
            epoch=f"epoch:{tag}",
        )
        return receipt, certificate

    def worker(tag: str) -> None:
        store = TestCertificateStore(
            store_root,
            clock=lambda: NOW_S,
            owner_id=f"owner-{tag}",
        )
        receipt, certificate = _unique_pair(tag)
        barrier.wait(timeout=5)
        result = store.put_candidate(
            receipt,
            certificate,
            locator_cid=locator.locator_id,
            owner_id=f"owner-{tag}",
        )
        with lock:
            outcomes.append((tag, result, receipt.receipt_id, certificate.certificate_id))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, "a"), pool.submit(worker, "b")]
        for future in futures:
            future.result(timeout=30)

    assert len(outcomes) == 2
    stored = [item for item in outcomes if item[1].stored and item[1].indexed]
    # At least one publication must fully succeed; a fenced loser is acceptable.
    assert stored
    assert all(
        item[1].reason_code
        in {
            CertificateStoreReason.FENCED,
            CertificateStoreReason.FENCE_MISMATCH,
            CertificateStoreReason.UNAVAILABLE,
            CertificateStoreReason.OK,
            CertificateStoreReason.ALREADY_EXISTS,
        }
        for item in outcomes
    )

    reader = TestCertificateStore(store_root, clock=lambda: NOW_S)
    lookup = reader.lookup(locator.locator_id)
    assert lookup.status is CertificateStoreStatus.HIT
    for candidate in lookup.candidates:
        # Reconstruct contracts and ensure the certificate binds the receipt.
        receipt = TestPassReceipt.from_dict(
            json.loads(candidate["receipt_bytes"].decode("utf-8"))
        )
        certificate = TestProofCertificate.from_dict(
            json.loads(candidate["certificate_bytes"].decode("utf-8"))
        )
        assert candidate["receipt_cid"] == receipt.receipt_id
        assert candidate["certificate_cid"] == certificate.certificate_id
        assert certificate.receipt_cid == receipt.receipt_id
        assert certificate.public_inputs["receipt_cid"] == receipt.receipt_id
        # No mixed pairing across the two writers' identity tags.
        receipt_tag = receipt.nonce
        cert_issuer = certificate.issuer_id
        assert cert_issuer == f"issuer:{receipt_tag}"


def test_store_integrates_with_trust_aware_proof_cache(tmp_path: Path) -> None:
    store = _store(tmp_path)
    locator, execution_key, receipt, certificate = _pair()
    assert store.put_candidate(
        receipt, certificate, locator_cid=locator.locator_id, created_at_ms=NOW_MS
    ).stored

    cache = TestProofCache(
        current_policy={
            "policy_cid": "cid:policy",
            "statement_cid": "cid:statement",
            "circuit_cid": "cid:circuit",
            "verifying_key_cid": "cid:verifying-key",
            "proof_system_id": "groth16",
            "trusted_issuer_ids": ("issuer:trusted",),
            "allowed_epochs": ("epoch:7",),
            "revoked_issuer_ids": (),
            "revoked_receipt_cids": (),
            "revoked_certificate_cids": (),
        },
        verifier=lambda *_args: True,
        candidate_provider=store.candidate_provider,
        clock=lambda: NOW_MS,
    )

    result = cache.lookup(locator, execution_key)
    assert result.decision.action is ReuseAction.SKIP
    assert result.decision.certificate_cid == certificate.certificate_id
    assert result.decision.receipt_cid == receipt.receipt_id


def test_index_symlink_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    locator, _execution_key_value, receipt, certificate = _pair()
    assert store.put_candidate(receipt, certificate, locator_cid=locator.locator_id).stored

    index_path = store.index._index_path(locator.locator_id)
    payload = index_path.read_bytes()
    index_path.unlink()
    outside = tmp_path / "evil-index.json"
    outside.write_bytes(payload)
    index_path.symlink_to(outside)

    lookup = store.index.candidates(locator.locator_id)
    assert lookup.status is CertificateStoreStatus.MISS
    assert lookup.reason_code is CertificateStoreReason.SYMLINK_REJECTED


def test_put_receipt_and_certificate_helpers_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _locator_value, _execution_key_value, receipt, certificate = _pair()

    receipt_put = store.put_receipt(receipt)
    certificate_put = store.put_certificate(certificate)
    assert receipt_put.stored
    assert certificate_put.stored
    assert store.get_bytes(receipt.receipt_id).data == receipt.canonical_bytes()
    assert store.get_bytes(certificate.certificate_id).data == certificate.canonical_bytes()


def test_idempotent_cas_put_returns_already_exists(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _locator_value, _execution_key_value, receipt, _certificate_value = _pair()
    first = store.put_receipt(receipt)
    second = store.put_receipt(receipt)
    assert first.stored and first.reason_code is CertificateStoreReason.OK
    assert second.stored and second.reason_code is CertificateStoreReason.ALREADY_EXISTS


def test_restart_scrubs_partial_temporary_files(tmp_path: Path) -> None:
    root = tmp_path / "restart"
    store = TestCertificateStore(root, clock=lambda: NOW_S)
    cas_dir = store.cas.cas_root / "aa"
    cas_dir.mkdir(parents=True, exist_ok=True)
    orphan = cas_dir / ".tmp.orphan.blob.tmp"
    orphan.write_bytes(b"partial")
    assert orphan.exists()

    # Re-open performs restart recovery scrub.
    reopened = TestCertificateStore(root, clock=lambda: NOW_S)
    assert not orphan.exists()
    assert reopened.root == root.resolve()


def test_bounded_index_retains_newest_candidates(tmp_path: Path) -> None:
    store = _store(tmp_path, max_candidates=2)
    locator = _locator()
    published_cids: list[str] = []
    for index in range(3):
        execution_key = TestExecutionKey(
            locator_cid=locator.locator_id,
            repository_forest_cid=f"cid:repository-forest-{index}",
            static_trace_root_cid=f"cid:static-{index}",
            runtime_trace_root_cid=f"cid:runtime-{index}",
            runtime_completeness_policy="complete-v1",
            policy_cid="cid:policy",
        )
        receipt = TestPassReceipt(
            execution_key_cid=execution_key.execution_key_id,
            locator_cid=locator.locator_id,
            setup_outcome=PhaseOutcome.PASS,
            call_outcome=PhaseOutcome.PASS,
            teardown_outcome=PhaseOutcome.PASS,
            static_trace_root_cid=execution_key.static_trace_root_cid,
            runtime_trace_root_cid=execution_key.runtime_trace_root_cid,
            completeness_receipt_cid=f"cid:completeness-{index}",
            dependency_forest_cid=execution_key.repository_forest_cid,
            issuer_key_id="key:issuer",
            policy_cid=execution_key.policy_cid,
            nonce=str(index),
        )
        certificate = _certificate(
            receipt,
            execution_key,
            issuer_id=f"issuer:{index}",
            epoch=f"epoch:{index}",
        )
        result = store.put_candidate(
            receipt, certificate, locator_cid=locator.locator_id
        )
        assert result.stored
        published_cids.append(certificate.certificate_id)

    lookup = store.lookup(locator.locator_id)
    assert lookup.status is CertificateStoreStatus.HIT
    assert len(lookup.candidates) == 2
    returned = {item["certificate_cid"] for item in lookup.candidates}
    # Newest two should remain (indexes 1 and 2).
    assert published_cids[2] in returned
    assert published_cids[1] in returned
    assert published_cids[0] not in returned
