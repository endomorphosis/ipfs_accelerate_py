"""Security and concurrency assurance for proof-backed test reuse (PTR-092).

These tests deliberately cross the storage/admission boundary.  Mutable index
entries and serialized claims are treated only as candidate hints: a test may
be skipped only after immutable bytes are re-read and current local authority
admits them.  Every hostile, incomplete, unavailable, or revoked case below
must therefore execute the real test body.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pytest
from ipfs_accelerate_py.agent_supervisor.proof.test_certificate_store import (
    CertificateStoreReason,
    CertificateStoreStatus,
    CertificateWriteFence,
    TestCertificateStore,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    CertificateAuthority,
    PhaseOutcome,
    ProofBackendMode,
    ReuseAction,
    ReuseReasonCode,
    TestExecutionKey,
    TestLocatorKey,
    TestPassReceipt,
    TestProofCertificate,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_proof_cache import (
    TestProofCache,
    TestProofCacheLookup,
)


NOW_S = 10.0
NOW_MS = 10_000
DIAGNOSTIC_BUDGET_BYTES = 4_096
SECRET = "private-value-that-must-never-reach-diagnostics"


def _locator() -> TestLocatorKey:
    return TestLocatorKey(
        repository_id="repository:security-assurance",
        package_identity="package:security-assurance",
        node_id="test/api/test_security_assurance.py::test_authority",
    )


def _pair(
    tag: str = "trusted",
) -> tuple[TestLocatorKey, TestExecutionKey, TestPassReceipt, TestProofCertificate]:
    locator = _locator()
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
    issuer_id = f"issuer:{tag}"
    epoch = f"epoch:{tag}"
    certificate = TestProofCertificate(
        receipt_cid=receipt.receipt_id,
        execution_key_cid=execution_key.execution_key_id,
        policy_cid=execution_key.policy_cid,
        statement_cid="cid:statement",
        circuit_cid="cid:circuit",
        verifying_key_cid="cid:verifying-key",
        proof_artifact_cid="cid:proof-trusted",
        issuer_id=issuer_id,
        epoch=epoch,
        proof_system_id="groth16",
        backend_mode=ProofBackendMode.CRYPTOGRAPHIC,
        authority=CertificateAuthority.AUTHORITATIVE,
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
    return locator, execution_key, receipt, certificate


def _policy(
    certificate: TestProofCertificate,
    *,
    revoked_certificate_cids: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "policy_cid": "cid:policy",
        "statement_cid": "cid:statement",
        "circuit_cid": "cid:circuit",
        "verifying_key_cid": "cid:verifying-key",
        "proof_system_id": "groth16",
        "trusted_issuer_ids": (certificate.issuer_id,),
        "allowed_epochs": (certificate.epoch,),
        "revoked_issuer_ids": (),
        "revoked_receipt_cids": (),
        "revoked_certificate_cids": revoked_certificate_cids,
    }


def _verifier(
    certificate: TestProofCertificate,
    _receipt: TestPassReceipt,
    _policy_value: dict[str, Any],
) -> bool:
    """Stand in for local cryptographic verification, including proof binding."""

    return certificate.proof_artifact_cid == "cid:proof-trusted"


def _cache(
    certificate: TestProofCertificate,
    *,
    candidate_provider: Any = None,
    verifier: Any = _verifier,
    revocation_checker: Callable[..., Any] | None = None,
    policy: dict[str, Any] | None = None,
    max_candidates: int = 8,
    max_blob_bytes: int = 1_048_576,
) -> TestProofCache:
    return TestProofCache(
        current_policy=policy or _policy(certificate),
        verifier=verifier,
        revocation_checker=revocation_checker,
        candidate_provider=candidate_provider,
        clock=lambda: NOW_MS,
        max_candidates=max_candidates,
        max_blob_bytes=max_blob_bytes,
    )


def _assert_run_executes_real_test(
    result: TestProofCacheLookup,
    *,
    forbidden_text: str = SECRET,
) -> None:
    """Assert fail-open-to-run without reflecting attacker-controlled details."""

    assert result.decision.action is ReuseAction.RUN
    assert not result.hit
    assert result.decision.certificate_cid == ""
    assert result.decision.receipt_cid == ""
    rendered = result.decision.canonical_bytes()
    assert len(rendered) <= DIAGNOSTIC_BUDGET_BYTES
    assert forbidden_text.encode("utf-8") not in rendered

    calls: list[str] = []

    def real_test_body() -> None:
        calls.append("executed")

    # This is the supervisor's consequential branch: only SKIP suppresses it.
    if result.decision.action is not ReuseAction.SKIP:
        real_test_body()
    assert calls == ["executed"]


def _publisher_process(
    root: str,
    tag: str,
    start: multiprocessing.synchronize.Event,
    outcomes: multiprocessing.queues.Queue,
) -> None:
    """Publish one complete authority unit from a separate worker process."""

    try:
        locator, _execution_key, receipt, certificate = _pair(tag)
        store = TestCertificateStore(
            root,
            clock=lambda: NOW_S,
            owner_id=f"process:{tag}",
        )
        if not start.wait(timeout=10):
            outcomes.put(("error", tag, "start_timeout"))
            return
        result = store.put_candidate(
            receipt,
            certificate,
            locator_cid=locator.locator_id,
            owner_id=f"process:{tag}",
        )
        outcomes.put(
            (
                "result",
                tag,
                result.stored,
                result.indexed,
                result.reason_code.value,
                receipt.receipt_id,
                certificate.certificate_id,
            )
        )
    except BaseException as exc:  # pragma: no cover - reported to parent
        outcomes.put(("error", tag, type(exc).__name__))


def _crash_with_fence(
    root: str,
    locator_cid: str,
    ready: multiprocessing.synchronize.Event,
) -> None:
    """Die after taking a publication fence and leaving an incomplete temp."""

    fence = CertificateWriteFence(root, default_ttl_ms=100)
    fence.acquire(f"locator:{locator_cid}", owner_id="crashed-worker")
    shard = Path(root) / "cas" / "ff"
    shard.mkdir(parents=True, exist_ok=True)
    (shard / ".tmp.crashed-authority.blob.tmp").write_bytes(b"incomplete")
    ready.set()
    os._exit(23)


@pytest.mark.parametrize(
    "attack",
    (
        "forged_receipt",
        "forged_proof",
        "forged_claimed_cids",
        "noncanonical_receipt",
        "partial_certificate",
        "private_metadata",
        "oversized_artifact",
    ),
)
def test_hostile_artifacts_never_authorize_skip(attack: str) -> None:
    locator, execution_key, receipt, certificate = _pair()
    candidate = TestProofCache.candidate(
        receipt,
        certificate,
        created_at_ms=NOW_MS,
        metadata={"source": "hostile-test"},
    )
    max_blob_bytes = 1_048_576

    if attack == "forged_receipt":
        forged = replace(receipt, nonce="attacker-controlled")
        candidate["receipt_bytes"] = forged.canonical_bytes()
        candidate["receipt_cid"] = forged.receipt_id
    elif attack == "forged_proof":
        forged = replace(certificate, proof_artifact_cid="cid:forged-proof")
        candidate["certificate_bytes"] = forged.canonical_bytes()
        candidate["certificate_cid"] = forged.certificate_id
    elif attack == "forged_claimed_cids":
        candidate["receipt_cid"] = "a" * 64
        candidate["certificate_cid"] = "b" * 64
    elif attack == "noncanonical_receipt":
        candidate["receipt_bytes"] = b" " + receipt.canonical_bytes()
    elif attack == "partial_certificate":
        candidate["certificate_bytes"] = certificate.canonical_bytes()[:-19]
    elif attack == "private_metadata":
        candidate["metadata"] = {"private_witness": SECRET}
    elif attack == "oversized_artifact":
        max_blob_bytes = len(receipt.canonical_bytes()) - 1
    else:  # pragma: no cover - parametrization is closed
        raise AssertionError(attack)

    result = _cache(
        certificate,
        max_blob_bytes=max_blob_bytes,
    ).lookup(locator, execution_key, candidates=[candidate])

    _assert_run_executes_real_test(result)
    assert result.reason_code in {
        ReuseReasonCode.CANDIDATE_INTEGRITY_FAILED,
        ReuseReasonCode.EXECUTION_KEY_MISMATCH,
        ReuseReasonCode.MALFORMED_ARTIFACT,
        ReuseReasonCode.OVER_BUDGET,
        ReuseReasonCode.PRIVATE_MATERIAL,
        ReuseReasonCode.TRUST_POLICY_REJECTED,
    }


def test_serialized_authority_flags_and_failure_details_are_non_authoritative() -> None:
    locator, execution_key, receipt, certificate = _pair()
    candidate = TestProofCache.candidate(
        receipt,
        certificate,
        metadata={"trusted": True, "authoritative": True, "verified": True},
    )

    # A truthy provider object is not the exact local-verifier True result.
    asserted = _cache(
        certificate,
        verifier=lambda *_args: {"verified": True, "authoritative": True},
    ).lookup(locator, execution_key, candidates=[candidate])
    _assert_run_executes_real_test(asserted)
    assert asserted.reason_code is ReuseReasonCode.TRUST_POLICY_REJECTED

    def failed_provider(_locator_cid: str) -> list[dict[str, Any]]:
        raise RuntimeError(SECRET)

    def failed_verifier(*_args: Any) -> bool:
        raise RuntimeError(SECRET)

    def failed_revocation(*_args: Any) -> bool:
        raise TimeoutError(SECRET)

    def failed_iteration() -> Any:
        raise RuntimeError(SECRET)
        yield candidate  # pragma: no cover

    failures = (
        _cache(certificate, candidate_provider=failed_provider).lookup(
            locator, execution_key
        ),
        _cache(certificate, verifier=failed_verifier).lookup(
            locator, execution_key, candidates=[candidate]
        ),
        _cache(certificate, revocation_checker=failed_revocation).lookup(
            locator, execution_key, candidates=[candidate]
        ),
        _cache(
            certificate,
            candidate_provider=lambda _locator_cid: failed_iteration(),
        ).lookup(locator, execution_key),
        _cache(certificate, max_candidates=1).lookup(
            locator, execution_key, candidates=[candidate, candidate]
        ),
    )
    for result in failures:
        _assert_run_executes_real_test(result)


def test_path_escape_and_symlink_artifacts_cannot_escape_store_root(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text(SECRET, encoding="utf-8")
    original_outside = {path.name: path.read_bytes() for path in outside.iterdir()}

    store = TestCertificateStore(
        tmp_path / "path-store",
        clock=lambda: NOW_S,
    )
    locator, _execution_key, receipt, certificate = _pair()
    for hostile_cid in (
        "../escape",
        "../../outside/sentinel.txt",
        "a/b",
        "a\\b",
        "/absolute",
        "UPPERCASE",
        "has space",
        "",
    ):
        read = store.get_bytes(hostile_cid)
        assert read.status is CertificateStoreStatus.MISS
        assert read.reason_code is CertificateStoreReason.PATH_ESCAPE
        publish = store.index.publish(
            hostile_cid,
            certificate_cid=certificate.certificate_id,
            receipt_cid=receipt.receipt_id,
        )
        assert not publish.published
        assert publish.reason_code is CertificateStoreReason.PATH_ESCAPE

    escaped_put = store.put_candidate(
        receipt,
        certificate,
        locator_cid="../../outside",
    )
    assert not escaped_put.stored
    assert escaped_put.reason_code is CertificateStoreReason.PATH_ESCAPE

    # A symlinked shard must not redirect a CAS publication outside the root.
    shard_store = TestCertificateStore(
        tmp_path / "shard-store",
        clock=lambda: NOW_S,
    )
    shard = shard_store.cas.cas_root / receipt.receipt_id[:2]
    shard.symlink_to(outside, target_is_directory=True)
    redirected = shard_store.put_candidate(receipt, certificate)
    assert not redirected.stored
    assert redirected.reason_code in {
        CertificateStoreReason.PATH_ESCAPE,
        CertificateStoreReason.SYMLINK_REJECTED,
    }

    # A final blob symlink may contain valid bytes but is never an authority.
    blob_store = TestCertificateStore(
        tmp_path / "blob-store",
        clock=lambda: NOW_S,
    )
    assert blob_store.put_candidate(receipt, certificate).stored
    outside_blob = outside / "outside-certificate.json"
    outside_blob.write_bytes(certificate.canonical_bytes())
    blob_path = blob_store.cas.blob_path(certificate.certificate_id)
    blob_path.unlink()
    blob_path.symlink_to(outside_blob)
    blob_result = _cache(
        certificate,
        candidate_provider=blob_store.candidate_provider,
    ).lookup(locator, _execution_key)
    _assert_run_executes_real_test(blob_result)

    # A symlinked mutable index is likewise only a rejected hint.
    index_store = TestCertificateStore(
        tmp_path / "index-store",
        clock=lambda: NOW_S,
    )
    assert index_store.put_candidate(receipt, certificate).stored
    index_path = index_store.index._index_path(locator.locator_id)
    outside_index = outside / "outside-index.json"
    outside_index.write_bytes(index_path.read_bytes())
    index_path.unlink()
    index_path.symlink_to(outside_index)
    index_result = _cache(
        certificate,
        candidate_provider=index_store.candidate_provider,
    ).lookup(locator, _execution_key)
    _assert_run_executes_real_test(index_result)

    assert sentinel.read_text(encoding="utf-8") == SECRET
    assert original_outside["sentinel.txt"] == sentinel.read_bytes()
    assert not (tmp_path / "escape").exists()


def test_incomplete_publications_and_poisoned_indexes_remain_invisible(
    tmp_path: Path,
) -> None:
    locator, execution_key, receipt, certificate = _pair()
    root = tmp_path / "incomplete-store"
    store = TestCertificateStore(root, clock=lambda: NOW_S)

    # CAS-only writes are deliberately not discoverable without atomic index
    # publication, even though both immutable objects are complete.
    cas_only = store.put_candidate(receipt, certificate, publish_index=False)
    assert cas_only.stored and not cas_only.indexed
    absent = _cache(
        certificate,
        candidate_provider=store.candidate_provider,
    ).lookup(locator, execution_key)
    _assert_run_executes_real_test(absent)

    orphan_dir = store.cas.cas_root / "ee"
    orphan_dir.mkdir(parents=True)
    orphan = orphan_dir / ".tmp.interrupted-publication.blob.tmp"
    orphan.write_bytes(certificate.canonical_bytes())
    reopened = TestCertificateStore(root, clock=lambda: NOW_S)
    assert not orphan.exists()

    # An index hint cannot make a missing half of an authority unit visible.
    missing_certificate_cid = "c" * 64
    hinted = reopened.index.publish(
        locator.locator_id,
        certificate_cid=missing_certificate_cid,
        receipt_cid=receipt.receipt_id,
        created_at_ms=NOW_MS,
    )
    assert hinted.published
    missing_half = _cache(
        certificate,
        candidate_provider=reopened.candidate_provider,
    ).lookup(locator, execution_key)
    _assert_run_executes_real_test(missing_half)

    # Complete publication followed by a zero-length final blob is quarantined.
    corrupt_store = TestCertificateStore(
        tmp_path / "corrupt-final",
        clock=lambda: NOW_S,
    )
    assert corrupt_store.put_candidate(receipt, certificate).stored
    corrupt_store.cas.blob_path(certificate.certificate_id).write_bytes(b"")
    partial = _cache(
        certificate,
        candidate_provider=corrupt_store.candidate_provider,
    ).lookup(locator, execution_key)
    _assert_run_executes_real_test(partial)
    assert corrupt_store.cas.quarantine_path(certificate.certificate_id).exists()

    # Empty, malformed, duplicate-key, and oversized mutable index documents
    # are bounded misses, never alternate authority sources.
    poison_documents = (
        b"",
        b"{not-json",
        b'{"schema":"x","schema":"y"}',
        b"x" * 1_025,
    )
    for index, poison in enumerate(poison_documents):
        poison_store = TestCertificateStore(
            tmp_path / f"poison-{index}",
            clock=lambda: NOW_S,
            max_index_bytes=1_024,
        )
        poison_path = poison_store.index._index_path(locator.locator_id)
        poison_path.write_bytes(poison)
        poisoned = _cache(
            certificate,
            candidate_provider=poison_store.candidate_provider,
        ).lookup(locator, execution_key)
        _assert_run_executes_real_test(poisoned)


def test_worker_crash_restart_preserves_immutable_authority_and_recovers_fence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "crash-restart"
    locator, execution_key, receipt, certificate = _pair()
    store = TestCertificateStore(root)
    assert store.put_candidate(receipt, certificate).stored
    receipt_before = store.cas.blob_path(receipt.receipt_id).read_bytes()
    certificate_before = store.cas.blob_path(certificate.certificate_id).read_bytes()

    context = multiprocessing.get_context("fork")
    ready = context.Event()
    worker = context.Process(
        target=_crash_with_fence,
        args=(str(root), locator.locator_id, ready),
    )
    worker.start()
    assert ready.wait(timeout=10)
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert worker.exitcode == 23

    # Opening the store scrubs abandoned CAS temporaries but leaves immutable
    # authority bytes untouched.  A successor eventually fences off the dead
    # owner after the short lease expires.
    reopened = TestCertificateStore(root, fence_ttl_ms=100)
    assert list(reopened.cas.cas_root.rglob(".tmp.*")) == []
    _locator2, _execution_key2, receipt2, certificate2 = _pair("successor")
    deadline = time.monotonic() + 3
    while True:
        successor = reopened.put_candidate(receipt2, certificate2)
        if successor.stored:
            break
        assert successor.reason_code is CertificateStoreReason.FENCED
        if time.monotonic() >= deadline:
            pytest.fail("successor did not recover the expired crash fence")
        time.sleep(0.02)

    assert reopened.cas.blob_path(receipt.receipt_id).read_bytes() == receipt_before
    assert (
        reopened.cas.blob_path(certificate.certificate_id).read_bytes()
        == certificate_before
    )
    current_ms = time.time_ns() // 1_000_000
    cache = TestProofCache(
        current_policy=_policy(certificate),
        verifier=_verifier,
        candidate_provider=reopened.candidate_provider,
        clock=lambda: current_ms,
    )
    recovered = cache.lookup(locator, execution_key)
    assert recovered.decision.action is ReuseAction.SKIP
    assert recovered.decision.certificate_cid == certificate.certificate_id
    assert recovered.decision.receipt_cid == receipt.receipt_id


def test_concurrent_process_publishers_never_create_mixed_authority(
    tmp_path: Path,
) -> None:
    root = tmp_path / "parallel-processes"
    context = multiprocessing.get_context("fork")
    start = context.Event()
    outcomes = context.Queue()
    tags = ("alpha", "beta", "gamma", "delta")
    workers = [
        context.Process(
            target=_publisher_process,
            args=(str(root), tag, start, outcomes),
        )
        for tag in tags
    ]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(timeout=20)
        if worker.is_alive():  # Prevent a failed assertion from leaking workers.
            worker.terminate()
            worker.join(timeout=5)
            pytest.fail("parallel publisher exceeded its bounded deadline")
        assert worker.exitcode == 0

    reports: list[tuple[Any, ...]] = []
    for _worker in workers:
        try:
            reports.append(outcomes.get(timeout=10))
        except queue.Empty:
            pytest.fail("parallel publisher did not report an outcome")
    assert all(report[0] == "result" for report in reports), reports
    stored_reports = [
        report for report in reports if bool(report[2]) and bool(report[3])
    ]
    assert stored_reports
    allowed_reasons = {
        CertificateStoreReason.OK.value,
        CertificateStoreReason.FENCED.value,
        CertificateStoreReason.FENCE_MISMATCH.value,
        CertificateStoreReason.UNAVAILABLE.value,
    }
    assert all(report[4] in allowed_reasons for report in reports)

    reader = TestCertificateStore(root, clock=lambda: NOW_S)
    lookup = reader.lookup(_locator().locator_id)
    assert lookup.status is CertificateStoreStatus.HIT
    for candidate in lookup.candidates:
        stored_receipt = TestPassReceipt.from_dict(
            json.loads(candidate["receipt_bytes"].decode("utf-8"))
        )
        stored_certificate = TestProofCertificate.from_dict(
            json.loads(candidate["certificate_bytes"].decode("utf-8"))
        )
        assert stored_receipt.receipt_id == candidate["receipt_cid"]
        assert stored_certificate.certificate_id == candidate["certificate_cid"]
        assert stored_certificate.receipt_cid == stored_receipt.receipt_id
        assert stored_certificate.public_inputs["receipt_cid"] == stored_receipt.receipt_id
        assert stored_certificate.issuer_id == f"issuer:{stored_receipt.nonce}"

    # At least one complete authority unit survives full local re-admission.
    winning_tag = str(stored_reports[0][1])
    locator, execution_key, _receipt, certificate = _pair(winning_tag)
    admitted = _cache(
        certificate,
        candidate_provider=reader.candidate_provider,
    ).lookup(locator, execution_key)
    assert admitted.decision.action is ReuseAction.SKIP
    assert list(root.rglob(".tmp.*")) == []


def test_revocation_wins_replay_and_inflight_admission_races(
    tmp_path: Path,
) -> None:
    locator, execution_key, receipt, certificate = _pair()
    store = TestCertificateStore(
        tmp_path / "revocation-race",
        clock=lambda: NOW_S,
    )
    assert store.put_candidate(
        receipt,
        certificate,
        created_at_ms=NOW_MS,
    ).stored

    # A mutable hint may be replayed after its local index bit is revoked.  It
    # still cannot override current durable policy authority at admission.
    assert store.index.revoke_certificate(
        locator.locator_id,
        certificate.certificate_id,
    ).published
    replay = store.index.publish(
        locator.locator_id,
        certificate_cid=certificate.certificate_id,
        receipt_cid=receipt.receipt_id,
        created_at_ms=NOW_MS,
        metadata={"claimed_authoritative": True},
    )
    assert replay.published
    revoked_policy = _policy(
        certificate,
        revoked_certificate_cids=(certificate.certificate_id,),
    )
    replayed = _cache(
        certificate,
        candidate_provider=store.candidate_provider,
        policy=revoked_policy,
    ).lookup(locator, execution_key)
    _assert_run_executes_real_test(replayed)
    assert replayed.reason_code is ReuseReasonCode.EXPIRED_OR_REVOKED

    # Deterministically revoke while admission is in flight.  The verifier is
    # never reached after the revocation authority answers True.
    checker_entered = threading.Event()
    checker_release = threading.Event()
    revoked = threading.Event()
    verifier_calls: list[str] = []

    def racing_revocation(*_args: Any) -> bool:
        checker_entered.set()
        assert checker_release.wait(timeout=5)
        return revoked.is_set()

    def counted_verifier(*_args: Any) -> bool:
        verifier_calls.append("called")
        return True

    candidate = TestProofCache.candidate(receipt, certificate)
    racing_cache = _cache(
        certificate,
        verifier=counted_verifier,
        revocation_checker=racing_revocation,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            racing_cache.lookup,
            locator,
            execution_key,
            candidates=[candidate],
        )
        assert checker_entered.wait(timeout=5)
        revoked.set()
        checker_release.set()
        raced = future.result(timeout=10)

    _assert_run_executes_real_test(raced)
    assert raced.reason_code is ReuseReasonCode.EXPIRED_OR_REVOKED
    assert verifier_calls == []
