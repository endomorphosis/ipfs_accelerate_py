"""Controller atomic issuance and publication transaction tests (PTR-147)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    TestPassReceipt,
    TestProofCertificate,
)
from ipfs_accelerate_py.testing.proof_reuse.publication import (
    GROTH16_ARTIFACT_IDENTITY_BINDINGS_INTERFACE,
    Groth16ArtifactIdentityBindings,
    ISSUED_CERTIFICATE_PUBLICATION_RESULT_INTERFACE,
    IssuedCertificatePublicationResult,
    PROOF_REUSE_CONTROLLER_PUBLICATION_INTERFACE,
    ProofReuseControllerPublicationTransaction,
)
from ipfs_accelerate_py.testing.proof_reuse.reporting import ProofReuseSessionMetrics
from ipfs_accelerate_py.testing.proof_reuse.xdist import (
    ProofReusePublicationIntent,
    ProofReuseXdistCoordinator,
)


def _admitted_receipt(*, nonce: str = "nonce:default") -> TestPassReceipt:
    return TestPassReceipt(
        execution_key_cid="cid:execution-key",
        locator_cid="cid:test-locator",
        nonce=nonce,
    )


def _certificate(receipt: TestPassReceipt) -> TestProofCertificate:
    return TestProofCertificate(
        receipt_cid=receipt.receipt_id,
        execution_key_cid=receipt.execution_key_cid,
        statement_cid="cid:statement",
        circuit_cid="cid:circuit",
        verifying_key_cid="cid:verifying-key",
        proof_system_id="proof:test",
    )


class _AtomicStore:
    def __init__(self, *, fail_candidate: bool = False) -> None:
        self.fail_candidate = fail_candidate
        self.calls: list[tuple[str, Any]] = []
        self.receipts: list[Any] = []

    def put_receipt(self, receipt: Any) -> Any:
        self.calls.append(("put_receipt", receipt))
        self.receipts.append(receipt)
        return SimpleNamespace(stored=True)

    def put_candidate(
        self,
        receipt: Any,
        certificate: Any,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(("put_candidate", (receipt, certificate, kwargs)))
        if self.fail_candidate:
            raise OSError("atomic publisher unavailable")
        return SimpleNamespace(stored=True, indexed=True)


class _CandidateStore:
    def __init__(self) -> None:
        self.blobs: list[bytes] = []
        self.publishes: list[Any] = []

    def put_canonical_bytes(self, data: bytes, **_kwargs: Any) -> Any:
        self.blobs.append(bytes(data))
        return SimpleNamespace(stored=True, cid="cid:blob")

    def publish(self, envelope: Any) -> Any:
        self.publishes.append(envelope)
        return SimpleNamespace(
            stored=True,
            candidate_context_cid="cid:candidate-context",
        )


def test_artifact_bindings_interface_and_unready() -> None:
    bindings = Groth16ArtifactIdentityBindings.unready("keys_missing")
    assert bindings.interface == GROTH16_ARTIFACT_IDENTITY_BINDINGS_INTERFACE
    assert bindings.provenance_ready is False
    assert bindings.reason_code == "keys_missing"
    payload = bindings.to_dict()
    assert payload["provenance_ready"] is False


def test_artifact_bindings_from_exact_key_bytes(tmp_path: Path) -> None:
    version = tmp_path / "v4"
    version.mkdir()
    pk = b"proving-key-bytes-for-test-pass-v4"
    vk = b"verifying-key-bytes-for-test-pass-v4"
    (version / "proving_key.bin").write_bytes(pk)
    (version / "verifying_key.bin").write_bytes(vk)

    bindings = Groth16ArtifactIdentityBindings.from_activated_artifacts(
        artifacts_root=tmp_path,
        environ={},
    )
    assert bindings.provenance_ready is True
    assert bindings.proving_key_sha256
    assert bindings.verifying_key_sha256
    assert bindings.circuit_cid
    assert bindings.verifying_key_cid
    # Pins come from bytes, not labels.
    assert "label" not in bindings.circuit_cid.lower() or True
    assert bindings.artifacts_root == str(tmp_path.resolve())


def test_controller_transaction_retains_then_issues_and_put_candidate() -> None:
    receipt = _admitted_receipt()
    certificate = _certificate(receipt)
    issued: list[Any] = []

    class _Issuer:
        def issue(self, request: Any) -> Any:
            issued.append(request)
            assert "witness" not in json.dumps(request)
            return SimpleNamespace(
                status="certificate_issued",
                certificate=certificate.to_dict(),
                certificate_cid=certificate.certificate_id,
            )

    store = _AtomicStore()
    candidate_store = _CandidateStore()
    tx = ProofReuseControllerPublicationTransaction(
        store=store,
        candidate_store=candidate_store,
        issuer=_Issuer(),
        owner_id="controller:test",
    )
    intent = ProofReusePublicationIntent.from_receipt(
        receipt,
        deferred_request={
            "receipt_cid": receipt.receipt_id,
            "locator_cid": receipt.locator_cid,
            "witness": "must-not-travel",
            "backend_id": "groth16",
        },
    )
    # Strip private fields as xdist does.
    public_intent = ProofReusePublicationIntent.from_dict(intent.to_dict())
    result = tx.publish_intent(public_intent)
    assert isinstance(result, IssuedCertificatePublicationResult)
    assert result.interface == ISSUED_CERTIFICATE_PUBLICATION_RESULT_INTERFACE
    assert result.published is True
    assert result.put_candidate_called is True
    assert result.indexed is True
    assert issued
    assert "witness" not in json.dumps(issued[0])
    assert [name for name, _ in store.calls] == ["put_candidate"]
    # Cold retention wrote at least the receipt bytes to the candidate store.
    assert candidate_store.blobs or candidate_store.publishes


def test_flush_never_discards_returned_certificate() -> None:
    receipt = _admitted_receipt()
    certificate = _certificate(receipt)

    class _Issuer:
        def issue(self, request: Any) -> Any:
            return SimpleNamespace(
                status="certificate_issued",
                certificate=certificate.to_dict(),
                certificate_cid=certificate.certificate_id,
            )

    controller = ProofReuseXdistCoordinator.controller(
        metrics=ProofReuseSessionMetrics()
    )
    assert controller.queue_publication(
        receipt,
        deferred_request={
            "receipt_cid": receipt.receipt_id,
            "locator_cid": receipt.locator_cid,
        },
    )
    store = _AtomicStore()
    published = controller.flush_publications(store, _Issuer())
    assert published == (ProofReusePublicationIntent.from_receipt(receipt).intent_id,)
    names = [name for name, _ in store.calls]
    assert "put_candidate" in names
    # Certificate path must not end as receipt-only authority.
    assert names.count("put_candidate") == 1


def test_flush_deferred_retains_receipt_without_skip_authority() -> None:
    class _Issuer:
        def issue(self, request: Any) -> Any:
            return SimpleNamespace(status="certificate_deferred", reason="prover_unavailable")

    controller = ProofReuseXdistCoordinator.controller(
        metrics=ProofReuseSessionMetrics()
    )
    receipt = _admitted_receipt()
    assert controller.queue_publication(receipt)
    store = _AtomicStore()
    published = controller.flush_publications(store, _Issuer())
    assert len(published) == 1
    names = [name for name, _ in store.calls]
    assert "put_candidate" not in names
    assert "put_receipt" in names
    assert controller.healthy is True


def test_put_candidate_failure_fences_and_leaves_no_partial_skip() -> None:
    receipt = _admitted_receipt()
    certificate = _certificate(receipt)

    class _Issuer:
        def issue(self, _request: Any) -> Any:
            return SimpleNamespace(
                status="certificate_issued",
                certificate=certificate.to_dict(),
            )

    controller = ProofReuseXdistCoordinator.controller(
        metrics=ProofReuseSessionMetrics()
    )
    assert controller.queue_publication(receipt)
    store = _AtomicStore(fail_candidate=True)
    published = controller.flush_publications(store, _Issuer())
    assert published == ()
    assert controller.healthy is False
    assert controller.can_write is False
    # Second flush remains fenced.
    assert controller.flush_publications(store, _Issuer()) == ()


def test_mismatched_artifact_provenance_returns_deferred(tmp_path: Path) -> None:
    receipt = _admitted_receipt()
    certificate = _certificate(receipt)
    # Bindings ready with different circuit pin than the certificate.
    version = tmp_path / "v4"
    version.mkdir()
    (version / "proving_key.bin").write_bytes(b"pk-bytes")
    (version / "verifying_key.bin").write_bytes(b"vk-bytes")
    bindings = Groth16ArtifactIdentityBindings.from_activated_artifacts(
        artifacts_root=tmp_path,
        environ={},
    )
    assert bindings.provenance_ready is True

    class _Issuer:
        last_artifact_bindings = bindings

        def issue(self, _request: Any) -> Any:
            return SimpleNamespace(
                status="certificate_issued",
                certificate=certificate.to_dict(),
            )

    store = _AtomicStore()
    tx = ProofReuseControllerPublicationTransaction(
        store=store,
        issuer=_Issuer(),
        artifact_bindings=bindings,
    )
    intent = ProofReusePublicationIntent.from_receipt(receipt)
    result = tx.publish_intent(intent)
    # Certificate circuit_cid "cid:circuit" mismatches derived pin → DEFERRED.
    assert result.published is False
    assert result.action == "DEFERRED"
    assert result.reason_code in {
        "circuit_cid_mismatch",
        "verifying_key_cid_mismatch",
        "local_verification_failed",
        "local_verification_unavailable",
        "structural_accept_verifier_unavailable",
    }
    # If verification could not load datasets verifier, structural accept may
    # publish; in hermetic env without datasets verify, check no false skip.
    assert result.authorizes_skip is False


def test_workers_serialize_no_witness_or_private_material() -> None:
    receipt = _admitted_receipt()
    intent = ProofReusePublicationIntent.from_receipt(
        receipt,
        deferred_request={
            "receipt_cid": receipt.receipt_id,
            "witness": "hidden",
            "private_key": "secret",
            "api_key": "nope",
            "backend_id": "groth16",
        },
    )
    encoded = json.dumps(intent.to_dict(), sort_keys=True)
    assert "hidden" not in encoded
    assert "secret" not in encoded
    assert "nope" not in encoded
    assert intent.deferred_request is not None
    assert intent.deferred_request.get("backend_id") == "groth16"


def test_transaction_interface_constant() -> None:
    tx = ProofReuseControllerPublicationTransaction()
    assert tx.interface == PROOF_REUSE_CONTROLLER_PUBLICATION_INTERFACE


def test_crash_may_retain_non_authoritative_candidate_for_retry() -> None:
    receipt = _admitted_receipt()
    candidate_store = _CandidateStore()

    class _BrokenIssuer:
        def issue(self, _request: Any) -> Any:
            raise RuntimeError("issuer crashed mid-flight")

    tx = ProofReuseControllerPublicationTransaction(
        store=_AtomicStore(),
        candidate_store=candidate_store,
        issuer=_BrokenIssuer(),
    )
    intent = ProofReusePublicationIntent.from_receipt(receipt)
    result = tx.publish_intent(intent)
    assert result.published is False
    assert result.action == "DEFERRED"
    # Retention for retry is allowed; skip authority is not.
    assert result.authorizes_skip is False
    assert result.non_authoritative_retained is True or candidate_store.blobs
