"""Controller atomic issuance and publication transaction tests (PTR-147)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import ipfs_accelerate_py.testing.proof_reuse.publication as publication_module

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
from ipfs_accelerate_py.testing.proof_reuse.services import (
    DATASETS_GROTH16_LOCKED_SOURCE_IDENTITY,
    DATASETS_GROTH16_RELEASE_MANIFESTS_SHA256,
    DATASETS_GROTH16_TEST_PASS_CAPABILITY_PAYLOAD_SHA256,
    DATASETS_GROTH16_REVIEWED_SOURCE_FINGERPRINT,
    DATASETS_VERIFIER_REVISION,
    GROTH16_TEST_PASS_ARTIFACT_MANIFEST_INTERFACE,
    TEST_PASS_GROTH16_CIRCUIT_CID,
    TEST_PASS_GROTH16_CIRCUIT_IDENTITY_SHA256,
    TEST_PASS_GROTH16_PROVIDER_SOURCE_SHA256,
)
from ipfs_accelerate_py.testing.proof_reuse.reporting import ProofReuseSessionMetrics
from ipfs_accelerate_py.testing.proof_reuse.xdist import (
    ProofReusePublicationIntent,
    ProofReuseXdistCoordinator,
)

ACCELERATE_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = ACCELERATE_ROOT.parent / "ipfs_datasets"


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


def _test_ready_bindings(certificate: TestProofCertificate) -> Groth16ArtifactIdentityBindings:
    """Explicit non-production fixture for transaction sequencing tests."""

    return Groth16ArtifactIdentityBindings(
        circuit_cid=certificate.circuit_cid,
        verifying_key_cid=certificate.verifying_key_cid,
        artifacts_root="/test-only/reviewed-v4-artifacts",
        verifying_key_sha256="a" * 64,
        proving_key_sha256="b" * 64,
        backend_circuit_version=4,
        reviewed_revision=DATASETS_VERIFIER_REVISION,
        provenance_ready=True,
        reason_code="test_only_ready_fixture",
    )


def _patch_test_only_cryptographic_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def verified(_certificate: Any, **kwargs: Any) -> tuple[bool, str]:
        calls.append(dict(kwargs))
        assert kwargs.get("require_cryptographic_verify") is True
        assert kwargs["bindings"].provenance_ready is True
        return True, "verified_test_double"

    monkeypatch.setattr(publication_module, "_local_verify_certificate", verified)
    return calls


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


def test_arbitrary_v4_key_bytes_are_never_provenance(tmp_path: Path) -> None:
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
    assert bindings.provenance_ready is False
    assert bindings.reason_code == "artifact_manifest_pin_missing"
    assert bindings.circuit_cid == ""
    assert bindings.verifying_key_cid == ""
    assert bindings.diagnostics["arbitrary_keys_non_authoritative"] is True


def test_test_only_reviewed_manifest_binds_keys_provider_and_ptr151_native(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts_root = tmp_path / "artifacts"
    version = artifacts_root / "v4"
    version.mkdir(parents=True)
    pk = b"test-only-reviewed-proving-key"
    vk = b"test-only-reviewed-verifying-key"
    (version / "proving_key.bin").write_bytes(pk)
    (version / "verifying_key.bin").write_bytes(vk)
    binary = (
        DATASETS_ROOT
        / "ipfs_datasets_py"
        / "processors"
        / "groth16_backend"
        / "bin"
        / "linux-aarch64"
        / "groth16"
    )
    native_bytes = binary.read_bytes()
    manifest_payload = {
        "interface": GROTH16_TEST_PASS_ARTIFACT_MANIFEST_INTERFACE,
        "reviewed_datasets_revision": DATASETS_VERIFIER_REVISION,
        "reviewed_source_fingerprint": DATASETS_GROTH16_REVIEWED_SOURCE_FINGERPRINT,
        "provider_source_sha256": TEST_PASS_GROTH16_PROVIDER_SOURCE_SHA256,
        "circuit": {
            "version": 4,
            "identity_sha256": TEST_PASS_GROTH16_CIRCUIT_IDENTITY_SHA256,
            "circuit_cid": TEST_PASS_GROTH16_CIRCUIT_CID,
            "proof_system": "groth16",
            "ruleset_id": "test_pass_v2",
            "statement_interface": "TestPassStatementV2",
            "statement_version": 2,
        },
        "artifacts": {
            "proving_key": {
                "relative_path": "v4/proving_key.bin",
                "sha256": hashlib.sha256(pk).hexdigest(),
                "size": len(pk),
            },
            "verifying_key": {
                "relative_path": "v4/verifying_key.bin",
                "sha256": hashlib.sha256(vk).hexdigest(),
                "size": len(vk),
            },
        },
        "native": {
            "provenance": "reviewed_bundled_release",
            "binary_sha256": hashlib.sha256(native_bytes).hexdigest(),
            "binary_size": len(native_bytes),
            "supported_circuit_versions": [1, 2, 3, 4],
            "release_manifest_sha256": (
                DATASETS_GROTH16_RELEASE_MANIFESTS_SHA256["linux-aarch64"]
            ),
            "capability_payload_sha256": (
                DATASETS_GROTH16_TEST_PASS_CAPABILITY_PAYLOAD_SHA256
            ),
            "locked_source_identity": DATASETS_GROTH16_LOCKED_SOURCE_IDENTITY,
        },
    }
    manifest = tmp_path / "test-only-approved-manifest.json"
    manifest.write_text(
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(
        publication_module,
        "DATASETS_GROTH16_APPROVED_V4_KEY_MANIFESTS_SHA256",
        frozenset({manifest_digest}),
    )
    monkeypatch.setattr(
        publication_module,
        "_probe_native_capabilities",
        lambda *_args, **_kwargs: (True, "ready"),
    )
    original_datasets_modules = {
        module_name: module
        for module_name, module in tuple(sys.modules.items())
        if module_name == "ipfs_datasets_py"
        or module_name.startswith("ipfs_datasets_py.")
    }
    for module_name in original_datasets_modules:
        sys.modules.pop(module_name, None)
    monkeypatch.syspath_prepend(str(DATASETS_ROOT))
    monkeypatch.setenv("IPFS_DATASETS_PY_MINIMAL_IMPORTS", "1")
    env = {
        "GROTH16_BACKEND_ARTIFACTS_ROOT": str(artifacts_root),
        "IPFS_DATASETS_GROTH16_BINARY": str(binary),
        "IPFS_TEST_PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST": str(manifest),
        "IPFS_TEST_PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST_SHA256": manifest_digest,
    }

    bindings = Groth16ArtifactIdentityBindings.from_activated_artifacts(
        artifacts_root=artifacts_root,
        binary_path=binary,
        environ=env,
    )
    for module_name in tuple(sys.modules):
        if module_name == "ipfs_datasets_py" or module_name.startswith(
            "ipfs_datasets_py."
        ):
            sys.modules.pop(module_name, None)
    sys.modules.update(original_datasets_modules)

    assert bindings.provenance_ready is True
    assert bindings.reason_code == "ready"
    assert bindings.circuit_cid == TEST_PASS_GROTH16_CIRCUIT_CID
    assert bindings.proving_key_sha256 == hashlib.sha256(pk).hexdigest()
    assert bindings.verifying_key_sha256 == hashlib.sha256(vk).hexdigest()
    assert bindings.diagnostics["native_v4_capability_validated"] is True


def test_controller_transaction_retains_then_defers_positive_v4_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _admitted_receipt()
    certificate = _certificate(receipt)
    bindings = _test_ready_bindings(certificate)
    verification_calls = _patch_test_only_cryptographic_verifier(monkeypatch)
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
        artifact_bindings=bindings,
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
    assert result.published is False
    assert result.put_candidate_called is False
    assert result.indexed is False
    assert result.reason_code == "positive_v4_publication_pending_ptr155"
    assert result.action == "DEFERRED"
    assert result.non_authoritative_retained is True
    assert issued == []
    assert verification_calls == []
    assert store.calls == []
    # Cold retention wrote at least the receipt bytes to the candidate store.
    assert candidate_store.blobs or candidate_store.publishes


def test_self_asserted_bindings_and_fake_verifier_never_reach_candidate_store() -> None:
    calls: list[Any] = []

    class _Store:
        def put_candidate(self, *args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))
            return SimpleNamespace(stored=True, indexed=True)

    class _Issuer:
        def verify_certificate_locally(self, *_args: Any) -> Any:
            return SimpleNamespace(
                verified=True,
                authoritative=True,
                can_authorize_skip=True,
                status="verified",
                authority="authoritative",
            )

    bindings = Groth16ArtifactIdentityBindings(
        circuit_cid="cid:circuit",
        verifying_key_cid="cid:verifying-key",
        artifacts_root="/unreviewed",
        verifying_key_sha256="a" * 64,
        proving_key_sha256="b" * 64,
        backend_circuit_version=4,
        reviewed_revision=DATASETS_VERIFIER_REVISION,
        provenance_ready=True,
        reason_code="forged",
    )
    receipt = _admitted_receipt()
    certificate = _certificate(receipt)
    intent = ProofReusePublicationIntent.from_receipt(
        receipt,
        certificate=certificate.to_dict(),
        certificate_cid=certificate.certificate_id,
    )
    candidate_store = _CandidateStore()
    result = ProofReuseControllerPublicationTransaction(
        store=_Store(),
        candidate_store=candidate_store,
        issuer=_Issuer(),
        artifact_bindings=bindings,
    ).publish_intent(intent)

    assert result.published is False
    assert result.indexed is False
    assert result.put_candidate_called is False
    assert result.reason_code == "positive_v4_publication_pending_ptr155"
    assert result.certificate_cid == certificate.certificate_id
    assert calls == []
    assert any(
        json.loads(payload).get("interface") == "TestProofCertificate@1"
        for payload in candidate_store.blobs
    )


def test_flush_retains_receipt_without_positive_v4_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _admitted_receipt()
    certificate = _certificate(receipt)
    bindings = _test_ready_bindings(certificate)
    _patch_test_only_cryptographic_verifier(monkeypatch)

    class _Issuer:
        last_artifact_bindings = bindings

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
    assert names == ["put_receipt"]
    assert controller.healthy is True


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


def test_structural_certificate_without_artifact_provenance_never_indexes() -> None:
    receipt = _admitted_receipt()
    certificate = _certificate(receipt)
    store = _AtomicStore()
    tx = ProofReuseControllerPublicationTransaction(store=store)
    intent = ProofReusePublicationIntent.from_receipt(
        receipt,
        certificate=certificate.to_dict(),
        certificate_cid=certificate.certificate_id,
    )

    result = tx.publish_intent(intent)

    assert result.published is False
    assert result.reason_code == "positive_v4_publication_pending_ptr155"
    assert result.action == "DEFERRED"
    assert [name for name, _payload in store.calls] == []


def test_pending_positive_v4_never_probes_candidate_store_or_fences_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _admitted_receipt()
    certificate = _certificate(receipt)
    bindings = _test_ready_bindings(certificate)
    _patch_test_only_cryptographic_verifier(monkeypatch)

    class _Issuer:
        last_artifact_bindings = bindings

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
    assert published == (ProofReusePublicationIntent.from_receipt(receipt).intent_id,)
    assert [name for name, _payload in store.calls] == ["put_receipt"]
    assert controller.healthy is True
    assert controller.can_write is True


def test_put_candidate_once_does_not_retry_internal_type_error() -> None:
    calls: list[int] = []

    class _TypeErrorStore:
        def put_candidate(self, *_args: Any, **_kwargs: Any) -> Any:
            calls.append(1)
            raise TypeError("internal store failure after write began")

    transaction = ProofReuseControllerPublicationTransaction(
        store=_TypeErrorStore(),
        owner_id="controller:test",
    )
    result = transaction._put_candidate_once(
        receipt={"receipt_id": "cid:r"},
        certificate={"certificate_id": "cid:c"},
        locator_cid="cid:l",
    )

    assert result == (False, False, "put_candidate_failed")
    assert calls == [1]


def test_put_candidate_once_rejects_untyped_truthy_result() -> None:
    calls: list[int] = []

    class _TruthyStore:
        def put_candidate(self, _receipt: Any, _certificate: Any) -> Any:
            calls.append(1)
            return object()

    transaction = ProofReuseControllerPublicationTransaction(store=_TruthyStore())
    result = transaction._put_candidate_once(
        receipt={"receipt_id": "cid:r"},
        certificate={"certificate_id": "cid:c"},
        locator_cid="cid:l",
    )

    assert result == (False, False, "put_candidate_rejected")
    assert calls == [1]


def test_mismatched_artifact_provenance_returns_deferred(tmp_path: Path) -> None:
    receipt = _admitted_receipt()
    certificate = _certificate(receipt)
    # This test isolates the controller's pin comparison.  Production-ready
    # bindings can only be constructed through the reviewed manifest path.
    bindings = Groth16ArtifactIdentityBindings(
        circuit_cid="cid:different-circuit",
        verifying_key_cid="cid:different-verifying-key",
        artifacts_root=str(tmp_path.resolve()),
        verifying_key_sha256="a" * 64,
        proving_key_sha256="b" * 64,
        backend_circuit_version=4,
        reviewed_revision=DATASETS_VERIFIER_REVISION,
        provenance_ready=True,
        reason_code="test_fixture",
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
    # The PTR-155 authority gate runs before any mutable provider verification.
    assert result.published is False
    assert result.action == "DEFERRED"
    assert result.reason_code == "positive_v4_publication_pending_ptr155"
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
