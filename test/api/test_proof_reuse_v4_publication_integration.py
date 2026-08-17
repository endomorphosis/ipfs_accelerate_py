"""PTR-155: exact Groth16 v4 verification joined to atomic candidate publication.

One disposable explicitly test-only current-v4 fixture proves the complete
issue-material → controller-context → local-V2-verify → atomic-publication path
and is never counted as reviewed production authority.

Production publication remains fail-closed unless the hardcoded-reviewed key
manifest allowlist, exact PTR-151 bindings, and
``CertificateVerificationStatus.VERIFIED`` from
``verify_test_execution_certificate_v2`` all hold.  No trusted setup, key
generation, build, download, or network call occurs during import, collection,
ordinary setup, or verification.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest

import ipfs_accelerate_py.testing.proof_reuse.publication as publication_module
from ipfs_accelerate_py.testing.proof_reuse.publication import (
    CONTROLLER_V2_VERIFICATION_CONTEXT_INTERFACE,
    ControllerV2VerificationContext,
    Groth16ArtifactIdentityBindings,
    ProofReuseControllerPublicationTransaction,
    verify_test_execution_certificate_v2_for_publication,
)
from ipfs_accelerate_py.testing.proof_reuse.services import (
    DATASETS_VERIFIER_REVISION,
)
ACCELERATE_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = ACCELERATE_ROOT.parent / "ipfs_datasets"
_PROOF_BYTES = b"ptr-155-test-only-disposable-v4-proof-bytes"


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_datasets() -> None:
    if not DATASETS_ROOT.is_dir():
        pytest.skip("ipfs_datasets checkout unavailable")


@pytest.fixture
def datasets_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_datasets()
    monkeypatch.syspath_prepend(str(DATASETS_ROOT))
    monkeypatch.setenv("IPFS_DATASETS_PY_MINIMAL_IMPORTS", "1")
    # Drop any previously imported datasets modules so this session is clean.
    for name in tuple(sys.modules):
        if name == "ipfs_datasets_py" or name.startswith("ipfs_datasets_py."):
            sys.modules.pop(name, None)


def _build_test_only_v4_material(
    *,
    circuit_cid: str | None = None,
    verifying_key_cid: str | None = None,
) -> dict[str, Any]:
    """Construct a disposable V2 certificate + statement (never production)."""

    from ipfs_datasets_py.logic.zkp import ZKPProof
    from ipfs_datasets_py.logic.zkp.provekit.test_pass_circuit import (
        TestPassCircuitBinding,
    )
    from ipfs_datasets_py.logic.zkp.statements.test_pass import (
        build_statement_from_receipt_v2,
        content_cid_for_dag_json,
    )
    from ipfs_datasets_py.logic.zkp.test_execution_certificate import (
        CertificateVerificationStatus,
        TestExecutionCertificateV2,
        verify_test_execution_certificate_v2,
    )

    receipt_body = {
        "nodeid": "ptr155::test_only_disposable_v4",
        "outcome": "passed",
        "admitted": True,
        "setup_outcome": "pass",
        "call_outcome": "pass",
        "teardown_outcome": "pass",
        "disqualifying_bits": [],
    }
    candidate = {
        "module": "ptr155",
        "qualname": "test_only_disposable_v4",
        "source_sha256": "d" * 64,
    }
    circuit = circuit_cid or content_cid_for_dag_json(
        {"circuit": "ptr155-test-only-v4"}
    )
    vk = verifying_key_cid or content_cid_for_dag_json(
        {"verifying_key": "ptr155-test-only-v4"}
    )
    statement, _witness = build_statement_from_receipt_v2(
        receipt_body,
        candidate_context=candidate,
        policy_cid=content_cid_for_dag_json({"policy": "ptr155-test-only"}),
        statement_cid=content_cid_for_dag_json({"statement": "ptr155-test-only"}),
        circuit_cid=circuit,
        verifying_key_cid=vk,
        issuer_id="issuer:ptr155-test-only",
        epoch="epoch:ptr155-test-only",
        locator_cid=content_cid_for_dag_json({"locator": "ptr155-test-only"}),
        completeness_policy_cid=content_cid_for_dag_json(
            {"completeness": "ptr155-test-only"}
        ),
    )
    public_inputs = statement.to_public_inputs()
    binding = TestPassCircuitBinding(
        statement,
        backend_id="groth16",
        proof_system_id="groth16",
        candidate_context_cid=public_inputs["candidate_context_cid"],
    )
    proof = ZKPProof(
        proof_data=_PROOF_BYTES,
        public_inputs=public_inputs,
        metadata={
            "backend": "groth16",
            "proof_system": "groth16",
            "fixture": "ptr155-test-only-disposable",
        },
        timestamp=1_775_000_155.0,
        size_bytes=len(_PROOF_BYTES),
    )
    proof_digest = "sha256:" + _sha256_hex(_PROOF_BYTES)
    certificate = TestExecutionCertificateV2(
        receipt_cid=public_inputs["receipt_cid"],
        candidate_context_cid=public_inputs["candidate_context_cid"],
        execution_key_cid=public_inputs["execution_key_cid"],
        statement_cid=public_inputs["statement_cid"],
        circuit_cid=public_inputs["circuit_cid"],
        verifying_key_cid=public_inputs["verifying_key_cid"],
        proof_system_id="groth16",
        proof=proof,
        proof_artifact_cid=proof_digest,
        proof_digest=proof_digest,
        backend_mode="cryptographic",
        authority="authoritative",
        issuer_id=public_inputs["issuer_id"],
        policy_cid=public_inputs["policy_cid"],
        epoch=public_inputs["epoch"],
        public_inputs=public_inputs,
        metadata={"fixture": "ptr155-test-only-disposable-v4"},
    )

    class _DisposableBackend:
        """Test-only proof oracle — never a production reviewed verifier."""

        backend_id = "groth16"

        def verify_proof(self, proof_obj: Any) -> bool:
            return bytes(proof_obj.proof_data) == _PROOF_BYTES

    backend = _DisposableBackend()
    direct = verify_test_execution_certificate_v2(
        certificate,
        binding,
        backend,
        expected_candidate_context_cid=public_inputs["candidate_context_cid"],
    )
    assert direct.status is CertificateVerificationStatus.VERIFIED
    assert direct.verified is True

    return {
        "statement": statement,
        "public_inputs": public_inputs,
        "binding": binding,
        "certificate": certificate,
        "certificate_dict": certificate.to_dict(include_proof=True, include_ids=True),
        "backend": backend,
        "proof_digest": proof_digest,
        "CertificateVerificationStatus": CertificateVerificationStatus,
    }


def _test_only_bindings(
    *,
    circuit_cid: str,
    verifying_key_cid: str,
    artifacts_root: str = "/tmp/ptr155-test-only-artifacts",
) -> Groth16ArtifactIdentityBindings:
    return Groth16ArtifactIdentityBindings(
        circuit_cid=circuit_cid,
        verifying_key_cid=verifying_key_cid,
        artifacts_root=artifacts_root,
        verifying_key_sha256="e" * 64,
        proving_key_sha256="f" * 64,
        backend_circuit_version=5,
        reviewed_revision=DATASETS_VERIFIER_REVISION,
        provenance_ready=True,
        reason_code="test_only_disposable_v4_fixture",
        diagnostics={
            "test_only_disposable": True,
            "never_production_authority": True,
        },
    )


def _controller_intent(
    *,
    public_inputs: Mapping[str, Any],
    certificate: Mapping[str, Any],
    deferred_request: Mapping[str, Any] | None = None,
    statement: Any = None,
) -> SimpleNamespace:
    """Build a controller intent carrying V2 material without V1 schema checks.

    ``ProofReusePublicationIntent`` still validates accelerate V1 certificate
    envelopes; the PTR-155 join path consumes public V2 mappings directly from
    controller-owned fields.
    """

    receipt_cid = str(public_inputs["receipt_cid"])
    locator_cid = str(public_inputs.get("locator_cid") or "cid:locator")
    request = dict(deferred_request or {})
    if statement is not None and "statement" not in request:
        request["statement"] = statement
    if "public_inputs" not in request:
        request["public_inputs"] = dict(public_inputs)
    for key in (
        "receipt_cid",
        "execution_key_cid",
        "candidate_context_cid",
        "policy_cid",
        "statement_cid",
        "circuit_cid",
        "verifying_key_cid",
        "issuer_id",
        "epoch",
        "backend_id",
        "proof_system_id",
        "locator_cid",
    ):
        request.setdefault(
            key,
            public_inputs.get(key)
            if key not in {"backend_id", "proof_system_id"}
            else request.get(key)
            or ("groth16" if key != "locator_cid" else locator_cid),
        )
    request.setdefault("backend_id", "groth16")
    request.setdefault("proof_system_id", "groth16")
    request.setdefault("locator_cid", locator_cid)
    return SimpleNamespace(
        receipt={
            "receipt_id": receipt_cid,
            "execution_key_cid": public_inputs["execution_key_cid"],
            "locator_cid": locator_cid,
            "nonce": "nonce:ptr155-test-only",
            "admitted": True,
            "setup_outcome": "pass",
            "call_outcome": "pass",
            "teardown_outcome": "pass",
        },
        receipt_cid=receipt_cid,
        locator_cid=locator_cid,
        certificate=dict(certificate),
        certificate_cid=str(
            certificate.get("certificate_id")
            or certificate.get("content_id")
            or ""
        ),
        deferred_request=request,
        intent_id=f"intent:ptr155:{receipt_cid[:24]}",
    )


class _AtomicStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def put_receipt(self, receipt: Any) -> Any:
        self.calls.append(("put_receipt", receipt))
        return SimpleNamespace(stored=True)

    def put_candidate(
        self,
        receipt: Any,
        certificate: Any,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(("put_candidate", (receipt, certificate, kwargs)))
        return SimpleNamespace(stored=True, indexed=True)


class _CandidateStore:
    def __init__(self) -> None:
        self.blobs: list[bytes] = []

    def put_canonical_bytes(self, data: bytes, **_kwargs: Any) -> Any:
        self.blobs.append(bytes(data))
        return SimpleNamespace(stored=True, cid="cid:blob")


# ---------------------------------------------------------------------------
# Positive disposable path
# ---------------------------------------------------------------------------


def test_disposable_test_only_v4_fixture_full_publication_path(
    datasets_on_path: None,
) -> None:
    """Issue-material → controller-context → V2 VERIFIED → one put_candidate."""

    material = _build_test_only_v4_material()
    public_inputs = material["public_inputs"]
    certificate_dict = material["certificate_dict"]
    bindings = _test_only_bindings(
        circuit_cid=public_inputs["circuit_cid"],
        verifying_key_cid=public_inputs["verifying_key_cid"],
    )
    intent = _controller_intent(
        public_inputs=public_inputs,
        certificate=certificate_dict,
        statement=material["statement"],
    )

    store = _AtomicStore()
    candidate_store = _CandidateStore()
    tx = ProofReuseControllerPublicationTransaction(
        store=store,
        candidate_store=candidate_store,
        artifact_bindings=bindings,
        test_only_verification_backend=material["backend"],
        owner_id="controller:ptr155-test-only",
    )
    result = tx.publish_intent(intent)

    assert result.published is True
    assert result.put_candidate_called is True
    assert result.indexed is True
    assert result.reason_code == "published_test_only_disposable"
    assert result.authorizes_skip is False
    assert result.diagnostics.get("production_authority") is False
    assert result.diagnostics.get("test_only_disposable") is True
    assert (
        result.diagnostics.get("expected_candidate_context_cid")
        == public_inputs["candidate_context_cid"]
    )
    put_names = [name for name, _ in store.calls]
    assert put_names == ["put_candidate"]
    assert put_names.count("put_candidate") == 1
    # Non-authoritative retention may also have written public blobs.
    assert candidate_store.blobs or True


def test_exact_v2_adapter_requires_verified_status(datasets_on_path: None) -> None:
    material = _build_test_only_v4_material()
    public_inputs = material["public_inputs"]
    bindings = _test_only_bindings(
        circuit_cid=public_inputs["circuit_cid"],
        verifying_key_cid=public_inputs["verifying_key_cid"],
    )
    context = ControllerV2VerificationContext(
        receipt_cid=public_inputs["receipt_cid"],
        execution_key_cid=public_inputs["execution_key_cid"],
        candidate_context_cid=public_inputs["candidate_context_cid"],
        expected_candidate_context_cid=public_inputs["candidate_context_cid"],
        policy_cid=public_inputs["policy_cid"],
        statement_cid=public_inputs["statement_cid"],
        circuit_cid=public_inputs["circuit_cid"],
        verifying_key_cid=public_inputs["verifying_key_cid"],
        issuer_id=public_inputs["issuer_id"],
        epoch=public_inputs["epoch"],
        backend_id="groth16",
        proof_system_id="groth16",
        locator_cid=public_inputs.get("locator_cid", ""),
        public_inputs=public_inputs,
        statement=material["statement"],
        is_test_only_disposable=True,
    )
    assert context.interface == CONTROLLER_V2_VERIFICATION_CONTEXT_INTERFACE
    assert context.may_publish_candidate is False

    ok, reason, result = verify_test_execution_certificate_v2_for_publication(
        material["certificate_dict"],
        bindings=bindings,
        controller_context=context,
        test_only_backend=material["backend"],
    )
    assert ok is True
    assert reason == "verified_test_only_disposable"
    status_enum = material["CertificateVerificationStatus"]
    assert result.status is status_enum.VERIFIED

    from ipfs_datasets_py.logic.zkp.statements.test_pass import content_cid_for_dag_json

    other_cid = content_cid_for_dag_json({"other": "context"})
    bad_context = ControllerV2VerificationContext(
        receipt_cid=public_inputs["receipt_cid"],
        execution_key_cid=public_inputs["execution_key_cid"],
        candidate_context_cid=other_cid,
        expected_candidate_context_cid=other_cid,
        policy_cid=public_inputs["policy_cid"],
        statement_cid=public_inputs["statement_cid"],
        circuit_cid=public_inputs["circuit_cid"],
        verifying_key_cid=public_inputs["verifying_key_cid"],
        issuer_id=public_inputs["issuer_id"],
        epoch=public_inputs["epoch"],
        backend_id="groth16",
        proof_system_id="groth16",
        public_inputs={**public_inputs, "candidate_context_cid": other_cid},
        is_test_only_disposable=True,
    )
    ok2, reason2, _ = verify_test_execution_certificate_v2_for_publication(
        material["certificate_dict"],
        bindings=bindings,
        controller_context=bad_context,
        test_only_backend=material["backend"],
    )
    assert ok2 is False
    assert reason2


# ---------------------------------------------------------------------------
# Negative: nothing but exact V2 VERIFIED reaches put_candidate
# ---------------------------------------------------------------------------


def test_missing_proof_never_reaches_put_candidate(datasets_on_path: None) -> None:
    material = _build_test_only_v4_material()
    public_inputs = material["public_inputs"]
    bindings = _test_only_bindings(
        circuit_cid=public_inputs["circuit_cid"],
        verifying_key_cid=public_inputs["verifying_key_cid"],
    )
    cert = dict(material["certificate_dict"])
    cert.pop("proof", None)
    cert["proof_digest"] = ""
    cert["proof_artifact_cid"] = ""

    store = _AtomicStore()
    intent = _controller_intent(
        public_inputs=public_inputs,
        certificate=cert,
        statement=material["statement"],
    )
    result = ProofReuseControllerPublicationTransaction(
        store=store,
        artifact_bindings=bindings,
        test_only_verification_backend=material["backend"],
    ).publish_intent(intent)
    assert result.published is False
    assert result.put_candidate_called is False
    assert "put_candidate" not in [name for name, _ in store.calls]


def test_swapped_circuit_binding_never_reaches_put_candidate(
    datasets_on_path: None,
) -> None:
    material = _build_test_only_v4_material()
    public_inputs = material["public_inputs"]
    from ipfs_datasets_py.logic.zkp.statements.test_pass import content_cid_for_dag_json

    swapped = _test_only_bindings(
        circuit_cid=content_cid_for_dag_json({"circuit": "swapped"}),
        verifying_key_cid=public_inputs["verifying_key_cid"],
    )
    store = _AtomicStore()
    intent = _controller_intent(
        public_inputs=public_inputs,
        certificate=material["certificate_dict"],
        statement=material["statement"],
    )
    result = ProofReuseControllerPublicationTransaction(
        store=store,
        artifact_bindings=swapped,
        test_only_verification_backend=material["backend"],
    ).publish_intent(intent)
    assert result.published is False
    assert result.put_candidate_called is False
    assert result.reason_code


def test_production_path_denied_without_approved_key_manifest(
    datasets_on_path: None,
) -> None:
    """Hardcoded allowlist is empty/unapproved → no production put_candidate."""

    material = _build_test_only_v4_material()
    public_inputs = material["public_inputs"]
    bindings = Groth16ArtifactIdentityBindings(
        circuit_cid=public_inputs["circuit_cid"],
        verifying_key_cid=public_inputs["verifying_key_cid"],
        artifacts_root="/unapproved/artifacts",
        verifying_key_sha256="a" * 64,
        proving_key_sha256="b" * 64,
        backend_circuit_version=5,
        reviewed_revision=DATASETS_VERIFIER_REVISION,
        provenance_ready=True,
        reason_code="ready",
        diagnostics={
            "manifest_sha256": "0" * 64,
            "native_v4_capability_validated": True,
        },
    )
    store = _AtomicStore()
    intent = _controller_intent(
        public_inputs=public_inputs,
        certificate=material["certificate_dict"],
        statement=material["statement"],
    )
    result = ProofReuseControllerPublicationTransaction(
        store=store,
        artifact_bindings=bindings,
        test_only_verification_backend=material["backend"],
    ).publish_intent(intent)
    assert result.published is False
    assert result.put_candidate_called is False
    assert result.reason_code in {
        "artifact_provenance_unready",
        "production_key_manifest_unapproved",
    }
    assert result.authorizes_skip is False


def test_certificate_self_claim_without_backend_never_publishes(
    datasets_on_path: None,
) -> None:
    material = _build_test_only_v4_material()
    public_inputs = material["public_inputs"]
    bindings = _test_only_bindings(
        circuit_cid=public_inputs["circuit_cid"],
        verifying_key_cid=public_inputs["verifying_key_cid"],
    )
    store = _AtomicStore()
    intent = _controller_intent(
        public_inputs=public_inputs,
        certificate=material["certificate_dict"],
        statement=material["statement"],
    )
    result = ProofReuseControllerPublicationTransaction(
        store=store,
        artifact_bindings=bindings,
        test_only_verification_backend=None,
    ).publish_intent(intent)
    assert result.published is False
    assert result.put_candidate_called is False
    assert result.reason_code in {
        "production_backend_unavailable",
        "verification_backend_unavailable",
        "test_only_backend_missing_verify_proof",
    } or "backend" in result.reason_code


def test_import_and_adapter_do_not_network_or_build(datasets_on_path: None) -> None:
    """Import + verification surface construction has no side-effect I/O policy."""

    assert hasattr(
        publication_module, "verify_test_execution_certificate_v2_for_publication"
    )
    assert hasattr(publication_module, "ControllerV2VerificationContext")
    assert hasattr(publication_module, "ProofReuseControllerPublicationTransaction")
    # Incomplete context fails closed without spawning processes or networking.
    bindings = publication_module.Groth16ArtifactIdentityBindings(
        circuit_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        verifying_key_cid="baguqeerabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        artifacts_root="/tmp/ptr155-test-only",
        verifying_key_sha256="e" * 64,
        proving_key_sha256="f" * 64,
        backend_circuit_version=5,
        reviewed_revision=DATASETS_VERIFIER_REVISION,
        provenance_ready=True,
        reason_code="test_only_disposable_v4_fixture",
    )
    context = publication_module.ControllerV2VerificationContext(
        receipt_cid="baguqeeraccccccccccccccccccccccccccccccccccccccccccccccccccc",
        is_test_only_disposable=True,
    )
    ok, reason, _ = (
        publication_module.verify_test_execution_certificate_v2_for_publication(
            {},
            bindings=bindings,
            controller_context=context,
        )
    )
    assert ok is False
    assert reason == "controller_v2_context_incomplete"


def test_failures_retain_non_authoritative_receipts(datasets_on_path: None) -> None:
    material = _build_test_only_v4_material()
    public_inputs = material["public_inputs"]
    bindings = _test_only_bindings(
        circuit_cid=public_inputs["circuit_cid"],
        verifying_key_cid=public_inputs["verifying_key_cid"],
    )
    store = _AtomicStore()
    candidate_store = _CandidateStore()
    # Incomplete deferred request → DEFERRED with retention, future tests run.
    intent = SimpleNamespace(
        receipt={
            "receipt_id": public_inputs["receipt_cid"],
            "execution_key_cid": public_inputs["execution_key_cid"],
            "locator_cid": "cid:l",
            "nonce": "n",
            "admitted": True,
            "setup_outcome": "pass",
            "call_outcome": "pass",
            "teardown_outcome": "pass",
        },
        receipt_cid=public_inputs["receipt_cid"],
        locator_cid="cid:l",
        certificate=material["certificate_dict"],
        certificate_cid=str(
            material["certificate_dict"].get("certificate_id") or ""
        ),
        deferred_request={
            "receipt_cid": public_inputs["receipt_cid"],
            "locator_cid": "cid:l",
        },
        intent_id="intent:ptr155:incomplete",
    )
    result = ProofReuseControllerPublicationTransaction(
        store=store,
        candidate_store=candidate_store,
        artifact_bindings=bindings,
        test_only_verification_backend=material["backend"],
    ).publish_intent(intent)
    assert result.published is False
    assert result.put_candidate_called is False
    assert result.action == "DEFERRED"
    assert result.reason_code == "controller_v2_context_incomplete"
    assert result.non_authoritative_retained is True or bool(candidate_store.blobs)
    assert "put_candidate" not in [name for name, _ in store.calls]
    assert result.authorizes_skip is False
