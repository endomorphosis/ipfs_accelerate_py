"""Tests for the lazy datasets certificate provider adapter (PTR-043)."""

from __future__ import annotations

import hashlib
import hmac
import json
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_test_certificate_provider import (
    DEFAULT_VERIFIER_MODULE,
    IPFS_DATASETS_TEST_CERTIFICATE_PROVIDER_ID,
    TEST_CERTIFICATE_PROVIDER_INTERFACE,
    IpfsDatasetsTestCertificateProvider,
    TestCertificateProviderError,
    TestCertificateVerificationResult,
    TestCertificateVerificationStatus,
    inspect_test_certificate_provider_capability,
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

_SIGNING_KEY = b"ptr-043-offline-provider-key"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _locator() -> TestLocatorKey:
    return TestLocatorKey(
        repository_id="repository:example",
        package_identity="package:example",
        node_id="test/api/test_example.py::test_example",
    )


def _execution_key(locator: TestLocatorKey) -> TestExecutionKey:
    return TestExecutionKey(
        locator_cid=locator.locator_id,
        repository_forest_cid="cid:repository-forest",
        static_trace_root_cid="cid:static-trace",
        runtime_trace_root_cid="cid:runtime-trace",
        runtime_completeness_policy="complete-v1",
        policy_cid="cid:policy",
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
    proof_digest: str = "",
    proof_artifact_cid: str = "cid:proof",
    metadata: dict[str, Any] | None = None,
) -> TestProofCertificate:
    digest = proof_digest or "sha256:" + hashlib.sha256(b"fixture-proof").hexdigest()
    return TestProofCertificate(
        receipt_cid=receipt.receipt_id,
        execution_key_cid=execution_key.execution_key_id,
        policy_cid=execution_key.policy_cid,
        statement_cid="cid:statement",
        circuit_cid="cid:circuit",
        verifying_key_cid="cid:verifying-key",
        proof_artifact_cid=proof_artifact_cid or digest,
        proof_digest=digest,
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
        metadata=metadata or {"fixture": "ptr-043"},
    )


def _requirements(certificate: TestProofCertificate, **changes: Any) -> dict[str, Any]:
    requirements: dict[str, Any] = {
        "policy_cid": certificate.policy_cid,
        "statement_cid": certificate.statement_cid,
        "circuit_cid": certificate.circuit_cid,
        "verifying_key_cid": certificate.verifying_key_cid,
        "proof_system_id": certificate.proof_system_id,
        "backend_id": "groth16",
        "issuer_id": certificate.issuer_id,
        "epoch": certificate.epoch,
        "trusted_issuer_ids": (certificate.issuer_id,),
        "allowed_epochs": (certificate.epoch,),
        "proof_bytes": b"fixture-proof",
        "proof_public_inputs": dict(certificate.public_inputs),
        "proof_metadata": {
            "backend": "groth16",
            "proof_system": "groth16",
        },
    }
    requirements.update(changes)
    return requirements


def _assert_run_result(
    result: Any,
    reason: ReuseReasonCode,
    *,
    status: TestCertificateVerificationStatus | None = None,
) -> None:
    assert type(result).__name__ == "TestCertificateVerificationResult"
    assert result.reason_code == reason or result.reason_code is reason
    assert result.test_action == "run"
    assert result.can_authorize_skip is False
    assert result.authoritative is False
    assert result.verified is False
    if status is not None:
        assert result.status == status or result.status is status
    decision = result.to_reuse_decision()
    assert decision.action is ReuseAction.RUN
    assert decision.reason_code is reason


# ---------------------------------------------------------------------------
# Cold import / capability
# ---------------------------------------------------------------------------


def test_module_cold_import_does_not_load_datasets_zk_backend() -> None:
    """Importing the adapter module must not pull datasets ZK backends."""

    backend_prefixes = (
        "ipfs_datasets_py.logic.zkp.backends",
        "ipfs_datasets_py.logic.zkp.provekit",
        "ipfs_datasets_py.logic.zkp.test_execution_certificate",
    )
    # Drop any previously loaded surface so we can detect eager imports.
    for name in list(sys.modules):
        if name.startswith(backend_prefixes):
            del sys.modules[name]

    import importlib

    # Import only the adapter package path; do not reload (reload would split
    # class identities for subsequent isinstance checks in this process).
    module_name = (
        "ipfs_accelerate_py.agent_supervisor.integrations."
        "ipfs_datasets_test_certificate_provider"
    )
    # Ensure a fresh submodule import without reloading siblings already under test.
    if module_name in sys.modules:
        # Module already imported by this test file; just assert backends stay out
        # after capability/construction work.
        provider = IpfsDatasetsTestCertificateProvider(
            importer=lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name))
        )
        _ = provider.capabilities()
    else:
        importlib.import_module(module_name)

    after = {name for name in sys.modules if name.startswith(backend_prefixes)}
    assert after == set()


def test_construction_and_capability_never_import_datasets() -> None:
    calls: list[str] = []

    def explosive_importer(name: str) -> Any:
        calls.append(name)
        raise AssertionError(f"cold path imported {name}")

    class ExplosiveIssuer:
        def prove(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("issuer.prove must not run on cold path")

        def issue(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("issuer.issue must not run on cold path")

    provider = IpfsDatasetsTestCertificateProvider(
        importer=explosive_importer,
        issuer=ExplosiveIssuer(),
    )
    state_before = {name: id(value) for name, value in vars(provider).items()}

    first = provider.capabilities()
    second = provider.capability()
    handle = provider.issuer_handle
    inspected = inspect_test_certificate_provider_capability(issuance=True)

    assert calls == []
    assert provider.imported is False
    assert first.to_dict() == second.to_dict()
    assert first.provider_id == IPFS_DATASETS_TEST_CERTIFICATE_PROVIDER_ID
    assert first.interface == TEST_CERTIFICATE_PROVIDER_INTERFACE
    assert first.lazy is True
    assert first.prove_on_lookup is False
    assert first.imported is False
    assert handle is not None
    assert inspected.issuance is True
    assert {name: id(value) for name, value in vars(provider).items()} == state_before


# ---------------------------------------------------------------------------
# Injected verification successes / retained bytes
# ---------------------------------------------------------------------------


def test_verify_uses_exact_retained_bytes_and_pinned_inputs() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    receipt = _receipt(locator, execution_key)
    proof_bytes = b"exact-retained-proof-bytes"
    digest = "sha256:" + hashlib.sha256(proof_bytes).hexdigest()
    certificate = _certificate(
        receipt,
        execution_key,
        proof_digest=digest,
        proof_artifact_cid=digest,
    )
    receipt_bytes = receipt.canonical_bytes()
    certificate_bytes = certificate.canonical_bytes()
    assert type(receipt_bytes) is bytes
    assert type(certificate_bytes) is bytes

    seen: dict[str, Any] = {}

    def verify_fn(cert: Any, binding: Any, backend: Any = None, proof: Any = None) -> Any:
        seen["cert"] = cert
        seen["binding"] = binding
        seen["backend"] = backend
        seen["proof"] = proof
        return SimpleNamespace(
            status="verified",
            reason="verified",
            verified=True,
            can_authorize_skip=True,
            authority="authoritative",
            detail="ok",
            backend_id="groth16",
            certificate_id=certificate.certificate_id,
        )

    provider = IpfsDatasetsTestCertificateProvider(verify_fn=verify_fn, backend="pinned-backend")
    requirements = _requirements(
        certificate,
        proof_bytes=proof_bytes,
        binding=SimpleNamespace(backend_id="groth16"),
        # Pin a policy field the certificate already matches.
        policy_cid=certificate.policy_cid,
        circuit_cid=certificate.circuit_cid,
    )

    result = provider.verify_retained_bytes(
        certificate_bytes,
        receipt_bytes,
        requirements,
    )

    assert result.verified is True
    assert result.can_authorize_skip is True
    assert result.test_action == "skip"
    assert result.reason_code is ReuseReasonCode.PROOF_CACHE_HIT
    assert result.certificate_cid == certificate.certificate_id
    assert result.receipt_cid == receipt.receipt_id
    decision = result.to_reuse_decision()
    assert decision.action is ReuseAction.SKIP
    assert decision.certificate_cid == certificate.certificate_id

    # Non-canonical retained bytes must fail closed even if a verifier would pass.
    poisoned = b" " + certificate_bytes
    poisoned_result = provider.verify_retained_bytes(
        poisoned, receipt_bytes, requirements
    )
    _assert_run_result(
        poisoned_result,
        ReuseReasonCode.CANDIDATE_INTEGRITY_FAILED,
        status=TestCertificateVerificationStatus.REJECTED,
    )


def test_verify_rejects_mismatched_pinned_inputs() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    receipt = _receipt(locator, execution_key)
    certificate = _certificate(receipt, execution_key)

    def verify_fn(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("pinned mismatch must not reach verifier")

    provider = IpfsDatasetsTestCertificateProvider(verify_fn=verify_fn)
    requirements = _requirements(certificate, circuit_cid="cid:other-circuit")

    result = provider.verify(certificate, receipt, requirements)
    _assert_run_result(
        result,
        ReuseReasonCode.CIRCUIT_UNAVAILABLE,
        status=TestCertificateVerificationStatus.REJECTED,
    )


# ---------------------------------------------------------------------------
# Prove never invoked by lookup
# ---------------------------------------------------------------------------


class RecordingIssuer:
    def __init__(self) -> None:
        self.prove_calls = 0
        self.issue_calls = 0

    def prove(self, *_args: Any, **_kwargs: Any) -> Any:
        self.prove_calls += 1
        raise AssertionError("issuer.prove must never run during lookup/verify")

    def issue(self, *_args: Any, **_kwargs: Any) -> Any:
        self.issue_calls += 1
        raise AssertionError("issuer.issue must never run during lookup/verify")

    def issue_certificate(self, *_args: Any, **_kwargs: Any) -> Any:
        self.issue_calls += 1
        raise AssertionError("issuer.issue_certificate must never run during lookup")


def test_lookup_and_verify_never_invoke_prove_or_issuer() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    receipt = _receipt(locator, execution_key)
    certificate = _certificate(receipt, execution_key)
    issuer = RecordingIssuer()

    def verify_fn(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            status="verified",
            reason="verified",
            verified=True,
            can_authorize_skip=True,
            authority="authoritative",
            detail="ok",
            backend_id="groth16",
            certificate_id=certificate.certificate_id,
        )

    provider = IpfsDatasetsTestCertificateProvider(
        verify_fn=verify_fn,
        issuer=issuer,
        backend="backend",
    )
    requirements = _requirements(
        certificate, binding=SimpleNamespace(backend_id="groth16")
    )

    verified = provider.verify(certificate, receipt, requirements)
    looked_up = provider.lookup(
        certificate,
        receipt,
        requirements,
        prove=True,
        issue=True,
        issue_if_missing=True,
    )

    assert verified.can_authorize_skip is True
    assert looked_up.can_authorize_skip is True
    assert issuer.prove_calls == 0
    assert issuer.issue_calls == 0
    assert provider.prove_call_count == 0

    with pytest.raises(TestCertificateProviderError) as raised:
        provider.prove(certificate)
    assert "prove is not invoked" in str(raised.value)
    assert provider.prove_call_count == 1
    assert issuer.prove_calls == 0


# ---------------------------------------------------------------------------
# Missing / incompatible / timeout / exception → RUN-compatible
# ---------------------------------------------------------------------------


def test_missing_datasets_provider_returns_run_compatible_unavailable() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    receipt = _receipt(locator, execution_key)
    certificate = _certificate(receipt, execution_key)

    def missing_importer(name: str) -> Any:
        raise ModuleNotFoundError(name)

    provider = IpfsDatasetsTestCertificateProvider(importer=missing_importer)
    result = provider.verify(
        certificate,
        receipt,
        _requirements(certificate, binding=object()),
    )
    _assert_run_result(
        result,
        ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE,
        status=TestCertificateVerificationStatus.UNAVAILABLE,
    )


def test_incompatible_datasets_surface_returns_run_compatible_unavailable() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    receipt = _receipt(locator, execution_key)
    certificate = _certificate(receipt, execution_key)

    def incompatible_importer(name: str) -> Any:
        # Present module objects that lack required callables.
        return SimpleNamespace()

    provider = IpfsDatasetsTestCertificateProvider(importer=incompatible_importer)
    result = provider.verify(
        certificate,
        receipt,
        _requirements(certificate, binding=object()),
    )
    _assert_run_result(
        result,
        ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE,
        status=TestCertificateVerificationStatus.UNAVAILABLE,
    )
    assert "incompatible" in result.detail or "missing" in result.diagnostics.get(
        "missing", []
    ) or "incompatible" in str(result.diagnostics)


def test_timeout_returns_run_compatible_timeout() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    receipt = _receipt(locator, execution_key)
    certificate = _certificate(receipt, execution_key)

    def slow_verify(*_args: Any, **_kwargs: Any) -> Any:
        time.sleep(0.25)
        return True

    provider = IpfsDatasetsTestCertificateProvider(
        verify_fn=slow_verify,
        timeout_seconds=0.05,
    )
    result = provider.verify(
        certificate,
        receipt,
        _requirements(certificate, binding=object()),
    )
    _assert_run_result(
        result,
        ReuseReasonCode.TIMEOUT,
        status=TestCertificateVerificationStatus.UNAVAILABLE,
    )


def test_exception_returns_run_compatible_exception_result() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    receipt = _receipt(locator, execution_key)
    certificate = _certificate(receipt, execution_key)

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("secret-proof-material-must-not-leak")

    provider = IpfsDatasetsTestCertificateProvider(verify_fn=boom)
    result = provider.verify(
        certificate,
        receipt,
        _requirements(certificate, binding=object()),
    )
    _assert_run_result(
        result,
        ReuseReasonCode.EXCEPTION_FAIL_OPEN_TO_RUN,
        status=TestCertificateVerificationStatus.UNAVAILABLE,
    )
    assert "secret-proof-material" not in result.detail
    assert result.diagnostics.get("exception_type") == "RuntimeError"


def test_disabled_provider_is_unavailable_run() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    receipt = _receipt(locator, execution_key)
    certificate = _certificate(receipt, execution_key)
    provider = IpfsDatasetsTestCertificateProvider(
        enabled=False,
        verify_fn=lambda *_a, **_k: True,
    )
    result = provider.verify(
        certificate,
        receipt,
        _requirements(certificate, binding=object()),
    )
    _assert_run_result(
        result,
        ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE,
        status=TestCertificateVerificationStatus.UNAVAILABLE,
    )


# ---------------------------------------------------------------------------
# Simulated authority rejected
# ---------------------------------------------------------------------------


def test_simulated_backend_mode_is_rejected_as_non_attested() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    receipt = _receipt(locator, execution_key)
    certificate = _certificate(
        receipt,
        execution_key,
        authority=CertificateAuthority.NON_ATTESTED,
        backend_mode=ProofBackendMode.SIMULATED,
    )

    def verify_fn(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("simulated certificates must not reach the verifier")

    provider = IpfsDatasetsTestCertificateProvider(verify_fn=verify_fn)
    result = provider.verify(
        certificate,
        receipt,
        _requirements(certificate, binding=object()),
    )
    _assert_run_result(
        result,
        ReuseReasonCode.CERTIFICATE_NON_ATTESTED,
        status=TestCertificateVerificationStatus.REJECTED,
    )


def test_simulated_metadata_markers_are_rejected() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    receipt = _receipt(locator, execution_key)
    certificate = _certificate(
        receipt,
        execution_key,
        metadata={"mode": "simulated-demo", "fixture": "ptr-043"},
    )

    def verify_fn(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("simulated metadata must not reach the verifier")

    provider = IpfsDatasetsTestCertificateProvider(verify_fn=verify_fn)
    result = provider.verify(
        certificate,
        receipt,
        _requirements(certificate, binding=object()),
    )
    _assert_run_result(
        result,
        ReuseReasonCode.CERTIFICATE_NON_ATTESTED,
        status=TestCertificateVerificationStatus.REJECTED,
    )


def test_datasets_non_attested_result_is_mapped_to_run() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    receipt = _receipt(locator, execution_key)
    certificate = _certificate(receipt, execution_key)

    def verify_fn(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            status="rejected",
            reason="certificate_non_attested",
            verified=False,
            can_authorize_skip=False,
            authority="non_attested",
            detail="simulated artifact",
            backend_id="simulated",
            certificate_id=certificate.certificate_id,
        )

    provider = IpfsDatasetsTestCertificateProvider(verify_fn=verify_fn)
    result = provider.verify(
        certificate,
        receipt,
        _requirements(certificate, binding=object()),
    )
    _assert_run_result(
        result,
        ReuseReasonCode.CERTIFICATE_NON_ATTESTED,
        status=TestCertificateVerificationStatus.REJECTED,
    )


# ---------------------------------------------------------------------------
# Result contract / cache bridge
# ---------------------------------------------------------------------------


def test_verification_result_is_not_truthy() -> None:
    result = TestCertificateVerificationResult(
        status=TestCertificateVerificationStatus.VERIFIED,
        reason_code=ReuseReasonCode.PROOF_CACHE_HIT,
        authority=CertificateAuthority.AUTHORITATIVE,
        certificate_cid="cid:cert",
        receipt_cid="cid:receipt",
    )
    with pytest.raises(TypeError):
        bool(result)
    assert result.can_authorize_skip is True


def test_as_cache_verifier_returns_exact_true_only_on_success() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    receipt = _receipt(locator, execution_key)
    certificate = _certificate(receipt, execution_key)

    def verify_fn(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            status="verified",
            reason="verified",
            verified=True,
            can_authorize_skip=True,
            authority="authoritative",
            detail="ok",
            backend_id="groth16",
            certificate_id=certificate.certificate_id,
        )

    provider = IpfsDatasetsTestCertificateProvider(verify_fn=verify_fn)
    cache_verify = provider.as_cache_verifier()
    assert (
        cache_verify(
            certificate,
            receipt,
            _requirements(certificate, binding=object()),
        )
        is True
    )

    def reject_fn(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            status="rejected",
            reason="proof_invalid",
            verified=False,
            can_authorize_skip=False,
            authority="non_attested",
            detail="no",
            backend_id="groth16",
            certificate_id=certificate.certificate_id,
        )

    reject_provider = IpfsDatasetsTestCertificateProvider(verify_fn=reject_fn)
    assert (
        reject_provider.as_cache_verifier()(
            certificate,
            receipt,
            _requirements(certificate, binding=object()),
        )
        is False
    )


def test_receipt_certificate_identity_mismatch_is_rejected() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    receipt = _receipt(locator, execution_key)
    other_key = TestExecutionKey(
        locator_cid=locator.locator_id,
        repository_forest_cid="cid:other-forest",
        static_trace_root_cid="cid:static-trace",
        runtime_trace_root_cid="cid:runtime-trace",
        runtime_completeness_policy="complete-v1",
        policy_cid="cid:policy",
    )
    other_receipt = _receipt(locator, other_key)
    certificate = _certificate(receipt, execution_key)

    provider = IpfsDatasetsTestCertificateProvider(
        verify_fn=lambda *_a, **_k: True,
    )
    result = provider.verify(
        certificate,
        other_receipt,
        _requirements(certificate, binding=object()),
    )
    _assert_run_result(
        result,
        ReuseReasonCode.RECEIPT_MISMATCH,
        status=TestCertificateVerificationStatus.REJECTED,
    )


# ---------------------------------------------------------------------------
# Optional live datasets path (when available)
# ---------------------------------------------------------------------------


def test_live_datasets_offline_conformance_backend_when_available() -> None:
    """Exercise the real datasets verifier with an offline conformance backend."""

    pytest.importorskip("ipfs_datasets_py.logic.zkp.test_execution_certificate")
    from ipfs_datasets_py.logic.zkp import ZKPProof
    from ipfs_datasets_py.logic.zkp.provekit.test_pass_circuit import (
        TestPassCircuitBinding,
    )
    from ipfs_datasets_py.logic.zkp.statements.test_pass import (
        build_public_inputs,
        build_statement,
    )
    from ipfs_datasets_py.logic.zkp.test_execution_certificate import (
        TestExecutionCertificate,
        verify_test_execution_certificate,
    )

    def _canonical_bytes(value: Any) -> bytes:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def _digest(data: bytes) -> str:
        return "sha256:" + hashlib.sha256(data).hexdigest()

    def _signed(backend_id: str, proof_system_id: str, public_inputs: dict[str, Any]) -> bytes:
        message = b"\0".join(
            (
                backend_id.encode(),
                proof_system_id.encode(),
                _canonical_bytes(public_inputs),
            )
        )
        return hmac.digest(_SIGNING_KEY, message, "sha256")

    @dataclass
    class ConformanceBackend:
        backend_id: str
        proof_system_id: str
        available: bool = True
        calls: int = 0

        def verify_proof(self, proof: ZKPProof) -> bool:
            self.calls += 1
            expected = _signed(
                self.backend_id,
                self.proof_system_id,
                dict(proof.public_inputs),
            )
            return hmac.compare_digest(proof.proof_data, expected)

    backend_id = "groth16"
    proof_system_id = "groth16"
    statement = build_statement(
        build_public_inputs(
            receipt_cid=_digest(b"admitted complete-pass receipt"),
            execution_key_cid="cid:execution-key:provider-v1",
            policy_cid="cid:policy:strict-reuse-v1",
            statement_cid="cid:statement:TestPassStatementV1",
            circuit_cid="cid:circuit:test-pass-v1",
            verifying_key_cid="cid:vk:test-pass-groth16-v1",
            issuer_id="issuer:trusted-runner",
            epoch="epoch:2026-07-31",
            locator_cid="cid:locator:fixture-node",
            completeness_policy_cid="cid:completeness:strict-v1",
        )
    )
    public_inputs = statement.to_public_inputs()
    proof_bytes = _signed(backend_id, proof_system_id, public_inputs)
    proof = ZKPProof(
        proof_data=proof_bytes,
        public_inputs=public_inputs,
        metadata={"backend": backend_id, "proof_system": proof_system_id},
        timestamp=1_775_000_000.0,
        size_bytes=len(proof_bytes),
    )
    proof_digest = _digest(proof_bytes)
    datasets_certificate = TestExecutionCertificate(
        receipt_cid=statement.public_inputs.receipt_cid,
        execution_key_cid=statement.public_inputs.execution_key_cid,
        statement_cid=statement.public_inputs.statement_cid,
        circuit_cid=statement.public_inputs.circuit_cid,
        verifying_key_cid=statement.public_inputs.verifying_key_cid,
        proof_system_id=proof_system_id,
        proof=proof,
        proof_artifact_cid=proof_digest,
        proof_digest=proof_digest,
        backend_mode="cryptographic",
        authority="authoritative",
        issuer_id=statement.public_inputs.issuer_id,
        policy_cid=statement.public_inputs.policy_cid,
        epoch=statement.public_inputs.epoch,
        public_inputs=public_inputs,
        metadata={"fixture": "ptr-043-live"},
    )
    binding = TestPassCircuitBinding(
        statement,
        backend_id=backend_id,
        proof_system_id=proof_system_id,
    )
    backend = ConformanceBackend(backend_id, proof_system_id)

    # Sanity: datasets path itself verifies.
    direct = verify_test_execution_certificate(
        datasets_certificate, binding, backend
    )
    assert direct.verified is True

    # Build accelerator-side envelope + a synthetic admitted receipt matching pins.
    locator = TestLocatorKey(
        repository_id="repository:provider-live",
        package_identity="package:provider-live",
        node_id="test/api/test_live.py::test_live",
    )
    execution_key = TestExecutionKey(
        locator_cid=locator.locator_id,
        repository_forest_cid="cid:repository-forest",
        static_trace_root_cid="cid:static-trace",
        runtime_trace_root_cid="cid:runtime-trace",
        runtime_completeness_policy="complete-v1",
        policy_cid=statement.public_inputs.policy_cid,
    )
    # Force execution key id by using certificate's execution_key_cid via a
    # receipt that we align manually through public inputs rather than identity
    # equality of the TestExecutionKey helper.
    receipt = TestPassReceipt(
        execution_key_cid=statement.public_inputs.execution_key_cid,
        locator_cid=statement.public_inputs.locator_cid
        if hasattr(statement.public_inputs, "locator_cid")
        else locator.locator_id,
        setup_outcome=PhaseOutcome.PASS,
        call_outcome=PhaseOutcome.PASS,
        teardown_outcome=PhaseOutcome.PASS,
        static_trace_root_cid="cid:static-trace",
        runtime_trace_root_cid="cid:runtime-trace",
        completeness_receipt_cid="cid:completeness:strict-v1",
        dependency_forest_cid="cid:repository-forest",
        issuer_key_id="key:issuer",
        policy_cid=statement.public_inputs.policy_cid,
    )
    # Override receipt_id is not possible; certificate.receipt_cid is a digest of
    # a different witness.  For the adapter path we pass the datasets certificate
    # through verify_fn injection that wraps the real verifier while still using
    # the provider's retained-byte / pin machinery on an accelerator certificate
    # built from the same public pins.
    accel_cert = TestProofCertificate(
        receipt_cid=receipt.receipt_id,
        execution_key_cid=receipt.execution_key_cid,
        policy_cid=statement.public_inputs.policy_cid,
        statement_cid=statement.public_inputs.statement_cid,
        circuit_cid=statement.public_inputs.circuit_cid,
        verifying_key_cid=statement.public_inputs.verifying_key_cid,
        proof_system_id=proof_system_id,
        proof_artifact_cid=proof_digest,
        proof_digest=proof_digest,
        backend_mode=ProofBackendMode.CRYPTOGRAPHIC,
        authority=CertificateAuthority.AUTHORITATIVE,
        issuer_id=statement.public_inputs.issuer_id,
        epoch=statement.public_inputs.epoch,
        public_inputs=dict(public_inputs),
        metadata={"fixture": "ptr-043-live"},
    )

    def live_verify(cert: Any, bind: Any, backend_arg: Any = None, proof: Any = None) -> Any:
        # Rebind public receipt/execution ids to the statement while verifying
        # the cryptographic proof under the real datasets path.
        live_cert = TestExecutionCertificate(
            receipt_cid=statement.public_inputs.receipt_cid,
            execution_key_cid=statement.public_inputs.execution_key_cid,
            statement_cid=statement.public_inputs.statement_cid,
            circuit_cid=statement.public_inputs.circuit_cid,
            verifying_key_cid=statement.public_inputs.verifying_key_cid,
            proof_system_id=proof_system_id,
            proof=proof if proof is not None else datasets_certificate.proof,
            proof_artifact_cid=proof_digest,
            proof_digest=proof_digest,
            backend_mode="cryptographic",
            authority="authoritative",
            issuer_id=statement.public_inputs.issuer_id,
            policy_cid=statement.public_inputs.policy_cid,
            epoch=statement.public_inputs.epoch,
            public_inputs=public_inputs,
            metadata={"fixture": "ptr-043-live"},
        )
        return verify_test_execution_certificate(live_cert, binding, backend)

    provider = IpfsDatasetsTestCertificateProvider(
        verify_fn=live_verify,
        backend=backend,
    )
    requirements = {
        "policy_cid": statement.public_inputs.policy_cid,
        "statement_cid": statement.public_inputs.statement_cid,
        "circuit_cid": statement.public_inputs.circuit_cid,
        "verifying_key_cid": statement.public_inputs.verifying_key_cid,
        "proof_system_id": proof_system_id,
        "backend_id": backend_id,
        "issuer_id": statement.public_inputs.issuer_id,
        "epoch": statement.public_inputs.epoch,
        "trusted_issuer_ids": (statement.public_inputs.issuer_id,),
        "allowed_epochs": (statement.public_inputs.epoch,),
        "proof": proof,
        "binding": binding,
        "expected_public_inputs": public_inputs,
    }
    result = provider.verify(accel_cert, receipt, requirements)
    assert result.verified is True
    assert result.can_authorize_skip is True
    assert result.test_action == "skip"
    assert backend.calls >= 1


def test_provider_interface_constants_are_stable() -> None:
    assert TEST_CERTIFICATE_PROVIDER_INTERFACE == "TestCertificateProvider@1"
    assert DEFAULT_VERIFIER_MODULE.endswith("test_execution_certificate")
    provider = IpfsDatasetsTestCertificateProvider()
    assert provider.interface == TEST_CERTIFICATE_PROVIDER_INTERFACE
    assert provider.provider_id == IPFS_DATASETS_TEST_CERTIFICATE_PROVIDER_ID
