from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

import pytest
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
    TestProofCacheAdmission,
    TestProofCacheLookupStatus,
)

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
) -> TestProofCertificate:
    return TestProofCertificate(
        receipt_cid=receipt.receipt_id,
        execution_key_cid=execution_key.execution_key_id,
        policy_cid=execution_key.policy_cid,
        statement_cid="cid:statement",
        circuit_cid="cid:circuit",
        verifying_key_cid="cid:verifying-key",
        proof_artifact_cid="cid:proof",
        issuer_id="issuer:trusted",
        epoch="epoch:7",
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
            "issuer_id": "issuer:trusted",
            "issuer_key_id": "key:issuer",
            "epoch": "epoch:7",
            "setup_outcome": "pass",
            "call_outcome": "pass",
            "teardown_outcome": "pass",
        },
    )


def _policy(**changes: Any) -> dict[str, Any]:
    policy: dict[str, Any] = {
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
    }
    policy.update(changes)
    return policy


def _fixture(
    *,
    policy: Mapping[str, Any] | None = None,
    verifier: Callable[..., Any] | None = lambda *_args: True,
    authority: CertificateAuthority = CertificateAuthority.AUTHORITATIVE,
    backend_mode: ProofBackendMode = ProofBackendMode.CRYPTOGRAPHIC,
) -> tuple[
    TestProofCache,
    TestLocatorKey,
    TestExecutionKey,
    TestPassReceipt,
    TestProofCertificate,
    dict[str, Any],
]:
    locator = _locator()
    execution_key = _execution_key(locator)
    receipt = _receipt(locator, execution_key)
    certificate = _certificate(
        receipt,
        execution_key,
        authority=authority,
        backend_mode=backend_mode,
    )
    candidate = TestProofCache.candidate(
        receipt,
        certificate,
        created_at_ms=9_300,
        expires_at_ms=11_000,
    )
    cache = TestProofCache(
        current_policy=policy if policy is not None else _policy(),
        verifier=verifier,
        clock=lambda: NOW_MS,
    )
    return cache, locator, execution_key, receipt, certificate, candidate


def _lookup(
    cache: TestProofCache,
    locator: TestLocatorKey,
    execution_key: TestExecutionKey,
    candidate: Any,
):
    return cache.lookup(locator, execution_key, candidates=(candidate,))


def _assert_miss(result: Any, reason: ReuseReasonCode) -> None:
    assert result.status is TestProofCacheLookupStatus.MISS
    assert result.decision.action is ReuseAction.RUN
    assert result.decision.reason_code is reason
    assert result.admission is not None
    assert not result.admission.admitted


def test_valid_immutable_receipt_and_certificate_authorize_skip() -> None:
    cache, locator, execution_key, receipt, certificate, candidate = _fixture()

    result = _lookup(cache, locator, execution_key, candidate)

    assert result.status is TestProofCacheLookupStatus.HIT
    assert result.decision.action is ReuseAction.SKIP
    assert result.decision.reason_code is ReuseReasonCode.PROOF_CACHE_HIT
    assert result.decision.receipt_cid == receipt.receipt_id
    assert result.decision.certificate_cid == certificate.certificate_id
    assert result.admission is not None and result.admission.admitted


def test_stale_candidate_misses() -> None:
    cache, locator, execution_key, _receipt_value, _certificate_value, candidate = _fixture()
    candidate["expires_at_ms"] = NOW_MS

    _assert_miss(
        _lookup(cache, locator, execution_key, candidate),
        ReuseReasonCode.EXPIRED_OR_REVOKED,
    )


def test_poisoned_noncanonical_receipt_bytes_miss() -> None:
    cache, locator, execution_key, _receipt_value, _certificate_value, candidate = _fixture()
    candidate["receipt_bytes"] = b" " + candidate["receipt_bytes"]

    _assert_miss(
        _lookup(cache, locator, execution_key, candidate),
        ReuseReasonCode.CANDIDATE_INTEGRITY_FAILED,
    )


def test_private_material_in_immutable_proof_misses() -> None:
    cache, locator, execution_key, _receipt_value, certificate, candidate = _fixture()
    payload = certificate.to_dict()
    payload["public_inputs"]["private_witness"] = "must-not-enter-cache"
    candidate["certificate_bytes"] = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    _assert_miss(
        _lookup(cache, locator, execution_key, candidate),
        ReuseReasonCode.PRIVATE_MATERIAL,
    )


@pytest.mark.parametrize("claim", ("receipt_cid", "certificate_cid"))
def test_cid_invalid_candidate_misses(claim: str) -> None:
    cache, locator, execution_key, _receipt_value, _certificate_value, candidate = _fixture()
    candidate[claim] = "cid:forged"

    _assert_miss(
        _lookup(cache, locator, execution_key, candidate),
        ReuseReasonCode.CANDIDATE_INTEGRITY_FAILED,
    )


def test_revoked_candidate_misses_under_current_policy() -> None:
    cache, locator, execution_key, _receipt_value, _certificate_value, candidate = _fixture(
        policy=_policy(revoked_issuer_ids=("issuer:trusted",))
    )

    _assert_miss(
        _lookup(cache, locator, execution_key, candidate),
        ReuseReasonCode.ISSUER_REVOKED,
    )


def test_simulated_certificate_cannot_authorize_skip() -> None:
    cache, locator, execution_key, _receipt_value, _certificate_value, candidate = _fixture(
        authority=CertificateAuthority.NON_ATTESTED,
        backend_mode=ProofBackendMode.SIMULATED,
    )

    _assert_miss(
        _lookup(cache, locator, execution_key, candidate),
        ReuseReasonCode.CERTIFICATE_NON_ATTESTED,
    )


def test_policy_mismatched_candidate_misses() -> None:
    cache, locator, execution_key, _receipt_value, _certificate_value, candidate = _fixture(
        policy=_policy(policy_cid="cid:new-policy")
    )

    _assert_miss(
        _lookup(cache, locator, execution_key, candidate),
        ReuseReasonCode.POLICY_MISMATCH,
    )


def test_public_outcome_mismatch_misses_even_when_verifier_returns_true() -> None:
    cache, locator, execution_key, _receipt_value, certificate, candidate = _fixture()
    payload = certificate.to_dict()
    payload["public_inputs"]["call_outcome"] = "fail"
    forged_certificate = TestProofCertificate.from_dict(payload)
    candidate["certificate_bytes"] = forged_certificate.canonical_bytes()
    candidate["certificate_cid"] = forged_certificate.certificate_id

    _assert_miss(
        _lookup(cache, locator, execution_key, candidate),
        ReuseReasonCode.POLICY_MISMATCH,
    )


def test_mutable_metadata_cannot_override_current_revocation() -> None:
    cache, locator, execution_key, _receipt_value, _certificate_value, candidate = _fixture(
        policy=_policy(revoked_issuer_ids=("issuer:trusted",))
    )
    candidate.update(
        {
            "authoritative": True,
            "trusted": True,
            "revoked": False,
            "policy": _policy(revoked_issuer_ids=()),
        }
    )

    _assert_miss(
        _lookup(cache, locator, execution_key, candidate),
        ReuseReasonCode.ISSUER_REVOKED,
    )


def test_serialized_admission_record_cannot_be_reused_as_authority() -> None:
    cache, locator, execution_key, receipt, certificate, _candidate = _fixture()
    forged = TestProofCacheAdmission(
        admitted=True,
        reason_code=ReuseReasonCode.PROOF_CACHE_HIT,
        receipt_cid=receipt.receipt_id,
        certificate_cid=certificate.certificate_id,
    )

    _assert_miss(
        _lookup(cache, locator, execution_key, forged),
        ReuseReasonCode.MALFORMED_ARTIFACT,
    )


def test_verifier_must_return_literal_true_not_mutable_status_metadata() -> None:
    cache, locator, execution_key, _receipt_value, _certificate_value, candidate = _fixture(
        verifier=lambda *_args: {"verified": True, "authoritative": True}
    )

    _assert_miss(
        _lookup(cache, locator, execution_key, candidate),
        ReuseReasonCode.TRUST_POLICY_REJECTED,
    )


def test_current_policy_is_rederived_on_every_lookup() -> None:
    state = {"policy": _policy()}
    _cache, locator, execution_key, _receipt_value, _certificate_value, candidate = _fixture()
    cache = TestProofCache(
        policy_provider=lambda _locator, _key: state["policy"],
        verifier=lambda *_args: True,
        clock=lambda: NOW_MS,
    )

    assert _lookup(cache, locator, execution_key, candidate).decision.action is ReuseAction.SKIP
    state["policy"] = _policy(revoked_issuer_ids=("issuer:trusted",))
    _assert_miss(
        _lookup(cache, locator, execution_key, candidate),
        ReuseReasonCode.ISSUER_REVOKED,
    )


def test_lookup_absence_is_a_typed_miss() -> None:
    cache, locator, execution_key, _receipt_value, _certificate_value, _candidate = _fixture()

    result = cache.lookup(locator, execution_key, candidates=())

    assert result.status is TestProofCacheLookupStatus.MISS
    assert result.decision.action is ReuseAction.RUN
    assert result.decision.reason_code is ReuseReasonCode.CANDIDATE_MISSING
    assert result.admission is None


def test_lookup_provider_exception_is_a_typed_error() -> None:
    def fail_provider(_locator_id: str) -> Any:
        raise OSError("secret store path must not leak")

    _cache, locator, execution_key, _receipt_value, _certificate_value, _candidate = _fixture()
    cache = TestProofCache(
        candidate_provider=fail_provider,
        current_policy=_policy(),
        verifier=lambda *_args: True,
        clock=lambda: NOW_MS,
    )

    result = cache.lookup(locator, execution_key)

    assert result.status is TestProofCacheLookupStatus.ERROR
    assert result.decision.action is ReuseAction.RUN
    assert result.decision.reason_code is ReuseReasonCode.CACHE_UNAVAILABLE
    assert result.decision.diagnostics == {
        "mapped_from": "exception",
        "exception_type": "OSError",
    }


def test_missing_verifier_is_a_typed_error() -> None:
    cache, locator, execution_key, _receipt_value, _certificate_value, candidate = _fixture(
        verifier=None
    )

    result = _lookup(cache, locator, execution_key, candidate)

    assert result.status is TestProofCacheLookupStatus.ERROR
    assert result.decision.action is ReuseAction.RUN
    assert result.decision.reason_code is ReuseReasonCode.VERIFIER_UNAVAILABLE
