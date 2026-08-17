"""PTR-160 vectors for locally pinned runner pass attestations."""

from __future__ import annotations

import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from multiformats import CID, multihash

from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import TestPassReceipt
from ipfs_accelerate_py.testing.proof_reuse.runner_pass_attestation import (
    ATTESTATION_DOMAIN,
    AttestationNonceRegistry,
    RunnerAttestationError,
    RunnerKeyRecord,
    RunnerPassAttestation,
    RunnerPublicKey,
    RunnerTrustPolicy,
    attest_test_pass_receipt,
    canonical_dag_cbor,
    dag_cbor_cid,
    decode_canonical_dag_cbor,
    verify_runner_pass_attestation,
    verify_runner_pass_attestation_with_key,
)


NOW = 1_800_000_000


def _material() -> tuple[Ed25519PrivateKey, RunnerPublicKey, RunnerTrustPolicy]:
    private = Ed25519PrivateKey.generate()
    public = RunnerPublicKey.from_public_key(private.public_key())
    policy = RunnerTrustPolicy(
        trust_domain="pytest.local",
        active_key_epoch="epoch-7",
        keys=(
            RunnerKeyRecord(
                public_key_cid=public.cid,
                public_key_material=public.material,
                key_epoch="epoch-7",
                not_before=NOW - 60,
                not_after=NOW + 60,
            ),
        ),
        policy_epoch="policy-3",
    )
    return private, public, policy


def _candidate() -> str:
    return dag_cbor_cid({"candidate": "context-v1"})


def _receipt(policy: RunnerTrustPolicy) -> TestPassReceipt:
    return TestPassReceipt(
        execution_key_cid=dag_cbor_cid({"execution": "exact-v1"}),
        locator_cid=dag_cbor_cid({"locator": "exact-v1"}),
        static_trace_root_cid=dag_cbor_cid({"trace": "static"}),
        runtime_trace_root_cid=dag_cbor_cid({"trace": "runtime"}),
        completeness_receipt_cid=dag_cbor_cid({"trace": "complete"}),
        runner_identity="runner:pytest",
        trust_domain=policy.trust_domain,
        issuer_key_id="runner-key",
        nonce="receipt-nonce",
        epoch_policy_id="epoch-policy-7",
        policy_cid=policy.cid,
    )


def _attested() -> tuple[TestPassReceipt, RunnerTrustPolicy, RunnerPassAttestation, AttestationNonceRegistry, RunnerPublicKey]:
    private, public, policy = _material()
    receipt = _receipt(policy)
    registry = AttestationNonceRegistry()
    attestation = attest_test_pass_receipt(
        receipt,
        private_key=private,
        policy=policy,
        candidate_context_cid=_candidate(),
        issuance_nonce="nonce-for-one-immutable-issuance",
        issued_at=NOW,
        nonce_registry=registry,
    )
    return receipt, policy, attestation, registry, public


def _verify(receipt: TestPassReceipt, policy: RunnerTrustPolicy, attestation: RunnerPassAttestation, registry: AttestationNonceRegistry | None = None):
    return verify_runner_pass_attestation(
        attestation,
        receipt=receipt,
        policy=policy,
        pinned_policy_cid=policy.cid,
        current_execution_key_cid=receipt.execution_key_cid,
        current_candidate_context_cid=attestation.candidate_context_cid,
        now=NOW,
        nonce_registry=registry,
    )


def test_sole_v1_ed25519_suite_and_strict_dag_cbor_cids() -> None:
    receipt, policy, attestation, registry, public = _attested()

    public_key_cid = CID("base32", 1, "raw", multihash.digest(public.material, "sha2-256")).encode()
    assert public.cid == public_key_cid
    assert policy.cid.startswith("bafy")
    assert attestation.unsigned_cid.startswith("bafy")
    assert attestation.cid.startswith("bafy")
    assert attestation.unsigned_bytes() == canonical_dag_cbor(attestation.unsigned_dict())
    # A direct Ed25519 verification of exactly the documented domain-separated
    # digest confirms the wire suite has no alternate v1 preimage.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    Ed25519PublicKey.from_public_bytes(public.raw_key).verify(
        attestation.signature,
        ATTESTATION_DOMAIN + hashlib.sha256(attestation.unsigned_bytes()).digest(),
    )
    verified = _verify(receipt, policy, attestation, registry)
    assert verified.valid and verified.signed_receipt is not None
    assert verified.signed_receipt.runner_attestation_cid == attestation.cid


def test_canonical_decoding_rejects_alternative_and_tampered_attestation_bytes() -> None:
    _, _, attestation, _, _ = _attested()
    encoded = attestation.canonical_bytes()
    assert decode_canonical_dag_cbor(encoded)["interface"] == "RunnerPassAttestation@1"
    with pytest.raises(RunnerAttestationError):
        decode_canonical_dag_cbor(encoded + b"\x00")
    altered = bytearray(encoded)
    altered[-1] ^= 1
    with pytest.raises(RunnerAttestationError):
        RunnerPassAttestation.from_bytes(bytes(altered), expected_cid=attestation.cid)


def test_pinned_policy_and_exact_context_are_required_before_signature_authority() -> None:
    receipt, policy, attestation, registry, _ = _attested()
    assert _verify(receipt, policy, attestation, registry).valid
    wrong_policy = dag_cbor_cid({"not": "the-local-pin"})
    result = verify_runner_pass_attestation(
        attestation,
        receipt=receipt,
        policy=policy,
        pinned_policy_cid=wrong_policy,
        current_execution_key_cid=receipt.execution_key_cid,
        current_candidate_context_cid=attestation.candidate_context_cid,
        now=NOW,
        nonce_registry=registry,
    )
    assert not result.valid
    cross_context = verify_runner_pass_attestation(
        attestation,
        receipt=receipt,
        policy=policy,
        pinned_policy_cid=policy.cid,
        current_execution_key_cid=receipt.execution_key_cid,
        current_candidate_context_cid=dag_cbor_cid({"candidate": "other"}),
        now=NOW,
        nonce_registry=registry,
    )
    assert not cross_context.valid and "context" in cross_context.reason


def test_expired_revoked_wrong_key_and_altered_receipts_run() -> None:
    receipt, policy, attestation, registry, public = _attested()
    assert _verify(receipt, policy, attestation, registry).valid
    expired = _verify(receipt, policy, attestation, registry)
    assert expired.valid
    expiry = verify_runner_pass_attestation(
        attestation, receipt=receipt, policy=policy, pinned_policy_cid=policy.cid,
        current_execution_key_cid=receipt.execution_key_cid, current_candidate_context_cid=attestation.candidate_context_cid,
        now=NOW + 61, nonce_registry=registry,
    )
    assert not expiry.valid
    revoked_policy = RunnerTrustPolicy(policy.trust_domain, policy.active_key_epoch, policy.keys, policy.policy_epoch, (public.cid,))
    revoked = verify_runner_pass_attestation(
        attestation, receipt=receipt, policy=revoked_policy, pinned_policy_cid=revoked_policy.cid,
        current_execution_key_cid=receipt.execution_key_cid, current_candidate_context_cid=attestation.candidate_context_cid,
        now=NOW, nonce_registry=registry,
    )
    assert not revoked.valid
    changed_receipt = TestPassReceipt.from_dict({**receipt.to_dict(), "execution_key_cid": dag_cbor_cid({"execution": "substituted"})})
    assert not _verify(changed_receipt, policy, attestation, registry).valid


def test_policy_scopes_keys_to_pytest_passes_and_validates_rotation() -> None:
    _, public, _ = _material()
    with pytest.raises(RunnerAttestationError, match="restricted"):
        RunnerKeyRecord(
            public_key_cid=public.cid,
            public_key_material=public.material,
            key_epoch="epoch-7",
            not_before=NOW - 1,
            not_after=NOW + 1,
            usages=("artifact-signing",),
        )

    predecessor = RunnerPublicKey.from_public_key(Ed25519PrivateKey.generate().public_key())
    successor = RunnerPublicKey.from_public_key(Ed25519PrivateKey.generate().public_key())
    rotating_key = RunnerKeyRecord(
        public_key_cid=successor.cid,
        public_key_material=successor.material,
        key_epoch="epoch-8",
        not_before=NOW - 1,
        not_after=NOW + 1,
        replaces_key_cid=predecessor.cid,
    )
    with pytest.raises(RunnerAttestationError, match="predecessor"):
        RunnerTrustPolicy("pytest.local", "epoch-8", (rotating_key,))


def test_nonce_is_issuance_binding_not_consume_on_warm_cache_read() -> None:
    receipt, policy, attestation, registry, _ = _attested()
    before = registry.snapshot()
    assert _verify(receipt, policy, attestation, registry).valid
    assert _verify(receipt, policy, attestation, registry).valid
    assert registry.snapshot() == before

    # The registry binds an issuance nonce to one immutable artifact.  This
    # deliberately malformed replacement is enough to demonstrate that the
    # registry rejects an attempted reissue; verification itself never mutates
    # the registry and therefore never consumes a nonce on a warm cache read.
    replacement = RunnerPassAttestation(
        receipt_cid=attestation.receipt_cid,
        execution_key_cid=attestation.execution_key_cid,
        candidate_context_cid=attestation.candidate_context_cid,
        phase_root_cid=attestation.phase_root_cid,
        trace_root_cid=attestation.trace_root_cid,
        policy_cid=attestation.policy_cid,
        trust_domain=attestation.trust_domain,
        signer_key_cid=attestation.signer_key_cid,
        key_epoch=attestation.key_epoch,
        issuance_nonce=attestation.issuance_nonce,
        issued_at=attestation.issued_at + 1,
        signature=b"\x00" * 64,
    )
    with pytest.raises(RunnerAttestationError, match="nonce"):
        registry.register_issuance(replacement)


def test_external_local_key_pin_is_an_additional_non_tofu_check_and_no_secret_is_public() -> None:
    receipt, policy, attestation, registry, public = _attested()
    verified = verify_runner_pass_attestation_with_key(
        attestation, receipt=receipt, policy=policy, pinned_policy_cid=policy.cid,
        current_execution_key_cid=receipt.execution_key_cid, current_candidate_context_cid=attestation.candidate_context_cid,
        pinned_public_key_material=public.material, now=NOW, nonce_registry=registry,
    )
    assert verified.valid
    wrong = RunnerPublicKey.from_public_key(Ed25519PrivateKey.generate().public_key())
    assert not verify_runner_pass_attestation_with_key(
        attestation, receipt=receipt, policy=policy, pinned_policy_cid=policy.cid,
        current_execution_key_cid=receipt.execution_key_cid, current_candidate_context_cid=attestation.candidate_context_cid,
        pinned_public_key_material=wrong.material, now=NOW, nonce_registry=registry,
    ).valid
    public_artifact = attestation.canonical_bytes() + policy.canonical_bytes()
    assert b"private_key" not in public_artifact and b"witness" not in public_artifact
