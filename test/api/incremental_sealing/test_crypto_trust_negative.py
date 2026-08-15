"""IPS-049: reject cryptographic, signature, key, circuit, and claim tampering.

Old/unallowlisted keys, old circuit, modified public input, valid-format
invalid cryptography, absent/invalid/untrusted receipt signature, unknown
proof system, simulated-as-real, exposed secrets, and receipt-as-execution
claims must fail closed with typed reasons.  No production path may treat
simulated evidence or receipt aggregation as direct execution.

Evidence subset: ``ips/crypto-trust-negative@1``.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.admission import (
    DEFAULT_ALLOWED_PROOF_SYSTEMS,
    AdmissionPolicy,
    EvidenceCandidate,
    RejectionReason,
    verify_for_admission,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.aggregation import (
    AggregationReason,
    VerifiedUnit,
    aggregate_verified_units,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint import (
    FullCheckpointReason,
    RequiredUnitEvidence,
    RepositoryStateView,
    VerificationPolicyView,
    create_full_checkpoint,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.provers import (
    IncrementalProofBackendAdapter,
    ProgramRegistry,
    ProverInvocation,
    ProverReasonCode,
    ProverStatus,
    RegisteredProgram,
    VerificationInvocation,
    assert_no_sensitive_material,
    prove,
    verify,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.trust import (
    ProvingKeyHandle,
    ProvingKeyRecord,
    SetupOrigin,
    SignerTrustRecord,
    TrustError,
    TrustRejectionReason,
    TrustedProofPolicy,
    VerificationKeyRecord,
    build_production_policy,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.verification import (
    SealVerificationReason,
    UnitProofView,
    verify_seal,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    DirectExecutionProof,
    IntegrityCommitment,
    ProofMode,
    ProofTerminalStatus,
    ReceiptAggregationZkProof,
    SealStatus,
    SignedExecutionReceipt,
)

# ---------------------------------------------------------------------------
# Evidence / closed contracts
# ---------------------------------------------------------------------------

CRYPTO_TRUST_NEGATIVE_EVIDENCE = "ips/crypto-trust-negative@1"
EVIDENCE_SUBSETS = (CRYPTO_TRUST_NEGATIVE_EVIDENCE,)

_DIGEST_A = "sha256:" + ("aa" * 32)
_DIGEST_B = "sha256:" + ("bb" * 32)
_DIGEST_C = "sha256:" + ("cc" * 32)
_DIGEST_D = "sha256:" + ("dd" * 32)
_DIGEST_E = "sha256:" + ("ee" * 32)
_DIGEST_F = "sha256:" + ("ff" * 32)
_DIGEST_1 = "sha256:" + ("11" * 32)
_DIGEST_2 = "sha256:" + ("22" * 32)

_VK_ID = "vk/crypto-trust-1"
_VK_CID = "bafybeigverificationkeycryptotrust0000000000001"
_VK_CID_B = "bafybeigverificationkeycryptotrust0000000000002"
_PK_ID = "pk/crypto-trust-1"
_PK_CID = "bafybeigprovingkeycryptotrust00000000000000001"
_PK_CID_B = "bafybeigprovingkeycryptotrust00000000000000002"
_CIRCUIT = "circuit:ips-crypto-trust@1"
_CIRCUIT_OLD = "circuit:ips-crypto-trust@0"
_CIRCUIT_OTHER = "circuit:ips-other@9"
_SIGNER = "allowlist/crypto-trust-operator"
_SIGNER_UNTRUSTED = "allowlist/crypto-trust-unknown"

_PROGRAM_ID = "program:ips-hermetic-hmac@1"
_HERMETIC_CIRCUIT = "circuit:ips-hermetic-hmac@1"
_HERMETIC_BACKEND = "hermetic_hmac"
_HERMETIC_VK_ID = "vk/hermetic-1"
_HERMETIC_PK_ID = "pk/hermetic-1"
_HERMETIC_VK_CID = "bafybeigverificationkeyhermetic0000000000000001"
_HERMETIC_PK_CID = "bafybeigprovingkeyhermetic000000000000000000001"
_PUBLIC = b"ips-crypto-trust-public-input-v1\ncommitted\n"
_WITNESS = b"ips-crypto-trust-witness-secret-v1\nnever-export\n"
_POLICY_CID = _DIGEST_D
_TRUSTED_KEYS = (_VK_ID, "n/a")


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------


def _vk(**overrides: object) -> VerificationKeyRecord:
    payload: dict[str, object] = {
        "key_id": _VK_ID,
        "key_cid": _VK_CID,
        "circuit_ids": frozenset({_CIRCUIT}),
        "setup_origin": SetupOrigin.OPERATOR_REVIEWED,
        "test_only": False,
        "epoch": 2,
    }
    payload.update(overrides)
    return VerificationKeyRecord(**payload)  # type: ignore[arg-type]


def _pk(**overrides: object) -> ProvingKeyRecord:
    payload: dict[str, object] = {
        "key_id": _PK_ID,
        "key_cid": _PK_CID,
        "circuit_ids": frozenset({_CIRCUIT}),
        "setup_origin": SetupOrigin.OPERATOR_REVIEWED,
        "test_only": False,
        "paired_verification_key_id": _VK_ID,
        "epoch": 2,
    }
    payload.update(overrides)
    return ProvingKeyRecord(**payload)  # type: ignore[arg-type]


def _signer(**overrides: object) -> SignerTrustRecord:
    payload: dict[str, object] = {
        "signer_id": _SIGNER,
        "scopes": frozenset({"seal", "receipt"}),
        "trusted": True,
        "test_only": False,
        "revocation_epoch": None,
    }
    payload.update(overrides)
    return SignerTrustRecord(**payload)  # type: ignore[arg-type]


def _production_policy(**kwargs: object) -> TrustedProofPolicy:
    return build_production_policy(
        verification_keys=kwargs.pop("verification_keys", (_vk(),)),  # type: ignore[arg-type]
        proving_keys=kwargs.pop("proving_keys", (_pk(),)),  # type: ignore[arg-type]
        signers=kwargs.pop("signers", (_signer(),)),  # type: ignore[arg-type]
        current_epoch=kwargs.pop("current_epoch", 5),  # type: ignore[arg-type]
        minimum_key_epoch=kwargs.pop("minimum_key_epoch", 2),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def _integrity(**overrides: object) -> IntegrityCommitment:
    payload: dict[str, object] = {
        "digest": _DIGEST_A,
        "cid": _DIGEST_B,
        "merkle_inclusion": "leaf:0",
        "byte_length": 32,
    }
    payload.update(overrides)
    return IntegrityCommitment(**payload)  # type: ignore[arg-type]


def _signed(**overrides: object) -> SignedExecutionReceipt:
    payload: dict[str, object] = {
        "signer_id": _SIGNER,
        "receipt_digest": _DIGEST_A,
        "signature": "ed25519:sig-valid",
        "statement": "pytest node completed",
    }
    payload.update(overrides)
    return SignedExecutionReceipt(**payload)  # type: ignore[arg-type]


def _aggregation(**overrides: object) -> ReceiptAggregationZkProof:
    payload: dict[str, object] = {
        "circuit_id": "agg@v1",
        "receipt_digests": (_DIGEST_A, _DIGEST_B),
        "proof_cid": _DIGEST_C,
    }
    payload.update(overrides)
    return ReceiptAggregationZkProof(**payload)  # type: ignore[arg-type]


def _direct(**overrides: object) -> DirectExecutionProof:
    payload: dict[str, object] = {
        "program_id": "prog/direct-1",
        "input_commitment": _DIGEST_A,
        "output_commitment": _DIGEST_B,
        "proof_system_id": "groth16",
        "proof_cid": _DIGEST_C,
    }
    payload.update(overrides)
    return DirectExecutionProof(**payload)  # type: ignore[arg-type]


def _candidate(evidence: object, **overrides: object) -> EvidenceCandidate:
    base: dict[str, object] = {
        "evidence": evidence,
        "proof_system_id": "integrity",
        "public_input_cid": _DIGEST_A,
        "proof_unit_id": "unit/crypto-trust-1",
        "proof_object_cid": _DIGEST_C,
        "required_for_seal": True,
        "proof_mode": ProofMode.INTEGRITY_ONLY,
        "terminal_status": ProofTerminalStatus.INTEGRITY_VERIFIED,
        "expected_digest": _DIGEST_A,
        "observed_digest": _DIGEST_A,
        "logical_epoch": 1,
    }
    base.update(overrides)
    return EvidenceCandidate(**base)  # type: ignore[arg-type]


def _state(**overrides: object) -> RepositoryStateView:
    payload: dict[str, object] = {
        "repository_id": "repo/accelerate",
        "revision": "rev-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "source_root_cid": _DIGEST_A,
        "repository_state_cid": _DIGEST_B,
        "environment_cid": _DIGEST_C,
        "parent_revision_ids": (),
    }
    payload.update(overrides)
    return RepositoryStateView(**payload)  # type: ignore[arg-type]


def _policy(**overrides: object) -> VerificationPolicyView:
    payload: dict[str, object] = {
        "policy_cid": _POLICY_CID,
        "proof_schema_version": "1",
        "canonicalization_version": "1",
        "dependency_graph_schema_version": "graph@1",
        "circuit_id": _CIRCUIT,
        "verification_key_id": _VK_ID,
    }
    payload.update(overrides)
    return VerificationPolicyView(**payload)  # type: ignore[arg-type]


def _unit(unit_id: str, **overrides: object) -> RequiredUnitEvidence:
    payload: dict[str, object] = {
        "unit_id": unit_id,
        "proof_object_cid": _DIGEST_E,
        "category": "unit_test",
        "terminal_status": ProofTerminalStatus.INTEGRITY_VERIFIED.value,
        "proof_mode": ProofMode.INTEGRITY_ONLY.value,
        "required_for_seal": True,
        "freshly_verified": True,
        "cache_reused_without_fresh_verification": False,
        "circuit_id": _CIRCUIT,
        "verification_key_id": _VK_ID,
    }
    payload.update(overrides)
    return RequiredUnitEvidence(**payload)  # type: ignore[arg-type]


def _full_seal(**overrides: object):
    return create_full_checkpoint(
        _state(),
        _policy(),
        units=(
            _unit("unit/a"),
            _unit(
                "unit/b",
                category="static_analysis",
                proof_object_cid=_DIGEST_F,
                terminal_status=ProofTerminalStatus.PROVED.value,
                proof_mode=ProofMode.DIRECT_EXECUTION_PROOF.value,
            ),
        ),
        expected_unit_ids=("unit/a", "unit/b"),
        parent_seal_cid=None,
        fallback_reasons=("first_state", "missing_parent"),
        **overrides,
    )


def _hermetic_handle(**overrides: object) -> ProvingKeyHandle:
    payload: dict[str, object] = {
        "key_id": _HERMETIC_PK_ID,
        "key_cid": _HERMETIC_PK_CID,
        "circuit_ids": frozenset({_HERMETIC_CIRCUIT}),
        "setup_origin": SetupOrigin.OPERATOR_REVIEWED,
        "test_only": False,
        "paired_verification_key_id": _HERMETIC_VK_ID,
        "epoch": 1,
    }
    payload.update(overrides)
    return ProvingKeyHandle(**payload)  # type: ignore[arg-type]


def _prove_invocation(**overrides: object) -> ProverInvocation:
    payload: dict[str, object] = {
        "program_id": _PROGRAM_ID,
        "circuit_id": _HERMETIC_CIRCUIT,
        "public_input": _PUBLIC,
        "witness": _WITNESS,
        "proving_key_handle": _hermetic_handle(),
        "verification_key_id": _HERMETIC_VK_ID,
        "verification_key_cid": _HERMETIC_VK_CID,
        "backend_id": _HERMETIC_BACKEND,
        "proof_unit_id": "unit/crypto-trust-hermetic",
        "production": True,
    }
    payload.update(overrides)
    return ProverInvocation(**payload)  # type: ignore[arg-type]


def _verify_invocation(
    proof_bytes: bytes,
    *,
    public_input: bytes = _PUBLIC,
    **overrides: object,
) -> VerificationInvocation:
    payload: dict[str, object] = {
        "program_id": _PROGRAM_ID,
        "circuit_id": _HERMETIC_CIRCUIT,
        "public_input": public_input,
        "proof_bytes": proof_bytes,
        "verification_key_id": _HERMETIC_VK_ID,
        "verification_key_cid": _HERMETIC_VK_CID,
        "backend_id": _HERMETIC_BACKEND,
        "proof_unit_id": "unit/crypto-trust-hermetic",
        "production": True,
        "metadata": {"proving_key_cid": _HERMETIC_PK_CID},
    }
    payload.update(overrides)
    return VerificationInvocation(**payload)  # type: ignore[arg-type]


def _agg_unit(unit_id: str, **overrides: object) -> VerifiedUnit:
    payload: dict[str, object] = {
        "unit_id": unit_id,
        "proof_object_cid": _DIGEST_A,
        "category": "unit_test",
        "terminal_status": "integrity_verified",
        "repository_state_cid": _DIGEST_1,
        "environment_cid": _DIGEST_2,
    }
    payload.update(overrides)
    return VerifiedUnit(**payload)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Evidence subset
# ---------------------------------------------------------------------------


def test_crypto_trust_negative_evidence_subset() -> None:
    assert CRYPTO_TRUST_NEGATIVE_EVIDENCE == "ips/crypto-trust-negative@1"
    assert CRYPTO_TRUST_NEGATIVE_EVIDENCE in EVIDENCE_SUBSETS


# ---------------------------------------------------------------------------
# Old / unallowlisted keys
# ---------------------------------------------------------------------------


def test_unallowlisted_verification_key_rejects() -> None:
    policy = _production_policy()
    decision = policy.select_verification_key("vk/unknown")
    assert decision.accepted is False
    assert decision.reason_code == TrustRejectionReason.UNALLOWLISTED_VERIFICATION_KEY.value
    assert "not on the allowlist" in decision.message


def test_old_superseded_verification_key_rejects() -> None:
    policy = _production_policy(
        verification_keys=(
            _vk(superseded_by="vk/crypto-trust-2"),
            _vk(key_id="vk/crypto-trust-2", key_cid=_VK_CID_B, epoch=3),
        )
    )
    decision = policy.select_verification_key(_VK_ID, key_cid=_VK_CID)
    assert decision.accepted is False
    assert decision.reason_code == TrustRejectionReason.OLD_VERIFICATION_KEY.value
    assert "superseded" in decision.message


def test_old_epoch_verification_key_rejects() -> None:
    policy = _production_policy(
        minimum_key_epoch=9,
        verification_keys=(_vk(epoch=2),),
    )
    decision = policy.select_verification_key(_VK_ID, key_cid=_VK_CID)
    assert decision.accepted is False
    assert decision.reason_code == TrustRejectionReason.OLD_VERIFICATION_KEY.value


def test_unallowlisted_proving_key_rejects() -> None:
    policy = _production_policy()
    decision, handle = policy.select_proving_key_handle("pk/unknown")
    assert decision.accepted is False
    assert handle is None
    assert decision.reason_code == TrustRejectionReason.UNALLOWLISTED_PROVING_KEY.value


def test_old_proving_key_rejects() -> None:
    policy = _production_policy(
        proving_keys=(_pk(superseded_by="pk/crypto-trust-2"),),
    )
    decision, handle = policy.select_proving_key_handle(_PK_ID, key_cid=_PK_CID)
    assert decision.accepted is False
    assert handle is None
    assert decision.reason_code == TrustRejectionReason.OLD_PROVING_KEY.value


def test_substituted_verification_key_rejects() -> None:
    policy = _production_policy()
    decision = policy.select_verification_key(
        _VK_ID, key_cid=_VK_CID_B, circuit_id=_CIRCUIT
    )
    assert decision.accepted is False
    assert decision.reason_code == TrustRejectionReason.SUBSTITUTED_VERIFICATION_KEY.value


def test_seal_unallowlisted_verification_key_rejects() -> None:
    seal = _full_seal()
    result = verify_seal(seal, ("vk/other",), _policy())
    assert result.accepted is False
    assert result.reason is SealVerificationReason.UNALLOWLISTED_VERIFICATION_KEY
    assert result.failed_stage == "key"


def test_unit_unallowlisted_verification_key_rejects_at_crypto_stage() -> None:
    seal = _full_seal()
    result = verify_seal(
        seal,
        _TRUSTED_KEYS,
        _policy(),
        unit_proofs=(
            UnitProofView(
                unit_id="unit/a",
                proof_object_cid=_DIGEST_E,
                proof_system_id="integrity",
                verification_key_id="vk/injected-unallowlisted",
                freshly_verified=True,
            ),
        ),
    )
    assert result.accepted is False
    assert result.reason is SealVerificationReason.UNALLOWLISTED_VERIFICATION_KEY


# ---------------------------------------------------------------------------
# Old / incompatible circuit
# ---------------------------------------------------------------------------


def test_old_circuit_incompatible_with_verification_key_rejects() -> None:
    policy = _production_policy()
    decision = policy.select_verification_key(
        _VK_ID, key_cid=_VK_CID, circuit_id=_CIRCUIT_OLD
    )
    assert decision.accepted is False
    assert decision.reason_code == TrustRejectionReason.CIRCUIT_INCOMPATIBLE.value


def test_old_circuit_incompatible_with_proving_key_rejects() -> None:
    policy = _production_policy()
    decision, handle = policy.select_proving_key_handle(
        _PK_ID, key_cid=_PK_CID, circuit_id=_CIRCUIT_OTHER
    )
    assert decision.accepted is False
    assert handle is None
    assert decision.reason_code == TrustRejectionReason.CIRCUIT_INCOMPATIBLE.value


def test_seal_old_circuit_rejects_against_current_policy() -> None:
    seal = _full_seal()
    # Seal was built under circuit@current; verify against an older circuit binding.
    old_policy = _policy(circuit_id=_CIRCUIT_OLD)
    result = verify_seal(seal, _TRUSTED_KEYS, old_policy)
    assert result.accepted is False
    assert result.reason is SealVerificationReason.WRONG_POLICY
    assert "circuit_id" in result.message


def test_prover_old_circuit_rejects() -> None:
    adapter = IncrementalProofBackendAdapter()
    outcome = adapter.prove(_prove_invocation(circuit_id=_CIRCUIT_OLD))
    assert outcome.proved is False
    assert outcome.reason_code == ProverReasonCode.UNREGISTERED_CIRCUIT.value


# ---------------------------------------------------------------------------
# Modified public input
# ---------------------------------------------------------------------------


def test_modified_public_input_rejects_admission() -> None:
    decision = verify_for_admission(
        _candidate(
            _integrity(),
            public_input_cid=_DIGEST_A,
            observed_public_input_cid=_DIGEST_B,
        )
    )
    assert decision.admitted is False
    assert decision.reason_code == RejectionReason.PUBLIC_INPUT_MISMATCH.value


def test_modified_direct_execution_public_input_rejects_admission() -> None:
    decision = verify_for_admission(
        _candidate(
            _direct(input_commitment=_DIGEST_A),
            proof_system_id="groth16",
            public_input_cid=_DIGEST_B,
            proof_mode=ProofMode.DIRECT_EXECUTION_PROOF,
            terminal_status=ProofTerminalStatus.PROVED,
            proof_object_cid=_DIGEST_C,
        )
    )
    assert decision.admitted is False
    assert decision.reason_code == RejectionReason.PUBLIC_INPUT_MISMATCH.value


def test_modified_public_input_fails_hermetic_verification() -> None:
    adapter = IncrementalProofBackendAdapter()
    proved = adapter.prove(_prove_invocation())
    assert proved.proved is True

    tampered = adapter.verify(
        _verify_invocation(
            proved.proof_bytes or b"",
            public_input=b"ips-crypto-trust-public-input-v1\nTAMPERED\n",
        )
    )
    assert tampered.proved is False
    assert tampered.verified is False
    assert tampered.status is ProverStatus.VERIFICATION_FAILED
    assert tampered.reason_code == ProverReasonCode.INVALID_CRYPTOGRAPHY.value


def test_modified_public_input_rejects_seal_verification() -> None:
    seal = _full_seal()
    result = verify_seal(
        seal,
        _TRUSTED_KEYS,
        _policy(),
        unit_proofs=(
            UnitProofView(
                unit_id="unit/a",
                proof_object_cid=_DIGEST_E,
                public_input_cid=_DIGEST_A,
                observed_public_input_cid=_DIGEST_1,
                proof_system_id="integrity",
            ),
        ),
    )
    assert result.accepted is False
    assert result.reason is SealVerificationReason.MODIFIED_INPUTS
    assert result.failed_stage == "inputs"


# ---------------------------------------------------------------------------
# Valid-format invalid cryptography
# ---------------------------------------------------------------------------


def test_valid_format_invalid_proof_bytes_fail_verification() -> None:
    """Well-formed byte length that fails cryptographic re-check must reject."""
    adapter = IncrementalProofBackendAdapter()
    # 64-byte payload is a valid-looking length for the hermetic engine but
    # is not a correct MAC under the committed public input / key binding.
    garbage = adapter.verify(
        _verify_invocation(b"\x00" * 64)
    )
    assert garbage.proved is False
    assert garbage.status is ProverStatus.VERIFICATION_FAILED
    assert garbage.reason_code == ProverReasonCode.INVALID_CRYPTOGRAPHY.value


def test_valid_format_invalid_cryptography_rejects_seal_digest() -> None:
    seal = _full_seal()
    proof_bytes = b"valid-looking-proof-material-v1"
    # Present a correctly shaped sha256: digest that does not match the bytes.
    wrong_digest = "sha256:" + ("00" * 32)
    result = verify_seal(
        seal,
        _TRUSTED_KEYS,
        _policy(),
        unit_proofs=(
            UnitProofView(
                unit_id="unit/a",
                proof_object_cid=wrong_digest,
                proof_bytes=proof_bytes,
                expected_proof_digest=wrong_digest,
                proof_system_id="integrity",
            ),
        ),
    )
    assert result.accepted is False
    assert result.reason is SealVerificationReason.CRYPTOGRAPHIC_FAILURE
    assert result.failed_stage == "cryptography"
    computed = "sha256:" + hashlib.sha256(proof_bytes).hexdigest()
    assert computed != wrong_digest


def test_structural_cryptographic_failure_rejects_direct_execution() -> None:
    decision = verify_for_admission(
        _candidate(
            _direct(proof_cid=_DIGEST_C),
            proof_system_id="groth16",
            public_input_cid=_DIGEST_A,
            proof_mode=ProofMode.DIRECT_EXECUTION_PROOF,
            terminal_status=ProofTerminalStatus.PROVED,
            # Valid-format CID that does not match the evidence proof object.
            proof_object_cid=_DIGEST_1,
        )
    )
    assert decision.admitted is False
    assert decision.reason_code == RejectionReason.VERIFIER_FAILURE.value


def test_injected_verifier_false_is_cryptographic_failure() -> None:
    decision = verify_for_admission(
        _candidate(
            _direct(),
            proof_system_id="groth16",
            public_input_cid=_DIGEST_A,
            proof_mode=ProofMode.DIRECT_EXECUTION_PROOF,
            terminal_status=ProofTerminalStatus.PROVED,
            proof_object_cid=_DIGEST_C,
        ),
        verifier=lambda _evidence, _meta: False,
    )
    assert decision.admitted is False
    assert decision.reason_code == RejectionReason.VERIFIER_FAILURE.value


# ---------------------------------------------------------------------------
# Absent / invalid / untrusted receipt signature
# ---------------------------------------------------------------------------


def test_absent_receipt_signature_rejects() -> None:
    policy = AdmissionPolicy(
        allowed_proof_systems=DEFAULT_ALLOWED_PROOF_SYSTEMS,
        allowed_signers=frozenset({_SIGNER}),
    )
    decision = verify_for_admission(
        _candidate(
            _signed(signature="unsigned"),
            proof_system_id="signed_receipt",
            proof_mode=ProofMode.SIGNED_RECEIPT,
            terminal_status=ProofTerminalStatus.SIGNED_ASSERTION_VERIFIED,
        ),
        policy=policy,
    )
    assert decision.admitted is False
    assert decision.reason_code == RejectionReason.UNSIGNED_REQUIRED_RECEIPT.value
    assert decision.cache_admission_record is None


def test_invalid_receipt_signature_marker_rejects() -> None:
    policy = AdmissionPolicy(allowed_signers=frozenset({_SIGNER}))
    for marker in ("none", "null", "n/a", "missing"):
        decision = verify_for_admission(
            _candidate(
                _signed(signature=marker),
                proof_system_id="signed_receipt",
                proof_mode=ProofMode.SIGNED_RECEIPT,
                terminal_status=ProofTerminalStatus.SIGNED_ASSERTION_VERIFIED,
            ),
            policy=policy,
        )
        assert decision.admitted is False, marker
        assert (
            decision.reason_code == RejectionReason.UNSIGNED_REQUIRED_RECEIPT.value
        ), marker


def test_untrusted_receipt_signer_rejects_admission() -> None:
    policy = AdmissionPolicy(allowed_signers=frozenset({_SIGNER}))
    decision = verify_for_admission(
        _candidate(
            _signed(signer_id=_SIGNER_UNTRUSTED),
            proof_system_id="signed_receipt",
            proof_mode=ProofMode.SIGNED_RECEIPT,
            terminal_status=ProofTerminalStatus.SIGNED_ASSERTION_VERIFIED,
        ),
        policy=policy,
    )
    assert decision.admitted is False
    assert decision.reason_code == RejectionReason.UNALLOWLISTED_SIGNER.value


def test_untrusted_signer_rejects_trust_policy() -> None:
    policy = _production_policy()
    decision = policy.select_signer(_SIGNER_UNTRUSTED, scope="receipt")
    assert decision.accepted is False
    assert decision.reason_code == TrustRejectionReason.UNTRUSTED_SIGNER.value


def test_absent_signature_rejects_seal_verification() -> None:
    seal = _full_seal()
    result = verify_seal(
        seal,
        _TRUSTED_KEYS,
        _policy(),
        unit_proofs=(
            UnitProofView(
                unit_id="unit/a",
                proof_object_cid=_DIGEST_E,
                proof_system_id="signed_receipt",
                signature="unsigned",
            ),
        ),
    )
    assert result.accepted is False
    assert result.reason is SealVerificationReason.SIGNATURE_FAILURE
    assert result.failed_stage == "signature"


def test_invalid_signature_rejects_when_cryptographic_verifier_injected() -> None:
    def reject_bad_sig(evidence: Any, _meta: Any) -> bool:
        if isinstance(evidence, SignedExecutionReceipt):
            return evidence.signature == "ed25519:sig-valid"
        return False

    policy = AdmissionPolicy(allowed_signers=frozenset({_SIGNER}))
    decision = verify_for_admission(
        _candidate(
            _signed(signature="ed25519:sig-forged-but-well-formed"),
            proof_system_id="signed_receipt",
            proof_mode=ProofMode.SIGNED_RECEIPT,
            terminal_status=ProofTerminalStatus.SIGNED_ASSERTION_VERIFIED,
        ),
        policy=policy,
        verifier=reject_bad_sig,
    )
    assert decision.admitted is False
    assert decision.reason_code == RejectionReason.VERIFIER_FAILURE.value


# ---------------------------------------------------------------------------
# Unknown proof system
# ---------------------------------------------------------------------------


def test_unknown_proof_system_rejects_admission() -> None:
    decision = verify_for_admission(
        _candidate(
            _integrity(),
            proof_system_id="exotic-unknown-system",
        )
    )
    assert decision.admitted is False
    assert decision.outcome.value == "rejected"
    assert decision.reason_code == RejectionReason.UNKNOWN_PROOF_SYSTEM.value
    assert decision.cache_admission_record is None


def test_unknown_proof_system_rejects_seal_verification() -> None:
    seal = _full_seal()
    result = verify_seal(
        seal,
        _TRUSTED_KEYS,
        _policy(),
        unit_proofs=(
            UnitProofView(
                unit_id="unit/a",
                proof_object_cid=_DIGEST_E,
                proof_system_id="invented_system",
            ),
        ),
    )
    assert result.accepted is False
    assert result.reason is SealVerificationReason.UNKNOWN_PROOF_SYSTEM
    assert result.failed_stage == "proof_system"


def test_unknown_prover_backend_rejects() -> None:
    adapter = IncrementalProofBackendAdapter()
    outcome = adapter.prove(_prove_invocation(backend_id="exotic-snark"))
    assert outcome.proved is False
    assert outcome.status is ProverStatus.UNKNOWN
    assert outcome.reason_code == ProverReasonCode.UNKNOWN_BACKEND.value


# ---------------------------------------------------------------------------
# Simulated presented as real
# ---------------------------------------------------------------------------


def test_simulated_required_unit_rejects_admission() -> None:
    decision = verify_for_admission(
        _candidate(
            _integrity(),
            proof_mode=ProofMode.SIMULATED,
            terminal_status=ProofTerminalStatus.SIMULATED,
            required_for_seal=True,
        )
    )
    assert decision.admitted is False
    assert decision.reason_code == RejectionReason.SIMULATED_REQUIRED_UNIT.value
    assert "direct-execution" in decision.message


def test_simulated_status_alone_rejects_as_real() -> None:
    decision = verify_for_admission(
        _candidate(
            _integrity(),
            proof_mode=ProofMode.INTEGRITY_ONLY,
            terminal_status=ProofTerminalStatus.SIMULATED,
            required_for_seal=True,
        )
    )
    assert decision.admitted is False
    assert decision.reason_code == RejectionReason.SIMULATED_REQUIRED_UNIT.value


def test_simulated_required_unit_prevents_sealed_full() -> None:
    units = (
        _unit("unit/a"),
        _unit(
            "unit/sim",
            proof_mode=ProofMode.SIMULATED.value,
            terminal_status=ProofTerminalStatus.SIMULATED.value,
        ),
    )
    seal = create_full_checkpoint(_state(), _policy(), units=units)
    assert seal.sealed is False
    assert seal.seal_status is SealStatus.SIMULATED_ONLY
    assert seal.reason is FullCheckpointReason.SIMULATED_REQUIRED_UNIT
    assert "unit/sim" in seal.rejected_unit_ids
    assert seal.seal_status is not SealStatus.SEALED_FULL


def test_simulated_backend_forbidden_as_production_proof() -> None:
    registry = ProgramRegistry(
        (
            RegisteredProgram(
                program_id="program:sim@1",
                circuit_id="circuit:sim@1",
                backend_id="simulated",
                production_allowed=True,
            ),
        )
    )
    adapter = IncrementalProofBackendAdapter(programs=registry)
    outcome = adapter.prove(
        _prove_invocation(
            program_id="program:sim@1",
            circuit_id="circuit:sim@1",
            backend_id="simulated",
            production=True,
        )
    )
    assert outcome.proved is False
    assert outcome.status is ProverStatus.SIMULATED
    assert outcome.reason_code == ProverReasonCode.SIMULATED_FORBIDDEN.value


# ---------------------------------------------------------------------------
# Exposed secrets
# ---------------------------------------------------------------------------


def test_proving_key_handle_never_exports_secret_bytes() -> None:
    policy = _production_policy()
    decision, handle = policy.select_proving_key_handle(
        _PK_ID,
        key_cid=_PK_CID,
        circuit_id=_CIRCUIT,
        paired_verification_key_id=_VK_ID,
    )
    assert decision.accepted is True
    assert handle is not None
    assert handle.exportable is False
    assert handle.bytes_available is False
    with pytest.raises(TrustError, match="nonexportable"):
        handle.export_bytes()
    public = handle.to_public_api()
    assert public["proving_key_exported"] is False
    for forbidden in (
        "proving_key_bytes",
        "key_bytes",
        "private_key",
        "witness",
        "trapdoor",
        "secret",
    ):
        assert forbidden not in public
    assert_no_sensitive_material(public)


def test_prover_receipts_and_logs_never_expose_witness_or_keys() -> None:
    outcome = prove(_prove_invocation())
    assert outcome.proved is True

    receipt = outcome.to_receipt()
    canonical = outcome.to_canonical()
    canonical_json = outcome.to_canonical_json()
    invocation = _prove_invocation().to_canonical()

    for surface in (receipt, canonical, invocation):
        assert surface["witness_exported"] is False
        assert surface["proving_key_exported"] is False
        assert "witness" not in surface
        assert "witness_bytes" not in surface
        assert "proving_key_bytes" not in surface
        assert "private_key" not in surface
        assert_no_sensitive_material(surface)

    assert "never-export" not in canonical_json
    assert _WITNESS.hex() not in canonical_json
    assert "witness=" not in canonical_json

    for line in outcome.log_lines:
        assert "never-export" not in line
        assert "witness=" not in line.casefold()
        assert "proving_key_bytes" not in line.casefold()
        assert_no_sensitive_material(line)


def test_seal_unit_proof_view_never_exports_proof_bytes() -> None:
    view = UnitProofView(
        unit_id="unit/a",
        proof_object_cid=_DIGEST_E,
        proof_bytes=b"secret-proof-material",
        proof_system_id="integrity",
    )
    canonical = view.to_canonical()
    assert canonical["proof_bytes_exported"] is False
    assert "proof_bytes" not in canonical
    assert "secret-proof-material" not in str(canonical)
    assert_no_sensitive_material(canonical)


def test_trust_decision_details_never_embed_secrets() -> None:
    policy = _production_policy()
    decision = policy.select_verification_key(_VK_ID, key_cid=_VK_CID, circuit_id=_CIRCUIT)
    assert decision.accepted is True
    details = dict(decision.details)
    for forbidden in (
        "proving_key_bytes",
        "witness",
        "private_key",
        "trapdoor",
        "secret",
    ):
        assert forbidden not in details
    assert_no_sensitive_material(details)


# ---------------------------------------------------------------------------
# Receipt-as-execution claims
# ---------------------------------------------------------------------------


def test_receipt_aggregation_never_claims_tests_executed_on_admission() -> None:
    evidence = _aggregation()
    decision = verify_for_admission(
        _candidate(
            evidence,
            proof_system_id="receipt_aggregation",
            proof_mode=ProofMode.RECEIPT_AGGREGATION,
            terminal_status=ProofTerminalStatus.PROVED,
            public_input_cid=_DIGEST_A,
            proof_object_cid=evidence.proof_cid,
        )
    )
    assert decision.admitted is True
    establishes = decision.establishes.casefold()
    does_not = decision.does_not_establish.casefold()
    assert "tests executed" not in establishes
    assert "tests ran" not in establishes
    assert "tests ran" in does_not or "test execution" in does_not
    record = decision.cache_admission_record
    assert record is not None
    assert "tests ran" in record.does_not_establish.casefold() or (
        "test execution" in record.does_not_establish.casefold()
    )


def test_receipt_aggregation_execution_claim_rejects() -> None:
    result = aggregate_verified_units(
        (_agg_unit("unit/a"), _agg_unit("unit/b")),
        receipt_claim="receipt aggregation proves tests executed",
    )
    assert result.accepted is False
    assert result.reason is AggregationReason.EXECUTION_OVERCLAIM


def test_lower_class_cannot_claim_direct_computation() -> None:
    decision = verify_for_admission(
        _candidate(
            _integrity(),
            metadata={"direct_computation_claim": True},
        )
    )
    assert decision.admitted is False
    assert decision.reason_code == RejectionReason.OVERCLAIM.value


def test_signed_receipt_does_not_establish_independent_execution() -> None:
    policy = AdmissionPolicy(allowed_signers=frozenset({_SIGNER}))
    decision = verify_for_admission(
        _candidate(
            _signed(),
            proof_system_id="signed_receipt",
            proof_mode=ProofMode.SIGNED_RECEIPT,
            terminal_status=ProofTerminalStatus.SIGNED_ASSERTION_VERIFIED,
        ),
        policy=policy,
    )
    assert decision.admitted is True
    does_not = decision.does_not_establish.casefold()
    assert "without trusting the signer" in does_not or "independent" in does_not
    establishes = decision.establishes.casefold()
    assert "tests executed" not in establishes
    assert "direct execution" not in establishes or "signer" in establishes


# ---------------------------------------------------------------------------
# Consolidated acceptance matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "fn"),
    [
        ("old_verification_key", lambda: (
            _production_policy(
                verification_keys=(_vk(superseded_by="vk/next"),),
            ).select_verification_key(_VK_ID).accepted
        )),
        ("unallowlisted_verification_key", lambda: (
            _production_policy().select_verification_key("vk/x").accepted
        )),
        ("unallowlisted_proving_key", lambda: (
            _production_policy().select_proving_key_handle("pk/x")[0].accepted
        )),
        ("old_circuit", lambda: (
            _production_policy()
            .select_verification_key(_VK_ID, circuit_id=_CIRCUIT_OLD)
            .accepted
        )),
        ("modified_public_input", lambda: (
            verify_for_admission(
                _candidate(
                    _integrity(),
                    public_input_cid=_DIGEST_A,
                    observed_public_input_cid=_DIGEST_B,
                )
            ).admitted
        )),
        ("unknown_proof_system", lambda: (
            verify_for_admission(
                _candidate(_integrity(), proof_system_id="nope")
            ).admitted
        )),
        ("absent_receipt_signature", lambda: (
            verify_for_admission(
                _candidate(
                    _signed(signature="unsigned"),
                    proof_system_id="signed_receipt",
                    proof_mode=ProofMode.SIGNED_RECEIPT,
                    terminal_status=ProofTerminalStatus.SIGNED_ASSERTION_VERIFIED,
                ),
                policy=AdmissionPolicy(allowed_signers=frozenset({_SIGNER})),
            ).admitted
        )),
        ("untrusted_receipt_signer", lambda: (
            verify_for_admission(
                _candidate(
                    _signed(signer_id=_SIGNER_UNTRUSTED),
                    proof_system_id="signed_receipt",
                    proof_mode=ProofMode.SIGNED_RECEIPT,
                    terminal_status=ProofTerminalStatus.SIGNED_ASSERTION_VERIFIED,
                ),
                policy=AdmissionPolicy(allowed_signers=frozenset({_SIGNER})),
            ).admitted
        )),
        ("simulated_as_real", lambda: (
            verify_for_admission(
                _candidate(
                    _integrity(),
                    proof_mode=ProofMode.SIMULATED,
                    terminal_status=ProofTerminalStatus.SIMULATED,
                )
            ).admitted
        )),
        ("receipt_as_execution_overclaim", lambda: (
            verify_for_admission(
                _candidate(
                    _integrity(),
                    metadata={"direct_computation_claim": True},
                )
            ).admitted
        )),
    ],
)
def test_acceptance_matrix_all_crypto_trust_negatives_reject(
    label: str, fn: Any
) -> None:
    accepted_or_admitted = fn()
    assert accepted_or_admitted is False, f"{label} must reject"


def test_module_level_prove_verify_still_round_trips_for_control() -> None:
    """Positive control: hermetic crypto still works when inputs are honest."""
    outcome = prove(_prove_invocation())
    assert outcome.proved is True
    recheck = verify(
        _verify_invocation(
            outcome.proof_bytes or b"",
            metadata={"proving_key_cid": _HERMETIC_PK_CID},
        )
    )
    assert recheck.proved is True
    assert recheck.reason_code == ProverReasonCode.VERIFIED.value
