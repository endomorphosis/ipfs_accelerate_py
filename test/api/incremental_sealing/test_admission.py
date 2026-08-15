"""IPS-028: evidence-class verification and cache-admission decisions."""

from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.admission import (
    ADMISSION_SCHEMA,
    DEFAULT_ALLOWED_PROOF_SYSTEMS,
    EVIDENCE_SUBSET,
    AdmissionDecision,
    AdmissionError,
    AdmissionOutcome,
    AdmissionPolicy,
    CacheAdmissionRecord,
    EvidenceCandidate,
    EvidenceVerifier,
    RejectionReason,
    closed_rejection_reasons,
    issue_cache_admission_record,
    verify_for_admission,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    DirectExecutionProof,
    IncrementalCommitSeal,
    IntegrityCommitment,
    ProofMode,
    ProofTerminalStatus,
    ReceiptAggregationZkProof,
    SignedExecutionReceipt,
)

_DIGEST = "sha256:" + ("ab" * 32)
_DIGEST_B = "sha256:" + ("cd" * 32)
_DIGEST_C = "sha256:" + ("ef" * 32)
_DIGEST_D = "sha256:" + ("11" * 32)
_DIGEST_E = "sha256:" + ("22" * 32)


def _integrity(**overrides) -> IntegrityCommitment:
    payload = {
        "digest": _DIGEST,
        "cid": _DIGEST_B,
        "merkle_inclusion": "leaf:0",
        "byte_length": 32,
    }
    payload.update(overrides)
    return IntegrityCommitment(**payload)


def _signed(**overrides) -> SignedExecutionReceipt:
    payload = {
        "signer_id": "allowlist/operator-1",
        "receipt_digest": _DIGEST,
        "signature": "ed25519:sig-valid",
        "statement": "pytest node completed",
    }
    payload.update(overrides)
    return SignedExecutionReceipt(**payload)


def _aggregation(**overrides) -> ReceiptAggregationZkProof:
    payload = {
        "circuit_id": "agg@v1",
        "receipt_digests": (_DIGEST, _DIGEST_B),
        "proof_cid": _DIGEST_C,
    }
    payload.update(overrides)
    return ReceiptAggregationZkProof(**payload)


def _direct(**overrides) -> DirectExecutionProof:
    payload = {
        "program_id": "prog/direct-1",
        "input_commitment": _DIGEST,
        "output_commitment": _DIGEST_B,
        "proof_system_id": "groth16",
        "proof_cid": _DIGEST_C,
    }
    payload.update(overrides)
    return DirectExecutionProof(**payload)


def _seal(**overrides) -> IncrementalCommitSeal:
    payload = {
        "parent_seal_cid": _DIGEST,
        "transition_id": "t/1",
        "reused_leaf_cids": (_DIGEST_B,),
        "replacement_leaf_cids": (_DIGEST_C,),
        "manifest_cid": _DIGEST_D,
        "verification_root": _DIGEST_E,
    }
    payload.update(overrides)
    return IncrementalCommitSeal(**payload)


def _candidate(evidence, **overrides) -> EvidenceCandidate:
    base = {
        "evidence": evidence,
        "proof_system_id": "integrity",
        "public_input_cid": _DIGEST,
        "proof_unit_id": "unit/test-1",
        "proof_object_cid": _DIGEST_C,
        "required_for_seal": True,
        "proof_mode": ProofMode.INTEGRITY_ONLY,
        "terminal_status": ProofTerminalStatus.INTEGRITY_VERIFIED,
        "expected_digest": _DIGEST,
        "observed_digest": _DIGEST,
        "logical_epoch": 1,
    }
    base.update(overrides)
    return EvidenceCandidate(**base)


def test_evidence_subset_and_closed_rejection_reasons() -> None:
    assert EVIDENCE_SUBSET == "ips/evidence-admission@1"
    reasons = closed_rejection_reasons()
    for required in (
        "unknown_proof_system",
        "malformed_evidence",
        "nonterminal_required_unit",
        "failed_required_unit",
        "simulated_required_unit",
        "unsigned_required_receipt",
        "public_input_mismatch",
        "verifier_failure",
    ):
        assert required in reasons


def test_unknown_proof_system_rejects_without_admission_record() -> None:
    decision = verify_for_admission(
        _candidate(
            _integrity(),
            proof_system_id="exotic-unknown-system",
        )
    )
    assert decision.admitted is False
    assert decision.outcome is AdmissionOutcome.REJECTED
    assert decision.reason_code == RejectionReason.UNKNOWN_PROOF_SYSTEM.value
    assert decision.cache_admission_record is None
    with pytest.raises(AdmissionError, match="successful verification"):
        issue_cache_admission_record(decision)


def test_malformed_evidence_rejects() -> None:
    decision = verify_for_admission(
        _candidate(
            {
                "evidence_class": "NotARealClass",
                "digest": _DIGEST,
            },
            proof_system_id="integrity",
        )
    )
    assert decision.admitted is False
    assert decision.reason_code == RejectionReason.MALFORMED_EVIDENCE.value
    assert decision.cache_admission_record is None


def test_malformed_mapping_missing_fields_rejects() -> None:
    decision = verify_for_admission(
        _candidate(
            {
                "evidence_class": "IntegrityCommitment",
                "digest": "bad",
            }
        )
    )
    assert decision.admitted is False
    assert decision.reason_code == RejectionReason.MALFORMED_EVIDENCE.value


@pytest.mark.parametrize(
    "status",
    [
        ProofTerminalStatus.UNKNOWN,
        ProofTerminalStatus.TIMEOUT,
        ProofTerminalStatus.UNAVAILABLE,
        ProofTerminalStatus.CANCELLED,
        ProofTerminalStatus.NOT_MODELED,
    ],
)
def test_nonterminal_required_unit_rejects(status: ProofTerminalStatus) -> None:
    decision = verify_for_admission(
        _candidate(
            _integrity(),
            terminal_status=status,
            required_for_seal=True,
        )
    )
    assert decision.admitted is False
    assert decision.reason_code == RejectionReason.NONTERMINAL_REQUIRED_UNIT.value
    assert decision.cache_admission_record is None


@pytest.mark.parametrize(
    "status",
    [
        ProofTerminalStatus.FAILED,
        ProofTerminalStatus.PROOF_FAILED,
        ProofTerminalStatus.INVALID,
        ProofTerminalStatus.STALE,
        ProofTerminalStatus.DISPROVED,
    ],
)
def test_failed_required_unit_rejects(status: ProofTerminalStatus) -> None:
    decision = verify_for_admission(
        _candidate(
            _integrity(),
            terminal_status=status,
            required_for_seal=True,
        )
    )
    assert decision.admitted is False
    assert decision.reason_code == RejectionReason.FAILED_REQUIRED_UNIT.value


def test_simulated_required_unit_rejects() -> None:
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


def test_simulated_status_alone_rejects_required_unit() -> None:
    decision = verify_for_admission(
        _candidate(
            _integrity(),
            proof_mode=ProofMode.INTEGRITY_ONLY,
            terminal_status=ProofTerminalStatus.SIMULATED,
            required_for_seal=True,
        )
    )
    assert decision.reason_code == RejectionReason.SIMULATED_REQUIRED_UNIT.value


def test_unsigned_required_receipt_rejects() -> None:
    policy = AdmissionPolicy(
        allowed_proof_systems=DEFAULT_ALLOWED_PROOF_SYSTEMS,
        allowed_signers=frozenset({"allowlist/operator-1"}),
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


def test_empty_signature_is_unsigned() -> None:
    # Construction of SignedExecutionReceipt rejects empty signature; use
    # a canonical payload that bypasses dataclass construction via mapping
    # is also rejected as malformed.  Use a whitespace-only marker that the
    # receipt accepts but admission treats as unsigned.
    with pytest.raises(Exception):
        _signed(signature="")

    decision = verify_for_admission(
        _candidate(
            _signed(signature="none"),
            proof_system_id="signed_receipt",
            proof_mode=ProofMode.SIGNED_RECEIPT,
            terminal_status=ProofTerminalStatus.SIGNED_ASSERTION_VERIFIED,
        ),
        policy=AdmissionPolicy(
            allowed_signers=frozenset({"allowlist/operator-1"}),
        ),
    )
    assert decision.reason_code == RejectionReason.UNSIGNED_REQUIRED_RECEIPT.value


def test_public_input_mismatch_rejects() -> None:
    decision = verify_for_admission(
        _candidate(
            _integrity(),
            public_input_cid=_DIGEST,
            observed_public_input_cid=_DIGEST_B,
        )
    )
    assert decision.admitted is False
    assert decision.reason_code == RejectionReason.PUBLIC_INPUT_MISMATCH.value


def test_direct_execution_public_input_mismatch_rejects() -> None:
    decision = verify_for_admission(
        _candidate(
            _direct(input_commitment=_DIGEST),
            proof_system_id="groth16",
            public_input_cid=_DIGEST_B,
            proof_mode=ProofMode.DIRECT_EXECUTION_PROOF,
            terminal_status=ProofTerminalStatus.PROVED,
            proof_object_cid=_DIGEST_C,
        )
    )
    assert decision.admitted is False
    assert decision.reason_code == RejectionReason.PUBLIC_INPUT_MISMATCH.value


def test_verifier_failure_rejects() -> None:
    def boom(_evidence, _meta):
        raise RuntimeError("backend exploded")

    decision = verify_for_admission(
        _candidate(
            _direct(),
            proof_system_id="groth16",
            public_input_cid=_DIGEST,
            proof_mode=ProofMode.DIRECT_EXECUTION_PROOF,
            terminal_status=ProofTerminalStatus.PROVED,
            proof_object_cid=_DIGEST_C,
        ),
        verifier=boom,
    )
    assert decision.admitted is False
    assert decision.reason_code == RejectionReason.VERIFIER_FAILURE.value
    assert "backend exploded" in decision.message


def test_verifier_false_rejects() -> None:
    decision = verify_for_admission(
        _candidate(
            _direct(),
            proof_system_id="groth16",
            public_input_cid=_DIGEST,
            proof_mode=ProofMode.DIRECT_EXECUTION_PROOF,
            terminal_status=ProofTerminalStatus.PROVED,
            proof_object_cid=_DIGEST_C,
        ),
        verifier=lambda _e, _m: False,
    )
    assert decision.reason_code == RejectionReason.VERIFIER_FAILURE.value


def test_receipt_aggregation_never_claims_tests_executed() -> None:
    evidence = _aggregation()
    decision = verify_for_admission(
        _candidate(
            evidence,
            proof_system_id="receipt_aggregation",
            proof_mode=ProofMode.RECEIPT_AGGREGATION,
            terminal_status=ProofTerminalStatus.PROVED,
            public_input_cid=_DIGEST,
            proof_object_cid=evidence.proof_cid,
        )
    )
    assert decision.admitted is True
    assert "tests ran" in decision.does_not_establish.casefold() or (
        "test execution" in decision.does_not_establish.casefold()
    )
    assert "tests executed" not in decision.establishes.casefold()
    assert "tests ran" not in decision.establishes.casefold()
    record = decision.cache_admission_record
    assert record is not None
    assert "tests ran" in record.does_not_establish.casefold() or (
        "test execution" in record.does_not_establish.casefold()
    )
    # Aggregation establishes receipt completeness, not execution.
    assert "aggregation" in decision.establishes.casefold() or (
        "receipt" in decision.establishes.casefold()
    )


def test_integrity_admission_issues_verified_record() -> None:
    decision = verify_for_admission(_candidate(_integrity()))
    assert decision.admitted is True
    assert decision.outcome is AdmissionOutcome.ADMITTED
    assert decision.reason_code is None
    assert decision.evidence_class == "IntegrityCommitment"
    assert "execution" in decision.does_not_establish.casefold()
    record = issue_cache_admission_record(decision)
    assert isinstance(record, CacheAdmissionRecord)
    assert record.verified is True
    assert record.schema == ADMISSION_SCHEMA
    assert record.verification_digest.startswith("sha256:")
    payload = json.loads(record.to_canonical_json())
    assert payload["cache_admission"] == "verified_only"
    assert payload["verified"] is True


def test_signed_receipt_admission_with_allowlisted_signer() -> None:
    policy = AdmissionPolicy(
        allowed_signers=frozenset({"allowlist/operator-1"}),
    )
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
    assert "signer" in decision.establishes.casefold()
    assert "without trusting the signer" in decision.does_not_establish.casefold()


def test_unallowlisted_signer_rejects() -> None:
    policy = AdmissionPolicy(
        allowed_signers=frozenset({"allowlist/other"}),
    )
    decision = verify_for_admission(
        _candidate(
            _signed(),
            proof_system_id="signed_receipt",
            proof_mode=ProofMode.SIGNED_RECEIPT,
            terminal_status=ProofTerminalStatus.SIGNED_ASSERTION_VERIFIED,
        ),
        policy=policy,
    )
    assert decision.admitted is False
    assert decision.reason_code == RejectionReason.UNALLOWLISTED_SIGNER.value


def test_direct_execution_admission_retains_class_claims() -> None:
    evidence = _direct()
    decision = verify_for_admission(
        _candidate(
            evidence,
            proof_system_id="groth16",
            public_input_cid=evidence.input_commitment,
            proof_mode=ProofMode.DIRECT_EXECUTION_PROOF,
            terminal_status=ProofTerminalStatus.PROVED,
            proof_object_cid=evidence.proof_cid,
        )
    )
    assert decision.admitted is True
    assert decision.evidence_class == "DirectExecutionProof"
    assert "declared program" in decision.establishes.casefold() or (
        "program/verifier" in decision.establishes.casefold()
    )


def test_lower_class_cannot_claim_direct_computation() -> None:
    decision = verify_for_admission(
        _candidate(
            _integrity(),
            metadata={"direct_computation_claim": True},
        )
    )
    assert decision.admitted is False
    assert decision.reason_code == RejectionReason.OVERCLAIM.value


def test_integrity_mismatch_rejects() -> None:
    decision = verify_for_admission(
        _candidate(
            _integrity(digest=_DIGEST),
            expected_digest=_DIGEST,
            observed_digest=_DIGEST_B,
        )
    )
    assert decision.admitted is False
    assert decision.reason_code == RejectionReason.INTEGRITY_MISMATCH.value


def test_incremental_seal_admission() -> None:
    evidence = _seal()
    decision = verify_for_admission(
        _candidate(
            evidence,
            proof_system_id="incremental_seal",
            proof_mode=ProofMode.THEOREM_CERTIFICATE,
            terminal_status=ProofTerminalStatus.PROVED,
            public_input_cid=_DIGEST,
            proof_object_cid=evidence.manifest_cid,
        )
    )
    assert decision.admitted is True
    assert decision.evidence_class == "IncrementalCommitSeal"
    assert "direct test execution" in decision.does_not_establish.casefold() or (
        "test execution" in decision.does_not_establish.casefold()
    )


def test_canonical_evidence_mapping_round_trip_admission() -> None:
    evidence = _integrity()
    decision = verify_for_admission(
        _candidate(evidence.to_canonical())
    )
    assert decision.admitted is True
    assert decision.evidence_class == "IntegrityCommitment"


def test_evidence_verifier_class_matches_module_function() -> None:
    candidate = _candidate(_integrity())
    via_class = EvidenceVerifier().verify(candidate)
    via_fn = verify_for_admission(candidate)
    assert via_class.admitted == via_fn.admitted
    assert via_class.evidence_class == via_fn.evidence_class
    assert (
        via_class.cache_admission_record.verification_digest
        == via_fn.cache_admission_record.verification_digest
    )


def test_rejected_decision_cannot_embed_admission_record() -> None:
    with pytest.raises(AdmissionError, match="must not carry"):
        AdmissionDecision(
            outcome=AdmissionOutcome.REJECTED,
            admitted=False,
            reason_code=RejectionReason.VERIFIER_FAILURE.value,
            evidence_class="IntegrityCommitment",
            establishes="x",
            does_not_establish="y",
            message="no",
            cache_admission_record=CacheAdmissionRecord(
                schema=ADMISSION_SCHEMA,
                proof_unit_id="u",
                evidence_class="IntegrityCommitment",
                proof_system_id="integrity",
                proof_object_cid=_DIGEST,
                public_input_cid=_DIGEST,
                verification_digest=_DIGEST_B,
                establishes="x",
                does_not_establish="y",
                logical_epoch=0,
            ),
            proof_unit_id="u",
            proof_system_id="integrity",
            public_input_cid=_DIGEST,
        )


def test_admitted_decision_requires_record() -> None:
    with pytest.raises(AdmissionError, match="require a CacheAdmissionRecord"):
        AdmissionDecision(
            outcome=AdmissionOutcome.ADMITTED,
            admitted=True,
            reason_code=None,
            evidence_class="IntegrityCommitment",
            establishes="x",
            does_not_establish="y",
            message="ok",
            cache_admission_record=None,
            proof_unit_id="u",
            proof_system_id="integrity",
            public_input_cid=_DIGEST,
        )


def test_status_class_mismatch_rejects() -> None:
    # Integrity evidence with "proved" status does not satisfy the class.
    decision = verify_for_admission(
        _candidate(
            _integrity(),
            terminal_status=ProofTerminalStatus.PROVED,
            required_for_seal=True,
        )
    )
    assert decision.admitted is False
    assert decision.reason_code == RejectionReason.STATUS_CLASS_MISMATCH.value


def test_decision_to_canonical_includes_subset() -> None:
    decision = verify_for_admission(_candidate(_integrity()))
    payload = decision.to_canonical()
    assert payload["evidence_subset"] == EVIDENCE_SUBSET
    assert payload["admitted"] is True
    assert payload["cache_admission_record"]["verified"] is True


def test_injected_verifier_success_path() -> None:
    calls: list[str] = []

    def ok(evidence, meta):
        calls.append(type(evidence).__name__)
        return True

    decision = verify_for_admission(
        _candidate(
            _direct(),
            proof_system_id="groth16",
            public_input_cid=_DIGEST,
            proof_mode=ProofMode.DIRECT_EXECUTION_PROOF,
            terminal_status=ProofTerminalStatus.PROVED,
            proof_object_cid=_DIGEST_C,
        ),
        verifier=ok,
    )
    assert decision.admitted is True
    assert calls == ["DirectExecutionProof"]
