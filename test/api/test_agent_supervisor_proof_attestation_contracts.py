from __future__ import annotations

import json
import pickle
from collections.abc import Mapping

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
    ContractValidationError,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
    assess_assurance,
)
from ipfs_accelerate_py.agent_supervisor.proof_attestation import (
    AttestationBackendMode,
    AttestationGate,
    AttestationTrust,
    AttestationValidationError,
    AttestationVerification,
    PrivateAttestationWitness,
    ReceiptAttestationEnvelope,
    ReceiptAttestationStatement,
    WitnessDisclosureError,
    attestation_satisfies_gate,
    build_receipt_attestation_statement,
    create_attestation_envelope,
    prepare_receipt_attestation,
    public_artifact_contains,
    public_attestation_artifact,
    record_attestation_verification,
)


def _kernel_evidence(
    obligation_id: str,
    *,
    freshness: EvidenceFreshness = EvidenceFreshness.CURRENT,
    simulated: bool = False,
) -> ProofEvidence:
    return ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:lean-checked-proof",
        subject_id=obligation_id,
        verifier_id="kernel:lean@4.19",
        freshness=freshness,
        independent=True,
        simulated=simulated,
    )


def _receipt(
    *,
    evidence: tuple[ProofEvidence, ...] | None = None,
    verdict: ProofVerdict = ProofVerdict.PROVED,
    freshness: EvidenceFreshness = EvidenceFreshness.CURRENT,
) -> ProofReceipt:
    obligation_id = "obligation:lease-fencing@2"
    return ProofReceipt(
        obligation_id=obligation_id,
        plan_id="plan:formal-v1",
        attempt_id="attempt:kernel",
        repository_id="repo:complaint-generator",
        repository_tree_id="git-tree:abc123",
        ast_scope_ids=("scope:lease",),
        premise_ids=("premise:fencing-order",),
        translator_id="translator:python-to-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id="kernel:lean@4.19",
        toolchain_id="toolchain:nix-lock",
        policy_id="policy:proof-production@3",
        resource_budget=ResourceBudget(
            wall_time_ms=30_000,
            memory_bytes=512 * 1024 * 1024,
            max_processes=2,
            network_allowed=False,
        ),
        verdict=verdict,
        evidence=(
            evidence
            if evidence is not None
            else (_kernel_evidence(obligation_id, freshness=freshness),)
        ),
        freshness=freshness,
    )


def _statement(receipt: ProofReceipt | None = None) -> ReceiptAttestationStatement:
    return build_receipt_attestation_statement(
        receipt or _receipt(),
        circuit_id="circuit:receipt-binding@1:sha256-cafe",
        backend_id="backend:provekit@0.1",
        verification_key_id="vk:receipt-binding@1:sha256-beef",
    )


def _request(
    receipt: ProofReceipt | None = None,
    *,
    secret: str = "premise-private-9f2937",
):
    return prepare_receipt_attestation(
        receipt or _receipt(),
        circuit_id="circuit:receipt-binding@1:sha256-cafe",
        backend_id="backend:provekit@0.1",
        verification_key_id="vk:receipt-binding@1:sha256-beef",
        witness=PrivateAttestationWitness(
            {
                "private_premise": secret,
                "kernel_transcript": b"private-kernel-transcript",
            }
        ),
    )


def _envelope(
    *,
    mode: AttestationBackendMode = AttestationBackendMode.CRYPTOGRAPHIC,
) -> ReceiptAttestationEnvelope:
    return create_attestation_envelope(
        _request(),
        backend_mode=mode,
        proof_artifact_id="artifact:zk-proof",
        proof_digest="sha256:proof",
        prover_id="prover:provekit-worker",
    )


def test_public_statement_binds_all_required_identities_and_is_canonical() -> None:
    receipt = _receipt()
    statement = _statement(receipt)

    assert statement.public_inputs == {
        "repository_tree_id": receipt.repository_tree_id,
        "obligation_id": receipt.obligation_id,
        "policy_id": receipt.policy_id,
        "kernel_id": receipt.kernel_id,
        "receipt_id": receipt.receipt_id,
        "circuit_id": "circuit:receipt-binding@1:sha256-cafe",
        "backend_id": "backend:provekit@0.1",
        "verification_key_id": "vk:receipt-binding@1:sha256-beef",
    }
    assert statement.statement_id.startswith("baguqeera")
    assert statement.public_input_digest.startswith("baguqeera")
    assert ReceiptAttestationStatement.from_dict(statement.to_dict()) == statement
    assert json.loads(statement.to_json())["receipt_id"] == receipt.receipt_id


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("repository_tree_id", "git-tree:different"),
        ("obligation_id", "obligation:different"),
        ("policy_id", "policy:different"),
        ("kernel_id", "kernel:different"),
        ("receipt_id", "receipt:different"),
        ("circuit_id", "circuit:different"),
        ("backend_id", "backend:different"),
        ("verification_key_id", "vk:different"),
    ),
)
def test_every_public_identity_changes_statement_identity(
    field_name: str, replacement: str
) -> None:
    baseline = _statement()
    values = baseline.public_inputs
    values[field_name] = replacement
    changed = ReceiptAttestationStatement(**values)

    assert changed.statement_id != baseline.statement_id
    assert changed.public_input_digest != baseline.public_input_digest


def test_statement_rejects_forged_identity_and_public_input_digest() -> None:
    statement = _statement()
    forged = statement.to_public_artifact()
    forged["receipt_id"] = "receipt:forged"
    with pytest.raises(AttestationValidationError, match="statement identity"):
        ReceiptAttestationStatement.from_dict(forged)

    forged = statement.to_public_artifact()
    forged["public_input_digest"] = "digest:forged"
    with pytest.raises(AttestationValidationError, match="public-input digest"):
        ReceiptAttestationStatement.from_dict(forged)


@pytest.mark.parametrize(
    "receipt",
    (
        _receipt(evidence=()),
        _receipt(
            evidence=(
                ProofEvidence(
                    kind=EvidenceKind.SMT_CANDIDATE,
                    authority=EvidenceAuthority.SMT,
                    verdict=EvidenceVerdict.ACCEPTED,
                    artifact_id="artifact:smt",
                    subject_id="obligation:lease-fencing@2",
                    verifier_id="solver:z3",
                    independent=True,
                ),
            )
        ),
        _receipt(verdict=ProofVerdict.INCONCLUSIVE),
        _receipt(freshness=EvidenceFreshness.STALE),
    ),
)
def test_attestation_is_unavailable_without_current_kernel_verified_receipt(
    receipt: ProofReceipt,
) -> None:
    assert not receipt.can_be_attested
    with pytest.raises(ContractValidationError, match="attestation requires"):
        _statement(receipt)
    with pytest.raises(ContractValidationError, match="attestation requires"):
        _request(receipt)


def test_statement_builder_requires_an_existing_receipt_object() -> None:
    with pytest.raises(AttestationValidationError, match="ProofReceipt"):
        build_receipt_attestation_statement(  # type: ignore[arg-type]
            "receipt:id-only",
            circuit_id="circuit:id",
            backend_id="backend:id",
            verification_key_id="vk:id",
        )


def test_private_witness_is_redacted_non_mapping_and_non_serializable() -> None:
    secret = "never-log-private-premise"
    witness = PrivateAttestationWitness({"hidden_field_name": secret})

    assert secret not in repr(witness)
    assert "hidden_field_name" not in repr(witness)
    assert not isinstance(witness, Mapping)
    assert witness.redacted() == {"private_witness_redacted": True}
    with pytest.raises(WitnessDisclosureError):
        witness.to_dict()
    with pytest.raises(WitnessDisclosureError):
        pickle.dumps(witness)
    with pytest.raises(WitnessDisclosureError):
        public_attestation_artifact(witness)


def test_proving_callback_can_use_read_only_witness_without_publication() -> None:
    request = _request(secret="callback-only-secret")

    def consume(values: Mapping[str, object]) -> tuple[str, bool]:
        with pytest.raises(TypeError):
            values["extra"] = "forbidden"  # type: ignore[index]
        return str(values["private_premise"]), "kernel_transcript" in values

    assert request.use_witness(consume) == ("callback-only-secret", True)


def test_witness_fields_are_absent_from_logs_context_and_public_artifacts() -> None:
    secret = "s3cr3t-witness-value-unique"
    request = _request(secret=secret)

    for artifact in (
        request.to_dict(),
        request.to_log_record(),
        request.to_context_capsule(),
        public_attestation_artifact({"request": request}),
    ):
        encoded = json.dumps(artifact, sort_keys=True)
        assert secret not in encoded
        assert "private_premise" not in encoded
        assert "kernel_transcript" not in encoded
        assert not public_artifact_contains(artifact, secret)

    assert secret not in repr(request)
    with pytest.raises(WitnessDisclosureError, match="cannot be cached"):
        request.to_cache_record()
    with pytest.raises(WitnessDisclosureError):
        pickle.dumps(request)


def test_generated_envelope_is_non_authoritative_until_independent_verification() -> None:
    envelope = _envelope()
    payload = envelope.to_dict()

    assert envelope.authoritative is False
    assert envelope.trust is AttestationTrust.NON_AUTHORITATIVE
    assert payload["authoritative"] is False
    assert payload["non_authoritative_reason"] == "independent_verification_required"
    assert ReceiptAttestationEnvelope.from_dict(payload) == envelope

    forged = dict(payload)
    forged["authoritative"] = True
    with pytest.raises(AttestationValidationError, match="cannot assert authority"):
        ReceiptAttestationEnvelope.from_dict(forged)


def test_simulated_zkp_is_explicitly_non_authoritative_and_gate_limited() -> None:
    envelope = _envelope(mode=AttestationBackendMode.SIMULATED)
    verification = record_attestation_verification(
        envelope,
        verified=True,
        verifier_id="verifier:simulated-v0.1",
    )
    payload = verification.to_dict()

    assert envelope.simulated is True
    assert envelope.non_authoritative_reason == "simulated_zkp_is_non_authoritative"
    assert verification.verified is True
    assert verification.simulated is True
    assert verification.authoritative is False
    assert verification.trust is AttestationTrust.NON_AUTHORITATIVE
    assert verification.authoritative_assurance is AssuranceLevel.UNVERIFIED
    assert verification.satisfies_gate(AttestationGate.SERIALIZATION)
    assert verification.satisfies_gate(AttestationGate.TEST)
    assert not verification.satisfies_production_gate()
    assert not verification.satisfies_completion_gate()
    assert payload["simulated"] is True
    assert payload["authoritative"] is False
    assert payload["trust"] == "non_authoritative"

    evidence = verification.to_evidence()
    assert evidence.simulated is True
    assert evidence.authority is EvidenceAuthority.ATTESTATION_VERIFIER
    assert evidence.subject_id == envelope.statement.receipt_id


def test_simulated_backend_identity_cannot_claim_cryptographic_mode() -> None:
    request = prepare_receipt_attestation(
        _receipt(),
        circuit_id="circuit:receipt-binding@1",
        backend_id="backend:simulated@0.1",
        verification_key_id="vk:simulated",
        witness=PrivateAttestationWitness({"private_premise": "secret"}),
    )

    with pytest.raises(AttestationValidationError, match="simulated backend"):
        create_attestation_envelope(
            request,
            backend_mode=AttestationBackendMode.CRYPTOGRAPHIC,
            proof_artifact_id="artifact:fake",
            proof_digest="sha256:fake",
        )


def test_real_independent_verification_can_satisfy_attestation_gates() -> None:
    envelope = _envelope()
    verification = record_attestation_verification(
        envelope,
        verified=True,
        verifier_id="verifier:provekit-production@1",
        independent=True,
    )

    assert verification.authoritative is True
    assert verification.trust is AttestationTrust.AUTHORITATIVE
    assert verification.authoritative_assurance is AssuranceLevel.ATTESTED
    assert verification.satisfies_production_gate()
    assert verification.satisfies_completion_gate()
    assert attestation_satisfies_gate(
        verification, AttestationGate.COMPLETION
    )
    assert AttestationVerification.from_dict(verification.to_dict()) == verification


@pytest.mark.parametrize(
    ("verified", "independent"),
    ((False, True), (True, False), (False, False)),
)
def test_failed_or_non_independent_verification_is_not_authoritative(
    verified: bool, independent: bool
) -> None:
    result = record_attestation_verification(
        _envelope(),
        verified=verified,
        verifier_id="verifier:worker",
        independent=independent,
    )

    assert result.authoritative is False
    assert not result.satisfies_production_gate()
    assert not result.satisfies_completion_gate()


def test_verified_attestation_evidence_upgrades_only_the_bound_kernel_receipt() -> None:
    receipt = _receipt()
    request = prepare_receipt_attestation(
        receipt,
        circuit_id="circuit:receipt-binding@1:sha256-cafe",
        backend_id="backend:provekit@0.1",
        verification_key_id="vk:receipt-binding@1:sha256-beef",
        witness=PrivateAttestationWitness({"private_premise": "secret"}),
    )
    verification = record_attestation_verification(
        create_attestation_envelope(
            request,
            backend_mode=AttestationBackendMode.CRYPTOGRAPHIC,
            proof_artifact_id="artifact:proof",
            proof_digest="sha256:proof",
        ),
        verified=True,
        verifier_id="verifier:production",
    )
    attestation = verification.to_evidence()
    kernel = _kernel_evidence(receipt.obligation_id)

    assessment = assess_assurance(
        (kernel, attestation),
        obligation_id=receipt.obligation_id,
        kernel_id=receipt.kernel_id,
        kernel_receipt_id=receipt.receipt_id,
    )
    assert assessment.level is AssuranceLevel.ATTESTED

    wrong_receipt = assess_assurance(
        (kernel, attestation),
        obligation_id=receipt.obligation_id,
        kernel_id=receipt.kernel_id,
        kernel_receipt_id="receipt:different",
    )
    assert wrong_receipt.level is AssuranceLevel.KERNEL_VERIFIED


def test_public_artifacts_expose_no_generic_metadata_or_witness_channel() -> None:
    assert "metadata" not in _statement().to_dict()
    assert "metadata" not in _envelope().to_dict()
    assert "witness" not in json.dumps(_envelope().to_public_artifact()).lower()
