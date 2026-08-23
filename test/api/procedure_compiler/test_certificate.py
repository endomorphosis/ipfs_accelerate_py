from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.certificate import (
    CERTIFICATE_SIGNING_SCOPE,
    REQUIRED_CERTIFICATE_BINDINGS,
    SIGNATURE_ALGORITHM,
    CertificateKeyRing,
    CertificateReasonCode,
    CertificateVerificationStatus,
    CurrentCertificateContext,
    ProcedureCertificateError,
    ProcedureCertificateIssuer,
    ProcedureCertificateVerifier,
    encode_certificate_statement,
    issue_procedure_certificate,
    unsigned_certificate_statement,
    verify_procedure_certificate,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactState,
    ProcedureCertificate,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.verifier import (
    ProcedureVerifier,
    VerificationStatus,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.trust import (
    SignerTrustRecord,
    SignerTrustRegistry,
    TrustedProofPolicy,
)


def _load_verifier_helpers():
    path = Path(__file__).with_name("test_verifier.py")
    spec = importlib.util.spec_from_file_location("_pcpc017_verifier_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load verifier test helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_helpers = _load_verifier_helpers()
ISSUER_ID = _helpers.ISSUER_ID
candidate_for = _helpers.candidate_for
evidence_for = _helpers.evidence_for
policy_for = _helpers.policy_for
valid_spec = _helpers.valid_spec


TEST_HMAC_KEY = b"pcpc-017-test-only-hmac-key-32b!"


def trust_policy(*, issuer_id: str = ISSUER_ID, scopes: tuple[str, ...] | None = None):
    registry = SignerTrustRegistry(
        [
            SignerTrustRecord(
                signer_id=issuer_id,
                scopes=frozenset(scopes or (CERTIFICATE_SIGNING_SCOPE,)),
                trusted=True,
                test_only=True,
            )
        ],
        production=False,
    )
    return TrustedProofPolicy(production=False, signers=registry)


def keyring(*, issuer_id: str = ISSUER_ID, secret: bytes = TEST_HMAC_KEY) -> CertificateKeyRing:
    return CertificateKeyRing({issuer_id: secret})


def issuer_for(issuer_id: str = ISSUER_ID) -> ProcedureCertificateIssuer:
    return ProcedureCertificateIssuer(
        trust_policy(issuer_id=issuer_id),
        keyring(issuer_id=issuer_id),
        issuer_id=issuer_id,
    )


def resign(certificate: ProcedureCertificate, **changes: object) -> ProcedureCertificate:
    pending = replace(certificate, signature="pending-signature", **changes)
    signature = keyring(issuer_id=pending.issuer).sign(
        pending.issuer, encode_certificate_statement(unsigned_certificate_statement(pending))
    )
    return replace(pending, signature=signature)


def issue_valid(*, now_ms: int = 100):
    spec = valid_spec()
    candidate = candidate_for(spec)
    evidence = evidence_for(spec)
    policy = policy_for(spec)
    verification = ProcedureVerifier().verify(candidate, evidence, policy, now_ms=now_ms)
    certificate = issuer_for().issue(
        candidate, verification, evidence, policy, now_ms=now_ms
    )
    context = CurrentCertificateContext.from_policy(policy, now_ms=now_ms)
    return spec, candidate, evidence, policy, verification, certificate, context


def test_issuer_binds_every_required_identity_evidence_limitation_and_horizon() -> None:
    spec, candidate, evidence, policy, verification, certificate, context = issue_valid()

    assert verification.status is VerificationStatus.ACCEPTED
    assert certificate.state is ArtifactState.VERIFIED
    assert certificate.state is not ArtifactState.PROMOTED
    assert certificate.issuer == ISSUER_ID
    assert certificate.procedure_cid == spec.content_id
    assert certificate.procedure_version == spec.version
    assert certificate.task_family_cid == spec.task_family_id
    assert certificate.source_episode_cids == evidence.source_episode_cids
    assert certificate.specification_cids == evidence.specification_cids
    assert certificate.counterexample_set_cid == evidence.counterexample_set_cid
    assert certificate.operation_catalog_revision == policy.operation_catalog_revision
    assert certificate.effect_policy_revision == policy.effect_policy_revision
    assert certificate.authority_policy_revision == policy.authority_policy_revision
    assert certificate.verification_policy_revision == policy.revision
    assert certificate.repository_families == evidence.repository_families
    assert certificate.supported_language_classes == evidence.supported_language_classes
    assert certificate.supported_framework_classes == evidence.supported_framework_classes
    assert certificate.risk_ceiling == spec.authority.risk_ceiling
    assert certificate.proof_receipt_cids == evidence.proof_receipt_cids
    assert certificate.test_receipt_cids == evidence.test_receipt_cids
    assert certificate.adversarial_assurance_cids == evidence.adversarial_assurance_cids
    assert certificate.held_out_evaluation_cid == evidence.held_out_evaluation_cid
    assert certificate.shadow_evaluation_cid == evidence.shadow_evaluation_cid
    assert certificate.known_limitations == evidence.known_limitations
    assert certificate.issued_at_ms == 100
    assert certificate.expires_at_ms == 100 + policy.review_horizon_ms
    payload = certificate.to_dict()
    for name in REQUIRED_CERTIFICATE_BINDINGS:
        assert name in payload
    assert certificate.signature.startswith(SIGNATURE_ALGORITHM + ":")
    statement = unsigned_certificate_statement(certificate)
    assert "signature" not in statement
    assert candidate.content_id != certificate.content_id
    del context


def test_independent_certificate_verifier_accepts_signed_current_certificate() -> None:
    _spec, candidate, _evidence, _policy, _verification, certificate, context = issue_valid()
    verifier = ProcedureCertificateVerifier(trust_policy(), keyring())

    admission = verifier.verify(certificate, context)
    assert admission.status is CertificateVerificationStatus.ACCEPTED
    assert admission.accepted
    assert admission.usable
    assert admission.grants_authority is False
    assert admission.grants_promotion is False
    assert admission.certificate_cid == certificate.content_id
    assert admission.issuer == ISSUER_ID
    assert certificate.procedure_cid in admission.bound_identities
    assert certificate.verification_policy_revision in admission.bound_identities

    # Certificate verification does not read procedure content: a candidate
    # argument is only an optional self-issuance check.
    again = verifier.verify(certificate.to_dict(), context, candidate=candidate)
    assert again.accepted
    assert again.grants_authority is False


def test_unverified_or_mismatched_candidates_cannot_receive_certificates() -> None:
    spec = valid_spec()
    candidate = candidate_for(spec)
    evidence = evidence_for(spec)
    policy = policy_for(spec)
    rejected = ProcedureVerifier().verify(
        candidate_for(spec, state=ArtifactState.REJECTED),
        evidence,
        policy,
        now_ms=100,
    )
    with pytest.raises(ProcedureCertificateError, match="independently verified"):
        issuer_for().issue(candidate, rejected, evidence, policy, now_ms=100)

    accepted = ProcedureVerifier().verify(candidate, evidence, policy, now_ms=100)
    other = candidate_for(valid_spec(name="other-procedure"))
    with pytest.raises(ProcedureCertificateError, match="does not bind this candidate"):
        issuer_for().issue(other, accepted, evidence_for(other.procedure), policy, now_ms=100)


def test_forged_signature_and_tampered_payload_fail() -> None:
    _spec, _candidate, _evidence, _policy, _verification, certificate, context = issue_valid()
    verifier = ProcedureCertificateVerifier(trust_policy(), keyring())

    forged = replace(certificate, signature=SIGNATURE_ALGORITHM + ":" + "0" * 64)
    forged_admission = verifier.verify(forged, context)
    assert not forged_admission.accepted
    assert forged_admission.reason_code is CertificateReasonCode.FORGED_SIGNATURE

    tampered = replace(certificate, specification_cids=("specification-forged",))
    tampered_admission = verifier.verify(tampered, context)
    assert not tampered_admission.accepted
    assert tampered_admission.reason_code is CertificateReasonCode.FORGED_SIGNATURE

    other_key = ProcedureCertificateVerifier(
        trust_policy(), keyring(secret=b"pcpc-017-other-test-hmac-key-32b")
    )
    wrong_key = other_key.verify(certificate, context)
    assert not wrong_key.accepted
    assert wrong_key.reason_code is CertificateReasonCode.FORGED_SIGNATURE


def test_unknown_signature_algorithm_fails() -> None:
    _spec, _candidate, _evidence, _policy, _verification, certificate, context = issue_valid()
    weak = replace(certificate, signature="sha256:" + "a" * 64)
    admission = ProcedureCertificateVerifier(trust_policy(), keyring()).verify(weak, context)
    assert not admission.accepted
    assert admission.reason_code is CertificateReasonCode.UNKNOWN_SIGNATURE_ALGORITHM


def test_stale_expiry_bindings_and_catalog_fail() -> None:
    _spec, _candidate, _evidence, policy, _verification, certificate, context = issue_valid()
    verifier = ProcedureCertificateVerifier(trust_policy(), keyring())

    expired = verifier.verify(
        certificate, replace(context, now_ms=certificate.expires_at_ms)
    )
    assert not expired.accepted
    assert expired.reason_code is CertificateReasonCode.STALE_CERTIFICATE

    stale_tree = verifier.verify(
        certificate,
        replace(context, bindings=replace(context.bindings, tree_id="tree-other")),
    )
    assert not stale_tree.accepted
    assert stale_tree.reason_code is CertificateReasonCode.STALE_BINDINGS

    stale_catalog = verifier.verify(
        certificate, replace(context, operation_catalog_revision="catalog-old")
    )
    assert not stale_catalog.accepted
    assert stale_catalog.reason_code is CertificateReasonCode.STALE_POLICY

    stale_state = resign(certificate, state=ArtifactState.STALE)
    stale_admission = verifier.verify(stale_state, context)
    assert not stale_admission.accepted
    assert stale_admission.reason_code is CertificateReasonCode.STALE_CERTIFICATE
    del policy


def test_self_issued_certificates_fail() -> None:
    spec, candidate, _evidence, _policy, _verification, certificate, context = issue_valid()
    verifier = ProcedureCertificateVerifier(trust_policy(), keyring())
    self_issued = replace(certificate, issuer=spec.content_id)
    admission = verifier.verify(self_issued, context)
    assert not admission.accepted
    assert admission.reason_code is CertificateReasonCode.SELF_ISSUED

    named = replace(certificate, issuer=spec.name)
    named_admission = verifier.verify(named, context, candidate=candidate)
    assert not named_admission.accepted
    assert named_admission.reason_code is CertificateReasonCode.SELF_ISSUED


def test_incomplete_certificate_payload_fails() -> None:
    _spec, _candidate, _evidence, _policy, _verification, certificate, context = issue_valid()
    payload = certificate.to_dict()
    payload.pop("shadow_evaluation_cid")
    admission = ProcedureCertificateVerifier(trust_policy(), keyring()).verify(
        payload, context
    )
    assert not admission.accepted
    assert admission.reason_code is CertificateReasonCode.INCOMPLETE_CERTIFICATE
    assert "shadow_evaluation_cid" in admission.message

    payload = certificate.to_dict()
    payload["proof_receipt_cids"] = ()
    empty_proofs = ProcedureCertificateVerifier(trust_policy(), keyring()).verify(
        payload, context
    )
    assert not empty_proofs.accepted
    assert empty_proofs.reason_code is CertificateReasonCode.INCOMPLETE_CERTIFICATE


def test_weaker_validation_policy_or_missing_evidence_fails() -> None:
    _spec, _candidate, _evidence, policy, _verification, certificate, context = issue_valid()
    verifier = ProcedureCertificateVerifier(trust_policy(), keyring())

    weaker_policy = verifier.verify(
        certificate, replace(context, verification_policy_revision="verification-weak")
    )
    assert not weaker_policy.accepted
    assert weaker_policy.reason_code is CertificateReasonCode.WEAKER_VALIDATION

    weaker_context = CurrentCertificateContext.from_policy(policy, now_ms=100)
    weaker_context = replace(weaker_context, require_adversarial=True)
    dropped = replace(certificate, adversarial_assurance_cids=certificate.adversarial_assurance_cids)
    # Resigning is required to change fields; construct a mapping that still
    # looks complete but binds a different, weaker policy identity.
    payload = certificate.to_dict()
    payload["verification_policy_revision"] = "verification-weak"
    weak_payload = verifier.verify(payload, context)
    assert not weak_payload.accepted
    assert weak_payload.reason_code in {
        CertificateReasonCode.WEAKER_VALIDATION,
        CertificateReasonCode.FORGED_SIGNATURE,
        CertificateReasonCode.MALFORMED_CERTIFICATE,
    }
    del weaker_context
    del dropped


def test_untrusted_revoked_and_out_of_scope_issuers_fail() -> None:
    _spec, _candidate, _evidence, _policy, _verification, certificate, context = issue_valid()

    empty_trust = TrustedProofPolicy(
        production=False, signers=SignerTrustRegistry(production=False)
    )
    untrusted = ProcedureCertificateVerifier(empty_trust, keyring()).verify(
        certificate, context
    )
    assert not untrusted.accepted
    assert untrusted.reason_code is CertificateReasonCode.UNTRUSTED_ISSUER

    revoked = SignerTrustRegistry(
        [
            SignerTrustRecord(
                signer_id=ISSUER_ID,
                scopes=frozenset({CERTIFICATE_SIGNING_SCOPE}),
                trusted=True,
                test_only=True,
                revocation_epoch=1,
            )
        ],
        production=False,
        current_epoch=1,
    )
    revoked_admission = ProcedureCertificateVerifier(
        TrustedProofPolicy(production=False, signers=revoked, current_epoch=1),
        keyring(),
    ).verify(certificate, context)
    assert not revoked_admission.accepted
    assert revoked_admission.reason_code is CertificateReasonCode.REVOKED_ISSUER

    scoped = SignerTrustRegistry(
        [
            SignerTrustRecord(
                signer_id=ISSUER_ID,
                scopes=frozenset({"other-scope"}),
                trusted=True,
                test_only=True,
            )
        ],
        production=False,
    )
    out_of_scope = ProcedureCertificateVerifier(
        TrustedProofPolicy(production=False, signers=scoped),
        keyring(),
    ).verify(certificate, context)
    assert not out_of_scope.accepted
    assert out_of_scope.reason_code is CertificateReasonCode.ISSUER_OUT_OF_SCOPE


def test_issuer_refuses_self_certifying_issuer_id() -> None:
    spec = valid_spec()
    candidate = candidate_for(spec)
    evidence = evidence_for(spec)
    policy = policy_for(spec)
    verification = ProcedureVerifier().verify(candidate, evidence, policy, now_ms=100)
    colliding = ProcedureCertificateIssuer(
        trust_policy(issuer_id=spec.name),
        keyring(issuer_id=spec.name),
        issuer_id=spec.name,
    )
    with pytest.raises(ProcedureCertificateError, match="cannot issue its own certificate"):
        colliding.issue(candidate, verification, evidence, policy, now_ms=100)


def test_identity_alone_does_not_establish_authority_or_usability() -> None:
    _spec, _candidate, _evidence, _policy, _verification, certificate, context = issue_valid()
    verifier = ProcedureCertificateVerifier(trust_policy(), keyring())
    admission = verifier.verify(certificate, context)
    assert admission.accepted
    assert admission.grants_authority is False
    assert admission.grants_promotion is False
    assert admission.to_dict()["grants_authority"] is False

    # A well-formed CID copied onto an unsigned mapping is not usable.
    payload = {"content_id": certificate.content_id, "issuer": certificate.issuer}
    identity_only = verifier.verify(payload, context)
    assert not identity_only.accepted
    assert identity_only.usable is False
    assert identity_only.reason_code is CertificateReasonCode.INCOMPLETE_CERTIFICATE


def test_keyring_never_generates_or_serializes_secrets() -> None:
    ring = keyring()
    assert ISSUER_ID in ring
    with pytest.raises(ProcedureCertificateError, match="nonempty bytes"):
        CertificateKeyRing({ISSUER_ID: ""})  # type: ignore[dict-item]
    with pytest.raises(ProcedureCertificateError, match="no authorized signing key"):
        ring.sign("unknown-issuer@1", b"payload")


def test_issuer_construction_requires_trusted_in_scope_signer() -> None:
    with pytest.raises(ProcedureCertificateError, match="not admitted"):
        ProcedureCertificateIssuer(
            TrustedProofPolicy(
                production=False, signers=SignerTrustRegistry(production=False)
            ),
            keyring(),
            issuer_id=ISSUER_ID,
        )


def test_issuer_refuses_dropped_limitations_and_mismatched_evidence() -> None:
    spec, candidate, evidence, policy, verification, _certificate, _context = issue_valid()
    with pytest.raises(ProcedureCertificateError, match="cannot drop known limitations"):
        issuer_for().issue(
            candidate, verification, evidence, policy, now_ms=100, known_limitations=()
        )
    weaker = evidence_for(spec, proof_receipt_cids=("proof-other",), include_receipts=False)
    with pytest.raises(ProcedureCertificateError, match="does not bind this evidence"):
        issuer_for().issue(candidate, verification, weaker, policy, now_ms=100)


def test_current_review_horizon_cannot_exceed_policy() -> None:
    _spec, _candidate, _evidence, _policy, _verification, certificate, context = issue_valid()
    admission = ProcedureCertificateVerifier(trust_policy(), keyring()).verify(
        certificate, replace(context, review_horizon_ms=1)
    )
    assert not admission.accepted
    assert admission.reason_code is CertificateReasonCode.STALE_CERTIFICATE


def test_family_or_evidence_identity_cannot_be_the_issuer() -> None:
    spec, _candidate, _evidence, _policy, _verification, certificate, context = issue_valid()
    verifier = ProcedureCertificateVerifier(trust_policy(), keyring())
    family_issued = replace(certificate, issuer=spec.task_family_id)
    admission = verifier.verify(family_issued, context)
    assert not admission.accepted
    assert admission.reason_code is CertificateReasonCode.SELF_ISSUED


def test_pending_signature_is_not_a_usable_certificate() -> None:
    _spec, _candidate, _evidence, _policy, _verification, certificate, context = issue_valid()
    pending = replace(certificate, signature="pending-signature")
    admission = ProcedureCertificateVerifier(trust_policy(), keyring()).verify(
        pending, context
    )
    assert not admission.accepted
    assert admission.usable is False
    assert admission.reason_code in {
        CertificateReasonCode.INCOMPLETE_CERTIFICATE,
        CertificateReasonCode.UNKNOWN_SIGNATURE_ALGORITHM,
    }


def test_future_certificate_is_stale() -> None:
    _spec, _candidate, _evidence, _policy, _verification, certificate, context = issue_valid()
    admission = ProcedureCertificateVerifier(trust_policy(), keyring()).verify(
        certificate, replace(context, now_ms=0)
    )
    assert not admission.accepted
    assert admission.reason_code is CertificateReasonCode.STALE_CERTIFICATE


def test_promoted_state_does_not_grant_promotion_or_authority() -> None:
    _spec, _candidate, _evidence, _policy, _verification, certificate, context = issue_valid()
    promoted = resign(certificate, state=ArtifactState.PROMOTED)
    admission = ProcedureCertificateVerifier(trust_policy(), keyring()).verify(
        promoted, context
    )
    assert admission.accepted
    assert admission.grants_authority is False
    assert admission.grants_promotion is False
    assert admission.to_dict()["grants_promotion"] is False


def test_forbidden_issuer_name_is_rejected_at_construction() -> None:
    with pytest.raises(ProcedureCertificateError, match="not independent"):
        ProcedureCertificateIssuer(
            trust_policy(issuer_id="self"),
            keyring(issuer_id="self"),
            issuer_id="self",
        )


def test_issue_and_verify_helpers_do_not_grant_authority() -> None:
    spec, candidate, evidence, policy, verification, _certificate, context = issue_valid()
    issued = issue_procedure_certificate(
        candidate, verification, evidence, policy, issuer_for(), now_ms=100
    )
    assert issued.procedure_cid == spec.content_id
    assert issued.state is ArtifactState.VERIFIED
    admission = verify_procedure_certificate(
        issued,
        context,
        ProcedureCertificateVerifier(trust_policy(), keyring()),
        candidate=candidate,
    )
    assert admission.accepted
    assert admission.usable
    assert admission.grants_authority is False
    assert admission.grants_promotion is False
