"""Independent procedure certificate issuance and verification (PCPC-017).

``ProcedureCertificateIssuer`` signs a certificate only after independent
verification accepted the candidate.  ``ProcedureCertificateVerifier``
rechecks signature, freshness, completeness, issuer independence, and
current policy/evidence strength without reading procedure bodies.

Identity, including a well-formed CID, never grants authority, promotion,
or usability.  The issuer never promotes.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from ..proof.formal_verification_contracts import canonical_json_bytes
from ..proof.incremental_sealing.trust import (
    SignerTrustRegistry,
    TrustedProofPolicy,
)
from .contracts import (
    ArtifactBindings,
    ArtifactState,
    ProcedureCandidate,
    ProcedureCertificate,
    ProcedureContractError,
    _enum,
    _identifier,
    _nested,
    _nonnegative_int,
    _strings,
    _text,
)
from .verifier import (
    FORBIDDEN_SELF_PRODUCERS,
    REQUIRED_EVIDENCE_KINDS,
    REQUIRED_VERIFICATION_LAYERS,
    IndependentEvidence,
    ProcedureVerification,
    VerificationPolicy,
    VerificationStatus,
    _self_identities,
)


ISSUER_REVISION: Final[str] = "ProcedureCertificateIssuer@1"
CERTIFICATE_VERIFIER_REVISION: Final[str] = "ProcedureCertificateVerifier@1"
CERTIFICATE_SIGNING_SCOPE: Final[str] = "procedure-certificate"
SIGNATURE_ALGORITHM: Final[str] = "hmac-sha256"
PENDING_SIGNATURE: Final[str] = "pending-signature"
REQUIRED_CERTIFICATE_BINDINGS: Final[tuple[str, ...]] = (
    "procedure_cid",
    "procedure_version",
    "task_family_cid",
    "source_episode_cids",
    "specification_cids",
    "counterexample_set_cid",
    "operation_catalog_revision",
    "effect_policy_revision",
    "authority_policy_revision",
    "verification_policy_revision",
    "repository_families",
    "supported_language_classes",
    "supported_framework_classes",
    "risk_ceiling",
    "proof_receipt_cids",
    "test_receipt_cids",
    "adversarial_assurance_cids",
    "held_out_evaluation_cid",
    "shadow_evaluation_cid",
    "known_limitations",
    "issuer",
    "signature",
    "issued_at_ms",
    "expires_at_ms",
)
_SEQUENCE_BINDINGS: Final[frozenset[str]] = frozenset(
    {
        "source_episode_cids",
        "specification_cids",
        "repository_families",
        "supported_language_classes",
        "supported_framework_classes",
        "proof_receipt_cids",
        "test_receipt_cids",
        "adversarial_assurance_cids",
        "known_limitations",
    }
)
_NONEMPTY_SEQUENCE_BINDINGS: Final[frozenset[str]] = _SEQUENCE_BINDINGS - {"known_limitations"}


class ProcedureCertificateError(ProcedureContractError):
    """Certificate issuance or verification failed closed."""


class CertificateReasonCode(str, Enum):
    ACCEPTED = "accepted"
    MALFORMED_CERTIFICATE = "malformed-certificate"
    INCOMPLETE_CERTIFICATE = "incomplete-certificate"
    FORGED_SIGNATURE = "forged-signature"
    UNKNOWN_SIGNATURE_ALGORITHM = "unknown-signature-algorithm"
    UNTRUSTED_ISSUER = "untrusted-issuer"
    SELF_ISSUED = "self-issued"
    STALE_CERTIFICATE = "stale-certificate"
    STALE_BINDINGS = "stale-bindings"
    STALE_POLICY = "stale-policy"
    STALE_EVIDENCE = "stale-evidence"
    WEAKER_VALIDATION = "weaker-validation"
    MISSING_VERIFICATION = "missing-verification"
    VERIFICATION_MISMATCH = "verification-mismatch"
    ISSUER_OUT_OF_SCOPE = "issuer-out-of-scope"
    REVOKED_ISSUER = "revoked-issuer"
    IDENTITY_IS_NOT_AUTHORITY = "identity-is-not-authority"
    PROMOTION_FORBIDDEN = "promotion-forbidden"


class CertificateVerificationStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


def _bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ProcedureCertificateError("{} must be a boolean".format(field_name))
    return value


def _signer_registry(trust: TrustedProofPolicy | SignerTrustRegistry) -> SignerTrustRegistry:
    if isinstance(trust, TrustedProofPolicy):
        registry = trust.signers
        if not isinstance(registry, SignerTrustRegistry):
            raise ProcedureCertificateError("trusted-proof policy has no signer registry")
        return registry
    if isinstance(trust, SignerTrustRegistry):
        return trust
    raise ProcedureCertificateError("trust policy must be TrustedProofPolicy or SignerTrustRegistry")


def unsigned_certificate_statement(
    certificate: ProcedureCertificate | Mapping[str, Any],
) -> dict[str, Any]:
    """Return the canonical certificate payload excluding the signature MAC."""

    if isinstance(certificate, ProcedureCertificate):
        payload = dict(certificate.to_dict())
    elif isinstance(certificate, Mapping):
        payload = dict(certificate)
    else:
        raise ProcedureCertificateError("certificate statement requires a certificate or mapping")
    payload.pop("signature", None)
    payload.pop("content_id", None)
    return payload


def encode_certificate_statement(statement: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(statement))


class CertificateKeyRing:
    """In-memory HMAC keys for allowlisted issuers.

    Key bytes never appear on certificate artifacts.  Production callers inject
    existing authorized material; this type never generates or downloads keys.
    """

    def __init__(self, keys: Mapping[str, bytes] | None = None) -> None:
        material: dict[str, bytes] = {}
        for issuer_id, secret in dict(keys or {}).items():
            identity = _identifier(issuer_id, "issuer_id")
            if not isinstance(secret, (bytes, bytearray)) or not secret:
                raise ProcedureCertificateError("issuer key material must be nonempty bytes")
            material[identity] = bytes(secret)
        self._keys = material

    def __contains__(self, issuer_id: object) -> bool:
        return isinstance(issuer_id, str) and issuer_id in self._keys

    def sign(self, issuer_id: str, payload: bytes) -> str:
        issuer_id = _identifier(issuer_id, "issuer_id")
        secret = self._keys.get(issuer_id)
        if secret is None:
            raise ProcedureCertificateError("issuer has no authorized signing key")
        if not isinstance(payload, (bytes, bytearray)):
            raise ProcedureCertificateError("signed payload must be bytes")
        digest = hmac.new(secret, bytes(payload), hashlib.sha256).hexdigest()
        return "{}:{}".format(SIGNATURE_ALGORITHM, digest)

    def verify(self, issuer_id: str, payload: bytes, signature: str) -> bool:
        try:
            expected = self.sign(issuer_id, payload)
        except ProcedureCertificateError:
            return False
        if not isinstance(signature, str) or not signature:
            return False
        return hmac.compare_digest(expected, signature)


@dataclass(frozen=True)
class CurrentCertificateContext:
    """Current-tree identities a certificate must still match."""

    bindings: ArtifactBindings
    operation_catalog_revision: str
    effect_policy_revision: str
    authority_policy_revision: str
    verification_policy_revision: str
    now_ms: int
    required_test_contracts: tuple[str, ...] = ()
    required_proof_contracts: tuple[str, ...] = ()
    require_adversarial: bool = True
    require_held_out: bool = True
    require_shadow: bool = True
    required_evidence_kinds: tuple[str, ...] = REQUIRED_EVIDENCE_KINDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings"))
        for name in (
            "operation_catalog_revision",
            "effect_policy_revision",
            "authority_policy_revision",
            "verification_policy_revision",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "now_ms", _nonnegative_int(self.now_ms, "now_ms"))
        object.__setattr__(
            self,
            "required_test_contracts",
            _strings(self.required_test_contracts, "required_test_contracts", identifiers=True),
        )
        object.__setattr__(
            self,
            "required_proof_contracts",
            _strings(self.required_proof_contracts, "required_proof_contracts", identifiers=True),
        )
        for name in ("require_adversarial", "require_held_out", "require_shadow"):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        kinds = _strings(
            self.required_evidence_kinds,
            "required_evidence_kinds",
            identifiers=True,
            required=True,
        )
        if set(kinds) != set(REQUIRED_EVIDENCE_KINDS):
            raise ProcedureCertificateError("current context omitted a required evidence kind")
        object.__setattr__(self, "required_evidence_kinds", tuple(REQUIRED_EVIDENCE_KINDS))
        if self.authority_policy_revision != self.bindings.policy_revision:
            raise ProcedureCertificateError("authority policy is not exact-binding current")

    @classmethod
    def from_policy(
        cls,
        policy: VerificationPolicy,
        *,
        now_ms: int,
    ) -> CurrentCertificateContext:
        return cls(
            bindings=policy.bindings,
            operation_catalog_revision=policy.operation_catalog_revision,
            effect_policy_revision=policy.effect_policy_revision,
            authority_policy_revision=policy.authority_policy_revision,
            verification_policy_revision=policy.revision,
            now_ms=now_ms,
            required_test_contracts=policy.required_test_contracts,
            required_proof_contracts=policy.required_proof_contracts,
            require_adversarial=policy.require_adversarial,
            require_held_out=policy.require_held_out,
            require_shadow=policy.require_shadow,
            required_evidence_kinds=policy.required_evidence_kinds,
        )


@dataclass(frozen=True)
class CertificateAdmission:
    """Independent certificate verdict.  Never an authority or promotion grant."""

    status: CertificateVerificationStatus
    reason_code: CertificateReasonCode
    certificate_cid: str
    issuer: str
    accepted: bool
    usable: bool
    grants_authority: bool
    grants_promotion: bool
    message: str = ""
    bound_identities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "status", _enum(self.status, CertificateVerificationStatus, "status")
        )
        object.__setattr__(
            self, "reason_code", _enum(self.reason_code, CertificateReasonCode, "reason_code")
        )
        object.__setattr__(
            self, "certificate_cid", _identifier(self.certificate_cid, "certificate_cid")
        )
        object.__setattr__(self, "issuer", _identifier(self.issuer, "issuer", required=False))
        for name in ("accepted", "usable", "grants_authority", "grants_promotion"):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        object.__setattr__(self, "message", _text(self.message, "message", required=False))
        object.__setattr__(
            self,
            "bound_identities",
            _strings(self.bound_identities, "bound_identities", identifiers=True),
        )
        if self.grants_authority or self.grants_promotion:
            raise ProcedureCertificateError("certificate admission cannot grant authority or promotion")
        if self.accepted and self.status is not CertificateVerificationStatus.ACCEPTED:
            raise ProcedureCertificateError("accepted admission requires accepted status")
        if self.usable and not self.accepted:
            raise ProcedureCertificateError("unaccepted certificates are not usable")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason_code": self.reason_code.value,
            "certificate_cid": self.certificate_cid,
            "issuer": self.issuer,
            "accepted": self.accepted,
            "usable": self.usable,
            "grants_authority": False,
            "grants_promotion": False,
            "message": self.message,
            "bound_identities": self.bound_identities,
        }


def _reject_admission(
    *,
    reason: CertificateReasonCode,
    message: str,
    certificate_cid: str = "unverified-certificate",
    issuer: str = "",
    bound_identities: Sequence[str] = (),
) -> CertificateAdmission:
    return CertificateAdmission(
        status=CertificateVerificationStatus.REJECTED,
        reason_code=reason,
        certificate_cid=certificate_cid or "unverified-certificate",
        issuer=issuer,
        accepted=False,
        usable=False,
        grants_authority=False,
        grants_promotion=False,
        message=message,
        bound_identities=tuple(bound_identities),
    )


def _is_self_issuer(issuer: str, candidate: ProcedureCandidate | None, procedure_cid: str) -> bool:
    normalized = issuer.lower()
    if normalized in FORBIDDEN_SELF_PRODUCERS:
        return True
    if normalized == procedure_cid.lower():
        return True
    if candidate is not None and normalized in _self_identities(candidate):
        return True
    return False


def _decode_certificate(value: ProcedureCertificate | Mapping[str, Any]) -> ProcedureCertificate:
    if isinstance(value, ProcedureCertificate):
        return value
    if isinstance(value, Mapping):
        try:
            return ProcedureCertificate.from_dict(value)
        except ProcedureContractError as exc:
            raise ProcedureCertificateError(str(exc)) from exc
    raise ProcedureCertificateError("certificate must be a ProcedureCertificate")


def _missing_bindings(payload: Mapping[str, Any]) -> tuple[str, ...]:
    missing: list[str] = []
    for name in REQUIRED_CERTIFICATE_BINDINGS:
        if name not in payload:
            missing.append(name)
            continue
        value = payload[name]
        if name == "known_limitations":
            if value is None:
                missing.append(name)
            continue
        if name == "signature":
            if not isinstance(value, str) or not value.strip() or value.strip() == PENDING_SIGNATURE:
                missing.append(name)
            continue
        if name in _NONEMPTY_SEQUENCE_BINDINGS:
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or not value:
                missing.append(name)
            continue
        if name == "procedure_version":
            if value in (None, "", {}, ()):
                missing.append(name)
            continue
        if name in {"issued_at_ms", "expires_at_ms"}:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                missing.append(name)
            continue
        if value in (None, ""):
            missing.append(name)
    return tuple(missing)


def _certificate_identities(certificate: ProcedureCertificate) -> tuple[str, ...]:
    return (
        certificate.procedure_cid,
        certificate.task_family_cid,
        certificate.counterexample_set_cid,
        certificate.operation_catalog_revision,
        certificate.effect_policy_revision,
        certificate.authority_policy_revision,
        certificate.verification_policy_revision,
        certificate.held_out_evaluation_cid,
        certificate.shadow_evaluation_cid,
        certificate.issuer,
        *certificate.source_episode_cids,
        *certificate.specification_cids,
        *certificate.repository_families,
        *certificate.supported_language_classes,
        *certificate.supported_framework_classes,
        *certificate.proof_receipt_cids,
        *certificate.test_receipt_cids,
        *certificate.adversarial_assurance_cids,
    )


class ProcedureCertificateIssuer:
    """Authorized issuer of independently verified procedure certificates."""

    revision: Final[str] = ISSUER_REVISION
    signing_scope: Final[str] = CERTIFICATE_SIGNING_SCOPE

    def __init__(
        self,
        trust: TrustedProofPolicy | SignerTrustRegistry,
        keyring: CertificateKeyRing,
        *,
        issuer_id: str,
    ) -> None:
        if not isinstance(keyring, CertificateKeyRing):
            raise ProcedureCertificateError("issuer requires a CertificateKeyRing")
        self._trust = trust
        self._registry = _signer_registry(trust)
        self._keyring = keyring
        self._issuer_id = _identifier(issuer_id, "issuer_id")
        if self._issuer_id not in self._keyring:
            raise ProcedureCertificateError("issuer has no authorized signing key")
        decision = self._registry.evaluate(
            self._issuer_id, scope=CERTIFICATE_SIGNING_SCOPE, required_trusted=True
        )
        if not decision.accepted:
            raise ProcedureCertificateError(
                "issuer {} is not admitted for procedure certificates: {}".format(
                    self._issuer_id, decision.reason_code or decision.message
                )
            )

    @property
    def issuer_id(self) -> str:
        return self._issuer_id

    def issue(
        self,
        candidate: ProcedureCandidate,
        verification: ProcedureVerification,
        evidence: IndependentEvidence,
        policy: VerificationPolicy,
        *,
        now_ms: int,
        known_limitations: Sequence[str] | None = None,
    ) -> ProcedureCertificate:
        if not isinstance(candidate, ProcedureCandidate):
            raise ProcedureCertificateError("candidate must be a ProcedureCandidate")
        if not isinstance(verification, ProcedureVerification):
            raise ProcedureCertificateError("verification must be a ProcedureVerification")
        if not isinstance(evidence, IndependentEvidence):
            raise ProcedureCertificateError("evidence must be IndependentEvidence")
        if not isinstance(policy, VerificationPolicy):
            raise ProcedureCertificateError("policy must be a VerificationPolicy")
        issued_at = _nonnegative_int(now_ms, "now_ms")
        if verification.status is not VerificationStatus.ACCEPTED or not verification.accepted:
            raise ProcedureCertificateError("only independently verified candidates receive certificates")
        if verification.candidate_cid != candidate.content_id:
            raise ProcedureCertificateError("verification does not bind this candidate")
        if verification.procedure_cid != candidate.procedure.content_id:
            raise ProcedureCertificateError("verification does not bind this procedure")
        if verification.policy_revision != policy.revision:
            raise ProcedureCertificateError("verification policy is not the current policy")
        if not all(item.accepted for item in verification.layers):
            raise ProcedureCertificateError("verification omitted a required layer")
        reported = tuple(item.layer.value for item in verification.layers)
        if reported != REQUIRED_VERIFICATION_LAYERS:
            raise ProcedureCertificateError("verification layers do not bind every required obligation")
        if _is_self_issuer(self._issuer_id, candidate, candidate.procedure.content_id):
            raise ProcedureCertificateError("a procedure cannot issue its own certificate")
        if evidence.producer_id == self._issuer_id:
            # The evidence producer may be a distinct campaign; the issuer must
            # still be an independent signing authority, not the evidence bundle.
            pass
        if _is_self_issuer(evidence.producer_id, candidate, candidate.procedure.content_id):
            raise ProcedureCertificateError("self-produced evidence cannot be certified")
        limitations = evidence.known_limitations if known_limitations is None else _strings(
            known_limitations, "known_limitations", limit=64
        )
        expires_at = issued_at + policy.review_horizon_ms
        if expires_at <= issued_at:
            raise ProcedureCertificateError("certificate review horizon must be positive")
        procedure = candidate.procedure
        pending = ProcedureCertificate(
            bindings=procedure.bindings,
            procedure_cid=procedure.content_id,
            procedure_version=procedure.version,
            task_family_cid=procedure.task_family_id,
            source_episode_cids=evidence.source_episode_cids,
            specification_cids=evidence.specification_cids,
            counterexample_set_cid=evidence.counterexample_set_cid,
            operation_catalog_revision=policy.operation_catalog_revision,
            effect_policy_revision=policy.effect_policy_revision,
            authority_policy_revision=policy.authority_policy_revision,
            verification_policy_revision=policy.revision,
            repository_families=evidence.repository_families,
            supported_language_classes=evidence.supported_language_classes,
            supported_framework_classes=evidence.supported_framework_classes,
            risk_ceiling=procedure.authority.risk_ceiling,
            proof_receipt_cids=evidence.proof_receipt_cids,
            test_receipt_cids=evidence.test_receipt_cids,
            adversarial_assurance_cids=evidence.adversarial_assurance_cids,
            held_out_evaluation_cid=evidence.held_out_evaluation_cid,
            shadow_evaluation_cid=evidence.shadow_evaluation_cid,
            known_limitations=limitations,
            issuer=self._issuer_id,
            signature=PENDING_SIGNATURE,
            issued_at_ms=issued_at,
            expires_at_ms=expires_at,
            state=ArtifactState.VERIFIED,
        )
        statement = unsigned_certificate_statement(pending)
        missing = _missing_bindings({**statement, "signature": "present"})
        if missing:
            raise ProcedureCertificateError(
                "certificate omitted required bindings: " + ",".join(missing)
            )
        signature = self._keyring.sign(self._issuer_id, encode_certificate_statement(statement))
        certificate = ProcedureCertificate(
            bindings=pending.bindings,
            procedure_cid=pending.procedure_cid,
            procedure_version=pending.procedure_version,
            task_family_cid=pending.task_family_cid,
            source_episode_cids=pending.source_episode_cids,
            specification_cids=pending.specification_cids,
            counterexample_set_cid=pending.counterexample_set_cid,
            operation_catalog_revision=pending.operation_catalog_revision,
            effect_policy_revision=pending.effect_policy_revision,
            authority_policy_revision=pending.authority_policy_revision,
            verification_policy_revision=pending.verification_policy_revision,
            repository_families=pending.repository_families,
            supported_language_classes=pending.supported_language_classes,
            supported_framework_classes=pending.supported_framework_classes,
            risk_ceiling=pending.risk_ceiling,
            proof_receipt_cids=pending.proof_receipt_cids,
            test_receipt_cids=pending.test_receipt_cids,
            adversarial_assurance_cids=pending.adversarial_assurance_cids,
            held_out_evaluation_cid=pending.held_out_evaluation_cid,
            shadow_evaluation_cid=pending.shadow_evaluation_cid,
            known_limitations=pending.known_limitations,
            issuer=pending.issuer,
            signature=signature,
            issued_at_ms=pending.issued_at_ms,
            expires_at_ms=pending.expires_at_ms,
            state=ArtifactState.VERIFIED,
        )
        if certificate.state is ArtifactState.PROMOTED:
            raise ProcedureCertificateError("certificate issuance cannot promote")
        return certificate


class ProcedureCertificateVerifier:
    """Verify certificates independently of procedure content."""

    revision: Final[str] = CERTIFICATE_VERIFIER_REVISION
    signing_scope: Final[str] = CERTIFICATE_SIGNING_SCOPE

    def __init__(
        self,
        trust: TrustedProofPolicy | SignerTrustRegistry,
        keyring: CertificateKeyRing,
    ) -> None:
        if not isinstance(keyring, CertificateKeyRing):
            raise ProcedureCertificateError("verifier requires a CertificateKeyRing")
        self._trust = trust
        self._registry = _signer_registry(trust)
        self._keyring = keyring

    def verify(
        self,
        certificate: ProcedureCertificate | Mapping[str, Any],
        context: CurrentCertificateContext,
        *,
        candidate: ProcedureCandidate | None = None,
    ) -> CertificateAdmission:
        """Admit a certificate without consulting procedure bodies.

        ``candidate`` is optional and used only to reject self-issuance against
        known candidate identities.  Procedure IR, steps, and effects are never
        read.
        """

        if not isinstance(context, CurrentCertificateContext):
            return _reject_admission(
                reason=CertificateReasonCode.MALFORMED_CERTIFICATE,
                message="certificate context is untyped",
            )
        payload: Mapping[str, Any]
        if isinstance(certificate, ProcedureCertificate):
            payload = certificate.to_dict()
            certificate_cid = certificate.content_id
        elif isinstance(certificate, Mapping):
            payload = certificate
            certificate_cid = str(payload.get("content_id") or "unverified-certificate")
        else:
            return _reject_admission(
                reason=CertificateReasonCode.MALFORMED_CERTIFICATE,
                message="certificate must be a ProcedureCertificate",
            )
        missing = _missing_bindings(payload)
        if missing:
            return _reject_admission(
                reason=CertificateReasonCode.INCOMPLETE_CERTIFICATE,
                message="certificate omitted required bindings: " + ",".join(missing),
                certificate_cid=certificate_cid if certificate_cid else "unverified-certificate",
                issuer=str(payload.get("issuer") or ""),
            )
        try:
            parsed = _decode_certificate(certificate)
        except ProcedureCertificateError as exc:
            return _reject_admission(
                reason=CertificateReasonCode.MALFORMED_CERTIFICATE,
                message=str(exc),
                certificate_cid=certificate_cid,
                issuer=str(payload.get("issuer") or ""),
            )
        certificate_cid = parsed.content_id
        issuer = parsed.issuer
        identities = _certificate_identities(parsed)
        if parsed.signature == PENDING_SIGNATURE or not parsed.signature.startswith(
            SIGNATURE_ALGORITHM + ":"
        ):
            return _reject_admission(
                reason=CertificateReasonCode.UNKNOWN_SIGNATURE_ALGORITHM,
                message="certificate signature algorithm is not hmac-sha256",
                certificate_cid=certificate_cid,
                issuer=issuer,
                bound_identities=identities,
            )
        if _is_self_issuer(issuer, candidate, parsed.procedure_cid):
            return _reject_admission(
                reason=CertificateReasonCode.SELF_ISSUED,
                message="certificate issuer is not independent of the procedure",
                certificate_cid=certificate_cid,
                issuer=issuer,
                bound_identities=identities,
            )
        decision = self._registry.evaluate(
            issuer, scope=CERTIFICATE_SIGNING_SCOPE, required_trusted=True
        )
        if not decision.accepted:
            reason = CertificateReasonCode.UNTRUSTED_ISSUER
            code = decision.reason_code or ""
            if code == "revoked_signer":
                reason = CertificateReasonCode.REVOKED_ISSUER
            elif code == "out_of_scope_signer":
                reason = CertificateReasonCode.ISSUER_OUT_OF_SCOPE
            return _reject_admission(
                reason=reason,
                message=decision.message,
                certificate_cid=certificate_cid,
                issuer=issuer,
                bound_identities=identities,
            )
        statement = unsigned_certificate_statement(parsed)
        if not self._keyring.verify(
            issuer, encode_certificate_statement(statement), parsed.signature
        ):
            return _reject_admission(
                reason=CertificateReasonCode.FORGED_SIGNATURE,
                message="certificate signature does not bind the canonical payload",
                certificate_cid=certificate_cid,
                issuer=issuer,
                bound_identities=identities,
            )
        if parsed.state in {
            ArtifactState.STALE,
            ArtifactState.REVOKED,
            ArtifactState.SUPERSEDED,
            ArtifactState.REJECTED,
            ArtifactState.DEGRADED,
        }:
            return _reject_admission(
                reason=CertificateReasonCode.STALE_CERTIFICATE,
                message="certificate lifecycle state is not current",
                certificate_cid=certificate_cid,
                issuer=issuer,
                bound_identities=identities,
            )
        if parsed.expires_at_ms <= context.now_ms or parsed.issued_at_ms > context.now_ms:
            return _reject_admission(
                reason=CertificateReasonCode.STALE_CERTIFICATE,
                message="certificate is expired or not yet valid",
                certificate_cid=certificate_cid,
                issuer=issuer,
                bound_identities=identities,
            )
        if parsed.bindings != context.bindings:
            return _reject_admission(
                reason=CertificateReasonCode.STALE_BINDINGS,
                message="certificate bindings are not current",
                certificate_cid=certificate_cid,
                issuer=issuer,
                bound_identities=identities,
            )
        if parsed.authority_policy_revision != context.authority_policy_revision:
            return _reject_admission(
                reason=CertificateReasonCode.STALE_POLICY,
                message="certificate authority policy is stale",
                certificate_cid=certificate_cid,
                issuer=issuer,
                bound_identities=identities,
            )
        if parsed.operation_catalog_revision != context.operation_catalog_revision:
            return _reject_admission(
                reason=CertificateReasonCode.STALE_POLICY,
                message="certificate operation catalog is stale",
                certificate_cid=certificate_cid,
                issuer=issuer,
                bound_identities=identities,
            )
        if parsed.effect_policy_revision != context.effect_policy_revision:
            return _reject_admission(
                reason=CertificateReasonCode.STALE_POLICY,
                message="certificate effect policy is stale",
                certificate_cid=certificate_cid,
                issuer=issuer,
                bound_identities=identities,
            )
        if parsed.verification_policy_revision != context.verification_policy_revision:
            return _reject_admission(
                reason=CertificateReasonCode.WEAKER_VALIDATION,
                message="certificate verification policy is not the current policy",
                certificate_cid=certificate_cid,
                issuer=issuer,
                bound_identities=identities,
            )
        if context.require_adversarial and not parsed.adversarial_assurance_cids:
            return _reject_admission(
                reason=CertificateReasonCode.WEAKER_VALIDATION,
                message="certificate omitted required adversarial-assurance evidence",
                certificate_cid=certificate_cid,
                issuer=issuer,
                bound_identities=identities,
            )
        if context.require_held_out and not parsed.held_out_evaluation_cid:
            return _reject_admission(
                reason=CertificateReasonCode.WEAKER_VALIDATION,
                message="certificate omitted required held-out evaluation",
                certificate_cid=certificate_cid,
                issuer=issuer,
                bound_identities=identities,
            )
        if context.require_shadow and not parsed.shadow_evaluation_cid:
            return _reject_admission(
                reason=CertificateReasonCode.WEAKER_VALIDATION,
                message="certificate omitted required shadow evaluation",
                certificate_cid=certificate_cid,
                issuer=issuer,
                bound_identities=identities,
            )
        if not parsed.proof_receipt_cids or not parsed.test_receipt_cids:
            return _reject_admission(
                reason=CertificateReasonCode.WEAKER_VALIDATION,
                message="certificate omitted required proof or test receipts",
                certificate_cid=certificate_cid,
                issuer=issuer,
                bound_identities=identities,
            )
        if parsed.state is ArtifactState.PROMOTED:
            # Promoted is a certificate-tier state the registry may later
            # assign; this verifier still never treats promotion as granted.
            pass
        return CertificateAdmission(
            status=CertificateVerificationStatus.ACCEPTED,
            reason_code=CertificateReasonCode.ACCEPTED,
            certificate_cid=certificate_cid,
            issuer=issuer,
            accepted=True,
            usable=True,
            grants_authority=False,
            grants_promotion=False,
            message="",
            bound_identities=identities,
        )


def issue_procedure_certificate(
    candidate: ProcedureCandidate,
    verification: ProcedureVerification,
    evidence: IndependentEvidence,
    policy: VerificationPolicy,
    issuer: ProcedureCertificateIssuer,
    *,
    now_ms: int,
) -> ProcedureCertificate:
    return issuer.issue(candidate, verification, evidence, policy, now_ms=now_ms)


def verify_procedure_certificate(
    certificate: ProcedureCertificate | Mapping[str, Any],
    context: CurrentCertificateContext,
    verifier: ProcedureCertificateVerifier,
    *,
    candidate: ProcedureCandidate | None = None,
) -> CertificateAdmission:
    return verifier.verify(certificate, context, candidate=candidate)


__all__ = [
    "CERTIFICATE_SIGNING_SCOPE",
    "CERTIFICATE_VERIFIER_REVISION",
    "ISSUER_REVISION",
    "PENDING_SIGNATURE",
    "REQUIRED_CERTIFICATE_BINDINGS",
    "SIGNATURE_ALGORITHM",
    "CertificateAdmission",
    "CertificateKeyRing",
    "CertificateReasonCode",
    "CertificateVerificationStatus",
    "CurrentCertificateContext",
    "ProcedureCertificateError",
    "ProcedureCertificateIssuer",
    "ProcedureCertificateVerifier",
    "encode_certificate_statement",
    "issue_procedure_certificate",
    "unsigned_certificate_statement",
    "verify_procedure_certificate",
]
