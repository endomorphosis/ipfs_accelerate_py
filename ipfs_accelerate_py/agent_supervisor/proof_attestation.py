"""Receipt-bound zero-knowledge attestation contracts.

This module defines the supervisor side of the ZKP trust boundary.  It does
not claim that a ZKP proves arbitrary Python correctness.  A statement may be
prepared only from an existing, current, independently kernel-verified
``ProofReceipt``.  The ZKP then attests that immutable receipt and its public
provenance identities.

Private witness material deliberately uses a separate, non-serializable
object.  It is not a field of any canonical contract, public artifact, cache
key, context capsule, log representation, or proof evidence record.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Dict, TypeVar

from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    ContractValidationError,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    _canonical_value,
    _enum,
    _schema,
    _text,
    canonical_json_bytes,
    content_identity,
)


PROOF_ATTESTATION_CONTRACT_VERSION = 1
PROOF_ATTESTATION_STATEMENT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proof-attestation-statement@1"
)
PROOF_ATTESTATION_ENVELOPE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proof-attestation-envelope@1"
)
PROOF_ATTESTATION_VERIFICATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proof-attestation-verification@1"
)

# Compatibility names used by callers which emphasize that this is a ZKP.
ZKP_RECEIPT_ATTESTATION_STATEMENT_SCHEMA = PROOF_ATTESTATION_STATEMENT_SCHEMA
ZKP_RECEIPT_ATTESTATION_ENVELOPE_SCHEMA = PROOF_ATTESTATION_ENVELOPE_SCHEMA
ZKP_RECEIPT_ATTESTATION_VERIFICATION_SCHEMA = PROOF_ATTESTATION_VERIFICATION_SCHEMA
RECEIPT_ATTESTATION_STATEMENT_SCHEMA = PROOF_ATTESTATION_STATEMENT_SCHEMA
RECEIPT_ATTESTATION_ENVELOPE_SCHEMA = PROOF_ATTESTATION_ENVELOPE_SCHEMA
ATTESTATION_VERIFICATION_SCHEMA = PROOF_ATTESTATION_VERIFICATION_SCHEMA


class AttestationValidationError(ContractValidationError):
    """Raised when receipt-attestation data violates the trust contract."""


class WitnessDisclosureError(AttestationValidationError):
    """Raised when private witness material reaches a serialization boundary."""


class AttestationBackendMode(str, Enum):
    """Trust class of the proof backend, independent of its product name."""

    CRYPTOGRAPHIC = "cryptographic"
    PRODUCTION = "cryptographic"  # compatibility spelling
    SIMULATED = "simulated"


ZKPBackendMode = AttestationBackendMode


class AttestationTrust(str, Enum):
    """Authority derived from verification, never asserted by a provider."""

    NON_AUTHORITATIVE = "non_authoritative"
    AUTHORITATIVE = "authoritative"


class AttestationGate(str, Enum):
    """Supervisor gates at which an attestation might be consumed."""

    SERIALIZATION = "serialization"
    TEST = "test"
    PRODUCTION = "production"
    COMPLETION = "completion"


class AttestationVerificationVerdict(str, Enum):
    """Result returned by the independent attestation verifier."""

    VERIFIED = "verified"
    REJECTED = "rejected"
    ERROR = "error"


AttestationMode = AttestationBackendMode


def _bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise AttestationValidationError("%s must be a boolean" % field_name)
    return value


def _statement(value: Any) -> "ReceiptAttestationStatement":
    if isinstance(value, ReceiptAttestationStatement):
        return value
    if isinstance(value, Mapping):
        return ReceiptAttestationStatement.from_dict(value)
    raise AttestationValidationError(
        "statement must be a ReceiptAttestationStatement or mapping"
    )


def _backend_id_is_explicitly_simulated(backend_id: str) -> bool:
    """Recognize backend identities which can never be cryptographic."""

    normalized = (
        backend_id.strip()
        .lower()
        .replace("/", ":")
        .replace("@", ":")
        .replace("_", "-")
    )
    tokens = normalized.split(":")
    return any(
        token in {"sim", "simulated", "mock", "fake", "demo", "educational"}
        or token.startswith("simulated-")
        for token in tokens
    )


@dataclass(frozen=True)
class ReceiptAttestationStatement(CanonicalContract):
    """The complete public statement proven by a receipt attestation.

    The eight identity fields are intentionally explicit rather than hidden in
    an opaque digest.  ``statement_id`` and ``public_inputs_digest`` commit to
    their canonical encoding for circuit adapters that require a single field.
    No witness values or witness field names are accepted by this contract.
    """

    SCHEMA: ClassVar[str] = PROOF_ATTESTATION_STATEMENT_SCHEMA

    repository_tree_id: str
    obligation_id: str
    policy_id: str
    kernel_id: str
    receipt_id: str
    circuit_id: str
    backend_id: str
    verification_key_id: str

    def __post_init__(self) -> None:
        for name in (
            "repository_tree_id",
            "obligation_id",
            "policy_id",
            "kernel_id",
            "receipt_id",
            "circuit_id",
            "backend_id",
            "verification_key_id",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=True),
            )

    @property
    def tree_id(self) -> str:
        return self.repository_tree_id

    @property
    def statement_id(self) -> str:
        return self.content_id

    @property
    def public_input_digest(self) -> str:
        """Content identity of the circuit-facing public input object."""

        return content_identity(self.public_inputs)

    @property
    def public_inputs_digest(self) -> str:
        """Compatibility plural spelling for :attr:`public_input_digest`."""

        return self.public_input_digest

    @property
    def public_inputs(self) -> Dict[str, str]:
        """Return exactly the public identities supplied to the circuit."""

        return {
            "backend_id": self.backend_id,
            "circuit_id": self.circuit_id,
            "kernel_id": self.kernel_id,
            "obligation_id": self.obligation_id,
            "policy_id": self.policy_id,
            "receipt_id": self.receipt_id,
            "repository_tree_id": self.repository_tree_id,
            "verification_key_id": self.verification_key_id,
        }

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": PROOF_ATTESTATION_CONTRACT_VERSION,
            **self.public_inputs,
        }

    @classmethod
    def from_receipt(
        cls,
        receipt: ProofReceipt,
        *,
        circuit_id: str,
        backend_id: str,
        verification_key_id: str,
    ) -> "ReceiptAttestationStatement":
        """Build a public statement after enforcing receipt eligibility."""

        if not isinstance(receipt, ProofReceipt):
            raise AttestationValidationError(
                "attestation requires an existing ProofReceipt object"
            )
        receipt.require_kernel_verified()
        return cls(
            repository_tree_id=receipt.repository_tree_id,
            obligation_id=receipt.obligation_id,
            policy_id=receipt.policy_id,
            kernel_id=receipt.kernel_id,
            receipt_id=receipt.receipt_id,
            circuit_id=circuit_id,
            backend_id=backend_id,
            verification_key_id=verification_key_id,
        )

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ReceiptAttestationStatement":
        _schema(payload, cls.SCHEMA)
        result = cls(
            repository_tree_id=payload.get(
                "repository_tree_id", payload.get("tree_id", "")
            ),
            obligation_id=payload.get("obligation_id", ""),
            policy_id=payload.get("policy_id", ""),
            kernel_id=payload.get("kernel_id", ""),
            receipt_id=payload.get("receipt_id", ""),
            circuit_id=payload.get("circuit_id", ""),
            backend_id=payload.get("backend_id", ""),
            verification_key_id=payload.get("verification_key_id", ""),
        )
        claimed = payload.get("statement_id") or payload.get("content_id")
        if claimed and claimed != result.statement_id:
            raise AttestationValidationError(
                "attestation statement identity does not match payload"
            )
        claimed_digest = payload.get("public_input_digest") or payload.get(
            "public_inputs_digest"
        )
        if claimed_digest and claimed_digest != result.public_input_digest:
            raise AttestationValidationError(
                "attestation public-input digest does not match payload"
            )
        return result

    def to_public_artifact(self) -> Dict[str, Any]:
        """Return the public statement plus its derived identities."""

        return {
            **self.to_dict(),
            "statement_id": self.statement_id,
            "public_input_digest": self.public_input_digest,
        }

    to_context_capsule = to_public_artifact
    to_cache_record = to_public_artifact
    to_log_record = to_public_artifact


ZKPReceiptAttestationStatement = ReceiptAttestationStatement
ProofAttestationStatement = ReceiptAttestationStatement


T = TypeVar("T")


class PrivateAttestationWitness:
    """Opaque, redacted, and non-serializable private proving inputs.

    A backend receives the values only inside :meth:`use`.  The wrapper has no
    mapping protocol, iterator, public value property, dataclass fields, JSON
    method, or pickle representation.  This keeps generic logging, context,
    artifact, and cache serializers from accidentally traversing its secrets.
    """

    __slots__ = ("__values",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        if not isinstance(values, Mapping):
            raise AttestationValidationError("witness values must be a mapping")
        normalized: Dict[str, Any] = {}
        for raw_name, value in values.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise AttestationValidationError(
                    "witness field names must be non-empty strings"
                )
            normalized[raw_name] = value
        if not normalized:
            raise AttestationValidationError("witness values must not be empty")
        # A private copy prevents later mutation through the caller's mapping.
        self.__values = dict(normalized)

    def __repr__(self) -> str:
        return "<PrivateAttestationWitness redacted>"

    __str__ = __repr__

    def __copy__(self) -> "PrivateAttestationWitness":
        raise WitnessDisclosureError("private witness cannot be copied")

    def __deepcopy__(self, memo: Any) -> "PrivateAttestationWitness":
        del memo
        raise WitnessDisclosureError("private witness cannot be copied")

    def __reduce_ex__(self, protocol: int) -> Any:
        del protocol
        raise WitnessDisclosureError(
            "private witness cannot be serialized or cached"
        )

    def __getstate__(self) -> Any:
        raise WitnessDisclosureError(
            "private witness cannot be serialized or cached"
        )

    def to_dict(self) -> Dict[str, Any]:
        raise WitnessDisclosureError(
            "private witness has no public dictionary representation"
        )

    def use(self, consumer: Callable[[Mapping[str, Any]], T]) -> T:
        """Invoke a local prover callback with a read-only witness view."""

        if not callable(consumer):
            raise AttestationValidationError("witness consumer must be callable")
        return consumer(MappingProxyType(self.__values))

    def redacted(self) -> Dict[str, bool]:
        """Return a constant safe marker which reveals no field names."""

        return {"private_witness_redacted": True}


ReceiptAttestationWitness = PrivateAttestationWitness
ZKPWitness = PrivateAttestationWitness


@dataclass(frozen=True, repr=False)
class ReceiptAttestationRequest:
    """Ephemeral proving request; only its public statement is serializable."""

    statement: ReceiptAttestationStatement
    _witness: PrivateAttestationWitness = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "statement", _statement(self.statement))
        if not isinstance(self._witness, PrivateAttestationWitness):
            raise AttestationValidationError(
                "_witness must be a PrivateAttestationWitness"
            )

    def __repr__(self) -> str:
        return (
            "ReceiptAttestationRequest("
            "statement_id=%r, witness=<redacted>)" % self.statement.statement_id
        )

    __str__ = __repr__

    def __reduce_ex__(self, protocol: int) -> Any:
        del protocol
        raise WitnessDisclosureError(
            "attestation proving requests cannot be serialized or cached"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return only public proving inputs, never the witness."""

        return {
            "statement": self.statement.to_public_artifact(),
            "private_witness_redacted": True,
        }

    to_public_artifact = to_dict
    to_context_capsule = to_dict
    to_log_record = to_dict

    def to_cache_record(self) -> Dict[str, Any]:
        raise WitnessDisclosureError(
            "attestation proving requests containing a witness cannot be cached"
        )

    def use_witness(self, consumer: Callable[[Mapping[str, Any]], T]) -> T:
        """Pass the witness to an in-process proving callback."""

        return self._witness.use(consumer)


ZKPReceiptAttestationRequest = ReceiptAttestationRequest
ProofAttestationRequest = ReceiptAttestationRequest


@dataclass(frozen=True)
class ReceiptAttestationEnvelope(CanonicalContract):
    """Public proof envelope produced for one immutable statement.

    Generation is not verification: even a cryptographic envelope is
    non-authoritative until an independent verifier records a successful
    :class:`AttestationVerification`.
    """

    SCHEMA: ClassVar[str] = PROOF_ATTESTATION_ENVELOPE_SCHEMA

    statement: ReceiptAttestationStatement
    backend_mode: AttestationBackendMode
    proof_artifact_id: str
    proof_digest: str
    prover_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "statement", _statement(self.statement))
        object.__setattr__(
            self,
            "backend_mode",
            _enum(
                self.backend_mode,
                AttestationBackendMode,
                field_name="backend_mode",
            ),
        )
        for name in ("proof_artifact_id", "proof_digest"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=True),
            )
        object.__setattr__(
            self, "prover_id", _text(self.prover_id, field_name="prover_id")
        )
        if (
            self.backend_mode is AttestationBackendMode.CRYPTOGRAPHIC
            and _backend_id_is_explicitly_simulated(self.statement.backend_id)
        ):
            raise AttestationValidationError(
                "a simulated backend identity cannot use cryptographic mode"
            )

    @property
    def envelope_id(self) -> str:
        return self.content_id

    @property
    def simulated(self) -> bool:
        return self.backend_mode is AttestationBackendMode.SIMULATED

    @property
    def is_simulated(self) -> bool:
        return self.simulated

    @property
    def authoritative(self) -> bool:
        # Proof generation and provider claims never cross the verifier boundary.
        return False

    @property
    def is_authoritative(self) -> bool:
        return self.authoritative

    @property
    def trust(self) -> AttestationTrust:
        return AttestationTrust.NON_AUTHORITATIVE

    @property
    def non_authoritative_reason(self) -> str:
        if self.simulated:
            return "simulated_zkp_is_non_authoritative"
        return "independent_verification_required"

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": PROOF_ATTESTATION_CONTRACT_VERSION,
            "statement": self.statement,
            "statement_id": self.statement.statement_id,
            "public_input_digest": self.statement.public_input_digest,
            "backend_mode": self.backend_mode,
            "proof_artifact_id": self.proof_artifact_id,
            "proof_digest": self.proof_digest,
            "prover_id": self.prover_id,
            "simulated": self.simulated,
            "authoritative": False,
            "trust": self.trust,
            "non_authoritative_reason": self.non_authoritative_reason,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ReceiptAttestationEnvelope":
        _schema(payload, cls.SCHEMA)
        result = cls(
            statement=_statement(payload.get("statement") or {}),
            backend_mode=payload.get(
                "backend_mode", AttestationBackendMode.SIMULATED
            ),
            proof_artifact_id=payload.get("proof_artifact_id", ""),
            proof_digest=payload.get("proof_digest", ""),
            prover_id=payload.get("prover_id", ""),
        )
        if payload.get("authoritative") not in (None, False):
            raise AttestationValidationError(
                "a generated attestation envelope cannot assert authority"
            )
        claimed_trust = payload.get("trust")
        if claimed_trust not in (None, "", AttestationTrust.NON_AUTHORITATIVE.value):
            raise AttestationValidationError(
                "a generated attestation envelope must be non-authoritative"
            )
        if payload.get("simulated") not in (None, result.simulated):
            raise AttestationValidationError(
                "attestation simulation label does not match backend mode"
            )
        if payload.get("statement_id") not in (None, "", result.statement.statement_id):
            raise AttestationValidationError(
                "attestation statement identity does not match envelope"
            )
        if payload.get("public_input_digest") not in (
            None,
            "",
            result.statement.public_input_digest,
        ):
            raise AttestationValidationError(
                "attestation public-input digest does not match envelope"
            )
        claimed_id = payload.get("envelope_id") or payload.get("content_id")
        if claimed_id and claimed_id != result.envelope_id:
            raise AttestationValidationError(
                "attestation envelope identity does not match payload"
            )
        return result

    def to_public_artifact(self) -> Dict[str, Any]:
        return {**self.to_dict(), "envelope_id": self.envelope_id}

    to_context_capsule = to_public_artifact
    to_cache_record = to_public_artifact
    to_log_record = to_public_artifact


ZKPReceiptAttestation = ReceiptAttestationEnvelope
ProofAttestationEnvelope = ReceiptAttestationEnvelope
AttestationEnvelope = ReceiptAttestationEnvelope


def _envelope(value: Any) -> ReceiptAttestationEnvelope:
    if isinstance(value, ReceiptAttestationEnvelope):
        return value
    if isinstance(value, Mapping):
        return ReceiptAttestationEnvelope.from_dict(value)
    raise AttestationValidationError(
        "envelope must be a ReceiptAttestationEnvelope or mapping"
    )


@dataclass(frozen=True)
class AttestationVerification(CanonicalContract):
    """Independent verification result and its derived gate authority."""

    SCHEMA: ClassVar[str] = PROOF_ATTESTATION_VERIFICATION_SCHEMA

    envelope: ReceiptAttestationEnvelope
    verdict: AttestationVerificationVerdict
    verifier_id: str
    independent: bool = True
    diagnostic_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "envelope", _envelope(self.envelope))
        object.__setattr__(
            self,
            "verdict",
            _enum(
                self.verdict,
                AttestationVerificationVerdict,
                field_name="verdict",
            ),
        )
        object.__setattr__(
            self,
            "verifier_id",
            _text(self.verifier_id, field_name="verifier_id", required=True),
        )
        object.__setattr__(
            self,
            "diagnostic_code",
            _text(self.diagnostic_code, field_name="diagnostic_code"),
        )
        object.__setattr__(
            self, "independent", _bool(self.independent, field_name="independent")
        )

    @property
    def verification_id(self) -> str:
        return self.content_id

    @property
    def verified(self) -> bool:
        return self.verdict is AttestationVerificationVerdict.VERIFIED

    @property
    def simulated(self) -> bool:
        return self.envelope.simulated

    @property
    def is_simulated(self) -> bool:
        return self.simulated

    @property
    def authoritative(self) -> bool:
        return (
            self.verified
            and self.independent
            and self.envelope.backend_mode
            is AttestationBackendMode.CRYPTOGRAPHIC
        )

    @property
    def is_authoritative(self) -> bool:
        return self.authoritative

    @property
    def trust(self) -> AttestationTrust:
        if self.authoritative:
            return AttestationTrust.AUTHORITATIVE
        return AttestationTrust.NON_AUTHORITATIVE

    @property
    def authoritative_assurance(self) -> AssuranceLevel:
        """Assurance contributed by the envelope alone.

        An authoritative result contributes ``ATTESTED`` only because statement
        creation required the bound base receipt to already be kernel verified.
        A simulated or rejected result contributes no assurance.
        """

        if self.authoritative:
            return AssuranceLevel.ATTESTED
        return AssuranceLevel.UNVERIFIED

    def satisfies_gate(self, gate: AttestationGate) -> bool:
        normalized = _enum(gate, AttestationGate, field_name="gate")
        if normalized in (AttestationGate.SERIALIZATION, AttestationGate.TEST):
            return self.verified
        return self.authoritative

    def satisfies_production_gate(self) -> bool:
        return self.satisfies_gate(AttestationGate.PRODUCTION)

    def satisfies_completion_gate(self) -> bool:
        return self.satisfies_gate(AttestationGate.COMPLETION)

    def to_evidence(self) -> ProofEvidence:
        """Project the result into the shared proof-assurance vocabulary."""

        statement = self.envelope.statement
        verdict = (
            EvidenceVerdict.ACCEPTED
            if self.verified
            else (
                EvidenceVerdict.REJECTED
                if self.verdict is AttestationVerificationVerdict.REJECTED
                else EvidenceVerdict.ERROR
            )
        )
        return ProofEvidence(
            kind=EvidenceKind.CRYPTOGRAPHIC_ATTESTATION,
            authority=EvidenceAuthority.ATTESTATION_VERIFIER,
            verdict=verdict,
            artifact_id=self.envelope.proof_artifact_id,
            subject_id=statement.receipt_id,
            verifier_id=self.verifier_id,
            freshness=EvidenceFreshness.CURRENT,
            independent=self.independent,
            simulated=self.simulated,
            metadata={
                "attestation_contract_version": PROOF_ATTESTATION_CONTRACT_VERSION,
                "attestation_envelope_id": self.envelope.envelope_id,
                "attestation_statement_id": statement.statement_id,
                "backend_id": statement.backend_id,
                "circuit_id": statement.circuit_id,
                "kernel_id": statement.kernel_id,
                "obligation_id": statement.obligation_id,
                "policy_id": statement.policy_id,
                "public_input_digest": statement.public_input_digest,
                "repository_tree_id": statement.repository_tree_id,
                "verification_key_id": statement.verification_key_id,
            },
        )

    as_evidence = to_evidence

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": PROOF_ATTESTATION_CONTRACT_VERSION,
            "envelope": self.envelope,
            "envelope_id": self.envelope.envelope_id,
            "verdict": self.verdict,
            "verifier_id": self.verifier_id,
            "independent": self.independent,
            "diagnostic_code": self.diagnostic_code,
            "simulated": self.simulated,
            "authoritative": self.authoritative,
            "trust": self.trust,
            "authoritative_assurance": self.authoritative_assurance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AttestationVerification":
        _schema(payload, cls.SCHEMA)
        result = cls(
            envelope=_envelope(payload.get("envelope") or {}),
            verdict=payload.get(
                "verdict", AttestationVerificationVerdict.ERROR
            ),
            verifier_id=payload.get("verifier_id", ""),
            independent=payload.get("independent", True),
            diagnostic_code=payload.get("diagnostic_code", ""),
        )
        derived_claims = {
            "envelope_id": result.envelope.envelope_id,
            "simulated": result.simulated,
            "authoritative": result.authoritative,
            "trust": result.trust.value,
            "authoritative_assurance": result.authoritative_assurance.value,
        }
        for name, expected in derived_claims.items():
            supplied = payload.get(name)
            if supplied not in (None, "", expected):
                raise AttestationValidationError(
                    "attestation verification %s does not match derived value"
                    % name
                )
        claimed_id = payload.get("verification_id") or payload.get("content_id")
        if claimed_id and claimed_id != result.verification_id:
            raise AttestationValidationError(
                "attestation verification identity does not match payload"
            )
        return result

    def to_public_artifact(self) -> Dict[str, Any]:
        return {**self.to_dict(), "verification_id": self.verification_id}

    to_context_capsule = to_public_artifact
    to_cache_record = to_public_artifact
    to_log_record = to_public_artifact


ReceiptAttestationVerification = AttestationVerification
ZKPReceiptAttestationVerification = AttestationVerification
ProofAttestationVerification = AttestationVerification
AttestationVerificationResult = AttestationVerification


def build_receipt_attestation_statement(
    receipt: ProofReceipt,
    *,
    circuit_id: str,
    backend_id: str,
    verification_key_id: str,
) -> ReceiptAttestationStatement:
    """Prepare the canonical public statement for an eligible receipt."""

    return ReceiptAttestationStatement.from_receipt(
        receipt,
        circuit_id=circuit_id,
        backend_id=backend_id,
        verification_key_id=verification_key_id,
    )


def prepare_receipt_attestation(
    receipt: ProofReceipt,
    *,
    circuit_id: str,
    backend_id: str,
    verification_key_id: str,
    witness: PrivateAttestationWitness,
) -> ReceiptAttestationRequest:
    """Create an ephemeral proving request after the kernel receipt gate."""

    statement = build_receipt_attestation_statement(
        receipt,
        circuit_id=circuit_id,
        backend_id=backend_id,
        verification_key_id=verification_key_id,
    )
    return ReceiptAttestationRequest(statement=statement, _witness=witness)


prepare_proof_attestation = prepare_receipt_attestation
prepare_attestation = prepare_receipt_attestation
build_attestation_statement = build_receipt_attestation_statement


def create_attestation_envelope(
    request: ReceiptAttestationRequest,
    *,
    backend_mode: AttestationBackendMode,
    proof_artifact_id: str,
    proof_digest: str,
    prover_id: str = "",
) -> ReceiptAttestationEnvelope:
    """Record public prover output for an already-authorized request."""

    if not isinstance(request, ReceiptAttestationRequest):
        raise AttestationValidationError(
            "envelope generation requires a prepared receipt-attestation request"
        )
    return ReceiptAttestationEnvelope(
        statement=request.statement,
        backend_mode=backend_mode,
        proof_artifact_id=proof_artifact_id,
        proof_digest=proof_digest,
        prover_id=prover_id,
    )


def record_attestation_verification(
    envelope: ReceiptAttestationEnvelope,
    *,
    verified: bool,
    verifier_id: str,
    independent: bool = True,
    diagnostic_code: str = "",
) -> AttestationVerification:
    """Create a fail-closed independent verification result."""

    checked = _bool(verified, field_name="verified")
    return AttestationVerification(
        envelope=envelope,
        verdict=(
            AttestationVerificationVerdict.VERIFIED
            if checked
            else AttestationVerificationVerdict.REJECTED
        ),
        verifier_id=verifier_id,
        independent=independent,
        diagnostic_code=diagnostic_code,
    )


def attestation_satisfies_gate(
    verification: AttestationVerification, gate: AttestationGate
) -> bool:
    """Return whether a verified envelope can satisfy ``gate``."""

    if not isinstance(verification, AttestationVerification):
        return False
    return verification.satisfies_gate(gate)


def public_attestation_artifact(value: Any) -> Any:
    """Return a canonical public value or reject witness-bearing objects.

    This helper is intended for log, context-capsule, artifact, and cache
    boundaries.  Public attestation contracts are supported directly.  A
    proving request is reduced to its explicitly redacted public view.
    """

    if isinstance(value, PrivateAttestationWitness):
        raise WitnessDisclosureError(
            "private witness cannot enter a public artifact"
        )
    if isinstance(value, ReceiptAttestationRequest):
        return value.to_public_artifact()
    if isinstance(
        value,
        (
            ReceiptAttestationStatement,
            ReceiptAttestationEnvelope,
            AttestationVerification,
        ),
    ):
        return value.to_public_artifact()
    if isinstance(value, Mapping):
        return {
            str(key): public_attestation_artifact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [public_attestation_artifact(item) for item in value]
    return _canonical_value(value)


def public_artifact_contains(
    artifact: Any, secret: str | bytes
) -> bool:
    """Test helper for witness no-leak assertions."""

    needle = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
    if not needle:
        raise AttestationValidationError("secret probe must not be empty")
    encoded = canonical_json_bytes(public_attestation_artifact(artifact))
    return needle in encoded


__all__ = [
    "PROOF_ATTESTATION_CONTRACT_VERSION",
    "PROOF_ATTESTATION_ENVELOPE_SCHEMA",
    "PROOF_ATTESTATION_STATEMENT_SCHEMA",
    "PROOF_ATTESTATION_VERIFICATION_SCHEMA",
    "RECEIPT_ATTESTATION_ENVELOPE_SCHEMA",
    "RECEIPT_ATTESTATION_STATEMENT_SCHEMA",
    "ATTESTATION_VERIFICATION_SCHEMA",
    "ZKP_RECEIPT_ATTESTATION_ENVELOPE_SCHEMA",
    "ZKP_RECEIPT_ATTESTATION_STATEMENT_SCHEMA",
    "ZKP_RECEIPT_ATTESTATION_VERIFICATION_SCHEMA",
    "AttestationBackendMode",
    "AttestationEnvelope",
    "AttestationGate",
    "AttestationMode",
    "AttestationTrust",
    "AttestationValidationError",
    "AttestationVerification",
    "AttestationVerificationResult",
    "AttestationVerificationVerdict",
    "PrivateAttestationWitness",
    "ProofAttestationEnvelope",
    "ProofAttestationRequest",
    "ProofAttestationStatement",
    "ProofAttestationVerification",
    "ReceiptAttestationEnvelope",
    "ReceiptAttestationRequest",
    "ReceiptAttestationStatement",
    "ReceiptAttestationVerification",
    "ReceiptAttestationWitness",
    "WitnessDisclosureError",
    "ZKPBackendMode",
    "ZKPReceiptAttestation",
    "ZKPReceiptAttestationRequest",
    "ZKPReceiptAttestationStatement",
    "ZKPReceiptAttestationVerification",
    "ZKPWitness",
    "attestation_satisfies_gate",
    "build_receipt_attestation_statement",
    "build_attestation_statement",
    "create_attestation_envelope",
    "prepare_proof_attestation",
    "prepare_attestation",
    "prepare_receipt_attestation",
    "public_artifact_contains",
    "public_attestation_artifact",
    "record_attestation_verification",
]
