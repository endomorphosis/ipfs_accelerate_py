"""Evidence-class verification and cache-admission decisions (IPS-028).

Accelerate alone decides whether closed evidence may enter the proof-unit
cache.  Every candidate is verified according to its declared evidence class;
hashes, signed receipts, structural checks, and simulations retain that class
and are never promoted to direct-execution evidence.

A :class:`CacheAdmissionRecord` is issued only after successful verification.
Receipt aggregation never claims that underlying tests executed.

Interfaces: ``EvidenceVerifier``, ``AdmissionDecision``, ``verify_for_admission``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    DirectExecutionProof,
    EvidenceClass,
    EvidenceClassError,
    IncrementalCommitSeal,
    IntegrityCommitment,
    ProofMode,
    ProofTerminalStatus,
    ReceiptAggregationZkProof,
    SignedExecutionReceipt,
    evidence_from_canonical,
    status_satisfies_class,
)

EVIDENCE_SUBSET: Final[str] = "ips/evidence-admission@1"
ADMISSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "cache-admission-record@1"
)
VERIFICATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "evidence-verification@1"
)

# Default closed proof systems admitted by production policy.  Unknown systems
# always reject; later key-registry work may refine the allowlist.
DEFAULT_ALLOWED_PROOF_SYSTEMS: Final[frozenset[str]] = frozenset(
    {
        "integrity",
        "signed_receipt",
        "merkle_manifest_aggregation",
        "groth16",
        "receipt_aggregation",
        "incremental_seal",
    }
)

_NONTERMINAL_STATUSES: Final[frozenset[ProofTerminalStatus]] = frozenset(
    {
        ProofTerminalStatus.UNKNOWN,
        ProofTerminalStatus.TIMEOUT,
        ProofTerminalStatus.UNAVAILABLE,
        ProofTerminalStatus.CANCELLED,
        ProofTerminalStatus.NOT_MODELED,
    }
)

_FAILED_STATUSES: Final[frozenset[ProofTerminalStatus]] = frozenset(
    {
        ProofTerminalStatus.FAILED,
        ProofTerminalStatus.PROOF_FAILED,
        ProofTerminalStatus.INVALID,
        ProofTerminalStatus.STALE,
        ProofTerminalStatus.DISPROVED,
    }
)

_UNSIGNED_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "",
        "unsigned",
        "none",
        "null",
        "n/a",
        "missing",
    }
)

# Language that receipt aggregation must never use for acceptance claims.
_AGGREGATION_FORBIDDEN_CLAIM_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "tests executed",
        "tests ran",
        "test execution",
        "pytest executed",
        "pytest ran",
        "underlying tests ran",
        "direct execution",
        "tests_executed",
    }
)


class AdmissionError(ValueError):
    """Fail-closed admission contract violation."""


class RejectionReason(str, Enum):
    """Stable reason codes for rejected admission decisions."""

    UNKNOWN_PROOF_SYSTEM = "unknown_proof_system"
    MALFORMED_EVIDENCE = "malformed_evidence"
    NONTERMINAL_REQUIRED_UNIT = "nonterminal_required_unit"
    FAILED_REQUIRED_UNIT = "failed_required_unit"
    SIMULATED_REQUIRED_UNIT = "simulated_required_unit"
    UNSIGNED_REQUIRED_RECEIPT = "unsigned_required_receipt"
    PUBLIC_INPUT_MISMATCH = "public_input_mismatch"
    VERIFIER_FAILURE = "verifier_failure"
    UNKNOWN_EVIDENCE_CLASS = "unknown_evidence_class"
    UNALLOWLISTED_SIGNER = "unallowlisted_signer"
    INTEGRITY_MISMATCH = "integrity_mismatch"
    STATUS_CLASS_MISMATCH = "status_class_mismatch"
    OVERCLAIM = "overclaim"


class AdmissionOutcome(str, Enum):
    ADMITTED = "admitted"
    REJECTED = "rejected"


EvidenceRecord = (
    IntegrityCommitment
    | SignedExecutionReceipt
    | ReceiptAggregationZkProof
    | DirectExecutionProof
    | IncrementalCommitSeal
)

# Optional cryptographic / signature verifier hook.
# Returns True on success; False or raised Exception is a verifier failure.
VerifierFn = Callable[[EvidenceRecord, Mapping[str, Any]], bool]


@dataclass(frozen=True, slots=True)
class AdmissionPolicy:
    """Closed allowlists and production flags for evidence admission."""

    allowed_proof_systems: frozenset[str] = DEFAULT_ALLOWED_PROOF_SYSTEMS
    allowed_signers: frozenset[str] = frozenset()
    production: bool = True
    require_signature_for_signed_receipts: bool = True

    def __post_init__(self) -> None:
        systems = self.allowed_proof_systems
        if not isinstance(systems, frozenset) or not systems:
            raise AdmissionError("allowed_proof_systems must be a non-empty frozenset")
        for item in systems:
            if not isinstance(item, str) or not item.strip():
                raise AdmissionError("allowed_proof_systems entries must be non-empty strings")
        signers = self.allowed_signers
        if not isinstance(signers, frozenset):
            raise AdmissionError("allowed_signers must be a frozenset")
        for item in signers:
            if not isinstance(item, str) or not item.strip():
                raise AdmissionError("allowed_signers entries must be non-empty strings")
        if type(self.production) is not bool:
            raise AdmissionError("production must be a boolean")
        if type(self.require_signature_for_signed_receipts) is not bool:
            raise AdmissionError(
                "require_signature_for_signed_receipts must be a boolean"
            )


@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    """Immutable candidate presented for verification and cache admission.

    ``evidence`` is either a closed evidence record or its canonical mapping.
    Unit metadata fields drive required-unit and public-input checks.
    """

    evidence: EvidenceRecord | Mapping[str, Any]
    proof_system_id: str
    public_input_cid: str
    proof_unit_id: str = "unit/unknown"
    proof_object_cid: str = "n/a"
    required_for_seal: bool = True
    proof_mode: ProofMode | str = ProofMode.INTEGRITY_ONLY
    terminal_status: ProofTerminalStatus | str = ProofTerminalStatus.INTEGRITY_VERIFIED
    expected_digest: str | None = None
    observed_digest: str | None = None
    observed_public_input_cid: str | None = None
    logical_epoch: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.proof_system_id, str) or not self.proof_system_id.strip():
            raise AdmissionError("proof_system_id must be a non-empty string")
        if not isinstance(self.public_input_cid, str) or not self.public_input_cid.strip():
            raise AdmissionError("public_input_cid must be a non-empty string")
        if not isinstance(self.proof_unit_id, str) or not self.proof_unit_id.strip():
            raise AdmissionError("proof_unit_id must be a non-empty string")
        if type(self.required_for_seal) is not bool:
            raise AdmissionError("required_for_seal must be a boolean")
        if type(self.logical_epoch) is not int or isinstance(self.logical_epoch, bool):
            raise AdmissionError("logical_epoch must be an int")
        if self.logical_epoch < 0:
            raise AdmissionError("logical_epoch must be non-negative")
        object.__setattr__(self, "proof_system_id", self.proof_system_id.strip())
        object.__setattr__(self, "public_input_cid", self.public_input_cid.strip())
        object.__setattr__(self, "proof_unit_id", self.proof_unit_id.strip())
        object.__setattr__(
            self, "proof_object_cid", str(self.proof_object_cid or "n/a").strip()
        )
        object.__setattr__(self, "proof_mode", _coerce_proof_mode(self.proof_mode))
        object.__setattr__(
            self, "terminal_status", _coerce_terminal_status(self.terminal_status)
        )
        if not isinstance(self.metadata, Mapping):
            raise AdmissionError("metadata must be a mapping")


@dataclass(frozen=True, slots=True)
class CacheAdmissionRecord:
    """Accelerate-issued record proving verification completed before cache write.

    Kit indexes may store this only after accelerate verification.  Lookup still
    requires fresh re-verification; the record is not a trust root.
    """

    schema: str
    proof_unit_id: str
    evidence_class: str
    proof_system_id: str
    proof_object_cid: str
    public_input_cid: str
    verification_digest: str
    establishes: str
    does_not_establish: str
    logical_epoch: int
    verified: bool = True

    def __post_init__(self) -> None:
        if self.verified is not True:
            raise AdmissionError("CacheAdmissionRecord requires verified=True")
        if not self.proof_unit_id or not self.evidence_class:
            raise AdmissionError("CacheAdmissionRecord identity fields are required")
        if not self.verification_digest:
            raise AdmissionError("CacheAdmissionRecord requires verification_digest")

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "proof_unit_id": self.proof_unit_id,
            "evidence_class": self.evidence_class,
            "proof_system_id": self.proof_system_id,
            "proof_object_cid": self.proof_object_cid,
            "public_input_cid": self.public_input_cid,
            "verification_digest": self.verification_digest,
            "establishes": self.establishes,
            "does_not_establish": self.does_not_establish,
            "logical_epoch": self.logical_epoch,
            "verified": True,
            "cache_admission": "verified_only",
        }

    def to_canonical_json(self) -> str:
        return json.dumps(
            self.to_canonical(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """Typed admission outcome with exact establishes / does-not-establish claims."""

    outcome: AdmissionOutcome
    admitted: bool
    reason_code: str | None
    evidence_class: str | None
    establishes: str
    does_not_establish: str
    message: str
    cache_admission_record: CacheAdmissionRecord | None
    proof_unit_id: str
    proof_system_id: str
    public_input_cid: str
    verification_digest: str | None = None

    def __post_init__(self) -> None:
        if self.admitted and self.cache_admission_record is None:
            raise AdmissionError(
                "admitted decisions require a CacheAdmissionRecord"
            )
        if not self.admitted and self.cache_admission_record is not None:
            raise AdmissionError(
                "rejected decisions must not carry a CacheAdmissionRecord"
            )
        if self.admitted and self.reason_code is not None:
            raise AdmissionError("admitted decisions must not carry a rejection reason")
        if not self.admitted and not self.reason_code:
            raise AdmissionError("rejected decisions require a reason_code")

    @property
    def rejected(self) -> bool:
        return not self.admitted

    def to_canonical(self) -> dict[str, Any]:
        record = None
        if self.cache_admission_record is not None:
            record = self.cache_admission_record.to_canonical()
        return {
            "schema": VERIFICATION_SCHEMA,
            "outcome": self.outcome.value,
            "admitted": self.admitted,
            "reason_code": self.reason_code,
            "evidence_class": self.evidence_class,
            "establishes": self.establishes,
            "does_not_establish": self.does_not_establish,
            "message": self.message,
            "proof_unit_id": self.proof_unit_id,
            "proof_system_id": self.proof_system_id,
            "public_input_cid": self.public_input_cid,
            "verification_digest": self.verification_digest,
            "cache_admission_record": record,
            "evidence_subset": EVIDENCE_SUBSET,
        }


def _coerce_proof_mode(value: ProofMode | str) -> ProofMode:
    if isinstance(value, ProofMode):
        return value
    if not isinstance(value, str) or not value.strip():
        raise AdmissionError("proof_mode must be a closed ProofMode string")
    try:
        return ProofMode(value.strip())
    except ValueError as exc:
        raise AdmissionError(f"unknown proof_mode {value!r}") from exc


def _coerce_terminal_status(value: ProofTerminalStatus | str) -> ProofTerminalStatus:
    if isinstance(value, ProofTerminalStatus):
        return value
    if not isinstance(value, str) or not value.strip():
        raise AdmissionError("terminal_status must be a closed status string")
    try:
        return ProofTerminalStatus(value.strip())
    except ValueError as exc:
        raise AdmissionError(f"unknown terminal_status {value!r}") from exc


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_hex(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _claims_for(evidence: EvidenceRecord) -> tuple[str, str]:
    return evidence.ESTABLISHES, evidence.DOES_NOT_ESTABLISH


def _is_unsigned_signature(signature: str) -> bool:
    return signature.strip().casefold() in _UNSIGNED_MARKERS


def _aggregation_claims_test_execution(establishes: str, does_not_establish: str) -> bool:
    establishes_cf = establishes.casefold()
    for token in _AGGREGATION_FORBIDDEN_CLAIM_TOKENS:
        if token in establishes_cf:
            return True
    # Require the class nonclaim to remain present.
    does_cf = does_not_establish.casefold()
    if "tests ran" not in does_cf and "test execution" not in does_cf:
        # Missing nonclaim is treated as an overclaim risk for aggregation.
        return True
    return False


def _default_integrity_verifier(
    evidence: IntegrityCommitment,
    candidate: EvidenceCandidate,
) -> bool:
    expected = candidate.expected_digest or evidence.digest
    observed = candidate.observed_digest
    if observed is None:
        # Without observed bytes, admit only when expected equals declared digest.
        return expected == evidence.digest
    return (
        hmac.compare_digest(str(observed), str(expected))
        and hmac.compare_digest(str(expected), evidence.digest)
    )


def _default_signature_verifier(
    evidence: SignedExecutionReceipt,
    policy: AdmissionPolicy,
) -> bool:
    if _is_unsigned_signature(evidence.signature):
        return False
    if policy.allowed_signers and evidence.signer_id not in policy.allowed_signers:
        return False
    # Structural signature presence + allowlist is the hermetic default.  A
    # production deployment injects a cryptographic VerifierFn.
    return True


def _default_zk_verifier(
    evidence: EvidenceRecord,
    candidate: EvidenceCandidate,
) -> bool:
    # Hermetic default: require proof object identity and public-input binding.
    # Real backends are injected via EvidenceVerifier(verifier=...).
    if isinstance(evidence, DirectExecutionProof):
        if not evidence.proof_cid or evidence.proof_cid == "n/a":
            return False
        if candidate.proof_object_cid not in {"n/a", evidence.proof_cid}:
            if candidate.proof_object_cid != evidence.proof_cid:
                return False
        return True
    if isinstance(evidence, ReceiptAggregationZkProof):
        return bool(evidence.proof_cid) and bool(evidence.receipt_digests)
    if isinstance(evidence, IncrementalCommitSeal):
        return bool(evidence.manifest_cid) and bool(evidence.verification_root)
    return False


class EvidenceVerifier:
    """Verifies closed evidence classes and issues cache-admission decisions."""

    def __init__(
        self,
        policy: AdmissionPolicy | None = None,
        *,
        verifier: VerifierFn | None = None,
    ) -> None:
        self._policy = policy or AdmissionPolicy()
        self._verifier = verifier

    @property
    def policy(self) -> AdmissionPolicy:
        return self._policy

    def verify(self, candidate: EvidenceCandidate) -> AdmissionDecision:
        return self.verify_for_admission(candidate)

    def verify_for_admission(self, candidate: EvidenceCandidate) -> AdmissionDecision:
        if not isinstance(candidate, EvidenceCandidate):
            raise AdmissionError("candidate must be EvidenceCandidate")

        try:
            evidence = self._load_evidence(candidate.evidence)
        except (AdmissionError, EvidenceClassError, TypeError, ValueError) as exc:
            return self._reject(
                candidate,
                RejectionReason.MALFORMED_EVIDENCE,
                message=f"malformed evidence: {exc}",
                evidence_class=None,
                establishes="",
                does_not_establish="any assurance; evidence could not be parsed",
            )

        establishes, does_not_establish = _claims_for(evidence)
        evidence_class = type(evidence).__name__

        # Receipt aggregation never claims tests executed.
        if isinstance(evidence, ReceiptAggregationZkProof):
            if _aggregation_claims_test_execution(establishes, does_not_establish):
                return self._reject(
                    candidate,
                    RejectionReason.OVERCLAIM,
                    message=(
                        "receipt aggregation must not claim tests executed; "
                        "it only establishes admitted receipt-field completeness"
                    ),
                    evidence_class=evidence_class,
                    establishes=establishes,
                    does_not_establish=does_not_establish,
                )

        if candidate.proof_system_id not in self._policy.allowed_proof_systems:
            return self._reject(
                candidate,
                RejectionReason.UNKNOWN_PROOF_SYSTEM,
                message=(
                    f"unknown proof system {candidate.proof_system_id!r}; "
                    f"allowed={sorted(self._policy.allowed_proof_systems)}"
                ),
                evidence_class=evidence_class,
                establishes=establishes,
                does_not_establish=does_not_establish,
            )

        # Required-unit production gates (fail closed).
        if candidate.required_for_seal:
            unit_reject = self._check_required_unit(candidate, evidence, evidence_class)
            if unit_reject is not None:
                return unit_reject

        # Public-input binding.
        observed_pi = candidate.observed_public_input_cid
        if observed_pi is not None and observed_pi != candidate.public_input_cid:
            return self._reject(
                candidate,
                RejectionReason.PUBLIC_INPUT_MISMATCH,
                message=(
                    f"public-input mismatch: expected {candidate.public_input_cid!r}, "
                    f"observed {observed_pi!r}"
                ),
                evidence_class=evidence_class,
                establishes=establishes,
                does_not_establish=does_not_establish,
            )
        if isinstance(evidence, DirectExecutionProof):
            if (
                evidence.input_commitment
                and evidence.input_commitment != candidate.public_input_cid
                and candidate.public_input_cid not in {evidence.input_commitment, "n/a"}
            ):
                # Bind direct-execution public input when both sides are concrete.
                if candidate.metadata.get("bind_direct_public_input", True):
                    declared = candidate.public_input_cid
                    if declared != evidence.input_commitment and not declared.startswith(
                        "statement/"
                    ):
                        # Only reject hard mismatch of digest-shaped values.
                        if declared.startswith("sha256:") and evidence.input_commitment.startswith(
                            "sha256:"
                        ):
                            return self._reject(
                                candidate,
                                RejectionReason.PUBLIC_INPUT_MISMATCH,
                                message=(
                                    "direct-execution public input does not match "
                                    "candidate public_input_cid"
                                ),
                                evidence_class=evidence_class,
                                establishes=establishes,
                                does_not_establish=does_not_establish,
                            )

        class_reject = self._verify_by_class(candidate, evidence, evidence_class)
        if class_reject is not None:
            return class_reject

        # Optional injected cryptographic verifier.
        if self._verifier is not None:
            try:
                ok = self._verifier(evidence, candidate.metadata)
            except Exception as exc:  # noqa: BLE001 - fail closed on any verifier fault
                return self._reject(
                    candidate,
                    RejectionReason.VERIFIER_FAILURE,
                    message=f"verifier failure: {exc}",
                    evidence_class=evidence_class,
                    establishes=establishes,
                    does_not_establish=does_not_establish,
                )
            if ok is not True:
                return self._reject(
                    candidate,
                    RejectionReason.VERIFIER_FAILURE,
                    message="verifier returned non-success",
                    evidence_class=evidence_class,
                    establishes=establishes,
                    does_not_establish=does_not_establish,
                )

        return self._admit(candidate, evidence, establishes, does_not_establish)

    def _load_evidence(
        self, raw: EvidenceRecord | Mapping[str, Any]
    ) -> EvidenceRecord:
        if isinstance(
            raw,
            (
                IntegrityCommitment,
                SignedExecutionReceipt,
                ReceiptAggregationZkProof,
                DirectExecutionProof,
                IncrementalCommitSeal,
            ),
        ):
            return raw
        if not isinstance(raw, Mapping):
            raise AdmissionError("evidence must be a closed record or mapping")
        try:
            return evidence_from_canonical(raw)
        except EvidenceClassError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AdmissionError(f"malformed evidence payload: {exc}") from exc

    def _check_required_unit(
        self,
        candidate: EvidenceCandidate,
        evidence: EvidenceRecord,
        evidence_class: str,
    ) -> AdmissionDecision | None:
        establishes, does_not_establish = _claims_for(evidence)
        mode = candidate.proof_mode
        status = candidate.terminal_status

        if mode is ProofMode.SIMULATED or status is ProofTerminalStatus.SIMULATED:
            return self._reject(
                candidate,
                RejectionReason.SIMULATED_REQUIRED_UNIT,
                message=(
                    "simulated required units cannot be admitted under production "
                    "policy and never become direct-execution evidence"
                ),
                evidence_class=evidence_class,
                establishes=establishes,
                does_not_establish=does_not_establish,
            )

        if status in _NONTERMINAL_STATUSES:
            return self._reject(
                candidate,
                RejectionReason.NONTERMINAL_REQUIRED_UNIT,
                message=f"required unit is nonterminal: {status.value}",
                evidence_class=evidence_class,
                establishes=establishes,
                does_not_establish=does_not_establish,
            )

        if status in _FAILED_STATUSES:
            return self._reject(
                candidate,
                RejectionReason.FAILED_REQUIRED_UNIT,
                message=f"required unit failed: {status.value}",
                evidence_class=evidence_class,
                establishes=establishes,
                does_not_establish=does_not_establish,
            )

        # Map evidence record to EvidenceClass for status satisfaction.
        try:
            eclass = EvidenceClass(evidence_class)
        except ValueError:
            return self._reject(
                candidate,
                RejectionReason.UNKNOWN_EVIDENCE_CLASS,
                message=f"unknown evidence class {evidence_class!r}",
                evidence_class=evidence_class,
                establishes=establishes,
                does_not_establish=does_not_establish,
            )

        if not status_satisfies_class(status, eclass):
            return self._reject(
                candidate,
                RejectionReason.STATUS_CLASS_MISMATCH,
                message=(
                    f"terminal status {status.value!r} does not satisfy "
                    f"evidence class {eclass.value!r}"
                ),
                evidence_class=evidence_class,
                establishes=establishes,
                does_not_establish=does_not_establish,
            )
        return None

    def _verify_by_class(
        self,
        candidate: EvidenceCandidate,
        evidence: EvidenceRecord,
        evidence_class: str,
    ) -> AdmissionDecision | None:
        establishes, does_not_establish = _claims_for(evidence)

        if isinstance(evidence, IntegrityCommitment):
            if not _default_integrity_verifier(evidence, candidate):
                return self._reject(
                    candidate,
                    RejectionReason.INTEGRITY_MISMATCH,
                    message="integrity digest/CID rehash check failed",
                    evidence_class=evidence_class,
                    establishes=establishes,
                    does_not_establish=does_not_establish,
                )
            return None

        if isinstance(evidence, SignedExecutionReceipt):
            if (
                self._policy.require_signature_for_signed_receipts
                and _is_unsigned_signature(evidence.signature)
            ):
                return self._reject(
                    candidate,
                    RejectionReason.UNSIGNED_REQUIRED_RECEIPT,
                    message="required signed receipt is unsigned",
                    evidence_class=evidence_class,
                    establishes=establishes,
                    does_not_establish=does_not_establish,
                )
            if (
                self._policy.allowed_signers
                and evidence.signer_id not in self._policy.allowed_signers
            ):
                return self._reject(
                    candidate,
                    RejectionReason.UNALLOWLISTED_SIGNER,
                    message=f"signer {evidence.signer_id!r} is not allowlisted",
                    evidence_class=evidence_class,
                    establishes=establishes,
                    does_not_establish=does_not_establish,
                )
            if not _default_signature_verifier(evidence, self._policy):
                return self._reject(
                    candidate,
                    RejectionReason.VERIFIER_FAILURE,
                    message="signed receipt verification failed",
                    evidence_class=evidence_class,
                    establishes=establishes,
                    does_not_establish=does_not_establish,
                )
            return None

        if isinstance(
            evidence,
            (DirectExecutionProof, ReceiptAggregationZkProof, IncrementalCommitSeal),
        ):
            # When no external verifier is injected, use hermetic structural check.
            if self._verifier is None and not _default_zk_verifier(evidence, candidate):
                return self._reject(
                    candidate,
                    RejectionReason.VERIFIER_FAILURE,
                    message="structural cryptographic verification failed",
                    evidence_class=evidence_class,
                    establishes=establishes,
                    does_not_establish=does_not_establish,
                )
            return None

        return self._reject(
            candidate,
            RejectionReason.UNKNOWN_EVIDENCE_CLASS,
            message=f"unsupported evidence class {evidence_class!r}",
            evidence_class=evidence_class,
            establishes=establishes,
            does_not_establish=does_not_establish,
        )

    def _admit(
        self,
        candidate: EvidenceCandidate,
        evidence: EvidenceRecord,
        establishes: str,
        does_not_establish: str,
    ) -> AdmissionDecision:
        evidence_class = type(evidence).__name__
        # Never upgrade lower classes to direct-execution claim language.
        if not isinstance(evidence, DirectExecutionProof):
            if "direct_computation_claim" in candidate.metadata:
                if candidate.metadata.get("direct_computation_claim") is True:
                    return self._reject(
                        candidate,
                        RejectionReason.OVERCLAIM,
                        message=(
                            f"{evidence_class} cannot claim direct computation; "
                            "only DirectExecutionProof may"
                        ),
                        evidence_class=evidence_class,
                        establishes=establishes,
                        does_not_establish=does_not_establish,
                    )

        preimage = _canonical_json(
            {
                "evidence_class": evidence_class,
                "proof_unit_id": candidate.proof_unit_id,
                "proof_system_id": candidate.proof_system_id,
                "public_input_cid": candidate.public_input_cid,
                "proof_object_cid": candidate.proof_object_cid,
                "establishes": establishes,
                "does_not_establish": does_not_establish,
                "logical_epoch": candidate.logical_epoch,
                "evidence": evidence.to_canonical(),
            }
        )
        verification_digest = _sha256_hex(preimage)
        record = CacheAdmissionRecord(
            schema=ADMISSION_SCHEMA,
            proof_unit_id=candidate.proof_unit_id,
            evidence_class=evidence_class,
            proof_system_id=candidate.proof_system_id,
            proof_object_cid=candidate.proof_object_cid,
            public_input_cid=candidate.public_input_cid,
            verification_digest=verification_digest,
            establishes=establishes,
            does_not_establish=does_not_establish,
            logical_epoch=candidate.logical_epoch,
            verified=True,
        )
        return AdmissionDecision(
            outcome=AdmissionOutcome.ADMITTED,
            admitted=True,
            reason_code=None,
            evidence_class=evidence_class,
            establishes=establishes,
            does_not_establish=does_not_establish,
            message="verified; cache admission record issued",
            cache_admission_record=record,
            proof_unit_id=candidate.proof_unit_id,
            proof_system_id=candidate.proof_system_id,
            public_input_cid=candidate.public_input_cid,
            verification_digest=verification_digest,
        )

    def _reject(
        self,
        candidate: EvidenceCandidate,
        reason: RejectionReason,
        *,
        message: str,
        evidence_class: str | None,
        establishes: str,
        does_not_establish: str,
    ) -> AdmissionDecision:
        return AdmissionDecision(
            outcome=AdmissionOutcome.REJECTED,
            admitted=False,
            reason_code=reason.value,
            evidence_class=evidence_class,
            establishes=establishes,
            does_not_establish=does_not_establish,
            message=message,
            cache_admission_record=None,
            proof_unit_id=candidate.proof_unit_id,
            proof_system_id=candidate.proof_system_id,
            public_input_cid=candidate.public_input_cid,
            verification_digest=None,
        )


def verify_for_admission(
    candidate: EvidenceCandidate,
    policy: AdmissionPolicy | None = None,
    *,
    verifier: VerifierFn | None = None,
) -> AdmissionDecision:
    """Verify closed evidence and decide cache admission.

    Returns an :class:`AdmissionDecision`.  A
    :class:`CacheAdmissionRecord` is present only when ``admitted`` is True.
    """

    return EvidenceVerifier(policy=policy, verifier=verifier).verify_for_admission(
        candidate
    )


def issue_cache_admission_record(
    decision: AdmissionDecision,
) -> CacheAdmissionRecord:
    """Return the verified admission record or raise if verification failed."""

    if not isinstance(decision, AdmissionDecision):
        raise AdmissionError("decision must be AdmissionDecision")
    if not decision.admitted or decision.cache_admission_record is None:
        raise AdmissionError(
            "cache admission record requires successful verification; "
            f"reason={decision.reason_code!r}"
        )
    return decision.cache_admission_record


def closed_rejection_reasons() -> frozenset[str]:
    return frozenset(item.value for item in RejectionReason)


__all__ = (
    "ADMISSION_SCHEMA",
    "DEFAULT_ALLOWED_PROOF_SYSTEMS",
    "EVIDENCE_SUBSET",
    "VERIFICATION_SCHEMA",
    "AdmissionDecision",
    "AdmissionError",
    "AdmissionOutcome",
    "AdmissionPolicy",
    "CacheAdmissionRecord",
    "EvidenceCandidate",
    "EvidenceVerifier",
    "RejectionReason",
    "VerifierFn",
    "closed_rejection_reasons",
    "issue_cache_admission_record",
    "verify_for_admission",
)
