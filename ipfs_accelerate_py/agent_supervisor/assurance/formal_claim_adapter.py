"""Compatibility adapter: supervisor legacy assurance records → FCA envelopes.

FACP-014. Normative vocabulary is ``facp/formal-claim-algebra-v1@1``. This
module is a conservative projection only: legacy total ladders may inform at
most the dimensions they historically carried; every other EvidenceEnvelope
dimension stays at its weakest honest default (unchecked / absent / none /
stale / not_started / hermetic / unreviewed). Reverse projection refuses any
collapse that would lose non-ladder dimensions or invent a stronger legacy
rank than the envelope supports.

This adapter does **not** claim deprecation of the legacy ladders and does not
edit them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Final, Iterable, Mapping, Optional, Sequence, Tuple, Union

from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
    ProofStatus,
)
from ipfs_accelerate_py.agent_supervisor.proof import database_repair_evidence as _repair_evidence
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import (
    ProofCacheEntry,
    UntrustedDraftCacheEntry,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    EvidenceFreshness,
    ProofReceipt,
    ProofVerdict,
)

VOCAB_SCHEMA: Final[str] = "facp/formal-claim-algebra-v1@1"
ADAPTER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/formal-claim-adapter@1"
)
TASK_ID: Final[str] = "FACP-014"
GOAL_ID: Final[str] = "FACP-G120"
BUNDLE: Final[str] = "facp/fca/accelerate-adapter"
UNSAFE_PROMOTION_DEFAULT: Final[bool] = False

DIMENSION_ORDER: Final[Tuple[str, ...]] = (
    "origin",
    "integrity",
    "authority",
    "policy",
    "proof",
    "freshness",
    "effect",
    "environment",
    "review",
)

# Seed maps from facp/formal-claim-algebra-v1@1 §8 / machine-readable appendix.
_ASSURANCE_LEVEL_PROOF_MAP: Final[Mapping[str, str]] = {
    "unverified": "none",
    "none": "none",
    "candidate": "candidate",
    "solver_checked": "candidate",
    "solver_verified": "candidate",
    "kernel_verified": "verified",
    "attested": "verified",
}

_REPAIR_ASSURANCE_PROOF_MAP: Final[Mapping[str, str]] = {
    "none": "none",
    "heuristic": "candidate",
    "validated": "candidate",
    "solver_checked": "candidate",
    "kernel_verified": "verified",
    "attested": "verified",
}

_PROOF_STATUS_MAP: Final[Mapping[str, Mapping[str, str]]] = {
    "unproved": {"proof": "none"},
    "candidate": {"proof": "candidate"},
    "solver_checked": {"proof": "candidate"},
    "kernel_verified": {"proof": "verified"},
    "validated_refuted": {"proof": "refuted"},
    "inconclusive": {"proof": "unknown"},
    "unsupported": {"proof": "verifier_unavailable"},
    "stale": {"proof": "unknown", "freshness": "stale"},
    "error": {"proof": "unknown"},
}

# EvidenceTier vocabulary (string form). The enum module has a circular import
# graph in some checkouts; adapters accept the enum instance or these spellings.
_EVIDENCE_TIER_PROOF_MAP: Final[Mapping[str, str]] = {
    # Ceiling vocabulary → proof only; bare tier is not a kernel receipt.
    "query_fact": "none",
    "graphrag_fact": "none",
    "observation": "candidate",
    "solver_candidate": "candidate",
    # Conservatively keep kernel/attestation *tiers* at candidate unless a
    # typed ProofReceipt supplies independent acceptance (see adapt_proof_receipt).
    "kernel_proof": "candidate",
    "cryptographic_attestation": "candidate",
}

# Reverse: FCA proof → formal_verification_contracts.AssuranceLevel.
# Never emit ATTESTED from proof.verified alone (attestation is not encoded in
# the proof dimension; choosing ATTESTED would be information-losing promotion).
_PROOF_TO_ASSURANCE_LEVEL: Final[Mapping[str, str]] = {
    "none": AssuranceLevel.UNVERIFIED.value,
    "candidate": AssuranceLevel.CANDIDATE.value,
    "verified": AssuranceLevel.KERNEL_VERIFIED.value,
}


class Origin(str, Enum):
    ABSENT = "absent"
    DECLARED = "declared"
    FIXTURE = "fixture"
    SIMULATED = "simulated"
    HERMETIC_OBSERVED = "hermetic_observed"
    LIVE_OBSERVED = "live_observed"


class Integrity(str, Enum):
    UNCHECKED = "unchecked"
    STRUCTURALLY_VALID = "structurally_valid"
    DIGEST_VALID = "digest_valid"
    SIGNATURE_VALID = "signature_valid"


class Authority(str, Enum):
    UNCHECKED = "unchecked"
    ABSENT = "absent"
    VALID = "valid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    DENIED = "denied"


class Policy(str, Enum):
    UNCHECKED = "unchecked"
    ALLOWED = "allowed"
    DENIED = "denied"
    ALLOWED_WITH_OBLIGATIONS = "allowed_with_obligations"
    INDETERMINATE = "indeterminate"


class Proof(str, Enum):
    NONE = "none"
    CANDIDATE = "candidate"
    VERIFIED = "verified"
    REFUTED = "refuted"
    UNKNOWN = "unknown"
    VERIFIER_UNAVAILABLE = "verifier_unavailable"


class Freshness(str, Enum):
    CURRENT = "current"
    STALE = "stale"
    SUPERSEDED = "superseded"
    WITHDRAWN = "withdrawn"


class Effect(str, Enum):
    NOT_STARTED = "not_started"
    RESERVED = "reserved"
    STARTED = "started"
    EXTERNALLY_UNKNOWN = "externally_unknown"
    OBSERVED = "observed"
    COMPENSATED = "compensated"
    FAILED = "failed"


class Environment(str, Enum):
    HERMETIC = "hermetic"
    CONDITIONAL = "conditional"
    LIVE = "live"


class Review(str, Enum):
    UNREVIEWED = "unreviewed"
    MACHINE_REVIEWED = "machine_reviewed"
    HUMAN_REVIEWED = "human_reviewed"


_DIMENSION_ENUMS: Final[Mapping[str, type[Enum]]] = {
    "origin": Origin,
    "integrity": Integrity,
    "authority": Authority,
    "policy": Policy,
    "proof": Proof,
    "freshness": Freshness,
    "effect": Effect,
    "environment": Environment,
    "review": Review,
}


@dataclass(frozen=True)
class EvidenceEnvelope:
    """Closed FCA evidence product (nine dimensions)."""

    origin: Origin = Origin.ABSENT
    integrity: Integrity = Integrity.UNCHECKED
    authority: Authority = Authority.UNCHECKED
    policy: Policy = Policy.UNCHECKED
    proof: Proof = Proof.NONE
    freshness: Freshness = Freshness.STALE
    effect: Effect = Effect.NOT_STARTED
    environment: Environment = Environment.HERMETIC
    review: Review = Review.UNREVIEWED

    @classmethod
    def weakest(cls) -> "EvidenceEnvelope":
        """Weakest honest defaults for every dimension (fail-closed start)."""

        return cls()

    def to_dict(self) -> dict[str, str]:
        return {
            "origin": self.origin.value,
            "integrity": self.integrity.value,
            "authority": self.authority.value,
            "policy": self.policy.value,
            "proof": self.proof.value,
            "freshness": self.freshness.value,
            "effect": self.effect.value,
            "environment": self.environment.value,
            "review": self.review.value,
        }

    def informed_overrides(self, weakest: Optional["EvidenceEnvelope"] = None) -> dict[str, str]:
        """Return only dimensions that differ from weakest defaults."""

        base = weakest or EvidenceEnvelope.weakest()
        out: dict[str, str] = {}
        for name in DIMENSION_ORDER:
            current = getattr(self, name)
            default = getattr(base, name)
            if current != default:
                out[name] = current.value if isinstance(current, Enum) else str(current)
        return out

    def with_updates(self, **updates: Union[str, Enum]) -> "EvidenceEnvelope":
        normalized: dict[str, Enum] = {}
        for key, raw in updates.items():
            if key not in _DIMENSION_ENUMS:
                raise FormalClaimAdapterError(
                    code="unknown_dimension",
                    message=f"unknown evidence dimension: {key!r}",
                    legacy_kind="envelope_update",
                )
            enum_type = _DIMENSION_ENUMS[key]
            if isinstance(raw, enum_type):
                normalized[key] = raw
            else:
                try:
                    normalized[key] = enum_type(str(raw))
                except ValueError as exc:
                    raise FormalClaimAdapterError(
                        code="unknown_enum_value",
                        message=f"unknown {key} value: {raw!r}",
                        legacy_kind="envelope_update",
                    ) from exc
        return replace(self, **normalized)


class FormalClaimAdapterError(ValueError):
    """Fail-closed adapter rejection (malformed input or illegal construction)."""

    def __init__(self, *, code: str, message: str, legacy_kind: str = "") -> None:
        self.code = str(code)
        self.legacy_kind = str(legacy_kind or "")
        super().__init__(message)


@dataclass(frozen=True)
class TypedIncompatibility:
    """Legacy record that cannot safely project into an FCA envelope (or reverse)."""

    code: str
    legacy_kind: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)
    unsafe_promotion: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ADAPTER_SCHEMA,
            "vocab_schema": VOCAB_SCHEMA,
            "kind": "typed_incompatibility",
            "code": self.code,
            "legacy_kind": self.legacy_kind,
            "message": self.message,
            "details": dict(self.details),
            "unsafe_promotion": bool(self.unsafe_promotion),
            "task_id": TASK_ID,
        }


@dataclass(frozen=True)
class EnvelopeAdaptation:
    """Successful conservative adaptation of one legacy record."""

    envelope: EvidenceEnvelope
    legacy_kind: str
    informed_dimensions: Tuple[str, ...]
    source_ref: str = ""
    notes: Tuple[str, ...] = ()
    unsafe_promotion: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ADAPTER_SCHEMA,
            "vocab_schema": VOCAB_SCHEMA,
            "kind": "envelope_adaptation",
            "legacy_kind": self.legacy_kind,
            "informed_dimensions": list(self.informed_dimensions),
            "envelope": self.envelope.to_dict(),
            "informed_overrides": self.envelope.informed_overrides(),
            "source_ref": self.source_ref,
            "notes": list(self.notes),
            "unsafe_promotion": bool(self.unsafe_promotion),
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
            "bundle": BUNDLE,
        }


AdaptationResult = Union[EnvelopeAdaptation, TypedIncompatibility]


def _token(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value or "").strip()


def _legacy_kind_name(record: Any) -> str:
    if isinstance(record, type) and issubclass(record, Enum):
        return record.__name__
    if isinstance(record, Enum):
        return type(record).__name__
    if isinstance(record, type):
        return record.__name__
    return type(record).__name__


def _success(
    envelope: EvidenceEnvelope,
    *,
    legacy_kind: str,
    informed: Iterable[str],
    source_ref: str = "",
    notes: Sequence[str] = (),
) -> EnvelopeAdaptation:
    informed_dims = tuple(name for name in DIMENSION_ORDER if name in set(informed))
    # Guard: never silently mark unsafe promotion.
    return EnvelopeAdaptation(
        envelope=envelope,
        legacy_kind=legacy_kind,
        informed_dimensions=informed_dims,
        source_ref=source_ref,
        notes=tuple(notes),
        unsafe_promotion=UNSAFE_PROMOTION_DEFAULT,
    )


def _incompat(
    *,
    code: str,
    legacy_kind: str,
    message: str,
    details: Optional[Mapping[str, Any]] = None,
    unsafe_promotion: bool = False,
) -> TypedIncompatibility:
    return TypedIncompatibility(
        code=code,
        legacy_kind=legacy_kind,
        message=message,
        details=dict(details or {}),
        unsafe_promotion=bool(unsafe_promotion),
    )


def _envelope_from_proof(
    proof: str,
    *,
    freshness: Optional[str] = None,
) -> EvidenceEnvelope:
    updates: dict[str, str] = {"proof": proof}
    if freshness is not None:
        updates["freshness"] = freshness
    return EvidenceEnvelope.weakest().with_updates(**updates)


def adapt_assurance_level(
    level: Union[AssuranceLevel, str],
) -> AdaptationResult:
    """Map formal_verification_contracts.AssuranceLevel → proof only."""

    raw = _token(level).casefold()
    # Compatibility aliases share values with primary spellings.
    if raw not in _ASSURANCE_LEVEL_PROOF_MAP:
        return _incompat(
            code="unknown_assurance_level",
            legacy_kind="AssuranceLevel",
            message=f"unknown AssuranceLevel value: {level!r}",
            details={"value": raw},
        )
    proof = _ASSURANCE_LEVEL_PROOF_MAP[raw]
    notes = ()
    if raw == "attested":
        notes = (
            "attested maps to proof.verified only; authority remains unchecked",
        )
    return _success(
        _envelope_from_proof(proof),
        legacy_kind="AssuranceLevel",
        informed=("proof",),
        source_ref=f"AssuranceLevel:{raw}",
        notes=notes,
    )


def adapt_database_repair_assurance_level(
    level: Union[_repair_evidence.AssuranceLevel, str],
) -> AdaptationResult:
    """Map database-repair AssuranceLevel → proof only (heuristic never verified)."""

    raw = _token(level).casefold()
    if raw not in _REPAIR_ASSURANCE_PROOF_MAP:
        return _incompat(
            code="unknown_repair_assurance_level",
            legacy_kind="database_repair.AssuranceLevel",
            message=f"unknown database-repair AssuranceLevel value: {level!r}",
            details={"value": raw},
        )
    proof = _REPAIR_ASSURANCE_PROOF_MAP[raw]
    notes = ()
    if raw in {"heuristic", "validated"}:
        notes = (
            f"{raw} maps to proof.candidate; never proof.verified",
        )
    return _success(
        _envelope_from_proof(proof),
        legacy_kind="database_repair.AssuranceLevel",
        informed=("proof",),
        source_ref=f"database_repair.AssuranceLevel:{raw}",
        notes=notes,
    )


def adapt_proof_status(status: Union[ProofStatus, str]) -> AdaptationResult:
    """Map ProofStatus → proof and, when STALE, freshness.stale."""

    raw = _token(status).casefold()
    mapped = _PROOF_STATUS_MAP.get(raw)
    if mapped is None:
        return _incompat(
            code="unknown_proof_status",
            legacy_kind="ProofStatus",
            message=f"unknown ProofStatus value: {status!r}",
            details={"value": raw},
        )
    informed = tuple(mapped.keys())
    return _success(
        EvidenceEnvelope.weakest().with_updates(**mapped),
        legacy_kind="ProofStatus",
        informed=informed,
        source_ref=f"ProofStatus:{raw}",
    )


def adapt_evidence_tier(tier: Any) -> AdaptationResult:
    """Map EvidenceTier ladder → proof only (bare tier is not a receipt)."""

    raw = _token(tier).casefold()
    proof = _EVIDENCE_TIER_PROOF_MAP.get(raw)
    if proof is None:
        return _incompat(
            code="unknown_evidence_tier",
            legacy_kind="EvidenceTier",
            message=f"unknown EvidenceTier value: {tier!r}",
            details={"value": raw},
        )
    notes = ()
    if raw in {"kernel_proof", "cryptographic_attestation"}:
        notes = (
            "bare EvidenceTier does not mint proof.verified; "
            "supply a ProofReceipt for kernel/attestation acceptance",
        )
    return _success(
        _envelope_from_proof(proof),
        legacy_kind="EvidenceTier",
        informed=("proof",),
        source_ref=f"EvidenceTier:{raw}",
        notes=notes,
    )


def _freshness_from_evidence_freshness(value: EvidenceFreshness | str) -> str:
    raw = _token(value).casefold()
    if raw == EvidenceFreshness.CURRENT.value:
        return Freshness.CURRENT.value
    if raw == EvidenceFreshness.STALE.value:
        return Freshness.STALE.value
    # UNKNOWN → weakest honest freshness (stale), never current.
    return Freshness.STALE.value


def adapt_proof_receipt(receipt: ProofReceipt) -> AdaptationResult:
    """Map ProofReceipt → proof + freshness from authoritative projections."""

    if not isinstance(receipt, ProofReceipt):
        return _incompat(
            code="not_a_proof_receipt",
            legacy_kind="ProofReceipt",
            message="adapt_proof_receipt requires a ProofReceipt instance",
        )

    authoritative = receipt.authoritative_assurance
    level_result = adapt_assurance_level(authoritative)
    if isinstance(level_result, TypedIncompatibility):
        return level_result

    proof = level_result.envelope.proof.value
    notes: list[str] = [
        "provider_claimed_assurance is ignored for envelope proof",
    ]

    # Verdict can strengthen refutation independently of assurance projection.
    verdict = receipt.authoritative_verdict
    if verdict is ProofVerdict.DISPROVED:
        proof = Proof.REFUTED.value
        notes.append("authoritative disproved verdict → proof.refuted")
    elif verdict is ProofVerdict.INCONCLUSIVE:
        if proof == Proof.NONE.value:
            proof = Proof.UNKNOWN.value
            notes.append("inconclusive verdict → proof.unknown")
    elif verdict in {ProofVerdict.UNSUPPORTED, ProofVerdict.ERROR}:
        if proof in {Proof.NONE.value, Proof.CANDIDATE.value}:
            proof = (
                Proof.VERIFIER_UNAVAILABLE.value
                if verdict is ProofVerdict.UNSUPPORTED
                else Proof.UNKNOWN.value
            )

    freshness = _freshness_from_evidence_freshness(receipt.freshness)
    if freshness != Freshness.CURRENT.value:
        # Stale/unknown receipts must not keep a stronger currency claim.
        notes.append(f"receipt freshness → freshness.{freshness}")

    informed: list[str] = ["proof"]
    updates: dict[str, str] = {"proof": proof, "freshness": freshness}
    informed.append("freshness")

    return _success(
        EvidenceEnvelope.weakest().with_updates(**updates),
        legacy_kind="ProofReceipt",
        informed=tuple(informed),
        source_ref=f"ProofReceipt:{receipt.receipt_id}",
        notes=tuple(notes),
    )


def adapt_proof_cache_entry(
    entry: ProofCacheEntry,
    *,
    now_ms: Optional[int] = None,
) -> AdaptationResult:
    """Map a proof-cache entry via its receipt; expired/incomplete → stale."""

    if not isinstance(entry, ProofCacheEntry):
        return _incompat(
            code="not_a_proof_cache_entry",
            legacy_kind="ProofCacheEntry",
            message="adapt_proof_cache_entry requires a ProofCacheEntry",
        )

    base = adapt_proof_receipt(entry.receipt)
    if isinstance(base, TypedIncompatibility):
        return base

    notes = list(base.notes)
    envelope = base.envelope
    informed = set(base.informed_dimensions)

    expired = False
    if now_ms is not None:
        try:
            clock = int(now_ms)
        except (TypeError, ValueError):
            return _incompat(
                code="invalid_now_ms",
                legacy_kind="ProofCacheEntry",
                message="now_ms must be an integer epoch-millis value",
            )
        if clock > int(entry.expires_at_ms):
            expired = True

    if expired or not entry.complete:
        envelope = envelope.with_updates(freshness=Freshness.STALE)
        informed.add("freshness")
        notes.append(
            "expired_or_incomplete_cache_entry → freshness.stale"
            if expired
            else "incomplete_cache_entry → freshness.stale"
        )

    return _success(
        envelope,
        legacy_kind="ProofCacheEntry",
        informed=tuple(informed),
        source_ref=f"ProofCacheEntry:{entry.key.key_id}",
        notes=tuple(notes),
    )


def adapt_untrusted_draft_cache_entry(
    entry: UntrustedDraftCacheEntry,
) -> AdaptationResult:
    """Draft cache entries never mint verified / current production proof."""

    if not isinstance(entry, UntrustedDraftCacheEntry):
        return _incompat(
            code="not_an_untrusted_draft_cache_entry",
            legacy_kind="UntrustedDraftCacheEntry",
            message="adapt_untrusted_draft_cache_entry requires UntrustedDraftCacheEntry",
        )
    # Draft material is at most a candidate under declared/simulated origin.
    return _success(
        EvidenceEnvelope.weakest().with_updates(
            origin=Origin.DECLARED,
            proof=Proof.CANDIDATE,
            freshness=Freshness.STALE,
        ),
        legacy_kind="UntrustedDraftCacheEntry",
        informed=("origin", "proof", "freshness"),
        source_ref="UntrustedDraftCacheEntry",
        notes=(
            "untrusted draft cache cannot satisfy proof.verified or freshness.current",
        ),
    )


def _is_named_type(obj: Any, type_name: str) -> bool:
    return type(obj).__name__ == type_name


def _permit_fields(permit: Any) -> Optional[tuple[int, int, str]]:
    """Extract issued/expires/admission_receipt from a permit or mapping."""

    if isinstance(permit, Mapping):
        try:
            issued = int(permit["issued_at_ms"])
            expires = int(permit["expires_at_ms"])
        except (KeyError, TypeError, ValueError):
            return None
        receipt = str(permit.get("admission_receipt_id") or "")
        return issued, expires, receipt
    try:
        issued = int(permit.issued_at_ms)
        expires = int(permit.expires_at_ms)
        receipt = str(getattr(permit, "admission_receipt_id", "") or "")
    except (AttributeError, TypeError, ValueError):
        return None
    return issued, expires, receipt


def adapt_execution_permit(
    permit: Any,
    *,
    now_ms: Optional[int] = None,
) -> AdaptationResult:
    """Map ExecutionPermit → authority (+ freshness when expired).

    Host policy remains unchecked: carrying ``policy_id`` is not a completed
    host-policy evaluation result. Effect stays ``not_started``.

    Accepts a real ``ExecutionPermit`` or a compact mapping with
    ``issued_at_ms`` / ``expires_at_ms`` (and optional ``admission_receipt_id``).
    """

    fields = _permit_fields(permit)
    if fields is None and not (
        _is_named_type(permit, "ExecutionPermit") or isinstance(permit, Mapping)
    ):
        return _incompat(
            code="not_an_execution_permit",
            legacy_kind="ExecutionPermit",
            message=(
                "adapt_execution_permit requires an ExecutionPermit or mapping "
                "with issued_at_ms/expires_at_ms"
            ),
        )
    if fields is None:
        return _incompat(
            code="malformed_execution_permit",
            legacy_kind="ExecutionPermit",
            message="execution permit is missing issued_at_ms/expires_at_ms",
        )

    issued_at_ms, expires_at_ms, admission_receipt_id = fields
    clock = int(now_ms) if now_ms is not None else issued_at_ms
    notes: list[str] = [
        "policy_id does not set policy.allowed",
        "permit does not set effect.observed",
    ]
    informed: list[str] = ["authority"]

    if clock > expires_at_ms:
        updates = {
            "authority": Authority.EXPIRED.value,
            "freshness": Freshness.STALE.value,
        }
        informed.append("freshness")
        notes.append("expired permit → authority.expired + freshness.stale")
    elif clock < issued_at_ms:
        updates = {"authority": Authority.ABSENT.value}
        notes.append("not-yet-valid permit → authority.absent")
    else:
        updates = {"authority": Authority.VALID.value}
        notes.append("current permit → authority.valid only")

    return _success(
        EvidenceEnvelope.weakest().with_updates(**updates),
        legacy_kind="ExecutionPermit",
        informed=tuple(informed),
        source_ref=f"ExecutionPermit:{admission_receipt_id or 'anonymous'}",
        notes=tuple(notes),
    )


def adapt_provider_capability_evidence(evidence: Any) -> AdaptationResult:
    """Map provider capability evidence as discovery — never live qualification."""

    if isinstance(evidence, Mapping):
        provider_id = str(evidence.get("provider_id") or "unknown")
        ready = bool(evidence.get("ready", False))
    elif _is_named_type(evidence, "ProviderCapabilityEvidence") or hasattr(
        evidence, "provider_id"
    ):
        provider_id = str(getattr(evidence, "provider_id", "unknown"))
        ready = bool(getattr(evidence, "ready", False))
    else:
        return _incompat(
            code="not_provider_capability_evidence",
            legacy_kind="ProviderCapabilityEvidence",
            message=(
                "adapt_provider_capability_evidence requires "
                "ProviderCapabilityEvidence or a compact mapping"
            ),
        )

    notes = [
        "capability readiness is discovery/inventory only",
        "never sets origin.live_observed or environment.live qualification",
    ]
    if not ready:
        notes.append("provider not ready; envelope stays non-production")
    return _success(
        EvidenceEnvelope.weakest().with_updates(origin=Origin.DECLARED),
        legacy_kind="ProviderCapabilityEvidence",
        informed=("origin",),
        source_ref=f"ProviderCapabilityEvidence:{provider_id}",
        notes=tuple(notes),
    )


def adapt_capability_evidence(evidence: Any) -> AdaptationResult:
    """Map a capability evidence bundle as discovery (product stays weak)."""

    if isinstance(evidence, Mapping):
        providers = evidence.get("providers") or {}
        attempt_cid = str(evidence.get("attempt_cid") or "")
        if isinstance(providers, Mapping):
            ready_count = sum(
                1
                for item in providers.values()
                if (isinstance(item, Mapping) and item.get("ready"))
                or getattr(item, "ready", False)
            )
        else:
            ready_count = 0
    elif _is_named_type(evidence, "CapabilityEvidence") or hasattr(
        evidence, "providers"
    ):
        providers = getattr(evidence, "providers", {}) or {}
        attempt_cid = str(getattr(evidence, "attempt_cid", "") or "")
        ready_count = sum(
            1 for item in getattr(providers, "values", lambda: [])() if getattr(item, "ready", False)
        )
    else:
        return _incompat(
            code="not_capability_evidence",
            legacy_kind="CapabilityEvidence",
            message=(
                "adapt_capability_evidence requires CapabilityEvidence or a "
                "compact mapping"
            ),
        )

    notes = [
        "CapabilityEvidence is discovery/inventory; not live qualification",
        f"ready_providers={ready_count}",
    ]
    if ready_count:
        notes.append("ready providers do not set production_supported dimensions")
    return _success(
        EvidenceEnvelope.weakest().with_updates(origin=Origin.DECLARED),
        legacy_kind="CapabilityEvidence",
        informed=("origin",),
        source_ref=f"CapabilityEvidence:{attempt_cid or 'anonymous'}",
        notes=tuple(notes),
    )


def adapt_stale_or_unknown_marker(
    marker: Union[str, EvidenceFreshness, ProofStatus],
) -> AdaptationResult:
    """Map explicit stale/unknown markers without inventing currency or proof."""

    raw = _token(marker).casefold()
    if raw in {"stale", EvidenceFreshness.STALE.value, ProofStatus.STALE.value}:
        return _success(
            EvidenceEnvelope.weakest().with_updates(
                proof=Proof.UNKNOWN,
                freshness=Freshness.STALE,
            ),
            legacy_kind="stale_marker",
            informed=("proof", "freshness"),
            source_ref=f"stale:{raw}",
            notes=("stale marker cannot become freshness.current",),
        )
    if raw in {"unknown", EvidenceFreshness.UNKNOWN.value}:
        return _success(
            EvidenceEnvelope.weakest().with_updates(
                proof=Proof.UNKNOWN,
                freshness=Freshness.STALE,
            ),
            legacy_kind="unknown_marker",
            informed=("proof", "freshness"),
            source_ref=f"unknown:{raw}",
            notes=("unknown currency stays freshness.stale",),
        )
    return _incompat(
        code="unrecognized_stale_unknown_marker",
        legacy_kind="stale_or_unknown_marker",
        message=f"unrecognized stale/unknown marker: {marker!r}",
        details={"value": raw},
    )


_FORBIDDEN_GENERIC_FIELDS: Final[frozenset[str]] = frozenset(
    {"success", "available", "supported", "verified", "proven"}
)


def adapt_generic_claim_mapping(
    payload: Mapping[str, Any],
) -> AdaptationResult:
    """Conservatively map forbidden generic legacy booleans.

    ``success`` may inform at most ``effect.started``. ``verified``/``proven``
    may inform at most ``proof.candidate``. ``available``/``supported`` are
    discovery-only and do not alter the product envelope dimensions.
    """

    if not isinstance(payload, Mapping):
        return _incompat(
            code="not_a_mapping",
            legacy_kind="generic_claim",
            message="adapt_generic_claim_mapping requires a mapping",
        )

    present = {
        key: payload[key]
        for key in _FORBIDDEN_GENERIC_FIELDS
        if key in payload
    }
    if not present:
        return _incompat(
            code="no_generic_claim_fields",
            legacy_kind="generic_claim",
            message="payload carries none of the forbidden generic claim fields",
            details={"allowed_keys": sorted(_FORBIDDEN_GENERIC_FIELDS)},
        )

    envelope = EvidenceEnvelope.weakest()
    informed: list[str] = []
    notes: list[str] = []

    if present.get("success") is True:
        envelope = envelope.with_updates(effect=Effect.STARTED)
        informed.append("effect")
        notes.append("success:true → effect.started only (not observed)")
    elif present.get("success") is False:
        notes.append("success:false leaves effect.not_started")

    for field_name in ("verified", "proven"):
        if present.get(field_name) is True:
            envelope = envelope.with_updates(proof=Proof.CANDIDATE)
            if "proof" not in informed:
                informed.append("proof")
            notes.append(f"{field_name}:true → proof.candidate (never verified)")

    for field_name in ("available", "supported"):
        if field_name in present:
            notes.append(
                f"{field_name} is discovery/inventory only; dimensions unchanged"
            )

    return _success(
        envelope,
        legacy_kind="generic_claim",
        informed=tuple(informed),
        source_ref="generic_claim",
        notes=tuple(notes),
    )


def adapt_legacy_record(
    record: Any,
    *,
    now_ms: Optional[int] = None,
    legacy_kind_hint: str = "",
) -> AdaptationResult:
    """Dispatch every supported supervisor legacy record to an FCA projection."""

    if isinstance(record, TypedIncompatibility):
        return record
    if isinstance(record, EnvelopeAdaptation):
        return record
    if isinstance(record, EvidenceEnvelope):
        return _success(
            record,
            legacy_kind="EvidenceEnvelope",
            informed=tuple(record.informed_overrides().keys()),
            source_ref="EvidenceEnvelope",
            notes=("passthrough of an existing envelope",),
        )

    hint = str(legacy_kind_hint or "").strip()

    # Enum ladders / statuses.
    if isinstance(record, AssuranceLevel) or (
        hint == "AssuranceLevel" and isinstance(record, str)
    ):
        return adapt_assurance_level(record)
    if isinstance(record, _repair_evidence.AssuranceLevel) or (
        hint in {"database_repair.AssuranceLevel", "RepairAssuranceLevel"}
        and isinstance(record, str)
    ):
        return adapt_database_repair_assurance_level(record)
    if isinstance(record, ProofStatus) or (
        hint == "ProofStatus" and isinstance(record, str)
    ):
        return adapt_proof_status(record)
    if hint == "EvidenceTier" or _is_named_type(record, "EvidenceTier"):
        return adapt_evidence_tier(record)
    if isinstance(record, str) and record.casefold() in _EVIDENCE_TIER_PROOF_MAP:
        # Bare EvidenceTier spelling without a hint.
        return adapt_evidence_tier(record)

    if isinstance(record, ProofReceipt) or _is_named_type(record, "ProofReceipt"):
        return adapt_proof_receipt(record)
    if isinstance(record, ProofCacheEntry) or _is_named_type(record, "ProofCacheEntry"):
        return adapt_proof_cache_entry(record, now_ms=now_ms)
    if isinstance(record, UntrustedDraftCacheEntry) or _is_named_type(
        record, "UntrustedDraftCacheEntry"
    ):
        return adapt_untrusted_draft_cache_entry(record)
    if hint == "ExecutionPermit" or _is_named_type(record, "ExecutionPermit"):
        return adapt_execution_permit(record, now_ms=now_ms)
    if hint == "ProviderCapabilityEvidence" or _is_named_type(
        record, "ProviderCapabilityEvidence"
    ):
        return adapt_provider_capability_evidence(record)
    if hint == "CapabilityEvidence" or _is_named_type(record, "CapabilityEvidence"):
        return adapt_capability_evidence(record)

    if isinstance(record, Mapping):
        map_hint = hint or str(record.get("legacy_kind") or "").strip()
        if map_hint == "AssuranceLevel" and "value" in record:
            return adapt_assurance_level(record["value"])
        if map_hint in {"database_repair.AssuranceLevel", "RepairAssuranceLevel"} and (
            "value" in record
        ):
            return adapt_database_repair_assurance_level(record["value"])
        if map_hint == "ProofStatus" and "value" in record:
            return adapt_proof_status(record["value"])
        if map_hint == "EvidenceTier" and "value" in record:
            return adapt_evidence_tier(record["value"])
        if map_hint == "ExecutionPermit" or (
            "issued_at_ms" in record and "expires_at_ms" in record
        ):
            return adapt_execution_permit(record, now_ms=now_ms)
        if map_hint == "ProviderCapabilityEvidence" or (
            "provider_id" in record and "ready" in record
        ):
            return adapt_provider_capability_evidence(record)
        if map_hint == "CapabilityEvidence" or "providers" in record:
            return adapt_capability_evidence(record)
        if map_hint in {"stale_marker", "unknown_marker"} or (
            "marker" in record
            and _token(record.get("marker")).casefold() in {"stale", "unknown"}
        ):
            return adapt_stale_or_unknown_marker(record.get("marker", map_hint))
        if any(key in record for key in _FORBIDDEN_GENERIC_FIELDS):
            return adapt_generic_claim_mapping(record)
        return _incompat(
            code="unsupported_mapping_record",
            legacy_kind=map_hint or "mapping",
            message="mapping is not a recognized legacy assurance record shape",
            details={"keys": sorted(str(k) for k in record.keys())},
        )

    if isinstance(record, str) and hint in {
        "stale_marker",
        "unknown_marker",
        "stale_or_unknown",
    }:
        return adapt_stale_or_unknown_marker(record)

    return _incompat(
        code="unsupported_legacy_record",
        legacy_kind=hint or _legacy_kind_name(record),
        message=(
            "no conservative FCA mapping for legacy record type "
            f"{type(record).__name__}"
        ),
        details={"type": type(record).__name__},
    )


def _non_proof_dimensions_are_weakest(envelope: EvidenceEnvelope) -> bool:
    weakest = EvidenceEnvelope.weakest()
    for name in DIMENSION_ORDER:
        if name == "proof":
            continue
        if getattr(envelope, name) != getattr(weakest, name):
            return False
    return True


def project_envelope_to_assurance_level(
    envelope: EvidenceEnvelope,
) -> Union[AssuranceLevel, TypedIncompatibility]:
    """Reverse-project an envelope to AssuranceLevel.

    Refuses when any non-proof dimension differs from weakest defaults
    (collapsing the product would lose information) and refuses any mapping
    that would invent a stronger ladder rank than ``proof`` supports.
    """

    if not isinstance(envelope, EvidenceEnvelope):
        return _incompat(
            code="not_an_evidence_envelope",
            legacy_kind="reverse_projection",
            message="project_envelope_to_assurance_level requires EvidenceEnvelope",
        )

    if not _non_proof_dimensions_are_weakest(envelope):
        return _incompat(
            code="information_losing_reverse_projection",
            legacy_kind="AssuranceLevel",
            message=(
                "refusing reverse projection: non-proof dimensions are set; "
                "collapsing to AssuranceLevel would lose product information"
            ),
            details={
                "informed_overrides": envelope.informed_overrides(),
                "refused_promotion": True,
            },
            unsafe_promotion=True,
        )

    proof = envelope.proof.value
    mapped = _PROOF_TO_ASSURANCE_LEVEL.get(proof)
    if mapped is None:
        return _incompat(
            code="proof_has_no_assurance_level_projection",
            legacy_kind="AssuranceLevel",
            message=(
                f"proof.{proof} has no lossless AssuranceLevel projection"
            ),
            details={"proof": proof},
            unsafe_promotion=False,
        )

    # Explicit anti-promotion: never emit ATTESTED from proof.verified alone.
    level = AssuranceLevel(mapped)
    if level is AssuranceLevel.ATTESTED:
        return _incompat(
            code="attested_promotion_refused",
            legacy_kind="AssuranceLevel",
            message="refusing information-losing promotion to AssuranceLevel.ATTESTED",
            unsafe_promotion=True,
        )
    return level


def project_envelope_to_proof_status(
    envelope: EvidenceEnvelope,
    *,
    informed_dimensions: Optional[Sequence[str]] = None,
) -> Union[ProofStatus, TypedIncompatibility]:
    """Reverse-project to ProofStatus when only proof(/freshness.stale) differ.

    Because weakest freshness is already ``stale``, callers should pass
    ``informed_dimensions`` from :class:`EnvelopeAdaptation` when available so
    ``ProofStatus.STALE`` (explicit freshness) stays distinct from
    ``ProofStatus.INCONCLUSIVE`` (proof.unknown only).
    """

    if not isinstance(envelope, EvidenceEnvelope):
        return _incompat(
            code="not_an_evidence_envelope",
            legacy_kind="reverse_projection",
            message="project_envelope_to_proof_status requires EvidenceEnvelope",
        )

    weakest = EvidenceEnvelope.weakest()
    for name in DIMENSION_ORDER:
        if name in {"proof", "freshness"}:
            continue
        if getattr(envelope, name) != getattr(weakest, name):
            return _incompat(
                code="information_losing_reverse_projection",
                legacy_kind="ProofStatus",
                message=(
                    "refusing reverse projection: dimensions beyond proof/freshness "
                    "are set; collapsing to ProofStatus would lose information"
                ),
                details={"informed_overrides": envelope.informed_overrides()},
                unsafe_promotion=True,
            )

    proof = envelope.proof.value
    freshness = envelope.freshness.value
    informed = {
        str(item)
        for item in (
            informed_dimensions
            if informed_dimensions is not None
            else envelope.informed_overrides().keys()
        )
    }

    # Explicit stale pairing from the seed map.
    if (
        proof == Proof.UNKNOWN.value
        and freshness == Freshness.STALE.value
        and "freshness" in informed
    ):
        return ProofStatus.STALE

    # Non-default freshness (current/superseded/withdrawn) cannot collapse.
    if freshness != weakest.freshness.value:
        return _incompat(
            code="information_losing_reverse_projection",
            legacy_kind="ProofStatus",
            message=(
                f"freshness.{freshness} with proof.{proof} cannot reverse-project "
                "to ProofStatus without information loss"
            ),
            details={"proof": proof, "freshness": freshness, "informed": sorted(informed)},
            unsafe_promotion=True,
        )

    # freshness.stale was explicitly informed alongside a strong proof value:
    # dropping the stale bit would lose information.
    if "freshness" in informed and proof not in {
        Proof.UNKNOWN.value,
        Proof.NONE.value,
    }:
        return _incompat(
            code="information_losing_reverse_projection",
            legacy_kind="ProofStatus",
            message=(
                "refusing to drop freshness.stale while projecting a strong "
                "proof value into ProofStatus"
            ),
            details={"proof": proof, "freshness": freshness},
            unsafe_promotion=True,
        )

    reverse = {
        Proof.NONE.value: ProofStatus.UNPROVED,
        Proof.CANDIDATE.value: ProofStatus.CANDIDATE,
        Proof.VERIFIED.value: ProofStatus.KERNEL_VERIFIED,
        Proof.REFUTED.value: ProofStatus.VALIDATED_REFUTED,
        Proof.UNKNOWN.value: ProofStatus.INCONCLUSIVE,
        Proof.VERIFIER_UNAVAILABLE.value: ProofStatus.UNSUPPORTED,
    }
    status = reverse.get(proof)
    if status is None:
        return _incompat(
            code="proof_has_no_proof_status_projection",
            legacy_kind="ProofStatus",
            message=f"proof.{proof} has no ProofStatus projection",
        )
    return status


__all__ = [
    "ADAPTER_SCHEMA",
    "AdaptationResult",
    "Authority",
    "BUNDLE",
    "DIMENSION_ORDER",
    "Effect",
    "EnvelopeAdaptation",
    "Environment",
    "EvidenceEnvelope",
    "FormalClaimAdapterError",
    "Freshness",
    "GOAL_ID",
    "Integrity",
    "Origin",
    "Policy",
    "Proof",
    "Review",
    "TASK_ID",
    "TypedIncompatibility",
    "UNSAFE_PROMOTION_DEFAULT",
    "VOCAB_SCHEMA",
    "adapt_assurance_level",
    "adapt_capability_evidence",
    "adapt_database_repair_assurance_level",
    "adapt_evidence_tier",
    "adapt_execution_permit",
    "adapt_generic_claim_mapping",
    "adapt_legacy_record",
    "adapt_proof_cache_entry",
    "adapt_proof_receipt",
    "adapt_proof_status",
    "adapt_provider_capability_evidence",
    "adapt_stale_or_unknown_marker",
    "adapt_untrusted_draft_cache_entry",
    "project_envelope_to_assurance_level",
    "project_envelope_to_proof_status",
]
