"""Typed claim/evidence semantics and lifecycle for codebase proof (CBP-025).

Interface: ``CodeClaimRecord@1``

This module is a normalization/lifecycle layer over existing
``formal_verification_contracts`` and ``code_proof_obligations`` types.  It
does **not** invent a second assurance model: authoritative assurance is always
re-derived via :func:`~.formal_verification_contracts.assess_assurance` /
:func:`~.formal_verification_contracts.derive_assurance` from typed
:class:`~.formal_verification_contracts.ProofEvidence`.

Normative rules:

* Cache miss is lifecycle ``open`` (or ``unknown``), never ``refuted``.
* Query / GraphRAG facts and bounded observations cannot independently mint
  kernel or attestation assurance.
* Arbitrary natural-language claims fail closed.
* Invalidation selectors are machine-readable; status ``stale`` is distinct
  from refutation and from cache miss.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .code_proof_obligations import (
    ImplementationEvidenceKind,
    ImplementationResultEvidence,
)
from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    CodeProofObligation,
    ContractValidationError,
    EvidenceFreshness,
    EvidenceKind,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    assess_assurance,
    assurance_satisfies,
    content_identity,
    derive_assurance,
    derive_verdict,
)


CODE_CLAIM_RECORD_INTERFACE: Final = "CodeClaimRecord@1"
CODE_CLAIM_RECORD_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-claim-record@1"
)
CODE_CLAIM_CONTRACT_VERSION: Final = 1
CLAIM_CATALOG_VERSION: Final = "1"

# Cache lookup outcomes are not claim verdicts.
CACHE_LOOKUP_HIT: Final = "hit"
CACHE_LOOKUP_MISS: Final = "miss"
CACHE_LOOKUP_STALE: Final = "stale"
CACHE_LOOKUP_REJECTED: Final = "rejected"


class CodeClaimContractError(ContractValidationError):
    """Raised when a claim/evidence contract is malformed or fails closed."""


class ClaimFamily(str, Enum):
    """Closed reviewed claim families (catalog-aligned, no NL invent)."""

    DEPENDENCY_REACHABILITY = "dependency_reachability"
    API_CONTRACT = "api_contract"
    BEHAVIORAL_INVARIANT = "behavioral_invariant"
    SECURITY_PROPERTY = "security_property"
    SEMANTIC_EQUIVALENCE = "semantic_equivalence"
    SUPERVISOR_LIFECYCLE = "supervisor_lifecycle"
    SRT_STRUCTURAL = "srt_structural"
    UNSUPPORTED = "unsupported"


class ClaimStatus(str, Enum):
    """Lifecycle of one content-addressed claim.

    Distinctions that must never collapse:

    * ``unknown`` — no claim evaluation has been started.
    * ``open`` — claim is admitted and awaits evidence (includes cache miss).
    * ``satisfied`` — settled positively at required assurance.
    * ``refuted`` — settled negatively with independent counterexample evidence.
    * ``unsupported`` — reviewed shape/template refuses the claim.
    * ``not_measured`` — measurement path not executed or out of bounds.
    * ``stale`` — prior evidence no longer binds the current selectors.

    A proof-cache miss is memoization absence, **not** refutation.
    """

    UNKNOWN = "unknown"
    OPEN = "open"
    SATISFIED = "satisfied"
    REFUTED = "refuted"
    UNSUPPORTED = "unsupported"
    NOT_MEASURED = "not_measured"
    STALE = "stale"

    @property
    def terminal(self) -> bool:
        """Whether the claim is settled (positively or negatively) or terminal."""

        return self in {
            ClaimStatus.SATISFIED,
            ClaimStatus.REFUTED,
            ClaimStatus.UNSUPPORTED,
        }

    @property
    def is_refutation(self) -> bool:
        return self is ClaimStatus.REFUTED


class EvidenceTier(str, Enum):
    """Authority ladder for evidence supporting a claim.

    Higher tiers may subsume lower ones when independently re-derived, but
    lower tiers never upgrade themselves into kernel or attestation assurance.
    """

    QUERY_FACT = "query_fact"
    GRAPHRAG_FACT = "graphrag_fact"
    OBSERVATION = "observation"  # test / runtime / static-analysis
    SOLVER_CANDIDATE = "solver_candidate"
    KERNEL_PROOF = "kernel_proof"
    CRYPTOGRAPHIC_ATTESTATION = "cryptographic_attestation"

    @property
    def rank(self) -> int:
        return {
            EvidenceTier.QUERY_FACT: 0,
            EvidenceTier.GRAPHRAG_FACT: 0,
            EvidenceTier.OBSERVATION: 1,
            EvidenceTier.SOLVER_CANDIDATE: 2,
            EvidenceTier.KERNEL_PROOF: 3,
            EvidenceTier.CRYPTOGRAPHIC_ATTESTATION: 4,
        }[self]

    @property
    def can_mint_kernel_assurance(self) -> bool:
        return self in {
            EvidenceTier.KERNEL_PROOF,
            EvidenceTier.CRYPTOGRAPHIC_ATTESTATION,
        }

    @property
    def max_assurance(self) -> AssuranceLevel:
        """Ceiling this tier may independently contribute without higher evidence."""

        return {
            EvidenceTier.QUERY_FACT: AssuranceLevel.UNVERIFIED,
            EvidenceTier.GRAPHRAG_FACT: AssuranceLevel.UNVERIFIED,
            EvidenceTier.OBSERVATION: AssuranceLevel.CANDIDATE,
            EvidenceTier.SOLVER_CANDIDATE: AssuranceLevel.SOLVER_CHECKED,
            EvidenceTier.KERNEL_PROOF: AssuranceLevel.KERNEL_VERIFIED,
            EvidenceTier.CRYPTOGRAPHIC_ATTESTATION: AssuranceLevel.ATTESTED,
        }[self]


class InvalidationSelectorKind(str, Enum):
    """Machine-readable reasons a claim becomes stale or must be re-proved."""

    REPOSITORY_TREE = "repository_tree"
    AST_SCOPE = "ast_scope"
    PREMISE_SET = "premise_set"
    ASSUMPTION_SET = "assumption_set"
    TOOLCHAIN = "toolchain"
    POLICY = "policy"
    CATALOG = "catalog"
    PROPERTY = "property"
    OBLIGATION = "obligation"
    EVIDENCE_FRESHNESS = "evidence_freshness"
    PRODUCER = "producer"
    REQUIRED_ASSURANCE = "required_assurance"
    CACHE_BINDING = "cache_binding"


_FAMILY_BY_INVARIANT: Mapping[str, ClaimFamily] = MappingProxyType(
    {
        "dependency": ClaimFamily.DEPENDENCY_REACHABILITY,
        "reachability": ClaimFamily.DEPENDENCY_REACHABILITY,
        "import": ClaimFamily.DEPENDENCY_REACHABILITY,
        "api": ClaimFamily.API_CONTRACT,
        "interface": ClaimFamily.API_CONTRACT,
        "contract": ClaimFamily.API_CONTRACT,
        "behavior": ClaimFamily.BEHAVIORAL_INVARIANT,
        "invariant": ClaimFamily.BEHAVIORAL_INVARIANT,
        "state_machine": ClaimFamily.BEHAVIORAL_INVARIANT,
        "security": ClaimFamily.SECURITY_PROPERTY,
        "authorization": ClaimFamily.SECURITY_PROPERTY,
        "lease": ClaimFamily.SECURITY_PROPERTY,
        "equivalence": ClaimFamily.SEMANTIC_EQUIVALENCE,
        "projection": ClaimFamily.SEMANTIC_EQUIVALENCE,
        "lifecycle": ClaimFamily.SUPERVISOR_LIFECYCLE,
        "supervisor": ClaimFamily.SUPERVISOR_LIFECYCLE,
        "merge": ClaimFamily.SUPERVISOR_LIFECYCLE,
        "srt": ClaimFamily.SRT_STRUCTURAL,
        "structural": ClaimFamily.SRT_STRUCTURAL,
        "non_vacuous": ClaimFamily.SRT_STRUCTURAL,
    }
)

_FAMILY_BY_TEMPLATE_HINT: Mapping[str, ClaimFamily] = MappingProxyType(
    {
        "lease-uniqueness-and-fencing": ClaimFamily.SECURITY_PROPERTY,
        "dag-acyclicity": ClaimFamily.BEHAVIORAL_INVARIANT,
        "merge-idempotence": ClaimFamily.SUPERVISOR_LIFECYCLE,
        "cache-key-completeness": ClaimFamily.BEHAVIORAL_INVARIANT,
        "evidence-freshness": ClaimFamily.BEHAVIORAL_INVARIANT,
        "projection-equivalence": ClaimFamily.SEMANTIC_EQUIVALENCE,
        "legal-state-transition": ClaimFamily.BEHAVIORAL_INVARIANT,
        "unsupported-proof-fail-closed": ClaimFamily.UNSUPPORTED,
    }
)

_NL_MARKERS: Final[tuple[str, ...]] = (
    "natural_language",
    "nl_claim",
    "freeform",
    "free_text",
    "prose_claim",
    "arbitrary_claim",
)


def _norm_text(value: Any, *, field_name: str, required: bool = False) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise CodeClaimContractError(f"{field_name} must be a string")
    if required and not text:
        raise CodeClaimContractError(f"{field_name} is required")
    return text


def _norm_ids(
    values: Any,
    *,
    field_name: str,
    required: bool = False,
) -> tuple[str, ...]:
    if values is None:
        items: Iterable[Any] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        items = values
    else:
        raise CodeClaimContractError(f"{field_name} must be a sequence of strings")
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _norm_text(item, field_name=field_name, required=True)
        if text not in seen:
            seen.add(text)
            out.append(text)
    result = tuple(sorted(out))
    if required and not result:
        raise CodeClaimContractError(f"{field_name} must not be empty")
    return result


def _norm_enum(value: Any, enum_type: type[Enum], *, field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(sorted({str(item.value) for item in enum_type}))
        raise CodeClaimContractError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _norm_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise CodeClaimContractError(f"{field_name} must be a mapping")
    # Detach and freeze; reject non-string keys.
    if not all(isinstance(k, str) for k in value):
        raise CodeClaimContractError(f"{field_name} keys must be strings")
    return MappingProxyType(dict(value))


def _looks_like_natural_language_claim(
    *,
    property_id: str,
    obligation_id: str,
    claim_family: ClaimFamily | str | None,
    statement: str,
    metadata: Mapping[str, Any] | None,
) -> bool:
    """Heuristic fail-closed detector for unreviewed freeform claims."""

    meta = dict(metadata or {})
    for key, raw in meta.items():
        key_l = str(key).strip().lower()
        if any(marker in key_l for marker in _NL_MARKERS):
            return True
        if isinstance(raw, str) and raw.strip().lower() in {"true", "1", "yes"}:
            if any(marker in key_l for marker in ("nl", "natural", "freeform", "prose")):
                return True
    if meta.get("reviewed") is False and not property_id and not obligation_id:
        return True
    if meta.get("natural_language") or meta.get("nl_claim"):
        return True
    # Bare prose without reviewed ids is rejected.
    if statement and not property_id and not obligation_id:
        if claim_family in (None, "", ClaimFamily.UNSUPPORTED):
            return True
        # Explicit freeform family markers
        family_text = str(getattr(claim_family, "value", claim_family) or "").lower()
        if family_text in {"natural_language", "freeform", "prose"}:
            return True
    return False


def resolve_claim_family(
    *,
    template_id: str = "",
    invariant_class: str = "",
    property_id: str = "",
    code_shape: str = "",
    explicit: ClaimFamily | str | None = None,
) -> ClaimFamily:
    """Map reviewed catalog/template hints onto a closed ClaimFamily."""

    if explicit is not None and str(explicit).strip():
        return ClaimFamily(_norm_enum(explicit, ClaimFamily, field_name="claim_family").value)  # type: ignore[arg-type]

    tid = (template_id or "").strip().lower()
    if tid in _FAMILY_BY_TEMPLATE_HINT:
        return _FAMILY_BY_TEMPLATE_HINT[tid]

    for hay in (invariant_class, property_id, code_shape, tid):
        text = (hay or "").strip().lower().replace("-", "_")
        if not text:
            continue
        for key, family in _FAMILY_BY_INVARIANT.items():
            if key in text:
                return family
    # Reviewed but uncategorized shapes still admit as behavioral by default
    # only when a property or template id is present; otherwise unsupported.
    if template_id or property_id:
        return ClaimFamily.BEHAVIORAL_INVARIANT
    return ClaimFamily.UNSUPPORTED


def evidence_kind_to_tier(kind: EvidenceKind | str) -> EvidenceTier:
    """Project a ProofEvidence kind onto the CBP evidence-tier ladder."""

    kind_e = (
        kind
        if isinstance(kind, EvidenceKind)
        else EvidenceKind(str(kind))
    )
    if kind_e in {
        EvidenceKind.LLM_OUTPUT,
        EvidenceKind.ATP_CANDIDATE,
        EvidenceKind.SMT_CANDIDATE,
        EvidenceKind.SOLVER_RESULT,
    }:
        return EvidenceTier.SOLVER_CANDIDATE
    if kind_e is EvidenceKind.KERNEL_VERIFICATION:
        return EvidenceTier.KERNEL_PROOF
    if kind_e is EvidenceKind.CRYPTOGRAPHIC_ATTESTATION:
        return EvidenceTier.CRYPTOGRAPHIC_ATTESTATION
    if kind_e in {
        EvidenceKind.TEST_RESULT,
        EvidenceKind.STATIC_ANALYSIS,
    }:
        return EvidenceTier.OBSERVATION
    if kind_e is EvidenceKind.CACHE_ENTRY:
        # Cache is memoization; tier of the cached body is not upgraded here.
        return EvidenceTier.SOLVER_CANDIDATE
    return EvidenceTier.QUERY_FACT


def implementation_kind_to_tier(
    kind: ImplementationEvidenceKind | str,
) -> EvidenceTier:
    """All implementation observations remain on the observation tier."""

    _ = kind  # all map to observation
    return EvidenceTier.OBSERVATION


def max_assurance_for_tiers(tiers: Iterable[EvidenceTier | str]) -> AssuranceLevel:
    """Ceiling reachable from the given tiers without inventing higher evidence."""

    ceiling = AssuranceLevel.UNVERIFIED
    for raw in tiers:
        tier = (
            raw
            if isinstance(raw, EvidenceTier)
            else EvidenceTier(str(raw))
        )
        if tier.max_assurance.rank > ceiling.rank:
            ceiling = tier.max_assurance
    return ceiling


def tiers_can_independently_mint_kernel(
    tiers: Iterable[EvidenceTier | str],
) -> bool:
    """Return whether any tier may independently establish kernel assurance."""

    for raw in tiers:
        tier = (
            raw
            if isinstance(raw, EvidenceTier)
            else EvidenceTier(str(raw))
        )
        if tier.can_mint_kernel_assurance:
            return True
    return False


def cache_miss_status(*, previously: ClaimStatus | str | None = None) -> ClaimStatus:
    """Map a proof-cache miss onto a lifecycle status (never refuted)."""

    if previously is None or previously == "" or previously == ClaimStatus.UNKNOWN:
        return ClaimStatus.OPEN
    status = (
        previously
        if isinstance(previously, ClaimStatus)
        else ClaimStatus(str(previously))
    )
    if status is ClaimStatus.REFUTED:
        # A miss after a prior refutation does not reaffirm refutation from cache;
        # reopen so independent re-proof can settle.
        return ClaimStatus.OPEN
    if status is ClaimStatus.SATISFIED:
        # Missing the cache does not unsatisfy; treat as open pending re-derive.
        return ClaimStatus.OPEN
    if status is ClaimStatus.STALE:
        return ClaimStatus.STALE
    if status is ClaimStatus.UNSUPPORTED:
        return ClaimStatus.UNSUPPORTED
    if status is ClaimStatus.NOT_MEASURED:
        return ClaimStatus.NOT_MEASURED
    return ClaimStatus.OPEN


@dataclass(frozen=True)
class InvalidationSelector:
    """One machine-readable binding that can invalidate a claim."""

    kind: InvalidationSelectorKind
    value: str
    reason_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _norm_enum(self.kind, InvalidationSelectorKind, field_name="kind"),
        )
        object.__setattr__(
            self, "value", _norm_text(self.value, field_name="value", required=True)
        )
        object.__setattr__(
            self,
            "reason_code",
            _norm_text(self.reason_code, field_name="reason_code"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value if isinstance(self.kind, Enum) else str(self.kind),
            "value": self.value,
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InvalidationSelector":
        if not isinstance(payload, Mapping):
            raise CodeClaimContractError("invalidation selector must be an object")
        return cls(
            kind=payload.get("kind", InvalidationSelectorKind.EVIDENCE_FRESHNESS),
            value=str(payload.get("value") or ""),
            reason_code=str(payload.get("reason_code") or ""),
        )

    def matches(self, *, kind: InvalidationSelectorKind | str, value: str) -> bool:
        kind_e = (
            kind
            if isinstance(kind, InvalidationSelectorKind)
            else InvalidationSelectorKind(str(kind))
        )
        return self.kind is kind_e and self.value == str(value).strip()


def build_invalidation_selectors(
    *,
    repository_tree_id: str = "",
    scope_ids: Sequence[str] = (),
    premise_ids: Sequence[str] = (),
    assumption_ids: Sequence[str] = (),
    toolchain_id: str = "",
    policy_id: str = "",
    catalog_version: str = "",
    property_id: str = "",
    obligation_id: str = "",
    producer_id: str = "",
    required_assurance: AssuranceLevel | str = "",
) -> tuple[InvalidationSelector, ...]:
    """Construct the default closed set of invalidation selectors for a claim."""

    selectors: list[InvalidationSelector] = []
    if repository_tree_id:
        selectors.append(
            InvalidationSelector(
                kind=InvalidationSelectorKind.REPOSITORY_TREE,
                value=repository_tree_id,
                reason_code="stale_candidate_tree",
            )
        )
    for scope in sorted({str(s).strip() for s in scope_ids if str(s).strip()}):
        selectors.append(
            InvalidationSelector(
                kind=InvalidationSelectorKind.AST_SCOPE,
                value=scope,
                reason_code="changed_ast_scope",
            )
        )
    if premise_ids:
        selectors.append(
            InvalidationSelector(
                kind=InvalidationSelectorKind.PREMISE_SET,
                value=content_identity({"premise_ids": sorted(premise_ids)}),
                reason_code="changed_premises",
            )
        )
    if assumption_ids:
        selectors.append(
            InvalidationSelector(
                kind=InvalidationSelectorKind.ASSUMPTION_SET,
                value=content_identity({"assumption_ids": sorted(assumption_ids)}),
                reason_code="changed_assumptions",
            )
        )
    if toolchain_id:
        selectors.append(
            InvalidationSelector(
                kind=InvalidationSelectorKind.TOOLCHAIN,
                value=toolchain_id,
                reason_code="toolchain_drift",
            )
        )
    if policy_id:
        selectors.append(
            InvalidationSelector(
                kind=InvalidationSelectorKind.POLICY,
                value=policy_id,
                reason_code="policy_drift",
            )
        )
    if catalog_version:
        selectors.append(
            InvalidationSelector(
                kind=InvalidationSelectorKind.CATALOG,
                value=catalog_version,
                reason_code="catalog_drift",
            )
        )
    if property_id:
        selectors.append(
            InvalidationSelector(
                kind=InvalidationSelectorKind.PROPERTY,
                value=property_id,
                reason_code="property_changed",
            )
        )
    if obligation_id:
        selectors.append(
            InvalidationSelector(
                kind=InvalidationSelectorKind.OBLIGATION,
                value=obligation_id,
                reason_code="obligation_changed",
            )
        )
    if producer_id:
        selectors.append(
            InvalidationSelector(
                kind=InvalidationSelectorKind.PRODUCER,
                value=producer_id,
                reason_code="producer_drift",
            )
        )
    if required_assurance:
        level = (
            required_assurance
            if isinstance(required_assurance, AssuranceLevel)
            else AssuranceLevel(str(required_assurance))
        )
        selectors.append(
            InvalidationSelector(
                kind=InvalidationSelectorKind.REQUIRED_ASSURANCE,
                value=level.value,
                reason_code="required_assurance_changed",
            )
        )
    # Stable order by kind then value.
    return tuple(
        sorted(selectors, key=lambda s: (s.kind.value, s.value, s.reason_code))
    )


@dataclass(frozen=True)
class CodeClaimRecord(CanonicalContract):
    """Content-addressed claim binding family, evidence, and invalidators.

    ``claim_id`` / ``content_id`` are derived from the canonical payload and are
    never trusted when supplied by a caller without matching the re-derived
    identity.
    """

    SCHEMA: ClassVar[str] = CODE_CLAIM_RECORD_SCHEMA

    claim_family: ClaimFamily
    status: ClaimStatus = ClaimStatus.UNKNOWN
    property_id: str = ""
    obligation_id: str = ""
    repository_id: str = ""
    repository_tree_id: str = ""
    scope_ids: tuple[str, ...] = ()
    premise_ids: tuple[str, ...] = ()
    assumption_ids: tuple[str, ...] = ()
    producer_id: str = ""
    toolchain_id: str = ""
    policy_id: str = ""
    catalog_version: str = CLAIM_CATALOG_VERSION
    evidence_ids: tuple[str, ...] = ()
    evidence_tiers: tuple[EvidenceTier, ...] = ()
    required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED
    derived_assurance: AssuranceLevel = AssuranceLevel.UNVERIFIED
    invalidation_selectors: tuple[InvalidationSelector, ...] = ()
    statement: str = ""
    cache_lookup: str = ""
    receipt_id: str = ""
    template_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "claim_family",
            _norm_enum(self.claim_family, ClaimFamily, field_name="claim_family"),
        )
        object.__setattr__(
            self,
            "status",
            _norm_enum(self.status, ClaimStatus, field_name="status"),
        )
        for name in (
            "property_id",
            "obligation_id",
            "repository_id",
            "repository_tree_id",
            "producer_id",
            "toolchain_id",
            "policy_id",
            "catalog_version",
            "statement",
            "cache_lookup",
            "receipt_id",
            "template_id",
        ):
            object.__setattr__(
                self, name, _norm_text(getattr(self, name), field_name=name)
            )
        if not self.catalog_version:
            object.__setattr__(self, "catalog_version", CLAIM_CATALOG_VERSION)

        object.__setattr__(
            self, "scope_ids", _norm_ids(self.scope_ids, field_name="scope_ids")
        )
        object.__setattr__(
            self, "premise_ids", _norm_ids(self.premise_ids, field_name="premise_ids")
        )
        object.__setattr__(
            self,
            "assumption_ids",
            _norm_ids(self.assumption_ids, field_name="assumption_ids"),
        )
        object.__setattr__(
            self,
            "evidence_ids",
            _norm_ids(self.evidence_ids, field_name="evidence_ids"),
        )

        tiers = tuple(
            _norm_enum(t, EvidenceTier, field_name="evidence_tiers")  # type: ignore[misc]
            for t in (self.evidence_tiers or ())
        )
        # Stable unique order by rank then value.
        unique_tiers: list[EvidenceTier] = []
        seen_tiers: set[EvidenceTier] = set()
        for tier in sorted(tiers, key=lambda t: (t.rank, t.value)):  # type: ignore[attr-defined]
            if tier not in seen_tiers:
                seen_tiers.add(tier)  # type: ignore[arg-type]
                unique_tiers.append(tier)  # type: ignore[arg-type]
        object.__setattr__(self, "evidence_tiers", tuple(unique_tiers))

        object.__setattr__(
            self,
            "required_assurance",
            _norm_enum(
                self.required_assurance, AssuranceLevel, field_name="required_assurance"
            ),
        )
        object.__setattr__(
            self,
            "derived_assurance",
            _norm_enum(
                self.derived_assurance, AssuranceLevel, field_name="derived_assurance"
            ),
        )

        selectors = tuple(
            item
            if isinstance(item, InvalidationSelector)
            else InvalidationSelector.from_dict(item)  # type: ignore[arg-type]
            for item in (self.invalidation_selectors or ())
        )
        object.__setattr__(
            self,
            "invalidation_selectors",
            tuple(
                sorted(
                    selectors,
                    key=lambda s: (s.kind.value, s.value, s.reason_code),
                )
            ),
        )
        object.__setattr__(
            self, "metadata", _norm_mapping(self.metadata, field_name="metadata")
        )

        # Fail closed on arbitrary natural-language claims.
        if _looks_like_natural_language_claim(
            property_id=self.property_id,
            obligation_id=self.obligation_id,
            claim_family=self.claim_family,  # type: ignore[arg-type]
            statement=self.statement,
            metadata=self.metadata,
        ):
            raise CodeClaimContractError(
                "arbitrary natural-language claims fail closed; bind a reviewed "
                "property_id or obligation_id"
            )

        # Query / observation tiers cannot independently claim kernel assurance.
        if self.derived_assurance in (
            AssuranceLevel.KERNEL_VERIFIED,
            AssuranceLevel.ATTESTED,
        ) and not tiers_can_independently_mint_kernel(self.evidence_tiers):  # type: ignore[arg-type]
            raise CodeClaimContractError(
                "query facts and observations cannot independently mint kernel "
                "or attestation assurance"
            )

        # Cache miss must never appear as refutation.
        if (
            self.cache_lookup == CACHE_LOOKUP_MISS
            and self.status is ClaimStatus.REFUTED
        ):
            raise CodeClaimContractError(
                "cache miss must not be treated as refutation"
            )

        # Unsupported family forces unsupported status when open/unknown would
        # otherwise imply a measurable claim.
        if (
            self.claim_family is ClaimFamily.UNSUPPORTED
            and self.status
            in {ClaimStatus.SATISFIED, ClaimStatus.REFUTED, ClaimStatus.OPEN}
            and not self.obligation_id
            and not self.property_id
        ):
            raise CodeClaimContractError(
                "unsupported claim family cannot be open/satisfied/refuted "
                "without a reviewed property or obligation binding"
            )

    @property
    def claim_id(self) -> str:
        return self.content_id

    @property
    def interface(self) -> str:
        return CODE_CLAIM_RECORD_INTERFACE

    @property
    def tree_id(self) -> str:
        return self.repository_tree_id

    @property
    def highest_evidence_tier(self) -> EvidenceTier | None:
        if not self.evidence_tiers:
            return None
        return max(self.evidence_tiers, key=lambda t: t.rank)  # type: ignore[arg-type]

    @property
    def tier_assurance_ceiling(self) -> AssuranceLevel:
        return max_assurance_for_tiers(self.evidence_tiers)  # type: ignore[arg-type]

    def satisfies_required_assurance(self) -> bool:
        return assurance_satisfies(self.derived_assurance, self.required_assurance)  # type: ignore[arg-type]

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CODE_CLAIM_CONTRACT_VERSION,
            "interface": CODE_CLAIM_RECORD_INTERFACE,
            "claim_family": self.claim_family,
            "status": self.status,
            "property_id": self.property_id,
            "obligation_id": self.obligation_id,
            "repository_id": self.repository_id,
            "repository_tree_id": self.repository_tree_id,
            "scope_ids": self.scope_ids,
            "premise_ids": self.premise_ids,
            "assumption_ids": self.assumption_ids,
            "producer_id": self.producer_id,
            "toolchain_id": self.toolchain_id,
            "policy_id": self.policy_id,
            "catalog_version": self.catalog_version,
            "evidence_ids": self.evidence_ids,
            "evidence_tiers": self.evidence_tiers,
            "required_assurance": self.required_assurance,
            "derived_assurance": self.derived_assurance,
            "invalidation_selectors": [
                s.to_dict() for s in self.invalidation_selectors
            ],
            "statement": self.statement,
            "cache_lookup": self.cache_lookup,
            "receipt_id": self.receipt_id,
            "template_id": self.template_id,
            "metadata": dict(self.metadata),
        }

    def to_record(self) -> dict[str, Any]:
        """Canonical payload plus non-recursive claim_id / content_id."""

        return {
            **self.to_dict(),
            "claim_id": self.claim_id,
            "content_id": self.content_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodeClaimRecord":
        if not isinstance(payload, Mapping):
            raise CodeClaimContractError("code claim record must be an object")
        schema = payload.get("schema")
        if schema not in (None, "", CODE_CLAIM_RECORD_SCHEMA):
            raise CodeClaimContractError(
                f"unsupported code-claim schema; use {CODE_CLAIM_RECORD_SCHEMA}"
            )
        version = payload.get("contract_version")
        if version not in (None, CODE_CLAIM_CONTRACT_VERSION):
            raise CodeClaimContractError(
                "unsupported code-claim contract version; rebuild with current contract"
            )

        raw_selectors = payload.get("invalidation_selectors") or ()
        selectors = tuple(
            item
            if isinstance(item, InvalidationSelector)
            else InvalidationSelector.from_dict(item)
            for item in raw_selectors
        )
        raw_tiers = payload.get("evidence_tiers") or ()
        result = cls(
            claim_family=payload.get("claim_family", ClaimFamily.UNSUPPORTED),
            status=payload.get("status", ClaimStatus.UNKNOWN),
            property_id=str(payload.get("property_id") or ""),
            obligation_id=str(payload.get("obligation_id") or ""),
            repository_id=str(payload.get("repository_id") or ""),
            repository_tree_id=str(
                payload.get("repository_tree_id") or payload.get("tree_id") or ""
            ),
            scope_ids=tuple(
                payload.get("scope_ids") or payload.get("ast_scope_ids") or ()
            ),
            premise_ids=tuple(payload.get("premise_ids") or ()),
            assumption_ids=tuple(payload.get("assumption_ids") or ()),
            producer_id=str(payload.get("producer_id") or ""),
            toolchain_id=str(payload.get("toolchain_id") or ""),
            policy_id=str(payload.get("policy_id") or ""),
            catalog_version=str(
                payload.get("catalog_version") or CLAIM_CATALOG_VERSION
            ),
            evidence_ids=tuple(payload.get("evidence_ids") or ()),
            evidence_tiers=tuple(raw_tiers),
            required_assurance=payload.get(
                "required_assurance", AssuranceLevel.KERNEL_VERIFIED
            ),
            derived_assurance=payload.get(
                "derived_assurance", AssuranceLevel.UNVERIFIED
            ),
            invalidation_selectors=selectors,
            statement=str(payload.get("statement") or ""),
            cache_lookup=str(payload.get("cache_lookup") or ""),
            receipt_id=str(payload.get("receipt_id") or ""),
            template_id=str(payload.get("template_id") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )
        claimed = payload.get("claim_id") or payload.get("content_id")
        if claimed and str(claimed) != result.claim_id:
            raise CodeClaimContractError(
                "code claim content identity does not match payload"
            )
        return result

    def with_updates(self, **changes: Any) -> "CodeClaimRecord":
        """Return a new record with selected fields replaced."""

        base = {
            "claim_family": self.claim_family,
            "status": self.status,
            "property_id": self.property_id,
            "obligation_id": self.obligation_id,
            "repository_id": self.repository_id,
            "repository_tree_id": self.repository_tree_id,
            "scope_ids": self.scope_ids,
            "premise_ids": self.premise_ids,
            "assumption_ids": self.assumption_ids,
            "producer_id": self.producer_id,
            "toolchain_id": self.toolchain_id,
            "policy_id": self.policy_id,
            "catalog_version": self.catalog_version,
            "evidence_ids": self.evidence_ids,
            "evidence_tiers": self.evidence_tiers,
            "required_assurance": self.required_assurance,
            "derived_assurance": self.derived_assurance,
            "invalidation_selectors": self.invalidation_selectors,
            "statement": self.statement,
            "cache_lookup": self.cache_lookup,
            "receipt_id": self.receipt_id,
            "template_id": self.template_id,
            "metadata": dict(self.metadata),
        }
        base.update(changes)
        return CodeClaimRecord(**base)


def claim_from_obligation(
    obligation: CodeProofObligation,
    *,
    property_id: str = "",
    claim_family: ClaimFamily | str | None = None,
    assumption_ids: Sequence[str] = (),
    producer_id: str = "",
    toolchain_id: str = "",
    policy_id: str = "",
    catalog_version: str = CLAIM_CATALOG_VERSION,
    status: ClaimStatus | str = ClaimStatus.OPEN,
    metadata: Mapping[str, Any] | None = None,
) -> CodeClaimRecord:
    """Normalize a reviewed obligation into an open CodeClaimRecord."""

    if not isinstance(obligation, CodeProofObligation):
        raise CodeClaimContractError("obligation must be a CodeProofObligation")
    family = resolve_claim_family(
        template_id=obligation.template_id,
        invariant_class=obligation.invariant_class,
        property_id=property_id,
        explicit=claim_family,
    )
    if family is ClaimFamily.UNSUPPORTED and obligation.template_id:
        # Explicit unsupported template remains unsupported status.
        status = ClaimStatus.UNSUPPORTED
    selectors = build_invalidation_selectors(
        repository_tree_id=obligation.repository_tree_id,
        scope_ids=obligation.ast_scope_ids,
        premise_ids=obligation.premise_ids,
        assumption_ids=assumption_ids,
        toolchain_id=toolchain_id,
        policy_id=policy_id or "",
        catalog_version=catalog_version,
        property_id=property_id,
        obligation_id=obligation.obligation_id,
        producer_id=producer_id,
        required_assurance=obligation.required_assurance,
    )
    return CodeClaimRecord(
        claim_family=family,
        status=status,
        property_id=property_id,
        obligation_id=obligation.obligation_id,
        repository_id=obligation.repository_id,
        repository_tree_id=obligation.repository_tree_id,
        scope_ids=obligation.ast_scope_ids,
        premise_ids=obligation.premise_ids,
        assumption_ids=tuple(assumption_ids),
        producer_id=producer_id,
        toolchain_id=toolchain_id,
        policy_id=policy_id,
        catalog_version=catalog_version,
        required_assurance=obligation.required_assurance,
        derived_assurance=AssuranceLevel.UNVERIFIED,
        invalidation_selectors=selectors,
        statement=obligation.statement,
        template_id=obligation.template_id,
        metadata=dict(metadata or {}),
    )


def claim_from_receipt(
    receipt: ProofReceipt,
    *,
    property_id: str = "",
    claim_family: ClaimFamily | str | None = None,
    assumption_ids: Sequence[str] = (),
    catalog_version: str = CLAIM_CATALOG_VERSION,
    prior: CodeClaimRecord | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CodeClaimRecord:
    """Project a ProofReceipt onto a CodeClaimRecord using existing assurance.

    Does not invent assurance: re-derives via assess_assurance / derive_verdict.
    """

    if not isinstance(receipt, ProofReceipt):
        raise CodeClaimContractError("receipt must be a ProofReceipt")

    assessment = assess_assurance(
        receipt.evidence,
        obligation_id=receipt.obligation_id,
        kernel_id=receipt.kernel_id,
        kernel_receipt_id=receipt.kernel_receipt_id,
        freshness=receipt.freshness,
    )
    derived = assessment.level
    verdict = derive_verdict(
        receipt.evidence,
        obligation_id=receipt.obligation_id,
        kernel_id=receipt.kernel_id,
        freshness=receipt.freshness,
    )

    tiers = tuple(evidence_kind_to_tier(item.kind) for item in receipt.evidence)
    # Cap derived assurance by tier ceiling (query/observation cannot mint kernel).
    ceiling = max_assurance_for_tiers(tiers) if tiers else AssuranceLevel.UNVERIFIED
    if derived.rank > ceiling.rank:
        derived = ceiling

    required = (
        prior.required_assurance
        if prior is not None
        else AssuranceLevel.KERNEL_VERIFIED
    )
    if receipt.freshness is not EvidenceFreshness.CURRENT:
        status = ClaimStatus.STALE
        # Stale evidence never retains authoritative assurance for gates.
        derived = AssuranceLevel.UNVERIFIED
    elif verdict is ProofVerdict.DISPROVED:
        status = ClaimStatus.REFUTED
    elif verdict is ProofVerdict.UNSUPPORTED:
        status = ClaimStatus.UNSUPPORTED
    elif verdict is ProofVerdict.PROVED and assurance_satisfies(derived, required):
        status = ClaimStatus.SATISFIED
    elif verdict is ProofVerdict.PROVED:
        status = ClaimStatus.OPEN  # proved below required assurance
    else:
        status = ClaimStatus.OPEN

    if claim_family is not None:
        family = resolve_claim_family(
            template_id=prior.template_id if prior else "",
            property_id=property_id or (prior.property_id if prior else ""),
            explicit=claim_family,
        )
    elif prior is not None:
        family = prior.claim_family  # type: ignore[assignment]
    else:
        family = resolve_claim_family(
            property_id=property_id,
            explicit=None,
        )

    evidence_ids = tuple(
        sorted({item.evidence_id for item in receipt.evidence if item.evidence_id})
    )
    scope_ids = receipt.ast_scope_ids
    premise_ids = receipt.premise_ids
    assumptions = tuple(assumption_ids) or (
        prior.assumption_ids if prior is not None else ()
    )
    prop_id = property_id or (prior.property_id if prior else "")
    catalog = catalog_version or (prior.catalog_version if prior else CLAIM_CATALOG_VERSION)

    selectors = build_invalidation_selectors(
        repository_tree_id=receipt.repository_tree_id,
        scope_ids=scope_ids,
        premise_ids=premise_ids,
        assumption_ids=assumptions,
        toolchain_id=receipt.toolchain_id,
        policy_id=receipt.policy_id,
        catalog_version=catalog,
        property_id=prop_id,
        obligation_id=receipt.obligation_id,
        producer_id=receipt.provider_id,
        required_assurance=required,
    )
    # Always include evidence-freshness selector.
    selectors = tuple(
        sorted(
            set(selectors)
            | {
                InvalidationSelector(
                    kind=InvalidationSelectorKind.EVIDENCE_FRESHNESS,
                    value=receipt.freshness.value,
                    reason_code="stale_or_unknown_evidence",
                )
            },
            key=lambda s: (s.kind.value, s.value, s.reason_code),
        )
    )

    meta = dict(prior.metadata if prior else {})
    meta.update(dict(metadata or {}))
    meta["assurance_reason_codes"] = list(assessment.reason_codes)
    meta["receipt_verdict"] = verdict.value

    return CodeClaimRecord(
        claim_family=family,
        status=status,
        property_id=prop_id,
        obligation_id=receipt.obligation_id,
        repository_id=receipt.repository_id,
        repository_tree_id=receipt.repository_tree_id,
        scope_ids=scope_ids,
        premise_ids=premise_ids,
        assumption_ids=assumptions,
        producer_id=receipt.provider_id,
        toolchain_id=receipt.toolchain_id,
        policy_id=receipt.policy_id,
        catalog_version=catalog,
        evidence_ids=evidence_ids,
        evidence_tiers=tiers,
        required_assurance=required,
        derived_assurance=derived,
        invalidation_selectors=selectors,
        statement=prior.statement if prior else "",
        cache_lookup=CACHE_LOOKUP_HIT if prior and prior.cache_lookup == CACHE_LOOKUP_HIT else "",
        receipt_id=receipt.receipt_id,
        template_id=prior.template_id if prior else "",
        metadata=meta,
    )


def claim_from_implementation_evidence(
    evidence: ImplementationResultEvidence,
    *,
    property_id: str = "",
    obligation_id: str = "",
    claim_family: ClaimFamily | str | None = None,
    required_assurance: AssuranceLevel | str = AssuranceLevel.KERNEL_VERIFIED,
    catalog_version: str = CLAIM_CATALOG_VERSION,
    policy_id: str = "",
    toolchain_id: str = "",
    status: ClaimStatus | str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CodeClaimRecord:
    """Bind a bounded observation; never mints kernel assurance."""

    if not isinstance(evidence, ImplementationResultEvidence):
        raise CodeClaimContractError(
            "evidence must be ImplementationResultEvidence"
        )
    tier = implementation_kind_to_tier(evidence.kind)
    family = resolve_claim_family(
        property_id=property_id,
        explicit=claim_family,
    )
    # Observations cannot satisfy kernel-required claims.
    required = (
        required_assurance
        if isinstance(required_assurance, AssuranceLevel)
        else AssuranceLevel(str(required_assurance))
    )
    derived = tier.max_assurance  # CANDIDATE ceiling
    if evidence.contradictory:
        claim_status = ClaimStatus.REFUTED
    elif status is not None:
        claim_status = (
            status if isinstance(status, ClaimStatus) else ClaimStatus(str(status))
        )
    elif evidence.passed:
        # Passing observation leaves claim open unless required assurance is
        # within the observation ceiling (rare).
        claim_status = (
            ClaimStatus.SATISFIED
            if assurance_satisfies(derived, required)
            else ClaimStatus.OPEN
        )
    else:
        claim_status = ClaimStatus.OPEN

    selectors = build_invalidation_selectors(
        repository_tree_id=evidence.repository_tree_id,
        scope_ids=evidence.scope_ids,
        assumption_ids=evidence.assumption_ids,
        toolchain_id=toolchain_id,
        policy_id=policy_id,
        catalog_version=catalog_version,
        property_id=property_id,
        obligation_id=obligation_id,
        producer_id=evidence.producer_id,
        required_assurance=required,
    )
    return CodeClaimRecord(
        claim_family=family if family is not ClaimFamily.UNSUPPORTED or property_id or obligation_id else ClaimFamily.BEHAVIORAL_INVARIANT,
        status=claim_status,
        property_id=property_id,
        obligation_id=obligation_id,
        repository_id=evidence.repository_id,
        repository_tree_id=evidence.repository_tree_id,
        scope_ids=evidence.scope_ids,
        premise_ids=(),
        assumption_ids=evidence.assumption_ids,
        producer_id=evidence.producer_id,
        toolchain_id=toolchain_id,
        policy_id=policy_id,
        catalog_version=catalog_version,
        evidence_ids=(evidence.evidence_id,),
        evidence_tiers=(tier,),
        required_assurance=required,
        derived_assurance=derived,
        invalidation_selectors=selectors,
        statement=evidence.subject or evidence.command,
        metadata={
            **dict(metadata or {}),
            "implementation_kind": evidence.kind.value,
            "passed": evidence.passed,
            "observation_only": True,
        },
    )


def claim_from_query_fact(
    *,
    fact_id: str,
    repository_id: str,
    repository_tree_id: str,
    scope_ids: Sequence[str] = (),
    property_id: str = "",
    obligation_id: str = "",
    claim_family: ClaimFamily | str = ClaimFamily.DEPENDENCY_REACHABILITY,
    graphrag: bool = False,
    catalog_version: str = CLAIM_CATALOG_VERSION,
    statement: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> CodeClaimRecord:
    """Bind a repository query / GraphRAG projection as a non-authoritative fact.

    Query facts never mint kernel assurance; derived assurance stays UNVERIFIED.
    """

    fact = _norm_text(fact_id, field_name="fact_id", required=True)
    tier = EvidenceTier.GRAPHRAG_FACT if graphrag else EvidenceTier.QUERY_FACT
    family = resolve_claim_family(
        property_id=property_id,
        explicit=claim_family,
    )
    # Ensure we have a reviewed binding or fail closed.
    if not property_id and not obligation_id:
        # Query facts may exist without obligations when property_id is set; if
        # neither is set, require an explicit reviewed family other than freeform.
        if family is ClaimFamily.UNSUPPORTED:
            raise CodeClaimContractError(
                "query facts require property_id, obligation_id, or a reviewed family"
            )
    selectors = build_invalidation_selectors(
        repository_tree_id=repository_tree_id,
        scope_ids=scope_ids,
        catalog_version=catalog_version,
        property_id=property_id,
        obligation_id=obligation_id,
    )
    return CodeClaimRecord(
        claim_family=family,
        status=ClaimStatus.OPEN,
        property_id=property_id,
        obligation_id=obligation_id,
        repository_id=repository_id,
        repository_tree_id=repository_tree_id,
        scope_ids=tuple(scope_ids),
        evidence_ids=(fact,),
        evidence_tiers=(tier,),
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
        derived_assurance=AssuranceLevel.UNVERIFIED,
        invalidation_selectors=selectors,
        statement=statement,
        catalog_version=catalog_version,
        metadata={
            **dict(metadata or {}),
            "query_fact": True,
            "graphrag": graphrag,
            "non_authoritative": True,
        },
    )


def apply_cache_lookup(
    claim: CodeClaimRecord,
    *,
    outcome: str,
    receipt: ProofReceipt | None = None,
) -> CodeClaimRecord:
    """Update claim lifecycle from a trust-aware proof-cache lookup.

    Miss → open (never refuted).  Hit with receipt → re-derive via
    :func:`claim_from_receipt`.  Stale/rejected cache entries → stale/open.
    """

    if not isinstance(claim, CodeClaimRecord):
        raise CodeClaimContractError("claim must be a CodeClaimRecord")
    normalized = str(outcome or "").strip().lower()
    if normalized in {CACHE_LOOKUP_MISS, "cache_miss", "miss"}:
        return claim.with_updates(
            status=cache_miss_status(previously=claim.status),  # type: ignore[arg-type]
            cache_lookup=CACHE_LOOKUP_MISS,
            # Do not clear prior evidence ids; miss only means no memoized hit.
            derived_assurance=AssuranceLevel.UNVERIFIED
            if claim.status is ClaimStatus.SATISFIED
            else claim.derived_assurance,
        )
    if normalized in {CACHE_LOOKUP_STALE, "stale_cache_entry", "stale"}:
        return claim.with_updates(
            status=ClaimStatus.STALE,
            cache_lookup=CACHE_LOOKUP_STALE,
            derived_assurance=AssuranceLevel.UNVERIFIED,
        )
    if normalized in {CACHE_LOOKUP_REJECTED, "rejected", "poisoned"}:
        return claim.with_updates(
            status=ClaimStatus.OPEN,
            cache_lookup=CACHE_LOOKUP_REJECTED,
            derived_assurance=AssuranceLevel.UNVERIFIED,
        )
    if normalized in {CACHE_LOOKUP_HIT, "hit"}:
        if receipt is None:
            raise CodeClaimContractError("cache hit requires a ProofReceipt")
        updated = claim_from_receipt(receipt, prior=claim)
        return updated.with_updates(cache_lookup=CACHE_LOOKUP_HIT)
    raise CodeClaimContractError(f"unknown cache lookup outcome: {outcome!r}")


def mark_claim_stale(
    claim: CodeClaimRecord,
    *,
    reason_code: str = "stale_or_unknown_evidence",
    selector: InvalidationSelector | None = None,
) -> CodeClaimRecord:
    """Transition a claim to stale without treating it as refuted."""

    selectors = list(claim.invalidation_selectors)
    if selector is not None:
        selectors.append(selector)
    else:
        selectors.append(
            InvalidationSelector(
                kind=InvalidationSelectorKind.EVIDENCE_FRESHNESS,
                value="stale",
                reason_code=reason_code,
            )
        )
    meta = dict(claim.metadata)
    meta["stale_reason"] = reason_code
    return claim.with_updates(
        status=ClaimStatus.STALE,
        derived_assurance=AssuranceLevel.UNVERIFIED,
        invalidation_selectors=tuple(selectors),
        metadata=meta,
    )


def evaluate_invalidation(
    claim: CodeClaimRecord,
    *,
    current_tree_id: str = "",
    current_toolchain_id: str = "",
    current_policy_id: str = "",
    current_catalog_version: str = "",
    current_premise_digest: str = "",
    current_assumption_digest: str = "",
) -> CodeClaimRecord:
    """Return claim unchanged or stale if any bound selector no longer matches."""

    reasons: list[str] = []
    for selector in claim.invalidation_selectors:
        kind = selector.kind
        if (
            kind is InvalidationSelectorKind.REPOSITORY_TREE
            and current_tree_id
            and selector.value != current_tree_id
        ):
            reasons.append(selector.reason_code or "stale_candidate_tree")
        elif (
            kind is InvalidationSelectorKind.TOOLCHAIN
            and current_toolchain_id
            and selector.value != current_toolchain_id
        ):
            reasons.append(selector.reason_code or "toolchain_drift")
        elif (
            kind is InvalidationSelectorKind.POLICY
            and current_policy_id
            and selector.value != current_policy_id
        ):
            reasons.append(selector.reason_code or "policy_drift")
        elif (
            kind is InvalidationSelectorKind.CATALOG
            and current_catalog_version
            and selector.value != current_catalog_version
        ):
            reasons.append(selector.reason_code or "catalog_drift")
        elif (
            kind is InvalidationSelectorKind.PREMISE_SET
            and current_premise_digest
            and selector.value != current_premise_digest
        ):
            reasons.append(selector.reason_code or "changed_premises")
        elif (
            kind is InvalidationSelectorKind.ASSUMPTION_SET
            and current_assumption_digest
            and selector.value != current_assumption_digest
        ):
            reasons.append(selector.reason_code or "changed_assumptions")
    if not reasons:
        return claim
    return mark_claim_stale(claim, reason_code=reasons[0])


def reject_natural_language_claim(
    statement: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed for arbitrary NL claims (always raises)."""

    raise CodeClaimContractError(
        "arbitrary natural-language claims fail closed; bind a reviewed "
        f"property_id or obligation_id (statement_len={len(statement or '')}, "
        f"metadata_keys={sorted((metadata or {}).keys())})"
    )


def build_open_claim(
    *,
    property_id: str,
    claim_family: ClaimFamily | str,
    repository_id: str,
    repository_tree_id: str,
    obligation_id: str = "",
    scope_ids: Sequence[str] = (),
    premise_ids: Sequence[str] = (),
    assumption_ids: Sequence[str] = (),
    producer_id: str = "",
    toolchain_id: str = "",
    policy_id: str = "",
    catalog_version: str = CLAIM_CATALOG_VERSION,
    required_assurance: AssuranceLevel | str = AssuranceLevel.KERNEL_VERIFIED,
    statement: str = "",
    template_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> CodeClaimRecord:
    """Construct an open claim bound to a reviewed property (fail-closed)."""

    prop = _norm_text(property_id, field_name="property_id", required=True)
    family = resolve_claim_family(
        template_id=template_id,
        property_id=prop,
        explicit=claim_family,
    )
    if family is ClaimFamily.UNSUPPORTED and not obligation_id:
        raise CodeClaimContractError(
            "cannot open an unsupported claim without a reviewed obligation"
        )
    selectors = build_invalidation_selectors(
        repository_tree_id=repository_tree_id,
        scope_ids=scope_ids,
        premise_ids=premise_ids,
        assumption_ids=assumption_ids,
        toolchain_id=toolchain_id,
        policy_id=policy_id,
        catalog_version=catalog_version,
        property_id=prop,
        obligation_id=obligation_id,
        producer_id=producer_id,
        required_assurance=required_assurance,
    )
    return CodeClaimRecord(
        claim_family=family,
        status=ClaimStatus.OPEN,
        property_id=prop,
        obligation_id=obligation_id,
        repository_id=repository_id,
        repository_tree_id=repository_tree_id,
        scope_ids=tuple(scope_ids),
        premise_ids=tuple(premise_ids),
        assumption_ids=tuple(assumption_ids),
        producer_id=producer_id,
        toolchain_id=toolchain_id,
        policy_id=policy_id,
        catalog_version=catalog_version,
        required_assurance=required_assurance,
        derived_assurance=AssuranceLevel.UNVERIFIED,
        invalidation_selectors=selectors,
        statement=statement,
        template_id=template_id,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "CODE_CLAIM_RECORD_INTERFACE",
    "CODE_CLAIM_RECORD_SCHEMA",
    "CODE_CLAIM_CONTRACT_VERSION",
    "CLAIM_CATALOG_VERSION",
    "CACHE_LOOKUP_HIT",
    "CACHE_LOOKUP_MISS",
    "CACHE_LOOKUP_STALE",
    "CACHE_LOOKUP_REJECTED",
    "CodeClaimContractError",
    "ClaimFamily",
    "ClaimStatus",
    "EvidenceTier",
    "InvalidationSelectorKind",
    "InvalidationSelector",
    "CodeClaimRecord",
    "resolve_claim_family",
    "evidence_kind_to_tier",
    "implementation_kind_to_tier",
    "max_assurance_for_tiers",
    "tiers_can_independently_mint_kernel",
    "cache_miss_status",
    "build_invalidation_selectors",
    "claim_from_obligation",
    "claim_from_receipt",
    "claim_from_implementation_evidence",
    "claim_from_query_fact",
    "apply_cache_lookup",
    "mark_claim_stale",
    "evaluate_invalidation",
    "reject_natural_language_claim",
    "build_open_claim",
    # Re-exports used by adapters / tests without a second assurance model.
    "AssuranceLevel",
    "ProofEvidence",
    "ProofReceipt",
    "ImplementationResultEvidence",
    "assess_assurance",
    "derive_assurance",
]
