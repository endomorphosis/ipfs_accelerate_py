"""Typed claim/evidence lifecycle for codebase-proof work (CBP-025).

Interface: ``CodeClaimRecord@1``

This module is a **normalization layer** over existing proof and implementation
evidence types.  It does **not** re-derive assurance, replace
:class:`ProofReceipt`, or act as a second proof-cache trust root.

Lifecycle statuses are explicit and fail-closed:

* ``unknown`` — no claim yet for the property/scope
* ``open`` — claim exists; required evidence not yet satisfied
* ``satisfied`` — bound evidence meets required assurance / observation rules
* ``refuted`` — bound evidence shows the property fails
* ``unsupported`` — no reviewed template/shape can express the claim
* ``not_measured`` — measurement intentionally deferred
* ``stale`` — previously satisfied, but invalidation selectors fired

A cache **miss** is not a refutation: callers should leave status ``open`` or
``not_measured`` until independent evidence is obtained.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .formal_verification_contracts import (
    AssuranceLevel,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    ProofEvidence,
    ProofReceipt,
    content_identity,
)
from .code_proof_obligations import ImplementationResultEvidence


CODE_CLAIM_RECORD_INTERFACE: Final = "CodeClaimRecord@1"
CODE_CLAIM_RECORD_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-claim-record@1"
)
CODE_CLAIM_CONTRACT_VERSION: Final = "1"


class CodeClaimContractError(ValueError):
    """Raised when a claim/evidence contract is malformed or unsafe."""


class ClaimFamily(str, Enum):
    """Closed set of claim families (no free-form NL invent)."""

    CODE_INVARIANT = "code_invariant"
    PROTOCOL = "protocol"
    SECURITY = "security"
    SEMANTIC_EQUIVALENCE = "semantic_equivalence"
    DEPENDENCY = "dependency"
    API_CONTRACT = "api_contract"
    STRUCTURAL = "structural"
    BENCHMARK = "benchmark"


class ClaimStatus(str, Enum):
    """Lifecycle of one claim relative to bound evidence."""

    UNKNOWN = "unknown"
    OPEN = "open"
    SATISFIED = "satisfied"
    REFUTED = "refuted"
    UNSUPPORTED = "unsupported"
    NOT_MEASURED = "not_measured"
    STALE = "stale"


class EvidenceTier(str, Enum):
    """Coarse evidence tiers used for policy (not assurance derivation)."""

    QUERY_FACT = "query_fact"
    OBSERVATION = "observation"
    SOLVER_CANDIDATE = "solver_candidate"
    KERNEL_PROOF = "kernel_proof"
    CRYPTOGRAPHIC_ATTESTATION = "cryptographic_attestation"


# Evidence kinds that may never independently mint kernel assurance.
_NON_KERNEL_MINTING_KINDS = frozenset(
    {
        EvidenceKind.LLM_OUTPUT,
        EvidenceKind.ATP_CANDIDATE,
        EvidenceKind.SMT_CANDIDATE,
        getattr(EvidenceKind, "QUERY_RESULT", None),
        getattr(EvidenceKind, "RETRIEVAL", None),
        getattr(EvidenceKind, "GRAPHRAG", None),
    }
    - {None}
)


def _norm_str(value: Any, *, field_name: str, required: bool = True) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise CodeClaimContractError(f"{field_name} must be a non-empty string")
    return text


def _sorted_ids(values: Iterable[Any], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        values = (values,)
    return tuple(sorted({str(v).strip() for v in values if str(v).strip()}))


def _enum(value: Any, enum_cls: type[Enum], *, field_name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value))
    except Exception as exc:  # noqa: BLE001
        raise CodeClaimContractError(
            f"{field_name} must be a valid {enum_cls.__name__}"
        ) from exc


def evidence_tier_for_proof_evidence(evidence: ProofEvidence) -> EvidenceTier:
    """Map a :class:`ProofEvidence` row to a coarse tier."""

    if not isinstance(evidence, ProofEvidence):
        raise CodeClaimContractError("evidence must be ProofEvidence")
    kind = evidence.kind
    if kind is EvidenceKind.CRYPTOGRAPHIC_ATTESTATION:
        return EvidenceTier.CRYPTOGRAPHIC_ATTESTATION
    if kind is EvidenceKind.KERNEL_VERIFICATION:
        return EvidenceTier.KERNEL_PROOF
    if kind in (
        EvidenceKind.ATP_CANDIDATE,
        EvidenceKind.SMT_CANDIDATE,
        EvidenceKind.SOLVER_RESULT,
        EvidenceKind.LLM_OUTPUT,
    ):
        return EvidenceTier.SOLVER_CANDIDATE
    if kind in (EvidenceKind.TEST_RESULT, EvidenceKind.STATIC_ANALYSIS):
        return EvidenceTier.OBSERVATION
    if kind is EvidenceKind.CACHE_ENTRY:
        return EvidenceTier.OBSERVATION
    # Unknown/query-like kinds stay non-kernel query facts.
    name = str(getattr(kind, "value", kind)).lower()
    if "query" in name or "retrieval" in name or "graph" in name:
        return EvidenceTier.QUERY_FACT
    return EvidenceTier.QUERY_FACT

def evidence_tier_for_implementation(
    evidence: ImplementationResultEvidence,
) -> EvidenceTier:
    """Implementation observations are always observation-tier."""

    if not isinstance(evidence, ImplementationResultEvidence):
        raise CodeClaimContractError(
            "evidence must be ImplementationResultEvidence"
        )
    return EvidenceTier.OBSERVATION


def can_mint_kernel_assurance(
    *,
    tiers: Sequence[EvidenceTier | str],
    proof_kinds: Sequence[EvidenceKind | str] = (),
) -> bool:
    """Return whether the tier/kind set may contribute to kernel assurance.

    Query facts, observations, and solver/model candidates alone cannot mint
    kernel assurance.  A kernel proof tier is required.
    """

    normalized = {
        t if isinstance(t, EvidenceTier) else EvidenceTier(str(t))
        for t in tiers
    }
    if EvidenceTier.KERNEL_PROOF in normalized:
        return True
    if EvidenceTier.CRYPTOGRAPHIC_ATTESTATION in normalized:
        # Attestation still requires an underlying kernel receipt elsewhere.
        return False
    for kind in proof_kinds:
        k = kind if isinstance(kind, EvidenceKind) else EvidenceKind(str(kind))
        if k is EvidenceKind.KERNEL_VERIFICATION:
            return True
        if k in _NON_KERNEL_MINTING_KINDS:
            continue
    return False


@dataclass(frozen=True)
class InvalidationSelector:
    """What changes force a satisfied claim into ``stale``."""

    repository_tree_ids: tuple[str, ...] = ()
    scope_ids: tuple[str, ...] = ()
    premise_ids: tuple[str, ...] = ()
    assumption_ids: tuple[str, ...] = ()
    policy_ids: tuple[str, ...] = ()
    toolchain_ids: tuple[str, ...] = ()
    catalog_versions: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_tree_ids": list(self.repository_tree_ids),
            "scope_ids": list(self.scope_ids),
            "premise_ids": list(self.premise_ids),
            "assumption_ids": list(self.assumption_ids),
            "policy_ids": list(self.policy_ids),
            "toolchain_ids": list(self.toolchain_ids),
            "catalog_versions": list(self.catalog_versions),
            "paths": list(self.paths),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "InvalidationSelector":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise CodeClaimContractError("invalidation selector must be an object")
        return cls(
            repository_tree_ids=_sorted_ids(
                payload.get("repository_tree_ids") or (),
                field_name="repository_tree_ids",
            ),
            scope_ids=_sorted_ids(payload.get("scope_ids") or (), field_name="scope_ids"),
            premise_ids=_sorted_ids(
                payload.get("premise_ids") or (), field_name="premise_ids"
            ),
            assumption_ids=_sorted_ids(
                payload.get("assumption_ids") or (), field_name="assumption_ids"
            ),
            policy_ids=_sorted_ids(
                payload.get("policy_ids") or (), field_name="policy_ids"
            ),
            toolchain_ids=_sorted_ids(
                payload.get("toolchain_ids") or (), field_name="toolchain_ids"
            ),
            catalog_versions=_sorted_ids(
                payload.get("catalog_versions") or (),
                field_name="catalog_versions",
            ),
            paths=_sorted_ids(payload.get("paths") or (), field_name="paths"),
        )

    def is_triggered_by(
        self,
        *,
        repository_tree_id: str = "",
        scope_ids: Sequence[str] = (),
        premise_ids: Sequence[str] = (),
        assumption_ids: Sequence[str] = (),
        policy_id: str = "",
        toolchain_id: str = "",
        catalog_version: str = "",
        paths: Sequence[str] = (),
    ) -> bool:
        """Return True when any selector dimension mismatches current inputs."""

        if self.repository_tree_ids and repository_tree_id:
            if repository_tree_id not in self.repository_tree_ids:
                return True
        if self.scope_ids and scope_ids:
            if not set(self.scope_ids).intersection(scope_ids):
                # scopes changed away from bound set
                if set(scope_ids) != set(self.scope_ids):
                    return True
        if self.premise_ids and premise_ids:
            if set(premise_ids) != set(self.premise_ids):
                return True
        if self.assumption_ids and assumption_ids:
            if set(assumption_ids) != set(self.assumption_ids):
                return True
        if self.policy_ids and policy_id and policy_id not in self.policy_ids:
            return True
        if (
            self.toolchain_ids
            and toolchain_id
            and toolchain_id not in self.toolchain_ids
        ):
            return True
        if (
            self.catalog_versions
            and catalog_version
            and catalog_version not in self.catalog_versions
        ):
            return True
        if self.paths and paths:
            if set(paths).intersection(self.paths) or set(paths) != set(self.paths):
                # any path change among watched paths
                if set(paths) != set(self.paths):
                    return True
        return False


@dataclass(frozen=True)
class CodeClaimRecord:
    """Content-addressed claim with bound evidence references and lifecycle."""

    property_id: str
    claim_family: ClaimFamily
    status: ClaimStatus
    repository_id: str
    repository_tree_id: str
    obligation_ids: tuple[str, ...] = ()
    scope_ids: tuple[str, ...] = ()
    premise_ids: tuple[str, ...] = ()
    assumption_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    evidence_tiers: tuple[EvidenceTier, ...] = ()
    required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED
    producer_id: str = ""
    toolchain_id: str = ""
    policy_id: str = ""
    catalog_version: str = ""
    invalidation: InvalidationSelector = field(default_factory=InvalidationSelector)
    natural_language_allowed: bool = False
    statement: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "property_id",
            _norm_str(self.property_id, field_name="property_id"),
        )
        object.__setattr__(
            self,
            "claim_family",
            _enum(self.claim_family, ClaimFamily, field_name="claim_family"),
        )
        object.__setattr__(
            self, "status", _enum(self.status, ClaimStatus, field_name="status")
        )
        object.__setattr__(
            self,
            "repository_id",
            _norm_str(self.repository_id, field_name="repository_id"),
        )
        object.__setattr__(
            self,
            "repository_tree_id",
            _norm_str(self.repository_tree_id, field_name="repository_tree_id"),
        )
        object.__setattr__(
            self,
            "obligation_ids",
            _sorted_ids(self.obligation_ids, field_name="obligation_ids"),
        )
        object.__setattr__(
            self, "scope_ids", _sorted_ids(self.scope_ids, field_name="scope_ids")
        )
        object.__setattr__(
            self,
            "premise_ids",
            _sorted_ids(self.premise_ids, field_name="premise_ids"),
        )
        object.__setattr__(
            self,
            "assumption_ids",
            _sorted_ids(self.assumption_ids, field_name="assumption_ids"),
        )
        object.__setattr__(
            self,
            "evidence_ids",
            _sorted_ids(self.evidence_ids, field_name="evidence_ids"),
        )
        tiers: list[EvidenceTier] = []
        for tier in self.evidence_tiers:
            tiers.append(
                tier
                if isinstance(tier, EvidenceTier)
                else EvidenceTier(str(tier))
            )
        object.__setattr__(
            self,
            "evidence_tiers",
            tuple(sorted(tiers, key=lambda item: item.value)),
        )
        assurance = self.required_assurance
        if not isinstance(assurance, AssuranceLevel):
            assurance = AssuranceLevel(str(assurance))
        object.__setattr__(self, "required_assurance", assurance)
        object.__setattr__(
            self, "producer_id", str(self.producer_id or "").strip()
        )
        object.__setattr__(
            self, "toolchain_id", str(self.toolchain_id or "").strip()
        )
        object.__setattr__(self, "policy_id", str(self.policy_id or "").strip())
        object.__setattr__(
            self, "catalog_version", str(self.catalog_version or "").strip()
        )
        if not isinstance(self.invalidation, InvalidationSelector):
            object.__setattr__(
                self,
                "invalidation",
                InvalidationSelector.from_dict(self.invalidation),  # type: ignore[arg-type]
            )
        if not isinstance(self.natural_language_allowed, bool):
            raise CodeClaimContractError(
                "natural_language_allowed must be a boolean"
            )
        statement = str(self.statement or "").strip()
        # Fail closed: free-form NL claims without a property id family path
        # are rejected when natural_language_allowed is false and statement is
        # the only content (property_id already required).
        if statement and not self.natural_language_allowed:
            # Allow short reviewed titles; reject long unstructured prose.
            if len(statement) > 280 or "\n" in statement:
                raise CodeClaimContractError(
                    "arbitrary natural-language claims fail closed; "
                    "set natural_language_allowed only for audited prose"
                )
        object.__setattr__(self, "statement", statement)
        if not isinstance(self.metadata, Mapping):
            raise CodeClaimContractError("metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

        # Kernel minting guard: observations/query alone cannot claim satisfied
        # at KERNEL_VERIFIED unless natural_language_allowed bypass is off and
        # tiers include kernel proof.
        if (
            self.status is ClaimStatus.SATISFIED
            and self.required_assurance
            in (AssuranceLevel.KERNEL_VERIFIED, AssuranceLevel.ATTESTED)
            and not can_mint_kernel_assurance(tiers=self.evidence_tiers)
        ):
            raise CodeClaimContractError(
                "query/observation/candidate tiers cannot independently mint "
                "kernel-level claim satisfaction"
            )

    @property
    def claim_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": CODE_CLAIM_RECORD_SCHEMA,
            "interface": CODE_CLAIM_RECORD_INTERFACE,
            "contract_version": CODE_CLAIM_CONTRACT_VERSION,
            "property_id": self.property_id,
            "claim_family": self.claim_family.value,
            "status": self.status.value,
            "repository_id": self.repository_id,
            "repository_tree_id": self.repository_tree_id,
            "obligation_ids": list(self.obligation_ids),
            "scope_ids": list(self.scope_ids),
            "premise_ids": list(self.premise_ids),
            "assumption_ids": list(self.assumption_ids),
            "evidence_ids": list(self.evidence_ids),
            "evidence_tiers": [tier.value for tier in self.evidence_tiers],
            "required_assurance": self.required_assurance.value,
            "producer_id": self.producer_id,
            "toolchain_id": self.toolchain_id,
            "policy_id": self.policy_id,
            "catalog_version": self.catalog_version,
            "invalidation": self.invalidation.to_dict(),
            "natural_language_allowed": self.natural_language_allowed,
            "statement": self.statement,
            "metadata": dict(self.metadata),
        }
        if include_id:
            payload["claim_id"] = content_identity(
                {k: v for k, v in payload.items() if k != "claim_id"}
            )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodeClaimRecord":
        if not isinstance(payload, Mapping):
            raise CodeClaimContractError("claim record must be an object")
        schema = payload.get("schema")
        if schema not in (None, CODE_CLAIM_RECORD_SCHEMA):
            raise CodeClaimContractError("unsupported code-claim schema")
        record = cls(
            property_id=str(payload.get("property_id") or ""),
            claim_family=ClaimFamily(str(payload.get("claim_family"))),
            status=ClaimStatus(str(payload.get("status"))),
            repository_id=str(payload.get("repository_id") or ""),
            repository_tree_id=str(payload.get("repository_tree_id") or ""),
            obligation_ids=tuple(payload.get("obligation_ids") or ()),
            scope_ids=tuple(payload.get("scope_ids") or ()),
            premise_ids=tuple(payload.get("premise_ids") or ()),
            assumption_ids=tuple(payload.get("assumption_ids") or ()),
            evidence_ids=tuple(payload.get("evidence_ids") or ()),
            evidence_tiers=tuple(
                EvidenceTier(str(t)) for t in (payload.get("evidence_tiers") or ())
            ),
            required_assurance=AssuranceLevel(
                str(
                    payload.get("required_assurance")
                    or AssuranceLevel.KERNEL_VERIFIED.value
                )
            ),
            producer_id=str(payload.get("producer_id") or ""),
            toolchain_id=str(payload.get("toolchain_id") or ""),
            policy_id=str(payload.get("policy_id") or ""),
            catalog_version=str(payload.get("catalog_version") or ""),
            invalidation=InvalidationSelector.from_dict(
                payload.get("invalidation")
            ),
            natural_language_allowed=bool(
                payload.get("natural_language_allowed", False)
            ),
            statement=str(payload.get("statement") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )
        claimed = payload.get("claim_id")
        if claimed is not None and str(claimed) != record.claim_id:
            raise CodeClaimContractError("claim_id does not match content")
        return record


def open_claim(
    *,
    property_id: str,
    claim_family: ClaimFamily | str,
    repository_id: str,
    repository_tree_id: str,
    obligation_ids: Sequence[str] = (),
    scope_ids: Sequence[str] = (),
    premise_ids: Sequence[str] = (),
    assumption_ids: Sequence[str] = (),
    required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED,
    producer_id: str = "",
    toolchain_id: str = "",
    policy_id: str = "",
    catalog_version: str = "",
    statement: str = "",
) -> CodeClaimRecord:
    """Construct an ``open`` claim with invalidation selectors pre-bound."""

    inv = InvalidationSelector(
        repository_tree_ids=(repository_tree_id,) if repository_tree_id else (),
        scope_ids=tuple(scope_ids),
        premise_ids=tuple(premise_ids),
        assumption_ids=tuple(assumption_ids),
        policy_ids=(policy_id,) if policy_id else (),
        toolchain_ids=(toolchain_id,) if toolchain_id else (),
        catalog_versions=(catalog_version,) if catalog_version else (),
    )
    return CodeClaimRecord(
        property_id=property_id,
        claim_family=claim_family,  # type: ignore[arg-type]
        status=ClaimStatus.OPEN,
        repository_id=repository_id,
        repository_tree_id=repository_tree_id,
        obligation_ids=tuple(obligation_ids),
        scope_ids=tuple(scope_ids),
        premise_ids=tuple(premise_ids),
        assumption_ids=tuple(assumption_ids),
        required_assurance=required_assurance,
        producer_id=producer_id,
        toolchain_id=toolchain_id,
        policy_id=policy_id,
        catalog_version=catalog_version,
        invalidation=inv,
        statement=statement,
    )


def claim_from_proof_receipt(
    receipt: ProofReceipt,
    *,
    property_id: str,
    claim_family: ClaimFamily | str = ClaimFamily.CODE_INVARIANT,
    catalog_version: str = "",
    status: ClaimStatus | str | None = None,
) -> CodeClaimRecord:
    """Project a :class:`ProofReceipt` into a claim record (no re-derivation)."""

    if not isinstance(receipt, ProofReceipt):
        raise CodeClaimContractError("receipt must be a ProofReceipt")
    tiers = tuple(
        evidence_tier_for_proof_evidence(item) for item in receipt.evidence
    )
    evidence_ids = tuple(
        sorted(
            {
                str(getattr(item, "artifact_id", "") or "").strip()
                for item in receipt.evidence
                if str(getattr(item, "artifact_id", "") or "").strip()
            }
            | {str(receipt.receipt_id)}
        )
    )
    if status is None:
        if getattr(receipt, "freshness", None) is EvidenceFreshness.STALE:
            resolved = ClaimStatus.STALE
        elif receipt.authoritative_assurance is AssuranceLevel.CANDIDATE:
            resolved = ClaimStatus.OPEN
        elif receipt.authoritative_assurance in (
            AssuranceLevel.UNVERIFIED,
        ):
            resolved = ClaimStatus.OPEN
        elif can_mint_kernel_assurance(tiers=tiers) and receipt.satisfies(
            AssuranceLevel.KERNEL_VERIFIED
        ):
            resolved = ClaimStatus.SATISFIED
        else:
            # Solver/observation-only remains open (cannot mint kernel alone).
            resolved = ClaimStatus.OPEN
    else:
        resolved = (
            status if isinstance(status, ClaimStatus) else ClaimStatus(str(status))
        )

    return CodeClaimRecord(
        property_id=property_id,
        claim_family=claim_family,  # type: ignore[arg-type]
        status=resolved,
        repository_id=receipt.repository_id,
        repository_tree_id=receipt.repository_tree_id,
        obligation_ids=(receipt.obligation_id,),
        scope_ids=tuple(receipt.ast_scope_ids),
        premise_ids=tuple(receipt.premise_ids),
        evidence_ids=evidence_ids,
        evidence_tiers=tiers,
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
        producer_id=str(getattr(receipt, "provider_id", "") or ""),
        toolchain_id=receipt.toolchain_id,
        policy_id=receipt.policy_id,
        catalog_version=catalog_version,
        invalidation=InvalidationSelector(
            repository_tree_ids=(receipt.repository_tree_id,),
            scope_ids=tuple(receipt.ast_scope_ids),
            premise_ids=tuple(receipt.premise_ids),
            policy_ids=(receipt.policy_id,) if receipt.policy_id else (),
            toolchain_ids=(receipt.toolchain_id,) if receipt.toolchain_id else (),
            catalog_versions=(catalog_version,) if catalog_version else (),
        ),
        metadata={
            "receipt_id": receipt.receipt_id,
            "authoritative_assurance": receipt.authoritative_assurance.value,
            "cache_miss_is_not_refutation": True,
        },
    )


def mark_stale_if_invalidated(
    claim: CodeClaimRecord,
    *,
    repository_tree_id: str = "",
    scope_ids: Sequence[str] = (),
    premise_ids: Sequence[str] = (),
    assumption_ids: Sequence[str] = (),
    policy_id: str = "",
    toolchain_id: str = "",
    catalog_version: str = "",
    paths: Sequence[str] = (),
) -> CodeClaimRecord:
    """Return ``stale`` when invalidation selectors fire; else the same claim."""

    if claim.status not in (ClaimStatus.SATISFIED, ClaimStatus.OPEN):
        return claim
    if claim.invalidation.is_triggered_by(
        repository_tree_id=repository_tree_id or claim.repository_tree_id,
        scope_ids=scope_ids or claim.scope_ids,
        premise_ids=premise_ids or claim.premise_ids,
        assumption_ids=assumption_ids or claim.assumption_ids,
        policy_id=policy_id or claim.policy_id,
        toolchain_id=toolchain_id or claim.toolchain_id,
        catalog_version=catalog_version or claim.catalog_version,
        paths=paths,
    ):
        return CodeClaimRecord(
            property_id=claim.property_id,
            claim_family=claim.claim_family,
            status=ClaimStatus.STALE,
            repository_id=claim.repository_id,
            repository_tree_id=claim.repository_tree_id,
            obligation_ids=claim.obligation_ids,
            scope_ids=claim.scope_ids,
            premise_ids=claim.premise_ids,
            assumption_ids=claim.assumption_ids,
            evidence_ids=claim.evidence_ids,
            evidence_tiers=claim.evidence_tiers,
            required_assurance=claim.required_assurance,
            producer_id=claim.producer_id,
            toolchain_id=claim.toolchain_id,
            policy_id=claim.policy_id,
            catalog_version=claim.catalog_version,
            invalidation=claim.invalidation,
            natural_language_allowed=claim.natural_language_allowed,
            statement=claim.statement,
            metadata={**dict(claim.metadata), "stale_reason": "invalidation_selector"},
        )
    return claim


def cache_miss_status() -> ClaimStatus:
    """Cache miss is never refuted — callers should treat as open/not_measured."""

    return ClaimStatus.OPEN


__all__ = [
    "CODE_CLAIM_RECORD_INTERFACE",
    "CODE_CLAIM_RECORD_SCHEMA",
    "CODE_CLAIM_CONTRACT_VERSION",
    "CodeClaimContractError",
    "ClaimFamily",
    "ClaimStatus",
    "EvidenceTier",
    "InvalidationSelector",
    "CodeClaimRecord",
    "open_claim",
    "claim_from_proof_receipt",
    "mark_stale_if_invalidated",
    "cache_miss_status",
    "evidence_tier_for_proof_evidence",
    "evidence_tier_for_implementation",
    "can_mint_kernel_assurance",
]
