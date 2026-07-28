"""Claim-centric codebase proof query API (CBP-040).

Interface: ``CodeProofQuery@1``

Queries project typed :class:`CodeClaimRecord` values (and optional obligation
compilations / proof-cache lookups) into bounded, content-addressed answers:

* ``properties_satisfied`` / ``open`` / ``refuted`` / ``unsupported`` /
  ``not_measured`` / ``stale``
* ``counterexamples``
* ``impact`` (via :class:`CodeImpactIndex` when provided)
* ``proof_delta`` (invalidated claims/obligations between parent and child trees)

Normative rules:

* Graph / GraphRAG projections are **non-authoritative** and cannot mint proof.
* A proof-cache **miss is never a refutation**.
* ``open`` means a supported claim has no current valid evidence at the required
  assurance (including cache miss).
* ``unsupported``, ``not_measured``, ``unknown``, and ``stale`` stay distinct.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from .code_claim_contracts import (
    ClaimStatus,
    CodeClaimRecord,
    cache_miss_status,
)
from .code_evidence_graph import (
    CodeEvidenceGraph,
    CodeImpactIndex,
    CodeImpactResult,
    materialize_code_evidence_graph,
)
from .code_proof_obligations import (
    CodeProofObligationCompilation,
    CompiledCodeProofItem,
    ObligationCompileStatus,
)
from .formal_verification_cache import (
    CacheLookupStatus,
    FormalVerificationCache,
    ProofCacheKey,
    TrustAwareProofCache,
)
from .formal_verification_contracts import (
    AssuranceLevel,
    content_identity,
)


CODE_PROOF_QUERY_INTERFACE: Final = "CodeProofQuery@1"
CODE_PROOF_QUERY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-proof-query@1"
)
CODE_PROOF_QUERY_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-proof-query-result@1"
)
CODE_PROOF_DELTA_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-proof-delta@1"
)
CODE_PROOF_QUERY_VERSION: Final = "1"

# Hard bounds for returned populations.
DEFAULT_MAX_RESULTS: Final = 256
HARD_MAX_RESULTS: Final = 1024


class CodeProofQueryError(ValueError):
    """Query input is malformed or violates fail-closed bounds."""


def _sorted_unique(values: Iterable[Any]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        values = (values,)
    return tuple(sorted({str(v).strip() for v in values if str(v).strip()}))


def _clamp_limit(limit: int | None) -> int:
    if limit is None:
        return DEFAULT_MAX_RESULTS
    try:
        value = int(limit)
    except (TypeError, ValueError) as exc:
        raise CodeProofQueryError("limit must be an integer") from exc
    if value < 1:
        raise CodeProofQueryError("limit must be >= 1")
    return min(value, HARD_MAX_RESULTS)


@dataclass(frozen=True)
class ClaimQueryHit:
    """One claim-centric query hit with provenance handles."""

    property_id: str
    status: ClaimStatus
    claim_id: str
    obligation_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    evidence_tiers: tuple[str, ...] = ()
    repository_tree_id: str = ""
    cache_key_id: str = ""
    receipt_id: str = ""
    reason_codes: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    counterexample: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "property_id": self.property_id,
            "status": self.status.value
            if isinstance(self.status, ClaimStatus)
            else str(self.status),
            "claim_id": self.claim_id,
            "obligation_ids": list(self.obligation_ids),
            "evidence_ids": list(self.evidence_ids),
            "evidence_tiers": list(self.evidence_tiers),
            "repository_tree_id": self.repository_tree_id,
            "cache_key_id": self.cache_key_id,
            "receipt_id": self.receipt_id,
            "reason_codes": list(self.reason_codes),
            "provenance": dict(self.provenance),
        }
        if self.counterexample is not None:
            payload["counterexample"] = dict(self.counterexample)
        return payload


@dataclass(frozen=True)
class CodeProofQueryResult:
    """Bounded, content-addressed query answer."""

    query: str
    hits: tuple[ClaimQueryHit, ...]
    repository_tree_id: str = ""
    truncated: bool = False
    notes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hits", tuple(self.hits))
        object.__setattr__(self, "notes", _sorted_unique(self.notes))
        if not isinstance(self.metadata, Mapping):
            raise CodeProofQueryError("metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def result_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": CODE_PROOF_QUERY_RESULT_SCHEMA,
            "interface": CODE_PROOF_QUERY_INTERFACE,
            "query": self.query,
            "repository_tree_id": self.repository_tree_id,
            "hits": [hit.to_dict() for hit in self.hits],
            "truncated": bool(self.truncated),
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
            "version": CODE_PROOF_QUERY_VERSION,
        }
        if include_id:
            payload["result_id"] = content_identity(
                {k: v for k, v in payload.items() if k != "result_id"}
            )
        return payload


@dataclass(frozen=True)
class ProofDeltaEntry:
    """One invalidated claim/obligation between parent and child trees."""

    property_id: str
    obligation_id: str
    claim_id: str
    reason_codes: tuple[str, ...]
    parent_tree_id: str
    child_tree_id: str
    parent_status: str = ""
    child_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "property_id": self.property_id,
            "obligation_id": self.obligation_id,
            "claim_id": self.claim_id,
            "reason_codes": list(self.reason_codes),
            "parent_tree_id": self.parent_tree_id,
            "child_tree_id": self.child_tree_id,
            "parent_status": self.parent_status,
            "child_status": self.child_status,
        }


@dataclass(frozen=True)
class ProofDeltaResult:
    """Content-addressed delta of invalidated obligations/claims."""

    parent_tree_id: str
    child_tree_id: str
    entries: tuple[ProofDeltaEntry, ...]
    notes: tuple[str, ...] = ()

    @property
    def delta_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": CODE_PROOF_DELTA_SCHEMA,
            "interface": CODE_PROOF_QUERY_INTERFACE,
            "parent_tree_id": self.parent_tree_id,
            "child_tree_id": self.child_tree_id,
            "entries": [entry.to_dict() for entry in self.entries],
            "notes": list(self.notes),
        }
        if include_id:
            payload["delta_id"] = content_identity(
                {k: v for k, v in payload.items() if k != "delta_id"}
            )
        return payload


def _claim_id(claim: CodeClaimRecord) -> str:
    if hasattr(claim, "claim_id"):
        try:
            return str(claim.claim_id)
        except Exception:  # noqa: BLE001
            pass
    return content_identity(claim.to_dict())


def _hit_from_claim(
    claim: CodeClaimRecord,
    *,
    cache_key_id: str = "",
    reason_codes: Sequence[str] = (),
    counterexample: Mapping[str, Any] | None = None,
    extra_provenance: Mapping[str, Any] | None = None,
) -> ClaimQueryHit:
    tiers = tuple(
        str(getattr(tier, "value", tier)) for tier in (claim.evidence_tiers or ())
    )
    provenance = {
        "producer_id": claim.producer_id,
        "toolchain_id": claim.toolchain_id,
        "policy_id": claim.policy_id,
        "catalog_version": claim.catalog_version,
        "required_assurance": (
            claim.required_assurance.value
            if isinstance(claim.required_assurance, AssuranceLevel)
            else str(claim.required_assurance)
        ),
        "derived_assurance": (
            claim.derived_assurance.value
            if isinstance(getattr(claim, "derived_assurance", None), AssuranceLevel)
            else str(getattr(claim, "derived_assurance", "") or "")
        ),
        "cache_lookup": str(getattr(claim, "cache_lookup", "") or ""),
    }
    if extra_provenance:
        provenance.update(dict(extra_provenance))
    obligation_ids = ()
    if getattr(claim, "obligation_id", ""):
        obligation_ids = (str(claim.obligation_id),)
    elif getattr(claim, "obligation_ids", ()):
        obligation_ids = tuple(claim.obligation_ids)
    return ClaimQueryHit(
        property_id=str(claim.property_id),
        status=claim.status
        if isinstance(claim.status, ClaimStatus)
        else ClaimStatus(str(claim.status)),
        claim_id=_claim_id(claim),
        obligation_ids=obligation_ids,
        evidence_ids=tuple(claim.evidence_ids or ()),
        evidence_tiers=tiers,
        repository_tree_id=str(claim.repository_tree_id or ""),
        cache_key_id=cache_key_id,
        receipt_id=str(getattr(claim, "receipt_id", "") or ""),
        reason_codes=_sorted_unique(reason_codes),
        provenance=provenance,
        counterexample=dict(counterexample) if counterexample else None,
    )


def _hit_from_item(item: CompiledCodeProofItem) -> ClaimQueryHit:
    claim = item.claim
    reasons = list(item.reason_codes or ())
    if item.status is ObligationCompileStatus.UNSUPPORTED:
        reasons.append("compile_unsupported")
    elif item.status is ObligationCompileStatus.NOT_MEASURED:
        reasons.append("compile_not_measured")
    elif item.status is ObligationCompileStatus.OPEN:
        reasons.append("compile_open")
    counterexample = None
    if claim is not None and claim.status is ClaimStatus.REFUTED:
        counterexample = {
            "kind": "claim_refutation",
            "property_id": item.property_id,
            "reason_codes": list(item.reason_codes or ()),
            "invalidation_selectors": [
                dict(selector) if isinstance(selector, Mapping) else {"value": selector}
                for selector in (item.invalidation_selectors or ())
            ],
        }
    if claim is None:
        # Synthetic open/unsupported without claim body.
        status = {
            ObligationCompileStatus.UNSUPPORTED: ClaimStatus.UNSUPPORTED,
            ObligationCompileStatus.NOT_MEASURED: ClaimStatus.NOT_MEASURED,
            ObligationCompileStatus.OPEN: ClaimStatus.OPEN,
        }.get(item.status, ClaimStatus.UNKNOWN)
        return ClaimQueryHit(
            property_id=item.property_id,
            status=status,
            claim_id=content_identity(
                {
                    "property_id": item.property_id,
                    "status": status.value,
                    "cache_key_id": item.cache_key_id,
                }
            ),
            obligation_ids=(
                (item.obligation.obligation_id,) if item.obligation is not None else ()
            ),
            cache_key_id=str(item.cache_key_id or ""),
            reason_codes=_sorted_unique(reasons),
            provenance={"source": "compilation_item"},
        )
    return _hit_from_claim(
        claim,
        cache_key_id=str(item.cache_key_id or ""),
        reason_codes=reasons,
        counterexample=counterexample,
        extra_provenance={"source": "compilation_item", "compile_status": item.status.value},
    )


def _claims_from_inputs(
    claims: Sequence[CodeClaimRecord] | None,
    compilation: CodeProofObligationCompilation | None,
) -> tuple[ClaimQueryHit, ...]:
    hits: list[ClaimQueryHit] = []
    if claims:
        for claim in claims:
            if not isinstance(claim, CodeClaimRecord):
                raise CodeProofQueryError("claims must be CodeClaimRecord instances")
            hits.append(_hit_from_claim(claim))
    if compilation is not None:
        if not isinstance(compilation, CodeProofObligationCompilation):
            raise CodeProofQueryError(
                "compilation must be a CodeProofObligationCompilation"
            )
        for item in compilation.items:
            hits.append(_hit_from_item(item))
    # Prefer claim_id uniqueness; last write wins for same claim_id.
    by_id: dict[str, ClaimQueryHit] = {}
    for hit in hits:
        by_id[hit.claim_id] = hit
    return tuple(
        sorted(by_id.values(), key=lambda item: (item.property_id, item.claim_id))
    )


def _filter_status(
    hits: Sequence[ClaimQueryHit],
    status: ClaimStatus,
    *,
    limit: int | None,
    query: str,
    repository_tree_id: str = "",
    notes: Sequence[str] = (),
) -> CodeProofQueryResult:
    limit_n = _clamp_limit(limit)
    matched = [hit for hit in hits if hit.status is status]
    truncated = len(matched) > limit_n
    return CodeProofQueryResult(
        query=query,
        hits=tuple(matched[:limit_n]),
        repository_tree_id=repository_tree_id,
        truncated=truncated,
        notes=tuple(notes),
        metadata={"status_filter": status.value, "population": len(matched)},
    )


@dataclass
class CodeProofQuery:
    """In-memory claim/compilation query surface.

    Construct from claims and/or a CBP-030 compilation.  Optional proof cache
    is used only to annotate open claims with miss/hit provenance — never to
    invent refutations.
    """

    claims: tuple[CodeClaimRecord, ...] = ()
    compilation: CodeProofObligationCompilation | None = None
    cache: FormalVerificationCache | TrustAwareProofCache | None = None
    impact_index: CodeImpactIndex | None = None
    graph: CodeEvidenceGraph | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "claims", tuple(self.claims or ()))
        object.__setattr__(
            self,
            "_hits",
            _claims_from_inputs(self.claims, self.compilation),
        )

    @property
    def hits(self) -> tuple[ClaimQueryHit, ...]:
        return getattr(self, "_hits")

    @property
    def repository_tree_id(self) -> str:
        if self.compilation is not None:
            return str(self.compilation.repository_tree_id or "")
        for hit in self.hits:
            if hit.repository_tree_id:
                return hit.repository_tree_id
        return ""

    def _annotate_cache(self, hit: ClaimQueryHit) -> ClaimQueryHit:
        """Attach cache miss/hit notes without changing claim status semantics."""

        if self.cache is None or not hit.cache_key_id:
            return hit
        # Without the full ProofCacheKey object we only record that a cache id
        # exists; callers may pass pre-annotated claims via metadata.
        return hit

    def properties_satisfied(
        self, *, limit: int | None = None
    ) -> CodeProofQueryResult:
        return _filter_status(
            self.hits,
            ClaimStatus.SATISFIED,
            limit=limit,
            query="properties_satisfied",
            repository_tree_id=self.repository_tree_id,
        )

    def properties_open(self, *, limit: int | None = None) -> CodeProofQueryResult:
        # Cache miss is open, not refuted.
        notes = ()
        if self.cache is not None:
            notes = ("cache_miss_is_not_refutation",)
        return _filter_status(
            self.hits,
            ClaimStatus.OPEN,
            limit=limit,
            query="properties_open",
            repository_tree_id=self.repository_tree_id,
            notes=notes,
        )

    def properties_refuted(
        self, *, limit: int | None = None
    ) -> CodeProofQueryResult:
        return _filter_status(
            self.hits,
            ClaimStatus.REFUTED,
            limit=limit,
            query="properties_refuted",
            repository_tree_id=self.repository_tree_id,
            notes=("cache_miss_is_not_refutation",),
        )

    def properties_unsupported(
        self, *, limit: int | None = None
    ) -> CodeProofQueryResult:
        return _filter_status(
            self.hits,
            ClaimStatus.UNSUPPORTED,
            limit=limit,
            query="properties_unsupported",
            repository_tree_id=self.repository_tree_id,
        )

    def properties_not_measured(
        self, *, limit: int | None = None
    ) -> CodeProofQueryResult:
        return _filter_status(
            self.hits,
            ClaimStatus.NOT_MEASURED,
            limit=limit,
            query="properties_not_measured",
            repository_tree_id=self.repository_tree_id,
        )

    def properties_stale(self, *, limit: int | None = None) -> CodeProofQueryResult:
        return _filter_status(
            self.hits,
            ClaimStatus.STALE,
            limit=limit,
            query="properties_stale",
            repository_tree_id=self.repository_tree_id,
        )

    def counterexamples(self, *, limit: int | None = None) -> CodeProofQueryResult:
        limit_n = _clamp_limit(limit)
        matched = [
            hit
            for hit in self.hits
            if hit.status is ClaimStatus.REFUTED or hit.counterexample is not None
        ]
        # Ensure refuted hits always carry a counterexample sketch.
        normalized: list[ClaimQueryHit] = []
        for hit in matched:
            if hit.counterexample is None:
                hit = ClaimQueryHit(
                    property_id=hit.property_id,
                    status=hit.status,
                    claim_id=hit.claim_id,
                    obligation_ids=hit.obligation_ids,
                    evidence_ids=hit.evidence_ids,
                    evidence_tiers=hit.evidence_tiers,
                    repository_tree_id=hit.repository_tree_id,
                    cache_key_id=hit.cache_key_id,
                    receipt_id=hit.receipt_id,
                    reason_codes=hit.reason_codes + ("refuted_without_model",),
                    provenance=hit.provenance,
                    counterexample={
                        "kind": "refuted_claim",
                        "property_id": hit.property_id,
                        "claim_id": hit.claim_id,
                    },
                )
            normalized.append(hit)
        truncated = len(normalized) > limit_n
        return CodeProofQueryResult(
            query="counterexamples",
            hits=tuple(normalized[:limit_n]),
            repository_tree_id=self.repository_tree_id,
            truncated=truncated,
            notes=("counterexamples_from_refuted_claims",),
            metadata={"population": len(normalized)},
        )

    def impact(
        self,
        *,
        changed_paths: Sequence[str] = (),
        changed_symbols: Sequence[str] = (),
    ) -> CodeImpactResult | CodeProofQueryResult:
        """Return path/symbol impact via CodeImpactIndex when available."""

        if self.impact_index is None:
            # Fallback: property hits whose provenance paths intersect.
            paths = set(_sorted_unique(changed_paths))
            hits = [
                hit
                for hit in self.hits
                if paths
                and any(
                    path in paths
                    for path in (
                        hit.provenance.get("paths")
                        if isinstance(hit.provenance.get("paths"), (list, tuple))
                        else ()
                    )
                )
            ]
            return CodeProofQueryResult(
                query="impact",
                hits=tuple(hits),
                repository_tree_id=self.repository_tree_id,
                notes=("impact_index_absent_property_path_fallback",),
                metadata={"changed_paths": list(paths)},
            )
        return self.impact_index.impact(
            changed_paths=changed_paths,
            changed_symbols=changed_symbols,
        )

    def proof_delta(
        self,
        parent: "CodeProofQuery | CodeProofObligationCompilation | Sequence[CodeClaimRecord]",
        child: "CodeProofQuery | CodeProofObligationCompilation | Sequence[CodeClaimRecord] | None" = None,
    ) -> ProofDeltaResult:
        """List claims/obligations invalidated between parent and child trees."""

        parent_q = _as_query(parent)
        child_q = _as_query(child if child is not None else self)
        parent_tree = parent_q.repository_tree_id
        child_tree = child_q.repository_tree_id
        parent_by_prop = {hit.property_id: hit for hit in parent_q.hits}
        child_by_prop = {hit.property_id: hit for hit in child_q.hits}
        entries: list[ProofDeltaEntry] = []

        for property_id, parent_hit in sorted(parent_by_prop.items()):
            child_hit = child_by_prop.get(property_id)
            reasons: list[str] = []
            if child_hit is None:
                reasons.append("missing_on_child_tree")
            else:
                if parent_hit.claim_id != child_hit.claim_id:
                    reasons.append("claim_identity_changed")
                if parent_hit.status is ClaimStatus.SATISFIED and child_hit.status in (
                    ClaimStatus.OPEN,
                    ClaimStatus.STALE,
                    ClaimStatus.REFUTED,
                    ClaimStatus.UNSUPPORTED,
                ):
                    reasons.append("satisfied_no_longer_holds")
                if child_hit.status is ClaimStatus.STALE:
                    reasons.append("child_stale")
                if (
                    parent_hit.repository_tree_id
                    and child_hit.repository_tree_id
                    and parent_hit.repository_tree_id != child_hit.repository_tree_id
                ):
                    reasons.append("repository_tree_changed")
                if parent_hit.cache_key_id and child_hit.cache_key_id:
                    if parent_hit.cache_key_id != child_hit.cache_key_id:
                        reasons.append("cache_key_changed")
            if not reasons:
                continue
            obligation_id = (
                parent_hit.obligation_ids[0] if parent_hit.obligation_ids else ""
            )
            if child_hit and child_hit.obligation_ids:
                obligation_id = child_hit.obligation_ids[0]
            entries.append(
                ProofDeltaEntry(
                    property_id=property_id,
                    obligation_id=obligation_id,
                    claim_id=(child_hit.claim_id if child_hit else parent_hit.claim_id),
                    reason_codes=_sorted_unique(reasons),
                    parent_tree_id=parent_tree,
                    child_tree_id=child_tree,
                    parent_status=parent_hit.status.value,
                    child_status=child_hit.status.value if child_hit else "",
                )
            )

        # New refutations on child that were not present on parent.
        for property_id, child_hit in sorted(child_by_prop.items()):
            if property_id in parent_by_prop:
                continue
            if child_hit.status in (
                ClaimStatus.REFUTED,
                ClaimStatus.OPEN,
                ClaimStatus.STALE,
            ):
                entries.append(
                    ProofDeltaEntry(
                        property_id=property_id,
                        obligation_id=(
                            child_hit.obligation_ids[0]
                            if child_hit.obligation_ids
                            else ""
                        ),
                        claim_id=child_hit.claim_id,
                        reason_codes=("introduced_on_child_tree",),
                        parent_tree_id=parent_tree,
                        child_tree_id=child_tree,
                        parent_status="",
                        child_status=child_hit.status.value,
                    )
                )

        entries_sorted = tuple(
            sorted(entries, key=lambda item: (item.property_id, item.obligation_id))
        )
        return ProofDeltaResult(
            parent_tree_id=parent_tree,
            child_tree_id=child_tree,
            entries=entries_sorted,
            notes=(
                "proof_delta_lists_only_invalidated_or_introduced_claims",
                "cache_miss_is_not_refutation",
            ),
        )

    def project_evidence_graph(
        self,
        *,
        tasks: Sequence[Any] = (),
        repository_trees: Sequence[Any] = (),
        obligations: Sequence[Any] = (),
        proof_receipts: Sequence[Any] = (),
    ) -> CodeEvidenceGraph:
        """Project claims into a non-authoritative evidence graph enrichment."""

        claim_records = []
        for hit in self.hits:
            claim_records.append(
                {
                    "node_kind": "claim",
                    "property_id": hit.property_id,
                    "claim_id": hit.claim_id,
                    "status": hit.status.value,
                    "obligation_ids": list(hit.obligation_ids),
                    "evidence_ids": list(hit.evidence_ids),
                    "provenance": "enrichment",
                    "repository_tree_id": hit.repository_tree_id
                    or self.repository_tree_id,
                }
            )
        # Enrichments cannot mint proof nodes — use related_to style enrichment.
        trees = repository_trees or (
            ({"repository_tree_id": self.repository_tree_id},)
            if self.repository_tree_id
            else ()
        )
        return materialize_code_evidence_graph(
            tasks=tasks,
            repository_trees=trees,
            obligations=obligations,
            proof_receipts=proof_receipts,
            enrichments=claim_records,
        )


def _as_query(
    value: CodeProofQuery
    | CodeProofObligationCompilation
    | Sequence[CodeClaimRecord]
    | None,
) -> CodeProofQuery:
    if value is None:
        return CodeProofQuery()
    if isinstance(value, CodeProofQuery):
        return value
    if isinstance(value, CodeProofObligationCompilation):
        return CodeProofQuery(compilation=value)
    if isinstance(value, (list, tuple)):
        return CodeProofQuery(claims=tuple(value))
    raise CodeProofQueryError(
        "expected CodeProofQuery, compilation, or sequence of claims"
    )


def build_code_proof_query(
    *,
    claims: Sequence[CodeClaimRecord] = (),
    compilation: CodeProofObligationCompilation | None = None,
    cache: FormalVerificationCache | TrustAwareProofCache | None = None,
    impact_index: CodeImpactIndex | None = None,
    graph: CodeEvidenceGraph | None = None,
) -> CodeProofQuery:
    """Factory for :class:`CodeProofQuery`."""

    return CodeProofQuery(
        claims=tuple(claims),
        compilation=compilation,
        cache=cache,
        impact_index=impact_index,
        graph=graph,
    )


# Explicit re-export of cache-miss doctrine for callers/tests.
CACHE_MISS_STATUS = cache_miss_status()


__all__ = [
    "CODE_PROOF_QUERY_INTERFACE",
    "CODE_PROOF_QUERY_SCHEMA",
    "CODE_PROOF_QUERY_RESULT_SCHEMA",
    "CODE_PROOF_DELTA_SCHEMA",
    "CODE_PROOF_QUERY_VERSION",
    "CACHE_MISS_STATUS",
    "CodeProofQueryError",
    "ClaimQueryHit",
    "CodeProofQueryResult",
    "ProofDeltaEntry",
    "ProofDeltaResult",
    "CodeProofQuery",
    "build_code_proof_query",
]
