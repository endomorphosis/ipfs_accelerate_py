"""Cache-aware re-proof and invalidation (CBP-050).

Builds on:

* :func:`prove_code_obligation_with_cache` (CBP-015) — lookup-before-provider,
  single-flight, re-derive on hit
* :func:`compile_code_proof_obligations` (CBP-030)
* :func:`CodeProofQuery.proof_delta` (CBP-040) for tree-to-tree claim deltas

Warm re-proof serves unchanged obligations from the trust-aware proof cache.
Any binding drift (tree, premises/assumptions, catalog, toolchain, policy,
required assurance, property id) forces a miss / stale disposition and a
re-solve.  Wrong-tree receipts never satisfy a foreign key.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .code_proof_obligations import (
    CachedProveResult,
    CodeProofObligationCompilation,
    CompiledCodeProofItem,
    ObligationCompileStatus,
    ProofCacheMetrics,
    build_code_proof_cache_key,
    prove_code_obligation_with_cache,
)
from .code_proof_query import CodeProofQuery, ProofDeltaResult, build_code_proof_query
from .formal_verification_cache import (
    FormalVerificationCache,
    ProofCacheKey,
    TrustAwareProofCache,
)
from .formal_verification_contracts import (
    AssuranceLevel,
    CodeProofObligation,
    ProofReceipt,
    content_identity,
)


CODE_PROOF_REPROOF_INTERFACE: Final = "CodeProofReproof@1"
CODE_PROOF_REPROOF_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-proof-reproof-report@1"
)
CODE_PROOF_REPROOF_VERSION: Final = "1"


class CodeProofReproofError(ValueError):
    """Re-proof input is malformed or unsafe."""


class ReproofDisposition(str, Enum):
    """Per-obligation re-proof outcome."""

    CACHE_HIT = "cache_hit"
    RE_SOLVED = "re_solved"
    SKIPPED_UNSUPPORTED = "skipped_unsupported"
    SKIPPED_NOT_MEASURED = "skipped_not_measured"
    SKIPPED_NO_OBLIGATION = "skipped_no_obligation"
    REJECTED = "rejected"
    FAILED = "failed"


# Stable invalidation reason vocabulary (also used as provenance edge labels).
class InvalidationReason(str, Enum):
    REPOSITORY_TREE_CHANGED = "repository_tree_changed"
    PREMISE_DIGEST_CHANGED = "premise_digest_changed"
    ASSUMPTION_DIGEST_CHANGED = "assumption_digest_changed"
    PROPERTY_CHANGED = "property_changed"
    CATALOG_CHANGED = "catalog_changed"
    TOOLCHAIN_CHANGED = "toolchain_changed"
    POLICY_CHANGED = "policy_changed"
    REQUIRED_ASSURANCE_CHANGED = "required_assurance_changed"
    CACHE_KEY_CHANGED = "cache_key_changed"
    AST_SCOPE_CHANGED = "ast_scope_changed"
    DEPENDENCY_EDGE_CHANGED = "dependency_edge_changed"
    PATH_CHANGED = "path_changed"
    FOREIGN_TREE_HIT_REJECTED = "foreign_tree_hit_rejected"
    AUTHORITATIVE_CACHE_HIT = "authoritative_cache_hit"
    PROVIDER_INVOKED = "provider_invoked"
    COLD_MISS = "cold_miss"


@dataclass(frozen=True)
class BindingFingerprint:
    """Compact binding surface used to explain invalidations."""

    repository_tree_id: str
    property_id: str
    cache_key_id: str
    premise_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]
    catalog_version: str
    toolchain_id: str
    policy_id: str
    required_assurance: str
    ast_scope_ids: tuple[str, ...] = ()
    residual_ref_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_tree_id": self.repository_tree_id,
            "property_id": self.property_id,
            "cache_key_id": self.cache_key_id,
            "premise_ids": list(self.premise_ids),
            "assumption_ids": list(self.assumption_ids),
            "catalog_version": self.catalog_version,
            "toolchain_id": self.toolchain_id,
            "policy_id": self.policy_id,
            "required_assurance": self.required_assurance,
            "ast_scope_ids": list(self.ast_scope_ids),
            "residual_ref_ids": list(self.residual_ref_ids),
        }


def binding_fingerprint_for_item(
    item: CompiledCodeProofItem,
    *,
    repository_tree_id: str = "",
    toolchain_id: str = "",
    policy_id: str = "",
) -> BindingFingerprint:
    obligation = item.obligation
    scopes = tuple(obligation.ast_scope_ids) if obligation is not None else ()
    assurance = item.required_assurance
    if isinstance(assurance, AssuranceLevel):
        assurance_s = assurance.value
    else:
        assurance_s = str(assurance or "")
    return BindingFingerprint(
        repository_tree_id=str(
            repository_tree_id
            or (obligation.repository_tree_id if obligation else "")
            or ""
        ),
        property_id=str(item.property_id or ""),
        cache_key_id=str(item.cache_key_id or ""),
        premise_ids=tuple(item.premise_ids or ()),
        assumption_ids=tuple(item.assumption_ids or ()),
        catalog_version=str(item.catalog_version or ""),
        toolchain_id=str(toolchain_id or ""),
        policy_id=str(policy_id or ""),
        required_assurance=assurance_s,
        ast_scope_ids=scopes,
        residual_ref_ids=tuple(item.residual_ref_ids or ()),
    )


def invalidation_reasons(
    previous: BindingFingerprint | None,
    current: BindingFingerprint,
    *,
    changed_paths: Sequence[str] = (),
    dependency_edge_changed: bool = False,
) -> tuple[str, ...]:
    """Return machine-readable reasons why a prior proof cannot be reused."""

    if previous is None:
        return (InvalidationReason.COLD_MISS.value,)
    reasons: list[str] = []
    if previous.repository_tree_id != current.repository_tree_id:
        reasons.append(InvalidationReason.REPOSITORY_TREE_CHANGED.value)
    if previous.premise_ids != current.premise_ids:
        reasons.append(InvalidationReason.PREMISE_DIGEST_CHANGED.value)
    if previous.assumption_ids != current.assumption_ids:
        reasons.append(InvalidationReason.ASSUMPTION_DIGEST_CHANGED.value)
    if previous.property_id != current.property_id:
        reasons.append(InvalidationReason.PROPERTY_CHANGED.value)
    if previous.catalog_version != current.catalog_version:
        reasons.append(InvalidationReason.CATALOG_CHANGED.value)
    if previous.toolchain_id != current.toolchain_id:
        reasons.append(InvalidationReason.TOOLCHAIN_CHANGED.value)
    if previous.policy_id != current.policy_id:
        reasons.append(InvalidationReason.POLICY_CHANGED.value)
    if previous.required_assurance != current.required_assurance:
        reasons.append(InvalidationReason.REQUIRED_ASSURANCE_CHANGED.value)
    if previous.cache_key_id != current.cache_key_id:
        reasons.append(InvalidationReason.CACHE_KEY_CHANGED.value)
    if previous.ast_scope_ids != current.ast_scope_ids:
        reasons.append(InvalidationReason.AST_SCOPE_CHANGED.value)
    if dependency_edge_changed:
        reasons.append(InvalidationReason.DEPENDENCY_EDGE_CHANGED.value)
    if changed_paths:
        reasons.append(InvalidationReason.PATH_CHANGED.value)
    return tuple(sorted(set(reasons)))


@dataclass(frozen=True)
class ReproofItemResult:
    """Outcome for one compiled obligation under cache-aware re-proof."""

    property_id: str
    disposition: ReproofDisposition
    from_cache: bool
    cache_key_id: str
    obligation_id: str = ""
    reason_codes: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    receipt_id: str = ""
    authoritative_assurance: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "property_id": self.property_id,
            "disposition": self.disposition.value,
            "from_cache": self.from_cache,
            "cache_key_id": self.cache_key_id,
            "obligation_id": self.obligation_id,
            "reason_codes": list(self.reason_codes),
            "provenance": dict(self.provenance),
            "receipt_id": self.receipt_id,
            "authoritative_assurance": self.authoritative_assurance,
        }


@dataclass(frozen=True)
class ReproofReport:
    """Content-addressed report for a re-proof pass."""

    repository_tree_id: str
    results: tuple[ReproofItemResult, ...]
    metrics: Mapping[str, Any] = field(default_factory=dict)
    proof_delta: Mapping[str, Any] | None = None
    notes: tuple[str, ...] = ()

    @property
    def report_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    @property
    def cache_hits(self) -> int:
        return sum(
            1
            for item in self.results
            if item.disposition is ReproofDisposition.CACHE_HIT
        )

    @property
    def re_solved(self) -> int:
        return sum(
            1
            for item in self.results
            if item.disposition is ReproofDisposition.RE_SOLVED
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        # Canonical contracts reject floats; coerce metric rates to millis.
        metrics: dict[str, Any] = {}
        for key, value in dict(self.metrics).items():
            if isinstance(value, float):
                metrics[key] = int(round(value * 1_000_000))
            else:
                metrics[key] = value
        payload = {
            "schema": CODE_PROOF_REPROOF_REPORT_SCHEMA,
            "interface": CODE_PROOF_REPROOF_INTERFACE,
            "version": CODE_PROOF_REPROOF_VERSION,
            "repository_tree_id": self.repository_tree_id,
            "results": [item.to_dict() for item in self.results],
            "metrics": metrics,
            "proof_delta": dict(self.proof_delta) if self.proof_delta else None,
            "notes": list(self.notes),
            "cache_hits": self.cache_hits,
            "re_solved": self.re_solved,
        }
        if include_id:
            payload["report_id"] = content_identity(
                {k: v for k, v in payload.items() if k != "report_id"}
            )
        return payload


def _key_for_item(
    item: CompiledCodeProofItem,
    compilation: CodeProofObligationCompilation,
    *,
    translator_id: str,
    solver_id: str,
    kernel_id: str,
    theorem_registry_id: str,
    resource_budget: Any,
) -> ProofCacheKey:
    if item.obligation is None:
        raise CodeProofReproofError(
            f"item {item.property_id!r} has no obligation to re-prove"
        )
    return build_code_proof_cache_key(
        item.obligation,
        translator_id=translator_id,
        solver_id=solver_id,
        kernel_id=kernel_id,
        toolchain_id=compilation.toolchain_id,
        theorem_registry_id=theorem_registry_id,
        policy_id=compilation.policy_id,
        resource_budget=resource_budget,
        candidate_tree=compilation.repository_tree_id,
        property_id=item.property_id,
        catalog_version=item.catalog_version or compilation.catalog_version,
        catalog_id=compilation.catalog_id,
        assumption_ids=item.assumption_ids,
        residual_ref_ids=item.residual_ref_ids,
    )


def _previous_by_property(
    previous: CodeProofObligationCompilation | None,
) -> dict[str, CompiledCodeProofItem]:
    if previous is None:
        return {}
    return {item.property_id: item for item in previous.items}


def reprove_code_proof_compilation(
    cache: FormalVerificationCache | TrustAwareProofCache,
    compilation: CodeProofObligationCompilation,
    *,
    prove: Callable[[CompiledCodeProofItem, ProofCacheKey], ProofReceipt],
    previous: CodeProofObligationCompilation | None = None,
    metrics: ProofCacheMetrics | None = None,
    translator_id: str = "translator:default",
    solver_id: str = "solver:default",
    kernel_id: str = "kernel:default",
    theorem_registry_id: str = "registry:default",
    resource_budget: Any = None,
    changed_paths: Sequence[str] = (),
    dependency_edge_changed: bool = False,
    prefer_cache_before_provider: bool = True,
) -> ReproofReport:
    """Re-prove each open compiled obligation with cache-first semantics.

    ``prove(item, key)`` is invoked only on cache miss/reject paths (or when
    ``prefer_cache_before_provider`` is false).  Single-flight is handled inside
    :func:`prove_code_obligation_with_cache`.
    """

    if not isinstance(compilation, CodeProofObligationCompilation):
        raise CodeProofReproofError(
            "compilation must be a CodeProofObligationCompilation"
        )
    if not callable(prove):
        raise CodeProofReproofError("prove must be callable")

    stats = metrics if metrics is not None else ProofCacheMetrics()
    prev_map = _previous_by_property(previous)
    results: list[ReproofItemResult] = []

    for item in compilation.items:
        if item.status is ObligationCompileStatus.UNSUPPORTED:
            results.append(
                ReproofItemResult(
                    property_id=item.property_id,
                    disposition=ReproofDisposition.SKIPPED_UNSUPPORTED,
                    from_cache=False,
                    cache_key_id=str(item.cache_key_id or ""),
                    reason_codes=("compile_unsupported",),
                    provenance={"edge": "skip", "status": item.status.value},
                )
            )
            continue
        if item.status is ObligationCompileStatus.NOT_MEASURED:
            results.append(
                ReproofItemResult(
                    property_id=item.property_id,
                    disposition=ReproofDisposition.SKIPPED_NOT_MEASURED,
                    from_cache=False,
                    cache_key_id=str(item.cache_key_id or ""),
                    reason_codes=("compile_not_measured",),
                    provenance={"edge": "skip", "status": item.status.value},
                )
            )
            continue
        if item.obligation is None:
            results.append(
                ReproofItemResult(
                    property_id=item.property_id,
                    disposition=ReproofDisposition.SKIPPED_NO_OBLIGATION,
                    from_cache=False,
                    cache_key_id=str(item.cache_key_id or ""),
                    reason_codes=("missing_obligation",),
                )
            )
            continue

        current_fp = binding_fingerprint_for_item(
            item,
            repository_tree_id=compilation.repository_tree_id,
            toolchain_id=compilation.toolchain_id,
            policy_id=compilation.policy_id,
        )
        prev_item = prev_map.get(item.property_id)
        prev_fp = (
            binding_fingerprint_for_item(
                prev_item,
                repository_tree_id=previous.repository_tree_id if previous else "",
                toolchain_id=previous.toolchain_id if previous else "",
                policy_id=previous.policy_id if previous else "",
            )
            if prev_item is not None
            else None
        )
        reasons = list(
            invalidation_reasons(
                prev_fp,
                current_fp,
                changed_paths=changed_paths,
                dependency_edge_changed=dependency_edge_changed,
            )
        )

        key = _key_for_item(
            item,
            compilation,
            translator_id=translator_id,
            solver_id=solver_id,
            kernel_id=kernel_id,
            theorem_registry_id=theorem_registry_id,
            resource_budget=resource_budget,
        )
        required = item.required_assurance
        if not isinstance(required, AssuranceLevel):
            required = AssuranceLevel(
                str(required or AssuranceLevel.KERNEL_VERIFIED.value)
            )

        provider_calls = {"n": 0}

        def _prove() -> ProofReceipt:
            provider_calls["n"] += 1
            return prove(item, key)

        try:
            outcome: CachedProveResult = prove_code_obligation_with_cache(
                cache,
                key,
                prove=_prove,
                required_assurance=required,
                metrics=stats,
                prefer_cache_before_provider=prefer_cache_before_provider,
            )
        except Exception as exc:  # noqa: BLE001 - surface as failed item
            results.append(
                ReproofItemResult(
                    property_id=item.property_id,
                    disposition=ReproofDisposition.FAILED,
                    from_cache=False,
                    cache_key_id=key.key_id,
                    obligation_id=item.obligation.obligation_id,
                    reason_codes=("reproof_exception", type(exc).__name__),
                    provenance={"error": str(exc)[:500]},
                )
            )
            continue

        # Wrong-tree foreign hit protection: receipt tree must match key tree.
        if outcome.receipt is not None:
            receipt_tree = str(outcome.receipt.repository_tree_id or "")
            key_tree = str(key.candidate_tree or "")
            if receipt_tree and key_tree and receipt_tree != key_tree:
                results.append(
                    ReproofItemResult(
                        property_id=item.property_id,
                        disposition=ReproofDisposition.REJECTED,
                        from_cache=outcome.from_cache,
                        cache_key_id=key.key_id,
                        obligation_id=item.obligation.obligation_id,
                        reason_codes=(
                            InvalidationReason.FOREIGN_TREE_HIT_REJECTED.value,
                            "stale_tree",
                        ),
                        provenance={
                            "receipt_tree": receipt_tree,
                            "key_tree": key_tree,
                            "edge": "reject_foreign_tree",
                        },
                        receipt_id=str(outcome.receipt.receipt_id),
                    )
                )
                continue

        if outcome.status == "hit" and outcome.from_cache:
            disposition = ReproofDisposition.CACHE_HIT
            reasons = [
                InvalidationReason.AUTHORITATIVE_CACHE_HIT.value,
                *outcome.reason_codes,
            ]
            # Provider must not have been called on authoritative hit.
            if provider_calls["n"] != 0:
                disposition = ReproofDisposition.FAILED
                reasons = ("provider_called_on_cache_hit",)
        elif outcome.status in {"proved", "hit"} and not outcome.from_cache:
            disposition = ReproofDisposition.RE_SOLVED
            reasons = [
                InvalidationReason.PROVIDER_INVOKED.value,
                *reasons,
                *outcome.reason_codes,
            ]
        elif outcome.status == "rejected":
            disposition = ReproofDisposition.REJECTED
            reasons = list(outcome.reason_codes) or reasons or ("rejected",)
        else:
            disposition = ReproofDisposition.RE_SOLVED
            reasons = [
                InvalidationReason.PROVIDER_INVOKED.value,
                *reasons,
                *outcome.reason_codes,
            ]

        results.append(
            ReproofItemResult(
                property_id=item.property_id,
                disposition=disposition,
                from_cache=bool(outcome.from_cache),
                cache_key_id=key.key_id,
                obligation_id=item.obligation.obligation_id,
                reason_codes=tuple(sorted(set(str(r) for r in reasons if r))),
                provenance={
                    "edge": "cache_hit" if outcome.from_cache else "re_solve",
                    "binding": current_fp.to_dict(),
                    "previous_binding": prev_fp.to_dict() if prev_fp else None,
                    "provider_calls": provider_calls["n"],
                    "outcome_status": outcome.status,
                },
                receipt_id=(
                    str(outcome.receipt.receipt_id) if outcome.receipt is not None else ""
                ),
                authoritative_assurance=(
                    outcome.receipt.authoritative_assurance.value
                    if outcome.receipt is not None
                    else ""
                ),
            )
        )

    delta_payload = None
    notes: list[str] = [
        "warm_path_uses_trust_aware_cache_with_rederive",
        "provider_not_called_on_authoritative_hit",
        "wrong_tree_binding_never_accepted",
    ]
    if previous is not None:
        delta: ProofDeltaResult = build_code_proof_query(
            compilation=compilation
        ).proof_delta(previous)
        delta_payload = delta.to_dict()
        notes.append("proof_delta_attached")

    return ReproofReport(
        repository_tree_id=compilation.repository_tree_id,
        results=tuple(results),
        metrics=stats.snapshot(),
        proof_delta=delta_payload,
        notes=tuple(notes),
    )


# Convenience re-export for query-side deltas during re-proof planning.
def plan_reproof_from_delta(
    parent: CodeProofObligationCompilation,
    child: CodeProofObligationCompilation,
) -> ProofDeltaResult:
    """Return claim/obligation delta that should drive re-proof selection."""

    return build_code_proof_query(compilation=child).proof_delta(parent)


__all__ = [
    "CODE_PROOF_REPROOF_INTERFACE",
    "CODE_PROOF_REPROOF_REPORT_SCHEMA",
    "CODE_PROOF_REPROOF_VERSION",
    "CodeProofReproofError",
    "ReproofDisposition",
    "InvalidationReason",
    "BindingFingerprint",
    "binding_fingerprint_for_item",
    "invalidation_reasons",
    "ReproofItemResult",
    "ReproofReport",
    "reprove_code_proof_compilation",
    "plan_reproof_from_delta",
]
