"""Assurance-aware ContextPack compilation over admitted capsules and raw source.

Projects semantic inputs into existing ``ContextReference`` tiers and accounts
tokens with ``ContextCompiler``. Exact target/edit/test spans are never
compressed. Substitutable capsules may replace unchanged dependency code only
when admission allows it, with visible caveats. Budget failures recommend
escalation instead of silent truncation. Capsule facts remain datasets-owned.

Accelerate admits freshness, reuse, and executor context. It verifies Datasets
semantic identity and Kit bytes/root without reminting Datasets CIDs or treating
durable storage as semantic proof.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    CalibratedTokenEstimator,
    ContextCompiler,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
    ContextBudget,
    ContextReference,
    ContextTier,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.capsules import (
    ADMISSION_CONSERVATIVE,
    ADMISSION_RAW,
    CapsuleAdmission,
    capsule_may_substitute,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    ContextPack,
    HarnessError,
    ModelRoute,
    _text,
    _unique_sorted_cids,
    _unique_sorted_texts,
    validate_opaque_cid,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload

# ---------------------------------------------------------------------------
# Interfaces / estimator identity
# ---------------------------------------------------------------------------

CONTEXT_PACK_INTERFACE = "ContextPack@1"
CONTEXT_PACK_RESULT_SCHEMA = "ipfs-accelerate.context-pack-result@1"
CONTEXT_COVERAGE_POLICY_SCHEMA = "ipfs-accelerate.context-coverage-policy@1"
CONTEXT_PACK_FRESHNESS_INTERFACE = "ContextPackFreshnessAdmission@1"
CONTEXT_PACK_FRESHNESS_SCHEMA = "ipfs-accelerate.context-pack-freshness-admission@1"
TOKEN_ESTIMATOR_VERSION = "context-compiler-calibrated_utf8@1"
DATASETS_CONTEXT_PACK_AUTHORITY = "ipfs_datasets_py.proof_context.context_pack"
KIT_CONTEXT_PACK_STORE_AUTHORITY = "ipfs_kit_py.proof_context.state_store"

EXACT_FRESHNESS_FIELDS: tuple[str, ...] = (
    "tree",
    "objective",
    "policy",
    "interface",
    "toolchain",
    "environment",
)
REQUIRED_SOURCE_KINDS: tuple[str, ...] = (
    "target_source",
    "surrounding_source",
    "test_source",
)

_REQUIRED_SOURCE_KINDS = frozenset(REQUIRED_SOURCE_KINDS)
_NEVER_COMPRESS = frozenset(
    {
        "target_source",
        "surrounding_source",
        "test_source",
    }
)


def _dedupe_sorted(values: Sequence[Any], name: str) -> tuple[str, ...]:
    seen: list[str] = []
    for item in values:
        text = _text(item, name)
        if text not in seen:
            seen.append(text)
    return _unique_sorted_texts(seen, name)


class ContextPackError(HarnessError):
    """Closed context-pack or coverage-policy violation."""

    reason_code = "invalid"


class StaleIdentityError(ContextPackError):
    """Exact tree/objective/policy/interface/toolchain/environment mismatch."""

    reason_code = "stale_identity"

    def __init__(
        self,
        message: str,
        *,
        stale_fields: Sequence[str] = (),
        reason_codes: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.stale_fields = _dedupe_sorted(stale_fields, "stale_fields")
        codes = list(reason_codes) or [f"stale:{field}" for field in self.stale_fields]
        self.reason_codes = _dedupe_sorted(codes, "reason_codes")


class ContextPackAuthorityUnavailable(ContextPackError):
    """Installed Datasets or Kit ContextPack authority is missing."""

    reason_code = "unavailable"


# ---------------------------------------------------------------------------
# Coverage policy and token estimate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextCoveragePolicy:
    """Hard coverage constraints for assurance-aware packing.

    Coverage is not a ranking score. Required source categories must appear as
    exact CIDs; never-compress categories may not be replaced by capsules or
    summaries.
    """

    required_kinds: tuple[str, ...] = (
        "target_source",
        "surrounding_source",
        "test_source",
    )
    never_compress_kinds: tuple[str, ...] = (
        "target_source",
        "surrounding_source",
        "test_source",
    )
    allow_capsule_substitution: bool = True
    require_exclusion_explanations: bool = True

    def __post_init__(self) -> None:
        required = _unique_sorted_texts(list(self.required_kinds), "required_kinds")
        never = _unique_sorted_texts(
            list(self.never_compress_kinds), "never_compress_kinds"
        )
        object.__setattr__(self, "required_kinds", required)
        object.__setattr__(self, "never_compress_kinds", never)
        if not isinstance(self.allow_capsule_substitution, bool):
            raise ContextPackError("allow_capsule_substitution must be a boolean")
        if not isinstance(self.require_exclusion_explanations, bool):
            raise ContextPackError("require_exclusion_explanations must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTEXT_COVERAGE_POLICY_SCHEMA,
            "required_kinds": list(self.required_kinds),
            "never_compress_kinds": list(self.never_compress_kinds),
            "allow_capsule_substitution": self.allow_capsule_substitution,
            "require_exclusion_explanations": self.require_exclusion_explanations,
        }


@dataclass(frozen=True)
class ContextTokenEstimate:
    """Deterministic per-category token accounting."""

    totals: Mapping[str, int]
    estimator_version: str
    total: int

    def __post_init__(self) -> None:
        if not isinstance(self.totals, Mapping):
            raise ContextPackError("totals must be an object")
        cleaned: dict[str, int] = {}
        for key, value in self.totals.items():
            name = _text(key, "token category")
            if type(value) is not int or isinstance(value, bool) or value < 0:
                raise ContextPackError(f"token total for {name} must be nonnegative int")
            cleaned[name] = value
        object.__setattr__(
            self, "totals", {key: cleaned[key] for key in sorted(cleaned)}
        )
        version = _text(self.estimator_version, "estimator_version")
        object.__setattr__(self, "estimator_version", version)
        expected_total = sum(self.totals.values())
        if type(self.total) is not int or isinstance(self.total, bool) or self.total < 0:
            raise ContextPackError("total must be a nonnegative integer")
        if self.total != expected_total:
            raise ContextPackError(
                f"token total {self.total} does not match category sum {expected_total}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "totals": dict(self.totals),
            "estimator_version": self.estimator_version,
            "total": self.total,
        }


@dataclass(frozen=True)
class ContextPackResult:
    """Compiled pack plus projection/accounting witnesses."""

    pack: ContextPack
    pack_cid: str
    references: tuple[ContextReference, ...]
    token_estimate: ContextTokenEstimate
    coverage_satisfied: bool
    production_slice: Any = None
    production_slice_cid: str | None = None
    budget_exceeded: bool = False
    decisions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTEXT_PACK_RESULT_SCHEMA,
            "interface": CONTEXT_PACK_INTERFACE,
            "pack": self.pack.to_dict(),
            "pack_cid": self.pack_cid,
            "references": [item.to_dict() for item in self.references],
            "token_estimate": self.token_estimate.to_dict(),
            "coverage_satisfied": self.coverage_satisfied,
            "production_slice_cid": self.production_slice_cid,
            "budget_exceeded": self.budget_exceeded,
            "decisions": list(self.decisions),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cid_list(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ContextPackError(f"{name} must be a list")
    return _unique_sorted_cids(list(values), name)


def _text_list(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ContextPackError(f"{name} must be a list")
    return _unique_sorted_texts(list(values), name)


def _estimate_category(
    estimator: CalibratedTokenEstimator,
    category: str,
    payload: Any,
) -> int:
    body = {"category": category, "payload": payload}
    return int(estimator.estimate(body))


def _reference(
    *,
    reference_id: str,
    kind: str,
    tier: ContextTier,
    content_id: str,
    summary: str,
    required: bool,
    metadata: Mapping[str, Any] | None = None,
    token_count: int = 0,
    byte_count: int = 0,
    path: str = "",
) -> ContextReference:
    meta: dict[str, Any] = dict(metadata or {})
    if required:
        meta["required"] = True
    meta.setdefault("coverage_ids", (kind,))
    return ContextReference(
        reference_id=reference_id,
        kind=kind,
        tier=tier,
        referenced_content_id=content_id,
        summary=summary,
        token_count=token_count,
        byte_count=byte_count,
        path=path,
        metadata=meta,
    )


def _admission_of(item: Any) -> CapsuleAdmission:
    if isinstance(item, CapsuleAdmission):
        return item
    if isinstance(item, Mapping):
        return CapsuleAdmission.from_dict(item)
    raise ContextPackError("dependency capsule admissions must be CapsuleAdmission records")


def _risk_for(
    *,
    lowest_confidence: str,
    obligation_count: int,
    raw_source_count: int,
    budget_exceeded: bool,
) -> str:
    if budget_exceeded:
        return "critical"
    if lowest_confidence in {"opaque", "heuristic"} or raw_source_count > 0:
        if obligation_count > 0:
            return "high"
        return "medium"
    if obligation_count > 0:
        return "medium"
    if lowest_confidence == "conservative":
        return "low"
    return "low"


def _route_for(
    *,
    budget_exceeded: bool,
    coverage_satisfied: bool,
    risk: str,
    obligation_count: int,
) -> str:
    if budget_exceeded or not coverage_satisfied:
        return ModelRoute.HUMAN_REVIEW_REQUIRED.value
    if risk in {"critical", "high"}:
        return ModelRoute.FRONTIER_MODEL.value
    if risk == "medium" or obligation_count > 0:
        return ModelRoute.MEDIUM_MODEL.value
    if risk == "low":
        return ModelRoute.SMALL_LOCAL_MODEL.value
    return ModelRoute.DETERMINISTIC_ONLY.value


def _escalation_for(
    *,
    budget_exceeded: bool,
    coverage_satisfied: bool,
    raw_forced: int,
    risk: str,
) -> str:
    if budget_exceeded:
        return (
            "budget_failure:escalate_or_human_review;"
            "required coverage cannot be truncated"
        )
    if not coverage_satisfied:
        return "coverage_failure:escalate_or_human_review"
    if raw_forced > 0:
        return "raw_source_included:review_opaque_or_stale_dependencies"
    if risk in {"high", "critical"}:
        return "assurance_risk:consider_human_review"
    return "none"


# ---------------------------------------------------------------------------
# Packer
# ---------------------------------------------------------------------------


@dataclass
class ContextPacker:
    """Compatibility/delegation surface for ContextPack compilation (PCCE-012).

    Production v0.1 pack identity is datasets-owned
    (``ipfs_datasets_py.proof_context.context_pack``). This class remains the
    legacy token/budget consumer and is not v0.1 construction authority.
    """

    V01_PRODUCTION_AUTHORITY: ClassVar[bool] = False
    V01_DELEGATE: ClassVar[str] = (
        "ipfs_datasets_py.proof_context.context_pack.build_context_pack"
    )

    budget: ContextBudget = field(default_factory=ContextBudget)
    policy: ContextCoveragePolicy = field(default_factory=ContextCoveragePolicy)
    estimator_version: str = TOKEN_ESTIMATOR_VERSION
    _compiler: ContextCompiler | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.budget, ContextBudget):
            if isinstance(self.budget, Mapping):
                self.budget = ContextBudget.from_dict(self.budget)
            else:
                raise ContextPackError("budget must be a ContextBudget")
        if not isinstance(self.policy, ContextCoveragePolicy):
            raise ContextPackError("policy must be a ContextCoveragePolicy")
        self.estimator_version = _text(self.estimator_version, "estimator_version")
        if self._compiler is None:
            self._compiler = ContextCompiler(self.budget)

    def pack_v01(self, **kwargs: Any) -> Any:
        """Delegate v0.1 construction to datasets. Does not reconstruct packs."""
        from ipfs_accelerate_py.proof_context.semantic_bridge import (
            build_v01_context_pack,
        )

        return build_v01_context_pack(**kwargs)

    @property
    def compiler(self) -> ContextCompiler:
        assert self._compiler is not None
        return self._compiler

    @property
    def effective_input_limit(self) -> int:
        return int(self.compiler.effective_input_limit)

    def pack(
        self,
        *,
        objective: str,
        target_source_cid: str,
        surrounding_source_cid: str,
        test_source_cid: str,
        dependency_admissions: Sequence[Any] = (),
        obligation_cids: Sequence[str] = (),
        counterexample_cids: Sequence[str] = (),
        delta_cid: str,
        interface_cids: Sequence[str] = (),
        assumptions: Sequence[str] = (),
        exclusions: Sequence[str] | None = None,
        raw_source_regions: Sequence[Mapping[str, Any]] = (),
        production_slice: Any = None,
        production_slice_builder: Any = None,
    ) -> ContextPackResult:
        """Compile one deterministic ContextPack.

        Required exact source CIDs are always included. Dependency capsules are
        included only when ``capsule_may_substitute`` is true; otherwise raw
        source regions or exclusions explain the fallback. Model summaries are
        never accepted as coverage for required kinds.
        """

        objective_text = _text(objective, "objective")
        target_cid = validate_opaque_cid(target_source_cid, "target_source_cid")
        surrounding_cid = validate_opaque_cid(
            surrounding_source_cid, "surrounding_source_cid"
        )
        test_cid = validate_opaque_cid(test_source_cid, "test_source_cid")
        delta = validate_opaque_cid(delta_cid, "delta_cid")
        obligations = _cid_list(list(obligation_cids), "obligation_cids")
        counterexamples = _cid_list(list(counterexample_cids), "counterexample_cids")
        interfaces = _cid_list(list(interface_cids), "interface_cids")
        assumption_list = _text_list(list(assumptions), "assumptions")

        admissions = [_admission_of(item) for item in dependency_admissions]
        # Deterministic order by capsule CID.
        admissions = sorted(admissions, key=lambda item: item.ref.capsule_cid)

        decisions: list[str] = []
        references: list[ContextReference] = []
        token_totals: dict[str, int] = {}
        estimator = self.compiler.estimator
        included_capsule_cids: list[str] = []
        exclusion_list: list[str] = list(exclusions or [])
        confidences: list[str] = []
        raw_forced = 0

        # --- Required exact source (never compressed) -----------------------
        for kind, cid, summary in (
            ("target_source", target_cid, "exact target source"),
            ("surrounding_source", surrounding_cid, "exact surrounding edit context"),
            ("test_source", test_cid, "exact directly edited tests"),
        ):
            tokens = _estimate_category(
                estimator, kind, {"source_cid": cid, "summary": summary}
            )
            token_totals[kind] = tokens
            references.append(
                _reference(
                    reference_id=f"required:{kind}",
                    kind=kind,
                    tier=ContextTier.INVARIANT,
                    content_id=cid,
                    summary=summary,
                    required=True,
                    token_count=tokens,
                    metadata={"never_compress": True, "source_cid": cid},
                )
            )
            decisions.append(f"include:{kind}:{cid}")

        # --- Dependency capsules / raw fallback -----------------------------
        dep_tokens = 0
        for admission in admissions:
            confidences.append(admission.ref.confidence)
            if (
                self.policy.allow_capsule_substitution
                and capsule_may_substitute(admission)
            ):
                included_capsule_cids.append(admission.ref.capsule_cid)
                payload = {
                    "capsule_cid": admission.ref.capsule_cid,
                    "admission": admission.admission,
                    "confidence": admission.ref.confidence,
                    "caveats": list(admission.caveats),
                }
                tokens = _estimate_category(estimator, "dependency_capsule", payload)
                dep_tokens += tokens
                tier = ContextTier.EVIDENCE
                if admission.admission == ADMISSION_CONSERVATIVE:
                    # Conservative remains visible.
                    assumption_list = _text_list(
                        list(assumption_list)
                        + [
                            f"conservative_capsule:{admission.ref.capsule_cid}"
                        ]
                        + [f"caveat:{item}" for item in admission.caveats],
                        "assumptions",
                    )
                references.append(
                    _reference(
                        reference_id=f"capsule:{admission.ref.capsule_cid}",
                        kind="dependency_capsule",
                        tier=tier,
                        content_id=admission.ref.capsule_cid,
                        summary=(
                            f"admitted {admission.admission} capsule "
                            f"({admission.ref.confidence})"
                        ),
                        required=False,
                        token_count=tokens,
                        metadata={
                            "admission": admission.admission,
                            "confidence": admission.ref.confidence,
                            "caveats": list(admission.caveats),
                            "raw_source_required": False,
                            "datasets_authority": True,
                        },
                    )
                )
                decisions.append(
                    f"substitute:capsule:{admission.ref.capsule_cid}:{admission.admission}"
                )
            else:
                raw_forced += 1
                reason = (
                    admission.admission
                    if admission.admission == ADMISSION_RAW
                    else "not_substitutable"
                )
                exclusion = (
                    f"excluded_capsule_body:{admission.ref.capsule_cid}:"
                    f"{reason}:retrieve_raw_source:{admission.ref.source_cid}"
                )
                exclusion_list.append(exclusion)
                # Include exact source CID as required raw evidence.
                tokens = _estimate_category(
                    estimator,
                    "raw_dependency_source",
                    {
                        "source_cid": admission.ref.source_cid,
                        "capsule_cid": admission.ref.capsule_cid,
                        "reason": reason,
                    },
                )
                dep_tokens += tokens
                references.append(
                    _reference(
                        reference_id=f"raw:{admission.ref.source_cid}",
                        kind="raw_dependency_source",
                        tier=ContextTier.INVARIANT,
                        content_id=admission.ref.source_cid,
                        summary=(
                            f"raw tree-bound source for non-substitutable capsule "
                            f"{admission.ref.capsule_cid}"
                        ),
                        required=True,
                        token_count=tokens,
                        metadata={
                            "capsule_cid": admission.ref.capsule_cid,
                            "confidence": admission.ref.confidence,
                            "admission": admission.admission,
                            "raw_source_required": True,
                            "datasets_authority": True,
                            "caveats": list(admission.caveats),
                        },
                    )
                )
                decisions.append(
                    f"raw_source:{admission.ref.capsule_cid}:{admission.ref.source_cid}"
                )
                for caveat in admission.caveats:
                    assumption_list = _text_list(
                        list(assumption_list) + [f"caveat:{caveat}"],
                        "assumptions",
                    )

        token_totals["dependency_capsules"] = dep_tokens

        # Explicit raw source regions (opaque symbols without admitted capsule).
        raw_extra_tokens = 0
        for index, region in enumerate(raw_source_regions or ()):
            if not isinstance(region, Mapping):
                raise ContextPackError("raw_source_regions entries must be objects")
            source_cid = validate_opaque_cid(
                region.get("source_cid"), f"raw_source_regions[{index}].source_cid"
            )
            reason = _text(
                region.get("reason", "opaque_or_missing_capsule"),
                f"raw_source_regions[{index}].reason",
            )
            tokens = _estimate_category(
                estimator,
                "raw_source_region",
                {"source_cid": source_cid, "reason": reason},
            )
            raw_extra_tokens += tokens
            raw_forced += 1
            references.append(
                _reference(
                    reference_id=f"raw-region:{source_cid}",
                    kind="raw_source_region",
                    tier=ContextTier.INVARIANT,
                    content_id=source_cid,
                    summary=f"exact scanned-tree source ({reason})",
                    required=True,
                    token_count=tokens,
                    path=str(region.get("path") or ""),
                    metadata={"reason": reason, "never_compress": True},
                )
            )
            decisions.append(f"raw_region:{source_cid}:{reason}")
        token_totals["raw_source_regions"] = raw_extra_tokens

        # Obligations / counterexamples / delta / interfaces
        token_totals["obligations"] = _estimate_category(
            estimator, "obligations", list(obligations)
        )
        token_totals["counterexamples"] = _estimate_category(
            estimator, "counterexamples", list(counterexamples)
        )
        token_totals["delta"] = _estimate_category(
            estimator, "delta", {"delta_cid": delta}
        )
        token_totals["interfaces"] = _estimate_category(
            estimator, "interfaces", list(interfaces)
        )
        token_totals["assumptions"] = _estimate_category(
            estimator, "assumptions", list(assumption_list)
        )

        for cid in obligations:
            references.append(
                _reference(
                    reference_id=f"obligation:{cid}",
                    kind="obligation",
                    tier=ContextTier.EVIDENCE,
                    content_id=cid,
                    summary="unresolved obligation",
                    required=True,
                    metadata={"required": True},
                )
            )
        for cid in counterexamples:
            references.append(
                _reference(
                    reference_id=f"counterexample:{cid}",
                    kind="counterexample",
                    tier=ContextTier.EVIDENCE,
                    content_id=cid,
                    summary="minimized counterexample",
                    required=False,
                )
            )
        references.append(
            _reference(
                reference_id=f"delta:{delta}",
                kind="repository_delta",
                tier=ContextTier.EVIDENCE,
                content_id=delta,
                summary="current repository-state delta",
                required=True,
            )
        )
        for cid in interfaces:
            references.append(
                _reference(
                    reference_id=f"interface:{cid}",
                    kind="interface_schema",
                    tier=ContextTier.EVIDENCE,
                    content_id=cid,
                    summary="MCP/public interface schema",
                    required=False,
                )
            )

        # Suggestions (LLM summaries) — never required, never raise confidence.
        # Callers may pass assumptions already; do not invent summaries here.

        exclusion_list = list(_text_list(exclusion_list, "exclusions"))
        if self.policy.require_exclusion_explanations:
            for item in exclusion_list:
                if ":" not in item:
                    raise ContextPackError(
                        f"exclusion must explain the omitted region: {item!r}"
                    )

        # Coverage: required kinds present as invariant references.
        present_kinds = {
            item.kind for item in references if item.tier is ContextTier.INVARIANT
        }
        coverage_satisfied = all(
            kind in present_kinds for kind in self.policy.required_kinds
        )
        # Exact CIDs must remain non-empty and distinct fields on the pack.
        if not (target_cid and surrounding_cid and test_cid and delta):
            coverage_satisfied = False

        token_estimate = ContextTokenEstimate(
            totals=token_totals,
            estimator_version=self.estimator_version,
            total=sum(token_totals.values()),
        )
        budget_exceeded = token_estimate.total > self.effective_input_limit
        if budget_exceeded:
            decisions.append(
                f"budget_exceeded:total={token_estimate.total}:"
                f"limit={self.effective_input_limit}"
            )
            # Do not truncate required coverage; escalate instead.
            coverage_for_route = False
        else:
            coverage_for_route = coverage_satisfied

        lowest = "exact"
        rank = {"exact": 0, "conservative": 1, "heuristic": 2, "opaque": 3}
        for conf in confidences:
            if rank.get(conf, 3) > rank.get(lowest, 0):
                lowest = conf
        if raw_forced and lowest in {"exact", "conservative"}:
            # Raw inclusion implies assurance no higher than heuristic path.
            lowest = "heuristic"

        risk = _risk_for(
            lowest_confidence=lowest,
            obligation_count=len(obligations),
            raw_source_count=raw_forced,
            budget_exceeded=budget_exceeded,
        )
        route = _route_for(
            budget_exceeded=budget_exceeded,
            coverage_satisfied=coverage_for_route and coverage_satisfied,
            risk=risk,
            obligation_count=len(obligations),
        )
        escalation = _escalation_for(
            budget_exceeded=budget_exceeded,
            coverage_satisfied=coverage_satisfied and not budget_exceeded,
            raw_forced=raw_forced,
            risk=risk,
        )

        # Stable sorted dependency capsule CIDs (only admitted substitutes).
        dep_cids = tuple(sorted(set(included_capsule_cids)))

        pack = ContextPack(
            objective=objective_text,
            target_source_cid=target_cid,
            surrounding_source_cid=surrounding_cid,
            test_source_cid=test_cid,
            dependency_capsule_cids=dep_cids,
            obligation_cids=obligations,
            counterexample_cids=counterexamples,
            delta_cid=delta,
            interface_cids=interfaces,
            assumptions=assumption_list,
            exclusions=tuple(exclusion_list),
            token_totals=dict(token_estimate.totals),
            estimator_version=token_estimate.estimator_version,
            risk=risk,
            route=route,
            escalation_recommendation=escalation,
        )
        pack_cid = cid_for_payload(pack.to_dict())

        # Optional production source-coverage proof.
        slice_obj = production_slice
        slice_cid: str | None = None
        if slice_obj is None and callable(production_slice_builder):
            slice_obj = production_slice_builder()
        if slice_obj is not None:
            slice_cid = _production_slice_cid(slice_obj)
            decisions.append(f"production_slice:{slice_cid}")

        # Deterministic decision order.
        decisions_sorted = tuple(sorted(set(decisions)))
        references_sorted = tuple(
            sorted(references, key=lambda item: item.reference_id)
        )

        return ContextPackResult(
            pack=pack,
            pack_cid=pack_cid,
            references=references_sorted,
            token_estimate=token_estimate,
            coverage_satisfied=coverage_satisfied and not budget_exceeded,
            production_slice=slice_obj,
            production_slice_cid=slice_cid,
            budget_exceeded=budget_exceeded,
            decisions=decisions_sorted,
        )


def _production_slice_cid(slice_obj: Any) -> str:
    if hasattr(slice_obj, "manifest_cid"):
        return validate_opaque_cid(slice_obj.manifest_cid, "production_slice.manifest_cid")
    if isinstance(slice_obj, Mapping):
        cid = slice_obj.get("manifest_cid")
        if cid:
            return validate_opaque_cid(cid, "production_slice.manifest_cid")
        return cid_for_payload(dict(slice_obj))
    raise ContextPackError("production_slice must expose manifest_cid or be a mapping")


def pack_context(
    *,
    objective: str,
    target_source_cid: str,
    surrounding_source_cid: str,
    test_source_cid: str,
    dependency_admissions: Sequence[Any] = (),
    obligation_cids: Sequence[str] = (),
    counterexample_cids: Sequence[str] = (),
    delta_cid: str,
    interface_cids: Sequence[str] = (),
    assumptions: Sequence[str] = (),
    exclusions: Sequence[str] | None = None,
    raw_source_regions: Sequence[Mapping[str, Any]] = (),
    budget: ContextBudget | Mapping[str, Any] | None = None,
    policy: ContextCoveragePolicy | None = None,
    production_slice: Any = None,
    production_slice_builder: Any = None,
    estimator_version: str = TOKEN_ESTIMATOR_VERSION,
) -> ContextPackResult:
    """Module-level entry point for assurance-aware context packing."""

    selected_budget: ContextBudget
    if budget is None:
        selected_budget = ContextBudget()
    elif isinstance(budget, ContextBudget):
        selected_budget = budget
    elif isinstance(budget, Mapping):
        selected_budget = ContextBudget.from_dict(budget)
    else:
        raise ContextPackError("budget must be a ContextBudget or mapping")

    packer = ContextPacker(
        budget=selected_budget,
        policy=policy or ContextCoveragePolicy(),
        estimator_version=estimator_version,
    )
    return packer.pack(
        objective=objective,
        target_source_cid=target_source_cid,
        surrounding_source_cid=surrounding_source_cid,
        test_source_cid=test_source_cid,
        dependency_admissions=dependency_admissions,
        obligation_cids=obligation_cids,
        counterexample_cids=counterexample_cids,
        delta_cid=delta_cid,
        interface_cids=interface_cids,
        assumptions=assumptions,
        exclusions=exclusions,
        raw_source_regions=raw_source_regions,
        production_slice=production_slice,
        production_slice_builder=production_slice_builder,
    )


def project_admission_to_reference(
    admission: CapsuleAdmission,
    *,
    token_count: int = 0,
) -> ContextReference:
    """Project one admission into a ContextReference tier."""

    if not isinstance(admission, CapsuleAdmission):
        raise ContextPackError("admission must be a CapsuleAdmission")
    if capsule_may_substitute(admission):
        tier = ContextTier.EVIDENCE
        kind = "dependency_capsule"
        content_id = admission.ref.capsule_cid
        required = False
        summary = f"admitted {admission.admission} capsule"
    else:
        tier = ContextTier.INVARIANT
        kind = "raw_dependency_source"
        content_id = admission.ref.source_cid
        required = True
        summary = "raw tree-bound source (non-substitutable capsule)"
    return _reference(
        reference_id=f"proj:{content_id}",
        kind=kind,
        tier=tier,
        content_id=content_id,
        summary=summary,
        required=required,
        token_count=token_count,
        metadata={
            "admission": admission.admission,
            "confidence": admission.ref.confidence,
            "caveats": list(admission.caveats),
            "datasets_authority": True,
            "raw_source_required": admission.requires_raw_source,
        },
    )


# ---------------------------------------------------------------------------
# Freshness admission: Datasets identity + Kit bytes/root (ASEH-033)
# ---------------------------------------------------------------------------


def _identity_tuple(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return _dedupe_sorted((values,), name)
    if not isinstance(values, (list, tuple)):
        raise ContextPackError(f"{name} must be a list")
    return _dedupe_sorted(list(values), name)


def _require_git_oid(value: Any, name: str) -> str:
    text = _text(value, name)
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise ContextPackError(f"{name} must be a lowercase git object id")
    return text


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or isinstance(value, (str, bytes)):
        raise ContextPackError(f"{name} must be an object")
    return value


def load_datasets_context_pack_authority() -> Any:
    """Load the installed Datasets ContextPack authority. Never remints CIDs."""

    try:
        from ipfs_datasets_py.proof_context.context_pack import (
            DatasetsContextPackAuthority,
            get_authority,
        )
    except ImportError as exc:
        raise ContextPackAuthorityUnavailable(
            "datasets ContextPack authority is unavailable; "
            "accelerator must not remint semantic identity"
        ) from exc
    authority = get_authority()
    if not isinstance(authority, DatasetsContextPackAuthority):
        raise ContextPackError("installed Datasets authority type is not admitted")
    if getattr(authority, "producer", None) != DATASETS_CONTEXT_PACK_AUTHORITY:
        raise ContextPackError("ownership drift: Datasets is not the sole builder")
    return authority


def load_kit_context_pack_store() -> Any:
    """Load the installed Kit ContextPack store module. Never bypasses bytes."""

    try:
        from ipfs_kit_py.proof_context import state_store as kit_store
    except ImportError as exc:
        raise ContextPackAuthorityUnavailable(
            "kit ContextPack store is unavailable; "
            "accelerator must not invent durable bytes or roots"
        ) from exc
    if getattr(kit_store, "CONTEXT_PACK_NAMESPACE", None) != "ContextPack":
        raise ContextPackError("kit ContextPack namespace mismatch")
    return kit_store


def encode_context_pack_envelope(envelope: Mapping[str, Any]) -> bytes:
    """Canonical JSON encoding of a Datasets envelope for Kit byte storage."""

    closed = _require_mapping(envelope, "envelope")
    try:
        return json.dumps(
            dict(closed),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContextPackError("ContextPack envelope is not JSON-serializable") from exc


def interface_identities_of(envelope: Mapping[str, Any]) -> tuple[str, ...]:
    """Exact interface/schema identities bound into a Datasets envelope."""

    closed = _require_mapping(envelope, "envelope")
    identity = _require_mapping(closed.get("identity"), "identity")
    bindings = _require_mapping(
        closed.get("freshness_bindings"), "freshness_bindings"
    )
    values = [
        _text(closed.get("interface"), "interface"),
        _text(
            identity.get("schema_and_interface_version"),
            "identity.schema_and_interface_version",
        ),
        _text(closed.get("schema"), "schema"),
    ]
    values.extend(
        _identity_tuple(bindings.get("schema_identities"), "schema_identities")
    )
    return _dedupe_sorted(values, "interface_identities")


@dataclass(frozen=True)
class CurrentPackIdentity:
    """Exact current-tree identities that a reusable pack must bind."""

    tree: str
    objective_identity: str
    objective_revision: str
    policy_identity: str
    interface_identities: tuple[str, ...]
    toolchain_identities: tuple[str, ...]
    environment_requirements: tuple[str, ...]
    required_source_cids: Mapping[str, str] = field(default_factory=dict)
    identity_kind: str = "live"
    evidence_kind: str = "real"
    execution_mode: str = "live"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree", _require_git_oid(self.tree, "tree"))
        object.__setattr__(
            self,
            "objective_identity",
            _text(self.objective_identity, "objective_identity"),
        )
        object.__setattr__(
            self,
            "objective_revision",
            _text(self.objective_revision, "objective_revision"),
        )
        object.__setattr__(
            self,
            "policy_identity",
            validate_opaque_cid(self.policy_identity, "policy_identity"),
        )
        object.__setattr__(
            self,
            "interface_identities",
            _identity_tuple(self.interface_identities, "interface_identities"),
        )
        if not self.interface_identities:
            raise ContextPackError("interface_identities must be nonempty")
        object.__setattr__(
            self,
            "toolchain_identities",
            _identity_tuple(self.toolchain_identities, "toolchain_identities"),
        )
        object.__setattr__(
            self,
            "environment_requirements",
            _identity_tuple(
                self.environment_requirements, "environment_requirements"
            ),
        )
        sources = _require_mapping(
            self.required_source_cids, "required_source_cids"
        )
        cleaned: dict[str, str] = {}
        for kind, cid in sources.items():
            name = _text(kind, "required source kind")
            cleaned[name] = validate_opaque_cid(cid, f"required_source_cids.{name}")
        object.__setattr__(
            self,
            "required_source_cids",
            {key: cleaned[key] for key in sorted(cleaned)},
        )
        object.__setattr__(
            self, "identity_kind", _text(self.identity_kind, "identity_kind")
        )
        object.__setattr__(
            self, "evidence_kind", _text(self.evidence_kind, "evidence_kind")
        )
        object.__setattr__(
            self, "execution_mode", _text(self.execution_mode, "execution_mode")
        )

    @classmethod
    def from_envelope(cls, envelope: Mapping[str, Any]) -> "CurrentPackIdentity":
        """Derive current-world identities from an already-verified envelope."""

        closed = _require_mapping(envelope, "envelope")
        identity = _require_mapping(closed.get("identity"), "identity")
        bindings = _require_mapping(
            closed.get("freshness_bindings"), "freshness_bindings"
        )
        sources = _require_mapping(
            closed.get("required_source_cids"), "required_source_cids"
        )
        return cls(
            tree=_text(identity.get("tree"), "identity.tree"),
            objective_identity=_text(
                identity.get("objective_identity"), "identity.objective_identity"
            ),
            objective_revision=_text(
                identity.get("objective_revision"), "identity.objective_revision"
            ),
            policy_identity=_text(
                identity.get("policy_identity"), "identity.policy_identity"
            ),
            interface_identities=interface_identities_of(closed),
            toolchain_identities=_identity_tuple(
                bindings.get("toolchain_identities"), "toolchain_identities"
            ),
            environment_requirements=_identity_tuple(
                bindings.get("environment_requirements"), "environment_requirements"
            ),
            required_source_cids=dict(sources),
            identity_kind=_text(closed.get("identity_kind"), "identity_kind"),
            evidence_kind=_text(closed.get("evidence_kind"), "evidence_kind"),
            execution_mode=_text(closed.get("execution_mode"), "execution_mode"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree": self.tree,
            "objective_identity": self.objective_identity,
            "objective_revision": self.objective_revision,
            "policy_identity": self.policy_identity,
            "interface_identities": list(self.interface_identities),
            "toolchain_identities": list(self.toolchain_identities),
            "environment_requirements": list(self.environment_requirements),
            "required_source_cids": dict(self.required_source_cids),
            "identity_kind": self.identity_kind,
            "evidence_kind": self.evidence_kind,
            "execution_mode": self.execution_mode,
        }


@dataclass(frozen=True)
class FreshnessVerdict:
    """Exact freshness comparison against current-tree identities."""

    fresh: bool
    stale_fields: tuple[str, ...]
    pack_cid: str
    identity_kind: str
    masquerade_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTEXT_PACK_FRESHNESS_SCHEMA,
            "interface": CONTEXT_PACK_FRESHNESS_INTERFACE,
            "fresh": self.fresh,
            "stale_fields": list(self.stale_fields),
            "pack_cid": self.pack_cid,
            "identity_kind": self.identity_kind,
            "masquerade_reasons": list(self.masquerade_reasons),
        }


def verify_datasets_semantic_identity(
    envelope: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-verify Datasets identity. Accelerate never remints ``pack_cid``."""

    claimed = envelope.get("pack_cid") if isinstance(envelope, Mapping) else None
    authority = load_datasets_context_pack_authority()
    try:
        validated = authority.validate_envelope(envelope)
    except Exception as exc:
        raise ContextPackError(
            f"datasets semantic identity rejected the envelope; "
            f"storage is not semantic proof: {exc}"
        ) from exc
    pack_cid = _text(validated.get("pack_cid"), "pack_cid")
    if claimed is None or _text(claimed, "pack_cid") != pack_cid:
        raise ContextPackError(
            "datasets pack_cid remint is forbidden; claimed identity must stand"
        )
    if validated.get("producer") != DATASETS_CONTEXT_PACK_AUTHORITY:
        raise ContextPackError("ownership drift: producer is not Datasets")
    if validated.get("interface") in {"SupervisorContextPack@1", "SupervisorContextPack"}:
        raise ContextPackError("ownership drift: competing ContextPack type")
    return validated


def decode_context_pack_envelope(data: bytes) -> dict[str, Any]:
    """Parse Kit-stored bytes and verify the Datasets envelope identity."""

    if type(data) is not bytes or not data:
        raise ContextPackError("ContextPack bytes must be nonempty")
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContextPackError(
            "stored bytes are not a ContextPack envelope; storage is not semantic proof"
        ) from exc
    if not isinstance(payload, dict):
        raise ContextPackError("ContextPack envelope must be an object")
    return verify_datasets_semantic_identity(payload)


def kit_context_pack_store_of(store: Any) -> Any:
    """Return the Kit ContextPack namespace from a store or parent port."""

    if store is None:
        raise ContextPackError("kit ContextPack store is required")
    if hasattr(store, "get_immutable") and hasattr(store, "current_root"):
        return store
    nested = getattr(store, "context_pack", None)
    if nested is not None and hasattr(nested, "get_immutable"):
        return nested
    raise ContextPackError("kit ContextPack store is required")


def verify_kit_bytes(
    store: Any,
    data: bytes,
    *,
    claimed_cid: str | None = None,
) -> str:
    """Recompute Kit CID of exact bytes. Storage is not semantic admission."""

    pack_store = kit_context_pack_store_of(store)
    if type(data) is not bytes or not data:
        raise ContextPackError("kit ContextPack bytes must be nonempty")
    actual = _text(pack_store.cid_for(data), "kit_cid")
    if claimed_cid is not None and validate_opaque_cid(claimed_cid, "kit_cid") != actual:
        raise ContextPackError("kit CID does not equal recomputed bytes CID")
    return actual


def load_verified_kit_bytes(
    store: Any,
    *,
    kit_cid: str,
    kind: Any,
) -> bytes:
    """Immutable Kit get: stored bytes must match the durable CID."""

    pack_store = kit_context_pack_store_of(store)
    kit_mod = load_kit_context_pack_store()
    try:
        from ipfs_kit_py.proof_seal_store.contracts import ArtifactKind, ArtifactReference
    except ImportError as exc:
        raise ContextPackAuthorityUnavailable(
            "kit artifact contracts are unavailable"
        ) from exc
    closed_kind = kind
    try:
        if not isinstance(closed_kind, ArtifactKind):
            closed_kind = ArtifactKind(kind)
        reference = ArtifactReference(
            cid=validate_opaque_cid(kit_cid, "kit_cid"),
            kind=closed_kind,
        )
        data = pack_store.get_immutable(reference)
    except ContextPackError:
        raise
    except Exception as exc:
        raise ContextPackError(f"kit immutable get failed closed: {exc}") from exc
    actual = pack_store.cid_for(data)
    if actual != reference.cid:
        raise ContextPackError("stored bytes do not match kit CID")
    if kit_mod.cid_for_bytes(data) != reference.cid:
        raise ContextPackError("kit CID verification bypass is forbidden")
    return data


def verify_kit_current_root(store: Any, kit_cid: str) -> Any:
    """Require ``kit_cid`` to be the published current ContextPack root."""

    pack_store = kit_context_pack_store_of(store)
    pointer = pack_store.current_root()
    expected = validate_opaque_cid(kit_cid, "kit_cid")
    if pointer is None:
        raise StaleIdentityError(
            "no current ContextPack root",
            stale_fields=("tree",),
            reason_codes=("stale:current_root",),
        )
    if pointer.seal_cid != expected:
        raise StaleIdentityError(
            "pack is not the current kit root",
            stale_fields=("tree",),
            reason_codes=("stale:current_root",),
        )
    return pointer


def evaluate_exact_freshness(
    envelope: Mapping[str, Any],
    current: CurrentPackIdentity,
) -> FreshnessVerdict:
    """Compare Datasets-bound identities to the current tree exactly."""

    if not isinstance(current, CurrentPackIdentity):
        raise ContextPackError("current identity must be a CurrentPackIdentity")
    closed = _require_mapping(envelope, "envelope")
    identity = _require_mapping(closed.get("identity"), "identity")
    bindings = _require_mapping(
        closed.get("freshness_bindings"), "freshness_bindings"
    )
    pack_cid = validate_opaque_cid(closed.get("pack_cid"), "pack_cid")
    stale: list[str] = []
    masquerade: list[str] = []

    pack_tree = _text(identity.get("tree"), "identity.tree")
    scanned = _text(closed.get("scanned_tree_oid"), "scanned_tree_oid")
    if pack_tree != current.tree or scanned != current.tree:
        stale.append("tree")

    if _text(identity.get("objective_identity"), "objective_identity") != (
        current.objective_identity
    ) or _text(identity.get("objective_revision"), "objective_revision") != (
        current.objective_revision
    ):
        stale.append("objective")

    if (
        validate_opaque_cid(identity.get("policy_identity"), "policy_identity")
        != current.policy_identity
    ):
        stale.append("policy")

    if interface_identities_of(closed) != current.interface_identities:
        stale.append("interface")

    if (
        _identity_tuple(bindings.get("toolchain_identities"), "toolchain_identities")
        != current.toolchain_identities
    ):
        stale.append("toolchain")

    if (
        _identity_tuple(
            bindings.get("environment_requirements"), "environment_requirements"
        )
        != current.environment_requirements
    ):
        stale.append("environment")

    identity_kind = _text(closed.get("identity_kind"), "identity_kind")
    evidence_kind = _text(closed.get("evidence_kind"), "evidence_kind")
    execution_mode = _text(closed.get("execution_mode"), "execution_mode")
    if identity_kind != current.identity_kind:
        if identity_kind == "fixture" and current.identity_kind == "live":
            masquerade.append("fixture_as_live")
        elif identity_kind == "synthetic" and current.identity_kind == "live":
            masquerade.append("synthetic_as_live")
        else:
            masquerade.append("identity_kind_mismatch")
    if evidence_kind != current.evidence_kind:
        masquerade.append("evidence_kind_mismatch")
    if execution_mode != current.execution_mode:
        masquerade.append("execution_mode_mismatch")

    stale_fields = _dedupe_sorted(stale, "stale_fields")
    masquerade_reasons = _dedupe_sorted(masquerade, "masquerade_reasons")
    fresh = not stale_fields and not masquerade_reasons
    return FreshnessVerdict(
        fresh=fresh,
        stale_fields=stale_fields,
        pack_cid=pack_cid,
        identity_kind=identity_kind,
        masquerade_reasons=masquerade_reasons,
    )


def require_exact_freshness(
    envelope: Mapping[str, Any],
    current: CurrentPackIdentity,
) -> FreshnessVerdict:
    """Fail closed on any exact-identity mismatch."""

    verdict = evaluate_exact_freshness(envelope, current)
    if verdict.fresh:
        return verdict
    reasons = [f"stale:{field}" for field in verdict.stale_fields]
    reasons.extend(verdict.masquerade_reasons)
    raise StaleIdentityError(
        "ContextPack identity is stale for the current tree",
        stale_fields=verdict.stale_fields,
        reason_codes=reasons,
    )


def required_sources_of(envelope: Mapping[str, Any]) -> dict[str, str]:
    sources = _require_mapping(
        envelope.get("required_source_cids"), "required_source_cids"
    )
    cleaned: dict[str, str] = {}
    for kind in REQUIRED_SOURCE_KINDS:
        cleaned[kind] = validate_opaque_cid(
            sources.get(kind), f"required_source_cids.{kind}"
        )
    extra = sorted(set(sources) - set(REQUIRED_SOURCE_KINDS))
    if extra:
        raise ContextPackError(f"unknown required source kind {extra[0]}")
    return cleaned


def pack_is_adequate(
    envelope: Mapping[str, Any],
    current: CurrentPackIdentity,
) -> bool:
    """Adequate packs cover current required sources and do not omit them."""

    sources = required_sources_of(envelope)
    if current.required_source_cids and dict(current.required_source_cids) != sources:
        return False
    questions = envelope.get("questions") if isinstance(envelope, Mapping) else None
    if isinstance(questions, Mapping):
        missing = questions.get("missing_evidence") or ()
        if missing:
            return False
    return True


def pack_minimality_key(envelope: Mapping[str, Any], byte_length: int) -> tuple[Any, ...]:
    """Deterministic minimality order: fewer capsules, then fewer bytes, then CID."""

    closed = _require_mapping(envelope, "envelope")
    capsules = closed.get("capsule_cids") or ()
    if not isinstance(capsules, (list, tuple)):
        raise ContextPackError("capsule_cids must be a list")
    pack_cid = validate_opaque_cid(closed.get("pack_cid"), "pack_cid")
    if type(byte_length) is not int or isinstance(byte_length, bool) or byte_length < 0:
        raise ContextPackError("byte_length must be a nonnegative integer")
    return (len(tuple(capsules)), byte_length, pack_cid)


__all__ = [
    "CONTEXT_COVERAGE_POLICY_SCHEMA",
    "CONTEXT_PACK_FRESHNESS_INTERFACE",
    "CONTEXT_PACK_FRESHNESS_SCHEMA",
    "CONTEXT_PACK_INTERFACE",
    "CONTEXT_PACK_RESULT_SCHEMA",
    "DATASETS_CONTEXT_PACK_AUTHORITY",
    "EXACT_FRESHNESS_FIELDS",
    "KIT_CONTEXT_PACK_STORE_AUTHORITY",
    "REQUIRED_SOURCE_KINDS",
    "TOKEN_ESTIMATOR_VERSION",
    "ContextCoveragePolicy",
    "ContextPackAuthorityUnavailable",
    "ContextPackError",
    "ContextPackResult",
    "ContextPacker",
    "ContextTokenEstimate",
    "CurrentPackIdentity",
    "FreshnessVerdict",
    "StaleIdentityError",
    "decode_context_pack_envelope",
    "encode_context_pack_envelope",
    "evaluate_exact_freshness",
    "interface_identities_of",
    "kit_context_pack_store_of",
    "load_datasets_context_pack_authority",
    "load_kit_context_pack_store",
    "load_verified_kit_bytes",
    "pack_context",
    "pack_is_adequate",
    "pack_minimality_key",
    "project_admission_to_reference",
    "require_exact_freshness",
    "required_sources_of",
    "verify_datasets_semantic_identity",
    "verify_kit_bytes",
    "verify_kit_current_root",
]
