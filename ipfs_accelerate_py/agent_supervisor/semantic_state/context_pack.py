"""Assurance-aware ContextPack compilation over admitted capsules and raw source.

Projects semantic inputs into existing ``ContextReference`` tiers and accounts
tokens with ``ContextCompiler``. Exact target/edit/test spans are never
compressed. Substitutable capsules may replace unchanged dependency code only
when admission allows it, with visible caveats. Budget failures recommend
escalation instead of silent truncation. Capsule facts remain datasets-owned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

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
TOKEN_ESTIMATOR_VERSION = "context-compiler-calibrated_utf8@1"

_REQUIRED_SOURCE_KINDS = frozenset(
    {
        "target_source",
        "surrounding_source",
        "test_source",
    }
)
_NEVER_COMPRESS = frozenset(
    {
        "target_source",
        "surrounding_source",
        "test_source",
    }
)


class ContextPackError(HarnessError):
    """Closed context-pack or coverage-policy violation."""


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
    """Compile assurance-aware ContextPacks via existing context infrastructure."""

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


__all__ = [
    "CONTEXT_COVERAGE_POLICY_SCHEMA",
    "CONTEXT_PACK_INTERFACE",
    "CONTEXT_PACK_RESULT_SCHEMA",
    "TOKEN_ESTIMATOR_VERSION",
    "ContextCoveragePolicy",
    "ContextPackError",
    "ContextPackResult",
    "ContextPacker",
    "ContextTokenEstimate",
    "pack_context",
    "project_admission_to_reference",
]
