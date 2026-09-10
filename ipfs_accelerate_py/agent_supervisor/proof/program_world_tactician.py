"""Program-world Tactician integration over landed proof authorities (SAWM-018).

Interface: ``ProgramWorldTactician@1``
Evidence: ``sawm/tactician-hammer@1``

Compile transition and repair uncertainties into a finite goal inventory and a
content-addressed premise plan, then run the current Tactician.  This module
does not create a second tactic engine, prover, logic family, or acceptance
gate.  Datasets remains the semantic/formal authority; accelerate only
orchestrates operational proof search.

Normative constraints:

* Finite positive/negative/counterexample obligations only.
* Premise corpus identity is content-addressed and current-tree bound.
* Tactician plans are advisory; ``semantic_authority`` is always false.
* Models may nominate premises but never mint axioms, proofs, or authority.
* Contradictory admitted premises abstain (no ex-falso).
* Unknown, unsupported, and dynamic frontiers stay explicit residuals.
* Importing this module performs no I/O and starts no analysis.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..analysis.program_logic_premise_corpus import (
    ConsistencyDisposition,
    PremiseAuthority,
    PremiseSourceClass,
    ProgramLogicPremise,
    ProgramLogicPremiseCorpus,
    ProgramLogicPremiseCorpusBuilder,
)
from ..analysis.program_logic_prediction_contracts import (
    GoalDisposition,
    GoalFamily,
    LogicFacetKind,
    LogicFacetRef,
    LogicSubgoal,
    ProgramLogicAuthorityRoots,
    ProgramLogicGoal,
    ProofStatus,
    SourceAuthorityClass,
    SourceRouteKind,
    SubgoalDisposition,
    TacticianSearchPlan,
)
from ..analysis.program_successors import StaticSuccessorPlan, SuccessorCandidate
from .formal_verification_contracts import CanonicalContract, content_identity


PROGRAM_WORLD_TACTICIAN_INTERFACE: Final[str] = "ProgramWorldTactician@1"
PROGRAM_WORLD_PREMISE_CORPUS_INTERFACE: Final[str] = "ProgramWorldPremiseCorpus@1"
SAWM_TACTICIAN_HAMMER_EVIDENCE: Final[str] = "sawm/tactician-hammer@1"

PROGRAM_WORLD_GOAL_COMPILATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-goal-compilation@1"
)
PROGRAM_WORLD_PREMISE_CORPUS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-premise-corpus@1"
)
PROGRAM_WORLD_OBLIGATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-obligation@1"
)

PRODUCER_ID: Final[str] = "program-world-tactician@1"
PLANNER_ID: Final[str] = "program-world-tactician@1"
TACTICIAN_VERSION: Final[str] = "1"

MAX_GOALS: Final[int] = 512
MAX_OBLIGATIONS: Final[int] = 512
MAX_PREMISES: Final[int] = 10_000
MAX_NOMINATIONS: Final[int] = 256
MAX_TEXT_CHARS: Final[int] = 4_096
DEFAULT_TOP_K: Final[int] = 32

DETERMINISTIC_SOURCE_ROUTES: Final[tuple[SourceRouteKind, ...]] = (
    SourceRouteKind.LOCAL_STATIC,
    SourceRouteKind.REVIEWED_CONTRACT,
    SourceRouteKind.REVIEWED_TEST,
    SourceRouteKind.DATAFLOW,
    SourceRouteKind.GRAPH,
)
NOMINATING_SOURCE_ROUTES: Final[tuple[SourceRouteKind, ...]] = (
    SourceRouteKind.TACTICIAN,
    SourceRouteKind.VECTOR,
    SourceRouteKind.KNOWLEDGE_GRAPH,
    SourceRouteKind.LLM,
)
ORDERED_SOURCE_ROUTES: Final[tuple[SourceRouteKind, ...]] = (
    *DETERMINISTIC_SOURCE_ROUTES,
    *NOMINATING_SOURCE_ROUTES,
)

_AUTHORITY_BOOLEAN_CLAIMS: Final[frozenset[str]] = frozenset(
    {
        "admitted",
        "authoritative",
        "complete",
        "kernel_checked",
        "proof_success",
        "semantic_authority",
        "trusted",
        "verified",
    }
)


class ProgramWorldTacticianError(ValueError):
    """Program-world Tactician inputs cannot produce a trustworthy plan."""


class ProgramWorldTacticianBoundsError(ProgramWorldTacticianError):
    """A compilation exceeded a deterministic compactness bound."""


class ProgramWorldTacticianAuthorityError(ProgramWorldTacticianError):
    """A nominating or stale source attempted to claim proof authority."""


class ProgramWorldCompilationDisposition(str, Enum):
    """Closed outcomes for one program-world goal compilation."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    CONFLICT = "conflict"
    ABSTAINED = "abstained"


class ProgramWorldObligationKind(str, Enum):
    """Closed families of compiled program-world proof obligations."""

    TRANSITION = "transition"
    REPAIR = "repair"
    CONSISTENCY = "consistency"
    COUNTEREXAMPLE = "counterexample"
    RESIDUAL = "residual"


class ProgramWorldNominationKind(str, Enum):
    """Closed nominating sources.  Never axioms or proof authority."""

    MODEL = "model"
    VECTOR = "vector"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    TACTICIAN = "tactician"
    HISTORY = "history"


def _text(value: Any, field_name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise ProgramWorldTacticianError(f"{field_name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise ProgramWorldTacticianError(f"{field_name} must not be empty")
    if len(text) > MAX_TEXT_CHARS:
        raise ProgramWorldTacticianBoundsError(f"{field_name} exceeds its character bound")
    return text


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ProgramWorldTacticianError(f"{field_name} must be a boolean")
    return value


def _compact_id(value: str, *, prefix: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}:{digest}"


def _roots(value: Any) -> ProgramLogicAuthorityRoots:
    if isinstance(value, ProgramLogicAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return (
            ProgramLogicAuthorityRoots.from_dict(value)
            if "schema" in value
            else ProgramLogicAuthorityRoots(**dict(value))
        )
    raise ProgramWorldTacticianError("roots must be ProgramLogicAuthorityRoots")


def _assert_current_tree(
    roots: ProgramLogicAuthorityRoots, expected_tree_id: str | None
) -> None:
    expected = _text(expected_tree_id or "", "expected_tree_id", required=False)
    if expected and expected != roots.tree_id:
        raise ProgramWorldTacticianAuthorityError(
            "compilation roots are not bound to the current tree"
        )


def _reject_authority_claims(payload: Mapping[str, Any], *, field_name: str) -> None:
    for key, value in payload.items():
        marker = str(key).strip().lower()
        if marker in _AUTHORITY_BOOLEAN_CLAIMS and value is True:
            raise ProgramWorldTacticianAuthorityError(
                f"{field_name} cannot claim {marker}"
            )
        if isinstance(value, Mapping):
            _reject_authority_claims(value, field_name=field_name)


def _facet(goal_id: str, kind: LogicFacetKind, subject: str) -> LogicFacetRef:
    return LogicFacetRef(
        facet_id=_compact_id(f"{goal_id}:{kind.value}:{subject}", prefix="facet"),
        kind=kind,
        subject_symbol_id=subject if " " not in subject else _compact_id(subject, prefix="sym"),
    )


@dataclass(frozen=True)
class ProgramWorldObligation(CanonicalContract):
    """One finite compiled transition/repair/consistency obligation."""

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_OBLIGATION_SCHEMA

    obligation_id: str
    kind: ProgramWorldObligationKind
    goal_id: str
    subject_cid: str
    evidence_cid: str
    complete: bool = True
    unknown: bool = False
    unavailable_dimensions: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "obligation_id", _text(self.obligation_id, "obligation_id")
        )
        kind = (
            self.kind
            if isinstance(self.kind, ProgramWorldObligationKind)
            else ProgramWorldObligationKind(str(self.kind))
        )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "goal_id", _text(self.goal_id, "goal_id"))
        object.__setattr__(self, "subject_cid", _text(self.subject_cid, "subject_cid"))
        object.__setattr__(self, "evidence_cid", _text(self.evidence_cid, "evidence_cid"))
        object.__setattr__(self, "complete", _bool(self.complete, "complete"))
        object.__setattr__(self, "unknown", _bool(self.unknown, "unknown"))
        dims = tuple(
            sorted(
                {
                    _text(item, "unavailable_dimensions")
                    for item in (self.unavailable_dimensions or ())
                }
            )
        )
        object.__setattr__(self, "unavailable_dimensions", dims)
        meta = dict(self.metadata or {})
        _reject_authority_claims(meta, field_name="obligation.metadata")
        object.__setattr__(self, "metadata", MappingProxyType(meta))
        if self.unknown and self.complete:
            raise ProgramWorldTacticianError(
                "unknown obligations cannot be marked complete"
            )
        if self.kind is ProgramWorldObligationKind.RESIDUAL and self.complete:
            raise ProgramWorldTacticianError("residual obligations cannot be complete")

    def _payload(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "kind": self.kind.value,
            "goal_id": self.goal_id,
            "subject_cid": self.subject_cid,
            "evidence_cid": self.evidence_cid,
            "complete": self.complete,
            "unknown": self.unknown,
            "unavailable_dimensions": list(self.unavailable_dimensions),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProgramWorldPremiseCorpus(CanonicalContract):
    """Current-tree content-addressed premise corpus for program-world search."""

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_PREMISE_CORPUS_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_WORLD_PREMISE_CORPUS_INTERFACE

    inner: ProgramLogicPremiseCorpus
    successor_plan_cid: str = ""
    nominated_premise_ids: tuple[str, ...] = ()
    axiomatic_premise_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.inner, ProgramLogicPremiseCorpus):
            raise ProgramWorldTacticianError(
                "inner must be a ProgramLogicPremiseCorpus"
            )
        object.__setattr__(
            self,
            "successor_plan_cid",
            _text(self.successor_plan_cid, "successor_plan_cid", required=False),
        )
        nominated = tuple(
            _text(item, "nominated_premise_ids")
            for item in (self.nominated_premise_ids or ())
        )
        axiomatic = tuple(
            _text(item, "axiomatic_premise_ids")
            for item in (self.axiomatic_premise_ids or ())
        )
        if set(nominated) & set(axiomatic):
            raise ProgramWorldTacticianAuthorityError(
                "nominated premises cannot also be axiomatic"
            )
        live = {item.premise_id for item in self.inner.premises}
        extra = (set(nominated) | set(axiomatic)) - live
        if extra:
            raise ProgramWorldTacticianError(
                "corpus premise ids must exist in the inner corpus"
            )
        for premise in self.inner.premises:
            if (
                premise.premise_id in axiomatic
                and premise.authority is PremiseAuthority.HYPOTHESIS
            ):
                raise ProgramWorldTacticianAuthorityError(
                    "hypothesis premises cannot be axiomatic"
                )
            if premise.semantic_authority:
                raise ProgramWorldTacticianAuthorityError(
                    "premises cannot claim semantic authority"
                )
        object.__setattr__(self, "nominated_premise_ids", nominated)
        object.__setattr__(self, "axiomatic_premise_ids", axiomatic)

    @property
    def roots(self) -> ProgramLogicAuthorityRoots:
        return self.inner.roots

    @property
    def corpus_cid(self) -> str:
        return self.inner.corpus_id

    @property
    def premises(self) -> tuple[ProgramLogicPremise, ...]:
        return self.inner.premises

    @property
    def consistency_disposition(self) -> ConsistencyDisposition:
        return self.inner.consistency_disposition

    @property
    def conflicting(self) -> bool:
        return self.inner.consistency_disposition in {
            ConsistencyDisposition.LOGICAL_CONFLICT_PROVED,
            ConsistencyDisposition.SUSPECTED_AUTHORITATIVE_CONTRADICTION,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "inner": self.inner.to_dict(),
            "successor_plan_cid": self.successor_plan_cid,
            "nominated_premise_ids": list(self.nominated_premise_ids),
            "axiomatic_premise_ids": list(self.axiomatic_premise_ids),
            "corpus_cid": self.corpus_cid,
        }


@dataclass(frozen=True)
class ProgramWorldGoalCompilation(CanonicalContract):
    """Finite obligation inventory, premise corpus, and advisory tactic plan."""

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_GOAL_COMPILATION_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_WORLD_TACTICIAN_INTERFACE

    roots: ProgramLogicAuthorityRoots
    compilation_id: str
    disposition: ProgramWorldCompilationDisposition
    goals: tuple[ProgramLogicGoal, ...]
    obligations: tuple[ProgramWorldObligation, ...]
    corpus: ProgramWorldPremiseCorpus
    tactic_plan: TacticianSearchPlan
    residual_goal_ids: tuple[str, ...] = ()
    unsupported_goal_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "compilation_id", _text(self.compilation_id, "compilation_id")
        )
        disposition = (
            self.disposition
            if isinstance(self.disposition, ProgramWorldCompilationDisposition)
            else ProgramWorldCompilationDisposition(str(self.disposition))
        )
        object.__setattr__(self, "disposition", disposition)
        goals = tuple(self.goals)
        if len(goals) > MAX_GOALS:
            raise ProgramWorldTacticianBoundsError("goals exceed compilation bound")
        if not all(isinstance(item, ProgramLogicGoal) for item in goals):
            raise ProgramWorldTacticianError("goals must be ProgramLogicGoal values")
        goal_ids = [item.goal_id for item in goals]
        if len(goal_ids) != len(set(goal_ids)):
            raise ProgramWorldTacticianError("goals must have unique goal_ids")
        for goal in goals:
            if goal.roots.content_id != self.roots.content_id:
                raise ProgramWorldTacticianAuthorityError(
                    "compiled goals must bind the current authority roots"
                )
            if goal.proof_status in {
                ProofStatus.KERNEL_VERIFIED,
                ProofStatus.VALIDATED_REFUTED,
            }:
                raise ProgramWorldTacticianAuthorityError(
                    "Tactician compilation cannot mint verified or refuted proof status"
                )
        object.__setattr__(self, "goals", tuple(sorted(goals, key=lambda item: item.goal_id)))
        obligations = tuple(self.obligations)
        if len(obligations) > MAX_OBLIGATIONS:
            raise ProgramWorldTacticianBoundsError(
                "obligations exceed compilation bound"
            )
        if not all(isinstance(item, ProgramWorldObligation) for item in obligations):
            raise ProgramWorldTacticianError(
                "obligations must be ProgramWorldObligation values"
            )
        object.__setattr__(
            self,
            "obligations",
            tuple(sorted(obligations, key=lambda item: item.obligation_id)),
        )
        if not isinstance(self.corpus, ProgramWorldPremiseCorpus):
            raise ProgramWorldTacticianError("corpus must be ProgramWorldPremiseCorpus")
        if self.corpus.roots.content_id != self.roots.content_id:
            raise ProgramWorldTacticianAuthorityError(
                "premise corpus roots must match compilation roots"
            )
        if not isinstance(self.tactic_plan, TacticianSearchPlan):
            raise ProgramWorldTacticianError("tactic_plan must be TacticianSearchPlan")
        if self.tactic_plan.semantic_authority:
            raise ProgramWorldTacticianAuthorityError(
                "tactic plans cannot claim semantic authority"
            )
        if self.tactic_plan.roots.content_id != self.roots.content_id:
            raise ProgramWorldTacticianAuthorityError(
                "tactic plan roots must match compilation roots"
            )
        object.__setattr__(
            self,
            "residual_goal_ids",
            tuple(_text(item, "residual_goal_ids") for item in (self.residual_goal_ids or ())),
        )
        object.__setattr__(
            self,
            "unsupported_goal_ids",
            tuple(
                _text(item, "unsupported_goal_ids")
                for item in (self.unsupported_goal_ids or ())
            ),
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(_text(item, "reason_codes") for item in (self.reason_codes or ())),
        )
        object.__setattr__(self, "producer_id", _text(self.producer_id, "producer_id"))
        if (
            self.disposition is ProgramWorldCompilationDisposition.COMPLETE
            and (self.residual_goal_ids or self.unsupported_goal_ids)
        ):
            raise ProgramWorldTacticianError(
                "complete compilations cannot retain residual or unsupported goals"
            )
        if (
            self.disposition is ProgramWorldCompilationDisposition.CONFLICT
            and "contradiction" not in self.reason_codes
            and "conflict" not in self.reason_codes
        ):
            raise ProgramWorldTacticianError(
                "conflict compilations require a contradiction/conflict reason"
            )

    @property
    def finite(self) -> bool:
        return len(self.goals) <= MAX_GOALS and len(self.obligations) <= MAX_OBLIGATIONS

    @property
    def premise_corpus_cid(self) -> str:
        return self.corpus.corpus_cid

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "evidence": SAWM_TACTICIAN_HAMMER_EVIDENCE,
            "producer_id": self.producer_id,
            "tactician_version": TACTICIAN_VERSION,
            "roots": self.roots.to_dict(),
            "compilation_id": self.compilation_id,
            "disposition": self.disposition.value,
            "goals": [item.to_dict() for item in self.goals],
            "obligations": [item.to_dict() for item in self.obligations],
            "corpus": self.corpus.to_dict(),
            "tactic_plan": self.tactic_plan.to_dict(),
            "residual_goal_ids": list(self.residual_goal_ids),
            "unsupported_goal_ids": list(self.unsupported_goal_ids),
            "reason_codes": list(self.reason_codes),
            "premise_corpus_cid": self.premise_corpus_cid,
            "finite": self.finite,
        }


def _goal_from_successor(
    roots: ProgramLogicAuthorityRoots,
    candidate: SuccessorCandidate,
    *,
    plan_cid: str,
) -> tuple[ProgramLogicGoal, ProgramWorldObligation]:
    subject = candidate.subject_node_cid
    target = candidate.target_node_cid
    unknown = bool(candidate.unknown) or not bool(candidate.complete)
    goal_id = _compact_id(
        f"{plan_cid}:{candidate.candidate_cid}:{target}", prefix="goal"
    )
    dims = tuple(candidate.unavailable_dimensions)
    if unknown:
        family = GoalFamily.COUNTEREXAMPLE if candidate.unknown else GoalFamily.POSITIVE
        disposition = (
            GoalDisposition.UNSUPPORTED if candidate.unknown else GoalDisposition.RESIDUAL
        )
        kind = (
            ProgramWorldObligationKind.COUNTEREXAMPLE
            if candidate.unknown
            else ProgramWorldObligationKind.RESIDUAL
        )
        proof_status = (
            ProofStatus.UNSUPPORTED if candidate.unknown else ProofStatus.UNPROVED
        )
    else:
        family = GoalFamily.POSITIVE
        disposition = GoalDisposition.OPEN
        kind = ProgramWorldObligationKind.TRANSITION
        proof_status = ProofStatus.UNPROVED
    statement = _compact_id(f"stmt:{candidate.candidate_cid}", prefix="stmt")
    goal = ProgramLogicGoal(
        roots=roots,
        goal_id=goal_id,
        family=family,
        disposition=disposition,
        positive_statement_ref=statement,
        negative_target_ref=(
            _compact_id(f"neg:{candidate.candidate_cid}", prefix="neg")
            if family is GoalFamily.NEGATIVE
            else ""
        ),
        counterexample_target_ref=(
            _compact_id(f"cex:{candidate.candidate_cid}", prefix="cex")
            if family is GoalFamily.COUNTEREXAMPLE
            else ""
        ),
        affected_symbol_ids=(subject, target),
        source_refs=(candidate.candidate_cid, plan_cid),
        required_facets=()
        if unknown
        else (_facet(goal_id, LogicFacetKind.STATE, subject),),
        unsupported_facets=(
            (
                LogicFacetRef(
                    facet_id=_compact_id(f"{goal_id}:dynamic", prefix="facet"),
                    kind=LogicFacetKind.STATE,
                    subject_symbol_id=_compact_id(target, prefix="sym"),
                    unsupported=True,
                ),
            )
            if unknown
            else ()
        ),
        assumption_refs=(),
        assumption_authority=SourceAuthorityClass.AUTHORITATIVE,
        proof_status=proof_status,
        logic_family_refs=("logic-family:fol-core",),
        bound_refs=(roots.tree_id, roots.corpus_id, roots.policy_id),
        invalidation_refs=(roots.tree_id, roots.graph_id, plan_cid),
    )
    obligation = ProgramWorldObligation(
        obligation_id=_compact_id(goal_id, prefix="obl"),
        kind=kind,
        goal_id=goal_id,
        subject_cid=subject,
        evidence_cid=candidate.candidate_cid,
        complete=not unknown,
        unknown=unknown,
        unavailable_dimensions=dims,
        metadata={
            "successor_kind": getattr(candidate.kind, "value", str(candidate.kind)),
            "target_node_cid": target,
        },
    )
    return goal, obligation


def _goal_from_repair(
    roots: ProgramLogicAuthorityRoots,
    payload: Mapping[str, Any],
) -> tuple[ProgramLogicGoal, ProgramWorldObligation]:
    subject = _text(payload.get("subject_cid") or payload.get("subject_id"), "repair.subject")
    evidence = _text(
        payload.get("evidence_cid") or payload.get("repair_id") or subject,
        "repair.evidence",
    )
    goal_id = _text(
        payload.get("goal_id") or "",
        "repair.goal_id",
        required=False,
    ) or _compact_id(f"repair:{subject}:{evidence}", prefix="goal")
    statement = _text(
        payload.get("statement_ref") or "",
        "repair.statement_ref",
        required=False,
    ) or _compact_id(evidence, prefix="stmt")
    unsupported = bool(payload.get("unsupported"))
    goal = ProgramLogicGoal(
        roots=roots,
        goal_id=goal_id,
        family=GoalFamily.BEHAVIOR,
        disposition=(
            GoalDisposition.UNSUPPORTED if unsupported else GoalDisposition.OPEN
        ),
        positive_statement_ref=statement,
        affected_symbol_ids=(subject,),
        source_refs=(evidence,),
        required_facets=(_facet(goal_id, LogicFacetKind.STATE, subject),),
        assumption_authority=SourceAuthorityClass.AUTHORITATIVE,
        proof_status=(
            ProofStatus.UNSUPPORTED if unsupported else ProofStatus.UNPROVED
        ),
        logic_family_refs=("logic-family:fol-core",),
        bound_refs=(roots.tree_id, roots.policy_id),
        invalidation_refs=(roots.tree_id, evidence),
    )
    obligation = ProgramWorldObligation(
        obligation_id=_compact_id(goal_id, prefix="obl"),
        kind=(
            ProgramWorldObligationKind.RESIDUAL
            if unsupported
            else ProgramWorldObligationKind.REPAIR
        ),
        goal_id=goal_id,
        subject_cid=subject,
        evidence_cid=evidence,
        complete=not unsupported,
        unknown=unsupported,
        unavailable_dimensions=tuple(
            str(item) for item in (payload.get("unavailable_dimensions") or ())
        ),
    )
    return goal, obligation


def _decode_goals(
    roots: ProgramLogicAuthorityRoots,
    goals: Sequence[ProgramLogicGoal | Mapping[str, Any]],
) -> list[ProgramLogicGoal]:
    decoded: list[ProgramLogicGoal] = []
    for item in goals:
        if isinstance(item, ProgramLogicGoal):
            goal = item
        elif isinstance(item, Mapping):
            goal = (
                ProgramLogicGoal.from_dict(item)
                if "schema" in item
                else ProgramLogicGoal(**dict(item))
            )
        else:
            raise ProgramWorldTacticianError("goals must be ProgramLogicGoal values")
        if goal.roots.content_id != roots.content_id:
            raise ProgramWorldTacticianAuthorityError(
                "supplied goals must bind the current authority roots"
            )
        decoded.append(goal)
    return decoded


def _add_static_premise(
    builder: ProgramLogicPremiseCorpusBuilder,
    *,
    premise_id: str,
    statement_ref: str,
    evidence_cid: str,
    source_class: PremiseSourceClass = PremiseSourceClass.PROGRAM_GRAPH,
) -> None:
    builder.add_static_fact(
        premise_id=premise_id,
        source_class=source_class,
        statement_ref=statement_ref,
        lowering_ref=_compact_id(evidence_cid, prefix="lowering"),
        graph_identity=builder.roots.graph_id,
    )


def _add_expectation_premise(
    builder: ProgramLogicPremiseCorpusBuilder,
    payload: Mapping[str, Any],
) -> None:
    builder.add_expectation(
        premise_id=_text(payload.get("premise_id"), "expectation.premise_id"),
        source_class=payload.get("source_class", PremiseSourceClass.REVIEWED_CONTRACT),
        statement_ref=_text(payload.get("statement_ref"), "expectation.statement_ref"),
        lowering_ref=_text(
            payload.get("lowering_ref")
            or _compact_id(str(payload.get("premise_id")), prefix="lowering"),
            "expectation.lowering_ref",
        ),
        conflicts_with=tuple(
            str(item) for item in (payload.get("conflicts_with") or ())
        ),
        contract_identity=str(payload.get("contract_identity") or ""),
        graph_identity=builder.roots.graph_id,
    )


def _nomination_source_class(kind: ProgramWorldNominationKind) -> PremiseSourceClass:
    mapping = {
        ProgramWorldNominationKind.MODEL: PremiseSourceClass.MODEL_HYPOTHESIS,
        ProgramWorldNominationKind.VECTOR: PremiseSourceClass.VECTOR_ANALOGUE,
        ProgramWorldNominationKind.KNOWLEDGE_GRAPH: PremiseSourceClass.KNOWLEDGE_GRAPH,
        ProgramWorldNominationKind.TACTICIAN: PremiseSourceClass.THEOREM_CORPUS,
        ProgramWorldNominationKind.HISTORY: PremiseSourceClass.HISTORY,
    }
    return mapping[kind]


def _add_nomination(
    builder: ProgramLogicPremiseCorpusBuilder,
    payload: Mapping[str, Any],
) -> str:
    _reject_authority_claims(payload, field_name="nomination")
    kind_raw = payload.get("kind") or payload.get("source") or "model"
    kind = (
        kind_raw
        if isinstance(kind_raw, ProgramWorldNominationKind)
        else ProgramWorldNominationKind(str(kind_raw))
    )
    premise_id = _text(
        payload.get("premise_id") or "",
        "nomination.premise_id",
        required=False,
    ) or _compact_id(str(payload.get("statement_ref") or payload), prefix="nom")
    if payload.get("semantic_authority") or payload.get("expectation_authority"):
        raise ProgramWorldTacticianAuthorityError(
            "nominations cannot claim expectation or semantic authority"
        )
    builder.add_hypothesis(
        premise_id=premise_id,
        source_class=_nomination_source_class(kind),
        statement_ref=_text(
            payload.get("statement_ref") or premise_id, "nomination.statement_ref"
        ),
        lowering_ref=_text(
            payload.get("lowering_ref") or _compact_id(premise_id, prefix="lowering"),
            "nomination.lowering_ref",
        ),
        graph_identity=builder.roots.graph_id,
    )
    return premise_id


def _build_subgoals(
    goals: Sequence[ProgramLogicGoal],
    *,
    axiomatic: Sequence[str],
) -> tuple[LogicSubgoal, ...]:
    subgoals: list[LogicSubgoal] = []
    axiom_ref = axiomatic[0] if axiomatic else "claim:open"
    for goal in goals:
        route = (
            SourceRouteKind.LOCAL_STATIC
            if goal.disposition is GoalDisposition.OPEN
            else SourceRouteKind.GRAPH
        )
        authority = (
            SourceAuthorityClass.AUTHORITATIVE
            if route in DETERMINISTIC_SOURCE_ROUTES
            else SourceAuthorityClass.NOMINATING
        )
        if goal.disposition in {GoalDisposition.UNSUPPORTED, GoalDisposition.RESIDUAL}:
            route = SourceRouteKind.GRAPH
            authority = SourceAuthorityClass.DIAGNOSTIC
        subgoals.append(
            LogicSubgoal(
                subgoal_id=_compact_id(goal.goal_id, prefix="subgoal"),
                goal_id=goal.goal_id,
                disposition=SubgoalDisposition.PENDING,
                claim_ref=axiom_ref if axiom_ref.startswith("claim:") or ":" in axiom_ref else _compact_id(axiom_ref, prefix="claim"),
                source_route=route,
                source_authority=authority,
                proof_status=ProofStatus.UNPROVED,
            )
        )
    return tuple(subgoals)


def _rank_and_select(
    corpus: ProgramLogicPremiseCorpus,
    *,
    top_k: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Rank axiomatic premises; nominated hypotheses stay excluded."""

    scored: list[tuple[int, str]] = []
    nominated: list[str] = []
    for index, premise in enumerate(corpus.premises):
        if premise.authority is PremiseAuthority.HYPOTHESIS:
            nominated.append(premise.premise_id)
            continue
        if premise.authority is PremiseAuthority.EXPECTATION:
            score = 1_000_000 - index
        else:
            score = 500_000 - index
        scored.append((max(score, 0), premise.premise_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = tuple(item_id for _score, item_id in scored[:top_k])
    return selected, tuple(nominated)


class ProgramWorldTactician:
    """Compile finite program-world goals and an advisory Tactician plan."""

    interface: ClassVar[str] = PROGRAM_WORLD_TACTICIAN_INTERFACE
    evidence: ClassVar[str] = SAWM_TACTICIAN_HAMMER_EVIDENCE
    version: ClassVar[str] = TACTICIAN_VERSION

    def __init__(
        self,
        roots: ProgramLogicAuthorityRoots | Mapping[str, Any],
        *,
        expected_tree_id: str | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.roots = _roots(roots)
        _assert_current_tree(self.roots, expected_tree_id)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ProgramWorldTacticianError("top_k must be a positive integer")
        self.top_k = min(int(top_k), MAX_NOMINATIONS)
        self.expected_tree_id = expected_tree_id or self.roots.tree_id

    def compile(
        self,
        *,
        successor_plan: StaticSuccessorPlan | Mapping[str, Any] | None = None,
        goals: Sequence[ProgramLogicGoal | Mapping[str, Any]] = (),
        repair_residuals: Sequence[Mapping[str, Any]] = (),
        reviewed_premises: Sequence[Mapping[str, Any]] = (),
        static_premises: Sequence[Mapping[str, Any]] = (),
        nominations: Sequence[Mapping[str, Any]] = (),
        previous_corpus: ProgramLogicPremiseCorpus | Mapping[str, Any] | None = None,
        logic_family_refs: Sequence[str] = ("logic-family:fol-core",),
        expected_tree_id: str | None = None,
    ) -> ProgramWorldGoalCompilation:
        """Compile finite obligations and a content-addressed Tactician plan."""

        _assert_current_tree(self.roots, expected_tree_id or self.expected_tree_id)
        if len(nominations) > MAX_NOMINATIONS:
            raise ProgramWorldTacticianBoundsError("nominations exceed their bound")

        compiled_goals: list[ProgramLogicGoal] = _decode_goals(self.roots, goals)
        obligations: list[ProgramWorldObligation] = []
        plan_cid = ""
        if successor_plan is not None:
            plan = (
                successor_plan
                if isinstance(successor_plan, StaticSuccessorPlan)
                else StaticSuccessorPlan(**dict(successor_plan))
            )
            if plan.graph_cid and plan.graph_cid != self.roots.graph_id:
                raise ProgramWorldTacticianAuthorityError(
                    "successor plan graph is not the current authority graph"
                )
            plan_cid = plan.plan_cid
            for candidate in plan.candidates:
                goal, obligation = _goal_from_successor(
                    self.roots, candidate, plan_cid=plan_cid
                )
                compiled_goals.append(goal)
                obligations.append(obligation)

        for residual in repair_residuals:
            if not isinstance(residual, Mapping):
                raise ProgramWorldTacticianError("repair residuals must be objects")
            _reject_authority_claims(residual, field_name="repair_residual")
            goal, obligation = _goal_from_repair(self.roots, residual)
            compiled_goals.append(goal)
            obligations.append(obligation)

        unique_goals: dict[str, ProgramLogicGoal] = {}
        for goal in compiled_goals:
            existing = unique_goals.get(goal.goal_id)
            if existing is None:
                unique_goals[goal.goal_id] = goal
            elif existing.content_id != goal.content_id:
                raise ProgramWorldTacticianError(
                    "duplicate goal_id with conflicting content identity"
                )
        compiled_goals = list(unique_goals.values())

        builder = ProgramLogicPremiseCorpusBuilder(self.roots)
        if previous_corpus is not None:
            builder.with_previous(previous_corpus)

        for payload in reviewed_premises:
            if not isinstance(payload, Mapping):
                raise ProgramWorldTacticianError("reviewed premises must be objects")
            _add_expectation_premise(builder, payload)
        for payload in static_premises:
            if not isinstance(payload, Mapping):
                raise ProgramWorldTacticianError("static premises must be objects")
            _add_static_premise(
                builder,
                premise_id=_text(payload.get("premise_id"), "static.premise_id"),
                statement_ref=_text(
                    payload.get("statement_ref"), "static.statement_ref"
                ),
                evidence_cid=_text(
                    payload.get("evidence_cid") or payload.get("premise_id"),
                    "static.evidence_cid",
                ),
                source_class=payload.get(
                    "source_class", PremiseSourceClass.PROGRAM_GRAPH
                ),
            )
        if successor_plan is not None:
            plan = (
                successor_plan
                if isinstance(successor_plan, StaticSuccessorPlan)
                else StaticSuccessorPlan(**dict(successor_plan))
            )
            for candidate in plan.candidates:
                if candidate.unknown:
                    continue
                _add_static_premise(
                    builder,
                    premise_id=_compact_id(candidate.candidate_cid, prefix="premise"),
                    statement_ref=candidate.candidate_cid,
                    evidence_cid=candidate.candidate_cid,
                )

        nominated_ids: list[str] = []
        for payload in nominations:
            if not isinstance(payload, Mapping):
                raise ProgramWorldTacticianError("nominations must be objects")
            nominated_ids.append(_add_nomination(builder, payload))

        inner = builder.build()
        if len(inner.premises) > MAX_PREMISES:
            raise ProgramWorldTacticianBoundsError("premise corpus exceeds its bound")

        selected, excluded = _rank_and_select(inner, top_k=self.top_k)
        axiomatic = tuple(
            item.premise_id
            for item in inner.premises
            if item.authority is not PremiseAuthority.HYPOTHESIS
        )
        nominated = tuple(
            item.premise_id
            for item in inner.premises
            if item.authority is PremiseAuthority.HYPOTHESIS
        )
        corpus = ProgramWorldPremiseCorpus(
            inner=inner,
            successor_plan_cid=plan_cid,
            nominated_premise_ids=nominated,
            axiomatic_premise_ids=axiomatic,
        )

        residual_ids = tuple(
            goal.goal_id
            for goal in compiled_goals
            if goal.disposition is GoalDisposition.RESIDUAL
        )
        unsupported_ids = tuple(
            goal.goal_id
            for goal in compiled_goals
            if goal.disposition is GoalDisposition.UNSUPPORTED
        )
        reason_codes: list[str] = []
        if corpus.conflicting or inner.consistency_obligations:
            reason_codes.extend(("contradiction", "conflict"))
        if inner.consistency_obligations:
            reason_codes.append("consistency_obligation")
        if residual_ids:
            reason_codes.append("residual_frontier")
        if unsupported_ids:
            reason_codes.append("unsupported_logic")
        if nominated:
            reason_codes.append("nominations_non_axiomatic")
        if not compiled_goals:
            reason_codes.append("empty_obligation_inventory")

        if corpus.conflicting or inner.consistency_obligations:
            disposition = ProgramWorldCompilationDisposition.CONFLICT
        elif not compiled_goals:
            disposition = ProgramWorldCompilationDisposition.ABSTAINED
        elif residual_ids or unsupported_ids:
            disposition = ProgramWorldCompilationDisposition.PARTIAL
        else:
            disposition = ProgramWorldCompilationDisposition.COMPLETE

        families = tuple(
            _text(item, "logic_family_refs")
            for item in (logic_family_refs or ("logic-family:fol-core",))
        )
        invalidation = (
            self.roots.tree_id,
            self.roots.corpus_id,
            self.roots.policy_id,
            self.roots.environment_id,
            corpus.corpus_cid,
        )
        tactic_plan = TacticianSearchPlan(
            roots=self.roots,
            plan_id=_compact_id(
                f"{self.roots.tree_id}:{corpus.corpus_cid}:{','.join(sorted(unique_goals))}",
                prefix="plan",
            ),
            goal_ids=tuple(item.goal_id for item in compiled_goals)
            or ("goal:empty-inventory",),
            ordered_source_routes=ORDERED_SOURCE_ROUTES,
            query_refs=tuple(item.obligation_id for item in obligations),
            selected_premise_ids=selected,
            excluded_premise_ids=excluded or nominated,
            exclusion_rationale_refs=(
                ("rationale:nominating-not-axiom",) if nominated else ()
            ),
            subgoals=_build_subgoals(compiled_goals, axiomatic=axiomatic)
            if compiled_goals
            else (
                LogicSubgoal(
                    subgoal_id="subgoal:empty-inventory",
                    goal_id="goal:empty-inventory",
                    disposition=SubgoalDisposition.ABSTAINED,
                    claim_ref="claim:empty-inventory",
                    source_route=SourceRouteKind.LOCAL_STATIC,
                    source_authority=SourceAuthorityClass.DIAGNOSTIC,
                    proof_status=ProofStatus.INCONCLUSIVE,
                ),
            ),
            planned_logic_family_refs=families,
            stop_policy_ref="policy:stop-on-kernel-or-replay",
            escalation_policy_ref="policy:no-model-proof-authority",
            abstention_policy_ref="policy:contradiction-abstains",
            resource_policy_ref="policy:bounded-hammer",
            planner_id=PLANNER_ID,
            model_id="model:none",
            config_id="config:program-world-tactician@1",
            semantic_authority=False,
            invalidation_refs=invalidation,
        )
        if not compiled_goals:
            empty_goal = ProgramLogicGoal(
                roots=self.roots,
                goal_id="goal:empty-inventory",
                family=GoalFamily.CONSISTENCY,
                disposition=GoalDisposition.ABSTAINED,
                positive_statement_ref="stmt:empty-inventory",
                assumption_authority=SourceAuthorityClass.NONE,
                proof_status=ProofStatus.INCONCLUSIVE,
                bound_refs=(self.roots.tree_id,),
                invalidation_refs=(self.roots.tree_id,),
            )
            compiled_goals = [empty_goal]

        compilation_id = content_identity(
            {
                "roots": self.roots.content_id,
                "goals": [item.goal_id for item in compiled_goals],
                "corpus": corpus.corpus_cid,
                "plan": tactic_plan.plan_id,
                "disposition": disposition.value,
            }
        )
        return ProgramWorldGoalCompilation(
            roots=self.roots,
            compilation_id=_compact_id(compilation_id, prefix="compilation"),
            disposition=disposition,
            goals=tuple(compiled_goals),
            obligations=tuple(obligations),
            corpus=corpus,
            tactic_plan=tactic_plan,
            residual_goal_ids=residual_ids,
            unsupported_goal_ids=unsupported_ids,
            reason_codes=tuple(dict.fromkeys(reason_codes)),
        )


def compile_program_world_goals(
    roots: ProgramLogicAuthorityRoots | Mapping[str, Any],
    *,
    successor_plan: StaticSuccessorPlan | Mapping[str, Any] | None = None,
    goals: Sequence[ProgramLogicGoal | Mapping[str, Any]] = (),
    repair_residuals: Sequence[Mapping[str, Any]] = (),
    reviewed_premises: Sequence[Mapping[str, Any]] = (),
    static_premises: Sequence[Mapping[str, Any]] = (),
    nominations: Sequence[Mapping[str, Any]] = (),
    previous_corpus: ProgramLogicPremiseCorpus | Mapping[str, Any] | None = None,
    logic_family_refs: Sequence[str] = ("logic-family:fol-core",),
    expected_tree_id: str | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> ProgramWorldGoalCompilation:
    """Compile finite program-world goals and an advisory Tactician plan."""

    tactician = ProgramWorldTactician(
        roots, expected_tree_id=expected_tree_id, top_k=top_k
    )
    return tactician.compile(
        successor_plan=successor_plan,
        goals=goals,
        repair_residuals=repair_residuals,
        reviewed_premises=reviewed_premises,
        static_premises=static_premises,
        nominations=nominations,
        previous_corpus=previous_corpus,
        logic_family_refs=logic_family_refs,
        expected_tree_id=expected_tree_id,
    )


__all__ = [
    "DETERMINISTIC_SOURCE_ROUTES",
    "NOMINATING_SOURCE_ROUTES",
    "ORDERED_SOURCE_ROUTES",
    "PROGRAM_WORLD_PREMISE_CORPUS_INTERFACE",
    "PROGRAM_WORLD_TACTICIAN_INTERFACE",
    "SAWM_TACTICIAN_HAMMER_EVIDENCE",
    "ProgramWorldCompilationDisposition",
    "ProgramWorldGoalCompilation",
    "ProgramWorldNominationKind",
    "ProgramWorldObligation",
    "ProgramWorldObligationKind",
    "ProgramWorldPremiseCorpus",
    "ProgramWorldTactician",
    "ProgramWorldTacticianAuthorityError",
    "ProgramWorldTacticianBoundsError",
    "ProgramWorldTacticianError",
    "compile_program_world_goals",
]
