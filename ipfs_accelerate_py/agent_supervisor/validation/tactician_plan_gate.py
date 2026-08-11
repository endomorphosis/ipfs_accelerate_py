"""Fail-closed Tactician plan security gate (LPR-010).

Admits only current, acyclic, source-authorized, complete, and
policy-bounded ``TacticianSearchPlan@1`` records for later obligation
lowering.  This module never runs solvers, never mutates plan/corpus
inputs, never grants write authority, and never claims semantic
authority.

Hard failures reject the plan.  Structural conflicts and unknown
consistency abstain.  Suspected logical contradiction emits a
consistency subgoal and permits *only* that consistency proof plan to
proceed; semantic prediction admission stays blocked until a separately
validated unsat-core / native conflict receipt (LPR-012) is supplied.
Learned, vector, and model scores cannot override a hard failure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ..analysis.program_logic_prediction_contracts import (
    GoalDisposition,
    GoalFamily,
    HypothesisDisposition,
    LogicGap,
    LogicHypothesis,
    LogicSubgoal,
    ProgramLogicAuthorityRoots,
    ProgramLogicGoal,
    ProgramLogicPredictionError,
    SourceAuthorityClass,
    SourceRouteKind,
    TacticianSearchPlan,
)
from ..analysis.program_logic_premise_corpus import (
    ConsistencyDisposition,
    PremiseAuthority,
    PremiseConflictReceipt,
    PremiseSourceClass,
    ProgramLogicPremise,
    ProgramLogicPremiseCorpus,
)
from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)


# ---------------------------------------------------------------------------
# Schemas / bounds
# ---------------------------------------------------------------------------

TACTICIAN_PLAN_GATE_INTERFACE: Final[str] = "TacticianPlanGate@1"
TACTICIAN_PLAN_GATE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/tactician-plan-gate-receipt@1"
)
TACTICIAN_PLAN_GATE_BOUNDS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/tactician-plan-gate-bounds@1"
)
GOAL_DISPOSITION_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/tactician-plan-goal-disposition@1"
)
CONSISTENCY_SUBGOAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/tactician-plan-consistency-subgoal@1"
)

PRODUCER_ID: Final[str] = "tactician-plan-gate@1"
CONTRACT_VERSION: Final[int] = 1

MAX_GATE_RECEIPT_BYTES: Final[int] = 131_072
MAX_GOALS: Final[int] = 256
MAX_HYPOTHESES: Final[int] = 512
MAX_REJECTION_REASONS: Final[int] = 64
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_REF_BYTES: Final[int] = 512
MAX_REFERENCE_COUNT: Final[int] = 256

DEFAULT_MAX_SUBGOALS: Final[int] = 64
DEFAULT_MAX_PREMISES: Final[int] = 256
DEFAULT_MAX_ROUTES: Final[int] = 64
DEFAULT_MAX_QUERIES: Final[int] = 64
DEFAULT_MAX_SCORE_MILLIPERCENT: Final[int] = 100_000

_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "source",
        "source_body",
        "source_text",
        "source_code",
        "contents",
        "content",
        "snippet",
        "code",
        "file_text",
        "raw",
        "raw_text",
        "ast",
        "ast_body",
        "embedding",
        "query_vector",
        "model_output",
        "completion",
        "prompt",
        "prompt_body",
        "transcript",
        "theorem_text",
        "proof_script",
    }
)

_SECRET_KEY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "private_key",
        "secret",
        "secret_key",
        "access_token",
        "refresh_token",
        "bearer",
        "credential",
        "ssh_key",
        "client_secret",
        "session_token",
        "cookie",
        "token",
        "passwd",
        "private_witness",
        "private_premise",
    }
)

_SECRET_VALUE_MARKERS: Final[tuple[str, ...]] = (
    "api_key=",
    "apikey=",
    "password=",
    "secret=",
    "private_key",
    "authorization:",
    "bearer ",
    "-----begin",
    "client_secret=",
)

_PROMPT_DIRECTIVE_MARKERS: Final[tuple[str, ...]] = (
    "ignore previous",
    "ignore all previous",
    "system:",
    "you are now",
    "disregard",
    "override policy",
    "jailbreak",
    "as an ai",
    "do anything now",
    "prompt:",
    "instruction:",
    "# policy",
    "treat this as policy",
    "use this as policy",
    "act as policy",
)

_PROMPT_POLICY_REF_PREFIXES: Final[tuple[str, ...]] = (
    "prompt:",
    "llm:",
    "completion:",
    "chat:",
    "directive:",
    "instruction:",
    "freeform:",
    "nl:",
    "natural-language:",
)

_AUTHORITATIVE_ROUTES: Final[frozenset[SourceRouteKind]] = frozenset(
    {
        SourceRouteKind.LOCAL_STATIC,
        SourceRouteKind.REVIEWED_CONTRACT,
        SourceRouteKind.NORMATIVE_SPEC,
        SourceRouteKind.REVIEWED_TEST,
        SourceRouteKind.DATAFLOW,
        SourceRouteKind.GRAPH,
    }
)

_NOMINATING_ROUTES: Final[frozenset[SourceRouteKind]] = frozenset(
    {
        SourceRouteKind.VECTOR,
        SourceRouteKind.KNOWLEDGE_GRAPH,
        SourceRouteKind.TACTICIAN,
        SourceRouteKind.LLM,
        SourceRouteKind.SOLVER,
        SourceRouteKind.RUNTIME_WITNESS,
        SourceRouteKind.HISTORY,
    }
)

_HYPOTHESIS_SOURCE_CLASSES: Final[frozenset[PremiseSourceClass]] = frozenset(
    {
        PremiseSourceClass.CANDIDATE_IMPLEMENTATION,
        PremiseSourceClass.COMMENT,
        PremiseSourceClass.RUNTIME_WITNESS,
        PremiseSourceClass.HISTORY,
        PremiseSourceClass.VECTOR_ANALOGUE,
        PremiseSourceClass.KNOWLEDGE_GRAPH,
        PremiseSourceClass.MODEL_HYPOTHESIS,
        PremiseSourceClass.GIT_LINEAGE,
    }
)

_GOAL_DISPOSITIONS_REQUIRING_COVERAGE: Final[frozenset[GoalDisposition]] = frozenset(
    {
        GoalDisposition.OPEN,
        GoalDisposition.PLANNED,
        GoalDisposition.ADMITTED,
        GoalDisposition.RESIDUAL,
    }
)

_RESIDUAL_GOAL_DISPOSITIONS: Final[frozenset[GoalDisposition]] = frozenset(
    {
        GoalDisposition.RESIDUAL,
        GoalDisposition.UNSUPPORTED,
    }
)

_SCORE_OVERRIDE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "score_overrides_hard_gate",
        "learned_score_admits",
        "vector_score_admits",
        "model_score_admits",
        "score_establishes_admission",
        "hard_gate_passed_by_score",
        "ranking_admits",
        "similarity_admits",
    }
)


# ---------------------------------------------------------------------------
# Errors / closed taxonomies
# ---------------------------------------------------------------------------


class TacticianPlanGateError(ContractValidationError):
    """Raised when a plan cannot be safely lowered."""


class TacticianPlanGateBindingError(TacticianPlanGateError):
    """Inputs are cross-root, forged, or incompletely bound."""


class TacticianPlanGateBoundsError(TacticianPlanGateError):
    """A declared policy budget would be exceeded or escalated."""


class TacticianPlanRejectionReason(str, Enum):
    """Stable rejection / abstention reason codes (plan §6 + LPR-010)."""

    MALFORMED_INPUT = "malformed_input"
    FORGED_IDENTITY = "forged_identity"
    CHANGED_ROOTS = "changed_roots"
    CROSS_ROOT_BINDING = "cross_root_binding"
    OMITTED_GOAL_DISPOSITION = "omitted_goal_disposition"
    OMITTED_RESIDUAL_DISPOSITION = "omitted_residual_disposition"
    OMITTED_FACET = "omitted_facet"
    TACTICIAN_PLAN_INVALID = "tactician_plan_invalid"
    TACTICIAN_PLAN_CYCLIC = "tactician_plan_cyclic"
    DUPLICATED_SUBGOAL_IDENTITY = "duplicated_subgoal_identity"
    SELF_AUTHORING_CANDIDATE_PREMISE = "self_authoring_candidate_premise"
    PREMISE_SELF_REFERENTIAL = "premise_self_referential"
    UNAUTHORIZED_SOURCE = "unauthorized_source"
    PREMISE_UNTRUSTED = "premise_untrusted"
    PROMPT_DIRECTIVE_AS_POLICY = "prompt_directive_as_policy"
    SECRET_OR_BODY_LEAKAGE = "secret_or_body_leakage"
    FORGED_EXCLUSION = "forged_exclusion"
    BUDGET_ESCALATION = "budget_escalation"
    SEMANTIC_AUTHORITY_CLAIM = "semantic_authority_claim"
    WRITE_AUTHORITY_CLAIM = "write_authority_claim"
    STRUCTURAL_CONFLICT = "structural_conflict"
    PREMISE_CORPUS_INCONSISTENT = "premise_corpus_inconsistent"
    SUSPECTED_LOGICAL_CONTRADICTION = "suspected_logical_contradiction"
    UNKNOWN_CONSISTENCY = "unknown_consistency"
    SCORE_OVERRIDE_ATTEMPT = "score_override_attempt"
    PREDICTION_ADMISSION_BLOCKED = "prediction_admission_blocked"
    UNAUTHORIZED_PREMISE = "unauthorized_premise"
    HYPOTHESIS_AMBIGUOUS = "hypothesis_ambiguous"
    STALE_EVIDENCE = "stale_evidence"


class TacticianPlanGateDisposition(str, Enum):
    """Closed gate outcomes; only ADMITTED / CONSISTENCY_ONLY may lower."""

    ADMITTED = "admitted"
    ABSTAINED = "abstained"
    REJECTED = "rejected"
    CONSISTENCY_ONLY = "consistency_only"


# Reasons that abstain rather than hard-reject.
_ABSTAIN_REASONS: Final[frozenset[TacticianPlanRejectionReason]] = frozenset(
    {
        TacticianPlanRejectionReason.STRUCTURAL_CONFLICT,
        TacticianPlanRejectionReason.UNKNOWN_CONSISTENCY,
        TacticianPlanRejectionReason.PREMISE_CORPUS_INCONSISTENT,
    }
)

# Soft structural flags that still permit a consistency-only path.
_CONSISTENCY_ONLY_REASONS: Final[frozenset[TacticianPlanRejectionReason]] = frozenset(
    {
        TacticianPlanRejectionReason.SUSPECTED_LOGICAL_CONTRADICTION,
    }
)

# Hard failures that learned/vector/model scores cannot override.
_HARD_FAILURE_REASONS: Final[frozenset[TacticianPlanRejectionReason]] = frozenset(
    {
        TacticianPlanRejectionReason.MALFORMED_INPUT,
        TacticianPlanRejectionReason.FORGED_IDENTITY,
        TacticianPlanRejectionReason.CHANGED_ROOTS,
        TacticianPlanRejectionReason.CROSS_ROOT_BINDING,
        TacticianPlanRejectionReason.OMITTED_GOAL_DISPOSITION,
        TacticianPlanRejectionReason.OMITTED_RESIDUAL_DISPOSITION,
        TacticianPlanRejectionReason.OMITTED_FACET,
        TacticianPlanRejectionReason.TACTICIAN_PLAN_INVALID,
        TacticianPlanRejectionReason.TACTICIAN_PLAN_CYCLIC,
        TacticianPlanRejectionReason.DUPLICATED_SUBGOAL_IDENTITY,
        TacticianPlanRejectionReason.SELF_AUTHORING_CANDIDATE_PREMISE,
        TacticianPlanRejectionReason.PREMISE_SELF_REFERENTIAL,
        TacticianPlanRejectionReason.UNAUTHORIZED_SOURCE,
        TacticianPlanRejectionReason.PREMISE_UNTRUSTED,
        TacticianPlanRejectionReason.PROMPT_DIRECTIVE_AS_POLICY,
        TacticianPlanRejectionReason.SECRET_OR_BODY_LEAKAGE,
        TacticianPlanRejectionReason.FORGED_EXCLUSION,
        TacticianPlanRejectionReason.BUDGET_ESCALATION,
        TacticianPlanRejectionReason.SEMANTIC_AUTHORITY_CLAIM,
        TacticianPlanRejectionReason.WRITE_AUTHORITY_CLAIM,
        TacticianPlanRejectionReason.SCORE_OVERRIDE_ATTEMPT,
        TacticianPlanRejectionReason.UNAUTHORIZED_PREMISE,
        TacticianPlanRejectionReason.STALE_EVIDENCE,
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, field_name: str, *, required: bool = False, limit: int = MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str):
        raise TacticianPlanGateError(f"{field_name} must be a string")
    value = value.strip()
    if required and not value:
        raise TacticianPlanGateError(f"{field_name} is required")
    if len(value.encode("utf-8")) > limit:
        raise TacticianPlanGateBoundsError(f"{field_name} exceeds its byte bound")
    return value


def _identifier(value: Any, field_name: str) -> str:
    value = _text(value, field_name, required=True, limit=MAX_REF_BYTES)
    if any(char.isspace() for char in value):
        raise TacticianPlanGateError(f"{field_name} must be a compact identifier")
    return value


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TacticianPlanGateError(f"{field_name} must be a boolean")
    return value


def _nonneg_int(value: Any, field_name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TacticianPlanGateError(f"{field_name} must be a finite integer")
    if value < 0:
        raise TacticianPlanGateBoundsError(f"{field_name} must be non-negative")
    if maximum is not None and value > maximum:
        raise TacticianPlanGateBoundsError(f"{field_name} exceeds its bound")
    return value


def _ids(
    values: Any,
    field_name: str,
    *,
    required: bool = False,
    limit: int = MAX_REFERENCE_COUNT,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        raw = values
    else:
        raise TacticianPlanGateError(f"{field_name} must be a sequence")
    if len(raw) > limit:
        raise TacticianPlanGateBoundsError(f"{field_name} exceeds its item bound")
    items: list[str] = []
    seen: set[str] = set()
    for item in raw:
        ident = _identifier(item, field_name)
        if preserve_order:
            if ident not in seen:
                seen.add(ident)
                items.append(ident)
        else:
            if ident not in seen:
                seen.add(ident)
                items.append(ident)
    if not preserve_order:
        items = sorted(items)
    if required and not items:
        raise TacticianPlanGateError(f"{field_name} must not be empty")
    return tuple(items)


def _roots(value: Any) -> ProgramLogicAuthorityRoots:
    if isinstance(value, ProgramLogicAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return (
            ProgramLogicAuthorityRoots.from_dict(value)
            if "schema" in value
            else ProgramLogicAuthorityRoots(**value)
        )
    raise TacticianPlanGateError("roots must be ProgramLogicAuthorityRoots")


def _roots_equal(left: ProgramLogicAuthorityRoots, right: ProgramLogicAuthorityRoots) -> bool:
    return left.content_id == right.content_id


def _normalize_key(key: str) -> str:
    return key.lower().replace("-", "_").strip()


def _contains_secret_text(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _SECRET_VALUE_MARKERS)


def _walk_for_body_or_secret(value: Any, *, path: str = "payload") -> str | None:
    """Return a diagnostic path if body/secret material is present."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                return f"{path}.<non-string-key>"
            normalized = _normalize_key(key)
            if normalized in _BODY_MARKERS:
                return f"{path}.{key}"
            if normalized in _SECRET_KEY_MARKERS:
                return f"{path}.{key}"
            if normalized in {
                "semantic_authority",
                "write_authority",
                "proof_authority",
            } and item is True:
                # Authority flags are handled separately; not body leakage.
                continue
            nested = _walk_for_body_or_secret(item, path=f"{path}.{key}")
            if nested:
                return nested
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            nested = _walk_for_body_or_secret(item, path=f"{path}[{index}]")
            if nested:
                return nested
    elif isinstance(value, (bytes, bytearray)):
        return path
    elif isinstance(value, str) and _contains_secret_text(value):
        return path
    return None


def _contains_prompt_directive(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _PROMPT_DIRECTIVE_MARKERS)


def _is_prompt_policy_ref(ref: str) -> bool:
    lowered = ref.lower().strip()
    if any(lowered.startswith(prefix) for prefix in _PROMPT_POLICY_REF_PREFIXES):
        return True
    return _contains_prompt_directive(lowered)


def _decode_goals(values: Any) -> tuple[ProgramLogicGoal, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        raw = values
    else:
        raise TacticianPlanGateError("goals must be a sequence")
    if len(raw) > MAX_GOALS:
        raise TacticianPlanGateBoundsError("goals exceeds its item bound")
    goals: list[ProgramLogicGoal] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, ProgramLogicGoal):
            goal = item
        elif isinstance(item, Mapping):
            goal = (
                ProgramLogicGoal.from_dict(item)
                if "schema" in item
                else ProgramLogicGoal(**item)
            )
        else:
            raise TacticianPlanGateError("goals must contain ProgramLogicGoal values")
        if goal.goal_id in seen:
            raise TacticianPlanGateBindingError(
                f"duplicate goal identity {goal.goal_id!r}"
            )
        seen.add(goal.goal_id)
        goals.append(goal)
    return tuple(goals)


def _decode_hypotheses(values: Any) -> tuple[LogicHypothesis, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        raw = values
    else:
        raise TacticianPlanGateError("candidates must be a sequence")
    if len(raw) > MAX_HYPOTHESES:
        raise TacticianPlanGateBoundsError("candidates exceeds its item bound")
    items: list[LogicHypothesis] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, LogicHypothesis):
            hypothesis = item
        elif isinstance(item, Mapping):
            # Support nomination wrappers that embed a hypothesis.
            if "hypothesis" in item and isinstance(item["hypothesis"], (Mapping, LogicHypothesis)):
                embedded = item["hypothesis"]
                hypothesis = (
                    embedded
                    if isinstance(embedded, LogicHypothesis)
                    else (
                        LogicHypothesis.from_dict(embedded)
                        if "schema" in embedded
                        else LogicHypothesis(**embedded)
                    )
                )
            else:
                hypothesis = (
                    LogicHypothesis.from_dict(item)
                    if "schema" in item
                    else LogicHypothesis(**item)
                )
        else:
            # Nomination objects with .hypothesis attribute.
            embedded = getattr(item, "hypothesis", None)
            if isinstance(embedded, LogicHypothesis):
                hypothesis = embedded
            else:
                raise TacticianPlanGateError(
                    "candidates must contain LogicHypothesis values"
                )
        if hypothesis.hypothesis_id in seen:
            raise TacticianPlanGateBindingError(
                f"duplicate candidate identity {hypothesis.hypothesis_id!r}"
            )
        seen.add(hypothesis.hypothesis_id)
        items.append(hypothesis)
    return tuple(items)


def _decode_gaps(values: Any) -> tuple[LogicGap, ...]:
    if values is None:
        return ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise TacticianPlanGateError("gaps must be a sequence")
    gaps: list[LogicGap] = []
    for item in values:
        if isinstance(item, LogicGap):
            gaps.append(item)
        elif isinstance(item, Mapping):
            gaps.append(
                LogicGap.from_dict(item) if "schema" in item else LogicGap(**item)
            )
        else:
            raise TacticianPlanGateError("gaps must contain LogicGap values")
    return tuple(gaps)


def _decode_corpus(value: Any) -> ProgramLogicPremiseCorpus | None:
    if value is None:
        return None
    if isinstance(value, ProgramLogicPremiseCorpus):
        return value
    if isinstance(value, Mapping):
        return (
            ProgramLogicPremiseCorpus.from_dict(value)
            if "schema" in value
            else ProgramLogicPremiseCorpus(**value)
        )
    raise TacticianPlanGateError("corpus must be ProgramLogicPremiseCorpus")


def _decode_plan(value: Any) -> TacticianSearchPlan:
    if isinstance(value, TacticianSearchPlan):
        return value
    if isinstance(value, Mapping):
        return (
            TacticianSearchPlan.from_dict(value)
            if "schema" in value
            else TacticianSearchPlan(**value)
        )
    raise TacticianPlanGateError("plan must be TacticianSearchPlan")


def _decode_conflict_receipts(values: Any) -> tuple[PremiseConflictReceipt, ...]:
    if values is None:
        return ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise TacticianPlanGateError("conflict_receipts must be a sequence")
    items: list[PremiseConflictReceipt] = []
    for item in values:
        if isinstance(item, PremiseConflictReceipt):
            items.append(item)
        elif isinstance(item, Mapping):
            items.append(
                PremiseConflictReceipt.from_dict(item)
                if "schema" in item
                else PremiseConflictReceipt(**item)
            )
        else:
            raise TacticianPlanGateError(
                "conflict_receipts must contain PremiseConflictReceipt"
            )
    return tuple(items)


def _subgoal_has_cycle(subgoals: Sequence[LogicSubgoal]) -> bool:
    """Detect cycles via depends_on / parent edges (defensive re-check)."""
    by_id = {item.subgoal_id: item for item in subgoals}
    if len(by_id) != len(subgoals):
        return False  # duplicates handled elsewhere
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in visited:
            return False
        if node_id in visiting:
            return True
        visiting.add(node_id)
        node = by_id.get(node_id)
        if node is not None:
            for dep in node.depends_on:
                if dep in by_id and visit(dep):
                    return True
            parent = node.parent_subgoal_id
            if parent and parent in by_id and visit(parent):
                return True
        visiting.remove(node_id)
        visited.add(node_id)
        return False

    return any(visit(item.subgoal_id) for item in subgoals)


def _recompute_identity(record: Any) -> str:
    """Recompute a content identity from a typed contract or mapping payload."""
    if hasattr(record, "content_id") and hasattr(record, "to_dict"):
        # Prefer payload without forged content_id.
        payload = record.to_dict()
        payload.pop("content_id", None)
        payload.pop("receipt_id", None)
        return content_identity(payload)
    if isinstance(record, Mapping):
        payload = dict(record)
        payload.pop("content_id", None)
        payload.pop("receipt_id", None)
        return content_identity(payload)
    raise TacticianPlanGateError("cannot recompute identity for unsupported record")


# ---------------------------------------------------------------------------
# Bounds / policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TacticianPlanGateBounds(CanonicalContract):
    """Hard policy ceilings the plan may not escalate past."""

    SCHEMA: ClassVar[str] = TACTICIAN_PLAN_GATE_BOUNDS_SCHEMA

    max_subgoals: int = DEFAULT_MAX_SUBGOALS
    max_premises: int = DEFAULT_MAX_PREMISES
    max_routes: int = DEFAULT_MAX_ROUTES
    max_queries: int = DEFAULT_MAX_QUERIES
    max_score_millipercent: int = DEFAULT_MAX_SCORE_MILLIPERCENT
    allow_model_hypothesis: bool = False
    allow_approximate_routes: bool = True
    network_allowed: bool = False
    write_allowed: bool = False
    proof_execution_allowed: bool = False
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_subgoals",
            _nonneg_int(self.max_subgoals, "max_subgoals", maximum=MAX_GOALS),
        )
        object.__setattr__(
            self,
            "max_premises",
            _nonneg_int(self.max_premises, "max_premises", maximum=MAX_HYPOTHESES),
        )
        object.__setattr__(
            self,
            "max_routes",
            _nonneg_int(self.max_routes, "max_routes", maximum=MAX_REFERENCE_COUNT),
        )
        object.__setattr__(
            self,
            "max_queries",
            _nonneg_int(self.max_queries, "max_queries", maximum=MAX_REFERENCE_COUNT),
        )
        object.__setattr__(
            self,
            "max_score_millipercent",
            _nonneg_int(
                self.max_score_millipercent,
                "max_score_millipercent",
                maximum=DEFAULT_MAX_SCORE_MILLIPERCENT,
            ),
        )
        for name in (
            "allow_model_hypothesis",
            "allow_approximate_routes",
            "network_allowed",
            "write_allowed",
            "proof_execution_allowed",
            "semantic_authority",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        # Authority-like capability flags remain closed at the gate.
        if self.semantic_authority is not False:
            raise TacticianPlanGateError("gate bounds cannot claim semantic authority")
        if self.write_allowed is not False:
            raise TacticianPlanGateError("gate bounds cannot authorize writes")
        if self.network_allowed is not False:
            raise TacticianPlanGateError("gate bounds cannot authorize network")
        if self.proof_execution_allowed is not False:
            raise TacticianPlanGateError(
                "gate bounds cannot authorize native proof execution"
            )
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "write_allowed", False)
        object.__setattr__(self, "network_allowed", False)
        object.__setattr__(self, "proof_execution_allowed", False)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "max_subgoals": self.max_subgoals,
            "max_premises": self.max_premises,
            "max_routes": self.max_routes,
            "max_queries": self.max_queries,
            "max_score_millipercent": self.max_score_millipercent,
            "allow_model_hypothesis": self.allow_model_hypothesis,
            "allow_approximate_routes": self.allow_approximate_routes,
            "network_allowed": False,
            "write_allowed": False,
            "proof_execution_allowed": False,
            "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TacticianPlanGateBounds":
        fields = (
            "max_subgoals",
            "max_premises",
            "max_routes",
            "max_queries",
            "max_score_millipercent",
            "allow_model_hypothesis",
            "allow_approximate_routes",
            "network_allowed",
            "write_allowed",
            "proof_execution_allowed",
            "semantic_authority",
        )
        values = {name: payload[name] for name in fields if name in payload}
        return cls(**values)


@dataclass(frozen=True)
class GoalDispositionBinding(CanonicalContract):
    """One disposition binding for an original goal or residual."""

    SCHEMA: ClassVar[str] = GOAL_DISPOSITION_BINDING_SCHEMA

    goal_id: str
    disposition: str
    is_residual: bool = False
    subgoal_ids: tuple[str, ...] = ()
    facet_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _identifier(self.goal_id, "goal_id"))
        object.__setattr__(
            self, "disposition", _identifier(self.disposition, "disposition")
        )
        object.__setattr__(self, "is_residual", _bool(self.is_residual, "is_residual"))
        object.__setattr__(
            self, "subgoal_ids", _ids(self.subgoal_ids, "subgoal_ids", preserve_order=True)
        )
        object.__setattr__(
            self, "facet_ids", _ids(self.facet_ids, "facet_ids", preserve_order=True)
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "goal_id": self.goal_id,
            "disposition": self.disposition,
            "is_residual": self.is_residual,
            "subgoal_ids": list(self.subgoal_ids),
            "facet_ids": list(self.facet_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalDispositionBinding":
        return cls(
            goal_id=payload["goal_id"],
            disposition=payload["disposition"],
            is_residual=payload.get("is_residual", False),
            subgoal_ids=tuple(payload.get("subgoal_ids", ())),
            facet_ids=tuple(payload.get("facet_ids", ())),
        )


@dataclass(frozen=True)
class ConsistencySubgoalPlan(CanonicalContract):
    """Bounded consistency-only plan fragment permitted under contradiction."""

    SCHEMA: ClassVar[str] = CONSISTENCY_SUBGOAL_SCHEMA

    subgoal_id: str
    goal_id: str
    premise_ids: tuple[str, ...]
    obligation_ids: tuple[str, ...] = ()
    claim_ref: str = ""
    semantic_prediction_admission_blocked: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "subgoal_id", _identifier(self.subgoal_id, "subgoal_id")
        )
        object.__setattr__(self, "goal_id", _identifier(self.goal_id, "goal_id"))
        object.__setattr__(
            self,
            "premise_ids",
            _ids(self.premise_ids, "premise_ids", required=True, preserve_order=True),
        )
        object.__setattr__(
            self,
            "obligation_ids",
            _ids(self.obligation_ids, "obligation_ids", preserve_order=True),
        )
        object.__setattr__(
            self,
            "claim_ref",
            _text(self.claim_ref, "claim_ref", limit=MAX_REF_BYTES),
        )
        if self.semantic_prediction_admission_blocked is not True:
            raise TacticianPlanGateError(
                "consistency subgoal plans must keep semantic prediction admission blocked"
            )
        object.__setattr__(self, "semantic_prediction_admission_blocked", True)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "subgoal_id": self.subgoal_id,
            "goal_id": self.goal_id,
            "premise_ids": list(self.premise_ids),
            "obligation_ids": list(self.obligation_ids),
            "claim_ref": self.claim_ref,
            "semantic_prediction_admission_blocked": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConsistencySubgoalPlan":
        return cls(
            subgoal_id=payload["subgoal_id"],
            goal_id=payload["goal_id"],
            premise_ids=tuple(payload["premise_ids"]),
            obligation_ids=tuple(payload.get("obligation_ids", ())),
            claim_ref=payload.get("claim_ref", ""),
            semantic_prediction_admission_blocked=payload.get(
                "semantic_prediction_admission_blocked", True
            ),
        )


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TacticianPlanGateReceipt(CanonicalContract):
    """Side-effect-free receipt authorizing plan lowering (or abstention)."""

    SCHEMA: ClassVar[str] = TACTICIAN_PLAN_GATE_RECEIPT_SCHEMA

    roots: ProgramLogicAuthorityRoots
    plan_id: str
    plan_content_id: str
    corpus_content_id: str
    goal_content_ids: tuple[str, ...]
    candidate_content_ids: tuple[str, ...]
    disposition: TacticianPlanGateDisposition
    reasons: tuple[TacticianPlanRejectionReason, ...] = ()
    goal_dispositions: tuple[GoalDispositionBinding, ...] = ()
    permitted_subgoal_ids: tuple[str, ...] = ()
    consistency_subgoal: ConsistencySubgoalPlan | None = None
    semantic_authority: bool = False
    write_authority: bool = False
    semantic_prediction_admission_blocked: bool = True
    scores_cannot_override_hard_failure: bool = True
    producer_id: str = PRODUCER_ID
    bounds: TacticianPlanGateBounds | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(
            self, "plan_content_id", _identifier(self.plan_content_id, "plan_content_id")
        )
        object.__setattr__(
            self,
            "corpus_content_id",
            _text(self.corpus_content_id, "corpus_content_id", limit=MAX_REF_BYTES),
        )
        object.__setattr__(
            self,
            "goal_content_ids",
            _ids(self.goal_content_ids, "goal_content_ids", preserve_order=True),
        )
        object.__setattr__(
            self,
            "candidate_content_ids",
            _ids(
                self.candidate_content_ids,
                "candidate_content_ids",
                preserve_order=True,
            ),
        )
        if isinstance(self.disposition, TacticianPlanGateDisposition):
            disposition = self.disposition
        else:
            disposition = TacticianPlanGateDisposition(str(self.disposition))
        object.__setattr__(self, "disposition", disposition)

        reasons: list[TacticianPlanRejectionReason] = []
        for item in self.reasons or ():
            if isinstance(item, TacticianPlanRejectionReason):
                reasons.append(item)
            else:
                reasons.append(TacticianPlanRejectionReason(str(item)))
        if len(reasons) > MAX_REJECTION_REASONS:
            raise TacticianPlanGateBoundsError("reasons exceeds its item bound")
        object.__setattr__(self, "reasons", tuple(reasons))

        bindings: list[GoalDispositionBinding] = []
        for item in self.goal_dispositions or ():
            if isinstance(item, GoalDispositionBinding):
                bindings.append(item)
            elif isinstance(item, Mapping):
                bindings.append(GoalDispositionBinding.from_dict(item))
            else:
                raise TacticianPlanGateError(
                    "goal_dispositions must contain GoalDispositionBinding"
                )
        object.__setattr__(self, "goal_dispositions", tuple(bindings))

        object.__setattr__(
            self,
            "permitted_subgoal_ids",
            _ids(
                self.permitted_subgoal_ids,
                "permitted_subgoal_ids",
                preserve_order=True,
            ),
        )

        consistency = self.consistency_subgoal
        if consistency is not None and not isinstance(consistency, ConsistencySubgoalPlan):
            if isinstance(consistency, Mapping):
                consistency = ConsistencySubgoalPlan.from_dict(consistency)
            else:
                raise TacticianPlanGateError(
                    "consistency_subgoal must be ConsistencySubgoalPlan"
                )
        object.__setattr__(self, "consistency_subgoal", consistency)

        if self.semantic_authority is not False:
            raise TacticianPlanGateError("gate receipts cannot claim semantic authority")
        if self.write_authority is not False:
            raise TacticianPlanGateError("gate receipts cannot claim write authority")
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "write_authority", False)

        object.__setattr__(
            self,
            "semantic_prediction_admission_blocked",
            _bool(
                self.semantic_prediction_admission_blocked,
                "semantic_prediction_admission_blocked",
            ),
        )
        object.__setattr__(
            self,
            "scores_cannot_override_hard_failure",
            _bool(
                self.scores_cannot_override_hard_failure,
                "scores_cannot_override_hard_failure",
            ),
        )
        if self.scores_cannot_override_hard_failure is not True:
            raise TacticianPlanGateError(
                "gate receipts must record that scores cannot override hard failures"
            )
        object.__setattr__(self, "scores_cannot_override_hard_failure", True)

        object.__setattr__(
            self, "producer_id", _identifier(self.producer_id, "producer_id")
        )

        bounds = self.bounds
        if bounds is not None and not isinstance(bounds, TacticianPlanGateBounds):
            if isinstance(bounds, Mapping):
                bounds = TacticianPlanGateBounds.from_dict(bounds)
            else:
                raise TacticianPlanGateError("bounds must be TacticianPlanGateBounds")
        object.__setattr__(self, "bounds", bounds)

        # Disposition invariants.
        if disposition is TacticianPlanGateDisposition.ADMITTED:
            if any(reason in _HARD_FAILURE_REASONS for reason in reasons):
                raise TacticianPlanGateError(
                    "admitted receipts cannot carry hard-failure reasons"
                )
            if any(reason in _ABSTAIN_REASONS for reason in reasons):
                raise TacticianPlanGateError(
                    "admitted receipts cannot carry abstention reasons"
                )
            if consistency is not None:
                raise TacticianPlanGateError(
                    "admitted receipts cannot carry a consistency-only subgoal"
                )
        if disposition is TacticianPlanGateDisposition.CONSISTENCY_ONLY:
            if consistency is None:
                raise TacticianPlanGateError(
                    "consistency_only receipts require a consistency subgoal plan"
                )
            if self.semantic_prediction_admission_blocked is not True:
                raise TacticianPlanGateError(
                    "consistency_only receipts must block semantic prediction admission"
                )
            if any(reason in _HARD_FAILURE_REASONS for reason in reasons):
                raise TacticianPlanGateError(
                    "consistency_only path is blocked by hard failures"
                )
        if disposition is TacticianPlanGateDisposition.REJECTED and not reasons:
            raise TacticianPlanGateError("rejected receipts require reason codes")
        if disposition is TacticianPlanGateDisposition.ABSTAINED and not reasons:
            raise TacticianPlanGateError("abstained receipts require reason codes")

        if len(canonical_json_bytes(self.to_dict())) > MAX_GATE_RECEIPT_BYTES:
            raise TacticianPlanGateBoundsError("receipt exceeds its serialized byte bound")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": TACTICIAN_PLAN_GATE_INTERFACE,
            "contract_version": CONTRACT_VERSION,
            "roots": self.roots.to_dict(),
            "plan_id": self.plan_id,
            "plan_content_id": self.plan_content_id,
            "corpus_content_id": self.corpus_content_id,
            "goal_content_ids": list(self.goal_content_ids),
            "candidate_content_ids": list(self.candidate_content_ids),
            "disposition": self.disposition.value,
            "reasons": [item.value for item in self.reasons],
            "goal_dispositions": [item.to_dict() for item in self.goal_dispositions],
            "permitted_subgoal_ids": list(self.permitted_subgoal_ids),
            "consistency_subgoal": (
                self.consistency_subgoal.to_dict()
                if self.consistency_subgoal is not None
                else None
            ),
            "semantic_authority": False,
            "write_authority": False,
            "semantic_prediction_admission_blocked": (
                self.semantic_prediction_admission_blocked
            ),
            "scores_cannot_override_hard_failure": True,
            "producer_id": self.producer_id,
            "bounds": self.bounds.to_dict() if self.bounds is not None else None,
            "provider_invoked": False,
            "solver_invoked": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return self._payload()

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict())

    @property
    def content_id(self) -> str:
        return self.receipt_id

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "receipt_id": self.receipt_id, "content_id": self.content_id}

    @property
    def admitted(self) -> bool:
        return self.disposition is TacticianPlanGateDisposition.ADMITTED

    @property
    def may_lower_obligations(self) -> bool:
        return self.disposition in {
            TacticianPlanGateDisposition.ADMITTED,
            TacticianPlanGateDisposition.CONSISTENCY_ONLY,
        }

    @property
    def write_paths(self) -> tuple[str, ...]:
        return ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TacticianPlanGateReceipt":
        allowed = {
            "schema",
            "interface",
            "contract_version",
            "roots",
            "plan_id",
            "plan_content_id",
            "corpus_content_id",
            "goal_content_ids",
            "candidate_content_ids",
            "disposition",
            "reasons",
            "goal_dispositions",
            "permitted_subgoal_ids",
            "consistency_subgoal",
            "semantic_authority",
            "write_authority",
            "semantic_prediction_admission_blocked",
            "scores_cannot_override_hard_failure",
            "producer_id",
            "bounds",
            "provider_invoked",
            "solver_invoked",
            "receipt_id",
            "content_id",
        }
        if not isinstance(payload, Mapping) or set(payload).difference(allowed):
            raise TacticianPlanGateError("receipt contains unsupported fields")
        if payload.get("schema") not in (None, "", TACTICIAN_PLAN_GATE_RECEIPT_SCHEMA):
            if payload.get("schema") != TACTICIAN_PLAN_GATE_RECEIPT_SCHEMA:
                raise TacticianPlanGateError("receipt has an unsupported schema")
        if payload.get("interface") not in (None, "", TACTICIAN_PLAN_GATE_INTERFACE):
            if payload.get("interface") != TACTICIAN_PLAN_GATE_INTERFACE:
                raise TacticianPlanGateError("receipt has an unsupported interface")
        if payload.get("provider_invoked", False) is not False:
            raise TacticianPlanGateError("gate receipt cannot claim provider invocation")
        if payload.get("solver_invoked", False) is not False:
            raise TacticianPlanGateError("gate receipt cannot claim solver invocation")
        if payload.get("semantic_authority", False) is not False:
            raise TacticianPlanGateError("gate receipt cannot claim semantic authority")
        if payload.get("write_authority", False) is not False:
            raise TacticianPlanGateError("gate receipt cannot claim write authority")

        leak = _walk_for_body_or_secret(payload)
        if leak:
            raise TacticianPlanGateError(f"receipt leaks body/secret material at {leak}")

        try:
            consistency_raw = payload.get("consistency_subgoal")
            consistency = None
            if consistency_raw is not None:
                consistency = (
                    ConsistencySubgoalPlan.from_dict(consistency_raw)
                    if isinstance(consistency_raw, Mapping)
                    else consistency_raw
                )
            bounds_raw = payload.get("bounds")
            bounds = None
            if bounds_raw is not None:
                bounds = (
                    TacticianPlanGateBounds.from_dict(bounds_raw)
                    if isinstance(bounds_raw, Mapping)
                    else bounds_raw
                )
            receipt = cls(
                roots=_roots(payload["roots"]),
                plan_id=payload["plan_id"],
                plan_content_id=payload["plan_content_id"],
                corpus_content_id=payload.get("corpus_content_id", ""),
                goal_content_ids=tuple(payload.get("goal_content_ids", ())),
                candidate_content_ids=tuple(payload.get("candidate_content_ids", ())),
                disposition=payload["disposition"],
                reasons=tuple(payload.get("reasons", ())),
                goal_dispositions=tuple(payload.get("goal_dispositions", ())),
                permitted_subgoal_ids=tuple(payload.get("permitted_subgoal_ids", ())),
                consistency_subgoal=consistency,
                semantic_authority=False,
                write_authority=False,
                semantic_prediction_admission_blocked=payload.get(
                    "semantic_prediction_admission_blocked", True
                ),
                scores_cannot_override_hard_failure=payload.get(
                    "scores_cannot_override_hard_failure", True
                ),
                producer_id=payload.get("producer_id", PRODUCER_ID),
                bounds=bounds,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TacticianPlanGateError("receipt is malformed") from exc

        claimed = payload.get("receipt_id") or payload.get("content_id")
        if claimed not in (None, "", receipt.receipt_id):
            raise TacticianPlanGateBindingError("receipt identity is forged")
        return receipt


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


class TacticianPlanGate:
    """Validate a Tactician plan against axiom smuggling and stale evidence.

    The gate recomputes identities, requires complete goal/residual
    dispositions, and rejects or abstains according to the LPR-010
    acceptance criteria.  It never dispatches a provider or solver.
    """

    def __init__(
        self,
        bounds: TacticianPlanGateBounds | Mapping[str, Any] | None = None,
    ) -> None:
        if bounds is None:
            self._bounds = TacticianPlanGateBounds()
        elif isinstance(bounds, TacticianPlanGateBounds):
            self._bounds = bounds
        elif isinstance(bounds, Mapping):
            self._bounds = TacticianPlanGateBounds.from_dict(bounds)
        else:
            raise TacticianPlanGateError("bounds must be TacticianPlanGateBounds")

    @property
    def bounds(self) -> TacticianPlanGateBounds:
        return self._bounds

    def validate(
        self,
        *,
        plan: TacticianSearchPlan | Mapping[str, Any],
        goals: Sequence[ProgramLogicGoal | Mapping[str, Any]] | None = None,
        candidates: Sequence[Any] | None = None,
        corpus: ProgramLogicPremiseCorpus | Mapping[str, Any] | None = None,
        gaps: Sequence[LogicGap | Mapping[str, Any]] | None = None,
        current_roots: ProgramLogicAuthorityRoots | Mapping[str, Any] | None = None,
        conflict_receipts: Sequence[Any] | None = None,
        claimed_identities: Mapping[str, str] | None = None,
        score_override_attempt: bool = False,
        extra_payload: Mapping[str, Any] | None = None,
    ) -> list[TacticianPlanRejectionReason]:
        """Return ordered rejection/abstention reasons (empty = fully admitted path)."""
        receipt = self.evaluate(
            plan=plan,
            goals=goals,
            candidates=candidates,
            corpus=corpus,
            gaps=gaps,
            current_roots=current_roots,
            conflict_receipts=conflict_receipts,
            claimed_identities=claimed_identities,
            score_override_attempt=score_override_attempt,
            extra_payload=extra_payload,
        )
        return list(receipt.reasons)

    def require_valid(
        self,
        *,
        plan: TacticianSearchPlan | Mapping[str, Any],
        goals: Sequence[ProgramLogicGoal | Mapping[str, Any]] | None = None,
        candidates: Sequence[Any] | None = None,
        corpus: ProgramLogicPremiseCorpus | Mapping[str, Any] | None = None,
        gaps: Sequence[LogicGap | Mapping[str, Any]] | None = None,
        current_roots: ProgramLogicAuthorityRoots | Mapping[str, Any] | None = None,
        conflict_receipts: Sequence[Any] | None = None,
        claimed_identities: Mapping[str, str] | None = None,
        score_override_attempt: bool = False,
        extra_payload: Mapping[str, Any] | None = None,
    ) -> TacticianPlanGateReceipt:
        """Return an admitted or consistency-only receipt, else raise."""
        receipt = self.evaluate(
            plan=plan,
            goals=goals,
            candidates=candidates,
            corpus=corpus,
            gaps=gaps,
            current_roots=current_roots,
            conflict_receipts=conflict_receipts,
            claimed_identities=claimed_identities,
            score_override_attempt=score_override_attempt,
            extra_payload=extra_payload,
        )
        if receipt.disposition is TacticianPlanGateDisposition.REJECTED:
            codes = ",".join(item.value for item in receipt.reasons) or "rejected"
            raise TacticianPlanGateError(f"tactician plan rejected: {codes}")
        if receipt.disposition is TacticianPlanGateDisposition.ABSTAINED:
            codes = ",".join(item.value for item in receipt.reasons) or "abstained"
            raise TacticianPlanGateError(f"tactician plan abstained: {codes}")
        return receipt

    def evaluate(
        self,
        *,
        plan: TacticianSearchPlan | Mapping[str, Any],
        goals: Sequence[ProgramLogicGoal | Mapping[str, Any]] | None = None,
        candidates: Sequence[Any] | None = None,
        corpus: ProgramLogicPremiseCorpus | Mapping[str, Any] | None = None,
        gaps: Sequence[LogicGap | Mapping[str, Any]] | None = None,
        current_roots: ProgramLogicAuthorityRoots | Mapping[str, Any] | None = None,
        conflict_receipts: Sequence[Any] | None = None,
        claimed_identities: Mapping[str, str] | None = None,
        score_override_attempt: bool = False,
        extra_payload: Mapping[str, Any] | None = None,
    ) -> TacticianPlanGateReceipt:
        """Full fail-closed evaluation producing a typed receipt."""
        reasons: list[TacticianPlanRejectionReason] = []

        try:
            typed_plan = _decode_plan(plan)
            typed_goals = _decode_goals(goals)
            typed_candidates = _decode_hypotheses(candidates)
            typed_corpus = _decode_corpus(corpus)
            typed_gaps = _decode_gaps(gaps)
            typed_conflicts = _decode_conflict_receipts(conflict_receipts)
            if current_roots is not None:
                expected_roots = _roots(current_roots)
            else:
                expected_roots = typed_plan.roots
        except (TacticianPlanGateError, ProgramLogicPredictionError, TypeError, ValueError) as exc:
            # Malformed inputs are hard failures.
            raise TacticianPlanGateError(f"malformed_input: {exc}") from exc

        # ---- Identity recomputation ---------------------------------------
        plan_content_id = typed_plan.content_id
        recomputed_plan_id = _recompute_identity(typed_plan)
        if plan_content_id != recomputed_plan_id:
            reasons.append(TacticianPlanRejectionReason.FORGED_IDENTITY)

        goal_content_ids: list[str] = []
        for goal in typed_goals:
            cid = goal.content_id
            if cid != _recompute_identity(goal):
                reasons.append(TacticianPlanRejectionReason.FORGED_IDENTITY)
            goal_content_ids.append(cid)

        candidate_content_ids: list[str] = []
        for hypothesis in typed_candidates:
            cid = hypothesis.content_id
            if cid != _recompute_identity(hypothesis):
                reasons.append(TacticianPlanRejectionReason.FORGED_IDENTITY)
            candidate_content_ids.append(cid)

        corpus_content_id = ""
        if typed_corpus is not None:
            corpus_content_id = typed_corpus.content_id
            if corpus_content_id != _recompute_identity(typed_corpus):
                reasons.append(TacticianPlanRejectionReason.FORGED_IDENTITY)

        if claimed_identities:
            checks = {
                "plan": plan_content_id,
                "corpus": corpus_content_id,
            }
            for index, cid in enumerate(goal_content_ids):
                checks[f"goal:{index}"] = cid
                checks[f"goal_id:{typed_goals[index].goal_id}"] = cid
            for index, cid in enumerate(candidate_content_ids):
                checks[f"candidate:{index}"] = cid
                checks[f"hypothesis:{typed_candidates[index].hypothesis_id}"] = cid
            for key, claimed in claimed_identities.items():
                expected = checks.get(key)
                if expected is None:
                    # Allow direct content-id claims by role.
                    if key in ("plan_content_id", "plan"):
                        expected = plan_content_id
                    elif key in ("corpus_content_id", "corpus"):
                        expected = corpus_content_id
                    else:
                        continue
                if claimed and claimed != expected:
                    reasons.append(TacticianPlanRejectionReason.FORGED_IDENTITY)

        # ---- Root integrity -----------------------------------------------
        if not _roots_equal(typed_plan.roots, expected_roots):
            reasons.append(TacticianPlanRejectionReason.CHANGED_ROOTS)

        for goal in typed_goals:
            if not _roots_equal(goal.roots, expected_roots):
                reasons.append(TacticianPlanRejectionReason.CROSS_ROOT_BINDING)

        for hypothesis in typed_candidates:
            if not _roots_equal(hypothesis.roots, expected_roots):
                reasons.append(TacticianPlanRejectionReason.CROSS_ROOT_BINDING)

        if typed_corpus is not None and not _roots_equal(
            typed_corpus.roots, expected_roots
        ):
            reasons.append(TacticianPlanRejectionReason.CROSS_ROOT_BINDING)

        for gap in typed_gaps:
            if not _roots_equal(gap.roots, expected_roots):
                reasons.append(TacticianPlanRejectionReason.CROSS_ROOT_BINDING)

        for receipt in typed_conflicts:
            if not _roots_equal(receipt.roots, expected_roots):
                reasons.append(TacticianPlanRejectionReason.CROSS_ROOT_BINDING)

        # Plan corpus_id root binding.
        if typed_corpus is not None:
            if typed_plan.roots.corpus_id not in {
                typed_corpus.roots.corpus_id,
                typed_corpus.content_id,
            } and typed_corpus.roots.corpus_id != expected_roots.corpus_id:
                reasons.append(TacticianPlanRejectionReason.STALE_EVIDENCE)

        # ---- Body / secret leakage ----------------------------------------
        payloads_to_scan: list[Any] = [
            typed_plan.to_dict(),
            *[goal.to_dict() for goal in typed_goals],
            *[item.to_dict() for item in typed_candidates],
        ]
        if typed_corpus is not None:
            payloads_to_scan.append(typed_corpus.to_dict())
        for gap in typed_gaps:
            payloads_to_scan.append(gap.to_dict())
        if extra_payload is not None:
            payloads_to_scan.append(extra_payload)
        for payload in payloads_to_scan:
            if _walk_for_body_or_secret(payload):
                reasons.append(TacticianPlanRejectionReason.SECRET_OR_BODY_LEAKAGE)
                break

        # ---- Semantic / write authority flags -----------------------------
        if typed_plan.semantic_authority is not False:
            reasons.append(TacticianPlanRejectionReason.SEMANTIC_AUTHORITY_CLAIM)
        for hypothesis in typed_candidates:
            if hypothesis.semantic_authority is not False:
                reasons.append(TacticianPlanRejectionReason.SEMANTIC_AUTHORITY_CLAIM)
            # Hypotheses never carry write paths; reject smuggled fields via dict.
        if extra_payload:
            for key, value in extra_payload.items():
                normalized = _normalize_key(str(key))
                if normalized in {"semantic_authority", "proof_authority"} and value is True:
                    reasons.append(TacticianPlanRejectionReason.SEMANTIC_AUTHORITY_CLAIM)
                if normalized in {
                    "write_authority",
                    "write_allowed",
                    "authorized_write",
                } and value is True:
                    reasons.append(TacticianPlanRejectionReason.WRITE_AUTHORITY_CLAIM)
                if normalized in {"write_paths", "permitted_write_paths", "authorized_paths"}:
                    if value:
                        reasons.append(TacticianPlanRejectionReason.WRITE_AUTHORITY_CLAIM)
                if normalized in _SCORE_OVERRIDE_KEYS and value:
                    reasons.append(TacticianPlanRejectionReason.SCORE_OVERRIDE_ATTEMPT)

        if score_override_attempt:
            reasons.append(TacticianPlanRejectionReason.SCORE_OVERRIDE_ATTEMPT)

        # ---- Prompt directives treated as policy --------------------------
        for ref_name in (
            "stop_policy_ref",
            "escalation_policy_ref",
            "abstention_policy_ref",
            "resource_policy_ref",
            "planner_id",
            "model_id",
            "config_id",
        ):
            ref_value = getattr(typed_plan, ref_name, "") or ""
            if ref_value and _is_prompt_policy_ref(ref_value):
                reasons.append(TacticianPlanRejectionReason.PROMPT_DIRECTIVE_AS_POLICY)
                break
        for ref in typed_plan.invalidation_refs:
            if _is_prompt_policy_ref(ref):
                reasons.append(TacticianPlanRejectionReason.PROMPT_DIRECTIVE_AS_POLICY)
                break

        # ---- Budget escalation --------------------------------------------
        if len(typed_plan.subgoals) > self._bounds.max_subgoals:
            reasons.append(TacticianPlanRejectionReason.BUDGET_ESCALATION)
        if len(typed_plan.ordered_source_routes) > self._bounds.max_routes:
            reasons.append(TacticianPlanRejectionReason.BUDGET_ESCALATION)
        if len(typed_plan.query_refs) > self._bounds.max_queries:
            reasons.append(TacticianPlanRejectionReason.BUDGET_ESCALATION)
        if len(typed_plan.selected_premise_ids) > self._bounds.max_premises:
            reasons.append(TacticianPlanRejectionReason.BUDGET_ESCALATION)
        for subgoal in typed_plan.subgoals:
            if subgoal.score_millipercent > self._bounds.max_score_millipercent:
                reasons.append(TacticianPlanRejectionReason.BUDGET_ESCALATION)
                break
        for hypothesis in typed_candidates:
            if (
                hypothesis.nomination_score_millipercent
                > self._bounds.max_score_millipercent
            ):
                reasons.append(TacticianPlanRejectionReason.BUDGET_ESCALATION)
                break
        # Escalation policy that claims unbounded or escalated budgets.
        esc = (typed_plan.escalation_policy_ref or "").lower()
        if any(
            token in esc
            for token in (
                "unbounded",
                "escalate:infinite",
                "budget:override",
                "max:unlimited",
                "no_limit",
            )
        ):
            reasons.append(TacticianPlanRejectionReason.BUDGET_ESCALATION)

        # ---- Subgoal identity / cycles ------------------------------------
        subgoal_ids = [item.subgoal_id for item in typed_plan.subgoals]
        if len(subgoal_ids) != len(set(subgoal_ids)):
            reasons.append(TacticianPlanRejectionReason.DUPLICATED_SUBGOAL_IDENTITY)
        if _subgoal_has_cycle(typed_plan.subgoals):
            reasons.append(TacticianPlanRejectionReason.TACTICIAN_PLAN_CYCLIC)

        # Subgoals must target plan goal_ids (contract already checks, re-assert).
        plan_goal_ids = set(typed_plan.goal_ids)
        for subgoal in typed_plan.subgoals:
            if subgoal.goal_id not in plan_goal_ids:
                reasons.append(TacticianPlanRejectionReason.TACTICIAN_PLAN_INVALID)

        # ---- Goal / residual dispositions ---------------------------------
        goal_by_id = {goal.goal_id: goal for goal in typed_goals}
        disposition_bindings: list[GoalDispositionBinding] = []
        covered_facets: dict[str, set[str]] = {
            goal.goal_id: set() for goal in typed_goals
        }

        # Index subgoals by goal.
        subgoals_by_goal: dict[str, list[LogicSubgoal]] = {}
        for subgoal in typed_plan.subgoals:
            subgoals_by_goal.setdefault(subgoal.goal_id, []).append(subgoal)

        # Goals that must have an explicit disposition in the plan.
        for goal in typed_goals:
            is_residual = goal.disposition in _RESIDUAL_GOAL_DISPOSITIONS
            requires_coverage = (
                goal.disposition in _GOAL_DISPOSITIONS_REQUIRING_COVERAGE
                or is_residual
            )
            if goal.goal_id not in plan_goal_ids:
                if requires_coverage:
                    if is_residual:
                        reasons.append(
                            TacticianPlanRejectionReason.OMITTED_RESIDUAL_DISPOSITION
                        )
                    else:
                        reasons.append(
                            TacticianPlanRejectionReason.OMITTED_GOAL_DISPOSITION
                        )
                continue

            related = subgoals_by_goal.get(goal.goal_id, [])
            # Residual goals need an explicit residual/abstained/unsupported subgoal
            # or the goal itself already carries residual disposition listed in plan.
            if requires_coverage and not related and not is_residual:
                # Plan lists the goal but has no subgoal disposition for open goals.
                if goal.disposition in {
                    GoalDisposition.OPEN,
                    GoalDisposition.PLANNED,
                    GoalDisposition.ADMITTED,
                }:
                    reasons.append(
                        TacticianPlanRejectionReason.OMITTED_GOAL_DISPOSITION
                    )
            if is_residual and not related and goal.goal_id not in plan_goal_ids:
                reasons.append(
                    TacticianPlanRejectionReason.OMITTED_RESIDUAL_DISPOSITION
                )

            # Facet coverage: required facets must appear via subgoal claim refs
            # or hypothesis evidence, or be recorded as unsupported on the goal.
            required_facet_ids = {
                facet.facet_id for facet in goal.required_facets if not facet.unsupported
            }
            unsupported_facet_ids = {
                facet.facet_id for facet in goal.unsupported_facets
            }
            # Subgoal claim_refs may reference facets as "facet:<id>" or the id.
            for subgoal in related:
                claim = subgoal.claim_ref
                if claim in required_facet_ids:
                    covered_facets[goal.goal_id].add(claim)
                elif claim.startswith("facet:") and claim[len("facet:") :] in required_facet_ids:
                    covered_facets[goal.goal_id].add(claim[len("facet:") :])
                # Also accept claim_ref equality with facet contract refs.
                for facet in goal.required_facets:
                    if facet.contract_ref and facet.contract_ref == claim:
                        covered_facets[goal.goal_id].add(facet.facet_id)
                    if facet.facet_id == claim or claim.endswith(facet.facet_id):
                        covered_facets[goal.goal_id].add(facet.facet_id)

            for hypothesis in typed_candidates:
                if hypothesis.target_goal_id != goal.goal_id:
                    continue
                for ref in hypothesis.evidence_refs:
                    for facet in goal.required_facets:
                        if facet.facet_id in ref or (
                            facet.contract_ref and facet.contract_ref in ref
                        ):
                            covered_facets[goal.goal_id].add(facet.facet_id)
                if hypothesis.claimed_consequence_ref:
                    for facet in goal.required_facets:
                        if facet.facet_id in hypothesis.claimed_consequence_ref:
                            covered_facets[goal.goal_id].add(facet.facet_id)

            # Unsupported facets are explicitly discharged.
            covered_facets[goal.goal_id].update(unsupported_facet_ids)

            missing = required_facet_ids - covered_facets[goal.goal_id]
            # Only enforce facet coverage for goals that are planned/admitted
            # with subgoals present (open inventory without facets is ok).
            if missing and related and goal.disposition in {
                GoalDisposition.PLANNED,
                GoalDisposition.ADMITTED,
                GoalDisposition.OPEN,
            }:
                reasons.append(TacticianPlanRejectionReason.OMITTED_FACET)

            plan_disposition = goal.disposition.value
            if related:
                # Prefer the most advanced non-pending subgoal disposition.
                plan_disposition = related[0].disposition.value
            disposition_bindings.append(
                GoalDispositionBinding(
                    goal_id=goal.goal_id,
                    disposition=plan_disposition,
                    is_residual=is_residual,
                    subgoal_ids=tuple(item.subgoal_id for item in related),
                    facet_ids=tuple(sorted(covered_facets[goal.goal_id])),
                )
            )

        # Plan may not introduce goal_ids without goal records when goals supplied.
        if typed_goals:
            unknown_plan_goals = plan_goal_ids - set(goal_by_id)
            if unknown_plan_goals:
                reasons.append(TacticianPlanRejectionReason.TACTICIAN_PLAN_INVALID)

        # ---- Source authority / unauthorized sources ----------------------
        premise_by_id: dict[str, ProgramLogicPremise] = {}
        if typed_corpus is not None:
            premise_by_id = {item.premise_id: item for item in typed_corpus.premises}

        for route in typed_plan.ordered_source_routes:
            if route is SourceRouteKind.LLM and not self._bounds.allow_model_hypothesis:
                reasons.append(TacticianPlanRejectionReason.UNAUTHORIZED_SOURCE)
            if (
                route in _NOMINATING_ROUTES
                and not self._bounds.allow_approximate_routes
                and route is not SourceRouteKind.LLM
            ):
                reasons.append(TacticianPlanRejectionReason.UNAUTHORIZED_SOURCE)

        for subgoal in typed_plan.subgoals:
            if (
                subgoal.source_route in _NOMINATING_ROUTES
                and subgoal.source_authority is SourceAuthorityClass.AUTHORITATIVE
            ):
                reasons.append(TacticianPlanRejectionReason.UNAUTHORIZED_SOURCE)
            if (
                subgoal.source_route is SourceRouteKind.LLM
                and not self._bounds.allow_model_hypothesis
            ):
                reasons.append(TacticianPlanRejectionReason.UNAUTHORIZED_SOURCE)

        for hypothesis in typed_candidates:
            for route in hypothesis.evidence_route_kinds:
                if (
                    route is SourceRouteKind.LLM
                    and not self._bounds.allow_model_hypothesis
                ):
                    reasons.append(TacticianPlanRejectionReason.UNAUTHORIZED_SOURCE)
                if (
                    route in _NOMINATING_ROUTES
                    and hypothesis.source_authority is SourceAuthorityClass.AUTHORITATIVE
                ):
                    reasons.append(TacticianPlanRejectionReason.UNAUTHORIZED_SOURCE)
            # Nominating hypotheses claiming completeness/proved are invalid.
            if (
                hypothesis.disposition is HypothesisDisposition.PROVED
                and hypothesis.source_authority
                in {
                    SourceAuthorityClass.NOMINATING,
                    SourceAuthorityClass.DIAGNOSTIC,
                    SourceAuthorityClass.NONE,
                }
            ):
                reasons.append(TacticianPlanRejectionReason.UNAUTHORIZED_SOURCE)

        # ---- Premises: self-authoring, unauthorized, exclusions -----------
        selected = set(typed_plan.selected_premise_ids)
        excluded = set(typed_plan.excluded_premise_ids)
        rationale_refs = list(typed_plan.exclusion_rationale_refs)

        if typed_corpus is not None:
            live_ids = set(premise_by_id)
            for premise_id in selected:
                if premise_id not in live_ids:
                    reasons.append(TacticianPlanRejectionReason.UNAUTHORIZED_PREMISE)
                    continue
                premise = premise_by_id[premise_id]
                if premise.self_validation:
                    reasons.append(
                        TacticianPlanRejectionReason.PREMISE_SELF_REFERENTIAL
                    )
                if premise.semantic_authority is not False:
                    reasons.append(
                        TacticianPlanRejectionReason.SEMANTIC_AUTHORITY_CLAIM
                    )
                # Hypothesis-class premises cannot be treated as authoritative
                # selected axioms for plan admission.
                if (
                    premise.source_class in _HYPOTHESIS_SOURCE_CLASSES
                    and premise.expectation_authority
                ):
                    reasons.append(TacticianPlanRejectionReason.PREMISE_UNTRUSTED)
                if premise.authority is PremiseAuthority.HYPOTHESIS and any(
                    route in _AUTHORITATIVE_ROUTES
                    for route in typed_plan.ordered_source_routes
                ):
                    # Selecting pure hypothesis premises as sole authority is ok
                    # for nomination, but self-authoring checks below catch loops.
                    pass

            # Forged exclusions: exclude IDs not in corpus, or missing rationales
            # when exclusions are present, or rationale that forges a non-ref.
            for premise_id in excluded:
                if premise_id not in live_ids and premise_id not in {
                    item.premise_id for item in typed_corpus.tombstones
                }:
                    reasons.append(TacticianPlanRejectionReason.FORGED_EXCLUSION)
            if excluded and not rationale_refs:
                reasons.append(TacticianPlanRejectionReason.FORGED_EXCLUSION)
            for ref in rationale_refs:
                if _is_prompt_policy_ref(ref) or _contains_secret_text(ref):
                    reasons.append(TacticianPlanRejectionReason.FORGED_EXCLUSION)
                if ref.startswith("forged:") or ref.startswith("fake:"):
                    reasons.append(TacticianPlanRejectionReason.FORGED_EXCLUSION)

        elif selected or excluded:
            # Plan references premises without a bound corpus.
            reasons.append(TacticianPlanRejectionReason.UNAUTHORIZED_PREMISE)

        # Self-authoring candidate premises: a hypothesis selects a premise
        # whose identity is the hypothesis itself or derives from its own claim.
        for hypothesis in typed_candidates:
            for premise_id in hypothesis.selected_premise_ids:
                if premise_id in {
                    hypothesis.hypothesis_id,
                    hypothesis.claimed_consequence_ref,
                    hypothesis.construction_ref,
                    hypothesis.value_ref,
                }:
                    reasons.append(
                        TacticianPlanRejectionReason.SELF_AUTHORING_CANDIDATE_PREMISE
                    )
                if premise_id.startswith("hypothesis:") or premise_id.startswith(
                    f"self:{hypothesis.hypothesis_id}"
                ):
                    reasons.append(
                        TacticianPlanRejectionReason.SELF_AUTHORING_CANDIDATE_PREMISE
                    )
                premise = premise_by_id.get(premise_id)
                if premise is not None:
                    if premise.self_validation:
                        reasons.append(
                            TacticianPlanRejectionReason.SELF_AUTHORING_CANDIDATE_PREMISE
                        )
                    # Candidate implementation premises authored by the same
                    # hypothesis construction cannot ground themselves.
                    if (
                        premise.source_class
                        is PremiseSourceClass.CANDIDATE_IMPLEMENTATION
                        and hypothesis.construction_ref
                        and (
                            hypothesis.construction_ref == premise.statement_ref
                            or hypothesis.construction_ref in premise.statement_ref
                            or premise.statement_ref in hypothesis.construction_ref
                        )
                    ):
                        reasons.append(
                            TacticianPlanRejectionReason.SELF_AUTHORING_CANDIDATE_PREMISE
                        )
                    if premise.statement_ref == hypothesis.claimed_consequence_ref:
                        reasons.append(
                            TacticianPlanRejectionReason.SELF_AUTHORING_CANDIDATE_PREMISE
                        )

            # Plan-level selected premises that equal hypothesis identity.
            for premise_id in selected:
                if premise_id == hypothesis.hypothesis_id:
                    reasons.append(
                        TacticianPlanRejectionReason.SELF_AUTHORING_CANDIDATE_PREMISE
                    )

        # ---- Consistency / structural conflict ----------------------------
        consistency_subgoal: ConsistencySubgoalPlan | None = None
        structural_conflict = False
        suspected_contradiction = False
        unknown_consistency = False
        logical_conflict_proved = False

        if typed_corpus is not None:
            disposition = typed_corpus.consistency_disposition
            if disposition is ConsistencyDisposition.STRUCTURAL_CONFLICT:
                structural_conflict = True
            elif disposition is ConsistencyDisposition.UNKNOWN:
                # Empty corpus stays unknown → abstain only when goals need premises.
                if typed_corpus.premises and selected:
                    unknown_consistency = True
                elif not typed_corpus.premises and selected:
                    unknown_consistency = True
            elif disposition is ConsistencyDisposition.SUSPECTED_AUTHORITATIVE_CONTRADICTION:
                suspected_contradiction = True
            elif disposition is ConsistencyDisposition.CONSISTENCY_OBLIGATION_EMITTED:
                suspected_contradiction = True
            elif disposition is ConsistencyDisposition.LOGICAL_CONFLICT_PROVED:
                logical_conflict_proved = True
                if not typed_corpus.conflict_receipts and not typed_conflicts:
                    # Forged logical-conflict claim without receipt.
                    reasons.append(TacticianPlanRejectionReason.FORGED_IDENTITY)
                    structural_conflict = True

            # Explicit conflicting authoritative premises in selected set.
            selected_premises = [
                premise_by_id[pid] for pid in selected if pid in premise_by_id
            ]
            authoritative_selected = [
                p
                for p in selected_premises
                if p.authority is PremiseAuthority.EXPECTATION
                or p.expectation_authority
            ]
            conflict_pairs: list[tuple[str, str]] = []
            by_id_auth = {p.premise_id: p for p in authoritative_selected}
            for premise in authoritative_selected:
                for other_id in premise.conflicts_with:
                    if other_id in by_id_auth:
                        pair = tuple(sorted((premise.premise_id, other_id)))
                        conflict_pairs.append(pair)  # type: ignore[arg-type]
            if conflict_pairs and not logical_conflict_proved:
                suspected_contradiction = True

            if structural_conflict:
                reasons.append(TacticianPlanRejectionReason.STRUCTURAL_CONFLICT)
            if unknown_consistency:
                reasons.append(TacticianPlanRejectionReason.UNKNOWN_CONSISTENCY)
            if suspected_contradiction:
                reasons.append(
                    TacticianPlanRejectionReason.SUSPECTED_LOGICAL_CONTRADICTION
                )
                # Emit consistency subgoal; permit only its proof plan.
                conflict_premise_ids = sorted(
                    {
                        pid
                        for pair in conflict_pairs
                        for pid in pair
                    }
                    or {
                        item
                        for obligation in typed_corpus.consistency_obligations
                        for item in obligation.premise_ids
                    }
                    or {p.premise_id for p in authoritative_selected}
                )
                obligation_ids = tuple(
                    item.obligation_id
                    for item in typed_corpus.consistency_obligations
                )
                consistency_goal_id = next(
                    (
                        g.goal_id
                        for g in typed_goals
                        if g.family is GoalFamily.CONSISTENCY
                    ),
                    typed_plan.goal_ids[0] if typed_plan.goal_ids else "goal:consistency",
                )
                # Prefer an existing consistency subgoal on the plan.
                existing_consistency = next(
                    (
                        sg
                        for sg in typed_plan.subgoals
                        if sg.claim_ref.startswith("consistency:")
                        or "consistency" in sg.subgoal_id
                        or (
                            sg.goal_id in goal_by_id
                            and goal_by_id[sg.goal_id].family is GoalFamily.CONSISTENCY
                        )
                    ),
                    None,
                )
                if existing_consistency is not None:
                    subgoal_id = existing_consistency.subgoal_id
                    claim_ref = existing_consistency.claim_ref
                    consistency_goal_id = existing_consistency.goal_id
                else:
                    subgoal_id = (
                        f"subgoal:consistency:{content_identity({'p': conflict_premise_ids})[:16]}"
                    )
                    claim_ref = "consistency:authoritative-premises"
                consistency_subgoal = ConsistencySubgoalPlan(
                    subgoal_id=subgoal_id,
                    goal_id=consistency_goal_id,
                    premise_ids=tuple(conflict_premise_ids)
                    or tuple(sorted(selected))
                    or ("premise:consistency-unknown",),
                    obligation_ids=obligation_ids,
                    claim_ref=claim_ref,
                    semantic_prediction_admission_blocked=True,
                )

            # Logical conflict proved: semantic prediction admission remains
            # blocked unless a separately validated receipt is present (LPR-012).
            validated_receipts = list(typed_corpus.conflict_receipts) + list(
                typed_conflicts
            )
            if logical_conflict_proved or validated_receipts:
                # Even with a receipt, this gate does not admit semantic
                # predictions; LPR-012 coordinates that.  We still allow
                # consistency-only lowering when conflict is proved.
                if not validated_receipts:
                    reasons.append(
                        TacticianPlanRejectionReason.PREDICTION_ADMISSION_BLOCKED
                    )
                else:
                    # Validated conflict: still block prediction admission at
                    # this gate (LPR-012 owns admission).
                    reasons.append(
                        TacticianPlanRejectionReason.PREDICTION_ADMISSION_BLOCKED
                    )
                    if TacticianPlanRejectionReason.SUSPECTED_LOGICAL_CONTRADICTION not in reasons:
                        # Treat proved conflict as consistency-only path.
                        reasons.append(
                            TacticianPlanRejectionReason.SUSPECTED_LOGICAL_CONTRADICTION
                        )
                        if consistency_subgoal is None:
                            premise_ids = sorted(
                                {
                                    pid
                                    for receipt in validated_receipts
                                    for pid in receipt.premise_ids
                                }
                            )
                            consistency_subgoal = ConsistencySubgoalPlan(
                                subgoal_id="subgoal:consistency:proved-conflict",
                                goal_id=(
                                    typed_plan.goal_ids[0]
                                    if typed_plan.goal_ids
                                    else "goal:consistency"
                                ),
                                premise_ids=tuple(premise_ids)
                                or ("premise:conflict",),
                                obligation_ids=(),
                                claim_ref="consistency:logical-conflict-proved",
                                semantic_prediction_admission_blocked=True,
                            )

        # Gaps marked consistency without resolution.
        for gap in typed_gaps:
            if (
                gap.missing_class.value == "consistency"
                and gap.disposition.value in {"required", "frontier"}
            ):
                if (
                    TacticianPlanRejectionReason.SUSPECTED_LOGICAL_CONTRADICTION
                    not in reasons
                    and TacticianPlanRejectionReason.UNKNOWN_CONSISTENCY not in reasons
                ):
                    reasons.append(TacticianPlanRejectionReason.UNKNOWN_CONSISTENCY)

        # ---- Score override cannot clear hard failures --------------------
        # Deduplicate while preserving order.
        ordered_reasons: list[TacticianPlanRejectionReason] = []
        seen_reasons: set[TacticianPlanRejectionReason] = set()
        for reason in reasons:
            if reason not in seen_reasons:
                seen_reasons.add(reason)
                ordered_reasons.append(reason)

        hard = [r for r in ordered_reasons if r in _HARD_FAILURE_REASONS]
        # High nomination scores never erase hard reasons (already true by construction).
        # If caller attempted score override, hard reasons stay and SCORE_OVERRIDE is present.
        if score_override_attempt and hard:
            if TacticianPlanRejectionReason.SCORE_OVERRIDE_ATTEMPT not in ordered_reasons:
                ordered_reasons.append(
                    TacticianPlanRejectionReason.SCORE_OVERRIDE_ATTEMPT
                )

        # ---- Final disposition --------------------------------------------
        has_hard = any(r in _HARD_FAILURE_REASONS for r in ordered_reasons)
        has_abstain = any(r in _ABSTAIN_REASONS for r in ordered_reasons)
        has_consistency = any(r in _CONSISTENCY_ONLY_REASONS for r in ordered_reasons)
        # PREDICTION_ADMISSION_BLOCKED alone under consistency is not a hard reject.
        prediction_blocked = (
            TacticianPlanRejectionReason.PREDICTION_ADMISSION_BLOCKED in ordered_reasons
        )

        if has_hard:
            disposition = TacticianPlanGateDisposition.REJECTED
            permitted: tuple[str, ...] = ()
            consistency_subgoal = None  # hard failures kill the consistency path
            semantic_prediction_blocked = True
        elif has_abstain and not has_consistency:
            disposition = TacticianPlanGateDisposition.ABSTAINED
            permitted = ()
            consistency_subgoal = None
            semantic_prediction_blocked = True
        elif has_consistency and consistency_subgoal is not None:
            disposition = TacticianPlanGateDisposition.CONSISTENCY_ONLY
            permitted = (consistency_subgoal.subgoal_id,)
            semantic_prediction_blocked = True
        elif has_abstain:
            disposition = TacticianPlanGateDisposition.ABSTAINED
            permitted = ()
            consistency_subgoal = None
            semantic_prediction_blocked = True
        else:
            disposition = TacticianPlanGateDisposition.ADMITTED
            permitted = tuple(item.subgoal_id for item in typed_plan.subgoals)
            # Clean plans may proceed to obligation lowering; prediction
            # admission for consequences still requires later stages.
            # Without contradiction, prediction is not specially blocked by LPR-012.
            semantic_prediction_blocked = prediction_blocked
            # Default for clean admit: prediction not yet admitted (false authority),
            # but the LPR-012 block is only mandatory under contradiction.
            if not prediction_blocked:
                semantic_prediction_blocked = False

        # Strip soft-only prediction block from admitted clean path reasons if
        # it was the sole non-hard reason and disposition is admitted.
        if (
            disposition is TacticianPlanGateDisposition.ADMITTED
            and ordered_reasons == [
                TacticianPlanRejectionReason.PREDICTION_ADMISSION_BLOCKED
            ]
        ):
            ordered_reasons = []

        return TacticianPlanGateReceipt(
            roots=expected_roots,
            plan_id=typed_plan.plan_id,
            plan_content_id=plan_content_id,
            corpus_content_id=corpus_content_id,
            goal_content_ids=tuple(goal_content_ids),
            candidate_content_ids=tuple(candidate_content_ids),
            disposition=disposition,
            reasons=tuple(ordered_reasons),
            goal_dispositions=tuple(disposition_bindings),
            permitted_subgoal_ids=permitted,
            consistency_subgoal=consistency_subgoal,
            semantic_authority=False,
            write_authority=False,
            semantic_prediction_admission_blocked=semantic_prediction_blocked,
            scores_cannot_override_hard_failure=True,
            producer_id=PRODUCER_ID,
            bounds=self._bounds,
        )


def gate_tactician_plan(
    *,
    plan: TacticianSearchPlan | Mapping[str, Any],
    goals: Sequence[ProgramLogicGoal | Mapping[str, Any]] | None = None,
    candidates: Sequence[Any] | None = None,
    corpus: ProgramLogicPremiseCorpus | Mapping[str, Any] | None = None,
    gaps: Sequence[LogicGap | Mapping[str, Any]] | None = None,
    current_roots: ProgramLogicAuthorityRoots | Mapping[str, Any] | None = None,
    bounds: TacticianPlanGateBounds | Mapping[str, Any] | None = None,
    conflict_receipts: Sequence[Any] | None = None,
    claimed_identities: Mapping[str, str] | None = None,
    score_override_attempt: bool = False,
    extra_payload: Mapping[str, Any] | None = None,
) -> TacticianPlanGateReceipt:
    """Module-level convenience entry point for the plan security gate."""
    return TacticianPlanGate(bounds=bounds).evaluate(
        plan=plan,
        goals=goals,
        candidates=candidates,
        corpus=corpus,
        gaps=gaps,
        current_roots=current_roots,
        conflict_receipts=conflict_receipts,
        claimed_identities=claimed_identities,
        score_override_attempt=score_override_attempt,
        extra_payload=extra_payload,
    )


__all__ = (
    "CONSISTENCY_SUBGOAL_SCHEMA",
    "DEFAULT_MAX_PREMISES",
    "DEFAULT_MAX_QUERIES",
    "DEFAULT_MAX_ROUTES",
    "DEFAULT_MAX_SCORE_MILLIPERCENT",
    "DEFAULT_MAX_SUBGOALS",
    "GOAL_DISPOSITION_BINDING_SCHEMA",
    "PRODUCER_ID",
    "TACTICIAN_PLAN_GATE_BOUNDS_SCHEMA",
    "TACTICIAN_PLAN_GATE_INTERFACE",
    "TACTICIAN_PLAN_GATE_RECEIPT_SCHEMA",
    "ConsistencySubgoalPlan",
    "GoalDispositionBinding",
    "TacticianPlanGate",
    "TacticianPlanGateBindingError",
    "TacticianPlanGateBounds",
    "TacticianPlanGateBoundsError",
    "TacticianPlanGateDisposition",
    "TacticianPlanGateError",
    "TacticianPlanGateReceipt",
    "TacticianPlanRejectionReason",
    "gate_tactician_plan",
)
