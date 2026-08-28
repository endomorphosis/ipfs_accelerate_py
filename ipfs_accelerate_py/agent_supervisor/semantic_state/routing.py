"""Deterministic model routing for the semantic-compression harness.

``routing.py`` scores only the declared inputs: context size, lowest relevant
confidence, risk class, affected dependency cone, unresolved obligations,
prior repair failures, and available proofs. Results are one of the five
closed ``ModelRoute`` values and always carry an ordered explanation.

Providers are never hardcoded here. ``deterministic_only`` means no model
invocation. ``human_review_required`` halts before provider dispatch or root
publication. Production promotion gates live in ``providers.py``.

Importing this module starts no threads, processes, databases, or network
calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    BOARD_NAMESPACE,
    ContextPack,
    HarnessError,
    ModelRoute,
    _bool,
    _closed,
    _enum,
    _nonneg_int,
    _text,
)

MODEL_ROUTING_INTERFACE = "ModelRouting@1"
MODEL_ROUTING_SCHEMA = "semantic-state-model-routing@1"
ADAPTER_ID = "semantic-model-routing"

# Closed confidence ranks (lower is weaker / less assured).
_CONFIDENCE_RANK: Mapping[str, int] = {
    "exact": 3,
    "conservative": 2,
    "heuristic": 1,
    "opaque": 0,
}

# Closed risk ranks (higher is more dangerous).
_RISK_RANK: Mapping[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}

# Default absolute ceilings for deterministic scoring.
_DEFAULT_SMALL_CONTEXT_TOKENS = 4_096
_DEFAULT_MEDIUM_CONTEXT_TOKENS = 16_384
_DEFAULT_FRONTIER_CONTEXT_TOKENS = 48_000
_DEFAULT_OVERSIZED_CONTEXT_TOKENS = 96_000
_DEFAULT_SMALL_CONE = 8
_DEFAULT_MEDIUM_CONE = 32
_DEFAULT_LARGE_CONE = 128
_DEFAULT_MAX_PRIOR_FAILURES = 2
_DEFAULT_MAX_OBLIGATIONS_WITHOUT_PROOF = 0

_MAX_EXPLANATION_CHARS = 512
_MAX_REASON_CODES = 32


class ConfidenceClass(str, Enum):
    """Closed datasets confidence classifications."""

    EXACT = "exact"
    CONSERVATIVE = "conservative"
    HEURISTIC = "heuristic"
    OPAQUE = "opaque"


class RiskClass(str, Enum):
    """Closed risk classes used by routing."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DecisionStage(str, Enum):
    """The closed, canonical receipt-to-human decision ladder.

    The order of this enum is normative.  It deliberately lives beside the
    existing semantic-state router so consumers do not acquire a second
    routing authority merely to obtain a route receipt.
    """

    EXACT_CURRENT_AUTHORITATIVE_CACHED_RECEIPT = (
        "exact_current_authoritative_cached_receipt"
    )
    AST_SYMBOL_DEPENDENCY_AND_IMPACT_ANALYSIS = (
        "ast_symbol_dependency_and_impact_analysis"
    )
    SCHEMA_TYPE_STATIC_LINT_AND_CONTRACT_CHECKS = (
        "schema_type_static_lint_and_contract_checks"
    )
    SELECTED_TESTS = "selected_tests"
    INCREMENTAL_SMT_OR_THEOREM_PROVER = "incremental_smt_or_theorem_prover"
    LOCAL_SMALL_SPECIALIST_MODEL = "local_small_specialist_model"
    LOCAL_OR_REMOTE_MEDIUM_MODEL = "local_or_remote_medium_model"
    REMOTE_STRONG_OR_FRONTIER_MODEL = "remote_strong_or_frontier_model"
    HUMAN_DECISION = "human_decision"


class StageDisposition(str, Enum):
    """Whether a ladder stage actually ran or was explicitly skipped."""

    RUN = "run"
    SKIP = "skip"


class DeterministicEvidenceOutcome(str, Enum):
    """Closed results admitted from a deterministic stage."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    NOT_APPLICABLE = "not_applicable"


class LadderReason(str, Enum):
    """Typed reasons used by every run or skip receipt entry."""

    EVIDENCE_EVALUATED = "evidence_evaluated"
    DETERMINISTIC_EVIDENCE_RESOLVED = "deterministic_evidence_resolved"
    DETERMINISTIC_EVIDENCE_UNRESOLVED = "deterministic_evidence_unresolved"
    DETERMINISTIC_STAGE_NOT_APPLICABLE = "deterministic_stage_not_applicable"
    PRIOR_DETERMINISTIC_STAGE_RESOLVED = "prior_deterministic_stage_resolved"
    UNRESOLVED_QUESTION_REQUIRES_MODEL = "unresolved_question_requires_model"
    SMALLER_CAPABILITY_INSUFFICIENT = "smaller_capability_insufficient"
    SMALLEST_ADEQUATE_EXECUTOR_SELECTED = "smallest_adequate_executor_selected"
    HUMAN_REVIEW_SELECTED = "human_review_selected"
    HUMAN_REVIEW_NOT_REQUIRED = "human_review_not_required"


_DETERMINISTIC_STAGES: tuple[DecisionStage, ...] = (
    DecisionStage.EXACT_CURRENT_AUTHORITATIVE_CACHED_RECEIPT,
    DecisionStage.AST_SYMBOL_DEPENDENCY_AND_IMPACT_ANALYSIS,
    DecisionStage.SCHEMA_TYPE_STATIC_LINT_AND_CONTRACT_CHECKS,
    DecisionStage.SELECTED_TESTS,
    DecisionStage.INCREMENTAL_SMT_OR_THEOREM_PROVER,
)
_MODEL_STAGES: tuple[DecisionStage, ...] = (
    DecisionStage.LOCAL_SMALL_SPECIALIST_MODEL,
    DecisionStage.LOCAL_OR_REMOTE_MEDIUM_MODEL,
    DecisionStage.REMOTE_STRONG_OR_FRONTIER_MODEL,
)
_MODEL_STAGE_FOR_ROUTE: Mapping[str, DecisionStage] = {
    ModelRoute.SMALL_LOCAL_MODEL.value: DecisionStage.LOCAL_SMALL_SPECIALIST_MODEL,
    ModelRoute.MEDIUM_MODEL.value: DecisionStage.LOCAL_OR_REMOTE_MEDIUM_MODEL,
    ModelRoute.FRONTIER_MODEL.value: DecisionStage.REMOTE_STRONG_OR_FRONTIER_MODEL,
}


def _ladder_text(value: Any, field: str) -> str:
    return _text(value, field).strip()


def _ladder_texts(value: Any, field: str, *, require_one: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise HarnessError(f"{field} must be a list")
    result = tuple(_ladder_text(item, field) for item in value)
    if require_one and not result:
        raise HarnessError(f"{field} must not be empty")
    return result


@dataclass(frozen=True)
class DeterministicStageEvidence:
    """One observed result for a deterministic stage in the canonical order."""

    stage: DecisionStage
    outcome: DeterministicEvidenceOutcome
    evidence: tuple[str, ...]

    _FIELDS = frozenset({"stage", "outcome", "evidence"})

    def __post_init__(self) -> None:
        stage = _enum(self.stage, DecisionStage, "stage")
        if stage not in _DETERMINISTIC_STAGES:
            raise HarnessError("deterministic evidence stage must be a deterministic stage")
        object.__setattr__(self, "stage", DecisionStage(stage))
        object.__setattr__(
            self,
            "outcome",
            DeterministicEvidenceOutcome(
                _enum(self.outcome, DeterministicEvidenceOutcome, "outcome")
            ),
        )
        object.__setattr__(self, "evidence", _ladder_texts(self.evidence, "evidence", require_one=True))

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage.value, "outcome": self.outcome.value, "evidence": list(self.evidence)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeterministicStageEvidence":
        payload = _closed(data, cls._FIELDS, "DeterministicStageEvidence")
        return cls(stage=payload["stage"], outcome=payload["outcome"], evidence=payload["evidence"])


@dataclass(frozen=True)
class UnresolvedQuestion:
    """Decision-relevant question required before any model stage may run."""

    question_id: str
    exact_question: str
    why_prior_deterministic_stages_could_not_resolve: tuple[str, ...]
    evidence_available: tuple[str, ...]
    evidence_missing: tuple[str, ...]
    candidate_decisions_answer_could_change: tuple[str, ...]
    minimum_model_capability: str
    context_budget: int
    response_schema: str
    deadline: str
    cost_budget: int

    _FIELDS = frozenset(
        {
            "question_id", "exact_question", "why_prior_deterministic_stages_could_not_resolve",
            "evidence_available", "evidence_missing", "candidate_decisions_answer_could_change",
            "minimum_model_capability", "context_budget", "response_schema", "deadline", "cost_budget",
        }
    )

    def __post_init__(self) -> None:
        for name in ("question_id", "exact_question", "response_schema", "deadline"):
            object.__setattr__(self, name, _ladder_text(getattr(self, name), name))
        for name, require_one in (
            ("why_prior_deterministic_stages_could_not_resolve", True),
            ("evidence_available", False),
            ("evidence_missing", True),
            ("candidate_decisions_answer_could_change", True),
        ):
            object.__setattr__(self, name, _ladder_texts(getattr(self, name), name, require_one=require_one))
        candidates = self.candidate_decisions_answer_could_change
        if len(candidates) < 2 or len(set(candidates)) != len(candidates):
            raise HarnessError(
                "candidate_decisions_answer_could_change must contain two distinct decisions"
            )
        capability = _enum(self.minimum_model_capability, ModelRoute, "minimum_model_capability")
        if capability not in _MODEL_STAGE_FOR_ROUTE:
            raise HarnessError("minimum_model_capability must be a model route")
        object.__setattr__(self, "minimum_model_capability", capability)
        object.__setattr__(self, "context_budget", _nonneg_int(self.context_budget, "context_budget"))
        object.__setattr__(self, "cost_budget", _nonneg_int(self.cost_budget, "cost_budget"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id, "exact_question": self.exact_question,
            "why_prior_deterministic_stages_could_not_resolve": list(self.why_prior_deterministic_stages_could_not_resolve),
            "evidence_available": list(self.evidence_available), "evidence_missing": list(self.evidence_missing),
            "candidate_decisions_answer_could_change": list(self.candidate_decisions_answer_could_change),
            "minimum_model_capability": self.minimum_model_capability, "context_budget": self.context_budget,
            "response_schema": self.response_schema, "deadline": self.deadline, "cost_budget": self.cost_budget,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UnresolvedQuestion":
        return cls(**_closed(data, cls._FIELDS, "UnresolvedQuestion"))


@dataclass(frozen=True)
class StageReceipt:
    """A single ordered stage receipt with a typed run/skip reason."""

    stage: DecisionStage
    disposition: StageDisposition
    reason: LadderReason
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", DecisionStage(_enum(self.stage, DecisionStage, "stage")))
        object.__setattr__(self, "disposition", StageDisposition(_enum(self.disposition, StageDisposition, "disposition")))
        object.__setattr__(self, "reason", LadderReason(_enum(self.reason, LadderReason, "reason")))
        object.__setattr__(self, "evidence", _ladder_texts(self.evidence, "evidence"))

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage.value, "disposition": self.disposition.value, "reason": self.reason.value, "evidence": list(self.evidence)}


@dataclass(frozen=True)
class RouteReceipt:
    """Closed receipt emitted by the one canonical decision ladder."""

    stages: tuple[StageReceipt, ...]
    decisive_evidence: tuple[str, ...]
    escalation_reason: str
    selected_executor: str
    actual_resource_use: Mapping[str, int]
    final_outcome: str
    unresolved_question: UnresolvedQuestion | None = None

    _FIELDS = frozenset(
        {
            "stages_run", "stages_skipped_and_reason", "decisive_evidence",
            "escalation_reason", "selected_executor", "actual_resource_use",
            "final_outcome", "unresolved_question",
        }
    )

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        if tuple(item.stage for item in stages) != tuple(DecisionStage):
            raise HarnessError("route receipt must contain all nine stages in canonical order")
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "decisive_evidence", _ladder_texts(self.decisive_evidence, "decisive_evidence"))
        object.__setattr__(self, "escalation_reason", LadderReason(_enum(self.escalation_reason, LadderReason, "escalation_reason")).value)
        executor = _enum(self.selected_executor, ModelRoute, "selected_executor")
        object.__setattr__(self, "selected_executor", executor)
        raw_use = self.actual_resource_use
        if not isinstance(raw_use, Mapping):
            raise HarnessError("actual_resource_use must be an object")
        object.__setattr__(
            self,
            "actual_resource_use",
            {_ladder_text(key, "actual_resource_use key"): _nonneg_int(value, "actual_resource_use value") for key, value in raw_use.items()},
        )
        if self.final_outcome not in {"deterministic_resolved", "model_pending", "human_review_required"}:
            raise HarnessError("final_outcome has unsupported value")
        question = self.unresolved_question
        if isinstance(question, Mapping):
            question = UnresolvedQuestion.from_dict(question)
        if question is not None and not isinstance(question, UnresolvedQuestion):
            raise HarnessError("unresolved_question must be UnresolvedQuestion, mapping, or None")
        if executor in _MODEL_STAGE_FOR_ROUTE and question is None:
            raise HarnessError("model receipt requires unresolved_question")
        if executor not in _MODEL_STAGE_FOR_ROUTE and question is not None:
            raise HarnessError("non-model receipt must not carry unresolved_question")
        object.__setattr__(self, "unresolved_question", question)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages_run": [item.to_dict() for item in self.stages if item.disposition is StageDisposition.RUN],
            "stages_skipped_and_reason": [item.to_dict() for item in self.stages if item.disposition is StageDisposition.SKIP],
            "decisive_evidence": list(self.decisive_evidence), "escalation_reason": self.escalation_reason,
            "selected_executor": self.selected_executor, "actual_resource_use": dict(self.actual_resource_use),
            "final_outcome": self.final_outcome,
            "unresolved_question": None if self.unresolved_question is None else self.unresolved_question.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RouteReceipt":
        payload = _closed(data, cls._FIELDS, "RouteReceipt")
        receipts: list[StageReceipt] = []
        for field, disposition in (
            ("stages_run", StageDisposition.RUN),
            ("stages_skipped_and_reason", StageDisposition.SKIP),
        ):
            raw_items = payload[field]
            if not isinstance(raw_items, list):
                raise HarnessError(f"{field} must be a list")
            for raw in raw_items:
                if not isinstance(raw, Mapping) or set(raw) != {"stage", "disposition", "reason", "evidence"}:
                    raise HarnessError(f"{field} entries must be closed stage receipts")
                if raw["disposition"] != disposition.value:
                    raise HarnessError(f"{field} entry has incorrect disposition")
                receipts.append(StageReceipt(raw["stage"], raw["disposition"], raw["reason"], raw["evidence"]))
        ordered = tuple(sorted(receipts, key=lambda item: tuple(DecisionStage).index(item.stage)))
        return cls(
            stages=ordered,
            decisive_evidence=_ladder_texts(payload["decisive_evidence"], "decisive_evidence"),
            escalation_reason=payload["escalation_reason"], selected_executor=payload["selected_executor"],
            actual_resource_use=payload["actual_resource_use"], final_outcome=_ladder_text(payload["final_outcome"], "final_outcome"),
            unresolved_question=payload["unresolved_question"],
        )


@dataclass(frozen=True)
class DeterministicLadderRequest:
    """Closed inputs to the canonical nine-stage decision ladder."""

    deterministic_evidence: tuple[DeterministicStageEvidence, ...]
    proposed_route: str
    unresolved_question: UnresolvedQuestion | None = None
    actual_resource_use: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        evidence = tuple(self.deterministic_evidence)
        if not evidence or len(evidence) > len(_DETERMINISTIC_STAGES):
            raise HarnessError("deterministic_evidence must contain one through five ordered stages")
        for index, item in enumerate(evidence):
            if not isinstance(item, DeterministicStageEvidence):
                raise HarnessError("deterministic_evidence items must be DeterministicStageEvidence")
            if item.stage is not _DETERMINISTIC_STAGES[index]:
                raise HarnessError("deterministic_evidence must follow the canonical stage order")
        resolved = next((index for index, item in enumerate(evidence) if item.outcome is DeterministicEvidenceOutcome.RESOLVED), None)
        if resolved is None and len(evidence) != len(_DETERMINISTIC_STAGES):
            raise HarnessError("all deterministic stages must run before escalation")
        if resolved is not None and len(evidence) != resolved + 1:
            raise HarnessError("stages after deterministic resolution must be skipped, not supplied")
        object.__setattr__(self, "deterministic_evidence", evidence)
        route = _enum(self.proposed_route, ModelRoute, "proposed_route")
        object.__setattr__(self, "proposed_route", route)
        question = self.unresolved_question
        if isinstance(question, Mapping):
            question = UnresolvedQuestion.from_dict(question)
        if question is not None and not isinstance(question, UnresolvedQuestion):
            raise HarnessError("unresolved_question must be UnresolvedQuestion, mapping, or None")
        object.__setattr__(self, "unresolved_question", question)
        raw_use = self.actual_resource_use or {}
        if not isinstance(raw_use, Mapping):
            raise HarnessError("actual_resource_use must be an object")
        use = { _ladder_text(key, "actual_resource_use key"): _nonneg_int(value, "actual_resource_use value") for key, value in raw_use.items() }
        object.__setattr__(self, "actual_resource_use", use)


def route_decision_ladder(request: DeterministicLadderRequest | Mapping[str, Any]) -> RouteReceipt:
    """Run the exact receipt-to-human ladder and return its closed receipt.

    This is the sole authority for ladder ordering.  A model route is accepted
    only after every deterministic stage is observed unresolved/not-applicable
    and a typed, decision-changing :class:`UnresolvedQuestion` is present.
    Availability is intentionally absent from this API: it can never justify
    skipping a deterministic stage or choosing a larger executor.
    """

    if isinstance(request, Mapping):
        allowed = {"deterministic_evidence", "proposed_route", "unresolved_question", "actual_resource_use"}
        payload = _closed(request, frozenset(allowed), "DeterministicLadderRequest")
        request = DeterministicLadderRequest(
            deterministic_evidence=tuple(DeterministicStageEvidence.from_dict(item) for item in payload["deterministic_evidence"]),
            proposed_route=payload["proposed_route"], unresolved_question=payload.get("unresolved_question"),
            actual_resource_use=payload.get("actual_resource_use"),
        )
    elif not isinstance(request, DeterministicLadderRequest):
        raise HarnessError("request must be DeterministicLadderRequest or mapping")

    stages: list[StageReceipt] = []
    decisive: tuple[str, ...] = ()
    resolved_index: int | None = None
    for index, item in enumerate(request.deterministic_evidence):
        reason = {
            DeterministicEvidenceOutcome.RESOLVED: LadderReason.DETERMINISTIC_EVIDENCE_RESOLVED,
            DeterministicEvidenceOutcome.UNRESOLVED: LadderReason.DETERMINISTIC_EVIDENCE_UNRESOLVED,
            DeterministicEvidenceOutcome.NOT_APPLICABLE: LadderReason.DETERMINISTIC_STAGE_NOT_APPLICABLE,
        }[item.outcome]
        stages.append(StageReceipt(item.stage, StageDisposition.RUN, reason, item.evidence))
        if item.outcome is DeterministicEvidenceOutcome.RESOLVED:
            resolved_index = index
            decisive = item.evidence
            break
    if resolved_index is not None:
        for stage in _DETERMINISTIC_STAGES[resolved_index + 1:] + _MODEL_STAGES + (DecisionStage.HUMAN_DECISION,):
            stages.append(StageReceipt(stage, StageDisposition.SKIP, LadderReason.PRIOR_DETERMINISTIC_STAGE_RESOLVED))
        return RouteReceipt(tuple(stages), decisive, LadderReason.DETERMINISTIC_EVIDENCE_RESOLVED.value,
                            ModelRoute.DETERMINISTIC_ONLY.value, request.actual_resource_use or {}, "deterministic_resolved")

    if request.proposed_route == ModelRoute.DETERMINISTIC_ONLY.value:
        raise HarnessError("deterministic_only requires resolving deterministic evidence")
    if request.proposed_route in _MODEL_STAGE_FOR_ROUTE:
        question = request.unresolved_question
        if question is None:
            raise HarnessError("a typed unresolved_question is required before any model route")
        required_index = _MODEL_STAGES.index(_MODEL_STAGE_FOR_ROUTE[question.minimum_model_capability])
        selected_index = _MODEL_STAGES.index(_MODEL_STAGE_FOR_ROUTE[request.proposed_route])
        if selected_index != required_index:
            raise HarnessError(
                "proposed_route must equal unresolved_question minimum_model_capability"
            )
        for index, stage in enumerate(_MODEL_STAGES):
            if index < selected_index:
                stages.append(StageReceipt(stage, StageDisposition.SKIP, LadderReason.SMALLER_CAPABILITY_INSUFFICIENT))
            elif index == selected_index:
                stages.append(StageReceipt(stage, StageDisposition.RUN, LadderReason.UNRESOLVED_QUESTION_REQUIRES_MODEL))
            else:
                stages.append(StageReceipt(stage, StageDisposition.SKIP, LadderReason.SMALLEST_ADEQUATE_EXECUTOR_SELECTED))
        stages.append(StageReceipt(DecisionStage.HUMAN_DECISION, StageDisposition.SKIP, LadderReason.HUMAN_REVIEW_NOT_REQUIRED))
        return RouteReceipt(tuple(stages), tuple(question.evidence_missing), LadderReason.UNRESOLVED_QUESTION_REQUIRES_MODEL.value,
                            request.proposed_route, request.actual_resource_use or {}, "model_pending", question)

    if request.proposed_route == ModelRoute.HUMAN_REVIEW_REQUIRED.value:
        for stage in _MODEL_STAGES:
            stages.append(StageReceipt(stage, StageDisposition.SKIP, LadderReason.HUMAN_REVIEW_SELECTED))
        stages.append(StageReceipt(DecisionStage.HUMAN_DECISION, StageDisposition.RUN, LadderReason.HUMAN_REVIEW_SELECTED))
        return RouteReceipt(tuple(stages), (), LadderReason.HUMAN_REVIEW_SELECTED.value,
                            ModelRoute.HUMAN_REVIEW_REQUIRED.value, request.actual_resource_use or {}, "human_review_required")
    raise HarnessError("unsupported proposed_route")


def _clip(text: str, *, maximum: int = _MAX_EXPLANATION_CHARS) -> str:
    value = str(text or "").strip() or "unspecified"
    if len(value) > maximum:
        return value[: maximum - 3] + "..."
    return value


def _normalize_confidence(value: Any) -> str:
    text = _text(value, "lowest_confidence").casefold()
    if text not in _CONFIDENCE_RANK:
        raise HarnessError(
            f"lowest_confidence has unsupported value {value!r}; "
            f"expected one of {sorted(_CONFIDENCE_RANK)}"
        )
    return text


def _normalize_risk(value: Any) -> str:
    text = _text(value, "risk").casefold()
    if text not in _RISK_RANK:
        raise HarnessError(
            f"risk has unsupported value {value!r}; "
            f"expected one of {sorted(_RISK_RANK)}"
        )
    return text


def _sorted_reason_codes(codes: Sequence[str]) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in codes:
        code = str(item or "").strip().casefold().replace(" ", "_")
        if not code or code in seen:
            continue
        seen.add(code)
        cleaned.append(code)
        if len(cleaned) >= _MAX_REASON_CODES:
            break
    return tuple(sorted(cleaned))


@dataclass(frozen=True)
class ModelRoutingPolicy:
    """Deterministic thresholds for model routing decisions.

    Thresholds are absolute and ordered. Identical inputs always produce the
    same route and explanation under the same policy.
    """

    small_context_tokens: int = _DEFAULT_SMALL_CONTEXT_TOKENS
    medium_context_tokens: int = _DEFAULT_MEDIUM_CONTEXT_TOKENS
    frontier_context_tokens: int = _DEFAULT_FRONTIER_CONTEXT_TOKENS
    oversized_context_tokens: int = _DEFAULT_OVERSIZED_CONTEXT_TOKENS
    small_dependency_cone: int = _DEFAULT_SMALL_CONE
    medium_dependency_cone: int = _DEFAULT_MEDIUM_CONE
    large_dependency_cone: int = _DEFAULT_LARGE_CONE
    max_prior_failures_before_human: int = _DEFAULT_MAX_PRIOR_FAILURES
    max_unresolved_obligations_without_proof: int = (
        _DEFAULT_MAX_OBLIGATIONS_WITHOUT_PROOF
    )

    _FIELDS = frozenset(
        {
            "small_context_tokens",
            "medium_context_tokens",
            "frontier_context_tokens",
            "oversized_context_tokens",
            "small_dependency_cone",
            "medium_dependency_cone",
            "large_dependency_cone",
            "max_prior_failures_before_human",
            "max_unresolved_obligations_without_proof",
        }
    )

    def __post_init__(self) -> None:
        for name in (
            "small_context_tokens",
            "medium_context_tokens",
            "frontier_context_tokens",
            "oversized_context_tokens",
            "small_dependency_cone",
            "medium_dependency_cone",
            "large_dependency_cone",
            "max_prior_failures_before_human",
            "max_unresolved_obligations_without_proof",
        ):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool) or value < 0:
                raise HarnessError(f"{name} must be a nonnegative integer")
        if not (
            self.small_context_tokens
            <= self.medium_context_tokens
            <= self.frontier_context_tokens
            <= self.oversized_context_tokens
        ):
            raise HarnessError(
                "context token thresholds must be nondecreasing: "
                "small <= medium <= frontier <= oversized"
            )
        if not (
            self.small_dependency_cone
            <= self.medium_dependency_cone
            <= self.large_dependency_cone
        ):
            raise HarnessError(
                "dependency cone thresholds must be nondecreasing: "
                "small <= medium <= large"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "small_context_tokens": self.small_context_tokens,
            "medium_context_tokens": self.medium_context_tokens,
            "frontier_context_tokens": self.frontier_context_tokens,
            "oversized_context_tokens": self.oversized_context_tokens,
            "small_dependency_cone": self.small_dependency_cone,
            "medium_dependency_cone": self.medium_dependency_cone,
            "large_dependency_cone": self.large_dependency_cone,
            "max_prior_failures_before_human": self.max_prior_failures_before_human,
            "max_unresolved_obligations_without_proof": (
                self.max_unresolved_obligations_without_proof
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelRoutingPolicy":
        payload = _closed(data, cls._FIELDS, "ModelRoutingPolicy")
        return cls(
            small_context_tokens=_nonneg_int(
                payload["small_context_tokens"], "small_context_tokens"
            ),
            medium_context_tokens=_nonneg_int(
                payload["medium_context_tokens"], "medium_context_tokens"
            ),
            frontier_context_tokens=_nonneg_int(
                payload["frontier_context_tokens"], "frontier_context_tokens"
            ),
            oversized_context_tokens=_nonneg_int(
                payload["oversized_context_tokens"], "oversized_context_tokens"
            ),
            small_dependency_cone=_nonneg_int(
                payload["small_dependency_cone"], "small_dependency_cone"
            ),
            medium_dependency_cone=_nonneg_int(
                payload["medium_dependency_cone"], "medium_dependency_cone"
            ),
            large_dependency_cone=_nonneg_int(
                payload["large_dependency_cone"], "large_dependency_cone"
            ),
            max_prior_failures_before_human=_nonneg_int(
                payload["max_prior_failures_before_human"],
                "max_prior_failures_before_human",
            ),
            max_unresolved_obligations_without_proof=_nonneg_int(
                payload["max_unresolved_obligations_without_proof"],
                "max_unresolved_obligations_without_proof",
            ),
        )

    @classmethod
    def default(cls) -> "ModelRoutingPolicy":
        return cls()


@dataclass(frozen=True)
class RoutingInputs:
    """Closed scoring inputs for a single model-routing decision.

    Only the plan-declared dimensions are admitted. Secret values, prompts,
    and source bodies never enter routing observations.
    """

    context_tokens: int
    lowest_confidence: str
    risk: str
    dependency_cone_size: int
    unresolved_obligations: int
    prior_repair_failures: int
    available_proofs: int
    prior_route_failed: bool = False

    _FIELDS = frozenset(
        {
            "context_tokens",
            "lowest_confidence",
            "risk",
            "dependency_cone_size",
            "unresolved_obligations",
            "prior_repair_failures",
            "available_proofs",
            "prior_route_failed",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_tokens": self.context_tokens,
            "lowest_confidence": self.lowest_confidence,
            "risk": self.risk,
            "dependency_cone_size": self.dependency_cone_size,
            "unresolved_obligations": self.unresolved_obligations,
            "prior_repair_failures": self.prior_repair_failures,
            "available_proofs": self.available_proofs,
            "prior_route_failed": self.prior_route_failed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoutingInputs":
        payload = _closed(data, cls._FIELDS, "RoutingInputs")
        return cls(
            context_tokens=_nonneg_int(payload["context_tokens"], "context_tokens"),
            lowest_confidence=_normalize_confidence(payload["lowest_confidence"]),
            risk=_normalize_risk(payload["risk"]),
            dependency_cone_size=_nonneg_int(
                payload["dependency_cone_size"], "dependency_cone_size"
            ),
            unresolved_obligations=_nonneg_int(
                payload["unresolved_obligations"], "unresolved_obligations"
            ),
            prior_repair_failures=_nonneg_int(
                payload["prior_repair_failures"], "prior_repair_failures"
            ),
            available_proofs=_nonneg_int(
                payload["available_proofs"], "available_proofs"
            ),
            prior_route_failed=_bool(
                payload["prior_route_failed"], "prior_route_failed"
            ),
        )

    @classmethod
    def from_context_pack(
        cls,
        pack: ContextPack | Mapping[str, Any],
        *,
        lowest_confidence: str,
        dependency_cone_size: int,
        prior_repair_failures: int = 0,
        available_proofs: int = 0,
        prior_route_failed: bool = False,
    ) -> "RoutingInputs":
        """Project a ContextPack plus closed assurance fields into routing inputs."""

        if isinstance(pack, ContextPack):
            token_totals = dict(pack.token_totals)
            risk = pack.risk
            unresolved = len(pack.obligation_cids)
        else:
            if not isinstance(pack, Mapping):
                raise HarnessError("context pack must be ContextPack or mapping")
            totals = pack.get("token_totals")
            if not isinstance(totals, Mapping):
                raise HarnessError("token_totals must be an object")
            token_totals = {
                str(key): int(value)
                for key, value in totals.items()
                if type(value) is int and not isinstance(value, bool) and value >= 0
            }
            risk = str(pack.get("risk") or "")
            obligations = pack.get("obligation_cids") or ()
            if not isinstance(obligations, (list, tuple)):
                raise HarnessError("obligation_cids must be a list")
            unresolved = len(obligations)

        context_tokens = int(token_totals.get("total", 0))
        if context_tokens <= 0:
            # Fall back to the sum of category totals when "total" is absent.
            context_tokens = sum(
                int(value)
                for key, value in token_totals.items()
                if key != "total" and type(value) is int and not isinstance(value, bool)
            )
        return cls.from_dict(
            {
                "context_tokens": context_tokens,
                "lowest_confidence": lowest_confidence,
                "risk": risk,
                "dependency_cone_size": dependency_cone_size,
                "unresolved_obligations": unresolved,
                "prior_repair_failures": prior_repair_failures,
                "available_proofs": available_proofs,
                "prior_route_failed": prior_route_failed,
            }
        )


@dataclass(frozen=True)
class RoutingDecision:
    """Deterministic, explained model-routing result."""

    route: str
    reason_codes: tuple[str, ...]
    explanation: str
    requires_provider: bool
    halt_before_dispatch: bool
    halt_before_root_publication: bool
    inputs: RoutingInputs
    policy: ModelRoutingPolicy

    _FIELDS = frozenset(
        {
            "route",
            "reason_codes",
            "explanation",
            "requires_provider",
            "halt_before_dispatch",
            "halt_before_root_publication",
            "inputs",
            "policy",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "reason_codes": list(self.reason_codes),
            "explanation": self.explanation,
            "requires_provider": self.requires_provider,
            "halt_before_dispatch": self.halt_before_dispatch,
            "halt_before_root_publication": self.halt_before_root_publication,
            "inputs": self.inputs.to_dict(),
            "policy": self.policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoutingDecision":
        payload = _closed(data, cls._FIELDS, "RoutingDecision")
        inputs_raw = payload["inputs"]
        policy_raw = payload["policy"]
        if not isinstance(inputs_raw, Mapping):
            raise HarnessError("inputs must be an object")
        if not isinstance(policy_raw, Mapping):
            raise HarnessError("policy must be an object")
        route = _enum(payload["route"], ModelRoute, "route")
        halt_dispatch = _bool(
            payload["halt_before_dispatch"], "halt_before_dispatch"
        )
        halt_publish = _bool(
            payload["halt_before_root_publication"],
            "halt_before_root_publication",
        )
        requires_provider = _bool(
            payload["requires_provider"], "requires_provider"
        )
        if route == ModelRoute.HUMAN_REVIEW_REQUIRED.value:
            if not halt_dispatch or not halt_publish:
                raise HarnessError(
                    "human_review_required must halt before dispatch and root publication"
                )
            if requires_provider:
                raise HarnessError(
                    "human_review_required must not require a provider"
                )
        if route == ModelRoute.DETERMINISTIC_ONLY.value and requires_provider:
            raise HarnessError("deterministic_only must not require a provider")
        if route not in {
            ModelRoute.DETERMINISTIC_ONLY.value,
            ModelRoute.HUMAN_REVIEW_REQUIRED.value,
        } and not requires_provider:
            raise HarnessError(f"{route} requires a provider")
        return cls(
            route=route,
            reason_codes=_sorted_reason_codes(
                payload["reason_codes"]
                if isinstance(payload["reason_codes"], list)
                else ()
            ),
            explanation=_clip(_text(payload["explanation"], "explanation")),
            requires_provider=requires_provider,
            halt_before_dispatch=halt_dispatch,
            halt_before_root_publication=halt_publish,
            inputs=RoutingInputs.from_dict(inputs_raw),
            policy=ModelRoutingPolicy.from_dict(policy_raw),
        )

    @property
    def may_invoke_provider(self) -> bool:
        return (
            self.requires_provider
            and not self.halt_before_dispatch
            and self.route
            not in {
                ModelRoute.DETERMINISTIC_ONLY.value,
                ModelRoute.HUMAN_REVIEW_REQUIRED.value,
            }
        )


def _human_review_reasons(
    inputs: RoutingInputs, policy: ModelRoutingPolicy
) -> list[str]:
    reasons: list[str] = []
    if inputs.risk in {RiskClass.HIGH.value, RiskClass.CRITICAL.value}:
        reasons.append(f"risk_{inputs.risk}")
    if inputs.lowest_confidence == ConfidenceClass.OPAQUE.value:
        reasons.append("confidence_opaque")
    if inputs.context_tokens > policy.oversized_context_tokens:
        reasons.append("context_oversized")
    if inputs.prior_repair_failures > policy.max_prior_failures_before_human:
        reasons.append("prior_repair_failures_exceeded")
    if inputs.prior_route_failed:
        reasons.append("prior_route_failed")
    if (
        inputs.unresolved_obligations
        > policy.max_unresolved_obligations_without_proof
        and inputs.available_proofs <= 0
        and inputs.risk
        in {RiskClass.HIGH.value, RiskClass.CRITICAL.value}
        and inputs.lowest_confidence
        in {ConfidenceClass.HEURISTIC.value, ConfidenceClass.OPAQUE.value}
    ):
        # High-risk assurance gap with weak confidence escalates rather than
        # silently guessing. Medium-risk gaps still route to a model class.
        reasons.append("unresolved_obligations_without_proof")
    if inputs.dependency_cone_size > policy.large_dependency_cone and (
        inputs.risk in {RiskClass.HIGH.value, RiskClass.CRITICAL.value}
        or inputs.lowest_confidence == ConfidenceClass.OPAQUE.value
    ):
        reasons.append("dependency_cone_too_large")
    return reasons


def _deterministic_only_eligible(
    inputs: RoutingInputs, policy: ModelRoutingPolicy
) -> bool:
    if inputs.prior_route_failed:
        return False
    if inputs.prior_repair_failures > 0:
        return False
    if inputs.risk != RiskClass.LOW.value:
        return False
    if inputs.lowest_confidence not in {
        ConfidenceClass.EXACT.value,
        ConfidenceClass.CONSERVATIVE.value,
    }:
        return False
    if inputs.context_tokens > policy.small_context_tokens:
        return False
    if inputs.dependency_cone_size > policy.small_dependency_cone:
        return False
    if inputs.unresolved_obligations > 0 and inputs.available_proofs <= 0:
        return False
    # Prefer deterministic work when proofs already cover remaining obligations
    # or there is nothing open to prove.
    if inputs.unresolved_obligations > 0:
        return inputs.available_proofs >= inputs.unresolved_obligations
    return True


def _score_model_route(
    inputs: RoutingInputs, policy: ModelRoutingPolicy
) -> tuple[str, list[str]]:
    """Return a non-human, non-deterministic model route and reason codes."""

    reasons: list[str] = []
    conf_rank = _CONFIDENCE_RANK[inputs.lowest_confidence]
    risk_rank = _RISK_RANK[inputs.risk]

    # Size pressure.
    size_score = 0
    if inputs.context_tokens > policy.frontier_context_tokens:
        size_score = 3
        reasons.append("context_frontier_size")
    elif inputs.context_tokens > policy.medium_context_tokens:
        size_score = 2
        reasons.append("context_medium_size")
    elif inputs.context_tokens > policy.small_context_tokens:
        size_score = 1
        reasons.append("context_small_exceeded")
    else:
        reasons.append("context_within_small")

    cone_score = 0
    if inputs.dependency_cone_size > policy.medium_dependency_cone:
        cone_score = 2
        reasons.append("dependency_cone_medium_exceeded")
    elif inputs.dependency_cone_size > policy.small_dependency_cone:
        cone_score = 1
        reasons.append("dependency_cone_small_exceeded")
    else:
        reasons.append("dependency_cone_within_small")

    conf_score = 0
    if conf_rank <= _CONFIDENCE_RANK[ConfidenceClass.HEURISTIC.value]:
        conf_score = 1
        reasons.append(f"confidence_{inputs.lowest_confidence}")
    else:
        reasons.append(f"confidence_{inputs.lowest_confidence}")

    risk_score = 0
    if risk_rank >= _RISK_RANK[RiskClass.MEDIUM.value]:
        risk_score = 1
        reasons.append(f"risk_{inputs.risk}")
    else:
        reasons.append(f"risk_{inputs.risk}")

    obligation_score = 0
    if inputs.unresolved_obligations > 0:
        obligation_score = 1
        reasons.append("unresolved_obligations_present")
        if inputs.available_proofs <= 0:
            reasons.append("proofs_absent")
        else:
            reasons.append("proofs_available")

    total = size_score + cone_score + conf_score + risk_score + obligation_score
    if total >= 4 or size_score >= 3:
        return ModelRoute.FRONTIER_MODEL.value, reasons + ["score_frontier"]
    if total >= 2 or size_score >= 2 or risk_score >= 1:
        return ModelRoute.MEDIUM_MODEL.value, reasons + ["score_medium"]
    return ModelRoute.SMALL_LOCAL_MODEL.value, reasons + ["score_small_local"]


def route_model(
    inputs: RoutingInputs | Mapping[str, Any],
    *,
    policy: ModelRoutingPolicy | Mapping[str, Any] | None = None,
) -> RoutingDecision:
    """Compute a deterministic, explained model route.

    Escalation order is fail-closed:

    1. high-risk / opaque / oversized / failed cases → ``human_review_required``
    2. exact/conservative low-risk proof-covered cases → ``deterministic_only``
    3. otherwise score into small / medium / frontier model classes
    """

    if isinstance(inputs, Mapping):
        inputs = RoutingInputs.from_dict(inputs)
    elif not isinstance(inputs, RoutingInputs):
        raise HarnessError("inputs must be RoutingInputs or mapping")

    if policy is None:
        policy_obj = ModelRoutingPolicy.default()
    elif isinstance(policy, Mapping):
        policy_obj = ModelRoutingPolicy.from_dict(policy)
    elif isinstance(policy, ModelRoutingPolicy):
        policy_obj = policy
    else:
        raise HarnessError("policy must be ModelRoutingPolicy, mapping, or None")

    human_reasons = _human_review_reasons(inputs, policy_obj)
    if human_reasons:
        explanation = _clip(
            "human review required: " + ", ".join(sorted(human_reasons))
        )
        return RoutingDecision(
            route=ModelRoute.HUMAN_REVIEW_REQUIRED.value,
            reason_codes=_sorted_reason_codes(
                ["human_review_required", *human_reasons]
            ),
            explanation=explanation,
            requires_provider=False,
            halt_before_dispatch=True,
            halt_before_root_publication=True,
            inputs=inputs,
            policy=policy_obj,
        )

    if _deterministic_only_eligible(inputs, policy_obj):
        reasons = [
            "deterministic_only",
            f"confidence_{inputs.lowest_confidence}",
            f"risk_{inputs.risk}",
            "proofs_cover_or_absent_obligations",
        ]
        return RoutingDecision(
            route=ModelRoute.DETERMINISTIC_ONLY.value,
            reason_codes=_sorted_reason_codes(reasons),
            explanation=_clip(
                "deterministic_only: exact/conservative low-risk work with "
                "proof coverage and small cone/context"
            ),
            requires_provider=False,
            halt_before_dispatch=True,
            halt_before_root_publication=False,
            inputs=inputs,
            policy=policy_obj,
        )

    route, reasons = _score_model_route(inputs, policy_obj)
    explanation = _clip(f"{route}: " + ", ".join(reasons))
    return RoutingDecision(
        route=route,
        reason_codes=_sorted_reason_codes([route, *reasons]),
        explanation=explanation,
        requires_provider=True,
        halt_before_dispatch=False,
        halt_before_root_publication=False,
        inputs=inputs,
        policy=policy_obj,
    )


def route_requires_human_review(decision: RoutingDecision | Mapping[str, Any]) -> bool:
    if isinstance(decision, Mapping):
        decision = RoutingDecision.from_dict(decision)
    return decision.route == ModelRoute.HUMAN_REVIEW_REQUIRED.value


def route_allows_provider_dispatch(
    decision: RoutingDecision | Mapping[str, Any],
) -> bool:
    if isinstance(decision, Mapping):
        decision = RoutingDecision.from_dict(decision)
    return decision.may_invoke_provider


def model_routing_descriptor() -> dict[str, Any]:
    """Closed interface metadata for ModelRouting@1."""

    return {
        "interface": MODEL_ROUTING_INTERFACE,
        "schema": MODEL_ROUTING_SCHEMA,
        "board_namespace": BOARD_NAMESPACE,
        "adapter_id": ADAPTER_ID,
        "routes": [item.value for item in ModelRoute],
        "confidence_classes": [item.value for item in ConfidenceClass],
        "risk_classes": [item.value for item in RiskClass],
        "records": [
            "ModelRoutingPolicy",
            "RoutingInputs",
            "RoutingDecision",
            "DeterministicStageEvidence",
            "UnresolvedQuestion",
            "RouteReceipt",
        ],
        "scoring_inputs": [
            "context_size",
            "lowest_relevant_confidence",
            "risk_class",
            "affected_dependency_cone",
            "unresolved_obligations",
            "prior_repair_failures",
            "available_proofs",
        ],
        "invariants": [
            "route_decision_is_deterministic_and_explained",
            "human_review_required_halts_before_dispatch_and_root_publication",
            "deterministic_only_never_invokes_a_provider",
            "high_risk_opaque_oversized_failed_cases_escalate",
            "providers_are_never_hardcoded",
            "nine_stage_ladder_runs_or_skips_every_stage_in_order",
            "model_escalation_requires_a_decision_relevant_unresolved_question",
            "deterministic_evidence_resolves_eligible_cases_before_models",
        ],
    }


__all__ = [
    "ADAPTER_ID",
    "BOARD_NAMESPACE",
    "ConfidenceClass",
    "DecisionStage",
    "DeterministicEvidenceOutcome",
    "DeterministicLadderRequest",
    "DeterministicStageEvidence",
    "LadderReason",
    "MODEL_ROUTING_INTERFACE",
    "MODEL_ROUTING_SCHEMA",
    "ModelRoute",
    "ModelRoutingPolicy",
    "RiskClass",
    "RouteReceipt",
    "StageDisposition",
    "StageReceipt",
    "UnresolvedQuestion",
    "RoutingDecision",
    "RoutingInputs",
    "model_routing_descriptor",
    "route_allows_provider_dispatch",
    "route_decision_ladder",
    "route_model",
    "route_requires_human_review",
]
