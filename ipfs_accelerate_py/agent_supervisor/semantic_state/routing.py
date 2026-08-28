"""Deterministic model routing for the semantic-compression harness.

``routing.py`` is the canonical receipt-to-human routing authority. It scores
only declared inputs: context size, lowest relevant confidence, risk class,
affected dependency cone, unresolved obligations, prior repair failures, and
available proofs. ``route_decision_ladder`` records the exact nine ordered
stages; ``route_model`` remains its compatibility-level capability scorer.

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


class LadderStage(str, Enum):
    """The closed, ordered receipt-to-human routing ladder.

    These values are intentionally capability-oriented rather than tied to a
    provider.  ``route_decision_ladder`` is the sole evaluator for this
    sequence; callers receive a receipt for every value on every decision.
    """

    AUTHORITATIVE_RECEIPT = "authoritative_receipt"
    AST_SYMBOL_DEPENDENCY_IMPACT = "ast_symbol_dependency_impact"
    SCHEMA_TYPE_STATIC_LINT_CONTRACT = "schema_type_static_lint_contract"
    SELECTED_TESTS = "selected_tests"
    INCREMENTAL_PROVER = "incremental_prover"
    LOCAL_SMALL_SPECIALIST = "local_small_specialist"
    LOCAL_REMOTE_MEDIUM = "local_remote_medium"
    REMOTE_FRONTIER = "remote_frontier"
    HUMAN_REVIEW = "human_review"


class StageAction(str, Enum):
    """Whether a ladder stage was evaluated or explicitly bypassed."""

    RUN = "run"
    SKIP = "skip"


class StageReason(str, Enum):
    """Closed reasons carried by every ladder-stage receipt."""

    DECISIVE_EVIDENCE = "decisive_evidence"
    INCONCLUSIVE_EVIDENCE = "inconclusive_evidence"
    EVIDENCE_UNAVAILABLE = "evidence_unavailable"
    NOT_APPLICABLE = "not_applicable"
    NO_EVIDENCE = "no_evidence"
    PRIOR_STAGE_RESOLVED = "prior_stage_resolved"
    DETERMINISTIC_ROUTE_ELIGIBLE = "deterministic_route_eligible"
    ROUTE_NOT_ELIGIBLE = "route_not_eligible"
    MODEL_STAGE_SELECTED = "model_stage_selected"
    UNRESOLVED_QUESTION_REQUIRED = "unresolved_question_required"
    QUESTION_NOT_DECISION_RELEVANT = "question_not_decision_relevant"
    HUMAN_REVIEW_REQUIRED = "human_review_required"


class EvidenceStatus(str, Enum):
    """Closed result of a deterministic-stage evidence attempt."""

    RESOLVED = "resolved"
    INCONCLUSIVE = "inconclusive"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


_LADDER_STAGES: tuple[LadderStage, ...] = tuple(LadderStage)
_DETERMINISTIC_STAGES: tuple[LadderStage, ...] = _LADDER_STAGES[:5]
_MODEL_STAGE_BY_ROUTE: Mapping[str, LadderStage] = {
    ModelRoute.SMALL_LOCAL_MODEL.value: LadderStage.LOCAL_SMALL_SPECIALIST,
    ModelRoute.MEDIUM_MODEL.value: LadderStage.LOCAL_REMOTE_MEDIUM,
    ModelRoute.FRONTIER_MODEL.value: LadderStage.REMOTE_FRONTIER,
}


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


@dataclass(frozen=True)
class DeterministicEvidence:
    """A bounded observation produced by one deterministic ladder stage.

    A ``resolved`` observation is decisive and prevents all later model
    stages from being selected.  The reference is deliberately an opaque,
    bounded label (for example an admitted receipt or test run identifier),
    rather than a prompt, source body, or provider response.
    """

    status: str
    reference: str = "unspecified"

    _FIELDS = frozenset({"status", "reference"})

    def __post_init__(self) -> None:
        status = _enum(self.status, EvidenceStatus, "status")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reference", _clip(_text(self.reference, "reference")))

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status, "reference": self.reference}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeterministicEvidence":
        payload = _closed(data, cls._FIELDS, "DeterministicEvidence")
        return cls(status=payload["status"], reference=payload["reference"])


@dataclass(frozen=True)
class UnresolvedQuestionRecord:
    """Closed, decision-relevant record required before model selection."""

    question_id: str
    question: str
    admissible_decisions: tuple[str, ...]
    answer_can_change_decision: bool

    _FIELDS = frozenset(
        {
            "question_id",
            "question",
            "admissible_decisions",
            "answer_can_change_decision",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "question_id", _clip(_text(self.question_id, "question_id")))
        object.__setattr__(self, "question", _clip(_text(self.question, "question")))
        decisions = self.admissible_decisions
        if isinstance(decisions, (str, bytes)) or not isinstance(decisions, tuple):
            raise HarnessError("admissible_decisions must be a tuple")
        normalized = tuple(
            _enum(item, ModelRoute, "admissible_decisions") for item in decisions
        )
        if len(set(normalized)) != len(normalized):
            raise HarnessError("admissible_decisions must not contain duplicates")
        if not normalized:
            raise HarnessError("admissible_decisions must not be empty")
        object.__setattr__(self, "admissible_decisions", normalized)
        object.__setattr__(
            self,
            "answer_can_change_decision",
            _bool(self.answer_can_change_decision, "answer_can_change_decision"),
        )

    @property
    def is_decision_relevant(self) -> bool:
        return self.answer_can_change_decision and len(self.admissible_decisions) >= 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "admissible_decisions": list(self.admissible_decisions),
            "answer_can_change_decision": self.answer_can_change_decision,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UnresolvedQuestionRecord":
        payload = _closed(data, cls._FIELDS, "UnresolvedQuestionRecord")
        raw_decisions = payload["admissible_decisions"]
        if isinstance(raw_decisions, (str, bytes)) or not isinstance(raw_decisions, list):
            raise HarnessError("admissible_decisions must be a list")
        return cls(
            question_id=payload["question_id"],
            question=payload["question"],
            admissible_decisions=tuple(raw_decisions),
            answer_can_change_decision=payload["answer_can_change_decision"],
        )


@dataclass(frozen=True)
class StageReceipt:
    """One typed run/skip receipt in the canonical decision ladder."""

    stage: str
    action: str
    reason: str
    decisive: bool = False
    evidence_reference: str = "unspecified"

    _FIELDS = frozenset(
        {"stage", "action", "reason", "decisive", "evidence_reference"}
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _enum(self.stage, LadderStage, "stage"))
        object.__setattr__(self, "action", _enum(self.action, StageAction, "action"))
        object.__setattr__(self, "reason", _enum(self.reason, StageReason, "reason"))
        object.__setattr__(self, "decisive", _bool(self.decisive, "decisive"))
        object.__setattr__(
            self,
            "evidence_reference",
            _clip(_text(self.evidence_reference, "evidence_reference")),
        )
        if self.decisive and self.action != StageAction.RUN.value:
            raise HarnessError("a decisive stage receipt must record action=run")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "action": self.action,
            "reason": self.reason,
            "decisive": self.decisive,
            "evidence_reference": self.evidence_reference,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageReceipt":
        payload = _closed(data, cls._FIELDS, "StageReceipt")
        return cls(
            stage=payload["stage"],
            action=payload["action"],
            reason=payload["reason"],
            decisive=payload["decisive"],
            evidence_reference=payload["evidence_reference"],
        )


@dataclass(frozen=True)
class DecisionLadderInputs:
    """Inputs to the single receipt-to-human routing authority."""

    routing_inputs: RoutingInputs
    deterministic_evidence: Mapping[str, DeterministicEvidence]
    unresolved_question: UnresolvedQuestionRecord | None = None

    _FIELDS = frozenset(
        {"routing_inputs", "deterministic_evidence", "unresolved_question"}
    )

    def __post_init__(self) -> None:
        if not isinstance(self.routing_inputs, RoutingInputs):
            raise HarnessError("routing_inputs must be RoutingInputs")
        if not isinstance(self.deterministic_evidence, Mapping):
            raise HarnessError("deterministic_evidence must be an object")
        normalized: dict[str, DeterministicEvidence] = {}
        for stage, evidence in self.deterministic_evidence.items():
            stage_token = _enum(stage, LadderStage, "deterministic_evidence stage")
            if stage_token not in {item.value for item in _DETERMINISTIC_STAGES}:
                raise HarnessError("deterministic_evidence is only valid for deterministic stages")
            if not isinstance(evidence, DeterministicEvidence):
                raise HarnessError("deterministic_evidence values must be DeterministicEvidence")
            normalized[stage_token] = evidence
        object.__setattr__(self, "deterministic_evidence", normalized)
        if self.unresolved_question is not None and not isinstance(
            self.unresolved_question, UnresolvedQuestionRecord
        ):
            raise HarnessError("unresolved_question must be UnresolvedQuestionRecord or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "routing_inputs": self.routing_inputs.to_dict(),
            "deterministic_evidence": {
                stage: evidence.to_dict()
                for stage, evidence in sorted(self.deterministic_evidence.items())
            },
            "unresolved_question": (
                None
                if self.unresolved_question is None
                else self.unresolved_question.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecisionLadderInputs":
        payload = _closed(data, cls._FIELDS, "DecisionLadderInputs")
        routing_inputs = payload["routing_inputs"]
        evidence = payload["deterministic_evidence"]
        question = payload["unresolved_question"]
        if not isinstance(routing_inputs, Mapping):
            raise HarnessError("routing_inputs must be an object")
        if not isinstance(evidence, Mapping):
            raise HarnessError("deterministic_evidence must be an object")
        normalized_evidence: dict[str, DeterministicEvidence] = {}
        for stage, item in evidence.items():
            if not isinstance(item, Mapping):
                raise HarnessError("deterministic_evidence values must be objects")
            normalized_evidence[str(stage)] = DeterministicEvidence.from_dict(item)
        if question is not None and not isinstance(question, Mapping):
            raise HarnessError("unresolved_question must be an object or null")
        return cls(
            routing_inputs=RoutingInputs.from_dict(routing_inputs),
            deterministic_evidence=normalized_evidence,
            unresolved_question=(
                None if question is None else UnresolvedQuestionRecord.from_dict(question)
            ),
        )


@dataclass(frozen=True)
class DecisionLadder:
    """Full ordered receipt and the selected executor for one decision."""

    selected_stage: str
    route: str
    stage_receipts: tuple[StageReceipt, ...]
    routing_decision: RoutingDecision

    _FIELDS = frozenset(
        {"selected_stage", "route", "stage_receipts", "routing_decision"}
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "selected_stage", _enum(self.selected_stage, LadderStage, "selected_stage")
        )
        object.__setattr__(self, "route", _enum(self.route, ModelRoute, "route"))
        if len(self.stage_receipts) != len(_LADDER_STAGES):
            raise HarnessError("stage_receipts must contain every ladder stage exactly once")
        expected = tuple(item.value for item in _LADDER_STAGES)
        actual = tuple(item.stage for item in self.stage_receipts)
        if actual != expected:
            raise HarnessError("stage_receipts must be in canonical ladder order")
        selected = [item for item in self.stage_receipts if item.stage == self.selected_stage]
        if len(selected) != 1 or selected[0].action != StageAction.RUN.value:
            raise HarnessError("selected_stage must have exactly one run receipt")
        if self.route == ModelRoute.HUMAN_REVIEW_REQUIRED.value:
            if self.selected_stage != LadderStage.HUMAN_REVIEW.value:
                raise HarnessError("human_review_required must select human_review")
        elif self.selected_stage == LadderStage.HUMAN_REVIEW.value:
            raise HarnessError("only human_review_required may select human_review")

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_stage": self.selected_stage,
            "route": self.route,
            "stage_receipts": [item.to_dict() for item in self.stage_receipts],
            "routing_decision": self.routing_decision.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecisionLadder":
        payload = _closed(data, cls._FIELDS, "DecisionLadder")
        receipts = payload["stage_receipts"]
        decision = payload["routing_decision"]
        if not isinstance(receipts, list):
            raise HarnessError("stage_receipts must be a list")
        if not isinstance(decision, Mapping):
            raise HarnessError("routing_decision must be an object")
        return cls(
            selected_stage=payload["selected_stage"],
            route=payload["route"],
            stage_receipts=tuple(StageReceipt.from_dict(item) for item in receipts),
            routing_decision=RoutingDecision.from_dict(decision),
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


def route_decision_ladder(
    inputs: DecisionLadderInputs | Mapping[str, Any],
    *,
    policy: ModelRoutingPolicy | Mapping[str, Any] | None = None,
) -> DecisionLadder:
    """Evaluate the one canonical nine-stage receipt-to-human ladder.

    All nine stages are represented, in declaration order, regardless of the
    selected executor.  Deterministic evidence is considered before the model
    score, and a decisive deterministic observation prevents model selection.
    A model stage additionally requires a closed, decision-relevant
    :class:`UnresolvedQuestionRecord`; model availability is deliberately not
    an input and therefore cannot justify skipping a prior stage.
    """

    if isinstance(inputs, Mapping):
        inputs = DecisionLadderInputs.from_dict(inputs)
    elif not isinstance(inputs, DecisionLadderInputs):
        raise HarnessError("inputs must be DecisionLadderInputs or mapping")

    base_decision = route_model(inputs.routing_inputs, policy=policy)
    receipts: list[StageReceipt] = []
    selected_stage: str | None = None

    for stage in _DETERMINISTIC_STAGES:
        evidence = inputs.deterministic_evidence.get(stage.value)
        if selected_stage is not None:
            receipts.append(
                StageReceipt(
                    stage=stage.value,
                    action=StageAction.SKIP.value,
                    reason=StageReason.PRIOR_STAGE_RESOLVED.value,
                )
            )
        elif evidence is None:
            receipts.append(
                StageReceipt(
                    stage=stage.value,
                    action=StageAction.SKIP.value,
                    reason=StageReason.NO_EVIDENCE.value,
                )
            )
        elif evidence.status == EvidenceStatus.RESOLVED.value:
            selected_stage = stage.value
            receipts.append(
                StageReceipt(
                    stage=stage.value,
                    action=StageAction.RUN.value,
                    reason=StageReason.DECISIVE_EVIDENCE.value,
                    decisive=True,
                    evidence_reference=evidence.reference,
                )
            )
        elif evidence.status == EvidenceStatus.INCONCLUSIVE.value:
            receipts.append(
                StageReceipt(
                    stage=stage.value,
                    action=StageAction.RUN.value,
                    reason=StageReason.INCONCLUSIVE_EVIDENCE.value,
                    evidence_reference=evidence.reference,
                )
            )
        elif evidence.status == EvidenceStatus.UNAVAILABLE.value:
            receipts.append(
                StageReceipt(
                    stage=stage.value,
                    action=StageAction.SKIP.value,
                    reason=StageReason.EVIDENCE_UNAVAILABLE.value,
                    evidence_reference=evidence.reference,
                )
            )
        else:
            receipts.append(
                StageReceipt(
                    stage=stage.value,
                    action=StageAction.SKIP.value,
                    reason=StageReason.NOT_APPLICABLE.value,
                    evidence_reference=evidence.reference,
                )
            )

    # A low-risk, proof-covered route remains deterministic even where no
    # individual deterministic evidence producer supplied a terminal result.
    # The final deterministic stage records that eligibility explicitly.
    if selected_stage is None and base_decision.route == ModelRoute.DETERMINISTIC_ONLY.value:
        selected_stage = LadderStage.INCREMENTAL_PROVER.value
        receipts[-1] = StageReceipt(
            stage=LadderStage.INCREMENTAL_PROVER.value,
            action=StageAction.RUN.value,
            reason=StageReason.DETERMINISTIC_ROUTE_ELIGIBLE.value,
            decisive=True,
        )

    if selected_stage is not None:
        for stage in _LADDER_STAGES[len(_DETERMINISTIC_STAGES) :]:
            receipts.append(
                StageReceipt(
                    stage=stage.value,
                    action=StageAction.SKIP.value,
                    reason=StageReason.PRIOR_STAGE_RESOLVED.value,
                )
            )
        return DecisionLadder(
            selected_stage=selected_stage,
            route=ModelRoute.DETERMINISTIC_ONLY.value,
            stage_receipts=tuple(receipts),
            routing_decision=base_decision,
        )

    target_stage = _MODEL_STAGE_BY_ROUTE.get(base_decision.route)
    question = inputs.unresolved_question
    question_reason: StageReason | None = None
    if target_stage is not None:
        if question is None:
            question_reason = StageReason.UNRESOLVED_QUESTION_REQUIRED
        elif not question.is_decision_relevant or base_decision.route not in question.admissible_decisions:
            question_reason = StageReason.QUESTION_NOT_DECISION_RELEVANT

    if target_stage is not None and question_reason is None:
        selected_stage = target_stage.value
        for stage in _LADDER_STAGES[len(_DETERMINISTIC_STAGES) : -1]:
            if stage == target_stage:
                receipts.append(
                    StageReceipt(
                        stage=stage.value,
                        action=StageAction.RUN.value,
                        reason=StageReason.MODEL_STAGE_SELECTED.value,
                        decisive=True,
                        evidence_reference=question.question_id if question else "unspecified",
                    )
                )
            elif _LADDER_STAGES.index(stage) < _LADDER_STAGES.index(target_stage):
                receipts.append(
                    StageReceipt(
                        stage=stage.value,
                        action=StageAction.SKIP.value,
                        reason=StageReason.ROUTE_NOT_ELIGIBLE.value,
                    )
                )
            else:
                receipts.append(
                    StageReceipt(
                        stage=stage.value,
                        action=StageAction.SKIP.value,
                        reason=StageReason.PRIOR_STAGE_RESOLVED.value,
                    )
                )
        receipts.append(
            StageReceipt(
                stage=LadderStage.HUMAN_REVIEW.value,
                action=StageAction.SKIP.value,
                reason=StageReason.PRIOR_STAGE_RESOLVED.value,
            )
        )
        return DecisionLadder(
            selected_stage=selected_stage,
            route=base_decision.route,
            stage_receipts=tuple(receipts),
            routing_decision=base_decision,
        )

    # A direct human route or a rejected model call both terminate at the
    # human stage.  The rejected model stages retain the typed reason, making
    # the failure auditable rather than silently falling through.
    model_reason = (
        question_reason.value
        if question_reason is not None
        else StageReason.ROUTE_NOT_ELIGIBLE.value
    )
    for stage in _LADDER_STAGES[len(_DETERMINISTIC_STAGES) : -1]:
        receipts.append(
            StageReceipt(
                stage=stage.value,
                action=StageAction.SKIP.value,
                reason=model_reason,
            )
        )
    receipts.append(
        StageReceipt(
            stage=LadderStage.HUMAN_REVIEW.value,
            action=StageAction.RUN.value,
            reason=StageReason.HUMAN_REVIEW_REQUIRED.value,
            decisive=True,
        )
    )
    return DecisionLadder(
        selected_stage=LadderStage.HUMAN_REVIEW.value,
        route=ModelRoute.HUMAN_REVIEW_REQUIRED.value,
        stage_receipts=tuple(receipts),
        routing_decision=base_decision,
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
            "DeterministicEvidence",
            "UnresolvedQuestionRecord",
            "StageReceipt",
            "DecisionLadderInputs",
            "DecisionLadder",
        ],
        "decision_ladder": [item.value for item in _LADDER_STAGES],
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
            "every_decision_records_every_ladder_stage_in_order",
            "deterministic_evidence_resolves_before_model_selection",
            "model_selection_requires_a_closed_decision_relevant_question",
            "high_risk_opaque_oversized_failed_cases_escalate",
            "providers_are_never_hardcoded",
        ],
    }


__all__ = [
    "ADAPTER_ID",
    "BOARD_NAMESPACE",
    "ConfidenceClass",
    "DecisionLadder",
    "DecisionLadderInputs",
    "DeterministicEvidence",
    "EvidenceStatus",
    "LadderStage",
    "MODEL_ROUTING_INTERFACE",
    "MODEL_ROUTING_SCHEMA",
    "ModelRoute",
    "ModelRoutingPolicy",
    "RiskClass",
    "RoutingDecision",
    "RoutingInputs",
    "StageAction",
    "StageReason",
    "StageReceipt",
    "UnresolvedQuestionRecord",
    "model_routing_descriptor",
    "route_allows_provider_dispatch",
    "route_decision_ladder",
    "route_model",
    "route_requires_human_review",
]
