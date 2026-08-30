"""Deterministic model routing for the semantic-compression harness.

``routing.py`` is the canonical ModelRouting@1 authority. Every decision
traverses the exact nine-stage receipt-to-human ladder in order, with typed
run/skip reasons. Deterministic evidence resolves eligible cases before any
model stage; later-model availability cannot skip an earlier stage.

Scoring inputs remain context size, lowest relevant confidence, risk class,
affected dependency cone, unresolved obligations, prior repair failures, and
available proofs. Results are one of the five closed ``ModelRoute`` values
and always carry an ordered explanation.

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


# ---------------------------------------------------------------------------
# Canonical nine-stage receipt-to-human ladder (ModelRouting@1)
# ---------------------------------------------------------------------------
#
# ASEH-020 selects this module as the sole routing authority. Verification
# ``model_route.py`` consumes these types; it must not mint a second ladder.

CANONICAL_ROUTING_AUTHORITY = MODEL_ROUTING_INTERFACE
RECEIPT_TO_HUMAN_LADDER_SCHEMA = "semantic-state-receipt-to-human-ladder@1"


class LadderStage(str, Enum):
    """Exact ordered stages from cached receipt through human decision."""

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


RECEIPT_TO_HUMAN_LADDER: tuple[LadderStage, ...] = (
    LadderStage.EXACT_CURRENT_AUTHORITATIVE_CACHED_RECEIPT,
    LadderStage.AST_SYMBOL_DEPENDENCY_AND_IMPACT_ANALYSIS,
    LadderStage.SCHEMA_TYPE_STATIC_LINT_AND_CONTRACT_CHECKS,
    LadderStage.SELECTED_TESTS,
    LadderStage.INCREMENTAL_SMT_OR_THEOREM_PROVER,
    LadderStage.LOCAL_SMALL_SPECIALIST_MODEL,
    LadderStage.LOCAL_OR_REMOTE_MEDIUM_MODEL,
    LadderStage.REMOTE_STRONG_OR_FRONTIER_MODEL,
    LadderStage.HUMAN_DECISION,
)
DETERMINISTIC_LADDER_STAGES: tuple[LadderStage, ...] = RECEIPT_TO_HUMAN_LADDER[:5]
MODEL_LADDER_STAGES: tuple[LadderStage, ...] = RECEIPT_TO_HUMAN_LADDER[5:8]
HUMAN_LADDER_STAGE = LadderStage.HUMAN_DECISION


class DeterministicEvidenceState(str, Enum):
    """Closed evidence states for a deterministic ladder stage."""

    RESOLVES = "resolves"
    UNRESOLVED = "unresolved"
    UNAVAILABLE = "unavailable"
    INELIGIBLE = "ineligible"


class StageAction(str, Enum):
    RUN = "run"
    SKIP = "skip"


class StageRunReason(str, Enum):
    """Typed reasons a ladder stage is run."""

    ELIGIBLE_DETERMINISTIC_EVIDENCE = "eligible_deterministic_evidence"
    UNRESOLVED_DETERMINISTIC_EVIDENCE = "unresolved_deterministic_evidence"
    SMALLEST_ADEQUATE_EXECUTOR = "smallest_adequate_executor"
    HUMAN_DECISION_REQUIRED = "human_decision_required"


class StageSkipReason(str, Enum):
    """Typed reasons a ladder stage is skipped.

    Availability of a later model is never a skip reason for an earlier stage.
    """

    RESOLVED_BY_PRIOR_STAGE = "resolved_by_prior_stage"
    EVIDENCE_UNAVAILABLE = "evidence_unavailable"
    EVIDENCE_INELIGIBLE = "evidence_ineligible"
    MODEL_NOT_REQUIRED = "model_not_required"
    NOT_SMALLEST_ADEQUATE_EXECUTOR = "not_smallest_adequate_executor"
    REQUIRED_TIER_UNAVAILABLE = "required_tier_unavailable"
    HUMAN_GATE_PREEMPTS_MODEL = "human_gate_preempts_model"


_RESOLVING_RUN_REASONS = frozenset(
    {
        StageRunReason.ELIGIBLE_DETERMINISTIC_EVIDENCE.value,
        StageRunReason.SMALLEST_ADEQUATE_EXECUTOR.value,
        StageRunReason.HUMAN_DECISION_REQUIRED.value,
    }
)

_DETERMINISTIC_EVIDENCE_FIELDS: Mapping[LadderStage, str] = {
    LadderStage.EXACT_CURRENT_AUTHORITATIVE_CACHED_RECEIPT: "cached_receipt",
    LadderStage.AST_SYMBOL_DEPENDENCY_AND_IMPACT_ANALYSIS: "ast_symbol_impact",
    LadderStage.SCHEMA_TYPE_STATIC_LINT_AND_CONTRACT_CHECKS: "schema_type_static",
    LadderStage.SELECTED_TESTS: "selected_tests",
    LadderStage.INCREMENTAL_SMT_OR_THEOREM_PROVER: "incremental_prover",
}

_MODEL_REQUIRED_FIELDS: Mapping[LadderStage, str] = {
    LadderStage.LOCAL_SMALL_SPECIALIST_MODEL: "small_model_required",
    LadderStage.LOCAL_OR_REMOTE_MEDIUM_MODEL: "medium_model_required",
    LadderStage.REMOTE_STRONG_OR_FRONTIER_MODEL: "frontier_model_required",
}

_MODEL_AVAILABLE_FIELDS: Mapping[LadderStage, str] = {
    LadderStage.LOCAL_SMALL_SPECIALIST_MODEL: "small_model_available",
    LadderStage.LOCAL_OR_REMOTE_MEDIUM_MODEL: "medium_model_available",
    LadderStage.REMOTE_STRONG_OR_FRONTIER_MODEL: "frontier_model_available",
}

_STAGE_TO_MODEL_ROUTE: Mapping[str, str] = {
    LadderStage.EXACT_CURRENT_AUTHORITATIVE_CACHED_RECEIPT.value: (
        ModelRoute.DETERMINISTIC_ONLY.value
    ),
    LadderStage.AST_SYMBOL_DEPENDENCY_AND_IMPACT_ANALYSIS.value: (
        ModelRoute.DETERMINISTIC_ONLY.value
    ),
    LadderStage.SCHEMA_TYPE_STATIC_LINT_AND_CONTRACT_CHECKS.value: (
        ModelRoute.DETERMINISTIC_ONLY.value
    ),
    LadderStage.SELECTED_TESTS.value: ModelRoute.DETERMINISTIC_ONLY.value,
    LadderStage.INCREMENTAL_SMT_OR_THEOREM_PROVER.value: (
        ModelRoute.DETERMINISTIC_ONLY.value
    ),
    LadderStage.LOCAL_SMALL_SPECIALIST_MODEL.value: (
        ModelRoute.SMALL_LOCAL_MODEL.value
    ),
    LadderStage.LOCAL_OR_REMOTE_MEDIUM_MODEL.value: ModelRoute.MEDIUM_MODEL.value,
    LadderStage.REMOTE_STRONG_OR_FRONTIER_MODEL.value: (
        ModelRoute.FRONTIER_MODEL.value
    ),
    LadderStage.HUMAN_DECISION.value: ModelRoute.HUMAN_REVIEW_REQUIRED.value,
}


def _normalize_evidence_state(value: Any, *, field_name: str) -> str:
    if isinstance(value, DeterministicEvidenceState):
        return value.value
    return _enum(value, DeterministicEvidenceState, field_name)


@dataclass(frozen=True)
class LadderEvidence:
    """Closed declared evidence for one receipt-to-human traversal.

    Model availability is recorded so unavailable required tiers fail closed.
    Availability of a larger model cannot skip or select an earlier stage.
    """

    cached_receipt: str = DeterministicEvidenceState.UNAVAILABLE.value
    ast_symbol_impact: str = DeterministicEvidenceState.UNAVAILABLE.value
    schema_type_static: str = DeterministicEvidenceState.UNAVAILABLE.value
    selected_tests: str = DeterministicEvidenceState.UNAVAILABLE.value
    incremental_prover: str = DeterministicEvidenceState.UNAVAILABLE.value
    small_model_required: bool = False
    medium_model_required: bool = False
    frontier_model_required: bool = False
    small_model_available: bool = False
    medium_model_available: bool = False
    frontier_model_available: bool = False
    human_review_required: bool = False

    _FIELDS = frozenset(
        {
            "cached_receipt",
            "ast_symbol_impact",
            "schema_type_static",
            "selected_tests",
            "incremental_prover",
            "small_model_required",
            "medium_model_required",
            "frontier_model_required",
            "small_model_available",
            "medium_model_available",
            "frontier_model_available",
            "human_review_required",
        }
    )

    def __post_init__(self) -> None:
        for name in (
            "cached_receipt",
            "ast_symbol_impact",
            "schema_type_static",
            "selected_tests",
            "incremental_prover",
        ):
            object.__setattr__(
                self,
                name,
                _normalize_evidence_state(getattr(self, name), field_name=name),
            )
        for name in (
            "small_model_required",
            "medium_model_required",
            "frontier_model_required",
            "small_model_available",
            "medium_model_available",
            "frontier_model_available",
            "human_review_required",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        required = [
            flag
            for flag, active in (
                ("small_model_required", self.small_model_required),
                ("medium_model_required", self.medium_model_required),
                ("frontier_model_required", self.frontier_model_required),
            )
            if active
        ]
        if len(required) > 1:
            raise HarnessError(
                "at most one model tier may be the smallest adequate executor; "
                f"got {required}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cached_receipt": self.cached_receipt,
            "ast_symbol_impact": self.ast_symbol_impact,
            "schema_type_static": self.schema_type_static,
            "selected_tests": self.selected_tests,
            "incremental_prover": self.incremental_prover,
            "small_model_required": self.small_model_required,
            "medium_model_required": self.medium_model_required,
            "frontier_model_required": self.frontier_model_required,
            "small_model_available": self.small_model_available,
            "medium_model_available": self.medium_model_available,
            "frontier_model_available": self.frontier_model_available,
            "human_review_required": self.human_review_required,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LadderEvidence":
        payload = _closed(data, cls._FIELDS, "LadderEvidence")
        return cls(
            cached_receipt=_normalize_evidence_state(
                payload["cached_receipt"], field_name="cached_receipt"
            ),
            ast_symbol_impact=_normalize_evidence_state(
                payload["ast_symbol_impact"], field_name="ast_symbol_impact"
            ),
            schema_type_static=_normalize_evidence_state(
                payload["schema_type_static"], field_name="schema_type_static"
            ),
            selected_tests=_normalize_evidence_state(
                payload["selected_tests"], field_name="selected_tests"
            ),
            incremental_prover=_normalize_evidence_state(
                payload["incremental_prover"], field_name="incremental_prover"
            ),
            small_model_required=_bool(
                payload["small_model_required"], "small_model_required"
            ),
            medium_model_required=_bool(
                payload["medium_model_required"], "medium_model_required"
            ),
            frontier_model_required=_bool(
                payload["frontier_model_required"], "frontier_model_required"
            ),
            small_model_available=_bool(
                payload["small_model_available"], "small_model_available"
            ),
            medium_model_available=_bool(
                payload["medium_model_available"], "medium_model_available"
            ),
            frontier_model_available=_bool(
                payload["frontier_model_available"], "frontier_model_available"
            ),
            human_review_required=_bool(
                payload["human_review_required"], "human_review_required"
            ),
        )

    @property
    def required_model_stage(self) -> LadderStage | None:
        if self.small_model_required:
            return LadderStage.LOCAL_SMALL_SPECIALIST_MODEL
        if self.medium_model_required:
            return LadderStage.LOCAL_OR_REMOTE_MEDIUM_MODEL
        if self.frontier_model_required:
            return LadderStage.REMOTE_STRONG_OR_FRONTIER_MODEL
        return None


@dataclass(frozen=True)
class LadderStageRecord:
    """One typed run/skip observation for a single ladder stage."""

    stage: str
    action: str
    reason: str
    resolved: bool

    _FIELDS = frozenset({"stage", "action", "reason", "resolved"})

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _enum(self.stage, LadderStage, "stage"))
        object.__setattr__(
            self, "action", _enum(self.action, StageAction, "action")
        )
        if self.action == StageAction.RUN.value:
            object.__setattr__(
                self, "reason", _enum(self.reason, StageRunReason, "reason")
            )
        else:
            object.__setattr__(
                self, "reason", _enum(self.reason, StageSkipReason, "reason")
            )
        object.__setattr__(self, "resolved", _bool(self.resolved, "resolved"))
        if self.resolved and self.action != StageAction.RUN.value:
            raise HarnessError("skipped stages cannot resolve a decision")
        if self.resolved and self.reason not in _RESOLVING_RUN_REASONS:
            raise HarnessError(
                f"reason {self.reason!r} cannot mark a stage as resolved"
            )
        if (
            self.action == StageAction.RUN.value
            and self.reason == StageRunReason.UNRESOLVED_DETERMINISTIC_EVIDENCE.value
            and self.resolved
        ):
            raise HarnessError("unresolved deterministic evidence cannot resolve")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "action": self.action,
            "reason": self.reason,
            "resolved": self.resolved,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LadderStageRecord":
        payload = _closed(data, cls._FIELDS, "LadderStageRecord")
        return cls(
            stage=payload["stage"],
            action=payload["action"],
            reason=payload["reason"],
            resolved=payload["resolved"],
        )


@dataclass(frozen=True)
class LadderDecision:
    """Complete ordered traversal of the receipt-to-human ladder."""

    stages: tuple[LadderStageRecord, ...]
    selected_stage: str
    selected_route: str
    resolved_by_deterministic_evidence: bool
    evidence: LadderEvidence

    _FIELDS = frozenset(
        {
            "stages",
            "selected_stage",
            "selected_route",
            "resolved_by_deterministic_evidence",
            "evidence",
        }
    )

    def __post_init__(self) -> None:
        if not isinstance(self.stages, tuple):
            raise HarnessError("stages must be a tuple of LadderStageRecord")
        if len(self.stages) != len(RECEIPT_TO_HUMAN_LADDER):
            raise HarnessError(
                "ladder decisions must record every stage in order; "
                f"expected {len(RECEIPT_TO_HUMAN_LADDER)}, got {len(self.stages)}"
            )
        for index, (record, expected) in enumerate(
            zip(self.stages, RECEIPT_TO_HUMAN_LADDER)
        ):
            if not isinstance(record, LadderStageRecord):
                raise HarnessError("stages must contain LadderStageRecord values")
            if record.stage != expected.value:
                raise HarnessError(
                    f"stage {index} must be {expected.value}, got {record.stage}"
                )
        object.__setattr__(
            self,
            "selected_stage",
            _enum(self.selected_stage, LadderStage, "selected_stage"),
        )
        object.__setattr__(
            self,
            "selected_route",
            _enum(self.selected_route, ModelRoute, "selected_route"),
        )
        object.__setattr__(
            self,
            "resolved_by_deterministic_evidence",
            _bool(
                self.resolved_by_deterministic_evidence,
                "resolved_by_deterministic_evidence",
            ),
        )
        if not isinstance(self.evidence, LadderEvidence):
            raise HarnessError("evidence must be LadderEvidence")
        resolved_records = [item for item in self.stages if item.resolved]
        if len(resolved_records) != 1:
            raise HarnessError("exactly one ladder stage must resolve")
        winner = resolved_records[0]
        if winner.stage != self.selected_stage:
            raise HarnessError("selected_stage must match the resolving record")
        expected_route = ladder_stage_to_model_route(self.selected_stage)
        if self.selected_route != expected_route:
            raise HarnessError(
                f"selected_route {self.selected_route!r} does not match "
                f"stage {self.selected_stage}"
            )
        det_selected = self.selected_stage in {
            stage.value for stage in DETERMINISTIC_LADDER_STAGES
        }
        if self.resolved_by_deterministic_evidence != det_selected:
            raise HarnessError(
                "resolved_by_deterministic_evidence must match a deterministic "
                "selected stage"
            )
        seen_resolution = False
        for record in self.stages:
            if record.resolved:
                seen_resolution = True
                continue
            if seen_resolution and (
                record.action != StageAction.SKIP.value
                or record.reason != StageSkipReason.RESOLVED_BY_PRIOR_STAGE.value
            ):
                raise HarnessError(
                    "stages after the resolver must skip with "
                    "resolved_by_prior_stage"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [item.to_dict() for item in self.stages],
            "selected_stage": self.selected_stage,
            "selected_route": self.selected_route,
            "resolved_by_deterministic_evidence": (
                self.resolved_by_deterministic_evidence
            ),
            "evidence": self.evidence.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LadderDecision":
        payload = _closed(data, cls._FIELDS, "LadderDecision")
        stages_raw = payload["stages"]
        if not isinstance(stages_raw, list):
            raise HarnessError("stages must be a list")
        evidence_raw = payload["evidence"]
        if not isinstance(evidence_raw, Mapping):
            raise HarnessError("evidence must be an object")
        return cls(
            stages=tuple(LadderStageRecord.from_dict(item) for item in stages_raw),
            selected_stage=payload["selected_stage"],
            selected_route=payload["selected_route"],
            resolved_by_deterministic_evidence=payload[
                "resolved_by_deterministic_evidence"
            ],
            evidence=LadderEvidence.from_dict(evidence_raw),
        )

    @property
    def run_records(self) -> tuple[LadderStageRecord, ...]:
        return tuple(
            item for item in self.stages if item.action == StageAction.RUN.value
        )

    @property
    def skip_records(self) -> tuple[LadderStageRecord, ...]:
        return tuple(
            item for item in self.stages if item.action == StageAction.SKIP.value
        )


def ladder_stage_to_model_route(stage: LadderStage | str) -> str:
    """Map one ladder stage onto the closed ModelRoute vocabulary."""

    token = stage.value if isinstance(stage, LadderStage) else str(stage)
    try:
        return _STAGE_TO_MODEL_ROUTE[token]
    except KeyError as exc:
        raise HarnessError(f"unsupported ladder stage {token!r}") from exc


def evaluate_receipt_to_human_ladder(
    evidence: LadderEvidence | Mapping[str, Any],
) -> LadderDecision:
    """Traverse every receipt-to-human stage in order with typed reasons.

    Deterministic evidence that can resolve does so before any model stage is
    selected. Later-stage model availability is never consulted when deciding
    an earlier stage.
    """

    if isinstance(evidence, Mapping):
        evidence_obj = LadderEvidence.from_dict(evidence)
    elif isinstance(evidence, LadderEvidence):
        evidence_obj = evidence
    else:
        raise HarnessError("evidence must be LadderEvidence or mapping")

    records: list[LadderStageRecord] = []
    selected: LadderStage | None = None
    required_model = evidence_obj.required_model_stage

    for stage in RECEIPT_TO_HUMAN_LADDER:
        if selected is not None:
            records.append(
                LadderStageRecord(
                    stage=stage.value,
                    action=StageAction.SKIP.value,
                    reason=StageSkipReason.RESOLVED_BY_PRIOR_STAGE.value,
                    resolved=False,
                )
            )
            continue

        if stage in DETERMINISTIC_LADDER_STAGES:
            field_name = _DETERMINISTIC_EVIDENCE_FIELDS[stage]
            state = getattr(evidence_obj, field_name)
            if state == DeterministicEvidenceState.RESOLVES.value:
                records.append(
                    LadderStageRecord(
                        stage=stage.value,
                        action=StageAction.RUN.value,
                        reason=StageRunReason.ELIGIBLE_DETERMINISTIC_EVIDENCE.value,
                        resolved=True,
                    )
                )
                selected = stage
            elif state == DeterministicEvidenceState.UNRESOLVED.value:
                records.append(
                    LadderStageRecord(
                        stage=stage.value,
                        action=StageAction.RUN.value,
                        reason=StageRunReason.UNRESOLVED_DETERMINISTIC_EVIDENCE.value,
                        resolved=False,
                    )
                )
            elif state == DeterministicEvidenceState.INELIGIBLE.value:
                records.append(
                    LadderStageRecord(
                        stage=stage.value,
                        action=StageAction.SKIP.value,
                        reason=StageSkipReason.EVIDENCE_INELIGIBLE.value,
                        resolved=False,
                    )
                )
            else:
                records.append(
                    LadderStageRecord(
                        stage=stage.value,
                        action=StageAction.SKIP.value,
                        reason=StageSkipReason.EVIDENCE_UNAVAILABLE.value,
                        resolved=False,
                    )
                )
            continue

        if stage in MODEL_LADDER_STAGES:
            if evidence_obj.human_review_required:
                skip_reason = StageSkipReason.HUMAN_GATE_PREEMPTS_MODEL.value
            elif required_model is None:
                skip_reason = StageSkipReason.MODEL_NOT_REQUIRED.value
            elif stage is not required_model:
                skip_reason = StageSkipReason.NOT_SMALLEST_ADEQUATE_EXECUTOR.value
            elif not bool(getattr(evidence_obj, _MODEL_AVAILABLE_FIELDS[stage])):
                skip_reason = StageSkipReason.REQUIRED_TIER_UNAVAILABLE.value
            else:
                skip_reason = ""
            if skip_reason:
                records.append(
                    LadderStageRecord(
                        stage=stage.value,
                        action=StageAction.SKIP.value,
                        reason=skip_reason,
                        resolved=False,
                    )
                )
                continue
            records.append(
                LadderStageRecord(
                    stage=stage.value,
                    action=StageAction.RUN.value,
                    reason=StageRunReason.SMALLEST_ADEQUATE_EXECUTOR.value,
                    resolved=True,
                )
            )
            selected = stage
            continue

        records.append(
            LadderStageRecord(
                stage=stage.value,
                action=StageAction.RUN.value,
                reason=StageRunReason.HUMAN_DECISION_REQUIRED.value,
                resolved=True,
            )
        )
        selected = stage

    if selected is None:
        raise HarnessError("ladder traversal failed to select a stage")

    return LadderDecision(
        stages=tuple(records),
        selected_stage=selected.value,
        selected_route=ladder_stage_to_model_route(selected),
        resolved_by_deterministic_evidence=selected in DETERMINISTIC_LADDER_STAGES,
        evidence=evidence_obj,
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


def ladder_evidence_from_routing_inputs(
    inputs: RoutingInputs,
    policy: ModelRoutingPolicy,
) -> LadderEvidence:
    """Project closed ModelRouting@1 inputs onto the canonical ladder.

    RoutingInputs does not carry per-stage execution receipts. Human-gated
    cases therefore mark deterministic stages ineligible rather than inventing
    a current-tree cache hit. Eligible deterministic cases resolve on the
    earliest declared evidence (cached proof receipt, otherwise schema/static).
    """

    unavailable = DeterministicEvidenceState.UNAVAILABLE.value
    ineligible = DeterministicEvidenceState.INELIGIBLE.value
    resolves = DeterministicEvidenceState.RESOLVES.value
    human_reasons = _human_review_reasons(inputs, policy)
    if human_reasons:
        return LadderEvidence(
            cached_receipt=ineligible,
            ast_symbol_impact=ineligible,
            schema_type_static=ineligible,
            selected_tests=ineligible,
            incremental_prover=ineligible,
            small_model_required=False,
            medium_model_required=False,
            frontier_model_required=False,
            small_model_available=False,
            medium_model_available=False,
            frontier_model_available=False,
            human_review_required=True,
        )
    if _deterministic_only_eligible(inputs, policy):
        cached = resolves if inputs.available_proofs > 0 else unavailable
        schema = unavailable if inputs.available_proofs > 0 else resolves
        return LadderEvidence(
            cached_receipt=cached,
            ast_symbol_impact=unavailable,
            schema_type_static=schema,
            selected_tests=unavailable,
            incremental_prover=unavailable,
            small_model_required=False,
            medium_model_required=False,
            frontier_model_required=False,
            small_model_available=False,
            medium_model_available=False,
            frontier_model_available=False,
            human_review_required=False,
        )
    scored, _reasons = _score_model_route(inputs, policy)
    return LadderEvidence(
        cached_receipt=unavailable,
        ast_symbol_impact=unavailable,
        schema_type_static=unavailable,
        selected_tests=unavailable,
        incremental_prover=unavailable,
        small_model_required=scored == ModelRoute.SMALL_LOCAL_MODEL.value,
        medium_model_required=scored == ModelRoute.MEDIUM_MODEL.value,
        frontier_model_required=scored == ModelRoute.FRONTIER_MODEL.value,
        small_model_available=True,
        medium_model_available=True,
        frontier_model_available=True,
        human_review_required=False,
    )


def evaluate_ladder_for_routing_inputs(
    inputs: RoutingInputs,
    policy: ModelRoutingPolicy,
) -> LadderDecision:
    """Traverse the canonical ladder for ModelRouting@1 scoring inputs."""

    return evaluate_receipt_to_human_ladder(
        ladder_evidence_from_routing_inputs(inputs, policy)
    )


def _assert_ladder_agrees_with_route(
    *,
    inputs: RoutingInputs,
    policy: ModelRoutingPolicy,
    route: str,
) -> LadderDecision:
    ladder = evaluate_ladder_for_routing_inputs(inputs, policy)
    if ladder.selected_route != route:
        raise HarnessError(
            "canonical receipt-to-human ladder selected "
            f"{ladder.selected_route} at {ladder.selected_stage}, "
            f"but the route table selected {route}"
        )
    return ladder


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
        decision = RoutingDecision(
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
        _assert_ladder_agrees_with_route(
            inputs=inputs, policy=policy_obj, route=decision.route
        )
        return decision

    if _deterministic_only_eligible(inputs, policy_obj):
        reasons = [
            "deterministic_only",
            f"confidence_{inputs.lowest_confidence}",
            f"risk_{inputs.risk}",
            "proofs_cover_or_absent_obligations",
        ]
        decision = RoutingDecision(
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
        _assert_ladder_agrees_with_route(
            inputs=inputs, policy=policy_obj, route=decision.route
        )
        return decision

    route, reasons = _score_model_route(inputs, policy_obj)
    explanation = _clip(f"{route}: " + ", ".join(reasons))
    decision = RoutingDecision(
        route=route,
        reason_codes=_sorted_reason_codes([route, *reasons]),
        explanation=explanation,
        requires_provider=True,
        halt_before_dispatch=False,
        halt_before_root_publication=False,
        inputs=inputs,
        policy=policy_obj,
    )
    _assert_ladder_agrees_with_route(
        inputs=inputs, policy=policy_obj, route=decision.route
    )
    return decision


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
            "every_decision_traverses_receipt_to_human_ladder_in_order",
            "typed_run_and_skip_reasons_required",
            "deterministic_evidence_resolves_before_models",
            "later_model_availability_cannot_skip_earlier_stages",
        ],
        "canonical_authority": CANONICAL_ROUTING_AUTHORITY,
        "receipt_to_human_ladder": [stage.value for stage in RECEIPT_TO_HUMAN_LADDER],
        "ladder_schema": RECEIPT_TO_HUMAN_LADDER_SCHEMA,
    }


__all__ = [
    "ADAPTER_ID",
    "BOARD_NAMESPACE",
    "CANONICAL_ROUTING_AUTHORITY",
    "ConfidenceClass",
    "DETERMINISTIC_LADDER_STAGES",
    "DeterministicEvidenceState",
    "HUMAN_LADDER_STAGE",
    "LadderDecision",
    "LadderEvidence",
    "LadderStage",
    "LadderStageRecord",
    "MODEL_LADDER_STAGES",
    "MODEL_ROUTING_INTERFACE",
    "MODEL_ROUTING_SCHEMA",
    "ModelRoute",
    "ModelRoutingPolicy",
    "RECEIPT_TO_HUMAN_LADDER",
    "RECEIPT_TO_HUMAN_LADDER_SCHEMA",
    "RiskClass",
    "RoutingDecision",
    "RoutingInputs",
    "StageAction",
    "StageRunReason",
    "StageSkipReason",
    "evaluate_ladder_for_routing_inputs",
    "evaluate_receipt_to_human_ladder",
    "ladder_evidence_from_routing_inputs",
    "ladder_stage_to_model_route",
    "model_routing_descriptor",
    "route_allows_provider_dispatch",
    "route_model",
    "route_requires_human_review",
]
