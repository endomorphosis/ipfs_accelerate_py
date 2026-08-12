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
        ],
    }


__all__ = [
    "ADAPTER_ID",
    "BOARD_NAMESPACE",
    "ConfidenceClass",
    "MODEL_ROUTING_INTERFACE",
    "MODEL_ROUTING_SCHEMA",
    "ModelRoute",
    "ModelRoutingPolicy",
    "RiskClass",
    "RoutingDecision",
    "RoutingInputs",
    "model_routing_descriptor",
    "route_allows_provider_dispatch",
    "route_model",
    "route_requires_human_review",
]
