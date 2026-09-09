"""Decision receipts and highest-level machine output.

This module extends the existing LGSWF ``emit_decision`` envelope with
observational route explanations for already-made
:class:`~ipfs_accelerate_py.agent_supervisor.verification.contracts.ModelRouteDecision`
values. It does **not** classify, escalate, or re-score routes: selection stays
on ``verification.model_route``. Explanations and receipts are
non-authoritative projections suitable for DOEP-G080.S2 observability and for
downstream unnecessary-escalation metrics.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    cid_for_dag_json,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    MODEL_ROUTE_DECISION_INTERFACE,
    MODEL_ROUTE_DECISION_SCHEMA,
    ModelRoute,
    ModelRouteDecision,
)
from ipfs_accelerate_py.agent_supervisor.verification.model_route import (
    DETERMINISTIC_FIRST_ROUTE_LADDER,
    REASON_SMALLER_ROUTE_FAILED,
)

# ---------------------------------------------------------------------------
# Schema / interface constants
# ---------------------------------------------------------------------------

DECISION_RECEIPT_SCHEMA: Final[str] = "lgswf/decision-receipt@1"
DECISION_RECEIPT_INTERFACE: Final[str] = "DecisionReceipt@1"

ROUTE_EXPLANATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/route-explanation@1"
)
ROUTE_EXPLANATION_INTERFACE: Final[str] = "RouteExplanation@1"

ROUTE_EXPLANATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/route-explanation-receipt@1"
)
ROUTE_EXPLANATION_RECEIPT_INTERFACE: Final[str] = "RouteExplanationReceipt@1"

ROUTE_EXPLANATION_EVIDENCE: Final[str] = "doep/route-explanation@1"

# ModelRoute capability class → G080 explanation rung. Pre-model ladder tokens
# (receipt/static/tests/proof) remain owned by sibling DOEP modules.
ROUTE_LADDER_RUNGS: Final[Mapping[ModelRoute, str]] = MappingProxyType(
    {
        ModelRoute.DETERMINISTIC_ONLY: "deterministic",
        ModelRoute.SMALL_LOCAL_MODEL: "small",
        ModelRoute.MEDIUM_MODEL: "medium",
        ModelRoute.FRONTIER_MODEL: "frontier",
        ModelRoute.HUMAN_REVIEW_REQUIRED: "human",
    }
)

_REASON_PHRASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "unresolved_authority": "unresolved authority requires human review",
        "unmodeled_high_risk": "unmodeled high-risk effects require human review",
        "scope_crossing": "scope crossing requires human review",
        "proof_test_conflict": "proof and test conflict requires human review",
        "unsafe_context": "unsafe context requires human review",
        "nonreproducible_environment": "non-reproducible environment requires human review",
        "verification_incomplete": "verification is incomplete",
        "mandatory_full_suite_pending": "mandatory full suite is still pending",
        "required_tier_unavailable": "required model tier is unavailable; no downgrade",
        "mechanical_exact_work": "mechanical exact work admits a deterministic route",
        "localized_exact_counterexample": (
            "localized exact work with a good counterexample admits a small route"
        ),
        "multi_file_synthesis": "multi-file synthesis requires a medium route",
        "ambiguous_work": "ambiguous work requires a frontier route",
        "broad_dependency_cone": "broad dependency cone requires a frontier route",
        "opaque_critical_dependency": (
            "opaque critical dependency requires a frontier route"
        ),
        "conflicting_proof_requirements": (
            "conflicting proof requirements require a frontier route"
        ),
        "smaller_route_failed": "a smaller route already failed; escalate",
        "context_overflow": "context overflow forces a higher route",
        "human_review_plan": "plan requires human review",
        "high_risk": "high risk requires human review",
        "route_selected": "route selected by the canonical classifier",
    }
)

_MAX_EXPLANATION_CHARS: Final[int] = 2_048
_PROVIDER_IDENTITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "provider",
        "provider_id",
        "providers",
        "vendor",
        "vendor_id",
        "vendors",
        "model",
        "model_id",
        "model_ids",
        "model_name",
        "models",
        "endpoint",
        "endpoint_id",
        "api_base",
        "base_url",
        "deployment",
        "deployment_id",
        "engine",
        "engine_id",
    }
)


class DecisionReceiptError(ValueError):
    """Fail-closed error for malformed decision or route-explanation input."""


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RouteExplanation:
    """Pure projection of one :class:`ModelRouteDecision` into an explanation."""

    route: ModelRoute
    ladder_rung: str
    explanation: str
    decisive_reason_codes: tuple[str, ...]
    considered_routes: tuple[ModelRoute, ...]
    required_capabilities: tuple[str, ...]
    requires_human_review: bool
    policy_cid: str
    decision_id: str
    escalated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ROUTE_EXPLANATION_SCHEMA,
            "interface": ROUTE_EXPLANATION_INTERFACE,
            "evidence": ROUTE_EXPLANATION_EVIDENCE,
            "route": self.route.value,
            "ladder_rung": self.ladder_rung,
            "explanation": self.explanation,
            "decisive_reason_codes": list(self.decisive_reason_codes),
            "considered_routes": [item.value for item in self.considered_routes],
            "required_capabilities": list(self.required_capabilities),
            "requires_human_review": self.requires_human_review,
            "policy_cid": self.policy_cid,
            "decision_id": self.decision_id,
            "escalated": self.escalated,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RouteExplanation":
        if not isinstance(payload, Mapping):
            raise DecisionReceiptError("route explanation payload must be a mapping")
        _reject_provider_identity(payload)
        schema = payload.get("schema")
        if schema not in (None, ROUTE_EXPLANATION_SCHEMA):
            raise DecisionReceiptError(
                f"unexpected route explanation schema: {schema!r}"
            )
        interface = payload.get("interface")
        if interface not in (None, ROUTE_EXPLANATION_INTERFACE):
            raise DecisionReceiptError(
                f"unexpected route explanation interface: {interface!r}"
            )
        route = _coerce_route(payload.get("route"))
        codes = _string_tuple(payload.get("decisive_reason_codes"), "decisive_reason_codes")
        considered = tuple(
            _coerce_route(item)
            for item in _sequence(payload.get("considered_routes"), "considered_routes")
        )
        caps = _string_tuple(
            payload.get("required_capabilities"), "required_capabilities"
        )
        requires_human = bool(payload.get("requires_human_review", False))
        if requires_human != (route is ModelRoute.HUMAN_REVIEW_REQUIRED):
            raise DecisionReceiptError(
                "requires_human_review must match human_review_required route"
            )
        ladder = str(payload.get("ladder_rung") or ROUTE_LADDER_RUNGS[route])
        if ladder != ROUTE_LADDER_RUNGS[route]:
            raise DecisionReceiptError(
                f"ladder_rung {ladder!r} does not match route {route.value!r}"
            )
        explanation = _clip(str(payload.get("explanation") or ""))
        if not explanation:
            raise DecisionReceiptError("explanation must be a nonempty string")
        policy_cid = _nonempty_str(payload.get("policy_cid"), "policy_cid")
        decision_id = _nonempty_str(payload.get("decision_id"), "decision_id")
        escalated = bool(payload.get("escalated", False))
        if escalated != (REASON_SMALLER_ROUTE_FAILED in codes):
            raise DecisionReceiptError(
                "escalated must match presence of smaller_route_failed"
            )
        return cls(
            route=route,
            ladder_rung=ladder,
            explanation=explanation,
            decisive_reason_codes=codes,
            considered_routes=considered,
            required_capabilities=caps,
            requires_human_review=requires_human,
            policy_cid=policy_cid,
            decision_id=decision_id,
            escalated=escalated,
        )


@dataclass(frozen=True, slots=True)
class RouteExplanationReceipt:
    """Content-addressed, non-authoritative route explanation receipt."""

    explanation: RouteExplanation
    metrics: Mapping[str, Any]
    receipt_id: str
    authoritative: bool = False
    completion_authoritative: bool = False

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": ROUTE_EXPLANATION_RECEIPT_SCHEMA,
            "interface": ROUTE_EXPLANATION_RECEIPT_INTERFACE,
            "evidence": ROUTE_EXPLANATION_EVIDENCE,
            "authoritative": False,
            "completion_authoritative": False,
            "route": self.explanation.route.value,
            "ladder_rung": self.explanation.ladder_rung,
            "explanation": self.explanation.explanation,
            "decisive_reason_codes": list(self.explanation.decisive_reason_codes),
            "considered_routes": [
                item.value for item in self.explanation.considered_routes
            ],
            "required_capabilities": list(self.explanation.required_capabilities),
            "requires_human_review": self.explanation.requires_human_review,
            "policy_cid": self.explanation.policy_cid,
            "decision_id": self.explanation.decision_id,
            "escalated": self.explanation.escalated,
            "metrics": dict(self.metrics),
            "deterministic_first_route_ladder": list(DETERMINISTIC_FIRST_ROUTE_LADDER),
        }
        if include_identity:
            payload["receipt_id"] = self.receipt_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RouteExplanationReceipt":
        if not isinstance(payload, Mapping):
            raise DecisionReceiptError("route explanation receipt must be a mapping")
        _reject_provider_identity(payload)
        schema = payload.get("schema")
        if schema not in (None, ROUTE_EXPLANATION_RECEIPT_SCHEMA):
            raise DecisionReceiptError(
                f"unexpected route explanation receipt schema: {schema!r}"
            )
        if payload.get("authoritative") is True:
            raise DecisionReceiptError(
                "route explanation receipts are observational and never authoritative"
            )
        if payload.get("completion_authoritative") is True:
            raise DecisionReceiptError(
                "route explanation receipts never authorize completion"
            )
        metrics_raw = payload.get("metrics") or {}
        if not isinstance(metrics_raw, Mapping):
            raise DecisionReceiptError("metrics must be a mapping")
        _reject_provider_identity(metrics_raw)
        explanation = RouteExplanation.from_dict(
            {
                "schema": ROUTE_EXPLANATION_SCHEMA,
                "interface": ROUTE_EXPLANATION_INTERFACE,
                "route": payload.get("route"),
                "ladder_rung": payload.get("ladder_rung"),
                "explanation": payload.get("explanation"),
                "decisive_reason_codes": payload.get("decisive_reason_codes"),
                "considered_routes": payload.get("considered_routes"),
                "required_capabilities": payload.get("required_capabilities"),
                "requires_human_review": payload.get("requires_human_review"),
                "policy_cid": payload.get("policy_cid"),
                "decision_id": payload.get("decision_id"),
                "escalated": payload.get("escalated"),
            }
        )
        receipt = cls(
            explanation=explanation,
            metrics=MappingProxyType(dict(metrics_raw)),
            receipt_id="",  # filled after identity check
            authoritative=False,
            completion_authoritative=False,
        )
        expected = _receipt_id_for(receipt.to_dict(include_identity=False))
        provided = payload.get("receipt_id")
        if provided is None:
            object.__setattr__(receipt, "receipt_id", expected)
            return receipt
        provided_text = _nonempty_str(provided, "receipt_id")
        if provided_text != expected:
            raise DecisionReceiptError("receipt_id does not match payload identity")
        object.__setattr__(receipt, "receipt_id", expected)
        return receipt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def explain_model_route(
    decision: ModelRouteDecision | Mapping[str, Any],
) -> RouteExplanation:
    """Project a canonical :class:`ModelRouteDecision` into a route explanation.

    Does not call the classifier, escalation policy, or any provider. Rejects
    competing routing objects (for example semantic-state harness decisions).
    """

    resolved = _coerce_model_route_decision(decision)
    codes = tuple(resolved.decisive_reason_codes)
    parts: list[str] = []
    for code in codes:
        phrase = _phrase_for(code)
        if phrase == code:
            parts.append(code)
        else:
            parts.append(f"{code} ({phrase})")
    explanation = _clip(f"{resolved.route.value}: " + "; ".join(parts))
    return RouteExplanation(
        route=resolved.route,
        ladder_rung=ROUTE_LADDER_RUNGS[resolved.route],
        explanation=explanation,
        decisive_reason_codes=codes,
        considered_routes=tuple(resolved.considered_routes),
        required_capabilities=tuple(resolved.required_capabilities),
        requires_human_review=bool(resolved.requires_human_review),
        policy_cid=str(resolved.policy_cid),
        decision_id=str(resolved.decision_id),
        escalated=REASON_SMALLER_ROUTE_FAILED in codes,
    )


def emit_route_explanation_receipt(
    decision: ModelRouteDecision | Mapping[str, Any],
    *,
    metrics: Mapping[str, Any] | None = None,
) -> MappingProxyType:
    """Emit a frozen, content-addressed route explanation receipt.

    The receipt is observational: ``authoritative`` and
    ``completion_authoritative`` are always false.
    """

    explanation = explain_model_route(decision)
    metrics_map = _freeze_metrics(metrics)
    provisional = RouteExplanationReceipt(
        explanation=explanation,
        metrics=metrics_map,
        receipt_id="",
    )
    receipt_id = _receipt_id_for(provisional.to_dict(include_identity=False))
    receipt = RouteExplanationReceipt(
        explanation=explanation,
        metrics=metrics_map,
        receipt_id=receipt_id,
    )
    return MappingProxyType(receipt.to_dict(include_identity=True))


def emit_decision(record: Mapping[str, Any] | None = None) -> MappingProxyType:
    """Emit the highest-level LGSWF decision receipt.

    Generic ``{decision, metrics}`` payloads keep the historical
    ``lgswf/decision-receipt@1`` shape. When ``record`` carries a
    :class:`ModelRouteDecision` (object or ``model-route-decision@1`` mapping)
    under ``decision`` or ``model_route_decision``, the receipt also nests
    ``route_explanation`` and ``route_receipt`` projections without changing
    the outer schema.
    """

    payload = dict(record or {})
    if not isinstance(payload, Mapping):
        raise DecisionReceiptError("decision record must be a mapping")
    _reject_provider_identity(payload)

    metrics_raw = payload.get("metrics") or {}
    if not isinstance(metrics_raw, Mapping):
        raise DecisionReceiptError("metrics must be a mapping")
    _reject_provider_identity(metrics_raw)
    metrics = dict(metrics_raw)

    decision_value = payload.get("decision")
    model_route_value = payload.get("model_route_decision", decision_value)

    result: dict[str, Any] = {
        "schema": DECISION_RECEIPT_SCHEMA,
        "interface": DECISION_RECEIPT_INTERFACE,
        "decision": decision_value,
        "metrics": metrics,
    }

    if _looks_like_model_route_decision(model_route_value):
        resolved = _coerce_model_route_decision(model_route_value)
        decision_record = resolved.to_record()
        explanation = explain_model_route(resolved)
        route_receipt = emit_route_explanation_receipt(resolved, metrics=metrics)
        # Prefer an explicit model_route_decision key; when decision itself was
        # the ModelRouteDecision, serialize it as a plain mapping.
        if _looks_like_model_route_decision(decision_value):
            result["decision"] = decision_record
        result["model_route_decision"] = decision_record
        result["route_explanation"] = explanation.to_dict()
        result["route_receipt"] = dict(route_receipt)
        result["decision_id"] = resolved.decision_id

    if payload.get("authoritative") is True or payload.get("completion_authoritative") is True:
        raise DecisionReceiptError(
            "decision receipts emitted here are observational and never authoritative"
        )
    result["authoritative"] = False
    result["completion_authoritative"] = False
    return MappingProxyType(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _phrase_for(code: str) -> str:
    return _REASON_PHRASES.get(code, code)


def _clip(text: str) -> str:
    value = text.strip()
    if len(value) <= _MAX_EXPLANATION_CHARS:
        return value
    return value[: _MAX_EXPLANATION_CHARS - 3] + "..."


def _nonempty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DecisionReceiptError(f"{field_name} must be a nonempty string")
    return value.strip()


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DecisionReceiptError(f"{field_name} must be a sequence")
    return value


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    items = _sequence(value if value is not None else (), field_name)
    out: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item:
            raise DecisionReceiptError(f"{field_name} items must be nonempty strings")
        out.append(item)
    return tuple(out)


def _coerce_route(value: Any) -> ModelRoute:
    if isinstance(value, ModelRoute):
        return value
    if isinstance(value, str):
        try:
            return ModelRoute(value)
        except ValueError as exc:
            raise DecisionReceiptError(f"unknown model route: {value!r}") from exc
    raise DecisionReceiptError("route must be a ModelRoute or route token string")


def _reject_provider_identity(payload: Mapping[str, Any]) -> None:
    forbidden = _PROVIDER_IDENTITY_FIELDS.intersection(payload.keys())
    if forbidden:
        raise DecisionReceiptError(
            "provider/vendor identity fields are forbidden on route receipts: "
            + ", ".join(sorted(forbidden))
        )


def _freeze_metrics(metrics: Mapping[str, Any] | None) -> MappingProxyType:
    raw = dict(metrics or {})
    if not isinstance(raw, MutableMapping):
        raise DecisionReceiptError("metrics must be a mapping")
    _reject_provider_identity(raw)
    return MappingProxyType(raw)


def _receipt_id_for(payload: Mapping[str, Any]) -> str:
    return cid_for_dag_json(dict(payload), for_identity=True)


def _looks_like_model_route_decision(value: Any) -> bool:
    if isinstance(value, ModelRouteDecision):
        return True
    if not isinstance(value, Mapping):
        return False
    schema = value.get("schema")
    interface = value.get("interface")
    if schema == MODEL_ROUTE_DECISION_SCHEMA:
        return True
    if interface == MODEL_ROUTE_DECISION_INTERFACE and "route" in value:
        return True
    # Accept closed decision records that omit schema but carry route fields.
    return (
        "route" in value
        and "decisive_reason_codes" in value
        and "policy_cid" in value
        and "considered_routes" in value
    )


def _coerce_model_route_decision(
    value: ModelRouteDecision | Mapping[str, Any],
) -> ModelRouteDecision:
    if isinstance(value, ModelRouteDecision):
        return value
    # Reject competing semantic-state harness routing objects.
    type_name = type(value).__name__
    module_name = getattr(type(value), "__module__", "") or ""
    if type_name == "RoutingDecision" or module_name.endswith("semantic_state.routing"):
        raise DecisionReceiptError(
            "competing semantic_state RoutingDecision is not admitted; "
            "pass a canonical ModelRouteDecision"
        )
    if not isinstance(value, Mapping):
        raise DecisionReceiptError(
            "decision must be a ModelRouteDecision or model-route-decision mapping"
        )
    _reject_provider_identity(value)
    try:
        if "schema" in value or "interface" in value:
            return ModelRouteDecision.from_dict(value)
        # Minimal closed mapping without header: rebuild through from_dict with header.
        enriched = {
            "schema": MODEL_ROUTE_DECISION_SCHEMA,
            "interface": MODEL_ROUTE_DECISION_INTERFACE,
            "contract_version": value.get("contract_version", 1),
            **dict(value),
        }
        return ModelRouteDecision.from_dict(enriched)
    except Exception as exc:  # noqa: BLE001 - fail closed with local error type
        raise DecisionReceiptError(
            f"invalid ModelRouteDecision payload: {exc}"
        ) from exc


__all__ = [
    "DECISION_RECEIPT_INTERFACE",
    "DECISION_RECEIPT_SCHEMA",
    "DETERMINISTIC_FIRST_ROUTE_LADDER",
    "DecisionReceiptError",
    "ROUTE_EXPLANATION_EVIDENCE",
    "ROUTE_EXPLANATION_INTERFACE",
    "ROUTE_EXPLANATION_RECEIPT_INTERFACE",
    "ROUTE_EXPLANATION_RECEIPT_SCHEMA",
    "ROUTE_EXPLANATION_SCHEMA",
    "ROUTE_LADDER_RUNGS",
    "RouteExplanation",
    "RouteExplanationReceipt",
    "emit_decision",
    "emit_route_explanation_receipt",
    "explain_model_route",
]
