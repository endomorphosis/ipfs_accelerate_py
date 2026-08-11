"""Provider-neutral next-repair model routing for incremental verification.

``ModelRoutePlanner`` / :func:`choose_model_route` select a *capability class*
only.  Model/provider/vendor resolution is intentionally out of scope and
must remain a separate, downstream concern.

Supported routes (closed):

* ``deterministic_only``
* ``small_local_model``
* ``medium_model``
* ``frontier_model``
* ``human_review_required``

Fail-closed precedence (normative):

1. Unresolved authority/policy, unmodeled high-risk effects, scope crossings,
   proof/test conflict, unsafe context, non-reproducible environment, or a
   pending mandatory full/broader suite => human review *before* any model
   route.
2. Exact mechanical formatting / import / codemod / rename work =>
   deterministic.
3. Localized bounded exact/conservative work with a good minimized
   counterexample and low/moderate risk => small local.
4. Several-file nontrivial synthesis without an opaque critical dependency =>
   medium.
5. Ambiguity, broad cones, opaque critical behavior, conflicting proof
   requirements, smaller-route failures, or context overflow => frontier.

``available_models`` is a provider-neutral inventory of capability tier,
context limit, locality, and current availability.  Vendor preference is
rejected.  When the safely required model tier is unavailable, routing does
**not** downgrade: it returns ``human_review_required``.

Importing this module performs no I/O and never invokes a model provider.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .contracts import (
    MAX_COLLECTION_ITEMS,
    MAX_RESOURCE_QUANTITY,
    ModelRoute,
    ModelRouteDecision,
    VerificationContractError,
    VerificationPlan,
)
from .contracts import _cid as _contract_cid
from .contracts import _integer as _contract_integer
from .contracts import _token as _contract_token
from ..core.multiformats_identity import cid_for_dag_json

# ---------------------------------------------------------------------------
# Evidence / schema constants
# ---------------------------------------------------------------------------

MODEL_ROUTE_PLANNER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-model-route-planner@1"
)
MODEL_ROUTE_PLANNER_INTERFACE: Final[str] = "ModelRoutePlanner@1"
MODEL_ROUTE_EVIDENCE: Final[str] = "ivp/model-route@1"
AVAILABLE_MODEL_CAPABILITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-available-model-capability@1"
)
MODEL_ROUTE_FACTS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-model-route-facts@1"
)
MODEL_ROUTE_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-model-route-policy@1"
)
PRIOR_REPAIR_ATTEMPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-prior-repair-attempt@1"
)

# Capability tiers that require an inventory entry (deterministic does not).
_MODEL_TIERS: Final[frozenset[ModelRoute]] = frozenset(
    {
        ModelRoute.SMALL_LOCAL_MODEL,
        ModelRoute.MEDIUM_MODEL,
        ModelRoute.FRONTIER_MODEL,
    }
)

_ALL_ROUTES: Final[tuple[ModelRoute, ...]] = (
    ModelRoute.DETERMINISTIC_ONLY,
    ModelRoute.SMALL_LOCAL_MODEL,
    ModelRoute.MEDIUM_MODEL,
    ModelRoute.FRONTIER_MODEL,
    ModelRoute.HUMAN_REVIEW_REQUIRED,
)

# Inventory / decision must never carry vendor or provider identity.
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
        "openai",
        "anthropic",
        "grok",
        "gemini",
        "codex",
        "ollama",
        "huggingface",
        "hf_model",
        "served_model_name",
    }
)

_MECHANICAL_KINDS: Final[frozenset[str]] = frozenset(
    {
        "mechanical_formatting",
        "mechanical_import",
        "mechanical_codemod",
        "mechanical_rename",
        "formatting",
        "import",
        "codemod",
        "rename",
    }
)

_LOCALIZED_KINDS: Final[frozenset[str]] = frozenset(
    {
        "localized_exact",
        "localized_conservative",
        "localized",
        "bounded_localized",
        "exact_localized",
    }
)

_MULTI_FILE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "multi_file_synthesis",
        "several_file_synthesis",
        "multi_file",
        "several_file",
        "synthesis",
    }
)

_AMBIGUOUS_KINDS: Final[frozenset[str]] = frozenset(
    {
        "ambiguous",
        "ambiguity",
        "broad",
        "opaque",
        "conflicting",
        "conflict",
        "unknown",
        "unclassified",
    }
)

_GOOD_COUNTEREXAMPLES: Final[frozenset[str]] = frozenset(
    {
        "good",
        "minimized",
        "high",
        "adequate",
        "present_good",
    }
)

_LOW_MODERATE_RISK: Final[frozenset[str]] = frozenset({"low", "moderate", "medium"})

_DEFAULT_MAX_CONTEXT_TOKENS: Final[int] = 128_000
_DEFAULT_SMALL_MAX_FILES: Final[int] = 2
_DEFAULT_SMALL_MAX_CONE: Final[int] = 8
_DEFAULT_MEDIUM_MAX_FILES: Final[int] = 12
_DEFAULT_MEDIUM_MAX_CONE: Final[int] = 64
_DEFAULT_OPAQUE_CRITICAL_THRESHOLD: Final[int] = 1

# Reason codes (canonical tokens; match VerificationContract _TOKEN_RE).
REASON_UNRESOLVED_AUTHORITY: Final[str] = "unresolved_authority"
REASON_UNMODELED_HIGH_RISK: Final[str] = "unmodeled_high_risk"
REASON_SCOPE_CROSSING: Final[str] = "scope_crossing"
REASON_PROOF_TEST_CONFLICT: Final[str] = "proof_test_conflict"
REASON_UNSAFE_CONTEXT: Final[str] = "unsafe_context"
REASON_NONREPRODUCIBLE_ENVIRONMENT: Final[str] = "nonreproducible_environment"
REASON_VERIFICATION_INCOMPLETE: Final[str] = "verification_incomplete"
REASON_MANDATORY_FULL_SUITE_PENDING: Final[str] = "mandatory_full_suite_pending"
REASON_REQUIRED_TIER_UNAVAILABLE: Final[str] = "required_tier_unavailable"
REASON_MECHANICAL_EXACT_WORK: Final[str] = "mechanical_exact_work"
REASON_LOCALIZED_EXACT_COUNTEREXAMPLE: Final[str] = "localized_exact_counterexample"
REASON_MULTI_FILE_SYNTHESIS: Final[str] = "multi_file_synthesis"
REASON_AMBIGUOUS_WORK: Final[str] = "ambiguous_work"
REASON_BROAD_DEPENDENCY_CONE: Final[str] = "broad_dependency_cone"
REASON_OPAQUE_CRITICAL_DEPENDENCY: Final[str] = "opaque_critical_dependency"
REASON_CONFLICTING_PROOF_REQUIREMENTS: Final[str] = "conflicting_proof_requirements"
REASON_SMALLER_ROUTE_FAILED: Final[str] = "smaller_route_failed"
REASON_CONTEXT_OVERFLOW: Final[str] = "context_overflow"
REASON_HUMAN_REVIEW_PLAN: Final[str] = "human_review_plan"
REASON_HIGH_RISK: Final[str] = "high_risk"

# Capability labels (provider-neutral).
CAP_MECHANICAL_TRANSFORM: Final[str] = "mechanical_transform"
CAP_BOUNDED_CONTEXT: Final[str] = "bounded_context"
CAP_LOCAL_EXECUTION: Final[str] = "local_execution"
CAP_MULTI_FILE_SYNTHESIS: Final[str] = "multi_file_synthesis"
CAP_FRONTIER_REASONING: Final[str] = "frontier_reasoning"
CAP_HUMAN_JUDGMENT: Final[str] = "human_judgment"


class ModelRouteError(VerificationContractError):
    """Fail-closed error while selecting a provider-neutral model route."""


class ModelRoutePolicyError(ModelRouteError):
    """Policy or inventory rejected as non-provider-neutral or malformed."""


# ---------------------------------------------------------------------------
# Closed vocabularies for routing facts
# ---------------------------------------------------------------------------


class AnalysisKind(str, Enum):
    """Closed analysis classification used by the route table."""

    MECHANICAL_FORMATTING = "mechanical_formatting"
    MECHANICAL_IMPORT = "mechanical_import"
    MECHANICAL_CODEMOD = "mechanical_codemod"
    MECHANICAL_RENAME = "mechanical_rename"
    LOCALIZED_EXACT = "localized_exact"
    LOCALIZED_CONSERVATIVE = "localized_conservative"
    MULTI_FILE_SYNTHESIS = "multi_file_synthesis"
    AMBIGUOUS = "ambiguous"
    BROAD = "broad"
    OPAQUE = "opaque"
    CONFLICTING = "conflicting"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class CounterexampleQuality(str, Enum):
    NONE = "none"
    POOR = "poor"
    GOOD = "good"
    MINIMIZED = "minimized"


class CapabilityLocality(str, Enum):
    """Provider-neutral placement class (not a vendor)."""

    LOCAL = "local"
    REMOTE = "remote"
    ANY = "any"


# ---------------------------------------------------------------------------
# Inventory and policy records
# ---------------------------------------------------------------------------


def _reject_provider_identity(payload: Mapping[str, Any], *, artifact: str) -> None:
    banned = sorted(key for key in payload if str(key).strip().lower() in _PROVIDER_IDENTITY_FIELDS)
    if banned:
        raise ModelRoutePolicyError(
            f"{artifact} must not carry provider identity fields: {banned}"
        )


def _as_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelRouteError(f"{field_name} must be an object")
    return value


def _enum_token(value: Any, enum_type: type[Enum], *, field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    if not isinstance(raw, str):
        raise ModelRouteError(f"{field_name} must be a string token")
    normalized = raw.strip().lower().replace("-", "_")
    try:
        return enum_type(normalized)
    except ValueError as exc:
        # Accept short aliases for analysis kinds / risk.
        aliases = _ANALYSIS_ALIASES if enum_type is AnalysisKind else {}
        if enum_type is RiskLevel:
            aliases = {"medium": RiskLevel.MODERATE}
        if enum_type is CounterexampleQuality:
            aliases = {
                "high": CounterexampleQuality.GOOD,
                "adequate": CounterexampleQuality.GOOD,
                "present_good": CounterexampleQuality.GOOD,
            }
        if enum_type is CapabilityLocality:
            aliases = {"on_device": CapabilityLocality.LOCAL, "cloud": CapabilityLocality.REMOTE}
        if normalized in aliases:
            return aliases[normalized]
        allowed = ", ".join(item.value for item in enum_type)
        raise ModelRouteError(f"{field_name} must be one of: {allowed}") from exc


_ANALYSIS_ALIASES: Final[dict[str, AnalysisKind]] = {
    "formatting": AnalysisKind.MECHANICAL_FORMATTING,
    "import": AnalysisKind.MECHANICAL_IMPORT,
    "codemod": AnalysisKind.MECHANICAL_CODEMOD,
    "rename": AnalysisKind.MECHANICAL_RENAME,
    "localized": AnalysisKind.LOCALIZED_EXACT,
    "bounded_localized": AnalysisKind.LOCALIZED_EXACT,
    "exact_localized": AnalysisKind.LOCALIZED_EXACT,
    "localized_conservative": AnalysisKind.LOCALIZED_CONSERVATIVE,
    "multi_file": AnalysisKind.MULTI_FILE_SYNTHESIS,
    "several_file": AnalysisKind.MULTI_FILE_SYNTHESIS,
    "several_file_synthesis": AnalysisKind.MULTI_FILE_SYNTHESIS,
    "synthesis": AnalysisKind.MULTI_FILE_SYNTHESIS,
    "ambiguity": AnalysisKind.AMBIGUOUS,
    "conflict": AnalysisKind.CONFLICTING,
    "unclassified": AnalysisKind.UNKNOWN,
}


def _bool(value: Any, *, field_name: str, default: bool | None = None) -> bool:
    if value is None and default is not None:
        return default
    if not isinstance(value, bool):
        raise ModelRouteError(f"{field_name} must be a boolean")
    return value


def _nonneg_int(value: Any, *, field_name: str, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    return _contract_integer(value, field_name=field_name, minimum=0, maximum=MAX_RESOURCE_QUANTITY)


def _route_enum(value: Any, *, field_name: str) -> ModelRoute:
    if isinstance(value, ModelRoute):
        return value
    raw = getattr(value, "value", value)
    try:
        return ModelRoute(raw)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in ModelRoute)
        raise ModelRouteError(f"{field_name} must be one of: {allowed}") from exc


@dataclass(frozen=True, slots=True)
class AvailableModelCapability:
    """One provider-neutral capability inventory row.

    Fields are limited to capability tier, context limit, locality, and
    availability.  Provider / vendor / model identifiers are rejected.
    """

    capability_tier: ModelRoute
    context_limit_tokens: int
    locality: CapabilityLocality
    available: bool
    schema: str = AVAILABLE_MODEL_CAPABILITY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capability_tier",
            _route_enum(self.capability_tier, field_name="capability_tier"),
        )
        if self.capability_tier is ModelRoute.HUMAN_REVIEW_REQUIRED:
            raise ModelRoutePolicyError(
                "available_models inventory must not list human_review_required as a tier"
            )
        object.__setattr__(
            self,
            "context_limit_tokens",
            _nonneg_int(self.context_limit_tokens, field_name="context_limit_tokens"),
        )
        object.__setattr__(
            self,
            "locality",
            _enum_token(self.locality, CapabilityLocality, field_name="locality"),
        )
        object.__setattr__(
            self, "available", _bool(self.available, field_name="available")
        )
        if self.schema != AVAILABLE_MODEL_CAPABILITY_SCHEMA:
            raise ModelRoutePolicyError("available model capability has unsupported schema")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "capability_tier": self.capability_tier.value,
            "context_limit_tokens": self.context_limit_tokens,
            "locality": self.locality.value
            if isinstance(self.locality, CapabilityLocality)
            else str(self.locality),
            "available": self.available,
        }

    @classmethod
    def from_value(cls, value: Any) -> AvailableModelCapability:
        if isinstance(value, cls):
            return value
        payload = _as_mapping(value, field_name="available_models item")
        _reject_provider_identity(payload, artifact="available_models item")
        unknown = set(payload) - {
            "schema",
            "capability_tier",
            "tier",
            "context_limit_tokens",
            "context_limit",
            "max_context_tokens",
            "locality",
            "available",
            "is_available",
        }
        if unknown:
            raise ModelRoutePolicyError(
                f"available_models item has unsupported fields: {sorted(unknown)}"
            )
        tier = payload.get("capability_tier", payload.get("tier"))
        context = payload.get(
            "context_limit_tokens",
            payload.get("context_limit", payload.get("max_context_tokens")),
        )
        available = payload.get("available", payload.get("is_available", True))
        locality = payload.get("locality", CapabilityLocality.ANY.value)
        return cls(
            capability_tier=tier,
            context_limit_tokens=context if context is not None else 0,
            locality=locality,
            available=available if available is not None else True,
            schema=str(payload.get("schema") or AVAILABLE_MODEL_CAPABILITY_SCHEMA),
        )


@dataclass(frozen=True, slots=True)
class PriorRepairAttempt:
    """One prior repair attempt used to escalate after smaller-route failure."""

    route: ModelRoute
    failed: bool
    reason_codes: tuple[str, ...] = ()
    schema: str = PRIOR_REPAIR_ATTEMPT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "route", _route_enum(self.route, field_name="route"))
        object.__setattr__(self, "failed", _bool(self.failed, field_name="failed"))
        codes = self.reason_codes or ()
        if isinstance(codes, (str, bytes)) or not isinstance(codes, Sequence):
            raise ModelRouteError("reason_codes must be a sequence")
        normalized: list[str] = []
        for item in codes:
            token = _contract_token(item, field_name="reason_codes")
            normalized.append(token)
        object.__setattr__(self, "reason_codes", tuple(normalized))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "route": self.route.value,
            "failed": self.failed,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_value(cls, value: Any) -> PriorRepairAttempt:
        if isinstance(value, cls):
            return value
        payload = _as_mapping(value, field_name="prior_attempts item")
        _reject_provider_identity(payload, artifact="prior_attempts item")
        return cls(
            route=payload.get("route", ""),
            failed=payload.get("failed", payload.get("success") is False),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            schema=str(payload.get("schema") or PRIOR_REPAIR_ATTEMPT_SCHEMA),
        )


@dataclass(frozen=True, slots=True)
class ModelRoutePolicy:
    """Bounded routing thresholds bound to a policy CID."""

    policy_cid: str
    max_context_tokens: int = _DEFAULT_MAX_CONTEXT_TOKENS
    small_max_changed_files: int = _DEFAULT_SMALL_MAX_FILES
    small_max_cone_size: int = _DEFAULT_SMALL_MAX_CONE
    medium_max_changed_files: int = _DEFAULT_MEDIUM_MAX_FILES
    medium_max_cone_size: int = _DEFAULT_MEDIUM_MAX_CONE
    opaque_critical_threshold: int = _DEFAULT_OPAQUE_CRITICAL_THRESHOLD
    schema: str = MODEL_ROUTE_POLICY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_cid", _contract_cid(self.policy_cid, field_name="policy_cid")
        )
        for name in (
            "max_context_tokens",
            "small_max_changed_files",
            "small_max_cone_size",
            "medium_max_changed_files",
            "medium_max_cone_size",
            "opaque_critical_threshold",
        ):
            object.__setattr__(
                self,
                name,
                _nonneg_int(getattr(self, name), field_name=name),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "policy_cid": self.policy_cid,
            "max_context_tokens": self.max_context_tokens,
            "small_max_changed_files": self.small_max_changed_files,
            "small_max_cone_size": self.small_max_cone_size,
            "medium_max_changed_files": self.medium_max_changed_files,
            "medium_max_cone_size": self.medium_max_cone_size,
            "opaque_critical_threshold": self.opaque_critical_threshold,
        }

    @classmethod
    def from_value(cls, value: Any) -> ModelRoutePolicy:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(policy_cid=value)
        payload = _as_mapping(value, field_name="policy")
        _reject_provider_identity(payload, artifact="policy")
        policy_cid = payload.get("policy_cid", payload.get("cid", ""))
        return cls(
            policy_cid=policy_cid,
            max_context_tokens=payload.get(
                "max_context_tokens", _DEFAULT_MAX_CONTEXT_TOKENS
            ),
            small_max_changed_files=payload.get(
                "small_max_changed_files", _DEFAULT_SMALL_MAX_FILES
            ),
            small_max_cone_size=payload.get(
                "small_max_cone_size", _DEFAULT_SMALL_MAX_CONE
            ),
            medium_max_changed_files=payload.get(
                "medium_max_changed_files", _DEFAULT_MEDIUM_MAX_FILES
            ),
            medium_max_cone_size=payload.get(
                "medium_max_cone_size", _DEFAULT_MEDIUM_MAX_CONE
            ),
            opaque_critical_threshold=payload.get(
                "opaque_critical_threshold", _DEFAULT_OPAQUE_CRITICAL_THRESHOLD
            ),
            schema=str(payload.get("schema") or MODEL_ROUTE_POLICY_SCHEMA),
        )


@dataclass(frozen=True, slots=True)
class ModelRouteFacts:
    """Normalized, table-driven inputs for route selection.

    Callers may construct this directly for hermetic unit tests, or let
    :func:`choose_model_route` derive it from context/plan/attempts.
    """

    context_token_estimate: int
    analysis_kind: AnalysisKind
    opaque_dependency_count: int = 0
    risk_level: RiskLevel = RiskLevel.LOW
    dependency_cone_size: int = 0
    unresolved_obligation_count: int = 0
    failure_kind: str = "none"
    counterexample_quality: CounterexampleQuality = CounterexampleQuality.NONE
    exact_contract_available: bool = False
    full_suite_required: bool = False
    full_suite_pending: bool = False
    environment_reproducible: bool = True
    scope_crossing: bool = False
    unresolved_authority: bool = False
    unmodeled_high_risk: bool = False
    proof_test_conflict: bool = False
    unsafe_context: bool = False
    changed_file_count: int = 0
    plan_human_review_required: bool = False
    plan_human_review_reason_codes: tuple[str, ...] = ()
    schema: str = MODEL_ROUTE_FACTS_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "context_token_estimate",
            _nonneg_int(self.context_token_estimate, field_name="context_token_estimate"),
        )
        object.__setattr__(
            self,
            "analysis_kind",
            _enum_token(self.analysis_kind, AnalysisKind, field_name="analysis_kind"),
        )
        object.__setattr__(
            self,
            "risk_level",
            _enum_token(self.risk_level, RiskLevel, field_name="risk_level"),
        )
        object.__setattr__(
            self,
            "counterexample_quality",
            _enum_token(
                self.counterexample_quality,
                CounterexampleQuality,
                field_name="counterexample_quality",
            ),
        )
        for name in (
            "opaque_dependency_count",
            "dependency_cone_size",
            "unresolved_obligation_count",
            "changed_file_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonneg_int(getattr(self, name), field_name=name),
            )
        for name in (
            "exact_contract_available",
            "full_suite_required",
            "full_suite_pending",
            "environment_reproducible",
            "scope_crossing",
            "unresolved_authority",
            "unmodeled_high_risk",
            "proof_test_conflict",
            "unsafe_context",
            "plan_human_review_required",
        ):
            object.__setattr__(
                self, name, _bool(getattr(self, name), field_name=name)
            )
        failure = self.failure_kind if self.failure_kind is not None else "none"
        object.__setattr__(
            self, "failure_kind", _contract_token(failure, field_name="failure_kind")
        )
        codes = self.plan_human_review_reason_codes or ()
        if isinstance(codes, (str, bytes)) or not isinstance(codes, Sequence):
            raise ModelRouteError("plan_human_review_reason_codes must be a sequence")
        object.__setattr__(
            self,
            "plan_human_review_reason_codes",
            tuple(_contract_token(item, field_name="plan_human_review_reason_codes") for item in codes),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "context_token_estimate": self.context_token_estimate,
            "analysis_kind": self.analysis_kind.value
            if isinstance(self.analysis_kind, AnalysisKind)
            else str(self.analysis_kind),
            "opaque_dependency_count": self.opaque_dependency_count,
            "risk_level": self.risk_level.value
            if isinstance(self.risk_level, RiskLevel)
            else str(self.risk_level),
            "dependency_cone_size": self.dependency_cone_size,
            "unresolved_obligation_count": self.unresolved_obligation_count,
            "failure_kind": self.failure_kind,
            "counterexample_quality": self.counterexample_quality.value
            if isinstance(self.counterexample_quality, CounterexampleQuality)
            else str(self.counterexample_quality),
            "exact_contract_available": self.exact_contract_available,
            "full_suite_required": self.full_suite_required,
            "full_suite_pending": self.full_suite_pending,
            "environment_reproducible": self.environment_reproducible,
            "scope_crossing": self.scope_crossing,
            "unresolved_authority": self.unresolved_authority,
            "unmodeled_high_risk": self.unmodeled_high_risk,
            "proof_test_conflict": self.proof_test_conflict,
            "unsafe_context": self.unsafe_context,
            "changed_file_count": self.changed_file_count,
            "plan_human_review_required": self.plan_human_review_required,
            "plan_human_review_reason_codes": list(self.plan_human_review_reason_codes),
        }

    @classmethod
    def from_value(cls, value: Any) -> ModelRouteFacts:
        if isinstance(value, cls):
            return value
        payload = _as_mapping(value, field_name="model_route_facts")
        _reject_provider_identity(payload, artifact="model_route_facts")
        return cls(
            context_token_estimate=payload.get("context_token_estimate", 0),
            analysis_kind=payload.get("analysis_kind", AnalysisKind.UNKNOWN.value),
            opaque_dependency_count=payload.get("opaque_dependency_count", 0),
            risk_level=payload.get("risk_level", RiskLevel.LOW.value),
            dependency_cone_size=payload.get("dependency_cone_size", 0),
            unresolved_obligation_count=payload.get("unresolved_obligation_count", 0),
            failure_kind=payload.get("failure_kind", "none"),
            counterexample_quality=payload.get(
                "counterexample_quality", CounterexampleQuality.NONE.value
            ),
            exact_contract_available=payload.get("exact_contract_available", False),
            full_suite_required=payload.get("full_suite_required", False),
            full_suite_pending=payload.get("full_suite_pending", False),
            environment_reproducible=payload.get("environment_reproducible", True),
            scope_crossing=payload.get("scope_crossing", False),
            unresolved_authority=payload.get("unresolved_authority", False),
            unmodeled_high_risk=payload.get("unmodeled_high_risk", False),
            proof_test_conflict=payload.get("proof_test_conflict", False),
            unsafe_context=payload.get("unsafe_context", False),
            changed_file_count=payload.get("changed_file_count", 0),
            plan_human_review_required=payload.get("plan_human_review_required", False),
            plan_human_review_reason_codes=tuple(
                payload.get("plan_human_review_reason_codes") or ()
            ),
            schema=str(payload.get("schema") or MODEL_ROUTE_FACTS_SCHEMA),
        )


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _mapping_get(payload: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return default


def _normalize_inventory(value: Any) -> tuple[AvailableModelCapability, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ModelRouteError("available_models must be a sequence")
    if len(value) > MAX_COLLECTION_ITEMS:
        raise ModelRouteError("available_models exceeds item bound")
    items = tuple(AvailableModelCapability.from_value(item) for item in value)
    return items


def _normalize_prior_attempts(value: Any) -> tuple[PriorRepairAttempt, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ModelRouteError("prior_attempts must be a sequence")
    if len(value) > MAX_COLLECTION_ITEMS:
        raise ModelRouteError("prior_attempts exceeds item bound")
    return tuple(PriorRepairAttempt.from_value(item) for item in value)


def _context_token_estimate(context_pack: Any) -> int:
    if context_pack is None:
        return 0
    if isinstance(context_pack, ModelRouteFacts):
        return context_pack.context_token_estimate
    if hasattr(context_pack, "token_estimate"):
        return _nonneg_int(
            getattr(context_pack, "token_estimate"), field_name="token_estimate"
        )
    if isinstance(context_pack, Mapping):
        return _nonneg_int(
            _mapping_get(
                context_pack,
                "token_estimate",
                "context_token_estimate",
                "context_tokens",
                default=0,
            ),
            field_name="context_token_estimate",
        )
    raise ModelRouteError("context_pack must be a mapping, view, or ModelRouteFacts")


def _exact_contracts_present(context_pack: Any) -> bool:
    if context_pack is None:
        return False
    if isinstance(context_pack, Mapping):
        if "exact_contract_available" in context_pack:
            return _bool(
                context_pack["exact_contract_available"],
                field_name="exact_contract_available",
            )
        contracts = context_pack.get("contracts")
        if isinstance(contracts, Sequence) and not isinstance(contracts, (str, bytes)):
            return len(contracts) > 0
        return bool(context_pack.get("exact_contracts"))
    contracts = getattr(context_pack, "contracts", None)
    if isinstance(contracts, Sequence) and not isinstance(contracts, (str, bytes)):
        return len(contracts) > 0
    return False


def derive_model_route_facts(
    context_pack: Any,
    verification_plan: Any,
    *,
    routing_hints: Mapping[str, Any] | None = None,
) -> ModelRouteFacts:
    """Derive :class:`ModelRouteFacts` from context/plan plus optional hints.

    ``routing_hints`` (or a facts object passed as ``context_pack``) supplies
    analysis classification and diagnostic signals that are not yet sealed on
    ``VerificationPlan`` itself.  When ``context_pack`` is already a
    :class:`ModelRouteFacts`, it is returned (after revalidation) unless the
    plan forces stronger human/full-suite flags.
    """

    hints: Mapping[str, Any] = MappingProxyType(dict(routing_hints or {}))
    if isinstance(context_pack, ModelRouteFacts):
        base = context_pack.to_dict()
    elif isinstance(context_pack, Mapping) and (
        "analysis_kind" in context_pack or context_pack.get("schema") == MODEL_ROUTE_FACTS_SCHEMA
    ):
        base = dict(context_pack)
    else:
        base = {
            "context_token_estimate": _context_token_estimate(context_pack),
            "exact_contract_available": _exact_contracts_present(context_pack),
        }

    # Overlay explicit routing hints (analysis kind, risk, cone, etc.).
    for key, value in hints.items():
        if key in _PROVIDER_IDENTITY_FIELDS:
            raise ModelRoutePolicyError(
                f"routing_hints must not carry provider identity field: {key}"
            )
        base[key] = value

    # Plan-derived authority flags (fail-closed merge: plan may only strengthen).
    if verification_plan is None:
        plan_map: Mapping[str, Any] = {}
    elif isinstance(verification_plan, VerificationPlan):
        plan_map = {
            "full_suite_required": verification_plan.full_suite_required,
            "full_suite_pending": bool(verification_plan.full_suite_required),
            "plan_human_review_required": verification_plan.human_review_required,
            "plan_human_review_reason_codes": verification_plan.human_review_reason_codes,
            "unresolved_obligation_count": len(
                verification_plan.affected_proof_obligation_cids
            ),
            "dependency_cone_size": max(
                len(verification_plan.affected_tests),
                len(verification_plan.fallback_tests),
            ),
            "policy_cid": verification_plan.policy_cid,
        }
        # full_suite_pending: required suite is still pending when plan requires it
        # and no execution bundle has cleared it; planner-time route assumes pending.
        if verification_plan.full_suite_required:
            plan_map["full_suite_pending"] = True
    elif isinstance(verification_plan, Mapping):
        _reject_provider_identity(verification_plan, artifact="verification_plan")
        plan_map = dict(verification_plan)
    else:
        raise ModelRouteError(
            "verification_plan must be a VerificationPlan, mapping, or None"
        )

    if plan_map.get("full_suite_required"):
        base["full_suite_required"] = True
        # Pending unless an explicit False is supplied in base/hints after plan.
        if "full_suite_pending" not in base and "full_suite_pending" not in hints:
            base["full_suite_pending"] = bool(
                plan_map.get("full_suite_pending", True)
            )
        elif plan_map.get("full_suite_pending") is True:
            base["full_suite_pending"] = True

    if plan_map.get("plan_human_review_required") or plan_map.get("human_review_required"):
        base["plan_human_review_required"] = True
        codes = plan_map.get("plan_human_review_reason_codes") or plan_map.get(
            "human_review_reason_codes"
        ) or ()
        existing = tuple(base.get("plan_human_review_reason_codes") or ())
        base["plan_human_review_reason_codes"] = tuple(existing) + tuple(codes)

    for key in (
        "unresolved_obligation_count",
        "dependency_cone_size",
        "scope_crossing",
        "environment_reproducible",
        "unresolved_authority",
        "unmodeled_high_risk",
        "proof_test_conflict",
        "unsafe_context",
        "opaque_dependency_count",
        "changed_file_count",
        "risk_level",
        "analysis_kind",
        "counterexample_quality",
        "failure_kind",
        "exact_contract_available",
        "context_token_estimate",
    ):
        if key in plan_map and key not in hints and (
            key not in base
            or base.get(key) in (None, 0, False, "", "none", "unknown", "low")
            and plan_map[key] not in (None, "", ())
        ):
            # Prefer explicit base/hints; fill gaps from plan mapping.
            if key not in base or base.get(key) in (None,):
                base[key] = plan_map[key]
            elif key in (
                "scope_crossing",
                "unresolved_authority",
                "unmodeled_high_risk",
                "proof_test_conflict",
                "unsafe_context",
            ) and plan_map[key]:
                base[key] = True
            elif key == "environment_reproducible" and plan_map[key] is False:
                base[key] = False
            elif key in (
                "unresolved_obligation_count",
                "dependency_cone_size",
                "opaque_dependency_count",
                "changed_file_count",
                "context_token_estimate",
            ):
                try:
                    base[key] = max(int(base.get(key) or 0), int(plan_map[key] or 0))
                except (TypeError, ValueError):
                    base[key] = plan_map[key]

    if "context_token_estimate" not in base:
        base["context_token_estimate"] = _context_token_estimate(context_pack)
    if "analysis_kind" not in base:
        base["analysis_kind"] = AnalysisKind.UNKNOWN.value

    return ModelRouteFacts.from_value(base)


# ---------------------------------------------------------------------------
# Decision table
# ---------------------------------------------------------------------------


def _failed_routes(prior_attempts: Sequence[PriorRepairAttempt]) -> frozenset[ModelRoute]:
    return frozenset(item.route for item in prior_attempts if item.failed)


def _tier_available(
    inventory: Sequence[AvailableModelCapability],
    tier: ModelRoute,
    *,
    context_token_estimate: int,
) -> bool:
    if tier is ModelRoute.DETERMINISTIC_ONLY:
        return True
    if tier is ModelRoute.HUMAN_REVIEW_REQUIRED:
        return True
    for item in inventory:
        if (
            item.available
            and item.capability_tier is tier
            and item.context_limit_tokens >= context_token_estimate
        ):
            return True
    return False


def _is_mechanical(kind: AnalysisKind | str) -> bool:
    value = kind.value if isinstance(kind, AnalysisKind) else str(kind)
    return value in _MECHANICAL_KINDS or value.startswith("mechanical_")


def _is_localized(kind: AnalysisKind | str) -> bool:
    value = kind.value if isinstance(kind, AnalysisKind) else str(kind)
    return value in _LOCALIZED_KINDS or value.startswith("localized_")


def _is_multi_file(kind: AnalysisKind | str) -> bool:
    value = kind.value if isinstance(kind, AnalysisKind) else str(kind)
    return value in _MULTI_FILE_KINDS


def _is_ambiguous_class(kind: AnalysisKind | str) -> bool:
    value = kind.value if isinstance(kind, AnalysisKind) else str(kind)
    return value in _AMBIGUOUS_KINDS


def _good_counterexample(quality: CounterexampleQuality | str) -> bool:
    value = quality.value if isinstance(quality, CounterexampleQuality) else str(quality)
    return value in _GOOD_COUNTEREXAMPLES


def _risk_token(level: RiskLevel | str) -> str:
    return level.value if isinstance(level, RiskLevel) else str(level)


def _capabilities_for(route: ModelRoute) -> tuple[str, ...]:
    if route is ModelRoute.DETERMINISTIC_ONLY:
        return (CAP_MECHANICAL_TRANSFORM,)
    if route is ModelRoute.SMALL_LOCAL_MODEL:
        return (CAP_BOUNDED_CONTEXT, CAP_LOCAL_EXECUTION)
    if route is ModelRoute.MEDIUM_MODEL:
        return (CAP_BOUNDED_CONTEXT, CAP_MULTI_FILE_SYNTHESIS)
    if route is ModelRoute.FRONTIER_MODEL:
        return (CAP_FRONTIER_REASONING, CAP_BOUNDED_CONTEXT)
    return (CAP_HUMAN_JUDGMENT,)


def _decision(
    *,
    route: ModelRoute,
    reasons: Sequence[str],
    context_token_estimate: int,
    policy_cid: str,
    considered: Sequence[ModelRoute] = _ALL_ROUTES,
) -> ModelRouteDecision:
    # Ensure selected route is present and unique in considered_routes.
    ordered: list[ModelRoute] = []
    for item in considered:
        if item not in ordered:
            ordered.append(item)
    if route not in ordered:
        ordered.append(route)
    reason_codes = tuple(dict.fromkeys(reasons))  # stable unique
    if not reason_codes:
        reason_codes = ("route_selected",)
    return ModelRouteDecision(
        route=route,
        considered_routes=tuple(ordered),
        decisive_reason_codes=reason_codes,
        required_capabilities=_capabilities_for(route),
        context_token_estimate=context_token_estimate,
        policy_cid=policy_cid,
    )


def select_required_route(
    facts: ModelRouteFacts,
    prior_attempts: Sequence[PriorRepairAttempt],
    policy: ModelRoutePolicy,
) -> tuple[ModelRoute, tuple[str, ...]]:
    """Return the safely required route and decisive reason codes.

    Availability is *not* applied here; callers enforce no-downgrade inventory
    checks after this pure table evaluation.
    """

    facts = ModelRouteFacts.from_value(facts)
    failed = _failed_routes(prior_attempts)
    reasons: list[str] = []

    # --- 1. Human-review gates (absolute precedence) ---
    if facts.unresolved_authority:
        return ModelRoute.HUMAN_REVIEW_REQUIRED, (REASON_UNRESOLVED_AUTHORITY,)
    if facts.unmodeled_high_risk:
        return ModelRoute.HUMAN_REVIEW_REQUIRED, (REASON_UNMODELED_HIGH_RISK,)
    if facts.scope_crossing:
        return ModelRoute.HUMAN_REVIEW_REQUIRED, (REASON_SCOPE_CROSSING,)
    if facts.proof_test_conflict:
        return ModelRoute.HUMAN_REVIEW_REQUIRED, (REASON_PROOF_TEST_CONFLICT,)
    if facts.unsafe_context:
        return ModelRoute.HUMAN_REVIEW_REQUIRED, (REASON_UNSAFE_CONTEXT,)
    if not facts.environment_reproducible:
        return ModelRoute.HUMAN_REVIEW_REQUIRED, (REASON_NONREPRODUCIBLE_ENVIRONMENT,)
    if facts.plan_human_review_required:
        codes = facts.plan_human_review_reason_codes or (REASON_HUMAN_REVIEW_PLAN,)
        return ModelRoute.HUMAN_REVIEW_REQUIRED, codes
    if facts.full_suite_required and facts.full_suite_pending:
        return (
            ModelRoute.HUMAN_REVIEW_REQUIRED,
            (REASON_VERIFICATION_INCOMPLETE, REASON_MANDATORY_FULL_SUITE_PENDING),
        )
    if _risk_token(facts.risk_level) == RiskLevel.HIGH.value and facts.unmodeled_high_risk:
        return ModelRoute.HUMAN_REVIEW_REQUIRED, (REASON_UNMODELED_HIGH_RISK,)
    # High risk without an explicit modeled-safe flag still fails closed when
    # analysis is opaque/unclassified.
    if (
        _risk_token(facts.risk_level) == RiskLevel.HIGH.value
        and _is_ambiguous_class(facts.analysis_kind)
    ):
        return ModelRoute.HUMAN_REVIEW_REQUIRED, (REASON_HIGH_RISK, REASON_UNMODELED_HIGH_RISK)

    # --- Context overflow forces frontier (or human later if unavailable) ---
    context_overflow = facts.context_token_estimate > policy.max_context_tokens

    # --- Escalate past routes that already failed ---
    def _not_failed(route: ModelRoute) -> bool:
        return route not in failed

    # --- 2. Mechanical exact work => deterministic ---
    # Mechanical formatting/import/codemod/rename is treated as exact by kind.
    # exact_contract_available remains an optional reinforcing signal.
    if (
        _is_mechanical(facts.analysis_kind)
        and _risk_token(facts.risk_level) in _LOW_MODERATE_RISK
        and not context_overflow
        and facts.opaque_dependency_count == 0
        and facts.changed_file_count <= policy.small_max_changed_files
        and _not_failed(ModelRoute.DETERMINISTIC_ONLY)
    ):
        kind_token = (
            facts.analysis_kind.value
            if isinstance(facts.analysis_kind, AnalysisKind)
            else str(facts.analysis_kind)
        )
        return ModelRoute.DETERMINISTIC_ONLY, (
            REASON_MECHANICAL_EXACT_WORK,
            kind_token,
        )

    # Mechanical without exact contracts, or after deterministic failure, falls through.
    if _is_mechanical(facts.analysis_kind) and ModelRoute.DETERMINISTIC_ONLY in failed:
        reasons.append(REASON_SMALLER_ROUTE_FAILED)

    # --- 3. Localized + good counterexample => small ---
    # Failed deterministic may escalate into small when the residual work is
    # still bounded/localized with a good counterexample.
    small_candidate = (
        _is_localized(facts.analysis_kind)
        or (
            _is_mechanical(facts.analysis_kind)
            and ModelRoute.DETERMINISTIC_ONLY in failed
        )
    )
    if (
        small_candidate
        and _good_counterexample(facts.counterexample_quality)
        and _risk_token(facts.risk_level) in _LOW_MODERATE_RISK
        and facts.changed_file_count <= policy.small_max_changed_files
        and facts.dependency_cone_size <= policy.small_max_cone_size
        and facts.opaque_dependency_count == 0
        and not context_overflow
        and _not_failed(ModelRoute.SMALL_LOCAL_MODEL)
    ):
        codes = [REASON_LOCALIZED_EXACT_COUNTEREXAMPLE]
        if ModelRoute.DETERMINISTIC_ONLY in failed:
            codes.insert(0, REASON_SMALLER_ROUTE_FAILED)
        return ModelRoute.SMALL_LOCAL_MODEL, tuple(codes)

    # Localized without good counterexample is not "small" — escalate.
    localized_without_good_cx = _is_localized(facts.analysis_kind) and not _good_counterexample(
        facts.counterexample_quality
    )

    # --- 4. Multi-file synthesis without opaque critical => medium ---
    smaller_failed = bool(
        failed
        & {
            ModelRoute.DETERMINISTIC_ONLY,
            ModelRoute.SMALL_LOCAL_MODEL,
        }
    )
    if smaller_failed:
        reasons.append(REASON_SMALLER_ROUTE_FAILED)

    opaque_critical = facts.opaque_dependency_count >= policy.opaque_critical_threshold
    if opaque_critical:
        reasons.append(REASON_OPAQUE_CRITICAL_DEPENDENCY)

    broad_cone = facts.dependency_cone_size > policy.medium_max_cone_size
    if broad_cone:
        reasons.append(REASON_BROAD_DEPENDENCY_CONE)

    if context_overflow:
        reasons.append(REASON_CONTEXT_OVERFLOW)

    conflicting = facts.analysis_kind is AnalysisKind.CONFLICTING or facts.failure_kind in {
        "proof_conflict",
        "conflicting_proof",
        "conflicting",
    }
    if conflicting:
        reasons.append(REASON_CONFLICTING_PROOF_REQUIREMENTS)

    if _is_ambiguous_class(facts.analysis_kind) or localized_without_good_cx:
        reasons.append(REASON_AMBIGUOUS_WORK)

    # Failed medium always forces frontier. Failed small forces frontier when
    # residual work is already opaque/broad/ambiguous/overflowing.
    frontier_forced = bool(
        opaque_critical
        or broad_cone
        or context_overflow
        or conflicting
        or _is_ambiguous_class(facts.analysis_kind)
        or ModelRoute.MEDIUM_MODEL in failed
        or (
            ModelRoute.SMALL_LOCAL_MODEL in failed
            and (
                opaque_critical
                or broad_cone
                or context_overflow
                or _is_ambiguous_class(facts.analysis_kind)
                or conflicting
            )
        )
    )

    medium_shape = (
        _is_multi_file(facts.analysis_kind)
        or (
            smaller_failed
            and (
                _is_localized(facts.analysis_kind)
                or _is_mechanical(facts.analysis_kind)
                or _is_multi_file(facts.analysis_kind)
            )
        )
        or (
            facts.changed_file_count > policy.small_max_changed_files
            and facts.changed_file_count <= policy.medium_max_changed_files
            and not _is_ambiguous_class(facts.analysis_kind)
        )
    )

    if (
        not frontier_forced
        and medium_shape
        and not opaque_critical
        and not broad_cone
        and not context_overflow
        and _not_failed(ModelRoute.MEDIUM_MODEL)
        and facts.changed_file_count <= policy.medium_max_changed_files
        and facts.dependency_cone_size <= policy.medium_max_cone_size
        and _risk_token(facts.risk_level) != RiskLevel.HIGH.value
    ):
        codes = [REASON_MULTI_FILE_SYNTHESIS]
        if smaller_failed:
            codes.insert(0, REASON_SMALLER_ROUTE_FAILED)
        return ModelRoute.MEDIUM_MODEL, tuple(dict.fromkeys(codes))

    # --- 5. Frontier for ambiguous / broad / opaque / failed-smaller / overflow ---
    frontier_reasons = list(reasons)
    if not frontier_reasons:
        if _is_ambiguous_class(facts.analysis_kind):
            frontier_reasons.append(REASON_AMBIGUOUS_WORK)
        elif opaque_critical:
            frontier_reasons.append(REASON_OPAQUE_CRITICAL_DEPENDENCY)
        elif broad_cone:
            frontier_reasons.append(REASON_BROAD_DEPENDENCY_CONE)
        elif context_overflow:
            frontier_reasons.append(REASON_CONTEXT_OVERFLOW)
        elif smaller_failed or ModelRoute.MEDIUM_MODEL in failed:
            frontier_reasons.append(REASON_SMALLER_ROUTE_FAILED)
        else:
            frontier_reasons.append(REASON_AMBIGUOUS_WORK)

    if ModelRoute.FRONTIER_MODEL in failed:
        # Even frontier failed: human review.
        return (
            ModelRoute.HUMAN_REVIEW_REQUIRED,
            tuple(dict.fromkeys([REASON_SMALLER_ROUTE_FAILED, *frontier_reasons])),
        )

    return ModelRoute.FRONTIER_MODEL, tuple(dict.fromkeys(frontier_reasons))


def apply_availability(
    *,
    required_route: ModelRoute,
    reasons: Sequence[str],
    facts: ModelRouteFacts,
    inventory: Sequence[AvailableModelCapability],
    policy: ModelRoutePolicy,
) -> ModelRouteDecision:
    """Enforce no-downgrade inventory availability for the required tier."""

    if required_route is ModelRoute.HUMAN_REVIEW_REQUIRED:
        return _decision(
            route=ModelRoute.HUMAN_REVIEW_REQUIRED,
            reasons=reasons,
            context_token_estimate=facts.context_token_estimate,
            policy_cid=policy.policy_cid,
        )

    if required_route is ModelRoute.DETERMINISTIC_ONLY:
        return _decision(
            route=ModelRoute.DETERMINISTIC_ONLY,
            reasons=reasons,
            context_token_estimate=facts.context_token_estimate,
            policy_cid=policy.policy_cid,
        )

    if _tier_available(
        inventory,
        required_route,
        context_token_estimate=facts.context_token_estimate,
    ):
        return _decision(
            route=required_route,
            reasons=reasons,
            context_token_estimate=facts.context_token_estimate,
            policy_cid=policy.policy_cid,
        )

    # Required model tier unavailable: never downgrade.
    return _decision(
        route=ModelRoute.HUMAN_REVIEW_REQUIRED,
        reasons=(REASON_REQUIRED_TIER_UNAVAILABLE, *reasons),
        context_token_estimate=facts.context_token_estimate,
        policy_cid=policy.policy_cid,
    )


def decide_model_route(
    facts: ModelRouteFacts | Mapping[str, Any],
    *,
    prior_attempts: Sequence[Any] = (),
    available_models: Sequence[Any] = (),
    policy: Any,
) -> ModelRouteDecision:
    """Pure decision entrypoint over already-normalized (or mappable) facts."""

    normalized_facts = ModelRouteFacts.from_value(facts)
    normalized_attempts = _normalize_prior_attempts(prior_attempts)
    inventory = _normalize_inventory(available_models)
    normalized_policy = ModelRoutePolicy.from_value(policy)
    required, reasons = select_required_route(
        normalized_facts, normalized_attempts, normalized_policy
    )
    decision = apply_availability(
        required_route=required,
        reasons=reasons,
        facts=normalized_facts,
        inventory=inventory,
        policy=normalized_policy,
    )
    _assert_provider_neutral_decision(decision)
    return decision


def _assert_provider_neutral_decision(decision: ModelRouteDecision) -> None:
    payload = decision.to_record()
    _reject_provider_identity(payload, artifact="ModelRouteDecision")
    # String-scan reason codes and capabilities for vendor tokens.
    banned_substrings = (
        "openai",
        "anthropic",
        "grok",
        "gemini",
        "codex",
        "ollama",
        "huggingface",
        "vendor",
        "provider",
    )
    for field_name in ("decisive_reason_codes", "required_capabilities"):
        for item in payload.get(field_name) or ():
            lowered = str(item).lower()
            for banned in banned_substrings:
                if banned in lowered:
                    raise ModelRoutePolicyError(
                        f"ModelRouteDecision {field_name} must not contain provider identity"
                    )


def choose_model_route(
    context_pack: Any,
    verification_plan: Any,
    prior_attempts: Sequence[Any],
    available_models: Sequence[Any],
    policy: Any,
    *,
    routing_hints: Mapping[str, Any] | None = None,
) -> ModelRouteDecision:
    """Select the provider-neutral next-repair capability class.

    Parameters mirror the sealed production API:

    * ``context_pack`` — context view/mapping or prebuilt :class:`ModelRouteFacts`
    * ``verification_plan`` — :class:`VerificationPlan` or mapping
    * ``prior_attempts`` — prior repair attempts (failed routes escalate)
    * ``available_models`` — provider-neutral capability inventory
    * ``policy`` — policy CID string or :class:`ModelRoutePolicy` mapping

    ``routing_hints`` is an optional test/integration escape hatch for analysis
    classification and diagnostic signals not yet sealed on the plan object.
    """

    facts = derive_model_route_facts(
        context_pack, verification_plan, routing_hints=routing_hints
    )
    return decide_model_route(
        facts,
        prior_attempts=prior_attempts,
        available_models=available_models,
        policy=policy,
    )


class ModelRoutePlanner:
    """Narrow collaborator that chooses provider-neutral repair routes."""

    SCHEMA: Final[str] = MODEL_ROUTE_PLANNER_SCHEMA
    INTERFACE: Final[str] = MODEL_ROUTE_PLANNER_INTERFACE
    EVIDENCE: Final[str] = MODEL_ROUTE_EVIDENCE

    def __init__(self, *, default_policy: Any | None = None) -> None:
        self._default_policy = (
            None if default_policy is None else ModelRoutePolicy.from_value(default_policy)
        )

    def choose(
        self,
        context_pack: Any,
        verification_plan: Any,
        prior_attempts: Sequence[Any],
        available_models: Sequence[Any],
        policy: Any | None = None,
        *,
        routing_hints: Mapping[str, Any] | None = None,
    ) -> ModelRouteDecision:
        effective_policy = policy
        if effective_policy is None:
            if self._default_policy is None:
                raise ModelRoutePolicyError("policy is required")
            effective_policy = self._default_policy
        return choose_model_route(
            context_pack,
            verification_plan,
            prior_attempts,
            available_models,
            effective_policy,
            routing_hints=routing_hints,
        )

    def decide(
        self,
        facts: ModelRouteFacts | Mapping[str, Any],
        *,
        prior_attempts: Sequence[Any] = (),
        available_models: Sequence[Any] = (),
        policy: Any | None = None,
    ) -> ModelRouteDecision:
        effective_policy = policy
        if effective_policy is None:
            if self._default_policy is None:
                raise ModelRoutePolicyError("policy is required")
            effective_policy = self._default_policy
        return decide_model_route(
            facts,
            prior_attempts=prior_attempts,
            available_models=available_models,
            policy=effective_policy,
        )


def default_inventory(
    *,
    include_deterministic: bool = True,
    small: bool = True,
    medium: bool = True,
    frontier: bool = True,
    context_limit_tokens: int = _DEFAULT_MAX_CONTEXT_TOKENS,
) -> tuple[AvailableModelCapability, ...]:
    """Build a hermetic provider-neutral inventory for tests and local runs."""

    rows: list[AvailableModelCapability] = []
    if include_deterministic:
        rows.append(
            AvailableModelCapability(
                capability_tier=ModelRoute.DETERMINISTIC_ONLY,
                context_limit_tokens=context_limit_tokens,
                locality=CapabilityLocality.LOCAL,
                available=True,
            )
        )
    if small:
        rows.append(
            AvailableModelCapability(
                capability_tier=ModelRoute.SMALL_LOCAL_MODEL,
                context_limit_tokens=context_limit_tokens,
                locality=CapabilityLocality.LOCAL,
                available=True,
            )
        )
    if medium:
        rows.append(
            AvailableModelCapability(
                capability_tier=ModelRoute.MEDIUM_MODEL,
                context_limit_tokens=context_limit_tokens,
                locality=CapabilityLocality.ANY,
                available=True,
            )
        )
    if frontier:
        rows.append(
            AvailableModelCapability(
                capability_tier=ModelRoute.FRONTIER_MODEL,
                context_limit_tokens=context_limit_tokens,
                locality=CapabilityLocality.ANY,
                available=True,
            )
        )
    return tuple(rows)


def policy_cid_for(label: str) -> str:
    """Deterministic policy CID helper for hermetic fixtures."""

    return cid_for_dag_json(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/verification-route-policy-label@1",
            "label": label,
        }
    )


__all__ = [
    "AVAILABLE_MODEL_CAPABILITY_SCHEMA",
    "AnalysisKind",
    "AvailableModelCapability",
    "CAP_BOUNDED_CONTEXT",
    "CAP_FRONTIER_REASONING",
    "CAP_HUMAN_JUDGMENT",
    "CAP_LOCAL_EXECUTION",
    "CAP_MECHANICAL_TRANSFORM",
    "CAP_MULTI_FILE_SYNTHESIS",
    "CapabilityLocality",
    "CounterexampleQuality",
    "MODEL_ROUTE_EVIDENCE",
    "MODEL_ROUTE_FACTS_SCHEMA",
    "MODEL_ROUTE_PLANNER_INTERFACE",
    "MODEL_ROUTE_PLANNER_SCHEMA",
    "MODEL_ROUTE_POLICY_SCHEMA",
    "ModelRoute",
    "ModelRouteDecision",
    "ModelRouteError",
    "ModelRouteFacts",
    "ModelRoutePlanner",
    "ModelRoutePolicy",
    "ModelRoutePolicyError",
    "PRIOR_REPAIR_ATTEMPT_SCHEMA",
    "PriorRepairAttempt",
    "REASON_AMBIGUOUS_WORK",
    "REASON_BROAD_DEPENDENCY_CONE",
    "REASON_CONFLICTING_PROOF_REQUIREMENTS",
    "REASON_CONTEXT_OVERFLOW",
    "REASON_LOCALIZED_EXACT_COUNTEREXAMPLE",
    "REASON_MANDATORY_FULL_SUITE_PENDING",
    "REASON_MECHANICAL_EXACT_WORK",
    "REASON_MULTI_FILE_SYNTHESIS",
    "REASON_NONREPRODUCIBLE_ENVIRONMENT",
    "REASON_OPAQUE_CRITICAL_DEPENDENCY",
    "REASON_PROOF_TEST_CONFLICT",
    "REASON_REQUIRED_TIER_UNAVAILABLE",
    "REASON_SCOPE_CROSSING",
    "REASON_SMALLER_ROUTE_FAILED",
    "REASON_UNMODELED_HIGH_RISK",
    "REASON_UNRESOLVED_AUTHORITY",
    "REASON_UNSAFE_CONTEXT",
    "REASON_VERIFICATION_INCOMPLETE",
    "RiskLevel",
    "apply_availability",
    "choose_model_route",
    "decide_model_route",
    "default_inventory",
    "derive_model_route_facts",
    "policy_cid_for",
    "select_required_route",
]
