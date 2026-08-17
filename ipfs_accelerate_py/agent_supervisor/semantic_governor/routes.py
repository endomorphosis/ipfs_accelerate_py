"""Calibrate model routes separately from context sufficiency (SCG-030).

Runtime route calibration over ``GovernorRunReceipt`` histories and
provider-neutral ``ModelRoutePlanner`` decisions. Empirical counters change
future *threshold proposals* and audit frequency only; they never mutate
production routes in place and never fold into context-sufficiency claims.

Normative fail-closed invariants:

* **Separate counters** — ``context_omission_failure_count`` and
  ``reasoning_failure_count`` are independent. Context sufficiency outcomes
  do not rewrite route-calibration identity or merge the two failure kinds.
* **No downgrade** — when the safely required capability tier is unavailable,
  routing escalates to ``human`` review; it never selects a weaker tier.
* **Proposals only** — ``propose_route_threshold_change`` emits sealed
  proposals. Direct production route mutation and provider-ID selection are
  rejected.
* **Capability tier only** — closed vocabulary is
  deterministic / small / medium / frontier / human. Vendor and provider
  identity fields are rejected from durable payloads.
* **High-risk floor** — high-risk requirements never auto-lower via normal
  proposals.

Importing this module performs no I/O, opens no sockets, and never invokes a
provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Iterable, Mapping, Sequence
import re
import unicodedata

from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_structured,
    validate_cid,
    validate_structured_value,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.audit_contracts import (
    BASIS_POINTS,
    RouteTier,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.base import (
    AssumptionKind,
    ArtifactProvenance,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    GovernorArtifactHeader,
    GovernorAssumption,
    GovernorTerminalStatus,
    SemanticGovernorBaseError,
    reject_private_and_model_authority,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.calibration_contracts import (
    EvidencePartition,
    ratio_to_basis_points,
)

from ipfs_accelerate_py.agent_supervisor.semantic_governor.contracts import (
    SemanticGovernorExecutionError,
)

# ---------------------------------------------------------------------------
# Evidence / interface / schema constants
# ---------------------------------------------------------------------------

SCG_ROUTE_CALIBRATION_EVIDENCE: Final[str] = "scg/route-calibration@1"

UPDATE_MODEL_ROUTE_CALIBRATION_INTERFACE: Final[str] = (
    "update_model_route_calibration@1"
)
PROPOSE_ROUTE_THRESHOLD_CHANGE_INTERFACE: Final[str] = (
    "propose_route_threshold_change@1"
)

ROUTE_CALIBRATION_STATE_INTERFACE: Final[str] = "ModelRouteCalibrationState@1"
ROUTE_CALIBRATION_STATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "model-route-calibration-state@1"
)
ROUTE_TIER_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "route-tier-metrics@1"
)
ROUTE_RUN_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "route-run-observation@1"
)
ROUTE_CALIBRATION_UPDATE_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "route-calibration-update-result@1"
)
ROUTE_THRESHOLD_PROPOSAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "route-threshold-proposal@1"
)
ROUTE_THRESHOLD_PROPOSAL_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "route-threshold-proposal-result@1"
)
ROUTE_AVAILABILITY_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "route-availability-decision@1"
)
ROUTE_THRESHOLD_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "route-threshold-policy@1"
)

GENERATOR_ID: Final[str] = "semantic_governor_route_calibration"
GENERATOR_VERSION: Final[str] = "1.0.0"
PRODUCER_ID: Final[str] = "semantic_governor"
PRODUCER_VERSION: Final[str] = "1"
TOOL_ID: Final[str] = "route_calibration.v1"

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_METADATA_KEYS: Final[int] = 64
MAX_CID_LIST: Final[int] = 4_096
MAX_OBSERVATIONS: Final[int] = 4_096
MAX_PROPOSALS: Final[int] = 64
MAX_REASON_CODES: Final[int] = 64
MAX_REVISION: Final[int] = 2**63 - 1
MAX_COUNTER: Final[int] = 2**63 - 1

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")

# Closed capability tiers (plan vocabulary). Deterministic/small/medium/frontier/human.
_ROUTE_TIERS: Final[tuple[str, ...]] = (
    RouteTier.DETERMINISTIC.value,
    RouteTier.SMALL.value,
    RouteTier.MEDIUM.value,
    RouteTier.FRONTIER.value,
    RouteTier.HUMAN.value,
)

# Capability rank for no-downgrade comparisons (higher = stronger).
_ROUTE_RANK: Final[Mapping[str, int]] = MappingProxyType(
    {
        RouteTier.DETERMINISTIC.value: 0,
        RouteTier.SMALL.value: 1,
        RouteTier.MEDIUM.value: 2,
        RouteTier.FRONTIER.value: 3,
        RouteTier.HUMAN.value: 4,
    }
)

# Fields that must never appear on durable route-calibration payloads.
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

# Context-sufficiency fields that must not be rewritten by route calibration.
_CONTEXT_SUFFICIENCY_AUTHORITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "sufficiency_state",
        "context_sufficiency_state",
        "sufficiency_claim_cid",
        "context_sufficiency_claim",
        "evaluate_context_sufficiency",
    }
)

# Default proposal thresholds (basis points / absolute floors).
DEFAULT_MIN_ACCEPTED_RATE_BP: Final[int] = 7_000
DEFAULT_MAX_OMISSION_RATE_BP: Final[int] = 2_000
DEFAULT_MAX_REASONING_FAILURE_RATE_BP: Final[int] = 2_000
DEFAULT_MAX_RETRY_RATE_BP: Final[int] = 4_000
DEFAULT_MIN_USES_FOR_PROPOSAL: Final[int] = 5
DEFAULT_ESCALATION_OMISSION_RATE_BP: Final[int] = 3_000
DEFAULT_ESCALATION_REASONING_RATE_BP: Final[int] = 3_000


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RouteCalibrationError(SemanticGovernorExecutionError):
    """Raised when route calibration input is malformed or fail-closed."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "route_calibration_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


class RouteCalibrationContractError(RouteCalibrationError):
    """Raised when a sealed route-calibration artifact fails validation."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "route_calibration_contract_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, reason_code=reason_code, details=details)


# ---------------------------------------------------------------------------
# Closed enumerations
# ---------------------------------------------------------------------------


class RouteCalibrationDisposition(str, Enum):
    """Closed dispositions for a calibration update attempt."""

    APPLIED = "applied"
    SKIPPED_EMPTY = "skipped_empty"
    SKIPPED_IDEMPOTENT = "skipped_idempotent"
    SKIPPED_SIMULATED = "skipped_simulated"
    REJECTED_PROVIDER_IDENTITY = "rejected_provider_identity"
    REJECTED_CONTEXT_SUFFICIENCY_MUTATION = "rejected_context_sufficiency_mutation"
    REJECTED_MALFORMED = "rejected_malformed"


class RouteThresholdDisposition(str, Enum):
    """Closed dispositions for propose_route_threshold_change."""

    PROPOSED = "proposed"
    NO_CHANGE = "no_change"
    REJECTED_HIGH_RISK_REDUCTION = "rejected_high_risk_reduction"
    REJECTED_DOWNGRADE = "rejected_downgrade"
    REJECTED_PROVIDER_IDENTITY = "rejected_provider_identity"
    REJECTED_PRODUCTION_MUTATION = "rejected_production_mutation"
    REJECTED_MALFORMED = "rejected_malformed"


class RouteThresholdParameter(str, Enum):
    """Closed parameters that may appear on a threshold proposal."""

    MIN_ACCEPTED_RATE_BP = "min_accepted_rate_bp"
    MAX_CONTEXT_OMISSION_RATE_BP = "max_context_omission_rate_bp"
    MAX_REASONING_FAILURE_RATE_BP = "max_reasoning_failure_rate_bp"
    MAX_RETRY_RATE_BP = "max_retry_rate_bp"
    MIN_ROUTE_TIER = "min_route_tier"
    ESCALATE_ON_UNAVAILABLE = "escalate_on_unavailable"
    SHADOW_SAMPLE_RATE_BP = "shadow_sample_rate_bp"


class RouteFailureKind(str, Enum):
    """Closed primary failure attribution for a single observation.

    Omission and reasoning remain independently countable even when neither
    is the primary kind (both counters stay zero) or when diagnostics carry
    both flags for audit — counters never merge.
    """

    NONE = "none"
    CONTEXT_OMISSION = "context_omission"
    REASONING_FAILURE = "reasoning_failure"
    VERIFICATION_FAILURE = "verification_failure"
    UNAVAILABLE_TIER = "unavailable_tier"
    OTHER = "other"


class RouteAvailabilityDisposition(str, Enum):
    """Closed availability resolution outcomes (no downgrade)."""

    AVAILABLE = "available"
    UNAVAILABLE_ESCALATED_TO_HUMAN = "unavailable_escalated_to_human"
    ALREADY_HUMAN = "already_human"
    DETERMINISTIC_ALWAYS_AVAILABLE = "deterministic_always_available"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        raise RouteCalibrationError(f"{name} must be a nonempty string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise RouteCalibrationError(f"{name} must be trimmed NFC text")
    if len(value) > MAX_TEXT_CHARS or any(not char.isprintable() for char in value):
        raise RouteCalibrationError(f"{name} contains invalid text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if _TOKEN_RE.fullmatch(text) is None:
        raise RouteCalibrationError(
            f"{name} must be a lowercase token matching {_TOKEN_RE.pattern}"
        )
    return text


def _cid(value: Any, name: str) -> str:
    try:
        return validate_cid(value)
    except Exception as exc:
        raise RouteCalibrationError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise RouteCalibrationError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise RouteCalibrationError(f"{name} must be a nonnegative integer")
    if value > MAX_COUNTER:
        raise RouteCalibrationError(f"{name} exceeds maximum")
    return value


def _basis_points(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise RouteCalibrationError(
            f"{name} must be an integer basis-point ratio in [0, {BASIS_POINTS}]"
        )
    if value < 0 or value > BASIS_POINTS:
        raise RouteCalibrationError(
            f"{name} must be an integer basis-point ratio in [0, {BASIS_POINTS}]"
        )
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    try:
        return enum_type(value).value
    except (TypeError, ValueError) as exc:
        raise RouteCalibrationError(f"{name} has unsupported value {value!r}") from exc


def _route_tier(value: Any, name: str = "route_tier") -> str:
    return _enum(value, RouteTier, name)


def _freeze_structured(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_structured(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_structured(item) for item in value)
    return value


def _thaw_structured(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_structured(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_structured(item) for item in value]
    return value


def _require_structured(value: Any, name: str) -> Any:
    thawed = _thaw_structured(value)
    try:
        validate_structured_value(thawed, path=name)
    except Exception as exc:
        raise RouteCalibrationError(
            f"{name} must be strict DAG-JSON without floats or host types"
        ) from exc
    reject_private_and_model_authority(thawed, path=name)
    return thawed


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RouteCalibrationError(f"{name} must be a mapping")
    if len(value) > MAX_METADATA_KEYS:
        raise RouteCalibrationError(f"{name} exceeds maximum keys")
    return _freeze_structured(_require_structured(dict(value), name))


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise RouteCalibrationError(f"{name} must be a mapping")
    actual = set(data)
    if actual != fields:
        raise RouteCalibrationError(
            f"{name} fields must be exactly {sorted(fields)}, got {sorted(actual)}"
        )
    return dict(data)


def _unique_sorted_cids(values: Iterable[Any], name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise RouteCalibrationError(f"{name} must be a list")
    ordered = tuple(sorted(_cid(value, name) for value in values))
    if len(ordered) > MAX_CID_LIST:
        raise RouteCalibrationError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise RouteCalibrationError(f"{name} must not contain duplicates")
    return ordered


def _unique_sorted_tokens(
    values: Iterable[Any],
    name: str,
    *,
    max_items: int = MAX_REASON_CODES,
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise RouteCalibrationError(f"{name} must be a list")
    tokens = tuple(sorted({_token(value, name) for value in values}))
    if len(tokens) > max_items:
        raise RouteCalibrationError(f"{name} exceeds maximum length")
    return tokens


def _add_counter(left: int, right: int, name: str) -> int:
    total = _nonneg_int(left, name) + _nonneg_int(right, name)
    if total > MAX_COUNTER:
        raise RouteCalibrationError(f"{name} overflow")
    return total


def _rate_bp(successes: int, trials: int) -> int:
    rate = ratio_to_basis_points(successes, trials)
    return 0 if rate is None else rate


def _reject_provider_identity(value: Any, *, path: str = "$") -> None:
    """Fail closed when durable payloads carry provider/vendor identity."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            if lowered in _PROVIDER_IDENTITY_FIELDS or any(
                marker in lowered
                for marker in ("provider", "vendor", "openai", "anthropic", "ollama")
            ):
                raise RouteCalibrationError(
                    f"{path}.{key_text} must not contain provider identity",
                    reason_code="provider_identity_forbidden",
                )
            _reject_provider_identity(item, path=f"{path}.{key_text}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_provider_identity(item, path=f"{path}[{index}]")
        return
    if type(value) is str:
        lowered = value.lower()
        for banned in (
            "openai",
            "anthropic",
            "gemini",
            "ollama",
            "huggingface",
            "provider_id",
            "vendor_id",
        ):
            if banned in lowered:
                raise RouteCalibrationError(
                    f"{path} must not contain provider identity token {banned!r}",
                    reason_code="provider_identity_forbidden",
                )


def _reject_context_sufficiency_mutation(value: Any, *, path: str = "$") -> None:
    """Route calibration must not claim authority over context sufficiency."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in _CONTEXT_SUFFICIENCY_AUTHORITY_FIELDS:
                raise RouteCalibrationError(
                    f"{path}.{key_text} cannot be mutated by route calibration; "
                    "context sufficiency is calibrated separately",
                    reason_code="context_sufficiency_mutation_forbidden",
                )
            _reject_context_sufficiency_mutation(item, path=f"{path}.{key_text}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_context_sufficiency_mutation(item, path=f"{path}[{index}]")


def _stable_cid(label: str) -> str:
    return cid_for_structured({"route_calibration_seed": label})


def _header(
    artifact_kind: str,
    *,
    input_cids: Sequence[str] = (),
    interface_id: str = UPDATE_MODEL_ROUTE_CALIBRATION_INTERFACE,
    execution_mode: ExecutionMode | str = ExecutionMode.LIVE,
    terminal_status: GovernorTerminalStatus | str = GovernorTerminalStatus.COMPLETE,
    metadata: Mapping[str, Any] | None = None,
    repository_state_cid: str | None = None,
    context_pack_cid: str | None = None,
    verification_bundle_cid: str | None = None,
) -> GovernorArtifactHeader:
    try:
        return GovernorArtifactHeader(
            artifact_kind=artifact_kind,
            repository_state_cid=repository_state_cid
            or _stable_cid("repository_state"),
            context_pack_cid=context_pack_cid or _stable_cid("context_pack"),
            verification_bundle_cid=verification_bundle_cid
            or _stable_cid("verification_bundle"),
            generator=GeneratorIdentity(
                generator_id=GENERATOR_ID,
                generator_version=GENERATOR_VERSION,
                interface_id=interface_id,
            ),
            provenance=ArtifactProvenance(
                producer_id=PRODUCER_ID,
                producer_version=PRODUCER_VERSION,
                execution_mode=execution_mode,
                authority_source=AuthoritySource.DETERMINISTIC,
                input_cids=tuple(sorted(set(input_cids))),
                tool_ids=(TOOL_ID,),
                policy_cid=None,
                notes=None,
            ),
            terminal_status=terminal_status,
            assumptions=(
                GovernorAssumption(
                    assumption_id="route_calibration_proposal_only",
                    kind=AssumptionKind.ROUTE,
                    statement=(
                        "Route calibration emits proposals only and never "
                        "mutates production routes or context sufficiency"
                    ),
                    supporting_cids=(),
                ),
            ),
            metadata=dict(metadata or {"track": "route-calibration"}),
        )
    except SemanticGovernorBaseError as exc:
        raise RouteCalibrationError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Route tier metrics (per capability tier)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RouteTierMetrics:
    """Per-tier empirical counters for model-route calibration.

    ``context_omission_failure_count`` and ``reasoning_failure_count`` are
    intentionally independent; neither is derived from the other.
    """

    route_tier: RouteTier | str
    total_uses: int = 0
    accepted_count: int = 0
    retry_count: int = 0
    expansion_count: int = 0
    verification_pass_count: int = 0
    verification_fail_count: int = 0
    context_omission_failure_count: int = 0
    reasoning_failure_count: int = 0
    unavailable_required_tier_count: int = 0
    cost_micros_total: int = 0
    latency_ms_total: int = 0
    source_receipt_cids: Sequence[str] = ()

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "route_tier",
            "total_uses",
            "accepted_count",
            "retry_count",
            "expansion_count",
            "verification_pass_count",
            "verification_fail_count",
            "context_omission_failure_count",
            "reasoning_failure_count",
            "unavailable_required_tier_count",
            "cost_micros_total",
            "latency_ms_total",
            "source_receipt_cids",
            "accepted_rate_bp",
            "retry_rate_bp",
            "expansion_rate_bp",
            "verification_pass_rate_bp",
            "context_omission_rate_bp",
            "reasoning_failure_rate_bp",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "route_tier", _route_tier(self.route_tier, "route_tier")
        )
        for name in (
            "total_uses",
            "accepted_count",
            "retry_count",
            "expansion_count",
            "verification_pass_count",
            "verification_fail_count",
            "context_omission_failure_count",
            "reasoning_failure_count",
            "unavailable_required_tier_count",
            "cost_micros_total",
            "latency_ms_total",
        ):
            object.__setattr__(self, name, _nonneg_int(getattr(self, name), name))
        if self.accepted_count > self.total_uses:
            raise RouteCalibrationError("accepted_count must not exceed total_uses")
        if self.context_omission_failure_count > self.total_uses:
            raise RouteCalibrationError(
                "context_omission_failure_count must not exceed total_uses"
            )
        if self.reasoning_failure_count > self.total_uses:
            raise RouteCalibrationError(
                "reasoning_failure_count must not exceed total_uses"
            )
        object.__setattr__(
            self,
            "source_receipt_cids",
            _unique_sorted_cids(
                list(self.source_receipt_cids), "source_receipt_cids"
            ),
        )

    @property
    def accepted_rate_bp(self) -> int:
        return _rate_bp(self.accepted_count, self.total_uses)

    @property
    def retry_rate_bp(self) -> int:
        return _rate_bp(self.retry_count, self.total_uses)

    @property
    def expansion_rate_bp(self) -> int:
        return _rate_bp(self.expansion_count, self.total_uses)

    @property
    def verification_pass_rate_bp(self) -> int:
        trials = self.verification_pass_count + self.verification_fail_count
        return _rate_bp(self.verification_pass_count, trials)

    @property
    def context_omission_rate_bp(self) -> int:
        return _rate_bp(self.context_omission_failure_count, self.total_uses)

    @property
    def reasoning_failure_rate_bp(self) -> int:
        return _rate_bp(self.reasoning_failure_count, self.total_uses)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": ROUTE_TIER_METRICS_SCHEMA,
            "route_tier": self.route_tier,
            "total_uses": self.total_uses,
            "accepted_count": self.accepted_count,
            "retry_count": self.retry_count,
            "expansion_count": self.expansion_count,
            "verification_pass_count": self.verification_pass_count,
            "verification_fail_count": self.verification_fail_count,
            "context_omission_failure_count": self.context_omission_failure_count,
            "reasoning_failure_count": self.reasoning_failure_count,
            "unavailable_required_tier_count": self.unavailable_required_tier_count,
            "cost_micros_total": self.cost_micros_total,
            "latency_ms_total": self.latency_ms_total,
            "source_receipt_cids": list(self.source_receipt_cids),
            "accepted_rate_bp": self.accepted_rate_bp,
            "retry_rate_bp": self.retry_rate_bp,
            "expansion_rate_bp": self.expansion_rate_bp,
            "verification_pass_rate_bp": self.verification_pass_rate_bp,
            "context_omission_rate_bp": self.context_omission_rate_bp,
            "reasoning_failure_rate_bp": self.reasoning_failure_rate_bp,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.identity_payload()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RouteTierMetrics":
        payload = dict(data)
        # Derived rate fields are optional on input; identity re-computes them.
        for derived in (
            "accepted_rate_bp",
            "retry_rate_bp",
            "expansion_rate_bp",
            "verification_pass_rate_bp",
            "context_omission_rate_bp",
            "reasoning_failure_rate_bp",
        ):
            payload.pop(derived, None)
        schema = payload.pop("schema", ROUTE_TIER_METRICS_SCHEMA)
        if schema != ROUTE_TIER_METRICS_SCHEMA:
            raise RouteCalibrationError("unsupported RouteTierMetrics schema version")
        return cls(
            route_tier=payload.get("route_tier", RouteTier.MEDIUM.value),
            total_uses=payload.get("total_uses", 0),
            accepted_count=payload.get("accepted_count", 0),
            retry_count=payload.get("retry_count", 0),
            expansion_count=payload.get("expansion_count", 0),
            verification_pass_count=payload.get("verification_pass_count", 0),
            verification_fail_count=payload.get("verification_fail_count", 0),
            context_omission_failure_count=payload.get(
                "context_omission_failure_count", 0
            ),
            reasoning_failure_count=payload.get("reasoning_failure_count", 0),
            unavailable_required_tier_count=payload.get(
                "unavailable_required_tier_count", 0
            ),
            cost_micros_total=payload.get("cost_micros_total", 0),
            latency_ms_total=payload.get("latency_ms_total", 0),
            source_receipt_cids=tuple(payload.get("source_receipt_cids") or ()),
        )

    @classmethod
    def empty(cls, route_tier: RouteTier | str) -> "RouteTierMetrics":
        return cls(route_tier=route_tier)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RouteRunObservation:
    """One sealed observation derived from a run receipt / route decision.

    Failure flags are independent: omission and reasoning can be recorded
    separately and are never collapsed into a single generic counter.
    """

    observation_id: str
    route_tier: RouteTier | str
    accepted: bool
    retried: bool = False
    expansion_used: bool = False
    verification_passed: bool | None = None
    context_omission_failure: bool = False
    reasoning_failure: bool = False
    required_route_tier: RouteTier | str | None = None
    required_tier_available: bool = True
    cost_micros: int = 0
    latency_ms: int = 0
    receipt_cid: str | None = None
    decision_cid: str | None = None
    failure_kind: RouteFailureKind | str = RouteFailureKind.NONE
    reason_codes: Sequence[str] = ()
    simulated: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "observation_id",
            "route_tier",
            "accepted",
            "retried",
            "expansion_used",
            "verification_passed",
            "context_omission_failure",
            "reasoning_failure",
            "required_route_tier",
            "required_tier_available",
            "cost_micros",
            "latency_ms",
            "receipt_cid",
            "decision_cid",
            "failure_kind",
            "reason_codes",
            "simulated",
            "metadata",
            "observation_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observation_id", _token(self.observation_id, "observation_id")
        )
        object.__setattr__(
            self, "route_tier", _route_tier(self.route_tier, "route_tier")
        )
        object.__setattr__(self, "accepted", _bool(self.accepted, "accepted"))
        object.__setattr__(self, "retried", _bool(self.retried, "retried"))
        object.__setattr__(
            self, "expansion_used", _bool(self.expansion_used, "expansion_used")
        )
        if self.verification_passed is not None:
            object.__setattr__(
                self,
                "verification_passed",
                _bool(self.verification_passed, "verification_passed"),
            )
        object.__setattr__(
            self,
            "context_omission_failure",
            _bool(self.context_omission_failure, "context_omission_failure"),
        )
        object.__setattr__(
            self,
            "reasoning_failure",
            _bool(self.reasoning_failure, "reasoning_failure"),
        )
        if self.required_route_tier is None:
            object.__setattr__(self, "required_route_tier", self.route_tier)
        else:
            object.__setattr__(
                self,
                "required_route_tier",
                _route_tier(self.required_route_tier, "required_route_tier"),
            )
        object.__setattr__(
            self,
            "required_tier_available",
            _bool(self.required_tier_available, "required_tier_available"),
        )
        object.__setattr__(
            self, "cost_micros", _nonneg_int(self.cost_micros, "cost_micros")
        )
        object.__setattr__(
            self, "latency_ms", _nonneg_int(self.latency_ms, "latency_ms")
        )
        object.__setattr__(
            self, "receipt_cid", _optional_cid(self.receipt_cid, "receipt_cid")
        )
        object.__setattr__(
            self, "decision_cid", _optional_cid(self.decision_cid, "decision_cid")
        )
        kind = _enum(self.failure_kind, RouteFailureKind, "failure_kind")
        # Normalize primary kind from explicit flags when caller left NONE.
        if kind == RouteFailureKind.NONE.value:
            if self.context_omission_failure and not self.reasoning_failure:
                kind = RouteFailureKind.CONTEXT_OMISSION.value
            elif self.reasoning_failure and not self.context_omission_failure:
                kind = RouteFailureKind.REASONING_FAILURE.value
            elif not self.required_tier_available:
                kind = RouteFailureKind.UNAVAILABLE_TIER.value
            elif self.verification_passed is False:
                kind = RouteFailureKind.VERIFICATION_FAILURE.value
        object.__setattr__(self, "failure_kind", kind)
        object.__setattr__(
            self,
            "reason_codes",
            _unique_sorted_tokens(list(self.reason_codes), "reason_codes"),
        )
        object.__setattr__(self, "simulated", _bool(self.simulated, "simulated"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        _reject_provider_identity(self.identity_payload())
        _reject_context_sufficiency_mutation(self.identity_payload())

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": ROUTE_RUN_OBSERVATION_SCHEMA,
            "observation_id": self.observation_id,
            "route_tier": self.route_tier,
            "accepted": self.accepted,
            "retried": self.retried,
            "expansion_used": self.expansion_used,
            "verification_passed": self.verification_passed,
            "context_omission_failure": self.context_omission_failure,
            "reasoning_failure": self.reasoning_failure,
            "required_route_tier": self.required_route_tier,
            "required_tier_available": self.required_tier_available,
            "cost_micros": self.cost_micros,
            "latency_ms": self.latency_ms,
            "receipt_cid": self.receipt_cid,
            "decision_cid": self.decision_cid,
            "failure_kind": self.failure_kind,
            "reason_codes": list(self.reason_codes),
            "simulated": self.simulated,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def observation_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "observation_cid": self.observation_cid}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RouteRunObservation":
        payload = dict(data)
        claimed = payload.pop("observation_cid", None)
        schema = payload.pop("schema", ROUTE_RUN_OBSERVATION_SCHEMA)
        if schema != ROUTE_RUN_OBSERVATION_SCHEMA:
            raise RouteCalibrationError(
                "unsupported RouteRunObservation schema version"
            )
        result = cls(
            observation_id=payload.get("observation_id", "obs"),
            route_tier=payload.get("route_tier", RouteTier.MEDIUM.value),
            accepted=payload.get("accepted", False),
            retried=payload.get("retried", False),
            expansion_used=payload.get("expansion_used", False),
            verification_passed=payload.get("verification_passed"),
            context_omission_failure=payload.get("context_omission_failure", False),
            reasoning_failure=payload.get("reasoning_failure", False),
            required_route_tier=payload.get("required_route_tier"),
            required_tier_available=payload.get("required_tier_available", True),
            cost_micros=payload.get("cost_micros", 0),
            latency_ms=payload.get("latency_ms", 0),
            receipt_cid=payload.get("receipt_cid"),
            decision_cid=payload.get("decision_cid"),
            failure_kind=payload.get("failure_kind", RouteFailureKind.NONE.value),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            simulated=payload.get("simulated", False),
            metadata=payload.get("metadata") or {},
        )
        if claimed is not None and claimed != result.observation_cid:
            raise RouteCalibrationError(
                "RouteRunObservation observation_cid does not verify"
            )
        return result

    @classmethod
    def from_value(cls, value: "RouteRunObservation | Mapping[str, Any]") -> "RouteRunObservation":
        if isinstance(value, RouteRunObservation):
            return value
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        raise RouteCalibrationError(
            "observation must be RouteRunObservation or mapping"
        )


# ---------------------------------------------------------------------------
# Calibration state
# ---------------------------------------------------------------------------


def _empty_tier_map() -> dict[str, RouteTierMetrics]:
    return {tier: RouteTierMetrics.empty(tier) for tier in _ROUTE_TIERS}


@dataclass(frozen=True, slots=True)
class ModelRouteCalibrationState:
    """Durable multi-tier route calibration state (proposal authority only)."""

    header: GovernorArtifactHeader
    state_id: str
    partition: EvidencePartition | str
    revision: int
    tier_metrics: Mapping[str, RouteTierMetrics | Mapping[str, Any]]
    applied_observation_cids: Sequence[str] = ()
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface_id",
            "header",
            "state_id",
            "partition",
            "revision",
            "tier_metrics",
            "applied_observation_cids",
            "notes",
            "metadata",
            "state_cid",
        }
    )

    def __post_init__(self) -> None:
        if not isinstance(self.header, GovernorArtifactHeader):
            if isinstance(self.header, Mapping):
                object.__setattr__(
                    self, "header", GovernorArtifactHeader.from_dict(self.header)
                )
            else:
                raise RouteCalibrationError("header must be GovernorArtifactHeader")
        if self.header.artifact_kind != "model_route_calibration_state":
            raise RouteCalibrationError(
                "header.artifact_kind must be model_route_calibration_state"
            )
        object.__setattr__(self, "state_id", _token(self.state_id, "state_id"))
        object.__setattr__(
            self, "partition", _enum(self.partition, EvidencePartition, "partition")
        )
        object.__setattr__(self, "revision", _nonneg_int(self.revision, "revision"))
        if not isinstance(self.tier_metrics, Mapping):
            raise RouteCalibrationError("tier_metrics must be a mapping")
        normalized: dict[str, RouteTierMetrics] = {}
        for tier in _ROUTE_TIERS:
            raw = self.tier_metrics.get(tier)
            if raw is None:
                metrics = RouteTierMetrics.empty(tier)
            elif isinstance(raw, RouteTierMetrics):
                metrics = raw
            elif isinstance(raw, Mapping):
                metrics = RouteTierMetrics.from_dict(raw)
            else:
                raise RouteCalibrationError(
                    f"tier_metrics[{tier}] must be RouteTierMetrics or mapping"
                )
            if metrics.route_tier != tier:
                raise RouteCalibrationError(
                    f"tier_metrics key {tier!r} does not match metrics.route_tier"
                )
            normalized[tier] = metrics
        # Reject unknown tier keys (closed vocabulary).
        unknown = set(self.tier_metrics) - set(_ROUTE_TIERS)
        if unknown:
            raise RouteCalibrationError(
                f"tier_metrics contains unsupported tiers {sorted(unknown)}"
            )
        object.__setattr__(self, "tier_metrics", MappingProxyType(normalized))
        object.__setattr__(
            self,
            "applied_observation_cids",
            _unique_sorted_cids(
                list(self.applied_observation_cids), "applied_observation_cids"
            ),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        _reject_provider_identity(self.identity_payload())
        _reject_context_sufficiency_mutation(self.identity_payload())

    def metrics_for(self, route_tier: RouteTier | str) -> RouteTierMetrics:
        tier = _route_tier(route_tier)
        return self.tier_metrics[tier]

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": ROUTE_CALIBRATION_STATE_SCHEMA,
            "interface_id": ROUTE_CALIBRATION_STATE_INTERFACE,
            "header": self.header.identity_payload(),
            "state_id": self.state_id,
            "partition": self.partition,
            "revision": self.revision,
            "tier_metrics": {
                tier: self.tier_metrics[tier].identity_payload()
                for tier in _ROUTE_TIERS
            },
            "applied_observation_cids": list(self.applied_observation_cids),
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def state_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ROUTE_CALIBRATION_STATE_SCHEMA,
            "interface_id": ROUTE_CALIBRATION_STATE_INTERFACE,
            "header": self.header.to_dict(),
            "state_id": self.state_id,
            "partition": self.partition,
            "revision": self.revision,
            "tier_metrics": {
                tier: self.tier_metrics[tier].to_dict() for tier in _ROUTE_TIERS
            },
            "applied_observation_cids": list(self.applied_observation_cids),
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
            "state_cid": self.state_cid,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelRouteCalibrationState":
        payload = dict(data)
        claimed = payload.pop("state_cid", None)
        schema = payload.pop("schema", ROUTE_CALIBRATION_STATE_SCHEMA)
        interface_id = payload.pop("interface_id", ROUTE_CALIBRATION_STATE_INTERFACE)
        if schema != ROUTE_CALIBRATION_STATE_SCHEMA:
            raise RouteCalibrationError(
                "unsupported ModelRouteCalibrationState schema version"
            )
        if interface_id != ROUTE_CALIBRATION_STATE_INTERFACE:
            raise RouteCalibrationError(
                "unsupported ModelRouteCalibrationState interface_id"
            )
        result = cls(
            header=payload["header"],
            state_id=payload["state_id"],
            partition=payload["partition"],
            revision=payload["revision"],
            tier_metrics=payload.get("tier_metrics") or {},
            applied_observation_cids=tuple(
                payload.get("applied_observation_cids") or ()
            ),
            notes=payload.get("notes"),
            metadata=payload.get("metadata") or {},
        )
        if claimed is not None and claimed != result.state_cid:
            raise RouteCalibrationError(
                "ModelRouteCalibrationState state_cid does not verify"
            )
        return result

    @classmethod
    def empty(
        cls,
        *,
        state_id: str = "route_calibration_default",
        partition: EvidencePartition | str = EvidencePartition.CALIBRATION,
        notes: str | None = None,
    ) -> "ModelRouteCalibrationState":
        return cls(
            header=_header("model_route_calibration_state"),
            state_id=state_id,
            partition=partition,
            revision=0,
            tier_metrics=_empty_tier_map(),
            applied_observation_cids=(),
            notes=notes,
            metadata={"track": "route-calibration"},
        )


# ---------------------------------------------------------------------------
# Threshold policy + proposals
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RouteThresholdPolicy:
    """Current route-threshold floors used only as proposal baselines."""

    policy_id: str
    min_accepted_rate_bp: int = DEFAULT_MIN_ACCEPTED_RATE_BP
    max_context_omission_rate_bp: int = DEFAULT_MAX_OMISSION_RATE_BP
    max_reasoning_failure_rate_bp: int = DEFAULT_MAX_REASONING_FAILURE_RATE_BP
    max_retry_rate_bp: int = DEFAULT_MAX_RETRY_RATE_BP
    min_route_tier: RouteTier | str = RouteTier.DETERMINISTIC
    escalate_on_unavailable: bool = True
    high_risk_min_route_tier: RouteTier | str = RouteTier.FRONTIER
    shadow_sample_rate_bp: int = 1_000
    min_uses_for_proposal: int = DEFAULT_MIN_USES_FOR_PROPOSAL
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _token(self.policy_id, "policy_id"))
        object.__setattr__(
            self,
            "min_accepted_rate_bp",
            _basis_points(self.min_accepted_rate_bp, "min_accepted_rate_bp"),
        )
        object.__setattr__(
            self,
            "max_context_omission_rate_bp",
            _basis_points(
                self.max_context_omission_rate_bp, "max_context_omission_rate_bp"
            ),
        )
        object.__setattr__(
            self,
            "max_reasoning_failure_rate_bp",
            _basis_points(
                self.max_reasoning_failure_rate_bp, "max_reasoning_failure_rate_bp"
            ),
        )
        object.__setattr__(
            self,
            "max_retry_rate_bp",
            _basis_points(self.max_retry_rate_bp, "max_retry_rate_bp"),
        )
        object.__setattr__(
            self, "min_route_tier", _route_tier(self.min_route_tier, "min_route_tier")
        )
        object.__setattr__(
            self,
            "escalate_on_unavailable",
            _bool(self.escalate_on_unavailable, "escalate_on_unavailable"),
        )
        object.__setattr__(
            self,
            "high_risk_min_route_tier",
            _route_tier(self.high_risk_min_route_tier, "high_risk_min_route_tier"),
        )
        object.__setattr__(
            self,
            "shadow_sample_rate_bp",
            _basis_points(self.shadow_sample_rate_bp, "shadow_sample_rate_bp"),
        )
        object.__setattr__(
            self,
            "min_uses_for_proposal",
            _nonneg_int(self.min_uses_for_proposal, "min_uses_for_proposal"),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        _reject_provider_identity(self.identity_payload())

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": ROUTE_THRESHOLD_POLICY_SCHEMA,
            "policy_id": self.policy_id,
            "min_accepted_rate_bp": self.min_accepted_rate_bp,
            "max_context_omission_rate_bp": self.max_context_omission_rate_bp,
            "max_reasoning_failure_rate_bp": self.max_reasoning_failure_rate_bp,
            "max_retry_rate_bp": self.max_retry_rate_bp,
            "min_route_tier": self.min_route_tier,
            "escalate_on_unavailable": self.escalate_on_unavailable,
            "high_risk_min_route_tier": self.high_risk_min_route_tier,
            "shadow_sample_rate_bp": self.shadow_sample_rate_bp,
            "min_uses_for_proposal": self.min_uses_for_proposal,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def policy_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "policy_cid": self.policy_cid}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RouteThresholdPolicy":
        payload = dict(data)
        payload.pop("policy_cid", None)
        schema = payload.pop("schema", ROUTE_THRESHOLD_POLICY_SCHEMA)
        if schema != ROUTE_THRESHOLD_POLICY_SCHEMA:
            raise RouteCalibrationError(
                "unsupported RouteThresholdPolicy schema version"
            )
        return cls(
            policy_id=payload.get("policy_id", "route_threshold_default"),
            min_accepted_rate_bp=payload.get(
                "min_accepted_rate_bp", DEFAULT_MIN_ACCEPTED_RATE_BP
            ),
            max_context_omission_rate_bp=payload.get(
                "max_context_omission_rate_bp", DEFAULT_MAX_OMISSION_RATE_BP
            ),
            max_reasoning_failure_rate_bp=payload.get(
                "max_reasoning_failure_rate_bp", DEFAULT_MAX_REASONING_FAILURE_RATE_BP
            ),
            max_retry_rate_bp=payload.get(
                "max_retry_rate_bp", DEFAULT_MAX_RETRY_RATE_BP
            ),
            min_route_tier=payload.get("min_route_tier", RouteTier.DETERMINISTIC.value),
            escalate_on_unavailable=payload.get("escalate_on_unavailable", True),
            high_risk_min_route_tier=payload.get(
                "high_risk_min_route_tier", RouteTier.FRONTIER.value
            ),
            shadow_sample_rate_bp=payload.get("shadow_sample_rate_bp", 1_000),
            min_uses_for_proposal=payload.get(
                "min_uses_for_proposal", DEFAULT_MIN_USES_FOR_PROPOSAL
            ),
            metadata=payload.get("metadata") or {},
        )

    @classmethod
    def from_value(
        cls, value: "RouteThresholdPolicy | Mapping[str, Any] | None"
    ) -> "RouteThresholdPolicy":
        if value is None:
            return default_route_threshold_policy()
        if isinstance(value, RouteThresholdPolicy):
            return value
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        raise RouteCalibrationError(
            "policy must be RouteThresholdPolicy, mapping, or None"
        )


def default_route_threshold_policy() -> RouteThresholdPolicy:
    return RouteThresholdPolicy(policy_id="route_threshold_default")


@dataclass(frozen=True, slots=True)
class RouteThresholdProposal:
    """One sealed route-threshold change proposal (never auto-applied)."""

    proposal_id: str
    route_tier: RouteTier | str
    parameter: RouteThresholdParameter | str
    current_value: str
    proposed_value: str
    reason_codes: Sequence[str]
    supporting_observation_cids: Sequence[str] = ()
    high_risk_assurance_reduced: bool = False
    mutates_production: bool = False
    is_proposal_only: bool = True
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "proposal_id", _token(self.proposal_id, "proposal_id")
        )
        object.__setattr__(
            self, "route_tier", _route_tier(self.route_tier, "route_tier")
        )
        object.__setattr__(
            self,
            "parameter",
            _enum(self.parameter, RouteThresholdParameter, "parameter"),
        )
        object.__setattr__(
            self, "current_value", _text(self.current_value, "current_value")
        )
        object.__setattr__(
            self, "proposed_value", _text(self.proposed_value, "proposed_value")
        )
        reasons = _unique_sorted_tokens(list(self.reason_codes), "reason_codes")
        if not reasons:
            raise RouteCalibrationError("reason_codes must not be empty")
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(
            self,
            "supporting_observation_cids",
            _unique_sorted_cids(
                list(self.supporting_observation_cids), "supporting_observation_cids"
            ),
        )
        object.__setattr__(
            self,
            "high_risk_assurance_reduced",
            _bool(self.high_risk_assurance_reduced, "high_risk_assurance_reduced"),
        )
        object.__setattr__(
            self,
            "mutates_production",
            _bool(self.mutates_production, "mutates_production"),
        )
        object.__setattr__(
            self, "is_proposal_only", _bool(self.is_proposal_only, "is_proposal_only")
        )
        # Hard invariants: changes are proposals only.
        if self.mutates_production:
            raise RouteCalibrationError(
                "route threshold proposals must not mutate production",
                reason_code="production_mutation_forbidden",
            )
        if not self.is_proposal_only:
            raise RouteCalibrationError(
                "route threshold changes are proposals only",
                reason_code="proposal_only_required",
            )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        _reject_provider_identity(self.identity_payload())

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": ROUTE_THRESHOLD_PROPOSAL_SCHEMA,
            "proposal_id": self.proposal_id,
            "route_tier": self.route_tier,
            "parameter": self.parameter,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
            "reason_codes": list(self.reason_codes),
            "supporting_observation_cids": list(self.supporting_observation_cids),
            "high_risk_assurance_reduced": self.high_risk_assurance_reduced,
            "mutates_production": self.mutates_production,
            "is_proposal_only": self.is_proposal_only,
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def proposal_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "proposal_cid": self.proposal_cid}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RouteThresholdProposal":
        payload = dict(data)
        claimed = payload.pop("proposal_cid", None)
        schema = payload.pop("schema", ROUTE_THRESHOLD_PROPOSAL_SCHEMA)
        if schema != ROUTE_THRESHOLD_PROPOSAL_SCHEMA:
            raise RouteCalibrationError(
                "unsupported RouteThresholdProposal schema version"
            )
        result = cls(
            proposal_id=payload["proposal_id"],
            route_tier=payload["route_tier"],
            parameter=payload["parameter"],
            current_value=payload["current_value"],
            proposed_value=payload["proposed_value"],
            reason_codes=tuple(payload.get("reason_codes") or ()),
            supporting_observation_cids=tuple(
                payload.get("supporting_observation_cids") or ()
            ),
            high_risk_assurance_reduced=payload.get(
                "high_risk_assurance_reduced", False
            ),
            mutates_production=payload.get("mutates_production", False),
            is_proposal_only=payload.get("is_proposal_only", True),
            notes=payload.get("notes"),
            metadata=payload.get("metadata") or {},
        )
        if claimed is not None and claimed != result.proposal_cid:
            raise RouteCalibrationError(
                "RouteThresholdProposal proposal_cid does not verify"
            )
        return result


@dataclass(frozen=True, slots=True)
class RouteCalibrationUpdateResult:
    """Sealed outcome of update_model_route_calibration."""

    disposition: RouteCalibrationDisposition | str
    state: ModelRouteCalibrationState
    applied_observation_cids: Sequence[str] = ()
    skipped_observation_cids: Sequence[str] = ()
    reason_codes: Sequence[str] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, RouteCalibrationDisposition, "disposition"),
        )
        if not isinstance(self.state, ModelRouteCalibrationState):
            if isinstance(self.state, Mapping):
                object.__setattr__(
                    self, "state", ModelRouteCalibrationState.from_dict(self.state)
                )
            else:
                raise RouteCalibrationError(
                    "state must be ModelRouteCalibrationState"
                )
        object.__setattr__(
            self,
            "applied_observation_cids",
            _unique_sorted_cids(
                list(self.applied_observation_cids), "applied_observation_cids"
            ),
        )
        object.__setattr__(
            self,
            "skipped_observation_cids",
            _unique_sorted_cids(
                list(self.skipped_observation_cids), "skipped_observation_cids"
            ),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _unique_sorted_tokens(list(self.reason_codes), "reason_codes"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": ROUTE_CALIBRATION_UPDATE_RESULT_SCHEMA,
            "interface_id": UPDATE_MODEL_ROUTE_CALIBRATION_INTERFACE,
            "disposition": self.disposition,
            "state_cid": self.state.state_cid,
            "applied_observation_cids": list(self.applied_observation_cids),
            "skipped_observation_cids": list(self.skipped_observation_cids),
            "reason_codes": list(self.reason_codes),
            "notes": self.notes,
        }

    @property
    def result_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "state": self.state.to_dict(),
            "result_cid": self.result_cid,
        }


@dataclass(frozen=True, slots=True)
class RouteThresholdProposalResult:
    """Sealed outcome of propose_route_threshold_change (proposals only)."""

    disposition: RouteThresholdDisposition | str
    proposals: Sequence[RouteThresholdProposal | Mapping[str, Any]]
    policy_cid: str
    state_cid: str
    reason_codes: Sequence[str] = ()
    mutates_production: bool = False
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, RouteThresholdDisposition, "disposition"),
        )
        if not isinstance(self.proposals, (list, tuple)):
            raise RouteCalibrationError("proposals must be a sequence")
        if len(self.proposals) > MAX_PROPOSALS:
            raise RouteCalibrationError("proposals exceeds maximum length")
        normalized: list[RouteThresholdProposal] = []
        for item in self.proposals:
            if isinstance(item, RouteThresholdProposal):
                normalized.append(item)
            elif isinstance(item, Mapping):
                normalized.append(RouteThresholdProposal.from_dict(item))
            else:
                raise RouteCalibrationError(
                    "proposals items must be RouteThresholdProposal or mapping"
                )
        object.__setattr__(self, "proposals", tuple(normalized))
        object.__setattr__(self, "policy_cid", _cid(self.policy_cid, "policy_cid"))
        object.__setattr__(self, "state_cid", _cid(self.state_cid, "state_cid"))
        object.__setattr__(
            self,
            "reason_codes",
            _unique_sorted_tokens(list(self.reason_codes), "reason_codes"),
        )
        mutates = _bool(self.mutates_production, "mutates_production")
        if mutates:
            raise RouteCalibrationError(
                "propose_route_threshold_change must not mutate production",
                reason_code="production_mutation_forbidden",
            )
        object.__setattr__(self, "mutates_production", mutates)
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": ROUTE_THRESHOLD_PROPOSAL_RESULT_SCHEMA,
            "interface_id": PROPOSE_ROUTE_THRESHOLD_CHANGE_INTERFACE,
            "disposition": self.disposition,
            "proposals": [item.identity_payload() for item in self.proposals],
            "policy_cid": self.policy_cid,
            "state_cid": self.state_cid,
            "reason_codes": list(self.reason_codes),
            "mutates_production": self.mutates_production,
            "notes": self.notes,
        }

    @property
    def result_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "proposals": [item.to_dict() for item in self.proposals],
            "result_cid": self.result_cid,
        }


@dataclass(frozen=True, slots=True)
class RouteAvailabilityDecision:
    """Resolution of a required capability tier against availability."""

    required_route_tier: RouteTier | str
    resolved_route_tier: RouteTier | str
    disposition: RouteAvailabilityDisposition | str
    available_route_tiers: Sequence[str]
    reason_codes: Sequence[str]
    downgraded: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_route_tier",
            _route_tier(self.required_route_tier, "required_route_tier"),
        )
        object.__setattr__(
            self,
            "resolved_route_tier",
            _route_tier(self.resolved_route_tier, "resolved_route_tier"),
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, RouteAvailabilityDisposition, "disposition"),
        )
        available = _unique_sorted_tokens(
            list(self.available_route_tiers),
            "available_route_tiers",
            max_items=len(_ROUTE_TIERS) + 4,
        )
        for tier in available:
            _route_tier(tier, "available_route_tiers")
        object.__setattr__(self, "available_route_tiers", available)
        reasons = _unique_sorted_tokens(list(self.reason_codes), "reason_codes")
        if not reasons:
            raise RouteCalibrationError("reason_codes must not be empty")
        object.__setattr__(self, "reason_codes", reasons)
        downgraded = _bool(self.downgraded, "downgraded")
        # Hard invariant: never report a capability downgrade as success.
        if downgraded:
            raise RouteCalibrationError(
                "unavailable required tier must never downgrade",
                reason_code="route_downgrade_forbidden",
            )
        if (
            _ROUTE_RANK[self.resolved_route_tier]
            < _ROUTE_RANK[self.required_route_tier]
        ):
            raise RouteCalibrationError(
                "resolved route tier is weaker than required tier",
                reason_code="route_downgrade_forbidden",
            )
        object.__setattr__(self, "downgraded", downgraded)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": ROUTE_AVAILABILITY_DECISION_SCHEMA,
            "required_route_tier": self.required_route_tier,
            "resolved_route_tier": self.resolved_route_tier,
            "disposition": self.disposition,
            "available_route_tiers": list(self.available_route_tiers),
            "reason_codes": list(self.reason_codes),
            "downgraded": self.downgraded,
        }

    @property
    def decision_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "decision_cid": self.decision_cid}


# ---------------------------------------------------------------------------
# Availability: never downgrade
# ---------------------------------------------------------------------------


def resolve_route_availability(
    required_route_tier: RouteTier | str,
    available_route_tiers: Sequence[str] | None = None,
) -> RouteAvailabilityDecision:
    """Resolve a required capability tier without ever downgrading.

    * Deterministic work is always available (no inventory required).
    * Human is always available as the no-downgrade escalation sink.
    * When the required model tier is missing from inventory, escalate to
      ``human`` — never select a weaker model tier.
    """

    required = _route_tier(required_route_tier, "required_route_tier")
    inventory_raw = list(available_route_tiers or ())
    # Normalize and validate inventory tiers.
    inventory: list[str] = []
    for item in inventory_raw:
        tier = _route_tier(item, "available_route_tiers")
        if tier not in inventory:
            inventory.append(tier)
    # Human is always an available escalation sink for policy purposes.
    if RouteTier.HUMAN.value not in inventory:
        inventory.append(RouteTier.HUMAN.value)

    if required == RouteTier.DETERMINISTIC.value:
        return RouteAvailabilityDecision(
            required_route_tier=required,
            resolved_route_tier=RouteTier.DETERMINISTIC.value,
            disposition=RouteAvailabilityDisposition.DETERMINISTIC_ALWAYS_AVAILABLE.value,
            available_route_tiers=tuple(inventory),
            reason_codes=("deterministic_always_available",),
            downgraded=False,
        )

    if required == RouteTier.HUMAN.value:
        return RouteAvailabilityDecision(
            required_route_tier=required,
            resolved_route_tier=RouteTier.HUMAN.value,
            disposition=RouteAvailabilityDisposition.ALREADY_HUMAN.value,
            available_route_tiers=tuple(inventory),
            reason_codes=("human_review_required",),
            downgraded=False,
        )

    if required in inventory:
        return RouteAvailabilityDecision(
            required_route_tier=required,
            resolved_route_tier=required,
            disposition=RouteAvailabilityDisposition.AVAILABLE.value,
            available_route_tiers=tuple(inventory),
            reason_codes=("required_tier_available",),
            downgraded=False,
        )

    # Unavailable: escalate to human — never pick a weaker model tier.
    return RouteAvailabilityDecision(
        required_route_tier=required,
        resolved_route_tier=RouteTier.HUMAN.value,
        disposition=RouteAvailabilityDisposition.UNAVAILABLE_ESCALATED_TO_HUMAN.value,
        available_route_tiers=tuple(inventory),
        reason_codes=("required_tier_unavailable", "escalated_to_human"),
        downgraded=False,
    )


# ---------------------------------------------------------------------------
# Observation application
# ---------------------------------------------------------------------------


def _apply_observation_to_metrics(
    metrics: RouteTierMetrics,
    obs: RouteRunObservation,
) -> RouteTierMetrics:
    receipt_cids = list(metrics.source_receipt_cids)
    if obs.receipt_cid is not None:
        receipt_cids.append(obs.receipt_cid)
    if obs.decision_cid is not None:
        receipt_cids.append(obs.decision_cid)

    verification_pass = metrics.verification_pass_count
    verification_fail = metrics.verification_fail_count
    if obs.verification_passed is True:
        verification_pass = _add_counter(verification_pass, 1, "verification_pass_count")
    elif obs.verification_passed is False:
        verification_fail = _add_counter(verification_fail, 1, "verification_fail_count")

    return RouteTierMetrics(
        route_tier=metrics.route_tier,
        total_uses=_add_counter(metrics.total_uses, 1, "total_uses"),
        accepted_count=_add_counter(
            metrics.accepted_count, 1 if obs.accepted else 0, "accepted_count"
        ),
        retry_count=_add_counter(
            metrics.retry_count, 1 if obs.retried else 0, "retry_count"
        ),
        expansion_count=_add_counter(
            metrics.expansion_count,
            1 if obs.expansion_used else 0,
            "expansion_count",
        ),
        verification_pass_count=verification_pass,
        verification_fail_count=verification_fail,
        # Separate counters: never fold omission into reasoning or vice versa.
        context_omission_failure_count=_add_counter(
            metrics.context_omission_failure_count,
            1 if obs.context_omission_failure else 0,
            "context_omission_failure_count",
        ),
        reasoning_failure_count=_add_counter(
            metrics.reasoning_failure_count,
            1 if obs.reasoning_failure else 0,
            "reasoning_failure_count",
        ),
        unavailable_required_tier_count=_add_counter(
            metrics.unavailable_required_tier_count,
            1 if not obs.required_tier_available else 0,
            "unavailable_required_tier_count",
        ),
        cost_micros_total=_add_counter(
            metrics.cost_micros_total, obs.cost_micros, "cost_micros_total"
        ),
        latency_ms_total=_add_counter(
            metrics.latency_ms_total, obs.latency_ms, "latency_ms_total"
        ),
        source_receipt_cids=tuple(sorted(set(receipt_cids))),
    )


def _normalize_state(
    value: ModelRouteCalibrationState | Mapping[str, Any] | None,
) -> ModelRouteCalibrationState:
    if value is None:
        return ModelRouteCalibrationState.empty()
    if isinstance(value, ModelRouteCalibrationState):
        return value
    if isinstance(value, Mapping):
        return ModelRouteCalibrationState.from_dict(value)
    raise RouteCalibrationError(
        "state must be ModelRouteCalibrationState, mapping, or None"
    )


def _normalize_observations(
    observations: Sequence[RouteRunObservation | Mapping[str, Any]] | None,
) -> tuple[RouteRunObservation, ...]:
    if observations is None:
        return ()
    if not isinstance(observations, (list, tuple)):
        raise RouteCalibrationError("observations must be a sequence")
    if len(observations) > MAX_OBSERVATIONS:
        raise RouteCalibrationError("observations exceeds maximum length")
    return tuple(RouteRunObservation.from_value(item) for item in observations)


# ---------------------------------------------------------------------------
# Public: update_model_route_calibration
# ---------------------------------------------------------------------------


def update_model_route_calibration(
    state: ModelRouteCalibrationState | Mapping[str, Any] | None,
    observations: Sequence[RouteRunObservation | Mapping[str, Any]] | None,
    *,
    notes: str | None = None,
) -> RouteCalibrationUpdateResult:
    """Update per-tier route counters from sealed run observations.

    Simulated observations are excluded from live quality counters.
    Already-applied observation CIDs are idempotent (skipped). Context
    sufficiency fields cannot be rewritten through this interface.
    """

    current = _normalize_state(state)
    obs_list = _normalize_observations(observations)

    if not obs_list:
        return RouteCalibrationUpdateResult(
            disposition=RouteCalibrationDisposition.SKIPPED_EMPTY.value,
            state=current,
            applied_observation_cids=(),
            skipped_observation_cids=(),
            reason_codes=("no_observations",),
            notes=notes or "no observations to apply",
        )

    applied: list[str] = []
    skipped: list[str] = []
    skipped_simulated = 0
    skipped_idempotent = 0
    tier_map = {tier: current.tier_metrics[tier] for tier in _ROUTE_TIERS}
    seen = set(current.applied_observation_cids)
    reasons: list[str] = []

    for obs in obs_list:
        obs_cid = obs.observation_cid
        if obs_cid in seen:
            skipped.append(obs_cid)
            skipped_idempotent += 1
            reasons.append("skipped_idempotent")
            continue
        if obs.simulated:
            skipped.append(obs_cid)
            skipped_simulated += 1
            reasons.append("skipped_simulated")
            continue

        # Availability invariant on the observation itself: when required tier
        # was unavailable, recorded route must not be a weaker model tier.
        if not obs.required_tier_available:
            required = obs.required_route_tier or obs.route_tier
            if (
                _ROUTE_RANK[obs.route_tier] < _ROUTE_RANK[required]
                and obs.route_tier != RouteTier.HUMAN.value
            ):
                raise RouteCalibrationError(
                    "observation records a downgrade for unavailable required tier",
                    reason_code="route_downgrade_forbidden",
                    details={
                        "required_route_tier": required,
                        "route_tier": obs.route_tier,
                    },
                )

        metrics = tier_map[obs.route_tier]
        tier_map[obs.route_tier] = _apply_observation_to_metrics(metrics, obs)
        applied.append(obs_cid)
        seen.add(obs_cid)

    if not applied:
        if skipped_simulated and not skipped_idempotent:
            disposition = RouteCalibrationDisposition.SKIPPED_SIMULATED.value
        elif skipped:
            disposition = RouteCalibrationDisposition.SKIPPED_IDEMPOTENT.value
        else:
            disposition = RouteCalibrationDisposition.SKIPPED_EMPTY.value
        return RouteCalibrationUpdateResult(
            disposition=disposition,
            state=current,
            applied_observation_cids=(),
            skipped_observation_cids=tuple(sorted(set(skipped))),
            reason_codes=tuple(sorted(set(reasons))) or ("no_live_observations",),
            notes=notes,
        )

    input_cids = list(current.applied_observation_cids) + applied
    header = _header(
        "model_route_calibration_state",
        input_cids=input_cids[-MAX_CID_LIST:],
        interface_id=UPDATE_MODEL_ROUTE_CALIBRATION_INTERFACE,
    )
    new_revision = current.revision + 1
    if new_revision > MAX_REVISION:
        raise RouteCalibrationError("revision exceeds maximum")

    new_state = ModelRouteCalibrationState(
        header=header,
        state_id=current.state_id,
        partition=current.partition,
        revision=new_revision,
        tier_metrics=tier_map,
        applied_observation_cids=tuple(sorted(seen))[-MAX_CID_LIST:],
        notes=notes if notes is not None else current.notes,
        metadata=_thaw_structured(current.metadata),
    )

    return RouteCalibrationUpdateResult(
        disposition=RouteCalibrationDisposition.APPLIED.value,
        state=new_state,
        applied_observation_cids=tuple(sorted(set(applied))),
        skipped_observation_cids=tuple(sorted(set(skipped))),
        reason_codes=tuple(sorted(set(reasons + ["calibration_applied"]))),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Public: propose_route_threshold_change
# ---------------------------------------------------------------------------


def _next_stronger_tier(tier: str) -> str | None:
    rank = _ROUTE_RANK[tier]
    for candidate, candidate_rank in _ROUTE_RANK.items():
        if candidate_rank == rank + 1:
            return candidate
    return None


def _would_reduce_high_risk(
    parameter: str,
    current_value: str,
    proposed_value: str,
    policy: RouteThresholdPolicy,
) -> bool:
    """Return True when a proposal would lower high-risk assurance."""

    if parameter == RouteThresholdParameter.MIN_ROUTE_TIER.value:
        try:
            current_rank = _ROUTE_RANK[_route_tier(current_value)]
            proposed_rank = _ROUTE_RANK[_route_tier(proposed_value)]
        except RouteCalibrationError:
            return True
        high_floor = _ROUTE_RANK[policy.high_risk_min_route_tier]
        # Lowering min tier below high-risk floor reduces assurance.
        if proposed_rank < current_rank:
            return True
        if proposed_rank < high_floor and current_rank >= high_floor:
            return True
        return False

    if parameter == RouteThresholdParameter.ESCALATE_ON_UNAVAILABLE.value:
        return current_value == "true" and proposed_value == "false"

    if parameter in {
        RouteThresholdParameter.MIN_ACCEPTED_RATE_BP.value,
        RouteThresholdParameter.SHADOW_SAMPLE_RATE_BP.value,
    }:
        try:
            return int(proposed_value) < int(current_value)
        except ValueError:
            return True

    if parameter in {
        RouteThresholdParameter.MAX_CONTEXT_OMISSION_RATE_BP.value,
        RouteThresholdParameter.MAX_REASONING_FAILURE_RATE_BP.value,
        RouteThresholdParameter.MAX_RETRY_RATE_BP.value,
    }:
        # Raising max failure ceilings reduces assurance.
        try:
            return int(proposed_value) > int(current_value)
        except ValueError:
            return True

    return False


def propose_route_threshold_change(
    state: ModelRouteCalibrationState | Mapping[str, Any] | None,
    policy: RouteThresholdPolicy | Mapping[str, Any] | None = None,
    *,
    high_risk: bool = False,
    notes: str | None = None,
) -> RouteThresholdProposalResult:
    """Emit route-threshold change *proposals* only — never mutate production.

    Separate omission and reasoning rates drive separate threshold proposals.
    High-risk floors are never auto-lowered. Unavailable-tier policy remains
    escalate-to-human (no downgrade).
    """

    current = _normalize_state(state)
    threshold_policy = RouteThresholdPolicy.from_value(policy)
    high_risk = _bool(high_risk, "high_risk")

    proposals: list[RouteThresholdProposal] = []
    rejected_reasons: list[str] = []

    # Always reinforce no-downgrade availability policy when disabled.
    if not threshold_policy.escalate_on_unavailable:
        rejected_reasons.append("rejected_disable_escalate_on_unavailable")
        # Propose re-enabling rather than accepting a downgrade policy.
        proposals.append(
            RouteThresholdProposal(
                proposal_id="restore_escalate_on_unavailable",
                route_tier=RouteTier.HUMAN.value,
                parameter=RouteThresholdParameter.ESCALATE_ON_UNAVAILABLE.value,
                current_value="false",
                proposed_value="true",
                reason_codes=(
                    "unavailable_required_tier_never_downgrades",
                    "restore_escalate_on_unavailable",
                ),
                high_risk_assurance_reduced=False,
                mutates_production=False,
                is_proposal_only=True,
                notes="Unavailable required tiers must escalate to human",
            )
        )

    for tier in _ROUTE_TIERS:
        metrics = current.metrics_for(tier)
        if metrics.total_uses < threshold_policy.min_uses_for_proposal:
            continue

        support = list(metrics.source_receipt_cids)[:16]

        # Accepted-rate floor: propose raising floor when observed acceptance is weak.
        if metrics.accepted_rate_bp < threshold_policy.min_accepted_rate_bp:
            proposed = str(
                min(
                    BASIS_POINTS,
                    max(
                        threshold_policy.min_accepted_rate_bp,
                        metrics.accepted_rate_bp + 500,
                    ),
                )
            )
            current_value = str(threshold_policy.min_accepted_rate_bp)
            if not _would_reduce_high_risk(
                RouteThresholdParameter.MIN_ACCEPTED_RATE_BP.value,
                current_value,
                proposed,
                threshold_policy,
            ):
                proposals.append(
                    RouteThresholdProposal(
                        proposal_id=f"raise_min_accepted_rate_{tier}",
                        route_tier=tier,
                        parameter=RouteThresholdParameter.MIN_ACCEPTED_RATE_BP.value,
                        current_value=current_value,
                        proposed_value=proposed,
                        reason_codes=(
                            "low_accepted_rate",
                            f"tier_{tier}",
                        ),
                        supporting_observation_cids=tuple(support),
                        notes=f"Observed accepted_rate_bp={metrics.accepted_rate_bp}",
                    )
                )

        # Context omission — separate from reasoning failure.
        if metrics.context_omission_rate_bp > threshold_policy.max_context_omission_rate_bp:
            # Tighten max omission ceiling (lower is stricter / safer).
            proposed = str(
                max(
                    0,
                    min(
                        threshold_policy.max_context_omission_rate_bp,
                        metrics.context_omission_rate_bp - 500,
                    ),
                )
            )
            current_value = str(threshold_policy.max_context_omission_rate_bp)
            if not _would_reduce_high_risk(
                RouteThresholdParameter.MAX_CONTEXT_OMISSION_RATE_BP.value,
                current_value,
                proposed,
                threshold_policy,
            ):
                proposals.append(
                    RouteThresholdProposal(
                        proposal_id=f"tighten_max_context_omission_{tier}",
                        route_tier=tier,
                        parameter=(
                            RouteThresholdParameter.MAX_CONTEXT_OMISSION_RATE_BP.value
                        ),
                        current_value=current_value,
                        proposed_value=proposed,
                        reason_codes=(
                            "elevated_context_omission_rate",
                            "omission_counter_separate",
                            f"tier_{tier}",
                        ),
                        supporting_observation_cids=tuple(support),
                        notes=(
                            "Context omission rate elevated; "
                            f"count={metrics.context_omission_failure_count} "
                            f"rate_bp={metrics.context_omission_rate_bp}"
                        ),
                    )
                )
            # Escalate min route tier when omission is severe (never lower).
            if metrics.context_omission_rate_bp >= DEFAULT_ESCALATION_OMISSION_RATE_BP:
                stronger = _next_stronger_tier(tier)
                if stronger is not None:
                    proposals.append(
                        RouteThresholdProposal(
                            proposal_id=f"escalate_min_route_on_omission_{tier}",
                            route_tier=tier,
                            parameter=RouteThresholdParameter.MIN_ROUTE_TIER.value,
                            current_value=tier,
                            proposed_value=stronger,
                            reason_codes=(
                                "context_omission_escalation",
                                "never_auto_lower",
                                f"tier_{tier}",
                            ),
                            supporting_observation_cids=tuple(support),
                            notes="Escalate capability tier after sustained omission",
                        )
                    )

        # Reasoning failure — separate counter, separate proposals.
        if (
            metrics.reasoning_failure_rate_bp
            > threshold_policy.max_reasoning_failure_rate_bp
        ):
            proposed = str(
                max(
                    0,
                    min(
                        threshold_policy.max_reasoning_failure_rate_bp,
                        metrics.reasoning_failure_rate_bp - 500,
                    ),
                )
            )
            current_value = str(threshold_policy.max_reasoning_failure_rate_bp)
            if not _would_reduce_high_risk(
                RouteThresholdParameter.MAX_REASONING_FAILURE_RATE_BP.value,
                current_value,
                proposed,
                threshold_policy,
            ):
                proposals.append(
                    RouteThresholdProposal(
                        proposal_id=f"tighten_max_reasoning_failure_{tier}",
                        route_tier=tier,
                        parameter=(
                            RouteThresholdParameter.MAX_REASONING_FAILURE_RATE_BP.value
                        ),
                        current_value=current_value,
                        proposed_value=proposed,
                        reason_codes=(
                            "elevated_reasoning_failure_rate",
                            "reasoning_counter_separate",
                            f"tier_{tier}",
                        ),
                        supporting_observation_cids=tuple(support),
                        notes=(
                            "Reasoning failure rate elevated; "
                            f"count={metrics.reasoning_failure_count} "
                            f"rate_bp={metrics.reasoning_failure_rate_bp}"
                        ),
                    )
                )
            if (
                metrics.reasoning_failure_rate_bp
                >= DEFAULT_ESCALATION_REASONING_RATE_BP
            ):
                stronger = _next_stronger_tier(tier)
                if stronger is not None:
                    proposals.append(
                        RouteThresholdProposal(
                            proposal_id=f"escalate_min_route_on_reasoning_{tier}",
                            route_tier=tier,
                            parameter=RouteThresholdParameter.MIN_ROUTE_TIER.value,
                            current_value=tier,
                            proposed_value=stronger,
                            reason_codes=(
                                "reasoning_failure_escalation",
                                "never_auto_lower",
                                f"tier_{tier}",
                            ),
                            supporting_observation_cids=tuple(support),
                            notes="Escalate capability tier after sustained reasoning failure",
                        )
                    )

        # Retry pressure can raise shadow sampling — never lowers assurance floors.
        if metrics.retry_rate_bp > threshold_policy.max_retry_rate_bp:
            proposed_shadow = str(
                min(
                    BASIS_POINTS,
                    max(
                        threshold_policy.shadow_sample_rate_bp,
                        threshold_policy.shadow_sample_rate_bp + 500,
                    ),
                )
            )
            proposals.append(
                RouteThresholdProposal(
                    proposal_id=f"raise_shadow_on_retry_{tier}",
                    route_tier=tier,
                    parameter=RouteThresholdParameter.SHADOW_SAMPLE_RATE_BP.value,
                    current_value=str(threshold_policy.shadow_sample_rate_bp),
                    proposed_value=proposed_shadow,
                    reason_codes=("elevated_retry_rate", f"tier_{tier}"),
                    supporting_observation_cids=tuple(support),
                    notes=f"retry_rate_bp={metrics.retry_rate_bp}",
                )
            )

        # Unavailable required tier pressure always reinforces human escalation.
        if metrics.unavailable_required_tier_count > 0:
            proposals.append(
                RouteThresholdProposal(
                    proposal_id=f"reinforce_no_downgrade_{tier}",
                    route_tier=tier,
                    parameter=RouteThresholdParameter.ESCALATE_ON_UNAVAILABLE.value,
                    current_value=(
                        "true" if threshold_policy.escalate_on_unavailable else "false"
                    ),
                    proposed_value="true",
                    reason_codes=(
                        "required_tier_unavailable_observed",
                        "never_downgrade",
                        f"tier_{tier}",
                    ),
                    supporting_observation_cids=tuple(support),
                    notes=(
                        "Unavailable required tier observations="
                        f"{metrics.unavailable_required_tier_count}"
                    ),
                )
            )

    # High-risk floor: reject any proposal that would lower high-risk min tier.
    if high_risk:
        filtered: list[RouteThresholdProposal] = []
        for proposal in proposals:
            if proposal.parameter == RouteThresholdParameter.MIN_ROUTE_TIER.value:
                if (
                    _ROUTE_RANK[proposal.proposed_value]
                    < _ROUTE_RANK[threshold_policy.high_risk_min_route_tier]
                ):
                    rejected_reasons.append("rejected_high_risk_reduction")
                    continue
            if proposal.high_risk_assurance_reduced:
                rejected_reasons.append("rejected_high_risk_reduction")
                continue
            filtered.append(proposal)
        proposals = filtered

    # De-duplicate by proposal_id while preserving order.
    seen_ids: set[str] = set()
    unique_proposals: list[RouteThresholdProposal] = []
    for proposal in proposals:
        if proposal.proposal_id in seen_ids:
            continue
        seen_ids.add(proposal.proposal_id)
        unique_proposals.append(proposal)
    unique_proposals = unique_proposals[:MAX_PROPOSALS]

    if not unique_proposals:
        disposition = RouteThresholdDisposition.NO_CHANGE.value
        reason_codes = tuple(sorted(set(rejected_reasons + ["no_threshold_change"])))
    else:
        disposition = RouteThresholdDisposition.PROPOSED.value
        reason_codes = tuple(
            sorted(set(rejected_reasons + ["threshold_proposals_emitted"]))
        )

    return RouteThresholdProposalResult(
        disposition=disposition,
        proposals=tuple(unique_proposals),
        policy_cid=threshold_policy.policy_cid,
        state_cid=current.state_cid,
        reason_codes=reason_codes,
        mutates_production=False,
        notes=notes or "route threshold changes are proposals only",
    )


# ---------------------------------------------------------------------------
# Helpers / vocabulary pins
# ---------------------------------------------------------------------------


def update_model_route_calibration_interface_id() -> str:
    return UPDATE_MODEL_ROUTE_CALIBRATION_INTERFACE


def propose_route_threshold_change_interface_id() -> str:
    return PROPOSE_ROUTE_THRESHOLD_CHANGE_INTERFACE


def route_calibration_evidence_id() -> str:
    return SCG_ROUTE_CALIBRATION_EVIDENCE


def route_tiers() -> tuple[str, ...]:
    return _ROUTE_TIERS


def route_failure_kinds() -> tuple[str, ...]:
    return tuple(kind.value for kind in RouteFailureKind)


def route_threshold_parameters() -> tuple[str, ...]:
    return tuple(param.value for param in RouteThresholdParameter)


def observation_from_receipt_fields(
    *,
    observation_id: str,
    route_tier: RouteTier | str,
    accepted: bool,
    receipt_cid: str | None = None,
    decision_cid: str | None = None,
    wall_time_ms: int = 0,
    spend_micros: int = 0,
    expansion_plan_cid: str | None = None,
    omission_evidence_cid: str | None = None,
    retried: bool = False,
    verification_passed: bool | None = None,
    context_omission_failure: bool = False,
    reasoning_failure: bool = False,
    required_route_tier: RouteTier | str | None = None,
    required_tier_available: bool = True,
    simulated: bool = False,
    reason_codes: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> RouteRunObservation:
    """Build a route observation from GovernorRunReceipt-shaped fields.

    Expansion plan presence signals expansion use. Omission evidence CID is
    *not* automatically treated as a context-omission failure — the caller
    must set ``context_omission_failure`` explicitly so reasoning failures
    remain distinguishable.
    """

    meta = dict(metadata or {})
    if expansion_plan_cid is not None:
        meta.setdefault("expansion_plan_cid", expansion_plan_cid)
    if omission_evidence_cid is not None:
        # Reference only; does not auto-increment omission counter.
        meta.setdefault("omission_evidence_cid", omission_evidence_cid)

    return RouteRunObservation(
        observation_id=observation_id,
        route_tier=route_tier,
        accepted=accepted,
        retried=retried,
        expansion_used=expansion_plan_cid is not None,
        verification_passed=verification_passed,
        context_omission_failure=context_omission_failure,
        reasoning_failure=reasoning_failure,
        required_route_tier=required_route_tier,
        required_tier_available=required_tier_available,
        cost_micros=spend_micros,
        latency_ms=wall_time_ms,
        receipt_cid=receipt_cid,
        decision_cid=decision_cid,
        reason_codes=tuple(reason_codes),
        simulated=simulated,
        metadata=meta,
    )


__all__ = [
    "DEFAULT_ESCALATION_OMISSION_RATE_BP",
    "DEFAULT_ESCALATION_REASONING_RATE_BP",
    "DEFAULT_MAX_OMISSION_RATE_BP",
    "DEFAULT_MAX_REASONING_FAILURE_RATE_BP",
    "DEFAULT_MAX_RETRY_RATE_BP",
    "DEFAULT_MIN_ACCEPTED_RATE_BP",
    "DEFAULT_MIN_USES_FOR_PROPOSAL",
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "PROPOSE_ROUTE_THRESHOLD_CHANGE_INTERFACE",
    "ROUTE_AVAILABILITY_DECISION_SCHEMA",
    "ROUTE_CALIBRATION_STATE_INTERFACE",
    "ROUTE_CALIBRATION_STATE_SCHEMA",
    "ROUTE_CALIBRATION_UPDATE_RESULT_SCHEMA",
    "ROUTE_RUN_OBSERVATION_SCHEMA",
    "ROUTE_THRESHOLD_POLICY_SCHEMA",
    "ROUTE_THRESHOLD_PROPOSAL_RESULT_SCHEMA",
    "ROUTE_THRESHOLD_PROPOSAL_SCHEMA",
    "ROUTE_TIER_METRICS_SCHEMA",
    "SCG_ROUTE_CALIBRATION_EVIDENCE",
    "UPDATE_MODEL_ROUTE_CALIBRATION_INTERFACE",
    "ModelRouteCalibrationState",
    "RouteAvailabilityDecision",
    "RouteAvailabilityDisposition",
    "RouteCalibrationContractError",
    "RouteCalibrationDisposition",
    "RouteCalibrationError",
    "RouteCalibrationUpdateResult",
    "RouteFailureKind",
    "RouteRunObservation",
    "RouteThresholdDisposition",
    "RouteThresholdParameter",
    "RouteThresholdPolicy",
    "RouteThresholdProposal",
    "RouteThresholdProposalResult",
    "RouteTierMetrics",
    "default_route_threshold_policy",
    "observation_from_receipt_fields",
    "propose_route_threshold_change",
    "propose_route_threshold_change_interface_id",
    "resolve_route_availability",
    "route_calibration_evidence_id",
    "route_failure_kinds",
    "route_threshold_parameters",
    "route_tiers",
    "update_model_route_calibration",
    "update_model_route_calibration_interface_id",
]
