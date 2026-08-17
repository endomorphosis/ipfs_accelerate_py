"""Risk- and information-value-aware shadow planning (SCG-025).

Builds a deterministic :class:`ShadowExecutionPlan` from task signals, a
versioned :class:`ShadowSamplingPolicy` (audit policy), repository state, and
privacy/resource gates.

Normative invariants (fail-closed):

* Development and high/critical risk may shadow at 100 percent when configured.
* Mature low-risk work samples at a lower configurable rate (never forced 100%).
* Forbidden expanded disclosure yields local-only evaluation or no external
  call (``allow_external_expanded_disclosure=False`` / disclosure-skip reason);
  never a policy bypass.
* Expanded output remains oracle/candidate only; isolated evaluation worktree
  is always required.
* Sampling is deterministic given an explicit random seed and task identity.
* Importing this module performs no I/O, opens no sockets, and never invokes a
  provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Iterable, Mapping, Sequence
import hashlib
import re
import unicodedata

from ipfs_accelerate_py.agent_supervisor.semantic_governor.contracts import (
    SHADOW_EXECUTION_PLAN_INTERFACE,
    ShadowExecutionPlan,
    ShadowSelectionReason,
    verify_plan_identity,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.privacy import (
    DisclosureDisposition,
    DisclosureForbiddenError,
    ProviderLocality,
    ShadowDisclosurePolicy,
    SourcePrivacyClass,
    authorize_shadow_disclosure,
    classify_provider_locality,
    classify_source_privacy,
    contains_private_source,
    default_shadow_disclosure_policy,
)
from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_structured,
    validate_cid,
    validate_structured_value,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.base import (
    ArtifactProvenance,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    GovernorArtifactHeader,
    GovernorAssumption,
    GovernorTerminalStatus,
    AssumptionKind,
    SemanticGovernorBaseError,
    reject_private_and_model_authority,
)

# ---------------------------------------------------------------------------
# Evidence / interface / schema constants
# ---------------------------------------------------------------------------

SCG_SHADOW_PLAN_EVIDENCE: Final[str] = "scg/shadow-plan@1"

CREATE_SHADOW_PLAN_INTERFACE: Final[str] = "create_shadow_plan@1"
SHADOW_SAMPLING_POLICY_INTERFACE: Final[str] = "ShadowSamplingPolicy@1"
SHADOW_SAMPLING_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "shadow-sampling-policy@1"
)
SHADOW_PLAN_DECISION_INTERFACE: Final[str] = "ShadowPlanDecision@1"
SHADOW_PLAN_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "shadow-plan-decision@1"
)

GENERATOR_ID: Final[str] = "shadow_plan"
GENERATOR_VERSION: Final[str] = "1.0.0"

# Basis points: 10_000 == 100 percent.
BASIS_POINTS_MAX: Final[int] = 10_000

# Default rates (configurable via ShadowSamplingPolicy).
DEFAULT_DEVELOPMENT_RATE_BP: Final[int] = BASIS_POINTS_MAX
DEFAULT_HIGH_RISK_RATE_BP: Final[int] = BASIS_POINTS_MAX
DEFAULT_CRITICAL_RISK_RATE_BP: Final[int] = BASIS_POINTS_MAX
DEFAULT_MEDIUM_RISK_RATE_BP: Final[int] = 1_000  # 10%
DEFAULT_MATURE_LOW_RISK_RATE_BP: Final[int] = 250  # 2.5%
DEFAULT_CAPSULE_UNCERTAINTY_RATE_BP: Final[int] = 5_000
DEFAULT_NOVELTY_RATE_BP: Final[int] = BASIS_POINTS_MAX
DEFAULT_TOKEN_SAVINGS_RATE_BP: Final[int] = 500
DEFAULT_PROOF_CACHE_REUSE_RATE_BP: Final[int] = 500
DEFAULT_RECENT_OMISSION_RATE_BP: Final[int] = 7_500
DEFAULT_RANDOM_QC_RATE_BP: Final[int] = 100  # 1%
DEFAULT_PROMOTION_EVALUATION_RATE_BP: Final[int] = BASIS_POINTS_MAX

DEFAULT_MAX_WALL_TIME_MS: Final[int] = 600_000
DEFAULT_MAX_MODEL_SPEND_MICROS: Final[int] = 50_000_000
DEFAULT_MAX_EXPANSION_TOKEN_BUDGET: Final[int] = 128_000
DEFAULT_RANDOM_SEED: Final[int] = 0

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_REASON_CODES: Final[int] = 64
MAX_METADATA_KEYS: Final[int] = 64
MAX_ROUTE_CHARS: Final[int] = 128

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")
_TASK_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z][A-Za-z0-9_.:/+-]{0,127}$"
)

HIGH_RISK_CLASSES: Final[frozenset[str]] = frozenset(
    {"high", "critical", "risk_high", "risk_critical"}
)
MEDIUM_RISK_CLASSES: Final[frozenset[str]] = frozenset(
    {"medium", "moderate", "risk_medium"}
)
LOW_RISK_CLASSES: Final[frozenset[str]] = frozenset(
    {"low", "minimal", "risk_low", "mature_low"}
)
DEVELOPMENT_ENVIRONMENTS: Final[frozenset[str]] = frozenset(
    {"development", "dev", "local_dev", "ci_development"}
)
MATURE_ENVIRONMENTS: Final[frozenset[str]] = frozenset(
    {"mature", "production", "prod", "staging", "canary"}
)


class SemanticGovernorShadowPlanError(SemanticGovernorBaseError):
    """Raised when shadow planning inputs or policy are malformed."""


class ShadowPlanNotSelected(SemanticGovernorShadowPlanError):
    """Raised when create_shadow_plan is asked to require a plan but sampling skipped."""


class ResourceGateError(SemanticGovernorShadowPlanError):
    """Raised when resource admission gates refuse a shadow plan."""


class LifecyclePhase(str, Enum):
    """Closed environment / maturity phase for rate selection."""

    DEVELOPMENT = "development"
    MATURE = "mature"
    PRODUCTION = "production"


class ShadowPlanDisposition(str, Enum):
    """Closed outcome of create_shadow_plan sampling."""

    SELECTED = "selected"
    SKIPPED = "skipped"
    DISCLOSURE_LOCAL_ONLY = "disclosure_local_only"
    DISCLOSURE_EXTERNAL_SKIPPED = "disclosure_external_skipped"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _normalize_token(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        raise SemanticGovernorShadowPlanError(f"{name} must be a nonempty string")
    text = _normalize_token(value)
    if text != value and value != value.strip():
        # Accept already-normalized input; reject dirty whitespace/control.
        raise SemanticGovernorShadowPlanError(f"{name} must be trimmed NFC text")
    if unicodedata.normalize("NFC", text) != text:
        raise SemanticGovernorShadowPlanError(f"{name} must be trimmed NFC text")
    if len(text) > MAX_TEXT_CHARS or any(not char.isprintable() for char in text):
        raise SemanticGovernorShadowPlanError(f"{name} contains invalid text")
    return text


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if _TOKEN_RE.fullmatch(text) is None:
        raise SemanticGovernorShadowPlanError(
            f"{name} must be a lowercase token matching {_TOKEN_RE.pattern}"
        )
    return text


def _task_id(value: Any, name: str = "task_id") -> str:
    text = _text(value, name)
    if _TASK_ID_RE.fullmatch(text) is None:
        raise SemanticGovernorShadowPlanError(
            f"{name} must match {_TASK_ID_RE.pattern}"
        )
    return text


def _cid(value: Any, name: str) -> str:
    try:
        return validate_cid(value)
    except Exception as exc:
        raise SemanticGovernorShadowPlanError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise SemanticGovernorShadowPlanError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise SemanticGovernorShadowPlanError(f"{name} must be a non-negative int")
    return value


def _basis_points(value: Any, name: str) -> int:
    rate = _nonneg_int(value, name)
    if rate > BASIS_POINTS_MAX:
        raise SemanticGovernorShadowPlanError(
            f"{name} must be between 0 and {BASIS_POINTS_MAX} inclusive"
        )
    return rate


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    try:
        return enum_type(value).value
    except (TypeError, ValueError) as exc:
        raise SemanticGovernorShadowPlanError(
            f"{name} has unsupported value {value!r}"
        ) from exc


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


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise SemanticGovernorShadowPlanError(f"{name} must be a mapping")
    actual = set(data)
    if actual != fields:
        raise SemanticGovernorShadowPlanError(
            f"{name} fields must be exactly {sorted(fields)}, got {sorted(actual)}"
        )
    return dict(data)


def _require_structured(value: Any, name: str) -> Any:
    thawed = _thaw_structured(value)
    try:
        validate_structured_value(thawed, path=name)
    except Exception as exc:
        raise SemanticGovernorShadowPlanError(
            f"{name} must be strict DAG-JSON without floats or host types"
        ) from exc
    try:
        reject_private_and_model_authority(thawed, path=name)
    except SemanticGovernorBaseError as exc:
        raise SemanticGovernorShadowPlanError(str(exc)) from exc
    return thawed


def _mapping(value: Any, name: str, *, frozen: bool = True) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticGovernorShadowPlanError(f"{name} must be a mapping")
    if len(value) > MAX_METADATA_KEYS:
        raise SemanticGovernorShadowPlanError(f"{name} exceeds maximum key count")
    result = _require_structured(dict(value), name)
    return _freeze_structured(result) if frozen else result


def _unique_sorted_tokens(
    values: Iterable[Any], name: str, *, max_items: int
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise SemanticGovernorShadowPlanError(f"{name} must be a list")
    ordered = tuple(sorted(_token(value, name) for value in values))
    if len(ordered) > max_items:
        raise SemanticGovernorShadowPlanError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise SemanticGovernorShadowPlanError(f"{name} must not contain duplicates")
    return ordered


def _mapping_get(data: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


# ---------------------------------------------------------------------------
# Deterministic sampling
# ---------------------------------------------------------------------------


def deterministic_sample_roll(
    *,
    random_seed: int,
    task_id: str,
    compressed_context_pack_cid: str,
    salt: str = "shadow-sample",
) -> int:
    """Return a deterministic roll in ``[0, BASIS_POINTS_MAX)``.

    Same seed and identity always yield the same roll; no global RNG is used.
    """

    seed = _nonneg_int(random_seed, "random_seed")
    tid = _task_id(task_id)
    pack = _cid(compressed_context_pack_cid, "compressed_context_pack_cid")
    salt_text = _text(salt, "salt")
    material = f"{seed}|{tid}|{pack}|{salt_text}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    # Use 32 bits of the digest; modular reduction is fine for sampling gates.
    value = int.from_bytes(digest[:4], "big")
    return value % BASIS_POINTS_MAX


def sample_hits(roll: int, rate_bp: int) -> bool:
    """True when *roll* is admitted by *rate_bp* (10000 always hits, 0 never)."""

    rate = _basis_points(rate_bp, "rate_bp")
    r = _nonneg_int(roll, "roll")
    if r >= BASIS_POINTS_MAX:
        raise SemanticGovernorShadowPlanError(
            f"roll must be in [0, {BASIS_POINTS_MAX})"
        )
    if rate >= BASIS_POINTS_MAX:
        return True
    if rate == 0:
        return False
    return r < rate


# ---------------------------------------------------------------------------
# ShadowSamplingPolicy (audit policy)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShadowSamplingPolicy:
    """Versioned audit policy for risk- and information-value-aware sampling.

    Rates are integer basis points (0–10000). Development and high/critical
    risk default to full shadowing; mature low-risk defaults to a low sample
    rate. Privacy zero-call / local-only is enforced via
    ``allow_external_expanded_disclosure`` defaults and the privacy gate.
    """

    policy_id: str = "shadow-sampling-default"
    lifecycle_phase: LifecyclePhase | str = LifecyclePhase.MATURE
    random_seed: int = DEFAULT_RANDOM_SEED
    development_sample_rate_bp: int = DEFAULT_DEVELOPMENT_RATE_BP
    high_risk_sample_rate_bp: int = DEFAULT_HIGH_RISK_RATE_BP
    critical_risk_sample_rate_bp: int = DEFAULT_CRITICAL_RISK_RATE_BP
    medium_risk_sample_rate_bp: int = DEFAULT_MEDIUM_RISK_RATE_BP
    mature_low_risk_sample_rate_bp: int = DEFAULT_MATURE_LOW_RISK_RATE_BP
    capsule_uncertainty_rate_bp: int = DEFAULT_CAPSULE_UNCERTAINTY_RATE_BP
    novelty_rate_bp: int = DEFAULT_NOVELTY_RATE_BP
    token_savings_rate_bp: int = DEFAULT_TOKEN_SAVINGS_RATE_BP
    proof_cache_reuse_rate_bp: int = DEFAULT_PROOF_CACHE_REUSE_RATE_BP
    recent_omission_rate_bp: int = DEFAULT_RECENT_OMISSION_RATE_BP
    random_quality_control_rate_bp: int = DEFAULT_RANDOM_QC_RATE_BP
    promotion_evaluation_rate_bp: int = DEFAULT_PROMOTION_EVALUATION_RATE_BP
    max_wall_time_ms: int = DEFAULT_MAX_WALL_TIME_MS
    max_model_spend_micros: int = DEFAULT_MAX_MODEL_SPEND_MICROS
    max_expansion_token_budget: int = DEFAULT_MAX_EXPANSION_TOKEN_BUDGET
    require_isolated_evaluation_worktree: bool = True
    expanded_is_oracle_candidate_only: bool = True
    # Default false: external expanded disclosure requires exact privacy authority.
    allow_external_expanded_disclosure: bool = False
    # When true, forbidden external disclosure skips the external call entirely
    # (plan still local-only with DISCLOSURE_FORBIDDEN_SKIP). When false, plan
    # is local-only without that skip reason when local expanded remains viable.
    zero_external_calls_when_disclosure_forbidden: bool = True
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface_id",
            "policy_id",
            "lifecycle_phase",
            "random_seed",
            "development_sample_rate_bp",
            "high_risk_sample_rate_bp",
            "critical_risk_sample_rate_bp",
            "medium_risk_sample_rate_bp",
            "mature_low_risk_sample_rate_bp",
            "capsule_uncertainty_rate_bp",
            "novelty_rate_bp",
            "token_savings_rate_bp",
            "proof_cache_reuse_rate_bp",
            "recent_omission_rate_bp",
            "random_quality_control_rate_bp",
            "promotion_evaluation_rate_bp",
            "max_wall_time_ms",
            "max_model_spend_micros",
            "max_expansion_token_budget",
            "require_isolated_evaluation_worktree",
            "expanded_is_oracle_candidate_only",
            "allow_external_expanded_disclosure",
            "zero_external_calls_when_disclosure_forbidden",
            "notes",
            "metadata",
            "policy_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _token(self.policy_id, "policy_id"))
        object.__setattr__(
            self,
            "lifecycle_phase",
            _enum(self.lifecycle_phase, LifecyclePhase, "lifecycle_phase"),
        )
        object.__setattr__(
            self, "random_seed", _nonneg_int(self.random_seed, "random_seed")
        )
        for name in (
            "development_sample_rate_bp",
            "high_risk_sample_rate_bp",
            "critical_risk_sample_rate_bp",
            "medium_risk_sample_rate_bp",
            "mature_low_risk_sample_rate_bp",
            "capsule_uncertainty_rate_bp",
            "novelty_rate_bp",
            "token_savings_rate_bp",
            "proof_cache_reuse_rate_bp",
            "recent_omission_rate_bp",
            "random_quality_control_rate_bp",
            "promotion_evaluation_rate_bp",
        ):
            object.__setattr__(self, name, _basis_points(getattr(self, name), name))
        object.__setattr__(
            self,
            "max_wall_time_ms",
            _nonneg_int(self.max_wall_time_ms, "max_wall_time_ms"),
        )
        object.__setattr__(
            self,
            "max_model_spend_micros",
            _nonneg_int(self.max_model_spend_micros, "max_model_spend_micros"),
        )
        object.__setattr__(
            self,
            "max_expansion_token_budget",
            _nonneg_int(self.max_expansion_token_budget, "max_expansion_token_budget"),
        )
        object.__setattr__(
            self,
            "require_isolated_evaluation_worktree",
            _bool(
                self.require_isolated_evaluation_worktree,
                "require_isolated_evaluation_worktree",
            ),
        )
        if not self.require_isolated_evaluation_worktree:
            raise SemanticGovernorShadowPlanError(
                "require_isolated_evaluation_worktree must be true"
            )
        object.__setattr__(
            self,
            "expanded_is_oracle_candidate_only",
            _bool(
                self.expanded_is_oracle_candidate_only,
                "expanded_is_oracle_candidate_only",
            ),
        )
        if not self.expanded_is_oracle_candidate_only:
            raise SemanticGovernorShadowPlanError(
                "expanded_is_oracle_candidate_only must be true by construction"
            )
        object.__setattr__(
            self,
            "allow_external_expanded_disclosure",
            _bool(
                self.allow_external_expanded_disclosure,
                "allow_external_expanded_disclosure",
            ),
        )
        object.__setattr__(
            self,
            "zero_external_calls_when_disclosure_forbidden",
            _bool(
                self.zero_external_calls_when_disclosure_forbidden,
                "zero_external_calls_when_disclosure_forbidden",
            ),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": SHADOW_SAMPLING_POLICY_SCHEMA,
            "interface_id": SHADOW_SAMPLING_POLICY_INTERFACE,
            "policy_id": self.policy_id,
            "lifecycle_phase": self.lifecycle_phase,
            "random_seed": self.random_seed,
            "development_sample_rate_bp": self.development_sample_rate_bp,
            "high_risk_sample_rate_bp": self.high_risk_sample_rate_bp,
            "critical_risk_sample_rate_bp": self.critical_risk_sample_rate_bp,
            "medium_risk_sample_rate_bp": self.medium_risk_sample_rate_bp,
            "mature_low_risk_sample_rate_bp": self.mature_low_risk_sample_rate_bp,
            "capsule_uncertainty_rate_bp": self.capsule_uncertainty_rate_bp,
            "novelty_rate_bp": self.novelty_rate_bp,
            "token_savings_rate_bp": self.token_savings_rate_bp,
            "proof_cache_reuse_rate_bp": self.proof_cache_reuse_rate_bp,
            "recent_omission_rate_bp": self.recent_omission_rate_bp,
            "random_quality_control_rate_bp": self.random_quality_control_rate_bp,
            "promotion_evaluation_rate_bp": self.promotion_evaluation_rate_bp,
            "max_wall_time_ms": self.max_wall_time_ms,
            "max_model_spend_micros": self.max_model_spend_micros,
            "max_expansion_token_budget": self.max_expansion_token_budget,
            "require_isolated_evaluation_worktree": (
                self.require_isolated_evaluation_worktree
            ),
            "expanded_is_oracle_candidate_only": self.expanded_is_oracle_candidate_only,
            "allow_external_expanded_disclosure": (
                self.allow_external_expanded_disclosure
            ),
            "zero_external_calls_when_disclosure_forbidden": (
                self.zero_external_calls_when_disclosure_forbidden
            ),
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def policy_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["policy_cid"] = self.policy_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShadowSamplingPolicy":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("policy_cid")
        schema = payload.pop("schema")
        interface_id = payload.pop("interface_id")
        if schema != SHADOW_SAMPLING_POLICY_SCHEMA:
            raise SemanticGovernorShadowPlanError(
                "unsupported ShadowSamplingPolicy schema version"
            )
        if interface_id != SHADOW_SAMPLING_POLICY_INTERFACE:
            raise SemanticGovernorShadowPlanError(
                "unsupported ShadowSamplingPolicy interface_id"
            )
        result = cls(
            policy_id=payload["policy_id"],
            lifecycle_phase=payload["lifecycle_phase"],
            random_seed=payload["random_seed"],
            development_sample_rate_bp=payload["development_sample_rate_bp"],
            high_risk_sample_rate_bp=payload["high_risk_sample_rate_bp"],
            critical_risk_sample_rate_bp=payload["critical_risk_sample_rate_bp"],
            medium_risk_sample_rate_bp=payload["medium_risk_sample_rate_bp"],
            mature_low_risk_sample_rate_bp=payload["mature_low_risk_sample_rate_bp"],
            capsule_uncertainty_rate_bp=payload["capsule_uncertainty_rate_bp"],
            novelty_rate_bp=payload["novelty_rate_bp"],
            token_savings_rate_bp=payload["token_savings_rate_bp"],
            proof_cache_reuse_rate_bp=payload["proof_cache_reuse_rate_bp"],
            recent_omission_rate_bp=payload["recent_omission_rate_bp"],
            random_quality_control_rate_bp=payload["random_quality_control_rate_bp"],
            promotion_evaluation_rate_bp=payload["promotion_evaluation_rate_bp"],
            max_wall_time_ms=payload["max_wall_time_ms"],
            max_model_spend_micros=payload["max_model_spend_micros"],
            max_expansion_token_budget=payload["max_expansion_token_budget"],
            require_isolated_evaluation_worktree=payload[
                "require_isolated_evaluation_worktree"
            ],
            expanded_is_oracle_candidate_only=payload[
                "expanded_is_oracle_candidate_only"
            ],
            allow_external_expanded_disclosure=payload[
                "allow_external_expanded_disclosure"
            ],
            zero_external_calls_when_disclosure_forbidden=payload[
                "zero_external_calls_when_disclosure_forbidden"
            ],
            notes=payload["notes"],
            metadata=payload["metadata"],
        )
        if claimed != result.policy_cid:
            raise SemanticGovernorShadowPlanError(
                "ShadowSamplingPolicy policy_cid does not verify"
            )
        return result


def default_shadow_sampling_policy(
    *,
    lifecycle_phase: LifecyclePhase | str = LifecyclePhase.MATURE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> ShadowSamplingPolicy:
    """Return the production-default mature sampling policy."""

    return ShadowSamplingPolicy(
        lifecycle_phase=lifecycle_phase,
        random_seed=random_seed,
    )


def development_shadow_sampling_policy(
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> ShadowSamplingPolicy:
    """Return a development policy with 100 percent shadowing by default."""

    return ShadowSamplingPolicy(
        policy_id="shadow-sampling-development",
        lifecycle_phase=LifecyclePhase.DEVELOPMENT,
        random_seed=random_seed,
        development_sample_rate_bp=BASIS_POINTS_MAX,
    )


def _coerce_sampling_policy(
    audit_policy: ShadowSamplingPolicy | Mapping[str, Any] | None,
) -> ShadowSamplingPolicy:
    if audit_policy is None:
        return default_shadow_sampling_policy()
    if isinstance(audit_policy, ShadowSamplingPolicy):
        return audit_policy
    if isinstance(audit_policy, Mapping):
        # Accept either full sealed dict or partial override dict.
        if "schema" in audit_policy and "interface_id" in audit_policy:
            return ShadowSamplingPolicy.from_dict(audit_policy)
        allowed = {
            f.name
            for f in ShadowSamplingPolicy.__dataclass_fields__.values()  # type: ignore[attr-defined]
        }
        kwargs = {k: v for k, v in audit_policy.items() if k in allowed}
        return ShadowSamplingPolicy(**kwargs)
    raise SemanticGovernorShadowPlanError(
        "audit_policy must be ShadowSamplingPolicy or mapping"
    )


# ---------------------------------------------------------------------------
# Input views
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShadowTaskView:
    """Bounded task projection for shadow planning."""

    task_id: str
    task_class: str = "default"
    risk_class: str = "low"
    environment: str | None = None
    route_id: str = "route.compressed"
    expanded_route_id: str = "route.expanded"
    promotion_evaluation: bool = False
    new_task_class: bool = False
    new_analyzer: bool = False
    new_route: bool = False
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _task_id(self.task_id))
        object.__setattr__(self, "task_class", _token(self.task_class, "task_class"))
        object.__setattr__(self, "risk_class", _token(self.risk_class, "risk_class"))
        if self.environment is not None:
            object.__setattr__(
                self, "environment", _token(self.environment, "environment")
            )
        object.__setattr__(self, "route_id", _token(self.route_id, "route_id"))
        object.__setattr__(
            self, "expanded_route_id", _token(self.expanded_route_id, "expanded_route_id")
        )
        object.__setattr__(
            self,
            "promotion_evaluation",
            _bool(self.promotion_evaluation, "promotion_evaluation"),
        )
        object.__setattr__(
            self, "new_task_class", _bool(self.new_task_class, "new_task_class")
        )
        object.__setattr__(
            self, "new_analyzer", _bool(self.new_analyzer, "new_analyzer")
        )
        object.__setattr__(self, "new_route", _bool(self.new_route, "new_route"))
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True)
class CompressedContextView:
    """Compressed ContextPack identity and information-value signals."""

    context_pack_cid: str
    capsule_uncertainty: bool = False
    token_savings_eligible: bool = False
    proof_cache_reuse: bool = False
    includes_private_source: bool = False
    expanded_context_pack_cid: str | None = None
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "context_pack_cid", _cid(self.context_pack_cid, "context_pack_cid")
        )
        object.__setattr__(
            self,
            "capsule_uncertainty",
            _bool(self.capsule_uncertainty, "capsule_uncertainty"),
        )
        object.__setattr__(
            self,
            "token_savings_eligible",
            _bool(self.token_savings_eligible, "token_savings_eligible"),
        )
        object.__setattr__(
            self,
            "proof_cache_reuse",
            _bool(self.proof_cache_reuse, "proof_cache_reuse"),
        )
        object.__setattr__(
            self,
            "includes_private_source",
            _bool(self.includes_private_source, "includes_private_source"),
        )
        object.__setattr__(
            self,
            "expanded_context_pack_cid",
            _optional_cid(self.expanded_context_pack_cid, "expanded_context_pack_cid"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True)
class RepositoryStateSignals:
    """Repository-state signals that affect shadow selection."""

    repository_state_cid: str
    recent_omission: bool = False
    recent_failure: bool = False
    verification_bundle_cid: str | None = None
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repository_state_cid",
            _cid(self.repository_state_cid, "repository_state_cid"),
        )
        object.__setattr__(
            self, "recent_omission", _bool(self.recent_omission, "recent_omission")
        )
        object.__setattr__(
            self, "recent_failure", _bool(self.recent_failure, "recent_failure")
        )
        object.__setattr__(
            self,
            "verification_bundle_cid",
            _optional_cid(self.verification_bundle_cid, "verification_bundle_cid"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))


def _coerce_task(task: ShadowTaskView | Mapping[str, Any] | str) -> ShadowTaskView:
    if isinstance(task, ShadowTaskView):
        return task
    if isinstance(task, str):
        return ShadowTaskView(task_id=task)
    if not isinstance(task, Mapping):
        raise SemanticGovernorShadowPlanError(
            "task must be ShadowTaskView, mapping, or task_id string"
        )
    return ShadowTaskView(
        task_id=_mapping_get(task, "task_id", "id", default="task.unknown"),
        task_class=_mapping_get(task, "task_class", "class", default="default"),
        risk_class=_mapping_get(task, "risk_class", "risk", default="low"),
        environment=_mapping_get(task, "environment", "env", "lifecycle_phase"),
        route_id=_mapping_get(
            task, "route_id", "compressed_route_id", default="route.compressed"
        ),
        expanded_route_id=_mapping_get(
            task, "expanded_route_id", default="route.expanded"
        ),
        promotion_evaluation=bool(
            _mapping_get(task, "promotion_evaluation", "promotion", default=False)
        ),
        new_task_class=bool(_mapping_get(task, "new_task_class", default=False)),
        new_analyzer=bool(_mapping_get(task, "new_analyzer", default=False)),
        new_route=bool(_mapping_get(task, "new_route", default=False)),
        notes=_mapping_get(task, "notes"),
        metadata=dict(_mapping_get(task, "metadata", default={}) or {}),
    )


def _coerce_compressed_context(
    compressed_context: CompressedContextView | Mapping[str, Any] | str,
) -> CompressedContextView:
    if isinstance(compressed_context, CompressedContextView):
        return compressed_context
    if isinstance(compressed_context, str):
        return CompressedContextView(context_pack_cid=compressed_context)
    if not isinstance(compressed_context, Mapping):
        raise SemanticGovernorShadowPlanError(
            "compressed_context must be CompressedContextView, mapping, or CID"
        )
    pack = _mapping_get(
        compressed_context,
        "context_pack_cid",
        "compressed_context_pack_cid",
        "cid",
    )
    if pack is None:
        raise SemanticGovernorShadowPlanError(
            "compressed_context requires context_pack_cid"
        )
    # Private-source hint from embedded material or explicit flag.
    includes_private = _mapping_get(
        compressed_context, "includes_private_source", default=None
    )
    if includes_private is None:
        includes_private = contains_private_source(compressed_context)
    else:
        includes_private = bool(includes_private)
    return CompressedContextView(
        context_pack_cid=pack,
        capsule_uncertainty=bool(
            _mapping_get(
                compressed_context,
                "capsule_uncertainty",
                "uncertainty",
                default=False,
            )
        ),
        token_savings_eligible=bool(
            _mapping_get(
                compressed_context,
                "token_savings_eligible",
                "token_savings",
                default=False,
            )
        ),
        proof_cache_reuse=bool(
            _mapping_get(
                compressed_context,
                "proof_cache_reuse",
                "cache_reuse",
                default=False,
            )
        ),
        includes_private_source=includes_private,
        expanded_context_pack_cid=_mapping_get(
            compressed_context,
            "expanded_context_pack_cid",
            "expanded_context_cid",
        ),
        notes=_mapping_get(compressed_context, "notes"),
        metadata=dict(_mapping_get(compressed_context, "metadata", default={}) or {}),
    )


def _coerce_repository_state(
    repository_state: RepositoryStateSignals | Mapping[str, Any] | str,
) -> RepositoryStateSignals:
    if isinstance(repository_state, RepositoryStateSignals):
        return repository_state
    if isinstance(repository_state, str):
        return RepositoryStateSignals(repository_state_cid=repository_state)
    if not isinstance(repository_state, Mapping):
        raise SemanticGovernorShadowPlanError(
            "repository_state must be RepositoryStateSignals, mapping, or CID"
        )
    repo_cid = _mapping_get(
        repository_state,
        "repository_state_cid",
        "repo_state_cid",
        "cid",
    )
    if repo_cid is None:
        raise SemanticGovernorShadowPlanError(
            "repository_state requires repository_state_cid"
        )
    return RepositoryStateSignals(
        repository_state_cid=repo_cid,
        recent_omission=bool(
            _mapping_get(
                repository_state,
                "recent_omission",
                "recent_omissions",
                default=False,
            )
        ),
        recent_failure=bool(
            _mapping_get(
                repository_state,
                "recent_failure",
                "recent_failures",
                default=False,
            )
        ),
        verification_bundle_cid=_mapping_get(
            repository_state,
            "verification_bundle_cid",
            "verification_cid",
        ),
        notes=_mapping_get(repository_state, "notes"),
        metadata=dict(_mapping_get(repository_state, "metadata", default={}) or {}),
    )


# ---------------------------------------------------------------------------
# Selection / information value
# ---------------------------------------------------------------------------


def _normalize_risk(risk_class: str) -> str:
    return risk_class.casefold().replace("-", "_")


def _is_development(
    task: ShadowTaskView,
    policy: ShadowSamplingPolicy,
) -> bool:
    if policy.lifecycle_phase == LifecyclePhase.DEVELOPMENT.value:
        return True
    env = (task.environment or "").casefold()
    return env in DEVELOPMENT_ENVIRONMENTS


def _risk_base_rate_bp(risk_class: str, policy: ShadowSamplingPolicy) -> int:
    risk = _normalize_risk(risk_class)
    if risk in {"critical", "risk_critical"}:
        return policy.critical_risk_sample_rate_bp
    if risk in HIGH_RISK_CLASSES:
        return policy.high_risk_sample_rate_bp
    if risk in MEDIUM_RISK_CLASSES:
        return policy.medium_risk_sample_rate_bp
    return policy.mature_low_risk_sample_rate_bp


def collect_selection_candidates(
    task: ShadowTaskView,
    compressed: CompressedContextView,
    repo: RepositoryStateSignals,
    policy: ShadowSamplingPolicy,
) -> list[tuple[str, int]]:
    """Return ordered (reason, rate_bp) candidates from information-value signals.

    Mandatory full-rate reasons are included when development or high risk
    applies. Random QC is always a candidate so mature low-risk work can still
    be sampled.
    """

    candidates: list[tuple[str, int]] = []

    if _is_development(task, policy):
        candidates.append(
            (
                ShadowSelectionReason.DEVELOPMENT_FULL_RATE.value,
                policy.development_sample_rate_bp,
            )
        )

    risk = _normalize_risk(task.risk_class)
    if risk in HIGH_RISK_CLASSES or risk in {"critical", "risk_critical"}:
        rate = _risk_base_rate_bp(task.risk_class, policy)
        candidates.append((ShadowSelectionReason.RISK_CLASS_MANDATORY.value, rate))

    if compressed.capsule_uncertainty:
        candidates.append(
            (
                ShadowSelectionReason.CAPSULE_UNCERTAINTY.value,
                policy.capsule_uncertainty_rate_bp,
            )
        )
    if task.new_analyzer:
        candidates.append(
            (ShadowSelectionReason.NEW_ANALYZER.value, policy.novelty_rate_bp)
        )
    if task.new_task_class:
        candidates.append(
            (ShadowSelectionReason.NEW_TASK_CLASS.value, policy.novelty_rate_bp)
        )
    if task.new_route:
        candidates.append(
            (ShadowSelectionReason.NEW_ROUTE.value, policy.novelty_rate_bp)
        )
    if compressed.token_savings_eligible:
        candidates.append(
            (
                ShadowSelectionReason.TOKEN_SAVINGS_SAMPLE.value,
                policy.token_savings_rate_bp,
            )
        )
    if compressed.proof_cache_reuse:
        candidates.append(
            (
                ShadowSelectionReason.PROOF_CACHE_REUSE.value,
                policy.proof_cache_reuse_rate_bp,
            )
        )
    if repo.recent_omission or repo.recent_failure:
        candidates.append(
            (
                ShadowSelectionReason.RECENT_OMISSION.value,
                policy.recent_omission_rate_bp,
            )
        )
    if task.promotion_evaluation:
        candidates.append(
            (
                ShadowSelectionReason.PROMOTION_EVALUATION.value,
                policy.promotion_evaluation_rate_bp,
            )
        )

    # Baseline risk-tier sample for non-mandatory classes (mature low / medium).
    # High/critical already contribute RISK_CLASS_MANDATORY above. Development
    # full-rate is separate. Mature low-risk therefore samples at the configured
    # low rate rather than 100 percent.
    if risk not in HIGH_RISK_CLASSES and risk not in {"critical", "risk_critical"}:
        base_rate = _risk_base_rate_bp(task.risk_class, policy)
        # Floor with explicit random QC so operators can raise the minimum sample.
        baseline = max(base_rate, policy.random_quality_control_rate_bp)
        candidates.append(
            (ShadowSelectionReason.RANDOM_QUALITY_CONTROL.value, baseline)
        )
    else:
        # High/critical still admit pure random QC as an additional label when hit.
        candidates.append(
            (
                ShadowSelectionReason.RANDOM_QUALITY_CONTROL.value,
                policy.random_quality_control_rate_bp,
            )
        )

    # De-duplicate keeping max rate per reason, stable sorted by reason.
    by_reason: dict[str, int] = {}
    for reason, rate in candidates:
        prev = by_reason.get(reason)
        if prev is None or rate > prev:
            by_reason[reason] = rate
    return sorted(by_reason.items(), key=lambda item: item[0])


def select_shadow_reasons(
    task: ShadowTaskView,
    compressed: CompressedContextView,
    repo: RepositoryStateSignals,
    policy: ShadowSamplingPolicy,
    *,
    roll: int | None = None,
) -> tuple[tuple[str, ...], int, int]:
    """Apply deterministic sampling to candidate reasons.

    Returns ``(selected_reasons, effective_rate_bp, roll)``.
    """

    candidates = collect_selection_candidates(task, compressed, repo, policy)
    if roll is None:
        roll = deterministic_sample_roll(
            random_seed=policy.random_seed,
            task_id=task.task_id,
            compressed_context_pack_cid=compressed.context_pack_cid,
        )
    else:
        roll = _nonneg_int(roll, "roll")
        if roll >= BASIS_POINTS_MAX:
            raise SemanticGovernorShadowPlanError(
                f"roll must be in [0, {BASIS_POINTS_MAX})"
            )

    selected: list[str] = []
    effective = 0
    for reason, rate in candidates:
        # Mandatory full-rate reasons always hit when their configured rate is 100%.
        if sample_hits(roll, rate):
            selected.append(reason)
            if rate > effective:
                effective = rate
        elif rate > effective:
            # Track highest considered rate for diagnostics even on miss.
            pass

    # If only random QC could apply and missed, selected may be empty.
    # When development/high risk rates are 100%, they always appear above.
    if not selected and candidates:
        # effective is max candidate rate for observability
        effective = max(rate for _, rate in candidates)

    return tuple(sorted(set(selected))), effective, roll


# ---------------------------------------------------------------------------
# Disclosure gate for planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DisclosurePlanGate:
    """Resolved disclosure posture for the planned expanded attempt."""

    allow_external_expanded_disclosure: bool
    disposition: str
    provider_id: str | None
    provider_locality: str | None
    reason_codes: tuple[str, ...]
    include_disclosure_skip_reason: bool


def resolve_disclosure_gate(
    *,
    policy: ShadowSamplingPolicy,
    disclosure_policy: ShadowDisclosurePolicy,
    expanded_provider_id: str | None,
    includes_private_source: bool,
    expanded_context: Any = None,
    worktree_id: str | None = "worktree-eval-plan",
) -> DisclosurePlanGate:
    """Resolve whether external expanded disclosure is allowed for this plan.

    Forbidden disclosure never yields a policy bypass: the plan is local-only
    and may record ``disclosure_forbidden_skip`` when external calls are zeroed.
    """

    # Sampling policy cannot self-elevate external disclosure without privacy.
    requested_external = bool(policy.allow_external_expanded_disclosure)

    if expanded_provider_id is None:
        # No external provider bound: always local-only expanded posture.
        return DisclosurePlanGate(
            allow_external_expanded_disclosure=False,
            disposition=DisclosureDisposition.LOCAL_ONLY.value,
            provider_id=None,
            provider_locality=ProviderLocality.LOCAL.value,
            reason_codes=("no_expanded_provider_bound", "local_only_default"),
            include_disclosure_skip_reason=False,
        )

    locality = classify_provider_locality(expanded_provider_id, disclosure_policy)

    # Public / non-private expanded context may still be external when requested
    # and provider is not unapproved for private residual — use authorize path.
    try:
        auth = authorize_shadow_disclosure(
            disclosure_policy,
            provider_id=expanded_provider_id,
            context=expanded_context,
            includes_private_source=includes_private_source,
            isolated_evaluation_worktree=True,
            worktree_id=worktree_id,
            raise_on_forbidden=False,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return DisclosurePlanGate(
            allow_external_expanded_disclosure=False,
            disposition=DisclosureDisposition.FORBIDDEN.value,
            provider_id=expanded_provider_id,
            provider_locality=locality.value,
            reason_codes=("disclosure_gate_error", type(exc).__name__.lower()),
            include_disclosure_skip_reason=True,
        )

    disposition = str(auth.disposition)
    reasons = tuple(auth.reason_codes)

    if disposition == DisclosureDisposition.ALLOWED.value and requested_external:
        # Exact privacy authorization plus sampling policy consent.
        if locality in {
            ProviderLocality.APPROVED_EXTERNAL,
            ProviderLocality.LOCAL,
            ProviderLocality.SIMULATED,
        }:
            # Local/simulated never need external flag; only approved external
            # sets allow_external_expanded_disclosure.
            allow_ext = locality is ProviderLocality.APPROVED_EXTERNAL
            return DisclosurePlanGate(
                allow_external_expanded_disclosure=allow_ext,
                disposition=disposition,
                provider_id=expanded_provider_id,
                provider_locality=locality.value,
                reason_codes=reasons + ("sampling_policy_allows_external",),
                include_disclosure_skip_reason=False,
            )

    if disposition in {
        DisclosureDisposition.LOCAL_ONLY.value,
        DisclosureDisposition.REDACTED_ONLY.value,
    }:
        return DisclosurePlanGate(
            allow_external_expanded_disclosure=False,
            disposition=disposition,
            provider_id=expanded_provider_id,
            provider_locality=locality.value,
            reason_codes=reasons + ("external_disclosure_not_admitted",),
            include_disclosure_skip_reason=False,
        )

    # FORBIDDEN or unapproved external private — never bypass.
    zero_external = policy.zero_external_calls_when_disclosure_forbidden
    include_skip = zero_external and locality in {
        ProviderLocality.UNAPPROVED_EXTERNAL,
        ProviderLocality.APPROVED_EXTERNAL,
    }
    return DisclosurePlanGate(
        allow_external_expanded_disclosure=False,
        disposition=DisclosureDisposition.FORBIDDEN.value,
        provider_id=expanded_provider_id,
        provider_locality=locality.value,
        reason_codes=reasons
        + (
            "external_disclosure_forbidden",
            "local_only_or_no_external_call",
        ),
        include_disclosure_skip_reason=include_skip,
    )


# ---------------------------------------------------------------------------
# Plan decision artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShadowPlanDecision:
    """Bounded decision from create_shadow_plan (selected or skipped)."""

    task_id: str
    disposition: ShadowPlanDisposition | str
    selected: bool
    selection_reasons: Sequence[str]
    effective_sample_rate_bp: int
    sample_roll: int
    audit_policy_cid: str
    compressed_context_pack_cid: str
    expanded_context_pack_cid: str | None
    allow_external_expanded_disclosure: bool
    disclosure_disposition: str
    disclosure_reason_codes: Sequence[str]
    plan: ShadowExecutionPlan | None = None
    plan_cid: str | None = None
    reason_codes: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface_id",
            "task_id",
            "disposition",
            "selected",
            "selection_reasons",
            "effective_sample_rate_bp",
            "sample_roll",
            "audit_policy_cid",
            "compressed_context_pack_cid",
            "expanded_context_pack_cid",
            "allow_external_expanded_disclosure",
            "disclosure_disposition",
            "disclosure_reason_codes",
            "plan",
            "plan_cid",
            "reason_codes",
            "metadata",
            "decision_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _task_id(self.task_id))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ShadowPlanDisposition, "disposition"),
        )
        object.__setattr__(self, "selected", _bool(self.selected, "selected"))
        reasons = list(self.selection_reasons)
        if not isinstance(self.selection_reasons, (list, tuple)):
            raise SemanticGovernorShadowPlanError(
                "selection_reasons must be a list"
            )
        # Validate against closed ShadowSelectionReason when non-empty.
        validated_reasons: list[str] = []
        for item in reasons:
            try:
                validated_reasons.append(
                    ShadowSelectionReason(item).value
                    if not isinstance(item, ShadowSelectionReason)
                    else item.value
                )
            except (TypeError, ValueError) as exc:
                raise SemanticGovernorShadowPlanError(
                    f"selection_reasons has unsupported value {item!r}"
                ) from exc
        object.__setattr__(
            self, "selection_reasons", tuple(sorted(set(validated_reasons)))
        )
        object.__setattr__(
            self,
            "effective_sample_rate_bp",
            _basis_points(self.effective_sample_rate_bp, "effective_sample_rate_bp"),
        )
        object.__setattr__(
            self, "sample_roll", _nonneg_int(self.sample_roll, "sample_roll")
        )
        if self.sample_roll >= BASIS_POINTS_MAX:
            raise SemanticGovernorShadowPlanError(
                f"sample_roll must be in [0, {BASIS_POINTS_MAX})"
            )
        object.__setattr__(
            self, "audit_policy_cid", _cid(self.audit_policy_cid, "audit_policy_cid")
        )
        object.__setattr__(
            self,
            "compressed_context_pack_cid",
            _cid(self.compressed_context_pack_cid, "compressed_context_pack_cid"),
        )
        object.__setattr__(
            self,
            "expanded_context_pack_cid",
            _optional_cid(self.expanded_context_pack_cid, "expanded_context_pack_cid"),
        )
        object.__setattr__(
            self,
            "allow_external_expanded_disclosure",
            _bool(
                self.allow_external_expanded_disclosure,
                "allow_external_expanded_disclosure",
            ),
        )
        object.__setattr__(
            self,
            "disclosure_disposition",
            _token(self.disclosure_disposition, "disclosure_disposition"),
        )
        object.__setattr__(
            self,
            "disclosure_reason_codes",
            _unique_sorted_tokens(
                list(self.disclosure_reason_codes),
                "disclosure_reason_codes",
                max_items=MAX_REASON_CODES,
            ),
        )
        if self.plan is not None and not isinstance(self.plan, ShadowExecutionPlan):
            raise SemanticGovernorShadowPlanError(
                "plan must be ShadowExecutionPlan or null"
            )
        if self.selected and self.plan is None:
            raise SemanticGovernorShadowPlanError(
                "selected decision requires a ShadowExecutionPlan"
            )
        if not self.selected and self.plan is not None:
            raise SemanticGovernorShadowPlanError(
                "skipped decision cannot carry a ShadowExecutionPlan"
            )
        object.__setattr__(
            self,
            "plan_cid",
            _optional_cid(self.plan_cid, "plan_cid")
            if self.plan_cid is not None
            else (self.plan.plan_cid if self.plan is not None else None),
        )
        if self.plan is not None and self.plan_cid != self.plan.plan_cid:
            raise SemanticGovernorShadowPlanError(
                "plan_cid must equal plan.plan_cid"
            )
        # Invariant: disclosure skip reason cannot allow external disclosure.
        if (
            ShadowSelectionReason.DISCLOSURE_FORBIDDEN_SKIP.value
            in self.selection_reasons
            and self.allow_external_expanded_disclosure
        ):
            raise SemanticGovernorShadowPlanError(
                "disclosure_forbidden_skip cannot allow external expanded disclosure"
            )
        object.__setattr__(
            self,
            "reason_codes",
            _unique_sorted_tokens(
                list(self.reason_codes),
                "reason_codes",
                max_items=MAX_REASON_CODES,
            ),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": SHADOW_PLAN_DECISION_SCHEMA,
            "interface_id": SHADOW_PLAN_DECISION_INTERFACE,
            "task_id": self.task_id,
            "disposition": self.disposition,
            "selected": self.selected,
            "selection_reasons": list(self.selection_reasons),
            "effective_sample_rate_bp": self.effective_sample_rate_bp,
            "sample_roll": self.sample_roll,
            "audit_policy_cid": self.audit_policy_cid,
            "compressed_context_pack_cid": self.compressed_context_pack_cid,
            "expanded_context_pack_cid": self.expanded_context_pack_cid,
            "allow_external_expanded_disclosure": (
                self.allow_external_expanded_disclosure
            ),
            "disclosure_disposition": self.disclosure_disposition,
            "disclosure_reason_codes": list(self.disclosure_reason_codes),
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "plan_cid": self.plan_cid,
            "reason_codes": list(self.reason_codes),
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def decision_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["decision_cid"] = self.decision_cid
        return payload


# ---------------------------------------------------------------------------
# Plan construction helpers
# ---------------------------------------------------------------------------


def _default_expanded_context_pack_cid(compressed_cid: str) -> str:
    """Derive a deterministic expanded-pack placeholder CID from compressed.

    Callers should supply a real expanded ContextPack CID when available. The
    derived CID is content-addressed metadata only (not raw source).
    """

    return cid_for_structured(
        {
            "kind": "expanded_context_pack_ref",
            "from_compressed_context_pack_cid": compressed_cid,
            "role": "expanded",
            "oracle_candidate_only": True,
        }
    )


def _build_plan_header(
    *,
    repository_state_cid: str,
    compressed_context_pack_cid: str,
    verification_bundle_cid: str,
    audit_policy_cid: str,
    task_id: str,
) -> GovernorArtifactHeader:
    generator = GeneratorIdentity(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        interface_id=CREATE_SHADOW_PLAN_INTERFACE,
    )
    provenance = ArtifactProvenance(
        producer_id="semantic_governor",
        producer_version="1",
        execution_mode=ExecutionMode.LIVE,
        authority_source=AuthoritySource.DETERMINISTIC,
        input_cids=(
            compressed_context_pack_cid,
            repository_state_cid,
            audit_policy_cid,
        ),
        tool_ids=("shadow_plan.v1",),
        policy_cid=audit_policy_cid,
        notes=None,
    )
    return GovernorArtifactHeader(
        artifact_kind="shadow_execution_plan",
        repository_state_cid=repository_state_cid,
        context_pack_cid=compressed_context_pack_cid,
        verification_bundle_cid=verification_bundle_cid,
        generator=generator,
        provenance=provenance,
        terminal_status=GovernorTerminalStatus.COMPLETE,
        assumptions=(
            GovernorAssumption(
                assumption_id="isolated_worktree",
                kind=AssumptionKind.ENVIRONMENT,
                statement=(
                    "Paired shadow runs use disposable evaluation worktrees"
                ),
                supporting_cids=(audit_policy_cid,),
            ),
            GovernorAssumption(
                assumption_id="expanded_oracle_only",
                kind=AssumptionKind.VERIFICATION,
                statement=(
                    "Expanded shadow output is oracle/candidate only and never "
                    "silently replaces the accepted patch"
                ),
                supporting_cids=(audit_policy_cid,),
            ),
        ),
        metadata={"task_id": task_id, "evidence": SCG_SHADOW_PLAN_EVIDENCE},
    )


def _admit_resource_gates(policy: ShadowSamplingPolicy) -> None:
    if policy.max_wall_time_ms == 0:
        raise ResourceGateError("max_wall_time_ms resource gate is zero")
    if policy.max_expansion_token_budget == 0:
        raise ResourceGateError("max_expansion_token_budget resource gate is zero")
    # Model spend may be zero for pure local/static expanded evaluation.


# ---------------------------------------------------------------------------
# Public API: create_shadow_plan
# ---------------------------------------------------------------------------


def create_shadow_plan(
    task: ShadowTaskView | Mapping[str, Any] | str,
    compressed_context: CompressedContextView | Mapping[str, Any] | str,
    repository_state: RepositoryStateSignals | Mapping[str, Any] | str,
    audit_policy: ShadowSamplingPolicy | Mapping[str, Any] | None = None,
    *,
    disclosure_policy: ShadowDisclosurePolicy | Mapping[str, Any] | None = None,
    expanded_provider_id: str | None = None,
    expanded_context: Any = None,
    expanded_context_pack_cid: str | None = None,
    worktree_id: str | None = "worktree-eval-plan",
    require_selected: bool = False,
    sample_roll: int | None = None,
) -> ShadowPlanDecision:
    """Create a risk- and information-value-aware shadow execution plan.

    Parameters mirror the release API::

        create_shadow_plan(task, compressed_context, repository_state, audit_policy)

    Optional keyword arguments bind privacy, expanded pack identity, and
    deterministic sampling overrides. Returns a :class:`ShadowPlanDecision`
    that either carries a sealed :class:`ShadowExecutionPlan` or records a
    deterministic skip.

    Disclosure never bypasses privacy policy: when expanded external disclosure
    is forbidden, the plan is local-only (and may record
    ``disclosure_forbidden_skip``) with ``allow_external_expanded_disclosure``
    forced false.
    """

    task_view = _coerce_task(task)
    compressed = _coerce_compressed_context(compressed_context)
    repo = _coerce_repository_state(repository_state)
    policy = _coerce_sampling_policy(audit_policy)

    if disclosure_policy is None:
        disc_policy = default_shadow_disclosure_policy()
    elif isinstance(disclosure_policy, ShadowDisclosurePolicy):
        disc_policy = disclosure_policy
    elif isinstance(disclosure_policy, Mapping):
        if "schema" in disclosure_policy and "interface_id" in disclosure_policy:
            disc_policy = ShadowDisclosurePolicy.from_dict(disclosure_policy)
        else:
            disc_policy = ShadowDisclosurePolicy(**dict(disclosure_policy))
    else:
        raise SemanticGovernorShadowPlanError(
            "disclosure_policy must be ShadowDisclosurePolicy or mapping"
        )

    _admit_resource_gates(policy)

    expanded_cid = expanded_context_pack_cid or compressed.expanded_context_pack_cid
    if expanded_cid is None:
        expanded_cid = _default_expanded_context_pack_cid(compressed.context_pack_cid)
    else:
        expanded_cid = _cid(expanded_cid, "expanded_context_pack_cid")

    includes_private = bool(compressed.includes_private_source)
    if expanded_context is not None and contains_private_source(expanded_context):
        includes_private = True
    if expanded_context is not None:
        source_class = classify_source_privacy(expanded_context)
        if source_class in {
            SourcePrivacyClass.PRIVATE,
            SourcePrivacyClass.RAW_PRIVATE,
        }:
            includes_private = True

    gate = resolve_disclosure_gate(
        policy=policy,
        disclosure_policy=disc_policy,
        expanded_provider_id=expanded_provider_id,
        includes_private_source=includes_private,
        expanded_context=expanded_context,
        worktree_id=worktree_id,
    )

    reasons, effective_rate, roll = select_shadow_reasons(
        task_view,
        compressed,
        repo,
        policy,
        roll=sample_roll,
    )

    # When disclosure forbids external expanded and zero-external is set,
    # still allow selection of the audit for local-only paired evaluation.
    # Attach disclosure skip reason without inventing selection if skipped.
    final_reasons = list(reasons)
    if gate.include_disclosure_skip_reason and reasons:
        final_reasons.append(
            ShadowSelectionReason.DISCLOSURE_FORBIDDEN_SKIP.value
        )

    selected = bool(final_reasons) and any(
        r != ShadowSelectionReason.DISCLOSURE_FORBIDDEN_SKIP.value
        for r in final_reasons
    )
    # If only disclosure skip would be present without other hits, not selected.
    if not reasons:
        selected = False
        final_reasons = []

    disposition: ShadowPlanDisposition
    decision_reasons: list[str] = []
    if not selected:
        disposition = ShadowPlanDisposition.SKIPPED
        decision_reasons.append("sample_miss")
        if not reasons:
            decision_reasons.append("no_selection_reasons")
    elif gate.include_disclosure_skip_reason:
        disposition = ShadowPlanDisposition.DISCLOSURE_EXTERNAL_SKIPPED
        decision_reasons.append("external_expanded_call_skipped")
    elif not gate.allow_external_expanded_disclosure:
        disposition = ShadowPlanDisposition.DISCLOSURE_LOCAL_ONLY
        decision_reasons.append("local_only_expanded")
    else:
        disposition = ShadowPlanDisposition.SELECTED
        decision_reasons.append("selected")

    plan: ShadowExecutionPlan | None = None
    plan_cid: str | None = None

    if selected:
        verification_cid = repo.verification_bundle_cid
        if verification_cid is None:
            verification_cid = cid_for_structured(
                {
                    "kind": "shadow_plan_verification_placeholder",
                    "task_id": task_view.task_id,
                    "repository_state_cid": repo.repository_state_cid,
                }
            )
        header = _build_plan_header(
            repository_state_cid=repo.repository_state_cid,
            compressed_context_pack_cid=compressed.context_pack_cid,
            verification_bundle_cid=verification_cid,
            audit_policy_cid=policy.policy_cid,
            task_id=task_view.task_id,
        )
        # Never allow external disclosure when gate forbids it (no bypass).
        allow_external = bool(gate.allow_external_expanded_disclosure)
        if (
            ShadowSelectionReason.DISCLOSURE_FORBIDDEN_SKIP.value in final_reasons
            and allow_external
        ):
            allow_external = False

        plan = ShadowExecutionPlan(
            header=header,
            task_id=task_view.task_id,
            audit_policy_cid=policy.policy_cid,
            compressed_context_pack_cid=compressed.context_pack_cid,
            expanded_context_pack_cid=expanded_cid,
            compressed_route_id=task_view.route_id,
            expanded_route_id=task_view.expanded_route_id,
            selection_reasons=tuple(sorted(set(final_reasons))),
            max_wall_time_ms=policy.max_wall_time_ms,
            max_model_spend_micros=policy.max_model_spend_micros,
            max_expansion_token_budget=policy.max_expansion_token_budget,
            isolated_evaluation_worktree_required=True,
            expanded_is_oracle_candidate_only=True,
            allow_external_expanded_disclosure=allow_external,
            metadata={
                "evidence": SCG_SHADOW_PLAN_EVIDENCE,
                "sample_roll": roll,
                "effective_sample_rate_bp": effective_rate,
                "disclosure_disposition": gate.disposition,
                "lifecycle_phase": policy.lifecycle_phase,
                "risk_class": task_view.risk_class,
            },
        )
        plan_cid = verify_plan_identity(plan)

    decision = ShadowPlanDecision(
        task_id=task_view.task_id,
        disposition=disposition,
        selected=selected,
        selection_reasons=tuple(sorted(set(final_reasons))),
        effective_sample_rate_bp=effective_rate
        if selected
        else (
            max((r for _, r in collect_selection_candidates(
                task_view, compressed, repo, policy
            )), default=0)
        ),
        sample_roll=roll,
        audit_policy_cid=policy.policy_cid,
        compressed_context_pack_cid=compressed.context_pack_cid,
        expanded_context_pack_cid=expanded_cid if selected else expanded_cid,
        allow_external_expanded_disclosure=(
            plan.allow_external_expanded_disclosure if plan is not None else False
        ),
        disclosure_disposition=gate.disposition,
        disclosure_reason_codes=gate.reason_codes,
        plan=plan,
        plan_cid=plan_cid,
        reason_codes=tuple(decision_reasons),
        metadata={
            "risk_class": task_view.risk_class,
            "lifecycle_phase": policy.lifecycle_phase,
            "provider_locality": gate.provider_locality or "none",
        },
    )

    if require_selected and not decision.selected:
        raise ShadowPlanNotSelected(
            f"shadow plan not selected for task {task_view.task_id!r} "
            f"(roll={roll}, effective_rate_bp={decision.effective_sample_rate_bp})"
        )

    return decision


def plan_allows_external_expanded_call(decision: ShadowPlanDecision) -> bool:
    """True only when a selected plan admits external expanded disclosure."""

    if not decision.selected or decision.plan is None:
        return False
    return bool(decision.plan.allow_external_expanded_disclosure)


def assert_no_disclosure_bypass(decision: ShadowPlanDecision) -> None:
    """Fail closed if a decision would bypass forbidden disclosure policy."""

    if (
        decision.disclosure_disposition == DisclosureDisposition.FORBIDDEN.value
        and decision.allow_external_expanded_disclosure
    ):
        raise DisclosureForbiddenError(
            "shadow plan must not allow external expanded disclosure when "
            "disclosure disposition is forbidden"
        )
    if (
        ShadowSelectionReason.DISCLOSURE_FORBIDDEN_SKIP.value
        in decision.selection_reasons
        and decision.allow_external_expanded_disclosure
    ):
        raise DisclosureForbiddenError(
            "disclosure_forbidden_skip cannot allow external expanded disclosure"
        )
    if decision.plan is not None:
        if (
            decision.disclosure_disposition == DisclosureDisposition.FORBIDDEN.value
            and decision.plan.allow_external_expanded_disclosure
        ):
            raise DisclosureForbiddenError(
                "plan.allow_external_expanded_disclosure bypasses forbidden disclosure"
            )


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = (
    "BASIS_POINTS_MAX",
    "CREATE_SHADOW_PLAN_INTERFACE",
    "CompressedContextView",
    "DEFAULT_MATURE_LOW_RISK_RATE_BP",
    "DisclosurePlanGate",
    "LifecyclePhase",
    "RepositoryStateSignals",
    "ResourceGateError",
    "SCG_SHADOW_PLAN_EVIDENCE",
    "SHADOW_PLAN_DECISION_INTERFACE",
    "SHADOW_PLAN_DECISION_SCHEMA",
    "SHADOW_SAMPLING_POLICY_INTERFACE",
    "SHADOW_SAMPLING_POLICY_SCHEMA",
    "SHADOW_EXECUTION_PLAN_INTERFACE",
    "SemanticGovernorShadowPlanError",
    "ShadowPlanDecision",
    "ShadowPlanDisposition",
    "ShadowPlanNotSelected",
    "ShadowSamplingPolicy",
    "ShadowTaskView",
    "assert_no_disclosure_bypass",
    "collect_selection_candidates",
    "create_shadow_plan",
    "default_shadow_sampling_policy",
    "deterministic_sample_roll",
    "development_shadow_sampling_policy",
    "plan_allows_external_expanded_call",
    "resolve_disclosure_gate",
    "sample_hits",
    "select_shadow_reasons",
)
