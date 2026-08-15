"""Active audit scheduling by expected information value (SCG-031).

Ranks candidate audits using a bounded, deterministic expected-information-value
(EIV) priority, admits work under resource capacity, and enforces configured
shadow sample rates plus privacy zero-call policy.

Normative fail-closed invariants:

* **Shadow rates** — admissions honor the versioned
  :class:`~shadow_plan.ShadowSamplingPolicy` rates. Development and
  high/critical risk may admit at full rate when configured; mature low-risk
  samples and is never forced to 100 percent.
* **Privacy zero-call** — when disclosure policy forbids external expanded
  calls, admitted audits are local-only with
  ``allow_external_expanded_disclosure=False`` (and may record
  ``disclosure_forbidden_skip``). Never a policy bypass.
* **Bounded queue** — the pending queue cannot grow without bound; overflow
  is rejected with a typed disposition.
* **Anti-starvation** — aged high-value candidates receive a starvation boost
  so mature repetitive low-risk work cannot monopolize audit spend forever.
* **Resource admission** — per-tick slot and spend budgets gate admissions.
* Importing this module performs no I/O, opens no sockets, and never invokes a
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

from ipfs_accelerate_py.agent_supervisor.semantic_governor.contracts import (
    ShadowSelectionReason,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.privacy import (
    DisclosureDisposition,
    ShadowDisclosurePolicy,
    default_shadow_disclosure_policy,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.shadow_plan import (
    BASIS_POINTS_MAX,
    CompressedContextView,
    LifecyclePhase,
    RepositoryStateSignals,
    ShadowPlanDecision,
    ShadowPlanDisposition,
    ShadowSamplingPolicy,
    ShadowTaskView,
    assert_no_disclosure_bypass,
    create_shadow_plan,
    default_shadow_sampling_policy,
    HIGH_RISK_CLASSES,
    LOW_RISK_CLASSES,
    MEDIUM_RISK_CLASSES,
)

# ---------------------------------------------------------------------------
# Evidence / interface / schema constants
# ---------------------------------------------------------------------------

SCG_ACTIVE_SCHEDULER_EVIDENCE: Final[str] = "scg/active-scheduler@1"

SCHEDULE_AUDITS_INTERFACE: Final[str] = "schedule_audits@1"
ACTIVE_AUDIT_SCHEDULER_INTERFACE: Final[str] = "ActiveAuditScheduler@1"
AUDIT_PRIORITY_INTERFACE: Final[str] = "AuditPriority@1"
AUDIT_CANDIDATE_INTERFACE: Final[str] = "AuditCandidate@1"
AUDIT_SCHEDULER_POLICY_INTERFACE: Final[str] = "AuditSchedulerPolicy@1"
AUDIT_SCHEDULE_RESULT_INTERFACE: Final[str] = "AuditScheduleResult@1"
AUDIT_ADMISSION_INTERFACE: Final[str] = "AuditAdmission@1"

AUDIT_PRIORITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/audit-priority@1"
)
AUDIT_CANDIDATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/audit-candidate@1"
)
AUDIT_SCHEDULER_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "audit-scheduler-policy@1"
)
AUDIT_SCHEDULE_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "audit-schedule-result@1"
)
AUDIT_ADMISSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/audit-admission@1"
)

GENERATOR_ID: Final[str] = "semantic_governor_active_scheduler"
GENERATOR_VERSION: Final[str] = "1.0.0"
PRODUCER_ID: Final[str] = "semantic_governor"
PRODUCER_VERSION: Final[str] = "1"
TOOL_ID: Final[str] = "active_audit_scheduler.v1"

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_REASON_CODES: Final[int] = 64
MAX_METADATA_KEYS: Final[int] = 64
MAX_CANDIDATES_PER_SCHEDULE: Final[int] = 4_096
MAX_QUEUE_DEPTH_HARD: Final[int] = 16_384
MAX_ADMISSIONS_HARD: Final[int] = 1_024
MAX_SPEND_MICROS_HARD: Final[int] = 10**15
MAX_CONE_SIZE: Final[int] = 1_000_000
MAX_SAMPLE_DEFICIT: Final[int] = BASIS_POINTS_MAX
MAX_AGE_MS: Final[int] = 86_400_000 * 30  # 30 days

# Default scheduler policy (resource / fairness bounds).
DEFAULT_MAX_QUEUE_DEPTH: Final[int] = 256
DEFAULT_MAX_ADMISSIONS_PER_TICK: Final[int] = 8
DEFAULT_MAX_SPEND_MICROS_PER_TICK: Final[int] = 50_000_000
DEFAULT_MAX_MATURE_LOW_RISK_FRACTION_BP: Final[int] = 2_500  # 25% of admissions
DEFAULT_STARVATION_AGE_MS: Final[int] = 60_000
DEFAULT_STARVATION_BOOST_BP: Final[int] = 2_500
DEFAULT_ESTIMATED_COST_MICROS: Final[int] = 1_000_000

# Factor weights for expected information value (sum to BASIS_POINTS_MAX).
WEIGHT_RISK_BP: Final[int] = 1_800
WEIGHT_UNCERTAINTY_BP: Final[int] = 1_400
WEIGHT_SAVINGS_BP: Final[int] = 800
WEIGHT_RULE_EXPOSURE_BP: Final[int] = 900
WEIGHT_SAMPLE_DEFICIT_BP: Final[int] = 1_200
WEIGHT_FAILURES_BP: Final[int] = 1_300
WEIGHT_COST_ESCALATION_BP: Final[int] = 700
WEIGHT_CONE_SIZE_BP: Final[int] = 600
WEIGHT_DYNAMIC_FEATURES_BP: Final[int] = 600
WEIGHT_POLICY_IMPORTANCE_BP: Final[int] = 700

_WEIGHT_SUM: Final[int] = (
    WEIGHT_RISK_BP
    + WEIGHT_UNCERTAINTY_BP
    + WEIGHT_SAVINGS_BP
    + WEIGHT_RULE_EXPOSURE_BP
    + WEIGHT_SAMPLE_DEFICIT_BP
    + WEIGHT_FAILURES_BP
    + WEIGHT_COST_ESCALATION_BP
    + WEIGHT_CONE_SIZE_BP
    + WEIGHT_DYNAMIC_FEATURES_BP
    + WEIGHT_POLICY_IMPORTANCE_BP
)
assert _WEIGHT_SUM == BASIS_POINTS_MAX, "EIV weights must sum to 10000"

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")
_TASK_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z][A-Za-z0-9_.:/+-]{0,127}$"
)


# ---------------------------------------------------------------------------
# Errors / closed enums
# ---------------------------------------------------------------------------


class SemanticGovernorSchedulerError(SemanticGovernorBaseError):
    """Raised when active audit scheduling inputs or policy fail closed."""


class AuditQueueOverflowError(SemanticGovernorSchedulerError):
    """Raised when the pending audit queue rejects further growth."""


class AuditResourceDeniedError(SemanticGovernorSchedulerError):
    """Raised when resource capacity refuses an admission (optional hard mode)."""


class AuditAdmissionDisposition(str, Enum):
    """Closed outcome for one candidate during a schedule tick."""

    ADMITTED = "admitted"
    DEFERRED = "deferred"
    SAMPLE_SKIPPED = "sample_skipped"
    PRIVACY_EXTERNAL_SKIPPED = "privacy_external_skipped"
    RESOURCE_DENIED = "resource_denied"
    SPEND_EXHAUSTED = "spend_exhausted"
    ANTI_MONOPOLY_DEFERRED = "anti_monopoly_deferred"
    QUEUE_OVERFLOW = "queue_overflow"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"


class FairnessClass(str, Enum):
    """Closed fairness classes used for anti-monopoly accounting."""

    DEVELOPMENT = "development"
    HIGH_RISK = "high_risk"
    MEDIUM_RISK = "medium_risk"
    MATURE_LOW_RISK = "mature_low_risk"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _normalize_token(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        raise SemanticGovernorSchedulerError(f"{name} must be a nonempty string")
    text = _normalize_token(value)
    if text != value.strip() and text != value:
        raise SemanticGovernorSchedulerError(f"{name} must be trimmed NFC text")
    if unicodedata.normalize("NFC", text) != text:
        raise SemanticGovernorSchedulerError(f"{name} must be trimmed NFC text")
    if len(text) > MAX_TEXT_CHARS or any(not char.isprintable() for char in text):
        raise SemanticGovernorSchedulerError(f"{name} contains invalid text")
    return text


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if _TOKEN_RE.fullmatch(text) is None:
        raise SemanticGovernorSchedulerError(
            f"{name} must be a lowercase token matching {_TOKEN_RE.pattern}"
        )
    return text


def _task_id(value: Any, name: str = "task_id") -> str:
    text = _text(value, name)
    if _TASK_ID_RE.fullmatch(text) is None:
        raise SemanticGovernorSchedulerError(
            f"{name} must match {_TASK_ID_RE.pattern}"
        )
    return text


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise SemanticGovernorSchedulerError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise SemanticGovernorSchedulerError(f"{name} must be a bool")
    return value


def _nonneg_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise SemanticGovernorSchedulerError(f"{name} must be a non-negative int")
    if maximum is not None and value > maximum:
        raise SemanticGovernorSchedulerError(
            f"{name} must be <= {maximum}"
        )
    return value


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    value = _nonneg_int(value, name, maximum=maximum)
    if value < 1:
        raise SemanticGovernorSchedulerError(f"{name} must be a positive int")
    return value


def _basis_points(value: Any, name: str) -> int:
    value = _nonneg_int(value, name, maximum=BASIS_POINTS_MAX)
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    if type(value) is not str:
        raise SemanticGovernorSchedulerError(
            f"{name} must be a {enum_type.__name__} value"
        )
    text = _normalize_token(value)
    try:
        return enum_type(text).value
    except ValueError as exc:
        allowed = ", ".join(sorted(item.value for item in enum_type))
        raise SemanticGovernorSchedulerError(
            f"{name} must be one of: {allowed}"
        ) from exc


def _freeze_structured(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(k): _freeze_structured(v) for k, v in sorted(value.items(), key=lambda i: str(i[0]))}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_structured(item) for item in value)
    return value


def _thaw_structured(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _thaw_structured(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_structured(item) for item in value]
    return value


def _require_structured(value: Any, name: str) -> Any:
    thawed = _thaw_structured(value)
    try:
        validate_structured_value(thawed)
    except Exception as exc:
        raise SemanticGovernorSchedulerError(
            f"{name} must be DAG-JSON structured"
        ) from exc
    reject_private_and_model_authority(thawed, path=name)
    return thawed


def _mapping(value: Any, name: str, *, frozen: bool = True) -> Mapping[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise SemanticGovernorSchedulerError(f"{name} must be a mapping")
    if len(value) > MAX_METADATA_KEYS:
        raise SemanticGovernorSchedulerError(
            f"{name} exceeds max keys {MAX_METADATA_KEYS}"
        )
    cleaned = _require_structured(dict(value), name)
    if not isinstance(cleaned, dict):
        raise SemanticGovernorSchedulerError(f"{name} must be a mapping")
    return MappingProxyType(cleaned) if frozen else cleaned


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise SemanticGovernorSchedulerError(f"{name} must be a mapping")
    unknown = sorted(set(data) - fields)
    if unknown:
        raise SemanticGovernorSchedulerError(
            f"{name} has unknown fields: {', '.join(unknown)}"
        )
    return dict(data)


def _unique_sorted_tokens(
    values: Iterable[Any], name: str, *, max_items: int = MAX_REASON_CODES
) -> tuple[str, ...]:
    items: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = _token(raw, name)
        if token not in seen:
            seen.add(token)
            items.append(token)
    if len(items) > max_items:
        raise SemanticGovernorSchedulerError(
            f"{name} exceeds max items {max_items}"
        )
    return tuple(sorted(items))


def _mapping_get(data: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def _stable_cid(label: str) -> str:
    return cid_for_structured({"active_scheduler_seed": label})


def _header(
    artifact_kind: str,
    *,
    input_cids: Sequence[str] = (),
    interface_id: str = SCHEDULE_AUDITS_INTERFACE,
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
                    assumption_id="active_scheduler_bounded_eiv",
                    kind=AssumptionKind.ROUTE,
                    statement=(
                        "Active audit scheduling ranks expected information "
                        "value under bounded resource admission; mature "
                        "low-risk work cannot monopolize audit spend"
                    ),
                    supporting_cids=(),
                ),
            ),
            metadata=dict(metadata or {"track": "audit-scheduler"}),
        )
    except SemanticGovernorBaseError as exc:
        raise SemanticGovernorSchedulerError(str(exc)) from exc


def _scale_weight(score_bp: int, weight_bp: int) -> int:
    """Scale a [0, 10000] factor by a weight without floats."""

    return (int(score_bp) * int(weight_bp)) // BASIS_POINTS_MAX


def _clamp_bp(value: int) -> int:
    if value < 0:
        return 0
    if value > BASIS_POINTS_MAX:
        return BASIS_POINTS_MAX
    return value


# ---------------------------------------------------------------------------
# Fairness helpers
# ---------------------------------------------------------------------------


def _normalize_risk(risk_class: str) -> str:
    return risk_class.strip().lower()


def classify_fairness_class(
    *,
    risk_class: str,
    environment: str | None,
    lifecycle_phase: str | None = None,
) -> str:
    """Map risk/environment into a closed fairness class.

    Fairness is a per-candidate property. Global sampling-policy lifecycle
    phase alone does not reclassify every task as development — only the
    candidate's own ``environment`` (or an explicit development lifecycle
    bound to that candidate) does.
    """

    env = (environment or "").strip().lower()
    # Candidate-local development markers only (not global audit-policy phase).
    if env in {"development", "dev", "local_dev", "ci_development"}:
        return FairnessClass.DEVELOPMENT.value
    phase = (lifecycle_phase or "").strip().lower()
    if env == "" and phase in {LifecyclePhase.DEVELOPMENT.value, "development"}:
        # Explicit candidate-bound development lifecycle only when environment
        # is unset; callers must not pass the global sampling policy phase here.
        return FairnessClass.DEVELOPMENT.value

    risk = _normalize_risk(risk_class)
    if risk in HIGH_RISK_CLASSES or risk in {"critical", "risk_critical"}:
        return FairnessClass.HIGH_RISK.value
    if risk in MEDIUM_RISK_CLASSES:
        return FairnessClass.MEDIUM_RISK.value
    if risk in LOW_RISK_CLASSES:
        return FairnessClass.MATURE_LOW_RISK.value
    return FairnessClass.OTHER.value


def is_mature_low_risk_fairness(fairness_class: str) -> bool:
    return fairness_class == FairnessClass.MATURE_LOW_RISK.value


# ---------------------------------------------------------------------------
# AuditSchedulerPolicy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuditSchedulerPolicy:
    """Bounded resource, queue, and fairness policy for active scheduling."""

    policy_id: str = "audit-scheduler-default"
    max_queue_depth: int = DEFAULT_MAX_QUEUE_DEPTH
    max_admissions_per_tick: int = DEFAULT_MAX_ADMISSIONS_PER_TICK
    max_spend_micros_per_tick: int = DEFAULT_MAX_SPEND_MICROS_PER_TICK
    max_mature_low_risk_fraction_bp: int = DEFAULT_MAX_MATURE_LOW_RISK_FRACTION_BP
    starvation_age_ms: int = DEFAULT_STARVATION_AGE_MS
    starvation_boost_bp: int = DEFAULT_STARVATION_BOOST_BP
    default_estimated_cost_micros: int = DEFAULT_ESTIMATED_COST_MICROS
    # When True, overflow enqueue raises; when False, records QUEUE_OVERFLOW.
    raise_on_queue_overflow: bool = False
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface_id",
            "policy_id",
            "max_queue_depth",
            "max_admissions_per_tick",
            "max_spend_micros_per_tick",
            "max_mature_low_risk_fraction_bp",
            "starvation_age_ms",
            "starvation_boost_bp",
            "default_estimated_cost_micros",
            "raise_on_queue_overflow",
            "notes",
            "metadata",
            "policy_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _token(self.policy_id, "policy_id"))
        object.__setattr__(
            self,
            "max_queue_depth",
            _positive_int(
                self.max_queue_depth, "max_queue_depth", maximum=MAX_QUEUE_DEPTH_HARD
            ),
        )
        object.__setattr__(
            self,
            "max_admissions_per_tick",
            _positive_int(
                self.max_admissions_per_tick,
                "max_admissions_per_tick",
                maximum=MAX_ADMISSIONS_HARD,
            ),
        )
        object.__setattr__(
            self,
            "max_spend_micros_per_tick",
            _nonneg_int(
                self.max_spend_micros_per_tick,
                "max_spend_micros_per_tick",
                maximum=MAX_SPEND_MICROS_HARD,
            ),
        )
        object.__setattr__(
            self,
            "max_mature_low_risk_fraction_bp",
            _basis_points(
                self.max_mature_low_risk_fraction_bp,
                "max_mature_low_risk_fraction_bp",
            ),
        )
        object.__setattr__(
            self,
            "starvation_age_ms",
            _nonneg_int(self.starvation_age_ms, "starvation_age_ms", maximum=MAX_AGE_MS),
        )
        object.__setattr__(
            self,
            "starvation_boost_bp",
            _basis_points(self.starvation_boost_bp, "starvation_boost_bp"),
        )
        object.__setattr__(
            self,
            "default_estimated_cost_micros",
            _nonneg_int(
                self.default_estimated_cost_micros,
                "default_estimated_cost_micros",
                maximum=MAX_SPEND_MICROS_HARD,
            ),
        )
        object.__setattr__(
            self,
            "raise_on_queue_overflow",
            _bool(self.raise_on_queue_overflow, "raise_on_queue_overflow"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": AUDIT_SCHEDULER_POLICY_SCHEMA,
            "interface_id": AUDIT_SCHEDULER_POLICY_INTERFACE,
            "policy_id": self.policy_id,
            "max_queue_depth": self.max_queue_depth,
            "max_admissions_per_tick": self.max_admissions_per_tick,
            "max_spend_micros_per_tick": self.max_spend_micros_per_tick,
            "max_mature_low_risk_fraction_bp": self.max_mature_low_risk_fraction_bp,
            "starvation_age_ms": self.starvation_age_ms,
            "starvation_boost_bp": self.starvation_boost_bp,
            "default_estimated_cost_micros": self.default_estimated_cost_micros,
            "raise_on_queue_overflow": self.raise_on_queue_overflow,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditSchedulerPolicy":
        payload = _closed(data, cls._FIELDS, "AuditSchedulerPolicy")
        claimed = payload.pop("policy_cid", None)
        result = cls(
            policy_id=payload.get("policy_id", "audit-scheduler-default"),
            max_queue_depth=payload.get("max_queue_depth", DEFAULT_MAX_QUEUE_DEPTH),
            max_admissions_per_tick=payload.get(
                "max_admissions_per_tick", DEFAULT_MAX_ADMISSIONS_PER_TICK
            ),
            max_spend_micros_per_tick=payload.get(
                "max_spend_micros_per_tick", DEFAULT_MAX_SPEND_MICROS_PER_TICK
            ),
            max_mature_low_risk_fraction_bp=payload.get(
                "max_mature_low_risk_fraction_bp",
                DEFAULT_MAX_MATURE_LOW_RISK_FRACTION_BP,
            ),
            starvation_age_ms=payload.get(
                "starvation_age_ms", DEFAULT_STARVATION_AGE_MS
            ),
            starvation_boost_bp=payload.get(
                "starvation_boost_bp", DEFAULT_STARVATION_BOOST_BP
            ),
            default_estimated_cost_micros=payload.get(
                "default_estimated_cost_micros", DEFAULT_ESTIMATED_COST_MICROS
            ),
            raise_on_queue_overflow=payload.get("raise_on_queue_overflow", False),
            notes=payload.get("notes"),
            metadata=payload.get("metadata") or {},
        )
        if claimed is not None and claimed != result.policy_cid:
            raise SemanticGovernorSchedulerError(
                "AuditSchedulerPolicy policy_cid does not verify"
            )
        return result


def default_audit_scheduler_policy(**overrides: Any) -> AuditSchedulerPolicy:
    """Return the production-default bounded scheduler policy."""

    return AuditSchedulerPolicy(**overrides)


# ---------------------------------------------------------------------------
# AuditCandidate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuditCandidate:
    """One candidate audit with expected-information-value signals.

    Factor signals are either basis-point intensities (0–10000) or boolean /
    count fields that are mapped into scores by :func:`compute_audit_priority`.
    """

    task_id: str
    task_class: str = "default"
    risk_class: str = "low"
    environment: str | None = None
    route_id: str = "route.compressed"
    expanded_route_id: str = "route.expanded"
    context_pack_cid: str | None = None
    repository_state_cid: str | None = None
    verification_bundle_cid: str | None = None
    # Information-value signals
    capsule_uncertainty: bool = False
    uncertainty_score_bp: int = 0
    token_savings_eligible: bool = False
    savings_score_bp: int = 0
    rule_exposure: bool = False
    rule_exposure_score_bp: int = 0
    sample_deficit_bp: int = 0
    recent_omission: bool = False
    recent_failure: bool = False
    failure_score_bp: int = 0
    cost_escalation_pressure_bp: int = 0
    cone_size: int = 0
    dynamic_features: bool = False
    dynamic_features_score_bp: int = 0
    policy_importance_bp: int = 0
    promotion_evaluation: bool = False
    new_task_class: bool = False
    new_analyzer: bool = False
    new_route: bool = False
    includes_private_source: bool = False
    queue_age_ms: int = 0
    estimated_cost_micros: int | None = None
    expanded_provider_id: str | None = None
    # Transient planning-only context (not part of durable candidate identity).
    # May carry private-source markers for the privacy gate; never sealed into
    # candidate_cid and never written to public reports by this module.
    expanded_context: Mapping[str, Any] | None = field(default=None, compare=False)
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface_id",
            "task_id",
            "task_class",
            "risk_class",
            "environment",
            "route_id",
            "expanded_route_id",
            "context_pack_cid",
            "repository_state_cid",
            "verification_bundle_cid",
            "capsule_uncertainty",
            "uncertainty_score_bp",
            "token_savings_eligible",
            "savings_score_bp",
            "rule_exposure",
            "rule_exposure_score_bp",
            "sample_deficit_bp",
            "recent_omission",
            "recent_failure",
            "failure_score_bp",
            "cost_escalation_pressure_bp",
            "cone_size",
            "dynamic_features",
            "dynamic_features_score_bp",
            "policy_importance_bp",
            "promotion_evaluation",
            "new_task_class",
            "new_analyzer",
            "new_route",
            "includes_private_source",
            "queue_age_ms",
            "estimated_cost_micros",
            "expanded_provider_id",
            "notes",
            "metadata",
            "candidate_cid",
        }
    )

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
            self, "context_pack_cid", _optional_cid(self.context_pack_cid, "context_pack_cid")
        )
        object.__setattr__(
            self,
            "repository_state_cid",
            _optional_cid(self.repository_state_cid, "repository_state_cid"),
        )
        object.__setattr__(
            self,
            "verification_bundle_cid",
            _optional_cid(self.verification_bundle_cid, "verification_bundle_cid"),
        )
        for flag_name in (
            "capsule_uncertainty",
            "token_savings_eligible",
            "rule_exposure",
            "recent_omission",
            "recent_failure",
            "dynamic_features",
            "promotion_evaluation",
            "new_task_class",
            "new_analyzer",
            "new_route",
            "includes_private_source",
        ):
            object.__setattr__(
                self, flag_name, _bool(getattr(self, flag_name), flag_name)
            )
        for bp_name in (
            "uncertainty_score_bp",
            "savings_score_bp",
            "rule_exposure_score_bp",
            "sample_deficit_bp",
            "failure_score_bp",
            "cost_escalation_pressure_bp",
            "dynamic_features_score_bp",
            "policy_importance_bp",
        ):
            object.__setattr__(
                self, bp_name, _basis_points(getattr(self, bp_name), bp_name)
            )
        object.__setattr__(
            self,
            "cone_size",
            _nonneg_int(self.cone_size, "cone_size", maximum=MAX_CONE_SIZE),
        )
        object.__setattr__(
            self,
            "queue_age_ms",
            _nonneg_int(self.queue_age_ms, "queue_age_ms", maximum=MAX_AGE_MS),
        )
        if self.estimated_cost_micros is not None:
            object.__setattr__(
                self,
                "estimated_cost_micros",
                _nonneg_int(
                    self.estimated_cost_micros,
                    "estimated_cost_micros",
                    maximum=MAX_SPEND_MICROS_HARD,
                ),
            )
        if self.expanded_provider_id is not None:
            object.__setattr__(
                self,
                "expanded_provider_id",
                _token(self.expanded_provider_id, "expanded_provider_id"),
            )
        # expanded_context is transient planning input: accept a mapping without
        # sealing private markers into durable identity. Fail closed on non-map.
        if self.expanded_context is not None:
            if not isinstance(self.expanded_context, Mapping):
                raise SemanticGovernorSchedulerError(
                    "expanded_context must be a mapping or None"
                )
            object.__setattr__(
                self,
                "expanded_context",
                MappingProxyType(dict(self.expanded_context)),
            )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def resolved_context_pack_cid(self) -> str:
        if self.context_pack_cid is not None:
            return self.context_pack_cid
        return _stable_cid(f"context-pack:{self.task_id}")

    def resolved_repository_state_cid(self) -> str:
        if self.repository_state_cid is not None:
            return self.repository_state_cid
        return _stable_cid(f"repository-state:{self.task_id}")

    def estimated_cost(self, policy: AuditSchedulerPolicy) -> int:
        if self.estimated_cost_micros is not None:
            return int(self.estimated_cost_micros)
        return int(policy.default_estimated_cost_micros)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": AUDIT_CANDIDATE_SCHEMA,
            "interface_id": AUDIT_CANDIDATE_INTERFACE,
            "task_id": self.task_id,
            "task_class": self.task_class,
            "risk_class": self.risk_class,
            "environment": self.environment,
            "route_id": self.route_id,
            "expanded_route_id": self.expanded_route_id,
            "context_pack_cid": self.context_pack_cid,
            "repository_state_cid": self.repository_state_cid,
            "verification_bundle_cid": self.verification_bundle_cid,
            "capsule_uncertainty": self.capsule_uncertainty,
            "uncertainty_score_bp": self.uncertainty_score_bp,
            "token_savings_eligible": self.token_savings_eligible,
            "savings_score_bp": self.savings_score_bp,
            "rule_exposure": self.rule_exposure,
            "rule_exposure_score_bp": self.rule_exposure_score_bp,
            "sample_deficit_bp": self.sample_deficit_bp,
            "recent_omission": self.recent_omission,
            "recent_failure": self.recent_failure,
            "failure_score_bp": self.failure_score_bp,
            "cost_escalation_pressure_bp": self.cost_escalation_pressure_bp,
            "cone_size": self.cone_size,
            "dynamic_features": self.dynamic_features,
            "dynamic_features_score_bp": self.dynamic_features_score_bp,
            "policy_importance_bp": self.policy_importance_bp,
            "promotion_evaluation": self.promotion_evaluation,
            "new_task_class": self.new_task_class,
            "new_analyzer": self.new_analyzer,
            "new_route": self.new_route,
            "includes_private_source": self.includes_private_source,
            "queue_age_ms": self.queue_age_ms,
            "estimated_cost_micros": self.estimated_cost_micros,
            "expanded_provider_id": self.expanded_provider_id,
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def candidate_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["candidate_cid"] = self.candidate_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditCandidate":
        payload = _closed(data, cls._FIELDS, "AuditCandidate")
        claimed = payload.pop("candidate_cid", None)
        result = cls(
            task_id=payload["task_id"],
            task_class=payload.get("task_class", "default"),
            risk_class=payload.get("risk_class", "low"),
            environment=payload.get("environment"),
            route_id=payload.get("route_id", "route.compressed"),
            expanded_route_id=payload.get("expanded_route_id", "route.expanded"),
            context_pack_cid=payload.get("context_pack_cid"),
            repository_state_cid=payload.get("repository_state_cid"),
            verification_bundle_cid=payload.get("verification_bundle_cid"),
            capsule_uncertainty=bool(payload.get("capsule_uncertainty", False)),
            uncertainty_score_bp=payload.get("uncertainty_score_bp", 0),
            token_savings_eligible=bool(payload.get("token_savings_eligible", False)),
            savings_score_bp=payload.get("savings_score_bp", 0),
            rule_exposure=bool(payload.get("rule_exposure", False)),
            rule_exposure_score_bp=payload.get("rule_exposure_score_bp", 0),
            sample_deficit_bp=payload.get("sample_deficit_bp", 0),
            recent_omission=bool(payload.get("recent_omission", False)),
            recent_failure=bool(payload.get("recent_failure", False)),
            failure_score_bp=payload.get("failure_score_bp", 0),
            cost_escalation_pressure_bp=payload.get(
                "cost_escalation_pressure_bp", 0
            ),
            cone_size=payload.get("cone_size", 0),
            dynamic_features=bool(payload.get("dynamic_features", False)),
            dynamic_features_score_bp=payload.get("dynamic_features_score_bp", 0),
            policy_importance_bp=payload.get("policy_importance_bp", 0),
            promotion_evaluation=bool(payload.get("promotion_evaluation", False)),
            new_task_class=bool(payload.get("new_task_class", False)),
            new_analyzer=bool(payload.get("new_analyzer", False)),
            new_route=bool(payload.get("new_route", False)),
            includes_private_source=bool(
                payload.get("includes_private_source", False)
            ),
            queue_age_ms=payload.get("queue_age_ms", 0),
            estimated_cost_micros=payload.get("estimated_cost_micros"),
            expanded_provider_id=payload.get("expanded_provider_id"),
            notes=payload.get("notes"),
            metadata=payload.get("metadata") or {},
        )
        if claimed is not None and claimed != result.candidate_cid:
            raise SemanticGovernorSchedulerError(
                "AuditCandidate candidate_cid does not verify"
            )
        return result

    @classmethod
    def from_value(cls, value: "AuditCandidate | Mapping[str, Any]") -> "AuditCandidate":
        if isinstance(value, AuditCandidate):
            return value
        if isinstance(value, Mapping):
            if "task_id" not in value:
                raise SemanticGovernorSchedulerError(
                    "AuditCandidate mapping requires task_id"
                )
            # Prefer sealed path when schema present; else constructor kwargs.
            if "schema" in value or "candidate_cid" in value:
                return cls.from_dict(value)
            allowed = {
                k: value[k]
                for k in cls._FIELDS
                if k in value and k not in {"schema", "interface_id", "candidate_cid"}
            }
            # Transient planning-only field accepted from recipes.
            if "expanded_context" in value:
                allowed["expanded_context"] = value["expanded_context"]
            return cls(**allowed)
        raise SemanticGovernorSchedulerError(
            "AuditCandidate must be AuditCandidate or mapping"
        )


# ---------------------------------------------------------------------------
# AuditPriority
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuditPriority:
    """Bounded deterministic priority derived from expected information value.

    ``information_value_bp`` is the weighted composite of the named factors.
    ``effective_priority_bp`` adds a starvation boost when age exceeds the
    policy threshold. Ranking is stable: higher effective priority first, then
    lower task_id for ties.
    """

    task_id: str
    information_value_bp: int
    effective_priority_bp: int
    risk_score_bp: int
    uncertainty_score_bp: int
    savings_score_bp: int
    rule_exposure_score_bp: int
    sample_deficit_score_bp: int
    failure_score_bp: int
    cost_escalation_score_bp: int
    cone_size_score_bp: int
    dynamic_features_score_bp: int
    policy_importance_score_bp: int
    starvation_boost_bp: int = 0
    fairness_class: str = FairnessClass.MATURE_LOW_RISK.value
    queue_age_ms: int = 0
    reason_codes: tuple[str, ...] = ()
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface_id",
            "task_id",
            "information_value_bp",
            "effective_priority_bp",
            "risk_score_bp",
            "uncertainty_score_bp",
            "savings_score_bp",
            "rule_exposure_score_bp",
            "sample_deficit_score_bp",
            "failure_score_bp",
            "cost_escalation_score_bp",
            "cone_size_score_bp",
            "dynamic_features_score_bp",
            "policy_importance_score_bp",
            "starvation_boost_bp",
            "fairness_class",
            "queue_age_ms",
            "reason_codes",
            "notes",
            "metadata",
            "priority_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _task_id(self.task_id))
        for name in (
            "information_value_bp",
            "effective_priority_bp",
            "risk_score_bp",
            "uncertainty_score_bp",
            "savings_score_bp",
            "rule_exposure_score_bp",
            "sample_deficit_score_bp",
            "failure_score_bp",
            "cost_escalation_score_bp",
            "cone_size_score_bp",
            "dynamic_features_score_bp",
            "policy_importance_score_bp",
            "starvation_boost_bp",
        ):
            # effective_priority may exceed BASIS_POINTS_MAX by starvation boost
            # but is still bounded to 2 * BASIS_POINTS_MAX.
            if name == "effective_priority_bp":
                value = _nonneg_int(
                    getattr(self, name),
                    name,
                    maximum=BASIS_POINTS_MAX * 2,
                )
            else:
                value = _basis_points(getattr(self, name), name)
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "fairness_class",
            _enum(self.fairness_class, FairnessClass, "fairness_class"),
        )
        object.__setattr__(
            self,
            "queue_age_ms",
            _nonneg_int(self.queue_age_ms, "queue_age_ms", maximum=MAX_AGE_MS),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _unique_sorted_tokens(self.reason_codes, "reason_codes"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def rank_key(self) -> tuple[int, str]:
        """Deterministic sort key: higher priority first, then task_id."""

        return (-int(self.effective_priority_bp), self.task_id)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": AUDIT_PRIORITY_SCHEMA,
            "interface_id": AUDIT_PRIORITY_INTERFACE,
            "task_id": self.task_id,
            "information_value_bp": self.information_value_bp,
            "effective_priority_bp": self.effective_priority_bp,
            "risk_score_bp": self.risk_score_bp,
            "uncertainty_score_bp": self.uncertainty_score_bp,
            "savings_score_bp": self.savings_score_bp,
            "rule_exposure_score_bp": self.rule_exposure_score_bp,
            "sample_deficit_score_bp": self.sample_deficit_score_bp,
            "failure_score_bp": self.failure_score_bp,
            "cost_escalation_score_bp": self.cost_escalation_score_bp,
            "cone_size_score_bp": self.cone_size_score_bp,
            "dynamic_features_score_bp": self.dynamic_features_score_bp,
            "policy_importance_score_bp": self.policy_importance_score_bp,
            "starvation_boost_bp": self.starvation_boost_bp,
            "fairness_class": self.fairness_class,
            "queue_age_ms": self.queue_age_ms,
            "reason_codes": list(self.reason_codes),
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def priority_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["priority_cid"] = self.priority_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditPriority":
        payload = _closed(data, cls._FIELDS, "AuditPriority")
        claimed = payload.pop("priority_cid", None)
        result = cls(
            task_id=payload["task_id"],
            information_value_bp=payload["information_value_bp"],
            effective_priority_bp=payload["effective_priority_bp"],
            risk_score_bp=payload["risk_score_bp"],
            uncertainty_score_bp=payload["uncertainty_score_bp"],
            savings_score_bp=payload["savings_score_bp"],
            rule_exposure_score_bp=payload["rule_exposure_score_bp"],
            sample_deficit_score_bp=payload["sample_deficit_score_bp"],
            failure_score_bp=payload["failure_score_bp"],
            cost_escalation_score_bp=payload["cost_escalation_score_bp"],
            cone_size_score_bp=payload["cone_size_score_bp"],
            dynamic_features_score_bp=payload["dynamic_features_score_bp"],
            policy_importance_score_bp=payload["policy_importance_score_bp"],
            starvation_boost_bp=payload.get("starvation_boost_bp", 0),
            fairness_class=payload.get(
                "fairness_class", FairnessClass.MATURE_LOW_RISK.value
            ),
            queue_age_ms=payload.get("queue_age_ms", 0),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            notes=payload.get("notes"),
            metadata=payload.get("metadata") or {},
        )
        if claimed is not None and claimed != result.priority_cid:
            raise SemanticGovernorSchedulerError(
                "AuditPriority priority_cid does not verify"
            )
        return result


def _risk_score_bp(risk_class: str) -> int:
    risk = _normalize_risk(risk_class)
    if risk in {"critical", "risk_critical"}:
        return BASIS_POINTS_MAX
    if risk in HIGH_RISK_CLASSES:
        return 8_500
    if risk in MEDIUM_RISK_CLASSES:
        return 5_000
    if risk in LOW_RISK_CLASSES:
        return 1_500
    return 2_500


def _cone_size_score_bp(cone_size: int) -> int:
    """Larger cones raise priority (more structural uncertainty) but saturate."""

    if cone_size <= 0:
        return 0
    # Linear map: 0..1024 -> 0..10000, then clamp.
    scaled = (int(cone_size) * BASIS_POINTS_MAX) // 1_024
    return _clamp_bp(scaled)


def compute_audit_priority(
    candidate: AuditCandidate | Mapping[str, Any],
    policy: AuditSchedulerPolicy | Mapping[str, Any] | None = None,
    *,
    lifecycle_phase: str | None = None,
) -> AuditPriority:
    """Compute bounded expected-information-value priority for one candidate.

    Factors ranked: risk, uncertainty, savings, rule exposure, sample deficit,
    failures, cost/escalation, cone size, dynamic features, policy importance.
    Starvation boost is applied when ``queue_age_ms >= policy.starvation_age_ms``.
    """

    cand = AuditCandidate.from_value(candidate)
    sched_policy = (
        AuditSchedulerPolicy.from_dict(policy)
        if isinstance(policy, Mapping)
        else (policy or default_audit_scheduler_policy())
    )
    if not isinstance(sched_policy, AuditSchedulerPolicy):
        raise SemanticGovernorSchedulerError(
            "policy must be AuditSchedulerPolicy or mapping"
        )

    reasons: list[str] = []

    risk_score = _risk_score_bp(cand.risk_class)
    if risk_score >= 8_500:
        reasons.append("risk_high")

    uncertainty = cand.uncertainty_score_bp
    if cand.capsule_uncertainty:
        uncertainty = max(uncertainty, 7_500)
        reasons.append("capsule_uncertainty")
    if cand.new_analyzer or cand.new_task_class or cand.new_route:
        uncertainty = max(uncertainty, 6_000)
        reasons.append("novelty")

    savings = cand.savings_score_bp
    if cand.token_savings_eligible:
        savings = max(savings, 5_000)
        reasons.append("token_savings")

    rule_exposure = cand.rule_exposure_score_bp
    if cand.rule_exposure or cand.promotion_evaluation:
        rule_exposure = max(rule_exposure, 7_000)
        reasons.append("rule_exposure")

    sample_deficit = cand.sample_deficit_bp
    if sample_deficit > 0:
        reasons.append("sample_deficit")

    failure = cand.failure_score_bp
    if cand.recent_omission or cand.recent_failure:
        failure = max(failure, 8_000)
        reasons.append("recent_failure")

    cost_escalation = cand.cost_escalation_pressure_bp
    if cost_escalation > 0:
        reasons.append("cost_escalation")

    cone_score = _cone_size_score_bp(cand.cone_size)
    if cone_score > 0:
        reasons.append("cone_size")

    dynamic = cand.dynamic_features_score_bp
    if cand.dynamic_features:
        dynamic = max(dynamic, 6_000)
        reasons.append("dynamic_features")

    policy_importance = cand.policy_importance_bp
    if cand.promotion_evaluation:
        policy_importance = max(policy_importance, 9_000)
        reasons.append("policy_importance")

    information_value = (
        _scale_weight(risk_score, WEIGHT_RISK_BP)
        + _scale_weight(uncertainty, WEIGHT_UNCERTAINTY_BP)
        + _scale_weight(savings, WEIGHT_SAVINGS_BP)
        + _scale_weight(rule_exposure, WEIGHT_RULE_EXPOSURE_BP)
        + _scale_weight(sample_deficit, WEIGHT_SAMPLE_DEFICIT_BP)
        + _scale_weight(failure, WEIGHT_FAILURES_BP)
        + _scale_weight(cost_escalation, WEIGHT_COST_ESCALATION_BP)
        + _scale_weight(cone_score, WEIGHT_CONE_SIZE_BP)
        + _scale_weight(dynamic, WEIGHT_DYNAMIC_FEATURES_BP)
        + _scale_weight(policy_importance, WEIGHT_POLICY_IMPORTANCE_BP)
    )
    information_value = _clamp_bp(information_value)

    starvation_boost = 0
    if (
        sched_policy.starvation_age_ms > 0
        and cand.queue_age_ms >= sched_policy.starvation_age_ms
    ):
        starvation_boost = int(sched_policy.starvation_boost_bp)
        reasons.append("starvation_boost")

    effective = min(BASIS_POINTS_MAX * 2, information_value + starvation_boost)

    fairness = classify_fairness_class(
        risk_class=cand.risk_class,
        environment=cand.environment,
        lifecycle_phase=lifecycle_phase,
    )

    return AuditPriority(
        task_id=cand.task_id,
        information_value_bp=information_value,
        effective_priority_bp=effective,
        risk_score_bp=risk_score,
        uncertainty_score_bp=_clamp_bp(uncertainty),
        savings_score_bp=_clamp_bp(savings),
        rule_exposure_score_bp=_clamp_bp(rule_exposure),
        sample_deficit_score_bp=_clamp_bp(sample_deficit),
        failure_score_bp=_clamp_bp(failure),
        cost_escalation_score_bp=_clamp_bp(cost_escalation),
        cone_size_score_bp=cone_score,
        dynamic_features_score_bp=_clamp_bp(dynamic),
        policy_importance_score_bp=_clamp_bp(policy_importance),
        starvation_boost_bp=starvation_boost,
        fairness_class=fairness,
        queue_age_ms=cand.queue_age_ms,
        reason_codes=tuple(reasons),
        metadata={"evidence": SCG_ACTIVE_SCHEDULER_EVIDENCE},
    )


# ---------------------------------------------------------------------------
# Admission / schedule result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuditAdmission:
    """One candidate's scheduling outcome for a tick."""

    task_id: str
    disposition: AuditAdmissionDisposition | str
    priority: AuditPriority | None = None
    plan_decision: ShadowPlanDecision | None = None
    allow_external_expanded_disclosure: bool = False
    estimated_cost_micros: int = 0
    fairness_class: str = FairnessClass.OTHER.value
    reason_codes: tuple[str, ...] = ()
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _task_id(self.task_id))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, AuditAdmissionDisposition, "disposition"),
        )
        if self.priority is not None and not isinstance(self.priority, AuditPriority):
            raise SemanticGovernorSchedulerError(
                "priority must be AuditPriority or None"
            )
        if self.plan_decision is not None and not isinstance(
            self.plan_decision, ShadowPlanDecision
        ):
            raise SemanticGovernorSchedulerError(
                "plan_decision must be ShadowPlanDecision or None"
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
            "estimated_cost_micros",
            _nonneg_int(
                self.estimated_cost_micros,
                "estimated_cost_micros",
                maximum=MAX_SPEND_MICROS_HARD,
            ),
        )
        object.__setattr__(
            self,
            "fairness_class",
            _enum(self.fairness_class, FairnessClass, "fairness_class"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _unique_sorted_tokens(self.reason_codes, "reason_codes"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

        # Fail closed: admitted privacy-skip / forbidden must not allow external.
        if self.allow_external_expanded_disclosure and self.disposition in {
            AuditAdmissionDisposition.PRIVACY_EXTERNAL_SKIPPED.value,
        }:
            raise SemanticGovernorSchedulerError(
                "privacy_external_skipped cannot allow external expanded disclosure"
            )
        if (
            self.plan_decision is not None
            and self.allow_external_expanded_disclosure
            and not self.plan_decision.allow_external_expanded_disclosure
        ):
            raise SemanticGovernorSchedulerError(
                "admission external flag disagrees with plan decision"
            )

    @property
    def admitted(self) -> bool:
        return self.disposition == AuditAdmissionDisposition.ADMITTED.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": AUDIT_ADMISSION_SCHEMA,
            "interface_id": AUDIT_ADMISSION_INTERFACE,
            "task_id": self.task_id,
            "disposition": self.disposition,
            "priority": self.priority.to_dict() if self.priority is not None else None,
            "plan_decision": (
                self.plan_decision.to_dict()
                if self.plan_decision is not None
                else None
            ),
            "allow_external_expanded_disclosure": (
                self.allow_external_expanded_disclosure
            ),
            "estimated_cost_micros": self.estimated_cost_micros,
            "fairness_class": self.fairness_class,
            "reason_codes": list(self.reason_codes),
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
            "admitted": self.admitted,
        }


@dataclass(frozen=True, slots=True)
class AuditScheduleResult:
    """Sealed outcome of one schedule_audits tick."""

    header: GovernorArtifactHeader
    schedule_id: str
    admissions: tuple[AuditAdmission, ...]
    deferred: tuple[AuditAdmission, ...]
    rejected: tuple[AuditAdmission, ...]
    admitted_task_ids: tuple[str, ...]
    queue_depth_before: int
    queue_depth_after: int
    admitted_count: int
    projected_spend_micros: int
    mature_low_risk_admitted_count: int
    audit_policy_cid: str
    scheduler_policy_cid: str
    reason_codes: tuple[str, ...] = ()
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.header, GovernorArtifactHeader):
            if isinstance(self.header, Mapping):
                object.__setattr__(
                    self, "header", GovernorArtifactHeader.from_dict(self.header)
                )
            else:
                raise SemanticGovernorSchedulerError(
                    "header must be GovernorArtifactHeader"
                )
        object.__setattr__(self, "schedule_id", _token(self.schedule_id, "schedule_id"))
        admissions = tuple(self.admissions)
        deferred = tuple(self.deferred)
        rejected = tuple(self.rejected)
        for group_name, group in (
            ("admissions", admissions),
            ("deferred", deferred),
            ("rejected", rejected),
        ):
            for item in group:
                if not isinstance(item, AuditAdmission):
                    raise SemanticGovernorSchedulerError(
                        f"{group_name} items must be AuditAdmission"
                    )
        object.__setattr__(self, "admissions", admissions)
        object.__setattr__(self, "deferred", deferred)
        object.__setattr__(self, "rejected", rejected)
        object.__setattr__(
            self,
            "admitted_task_ids",
            tuple(_task_id(t, "admitted_task_ids") for t in self.admitted_task_ids),
        )
        for name in (
            "queue_depth_before",
            "queue_depth_after",
            "admitted_count",
            "projected_spend_micros",
            "mature_low_risk_admitted_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonneg_int(getattr(self, name), name, maximum=MAX_SPEND_MICROS_HARD),
            )
        object.__setattr__(
            self, "audit_policy_cid", _cid(self.audit_policy_cid, "audit_policy_cid")
        )
        object.__setattr__(
            self,
            "scheduler_policy_cid",
            _cid(self.scheduler_policy_cid, "scheduler_policy_cid"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _unique_sorted_tokens(self.reason_codes, "reason_codes"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        if self.admitted_count != len(self.admissions):
            raise SemanticGovernorSchedulerError(
                "admitted_count must equal len(admissions)"
            )
        if tuple(a.task_id for a in self.admissions) != self.admitted_task_ids:
            raise SemanticGovernorSchedulerError(
                "admitted_task_ids must match admissions order"
            )
        # External disclosure never set on privacy skip admissions.
        for admission in self.admissions:
            if admission.allow_external_expanded_disclosure:
                if (
                    ShadowSelectionReason.DISCLOSURE_FORBIDDEN_SKIP.value
                    in (
                        admission.plan_decision.selection_reasons
                        if admission.plan_decision is not None
                        else ()
                    )
                ):
                    raise SemanticGovernorSchedulerError(
                        "admitted audit with disclosure_forbidden_skip cannot "
                        "allow external expanded disclosure"
                    )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": AUDIT_SCHEDULE_RESULT_SCHEMA,
            "interface_id": AUDIT_SCHEDULE_RESULT_INTERFACE,
            "header": self.header.to_dict(),
            "schedule_id": self.schedule_id,
            "admissions": [item.to_dict() for item in self.admissions],
            "deferred": [item.to_dict() for item in self.deferred],
            "rejected": [item.to_dict() for item in self.rejected],
            "admitted_task_ids": list(self.admitted_task_ids),
            "queue_depth_before": self.queue_depth_before,
            "queue_depth_after": self.queue_depth_after,
            "admitted_count": self.admitted_count,
            "projected_spend_micros": self.projected_spend_micros,
            "mature_low_risk_admitted_count": self.mature_low_risk_admitted_count,
            "audit_policy_cid": self.audit_policy_cid,
            "scheduler_policy_cid": self.scheduler_policy_cid,
            "reason_codes": list(self.reason_codes),
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def result_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["result_cid"] = self.result_cid
        return payload


# ---------------------------------------------------------------------------
# Planning helpers
# ---------------------------------------------------------------------------


def _coerce_sampling_policy(
    policy: ShadowSamplingPolicy | Mapping[str, Any] | None,
) -> ShadowSamplingPolicy:
    if policy is None:
        return default_shadow_sampling_policy()
    if isinstance(policy, ShadowSamplingPolicy):
        return policy
    if isinstance(policy, Mapping):
        if "schema" in policy or "policy_cid" in policy:
            return ShadowSamplingPolicy.from_dict(policy)
        return ShadowSamplingPolicy(**dict(policy))
    raise SemanticGovernorSchedulerError(
        "audit_policy must be ShadowSamplingPolicy or mapping"
    )


def _coerce_disclosure_policy(
    policy: ShadowDisclosurePolicy | Mapping[str, Any] | None,
) -> ShadowDisclosurePolicy:
    if policy is None:
        return default_shadow_disclosure_policy()
    if isinstance(policy, ShadowDisclosurePolicy):
        return policy
    if isinstance(policy, Mapping):
        if "schema" in policy and "interface_id" in policy:
            return ShadowDisclosurePolicy.from_dict(policy)
        return ShadowDisclosurePolicy(**dict(policy))
    raise SemanticGovernorSchedulerError(
        "disclosure_policy must be ShadowDisclosurePolicy or mapping"
    )


def _coerce_scheduler_policy(
    policy: AuditSchedulerPolicy | Mapping[str, Any] | None,
) -> AuditSchedulerPolicy:
    if policy is None:
        return default_audit_scheduler_policy()
    if isinstance(policy, AuditSchedulerPolicy):
        return policy
    if isinstance(policy, Mapping):
        return AuditSchedulerPolicy.from_dict(policy)
    raise SemanticGovernorSchedulerError(
        "scheduler_policy must be AuditSchedulerPolicy or mapping"
    )


def _plan_for_candidate(
    candidate: AuditCandidate,
    *,
    audit_policy: ShadowSamplingPolicy,
    disclosure_policy: ShadowDisclosurePolicy,
    sample_roll: int | None = None,
) -> ShadowPlanDecision:
    task = ShadowTaskView(
        task_id=candidate.task_id,
        task_class=candidate.task_class,
        risk_class=candidate.risk_class,
        environment=candidate.environment,
        route_id=candidate.route_id,
        expanded_route_id=candidate.expanded_route_id,
        promotion_evaluation=candidate.promotion_evaluation,
        new_task_class=candidate.new_task_class,
        new_analyzer=candidate.new_analyzer,
        new_route=candidate.new_route,
    )
    compressed = CompressedContextView(
        context_pack_cid=candidate.resolved_context_pack_cid(),
        capsule_uncertainty=candidate.capsule_uncertainty,
        token_savings_eligible=candidate.token_savings_eligible,
        includes_private_source=candidate.includes_private_source,
    )
    repo = RepositoryStateSignals(
        repository_state_cid=candidate.resolved_repository_state_cid(),
        recent_omission=candidate.recent_omission,
        recent_failure=candidate.recent_failure,
        verification_bundle_cid=candidate.verification_bundle_cid,
    )
    return create_shadow_plan(
        task,
        compressed,
        repo,
        audit_policy,
        disclosure_policy=disclosure_policy,
        expanded_provider_id=candidate.expanded_provider_id,
        expanded_context=candidate.expanded_context,
        sample_roll=sample_roll,
    )


def _max_mature_low_risk_slots(
    max_admissions: int, fraction_bp: int
) -> int:
    """Integer max mature-low-risk admissions for this tick (at least 0)."""

    if max_admissions <= 0:
        return 0
    # Ceiling of fraction so tiny budgets still allow one low-risk when budget>0
    # would be wrong for anti-monopoly — use floor, but allow 0.
    return (max_admissions * int(fraction_bp)) // BASIS_POINTS_MAX


# ---------------------------------------------------------------------------
# ActiveAuditScheduler
# ---------------------------------------------------------------------------


class ActiveAuditScheduler:
    """Stateful bounded queue for expected-information-value audit admission.

    Queue growth is hard-capped by ``AuditSchedulerPolicy.max_queue_depth``.
    Scheduling ranks candidates by :class:`AuditPriority`, applies shadow
    sample rates, privacy zero-call policy, resource spend/slot budgets, and
    anti-monopoly caps on mature low-risk work.
    """

    def __init__(
        self,
        *,
        scheduler_policy: AuditSchedulerPolicy | Mapping[str, Any] | None = None,
        audit_policy: ShadowSamplingPolicy | Mapping[str, Any] | None = None,
        disclosure_policy: ShadowDisclosurePolicy | Mapping[str, Any] | None = None,
    ) -> None:
        self._scheduler_policy = _coerce_scheduler_policy(scheduler_policy)
        self._audit_policy = _coerce_sampling_policy(audit_policy)
        self._disclosure_policy = _coerce_disclosure_policy(disclosure_policy)
        self._queue: list[AuditCandidate] = []
        self._pending_ids: set[str] = set()
        self._tick: int = 0
        self._overflow_count: int = 0
        self._total_admitted: int = 0

    @property
    def scheduler_policy(self) -> AuditSchedulerPolicy:
        return self._scheduler_policy

    @property
    def audit_policy(self) -> ShadowSamplingPolicy:
        return self._audit_policy

    @property
    def disclosure_policy(self) -> ShadowDisclosurePolicy:
        return self._disclosure_policy

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    @property
    def overflow_count(self) -> int:
        return self._overflow_count

    @property
    def total_admitted(self) -> int:
        return self._total_admitted

    def pending_task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._pending_ids))

    def clear(self) -> None:
        self._queue.clear()
        self._pending_ids.clear()

    def enqueue(
        self,
        candidate: AuditCandidate | Mapping[str, Any],
        *,
        replace: bool = False,
    ) -> AuditAdmission | None:
        """Enqueue one candidate. Returns overflow/duplicate admission or None."""

        cand = AuditCandidate.from_value(candidate)
        if cand.task_id in self._pending_ids:
            if replace:
                self._queue = [c for c in self._queue if c.task_id != cand.task_id]
                self._pending_ids.discard(cand.task_id)
            else:
                return AuditAdmission(
                    task_id=cand.task_id,
                    disposition=AuditAdmissionDisposition.DUPLICATE,
                    fairness_class=classify_fairness_class(
                        risk_class=cand.risk_class,
                        environment=cand.environment,
                    ),
                    reason_codes=("duplicate_task_id",),
                    estimated_cost_micros=cand.estimated_cost(self._scheduler_policy),
                )

        if len(self._queue) >= self._scheduler_policy.max_queue_depth:
            self._overflow_count += 1
            if self._scheduler_policy.raise_on_queue_overflow:
                raise AuditQueueOverflowError(
                    f"audit queue at capacity {self._scheduler_policy.max_queue_depth}"
                )
            return AuditAdmission(
                task_id=cand.task_id,
                disposition=AuditAdmissionDisposition.QUEUE_OVERFLOW,
                fairness_class=classify_fairness_class(
                    risk_class=cand.risk_class,
                    environment=cand.environment,
                ),
                reason_codes=("queue_capacity_reached", "unbounded_growth_prevented"),
                estimated_cost_micros=cand.estimated_cost(self._scheduler_policy),
                metadata={
                    "max_queue_depth": self._scheduler_policy.max_queue_depth,
                    "queue_depth": len(self._queue),
                },
            )

        self._queue.append(cand)
        self._pending_ids.add(cand.task_id)
        return None

    def enqueue_many(
        self, candidates: Sequence[AuditCandidate | Mapping[str, Any]]
    ) -> tuple[AuditAdmission, ...]:
        """Enqueue many candidates; returns overflow/duplicate outcomes."""

        outcomes: list[AuditAdmission] = []
        for raw in candidates:
            outcome = self.enqueue(raw)
            if outcome is not None:
                outcomes.append(outcome)
        return tuple(outcomes)

    def schedule_audits(
        self,
        candidates: Sequence[AuditCandidate | Mapping[str, Any]] | None = None,
        *,
        sample_rolls: Mapping[str, int] | None = None,
        schedule_id: str | None = None,
    ) -> AuditScheduleResult:
        """Rank, sample, and resource-admit pending (and optional new) candidates.

        Optional ``candidates`` are enqueued first (subject to queue bounds).
        ``sample_rolls`` overrides deterministic shadow sample rolls per task_id
        for hermetic tests.
        """

        rejected: list[AuditAdmission] = []
        if candidates:
            if len(candidates) > MAX_CANDIDATES_PER_SCHEDULE:
                raise SemanticGovernorSchedulerError(
                    f"candidates exceeds max {MAX_CANDIDATES_PER_SCHEDULE}"
                )
            for raw in candidates:
                outcome = self.enqueue(raw)
                if outcome is not None:
                    rejected.append(outcome)

        depth_before = len(self._queue)
        self._tick += 1
        sid = schedule_id or f"schedule.tick.{self._tick}"

        # Score and sort.
        scored: list[tuple[AuditPriority, AuditCandidate]] = []
        for cand in self._queue:
            # Fairness uses candidate environment only — never the global
            # sampling-policy lifecycle (which would reclassify all work as
            # development under a development audit policy).
            priority = compute_audit_priority(
                cand,
                self._scheduler_policy,
                lifecycle_phase=None,
            )
            scored.append((priority, cand))
        scored.sort(key=lambda item: item[0].rank_key())

        max_slots = self._scheduler_policy.max_admissions_per_tick
        max_spend = self._scheduler_policy.max_spend_micros_per_tick
        max_low = _max_mature_low_risk_slots(
            max_slots, self._scheduler_policy.max_mature_low_risk_fraction_bp
        )

        admitted: list[AuditAdmission] = []
        deferred: list[AuditAdmission] = []
        remaining: list[AuditCandidate] = []
        remaining_ids: set[str] = set()

        spend = 0
        low_count = 0
        rolls = dict(sample_rolls or {})

        for priority, cand in scored:
            cost = cand.estimated_cost(self._scheduler_policy)
            fairness = priority.fairness_class

            # Anti-monopoly first: mature low-risk cannot monopolize remaining
            # capacity even when slots are still open (or when the only open
            # slots would otherwise be taken by low-risk flood).
            if is_mature_low_risk_fairness(fairness) and low_count >= max_low:
                deferred.append(
                    AuditAdmission(
                        task_id=cand.task_id,
                        disposition=AuditAdmissionDisposition.ANTI_MONOPOLY_DEFERRED,
                        priority=priority,
                        estimated_cost_micros=cost,
                        fairness_class=fairness,
                        reason_codes=(
                            "mature_low_risk_monopoly_prevented",
                            "fairness_cap",
                        ),
                        metadata={
                            "max_mature_low_risk_slots": max_low,
                            "mature_low_risk_admitted": low_count,
                        },
                    )
                )
                remaining.append(cand)
                remaining_ids.add(cand.task_id)
                continue

            # Resource: admission slots.
            if len(admitted) >= max_slots:
                deferred.append(
                    AuditAdmission(
                        task_id=cand.task_id,
                        disposition=AuditAdmissionDisposition.RESOURCE_DENIED,
                        priority=priority,
                        estimated_cost_micros=cost,
                        fairness_class=fairness,
                        reason_codes=("admission_slots_exhausted",),
                    )
                )
                remaining.append(cand)
                remaining_ids.add(cand.task_id)
                continue

            # Resource: spend budget.
            if max_spend > 0 and spend + cost > max_spend:
                deferred.append(
                    AuditAdmission(
                        task_id=cand.task_id,
                        disposition=AuditAdmissionDisposition.SPEND_EXHAUSTED,
                        priority=priority,
                        estimated_cost_micros=cost,
                        fairness_class=fairness,
                        reason_codes=("spend_budget_exhausted",),
                    )
                )
                remaining.append(cand)
                remaining_ids.add(cand.task_id)
                continue

            # Shadow sampling + privacy via create_shadow_plan.
            roll = rolls.get(cand.task_id)
            decision = _plan_for_candidate(
                cand,
                audit_policy=self._audit_policy,
                disclosure_policy=self._disclosure_policy,
                sample_roll=roll,
            )
            assert_no_disclosure_bypass(decision)

            if not decision.selected:
                deferred.append(
                    AuditAdmission(
                        task_id=cand.task_id,
                        disposition=AuditAdmissionDisposition.SAMPLE_SKIPPED,
                        priority=priority,
                        plan_decision=decision,
                        allow_external_expanded_disclosure=False,
                        estimated_cost_micros=cost,
                        fairness_class=fairness,
                        reason_codes=("sample_miss", "shadow_rate_honored"),
                        metadata={
                            "effective_sample_rate_bp": (
                                decision.effective_sample_rate_bp
                            ),
                            "sample_roll": decision.sample_roll,
                        },
                    )
                )
                # Sample miss removes from queue (consumed decision); not re-queued.
                continue

            # Privacy zero-call: external forbidden still admits local-only.
            allow_external = bool(decision.allow_external_expanded_disclosure)
            disposition = AuditAdmissionDisposition.ADMITTED
            reason_codes = ["admitted", "shadow_rate_honored"]
            if (
                decision.disposition
                == ShadowPlanDisposition.DISCLOSURE_EXTERNAL_SKIPPED.value
                or ShadowSelectionReason.DISCLOSURE_FORBIDDEN_SKIP.value
                in decision.selection_reasons
            ):
                allow_external = False
                reason_codes.append("privacy_zero_call_honored")
                reason_codes.append("disclosure_forbidden_skip")
            elif (
                decision.disposition
                == ShadowPlanDisposition.DISCLOSURE_LOCAL_ONLY.value
            ):
                allow_external = False
                reason_codes.append("local_only_expanded")

            # Absolute fail-closed: never external when policy forbids.
            if (
                self._audit_policy.zero_external_calls_when_disclosure_forbidden
                and decision.disclosure_disposition
                == DisclosureDisposition.FORBIDDEN.value
            ):
                allow_external = False
                reason_codes.append("privacy_zero_call_honored")

            if allow_external and decision.plan is not None:
                if not decision.plan.allow_external_expanded_disclosure:
                    allow_external = False

            admission = AuditAdmission(
                task_id=cand.task_id,
                disposition=disposition,
                priority=priority,
                plan_decision=decision,
                allow_external_expanded_disclosure=allow_external,
                estimated_cost_micros=cost,
                fairness_class=fairness,
                reason_codes=tuple(reason_codes),
                metadata={
                    "evidence": SCG_ACTIVE_SCHEDULER_EVIDENCE,
                    "plan_cid": decision.plan_cid,
                    "effective_priority_bp": priority.effective_priority_bp,
                },
            )
            admitted.append(admission)
            spend += cost
            if is_mature_low_risk_fairness(fairness):
                low_count += 1
            # Admitted removed from pending queue.

        # Rebuild queue from deferred resource/anti-monopoly candidates only.
        # Sample-skipped are intentionally dropped (sampling decision consumed).
        self._queue = remaining
        self._pending_ids = remaining_ids
        self._total_admitted += len(admitted)

        reason_codes = ["schedule_complete"]
        if rejected:
            reason_codes.append("queue_overflow_or_duplicate")
        if any(
            a.disposition == AuditAdmissionDisposition.ANTI_MONOPOLY_DEFERRED.value
            for a in deferred
        ):
            reason_codes.append("anti_monopoly_active")
        if any(
            a.priority is not None and a.priority.starvation_boost_bp > 0
            for a in admitted
        ):
            reason_codes.append("starvation_boost_applied")

        header = _header(
            "audit_schedule_result",
            input_cids=(
                self._audit_policy.policy_cid,
                self._scheduler_policy.policy_cid,
            ),
            interface_id=SCHEDULE_AUDITS_INTERFACE,
            metadata={
                "track": "audit-scheduler",
                "evidence": SCG_ACTIVE_SCHEDULER_EVIDENCE,
                "tick": self._tick,
            },
        )

        return AuditScheduleResult(
            header=header,
            schedule_id=_token(sid, "schedule_id"),
            admissions=tuple(admitted),
            deferred=tuple(deferred),
            rejected=tuple(rejected),
            admitted_task_ids=tuple(a.task_id for a in admitted),
            queue_depth_before=depth_before,
            queue_depth_after=len(self._queue),
            admitted_count=len(admitted),
            projected_spend_micros=spend,
            mature_low_risk_admitted_count=low_count,
            audit_policy_cid=self._audit_policy.policy_cid,
            scheduler_policy_cid=self._scheduler_policy.policy_cid,
            reason_codes=tuple(reason_codes),
            metadata={
                "evidence": SCG_ACTIVE_SCHEDULER_EVIDENCE,
                "overflow_count": self._overflow_count,
                "total_admitted": self._total_admitted,
            },
        )


# ---------------------------------------------------------------------------
# Public free function
# ---------------------------------------------------------------------------


def schedule_audits(
    candidates: Sequence[AuditCandidate | Mapping[str, Any]],
    *,
    scheduler_policy: AuditSchedulerPolicy | Mapping[str, Any] | None = None,
    audit_policy: ShadowSamplingPolicy | Mapping[str, Any] | None = None,
    disclosure_policy: ShadowDisclosurePolicy | Mapping[str, Any] | None = None,
    sample_rolls: Mapping[str, int] | None = None,
    schedule_id: str = "schedule.ephemeral.1",
) -> AuditScheduleResult:
    """Schedule candidate audits by expected information value (stateless tick).

    Constructs a fresh :class:`ActiveAuditScheduler`, enqueues candidates, and
    returns one admission tick. For multi-tick fairness / starvation across
    ages, use :class:`ActiveAuditScheduler` directly.
    """

    scheduler = ActiveAuditScheduler(
        scheduler_policy=scheduler_policy,
        audit_policy=audit_policy,
        disclosure_policy=disclosure_policy,
    )
    return scheduler.schedule_audits(
        candidates,
        sample_rolls=sample_rolls,
        schedule_id=schedule_id,
    )


def schedule_audits_interface_id() -> str:
    return SCHEDULE_AUDITS_INTERFACE


def active_audit_scheduler_evidence_id() -> str:
    return SCG_ACTIVE_SCHEDULER_EVIDENCE


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = (
    "ACTIVE_AUDIT_SCHEDULER_INTERFACE",
    "AUDIT_ADMISSION_INTERFACE",
    "AUDIT_ADMISSION_SCHEMA",
    "AUDIT_CANDIDATE_INTERFACE",
    "AUDIT_CANDIDATE_SCHEMA",
    "AUDIT_PRIORITY_INTERFACE",
    "AUDIT_PRIORITY_SCHEMA",
    "AUDIT_SCHEDULE_RESULT_INTERFACE",
    "AUDIT_SCHEDULE_RESULT_SCHEMA",
    "AUDIT_SCHEDULER_POLICY_INTERFACE",
    "AUDIT_SCHEDULER_POLICY_SCHEMA",
    "BASIS_POINTS_MAX",
    "DEFAULT_MAX_ADMISSIONS_PER_TICK",
    "DEFAULT_MAX_MATURE_LOW_RISK_FRACTION_BP",
    "DEFAULT_MAX_QUEUE_DEPTH",
    "DEFAULT_MAX_SPEND_MICROS_PER_TICK",
    "DEFAULT_STARVATION_AGE_MS",
    "DEFAULT_STARVATION_BOOST_BP",
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "SCHEDULE_AUDITS_INTERFACE",
    "SCG_ACTIVE_SCHEDULER_EVIDENCE",
    "ActiveAuditScheduler",
    "AuditAdmission",
    "AuditAdmissionDisposition",
    "AuditCandidate",
    "AuditPriority",
    "AuditQueueOverflowError",
    "AuditResourceDeniedError",
    "AuditScheduleResult",
    "AuditSchedulerPolicy",
    "FairnessClass",
    "SemanticGovernorSchedulerError",
    "active_audit_scheduler_evidence_id",
    "classify_fairness_class",
    "compute_audit_priority",
    "default_audit_scheduler_policy",
    "is_mature_low_risk_fairness",
    "schedule_audits",
    "schedule_audits_interface_id",
)
