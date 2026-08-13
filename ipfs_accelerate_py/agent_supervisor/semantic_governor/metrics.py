"""Complete semantic-governor metrics aggregation (SCG-038).

Computes compression, quality, omission, routing, economic, and calibration
metrics over sealed audit/run/calibration receipts. The surface is
observability-only: it never grants acceptance, promotion, or route authority.

Normative fail-closed rules:

* **Cohort separation** — simulated and live observations never share quality
  counters or live savings claims. Each cohort is aggregated independently.
* **Exact integer accounting** — tokens, micros, basis points, and counters
  only. Durable payloads never carry host floats.
* **Reproducible percentiles** — nearest-rank on sorted samples with a fixed
  integer formula; identical ordered samples yield identical percentiles.
* **Net savings include audit overhead** — gross inference savings are reduced
  by audit, verification, and shadow compute before net savings are reported.
* **Unavailable is not zero** — empty populations yield missing percentiles /
  rates / unit costs, never fabricated success or zero cost-per-accepted.
* **Provenance** — every report binds source receipt CIDs and recomputes a
  content identity from the sealed payload.

Interfaces: :class:`GovernorMetricsCollector`, :class:`GovernorMetricReport`.

Importing this module performs no I/O and never invokes a provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import ceil, floor, sqrt
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
    RouteTier,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.base import (
    ExecutionMode,
    SemanticGovernorBaseError,
    reject_private_and_model_authority,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.calibration import (
    wilson_score_interval_bp,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.calibration_contracts import (
    BASIS_POINTS,
    EmpiricalRate,
    EvidencePartition,
    ratio_to_basis_points,
)

from ipfs_accelerate_py.agent_supervisor.semantic_governor.contracts import (
    AcceptanceDisposition,
    ComparativeOutcome,
    SemanticGovernorExecutionError,
)

# ---------------------------------------------------------------------------
# Evidence / interface / schema constants
# ---------------------------------------------------------------------------

SCG_METRICS_EVIDENCE: Final[str] = "scg/metrics@1"

GOVERNOR_METRICS_COLLECTOR_INTERFACE: Final[str] = "GovernorMetricsCollector@1"
GOVERNOR_METRIC_REPORT_INTERFACE: Final[str] = "GovernorMetricReport@1"

METRICS_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "metrics-observation@1"
)
INTEGER_PERCENTILE_SUMMARY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "integer-percentile-summary@1"
)
COMPRESSION_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "compression-metrics@1"
)
QUALITY_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "quality-metrics@1"
)
OMISSION_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "omission-metrics@1"
)
ROUTING_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "routing-metrics@1"
)
ECONOMIC_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "economic-metrics@1"
)
CALIBRATION_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "calibration-metrics@1"
)
COHORT_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "cohort-metrics@1"
)
GOVERNOR_METRIC_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "governor-metric-report@1"
)

GENERATOR_ID: Final[str] = "semantic_governor_metrics"
GENERATOR_VERSION: Final[str] = "1.0.0"
PRODUCER_ID: Final[str] = "semantic_governor"
PRODUCER_VERSION: Final[str] = "1"
TOOL_ID: Final[str] = "metrics.v1"

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_METADATA_KEYS: Final[int] = 64
MAX_CID_LIST: Final[int] = 4_096
MAX_OBSERVATIONS: Final[int] = 16_384
MAX_TASK_CLASSES: Final[int] = 512
MAX_COUNTER: Final[int] = 2**63 - 1
MAX_REVISION: Final[int] = 2**63 - 1

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")

_ROUTE_TIERS: Final[tuple[str, ...]] = (
    RouteTier.DETERMINISTIC.value,
    RouteTier.SMALL.value,
    RouteTier.MEDIUM.value,
    RouteTier.FRONTIER.value,
    RouteTier.HUMAN.value,
)

_COMPARATIVE_OUTCOMES: Final[tuple[str, ...]] = tuple(
    item.value for item in ComparativeOutcome
)

# Percentiles reported for every token/cost distribution (basis-point points).
_PERCENTILE_POINTS_BP: Final[tuple[int, ...]] = (
    5_000,  # p50
    9_000,  # p90
    9_500,  # p95
    9_900,  # p99
)

_PERCENTILE_LABELS: Final[Mapping[int, str]] = MappingProxyType(
    {
        5_000: "p50",
        9_000: "p90",
        9_500: "p95",
        9_900: "p99",
    }
)


# ---------------------------------------------------------------------------
# Errors and closed enumerations
# ---------------------------------------------------------------------------


class MetricsError(SemanticGovernorExecutionError):
    """Raised when metrics inputs are malformed or fail closed."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "metrics_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


class MetricsCohort(str, Enum):
    """Closed observation cohorts; live and simulated never merge."""

    LIVE = "live"
    SIMULATED = "simulated"


class MetricsIngestDisposition(str, Enum):
    """Closed dispositions for a single observation ingest attempt."""

    APPLIED = "applied"
    SKIPPED_IDEMPOTENT = "skipped_idempotent"
    REJECTED_MALFORMED = "rejected_malformed"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        raise MetricsError(f"{name} must be a nonempty string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise MetricsError(f"{name} must be trimmed NFC text")
    if len(value) > MAX_TEXT_CHARS or any(not char.isprintable() for char in value):
        raise MetricsError(f"{name} contains invalid text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if _TOKEN_RE.fullmatch(text) is None:
        raise MetricsError(
            f"{name} must be a lowercase token matching {_TOKEN_RE.pattern}"
        )
    return text


def _optional_token(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _token(value, name)


def _cid(value: Any, name: str) -> str:
    try:
        return validate_cid(value)
    except Exception as exc:
        raise MetricsError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise MetricsError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise MetricsError(f"{name} must be a nonnegative integer")
    if value > MAX_COUNTER:
        raise MetricsError(f"{name} exceeds maximum")
    return value


def _optional_nonneg_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _nonneg_int(value, name)


def _basis_points(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise MetricsError(
            f"{name} must be an integer basis-point ratio in [0, {BASIS_POINTS}]"
        )
    if value < 0 or value > BASIS_POINTS:
        raise MetricsError(
            f"{name} must be an integer basis-point ratio in [0, {BASIS_POINTS}]"
        )
    return value


def _optional_basis_points(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _basis_points(value, name)


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    try:
        return enum_type(value).value
    except (TypeError, ValueError) as exc:
        raise MetricsError(f"{name} has unsupported value {value!r}") from exc


def _route_tier(value: Any, name: str = "route_tier") -> str:
    return _enum(value, RouteTier, name)


def _cohort(value: Any, name: str = "cohort") -> str:
    # Accept ExecutionMode as a convenience synonym for cohort.
    if isinstance(value, ExecutionMode):
        value = value.value
    if value == ExecutionMode.LIVE.value:
        return MetricsCohort.LIVE.value
    if value == ExecutionMode.SIMULATED.value:
        return MetricsCohort.SIMULATED.value
    return _enum(value, MetricsCohort, name)


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


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MetricsError(f"{name} must be a mapping")
    payload = dict(value)
    if len(payload) > MAX_METADATA_KEYS:
        raise MetricsError(f"{name} exceeds maximum key count")
    try:
        validate_structured_value(payload, path=name)
    except Exception as exc:
        raise MetricsError(
            f"{name} must be strict DAG-JSON without floats or host types"
        ) from exc
    try:
        reject_private_and_model_authority(payload, path=name)
    except SemanticGovernorBaseError as exc:
        raise MetricsError(str(exc)) from exc
    return _freeze_structured(payload)


def _unique_sorted_cids(values: Iterable[Any], name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise MetricsError(f"{name} must be a list or tuple")
    ordered = tuple(sorted({_cid(value, name) for value in values}))
    if len(ordered) > MAX_CID_LIST:
        raise MetricsError(f"{name} exceeds maximum length")
    return ordered


def _rate_bp(numerator: int, denominator: int) -> int | None:
    """Return floor ratio in basis points, or None when denominator is zero."""

    return ratio_to_basis_points(numerator, denominator)


def _add_counter(current: int, delta: int, name: str) -> int:
    total = current + delta
    if total > MAX_COUNTER:
        raise MetricsError(f"{name} would exceed maximum counter")
    return total


def _empty_route_share() -> dict[str, int]:
    return {tier: 0 for tier in _ROUTE_TIERS}


def _empty_outcome_counts() -> dict[str, int]:
    return {outcome: 0 for outcome in _COMPARATIVE_OUTCOMES}


# ---------------------------------------------------------------------------
# Deterministic integer percentile helpers
# ---------------------------------------------------------------------------


def nearest_rank_percentile(
    samples: Sequence[int],
    percentile_bp: int,
) -> int | None:
    """Return the nearest-rank percentile of ``samples``.

    Formula (integer-only, reproducible):

    * empty sample → ``None`` (unavailable, not zero)
    * index = ceil(percentile_bp * n / 10000) - 1, clamped to ``[0, n-1]``
    * value at that index of the ascending sorted sample

    ``percentile_bp`` is a basis-point percentile (5000 = p50, 9500 = p95).
    """

    if not samples:
        return None
    p = _basis_points(percentile_bp, "percentile_bp")
    ordered = sorted(int(item) for item in samples)
    n = len(ordered)
    # Nearest-rank: rank = ceil(p/10000 * n), index = rank - 1.
    rank = ceil((p * n) / BASIS_POINTS)
    if rank < 1:
        rank = 1
    if rank > n:
        rank = n
    return ordered[rank - 1]


def build_percentile_summary(
    samples: Sequence[int],
    *,
    sample_kind: str = "tokens",
) -> "IntegerPercentileSummary":
    """Build a sealed integer percentile summary for a sample sequence."""

    ordered = tuple(sorted(int(item) for item in samples))
    count = len(ordered)
    total = sum(ordered) if ordered else 0
    values: dict[str, int | None] = {}
    for point in _PERCENTILE_POINTS_BP:
        label = _PERCENTILE_LABELS[point]
        values[label] = nearest_rank_percentile(ordered, point)
    return IntegerPercentileSummary(
        sample_kind=sample_kind,
        sample_count=count,
        total=total,
        min_value=ordered[0] if ordered else None,
        max_value=ordered[-1] if ordered else None,
        p50=values["p50"],
        p90=values["p90"],
        p95=values["p95"],
        p99=values["p99"],
    )


def build_empirical_rate(successes: int, trials: int) -> EmpiricalRate:
    """Build an EmpiricalRate with Wilson 95% integer bounds."""

    successes = _nonneg_int(successes, "successes")
    trials = _nonneg_int(trials, "trials")
    if successes > trials:
        raise MetricsError("successes must not exceed trials")
    rate_bp, lower_bp, upper_bp = wilson_score_interval_bp(successes, trials)
    return EmpiricalRate(
        successes=successes,
        trials=trials,
        rate_bp=rate_bp,
        interval_lower_bp=lower_bp,
        interval_upper_bp=upper_bp,
        interval_method="wilson_score_95",
    )


# ---------------------------------------------------------------------------
# Integer percentile summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IntegerPercentileSummary:
    """Deterministic integer percentile projection for one sample family.

    Empty populations leave percentile fields as ``None`` (unavailable).
    """

    sample_kind: str
    sample_count: int
    total: int
    min_value: int | None
    max_value: int | None
    p50: int | None
    p90: int | None
    p95: int | None
    p99: int | None

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "sample_kind",
            "sample_count",
            "total",
            "min_value",
            "max_value",
            "p50",
            "p90",
            "p95",
            "p99",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sample_kind", _token(self.sample_kind, "sample_kind")
        )
        object.__setattr__(
            self, "sample_count", _nonneg_int(self.sample_count, "sample_count")
        )
        object.__setattr__(self, "total", _nonneg_int(self.total, "total"))
        for name in ("min_value", "max_value", "p50", "p90", "p95", "p99"):
            object.__setattr__(
                self, name, _optional_nonneg_int(getattr(self, name), name)
            )
        if self.sample_count == 0:
            for name in ("min_value", "max_value", "p50", "p90", "p95", "p99"):
                if getattr(self, name) is not None:
                    raise MetricsError(
                        f"{name} must be missing when sample_count is 0"
                    )
            if self.total != 0:
                raise MetricsError("total must be 0 when sample_count is 0")
        else:
            for name in ("min_value", "max_value", "p50", "p90", "p95", "p99"):
                if getattr(self, name) is None:
                    raise MetricsError(
                        f"{name} must be present when sample_count > 0"
                    )
            if self.min_value is not None and self.max_value is not None:
                if self.min_value > self.max_value:
                    raise MetricsError("min_value must not exceed max_value")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": INTEGER_PERCENTILE_SUMMARY_SCHEMA,
            "sample_kind": self.sample_kind,
            "sample_count": self.sample_count,
            "total": self.total,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.identity_payload()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IntegerPercentileSummary":
        payload = dict(data)
        schema = payload.pop("schema", INTEGER_PERCENTILE_SUMMARY_SCHEMA)
        if schema != INTEGER_PERCENTILE_SUMMARY_SCHEMA:
            raise MetricsError("unsupported IntegerPercentileSummary schema")
        return cls(
            sample_kind=payload.get("sample_kind", "tokens"),
            sample_count=payload.get("sample_count", 0),
            total=payload.get("total", 0),
            min_value=payload.get("min_value"),
            max_value=payload.get("max_value"),
            p50=payload.get("p50"),
            p90=payload.get("p90"),
            p95=payload.get("p95"),
            p99=payload.get("p99"),
        )

    @classmethod
    def empty(cls, sample_kind: str = "tokens") -> "IntegerPercentileSummary":
        return cls(
            sample_kind=sample_kind,
            sample_count=0,
            total=0,
            min_value=None,
            max_value=None,
            p50=None,
            p90=None,
            p95=None,
            p99=None,
        )


# ---------------------------------------------------------------------------
# Observation (one sealed receipt contribution)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetricsObservation:
    """Closed integer observation derived from one audit/run receipt.

    Token and cost fields may be ``None`` to signal *unavailable* sensors.
    Unavailable is never rewritten to zero success or zero savings.
    """

    observation_id: str
    receipt_cid: str
    cohort: MetricsCohort | str
    route_tier: RouteTier | str = RouteTier.MEDIUM
    comparative_outcome: ComparativeOutcome | str = (
        ComparativeOutcome.EQUIVALENT_SUCCESS
    )
    acceptance_disposition: AcceptanceDisposition | str = (
        AcceptanceDisposition.NOT_ACCEPTED
    )
    # Compression tokens (optional / unavailable-aware)
    raw_tokens: int | None = None
    retrieval_tokens: int | None = None
    compressed_tokens: int | None = None
    expanded_tokens: int | None = None
    # Quality flags
    accepted_patch: bool = False
    regression: bool = False
    selected_test_false_negative: bool = False
    proof_failure: bool = False
    review_disagreement: bool = False
    # Omission / expansion attribution
    intentional_omission_present: bool = False
    omission_detected_before_execution: bool = False
    omission_detected_after_execution: bool = False
    critical_omission: bool = False
    critical_omission_accepted: bool = False
    expansion_used: bool = False
    expansion_true_positive: bool = False
    expansion_false_positive: bool = False
    expansion_false_negative: bool = False
    # Routing
    escalated: bool = False
    retried: bool = False
    # Economics (micros / tokens; optional)
    input_tokens: int | None = None
    output_tokens: int | None = None
    baseline_model_spend_micros: int | None = None
    model_spend_micros: int | None = None
    verification_compute_micros: int | None = None
    shadow_compute_micros: int | None = None
    audit_overhead_micros: int | None = None
    # Calibration
    calibration_use: bool = False
    calibration_revision: int | None = None
    omission_failure: bool = False
    task_class: str | None = None
    partition: EvidencePartition | str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "observation_id",
            "receipt_cid",
            "cohort",
            "route_tier",
            "comparative_outcome",
            "acceptance_disposition",
            "raw_tokens",
            "retrieval_tokens",
            "compressed_tokens",
            "expanded_tokens",
            "accepted_patch",
            "regression",
            "selected_test_false_negative",
            "proof_failure",
            "review_disagreement",
            "intentional_omission_present",
            "omission_detected_before_execution",
            "omission_detected_after_execution",
            "critical_omission",
            "critical_omission_accepted",
            "expansion_used",
            "expansion_true_positive",
            "expansion_false_positive",
            "expansion_false_negative",
            "escalated",
            "retried",
            "input_tokens",
            "output_tokens",
            "baseline_model_spend_micros",
            "model_spend_micros",
            "verification_compute_micros",
            "shadow_compute_micros",
            "audit_overhead_micros",
            "calibration_use",
            "calibration_revision",
            "omission_failure",
            "task_class",
            "partition",
            "metadata",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observation_id", _token(self.observation_id, "observation_id")
        )
        object.__setattr__(self, "receipt_cid", _cid(self.receipt_cid, "receipt_cid"))
        object.__setattr__(self, "cohort", _cohort(self.cohort, "cohort"))
        object.__setattr__(self, "route_tier", _route_tier(self.route_tier))
        object.__setattr__(
            self,
            "comparative_outcome",
            _enum(self.comparative_outcome, ComparativeOutcome, "comparative_outcome"),
        )
        object.__setattr__(
            self,
            "acceptance_disposition",
            _enum(
                self.acceptance_disposition,
                AcceptanceDisposition,
                "acceptance_disposition",
            ),
        )
        for name in (
            "raw_tokens",
            "retrieval_tokens",
            "compressed_tokens",
            "expanded_tokens",
            "input_tokens",
            "output_tokens",
            "baseline_model_spend_micros",
            "model_spend_micros",
            "verification_compute_micros",
            "shadow_compute_micros",
            "audit_overhead_micros",
            "calibration_revision",
        ):
            object.__setattr__(
                self, name, _optional_nonneg_int(getattr(self, name), name)
            )
        if self.calibration_revision is not None and self.calibration_revision > MAX_REVISION:
            raise MetricsError("calibration_revision exceeds maximum")
        for name in (
            "accepted_patch",
            "regression",
            "selected_test_false_negative",
            "proof_failure",
            "review_disagreement",
            "intentional_omission_present",
            "omission_detected_before_execution",
            "omission_detected_after_execution",
            "critical_omission",
            "critical_omission_accepted",
            "expansion_used",
            "expansion_true_positive",
            "expansion_false_positive",
            "expansion_false_negative",
            "escalated",
            "retried",
            "calibration_use",
            "omission_failure",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        object.__setattr__(
            self, "task_class", _optional_token(self.task_class, "task_class")
        )
        if self.partition is None:
            object.__setattr__(self, "partition", None)
        else:
            object.__setattr__(
                self,
                "partition",
                _enum(self.partition, EvidencePartition, "partition"),
            )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        # Fail closed on contradictory expansion attribution.
        positive_flags = sum(
            1
            for flag in (
                self.expansion_true_positive,
                self.expansion_false_positive,
                self.expansion_false_negative,
            )
            if flag
        )
        if positive_flags > 1:
            raise MetricsError(
                "expansion true/false positive/negative flags are mutually exclusive"
            )
        if self.critical_omission_accepted and not self.critical_omission:
            raise MetricsError(
                "critical_omission_accepted requires critical_omission"
            )
        if self.accepted_patch and self.acceptance_disposition != (
            AcceptanceDisposition.ACCEPTED.value
        ):
            raise MetricsError(
                "accepted_patch requires acceptance_disposition=accepted"
            )
        if (
            self.acceptance_disposition == AcceptanceDisposition.ACCEPTED.value
            and not self.accepted_patch
        ):
            object.__setattr__(self, "accepted_patch", True)

    @property
    def is_simulated(self) -> bool:
        return self.cohort == MetricsCohort.SIMULATED.value

    @property
    def is_live(self) -> bool:
        return self.cohort == MetricsCohort.LIVE.value

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": METRICS_OBSERVATION_SCHEMA,
            "observation_id": self.observation_id,
            "receipt_cid": self.receipt_cid,
            "cohort": self.cohort,
            "route_tier": self.route_tier,
            "comparative_outcome": self.comparative_outcome,
            "acceptance_disposition": self.acceptance_disposition,
            "raw_tokens": self.raw_tokens,
            "retrieval_tokens": self.retrieval_tokens,
            "compressed_tokens": self.compressed_tokens,
            "expanded_tokens": self.expanded_tokens,
            "accepted_patch": self.accepted_patch,
            "regression": self.regression,
            "selected_test_false_negative": self.selected_test_false_negative,
            "proof_failure": self.proof_failure,
            "review_disagreement": self.review_disagreement,
            "intentional_omission_present": self.intentional_omission_present,
            "omission_detected_before_execution": (
                self.omission_detected_before_execution
            ),
            "omission_detected_after_execution": (
                self.omission_detected_after_execution
            ),
            "critical_omission": self.critical_omission,
            "critical_omission_accepted": self.critical_omission_accepted,
            "expansion_used": self.expansion_used,
            "expansion_true_positive": self.expansion_true_positive,
            "expansion_false_positive": self.expansion_false_positive,
            "expansion_false_negative": self.expansion_false_negative,
            "escalated": self.escalated,
            "retried": self.retried,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "baseline_model_spend_micros": self.baseline_model_spend_micros,
            "model_spend_micros": self.model_spend_micros,
            "verification_compute_micros": self.verification_compute_micros,
            "shadow_compute_micros": self.shadow_compute_micros,
            "audit_overhead_micros": self.audit_overhead_micros,
            "calibration_use": self.calibration_use,
            "calibration_revision": self.calibration_revision,
            "omission_failure": self.omission_failure,
            "task_class": self.task_class,
            "partition": self.partition,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def observation_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["observation_cid"] = self.observation_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricsObservation":
        payload = dict(data)
        payload.pop("observation_cid", None)
        schema = payload.pop("schema", METRICS_OBSERVATION_SCHEMA)
        if schema != METRICS_OBSERVATION_SCHEMA:
            raise MetricsError("unsupported MetricsObservation schema version")
        return cls(**payload)


def observation_from_receipt_fields(
    *,
    observation_id: str,
    receipt_cid: str,
    cohort: MetricsCohort | ExecutionMode | str = MetricsCohort.LIVE,
    route_tier: RouteTier | str = RouteTier.MEDIUM,
    comparative_outcome: ComparativeOutcome | str = (
        ComparativeOutcome.EQUIVALENT_SUCCESS
    ),
    acceptance_disposition: AcceptanceDisposition | str = (
        AcceptanceDisposition.NOT_ACCEPTED
    ),
    raw_tokens: int | None = None,
    retrieval_tokens: int | None = None,
    compressed_tokens: int | None = None,
    expanded_tokens: int | None = None,
    accepted_patch: bool | None = None,
    regression: bool = False,
    selected_test_false_negative: bool = False,
    proof_failure: bool = False,
    review_disagreement: bool = False,
    intentional_omission_present: bool = False,
    omission_detected_before_execution: bool = False,
    omission_detected_after_execution: bool = False,
    critical_omission: bool = False,
    critical_omission_accepted: bool = False,
    expansion_used: bool = False,
    expansion_true_positive: bool = False,
    expansion_false_positive: bool = False,
    expansion_false_negative: bool = False,
    escalated: bool = False,
    retried: bool = False,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    baseline_model_spend_micros: int | None = None,
    model_spend_micros: int | None = None,
    verification_compute_micros: int | None = None,
    shadow_compute_micros: int | None = None,
    audit_overhead_micros: int | None = None,
    calibration_use: bool = False,
    calibration_revision: int | None = None,
    omission_failure: bool = False,
    task_class: str | None = None,
    partition: EvidencePartition | str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MetricsObservation:
    """Build a metrics observation from GovernorRunReceipt-shaped fields."""

    disposition = _enum(
        acceptance_disposition, AcceptanceDisposition, "acceptance_disposition"
    )
    accepted = (
        disposition == AcceptanceDisposition.ACCEPTED.value
        if accepted_patch is None
        else accepted_patch
    )
    return MetricsObservation(
        observation_id=observation_id,
        receipt_cid=receipt_cid,
        cohort=cohort,
        route_tier=route_tier,
        comparative_outcome=comparative_outcome,
        acceptance_disposition=disposition,
        raw_tokens=raw_tokens,
        retrieval_tokens=retrieval_tokens,
        compressed_tokens=compressed_tokens,
        expanded_tokens=expanded_tokens,
        accepted_patch=bool(accepted),
        regression=regression,
        selected_test_false_negative=selected_test_false_negative,
        proof_failure=proof_failure,
        review_disagreement=review_disagreement,
        intentional_omission_present=intentional_omission_present,
        omission_detected_before_execution=omission_detected_before_execution,
        omission_detected_after_execution=omission_detected_after_execution,
        critical_omission=critical_omission,
        critical_omission_accepted=critical_omission_accepted,
        expansion_used=expansion_used,
        expansion_true_positive=expansion_true_positive,
        expansion_false_positive=expansion_false_positive,
        expansion_false_negative=expansion_false_negative,
        escalated=escalated,
        retried=retried,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        baseline_model_spend_micros=baseline_model_spend_micros,
        model_spend_micros=model_spend_micros,
        verification_compute_micros=verification_compute_micros,
        shadow_compute_micros=shadow_compute_micros,
        audit_overhead_micros=audit_overhead_micros,
        calibration_use=calibration_use,
        calibration_revision=calibration_revision,
        omission_failure=omission_failure,
        task_class=task_class,
        partition=partition,
        metadata=dict(metadata or {}),
    )


# ---------------------------------------------------------------------------
# Metric family summaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompressionMetrics:
    """Token distributions, reduction, and expansion rate for one cohort."""

    observation_count: int
    raw_tokens_total: int
    retrieval_tokens_total: int
    compressed_tokens_total: int
    expanded_tokens_total: int
    raw_tokens_samples: int
    retrieval_tokens_samples: int
    compressed_tokens_samples: int
    expanded_tokens_samples: int
    raw_tokens_percentiles: IntegerPercentileSummary
    retrieval_tokens_percentiles: IntegerPercentileSummary
    compressed_tokens_percentiles: IntegerPercentileSummary
    expanded_tokens_percentiles: IntegerPercentileSummary
    # Reduction vs raw: floor((raw - final) * 10000 / raw); None if raw missing.
    median_context_reduction_bp: int | None
    mean_context_reduction_bp: int | None
    expansion_count: int
    expansion_rate_bp: int | None
    unavailable_token_fields: int

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": COMPRESSION_METRICS_SCHEMA,
            "observation_count": self.observation_count,
            "raw_tokens_total": self.raw_tokens_total,
            "retrieval_tokens_total": self.retrieval_tokens_total,
            "compressed_tokens_total": self.compressed_tokens_total,
            "expanded_tokens_total": self.expanded_tokens_total,
            "raw_tokens_samples": self.raw_tokens_samples,
            "retrieval_tokens_samples": self.retrieval_tokens_samples,
            "compressed_tokens_samples": self.compressed_tokens_samples,
            "expanded_tokens_samples": self.expanded_tokens_samples,
            "raw_tokens_percentiles": self.raw_tokens_percentiles.to_dict(),
            "retrieval_tokens_percentiles": (
                self.retrieval_tokens_percentiles.to_dict()
            ),
            "compressed_tokens_percentiles": (
                self.compressed_tokens_percentiles.to_dict()
            ),
            "expanded_tokens_percentiles": (
                self.expanded_tokens_percentiles.to_dict()
            ),
            "median_context_reduction_bp": self.median_context_reduction_bp,
            "mean_context_reduction_bp": self.mean_context_reduction_bp,
            "expansion_count": self.expansion_count,
            "expansion_rate_bp": self.expansion_rate_bp,
            "unavailable_token_fields": self.unavailable_token_fields,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.identity_payload()

    @classmethod
    def empty(cls) -> "CompressionMetrics":
        return cls(
            observation_count=0,
            raw_tokens_total=0,
            retrieval_tokens_total=0,
            compressed_tokens_total=0,
            expanded_tokens_total=0,
            raw_tokens_samples=0,
            retrieval_tokens_samples=0,
            compressed_tokens_samples=0,
            expanded_tokens_samples=0,
            raw_tokens_percentiles=IntegerPercentileSummary.empty("raw_tokens"),
            retrieval_tokens_percentiles=IntegerPercentileSummary.empty(
                "retrieval_tokens"
            ),
            compressed_tokens_percentiles=IntegerPercentileSummary.empty(
                "compressed_tokens"
            ),
            expanded_tokens_percentiles=IntegerPercentileSummary.empty(
                "expanded_tokens"
            ),
            median_context_reduction_bp=None,
            mean_context_reduction_bp=None,
            expansion_count=0,
            expansion_rate_bp=None,
            unavailable_token_fields=0,
        )


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    """Accepted-patch quality, regressions, and outcome distribution."""

    observation_count: int
    accepted_patch_count: int
    regression_count: int
    selected_test_false_negative_count: int
    proof_failure_count: int
    review_disagreement_count: int
    accepted_rate_bp: int | None
    regression_rate_bp: int | None
    selected_test_false_negative_rate_bp: int | None
    proof_failure_rate_bp: int | None
    review_disagreement_rate_bp: int | None
    outcome_counts: Mapping[str, int]

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": QUALITY_METRICS_SCHEMA,
            "observation_count": self.observation_count,
            "accepted_patch_count": self.accepted_patch_count,
            "regression_count": self.regression_count,
            "selected_test_false_negative_count": (
                self.selected_test_false_negative_count
            ),
            "proof_failure_count": self.proof_failure_count,
            "review_disagreement_count": self.review_disagreement_count,
            "accepted_rate_bp": self.accepted_rate_bp,
            "regression_rate_bp": self.regression_rate_bp,
            "selected_test_false_negative_rate_bp": (
                self.selected_test_false_negative_rate_bp
            ),
            "proof_failure_rate_bp": self.proof_failure_rate_bp,
            "review_disagreement_rate_bp": self.review_disagreement_rate_bp,
            "outcome_counts": {
                key: self.outcome_counts[key] for key in _COMPARATIVE_OUTCOMES
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return self.identity_payload()

    @classmethod
    def empty(cls) -> "QualityMetrics":
        return cls(
            observation_count=0,
            accepted_patch_count=0,
            regression_count=0,
            selected_test_false_negative_count=0,
            proof_failure_count=0,
            review_disagreement_count=0,
            accepted_rate_bp=None,
            regression_rate_bp=None,
            selected_test_false_negative_rate_bp=None,
            proof_failure_rate_bp=None,
            review_disagreement_rate_bp=None,
            outcome_counts=MappingProxyType(_empty_outcome_counts()),
        )


@dataclass(frozen=True, slots=True)
class OmissionMetrics:
    """Intentional-omission detection, critical acceptance, expansion PR."""

    observation_count: int
    intentional_omission_count: int
    detected_before_execution_count: int
    detected_after_execution_count: int
    critical_omission_count: int
    critical_omissions_accepted_count: int
    false_alarm_count: int
    expansion_true_positive_count: int
    expansion_false_positive_count: int
    expansion_false_negative_count: int
    detection_before_rate_bp: int | None
    detection_after_rate_bp: int | None
    critical_acceptance_rate_bp: int | None
    expansion_precision_bp: int | None
    expansion_recall_bp: int | None
    empirical_omission_rate: EmpiricalRate | None

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": OMISSION_METRICS_SCHEMA,
            "observation_count": self.observation_count,
            "intentional_omission_count": self.intentional_omission_count,
            "detected_before_execution_count": self.detected_before_execution_count,
            "detected_after_execution_count": self.detected_after_execution_count,
            "critical_omission_count": self.critical_omission_count,
            "critical_omissions_accepted_count": (
                self.critical_omissions_accepted_count
            ),
            "false_alarm_count": self.false_alarm_count,
            "expansion_true_positive_count": self.expansion_true_positive_count,
            "expansion_false_positive_count": self.expansion_false_positive_count,
            "expansion_false_negative_count": self.expansion_false_negative_count,
            "detection_before_rate_bp": self.detection_before_rate_bp,
            "detection_after_rate_bp": self.detection_after_rate_bp,
            "critical_acceptance_rate_bp": self.critical_acceptance_rate_bp,
            "expansion_precision_bp": self.expansion_precision_bp,
            "expansion_recall_bp": self.expansion_recall_bp,
            "empirical_omission_rate": (
                None
                if self.empirical_omission_rate is None
                else self.empirical_omission_rate.to_dict()
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.identity_payload()

    @classmethod
    def empty(cls) -> "OmissionMetrics":
        return cls(
            observation_count=0,
            intentional_omission_count=0,
            detected_before_execution_count=0,
            detected_after_execution_count=0,
            critical_omission_count=0,
            critical_omissions_accepted_count=0,
            false_alarm_count=0,
            expansion_true_positive_count=0,
            expansion_false_positive_count=0,
            expansion_false_negative_count=0,
            detection_before_rate_bp=None,
            detection_after_rate_bp=None,
            critical_acceptance_rate_bp=None,
            expansion_precision_bp=None,
            expansion_recall_bp=None,
            empirical_omission_rate=None,
        )


@dataclass(frozen=True, slots=True)
class RoutingMetrics:
    """Route-tier share, escalation, and retry rates."""

    observation_count: int
    route_share_counts: Mapping[str, int]
    route_share_bp: Mapping[str, int | None]
    escalation_count: int
    retry_count: int
    escalation_rate_bp: int | None
    retry_rate_bp: int | None

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": ROUTING_METRICS_SCHEMA,
            "observation_count": self.observation_count,
            "route_share_counts": {
                tier: self.route_share_counts[tier] for tier in _ROUTE_TIERS
            },
            "route_share_bp": {
                tier: self.route_share_bp[tier] for tier in _ROUTE_TIERS
            },
            "escalation_count": self.escalation_count,
            "retry_count": self.retry_count,
            "escalation_rate_bp": self.escalation_rate_bp,
            "retry_rate_bp": self.retry_rate_bp,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.identity_payload()

    @classmethod
    def empty(cls) -> "RoutingMetrics":
        empty_share = _empty_route_share()
        return cls(
            observation_count=0,
            route_share_counts=MappingProxyType(empty_share),
            route_share_bp=MappingProxyType({tier: None for tier in _ROUTE_TIERS}),
            escalation_count=0,
            retry_count=0,
            escalation_rate_bp=None,
            retry_rate_bp=None,
        )


@dataclass(frozen=True, slots=True)
class EconomicMetrics:
    """Token spend, model cost, audit overhead, and gross/net savings.

    Net savings always subtract audit overhead (including verification and
    shadow compute). Missing cost sensors leave savings as ``None``.
    """

    observation_count: int
    input_tokens_total: int
    output_tokens_total: int
    input_tokens_samples: int
    output_tokens_samples: int
    model_spend_micros_total: int
    baseline_model_spend_micros_total: int
    verification_compute_micros_total: int
    shadow_compute_micros_total: int
    audit_overhead_micros_total: int
    # Explicit composite: audit + verification + shadow.
    total_audit_overhead_micros: int
    model_spend_samples: int
    baseline_spend_samples: int
    model_spend_percentiles: IntegerPercentileSummary
    gross_savings_micros: int | None
    net_savings_micros: int | None
    cost_per_accepted_patch_micros: int | None
    accepted_patch_count: int
    unavailable_cost_fields: int

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": ECONOMIC_METRICS_SCHEMA,
            "observation_count": self.observation_count,
            "input_tokens_total": self.input_tokens_total,
            "output_tokens_total": self.output_tokens_total,
            "input_tokens_samples": self.input_tokens_samples,
            "output_tokens_samples": self.output_tokens_samples,
            "model_spend_micros_total": self.model_spend_micros_total,
            "baseline_model_spend_micros_total": (
                self.baseline_model_spend_micros_total
            ),
            "verification_compute_micros_total": (
                self.verification_compute_micros_total
            ),
            "shadow_compute_micros_total": self.shadow_compute_micros_total,
            "audit_overhead_micros_total": self.audit_overhead_micros_total,
            "total_audit_overhead_micros": self.total_audit_overhead_micros,
            "model_spend_samples": self.model_spend_samples,
            "baseline_spend_samples": self.baseline_spend_samples,
            "model_spend_percentiles": self.model_spend_percentiles.to_dict(),
            "gross_savings_micros": self.gross_savings_micros,
            "net_savings_micros": self.net_savings_micros,
            "cost_per_accepted_patch_micros": self.cost_per_accepted_patch_micros,
            "accepted_patch_count": self.accepted_patch_count,
            "unavailable_cost_fields": self.unavailable_cost_fields,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.identity_payload()

    @classmethod
    def empty(cls) -> "EconomicMetrics":
        return cls(
            observation_count=0,
            input_tokens_total=0,
            output_tokens_total=0,
            input_tokens_samples=0,
            output_tokens_samples=0,
            model_spend_micros_total=0,
            baseline_model_spend_micros_total=0,
            verification_compute_micros_total=0,
            shadow_compute_micros_total=0,
            audit_overhead_micros_total=0,
            total_audit_overhead_micros=0,
            model_spend_samples=0,
            baseline_spend_samples=0,
            model_spend_percentiles=IntegerPercentileSummary.empty(
                "model_spend_micros"
            ),
            gross_savings_micros=None,
            net_savings_micros=None,
            cost_per_accepted_patch_micros=None,
            accepted_patch_count=0,
            unavailable_cost_fields=0,
        )


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    """Calibration uses, omission-rate CI, revision, and task coverage."""

    observation_count: int
    calibration_use_count: int
    empirical_omission_rate: EmpiricalRate | None
    last_revision: int | None
    task_classes_observed: Sequence[str]
    task_class_counts: Mapping[str, int]
    task_coverage_count: int
    partition_counts: Mapping[str, int]

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": CALIBRATION_METRICS_SCHEMA,
            "observation_count": self.observation_count,
            "calibration_use_count": self.calibration_use_count,
            "empirical_omission_rate": (
                None
                if self.empirical_omission_rate is None
                else self.empirical_omission_rate.to_dict()
            ),
            "last_revision": self.last_revision,
            "task_classes_observed": list(self.task_classes_observed),
            "task_class_counts": {
                key: self.task_class_counts[key]
                for key in self.task_classes_observed
            },
            "task_coverage_count": self.task_coverage_count,
            "partition_counts": dict(self.partition_counts),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.identity_payload()

    @classmethod
    def empty(cls) -> "CalibrationMetrics":
        return cls(
            observation_count=0,
            calibration_use_count=0,
            empirical_omission_rate=None,
            last_revision=None,
            task_classes_observed=(),
            task_class_counts=MappingProxyType({}),
            task_coverage_count=0,
            partition_counts=MappingProxyType({}),
        )


# ---------------------------------------------------------------------------
# Cohort aggregate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CohortMetrics:
    """Full metric family bundle for one live or simulated cohort."""

    cohort: MetricsCohort | str
    observation_count: int
    source_receipt_cids: Sequence[str]
    compression: CompressionMetrics
    quality: QualityMetrics
    omission: OmissionMetrics
    routing: RoutingMetrics
    economic: EconomicMetrics
    calibration: CalibrationMetrics

    def __post_init__(self) -> None:
        object.__setattr__(self, "cohort", _cohort(self.cohort, "cohort"))
        object.__setattr__(
            self,
            "observation_count",
            _nonneg_int(self.observation_count, "observation_count"),
        )
        object.__setattr__(
            self,
            "source_receipt_cids",
            _unique_sorted_cids(
                list(self.source_receipt_cids), "source_receipt_cids"
            ),
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": COHORT_METRICS_SCHEMA,
            "cohort": self.cohort,
            "observation_count": self.observation_count,
            "source_receipt_cids": list(self.source_receipt_cids),
            "compression": self.compression.to_dict(),
            "quality": self.quality.to_dict(),
            "omission": self.omission.to_dict(),
            "routing": self.routing.to_dict(),
            "economic": self.economic.to_dict(),
            "calibration": self.calibration.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.identity_payload()

    @classmethod
    def empty(cls, cohort: MetricsCohort | str) -> "CohortMetrics":
        return cls(
            cohort=cohort,
            observation_count=0,
            source_receipt_cids=(),
            compression=CompressionMetrics.empty(),
            quality=QualityMetrics.empty(),
            omission=OmissionMetrics.empty(),
            routing=RoutingMetrics.empty(),
            economic=EconomicMetrics.empty(),
            calibration=CalibrationMetrics.empty(),
        )


# ---------------------------------------------------------------------------
# Aggregation internals
# ---------------------------------------------------------------------------


@dataclass
class _CohortAccumulator:
    """Mutable per-cohort sample store (not durable)."""

    cohort: str
    receipt_cids: list[str] = field(default_factory=list)
    # Compression samples
    raw_tokens: list[int] = field(default_factory=list)
    retrieval_tokens: list[int] = field(default_factory=list)
    compressed_tokens: list[int] = field(default_factory=list)
    expanded_tokens: list[int] = field(default_factory=list)
    reduction_bp_samples: list[int] = field(default_factory=list)
    expansion_count: int = 0
    unavailable_token_fields: int = 0
    # Quality
    accepted_patch_count: int = 0
    regression_count: int = 0
    selected_test_false_negative_count: int = 0
    proof_failure_count: int = 0
    review_disagreement_count: int = 0
    outcome_counts: dict[str, int] = field(default_factory=_empty_outcome_counts)
    # Omission
    intentional_omission_count: int = 0
    detected_before_execution_count: int = 0
    detected_after_execution_count: int = 0
    critical_omission_count: int = 0
    critical_omissions_accepted_count: int = 0
    false_alarm_count: int = 0
    expansion_true_positive_count: int = 0
    expansion_false_positive_count: int = 0
    expansion_false_negative_count: int = 0
    omission_failure_count: int = 0
    # Routing
    route_share_counts: dict[str, int] = field(default_factory=_empty_route_share)
    escalation_count: int = 0
    retry_count: int = 0
    # Economics
    input_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    model_spend: list[int] = field(default_factory=list)
    baseline_spend: list[int] = field(default_factory=list)
    verification_compute_total: int = 0
    shadow_compute_total: int = 0
    audit_overhead_total: int = 0
    paired_baseline_spend_total: int = 0
    paired_model_spend_total: int = 0
    paired_cost_samples: int = 0
    unavailable_cost_fields: int = 0
    # Calibration
    calibration_use_count: int = 0
    last_revision: int | None = None
    task_class_counts: dict[str, int] = field(default_factory=dict)
    partition_counts: dict[str, int] = field(default_factory=dict)

    @property
    def observation_count(self) -> int:
        return len(self.receipt_cids)

    def ingest(self, obs: MetricsObservation) -> None:
        self.receipt_cids.append(obs.receipt_cid)

        # Compression
        for attr, bucket in (
            ("raw_tokens", self.raw_tokens),
            ("retrieval_tokens", self.retrieval_tokens),
            ("compressed_tokens", self.compressed_tokens),
            ("expanded_tokens", self.expanded_tokens),
        ):
            value = getattr(obs, attr)
            if value is None:
                self.unavailable_token_fields = _add_counter(
                    self.unavailable_token_fields, 1, "unavailable_token_fields"
                )
            else:
                bucket.append(value)
        if obs.raw_tokens is not None and obs.compressed_tokens is not None:
            raw = obs.raw_tokens
            final = obs.compressed_tokens
            if obs.expansion_used and obs.expanded_tokens is not None:
                final = obs.expanded_tokens
            if raw > 0:
                # Clamp reduction to [0, 10000]; expansion beyond raw is 0 reduction.
                saved = raw - final
                if saved < 0:
                    saved = 0
                if saved > raw:
                    saved = raw
                reduction = (saved * BASIS_POINTS) // raw
                self.reduction_bp_samples.append(reduction)
        if obs.expansion_used:
            self.expansion_count = _add_counter(
                self.expansion_count, 1, "expansion_count"
            )

        # Quality
        if obs.accepted_patch:
            self.accepted_patch_count = _add_counter(
                self.accepted_patch_count, 1, "accepted_patch_count"
            )
        if obs.regression:
            self.regression_count = _add_counter(
                self.regression_count, 1, "regression_count"
            )
        if obs.selected_test_false_negative:
            self.selected_test_false_negative_count = _add_counter(
                self.selected_test_false_negative_count,
                1,
                "selected_test_false_negative_count",
            )
        if obs.proof_failure:
            self.proof_failure_count = _add_counter(
                self.proof_failure_count, 1, "proof_failure_count"
            )
        if obs.review_disagreement:
            self.review_disagreement_count = _add_counter(
                self.review_disagreement_count, 1, "review_disagreement_count"
            )
        outcome = obs.comparative_outcome
        self.outcome_counts[outcome] = _add_counter(
            self.outcome_counts.get(outcome, 0), 1, "outcome_counts"
        )

        # Omission
        if obs.intentional_omission_present:
            self.intentional_omission_count = _add_counter(
                self.intentional_omission_count, 1, "intentional_omission_count"
            )
        if obs.omission_detected_before_execution:
            self.detected_before_execution_count = _add_counter(
                self.detected_before_execution_count,
                1,
                "detected_before_execution_count",
            )
        if obs.omission_detected_after_execution:
            self.detected_after_execution_count = _add_counter(
                self.detected_after_execution_count,
                1,
                "detected_after_execution_count",
            )
        if obs.critical_omission:
            self.critical_omission_count = _add_counter(
                self.critical_omission_count, 1, "critical_omission_count"
            )
        if obs.critical_omission_accepted:
            self.critical_omissions_accepted_count = _add_counter(
                self.critical_omissions_accepted_count,
                1,
                "critical_omissions_accepted_count",
            )
        # False alarm: expansion or detection without intentional omission.
        if (
            obs.expansion_false_positive
            or (
                not obs.intentional_omission_present
                and (
                    obs.omission_detected_before_execution
                    or obs.omission_detected_after_execution
                )
            )
        ):
            self.false_alarm_count = _add_counter(
                self.false_alarm_count, 1, "false_alarm_count"
            )
        if obs.expansion_true_positive:
            self.expansion_true_positive_count = _add_counter(
                self.expansion_true_positive_count,
                1,
                "expansion_true_positive_count",
            )
        if obs.expansion_false_positive:
            self.expansion_false_positive_count = _add_counter(
                self.expansion_false_positive_count,
                1,
                "expansion_false_positive_count",
            )
        if obs.expansion_false_negative:
            self.expansion_false_negative_count = _add_counter(
                self.expansion_false_negative_count,
                1,
                "expansion_false_negative_count",
            )
        if obs.omission_failure:
            self.omission_failure_count = _add_counter(
                self.omission_failure_count, 1, "omission_failure_count"
            )

        # Routing
        tier = obs.route_tier
        self.route_share_counts[tier] = _add_counter(
            self.route_share_counts.get(tier, 0), 1, "route_share_counts"
        )
        if obs.escalated:
            self.escalation_count = _add_counter(
                self.escalation_count, 1, "escalation_count"
            )
        if obs.retried:
            self.retry_count = _add_counter(self.retry_count, 1, "retry_count")

        # Economics
        if obs.input_tokens is None:
            self.unavailable_cost_fields = _add_counter(
                self.unavailable_cost_fields, 1, "unavailable_cost_fields"
            )
        else:
            self.input_tokens.append(obs.input_tokens)
        if obs.output_tokens is None:
            self.unavailable_cost_fields = _add_counter(
                self.unavailable_cost_fields, 1, "unavailable_cost_fields"
            )
        else:
            self.output_tokens.append(obs.output_tokens)
        if obs.model_spend_micros is None:
            self.unavailable_cost_fields = _add_counter(
                self.unavailable_cost_fields, 1, "unavailable_cost_fields"
            )
        else:
            self.model_spend.append(obs.model_spend_micros)
        if obs.baseline_model_spend_micros is None:
            self.unavailable_cost_fields = _add_counter(
                self.unavailable_cost_fields, 1, "unavailable_cost_fields"
            )
        else:
            self.baseline_spend.append(obs.baseline_model_spend_micros)
        # Paired savings require both baseline and actual spend sensors.
        if (
            obs.baseline_model_spend_micros is not None
            and obs.model_spend_micros is not None
        ):
            self.paired_baseline_spend_total = _add_counter(
                self.paired_baseline_spend_total,
                obs.baseline_model_spend_micros,
                "paired_baseline_spend_total",
            )
            self.paired_model_spend_total = _add_counter(
                self.paired_model_spend_total,
                obs.model_spend_micros,
                "paired_model_spend_total",
            )
            self.paired_cost_samples = _add_counter(
                self.paired_cost_samples, 1, "paired_cost_samples"
            )
        # Overhead components: missing → count as 0 contribution for totals
        # only when the field is explicitly measured as 0; None is unavailable
        # and does not invent audit cost, but also is not treated as free.
        # For net savings we only subtract measured overhead components.
        if obs.verification_compute_micros is not None:
            self.verification_compute_total = _add_counter(
                self.verification_compute_total,
                obs.verification_compute_micros,
                "verification_compute_total",
            )
        else:
            self.unavailable_cost_fields = _add_counter(
                self.unavailable_cost_fields, 1, "unavailable_cost_fields"
            )
        if obs.shadow_compute_micros is not None:
            self.shadow_compute_total = _add_counter(
                self.shadow_compute_total,
                obs.shadow_compute_micros,
                "shadow_compute_total",
            )
        else:
            self.unavailable_cost_fields = _add_counter(
                self.unavailable_cost_fields, 1, "unavailable_cost_fields"
            )
        if obs.audit_overhead_micros is not None:
            self.audit_overhead_total = _add_counter(
                self.audit_overhead_total,
                obs.audit_overhead_micros,
                "audit_overhead_total",
            )
        else:
            self.unavailable_cost_fields = _add_counter(
                self.unavailable_cost_fields, 1, "unavailable_cost_fields"
            )

        # Calibration
        if obs.calibration_use:
            self.calibration_use_count = _add_counter(
                self.calibration_use_count, 1, "calibration_use_count"
            )
        if obs.calibration_revision is not None:
            if (
                self.last_revision is None
                or obs.calibration_revision > self.last_revision
            ):
                self.last_revision = obs.calibration_revision
        if obs.task_class is not None:
            if (
                obs.task_class not in self.task_class_counts
                and len(self.task_class_counts) >= MAX_TASK_CLASSES
            ):
                raise MetricsError("task_class_counts exceeds maximum")
            self.task_class_counts[obs.task_class] = _add_counter(
                self.task_class_counts.get(obs.task_class, 0),
                1,
                "task_class_counts",
            )
        if obs.partition is not None:
            self.partition_counts[obs.partition] = _add_counter(
                self.partition_counts.get(obs.partition, 0),
                1,
                "partition_counts",
            )

    def finalize(self) -> CohortMetrics:
        n = self.observation_count

        compression = CompressionMetrics(
            observation_count=n,
            raw_tokens_total=sum(self.raw_tokens),
            retrieval_tokens_total=sum(self.retrieval_tokens),
            compressed_tokens_total=sum(self.compressed_tokens),
            expanded_tokens_total=sum(self.expanded_tokens),
            raw_tokens_samples=len(self.raw_tokens),
            retrieval_tokens_samples=len(self.retrieval_tokens),
            compressed_tokens_samples=len(self.compressed_tokens),
            expanded_tokens_samples=len(self.expanded_tokens),
            raw_tokens_percentiles=build_percentile_summary(
                self.raw_tokens, sample_kind="raw_tokens"
            ),
            retrieval_tokens_percentiles=build_percentile_summary(
                self.retrieval_tokens, sample_kind="retrieval_tokens"
            ),
            compressed_tokens_percentiles=build_percentile_summary(
                self.compressed_tokens, sample_kind="compressed_tokens"
            ),
            expanded_tokens_percentiles=build_percentile_summary(
                self.expanded_tokens, sample_kind="expanded_tokens"
            ),
            median_context_reduction_bp=nearest_rank_percentile(
                self.reduction_bp_samples, 5_000
            ),
            mean_context_reduction_bp=(
                None
                if not self.reduction_bp_samples
                else sum(self.reduction_bp_samples) // len(self.reduction_bp_samples)
            ),
            expansion_count=self.expansion_count,
            expansion_rate_bp=_rate_bp(self.expansion_count, n) if n else None,
            unavailable_token_fields=self.unavailable_token_fields,
        )

        quality = QualityMetrics(
            observation_count=n,
            accepted_patch_count=self.accepted_patch_count,
            regression_count=self.regression_count,
            selected_test_false_negative_count=(
                self.selected_test_false_negative_count
            ),
            proof_failure_count=self.proof_failure_count,
            review_disagreement_count=self.review_disagreement_count,
            accepted_rate_bp=_rate_bp(self.accepted_patch_count, n) if n else None,
            regression_rate_bp=_rate_bp(self.regression_count, n) if n else None,
            selected_test_false_negative_rate_bp=(
                _rate_bp(self.selected_test_false_negative_count, n) if n else None
            ),
            proof_failure_rate_bp=(
                _rate_bp(self.proof_failure_count, n) if n else None
            ),
            review_disagreement_rate_bp=(
                _rate_bp(self.review_disagreement_count, n) if n else None
            ),
            outcome_counts=MappingProxyType(
                {key: self.outcome_counts.get(key, 0) for key in _COMPARATIVE_OUTCOMES}
            ),
        )

        intentional = self.intentional_omission_count
        tp = self.expansion_true_positive_count
        fp = self.expansion_false_positive_count
        fn = self.expansion_false_negative_count
        precision_den = tp + fp
        recall_den = tp + fn
        omission = OmissionMetrics(
            observation_count=n,
            intentional_omission_count=intentional,
            detected_before_execution_count=self.detected_before_execution_count,
            detected_after_execution_count=self.detected_after_execution_count,
            critical_omission_count=self.critical_omission_count,
            critical_omissions_accepted_count=(
                self.critical_omissions_accepted_count
            ),
            false_alarm_count=self.false_alarm_count,
            expansion_true_positive_count=tp,
            expansion_false_positive_count=fp,
            expansion_false_negative_count=fn,
            detection_before_rate_bp=(
                _rate_bp(self.detected_before_execution_count, intentional)
                if intentional
                else None
            ),
            detection_after_rate_bp=(
                _rate_bp(self.detected_after_execution_count, intentional)
                if intentional
                else None
            ),
            critical_acceptance_rate_bp=(
                _rate_bp(
                    self.critical_omissions_accepted_count,
                    self.critical_omission_count,
                )
                if self.critical_omission_count
                else None
            ),
            expansion_precision_bp=(
                _rate_bp(tp, precision_den) if precision_den else None
            ),
            expansion_recall_bp=_rate_bp(tp, recall_den) if recall_den else None,
            empirical_omission_rate=(
                build_empirical_rate(self.omission_failure_count, n) if n else None
            ),
        )

        route_counts = {
            tier: self.route_share_counts.get(tier, 0) for tier in _ROUTE_TIERS
        }
        routing = RoutingMetrics(
            observation_count=n,
            route_share_counts=MappingProxyType(route_counts),
            route_share_bp=MappingProxyType(
                {
                    tier: (_rate_bp(route_counts[tier], n) if n else None)
                    for tier in _ROUTE_TIERS
                }
            ),
            escalation_count=self.escalation_count,
            retry_count=self.retry_count,
            escalation_rate_bp=_rate_bp(self.escalation_count, n) if n else None,
            retry_rate_bp=_rate_bp(self.retry_count, n) if n else None,
        )

        model_spend_total = sum(self.model_spend)
        baseline_total = sum(self.baseline_spend)
        total_audit_overhead = (
            self.audit_overhead_total
            + self.verification_compute_total
            + self.shadow_compute_total
        )
        if self.paired_cost_samples > 0:
            gross = self.paired_baseline_spend_total - self.paired_model_spend_total
            # Gross savings may be negative if compressed spend exceeded baseline.
            net = gross - total_audit_overhead
            gross_savings: int | None = gross
            net_savings: int | None = net
        else:
            gross_savings = None
            net_savings = None
        accepted = self.accepted_patch_count
        if accepted > 0 and self.model_spend:
            cost_per_accepted: int | None = model_spend_total // accepted
        else:
            cost_per_accepted = None

        economic = EconomicMetrics(
            observation_count=n,
            input_tokens_total=sum(self.input_tokens),
            output_tokens_total=sum(self.output_tokens),
            input_tokens_samples=len(self.input_tokens),
            output_tokens_samples=len(self.output_tokens),
            model_spend_micros_total=model_spend_total,
            baseline_model_spend_micros_total=baseline_total,
            verification_compute_micros_total=self.verification_compute_total,
            shadow_compute_micros_total=self.shadow_compute_total,
            audit_overhead_micros_total=self.audit_overhead_total,
            total_audit_overhead_micros=total_audit_overhead,
            model_spend_samples=len(self.model_spend),
            baseline_spend_samples=len(self.baseline_spend),
            model_spend_percentiles=build_percentile_summary(
                self.model_spend, sample_kind="model_spend_micros"
            ),
            gross_savings_micros=gross_savings,
            net_savings_micros=net_savings,
            cost_per_accepted_patch_micros=cost_per_accepted,
            accepted_patch_count=accepted,
            unavailable_cost_fields=self.unavailable_cost_fields,
        )

        task_classes = tuple(sorted(self.task_class_counts))
        calibration = CalibrationMetrics(
            observation_count=n,
            calibration_use_count=self.calibration_use_count,
            empirical_omission_rate=(
                build_empirical_rate(self.omission_failure_count, n) if n else None
            ),
            last_revision=self.last_revision,
            task_classes_observed=task_classes,
            task_class_counts=MappingProxyType(
                {key: self.task_class_counts[key] for key in task_classes}
            ),
            task_coverage_count=len(task_classes),
            partition_counts=MappingProxyType(dict(sorted(self.partition_counts.items()))),
        )

        return CohortMetrics(
            cohort=self.cohort,
            observation_count=n,
            source_receipt_cids=tuple(self.receipt_cids),
            compression=compression,
            quality=quality,
            omission=omission,
            routing=routing,
            economic=economic,
            calibration=calibration,
        )


# ---------------------------------------------------------------------------
# Report + collector
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GovernorMetricReport:
    """Sealed dual-cohort metrics report with content identity.

    Live and simulated cohorts are always present as separate bundles so
    simulated quality cannot contaminate live claims.
    """

    live: CohortMetrics
    simulated: CohortMetrics
    total_observations: int
    applied_count: int
    skipped_idempotent_count: int
    source_receipt_cids: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.live.cohort != MetricsCohort.LIVE.value:
            raise MetricsError("live cohort metrics must have cohort=live")
        if self.simulated.cohort != MetricsCohort.SIMULATED.value:
            raise MetricsError(
                "simulated cohort metrics must have cohort=simulated"
            )
        object.__setattr__(
            self,
            "total_observations",
            _nonneg_int(self.total_observations, "total_observations"),
        )
        object.__setattr__(
            self, "applied_count", _nonneg_int(self.applied_count, "applied_count")
        )
        object.__setattr__(
            self,
            "skipped_idempotent_count",
            _nonneg_int(self.skipped_idempotent_count, "skipped_idempotent_count"),
        )
        object.__setattr__(
            self,
            "source_receipt_cids",
            _unique_sorted_cids(
                list(self.source_receipt_cids), "source_receipt_cids"
            ),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        expected_total = self.live.observation_count + self.simulated.observation_count
        if self.total_observations != expected_total:
            raise MetricsError(
                "total_observations must equal live + simulated observation counts"
            )
        if self.applied_count != expected_total:
            raise MetricsError(
                "applied_count must equal live + simulated observation counts"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": GOVERNOR_METRIC_REPORT_SCHEMA,
            "evidence": SCG_METRICS_EVIDENCE,
            "interface_id": GOVERNOR_METRIC_REPORT_INTERFACE,
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "producer_id": PRODUCER_ID,
            "producer_version": PRODUCER_VERSION,
            "tool_id": TOOL_ID,
            "live": self.live.to_dict(),
            "simulated": self.simulated.to_dict(),
            "total_observations": self.total_observations,
            "applied_count": self.applied_count,
            "skipped_idempotent_count": self.skipped_idempotent_count,
            "source_receipt_cids": list(self.source_receipt_cids),
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def report_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["report_cid"] = self.report_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GovernorMetricReport":
        payload = dict(data)
        claimed = payload.pop("report_cid", None)
        schema = payload.get("schema", GOVERNOR_METRIC_REPORT_SCHEMA)
        if schema != GOVERNOR_METRIC_REPORT_SCHEMA:
            raise MetricsError("unsupported GovernorMetricReport schema version")
        live_raw = payload.get("live") or {}
        sim_raw = payload.get("simulated") or {}
        report = cls(
            live=_cohort_from_dict(live_raw, MetricsCohort.LIVE.value),
            simulated=_cohort_from_dict(sim_raw, MetricsCohort.SIMULATED.value),
            total_observations=payload.get("total_observations", 0),
            applied_count=payload.get("applied_count", 0),
            skipped_idempotent_count=payload.get("skipped_idempotent_count", 0),
            source_receipt_cids=tuple(payload.get("source_receipt_cids") or ()),
            metadata=dict(payload.get("metadata") or {}),
        )
        if claimed is not None and claimed != report.report_cid:
            raise MetricsError("GovernorMetricReport report_cid does not verify")
        return report


def _percentile_from_dict(data: Any, sample_kind: str) -> IntegerPercentileSummary:
    if isinstance(data, IntegerPercentileSummary):
        return data
    if isinstance(data, Mapping):
        return IntegerPercentileSummary.from_dict(data)
    return IntegerPercentileSummary.empty(sample_kind)


def _empirical_from_dict(data: Any) -> EmpiricalRate | None:
    if data is None:
        return None
    if isinstance(data, EmpiricalRate):
        return data
    if isinstance(data, Mapping):
        return EmpiricalRate.from_dict(data)
    raise MetricsError("empirical rate must be EmpiricalRate, mapping, or null")


def _cohort_from_dict(data: Mapping[str, Any], expected_cohort: str) -> CohortMetrics:
    if not isinstance(data, Mapping):
        raise MetricsError("cohort metrics must be a mapping")
    cohort = data.get("cohort", expected_cohort)
    if _cohort(cohort) != expected_cohort:
        raise MetricsError(f"expected cohort {expected_cohort}, got {cohort!r}")
    compression_raw = dict(data.get("compression") or {})
    quality_raw = dict(data.get("quality") or {})
    omission_raw = dict(data.get("omission") or {})
    routing_raw = dict(data.get("routing") or {})
    economic_raw = dict(data.get("economic") or {})
    calibration_raw = dict(data.get("calibration") or {})

    compression = CompressionMetrics(
        observation_count=int(compression_raw.get("observation_count", 0)),
        raw_tokens_total=int(compression_raw.get("raw_tokens_total", 0)),
        retrieval_tokens_total=int(compression_raw.get("retrieval_tokens_total", 0)),
        compressed_tokens_total=int(
            compression_raw.get("compressed_tokens_total", 0)
        ),
        expanded_tokens_total=int(compression_raw.get("expanded_tokens_total", 0)),
        raw_tokens_samples=int(compression_raw.get("raw_tokens_samples", 0)),
        retrieval_tokens_samples=int(
            compression_raw.get("retrieval_tokens_samples", 0)
        ),
        compressed_tokens_samples=int(
            compression_raw.get("compressed_tokens_samples", 0)
        ),
        expanded_tokens_samples=int(
            compression_raw.get("expanded_tokens_samples", 0)
        ),
        raw_tokens_percentiles=_percentile_from_dict(
            compression_raw.get("raw_tokens_percentiles"), "raw_tokens"
        ),
        retrieval_tokens_percentiles=_percentile_from_dict(
            compression_raw.get("retrieval_tokens_percentiles"), "retrieval_tokens"
        ),
        compressed_tokens_percentiles=_percentile_from_dict(
            compression_raw.get("compressed_tokens_percentiles"),
            "compressed_tokens",
        ),
        expanded_tokens_percentiles=_percentile_from_dict(
            compression_raw.get("expanded_tokens_percentiles"), "expanded_tokens"
        ),
        median_context_reduction_bp=compression_raw.get(
            "median_context_reduction_bp"
        ),
        mean_context_reduction_bp=compression_raw.get("mean_context_reduction_bp"),
        expansion_count=int(compression_raw.get("expansion_count", 0)),
        expansion_rate_bp=compression_raw.get("expansion_rate_bp"),
        unavailable_token_fields=int(
            compression_raw.get("unavailable_token_fields", 0)
        ),
    )

    outcome_counts = dict(quality_raw.get("outcome_counts") or _empty_outcome_counts())
    for key in _COMPARATIVE_OUTCOMES:
        outcome_counts.setdefault(key, 0)
    quality = QualityMetrics(
        observation_count=int(quality_raw.get("observation_count", 0)),
        accepted_patch_count=int(quality_raw.get("accepted_patch_count", 0)),
        regression_count=int(quality_raw.get("regression_count", 0)),
        selected_test_false_negative_count=int(
            quality_raw.get("selected_test_false_negative_count", 0)
        ),
        proof_failure_count=int(quality_raw.get("proof_failure_count", 0)),
        review_disagreement_count=int(
            quality_raw.get("review_disagreement_count", 0)
        ),
        accepted_rate_bp=quality_raw.get("accepted_rate_bp"),
        regression_rate_bp=quality_raw.get("regression_rate_bp"),
        selected_test_false_negative_rate_bp=quality_raw.get(
            "selected_test_false_negative_rate_bp"
        ),
        proof_failure_rate_bp=quality_raw.get("proof_failure_rate_bp"),
        review_disagreement_rate_bp=quality_raw.get("review_disagreement_rate_bp"),
        outcome_counts=MappingProxyType(
            {key: int(outcome_counts[key]) for key in _COMPARATIVE_OUTCOMES}
        ),
    )

    omission = OmissionMetrics(
        observation_count=int(omission_raw.get("observation_count", 0)),
        intentional_omission_count=int(
            omission_raw.get("intentional_omission_count", 0)
        ),
        detected_before_execution_count=int(
            omission_raw.get("detected_before_execution_count", 0)
        ),
        detected_after_execution_count=int(
            omission_raw.get("detected_after_execution_count", 0)
        ),
        critical_omission_count=int(omission_raw.get("critical_omission_count", 0)),
        critical_omissions_accepted_count=int(
            omission_raw.get("critical_omissions_accepted_count", 0)
        ),
        false_alarm_count=int(omission_raw.get("false_alarm_count", 0)),
        expansion_true_positive_count=int(
            omission_raw.get("expansion_true_positive_count", 0)
        ),
        expansion_false_positive_count=int(
            omission_raw.get("expansion_false_positive_count", 0)
        ),
        expansion_false_negative_count=int(
            omission_raw.get("expansion_false_negative_count", 0)
        ),
        detection_before_rate_bp=omission_raw.get("detection_before_rate_bp"),
        detection_after_rate_bp=omission_raw.get("detection_after_rate_bp"),
        critical_acceptance_rate_bp=omission_raw.get("critical_acceptance_rate_bp"),
        expansion_precision_bp=omission_raw.get("expansion_precision_bp"),
        expansion_recall_bp=omission_raw.get("expansion_recall_bp"),
        empirical_omission_rate=_empirical_from_dict(
            omission_raw.get("empirical_omission_rate")
        ),
    )

    route_counts = dict(routing_raw.get("route_share_counts") or _empty_route_share())
    for tier in _ROUTE_TIERS:
        route_counts.setdefault(tier, 0)
    route_bp_raw = dict(routing_raw.get("route_share_bp") or {})
    routing = RoutingMetrics(
        observation_count=int(routing_raw.get("observation_count", 0)),
        route_share_counts=MappingProxyType(
            {tier: int(route_counts[tier]) for tier in _ROUTE_TIERS}
        ),
        route_share_bp=MappingProxyType(
            {tier: route_bp_raw.get(tier) for tier in _ROUTE_TIERS}
        ),
        escalation_count=int(routing_raw.get("escalation_count", 0)),
        retry_count=int(routing_raw.get("retry_count", 0)),
        escalation_rate_bp=routing_raw.get("escalation_rate_bp"),
        retry_rate_bp=routing_raw.get("retry_rate_bp"),
    )

    economic = EconomicMetrics(
        observation_count=int(economic_raw.get("observation_count", 0)),
        input_tokens_total=int(economic_raw.get("input_tokens_total", 0)),
        output_tokens_total=int(economic_raw.get("output_tokens_total", 0)),
        input_tokens_samples=int(economic_raw.get("input_tokens_samples", 0)),
        output_tokens_samples=int(economic_raw.get("output_tokens_samples", 0)),
        model_spend_micros_total=int(
            economic_raw.get("model_spend_micros_total", 0)
        ),
        baseline_model_spend_micros_total=int(
            economic_raw.get("baseline_model_spend_micros_total", 0)
        ),
        verification_compute_micros_total=int(
            economic_raw.get("verification_compute_micros_total", 0)
        ),
        shadow_compute_micros_total=int(
            economic_raw.get("shadow_compute_micros_total", 0)
        ),
        audit_overhead_micros_total=int(
            economic_raw.get("audit_overhead_micros_total", 0)
        ),
        total_audit_overhead_micros=int(
            economic_raw.get("total_audit_overhead_micros", 0)
        ),
        model_spend_samples=int(economic_raw.get("model_spend_samples", 0)),
        baseline_spend_samples=int(economic_raw.get("baseline_spend_samples", 0)),
        model_spend_percentiles=_percentile_from_dict(
            economic_raw.get("model_spend_percentiles"), "model_spend_micros"
        ),
        gross_savings_micros=economic_raw.get("gross_savings_micros"),
        net_savings_micros=economic_raw.get("net_savings_micros"),
        cost_per_accepted_patch_micros=economic_raw.get(
            "cost_per_accepted_patch_micros"
        ),
        accepted_patch_count=int(economic_raw.get("accepted_patch_count", 0)),
        unavailable_cost_fields=int(
            economic_raw.get("unavailable_cost_fields", 0)
        ),
    )

    task_classes = tuple(calibration_raw.get("task_classes_observed") or ())
    task_counts = dict(calibration_raw.get("task_class_counts") or {})
    calibration = CalibrationMetrics(
        observation_count=int(calibration_raw.get("observation_count", 0)),
        calibration_use_count=int(calibration_raw.get("calibration_use_count", 0)),
        empirical_omission_rate=_empirical_from_dict(
            calibration_raw.get("empirical_omission_rate")
        ),
        last_revision=calibration_raw.get("last_revision"),
        task_classes_observed=task_classes,
        task_class_counts=MappingProxyType(
            {key: int(task_counts.get(key, 0)) for key in task_classes}
        ),
        task_coverage_count=int(calibration_raw.get("task_coverage_count", 0)),
        partition_counts=MappingProxyType(
            {
                str(key): int(value)
                for key, value in dict(
                    calibration_raw.get("partition_counts") or {}
                ).items()
            }
        ),
    )

    return CohortMetrics(
        cohort=expected_cohort,
        observation_count=int(data.get("observation_count", 0)),
        source_receipt_cids=tuple(data.get("source_receipt_cids") or ()),
        compression=compression,
        quality=quality,
        omission=omission,
        routing=routing,
        economic=economic,
        calibration=calibration,
    )


class GovernorMetricsCollector:
    """Accumulate sealed observations and emit dual-cohort metric reports.

    Simulated observations never enter the live cohort. Replayed receipt CIDs
    are idempotent (skipped). Building a report is pure with respect to the
    current accumulator snapshot.
    """

    def __init__(self) -> None:
        self._live = _CohortAccumulator(cohort=MetricsCohort.LIVE.value)
        self._simulated = _CohortAccumulator(cohort=MetricsCohort.SIMULATED.value)
        self._seen_receipts: set[str] = set()
        self._applied = 0
        self._skipped_idempotent = 0
        self._ordered_receipts: list[str] = []

    @property
    def applied_count(self) -> int:
        return self._applied

    @property
    def skipped_idempotent_count(self) -> int:
        return self._skipped_idempotent

    @property
    def live_observation_count(self) -> int:
        return self._live.observation_count

    @property
    def simulated_observation_count(self) -> int:
        return self._simulated.observation_count

    def ingest(
        self, observation: MetricsObservation | Mapping[str, Any]
    ) -> MetricsIngestDisposition:
        """Ingest one observation into the matching cohort.

        Returns :attr:`MetricsIngestDisposition.APPLIED` or
        :attr:`MetricsIngestDisposition.SKIPPED_IDEMPOTENT`.
        """

        if isinstance(observation, Mapping):
            obs = MetricsObservation.from_dict(observation)
        elif isinstance(observation, MetricsObservation):
            obs = observation
        else:
            raise MetricsError(
                "observation must be MetricsObservation or mapping",
                reason_code="rejected_malformed",
            )
        if len(self._seen_receipts) >= MAX_OBSERVATIONS and (
            obs.receipt_cid not in self._seen_receipts
        ):
            raise MetricsError("observation budget exceeded")
        if obs.receipt_cid in self._seen_receipts:
            self._skipped_idempotent = _add_counter(
                self._skipped_idempotent, 1, "skipped_idempotent"
            )
            return MetricsIngestDisposition.SKIPPED_IDEMPOTENT
        if obs.cohort == MetricsCohort.SIMULATED.value:
            self._simulated.ingest(obs)
        else:
            self._live.ingest(obs)
        self._seen_receipts.add(obs.receipt_cid)
        self._ordered_receipts.append(obs.receipt_cid)
        self._applied = _add_counter(self._applied, 1, "applied")
        return MetricsIngestDisposition.APPLIED

    def ingest_many(
        self, observations: Iterable[MetricsObservation | Mapping[str, Any]]
    ) -> tuple[MetricsIngestDisposition, ...]:
        return tuple(self.ingest(item) for item in observations)

    def build_report(
        self, *, metadata: Mapping[str, Any] | None = None
    ) -> GovernorMetricReport:
        """Seal a dual-cohort report from the current accumulator state."""

        live = self._live.finalize()
        simulated = self._simulated.finalize()
        meta = dict(metadata or {})
        meta.setdefault("evidence", SCG_METRICS_EVIDENCE)
        meta.setdefault("track", "metrics")
        return GovernorMetricReport(
            live=live,
            simulated=simulated,
            total_observations=live.observation_count + simulated.observation_count,
            applied_count=self._applied,
            skipped_idempotent_count=self._skipped_idempotent,
            source_receipt_cids=tuple(self._ordered_receipts),
            metadata=meta,
        )

    def reset(self) -> None:
        """Clear all accumulators (does not affect previously sealed reports)."""

        self._live = _CohortAccumulator(cohort=MetricsCohort.LIVE.value)
        self._simulated = _CohortAccumulator(cohort=MetricsCohort.SIMULATED.value)
        self._seen_receipts.clear()
        self._ordered_receipts.clear()
        self._applied = 0
        self._skipped_idempotent = 0


def collect_metrics(
    observations: Iterable[MetricsObservation | Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> GovernorMetricReport:
    """One-shot dual-cohort metrics collection over sealed observations."""

    collector = GovernorMetricsCollector()
    collector.ingest_many(observations)
    return collector.build_report(metadata=metadata)


def governor_metrics_collector_interface_id() -> str:
    return GOVERNOR_METRICS_COLLECTOR_INTERFACE


def governor_metric_report_interface_id() -> str:
    return GOVERNOR_METRIC_REPORT_INTERFACE


def metrics_evidence_id() -> str:
    return SCG_METRICS_EVIDENCE


def metrics_cohorts() -> tuple[str, ...]:
    return tuple(item.value for item in MetricsCohort)


def metrics_ingest_dispositions() -> tuple[str, ...]:
    return tuple(item.value for item in MetricsIngestDisposition)


__all__ = [
    "BASIS_POINTS",
    "CALIBRATION_METRICS_SCHEMA",
    "COHORT_METRICS_SCHEMA",
    "COMPRESSION_METRICS_SCHEMA",
    "ECONOMIC_METRICS_SCHEMA",
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "GOVERNOR_METRIC_REPORT_INTERFACE",
    "GOVERNOR_METRIC_REPORT_SCHEMA",
    "GOVERNOR_METRICS_COLLECTOR_INTERFACE",
    "INTEGER_PERCENTILE_SUMMARY_SCHEMA",
    "METRICS_OBSERVATION_SCHEMA",
    "OMISSION_METRICS_SCHEMA",
    "PRODUCER_ID",
    "PRODUCER_VERSION",
    "QUALITY_METRICS_SCHEMA",
    "ROUTING_METRICS_SCHEMA",
    "SCG_METRICS_EVIDENCE",
    "TOOL_ID",
    "CalibrationMetrics",
    "CohortMetrics",
    "CompressionMetrics",
    "EconomicMetrics",
    "GovernorMetricReport",
    "GovernorMetricsCollector",
    "IntegerPercentileSummary",
    "MetricsCohort",
    "MetricsError",
    "MetricsIngestDisposition",
    "MetricsObservation",
    "OmissionMetrics",
    "QualityMetrics",
    "RoutingMetrics",
    "build_empirical_rate",
    "build_percentile_summary",
    "collect_metrics",
    "governor_metric_report_interface_id",
    "governor_metrics_collector_interface_id",
    "metrics_cohorts",
    "metrics_evidence_id",
    "metrics_ingest_dispositions",
    "nearest_rank_percentile",
    "observation_from_receipt_fields",
]
