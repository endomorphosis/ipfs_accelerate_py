"""Disjoint campaign metrics for the Adversarial Assurance Engine (AAE-058).

Interface surface:

* ``AssuranceMetrics@1`` — sealed, content-addressed campaign metrics binding
  five **disjoint** populations (mutation coverage, detection quality, gap,
  remediation, economics).
* ``compute_assurance_metrics`` — pure, deterministic aggregation over campaign
  outcomes, gaps, remediations, and cost records.

Normative properties (acceptance / plan §5, §15):

* Populations are pairwise disjoint by construction (distinct member-id
  prefixes) and re-checked at seal time.
* Kill-rate and risk-weighted denominators **exclude** invalid, uncompilable,
  infrastructure-failed, timeout, inconclusive, equivalent, and
  probably_equivalent mutants; those statuses never count as killed.
* Rates use integer basis points; empty denominators yield ``None`` (unknown),
  never fabricated zeros presented as success.
* No production policy change; cold import is side-effect free.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Iterable

from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_structured,
    validate_cid,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.analysis_contracts import (
    AssuranceGapClass,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.common import (
    AssuranceBaseError,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.execution_contracts import (
    DetectorKind,
    DetectorRole,
    MutationOutcomeStatus,
    counts_as_killed,
    killed_outcome_statuses,
    mutation_outcome_statuses,
    never_counted_as_killed_statuses,
)

# ---------------------------------------------------------------------------
# Schema / interface / evidence
# ---------------------------------------------------------------------------

ASSURANCE_METRICS_INTERFACE: Final[str] = "AssuranceMetrics@1"
ASSURANCE_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-metrics@1"
)
MUTATION_COVERAGE_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "mutation-coverage-metrics@1"
)
DETECTION_QUALITY_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "detection-quality-metrics@1"
)
GAP_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-gap-metrics@1"
)
REMEDIATION_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "remediation-metrics@1"
)
ECONOMICS_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "economics-metrics@1"
)
METRICS_POPULATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "metrics-population@1"
)

AAE_METRICS_EVIDENCE: Final[str] = "aae/metrics@1"
ADAPTER_ID: Final[str] = "aae-metrics"
BOARD_NAMESPACE: Final[str] = "adversarial-assurance-engine-v1"
GENERATOR_ID: Final[str] = "assurance_metrics"
GENERATOR_VERSION: Final[str] = "1.0.0"
PRODUCER_ID: Final[str] = "adversarial_assurance"
PRODUCER_VERSION: Final[str] = "1"
TOOL_ID: Final[str] = "metrics.v1"

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_LIST: Final[int] = 4_096
MAX_POPULATION: Final[int] = 16_384
MAX_COUNTER: Final[int] = 2**63 - 1
BASIS_POINTS: Final[int] = 10_000
MAX_RISK_WEIGHT_BP: Final[int] = 10_000

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")
_MEMBER_RE: Final[re.Pattern[str]] = re.compile(
    r"^[a-z][a-z0-9_.:/+@#$-]{0,255}$"
)

# ---------------------------------------------------------------------------
# Closed population / exclusion vocabularies
# ---------------------------------------------------------------------------


class MetricsPopulationKind(str, Enum):
    """Closed metric population kinds (plan §15). Must remain disjoint."""

    MUTATION_COVERAGE = "mutation_coverage"
    DETECTION_QUALITY = "detection_quality"
    GAP = "gap"
    REMEDIATION = "remediation"
    ECONOMICS = "economics"


METRICS_POPULATION_KINDS: Final[tuple[str, ...]] = tuple(
    item.value for item in MetricsPopulationKind
)

# Denominator exclusions for kill-rate / risk-weighted scoring (plan §5).
# These statuses never count as killed and are omitted from scoring denominators.
DENOMINATOR_EXCLUDED_OUTCOMES: Final[frozenset[str]] = frozenset(
    {
        MutationOutcomeStatus.INVALID_MUTANT.value,
        MutationOutcomeStatus.UNCOMPILABLE.value,
        MutationOutcomeStatus.INFRASTRUCTURE_FAILURE.value,
        MutationOutcomeStatus.TIMEOUT.value,
        MutationOutcomeStatus.INCONCLUSIVE.value,
        MutationOutcomeStatus.EQUIVALENT.value,
        MutationOutcomeStatus.PROBABLY_EQUIVALENT.value,
    }
)

# Infrastructure-style exclusions (subset of DENOMINATOR_EXCLUDED_OUTCOMES).
INFRASTRUCTURE_EXCLUDED_OUTCOMES: Final[frozenset[str]] = frozenset(
    {
        MutationOutcomeStatus.INVALID_MUTANT.value,
        MutationOutcomeStatus.UNCOMPILABLE.value,
        MutationOutcomeStatus.INFRASTRUCTURE_FAILURE.value,
        MutationOutcomeStatus.TIMEOUT.value,
        MutationOutcomeStatus.INCONCLUSIVE.value,
    }
)

EQUIVALENCE_EXCLUDED_OUTCOMES: Final[frozenset[str]] = frozenset(
    {
        MutationOutcomeStatus.EQUIVALENT.value,
        MutationOutcomeStatus.PROBABLY_EQUIVALENT.value,
    }
)

_TEST_LIKE_KINDS: Final[frozenset[str]] = frozenset(
    {
        DetectorKind.UNIT_TEST.value,
        DetectorKind.INTEGRATION_TEST.value,
        DetectorKind.PROPERTY_TEST.value,
    }
)
_PROOF_LIKE_KINDS: Final[frozenset[str]] = frozenset(
    {
        DetectorKind.FORMAL_OBLIGATION.value,
        DetectorKind.INCREMENTAL_SEAL.value,
    }
)
_POLICY_LIKE_KINDS: Final[frozenset[str]] = frozenset(
    {DetectorKind.POLICY_RULE.value}
)

_GAP_CLASS_VALUES: Final[frozenset[str]] = frozenset(
    item.value for item in AssuranceGapClass
)

_KILLED_STATUSES: Final[frozenset[str]] = frozenset(killed_outcome_statuses())
_NEVER_KILLED: Final[frozenset[str]] = frozenset(never_counted_as_killed_statuses())
_CLOSED_OUTCOMES: Final[frozenset[str]] = frozenset(mutation_outcome_statuses())


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MetricsError(AssuranceBaseError):
    """Raised when metrics inputs are malformed or violate disjointness."""

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


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        raise MetricsError(f"{name} must be a nonempty string", reason_code="invalid_type")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise MetricsError(f"{name} must be trimmed NFC text", reason_code="invalid_text")
    if len(value) > MAX_TEXT_CHARS or any(not char.isprintable() for char in value):
        raise MetricsError(f"{name} contains invalid text", reason_code="invalid_text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name, empty=True) if value == "" else _text(value, name)


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if not _TOKEN_RE.match(text):
        raise MetricsError(f"{name} is not a valid token", reason_code="invalid_token")
    return text


def _member_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if not _MEMBER_RE.match(text):
        raise MetricsError(
            f"{name} is not a valid population member id",
            reason_code="invalid_member_id",
        )
    return text


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        validate_cid(text)
    except Exception as exc:  # noqa: BLE001 — surface as metrics error
        raise MetricsError(
            f"{name} is not a valid CID",
            reason_code="invalid_cid",
            details={"value": text},
        ) from exc
    return text


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise MetricsError(f"{name} must be a bool", reason_code="invalid_type")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise MetricsError(
            f"{name} must be a non-negative int",
            reason_code="invalid_type",
        )
    if value > MAX_COUNTER:
        raise MetricsError(f"{name} exceeds maximum", reason_code="bounds")
    return value


def _optional_nonneg_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _nonneg_int(value, name)


def _basis_points(value: Any, name: str) -> int:
    n = _nonneg_int(value, name)
    if n > BASIS_POINTS:
        raise MetricsError(
            f"{name} must be in 0..{BASIS_POINTS}",
            reason_code="invalid_bp",
        )
    return n


def _optional_rate_bp(numerator: int, denominator: int) -> int | None:
    """Integer rate in basis points; ``None`` when denominator is empty."""

    if denominator <= 0:
        return None
    if numerator < 0 or numerator > denominator:
        # Allow numerator > denominator only for unexpected counts; clamp report.
        if numerator < 0:
            raise MetricsError("rate numerator must be non-negative", reason_code="invalid_rate")
    return (numerator * BASIS_POINTS) // denominator


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or isinstance(value, (str, bytes)):
        raise MetricsError(f"{name} must be a mapping", reason_code="invalid_type")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MetricsError(f"{name} must be a sequence", reason_code="invalid_type")
    if len(value) > MAX_LIST:
        raise MetricsError(f"{name} exceeds maximum length", reason_code="bounds")
    return value


def _stable_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in values:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


def _sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


# ---------------------------------------------------------------------------
# Outcome / population classification
# ---------------------------------------------------------------------------


def is_denominator_excluded(outcome_status: str | MutationOutcomeStatus) -> bool:
    """Return True when *outcome_status* must leave kill-rate denominators."""

    status = (
        outcome_status.value
        if isinstance(outcome_status, MutationOutcomeStatus)
        else str(outcome_status)
    )
    if status not in _CLOSED_OUTCOMES:
        raise MetricsError(
            f"unknown outcome_status {status!r}",
            reason_code="unknown_outcome",
        )
    return status in DENOMINATOR_EXCLUDED_OUTCOMES


def is_infrastructure_excluded(outcome_status: str | MutationOutcomeStatus) -> bool:
    status = (
        outcome_status.value
        if isinstance(outcome_status, MutationOutcomeStatus)
        else str(outcome_status)
    )
    if status not in _CLOSED_OUTCOMES:
        raise MetricsError(
            f"unknown outcome_status {status!r}",
            reason_code="unknown_outcome",
        )
    return status in INFRASTRUCTURE_EXCLUDED_OUTCOMES


def is_equivalence_excluded(outcome_status: str | MutationOutcomeStatus) -> bool:
    status = (
        outcome_status.value
        if isinstance(outcome_status, MutationOutcomeStatus)
        else str(outcome_status)
    )
    if status not in _CLOSED_OUTCOMES:
        raise MetricsError(
            f"unknown outcome_status {status!r}",
            reason_code="unknown_outcome",
        )
    return status in EQUIVALENCE_EXCLUDED_OUTCOMES


def population_member_id(kind: str | MetricsPopulationKind, entity_id: str) -> str:
    """Build a population-scoped member id (prefix enforces disjointness)."""

    kind_value = kind.value if isinstance(kind, MetricsPopulationKind) else str(kind)
    if kind_value not in METRICS_POPULATION_KINDS:
        raise MetricsError(
            f"unknown population kind {kind_value!r}",
            reason_code="unknown_population",
        )
    entity = _member_id(entity_id, "entity_id")
    return f"{kind_value}:{entity}"


def coverage_bucket(outcome_status: str) -> str:
    """Map a closed outcome status to a coverage bucket label."""

    if outcome_status not in _CLOSED_OUTCOMES:
        raise MetricsError(
            f"unknown outcome_status {outcome_status!r}",
            reason_code="unknown_outcome",
        )
    if outcome_status in _KILLED_STATUSES:
        return "killed"
    if outcome_status == MutationOutcomeStatus.SURVIVED_SELECTED_VERIFICATION.value:
        return "selected_survivor"
    if outcome_status == MutationOutcomeStatus.SURVIVED_FULL_VERIFICATION.value:
        return "full_survivor"
    if outcome_status in EQUIVALENCE_EXCLUDED_OUTCOMES:
        return "equivalent"
    if outcome_status == MutationOutcomeStatus.INVALID_MUTANT.value:
        return "invalid"
    if outcome_status == MutationOutcomeStatus.UNCOMPILABLE.value:
        return "uncompilable"
    if outcome_status == MutationOutcomeStatus.INFRASTRUCTURE_FAILURE.value:
        return "infrastructure_failure"
    if outcome_status == MutationOutcomeStatus.TIMEOUT.value:
        return "timeout"
    if outcome_status == MutationOutcomeStatus.INCONCLUSIVE.value:
        return "inconclusive"
    if outcome_status == MutationOutcomeStatus.HUMAN_REVIEW_REQUIRED.value:
        return "human_review_required"
    return "other"


# ---------------------------------------------------------------------------
# Sub-metric records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetricsPopulation:
    """Sealed membership for one metrics population."""

    kind: str
    member_ids: tuple[str, ...]
    count: int

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"schema", "kind", "member_ids", "count"}
    )

    def __post_init__(self) -> None:
        kind = _token(self.kind, "kind")
        if kind not in METRICS_POPULATION_KINDS:
            raise MetricsError(
                f"unknown population kind {kind!r}",
                reason_code="unknown_population",
            )
        object.__setattr__(self, "kind", kind)
        members = tuple(_member_id(item, "member_ids") for item in self.member_ids)
        if len(members) != len(set(members)):
            raise MetricsError(
                f"population {kind!r} has duplicate member ids",
                reason_code="duplicate_member",
            )
        if len(members) > MAX_POPULATION:
            raise MetricsError(
                f"population {kind!r} exceeds maximum size",
                reason_code="bounds",
            )
        # Prefix check: every member must start with kind:
        prefix = f"{kind}:"
        for mid in members:
            if not mid.startswith(prefix):
                raise MetricsError(
                    f"member {mid!r} is not scoped to population {kind!r}",
                    reason_code="population_scope",
                )
        object.__setattr__(self, "member_ids", tuple(sorted(members)))
        count = _nonneg_int(self.count, "count")
        if count != len(members):
            raise MetricsError(
                f"population {kind!r} count {count} != len(member_ids)",
                reason_code="count_mismatch",
            )
        object.__setattr__(self, "count", count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": METRICS_POPULATION_SCHEMA,
            "kind": self.kind,
            "member_ids": list(self.member_ids),
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricsPopulation":
        payload = dict(_mapping(data, "population"))
        payload.pop("schema", None)
        return cls(
            kind=payload["kind"],
            member_ids=tuple(payload.get("member_ids") or ()),
            count=int(payload["count"]),
        )


@dataclass(frozen=True, slots=True)
class MutationCoverageMetrics:
    """Mutation coverage counters and rates (plan §15)."""

    generated_count: int
    admitted_count: int
    invalid_count: int
    uncompilable_count: int
    infrastructure_failure_count: int
    timeout_count: int
    inconclusive_count: int
    equivalent_count: int
    probably_equivalent_count: int
    killed_count: int
    selected_survivor_count: int
    full_survivor_count: int
    human_review_count: int
    denominator_excluded_count: int
    scoring_denominator: int
    kill_rate_bp: int | None
    risk_weighted_score_bp: int | None
    class_kill_rates_bp: Mapping[str, int | None]
    outcome_counts: Mapping[str, int]
    member_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "generated_count",
            "admitted_count",
            "invalid_count",
            "uncompilable_count",
            "infrastructure_failure_count",
            "timeout_count",
            "inconclusive_count",
            "equivalent_count",
            "probably_equivalent_count",
            "killed_count",
            "selected_survivor_count",
            "full_survivor_count",
            "human_review_count",
            "denominator_excluded_count",
            "scoring_denominator",
        ):
            object.__setattr__(self, name, _nonneg_int(getattr(self, name), name))
        if self.kill_rate_bp is not None:
            object.__setattr__(
                self, "kill_rate_bp", _basis_points(self.kill_rate_bp, "kill_rate_bp")
            )
        if self.risk_weighted_score_bp is not None:
            object.__setattr__(
                self,
                "risk_weighted_score_bp",
                _basis_points(self.risk_weighted_score_bp, "risk_weighted_score_bp"),
            )
        rates: dict[str, int | None] = {}
        for key, value in dict(self.class_kill_rates_bp or {}).items():
            k = _token(key, "class_kill_rates_bp key")
            rates[k] = None if value is None else _basis_points(value, f"class_kill_rates_bp[{k}]")
        object.__setattr__(self, "class_kill_rates_bp", MappingProxyType(dict(sorted(rates.items()))))
        counts: dict[str, int] = {}
        for key, value in dict(self.outcome_counts or {}).items():
            k = _token(key, "outcome_counts key")
            counts[k] = _nonneg_int(value, f"outcome_counts[{k}]")
        object.__setattr__(self, "outcome_counts", MappingProxyType(dict(sorted(counts.items()))))
        members = tuple(_member_id(item, "member_ids") for item in self.member_ids)
        object.__setattr__(self, "member_ids", tuple(sorted(set(members))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MUTATION_COVERAGE_METRICS_SCHEMA,
            "generated_count": self.generated_count,
            "admitted_count": self.admitted_count,
            "invalid_count": self.invalid_count,
            "uncompilable_count": self.uncompilable_count,
            "infrastructure_failure_count": self.infrastructure_failure_count,
            "timeout_count": self.timeout_count,
            "inconclusive_count": self.inconclusive_count,
            "equivalent_count": self.equivalent_count,
            "probably_equivalent_count": self.probably_equivalent_count,
            "killed_count": self.killed_count,
            "selected_survivor_count": self.selected_survivor_count,
            "full_survivor_count": self.full_survivor_count,
            "human_review_count": self.human_review_count,
            "denominator_excluded_count": self.denominator_excluded_count,
            "scoring_denominator": self.scoring_denominator,
            "kill_rate_bp": self.kill_rate_bp,
            "risk_weighted_score_bp": self.risk_weighted_score_bp,
            "class_kill_rates_bp": dict(self.class_kill_rates_bp),
            "outcome_counts": dict(self.outcome_counts),
            "member_ids": list(self.member_ids),
        }


@dataclass(frozen=True, slots=True)
class DetectionQualityMetrics:
    """Predicted/observed/missed/unexpected detector quality (plan §15)."""

    predicted_detector_count: int
    selected_detector_count: int
    executed_detector_count: int
    observed_detector_count: int
    missed_detector_count: int
    unexpected_detector_count: int
    selected_test_rate_bp: int | None
    selected_proof_rate_bp: int | None
    selected_policy_rate_bp: int | None
    full_suite_only_detection_count: int
    full_suite_only_rate_bp: int | None
    member_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "predicted_detector_count",
            "selected_detector_count",
            "executed_detector_count",
            "observed_detector_count",
            "missed_detector_count",
            "unexpected_detector_count",
            "full_suite_only_detection_count",
        ):
            object.__setattr__(self, name, _nonneg_int(getattr(self, name), name))
        for name in (
            "selected_test_rate_bp",
            "selected_proof_rate_bp",
            "selected_policy_rate_bp",
            "full_suite_only_rate_bp",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _basis_points(value, name))
        members = tuple(_member_id(item, "member_ids") for item in self.member_ids)
        object.__setattr__(self, "member_ids", tuple(sorted(set(members))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DETECTION_QUALITY_METRICS_SCHEMA,
            "predicted_detector_count": self.predicted_detector_count,
            "selected_detector_count": self.selected_detector_count,
            "executed_detector_count": self.executed_detector_count,
            "observed_detector_count": self.observed_detector_count,
            "missed_detector_count": self.missed_detector_count,
            "unexpected_detector_count": self.unexpected_detector_count,
            "selected_test_rate_bp": self.selected_test_rate_bp,
            "selected_proof_rate_bp": self.selected_proof_rate_bp,
            "selected_policy_rate_bp": self.selected_policy_rate_bp,
            "full_suite_only_detection_count": self.full_suite_only_detection_count,
            "full_suite_only_rate_bp": self.full_suite_only_rate_bp,
            "member_ids": list(self.member_ids),
        }


@dataclass(frozen=True, slots=True)
class GapMetrics:
    """Assurance gap category counts (plan §15)."""

    total_gaps: int
    high_risk_survivor_gaps: int
    category_counts: Mapping[str, int]
    member_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "total_gaps", _nonneg_int(self.total_gaps, "total_gaps"))
        object.__setattr__(
            self,
            "high_risk_survivor_gaps",
            _nonneg_int(self.high_risk_survivor_gaps, "high_risk_survivor_gaps"),
        )
        counts: dict[str, int] = {}
        for key, value in dict(self.category_counts or {}).items():
            k = _token(key, "category_counts key")
            if k not in _GAP_CLASS_VALUES and k != "unspecified":
                raise MetricsError(
                    f"unknown gap category {k!r}",
                    reason_code="unknown_gap_class",
                )
            counts[k] = _nonneg_int(value, f"category_counts[{k}]")
        object.__setattr__(
            self, "category_counts", MappingProxyType(dict(sorted(counts.items())))
        )
        members = tuple(_member_id(item, "member_ids") for item in self.member_ids)
        object.__setattr__(self, "member_ids", tuple(sorted(set(members))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GAP_METRICS_SCHEMA,
            "total_gaps": self.total_gaps,
            "high_risk_survivor_gaps": self.high_risk_survivor_gaps,
            "category_counts": dict(self.category_counts),
            "member_ids": list(self.member_ids),
        }


@dataclass(frozen=True, slots=True)
class RemediationMetrics:
    """Remediation candidate / promotion metrics (plan §15)."""

    candidate_count: int
    evaluated_count: int
    held_out_kill_count: int
    regression_count: int
    overconstraint_count: int
    accepted_promotion_count: int
    rejected_promotion_count: int
    total_cost_cpu_ms: int
    total_cost_wall_ms: int
    member_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "candidate_count",
            "evaluated_count",
            "held_out_kill_count",
            "regression_count",
            "overconstraint_count",
            "accepted_promotion_count",
            "rejected_promotion_count",
            "total_cost_cpu_ms",
            "total_cost_wall_ms",
        ):
            object.__setattr__(self, name, _nonneg_int(getattr(self, name), name))
        members = tuple(_member_id(item, "member_ids") for item in self.member_ids)
        object.__setattr__(self, "member_ids", tuple(sorted(set(members))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REMEDIATION_METRICS_SCHEMA,
            "candidate_count": self.candidate_count,
            "evaluated_count": self.evaluated_count,
            "held_out_kill_count": self.held_out_kill_count,
            "regression_count": self.regression_count,
            "overconstraint_count": self.overconstraint_count,
            "accepted_promotion_count": self.accepted_promotion_count,
            "rejected_promotion_count": self.rejected_promotion_count,
            "total_cost_cpu_ms": self.total_cost_cpu_ms,
            "total_cost_wall_ms": self.total_cost_wall_ms,
            "member_ids": list(self.member_ids),
        }


@dataclass(frozen=True, slots=True)
class EconomicsMetrics:
    """Full versus incremental economics (plan §15)."""

    mutant_cost_records: int
    full_cpu_ms_total: int
    full_wall_ms_total: int
    incremental_cpu_ms_total: int
    incremental_wall_ms_total: int
    compute_saved_cpu_ms: int | None
    compute_saved_wall_ms: int | None
    savings_rate_bp: int | None
    proof_cache_hits: int
    proof_cache_misses: int
    proof_cache_reuse_rate_bp: int | None
    model_calls: int
    model_tokens: int
    cost_per_critical_gap_cpu_ms: int | None
    cost_per_promotion_cpu_ms: int | None
    avg_full_cost_per_mutant_cpu_ms: int | None
    avg_incremental_cost_per_mutant_cpu_ms: int | None
    member_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "mutant_cost_records",
            "full_cpu_ms_total",
            "full_wall_ms_total",
            "incremental_cpu_ms_total",
            "incremental_wall_ms_total",
            "proof_cache_hits",
            "proof_cache_misses",
            "model_calls",
            "model_tokens",
        ):
            object.__setattr__(self, name, _nonneg_int(getattr(self, name), name))
        for name in (
            "compute_saved_cpu_ms",
            "compute_saved_wall_ms",
            "cost_per_critical_gap_cpu_ms",
            "cost_per_promotion_cpu_ms",
            "avg_full_cost_per_mutant_cpu_ms",
            "avg_incremental_cost_per_mutant_cpu_ms",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonneg_int(value, name))
        for name in ("savings_rate_bp", "proof_cache_reuse_rate_bp"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _basis_points(value, name))
        members = tuple(_member_id(item, "member_ids") for item in self.member_ids)
        object.__setattr__(self, "member_ids", tuple(sorted(set(members))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ECONOMICS_METRICS_SCHEMA,
            "mutant_cost_records": self.mutant_cost_records,
            "full_cpu_ms_total": self.full_cpu_ms_total,
            "full_wall_ms_total": self.full_wall_ms_total,
            "incremental_cpu_ms_total": self.incremental_cpu_ms_total,
            "incremental_wall_ms_total": self.incremental_wall_ms_total,
            "compute_saved_cpu_ms": self.compute_saved_cpu_ms,
            "compute_saved_wall_ms": self.compute_saved_wall_ms,
            "savings_rate_bp": self.savings_rate_bp,
            "proof_cache_hits": self.proof_cache_hits,
            "proof_cache_misses": self.proof_cache_misses,
            "proof_cache_reuse_rate_bp": self.proof_cache_reuse_rate_bp,
            "model_calls": self.model_calls,
            "model_tokens": self.model_tokens,
            "cost_per_critical_gap_cpu_ms": self.cost_per_critical_gap_cpu_ms,
            "cost_per_promotion_cpu_ms": self.cost_per_promotion_cpu_ms,
            "avg_full_cost_per_mutant_cpu_ms": self.avg_full_cost_per_mutant_cpu_ms,
            "avg_incremental_cost_per_mutant_cpu_ms": (
                self.avg_incremental_cost_per_mutant_cpu_ms
            ),
            "member_ids": list(self.member_ids),
        }


# ---------------------------------------------------------------------------
# AssuranceMetrics@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AssuranceMetrics:
    """Sealed campaign metrics with five disjoint populations.

    Interface: ``AssuranceMetrics@1``
    """

    interface_id: str
    campaign_id: str
    plan_id: str | None
    plan_cid: str | None
    result_cid: str | None
    repository_state_cid: str | None
    mutation_coverage: MutationCoverageMetrics
    detection_quality: DetectionQualityMetrics
    gaps: GapMetrics
    remediation: RemediationMetrics
    economics: EconomicsMetrics
    populations: Mapping[str, MetricsPopulation]
    reason_codes: tuple[str, ...]
    notes: str | None = None
    production_policy_changed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface_id",
            "campaign_id",
            "plan_id",
            "plan_cid",
            "result_cid",
            "repository_state_cid",
            "mutation_coverage",
            "detection_quality",
            "gaps",
            "remediation",
            "economics",
            "populations",
            "reason_codes",
            "notes",
            "production_policy_changed",
            "metadata",
            "metrics_cid",
            "evidence",
            "populations_disjoint",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "interface_id", _text(self.interface_id, "interface_id")
        )
        if self.interface_id != ASSURANCE_METRICS_INTERFACE:
            raise MetricsError(
                "interface_id must be AssuranceMetrics@1",
                reason_code="invalid_interface",
            )
        object.__setattr__(self, "campaign_id", _token(self.campaign_id, "campaign_id"))
        object.__setattr__(
            self,
            "plan_id",
            None if self.plan_id is None else _token(self.plan_id, "plan_id"),
        )
        object.__setattr__(
            self, "plan_cid", _optional_cid(self.plan_cid, "plan_cid")
        )
        object.__setattr__(
            self, "result_cid", _optional_cid(self.result_cid, "result_cid")
        )
        object.__setattr__(
            self,
            "repository_state_cid",
            _optional_cid(self.repository_state_cid, "repository_state_cid"),
        )
        if not isinstance(self.mutation_coverage, MutationCoverageMetrics):
            raise MetricsError(
                "mutation_coverage must be MutationCoverageMetrics",
                reason_code="invalid_type",
            )
        if not isinstance(self.detection_quality, DetectionQualityMetrics):
            raise MetricsError(
                "detection_quality must be DetectionQualityMetrics",
                reason_code="invalid_type",
            )
        if not isinstance(self.gaps, GapMetrics):
            raise MetricsError("gaps must be GapMetrics", reason_code="invalid_type")
        if not isinstance(self.remediation, RemediationMetrics):
            raise MetricsError(
                "remediation must be RemediationMetrics",
                reason_code="invalid_type",
            )
        if not isinstance(self.economics, EconomicsMetrics):
            raise MetricsError(
                "economics must be EconomicsMetrics",
                reason_code="invalid_type",
            )

        pops: dict[str, MetricsPopulation] = {}
        for key, value in dict(self.populations or {}).items():
            k = _token(key, "populations key")
            if isinstance(value, MetricsPopulation):
                pop = value
            elif isinstance(value, Mapping):
                pop = MetricsPopulation.from_dict(value)
            else:
                raise MetricsError(
                    "population entries must be MetricsPopulation or mapping",
                    reason_code="invalid_type",
                )
            if pop.kind != k:
                raise MetricsError(
                    f"population key {k!r} does not match kind {pop.kind!r}",
                    reason_code="population_key_mismatch",
                )
            pops[k] = pop
        missing = set(METRICS_POPULATION_KINDS) - set(pops)
        if missing:
            raise MetricsError(
                f"missing required populations: {sorted(missing)}",
                reason_code="missing_population",
            )
        extra = set(pops) - set(METRICS_POPULATION_KINDS)
        if extra:
            raise MetricsError(
                f"unknown populations: {sorted(extra)}",
                reason_code="unknown_population",
            )
        object.__setattr__(
            self,
            "populations",
            MappingProxyType({k: pops[k] for k in METRICS_POPULATION_KINDS}),
        )
        assert_populations_disjoint(self.populations)

        # Cross-check population membership against sub-metric member_ids.
        _assert_member_alignment(
            self.populations[MetricsPopulationKind.MUTATION_COVERAGE.value],
            self.mutation_coverage.member_ids,
        )
        _assert_member_alignment(
            self.populations[MetricsPopulationKind.DETECTION_QUALITY.value],
            self.detection_quality.member_ids,
        )
        _assert_member_alignment(
            self.populations[MetricsPopulationKind.GAP.value],
            self.gaps.member_ids,
        )
        _assert_member_alignment(
            self.populations[MetricsPopulationKind.REMEDIATION.value],
            self.remediation.member_ids,
        )
        _assert_member_alignment(
            self.populations[MetricsPopulationKind.ECONOMICS.value],
            self.economics.member_ids,
        )

        codes = tuple(
            _token(item, "reason_codes") for item in (self.reason_codes or ())
        )
        object.__setattr__(self, "reason_codes", _stable_unique(codes))
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        changed = _bool(self.production_policy_changed, "production_policy_changed")
        if changed:
            raise MetricsError(
                "metrics must not claim production policy change",
                reason_code="production_policy_change",
            )
        object.__setattr__(self, "production_policy_changed", False)
        meta = self.metadata if isinstance(self.metadata, Mapping) else {}
        object.__setattr__(self, "metadata", MappingProxyType(dict(meta)))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": ASSURANCE_METRICS_SCHEMA,
            "interface_id": self.interface_id,
            "campaign_id": self.campaign_id,
            "plan_id": self.plan_id,
            "plan_cid": self.plan_cid,
            "result_cid": self.result_cid,
            "repository_state_cid": self.repository_state_cid,
            "mutation_coverage": self.mutation_coverage.to_dict(),
            "detection_quality": self.detection_quality.to_dict(),
            "gaps": self.gaps.to_dict(),
            "remediation": self.remediation.to_dict(),
            "economics": self.economics.to_dict(),
            "populations": {
                kind: self.populations[kind].to_dict()
                for kind in METRICS_POPULATION_KINDS
            },
            "reason_codes": list(self.reason_codes),
            "notes": self.notes,
            "production_policy_changed": False,
            "metadata": dict(self.metadata),
            "evidence": AAE_METRICS_EVIDENCE,
            "populations_disjoint": True,
        }

    @property
    def metrics_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["metrics_cid"] = self.metrics_cid
        return payload


def _assert_member_alignment(
    population: MetricsPopulation, member_ids: Sequence[str]
) -> None:
    expected = tuple(sorted(set(member_ids)))
    if population.member_ids != expected:
        raise MetricsError(
            f"population {population.kind!r} members disagree with sub-metric",
            reason_code="member_alignment",
            details={
                "population": list(population.member_ids),
                "sub_metric": list(expected),
            },
        )


def assert_populations_disjoint(
    populations: Mapping[str, MetricsPopulation],
) -> None:
    """Fail closed when any member id appears in more than one population."""

    seen: dict[str, str] = {}
    for kind in METRICS_POPULATION_KINDS:
        pop = populations.get(kind)
        if pop is None:
            raise MetricsError(
                f"missing population {kind!r}",
                reason_code="missing_population",
            )
        for mid in pop.member_ids:
            owner = seen.get(mid)
            if owner is not None:
                raise MetricsError(
                    f"population member {mid!r} appears in both {owner!r} and {kind!r}",
                    reason_code="populations_not_disjoint",
                    details={"member_id": mid, "populations": [owner, kind]},
                )
            seen[mid] = kind


def verify_assurance_metrics_identity(
    metrics: AssuranceMetrics | Mapping[str, Any],
) -> str:
    """Recompute and return metrics_cid; raise on forged or non-disjoint input."""

    declared_cid: str | None = None
    if isinstance(metrics, AssuranceMetrics):
        sealed = metrics
        declared_cid = metrics.metrics_cid
    elif isinstance(metrics, Mapping):
        raw = dict(metrics)
        declared = raw.get("metrics_cid")
        if declared is not None:
            declared_cid = _cid(declared, "metrics_cid")
        sealed = assurance_metrics_from_dict(raw)
    else:
        raise MetricsError(
            "metrics must be AssuranceMetrics or mapping",
            reason_code="invalid_type",
        )
    recomputed = cid_for_structured(sealed.identity_payload())
    if declared_cid is not None and recomputed != declared_cid:
        raise MetricsError(
            "metrics_cid identity mismatch with recomputed identity",
            reason_code="identity_mismatch",
            details={"declared": declared_cid, "recomputed": recomputed},
        )
    if recomputed != sealed.metrics_cid:
        raise MetricsError(
            "metrics_cid identity mismatch with recomputed identity",
            reason_code="identity_mismatch",
        )
    assert_populations_disjoint(sealed.populations)
    return recomputed


def assurance_metrics_from_dict(data: Mapping[str, Any]) -> AssuranceMetrics:
    """Rehydrate ``AssuranceMetrics`` from a sealed mapping."""

    payload = dict(_mapping(data, "metrics"))
    payload.pop("metrics_cid", None)
    payload.pop("evidence", None)
    payload.pop("populations_disjoint", None)
    schema = payload.pop("schema", ASSURANCE_METRICS_SCHEMA)
    if schema != ASSURANCE_METRICS_SCHEMA:
        raise MetricsError(
            f"unexpected metrics schema {schema!r}",
            reason_code="invalid_schema",
        )

    def _cov(raw: Any) -> MutationCoverageMetrics:
        d = dict(_mapping(raw, "mutation_coverage"))
        d.pop("schema", None)
        return MutationCoverageMetrics(
            generated_count=d["generated_count"],
            admitted_count=d["admitted_count"],
            invalid_count=d["invalid_count"],
            uncompilable_count=d.get("uncompilable_count", 0),
            infrastructure_failure_count=d.get("infrastructure_failure_count", 0),
            timeout_count=d.get("timeout_count", 0),
            inconclusive_count=d.get("inconclusive_count", 0),
            equivalent_count=d["equivalent_count"],
            probably_equivalent_count=d.get("probably_equivalent_count", 0),
            killed_count=d["killed_count"],
            selected_survivor_count=d.get("selected_survivor_count", 0),
            full_survivor_count=d.get("full_survivor_count", 0),
            human_review_count=d.get("human_review_count", 0),
            denominator_excluded_count=d["denominator_excluded_count"],
            scoring_denominator=d["scoring_denominator"],
            kill_rate_bp=d.get("kill_rate_bp"),
            risk_weighted_score_bp=d.get("risk_weighted_score_bp"),
            class_kill_rates_bp=d.get("class_kill_rates_bp") or {},
            outcome_counts=d.get("outcome_counts") or {},
            member_ids=tuple(d.get("member_ids") or ()),
        )

    def _det(raw: Any) -> DetectionQualityMetrics:
        d = dict(_mapping(raw, "detection_quality"))
        d.pop("schema", None)
        return DetectionQualityMetrics(
            predicted_detector_count=d["predicted_detector_count"],
            selected_detector_count=d["selected_detector_count"],
            executed_detector_count=d.get("executed_detector_count", 0),
            observed_detector_count=d["observed_detector_count"],
            missed_detector_count=d["missed_detector_count"],
            unexpected_detector_count=d["unexpected_detector_count"],
            selected_test_rate_bp=d.get("selected_test_rate_bp"),
            selected_proof_rate_bp=d.get("selected_proof_rate_bp"),
            selected_policy_rate_bp=d.get("selected_policy_rate_bp"),
            full_suite_only_detection_count=d.get("full_suite_only_detection_count", 0),
            full_suite_only_rate_bp=d.get("full_suite_only_rate_bp"),
            member_ids=tuple(d.get("member_ids") or ()),
        )

    def _gap(raw: Any) -> GapMetrics:
        d = dict(_mapping(raw, "gaps"))
        d.pop("schema", None)
        return GapMetrics(
            total_gaps=d["total_gaps"],
            high_risk_survivor_gaps=d.get("high_risk_survivor_gaps", 0),
            category_counts=d.get("category_counts") or {},
            member_ids=tuple(d.get("member_ids") or ()),
        )

    def _rem(raw: Any) -> RemediationMetrics:
        d = dict(_mapping(raw, "remediation"))
        d.pop("schema", None)
        return RemediationMetrics(
            candidate_count=d["candidate_count"],
            evaluated_count=d.get("evaluated_count", 0),
            held_out_kill_count=d.get("held_out_kill_count", 0),
            regression_count=d.get("regression_count", 0),
            overconstraint_count=d.get("overconstraint_count", 0),
            accepted_promotion_count=d.get("accepted_promotion_count", 0),
            rejected_promotion_count=d.get("rejected_promotion_count", 0),
            total_cost_cpu_ms=d.get("total_cost_cpu_ms", 0),
            total_cost_wall_ms=d.get("total_cost_wall_ms", 0),
            member_ids=tuple(d.get("member_ids") or ()),
        )

    def _eco(raw: Any) -> EconomicsMetrics:
        d = dict(_mapping(raw, "economics"))
        d.pop("schema", None)
        return EconomicsMetrics(
            mutant_cost_records=d["mutant_cost_records"],
            full_cpu_ms_total=d["full_cpu_ms_total"],
            full_wall_ms_total=d.get("full_wall_ms_total", 0),
            incremental_cpu_ms_total=d["incremental_cpu_ms_total"],
            incremental_wall_ms_total=d.get("incremental_wall_ms_total", 0),
            compute_saved_cpu_ms=d.get("compute_saved_cpu_ms"),
            compute_saved_wall_ms=d.get("compute_saved_wall_ms"),
            savings_rate_bp=d.get("savings_rate_bp"),
            proof_cache_hits=d.get("proof_cache_hits", 0),
            proof_cache_misses=d.get("proof_cache_misses", 0),
            proof_cache_reuse_rate_bp=d.get("proof_cache_reuse_rate_bp"),
            model_calls=d.get("model_calls", 0),
            model_tokens=d.get("model_tokens", 0),
            cost_per_critical_gap_cpu_ms=d.get("cost_per_critical_gap_cpu_ms"),
            cost_per_promotion_cpu_ms=d.get("cost_per_promotion_cpu_ms"),
            avg_full_cost_per_mutant_cpu_ms=d.get("avg_full_cost_per_mutant_cpu_ms"),
            avg_incremental_cost_per_mutant_cpu_ms=d.get(
                "avg_incremental_cost_per_mutant_cpu_ms"
            ),
            member_ids=tuple(d.get("member_ids") or ()),
        )

    pops_raw = dict(_mapping(payload.get("populations") or {}, "populations"))
    populations = {
        kind: MetricsPopulation.from_dict(pops_raw[kind])
        for kind in METRICS_POPULATION_KINDS
        if kind in pops_raw
    }

    return AssuranceMetrics(
        interface_id=payload.get("interface_id", ASSURANCE_METRICS_INTERFACE),
        campaign_id=payload["campaign_id"],
        plan_id=payload.get("plan_id"),
        plan_cid=payload.get("plan_cid"),
        result_cid=payload.get("result_cid"),
        repository_state_cid=payload.get("repository_state_cid"),
        mutation_coverage=_cov(payload["mutation_coverage"]),
        detection_quality=_det(payload["detection_quality"]),
        gaps=_gap(payload["gaps"]),
        remediation=_rem(payload["remediation"]),
        economics=_eco(payload["economics"]),
        populations=populations,
        reason_codes=tuple(payload.get("reason_codes") or ()),
        notes=payload.get("notes"),
        production_policy_changed=bool(
            payload.get("production_policy_changed", False)
        ),
        metadata=payload.get("metadata") or {},
    )


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------


def _outcome_status_of(item: Mapping[str, Any]) -> str:
    raw = (
        item.get("outcome_status")
        or item.get("terminal_status")
        or item.get("disposition")
    )
    if raw is None:
        raise MetricsError(
            "outcome requires outcome_status",
            reason_code="missing_outcome_status",
        )
    status = _token(raw, "outcome_status")
    if status not in _CLOSED_OUTCOMES:
        # Allow coarse campaign statuses used by CLI projections.
        coarse = {
            "killed": MutationOutcomeStatus.KILLED_BY_TEST.value,
            "survivor": MutationOutcomeStatus.SURVIVED_SELECTED_VERIFICATION.value,
            "survived": MutationOutcomeStatus.SURVIVED_SELECTED_VERIFICATION.value,
            "invalid": MutationOutcomeStatus.INVALID_MUTANT.value,
            "equivalent": MutationOutcomeStatus.EQUIVALENT.value,
            "inconclusive": MutationOutcomeStatus.INCONCLUSIVE.value,
            "timeout": MutationOutcomeStatus.TIMEOUT.value,
            "infrastructure_failure": (
                MutationOutcomeStatus.INFRASTRUCTURE_FAILURE.value
            ),
            "uncompilable": MutationOutcomeStatus.UNCOMPILABLE.value,
            "human_review_required": (
                MutationOutcomeStatus.HUMAN_REVIEW_REQUIRED.value
            ),
            "complete": MutationOutcomeStatus.INCONCLUSIVE.value,
        }
        if status in coarse:
            status = coarse[status]
        else:
            raise MetricsError(
                f"unknown outcome_status {status!r}",
                reason_code="unknown_outcome",
            )
    return status


def _candidate_id_of(item: Mapping[str, Any], index: int) -> str:
    raw = (
        item.get("candidate_id")
        or item.get("mutant_id")
        or item.get("outcome_id")
        or item.get("id")
    )
    if raw is None:
        return f"candidate_{index}"
    return _token(str(raw), "candidate_id")


def _detector_ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise MetricsError(
            "detector id list must be a sequence of strings",
            reason_code="invalid_type",
        )
    if isinstance(value, (set, frozenset)):
        value = sorted(value)
    elif not isinstance(value, Sequence):
        raise MetricsError(
            "detector id list must be a sequence",
            reason_code="invalid_type",
        )
    out: list[str] = []
    for item in value:
        out.append(_token(item, "detector_id"))
    return _stable_unique(out)


def _detector_kind_map(item: Mapping[str, Any]) -> Mapping[str, str]:
    """Return detector_id -> kind when provided on an outcome mapping."""

    raw = item.get("detector_kinds") or item.get("detector_kind_by_id") or {}
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        kid = _token(key, "detector_kinds key")
        kind = _token(value, "detector_kinds value")
        out[kid] = kind
    return out


def _classification_of(item: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    raw = item.get("detector_classification")
    if isinstance(raw, Mapping):
        return {
            "predicted": _detector_ids(raw.get("predicted_detector_ids")),
            "selected": _detector_ids(raw.get("selected_detector_ids")),
            "executed": _detector_ids(raw.get("executed_detector_ids")),
            "observed": _detector_ids(raw.get("observed_detector_ids")),
            "missed": _detector_ids(
                raw.get("missed_detector_ids")
                or (
                    set(_detector_ids(raw.get("predicted_detector_ids")))
                    - set(_detector_ids(raw.get("observed_detector_ids")))
                )
            ),
            "unexpected": _detector_ids(
                raw.get("unexpected_detector_ids")
                or (
                    set(_detector_ids(raw.get("observed_detector_ids")))
                    - set(_detector_ids(raw.get("predicted_detector_ids")))
                )
            ),
        }
    # Flat field fallbacks.
    predicted = _detector_ids(item.get("predicted_detector_ids"))
    selected = _detector_ids(item.get("selected_detector_ids") or predicted)
    executed = _detector_ids(item.get("executed_detector_ids") or selected)
    observed = _detector_ids(item.get("observed_detector_ids"))
    missed = _detector_ids(
        item.get("missed_detector_ids") or (set(predicted) - set(observed))
    )
    unexpected = _detector_ids(
        item.get("unexpected_detector_ids") or (set(observed) - set(predicted))
    )
    return {
        "predicted": predicted,
        "selected": selected,
        "executed": executed,
        "observed": observed,
        "missed": missed,
        "unexpected": unexpected,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def compute_assurance_metrics(
    *,
    campaign_id: str,
    outcomes: Sequence[Mapping[str, Any]] | None = None,
    gaps: Sequence[Mapping[str, Any]] | None = None,
    remediations: Sequence[Mapping[str, Any]] | None = None,
    economics_records: Sequence[Mapping[str, Any]] | None = None,
    plan_id: str | None = None,
    plan_cid: str | None = None,
    result_cid: str | None = None,
    repository_state_cid: str | None = None,
    generated_count: int | None = None,
    admitted_count: int | None = None,
    critical_gap_count: int | None = None,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AssuranceMetrics:
    """Aggregate disjoint campaign metrics from pure input records.

    Parameters accept plain mappings so CLI campaign results, sealed
    ``MutationOutcome`` projections, gap reports, remediation evaluations, and
    cost accounts can feed the same path without host I/O.
    """

    outcome_items = list(_sequence(outcomes, "outcomes"))
    gap_items = list(_sequence(gaps, "gaps"))
    rem_items = list(_sequence(remediations, "remediations"))
    eco_items = list(_sequence(economics_records, "economics_records"))

    # --- mutation coverage population ---
    coverage_members: list[str] = []
    outcome_counts: dict[str, int] = {status: 0 for status in mutation_outcome_statuses()}
    class_totals: dict[str, int] = {}
    class_kills: dict[str, int] = {}
    risk_weight_total = 0
    risk_weight_killed = 0
    scoring_denominator = 0
    killed_count = 0
    selected_survivor_count = 0
    full_survivor_count = 0
    invalid_count = 0
    uncompilable_count = 0
    infra_count = 0
    timeout_count = 0
    inconclusive_count = 0
    equivalent_count = 0
    probably_equivalent_count = 0
    human_review_count = 0
    excluded_count = 0

    # --- detection quality population ---
    detection_members: list[str] = []
    predicted_n = selected_n = executed_n = observed_n = missed_n = unexpected_n = 0
    selected_test = selected_proof = selected_policy = 0
    full_suite_only = 0
    selected_kind_total = 0

    for index, raw in enumerate(outcome_items):
        item = dict(_mapping(raw, f"outcomes[{index}]"))
        candidate_id = _candidate_id_of(item, index)
        status = _outcome_status_of(item)
        mid = population_member_id(MetricsPopulationKind.MUTATION_COVERAGE, candidate_id)
        coverage_members.append(mid)
        outcome_counts[status] = outcome_counts.get(status, 0) + 1

        bucket = coverage_bucket(status)
        if bucket == "killed":
            killed_count += 1
        elif bucket == "selected_survivor":
            selected_survivor_count += 1
        elif bucket == "full_survivor":
            full_survivor_count += 1
        elif status == MutationOutcomeStatus.INVALID_MUTANT.value:
            invalid_count += 1
        elif status == MutationOutcomeStatus.UNCOMPILABLE.value:
            uncompilable_count += 1
        elif status == MutationOutcomeStatus.INFRASTRUCTURE_FAILURE.value:
            infra_count += 1
        elif status == MutationOutcomeStatus.TIMEOUT.value:
            timeout_count += 1
        elif status == MutationOutcomeStatus.INCONCLUSIVE.value:
            inconclusive_count += 1
        elif status == MutationOutcomeStatus.EQUIVALENT.value:
            equivalent_count += 1
        elif status == MutationOutcomeStatus.PROBABLY_EQUIVALENT.value:
            probably_equivalent_count += 1
        elif status == MutationOutcomeStatus.HUMAN_REVIEW_REQUIRED.value:
            human_review_count += 1

        excluded = is_denominator_excluded(status)
        if excluded:
            excluded_count += 1
        else:
            scoring_denominator += 1
            # Risk-weighted score uses risk_weight_bp when present (default 1).
            weight = item.get("risk_weight_bp", 1)
            weight_i = _nonneg_int(weight, "risk_weight_bp")
            if weight_i > MAX_RISK_WEIGHT_BP:
                weight_i = MAX_RISK_WEIGHT_BP
            risk_weight_total += weight_i
            if counts_as_killed(status):
                risk_weight_killed += weight_i

            operator_class = item.get("operator_class") or item.get("mutation_class") or "unspecified"
            op = _token(str(operator_class), "operator_class")
            class_totals[op] = class_totals.get(op, 0) + 1
            if counts_as_killed(status):
                class_kills[op] = class_kills.get(op, 0) + 1

        # Detection quality — separate population members per detector event.
        classification = _classification_of(item)
        kind_map = _detector_kind_map(item)
        killing_kind = item.get("killing_detector_kind")
        if killing_kind is not None and item.get("killing_detector_id"):
            kind_map = dict(kind_map)
            kind_map[_token(item["killing_detector_id"], "killing_detector_id")] = _token(
                killing_kind, "killing_detector_kind"
            )

        for role, ids in classification.items():
            for detector_id in ids:
                det_member = population_member_id(
                    MetricsPopulationKind.DETECTION_QUALITY,
                    f"{candidate_id}:{detector_id}:{role}",
                )
                detection_members.append(det_member)

        predicted_n += len(classification["predicted"])
        selected_n += len(classification["selected"])
        executed_n += len(classification["executed"])
        observed_n += len(classification["observed"])
        missed_n += len(classification["missed"])
        unexpected_n += len(classification["unexpected"])

        for detector_id in classification["selected"]:
            kind = kind_map.get(detector_id)
            selected_kind_total += 1
            if kind in _TEST_LIKE_KINDS:
                selected_test += 1
            elif kind in _PROOF_LIKE_KINDS:
                selected_proof += 1
            elif kind in _POLICY_LIKE_KINDS:
                selected_policy += 1

        # Full-suite-only: killed only by full suite (not by selected detectors).
        if status == MutationOutcomeStatus.KILLED_BY_FULL_SUITE.value:
            full_suite_only += 1
        elif (
            status in _KILLED_STATUSES
            and item.get("killing_detector_kind") == DetectorKind.FULL_SUITE.value
        ):
            full_suite_only += 1

    admitted = (
        len(outcome_items)
        if admitted_count is None
        else _nonneg_int(admitted_count, "admitted_count")
    )
    generated = (
        admitted if generated_count is None else _nonneg_int(generated_count, "generated_count")
    )
    if generated < admitted:
        # Generated cannot be less than admitted; treat declared generated as floor.
        generated = max(generated, admitted)

    kill_rate_bp = _optional_rate_bp(killed_count, scoring_denominator)
    if risk_weight_total > 0:
        risk_weighted_score_bp = (risk_weight_killed * BASIS_POINTS) // risk_weight_total
        if risk_weighted_score_bp > BASIS_POINTS:
            risk_weighted_score_bp = BASIS_POINTS
    else:
        risk_weighted_score_bp = None

    class_kill_rates: dict[str, int | None] = {}
    for op in sorted(class_totals):
        class_kill_rates[op] = _optional_rate_bp(
            class_kills.get(op, 0), class_totals[op]
        )

    coverage = MutationCoverageMetrics(
        generated_count=generated,
        admitted_count=admitted,
        invalid_count=invalid_count,
        uncompilable_count=uncompilable_count,
        infrastructure_failure_count=infra_count,
        timeout_count=timeout_count,
        inconclusive_count=inconclusive_count,
        equivalent_count=equivalent_count,
        probably_equivalent_count=probably_equivalent_count,
        killed_count=killed_count,
        selected_survivor_count=selected_survivor_count,
        full_survivor_count=full_survivor_count,
        human_review_count=human_review_count,
        denominator_excluded_count=excluded_count,
        scoring_denominator=scoring_denominator,
        kill_rate_bp=kill_rate_bp,
        risk_weighted_score_bp=risk_weighted_score_bp,
        class_kill_rates_bp=class_kill_rates,
        outcome_counts={k: v for k, v in sorted(outcome_counts.items()) if v > 0},
        member_ids=tuple(coverage_members),
    )

    detection = DetectionQualityMetrics(
        predicted_detector_count=predicted_n,
        selected_detector_count=selected_n,
        executed_detector_count=executed_n,
        observed_detector_count=observed_n,
        missed_detector_count=missed_n,
        unexpected_detector_count=unexpected_n,
        selected_test_rate_bp=_optional_rate_bp(selected_test, selected_kind_total),
        selected_proof_rate_bp=_optional_rate_bp(selected_proof, selected_kind_total),
        selected_policy_rate_bp=_optional_rate_bp(selected_policy, selected_kind_total),
        full_suite_only_detection_count=full_suite_only,
        full_suite_only_rate_bp=_optional_rate_bp(full_suite_only, max(killed_count, 0)),
        member_ids=tuple(detection_members),
    )

    # --- gap population ---
    gap_members: list[str] = []
    category_counts: dict[str, int] = {}
    high_risk_gaps = 0
    for index, raw in enumerate(gap_items):
        item = dict(_mapping(raw, f"gaps[{index}]"))
        gap_id = item.get("gap_id") or item.get("id") or f"gap_{index}"
        gap_id = _token(str(gap_id), "gap_id")
        gap_members.append(
            population_member_id(MetricsPopulationKind.GAP, gap_id)
        )
        gap_class = (
            item.get("gap_class")
            or item.get("category")
            or item.get("assurance_gap_class")
            or "unknown"
        )
        gap_class = _token(str(gap_class), "gap_class")
        if gap_class not in _GAP_CLASS_VALUES:
            gap_class = AssuranceGapClass.UNKNOWN.value
        category_counts[gap_class] = category_counts.get(gap_class, 0) + 1
        risk = str(item.get("risk_class") or item.get("severity") or "").lower()
        if risk in {
            "critical",
            "critical_security",
            "authorization",
            "financial_legal",
            "high",
        }:
            high_risk_gaps += 1
        elif item.get("high_risk") is True:
            high_risk_gaps += 1

    gap_metrics = GapMetrics(
        total_gaps=len(gap_items),
        high_risk_survivor_gaps=high_risk_gaps,
        category_counts=category_counts,
        member_ids=tuple(gap_members),
    )

    # --- remediation population ---
    rem_members: list[str] = []
    held_out_kills = 0
    regressions = 0
    overconstraints = 0
    accepted = 0
    rejected = 0
    evaluated = 0
    rem_cpu = 0
    rem_wall = 0
    for index, raw in enumerate(rem_items):
        item = dict(_mapping(raw, f"remediations[{index}]"))
        rem_id = (
            item.get("remediation_id")
            or item.get("candidate_id")
            or item.get("plan_id")
            or item.get("id")
            or f"remediation_{index}"
        )
        rem_id = _token(str(rem_id), "remediation_id")
        rem_members.append(
            population_member_id(MetricsPopulationKind.REMEDIATION, rem_id)
        )
        disposition = str(
            item.get("disposition")
            or item.get("verdict")
            or item.get("status")
            or ""
        ).lower()
        if item.get("evaluated") is True or disposition in {
            "evaluated",
            "qualified",
            "rejected",
            "accepted",
            "promoted",
            "failed",
        }:
            evaluated += 1
        held_out_kills += _nonneg_int(
            item.get("held_out_kill_count", 0), "held_out_kill_count"
        )
        if item.get("regression") is True or "regression" in disposition:
            regressions += 1
        if item.get("overconstraint") is True or "overconstraint" in disposition:
            overconstraints += 1
        if disposition in {"accepted", "promoted", "qualified"}:
            accepted += 1
        if disposition in {"rejected", "failed", "blocked"}:
            rejected += 1
        rem_cpu += _nonneg_int(item.get("cost_cpu_ms", 0), "cost_cpu_ms")
        rem_wall += _nonneg_int(item.get("cost_wall_ms", 0), "cost_wall_ms")

    remediation = RemediationMetrics(
        candidate_count=len(rem_items),
        evaluated_count=evaluated,
        held_out_kill_count=held_out_kills,
        regression_count=regressions,
        overconstraint_count=overconstraints,
        accepted_promotion_count=accepted,
        rejected_promotion_count=rejected,
        total_cost_cpu_ms=rem_cpu,
        total_cost_wall_ms=rem_wall,
        member_ids=tuple(rem_members),
    )

    # --- economics population ---
    eco_members: list[str] = []
    full_cpu = full_wall = inc_cpu = inc_wall = 0
    cache_hits = cache_misses = 0
    model_calls = model_tokens = 0
    saved_cpu: int | None = 0
    saved_wall: int | None = 0
    measured_any = False
    for index, raw in enumerate(eco_items):
        item = dict(_mapping(raw, f"economics_records[{index}]"))
        eco_id = (
            item.get("economics_id")
            or item.get("candidate_id")
            or item.get("mutant_id")
            or item.get("id")
            or f"economics_{index}"
        )
        eco_id = _token(str(eco_id), "economics_id")
        eco_members.append(
            population_member_id(MetricsPopulationKind.ECONOMICS, eco_id)
        )
        measured_any = True
        full_cpu += _nonneg_int(item.get("full_cpu_ms", 0), "full_cpu_ms")
        full_wall += _nonneg_int(item.get("full_wall_ms", 0), "full_wall_ms")
        inc_cpu += _nonneg_int(
            item.get("incremental_cpu_ms", 0), "incremental_cpu_ms"
        )
        inc_wall += _nonneg_int(
            item.get("incremental_wall_ms", 0), "incremental_wall_ms"
        )
        cache_hits += _nonneg_int(item.get("cache_hits", 0), "cache_hits")
        cache_misses += _nonneg_int(item.get("cache_misses", 0), "cache_misses")
        model_calls += _nonneg_int(item.get("model_calls", 0), "model_calls")
        model_tokens += _nonneg_int(item.get("model_tokens", 0), "model_tokens")
        if item.get("compute_saved_cpu_ms") is None and item.get("measured") is False:
            saved_cpu = None if saved_cpu is None else saved_cpu
        else:
            if saved_cpu is not None:
                raw_saved = item.get("compute_saved_cpu_ms")
                if raw_saved is None:
                    # Derive when both sides present.
                    derived = full_cpu  # placeholder; recompute below
                    del derived
                else:
                    saved_cpu += _nonneg_int(raw_saved, "compute_saved_cpu_ms")
            if saved_wall is not None:
                raw_saved_w = item.get("compute_saved_wall_ms")
                if raw_saved_w is not None:
                    saved_wall += _nonneg_int(raw_saved_w, "compute_saved_wall_ms")

    n_eco = len(eco_items)
    if measured_any and n_eco > 0:
        # Prefer explicit saved sums; otherwise derive from totals.
        if saved_cpu == 0 and full_cpu >= inc_cpu:
            saved_cpu = full_cpu - inc_cpu
        if saved_wall == 0 and full_wall >= inc_wall:
            saved_wall = full_wall - inc_wall
        savings_rate_bp = _optional_rate_bp(saved_cpu or 0, full_cpu)
        cache_total = cache_hits + cache_misses
        cache_rate = _optional_rate_bp(cache_hits, cache_total)
        avg_full = full_cpu // n_eco if n_eco else None
        avg_inc = inc_cpu // n_eco if n_eco else None
    else:
        saved_cpu = None
        saved_wall = None
        savings_rate_bp = None
        cache_rate = None
        avg_full = None
        avg_inc = None

    crit = (
        high_risk_gaps
        if critical_gap_count is None
        else _nonneg_int(critical_gap_count, "critical_gap_count")
    )
    cost_per_gap = (inc_cpu // crit) if crit > 0 else None
    promo_n = accepted + rejected
    cost_per_promo = (rem_cpu // promo_n) if promo_n > 0 else None

    economics = EconomicsMetrics(
        mutant_cost_records=n_eco,
        full_cpu_ms_total=full_cpu,
        full_wall_ms_total=full_wall,
        incremental_cpu_ms_total=inc_cpu,
        incremental_wall_ms_total=inc_wall,
        compute_saved_cpu_ms=saved_cpu,
        compute_saved_wall_ms=saved_wall,
        savings_rate_bp=savings_rate_bp,
        proof_cache_hits=cache_hits,
        proof_cache_misses=cache_misses,
        proof_cache_reuse_rate_bp=cache_rate,
        model_calls=model_calls,
        model_tokens=model_tokens,
        cost_per_critical_gap_cpu_ms=cost_per_gap,
        cost_per_promotion_cpu_ms=cost_per_promo,
        avg_full_cost_per_mutant_cpu_ms=avg_full,
        avg_incremental_cost_per_mutant_cpu_ms=avg_inc,
        member_ids=tuple(eco_members),
    )

    coverage_members_u = _sorted_unique(coverage_members)
    detection_members_u = _sorted_unique(detection_members)
    gap_members_u = _sorted_unique(gap_members)
    rem_members_u = _sorted_unique(rem_members)
    eco_members_u = _sorted_unique(eco_members)

    # Sub-metrics store the same unique membership as sealed populations.
    coverage = MutationCoverageMetrics(
        generated_count=coverage.generated_count,
        admitted_count=coverage.admitted_count,
        invalid_count=coverage.invalid_count,
        uncompilable_count=coverage.uncompilable_count,
        infrastructure_failure_count=coverage.infrastructure_failure_count,
        timeout_count=coverage.timeout_count,
        inconclusive_count=coverage.inconclusive_count,
        equivalent_count=coverage.equivalent_count,
        probably_equivalent_count=coverage.probably_equivalent_count,
        killed_count=coverage.killed_count,
        selected_survivor_count=coverage.selected_survivor_count,
        full_survivor_count=coverage.full_survivor_count,
        human_review_count=coverage.human_review_count,
        denominator_excluded_count=coverage.denominator_excluded_count,
        scoring_denominator=coverage.scoring_denominator,
        kill_rate_bp=coverage.kill_rate_bp,
        risk_weighted_score_bp=coverage.risk_weighted_score_bp,
        class_kill_rates_bp=dict(coverage.class_kill_rates_bp),
        outcome_counts=dict(coverage.outcome_counts),
        member_ids=coverage_members_u,
    )
    detection = DetectionQualityMetrics(
        predicted_detector_count=detection.predicted_detector_count,
        selected_detector_count=detection.selected_detector_count,
        executed_detector_count=detection.executed_detector_count,
        observed_detector_count=detection.observed_detector_count,
        missed_detector_count=detection.missed_detector_count,
        unexpected_detector_count=detection.unexpected_detector_count,
        selected_test_rate_bp=detection.selected_test_rate_bp,
        selected_proof_rate_bp=detection.selected_proof_rate_bp,
        selected_policy_rate_bp=detection.selected_policy_rate_bp,
        full_suite_only_detection_count=detection.full_suite_only_detection_count,
        full_suite_only_rate_bp=detection.full_suite_only_rate_bp,
        member_ids=detection_members_u,
    )
    gap_metrics = GapMetrics(
        total_gaps=gap_metrics.total_gaps,
        high_risk_survivor_gaps=gap_metrics.high_risk_survivor_gaps,
        category_counts=dict(gap_metrics.category_counts),
        member_ids=gap_members_u,
    )
    remediation = RemediationMetrics(
        candidate_count=remediation.candidate_count,
        evaluated_count=remediation.evaluated_count,
        held_out_kill_count=remediation.held_out_kill_count,
        regression_count=remediation.regression_count,
        overconstraint_count=remediation.overconstraint_count,
        accepted_promotion_count=remediation.accepted_promotion_count,
        rejected_promotion_count=remediation.rejected_promotion_count,
        total_cost_cpu_ms=remediation.total_cost_cpu_ms,
        total_cost_wall_ms=remediation.total_cost_wall_ms,
        member_ids=rem_members_u,
    )
    economics = EconomicsMetrics(
        mutant_cost_records=economics.mutant_cost_records,
        full_cpu_ms_total=economics.full_cpu_ms_total,
        full_wall_ms_total=economics.full_wall_ms_total,
        incremental_cpu_ms_total=economics.incremental_cpu_ms_total,
        incremental_wall_ms_total=economics.incremental_wall_ms_total,
        compute_saved_cpu_ms=economics.compute_saved_cpu_ms,
        compute_saved_wall_ms=economics.compute_saved_wall_ms,
        savings_rate_bp=economics.savings_rate_bp,
        proof_cache_hits=economics.proof_cache_hits,
        proof_cache_misses=economics.proof_cache_misses,
        proof_cache_reuse_rate_bp=economics.proof_cache_reuse_rate_bp,
        model_calls=economics.model_calls,
        model_tokens=economics.model_tokens,
        cost_per_critical_gap_cpu_ms=economics.cost_per_critical_gap_cpu_ms,
        cost_per_promotion_cpu_ms=economics.cost_per_promotion_cpu_ms,
        avg_full_cost_per_mutant_cpu_ms=economics.avg_full_cost_per_mutant_cpu_ms,
        avg_incremental_cost_per_mutant_cpu_ms=(
            economics.avg_incremental_cost_per_mutant_cpu_ms
        ),
        member_ids=eco_members_u,
    )

    populations = {
        MetricsPopulationKind.MUTATION_COVERAGE.value: MetricsPopulation(
            kind=MetricsPopulationKind.MUTATION_COVERAGE.value,
            member_ids=coverage_members_u,
            count=len(coverage_members_u),
        ),
        MetricsPopulationKind.DETECTION_QUALITY.value: MetricsPopulation(
            kind=MetricsPopulationKind.DETECTION_QUALITY.value,
            member_ids=detection_members_u,
            count=len(detection_members_u),
        ),
        MetricsPopulationKind.GAP.value: MetricsPopulation(
            kind=MetricsPopulationKind.GAP.value,
            member_ids=gap_members_u,
            count=len(gap_members_u),
        ),
        MetricsPopulationKind.REMEDIATION.value: MetricsPopulation(
            kind=MetricsPopulationKind.REMEDIATION.value,
            member_ids=rem_members_u,
            count=len(rem_members_u),
        ),
        MetricsPopulationKind.ECONOMICS.value: MetricsPopulation(
            kind=MetricsPopulationKind.ECONOMICS.value,
            member_ids=eco_members_u,
            count=len(eco_members_u),
        ),
    }

    reasons = [
        "metrics_computed",
        "populations_disjoint",
        "denominators_exclude_invalid_equivalent_infrastructure",
        "no_production_policy_change",
    ]
    if excluded_count:
        reasons.append("denominator_exclusions_applied")
    if kill_rate_bp is None:
        reasons.append("kill_rate_unavailable_empty_denominator")

    return AssuranceMetrics(
        interface_id=ASSURANCE_METRICS_INTERFACE,
        campaign_id=_token(campaign_id, "campaign_id"),
        plan_id=None if plan_id is None else _token(plan_id, "plan_id"),
        plan_cid=_optional_cid(plan_cid, "plan_cid") if plan_cid else None,
        result_cid=_optional_cid(result_cid, "result_cid") if result_cid else None,
        repository_state_cid=(
            _optional_cid(repository_state_cid, "repository_state_cid")
            if repository_state_cid
            else None
        ),
        mutation_coverage=coverage,
        detection_quality=detection,
        gaps=gap_metrics,
        remediation=remediation,
        economics=economics,
        populations=populations,
        reason_codes=tuple(reasons),
        notes=_optional_text(notes, "notes"),
        production_policy_changed=False,
        metadata=metadata or {},
    )


def metrics_descriptor() -> Mapping[str, Any]:
    """Describe the metrics surface for discovery / CLI wiring."""

    return MappingProxyType(
        {
            "interface": ASSURANCE_METRICS_INTERFACE,
            "schema": ASSURANCE_METRICS_SCHEMA,
            "evidence": AAE_METRICS_EVIDENCE,
            "adapter_id": ADAPTER_ID,
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "populations": list(METRICS_POPULATION_KINDS),
            "denominator_excluded_outcomes": sorted(DENOMINATOR_EXCLUDED_OUTCOMES),
            "killed_outcome_statuses": list(killed_outcome_statuses()),
            "never_counted_as_killed": list(never_counted_as_killed_statuses()),
            "api": "compute_assurance_metrics",
            "production_policy_change": False,
        }
    )


def denominator_excluded_outcomes() -> tuple[str, ...]:
    """Return outcome statuses excluded from kill-rate denominators."""

    return tuple(sorted(DENOMINATOR_EXCLUDED_OUTCOMES))


def metrics_population_kinds() -> tuple[str, ...]:
    """Return the closed metrics population vocabulary."""

    return METRICS_POPULATION_KINDS


__all__ = [
    "AAE_METRICS_EVIDENCE",
    "ADAPTER_ID",
    "ASSURANCE_METRICS_INTERFACE",
    "ASSURANCE_METRICS_SCHEMA",
    "BASIS_POINTS",
    "BOARD_NAMESPACE",
    "DENOMINATOR_EXCLUDED_OUTCOMES",
    "DETECTION_QUALITY_METRICS_SCHEMA",
    "ECONOMICS_METRICS_SCHEMA",
    "EQUIVALENCE_EXCLUDED_OUTCOMES",
    "GAP_METRICS_SCHEMA",
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "INFRASTRUCTURE_EXCLUDED_OUTCOMES",
    "METRICS_POPULATION_KINDS",
    "METRICS_POPULATION_SCHEMA",
    "MUTATION_COVERAGE_METRICS_SCHEMA",
    "PRODUCER_ID",
    "PRODUCER_VERSION",
    "REMEDIATION_METRICS_SCHEMA",
    "TOOL_ID",
    "AssuranceMetrics",
    "DetectionQualityMetrics",
    "EconomicsMetrics",
    "GapMetrics",
    "MetricsError",
    "MetricsPopulation",
    "MetricsPopulationKind",
    "MutationCoverageMetrics",
    "RemediationMetrics",
    "assert_populations_disjoint",
    "assurance_metrics_from_dict",
    "compute_assurance_metrics",
    "coverage_bucket",
    "denominator_excluded_outcomes",
    "is_denominator_excluded",
    "is_equivalence_excluded",
    "is_infrastructure_excluded",
    "metrics_descriptor",
    "metrics_population_kinds",
    "population_member_id",
    "verify_assurance_metrics_identity",
]
