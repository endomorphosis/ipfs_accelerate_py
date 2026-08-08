"""State, latency, and LLM-churn baselines for DuckDB/Quack control plane (DQP-009).

Interfaces:
* ``SupervisorStateBaseline@1``
* ``LLMChurnBaseline@1``

This module is measurement-only.  It never grants completion, mutation,
promotion, or provider authority.  Baselines bind tree, environment, workload,
and metric definitions into a content-addressed record so later canaries
(DQP-036) compare against a fixed hermetic reference.

Contract rules (fail-closed):

* Every observation is either ``measured`` (value + sensor) or ``unavailable``
  (reason + sensor).  Missing telemetry is never encoded as numeric zero.
* Measured zeros require an explicit sensor receipt.
* Rejected, retry, and abandoned provider usage is charged, not dropped.
* Regeneration refuses any weakening of locked safety, durability, or quality
  criteria relative to the prior baseline.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final


# ---------------------------------------------------------------------------
# Interface / schema identities
# ---------------------------------------------------------------------------

SUPERVISOR_STATE_BASELINE_INTERFACE: Final[str] = "SupervisorStateBaseline@1"
LLM_CHURN_BASELINE_INTERFACE: Final[str] = "LLMChurnBaseline@1"
DUCKDB_QUACK_BASELINE_INTERFACE: Final[str] = "DuckDBQuackBaseline@1"

BASELINE_CONTRACT_VERSION: Final[int] = 1
BASELINE_TASK_ID: Final[str] = "DQP-009"
BASELINE_GOAL_ID: Final[str] = "DQP-G050"
BASELINE_EVIDENCE: Final[str] = "dqp/duckdb-quack-baseline@1"

SCHEMA_PREFIX: Final[str] = "ipfs_accelerate_py/agent-supervisor"
BASELINE_BINDING_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/duckdb-quack-baseline-binding@1"
METRIC_DEFINITION_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/duckdb-quack-metric-definition@1"
METRIC_OBSERVATION_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/duckdb-quack-metric-observation@1"
PROVIDER_USAGE_CHARGE_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/duckdb-quack-provider-usage-charge@1"
)
LOCKED_CRITERIA_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/duckdb-quack-locked-criteria@1"
STATE_BASELINE_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/supervisor-state-baseline@1"
LLM_CHURN_BASELINE_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/llm-churn-baseline@1"
COMBINED_BASELINE_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/duckdb-quack-baseline@1"

MAX_TEXT_BYTES: Final[int] = 512
MAX_COUNTER: Final[int] = 10**18
MAX_OBSERVATIONS: Final[int] = 100_000
MAX_SAMPLES: Final[int] = 10_000
BASIS_POINTS: Final[int] = 10_000

MISSING_TELEMETRY_SENSOR: Final[str] = "sensor:missing-telemetry@1"
HERMETIC_WORKLOAD_SENSOR: Final[str] = "sensor:hermetic-workload@1"
PROVIDER_LEDGER_SENSOR: Final[str] = "sensor:provider-usage-ledger@1"


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class SampleStatus(str, Enum):
    """Whether a metric observation was measured or is typed unavailable."""

    MEASURED = "measured"
    UNAVAILABLE = "unavailable"


class UnavailableReason(str, Enum):
    """Closed reasons for missing metric telemetry."""

    TELEMETRY_MISSING = "telemetry-missing"
    SENSOR_ABSENT = "sensor-absent"
    PROVIDER_OMITTED = "provider-omitted"
    COLLECTION_FAILED = "collection-failed"
    STRATUM_NOT_RUN = "stratum-not-run"


class BaselineStratum(str, Enum):
    """Measurement strata required by the DQP-009 evidence subset."""

    COLD = "cold"
    WARM = "warm"
    RESTART = "restart"
    PARALLEL = "parallel"


class MetricCategory(str, Enum):
    """Closed metric categories for state, latency, churn, and floors."""

    STATE = "state"
    LATENCY = "latency"
    LLM_CHURN = "llm_churn"
    QUALITY = "quality"
    SAFETY = "safety"
    DURABILITY = "durability"
    PROVIDER_USAGE = "provider_usage"


class MetricUnit(str, Enum):
    COUNT = "count"
    BYTES = "bytes"
    TOKENS = "tokens"
    MILLISECONDS = "milliseconds"
    BASIS_POINTS = "basis_points"
    RATIO_MILLIONTHS = "ratio_millionths"


class ProviderUsageDisposition(str, Enum):
    """Provider usage outcomes that remain charged in churn accounting.

    Rejected, retry, and abandoned work is never free: every disposition is
    counted so token/call reductions cannot hide failed spend.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    RETRY = "retry"
    ABANDONED = "abandoned"

    @property
    def is_charged(self) -> bool:
        """All dispositions are charged (including rejected/retry/abandoned)."""

        return True


class FloorKind(str, Enum):
    """How locked criteria compare on regeneration."""

    MAXIMUM = "maximum"  # safety/durability: cannot raise the allowance
    MINIMUM = "minimum"  # quality: cannot lower the floor


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DuckDBQuackBaselineError(ValueError):
    """Raised when baseline inputs are unsafe, incomplete, or weakened."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, required: bool = True, maximum: int = MAX_TEXT_BYTES) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        raise DuckDBQuackBaselineError(f"{name} must be text")
    result = value.strip()
    if required and not result:
        raise DuckDBQuackBaselineError(f"{name} must not be empty")
    if "\x00" in result:
        raise DuckDBQuackBaselineError(f"{name} contains a NUL byte")
    if len(result.encode("utf-8")) > maximum:
        raise DuckDBQuackBaselineError(f"{name} exceeds its {maximum}-byte bound")
    return result


def _nonnegative_int(
    value: Any,
    name: str,
    *,
    maximum: int = MAX_COUNTER,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DuckDBQuackBaselineError(
            f"{name} must be a non-negative integer (negative counts rejected)"
        )
    if value < 0:
        raise DuckDBQuackBaselineError(
            f"{name} must be a non-negative integer (negative counts rejected)"
        )
    if value > maximum:
        raise DuckDBQuackBaselineError(f"{name} exceeds its maximum of {maximum}")
    return value


def _parse_enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, Enum):
        value = value.value
    text = _text(value, name)
    try:
        return enum_type(text)
    except ValueError as exc:
        allowed = ", ".join(sorted(item.value for item in enum_type))
        raise DuckDBQuackBaselineError(
            f"{name} must be one of {{{allowed}}}; got {text!r}"
        ) from exc


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DuckDBQuackBaselineError(
            "baseline payloads require canonical JSON values"
        ) from exc


def content_identity(value: Any) -> str:
    """Return a stable sha256 content identity for a JSON-compatible value."""

    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# ---------------------------------------------------------------------------
# Locked safety / durability / quality criteria
# ---------------------------------------------------------------------------

# Release safety floors from the DuckDB/Quack control-plane plan and scheduler.
# All are maximums of zero (non-compensable).
DEFAULT_SAFETY_FLOORS: Final[Mapping[str, int]] = {
    "duplicate_non_idempotent_effects": 0,
    "stale_lease_writes": 0,
    "unauthorized_sql_effects": 0,
    "secret_leaks": 0,
    "false_completions": 0,
    "incomplete_mutation_lineage_admissions": 0,
    "open_impact_frontier_automatic_mutations": 0,
    "legacy_export_authority_reads_in_canary": 0,
    "silent_quack_to_file_fallbacks": 0,
}

DEFAULT_DURABILITY_FLOORS: Final[Mapping[str, int]] = {
    "accepted_state_losses": 0,
    "event_projection_divergences": 0,
}

# Quality floors are minimums (basis points).  Regeneration cannot lower them.
DEFAULT_QUALITY_FLOORS: Final[Mapping[str, int]] = {
    "accepted_mutation_quality_bps": 8_000,
    "cache_reuse_rate_bps": 0,
}


# ---------------------------------------------------------------------------
# Metric definition catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricDefinition:
    """One named metric definition bound into a baseline."""

    SCHEMA: ClassVar[str] = METRIC_DEFINITION_SCHEMA

    name: str
    category: MetricCategory
    unit: MetricUnit
    description: str
    higher_is_better: bool = False
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "name", maximum=128))
        object.__setattr__(
            self, "category", _parse_enum(self.category, MetricCategory, "category")
        )
        object.__setattr__(self, "unit", _parse_enum(self.unit, MetricUnit, "unit"))
        object.__setattr__(
            self, "description", _text(self.description, "description", maximum=512)
        )
        if not isinstance(self.higher_is_better, bool):
            raise DuckDBQuackBaselineError("higher_is_better must be a boolean")
        if not isinstance(self.required, bool):
            raise DuckDBQuackBaselineError("required must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "name": self.name,
            "category": self.category.value,
            "unit": self.unit.value,
            "description": self.description,
            "higher_is_better": self.higher_is_better,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetricDefinition":
        if not isinstance(payload, Mapping):
            raise DuckDBQuackBaselineError("metric definition must be an object")
        claimed = payload.get("schema")
        if claimed is not None and claimed != cls.SCHEMA:
            raise DuckDBQuackBaselineError(
                f"metric definition has foreign schema {claimed!r}"
            )
        return cls(
            name=payload.get("name", ""),
            category=payload.get("category", ""),
            unit=payload.get("unit", ""),
            description=payload.get("description", ""),
            higher_is_better=bool(payload.get("higher_is_better", False)),
            required=bool(payload.get("required", True)),
        )


def default_state_metric_definitions() -> tuple[MetricDefinition, ...]:
    """State and latency metrics measured by SupervisorStateBaseline@1."""

    return (
        MetricDefinition(
            name="file_reads",
            category=MetricCategory.STATE,
            unit=MetricUnit.COUNT,
            description="File reads during the hermetic workload",
        ),
        MetricDefinition(
            name="file_writes",
            category=MetricCategory.STATE,
            unit=MetricUnit.COUNT,
            description="File writes during the hermetic workload",
        ),
        MetricDefinition(
            name="file_parses",
            category=MetricCategory.STATE,
            unit=MetricUnit.COUNT,
            description="File parses during the hermetic workload",
        ),
        MetricDefinition(
            name="independent_db_opens",
            category=MetricCategory.STATE,
            unit=MetricUnit.COUNT,
            description="Independent database open operations",
        ),
        MetricDefinition(
            name="lock_wait_ms",
            category=MetricCategory.LATENCY,
            unit=MetricUnit.MILLISECONDS,
            description="Aggregate lock wait time in milliseconds",
        ),
        MetricDefinition(
            name="noop_poll_count",
            category=MetricCategory.STATE,
            unit=MetricUnit.COUNT,
            description="No-op polling cycles that observed no change",
        ),
        MetricDefinition(
            name="task_claim_latency_ms",
            category=MetricCategory.LATENCY,
            unit=MetricUnit.MILLISECONDS,
            description="Task claim latency in milliseconds",
        ),
        MetricDefinition(
            name="queue_latency_ms",
            category=MetricCategory.LATENCY,
            unit=MetricUnit.MILLISECONDS,
            description="Queue wait latency in milliseconds",
        ),
        MetricDefinition(
            name="rollback_rate_bps",
            category=MetricCategory.QUALITY,
            unit=MetricUnit.BASIS_POINTS,
            description="Rollback rate in basis points",
        ),
        MetricDefinition(
            name="failure_rate_bps",
            category=MetricCategory.QUALITY,
            unit=MetricUnit.BASIS_POINTS,
            description="Failure rate in basis points",
        ),
        MetricDefinition(
            name="accepted_mutation_quality_bps",
            category=MetricCategory.QUALITY,
            unit=MetricUnit.BASIS_POINTS,
            description="Accepted mutation quality in basis points",
            higher_is_better=True,
        ),
    )


def default_llm_churn_metric_definitions() -> tuple[MetricDefinition, ...]:
    """LLM-churn metrics measured by LLMChurnBaseline@1."""

    return (
        MetricDefinition(
            name="context_bytes",
            category=MetricCategory.LLM_CHURN,
            unit=MetricUnit.BYTES,
            description="Context bytes sent or assembled for provider work",
        ),
        MetricDefinition(
            name="provider_calls",
            category=MetricCategory.LLM_CHURN,
            unit=MetricUnit.COUNT,
            description="Total charged provider calls including non-accepted outcomes",
        ),
        MetricDefinition(
            name="provider_input_tokens",
            category=MetricCategory.LLM_CHURN,
            unit=MetricUnit.TOKENS,
            description="Charged provider input tokens",
        ),
        MetricDefinition(
            name="provider_output_tokens",
            category=MetricCategory.LLM_CHURN,
            unit=MetricUnit.TOKENS,
            description="Charged provider output tokens",
        ),
        MetricDefinition(
            name="duplicate_semantic_inputs",
            category=MetricCategory.LLM_CHURN,
            unit=MetricUnit.COUNT,
            description="Provider inputs that were semantic duplicates of prior calls",
        ),
        MetricDefinition(
            name="cache_reuse_count",
            category=MetricCategory.LLM_CHURN,
            unit=MetricUnit.COUNT,
            description="Cache reuses that avoided a new provider call",
            higher_is_better=True,
        ),
        MetricDefinition(
            name="rejected_provider_usage",
            category=MetricCategory.PROVIDER_USAGE,
            unit=MetricUnit.COUNT,
            description="Provider usage charged under rejected disposition",
        ),
        MetricDefinition(
            name="retry_provider_usage",
            category=MetricCategory.PROVIDER_USAGE,
            unit=MetricUnit.COUNT,
            description="Provider usage charged under retry disposition",
        ),
        MetricDefinition(
            name="abandoned_provider_usage",
            category=MetricCategory.PROVIDER_USAGE,
            unit=MetricUnit.COUNT,
            description="Provider usage charged under abandoned disposition",
        ),
        MetricDefinition(
            name="accepted_provider_usage",
            category=MetricCategory.PROVIDER_USAGE,
            unit=MetricUnit.COUNT,
            description="Provider usage charged under accepted disposition",
        ),
        MetricDefinition(
            name="unchanged_reprompt_count",
            category=MetricCategory.LLM_CHURN,
            unit=MetricUnit.COUNT,
            description="Reprompts against unchanged semantic state",
        ),
    )


def default_metric_definitions() -> tuple[MetricDefinition, ...]:
    """Full metric catalog bound into a combined baseline."""

    return default_state_metric_definitions() + default_llm_churn_metric_definitions()


def metric_definitions_digest(
    definitions: Sequence[MetricDefinition] | None = None,
) -> str:
    """Content digest of the ordered metric definition catalog."""

    catalog = definitions if definitions is not None else default_metric_definitions()
    payload = [item.to_dict() for item in catalog]
    return content_identity(payload)


# ---------------------------------------------------------------------------
# Metric observation (measured vs unavailable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricObservation:
    """One metric observation: measured count or typed unavailable.

    Unavailable is never encoded as a measured numeric zero.  A measured zero
    is only valid when a sensor (including the hermetic workload) produced it.
    """

    SCHEMA: ClassVar[str] = METRIC_OBSERVATION_SCHEMA

    metric_name: str
    status: SampleStatus
    sensor_id: str
    stratum: BaselineStratum
    value: int = 0
    reason_code: str = ""
    sample_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metric_name", _text(self.metric_name, "metric_name", maximum=128)
        )
        object.__setattr__(
            self, "status", _parse_enum(self.status, SampleStatus, "status")
        )
        object.__setattr__(self, "sensor_id", _text(self.sensor_id, "sensor_id"))
        object.__setattr__(
            self, "stratum", _parse_enum(self.stratum, BaselineStratum, "stratum")
        )
        object.__setattr__(
            self, "value", _nonnegative_int(self.value, "value")
        )
        object.__setattr__(
            self,
            "sample_count",
            _nonnegative_int(self.sample_count, "sample_count", maximum=MAX_SAMPLES),
        )
        if self.sample_count < 1 and self.status is SampleStatus.MEASURED:
            raise DuckDBQuackBaselineError(
                "measured observation requires sample_count >= 1"
            )

        reason = self.reason_code
        if reason in (None, ""):
            object.__setattr__(self, "reason_code", "")
        else:
            object.__setattr__(
                self,
                "reason_code",
                _parse_enum(reason, UnavailableReason, "reason_code").value,
            )

        if self.status is SampleStatus.MEASURED:
            if self.reason_code:
                raise DuckDBQuackBaselineError(
                    "measured observation cannot carry an unavailable reason_code"
                )
        else:
            if not self.reason_code:
                raise DuckDBQuackBaselineError(
                    "unavailable observation requires a reason_code"
                )
            if self.value != 0:
                raise DuckDBQuackBaselineError(
                    "unavailable observation must not encode a numeric value"
                )

    @classmethod
    def measured(
        cls,
        metric_name: str,
        value: int,
        *,
        sensor_id: str,
        stratum: BaselineStratum | str = BaselineStratum.COLD,
        sample_count: int = 1,
    ) -> "MetricObservation":
        return cls(
            metric_name=metric_name,
            status=SampleStatus.MEASURED,
            sensor_id=sensor_id,
            stratum=stratum,
            value=_nonnegative_int(value, metric_name),
            sample_count=sample_count,
        )

    @classmethod
    def unavailable(
        cls,
        metric_name: str,
        reason: UnavailableReason | str,
        *,
        sensor_id: str = MISSING_TELEMETRY_SENSOR,
        stratum: BaselineStratum | str = BaselineStratum.COLD,
        sample_count: int = 0,
    ) -> "MetricObservation":
        return cls(
            metric_name=metric_name,
            status=SampleStatus.UNAVAILABLE,
            sensor_id=sensor_id,
            stratum=stratum,
            value=0,
            reason_code=_parse_enum(reason, UnavailableReason, "reason_code").value,
            sample_count=sample_count,
        )

    @property
    def is_measured(self) -> bool:
        return self.status is SampleStatus.MEASURED

    @property
    def is_unavailable(self) -> bool:
        return self.status is SampleStatus.UNAVAILABLE

    def measured_value(self) -> int | None:
        """Return the measured integer, or ``None`` when unavailable.

        Distinguishes missing (``None``) from measured zero (``0``).
        """

        if self.is_unavailable:
            return None
        return self.value

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.SCHEMA,
            "metric_name": self.metric_name,
            "status": self.status.value,
            "sensor_id": self.sensor_id,
            "stratum": self.stratum.value,
            "sample_count": self.sample_count,
        }
        if self.is_measured:
            payload["value"] = self.value
        else:
            payload["reason_code"] = self.reason_code
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetricObservation":
        if not isinstance(payload, Mapping):
            raise DuckDBQuackBaselineError("metric observation must be an object")
        claimed = payload.get("schema")
        if claimed is not None and claimed != cls.SCHEMA:
            raise DuckDBQuackBaselineError(
                f"metric observation has foreign schema {claimed!r}"
            )
        status = _parse_enum(
            payload.get("status", "measured"), SampleStatus, "status"
        )
        if status is SampleStatus.MEASURED:
            if "value" not in payload:
                raise DuckDBQuackBaselineError(
                    "measured observation requires value"
                )
            return cls.measured(
                str(payload.get("metric_name", "")),
                payload["value"],
                sensor_id=str(payload.get("sensor_id", "")),
                stratum=payload.get("stratum", BaselineStratum.COLD),
                sample_count=int(payload.get("sample_count", 1)),
            )
        return cls.unavailable(
            str(payload.get("metric_name", "")),
            payload.get("reason_code", UnavailableReason.TELEMETRY_MISSING),
            sensor_id=str(payload.get("sensor_id", MISSING_TELEMETRY_SENSOR)),
            stratum=payload.get("stratum", BaselineStratum.COLD),
            sample_count=int(payload.get("sample_count", 0)),
        )


# ---------------------------------------------------------------------------
# Provider usage charging
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderUsageCharge:
    """One charged provider-usage event under a closed disposition.

    Rejected, retry, and abandoned usage remains in the ledger so churn
    reductions cannot hide failed spend.
    """

    SCHEMA: ClassVar[str] = PROVIDER_USAGE_CHARGE_SCHEMA

    disposition: ProviderUsageDisposition
    call_count: int
    input_tokens: int = 0
    output_tokens: int = 0
    sensor_id: str = PROVIDER_LEDGER_SENSOR

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _parse_enum(self.disposition, ProviderUsageDisposition, "disposition"),
        )
        object.__setattr__(
            self, "call_count", _nonnegative_int(self.call_count, "call_count")
        )
        object.__setattr__(
            self, "input_tokens", _nonnegative_int(self.input_tokens, "input_tokens")
        )
        object.__setattr__(
            self, "output_tokens", _nonnegative_int(self.output_tokens, "output_tokens")
        )
        object.__setattr__(self, "sensor_id", _text(self.sensor_id, "sensor_id"))
        if not self.disposition.is_charged:
            raise DuckDBQuackBaselineError(
                f"disposition {self.disposition.value!r} is not chargeable"
            )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "disposition": self.disposition.value,
            "call_count": self.call_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "sensor_id": self.sensor_id,
            "charged": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderUsageCharge":
        if not isinstance(payload, Mapping):
            raise DuckDBQuackBaselineError("provider usage charge must be an object")
        claimed = payload.get("schema")
        if claimed is not None and claimed != cls.SCHEMA:
            raise DuckDBQuackBaselineError(
                f"provider usage charge has foreign schema {claimed!r}"
            )
        return cls(
            disposition=payload.get("disposition", ""),
            call_count=payload.get("call_count", 0),
            input_tokens=payload.get("input_tokens", 0),
            output_tokens=payload.get("output_tokens", 0),
            sensor_id=str(payload.get("sensor_id", PROVIDER_LEDGER_SENSOR)),
        )


@dataclass(frozen=True)
class ProviderUsageTotals:
    """Aggregated charged provider usage by disposition."""

    accepted_calls: int = 0
    rejected_calls: int = 0
    retry_calls: int = 0
    abandoned_calls: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    retry_tokens: int = 0
    abandoned_tokens: int = 0

    def __post_init__(self) -> None:
        for name in (
            "accepted_calls",
            "rejected_calls",
            "retry_calls",
            "abandoned_calls",
            "accepted_tokens",
            "rejected_tokens",
            "retry_tokens",
            "abandoned_tokens",
        ):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name)
            )

    @property
    def total_calls(self) -> int:
        return (
            self.accepted_calls
            + self.rejected_calls
            + self.retry_calls
            + self.abandoned_calls
        )

    @property
    def total_tokens(self) -> int:
        return (
            self.accepted_tokens
            + self.rejected_tokens
            + self.retry_tokens
            + self.abandoned_tokens
        )

    @property
    def non_accepted_calls(self) -> int:
        return self.rejected_calls + self.retry_calls + self.abandoned_calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_calls": self.accepted_calls,
            "rejected_calls": self.rejected_calls,
            "retry_calls": self.retry_calls,
            "abandoned_calls": self.abandoned_calls,
            "accepted_tokens": self.accepted_tokens,
            "rejected_tokens": self.rejected_tokens,
            "retry_tokens": self.retry_tokens,
            "abandoned_tokens": self.abandoned_tokens,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "non_accepted_calls": self.non_accepted_calls,
            "all_dispositions_charged": True,
        }


def charge_provider_usage(
    charges: Sequence[ProviderUsageCharge | Mapping[str, Any]],
) -> ProviderUsageTotals:
    """Aggregate provider usage charges.  Every disposition remains charged."""

    if not isinstance(charges, Sequence) or isinstance(charges, (str, bytes)):
        raise DuckDBQuackBaselineError("charges must be a sequence")
    if len(charges) > MAX_OBSERVATIONS:
        raise DuckDBQuackBaselineError("charges exceed observation bound")

    accepted_calls = rejected_calls = retry_calls = abandoned_calls = 0
    accepted_tokens = rejected_tokens = retry_tokens = abandoned_tokens = 0

    for item in charges:
        if isinstance(item, Mapping):
            charge = ProviderUsageCharge.from_dict(item)
        elif isinstance(item, ProviderUsageCharge):
            charge = item
        else:
            raise DuckDBQuackBaselineError(
                "charge entries must be ProviderUsageCharge or mapping"
            )
        tokens = charge.total_tokens
        if charge.disposition is ProviderUsageDisposition.ACCEPTED:
            accepted_calls += charge.call_count
            accepted_tokens += tokens
        elif charge.disposition is ProviderUsageDisposition.REJECTED:
            rejected_calls += charge.call_count
            rejected_tokens += tokens
        elif charge.disposition is ProviderUsageDisposition.RETRY:
            retry_calls += charge.call_count
            retry_tokens += tokens
        elif charge.disposition is ProviderUsageDisposition.ABANDONED:
            abandoned_calls += charge.call_count
            abandoned_tokens += tokens
        else:  # pragma: no cover - closed enum
            raise DuckDBQuackBaselineError(
                f"unsupported disposition {charge.disposition!r}"
            )

    return ProviderUsageTotals(
        accepted_calls=accepted_calls,
        rejected_calls=rejected_calls,
        retry_calls=retry_calls,
        abandoned_calls=abandoned_calls,
        accepted_tokens=accepted_tokens,
        rejected_tokens=rejected_tokens,
        retry_tokens=retry_tokens,
        abandoned_tokens=abandoned_tokens,
    )


# ---------------------------------------------------------------------------
# Locked criteria (cannot be weakened on regenerate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LockedCriteria:
    """Safety, durability, and quality floors sealed into a baseline.

    * Safety / durability floors are **maximums** (allowance cannot rise).
    * Quality floors are **minimums** (threshold cannot fall).

    Regeneration with any weakened criterion is rejected fail-closed.
    """

    SCHEMA: ClassVar[str] = LOCKED_CRITERIA_SCHEMA

    safety_floors: Mapping[str, int]
    durability_floors: Mapping[str, int]
    quality_floors: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "safety_floors",
            _normalize_floor_map(self.safety_floors, "safety_floors"),
        )
        object.__setattr__(
            self,
            "durability_floors",
            _normalize_floor_map(self.durability_floors, "durability_floors"),
        )
        object.__setattr__(
            self,
            "quality_floors",
            _normalize_floor_map(self.quality_floors, "quality_floors"),
        )

    @classmethod
    def defaults(cls) -> "LockedCriteria":
        return cls(
            safety_floors=dict(DEFAULT_SAFETY_FLOORS),
            durability_floors=dict(DEFAULT_DURABILITY_FLOORS),
            quality_floors=dict(DEFAULT_QUALITY_FLOORS),
        )

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "safety_floors": dict(sorted(self.safety_floors.items())),
            "durability_floors": dict(sorted(self.durability_floors.items())),
            "quality_floors": dict(sorted(self.quality_floors.items())),
            "floor_kinds": {
                "safety_floors": FloorKind.MAXIMUM.value,
                "durability_floors": FloorKind.MAXIMUM.value,
                "quality_floors": FloorKind.MINIMUM.value,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LockedCriteria":
        if not isinstance(payload, Mapping):
            raise DuckDBQuackBaselineError("locked criteria must be an object")
        claimed = payload.get("schema")
        if claimed is not None and claimed != cls.SCHEMA:
            raise DuckDBQuackBaselineError(
                f"locked criteria has foreign schema {claimed!r}"
            )
        return cls(
            safety_floors=payload.get("safety_floors") or {},
            durability_floors=payload.get("durability_floors") or {},
            quality_floors=payload.get("quality_floors") or {},
        )

    def weakening_violations(self, candidate: "LockedCriteria") -> tuple[str, ...]:
        """Return human-readable reasons that ``candidate`` weakens ``self``."""

        violations: list[str] = []
        for name, prior in self.safety_floors.items():
            next_value = int(candidate.safety_floors.get(name, prior))
            if next_value > prior:
                violations.append(
                    f"safety_floors.{name}: allowance raised {prior} -> {next_value}"
                )
        for name, prior in self.durability_floors.items():
            next_value = int(candidate.durability_floors.get(name, prior))
            if next_value > prior:
                violations.append(
                    f"durability_floors.{name}: allowance raised {prior} -> {next_value}"
                )
        for name, prior in self.quality_floors.items():
            next_value = int(candidate.quality_floors.get(name, prior))
            if next_value < prior:
                violations.append(
                    f"quality_floors.{name}: threshold lowered {prior} -> {next_value}"
                )
        return tuple(violations)

    def assert_not_weaker_than(self, prior: "LockedCriteria") -> None:
        """Reject when ``self`` weakens ``prior`` criteria."""

        violations = prior.weakening_violations(self)
        if violations:
            detail = "; ".join(violations)
            raise DuckDBQuackBaselineError(
                "cannot regenerate baseline with weakened safety, durability, "
                f"or quality criteria: {detail}"
            )


def _normalize_floor_map(
    value: Mapping[str, Any] | None,
    name: str,
) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DuckDBQuackBaselineError(f"{name} must be an object")
    result: dict[str, int] = {}
    for key, raw in value.items():
        key_text = _text(str(key), f"{name} key", maximum=128)
        result[key_text] = _nonnegative_int(raw, f"{name}.{key_text}")
    return dict(sorted(result.items()))


# ---------------------------------------------------------------------------
# Baseline binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineBinding:
    """Binds tree, environment, workload, and metric definitions together."""

    SCHEMA: ClassVar[str] = BASELINE_BINDING_SCHEMA

    tree_id: str
    environment_id: str
    workload_id: str
    metric_definitions_digest: str
    seed: str = "seed:default"
    policy_id: str = "policy:duckdb-quack-baseline"
    policy_revision: str = "policy:duckdb-quack-baseline@1"
    repository_id: str = "repository:ipfs-accelerate"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(
            self, "environment_id", _text(self.environment_id, "environment_id")
        )
        object.__setattr__(self, "workload_id", _text(self.workload_id, "workload_id"))
        object.__setattr__(
            self,
            "metric_definitions_digest",
            _text(self.metric_definitions_digest, "metric_definitions_digest"),
        )
        object.__setattr__(self, "seed", _text(self.seed, "seed"))
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(
            self, "policy_revision", _text(self.policy_revision, "policy_revision")
        )
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        if not self.metric_definitions_digest.startswith("sha256:"):
            raise DuckDBQuackBaselineError(
                "metric_definitions_digest must be a sha256 content identity"
            )

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "tree_id": self.tree_id,
            "environment_id": self.environment_id,
            "workload_id": self.workload_id,
            "metric_definitions_digest": self.metric_definitions_digest,
            "seed": self.seed,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "repository_id": self.repository_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BaselineBinding":
        if not isinstance(payload, Mapping):
            raise DuckDBQuackBaselineError("baseline binding must be an object")
        claimed = payload.get("schema")
        if claimed is not None and claimed != cls.SCHEMA:
            raise DuckDBQuackBaselineError(
                f"baseline binding has foreign schema {claimed!r}"
            )
        return cls(
            tree_id=payload.get("tree_id", ""),
            environment_id=payload.get("environment_id", ""),
            workload_id=payload.get("workload_id", ""),
            metric_definitions_digest=payload.get("metric_definitions_digest", ""),
            seed=payload.get("seed", "seed:default"),
            policy_id=payload.get("policy_id", "policy:duckdb-quack-baseline"),
            policy_revision=payload.get(
                "policy_revision", "policy:duckdb-quack-baseline@1"
            ),
            repository_id=payload.get("repository_id", "repository:ipfs-accelerate"),
        )


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def _index_observations(
    observations: Sequence[MetricObservation | Mapping[str, Any]],
) -> dict[tuple[str, str], MetricObservation]:
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise DuckDBQuackBaselineError("observations must be a sequence")
    if len(observations) > MAX_OBSERVATIONS:
        raise DuckDBQuackBaselineError("observations exceed bound")

    indexed: dict[tuple[str, str], MetricObservation] = {}
    for item in observations:
        if isinstance(item, Mapping):
            observation = MetricObservation.from_dict(item)
        elif isinstance(item, MetricObservation):
            observation = item
        else:
            raise DuckDBQuackBaselineError(
                "observations must be MetricObservation or mapping"
            )
        key = (observation.metric_name, observation.stratum.value)
        if key in indexed:
            raise DuckDBQuackBaselineError(
                f"duplicate observation for {observation.metric_name!r} "
                f"stratum {observation.stratum.value!r}"
            )
        indexed[key] = observation
    return indexed


def _resolve_required_observations(
    *,
    definitions: Sequence[MetricDefinition],
    observations: Sequence[MetricObservation | Mapping[str, Any]],
    strata: Sequence[BaselineStratum],
) -> tuple[MetricObservation, ...]:
    indexed = _index_observations(observations)
    resolved: list[MetricObservation] = []
    for definition in definitions:
        for stratum in strata:
            key = (definition.name, stratum.value)
            if key in indexed:
                resolved.append(indexed[key])
            elif definition.required:
                # Required but missing: typed unavailable, never invent zero.
                resolved.append(
                    MetricObservation.unavailable(
                        definition.name,
                        UnavailableReason.TELEMETRY_MISSING,
                        stratum=stratum,
                    )
                )
            else:
                resolved.append(
                    MetricObservation.unavailable(
                        definition.name,
                        UnavailableReason.STRATUM_NOT_RUN,
                        stratum=stratum,
                    )
                )
    return tuple(resolved)


def _observation_map(
    observations: Sequence[MetricObservation],
) -> dict[str, MetricObservation]:
    """Collapse observations preferring measured over unavailable per metric.

    When multiple strata exist, the first measured value wins for summary
    accessors; full multi-stratum detail remains on the observation list.
    """

    result: dict[str, MetricObservation] = {}
    for item in observations:
        existing = result.get(item.metric_name)
        if existing is None:
            result[item.metric_name] = item
            continue
        if existing.is_unavailable and item.is_measured:
            result[item.metric_name] = item
    return result


# ---------------------------------------------------------------------------
# SupervisorStateBaseline@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SupervisorStateBaseline:
    """State and latency baseline bound to tree/environment/workload/metrics."""

    INTERFACE: ClassVar[str] = SUPERVISOR_STATE_BASELINE_INTERFACE
    SCHEMA: ClassVar[str] = STATE_BASELINE_SCHEMA
    VERSION: ClassVar[int] = BASELINE_CONTRACT_VERSION

    binding: BaselineBinding
    metric_definitions: tuple[MetricDefinition, ...]
    observations: tuple[MetricObservation, ...]
    locked_criteria: LockedCriteria
    strata: tuple[BaselineStratum, ...] = (
        BaselineStratum.COLD,
        BaselineStratum.WARM,
        BaselineStratum.RESTART,
        BaselineStratum.PARALLEL,
    )
    sample_count: int = 1
    confidence_bps: int = 9_500
    task_id: str = BASELINE_TASK_ID
    goal_id: str = BASELINE_GOAL_ID

    def __post_init__(self) -> None:
        if not isinstance(self.binding, BaselineBinding):
            raise DuckDBQuackBaselineError("binding must be a BaselineBinding")
        if not self.metric_definitions:
            raise DuckDBQuackBaselineError("metric_definitions must be non-empty")
        defs = tuple(
            item if isinstance(item, MetricDefinition) else MetricDefinition.from_dict(item)
            for item in self.metric_definitions
        )
        object.__setattr__(self, "metric_definitions", defs)

        digest = metric_definitions_digest(defs)
        if self.binding.metric_definitions_digest != digest:
            raise DuckDBQuackBaselineError(
                "binding.metric_definitions_digest does not match metric_definitions"
            )

        strata = tuple(
            _parse_enum(item, BaselineStratum, "stratum") for item in self.strata
        )
        if not strata:
            raise DuckDBQuackBaselineError("strata must be non-empty")
        object.__setattr__(self, "strata", strata)

        observations = tuple(
            item if isinstance(item, MetricObservation) else MetricObservation.from_dict(item)
            for item in self.observations
        )
        object.__setattr__(self, "observations", observations)

        if not isinstance(self.locked_criteria, LockedCriteria):
            raise DuckDBQuackBaselineError("locked_criteria must be LockedCriteria")

        object.__setattr__(
            self,
            "sample_count",
            _nonnegative_int(self.sample_count, "sample_count", maximum=MAX_SAMPLES),
        )
        if self.sample_count < 1:
            raise DuckDBQuackBaselineError("sample_count must be >= 1")
        object.__setattr__(
            self,
            "confidence_bps",
            _nonnegative_int(self.confidence_bps, "confidence_bps", maximum=BASIS_POINTS),
        )
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "goal_id", _text(self.goal_id, "goal_id"))

    def observation(
        self,
        metric_name: str,
        *,
        stratum: BaselineStratum | str | None = None,
    ) -> MetricObservation | None:
        if stratum is None:
            return _observation_map(self.observations).get(metric_name)
        stratum_value = _parse_enum(stratum, BaselineStratum, "stratum").value
        for item in self.observations:
            if item.metric_name == metric_name and item.stratum.value == stratum_value:
                return item
        return None

    def measured_or_none(self, metric_name: str) -> int | None:
        """Return measured value or ``None`` when missing (never invent zero)."""

        item = self.observation(metric_name)
        if item is None:
            return None
        return item.measured_value()

    def unavailable_metrics(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    item.metric_name
                    for item in self.observations
                    if item.is_unavailable
                }
            )
        )

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "contract_version": self.VERSION,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "evidence": BASELINE_EVIDENCE,
            "binding": self.binding.to_dict(),
            "metric_definitions": [item.to_dict() for item in self.metric_definitions],
            "observations": [item.to_dict() for item in self.observations],
            "locked_criteria": self.locked_criteria.to_dict(),
            "strata": [item.value for item in self.strata],
            "sample_count": self.sample_count,
            "confidence_bps": self.confidence_bps,
            "unavailable_metrics": list(self.unavailable_metrics()),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SupervisorStateBaseline":
        if not isinstance(payload, Mapping):
            raise DuckDBQuackBaselineError("state baseline must be an object")
        claimed = payload.get("schema")
        if claimed is not None and claimed != cls.SCHEMA:
            raise DuckDBQuackBaselineError(
                f"state baseline has foreign schema {claimed!r}"
            )
        return cls(
            binding=BaselineBinding.from_dict(payload.get("binding") or {}),
            metric_definitions=tuple(
                MetricDefinition.from_dict(item)
                for item in (payload.get("metric_definitions") or ())
            ),
            observations=tuple(
                MetricObservation.from_dict(item)
                for item in (payload.get("observations") or ())
            ),
            locked_criteria=LockedCriteria.from_dict(
                payload.get("locked_criteria") or LockedCriteria.defaults().to_dict()
            ),
            strata=tuple(payload.get("strata") or ()),
            sample_count=int(payload.get("sample_count", 1)),
            confidence_bps=int(payload.get("confidence_bps", 9_500)),
            task_id=str(payload.get("task_id", BASELINE_TASK_ID)),
            goal_id=str(payload.get("goal_id", BASELINE_GOAL_ID)),
        )


# ---------------------------------------------------------------------------
# LLMChurnBaseline@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMChurnBaseline:
    """LLM-churn baseline including rejected/retry/abandoned provider usage."""

    INTERFACE: ClassVar[str] = LLM_CHURN_BASELINE_INTERFACE
    SCHEMA: ClassVar[str] = LLM_CHURN_BASELINE_SCHEMA
    VERSION: ClassVar[int] = BASELINE_CONTRACT_VERSION

    binding: BaselineBinding
    metric_definitions: tuple[MetricDefinition, ...]
    observations: tuple[MetricObservation, ...]
    provider_usage: ProviderUsageTotals
    locked_criteria: LockedCriteria
    strata: tuple[BaselineStratum, ...] = (
        BaselineStratum.COLD,
        BaselineStratum.WARM,
        BaselineStratum.RESTART,
        BaselineStratum.PARALLEL,
    )
    sample_count: int = 1
    confidence_bps: int = 9_500
    task_id: str = BASELINE_TASK_ID
    goal_id: str = BASELINE_GOAL_ID

    def __post_init__(self) -> None:
        if not isinstance(self.binding, BaselineBinding):
            raise DuckDBQuackBaselineError("binding must be a BaselineBinding")
        if not self.metric_definitions:
            raise DuckDBQuackBaselineError("metric_definitions must be non-empty")
        defs = tuple(
            item if isinstance(item, MetricDefinition) else MetricDefinition.from_dict(item)
            for item in self.metric_definitions
        )
        object.__setattr__(self, "metric_definitions", defs)

        digest = metric_definitions_digest(defs)
        if self.binding.metric_definitions_digest != digest:
            raise DuckDBQuackBaselineError(
                "binding.metric_definitions_digest does not match metric_definitions"
            )

        strata = tuple(
            _parse_enum(item, BaselineStratum, "stratum") for item in self.strata
        )
        if not strata:
            raise DuckDBQuackBaselineError("strata must be non-empty")
        object.__setattr__(self, "strata", strata)

        observations = tuple(
            item if isinstance(item, MetricObservation) else MetricObservation.from_dict(item)
            for item in self.observations
        )
        object.__setattr__(self, "observations", observations)

        if not isinstance(self.provider_usage, ProviderUsageTotals):
            raise DuckDBQuackBaselineError("provider_usage must be ProviderUsageTotals")
        if not isinstance(self.locked_criteria, LockedCriteria):
            raise DuckDBQuackBaselineError("locked_criteria must be LockedCriteria")

        object.__setattr__(
            self,
            "sample_count",
            _nonnegative_int(self.sample_count, "sample_count", maximum=MAX_SAMPLES),
        )
        if self.sample_count < 1:
            raise DuckDBQuackBaselineError("sample_count must be >= 1")
        object.__setattr__(
            self,
            "confidence_bps",
            _nonnegative_int(self.confidence_bps, "confidence_bps", maximum=BASIS_POINTS),
        )
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "goal_id", _text(self.goal_id, "goal_id"))

        # Provider usage disposition totals must agree with charged observations
        # when those metrics are measured.
        self._assert_provider_usage_consistency()

    def _assert_provider_usage_consistency(self) -> None:
        mapping = _observation_map(self.observations)
        checks = (
            ("rejected_provider_usage", self.provider_usage.rejected_calls),
            ("retry_provider_usage", self.provider_usage.retry_calls),
            ("abandoned_provider_usage", self.provider_usage.abandoned_calls),
            ("accepted_provider_usage", self.provider_usage.accepted_calls),
        )
        for metric_name, expected in checks:
            item = mapping.get(metric_name)
            if item is None or item.is_unavailable:
                continue
            if item.value != expected:
                raise DuckDBQuackBaselineError(
                    f"{metric_name} observation {item.value} does not match "
                    f"provider_usage total {expected}"
                )

        provider_calls = mapping.get("provider_calls")
        if (
            provider_calls is not None
            and provider_calls.is_measured
            and provider_calls.value != self.provider_usage.total_calls
        ):
            raise DuckDBQuackBaselineError(
                "provider_calls observation must equal charged total_calls "
                f"({provider_calls.value} != {self.provider_usage.total_calls})"
            )

    def observation(
        self,
        metric_name: str,
        *,
        stratum: BaselineStratum | str | None = None,
    ) -> MetricObservation | None:
        if stratum is None:
            return _observation_map(self.observations).get(metric_name)
        stratum_value = _parse_enum(stratum, BaselineStratum, "stratum").value
        for item in self.observations:
            if item.metric_name == metric_name and item.stratum.value == stratum_value:
                return item
        return None

    def measured_or_none(self, metric_name: str) -> int | None:
        item = self.observation(metric_name)
        if item is None:
            return None
        return item.measured_value()

    def unavailable_metrics(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    item.metric_name
                    for item in self.observations
                    if item.is_unavailable
                }
            )
        )

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "contract_version": self.VERSION,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "evidence": BASELINE_EVIDENCE,
            "binding": self.binding.to_dict(),
            "metric_definitions": [item.to_dict() for item in self.metric_definitions],
            "observations": [item.to_dict() for item in self.observations],
            "provider_usage": self.provider_usage.to_dict(),
            "locked_criteria": self.locked_criteria.to_dict(),
            "strata": [item.value for item in self.strata],
            "sample_count": self.sample_count,
            "confidence_bps": self.confidence_bps,
            "unavailable_metrics": list(self.unavailable_metrics()),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LLMChurnBaseline":
        if not isinstance(payload, Mapping):
            raise DuckDBQuackBaselineError("llm churn baseline must be an object")
        claimed = payload.get("schema")
        if claimed is not None and claimed != cls.SCHEMA:
            raise DuckDBQuackBaselineError(
                f"llm churn baseline has foreign schema {claimed!r}"
            )
        usage_payload = payload.get("provider_usage") or {}
        if not isinstance(usage_payload, Mapping):
            raise DuckDBQuackBaselineError("provider_usage must be an object")
        return cls(
            binding=BaselineBinding.from_dict(payload.get("binding") or {}),
            metric_definitions=tuple(
                MetricDefinition.from_dict(item)
                for item in (payload.get("metric_definitions") or ())
            ),
            observations=tuple(
                MetricObservation.from_dict(item)
                for item in (payload.get("observations") or ())
            ),
            provider_usage=ProviderUsageTotals(
                accepted_calls=int(usage_payload.get("accepted_calls", 0)),
                rejected_calls=int(usage_payload.get("rejected_calls", 0)),
                retry_calls=int(usage_payload.get("retry_calls", 0)),
                abandoned_calls=int(usage_payload.get("abandoned_calls", 0)),
                accepted_tokens=int(usage_payload.get("accepted_tokens", 0)),
                rejected_tokens=int(usage_payload.get("rejected_tokens", 0)),
                retry_tokens=int(usage_payload.get("retry_tokens", 0)),
                abandoned_tokens=int(usage_payload.get("abandoned_tokens", 0)),
            ),
            locked_criteria=LockedCriteria.from_dict(
                payload.get("locked_criteria") or LockedCriteria.defaults().to_dict()
            ),
            strata=tuple(payload.get("strata") or ()),
            sample_count=int(payload.get("sample_count", 1)),
            confidence_bps=int(payload.get("confidence_bps", 9_500)),
            task_id=str(payload.get("task_id", BASELINE_TASK_ID)),
            goal_id=str(payload.get("goal_id", BASELINE_GOAL_ID)),
        )


# ---------------------------------------------------------------------------
# Combined baseline report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DuckDBQuackBaseline:
    """Combined state + LLM-churn baseline used as the DQP-009 deliverable."""

    INTERFACE: ClassVar[str] = DUCKDB_QUACK_BASELINE_INTERFACE
    SCHEMA: ClassVar[str] = COMBINED_BASELINE_SCHEMA
    VERSION: ClassVar[int] = BASELINE_CONTRACT_VERSION

    binding: BaselineBinding
    state: SupervisorStateBaseline
    llm_churn: LLMChurnBaseline
    locked_criteria: LockedCriteria
    metric_definitions: tuple[MetricDefinition, ...]
    task_id: str = BASELINE_TASK_ID
    goal_id: str = BASELINE_GOAL_ID

    def __post_init__(self) -> None:
        if not isinstance(self.binding, BaselineBinding):
            raise DuckDBQuackBaselineError("binding must be a BaselineBinding")
        if not isinstance(self.state, SupervisorStateBaseline):
            raise DuckDBQuackBaselineError("state must be SupervisorStateBaseline")
        if not isinstance(self.llm_churn, LLMChurnBaseline):
            raise DuckDBQuackBaselineError("llm_churn must be LLMChurnBaseline")
        if not isinstance(self.locked_criteria, LockedCriteria):
            raise DuckDBQuackBaselineError("locked_criteria must be LockedCriteria")

        # Component baselines bind their own metric-definition subsets; the
        # combined record requires shared tree/environment/workload/seed identity.
        self._assert_binding_identity(self.state.binding, "state")
        self._assert_binding_identity(self.llm_churn.binding, "llm_churn")
        if self.state.locked_criteria.content_id != self.locked_criteria.content_id:
            raise DuckDBQuackBaselineError(
                "state locked_criteria must match combined locked_criteria"
            )
        if self.llm_churn.locked_criteria.content_id != self.locked_criteria.content_id:
            raise DuckDBQuackBaselineError(
                "llm_churn locked_criteria must match combined locked_criteria"
            )

        defs = tuple(
            item if isinstance(item, MetricDefinition) else MetricDefinition.from_dict(item)
            for item in self.metric_definitions
        )
        if not defs:
            raise DuckDBQuackBaselineError("metric_definitions must be non-empty")
        object.__setattr__(self, "metric_definitions", defs)
        digest = metric_definitions_digest(defs)
        if self.binding.metric_definitions_digest != digest:
            raise DuckDBQuackBaselineError(
                "binding.metric_definitions_digest does not match metric_definitions"
            )

        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "goal_id", _text(self.goal_id, "goal_id"))

    def _assert_binding_identity(self, other: BaselineBinding, name: str) -> None:
        for field_name in (
            "tree_id",
            "environment_id",
            "workload_id",
            "seed",
            "repository_id",
        ):
            if getattr(other, field_name) != getattr(self.binding, field_name):
                raise DuckDBQuackBaselineError(
                    f"{name} baseline binding.{field_name} must match combined binding"
                )
    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "contract_version": self.VERSION,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "evidence": BASELINE_EVIDENCE,
            "binding": self.binding.to_dict(),
            "metric_definitions": [item.to_dict() for item in self.metric_definitions],
            "locked_criteria": self.locked_criteria.to_dict(),
            "state": self.state.to_dict(),
            "llm_churn": self.llm_churn.to_dict(),
            "interfaces": [
                SUPERVISOR_STATE_BASELINE_INTERFACE,
                LLM_CHURN_BASELINE_INTERFACE,
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DuckDBQuackBaseline":
        if not isinstance(payload, Mapping):
            raise DuckDBQuackBaselineError("combined baseline must be an object")
        claimed = payload.get("schema")
        if claimed is not None and claimed != cls.SCHEMA:
            raise DuckDBQuackBaselineError(
                f"combined baseline has foreign schema {claimed!r}"
            )
        return cls(
            binding=BaselineBinding.from_dict(payload.get("binding") or {}),
            state=SupervisorStateBaseline.from_dict(payload.get("state") or {}),
            llm_churn=LLMChurnBaseline.from_dict(payload.get("llm_churn") or {}),
            locked_criteria=LockedCriteria.from_dict(
                payload.get("locked_criteria") or LockedCriteria.defaults().to_dict()
            ),
            metric_definitions=tuple(
                MetricDefinition.from_dict(item)
                for item in (payload.get("metric_definitions") or ())
            ),
            task_id=str(payload.get("task_id", BASELINE_TASK_ID)),
            goal_id=str(payload.get("goal_id", BASELINE_GOAL_ID)),
        )


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def establish_state_baseline(
    *,
    tree_id: str,
    environment_id: str,
    workload_id: str,
    observations: Sequence[MetricObservation | Mapping[str, Any]] | None = None,
    locked_criteria: LockedCriteria | Mapping[str, Any] | None = None,
    strata: Sequence[BaselineStratum | str] | None = None,
    seed: str = "seed:default",
    sample_count: int = 1,
    confidence_bps: int = 9_500,
    metric_definitions: Sequence[MetricDefinition] | None = None,
    repository_id: str = "repository:ipfs-accelerate",
) -> SupervisorStateBaseline:
    """Establish a SupervisorStateBaseline@1 bound to tree/env/workload/metrics."""

    definitions = tuple(metric_definitions or default_state_metric_definitions())
    resolved_strata = tuple(
        _parse_enum(item, BaselineStratum, "stratum")
        for item in (
            strata
            or (
                BaselineStratum.COLD,
                BaselineStratum.WARM,
                BaselineStratum.RESTART,
                BaselineStratum.PARALLEL,
            )
        )
    )
    resolved_observations = _resolve_required_observations(
        definitions=definitions,
        observations=observations or (),
        strata=resolved_strata,
    )
    criteria = (
        LockedCriteria.defaults()
        if locked_criteria is None
        else (
            locked_criteria
            if isinstance(locked_criteria, LockedCriteria)
            else LockedCriteria.from_dict(locked_criteria)
        )
    )
    binding = BaselineBinding(
        tree_id=tree_id,
        environment_id=environment_id,
        workload_id=workload_id,
        metric_definitions_digest=metric_definitions_digest(definitions),
        seed=seed,
        repository_id=repository_id,
    )
    return SupervisorStateBaseline(
        binding=binding,
        metric_definitions=definitions,
        observations=resolved_observations,
        locked_criteria=criteria,
        strata=resolved_strata,
        sample_count=sample_count,
        confidence_bps=confidence_bps,
    )


def establish_llm_churn_baseline(
    *,
    tree_id: str,
    environment_id: str,
    workload_id: str,
    provider_charges: Sequence[ProviderUsageCharge | Mapping[str, Any]] | None = None,
    observations: Sequence[MetricObservation | Mapping[str, Any]] | None = None,
    locked_criteria: LockedCriteria | Mapping[str, Any] | None = None,
    strata: Sequence[BaselineStratum | str] | None = None,
    seed: str = "seed:default",
    sample_count: int = 1,
    confidence_bps: int = 9_500,
    metric_definitions: Sequence[MetricDefinition] | None = None,
    repository_id: str = "repository:ipfs-accelerate",
) -> LLMChurnBaseline:
    """Establish an LLMChurnBaseline@1 with charged provider usage totals."""

    definitions = tuple(metric_definitions or default_llm_churn_metric_definitions())
    resolved_strata = tuple(
        _parse_enum(item, BaselineStratum, "stratum")
        for item in (
            strata
            or (
                BaselineStratum.COLD,
                BaselineStratum.WARM,
                BaselineStratum.RESTART,
                BaselineStratum.PARALLEL,
            )
        )
    )
    usage = charge_provider_usage(provider_charges or ())

    # Auto-fill charged disposition observations for the primary (cold) stratum
    # when callers supply charges but omit those metrics.
    primary = resolved_strata[0]
    supplied = list(observations or ())
    supplied_names = {
        (
            item.metric_name
            if isinstance(item, MetricObservation)
            else str((item or {}).get("metric_name", ""))  # type: ignore[union-attr]
        )
        for item in supplied
    }

    auto: list[MetricObservation] = []
    disposition_metrics = (
        ("accepted_provider_usage", usage.accepted_calls),
        ("rejected_provider_usage", usage.rejected_calls),
        ("retry_provider_usage", usage.retry_calls),
        ("abandoned_provider_usage", usage.abandoned_calls),
        ("provider_calls", usage.total_calls),
    )
    for metric_name, value in disposition_metrics:
        if metric_name not in supplied_names:
            auto.append(
                MetricObservation.measured(
                    metric_name,
                    value,
                    sensor_id=PROVIDER_LEDGER_SENSOR,
                    stratum=primary,
                    sample_count=sample_count,
                )
            )

    normalized_charges = [
        item
        if isinstance(item, ProviderUsageCharge)
        else ProviderUsageCharge.from_dict(item)
        for item in (provider_charges or ())
    ]
    input_total = sum(item.input_tokens for item in normalized_charges)
    output_total = sum(item.output_tokens for item in normalized_charges)
    if "provider_input_tokens" not in supplied_names:
        auto.append(
            MetricObservation.measured(
                "provider_input_tokens",
                input_total,
                sensor_id=PROVIDER_LEDGER_SENSOR,
                stratum=primary,
                sample_count=sample_count,
            )
        )
    if "provider_output_tokens" not in supplied_names:
        auto.append(
            MetricObservation.measured(
                "provider_output_tokens",
                output_total,
                sensor_id=PROVIDER_LEDGER_SENSOR,
                stratum=primary,
                sample_count=sample_count,
            )
        )
    resolved_observations = _resolve_required_observations(
        definitions=definitions,
        observations=list(supplied) + auto,
        strata=resolved_strata,
    )
    criteria = (
        LockedCriteria.defaults()
        if locked_criteria is None
        else (
            locked_criteria
            if isinstance(locked_criteria, LockedCriteria)
            else LockedCriteria.from_dict(locked_criteria)
        )
    )
    binding = BaselineBinding(
        tree_id=tree_id,
        environment_id=environment_id,
        workload_id=workload_id,
        metric_definitions_digest=metric_definitions_digest(definitions),
        seed=seed,
        repository_id=repository_id,
    )
    return LLMChurnBaseline(
        binding=binding,
        metric_definitions=definitions,
        observations=resolved_observations,
        provider_usage=usage,
        locked_criteria=criteria,
        strata=resolved_strata,
        sample_count=sample_count,
        confidence_bps=confidence_bps,
    )


def establish_baseline(
    *,
    tree_id: str,
    environment_id: str,
    workload_id: str,
    state_observations: Sequence[MetricObservation | Mapping[str, Any]] | None = None,
    churn_observations: Sequence[MetricObservation | Mapping[str, Any]] | None = None,
    provider_charges: Sequence[ProviderUsageCharge | Mapping[str, Any]] | None = None,
    locked_criteria: LockedCriteria | Mapping[str, Any] | None = None,
    strata: Sequence[BaselineStratum | str] | None = None,
    seed: str = "seed:default",
    sample_count: int = 1,
    confidence_bps: int = 9_500,
    repository_id: str = "repository:ipfs-accelerate",
) -> DuckDBQuackBaseline:
    """Establish the combined DQP-009 baseline for state, latency, and LLM churn."""

    criteria = (
        LockedCriteria.defaults()
        if locked_criteria is None
        else (
            locked_criteria
            if isinstance(locked_criteria, LockedCriteria)
            else LockedCriteria.from_dict(locked_criteria)
        )
    )
    common = dict(
        tree_id=tree_id,
        environment_id=environment_id,
        workload_id=workload_id,
        locked_criteria=criteria,
        strata=strata,
        seed=seed,
        sample_count=sample_count,
        confidence_bps=confidence_bps,
        repository_id=repository_id,
    )
    state = establish_state_baseline(observations=state_observations, **common)
    llm_churn = establish_llm_churn_baseline(
        provider_charges=provider_charges,
        observations=churn_observations,
        **common,
    )
    # Combined binding uses the full metric catalog digest.  Component baselines
    # keep subset digests for their own interfaces; identity fields align.
    all_definitions = default_metric_definitions()
    binding = BaselineBinding(
        tree_id=tree_id,
        environment_id=environment_id,
        workload_id=workload_id,
        metric_definitions_digest=metric_definitions_digest(all_definitions),
        seed=seed,
        repository_id=repository_id,
    )
    return DuckDBQuackBaseline(
        binding=binding,
        state=state,
        llm_churn=llm_churn,
        locked_criteria=criteria,
        metric_definitions=all_definitions,
    )

def regenerate_baseline(
    prior: DuckDBQuackBaseline,
    *,
    tree_id: str | None = None,
    environment_id: str | None = None,
    workload_id: str | None = None,
    state_observations: Sequence[MetricObservation | Mapping[str, Any]] | None = None,
    churn_observations: Sequence[MetricObservation | Mapping[str, Any]] | None = None,
    provider_charges: Sequence[ProviderUsageCharge | Mapping[str, Any]] | None = None,
    locked_criteria: LockedCriteria | Mapping[str, Any] | None = None,
    strata: Sequence[BaselineStratum | str] | None = None,
    seed: str | None = None,
    sample_count: int | None = None,
    confidence_bps: int | None = None,
    repository_id: str | None = None,
) -> DuckDBQuackBaseline:
    """Regenerate a baseline, refusing any weakened locked criteria.

    Measurement values may change; safety/durability allowances cannot rise and
    quality floors cannot fall relative to ``prior.locked_criteria``.
    """

    if not isinstance(prior, DuckDBQuackBaseline):
        raise DuckDBQuackBaselineError("prior must be a DuckDBQuackBaseline")

    if locked_criteria is None:
        candidate_criteria = prior.locked_criteria
    elif isinstance(locked_criteria, LockedCriteria):
        candidate_criteria = locked_criteria
    else:
        candidate_criteria = LockedCriteria.from_dict(locked_criteria)

    candidate_criteria.assert_not_weaker_than(prior.locked_criteria)

    return establish_baseline(
        tree_id=tree_id if tree_id is not None else prior.binding.tree_id,
        environment_id=(
            environment_id
            if environment_id is not None
            else prior.binding.environment_id
        ),
        workload_id=workload_id if workload_id is not None else prior.binding.workload_id,
        state_observations=state_observations,
        churn_observations=churn_observations,
        provider_charges=provider_charges,
        locked_criteria=candidate_criteria,
        strata=strata if strata is not None else prior.state.strata,
        seed=seed if seed is not None else prior.binding.seed,
        sample_count=(
            sample_count if sample_count is not None else prior.state.sample_count
        ),
        confidence_bps=(
            confidence_bps if confidence_bps is not None else prior.state.confidence_bps
        ),
        repository_id=(
            repository_id if repository_id is not None else prior.binding.repository_id
        ),
    )


def hermetic_fixture_baseline(
    *,
    tree_id: str = "tree:dqp-009-hermetic",
    environment_id: str = "environment:hermetic-validation",
    workload_id: str = "workload:dqp-009-fixed-hermetic@1",
    seed: str = "seed:dqp-009",
) -> DuckDBQuackBaseline:
    """Build a deterministic hermetic baseline used by unit tests and canaries.

    Values are fixed fixture measurements (including explicit zeros) so tests
    can distinguish measured zero from missing telemetry without live I/O.
    """

    cold = BaselineStratum.COLD
    sensor = HERMETIC_WORKLOAD_SENSOR

    state_observations = [
        MetricObservation.measured("file_reads", 42, sensor_id=sensor, stratum=cold),
        MetricObservation.measured("file_writes", 7, sensor_id=sensor, stratum=cold),
        MetricObservation.measured("file_parses", 42, sensor_id=sensor, stratum=cold),
        MetricObservation.measured(
            "independent_db_opens", 3, sensor_id=sensor, stratum=cold
        ),
        MetricObservation.measured("lock_wait_ms", 15, sensor_id=sensor, stratum=cold),
        MetricObservation.measured(
            "noop_poll_count", 0, sensor_id=sensor, stratum=cold
        ),  # measured zero
        MetricObservation.measured(
            "task_claim_latency_ms", 12, sensor_id=sensor, stratum=cold
        ),
        MetricObservation.measured(
            "queue_latency_ms", 8, sensor_id=sensor, stratum=cold
        ),
        MetricObservation.measured(
            "rollback_rate_bps", 0, sensor_id=sensor, stratum=cold
        ),
        MetricObservation.measured(
            "failure_rate_bps", 0, sensor_id=sensor, stratum=cold
        ),
        MetricObservation.measured(
            "accepted_mutation_quality_bps",
            9_200,
            sensor_id=sensor,
            stratum=cold,
        ),
    ]

    charges = [
        ProviderUsageCharge(
            disposition=ProviderUsageDisposition.ACCEPTED,
            call_count=2,
            input_tokens=1_200,
            output_tokens=400,
        ),
        ProviderUsageCharge(
            disposition=ProviderUsageDisposition.REJECTED,
            call_count=1,
            input_tokens=300,
            output_tokens=0,
        ),
        ProviderUsageCharge(
            disposition=ProviderUsageDisposition.RETRY,
            call_count=1,
            input_tokens=350,
            output_tokens=50,
        ),
        ProviderUsageCharge(
            disposition=ProviderUsageDisposition.ABANDONED,
            call_count=1,
            input_tokens=200,
            output_tokens=0,
        ),
    ]

    churn_observations = [
        MetricObservation.measured(
            "context_bytes", 24_000, sensor_id=sensor, stratum=cold
        ),
        MetricObservation.measured(
            "duplicate_semantic_inputs", 1, sensor_id=sensor, stratum=cold
        ),
        MetricObservation.measured(
            "cache_reuse_count", 0, sensor_id=sensor, stratum=cold
        ),  # measured zero
        MetricObservation.measured(
            "unchanged_reprompt_count", 1, sensor_id=sensor, stratum=cold
        ),
    ]

    return establish_baseline(
        tree_id=tree_id,
        environment_id=environment_id,
        workload_id=workload_id,
        state_observations=state_observations,
        churn_observations=churn_observations,
        provider_charges=charges,
        seed=seed,
        sample_count=3,
        confidence_bps=9_500,
        # Hermetic fixture focuses on cold stratum for compact determinism;
        # remaining strata are recorded as typed unavailable (not zero).
        strata=(BaselineStratum.COLD,),
    )


__all__ = [
    "BASELINE_CONTRACT_VERSION",
    "BASELINE_EVIDENCE",
    "BASELINE_GOAL_ID",
    "BASELINE_TASK_ID",
    "BaselineBinding",
    "BaselineStratum",
    "DEFAULT_DURABILITY_FLOORS",
    "DEFAULT_QUALITY_FLOORS",
    "DEFAULT_SAFETY_FLOORS",
    "DuckDBQuackBaseline",
    "DuckDBQuackBaselineError",
    "FloorKind",
    "HERMETIC_WORKLOAD_SENSOR",
    "LLM_CHURN_BASELINE_INTERFACE",
    "LLMChurnBaseline",
    "LockedCriteria",
    "MISSING_TELEMETRY_SENSOR",
    "MetricCategory",
    "MetricDefinition",
    "MetricObservation",
    "MetricUnit",
    "PROVIDER_LEDGER_SENSOR",
    "ProviderUsageCharge",
    "ProviderUsageDisposition",
    "ProviderUsageTotals",
    "SUPERVISOR_STATE_BASELINE_INTERFACE",
    "SampleStatus",
    "SupervisorStateBaseline",
    "UnavailableReason",
    "charge_provider_usage",
    "content_identity",
    "default_llm_churn_metric_definitions",
    "default_metric_definitions",
    "default_state_metric_definitions",
    "establish_baseline",
    "establish_llm_churn_baseline",
    "establish_state_baseline",
    "hermetic_fixture_baseline",
    "metric_definitions_digest",
    "regenerate_baseline",
]
