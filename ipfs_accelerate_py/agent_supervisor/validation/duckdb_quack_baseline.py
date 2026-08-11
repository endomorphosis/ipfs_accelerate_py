"""State, latency, and LLM-churn baselines for the DuckDB/Quack control plane.

Interfaces: ``SupervisorStateBaseline@1``, ``LLMChurnBaseline@1``

Task: DQP-009 / Goal: DQP-G050

This module is a measurement boundary only.  It never grants completion,
mutation, promotion, provider, or process authority.  Baselines bind tree,
environment, workload, and metric definitions; encode measured zero separately
from missing telemetry; count rejected / retry / abandoned provider usage; and
seal safety, durability, and quality criteria so a candidate cannot regenerate
a baseline by weakening those floors.

Hermetic workloads are deterministic synthetic fixtures.  Live production paths
are not optimized or mutated here.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

# ---------------------------------------------------------------------------
# Interface / schema identities
# ---------------------------------------------------------------------------

SUPERVISOR_STATE_BASELINE_INTERFACE: Final[str] = "SupervisorStateBaseline@1"
LLM_CHURN_BASELINE_INTERFACE: Final[str] = "LLMChurnBaseline@1"
BASELINE_CONTRACT_VERSION: Final[int] = 1
TASK_ID: Final[str] = "DQP-009"
GOAL_ID: Final[str] = "DQP-G050"
EVIDENCE: Final[str] = "dqp/duckdb-quack-baseline@1"

SCHEMA_PREFIX: Final[str] = "ipfs_accelerate_py/agent-supervisor"
SUPERVISOR_STATE_BASELINE_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/supervisor-state-baseline@1"
)
LLM_CHURN_BASELINE_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/llm-churn-baseline@1"
METRIC_DEFINITION_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/baseline-metric-definition@1"
METRIC_SAMPLE_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/baseline-metric-sample@1"
ENVIRONMENT_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/baseline-environment@1"
WORKLOAD_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/baseline-workload@1"
BINDING_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/baseline-binding@1"
CRITERIA_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/baseline-criteria@1"
PROVIDER_USAGE_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/baseline-provider-usage@1"
STRATUM_OBSERVATION_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/baseline-stratum-observation@1"
COMBINED_BASELINE_REPORT_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/duckdb-quack-baseline-report@1"
)

MAX_TEXT_BYTES: Final[int] = 512
MAX_COUNTER: Final[int] = 10**18
MAX_SAMPLES: Final[int] = 100_000
MAX_METRICS: Final[int] = 256
BASIS_POINTS: Final[int] = 10_000
MISSING_TELEMETRY_SENSOR: Final[str] = "sensor:missing-telemetry@1"
HERMETIC_SENSOR: Final[str] = "sensor:hermetic-workload@1"

# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class TelemetryStatus(str, Enum):
    """Whether a counter was measured or is typed unavailable.

    Measured zero is a valid observation.  Missing telemetry must never be
    encoded as a measured numeric zero.
    """

    MEASURED = "measured"
    UNAVAILABLE = "unavailable"


class UnavailableReason(str, Enum):
    """Closed reasons for missing telemetry."""

    TELEMETRY_MISSING = "telemetry-missing"
    SENSOR_ABSENT = "sensor-absent"
    COLLECTION_FAILED = "collection-failed"
    STRATUM_NOT_RUN = "stratum-not-run"
    PROVIDER_OMITTED = "provider-omitted"


class BaselineStratum(str, Enum):
    """Cold / warm / restart / parallel measurement strata."""

    COLD = "cold"
    WARM = "warm"
    RESTART = "restart"
    PARALLEL = "parallel"


class MetricUnit(str, Enum):
    COUNT = "count"
    BYTES = "bytes"
    MILLISECONDS = "milliseconds"
    BASIS_POINTS = "basis_points"
    RATIO = "ratio"


class MetricKind(str, Enum):
    """How a metric aggregates and whether zero is meaningful."""

    COUNTER = "counter"
    LATENCY = "latency"
    RATE = "rate"
    QUALITY = "quality"
    SIZE = "size"


class ProviderUsageOutcome(str, Enum):
    """Closed provider-usage disposition for churn accounting."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    RETRY = "retry"
    ABANDONED = "abandoned"


class BaselineVerdict(str, Enum):
    """Measurement conclusion; not a promotion decision."""

    ESTABLISHED = "established"
    REJECTED = "rejected"
    INSUFFICIENT = "insufficient"


_UNAVAILABLE_REASONS: Final[frozenset[str]] = frozenset(
    reason.value for reason in UnavailableReason
)
_REQUIRED_STRATA: Final[tuple[str, ...]] = tuple(s.value for s in BaselineStratum)
_PROVIDER_OUTCOMES: Final[tuple[str, ...]] = tuple(
    outcome.value for outcome in ProviderUsageOutcome
)

# Metric catalogs sealed into baseline identity.
STATE_METRIC_NAMES: Final[tuple[str, ...]] = (
    "file_reads",
    "file_writes",
    "file_parses",
    "independent_db_opens",
    "lock_waits_ms",
    "noop_polls",
    "task_claim_latency_ms",
    "queue_latency_ms",
    "rollback_count",
    "failure_count",
)

LLM_CHURN_METRIC_NAMES: Final[tuple[str, ...]] = (
    "context_bytes",
    "provider_calls",
    "input_tokens",
    "output_tokens",
    "duplicate_semantic_inputs",
    "cache_reuse_hits",
    "cache_reuse_misses",
    "accepted_mutations",
    "rejected_provider_calls",
    "retry_provider_calls",
    "abandoned_provider_calls",
    "accepted_mutation_quality_bps",
    "rollback_rate_bps",
    "failure_rate_bps",
)

# Absolute-zero safety floors (violations must remain zero).
SAFETY_FLOOR_KEYS: Final[tuple[str, ...]] = (
    "duplicate_non_idempotent_effects",
    "stale_lease_writes",
    "unauthorized_sql",
    "secret_leakage",
    "false_completion",
    "missing_impact_frontier_admission",
    "ast_mutation_misbinding",
    "event_projection_divergence",
    "accepted_state_loss",
    "safety_floor_violations",
)

# Durability floors (minimum required guarantees; cannot be lowered).
DURABILITY_FLOOR_KEYS: Final[tuple[str, ...]] = (
    "fsync_on_commit",
    "crash_recovery_verified",
    "lease_fence_enforced",
    "idempotency_keys_required",
    "rollback_receipt_required",
)

# Quality floors in basis points (cannot be lowered by candidate regeneration).
DEFAULT_MIN_ACCEPTED_MUTATION_QUALITY_BPS: Final[int] = 8_000
DEFAULT_MAX_ROLLBACK_RATE_BPS: Final[int] = 2_000
DEFAULT_MAX_FAILURE_RATE_BPS: Final[int] = 2_000
DEFAULT_MIN_SAMPLES: Final[int] = 4
DEFAULT_WORKLOAD_SEED: Final[int] = 0xD0_09_BA_5E
DEFAULT_SAMPLE_COUNT: Final[int] = 8


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DuckDBQuackBaselineError(ValueError):
    """Raised when baseline inputs are unsafe, incomplete, or weakened."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        raise DuckDBQuackBaselineError(f"{name} must be text")
    result = value.strip()
    if not result:
        raise DuckDBQuackBaselineError(f"{name} must not be empty")
    if "\x00" in result:
        raise DuckDBQuackBaselineError(f"{name} contains a NUL byte")
    if len(result.encode("utf-8")) > maximum:
        raise DuckDBQuackBaselineError(
            f"{name} exceeds its {maximum}-byte bound"
        )
    return result


def _nonnegative_int(
    value: Any,
    name: str,
    *,
    maximum: int = MAX_COUNTER,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DuckDBQuackBaselineError(
            f"{name} must be a non-negative integer"
        )
    if value < 0:
        raise DuckDBQuackBaselineError(
            f"{name} must be a non-negative integer"
        )
    if value > maximum:
        raise DuckDBQuackBaselineError(f"{name} exceeds its maximum of {maximum}")
    return value


def _positive_int(
    value: Any,
    name: str,
    *,
    maximum: int = MAX_COUNTER,
) -> int:
    result = _nonnegative_int(value, name, maximum=maximum)
    if result < 1:
        raise DuckDBQuackBaselineError(f"{name} must be a positive integer")
    return result


def _bps(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name, maximum=BASIS_POINTS)
    return result


def _parse_enum(enum_cls: type[Enum], value: Any, name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, Enum):
        value = value.value
    text = _text(value, name)
    try:
        return enum_cls(text)
    except ValueError as exc:
        allowed = ", ".join(sorted(item.value for item in enum_cls))
        raise DuckDBQuackBaselineError(
            f"{name} must be one of {{{allowed}}}; got {text!r}"
        ) from exc


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _plain(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DuckDBQuackBaselineError(
            "baseline payloads require canonical JSON values"
        ) from exc


def content_identity(value: Any) -> str:
    """Return a stable sha256 content identity for a JSON-compatible value."""

    digest = hashlib.sha256(_canonical_bytes(value)).hexdigest()
    return f"sha256:{digest}"


def _mapping_proxy(data: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(data))


# ---------------------------------------------------------------------------
# Metric definition and sample (missing vs zero)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricDefinition:
    """Sealed definition of one baseline metric.

    Zero is a valid measured value only when a sensor produced it.  Absence of
    a sensor is encoded as ``unavailable``, never as measured zero.
    """

    SCHEMA: ClassVar[str] = METRIC_DEFINITION_SCHEMA

    name: str
    unit: MetricUnit
    kind: MetricKind
    description: str
    zero_is_meaningful: bool = True
    higher_is_better: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "name", maximum=128))
        object.__setattr__(
            self, "unit", _parse_enum(MetricUnit, self.unit, "unit")
        )
        object.__setattr__(
            self, "kind", _parse_enum(MetricKind, self.kind, "kind")
        )
        object.__setattr__(
            self,
            "description",
            _text(self.description, "description", maximum=MAX_TEXT_BYTES),
        )
        if not isinstance(self.zero_is_meaningful, bool):
            raise DuckDBQuackBaselineError("zero_is_meaningful must be bool")
        if not isinstance(self.higher_is_better, bool):
            raise DuckDBQuackBaselineError("higher_is_better must be bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "name": self.name,
            "unit": self.unit.value if isinstance(self.unit, Enum) else self.unit,
            "kind": self.kind.value if isinstance(self.kind, Enum) else self.kind,
            "description": self.description,
            "zero_is_meaningful": self.zero_is_meaningful,
            "higher_is_better": self.higher_is_better,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetricDefinition":
        if not isinstance(payload, Mapping):
            raise DuckDBQuackBaselineError("metric definition must be an object")
        return cls(
            name=str(payload.get("name") or ""),
            unit=payload.get("unit", MetricUnit.COUNT),
            kind=payload.get("kind", MetricKind.COUNTER),
            description=str(payload.get("description") or "metric"),
            zero_is_meaningful=bool(payload.get("zero_is_meaningful", True)),
            higher_is_better=bool(payload.get("higher_is_better", False)),
        )


@dataclass(frozen=True)
class MetricSample:
    """One metric observation: measured count (including zero) or unavailable."""

    SCHEMA: ClassVar[str] = METRIC_SAMPLE_SCHEMA

    metric_name: str
    status: TelemetryStatus
    sensor_id: str
    value: int = 0
    reason_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metric_name", _text(self.metric_name, "metric_name", maximum=128)
        )
        object.__setattr__(
            self, "status", _parse_enum(TelemetryStatus, self.status, "status")
        )
        object.__setattr__(
            self, "sensor_id", _text(self.sensor_id, "sensor_id", maximum=192)
        )
        object.__setattr__(self, "value", _nonnegative_int(self.value, "value"))
        if self.reason_code in (None, ""):
            object.__setattr__(self, "reason_code", "")
        else:
            reason = _text(self.reason_code, "reason_code", maximum=64)
            if reason not in _UNAVAILABLE_REASONS:
                allowed = ", ".join(sorted(_UNAVAILABLE_REASONS))
                raise DuckDBQuackBaselineError(
                    f"reason_code must be one of {{{allowed}}}; got {reason!r}"
                )
            object.__setattr__(self, "reason_code", reason)

        if self.status is TelemetryStatus.MEASURED:
            if self.reason_code:
                raise DuckDBQuackBaselineError(
                    "measured sample cannot carry an unavailable reason_code"
                )
        else:
            if not self.reason_code:
                raise DuckDBQuackBaselineError(
                    "unavailable sample requires a reason_code"
                )
            if self.value != 0:
                raise DuckDBQuackBaselineError(
                    "unavailable sample must not encode a numeric value"
                )

    @classmethod
    def measured(
        cls,
        metric_name: str,
        value: int,
        *,
        sensor_id: str = HERMETIC_SENSOR,
    ) -> "MetricSample":
        return cls(
            metric_name=metric_name,
            status=TelemetryStatus.MEASURED,
            sensor_id=sensor_id,
            value=_nonnegative_int(value, metric_name),
        )

    @classmethod
    def unavailable(
        cls,
        metric_name: str,
        reason: UnavailableReason | str,
        *,
        sensor_id: str = MISSING_TELEMETRY_SENSOR,
    ) -> "MetricSample":
        if isinstance(reason, UnavailableReason):
            reason_code = reason.value
        else:
            reason_code = _text(reason, "reason_code", maximum=64)
        return cls(
            metric_name=metric_name,
            status=TelemetryStatus.UNAVAILABLE,
            sensor_id=sensor_id,
            value=0,
            reason_code=reason_code,
        )

    @property
    def is_measured(self) -> bool:
        return self.status is TelemetryStatus.MEASURED

    @property
    def is_unavailable(self) -> bool:
        return self.status is TelemetryStatus.UNAVAILABLE

    def measured_value(self) -> int | None:
        """Return the measured integer, or ``None`` when unavailable."""

        if self.is_unavailable:
            return None
        return self.value

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.SCHEMA,
            "metric_name": self.metric_name,
            "status": self.status.value
            if isinstance(self.status, Enum)
            else self.status,
            "sensor_id": self.sensor_id,
        }
        if self.is_measured:
            payload["value"] = self.value
        else:
            payload["reason_code"] = self.reason_code
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetricSample":
        if not isinstance(payload, Mapping):
            raise DuckDBQuackBaselineError("metric sample must be an object")
        status = _parse_enum(
            TelemetryStatus, payload.get("status", "measured"), "status"
        )
        if status is TelemetryStatus.MEASURED:
            if "value" not in payload:
                raise DuckDBQuackBaselineError(
                    "measured metric sample requires value"
                )
            return cls.measured(
                str(payload.get("metric_name") or ""),
                payload["value"],
                sensor_id=str(payload.get("sensor_id") or HERMETIC_SENSOR),
            )
        return cls.unavailable(
            str(payload.get("metric_name") or ""),
            payload.get("reason_code", UnavailableReason.TELEMETRY_MISSING),
            sensor_id=str(payload.get("sensor_id") or MISSING_TELEMETRY_SENSOR),
        )


# ---------------------------------------------------------------------------
# Environment, workload, criteria, binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineEnvironment:
    """Pinned measurement environment binding."""

    SCHEMA: ClassVar[str] = ENVIRONMENT_SCHEMA

    python_version: str
    platform_name: str
    implementation: str
    path_fingerprint: str
    duckdb_version: str = "unavailable"
    extra: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "python_version", _text(self.python_version, "python_version")
        )
        object.__setattr__(
            self, "platform_name", _text(self.platform_name, "platform_name")
        )
        object.__setattr__(
            self, "implementation", _text(self.implementation, "implementation")
        )
        object.__setattr__(
            self,
            "path_fingerprint",
            _text(self.path_fingerprint, "path_fingerprint", maximum=96),
        )
        object.__setattr__(
            self,
            "duckdb_version",
            _text(self.duckdb_version, "duckdb_version", maximum=96),
        )
        cleaned: dict[str, str] = {}
        for key, value in dict(self.extra or {}).items():
            cleaned[_text(key, "extra.key", maximum=64)] = _text(
                value, f"extra.{key}", maximum=192
            )
        object.__setattr__(self, "extra", _mapping_proxy(cleaned))

    @property
    def identity_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "python_version": self.python_version,
            "platform_name": self.platform_name,
            "implementation": self.implementation,
            "path_fingerprint": self.path_fingerprint,
            "duckdb_version": self.duckdb_version,
            "extra": dict(self.extra),
        }
        if include_identity:
            payload["identity_id"] = self.identity_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BaselineEnvironment":
        if not isinstance(payload, Mapping):
            raise DuckDBQuackBaselineError("environment must be an object")
        return cls(
            python_version=str(payload.get("python_version") or ""),
            platform_name=str(payload.get("platform_name") or ""),
            implementation=str(payload.get("implementation") or ""),
            path_fingerprint=str(payload.get("path_fingerprint") or ""),
            duckdb_version=str(payload.get("duckdb_version") or "unavailable"),
            extra=dict(payload.get("extra") or {}),
        )

    @classmethod
    def capture(cls, *, path_entries: Sequence[str] | None = None) -> "BaselineEnvironment":
        """Capture the current process environment without side effects."""

        path = list(path_entries) if path_entries is not None else list(
            (sys.path or [])[:16]
        )
        path_fp = content_identity(path)
        duckdb_version = "unavailable"
        try:
            import duckdb  # type: ignore

            duckdb_version = str(getattr(duckdb, "__version__", "unknown"))
        except Exception:
            duckdb_version = "unavailable"
        return cls(
            python_version=".".join(str(part) for part in sys.version_info[:3]),
            platform_name=platform.platform(),
            implementation=platform.python_implementation(),
            path_fingerprint=path_fp,
            duckdb_version=duckdb_version,
            extra={
                "machine": platform.machine() or "unknown",
                "system": platform.system() or "unknown",
            },
        )


@dataclass(frozen=True)
class WorkloadDefinition:
    """Fixed hermetic workload bound into the baseline identity."""

    SCHEMA: ClassVar[str] = WORKLOAD_SCHEMA

    workload_id: str
    seed: int
    sample_count: int
    strata: tuple[str, ...]
    operations: tuple[str, ...]
    description: str = "hermetic duckdb/quack baseline workload"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "workload_id", _text(self.workload_id, "workload_id", maximum=128)
        )
        object.__setattr__(self, "seed", _nonnegative_int(self.seed, "seed"))
        object.__setattr__(
            self,
            "sample_count",
            _positive_int(self.sample_count, "sample_count", maximum=MAX_SAMPLES),
        )
        strata = tuple(
            _text(item, "strata.item", maximum=32).lower() for item in self.strata
        )
        if not strata:
            raise DuckDBQuackBaselineError("workload strata must be non-empty")
        unknown = [item for item in strata if item not in _REQUIRED_STRATA]
        if unknown:
            raise DuckDBQuackBaselineError(
                f"unknown strata {unknown!r}; allowed {_REQUIRED_STRATA}"
            )
        missing = [item for item in _REQUIRED_STRATA if item not in strata]
        if missing:
            raise DuckDBQuackBaselineError(
                f"workload must cover required strata; missing {missing}"
            )
        object.__setattr__(self, "strata", strata)
        ops = tuple(
            _text(item, "operations.item", maximum=64) for item in self.operations
        )
        if not ops:
            raise DuckDBQuackBaselineError("workload operations must be non-empty")
        if len(ops) > MAX_METRICS:
            raise DuckDBQuackBaselineError("workload operations exceed bound")
        object.__setattr__(self, "operations", ops)
        object.__setattr__(
            self,
            "description",
            _text(self.description, "description"),
        )

    @property
    def identity_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "workload_id": self.workload_id,
            "seed": self.seed,
            "sample_count": self.sample_count,
            "strata": list(self.strata),
            "operations": list(self.operations),
            "description": self.description,
        }
        if include_identity:
            payload["identity_id"] = self.identity_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkloadDefinition":
        if not isinstance(payload, Mapping):
            raise DuckDBQuackBaselineError("workload must be an object")
        return cls(
            workload_id=str(payload.get("workload_id") or ""),
            seed=int(payload.get("seed", 0)),
            sample_count=int(payload.get("sample_count", DEFAULT_SAMPLE_COUNT)),
            strata=tuple(payload.get("strata") or _REQUIRED_STRATA),
            operations=tuple(payload.get("operations") or ()),
            description=str(
                payload.get("description")
                or "hermetic duckdb/quack baseline workload"
            ),
        )


@dataclass(frozen=True)
class BaselineCriteria:
    """Sealed safety, durability, and quality floors.

    Candidate regeneration may not weaken any floor.  Weakening means:

    * raising any safety floor above zero (allowing violations);
    * setting any durability guarantee to false;
    * lowering minimum accepted-mutation quality;
    * raising maximum rollback or failure rates;
    * lowering the minimum sample requirement.
    """

    SCHEMA: ClassVar[str] = CRITERIA_SCHEMA

    safety_floors: Mapping[str, int] = field(default_factory=dict)
    durability_floors: Mapping[str, bool] = field(default_factory=dict)
    min_accepted_mutation_quality_bps: int = DEFAULT_MIN_ACCEPTED_MUTATION_QUALITY_BPS
    max_rollback_rate_bps: int = DEFAULT_MAX_ROLLBACK_RATE_BPS
    max_failure_rate_bps: int = DEFAULT_MAX_FAILURE_RATE_BPS
    min_samples: int = DEFAULT_MIN_SAMPLES
    require_zero_safety_floors: bool = True
    completion_authoritative: bool = False
    mutation_authorized: bool = False

    def __post_init__(self) -> None:
        # Construction admits candidate payloads so weakening can be detected
        # and rejected at regeneration/establishment time.  Unknown keys still
        # fail closed.
        floors = {
            key: _nonnegative_int(
                dict(self.safety_floors or {}).get(key, 0), f"safety_floors.{key}"
            )
            for key in SAFETY_FLOOR_KEYS
        }
        for key in dict(self.safety_floors or {}):
            if key not in SAFETY_FLOOR_KEYS:
                raise DuckDBQuackBaselineError(
                    f"unknown safety floor key {key!r}"
                )
        object.__setattr__(self, "safety_floors", _mapping_proxy(floors))

        durability = {
            key: bool(dict(self.durability_floors or {}).get(key, True))
            for key in DURABILITY_FLOOR_KEYS
        }
        for key in dict(self.durability_floors or {}):
            if key not in DURABILITY_FLOOR_KEYS:
                raise DuckDBQuackBaselineError(
                    f"unknown durability floor key {key!r}"
                )
        object.__setattr__(self, "durability_floors", _mapping_proxy(durability))

        object.__setattr__(
            self,
            "min_accepted_mutation_quality_bps",
            _bps(self.min_accepted_mutation_quality_bps, "min_accepted_mutation_quality_bps"),
        )
        object.__setattr__(
            self,
            "max_rollback_rate_bps",
            _bps(self.max_rollback_rate_bps, "max_rollback_rate_bps"),
        )
        object.__setattr__(
            self,
            "max_failure_rate_bps",
            _bps(self.max_failure_rate_bps, "max_failure_rate_bps"),
        )
        object.__setattr__(
            self,
            "min_samples",
            _positive_int(self.min_samples, "min_samples", maximum=MAX_SAMPLES),
        )
        if not isinstance(self.require_zero_safety_floors, bool):
            raise DuckDBQuackBaselineError("require_zero_safety_floors must be bool")
        if not isinstance(self.completion_authoritative, bool):
            raise DuckDBQuackBaselineError("completion_authoritative must be bool")
        if not isinstance(self.mutation_authorized, bool):
            raise DuckDBQuackBaselineError("mutation_authorized must be bool")

    def assert_establishment_safe(self) -> None:
        """Reject criteria that cannot establish an authoritative sealed baseline."""

        if self.completion_authoritative or self.mutation_authorized:
            raise DuckDBQuackBaselineError(
                "baseline criteria must not authorize completion or mutation"
            )
        if self.require_zero_safety_floors and any(
            value != 0 for value in self.safety_floors.values()
        ):
            raise DuckDBQuackBaselineError(
                "safety floors must be absolute zero when require_zero_safety_floors"
            )
        if not all(self.durability_floors.values()):
            raise DuckDBQuackBaselineError(
                "durability floors cannot be disabled when establishing a baseline"
            )
        sealed = BaselineCriteria.sealed_defaults()
        weakened, reasons = self.is_weakened_relative_to(sealed)
        if weakened:
            raise DuckDBQuackBaselineError(
                "criteria weaken sealed defaults: " + ", ".join(reasons)
            )

    @property
    def identity_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "safety_floors": dict(self.safety_floors),
            "durability_floors": dict(self.durability_floors),
            "min_accepted_mutation_quality_bps": self.min_accepted_mutation_quality_bps,
            "max_rollback_rate_bps": self.max_rollback_rate_bps,
            "max_failure_rate_bps": self.max_failure_rate_bps,
            "min_samples": self.min_samples,
            "require_zero_safety_floors": self.require_zero_safety_floors,
            "completion_authoritative": self.completion_authoritative,
            "mutation_authorized": self.mutation_authorized,
        }
        if include_identity:
            payload["identity_id"] = self.identity_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BaselineCriteria":
        if not isinstance(payload, Mapping):
            raise DuckDBQuackBaselineError("criteria must be an object")
        return cls(
            safety_floors=dict(payload.get("safety_floors") or {}),
            durability_floors=dict(payload.get("durability_floors") or {}),
            min_accepted_mutation_quality_bps=int(
                payload.get(
                    "min_accepted_mutation_quality_bps",
                    DEFAULT_MIN_ACCEPTED_MUTATION_QUALITY_BPS,
                )
            ),
            max_rollback_rate_bps=int(
                payload.get("max_rollback_rate_bps", DEFAULT_MAX_ROLLBACK_RATE_BPS)
            ),
            max_failure_rate_bps=int(
                payload.get("max_failure_rate_bps", DEFAULT_MAX_FAILURE_RATE_BPS)
            ),
            min_samples=int(payload.get("min_samples", DEFAULT_MIN_SAMPLES)),
            require_zero_safety_floors=bool(
                payload.get("require_zero_safety_floors", True)
            ),
            completion_authoritative=bool(
                payload.get("completion_authoritative", False)
            ),
            mutation_authorized=bool(payload.get("mutation_authorized", False)),
        )

    @classmethod
    def sealed_defaults(cls) -> "BaselineCriteria":
        return cls(
            safety_floors={key: 0 for key in SAFETY_FLOOR_KEYS},
            durability_floors={key: True for key in DURABILITY_FLOOR_KEYS},
        )

    def is_weakened_relative_to(self, sealed: "BaselineCriteria") -> tuple[bool, tuple[str, ...]]:
        """Return whether ``self`` weakens ``sealed`` and the reason codes."""

        reasons: list[str] = []
        for key in SAFETY_FLOOR_KEYS:
            candidate = int(self.safety_floors.get(key, 0))
            baseline = int(sealed.safety_floors.get(key, 0))
            if candidate > baseline:
                reasons.append(f"safety_floor_raised:{key}")
        if (
            sealed.require_zero_safety_floors
            and not self.require_zero_safety_floors
        ):
            reasons.append("require_zero_safety_floors_disabled")
        for key in DURABILITY_FLOOR_KEYS:
            if sealed.durability_floors.get(key, True) and not self.durability_floors.get(
                key, False
            ):
                reasons.append(f"durability_disabled:{key}")
        if (
            self.min_accepted_mutation_quality_bps
            < sealed.min_accepted_mutation_quality_bps
        ):
            reasons.append("quality_floor_lowered")
        if self.max_rollback_rate_bps > sealed.max_rollback_rate_bps:
            reasons.append("rollback_rate_ceiling_raised")
        if self.max_failure_rate_bps > sealed.max_failure_rate_bps:
            reasons.append("failure_rate_ceiling_raised")
        if self.min_samples < sealed.min_samples:
            reasons.append("min_samples_lowered")
        if self.completion_authoritative and not sealed.completion_authoritative:
            reasons.append("completion_authority_granted")
        if self.mutation_authorized and not sealed.mutation_authorized:
            reasons.append("mutation_authority_granted")
        return (bool(reasons), tuple(reasons))


def default_state_metric_definitions() -> tuple[MetricDefinition, ...]:
    return (
        MetricDefinition(
            "file_reads", MetricUnit.COUNT, MetricKind.COUNTER, "File read operations"
        ),
        MetricDefinition(
            "file_writes", MetricUnit.COUNT, MetricKind.COUNTER, "File write operations"
        ),
        MetricDefinition(
            "file_parses", MetricUnit.COUNT, MetricKind.COUNTER, "File parse operations"
        ),
        MetricDefinition(
            "independent_db_opens",
            MetricUnit.COUNT,
            MetricKind.COUNTER,
            "Independent database open events",
        ),
        MetricDefinition(
            "lock_waits_ms",
            MetricUnit.MILLISECONDS,
            MetricKind.LATENCY,
            "Aggregate lock wait time",
            higher_is_better=False,
        ),
        MetricDefinition(
            "noop_polls",
            MetricUnit.COUNT,
            MetricKind.COUNTER,
            "No-op polling cycles with no ready work",
        ),
        MetricDefinition(
            "task_claim_latency_ms",
            MetricUnit.MILLISECONDS,
            MetricKind.LATENCY,
            "Task claim latency",
            higher_is_better=False,
        ),
        MetricDefinition(
            "queue_latency_ms",
            MetricUnit.MILLISECONDS,
            MetricKind.LATENCY,
            "Queue wait latency",
            higher_is_better=False,
        ),
        MetricDefinition(
            "rollback_count",
            MetricUnit.COUNT,
            MetricKind.COUNTER,
            "Rollback events",
            zero_is_meaningful=True,
        ),
        MetricDefinition(
            "failure_count",
            MetricUnit.COUNT,
            MetricKind.COUNTER,
            "Hard failure events",
            zero_is_meaningful=True,
        ),
    )


def default_llm_metric_definitions() -> tuple[MetricDefinition, ...]:
    return (
        MetricDefinition(
            "context_bytes", MetricUnit.BYTES, MetricKind.SIZE, "Context packet bytes"
        ),
        MetricDefinition(
            "provider_calls",
            MetricUnit.COUNT,
            MetricKind.COUNTER,
            "Provider call attempts (all outcomes)",
        ),
        MetricDefinition(
            "input_tokens", MetricUnit.COUNT, MetricKind.COUNTER, "Input tokens"
        ),
        MetricDefinition(
            "output_tokens", MetricUnit.COUNT, MetricKind.COUNTER, "Output tokens"
        ),
        MetricDefinition(
            "duplicate_semantic_inputs",
            MetricUnit.COUNT,
            MetricKind.COUNTER,
            "Duplicate semantic provider inputs",
            zero_is_meaningful=True,
        ),
        MetricDefinition(
            "cache_reuse_hits",
            MetricUnit.COUNT,
            MetricKind.COUNTER,
            "Decision/context cache hits",
            higher_is_better=True,
        ),
        MetricDefinition(
            "cache_reuse_misses",
            MetricUnit.COUNT,
            MetricKind.COUNTER,
            "Decision/context cache misses",
        ),
        MetricDefinition(
            "accepted_mutations",
            MetricUnit.COUNT,
            MetricKind.COUNTER,
            "Accepted mutations",
            higher_is_better=True,
        ),
        MetricDefinition(
            "rejected_provider_calls",
            MetricUnit.COUNT,
            MetricKind.COUNTER,
            "Provider calls rejected by policy/gate",
            zero_is_meaningful=True,
        ),
        MetricDefinition(
            "retry_provider_calls",
            MetricUnit.COUNT,
            MetricKind.COUNTER,
            "Provider calls that required retry",
            zero_is_meaningful=True,
        ),
        MetricDefinition(
            "abandoned_provider_calls",
            MetricUnit.COUNT,
            MetricKind.COUNTER,
            "Provider calls abandoned after budget/timeout",
            zero_is_meaningful=True,
        ),
        MetricDefinition(
            "accepted_mutation_quality_bps",
            MetricUnit.BASIS_POINTS,
            MetricKind.QUALITY,
            "Accepted mutation quality in basis points",
            higher_is_better=True,
        ),
        MetricDefinition(
            "rollback_rate_bps",
            MetricUnit.BASIS_POINTS,
            MetricKind.RATE,
            "Rollback rate in basis points",
            higher_is_better=False,
        ),
        MetricDefinition(
            "failure_rate_bps",
            MetricUnit.BASIS_POINTS,
            MetricKind.RATE,
            "Failure rate in basis points",
            higher_is_better=False,
        ),
    )


@dataclass(frozen=True)
class BaselineBinding:
    """Binds tree, environment, workload, metric definitions, and criteria."""

    SCHEMA: ClassVar[str] = BINDING_SCHEMA

    tree_id: str
    environment: BaselineEnvironment
    workload: WorkloadDefinition
    metric_definitions: tuple[MetricDefinition, ...]
    criteria: BaselineCriteria
    repository_id: str = "repository:local"
    policy_id: str = "policy:duckdb-quack-baseline@1"
    policy_revision: str = "revision:1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        if not isinstance(self.environment, BaselineEnvironment):
            raise DuckDBQuackBaselineError("environment must be BaselineEnvironment")
        if not isinstance(self.workload, WorkloadDefinition):
            raise DuckDBQuackBaselineError("workload must be WorkloadDefinition")
        if not isinstance(self.criteria, BaselineCriteria):
            raise DuckDBQuackBaselineError("criteria must be BaselineCriteria")
        definitions = tuple(self.metric_definitions)
        if not definitions:
            raise DuckDBQuackBaselineError("metric_definitions must be non-empty")
        if len(definitions) > MAX_METRICS:
            raise DuckDBQuackBaselineError("metric_definitions exceed bound")
        names: list[str] = []
        for item in definitions:
            if not isinstance(item, MetricDefinition):
                raise DuckDBQuackBaselineError(
                    "metric_definitions entries must be MetricDefinition"
                )
            names.append(item.name)
        if len(set(names)) != len(names):
            raise DuckDBQuackBaselineError("metric_definitions names must be unique")
        object.__setattr__(self, "metric_definitions", definitions)
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(
            self, "policy_revision", _text(self.policy_revision, "policy_revision")
        )

    @property
    def metric_definition_identity(self) -> str:
        return content_identity([item.to_dict() for item in self.metric_definitions])

    @property
    def identity_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "tree_id": self.tree_id,
            "repository_id": self.repository_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "environment": self.environment.to_dict(),
            "workload": self.workload.to_dict(),
            "metric_definitions": [item.to_dict() for item in self.metric_definitions],
            "metric_definition_identity": self.metric_definition_identity,
            "criteria": self.criteria.to_dict(),
            "criteria_identity": self.criteria.identity_id,
        }
        if include_identity:
            payload["identity_id"] = self.identity_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BaselineBinding":
        if not isinstance(payload, Mapping):
            raise DuckDBQuackBaselineError("binding must be an object")
        raw_defs = payload.get("metric_definitions") or ()
        if not isinstance(raw_defs, Sequence) or isinstance(raw_defs, (str, bytes)):
            raise DuckDBQuackBaselineError("metric_definitions must be a sequence")
        return cls(
            tree_id=str(payload.get("tree_id") or ""),
            repository_id=str(payload.get("repository_id") or "repository:local"),
            policy_id=str(payload.get("policy_id") or "policy:duckdb-quack-baseline@1"),
            policy_revision=str(payload.get("policy_revision") or "revision:1"),
            environment=BaselineEnvironment.from_dict(
                payload.get("environment") or {}
            ),
            workload=WorkloadDefinition.from_dict(payload.get("workload") or {}),
            metric_definitions=tuple(
                MetricDefinition.from_dict(item) for item in raw_defs
            ),
            criteria=BaselineCriteria.from_dict(payload.get("criteria") or {}),
        )


# ---------------------------------------------------------------------------
# Provider usage and stratum observations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderUsageCounters:
    """Counts accepted / rejected / retry / abandoned provider usage."""

    SCHEMA: ClassVar[str] = PROVIDER_USAGE_SCHEMA

    accepted: int = 0
    rejected: int = 0
    retry: int = 0
    abandoned: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted", _nonnegative_int(self.accepted, "accepted"))
        object.__setattr__(self, "rejected", _nonnegative_int(self.rejected, "rejected"))
        object.__setattr__(self, "retry", _nonnegative_int(self.retry, "retry"))
        object.__setattr__(
            self, "abandoned", _nonnegative_int(self.abandoned, "abandoned")
        )

    @property
    def total(self) -> int:
        return self.accepted + self.rejected + self.retry + self.abandoned

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "retry": self.retry,
            "abandoned": self.abandoned,
            "total": self.total,
            "outcomes": list(_PROVIDER_OUTCOMES),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderUsageCounters":
        if not isinstance(payload, Mapping):
            raise DuckDBQuackBaselineError("provider usage must be an object")
        return cls(
            accepted=int(payload.get("accepted", 0)),
            rejected=int(payload.get("rejected", 0)),
            retry=int(payload.get("retry", 0)),
            abandoned=int(payload.get("abandoned", 0)),
        )

    def add(self, other: "ProviderUsageCounters") -> "ProviderUsageCounters":
        return ProviderUsageCounters(
            accepted=self.accepted + other.accepted,
            rejected=self.rejected + other.rejected,
            retry=self.retry + other.retry,
            abandoned=self.abandoned + other.abandoned,
        )


@dataclass(frozen=True)
class StratumObservation:
    """Metrics collected for one measurement stratum."""

    SCHEMA: ClassVar[str] = STRATUM_OBSERVATION_SCHEMA

    stratum: BaselineStratum
    samples: int
    metrics: Mapping[str, MetricSample]
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "stratum", _parse_enum(BaselineStratum, self.stratum, "stratum")
        )
        object.__setattr__(
            self, "samples", _nonnegative_int(self.samples, "samples", maximum=MAX_SAMPLES)
        )
        object.__setattr__(self, "seed", _nonnegative_int(self.seed, "seed"))
        cleaned: dict[str, MetricSample] = {}
        for key, sample in dict(self.metrics or {}).items():
            name = _text(key, "metrics.key", maximum=128)
            if isinstance(sample, Mapping):
                resolved = MetricSample.from_dict(sample)
            elif isinstance(sample, MetricSample):
                resolved = sample
            else:
                raise DuckDBQuackBaselineError(
                    "metrics values must be MetricSample or mapping"
                )
            if resolved.metric_name != name:
                raise DuckDBQuackBaselineError(
                    f"metric key {name!r} does not match sample name "
                    f"{resolved.metric_name!r}"
                )
            cleaned[name] = resolved
        object.__setattr__(self, "metrics", _mapping_proxy(cleaned))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "stratum": self.stratum.value
            if isinstance(self.stratum, Enum)
            else self.stratum,
            "samples": self.samples,
            "seed": self.seed,
            "metrics": {
                name: sample.to_dict() for name, sample in self.metrics.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StratumObservation":
        if not isinstance(payload, Mapping):
            raise DuckDBQuackBaselineError("stratum observation must be an object")
        raw_metrics = payload.get("metrics") or {}
        if not isinstance(raw_metrics, Mapping):
            raise DuckDBQuackBaselineError("metrics must be an object")
        return cls(
            stratum=payload.get("stratum", BaselineStratum.COLD),
            samples=int(payload.get("samples", 0)),
            seed=int(payload.get("seed", 0)),
            metrics={
                str(name): MetricSample.from_dict(sample)
                for name, sample in raw_metrics.items()
            },
        )


# ---------------------------------------------------------------------------
# Baseline reports
# ---------------------------------------------------------------------------


def _aggregate_measured(
    observations: Sequence[StratumObservation],
    metric_name: str,
) -> MetricSample:
    """Sum measured values across strata; unavailable if any stratum missing."""

    total = 0
    saw_measured = False
    for observation in observations:
        sample = observation.metrics.get(metric_name)
        if sample is None:
            return MetricSample.unavailable(
                metric_name,
                UnavailableReason.TELEMETRY_MISSING,
            )
        if sample.is_unavailable:
            return MetricSample.unavailable(
                metric_name,
                sample.reason_code or UnavailableReason.TELEMETRY_MISSING.value,
                sensor_id=sample.sensor_id,
            )
        total += sample.value
        saw_measured = True
    if not saw_measured:
        return MetricSample.unavailable(
            metric_name, UnavailableReason.STRATUM_NOT_RUN
        )
    return MetricSample.measured(metric_name, total)


@dataclass(frozen=True)
class SupervisorStateBaseline:
    """``SupervisorStateBaseline@1`` — state and latency baseline report."""

    SCHEMA: ClassVar[str] = SUPERVISOR_STATE_BASELINE_SCHEMA
    INTERFACE: ClassVar[str] = SUPERVISOR_STATE_BASELINE_INTERFACE

    binding: BaselineBinding
    strata: tuple[StratumObservation, ...]
    aggregates: Mapping[str, MetricSample]
    sample_count: int
    confidence_bps: int
    verdict: BaselineVerdict
    reason_codes: tuple[str, ...] = ()
    evidence: str = EVIDENCE
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID

    def __post_init__(self) -> None:
        if not isinstance(self.binding, BaselineBinding):
            raise DuckDBQuackBaselineError("binding must be BaselineBinding")
        strata = tuple(self.strata)
        if not strata:
            raise DuckDBQuackBaselineError("strata must be non-empty")
        seen: set[str] = set()
        for item in strata:
            if not isinstance(item, StratumObservation):
                raise DuckDBQuackBaselineError(
                    "strata entries must be StratumObservation"
                )
            key = item.stratum.value if isinstance(item.stratum, Enum) else str(item.stratum)
            if key in seen:
                raise DuckDBQuackBaselineError(f"duplicate stratum {key!r}")
            seen.add(key)
        object.__setattr__(self, "strata", strata)
        cleaned = {
            _text(name, "aggregates.key", maximum=128): (
                sample
                if isinstance(sample, MetricSample)
                else MetricSample.from_dict(sample)
            )
            for name, sample in dict(self.aggregates or {}).items()
        }
        object.__setattr__(self, "aggregates", _mapping_proxy(cleaned))
        object.__setattr__(
            self,
            "sample_count",
            _nonnegative_int(self.sample_count, "sample_count", maximum=MAX_SAMPLES),
        )
        object.__setattr__(
            self, "confidence_bps", _bps(self.confidence_bps, "confidence_bps")
        )
        object.__setattr__(
            self, "verdict", _parse_enum(BaselineVerdict, self.verdict, "verdict")
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(_text(code, "reason_codes.item", maximum=96) for code in self.reason_codes),
        )
        object.__setattr__(self, "evidence", _text(self.evidence, "evidence"))
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id", maximum=32))
        object.__setattr__(self, "goal_id", _text(self.goal_id, "goal_id", maximum=32))

    @property
    def identity_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def measured_aggregate(self, name: str) -> int | None:
        sample = self.aggregates.get(name)
        if sample is None:
            return None
        return sample.measured_value()

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "contract_version": BASELINE_CONTRACT_VERSION,
            "evidence": self.evidence,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "binding": self.binding.to_dict(),
            "strata": [item.to_dict() for item in self.strata],
            "aggregates": {
                name: sample.to_dict() for name, sample in self.aggregates.items()
            },
            "sample_count": self.sample_count,
            "confidence_bps": self.confidence_bps,
            "verdict": self.verdict.value
            if isinstance(self.verdict, Enum)
            else self.verdict,
            "reason_codes": list(self.reason_codes),
        }
        if include_identity:
            payload["identity_id"] = self.identity_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SupervisorStateBaseline":
        if not isinstance(payload, Mapping):
            raise DuckDBQuackBaselineError("state baseline must be an object")
        raw_strata = payload.get("strata") or ()
        raw_agg = payload.get("aggregates") or {}
        if not isinstance(raw_strata, Sequence) or isinstance(raw_strata, (str, bytes)):
            raise DuckDBQuackBaselineError("strata must be a sequence")
        if not isinstance(raw_agg, Mapping):
            raise DuckDBQuackBaselineError("aggregates must be an object")
        return cls(
            binding=BaselineBinding.from_dict(payload.get("binding") or {}),
            strata=tuple(StratumObservation.from_dict(item) for item in raw_strata),
            aggregates={
                str(name): MetricSample.from_dict(sample)
                for name, sample in raw_agg.items()
            },
            sample_count=int(payload.get("sample_count", 0)),
            confidence_bps=int(payload.get("confidence_bps", 0)),
            verdict=payload.get("verdict", BaselineVerdict.REJECTED),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            evidence=str(payload.get("evidence") or EVIDENCE),
            task_id=str(payload.get("task_id") or TASK_ID),
            goal_id=str(payload.get("goal_id") or GOAL_ID),
        )


@dataclass(frozen=True)
class LLMChurnBaseline:
    """``LLMChurnBaseline@1`` — provider/context churn baseline report."""

    SCHEMA: ClassVar[str] = LLM_CHURN_BASELINE_SCHEMA
    INTERFACE: ClassVar[str] = LLM_CHURN_BASELINE_INTERFACE

    binding: BaselineBinding
    strata: tuple[StratumObservation, ...]
    aggregates: Mapping[str, MetricSample]
    provider_usage: ProviderUsageCounters
    sample_count: int
    confidence_bps: int
    verdict: BaselineVerdict
    reason_codes: tuple[str, ...] = ()
    evidence: str = EVIDENCE
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID

    def __post_init__(self) -> None:
        if not isinstance(self.binding, BaselineBinding):
            raise DuckDBQuackBaselineError("binding must be BaselineBinding")
        if not isinstance(self.provider_usage, ProviderUsageCounters):
            raise DuckDBQuackBaselineError(
                "provider_usage must be ProviderUsageCounters"
            )
        strata = tuple(self.strata)
        if not strata:
            raise DuckDBQuackBaselineError("strata must be non-empty")
        object.__setattr__(self, "strata", strata)
        cleaned = {
            _text(name, "aggregates.key", maximum=128): (
                sample
                if isinstance(sample, MetricSample)
                else MetricSample.from_dict(sample)
            )
            for name, sample in dict(self.aggregates or {}).items()
        }
        object.__setattr__(self, "aggregates", _mapping_proxy(cleaned))
        object.__setattr__(
            self,
            "sample_count",
            _nonnegative_int(self.sample_count, "sample_count", maximum=MAX_SAMPLES),
        )
        object.__setattr__(
            self, "confidence_bps", _bps(self.confidence_bps, "confidence_bps")
        )
        object.__setattr__(
            self, "verdict", _parse_enum(BaselineVerdict, self.verdict, "verdict")
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(_text(code, "reason_codes.item", maximum=96) for code in self.reason_codes),
        )
        object.__setattr__(self, "evidence", _text(self.evidence, "evidence"))
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id", maximum=32))
        object.__setattr__(self, "goal_id", _text(self.goal_id, "goal_id", maximum=32))

    @property
    def identity_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def measured_aggregate(self, name: str) -> int | None:
        sample = self.aggregates.get(name)
        if sample is None:
            return None
        return sample.measured_value()

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "contract_version": BASELINE_CONTRACT_VERSION,
            "evidence": self.evidence,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "binding": self.binding.to_dict(),
            "strata": [item.to_dict() for item in self.strata],
            "aggregates": {
                name: sample.to_dict() for name, sample in self.aggregates.items()
            },
            "provider_usage": self.provider_usage.to_dict(),
            "sample_count": self.sample_count,
            "confidence_bps": self.confidence_bps,
            "verdict": self.verdict.value
            if isinstance(self.verdict, Enum)
            else self.verdict,
            "reason_codes": list(self.reason_codes),
        }
        if include_identity:
            payload["identity_id"] = self.identity_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LLMChurnBaseline":
        if not isinstance(payload, Mapping):
            raise DuckDBQuackBaselineError("llm churn baseline must be an object")
        raw_strata = payload.get("strata") or ()
        raw_agg = payload.get("aggregates") or {}
        if not isinstance(raw_strata, Sequence) or isinstance(raw_strata, (str, bytes)):
            raise DuckDBQuackBaselineError("strata must be a sequence")
        if not isinstance(raw_agg, Mapping):
            raise DuckDBQuackBaselineError("aggregates must be an object")
        return cls(
            binding=BaselineBinding.from_dict(payload.get("binding") or {}),
            strata=tuple(StratumObservation.from_dict(item) for item in raw_strata),
            aggregates={
                str(name): MetricSample.from_dict(sample)
                for name, sample in raw_agg.items()
            },
            provider_usage=ProviderUsageCounters.from_dict(
                payload.get("provider_usage") or {}
            ),
            sample_count=int(payload.get("sample_count", 0)),
            confidence_bps=int(payload.get("confidence_bps", 0)),
            verdict=payload.get("verdict", BaselineVerdict.REJECTED),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            evidence=str(payload.get("evidence") or EVIDENCE),
            task_id=str(payload.get("task_id") or TASK_ID),
            goal_id=str(payload.get("goal_id") or GOAL_ID),
        )


@dataclass(frozen=True)
class DuckDBQuackBaselineReport:
    """Combined sealed baseline report for DQP-009."""

    SCHEMA: ClassVar[str] = COMBINED_BASELINE_REPORT_SCHEMA

    state_baseline: SupervisorStateBaseline
    llm_churn_baseline: LLMChurnBaseline
    criteria: BaselineCriteria
    verdict: BaselineVerdict
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.state_baseline, SupervisorStateBaseline):
            raise DuckDBQuackBaselineError(
                "state_baseline must be SupervisorStateBaseline"
            )
        if not isinstance(self.llm_churn_baseline, LLMChurnBaseline):
            raise DuckDBQuackBaselineError(
                "llm_churn_baseline must be LLMChurnBaseline"
            )
        if not isinstance(self.criteria, BaselineCriteria):
            raise DuckDBQuackBaselineError("criteria must be BaselineCriteria")
        object.__setattr__(
            self, "verdict", _parse_enum(BaselineVerdict, self.verdict, "verdict")
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(_text(code, "reason_codes.item", maximum=96) for code in self.reason_codes),
        )

    @property
    def identity_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "contract_version": BASELINE_CONTRACT_VERSION,
            "evidence": EVIDENCE,
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
            "interfaces": [
                SUPERVISOR_STATE_BASELINE_INTERFACE,
                LLM_CHURN_BASELINE_INTERFACE,
            ],
            "state_baseline": self.state_baseline.to_dict(),
            "llm_churn_baseline": self.llm_churn_baseline.to_dict(),
            "criteria": self.criteria.to_dict(),
            "criteria_identity": self.criteria.identity_id,
            "verdict": self.verdict.value
            if isinstance(self.verdict, Enum)
            else self.verdict,
            "reason_codes": list(self.reason_codes),
        }
        if include_identity:
            payload["identity_id"] = self.identity_id
        return payload


# ---------------------------------------------------------------------------
# Hermetic workload simulation (deterministic, no I/O side effects)
# ---------------------------------------------------------------------------


def default_workload(
    *,
    seed: int = DEFAULT_WORKLOAD_SEED,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
) -> WorkloadDefinition:
    return WorkloadDefinition(
        workload_id="workload:duckdb-quack-baseline-hermetic@1",
        seed=seed,
        sample_count=sample_count,
        strata=_REQUIRED_STRATA,
        operations=(
            "read_plan",
            "parse_taskboard",
            "open_db",
            "claim_task",
            "poll_ready",
            "compile_context",
            "provider_call",
            "cache_lookup",
            "accept_mutation",
            "rollback",
        ),
        description=(
            "Fixed hermetic workload measuring state I/O, lock/claim latency, "
            "and LLM churn under cold/warm/restart/parallel strata"
        ),
    )


def _stratum_factor(stratum: str) -> int:
    return {
        BaselineStratum.COLD.value: 4,
        BaselineStratum.WARM.value: 2,
        BaselineStratum.RESTART.value: 3,
        BaselineStratum.PARALLEL.value: 5,
    }[stratum]


def _simulate_state_stratum(
    *,
    stratum: str,
    seed: int,
    samples: int,
) -> StratumObservation:
    """Deterministic state/latency observation for one stratum."""

    factor = _stratum_factor(stratum)
    # Warm reuses cut file I/O and opens relative to cold; salt stays small so
    # the warm < cold relationship is stable for the hermetic workload.
    is_warm = stratum == BaselineStratum.WARM.value
    read_base = samples * (3 if is_warm else 12)
    parse_base = samples * (2 if is_warm else 8)
    open_base = samples * (1 if is_warm else factor)
    salt = (seed ^ (factor * 0x9E37)) & 0x3  # 0..3, cannot invert warm/cold gap
    metrics = {
        "file_reads": MetricSample.measured("file_reads", read_base + salt),
        "file_writes": MetricSample.measured(
            "file_writes", samples * max(1, factor // 2)
        ),
        "file_parses": MetricSample.measured("file_parses", parse_base + salt),
        "independent_db_opens": MetricSample.measured(
            "independent_db_opens", open_base
        ),
        "lock_waits_ms": MetricSample.measured(
            "lock_waits_ms",
            samples * factor * (12 if stratum == BaselineStratum.PARALLEL.value else 4),
        ),
        "noop_polls": MetricSample.measured(
            "noop_polls", samples * (2 if stratum == BaselineStratum.COLD.value else 1)
        ),
        "task_claim_latency_ms": MetricSample.measured(
            "task_claim_latency_ms",
            samples * factor * (3 if is_warm else 8),
        ),
        "queue_latency_ms": MetricSample.measured(
            "queue_latency_ms",
            samples * factor * (6 if stratum == BaselineStratum.PARALLEL.value else 2),
        ),
        # Measured zeros are intentional and distinct from missing telemetry.
        "rollback_count": MetricSample.measured("rollback_count", 0),
        "failure_count": MetricSample.measured("failure_count", 0),
    }
    return StratumObservation(
        stratum=stratum,
        samples=samples,
        seed=seed ^ factor,
        metrics=metrics,
    )


def _simulate_llm_stratum(
    *,
    stratum: str,
    seed: int,
    samples: int,
) -> tuple[StratumObservation, ProviderUsageCounters]:
    """Deterministic LLM-churn observation and provider usage for one stratum."""

    factor = _stratum_factor(stratum)
    warm = stratum == BaselineStratum.WARM.value
    # Provider outcomes per sample (deterministic mix).
    # Rejected / retry / abandoned are first-class and may be measured zero
    # when the stratum sample budget is too small; non-warm strata always
    # produce at least one abandoned call once samples >= 1 so accounting is
    # exercised by the default hermetic workload.
    accepted = samples * (1 if warm else 2)
    rejected = samples // 4  # measured zero when samples < 4
    retry = samples // 3
    abandoned = 0 if warm else max(1, samples // 4)
    provider_usage = ProviderUsageCounters(
        accepted=accepted,
        rejected=rejected,
        retry=retry,
        abandoned=abandoned,
    )
    total_calls = provider_usage.total
    # Warm context is strictly smaller than cold for the same sample budget.
    context_bytes = samples * (1_200 if warm else factor * 2_400)
    cache_hits = samples * (3 if warm else 1)
    cache_misses = samples * (1 if warm else 3)
    quality_bps = 9_200 if warm else 8_500
    metrics = {
        "context_bytes": MetricSample.measured("context_bytes", context_bytes),
        "provider_calls": MetricSample.measured("provider_calls", total_calls),
        "input_tokens": MetricSample.measured(
            "input_tokens", context_bytes // 4 + total_calls * 40
        ),
        "output_tokens": MetricSample.measured(
            "output_tokens", total_calls * 120 + (seed % 7)
        ),
        "duplicate_semantic_inputs": MetricSample.measured(
            "duplicate_semantic_inputs",
            0 if warm else samples // 2,
        ),
        "cache_reuse_hits": MetricSample.measured("cache_reuse_hits", cache_hits),
        "cache_reuse_misses": MetricSample.measured("cache_reuse_misses", cache_misses),
        "accepted_mutations": MetricSample.measured("accepted_mutations", accepted),
        "rejected_provider_calls": MetricSample.measured(
            "rejected_provider_calls", rejected
        ),
        "retry_provider_calls": MetricSample.measured("retry_provider_calls", retry),
        "abandoned_provider_calls": MetricSample.measured(
            "abandoned_provider_calls", abandoned
        ),
        "accepted_mutation_quality_bps": MetricSample.measured(
            "accepted_mutation_quality_bps", quality_bps
        ),
        "rollback_rate_bps": MetricSample.measured("rollback_rate_bps", 0),
        "failure_rate_bps": MetricSample.measured("failure_rate_bps", 0),
    }
    observation = StratumObservation(
        stratum=stratum,
        samples=samples,
        seed=seed ^ (factor * 17),
        metrics=metrics,
    )
    return observation, provider_usage


def _confidence_bps(sample_count: int, min_samples: int) -> int:
    if sample_count <= 0:
        return 0
    if sample_count < min_samples:
        return max(1, (sample_count * BASIS_POINTS) // max(1, min_samples) // 2)
    # Saturate toward high confidence with more samples; still not promotion.
    ratio = min(sample_count, min_samples * 4) / float(min_samples * 4)
    return int(6_000 + ratio * 3_500)


def _evaluate_state_verdict(
    *,
    sample_count: int,
    criteria: BaselineCriteria,
    aggregates: Mapping[str, MetricSample],
) -> tuple[BaselineVerdict, tuple[str, ...]]:
    reasons: list[str] = []
    if sample_count < criteria.min_samples:
        reasons.append("insufficient_samples")
    for key in ("rollback_count", "failure_count"):
        sample = aggregates.get(key)
        if sample is None or sample.is_unavailable:
            reasons.append(f"missing_metric:{key}")
        elif sample.value != 0:
            reasons.append(f"nonzero_{key}")
    for key in SAFETY_FLOOR_KEYS:
        if int(criteria.safety_floors.get(key, 0)) != 0:
            reasons.append(f"safety_floor_nonzero:{key}")
    if reasons:
        if "insufficient_samples" in reasons and len(reasons) == 1:
            return BaselineVerdict.INSUFFICIENT, tuple(reasons)
        return BaselineVerdict.REJECTED, tuple(reasons)
    return BaselineVerdict.ESTABLISHED, ()


def _evaluate_llm_verdict(
    *,
    sample_count: int,
    criteria: BaselineCriteria,
    aggregates: Mapping[str, MetricSample],
    provider_usage: ProviderUsageCounters,
) -> tuple[BaselineVerdict, tuple[str, ...]]:
    reasons: list[str] = []
    if sample_count < criteria.min_samples:
        reasons.append("insufficient_samples")
    quality = aggregates.get("accepted_mutation_quality_bps")
    if quality is None or quality.is_unavailable:
        reasons.append("missing_metric:accepted_mutation_quality_bps")
    elif quality.value < criteria.min_accepted_mutation_quality_bps:
        reasons.append("quality_below_floor")
    for rate_key, ceiling in (
        ("rollback_rate_bps", criteria.max_rollback_rate_bps),
        ("failure_rate_bps", criteria.max_failure_rate_bps),
    ):
        sample = aggregates.get(rate_key)
        if sample is None or sample.is_unavailable:
            reasons.append(f"missing_metric:{rate_key}")
        elif sample.value > ceiling:
            reasons.append(f"{rate_key}_above_ceiling")
    # Provider usage accounting must be complete and consistent.
    calls = aggregates.get("provider_calls")
    if calls is None or calls.is_unavailable:
        reasons.append("missing_metric:provider_calls")
    elif calls.value != provider_usage.total:
        reasons.append("provider_usage_total_mismatch")
    for key in (
        "rejected_provider_calls",
        "retry_provider_calls",
        "abandoned_provider_calls",
    ):
        sample = aggregates.get(key)
        if sample is None or sample.is_unavailable:
            reasons.append(f"missing_metric:{key}")
    if reasons:
        if "insufficient_samples" in reasons and len(reasons) == 1:
            return BaselineVerdict.INSUFFICIENT, tuple(reasons)
        return BaselineVerdict.REJECTED, tuple(reasons)
    return BaselineVerdict.ESTABLISHED, ()


def establish_supervisor_state_baseline(
    *,
    tree_id: str,
    environment: BaselineEnvironment | None = None,
    workload: WorkloadDefinition | None = None,
    criteria: BaselineCriteria | None = None,
    repository_id: str = "repository:local",
    missing_metrics: Iterable[str] = (),
) -> SupervisorStateBaseline:
    """Establish a hermetic ``SupervisorStateBaseline@1`` measurement."""

    env = environment or BaselineEnvironment.capture()
    work = workload or default_workload()
    crit = criteria or BaselineCriteria.sealed_defaults()
    crit.assert_establishment_safe()
    definitions = default_state_metric_definitions()
    binding = BaselineBinding(
        tree_id=tree_id,
        environment=env,
        workload=work,
        metric_definitions=definitions,
        criteria=crit,
        repository_id=repository_id,
    )
    missing = frozenset(_text(name, "missing_metrics.item", maximum=128) for name in missing_metrics)
    strata: list[StratumObservation] = []
    samples_per_stratum = max(1, work.sample_count // len(work.strata))
    for index, stratum in enumerate(work.strata):
        observation = _simulate_state_stratum(
            stratum=stratum,
            seed=work.seed + index,
            samples=samples_per_stratum,
        )
        if missing:
            metrics = dict(observation.metrics)
            for name in missing:
                if name in metrics:
                    metrics[name] = MetricSample.unavailable(
                        name, UnavailableReason.TELEMETRY_MISSING
                    )
            observation = StratumObservation(
                stratum=observation.stratum,
                samples=observation.samples,
                seed=observation.seed,
                metrics=metrics,
            )
        strata.append(observation)
    aggregates = {
        name: _aggregate_measured(strata, name) for name in STATE_METRIC_NAMES
    }
    sample_count = sum(item.samples for item in strata)
    verdict, reasons = _evaluate_state_verdict(
        sample_count=sample_count,
        criteria=crit,
        aggregates=aggregates,
    )
    return SupervisorStateBaseline(
        binding=binding,
        strata=tuple(strata),
        aggregates=aggregates,
        sample_count=sample_count,
        confidence_bps=_confidence_bps(sample_count, crit.min_samples),
        verdict=verdict,
        reason_codes=reasons,
    )


def establish_llm_churn_baseline(
    *,
    tree_id: str,
    environment: BaselineEnvironment | None = None,
    workload: WorkloadDefinition | None = None,
    criteria: BaselineCriteria | None = None,
    repository_id: str = "repository:local",
    missing_metrics: Iterable[str] = (),
) -> LLMChurnBaseline:
    """Establish a hermetic ``LLMChurnBaseline@1`` measurement."""

    env = environment or BaselineEnvironment.capture()
    work = workload or default_workload()
    crit = criteria or BaselineCriteria.sealed_defaults()
    crit.assert_establishment_safe()
    definitions = default_llm_metric_definitions()
    binding = BaselineBinding(
        tree_id=tree_id,
        environment=env,
        workload=work,
        metric_definitions=definitions,
        criteria=crit,
        repository_id=repository_id,
    )
    missing = frozenset(
        _text(name, "missing_metrics.item", maximum=128) for name in missing_metrics
    )
    strata: list[StratumObservation] = []
    usage = ProviderUsageCounters()
    samples_per_stratum = max(1, work.sample_count // len(work.strata))
    for index, stratum in enumerate(work.strata):
        observation, stratum_usage = _simulate_llm_stratum(
            stratum=stratum,
            seed=work.seed + index * 31,
            samples=samples_per_stratum,
        )
        if missing:
            metrics = dict(observation.metrics)
            for name in missing:
                if name in metrics:
                    metrics[name] = MetricSample.unavailable(
                        name, UnavailableReason.TELEMETRY_MISSING
                    )
            observation = StratumObservation(
                stratum=observation.stratum,
                samples=observation.samples,
                seed=observation.seed,
                metrics=metrics,
            )
        strata.append(observation)
        usage = usage.add(stratum_usage)
    aggregates = {
        name: _aggregate_measured(strata, name) for name in LLM_CHURN_METRIC_NAMES
    }
    # Quality / rate metrics are not summed; recompute from mean of measured strata.
    for rate_name in (
        "accepted_mutation_quality_bps",
        "rollback_rate_bps",
        "failure_rate_bps",
    ):
        values: list[int] = []
        unavailable = False
        for observation in strata:
            sample = observation.metrics.get(rate_name)
            if sample is None or sample.is_unavailable:
                unavailable = True
                break
            values.append(sample.value)
        if unavailable or not values:
            aggregates[rate_name] = MetricSample.unavailable(
                rate_name, UnavailableReason.TELEMETRY_MISSING
            )
        else:
            aggregates[rate_name] = MetricSample.measured(
                rate_name, sum(values) // len(values)
            )
    sample_count = sum(item.samples for item in strata)
    verdict, reasons = _evaluate_llm_verdict(
        sample_count=sample_count,
        criteria=crit,
        aggregates=aggregates,
        provider_usage=usage,
    )
    return LLMChurnBaseline(
        binding=binding,
        strata=tuple(strata),
        aggregates=aggregates,
        provider_usage=usage,
        sample_count=sample_count,
        confidence_bps=_confidence_bps(sample_count, crit.min_samples),
        verdict=verdict,
        reason_codes=reasons,
    )


def establish_duckdb_quack_baselines(
    *,
    tree_id: str,
    environment: BaselineEnvironment | None = None,
    workload: WorkloadDefinition | None = None,
    criteria: BaselineCriteria | None = None,
    repository_id: str = "repository:local",
) -> DuckDBQuackBaselineReport:
    """Establish both state and LLM-churn baselines under one sealed binding."""

    env = environment or BaselineEnvironment.capture()
    work = workload or default_workload()
    crit = criteria or BaselineCriteria.sealed_defaults()
    state = establish_supervisor_state_baseline(
        tree_id=tree_id,
        environment=env,
        workload=work,
        criteria=crit,
        repository_id=repository_id,
    )
    llm = establish_llm_churn_baseline(
        tree_id=tree_id,
        environment=env,
        workload=work,
        criteria=crit,
        repository_id=repository_id,
    )
    reasons: list[str] = []
    if state.verdict is not BaselineVerdict.ESTABLISHED:
        reasons.append(f"state:{state.verdict.value}")
        reasons.extend(f"state:{code}" for code in state.reason_codes)
    if llm.verdict is not BaselineVerdict.ESTABLISHED:
        reasons.append(f"llm:{llm.verdict.value}")
        reasons.extend(f"llm:{code}" for code in llm.reason_codes)
    if reasons:
        verdict = BaselineVerdict.REJECTED
        if (
            state.verdict is BaselineVerdict.INSUFFICIENT
            or llm.verdict is BaselineVerdict.INSUFFICIENT
        ) and state.verdict is not BaselineVerdict.REJECTED and llm.verdict is not BaselineVerdict.REJECTED:
            verdict = BaselineVerdict.INSUFFICIENT
    else:
        verdict = BaselineVerdict.ESTABLISHED
    return DuckDBQuackBaselineReport(
        state_baseline=state,
        llm_churn_baseline=llm,
        criteria=crit,
        verdict=verdict,
        reason_codes=tuple(reasons),
    )


def assert_criteria_not_weakened(
    sealed: BaselineCriteria | Mapping[str, Any],
    candidate: BaselineCriteria | Mapping[str, Any],
) -> None:
    """Reject candidate criteria that weaken sealed safety/durability/quality floors.

    Used when regenerating or comparing baselines so a candidate change cannot
    re-establish a baseline by lowering the bar.
    """

    sealed_criteria = (
        sealed
        if isinstance(sealed, BaselineCriteria)
        else BaselineCriteria.from_dict(sealed)
    )
    candidate_criteria = (
        candidate
        if isinstance(candidate, BaselineCriteria)
        else BaselineCriteria.from_dict(candidate)
    )
    weakened, reasons = candidate_criteria.is_weakened_relative_to(sealed_criteria)
    if weakened:
        raise DuckDBQuackBaselineError(
            "candidate criteria weaken sealed baseline floors: "
            + ", ".join(reasons)
        )


def regenerate_baseline_with_criteria(
    *,
    tree_id: str,
    sealed_criteria: BaselineCriteria,
    candidate_criteria: BaselineCriteria,
    environment: BaselineEnvironment | None = None,
    workload: WorkloadDefinition | None = None,
    repository_id: str = "repository:local",
) -> DuckDBQuackBaselineReport:
    """Regenerate baselines only when candidate criteria do not weaken sealed floors."""

    assert_criteria_not_weakened(sealed_criteria, candidate_criteria)
    return establish_duckdb_quack_baselines(
        tree_id=tree_id,
        environment=environment,
        workload=workload,
        criteria=candidate_criteria,
        repository_id=repository_id,
    )


def metric_sample_distinguishes_missing_from_zero() -> bool:
    """Documented contract helper used by tests and importers."""

    zero = MetricSample.measured("probe", 0)
    missing = MetricSample.unavailable(
        "probe", UnavailableReason.TELEMETRY_MISSING
    )
    return (
        zero.is_measured
        and zero.value == 0
        and zero.measured_value() == 0
        and missing.is_unavailable
        and missing.measured_value() is None
        and zero.to_dict() != missing.to_dict()
    )


__all__ = (
    "BASELINE_CONTRACT_VERSION",
    "BASIS_POINTS",
    "BaselineBinding",
    "BaselineCriteria",
    "BaselineEnvironment",
    "BaselineStratum",
    "BaselineVerdict",
    "DEFAULT_MAX_FAILURE_RATE_BPS",
    "DEFAULT_MAX_ROLLBACK_RATE_BPS",
    "DEFAULT_MIN_ACCEPTED_MUTATION_QUALITY_BPS",
    "DEFAULT_MIN_SAMPLES",
    "DEFAULT_SAMPLE_COUNT",
    "DEFAULT_WORKLOAD_SEED",
    "DURABILITY_FLOOR_KEYS",
    "DuckDBQuackBaselineError",
    "DuckDBQuackBaselineReport",
    "EVIDENCE",
    "GOAL_ID",
    "HERMETIC_SENSOR",
    "LLM_CHURN_BASELINE_INTERFACE",
    "LLM_CHURN_BASELINE_SCHEMA",
    "LLM_CHURN_METRIC_NAMES",
    "LLMChurnBaseline",
    "MISSING_TELEMETRY_SENSOR",
    "MetricDefinition",
    "MetricKind",
    "MetricSample",
    "MetricUnit",
    "ProviderUsageCounters",
    "ProviderUsageOutcome",
    "SAFETY_FLOOR_KEYS",
    "STATE_METRIC_NAMES",
    "SUPERVISOR_STATE_BASELINE_INTERFACE",
    "SUPERVISOR_STATE_BASELINE_SCHEMA",
    "StratumObservation",
    "SupervisorStateBaseline",
    "TASK_ID",
    "TelemetryStatus",
    "UnavailableReason",
    "WorkloadDefinition",
    "assert_criteria_not_weakened",
    "content_identity",
    "default_llm_metric_definitions",
    "default_state_metric_definitions",
    "default_workload",
    "establish_duckdb_quack_baselines",
    "establish_llm_churn_baseline",
    "establish_supervisor_state_baseline",
    "metric_sample_distinguishes_missing_from_zero",
    "regenerate_baseline_with_criteria",
)
