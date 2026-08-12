"""LLM-avoidance metrics and attempt attribution (WPD-050).

``LlmAvoidanceMetrics@1`` / ``wpd/llm-avoidance-metrics@1``

Observability-only surface that attributes every implementation attempt to a
closed disposition class and records provider call / token estimates.  This
module never grants completion, mutation, or provider authority.

Contract rules (fail-closed):

* All counters are non-negative integers; negative counts reject.
* ``closed_deterministic``, ``abstain_review``, and ``defer_capability``
  attribute **zero** provider calls (measured zero with disposition sensor).
* ``residual_llm_authorized`` never invents zero-success when telemetry is
  absent — missing provider/token sensors are marked ``unavailable``.
* Aggregates recompute from per-attempt records; they are not completion proof.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final


# ---------------------------------------------------------------------------
# Interface / schema identities
# ---------------------------------------------------------------------------

LLM_AVOIDANCE_METRICS_INTERFACE: Final[str] = "LlmAvoidanceMetrics@1"
LLM_AVOIDANCE_METRICS_VERSION: Final[int] = 1
LLM_AVOIDANCE_METRICS_EVIDENCE: Final[str] = "wpd/llm-avoidance-metrics@1"

SCHEMA_PREFIX: Final[str] = "ipfs_accelerate_py/agent-supervisor"
LLM_AVOIDANCE_METRICS_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/llm-avoidance-metrics@1"
)
ATTEMPT_ATTRIBUTION_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/llm-avoidance-attempt-attribution@1"
)
PROVIDER_TELEMETRY_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/llm-avoidance-provider-telemetry@1"
)
LLM_AVOIDANCE_REPORT_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/llm-avoidance-metrics-report@1"
)
DISPOSITION_COUNTERS_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/llm-avoidance-disposition-counters@1"
)

MAX_TEXT_BYTES: Final[int] = 512
MAX_ATTEMPTS: Final[int] = 100_000
MAX_COUNTER: Final[int] = 10**18
MAX_ATTEMPT_INDEX: Final[int] = 2**31 - 1
BASIS_POINTS: Final[int] = 10_000

# Sensor identity used when disposition policy attributes a measured zero.
DISPOSITION_POLICY_SENSOR: Final[str] = "sensor:disposition-policy@1"
MISSING_TELEMETRY_SENSOR: Final[str] = "sensor:missing-telemetry@1"


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class AttemptDisposition(str, Enum):
    """Closed disposition classes used for LLM-avoidance attribution.

    Matches the live pre-implementation disposition vocabulary so metrics can
    join disposition events without importing worker packages.
    """

    CLOSED_DETERMINISTIC = "closed_deterministic"
    RESIDUAL_LLM_AUTHORIZED = "residual_llm_authorized"
    ABSTAIN_REVIEW = "abstain_review"
    DEFER_CAPABILITY = "defer_capability"

    @property
    def attributes_zero_provider_calls(self) -> bool:
        """Whether this disposition must attribute zero provider calls."""

        return self is not AttemptDisposition.RESIDUAL_LLM_AUTHORIZED

    @property
    def authorizes_provider(self) -> bool:
        return self is AttemptDisposition.RESIDUAL_LLM_AUTHORIZED


class TelemetryStatus(str, Enum):
    """Whether a counter was measured or is typed unavailable."""

    MEASURED = "measured"
    UNAVAILABLE = "unavailable"


class UnavailableReason(str, Enum):
    """Closed reasons for missing provider/token telemetry."""

    TELEMETRY_MISSING = "telemetry-missing"
    PROVIDER_OMITTED = "provider-omitted"
    SENSOR_ABSENT = "sensor-absent"
    COLLECTION_FAILED = "collection-failed"


_UNAVAILABLE_REASONS: Final[frozenset[str]] = frozenset(
    reason.value for reason in UnavailableReason
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LlmAvoidanceMetricsError(ValueError):
    """Raised when LLM-avoidance metrics inputs are unsafe or inconsistent."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        raise LlmAvoidanceMetricsError(f"{name} must be text")
    result = value.strip()
    if required and not result:
        raise LlmAvoidanceMetricsError(f"{name} must not be empty")
    if "\x00" in result:
        raise LlmAvoidanceMetricsError(f"{name} contains a NUL byte")
    if len(result.encode("utf-8")) > MAX_TEXT_BYTES:
        raise LlmAvoidanceMetricsError(f"{name} exceeds its {MAX_TEXT_BYTES}-byte bound")
    return result


def _nonnegative_int(
    value: Any,
    name: str,
    *,
    maximum: int = MAX_COUNTER,
) -> int:
    """Reject bools, floats, and negative counts fail-closed."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise LlmAvoidanceMetricsError(
            f"{name} must be a non-negative integer (negative counts rejected)"
        )
    if value < 0:
        raise LlmAvoidanceMetricsError(
            f"{name} must be a non-negative integer (negative counts rejected)"
        )
    if value > maximum:
        raise LlmAvoidanceMetricsError(f"{name} exceeds its maximum of {maximum}")
    return value


def _parse_disposition(value: Any) -> AttemptDisposition:
    if isinstance(value, AttemptDisposition):
        return value
    if isinstance(value, Enum):
        value = value.value
    text = _text(value, "disposition")
    try:
        return AttemptDisposition(text)
    except ValueError as exc:
        allowed = ", ".join(sorted(d.value for d in AttemptDisposition))
        raise LlmAvoidanceMetricsError(
            f"disposition must be one of {{{allowed}}}; got {text!r}"
        ) from exc


def _parse_telemetry_status(value: Any) -> TelemetryStatus:
    if isinstance(value, TelemetryStatus):
        return value
    if isinstance(value, Enum):
        value = value.value
    text = _text(value, "status")
    try:
        return TelemetryStatus(text)
    except ValueError as exc:
        raise LlmAvoidanceMetricsError(
            f"status must be 'measured' or 'unavailable'; got {text!r}"
        ) from exc


def _parse_unavailable_reason(value: Any) -> str:
    if isinstance(value, UnavailableReason):
        return value.value
    text = _text(value, "reason_code")
    if text not in _UNAVAILABLE_REASONS:
        allowed = ", ".join(sorted(_UNAVAILABLE_REASONS))
        raise LlmAvoidanceMetricsError(
            f"reason_code must be one of {{{allowed}}}; got {text!r}"
        )
    return text


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
        raise LlmAvoidanceMetricsError(
            "metrics payloads require canonical JSON values"
        ) from exc


def content_identity(value: Any) -> str:
    """Return a stable sha256 content identity for a JSON-compatible value."""

    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def closed_disposition_values() -> frozenset[str]:
    """Return the closed set of disposition wire values."""

    return frozenset(d.value for d in AttemptDisposition)


def attributes_zero_provider_calls(disposition: AttemptDisposition | str) -> bool:
    """Whether ``disposition`` must attribute zero provider calls."""

    return _parse_disposition(disposition).attributes_zero_provider_calls


def expected_provider_call_floor(
    disposition: AttemptDisposition | str,
) -> int | None:
    """Return the attributed provider-call floor for ``disposition``.

    * Non-residual dispositions → ``0`` (measured zero by policy).
    * Residual LLM → ``None`` (observation required; never invent zero-success).
    """

    parsed = _parse_disposition(disposition)
    if parsed.attributes_zero_provider_calls:
        return 0
    return None


# ---------------------------------------------------------------------------
# Provider telemetry sample
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderTelemetrySample:
    """One provider metric observation: measured count or typed unavailable.

    Unavailable is never encoded as a measured numeric zero.  A measured zero
    is only valid when a sensor (including disposition policy) produced it.
    """

    SCHEMA: ClassVar[str] = PROVIDER_TELEMETRY_SCHEMA

    metric_name: str
    status: TelemetryStatus
    sensor_id: str
    value: int = 0
    reason_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metric_name", _text(self.metric_name, "metric_name")
        )
        object.__setattr__(
            self, "status", _parse_telemetry_status(self.status)
        )
        object.__setattr__(
            self, "sensor_id", _text(self.sensor_id, "sensor_id")
        )
        object.__setattr__(
            self, "value", _nonnegative_int(self.value, "value")
        )
        reason = self.reason_code
        if reason in (None, ""):
            object.__setattr__(self, "reason_code", "")
        else:
            object.__setattr__(
                self, "reason_code", _parse_unavailable_reason(reason)
            )

        if self.status is TelemetryStatus.MEASURED:
            if self.reason_code:
                raise LlmAvoidanceMetricsError(
                    "measured sample cannot carry an unavailable reason_code"
                )
        else:
            if not self.reason_code:
                raise LlmAvoidanceMetricsError(
                    "unavailable sample requires a reason_code"
                )
            # Unavailable must never be encoded as a numeric zero-success.
            if self.value != 0:
                raise LlmAvoidanceMetricsError(
                    "unavailable sample must not encode a numeric value"
                )

    @classmethod
    def measured(
        cls,
        metric_name: str,
        value: int,
        *,
        sensor_id: str,
    ) -> "ProviderTelemetrySample":
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
    ) -> "ProviderTelemetrySample":
        return cls(
            metric_name=metric_name,
            status=TelemetryStatus.UNAVAILABLE,
            sensor_id=sensor_id,
            value=0,
            reason_code=_parse_unavailable_reason(reason),
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
            "status": self.status.value,
            "sensor_id": self.sensor_id,
        }
        if self.is_measured:
            payload["value"] = self.value
        else:
            payload["reason_code"] = self.reason_code
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderTelemetrySample":
        if not isinstance(payload, Mapping):
            raise LlmAvoidanceMetricsError("provider telemetry must be an object")
        claimed = payload.get("schema")
        if claimed is not None and claimed != cls.SCHEMA:
            raise LlmAvoidanceMetricsError(
                f"provider telemetry has foreign schema {claimed!r}"
            )
        status = _parse_telemetry_status(payload.get("status", "measured"))
        if status is TelemetryStatus.MEASURED:
            if "value" not in payload:
                raise LlmAvoidanceMetricsError(
                    "measured provider telemetry requires value"
                )
            return cls.measured(
                str(payload.get("metric_name", "")),
                payload["value"],
                sensor_id=str(payload.get("sensor_id", "")),
            )
        return cls.unavailable(
            str(payload.get("metric_name", "")),
            payload.get("reason_code", UnavailableReason.TELEMETRY_MISSING),
            sensor_id=str(payload.get("sensor_id", MISSING_TELEMETRY_SENSOR)),
        )


def _zero_by_disposition_policy(metric_name: str) -> ProviderTelemetrySample:
    return ProviderTelemetrySample.measured(
        metric_name,
        0,
        sensor_id=DISPOSITION_POLICY_SENSOR,
    )


def _unavailable_missing(metric_name: str) -> ProviderTelemetrySample:
    return ProviderTelemetrySample.unavailable(
        metric_name,
        UnavailableReason.TELEMETRY_MISSING,
        sensor_id=MISSING_TELEMETRY_SENSOR,
    )


def _resolve_provider_sample(
    *,
    metric_name: str,
    disposition: AttemptDisposition,
    observed: int | None,
    sample: ProviderTelemetrySample | Mapping[str, Any] | None,
) -> ProviderTelemetrySample:
    """Resolve one provider metric under disposition attribution rules."""

    if sample is not None:
        if isinstance(sample, Mapping):
            resolved = ProviderTelemetrySample.from_dict(sample)
        elif isinstance(sample, ProviderTelemetrySample):
            resolved = sample
        else:
            raise LlmAvoidanceMetricsError(
                f"{metric_name} sample must be ProviderTelemetrySample or mapping"
            )
        if resolved.metric_name != metric_name:
            raise LlmAvoidanceMetricsError(
                f"sample metric_name {resolved.metric_name!r} does not match "
                f"{metric_name!r}"
            )
        if disposition.attributes_zero_provider_calls and metric_name == "provider_calls":
            if resolved.is_unavailable:
                # Policy floor still attributes zero for non-residual paths.
                return _zero_by_disposition_policy(metric_name)
            if resolved.value != 0:
                raise LlmAvoidanceMetricsError(
                    f"{disposition.value} must attribute zero provider calls; "
                    f"got measured {resolved.value}"
                )
        return resolved

    if observed is not None:
        count = _nonnegative_int(observed, metric_name)
        if disposition.attributes_zero_provider_calls and metric_name == "provider_calls":
            if count != 0:
                raise LlmAvoidanceMetricsError(
                    f"{disposition.value} must attribute zero provider calls; "
                    f"got {count}"
                )
            return _zero_by_disposition_policy(metric_name)
        return ProviderTelemetrySample.measured(
            metric_name,
            count,
            sensor_id="sensor:observed@1",
        )

    # No observation supplied.
    if disposition.attributes_zero_provider_calls and metric_name == "provider_calls":
        return _zero_by_disposition_policy(metric_name)
    # Residual (or token estimates without observation): unavailable, not zero.
    return _unavailable_missing(metric_name)


# ---------------------------------------------------------------------------
# Per-attempt attribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttemptAttribution:
    """Public attribution of one implementation attempt.

    Carries disposition class plus provider call / token telemetry samples.
    Not completion authority.
    """

    SCHEMA: ClassVar[str] = ATTEMPT_ATTRIBUTION_SCHEMA

    attempt_id: str
    task_cid: str
    disposition: AttemptDisposition
    attempt_index: int
    provider_calls: ProviderTelemetrySample
    input_tokens: ProviderTelemetrySample
    output_tokens: ProviderTelemetrySample
    reused_tokens: ProviderTelemetrySample
    retry_tokens: ProviderTelemetrySample

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "attempt_id", _text(self.attempt_id, "attempt_id")
        )
        object.__setattr__(self, "task_cid", _text(self.task_cid, "task_cid"))
        object.__setattr__(
            self, "disposition", _parse_disposition(self.disposition)
        )
        object.__setattr__(
            self,
            "attempt_index",
            _nonnegative_int(
                self.attempt_index, "attempt_index", maximum=MAX_ATTEMPT_INDEX
            ),
        )
        for field_name in (
            "provider_calls",
            "input_tokens",
            "output_tokens",
            "reused_tokens",
            "retry_tokens",
        ):
            sample = getattr(self, field_name)
            if not isinstance(sample, ProviderTelemetrySample):
                raise LlmAvoidanceMetricsError(
                    f"{field_name} must be a ProviderTelemetrySample"
                )
            if sample.metric_name != field_name:
                raise LlmAvoidanceMetricsError(
                    f"{field_name} sample metric_name must be {field_name!r}"
                )

        # Disposition floor: non-residual paths attribute zero provider calls.
        if self.disposition.attributes_zero_provider_calls:
            if self.provider_calls.is_unavailable:
                raise LlmAvoidanceMetricsError(
                    f"{self.disposition.value} must attribute measured zero "
                    "provider calls, not unavailable"
                )
            if self.provider_calls.value != 0:
                raise LlmAvoidanceMetricsError(
                    f"{self.disposition.value} must attribute zero provider "
                    f"calls; got {self.provider_calls.value}"
                )

    @property
    def provider_calls_measured(self) -> int | None:
        return self.provider_calls.measured_value()

    @property
    def attributes_zero_provider_calls(self) -> bool:
        return self.disposition.attributes_zero_provider_calls

    def metric_labels(self) -> dict[str, str]:
        return {
            "disposition": self.disposition.value,
            "provider_authorized": (
                "true" if self.disposition.authorizes_provider else "false"
            ),
            "task_cid": self.task_cid,
            "attempt_id": self.attempt_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": LLM_AVOIDANCE_METRICS_VERSION,
            "attempt_id": self.attempt_id,
            "task_cid": self.task_cid,
            "disposition": self.disposition.value,
            "attempt_index": self.attempt_index,
            "provider_calls": self.provider_calls.to_dict(),
            "input_tokens": self.input_tokens.to_dict(),
            "output_tokens": self.output_tokens.to_dict(),
            "reused_tokens": self.reused_tokens.to_dict(),
            "retry_tokens": self.retry_tokens.to_dict(),
            "metric_labels": self.metric_labels(),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AttemptAttribution":
        if not isinstance(payload, Mapping):
            raise LlmAvoidanceMetricsError("attempt attribution must be an object")
        claimed = payload.get("schema")
        if claimed is not None and claimed != cls.SCHEMA:
            raise LlmAvoidanceMetricsError(
                f"attempt attribution has foreign schema {claimed!r}"
            )
        return attribute_attempt(
            attempt_id=payload.get("attempt_id", ""),
            task_cid=payload.get("task_cid", ""),
            disposition=payload.get("disposition", ""),
            attempt_index=payload.get("attempt_index", 0),
            provider_calls=payload.get("provider_calls"),
            input_tokens=payload.get("input_tokens"),
            output_tokens=payload.get("output_tokens"),
            reused_tokens=payload.get("reused_tokens"),
            retry_tokens=payload.get("retry_tokens"),
        )


def attribute_attempt(
    *,
    attempt_id: str,
    task_cid: str,
    disposition: AttemptDisposition | str,
    attempt_index: int = 0,
    provider_calls: int | ProviderTelemetrySample | Mapping[str, Any] | None = None,
    input_tokens: int | ProviderTelemetrySample | Mapping[str, Any] | None = None,
    output_tokens: int | ProviderTelemetrySample | Mapping[str, Any] | None = None,
    reused_tokens: int | ProviderTelemetrySample | Mapping[str, Any] | None = None,
    retry_tokens: int | ProviderTelemetrySample | Mapping[str, Any] | None = None,
) -> AttemptAttribution:
    """Build a validated attempt attribution under disposition rules.

    Integer observations are accepted directly.  ``None`` means telemetry was
    not collected: non-residual dispositions still attribute measured-zero
    provider calls by policy; residual and all token fields become
    ``unavailable`` rather than zero-success.
    """

    parsed = _parse_disposition(disposition)

    def _resolve(
        metric_name: str,
        observation: int | ProviderTelemetrySample | Mapping[str, Any] | None,
    ) -> ProviderTelemetrySample:
        if isinstance(observation, ProviderTelemetrySample):
            return _resolve_provider_sample(
                metric_name=metric_name,
                disposition=parsed,
                observed=None,
                sample=observation,
            )
        if isinstance(observation, Mapping):
            return _resolve_provider_sample(
                metric_name=metric_name,
                disposition=parsed,
                observed=None,
                sample=observation,
            )
        return _resolve_provider_sample(
            metric_name=metric_name,
            disposition=parsed,
            observed=observation,
            sample=None,
        )

    return AttemptAttribution(
        attempt_id=attempt_id,
        task_cid=task_cid,
        disposition=parsed,
        attempt_index=attempt_index,
        provider_calls=_resolve("provider_calls", provider_calls),
        input_tokens=_resolve("input_tokens", input_tokens),
        output_tokens=_resolve("output_tokens", output_tokens),
        reused_tokens=_resolve("reused_tokens", reused_tokens),
        retry_tokens=_resolve("retry_tokens", retry_tokens),
    )


# ---------------------------------------------------------------------------
# Aggregate counters and report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispositionCounters:
    """Additive counters for one disposition class."""

    SCHEMA: ClassVar[str] = DISPOSITION_COUNTERS_SCHEMA

    disposition: AttemptDisposition
    attempt_count: int = 0
    measured_provider_call_attempts: int = 0
    unavailable_provider_call_attempts: int = 0
    provider_calls_total: int = 0
    measured_input_token_attempts: int = 0
    unavailable_input_token_attempts: int = 0
    input_tokens_total: int = 0
    measured_output_token_attempts: int = 0
    unavailable_output_token_attempts: int = 0
    output_tokens_total: int = 0
    reused_tokens_total: int = 0
    retry_tokens_total: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "disposition", _parse_disposition(self.disposition)
        )
        for name in (
            "attempt_count",
            "measured_provider_call_attempts",
            "unavailable_provider_call_attempts",
            "provider_calls_total",
            "measured_input_token_attempts",
            "unavailable_input_token_attempts",
            "input_tokens_total",
            "measured_output_token_attempts",
            "unavailable_output_token_attempts",
            "output_tokens_total",
            "reused_tokens_total",
            "retry_tokens_total",
        ):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "disposition": self.disposition.value,
            "attempt_count": self.attempt_count,
            "measured_provider_call_attempts": self.measured_provider_call_attempts,
            "unavailable_provider_call_attempts": (
                self.unavailable_provider_call_attempts
            ),
            "provider_calls_total": self.provider_calls_total,
            "measured_input_token_attempts": self.measured_input_token_attempts,
            "unavailable_input_token_attempts": (
                self.unavailable_input_token_attempts
            ),
            "input_tokens_total": self.input_tokens_total,
            "measured_output_token_attempts": self.measured_output_token_attempts,
            "unavailable_output_token_attempts": (
                self.unavailable_output_token_attempts
            ),
            "output_tokens_total": self.output_tokens_total,
            "reused_tokens_total": self.reused_tokens_total,
            "retry_tokens_total": self.retry_tokens_total,
        }


@dataclass(frozen=True)
class LlmAvoidanceReport:
    """Aggregated LLM-avoidance metrics over a closed attempt population.

    Observability only — never completion or promotion authority.
    """

    SCHEMA: ClassVar[str] = LLM_AVOIDANCE_REPORT_SCHEMA

    total_attempts: int
    disposition_counters: tuple[DispositionCounters, ...]
    closed_deterministic_attempts: int
    residual_llm_attempts: int
    abstain_review_attempts: int
    defer_capability_attempts: int
    measured_provider_calls_total: int
    unavailable_provider_call_attempts: int
    unavailable_token_attempts: int
    measured_input_tokens_total: int
    measured_output_tokens_total: int
    llm_avoidance_ratio_bps: int | None
    llm_avoidance_ratio_status: TelemetryStatus
    llm_avoidance_ratio_reason_code: str
    attempt_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "total_attempts",
            _nonnegative_int(self.total_attempts, "total_attempts"),
        )
        for name in (
            "closed_deterministic_attempts",
            "residual_llm_attempts",
            "abstain_review_attempts",
            "defer_capability_attempts",
            "measured_provider_calls_total",
            "unavailable_provider_call_attempts",
            "unavailable_token_attempts",
            "measured_input_tokens_total",
            "measured_output_tokens_total",
        ):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "llm_avoidance_ratio_status",
            _parse_telemetry_status(self.llm_avoidance_ratio_status),
        )
        if self.llm_avoidance_ratio_status is TelemetryStatus.MEASURED:
            if self.llm_avoidance_ratio_bps is None:
                raise LlmAvoidanceMetricsError(
                    "measured llm_avoidance_ratio requires llm_avoidance_ratio_bps"
                )
            object.__setattr__(
                self,
                "llm_avoidance_ratio_bps",
                _nonnegative_int(
                    self.llm_avoidance_ratio_bps,
                    "llm_avoidance_ratio_bps",
                    maximum=BASIS_POINTS,
                ),
            )
            if self.llm_avoidance_ratio_reason_code:
                raise LlmAvoidanceMetricsError(
                    "measured ratio cannot carry an unavailable reason"
                )
        else:
            if self.llm_avoidance_ratio_bps is not None:
                raise LlmAvoidanceMetricsError(
                    "unavailable llm_avoidance_ratio must not encode a numeric value"
                )
            object.__setattr__(
                self,
                "llm_avoidance_ratio_reason_code",
                _parse_unavailable_reason(self.llm_avoidance_ratio_reason_code),
            )
        counters = tuple(self.disposition_counters)
        if len(counters) != len(AttemptDisposition):
            raise LlmAvoidanceMetricsError(
                "report must include counters for every closed disposition"
            )
        seen: set[AttemptDisposition] = set()
        for counter in counters:
            if not isinstance(counter, DispositionCounters):
                raise LlmAvoidanceMetricsError(
                    "disposition_counters entries must be DispositionCounters"
                )
            if counter.disposition in seen:
                raise LlmAvoidanceMetricsError(
                    f"duplicate disposition counters for {counter.disposition.value}"
                )
            seen.add(counter.disposition)
        object.__setattr__(self, "disposition_counters", counters)
        ids = tuple(_text(item, "attempt_id") for item in self.attempt_ids)
        if len(ids) != len(set(ids)):
            raise LlmAvoidanceMetricsError("attempt_ids must be unique")
        object.__setattr__(self, "attempt_ids", ids)

    def counters_for(
        self, disposition: AttemptDisposition | str
    ) -> DispositionCounters:
        parsed = _parse_disposition(disposition)
        for counter in self.disposition_counters:
            if counter.disposition is parsed:
                return counter
        raise LlmAvoidanceMetricsError(
            f"missing counters for disposition {parsed.value}"
        )

    def to_dict(self) -> dict[str, Any]:
        ratio: dict[str, Any] = {
            "status": self.llm_avoidance_ratio_status.value,
        }
        if self.llm_avoidance_ratio_status is TelemetryStatus.MEASURED:
            ratio["value_bps"] = self.llm_avoidance_ratio_bps
            ratio["unit"] = "basis_points"
        else:
            ratio["reason_code"] = self.llm_avoidance_ratio_reason_code

        return {
            "schema": self.SCHEMA,
            "interface": LLM_AVOIDANCE_METRICS_INTERFACE,
            "contract_version": LLM_AVOIDANCE_METRICS_VERSION,
            "evidence": LLM_AVOIDANCE_METRICS_EVIDENCE,
            "completion_authority": False,
            "total_attempts": self.total_attempts,
            "closed_deterministic_attempts": self.closed_deterministic_attempts,
            "residual_llm_attempts": self.residual_llm_attempts,
            "abstain_review_attempts": self.abstain_review_attempts,
            "defer_capability_attempts": self.defer_capability_attempts,
            "measured_provider_calls_total": self.measured_provider_calls_total,
            "unavailable_provider_call_attempts": (
                self.unavailable_provider_call_attempts
            ),
            "unavailable_token_attempts": self.unavailable_token_attempts,
            "measured_input_tokens_total": self.measured_input_tokens_total,
            "measured_output_tokens_total": self.measured_output_tokens_total,
            "llm_avoidance_ratio": ratio,
            "disposition_counters": [
                counter.to_dict() for counter in self.disposition_counters
            ],
            "attempt_ids": list(self.attempt_ids),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())


class LlmAvoidanceMetrics:
    """Collector that attributes attempts and aggregates LLM-avoidance metrics.

    Interface: :data:`LLM_AVOIDANCE_METRICS_INTERFACE`.
    """

    INTERFACE: ClassVar[str] = LLM_AVOIDANCE_METRICS_INTERFACE
    VERSION: ClassVar[int] = LLM_AVOIDANCE_METRICS_VERSION
    EVIDENCE: ClassVar[str] = LLM_AVOIDANCE_METRICS_EVIDENCE
    SCHEMA: ClassVar[str] = LLM_AVOIDANCE_METRICS_SCHEMA

    def __init__(self) -> None:
        self._attempts: dict[str, AttemptAttribution] = {}

    def __len__(self) -> int:
        return len(self._attempts)

    @property
    def attempts(self) -> tuple[AttemptAttribution, ...]:
        return tuple(
            self._attempts[key]
            for key in sorted(self._attempts)
        )

    def record(
        self,
        attribution: AttemptAttribution | Mapping[str, Any],
    ) -> AttemptAttribution:
        """Record one attempt attribution.  Duplicate attempt_id rejects."""

        if isinstance(attribution, Mapping):
            record = AttemptAttribution.from_dict(attribution)
        elif isinstance(attribution, AttemptAttribution):
            record = attribution
        else:
            raise LlmAvoidanceMetricsError(
                "attribution must be AttemptAttribution or mapping"
            )
        if len(self._attempts) >= MAX_ATTEMPTS and record.attempt_id not in self._attempts:
            raise LlmAvoidanceMetricsError(
                f"attempt population exceeds maximum of {MAX_ATTEMPTS}"
            )
        existing = self._attempts.get(record.attempt_id)
        if existing is not None:
            if existing.content_id != record.content_id:
                raise LlmAvoidanceMetricsError(
                    f"attempt_id {record.attempt_id!r} already recorded with "
                    "different attribution"
                )
            return existing
        self._attempts[record.attempt_id] = record
        return record

    def record_attempt(
        self,
        *,
        attempt_id: str,
        task_cid: str,
        disposition: AttemptDisposition | str,
        attempt_index: int = 0,
        provider_calls: int | ProviderTelemetrySample | Mapping[str, Any] | None = None,
        input_tokens: int | ProviderTelemetrySample | Mapping[str, Any] | None = None,
        output_tokens: int | ProviderTelemetrySample | Mapping[str, Any] | None = None,
        reused_tokens: int | ProviderTelemetrySample | Mapping[str, Any] | None = None,
        retry_tokens: int | ProviderTelemetrySample | Mapping[str, Any] | None = None,
    ) -> AttemptAttribution:
        """Attribute and record one attempt under disposition rules."""

        return self.record(
            attribute_attempt(
                attempt_id=attempt_id,
                task_cid=task_cid,
                disposition=disposition,
                attempt_index=attempt_index,
                provider_calls=provider_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reused_tokens=reused_tokens,
                retry_tokens=retry_tokens,
            )
        )

    def extend(
        self,
        attributions: Iterable[AttemptAttribution | Mapping[str, Any]],
    ) -> None:
        for item in attributions:
            self.record(item)

    def aggregate(self) -> LlmAvoidanceReport:
        """Recompute disposition counters and avoidance ratio from attempts."""

        by_disposition: dict[AttemptDisposition, list[AttemptAttribution]] = {
            disposition: [] for disposition in AttemptDisposition
        }
        for attempt in self._attempts.values():
            by_disposition[attempt.disposition].append(attempt)

        counters: list[DispositionCounters] = []
        measured_provider_calls_total = 0
        unavailable_provider_call_attempts = 0
        unavailable_token_attempts = 0
        measured_input_tokens_total = 0
        measured_output_tokens_total = 0

        for disposition in AttemptDisposition:
            items = by_disposition[disposition]
            measured_pc = 0
            unavailable_pc = 0
            provider_total = 0
            measured_in = 0
            unavailable_in = 0
            input_total = 0
            measured_out = 0
            unavailable_out = 0
            output_total = 0
            reused_total = 0
            retry_total = 0

            for item in items:
                if item.provider_calls.is_measured:
                    measured_pc += 1
                    provider_total = _checked_add(
                        provider_total, item.provider_calls.value, "provider_calls"
                    )
                else:
                    unavailable_pc += 1

                if item.input_tokens.is_measured:
                    measured_in += 1
                    input_total = _checked_add(
                        input_total, item.input_tokens.value, "input_tokens"
                    )
                else:
                    unavailable_in += 1
                    unavailable_token_attempts += 1

                if item.output_tokens.is_measured:
                    measured_out += 1
                    output_total = _checked_add(
                        output_total, item.output_tokens.value, "output_tokens"
                    )
                else:
                    unavailable_out += 1
                    unavailable_token_attempts += 1

                if item.reused_tokens.is_measured:
                    reused_total = _checked_add(
                        reused_total, item.reused_tokens.value, "reused_tokens"
                    )
                else:
                    unavailable_token_attempts += 1

                if item.retry_tokens.is_measured:
                    retry_total = _checked_add(
                        retry_total, item.retry_tokens.value, "retry_tokens"
                    )
                else:
                    unavailable_token_attempts += 1

            counters.append(
                DispositionCounters(
                    disposition=disposition,
                    attempt_count=len(items),
                    measured_provider_call_attempts=measured_pc,
                    unavailable_provider_call_attempts=unavailable_pc,
                    provider_calls_total=provider_total,
                    measured_input_token_attempts=measured_in,
                    unavailable_input_token_attempts=unavailable_in,
                    input_tokens_total=input_total,
                    measured_output_token_attempts=measured_out,
                    unavailable_output_token_attempts=unavailable_out,
                    output_tokens_total=output_total,
                    reused_tokens_total=reused_total,
                    retry_tokens_total=retry_total,
                )
            )
            measured_provider_calls_total = _checked_add(
                measured_provider_calls_total, provider_total, "provider_calls"
            )
            unavailable_provider_call_attempts = _checked_add(
                unavailable_provider_call_attempts,
                unavailable_pc,
                "unavailable_provider_call_attempts",
            )
            measured_input_tokens_total = _checked_add(
                measured_input_tokens_total, input_total, "input_tokens"
            )
            measured_output_tokens_total = _checked_add(
                measured_output_tokens_total, output_total, "output_tokens"
            )

        total = len(self._attempts)
        closed = len(by_disposition[AttemptDisposition.CLOSED_DETERMINISTIC])
        residual = len(by_disposition[AttemptDisposition.RESIDUAL_LLM_AUTHORIZED])
        abstain = len(by_disposition[AttemptDisposition.ABSTAIN_REVIEW])
        defer = len(by_disposition[AttemptDisposition.DEFER_CAPABILITY])

        if total > 0:
            # Avoidance = attempts closed without residual LLM authorization.
            avoided = closed + abstain + defer
            ratio_bps = (avoided * BASIS_POINTS) // total
            ratio_status = TelemetryStatus.MEASURED
            ratio_reason = ""
            ratio_value: int | None = ratio_bps
        else:
            # Empty population: unavailable, not zero-success.
            ratio_status = TelemetryStatus.UNAVAILABLE
            ratio_reason = UnavailableReason.SENSOR_ABSENT.value
            ratio_value = None

        return LlmAvoidanceReport(
            total_attempts=total,
            disposition_counters=tuple(counters),
            closed_deterministic_attempts=closed,
            residual_llm_attempts=residual,
            abstain_review_attempts=abstain,
            defer_capability_attempts=defer,
            measured_provider_calls_total=measured_provider_calls_total,
            unavailable_provider_call_attempts=unavailable_provider_call_attempts,
            unavailable_token_attempts=unavailable_token_attempts,
            measured_input_tokens_total=measured_input_tokens_total,
            measured_output_tokens_total=measured_output_tokens_total,
            llm_avoidance_ratio_bps=ratio_value,
            llm_avoidance_ratio_status=ratio_status,
            llm_avoidance_ratio_reason_code=ratio_reason,
            attempt_ids=tuple(sorted(self._attempts)),
        )

    def to_dict(self) -> dict[str, Any]:
        report = self.aggregate()
        payload = report.to_dict()
        payload["schema"] = self.SCHEMA
        payload["attempts"] = [item.to_dict() for item in self.attempts]
        return payload


def _checked_add(left: int, right: int, name: str) -> int:
    total = left + right
    if total > MAX_COUNTER:
        raise LlmAvoidanceMetricsError(f"{name} aggregate exceeds maximum")
    return total


def aggregate_attempt_attributions(
    attributions: Sequence[AttemptAttribution | Mapping[str, Any]],
) -> LlmAvoidanceReport:
    """Aggregate a closed sequence of attempt attributions."""

    metrics = LlmAvoidanceMetrics()
    metrics.extend(attributions)
    return metrics.aggregate()


__all__ = (
    "ATTEMPT_ATTRIBUTION_SCHEMA",
    "AttemptAttribution",
    "AttemptDisposition",
    "BASIS_POINTS",
    "DISPOSITION_COUNTERS_SCHEMA",
    "DISPOSITION_POLICY_SENSOR",
    "DispositionCounters",
    "LLM_AVOIDANCE_METRICS_EVIDENCE",
    "LLM_AVOIDANCE_METRICS_INTERFACE",
    "LLM_AVOIDANCE_METRICS_SCHEMA",
    "LLM_AVOIDANCE_METRICS_VERSION",
    "LLM_AVOIDANCE_REPORT_SCHEMA",
    "LlmAvoidanceMetrics",
    "LlmAvoidanceMetricsError",
    "LlmAvoidanceReport",
    "MISSING_TELEMETRY_SENSOR",
    "PROVIDER_TELEMETRY_SCHEMA",
    "ProviderTelemetrySample",
    "TelemetryStatus",
    "UnavailableReason",
    "aggregate_attempt_attributions",
    "attribute_attempt",
    "attributes_zero_provider_calls",
    "closed_disposition_values",
    "content_identity",
    "expected_provider_call_floor",
)
