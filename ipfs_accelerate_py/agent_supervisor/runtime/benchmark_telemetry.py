"""Benchmark causal-span telemetry for Planner/Doctor live runs.

This module attributes wall-clock, provider tokens, process-tree resources,
GPU, I/O, network, and cost to immutable causal spans.  It joins existing
scheduler metrics and the supervisor token ledger by span identity rather
than replacing capacity admission or inventing zero observations.

Contract rules:

* Every sample is either ``measured`` (value + unit + sensor receipt) or
  ``unavailable`` (reason code + sensor identity).  Unavailable is never
  encoded as a numeric zero.
* Measured zeros still require a sensor receipt.
* Provider usage estimates are a separate labeled-estimate record and must
  never be stored as measured samples.
* Provider responses bind only to admitted causal task spans.  Unbound
  usage, credential leakage, estimated-as-measured data, and missing causal
  identity quarantine the telemetry record instead of admitting it.
* Kill, cancel, retry, and daemon children are attributed exactly once.
* Serialized counter aggregates cannot self-certify; certification requires
  replaying the source span population that produced the measurement.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import time
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Final, Optional

from ..proof.formal_verification_contracts import CanonicalContract
from .efficiency_receipts import (
    ESTIMATOR_METHODS,
    IDENTITY_REF_RE,
    MODEL_CLASSES,
    estimated_quantity as receipt_estimated_quantity,
    measured_quantity as receipt_measured_quantity,
    observed_identity as receipt_observed_identity,
    observed_request_ids as receipt_observed_request_ids,
    unavailable as receipt_unavailable,
)


# ---------------------------------------------------------------------------
# Schemas and interface identities
# ---------------------------------------------------------------------------

BENCHMARK_TELEMETRY_CONTRACT_VERSION: Final[int] = 1
SCHEMA_VERSION: Final[int] = BENCHMARK_TELEMETRY_CONTRACT_VERSION

BENCHMARK_CAUSAL_SPAN_INTERFACE: Final[str] = "BenchmarkCausalSpan@1"
BENCHMARK_RESOURCE_MEASUREMENT_INTERFACE: Final[str] = (
    "BenchmarkResourceMeasurement@1"
)

BENCHMARK_CAUSAL_SPAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/benchmark-causal-span@1"
)
BENCHMARK_RESOURCE_MEASUREMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/benchmark-resource-measurement@1"
)
TELEMETRY_SAMPLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/benchmark-telemetry-sample@1"
)
HARDWARE_PROFILE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/benchmark-hardware-profile@1"
)
PROVIDER_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/benchmark-provider-binding@1"
)
SPAN_REPLAY_CERTIFICATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/benchmark-span-replay-certificate@1"
)
BENCHMARK_TELEMETRY_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/benchmark-telemetry-receipt@1"
)
PROVIDER_USAGE_RECORD_INTERFACE: Final[str] = "ProviderUsageRecord@1"
PROVIDER_USAGE_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/provider-usage-record@1"
)
LABELED_ESTIMATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/labeled-estimate@1"
)

MAX_TEXT_BYTES: Final[int] = 512
MAX_SAMPLES: Final[int] = 10_000
MAX_CHILDREN: Final[int] = 100_000
MAX_SPAN_DEPTH: Final[int] = 256
MAX_INTEGER: Final[int] = 10**18
GIB_BYTES: Final[int] = 1_073_741_824
MILLIONTHS: Final[int] = 1_000_000

# Metric names required by the preregistered benchmark metric registry.
CLOCK_METRIC_NAMES: Final[tuple[str, ...]] = (
    "end_to_end_makespan_seconds",
    "critical_path_seconds",
    "speedup_vs_same_arm_concurrency_one",
    "parallel_efficiency",
    "worker_occupancy_ratio",
    "queue_latency_p50_seconds",
    "queue_latency_p95_seconds",
    "ready_width",
    "admitted_width",
    "observed_width",
    "merge_conflict_serialization_seconds",
    "time_to_first_useful_counterexample_seconds",
    "accepted_criteria_per_hour",
)

TOKEN_METRIC_NAMES: Final[tuple[str, ...]] = (
    "provider_native_input_tokens",
    "provider_native_output_tokens",
    "provider_native_reused_tokens",
    "provider_native_retry_tokens",
    "provider_native_cancelled_tokens",
    "provider_native_cached_input_tokens",
    "provider_native_reasoning_tokens",
    "model_call_count",
    "tokenizer_identity",
    "context_bytes",
    "cache_reuse_count",
    "tokens_per_accepted_criterion",
    "tokens_per_proved_obligation",
    "provider_cost_per_accepted_criterion",
    "provider_cost_per_proved_obligation",
    "deterministic_llm_avoidance_ratio",
)

PROCESS_TREE_METRIC_NAMES: Final[tuple[str, ...]] = (
    "user_cpu_seconds",
    "system_cpu_seconds",
    "total_cpu_seconds",
    "peak_rss_bytes",
    "memory_gib_seconds",
    "read_bytes",
    "write_bytes",
    "disk_artifact_growth_bytes",
    "peak_process_count",
    "network_rx_bytes",
    "network_tx_bytes",
    "provider_quota_units",
    "provider_cost_microusd",
    "energy_joules_optional",
)

GPU_METRIC_NAMES: Final[tuple[str, ...]] = (
    "gpu_utilization_time_weighted_ratio",
    "peak_vram_bytes",
    "gpu_seconds",
    "gpu_energy_joules_optional",
)

# Units used in measured samples (integer-only contracts).
UNIT_SECONDS_MILLIONTHS: Final[str] = "seconds_millionths"
UNIT_COUNT: Final[str] = "count"
UNIT_BYTES: Final[str] = "bytes"
UNIT_TOKENS: Final[str] = "tokens"
UNIT_MICROUSD: Final[str] = "microusd"
UNIT_RATIO_MILLIONTHS: Final[str] = "ratio_millionths"
UNIT_GIB_SECONDS_MILLIONTHS: Final[str] = "gib_seconds_millionths"
UNIT_JOULES_MILLIONTHS: Final[str] = "joules_millionths"
UNIT_IDENTITY: Final[str] = "identity_digest"
UNIT_QUOTA: Final[str] = "quota_units"


class BenchmarkTelemetryError(ValueError):
    """Benchmark telemetry is malformed, duplicated, or detached."""


class SampleStatus(str, Enum):
    MEASURED = "measured"
    UNAVAILABLE = "unavailable"


class UnavailableReason(str, Enum):
    SENSOR_ABSENT = "sensor-absent"
    PERMISSION_DENIED = "permission-denied"
    HARDWARE_ABSENT = "hardware-absent"
    PROVIDER_OMITTED = "provider-omitted"
    COLLECTION_FAILED = "collection-failed"
    NOT_REPORTED = "not-reported"
    UNSAFE_REQUEST_ID = "unsafe-request-id"


class SpanKind(str, Enum):
    RUN = "run"
    CASE = "case"
    ARM = "arm"
    TASK = "task"
    ATTEMPT = "attempt"
    PROCESS = "process"
    PROVIDER_CALL = "provider_call"
    MERGE = "merge"
    QUEUE = "queue"
    DAEMON = "daemon"
    RETRY = "retry"
    CANCEL = "cancel"
    SENSOR = "sensor"


class AttributionRole(str, Enum):
    ROOT = "root"
    WORKER = "worker"
    DAEMON_CHILD = "daemon_child"
    RETRY = "retry"
    CANCELLED = "cancelled"
    KILLED = "killed"
    PROVIDER = "provider"
    ORACLE = "oracle"
    TELEMETRY = "telemetry"


# ---------------------------------------------------------------------------
# Validation helpers (integer-only, content-addressed)
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise BenchmarkTelemetryError(f"{name} must be text")
    result = value.strip()
    if required and not result:
        raise BenchmarkTelemetryError(f"{name} must not be empty")
    if "\x00" in result or len(result.encode("utf-8")) > MAX_TEXT_BYTES:
        raise BenchmarkTelemetryError(f"{name} is unsafe or too large")
    return result


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_INTEGER,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkTelemetryError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise BenchmarkTelemetryError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(raw)
    except (TypeError, ValueError) as exc:
        raise BenchmarkTelemetryError(
            f"{name} is not a supported {enum_type.__name__}"
        ) from exc


def _closed(
    payload: Mapping[str, Any],
    *,
    schema: str,
    allowed: set[str],
    name: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise BenchmarkTelemetryError(f"{name} must be an object")
    claimed = payload.get("schema")
    if claimed is not None and claimed != schema:
        raise BenchmarkTelemetryError(f"{name} has foreign schema")
    unknown = set(payload) - allowed
    if unknown:
        raise BenchmarkTelemetryError(
            f"{name} contains unknown fields: {sorted(unknown)}"
        )


def _claim(payload: Mapping[str, Any], actual: str, *names: str) -> None:
    for name in names:
        if name in payload and payload[name] != actual:
            raise BenchmarkTelemetryError(
                f"{name} does not match content identity"
            )


def _sensor_id(*parts: str) -> str:
    body = json.dumps(list(parts), separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f"sensor:sha256:{digest}"


def _identity_digest(value: str) -> int:
    """Map free-form identity text to a stable measured-integer digest.

    An 8-byte SHA-256 prefix is a uint64 and routinely exceeds ``MAX_INTEGER``.
    Fold into the measured-integer domain so identity samples cannot fail
    closed as ``BenchmarkTelemetryError`` after a successful provider map.
    """

    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (MAX_INTEGER + 1)


def seconds_to_millionths(seconds: float | int) -> int:
    """Convert wall/CPU seconds to integer millionths without float payloads."""

    if isinstance(seconds, bool):
        raise BenchmarkTelemetryError("seconds must be numeric")
    if isinstance(seconds, int):
        return _integer(seconds * MILLIONTHS, "seconds_millionths")
    if not isinstance(seconds, (int, float)):
        raise BenchmarkTelemetryError("seconds must be numeric")
    if seconds < 0:
        raise BenchmarkTelemetryError("seconds must be non-negative")
    return _integer(int(round(float(seconds) * MILLIONTHS)), "seconds_millionths")


def millionths_to_seconds_int(millionths: int) -> int:
    """Floor millionths back to whole seconds for coarse aggregates."""

    return _integer(millionths, "millionths") // MILLIONTHS


# ---------------------------------------------------------------------------
# Core contracts
# ---------------------------------------------------------------------------


class _TelemetryContract(CanonicalContract):
    @property
    def schema_version(self) -> int:
        return BENCHMARK_TELEMETRY_CONTRACT_VERSION


@dataclass(frozen=True)
class TelemetrySample(_TelemetryContract):
    """One metric observation: measured with receipt, or unavailable."""

    SCHEMA: ClassVar[str] = TELEMETRY_SAMPLE_SCHEMA

    metric_name: str
    status: SampleStatus
    sensor_id: str
    unit: str = ""
    value: int = 0
    reason_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metric_name", _text(self.metric_name, "metric_name")
        )
        object.__setattr__(
            self, "status", _enum(self.status, SampleStatus, "status")
        )
        object.__setattr__(
            self, "sensor_id", _text(self.sensor_id, "sensor_id")
        )
        object.__setattr__(
            self, "unit", _text(self.unit, "unit", required=False)
        )
        object.__setattr__(
            self, "value", _integer(self.value, "value", minimum=0)
        )
        object.__setattr__(
            self,
            "reason_code",
            _text(self.reason_code, "reason_code", required=False),
        )
        if self.status is SampleStatus.MEASURED:
            if not self.unit:
                raise BenchmarkTelemetryError(
                    "measured sample requires a unit"
                )
            if self.reason_code:
                raise BenchmarkTelemetryError(
                    "measured sample cannot carry an unavailable reason"
                )
        else:
            if not self.reason_code:
                raise BenchmarkTelemetryError(
                    "unavailable sample requires a reason_code"
                )
            try:
                UnavailableReason(self.reason_code)
            except ValueError as exc:
                raise BenchmarkTelemetryError(
                    "reason_code is not a supported unavailable reason"
                ) from exc
            # Unavailable must never be encoded as a numeric zero observation.
            if self.value != 0 or self.unit:
                raise BenchmarkTelemetryError(
                    "unavailable sample must not encode a numeric value or unit"
                )

    @classmethod
    def measured(
        cls,
        metric_name: str,
        value: int,
        *,
        unit: str,
        sensor_id: str,
    ) -> "TelemetrySample":
        return cls(
            metric_name=metric_name,
            status=SampleStatus.MEASURED,
            sensor_id=sensor_id,
            unit=unit,
            value=_integer(value, "value"),
        )

    @classmethod
    def unavailable(
        cls,
        metric_name: str,
        reason: UnavailableReason | str,
        *,
        sensor_id: str | None = None,
    ) -> "TelemetrySample":
        reason_code = (
            reason.value if isinstance(reason, UnavailableReason) else str(reason)
        )
        return cls(
            metric_name=metric_name,
            status=SampleStatus.UNAVAILABLE,
            sensor_id=sensor_id
            or _sensor_id("unavailable", metric_name, reason_code),
            reason_code=reason_code,
        )

    def _payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract_version": BENCHMARK_TELEMETRY_CONTRACT_VERSION,
            "metric_name": self.metric_name,
            "status": self.status.value,
            "sensor_id": self.sensor_id,
        }
        if self.status is SampleStatus.MEASURED:
            payload["unit"] = self.unit
            payload["value"] = self.value
        else:
            payload["reason_code"] = self.reason_code
        return payload

    def to_envelope(self) -> dict[str, Any]:
        """Telemetry contract sample envelope (measured | unavailable)."""

        if self.status is SampleStatus.MEASURED:
            return {
                "status": SampleStatus.MEASURED.value,
                "value": self.value,
                "unit": self.unit,
                "sensor_id": self.sensor_id,
            }
        return {
            "status": SampleStatus.UNAVAILABLE.value,
            "reason_code": self.reason_code,
            "sensor_id": self.sensor_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TelemetrySample":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "metric_name",
            "status",
            "sensor_id",
            "unit",
            "value",
            "reason_code",
            "content_id",
        }
        _closed(
            payload,
            schema=cls.SCHEMA,
            allowed=allowed,
            name="telemetry sample",
        )
        status = _enum(payload.get("status", ""), SampleStatus, "status")
        if status is SampleStatus.MEASURED:
            result = cls.measured(
                str(payload.get("metric_name", "")),
                int(payload.get("value", 0)),
                unit=str(payload.get("unit", "")),
                sensor_id=str(payload.get("sensor_id", "")),
            )
        else:
            result = cls.unavailable(
                str(payload.get("metric_name", "")),
                str(payload.get("reason_code", "")),
                sensor_id=str(payload.get("sensor_id", "")),
            )
        _claim(payload, result.content_id, "content_id")
        return result


@dataclass(frozen=True)
class BenchmarkHardwareProfile(_TelemetryContract):
    """Host/hardware binding sealed into every resource measurement."""

    SCHEMA: ClassVar[str] = HARDWARE_PROFILE_SCHEMA

    profile_id: str
    hostname_alias: str
    cpu_model_id: str
    cpu_count: int
    memory_bytes: int
    accelerator_present: bool
    accelerator_model_id: str = ""
    accelerator_count: int = 0
    platform: str = ""

    def __post_init__(self) -> None:
        for name in ("profile_id", "hostname_alias", "cpu_model_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "cpu_count", _integer(self.cpu_count, "cpu_count", minimum=1)
        )
        object.__setattr__(
            self,
            "memory_bytes",
            _integer(self.memory_bytes, "memory_bytes", minimum=0),
        )
        if not isinstance(self.accelerator_present, bool):
            raise BenchmarkTelemetryError(
                "accelerator_present must be a boolean"
            )
        object.__setattr__(
            self,
            "accelerator_model_id",
            _text(
                self.accelerator_model_id,
                "accelerator_model_id",
                required=self.accelerator_present,
            ),
        )
        object.__setattr__(
            self,
            "accelerator_count",
            _integer(
                self.accelerator_count,
                "accelerator_count",
                minimum=1 if self.accelerator_present else 0,
            ),
        )
        object.__setattr__(
            self,
            "platform",
            _text(self.platform, "platform", required=False),
        )
        if not self.accelerator_present and (
            self.accelerator_count or self.accelerator_model_id
        ):
            raise BenchmarkTelemetryError(
                "accelerator fields require accelerator_present"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": BENCHMARK_TELEMETRY_CONTRACT_VERSION,
            "profile_id": self.profile_id,
            "hostname_alias": self.hostname_alias,
            "cpu_model_id": self.cpu_model_id,
            "cpu_count": self.cpu_count,
            "memory_bytes": self.memory_bytes,
            "accelerator_present": self.accelerator_present,
            "accelerator_model_id": self.accelerator_model_id,
            "accelerator_count": self.accelerator_count,
            "platform": self.platform,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkHardwareProfile":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "profile_id",
            "hostname_alias",
            "cpu_model_id",
            "cpu_count",
            "memory_bytes",
            "accelerator_present",
            "accelerator_model_id",
            "accelerator_count",
            "platform",
            "content_id",
        }
        _closed(
            payload, schema=cls.SCHEMA, allowed=allowed, name="hardware profile"
        )
        result = cls(
            profile_id=payload.get("profile_id", ""),
            hostname_alias=payload.get("hostname_alias", ""),
            cpu_model_id=payload.get("cpu_model_id", ""),
            cpu_count=payload.get("cpu_count", 0),
            memory_bytes=payload.get("memory_bytes", 0),
            accelerator_present=bool(payload.get("accelerator_present", False)),
            accelerator_model_id=payload.get("accelerator_model_id", ""),
            accelerator_count=payload.get("accelerator_count", 0),
            platform=payload.get("platform", ""),
        )
        _claim(payload, result.content_id, "content_id")
        return result


@dataclass(frozen=True)
class BenchmarkProviderBinding(_TelemetryContract):
    """Tokenizer/model/endpoint binding for provider-native accounting."""

    SCHEMA: ClassVar[str] = PROVIDER_BINDING_SCHEMA

    provider_id: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    endpoint_id: str
    max_context_tokens: int = 0

    def __post_init__(self) -> None:
        for name in (
            "provider_id",
            "model_id",
            "model_revision",
            "tokenizer_id",
            "endpoint_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        object.__setattr__(
            self,
            "max_context_tokens",
            _integer(self.max_context_tokens, "max_context_tokens"),
        )

    @property
    def is_bound(self) -> bool:
        return bool(
            self.provider_id
            and self.model_id
            and self.tokenizer_id
            and self.endpoint_id
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": BENCHMARK_TELEMETRY_CONTRACT_VERSION,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_id": self.tokenizer_id,
            "endpoint_id": self.endpoint_id,
            "max_context_tokens": self.max_context_tokens,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkProviderBinding":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "provider_id",
            "model_id",
            "model_revision",
            "tokenizer_id",
            "endpoint_id",
            "max_context_tokens",
            "content_id",
        }
        _closed(
            payload,
            schema=cls.SCHEMA,
            allowed=allowed,
            name="provider binding",
        )
        result = cls(
            provider_id=payload.get("provider_id", ""),
            model_id=payload.get("model_id", ""),
            model_revision=payload.get("model_revision", ""),
            tokenizer_id=payload.get("tokenizer_id", ""),
            endpoint_id=payload.get("endpoint_id", ""),
            max_context_tokens=payload.get("max_context_tokens", 0),
        )
        _claim(payload, result.content_id, "content_id")
        return result


@dataclass(frozen=True)
class BenchmarkCausalSpan(_TelemetryContract):
    """One causal unit of benchmark work with ancestry and identity bindings.

    Interface: BenchmarkCausalSpan@1
    """

    SCHEMA: ClassVar[str] = BENCHMARK_CAUSAL_SPAN_SCHEMA
    INTERFACE: ClassVar[str] = BENCHMARK_CAUSAL_SPAN_INTERFACE

    span_id: str
    kind: SpanKind
    run_id: str
    case_id: str
    arm_id: str
    task_id: str
    attempt: int
    process_id: str
    parent_span_id: str = ""
    ancestry: tuple[str, ...] = ()
    role: AttributionRole = AttributionRole.ROOT
    provider: BenchmarkProviderBinding | None = None
    hardware: BenchmarkHardwareProfile | None = None
    started_at_mono_ns: int = 0
    finished_at_mono_ns: int = 0
    monotonic_clock: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "span_id", _text(self.span_id, "span_id"))
        object.__setattr__(self, "kind", _enum(self.kind, SpanKind, "kind"))
        for name in ("run_id", "case_id", "arm_id", "task_id", "process_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        object.__setattr__(
            self,
            "attempt",
            _integer(self.attempt, "attempt", minimum=0, maximum=100_000),
        )
        object.__setattr__(
            self,
            "parent_span_id",
            _text(self.parent_span_id, "parent_span_id", required=False),
        )
        ancestry = tuple(
            _text(item, "ancestry_item") for item in (self.ancestry or ())
        )
        if len(ancestry) > MAX_SPAN_DEPTH:
            raise BenchmarkTelemetryError("span ancestry exceeds depth bound")
        if len(set(ancestry)) != len(ancestry):
            raise BenchmarkTelemetryError("span ancestry contains duplicates")
        if self.span_id in ancestry:
            raise BenchmarkTelemetryError("span cannot ancestor itself")
        if self.parent_span_id:
            if not ancestry or ancestry[-1] != self.parent_span_id:
                raise BenchmarkTelemetryError(
                    "parent_span_id must be the terminal ancestry entry"
                )
        elif ancestry:
            raise BenchmarkTelemetryError(
                "root spans cannot declare ancestry"
            )
        object.__setattr__(self, "ancestry", ancestry)
        object.__setattr__(
            self, "role", _enum(self.role, AttributionRole, "role")
        )
        provider = self.provider
        if isinstance(provider, Mapping):
            provider = BenchmarkProviderBinding.from_dict(provider)
        if provider is not None and not isinstance(
            provider, BenchmarkProviderBinding
        ):
            raise BenchmarkTelemetryError(
                "provider must be BenchmarkProviderBinding"
            )
        object.__setattr__(self, "provider", provider)
        hardware = self.hardware
        if isinstance(hardware, Mapping):
            hardware = BenchmarkHardwareProfile.from_dict(hardware)
        if hardware is not None and not isinstance(
            hardware, BenchmarkHardwareProfile
        ):
            raise BenchmarkTelemetryError(
                "hardware must be BenchmarkHardwareProfile"
            )
        object.__setattr__(self, "hardware", hardware)
        object.__setattr__(
            self,
            "started_at_mono_ns",
            _integer(self.started_at_mono_ns, "started_at_mono_ns"),
        )
        object.__setattr__(
            self,
            "finished_at_mono_ns",
            _integer(self.finished_at_mono_ns, "finished_at_mono_ns"),
        )
        if (
            self.finished_at_mono_ns
            and self.started_at_mono_ns
            and self.finished_at_mono_ns < self.started_at_mono_ns
        ):
            raise BenchmarkTelemetryError(
                "finished_at_mono_ns precedes started_at_mono_ns"
            )
        if not isinstance(self.monotonic_clock, bool):
            raise BenchmarkTelemetryError("monotonic_clock must be a boolean")
        if not self.monotonic_clock:
            raise BenchmarkTelemetryError(
                "benchmark spans require a monotonic clock source"
            )

    @property
    def duration_ns(self) -> int:
        if not self.finished_at_mono_ns or not self.started_at_mono_ns:
            return 0
        return self.finished_at_mono_ns - self.started_at_mono_ns

    @property
    def duration_seconds_millionths(self) -> int:
        return self.duration_ns // 1_000

    def child(
        self,
        *,
        span_id: str,
        kind: SpanKind,
        role: AttributionRole = AttributionRole.WORKER,
        task_id: str | None = None,
        attempt: int | None = None,
        process_id: str | None = None,
        started_at_mono_ns: int = 0,
        finished_at_mono_ns: int = 0,
        provider: BenchmarkProviderBinding | None = None,
    ) -> "BenchmarkCausalSpan":
        """Derive a child span that extends this span's ancestry."""

        return BenchmarkCausalSpan(
            span_id=span_id,
            kind=kind,
            run_id=self.run_id,
            case_id=self.case_id,
            arm_id=self.arm_id,
            task_id=self.task_id if task_id is None else task_id,
            attempt=self.attempt if attempt is None else attempt,
            process_id=self.process_id if process_id is None else process_id,
            parent_span_id=self.span_id,
            ancestry=self.ancestry + (self.span_id,),
            role=role,
            provider=provider if provider is not None else self.provider,
            hardware=self.hardware,
            started_at_mono_ns=started_at_mono_ns,
            finished_at_mono_ns=finished_at_mono_ns,
            monotonic_clock=True,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": BENCHMARK_TELEMETRY_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "span_id": self.span_id,
            "kind": self.kind.value,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "arm_id": self.arm_id,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "process_id": self.process_id,
            "parent_span_id": self.parent_span_id,
            "ancestry": list(self.ancestry),
            "role": self.role.value,
            "provider": (
                None if self.provider is None else self.provider.to_record()
            ),
            "hardware": (
                None if self.hardware is None else self.hardware.to_record()
            ),
            "started_at_mono_ns": self.started_at_mono_ns,
            "finished_at_mono_ns": self.finished_at_mono_ns,
            "monotonic_clock": self.monotonic_clock,
            "duration_ns": self.duration_ns,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkCausalSpan":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "interface",
            "span_id",
            "kind",
            "run_id",
            "case_id",
            "arm_id",
            "task_id",
            "attempt",
            "process_id",
            "parent_span_id",
            "ancestry",
            "role",
            "provider",
            "hardware",
            "started_at_mono_ns",
            "finished_at_mono_ns",
            "monotonic_clock",
            "duration_ns",
            "content_id",
        }
        _closed(
            payload, schema=cls.SCHEMA, allowed=allowed, name="causal span"
        )
        result = cls(
            span_id=payload.get("span_id", ""),
            kind=payload.get("kind", ""),
            run_id=payload.get("run_id", ""),
            case_id=payload.get("case_id", ""),
            arm_id=payload.get("arm_id", ""),
            task_id=payload.get("task_id", ""),
            attempt=payload.get("attempt", 0),
            process_id=payload.get("process_id", ""),
            parent_span_id=payload.get("parent_span_id", ""),
            ancestry=tuple(payload.get("ancestry") or ()),
            role=payload.get("role", AttributionRole.ROOT),
            provider=payload.get("provider"),
            hardware=payload.get("hardware"),
            started_at_mono_ns=payload.get("started_at_mono_ns", 0),
            finished_at_mono_ns=payload.get("finished_at_mono_ns", 0),
            monotonic_clock=bool(payload.get("monotonic_clock", True)),
        )
        if payload.get("interface", result.INTERFACE) != result.INTERFACE:
            raise BenchmarkTelemetryError("span interface mismatch")
        if payload.get("duration_ns", result.duration_ns) != result.duration_ns:
            raise BenchmarkTelemetryError("duration_ns does not match span bounds")
        _claim(payload, result.content_id, "content_id")
        return result


@dataclass(frozen=True)
class BenchmarkResourceMeasurement(_TelemetryContract):
    """Process-tree, provider, GPU, I/O and cost samples bound to one span.

    Interface: BenchmarkResourceMeasurement@1
    """

    SCHEMA: ClassVar[str] = BENCHMARK_RESOURCE_MEASUREMENT_SCHEMA
    INTERFACE: ClassVar[str] = BENCHMARK_RESOURCE_MEASUREMENT_INTERFACE

    measurement_id: str
    span: BenchmarkCausalSpan
    samples: tuple[TelemetrySample, ...]
    attributed_process_ids: tuple[str, ...] = ()
    attributed_measurement_ids: tuple[str, ...] = ()
    source_span_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "measurement_id", _text(self.measurement_id, "measurement_id")
        )
        span = self.span
        if isinstance(span, Mapping):
            span = BenchmarkCausalSpan.from_dict(span)
        if not isinstance(span, BenchmarkCausalSpan):
            raise BenchmarkTelemetryError("span must be BenchmarkCausalSpan")
        object.__setattr__(self, "span", span)

        samples: list[TelemetrySample] = []
        for item in self.samples or ():
            if isinstance(item, Mapping):
                samples.append(TelemetrySample.from_dict(item))
            elif isinstance(item, TelemetrySample):
                samples.append(item)
            else:
                raise BenchmarkTelemetryError(
                    "samples must be TelemetrySample records"
                )
        if len(samples) > MAX_SAMPLES:
            raise BenchmarkTelemetryError("sample population exceeds bound")
        names = [item.metric_name for item in samples]
        if len(names) != len(set(names)):
            raise BenchmarkTelemetryError(
                "measurement contains duplicate metric names"
            )
        object.__setattr__(self, "samples", tuple(samples))

        process_ids = tuple(
            _text(item, "process_id")
            for item in (self.attributed_process_ids or ())
        )
        if len(process_ids) != len(set(process_ids)):
            raise BenchmarkTelemetryError(
                "process attribution contains duplicates"
            )
        if len(process_ids) > MAX_CHILDREN:
            raise BenchmarkTelemetryError("process attribution exceeds bound")
        object.__setattr__(self, "attributed_process_ids", process_ids)

        measurement_ids = tuple(
            _text(item, "measurement_id")
            for item in (self.attributed_measurement_ids or ())
        )
        if len(measurement_ids) != len(set(measurement_ids)):
            raise BenchmarkTelemetryError(
                "nested measurement attribution contains duplicates"
            )
        object.__setattr__(self, "attributed_measurement_ids", measurement_ids)

        source_span_ids = tuple(
            _text(item, "source_span_id")
            for item in (self.source_span_ids or (span.span_id,))
        )
        if span.span_id not in source_span_ids:
            raise BenchmarkTelemetryError(
                "source_span_ids must include the bound span"
            )
        if len(source_span_ids) != len(set(source_span_ids)):
            raise BenchmarkTelemetryError(
                "source_span_ids contains duplicates"
            )
        object.__setattr__(self, "source_span_ids", source_span_ids)

    def sample(self, metric_name: str) -> TelemetrySample | None:
        for item in self.samples:
            if item.metric_name == metric_name:
                return item
        return None

    def require_sample(self, metric_name: str) -> TelemetrySample:
        item = self.sample(metric_name)
        if item is None:
            raise BenchmarkTelemetryError(
                f"measurement is missing metric {metric_name!r}"
            )
        return item

    def measured_value(self, metric_name: str) -> int | None:
        item = self.sample(metric_name)
        if item is None or item.status is not SampleStatus.MEASURED:
            return None
        return item.value

    def to_envelopes(self) -> dict[str, dict[str, Any]]:
        return {item.metric_name: item.to_envelope() for item in self.samples}

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": BENCHMARK_TELEMETRY_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "measurement_id": self.measurement_id,
            "span": self.span.to_record(),
            "samples": [item.to_record() for item in self.samples],
            "attributed_process_ids": list(self.attributed_process_ids),
            "attributed_measurement_ids": list(self.attributed_measurement_ids),
            "source_span_ids": list(self.source_span_ids),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "BenchmarkResourceMeasurement":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "interface",
            "measurement_id",
            "span",
            "samples",
            "attributed_process_ids",
            "attributed_measurement_ids",
            "source_span_ids",
            "content_id",
        }
        _closed(
            payload,
            schema=cls.SCHEMA,
            allowed=allowed,
            name="resource measurement",
        )
        result = cls(
            measurement_id=payload.get("measurement_id", ""),
            span=payload.get("span", {}),
            samples=tuple(payload.get("samples") or ()),
            attributed_process_ids=tuple(
                payload.get("attributed_process_ids") or ()
            ),
            attributed_measurement_ids=tuple(
                payload.get("attributed_measurement_ids") or ()
            ),
            source_span_ids=tuple(payload.get("source_span_ids") or ()),
        )
        if payload.get("interface", result.INTERFACE) != result.INTERFACE:
            raise BenchmarkTelemetryError("measurement interface mismatch")
        _claim(payload, result.content_id, "content_id")
        return result


@dataclass(frozen=True)
class SpanReplayCertificate(_TelemetryContract):
    """Proves a measurement was derived by replaying source spans.

    Serialized counter aggregates alone cannot produce a valid certificate.
    """

    SCHEMA: ClassVar[str] = SPAN_REPLAY_CERTIFICATE_SCHEMA

    measurement_id: str
    measurement_content_id: str
    source_span_content_ids: tuple[str, ...]
    replay_digest: str
    certified: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "measurement_id", _text(self.measurement_id, "measurement_id")
        )
        object.__setattr__(
            self,
            "measurement_content_id",
            _text(self.measurement_content_id, "measurement_content_id"),
        )
        spans = tuple(
            _text(item, "source_span_content_id")
            for item in (self.source_span_content_ids or ())
        )
        if not spans:
            raise BenchmarkTelemetryError(
                "replay certificate requires at least one source span"
            )
        if len(spans) != len(set(spans)):
            raise BenchmarkTelemetryError(
                "source_span_content_ids contains duplicates"
            )
        object.__setattr__(self, "source_span_content_ids", spans)
        object.__setattr__(
            self, "replay_digest", _text(self.replay_digest, "replay_digest")
        )
        if not isinstance(self.certified, bool):
            raise BenchmarkTelemetryError("certified must be a boolean")
        expected = _replay_digest(
            self.measurement_content_id, self.source_span_content_ids
        )
        if self.certified and self.replay_digest != expected:
            raise BenchmarkTelemetryError(
                "certified replay digest does not match source spans"
            )
        if not self.certified and self.replay_digest == expected:
            # Refuse to mark an exactly matching replay as uncertified when
            # the operator supplies the correct digest; callers must omit the
            # digest or leave certified=False only when spans are missing.
            pass

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": BENCHMARK_TELEMETRY_CONTRACT_VERSION,
            "measurement_id": self.measurement_id,
            "measurement_content_id": self.measurement_content_id,
            "source_span_content_ids": list(self.source_span_content_ids),
            "replay_digest": self.replay_digest,
            "certified": self.certified,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpanReplayCertificate":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "measurement_id",
            "measurement_content_id",
            "source_span_content_ids",
            "replay_digest",
            "certified",
            "content_id",
        }
        _closed(
            payload,
            schema=cls.SCHEMA,
            allowed=allowed,
            name="span replay certificate",
        )
        result = cls(
            measurement_id=payload.get("measurement_id", ""),
            measurement_content_id=payload.get("measurement_content_id", ""),
            source_span_content_ids=tuple(
                payload.get("source_span_content_ids") or ()
            ),
            replay_digest=payload.get("replay_digest", ""),
            certified=bool(payload.get("certified", False)),
        )
        _claim(payload, result.content_id, "content_id")
        return result


def _replay_digest(
    measurement_content_id: str, source_span_content_ids: Sequence[str]
) -> str:
    body = json.dumps(
        {
            "measurement_content_id": measurement_content_id,
            "source_span_content_ids": list(source_span_content_ids),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def certify_measurement_from_source_spans(
    measurement: BenchmarkResourceMeasurement,
    source_spans: Sequence[BenchmarkCausalSpan | Mapping[str, Any]],
) -> SpanReplayCertificate:
    """Certify a measurement only when its source spans are replayed.

    Passing only serialized counter aggregates (or a measurement without the
    spans that produced it) cannot yield ``certified=True``.
    """

    if not isinstance(measurement, BenchmarkResourceMeasurement):
        raise BenchmarkTelemetryError(
            "measurement must be BenchmarkResourceMeasurement"
        )
    spans: list[BenchmarkCausalSpan] = []
    for item in source_spans:
        if isinstance(item, Mapping):
            spans.append(BenchmarkCausalSpan.from_dict(item))
        elif isinstance(item, BenchmarkCausalSpan):
            spans.append(item)
        else:
            raise BenchmarkTelemetryError(
                "source spans must be BenchmarkCausalSpan records"
            )
    span_ids = {item.span_id for item in spans}
    required = set(measurement.source_span_ids)
    if not required.issubset(span_ids):
        return SpanReplayCertificate(
            measurement_id=measurement.measurement_id,
            measurement_content_id=measurement.content_id,
            source_span_content_ids=tuple(
                sorted(item.content_id for item in spans)
            )
            or (measurement.span.content_id,),
            replay_digest="uncertified:missing-source-spans",
            certified=False,
        )
    # Bound span identity must match the replayed span population.
    by_id = {item.span_id: item for item in spans}
    bound = by_id[measurement.span.span_id]
    if bound.content_id != measurement.span.content_id:
        return SpanReplayCertificate(
            measurement_id=measurement.measurement_id,
            measurement_content_id=measurement.content_id,
            source_span_content_ids=tuple(
                sorted(item.content_id for item in spans)
            ),
            replay_digest="uncertified:span-identity-mismatch",
            certified=False,
        )
    source_ids = tuple(
        sorted(by_id[span_id].content_id for span_id in required)
    )
    digest = _replay_digest(measurement.content_id, source_ids)
    return SpanReplayCertificate(
        measurement_id=measurement.measurement_id,
        measurement_content_id=measurement.content_id,
        source_span_content_ids=source_ids,
        replay_digest=digest,
        certified=True,
    )


def reject_self_certified_counters(
    *,
    serialized_counters: Mapping[str, Any],
    source_spans: Sequence[Any] | None = None,
) -> None:
    """Fail closed when only serialized counters are offered as evidence."""

    if source_spans:
        return
    if not serialized_counters:
        return
    raise BenchmarkTelemetryError(
        "serialized counters cannot self-certify without source span replay"
    )


# ---------------------------------------------------------------------------
# Exactly-once attribution session
# ---------------------------------------------------------------------------


class BenchmarkTelemetrySession:
    """Mutable collector that attributes children and measurements once."""

    def __init__(self, root: BenchmarkCausalSpan) -> None:
        if not isinstance(root, BenchmarkCausalSpan):
            raise BenchmarkTelemetryError("root must be BenchmarkCausalSpan")
        self._root = root
        self._spans: dict[str, BenchmarkCausalSpan] = {root.span_id: root}
        self._process_owners: dict[str, str] = {}
        self._measurement_owners: dict[str, str] = {}
        self._measurements: dict[str, BenchmarkResourceMeasurement] = {}
        if root.process_id:
            self._process_owners[root.process_id] = root.span_id

    @property
    def root(self) -> BenchmarkCausalSpan:
        return self._root

    @property
    def spans(self) -> tuple[BenchmarkCausalSpan, ...]:
        return tuple(self._spans[key] for key in sorted(self._spans))

    @property
    def measurements(self) -> tuple[BenchmarkResourceMeasurement, ...]:
        return tuple(
            self._measurements[key] for key in sorted(self._measurements)
        )

    def register_span(self, span: BenchmarkCausalSpan) -> BenchmarkCausalSpan:
        if span.span_id in self._spans:
            existing = self._spans[span.span_id]
            if existing.content_id != span.content_id:
                raise BenchmarkTelemetryError(
                    f"span {span.span_id!r} is already registered with a "
                    "different identity"
                )
            return existing
        if span.parent_span_id and span.parent_span_id not in self._spans:
            raise BenchmarkTelemetryError(
                f"parent span {span.parent_span_id!r} is not registered"
            )
        self._spans[span.span_id] = span
        if span.process_id:
            self.attribute_process(span.process_id, span.span_id)
        return span

    def attribute_process(self, process_id: str, span_id: str) -> None:
        """Attribute a process (including daemon children) exactly once."""

        process_id = _text(process_id, "process_id")
        span_id = _text(span_id, "span_id")
        if span_id not in self._spans:
            raise BenchmarkTelemetryError(
                f"cannot attribute process to unknown span {span_id!r}"
            )
        owner = self._process_owners.get(process_id)
        if owner is not None and owner != span_id:
            raise BenchmarkTelemetryError(
                f"process {process_id!r} is already attributed to span "
                f"{owner!r}"
            )
        self._process_owners[process_id] = span_id

    def attribute_measurement(
        self, measurement_id: str, span_id: str
    ) -> None:
        measurement_id = _text(measurement_id, "measurement_id")
        span_id = _text(span_id, "span_id")
        if span_id not in self._spans:
            raise BenchmarkTelemetryError(
                f"cannot attribute measurement to unknown span {span_id!r}"
            )
        owner = self._measurement_owners.get(measurement_id)
        if owner is not None and owner != span_id:
            raise BenchmarkTelemetryError(
                f"measurement {measurement_id!r} is already attributed to "
                f"span {owner!r}"
            )
        self._measurement_owners[measurement_id] = span_id

    def record_measurement(
        self, measurement: BenchmarkResourceMeasurement
    ) -> BenchmarkResourceMeasurement:
        if not isinstance(measurement, BenchmarkResourceMeasurement):
            raise BenchmarkTelemetryError(
                "measurement must be BenchmarkResourceMeasurement"
            )
        if measurement.span.span_id not in self._spans:
            self.register_span(measurement.span)
        else:
            existing = self._spans[measurement.span.span_id]
            if existing.content_id != measurement.span.content_id:
                raise BenchmarkTelemetryError(
                    "measurement span does not match registered span"
                )
        self.attribute_measurement(
            measurement.measurement_id, measurement.span.span_id
        )
        for process_id in measurement.attributed_process_ids:
            self.attribute_process(process_id, measurement.span.span_id)
        for nested_id in measurement.attributed_measurement_ids:
            self.attribute_measurement(nested_id, measurement.span.span_id)
        if measurement.measurement_id in self._measurements:
            prior = self._measurements[measurement.measurement_id]
            if prior.content_id != measurement.content_id:
                raise BenchmarkTelemetryError(
                    "measurement_id collides with a different body"
                )
            return prior
        self._measurements[measurement.measurement_id] = measurement
        return measurement

    def record_provider_responses(
        self, responses: Sequence[Mapping[str, Any]]
    ) -> tuple["ProviderUsageRecord", ...]:
        """Map provider responses onto admitted spans in this session."""

        return admit_provider_usage_into_session(self, responses)

    def seal_receipt(self) -> "BenchmarkTelemetryReceipt":
        return BenchmarkTelemetryReceipt(
            run_id=self._root.run_id,
            root_span_id=self._root.span_id,
            spans=self.spans,
            measurements=self.measurements,
            process_attributions=tuple(
                sorted(
                    (process_id, span_id)
                    for process_id, span_id in self._process_owners.items()
                )
            ),
        )


@dataclass(frozen=True)
class BenchmarkTelemetryReceipt(_TelemetryContract):
    """Sealed population of spans and measurements for one benchmark run."""

    SCHEMA: ClassVar[str] = BENCHMARK_TELEMETRY_RECEIPT_SCHEMA

    run_id: str
    root_span_id: str
    spans: tuple[BenchmarkCausalSpan, ...]
    measurements: tuple[BenchmarkResourceMeasurement, ...]
    process_attributions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id"))
        object.__setattr__(
            self, "root_span_id", _text(self.root_span_id, "root_span_id")
        )
        spans: list[BenchmarkCausalSpan] = []
        for item in self.spans or ():
            if isinstance(item, Mapping):
                spans.append(BenchmarkCausalSpan.from_dict(item))
            elif isinstance(item, BenchmarkCausalSpan):
                spans.append(item)
            else:
                raise BenchmarkTelemetryError("spans must be causal spans")
        if not spans:
            raise BenchmarkTelemetryError("receipt requires a non-empty span population")
        span_ids = [item.span_id for item in spans]
        if len(span_ids) != len(set(span_ids)):
            raise BenchmarkTelemetryError("receipt span population has duplicates")
        if self.root_span_id not in span_ids:
            raise BenchmarkTelemetryError("root_span_id is missing from spans")
        for item in spans:
            if item.run_id != self.run_id:
                raise BenchmarkTelemetryError("span is foreign to receipt run_id")
        object.__setattr__(
            self,
            "spans",
            tuple(sorted(spans, key=lambda item: item.span_id)),
        )

        measurements: list[BenchmarkResourceMeasurement] = []
        for item in self.measurements or ():
            if isinstance(item, Mapping):
                measurements.append(BenchmarkResourceMeasurement.from_dict(item))
            elif isinstance(item, BenchmarkResourceMeasurement):
                measurements.append(item)
            else:
                raise BenchmarkTelemetryError(
                    "measurements must be resource measurements"
                )
        measurement_ids = [item.measurement_id for item in measurements]
        if len(measurement_ids) != len(set(measurement_ids)):
            raise BenchmarkTelemetryError(
                "receipt measurement population has duplicates"
            )
        known_spans = {item.span_id for item in self.spans}
        for item in measurements:
            if item.span.span_id not in known_spans:
                raise BenchmarkTelemetryError(
                    "measurement references an unregistered span"
                )
        object.__setattr__(
            self,
            "measurements",
            tuple(sorted(measurements, key=lambda item: item.measurement_id)),
        )

        attributions: list[tuple[str, str]] = []
        seen_processes: set[str] = set()
        for pair in self.process_attributions or ():
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise BenchmarkTelemetryError(
                    "process attributions must be (process_id, span_id) pairs"
                )
            process_id = _text(pair[0], "process_id")
            span_id = _text(pair[1], "span_id")
            if process_id in seen_processes:
                raise BenchmarkTelemetryError(
                    "process attribution population has duplicates"
                )
            if span_id not in known_spans:
                raise BenchmarkTelemetryError(
                    "process attribution references an unknown span"
                )
            seen_processes.add(process_id)
            attributions.append((process_id, span_id))
        object.__setattr__(
            self,
            "process_attributions",
            tuple(sorted(attributions)),
        )

    def certify_all(self) -> tuple[SpanReplayCertificate, ...]:
        certificates = []
        for measurement in self.measurements:
            certificates.append(
                certify_measurement_from_source_spans(
                    measurement, self.spans
                )
            )
        return tuple(certificates)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": BENCHMARK_TELEMETRY_CONTRACT_VERSION,
            "run_id": self.run_id,
            "root_span_id": self.root_span_id,
            "spans": [item.to_record() for item in self.spans],
            "measurements": [item.to_record() for item in self.measurements],
            "process_attributions": [
                [process_id, span_id]
                for process_id, span_id in self.process_attributions
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkTelemetryReceipt":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "run_id",
            "root_span_id",
            "spans",
            "measurements",
            "process_attributions",
            "content_id",
        }
        _closed(
            payload, schema=cls.SCHEMA, allowed=allowed, name="telemetry receipt"
        )
        result = cls(
            run_id=payload.get("run_id", ""),
            root_span_id=payload.get("root_span_id", ""),
            spans=tuple(payload.get("spans") or ()),
            measurements=tuple(payload.get("measurements") or ()),
            process_attributions=tuple(
                tuple(item)
                for item in (payload.get("process_attributions") or ())
            ),
        )
        _claim(payload, result.content_id, "content_id")
        return result


# ---------------------------------------------------------------------------
# Process-tree / GPU / network sensors
# ---------------------------------------------------------------------------


def _read_proc_stat(pid: int) -> tuple[int, int, int] | None:
    """Return (utime, stime, starttime) ticks or None on failure."""

    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (OSError, PermissionError):
        return None
    close = raw.rfind(")")
    if close < 0:
        return None
    fields = raw[close + 2 :].split()
    if len(fields) < 20:
        return None
    try:
        # fields: state ppid pgrp session tty_nr tpgid flags minflt cminflt
        # majflt cmajflt utime stime cutime cstime priority nice num_threads
        # itrealvalue starttime ...
        utime = int(fields[11])
        stime = int(fields[12])
        starttime = int(fields[19])
    except (TypeError, ValueError, IndexError):
        return None
    return utime, stime, starttime


def _read_proc_status_rss_bytes(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except (OSError, PermissionError):
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1]) * 1024
                except ValueError:
                    return None
    return None


def _read_proc_io(pid: int) -> tuple[int, int] | None:
    try:
        text = Path(f"/proc/{pid}/io").read_text(encoding="utf-8")
    except (OSError, PermissionError):
        return None
    read_bytes = write_bytes = None
    for line in text.splitlines():
        if line.startswith("read_bytes:"):
            try:
                read_bytes = int(line.split()[1])
            except (IndexError, ValueError):
                return None
        elif line.startswith("write_bytes:"):
            try:
                write_bytes = int(line.split()[1])
            except (IndexError, ValueError):
                return None
    if read_bytes is None or write_bytes is None:
        return None
    return read_bytes, write_bytes


def _list_children(pid: int) -> list[int]:
    children: list[int] = []
    task_dir = Path(f"/proc/{pid}/task")
    try:
        for task in task_dir.iterdir():
            children_path = task / "children"
            try:
                raw = children_path.read_text(encoding="utf-8").strip()
            except (OSError, PermissionError):
                continue
            if not raw:
                continue
            for token in raw.split():
                try:
                    children.append(int(token))
                except ValueError:
                    continue
    except (OSError, PermissionError):
        return []
    return children


def collect_descendant_pids(root_pid: int) -> tuple[int, ...] | None:
    """Walk the process tree under root_pid. None if root is unreadable."""

    if _read_proc_stat(root_pid) is None:
        return None
    seen: set[int] = {root_pid}
    queue = [root_pid]
    while queue:
        current = queue.pop()
        for child in _list_children(current):
            if child in seen:
                continue
            seen.add(child)
            queue.append(child)
            if len(seen) > MAX_CHILDREN:
                break
    return tuple(sorted(seen))


def _clock_ticks_per_second() -> int:
    try:
        return int(os.sysconf("SC_CLK_TCK"))
    except (AttributeError, OSError, ValueError):
        return 100


def sample_process_tree_resources(
    root_pid: int,
    *,
    wall_seconds_millionths: int,
    artifact_bytes_before: int = 0,
    artifact_bytes_after: int = 0,
    sensor_prefix: str = "procfs",
) -> dict[str, TelemetrySample]:
    """Sample the entire descendant process tree for one root PID.

    Missing or permission-denied sensors become ``unavailable`` samples, never
    numeric zeros.
    """

    sensor_cpu = _sensor_id(sensor_prefix, "cpu", str(root_pid))
    sensor_rss = _sensor_id(sensor_prefix, "rss", str(root_pid))
    sensor_io = _sensor_id(sensor_prefix, "io", str(root_pid))
    sensor_count = _sensor_id(sensor_prefix, "count", str(root_pid))
    sensor_disk = _sensor_id(sensor_prefix, "disk", str(root_pid))
    sensor_memtime = _sensor_id(sensor_prefix, "gib-seconds", str(root_pid))

    samples: dict[str, TelemetrySample] = {}
    pids = collect_descendant_pids(root_pid)
    if pids is None:
        reason = UnavailableReason.PERMISSION_DENIED
        # Distinguish absent root vs permission when possible.
        if not Path(f"/proc/{root_pid}").exists():
            reason = UnavailableReason.SENSOR_ABSENT
        for name in (
            "user_cpu_seconds",
            "system_cpu_seconds",
            "total_cpu_seconds",
            "peak_rss_bytes",
            "memory_gib_seconds",
            "read_bytes",
            "write_bytes",
            "peak_process_count",
        ):
            samples[name] = TelemetrySample.unavailable(
                name, reason, sensor_id=_sensor_id(sensor_prefix, name, str(root_pid))
            )
    else:
        ticks = _clock_ticks_per_second()
        user_ticks = 0
        system_ticks = 0
        peak_rss = 0
        rss_sum = 0
        rss_samples = 0
        read_total = 0
        write_total = 0
        io_ok = True
        cpu_ok = True
        rss_ok = True
        for pid in pids:
            stat = _read_proc_stat(pid)
            if stat is None:
                cpu_ok = False
            else:
                user_ticks += stat[0]
                system_ticks += stat[1]
            rss = _read_proc_status_rss_bytes(pid)
            if rss is None:
                rss_ok = False
            else:
                peak_rss = max(peak_rss, rss)
                rss_sum += rss
                rss_samples += 1
            io = _read_proc_io(pid)
            if io is None:
                io_ok = False
            else:
                read_total += io[0]
                write_total += io[1]

        if cpu_ok:
            user_millionths = (user_ticks * MILLIONTHS) // ticks
            system_millionths = (system_ticks * MILLIONTHS) // ticks
            samples["user_cpu_seconds"] = TelemetrySample.measured(
                "user_cpu_seconds",
                user_millionths,
                unit=UNIT_SECONDS_MILLIONTHS,
                sensor_id=sensor_cpu,
            )
            samples["system_cpu_seconds"] = TelemetrySample.measured(
                "system_cpu_seconds",
                system_millionths,
                unit=UNIT_SECONDS_MILLIONTHS,
                sensor_id=sensor_cpu,
            )
            samples["total_cpu_seconds"] = TelemetrySample.measured(
                "total_cpu_seconds",
                user_millionths + system_millionths,
                unit=UNIT_SECONDS_MILLIONTHS,
                sensor_id=sensor_cpu,
            )
        else:
            for name in (
                "user_cpu_seconds",
                "system_cpu_seconds",
                "total_cpu_seconds",
            ):
                samples[name] = TelemetrySample.unavailable(
                    name,
                    UnavailableReason.PERMISSION_DENIED,
                    sensor_id=sensor_cpu,
                )

        if rss_ok:
            samples["peak_rss_bytes"] = TelemetrySample.measured(
                "peak_rss_bytes",
                peak_rss,
                unit=UNIT_BYTES,
                sensor_id=sensor_rss,
            )
            # Approximate GiB-seconds from average RSS * wall time.
            avg_rss = rss_sum // max(1, rss_samples)
            gib_seconds_millionths = (
                avg_rss * wall_seconds_millionths
            ) // GIB_BYTES
            samples["memory_gib_seconds"] = TelemetrySample.measured(
                "memory_gib_seconds",
                gib_seconds_millionths,
                unit=UNIT_GIB_SECONDS_MILLIONTHS,
                sensor_id=sensor_memtime,
            )
        else:
            samples["peak_rss_bytes"] = TelemetrySample.unavailable(
                "peak_rss_bytes",
                UnavailableReason.PERMISSION_DENIED,
                sensor_id=sensor_rss,
            )
            samples["memory_gib_seconds"] = TelemetrySample.unavailable(
                "memory_gib_seconds",
                UnavailableReason.PERMISSION_DENIED,
                sensor_id=sensor_memtime,
            )

        if io_ok:
            samples["read_bytes"] = TelemetrySample.measured(
                "read_bytes", read_total, unit=UNIT_BYTES, sensor_id=sensor_io
            )
            samples["write_bytes"] = TelemetrySample.measured(
                "write_bytes", write_total, unit=UNIT_BYTES, sensor_id=sensor_io
            )
        else:
            samples["read_bytes"] = TelemetrySample.unavailable(
                "read_bytes",
                UnavailableReason.PERMISSION_DENIED,
                sensor_id=sensor_io,
            )
            samples["write_bytes"] = TelemetrySample.unavailable(
                "write_bytes",
                UnavailableReason.PERMISSION_DENIED,
                sensor_id=sensor_io,
            )

        samples["peak_process_count"] = TelemetrySample.measured(
            "peak_process_count",
            len(pids),
            unit=UNIT_COUNT,
            sensor_id=sensor_count,
        )

    growth = max(0, artifact_bytes_after - artifact_bytes_before)
    samples["disk_artifact_growth_bytes"] = TelemetrySample.measured(
        "disk_artifact_growth_bytes",
        growth,
        unit=UNIT_BYTES,
        sensor_id=sensor_disk,
    )
    return samples


def sample_self_process_tree_resources(
    *,
    wall_seconds_millionths: int,
    artifact_bytes_before: int = 0,
    artifact_bytes_after: int = 0,
) -> dict[str, TelemetrySample]:
    """Sample the current process tree (self + descendants)."""

    return sample_process_tree_resources(
        os.getpid(),
        wall_seconds_millionths=wall_seconds_millionths,
        artifact_bytes_before=artifact_bytes_before,
        artifact_bytes_after=artifact_bytes_after,
        sensor_prefix="procfs-self",
    )


def sample_gpu_resources(
    *,
    accelerator_present: bool,
    observation_seconds_millionths: int = 0,
) -> dict[str, TelemetrySample]:
    """Sample GPU utilization/VRAM/GPU-seconds when an accelerator is present.

    When the hardware profile declares no accelerator, GPU metrics are
    ``unavailable`` with ``hardware-absent``.  When an accelerator is declared
    but the driver/library is missing, they are ``sensor-absent`` or
    ``collection-failed`` — never numeric zero.
    """

    samples: dict[str, TelemetrySample] = {}
    if not accelerator_present:
        for name in GPU_METRIC_NAMES:
            samples[name] = TelemetrySample.unavailable(
                name,
                UnavailableReason.HARDWARE_ABSENT,
                sensor_id=_sensor_id("gpu", "absent", name),
            )
        return samples

    try:
        import pynvml  # type: ignore[import-not-found]
    except ImportError:
        for name in GPU_METRIC_NAMES:
            samples[name] = TelemetrySample.unavailable(
                name,
                UnavailableReason.SENSOR_ABSENT,
                sensor_id=_sensor_id("gpu", "pynvml-missing", name),
            )
        return samples

    try:
        pynvml.nvmlInit()
        try:
            count = int(pynvml.nvmlDeviceGetCount())
            if count <= 0:
                for name in GPU_METRIC_NAMES:
                    samples[name] = TelemetrySample.unavailable(
                        name,
                        UnavailableReason.HARDWARE_ABSENT,
                        sensor_id=_sensor_id("gpu", "zero-devices", name),
                    )
                return samples
            util_sum = 0
            util_samples = 0
            peak_vram = 0
            for index in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    util_sum += int(getattr(util, "gpu", 0))
                    util_samples += 1
                except Exception:
                    pass
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    used = int(getattr(mem, "used", 0))
                    peak_vram = max(peak_vram, used)
                except Exception:
                    pass
            sensor = _sensor_id("gpu", "nvml", str(count))
            if util_samples:
                avg_util_millionths = (util_sum * MILLIONTHS) // (
                    util_samples * 100
                )
                samples["gpu_utilization_time_weighted_ratio"] = (
                    TelemetrySample.measured(
                        "gpu_utilization_time_weighted_ratio",
                        avg_util_millionths,
                        unit=UNIT_RATIO_MILLIONTHS,
                        sensor_id=sensor,
                    )
                )
                gpu_seconds = (
                    avg_util_millionths * observation_seconds_millionths
                ) // MILLIONTHS
                samples["gpu_seconds"] = TelemetrySample.measured(
                    "gpu_seconds",
                    gpu_seconds,
                    unit=UNIT_SECONDS_MILLIONTHS,
                    sensor_id=sensor,
                )
            else:
                samples["gpu_utilization_time_weighted_ratio"] = (
                    TelemetrySample.unavailable(
                        "gpu_utilization_time_weighted_ratio",
                        UnavailableReason.COLLECTION_FAILED,
                        sensor_id=sensor,
                    )
                )
                samples["gpu_seconds"] = TelemetrySample.unavailable(
                    "gpu_seconds",
                    UnavailableReason.COLLECTION_FAILED,
                    sensor_id=sensor,
                )
            samples["peak_vram_bytes"] = TelemetrySample.measured(
                "peak_vram_bytes",
                peak_vram,
                unit=UNIT_BYTES,
                sensor_id=sensor,
            )
            samples["gpu_energy_joules_optional"] = TelemetrySample.unavailable(
                "gpu_energy_joules_optional",
                UnavailableReason.SENSOR_ABSENT,
                sensor_id=_sensor_id("gpu", "energy", "absent"),
            )
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        for name in GPU_METRIC_NAMES:
            samples[name] = TelemetrySample.unavailable(
                name,
                UnavailableReason.COLLECTION_FAILED,
                sensor_id=_sensor_id("gpu", "nvml-failed", name),
            )
    return samples


def sample_network_bytes(
    *,
    rx_bytes: int | None = None,
    tx_bytes: int | None = None,
    sensor_id: str | None = None,
) -> dict[str, TelemetrySample]:
    """Record network namespace counters or mark them unavailable."""

    sensor = sensor_id or _sensor_id("network", "namespace")
    samples: dict[str, TelemetrySample] = {}
    if rx_bytes is None:
        samples["network_rx_bytes"] = TelemetrySample.unavailable(
            "network_rx_bytes",
            UnavailableReason.SENSOR_ABSENT,
            sensor_id=sensor,
        )
    else:
        samples["network_rx_bytes"] = TelemetrySample.measured(
            "network_rx_bytes",
            _integer(rx_bytes, "rx_bytes"),
            unit=UNIT_BYTES,
            sensor_id=sensor,
        )
    if tx_bytes is None:
        samples["network_tx_bytes"] = TelemetrySample.unavailable(
            "network_tx_bytes",
            UnavailableReason.SENSOR_ABSENT,
            sensor_id=sensor,
        )
    else:
        samples["network_tx_bytes"] = TelemetrySample.measured(
            "network_tx_bytes",
            _integer(tx_bytes, "tx_bytes"),
            unit=UNIT_BYTES,
            sensor_id=sensor,
        )
    return samples


def sample_energy_optional(
    joules_millionths: int | None = None,
    *,
    sensor_id: str | None = None,
) -> TelemetrySample:
    """Optional energy estimate; absence does not invent zero joules."""

    if joules_millionths is None:
        return TelemetrySample.unavailable(
            "energy_joules_optional",
            UnavailableReason.SENSOR_ABSENT,
            sensor_id=sensor_id or _sensor_id("energy", "absent"),
        )
    return TelemetrySample.measured(
        "energy_joules_optional",
        _integer(joules_millionths, "joules_millionths"),
        unit=UNIT_JOULES_MILLIONTHS,
        sensor_id=sensor_id or _sensor_id("energy", "rapl-or-estimate"),
    )


def sample_rusage_self(
    *,
    wall_seconds_millionths: int,
) -> dict[str, TelemetrySample]:
    """Portable self rusage sample used when full /proc walks are denied."""

    sensor = _sensor_id("rusage", "self")
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
    except (AttributeError, OSError, ValueError):
        return {
            name: TelemetrySample.unavailable(
                name,
                UnavailableReason.COLLECTION_FAILED,
                sensor_id=sensor,
            )
            for name in (
                "user_cpu_seconds",
                "system_cpu_seconds",
                "total_cpu_seconds",
                "peak_rss_bytes",
                "memory_gib_seconds",
            )
        }

    user = seconds_to_millionths(usage.ru_utime + children.ru_utime)
    system = seconds_to_millionths(usage.ru_stime + children.ru_stime)
    # ru_maxrss is kilobytes on Linux.
    peak_rss = int(max(usage.ru_maxrss, children.ru_maxrss)) * 1024
    gib_seconds = (peak_rss * wall_seconds_millionths) // GIB_BYTES
    return {
        "user_cpu_seconds": TelemetrySample.measured(
            "user_cpu_seconds", user, unit=UNIT_SECONDS_MILLIONTHS, sensor_id=sensor
        ),
        "system_cpu_seconds": TelemetrySample.measured(
            "system_cpu_seconds",
            system,
            unit=UNIT_SECONDS_MILLIONTHS,
            sensor_id=sensor,
        ),
        "total_cpu_seconds": TelemetrySample.measured(
            "total_cpu_seconds",
            user + system,
            unit=UNIT_SECONDS_MILLIONTHS,
            sensor_id=sensor,
        ),
        "peak_rss_bytes": TelemetrySample.measured(
            "peak_rss_bytes", peak_rss, unit=UNIT_BYTES, sensor_id=sensor
        ),
        "memory_gib_seconds": TelemetrySample.measured(
            "memory_gib_seconds",
            gib_seconds,
            unit=UNIT_GIB_SECONDS_MILLIONTHS,
            sensor_id=sensor,
        ),
    }


# ---------------------------------------------------------------------------
# Joins: scheduler metrics + token ledger by causal span
# ---------------------------------------------------------------------------


def _percentile_millionths(values: Sequence[int], percentile: int) -> int:
    if not values:
        return 0
    ordered = sorted(int(item) for item in values)
    if percentile <= 0:
        return ordered[0]
    if percentile >= 100:
        return ordered[-1]
    # Nearest-rank.
    rank = max(1, int((percentile / 100) * len(ordered) + 0.999999))
    return ordered[min(len(ordered), rank) - 1]


def project_scheduler_clock_samples(
    snapshot: Mapping[str, Any],
    span: BenchmarkCausalSpan,
    *,
    concurrency_one_makespan_seconds_millionths: int | None = None,
    accepted_criteria: int = 0,
    time_to_first_useful_counterexample_seconds_millionths: int | None = None,
    sensor_id: str | None = None,
) -> dict[str, TelemetrySample]:
    """Join scheduler snapshot queue/merge/makespan metrics onto a span.

    Does not invent zeros for missing sensors: when the snapshot lacks the
    dimension for the span identity, the sample is ``unavailable``.
    """

    if not isinstance(snapshot, Mapping):
        raise BenchmarkTelemetryError("snapshot must be a mapping")
    sensor = sensor_id or _sensor_id(
        "scheduler-metrics", span.span_id, span.task_id
    )
    metrics = list(snapshot.get("metrics") or [])
    if not isinstance(metrics, list):
        metrics = []

    def _row_matches(row: Mapping[str, Any]) -> bool:
        if span.task_id:
            task_cid = str(row.get("task_cid") or row.get("task_id") or "")
            if task_cid and task_cid not in (span.task_id, f"task:{span.task_id}"):
                # Allow exact or suffix match on bare ids.
                if not (
                    task_cid.endswith(span.task_id)
                    or span.task_id.endswith(task_cid)
                ):
                    return False
        return True

    matched = [
        row
        for row in metrics
        if isinstance(row, Mapping) and _row_matches(row)
    ]

    samples: dict[str, TelemetrySample] = {}

    if span.duration_ns > 0:
        samples["end_to_end_makespan_seconds"] = TelemetrySample.measured(
            "end_to_end_makespan_seconds",
            span.duration_seconds_millionths,
            unit=UNIT_SECONDS_MILLIONTHS,
            sensor_id=sensor,
        )
    elif matched:
        # Fall back to sum of implementation+validation+queue+merge when the
        # span itself is unbound in time.
        total = 0
        for row in matched:
            for key in (
                "queue_wait_seconds",
                "implementation_duration_seconds",
                "validation_duration_seconds",
                "merge_wait_seconds",
            ):
                try:
                    total += seconds_to_millionths(float(row.get(key) or 0.0))
                except (TypeError, ValueError, BenchmarkTelemetryError):
                    continue
        if total > 0:
            samples["end_to_end_makespan_seconds"] = TelemetrySample.measured(
                "end_to_end_makespan_seconds",
                total,
                unit=UNIT_SECONDS_MILLIONTHS,
                sensor_id=sensor,
            )
        else:
            samples["end_to_end_makespan_seconds"] = TelemetrySample.unavailable(
                "end_to_end_makespan_seconds",
                UnavailableReason.COLLECTION_FAILED,
                sensor_id=sensor,
            )
    else:
        samples["end_to_end_makespan_seconds"] = TelemetrySample.unavailable(
            "end_to_end_makespan_seconds",
            UnavailableReason.SENSOR_ABSENT,
            sensor_id=sensor,
        )

    # Critical path: longest single-task chain of implementation+validation.
    if matched:
        path_values = []
        for row in matched:
            try:
                path_values.append(
                    seconds_to_millionths(
                        float(row.get("implementation_duration_seconds") or 0.0)
                        + float(row.get("validation_duration_seconds") or 0.0)
                    )
                )
            except (TypeError, ValueError, BenchmarkTelemetryError):
                continue
        if path_values:
            samples["critical_path_seconds"] = TelemetrySample.measured(
                "critical_path_seconds",
                max(path_values),
                unit=UNIT_SECONDS_MILLIONTHS,
                sensor_id=sensor,
            )
        else:
            samples["critical_path_seconds"] = TelemetrySample.unavailable(
                "critical_path_seconds",
                UnavailableReason.COLLECTION_FAILED,
                sensor_id=sensor,
            )
    else:
        samples["critical_path_seconds"] = TelemetrySample.unavailable(
            "critical_path_seconds",
            UnavailableReason.SENSOR_ABSENT,
            sensor_id=sensor,
        )

    makespan = samples["end_to_end_makespan_seconds"]
    if (
        makespan.status is SampleStatus.MEASURED
        and concurrency_one_makespan_seconds_millionths is not None
        and concurrency_one_makespan_seconds_millionths > 0
        and makespan.value > 0
    ):
        speedup = (
            concurrency_one_makespan_seconds_millionths * MILLIONTHS
        ) // makespan.value
        samples["speedup_vs_same_arm_concurrency_one"] = TelemetrySample.measured(
            "speedup_vs_same_arm_concurrency_one",
            speedup,
            unit=UNIT_RATIO_MILLIONTHS,
            sensor_id=sensor,
        )
        # Efficiency = speedup / observed width when width is known later.
        samples["parallel_efficiency"] = TelemetrySample.measured(
            "parallel_efficiency",
            speedup,  # refined below when width is known
            unit=UNIT_RATIO_MILLIONTHS,
            sensor_id=sensor,
        )
    else:
        samples["speedup_vs_same_arm_concurrency_one"] = (
            TelemetrySample.unavailable(
                "speedup_vs_same_arm_concurrency_one",
                UnavailableReason.SENSOR_ABSENT,
                sensor_id=sensor,
            )
        )
        samples["parallel_efficiency"] = TelemetrySample.unavailable(
            "parallel_efficiency",
            UnavailableReason.SENSOR_ABSENT,
            sensor_id=sensor,
        )

    queue_values = []
    merge_values = []
    for row in matched:
        try:
            queue_values.append(
                seconds_to_millionths(float(row.get("queue_wait_seconds") or 0.0))
            )
        except (TypeError, ValueError, BenchmarkTelemetryError):
            pass
        try:
            merge_values.append(
                seconds_to_millionths(float(row.get("merge_wait_seconds") or 0.0))
            )
        except (TypeError, ValueError, BenchmarkTelemetryError):
            pass

    if queue_values:
        samples["queue_latency_p50_seconds"] = TelemetrySample.measured(
            "queue_latency_p50_seconds",
            _percentile_millionths(queue_values, 50),
            unit=UNIT_SECONDS_MILLIONTHS,
            sensor_id=sensor,
        )
        samples["queue_latency_p95_seconds"] = TelemetrySample.measured(
            "queue_latency_p95_seconds",
            _percentile_millionths(queue_values, 95),
            unit=UNIT_SECONDS_MILLIONTHS,
            sensor_id=sensor,
        )
    else:
        samples["queue_latency_p50_seconds"] = TelemetrySample.unavailable(
            "queue_latency_p50_seconds",
            UnavailableReason.SENSOR_ABSENT,
            sensor_id=sensor,
        )
        samples["queue_latency_p95_seconds"] = TelemetrySample.unavailable(
            "queue_latency_p95_seconds",
            UnavailableReason.SENSOR_ABSENT,
            sensor_id=sensor,
        )

    if merge_values:
        samples["merge_conflict_serialization_seconds"] = TelemetrySample.measured(
            "merge_conflict_serialization_seconds",
            sum(merge_values),
            unit=UNIT_SECONDS_MILLIONTHS,
            sensor_id=sensor,
        )
    else:
        samples["merge_conflict_serialization_seconds"] = (
            TelemetrySample.unavailable(
                "merge_conflict_serialization_seconds",
                UnavailableReason.SENSOR_ABSENT,
                sensor_id=sensor,
            )
        )

    # Widths from snapshot phase/state if present.
    phase_counts = snapshot.get("phase_counts") or {}
    if isinstance(phase_counts, Mapping):
        ready = int(phase_counts.get("ready") or 0)
        active = int(phase_counts.get("active") or 0)
        samples["ready_width"] = TelemetrySample.measured(
            "ready_width", ready, unit=UNIT_COUNT, sensor_id=sensor
        )
        samples["observed_width"] = TelemetrySample.measured(
            "observed_width",
            max(active, 0),
            unit=UNIT_COUNT,
            sensor_id=sensor,
        )
        # Admitted width is not inventable from phase gauges alone.
        admitted = snapshot.get("admitted_width")
        if isinstance(admitted, int) and not isinstance(admitted, bool):
            samples["admitted_width"] = TelemetrySample.measured(
                "admitted_width", admitted, unit=UNIT_COUNT, sensor_id=sensor
            )
        else:
            samples["admitted_width"] = TelemetrySample.unavailable(
                "admitted_width",
                UnavailableReason.SENSOR_ABSENT,
                sensor_id=sensor,
            )
        if active > 0 and makespan.status is SampleStatus.MEASURED:
            # Occupancy: active / (active) trivial; use completions proxy if any.
            samples["worker_occupancy_ratio"] = TelemetrySample.measured(
                "worker_occupancy_ratio",
                MILLIONTHS,  # fully occupied while active > 0 under observation
                unit=UNIT_RATIO_MILLIONTHS,
                sensor_id=sensor,
            )
        else:
            samples["worker_occupancy_ratio"] = TelemetrySample.unavailable(
                "worker_occupancy_ratio",
                UnavailableReason.SENSOR_ABSENT,
                sensor_id=sensor,
            )
        # Refine parallel efficiency by observed width when possible.
        if (
            samples["parallel_efficiency"].status is SampleStatus.MEASURED
            and active > 0
        ):
            efficiency = samples["parallel_efficiency"].value // active
            samples["parallel_efficiency"] = TelemetrySample.measured(
                "parallel_efficiency",
                efficiency,
                unit=UNIT_RATIO_MILLIONTHS,
                sensor_id=sensor,
            )
    else:
        for name in (
            "ready_width",
            "admitted_width",
            "observed_width",
            "worker_occupancy_ratio",
        ):
            samples[name] = TelemetrySample.unavailable(
                name, UnavailableReason.SENSOR_ABSENT, sensor_id=sensor
            )

    if time_to_first_useful_counterexample_seconds_millionths is None:
        samples["time_to_first_useful_counterexample_seconds"] = (
            TelemetrySample.unavailable(
                "time_to_first_useful_counterexample_seconds",
                UnavailableReason.SENSOR_ABSENT,
                sensor_id=sensor,
            )
        )
    else:
        samples["time_to_first_useful_counterexample_seconds"] = (
            TelemetrySample.measured(
                "time_to_first_useful_counterexample_seconds",
                _integer(
                    time_to_first_useful_counterexample_seconds_millionths,
                    "ttfu",
                ),
                unit=UNIT_SECONDS_MILLIONTHS,
                sensor_id=sensor,
            )
        )

    if (
        makespan.status is SampleStatus.MEASURED
        and makespan.value > 0
        and accepted_criteria >= 0
    ):
        # criteria/hour = criteria * 3600 / seconds
        criteria_per_hour = (
            accepted_criteria * 3600 * MILLIONTHS
        ) // makespan.value
        samples["accepted_criteria_per_hour"] = TelemetrySample.measured(
            "accepted_criteria_per_hour",
            criteria_per_hour,
            unit=UNIT_COUNT,
            sensor_id=sensor,
        )
    else:
        samples["accepted_criteria_per_hour"] = TelemetrySample.unavailable(
            "accepted_criteria_per_hour",
            UnavailableReason.SENSOR_ABSENT,
            sensor_id=sensor,
        )

    return samples


def project_token_ledger_samples(
    ledger: Any,
    span: BenchmarkCausalSpan,
    *,
    provider_called: bool | None = None,
    proved_obligations: int = 0,
    deterministic_stages: int = 0,
    llm_stages: int = 0,
    sensor_id: str | None = None,
) -> dict[str, TelemetrySample]:
    """Join a supervisor token ledger onto a causal span as telemetry samples.

    Provider token metrics are required only when the arm emitted a provider
    call.  When no call occurred, tokens are ``provider-omitted`` rather than
    zero.
    """

    sensor = sensor_id or _sensor_id(
        "token-ledger", span.span_id, getattr(ledger, "ledger_id", "ledger")
    )
    # Lazy import surface: ledger may be passed as duck-typed report object.
    report = getattr(ledger, "report", None)
    if report is None and hasattr(ledger, "build_report"):
        report = ledger.build_report()

    called = provider_called
    if called is None:
        if report is None:
            called = False
        else:
            called = int(getattr(report, "total_tokens", 0) or 0) > 0 or int(
                getattr(report, "lifecycle_event_count", 0) or 0
            ) > 0

    samples: dict[str, TelemetrySample] = {}
    if not called or report is None:
        reason = (
            UnavailableReason.PROVIDER_OMITTED
            if not called
            else UnavailableReason.SENSOR_ABSENT
        )
        for name in TOKEN_METRIC_NAMES:
            samples[name] = TelemetrySample.unavailable(
                name, reason, sensor_id=sensor
            )
        return samples

    def _measured(name: str, value: int, unit: str) -> TelemetrySample:
        return TelemetrySample.measured(
            name, _integer(value, name), unit=unit, sensor_id=sensor
        )

    samples["provider_native_input_tokens"] = _measured(
        "provider_native_input_tokens",
        int(getattr(report, "input_tokens", 0)),
        UNIT_TOKENS,
    )
    samples["provider_native_output_tokens"] = _measured(
        "provider_native_output_tokens",
        int(getattr(report, "output_tokens", 0)),
        UNIT_TOKENS,
    )
    samples["provider_native_reused_tokens"] = _measured(
        "provider_native_reused_tokens",
        int(getattr(report, "reused_tokens", 0)),
        UNIT_TOKENS,
    )
    samples["provider_native_retry_tokens"] = _measured(
        "provider_native_retry_tokens",
        int(getattr(report, "retry_tokens", 0)),
        UNIT_TOKENS,
    )
    cancelled = int(
        getattr(
            report,
            "cancelled_tokens",
            getattr(report, "abandoned_tokens", 0),
        )
    )
    samples["provider_native_cancelled_tokens"] = _measured(
        "provider_native_cancelled_tokens", cancelled, UNIT_TOKENS
    )
    cached = getattr(report, "cached_input_tokens", None)
    if type(cached) is int:
        samples["provider_native_cached_input_tokens"] = _measured(
            "provider_native_cached_input_tokens", cached, UNIT_TOKENS
        )
    else:
        samples["provider_native_cached_input_tokens"] = TelemetrySample.unavailable(
            "provider_native_cached_input_tokens",
            UnavailableReason.PROVIDER_OMITTED,
            sensor_id=sensor,
        )
    reasoning = getattr(report, "reasoning_tokens", None)
    if type(reasoning) is int:
        samples["provider_native_reasoning_tokens"] = _measured(
            "provider_native_reasoning_tokens", reasoning, UNIT_TOKENS
        )
    else:
        samples["provider_native_reasoning_tokens"] = TelemetrySample.unavailable(
            "provider_native_reasoning_tokens",
            UnavailableReason.PROVIDER_OMITTED,
            sensor_id=sensor,
        )
    samples["model_call_count"] = _measured(
        "model_call_count",
        int(getattr(report, "lifecycle_event_count", 0)),
        UNIT_COUNT,
    )
    samples["cache_reuse_count"] = _measured(
        "cache_reuse_count",
        int(getattr(report, "reused_tokens", 0)),
        UNIT_COUNT,
    )
    samples["provider_cost_microusd"] = _measured(
        "provider_cost_microusd",
        int(getattr(report, "total_cost_microunits", 0)),
        UNIT_MICROUSD,
    )
    # Alias under process-tree registry name.
    samples["provider_quota_units"] = _measured(
        "provider_quota_units",
        int(getattr(report, "total_tokens", 0)),
        UNIT_QUOTA,
    )

    tokenizer = ""
    if span.provider is not None and span.provider.tokenizer_id:
        tokenizer = span.provider.tokenizer_id
    else:
        # Best-effort from first attribution envelope if present.
        attributions = getattr(ledger, "attributions", ()) or ()
        for attribution in attributions:
            usage = getattr(attribution, "usage", None)
            envelope = getattr(usage, "envelope", None)
            tokenizer = str(getattr(envelope, "tokenizer_id", "") or "")
            if tokenizer:
                break
    if tokenizer:
        samples["tokenizer_identity"] = _measured(
            "tokenizer_identity",
            _identity_digest(tokenizer),
            UNIT_IDENTITY,
        )
    else:
        samples["tokenizer_identity"] = TelemetrySample.unavailable(
            "tokenizer_identity",
            UnavailableReason.SENSOR_ABSENT,
            sensor_id=sensor,
        )

    # Context bytes are not native to the ledger; mark unavailable rather than
    # inventing a conversion.
    samples["context_bytes"] = TelemetrySample.unavailable(
        "context_bytes",
        UnavailableReason.SENSOR_ABSENT,
        sensor_id=sensor,
    )

    accepted = int(getattr(report, "accepted_criterion_count", 0))
    total_tokens = int(getattr(report, "total_tokens", 0))
    total_cost = int(getattr(report, "total_cost_microunits", 0))
    if accepted > 0:
        samples["tokens_per_accepted_criterion"] = _measured(
            "tokens_per_accepted_criterion",
            total_tokens // accepted,
            UNIT_TOKENS,
        )
        samples["provider_cost_per_accepted_criterion"] = _measured(
            "provider_cost_per_accepted_criterion",
            total_cost // accepted,
            UNIT_MICROUSD,
        )
    else:
        samples["tokens_per_accepted_criterion"] = TelemetrySample.unavailable(
            "tokens_per_accepted_criterion",
            UnavailableReason.COLLECTION_FAILED,
            sensor_id=sensor,
        )
        samples["provider_cost_per_accepted_criterion"] = (
            TelemetrySample.unavailable(
                "provider_cost_per_accepted_criterion",
                UnavailableReason.COLLECTION_FAILED,
                sensor_id=sensor,
            )
        )

    if proved_obligations > 0:
        samples["tokens_per_proved_obligation"] = _measured(
            "tokens_per_proved_obligation",
            total_tokens // proved_obligations,
            UNIT_TOKENS,
        )
        samples["provider_cost_per_proved_obligation"] = _measured(
            "provider_cost_per_proved_obligation",
            total_cost // proved_obligations,
            UNIT_MICROUSD,
        )
    else:
        samples["tokens_per_proved_obligation"] = TelemetrySample.unavailable(
            "tokens_per_proved_obligation",
            UnavailableReason.SENSOR_ABSENT,
            sensor_id=sensor,
        )
        samples["provider_cost_per_proved_obligation"] = (
            TelemetrySample.unavailable(
                "provider_cost_per_proved_obligation",
                UnavailableReason.SENSOR_ABSENT,
                sensor_id=sensor,
            )
        )

    stages = deterministic_stages + llm_stages
    if stages > 0:
        samples["deterministic_llm_avoidance_ratio"] = _measured(
            "deterministic_llm_avoidance_ratio",
            (deterministic_stages * MILLIONTHS) // stages,
            UNIT_RATIO_MILLIONTHS,
        )
    else:
        samples["deterministic_llm_avoidance_ratio"] = TelemetrySample.unavailable(
            "deterministic_llm_avoidance_ratio",
            UnavailableReason.SENSOR_ABSENT,
            sensor_id=sensor,
        )

    return samples


def build_resource_measurement(
    *,
    measurement_id: str,
    span: BenchmarkCausalSpan,
    samples: Mapping[str, TelemetrySample] | Sequence[TelemetrySample],
    attributed_process_ids: Sequence[str] = (),
    attributed_measurement_ids: Sequence[str] = (),
    source_span_ids: Sequence[str] | None = None,
) -> BenchmarkResourceMeasurement:
    """Assemble a resource measurement from named or ordered samples."""

    if isinstance(samples, Mapping):
        ordered = tuple(samples[name] for name in sorted(samples))
    else:
        ordered = tuple(samples)
    return BenchmarkResourceMeasurement(
        measurement_id=measurement_id,
        span=span,
        samples=ordered,
        attributed_process_ids=tuple(attributed_process_ids),
        attributed_measurement_ids=tuple(attributed_measurement_ids),
        source_span_ids=tuple(source_span_ids or (span.span_id,)),
    )


def build_span_joined_measurement(
    *,
    measurement_id: str,
    span: BenchmarkCausalSpan,
    scheduler_snapshot: Mapping[str, Any] | None = None,
    token_ledger: Any | None = None,
    process_samples: Mapping[str, TelemetrySample] | None = None,
    gpu_samples: Mapping[str, TelemetrySample] | None = None,
    network_samples: Mapping[str, TelemetrySample] | None = None,
    energy_sample: TelemetrySample | None = None,
    provider_cost_microusd: int | None = None,
    provider_quota_units: int | None = None,
    concurrency_one_makespan_seconds_millionths: int | None = None,
    accepted_criteria: int = 0,
    provider_called: bool | None = None,
    attributed_process_ids: Sequence[str] = (),
) -> BenchmarkResourceMeasurement:
    """Join scheduler, ledger, and sensor samples onto one causal span."""

    merged: dict[str, TelemetrySample] = {}
    if scheduler_snapshot is not None:
        merged.update(
            project_scheduler_clock_samples(
                scheduler_snapshot,
                span,
                concurrency_one_makespan_seconds_millionths=(
                    concurrency_one_makespan_seconds_millionths
                ),
                accepted_criteria=accepted_criteria,
            )
        )
    if token_ledger is not None:
        merged.update(
            project_token_ledger_samples(
                token_ledger,
                span,
                provider_called=provider_called,
            )
        )
    if process_samples:
        merged.update(process_samples)
    if gpu_samples:
        merged.update(gpu_samples)
    if network_samples:
        merged.update(network_samples)
    if energy_sample is not None:
        merged[energy_sample.metric_name] = energy_sample

    sensor = _sensor_id("provider-cost", span.span_id)
    if provider_cost_microusd is not None and "provider_cost_microusd" not in merged:
        merged["provider_cost_microusd"] = TelemetrySample.measured(
            "provider_cost_microusd",
            provider_cost_microusd,
            unit=UNIT_MICROUSD,
            sensor_id=sensor,
        )
    if provider_quota_units is not None and "provider_quota_units" not in merged:
        merged["provider_quota_units"] = TelemetrySample.measured(
            "provider_quota_units",
            provider_quota_units,
            unit=UNIT_QUOTA,
            sensor_id=sensor,
        )

    return build_resource_measurement(
        measurement_id=measurement_id,
        span=span,
        samples=merged,
        attributed_process_ids=attributed_process_ids,
    )


# ---------------------------------------------------------------------------
# ASEH-011: provider/model usage mapped onto admitted causal task spans
# ---------------------------------------------------------------------------

PROVIDER_USAGE_SAMPLE_NAMES: Final[tuple[str, ...]] = (
    "provider_native_input_tokens",
    "provider_native_output_tokens",
    "provider_native_cached_input_tokens",
    "provider_native_reasoning_tokens",
    "model_call_count",
    "provider_reported_charge_microusd",
)

_INPUT_TOKEN_KEYS: Final[tuple[str, ...]] = (
    "input_tokens",
    "prompt_tokens",
    "input",
)
_OUTPUT_TOKEN_KEYS: Final[tuple[str, ...]] = (
    "output_tokens",
    "completion_tokens",
    "output",
)
_CACHED_TOKEN_KEYS: Final[tuple[str, ...]] = (
    "cached_input_tokens",
    "cached_tokens",
    "prompt_cache_hit_tokens",
    "cache_read_input_tokens",
    "cached_prompt_tokens",
)
_REASONING_TOKEN_KEYS: Final[tuple[str, ...]] = (
    "reasoning_tokens",
    "reasoning_output_tokens",
    "thought_tokens",
)
_CHARGE_KEYS: Final[tuple[str, ...]] = (
    "reported_charge_microusd",
    "provider_reported_charge_microusd",
    "cost_microusd",
    "cost_microunits",
    "charge_microusd",
    "total_cost_microusd",
)
_REQUEST_ID_KEYS: Final[tuple[str, ...]] = (
    "request_id",
    "safe_request_id",
    "provider_request_id",
    "x_request_id",
    "x-request-id",
)
_SPAN_ID_KEYS: Final[tuple[str, ...]] = (
    "span_id",
    "causal_span_id",
    "benchmark_span_id",
)
_MODEL_REVISION_KEYS: Final[tuple[str, ...]] = (
    "model_revision",
    "system_fingerprint",
    "model_version",
    "revision",
)
_ESTIMATE_KEYS: Final[tuple[str, ...]] = (
    "estimate",
    "estimated_charge",
    "token_based_estimated_charge",
    "estimated_cost_microusd",
)
_SECRET_KEY_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?key|secret|password|passwd|"
    r"private[_-]?key|authorization|bearer|client[_-]?secret|"
    r"access[_-]?token|refresh[_-]?token|session[_-]?token|"
    r"raw[_-]?secret|credential)(?:$|[_-])",
    re.IGNORECASE,
)
_SECRET_VALUE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:bearer\s+\S{12,}|sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"hf_[A-Za-z0-9]{24,}|xox[baprs]-[A-Za-z0-9-]{20,}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)",
    re.IGNORECASE,
)
_SAFE_REQUEST_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9._:-]{8,128}$"
)
_RECEIPT_REASON_BY_UNAVAILABLE: Final[dict[str, str]] = {
    UnavailableReason.SENSOR_ABSENT.value: "sensor_absent",
    UnavailableReason.PERMISSION_DENIED.value: "permission_denied",
    UnavailableReason.HARDWARE_ABSENT.value: "hardware_absent",
    UnavailableReason.PROVIDER_OMITTED.value: "provider_omitted",
    UnavailableReason.COLLECTION_FAILED.value: "collection_failed",
    UnavailableReason.NOT_REPORTED.value: "not_reported",
    UnavailableReason.UNSAFE_REQUEST_ID.value: "not_reported",
}


class ProviderUsageDisposition(str, Enum):
    ADMITTED = "admitted"
    QUARANTINED = "quarantined"


class ProviderUsageQuarantineReason(str, Enum):
    UNBOUND_USAGE = "unbound-usage"
    CREDENTIAL_LEAKAGE = "credential-leakage"
    ESTIMATED_AS_MEASURED = "estimated-as-measured"
    MISSING_CAUSAL_IDENTITY = "missing-causal-identity"


def _receipt_reason(reason_code: str) -> str:
    return _RECEIPT_REASON_BY_UNAVAILABLE.get(
        reason_code, reason_code.replace("-", "_")
    )


def _optional_text_field(
    payload: Mapping[str, Any], names: Sequence[str]
) -> str:
    for name in names:
        if name not in payload:
            continue
        value = payload[name]
        if value is None:
            continue
        if not isinstance(value, str):
            raise BenchmarkTelemetryError(f"{name} must be text")
        text = value.strip()
        if text:
            return _text(text, name)
    return ""


def _lookup_reported_int(
    payload: Mapping[str, Any], names: Sequence[str], *, field: str
) -> int | None:
    """Return an explicit integer when present; None means the provider omitted it."""

    for name in names:
        if name not in payload:
            continue
        value = payload[name]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise BenchmarkTelemetryError(f"{field} must be an integer")
        return _integer(value, field)
    return None


def _nested_mapping(payload: Mapping[str, Any], *names: str) -> Mapping[str, Any]:
    current: Any = payload
    for name in names:
        if not isinstance(current, Mapping) or name not in current:
            return {}
        current = current[name]
    return current if isinstance(current, Mapping) else {}


def _usage_mapping(response: Mapping[str, Any]) -> Mapping[str, Any]:
    usage = response.get("usage")
    if usage is None:
        return {}
    if not isinstance(usage, Mapping):
        raise BenchmarkTelemetryError("usage must be an object")
    return usage


def _contains_credential_material(value: Any, *, path: str = "$") -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            lowered = key_text.casefold()
            if _SECRET_KEY_RE.search(key_text) or lowered in {
                "authorization",
                "api_key",
                "apikey",
                "access_token",
                "client_secret",
                "private_key",
                "password",
            }:
                if child not in (None, ""):
                    return True
            if _contains_credential_material(child, path=f"{path}.{key_text}"):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(
            _contains_credential_material(item, path=f"{path}[]") for item in value
        )
    if isinstance(value, str):
        return _SECRET_VALUE_RE.search(value) is not None
    return False


def _request_id_is_credential(value: str) -> bool:
    return _SECRET_VALUE_RE.search(value) is not None or value.casefold().startswith(
        ("sk-", "bearer ", "ghp_", "gho_", "hf_")
    )


def extract_safe_request_id(value: Any) -> str | None:
    """Return an opaque provider request id, or None when it is absent or unsafe.

    Credential-shaped values raise so the mapper can quarantine the record
    rather than persist the secret as telemetry.
    """

    if value is None:
        return None
    if not isinstance(value, str):
        raise BenchmarkTelemetryError("request_id must be text")
    text = value.strip()
    if not text:
        return None
    if _request_id_is_credential(text) or _SECRET_VALUE_RE.search(text):
        raise BenchmarkTelemetryError("request_id contains credential-shaped data")
    if _SAFE_REQUEST_ID_RE.fullmatch(text) is None:
        return None
    if "\x00" in text or len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise BenchmarkTelemetryError("request_id is unsafe or too large")
    return text


def _extract_request_ids(response: Mapping[str, Any]) -> tuple[str, ...]:
    found: list[str] = []
    raw_values: list[Any] = []
    for key in _REQUEST_ID_KEYS:
        if key in response:
            raw_values.append(response[key])
    if "id" in response:
        raw_values.append(response["id"])
    nested = response.get("safe_provider_request_ids")
    if isinstance(nested, Mapping) and isinstance(nested.get("values"), list):
        raw_values.extend(nested["values"])
    elif isinstance(nested, (list, tuple)):
        raw_values.extend(nested)
    for raw in raw_values:
        if raw is None:
            continue
        extracted = extract_safe_request_id(raw)
        if extracted and extracted not in found:
            found.append(extracted)
    return tuple(found)


def _sample_or_unavailable(
    metric_name: str,
    value: int | None,
    *,
    unit: str,
    sensor_id: str,
    reason: UnavailableReason = UnavailableReason.PROVIDER_OMITTED,
) -> TelemetrySample:
    if value is None:
        return TelemetrySample.unavailable(
            metric_name, reason, sensor_id=sensor_id
        )
    return TelemetrySample.measured(
        metric_name, value, unit=unit, sensor_id=sensor_id
    )


def _blank_provider_usage_samples(
    sensor_id: str,
    reason: UnavailableReason = UnavailableReason.PROVIDER_OMITTED,
) -> dict[str, TelemetrySample]:
    samples = {
        name: TelemetrySample.unavailable(name, reason, sensor_id=sensor_id)
        for name in PROVIDER_USAGE_SAMPLE_NAMES
    }
    return samples


def _calls_by_model_class_samples(
    *,
    model_class: str,
    call_count: int | None,
    sensor_id: str,
    reason: UnavailableReason = UnavailableReason.PROVIDER_OMITTED,
) -> dict[str, TelemetrySample]:
    samples: dict[str, TelemetrySample] = {}
    if not model_class or call_count is None:
        for name in MODEL_CLASSES:
            samples[name] = TelemetrySample.unavailable(
                f"calls_by_model_class.{name}",
                reason,
                sensor_id=sensor_id,
            )
        return samples
    if model_class not in MODEL_CLASSES:
        raise BenchmarkTelemetryError("model_class is not a closed value")
    for name in MODEL_CLASSES:
        value = call_count if name == model_class else 0
        samples[name] = TelemetrySample.measured(
            f"calls_by_model_class.{name}",
            value,
            unit=UNIT_COUNT,
            sensor_id=sensor_id,
        )
    return samples


def _quantity_fragment(sample: TelemetrySample) -> dict[str, Any]:
    if sample.status is SampleStatus.MEASURED:
        return receipt_measured_quantity(
            sample.value, unit=sample.unit, sensor_id=sample.sensor_id
        )
    return receipt_unavailable(_receipt_reason(sample.reason_code))


def _assert_estimate_not_labeled_measured(payload: Mapping[str, Any]) -> None:
    truth = payload.get("truth_state")
    if truth == "measured" or "sensor_id" in payload:
        raise BenchmarkTelemetryError("estimated values cannot be labeled measured")
    if payload.get("estimated_as_measured") is True:
        raise BenchmarkTelemetryError("estimated values cannot be labeled measured")


@dataclass(frozen=True)
class LabeledEstimate(_TelemetryContract):
    """Token/cost estimate that cannot masquerade as a measured sample."""

    SCHEMA: ClassVar[str] = LABELED_ESTIMATE_SCHEMA

    metric_name: str
    value: int
    unit: str
    estimator_id: str
    method: str
    price_snapshot_identity: str = "unavailable"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metric_name", _text(self.metric_name, "metric_name")
        )
        object.__setattr__(self, "value", _integer(self.value, "value"))
        object.__setattr__(self, "unit", _text(self.unit, "unit"))
        object.__setattr__(
            self, "estimator_id", _text(self.estimator_id, "estimator_id")
        )
        method = _text(self.method, "method")
        if method not in ESTIMATOR_METHODS:
            raise BenchmarkTelemetryError("estimator method is not a closed value")
        object.__setattr__(self, "method", method)
        identity = _text(
            self.price_snapshot_identity, "price_snapshot_identity"
        )
        if IDENTITY_REF_RE.fullmatch(identity) is None:
            raise BenchmarkTelemetryError(
                "price_snapshot_identity is not a closed identity ref"
            )
        object.__setattr__(self, "price_snapshot_identity", identity)

    def to_quantity(self) -> dict[str, Any]:
        return receipt_estimated_quantity(
            self.value,
            unit=self.unit,
            estimator_id=self.estimator_id,
            method=self.method,
            price_snapshot_identity=self.price_snapshot_identity,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": BENCHMARK_TELEMETRY_CONTRACT_VERSION,
            "metric_name": self.metric_name,
            "truth_state": "estimated",
            "value": self.value,
            "unit": self.unit,
            "estimator_id": self.estimator_id,
            "method": self.method,
            "price_snapshot_identity": self.price_snapshot_identity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LabeledEstimate":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "metric_name",
            "truth_state",
            "value",
            "unit",
            "estimator_id",
            "method",
            "price_snapshot_identity",
            "sensor_id",
            "content_id",
        }
        _closed(
            payload, schema=cls.SCHEMA, allowed=allowed, name="labeled estimate"
        )
        if payload.get("truth_state", "estimated") != "estimated":
            raise BenchmarkTelemetryError(
                "estimated values cannot be labeled measured"
            )
        if "sensor_id" in payload:
            raise BenchmarkTelemetryError(
                "estimated values cannot be labeled measured"
            )
        result = cls(
            metric_name=payload.get("metric_name", ""),
            value=payload.get("value", 0),
            unit=payload.get("unit", ""),
            estimator_id=payload.get("estimator_id", ""),
            method=payload.get("method", ""),
            price_snapshot_identity=payload.get(
                "price_snapshot_identity", "unavailable"
            ),
        )
        _claim(payload, result.content_id, "content_id")
        return result


def _parse_labeled_estimate(
    response: Mapping[str, Any],
) -> LabeledEstimate | None:
    blob: Any = None
    for key in _ESTIMATE_KEYS:
        if key in response:
            blob = response[key]
            break
    usage = _usage_mapping(response)
    if blob is None:
        for key in _ESTIMATE_KEYS:
            if key in usage:
                blob = usage[key]
                break
    if blob is None:
        return None
    if isinstance(blob, bool):
        raise BenchmarkTelemetryError("labeled estimate must be an integer")
    if isinstance(blob, int):
        estimator_id = _optional_text_field(
            response, ("estimator_id", "price_estimator_id")
        )
        method = _optional_text_field(response, ("estimator_method", "method"))
        if not estimator_id or not method:
            raise BenchmarkTelemetryError(
                "labeled estimate requires estimator_id and method"
            )
        return LabeledEstimate(
            metric_name="token_based_estimated_charge_microusd",
            value=_integer(blob, "estimated_charge"),
            unit=UNIT_MICROUSD,
            estimator_id=estimator_id,
            method=method,
            price_snapshot_identity=_optional_text_field(
                response, ("price_snapshot_identity",)
            )
            or "unavailable",
        )
    if not isinstance(blob, Mapping):
        raise BenchmarkTelemetryError("labeled estimate must be an object")
    _assert_estimate_not_labeled_measured(blob)
    value = _lookup_reported_int(
        blob, ("value", "amount", "microusd"), field="estimated_charge"
    )
    if value is None:
        return None
    estimator_id = _optional_text_field(
        blob, ("estimator_id", "price_estimator_id")
    ) or _optional_text_field(response, ("estimator_id",))
    method = _optional_text_field(
        blob, ("method", "estimator_method")
    ) or _optional_text_field(response, ("estimator_method", "method"))
    if not estimator_id or not method:
        raise BenchmarkTelemetryError(
            "labeled estimate requires estimator_id and method"
        )
    return LabeledEstimate(
        metric_name=str(
            blob.get("metric_name") or "token_based_estimated_charge_microusd"
        ),
        value=value,
        unit=str(blob.get("unit") or UNIT_MICROUSD),
        estimator_id=estimator_id,
        method=method,
        price_snapshot_identity=_optional_text_field(
            blob, ("price_snapshot_identity",)
        )
        or _optional_text_field(response, ("price_snapshot_identity",))
        or "unavailable",
    )


def _admitted_span_index(
    admitted: (
        "BenchmarkTelemetrySession"
        | Mapping[str, BenchmarkCausalSpan]
        | Sequence[BenchmarkCausalSpan]
    ),
) -> dict[str, BenchmarkCausalSpan]:
    if isinstance(admitted, BenchmarkTelemetrySession):
        spans = admitted.spans
    elif isinstance(admitted, Mapping):
        spans = []
        for key, span in admitted.items():
            if isinstance(span, Mapping):
                span = BenchmarkCausalSpan.from_dict(span)
            if not isinstance(span, BenchmarkCausalSpan):
                raise BenchmarkTelemetryError(
                    "admitted spans must be BenchmarkCausalSpan records"
                )
            if key != span.span_id:
                raise BenchmarkTelemetryError(
                    "admitted span index key must match span_id"
                )
            spans.append(span)
    else:
        spans = []
        for span in admitted:
            if isinstance(span, Mapping):
                span = BenchmarkCausalSpan.from_dict(span)
            if not isinstance(span, BenchmarkCausalSpan):
                raise BenchmarkTelemetryError(
                    "admitted spans must be BenchmarkCausalSpan records"
                )
            spans.append(span)
    index: dict[str, BenchmarkCausalSpan] = {}
    for span in spans:
        if span.span_id in index and index[span.span_id].content_id != span.content_id:
            raise BenchmarkTelemetryError(
                f"admitted span {span.span_id!r} has conflicting identities"
            )
        index[span.span_id] = span
    return index


def _resolve_admitted_span(
    response: Mapping[str, Any],
    admitted: Mapping[str, BenchmarkCausalSpan],
) -> tuple[BenchmarkCausalSpan | None, ProviderUsageQuarantineReason | None]:
    span_id = _optional_text_field(response, _SPAN_ID_KEYS)
    task_id = _optional_text_field(response, ("task_id",))
    if span_id:
        span = admitted.get(span_id)
        if span is None:
            return None, ProviderUsageQuarantineReason.UNBOUND_USAGE
        if task_id and span.task_id and task_id != span.task_id:
            return None, ProviderUsageQuarantineReason.MISSING_CAUSAL_IDENTITY
        return span, None
    if not task_id:
        return None, ProviderUsageQuarantineReason.MISSING_CAUSAL_IDENTITY
    matches = [span for span in admitted.values() if span.task_id == task_id]
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        return None, ProviderUsageQuarantineReason.UNBOUND_USAGE
    return None, ProviderUsageQuarantineReason.MISSING_CAUSAL_IDENTITY


def _reported_tokens(
    response: Mapping[str, Any],
) -> dict[str, int | None]:
    usage = _usage_mapping(response)
    sources: tuple[Mapping[str, Any], ...] = (usage, response)
    details_cached = _nested_mapping(usage, "prompt_tokens_details")
    details_reasoning = _nested_mapping(usage, "completion_tokens_details")
    output_details = _nested_mapping(usage, "output_tokens_details")
    input_details = _nested_mapping(usage, "input_tokens_details")

    def first(keys: Sequence[str], extra: Mapping[str, Any] | None = None) -> int | None:
        if extra:
            value = _lookup_reported_int(extra, keys, field=keys[0])
            if value is not None:
                return value
        for source in sources:
            value = _lookup_reported_int(source, keys, field=keys[0])
            if value is not None:
                return value
        return None

    return {
        "input": first(_INPUT_TOKEN_KEYS),
        "output": first(_OUTPUT_TOKEN_KEYS),
        "cached": first(_CACHED_TOKEN_KEYS, extra={**input_details, **details_cached}),
        "reasoning": first(
            _REASONING_TOKEN_KEYS, extra={**output_details, **details_reasoning}
        ),
    }


def _reported_charge(response: Mapping[str, Any]) -> int | None:
    usage = _usage_mapping(response)
    charge_blob = response.get("charge")
    if isinstance(charge_blob, Mapping):
        value = _lookup_reported_int(
            charge_blob,
            ("reported_microusd", "microusd", "value", *_CHARGE_KEYS),
            field="provider_reported_charge",
        )
        if value is not None:
            return value
        if any(key in charge_blob for key in ("usd", "cost_usd", "amount_usd")):
            raise BenchmarkTelemetryError(
                "provider_reported_charge must be integer microusd"
            )
    elif charge_blob is not None and not isinstance(charge_blob, Mapping):
        if isinstance(charge_blob, bool) or not isinstance(charge_blob, int):
            raise BenchmarkTelemetryError(
                "provider_reported_charge must be an integer"
            )
        return _integer(charge_blob, "provider_reported_charge")
    for source in (usage, response):
        value = _lookup_reported_int(source, _CHARGE_KEYS, field="provider_reported_charge")
        if value is not None:
            return value
        if any(key in source for key in ("cost_usd", "usd", "amount_usd")):
            raise BenchmarkTelemetryError(
                "provider_reported_charge must be integer microusd"
            )
    return None


def _provider_identity(
    response: Mapping[str, Any], span: BenchmarkCausalSpan | None
) -> tuple[str, str, str, str]:
    provider_id = _optional_text_field(
        response, ("provider_id", "provider")
    )
    model_id = _optional_text_field(response, ("model_id", "model"))
    model_revision = _optional_text_field(response, _MODEL_REVISION_KEYS)
    model_class = _optional_text_field(
        response, ("model_class", "call_class", "class")
    )
    binding = None if span is None else span.provider
    if binding is not None:
        provider_id = provider_id or binding.provider_id
        model_id = model_id or binding.model_id
        model_revision = model_revision or binding.model_revision
    if model_class and model_class not in MODEL_CLASSES:
        raise BenchmarkTelemetryError("model_class is not a closed value")
    return provider_id, model_id, model_revision, model_class


def _estimated_as_measured(response: Mapping[str, Any]) -> bool:
    if response.get("estimated_as_measured") is True:
        return True
    for key in ("usage", "charge", "estimate", "estimated_charge"):
        blob = response.get(key)
        if not isinstance(blob, Mapping):
            continue
        if blob.get("estimated_as_measured") is True:
            return True
        truth = blob.get("truth_state")
        source = str(blob.get("source") or blob.get("usage_source") or "")
        estimated_source = source in {
            "estimated",
            "estimate",
            "calibrated_fallback",
            "token_price_snapshot",
        }
        if estimated_source and (
            truth == "measured" or "sensor_id" in blob
        ):
            return True
        if truth == "estimated" and "sensor_id" in blob:
            return True
        if truth == "measured" and (
            "estimator_id" in blob or "method" in blob or estimated_source
        ):
            return True
    return False


def _observation_id_for(
    response: Mapping[str, Any],
    *,
    span_id: str,
    request_ids: Sequence[str],
    index: int,
) -> str:
    explicit = _optional_text_field(
        response, ("observation_id", "measurement_id", "usage_id")
    )
    if explicit:
        return explicit
    body = json.dumps(
        {
            "span_id": span_id,
            "request_ids": list(request_ids),
            "index": index,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f"provider-usage:{digest[:32]}"


@dataclass(frozen=True)
class ProviderUsageRecord(_TelemetryContract):
    """One provider response mapped onto an admitted causal task span."""

    SCHEMA: ClassVar[str] = PROVIDER_USAGE_RECORD_SCHEMA
    INTERFACE: ClassVar[str] = PROVIDER_USAGE_RECORD_INTERFACE

    observation_id: str
    disposition: ProviderUsageDisposition
    span_id: str = ""
    task_id: str = ""
    provider_id: str = ""
    model_id: str = ""
    model_revision: str = ""
    model_class: str = ""
    request_ids: tuple[str, ...] = ()
    samples: tuple[TelemetrySample, ...] = ()
    estimates: tuple[LabeledEstimate, ...] = ()
    calls_by_model_class: tuple[tuple[str, TelemetrySample], ...] = ()
    quarantine_reason: str = ""
    span: BenchmarkCausalSpan | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observation_id", _text(self.observation_id, "observation_id")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ProviderUsageDisposition, "disposition"),
        )
        for name in (
            "span_id",
            "task_id",
            "provider_id",
            "model_id",
            "model_revision",
            "model_class",
            "quarantine_reason",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        request_ids = tuple(
            _text(item, "request_id") for item in (self.request_ids or ())
        )
        if len(request_ids) != len(set(request_ids)):
            raise BenchmarkTelemetryError("request_ids contains duplicates")
        if len(request_ids) > MAX_SAMPLES:
            raise BenchmarkTelemetryError("request_ids exceeds bound")
        for item in request_ids:
            if extract_safe_request_id(item) != item:
                raise BenchmarkTelemetryError(
                    "request_ids must contain only safe opaque identifiers"
                )
        object.__setattr__(self, "request_ids", request_ids)

        samples: list[TelemetrySample] = []
        for item in self.samples or ():
            if isinstance(item, Mapping):
                samples.append(TelemetrySample.from_dict(item))
            elif isinstance(item, TelemetrySample):
                samples.append(item)
            else:
                raise BenchmarkTelemetryError(
                    "samples must be TelemetrySample records"
                )
        names = [item.metric_name for item in samples]
        if len(names) != len(set(names)):
            raise BenchmarkTelemetryError(
                "provider usage contains duplicate metric names"
            )
        missing = [
            name for name in PROVIDER_USAGE_SAMPLE_NAMES if name not in names
        ]
        if missing:
            raise BenchmarkTelemetryError(
                "provider usage is missing required samples: "
                + ", ".join(missing)
            )
        for sample in samples:
            if sample.status is SampleStatus.UNAVAILABLE and (
                sample.value != 0 or sample.unit
            ):
                raise BenchmarkTelemetryError(
                    "unavailable sample must not encode a numeric value or unit"
                )
        object.__setattr__(self, "samples", tuple(samples))

        estimates: list[LabeledEstimate] = []
        for item in self.estimates or ():
            if isinstance(item, Mapping):
                estimates.append(LabeledEstimate.from_dict(item))
            elif isinstance(item, LabeledEstimate):
                estimates.append(item)
            else:
                raise BenchmarkTelemetryError(
                    "estimates must be LabeledEstimate records"
                )
        object.__setattr__(self, "estimates", tuple(estimates))

        calls: list[tuple[str, TelemetrySample]] = []
        raw_calls = self.calls_by_model_class or ()
        if isinstance(raw_calls, Mapping):
            raw_items = raw_calls.items()
        else:
            raw_items = raw_calls
        for item in raw_items:
            if isinstance(item, Mapping) and "model_class" in item:
                class_name = _text(item.get("model_class", ""), "model_class")
                sample_payload = item.get("sample") or item
                sample = (
                    TelemetrySample.from_dict(sample_payload)
                    if isinstance(sample_payload, Mapping)
                    else sample_payload
                )
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                class_name = _text(item[0], "model_class")
                sample = item[1]
                if isinstance(sample, Mapping):
                    sample = TelemetrySample.from_dict(sample)
            else:
                raise BenchmarkTelemetryError(
                    "calls_by_model_class entries must be (class, sample) pairs"
                )
            if not isinstance(sample, TelemetrySample):
                raise BenchmarkTelemetryError(
                    "calls_by_model_class samples must be TelemetrySample records"
                )
            if class_name not in MODEL_CLASSES:
                raise BenchmarkTelemetryError("model_class is not a closed value")
            calls.append((class_name, sample))
        present = [name for name, _sample in calls]
        if set(present) != set(MODEL_CLASSES) or len(present) != len(MODEL_CLASSES):
            raise BenchmarkTelemetryError(
                "calls_by_model_class must cover every closed model class"
            )
        object.__setattr__(self, "calls_by_model_class", tuple(calls))

        span = self.span
        if isinstance(span, Mapping):
            span = BenchmarkCausalSpan.from_dict(span)
        if span is not None and not isinstance(span, BenchmarkCausalSpan):
            raise BenchmarkTelemetryError("span must be BenchmarkCausalSpan")
        object.__setattr__(self, "span", span)

        if self.disposition is ProviderUsageDisposition.ADMITTED:
            if self.quarantine_reason:
                raise BenchmarkTelemetryError(
                    "admitted provider usage cannot carry a quarantine reason"
                )
            if span is None or not self.span_id:
                raise BenchmarkTelemetryError(
                    "admitted provider usage requires a causal span"
                )
            if span.span_id != self.span_id:
                raise BenchmarkTelemetryError(
                    "span_id does not match the bound causal span"
                )
            if self.task_id and span.task_id and self.task_id != span.task_id:
                raise BenchmarkTelemetryError(
                    "task_id does not match the bound causal span"
                )
        else:
            if not self.quarantine_reason:
                raise BenchmarkTelemetryError(
                    "quarantined provider usage requires a quarantine_reason"
                )
            try:
                ProviderUsageQuarantineReason(self.quarantine_reason)
            except ValueError as exc:
                raise BenchmarkTelemetryError(
                    "quarantine_reason is not a supported quarantine reason"
                ) from exc

    @property
    def is_admitted(self) -> bool:
        return self.disposition is ProviderUsageDisposition.ADMITTED

    def sample(self, metric_name: str) -> TelemetrySample | None:
        for item in self.samples:
            if item.metric_name == metric_name:
                return item
        return None

    def require_sample(self, metric_name: str) -> TelemetrySample:
        item = self.sample(metric_name)
        if item is None:
            raise BenchmarkTelemetryError(
                f"provider usage is missing metric {metric_name!r}"
            )
        return item

    def call_class_sample(self, model_class: str) -> TelemetrySample:
        for name, sample in self.calls_by_model_class:
            if name == model_class:
                return sample
        raise BenchmarkTelemetryError(
            f"provider usage is missing call class {model_class!r}"
        )

    def unavailable_fields(self) -> tuple[str, ...]:
        names = [
            item.metric_name
            for item in self.samples
            if item.status is SampleStatus.UNAVAILABLE
        ]
        if not self.request_ids:
            names.append("safe_provider_request_ids")
        if not self.model_revision:
            names.append("model_revision")
        if not self.estimates:
            names.append("token_based_estimated_charge_microusd")
        for class_name, sample in self.calls_by_model_class:
            if sample.status is SampleStatus.UNAVAILABLE:
                names.append(f"calls_by_model_class.{class_name}")
        return tuple(names)

    def to_model_use_fragment(self) -> dict[str, Any]:
        """Efficiency-receipt model_use overlay. Quarantined records stay unavailable."""

        if not self.is_admitted:
            blank = receipt_unavailable("not_admitted")
            return {
                "input_tokens": dict(blank),
                "output_tokens": dict(blank),
                "cached_input_tokens": dict(blank),
                "reasoning_tokens": dict(blank),
                "number_of_calls": dict(blank),
                "calls_by_model_class": {
                    name: dict(blank) for name in MODEL_CLASSES
                },
                "safe_provider_request_ids": dict(blank),
                "provider_reported_usage": dict(blank),
            }

        quantities = {
            "input_tokens": _quantity_fragment(
                self.require_sample("provider_native_input_tokens")
            ),
            "output_tokens": _quantity_fragment(
                self.require_sample("provider_native_output_tokens")
            ),
            "cached_input_tokens": _quantity_fragment(
                self.require_sample("provider_native_cached_input_tokens")
            ),
            "reasoning_tokens": _quantity_fragment(
                self.require_sample("provider_native_reasoning_tokens")
            ),
            "number_of_calls": _quantity_fragment(
                self.require_sample("model_call_count")
            ),
        }
        classes = {
            name: _quantity_fragment(sample)
            for name, sample in self.calls_by_model_class
        }
        if self.request_ids:
            request_ids: dict[str, Any] = receipt_observed_request_ids(
                self.request_ids
            )
        else:
            request_ids = receipt_unavailable("not_reported")
        token_states = [
            quantities["input_tokens"],
            quantities["output_tokens"],
            quantities["cached_input_tokens"],
            quantities["reasoning_tokens"],
        ]
        if any(item.get("truth_state") == "measured" for item in token_states):
            reported = {
                "truth_state": "observed",
                "input_tokens": quantities["input_tokens"],
                "output_tokens": quantities["output_tokens"],
                "cached_input_tokens": quantities["cached_input_tokens"],
                "reasoning_tokens": quantities["reasoning_tokens"],
            }
        else:
            reported = receipt_unavailable("provider_omitted")
        return {
            **quantities,
            "calls_by_model_class": classes,
            "safe_provider_request_ids": request_ids,
            "provider_reported_usage": reported,
        }

    def to_cost_fragment(self) -> dict[str, Any]:
        if not self.is_admitted:
            blank = receipt_unavailable("not_admitted")
            return {
                "provider_reported_charge": dict(blank),
                "token_based_estimated_charge": dict(blank),
            }
        charge = _quantity_fragment(
            self.require_sample("provider_reported_charge_microusd")
        )
        if self.estimates:
            estimated = self.estimates[0].to_quantity()
        else:
            estimated = receipt_unavailable("not_reported")
        return {
            "provider_reported_charge": charge,
            "token_based_estimated_charge": estimated,
        }

    def to_provider_model_fragment(self) -> dict[str, Any]:
        revision = (
            receipt_observed_identity(self.model_revision)
            if self.model_revision
            else receipt_unavailable("not_reported")
        )
        if not self.provider_id or not self.model_id or self.model_class not in MODEL_CLASSES:
            raise BenchmarkTelemetryError(
                "provider model identity is incomplete"
            )
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "model_revision": revision,
            "model_class": self.model_class,
        }

    def to_resource_measurement(
        self, *, measurement_id: str | None = None
    ) -> BenchmarkResourceMeasurement:
        if not self.is_admitted or self.span is None:
            raise BenchmarkTelemetryError(
                "quarantined or unbound usage cannot be admitted onto a span"
            )
        return build_resource_measurement(
            measurement_id=measurement_id or f"meas:{self.observation_id}",
            span=self.span,
            samples=self.samples,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": BENCHMARK_TELEMETRY_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "observation_id": self.observation_id,
            "disposition": self.disposition.value,
            "span_id": self.span_id,
            "task_id": self.task_id,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_class": self.model_class,
            "request_ids": list(self.request_ids),
            "samples": [item.to_record() for item in self.samples],
            "estimates": [item.to_record() for item in self.estimates],
            "calls_by_model_class": [
                {"model_class": name, "sample": sample.to_record()}
                for name, sample in self.calls_by_model_class
            ],
            "quarantine_reason": self.quarantine_reason,
            "span": None if self.span is None else self.span.to_record(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderUsageRecord":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "interface",
            "observation_id",
            "disposition",
            "span_id",
            "task_id",
            "provider_id",
            "model_id",
            "model_revision",
            "model_class",
            "request_ids",
            "samples",
            "estimates",
            "calls_by_model_class",
            "quarantine_reason",
            "span",
            "content_id",
        }
        _closed(
            payload,
            schema=cls.SCHEMA,
            allowed=allowed,
            name="provider usage record",
        )
        calls_payload = payload.get("calls_by_model_class") or ()
        if isinstance(calls_payload, Mapping):
            calls: Any = tuple(calls_payload.items())
        else:
            calls = tuple(calls_payload)
        result = cls(
            observation_id=payload.get("observation_id", ""),
            disposition=payload.get("disposition", ""),
            span_id=payload.get("span_id", ""),
            task_id=payload.get("task_id", ""),
            provider_id=payload.get("provider_id", ""),
            model_id=payload.get("model_id", ""),
            model_revision=payload.get("model_revision", ""),
            model_class=payload.get("model_class", ""),
            request_ids=tuple(payload.get("request_ids") or ()),
            samples=tuple(payload.get("samples") or ()),
            estimates=tuple(payload.get("estimates") or ()),
            calls_by_model_class=calls,
            quarantine_reason=payload.get("quarantine_reason", ""),
            span=payload.get("span"),
        )
        if payload.get("interface", result.INTERFACE) != result.INTERFACE:
            raise BenchmarkTelemetryError("provider usage interface mismatch")
        _claim(payload, result.content_id, "content_id")
        return result


def _build_usage_samples(
    *,
    tokens: Mapping[str, int | None],
    charge: int | None,
    call_count: int | None,
    sensor_id: str,
    reason: UnavailableReason = UnavailableReason.PROVIDER_OMITTED,
) -> dict[str, TelemetrySample]:
    return {
        "provider_native_input_tokens": _sample_or_unavailable(
            "provider_native_input_tokens",
            tokens.get("input"),
            unit=UNIT_TOKENS,
            sensor_id=sensor_id,
            reason=reason,
        ),
        "provider_native_output_tokens": _sample_or_unavailable(
            "provider_native_output_tokens",
            tokens.get("output"),
            unit=UNIT_TOKENS,
            sensor_id=sensor_id,
            reason=reason,
        ),
        "provider_native_cached_input_tokens": _sample_or_unavailable(
            "provider_native_cached_input_tokens",
            tokens.get("cached"),
            unit=UNIT_TOKENS,
            sensor_id=sensor_id,
            reason=reason,
        ),
        "provider_native_reasoning_tokens": _sample_or_unavailable(
            "provider_native_reasoning_tokens",
            tokens.get("reasoning"),
            unit=UNIT_TOKENS,
            sensor_id=sensor_id,
            reason=reason,
        ),
        "model_call_count": _sample_or_unavailable(
            "model_call_count",
            call_count,
            unit=UNIT_COUNT,
            sensor_id=sensor_id,
            reason=reason,
        ),
        "provider_reported_charge_microusd": _sample_or_unavailable(
            "provider_reported_charge_microusd",
            charge,
            unit=UNIT_MICROUSD,
            sensor_id=sensor_id,
            reason=reason,
        ),
    }


def _quarantined_record(
    *,
    observation_id: str,
    reason: ProviderUsageQuarantineReason,
    span: BenchmarkCausalSpan | None = None,
    span_id: str = "",
    task_id: str = "",
    provider_id: str = "",
    model_id: str = "",
    model_revision: str = "",
    model_class: str = "",
) -> ProviderUsageRecord:
    sensor_id = _sensor_id("provider-usage", "quarantine", reason.value)
    samples = _blank_provider_usage_samples(sensor_id)
    classes = _calls_by_model_class_samples(
        model_class="",
        call_count=None,
        sensor_id=sensor_id,
    )
    return ProviderUsageRecord(
        observation_id=observation_id,
        disposition=ProviderUsageDisposition.QUARANTINED,
        span_id=span.span_id if span is not None else span_id,
        task_id=(span.task_id if span is not None else task_id),
        provider_id=provider_id,
        model_id=model_id,
        model_revision=model_revision,
        model_class=model_class,
        request_ids=(),
        samples=tuple(samples[name] for name in PROVIDER_USAGE_SAMPLE_NAMES),
        estimates=(),
        calls_by_model_class=tuple(
            (name, classes[name]) for name in MODEL_CLASSES
        ),
        quarantine_reason=reason.value,
        span=None,
    )


def map_provider_response_to_span(
    response: Mapping[str, Any],
    *,
    admitted_spans: (
        BenchmarkTelemetrySession
        | Mapping[str, BenchmarkCausalSpan]
        | Sequence[BenchmarkCausalSpan]
    ),
    observation_id: str | None = None,
    index: int = 0,
) -> ProviderUsageRecord:
    """Map one provider response onto an admitted causal task span.

    Unbound usage, credential leakage, estimated-as-measured data, and
    missing causal identity quarantine the record. Absent usage and cost
    fields become ``unavailable`` samples rather than numeric zero.
    """

    if not isinstance(response, Mapping):
        raise BenchmarkTelemetryError("provider response must be an object")
    admitted = _admitted_span_index(admitted_spans)
    span_hint = _optional_text_field(response, _SPAN_ID_KEYS)
    task_hint = _optional_text_field(response, ("task_id",))

    if _contains_credential_material(response):
        return _quarantined_record(
            observation_id=observation_id or f"provider-usage:quarantine:{index}",
            reason=ProviderUsageQuarantineReason.CREDENTIAL_LEAKAGE,
        )
    try:
        request_ids = _extract_request_ids(response)
    except BenchmarkTelemetryError as exc:
        if "credential" in str(exc):
            return _quarantined_record(
                observation_id=observation_id
                or f"provider-usage:quarantine:{index}",
                reason=ProviderUsageQuarantineReason.CREDENTIAL_LEAKAGE,
            )
        raise
    obs_id = observation_id or _observation_id_for(
        response, span_id=span_hint, request_ids=request_ids, index=index
    )
    if _estimated_as_measured(response):
        return _quarantined_record(
            observation_id=obs_id,
            reason=ProviderUsageQuarantineReason.ESTIMATED_AS_MEASURED,
            span_id=span_hint,
            task_id=task_hint,
        )

    span, quarantine = _resolve_admitted_span(response, admitted)
    if quarantine is not None:
        return _quarantined_record(
            observation_id=obs_id,
            reason=quarantine,
            span=span,
            span_id=span_hint,
            task_id=task_hint,
        )
    if span is None:
        return _quarantined_record(
            observation_id=obs_id,
            reason=ProviderUsageQuarantineReason.MISSING_CAUSAL_IDENTITY,
            span_id=span_hint,
            task_id=task_hint,
        )

    provider_id, model_id, model_revision, model_class = _provider_identity(
        response, span
    )
    sensor_id = _sensor_id(
        "provider-usage", span.span_id, provider_id or "unbound-provider"
    )
    tokens = _reported_tokens(response)
    charge = _reported_charge(response)
    try:
        estimate = _parse_labeled_estimate(response)
    except BenchmarkTelemetryError as exc:
        if "labeled measured" in str(exc):
            return _quarantined_record(
                observation_id=obs_id,
                reason=ProviderUsageQuarantineReason.ESTIMATED_AS_MEASURED,
                span=span,
                task_id=span.task_id,
                provider_id=provider_id,
                model_id=model_id,
                model_revision=model_revision,
                model_class=model_class,
            )
        raise
    samples = _build_usage_samples(
        tokens=tokens,
        charge=charge,
        call_count=1,
        sensor_id=sensor_id,
    )
    classes = _calls_by_model_class_samples(
        model_class=model_class,
        call_count=1 if model_class else None,
        sensor_id=sensor_id,
    )
    obs_id = observation_id or _observation_id_for(
        response,
        span_id=span.span_id,
        request_ids=request_ids,
        index=index,
    )
    return ProviderUsageRecord(
        observation_id=obs_id,
        disposition=ProviderUsageDisposition.ADMITTED,
        span_id=span.span_id,
        task_id=span.task_id,
        provider_id=provider_id,
        model_id=model_id,
        model_revision=model_revision,
        model_class=model_class,
        request_ids=request_ids,
        samples=tuple(samples[name] for name in PROVIDER_USAGE_SAMPLE_NAMES),
        estimates=() if estimate is None else (estimate,),
        calls_by_model_class=tuple(
            (name, classes[name]) for name in MODEL_CLASSES
        ),
        span=span,
    )


def map_provider_responses(
    responses: Sequence[Mapping[str, Any]],
    *,
    admitted_spans: (
        BenchmarkTelemetrySession
        | Mapping[str, BenchmarkCausalSpan]
        | Sequence[BenchmarkCausalSpan]
    ),
) -> tuple[ProviderUsageRecord, ...]:
    """Map a population of provider responses onto admitted task spans."""

    if not isinstance(responses, Sequence) or isinstance(responses, (str, bytes)):
        raise BenchmarkTelemetryError("provider responses must be a sequence")
    records = []
    for index, response in enumerate(responses):
        records.append(
            map_provider_response_to_span(
                response, admitted_spans=admitted_spans, index=index
            )
        )
    return tuple(records)


def aggregate_provider_usage_for_span(
    records: Sequence[ProviderUsageRecord],
    span: BenchmarkCausalSpan,
    *,
    observation_id: str | None = None,
) -> ProviderUsageRecord:
    """Aggregate admitted records for one span without filling omissions as zero."""

    if not isinstance(span, BenchmarkCausalSpan):
        raise BenchmarkTelemetryError("span must be BenchmarkCausalSpan")
    admitted = [
        record
        for record in records
        if record.is_admitted and record.span_id == span.span_id
    ]
    sensor_id = _sensor_id("provider-usage", "aggregate", span.span_id)
    if not admitted:
        samples = _blank_provider_usage_samples(
            sensor_id, UnavailableReason.PROVIDER_OMITTED
        )
        classes = _calls_by_model_class_samples(
            model_class="", call_count=None, sensor_id=sensor_id
        )
        return ProviderUsageRecord(
            observation_id=observation_id
            or f"provider-usage-aggregate:{span.span_id}",
            disposition=ProviderUsageDisposition.ADMITTED,
            span_id=span.span_id,
            task_id=span.task_id,
            provider_id=(span.provider.provider_id if span.provider else ""),
            model_id=(span.provider.model_id if span.provider else ""),
            model_revision=(
                span.provider.model_revision if span.provider else ""
            ),
            model_class="",
            request_ids=(),
            samples=tuple(samples[name] for name in PROVIDER_USAGE_SAMPLE_NAMES),
            estimates=(),
            calls_by_model_class=tuple(
                (name, classes[name]) for name in MODEL_CLASSES
            ),
            span=span,
        )

    def _sum_or_unavailable(metric_name: str, unit: str) -> TelemetrySample:
        values: list[int] = []
        for record in admitted:
            sample = record.require_sample(metric_name)
            if sample.status is SampleStatus.UNAVAILABLE:
                return TelemetrySample.unavailable(
                    metric_name,
                    UnavailableReason.PROVIDER_OMITTED,
                    sensor_id=sensor_id,
                )
            values.append(sample.value)
        return TelemetrySample.measured(
            metric_name, sum(values), unit=unit, sensor_id=sensor_id
        )

    samples = {
        "provider_native_input_tokens": _sum_or_unavailable(
            "provider_native_input_tokens", UNIT_TOKENS
        ),
        "provider_native_output_tokens": _sum_or_unavailable(
            "provider_native_output_tokens", UNIT_TOKENS
        ),
        "provider_native_cached_input_tokens": _sum_or_unavailable(
            "provider_native_cached_input_tokens", UNIT_TOKENS
        ),
        "provider_native_reasoning_tokens": _sum_or_unavailable(
            "provider_native_reasoning_tokens", UNIT_TOKENS
        ),
        "model_call_count": TelemetrySample.measured(
            "model_call_count",
            len(admitted),
            unit=UNIT_COUNT,
            sensor_id=sensor_id,
        ),
        "provider_reported_charge_microusd": _sum_or_unavailable(
            "provider_reported_charge_microusd", UNIT_MICROUSD
        ),
    }

    class_counts: dict[str, int | None] = {name: 0 for name in MODEL_CLASSES}
    any_class = False
    for record in admitted:
        if not record.model_class:
            class_counts = {name: None for name in MODEL_CLASSES}
            break
        any_class = True
        class_counts[record.model_class] = (
            class_counts.get(record.model_class) or 0
        ) + 1
    if not any_class:
        classes = _calls_by_model_class_samples(
            model_class="", call_count=None, sensor_id=sensor_id
        )
    else:
        classes = {}
        for name in MODEL_CLASSES:
            count = class_counts[name]
            if count is None:
                classes[name] = TelemetrySample.unavailable(
                    f"calls_by_model_class.{name}",
                    UnavailableReason.PROVIDER_OMITTED,
                    sensor_id=sensor_id,
                )
            else:
                classes[name] = TelemetrySample.measured(
                    f"calls_by_model_class.{name}",
                    count,
                    unit=UNIT_COUNT,
                    sensor_id=sensor_id,
                )

    request_ids: list[str] = []
    for record in admitted:
        for item in record.request_ids:
            if item not in request_ids:
                request_ids.append(item)
    estimates = tuple(
        estimate for record in admitted for estimate in record.estimates
    )
    provider_ids = {record.provider_id for record in admitted if record.provider_id}
    model_ids = {record.model_id for record in admitted if record.model_id}
    revisions = {
        record.model_revision for record in admitted if record.model_revision
    }
    classes_seen = {
        record.model_class for record in admitted if record.model_class
    }
    return ProviderUsageRecord(
        observation_id=observation_id
        or f"provider-usage-aggregate:{span.span_id}",
        disposition=ProviderUsageDisposition.ADMITTED,
        span_id=span.span_id,
        task_id=span.task_id,
        provider_id=next(iter(provider_ids), ""),
        model_id=next(iter(model_ids), ""),
        model_revision=next(iter(revisions), ""),
        model_class=next(iter(classes_seen), ""),
        request_ids=tuple(request_ids),
        samples=tuple(samples[name] for name in PROVIDER_USAGE_SAMPLE_NAMES),
        estimates=estimates[:1],
        calls_by_model_class=tuple(
            (name, classes[name]) for name in MODEL_CLASSES
        ),
        span=span,
    )


def admit_provider_usage_into_session(
    session: BenchmarkTelemetrySession,
    responses: Sequence[Mapping[str, Any]],
) -> tuple[ProviderUsageRecord, ...]:
    """Record only admitted provider usage onto session spans."""

    if not isinstance(session, BenchmarkTelemetrySession):
        raise BenchmarkTelemetryError("session must be BenchmarkTelemetrySession")
    records = map_provider_responses(responses, admitted_spans=session)
    for record in records:
        if record.is_admitted:
            session.record_measurement(record.to_resource_measurement())
    return records


def mono_ns() -> int:
    """Monotonic nanoseconds for span bounds."""

    return time.monotonic_ns()


def observe_wall_seconds_millionths(started_mono_ns: int, finished_mono_ns: int) -> int:
    if finished_mono_ns < started_mono_ns:
        raise BenchmarkTelemetryError("finished precedes started")
    return (finished_mono_ns - started_mono_ns) // 1_000


__all__ = [
    "AttributionRole",
    "BENCHMARK_CAUSAL_SPAN_INTERFACE",
    "BENCHMARK_CAUSAL_SPAN_SCHEMA",
    "BENCHMARK_RESOURCE_MEASUREMENT_INTERFACE",
    "BENCHMARK_RESOURCE_MEASUREMENT_SCHEMA",
    "BENCHMARK_TELEMETRY_CONTRACT_VERSION",
    "BENCHMARK_TELEMETRY_RECEIPT_SCHEMA",
    "BenchmarkCausalSpan",
    "BenchmarkHardwareProfile",
    "BenchmarkProviderBinding",
    "BenchmarkResourceMeasurement",
    "BenchmarkTelemetryError",
    "BenchmarkTelemetryReceipt",
    "BenchmarkTelemetrySession",
    "CLOCK_METRIC_NAMES",
    "GPU_METRIC_NAMES",
    "LABELED_ESTIMATE_SCHEMA",
    "LabeledEstimate",
    "MILLIONTHS",
    "PROCESS_TREE_METRIC_NAMES",
    "PROVIDER_USAGE_RECORD_INTERFACE",
    "PROVIDER_USAGE_RECORD_SCHEMA",
    "PROVIDER_USAGE_SAMPLE_NAMES",
    "ProviderUsageDisposition",
    "ProviderUsageQuarantineReason",
    "ProviderUsageRecord",
    "SCHEMA_VERSION",
    "SPAN_REPLAY_CERTIFICATE_SCHEMA",
    "SampleStatus",
    "SpanKind",
    "SpanReplayCertificate",
    "TOKEN_METRIC_NAMES",
    "TELEMETRY_SAMPLE_SCHEMA",
    "TelemetrySample",
    "UNIT_BYTES",
    "UNIT_COUNT",
    "UNIT_GIB_SECONDS_MILLIONTHS",
    "UNIT_IDENTITY",
    "UNIT_JOULES_MILLIONTHS",
    "UNIT_MICROUSD",
    "UNIT_QUOTA",
    "UNIT_RATIO_MILLIONTHS",
    "UNIT_SECONDS_MILLIONTHS",
    "UNIT_TOKENS",
    "UnavailableReason",
    "admit_provider_usage_into_session",
    "aggregate_provider_usage_for_span",
    "build_resource_measurement",
    "build_span_joined_measurement",
    "certify_measurement_from_source_spans",
    "collect_descendant_pids",
    "extract_safe_request_id",
    "map_provider_response_to_span",
    "map_provider_responses",
    "mono_ns",
    "observe_wall_seconds_millionths",
    "project_scheduler_clock_samples",
    "project_token_ledger_samples",
    "reject_self_certified_counters",
    "sample_energy_optional",
    "sample_gpu_resources",
    "sample_network_bytes",
    "sample_process_tree_resources",
    "sample_rusage_self",
    "sample_self_process_tree_resources",
    "seconds_to_millionths",
]
