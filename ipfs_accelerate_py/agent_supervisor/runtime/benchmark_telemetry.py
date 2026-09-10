"""Benchmark causal-span telemetry for Planner/Doctor live runs.

This module attributes wall-clock, provider tokens, process-tree resources,
GPU, I/O, network, cost, and work (tests, proofs, retries, rescue, merge,
human action, validation, and patch disposition, including audit overhead)
to immutable causal spans.  It joins existing scheduler metrics and the
supervisor token ledger by span identity rather than replacing capacity
admission or inventing zero observations.

Contract rules:

* Every sample is either ``measured`` (value + unit + sensor receipt) or
  ``unavailable`` (reason code + sensor identity).  Unavailable is never
  encoded as a numeric zero.
* Measured zeros still require a sensor receipt.
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
from collections.abc import Container, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Final, Optional

from ..proof.formal_verification_contracts import CanonicalContract


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
PROVIDER_USAGE_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/benchmark-provider-usage-record@1"
)
PROVIDER_USAGE_RECORD_INTERFACE: Final[str] = "BenchmarkProviderUsageRecord@1"
OPERATION_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/benchmark-operation-observation@1"
)
WORK_TELEMETRY_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/benchmark-work-telemetry-record@1"
)
WORK_TELEMETRY_RECORD_INTERFACE: Final[str] = "BenchmarkWorkTelemetryRecord@1"

CID_RE: Final[re.Pattern[str]] = re.compile(r"^b[a-z2-7]{20,}$")
MAX_CID_BYTES: Final[int] = 128
MAX_VERIFIERS: Final[int] = 256

MAX_TEXT_BYTES: Final[int] = 512
MAX_SAMPLES: Final[int] = 10_000
MAX_CHILDREN: Final[int] = 100_000
MAX_SPAN_DEPTH: Final[int] = 256
MAX_INTEGER: Final[int] = 10**18
MAX_REQUEST_IDS: Final[int] = 256
MAX_USAGE_SCAN_DEPTH: Final[int] = 8
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

PROVIDER_USAGE_METRIC_NAMES: Final[tuple[str, ...]] = (
    "provider_native_input_tokens",
    "provider_native_output_tokens",
    "provider_native_cached_input_tokens",
    "provider_native_reasoning_tokens",
    "model_call_count",
    "provider_model_revision_identity",
    "safe_provider_request_id_count",
    "provider_reported_charge_microusd",
    "token_based_estimated_charge_microusd",
    "calls_deterministic",
    "calls_local_small_model",
    "calls_local_medium_model",
    "calls_remote_standard_model",
    "calls_remote_frontier_model",
    "calls_human",
)

MODEL_CALL_CLASSES: Final[tuple[str, ...]] = (
    "deterministic",
    "local_small_model",
    "local_medium_model",
    "remote_standard_model",
    "remote_frontier_model",
    "human",
)

ESTIMATOR_METHODS: Final[tuple[str, ...]] = (
    "token_price_snapshot",
    "provider_quote",
    "local_compute_model",
    "manual_audit",
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

COMPUTE_FIELD_NAMES: Final[tuple[str, ...]] = (
    "cpu_seconds",
    "gpu_seconds",
    "peak_memory",
    "wall_clock_duration",
    "test_execution_time",
    "prover_execution_time",
    "static_analysis_time",
    "indexing_retrieval_time",
    "bytes_read",
    "bytes_written",
    "process_count",
    "concurrency",
)

WORK_OPERATION_FIELD_NAMES: Final[tuple[str, ...]] = (
    "tests_selected",
    "tests_executed",
    "full_suite_tests",
    "type_static_schema_checks",
    "proof_obligations_selected",
    "proof_obligations_executed",
    "proof_receipts_reused",
    "retries",
    "rescue_attempts",
    "merge_conflicts",
    "manual_recovery",
    "human_interventions",
    "validation_result",
)

WORK_ENUM_FIELD_NAMES: Final[tuple[str, ...]] = (
    "final_task_outcome",
    "patch_disposition",
)

WORK_FIELD_NAMES: Final[tuple[str, ...]] = (
    WORK_OPERATION_FIELD_NAMES + WORK_ENUM_FIELD_NAMES
)

WORK_OVERHEAD_FIELD_NAMES: Final[tuple[str, ...]] = (
    "audit_and_verification_overhead",
    "time_to_terminal_outcome",
)

WORK_METRIC_NAMES: Final[tuple[str, ...]] = (
    COMPUTE_FIELD_NAMES + WORK_OPERATION_FIELD_NAMES + WORK_OVERHEAD_FIELD_NAMES
)

_PROCESS_TREE_TO_COMPUTE: Final[dict[str, str]] = {
    "total_cpu_seconds": "cpu_seconds",
    "peak_rss_bytes": "peak_memory",
    "read_bytes": "bytes_read",
    "write_bytes": "bytes_written",
    "peak_process_count": "process_count",
}

_WORK_FIELD_SPAN_KINDS: Final[dict[str, tuple[str, ...]]] = {
    "cpu_seconds": ("process", "task", "attempt"),
    "gpu_seconds": ("process", "task", "attempt"),
    "peak_memory": ("process", "task", "attempt"),
    "wall_clock_duration": ("task", "attempt", "run"),
    "test_execution_time": ("check", "validation"),
    "prover_execution_time": ("proof",),
    "static_analysis_time": ("check",),
    "indexing_retrieval_time": ("check", "task"),
    "bytes_read": ("process", "task"),
    "bytes_written": ("process", "task"),
    "process_count": ("process", "task"),
    "concurrency": ("task", "run"),
    "tests_selected": ("check", "validation"),
    "tests_executed": ("check", "validation"),
    "full_suite_tests": ("check", "validation"),
    "type_static_schema_checks": ("check",),
    "proof_obligations_selected": ("proof",),
    "proof_obligations_executed": ("proof",),
    "proof_receipts_reused": ("proof",),
    "retries": ("retry",),
    "rescue_attempts": ("rescue",),
    "merge_conflicts": ("merge",),
    "manual_recovery": ("recovery",),
    "human_interventions": ("human",),
    "validation_result": ("validation",),
    "audit_and_verification_overhead": ("validation", "check", "proof", "sensor"),
    "time_to_terminal_outcome": ("task", "attempt", "run"),
}

_SELECTED_EXECUTED_BOUNDS: Final[tuple[tuple[str, str], ...]] = (
    ("tests_selected", "tests_executed"),
    ("proof_obligations_selected", "proof_obligations_executed"),
    ("proof_obligations_selected", "proof_receipts_reused"),
)

OPERATION_REASON_CODES: Final[tuple[str, ...]] = (
    "not_reported",
    "sensor_absent",
    "provider_omitted",
    "collection_failed",
    "permission_denied",
    "hardware_absent",
    "not_applicable",
    "not_yet_measured",
    "fixture_only",
    "deadline_elapsed",
    "not_sealed",
    "not_admitted",
)

TASK_OUTCOMES: Final[tuple[str, ...]] = (
    "succeeded",
    "failed",
    "retried",
    "rescued",
    "conflicted",
    "human_escalated",
    "quarantined",
    "compensated",
)

PATCH_DISPOSITIONS: Final[tuple[str, ...]] = (
    "accepted",
    "rejected",
    "quarantined",
    "reverted",
)

VERIFICATION_WORK_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "tests_executed",
        "full_suite_tests",
        "type_static_schema_checks",
        "proof_obligations_executed",
        "proof_receipts_reused",
        "validation_result",
    }
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
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


class UnavailableReason(str, Enum):
    SENSOR_ABSENT = "sensor-absent"
    PERMISSION_DENIED = "permission-denied"
    HARDWARE_ABSENT = "hardware-absent"
    PROVIDER_OMITTED = "provider-omitted"
    COLLECTION_FAILED = "collection-failed"
    NOT_REPORTED = "not-reported"
    NOT_ADMITTED = "not-admitted"
    NOT_APPLICABLE = "not-applicable"
    NOT_YET_MEASURED = "not-yet-measured"
    FIXTURE_ONLY = "fixture-only"
    DEADLINE_ELAPSED = "deadline-elapsed"
    NOT_SEALED = "not-sealed"


class EstimatorMethod(str, Enum):
    TOKEN_PRICE_SNAPSHOT = "token_price_snapshot"
    PROVIDER_QUOTE = "provider_quote"
    LOCAL_COMPUTE_MODEL = "local_compute_model"
    MANUAL_AUDIT = "manual_audit"


class ModelCallClass(str, Enum):
    DETERMINISTIC = "deterministic"
    LOCAL_SMALL_MODEL = "local_small_model"
    LOCAL_MEDIUM_MODEL = "local_medium_model"
    REMOTE_STANDARD_MODEL = "remote_standard_model"
    REMOTE_FRONTIER_MODEL = "remote_frontier_model"
    HUMAN = "human"


class ProviderUsageDisposition(str, Enum):
    ADMITTED = "admitted"
    QUARANTINED = "quarantined"


class ProviderUsageQuarantineReason(str, Enum):
    UNBOUND_USAGE = "unbound_usage"
    CREDENTIAL_LEAKAGE = "credential_leakage"
    ESTIMATED_AS_MEASURED = "estimated_as_measured"
    MISSING_CAUSAL_IDENTITY = "missing_causal_identity"


class OperationTruthState(str, Enum):
    ATTEMPTED = "attempted"
    OBSERVED = "observed"
    VERIFIED = "verified"
    UNAVAILABLE = "unavailable"
    SIMULATED = "simulated"


class TaskOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRIED = "retried"
    RESCUED = "rescued"
    CONFLICTED = "conflicted"
    HUMAN_ESCALATED = "human_escalated"
    QUARANTINED = "quarantined"
    COMPENSATED = "compensated"


class PatchDisposition(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"
    REVERTED = "reverted"


class WorkTelemetryDisposition(str, Enum):
    ADMITTED = "admitted"
    QUARANTINED = "quarantined"


class WorkTelemetryQuarantineReason(str, Enum):
    UNBOUND_WORK = "unbound_work"
    MISSING_CAUSAL_IDENTITY = "missing_causal_identity"
    ATTEMPTED_AS_OBSERVED = "attempted_as_observed"
    OBSERVED_AS_VERIFIED = "observed_as_verified"
    MISSING_OVERHEAD = "missing_overhead"
    DUPLICATE_TERMINAL_ACCOUNTING = "duplicate_terminal_accounting"
    INVALID_BOUNDS = "invalid_bounds"


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
    CHECK = "check"
    PROOF = "proof"
    VALIDATION = "validation"
    HUMAN = "human"
    RESCUE = "rescue"
    RECOVERY = "recovery"


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
    HUMAN = "human"
    VERIFIER = "verifier"


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
    """One metric observation: measured, labeled-estimated, or unavailable."""

    SCHEMA: ClassVar[str] = TELEMETRY_SAMPLE_SCHEMA

    metric_name: str
    status: SampleStatus
    sensor_id: str
    unit: str = ""
    value: int = 0
    reason_code: str = ""
    estimator_id: str = ""
    method: str = ""
    price_snapshot_identity: str = ""

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
        object.__setattr__(
            self,
            "estimator_id",
            _text(self.estimator_id, "estimator_id", required=False),
        )
        object.__setattr__(
            self, "method", _text(self.method, "method", required=False)
        )
        object.__setattr__(
            self,
            "price_snapshot_identity",
            _text(
                self.price_snapshot_identity,
                "price_snapshot_identity",
                required=False,
            ),
        )
        estimator_present = bool(
            self.estimator_id or self.method or self.price_snapshot_identity
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
            if estimator_present:
                raise BenchmarkTelemetryError(
                    "measured sample cannot carry estimator labels"
                )
        elif self.status is SampleStatus.ESTIMATED:
            if not self.unit:
                raise BenchmarkTelemetryError(
                    "estimated sample requires a unit"
                )
            if self.reason_code:
                raise BenchmarkTelemetryError(
                    "estimated sample cannot carry an unavailable reason"
                )
            if not self.estimator_id or not self.method:
                raise BenchmarkTelemetryError(
                    "estimated sample requires estimator_id and method"
                )
            try:
                EstimatorMethod(self.method)
            except ValueError as exc:
                raise BenchmarkTelemetryError(
                    "method is not a supported estimator method"
                ) from exc
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
            if estimator_present:
                raise BenchmarkTelemetryError(
                    "unavailable sample must not encode estimator labels"
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
    def estimated(
        cls,
        metric_name: str,
        value: int,
        *,
        unit: str,
        estimator_id: str,
        method: EstimatorMethod | str,
        price_snapshot_identity: str = "unavailable",
        sensor_id: str | None = None,
    ) -> "TelemetrySample":
        method_code = (
            method.value if isinstance(method, EstimatorMethod) else str(method)
        )
        return cls(
            metric_name=metric_name,
            status=SampleStatus.ESTIMATED,
            sensor_id=sensor_id or estimator_id,
            unit=unit,
            value=_integer(value, "value"),
            estimator_id=estimator_id,
            method=method_code,
            price_snapshot_identity=price_snapshot_identity,
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
        elif self.status is SampleStatus.ESTIMATED:
            payload["unit"] = self.unit
            payload["value"] = self.value
            payload["estimator_id"] = self.estimator_id
            payload["method"] = self.method
            payload["price_snapshot_identity"] = self.price_snapshot_identity
        else:
            payload["reason_code"] = self.reason_code
        return payload

    def to_envelope(self) -> dict[str, Any]:
        """Telemetry contract sample envelope (measured | estimated | unavailable)."""

        if self.status is SampleStatus.MEASURED:
            return {
                "status": SampleStatus.MEASURED.value,
                "value": self.value,
                "unit": self.unit,
                "sensor_id": self.sensor_id,
            }
        if self.status is SampleStatus.ESTIMATED:
            return {
                "status": SampleStatus.ESTIMATED.value,
                "value": self.value,
                "unit": self.unit,
                "estimator_id": self.estimator_id,
                "method": self.method,
                "price_snapshot_identity": self.price_snapshot_identity,
            }
        return {
            "status": SampleStatus.UNAVAILABLE.value,
            "reason_code": self.reason_code,
            "sensor_id": self.sensor_id,
        }

    def to_quantity_envelope(self) -> dict[str, Any]:
        """Efficiency-receipt quantity envelope; unavailable never encodes zero."""

        if self.status is SampleStatus.MEASURED:
            return {
                "truth_state": SampleStatus.MEASURED.value,
                "value": self.value,
                "unit": self.unit,
                "sensor_id": self.sensor_id,
            }
        if self.status is SampleStatus.ESTIMATED:
            return {
                "truth_state": SampleStatus.ESTIMATED.value,
                "value": self.value,
                "unit": self.unit,
                "estimator_id": self.estimator_id,
                "method": self.method,
                "price_snapshot_identity": self.price_snapshot_identity,
            }
        return {
            "truth_state": SampleStatus.UNAVAILABLE.value,
            "reason_code": self.reason_code.replace("-", "_"),
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
            "estimator_id",
            "method",
            "price_snapshot_identity",
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
        elif status is SampleStatus.ESTIMATED:
            result = cls.estimated(
                str(payload.get("metric_name", "")),
                int(payload.get("value", 0)),
                unit=str(payload.get("unit", "")),
                estimator_id=str(payload.get("estimator_id", "")),
                method=str(payload.get("method", "")),
                price_snapshot_identity=str(
                    payload.get("price_snapshot_identity", "unavailable")
                ),
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
        self._admitted_usage_span_ids: set[str] = set()
        self._provider_usage: dict[str, ProviderUsageRecord] = {}
        self._admitted_verifiers: dict[str, str] = {}
        self._work_telemetry: dict[str, WorkTelemetryRecord] = {}
        self._terminalized_span_ids: set[str] = set()
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

    @property
    def admitted_usage_span_ids(self) -> frozenset[str]:
        return frozenset(self._admitted_usage_span_ids)

    @property
    def provider_usage(self) -> tuple["ProviderUsageRecord", ...]:
        return tuple(
            self._provider_usage[key] for key in sorted(self._provider_usage)
        )

    @property
    def admitted_task_span_ids(self) -> frozenset[str]:
        return self.admitted_usage_span_ids

    @property
    def admitted_verifiers(self) -> dict[str, str]:
        return dict(self._admitted_verifiers)

    @property
    def work_telemetry(self) -> tuple["WorkTelemetryRecord", ...]:
        return tuple(
            self._work_telemetry[key] for key in sorted(self._work_telemetry)
        )

    @property
    def terminalized_span_ids(self) -> frozenset[str]:
        return frozenset(self._terminalized_span_ids)

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

    def admit_task_span(self, span: BenchmarkCausalSpan) -> BenchmarkCausalSpan:
        """Admit a causal task span as the only legal usage/work binding."""

        registered = self.register_span(span)
        if not registered.task_id:
            raise BenchmarkTelemetryError(
                "task span is missing causal task identity"
            )
        self._admitted_usage_span_ids.add(registered.span_id)
        return registered

    def admit_verifier(
        self, verifier_id: str, verifier_receipt_cid: str
    ) -> tuple[str, str]:
        """Admit a verifier identity and receipt CID for work-field linkage."""

        verifier_id = _text(verifier_id, "verifier_id")
        cid = _require_cid(verifier_receipt_cid, "verifier_receipt_cid")
        if len(self._admitted_verifiers) >= MAX_VERIFIERS and (
            verifier_id not in self._admitted_verifiers
        ):
            raise BenchmarkTelemetryError("admitted verifier population exceeds bound")
        existing = self._admitted_verifiers.get(verifier_id)
        if existing is not None and existing != cid:
            raise BenchmarkTelemetryError(
                f"verifier {verifier_id!r} is already admitted with a different receipt"
            )
        self._admitted_verifiers[verifier_id] = cid
        return verifier_id, cid

    def record_provider_response(
        self,
        response: Mapping[str, Any],
        *,
        span: BenchmarkCausalSpan,
        record_id: str | None = None,
        labeled_estimate: Mapping[str, Any] | None = None,
        model_class: ModelCallClass | str | None = None,
    ) -> "ProviderUsageRecord":
        """Map one provider response onto an admitted task span.

        Quarantined records are retained but never sealed as measured usage.
        """

        record = map_provider_response_to_admitted_span(
            response,
            span,
            session=self,
            record_id=record_id,
            labeled_estimate=labeled_estimate,
            model_class=model_class,
        )
        if record.record_id in self._provider_usage:
            prior = self._provider_usage[record.record_id]
            if prior.content_id != record.content_id:
                raise BenchmarkTelemetryError(
                    "provider usage record_id collides with a different body"
                )
            return prior
        self._provider_usage[record.record_id] = record
        if record.disposition is ProviderUsageDisposition.ADMITTED:
            self.record_measurement(record.to_resource_measurement())
        return record

    def record_work_and_compute(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        span: BenchmarkCausalSpan,
        record_id: str | None = None,
        compute: Mapping[str, Any] | None = None,
        work: Mapping[str, Any] | None = None,
        audit_and_verification_overhead: Mapping[str, Any]
        | TelemetrySample
        | None = None,
        time_to_terminal_outcome: Mapping[str, Any] | TelemetrySample | None = None,
        process_samples: Mapping[str, TelemetrySample] | None = None,
        gpu_samples: Mapping[str, TelemetrySample] | None = None,
        terminalized: bool | None = None,
    ) -> "WorkTelemetryRecord":
        """Map work/compute fields onto an admitted causal task span.

        Attempted-as-observed, observed-as-verified, missing audit overhead,
        duplicate terminal accounting, and invalid bounds fail closed.
        """

        record = map_work_and_compute_to_admitted_span(
            payload,
            span,
            session=self,
            record_id=record_id,
            compute=compute,
            work=work,
            audit_and_verification_overhead=audit_and_verification_overhead,
            time_to_terminal_outcome=time_to_terminal_outcome,
            process_samples=process_samples,
            gpu_samples=gpu_samples,
            terminalized=terminalized,
        )
        if record.record_id in self._work_telemetry:
            prior = self._work_telemetry[record.record_id]
            if prior.content_id != record.content_id:
                raise BenchmarkTelemetryError(
                    "work telemetry record_id collides with a different body"
                )
            return prior
        if (
            record.disposition is WorkTelemetryDisposition.ADMITTED
            and record.terminalized
        ):
            if span.span_id in self._terminalized_span_ids:
                raise BenchmarkTelemetryError(
                    "duplicate terminal accounting for admitted work span"
                )
            self._terminalized_span_ids.add(span.span_id)
        self._work_telemetry[record.record_id] = record
        if record.disposition is WorkTelemetryDisposition.ADMITTED:
            self.record_measurement(record.to_resource_measurement())
        return record

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


def mono_ns() -> int:
    """Monotonic nanoseconds for span bounds."""

    return time.monotonic_ns()


def observe_wall_seconds_millionths(started_mono_ns: int, finished_mono_ns: int) -> int:
    if finished_mono_ns < started_mono_ns:
        raise BenchmarkTelemetryError("finished precedes started")
    return (finished_mono_ns - started_mono_ns) // 1_000


# ---------------------------------------------------------------------------
# Provider/model usage mapped onto admitted causal task spans (ASEH-011)
# ---------------------------------------------------------------------------

_CREDENTIAL_KEY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "auth_token",
        "bearer",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "passphrase",
        "private_key",
        "refresh_token",
        "secret",
        "secrets",
        "secret_handle",
        "session_token",
        "x-api-key",
        "x_api_key",
    }
)

_UNSAFE_REQUEST_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(sk-|bearer\s+|api[_-]?key|secret|token=|-----BEGIN)"
)
_JWTISH_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}$"
)
_SAFE_REQUEST_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$"
)

_INPUT_TOKEN_KEYS: Final[tuple[str, ...]] = (
    "input_tokens",
    "prompt_tokens",
    "inputTokenCount",
    "prompt_token_count",
)
_OUTPUT_TOKEN_KEYS: Final[tuple[str, ...]] = (
    "output_tokens",
    "completion_tokens",
    "outputTokenCount",
    "completion_token_count",
)
_CACHED_TOKEN_KEYS: Final[tuple[str, ...]] = (
    "cached_input_tokens",
    "cached_tokens",
    "cache_read_input_tokens",
    "prompt_cache_hit_tokens",
    "cache_read_tokens",
)
_REASONING_TOKEN_KEYS: Final[tuple[str, ...]] = (
    "reasoning_tokens",
    "thinking_tokens",
    "thought_tokens",
)
_CHARGE_MICROUSD_KEYS: Final[tuple[str, ...]] = (
    "provider_reported_charge_microusd",
    "reported_charge_microusd",
    "charge_microusd",
    "cost_microusd",
    "cost_microunits",
)
_REQUEST_ID_KEYS: Final[tuple[str, ...]] = (
    "id",
    "request_id",
    "requestId",
    "x-request-id",
    "x_request_id",
    "response_id",
)
_CALL_CLASS_METRIC: Final[dict[str, str]] = {
    "deterministic": "calls_deterministic",
    "local_small_model": "calls_local_small_model",
    "local_medium_model": "calls_local_medium_model",
    "remote_standard_model": "calls_remote_standard_model",
    "remote_frontier_model": "calls_remote_frontier_model",
    "human": "calls_human",
}


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _walk_mappings(
    payload: Any, *, depth: int = 0
) -> Iterable[Mapping[str, Any]]:
    if depth > MAX_USAGE_SCAN_DEPTH or not isinstance(payload, Mapping):
        return
    yield payload
    for child in payload.values():
        if isinstance(child, Mapping):
            yield from _walk_mappings(child, depth=depth + 1)
        elif isinstance(child, Sequence) and not isinstance(
            child, (str, bytes, bytearray)
        ):
            for item in child:
                if isinstance(item, Mapping):
                    yield from _walk_mappings(item, depth=depth + 1)


def _key_looks_like_credential(key: str) -> bool:
    normalized = _normalize_key(key)
    if normalized in _CREDENTIAL_KEY_MARKERS:
        return True
    return normalized.endswith(("_secret", "_api_key", "_private_key", "_password"))


def _value_looks_like_credential(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if not candidate:
        return False
    if "BEGIN" in candidate.upper() and "PRIVATE" in candidate.upper():
        return True
    if _JWTISH_RE.match(candidate):
        return True
    return bool(_UNSAFE_REQUEST_ID_RE.search(candidate))


def provider_response_contains_credentials(payload: Any, *, depth: int = 0) -> bool:
    """True when a provider payload carries secret material that must not persist."""

    if depth > MAX_USAGE_SCAN_DEPTH:
        return False
    if isinstance(payload, Mapping):
        for key, child in payload.items():
            if not isinstance(key, str):
                continue
            if _key_looks_like_credential(key) and child not in (None, "", []):
                return True
            normalized = _normalize_key(key)
            if (
                normalized in {_normalize_key(item) for item in _REQUEST_ID_KEYS}
                or normalized.endswith("request_id")
                or normalized.endswith("requestid")
            ) and _value_looks_like_credential(child):
                return True
            if provider_response_contains_credentials(child, depth=depth + 1):
                return True
        return False
    if isinstance(payload, str):
        return bool(
            _UNSAFE_REQUEST_ID_RE.search(payload) and "BEGIN" in payload.upper()
        )
    if isinstance(payload, Sequence) and not isinstance(
        payload, (str, bytes, bytearray)
    ):
        return any(
            provider_response_contains_credentials(item, depth=depth + 1)
            for item in payload
        )
    return False


def is_safe_provider_request_id(value: Any) -> bool:
    """Opaque provider request IDs only; credentials and JWTs are rejected."""

    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if not candidate or "\x00" in candidate:
        return False
    if len(candidate.encode("utf-8")) > MAX_TEXT_BYTES:
        return False
    if _UNSAFE_REQUEST_ID_RE.search(candidate) or _JWTISH_RE.match(candidate):
        return False
    return _SAFE_REQUEST_ID_RE.fullmatch(candidate) is not None


def extract_safe_provider_request_ids(
    payload: Mapping[str, Any],
) -> tuple[str, ...]:
    """Collect unique request IDs that are safe to persist."""

    found: list[str] = []
    seen: set[str] = set()

    def _consider(raw: Any) -> None:
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            for item in raw:
                _consider(item)
            return
        if not is_safe_provider_request_id(raw):
            return
        candidate = str(raw).strip()
        if candidate in seen:
            return
        seen.add(candidate)
        found.append(candidate)

    for mapping in _walk_mappings(payload):
        for key, child in mapping.items():
            if not isinstance(key, str):
                continue
            normalized = _normalize_key(key)
            if normalized in {_normalize_key(item) for item in _REQUEST_ID_KEYS} or (
                normalized.endswith("request_id") or normalized.endswith("requestid")
            ):
                _consider(child)
        headers = mapping.get("headers")
        if isinstance(headers, Mapping):
            for key, child in headers.items():
                if isinstance(key, str) and _normalize_key(key) in {
                    "x_request_id",
                    "request_id",
                    "x_openai_request_id",
                }:
                    _consider(child)
    if len(found) > MAX_REQUEST_IDS:
        raise BenchmarkTelemetryError("safe_provider_request_ids exceeds bound")
    return tuple(found)


def _first_present_integer(
    mappings: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> int | None:
    key_set = {_normalize_key(key) for key in keys}
    for mapping in mappings:
        for key, child in mapping.items():
            if not isinstance(key, str) or _normalize_key(key) not in key_set:
                continue
            if child is None:
                continue
            if isinstance(child, bool) or not isinstance(child, int):
                raise BenchmarkTelemetryError(
                    f"{key} must be an integer token or charge count"
                )
            if child < 0:
                raise BenchmarkTelemetryError(f"{key} must be non-negative")
            return _integer(child, key)
    return None


def _nested_integer(
    mappings: Sequence[Mapping[str, Any]],
    parent_keys: Sequence[str],
    child_keys: Sequence[str],
) -> int | None:
    parents = {_normalize_key(key) for key in parent_keys}
    for mapping in mappings:
        for key, child in mapping.items():
            if not isinstance(key, str) or _normalize_key(key) not in parents:
                continue
            if isinstance(child, Mapping):
                found = _first_present_integer((child,), child_keys)
                if found is not None:
                    return found
    return None


def _optional_text(
    mappings: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> str:
    key_set = {_normalize_key(key) for key in keys}
    for mapping in mappings:
        for key, child in mapping.items():
            if not isinstance(key, str) or _normalize_key(key) not in key_set:
                continue
            if isinstance(child, str) and child.strip():
                return _text(child, key)
    return ""


def _usage_sensor(span_id: str, field: str) -> str:
    return _sensor_id("provider-usage", span_id, field)


def _quantity_sample(
    metric_name: str,
    value: int | None,
    *,
    unit: str,
    span_id: str,
    absent: UnavailableReason = UnavailableReason.NOT_REPORTED,
    quarantined: bool = False,
) -> TelemetrySample:
    sensor = _usage_sensor(span_id or "unbound", metric_name)
    if quarantined:
        return TelemetrySample.unavailable(
            metric_name,
            UnavailableReason.NOT_ADMITTED,
            sensor_id=sensor,
        )
    if value is None:
        return TelemetrySample.unavailable(metric_name, absent, sensor_id=sensor)
    return TelemetrySample.measured(
        metric_name, value, unit=unit, sensor_id=sensor
    )


def _labeled_estimate_sample(
    metric_name: str,
    estimate: Mapping[str, Any] | None,
    *,
    span_id: str,
    quarantined: bool = False,
) -> TelemetrySample:
    sensor = _usage_sensor(span_id or "unbound", metric_name)
    if quarantined:
        return TelemetrySample.unavailable(
            metric_name,
            UnavailableReason.NOT_ADMITTED,
            sensor_id=sensor,
        )
    if estimate is None:
        return TelemetrySample.unavailable(
            metric_name,
            UnavailableReason.NOT_REPORTED,
            sensor_id=sensor,
        )
    if not isinstance(estimate, Mapping):
        raise BenchmarkTelemetryError("labeled_estimate must be an object")
    claimed_state = str(estimate.get("truth_state") or estimate.get("status") or "")
    if claimed_state == SampleStatus.MEASURED.value:
        raise BenchmarkTelemetryError(
            "estimated charge cannot be labeled measured"
        )
    if "sensor_id" in estimate and claimed_state != SampleStatus.ESTIMATED.value:
        raise BenchmarkTelemetryError(
            "estimated charge cannot carry a measured sensor_id"
        )
    estimator_id = estimate.get("estimator_id")
    method = estimate.get("method")
    if not isinstance(estimator_id, str) or not estimator_id.strip():
        raise BenchmarkTelemetryError("labeled estimate requires estimator_id")
    if not isinstance(method, str) or not method.strip():
        raise BenchmarkTelemetryError("labeled estimate requires method")
    value = estimate.get("value")
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkTelemetryError("labeled estimate value must be an integer")
    unit = str(estimate.get("unit") or UNIT_MICROUSD)
    identity = estimate.get("price_snapshot_identity", "unavailable")
    if not isinstance(identity, str) or not identity.strip():
        identity = "unavailable"
    return TelemetrySample.estimated(
        metric_name,
        value,
        unit=unit,
        estimator_id=estimator_id,
        method=method,
        price_snapshot_identity=identity,
        sensor_id=sensor,
    )


def _calls_by_model_class_samples(
    *,
    model_class: str,
    call_count: int | None,
    span_id: str,
    quarantined: bool = False,
) -> dict[str, TelemetrySample]:
    samples: dict[str, TelemetrySample] = {}
    known = model_class in MODEL_CALL_CLASSES
    for name in MODEL_CALL_CLASSES:
        metric = _CALL_CLASS_METRIC[name]
        if quarantined or call_count is None:
            samples[metric] = _quantity_sample(
                metric,
                None,
                unit=UNIT_COUNT,
                span_id=span_id,
                quarantined=quarantined,
                absent=(
                    UnavailableReason.NOT_ADMITTED
                    if quarantined
                    else UnavailableReason.NOT_REPORTED
                ),
            )
            continue
        if not known:
            samples[metric] = TelemetrySample.unavailable(
                metric,
                UnavailableReason.NOT_REPORTED,
                sensor_id=_usage_sensor(span_id, metric),
            )
            continue
        samples[metric] = TelemetrySample.measured(
            metric,
            call_count if name == model_class else 0,
            unit=UNIT_COUNT,
            sensor_id=_usage_sensor(span_id, metric),
        )
    return samples


def _span_has_causal_task_identity(span: BenchmarkCausalSpan | None) -> bool:
    return span is not None and bool(span.task_id)


def _span_is_admitted_for_usage(
    span: BenchmarkCausalSpan | None,
    *,
    session: BenchmarkTelemetrySession | None,
    admitted_span_ids: Container[str] | None,
) -> bool:
    if span is None:
        return False
    admitted: set[str] = set()
    if admitted_span_ids is not None:
        admitted.update(str(item) for item in admitted_span_ids)
    if session is not None:
        admitted.update(session.admitted_usage_span_ids)
        if span.span_id not in {item.span_id for item in session.spans}:
            return False
    if not admitted:
        return False
    if span.span_id in admitted:
        return True
    return any(ancestor in admitted for ancestor in span.ancestry)


@dataclass(frozen=True)
class ProviderUsageRecord(_TelemetryContract):
    """One provider response bound (or quarantined) against a causal task span.

    Interface: BenchmarkProviderUsageRecord@1

    Admitted records map provider-reported usage onto an admitted task span.
    Absent token/cost fields remain ``unavailable`` and are never numeric zero.
    Unbound usage, credential leakage, estimated-as-measured data, and missing
    causal identity quarantine the record instead of admitting it.
    """

    SCHEMA: ClassVar[str] = PROVIDER_USAGE_RECORD_SCHEMA
    INTERFACE: ClassVar[str] = PROVIDER_USAGE_RECORD_INTERFACE

    record_id: str
    disposition: ProviderUsageDisposition
    span: BenchmarkCausalSpan | None
    provider_id: str
    model_id: str
    model_revision: str
    model_class: str
    safe_request_ids: tuple[str, ...]
    input_tokens: TelemetrySample
    output_tokens: TelemetrySample
    cached_input_tokens: TelemetrySample
    reasoning_tokens: TelemetrySample
    number_of_calls: TelemetrySample
    calls_by_model_class: tuple[TelemetrySample, ...]
    provider_reported_charge: TelemetrySample
    token_based_estimated_charge: TelemetrySample
    model_revision_sample: TelemetrySample
    quarantine_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _text(self.record_id, "record_id"))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ProviderUsageDisposition, "disposition"),
        )
        span = self.span
        if isinstance(span, Mapping):
            span = BenchmarkCausalSpan.from_dict(span)
        if span is not None and not isinstance(span, BenchmarkCausalSpan):
            raise BenchmarkTelemetryError("span must be BenchmarkCausalSpan")
        object.__setattr__(self, "span", span)
        for name in ("provider_id", "model_id", "model_revision", "model_class"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        if self.model_class and self.model_class not in MODEL_CALL_CLASSES:
            raise BenchmarkTelemetryError("model_class is not a supported call class")
        request_ids = tuple(
            _text(item, "request_id") for item in (self.safe_request_ids or ())
        )
        if len(request_ids) != len(set(request_ids)):
            raise BenchmarkTelemetryError("safe_request_ids contains duplicates")
        if len(request_ids) > MAX_REQUEST_IDS:
            raise BenchmarkTelemetryError("safe_request_ids exceeds bound")
        for item in request_ids:
            if not is_safe_provider_request_id(item):
                raise BenchmarkTelemetryError(
                    "safe_request_ids contains an unsafe request id"
                )
        object.__setattr__(self, "safe_request_ids", request_ids)

        def _sample(value: Any, name: str) -> TelemetrySample:
            if isinstance(value, Mapping):
                return TelemetrySample.from_dict(value)
            if isinstance(value, TelemetrySample):
                return value
            raise BenchmarkTelemetryError(f"{name} must be a TelemetrySample")

        for name in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
            "number_of_calls",
            "provider_reported_charge",
            "token_based_estimated_charge",
            "model_revision_sample",
        ):
            object.__setattr__(self, name, _sample(getattr(self, name), name))

        classes: list[TelemetrySample] = []
        for item in self.calls_by_model_class or ():
            classes.append(_sample(item, "calls_by_model_class"))
        names = [item.metric_name for item in classes]
        if len(names) != len(set(names)):
            raise BenchmarkTelemetryError(
                "calls_by_model_class contains duplicate metrics"
            )
        object.__setattr__(self, "calls_by_model_class", tuple(classes))
        object.__setattr__(
            self,
            "quarantine_reason",
            _text(self.quarantine_reason, "quarantine_reason", required=False),
        )

        if self.disposition is ProviderUsageDisposition.ADMITTED:
            if span is None or not span.task_id:
                raise BenchmarkTelemetryError(
                    "admitted usage requires a causal task span"
                )
            if self.quarantine_reason:
                raise BenchmarkTelemetryError(
                    "admitted usage cannot carry a quarantine reason"
                )
            if self.token_based_estimated_charge.status is SampleStatus.MEASURED:
                raise BenchmarkTelemetryError(
                    "estimated charge cannot be admitted as measured"
                )
            if self.provider_reported_charge.status is SampleStatus.ESTIMATED:
                raise BenchmarkTelemetryError(
                    "provider-reported charge cannot be labeled estimated"
                )
        else:
            if not self.quarantine_reason:
                raise BenchmarkTelemetryError(
                    "quarantined usage requires a quarantine_reason"
                )
            try:
                ProviderUsageQuarantineReason(self.quarantine_reason)
            except ValueError as exc:
                raise BenchmarkTelemetryError(
                    "quarantine_reason is not a supported quarantine reason"
                ) from exc

        for sample in (
            self.input_tokens,
            self.output_tokens,
            self.cached_input_tokens,
            self.reasoning_tokens,
            self.provider_reported_charge,
        ):
            if (
                sample.status is SampleStatus.UNAVAILABLE
                and (sample.value != 0 or sample.unit)
            ):
                raise BenchmarkTelemetryError(
                    "unavailable usage/cost fields must not encode numeric zero"
                )

    @property
    def span_id(self) -> str:
        return "" if self.span is None else self.span.span_id

    @property
    def task_id(self) -> str:
        return "" if self.span is None else self.span.task_id

    @property
    def admitted(self) -> bool:
        return self.disposition is ProviderUsageDisposition.ADMITTED

    def sample_map(self) -> dict[str, TelemetrySample]:
        samples = {
            self.input_tokens.metric_name: self.input_tokens,
            self.output_tokens.metric_name: self.output_tokens,
            self.cached_input_tokens.metric_name: self.cached_input_tokens,
            self.reasoning_tokens.metric_name: self.reasoning_tokens,
            self.number_of_calls.metric_name: self.number_of_calls,
            self.provider_reported_charge.metric_name: self.provider_reported_charge,
            self.token_based_estimated_charge.metric_name: (
                self.token_based_estimated_charge
            ),
            self.model_revision_sample.metric_name: self.model_revision_sample,
        }
        for item in self.calls_by_model_class:
            samples[item.metric_name] = item
        request_metric = "safe_provider_request_id_count"
        if self.disposition is ProviderUsageDisposition.ADMITTED:
            samples[request_metric] = TelemetrySample.measured(
                request_metric,
                len(self.safe_request_ids),
                unit=UNIT_COUNT,
                sensor_id=_usage_sensor(self.span_id, request_metric),
            )
        else:
            samples[request_metric] = TelemetrySample.unavailable(
                request_metric,
                UnavailableReason.NOT_ADMITTED,
                sensor_id=_usage_sensor(self.span_id or "unbound", request_metric),
            )
        return samples

    def unavailable_fields(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                name
                for name, sample in self.sample_map().items()
                if sample.status is SampleStatus.UNAVAILABLE
            )
        )

    def to_resource_measurement(self) -> BenchmarkResourceMeasurement:
        if (
            self.disposition is not ProviderUsageDisposition.ADMITTED
            or self.span is None
        ):
            raise BenchmarkTelemetryError(
                "quarantined unbound usage cannot seal a resource measurement"
            )
        return build_resource_measurement(
            measurement_id=f"meas:provider-usage:{self.record_id}",
            span=self.span,
            samples=self.sample_map(),
        )

    def to_model_use_fields(self) -> dict[str, Any]:
        """Closed model-use projection used by task efficiency receipts."""

        calls = {
            item.metric_name.removeprefix("calls_"): item.to_quantity_envelope()
            for item in self.calls_by_model_class
        }
        request_ids: dict[str, Any]
        if self.disposition is ProviderUsageDisposition.ADMITTED:
            request_ids = {
                "truth_state": "observed",
                "values": list(self.safe_request_ids),
            }
        else:
            request_ids = {
                "truth_state": "unavailable",
                "reason_code": "not_admitted",
            }
        reported = {
            "truth_state": "observed",
            "input_tokens": self.input_tokens.to_quantity_envelope(),
            "output_tokens": self.output_tokens.to_quantity_envelope(),
            "cached_input_tokens": self.cached_input_tokens.to_quantity_envelope(),
            "reasoning_tokens": self.reasoning_tokens.to_quantity_envelope(),
        }
        if self.disposition is not ProviderUsageDisposition.ADMITTED:
            reported = {
                "truth_state": "unavailable",
                "reason_code": "not_admitted",
            }
        return {
            "input_tokens": self.input_tokens.to_quantity_envelope(),
            "output_tokens": self.output_tokens.to_quantity_envelope(),
            "cached_input_tokens": self.cached_input_tokens.to_quantity_envelope(),
            "reasoning_tokens": self.reasoning_tokens.to_quantity_envelope(),
            "number_of_calls": self.number_of_calls.to_quantity_envelope(),
            "calls_by_model_class": calls,
            "safe_provider_request_ids": request_ids,
            "provider_reported_usage": reported,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": BENCHMARK_TELEMETRY_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "record_id": self.record_id,
            "disposition": self.disposition.value,
            "span": None if self.span is None else self.span.to_record(),
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_class": self.model_class,
            "safe_request_ids": list(self.safe_request_ids),
            "input_tokens": self.input_tokens.to_record(),
            "output_tokens": self.output_tokens.to_record(),
            "cached_input_tokens": self.cached_input_tokens.to_record(),
            "reasoning_tokens": self.reasoning_tokens.to_record(),
            "number_of_calls": self.number_of_calls.to_record(),
            "calls_by_model_class": [
                item.to_record() for item in self.calls_by_model_class
            ],
            "provider_reported_charge": self.provider_reported_charge.to_record(),
            "token_based_estimated_charge": (
                self.token_based_estimated_charge.to_record()
            ),
            "model_revision_sample": self.model_revision_sample.to_record(),
            "quarantine_reason": self.quarantine_reason,
            "unavailable_fields": list(self.unavailable_fields()),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderUsageRecord":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "interface",
            "record_id",
            "disposition",
            "span",
            "provider_id",
            "model_id",
            "model_revision",
            "model_class",
            "safe_request_ids",
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
            "number_of_calls",
            "calls_by_model_class",
            "provider_reported_charge",
            "token_based_estimated_charge",
            "model_revision_sample",
            "quarantine_reason",
            "unavailable_fields",
            "content_id",
        }
        _closed(
            payload, schema=cls.SCHEMA, allowed=allowed, name="provider usage record"
        )
        result = cls(
            record_id=payload.get("record_id", ""),
            disposition=payload.get("disposition", ""),
            span=payload.get("span"),
            provider_id=payload.get("provider_id", ""),
            model_id=payload.get("model_id", ""),
            model_revision=payload.get("model_revision", ""),
            model_class=payload.get("model_class", ""),
            safe_request_ids=tuple(payload.get("safe_request_ids") or ()),
            input_tokens=payload.get("input_tokens", {}),
            output_tokens=payload.get("output_tokens", {}),
            cached_input_tokens=payload.get("cached_input_tokens", {}),
            reasoning_tokens=payload.get("reasoning_tokens", {}),
            number_of_calls=payload.get("number_of_calls", {}),
            calls_by_model_class=tuple(payload.get("calls_by_model_class") or ()),
            provider_reported_charge=payload.get("provider_reported_charge", {}),
            token_based_estimated_charge=payload.get(
                "token_based_estimated_charge", {}
            ),
            model_revision_sample=payload.get("model_revision_sample", {}),
            quarantine_reason=payload.get("quarantine_reason", ""),
        )
        if payload.get("interface", result.INTERFACE) != result.INTERFACE:
            raise BenchmarkTelemetryError("provider usage interface mismatch")
        claimed = payload.get("unavailable_fields")
        if claimed is not None and tuple(claimed) != result.unavailable_fields():
            raise BenchmarkTelemetryError(
                "unavailable_fields does not match usage samples"
            )
        _claim(payload, result.content_id, "content_id")
        return result


def _quarantine_usage_record(
    *,
    record_id: str,
    span: BenchmarkCausalSpan | None,
    reason: ProviderUsageQuarantineReason,
    provider_id: str = "",
    model_id: str = "",
    model_revision: str = "",
    model_class: str = "",
) -> ProviderUsageRecord:
    span_id = "" if span is None else span.span_id

    def _blank(metric: str, unit: str = UNIT_TOKENS) -> TelemetrySample:
        return _quantity_sample(
            metric,
            None,
            unit=unit,
            span_id=span_id,
            quarantined=True,
            absent=UnavailableReason.NOT_ADMITTED,
        )

    class_samples = _calls_by_model_class_samples(
        model_class=model_class,
        call_count=None,
        span_id=span_id,
        quarantined=True,
    )
    return ProviderUsageRecord(
        record_id=record_id,
        disposition=ProviderUsageDisposition.QUARANTINED,
        span=span,
        provider_id=provider_id,
        model_id=model_id,
        model_revision=model_revision,
        model_class=model_class if model_class in MODEL_CALL_CLASSES else "",
        safe_request_ids=(),
        input_tokens=_blank("provider_native_input_tokens"),
        output_tokens=_blank("provider_native_output_tokens"),
        cached_input_tokens=_blank("provider_native_cached_input_tokens"),
        reasoning_tokens=_blank("provider_native_reasoning_tokens"),
        number_of_calls=_blank("model_call_count", UNIT_COUNT),
        calls_by_model_class=tuple(
            class_samples[name] for name in PROVIDER_USAGE_METRIC_NAMES
            if name.startswith("calls_")
        ),
        provider_reported_charge=_blank(
            "provider_reported_charge_microusd", UNIT_MICROUSD
        ),
        token_based_estimated_charge=_blank(
            "token_based_estimated_charge_microusd", UNIT_MICROUSD
        ),
        model_revision_sample=_blank(
            "provider_model_revision_identity", UNIT_IDENTITY
        ),
        quarantine_reason=reason.value,
    )


def map_provider_response_to_admitted_span(
    response: Mapping[str, Any],
    span: BenchmarkCausalSpan | None,
    *,
    session: BenchmarkTelemetrySession | None = None,
    admitted_span_ids: Container[str] | None = None,
    record_id: str | None = None,
    labeled_estimate: Mapping[str, Any] | None = None,
    model_class: ModelCallClass | str | None = None,
) -> ProviderUsageRecord:
    """Bind one provider response to an admitted task span.

    Provider-reported usage is measured only when the integer field is present.
    Absent usage and cost fields stay ``unavailable`` rather than numeric zero.
    Estimates require an explicit estimator identity and never overwrite
    reported measurements. Fail-closed quarantine covers unbound usage,
    credential leakage, estimated-as-measured data, and missing causal identity.
    """

    if not isinstance(response, Mapping):
        raise BenchmarkTelemetryError("provider response must be an object")
    identity = record_id or _sensor_id(
        "provider-usage-record",
        "" if span is None else span.span_id,
        str(response.get("id") or response.get("request_id") or "anonymous"),
    )

    provider_id = ""
    model_id = ""
    model_revision = ""
    if span is not None and span.provider is not None:
        provider_id = span.provider.provider_id
        model_id = span.provider.model_id
        model_revision = span.provider.model_revision

    resolved_class = ""
    if model_class is not None:
        resolved_class = (
            model_class.value
            if isinstance(model_class, ModelCallClass)
            else _text(model_class, "model_class", required=False)
        )

    if provider_response_contains_credentials(response):
        return _quarantine_usage_record(
            record_id=identity,
            span=span,
            reason=ProviderUsageQuarantineReason.CREDENTIAL_LEAKAGE,
            provider_id=provider_id,
            model_id=model_id,
            model_revision=model_revision,
            model_class=resolved_class,
        )

    claimed_measured_estimate = False
    usage_nodes = list(_walk_mappings(response))
    for node in usage_nodes:
        for key, child in node.items():
            if not isinstance(key, str) or not isinstance(child, Mapping):
                continue
            normalized = _normalize_key(key)
            if "estimat" not in normalized:
                continue
            state = str(child.get("truth_state") or child.get("status") or "")
            if state == SampleStatus.MEASURED.value or "sensor_id" in child:
                claimed_measured_estimate = True
    if labeled_estimate is not None:
        state = str(
            labeled_estimate.get("truth_state")
            or labeled_estimate.get("status")
            or ""
        )
        if state == SampleStatus.MEASURED.value or (
            "sensor_id" in labeled_estimate and state != SampleStatus.ESTIMATED.value
        ):
            claimed_measured_estimate = True
    if claimed_measured_estimate:
        return _quarantine_usage_record(
            record_id=identity,
            span=span,
            reason=ProviderUsageQuarantineReason.ESTIMATED_AS_MEASURED,
            provider_id=provider_id,
            model_id=model_id,
            model_revision=model_revision,
            model_class=resolved_class,
        )

    if not _span_has_causal_task_identity(span):
        return _quarantine_usage_record(
            record_id=identity,
            span=span,
            reason=ProviderUsageQuarantineReason.MISSING_CAUSAL_IDENTITY,
            provider_id=provider_id,
            model_id=model_id,
            model_revision=model_revision,
            model_class=resolved_class,
        )

    if not _span_is_admitted_for_usage(
        span, session=session, admitted_span_ids=admitted_span_ids
    ):
        return _quarantine_usage_record(
            record_id=identity,
            span=span,
            reason=ProviderUsageQuarantineReason.UNBOUND_USAGE,
            provider_id=provider_id,
            model_id=model_id,
            model_revision=model_revision,
            model_class=resolved_class,
        )

    assert span is not None
    span_id = span.span_id
    provider_id = _optional_text(
        usage_nodes, ("provider_id", "provider", "provider_name")
    ) or provider_id
    model_id = _optional_text(
        usage_nodes, ("model_id", "model", "model_name")
    ) or model_id
    model_revision = _optional_text(
        usage_nodes,
        (
            "model_revision",
            "system_fingerprint",
            "model_version",
            "revision",
        ),
    ) or model_revision
    if not resolved_class:
        resolved_class = _optional_text(usage_nodes, ("model_class", "call_class"))

    input_tokens = _first_present_integer(usage_nodes, _INPUT_TOKEN_KEYS)
    output_tokens = _first_present_integer(usage_nodes, _OUTPUT_TOKEN_KEYS)
    cached_tokens = _first_present_integer(usage_nodes, _CACHED_TOKEN_KEYS)
    if cached_tokens is None:
        cached_tokens = _nested_integer(
            usage_nodes,
            ("prompt_tokens_details", "input_tokens_details", "cache_details"),
            _CACHED_TOKEN_KEYS + ("cached_tokens",),
        )
    reasoning_tokens = _first_present_integer(usage_nodes, _REASONING_TOKEN_KEYS)
    if reasoning_tokens is None:
        reasoning_tokens = _nested_integer(
            usage_nodes,
            (
                "completion_tokens_details",
                "output_tokens_details",
                "reasoning_details",
            ),
            _REASONING_TOKEN_KEYS,
        )
    reported_charge = _first_present_integer(usage_nodes, _CHARGE_MICROUSD_KEYS)
    call_count = _first_present_integer(
        usage_nodes, ("number_of_calls", "call_count", "n")
    )
    if call_count is None:
        call_count = 1

    try:
        estimate_sample = _labeled_estimate_sample(
            "token_based_estimated_charge_microusd",
            labeled_estimate,
            span_id=span_id,
        )
    except BenchmarkTelemetryError as exc:
        message = str(exc)
        if "labeled measured" in message or "measured sensor_id" in message:
            return _quarantine_usage_record(
                record_id=identity,
                span=span,
                reason=ProviderUsageQuarantineReason.ESTIMATED_AS_MEASURED,
                provider_id=provider_id,
                model_id=model_id,
                model_revision=model_revision,
                model_class=resolved_class,
            )
        raise

    if model_revision:
        revision_sample = TelemetrySample.measured(
            "provider_model_revision_identity",
            _identity_digest(model_revision),
            unit=UNIT_IDENTITY,
            sensor_id=_usage_sensor(span_id, "provider_model_revision_identity"),
        )
    else:
        revision_sample = TelemetrySample.unavailable(
            "provider_model_revision_identity",
            UnavailableReason.NOT_REPORTED,
            sensor_id=_usage_sensor(span_id, "provider_model_revision_identity"),
        )

    class_samples = _calls_by_model_class_samples(
        model_class=resolved_class,
        call_count=call_count,
        span_id=span_id,
    )
    return ProviderUsageRecord(
        record_id=identity,
        disposition=ProviderUsageDisposition.ADMITTED,
        span=span,
        provider_id=provider_id,
        model_id=model_id,
        model_revision=model_revision,
        model_class=resolved_class if resolved_class in MODEL_CALL_CLASSES else "",
        safe_request_ids=extract_safe_provider_request_ids(response),
        input_tokens=_quantity_sample(
            "provider_native_input_tokens",
            input_tokens,
            unit=UNIT_TOKENS,
            span_id=span_id,
        ),
        output_tokens=_quantity_sample(
            "provider_native_output_tokens",
            output_tokens,
            unit=UNIT_TOKENS,
            span_id=span_id,
        ),
        cached_input_tokens=_quantity_sample(
            "provider_native_cached_input_tokens",
            cached_tokens,
            unit=UNIT_TOKENS,
            span_id=span_id,
        ),
        reasoning_tokens=_quantity_sample(
            "provider_native_reasoning_tokens",
            reasoning_tokens,
            unit=UNIT_TOKENS,
            span_id=span_id,
        ),
        number_of_calls=_quantity_sample(
            "model_call_count",
            call_count,
            unit=UNIT_COUNT,
            span_id=span_id,
        ),
        calls_by_model_class=tuple(
            class_samples[_CALL_CLASS_METRIC[name]] for name in MODEL_CALL_CLASSES
        ),
        provider_reported_charge=_quantity_sample(
            "provider_reported_charge_microusd",
            reported_charge,
            unit=UNIT_MICROUSD,
            span_id=span_id,
        ),
        token_based_estimated_charge=estimate_sample,
        model_revision_sample=revision_sample,
    )


def project_provider_usage_samples(
    record: ProviderUsageRecord,
) -> dict[str, TelemetrySample]:
    """Project a provider-usage record onto named telemetry samples."""

    if not isinstance(record, ProviderUsageRecord):
        raise BenchmarkTelemetryError("record must be ProviderUsageRecord")
    return record.sample_map()


# ---------------------------------------------------------------------------
# Work / compute telemetry (ASEH-012)
# ---------------------------------------------------------------------------


def _require_cid(value: Any, name: str) -> str:
    cid = _text(value, name)
    if len(cid.encode("utf-8")) > MAX_CID_BYTES or CID_RE.fullmatch(cid) is None:
        raise BenchmarkTelemetryError(f"{name} must be a CIDv1")
    return cid


def _operation_reason(value: Any) -> str:
    code = _text(value, "reason_code").replace("-", "_")
    if code not in OPERATION_REASON_CODES:
        raise BenchmarkTelemetryError("reason_code is not a supported unavailable reason")
    return code


def _unavailable_reason_from_payload(value: Any) -> UnavailableReason:
    code = _text(value, "reason_code").replace("_", "-")
    return _enum(code, UnavailableReason, "reason_code")


def _work_sensor(span_id: str, field: str) -> str:
    return _sensor_id("work-telemetry", span_id or "unbound", field)


def _rename_sample(sample: TelemetrySample, metric_name: str) -> TelemetrySample:
    if sample.status is SampleStatus.MEASURED:
        return TelemetrySample.measured(
            metric_name,
            sample.value,
            unit=sample.unit,
            sensor_id=sample.sensor_id,
        )
    if sample.status is SampleStatus.ESTIMATED:
        return TelemetrySample.estimated(
            metric_name,
            sample.value,
            unit=sample.unit,
            estimator_id=sample.estimator_id,
            method=sample.method,
            price_snapshot_identity=sample.price_snapshot_identity,
            sensor_id=sample.sensor_id,
        )
    return TelemetrySample.unavailable(
        metric_name,
        sample.reason_code,
        sensor_id=sample.sensor_id,
    )


def _blank_compute_sample(
    field_name: str,
    *,
    span_id: str,
    quarantined: bool = False,
    reason: UnavailableReason = UnavailableReason.NOT_REPORTED,
) -> TelemetrySample:
    return TelemetrySample.unavailable(
        field_name,
        UnavailableReason.NOT_ADMITTED if quarantined else reason,
        sensor_id=_work_sensor(span_id, field_name),
    )


@dataclass(frozen=True)
class OperationObservation(_TelemetryContract):
    """Attempted, observed, verified, unavailable, or simulated work operation.

    Attempted cannot carry observer or verifier fields. Observed cannot carry
    admitted verifier linkage. Verified requires an admitted verifier identity
    and receipt CID. Unavailable never encodes a numeric count or zero.
    """

    SCHEMA: ClassVar[str] = OPERATION_OBSERVATION_SCHEMA

    field_name: str
    truth_state: OperationTruthState
    operation_id: str = ""
    attempt_count: int = 0
    observer_id: str = ""
    count: int = 0
    verifier_id: str = ""
    verifier_receipt_cid: str = ""
    reason_code: str = ""
    fixture_id: str = ""
    covering_span_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "field_name", _text(self.field_name, "field_name")
        )
        if self.field_name not in WORK_OPERATION_FIELD_NAMES:
            raise BenchmarkTelemetryError(
                f"{self.field_name!r} is not a work operation field"
            )
        object.__setattr__(
            self,
            "truth_state",
            _enum(self.truth_state, OperationTruthState, "truth_state"),
        )
        object.__setattr__(
            self,
            "operation_id",
            _text(self.operation_id, "operation_id", required=False),
        )
        object.__setattr__(
            self,
            "attempt_count",
            _integer(self.attempt_count, "attempt_count"),
        )
        object.__setattr__(
            self,
            "observer_id",
            _text(self.observer_id, "observer_id", required=False),
        )
        object.__setattr__(self, "count", _integer(self.count, "count"))
        object.__setattr__(
            self,
            "verifier_id",
            _text(self.verifier_id, "verifier_id", required=False),
        )
        cid = self.verifier_receipt_cid
        if cid:
            cid = _require_cid(cid, "verifier_receipt_cid")
        else:
            cid = _text(cid, "verifier_receipt_cid", required=False)
        object.__setattr__(self, "verifier_receipt_cid", cid)
        object.__setattr__(
            self,
            "reason_code",
            _text(self.reason_code, "reason_code", required=False),
        )
        if self.reason_code:
            object.__setattr__(self, "reason_code", _operation_reason(self.reason_code))
        object.__setattr__(
            self,
            "fixture_id",
            _text(self.fixture_id, "fixture_id", required=False),
        )
        object.__setattr__(
            self,
            "covering_span_id",
            _text(self.covering_span_id, "covering_span_id", required=False),
        )
        self._assert_truth_state()

    def _assert_truth_state(self) -> None:
        state = self.truth_state
        if state is OperationTruthState.ATTEMPTED:
            if not self.operation_id or self.attempt_count < 1:
                raise BenchmarkTelemetryError(
                    f"{self.field_name}: attempted operations require operation_id "
                    "and a positive attempt_count"
                )
            if self.observer_id or self.verifier_id or self.verifier_receipt_cid:
                raise BenchmarkTelemetryError(
                    f"{self.field_name}: attempted operations cannot be represented "
                    "as observed or verified"
                )
            if self.count:
                raise BenchmarkTelemetryError(
                    f"{self.field_name}: attempted operations cannot carry an "
                    "observed count"
                )
            if self.reason_code or self.fixture_id:
                raise BenchmarkTelemetryError(
                    f"{self.field_name}: attempted operations cannot carry "
                    "unavailable or simulated labels"
                )
            return
        if state is OperationTruthState.OBSERVED:
            if not self.operation_id or not self.observer_id:
                raise BenchmarkTelemetryError(
                    f"{self.field_name}: observed operations require operation_id "
                    "and observer_id"
                )
            if self.verifier_id or self.verifier_receipt_cid:
                raise BenchmarkTelemetryError(
                    f"{self.field_name}: observed operations cannot be represented "
                    "as verified"
                )
            if self.attempt_count:
                raise BenchmarkTelemetryError(
                    f"{self.field_name}: attempted operations cannot be represented "
                    "as observed"
                )
            if self.reason_code or self.fixture_id:
                raise BenchmarkTelemetryError(
                    f"{self.field_name}: observed operations cannot carry "
                    "unavailable or simulated labels"
                )
            return
        if state is OperationTruthState.VERIFIED:
            if (
                not self.operation_id
                or not self.observer_id
                or not self.verifier_id
                or not self.verifier_receipt_cid
            ):
                raise BenchmarkTelemetryError(
                    f"{self.field_name}: verified operations require observer and "
                    "admitted verifier linkage"
                )
            if self.attempt_count:
                raise BenchmarkTelemetryError(
                    f"{self.field_name}: attempted operations cannot be represented "
                    "as verified"
                )
            if self.reason_code or self.fixture_id:
                raise BenchmarkTelemetryError(
                    f"{self.field_name}: verified operations cannot carry "
                    "unavailable or simulated labels"
                )
            return
        if state is OperationTruthState.UNAVAILABLE:
            if not self.reason_code:
                raise BenchmarkTelemetryError(
                    f"{self.field_name}: unavailable operations require a reason_code"
                )
            if (
                self.operation_id
                or self.attempt_count
                or self.observer_id
                or self.count
                or self.verifier_id
                or self.verifier_receipt_cid
                or self.fixture_id
            ):
                raise BenchmarkTelemetryError(
                    f"{self.field_name}: unavailable evidence cannot encode a "
                    "numeric value or operation identity"
                )
            return
        if not self.fixture_id or not self.reason_code:
            raise BenchmarkTelemetryError(
                f"{self.field_name}: simulated operations require fixture_id "
                "and reason_code"
            )
        if (
            self.observer_id
            or self.verifier_id
            or self.verifier_receipt_cid
            or self.count
            or self.attempt_count
        ):
            raise BenchmarkTelemetryError(
                f"{self.field_name}: simulated evidence cannot claim live "
                "observation or verification"
            )

    @classmethod
    def attempted(
        cls,
        field_name: str,
        operation_id: str,
        *,
        attempt_count: int = 1,
        covering_span_id: str = "",
    ) -> "OperationObservation":
        return cls(
            field_name=field_name,
            truth_state=OperationTruthState.ATTEMPTED,
            operation_id=operation_id,
            attempt_count=attempt_count,
            covering_span_id=covering_span_id,
        )

    @classmethod
    def observed(
        cls,
        field_name: str,
        operation_id: str,
        *,
        observer_id: str,
        count: int,
        covering_span_id: str = "",
    ) -> "OperationObservation":
        return cls(
            field_name=field_name,
            truth_state=OperationTruthState.OBSERVED,
            operation_id=operation_id,
            observer_id=observer_id,
            count=count,
            covering_span_id=covering_span_id,
        )

    @classmethod
    def verified(
        cls,
        field_name: str,
        operation_id: str,
        *,
        observer_id: str,
        count: int,
        verifier_id: str,
        verifier_receipt_cid: str,
        covering_span_id: str = "",
    ) -> "OperationObservation":
        return cls(
            field_name=field_name,
            truth_state=OperationTruthState.VERIFIED,
            operation_id=operation_id,
            observer_id=observer_id,
            count=count,
            verifier_id=verifier_id,
            verifier_receipt_cid=verifier_receipt_cid,
            covering_span_id=covering_span_id,
        )

    @classmethod
    def unavailable(
        cls,
        field_name: str,
        reason: UnavailableReason | str,
        *,
        covering_span_id: str = "",
    ) -> "OperationObservation":
        if isinstance(reason, UnavailableReason):
            code = reason.value.replace("-", "_")
        else:
            code = str(reason)
        return cls(
            field_name=field_name,
            truth_state=OperationTruthState.UNAVAILABLE,
            reason_code=code,
            covering_span_id=covering_span_id,
        )

    @classmethod
    def simulated(
        cls,
        field_name: str,
        fixture_id: str,
        reason: UnavailableReason | str = UnavailableReason.FIXTURE_ONLY,
        *,
        covering_span_id: str = "",
    ) -> "OperationObservation":
        if isinstance(reason, UnavailableReason):
            code = reason.value.replace("-", "_")
        else:
            code = str(reason)
        return cls(
            field_name=field_name,
            truth_state=OperationTruthState.SIMULATED,
            fixture_id=fixture_id,
            reason_code=code,
            covering_span_id=covering_span_id,
        )

    def to_operation_envelope(self) -> dict[str, Any]:
        """Efficiency-receipt operationObservation envelope."""

        if self.truth_state is OperationTruthState.ATTEMPTED:
            return {
                "truth_state": OperationTruthState.ATTEMPTED.value,
                "operation_id": self.operation_id,
                "attempt_count": self.attempt_count,
            }
        if self.truth_state is OperationTruthState.OBSERVED:
            return {
                "truth_state": OperationTruthState.OBSERVED.value,
                "operation_id": self.operation_id,
                "observer_id": self.observer_id,
                "count": self.count,
            }
        if self.truth_state is OperationTruthState.VERIFIED:
            return {
                "truth_state": OperationTruthState.VERIFIED.value,
                "operation_id": self.operation_id,
                "observer_id": self.observer_id,
                "count": self.count,
                "verifier_id": self.verifier_id,
                "verifier_receipt_cid": self.verifier_receipt_cid,
            }
        if self.truth_state is OperationTruthState.SIMULATED:
            return {
                "truth_state": OperationTruthState.SIMULATED.value,
                "fixture_id": self.fixture_id,
                "reason_code": self.reason_code,
            }
        return {
            "truth_state": OperationTruthState.UNAVAILABLE.value,
            "reason_code": self.reason_code,
        }

    def counted_value(self) -> int | None:
        if self.truth_state is OperationTruthState.OBSERVED:
            return self.count
        if self.truth_state is OperationTruthState.VERIFIED:
            return self.count
        if self.truth_state is OperationTruthState.ATTEMPTED:
            return self.attempt_count
        return None

    def to_quantity_sample(self, *, span_id: str) -> TelemetrySample:
        """Project an operation onto a quantity sample without collapsing states.

        Observed and verified counts are measured. Attempted, unavailable, and
        simulated stay unavailable rather than inventing a measured zero.
        """

        sensor = _work_sensor(span_id or self.covering_span_id, self.field_name)
        if self.truth_state in {
            OperationTruthState.OBSERVED,
            OperationTruthState.VERIFIED,
        }:
            return TelemetrySample.measured(
                self.field_name,
                self.count,
                unit=UNIT_COUNT,
                sensor_id=sensor,
            )
        if self.truth_state is OperationTruthState.UNAVAILABLE:
            return TelemetrySample.unavailable(
                self.field_name,
                self.reason_code.replace("_", "-"),
                sensor_id=sensor,
            )
        if self.truth_state is OperationTruthState.SIMULATED:
            return TelemetrySample.unavailable(
                self.field_name,
                UnavailableReason.FIXTURE_ONLY,
                sensor_id=sensor,
            )
        return TelemetrySample.unavailable(
            self.field_name,
            UnavailableReason.NOT_YET_MEASURED,
            sensor_id=sensor,
        )

    def _payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract_version": BENCHMARK_TELEMETRY_CONTRACT_VERSION,
            "field_name": self.field_name,
            "truth_state": self.truth_state.value,
            "covering_span_id": self.covering_span_id,
        }
        payload.update(self.to_operation_envelope())
        payload["truth_state"] = self.truth_state.value
        payload["field_name"] = self.field_name
        payload["covering_span_id"] = self.covering_span_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperationObservation":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "field_name",
            "truth_state",
            "operation_id",
            "attempt_count",
            "observer_id",
            "count",
            "verifier_id",
            "verifier_receipt_cid",
            "reason_code",
            "fixture_id",
            "covering_span_id",
            "content_id",
        }
        _closed(
            payload,
            schema=cls.SCHEMA,
            allowed=allowed,
            name="operation observation",
        )
        _detect_operation_fail_closed(
            str(payload.get("field_name") or "operation"), payload
        )
        result = cls(
            field_name=payload.get("field_name", ""),
            truth_state=payload.get("truth_state", ""),
            operation_id=payload.get("operation_id", ""),
            attempt_count=payload.get("attempt_count", 0),
            observer_id=payload.get("observer_id", ""),
            count=payload.get("count", 0),
            verifier_id=payload.get("verifier_id", ""),
            verifier_receipt_cid=payload.get("verifier_receipt_cid", ""),
            reason_code=payload.get("reason_code", ""),
            fixture_id=payload.get("fixture_id", ""),
            covering_span_id=payload.get("covering_span_id", ""),
        )
        _claim(payload, result.content_id, "content_id")
        return result


def _detect_operation_fail_closed(
    field_name: str, payload: Mapping[str, Any]
) -> None:
    """Raise on attempted-as-observed or observed-as-verified mislabels."""

    claimed = str(payload.get("truth_state") or payload.get("status") or "")
    has_observer = bool(payload.get("observer_id"))
    has_count = "count" in payload
    has_attempt = bool(payload.get("attempt_count"))
    has_verifier = bool(
        payload.get("verifier_id") or payload.get("verifier_receipt_cid")
    )
    if claimed == OperationTruthState.OBSERVED.value:
        if has_attempt or not has_observer:
            raise BenchmarkTelemetryError(
                f"{field_name}: attempted operations cannot be represented as observed"
            )
        if has_verifier:
            raise BenchmarkTelemetryError(
                f"{field_name}: observed operations cannot be represented as verified"
            )
        return
    if claimed == OperationTruthState.VERIFIED.value:
        if has_attempt:
            raise BenchmarkTelemetryError(
                f"{field_name}: attempted operations cannot be represented as verified"
            )
        if not (
            payload.get("verifier_id") and payload.get("verifier_receipt_cid")
        ):
            raise BenchmarkTelemetryError(
                f"{field_name}: observed operations cannot be represented as "
                "verified without admitted verifier evidence"
            )
        return
    if claimed == OperationTruthState.ATTEMPTED.value and (
        has_observer or has_count or has_verifier
    ):
        raise BenchmarkTelemetryError(
            f"{field_name}: attempted operations cannot be represented as "
            "observed or verified"
        )
    if claimed == OperationTruthState.UNAVAILABLE.value:
        for forbidden in (
            "count",
            "attempt_count",
            "value",
            "unit",
            "observer_id",
            "operation_id",
            "verifier_id",
            "verifier_receipt_cid",
            "fixture_id",
        ):
            if forbidden in payload:
                raise BenchmarkTelemetryError(
                    f"{field_name}: unavailable evidence cannot encode a "
                    "numeric value or operation identity"
                )
    if claimed == OperationTruthState.SIMULATED.value and (
        has_observer or has_count or has_verifier or has_attempt
    ):
        raise BenchmarkTelemetryError(
            f"{field_name}: simulated evidence cannot claim live observation "
            "or verification"
        )


def _operation_from_payload(
    field_name: str,
    payload: Any,
    *,
    covering_span_id: str,
    quarantined: bool = False,
) -> OperationObservation:
    if quarantined:
        return OperationObservation.unavailable(
            field_name,
            UnavailableReason.NOT_ADMITTED,
            covering_span_id=covering_span_id,
        )
    if payload is None:
        return OperationObservation.unavailable(
            field_name,
            UnavailableReason.NOT_REPORTED,
            covering_span_id=covering_span_id,
        )
    if isinstance(payload, OperationObservation):
        if payload.field_name != field_name:
            raise BenchmarkTelemetryError(
                f"{field_name}: operation field_name does not match"
            )
        if covering_span_id and not payload.covering_span_id:
            return OperationObservation(
                field_name=payload.field_name,
                truth_state=payload.truth_state,
                operation_id=payload.operation_id,
                attempt_count=payload.attempt_count,
                observer_id=payload.observer_id,
                count=payload.count,
                verifier_id=payload.verifier_id,
                verifier_receipt_cid=payload.verifier_receipt_cid,
                reason_code=payload.reason_code,
                fixture_id=payload.fixture_id,
                covering_span_id=covering_span_id,
            )
        return payload
    if not isinstance(payload, Mapping):
        raise BenchmarkTelemetryError(f"{field_name} must be an operation object")
    _detect_operation_fail_closed(field_name, payload)
    body = dict(payload)
    body["field_name"] = field_name
    if covering_span_id and not body.get("covering_span_id"):
        body["covering_span_id"] = covering_span_id
    if "truth_state" not in body and "status" in body:
        body["truth_state"] = body["status"]
    try:
        return OperationObservation(
            field_name=field_name,
            truth_state=body.get("truth_state", ""),
            operation_id=body.get("operation_id", ""),
            attempt_count=body.get("attempt_count", 0),
            observer_id=body.get("observer_id", ""),
            count=body.get("count", 0),
            verifier_id=body.get("verifier_id", ""),
            verifier_receipt_cid=body.get("verifier_receipt_cid", ""),
            reason_code=body.get("reason_code", ""),
            fixture_id=body.get("fixture_id", ""),
            covering_span_id=body.get("covering_span_id", covering_span_id),
        )
    except BenchmarkTelemetryError:
        raise
    except Exception as exc:
        raise BenchmarkTelemetryError(
            f"{field_name}: invalid operation observation"
        ) from exc


def _compute_sample_from_payload(
    field_name: str,
    payload: Any,
    *,
    span_id: str,
    quarantined: bool = False,
    default_unit: str = UNIT_SECONDS_MILLIONTHS,
) -> TelemetrySample:
    if quarantined:
        return _blank_compute_sample(field_name, span_id=span_id, quarantined=True)
    if payload is None:
        return _blank_compute_sample(field_name, span_id=span_id)
    if isinstance(payload, TelemetrySample):
        if payload.metric_name != field_name:
            return _rename_sample(payload, field_name)
        return payload
    if not isinstance(payload, Mapping):
        raise BenchmarkTelemetryError(f"{field_name} must be a quantity object")
    claimed = str(payload.get("truth_state") or payload.get("status") or "")
    if claimed in {
        OperationTruthState.ATTEMPTED.value,
        OperationTruthState.OBSERVED.value,
        OperationTruthState.VERIFIED.value,
    }:
        raise BenchmarkTelemetryError(
            f"{field_name}: compute fields cannot carry operation truth states"
        )
    if claimed == SampleStatus.ESTIMATED.value or (
        "estimator_id" in payload and claimed != SampleStatus.MEASURED.value
    ):
        if claimed == SampleStatus.MEASURED.value or (
            "sensor_id" in payload and claimed != SampleStatus.ESTIMATED.value
        ):
            raise BenchmarkTelemetryError(
                f"{field_name}: estimated values cannot be labeled measured"
            )
        return TelemetrySample.estimated(
            field_name,
            payload.get("value", 0),
            unit=str(payload.get("unit") or default_unit),
            estimator_id=str(payload.get("estimator_id", "")),
            method=str(payload.get("method", "")),
            price_snapshot_identity=str(
                payload.get("price_snapshot_identity", "unavailable")
            ),
            sensor_id=str(payload.get("sensor_id") or _work_sensor(span_id, field_name)),
        )
    if claimed == SampleStatus.UNAVAILABLE.value or claimed == "":
        if claimed == "" and "value" in payload:
            claimed = SampleStatus.MEASURED.value
        elif claimed != SampleStatus.MEASURED.value:
            reason = payload.get("reason_code") or UnavailableReason.NOT_REPORTED.value
            return TelemetrySample.unavailable(
                field_name,
                _unavailable_reason_from_payload(reason),
                sensor_id=str(
                    payload.get("sensor_id") or _work_sensor(span_id, field_name)
                ),
            )
    if claimed == SampleStatus.MEASURED.value or "value" in payload:
        if "estimator_id" in payload or "method" in payload:
            raise BenchmarkTelemetryError(
                f"{field_name}: measured values cannot carry estimator labels"
            )
        sensor = payload.get("sensor_id")
        if not isinstance(sensor, str) or not sensor.strip():
            raise BenchmarkTelemetryError(
                f"{field_name}: measured sample requires a sensor_id"
            )
        return TelemetrySample.measured(
            field_name,
            payload.get("value", 0),
            unit=str(payload.get("unit") or default_unit),
            sensor_id=sensor,
        )
    reason = payload.get("reason_code") or UnavailableReason.NOT_REPORTED.value
    return TelemetrySample.unavailable(
        field_name,
        _unavailable_reason_from_payload(reason),
        sensor_id=str(payload.get("sensor_id") or _work_sensor(span_id, field_name)),
    )


_COMPUTE_UNITS: Final[dict[str, str]] = {
    "cpu_seconds": UNIT_SECONDS_MILLIONTHS,
    "gpu_seconds": UNIT_SECONDS_MILLIONTHS,
    "peak_memory": UNIT_BYTES,
    "wall_clock_duration": UNIT_SECONDS_MILLIONTHS,
    "test_execution_time": UNIT_SECONDS_MILLIONTHS,
    "prover_execution_time": UNIT_SECONDS_MILLIONTHS,
    "static_analysis_time": UNIT_SECONDS_MILLIONTHS,
    "indexing_retrieval_time": UNIT_SECONDS_MILLIONTHS,
    "bytes_read": UNIT_BYTES,
    "bytes_written": UNIT_BYTES,
    "process_count": UNIT_COUNT,
    "concurrency": UNIT_COUNT,
    "audit_and_verification_overhead": UNIT_MICROUSD,
    "time_to_terminal_outcome": UNIT_SECONDS_MILLIONTHS,
}


def _covering_span_id_for_field(
    field_name: str,
    span: BenchmarkCausalSpan,
    *,
    session: BenchmarkTelemetrySession | None,
) -> str:
    kinds = _WORK_FIELD_SPAN_KINDS.get(field_name, ())
    if session is None or not kinds:
        return span.span_id
    children = [
        item
        for item in session.spans
        if item.parent_span_id == span.span_id and item.kind.value in kinds
    ]
    if not children:
        descendants = [
            item
            for item in session.spans
            if span.span_id in item.ancestry and item.kind.value in kinds
        ]
        children = descendants
    if not children:
        return span.span_id
    return sorted(children, key=lambda item: item.span_id)[0].span_id


def _child_span_duration_sample(
    field_name: str,
    span: BenchmarkCausalSpan,
    *,
    session: BenchmarkTelemetrySession | None,
) -> TelemetrySample | None:
    kinds = _WORK_FIELD_SPAN_KINDS.get(field_name, ())
    if session is None or not kinds:
        return None
    matching = [
        item
        for item in session.spans
        if (item.parent_span_id == span.span_id or span.span_id in item.ancestry)
        and item.kind.value in kinds
        and item.span_id != span.span_id
        and item.started_at_mono_ns
        and item.finished_at_mono_ns
    ]
    if not matching:
        return None
    total = sum(item.duration_seconds_millionths for item in matching)
    sensor = _work_sensor(span.span_id, field_name)
    return TelemetrySample.measured(
        field_name,
        total,
        unit=UNIT_SECONDS_MILLIONTHS,
        sensor_id=sensor,
    )


def project_compute_from_sensors(
    span: BenchmarkCausalSpan,
    *,
    process_samples: Mapping[str, TelemetrySample] | None = None,
    gpu_samples: Mapping[str, TelemetrySample] | None = None,
    extras: Mapping[str, TelemetrySample] | None = None,
    session: BenchmarkTelemetrySession | None = None,
) -> dict[str, TelemetrySample]:
    """Join process-tree, GPU, span-clock, and child-span timings onto compute fields.

    Absent sensors remain ``unavailable`` rather than numeric zero.
    """

    samples: dict[str, TelemetrySample] = {}
    process_samples = process_samples or {}
    gpu_samples = gpu_samples or {}
    extras = extras or {}
    span_id = span.span_id

    for source_name, dest_name in _PROCESS_TREE_TO_COMPUTE.items():
        source = process_samples.get(source_name)
        if source is not None:
            samples[dest_name] = _rename_sample(source, dest_name)

    gpu = gpu_samples.get("gpu_seconds")
    if gpu is not None:
        samples["gpu_seconds"] = _rename_sample(gpu, "gpu_seconds")
    elif span.hardware is not None and not span.hardware.accelerator_present:
        samples["gpu_seconds"] = TelemetrySample.unavailable(
            "gpu_seconds",
            UnavailableReason.HARDWARE_ABSENT,
            sensor_id=_work_sensor(span_id, "gpu_seconds"),
        )

    if span.started_at_mono_ns and span.finished_at_mono_ns:
        samples["wall_clock_duration"] = TelemetrySample.measured(
            "wall_clock_duration",
            span.duration_seconds_millionths,
            unit=UNIT_SECONDS_MILLIONTHS,
            sensor_id=_sensor_id("span-clock", span_id),
        )
        samples.setdefault(
            "time_to_terminal_outcome",
            TelemetrySample.measured(
                "time_to_terminal_outcome",
                span.duration_seconds_millionths,
                unit=UNIT_SECONDS_MILLIONTHS,
                sensor_id=_sensor_id("span-clock", span_id, "terminal"),
            ),
        )

    for timing_field in (
        "test_execution_time",
        "prover_execution_time",
        "static_analysis_time",
        "indexing_retrieval_time",
    ):
        derived = _child_span_duration_sample(
            timing_field, span, session=session
        )
        if derived is not None:
            samples.setdefault(timing_field, derived)

    for name, sample in extras.items():
        if name in COMPUTE_FIELD_NAMES or name in WORK_OVERHEAD_FIELD_NAMES:
            samples[name] = (
                sample if sample.metric_name == name else _rename_sample(sample, name)
            )

    for name in COMPUTE_FIELD_NAMES:
        samples.setdefault(
            name,
            _blank_compute_sample(
                name,
                span_id=span_id,
                reason=UnavailableReason.NOT_REPORTED,
            ),
        )
    return samples


def _assert_selected_executed_bounds(
    operations: Mapping[str, OperationObservation],
) -> None:
    for selected_name, executed_name in _SELECTED_EXECUTED_BOUNDS:
        selected = operations.get(selected_name)
        executed = operations.get(executed_name)
        if selected is None or executed is None:
            continue
        selected_count = selected.counted_value()
        executed_count = executed.counted_value()
        if selected_count is None or executed_count is None:
            continue
        if selected.truth_state is OperationTruthState.ATTEMPTED:
            continue
        if executed.truth_state is OperationTruthState.ATTEMPTED:
            continue
        if executed_count > selected_count:
            raise BenchmarkTelemetryError(
                f"{executed_name} count exceeds {selected_name} bound"
            )


def _verification_occurred(operations: Mapping[str, OperationObservation]) -> bool:
    for name in VERIFICATION_WORK_FIELDS:
        item = operations.get(name)
        if item is None:
            continue
        if item.truth_state in {
            OperationTruthState.OBSERVED,
            OperationTruthState.VERIFIED,
        }:
            return True
    return False


def _assert_admitted_verifier_linkage(
    operations: Mapping[str, OperationObservation],
    *,
    session: BenchmarkTelemetrySession | None,
    admitted_verifiers: Mapping[str, str] | None,
) -> None:
    admitted = dict(admitted_verifiers or {})
    if session is not None:
        admitted.update(session.admitted_verifiers)
    for name, item in operations.items():
        if item.truth_state is not OperationTruthState.VERIFIED:
            continue
        bound = admitted.get(item.verifier_id)
        if bound != item.verifier_receipt_cid:
            raise BenchmarkTelemetryError(
                f"{name}: observed operations cannot be represented as verified "
                "without admitted verifier linkage"
            )


@dataclass(frozen=True)
class WorkTelemetryRecord(_TelemetryContract):
    """Work and compute telemetry bound (or quarantined) against a causal span.

    Interface: BenchmarkWorkTelemetryRecord@1

    Admitted records cover every requested compute and work field with explicit
    availability. Verified operations require admitted verifier linkage. Missing
    audit overhead, duplicate terminal accounting, attempted-as-observed,
    observed-as-verified, and invalid bounds fail closed.
    """

    SCHEMA: ClassVar[str] = WORK_TELEMETRY_RECORD_SCHEMA
    INTERFACE: ClassVar[str] = WORK_TELEMETRY_RECORD_INTERFACE

    record_id: str
    disposition: WorkTelemetryDisposition
    span: BenchmarkCausalSpan | None
    compute: tuple[TelemetrySample, ...]
    operations: tuple[OperationObservation, ...]
    final_task_outcome: str
    patch_disposition: str
    audit_and_verification_overhead: TelemetrySample
    time_to_terminal_outcome: TelemetrySample
    terminalized: bool = True
    single_terminalization: bool = True
    covering_span_ids: tuple[tuple[str, str], ...] = ()
    admitted_verifier_ids: tuple[tuple[str, str], ...] = ()
    quarantine_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _text(self.record_id, "record_id"))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, WorkTelemetryDisposition, "disposition"),
        )
        span = self.span
        if isinstance(span, Mapping):
            span = BenchmarkCausalSpan.from_dict(span)
        if span is not None and not isinstance(span, BenchmarkCausalSpan):
            raise BenchmarkTelemetryError("span must be BenchmarkCausalSpan")
        object.__setattr__(self, "span", span)

        compute_items: list[TelemetrySample] = []
        for item in self.compute or ():
            if isinstance(item, Mapping):
                compute_items.append(TelemetrySample.from_dict(item))
            elif isinstance(item, TelemetrySample):
                compute_items.append(item)
            else:
                raise BenchmarkTelemetryError("compute samples must be TelemetrySample")
        compute_names = [item.metric_name for item in compute_items]
        if len(compute_names) != len(set(compute_names)):
            raise BenchmarkTelemetryError("compute contains duplicate metric names")
        missing_compute = [
            name for name in COMPUTE_FIELD_NAMES if name not in compute_names
        ]
        if missing_compute:
            raise BenchmarkTelemetryError(
                f"compute is missing required fields: {missing_compute}"
            )
        extra_compute = [
            name for name in compute_names if name not in COMPUTE_FIELD_NAMES
        ]
        if extra_compute:
            raise BenchmarkTelemetryError(
                f"compute contains unknown fields: {extra_compute}"
            )
        object.__setattr__(self, "compute", tuple(compute_items))

        operations: list[OperationObservation] = []
        for item in self.operations or ():
            if isinstance(item, Mapping):
                operations.append(OperationObservation.from_dict(item))
            elif isinstance(item, OperationObservation):
                operations.append(item)
            else:
                raise BenchmarkTelemetryError(
                    "operations must be OperationObservation records"
                )
        op_names = [item.field_name for item in operations]
        if len(op_names) != len(set(op_names)):
            raise BenchmarkTelemetryError("operations contain duplicate field names")
        missing_ops = [
            name for name in WORK_OPERATION_FIELD_NAMES if name not in op_names
        ]
        if missing_ops:
            raise BenchmarkTelemetryError(
                f"work is missing required fields: {missing_ops}"
            )
        extra_ops = [
            name for name in op_names if name not in WORK_OPERATION_FIELD_NAMES
        ]
        if extra_ops:
            raise BenchmarkTelemetryError(
                f"work contains unknown fields: {extra_ops}"
            )
        object.__setattr__(self, "operations", tuple(operations))

        object.__setattr__(
            self,
            "final_task_outcome",
            _text(self.final_task_outcome, "final_task_outcome", required=False),
        )
        object.__setattr__(
            self,
            "patch_disposition",
            _text(self.patch_disposition, "patch_disposition", required=False),
        )
        if self.final_task_outcome and self.final_task_outcome not in TASK_OUTCOMES:
            raise BenchmarkTelemetryError("final_task_outcome is not a closed value")
        if (
            self.patch_disposition
            and self.patch_disposition not in PATCH_DISPOSITIONS
        ):
            raise BenchmarkTelemetryError("patch_disposition is not a closed value")

        def _sample(value: Any, name: str) -> TelemetrySample:
            if isinstance(value, Mapping):
                sample = TelemetrySample.from_dict(value)
            elif isinstance(value, TelemetrySample):
                sample = value
            else:
                raise BenchmarkTelemetryError(f"{name} must be a TelemetrySample")
            if sample.metric_name != name:
                sample = _rename_sample(sample, name)
            return sample

        object.__setattr__(
            self,
            "audit_and_verification_overhead",
            _sample(
                self.audit_and_verification_overhead,
                "audit_and_verification_overhead",
            ),
        )
        object.__setattr__(
            self,
            "time_to_terminal_outcome",
            _sample(self.time_to_terminal_outcome, "time_to_terminal_outcome"),
        )
        if not isinstance(self.terminalized, bool):
            raise BenchmarkTelemetryError("terminalized must be a boolean")
        if not isinstance(self.single_terminalization, bool):
            raise BenchmarkTelemetryError("single_terminalization must be a boolean")
        if self.terminalized and not self.single_terminalization:
            raise BenchmarkTelemetryError(
                "duplicate terminal accounting: single_terminalization must be true"
            )

        covering: list[tuple[str, str]] = []
        seen_fields: set[str] = set()
        for pair in self.covering_span_ids or ():
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise BenchmarkTelemetryError(
                    "covering_span_ids must be (field, span_id) pairs"
                )
            field_name = _text(pair[0], "covering_field")
            span_id = _text(pair[1], "covering_span_id")
            if field_name in seen_fields:
                raise BenchmarkTelemetryError(
                    "covering_span_ids contains duplicate fields"
                )
            seen_fields.add(field_name)
            covering.append((field_name, span_id))
        object.__setattr__(self, "covering_span_ids", tuple(sorted(covering)))

        verifiers: list[tuple[str, str]] = []
        seen_verifiers: set[str] = set()
        for pair in self.admitted_verifier_ids or ():
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise BenchmarkTelemetryError(
                    "admitted_verifier_ids must be (verifier_id, cid) pairs"
                )
            verifier_id = _text(pair[0], "verifier_id")
            cid = _require_cid(pair[1], "verifier_receipt_cid")
            if verifier_id in seen_verifiers:
                raise BenchmarkTelemetryError(
                    "admitted_verifier_ids contains duplicate verifier identities"
                )
            seen_verifiers.add(verifier_id)
            verifiers.append((verifier_id, cid))
        object.__setattr__(
            self, "admitted_verifier_ids", tuple(sorted(verifiers))
        )
        object.__setattr__(
            self,
            "quarantine_reason",
            _text(self.quarantine_reason, "quarantine_reason", required=False),
        )

        if self.disposition is WorkTelemetryDisposition.ADMITTED:
            if span is None or not span.task_id:
                raise BenchmarkTelemetryError(
                    "admitted work telemetry requires a causal task span"
                )
            if self.quarantine_reason:
                raise BenchmarkTelemetryError(
                    "admitted work telemetry cannot carry a quarantine reason"
                )
            if not self.final_task_outcome or not self.patch_disposition:
                raise BenchmarkTelemetryError(
                    "admitted work telemetry requires outcome and patch disposition"
                )
            if not self.terminalized:
                raise BenchmarkTelemetryError(
                    "admitted work telemetry must be terminalized exactly once"
                )
            _assert_selected_executed_bounds(
                {item.field_name: item for item in self.operations}
            )
            if _verification_occurred(
                {item.field_name: item for item in self.operations}
            ) and self.audit_and_verification_overhead.status is SampleStatus.UNAVAILABLE:
                raise BenchmarkTelemetryError(
                    "missing audit and verification overhead"
                )
            required_cover = set(WORK_METRIC_NAMES) | set(WORK_ENUM_FIELD_NAMES)
            covered = {field for field, _span in self.covering_span_ids}
            if not required_cover <= covered:
                missing = sorted(required_cover - covered)
                raise BenchmarkTelemetryError(
                    f"causal spans do not cover required fields: {missing}"
                )
        else:
            if not self.quarantine_reason:
                raise BenchmarkTelemetryError(
                    "quarantined work telemetry requires a quarantine_reason"
                )
            try:
                WorkTelemetryQuarantineReason(self.quarantine_reason)
            except ValueError as exc:
                raise BenchmarkTelemetryError(
                    "quarantine_reason is not a supported quarantine reason"
                ) from exc

        for sample in self.compute:
            if (
                sample.status is SampleStatus.UNAVAILABLE
                and (sample.value != 0 or sample.unit)
            ):
                raise BenchmarkTelemetryError(
                    "unavailable compute fields must not encode numeric zero"
                )

    @property
    def span_id(self) -> str:
        return "" if self.span is None else self.span.span_id

    @property
    def task_id(self) -> str:
        return "" if self.span is None else self.span.task_id

    @property
    def admitted(self) -> bool:
        return self.disposition is WorkTelemetryDisposition.ADMITTED

    def compute_map(self) -> dict[str, TelemetrySample]:
        return {item.metric_name: item for item in self.compute}

    def operation_map(self) -> dict[str, OperationObservation]:
        return {item.field_name: item for item in self.operations}

    def covering_span_map(self) -> dict[str, str]:
        return {field: span_id for field, span_id in self.covering_span_ids}

    def explicit_availability(self) -> dict[str, str]:
        availability: dict[str, str] = {
            name: sample.status.value for name, sample in self.compute_map().items()
        }
        for name, operation in self.operation_map().items():
            availability[name] = operation.truth_state.value
        availability["audit_and_verification_overhead"] = (
            self.audit_and_verification_overhead.status.value
        )
        availability["time_to_terminal_outcome"] = (
            self.time_to_terminal_outcome.status.value
        )
        availability["final_task_outcome"] = "observed"
        availability["patch_disposition"] = "observed"
        return availability

    def sample_map(self) -> dict[str, TelemetrySample]:
        samples = dict(self.compute_map())
        samples["audit_and_verification_overhead"] = (
            self.audit_and_verification_overhead
        )
        samples["time_to_terminal_outcome"] = self.time_to_terminal_outcome
        for operation in self.operations:
            samples[operation.field_name] = operation.to_quantity_sample(
                span_id=self.span_id
            )
        return samples

    def unavailable_fields(self) -> tuple[str, ...]:
        names: list[str] = []
        for name, sample in self.compute_map().items():
            if sample.status is SampleStatus.UNAVAILABLE:
                names.append(f"compute.{name}")
        for name, operation in self.operation_map().items():
            if operation.truth_state is OperationTruthState.UNAVAILABLE:
                names.append(f"work.{name}")
        if self.audit_and_verification_overhead.status is SampleStatus.UNAVAILABLE:
            names.append("cost.audit_and_verification_overhead")
        if self.time_to_terminal_outcome.status is SampleStatus.UNAVAILABLE:
            names.append("terminal.time_to_terminal_outcome")
        return tuple(sorted(names))

    def to_compute_fields(self) -> dict[str, Any]:
        return {
            name: self.compute_map()[name].to_quantity_envelope()
            for name in COMPUTE_FIELD_NAMES
        }

    def to_work_fields(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            name: self.operation_map()[name].to_operation_envelope()
            for name in WORK_OPERATION_FIELD_NAMES
        }
        payload["final_task_outcome"] = self.final_task_outcome
        payload["patch_disposition"] = self.patch_disposition
        return payload

    def to_terminal_fields(self) -> dict[str, Any]:
        return {
            "final_task_outcome": self.final_task_outcome,
            "validation_result": self.operation_map()[
                "validation_result"
            ].to_operation_envelope(),
            "patch_disposition": self.patch_disposition,
            "time_to_terminal_outcome": (
                self.time_to_terminal_outcome.to_quantity_envelope()
            ),
            "terminalized": True if self.admitted else self.terminalized,
            "single_terminalization": self.single_terminalization,
        }

    def to_resource_measurement(self) -> BenchmarkResourceMeasurement:
        if (
            self.disposition is not WorkTelemetryDisposition.ADMITTED
            or self.span is None
        ):
            raise BenchmarkTelemetryError(
                "quarantined unbound work cannot seal a resource measurement"
            )
        source_ids = [self.span.span_id]
        for _field, span_id in self.covering_span_ids:
            if span_id not in source_ids:
                source_ids.append(span_id)
        attributed = [
            self.span.process_id
        ] if self.span.process_id else []
        return build_resource_measurement(
            measurement_id=f"meas:work-telemetry:{self.record_id}",
            span=self.span,
            samples=self.sample_map(),
            attributed_process_ids=attributed,
            source_span_ids=source_ids,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": BENCHMARK_TELEMETRY_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "record_id": self.record_id,
            "disposition": self.disposition.value,
            "span": None if self.span is None else self.span.to_record(),
            "compute": [item.to_record() for item in self.compute],
            "operations": [item.to_record() for item in self.operations],
            "final_task_outcome": self.final_task_outcome,
            "patch_disposition": self.patch_disposition,
            "audit_and_verification_overhead": (
                self.audit_and_verification_overhead.to_record()
            ),
            "time_to_terminal_outcome": self.time_to_terminal_outcome.to_record(),
            "terminalized": self.terminalized,
            "single_terminalization": self.single_terminalization,
            "covering_span_ids": [list(pair) for pair in self.covering_span_ids],
            "admitted_verifier_ids": [
                list(pair) for pair in self.admitted_verifier_ids
            ],
            "quarantine_reason": self.quarantine_reason,
            "unavailable_fields": list(self.unavailable_fields()),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkTelemetryRecord":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "interface",
            "record_id",
            "disposition",
            "span",
            "compute",
            "operations",
            "final_task_outcome",
            "patch_disposition",
            "audit_and_verification_overhead",
            "time_to_terminal_outcome",
            "terminalized",
            "single_terminalization",
            "covering_span_ids",
            "admitted_verifier_ids",
            "quarantine_reason",
            "unavailable_fields",
            "content_id",
        }
        _closed(
            payload,
            schema=cls.SCHEMA,
            allowed=allowed,
            name="work telemetry record",
        )
        result = cls(
            record_id=payload.get("record_id", ""),
            disposition=payload.get("disposition", ""),
            span=payload.get("span"),
            compute=tuple(payload.get("compute") or ()),
            operations=tuple(payload.get("operations") or ()),
            final_task_outcome=payload.get("final_task_outcome", ""),
            patch_disposition=payload.get("patch_disposition", ""),
            audit_and_verification_overhead=payload.get(
                "audit_and_verification_overhead", {}
            ),
            time_to_terminal_outcome=payload.get("time_to_terminal_outcome", {}),
            terminalized=bool(payload.get("terminalized", True)),
            single_terminalization=bool(payload.get("single_terminalization", True)),
            covering_span_ids=tuple(
                tuple(item)
                for item in (payload.get("covering_span_ids") or ())
            ),
            admitted_verifier_ids=tuple(
                tuple(item)
                for item in (payload.get("admitted_verifier_ids") or ())
            ),
            quarantine_reason=payload.get("quarantine_reason", ""),
        )
        if payload.get("interface", result.INTERFACE) != result.INTERFACE:
            raise BenchmarkTelemetryError("work telemetry interface mismatch")
        claimed = payload.get("unavailable_fields")
        if claimed is not None and tuple(claimed) != result.unavailable_fields():
            raise BenchmarkTelemetryError(
                "unavailable_fields does not match work/compute samples"
            )
        _claim(payload, result.content_id, "content_id")
        return result


def _quarantine_work_record(
    *,
    record_id: str,
    span: BenchmarkCausalSpan | None,
    reason: WorkTelemetryQuarantineReason,
) -> WorkTelemetryRecord:
    span_id = "" if span is None else span.span_id
    compute = tuple(
        _blank_compute_sample(name, span_id=span_id, quarantined=True)
        for name in COMPUTE_FIELD_NAMES
    )
    operations = tuple(
        OperationObservation.unavailable(
            name,
            UnavailableReason.NOT_ADMITTED,
            covering_span_id=span_id,
        )
        for name in WORK_OPERATION_FIELD_NAMES
    )
    covering = tuple(
        (name, span_id or "unbound")
        for name in (*WORK_METRIC_NAMES, *WORK_ENUM_FIELD_NAMES)
    )
    return WorkTelemetryRecord(
        record_id=record_id,
        disposition=WorkTelemetryDisposition.QUARANTINED,
        span=span,
        compute=compute,
        operations=operations,
        final_task_outcome=TaskOutcome.QUARANTINED.value,
        patch_disposition=PatchDisposition.QUARANTINED.value,
        audit_and_verification_overhead=_blank_compute_sample(
            "audit_and_verification_overhead",
            span_id=span_id,
            quarantined=True,
        ),
        time_to_terminal_outcome=_blank_compute_sample(
            "time_to_terminal_outcome",
            span_id=span_id,
            quarantined=True,
        ),
        terminalized=False,
        single_terminalization=True,
        covering_span_ids=covering,
        admitted_verifier_ids=(),
        quarantine_reason=reason.value,
    )


def _merge_mapping(
    base: Mapping[str, Any] | None, overlay: Mapping[str, Any] | None
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(base, Mapping):
        merged.update(base)
    if isinstance(overlay, Mapping):
        merged.update(overlay)
    return merged


def map_work_and_compute_to_admitted_span(
    payload: Mapping[str, Any] | None,
    span: BenchmarkCausalSpan | None,
    *,
    session: BenchmarkTelemetrySession | None = None,
    admitted_span_ids: Container[str] | None = None,
    admitted_verifiers: Mapping[str, str] | None = None,
    record_id: str | None = None,
    compute: Mapping[str, Any] | None = None,
    work: Mapping[str, Any] | None = None,
    audit_and_verification_overhead: Mapping[str, Any]
    | TelemetrySample
    | None = None,
    time_to_terminal_outcome: Mapping[str, Any] | TelemetrySample | None = None,
    process_samples: Mapping[str, TelemetrySample] | None = None,
    gpu_samples: Mapping[str, TelemetrySample] | None = None,
    terminalized: bool | None = None,
) -> WorkTelemetryRecord:
    """Bind work and compute fields to an admitted causal task span.

    Every requested field is present with explicit availability. Verified
    operations require admitted verifier linkage. Fail-closed errors cover
    attempted-as-observed, observed-as-verified, missing overhead, duplicate
    terminal accounting, and invalid bounds. Unbound or anonymous spans are
    quarantined rather than admitted.
    """

    if payload is not None and not isinstance(payload, Mapping):
        raise BenchmarkTelemetryError("work telemetry payload must be an object")
    body = dict(payload or {})
    identity = record_id or _sensor_id(
        "work-telemetry-record",
        "" if span is None else span.span_id,
        str(body.get("record_id") or "anonymous"),
    )

    if not _span_has_causal_task_identity(span):
        return _quarantine_work_record(
            record_id=identity,
            span=span,
            reason=WorkTelemetryQuarantineReason.MISSING_CAUSAL_IDENTITY,
        )
    if not _span_is_admitted_for_usage(
        span, session=session, admitted_span_ids=admitted_span_ids
    ):
        return _quarantine_work_record(
            record_id=identity,
            span=span,
            reason=WorkTelemetryQuarantineReason.UNBOUND_WORK,
        )

    assert span is not None
    span_id = span.span_id
    if (
        terminalized is not False
        and (body.get("terminalized", True) is True)
        and session is not None
        and span_id in session.terminalized_span_ids
    ):
        raise BenchmarkTelemetryError(
            "duplicate terminal accounting for admitted work span"
        )

    compute_payload = _merge_mapping(body.get("compute"), compute)
    work_payload = _merge_mapping(body.get("work"), work)
    for name in WORK_FIELD_NAMES:
        if name in body and name not in work_payload:
            work_payload[name] = body[name]
    for name in COMPUTE_FIELD_NAMES:
        if name in body and name not in compute_payload:
            compute_payload[name] = body[name]

    extras: dict[str, TelemetrySample] = {}
    for name, raw in compute_payload.items():
        if name not in COMPUTE_FIELD_NAMES:
            raise BenchmarkTelemetryError(
                f"compute contains unknown fields: {[name]}"
            )
        extras[name] = _compute_sample_from_payload(
            name,
            raw,
            span_id=span_id,
            default_unit=_COMPUTE_UNITS[name],
        )

    projected = project_compute_from_sensors(
        span,
        process_samples=process_samples,
        gpu_samples=gpu_samples,
        extras=extras,
        session=session,
    )
    compute_samples = tuple(projected[name] for name in COMPUTE_FIELD_NAMES)

    operations: dict[str, OperationObservation] = {}
    for name in WORK_OPERATION_FIELD_NAMES:
        covering = _covering_span_id_for_field(name, span, session=session)
        operations[name] = _operation_from_payload(
            name,
            work_payload.get(name),
            covering_span_id=covering,
        )
    extra_work = [
        key
        for key in work_payload
        if key not in WORK_FIELD_NAMES
    ]
    if extra_work:
        raise BenchmarkTelemetryError(
            f"work contains unknown fields: {sorted(extra_work)}"
        )

    _assert_selected_executed_bounds(operations)
    _assert_admitted_verifier_linkage(
        operations, session=session, admitted_verifiers=admitted_verifiers
    )

    outcome = work_payload.get("final_task_outcome", body.get("final_task_outcome"))
    disposition_value = work_payload.get(
        "patch_disposition", body.get("patch_disposition")
    )
    if not isinstance(outcome, str) or not outcome.strip():
        raise BenchmarkTelemetryError("final_task_outcome is required")
    if not isinstance(disposition_value, str) or not disposition_value.strip():
        raise BenchmarkTelemetryError("patch_disposition is required")
    outcome = _enum(outcome, TaskOutcome, "final_task_outcome").value
    disposition_value = _enum(
        disposition_value, PatchDisposition, "patch_disposition"
    ).value

    overhead_raw = audit_and_verification_overhead
    if overhead_raw is None:
        overhead_raw = body.get("audit_and_verification_overhead")
    if overhead_raw is None and "cost" in body and isinstance(body["cost"], Mapping):
        overhead_raw = body["cost"].get("audit_and_verification_overhead")
    if overhead_raw is None:
        raise BenchmarkTelemetryError("missing audit and verification overhead")
    overhead = _compute_sample_from_payload(
        "audit_and_verification_overhead",
        overhead_raw,
        span_id=span_id,
        default_unit=UNIT_MICROUSD,
    )
    if _verification_occurred(operations) and overhead.status is SampleStatus.UNAVAILABLE:
        raise BenchmarkTelemetryError("missing audit and verification overhead")

    terminal_raw = time_to_terminal_outcome
    if terminal_raw is None:
        terminal_raw = body.get("time_to_terminal_outcome")
    if terminal_raw is None and "terminal" in body and isinstance(body["terminal"], Mapping):
        terminal_raw = body["terminal"].get("time_to_terminal_outcome")
    if terminal_raw is None:
        terminal_sample = projected.get(
            "time_to_terminal_outcome",
            _blank_compute_sample("time_to_terminal_outcome", span_id=span_id),
        )
        if terminal_sample.metric_name != "time_to_terminal_outcome":
            terminal_sample = _rename_sample(
                terminal_sample, "time_to_terminal_outcome"
            )
    else:
        terminal_sample = _compute_sample_from_payload(
            "time_to_terminal_outcome",
            terminal_raw,
            span_id=span_id,
            default_unit=UNIT_SECONDS_MILLIONTHS,
        )

    is_terminal = True if terminalized is None else bool(terminalized)
    if "terminalized" in body:
        if not isinstance(body["terminalized"], bool):
            raise BenchmarkTelemetryError("terminalized must be a boolean")
        is_terminal = bool(body["terminalized"])
    single = True
    if "single_terminalization" in body:
        if body["single_terminalization"] is not True:
            raise BenchmarkTelemetryError(
                "duplicate terminal accounting: single_terminalization must be true"
            )

    covering_pairs: list[tuple[str, str]] = []
    for name in COMPUTE_FIELD_NAMES:
        covering_pairs.append(
            (name, _covering_span_id_for_field(name, span, session=session))
        )
    for name in WORK_OPERATION_FIELD_NAMES:
        covering_pairs.append((name, operations[name].covering_span_id or span_id))
    covering_pairs.append(
        (
            "audit_and_verification_overhead",
            _covering_span_id_for_field(
                "audit_and_verification_overhead", span, session=session
            ),
        )
    )
    covering_pairs.append(
        (
            "time_to_terminal_outcome",
            _covering_span_id_for_field(
                "time_to_terminal_outcome", span, session=session
            ),
        )
    )
    covering_pairs.append(("final_task_outcome", span_id))
    covering_pairs.append(("patch_disposition", span_id))

    verifier_pairs: list[tuple[str, str]] = []
    admitted = dict(admitted_verifiers or {})
    if session is not None:
        admitted.update(session.admitted_verifiers)
    for item in operations.values():
        if item.truth_state is OperationTruthState.VERIFIED:
            verifier_pairs.append((item.verifier_id, item.verifier_receipt_cid))

    return WorkTelemetryRecord(
        record_id=identity,
        disposition=WorkTelemetryDisposition.ADMITTED,
        span=span,
        compute=compute_samples,
        operations=tuple(operations[name] for name in WORK_OPERATION_FIELD_NAMES),
        final_task_outcome=outcome,
        patch_disposition=disposition_value,
        audit_and_verification_overhead=overhead,
        time_to_terminal_outcome=terminal_sample,
        terminalized=is_terminal,
        single_terminalization=single,
        covering_span_ids=tuple(covering_pairs),
        admitted_verifier_ids=tuple(sorted(set(verifier_pairs))),
    )


def project_work_telemetry_samples(
    record: WorkTelemetryRecord,
) -> dict[str, TelemetrySample]:
    """Project a work-telemetry record onto named telemetry samples."""

    if not isinstance(record, WorkTelemetryRecord):
        raise BenchmarkTelemetryError("record must be WorkTelemetryRecord")
    return record.sample_map()


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
    "COMPUTE_FIELD_NAMES",
    "ESTIMATOR_METHODS",
    "EstimatorMethod",
    "GPU_METRIC_NAMES",
    "MODEL_CALL_CLASSES",
    "MILLIONTHS",
    "ModelCallClass",
    "OPERATION_OBSERVATION_SCHEMA",
    "OPERATION_REASON_CODES",
    "OperationObservation",
    "OperationTruthState",
    "PATCH_DISPOSITIONS",
    "PROCESS_TREE_METRIC_NAMES",
    "PROVIDER_USAGE_METRIC_NAMES",
    "PROVIDER_USAGE_RECORD_INTERFACE",
    "PROVIDER_USAGE_RECORD_SCHEMA",
    "PatchDisposition",
    "ProviderUsageDisposition",
    "ProviderUsageQuarantineReason",
    "ProviderUsageRecord",
    "SCHEMA_VERSION",
    "SPAN_REPLAY_CERTIFICATE_SCHEMA",
    "SampleStatus",
    "SpanKind",
    "SpanReplayCertificate",
    "TASK_OUTCOMES",
    "TOKEN_METRIC_NAMES",
    "TELEMETRY_SAMPLE_SCHEMA",
    "TaskOutcome",
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
    "WORK_FIELD_NAMES",
    "WORK_METRIC_NAMES",
    "WORK_OPERATION_FIELD_NAMES",
    "WORK_TELEMETRY_RECORD_INTERFACE",
    "WORK_TELEMETRY_RECORD_SCHEMA",
    "WorkTelemetryDisposition",
    "WorkTelemetryQuarantineReason",
    "WorkTelemetryRecord",
    "build_resource_measurement",
    "build_span_joined_measurement",
    "certify_measurement_from_source_spans",
    "collect_descendant_pids",
    "extract_safe_provider_request_ids",
    "is_safe_provider_request_id",
    "map_provider_response_to_admitted_span",
    "map_work_and_compute_to_admitted_span",
    "mono_ns",
    "observe_wall_seconds_millionths",
    "project_compute_from_sensors",
    "project_provider_usage_samples",
    "project_scheduler_clock_samples",
    "project_token_ledger_samples",
    "project_work_telemetry_samples",
    "provider_response_contains_credentials",
    "reject_self_certified_counters",
    "sample_energy_optional",
    "sample_gpu_resources",
    "sample_network_bytes",
    "sample_process_tree_resources",
    "sample_rusage_self",
    "sample_self_process_tree_resources",
    "seconds_to_millionths",
]
