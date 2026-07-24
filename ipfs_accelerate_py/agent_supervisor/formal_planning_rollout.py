"""Assurance- and throughput-aware rollout gate for formal planning.

The gate consumes only the bounded projection produced by
``formal_planning_metrics``.  It is conservative by construction:

* shadow, canary, and enforcement limits are separately reviewed values;
* unavailable and consistently low-value lanes stay advisory;
* a scoped override may waive named metric failures but cannot manufacture
  availability, value, evidence, or assurance; and
* the operator view exposes identities, counts, reasons, and decisions without
  prompts, raw context, proof bodies, or trace payloads.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Final

from .formal_planning_metrics import (
    BenchmarkMode,
    FormalPlanningBenchmarkReport,
    FormalPlanningMetricDimensions,
)
from .formal_verification_contracts import AssuranceLevel
from .formal_verification_policy import RolloutMode


FORMAL_PLANNING_ROLLOUT_VERSION: Final = 1
FORMAL_PLANNING_ROLLOUT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-planning-rollout-decision@1"
)
FORMAL_PLANNING_OPERATOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-planning-operator-projection@1"
)
FORMAL_PLANNING_OVERRIDE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-planning-rollout-override@1"
)
MAX_OVERRIDE_SECONDS: Final = 7 * 24 * 60 * 60
MAX_OPERATOR_ITEMS: Final = 512
MAX_OPERATOR_BYTES: Final = 2 * 1024 * 1024
NON_WAIVABLE_FAILURES: Final = frozenset(
    {
        "authoritative_assurance_below_threshold",
        "benchmark_measurement_missing",
        "cold_benchmark_missing",
        "warm_benchmark_missing",
        "cold_sample_count_below_threshold",
        "warm_sample_count_below_threshold",
    }
)


class FormalPlanningRolloutError(ValueError):
    """Raised for unsafe thresholds, overrides, or rollout projections."""


class RolloutDisposition(str, Enum):
    PROMOTE = "promote"
    HOLD = "hold"
    ADVISORY = "advisory"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _text(value: Any, name: str) -> str:
    if isinstance(value, Enum):
        value = value.value
    result = str(value or "").strip()
    if not result:
        raise FormalPlanningRolloutError(f"{name} must be non-empty")
    if len(result.encode()) > 512:
        raise FormalPlanningRolloutError(f"{name} exceeds its byte bound")
    return result


def _timestamp(value: datetime | str | None = None) -> tuple[str, datetime]:
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise FormalPlanningRolloutError("invalid ISO timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat().replace("+00:00", "Z"), parsed


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise FormalPlanningRolloutError(f"{name} must be a fraction")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FormalPlanningRolloutError(f"{name} must be a fraction") from exc
    if not 0 <= number <= 1:
        raise FormalPlanningRolloutError(f"{name} must be between zero and one")
    return number


@dataclass(frozen=True)
class RolloutThresholds:
    """Explicit promotion thresholds for one rollout mode."""

    min_context_token_reduction: float
    min_pre_dispatch_defects: int
    min_proof_support_rate: float
    min_counterexample_quality: float
    min_warm_cache_reuse_rate: float
    max_queue_latency_ms: float
    max_cpu_saturation_percent: float
    max_memory_peak_bytes: int
    min_throughput_ratio: float
    minimum_assurance: AssuranceLevel | str
    require_cold_and_warm: bool = True
    min_samples_per_mode: int = 1

    def __post_init__(self) -> None:
        for name in (
            "min_context_token_reduction",
            "min_proof_support_rate",
            "min_counterexample_quality",
            "min_warm_cache_reuse_rate",
            "min_throughput_ratio",
        ):
            object.__setattr__(self, name, _fraction(getattr(self, name), name))
        for name in (
            "min_pre_dispatch_defects",
            "max_memory_peak_bytes",
            "min_samples_per_mode",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise FormalPlanningRolloutError(
                    f"{name} must be a non-negative integer"
                )
        if self.max_memory_peak_bytes <= 0 or self.min_samples_per_mode <= 0:
            raise FormalPlanningRolloutError(
                "memory and sample thresholds must be positive"
            )
        for name in ("max_queue_latency_ms", "max_cpu_saturation_percent"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError, OverflowError) as exc:
                raise FormalPlanningRolloutError(f"{name} must be positive") from exc
            if value <= 0:
                raise FormalPlanningRolloutError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.max_cpu_saturation_percent > 100:
            raise FormalPlanningRolloutError(
                "max_cpu_saturation_percent cannot exceed 100"
            )
        try:
            object.__setattr__(
                self, "minimum_assurance", AssuranceLevel(self.minimum_assurance)
            )
        except ValueError as exc:
            raise FormalPlanningRolloutError("invalid minimum assurance") from exc
        if not isinstance(self.require_cold_and_warm, bool):
            raise FormalPlanningRolloutError(
                "require_cold_and_warm must be a boolean"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: (
                value.value if isinstance(value := getattr(self, name), Enum) else value
            )
            for name in self.__dataclass_fields__
        }


DEFAULT_ROLLOUT_THRESHOLDS: Final[Mapping[RolloutMode, RolloutThresholds]] = {
    RolloutMode.SHADOW: RolloutThresholds(
        min_context_token_reduction=0.05,
        min_pre_dispatch_defects=0,
        min_proof_support_rate=0.40,
        min_counterexample_quality=0.30,
        min_warm_cache_reuse_rate=0.20,
        max_queue_latency_ms=750,
        max_cpu_saturation_percent=95,
        max_memory_peak_bytes=4 * 1024**3,
        min_throughput_ratio=0.65,
        minimum_assurance=AssuranceLevel.CANDIDATE,
    ),
    RolloutMode.CANARY: RolloutThresholds(
        min_context_token_reduction=0.15,
        min_pre_dispatch_defects=1,
        min_proof_support_rate=0.65,
        min_counterexample_quality=0.55,
        min_warm_cache_reuse_rate=0.45,
        max_queue_latency_ms=400,
        max_cpu_saturation_percent=90,
        max_memory_peak_bytes=3 * 1024**3,
        min_throughput_ratio=0.80,
        minimum_assurance=AssuranceLevel.SOLVER_CHECKED,
    ),
    RolloutMode.ENFORCEMENT: RolloutThresholds(
        min_context_token_reduction=0.25,
        min_pre_dispatch_defects=1,
        min_proof_support_rate=0.80,
        min_counterexample_quality=0.70,
        min_warm_cache_reuse_rate=0.60,
        max_queue_latency_ms=250,
        max_cpu_saturation_percent=85,
        max_memory_peak_bytes=2 * 1024**3,
        min_throughput_ratio=0.90,
        minimum_assurance=AssuranceLevel.KERNEL_VERIFIED,
    ),
}


@dataclass(frozen=True)
class FormalPlanningRolloutPolicy:
    """Reviewed thresholds for every mode that can execute formal work."""

    policy_id: str = "formal-planning-rollout-v1"
    shadow: RolloutThresholds = field(
        default_factory=lambda: DEFAULT_ROLLOUT_THRESHOLDS[RolloutMode.SHADOW]
    )
    canary: RolloutThresholds = field(
        default_factory=lambda: DEFAULT_ROLLOUT_THRESHOLDS[RolloutMode.CANARY]
    )
    enforcement: RolloutThresholds = field(
        default_factory=lambda: DEFAULT_ROLLOUT_THRESHOLDS[RolloutMode.ENFORCEMENT]
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        for name in ("shadow", "canary", "enforcement"):
            value = getattr(self, name)
            if not isinstance(value, RolloutThresholds):
                object.__setattr__(self, name, RolloutThresholds(**dict(value)))

    def for_mode(self, mode: RolloutMode | str) -> RolloutThresholds:
        normalized = RolloutMode(mode)
        if normalized is RolloutMode.DISABLED:
            raise FormalPlanningRolloutError("disabled mode has no promotion threshold")
        return {
            RolloutMode.SHADOW: self.shadow,
            RolloutMode.CANARY: self.canary,
            RolloutMode.ENFORCEMENT: self.enforcement,
        }[normalized]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "thresholds": {
                RolloutMode.SHADOW.value: self.shadow.to_dict(),
                RolloutMode.CANARY.value: self.canary.to_dict(),
                RolloutMode.ENFORCEMENT.value: self.enforcement.to_dict(),
            },
        }


@dataclass(frozen=True)
class FormalPlanningRolloutOverride:
    """Content-addressed, expiring waiver for one exact matrix lane."""

    policy_id: str
    lane_id: str
    target_mode: RolloutMode | str
    waived_reason_codes: tuple[str, ...]
    actor: str
    reason: str
    issued_at: str
    expires_at: str
    dimensions: FormalPlanningMetricDimensions | Mapping[str, Any]
    ticket_id: str = ""

    def __post_init__(self) -> None:
        for name in ("policy_id", "lane_id", "actor", "reason"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        try:
            mode = RolloutMode(self.target_mode)
        except ValueError as exc:
            raise FormalPlanningRolloutError("invalid override target mode") from exc
        if mode not in (RolloutMode.CANARY, RolloutMode.ENFORCEMENT):
            raise FormalPlanningRolloutError(
                "overrides apply only to canary or enforcement"
            )
        object.__setattr__(self, "target_mode", mode)
        reasons = tuple(
            sorted({_text(item, "waived_reason_code") for item in self.waived_reason_codes})
        )
        if not reasons:
            raise FormalPlanningRolloutError("override must name waived failures")
        object.__setattr__(self, "waived_reason_codes", reasons)
        dimensions = FormalPlanningMetricDimensions.from_mapping(self.dimensions)
        if dimensions.lane_id != self.lane_id:
            raise FormalPlanningRolloutError("override dimensions do not match lane_id")
        object.__setattr__(self, "dimensions", dimensions)
        issued_text, issued = _timestamp(self.issued_at)
        expires_text, expires = _timestamp(self.expires_at)
        if expires <= issued or (expires - issued).total_seconds() > MAX_OVERRIDE_SECONDS:
            raise FormalPlanningRolloutError("override lifetime is invalid")
        object.__setattr__(self, "issued_at", issued_text)
        object.__setattr__(self, "expires_at", expires_text)
        if self.ticket_id:
            object.__setattr__(self, "ticket_id", _text(self.ticket_id, "ticket_id"))

    @property
    def receipt_id(self) -> str:
        return _digest(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": FORMAL_PLANNING_OVERRIDE_SCHEMA,
            "policy_id": self.policy_id,
            "lane_id": self.lane_id,
            "target_mode": self.target_mode.value,
            "waived_reason_codes": list(self.waived_reason_codes),
            "actor": self.actor,
            "reason": self.reason,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "dimensions": self.dimensions.to_dict(),
            "ticket_id": self.ticket_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "receipt_id": self.receipt_id}

    def is_valid(
        self,
        *,
        policy_id: str,
        lane_id: str,
        target_mode: RolloutMode,
        now: datetime | str | None = None,
    ) -> bool:
        _, current = _timestamp(now)
        _, issued = _timestamp(self.issued_at)
        _, expires = _timestamp(self.expires_at)
        return (
            self.policy_id == policy_id
            and self.lane_id == lane_id
            and self.target_mode is target_mode
            and issued <= current < expires
        )

    @classmethod
    def create(
        cls,
        *,
        policy_id: str,
        dimensions: FormalPlanningMetricDimensions | Mapping[str, Any],
        target_mode: RolloutMode | str,
        waived_reason_codes: Sequence[str],
        actor: str,
        reason: str,
        ttl_seconds: int,
        now: datetime | str | None = None,
        ticket_id: str = "",
    ) -> "FormalPlanningRolloutOverride":
        if isinstance(ttl_seconds, bool) or not 0 < ttl_seconds <= MAX_OVERRIDE_SECONDS:
            raise FormalPlanningRolloutError("ttl_seconds is outside its hard bound")
        normalized = FormalPlanningMetricDimensions.from_mapping(dimensions)
        issued_text, issued = _timestamp(now)
        expires = issued + timedelta(seconds=ttl_seconds)
        return cls(
            policy_id=policy_id,
            lane_id=normalized.lane_id,
            target_mode=target_mode,
            waived_reason_codes=tuple(waived_reason_codes),
            actor=actor,
            reason=reason,
            issued_at=issued_text,
            expires_at=expires.isoformat().replace("+00:00", "Z"),
            dimensions=normalized,
            ticket_id=ticket_id,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FormalPlanningRolloutOverride":
        if payload.get("schema") != FORMAL_PLANNING_OVERRIDE_SCHEMA:
            raise FormalPlanningRolloutError("invalid override schema")
        result = cls(
            policy_id=payload.get("policy_id", ""),
            lane_id=payload.get("lane_id", ""),
            target_mode=payload.get("target_mode", ""),
            waived_reason_codes=tuple(payload.get("waived_reason_codes") or ()),
            actor=payload.get("actor", ""),
            reason=payload.get("reason", ""),
            issued_at=payload.get("issued_at", ""),
            expires_at=payload.get("expires_at", ""),
            dimensions=payload.get("dimensions") or {},
            ticket_id=payload.get("ticket_id", ""),
        )
        if payload.get("receipt_id") not in (None, result.receipt_id):
            raise FormalPlanningRolloutError("override receipt identity is inconsistent")
        return result


class FormalPlanningOverrideStore:
    """Append-only local store for durable scoped override receipts."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

    def persist(self, receipt: FormalPlanningRolloutOverride) -> Path:
        if not isinstance(receipt, FormalPlanningRolloutOverride):
            raise FormalPlanningRolloutError("only formal rollout overrides can be stored")
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"{receipt.receipt_id}.json"
        encoded = (_canonical_json(receipt.to_dict()) + "\n").encode()
        try:
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if self.load(receipt.receipt_id) != receipt:
                raise FormalPlanningRolloutError("stored override identity collision")
            return destination
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            try:
                destination.unlink()
            except OSError:
                pass
            raise
        return destination

    def load(self, receipt_id: str) -> FormalPlanningRolloutOverride:
        normalized = _text(receipt_id, "receipt_id")
        if "/" in normalized or "\\" in normalized:
            raise FormalPlanningRolloutError("unsafe override receipt identity")
        path = self.directory / f"{normalized}.json"
        if path.is_symlink():
            raise FormalPlanningRolloutError("override receipt cannot be a symlink")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FormalPlanningRolloutError("override receipt is unavailable") from exc
        receipt = FormalPlanningRolloutOverride.from_dict(payload)
        if receipt.receipt_id != normalized:
            raise FormalPlanningRolloutError("override filename identity mismatch")
        return receipt


@dataclass(frozen=True)
class FormalPlanningRolloutDecision(Mapping[str, Any]):
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        copied = json.loads(_canonical_json(dict(self.payload)))
        if copied.get("schema") != FORMAL_PLANNING_ROLLOUT_SCHEMA:
            raise FormalPlanningRolloutError("invalid rollout decision schema")
        lanes = copied.get("lane_decisions")
        if not isinstance(lanes, list):
            raise FormalPlanningRolloutError("lane_decisions must be a list")
        if copied.get("lane_count") != len(lanes):
            raise FormalPlanningRolloutError("lane decision count is inconsistent")
        expected = _digest(
            {
                key: value
                for key, value in copied.items()
                if key not in {"decision_id", "generated_at"}
            }
        )
        if copied.get("decision_id") != expected:
            raise FormalPlanningRolloutError("rollout decision identity is inconsistent")
        object.__setattr__(self, "payload", copied)

    @property
    def promotion_allowed(self) -> bool:
        return bool(self.payload["promotion_allowed"])

    @property
    def decision_id(self) -> str:
        return str(self.payload["decision_id"])

    def to_dict(self) -> dict[str, Any]:
        return json.loads(_canonical_json(self.payload))

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.payload)

    def __len__(self) -> int:
        return len(self.payload)


def _threshold_failures(
    lane: Mapping[str, Any],
    thresholds: RolloutThresholds,
) -> list[str]:
    failures: set[str] = set()
    modes = lane.get("benchmark_modes") or {}
    if thresholds.require_cold_and_warm:
        for mode in BenchmarkMode:
            cohort = modes.get(mode.value)
            if not cohort:
                failures.add(f"{mode.value}_benchmark_missing")
            elif cohort.get("sample_count", 0) < thresholds.min_samples_per_mode:
                failures.add(f"{mode.value}_sample_count_below_threshold")
    selected = [item for item in modes.values() if isinstance(item, Mapping)]
    if not selected:
        return sorted(failures | {"benchmark_measurement_missing"})
    if min(item.get("context_token_reduction", 0) for item in selected) < (
        thresholds.min_context_token_reduction
    ):
        failures.add("context_token_reduction_below_threshold")
    if sum(item.get("defects_found_before_dispatch", 0) for item in selected) < (
        thresholds.min_pre_dispatch_defects
    ):
        failures.add("pre_dispatch_defects_below_threshold")
    if min(item.get("proof_support_rate", 0) for item in selected) < (
        thresholds.min_proof_support_rate
    ):
        failures.add("proof_support_below_threshold")
    counterexample_cohorts = [
        item for item in selected if item.get("counterexamples", 0) > 0
    ]
    if (
        counterexample_cohorts
        and min(item.get("counterexample_quality", 0) for item in counterexample_cohorts)
        < thresholds.min_counterexample_quality
    ):
        failures.add("counterexample_quality_below_threshold")
    warm = modes.get(BenchmarkMode.WARM.value)
    if (
        warm
        and warm.get("cache_reuse_rate", 0) < thresholds.min_warm_cache_reuse_rate
    ):
        failures.add("warm_cache_reuse_below_threshold")
    if max(item.get("queue_latency_ms_max", 0) for item in selected) > (
        thresholds.max_queue_latency_ms
    ):
        failures.add("queue_latency_above_threshold")
    if max(item.get("cpu_saturation_percent_max", 0) for item in selected) > (
        thresholds.max_cpu_saturation_percent
    ):
        failures.add("cpu_saturation_above_threshold")
    if max(item.get("memory_peak_bytes", 0) for item in selected) > (
        thresholds.max_memory_peak_bytes
    ):
        failures.add("memory_above_threshold")
    if min(item.get("throughput_ratio", 0) for item in selected) < (
        thresholds.min_throughput_ratio
    ):
        failures.add("accepted_task_throughput_below_threshold")
    dimensions = FormalPlanningMetricDimensions.from_mapping(lane["dimensions"])
    if dimensions.authoritative_assurance.rank < thresholds.minimum_assurance.rank:
        failures.add("authoritative_assurance_below_threshold")
    return sorted(failures)


class FormalPlanningRolloutGate:
    """Evaluate every executable matrix lane against one target mode."""

    def __init__(self, policy: FormalPlanningRolloutPolicy | None = None) -> None:
        self.policy = policy or FormalPlanningRolloutPolicy()

    def evaluate(
        self,
        report: FormalPlanningBenchmarkReport | Mapping[str, Any],
        *,
        target_mode: RolloutMode | str,
        overrides: Iterable[
            FormalPlanningRolloutOverride | Mapping[str, Any]
        ] = (),
        now: datetime | str | None = None,
    ) -> FormalPlanningRolloutDecision:
        if not isinstance(report, FormalPlanningBenchmarkReport):
            report = FormalPlanningBenchmarkReport(report)
        try:
            mode = RolloutMode(target_mode)
        except ValueError as exc:
            raise FormalPlanningRolloutError("invalid rollout target mode") from exc
        if mode is RolloutMode.DISABLED:
            raise FormalPlanningRolloutError("disabled is not a promotion target")
        thresholds = self.policy.for_mode(mode)
        normalized_overrides = [
            item
            if isinstance(item, FormalPlanningRolloutOverride)
            else FormalPlanningRolloutOverride.from_dict(item)
            for item in overrides
        ]

        lane_decisions: list[dict[str, Any]] = []
        for lane in report.matrix:
            lane_id = str(lane["lane_id"])
            dimensions = FormalPlanningMetricDimensions.from_mapping(
                lane["dimensions"]
            )
            hard_advisory: list[str] = []
            if not lane.get("available", False):
                hard_advisory.append("lane_unavailable")
            if lane.get("low_value", False):
                hard_advisory.append("lane_low_value")
            failures = _threshold_failures(lane, thresholds)
            applicable = [
                item
                for item in normalized_overrides
                if item.is_valid(
                    policy_id=self.policy.policy_id,
                    lane_id=lane_id,
                    target_mode=mode,
                    now=now,
                )
            ]
            waived = sorted(
                {
                    failure
                    for override in applicable
                    for failure in override.waived_reason_codes
                    if failure in failures and failure not in NON_WAIVABLE_FAILURES
                }
            )
            remaining = [failure for failure in failures if failure not in waived]
            if hard_advisory:
                disposition = RolloutDisposition.ADVISORY
                # An override is never allowed to manufacture an executable lane.
                remaining = sorted(set(remaining) | set(hard_advisory))
            elif remaining:
                disposition = RolloutDisposition.HOLD
            else:
                disposition = RolloutDisposition.PROMOTE
            lane_decisions.append(
                {
                    "lane_id": lane_id,
                    "dimensions": dimensions.to_dict(),
                    "target_mode": mode.value,
                    "disposition": disposition.value,
                    "eligible_for_blocking": disposition is not RolloutDisposition.ADVISORY,
                    "failure_reason_codes": failures,
                    "waived_reason_codes": waived,
                    "remaining_reason_codes": remaining,
                    "override_receipt_ids": sorted(
                        item.receipt_id for item in applicable
                    ),
                    "degraded_reason_codes": sorted(
                        set(lane.get("degraded_reason_codes") or ()) | set(remaining)
                    ),
                }
            )

        eligible = [
            item for item in lane_decisions if item["eligible_for_blocking"]
        ]
        allowed = bool(eligible) and all(
            item["disposition"] == RolloutDisposition.PROMOTE.value
            for item in eligible
        )
        material: dict[str, Any] = {
            "schema": FORMAL_PLANNING_ROLLOUT_SCHEMA,
            "schema_version": FORMAL_PLANNING_ROLLOUT_VERSION,
            "generated_at": _timestamp(now)[0],
            "policy_id": self.policy.policy_id,
            "benchmark_report_id": report.report_id,
            "target_mode": mode.value,
            "thresholds": thresholds.to_dict(),
            "all_thresholds": self.policy.to_dict()["thresholds"],
            "promotion_allowed": allowed,
            "lane_count": len(lane_decisions),
            "executable_lane_count": len(eligible),
            "advisory_lane_count": len(lane_decisions) - len(eligible),
            "lane_decisions": lane_decisions,
            "degraded_reason_codes": sorted(
                {
                    reason
                    for item in lane_decisions
                    for reason in item["degraded_reason_codes"]
                }
            ),
        }
        material["decision_id"] = _digest(
            {
                key: value
                for key, value in material.items()
                if key not in {"decision_id", "generated_at"}
            }
        )
        return FormalPlanningRolloutDecision(material)


def _project_items(
    values: Iterable[Any],
    *,
    id_fields: Sequence[str],
    allowed_fields: Sequence[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for ordinal, value in enumerate(values):
        if ordinal >= MAX_OPERATOR_ITEMS:
            raise FormalPlanningRolloutError("operator projection item bound exceeded")
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        if not isinstance(value, Mapping):
            value = {"id": str(value)}
        projected: dict[str, Any] = {}
        for name in (*id_fields, *allowed_fields):
            item = value.get(name)
            if item is None:
                continue
            if isinstance(item, Enum):
                item = item.value
            if isinstance(item, (str, int, float, bool)):
                projected[name] = item
            elif isinstance(item, (list, tuple, set)):
                projected[name] = [
                    member.value if isinstance(member, Enum) else str(member)
                    for member in list(item)[:64]
                ]
        if not any(field in projected for field in id_fields):
            projected["id"] = f"item-{ordinal + 1}"
        result.append(projected)
    return result


def build_formal_planning_operator_projection(
    report: FormalPlanningBenchmarkReport | Mapping[str, Any],
    decision: FormalPlanningRolloutDecision | Mapping[str, Any],
    *,
    active_formal_plans: Iterable[Any] = (),
    unmet_obligations: Iterable[Any] = (),
    trace_violations: Iterable[Any] = (),
    generated_at: datetime | str | None = None,
) -> dict[str, Any]:
    """Build the bounded, raw-context-free operator status projection."""

    if not isinstance(report, FormalPlanningBenchmarkReport):
        report = FormalPlanningBenchmarkReport(report)
    if not isinstance(decision, FormalPlanningRolloutDecision):
        decision = FormalPlanningRolloutDecision(decision)
    if decision["benchmark_report_id"] != report.report_id:
        raise FormalPlanningRolloutError(
            "operator projection report and decision do not match"
        )
    plans = _project_items(
        active_formal_plans,
        id_fields=("plan_id", "workflow_id", "id"),
        allowed_fields=("status", "task_count", "active_node_count", "assurance"),
    )
    obligations = _project_items(
        unmet_obligations,
        id_fields=("obligation_id", "requirement_id", "id"),
        allowed_fields=(
            "plan_id",
            "property_class",
            "required_assurance",
            "actual_assurance",
            "status",
            "reason_code",
        ),
    )
    violations = _project_items(
        trace_violations,
        id_fields=("violation_id", "trace_id", "id"),
        allowed_fields=(
            "plan_id",
            "property_class",
            "verdict",
            "reason_code",
            "event_count",
            "finite_bound",
        ),
    )
    projection: dict[str, Any] = {
        "schema": FORMAL_PLANNING_OPERATOR_SCHEMA,
        "schema_version": FORMAL_PLANNING_ROLLOUT_VERSION,
        "generated_at": _timestamp(generated_at)[0],
        "policy_id": decision["policy_id"],
        "benchmark_report_id": report.report_id,
        "rollout_decision_id": decision.decision_id,
        "target_mode": decision["target_mode"],
        "promotion_allowed": decision.promotion_allowed,
        "executable_matrix": [
            {
                "lane_id": item["lane_id"],
                "dimensions": item["dimensions"],
                "disposition": item["disposition"],
                "eligible_for_blocking": item["eligible_for_blocking"],
                "degraded_reason_codes": item["degraded_reason_codes"],
            }
            for item in decision["lane_decisions"]
        ],
        "degraded_reasons": decision["degraded_reason_codes"],
        "active_formal_plans": plans,
        "unmet_obligations": obligations,
        "trace_violations": violations,
        "rollout_decisions": [
            {
                "lane_id": item["lane_id"],
                "target_mode": item["target_mode"],
                "disposition": item["disposition"],
                "remaining_reason_codes": item["remaining_reason_codes"],
                "override_receipt_ids": item["override_receipt_ids"],
            }
            for item in decision["lane_decisions"]
        ],
        "counts": {
            "executable_lanes": decision["executable_lane_count"],
            "advisory_lanes": decision["advisory_lane_count"],
            "active_formal_plans": len(plans),
            "unmet_obligations": len(obligations),
            "trace_violations": len(violations),
        },
        "contains_raw_context": False,
        "contains_proof_bodies": False,
        "contains_trace_payloads": False,
    }
    projection["projection_id"] = _digest(
        {key: value for key, value in projection.items() if key != "generated_at"}
    )
    encoded = _canonical_json(projection).encode()
    if len(encoded) > MAX_OPERATOR_BYTES:
        raise FormalPlanningRolloutError("operator projection exceeds its byte bound")
    # Reuse the metric projection's strict private-field validator by attempting
    # a harmless one-sample report is unnecessary; explicit allow-lists above
    # are the stronger boundary here.
    return projection


def gate_formal_planning_rollout(
    report: FormalPlanningBenchmarkReport | Mapping[str, Any],
    *,
    target_mode: RolloutMode | str,
    policy: FormalPlanningRolloutPolicy | None = None,
    overrides: Iterable[FormalPlanningRolloutOverride | Mapping[str, Any]] = (),
    now: datetime | str | None = None,
) -> FormalPlanningRolloutDecision:
    return FormalPlanningRolloutGate(policy).evaluate(
        report, target_mode=target_mode, overrides=overrides, now=now
    )


# Compatibility-friendly aliases.
FormalPlanningThresholds = RolloutThresholds
FormalPlanningRolloutThresholds = RolloutThresholds
FormalPlanningOverrideReceipt = FormalPlanningRolloutOverride
FormalPlanningRolloutOverrideStore = FormalPlanningOverrideStore
RolloutOverride = FormalPlanningRolloutOverride
RolloutOverrideStore = FormalPlanningOverrideStore
FormalPlanningOperatorProjection = dict
evaluate_formal_planning_rollout = gate_formal_planning_rollout
project_formal_planning_rollout = build_formal_planning_operator_projection


__all__ = [
    "DEFAULT_ROLLOUT_THRESHOLDS",
    "FORMAL_PLANNING_OPERATOR_SCHEMA",
    "FORMAL_PLANNING_OVERRIDE_SCHEMA",
    "FORMAL_PLANNING_ROLLOUT_SCHEMA",
    "FormalPlanningOperatorProjection",
    "FormalPlanningOverrideStore",
    "FormalPlanningOverrideReceipt",
    "FormalPlanningRolloutDecision",
    "FormalPlanningRolloutError",
    "FormalPlanningRolloutGate",
    "FormalPlanningRolloutOverride",
    "FormalPlanningRolloutOverrideStore",
    "FormalPlanningRolloutPolicy",
    "FormalPlanningThresholds",
    "FormalPlanningRolloutThresholds",
    "RolloutDisposition",
    "RolloutOverride",
    "RolloutOverrideStore",
    "RolloutThresholds",
    "build_formal_planning_operator_projection",
    "evaluate_formal_planning_rollout",
    "gate_formal_planning_rollout",
    "project_formal_planning_rollout",
]
