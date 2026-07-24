"""Bounded benchmark metrics for proof-carrying formal planning.

The lower-level :mod:`proof_metrics` module measures individual proof work.
This module measures the planning outcome that an operator actually cares
about: model context avoided, defects rejected before model dispatch, useful
proof/counterexample evidence, cache reuse, and host/task throughput.

Every sample carries the complete rollout matrix key.  Aggregation never
silently combines different property classes, translators, provers, kernels,
bounds, rollout modes, risks, or authoritative assurance levels.  Samples and
reports are deliberately public projections: they contain counts and resource
measurements, not prompts, source context, proof bodies, or counterexamples.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Final

from .formal_verification_contracts import AssuranceLevel
from .formal_verification_policy import RiskLevel, RolloutMode


FORMAL_PLANNING_METRICS_VERSION: Final = 1
FORMAL_PLANNING_BENCHMARK_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-planning-benchmark@1"
)
FORMAL_PLANNING_SAMPLE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-planning-benchmark-sample@1"
)
FORMAL_PLANNING_METRIC_DIMENSIONS: Final = (
    "property_class",
    "translator_profile",
    "prover",
    "kernel",
    "finite_bound",
    "rollout_mode",
    "task_risk",
    "authoritative_assurance",
)
MAX_FORMAL_PLANNING_SAMPLES: Final = 2_048
MAX_PUBLIC_REPORT_BYTES: Final = 4 * 1024 * 1024
_PRIVATE_MARKERS: Final = (
    "prompt",
    "raw_context",
    "context_dump",
    "proof_body",
    "proof_term",
    "counterexample_body",
    "source_text",
    "transcript",
    "witness",
    "secret",
    "credential",
    "token_value",
)


class FormalPlanningMetricsError(ValueError):
    """Raised when a benchmark record is incomplete, unsafe, or inconsistent."""


class BenchmarkMode(str, Enum):
    COLD = "cold"
    WARM = "warm"


def _text(value: Any, name: str, *, default: str = "") -> str:
    if isinstance(value, Enum):
        value = value.value
    result = str(value if value is not None else default).strip()
    if not result:
        raise FormalPlanningMetricsError(f"{name} must be non-empty")
    if len(result.encode("utf-8")) > 512:
        raise FormalPlanningMetricsError(f"{name} exceeds its byte bound")
    return result


def _integer(value: Any, name: str, *, default: int = 0) -> int:
    if value is None:
        value = default
    if isinstance(value, bool):
        raise FormalPlanningMetricsError(f"{name} must be a non-negative integer")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FormalPlanningMetricsError(
            f"{name} must be a non-negative integer"
        ) from exc
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise FormalPlanningMetricsError(f"{name} must be a non-negative integer")
    return int(number)


def _number(value: Any, name: str, *, default: float = 0.0) -> float:
    if value is None:
        value = default
    if isinstance(value, bool):
        raise FormalPlanningMetricsError(f"{name} must be a non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FormalPlanningMetricsError(
            f"{name} must be a non-negative number"
        ) from exc
    if not math.isfinite(number) or number < 0:
        raise FormalPlanningMetricsError(f"{name} must be a non-negative number")
    return number


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(max(0.0, numerator) / denominator, 6)


def _timestamp(value: datetime | str | None = None) -> str:
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise FormalPlanningMetricsError("generated_at is not an ISO timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _identity(value: Mapping[str, Any], *excluded: str) -> str:
    material = {key: item for key, item in value.items() if key not in excluded}
    return "sha256:" + hashlib.sha256(
        _canonical_json(material).encode("utf-8")
    ).hexdigest()


def _assert_public(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            negative_declaration = (
                normalized
                in {
                    "contains_raw_context",
                    "contains_proof_bodies",
                    "contains_counterexample_bodies",
                }
                and item is False
            )
            if (
                any(marker in normalized for marker in _PRIVATE_MARKERS)
                and not negative_declaration
            ):
                raise FormalPlanningMetricsError(
                    f"private benchmark field is not allowed at {path}.{key}"
                )
            _assert_public(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_public(item, f"{path}[{index}]")


@dataclass(frozen=True)
class FormalPlanningMetricDimensions:
    """The irreducible key of one formal-planning benchmark lane."""

    property_class: str
    translator_profile: str
    prover: str
    kernel: str
    finite_bound: int | None
    rollout_mode: RolloutMode | str
    task_risk: RiskLevel | str
    authoritative_assurance: AssuranceLevel | str

    def __post_init__(self) -> None:
        for name in ("property_class", "translator_profile", "prover", "kernel"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.finite_bound is not None:
            object.__setattr__(
                self, "finite_bound", _integer(self.finite_bound, "finite_bound")
            )
        try:
            object.__setattr__(self, "rollout_mode", RolloutMode(self.rollout_mode))
            object.__setattr__(self, "task_risk", RiskLevel(self.task_risk))
            object.__setattr__(
                self,
                "authoritative_assurance",
                AssuranceLevel(self.authoritative_assurance),
            )
        except ValueError as exc:
            raise FormalPlanningMetricsError(
                "invalid rollout mode, task risk, or assurance dimension"
            ) from exc

    @property
    def lane_id(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "property_class": self.property_class,
            "translator_profile": self.translator_profile,
            "prover": self.prover,
            "kernel": self.kernel,
            "finite_bound": self.finite_bound,
            "rollout_mode": self.rollout_mode.value,
            "task_risk": self.task_risk.value,
            "authoritative_assurance": self.authoritative_assurance.value,
        }

    @classmethod
    def from_mapping(
        cls, value: "FormalPlanningMetricDimensions | Mapping[str, Any]"
    ) -> "FormalPlanningMetricDimensions":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise FormalPlanningMetricsError("dimensions must be a mapping")
        return cls(
            property_class=value.get("property_class", value.get("property_kind", "unknown")),
            translator_profile=value.get(
                "translator_profile", value.get("translator", "unknown")
            ),
            prover=value.get("prover", value.get("prover_id", "unknown")),
            kernel=value.get("kernel", value.get("kernel_id", "none")),
            finite_bound=value.get("finite_bound", value.get("bound")),
            rollout_mode=value.get("rollout_mode", RolloutMode.SHADOW),
            task_risk=value.get("task_risk", value.get("risk", RiskLevel.LOW)),
            authoritative_assurance=value.get(
                "authoritative_assurance",
                value.get("assurance", AssuranceLevel.UNVERIFIED),
            ),
        )


@dataclass(frozen=True)
class FormalPlanningBenchmarkSample:
    """One cold or warm measurement for exactly one matrix lane."""

    sample_id: str
    benchmark_mode: BenchmarkMode | str
    dimensions: FormalPlanningMetricDimensions | Mapping[str, Any]
    baseline_context_tokens: int
    formal_context_tokens: int
    plans_evaluated: int
    defects_found_before_dispatch: int
    obligations: int
    proof_supported_obligations: int
    counterexamples: int
    actionable_counterexamples: int
    minimized_counterexamples: int
    cache_lookups: int
    cache_hits: int
    queue_latency_ms: float
    cpu_saturation_percent: float
    memory_peak_bytes: int
    accepted_tasks: int
    elapsed_seconds: float
    baseline_accepted_tasks_per_second: float
    available: bool = True
    low_value: bool = False
    advisory_reason_codes: tuple[str, ...] = ()
    counterexample_quality_score: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", _text(self.sample_id, "sample_id"))
        try:
            object.__setattr__(
                self, "benchmark_mode", BenchmarkMode(self.benchmark_mode)
            )
        except ValueError as exc:
            raise FormalPlanningMetricsError(
                "benchmark_mode must be cold or warm"
            ) from exc
        object.__setattr__(
            self,
            "dimensions",
            FormalPlanningMetricDimensions.from_mapping(self.dimensions),
        )
        integer_fields = (
            "baseline_context_tokens",
            "formal_context_tokens",
            "plans_evaluated",
            "defects_found_before_dispatch",
            "obligations",
            "proof_supported_obligations",
            "counterexamples",
            "actionable_counterexamples",
            "minimized_counterexamples",
            "cache_lookups",
            "cache_hits",
            "memory_peak_bytes",
            "accepted_tasks",
        )
        for name in integer_fields:
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        for name in (
            "queue_latency_ms",
            "cpu_saturation_percent",
            "elapsed_seconds",
            "baseline_accepted_tasks_per_second",
        ):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.formal_context_tokens > self.baseline_context_tokens:
            # A regression is valid benchmark data, but reduction is clamped at
            # a negative value below rather than rejected.
            pass
        if self.defects_found_before_dispatch > self.plans_evaluated:
            raise FormalPlanningMetricsError("defects cannot exceed evaluated plans")
        if self.proof_supported_obligations > self.obligations:
            raise FormalPlanningMetricsError(
                "proof-supported obligations cannot exceed obligations"
            )
        if self.actionable_counterexamples > self.counterexamples:
            raise FormalPlanningMetricsError(
                "actionable counterexamples cannot exceed counterexamples"
            )
        if self.minimized_counterexamples > self.counterexamples:
            raise FormalPlanningMetricsError(
                "minimized counterexamples cannot exceed counterexamples"
            )
        if self.cache_hits > self.cache_lookups:
            raise FormalPlanningMetricsError("cache hits cannot exceed lookups")
        if self.cpu_saturation_percent > 100:
            raise FormalPlanningMetricsError("cpu_saturation_percent cannot exceed 100")
        if not isinstance(self.available, bool) or not isinstance(self.low_value, bool):
            raise FormalPlanningMetricsError("availability flags must be booleans")
        reasons = tuple(
            sorted({_text(item, "advisory_reason_code") for item in self.advisory_reason_codes})
        )
        object.__setattr__(self, "advisory_reason_codes", reasons)
        quality = self.counterexample_quality_score
        if quality is not None:
            quality = _number(quality, "counterexample_quality_score")
            if quality > 1:
                raise FormalPlanningMetricsError(
                    "counterexample_quality_score cannot exceed one"
                )
            object.__setattr__(self, "counterexample_quality_score", quality)

    @property
    def context_token_reduction(self) -> float:
        if not self.baseline_context_tokens:
            return 0.0
        return round(
            (self.baseline_context_tokens - self.formal_context_tokens)
            / self.baseline_context_tokens,
            6,
        )

    @property
    def pre_dispatch_defect_rate(self) -> float:
        return _ratio(self.defects_found_before_dispatch, self.plans_evaluated)

    @property
    def proof_support_rate(self) -> float:
        return _ratio(self.proof_supported_obligations, self.obligations)

    @property
    def counterexample_quality(self) -> float:
        if self.counterexample_quality_score is not None:
            return self.counterexample_quality_score
        return _ratio(
            self.actionable_counterexamples + self.minimized_counterexamples,
            2 * self.counterexamples,
        )

    @property
    def cache_reuse_rate(self) -> float:
        return _ratio(self.cache_hits, self.cache_lookups)

    @property
    def accepted_tasks_per_second(self) -> float:
        return _ratio(self.accepted_tasks, self.elapsed_seconds)

    @property
    def throughput_ratio(self) -> float:
        return _ratio(
            self.accepted_tasks_per_second,
            self.baseline_accepted_tasks_per_second,
        )

    @property
    def advisory(self) -> bool:
        return not self.available or self.low_value

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": FORMAL_PLANNING_SAMPLE_SCHEMA,
            "sample_id": self.sample_id,
            "benchmark_mode": self.benchmark_mode.value,
            "dimensions": self.dimensions.to_dict(),
            "lane_id": self.dimensions.lane_id,
            "baseline_context_tokens": self.baseline_context_tokens,
            "formal_context_tokens": self.formal_context_tokens,
            "context_token_reduction": self.context_token_reduction,
            "plans_evaluated": self.plans_evaluated,
            "defects_found_before_dispatch": self.defects_found_before_dispatch,
            "pre_dispatch_defect_rate": self.pre_dispatch_defect_rate,
            "obligations": self.obligations,
            "proof_supported_obligations": self.proof_supported_obligations,
            "proof_support_rate": self.proof_support_rate,
            "counterexamples": self.counterexamples,
            "actionable_counterexamples": self.actionable_counterexamples,
            "minimized_counterexamples": self.minimized_counterexamples,
            "counterexample_quality": self.counterexample_quality,
            "cache_lookups": self.cache_lookups,
            "cache_hits": self.cache_hits,
            "cache_reuse_rate": self.cache_reuse_rate,
            "queue_latency_ms": round(self.queue_latency_ms, 6),
            "cpu_saturation_percent": round(self.cpu_saturation_percent, 6),
            "memory_peak_bytes": self.memory_peak_bytes,
            "accepted_tasks": self.accepted_tasks,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "accepted_tasks_per_second": self.accepted_tasks_per_second,
            "baseline_accepted_tasks_per_second": round(
                self.baseline_accepted_tasks_per_second, 6
            ),
            "throughput_ratio": self.throughput_ratio,
            "available": self.available,
            "low_value": self.low_value,
            "advisory": self.advisory,
            "advisory_reason_codes": list(self.advisory_reason_codes),
        }
        _assert_public(payload)
        return payload

    @classmethod
    def from_mapping(
        cls, value: "FormalPlanningBenchmarkSample | Mapping[str, Any]"
    ) -> "FormalPlanningBenchmarkSample":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise FormalPlanningMetricsError("benchmark sample must be a mapping")

        def first(*names: str, default: Any = 0) -> Any:
            for name in names:
                if name in value:
                    return value[name]
            return default

        dimensions = value.get("dimensions")
        if not isinstance(dimensions, Mapping):
            dimensions = {
                name: value.get(name)
                for name in FORMAL_PLANNING_METRIC_DIMENSIONS
                if name in value
            }
        return cls(
            sample_id=first("sample_id", "id", default="sample"),
            benchmark_mode=first("benchmark_mode", "mode", default=BenchmarkMode.COLD),
            dimensions=dimensions,
            baseline_context_tokens=first(
                "baseline_context_tokens", "raw_context_tokens"
            ),
            formal_context_tokens=first(
                "formal_context_tokens", "planning_context_tokens", "capsule_context_tokens"
            ),
            plans_evaluated=first("plans_evaluated", "attempted_plans"),
            defects_found_before_dispatch=first(
                "defects_found_before_dispatch", "pre_dispatch_defects"
            ),
            obligations=first("obligations", "obligation_count"),
            proof_supported_obligations=first(
                "proof_supported_obligations", "supported_obligations"
            ),
            counterexamples=first("counterexamples", "counterexample_count"),
            actionable_counterexamples=first(
                "actionable_counterexamples", "useful_counterexamples"
            ),
            minimized_counterexamples=first("minimized_counterexamples"),
            counterexample_quality_score=value.get("counterexample_quality_score"),
            cache_lookups=first("cache_lookups"),
            cache_hits=first("cache_hits", "cache_reuses"),
            queue_latency_ms=first("queue_latency_ms"),
            cpu_saturation_percent=first("cpu_saturation_percent", "cpu_percent"),
            memory_peak_bytes=first("memory_peak_bytes"),
            accepted_tasks=first("accepted_tasks"),
            elapsed_seconds=first("elapsed_seconds", "wall_time_seconds"),
            baseline_accepted_tasks_per_second=first(
                "baseline_accepted_tasks_per_second", "baseline_throughput"
            ),
            available=first("available", default=True),
            low_value=first("low_value", default=False),
            advisory_reason_codes=tuple(value.get("advisory_reason_codes") or ()),
        )


def _cohort(samples: list[FormalPlanningBenchmarkSample]) -> dict[str, Any]:
    baseline_tokens = sum(item.baseline_context_tokens for item in samples)
    formal_tokens = sum(item.formal_context_tokens for item in samples)
    plans = sum(item.plans_evaluated for item in samples)
    defects = sum(item.defects_found_before_dispatch for item in samples)
    obligations = sum(item.obligations for item in samples)
    supported = sum(item.proof_supported_obligations for item in samples)
    counterexamples = sum(item.counterexamples for item in samples)
    actionable = sum(item.actionable_counterexamples for item in samples)
    minimized = sum(item.minimized_counterexamples for item in samples)
    lookups = sum(item.cache_lookups for item in samples)
    hits = sum(item.cache_hits for item in samples)
    accepted = sum(item.accepted_tasks for item in samples)
    elapsed = sum(item.elapsed_seconds for item in samples)
    baseline_throughput = (
        sum(item.baseline_accepted_tasks_per_second for item in samples) / len(samples)
    )
    actual_throughput = _ratio(accepted, elapsed)
    return {
        "sample_count": len(samples),
        "sample_ids": sorted(item.sample_id for item in samples),
        "baseline_context_tokens": baseline_tokens,
        "formal_context_tokens": formal_tokens,
        "context_token_reduction": (
            round((baseline_tokens - formal_tokens) / baseline_tokens, 6)
            if baseline_tokens
            else 0.0
        ),
        "plans_evaluated": plans,
        "defects_found_before_dispatch": defects,
        "pre_dispatch_defect_rate": _ratio(defects, plans),
        "obligations": obligations,
        "proof_supported_obligations": supported,
        "proof_support_rate": _ratio(supported, obligations),
        "counterexamples": counterexamples,
        "counterexample_quality": _ratio(
            actionable + minimized, 2 * counterexamples
        ),
        "cache_lookups": lookups,
        "cache_hits": hits,
        "cache_reuse_rate": _ratio(hits, lookups),
        "queue_latency_ms_max": round(max(item.queue_latency_ms for item in samples), 6),
        "cpu_saturation_percent_max": round(
            max(item.cpu_saturation_percent for item in samples), 6
        ),
        "memory_peak_bytes": max(item.memory_peak_bytes for item in samples),
        "accepted_tasks": accepted,
        "elapsed_seconds": round(elapsed, 6),
        "accepted_tasks_per_second": actual_throughput,
        "baseline_accepted_tasks_per_second": round(baseline_throughput, 6),
        "throughput_ratio": _ratio(actual_throughput, baseline_throughput),
        "available": all(item.available for item in samples),
        "low_value": all(item.low_value for item in samples),
        "advisory": all(item.advisory for item in samples),
        "advisory_reason_codes": sorted(
            {reason for item in samples for reason in item.advisory_reason_codes}
        ),
    }


@dataclass(frozen=True)
class FormalPlanningBenchmarkReport(Mapping[str, Any]):
    """Immutable, content-addressed public benchmark projection."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        copied = json.loads(_canonical_json(dict(self.payload)))
        if copied.get("schema") != FORMAL_PLANNING_BENCHMARK_SCHEMA:
            raise FormalPlanningMetricsError("invalid formal-planning report schema")
        if copied.get("schema_version") != FORMAL_PLANNING_METRICS_VERSION:
            raise FormalPlanningMetricsError("unsupported report schema version")
        samples = copied.get("samples")
        matrix = copied.get("matrix")
        if not isinstance(samples, list) or not 0 < len(samples) <= MAX_FORMAL_PLANNING_SAMPLES:
            raise FormalPlanningMetricsError("report samples must be non-empty and bounded")
        if not isinstance(matrix, list):
            raise FormalPlanningMetricsError("report matrix must be a list")
        if copied.get("sample_count") != len(samples):
            raise FormalPlanningMetricsError("report sample count is inconsistent")
        if copied.get("contains_raw_context") is not False:
            raise FormalPlanningMetricsError("reports cannot contain raw context")
        if copied.get("contains_proof_bodies") is not False:
            raise FormalPlanningMetricsError("reports cannot contain proof bodies")
        _assert_public(copied)
        claimed = copied.get("report_id")
        expected = _identity(copied, "report_id", "generated_at")
        if claimed != expected:
            raise FormalPlanningMetricsError("report identity is inconsistent")
        if len(_canonical_json(copied).encode("utf-8")) > MAX_PUBLIC_REPORT_BYTES:
            raise FormalPlanningMetricsError("report exceeds its public byte bound")
        object.__setattr__(self, "payload", copied)

    @property
    def report_id(self) -> str:
        return str(self.payload["report_id"])

    @property
    def samples(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.payload["samples"])

    @property
    def matrix(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.payload["matrix"])

    def to_dict(self) -> dict[str, Any]:
        return json.loads(_canonical_json(self.payload))

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.payload)

    def __len__(self) -> int:
        return len(self.payload)


def build_formal_planning_benchmark_report(
    samples: Iterable[FormalPlanningBenchmarkSample | Mapping[str, Any]],
    *,
    generated_at: datetime | str | None = None,
) -> FormalPlanningBenchmarkReport:
    """Normalize samples and retain cold/warm results per complete lane key."""

    normalized: list[FormalPlanningBenchmarkSample] = []
    for ordinal, sample in enumerate(samples, start=1):
        if ordinal > MAX_FORMAL_PLANNING_SAMPLES:
            raise FormalPlanningMetricsError("sample count exceeds its bound")
        normalized.append(FormalPlanningBenchmarkSample.from_mapping(sample))
    if not normalized:
        raise FormalPlanningMetricsError("at least one benchmark sample is required")
    ids = [item.sample_id for item in normalized]
    if len(ids) != len(set(ids)):
        raise FormalPlanningMetricsError("sample_id values must be unique")

    grouped: dict[str, list[FormalPlanningBenchmarkSample]] = {}
    dimensions_by_lane: dict[str, FormalPlanningMetricDimensions] = {}
    for sample in normalized:
        lane_id = sample.dimensions.lane_id
        grouped.setdefault(lane_id, []).append(sample)
        dimensions_by_lane[lane_id] = sample.dimensions

    matrix: list[dict[str, Any]] = []
    for lane_id in sorted(grouped):
        lane_samples = grouped[lane_id]
        modes: dict[str, Any] = {}
        for mode in BenchmarkMode:
            selected = [item for item in lane_samples if item.benchmark_mode is mode]
            if selected:
                modes[mode.value] = _cohort(selected)
        cold = modes.get(BenchmarkMode.COLD.value)
        warm = modes.get(BenchmarkMode.WARM.value)
        cold_to_warm: dict[str, Any] = {}
        if cold and warm:
            cold_to_warm = {
                "queue_latency_reduction": (
                    round(
                        (cold["queue_latency_ms_max"] - warm["queue_latency_ms_max"])
                        / cold["queue_latency_ms_max"],
                        6,
                    )
                    if cold["queue_latency_ms_max"]
                    else 0.0
                ),
                "cache_reuse_improvement": round(
                    warm["cache_reuse_rate"] - cold["cache_reuse_rate"], 6
                ),
                "throughput_ratio_improvement": round(
                    warm["throughput_ratio"] - cold["throughput_ratio"], 6
                ),
            }
        matrix.append(
            {
                "lane_id": lane_id,
                "dimensions": dimensions_by_lane[lane_id].to_dict(),
                "benchmark_modes": modes,
                "cold_to_warm": cold_to_warm,
                "available": all(item.available for item in lane_samples),
                "low_value": all(item.low_value for item in lane_samples),
                "advisory": all(item.advisory for item in lane_samples),
                "degraded_reason_codes": sorted(
                    {
                        reason
                        for item in lane_samples
                        for reason in item.advisory_reason_codes
                    }
                    | ({"lane_unavailable"} if not all(item.available for item in lane_samples) else set())
                    | ({"lane_low_value"} if all(item.low_value for item in lane_samples) else set())
                ),
            }
        )

    total_baseline = sum(item.baseline_context_tokens for item in normalized)
    total_formal = sum(item.formal_context_tokens for item in normalized)
    material: dict[str, Any] = {
        "schema": FORMAL_PLANNING_BENCHMARK_SCHEMA,
        "schema_version": FORMAL_PLANNING_METRICS_VERSION,
        "generated_at": _timestamp(generated_at),
        "sample_count": len(normalized),
        "lane_count": len(matrix),
        "benchmark_modes": sorted(
            {item.benchmark_mode.value for item in normalized}
        ),
        "metric_dimensions": list(FORMAL_PLANNING_METRIC_DIMENSIONS),
        "samples": [
            item.to_dict()
            for item in sorted(
                normalized,
                key=lambda item: (
                    item.dimensions.lane_id,
                    item.benchmark_mode.value,
                    item.sample_id,
                ),
            )
        ],
        "matrix": matrix,
        "summary": {
            "baseline_context_tokens": total_baseline,
            "formal_context_tokens": total_formal,
            "context_token_reduction": (
                round((total_baseline - total_formal) / total_baseline, 6)
                if total_baseline
                else 0.0
            ),
            "defects_found_before_dispatch": sum(
                item.defects_found_before_dispatch for item in normalized
            ),
            "accepted_tasks": sum(item.accepted_tasks for item in normalized),
            "unavailable_lane_count": sum(not item["available"] for item in matrix),
            "low_value_lane_count": sum(item["low_value"] for item in matrix),
        },
        "bounded": True,
        "contains_raw_context": False,
        "contains_proof_bodies": False,
        "contains_counterexample_bodies": False,
    }
    material["report_id"] = _identity(material, "generated_at")
    return FormalPlanningBenchmarkReport(material)


@dataclass
class FormalPlanningMetricsCollector:
    """Thread-safe bounded collector used by cold and warm benchmark runners."""

    max_samples: int = MAX_FORMAL_PLANNING_SAMPLES
    _samples: list[FormalPlanningBenchmarkSample] = field(
        default_factory=list, init=False, repr=False
    )
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.max_samples = _integer(self.max_samples, "max_samples")
        if not 0 < self.max_samples <= MAX_FORMAL_PLANNING_SAMPLES:
            raise FormalPlanningMetricsError("max_samples is outside its hard bound")

    def record(
        self, sample: FormalPlanningBenchmarkSample | Mapping[str, Any]
    ) -> FormalPlanningBenchmarkSample:
        normalized = FormalPlanningBenchmarkSample.from_mapping(sample)
        with self._lock:
            if len(self._samples) >= self.max_samples:
                raise FormalPlanningMetricsError("collector sample bound reached")
            if any(item.sample_id == normalized.sample_id for item in self._samples):
                raise FormalPlanningMetricsError("sample_id values must be unique")
            self._samples.append(normalized)
        return normalized

    def report(
        self, *, generated_at: datetime | str | None = None
    ) -> FormalPlanningBenchmarkReport:
        with self._lock:
            samples = tuple(self._samples)
        return build_formal_planning_benchmark_report(
            samples, generated_at=generated_at
        )

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()


# Compatibility-friendly names for callers that use the shorter vocabulary.
BenchmarkDimensions = FormalPlanningMetricDimensions
FormalPlanningDimensions = FormalPlanningMetricDimensions
FormalPlanningBenchmarkDimensions = FormalPlanningMetricDimensions
FormalPlanningBenchmarkMode = BenchmarkMode
BenchmarkSample = FormalPlanningBenchmarkSample
FormalPlanningMetrics = FormalPlanningMetricsCollector
build_formal_planning_benchmark = build_formal_planning_benchmark_report
collect_formal_planning_metrics = build_formal_planning_benchmark_report


__all__ = [
    "BenchmarkDimensions",
    "BenchmarkMode",
    "BenchmarkSample",
    "FORMAL_PLANNING_BENCHMARK_SCHEMA",
    "FORMAL_PLANNING_METRICS_VERSION",
    "FORMAL_PLANNING_METRIC_DIMENSIONS",
    "FORMAL_PLANNING_SAMPLE_SCHEMA",
    "FormalPlanningBenchmarkReport",
    "FormalPlanningBenchmarkSample",
    "FormalPlanningBenchmarkDimensions",
    "FormalPlanningBenchmarkMode",
    "FormalPlanningDimensions",
    "FormalPlanningMetricDimensions",
    "FormalPlanningMetrics",
    "FormalPlanningMetricsCollector",
    "FormalPlanningMetricsError",
    "MAX_FORMAL_PLANNING_SAMPLES",
    "build_formal_planning_benchmark",
    "build_formal_planning_benchmark_report",
    "collect_formal_planning_metrics",
]
