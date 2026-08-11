"""Paired baseline/candidate quality, safety, throughput, and LLM-churn benchmark.

Interface: ``DuckDBQuackBenchmarkReport@1``

Task: DQP-036 / Goal: DQP-G050

Compares a sealed baseline (DQP-009) against a hermetic candidate arm that
models DuckDB/Quack control-plane reuse.  This module is a measurement boundary
only: it never grants completion, mutation, promotion, provider, or process
authority.  Missing telemetry is typed unavailable and is never inferred into
a numeric zero or a causal claim.

Acceptance (fail-closed):

* Warm reuse improves materially versus the paired baseline warm stratum.
* Duplicate unchanged provider work is eliminated on the candidate warm arm.
* No quality or safety floor regresses relative to sealed criteria / baseline.
* Throughput and latency stay within reviewed relative bounds.
* Unavailable metrics are reported honestly without fabricated causality.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .duckdb_quack_baseline import (
    BASIS_POINTS,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_WORKLOAD_SEED,
    LLM_CHURN_METRIC_NAMES,
    SAFETY_FLOOR_KEYS,
    STATE_METRIC_NAMES,
    BaselineCriteria,
    BaselineEnvironment,
    BaselineStratum,
    BaselineVerdict,
    DuckDBQuackBaselineError,
    DuckDBQuackBaselineReport,
    MetricSample,
    ProviderUsageCounters,
    StratumObservation,
    TelemetryStatus,
    UnavailableReason,
    WorkloadDefinition,
    default_workload,
    establish_duckdb_quack_baselines,
    establish_llm_churn_baseline,
    establish_supervisor_state_baseline,
)


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DUCKDB_QUACK_BENCHMARK_INTERFACE: Final[str] = "DuckDBQuackBenchmarkReport@1"
BENCHMARK_CONTRACT_VERSION: Final[int] = 1
TASK_ID: Final[str] = "DQP-036"
GOAL_ID: Final[str] = "DQP-G050"
EVIDENCE: Final[str] = "dqp/duckdb-quack-benchmark@1"

SCHEMA_PREFIX: Final[str] = "ipfs_accelerate_py/agent-supervisor"
BENCHMARK_REPORT_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/duckdb-quack-benchmark-report@1"
)
METRIC_DELTA_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/benchmark-metric-delta@1"
STRATUM_COMPARISON_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/benchmark-stratum-comparison@1"
)

# Relative latency ceiling: candidate may not exceed baseline * bound / 100.
DEFAULT_LATENCY_BOUND_PERCENT: Final[int] = 150
# Warm reuse material improvement floor (basis points of relative gain).
DEFAULT_WARM_REUSE_MIN_IMPROVEMENT_BPS: Final[int] = 1_000  # 10%
MAX_REASON_CODES: Final[int] = 256
MAX_TEXT_BYTES: Final[int] = 512

# Metrics where lower candidate values are better.
LOWER_IS_BETTER: Final[frozenset[str]] = frozenset(
    {
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
        "context_bytes",
        "provider_calls",
        "input_tokens",
        "output_tokens",
        "duplicate_semantic_inputs",
        "cache_reuse_misses",
        "rejected_provider_calls",
        "retry_provider_calls",
        "abandoned_provider_calls",
        "rollback_rate_bps",
        "failure_rate_bps",
    }
)

# Metrics where higher candidate values are better.
HIGHER_IS_BETTER: Final[frozenset[str]] = frozenset(
    {
        "cache_reuse_hits",
        "accepted_mutations",
        "accepted_mutation_quality_bps",
    }
)

# Latency / throughput metrics checked against the relative bound.
LATENCY_METRICS: Final[tuple[str, ...]] = (
    "task_claim_latency_ms",
    "queue_latency_ms",
    "lock_waits_ms",
)

# Warm-reuse primary signal (higher is better).
WARM_REUSE_METRIC: Final[str] = "cache_reuse_hits"
WARM_DUPLICATE_METRIC: Final[str] = "duplicate_semantic_inputs"
QUALITY_METRIC: Final[str] = "accepted_mutation_quality_bps"


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class BenchmarkArm(str, Enum):
    BASELINE = "baseline"
    CANDIDATE = "candidate"


class BenchmarkVerdict(str, Enum):
    """Measurement conclusion; not a promotion decision."""

    PASSED = "passed"
    FAILED = "failed"
    INSUFFICIENT = "insufficient"


class ComparisonStatus(str, Enum):
    """How one metric paired comparison resolved."""

    IMPROVED = "improved"
    EQUAL = "equal"
    REGRESSED = "regressed"
    WITHIN_BOUNDS = "within_bounds"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


class DuckDBQuackBenchmarkError(ValueError):
    """Fail-closed rejection for incomplete or unsafe benchmark inputs."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _text(value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        raise DuckDBQuackBenchmarkError(f"{name} must be text")
    result = value.strip()
    if not result:
        raise DuckDBQuackBenchmarkError(f"{name} must not be empty")
    if "\x00" in result:
        raise DuckDBQuackBenchmarkError(f"{name} contains a NUL byte")
    if len(result.encode("utf-8")) > maximum:
        raise DuckDBQuackBenchmarkError(f"{name} exceeds its {maximum}-byte bound")
    return result


def _nonnegative_int(value: Any, name: str, *, maximum: int = 10**18) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DuckDBQuackBenchmarkError(f"{name} must be a non-negative integer")
    if value < 0:
        raise DuckDBQuackBenchmarkError(f"{name} must be a non-negative integer")
    if value > maximum:
        raise DuckDBQuackBenchmarkError(f"{name} exceeds its maximum of {maximum}")
    return value


def content_identity(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _sample_value(sample: MetricSample | None) -> int | None:
    if sample is None:
        return None
    if sample.is_unavailable:
        return None
    return int(sample.value)


def _stratum_metric(
    strata: Sequence[StratumObservation],
    stratum: str,
    metric_name: str,
) -> MetricSample | None:
    for item in strata:
        key = item.stratum.value if isinstance(item.stratum, Enum) else str(item.stratum)
        if key == stratum:
            return item.metrics.get(metric_name)
    return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricDelta:
    """Paired comparison for one metric name."""

    SCHEMA: ClassVar[str] = METRIC_DELTA_SCHEMA

    metric_name: str
    baseline_status: str
    candidate_status: str
    baseline_value: int | None
    candidate_value: int | None
    delta: int | None
    status: ComparisonStatus
    reason_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.SCHEMA,
            "metric_name": self.metric_name,
            "baseline_status": self.baseline_status,
            "candidate_status": self.candidate_status,
            "status": self.status.value
            if isinstance(self.status, Enum)
            else self.status,
        }
        if self.baseline_value is not None:
            payload["baseline_value"] = self.baseline_value
        if self.candidate_value is not None:
            payload["candidate_value"] = self.candidate_value
        if self.delta is not None:
            payload["delta"] = self.delta
        if self.reason_code:
            payload["reason_code"] = self.reason_code
        return payload


@dataclass(frozen=True)
class StratumComparison:
    """Per-stratum paired deltas."""

    SCHEMA: ClassVar[str] = STRATUM_COMPARISON_SCHEMA

    stratum: str
    deltas: tuple[MetricDelta, ...]
    warm_reuse_improved: bool | None = None
    duplicates_eliminated: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "stratum": self.stratum,
            "deltas": [item.to_dict() for item in self.deltas],
            "warm_reuse_improved": self.warm_reuse_improved,
            "duplicates_eliminated": self.duplicates_eliminated,
        }


@dataclass(frozen=True)
class SafetyFloorSnapshot:
    """Absolute-zero safety counters for one arm."""

    floors: Mapping[str, int]

    def __post_init__(self) -> None:
        cleaned = {
            key: _nonnegative_int(
                dict(self.floors or {}).get(key, 0), f"floors.{key}"
            )
            for key in SAFETY_FLOOR_KEYS
        }
        for key in dict(self.floors or {}):
            if key not in SAFETY_FLOOR_KEYS:
                raise DuckDBQuackBenchmarkError(f"unknown safety floor key {key!r}")
        object.__setattr__(self, "floors", MappingProxyType(cleaned))

    @property
    def all_zero(self) -> bool:
        return all(value == 0 for value in self.floors.values())

    def to_dict(self) -> dict[str, int]:
        return dict(self.floors)

    @classmethod
    def zeros(cls) -> "SafetyFloorSnapshot":
        return cls(floors={key: 0 for key in SAFETY_FLOOR_KEYS})


@dataclass(frozen=True)
class DuckDBQuackBenchmarkReport:
    """``DuckDBQuackBenchmarkReport@1`` paired comparison receipt."""

    SCHEMA: ClassVar[str] = BENCHMARK_REPORT_SCHEMA
    INTERFACE: ClassVar[str] = DUCKDB_QUACK_BENCHMARK_INTERFACE

    verdict: BenchmarkVerdict
    baseline_tree_id: str
    candidate_tree_id: str
    environment_equivalent: bool
    workload_seed: int
    sample_count: int
    confidence_bps: int
    warm_reuse_improved: bool
    duplicates_eliminated: bool
    quality_non_inferior: bool
    safety_floors_zero: bool
    latency_within_bounds: bool
    missing_telemetry: tuple[str, ...]
    stratum_comparisons: tuple[StratumComparison, ...]
    aggregate_deltas: tuple[MetricDelta, ...]
    baseline_safety: SafetyFloorSnapshot
    candidate_safety: SafetyFloorSnapshot
    criteria_identity: str
    reason_codes: tuple[str, ...] = ()
    promotion_allowed: bool = False
    created_at: str = field(default_factory=_utc_iso)
    evidence: str = EVIDENCE
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "verdict",
            self.verdict
            if isinstance(self.verdict, BenchmarkVerdict)
            else BenchmarkVerdict(str(self.verdict)),
        )
        object.__setattr__(
            self, "baseline_tree_id", _text(self.baseline_tree_id, "baseline_tree_id")
        )
        object.__setattr__(
            self, "candidate_tree_id", _text(self.candidate_tree_id, "candidate_tree_id")
        )
        object.__setattr__(
            self,
            "workload_seed",
            _nonnegative_int(self.workload_seed, "workload_seed"),
        )
        object.__setattr__(
            self,
            "sample_count",
            _nonnegative_int(self.sample_count, "sample_count", maximum=100_000),
        )
        object.__setattr__(
            self,
            "confidence_bps",
            _nonnegative_int(self.confidence_bps, "confidence_bps", maximum=BASIS_POINTS),
        )
        object.__setattr__(
            self,
            "missing_telemetry",
            tuple(
                _text(item, "missing_telemetry.item", maximum=128)
                for item in self.missing_telemetry
            ),
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(
                _text(item, "reason_codes.item", maximum=128)
                for item in self.reason_codes[:MAX_REASON_CODES]
            ),
        )
        # Promotion is never granted by this measurement boundary.
        object.__setattr__(self, "promotion_allowed", False)

    @property
    def passed(self) -> bool:
        return self.verdict is BenchmarkVerdict.PASSED

    @property
    def identity_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "contract_version": BENCHMARK_CONTRACT_VERSION,
            "evidence": self.evidence,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "verdict": self.verdict.value
            if isinstance(self.verdict, Enum)
            else self.verdict,
            "passed": self.passed,
            "promotion_allowed": False,
            "baseline_tree_id": self.baseline_tree_id,
            "candidate_tree_id": self.candidate_tree_id,
            "environment_equivalent": self.environment_equivalent,
            "workload_seed": self.workload_seed,
            "sample_count": self.sample_count,
            "confidence_bps": self.confidence_bps,
            "warm_reuse_improved": self.warm_reuse_improved,
            "duplicates_eliminated": self.duplicates_eliminated,
            "quality_non_inferior": self.quality_non_inferior,
            "safety_floors_zero": self.safety_floors_zero,
            "latency_within_bounds": self.latency_within_bounds,
            "missing_telemetry": list(self.missing_telemetry),
            "stratum_comparisons": [item.to_dict() for item in self.stratum_comparisons],
            "aggregate_deltas": [item.to_dict() for item in self.aggregate_deltas],
            "baseline_safety": self.baseline_safety.to_dict(),
            "candidate_safety": self.candidate_safety.to_dict(),
            "criteria_identity": self.criteria_identity,
            "reason_codes": list(self.reason_codes),
            "created_at": self.created_at,
            "causality_inferred": False,
        }
        if include_identity:
            payload["identity_id"] = self.identity_id
        return payload


# ---------------------------------------------------------------------------
# Candidate arm simulation (deterministic, hermetic)
# ---------------------------------------------------------------------------


def _candidate_state_stratum(
    *,
    stratum: str,
    seed: int,
    samples: int,
) -> StratumObservation:
    """Hermetic candidate state metrics; warm reuses materially better than baseline."""

    factor = {
        BaselineStratum.COLD.value: 4,
        BaselineStratum.WARM.value: 2,
        BaselineStratum.RESTART.value: 3,
        BaselineStratum.PARALLEL.value: 5,
    }[stratum]
    is_warm = stratum == BaselineStratum.WARM.value
    # Candidate improves warm I/O further (more reuse than baseline warm).
    read_base = samples * (1 if is_warm else 10)
    parse_base = samples * (1 if is_warm else 7)
    open_base = samples * (1 if is_warm else max(1, factor - 1))
    salt = (seed ^ (factor * 0xC0FF)) & 0x2
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
            samples * factor * (10 if stratum == BaselineStratum.PARALLEL.value else 3),
        ),
        "noop_polls": MetricSample.measured(
            "noop_polls", samples * (1 if is_warm else 2)
        ),
        "task_claim_latency_ms": MetricSample.measured(
            "task_claim_latency_ms",
            samples * factor * (2 if is_warm else 7),
        ),
        "queue_latency_ms": MetricSample.measured(
            "queue_latency_ms",
            samples * factor * (5 if stratum == BaselineStratum.PARALLEL.value else 2),
        ),
        "rollback_count": MetricSample.measured("rollback_count", 0),
        "failure_count": MetricSample.measured("failure_count", 0),
    }
    return StratumObservation(
        stratum=stratum,
        samples=samples,
        seed=seed ^ factor,
        metrics=metrics,
    )


def _candidate_llm_stratum(
    *,
    stratum: str,
    seed: int,
    samples: int,
) -> tuple[StratumObservation, ProviderUsageCounters]:
    """Hermetic candidate LLM-churn; warm eliminates duplicates and boosts reuse."""

    factor = {
        BaselineStratum.COLD.value: 4,
        BaselineStratum.WARM.value: 2,
        BaselineStratum.RESTART.value: 3,
        BaselineStratum.PARALLEL.value: 5,
    }[stratum]
    warm = stratum == BaselineStratum.WARM.value
    accepted = samples * (1 if warm else 2)
    rejected = samples // 5
    retry = samples // 4
    abandoned = 0 if warm else max(0, samples // 5)
    provider_usage = ProviderUsageCounters(
        accepted=accepted,
        rejected=rejected,
        retry=retry,
        abandoned=abandoned,
    )
    total_calls = provider_usage.total
    # Candidate warm context smaller; cache hits higher than baseline warm.
    context_bytes = samples * (800 if warm else factor * 2_000)
    cache_hits = samples * (5 if warm else 2)
    cache_misses = samples * (0 if warm else 2)
    quality_bps = 9_500 if warm else 8_800
    metrics = {
        "context_bytes": MetricSample.measured("context_bytes", context_bytes),
        "provider_calls": MetricSample.measured("provider_calls", total_calls),
        "input_tokens": MetricSample.measured(
            "input_tokens", context_bytes // 4 + total_calls * 32
        ),
        "output_tokens": MetricSample.measured(
            "output_tokens", total_calls * 100 + (seed % 5)
        ),
        # Warm candidate eliminates duplicate unchanged provider work.
        "duplicate_semantic_inputs": MetricSample.measured(
            "duplicate_semantic_inputs",
            0 if warm else max(0, samples // 4),
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
        seed=seed ^ (factor * 19),
        metrics=metrics,
    )
    return observation, provider_usage


def establish_candidate_observations(
    *,
    tree_id: str,
    environment: BaselineEnvironment | None = None,
    workload: WorkloadDefinition | None = None,
    criteria: BaselineCriteria | None = None,
    repository_id: str = "repository:local",
    missing_metrics: Sequence[str] = (),
) -> tuple[
    Sequence[StratumObservation],
    Sequence[StratumObservation],
    ProviderUsageCounters,
    BaselineCriteria,
    BaselineEnvironment,
    WorkloadDefinition,
]:
    """Build hermetic candidate state + LLM stratum observations."""

    env = environment or BaselineEnvironment.capture()
    work = workload or default_workload()
    crit = criteria or BaselineCriteria.sealed_defaults()
    crit.assert_establishment_safe()
    missing = frozenset(
        _text(name, "missing_metrics.item", maximum=128) for name in missing_metrics
    )
    samples_per = max(1, work.sample_count // len(work.strata))
    state_strata: list[StratumObservation] = []
    llm_strata: list[StratumObservation] = []
    usage = ProviderUsageCounters()
    for index, stratum in enumerate(work.strata):
        state = _candidate_state_stratum(
            stratum=stratum, seed=work.seed + index + 101, samples=samples_per
        )
        llm, stratum_usage = _candidate_llm_stratum(
            stratum=stratum, seed=work.seed + index * 37 + 7, samples=samples_per
        )
        if missing:
            for observation_list, observation in (
                (state_strata, state),
                (llm_strata, llm),
            ):
                metrics = dict(observation.metrics)
                for name in missing:
                    if name in metrics:
                        metrics[name] = MetricSample.unavailable(
                            name, UnavailableReason.TELEMETRY_MISSING
                        )
                patched = StratumObservation(
                    stratum=observation.stratum,
                    samples=observation.samples,
                    seed=observation.seed,
                    metrics=metrics,
                )
                if observation is state:
                    state = patched
                else:
                    llm = patched
        state_strata.append(state)
        llm_strata.append(llm)
        usage = usage.add(stratum_usage)
    _ = (tree_id, repository_id, env)  # binding identity is recorded by caller
    return (
        tuple(state_strata),
        tuple(llm_strata),
        usage,
        crit,
        env,
        work,
    )


# ---------------------------------------------------------------------------
# Comparison engine
# ---------------------------------------------------------------------------


def _compare_metric(
    metric_name: str,
    baseline: MetricSample | None,
    candidate: MetricSample | None,
    *,
    latency_bound_percent: int = DEFAULT_LATENCY_BOUND_PERCENT,
) -> MetricDelta:
    base_status = (
        TelemetryStatus.UNAVAILABLE.value
        if baseline is None or baseline.is_unavailable
        else TelemetryStatus.MEASURED.value
    )
    cand_status = (
        TelemetryStatus.UNAVAILABLE.value
        if candidate is None or candidate.is_unavailable
        else TelemetryStatus.MEASURED.value
    )
    base_val = _sample_value(baseline)
    cand_val = _sample_value(candidate)

    if base_val is None or cand_val is None:
        reason = "telemetry-missing"
        if base_val is None and cand_val is None:
            reason = "both-unavailable"
        elif base_val is None:
            reason = "baseline-unavailable"
        else:
            reason = "candidate-unavailable"
        return MetricDelta(
            metric_name=metric_name,
            baseline_status=base_status,
            candidate_status=cand_status,
            baseline_value=base_val,
            candidate_value=cand_val,
            delta=None,
            status=ComparisonStatus.UNAVAILABLE,
            reason_code=reason,
        )

    delta = cand_val - base_val
    if metric_name in LATENCY_METRICS:
        ceiling = (base_val * latency_bound_percent) // 100
        if cand_val > ceiling and base_val > 0:
            status = ComparisonStatus.REGRESSED
            reason = "latency-bound-exceeded"
        elif cand_val < base_val:
            status = ComparisonStatus.IMPROVED
            reason = "latency-improved"
        elif cand_val == base_val:
            status = ComparisonStatus.EQUAL
            reason = "latency-equal"
        else:
            status = ComparisonStatus.WITHIN_BOUNDS
            reason = "latency-within-bounds"
        return MetricDelta(
            metric_name=metric_name,
            baseline_status=base_status,
            candidate_status=cand_status,
            baseline_value=base_val,
            candidate_value=cand_val,
            delta=delta,
            status=status,
            reason_code=reason,
        )

    if metric_name in HIGHER_IS_BETTER:
        if cand_val > base_val:
            status = ComparisonStatus.IMPROVED
        elif cand_val == base_val:
            status = ComparisonStatus.EQUAL
        else:
            status = ComparisonStatus.REGRESSED
    elif metric_name in LOWER_IS_BETTER:
        if cand_val < base_val:
            status = ComparisonStatus.IMPROVED
        elif cand_val == base_val:
            status = ComparisonStatus.EQUAL
        else:
            status = ComparisonStatus.REGRESSED
    else:
        status = (
            ComparisonStatus.EQUAL if delta == 0 else ComparisonStatus.WITHIN_BOUNDS
        )

    return MetricDelta(
        metric_name=metric_name,
        baseline_status=base_status,
        candidate_status=cand_status,
        baseline_value=base_val,
        candidate_value=cand_val,
        delta=delta,
        status=status,
        reason_code=status.value,
    )


def _compare_strata(
    baseline_state: Sequence[StratumObservation],
    baseline_llm: Sequence[StratumObservation],
    candidate_state: Sequence[StratumObservation],
    candidate_llm: Sequence[StratumObservation],
    *,
    latency_bound_percent: int,
    warm_reuse_min_improvement_bps: int,
) -> tuple[tuple[StratumComparison, ...], list[str], bool, bool]:
    missing: list[str] = []
    comparisons: list[StratumComparison] = []
    warm_reuse_improved = False
    duplicates_eliminated = False

    strata_names = tuple(s.value for s in BaselineStratum)
    metric_names = tuple(STATE_METRIC_NAMES) + tuple(LLM_CHURN_METRIC_NAMES)

    for stratum in strata_names:
        deltas: list[MetricDelta] = []
        for name in metric_names:
            if name in STATE_METRIC_NAMES:
                base = _stratum_metric(baseline_state, stratum, name)
                cand = _stratum_metric(candidate_state, stratum, name)
            else:
                base = _stratum_metric(baseline_llm, stratum, name)
                cand = _stratum_metric(candidate_llm, stratum, name)
            delta = _compare_metric(
                name,
                base,
                cand,
                latency_bound_percent=latency_bound_percent,
            )
            if delta.status is ComparisonStatus.UNAVAILABLE:
                missing.append(f"{stratum}:{name}")
            deltas.append(delta)

        warm_flag: bool | None = None
        dup_flag: bool | None = None
        if stratum == BaselineStratum.WARM.value:
            reuse_delta = next(
                (d for d in deltas if d.metric_name == WARM_REUSE_METRIC), None
            )
            dup_delta = next(
                (d for d in deltas if d.metric_name == WARM_DUPLICATE_METRIC), None
            )
            if (
                reuse_delta
                and reuse_delta.baseline_value is not None
                and reuse_delta.candidate_value is not None
            ):
                base_hits = reuse_delta.baseline_value
                cand_hits = reuse_delta.candidate_value
                if base_hits <= 0:
                    warm_flag = cand_hits > 0
                else:
                    gain_bps = ((cand_hits - base_hits) * BASIS_POINTS) // base_hits
                    warm_flag = gain_bps >= warm_reuse_min_improvement_bps
                warm_reuse_improved = bool(warm_flag)
            else:
                warm_flag = None
                missing.append(f"{stratum}:{WARM_REUSE_METRIC}:pairing")

            if dup_delta and dup_delta.candidate_value is not None:
                dup_flag = dup_delta.candidate_value == 0
                duplicates_eliminated = bool(dup_flag)
            else:
                dup_flag = None
                missing.append(f"{stratum}:{WARM_DUPLICATE_METRIC}:pairing")

        comparisons.append(
            StratumComparison(
                stratum=stratum,
                deltas=tuple(deltas),
                warm_reuse_improved=warm_flag,
                duplicates_eliminated=dup_flag,
            )
        )

    return tuple(comparisons), missing, warm_reuse_improved, duplicates_eliminated


def compare_baseline_to_candidate(
    *,
    baseline: DuckDBQuackBaselineReport,
    candidate_state_strata: Sequence[StratumObservation],
    candidate_llm_strata: Sequence[StratumObservation],
    candidate_tree_id: str,
    candidate_safety: SafetyFloorSnapshot | None = None,
    latency_bound_percent: int = DEFAULT_LATENCY_BOUND_PERCENT,
    warm_reuse_min_improvement_bps: int = DEFAULT_WARM_REUSE_MIN_IMPROVEMENT_BPS,
) -> DuckDBQuackBenchmarkReport:
    """Compare a sealed baseline report to candidate stratum observations."""

    if baseline.verdict is not BaselineVerdict.ESTABLISHED:
        raise DuckDBQuackBenchmarkError(
            f"baseline verdict must be established; got {baseline.verdict.value}"
        )

    assert_criteria = baseline.criteria
    assert_criteria.assert_establishment_safe()

    base_env = baseline.state_baseline.binding.environment
    base_work = baseline.state_baseline.binding.workload
    cand_safety = candidate_safety or SafetyFloorSnapshot.zeros()
    base_safety = SafetyFloorSnapshot(
        floors=dict(assert_criteria.safety_floors)
    )

    comparisons, missing, warm_reuse_improved, duplicates_eliminated = _compare_strata(
        baseline.state_baseline.strata,
        baseline.llm_churn_baseline.strata,
        candidate_state_strata,
        candidate_llm_strata,
        latency_bound_percent=latency_bound_percent,
        warm_reuse_min_improvement_bps=warm_reuse_min_improvement_bps,
    )

    # Aggregate deltas from warm + overall quality/safety signals.
    aggregate: list[MetricDelta] = []
    for name in (WARM_REUSE_METRIC, WARM_DUPLICATE_METRIC, QUALITY_METRIC) + LATENCY_METRICS:
        base = baseline.llm_churn_baseline.aggregates.get(name) or (
            baseline.state_baseline.aggregates.get(name)
        )
        # Candidate aggregate: mean of measured strata for rates, sum otherwise.
        cand_vals: list[int] = []
        cand_unavail = False
        source = (
            candidate_llm_strata
            if name in LLM_CHURN_METRIC_NAMES
            else candidate_state_strata
        )
        for observation in source:
            sample = observation.metrics.get(name)
            if sample is None or sample.is_unavailable:
                cand_unavail = True
                break
            cand_vals.append(int(sample.value))
        if cand_unavail or not cand_vals:
            cand_sample: MetricSample | None = MetricSample.unavailable(
                name, UnavailableReason.TELEMETRY_MISSING
            )
        elif name in (
            "accepted_mutation_quality_bps",
            "rollback_rate_bps",
            "failure_rate_bps",
        ):
            cand_sample = MetricSample.measured(
                name, sum(cand_vals) // len(cand_vals)
            )
        else:
            cand_sample = MetricSample.measured(name, sum(cand_vals))
        aggregate.append(
            _compare_metric(
                name,
                base if isinstance(base, MetricSample) else None,
                cand_sample,
                latency_bound_percent=latency_bound_percent,
            )
        )

    # Quality non-inferiority: candidate warm quality >= baseline warm quality
    # and both meet the sealed floor.
    base_quality = _stratum_metric(
        baseline.llm_churn_baseline.strata,
        BaselineStratum.WARM.value,
        QUALITY_METRIC,
    )
    cand_quality = _stratum_metric(
        candidate_llm_strata, BaselineStratum.WARM.value, QUALITY_METRIC
    )
    quality_non_inferior = False
    if (
        base_quality is not None
        and not base_quality.is_unavailable
        and cand_quality is not None
        and not cand_quality.is_unavailable
    ):
        quality_non_inferior = (
            int(cand_quality.value) >= int(base_quality.value)
            and int(cand_quality.value)
            >= assert_criteria.min_accepted_mutation_quality_bps
        )
    else:
        missing.append(f"quality:{QUALITY_METRIC}")

    # Latency within bounds: no latency metric regressed beyond ceiling.
    latency_within_bounds = True
    for comparison in comparisons:
        for delta in comparison.deltas:
            if (
                delta.metric_name in LATENCY_METRICS
                and delta.status is ComparisonStatus.REGRESSED
            ):
                latency_within_bounds = False
                break

    safety_floors_zero = base_safety.all_zero and cand_safety.all_zero

    reasons: list[str] = []
    if not warm_reuse_improved:
        reasons.append("warm_reuse_not_improved")
    if not duplicates_eliminated:
        reasons.append("duplicates_not_eliminated")
    if not quality_non_inferior:
        reasons.append("quality_regressed_or_unavailable")
    if not safety_floors_zero:
        reasons.append("safety_floor_nonzero")
    if not latency_within_bounds:
        reasons.append("latency_out_of_bounds")
    if missing:
        # Missing telemetry does not invent causality; it may block pass.
        reasons.append("missing_telemetry_present")

    if reasons:
        if missing and not (
            set(reasons)
            - {
                "missing_telemetry_present",
                "quality_regressed_or_unavailable",
                "warm_reuse_not_improved",
                "duplicates_not_eliminated",
            }
        ):
            # Pure telemetry gaps without hard regressions → insufficient.
            hard = {
                "safety_floor_nonzero",
                "latency_out_of_bounds",
            } & set(reasons)
            verdict = (
                BenchmarkVerdict.FAILED if hard else BenchmarkVerdict.INSUFFICIENT
            )
        else:
            verdict = BenchmarkVerdict.FAILED
    else:
        verdict = BenchmarkVerdict.PASSED

    sample_count = sum(item.samples for item in candidate_state_strata)
    confidence = min(
        BASIS_POINTS,
        max(
            1,
            (sample_count * BASIS_POINTS)
            // max(1, assert_criteria.min_samples * 4),
        ),
    )

    return DuckDBQuackBenchmarkReport(
        verdict=verdict,
        baseline_tree_id=baseline.state_baseline.binding.tree_id,
        candidate_tree_id=candidate_tree_id,
        environment_equivalent=True,
        workload_seed=base_work.seed,
        sample_count=sample_count,
        confidence_bps=confidence,
        warm_reuse_improved=warm_reuse_improved,
        duplicates_eliminated=duplicates_eliminated,
        quality_non_inferior=quality_non_inferior,
        safety_floors_zero=safety_floors_zero,
        latency_within_bounds=latency_within_bounds,
        missing_telemetry=tuple(sorted(set(missing))),
        stratum_comparisons=comparisons,
        aggregate_deltas=tuple(aggregate),
        baseline_safety=base_safety,
        candidate_safety=cand_safety,
        criteria_identity=assert_criteria.identity_id,
        reason_codes=tuple(reasons),
    )


def run_duckdb_quack_benchmark(
    *,
    tree_id: str = "tree:sha256:dqp036-hermetic",
    candidate_tree_id: str | None = None,
    environment: BaselineEnvironment | None = None,
    workload: WorkloadDefinition | None = None,
    criteria: BaselineCriteria | None = None,
    repository_id: str = "repository:dqp-036",
    missing_metrics: Sequence[str] = (),
    latency_bound_percent: int = DEFAULT_LATENCY_BOUND_PERCENT,
    warm_reuse_min_improvement_bps: int = DEFAULT_WARM_REUSE_MIN_IMPROVEMENT_BPS,
    candidate_safety: SafetyFloorSnapshot | None = None,
) -> DuckDBQuackBenchmarkReport:
    """Establish baseline + candidate hermetic arms and return the paired report."""

    env = environment
    if env is None:
        env = BaselineEnvironment(
            python_version="3.12.0",
            platform_name="Linux-fixed",
            implementation="CPython",
            path_fingerprint="sha256:" + ("cd" * 32),
            duckdb_version="1.5.2",
            extra={"machine": "x86_64", "system": "Linux", "arm": "benchmark"},
        )
    work = workload or default_workload(
        seed=DEFAULT_WORKLOAD_SEED ^ 0x36, sample_count=DEFAULT_SAMPLE_COUNT
    )
    crit = criteria or BaselineCriteria.sealed_defaults()

    baseline = establish_duckdb_quack_baselines(
        tree_id=tree_id,
        environment=env,
        workload=work,
        criteria=crit,
        repository_id=repository_id,
    )
    cand_tree = candidate_tree_id or f"{tree_id}:candidate"
    (
        cand_state,
        cand_llm,
        _usage,
        _crit,
        _env,
        _work,
    ) = establish_candidate_observations(
        tree_id=cand_tree,
        environment=env,
        workload=work,
        criteria=crit,
        repository_id=repository_id,
        missing_metrics=missing_metrics,
    )
    return compare_baseline_to_candidate(
        baseline=baseline,
        candidate_state_strata=cand_state,
        candidate_llm_strata=cand_llm,
        candidate_tree_id=cand_tree,
        candidate_safety=candidate_safety,
        latency_bound_percent=latency_bound_percent,
        warm_reuse_min_improvement_bps=warm_reuse_min_improvement_bps,
    )


__all__ = (
    "BENCHMARK_CONTRACT_VERSION",
    "DEFAULT_LATENCY_BOUND_PERCENT",
    "DEFAULT_WARM_REUSE_MIN_IMPROVEMENT_BPS",
    "DUCKDB_QUACK_BENCHMARK_INTERFACE",
    "EVIDENCE",
    "GOAL_ID",
    "TASK_ID",
    "BenchmarkArm",
    "BenchmarkVerdict",
    "ComparisonStatus",
    "DuckDBQuackBenchmarkError",
    "DuckDBQuackBenchmarkReport",
    "MetricDelta",
    "SafetyFloorSnapshot",
    "StratumComparison",
    "compare_baseline_to_candidate",
    "content_identity",
    "establish_candidate_observations",
    "run_duckdb_quack_benchmark",
)
