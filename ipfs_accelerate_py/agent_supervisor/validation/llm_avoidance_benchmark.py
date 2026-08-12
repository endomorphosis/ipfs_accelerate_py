"""Paired LLM-avoidance benchmark (WPD-051 / WorkerPlannerDoctorBenchmark@1).

Compares a provider-first **baseline** mock against a kernel-first **challenger**
on a fixed holdout fixture corpus.  This module never grants completion,
mutation, promotion, or process authority.

Acceptance rules (fail-closed):

* Challenger must show strictly lower measured provider calls on the analytical
  corpus (when baseline has at least one call).
* Quality oracle must be non-inferior (challenger quality >= baseline - slack).
* Safety floors (unauthorized provider, scope escape, free re-prompt) must be
  zero for both arms.
* Synthetic-only runs cannot promote (``promotion_allowed`` stays false).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

from .llm_avoidance_metrics import (
    AttemptDisposition,
    AttemptAttribution,
    aggregate_attempt_attributions,
    attribute_attempt,
)


WORKER_PLANNER_DOCTOR_BENCHMARK_INTERFACE: Final[str] = (
    "WorkerPlannerDoctorBenchmark@1"
)
WORKER_PLANNER_DOCTOR_BENCHMARK_VERSION: Final[int] = 1
WORKER_PLANNER_DOCTOR_BENCHMARK_EVIDENCE: Final[str] = (
    "wpd/llm-avoidance-benchmark@1"
)
BENCHMARK_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/worker-planner-doctor-benchmark-result@1"
)
HOLDOUT_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/worker-planner-doctor-holdout-manifest@1"
)

DEFAULT_QUALITY_SLACK: Final[float] = 0.0
MAX_FIXTURES: Final[int] = 10_000


class BenchmarkArm(str, Enum):
    BASELINE = "baseline"  # provider-first mock
    CHALLENGER = "challenger"  # kernel-first


class BenchmarkVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SYNTHETIC_BLOCKED = "synthetic_blocked"


class LlmAvoidanceBenchmarkError(ValueError):
    """Fail-closed rejection for an unsafe or incomplete benchmark run."""


@dataclass(frozen=True)
class HoldoutFixture:
    """One fixed analytical fixture in the holdout corpus."""

    fixture_id: str
    analytical_unique: bool = True
    quality_score: float = 1.0
    baseline_provider_calls: int = 1
    challenger_provider_calls: int = 0
    baseline_disposition: str = AttemptDisposition.RESIDUAL_LLM_AUTHORIZED.value
    challenger_disposition: str = AttemptDisposition.CLOSED_DETERMINISTIC.value
    safety_floor_violations: int = 0

    def __post_init__(self) -> None:
        if not str(self.fixture_id or "").strip():
            raise LlmAvoidanceBenchmarkError("fixture_id is required")
        if self.baseline_provider_calls < 0 or self.challenger_provider_calls < 0:
            raise LlmAvoidanceBenchmarkError("provider_calls must be non-negative")
        if self.safety_floor_violations < 0:
            raise LlmAvoidanceBenchmarkError("safety_floor_violations must be non-negative")
        if not (0.0 <= float(self.quality_score) <= 1.0):
            raise LlmAvoidanceBenchmarkError("quality_score must be in [0, 1]")


@dataclass(frozen=True)
class ArmResult:
    arm: BenchmarkArm
    provider_calls: int
    mean_quality: float
    safety_floor_violations: int
    attempt_count: int
    disposition_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "provider_calls": self.provider_calls,
            "mean_quality": self.mean_quality,
            "safety_floor_violations": self.safety_floor_violations,
            "attempt_count": self.attempt_count,
            "disposition_counts": dict(self.disposition_counts),
        }


@dataclass(frozen=True)
class BenchmarkResult:
    verdict: BenchmarkVerdict
    baseline: ArmResult
    challenger: ArmResult
    provider_call_reduction: int
    quality_non_inferior: bool
    safety_floors_zero: bool
    promotion_allowed: bool
    reason_codes: tuple[str, ...]
    fixture_count: int
    synthetic_only: bool
    evidence: str = WORKER_PLANNER_DOCTOR_BENCHMARK_EVIDENCE
    interface: str = WORKER_PLANNER_DOCTOR_BENCHMARK_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BENCHMARK_RESULT_SCHEMA,
            "contract_version": WORKER_PLANNER_DOCTOR_BENCHMARK_VERSION,
            "interface": self.interface,
            "evidence": self.evidence,
            "verdict": self.verdict.value,
            "baseline": self.baseline.to_dict(),
            "challenger": self.challenger.to_dict(),
            "provider_call_reduction": self.provider_call_reduction,
            "quality_non_inferior": self.quality_non_inferior,
            "safety_floors_zero": self.safety_floors_zero,
            "promotion_allowed": self.promotion_allowed,
            "reason_codes": list(self.reason_codes),
            "fixture_count": self.fixture_count,
            "synthetic_only": self.synthetic_only,
        }


def load_holdout_manifest(path: Path | str) -> tuple[HoldoutFixture, ...]:
    """Load a sealed holdout manifest from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise LlmAvoidanceBenchmarkError("holdout manifest must be an object")
    schema = str(payload.get("schema") or "")
    if schema and schema != HOLDOUT_MANIFEST_SCHEMA:
        raise LlmAvoidanceBenchmarkError(
            f"unsupported holdout manifest schema: {schema}"
        )
    raw = payload.get("fixtures")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise LlmAvoidanceBenchmarkError("fixtures must be a sequence")
    if len(raw) == 0:
        raise LlmAvoidanceBenchmarkError("holdout corpus must be non-empty")
    if len(raw) > MAX_FIXTURES:
        raise LlmAvoidanceBenchmarkError("holdout corpus exceeds max fixtures")
    fixtures: list[HoldoutFixture] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise LlmAvoidanceBenchmarkError("fixture entries must be objects")
        fixtures.append(
            HoldoutFixture(
                fixture_id=str(item.get("fixture_id") or ""),
                analytical_unique=bool(item.get("analytical_unique", True)),
                quality_score=float(item.get("quality_score", 1.0)),
                baseline_provider_calls=int(item.get("baseline_provider_calls", 1)),
                challenger_provider_calls=int(
                    item.get("challenger_provider_calls", 0)
                ),
                baseline_disposition=str(
                    item.get("baseline_disposition")
                    or AttemptDisposition.RESIDUAL_LLM_AUTHORIZED.value
                ),
                challenger_disposition=str(
                    item.get("challenger_disposition")
                    or AttemptDisposition.CLOSED_DETERMINISTIC.value
                ),
                safety_floor_violations=int(item.get("safety_floor_violations", 0)),
            )
        )
    return tuple(fixtures)


def _run_arm(
    arm: BenchmarkArm,
    fixtures: Sequence[HoldoutFixture],
) -> ArmResult:
    attributions: list[AttemptAttribution] = []
    quality_sum = 0.0
    floors = 0
    disposition_counts: dict[str, int] = {}
    for index, fixture in enumerate(fixtures, start=1):
        if arm is BenchmarkArm.BASELINE:
            disposition = fixture.baseline_disposition
            calls = fixture.baseline_provider_calls
        else:
            disposition = fixture.challenger_disposition
            calls = fixture.challenger_provider_calls
        floors += int(fixture.safety_floor_violations)
        quality_sum += float(fixture.quality_score)
        if AttemptDisposition(disposition).attributes_zero_provider_calls:
            # Disposition policy attributes measured zero regardless of mock.
            calls = 0
        attribution = attribute_attempt(
            attempt_id=f"{arm.value}:{fixture.fixture_id}:{index}",
            task_cid=f"fixture:{fixture.fixture_id}",
            disposition=disposition,
            provider_calls=calls,
            input_tokens=0 if calls == 0 else max(1, calls * 100),
            output_tokens=0 if calls == 0 else max(1, calls * 40),
        )
        attributions.append(attribution)
        key = attribution.disposition.value
        disposition_counts[key] = disposition_counts.get(key, 0) + 1
    mean_quality = quality_sum / max(1, len(fixtures))
    provider_calls = 0
    for item in attributions:
        measured = item.provider_calls_measured
        if measured is not None:
            provider_calls += int(measured)
    # Touch aggregate for smoke / side-effect free recompute validation.
    _ = aggregate_attempt_attributions(attributions)
    return ArmResult(
        arm=arm,
        provider_calls=provider_calls,
        mean_quality=mean_quality,
        safety_floor_violations=floors,
        attempt_count=len(attributions),
        disposition_counts=disposition_counts,
    )


def run_paired_benchmark(
    fixtures: Sequence[HoldoutFixture],
    *,
    quality_slack: float = DEFAULT_QUALITY_SLACK,
    synthetic_only: bool = True,
) -> BenchmarkResult:
    """Run baseline vs challenger on the fixed fixture corpus."""

    if not fixtures:
        raise LlmAvoidanceBenchmarkError("fixtures must be non-empty")
    baseline = _run_arm(BenchmarkArm.BASELINE, fixtures)
    challenger = _run_arm(BenchmarkArm.CHALLENGER, fixtures)
    reduction = baseline.provider_calls - challenger.provider_calls
    quality_ok = challenger.mean_quality + 1e-12 >= (
        baseline.mean_quality - float(quality_slack)
    )
    floors_zero = (
        baseline.safety_floor_violations == 0
        and challenger.safety_floor_violations == 0
    )
    reasons: list[str] = []
    if baseline.provider_calls > 0 and reduction <= 0:
        reasons.append("challenger_provider_calls_not_reduced")
    if baseline.provider_calls == 0 and challenger.provider_calls > 0:
        reasons.append("challenger_introduced_provider_calls")
    if not quality_ok:
        reasons.append("quality_inferior")
    if not floors_zero:
        reasons.append("safety_floor_nonzero")
    if synthetic_only:
        reasons.append("synthetic_only_cannot_promote")

    promotion = (
        not reasons
        and not synthetic_only
        and reduction > 0
        and quality_ok
        and floors_zero
    )
    observation_ok = (
        (reduction > 0 or baseline.provider_calls == 0)
        and quality_ok
        and floors_zero
    )
    if not observation_ok:
        verdict = BenchmarkVerdict.FAIL
    elif synthetic_only:
        verdict = BenchmarkVerdict.SYNTHETIC_BLOCKED
        if "synthetic_only_cannot_promote" not in reasons:
            reasons.append("synthetic_only_cannot_promote")
    else:
        verdict = BenchmarkVerdict.PASS if promotion else BenchmarkVerdict.FAIL

    return BenchmarkResult(
        verdict=verdict,
        baseline=baseline,
        challenger=challenger,
        provider_call_reduction=reduction,
        quality_non_inferior=quality_ok,
        safety_floors_zero=floors_zero,
        promotion_allowed=bool(promotion),
        reason_codes=tuple(dict.fromkeys(reasons)),
        fixture_count=len(fixtures),
        synthetic_only=bool(synthetic_only),
    )


def default_analytical_holdout() -> tuple[HoldoutFixture, ...]:
    """Built-in analytical corpus used when no external manifest is supplied."""

    return (
        HoldoutFixture(
            fixture_id="analytical-unique-1",
            analytical_unique=True,
            quality_score=1.0,
            baseline_provider_calls=2,
            challenger_provider_calls=0,
        ),
        HoldoutFixture(
            fixture_id="analytical-unique-2",
            analytical_unique=True,
            quality_score=0.95,
            baseline_provider_calls=1,
            challenger_provider_calls=0,
        ),
        HoldoutFixture(
            fixture_id="analytical-unique-3",
            analytical_unique=True,
            quality_score=1.0,
            baseline_provider_calls=3,
            challenger_provider_calls=0,
        ),
    )


__all__ = [
    "BENCHMARK_RESULT_SCHEMA",
    "HOLDOUT_MANIFEST_SCHEMA",
    "WORKER_PLANNER_DOCTOR_BENCHMARK_EVIDENCE",
    "WORKER_PLANNER_DOCTOR_BENCHMARK_INTERFACE",
    "WORKER_PLANNER_DOCTOR_BENCHMARK_VERSION",
    "ArmResult",
    "BenchmarkArm",
    "BenchmarkResult",
    "BenchmarkVerdict",
    "HoldoutFixture",
    "LlmAvoidanceBenchmarkError",
    "default_analytical_holdout",
    "load_holdout_manifest",
    "run_paired_benchmark",
]
