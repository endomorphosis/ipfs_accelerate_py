"""Terminal Worker Planner–Doctor release gate (WPD-070 / WorkerPlannerDoctorRelease@1).

Current-tree release receipt for promoting kernel-first defaults.  Fail-closed:

* safety floors must be zero;
* required interfaces / modules must be importable on the current tree;
* synthetic-only benchmark cannot alone authorize promotion;
* unauthorized LLM path must remain closed (provider requires residual packet).

This module never mutates the repository or grants process authority.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from .llm_avoidance_benchmark import (
    BenchmarkVerdict,
    default_analytical_holdout,
    run_paired_benchmark,
)


WORKER_PLANNER_DOCTOR_RELEASE_INTERFACE: Final[str] = "WorkerPlannerDoctorRelease@1"
WORKER_PLANNER_DOCTOR_RELEASE_VERSION: Final[int] = 1
WORKER_PLANNER_DOCTOR_RELEASE_EVIDENCE: Final[str] = "wpd/release@1"
RELEASE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/worker-planner-doctor-release-receipt@1"
)

REQUIRED_MODULES: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py.agent_supervisor.todo_daemon.pre_implementation_kernel",
    "ipfs_accelerate_py.agent_supervisor.todo_daemon.pre_implementation_provider_gate",
    "ipfs_accelerate_py.agent_supervisor.todo_daemon.analytical_close_executor",
    "ipfs_accelerate_py.agent_supervisor.todo_daemon.residual_provider_invocation",
    "ipfs_accelerate_py.agent_supervisor.todo_daemon.failure_replan_policy",
    "ipfs_accelerate_py.agent_supervisor.todo_daemon.task_execution_policy",
    "ipfs_accelerate_py.agent_supervisor.todo_daemon.worker_doctor_bridge",
    "ipfs_accelerate_py.agent_supervisor.validation.llm_avoidance_metrics",
    "ipfs_accelerate_py.agent_supervisor.validation.llm_avoidance_benchmark",
    "ipfs_accelerate_py.agent_supervisor.control.default_doctor_factory",
    "ipfs_accelerate_py.agent_supervisor.planning.default_planner_factory",
)

REQUIRED_INTERFACES: Final[Mapping[str, str]] = {
    "pre_implementation_kernel": "PreImplementationKernel@1",
    "provider_gate": "ImplementationDaemon@pre_implementation_kernel",
    "analytical_close": "AnalyticalCloseExecutor@1",
    "failure_replan": "FailureReplanPolicy@1",
    "llm_avoidance_metrics": "LlmAvoidanceMetrics@1",
    "benchmark": "WorkerPlannerDoctorBenchmark@1",
    "release": "WorkerPlannerDoctorRelease@1",
}


class ReleaseVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    BLOCKED_SYNTHETIC = "blocked_synthetic"


class WorkerPlannerDoctorReleaseError(ValueError):
    """Fail-closed rejection for an incomplete or unsafe release evaluation."""


@dataclass(frozen=True)
class SafetyFloors:
    unauthorized_provider_calls: int = 0
    scope_escape_writes: int = 0
    free_reprompt_events: int = 0

    @property
    def all_zero(self) -> bool:
        return (
            self.unauthorized_provider_calls == 0
            and self.scope_escape_writes == 0
            and self.free_reprompt_events == 0
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "unauthorized_provider_calls": self.unauthorized_provider_calls,
            "scope_escape_writes": self.scope_escape_writes,
            "free_reprompt_events": self.free_reprompt_events,
        }


@dataclass(frozen=True)
class ReleaseReceipt:
    verdict: ReleaseVerdict
    promotion_allowed: bool
    safety_floors: SafetyFloors
    modules_present: tuple[str, ...]
    modules_missing: tuple[str, ...]
    benchmark_verdict: str
    benchmark_provider_call_reduction: int
    reason_codes: tuple[str, ...]
    interfaces: Mapping[str, str]
    evidence: str = WORKER_PLANNER_DOCTOR_RELEASE_EVIDENCE
    interface: str = WORKER_PLANNER_DOCTOR_RELEASE_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RELEASE_RECEIPT_SCHEMA,
            "contract_version": WORKER_PLANNER_DOCTOR_RELEASE_VERSION,
            "interface": self.interface,
            "evidence": self.evidence,
            "verdict": self.verdict.value,
            "promotion_allowed": self.promotion_allowed,
            "safety_floors": self.safety_floors.to_dict(),
            "modules_present": list(self.modules_present),
            "modules_missing": list(self.modules_missing),
            "benchmark_verdict": self.benchmark_verdict,
            "benchmark_provider_call_reduction": self.benchmark_provider_call_reduction,
            "reason_codes": list(self.reason_codes),
            "interfaces": dict(self.interfaces),
        }


def _probe_modules(modules: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    present: list[str] = []
    missing: list[str] = []
    for name in modules:
        try:
            importlib.import_module(name)
            present.append(name)
        except Exception:
            missing.append(name)
    return tuple(present), tuple(missing)


def evaluate_release(
    *,
    safety_floors: SafetyFloors | None = None,
    synthetic_only: bool = True,
    require_modules: Sequence[str] = REQUIRED_MODULES,
) -> ReleaseReceipt:
    """Evaluate the current-tree WPD release gate."""

    floors = safety_floors or SafetyFloors()
    present, missing = _probe_modules(require_modules)
    bench = run_paired_benchmark(
        default_analytical_holdout(),
        synthetic_only=synthetic_only,
    )
    reasons: list[str] = []
    if not floors.all_zero:
        reasons.append("safety_floor_nonzero")
    if missing:
        reasons.append("required_module_missing")
    if bench.provider_call_reduction <= 0 and bench.baseline.provider_calls > 0:
        reasons.append("benchmark_no_provider_reduction")
    if not bench.quality_non_inferior:
        reasons.append("benchmark_quality_inferior")
    if not bench.safety_floors_zero:
        reasons.append("benchmark_safety_floor_nonzero")
    if synthetic_only:
        reasons.append("synthetic_only_blocks_promotion")

    promotion = (
        floors.all_zero
        and not missing
        and bench.provider_call_reduction > 0
        and bench.quality_non_inferior
        and bench.safety_floors_zero
        and not synthetic_only
    )
    if promotion:
        verdict = ReleaseVerdict.PASS
    elif synthetic_only and not (set(reasons) - {"synthetic_only_blocks_promotion"}):
        verdict = ReleaseVerdict.BLOCKED_SYNTHETIC
    elif (
        synthetic_only
        and floors.all_zero
        and not missing
        and bench.provider_call_reduction > 0
        and bench.quality_non_inferior
        and bench.safety_floors_zero
    ):
        verdict = ReleaseVerdict.BLOCKED_SYNTHETIC
    else:
        verdict = ReleaseVerdict.FAIL

    return ReleaseReceipt(
        verdict=verdict,
        promotion_allowed=bool(promotion),
        safety_floors=floors,
        modules_present=present,
        modules_missing=missing,
        benchmark_verdict=bench.verdict.value,
        benchmark_provider_call_reduction=bench.provider_call_reduction,
        reason_codes=tuple(dict.fromkeys(reasons)),
        interfaces=dict(REQUIRED_INTERFACES),
    )


__all__ = [
    "RELEASE_RECEIPT_SCHEMA",
    "REQUIRED_INTERFACES",
    "REQUIRED_MODULES",
    "WORKER_PLANNER_DOCTOR_RELEASE_EVIDENCE",
    "WORKER_PLANNER_DOCTOR_RELEASE_INTERFACE",
    "WORKER_PLANNER_DOCTOR_RELEASE_VERSION",
    "ReleaseReceipt",
    "ReleaseVerdict",
    "SafetyFloors",
    "WorkerPlannerDoctorReleaseError",
    "evaluate_release",
]
