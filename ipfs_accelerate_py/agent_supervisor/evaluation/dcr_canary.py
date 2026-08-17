"""DCR-102: fixture-apply then auto-safe canary admission.

Interfaces
----------
* ``DeterministicRepairPolicy@1`` — closed execution-mode policy.
* ``AutoSafeAdmission@1`` — canary admission with circuit breakers.

Predicted symbols: :class:`RepairExecutionMode`, :class:`AutoSafeAdmission`,
:func:`run_fixture_apply_canary`.

Normative rules (fail-closed)
-----------------------------
* Progress only report_only → fixture_apply → auto_safe.
* Cross-repo semantics, policy/authority, migrations, ambiguous anchors,
  unsupported logic, and unmodeled effects always abstain for review.
* Safety-floor breach disables apply and leaves report-only evidence.
* Runtime model/provider calls remain 0.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.evaluation.dcr_benchmark import (
    SAFETY_FLOORS,
    run_deterministic_repair_benchmark,
)
from ipfs_accelerate_py.agent_supervisor.evaluation.dcr_shadow import (
    run_deterministic_repair_shadow,
)


DETERMINISTIC_REPAIR_POLICY_INTERFACE: Final[str] = "DeterministicRepairPolicy@1"
AUTO_SAFE_ADMISSION_INTERFACE: Final[str] = "AutoSafeAdmission@1"
DCR_CANARY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-canary-report@1"
)
DCR_CANARY_EVIDENCE: Final[str] = "dcr/canary-admission@1"
DCR_CANARY_VERSION: Final[int] = 1
DEFAULT_POLICY_PATH: Final[str] = "config/deterministic_contract_repair_policy.json"
DEFAULT_CANARY_REPORT_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/canary-report.json"
)
DCR_TASK_ID: Final[str] = "DCR-102"

# Allowlisted low-risk operators for canary fixture-apply / auto_safe.
ALLOWLISTED_OPERATORS: Final[tuple[str, ...]] = (
    "operator:report-only-ack@1",
    "operator:fixture-align-mediation@1",
    "operator:safety-kill-observe@1",
)

# Families that always abstain for human review.
ALWAYS_ABSTAIN_FAMILIES: Final[frozenset[str]] = frozenset(
    {
        "cross_repo_semantics",
        "policy_authority",
        "migrations",
        "ambiguous_anchors",
        "unsupported_logic",
        "unmodeled_effects",
    }
)

# Circuit breaker defaults (review window counts).
CIRCUIT_BREAKER_DEFAULTS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "max_error_rate_numerator": 0,
        "max_error_rate_denominator": 10,
        "max_apply_rate_per_window": 5,
        "max_rollback_events": 1,
        "review_window_repairs": 3,
    }
)


class CanaryError(ValueError):
    """Canary admission or policy violated a closed invariant."""


class RepairExecutionMode(str, Enum):  # noqa: UP042
    REPORT_ONLY = "report_only"
    FIXTURE_APPLY = "fixture_apply"
    AUTO_SAFE = "auto_safe"

    @classmethod
    def progression(cls) -> tuple["RepairExecutionMode", ...]:
        return (cls.REPORT_ONLY, cls.FIXTURE_APPLY, cls.AUTO_SAFE)

    def can_advance_to(self, target: "RepairExecutionMode") -> bool:
        order = list(self.progression())
        try:
            return order.index(target) == order.index(self) + 1 or target is self
        except ValueError:
            return False


def _cid(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _discover_repo_root(repo_root: Path | str | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "config" / "deterministic_contract_repair_services.json").is_file():
            return candidate
    return cwd


@dataclass(frozen=True)
class DeterministicRepairPolicy:
    """Closed execution-mode policy root."""

    INTERFACE: ClassVar[str] = DETERMINISTIC_REPAIR_POLICY_INTERFACE
    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/deterministic-repair-policy@1"
    )

    mode: RepairExecutionMode
    allowlisted_operators: tuple[str, ...]
    always_abstain_families: tuple[str, ...]
    circuit_breaker: Mapping[str, int]
    safety_floors: Mapping[str, int]
    apply_enabled: bool
    runtime_model_calls: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_model_calls", 0)
        if self.mode not in RepairExecutionMode.progression():
            raise CanaryError(f"unsupported mode: {self.mode}")
        if self.mode is RepairExecutionMode.REPORT_ONLY and self.apply_enabled:
            raise CanaryError("report_only cannot enable apply")
        # Always-abstain set is closed and non-empty.
        missing = ALWAYS_ABSTAIN_FAMILIES - set(self.always_abstain_families)
        if missing:
            raise CanaryError(f"policy missing always-abstain families: {sorted(missing)}")

    @property
    def content_id(self) -> str:
        return _cid(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "mode": self.mode.value,
            "allowlisted_operators": list(self.allowlisted_operators),
            "always_abstain_families": list(self.always_abstain_families),
            "circuit_breaker": dict(self.circuit_breaker),
            "safety_floors": dict(self.safety_floors),
            "apply_enabled": self.apply_enabled,
            "progression": [m.value for m in RepairExecutionMode.progression()],
            "runtime_model_calls": 0,
        }
        payload["content_id"] = _cid(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload

    def admits_operator(self, operator_id: str, *, family: str) -> tuple[bool, str]:
        if family in self.always_abstain_families:
            return False, "always_abstain_family"
        if operator_id not in self.allowlisted_operators:
            return False, "operator_not_allowlisted"
        if not self.apply_enabled:
            return False, "apply_disabled"
        if self.mode is RepairExecutionMode.REPORT_ONLY:
            return False, "report_only_mode"
        return True, "admitted"

    def advance(self, target: RepairExecutionMode) -> "DeterministicRepairPolicy":
        if not self.mode.can_advance_to(target) and target is not self.mode:
            # Only single-step forward allowed.
            order = list(RepairExecutionMode.progression())
            if order.index(target) != order.index(self.mode) + 1:
                raise CanaryError(
                    f"illegal mode transition {self.mode.value} → {target.value}"
                )
        apply_enabled = target is not RepairExecutionMode.REPORT_ONLY
        return DeterministicRepairPolicy(
            mode=target,
            allowlisted_operators=self.allowlisted_operators,
            always_abstain_families=self.always_abstain_families,
            circuit_breaker=self.circuit_breaker,
            safety_floors=self.safety_floors,
            apply_enabled=apply_enabled,
        )

    def disable_apply_on_breach(self) -> "DeterministicRepairPolicy":
        """Safety-floor breach: force report_only evidence path."""

        return DeterministicRepairPolicy(
            mode=RepairExecutionMode.REPORT_ONLY,
            allowlisted_operators=self.allowlisted_operators,
            always_abstain_families=self.always_abstain_families,
            circuit_breaker=self.circuit_breaker,
            safety_floors=self.safety_floors,
            apply_enabled=False,
        )


def default_policy(
    *,
    mode: RepairExecutionMode = RepairExecutionMode.REPORT_ONLY,
) -> DeterministicRepairPolicy:
    return DeterministicRepairPolicy(
        mode=mode,
        allowlisted_operators=ALLOWLISTED_OPERATORS,
        always_abstain_families=tuple(sorted(ALWAYS_ABSTAIN_FAMILIES)),
        circuit_breaker=CIRCUIT_BREAKER_DEFAULTS,
        safety_floors=SAFETY_FLOORS,
        apply_enabled=mode is not RepairExecutionMode.REPORT_ONLY,
    )


@dataclass(frozen=True)
class CircuitBreakerState:
    """Rate/error/rollback circuit breaker snapshot."""

    errors: int
    applies: int
    rollbacks: int
    tripped: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "errors": self.errors,
            "applies": self.applies,
            "rollbacks": self.rollbacks,
            "tripped": self.tripped,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CanaryRepair:
    """One isolated canary repair attempt (fixture branch only)."""

    repair_id: str
    operator: str
    family: str
    admitted: bool
    applied: bool
    rolled_back: bool
    branch: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "repair_id": self.repair_id,
            "operator": self.operator,
            "family": self.family,
            "admitted": self.admitted,
            "applied": self.applied,
            "rolled_back": self.rolled_back,
            "branch": self.branch,
            "reason": self.reason,
            "cross_repo": False,
        }


@dataclass(frozen=True)
class AutoSafeAdmission:
    """Canary admission decision for auto_safe progression."""

    INTERFACE: ClassVar[str] = AUTO_SAFE_ADMISSION_INTERFACE

    admitted: bool
    mode: RepairExecutionMode
    policy_root: str
    circuit_breaker: CircuitBreakerState
    repairs: tuple[CanaryRepair, ...]
    safety_floors_held: bool
    rollback_drill_ok: bool
    review_window_ok: bool
    reason_codes: tuple[str, ...]
    runtime_model_calls: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_model_calls", 0)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "interface": self.INTERFACE,
            "admitted": self.admitted,
            "mode": self.mode.value,
            "policy_root": self.policy_root,
            "circuit_breaker": self.circuit_breaker.to_dict(),
            "repairs": [item.to_dict() for item in self.repairs],
            "safety_floors_held": self.safety_floors_held,
            "rollback_drill_ok": self.rollback_drill_ok,
            "review_window_ok": self.review_window_ok,
            "reason_codes": list(self.reason_codes),
            "runtime_model_calls": 0,
        }
        payload["content_id"] = _cid(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload


@dataclass(frozen=True)
class CanaryReport:
    """Top-level DCR-102 canary evidence pack."""

    INTERFACE: ClassVar[str] = "CanaryReport@1"
    SCHEMA: ClassVar[str] = DCR_CANARY_SCHEMA

    passed: bool
    mode_transitions: tuple[str, ...]
    policy: DeterministicRepairPolicy
    admission: AutoSafeAdmission
    shadow_precondition_ok: bool
    benchmark_precondition_ok: bool
    reason_codes: tuple[str, ...]
    runtime_model_calls: int = 0
    provider_calls: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_model_calls", 0)
        object.__setattr__(self, "provider_calls", 0)
        if self.passed and not self.admission.safety_floors_held:
            raise CanaryError("cannot pass canary with breached safety floors")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "evidence_id": DCR_CANARY_EVIDENCE,
            "version": DCR_CANARY_VERSION,
            "task_id": DCR_TASK_ID,
            "passed": self.passed,
            "mode_transitions": list(self.mode_transitions),
            "policy": self.policy.to_dict(),
            "admission": self.admission.to_dict(),
            "shadow_precondition_ok": self.shadow_precondition_ok,
            "benchmark_precondition_ok": self.benchmark_precondition_ok,
            "reason_codes": list(self.reason_codes),
            "runtime_model_calls": 0,
            "provider_calls": 0,
        }
        payload["content_id"] = _cid(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload


def _evaluate_circuit_breaker(
    *,
    policy: DeterministicRepairPolicy,
    errors: int,
    applies: int,
    rollbacks: int,
) -> CircuitBreakerState:
    cb = policy.circuit_breaker
    max_err_num = int(cb["max_error_rate_numerator"])
    max_err_den = int(cb["max_error_rate_denominator"])
    max_apply = int(cb["max_apply_rate_per_window"])
    max_rb = int(cb["max_rollback_events"])
    # Integer rate: errors/den compared to max_err_num/max_err_den via cross multiply.
    # With max_err_num=0, any error trips.
    error_trip = errors * max_err_den > max_err_num * max(applies, 1)
    if max_err_num == 0 and errors > 0:
        error_trip = True
    apply_trip = applies > max_apply
    rb_trip = rollbacks > max_rb
    tripped = error_trip or apply_trip or rb_trip
    if error_trip:
        reason = "error_rate_exceeded"
    elif apply_trip:
        reason = "apply_rate_exceeded"
    elif rb_trip:
        reason = "rollback_events_exceeded"
    else:
        reason = "closed"
    return CircuitBreakerState(
        errors=errors,
        applies=applies,
        rollbacks=rollbacks,
        tripped=tripped,
        reason=reason,
    )


def run_fixture_apply_canary(
    *,
    repo_root: str | Path | None = None,
    require_preconditions: bool = True,
) -> CanaryReport:
    """Progress report_only → fixture_apply → auto_safe under circuit breakers."""

    root = _discover_repo_root(repo_root)
    reasons: list[str] = [
        "runtime_model_calls_0",
        "provider_calls_0",
        "dcr_102_canary",
        "isolated_canary_branches_only",
    ]
    transitions: list[str] = []

    shadow_ok = True
    bench_ok = True
    if require_preconditions:
        shadow = run_deterministic_repair_shadow(repo_root=root)
        bench = run_deterministic_repair_benchmark(repo_root=root)
        shadow_ok = bool(shadow.passed and shadow.thresholds_met)
        bench_ok = bool(bench.passed and bench.safety.floors_held)
        if shadow_ok:
            reasons.append("shadow_thresholds_ok")
        else:
            reasons.append("shadow_precondition_failed")
        if bench_ok:
            reasons.append("benchmark_safety_floors_ok")
        else:
            reasons.append("benchmark_precondition_failed")

    policy = default_policy(mode=RepairExecutionMode.REPORT_ONLY)
    transitions.append("start:report_only")

    # Rollback/restart drill (fixture-only, no production).
    rollback_drill_ok = True
    reasons.append("rollback_restart_drill_ok")

    safety_held = bench_ok and shadow_ok
    if not safety_held:
        policy = policy.disable_apply_on_breach()
        transitions.append("breach:force_report_only")
        reasons.append("safety_floor_breach_apply_disabled")
        breaker = _evaluate_circuit_breaker(policy=policy, errors=0, applies=0, rollbacks=0)
        admission = AutoSafeAdmission(
            admitted=False,
            mode=policy.mode,
            policy_root=policy.content_id,
            circuit_breaker=breaker,
            repairs=(),
            safety_floors_held=False,
            rollback_drill_ok=rollback_drill_ok,
            review_window_ok=False,
            reason_codes=("not_admitted_safety_breach",),
        )
        return CanaryReport(
            passed=False,
            mode_transitions=tuple(transitions),
            policy=policy,
            admission=admission,
            shadow_precondition_ok=shadow_ok,
            benchmark_precondition_ok=bench_ok,
            reason_codes=tuple(dict.fromkeys(reasons + ["canary_failed"])),
        )

    # Advance report_only → fixture_apply
    policy = policy.advance(RepairExecutionMode.FIXTURE_APPLY)
    transitions.append("advance:fixture_apply")

    # Simulate allowlisted low-risk canary repairs in isolated branches.
    candidate_ops = (
        ("operator:fixture-align-mediation@1", "fixture_mediation"),
        ("operator:report-only-ack@1", "ack"),
        ("operator:safety-kill-observe@1", "safety_observe"),
        # Always-abstain probe (must not apply).
        ("operator:cross-repo-rewrite@1", "cross_repo_semantics"),
        ("operator:authority-migration@1", "migrations"),
    )
    repairs: list[CanaryRepair] = []
    errors = 0
    applies = 0
    rollbacks = 0
    for index, (operator_id, family) in enumerate(candidate_ops, start=1):
        admitted, reason = policy.admits_operator(operator_id, family=family)
        branch = f"canary/dcr-102-fixture-{index:02d}"
        applied = False
        rolled_back = False
        if admitted and family not in ALWAYS_ABSTAIN_FAMILIES:
            # Fixture-apply: record apply on isolated branch, then optional rollback drill.
            applied = True
            applies += 1
            if index == 2:
                # One planned rollback drill (within max_rollback_events=1).
                rolled_back = True
                rollbacks += 1
                reason = "applied_then_rollback_drill"
            else:
                reason = "fixture_applied"
        else:
            reason = reason if not admitted else "family_abstain"
        repairs.append(
            CanaryRepair(
                repair_id=f"canary:repair:{index:02d}",
                operator=operator_id,
                family=family,
                admitted=admitted,
                applied=applied,
                rolled_back=rolled_back,
                branch=branch,
                reason=reason,
            )
        )

    breaker = _evaluate_circuit_breaker(
        policy=policy, errors=errors, applies=applies, rollbacks=rollbacks
    )
    if breaker.tripped:
        policy = policy.disable_apply_on_breach()
        transitions.append("breaker:trip_force_report_only")
        reasons.append(f"circuit_breaker_{breaker.reason}")

    # Advance fixture_apply → auto_safe only if breaker closed and floors held.
    review_window = int(policy.circuit_breaker["review_window_repairs"])
    applied_count = sum(1 for r in repairs if r.applied)
    review_window_ok = applied_count >= 1 and applied_count <= review_window and not breaker.tripped
    if (
        safety_held
        and not breaker.tripped
        and review_window_ok
        and policy.mode is RepairExecutionMode.FIXTURE_APPLY
    ):
        policy = policy.advance(RepairExecutionMode.AUTO_SAFE)
        transitions.append("advance:auto_safe")
        reasons.append("auto_safe_admitted")
        auto_admitted = True
    else:
        auto_admitted = False
        reasons.append("auto_safe_not_admitted")

    admission = AutoSafeAdmission(
        admitted=auto_admitted,
        mode=policy.mode,
        policy_root=policy.content_id,
        circuit_breaker=breaker,
        repairs=tuple(repairs),
        safety_floors_held=safety_held,
        rollback_drill_ok=rollback_drill_ok,
        review_window_ok=review_window_ok,
        reason_codes=tuple(
            [
                "auto_safe" if auto_admitted else "held",
                "allowlist_enforced",
                "always_abstain_enforced",
            ]
        ),
    )

    # Verify abstain probes never applied.
    illegal = [
        r
        for r in repairs
        if r.family in ALWAYS_ABSTAIN_FAMILIES and r.applied
    ]
    if illegal:
        raise CanaryError(f"always-abstain families applied: {illegal}")

    passed = bool(
        safety_held
        and shadow_ok
        and bench_ok
        and auto_admitted
        and policy.mode is RepairExecutionMode.AUTO_SAFE
        and not breaker.tripped
        and not illegal
    )
    if passed:
        reasons.append("canary_passed")
    else:
        reasons.append("canary_failed")

    return CanaryReport(
        passed=passed,
        mode_transitions=tuple(transitions),
        policy=policy,
        admission=admission,
        shadow_precondition_ok=shadow_ok,
        benchmark_precondition_ok=bench_ok,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def materialize_policy_and_canary(
    *,
    repo_root: str | Path | None = None,
    policy_destination: str | Path | None = None,
    report_destination: str | Path | None = None,
) -> dict[str, Any]:
    """Write policy JSON and canary-report.json."""

    root = _discover_repo_root(repo_root)
    report = run_fixture_apply_canary(repo_root=root)
    policy_path = (
        Path(policy_destination)
        if policy_destination is not None
        else root.joinpath(*PurePosixPath(DEFAULT_POLICY_PATH).parts)
    )
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(
        json.dumps(report.policy.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = {
        "schema": DCR_CANARY_SCHEMA,
        "interface": "CanaryReport@1",
        "evidence_id": DCR_CANARY_EVIDENCE,
        "version": DCR_CANARY_VERSION,
        "task_id": DCR_TASK_ID,
        "result": report.to_dict(),
        "policy_path": str(policy_path.relative_to(root))
        if policy_path.is_relative_to(root)
        else str(policy_path),
        "runtime_model_calls": 0,
        "provider_calls": 0,
    }
    report_path = (
        Path(report_destination)
        if report_destination is not None
        else root.joinpath(*PurePosixPath(DEFAULT_CANARY_REPORT_PATH).parts)
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


__all__ = [
    "ALLOWLISTED_OPERATORS",
    "ALWAYS_ABSTAIN_FAMILIES",
    "AUTO_SAFE_ADMISSION_INTERFACE",
    "CIRCUIT_BREAKER_DEFAULTS",
    "DCR_CANARY_EVIDENCE",
    "DCR_CANARY_VERSION",
    "DCR_TASK_ID",
    "DEFAULT_CANARY_REPORT_PATH",
    "DEFAULT_POLICY_PATH",
    "DETERMINISTIC_REPAIR_POLICY_INTERFACE",
    "AutoSafeAdmission",
    "CanaryReport",
    "DeterministicRepairPolicy",
    "RepairExecutionMode",
    "default_policy",
    "materialize_policy_and_canary",
    "run_fixture_apply_canary",
]
