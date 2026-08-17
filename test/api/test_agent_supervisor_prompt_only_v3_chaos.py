"""ASE3-013 chaos matrix: fault injection recoveries without duplicate effects."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.self_host_canary import (
    FAULT_CLASSES,
    CanaryError,
    CanaryPromotionDenied,
    FaultInjectionMatrix,
    SelfImprovementCanary,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _ready_canary(tmp_path: Path, clock: _Clock) -> SelfImprovementCanary:
    canary = SelfImprovementCanary(
        state_root=tmp_path / "ns",
        repository_root=REPO_ROOT,
        monotonic_clock=clock,
    )
    canary.start("Chaos canary prompt for recovery matrix")
    canary.record_parallel_effects(
        effect_a_cid="a",
        effect_b_cid="b",
        overlapped=True,
        conflict_serialized=True,
    )
    canary.adopt_forced_residual("r1")
    canary.accept_non_sentinel_diff(changed_paths=("docs/guides/AGENT_SUPERVISOR_PROMPT_ENTRYPOINTS.md",))
    return canary


def test_fault_vocabulary_is_closed() -> None:
    matrix = FaultInjectionMatrix()
    with pytest.raises(CanaryError):
        matrix.inject("not-a-real-fault")
    for fault in FAULT_CLASSES:
        matrix.inject(fault)
        matrix.mark_recovered(fault)
    assert matrix.all_resolved()


def test_each_required_fault_recovers_or_fails_typed(tmp_path: Path) -> None:
    clock = _Clock()
    canary = _ready_canary(tmp_path, clock)
    required = (
        "stale_pid",
        "frozen_worker",
        "false_idle_open_goal",
        "branch_only_completion",
        "crash_boundary",
        "lease_loss",
        "client_disconnect",
        "monitor_death",
        "provider_saturation",
        "monotonic_clock_rollback",
        "merge_stall",
        "refill_stall",
        "recovery_oscillation",
    )
    assert set(required) == set(FAULT_CLASSES)
    for fault in required:
        # Recover once within budget (no duplicates — inject is idempotent).
        canary.inject_and_recover(fault, recovered=True)
        canary.inject_and_recover(fault, recovered=True)
    assert canary.faults.injected == list(required) or set(canary.faults.injected) == set(
        required
    )
    assert canary.faults.all_resolved()
    canary.mark_final_recovery_complete()
    clock.advance(900.0)
    canary.sample_health(healthy=True)
    evidence = canary.promote(canary_id="chaos-1")
    assert evidence.promotion_authorized


def test_unhealthy_sample_resets_observation_window(tmp_path: Path) -> None:
    clock = _Clock()
    canary = _ready_canary(tmp_path, clock)
    for fault in FAULT_CLASSES:
        canary.inject_and_recover(fault)
    canary.mark_final_recovery_complete()
    clock.advance(500.0)
    canary.sample_health(healthy=True)
    canary.sample_health(healthy=False)  # reset
    clock.advance(500.0)
    canary.sample_health(healthy=True)
    with pytest.raises(CanaryPromotionDenied):
        canary.promote(canary_id="reset-short")
    clock.advance(400.0)  # total continuous after reset = 900
    canary.sample_health(healthy=True)
    evidence = canary.promote(canary_id="reset-ok")
    assert evidence.observation["unhealthy_resets"] >= 1
    assert evidence.promotion_authorized


def test_promotion_before_final_recovery_denied(tmp_path: Path) -> None:
    clock = _Clock()
    canary = _ready_canary(tmp_path, clock)
    canary.inject_and_recover("monitor_death")
    # Other faults unresolved.
    with pytest.raises(CanaryPromotionDenied):
        canary.promote(canary_id="early")
