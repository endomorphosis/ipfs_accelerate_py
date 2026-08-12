"""ASE3-013 load/budget canary: multi-fault burst and observation durability."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.self_host_canary import (
    FAULT_CLASSES,
    CanaryPromotionDenied,
    SelfImprovementCanary,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Clock:
    def __init__(self) -> None:
        self.t = 10_000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_load_burst_recoveries_then_900s_health(tmp_path: Path) -> None:
    clock = _Clock()
    canary = SelfImprovementCanary(
        state_root=tmp_path / "load-ns",
        repository_root=REPO_ROOT,
        monotonic_clock=clock,
    )
    canary.start("Load canary: sustain health after recovery burst")
    canary.record_parallel_effects(
        effect_a_cid="load-a",
        effect_b_cid="load-b",
        overlapped=True,
        conflict_serialized=True,
    )
    canary.adopt_forced_residual("load-residual")
    canary.accept_non_sentinel_diff(
        changed_paths=("test/api/test_agent_supervisor_prompt_only_v3_load.py",)
    )
    # Burst inject all faults; each recovers once.
    for fault in FAULT_CLASSES:
        canary.inject_and_recover(fault, recovered=True)
    canary.mark_final_recovery_complete()
    # Continuous samples across the signed window.
    steps = 9
    for _ in range(steps):
        clock.advance(100.0)
        canary.sample_health(healthy=True)
    evidence = canary.promote(canary_id="load-1")
    assert evidence.promotion_authorized
    assert evidence.observation["healthy_samples"] >= steps
    assert evidence.observation["elapsed_seconds"] >= 900.0 - 1e-6


def test_fabricated_short_window_denied(tmp_path: Path) -> None:
    clock = _Clock()
    canary = SelfImprovementCanary(
        state_root=tmp_path / "load-ns2",
        repository_root=REPO_ROOT,
        monotonic_clock=clock,
        canary_observation_seconds=900,
    )
    canary.start("Load canary short window")
    canary.record_parallel_effects(
        effect_a_cid="a",
        effect_b_cid="b",
        overlapped=True,
        conflict_serialized=True,
    )
    canary.adopt_forced_residual("r")
    canary.accept_non_sentinel_diff(changed_paths=("README.md",))
    for fault in FAULT_CLASSES:
        canary.inject_and_recover(fault)
    canary.mark_final_recovery_complete()
    clock.advance(899.0)
    canary.sample_health(healthy=True)
    with pytest.raises(CanaryPromotionDenied, match="observation_window"):
        canary.promote(canary_id="too-short")
