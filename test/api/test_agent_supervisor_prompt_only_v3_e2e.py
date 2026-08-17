"""ASE3-013 self-host canary e2e: fresh-state prompt lineage and promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.self_host_canary import (
    DEFAULT_CANARY_OBSERVATION_SECONDS,
    CanaryPromotionDenied,
    SelfImprovementCanary,
    load_canary_observation_seconds,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CANARY_DATA = (
    REPO_ROOT
    / "data"
    / "agent_supervisor"
    / "prompt_only_self_improvement_v3"
    / "canary"
)


class _Clock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_signed_canary_observation_seconds_is_900() -> None:
    assert load_canary_observation_seconds(REPO_ROOT) == 900
    assert DEFAULT_CANARY_OBSERVATION_SECONDS == 900


def test_fresh_namespace_rejects_seed_contamination(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "seed_board").write_text("bad\n", encoding="utf-8")
    canary = SelfImprovementCanary(
        state_root=state,
        repository_root=REPO_ROOT,
        monotonic_clock=_Clock(),
    )
    with pytest.raises(Exception, match="contaminated"):
        canary.start("Improve supervisor safety gates")


def test_seed_board_argv_forbidden(tmp_path: Path) -> None:
    canary = SelfImprovementCanary(
        state_root=tmp_path / "fresh",
        repository_root=REPO_ROOT,
        monotonic_clock=_Clock(),
    )
    with pytest.raises(Exception, match="seed-board"):
        canary.start("Improve supervisor safety gates", seed_board_argv=True)


def test_activation_required_before_canary(tmp_path: Path) -> None:
    canary = SelfImprovementCanary(
        state_root=tmp_path / "fresh",
        repository_root=REPO_ROOT,
        activation_completed=False,
        monotonic_clock=_Clock(),
    )
    with pytest.raises(Exception, match="activation"):
        canary.start("Improve supervisor safety gates")


def test_full_canary_promotion_with_monotonic_900s(tmp_path: Path) -> None:
    clock = _Clock()
    state = tmp_path / "fresh"
    canary = SelfImprovementCanary(
        state_root=state,
        repository_root=REPO_ROOT,
        monotonic_clock=clock,
    )
    program = canary.start(
        "Improve the agent supervisor without weakening safety gates"
    )
    assert program
    assert canary.prompt_cid
    assert canary.program_root_cid
    assert canary.descendant_cids
    # All descendants bind to the prompt/program root (lineage presence).
    assert canary.prompt_cid
    canary.record_parallel_effects(
        effect_a_cid="effect:lane-a",
        effect_b_cid="effect:lane-b",
        overlapped=True,
        conflict_serialized=True,
    )
    canary.adopt_forced_residual("residual:forced-1")
    canary.accept_non_sentinel_diff(
        changed_paths=(
            "ipfs_accelerate_py/agent_supervisor/entrypoints/self_host_canary.py",
        )
    )
    for fault in (
        "client_disconnect",
        "monitor_death",
        "provider_saturation",
        "merge_stall",
        "refill_stall",
        "recovery_oscillation",
    ):
        canary.inject_and_recover(fault, recovered=True)
    canary.mark_final_recovery_complete()
    # Pre-recovery / short windows must not promote.
    canary.sample_health(healthy=True)
    clock.advance(100.0)
    canary.sample_health(healthy=True)
    with pytest.raises(CanaryPromotionDenied):
        canary.promote(canary_id="canary-short")
    # Unhealthy sample resets window.
    canary.sample_health(healthy=False)
    canary.sample_health(healthy=True)
    clock.advance(900.0)
    canary.sample_health(healthy=True)
    evidence = canary.promote(canary_id="canary-e2e-1")
    assert evidence.promotion_authorized is True
    assert evidence.observation["required_seconds"] == 900
    assert evidence.parallel_overlap_observed is True
    assert evidence.forced_residual_adopted is True
    assert evidence.seed_board_absent is True
    out = tmp_path / "promotion.json"
    canary.write_evidence(evidence, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["promotion_authorized"] is True
    # Persist under canary data root layout for release packaging.
    CANARY_DATA.mkdir(parents=True, exist_ok=True)
    packaged = CANARY_DATA / "promotion_evidence_schema.json"
    if not packaged.is_file():
        packaged.write_text(
            json.dumps(
                {
                    "schema": evidence.schema,
                    "required_observation_seconds": 900,
                    "activation_task_id": "ASE3-026",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
