from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.autonomy.observation_preservation import (
    OBSERVATIONS_NOT_PRESERVED,
    OBSERVATIONS_PRESERVED,
    UNDECLARED_PROFILE,
    WORKER_CHANGED_ROLLOUT,
    observation_preservation_view,
    observations_block_completion,
)


def test_missing_wave_payload_does_not_block() -> None:
    view = observation_preservation_view({})
    assert view["claimed"] is False
    assert view["blocks_completion"] is False
    assert view["completes_task"] is False
    assert observations_block_completion(None) is False


def test_preserved_observations_under_declared_profile_do_not_complete() -> None:
    view = observation_preservation_view(
        {
            "refactor_wave": True,
            "declared_profile": "profile:spar-w4",
            "observation_cids": ("obs:1", "obs:2"),
            "current_observation_cids": ("obs:1", "obs:2"),
        }
    )
    assert view["preserved"] is True
    assert view["blocks_completion"] is False
    assert view["completes_task"] is False
    assert view["reason_code"] == OBSERVATIONS_PRESERVED


def test_mutated_observations_cannot_complete() -> None:
    view = observation_preservation_view(
        {
            "refactor_wave": True,
            "declared_profile": "profile:spar-w4",
            "observation_cids": ("obs:1", "obs:2"),
            "wave_observation_cids": ("obs:1", "obs:mutated"),
        }
    )
    assert view["preserved"] is False
    assert view["blocks_completion"] is True
    assert view["reason_code"] == OBSERVATIONS_NOT_PRESERVED


def test_wave_without_declared_profile_cannot_complete() -> None:
    view = observation_preservation_view(
        {
            "refactor_wave": True,
            "observation_cids": ("obs:1",),
        }
    )
    assert view["blocks_completion"] is True
    assert view["reason_code"] == UNDECLARED_PROFILE


def test_worker_cannot_change_rollout() -> None:
    view = observation_preservation_view(
        {
            "refactor_wave": True,
            "declared_profile": "profile:spar-w4",
            "observation_cids": ("obs:1",),
            "worker_may_change_mode": True,
        }
    )
    assert view["blocks_completion"] is True
    assert view["reason_code"] == WORKER_CHANGED_ROLLOUT
    assert view["accepted_as_authority"] is False
