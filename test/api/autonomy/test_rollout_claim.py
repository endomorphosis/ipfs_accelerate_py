from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.autonomy.rollout_claim import (
    ROLLOUT_NOT_REQUIRED,
    ROLLOUT_RESPECTED,
    rollout_claim_view,
)


def test_missing_rollout_claim_does_not_block() -> None:
    view = rollout_claim_view({})
    assert view["claimed"] is False
    assert view["blocks_completion"] is False
    assert view["completes_task"] is False


def test_bootstrap_cannot_claim_required() -> None:
    view = rollout_claim_view(
        {
            "claimed_rollout": "required",
            "current_rollout_mode": "bootstrap",
        }
    )
    assert view["blocks_completion"] is True
    assert view["reason_code"] == ROLLOUT_NOT_REQUIRED
    assert view["observed_rollout"] == "off"
    assert view["completes_task"] is False


def test_shadow_cannot_claim_required() -> None:
    view = rollout_claim_view(
        {
            "rollout_required": True,
            "current_rollout_mode": "shadow_plan",
        }
    )
    assert view["blocks_completion"] is True
    assert view["observed_rollout"] == "shadow"


def test_unobserved_mode_does_not_infer_required() -> None:
    view = rollout_claim_view({"claimed_rollout": "required"})
    assert view["blocks_completion"] is True
    assert view["reason_code"] == ROLLOUT_NOT_REQUIRED


def test_observed_required_is_respected_and_does_not_complete() -> None:
    view = rollout_claim_view(
        {
            "claimed_rollout": "required",
            "current_rollout_mode": "required",
        }
    )
    assert view["respected"] is True
    assert view["blocks_completion"] is False
    assert view["completes_task"] is False
    assert view["reason_code"] == ROLLOUT_RESPECTED
    assert view["accepted_as_authority"] is False
