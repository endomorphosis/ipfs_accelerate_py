from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.autonomy.bounded_procedure import (
    BOUNDED_PROCEDURE,
    REANALYSIS_REQUIRED,
    UNBOUNDED_PROCEDURE,
    UNVERIFIED_EPISODE,
    bounded_procedure_view,
)


def test_missing_procedure_payload_does_not_block() -> None:
    view = bounded_procedure_view({})
    assert view["claimed"] is False
    assert view["blocks_completion"] is False
    assert view["completes_task"] is False


def test_unbounded_procedure_cannot_complete() -> None:
    view = bounded_procedure_view({"unbounded": True})
    assert view["blocks_completion"] is True
    assert view["reason_code"] == UNBOUNDED_PROCEDURE
    assert view["completes_task"] is False


def test_procedure_without_verified_episode_cannot_complete() -> None:
    view = bounded_procedure_view({"bounded": True, "verified_episode": False})
    assert view["reason_code"] == UNVERIFIED_EPISODE
    assert view["blocks_completion"] is True


def test_skip_reanalysis_cannot_claim_fixed_point() -> None:
    view = bounded_procedure_view(
        {
            "bounded": True,
            "verified_episode": True,
            "skip_reanalysis": True,
            "fixed_point": True,
        }
    )
    assert view["reason_code"] == REANALYSIS_REQUIRED
    assert view["blocks_completion"] is True
    assert view["accepted_as_authority"] is False


def test_bounded_verified_reanalyzed_procedure_does_not_complete() -> None:
    view = bounded_procedure_view(
        {
            "bounded": True,
            "verified_episode": True,
            "reanalysis": True,
        }
    )
    assert view["bounded"] is True
    assert view["blocks_completion"] is False
    assert view["completes_task"] is False
    assert view["reason_code"] == BOUNDED_PROCEDURE
