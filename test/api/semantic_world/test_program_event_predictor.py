"""SAWM-026 next-event and next-state prediction."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.program_event_predictor import (
    EventPredictionError,
    predict_next_program_event,
)


def test_structured_event_is_proposal_only() -> None:
    result = predict_next_program_event(
        {
            "event_type": "call",
            "current_state": "s0",
            "target": "mod.fn",
            "allowed_targets": ("mod.fn", "mod.other"),
            "observed_event": "call",
        }
    )
    assert result["event_type"] == "call"
    assert result["proposal_only"] is True
    assert result["observation"] is False
    assert result["completion_authority"] is False
    assert result["matches_observation"] is True


def test_impossible_event_is_rejected() -> None:
    with pytest.raises(EventPredictionError, match="impossible event"):
        predict_next_program_event({"event_type": "teleport"})


def test_ood_and_unavailable_checkpoint_abstain() -> None:
    ood = predict_next_program_event({"event_type": "call", "ood": True})
    assert ood["abstained"] is True
    assert ood["reason_code"] == "ood_rejected"
    missing = predict_next_program_event(
        {"event_type": "call", "checkpoint_unavailable": True}
    )
    assert missing["reason_code"] == "checkpoint_unavailable"
    assert missing["completion_authority"] is False
