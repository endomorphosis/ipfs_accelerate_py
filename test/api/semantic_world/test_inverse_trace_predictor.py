"""SAWM-027 inverse trace reconstruction."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.analysis.inverse_trace_predictor import (
    rank_inverse_trace_predecessors,
)


def test_ranks_multiple_valid_predecessors_as_proposal() -> None:
    result = rank_inverse_trace_predecessors(
        {
            "predecessor_states": [
                {"state_id": "s1", "score": 0.2},
                {"state_id": "s0", "score": 0.9},
            ],
            "predecessor_events": [{"event_id": "call", "score": 1.0}],
            "missing_inputs": ["arg0"],
            "reproducing_inputs": ["arg0", "arg1"],
        }
    )
    assert result["states"] == ["s0", "s1"]
    assert result["multiple_valid"] is True
    assert result["reproduction_useful"] is True
    assert result["proposal_only"] is True
    assert result["completion_authority"] is False


def test_stale_and_ood_abstain() -> None:
    stale = rank_inverse_trace_predecessors({"stale": True})
    assert stale["abstained"] is True
    assert stale["reason_code"] == "stale_snapshot"
    ood = rank_inverse_trace_predecessors({"ood": True})
    assert ood["reason_code"] == "ood_rejected"
