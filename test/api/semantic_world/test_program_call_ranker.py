"""SAWM-025 call-target ranking specialist."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.program_call_ranker import (
    CallTargetRankingError,
    CallTargetRankingRequest,
    rank_program_call_targets,
)


def test_ranks_only_static_candidates_and_unknown() -> None:
    result = rank_program_call_targets(
        CallTargetRankingRequest(
            current_symbol="mod.fn",
            static_candidates=("mod.a", "mod.b"),
            unknown_frontier=True,
            scores={"mod.b": 0.9, "mod.a": 0.1},
        )
    )
    assert result.ranked == ("mod.b", "mod.a", "unknown")
    assert result.proposal_only is True
    assert result.admitted is False


def test_cannot_introduce_non_candidate_symbols() -> None:
    with pytest.raises(CallTargetRankingError, match="non-candidate"):
        rank_program_call_targets(
            CallTargetRankingRequest(
                current_symbol="mod.fn",
                static_candidates=("mod.a",),
                scores={"mod.forged": 1.0},
            )
        )


def test_stale_and_ood_abstain() -> None:
    stale = rank_program_call_targets(
        CallTargetRankingRequest(
            current_symbol="mod.fn",
            static_candidates=("mod.a",),
            stale=True,
        )
    )
    assert stale.abstained is True
    assert stale.reason_code == "stale_snapshot"
    ood = rank_program_call_targets(
        {"current_symbol": "mod.fn", "static_candidates": ["mod.a"], "ood": True}
    )
    assert ood.abstained is True
    assert ood.reason_code == "ood_rejected"
