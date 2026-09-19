"""SAWM-028 repair-operator and graph-delta prediction."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.program_delta_predictor import (
    RepairPredictionError,
    predict_program_graph_delta,
    rank_repair_operators,
)


def test_ranks_operators_only_after_analytical_route_is_closed() -> None:
    blocked = rank_repair_operators(
        {"operators": ["rename", "inline"], "analytical_route_open": True}
    )
    assert blocked["abstained"] is True
    result = rank_repair_operators(
        {"operators": ["rename", "inline"], "scores": {"inline": 0.8, "rename": 0.1}}
    )
    assert result["ranked"] == ("inline", "rename")
    assert result["proposal_only"] is True
    assert result["completion_authority"] is False


def test_protected_path_and_test_weakening_are_rejected() -> None:
    with pytest.raises(RepairPredictionError, match="protected path"):
        predict_program_graph_delta({"sketch": {"path": "run-r2-m27/control.duckdb"}})
    with pytest.raises(RepairPredictionError, match="test weakening"):
        predict_program_graph_delta(
            {"sketch": {"path": "test/api/foo.py", "weakens_tests": True}}
        )
