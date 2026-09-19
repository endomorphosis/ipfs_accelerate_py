"""SAWM-041 ablation and efficiency benchmark."""

from __future__ import annotations

import pytest

from benchmarks.agent_supervisor.semantic_addressed_world_model.ablation import (
    AblationError,
    run_semantic_world_ablation,
    validate_semantic_world_ablation,
)


def test_ablation_reports_efficiency_without_completion_authority() -> None:
    result = validate_semantic_world_ablation(
        run_semantic_world_ablation(
            [
                {"rung": "A", "model_calls": 1, "tokens": 10, "prefix_reuse": 0.5, "avoided_calls": 2},
                {"rung": "B", "model_calls": 0, "tokens": 4, "prefix_reuse": 1.0, "avoided_calls": 3},
            ]
        )
    )
    assert result["completion_authority"] is False
    assert result["efficiency"].model_calls == 1
    assert result["safety"].test_weakening == 0


def test_safety_weakening_fails_validation() -> None:
    with pytest.raises(AblationError, match="safety"):
        validate_semantic_world_ablation(
            run_semantic_world_ablation([{"rung": "A", "test_weakening": 1}])
        )
