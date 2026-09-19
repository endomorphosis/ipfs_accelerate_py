"""DOEP-092 bounded model-assisted synthesis."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.program_repair_synthesis import (
    ProgramRepairAuthorityError,
    ProgramRepairBoundsError,
    bounded_model_assisted_synthesis,
)


def test_residual_syntax_is_proposal_only() -> None:
    receipt = bounded_model_assisted_synthesis(
        residual_only=True,
        max_model_calls=1,
        model_calls_used=1,
    )
    assert receipt["proposal_only"] is True
    assert receipt["admitted"] is False
    assert receipt["completion_authority"] is False
    assert receipt["model_calls_used"] == 1


def test_behavior_change_and_self_admission_are_refused() -> None:
    with pytest.raises(ProgramRepairAuthorityError, match="residual"):
        bounded_model_assisted_synthesis(
            residual_only=False, max_model_calls=1
        )
    with pytest.raises(ProgramRepairAuthorityError, match="self-admit"):
        bounded_model_assisted_synthesis(
            residual_only=True, max_model_calls=1, self_admits=True
        )


def test_model_call_budget_is_enforced() -> None:
    with pytest.raises(ProgramRepairBoundsError, match="budget exhausted"):
        bounded_model_assisted_synthesis(
            residual_only=True, max_model_calls=1, model_calls_used=2
        )
