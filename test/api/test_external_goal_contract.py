"""EAAEF-070: typed goal contracts from handoff objectives."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.external_goal_contract import (
    ExternalGoalContract,
    GoalContractError,
)


def _objective(**overrides):
    payload = {
        "objective_id": "EAAEF-G020",
        "desired_outcomes": ("normalize export", "preserve identities"),
        "prohibited_outcomes": ("self_approve", "hidden_chain_of_thought"),
        "write_scope": ("ipfs_accelerate_py/agent_supervisor/handoff/adapters/codex.py",),
        "authority_ceiling": "preview_only",
        "verification_requirements": ("focused pytest",),
        "proof_requirements": ("content identity",),
        "review_requirements": ("independent supervisor",),
        "completion_evidence": ("test receipt", "patch identity"),
        "timeout_seconds": 7200,
        "cpu_millicores": 4000,
        "ram_mib": 8192,
    }
    payload.update(overrides)
    return payload


def test_compile_goal_contract() -> None:
    contract = ExternalGoalContract.compile(_objective())
    assert contract.objective_id == "EAAEF-G020"
    assert contract.authority_ceiling == "preview_only"
    assert contract.content_id.startswith("b")
    clone = ExternalGoalContract.compile(_objective())
    assert clone.content_id == contract.content_id


def test_unbounded_authority_is_rejected() -> None:
    with pytest.raises(GoalContractError, match="authority"):
        ExternalGoalContract.compile(_objective(authority_ceiling="unbounded"))
    with pytest.raises(GoalContractError, match="self-grant"):
        ExternalGoalContract.compile(_objective(self_granted_authority=True))


def test_overlap_and_missing_evidence_fail() -> None:
    with pytest.raises(GoalContractError, match="overlap"):
        ExternalGoalContract.compile(
            _objective(
                desired_outcomes=("self_approve",),
                prohibited_outcomes=("self_approve",),
            )
        )
    with pytest.raises(GoalContractError, match="evidence"):
        ExternalGoalContract.compile(_objective(completion_evidence=()))


def test_budgets_must_be_positive() -> None:
    with pytest.raises(GoalContractError, match="timeout"):
        ExternalGoalContract.compile(_objective(timeout_seconds=0))
