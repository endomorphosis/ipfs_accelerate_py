"""Focused LGSWF plan-transform proposal checks."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.parallel_plan_compiler import (
    PlanTransformError,
    propose_plan_transform,
)


def test_split_coalesce_rewire_are_proposal_only() -> None:
    for kind in ("split", "coalesce", "rewire"):
        proposal = propose_plan_transform(
            {
                "kind": kind,
                "lifecycle": "future",
                "target": "T1",
                "coverage_equivalent": True,
            }
        )
        assert proposal["proposal"] is True
        assert proposal["accepted"] is False
        assert proposal["coverage_equivalent"] is True


def test_immutable_lifecycle_and_canonical_speculation_rejected() -> None:
    with pytest.raises(PlanTransformError, match="immutable"):
        propose_plan_transform({"kind": "split", "lifecycle": "claimed", "target": "T1"})
    with pytest.raises(PlanTransformError, match="canonical"):
        propose_plan_transform(
            {"kind": "speculate", "lifecycle": "future", "mutate_canonical": True}
        )
    with pytest.raises(PlanTransformError, match="unbounded"):
        propose_plan_transform(
            {
                "kind": "split",
                "lifecycle": "future",
                "amplification": 9,
                "amplification_bound": 4,
            }
        )
