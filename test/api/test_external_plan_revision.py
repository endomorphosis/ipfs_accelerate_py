"""EAAEF-103: plan revisions never edit claimed history."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.external_plan_revision import (
    PlanRevisionError,
    apply_ops,
)


def test_future_ops_compile() -> None:
    ops = apply_ops(
        (
            {"op": "add", "target_id": "EAAEF-200"},
            {"op": "supersede", "target_id": "EAAEF-199"},
            {"op": "cancel_future", "target_id": "EAAEF-198"},
        )
    )
    assert [item.op for item in ops] == ["add", "supersede", "cancel_future"]


def test_claimed_and_accepted_history_cannot_be_edited() -> None:
    with pytest.raises(PlanRevisionError, match="claimed"):
        apply_ops(({"op": "rewire", "target_id": "EAAEF-010", "claimed": True},))
    with pytest.raises(PlanRevisionError, match="accepted"):
        apply_ops(({"op": "cancel_future", "target_id": "EAAEF-010", "accepted": True},))


def test_unknown_op_fails() -> None:
    with pytest.raises(PlanRevisionError, match="unknown"):
        apply_ops(({"op": "rewrite_history", "target_id": "x"},))
