"""Focused readiness-gate and frontier-construction checks."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.planning.conflict_free_frontier import (
    READINESS_PREDICATES,
    construct_frontier,
    evaluate_readiness,
)


def _ready(task_id: str, **overrides: bool) -> dict:
    payload = {name: True for name in READINESS_PREDICATES}
    payload.update(overrides)
    payload["task_id"] = task_id
    return payload


def test_each_readiness_predicate_rejects() -> None:
    assert evaluate_readiness(_ready("T0"))["ready"] is True
    for predicate in READINESS_PREDICATES:
        result = evaluate_readiness(_ready("T1", **{predicate: False}))
        assert result["ready"] is False
        assert predicate in result["reasons"]


def test_conflict_free_deterministic_ordering() -> None:
    tasks = [_ready("B"), _ready("A"), _ready("C", legal_lifecycle=False)]
    first = construct_frontier(tasks, conflicts=(("A", "B"),))
    second = construct_frontier(list(reversed(tasks)), conflicts=(("B", "A"),))
    assert first == second
    assert first["candidates"] == ("A",)
    assert "C" in first["rejected"]
    assert "conflict" in first["rejected"]["B"]
