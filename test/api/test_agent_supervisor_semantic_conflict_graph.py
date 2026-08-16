"""Focused LGSWF conflict-graph extension checks."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import (
    LGSWF_CONFLICT_SCOPES,
    admit_conflict_free_frontier,
    evaluate_semantic_conflict,
)


def test_readers_are_compatible_and_same_symbol_writers_conflict() -> None:
    assert evaluate_semantic_conflict(
        {"mode": "read", "symbol": "A"}, {"mode": "read", "symbol": "A"}
    )["conflict"] is False
    same = evaluate_semantic_conflict(
        {"mode": "write", "symbol": "A"}, {"mode": "write", "symbol": "A"}
    )
    assert same["conflict"] is True
    assert same["scope"] == "exact_symbol"


def test_opaque_fallback_and_exclusive_resource() -> None:
    opaque = evaluate_semantic_conflict(
        {"mode": "write", "opaque": True, "path": "a.py"},
        {"mode": "write", "path": "b.py"},
    )
    assert opaque["conflict"] is True
    assert opaque["reason"] == "opaque-conservative-fallback"
    resource = evaluate_semantic_conflict(
        {"mode": "write", "exclusive_resource": "gpu0"},
        {"mode": "write", "exclusive_resource": "gpu0"},
    )
    assert resource["scope"] == "exclusive_resource"
    assert "opaque_file" in LGSWF_CONFLICT_SCOPES


def test_frontier_rejects_overlapping_writes_deterministically() -> None:
    first = admit_conflict_free_frontier(
        [
            {"task_id": "T1", "mode": "write", "symbol": "A"},
            {"task_id": "T2", "mode": "write", "symbol": "A"},
            {"task_id": "T3", "mode": "write", "symbol": "B"},
        ]
    )
    second = admit_conflict_free_frontier(
        [
            {"task_id": "T1", "mode": "write", "symbol": "A"},
            {"task_id": "T2", "mode": "write", "symbol": "A"},
            {"task_id": "T3", "mode": "write", "symbol": "B"},
        ]
    )
    assert first == second
    assert first["admitted"] == ("T1", "T3")
    assert first["rejected"][0]["task_id"] == "T2"
