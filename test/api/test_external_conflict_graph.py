"""EAAEF-081: conservative semantic conflict sets."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.external_conflict_graph import (
    ConflictGraph,
    ConflictGraphError,
)


def _scope(task_id: str, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "task_id": task_id,
        "write_scope": ("owned_a.py",),
        "effect_scope": (f"isolated_write:{task_id}",),
    }
    payload.update(overrides)
    return payload


def test_overlapping_files_conflict() -> None:
    graph = ConflictGraph.derive(
        _scope("task-a", write_scope=("shared.py",)),
        _scope("task-b", write_scope=("shared.py",)),
    )
    assert graph.conflicts is True
    assert graph.must_serialize is True
    assert graph.overlaps["files"] == ("shared.py",)
    assert any("files" in reason for reason in graph.reasons)
    assert graph.content_id.startswith("b")


def test_overlapping_effects_conflict() -> None:
    graph = ConflictGraph.derive(
        _scope("task-a", effect_scope=("merge-queue",)),
        _scope("task-b", write_scope=("other.py",), effect_scope=("merge-queue",)),
    )
    assert graph.conflicts is True
    assert graph.overlaps["effects"] == ("merge-queue",)


def test_disjoint_scopes_do_not_conflict() -> None:
    graph = ConflictGraph.derive(_scope("task-a"), _scope("task-b", write_scope=("owned_b.py",)))
    assert graph.conflicts is False
    assert graph.overlaps == {}
    assert graph.reasons == ()


def test_unknown_scope_conflicts() -> None:
    flagged = ConflictGraph.derive(_scope("task-a"), {"task_id": "task-b", "unknown": True})
    assert flagged.conflicts is True
    assert "unknown scope" in flagged.reasons

    token = ConflictGraph.derive(
        _scope("task-a"),
        _scope("task-b", write_scope=("unknown",)),
    )
    assert token.conflicts is True
    assert token.left.unknown is False
    assert token.right.unknown is True

    omitted = ConflictGraph.derive(_scope("task-a"), {"task_id": "task-b"})
    assert omitted.conflicts is True
    assert "unknown scope" in omitted.reasons


def test_named_merge_contract_admits_specific_overlap() -> None:
    left = _scope("task-a", write_scope=("shared.py",))
    right = _scope("task-b", write_scope=("shared.py",))
    blocked = ConflictGraph.derive(left, right)
    assert blocked.conflicts is True

    admitted = ConflictGraph.derive(
        left,
        right,
        merge_contracts=(
            {"name": "serialized_forward_extension", "files": ("shared.py",)},
        ),
    )
    assert admitted.conflicts is False
    assert admitted.admitted["files"] == ("shared.py",)
    assert admitted.admitted_by == ("serialized_forward_extension",)

    other_item = ConflictGraph.derive(
        left,
        right,
        merge_contracts=({"name": "serialized_forward_extension", "files": ("other.py",)},),
    )
    assert other_item.conflicts is True

    with pytest.raises(ConflictGraphError, match="name"):
        ConflictGraph.derive(left, right, merge_contracts=({"files": ("shared.py",)},))
