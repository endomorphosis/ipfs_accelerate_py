"""EAAEF-080: composite graph edges keep distinct meanings."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.external_composite_graph import (
    CompositeGraph,
    CompositeGraphError,
    EDGE_KINDS,
)


def test_compose_distinct_edge_kinds() -> None:
    graph = CompositeGraph.compose(
        ("goal-a", "task-1", "file.py", "merge-lane"),
        (
            {"kind": "goal", "source": "goal-a", "target": "task-1", "meaning": "child_covers_parent"},
            {"kind": "task", "source": "task-1", "target": "file.py", "meaning": "write_scope"},
            {"kind": "effect", "source": "task-1", "target": "file.py", "meaning": "isolated_write"},
            {"kind": "merge", "source": "task-1", "target": "merge-lane", "meaning": "single_admitted_lane"},
        ),
    )
    assert set(edge.kind for edge in graph.edges) <= EDGE_KINDS
    assert graph.content_id.startswith("b")


def test_unknown_kind_and_authority_inference_fail() -> None:
    with pytest.raises(CompositeGraphError, match="unknown"):
        CompositeGraph.compose(
            ("a", "b"),
            ({"kind": "authority", "source": "a", "target": "b", "meaning": "owns"},),
        )
    with pytest.raises(CompositeGraphError, match="authority"):
        CompositeGraph.compose(
            ("a", "b"),
            ({"kind": "goal", "source": "a", "target": "b", "meaning": "grants authority"},),
        )


def test_effect_and_merge_cannot_share_identity() -> None:
    with pytest.raises(CompositeGraphError, match="conflate"):
        CompositeGraph.compose(
            ("a", "b"),
            (
                {"kind": "effect", "source": "a", "target": "b", "meaning": "same"},
                {"kind": "merge", "source": "a", "target": "b", "meaning": "same"},
            ),
        )


def test_missing_node_and_self_edge_fail() -> None:
    with pytest.raises(CompositeGraphError, match="missing"):
        CompositeGraph.compose(
            ("a",),
            ({"kind": "task", "source": "a", "target": "missing", "meaning": "dep"},),
        )
    with pytest.raises(CompositeGraphError, match="self-edges"):
        CompositeGraph.compose(
            ("a",),
            ({"kind": "task", "source": "a", "target": "a", "meaning": "loop"},),
        )
