"""Hermetic PCAR-002 ArchitectureIR identity and round-trip tests."""

from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.architecture_ir import (
    ArchitectureEdge,
    ArchitectureIR,
    ArchitectureIRError,
    ArchitectureNode,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.contracts import (
    ARCHITECTURE_IR_SCHEMA,
    ARCHITECTURE_IR_VERSION,
    Confidence,
    EdgeKind,
    NodeKind,
    SourceFactIdentity,
    SourceSpan,
)
from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

_TREE = "a698da9e4b54e2929adacb613bc61ba3e72eed58"
_FRESHNESS = "pcar-002-round-trip"
_EXTRACTOR = "pcar-002-fixture"
_SPAN_PATH = "ipfs_accelerate_py/agent_supervisor/architecture_refactorer/architecture_ir.py"


def _span(start: int, end: int) -> SourceSpan:
    return SourceSpan(_SPAN_PATH, start, end)


def _fact(*, start: int = 1, end: int = 20, confidence: Confidence = Confidence.EXACT) -> SourceFactIdentity:
    return SourceFactIdentity(
        extractor_identity=_EXTRACTOR,
        span=_span(start, end),
        confidence=confidence,
        freshness=_FRESHNESS,
        repository_tree=_TREE,
    )


def _node(node_id: str, kind: NodeKind, *, start: int, end: int) -> ArchitectureNode:
    return ArchitectureNode(node_id=node_id, kind=kind, provenance=_fact(start=start, end=end))


def _edge(
    edge_id: str,
    kind: EdgeKind,
    source: str,
    target: str,
    *,
    start: int,
    end: int,
) -> ArchitectureEdge:
    return ArchitectureEdge(
        edge_id=edge_id,
        kind=kind,
        source=source,
        target=target,
        provenance=_fact(start=start, end=end),
    )


def _sample() -> ArchitectureIR:
    module = _node("n-module", NodeKind.MODULE, start=1, end=20)
    symbol = _node("n-symbol", NodeKind.SYMBOL, start=21, end=40)
    edge = _edge("e-contains", EdgeKind.CONTAINS, "n-module", "n-symbol", start=21, end=21)
    return ArchitectureIR.from_parts(
        repository_tree=_TREE,
        freshness=_FRESHNESS,
        nodes=(symbol, module),
        edges=(edge,),
    )


def test_deterministic_round_trip() -> None:
    graph = _sample()
    payload = graph.to_dict()
    restored = ArchitectureIR.from_mapping(payload)
    assert restored == graph
    assert restored.to_dict() == payload
    assert restored.to_json() == json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert ArchitectureIR.from_json(restored.to_json()) == graph
    assert restored.nodes[0].node_id == "n-module"
    assert restored.nodes[1].node_id == "n-symbol"
    assert restored.schema == ARCHITECTURE_IR_SCHEMA
    assert restored.version == ARCHITECTURE_IR_VERSION
    assert restored.nodes[0].provenance.extractor_identity == _EXTRACTOR
    assert restored.edges[0].provenance.repository_tree == _TREE
    assert restored.edges[0].provenance.freshness == _FRESHNESS


def test_canonical_identity() -> None:
    graph = _sample()
    payload = graph.to_dict()
    claimed = payload.pop("content_identity")
    validate_cid(claimed, codecs=("dag-json",))
    assert claimed == cid_for_dag_json(payload)
    assert claimed == graph.content_identity
    assert not claimed.startswith("sha256:")
    reversed_graph = ArchitectureIR.from_parts(
        repository_tree=_TREE,
        freshness=_FRESHNESS,
        nodes=tuple(reversed(graph.nodes)),
        edges=tuple(reversed(graph.edges)),
    )
    assert reversed_graph.content_identity == graph.content_identity
    assert reversed_graph.to_dict() == graph.to_dict()


def test_unknown_field_rejection() -> None:
    payload = _sample().to_dict()
    payload["unexpected"] = True
    with pytest.raises(ArchitectureIRError, match="unknown ArchitectureIR field"):
        ArchitectureIR.from_mapping(payload)
    node_payload = _sample().to_dict()
    node_payload["nodes"][0]["hidden"] = True
    with pytest.raises(ArchitectureIRError, match="unknown ArchitectureIR field"):
        ArchitectureIR.from_mapping(node_payload)
    edge_payload = _sample().to_dict()
    edge_payload["edges"][0]["hidden"] = True
    with pytest.raises(ArchitectureIRError, match="unknown ArchitectureIR field"):
        ArchitectureIR.from_mapping(edge_payload)
    span_payload = _sample().to_dict()
    span_payload["nodes"][0]["provenance"]["span"]["column"] = 3
    with pytest.raises(ArchitectureIRError, match="unknown ArchitectureIR field"):
        ArchitectureIR.from_mapping(span_payload)


def test_canonical_identity_mismatch_and_unknown_edge() -> None:
    payload = _sample().to_dict()
    payload["content_identity"] = "sha256:" + ("00" * 32)
    with pytest.raises(ArchitectureIRError, match="content identity mismatch"):
        ArchitectureIR.from_mapping(payload)
    forged = dict(_sample().to_dict())
    identity_payload = {key: value for key, value in forged.items() if key != "content_identity"}
    forged["content_identity"] = cid_for_dag_json({**identity_payload, "freshness": "other"})
    with pytest.raises(ArchitectureIRError, match="content identity mismatch"):
        ArchitectureIR.from_mapping(forged)
    with pytest.raises(ValueError):
        EdgeKind("not-an-edge")
    with pytest.raises(ArchitectureIRError, match="unsupported ArchitectureIR edge kind"):
        ArchitectureEdge(
            edge_id="e-bad",
            kind="not-an-edge",
            source="n-module",
            target="n-symbol",
            provenance=_fact(),
        )


def test_graph_rejects_duplicate_ids_dangling_edges_and_tree_mismatch() -> None:
    module = _node("n-module", NodeKind.MODULE, start=1, end=4)
    other = _node("n-module", NodeKind.SYMBOL, start=5, end=8)
    with pytest.raises(ArchitectureIRError, match="node ids must be unique"):
        ArchitectureIR.from_parts(
            repository_tree=_TREE,
            freshness=_FRESHNESS,
            nodes=(module, other),
            edges=(),
        )
    symbol = _node("n-symbol", NodeKind.SYMBOL, start=5, end=8)
    duplicate_edges = (
        _edge("e-dup", EdgeKind.CONTAINS, "n-module", "n-symbol", start=5, end=5),
        _edge("e-dup", EdgeKind.CALLS, "n-module", "n-symbol", start=6, end=6),
    )
    with pytest.raises(ArchitectureIRError, match="edge ids must be unique"):
        ArchitectureIR.from_parts(
            repository_tree=_TREE,
            freshness=_FRESHNESS,
            nodes=(module, symbol),
            edges=duplicate_edges,
        )
    with pytest.raises(ArchitectureIRError, match="unknown node"):
        ArchitectureIR.from_parts(
            repository_tree=_TREE,
            freshness=_FRESHNESS,
            nodes=(module, symbol),
            edges=(_edge("e-missing", EdgeKind.IMPORTS, "n-module", "n-absent", start=1, end=1),),
        )
    mismatched = ArchitectureNode(
        node_id="n-other-tree",
        kind=NodeKind.MODULE,
        provenance=SourceFactIdentity(
            extractor_identity=_EXTRACTOR,
            span=_span(1, 2),
            confidence=Confidence.EXACT,
            freshness=_FRESHNESS,
            repository_tree="deadbeef" * 5,
        ),
    )
    with pytest.raises(ArchitectureIRError, match="repository_tree must match"):
        ArchitectureIR.from_parts(
            repository_tree=_TREE,
            freshness=_FRESHNESS,
            nodes=(mismatched,),
            edges=(),
        )


def test_all_closed_node_and_edge_kinds_round_trip() -> None:
    nodes = tuple(
        _node(f"n-{kind.value}", kind, start=index + 1, end=index + 1)
        for index, kind in enumerate(NodeKind)
    )
    kinds = tuple(NodeKind)
    edges = tuple(
        _edge(
            f"e-{edge_kind.value}",
            edge_kind,
            nodes[index % len(nodes)].node_id,
            nodes[(index + 1) % len(nodes)].node_id,
            start=index + 1,
            end=index + 1,
        )
        for index, edge_kind in enumerate(EdgeKind)
    )
    graph = ArchitectureIR.from_parts(
        repository_tree=_TREE,
        freshness=_FRESHNESS,
        nodes=tuple(reversed(nodes)),
        edges=tuple(reversed(edges)),
    )
    restored = ArchitectureIR.from_mapping(graph.to_dict())
    assert restored == graph
    assert {node.kind for node in restored.nodes} == set(kinds)
    assert {edge.kind for edge in restored.edges} == set(EdgeKind)
    payload = graph.to_dict()
    claimed = payload.pop("content_identity")
    assert claimed == cid_for_dag_json(payload)
    for node in restored.nodes:
        node_payload = dict(node.to_dict())
        node_claimed = node_payload.pop("content_identity")
        assert node_claimed == cid_for_dag_json(node_payload)
    for edge in restored.edges:
        edge_payload = dict(edge.to_dict())
        edge_claimed = edge_payload.pop("content_identity")
        assert edge_claimed == cid_for_dag_json(edge_payload)


def test_versioned_schema_is_closed() -> None:
    payload = _sample().to_dict()
    payload["schema"] = payload["schema"] + "-extra"
    identity_payload = {key: value for key, value in payload.items() if key != "content_identity"}
    payload["content_identity"] = cid_for_dag_json(identity_payload)
    with pytest.raises(ArchitectureIRError, match="unexpected ArchitectureIR schema"):
        ArchitectureIR.from_mapping(payload)
    versioned = _sample().to_dict()
    versioned["version"] = 2
    identity_payload = {key: value for key, value in versioned.items() if key != "content_identity"}
    versioned["content_identity"] = cid_for_dag_json(identity_payload)
    with pytest.raises(ArchitectureIRError, match="unexpected ArchitectureIR version"):
        ArchitectureIR.from_mapping(versioned)
    with pytest.raises(ArchitectureIRError, match="missing ArchitectureIR field"):
        ArchitectureIR.from_mapping(
            {key: value for key, value in _sample().to_dict().items() if key != "freshness"}
        )


def test_heuristic_and_opaque_facts_are_representable() -> None:
    node = ArchitectureNode(
        node_id="n-opaque",
        kind=NodeKind.PROVIDER,
        provenance=_fact(confidence=Confidence.OPAQUE),
    )
    edge = ArchitectureEdge(
        edge_id="e-heuristic",
        kind=EdgeKind.FALLBACKS_TO,
        source="n-opaque",
        target="n-opaque",
        provenance=_fact(confidence=Confidence.HEURISTIC),
    )
    graph = ArchitectureIR.from_parts(
        repository_tree=_TREE,
        freshness=_FRESHNESS,
        nodes=(node,),
        edges=(edge,),
    )
    restored = ArchitectureIR.from_json(graph.to_json())
    assert restored.nodes[0].provenance.confidence is Confidence.OPAQUE
    assert restored.edges[0].provenance.confidence is Confidence.HEURISTIC
