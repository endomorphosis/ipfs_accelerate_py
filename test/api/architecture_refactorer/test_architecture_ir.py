"""Hermetic PCAR-002 ArchitectureIR identity and round-trip tests."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.architecture_ir import (
    ArchitectureEdge,
    ArchitectureIR,
    ArchitectureIRError,
    ArchitectureNode,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.contracts import (
    Confidence,
    EdgeKind,
    NodeKind,
    SourceSpan,
)


def _sample() -> ArchitectureIR:
    module = ArchitectureNode(
        node_id="n-module",
        kind=NodeKind.MODULE,
        span=SourceSpan("ipfs_accelerate_py/agent_supervisor/architecture_refactorer/architecture_ir.py", 1, 20),
        confidence=Confidence.EXACT,
        provenance="pcar-002-fixture",
        content_identity="sha256:" + ("ab" * 32),
    )
    function = ArchitectureNode(
        node_id="n-function",
        kind=NodeKind.FUNCTION,
        span=SourceSpan("ipfs_accelerate_py/agent_supervisor/architecture_refactorer/architecture_ir.py", 21, 40),
        confidence=Confidence.EXACT,
        provenance="pcar-002-fixture",
        content_identity="sha256:" + ("cd" * 32),
    )
    edge = ArchitectureEdge(
        edge_id="e-contains",
        kind=EdgeKind.CONTAINS,
        source="n-module",
        target="n-function",
        confidence=Confidence.EXACT,
        provenance="pcar-002-fixture",
    )
    return ArchitectureIR.from_parts(
        repository_tree="HEAD^{tree}",
        freshness="pcar-002-round-trip",
        nodes=(function, module),
        edges=(edge,),
    )


def test_deterministic_round_trip() -> None:
    graph = _sample()
    payload = graph.to_dict()
    restored = ArchitectureIR.from_mapping(payload)
    assert restored == graph
    assert restored.to_dict() == payload
    assert restored.content_identity.startswith("sha256:")
    assert restored.nodes[0].node_id == "n-function"
    assert restored.nodes[1].node_id == "n-module"


def test_unknown_field_rejection() -> None:
    payload = _sample().to_dict()
    payload["unexpected"] = True
    with pytest.raises(ArchitectureIRError, match="unknown ArchitectureIR field"):
        ArchitectureIR.from_mapping(payload)


def test_canonical_identity_mismatch_and_unknown_edge() -> None:
    payload = _sample().to_dict()
    payload["content_identity"] = "sha256:" + ("00" * 32)
    with pytest.raises(ArchitectureIRError, match="content identity mismatch"):
        ArchitectureIR.from_mapping(payload)
    with pytest.raises(ValueError):
        EdgeKind("not-an-edge")
