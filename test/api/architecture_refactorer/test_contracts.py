"""Hermetic PCAR-002 closed-vocabulary contract tests."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.contracts import (
    ARCHITECTURE_IR_EVIDENCE,
    ARCHITECTURE_IR_SCHEMA,
    ARCHITECTURE_IR_VERSION,
    CLOSED_CONFIDENCE,
    CLOSED_EDGE_KINDS,
    CLOSED_NODE_KINDS,
    NON_PROBATIVE_CONFIDENCE,
    ArchitectureContractError,
    Confidence,
    EdgeKind,
    NodeKind,
    SourceFactIdentity,
    SourceSpan,
)


def test_closed_contracts() -> None:
    assert ARCHITECTURE_IR_SCHEMA == "ipfs_accelerate_py/agent-supervisor/architecture-ir@1"
    assert ARCHITECTURE_IR_SCHEMA.endswith("architecture-ir@1")
    assert ARCHITECTURE_IR_VERSION == 1
    assert ARCHITECTURE_IR_EVIDENCE == "pcar/architecture-ir@1"
    assert {kind.value for kind in NodeKind} == {
        "repository",
        "package",
        "module",
        "file",
        "symbol",
        "interface",
        "schema",
        "operation",
        "effect",
        "authority",
        "policy",
        "state",
        "receipt",
        "test",
        "proof",
        "provider",
        "entrypoint",
        "artifact",
        "compatibility",
        "simulation",
        "generated",
    }
    assert CLOSED_NODE_KINDS == {kind.value for kind in NodeKind}
    assert {kind.value for kind in EdgeKind} == {
        "contains",
        "imports",
        "calls",
        "constructs",
        "reads",
        "writes",
        "mutates",
        "authorizes",
        "evaluates_policy",
        "confirms",
        "executes",
        "observes",
        "persists",
        "serializes",
        "deserializes",
        "generates",
        "tests",
        "proves",
        "invalidates",
        "implements",
        "adapts",
        "reexports",
        "duplicates",
        "shadows",
        "supersedes",
        "deprecates",
        "fallbacks_to",
    }
    assert CLOSED_EDGE_KINDS == {kind.value for kind in EdgeKind}
    assert {item.value for item in Confidence} == {
        "exact",
        "conservative",
        "heuristic",
        "opaque",
    }
    assert CLOSED_CONFIDENCE == {item.value for item in Confidence}
    assert NON_PROBATIVE_CONFIDENCE == {Confidence.HEURISTIC, Confidence.OPAQUE}


def test_source_span_rejects_unknown_and_invalid() -> None:
    span = SourceSpan.from_mapping(
        {"path": "ipfs_accelerate_py/agent_supervisor/x.py", "start_line": 1, "end_line": 4}
    )
    assert span.to_dict() == {
        "path": "ipfs_accelerate_py/agent_supervisor/x.py",
        "start_line": 1,
        "end_line": 4,
    }
    assert SourceSpan.from_mapping(span.to_dict()) == span
    with pytest.raises(ArchitectureContractError, match="unknown ArchitectureIR field"):
        SourceSpan.from_mapping(
            {
                "path": "ipfs_accelerate_py/agent_supervisor/x.py",
                "start_line": 1,
                "end_line": 4,
                "extra": True,
            }
        )
    with pytest.raises(ArchitectureContractError, match="missing ArchitectureIR field"):
        SourceSpan.from_mapping({"path": "x.py", "start_line": 1})
    with pytest.raises(ArchitectureContractError):
        SourceSpan(path="x.py", start_line=0, end_line=1)
    with pytest.raises(ArchitectureContractError):
        SourceSpan(path="x.py", start_line=4, end_line=1)
    with pytest.raises(ArchitectureContractError):
        SourceSpan(path="/abs/x.py", start_line=1, end_line=1)
    with pytest.raises(ArchitectureContractError):
        SourceSpan(path="../escape.py", start_line=1, end_line=1)
    with pytest.raises(ArchitectureContractError):
        SourceSpan(path="x.py", start_line=True, end_line=2)
    with pytest.raises(ValueError):
        NodeKind("not-a-node")
    with pytest.raises(ValueError):
        EdgeKind("not-an-edge")
    with pytest.raises(ValueError):
        Confidence("guess")


def test_source_fact_identity_round_trip_and_unknown_fields() -> None:
    fact = SourceFactIdentity.from_mapping(
        {
            "extractor_identity": "pcar-002-extractor",
            "span": {
                "path": "ipfs_accelerate_py/agent_supervisor/architecture_refactorer/contracts.py",
                "start_line": 1,
                "end_line": 8,
            },
            "confidence": "exact",
            "freshness": "pcar-002-round-trip",
            "repository_tree": "a698da9e4b54e2929adacb613bc61ba3e72eed58",
        }
    )
    assert fact.to_dict()["extractor_identity"] == "pcar-002-extractor"
    assert SourceFactIdentity.from_mapping(fact.to_dict()) == fact
    with pytest.raises(ArchitectureContractError, match="unknown ArchitectureIR field"):
        SourceFactIdentity.from_mapping({**fact.to_dict(), "hidden": 1})
    with pytest.raises(ArchitectureContractError):
        SourceFactIdentity.from_mapping(
            {
                **fact.to_dict(),
                "confidence": "probable",
            }
        )
