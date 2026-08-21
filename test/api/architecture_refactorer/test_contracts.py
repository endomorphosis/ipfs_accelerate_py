"""Hermetic PCAR-002 closed-vocabulary contract tests."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.contracts import (
    ARCHITECTURE_IR_SCHEMA,
    ArchitectureContractError,
    Confidence,
    EdgeKind,
    NodeKind,
    SourceSpan,
)


def test_closed_contracts() -> None:
    assert ARCHITECTURE_IR_SCHEMA.endswith("architecture-ir@1")
    assert {kind.value for kind in NodeKind} >= {
        "module",
        "class",
        "function",
        "schema",
        "operation",
        "test",
        "proof",
        "store",
        "entrypoint",
    }
    assert {kind.value for kind in EdgeKind} >= {
        "imports",
        "calls",
        "contains",
        "tests",
        "proves",
        "writes",
        "reads",
        "effects",
    }
    assert {item.value for item in Confidence} == {
        "exact",
        "conservative",
        "heuristic",
        "opaque",
    }


def test_source_span_rejects_unknown_and_invalid() -> None:
    span = SourceSpan.from_mapping(
        {"path": "ipfs_accelerate_py/agent_supervisor/x.py", "start_line": 1, "end_line": 4}
    )
    assert span.to_dict()["end_line"] == 4
    with pytest.raises(ArchitectureContractError, match="unknown ArchitectureIR field"):
        SourceSpan.from_mapping(
            {
                "path": "ipfs_accelerate_py/agent_supervisor/x.py",
                "start_line": 1,
                "end_line": 4,
                "extra": True,
            }
        )
    with pytest.raises(ArchitectureContractError):
        SourceSpan(path="x.py", start_line=0, end_line=1)
    with pytest.raises(ValueError):
        NodeKind("not-a-node")
