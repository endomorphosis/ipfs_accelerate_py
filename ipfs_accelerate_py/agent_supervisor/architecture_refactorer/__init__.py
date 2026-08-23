"""Proof-carrying architecture refactorer contracts and ArchitectureIR."""

from .architecture_ir import ArchitectureEdge, ArchitectureIR, ArchitectureIRError, ArchitectureNode
from .contracts import (
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

__all__ = [
    "ARCHITECTURE_IR_EVIDENCE",
    "ARCHITECTURE_IR_SCHEMA",
    "ARCHITECTURE_IR_VERSION",
    "ArchitectureContractError",
    "ArchitectureEdge",
    "ArchitectureIR",
    "ArchitectureIRError",
    "ArchitectureNode",
    "CLOSED_CONFIDENCE",
    "CLOSED_EDGE_KINDS",
    "CLOSED_NODE_KINDS",
    "Confidence",
    "EdgeKind",
    "NON_PROBATIVE_CONFIDENCE",
    "NodeKind",
    "SourceFactIdentity",
    "SourceSpan",
]
