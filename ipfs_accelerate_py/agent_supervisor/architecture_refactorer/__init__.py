"""Proof-carrying architecture refactorer contracts and ArchitectureIR."""

from .architecture_ir import ArchitectureIR, ArchitectureIRError
from .contracts import (
    ARCHITECTURE_IR_SCHEMA,
    Confidence,
    EdgeKind,
    NodeKind,
    SourceSpan,
)

__all__ = [
    "ARCHITECTURE_IR_SCHEMA",
    "ArchitectureIR",
    "ArchitectureIRError",
    "Confidence",
    "EdgeKind",
    "NodeKind",
    "SourceSpan",
]
