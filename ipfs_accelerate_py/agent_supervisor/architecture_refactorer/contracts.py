"""Closed ArchitectureIR vocabulary and source-span contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

ARCHITECTURE_IR_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/architecture-ir@1"
)
ARCHITECTURE_IR_VERSION = 1

_UNKNOWN_FIELD_MESSAGE = "unknown ArchitectureIR field"


class ArchitectureContractError(ValueError):
    """Fail-closed contract violation for ArchitectureIR vocabulary."""


class NodeKind(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    SCHEMA = "schema"
    OPERATION = "operation"
    EFFECT = "effect"
    TEST = "test"
    PROOF = "proof"
    STORE = "store"
    ENTRYPOINT = "entrypoint"


class EdgeKind(str, Enum):
    IMPORTS = "imports"
    CALLS = "calls"
    CONTAINS = "contains"
    IMPLEMENTS = "implements"
    TESTS = "tests"
    PROVES = "proves"
    WRITES = "writes"
    READS = "reads"
    EFFECTS = "effects"


class Confidence(str, Enum):
    EXACT = "exact"
    CONSERVATIVE = "conservative"
    HEURISTIC = "heuristic"
    OPAQUE = "opaque"


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str]) -> None:
    extra = sorted(set(payload) - allowed)
    if extra:
        raise ArchitectureContractError(f"{_UNKNOWN_FIELD_MESSAGE}: {extra}")


@dataclass(frozen=True)
class SourceSpan:
    """Exact inclusive source span bound to one repository path."""

    path: str
    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path:
            raise ArchitectureContractError("source span path must be a nonempty string")
        if type(self.start_line) is not int or type(self.end_line) is not int:
            raise ArchitectureContractError("source span lines must be integers")
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ArchitectureContractError("source span lines must be a closed interval")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SourceSpan":
        _reject_unknown(payload, {"path", "start_line", "end_line"})
        return cls(
            path=payload["path"],
            start_line=payload["start_line"],
            end_line=payload["end_line"],
        )
