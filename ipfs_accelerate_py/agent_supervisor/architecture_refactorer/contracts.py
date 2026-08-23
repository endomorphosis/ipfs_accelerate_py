"""Closed ArchitectureIR vocabulary, source-span, and fact-identity contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, TypeVar

ARCHITECTURE_IR_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/architecture-ir@1"
)
ARCHITECTURE_IR_VERSION = 1
ARCHITECTURE_IR_EVIDENCE = "pcar/architecture-ir@1"

_UNKNOWN_FIELD_MESSAGE = "unknown ArchitectureIR field"
_MISSING_FIELD_MESSAGE = "missing ArchitectureIR field"

_EnumT = TypeVar("_EnumT", bound=Enum)


class ArchitectureContractError(ValueError):
    """Fail-closed contract violation for ArchitectureIR vocabulary."""


class NodeKind(str, Enum):
    """Closed ArchitectureIR node vocabulary (PCAR-PLAN-R1)."""

    REPOSITORY = "repository"
    PACKAGE = "package"
    MODULE = "module"
    FILE = "file"
    SYMBOL = "symbol"
    INTERFACE = "interface"
    SCHEMA = "schema"
    OPERATION = "operation"
    EFFECT = "effect"
    AUTHORITY = "authority"
    POLICY = "policy"
    STATE = "state"
    RECEIPT = "receipt"
    TEST = "test"
    PROOF = "proof"
    PROVIDER = "provider"
    ENTRYPOINT = "entrypoint"
    ARTIFACT = "artifact"
    COMPATIBILITY = "compatibility"
    SIMULATION = "simulation"
    GENERATED = "generated"


class EdgeKind(str, Enum):
    """Closed ArchitectureIR edge vocabulary (PCAR-PLAN-R1)."""

    CONTAINS = "contains"
    IMPORTS = "imports"
    CALLS = "calls"
    CONSTRUCTS = "constructs"
    READS = "reads"
    WRITES = "writes"
    MUTATES = "mutates"
    AUTHORIZES = "authorizes"
    EVALUATES_POLICY = "evaluates_policy"
    CONFIRMS = "confirms"
    EXECUTES = "executes"
    OBSERVES = "observes"
    PERSISTS = "persists"
    SERIALIZES = "serializes"
    DESERIALIZES = "deserializes"
    GENERATES = "generates"
    TESTS = "tests"
    PROVES = "proves"
    INVALIDATES = "invalidates"
    IMPLEMENTS = "implements"
    ADAPTS = "adapts"
    REEXPORTS = "reexports"
    DUPLICATES = "duplicates"
    SHADOWS = "shadows"
    SUPERSEDES = "supersedes"
    DEPRECATES = "deprecates"
    FALLBACKS_TO = "fallbacks_to"


class Confidence(str, Enum):
    """Closed ArchitectureIR confidence vocabulary."""

    EXACT = "exact"
    CONSERVATIVE = "conservative"
    HEURISTIC = "heuristic"
    OPAQUE = "opaque"


CLOSED_NODE_KINDS: frozenset[str] = frozenset(kind.value for kind in NodeKind)
CLOSED_EDGE_KINDS: frozenset[str] = frozenset(kind.value for kind in EdgeKind)
CLOSED_CONFIDENCE: frozenset[str] = frozenset(item.value for item in Confidence)
NON_PROBATIVE_CONFIDENCE: frozenset[Confidence] = frozenset(
    {Confidence.HEURISTIC, Confidence.OPAQUE}
)

_SOURCE_SPAN_FIELDS = frozenset({"path", "start_line", "end_line"})
_SOURCE_FACT_FIELDS = frozenset(
    {
        "extractor_identity",
        "span",
        "confidence",
        "freshness",
        "repository_tree",
    }
)


def _reject_unknown(
    payload: Mapping[str, Any],
    allowed: Iterable[str],
    *,
    error_type: type[ArchitectureContractError] = ArchitectureContractError,
) -> None:
    allowed_fields = set(allowed)
    extra = sorted(set(payload) - allowed_fields)
    if extra:
        raise error_type(f"{_UNKNOWN_FIELD_MESSAGE}: {extra}")


def _require_mapping(
    payload: Any,
    *,
    error_type: type[ArchitectureContractError] = ArchitectureContractError,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or isinstance(payload, (str, bytes, bytearray)):
        raise error_type("ArchitectureIR payload must be an object")
    return payload


def _require_exact_fields(
    payload: Mapping[str, Any],
    allowed: Iterable[str],
    *,
    error_type: type[ArchitectureContractError] = ArchitectureContractError,
) -> None:
    allowed_fields = set(allowed)
    _reject_unknown(payload, allowed_fields, error_type=error_type)
    missing = sorted(allowed_fields - set(payload))
    if missing:
        raise error_type(f"{_MISSING_FIELD_MESSAGE}: {missing}")


def _require_text(
    value: Any,
    name: str,
    *,
    error_type: type[ArchitectureContractError] = ArchitectureContractError,
) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise error_type(f"{name} must be a nonempty string")
    return value


def _require_int(
    value: Any,
    name: str,
    *,
    error_type: type[ArchitectureContractError] = ArchitectureContractError,
) -> int:
    if type(value) is not int:
        raise error_type(f"{name} must be an integer")
    return value


def _closed_enum(
    value: Any,
    enum_type: type[_EnumT],
    name: str,
    *,
    error_type: type[ArchitectureContractError] = ArchitectureContractError,
) -> _EnumT:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise error_type(f"unsupported ArchitectureIR {name}: {value!r}") from exc


def _repository_relative_path(
    value: Any,
    name: str,
    *,
    error_type: type[ArchitectureContractError] = ArchitectureContractError,
) -> str:
    text = _require_text(value, name, error_type=error_type).replace("\\", "/")
    parts = tuple(part for part in text.split("/") if part not in ("", "."))
    if not parts or text.startswith("/") or any(part == ".." for part in parts):
        raise error_type(f"{name} must be a repository-relative path")
    return "/".join(parts)


@dataclass(frozen=True)
class SourceSpan:
    """Exact inclusive source span bound to one repository path."""

    path: str
    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path",
            _repository_relative_path(self.path, "source span path"),
        )
        start_line = _require_int(self.start_line, "source span start_line")
        end_line = _require_int(self.end_line, "source span end_line")
        if start_line < 1 or end_line < start_line:
            raise ArchitectureContractError("source span lines must be a closed interval")
        object.__setattr__(self, "start_line", start_line)
        object.__setattr__(self, "end_line", end_line)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SourceSpan":
        mapping = _require_mapping(payload)
        _require_exact_fields(mapping, _SOURCE_SPAN_FIELDS)
        return cls(
            path=mapping["path"],
            start_line=mapping["start_line"],
            end_line=mapping["end_line"],
        )

    from_dict = from_mapping


@dataclass(frozen=True)
class SourceFactIdentity:
    """Provenance carried by every ArchitectureIR node and edge fact."""

    extractor_identity: str
    span: SourceSpan
    confidence: Confidence
    freshness: str
    repository_tree: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "extractor_identity",
            _require_text(self.extractor_identity, "extractor_identity"),
        )
        span = self.span if isinstance(self.span, SourceSpan) else SourceSpan.from_mapping(self.span)
        object.__setattr__(self, "span", span)
        object.__setattr__(
            self,
            "confidence",
            _closed_enum(self.confidence, Confidence, "confidence"),
        )
        object.__setattr__(self, "freshness", _require_text(self.freshness, "freshness"))
        object.__setattr__(
            self,
            "repository_tree",
            _require_text(self.repository_tree, "repository_tree"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "extractor_identity": self.extractor_identity,
            "span": self.span.to_dict(),
            "confidence": self.confidence.value,
            "freshness": self.freshness,
            "repository_tree": self.repository_tree,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SourceFactIdentity":
        mapping = _require_mapping(payload)
        _require_exact_fields(mapping, _SOURCE_FACT_FIELDS)
        span_payload = mapping["span"]
        if not isinstance(span_payload, Mapping):
            raise ArchitectureContractError("source fact span must be an object")
        return cls(
            extractor_identity=mapping["extractor_identity"],
            span=SourceSpan.from_mapping(span_payload),
            confidence=mapping["confidence"],
            freshness=mapping["freshness"],
            repository_tree=mapping["repository_tree"],
        )

    from_dict = from_mapping
