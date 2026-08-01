"""Bounded value-provenance compiler: reaching defs, dominance, path conditions.

RPR-033: compile fail-closed intraprocedural data-flow facts that distinguish a
merely assignable name from a value that exists on every relevant path and
carries the required information safely.

Proved only for supported AST / control-flow shapes.  Branch-local absence,
alias ambiguity, loops beyond the unroll bound, exceptions, concurrency,
reflection / native calls, and incomplete interprocedural routes remain
explicit :class:`UnknownReason` facts — never invented edges.

Exact authority roots and producer identity bind every emission so stale
graphs cannot be reused across tree/config/toolchain drift.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Protocol, runtime_checkable

from ..program_graph import Completeness, ProgramGraphRoots


# ---------------------------------------------------------------------------
# Schemas / bounds / producer
# ---------------------------------------------------------------------------

VALUE_PROVENANCE_GRAPH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/value-provenance-graph@1"
)
VALUE_PROVENANCE_GRAPH_VERSION: Final[str] = "value-provenance-graph@1"
REACHING_DEFINITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/reaching-definition@1"
)
DOMINANCE_FACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dominance-fact@1"
)
PATH_CONDITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/path-condition@1"
)
INFORMATION_PROVENANCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/information-provenance@1"
)
DEF_USE_CHAIN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/def-use-chain@1"
)
PRODUCER_ID: Final[str] = "value-provenance-graph@1"

DEFAULT_MAX_BLOCKS: Final[int] = 4_096
DEFAULT_MAX_DEFS: Final[int] = 16_384
DEFAULT_MAX_USES: Final[int] = 32_768
DEFAULT_MAX_EDGES: Final[int] = 65_536
DEFAULT_MAX_LOOP_UNROLL: Final[int] = 2
DEFAULT_MAX_INTERPROCEDURAL_DEPTH: Final[int] = 4
DEFAULT_MAX_SOURCE_BYTES: Final[int] = 1_048_576
DEFAULT_MAX_TEXT_BYTES: Final[int] = 4_096
DEFAULT_MAX_REFS: Final[int] = 256
DEFAULT_MAX_PROCEDURES: Final[int] = 1_024

_CONFIG_CALL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "getenv",
        "get_env",
        "get_config",
        "load_config",
        "read_config",
        "config",
        "settings",
        "feature_flag",
        "get_setting",
        "environ",
        "os.environ",
        "os.getenv",
    }
)
_DI_CALL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "inject",
        "provide",
        "resolve",
        "get_service",
        "get_instance",
        "container.get",
        "registry.get",
        "lookup",
        "wire",
        "autowire",
        "depends",
        "Depends",
    }
)
_CONSTRUCTOR_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "__init__",
        "create",
        "make",
        "build",
        "from_dict",
        "from_json",
        "from_config",
        "from_row",
        "parse",
        "construct",
        "factory",
        "builder",
        "new",
    }
)
_CONVERSION_NAMES: Final[frozenset[str]] = frozenset(
    {
        "int",
        "str",
        "float",
        "bool",
        "bytes",
        "list",
        "dict",
        "set",
        "tuple",
        "frozenset",
        "complex",
        "bytes",
        "bytearray",
        "ord",
        "chr",
        "hex",
        "oct",
        "bin",
        "json.loads",
        "json.dumps",
        "ast.literal_eval",
    }
)
_REFLECTION_NAMES: Final[frozenset[str]] = frozenset(
    {
        "getattr",
        "setattr",
        "hasattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "eval",
        "exec",
        "compile",
        "__import__",
        "importlib.import_module",
        "type",
        "isinstance",  # isinstance is a guard, not reflection when used as test
    }
)
_NATIVE_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "ctypes",
        "cffi",
        "cdll",
        "pydll",
        "windll",
        "pythonapi",
        "ctypes.cdll",
        "ctypes.windll",
        "ctypes.pydll",
        "ctypes.cfunctype",
    }
)
_NATIVE_EXACT: Final[frozenset[str]] = frozenset(
    {
        "cdll",
        "pydll",
        "windll",
        "cdll.loadlibrary",
        "ffi",
    }
)
_CONCURRENCY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "threading",
        "multiprocessing",
        "asyncio",
        "concurrent",
        "Thread",
        "Process",
        "create_task",
        "ensure_future",
        "run_in_executor",
        "Lock",
        "RLock",
        "Semaphore",
        "Queue",
    }
)
_EFFECT_MARKERS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "open": "io",
        "print": "stdout",
        "write": "io",
        "read": "io",
        "send": "network",
        "recv": "network",
        "connect": "network",
        "execute": "db",
        "query": "db",
        "commit": "db",
        "os.system": "process",
        "subprocess": "process",
        "os.remove": "fs_mutate",
        "os.rename": "fs_mutate",
        "shutil": "fs_mutate",
    }
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ValueProvenanceError(ValueError):
    """Malformed provenance input or invariant failure."""


class ValueProvenanceBoundsError(ValueProvenanceError):
    """A provenance record exceeded its deterministic compactness bound."""


class ValueProvenanceAuthorityError(ValueProvenanceError):
    """Roots, producer identity, or reuse boundary was violated."""


class ValueProvenanceUnsupportedError(ValueProvenanceError):
    """Requested analysis shape is outside the proved fragment."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class DefinitionKind(str, Enum):
    """Closed vocabulary of value definition producers."""

    PARAMETER = "parameter"
    ASSIGNMENT = "assignment"
    ANN_ASSIGN = "ann_assign"
    AUG_ASSIGN = "aug_assign"
    FIELD_WRITE = "field_write"
    RETURN = "return"
    CONSTRUCTOR = "constructor"
    CONVERSION = "conversion"
    CONFIG_SOURCE = "config_source"
    DI_SOURCE = "di_source"
    ALIAS = "alias"
    IMPORT = "import"
    CONSTANT = "constant"
    CALL_RESULT = "call_result"
    COMPREHENSION = "comprehension"
    EXCEPTION_BIND = "exception_bind"
    UNKNOWN = "unknown"


class UseKind(str, Enum):
    """Closed vocabulary of value uses."""

    LOAD = "load"
    CALL_ARG = "call_arg"
    RETURN = "return"
    FIELD_READ = "field_read"
    CONDITION = "condition"
    CONVERSION = "conversion"
    INTERPROCEDURAL_THREAD = "interprocedural_thread"
    STORE_RHS = "store_rhs"
    AUG_OPERAND = "aug_operand"
    UNKNOWN = "unknown"


class DominanceKind(str, Enum):
    DOMINATES = "dominates"
    POST_DOMINATES = "post_dominates"
    STRICTLY_DOMINATES = "strictly_dominates"
    STRICTLY_POST_DOMINATES = "strictly_post_dominates"
    IMMEDIATE_DOMINATOR = "immediate_dominator"
    IMMEDIATE_POST_DOMINATOR = "immediate_post_dominator"


class ProvenanceStatus(str, Enum):
    """Whether a fact is proved, partial, or intentionally unknown."""

    PROVED = "proved"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


class UnknownReason(str, Enum):
    """Closed reasons a fact remains unknown (fail-closed)."""

    BRANCH_LOCAL_ABSENCE = "branch_local_absence"
    ALIAS_AMBIGUITY = "alias_ambiguity"
    LOOP_BEYOND_BOUNDS = "loop_beyond_bounds"
    EXCEPTION_PATH = "exception_path"
    CONCURRENCY = "concurrency"
    REFLECTION = "reflection"
    NATIVE_CALL = "native_call"
    INCOMPLETE_INTERPROCEDURAL = "incomplete_interprocedural"
    UNSUPPORTED_CFG = "unsupported_cfg"
    UNSUPPORTED_AST = "unsupported_ast"
    MULTIPLE_REACHING = "multiple_reaching"
    DYNAMIC_TARGET = "dynamic_target"
    MISSING_CALLEE = "missing_callee"
    STALE_ROOTS = "stale_roots"


class DependencyDirection(str, Enum):
    """Closed dependency / flow direction labels on provenance edges."""

    DEFINES = "defines"
    USES = "uses"
    FLOWS_TO = "flows_to"
    THREADS_TO = "threads_to"
    ALIASES = "aliases"
    DOMINATES = "dominates"
    POST_DOMINATES = "post_dominates"
    GUARDS = "guards"
    REFINES = "refines"
    CONSTRUCTS = "constructs"
    CONVERTS = "converts"
    CONFIGURES = "configures"
    INJECTS = "injects"
    RETURNS = "returns"
    FIELD_OF = "field_of"
    PARAMETER_OF = "parameter_of"


class InformationOriginKind(str, Enum):
    """Where information content is claimed to originate."""

    PARAMETER = "parameter"
    LOCAL = "local"
    FIELD = "field"
    CONSTRUCTOR = "constructor"
    CONVERSION = "conversion"
    CONFIG = "config"
    DI_REGISTRY = "di_registry"
    REQUEST_CONTEXT = "request_context"
    CONSTANT = "constant"
    RETURN_VALUE = "return_value"
    ALIAS = "alias"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


class Nullability(str, Enum):
    NONNULL = "nonnull"
    NULLABLE = "nullable"
    UNKNOWN = "unknown"


class MutationKind(str, Enum):
    IMMUTABLE = "immutable"
    MUTABLE = "mutable"
    UNKNOWN = "unknown"


class OwnershipKind(str, Enum):
    OWNED = "owned"
    BORROWED = "borrowed"
    SHARED = "shared"
    TRANSFERRED = "transferred"
    UNKNOWN = "unknown"


class ConcurrencyKind(str, Enum):
    SEQUENTIAL = "sequential"
    SHARED_MUTABLE = "shared_mutable"
    MESSAGE_PASSING = "message_passing"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


class InterproceduralCompleteness(str, Enum):
    """Explicit completeness of an interprocedural threading fact."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"
    FRONTIER = "frontier"


class CfgShapeSupport(str, Enum):
    """Whether a CFG shape is inside the proved fragment."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _identity(namespace: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"{namespace}:sha256:{digest}"


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    limit: int = DEFAULT_MAX_TEXT_BYTES,
) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)
    text = text.strip()
    if required and not text:
        raise ValueProvenanceError(f"{name} must not be empty")
    if len(text.encode("utf-8")) > limit:
        raise ValueProvenanceBoundsError(f"{name} exceeds its byte bound")
    return text


def _enum(value: Any, enum: type[Enum], name: str) -> Enum:
    if isinstance(value, enum):
        return value
    try:
        return enum(str(value))
    except (TypeError, ValueError) as exc:
        choices = ", ".join(member.value for member in enum)
        raise ValueProvenanceError(f"{name} must be one of: {choices}") from exc


def _string_tuple(
    values: Any,
    name: str,
    *,
    limit: int = DEFAULT_MAX_REFS,
    required: bool = False,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, str):
        raw = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        raw = values
    else:
        raise ValueProvenanceError(f"{name} must be a sequence of strings")
    if len(raw) > limit:
        raise ValueProvenanceBoundsError(f"{name} exceeds its item bound")
    result = tuple(sorted({_text(item, name, required=False) for item in raw if str(item or "").strip()}))
    if required and not result:
        raise ValueProvenanceError(f"{name} must not be empty")
    return result


def _expr_name(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _expr_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _expr_name(node.func)
    if isinstance(node, ast.Subscript):
        return _expr_name(node.value)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return repr(node.value)
        return repr(node.value)
    return ""


def _simple_name(reference: str) -> str:
    text = str(reference or "").strip()
    if not text:
        return ""
    return text.rsplit(".", 1)[-1]


def _is_constructor_call(func_name: str) -> bool:
    simple = _simple_name(func_name)
    if not simple:
        return False
    # Concurrency / native constructors are classified separately.
    if _is_concurrency_call(func_name) or _is_native_call(func_name):
        return False
    if simple[:1].isupper() and simple.isidentifier():
        return True
    return simple in _CONSTRUCTOR_MARKERS or simple.startswith("from_")


def _is_conversion_call(func_name: str) -> bool:
    return func_name in _CONVERSION_NAMES or _simple_name(func_name) in _CONVERSION_NAMES


def _is_config_call(func_name: str) -> bool:
    simple = _simple_name(func_name)
    return (
        func_name in _CONFIG_CALL_NAMES
        or simple in _CONFIG_CALL_NAMES
        or "environ" in func_name
        or func_name.endswith(".getenv")
    )


def _is_di_call(func_name: str) -> bool:
    simple = _simple_name(func_name)
    return func_name in _DI_CALL_NAMES or simple in _DI_CALL_NAMES


def _is_reflection_call(func_name: str) -> bool:
    simple = _simple_name(func_name)
    # isinstance used as a type guard is supported; bare reflection builtins are not.
    if simple == "isinstance":
        return False
    return func_name in _REFLECTION_NAMES or simple in _REFLECTION_NAMES


def _is_native_call(func_name: str) -> bool:
    lower = func_name.lower()
    simple = _simple_name(func_name).lower()
    if simple in {"cdll", "pydll", "windll"} or lower in _NATIVE_EXACT:
        return True
    if lower.startswith("ctypes.") or "ctypes." in lower:
        return True
    if lower.startswith("cffi.") or simple == "ffi":
        return True
    return any(marker == lower or lower.endswith("." + marker) for marker in _NATIVE_MARKERS)


def _is_concurrency_call(func_name: str) -> bool:
    simple = _simple_name(func_name)
    return (
        simple in _CONCURRENCY_MARKERS
        or any(marker in func_name for marker in _CONCURRENCY_MARKERS)
    )


def _effects_for_call(func_name: str) -> tuple[str, ...]:
    hits: set[str] = set()
    simple = _simple_name(func_name)
    for key, effect in _EFFECT_MARKERS.items():
        if key == func_name or key == simple or key in func_name:
            hits.add(effect)
    return tuple(sorted(hits))


def _annotation_text(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return _expr_name(node) or type(node).__name__


def _lineno(node: ast.AST) -> int:
    return int(getattr(node, "lineno", 0) or 0)


def _end_lineno(node: ast.AST) -> int:
    return int(getattr(node, "end_lineno", 0) or _lineno(node) or 0)


# ---------------------------------------------------------------------------
# Core fact records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceLocation:
    """Relative path + line span for one definition or use."""

    path: str
    line_start: int = 0
    line_end: int = 0
    column_start: int = 0
    column_end: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _text(self.path, "path", required=False) or "<unknown>")
        object.__setattr__(self, "line_start", max(0, int(self.line_start)))
        object.__setattr__(self, "line_end", max(0, int(self.line_end)))
        object.__setattr__(self, "column_start", max(0, int(self.column_start)))
        object.__setattr__(self, "column_end", max(0, int(self.column_end)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "column_start": self.column_start,
            "column_end": self.column_end,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceLocation":
        return cls(
            path=str(payload.get("path") or ""),
            line_start=int(payload.get("line_start") or 0),
            line_end=int(payload.get("line_end") or 0),
            column_start=int(payload.get("column_start") or 0),
            column_end=int(payload.get("column_end") or 0),
        )

    @classmethod
    def from_ast(cls, path: str, node: ast.AST) -> "SourceLocation":
        return cls(
            path=path,
            line_start=_lineno(node),
            line_end=_end_lineno(node),
            column_start=int(getattr(node, "col_offset", 0) or 0),
            column_end=int(getattr(node, "end_col_offset", 0) or 0),
        )


@dataclass(frozen=True)
class CfgBlock:
    """One basic block in the intraprocedural CFG."""

    block_id: str
    procedure_id: str
    label: str
    statements: tuple[str, ...] = ()
    successors: tuple[str, ...] = ()
    predecessors: tuple[str, ...] = ()
    shape_support: CfgShapeSupport = CfgShapeSupport.SUPPORTED
    unknown_reasons: tuple[str, ...] = ()
    is_entry: bool = False
    is_exit: bool = False
    is_join: bool = False
    loop_header: bool = False
    exception_handler: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_id", _text(self.block_id, "block_id"))
        object.__setattr__(self, "procedure_id", _text(self.procedure_id, "procedure_id"))
        object.__setattr__(self, "label", _text(self.label, "label", required=False) or self.block_id)
        object.__setattr__(
            self,
            "statements",
            tuple(str(item) for item in self.statements[:DEFAULT_MAX_REFS]),
        )
        object.__setattr__(self, "successors", _string_tuple(self.successors, "successors"))
        object.__setattr__(
            self, "predecessors", _string_tuple(self.predecessors, "predecessors")
        )
        object.__setattr__(
            self, "shape_support", _enum(self.shape_support, CfgShapeSupport, "shape_support")
        )
        object.__setattr__(
            self, "unknown_reasons", _string_tuple(self.unknown_reasons, "unknown_reasons")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "procedure_id": self.procedure_id,
            "label": self.label,
            "statements": list(self.statements),
            "successors": list(self.successors),
            "predecessors": list(self.predecessors),
            "shape_support": self.shape_support.value,
            "unknown_reasons": list(self.unknown_reasons),
            "is_entry": self.is_entry,
            "is_exit": self.is_exit,
            "is_join": self.is_join,
            "loop_header": self.loop_header,
            "exception_handler": self.exception_handler,
        }


@dataclass(frozen=True)
class ReachingDefinition:
    """One definition that may reach a program point."""

    def_id: str
    variable: str
    kind: DefinitionKind
    block_id: str
    procedure_id: str
    location: SourceLocation
    producer_id: str = PRODUCER_ID
    roots_id: str = ""
    expression_ref: str = ""
    type_annotation: str = ""
    status: ProvenanceStatus = ProvenanceStatus.PROVED
    unknown_reasons: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    field_path: str = ""
    source_kind_detail: str = ""
    interprocedural_completeness: InterproceduralCompleteness = (
        InterproceduralCompleteness.COMPLETE
    )
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "def_id", _text(self.def_id, "def_id"))
        object.__setattr__(self, "variable", _text(self.variable, "variable"))
        object.__setattr__(self, "kind", _enum(self.kind, DefinitionKind, "kind"))
        object.__setattr__(self, "block_id", _text(self.block_id, "block_id"))
        object.__setattr__(self, "procedure_id", _text(self.procedure_id, "procedure_id"))
        if not isinstance(self.location, SourceLocation):
            raise ValueProvenanceError("location must be SourceLocation")
        object.__setattr__(self, "producer_id", _text(self.producer_id, "producer_id"))
        object.__setattr__(
            self, "roots_id", _text(self.roots_id, "roots_id", required=False)
        )
        object.__setattr__(
            self,
            "expression_ref",
            _text(self.expression_ref, "expression_ref", required=False),
        )
        object.__setattr__(
            self,
            "type_annotation",
            _text(self.type_annotation, "type_annotation", required=False),
        )
        object.__setattr__(self, "status", _enum(self.status, ProvenanceStatus, "status"))
        object.__setattr__(
            self, "unknown_reasons", _string_tuple(self.unknown_reasons, "unknown_reasons")
        )
        object.__setattr__(self, "aliases", _string_tuple(self.aliases, "aliases"))
        object.__setattr__(
            self, "field_path", _text(self.field_path, "field_path", required=False)
        )
        object.__setattr__(
            self,
            "source_kind_detail",
            _text(self.source_kind_detail, "source_kind_detail", required=False),
        )
        object.__setattr__(
            self,
            "interprocedural_completeness",
            _enum(
                self.interprocedural_completeness,
                InterproceduralCompleteness,
                "interprocedural_completeness",
            ),
        )
        attrs = self.attributes or {}
        if not isinstance(attrs, Mapping):
            raise ValueProvenanceError("attributes must be a mapping")
        object.__setattr__(
            self,
            "attributes",
            MappingProxyType({str(k): attrs[k] for k in sorted(attrs, key=str)}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REACHING_DEFINITION_SCHEMA,
            "def_id": self.def_id,
            "variable": self.variable,
            "kind": self.kind.value,
            "block_id": self.block_id,
            "procedure_id": self.procedure_id,
            "location": self.location.to_dict(),
            "producer_id": self.producer_id,
            "roots_id": self.roots_id,
            "expression_ref": self.expression_ref,
            "type_annotation": self.type_annotation,
            "status": self.status.value,
            "unknown_reasons": list(self.unknown_reasons),
            "aliases": list(self.aliases),
            "field_path": self.field_path,
            "source_kind_detail": self.source_kind_detail,
            "interprocedural_completeness": self.interprocedural_completeness.value,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReachingDefinition":
        return cls(
            def_id=str(payload.get("def_id") or ""),
            variable=str(payload.get("variable") or ""),
            kind=payload.get("kind", DefinitionKind.UNKNOWN),
            block_id=str(payload.get("block_id") or ""),
            procedure_id=str(payload.get("procedure_id") or ""),
            location=SourceLocation.from_dict(payload.get("location") or {}),
            producer_id=str(payload.get("producer_id") or PRODUCER_ID),
            roots_id=str(payload.get("roots_id") or ""),
            expression_ref=str(payload.get("expression_ref") or ""),
            type_annotation=str(payload.get("type_annotation") or ""),
            status=payload.get("status", ProvenanceStatus.PROVED),
            unknown_reasons=tuple(payload.get("unknown_reasons") or ()),
            aliases=tuple(payload.get("aliases") or ()),
            field_path=str(payload.get("field_path") or ""),
            source_kind_detail=str(payload.get("source_kind_detail") or ""),
            interprocedural_completeness=payload.get(
                "interprocedural_completeness", InterproceduralCompleteness.COMPLETE
            ),
            attributes=payload.get("attributes") or {},
        )


@dataclass(frozen=True)
class ValueUse:
    """One use of a name at a program point."""

    use_id: str
    variable: str
    kind: UseKind
    block_id: str
    procedure_id: str
    location: SourceLocation
    expression_ref: str = ""
    status: ProvenanceStatus = ProvenanceStatus.PROVED
    unknown_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "use_id", _text(self.use_id, "use_id"))
        object.__setattr__(self, "variable", _text(self.variable, "variable"))
        object.__setattr__(self, "kind", _enum(self.kind, UseKind, "kind"))
        object.__setattr__(self, "block_id", _text(self.block_id, "block_id"))
        object.__setattr__(self, "procedure_id", _text(self.procedure_id, "procedure_id"))
        if not isinstance(self.location, SourceLocation):
            raise ValueProvenanceError("location must be SourceLocation")
        object.__setattr__(
            self,
            "expression_ref",
            _text(self.expression_ref, "expression_ref", required=False),
        )
        object.__setattr__(self, "status", _enum(self.status, ProvenanceStatus, "status"))
        object.__setattr__(
            self, "unknown_reasons", _string_tuple(self.unknown_reasons, "unknown_reasons")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_id": self.use_id,
            "variable": self.variable,
            "kind": self.kind.value,
            "block_id": self.block_id,
            "procedure_id": self.procedure_id,
            "location": self.location.to_dict(),
            "expression_ref": self.expression_ref,
            "status": self.status.value,
            "unknown_reasons": list(self.unknown_reasons),
        }


@dataclass(frozen=True)
class DefUseChain:
    """One definition-use edge with reaching confidence."""

    chain_id: str
    def_id: str
    use_id: str
    variable: str
    status: ProvenanceStatus = ProvenanceStatus.PROVED
    unknown_reasons: tuple[str, ...] = ()
    path_condition_id: str = ""
    dependency_direction: DependencyDirection = DependencyDirection.FLOWS_TO

    def __post_init__(self) -> None:
        object.__setattr__(self, "chain_id", _text(self.chain_id, "chain_id"))
        object.__setattr__(self, "def_id", _text(self.def_id, "def_id"))
        object.__setattr__(self, "use_id", _text(self.use_id, "use_id"))
        object.__setattr__(self, "variable", _text(self.variable, "variable"))
        object.__setattr__(self, "status", _enum(self.status, ProvenanceStatus, "status"))
        object.__setattr__(
            self, "unknown_reasons", _string_tuple(self.unknown_reasons, "unknown_reasons")
        )
        object.__setattr__(
            self,
            "path_condition_id",
            _text(self.path_condition_id, "path_condition_id", required=False),
        )
        object.__setattr__(
            self,
            "dependency_direction",
            _enum(self.dependency_direction, DependencyDirection, "dependency_direction"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DEF_USE_CHAIN_SCHEMA,
            "chain_id": self.chain_id,
            "def_id": self.def_id,
            "use_id": self.use_id,
            "variable": self.variable,
            "status": self.status.value,
            "unknown_reasons": list(self.unknown_reasons),
            "path_condition_id": self.path_condition_id,
            "dependency_direction": self.dependency_direction.value,
        }


@dataclass(frozen=True)
class DominanceFact:
    """One dominance or post-dominance relation between CFG blocks."""

    fact_id: str
    kind: DominanceKind
    dominator_block_id: str
    dominated_block_id: str
    procedure_id: str
    status: ProvenanceStatus = ProvenanceStatus.PROVED
    unknown_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact_id", _text(self.fact_id, "fact_id"))
        object.__setattr__(self, "kind", _enum(self.kind, DominanceKind, "kind"))
        object.__setattr__(
            self, "dominator_block_id", _text(self.dominator_block_id, "dominator_block_id")
        )
        object.__setattr__(
            self, "dominated_block_id", _text(self.dominated_block_id, "dominated_block_id")
        )
        object.__setattr__(self, "procedure_id", _text(self.procedure_id, "procedure_id"))
        object.__setattr__(self, "status", _enum(self.status, ProvenanceStatus, "status"))
        object.__setattr__(
            self, "unknown_reasons", _string_tuple(self.unknown_reasons, "unknown_reasons")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOMINANCE_FACT_SCHEMA,
            "fact_id": self.fact_id,
            "kind": self.kind.value,
            "dominator_block_id": self.dominator_block_id,
            "dominated_block_id": self.dominated_block_id,
            "procedure_id": self.procedure_id,
            "status": self.status.value,
            "unknown_reasons": list(self.unknown_reasons),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DominanceFact":
        return cls(
            fact_id=str(payload.get("fact_id") or ""),
            kind=payload.get("kind", DominanceKind.DOMINATES),
            dominator_block_id=str(payload.get("dominator_block_id") or ""),
            dominated_block_id=str(payload.get("dominated_block_id") or ""),
            procedure_id=str(payload.get("procedure_id") or ""),
            status=payload.get("status", ProvenanceStatus.PROVED),
            unknown_reasons=tuple(payload.get("unknown_reasons") or ()),
        )


@dataclass(frozen=True)
class PathCondition:
    """Predicate that holds on a control-flow edge or block entry."""

    condition_id: str
    procedure_id: str
    block_id: str
    predicate_ref: str
    polarity: bool = True
    guard_variable: str = ""
    type_refinement: str = ""
    status: ProvenanceStatus = ProvenanceStatus.PROVED
    unknown_reasons: tuple[str, ...] = ()
    predecessor_block_id: str = ""
    branch_label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition_id", _text(self.condition_id, "condition_id"))
        object.__setattr__(self, "procedure_id", _text(self.procedure_id, "procedure_id"))
        object.__setattr__(self, "block_id", _text(self.block_id, "block_id"))
        object.__setattr__(
            self, "predicate_ref", _text(self.predicate_ref, "predicate_ref", required=False)
        )
        if not isinstance(self.polarity, bool):
            raise ValueProvenanceError("polarity must be a boolean")
        object.__setattr__(
            self, "guard_variable", _text(self.guard_variable, "guard_variable", required=False)
        )
        object.__setattr__(
            self,
            "type_refinement",
            _text(self.type_refinement, "type_refinement", required=False),
        )
        object.__setattr__(self, "status", _enum(self.status, ProvenanceStatus, "status"))
        object.__setattr__(
            self, "unknown_reasons", _string_tuple(self.unknown_reasons, "unknown_reasons")
        )
        object.__setattr__(
            self,
            "predecessor_block_id",
            _text(self.predecessor_block_id, "predecessor_block_id", required=False),
        )
        object.__setattr__(
            self, "branch_label", _text(self.branch_label, "branch_label", required=False)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PATH_CONDITION_SCHEMA,
            "condition_id": self.condition_id,
            "procedure_id": self.procedure_id,
            "block_id": self.block_id,
            "predicate_ref": self.predicate_ref,
            "polarity": self.polarity,
            "guard_variable": self.guard_variable,
            "type_refinement": self.type_refinement,
            "status": self.status.value,
            "unknown_reasons": list(self.unknown_reasons),
            "predecessor_block_id": self.predecessor_block_id,
            "branch_label": self.branch_label,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PathCondition":
        return cls(
            condition_id=str(payload.get("condition_id") or ""),
            procedure_id=str(payload.get("procedure_id") or ""),
            block_id=str(payload.get("block_id") or ""),
            predicate_ref=str(payload.get("predicate_ref") or ""),
            polarity=bool(payload.get("polarity", True)),
            guard_variable=str(payload.get("guard_variable") or ""),
            type_refinement=str(payload.get("type_refinement") or ""),
            status=payload.get("status", ProvenanceStatus.PROVED),
            unknown_reasons=tuple(payload.get("unknown_reasons") or ()),
            predecessor_block_id=str(payload.get("predecessor_block_id") or ""),
            branch_label=str(payload.get("branch_label") or ""),
        )


@dataclass(frozen=True)
class TypeRefinement:
    """Type / nullability refinement introduced by a guard."""

    refinement_id: str
    variable: str
    refined_type: str
    nullability: Nullability
    path_condition_id: str
    procedure_id: str
    status: ProvenanceStatus = ProvenanceStatus.PROVED

    def __post_init__(self) -> None:
        object.__setattr__(self, "refinement_id", _text(self.refinement_id, "refinement_id"))
        object.__setattr__(self, "variable", _text(self.variable, "variable"))
        object.__setattr__(
            self, "refined_type", _text(self.refined_type, "refined_type", required=False)
        )
        object.__setattr__(
            self, "nullability", _enum(self.nullability, Nullability, "nullability")
        )
        object.__setattr__(
            self, "path_condition_id", _text(self.path_condition_id, "path_condition_id")
        )
        object.__setattr__(self, "procedure_id", _text(self.procedure_id, "procedure_id"))
        object.__setattr__(self, "status", _enum(self.status, ProvenanceStatus, "status"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "refinement_id": self.refinement_id,
            "variable": self.variable,
            "refined_type": self.refined_type,
            "nullability": self.nullability.value,
            "path_condition_id": self.path_condition_id,
            "procedure_id": self.procedure_id,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class InformationProvenance:
    """Information-origin and safety facets attached to one definition."""

    provenance_id: str
    def_id: str
    variable: str
    origin_kind: InformationOriginKind
    type_ref: str = ""
    schema_ref: str = ""
    range_ref: str = ""
    nullability: Nullability = Nullability.UNKNOWN
    origin_labels: tuple[str, ...] = ()
    effect_refs: tuple[str, ...] = ()
    capability_refs: tuple[str, ...] = ()
    authorization_refs: tuple[str, ...] = ()
    ownership: OwnershipKind = OwnershipKind.UNKNOWN
    lifetime_ref: str = ""
    mutation: MutationKind = MutationKind.UNKNOWN
    concurrency: ConcurrencyKind = ConcurrencyKind.SEQUENTIAL
    dependency_direction: DependencyDirection = DependencyDirection.DEFINES
    memory_safety_facet_ref: str = ""
    status: ProvenanceStatus = ProvenanceStatus.PROVED
    unknown_reasons: tuple[str, ...] = ()
    producer_id: str = PRODUCER_ID
    roots_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance_id", _text(self.provenance_id, "provenance_id"))
        object.__setattr__(self, "def_id", _text(self.def_id, "def_id"))
        object.__setattr__(self, "variable", _text(self.variable, "variable"))
        object.__setattr__(
            self, "origin_kind", _enum(self.origin_kind, InformationOriginKind, "origin_kind")
        )
        object.__setattr__(self, "type_ref", _text(self.type_ref, "type_ref", required=False))
        object.__setattr__(
            self, "schema_ref", _text(self.schema_ref, "schema_ref", required=False)
        )
        object.__setattr__(
            self, "range_ref", _text(self.range_ref, "range_ref", required=False)
        )
        object.__setattr__(
            self, "nullability", _enum(self.nullability, Nullability, "nullability")
        )
        object.__setattr__(
            self, "origin_labels", _string_tuple(self.origin_labels, "origin_labels")
        )
        object.__setattr__(self, "effect_refs", _string_tuple(self.effect_refs, "effect_refs"))
        object.__setattr__(
            self, "capability_refs", _string_tuple(self.capability_refs, "capability_refs")
        )
        object.__setattr__(
            self,
            "authorization_refs",
            _string_tuple(self.authorization_refs, "authorization_refs"),
        )
        object.__setattr__(
            self, "ownership", _enum(self.ownership, OwnershipKind, "ownership")
        )
        object.__setattr__(
            self, "lifetime_ref", _text(self.lifetime_ref, "lifetime_ref", required=False)
        )
        object.__setattr__(
            self, "mutation", _enum(self.mutation, MutationKind, "mutation")
        )
        object.__setattr__(
            self, "concurrency", _enum(self.concurrency, ConcurrencyKind, "concurrency")
        )
        object.__setattr__(
            self,
            "dependency_direction",
            _enum(self.dependency_direction, DependencyDirection, "dependency_direction"),
        )
        object.__setattr__(
            self,
            "memory_safety_facet_ref",
            _text(self.memory_safety_facet_ref, "memory_safety_facet_ref", required=False),
        )
        object.__setattr__(self, "status", _enum(self.status, ProvenanceStatus, "status"))
        object.__setattr__(
            self, "unknown_reasons", _string_tuple(self.unknown_reasons, "unknown_reasons")
        )
        object.__setattr__(self, "producer_id", _text(self.producer_id, "producer_id"))
        object.__setattr__(
            self, "roots_id", _text(self.roots_id, "roots_id", required=False)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": INFORMATION_PROVENANCE_SCHEMA,
            "provenance_id": self.provenance_id,
            "def_id": self.def_id,
            "variable": self.variable,
            "origin_kind": self.origin_kind.value,
            "type_ref": self.type_ref,
            "schema_ref": self.schema_ref,
            "range_ref": self.range_ref,
            "nullability": self.nullability.value,
            "origin_labels": list(self.origin_labels),
            "effect_refs": list(self.effect_refs),
            "capability_refs": list(self.capability_refs),
            "authorization_refs": list(self.authorization_refs),
            "ownership": self.ownership.value,
            "lifetime_ref": self.lifetime_ref,
            "mutation": self.mutation.value,
            "concurrency": self.concurrency.value,
            "dependency_direction": self.dependency_direction.value,
            "memory_safety_facet_ref": self.memory_safety_facet_ref,
            "status": self.status.value,
            "unknown_reasons": list(self.unknown_reasons),
            "producer_id": self.producer_id,
            "roots_id": self.roots_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InformationProvenance":
        return cls(
            provenance_id=str(payload.get("provenance_id") or ""),
            def_id=str(payload.get("def_id") or ""),
            variable=str(payload.get("variable") or ""),
            origin_kind=payload.get("origin_kind", InformationOriginKind.UNKNOWN),
            type_ref=str(payload.get("type_ref") or ""),
            schema_ref=str(payload.get("schema_ref") or ""),
            range_ref=str(payload.get("range_ref") or ""),
            nullability=payload.get("nullability", Nullability.UNKNOWN),
            origin_labels=tuple(payload.get("origin_labels") or ()),
            effect_refs=tuple(payload.get("effect_refs") or ()),
            capability_refs=tuple(payload.get("capability_refs") or ()),
            authorization_refs=tuple(payload.get("authorization_refs") or ()),
            ownership=payload.get("ownership", OwnershipKind.UNKNOWN),
            lifetime_ref=str(payload.get("lifetime_ref") or ""),
            mutation=payload.get("mutation", MutationKind.UNKNOWN),
            concurrency=payload.get("concurrency", ConcurrencyKind.SEQUENTIAL),
            dependency_direction=payload.get(
                "dependency_direction", DependencyDirection.DEFINES
            ),
            memory_safety_facet_ref=str(payload.get("memory_safety_facet_ref") or ""),
            status=payload.get("status", ProvenanceStatus.PROVED),
            unknown_reasons=tuple(payload.get("unknown_reasons") or ()),
            producer_id=str(payload.get("producer_id") or PRODUCER_ID),
            roots_id=str(payload.get("roots_id") or ""),
        )


@dataclass(frozen=True)
class InterproceduralThread:
    """Explicit threading of a value across a call boundary."""

    thread_id: str
    source_def_id: str
    source_procedure_id: str
    target_procedure_id: str
    parameter_name: str
    call_site_ref: str
    completeness: InterproceduralCompleteness
    status: ProvenanceStatus = ProvenanceStatus.PARTIAL
    unknown_reasons: tuple[str, ...] = ()
    depth: int = 1
    dependency_direction: DependencyDirection = DependencyDirection.THREADS_TO

    def __post_init__(self) -> None:
        object.__setattr__(self, "thread_id", _text(self.thread_id, "thread_id"))
        object.__setattr__(self, "source_def_id", _text(self.source_def_id, "source_def_id"))
        object.__setattr__(
            self, "source_procedure_id", _text(self.source_procedure_id, "source_procedure_id")
        )
        object.__setattr__(
            self, "target_procedure_id", _text(self.target_procedure_id, "target_procedure_id")
        )
        object.__setattr__(
            self, "parameter_name", _text(self.parameter_name, "parameter_name", required=False)
        )
        object.__setattr__(
            self, "call_site_ref", _text(self.call_site_ref, "call_site_ref", required=False)
        )
        object.__setattr__(
            self,
            "completeness",
            _enum(self.completeness, InterproceduralCompleteness, "completeness"),
        )
        object.__setattr__(self, "status", _enum(self.status, ProvenanceStatus, "status"))
        object.__setattr__(
            self, "unknown_reasons", _string_tuple(self.unknown_reasons, "unknown_reasons")
        )
        depth = int(self.depth)
        if depth < 0 or depth > DEFAULT_MAX_INTERPROCEDURAL_DEPTH * 4:
            raise ValueProvenanceBoundsError("interprocedural depth out of bound")
        object.__setattr__(self, "depth", depth)
        object.__setattr__(
            self,
            "dependency_direction",
            _enum(self.dependency_direction, DependencyDirection, "dependency_direction"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "source_def_id": self.source_def_id,
            "source_procedure_id": self.source_procedure_id,
            "target_procedure_id": self.target_procedure_id,
            "parameter_name": self.parameter_name,
            "call_site_ref": self.call_site_ref,
            "completeness": self.completeness.value,
            "status": self.status.value,
            "unknown_reasons": list(self.unknown_reasons),
            "depth": self.depth,
            "dependency_direction": self.dependency_direction.value,
        }


@dataclass(frozen=True)
class UnknownFrontierFact:
    """Explicit unknown left open by the fail-closed analyzer."""

    fact_id: str
    reason: UnknownReason
    procedure_id: str
    block_id: str = ""
    variable: str = ""
    detail: str = ""
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact_id", _text(self.fact_id, "fact_id"))
        object.__setattr__(self, "reason", _enum(self.reason, UnknownReason, "reason"))
        object.__setattr__(
            self, "procedure_id", _text(self.procedure_id, "procedure_id", required=False)
        )
        object.__setattr__(self, "block_id", _text(self.block_id, "block_id", required=False))
        object.__setattr__(self, "variable", _text(self.variable, "variable", required=False))
        object.__setattr__(self, "detail", _text(self.detail, "detail", required=False))
        object.__setattr__(
            self, "evidence_refs", _string_tuple(self.evidence_refs, "evidence_refs")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "reason": self.reason.value,
            "procedure_id": self.procedure_id,
            "block_id": self.block_id,
            "variable": self.variable,
            "detail": self.detail,
            "evidence_refs": list(self.evidence_refs),
        }


# ---------------------------------------------------------------------------
# Graph aggregate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValueProvenanceGraph:
    """Root- and producer-bound collection of provenance facts for one analysis.

    Identity is content-addressed over roots, producer, procedures, and the
    canonical serialization of every fact.  Reuse across different roots or
    producers is rejected.
    """

    roots: ProgramGraphRoots
    producer_id: str
    procedures: tuple[str, ...]
    blocks: tuple[CfgBlock, ...]
    definitions: tuple[ReachingDefinition, ...]
    uses: tuple[ValueUse, ...]
    def_use_chains: tuple[DefUseChain, ...]
    dominance_facts: tuple[DominanceFact, ...]
    path_conditions: tuple[PathCondition, ...]
    type_refinements: tuple[TypeRefinement, ...]
    information_provenances: tuple[InformationProvenance, ...]
    interprocedural_threads: tuple[InterproceduralThread, ...]
    unknown_frontier: tuple[UnknownFrontierFact, ...]
    completeness: Completeness = Completeness.PARTIAL
    max_loop_unroll: int = DEFAULT_MAX_LOOP_UNROLL
    schema: str = VALUE_PROVENANCE_GRAPH_SCHEMA
    version: str = VALUE_PROVENANCE_GRAPH_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.roots, ProgramGraphRoots):
            raise ValueProvenanceAuthorityError("roots must be ProgramGraphRoots")
        object.__setattr__(self, "producer_id", _text(self.producer_id, "producer_id"))
        if self.producer_id != PRODUCER_ID:
            # Allow only the known producer for this schema version.
            raise ValueProvenanceAuthorityError(
                f"unsupported producer_id: {self.producer_id!r}"
            )
        object.__setattr__(
            self, "procedures", _string_tuple(self.procedures, "procedures", limit=DEFAULT_MAX_PROCEDURES)
        )
        for collection, limit, name in (
            (self.blocks, DEFAULT_MAX_BLOCKS, "blocks"),
            (self.definitions, DEFAULT_MAX_DEFS, "definitions"),
            (self.uses, DEFAULT_MAX_USES, "uses"),
            (self.def_use_chains, DEFAULT_MAX_EDGES, "def_use_chains"),
            (self.dominance_facts, DEFAULT_MAX_EDGES, "dominance_facts"),
            (self.path_conditions, DEFAULT_MAX_EDGES, "path_conditions"),
            (self.type_refinements, DEFAULT_MAX_DEFS, "type_refinements"),
            (self.information_provenances, DEFAULT_MAX_DEFS, "information_provenances"),
            (self.interprocedural_threads, DEFAULT_MAX_EDGES, "interprocedural_threads"),
            (self.unknown_frontier, DEFAULT_MAX_EDGES, "unknown_frontier"),
        ):
            if len(collection) > limit:
                raise ValueProvenanceBoundsError(f"{name} exceeds hard bound {limit}")
        object.__setattr__(
            self, "completeness", _enum(self.completeness, Completeness, "completeness")
        )
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        object.__setattr__(self, "version", _text(self.version, "version"))
        if self.schema != VALUE_PROVENANCE_GRAPH_SCHEMA:
            raise ValueProvenanceError(f"unsupported schema: {self.schema}")
        roots_id = self.roots.roots_id
        fixed_defs: list[ReachingDefinition] = []
        for item in self.definitions:
            if item.producer_id != self.producer_id:
                raise ValueProvenanceAuthorityError(
                    "definition producer_id does not match graph producer"
                )
            if item.roots_id and item.roots_id != roots_id:
                raise ValueProvenanceAuthorityError(
                    "definition roots_id does not match graph roots"
                )
            if item.roots_id == roots_id:
                fixed_defs.append(item)
            else:
                fixed_defs.append(
                    ReachingDefinition(
                        def_id=item.def_id,
                        variable=item.variable,
                        kind=item.kind,
                        block_id=item.block_id,
                        procedure_id=item.procedure_id,
                        location=item.location,
                        producer_id=item.producer_id,
                        roots_id=roots_id,
                        expression_ref=item.expression_ref,
                        type_annotation=item.type_annotation,
                        status=item.status,
                        unknown_reasons=item.unknown_reasons,
                        aliases=item.aliases,
                        field_path=item.field_path,
                        source_kind_detail=item.source_kind_detail,
                        interprocedural_completeness=item.interprocedural_completeness,
                        attributes=dict(item.attributes),
                    )
                )
        object.__setattr__(self, "definitions", tuple(fixed_defs))
        fixed_info: list[InformationProvenance] = []
        for item in self.information_provenances:
            if item.producer_id != self.producer_id:
                raise ValueProvenanceAuthorityError(
                    "information provenance producer_id does not match graph producer"
                )
            if item.roots_id and item.roots_id != roots_id:
                raise ValueProvenanceAuthorityError(
                    "information provenance roots_id does not match graph roots"
                )
            if item.roots_id == roots_id:
                fixed_info.append(item)
            else:
                fixed_info.append(
                    InformationProvenance(
                        provenance_id=item.provenance_id,
                        def_id=item.def_id,
                        variable=item.variable,
                        origin_kind=item.origin_kind,
                        type_ref=item.type_ref,
                        schema_ref=item.schema_ref,
                        range_ref=item.range_ref,
                        nullability=item.nullability,
                        origin_labels=item.origin_labels,
                        effect_refs=item.effect_refs,
                        capability_refs=item.capability_refs,
                        authorization_refs=item.authorization_refs,
                        ownership=item.ownership,
                        lifetime_ref=item.lifetime_ref,
                        mutation=item.mutation,
                        concurrency=item.concurrency,
                        dependency_direction=item.dependency_direction,
                        memory_safety_facet_ref=item.memory_safety_facet_ref,
                        status=item.status,
                        unknown_reasons=item.unknown_reasons,
                        producer_id=item.producer_id,
                        roots_id=roots_id,
                    )
                )
        object.__setattr__(self, "information_provenances", tuple(fixed_info))

    @property
    def roots_id(self) -> str:
        return self.roots.roots_id

    @property
    def graph_id(self) -> str:
        return _identity("value-provenance-graph", self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "roots_id": self.roots.roots_id,
            "producer_id": self.producer_id,
            "procedures": list(self.procedures),
            "blocks": [item.to_dict() for item in self.blocks],
            "definitions": [item.to_dict() for item in self.definitions],
            "uses": [item.to_dict() for item in self.uses],
            "def_use_chains": [item.to_dict() for item in self.def_use_chains],
            "dominance_facts": [item.to_dict() for item in self.dominance_facts],
            "path_conditions": [item.to_dict() for item in self.path_conditions],
            "type_refinements": [item.to_dict() for item in self.type_refinements],
            "information_provenances": [
                item.to_dict() for item in self.information_provenances
            ],
            "interprocedural_threads": [
                item.to_dict() for item in self.interprocedural_threads
            ],
            "unknown_frontier": [item.to_dict() for item in self.unknown_frontier],
            "completeness": self.completeness.value
            if isinstance(self.completeness, Completeness)
            else str(self.completeness),
            "max_loop_unroll": int(self.max_loop_unroll),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["graph_id"] = self.graph_id
        payload["roots"] = self.roots.to_dict()
        return payload

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValueProvenanceGraph":
        roots_payload = payload.get("roots")
        if not isinstance(roots_payload, Mapping):
            raise ValueProvenanceAuthorityError("roots payload is required")
        roots = ProgramGraphRoots.from_dict(roots_payload)
        graph = cls(
            roots=roots,
            producer_id=str(payload.get("producer_id") or PRODUCER_ID),
            procedures=tuple(payload.get("procedures") or ()),
            blocks=tuple(
                CfgBlock(
                    block_id=str(item.get("block_id") or ""),
                    procedure_id=str(item.get("procedure_id") or ""),
                    label=str(item.get("label") or ""),
                    statements=tuple(item.get("statements") or ()),
                    successors=tuple(item.get("successors") or ()),
                    predecessors=tuple(item.get("predecessors") or ()),
                    shape_support=item.get("shape_support", CfgShapeSupport.SUPPORTED),
                    unknown_reasons=tuple(item.get("unknown_reasons") or ()),
                    is_entry=bool(item.get("is_entry")),
                    is_exit=bool(item.get("is_exit")),
                    is_join=bool(item.get("is_join")),
                    loop_header=bool(item.get("loop_header")),
                    exception_handler=bool(item.get("exception_handler")),
                )
                for item in (payload.get("blocks") or ())
                if isinstance(item, Mapping)
            ),
            definitions=tuple(
                ReachingDefinition.from_dict(item)
                for item in (payload.get("definitions") or ())
                if isinstance(item, Mapping)
            ),
            uses=tuple(
                ValueUse(
                    use_id=str(item.get("use_id") or ""),
                    variable=str(item.get("variable") or ""),
                    kind=item.get("kind", UseKind.LOAD),
                    block_id=str(item.get("block_id") or ""),
                    procedure_id=str(item.get("procedure_id") or ""),
                    location=SourceLocation.from_dict(item.get("location") or {}),
                    expression_ref=str(item.get("expression_ref") or ""),
                    status=item.get("status", ProvenanceStatus.PROVED),
                    unknown_reasons=tuple(item.get("unknown_reasons") or ()),
                )
                for item in (payload.get("uses") or ())
                if isinstance(item, Mapping)
            ),
            def_use_chains=tuple(
                DefUseChain(
                    chain_id=str(item.get("chain_id") or ""),
                    def_id=str(item.get("def_id") or ""),
                    use_id=str(item.get("use_id") or ""),
                    variable=str(item.get("variable") or ""),
                    status=item.get("status", ProvenanceStatus.PROVED),
                    unknown_reasons=tuple(item.get("unknown_reasons") or ()),
                    path_condition_id=str(item.get("path_condition_id") or ""),
                    dependency_direction=item.get(
                        "dependency_direction", DependencyDirection.FLOWS_TO
                    ),
                )
                for item in (payload.get("def_use_chains") or ())
                if isinstance(item, Mapping)
            ),
            dominance_facts=tuple(
                DominanceFact.from_dict(item)
                for item in (payload.get("dominance_facts") or ())
                if isinstance(item, Mapping)
            ),
            path_conditions=tuple(
                PathCondition.from_dict(item)
                for item in (payload.get("path_conditions") or ())
                if isinstance(item, Mapping)
            ),
            type_refinements=tuple(
                TypeRefinement(
                    refinement_id=str(item.get("refinement_id") or ""),
                    variable=str(item.get("variable") or ""),
                    refined_type=str(item.get("refined_type") or ""),
                    nullability=item.get("nullability", Nullability.UNKNOWN),
                    path_condition_id=str(item.get("path_condition_id") or ""),
                    procedure_id=str(item.get("procedure_id") or ""),
                    status=item.get("status", ProvenanceStatus.PROVED),
                )
                for item in (payload.get("type_refinements") or ())
                if isinstance(item, Mapping)
            ),
            information_provenances=tuple(
                InformationProvenance.from_dict(item)
                for item in (payload.get("information_provenances") or ())
                if isinstance(item, Mapping)
            ),
            interprocedural_threads=tuple(
                InterproceduralThread(
                    thread_id=str(item.get("thread_id") or ""),
                    source_def_id=str(item.get("source_def_id") or ""),
                    source_procedure_id=str(item.get("source_procedure_id") or ""),
                    target_procedure_id=str(item.get("target_procedure_id") or ""),
                    parameter_name=str(item.get("parameter_name") or ""),
                    call_site_ref=str(item.get("call_site_ref") or ""),
                    completeness=item.get(
                        "completeness", InterproceduralCompleteness.UNKNOWN
                    ),
                    status=item.get("status", ProvenanceStatus.PARTIAL),
                    unknown_reasons=tuple(item.get("unknown_reasons") or ()),
                    depth=int(item.get("depth") or 1),
                    dependency_direction=item.get(
                        "dependency_direction", DependencyDirection.THREADS_TO
                    ),
                )
                for item in (payload.get("interprocedural_threads") or ())
                if isinstance(item, Mapping)
            ),
            unknown_frontier=tuple(
                UnknownFrontierFact(
                    fact_id=str(item.get("fact_id") or ""),
                    reason=item.get("reason", UnknownReason.UNSUPPORTED_AST),
                    procedure_id=str(item.get("procedure_id") or ""),
                    block_id=str(item.get("block_id") or ""),
                    variable=str(item.get("variable") or ""),
                    detail=str(item.get("detail") or ""),
                    evidence_refs=tuple(item.get("evidence_refs") or ()),
                )
                for item in (payload.get("unknown_frontier") or ())
                if isinstance(item, Mapping)
            ),
            completeness=payload.get("completeness", Completeness.PARTIAL),
            max_loop_unroll=int(payload.get("max_loop_unroll") or DEFAULT_MAX_LOOP_UNROLL),
            schema=str(payload.get("schema") or VALUE_PROVENANCE_GRAPH_SCHEMA),
            version=str(payload.get("version") or VALUE_PROVENANCE_GRAPH_VERSION),
        )
        claimed = str(payload.get("graph_id") or "")
        if claimed and claimed != graph.graph_id:
            raise ValueProvenanceAuthorityError(
                "value provenance graph identity does not match payload"
            )
        return graph

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def definitions_for(self, variable: str, *, procedure_id: str = "") -> tuple[ReachingDefinition, ...]:
        variable = _text(variable, "variable")
        return tuple(
            item
            for item in self.definitions
            if item.variable == variable
            and (not procedure_id or item.procedure_id == procedure_id)
        )

    def uses_for(self, variable: str, *, procedure_id: str = "") -> tuple[ValueUse, ...]:
        variable = _text(variable, "variable")
        return tuple(
            item
            for item in self.uses
            if item.variable == variable
            and (not procedure_id or item.procedure_id == procedure_id)
        )

    def reaching_at_use(self, use_id: str) -> tuple[ReachingDefinition, ...]:
        use_id = _text(use_id, "use_id")
        def_ids = {
            chain.def_id for chain in self.def_use_chains if chain.use_id == use_id
        }
        return tuple(item for item in self.definitions if item.def_id in def_ids)

    def dominates(self, a: str, b: str, *, procedure_id: str = "") -> bool:
        """Return True if block *a* dominates block *b* (proved facts only)."""
        for fact in self.dominance_facts:
            if fact.status is not ProvenanceStatus.PROVED:
                continue
            if procedure_id and fact.procedure_id != procedure_id:
                continue
            if fact.kind in {
                DominanceKind.DOMINATES,
                DominanceKind.STRICTLY_DOMINATES,
                DominanceKind.IMMEDIATE_DOMINATOR,
            }:
                if fact.dominator_block_id == a and fact.dominated_block_id == b:
                    return True
                if (
                    fact.kind is DominanceKind.IMMEDIATE_DOMINATOR
                    and fact.dominator_block_id == a
                    and fact.dominated_block_id == b
                ):
                    return True
        # Self-dominance is always true when both blocks exist.
        if a == b and any(block.block_id == a for block in self.blocks):
            return True
        return False

    def post_dominates(self, a: str, b: str, *, procedure_id: str = "") -> bool:
        for fact in self.dominance_facts:
            if fact.status is not ProvenanceStatus.PROVED:
                continue
            if procedure_id and fact.procedure_id != procedure_id:
                continue
            if fact.kind in {
                DominanceKind.POST_DOMINATES,
                DominanceKind.STRICTLY_POST_DOMINATES,
                DominanceKind.IMMEDIATE_POST_DOMINATOR,
            }:
                if fact.dominator_block_id == a and fact.dominated_block_id == b:
                    return True
        if a == b and any(block.block_id == a for block in self.blocks):
            return True
        return False

    def path_conditions_for_block(self, block_id: str) -> tuple[PathCondition, ...]:
        block_id = _text(block_id, "block_id")
        return tuple(item for item in self.path_conditions if item.block_id == block_id)

    def information_for_def(self, def_id: str) -> InformationProvenance | None:
        def_id = _text(def_id, "def_id")
        for item in self.information_provenances:
            if item.def_id == def_id:
                return item
        return None

    def unknown_reasons(self) -> frozenset[UnknownReason]:
        reasons: set[UnknownReason] = set()
        for item in self.unknown_frontier:
            reasons.add(item.reason)
        return frozenset(reasons)

    def available_on_all_paths(
        self,
        variable: str,
        *,
        procedure_id: str,
        use_block_id: str = "",
    ) -> tuple[bool, ProvenanceStatus, tuple[str, ...]]:
        """Check whether *variable* has a single proved definition on every path.

        A reaching definition alone is not enough: the definition must
        *dominate* the use (or be a procedure parameter).  Branch-local
        absence and multiple reaching definitions fail closed.

        Returns ``(available, status, reason_codes)``.
        """
        variable = _text(variable, "variable")
        procedure_id = _text(procedure_id, "procedure_id")
        params = [
            item
            for item in self.definitions
            if item.variable == variable
            and item.procedure_id == procedure_id
            and item.kind is DefinitionKind.PARAMETER
            and item.status is ProvenanceStatus.PROVED
        ]
        uses = [
            item
            for item in self.uses
            if item.variable == variable
            and item.procedure_id == procedure_id
            and (not use_block_id or item.block_id == use_block_id)
        ]
        if not uses:
            if params and not use_block_id:
                return True, ProvenanceStatus.PROVED, ()
            return False, ProvenanceStatus.UNKNOWN, (UnknownReason.BRANCH_LOCAL_ABSENCE.value,)

        reasons: list[str] = []
        for use in uses:
            reaching = self.reaching_at_use(use.use_id)
            proved = [item for item in reaching if item.status is ProvenanceStatus.PROVED]
            if not proved:
                reasons.append(UnknownReason.BRANCH_LOCAL_ABSENCE.value)
                continue
            if len(proved) > 1:
                reasons.append(UnknownReason.MULTIPLE_REACHING.value)
                continue
            sole = proved[0]
            for reason in sole.unknown_reasons:
                reasons.append(reason)
            # Parameters always dominate body uses of the same procedure.
            if sole.kind is DefinitionKind.PARAMETER:
                continue
            if not self.dominates(
                sole.block_id, use.block_id, procedure_id=procedure_id
            ):
                # Defined on some path only (e.g. then-branch without else).
                reasons.append(UnknownReason.BRANCH_LOCAL_ABSENCE.value)
        if reasons:
            unique = tuple(sorted(set(reasons)))
            return False, ProvenanceStatus.UNKNOWN, unique
        return True, ProvenanceStatus.PROVED, ()

    def compatible_with_requirement(
        self,
        requirement: Any,
        *,
        procedure_id: str,
        candidate_variable: str,
    ) -> tuple[bool, ProvenanceStatus, tuple[str, ...]]:
        """Conservative check against a MissingInputRequirement-like object.

        Type assignability alone is never sufficient: information origin,
        availability on paths, and unknown frontiers must also fit.
        """
        candidate_variable = _text(candidate_variable, "candidate_variable")
        procedure_id = _text(procedure_id, "procedure_id")
        available, status, reasons = self.available_on_all_paths(
            candidate_variable, procedure_id=procedure_id
        )
        if not available:
            return False, status, reasons

        type_ref = str(getattr(requirement, "type_ref", "") or "")
        info_ref = str(getattr(requirement, "information_content_ref", "") or "")
        nullability = str(getattr(requirement, "nullability", "") or "")
        parameter_name = str(getattr(requirement, "parameter_name", "") or "")

        defs = self.definitions_for(candidate_variable, procedure_id=procedure_id)
        if not defs:
            return False, ProvenanceStatus.UNKNOWN, (UnknownReason.BRANCH_LOCAL_ABSENCE.value,)

        # Prefer a parameter / proved assignment that matches type_ref when present.
        matching: list[ReachingDefinition] = []
        for item in defs:
            if item.status is not ProvenanceStatus.PROVED:
                continue
            if type_ref and item.type_annotation and item.type_annotation != type_ref:
                # Annotation mismatch is not an automatic refutation when empty.
                if item.type_annotation not in type_ref and type_ref not in item.type_annotation:
                    continue
            matching.append(item)
        if not matching:
            matching = [item for item in defs if item.status is ProvenanceStatus.PROVED]
        if not matching:
            return False, ProvenanceStatus.UNKNOWN, (UnknownReason.BRANCH_LOCAL_ABSENCE.value,)

        for item in matching:
            info = self.information_for_def(item.def_id)
            if info is None:
                continue
            if info_ref and info_ref not in info.origin_labels and info_ref != info.type_ref:
                # Same-typed wrong information fails closed.
                if type_ref and (info.type_ref == type_ref or item.type_annotation == type_ref):
                    return (
                        False,
                        ProvenanceStatus.UNKNOWN,
                        ("wrong_information_content",),
                    )
            if nullability and info.nullability.value != nullability:
                if (
                    nullability == Nullability.NONNULL.value
                    and info.nullability is Nullability.NULLABLE
                ):
                    return False, ProvenanceStatus.UNKNOWN, ("nullability_mismatch",)
            if parameter_name and item.kind is DefinitionKind.PARAMETER:
                if item.variable == parameter_name or candidate_variable == parameter_name:
                    return True, ProvenanceStatus.PROVED, ()
            if info.status is ProvenanceStatus.PROVED:
                return True, ProvenanceStatus.PROVED, ()
        return True, ProvenanceStatus.PARTIAL, ()


# ---------------------------------------------------------------------------
# Optional protocols for interface composition
# ---------------------------------------------------------------------------


@runtime_checkable
class ProgramDependencyGraphLike(Protocol):
    """Minimal façade over ProgramDependencyGraph for optional enrichment."""

    @property
    def roots(self) -> ProgramGraphRoots: ...

    @property
    def graph(self) -> Any: ...


@runtime_checkable
class ProgramCallResolverLike(Protocol):
    def resolve_reference(self, reference: str, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class MissingInputRequirementLike(Protocol):
    parameter_name: str
    type_ref: str
    nullability: str
    information_content_ref: str


@runtime_checkable
class MemorySafetyFacetLike(Protocol):
    @property
    def disposition(self) -> Any: ...


# ---------------------------------------------------------------------------
# Internal mutable CFG builder state
# ---------------------------------------------------------------------------


@dataclass
class _MutableBlock:
    block_id: str
    procedure_id: str
    label: str
    stmts: list[ast.stmt] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)
    predecessors: list[str] = field(default_factory=list)
    shape_support: CfgShapeSupport = CfgShapeSupport.SUPPORTED
    unknown_reasons: list[str] = field(default_factory=list)
    is_entry: bool = False
    is_exit: bool = False
    is_join: bool = False
    loop_header: bool = False
    exception_handler: bool = False


@dataclass
class _ProcIR:
    procedure_id: str
    path: str
    qualname: str
    args: list[ast.arg]
    defaults_offset: int
    body: list[ast.stmt]
    returns_annotation: ast.AST | None
    is_method: bool
    lineno: int


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


class ValueProvenanceCompiler:
    """Compile bounded value-provenance facts from Python sources.

    Parameters
    ----------
    roots:
        Authority roots; drift invalidates every compiled graph.
    max_loop_unroll:
        Structural loop bound.  Beyond this, loop bodies are recorded as
        :attr:`UnknownReason.LOOP_BEYOND_BOUNDS` rather than unrolled further.
    max_interprocedural_depth:
        Depth at which interprocedural threading becomes an explicit frontier.
    dependency_graph / call_resolver:
        Optional enrichment interfaces; never authoritative for proved edges.
    """

    def __init__(
        self,
        roots: ProgramGraphRoots | Mapping[str, Any],
        *,
        max_loop_unroll: int = DEFAULT_MAX_LOOP_UNROLL,
        max_interprocedural_depth: int = DEFAULT_MAX_INTERPROCEDURAL_DEPTH,
        dependency_graph: ProgramDependencyGraphLike | None = None,
        call_resolver: ProgramCallResolverLike | None = None,
    ) -> None:
        if isinstance(roots, Mapping):
            roots = ProgramGraphRoots.from_dict(roots)
        if not isinstance(roots, ProgramGraphRoots):
            raise ValueProvenanceAuthorityError("roots must be ProgramGraphRoots")
        self._roots = roots
        self._max_loop_unroll = max(0, int(max_loop_unroll))
        self._max_ip_depth = max(0, int(max_interprocedural_depth))
        self._dependency_graph = dependency_graph
        self._call_resolver = call_resolver
        if dependency_graph is not None:
            dep_roots = getattr(dependency_graph, "roots", None)
            if dep_roots is not None and getattr(dep_roots, "roots_id", "") not in {
                "",
                roots.roots_id,
            }:
                raise ValueProvenanceAuthorityError(
                    "dependency_graph roots do not match compiler roots"
                )

    @property
    def roots(self) -> ProgramGraphRoots:
        return self._roots

    @property
    def producer_id(self) -> str:
        return PRODUCER_ID

    def compile_sources(
        self,
        files: Mapping[str, str],
        *,
        memory_safety_facets: Mapping[str, MemorySafetyFacetLike] | None = None,
    ) -> ValueProvenanceGraph:
        """Compile provenance for every top-level and nested function in *files*."""
        if not isinstance(files, Mapping) or not files:
            raise ValueProvenanceError("files must be a non-empty path->source mapping")
        procedures: list[_ProcIR] = []
        for path, source in sorted(files.items(), key=lambda item: item[0]):
            path = _text(path, "path")
            source = str(source or "")
            if len(source.encode("utf-8")) > DEFAULT_MAX_SOURCE_BYTES:
                raise ValueProvenanceBoundsError(f"source for {path!r} exceeds hard bound")
            try:
                tree = ast.parse(source, filename=path)
            except SyntaxError as exc:
                raise ValueProvenanceError(f"syntax error in {path}: {exc}") from exc
            procedures.extend(self._extract_procedures(path, tree))
        if len(procedures) > DEFAULT_MAX_PROCEDURES:
            raise ValueProvenanceBoundsError("procedure count exceeds hard bound")
        return self._compile_procedures(
            procedures, memory_safety_facets=memory_safety_facets or {}
        )

    def compile_procedure(
        self,
        source: str,
        *,
        path: str = "snippet.py",
        procedure_name: str = "",
        memory_safety_facets: Mapping[str, MemorySafetyFacetLike] | None = None,
    ) -> ValueProvenanceGraph:
        """Compile a single procedure (by name, or the first function found)."""
        files = {path: source}
        graph = self.compile_sources(files, memory_safety_facets=memory_safety_facets)
        if procedure_name:
            procedure_name = _text(procedure_name, "procedure_name")
            matching = [p for p in graph.procedures if p.endswith(procedure_name) or p == procedure_name]
            if not matching:
                # Also match qualname suffix.
                matching = [
                    p for p in graph.procedures if p.rsplit(".", 1)[-1] == procedure_name
                ]
            if not matching:
                raise ValueProvenanceError(
                    f"procedure {procedure_name!r} not found in {path}"
                )
        return graph

    # ------------------------------------------------------------------
    # Procedure extraction
    # ------------------------------------------------------------------

    def _extract_procedures(self, path: str, tree: ast.AST) -> list[_ProcIR]:
        results: list[_ProcIR] = []

        def walk(nodes: Sequence[ast.stmt], prefix: str) -> None:
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qual = f"{prefix}.{node.name}" if prefix else node.name
                    proc_id = f"{path}::{qual}"
                    results.append(
                        _ProcIR(
                            procedure_id=proc_id,
                            path=path,
                            qualname=qual,
                            args=list(node.args.args) + list(node.args.kwonlyargs),
                            defaults_offset=len(node.args.args) - len(node.args.defaults),
                            body=list(node.body),
                            returns_annotation=node.returns,
                            is_method=bool(prefix),
                            lineno=_lineno(node),
                        )
                    )
                    walk(node.body, qual)
                elif isinstance(node, ast.ClassDef):
                    walk(node.body, f"{prefix}.{node.name}" if prefix else node.name)
                elif isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    for attr in ("body", "orelse", "finalbody"):
                        child = getattr(node, attr, None)
                        if child:
                            walk(child, prefix)
                    for handler in getattr(node, "handlers", ()) or ():
                        walk(handler.body, prefix)

        if isinstance(tree, ast.Module):
            walk(tree.body, "")
        return results

    # ------------------------------------------------------------------
    # Main compilation
    # ------------------------------------------------------------------

    def _compile_procedures(
        self,
        procedures: Sequence[_ProcIR],
        *,
        memory_safety_facets: Mapping[str, MemorySafetyFacetLike],
    ) -> ValueProvenanceGraph:
        all_blocks: list[CfgBlock] = []
        all_defs: list[ReachingDefinition] = []
        all_uses: list[ValueUse] = []
        all_chains: list[DefUseChain] = []
        all_dom: list[DominanceFact] = []
        all_pc: list[PathCondition] = []
        all_ref: list[TypeRefinement] = []
        all_info: list[InformationProvenance] = []
        all_threads: list[InterproceduralThread] = []
        all_unknown: list[UnknownFrontierFact] = []
        proc_ids: list[str] = []
        # Callee signature map for interprocedural threading.
        sig_map: dict[str, _ProcIR] = {}
        for proc in procedures:
            sig_map[proc.qualname] = proc
            sig_map[proc.procedure_id] = proc
            sig_map[_simple_name(proc.qualname)] = proc

        for proc in procedures:
            proc_ids.append(proc.procedure_id)
            (
                blocks,
                definitions,
                uses,
                chains,
                dom,
                pcs,
                refs,
                infos,
                threads,
                unknowns,
            ) = self._compile_one(proc, sig_map=sig_map, facets=memory_safety_facets)
            all_blocks.extend(blocks)
            all_defs.extend(definitions)
            all_uses.extend(uses)
            all_chains.extend(chains)
            all_dom.extend(dom)
            all_pc.extend(pcs)
            all_ref.extend(refs)
            all_info.extend(infos)
            all_threads.extend(threads)
            all_unknown.extend(unknowns)

        completeness = Completeness.COMPLETE
        if all_unknown:
            completeness = Completeness.PARTIAL
        # Unsupported / frontier unknowns force frontier completeness.
        if any(
            item.reason
            in {
                UnknownReason.UNSUPPORTED_CFG,
                UnknownReason.UNSUPPORTED_AST,
                UnknownReason.INCOMPLETE_INTERPROCEDURAL,
                UnknownReason.REFLECTION,
                UnknownReason.NATIVE_CALL,
                UnknownReason.CONCURRENCY,
                UnknownReason.EXCEPTION_PATH,
            }
            for item in all_unknown
        ):
            completeness = Completeness.FRONTIER

        return ValueProvenanceGraph(
            roots=self._roots,
            producer_id=PRODUCER_ID,
            procedures=tuple(sorted(set(proc_ids))),
            blocks=tuple(all_blocks),
            definitions=tuple(all_defs),
            uses=tuple(all_uses),
            def_use_chains=tuple(all_chains),
            dominance_facts=tuple(all_dom),
            path_conditions=tuple(all_pc),
            type_refinements=tuple(all_ref),
            information_provenances=tuple(all_info),
            interprocedural_threads=tuple(all_threads),
            unknown_frontier=tuple(all_unknown),
            completeness=completeness,
            max_loop_unroll=self._max_loop_unroll,
        )

    def _compile_one(
        self,
        proc: _ProcIR,
        *,
        sig_map: Mapping[str, _ProcIR],
        facets: Mapping[str, MemorySafetyFacetLike],
    ) -> tuple[
        list[CfgBlock],
        list[ReachingDefinition],
        list[ValueUse],
        list[DefUseChain],
        list[DominanceFact],
        list[PathCondition],
        list[TypeRefinement],
        list[InformationProvenance],
        list[InterproceduralThread],
        list[UnknownFrontierFact],
    ]:
        blocks_m, edge_labels, unknowns = self._build_cfg(proc)
        blocks = self._freeze_blocks(blocks_m)
        definitions, uses = self._collect_defs_uses(proc, blocks_m)
        # Parameter definitions always reach the entry block.
        roots_id = self._roots.roots_id
        for index, arg in enumerate(proc.args):
            if proc.is_method and index == 0 and arg.arg in {"self", "cls"}:
                # Still track self/cls as parameters.
                pass
            def_id = _identity(
                "vpg-def",
                {
                    "proc": proc.procedure_id,
                    "var": arg.arg,
                    "kind": DefinitionKind.PARAMETER.value,
                    "line": proc.lineno,
                    "index": index,
                },
            )
            ann = _annotation_text(arg.annotation)
            definitions.append(
                ReachingDefinition(
                    def_id=def_id,
                    variable=arg.arg,
                    kind=DefinitionKind.PARAMETER,
                    block_id=blocks_m[0].block_id if blocks_m else f"{proc.procedure_id}::entry",
                    procedure_id=proc.procedure_id,
                    location=SourceLocation(path=proc.path, line_start=proc.lineno, line_end=proc.lineno),
                    producer_id=PRODUCER_ID,
                    roots_id=roots_id,
                    expression_ref=f"param:{arg.arg}",
                    type_annotation=ann,
                    status=ProvenanceStatus.PROVED,
                    source_kind_detail="parameter",
                )
            )

        gen, kill = self._gen_kill(definitions, blocks_m)
        reaching_in, reaching_out = self._reaching_definitions(blocks_m, gen, kill)
        chains = self._build_def_use_chains(uses, reaching_in, definitions, edge_labels)
        dom_facts = self._dominance_facts(blocks_m, proc.procedure_id)
        path_conditions, refinements = self._path_conditions_and_refinements(
            proc, blocks_m, edge_labels
        )
        infos = self._information_provenances(definitions, facets)
        threads, thread_unknowns = self._interprocedural_threads(
            proc, uses, definitions, reaching_in, sig_map
        )
        unknowns.extend(thread_unknowns)
        unknowns.extend(self._scan_unsupported_shapes(proc, blocks_m))
        unknowns.extend(self._scan_call_site_unknowns(proc, blocks_m))
        # Promote definition-level unknowns into the frontier.
        for item in definitions:
            for reason_text in item.unknown_reasons:
                try:
                    reason = UnknownReason(reason_text)
                except ValueError:
                    continue
                unknowns.append(
                    UnknownFrontierFact(
                        fact_id=_identity(
                            "vpg-unknown",
                            {
                                "reason": reason.value,
                                "def": item.def_id,
                            },
                        ),
                        reason=reason,
                        procedure_id=proc.procedure_id,
                        block_id=item.block_id,
                        variable=item.variable,
                        detail=item.source_kind_detail or item.expression_ref,
                        evidence_refs=(item.def_id,),
                    )
                )

        # Mark uses that have zero reaching defs as branch-local absence.
        def_by_id = {item.def_id: item for item in definitions}
        use_has_chain = {chain.use_id for chain in chains}
        for use in uses:
            if use.use_id not in use_has_chain:
                unknowns.append(
                    UnknownFrontierFact(
                        fact_id=_identity(
                            "vpg-unknown",
                            {
                                "reason": UnknownReason.BRANCH_LOCAL_ABSENCE.value,
                                "use": use.use_id,
                            },
                        ),
                        reason=UnknownReason.BRANCH_LOCAL_ABSENCE,
                        procedure_id=proc.procedure_id,
                        block_id=use.block_id,
                        variable=use.variable,
                        detail="no reaching definition at use",
                        evidence_refs=(use.use_id,),
                    )
                )

        # Multiple reaching definitions at a use → alias / multi-def ambiguity.
        chains_by_use: dict[str, list[DefUseChain]] = defaultdict(list)
        for chain in chains:
            chains_by_use[chain.use_id].append(chain)
        for use_id, group in chains_by_use.items():
            if len(group) > 1:
                use = next((u for u in uses if u.use_id == use_id), None)
                # Field / alias ambiguity when kinds differ or aliases present.
                kinds = {
                    def_by_id[c.def_id].kind
                    for c in group
                    if c.def_id in def_by_id
                }
                reason = (
                    UnknownReason.ALIAS_AMBIGUITY
                    if DefinitionKind.ALIAS in kinds or len(kinds) > 1
                    else UnknownReason.MULTIPLE_REACHING
                )
                unknowns.append(
                    UnknownFrontierFact(
                        fact_id=_identity(
                            "vpg-unknown",
                            {"reason": reason.value, "use": use_id},
                        ),
                        reason=reason,
                        procedure_id=proc.procedure_id,
                        block_id=use.block_id if use else "",
                        variable=use.variable if use else "",
                        detail=f"{len(group)} reaching definitions",
                        evidence_refs=tuple(sorted(c.def_id for c in group)),
                    )
                )

        frozen_blocks = self._freeze_blocks(blocks_m)
        return (
            frozen_blocks,
            definitions,
            uses,
            chains,
            dom_facts,
            path_conditions,
            refinements,
            infos,
            threads,
            unknowns,
        )

    # ------------------------------------------------------------------
    # CFG construction
    # ------------------------------------------------------------------

    def _build_cfg(
        self, proc: _ProcIR
    ) -> tuple[list[_MutableBlock], dict[tuple[str, str], dict[str, Any]], list[UnknownFrontierFact]]:
        blocks: list[_MutableBlock] = []
        edge_labels: dict[tuple[str, str], dict[str, Any]] = {}
        unknowns: list[UnknownFrontierFact] = []
        counter = {"n": 0}

        def new_block(
            label: str,
            *,
            is_entry: bool = False,
            is_exit: bool = False,
            loop_header: bool = False,
            exception_handler: bool = False,
            shape: CfgShapeSupport = CfgShapeSupport.SUPPORTED,
            reasons: Sequence[str] = (),
        ) -> _MutableBlock:
            counter["n"] += 1
            block = _MutableBlock(
                block_id=f"{proc.procedure_id}::b{counter['n']}",
                procedure_id=proc.procedure_id,
                label=label,
                is_entry=is_entry,
                is_exit=is_exit,
                loop_header=loop_header,
                exception_handler=exception_handler,
                shape_support=shape,
                unknown_reasons=list(reasons),
            )
            blocks.append(block)
            return block

        def link(
            src: _MutableBlock,
            dst: _MutableBlock,
            *,
            predicate: str = "",
            polarity: bool = True,
            branch_label: str = "",
            guard_variable: str = "",
            type_refinement: str = "",
        ) -> None:
            if dst.block_id not in src.successors:
                src.successors.append(dst.block_id)
            if src.block_id not in dst.predecessors:
                dst.predecessors.append(src.block_id)
            edge_labels[(src.block_id, dst.block_id)] = {
                "predicate": predicate,
                "polarity": polarity,
                "branch_label": branch_label,
                "guard_variable": guard_variable,
                "type_refinement": type_refinement,
            }

        entry = new_block("entry", is_entry=True)
        exit_block = new_block("exit", is_exit=True)

        def lower(
            stmts: Sequence[ast.stmt],
            current: _MutableBlock,
            *,
            loop_depth: int = 0,
        ) -> _MutableBlock:
            for stmt in stmts:
                if isinstance(stmt, (ast.Pass, ast.Expr)) and not isinstance(
                    getattr(stmt, "value", None), ast.Call
                ):
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, (ast.Constant,)):
                        current.stmts.append(stmt)
                        continue
                    if isinstance(stmt, ast.Pass):
                        current.stmts.append(stmt)
                        continue

                if isinstance(
                    stmt,
                    (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return, ast.Delete),
                ):
                    current.stmts.append(stmt)
                    continue

                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    current.stmts.append(stmt)
                    func_name = _expr_name(stmt.value.func)
                    if _is_reflection_call(func_name):
                        unknowns.append(
                            UnknownFrontierFact(
                                fact_id=_identity(
                                    "vpg-unknown",
                                    {
                                        "reason": UnknownReason.REFLECTION.value,
                                        "proc": proc.procedure_id,
                                        "line": _lineno(stmt),
                                    },
                                ),
                                reason=UnknownReason.REFLECTION,
                                procedure_id=proc.procedure_id,
                                block_id=current.block_id,
                                detail=func_name,
                            )
                        )
                        current.unknown_reasons.append(UnknownReason.REFLECTION.value)
                        current.shape_support = CfgShapeSupport.PARTIAL
                    if _is_native_call(func_name):
                        unknowns.append(
                            UnknownFrontierFact(
                                fact_id=_identity(
                                    "vpg-unknown",
                                    {
                                        "reason": UnknownReason.NATIVE_CALL.value,
                                        "proc": proc.procedure_id,
                                        "line": _lineno(stmt),
                                    },
                                ),
                                reason=UnknownReason.NATIVE_CALL,
                                procedure_id=proc.procedure_id,
                                block_id=current.block_id,
                                detail=func_name,
                            )
                        )
                        current.unknown_reasons.append(UnknownReason.NATIVE_CALL.value)
                        current.shape_support = CfgShapeSupport.PARTIAL
                    if _is_concurrency_call(func_name):
                        unknowns.append(
                            UnknownFrontierFact(
                                fact_id=_identity(
                                    "vpg-unknown",
                                    {
                                        "reason": UnknownReason.CONCURRENCY.value,
                                        "proc": proc.procedure_id,
                                        "line": _lineno(stmt),
                                    },
                                ),
                                reason=UnknownReason.CONCURRENCY,
                                procedure_id=proc.procedure_id,
                                block_id=current.block_id,
                                detail=func_name,
                            )
                        )
                        current.unknown_reasons.append(UnknownReason.CONCURRENCY.value)
                        current.shape_support = CfgShapeSupport.PARTIAL
                    continue

                if isinstance(stmt, ast.If):
                    # end current block; create then/else/join
                    pred = _annotation_text(stmt.test)
                    guard_var, refinement = self._extract_guard(stmt.test)
                    then_block = new_block("then")
                    else_block = new_block("else")
                    join_block = new_block("join")
                    join_block.is_join = True
                    link(
                        current,
                        then_block,
                        predicate=pred,
                        polarity=True,
                        branch_label="then",
                        guard_variable=guard_var,
                        type_refinement=refinement,
                    )
                    link(
                        current,
                        else_block,
                        predicate=pred,
                        polarity=False,
                        branch_label="else",
                        guard_variable=guard_var,
                    )
                    then_end = lower(stmt.body, then_block, loop_depth=loop_depth)
                    else_end = lower(stmt.orelse, else_block, loop_depth=loop_depth)
                    link(then_end, join_block)
                    link(else_end, join_block)
                    current = join_block
                    continue

                if isinstance(stmt, (ast.For, ast.While)):
                    if loop_depth >= self._max_loop_unroll:
                        # Beyond bound: single partial block, no unroll.
                        loop_block = new_block(
                            "loop_beyond_bounds",
                            loop_header=True,
                            shape=CfgShapeSupport.PARTIAL,
                            reasons=(UnknownReason.LOOP_BEYOND_BOUNDS.value,),
                        )
                        link(current, loop_block)
                        loop_block.stmts.append(stmt)
                        unknowns.append(
                            UnknownFrontierFact(
                                fact_id=_identity(
                                    "vpg-unknown",
                                    {
                                        "reason": UnknownReason.LOOP_BEYOND_BOUNDS.value,
                                        "proc": proc.procedure_id,
                                        "line": _lineno(stmt),
                                    },
                                ),
                                reason=UnknownReason.LOOP_BEYOND_BOUNDS,
                                procedure_id=proc.procedure_id,
                                block_id=loop_block.block_id,
                                detail=f"loop depth {loop_depth} exceeds bound {self._max_loop_unroll}",
                            )
                        )
                        after = new_block("after_loop")
                        link(loop_block, after)
                        # Also assume loop may be skipped.
                        link(current, after)
                        current = after
                        continue
                    header = new_block("loop_header", loop_header=True)
                    body = new_block("loop_body")
                    after = new_block("after_loop")
                    link(current, header)
                    pred = (
                        _annotation_text(stmt.test)
                        if isinstance(stmt, ast.While)
                        else f"iter:{_annotation_text(stmt.iter)}"
                    )
                    link(
                        header,
                        body,
                        predicate=pred,
                        polarity=True,
                        branch_label="loop_enter",
                    )
                    link(
                        header,
                        after,
                        predicate=pred,
                        polarity=False,
                        branch_label="loop_exit",
                    )
                    body_end = lower(stmt.body, body, loop_depth=loop_depth + 1)
                    link(body_end, header)  # back-edge
                    # orelse of for/while executes when loop ends without break;
                    # we model it as optional after_loop prefix (partial).
                    if stmt.orelse:
                        orelse_block = new_block(
                            "loop_orelse",
                            shape=CfgShapeSupport.PARTIAL,
                            reasons=(UnknownReason.UNSUPPORTED_CFG.value,),
                        )
                        link(header, orelse_block, branch_label="orelse")
                        orelse_end = lower(stmt.orelse, orelse_block, loop_depth=loop_depth)
                        link(orelse_end, after)
                    current = after
                    continue

                if isinstance(stmt, ast.Try):
                    try_block = new_block(
                        "try",
                        shape=CfgShapeSupport.PARTIAL,
                        reasons=(UnknownReason.EXCEPTION_PATH.value,),
                    )
                    link(current, try_block)
                    try_end = lower(stmt.body, try_block, loop_depth=loop_depth)
                    join = new_block("try_join")
                    join.is_join = True
                    link(try_end, join)
                    unknowns.append(
                        UnknownFrontierFact(
                            fact_id=_identity(
                                "vpg-unknown",
                                {
                                    "reason": UnknownReason.EXCEPTION_PATH.value,
                                    "proc": proc.procedure_id,
                                    "line": _lineno(stmt),
                                },
                            ),
                            reason=UnknownReason.EXCEPTION_PATH,
                            procedure_id=proc.procedure_id,
                            block_id=try_block.block_id,
                            detail="exception edges are not proved",
                        )
                    )
                    for handler in stmt.handlers:
                        hblock = new_block(
                            "except",
                            exception_handler=True,
                            shape=CfgShapeSupport.PARTIAL,
                            reasons=(UnknownReason.EXCEPTION_PATH.value,),
                        )
                        # Exception edge from try is unknown, not a proved edge.
                        # We still lower the handler body for local defs.
                        if handler.name:
                            # Exception bind is an unknown-origin definition.
                            pass
                        hend = lower(handler.body, hblock, loop_depth=loop_depth)
                        link(hend, join)
                        # Soft edge from try (unknown exception transfer).
                        link(try_block, hblock, branch_label="except_unknown")
                    if stmt.orelse:
                        oblock = new_block("try_else")
                        link(try_end, oblock)
                        oend = lower(stmt.orelse, oblock, loop_depth=loop_depth)
                        link(oend, join)
                    if stmt.finalbody:
                        fblock = new_block(
                            "finally",
                            shape=CfgShapeSupport.PARTIAL,
                            reasons=(UnknownReason.EXCEPTION_PATH.value,),
                        )
                        link(join, fblock)
                        current = lower(stmt.finalbody, fblock, loop_depth=loop_depth)
                    else:
                        current = join
                    continue

                if isinstance(stmt, (ast.With, ast.AsyncWith)):
                    # Supported as sequential body; context-manager effects unknown.
                    current.stmts.append(stmt)
                    current = lower(stmt.body, current, loop_depth=loop_depth)
                    continue

                if isinstance(stmt, (ast.Break, ast.Continue)):
                    current.stmts.append(stmt)
                    current.shape_support = CfgShapeSupport.PARTIAL
                    current.unknown_reasons.append(UnknownReason.UNSUPPORTED_CFG.value)
                    unknowns.append(
                        UnknownFrontierFact(
                            fact_id=_identity(
                                "vpg-unknown",
                                {
                                    "reason": UnknownReason.UNSUPPORTED_CFG.value,
                                    "proc": proc.procedure_id,
                                    "line": _lineno(stmt),
                                    "stmt": type(stmt).__name__,
                                },
                            ),
                            reason=UnknownReason.UNSUPPORTED_CFG,
                            procedure_id=proc.procedure_id,
                            block_id=current.block_id,
                            detail=type(stmt).__name__,
                        )
                    )
                    continue

                if isinstance(stmt, (ast.Raise, ast.Assert, ast.Global, ast.Nonlocal)):
                    current.stmts.append(stmt)
                    if isinstance(stmt, ast.Raise):
                        current.unknown_reasons.append(UnknownReason.EXCEPTION_PATH.value)
                        unknowns.append(
                            UnknownFrontierFact(
                                fact_id=_identity(
                                    "vpg-unknown",
                                    {
                                        "reason": UnknownReason.EXCEPTION_PATH.value,
                                        "proc": proc.procedure_id,
                                        "line": _lineno(stmt),
                                    },
                                ),
                                reason=UnknownReason.EXCEPTION_PATH,
                                procedure_id=proc.procedure_id,
                                block_id=current.block_id,
                                detail="raise",
                            )
                        )
                    continue

                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # Nested definitions are extracted separately; ignore bodies here.
                    continue

                # Unsupported statement shape.
                current.stmts.append(stmt)
                current.shape_support = CfgShapeSupport.UNSUPPORTED
                current.unknown_reasons.append(UnknownReason.UNSUPPORTED_AST.value)
                unknowns.append(
                    UnknownFrontierFact(
                        fact_id=_identity(
                            "vpg-unknown",
                            {
                                "reason": UnknownReason.UNSUPPORTED_AST.value,
                                "proc": proc.procedure_id,
                                "line": _lineno(stmt),
                                "stmt": type(stmt).__name__,
                            },
                        ),
                        reason=UnknownReason.UNSUPPORTED_AST,
                        procedure_id=proc.procedure_id,
                        block_id=current.block_id,
                        detail=type(stmt).__name__,
                    )
                )
            return current

        end = lower(proc.body, entry)
        link(end, exit_block)
        return blocks, edge_labels, unknowns

    def _freeze_blocks(self, blocks: Sequence[_MutableBlock]) -> list[CfgBlock]:
        result: list[CfgBlock] = []
        for block in blocks:
            stmt_refs = []
            for stmt in block.stmts:
                kind = type(stmt).__name__
                stmt_refs.append(f"{kind}@{_lineno(stmt)}")
            # Mark joins.
            is_join = block.is_join or len(block.predecessors) > 1
            result.append(
                CfgBlock(
                    block_id=block.block_id,
                    procedure_id=block.procedure_id,
                    label=block.label,
                    statements=tuple(stmt_refs),
                    successors=tuple(block.successors),
                    predecessors=tuple(block.predecessors),
                    shape_support=block.shape_support,
                    unknown_reasons=tuple(block.unknown_reasons),
                    is_entry=block.is_entry,
                    is_exit=block.is_exit,
                    is_join=is_join,
                    loop_header=block.loop_header,
                    exception_handler=block.exception_handler,
                )
            )
        return result

    def _extract_guard(self, test: ast.AST) -> tuple[str, str]:
        """Return (guard_variable, type_refinement_text) from a branch test."""
        if isinstance(test, ast.Call) and _expr_name(test.func) in {
            "isinstance",
            "issubclass",
        }:
            if len(test.args) >= 2:
                var = _expr_name(test.args[0])
                typ = _annotation_text(test.args[1])
                return var, typ
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            left = _expr_name(test.left)
            if isinstance(test.ops[0], ast.Is) and test.comparators:
                right = test.comparators[0]
                if isinstance(right, ast.Constant) and right.value is None:
                    return left, "None"
            if isinstance(test.ops[0], ast.IsNot) and test.comparators:
                right = test.comparators[0]
                if isinstance(right, ast.Constant) and right.value is None:
                    return left, "NonNull"
            if isinstance(test.ops[0], (ast.Eq, ast.NotEq)):
                return left, ""
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            var, refinement = self._extract_guard(test.operand)
            return var, refinement
        if isinstance(test, ast.Name):
            return test.id, ""
        if isinstance(test, ast.Attribute):
            return _expr_name(test), ""
        return "", ""

    # ------------------------------------------------------------------
    # Def / use collection
    # ------------------------------------------------------------------

    def _collect_defs_uses(
        self, proc: _ProcIR, blocks: Sequence[_MutableBlock]
    ) -> tuple[list[ReachingDefinition], list[ValueUse]]:
        definitions: list[ReachingDefinition] = []
        uses: list[ValueUse] = []
        roots_id = self._roots.roots_id

        def add_use(
            variable: str,
            kind: UseKind,
            block: _MutableBlock,
            node: ast.AST,
            *,
            expression_ref: str = "",
            status: ProvenanceStatus = ProvenanceStatus.PROVED,
            reasons: Sequence[str] = (),
        ) -> None:
            if not variable or not variable.isidentifier() and "." not in variable:
                # Allow simple field paths like self.x
                if not variable or not re.match(r"^[A-Za-z_][\w.]*$", variable):
                    return
            use_id = _identity(
                "vpg-use",
                {
                    "proc": proc.procedure_id,
                    "var": variable,
                    "kind": kind.value,
                    "block": block.block_id,
                    "line": _lineno(node),
                    "col": int(getattr(node, "col_offset", 0) or 0),
                    "expr": expression_ref or variable,
                },
            )
            uses.append(
                ValueUse(
                    use_id=use_id,
                    variable=variable.split(".", 1)[0] if kind is UseKind.FIELD_READ else variable,
                    kind=kind,
                    block_id=block.block_id,
                    procedure_id=proc.procedure_id,
                    location=SourceLocation.from_ast(proc.path, node),
                    expression_ref=expression_ref or variable,
                    status=status,
                    unknown_reasons=tuple(reasons),
                )
            )

        def add_def(
            variable: str,
            kind: DefinitionKind,
            block: _MutableBlock,
            node: ast.AST,
            *,
            expression_ref: str = "",
            type_annotation: str = "",
            status: ProvenanceStatus = ProvenanceStatus.PROVED,
            reasons: Sequence[str] = (),
            aliases: Sequence[str] = (),
            field_path: str = "",
            detail: str = "",
            ip_complete: InterproceduralCompleteness = InterproceduralCompleteness.COMPLETE,
            attributes: Mapping[str, Any] | None = None,
        ) -> ReachingDefinition:
            def_id = _identity(
                "vpg-def",
                {
                    "proc": proc.procedure_id,
                    "var": variable,
                    "kind": kind.value,
                    "block": block.block_id,
                    "line": _lineno(node),
                    "col": int(getattr(node, "col_offset", 0) or 0),
                    "expr": expression_ref,
                    "field": field_path,
                },
            )
            item = ReachingDefinition(
                def_id=def_id,
                variable=variable,
                kind=kind,
                block_id=block.block_id,
                procedure_id=proc.procedure_id,
                location=SourceLocation.from_ast(proc.path, node),
                producer_id=PRODUCER_ID,
                roots_id=roots_id,
                expression_ref=expression_ref,
                type_annotation=type_annotation,
                status=status,
                unknown_reasons=tuple(reasons),
                aliases=tuple(aliases),
                field_path=field_path,
                source_kind_detail=detail or kind.value,
                interprocedural_completeness=ip_complete,
                attributes=attributes or {},
            )
            definitions.append(item)
            return item

        def walk_loads(expr: ast.AST, block: _MutableBlock, kind: UseKind) -> None:
            for sub in ast.walk(expr):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    add_use(sub.id, kind, block, sub)
                elif isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Load):
                    add_use(
                        _expr_name(sub),
                        UseKind.FIELD_READ,
                        block,
                        sub,
                        expression_ref=_expr_name(sub),
                    )

        for block in blocks:
            for stmt in block.stmts:
                if isinstance(stmt, ast.Assign):
                    walk_loads(stmt.value, block, UseKind.STORE_RHS)
                    kind, detail, status, reasons, aliases, attrs = self._classify_rhs(
                        stmt.value
                    )
                    for target in stmt.targets:
                        self._define_target(
                            target,
                            kind,
                            block,
                            stmt,
                            add_def,
                            add_use,
                            expression_ref=_annotation_text(stmt.value),
                            status=status,
                            reasons=reasons,
                            aliases=aliases,
                            detail=detail,
                            attributes=attrs,
                        )
                elif isinstance(stmt, ast.AnnAssign):
                    if stmt.value is not None:
                        walk_loads(stmt.value, block, UseKind.STORE_RHS)
                    kind, detail, status, reasons, aliases, attrs = self._classify_rhs(
                        stmt.value
                    )
                    self._define_target(
                        stmt.target,
                        kind if stmt.value is not None else DefinitionKind.ANN_ASSIGN,
                        block,
                        stmt,
                        add_def,
                        add_use,
                        expression_ref=_annotation_text(stmt.value) if stmt.value else "",
                        type_annotation=_annotation_text(stmt.annotation),
                        status=status if stmt.value is not None else ProvenanceStatus.PROVED,
                        reasons=reasons,
                        aliases=aliases,
                        detail=detail or "ann_assign",
                        attributes=attrs,
                    )
                elif isinstance(stmt, ast.AugAssign):
                    walk_loads(stmt.value, block, UseKind.AUG_OPERAND)
                    if isinstance(stmt.target, ast.Name):
                        add_use(stmt.target.id, UseKind.AUG_OPERAND, block, stmt.target)
                    self._define_target(
                        stmt.target,
                        DefinitionKind.AUG_ASSIGN,
                        block,
                        stmt,
                        add_def,
                        add_use,
                        expression_ref=_annotation_text(stmt.value),
                        detail="aug_assign",
                    )
                elif isinstance(stmt, ast.Return):
                    if stmt.value is not None:
                        walk_loads(stmt.value, block, UseKind.RETURN)
                        add_def(
                            "<return>",
                            DefinitionKind.RETURN,
                            block,
                            stmt,
                            expression_ref=_annotation_text(stmt.value),
                            type_annotation=_annotation_text(proc.returns_annotation),
                            detail="return",
                        )
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    walk_loads(stmt.value, block, UseKind.CALL_ARG)
                    func_name = _expr_name(stmt.value.func)
                    for arg in stmt.value.args:
                        name = _expr_name(arg)
                        if name and re.match(r"^[A-Za-z_][\w.]*$", name):
                            add_use(
                                name.split(".", 1)[0],
                                UseKind.CALL_ARG,
                                block,
                                arg,
                                expression_ref=name,
                            )
                    for kw in stmt.value.keywords:
                        if kw.value is not None:
                            walk_loads(kw.value, block, UseKind.CALL_ARG)
                elif isinstance(stmt, ast.If):
                    walk_loads(stmt.test, block, UseKind.CONDITION)
                elif isinstance(stmt, (ast.For, ast.While)):
                    if isinstance(stmt, ast.While):
                        walk_loads(stmt.test, block, UseKind.CONDITION)
                    else:
                        walk_loads(stmt.iter, block, UseKind.LOAD)
                        # for-target is a definition.
                        self._define_target(
                            stmt.target,
                            DefinitionKind.ASSIGNMENT,
                            block,
                            stmt,
                            add_def,
                            add_use,
                            expression_ref=_annotation_text(stmt.iter),
                            detail="for_target",
                            status=ProvenanceStatus.PARTIAL,
                            reasons=(UnknownReason.LOOP_BEYOND_BOUNDS.value,),
                        )
                elif isinstance(stmt, ast.Delete):
                    for target in stmt.targets:
                        walk_loads(target, block, UseKind.LOAD)
                else:
                    # Generic load walk for residual statements.
                    for sub in ast.walk(stmt):
                        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                            add_use(sub.id, UseKind.LOAD, block, sub)

        return definitions, uses

    def _define_target(
        self,
        target: ast.AST,
        kind: DefinitionKind,
        block: _MutableBlock,
        node: ast.AST,
        add_def,
        add_use,
        *,
        expression_ref: str = "",
        type_annotation: str = "",
        status: ProvenanceStatus = ProvenanceStatus.PROVED,
        reasons: Sequence[str] = (),
        aliases: Sequence[str] = (),
        detail: str = "",
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(target, ast.Name):
            add_def(
                target.id,
                kind,
                block,
                node,
                expression_ref=expression_ref,
                type_annotation=type_annotation,
                status=status,
                reasons=reasons,
                aliases=aliases,
                detail=detail,
                attributes=attributes,
            )
        elif isinstance(target, ast.Attribute):
            base = _expr_name(target.value)
            field_path = f"{base}.{target.attr}" if base else target.attr
            add_def(
                base.split(".", 1)[0] if base else target.attr,
                DefinitionKind.FIELD_WRITE,
                block,
                node,
                expression_ref=expression_ref,
                type_annotation=type_annotation,
                status=status,
                reasons=reasons,
                aliases=aliases,
                field_path=field_path,
                detail=detail or "field_write",
                attributes=attributes,
            )
            if base:
                root = base.split(".", 1)[0]
                if root.isidentifier():
                    add_use(root, UseKind.FIELD_READ, block, target.value, expression_ref=base)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._define_target(
                    elt,
                    kind,
                    block,
                    node,
                    add_def,
                    add_use,
                    expression_ref=expression_ref,
                    status=status,
                    reasons=reasons,
                    detail=detail,
                    attributes=attributes,
                )
        elif isinstance(target, ast.Subscript):
            base = _expr_name(target.value)
            add_def(
                base.split(".", 1)[0] if base else "<subscript>",
                DefinitionKind.FIELD_WRITE,
                block,
                node,
                expression_ref=expression_ref,
                status=ProvenanceStatus.PARTIAL,
                reasons=tuple(reasons) + (UnknownReason.ALIAS_AMBIGUITY.value,),
                field_path=base,
                detail="subscript_write",
                attributes=attributes,
            )

    def _classify_rhs(
        self, value: ast.AST | None
    ) -> tuple[
        DefinitionKind,
        str,
        ProvenanceStatus,
        tuple[str, ...],
        tuple[str, ...],
        dict[str, Any],
    ]:
        if value is None:
            return (
                DefinitionKind.UNKNOWN,
                "",
                ProvenanceStatus.UNKNOWN,
                (UnknownReason.UNSUPPORTED_AST.value,),
                (),
                {},
            )
        if isinstance(value, ast.Constant):
            return DefinitionKind.CONSTANT, "constant", ProvenanceStatus.PROVED, (), (), {
                "constant": True
            }
        if isinstance(value, ast.Name):
            return (
                DefinitionKind.ALIAS,
                "alias",
                ProvenanceStatus.PROVED,
                (),
                (value.id,),
                {"alias_of": value.id},
            )
        if isinstance(value, ast.Call):
            func_name = _expr_name(value.func)
            attrs: dict[str, Any] = {"callee": func_name}
            effects = _effects_for_call(func_name)
            if effects:
                attrs["effects"] = list(effects)
            if _is_reflection_call(func_name):
                return (
                    DefinitionKind.CALL_RESULT,
                    func_name,
                    ProvenanceStatus.UNKNOWN,
                    (UnknownReason.REFLECTION.value,),
                    (),
                    attrs,
                )
            if _is_native_call(func_name):
                return (
                    DefinitionKind.CALL_RESULT,
                    func_name,
                    ProvenanceStatus.UNKNOWN,
                    (UnknownReason.NATIVE_CALL.value,),
                    (),
                    attrs,
                )
            if _is_concurrency_call(func_name):
                return (
                    DefinitionKind.CALL_RESULT,
                    func_name,
                    ProvenanceStatus.UNKNOWN,
                    (UnknownReason.CONCURRENCY.value,),
                    (),
                    attrs,
                )
            if _is_config_call(func_name):
                return (
                    DefinitionKind.CONFIG_SOURCE,
                    func_name,
                    ProvenanceStatus.PROVED,
                    (),
                    (),
                    {**attrs, "origin": "config"},
                )
            if _is_di_call(func_name):
                return (
                    DefinitionKind.DI_SOURCE,
                    func_name,
                    ProvenanceStatus.PROVED,
                    (),
                    (),
                    {**attrs, "origin": "di"},
                )
            if _is_conversion_call(func_name):
                return (
                    DefinitionKind.CONVERSION,
                    func_name,
                    ProvenanceStatus.PROVED,
                    (),
                    (),
                    {**attrs, "origin": "conversion"},
                )
            if _is_constructor_call(func_name):
                return (
                    DefinitionKind.CONSTRUCTOR,
                    func_name,
                    ProvenanceStatus.PROVED,
                    (),
                    (),
                    {**attrs, "origin": "constructor"},
                )
            return (
                DefinitionKind.CALL_RESULT,
                func_name,
                ProvenanceStatus.PARTIAL,
                (UnknownReason.INCOMPLETE_INTERPROCEDURAL.value,),
                (),
                attrs,
            )
        if isinstance(value, ast.Attribute):
            return (
                DefinitionKind.ASSIGNMENT,
                _expr_name(value),
                ProvenanceStatus.PROVED,
                (),
                (),
                {"field_read": _expr_name(value)},
            )
        if isinstance(value, (ast.List, ast.Dict, ast.Set, ast.Tuple)):
            return DefinitionKind.CONSTANT, type(value).__name__, ProvenanceStatus.PROVED, (), (), {
                "literal_container": True
            }
        if isinstance(value, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return (
                DefinitionKind.COMPREHENSION,
                type(value).__name__,
                ProvenanceStatus.PARTIAL,
                (UnknownReason.UNSUPPORTED_CFG.value,),
                (),
                {},
            )
        return DefinitionKind.ASSIGNMENT, type(value).__name__, ProvenanceStatus.PROVED, (), (), {}

    # ------------------------------------------------------------------
    # Reaching definitions
    # ------------------------------------------------------------------

    def _gen_kill(
        self,
        definitions: Sequence[ReachingDefinition],
        blocks: Sequence[_MutableBlock],
    ) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        by_block: dict[str, list[ReachingDefinition]] = defaultdict(list)
        by_var: dict[str, list[ReachingDefinition]] = defaultdict(list)
        for item in definitions:
            by_block[item.block_id].append(item)
            by_var[item.variable].append(item)
        gen: dict[str, set[str]] = {}
        kill: dict[str, set[str]] = {}
        for block in blocks:
            gens: set[str] = set()
            kills: set[str] = set()
            # Last definition of each variable in the block wins for gen.
            last_for_var: dict[str, str] = {}
            for item in by_block.get(block.block_id, ()):
                last_for_var[item.variable] = item.def_id
            for var, def_id in last_for_var.items():
                gens.add(def_id)
                for other in by_var.get(var, ()):
                    if other.def_id != def_id:
                        kills.add(other.def_id)
            gen[block.block_id] = gens
            kill[block.block_id] = kills
        return gen, kill

    def _reaching_definitions(
        self,
        blocks: Sequence[_MutableBlock],
        gen: Mapping[str, set[str]],
        kill: Mapping[str, set[str]],
    ) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        block_ids = [block.block_id for block in blocks]
        preds = {block.block_id: list(block.predecessors) for block in blocks}
        succs = {block.block_id: list(block.successors) for block in blocks}
        rin: dict[str, set[str]] = {bid: set() for bid in block_ids}
        rout: dict[str, set[str]] = {bid: set() for bid in block_ids}
        # Worklist.
        changed = deque(block_ids)
        in_queue = set(block_ids)
        iterations = 0
        max_iterations = max(1, len(block_ids) * max(1, len(block_ids)) * 4)
        while changed and iterations < max_iterations:
            iterations += 1
            bid = changed.popleft()
            in_queue.discard(bid)
            new_in: set[str] = set()
            for pred in preds.get(bid, ()):
                new_in |= rout.get(pred, set())
            new_out = set(gen.get(bid, set())) | (new_in - set(kill.get(bid, set())))
            if new_in != rin[bid] or new_out != rout[bid]:
                rin[bid] = new_in
                rout[bid] = new_out
                for succ in succs.get(bid, ()):
                    if succ not in in_queue:
                        changed.append(succ)
                        in_queue.add(succ)
        return rin, rout

    def _build_def_use_chains(
        self,
        uses: Sequence[ValueUse],
        reaching_in: Mapping[str, set[str]],
        definitions: Sequence[ReachingDefinition],
        edge_labels: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> list[DefUseChain]:
        def_by_id = {item.def_id: item for item in definitions}
        # Same-block defs indexed by (block, variable) for local order.
        same_block: dict[tuple[str, str], list[ReachingDefinition]] = defaultdict(list)
        for item in definitions:
            same_block[(item.block_id, item.variable)].append(item)
        for key in same_block:
            same_block[key].sort(
                key=lambda d: (
                    d.location.line_start,
                    d.location.column_start,
                    0 if d.kind is DefinitionKind.PARAMETER else 1,
                    d.def_id,
                )
            )

        chains: list[DefUseChain] = []
        for use in uses:
            reaching = set(reaching_in.get(use.block_id, set()))
            # Include same-block definitions that precede this use (line order),
            # and always include parameter definitions in the same block.
            local_defs = same_block.get((use.block_id, use.variable), [])
            for item in local_defs:
                if item.kind is DefinitionKind.PARAMETER:
                    reaching.add(item.def_id)
                elif item.location.line_start < use.location.line_start:
                    reaching.add(item.def_id)
                elif (
                    item.location.line_start == use.location.line_start
                    and item.location.column_start <= use.location.column_start
                    and item.kind is not DefinitionKind.PARAMETER
                ):
                    # Same-line: only if def is a parameter-like prior binding.
                    pass
            # Kill later redefinitions in the same block that strictly precede the use:
            # keep only the last local def at or before the use line.
            local_before = [
                item
                for item in local_defs
                if item.kind is DefinitionKind.PARAMETER
                or item.location.line_start < use.location.line_start
                or (
                    item.location.line_start == use.location.line_start
                    and item.kind is DefinitionKind.PARAMETER
                )
            ]
            if local_before:
                # Prefer the last non-parameter local def; fall back to parameters.
                non_params = [
                    item for item in local_before if item.kind is not DefinitionKind.PARAMETER
                ]
                if non_params:
                    last = non_params[-1]
                    for item in non_params:
                        if item.def_id != last.def_id:
                            reaching.discard(item.def_id)
                    reaching.add(last.def_id)
                else:
                    for item in local_before:
                        reaching.add(item.def_id)

            candidates = [
                def_id
                for def_id in reaching
                if def_id in def_by_id and def_by_id[def_id].variable == use.variable
            ]
            # Also accept field-path uses matching field writes.
            if use.kind is UseKind.FIELD_READ:
                expr = use.expression_ref
                candidates.extend(
                    def_id
                    for def_id, item in def_by_id.items()
                    if item.field_path == expr
                    and (
                        def_id in reaching
                        or item.block_id == use.block_id
                    )
                )
            status = ProvenanceStatus.PROVED
            reasons: list[str] = []
            unique_candidates = sorted(set(candidates))
            if len(unique_candidates) > 1:
                status = ProvenanceStatus.PARTIAL
                reasons.append(UnknownReason.MULTIPLE_REACHING.value)
            for def_id in unique_candidates:
                item = def_by_id[def_id]
                chain_status = status
                chain_reasons = list(reasons)
                if item.status is not ProvenanceStatus.PROVED:
                    chain_status = item.status
                    chain_reasons.extend(item.unknown_reasons)
                chain_id = _identity(
                    "vpg-chain",
                    {"def": def_id, "use": use.use_id, "var": use.variable},
                )
                chains.append(
                    DefUseChain(
                        chain_id=chain_id,
                        def_id=def_id,
                        use_id=use.use_id,
                        variable=use.variable,
                        status=chain_status,
                        unknown_reasons=tuple(sorted(set(chain_reasons))),
                        dependency_direction=DependencyDirection.FLOWS_TO,
                    )
                )
            if len(chains) > DEFAULT_MAX_EDGES:
                raise ValueProvenanceBoundsError("def-use chain count exceeds hard bound")
        return chains

    # ------------------------------------------------------------------
    # Dominance
    # ------------------------------------------------------------------

    def _dominance_facts(
        self, blocks: Sequence[_MutableBlock], procedure_id: str
    ) -> list[DominanceFact]:
        if not blocks:
            return []
        block_ids = [block.block_id for block in blocks]
        id_set = set(block_ids)
        preds = {block.block_id: [p for p in block.predecessors if p in id_set] for block in blocks}
        succs = {block.block_id: [s for s in block.successors if s in id_set] for block in blocks}
        entry = next((b.block_id for b in blocks if b.is_entry), block_ids[0])
        exits = [b.block_id for b in blocks if b.is_exit] or [block_ids[-1]]

        # Dominators.
        dom: dict[str, set[str]] = {bid: set(block_ids) for bid in block_ids}
        dom[entry] = {entry}
        changed = True
        guard = 0
        while changed and guard < len(block_ids) * len(block_ids) + 2:
            guard += 1
            changed = False
            for bid in block_ids:
                if bid == entry:
                    continue
                pred_list = preds.get(bid, [])
                if not pred_list:
                    new_set = {bid}
                else:
                    inter = set(block_ids)
                    for pred in pred_list:
                        inter &= dom[pred]
                    new_set = inter | {bid}
                if new_set != dom[bid]:
                    dom[bid] = new_set
                    changed = True

        # Immediate dominators.
        idom: dict[str, str] = {}
        for bid in block_ids:
            if bid == entry:
                continue
            candidates = dom[bid] - {bid}
            # idom is the unique dominator dominated by all other dominators.
            for cand in sorted(candidates):
                if all(c == cand or cand in dom.get(c, set()) for c in candidates):
                    # cand dominates all others in candidates? Actually:
                    # idom(n) is the unique node that strictly dominates n and
                    # is dominated by every other strict dominator of n.
                    if all(other == cand or other in dom[cand] for other in candidates):
                        idom[bid] = cand
                        break

        # Post-dominators: reverse graph, treat exits as entries.
        # For multiple exits, add a virtual exit if needed.
        rpreds = {bid: list(succs.get(bid, [])) for bid in block_ids}
        rsuccs = {bid: list(preds.get(bid, [])) for bid in block_ids}
        virtual_exit = None
        if len(exits) != 1:
            virtual_exit = f"{procedure_id}::virtual_exit"
            rpreds[virtual_exit] = []
            rsuccs[virtual_exit] = list(exits)
            for ex in exits:
                rpreds[ex] = list(rpreds.get(ex, [])) + [virtual_exit]
            post_nodes = block_ids + [virtual_exit]
            post_entry = virtual_exit
        else:
            post_nodes = list(block_ids)
            post_entry = exits[0]

        pdom: dict[str, set[str]] = {bid: set(post_nodes) for bid in post_nodes}
        pdom[post_entry] = {post_entry}
        changed = True
        guard = 0
        while changed and guard < len(post_nodes) * len(post_nodes) + 2:
            guard += 1
            changed = False
            for bid in post_nodes:
                if bid == post_entry:
                    continue
                pred_list = rpreds.get(bid, [])
                if not pred_list:
                    new_set = {bid}
                else:
                    inter = set(post_nodes)
                    for pred in pred_list:
                        if pred in pdom:
                            inter &= pdom[pred]
                    new_set = inter | {bid}
                if new_set != pdom.get(bid, set()):
                    pdom[bid] = new_set
                    changed = True

        facts: list[DominanceFact] = []
        for bid in block_ids:
            for d in sorted(dom[bid]):
                kind = (
                    DominanceKind.DOMINATES
                    if d == bid
                    else DominanceKind.STRICTLY_DOMINATES
                )
                if d == bid:
                    kind = DominanceKind.DOMINATES
                fact_id = _identity(
                    "vpg-dom",
                    {
                        "kind": kind.value,
                        "dom": d,
                        "n": bid,
                        "proc": procedure_id,
                    },
                )
                facts.append(
                    DominanceFact(
                        fact_id=fact_id,
                        kind=kind if d != bid else DominanceKind.DOMINATES,
                        dominator_block_id=d,
                        dominated_block_id=bid,
                        procedure_id=procedure_id,
                    )
                )
            if bid in idom:
                fact_id = _identity(
                    "vpg-idom",
                    {"dom": idom[bid], "n": bid, "proc": procedure_id},
                )
                facts.append(
                    DominanceFact(
                        fact_id=fact_id,
                        kind=DominanceKind.IMMEDIATE_DOMINATOR,
                        dominator_block_id=idom[bid],
                        dominated_block_id=bid,
                        procedure_id=procedure_id,
                    )
                )

        for bid in block_ids:
            for d in sorted(pdom.get(bid, set())):
                if virtual_exit and d == virtual_exit:
                    continue
                kind = (
                    DominanceKind.POST_DOMINATES
                    if d == bid
                    else DominanceKind.STRICTLY_POST_DOMINATES
                )
                fact_id = _identity(
                    "vpg-pdom",
                    {
                        "kind": kind.value,
                        "dom": d,
                        "n": bid,
                        "proc": procedure_id,
                    },
                )
                facts.append(
                    DominanceFact(
                        fact_id=fact_id,
                        kind=kind if d != bid else DominanceKind.POST_DOMINATES,
                        dominator_block_id=d,
                        dominated_block_id=bid,
                        procedure_id=procedure_id,
                    )
                )
        return facts

    # ------------------------------------------------------------------
    # Path conditions & refinements
    # ------------------------------------------------------------------

    def _path_conditions_and_refinements(
        self,
        proc: _ProcIR,
        blocks: Sequence[_MutableBlock],
        edge_labels: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> tuple[list[PathCondition], list[TypeRefinement]]:
        conditions: list[PathCondition] = []
        refinements: list[TypeRefinement] = []
        for (src, dst), meta in sorted(edge_labels.items()):
            pred = str(meta.get("predicate") or "")
            if not pred and not meta.get("branch_label"):
                continue
            polarity = bool(meta.get("polarity", True))
            guard_var = str(meta.get("guard_variable") or "")
            refinement = str(meta.get("type_refinement") or "")
            branch_label = str(meta.get("branch_label") or "")
            condition_id = _identity(
                "vpg-pc",
                {
                    "proc": proc.procedure_id,
                    "src": src,
                    "dst": dst,
                    "pred": pred,
                    "polarity": polarity,
                    "branch": branch_label,
                },
            )
            status = ProvenanceStatus.PROVED
            reasons: tuple[str, ...] = ()
            if not pred and branch_label in {"except_unknown"}:
                status = ProvenanceStatus.UNKNOWN
                reasons = (UnknownReason.EXCEPTION_PATH.value,)
            conditions.append(
                PathCondition(
                    condition_id=condition_id,
                    procedure_id=proc.procedure_id,
                    block_id=dst,
                    predicate_ref=pred,
                    polarity=polarity,
                    guard_variable=guard_var,
                    type_refinement=refinement if polarity else "",
                    status=status,
                    unknown_reasons=reasons,
                    predecessor_block_id=src,
                    branch_label=branch_label,
                )
            )
            # Recover is-None refinement from the predicate when the else edge
            # did not copy type_refinement (only the true edge carries it).
            is_none_guard = (
                refinement == "None"
                or (" is None" in pred or pred.endswith("is None") or "is None" in pred)
            )
            is_not_none_guard = refinement == "NonNull" or " is not None" in pred
            if guard_var and polarity and (refinement or is_none_guard or is_not_none_guard):
                nullability = Nullability.UNKNOWN
                refined_type = refinement
                if refinement == "None" or (is_none_guard and not is_not_none_guard and not refinement):
                    nullability = Nullability.NULLABLE
                    refined_type = "None"
                elif refinement == "NonNull" or is_not_none_guard:
                    nullability = Nullability.NONNULL
                    refined_type = ""
                elif refinement:
                    nullability = Nullability.NONNULL
                else:
                    nullability = Nullability.UNKNOWN
                if refined_type or nullability is not Nullability.UNKNOWN:
                    ref_id = _identity(
                        "vpg-ref",
                        {
                            "pc": condition_id,
                            "var": guard_var,
                            "type": refined_type,
                            "null": nullability.value,
                        },
                    )
                    refinements.append(
                        TypeRefinement(
                            refinement_id=ref_id,
                            variable=guard_var,
                            refined_type=refined_type,
                            nullability=nullability,
                            path_condition_id=condition_id,
                            procedure_id=proc.procedure_id,
                            status=ProvenanceStatus.PROVED,
                        )
                    )
            elif guard_var and not polarity and (is_none_guard or refinement == "None"):
                # else of `is None` → nonnull refinement
                ref_id = _identity(
                    "vpg-ref",
                    {
                        "pc": condition_id,
                        "var": guard_var,
                        "type": "",
                        "null": Nullability.NONNULL.value,
                    },
                )
                refinements.append(
                    TypeRefinement(
                        refinement_id=ref_id,
                        variable=guard_var,
                        refined_type="",
                        nullability=Nullability.NONNULL,
                        path_condition_id=condition_id,
                        procedure_id=proc.procedure_id,
                        status=ProvenanceStatus.PROVED,
                    )
                )
        return conditions, refinements

    # ------------------------------------------------------------------
    # Information provenance attachment
    # ------------------------------------------------------------------

    def _information_provenances(
        self,
        definitions: Sequence[ReachingDefinition],
        facets: Mapping[str, MemorySafetyFacetLike],
    ) -> list[InformationProvenance]:
        results: list[InformationProvenance] = []
        roots_id = self._roots.roots_id
        for item in definitions:
            origin = self._origin_for_kind(item.kind)
            labels = [origin.value, item.source_kind_detail or item.kind.value]
            if item.aliases:
                labels.extend(f"alias_of:{alias}" for alias in item.aliases)
            if item.field_path:
                labels.append(f"field:{item.field_path}")
            effects = tuple(
                str(e) for e in (item.attributes.get("effects") or ()) if str(e).strip()
            )
            concurrency = ConcurrencyKind.SEQUENTIAL
            if UnknownReason.CONCURRENCY.value in item.unknown_reasons:
                concurrency = ConcurrencyKind.UNSUPPORTED
            ownership = OwnershipKind.UNKNOWN
            mutation = MutationKind.UNKNOWN
            if item.kind is DefinitionKind.CONSTANT:
                ownership = OwnershipKind.OWNED
                mutation = MutationKind.IMMUTABLE
            elif item.kind is DefinitionKind.PARAMETER:
                ownership = OwnershipKind.BORROWED
                mutation = MutationKind.UNKNOWN
            elif item.kind in {
                DefinitionKind.CONSTRUCTOR,
                DefinitionKind.CONVERSION,
                DefinitionKind.CONFIG_SOURCE,
                DefinitionKind.DI_SOURCE,
            }:
                ownership = OwnershipKind.OWNED
                mutation = MutationKind.MUTABLE

            dep = DependencyDirection.DEFINES
            if item.kind is DefinitionKind.ALIAS:
                dep = DependencyDirection.ALIASES
            elif item.kind is DefinitionKind.CONSTRUCTOR:
                dep = DependencyDirection.CONSTRUCTS
            elif item.kind is DefinitionKind.CONVERSION:
                dep = DependencyDirection.CONVERTS
            elif item.kind is DefinitionKind.CONFIG_SOURCE:
                dep = DependencyDirection.CONFIGURES
            elif item.kind is DefinitionKind.DI_SOURCE:
                dep = DependencyDirection.INJECTS
            elif item.kind is DefinitionKind.RETURN:
                dep = DependencyDirection.RETURNS
            elif item.kind is DefinitionKind.FIELD_WRITE:
                dep = DependencyDirection.FIELD_OF
            elif item.kind is DefinitionKind.PARAMETER:
                dep = DependencyDirection.PARAMETER_OF

            facet_ref = ""
            facet_key = item.def_id
            if facet_key in facets:
                facet = facets[facet_key]
                disposition = getattr(getattr(facet, "disposition", None), "value", None) or str(
                    getattr(facet, "disposition", "") or ""
                )
                facet_ref = f"memory_safety:{disposition}"
            elif item.location.path in facets:
                facet = facets[item.location.path]
                disposition = getattr(getattr(facet, "disposition", None), "value", None) or str(
                    getattr(facet, "disposition", "") or ""
                )
                facet_ref = f"memory_safety:{disposition}"

            range_ref = ""
            if item.attributes.get("constant") is True:
                range_ref = "singleton"
            schema_ref = str(item.attributes.get("schema_ref") or "")

            status = item.status
            reasons = item.unknown_reasons
            provenance_id = _identity(
                "vpg-info",
                {
                    "def": item.def_id,
                    "origin": origin.value,
                    "type": item.type_annotation,
                    "roots": roots_id,
                },
            )
            results.append(
                InformationProvenance(
                    provenance_id=provenance_id,
                    def_id=item.def_id,
                    variable=item.variable,
                    origin_kind=origin,
                    type_ref=item.type_annotation,
                    schema_ref=schema_ref,
                    range_ref=range_ref,
                    nullability=(
                        Nullability.NONNULL
                        if item.kind is DefinitionKind.CONSTANT
                        else Nullability.UNKNOWN
                    ),
                    origin_labels=tuple(sorted(set(labels))),
                    effect_refs=effects,
                    capability_refs=(),
                    authorization_refs=(),
                    ownership=ownership,
                    lifetime_ref=f"proc:{item.procedure_id}",
                    mutation=mutation,
                    concurrency=concurrency,
                    dependency_direction=dep,
                    memory_safety_facet_ref=facet_ref,
                    status=status,
                    unknown_reasons=reasons,
                    producer_id=PRODUCER_ID,
                    roots_id=roots_id,
                )
            )
        return results

    def _origin_for_kind(self, kind: DefinitionKind) -> InformationOriginKind:
        return {
            DefinitionKind.PARAMETER: InformationOriginKind.PARAMETER,
            DefinitionKind.ASSIGNMENT: InformationOriginKind.LOCAL,
            DefinitionKind.ANN_ASSIGN: InformationOriginKind.LOCAL,
            DefinitionKind.AUG_ASSIGN: InformationOriginKind.LOCAL,
            DefinitionKind.FIELD_WRITE: InformationOriginKind.FIELD,
            DefinitionKind.RETURN: InformationOriginKind.RETURN_VALUE,
            DefinitionKind.CONSTRUCTOR: InformationOriginKind.CONSTRUCTOR,
            DefinitionKind.CONVERSION: InformationOriginKind.CONVERSION,
            DefinitionKind.CONFIG_SOURCE: InformationOriginKind.CONFIG,
            DefinitionKind.DI_SOURCE: InformationOriginKind.DI_REGISTRY,
            DefinitionKind.ALIAS: InformationOriginKind.ALIAS,
            DefinitionKind.IMPORT: InformationOriginKind.EXTERNAL,
            DefinitionKind.CONSTANT: InformationOriginKind.CONSTANT,
            DefinitionKind.CALL_RESULT: InformationOriginKind.EXTERNAL,
            DefinitionKind.COMPREHENSION: InformationOriginKind.LOCAL,
            DefinitionKind.EXCEPTION_BIND: InformationOriginKind.EXTERNAL,
            DefinitionKind.UNKNOWN: InformationOriginKind.UNKNOWN,
        }.get(kind, InformationOriginKind.UNKNOWN)

    # ------------------------------------------------------------------
    # Interprocedural threading
    # ------------------------------------------------------------------

    def _interprocedural_threads(
        self,
        proc: _ProcIR,
        uses: Sequence[ValueUse],
        definitions: Sequence[ReachingDefinition],
        reaching_in: Mapping[str, set[str]],
        sig_map: Mapping[str, _ProcIR],
    ) -> tuple[list[InterproceduralThread], list[UnknownFrontierFact]]:
        threads: list[InterproceduralThread] = []
        unknowns: list[UnknownFrontierFact] = []
        def_by_id = {item.def_id: item for item in definitions}

        # Walk call expressions in the original body for argument threading.
        for node in ast.walk(ast.Module(body=proc.body, type_ignores=[])):
            if not isinstance(node, ast.Call):
                continue
            func_name = _expr_name(node.func)
            simple = _simple_name(func_name)
            callee = sig_map.get(func_name) or sig_map.get(simple)
            # Find enclosing block by line proximity (best-effort).
            line = _lineno(node)
            block_id = ""
            # Uses of call args already recorded; match by line.
            if callee is None:
                # Incomplete interprocedural route.
                if not _is_conversion_call(func_name) and not _is_constructor_call(func_name):
                    if _is_reflection_call(func_name) or _is_native_call(func_name):
                        continue  # already frontiered elsewhere
                    unknowns.append(
                        UnknownFrontierFact(
                            fact_id=_identity(
                                "vpg-unknown",
                                {
                                    "reason": UnknownReason.INCOMPLETE_INTERPROCEDURAL.value,
                                    "proc": proc.procedure_id,
                                    "callee": func_name,
                                    "line": line,
                                },
                            ),
                            reason=UnknownReason.INCOMPLETE_INTERPROCEDURAL,
                            procedure_id=proc.procedure_id,
                            detail=f"missing callee body for {func_name}",
                            evidence_refs=(func_name,),
                        )
                    )
                continue

            # Depth check: same-module direct callee is depth 1.
            depth = 1
            if depth > self._max_ip_depth:
                unknowns.append(
                    UnknownFrontierFact(
                        fact_id=_identity(
                            "vpg-unknown",
                            {
                                "reason": UnknownReason.INCOMPLETE_INTERPROCEDURAL.value,
                                "proc": proc.procedure_id,
                                "callee": callee.procedure_id,
                                "depth": depth,
                            },
                        ),
                        reason=UnknownReason.INCOMPLETE_INTERPROCEDURAL,
                        procedure_id=proc.procedure_id,
                        detail=f"depth {depth} exceeds bound",
                    )
                )
                continue

            # Map positional args to callee parameters.
            callee_args = list(callee.args)
            # Skip self for methods when caller is method call.
            start = 0
            if callee.is_method and callee_args and callee_args[0].arg in {"self", "cls"}:
                # If call is attribute call, first formal is receiver.
                if isinstance(node.func, ast.Attribute):
                    start = 0  # receiver not in args
                    # Map receiver separately if Name
                    pass
                # Positional args map to args after self.
                formals = [a.arg for a in callee_args[1:]]
            else:
                formals = [a.arg for a in callee_args]

            for index, arg_node in enumerate(node.args):
                if index >= len(formals):
                    break
                formal = formals[index]
                actual_name = _expr_name(arg_node)
                if not actual_name or not actual_name.isidentifier():
                    # Complex expression: incomplete thread.
                    unknowns.append(
                        UnknownFrontierFact(
                            fact_id=_identity(
                                "vpg-unknown",
                                {
                                    "reason": UnknownReason.INCOMPLETE_INTERPROCEDURAL.value,
                                    "proc": proc.procedure_id,
                                    "formal": formal,
                                    "line": line,
                                },
                            ),
                            reason=UnknownReason.INCOMPLETE_INTERPROCEDURAL,
                            procedure_id=proc.procedure_id,
                            detail=f"non-name actual for {formal}",
                        )
                    )
                    continue
                # Find a reaching def for actual_name — approximate via any proved def.
                source_defs = [
                    item
                    for item in definitions
                    if item.variable == actual_name and item.status is ProvenanceStatus.PROVED
                ]
                if not source_defs:
                    unknowns.append(
                        UnknownFrontierFact(
                            fact_id=_identity(
                                "vpg-unknown",
                                {
                                    "reason": UnknownReason.INCOMPLETE_INTERPROCEDURAL.value,
                                    "proc": proc.procedure_id,
                                    "var": actual_name,
                                    "line": line,
                                },
                            ),
                            reason=UnknownReason.INCOMPLETE_INTERPROCEDURAL,
                            procedure_id=proc.procedure_id,
                            variable=actual_name,
                            detail="no proved source def for thread",
                        )
                    )
                    continue
                source = source_defs[0]
                completeness = InterproceduralCompleteness.COMPLETE
                status = ProvenanceStatus.PROVED
                reasons: tuple[str, ...] = ()
                # Without path-sensitive call-site reaching, mark partial when multiple defs.
                if len(source_defs) > 1:
                    completeness = InterproceduralCompleteness.PARTIAL
                    status = ProvenanceStatus.PARTIAL
                    reasons = (UnknownReason.MULTIPLE_REACHING.value,)
                thread_id = _identity(
                    "vpg-thread",
                    {
                        "src_def": source.def_id,
                        "src_proc": proc.procedure_id,
                        "tgt": callee.procedure_id,
                        "param": formal,
                        "line": line,
                    },
                )
                threads.append(
                    InterproceduralThread(
                        thread_id=thread_id,
                        source_def_id=source.def_id,
                        source_procedure_id=proc.procedure_id,
                        target_procedure_id=callee.procedure_id,
                        parameter_name=formal,
                        call_site_ref=f"{proc.path}:{line}:{func_name}",
                        completeness=completeness,
                        status=status,
                        unknown_reasons=reasons,
                        depth=depth,
                        dependency_direction=DependencyDirection.THREADS_TO,
                    )
                )

            # Keyword args.
            formal_set = set(formals)
            for kw in node.keywords:
                if not kw.arg or kw.arg not in formal_set:
                    if kw.arg is None:
                        unknowns.append(
                            UnknownFrontierFact(
                                fact_id=_identity(
                                    "vpg-unknown",
                                    {
                                        "reason": UnknownReason.INCOMPLETE_INTERPROCEDURAL.value,
                                        "proc": proc.procedure_id,
                                        "line": line,
                                        "splat": True,
                                    },
                                ),
                                reason=UnknownReason.INCOMPLETE_INTERPROCEDURAL,
                                procedure_id=proc.procedure_id,
                                detail="**kwargs splat",
                            )
                        )
                    continue
                actual_name = _expr_name(kw.value)
                if not actual_name or not actual_name.isidentifier():
                    continue
                source_defs = [
                    item
                    for item in definitions
                    if item.variable == actual_name and item.status is ProvenanceStatus.PROVED
                ]
                if not source_defs:
                    continue
                source = source_defs[0]
                thread_id = _identity(
                    "vpg-thread",
                    {
                        "src_def": source.def_id,
                        "src_proc": proc.procedure_id,
                        "tgt": callee.procedure_id,
                        "param": kw.arg,
                        "line": line,
                        "kw": True,
                    },
                )
                threads.append(
                    InterproceduralThread(
                        thread_id=thread_id,
                        source_def_id=source.def_id,
                        source_procedure_id=proc.procedure_id,
                        target_procedure_id=callee.procedure_id,
                        parameter_name=kw.arg,
                        call_site_ref=f"{proc.path}:{line}:{func_name}",
                        completeness=InterproceduralCompleteness.COMPLETE,
                        status=ProvenanceStatus.PROVED,
                        depth=depth,
                        dependency_direction=DependencyDirection.THREADS_TO,
                    )
                )

        return threads, unknowns

    def _scan_unsupported_shapes(
        self, proc: _ProcIR, blocks: Sequence[_MutableBlock]
    ) -> list[UnknownFrontierFact]:
        facts: list[UnknownFrontierFact] = []
        for block in blocks:
            if block.shape_support is CfgShapeSupport.UNSUPPORTED:
                facts.append(
                    UnknownFrontierFact(
                        fact_id=_identity(
                            "vpg-unknown",
                            {
                                "reason": UnknownReason.UNSUPPORTED_CFG.value,
                                "block": block.block_id,
                            },
                        ),
                        reason=UnknownReason.UNSUPPORTED_CFG,
                        procedure_id=proc.procedure_id,
                        block_id=block.block_id,
                        detail="unsupported cfg shape",
                        evidence_refs=tuple(block.unknown_reasons),
                    )
                )
        return facts

    def _scan_call_site_unknowns(
        self, proc: _ProcIR, blocks: Sequence[_MutableBlock]
    ) -> list[UnknownFrontierFact]:
        """Emit frontier facts for reflection / native / concurrency call sites."""
        facts: list[UnknownFrontierFact] = []
        for node in ast.walk(ast.Module(body=list(proc.body), type_ignores=[])):
            if not isinstance(node, ast.Call):
                continue
            func_name = _expr_name(node.func)
            line = _lineno(node)
            reason: UnknownReason | None = None
            if _is_reflection_call(func_name):
                reason = UnknownReason.REFLECTION
            elif _is_native_call(func_name):
                reason = UnknownReason.NATIVE_CALL
            elif _is_concurrency_call(func_name):
                reason = UnknownReason.CONCURRENCY
            if reason is None:
                continue
            facts.append(
                UnknownFrontierFact(
                    fact_id=_identity(
                        "vpg-unknown",
                        {
                            "reason": reason.value,
                            "proc": proc.procedure_id,
                            "callee": func_name,
                            "line": line,
                        },
                    ),
                    reason=reason,
                    procedure_id=proc.procedure_id,
                    detail=func_name,
                    evidence_refs=(f"{proc.path}:{line}:{func_name}",),
                )
            )
        return facts


# ---------------------------------------------------------------------------
# Public factory helpers
# ---------------------------------------------------------------------------


def build_value_provenance_graph(
    roots: ProgramGraphRoots | Mapping[str, Any],
    files: Mapping[str, str],
    *,
    max_loop_unroll: int = DEFAULT_MAX_LOOP_UNROLL,
    max_interprocedural_depth: int = DEFAULT_MAX_INTERPROCEDURAL_DEPTH,
    dependency_graph: ProgramDependencyGraphLike | None = None,
    call_resolver: ProgramCallResolverLike | None = None,
    memory_safety_facets: Mapping[str, MemorySafetyFacetLike] | None = None,
) -> ValueProvenanceGraph:
    """Compile a :class:`ValueProvenanceGraph` from path→source files."""
    compiler = ValueProvenanceCompiler(
        roots,
        max_loop_unroll=max_loop_unroll,
        max_interprocedural_depth=max_interprocedural_depth,
        dependency_graph=dependency_graph,
        call_resolver=call_resolver,
    )
    return compiler.compile_sources(files, memory_safety_facets=memory_safety_facets)


def compile_value_provenance(
    roots: ProgramGraphRoots | Mapping[str, Any],
    source: str,
    *,
    path: str = "snippet.py",
    procedure_name: str = "",
    **kwargs: Any,
) -> ValueProvenanceGraph:
    """Compile provenance for one source snippet (optionally one procedure)."""
    compiler = ValueProvenanceCompiler(roots, **{
        k: v
        for k, v in kwargs.items()
        if k
        in {
            "max_loop_unroll",
            "max_interprocedural_depth",
            "dependency_graph",
            "call_resolver",
        }
    })
    facets = kwargs.get("memory_safety_facets")
    return compiler.compile_procedure(
        source,
        path=path,
        procedure_name=procedure_name,
        memory_safety_facets=facets,
    )


__all__ = [
    "VALUE_PROVENANCE_GRAPH_SCHEMA",
    "VALUE_PROVENANCE_GRAPH_VERSION",
    "PRODUCER_ID",
    "ValueProvenanceError",
    "ValueProvenanceBoundsError",
    "ValueProvenanceAuthorityError",
    "ValueProvenanceUnsupportedError",
    "DefinitionKind",
    "UseKind",
    "DominanceKind",
    "ProvenanceStatus",
    "UnknownReason",
    "DependencyDirection",
    "InformationOriginKind",
    "Nullability",
    "MutationKind",
    "OwnershipKind",
    "ConcurrencyKind",
    "InterproceduralCompleteness",
    "CfgShapeSupport",
    "SourceLocation",
    "CfgBlock",
    "ReachingDefinition",
    "ValueUse",
    "DefUseChain",
    "DominanceFact",
    "PathCondition",
    "TypeRefinement",
    "InformationProvenance",
    "InterproceduralThread",
    "UnknownFrontierFact",
    "ValueProvenanceGraph",
    "ValueProvenanceCompiler",
    "build_value_provenance_graph",
    "compile_value_provenance",
]
