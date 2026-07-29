"""Cold, symbolic extraction of Python MCP and MCP++ surfaces.

The extractor intentionally operates on source text and paths only.  It never
imports a provider module, resolves an entry point, starts a server, or performs
network I/O.  Runtime ``tools/list`` observations are accepted through a
separate, capability-bound evidence API and never replace static evidence.

The supported static idioms include FastMCP decorators, MCP protocol
``list_tools``/``call_tool`` decorators, and common ``add_tool`` /
``register_tool`` calls.  When a registration name, schema, or handler depends
on runtime execution, the registration is retained as unresolved evidence
rather than being silently omitted.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Final, Iterable, Mapping, Sequence


PYTHON_MCP_SURFACE_EXTRACTOR_INTERFACE: Final = "PythonMcpSurfaceExtractor@1"
PYTHON_MCP_SURFACE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/python-mcp-surface@1"
)
PYTHON_MCP_TOOL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/python-mcp-tool@1"
)
PYTHON_MCP_UNRESOLVED_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/python-mcp-unresolved@1"
)
PYTHON_MCP_LIVE_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/python-mcp-live-tools-list@1"
)

DEFAULT_MAX_FILES = 20_000
DEFAULT_MAX_FILE_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 128 * 1024 * 1024

_REGISTRATION_METHODS = frozenset(
    {
        "add_tool",
        "register_tool",
        "register_mcp_tool",
        "tool",
    }
)
_PROTOCOL_DECORATORS = frozenset({"list_tools", "call_tool"})
_FACADE_NAMES = frozenset(
    {
        "list_categories",
        "list_tools",
        "get_schema",
        "dispatch",
        "tools.list_categories",
        "tools.list_tools",
        "tools.get_schema",
        "tools.dispatch",
    }
)
_POLICY_MARKERS = (
    "authorize",
    "authorization",
    "check_access",
    "check_capability",
    "check_permission",
    "enforce_policy",
    "permission",
    "policy",
    "require_capability",
    "validate_capability",
    "verify_capability",
    "ucan",
)
_TRANSPORT_MARKERS = {
    "stdio": "stdio",
    "sse": "sse",
    "streamable_http": "streamable_http",
    "websocket": "websocket",
    "web_socket": "websocket",
    "libp2p": "libp2p",
    "p2p": "libp2p",
    "http": "http",
}


class PythonMcpSurfaceError(ValueError):
    """Raised for malformed or unsafe extractor inputs."""


class ToolSurfaceKind(str, Enum):
    """The role of a statically visible MCP surface."""

    DOMAIN = "domain_tool"
    FACADE_META = "facade_meta_tool"
    DISCOVERY_HANDLER = "tools_list_handler"
    INVOCATION_HANDLER = "tools_call_handler"


class ResolutionState(str, Enum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


class UnresolvedReason(str, Enum):
    DYNAMIC_NAME = "dynamic_name"
    DYNAMIC_HANDLER = "dynamic_handler"
    DYNAMIC_SCHEMA = "dynamic_schema"
    DYNAMIC_DISCOVERY = "dynamic_discovery"
    PARSE_ERROR = "parse_error"
    READ_ERROR = "read_error"
    RESOURCE_LIMIT = "resource_limit"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _identity(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def _source_sha256(source: str) -> str:
    return "sha256:" + hashlib.sha256(
        source.encode("utf-8", errors="surrogatepass")
    ).hexdigest()


def _clean_path(path: str | os.PathLike[str]) -> str:
    raw = str(path).replace("\\", "/")
    normalized = PurePosixPath(raw).as_posix()
    if normalized in {"", "."}:
        raise PythonMcpSurfaceError("source path must be non-empty")
    if PurePosixPath(normalized).is_absolute() or ".." in PurePosixPath(normalized).parts:
        raise PythonMcpSurfaceError("source path must be relative and traversal-free")
    return normalized


def _json_value(value: Any) -> Any:
    """Return a deterministic JSON value or raise for Python-only literals."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PythonMcpSurfaceError("non-finite schema number is not JSON")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise PythonMcpSurfaceError("schema object keys must be strings")
        return {key: _json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise PythonMcpSurfaceError(
        f"schema contains non-JSON value {type(value).__name__}"
    )


def _json_mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        normalized = _json_value(value)
    except PythonMcpSurfaceError:
        return None
    return normalized if isinstance(normalized, dict) else None


@dataclass(frozen=True)
class SourceSpan:
    """Exact source coordinates for one extracted observation."""

    path: str
    source_sha256: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _clean_path(self.path))
        if not self.source_sha256.startswith("sha256:"):
            raise PythonMcpSurfaceError("source_sha256 must be sha256-prefixed")
        if self.start_line < 1 or self.end_line < self.start_line:
            raise PythonMcpSurfaceError("invalid source line span")
        if self.start_column < 0 or self.end_column < 0:
            raise PythonMcpSurfaceError("invalid source column span")

    @classmethod
    def from_node(cls, path: str, source_hash: str, node: ast.AST) -> "SourceSpan":
        return cls(
            path=path,
            source_sha256=source_hash,
            start_line=int(getattr(node, "lineno", 1)),
            start_column=int(getattr(node, "col_offset", 0)),
            end_line=int(getattr(node, "end_lineno", getattr(node, "lineno", 1))),
            end_column=int(getattr(node, "end_col_offset", getattr(node, "col_offset", 0))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "source_sha256": self.source_sha256,
            "start_line": self.start_line,
            "start_column": self.start_column,
            "end_line": self.end_line,
            "end_column": self.end_column,
        }


@dataclass(frozen=True)
class HandlerReachability:
    """A registered handler and its statically visible implementation calls."""

    symbol: str
    state: ResolutionState
    span: SourceSpan | None = None
    calls: tuple[str, ...] = ()
    policy_gates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", ResolutionState(self.state))
        object.__setattr__(self, "symbol", str(self.symbol or "").strip())
        object.__setattr__(self, "calls", tuple(sorted(set(self.calls))))
        object.__setattr__(
            self, "policy_gates", tuple(sorted(set(self.policy_gates)))
        )
        if self.state is ResolutionState.RESOLVED and not self.symbol:
            raise PythonMcpSurfaceError("resolved handler requires a symbol")

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "state": self.state.value,
            "span": self.span.to_dict() if self.span else None,
            "calls": list(self.calls),
            "policy_gates": list(self.policy_gates),
        }


@dataclass(frozen=True)
class PythonMcpToolSurface:
    """One static tool, facade, or MCP protocol handler."""

    provider: str
    declared_name: str
    canonical_name: str
    kind: ToolSurfaceKind
    registration_api: str
    registration_span: SourceSpan
    handler: HandlerReachability
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    schema_state: ResolutionState = ResolutionState.RESOLVED
    aliases: tuple[str, ...] = ()
    description: str = ""
    transports: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("provider", "declared_name", "canonical_name", "registration_api"):
            if not str(getattr(self, name) or "").strip():
                raise PythonMcpSurfaceError(f"{name} is required")
        object.__setattr__(self, "kind", ToolSurfaceKind(self.kind))
        object.__setattr__(self, "schema_state", ResolutionState(self.schema_state))
        normalized_schema = _json_mapping(self.input_schema)
        if normalized_schema is None:
            raise PythonMcpSurfaceError("input_schema must be a JSON object")
        object.__setattr__(self, "input_schema", normalized_schema)
        object.__setattr__(self, "aliases", tuple(sorted(set(self.aliases))))
        object.__setattr__(self, "transports", tuple(sorted(set(self.transports))))

    @property
    def tool_id(self) -> str:
        return _identity(
            "python-mcp-tool",
            {
                "provider": self.provider,
                "name": self.canonical_name,
                "kind": self.kind.value,
                "handler": self.handler.symbol,
                "registration": self.registration_span.to_dict(),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PYTHON_MCP_TOOL_SCHEMA,
            "tool_id": self.tool_id,
            "provider": self.provider,
            "declared_name": self.declared_name,
            "canonical_name": self.canonical_name,
            "kind": self.kind.value,
            "registration_api": self.registration_api,
            "registration_span": self.registration_span.to_dict(),
            "handler": self.handler.to_dict(),
            "input_schema": dict(self.input_schema),
            "schema_state": self.schema_state.value,
            "aliases": list(self.aliases),
            "description": self.description,
            "transports": list(self.transports),
        }


@dataclass(frozen=True)
class UnresolvedRegistration:
    """A registration-shaped construct that requires runtime resolution."""

    provider: str
    reason: UnresolvedReason
    expression: str
    registration_api: str
    span: SourceSpan
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", UnresolvedReason(self.reason))

    @property
    def unresolved_id(self) -> str:
        return _identity(
            "python-mcp-unresolved",
            {
                "provider": self.provider,
                "reason": self.reason.value,
                "expression": self.expression,
                "span": self.span.to_dict(),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PYTHON_MCP_UNRESOLVED_SCHEMA,
            "unresolved_id": self.unresolved_id,
            "provider": self.provider,
            "reason": self.reason.value,
            "expression": self.expression,
            "registration_api": self.registration_api,
            "span": self.span.to_dict(),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class LiveDiscoveryCapability:
    """Explicit authority binding for one externally obtained tools/list fixture."""

    capability_id: str
    provider: str
    transport: str
    endpoint_identity: str
    repository_tree_id: str

    def __post_init__(self) -> None:
        for name in (
            "capability_id",
            "provider",
            "transport",
            "endpoint_identity",
            "repository_tree_id",
        ):
            if not str(getattr(self, name) or "").strip():
                raise PythonMcpSurfaceError(f"{name} is required")


@dataclass(frozen=True)
class LiveToolsListEvidence:
    """Non-authoritative runtime observation, separate from the static surface."""

    capability_id: str
    provider: str
    transport: str
    endpoint_identity: str
    repository_tree_id: str
    tools: tuple[Mapping[str, Any], ...]
    fixture_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PYTHON_MCP_LIVE_EVIDENCE_SCHEMA,
            "authority": "observation",
            "capability_id": self.capability_id,
            "provider": self.provider,
            "transport": self.transport,
            "endpoint_identity": self.endpoint_identity,
            "repository_tree_id": self.repository_tree_id,
            "fixture_sha256": self.fixture_sha256,
            "tools": [dict(tool) for tool in self.tools],
        }


@dataclass(frozen=True)
class PythonMcpPackageSurface:
    """Deterministic static extraction result for one Python package."""

    provider: str
    repository_tree_id: str
    tools: tuple[PythonMcpToolSurface, ...]
    unresolved: tuple[UnresolvedRegistration, ...]
    source_files: tuple[Mapping[str, str], ...]

    @property
    def surface_id(self) -> str:
        return _identity(
            "python-mcp-surface",
            {
                "interface": PYTHON_MCP_SURFACE_EXTRACTOR_INTERFACE,
                "provider": self.provider,
                "repository_tree_id": self.repository_tree_id,
                "tools": [tool.to_dict() for tool in self.tools],
                "unresolved": [item.to_dict() for item in self.unresolved],
                "source_files": [dict(item) for item in self.source_files],
            },
        )

    def tools_named(self, name: str) -> tuple[PythonMcpToolSurface, ...]:
        selected = _canonical_tool_name(name)
        return tuple(
            tool
            for tool in self.tools
            if tool.canonical_name == selected or selected in tool.aliases
        )

    @property
    def domain_tools(self) -> tuple[PythonMcpToolSurface, ...]:
        return tuple(tool for tool in self.tools if tool.kind is ToolSurfaceKind.DOMAIN)

    @property
    def facade_tools(self) -> tuple[PythonMcpToolSurface, ...]:
        return tuple(
            tool for tool in self.tools if tool.kind is ToolSurfaceKind.FACADE_META
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PYTHON_MCP_SURFACE_SCHEMA,
            "interface": PYTHON_MCP_SURFACE_EXTRACTOR_INTERFACE,
            "surface_id": self.surface_id,
            "authority": "static_observation",
            "provider": self.provider,
            "repository_tree_id": self.repository_tree_id,
            "tools": [tool.to_dict() for tool in self.tools],
            "unresolved": [item.to_dict() for item in self.unresolved],
            "source_files": [dict(item) for item in self.source_files],
        }

    def catalog_registration_records(self) -> tuple[dict[str, Any], ...]:
        """Project tools to registration-tier McpContractCatalog source inputs."""

        return tuple(
            {
                "kind": "registration",
                "authority_class": "registration",
                "subject": f"tool:{tool.canonical_name}",
                "provider": self.provider,
                "repository_tree_id": self.repository_tree_id,
                "payload_fingerprint": tool.tool_id,
                "path": tool.registration_span.path,
                "source_span": tool.registration_span.to_dict(),
            }
            for tool in self.tools
        )


def bind_live_tools_list(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    capability: LiveDiscoveryCapability,
    provider: str | None = None,
    transport: str | None = None,
    endpoint_identity: str | None = None,
    repository_tree_id: str | None = None,
) -> LiveToolsListEvidence:
    """Bind a supplied tools/list fixture to explicit runtime capability facts."""

    if not isinstance(capability, LiveDiscoveryCapability):
        raise PythonMcpSurfaceError(
            "live tools/list evidence requires LiveDiscoveryCapability"
        )
    expected = {
        "provider": capability.provider,
        "transport": capability.transport,
        "endpoint_identity": capability.endpoint_identity,
        "repository_tree_id": capability.repository_tree_id,
    }
    supplied = {
        "provider": provider,
        "transport": transport,
        "endpoint_identity": endpoint_identity,
        "repository_tree_id": repository_tree_id,
    }
    for key, value in supplied.items():
        if value is not None and str(value) != expected[key]:
            raise PythonMcpSurfaceError(
                f"live tools/list {key} is outside the granted capability"
            )

    value: Any = payload
    if isinstance(value, Mapping) and "result" in value:
        value = value["result"]
    if isinstance(value, Mapping) and "tools" in value:
        value = value["tools"]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise PythonMcpSurfaceError("tools/list fixture must contain a tools sequence")
    tools: list[Mapping[str, Any]] = []
    for index, tool in enumerate(value):
        if not isinstance(tool, Mapping):
            raise PythonMcpSurfaceError(f"tools/list item {index} must be an object")
        name = tool.get("name")
        if not isinstance(name, str) or not name.strip():
            raise PythonMcpSurfaceError(f"tools/list item {index} requires a name")
        try:
            normalized = _json_value(tool)
        except PythonMcpSurfaceError as exc:
            raise PythonMcpSurfaceError(
                f"tools/list item {index} is not JSON-compatible"
            ) from exc
        if not isinstance(normalized, Mapping):
            raise PythonMcpSurfaceError(f"tools/list item {index} must be an object")
        tools.append(dict(normalized))
    tools.sort(key=lambda item: str(item["name"]))
    canonical = _canonical_json(tools)
    return LiveToolsListEvidence(
        capability_id=capability.capability_id,
        provider=capability.provider,
        transport=capability.transport,
        endpoint_identity=capability.endpoint_identity,
        repository_tree_id=capability.repository_tree_id,
        tools=tuple(tools),
        fixture_sha256="sha256:"
        + hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def _dotted(node: ast.AST | None) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _dotted(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    if isinstance(node, ast.Call):
        return _dotted(node.func)
    if isinstance(node, ast.Subscript):
        return _dotted(node.value)
    return ""


def _render(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def _literal(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None


def _string_literal(node: ast.AST | None) -> str:
    value = _literal(node)
    return value if isinstance(value, str) else ""


def _canonical_tool_name(name: str) -> str:
    result = str(name or "").strip().replace("/", ".")
    while ".." in result:
        result = result.replace("..", ".")
    result = result.strip(".")
    hierarchical_aliases = {
        "tools_list_categories": "tools.list_categories",
        "tools_list_tools": "tools.list_tools",
        "tools_get_schema": "tools.get_schema",
        "tools_dispatch": "tools.dispatch",
    }
    return hierarchical_aliases.get(result, result)


def _call_name(call: ast.Call) -> str:
    return _dotted(call.func)


def _keyword(call: ast.Call, *names: str) -> ast.AST | None:
    selected = set(names)
    for item in call.keywords:
        if item.arg in selected:
            return item.value
    return None


def _annotation(node: ast.AST | None) -> str:
    return _render(node) if node is not None else "Any"


def _signature_schema(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    arguments = list(node.args.posonlyargs) + list(node.args.args)
    defaults: list[ast.AST | None] = [None] * (
        len(arguments) - len(node.args.defaults)
    ) + list(node.args.defaults)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for argument, default in zip(arguments, defaults):
        if argument.arg in {"self", "cls", "ctx", "context"}:
            continue
        entry: dict[str, Any] = {"python_annotation": _annotation(argument.annotation)}
        if default is None:
            required.append(argument.arg)
        else:
            literal = _literal(default)
            if literal is not None or isinstance(default, ast.Constant):
                entry["default"] = literal
            else:
                entry["default_expression"] = _render(default)
        properties[argument.arg] = entry
    if node.args.vararg:
        properties[node.args.vararg.arg] = {
            "python_annotation": _annotation(node.args.vararg.annotation),
            "variadic": "positional",
        }
    for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        entry = {"python_annotation": _annotation(argument.annotation)}
        if default is None:
            required.append(argument.arg)
        else:
            literal = _literal(default)
            if literal is not None or isinstance(default, ast.Constant):
                entry["default"] = literal
            else:
                entry["default_expression"] = _render(default)
        properties[argument.arg] = entry
    if node.args.kwarg:
        properties[node.args.kwarg.arg] = {
            "python_annotation": _annotation(node.args.kwarg.annotation),
            "variadic": "keyword",
        }
    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": sorted(required),
    }
    if node.returns is not None:
        result["python_return_annotation"] = _annotation(node.returns)
    return result


def _calls_in(node: ast.AST) -> tuple[str, ...]:
    calls = {
        name
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        for name in (_call_name(child),)
        if name
    }
    return tuple(sorted(calls))


def _policy_calls(calls: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                call
                for call in calls
                if any(marker in call.casefold() for marker in _POLICY_MARKERS)
            }
        )
    )


def _transports_in(node: ast.AST) -> tuple[str, ...]:
    rendered = _render(node).casefold()
    return tuple(
        sorted(
            {
                transport
                for marker, transport in _TRANSPORT_MARKERS.items()
                if marker in rendered
            }
        )
    )


def _kind_for(name: str, protocol: str = "") -> ToolSurfaceKind:
    if protocol == "list_tools":
        return ToolSurfaceKind.DISCOVERY_HANDLER
    if protocol == "call_tool":
        return ToolSurfaceKind.INVOCATION_HANDLER
    canonical = _canonical_tool_name(name)
    if canonical in _FACADE_NAMES or (
        canonical.startswith("tools.")
        and canonical.rsplit(".", 1)[-1]
        in {"list_categories", "list_tools", "get_schema", "dispatch"}
    ):
        return ToolSurfaceKind.FACADE_META
    return ToolSurfaceKind.DOMAIN


@dataclass
class _FunctionFact:
    node: ast.FunctionDef | ast.AsyncFunctionDef
    symbol: str
    span: SourceSpan
    calls: tuple[str, ...]
    policy_gates: tuple[str, ...]


class _FunctionCollector(ast.NodeVisitor):
    def __init__(self, path: str, source_hash: str) -> None:
        self.path = path
        self.source_hash = source_hash
        self.scope: list[str] = []
        self.by_simple_name: dict[str, list[_FunctionFact]] = {}
        self.by_node: dict[int, _FunctionFact] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        symbol = ".".join((*self.scope, node.name))
        calls = _calls_in(node)
        fact = _FunctionFact(
            node=node,
            symbol=symbol,
            span=SourceSpan.from_node(self.path, self.source_hash, node),
            calls=calls,
            policy_gates=_policy_calls(calls),
        )
        self.by_node[id(node)] = fact
        self.by_simple_name.setdefault(node.name, []).append(fact)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function


class _StaticFileExtractor:
    def __init__(self, provider: str, path: str, source: str) -> None:
        self.provider = provider
        self.path = path
        self.source = source
        self.source_hash = _source_sha256(source)
        self.tools: list[PythonMcpToolSurface] = []
        self.unresolved: list[UnresolvedRegistration] = []
        self._decorator_call_ids: set[int] = set()

    def extract(self) -> tuple[list[PythonMcpToolSurface], list[UnresolvedRegistration]]:
        try:
            tree = ast.parse(self.source, filename=self.path, type_comments=True)
        except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
            line = max(1, int(getattr(exc, "lineno", 1) or 1))
            column = max(0, int(getattr(exc, "offset", 1) or 1) - 1)
            span = SourceSpan(
                self.path, self.source_hash, line, column, line, column
            )
            self.unresolved.append(
                UnresolvedRegistration(
                    provider=self.provider,
                    reason=UnresolvedReason.PARSE_ERROR,
                    expression="",
                    registration_api="ast.parse",
                    span=span,
                    detail=str(exc),
                )
            )
            return self.tools, self.unresolved

        functions = _FunctionCollector(self.path, self.source_hash)
        functions.visit(tree)
        module_transports = _transports_in(tree)
        self._decorator_call_ids = {
            id(decorator)
            for fact in functions.by_node.values()
            for decorator in fact.node.decorator_list
            if isinstance(decorator, ast.Call)
        }

        for fact in functions.by_node.values():
            self._extract_decorators(fact, functions, module_transports)
            self._extract_protocol_branches(fact, functions, module_transports)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self._extract_registration_call(
                    node, functions, module_transports
                )
        self._extract_dynamic_discovery(tree)
        return self.tools, self.unresolved

    def _handler(
        self,
        expression: ast.AST | None,
        functions: _FunctionCollector,
        *,
        decorated: _FunctionFact | None = None,
    ) -> HandlerReachability:
        if decorated is not None:
            return HandlerReachability(
                symbol=decorated.symbol,
                state=ResolutionState.RESOLVED,
                span=decorated.span,
                calls=decorated.calls,
                policy_gates=decorated.policy_gates,
            )
        symbol = _dotted(expression)
        simple = symbol.rsplit(".", 1)[-1] if symbol else ""
        matches = functions.by_simple_name.get(simple, ())
        if len(matches) == 1:
            fact = matches[0]
            return HandlerReachability(
                symbol=fact.symbol,
                state=ResolutionState.RESOLVED,
                span=fact.span,
                calls=fact.calls,
                policy_gates=fact.policy_gates,
            )
        if symbol:
            return HandlerReachability(symbol=symbol, state=ResolutionState.RESOLVED)
        return HandlerReachability(symbol=_render(expression), state=ResolutionState.UNRESOLVED)

    def _add_tool(
        self,
        *,
        name: str,
        registration_api: str,
        registration_node: ast.AST,
        handler: HandlerReachability,
        input_schema: Mapping[str, Any],
        schema_state: ResolutionState,
        description: str = "",
        aliases: Sequence[str] = (),
        protocol: str = "",
        transports: Sequence[str] = (),
    ) -> None:
        canonical = _canonical_tool_name(name)
        self.tools.append(
            PythonMcpToolSurface(
                provider=self.provider,
                declared_name=name,
                canonical_name=canonical,
                kind=_kind_for(canonical, protocol),
                registration_api=registration_api,
                registration_span=SourceSpan.from_node(
                    self.path, self.source_hash, registration_node
                ),
                handler=handler,
                input_schema=input_schema,
                schema_state=schema_state,
                aliases=tuple(_canonical_tool_name(alias) for alias in aliases if alias),
                description=description,
                transports=tuple(transports),
            )
        )

    def _extract_decorators(
        self,
        fact: _FunctionFact,
        functions: _FunctionCollector,
        module_transports: Sequence[str],
    ) -> None:
        node = fact.node
        for decorator in node.decorator_list:
            call = decorator if isinstance(decorator, ast.Call) else None
            dotted = _dotted(call.func if call else decorator)
            method = dotted.rsplit(".", 1)[-1]
            if method not in _REGISTRATION_METHODS | _PROTOCOL_DECORATORS:
                continue
            if method in _PROTOCOL_DECORATORS:
                name = method
                protocol = method
            else:
                name_node = (
                    _keyword(call, "name", "tool_name")
                    if call
                    else None
                )
                if name_node is None and call and call.args:
                    name_node = call.args[0]
                name = _string_literal(name_node)
                if name_node is not None and not name:
                    self.unresolved.append(
                        UnresolvedRegistration(
                            provider=self.provider,
                            reason=UnresolvedReason.DYNAMIC_NAME,
                            expression=_render(decorator),
                            registration_api=dotted,
                            span=SourceSpan.from_node(
                                self.path, self.source_hash, decorator
                            ),
                            detail="decorator tool name requires runtime evaluation",
                        )
                    )
                    continue
                name = name or node.name
                protocol = ""
            schema_node = _keyword(call, "input_schema", "schema", "parameters") if call else None
            literal_schema = _json_mapping(_literal(schema_node))
            schema = (
                literal_schema
                if literal_schema is not None
                else _signature_schema(node)
            )
            schema_state = (
                ResolutionState.UNRESOLVED
                if schema_node is not None and literal_schema is None
                else ResolutionState.RESOLVED
            )
            alias_value = _literal(_keyword(call, "aliases", "alias")) if call else None
            aliases = (
                [alias_value]
                if isinstance(alias_value, str)
                else list(alias_value)
                if isinstance(alias_value, (list, tuple))
                and all(isinstance(item, str) for item in alias_value)
                else []
            )
            self._add_tool(
                name=name,
                registration_api=dotted,
                registration_node=decorator,
                handler=self._handler(None, functions, decorated=fact),
                input_schema=schema,
                schema_state=schema_state,
                description=ast.get_docstring(node) or "",
                aliases=aliases,
                protocol=protocol,
                transports=module_transports,
            )

    def _extract_protocol_branches(
        self,
        fact: _FunctionFact,
        functions: _FunctionCollector,
        module_transports: Sequence[str],
    ) -> None:
        """Extract direct JSON-RPC tools/list and tools/call dispatch branches."""

        for candidate in ast.walk(fact.node):
            if not isinstance(candidate, ast.If):
                continue
            protocol_methods: set[str] = set()
            for comparison in ast.walk(candidate.test):
                if not isinstance(comparison, ast.Compare):
                    continue
                expressions = (comparison.left, *comparison.comparators)
                for expression in expressions:
                    literal = _literal(expression)
                    if isinstance(literal, str):
                        values = (literal,)
                    elif isinstance(literal, (tuple, list)):
                        values = tuple(
                            value for value in literal if isinstance(value, str)
                        )
                    else:
                        values = ()
                    protocol_methods.update(
                        value for value in values if value in {"tools/list", "tools/call"}
                    )
            for protocol_method in sorted(protocol_methods):
                protocol = (
                    "list_tools"
                    if protocol_method == "tools/list"
                    else "call_tool"
                )
                self._add_tool(
                    name=protocol_method,
                    registration_api="jsonrpc.method_branch",
                    registration_node=candidate.test,
                    handler=self._handler(None, functions, decorated=fact),
                    input_schema=_signature_schema(fact.node),
                    schema_state=ResolutionState.RESOLVED,
                    protocol=protocol,
                    transports=module_transports,
                )

    def _extract_registration_call(
        self,
        call: ast.Call,
        functions: _FunctionCollector,
        module_transports: Sequence[str],
    ) -> None:
        dotted = _call_name(call)
        method = dotted.rsplit(".", 1)[-1]
        if method not in _REGISTRATION_METHODS or id(call) in self._decorator_call_ids:
            return
        name_node = _keyword(call, "name", "tool_name")
        handler_node = _keyword(call, "handler", "func", "function", "callback")
        schema_node = _keyword(call, "schema", "input_schema", "parameters")

        if method == "add_tool":
            if handler_node is None and call.args:
                handler_node = call.args[0]
            if name_node is None and len(call.args) > 1:
                name_node = call.args[1]
        elif method in {"register_tool", "register_mcp_tool"}:
            if call.args:
                first_literal = _literal(call.args[0])
                if isinstance(first_literal, str):
                    name_node = name_node or call.args[0]
                    if handler_node is None and len(call.args) > 1:
                        handler_node = call.args[1]
                    if schema_node is None and len(call.args) > 3:
                        schema_node = call.args[3]
                elif isinstance(first_literal, Mapping):
                    schema_node = schema_node or call.args[0]
                    if handler_node is None and len(call.args) > 1:
                        handler_node = call.args[1]
                    schema_name = first_literal.get("name")
                    if isinstance(schema_name, str):
                        name_node = ast.Constant(schema_name)
                elif handler_node is None:
                    handler_node = call.args[0]
        else:  # direct ``mcp.tool(handler, ...)``
            if handler_node is None and call.args:
                handler_node = call.args[0]

        name = _string_literal(name_node)
        if name_node is not None and not name:
            self.unresolved.append(
                UnresolvedRegistration(
                    provider=self.provider,
                    reason=UnresolvedReason.DYNAMIC_NAME,
                    expression=_render(call),
                    registration_api=dotted,
                    span=SourceSpan.from_node(self.path, self.source_hash, call),
                    detail="explicit registration name requires runtime evaluation",
                )
            )
            return
        if not name and handler_node is not None:
            handler_symbol = _dotted(handler_node)
            # FastMCP add_tool(handler) defaults to the callable name.
            if method in {"add_tool", "tool"} and handler_symbol:
                name = handler_symbol.rsplit(".", 1)[-1]
        if not name:
            self.unresolved.append(
                UnresolvedRegistration(
                    provider=self.provider,
                    reason=UnresolvedReason.DYNAMIC_NAME,
                    expression=_render(call),
                    registration_api=dotted,
                    span=SourceSpan.from_node(self.path, self.source_hash, call),
                    detail="registration name requires runtime evaluation",
                )
            )
            return

        handler = self._handler(handler_node, functions)
        if handler.state is ResolutionState.UNRESOLVED:
            self.unresolved.append(
                UnresolvedRegistration(
                    provider=self.provider,
                    reason=UnresolvedReason.DYNAMIC_HANDLER,
                    expression=_render(handler_node),
                    registration_api=dotted,
                    span=SourceSpan.from_node(self.path, self.source_hash, call),
                    detail=f"handler for {name!r} requires runtime evaluation",
                )
            )
        schema_literal = _json_mapping(_literal(schema_node))
        if schema_literal is not None:
            schema: Mapping[str, Any] = schema_literal
            schema_state = ResolutionState.RESOLVED
        else:
            simple = handler.symbol.rsplit(".", 1)[-1]
            matches = functions.by_simple_name.get(simple, ())
            if len(matches) == 1:
                schema = _signature_schema(matches[0].node)
                schema_state = ResolutionState.RESOLVED
            else:
                schema = {}
                schema_state = ResolutionState.UNRESOLVED
        if schema_node is not None and schema_literal is None:
            schema_state = ResolutionState.UNRESOLVED
            self.unresolved.append(
                UnresolvedRegistration(
                    provider=self.provider,
                    reason=UnresolvedReason.DYNAMIC_SCHEMA,
                    expression=_render(schema_node),
                    registration_api=dotted,
                    span=SourceSpan.from_node(self.path, self.source_hash, call),
                    detail=f"schema for {name!r} requires runtime evaluation",
                )
            )
        aliases_literal = _literal(_keyword(call, "aliases", "alias"))
        aliases = (
            [aliases_literal]
            if isinstance(aliases_literal, str)
            else list(aliases_literal)
            if isinstance(aliases_literal, (list, tuple))
            and all(isinstance(item, str) for item in aliases_literal)
            else []
        )
        self._add_tool(
            name=name,
            registration_api=dotted,
            registration_node=call,
            handler=handler,
            input_schema=schema,
            schema_state=schema_state,
            aliases=aliases,
            transports=module_transports,
        )

    def _extract_dynamic_discovery(self, tree: ast.Module) -> None:
        """Retain runtime module/filesystem discovery as unresolved registration."""

        seen: set[tuple[int, int]] = set()
        for node in ast.walk(tree):
            reason = ""
            if isinstance(node, ast.Call):
                dotted = _call_name(node).casefold()
                if any(
                    marker in dotted
                    for marker in (
                        "import_module",
                        "iter_modules",
                        "walk_packages",
                        "entry_points",
                    )
                ):
                    reason = dotted
                elif dotted.endswith((".glob", ".rglob")):
                    literal = _literal(node.args[0]) if node.args else None
                    if isinstance(literal, str) and ".py" in literal:
                        reason = dotted
            if not reason:
                continue
            coordinates = (
                int(getattr(node, "lineno", 1)),
                int(getattr(node, "col_offset", 0)),
            )
            if coordinates in seen:
                continue
            seen.add(coordinates)
            self.unresolved.append(
                UnresolvedRegistration(
                    provider=self.provider,
                    reason=UnresolvedReason.DYNAMIC_DISCOVERY,
                    expression=_render(node),
                    registration_api=reason,
                    span=SourceSpan.from_node(self.path, self.source_hash, node),
                    detail="runtime discovery may add MCP tools; static absence is not inferred",
                )
            )


def extract_python_mcp_source(
    source: str | bytes,
    *,
    provider: str,
    path: str,
    repository_tree_id: str = "",
) -> PythonMcpPackageSurface:
    """Extract a single Python source body without importing it."""

    if isinstance(source, bytes):
        try:
            source = source.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise PythonMcpSurfaceError("Python source must be UTF-8") from exc
    if not isinstance(source, str):
        raise PythonMcpSurfaceError("source must be text or UTF-8 bytes")
    provider = str(provider or "").strip()
    if not provider:
        raise PythonMcpSurfaceError("provider is required")
    path = _clean_path(path)
    tools, unresolved = _StaticFileExtractor(provider, path, source).extract()
    return _build_surface(
        provider=provider,
        repository_tree_id=repository_tree_id,
        tools=tools,
        unresolved=unresolved,
        source_files=({"path": path, "source_sha256": _source_sha256(source)},),
    )


def _build_surface(
    *,
    provider: str,
    repository_tree_id: str,
    tools: Iterable[PythonMcpToolSurface],
    unresolved: Iterable[UnresolvedRegistration],
    source_files: Iterable[Mapping[str, str]],
) -> PythonMcpPackageSurface:
    materialized_tools = list(tools)
    handler_names: dict[tuple[str, ToolSurfaceKind], set[str]] = {}
    for tool in materialized_tools:
        if tool.handler.symbol:
            handler_names.setdefault(
                (tool.handler.symbol, tool.kind), set()
            ).add(tool.canonical_name)
    unique_tools: dict[str, PythonMcpToolSurface] = {}
    for tool in materialized_tools:
        sibling_names = handler_names.get((tool.handler.symbol, tool.kind), set())
        inferred_aliases = tuple(
            sorted(
                (set(tool.aliases) | sibling_names)
                - {tool.canonical_name}
            )
        )
        if inferred_aliases != tool.aliases:
            tool = replace(tool, aliases=inferred_aliases)
        unique_tools[tool.tool_id] = tool
    unique_unresolved: dict[str, UnresolvedRegistration] = {}
    for item in unresolved:
        unique_unresolved[item.unresolved_id] = item
    return PythonMcpPackageSurface(
        provider=provider,
        repository_tree_id=str(repository_tree_id or ""),
        tools=tuple(
            sorted(
                unique_tools.values(),
                key=lambda item: (
                    item.canonical_name,
                    item.kind.value,
                    item.registration_span.path,
                    item.registration_span.start_line,
                ),
            )
        ),
        unresolved=tuple(
            sorted(
                unique_unresolved.values(),
                key=lambda item: (
                    item.span.path,
                    item.span.start_line,
                    item.reason.value,
                    item.expression,
                ),
            )
        ),
        source_files=tuple(
            sorted(
                (dict(item) for item in source_files),
                key=lambda item: item["path"],
            )
        ),
    )


class PythonMcpSurfaceExtractor:
    """Bounded filesystem facade for cold Python source extraction."""

    interface = PYTHON_MCP_SURFACE_EXTRACTOR_INTERFACE
    schema = PYTHON_MCP_SURFACE_SCHEMA

    def __init__(
        self,
        *,
        max_files: int = DEFAULT_MAX_FILES,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    ) -> None:
        for name, value in (
            ("max_files", max_files),
            ("max_file_bytes", max_file_bytes),
            ("max_total_bytes", max_total_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise PythonMcpSurfaceError(f"{name} must be a positive integer")
        if max_file_bytes > max_total_bytes:
            raise PythonMcpSurfaceError("max_file_bytes cannot exceed max_total_bytes")
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes

    def extract_source(
        self,
        source: str | bytes,
        *,
        provider: str,
        path: str,
        repository_tree_id: str = "",
    ) -> PythonMcpPackageSurface:
        return extract_python_mcp_source(
            source,
            provider=provider,
            path=path,
            repository_tree_id=repository_tree_id,
        )

    @staticmethod
    def bind_live_tools_list(
        payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        capability: LiveDiscoveryCapability,
        provider: str | None = None,
        transport: str | None = None,
        endpoint_identity: str | None = None,
        repository_tree_id: str | None = None,
    ) -> LiveToolsListEvidence:
        """Convenience facade for the separate live-evidence boundary."""

        return bind_live_tools_list(
            payload,
            capability=capability,
            provider=provider,
            transport=transport,
            endpoint_identity=endpoint_identity,
            repository_tree_id=repository_tree_id,
        )

    def extract_package(
        self,
        root: str | os.PathLike[str],
        *,
        provider: str,
        repository_tree_id: str = "",
        paths: Iterable[str | os.PathLike[str]] | None = None,
    ) -> PythonMcpPackageSurface:
        """Read Python files beneath *root* without importing the package."""

        provider = str(provider or "").strip()
        if not provider:
            raise PythonMcpSurfaceError("provider is required")
        root_path = Path(root).resolve()
        if not root_path.is_dir():
            raise PythonMcpSurfaceError("package root must be an existing directory")
        if paths is None:
            selected = sorted(
                path for path in root_path.rglob("*.py") if path.is_file()
            )
        else:
            selected = []
            for raw in paths:
                candidate = (root_path / _clean_path(raw)).resolve()
                try:
                    candidate.relative_to(root_path)
                except ValueError as exc:
                    raise PythonMcpSurfaceError(
                        "source path escapes package root"
                    ) from exc
                if candidate.suffix != ".py" or not candidate.is_file():
                    raise PythonMcpSurfaceError(
                        f"source path is not a Python file: {raw}"
                    )
                selected.append(candidate)
            selected = sorted(set(selected))
        if len(selected) > self.max_files:
            raise PythonMcpSurfaceError("package exceeds max_files")

        tools: list[PythonMcpToolSurface] = []
        unresolved: list[UnresolvedRegistration] = []
        source_files: list[dict[str, str]] = []
        total = 0
        for file_path in selected:
            relative = file_path.relative_to(root_path).as_posix()
            try:
                file_path.resolve().relative_to(root_path)
            except ValueError as exc:
                raise PythonMcpSurfaceError(
                    f"source path escapes package root: {relative}"
                ) from exc
            try:
                data = file_path.read_bytes()
            except OSError as exc:
                # A stable placeholder hash binds the path even when it races away.
                missing_hash = _source_sha256("")
                unresolved.append(
                    UnresolvedRegistration(
                        provider=provider,
                        reason=UnresolvedReason.READ_ERROR,
                        expression="",
                        registration_api="Path.read_bytes",
                        span=SourceSpan(relative, missing_hash, 1, 0, 1, 0),
                        detail=str(exc),
                    )
                )
                continue
            if len(data) > self.max_file_bytes:
                source_hash = "sha256:" + hashlib.sha256(data).hexdigest()
                source_files.append({"path": relative, "source_sha256": source_hash})
                unresolved.append(
                    UnresolvedRegistration(
                        provider=provider,
                        reason=UnresolvedReason.RESOURCE_LIMIT,
                        expression="",
                        registration_api="bounded_read",
                        span=SourceSpan(relative, source_hash, 1, 0, 1, 0),
                        detail="file exceeds max_file_bytes",
                    )
                )
                continue
            total += len(data)
            if total > self.max_total_bytes:
                raise PythonMcpSurfaceError("package exceeds max_total_bytes")
            try:
                source = data.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                source_hash = "sha256:" + hashlib.sha256(data).hexdigest()
                source_files.append({"path": relative, "source_sha256": source_hash})
                unresolved.append(
                    UnresolvedRegistration(
                        provider=provider,
                        reason=UnresolvedReason.READ_ERROR,
                        expression="",
                        registration_api="utf8_decode",
                        span=SourceSpan(relative, source_hash, 1, 0, 1, 0),
                        detail=str(exc),
                    )
                )
                continue
            source_hash = _source_sha256(source)
            source_files.append({"path": relative, "source_sha256": source_hash})
            file_tools, file_unresolved = _StaticFileExtractor(
                provider, relative, source
            ).extract()
            tools.extend(file_tools)
            unresolved.extend(file_unresolved)
        return _build_surface(
            provider=provider,
            repository_tree_id=repository_tree_id,
            tools=tools,
            unresolved=unresolved,
            source_files=source_files,
        )


__all__ = [
    "PYTHON_MCP_SURFACE_EXTRACTOR_INTERFACE",
    "PYTHON_MCP_SURFACE_SCHEMA",
    "PYTHON_MCP_TOOL_SCHEMA",
    "PYTHON_MCP_UNRESOLVED_SCHEMA",
    "PYTHON_MCP_LIVE_EVIDENCE_SCHEMA",
    "PythonMcpSurfaceError",
    "ToolSurfaceKind",
    "ResolutionState",
    "UnresolvedReason",
    "SourceSpan",
    "HandlerReachability",
    "PythonMcpToolSurface",
    "UnresolvedRegistration",
    "LiveDiscoveryCapability",
    "LiveToolsListEvidence",
    "PythonMcpPackageSurface",
    "PythonMcpSurfaceExtractor",
    "extract_python_mcp_source",
    "bind_live_tools_list",
]
