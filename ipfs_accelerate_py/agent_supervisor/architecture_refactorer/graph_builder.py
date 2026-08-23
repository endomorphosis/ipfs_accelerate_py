"""Bounded ArchitectureIR graph extraction for Python sources (PCAR-003).

Parses declared text without importing inspected modules. Every emitted fact
carries exact source spans, extractor identity, freshness, repository tree, and
closed confidence. Dynamic dispatch widens edges conservatively; heuristic and
opaque facts never promote to exact. Symlink, submodule, protected, and
repository-escape paths fail closed before I/O.
"""

from __future__ import annotations

import ast
import json
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .architecture_ir import (
    ArchitectureEdge,
    ArchitectureIR,
    ArchitectureIRError,
    ArchitectureNode,
)
from .contracts import (
    Confidence,
    EdgeKind,
    NodeKind,
    SourceFactIdentity,
    SourceSpan,
    _require_text,
)

ARCHITECTURE_GRAPH_EVIDENCE = "pcar/architecture-graph@1"
ARCHITECTURE_GRAPH_BUILDER_VERSION = "architecture-graph-builder@1"
EXTRACTOR_IDENTITY = "pcar-003-architecture-graph-builder"
GRAPH_EXTRACTOR_IDENTITY = EXTRACTOR_IDENTITY
TASK_ID = "PCAR-003"
DEFAULT_FRESHNESS = "pcar-003-architecture-graph"
DEFAULT_MAX_SOURCE_BYTES = 1_048_576
DEFAULT_MAX_FILES = 10_000
DEFAULT_MAX_FACTS = 250_000

DEFAULT_PROTECTED_PATHS: tuple[str, ...] = (
    "docs/architecture/AGENT_SUPERVISOR_PROOF_CARRYING_ARCHITECTURE_REFACTORER_PLAN.md",
    "docs/architecture/agent_supervisor_architecture_refactorer.objectives.md",
    "docs/architecture/agent_supervisor_architecture_refactorer.todo.md",
    "config/agent_supervisor_architecture_refactorer_scheduler.json",
    "scripts/validate_agent_supervisor_architecture_refactorer_board.py",
    "scripts/run_agent_supervisor_architecture_refactorer.py",
    "test/api/architecture_refactorer/test_board.py",
)
DEFAULT_GITLINK_PREFIXES: tuple[str, ...] = (
    "ipfs_datasets_py",
    "ipfs_kit_py",
    "ipfs_accelerate_py/mcplusplus",
)
DEFAULT_WALK_EXCLUSIONS: tuple[str, ...] = (
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
)

_PYTHON_SUFFIXES = frozenset({".py", ".pyi"})
_JSON_SUFFIXES = frozenset({".json", ".jsonschema"})
_SKIP_CALL_LEAVES = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "frozenset",
        "hash",
        "id",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "max",
        "min",
        "next",
        "object",
        "property",
        "range",
        "repr",
        "reversed",
        "set",
        "sorted",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "zip",
        "classmethod",
        "staticmethod",
        "hasattr",
    }
)
_DYNAMIC_CALLEES = frozenset(
    {
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "eval",
        "exec",
        "compile",
        "__import__",
        "importlib.import_module",
        "import_module",
        "builtins.getattr",
        "builtins.setattr",
        "builtins.eval",
        "builtins.exec",
        "builtins.__import__",
    }
)
_CONSTRUCT_SKIP = frozenset(
    {
        "dict",
        "list",
        "tuple",
        "set",
        "frozenset",
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "bytearray",
        "type",
        "object",
        "Exception",
        "ValueError",
        "TypeError",
        "RuntimeError",
        "KeyError",
        "AttributeError",
        "ArchitectureIRError",
        "ArchitectureContractError",
        "ArchitectureGraphBuilderError",
        "ArchitectureGraphEscapeError",
    }
)
_PROTOCOL_BASES = frozenset({"Protocol", "ABC", "abc.ABC", "typing.Protocol"})
_TEST_PATH_MARKERS = ("/test/", "/tests/", "/testing/")
_PROOF_PATH_MARKERS = ("/proof/", "/proofs/", ".proof.")
_SCHEMA_PATH_MARKERS = ("schema", "openapi", "jsonschema")
_OPERATION_DECORATOR_MARKERS = ("operation", "endpoint", "command", "handler")
_PURE_BUILTINS_OPEN_WRITE_MODES = frozenset({"w", "wb", "a", "ab", "x", "xb", "wt", "at", "xt"})


class ArchitectureGraphBuilderError(ArchitectureIRError):
    """Fail-closed ArchitectureIR graph extraction error."""


class ArchitectureGraphEscapeError(ArchitectureGraphBuilderError):
    """Raised before I/O when a path escapes the declared extraction bound."""


def _error(
    message: str,
    *,
    error_type: type[ArchitectureGraphBuilderError] = ArchitectureGraphBuilderError,
) -> ArchitectureGraphBuilderError:
    return error_type(message)


def normalize_relative_path(relative: str, *, name: str = "path") -> str:
    """Return a repository-relative POSIX path with no escape components."""

    text = _require_text(relative, name, error_type=ArchitectureGraphEscapeError).replace(
        "\\", "/"
    )
    parts = tuple(part for part in text.split("/") if part not in ("", "."))
    if not parts or text.startswith("/") or any(part == ".." for part in parts):
        raise _error(
            f"{name} must be a repository-relative path",
            error_type=ArchitectureGraphEscapeError,
        )
    return "/".join(parts)


def logical_path_under(relative: str, prefix: str) -> bool:
    """Return whether ``relative`` is ``prefix`` or a descendant (no I/O)."""

    path = normalize_relative_path(relative)
    root = normalize_relative_path(prefix, name="prefix")
    return path == root or path.startswith(root + "/")


def module_name_from_path(relative: str) -> str:
    """Derive a dotted module name from a repository-relative Python path."""

    path = normalize_relative_path(relative)
    value = path
    for suffix in (".py", ".pyi"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    parts = list(PurePosixPath(value).parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    if not parts:
        return PurePosixPath(path).stem
    return ".".join(parts)


def package_name_from_module(module: str) -> str:
    if "." not in module:
        return module
    return module.rsplit(".", 1)[0]


def _posix_parent(relative: str) -> str:
    parent = str(PurePosixPath(relative).parent)
    return "" if parent == "." else parent


def _is_test_path(relative: str) -> bool:
    path = f"/{relative}/"
    name = PurePosixPath(relative).name
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or any(marker in path for marker in _TEST_PATH_MARKERS)
    )


def _is_proof_path(relative: str) -> bool:
    lowered = relative.replace("\\", "/").lower()
    name = PurePosixPath(relative).name.lower()
    return any(marker in f"/{lowered}/" or marker in name for marker in _PROOF_PATH_MARKERS) or (
        name.startswith("proof_") or name.endswith(".proof.json")
    )


def _is_schema_path(relative: str) -> bool:
    lowered = relative.replace("\\", "/").lower()
    return any(marker in lowered for marker in _SCHEMA_PATH_MARKERS)


def _file_span_lines(source: str) -> tuple[int, int]:
    if not source:
        return 1, 1
    if source.endswith("\n"):
        lines = source.count("\n")
        return 1, max(1, lines)
    return 1, source.count("\n") + 1


def _line_span(source: str, node: ast.AST) -> tuple[int, int]:
    start = int(getattr(node, "lineno", 1) or 1)
    end = int(getattr(node, "end_lineno", start) or start)
    if start < 1:
        start = 1
    if end < start:
        end = start
    _, last = _file_span_lines(source)
    return start, min(end, max(start, last))


def _expression_name(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expression_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Subscript):
        parent = _expression_name(node.value)
        return parent
    if isinstance(node, ast.Call):
        return _expression_name(node.func)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _constant_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return _expression_name(node.func)
    return _expression_name(node)


def _leaf(name: str) -> str:
    return name.rsplit(".", 1)[-1] if name else ""


def _node_id(kind: NodeKind, identity: str) -> str:
    return f"n:{kind.value}:{identity}"


def _edge_id(
    kind: EdgeKind,
    source: str,
    target: str,
    span: SourceSpan,
    disambiguator: int = 0,
) -> str:
    return (
        f"e:{kind.value}:{source}:{target}:{span.path}:"
        f"{span.start_line}:{span.end_line}:{disambiguator}"
    )


def nodes_of(graph: ArchitectureIR, *kinds: NodeKind) -> tuple[ArchitectureNode, ...]:
    wanted = frozenset(kinds)
    return tuple(node for node in graph.nodes if node.kind in wanted)


def edges_of(graph: ArchitectureIR, *kinds: EdgeKind) -> tuple[ArchitectureEdge, ...]:
    wanted = frozenset(kinds)
    return tuple(edge for edge in graph.edges if edge.kind in wanted)


def call_targets(
    graph: ArchitectureIR,
    source: str,
    *,
    kinds: Iterable[EdgeKind] = (EdgeKind.CALLS, EdgeKind.CONSTRUCTS, EdgeKind.EXECUTES),
) -> tuple[str, ...]:
    admitted = frozenset(kinds)
    return tuple(
        edge.target
        for edge in graph.edges
        if edge.source == source and edge.kind in admitted
    )


@dataclass(frozen=True)
class _ClassInfo:
    name: str
    qualified: str
    methods: tuple[str, ...]
    bases: tuple[str, ...]
    decorators: tuple[str, ...]
    span: tuple[int, int]
    has_dynamic_attr: bool


@dataclass
class _OpenHandle:
    mode: str
    span: tuple[int, int]


@dataclass
class _DynamicBinding:
    receiver: str
    attribute: str | None
    owner_qualified: str
    span: tuple[int, int]


class _Accumulator:
    """Deterministic node/edge accumulator for one extraction."""

    def __init__(
        self,
        *,
        repository_tree: str,
        freshness: str,
        extractor_identity: str,
        max_facts: int,
    ) -> None:
        self.repository_tree = repository_tree
        self.freshness = freshness
        self.extractor_identity = extractor_identity
        self.max_facts = max_facts
        self.nodes: dict[str, ArchitectureNode] = {}
        self.edges: dict[str, ArchitectureEdge] = {}
        self._edge_keys: dict[tuple[str, str, str, str, int, int], int] = {}

    def span(self, path: str, start: int, end: int) -> SourceSpan:
        return SourceSpan(path=path, start_line=start, end_line=end)

    def fact(
        self,
        path: str,
        start: int,
        end: int,
        confidence: Confidence,
    ) -> SourceFactIdentity:
        return SourceFactIdentity(
            extractor_identity=self.extractor_identity,
            span=self.span(path, start, end),
            confidence=confidence,
            freshness=self.freshness,
            repository_tree=self.repository_tree,
        )

    def add_node(
        self,
        kind: NodeKind,
        identity: str,
        path: str,
        start: int,
        end: int,
        *,
        confidence: Confidence = Confidence.EXACT,
    ) -> str:
        node_id = _node_id(kind, identity)
        if node_id in self.nodes:
            return node_id
        if len(self.nodes) + len(self.edges) >= self.max_facts:
            raise _error("graph fact bound exceeded")
        self.nodes[node_id] = ArchitectureNode(
            node_id=node_id,
            kind=kind,
            provenance=self.fact(path, start, end, confidence),
        )
        return node_id

    def add_edge(
        self,
        kind: EdgeKind,
        source: str,
        target: str,
        path: str,
        start: int,
        end: int,
        *,
        confidence: Confidence = Confidence.EXACT,
    ) -> str | None:
        if source not in self.nodes or target not in self.nodes:
            return None
        if source == target and kind is not EdgeKind.CALLS:
            return None
        key = (kind.value, source, target, path, start, end)
        disambiguator = self._edge_keys.get(key, 0)
        self._edge_keys[key] = disambiguator + 1
        edge_id = _edge_id(kind, source, target, self.span(path, start, end), disambiguator)
        if edge_id in self.edges:
            return edge_id
        if len(self.nodes) + len(self.edges) >= self.max_facts:
            raise _error("graph fact bound exceeded")
        self.edges[edge_id] = ArchitectureEdge(
            edge_id=edge_id,
            kind=kind,
            source=source,
            target=target,
            provenance=self.fact(path, start, end, confidence),
        )
        return edge_id

    def graph(self) -> ArchitectureIR:
        return ArchitectureIR.from_parts(
            repository_tree=self.repository_tree,
            freshness=self.freshness,
            nodes=tuple(self.nodes.values()),
            edges=tuple(self.edges.values()),
        )


def _open_mode(call: ast.Call) -> str:
    if len(call.args) >= 2:
        literal = _constant_str(call.args[1])
        if literal is not None:
            return literal
    for keyword in call.keywords:
        if keyword.arg == "mode":
            literal = _constant_str(keyword.value)
            if literal is not None:
                return literal
    return "r"


def _effect_rule(callee: str) -> tuple[EdgeKind, str, Confidence] | None:
    leaf = _leaf(callee)
    full = callee
    distinctive: dict[str, tuple[EdgeKind, str, Confidence]] = {
        "read_text": (EdgeKind.READS, "filesystem", Confidence.EXACT),
        "read_bytes": (EdgeKind.READS, "filesystem", Confidence.EXACT),
        "write_text": (EdgeKind.WRITES, "filesystem", Confidence.EXACT),
        "write_bytes": (EdgeKind.WRITES, "filesystem", Confidence.EXACT),
        "read_json": (EdgeKind.DESERIALIZES, "serialization", Confidence.CONSERVATIVE),
        "json.loads": (EdgeKind.DESERIALIZES, "serialization", Confidence.EXACT),
        "json.load": (EdgeKind.DESERIALIZES, "serialization", Confidence.EXACT),
        "json.dumps": (EdgeKind.SERIALIZES, "serialization", Confidence.EXACT),
        "json.dump": (EdgeKind.SERIALIZES, "serialization", Confidence.EXACT),
        "yaml.safe_load": (EdgeKind.DESERIALIZES, "serialization", Confidence.EXACT),
        "yaml.dump": (EdgeKind.SERIALIZES, "serialization", Confidence.EXACT),
        "subprocess.run": (EdgeKind.EXECUTES, "process", Confidence.EXACT),
        "subprocess.Popen": (EdgeKind.EXECUTES, "process", Confidence.EXACT),
        "subprocess.call": (EdgeKind.EXECUTES, "process", Confidence.EXACT),
        "subprocess.check_call": (EdgeKind.EXECUTES, "process", Confidence.EXACT),
        "subprocess.check_output": (EdgeKind.EXECUTES, "process", Confidence.EXACT),
        "os.system": (EdgeKind.EXECUTES, "process", Confidence.EXACT),
        "os.popen": (EdgeKind.EXECUTES, "process", Confidence.EXACT),
        "os.remove": (EdgeKind.WRITES, "filesystem", Confidence.EXACT),
        "os.unlink": (EdgeKind.WRITES, "filesystem", Confidence.EXACT),
        "pathlib.Path.read_text": (EdgeKind.READS, "filesystem", Confidence.EXACT),
        "pathlib.Path.write_text": (EdgeKind.WRITES, "filesystem", Confidence.EXACT),
        "logging.debug": (EdgeKind.OBSERVES, "logging", Confidence.EXACT),
        "logging.info": (EdgeKind.OBSERVES, "logging", Confidence.EXACT),
        "logging.warning": (EdgeKind.OBSERVES, "logging", Confidence.EXACT),
        "logging.error": (EdgeKind.OBSERVES, "logging", Confidence.EXACT),
        "print": (EdgeKind.OBSERVES, "logging", Confidence.EXACT),
        "requests.get": (EdgeKind.READS, "network", Confidence.EXACT),
        "requests.post": (EdgeKind.WRITES, "network", Confidence.EXACT),
        "requests.put": (EdgeKind.WRITES, "network", Confidence.EXACT),
        "requests.delete": (EdgeKind.WRITES, "network", Confidence.EXACT),
        "httpx.get": (EdgeKind.READS, "network", Confidence.EXACT),
        "httpx.post": (EdgeKind.WRITES, "network", Confidence.EXACT),
        "urllib.request.urlopen": (EdgeKind.READS, "network", Confidence.EXACT),
        "duckdb.connect": (EdgeKind.PERSISTS, "state", Confidence.EXACT),
        "sqlite3.connect": (EdgeKind.PERSISTS, "state", Confidence.EXACT),
        "connect": (EdgeKind.PERSISTS, "state", Confidence.CONSERVATIVE),
        "commit": (EdgeKind.PERSISTS, "state", Confidence.CONSERVATIVE),
        "execute": (EdgeKind.MUTATES, "state", Confidence.CONSERVATIVE),
        "executemany": (EdgeKind.MUTATES, "state", Confidence.CONSERVATIVE),
        "dumps": (EdgeKind.SERIALIZES, "serialization", Confidence.CONSERVATIVE),
        "loads": (EdgeKind.DESERIALIZES, "serialization", Confidence.CONSERVATIVE),
        "dump": (EdgeKind.SERIALIZES, "serialization", Confidence.CONSERVATIVE),
        "load": (EdgeKind.DESERIALIZES, "serialization", Confidence.CONSERVATIVE),
    }
    if full in distinctive:
        return distinctive[full]
    if leaf in distinctive:
        return distinctive[leaf]
    return None


class _PythonExtractor(ast.NodeVisitor):
    """Collect ArchitectureIR facts from one Python module."""

    def __init__(
        self,
        *,
        acc: _Accumulator,
        path: str,
        source: str,
        module_id: str,
        file_id: str,
        module_name: str,
        in_repo_modules: Mapping[str, str],
        in_repo_symbols: Mapping[str, str],
    ) -> None:
        self.acc = acc
        self.path = path
        self.source = source
        self.module_id = module_id
        self.file_id = file_id
        self.module_name = module_name
        self.in_repo_modules = in_repo_modules
        self.in_repo_symbols = dict(in_repo_symbols)
        self.scope: list[str] = []
        self.aliases: dict[str, str] = {}
        self.imported_modules: dict[str, str] = {}
        self.classes: dict[str, _ClassInfo] = {}
        self.local_symbols: dict[str, str] = {}
        self.dynamic_names: dict[str, _DynamicBinding] = {}
        self.file_handles: dict[str, _OpenHandle] = {}
        self.schema_names: set[str] = set()
        self.reexport_names: set[str] = set()

    def _owner_qualified(self) -> str:
        if not self.scope:
            return self.module_name
        return f"{self.module_name}.{'.'.join(self.scope)}"

    def _owner_symbol_id(self) -> str:
        qualified = self._owner_qualified()
        if qualified in self.local_symbols:
            return self.local_symbols[qualified]
        if self.scope:
            return self.acc.add_node(
                NodeKind.SYMBOL,
                qualified,
                self.path,
                1,
                1,
                confidence=Confidence.CONSERVATIVE,
            )
        return self.module_id

    def _span_of(self, node: ast.AST) -> tuple[int, int]:
        return _line_span(self.source, node)

    def _resolve_alias(self, name: str) -> str:
        if not name:
            return ""
        root, _, rest = name.partition(".")
        if root in self.aliases:
            target = self.aliases[root]
            return f"{target}.{rest}" if rest else target
        if root in self.imported_modules:
            module = self.imported_modules[root]
            return f"{module}.{rest}" if rest else module
        return name

    def _qualified_child(self, name: str) -> str:
        if self.scope:
            return f"{self.module_name}.{'.'.join((*self.scope, name))}"
        return f"{self.module_name}.{name}"

    def collect_definitions(self, tree: ast.AST) -> None:
        for node in tree.body if isinstance(tree, ast.Module) else ():
            if isinstance(node, ast.Assign):
                self._module_assign(node)
            elif isinstance(node, ast.AnnAssign):
                self._module_annassign(node)
        self._collect_scope(tree, ())

    def _collect_scope(self, node: ast.AST, prefix: tuple[str, ...]) -> None:
        body: Sequence[ast.AST]
        if isinstance(node, ast.Module):
            body = node.body
        elif isinstance(node, ast.ClassDef):
            body = node.body
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
        else:
            return
        for child in body:
            if isinstance(child, ast.ClassDef):
                qualified = ".".join((self.module_name, *prefix, child.name))
                methods: list[str] = []
                has_dynamic = False
                for item in child.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(item.name)
                        if item.name in {"__getattr__", "__getattribute__"}:
                            has_dynamic = True
                info = _ClassInfo(
                    name=child.name,
                    qualified=qualified,
                    methods=tuple(methods),
                    bases=tuple(_expression_name(base) for base in child.bases if _expression_name(base)),
                    decorators=tuple(_decorator_name(item) for item in child.decorator_list),
                    span=self._span_of(child),
                    has_dynamic_attr=has_dynamic,
                )
                self.classes[qualified] = info
                self.classes[child.name] = info
                self._collect_scope(child, (*prefix, child.name))
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._collect_scope(child, (*prefix, child.name))

    def _module_assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                names: list[str] = []
                if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                    for elt in node.value.elts:
                        literal = _constant_str(elt)
                        if literal:
                            names.append(literal)
                self.reexport_names.update(names)
            if isinstance(target, ast.Name) and self._looks_like_schema(target.id, node.value):
                self.schema_names.add(target.id)

    def _module_annassign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.value is not None:
            if self._looks_like_schema(node.target.id, node.value):
                self.schema_names.add(node.target.id)

    def _looks_like_schema(self, name: str, value: ast.AST) -> bool:
        if "schema" in name.lower() or name.isupper() and "SCHEMA" in name:
            return True
        if isinstance(value, ast.Dict):
            keys = {
                elt.value
                for elt in value.keys
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            }
            return bool({"type", "properties", "$schema", "required"} & keys)
        return False

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        start, end = self._span_of(node)
        qualified = self._qualified_child(node.name)
        symbol_id = self.acc.add_node(NodeKind.SYMBOL, qualified, self.path, start, end)
        self.local_symbols[qualified] = symbol_id
        self.local_symbols[node.name] = symbol_id
        self.in_repo_symbols.setdefault(qualified, symbol_id)
        self.acc.add_edge(EdgeKind.CONTAINS, self.module_id, symbol_id, self.path, start, end)
        self.acc.add_edge(EdgeKind.CONTAINS, self.file_id, symbol_id, self.path, start, end)
        parent = self._owner_symbol_id()
        if parent != symbol_id:
            self.acc.add_edge(EdgeKind.CONTAINS, parent, symbol_id, self.path, start, end)
        info = self.classes.get(qualified)
        decorator_names = [ _decorator_name(item) for item in node.decorator_list ]
        if info is not None:
            for base in info.bases:
                resolved = self._resolve_alias(base)
                confidence = Confidence.EXACT if resolved in self.classes else Confidence.CONSERVATIVE
                target_id = self._symbol_or_interface(resolved, start, end, confidence)
                self.acc.add_edge(
                    EdgeKind.IMPLEMENTS,
                    symbol_id,
                    target_id,
                    self.path,
                    start,
                    end,
                    confidence=confidence,
                )
            if any("dataclass" in name.lower() for name in info.decorators) or any(
                "dataclass" in name.lower() for name in decorator_names
            ):
                schema_id = self.acc.add_node(NodeKind.SCHEMA, qualified, self.path, start, end)
                self.acc.add_edge(EdgeKind.CONTAINS, self.module_id, schema_id, self.path, start, end)
                self.acc.add_edge(
                    EdgeKind.SERIALIZES, symbol_id, schema_id, self.path, start, end
                )
                self.acc.add_edge(
                    EdgeKind.DESERIALIZES, symbol_id, schema_id, self.path, start, end
                )
        if node.name.endswith("Protocol") or any(
            _leaf(base) in {"Protocol", "ABC"} for base in (_expression_name(b) for b in node.bases)
        ):
            iface = self.acc.add_node(NodeKind.INTERFACE, qualified, self.path, start, end)
            self.acc.add_edge(EdgeKind.IMPLEMENTS, symbol_id, iface, self.path, start, end)
            self.acc.add_edge(EdgeKind.CONTAINS, self.module_id, iface, self.path, start, end)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        start, end = self._span_of(node)
        qualified = self._qualified_child(node.name)
        symbol_id = self.acc.add_node(NodeKind.SYMBOL, qualified, self.path, start, end)
        self.local_symbols[qualified] = symbol_id
        self.local_symbols[node.name] = symbol_id
        self.in_repo_symbols.setdefault(qualified, symbol_id)
        parent = self._owner_symbol_id()
        self.acc.add_edge(EdgeKind.CONTAINS, parent, symbol_id, self.path, start, end)
        self.acc.add_edge(EdgeKind.CONTAINS, self.file_id, symbol_id, self.path, start, end)
        if parent != self.module_id:
            self.acc.add_edge(EdgeKind.CONTAINS, self.module_id, symbol_id, self.path, start, start)
        decorators = [_decorator_name(item) for item in node.decorator_list]
        is_operation = any(
            any(marker in name.lower() for marker in _OPERATION_DECORATOR_MARKERS)
            for name in decorators
        ) or node.name.startswith(("run_", "handle_", "execute_"))
        if is_operation:
            op_id = self.acc.add_node(NodeKind.OPERATION, qualified, self.path, start, end)
            self.acc.add_edge(EdgeKind.CONTAINS, self.module_id, op_id, self.path, start, end)
            self.acc.add_edge(EdgeKind.IMPLEMENTS, symbol_id, op_id, self.path, start, end)
            self.acc.add_edge(EdgeKind.EXECUTES, symbol_id, op_id, self.path, start, end)
        if node.name.startswith("test_") or _is_test_path(self.path):
            if node.name.startswith("test_"):
                test_id = self.acc.add_node(NodeKind.TEST, qualified, self.path, start, end)
                self.acc.add_edge(EdgeKind.CONTAINS, self.module_id, test_id, self.path, start, end)
                self.acc.add_edge(EdgeKind.TESTS, test_id, symbol_id, self.path, start, end)
        if node.name.startswith(("prove_", "proof_")):
            proof_id = self.acc.add_node(NodeKind.PROOF, qualified, self.path, start, end)
            self.acc.add_edge(EdgeKind.CONTAINS, self.module_id, proof_id, self.path, start, end)
            self.acc.add_edge(EdgeKind.PROVES, proof_id, symbol_id, self.path, start, end)
        if node.name == "main" and not self.scope:
            entry = self.acc.add_node(NodeKind.ENTRYPOINT, qualified, self.path, start, end)
            self.acc.add_edge(EdgeKind.CONTAINS, self.module_id, entry, self.path, start, end)
            self.acc.add_edge(EdgeKind.EXECUTES, entry, symbol_id, self.path, start, end)
        self.scope.append(node.name)
        prior_dynamic = dict(self.dynamic_names)
        prior_handles = dict(self.file_handles)
        self.generic_visit(node)
        self.dynamic_names = prior_dynamic
        self.file_handles = prior_handles
        self.scope.pop()

    def visit_Import(self, node: ast.Import) -> None:
        start, end = self._span_of(node)
        for alias in node.names:
            local = alias.asname or alias.name.split(".", 1)[0]
            self.aliases[local] = alias.name
            self.imported_modules[local] = alias.name.split(".", 1)[0] if alias.asname else alias.name
            target_id = self._imported_module_node(alias.name, start, end)
            self.acc.add_edge(
                EdgeKind.IMPORTS,
                self.module_id,
                target_id,
                self.path,
                start,
                end,
                confidence=Confidence.EXACT if alias.name in self.in_repo_modules else Confidence.CONSERVATIVE,
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        start, end = self._span_of(node)
        module = self._resolve_relative(node.module or "", int(node.level or 0))
        module_id = (
            self._imported_module_node(module, start, end)
            if module
            else None
        )
        if module_id is not None:
            confidence = (
                Confidence.EXACT if module in self.in_repo_modules else Confidence.CONSERVATIVE
            )
            self.acc.add_edge(
                EdgeKind.IMPORTS,
                self.module_id,
                module_id,
                self.path,
                start,
                end,
                confidence=confidence,
            )
        star = any(alias.name == "*" for alias in node.names)
        if star:
            opaque = self.acc.add_node(
                NodeKind.ARTIFACT,
                f"star-import:{self.module_name}:{module or '*'}",
                self.path,
                start,
                end,
                confidence=Confidence.OPAQUE,
            )
            self.acc.add_edge(
                EdgeKind.IMPORTS,
                self.module_id,
                opaque,
                self.path,
                start,
                end,
                confidence=Confidence.OPAQUE,
            )
            return
        init_module = PurePosixPath(self.path).name == "__init__.py"
        for alias in node.names:
            local = alias.asname or alias.name
            target_name = f"{module}.{alias.name}" if module else alias.name
            self.aliases[local] = target_name
            symbol_id = self._imported_symbol_node(target_name, start, end)
            confidence = (
                Confidence.EXACT if target_name in self.in_repo_symbols else Confidence.CONSERVATIVE
            )
            self.acc.add_edge(
                EdgeKind.IMPORTS,
                self.module_id,
                symbol_id,
                self.path,
                start,
                end,
                confidence=confidence,
            )
            if init_module or local in self.reexport_names or alias.name in self.reexport_names:
                self.acc.add_edge(
                    EdgeKind.REEXPORTS,
                    self.module_id,
                    symbol_id,
                    self.path,
                    start,
                    end,
                    confidence=confidence,
                )

    def visit_Assign(self, node: ast.Assign) -> None:
        if not self.scope and any(
            isinstance(target, ast.Name) and target.id in self.schema_names for target in node.targets
        ):
            start, end = self._span_of(node)
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in self.schema_names:
                    identity = f"{self.module_name}.{target.id}"
                    schema_id = self.acc.add_node(NodeKind.SCHEMA, identity, self.path, start, end)
                    self.acc.add_edge(EdgeKind.CONTAINS, self.module_id, schema_id, self.path, start, end)
                    self.acc.add_edge(EdgeKind.CONTAINS, self.file_id, schema_id, self.path, start, end)
        for target in node.targets:
            self._bind_value(target, node.value, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._bind_value(node.target, node.value, node)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self._with(node)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._with(node)
        self.generic_visit(node)

    def _with(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            expr = item.context_expr
            if isinstance(expr, ast.Call) and _leaf(_expression_name(expr.func)) == "open":
                mode = _open_mode(expr)
                if isinstance(item.optional_vars, ast.Name):
                    start, end = self._span_of(expr)
                    self.file_handles[item.optional_vars.id] = _OpenHandle(mode, (start, end))
                    self._emit_open_effect(expr, start, end)

    def visit_Call(self, node: ast.Call) -> None:
        start, end = self._span_of(node)
        owner_id = self._owner_symbol_id()
        raw = _expression_name(node.func)
        callee = self._resolve_alias(raw) if raw else ""
        leaf = _leaf(callee or raw)
        if isinstance(node.func, ast.Name) and node.func.id in self.dynamic_names:
            self._emit_dynamic_call(owner_id, self.dynamic_names[node.func.id], start, end)
            self.generic_visit(node)
            return
        if leaf == "open" or callee.endswith(".open"):
            self._emit_open_effect(node, start, end)
            if self.scope:
                # Assignment visitor handles named results; still record the call site.
                pass
        if leaf in {"getattr", "__import__", "import_module", "eval", "exec", "setattr", "delattr", "compile"} or (
            callee in _DYNAMIC_CALLEES
        ):
            self._emit_dynamic_callee(owner_id, node, callee or leaf, start, end)
            self.generic_visit(node)
            return
        if isinstance(node.func, ast.Name) and node.func.id in self.file_handles:
            handle = self.file_handles[node.func.id]
            self._emit_handle_effect(owner_id, handle, leaf, start, end)
            self.generic_visit(node)
            return
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            receiver = node.func.value.id
            if receiver in self.file_handles:
                self._emit_handle_effect(
                    owner_id, self.file_handles[receiver], node.func.attr, start, end
                )
                self.generic_visit(node)
                return
        effect = _effect_rule(callee or raw)
        if effect is not None:
            edge_kind, effect_class, confidence = effect
            effect_id = self.acc.add_node(
                NodeKind.EFFECT,
                f"{self.module_name}:{effect_class}",
                self.path,
                start,
                end,
                confidence=confidence,
            )
            self.acc.add_edge(EdgeKind.CONTAINS, self.module_id, effect_id, self.path, start, start)
            self.acc.add_edge(
                edge_kind, owner_id, effect_id, self.path, start, end, confidence=confidence
            )
            if effect_class == "state":
                state_id = self.acc.add_node(
                    NodeKind.STATE,
                    f"{self.module_name}:{leaf}",
                    self.path,
                    start,
                    end,
                    confidence=confidence,
                )
                self.acc.add_edge(EdgeKind.CONTAINS, self.module_id, state_id, self.path, start, start)
                persist_kind = EdgeKind.PERSISTS if edge_kind is EdgeKind.PERSISTS else EdgeKind.MUTATES
                self.acc.add_edge(
                    persist_kind, owner_id, state_id, self.path, start, end, confidence=confidence
                )
            if effect_class == "serialization":
                schema_id = self.acc.add_node(
                    NodeKind.SCHEMA,
                    f"{self.module_name}:serialized",
                    self.path,
                    start,
                    end,
                    confidence=Confidence.CONSERVATIVE,
                )
                self.acc.add_edge(EdgeKind.CONTAINS, self.module_id, schema_id, self.path, start, start)
                serde = (
                    EdgeKind.SERIALIZES
                    if edge_kind in {EdgeKind.SERIALIZES, EdgeKind.WRITES}
                    else EdgeKind.DESERIALIZES
                )
                self.acc.add_edge(serde, owner_id, schema_id, self.path, start, end, confidence=confidence)
        self._emit_ordinary_call(owner_id, node, callee or raw, start, end)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if self._is_main_guard(node.test):
            start, end = self._span_of(node)
            entry = self.acc.add_node(
                NodeKind.ENTRYPOINT, f"{self.module_name}:__main__", self.path, start, end
            )
            self.acc.add_edge(EdgeKind.CONTAINS, self.module_id, entry, self.path, start, end)
            self.acc.add_edge(EdgeKind.EXECUTES, entry, self.module_id, self.path, start, end)
        self.generic_visit(node)

    def _is_main_guard(self, test: ast.AST) -> bool:
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            return False
        left = test.left
        comparator = test.comparators[0]
        names = {_expression_name(left), _expression_name(comparator)}
        constants = {_constant_str(left), _constant_str(comparator)}
        return "__name__" in names and "__main__" in constants

    def _bind_value(self, target: ast.AST, value: ast.AST, node: ast.AST) -> None:
        if not isinstance(target, ast.Name):
            return
        start, end = self._span_of(node)
        if isinstance(value, ast.Call):
            leaf = _leaf(_expression_name(value.func))
            callee = self._resolve_alias(_expression_name(value.func))
            if leaf == "open" or callee.endswith(".open"):
                self.file_handles[target.id] = _OpenHandle(_open_mode(value), (start, end))
            if leaf == "getattr" or callee in {"getattr", "builtins.getattr"}:
                receiver = _expression_name(value.args[0]) if value.args else ""
                attribute = _constant_str(value.args[1]) if len(value.args) > 1 else None
                self.dynamic_names[target.id] = _DynamicBinding(
                    receiver=receiver,
                    attribute=attribute,
                    owner_qualified=self._owner_qualified(),
                    span=(start, end),
                )
            if leaf in {"__import__", "import_module"} or callee in {
                "importlib.import_module",
                "__import__",
            }:
                self.dynamic_names[target.id] = _DynamicBinding(
                    receiver=callee or leaf,
                    attribute=_constant_str(value.args[0]) if value.args else None,
                    owner_qualified=self._owner_qualified(),
                    span=(start, end),
                )

    def _emit_open_effect(self, call: ast.Call, start: int, end: int) -> None:
        owner_id = self._owner_symbol_id()
        mode = _open_mode(call)
        write = any(token in mode for token in _PURE_BUILTINS_OPEN_WRITE_MODES)
        edge_kind = EdgeKind.WRITES if write else EdgeKind.READS
        if "+" in mode:
            edge_kind = EdgeKind.MUTATES
        confidence = Confidence.EXACT if _constant_str(call.args[1] if len(call.args) > 1 else None) or any(
            keyword.arg == "mode" and _constant_str(keyword.value) for keyword in call.keywords
        ) or len(call.args) < 2 else Confidence.CONSERVATIVE
        if len(call.args) < 2 and not any(keyword.arg == "mode" for keyword in call.keywords):
            confidence = Confidence.EXACT
        effect_id = self.acc.add_node(
            NodeKind.EFFECT,
            f"{self.module_name}:filesystem",
            self.path,
            start,
            end,
            confidence=confidence,
        )
        self.acc.add_edge(EdgeKind.CONTAINS, self.module_id, effect_id, self.path, start, start)
        self.acc.add_edge(edge_kind, owner_id, effect_id, self.path, start, end, confidence=confidence)

    def _emit_handle_effect(
        self,
        owner_id: str,
        handle: _OpenHandle,
        attr: str,
        start: int,
        end: int,
    ) -> None:
        if attr in {"read", "readline", "readlines", "read1"}:
            kind = EdgeKind.READS
        elif attr in {"write", "writelines", "truncate", "flush"}:
            kind = EdgeKind.WRITES
        else:
            return
        effect_id = self.acc.add_node(
            NodeKind.EFFECT,
            f"{self.module_name}:filesystem",
            self.path,
            start,
            end,
        )
        self.acc.add_edge(EdgeKind.CONTAINS, self.module_id, effect_id, self.path, start, start)
        self.acc.add_edge(kind, owner_id, effect_id, self.path, start, end)

    def _emit_ordinary_call(
        self,
        owner_id: str,
        node: ast.Call,
        callee: str,
        start: int,
        end: int,
    ) -> None:
        if not callee:
            dynamic_id = self.acc.add_node(
                NodeKind.EFFECT,
                f"{self.module_name}:dynamic-dispatch",
                self.path,
                start,
                end,
                confidence=Confidence.OPAQUE,
            )
            self.acc.add_edge(EdgeKind.CONTAINS, self.module_id, dynamic_id, self.path, start, start)
            self.acc.add_edge(
                EdgeKind.CALLS, owner_id, dynamic_id, self.path, start, end, confidence=Confidence.OPAQUE
            )
            return
        leaf = _leaf(callee)
        if leaf in _SKIP_CALL_LEAVES:
            return
        if self._maybe_construct(owner_id, node, callee, start, end):
            return
        target_id, confidence = self._resolve_call_target(callee, node, start, end)
        if target_id is None:
            return
        self.acc.add_edge(
            EdgeKind.CALLS, owner_id, target_id, self.path, start, end, confidence=confidence
        )
        if _is_test_path(self.path) or (self.scope and self.scope[-1].startswith("test_")):
            test_qualified = self._owner_qualified()
            test_id = _node_id(NodeKind.TEST, test_qualified)
            if test_id not in self.acc.nodes and self.scope and self.scope[-1].startswith("test_"):
                test_id = self.acc.add_node(NodeKind.TEST, test_qualified, self.path, start, end)
                self.acc.add_edge(EdgeKind.CONTAINS, self.module_id, test_id, self.path, start, start)
            if test_id in self.acc.nodes:
                self.acc.add_edge(
                    EdgeKind.TESTS, test_id, target_id, self.path, start, end, confidence=confidence
                )
        if self.scope and self.scope[-1].startswith(("prove_", "proof_")):
            proof_id = _node_id(NodeKind.PROOF, self._owner_qualified())
            if proof_id in self.acc.nodes:
                self.acc.add_edge(
                    EdgeKind.PROVES, proof_id, target_id, self.path, start, end, confidence=confidence
                )

    def _maybe_construct(
        self,
        owner_id: str,
        node: ast.Call,
        callee: str,
        start: int,
        end: int,
    ) -> bool:
        leaf = _leaf(callee)
        if leaf in _CONSTRUCT_SKIP:
            return False
        class_info = self.classes.get(callee) or self.classes.get(leaf)
        imported_class = False
        if class_info is None and leaf[:1].isupper() and leaf[1:].replace("_", "").isalnum():
            imported_class = True
        if class_info is None and not imported_class:
            return False
        if class_info is not None:
            target_id = self.local_symbols.get(class_info.qualified) or self.acc.add_node(
                NodeKind.SYMBOL, class_info.qualified, self.path, start, end
            )
            confidence = Confidence.EXACT
        else:
            resolved = self._resolve_alias(callee)
            target_id, confidence = self._resolve_call_target(resolved, node, start, end)
            if target_id is None:
                target_id = self.acc.add_node(
                    NodeKind.SYMBOL,
                    f"external:{resolved}",
                    self.path,
                    start,
                    end,
                    confidence=Confidence.CONSERVATIVE,
                )
                confidence = Confidence.CONSERVATIVE
            else:
                confidence = Confidence.CONSERVATIVE
        self.acc.add_edge(
            EdgeKind.CONSTRUCTS, owner_id, target_id, self.path, start, end, confidence=confidence
        )
        return True

    def _resolve_call_target(
        self,
        callee: str,
        node: ast.Call,
        start: int,
        end: int,
    ) -> tuple[str | None, Confidence]:
        if not callee:
            return None, Confidence.OPAQUE
        if callee in self.local_symbols:
            return self.local_symbols[callee], Confidence.EXACT
        leaf = _leaf(callee)
        if self.scope:
            nested = f"{self.module_name}.{'.'.join(self.scope)}.{leaf}"
            if nested in self.local_symbols:
                return self.local_symbols[nested], Confidence.EXACT
            class_qualified = f"{self.module_name}.{self.scope[0]}.{leaf}" if self.scope else ""
            if class_qualified in self.local_symbols and (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"self", "cls"}
            ):
                return self.local_symbols[class_qualified], Confidence.EXACT
        if leaf in self.local_symbols and "." not in callee:
            return self.local_symbols[leaf], Confidence.EXACT
        if callee in self.in_repo_symbols:
            return self.in_repo_symbols[callee], Confidence.EXACT
        if callee in self.in_repo_modules:
            return self.in_repo_modules[callee], Confidence.CONSERVATIVE
        module_qualified = f"{self.module_name}.{leaf}"
        if module_qualified in self.local_symbols:
            confidence = Confidence.CONSERVATIVE if "." in callee else Confidence.EXACT
            return self.local_symbols[module_qualified], confidence
        if leaf in _SKIP_CALL_LEAVES:
            return None, Confidence.EXACT
        external = self.acc.add_node(
            NodeKind.SYMBOL,
            f"external:{callee}",
            self.path,
            start,
            end,
            confidence=Confidence.CONSERVATIVE,
        )
        return external, Confidence.CONSERVATIVE

    def _emit_dynamic_callee(
        self,
        owner_id: str,
        node: ast.Call,
        callee: str,
        start: int,
        end: int,
    ) -> None:
        leaf = _leaf(callee)
        if leaf in {"eval", "exec", "compile"}:
            effect_id = self.acc.add_node(
                NodeKind.EFFECT,
                f"{self.module_name}:dynamic-exec",
                self.path,
                start,
                end,
                confidence=Confidence.OPAQUE,
            )
            self.acc.add_edge(EdgeKind.CONTAINS, self.module_id, effect_id, self.path, start, start)
            self.acc.add_edge(
                EdgeKind.EXECUTES, owner_id, effect_id, self.path, start, end, confidence=Confidence.OPAQUE
            )
            self.acc.add_edge(
                EdgeKind.CALLS, owner_id, effect_id, self.path, start, end, confidence=Confidence.OPAQUE
            )
            return
        if leaf in {"__import__", "import_module"}:
            literal = _constant_str(node.args[0]) if node.args else None
            if literal:
                target = self._imported_module_node(literal, start, end)
                self.acc.add_edge(
                    EdgeKind.IMPORTS,
                    self.module_id,
                    target,
                    self.path,
                    start,
                    end,
                    confidence=Confidence.CONSERVATIVE,
                )
                self.acc.add_edge(
                    EdgeKind.CALLS,
                    owner_id,
                    target,
                    self.path,
                    start,
                    end,
                    confidence=Confidence.CONSERVATIVE,
                )
            else:
                opaque = self.acc.add_node(
                    NodeKind.EFFECT,
                    f"{self.module_name}:dynamic-import",
                    self.path,
                    start,
                    end,
                    confidence=Confidence.OPAQUE,
                )
                self.acc.add_edge(EdgeKind.CONTAINS, self.module_id, opaque, self.path, start, start)
                self.acc.add_edge(
                    EdgeKind.IMPORTS, self.module_id, opaque, self.path, start, end, confidence=Confidence.OPAQUE
                )
                self.acc.add_edge(
                    EdgeKind.CALLS, owner_id, opaque, self.path, start, end, confidence=Confidence.OPAQUE
                )
            return
        if leaf == "setattr":
            effect_id = self.acc.add_node(
                NodeKind.EFFECT,
                f"{self.module_name}:dynamic-dispatch",
                self.path,
                start,
                end,
                confidence=Confidence.OPAQUE,
            )
            self.acc.add_edge(EdgeKind.CONTAINS, self.module_id, effect_id, self.path, start, start)
            self.acc.add_edge(
                EdgeKind.MUTATES, owner_id, effect_id, self.path, start, end, confidence=Confidence.OPAQUE
            )
            self.acc.add_edge(
                EdgeKind.CALLS, owner_id, effect_id, self.path, start, end, confidence=Confidence.OPAQUE
            )
            if node.args:
                receiver = _expression_name(node.args[0])
                attribute = _constant_str(node.args[1]) if len(node.args) > 1 else None
                self._widen_class_methods(
                    owner_id,
                    receiver,
                    attribute,
                    start,
                    end,
                    edge=EdgeKind.MUTATES,
                )
            return
        if leaf == "getattr":
            receiver = _expression_name(node.args[0]) if node.args else ""
            attribute = _constant_str(node.args[1]) if len(node.args) > 1 else None
            binding = _DynamicBinding(
                receiver=receiver,
                attribute=attribute,
                owner_qualified=self._owner_qualified(),
                span=(start, end),
            )
            self._emit_dynamic_call(owner_id, binding, start, end)

    def _emit_dynamic_call(
        self,
        owner_id: str,
        binding: _DynamicBinding,
        start: int,
        end: int,
    ) -> None:
        dynamic_id = self.acc.add_node(
            NodeKind.EFFECT,
            f"{self.module_name}:dynamic-dispatch",
            self.path,
            start,
            end,
            confidence=Confidence.OPAQUE,
        )
        self.acc.add_edge(EdgeKind.CONTAINS, self.module_id, dynamic_id, self.path, start, start)
        self.acc.add_edge(
            EdgeKind.CALLS, owner_id, dynamic_id, self.path, start, end, confidence=Confidence.OPAQUE
        )
        self._widen_class_methods(
            owner_id,
            binding.receiver,
            binding.attribute,
            start,
            end,
            edge=EdgeKind.CALLS,
        )

    def _widen_class_methods(
        self,
        owner_id: str,
        receiver: str,
        attribute: str | None,
        start: int,
        end: int,
        *,
        edge: EdgeKind,
    ) -> None:
        class_info = self._class_for_receiver(receiver)
        if class_info is None:
            if attribute:
                target_id, confidence = self._resolve_call_target(attribute, ast.Call(func=ast.Name(id=attribute, ctx=ast.Load()), args=[], keywords=[]), start, end)
                if target_id is not None:
                    self.acc.add_edge(
                        edge,
                        owner_id,
                        target_id,
                        self.path,
                        start,
                        end,
                        confidence=Confidence.CONSERVATIVE,
                    )
            return
        if attribute:
            method_qualified = f"{class_info.qualified}.{attribute}"
            if method_qualified in self.local_symbols:
                self.acc.add_edge(
                    edge,
                    owner_id,
                    self.local_symbols[method_qualified],
                    self.path,
                    start,
                    end,
                    confidence=Confidence.CONSERVATIVE,
                )
                return
        for method in class_info.methods:
            method_qualified = f"{class_info.qualified}.{method}"
            target_id = self.local_symbols.get(method_qualified)
            if target_id is None:
                target_id = self.acc.add_node(
                    NodeKind.SYMBOL, method_qualified, self.path, start, end, confidence=Confidence.CONSERVATIVE
                )
                self.local_symbols[method_qualified] = target_id
            self.acc.add_edge(
                edge,
                owner_id,
                target_id,
                self.path,
                start,
                end,
                confidence=Confidence.CONSERVATIVE,
            )
        if class_info.has_dynamic_attr:
            opaque = self.acc.add_node(
                NodeKind.EFFECT,
                f"{class_info.qualified}:opaque-attr",
                self.path,
                start,
                end,
                confidence=Confidence.OPAQUE,
            )
            self.acc.add_edge(
                edge, owner_id, opaque, self.path, start, end, confidence=Confidence.OPAQUE
            )

    def _class_for_receiver(self, receiver: str) -> _ClassInfo | None:
        if receiver in {"self", "cls"} and self.scope:
            qualified = f"{self.module_name}.{self.scope[0]}"
            return self.classes.get(qualified)
        resolved = self._resolve_alias(receiver)
        return self.classes.get(resolved) or self.classes.get(_leaf(resolved))

    def _resolve_relative(self, module: str, level: int) -> str:
        if level <= 0:
            return module
        parts = self.module_name.split(".")
        if PurePosixPath(self.path).name != "__init__.py":
            parts = parts[:-1]
        if level - 1 > len(parts):
            return module
        prefix = parts[: len(parts) - (level - 1)]
        if module:
            prefix.append(module)
        return ".".join(part for part in prefix if part)

    def _imported_module_node(self, module: str, start: int, end: int) -> str:
        if module in self.in_repo_modules:
            return self.in_repo_modules[module]
        confidence = Confidence.CONSERVATIVE
        return self.acc.add_node(
            NodeKind.MODULE,
            f"external:{module}",
            self.path,
            start,
            end,
            confidence=confidence,
        )

    def _imported_symbol_node(self, qualified: str, start: int, end: int) -> str:
        if qualified in self.in_repo_symbols:
            return self.in_repo_symbols[qualified]
        if qualified in self.local_symbols:
            return self.local_symbols[qualified]
        return self.acc.add_node(
            NodeKind.SYMBOL,
            f"external:{qualified}",
            self.path,
            start,
            end,
            confidence=Confidence.CONSERVATIVE,
        )

    def _symbol_or_interface(
        self,
        name: str,
        start: int,
        end: int,
        confidence: Confidence,
    ) -> str:
        if name in self.local_symbols:
            return self.local_symbols[name]
        if name in self.classes:
            qualified = self.classes[name].qualified
            if qualified in self.local_symbols:
                return self.local_symbols[qualified]
        if _leaf(name) in {"Protocol", "ABC"} or name in _PROTOCOL_BASES:
            return self.acc.add_node(
                NodeKind.INTERFACE, name, self.path, start, end, confidence=confidence
            )
        return self.acc.add_node(
            NodeKind.SYMBOL,
            f"external:{name}",
            self.path,
            start,
            end,
            confidence=confidence,
        )


class ArchitectureGraphBuilder:
    """Extract a bounded ArchitectureIR graph from declared repository sources."""

    def __init__(
        self,
        repository_root: str | Path | None = None,
        *,
        repository_tree: str,
        freshness: str = DEFAULT_FRESHNESS,
        sources: Mapping[str, str] | None = None,
        inclusions: Sequence[str] | None = None,
        exclusions: Sequence[str] | None = None,
        protected_paths: Sequence[str] | None = None,
        gitlink_prefixes: Sequence[str] | None = None,
        max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
        max_files: int = DEFAULT_MAX_FILES,
        max_facts: int = DEFAULT_MAX_FACTS,
        extractor_identity: str = EXTRACTOR_IDENTITY,
    ) -> None:
        if not isinstance(max_source_bytes, int) or isinstance(max_source_bytes, bool) or max_source_bytes < 1:
            raise _error("max_source_bytes must be a positive integer")
        if not isinstance(max_files, int) or isinstance(max_files, bool) or max_files < 1:
            raise _error("max_files must be a positive integer")
        if not isinstance(max_facts, int) or isinstance(max_facts, bool) or max_facts < 1:
            raise _error("max_facts must be a positive integer")
        self.repository_tree = _require_text(
            repository_tree, "repository_tree", error_type=ArchitectureGraphBuilderError
        )
        self.freshness = _require_text(
            freshness, "freshness", error_type=ArchitectureGraphBuilderError
        )
        self.extractor_identity = _require_text(
            extractor_identity, "extractor_identity", error_type=ArchitectureGraphBuilderError
        )
        self.max_source_bytes = max_source_bytes
        self.max_files = max_files
        self.max_facts = max_facts
        self.protected_paths = tuple(
            normalize_relative_path(item, name="protected path")
            for item in (protected_paths if protected_paths is not None else DEFAULT_PROTECTED_PATHS)
        )
        self.gitlink_prefixes = tuple(
            normalize_relative_path(item, name="gitlink prefix")
            for item in (gitlink_prefixes if gitlink_prefixes is not None else DEFAULT_GITLINK_PREFIXES)
        )
        self.exclusions = tuple(
            normalize_relative_path(item, name="exclusion")
            for item in (exclusions or ())
        ) + DEFAULT_WALK_EXCLUSIONS
        self.inclusions = tuple(
            normalize_relative_path(item, name="inclusion")
            for item in (inclusions or ())
        )
        self._sources = {
            normalize_relative_path(path, name="source path"): text
            for path, text in dict(sources or {}).items()
        }
        for path, text in self._sources.items():
            if not isinstance(text, str):
                raise _error(f"source for {path!r} must be text")
            self._reject_logical(path)
            encoded = text.encode("utf-8")
            if len(encoded) > self.max_source_bytes:
                raise _error(f"source for {path!r} exceeds the hard byte bound")
        if repository_root is None:
            self.root: Path | None = None
        else:
            root = Path(repository_root)
            if not root.exists() or not root.is_dir():
                raise _error("repository_root must be an existing directory")
            self.root = root.resolve()

    @classmethod
    def from_sources(
        cls,
        sources: Mapping[str, str],
        *,
        repository_tree: str,
        freshness: str = DEFAULT_FRESHNESS,
        **kwargs: Any,
    ) -> "ArchitectureGraphBuilder":
        return cls(
            None,
            repository_tree=repository_tree,
            freshness=freshness,
            sources=sources,
            **kwargs,
        )

    def build(self) -> ArchitectureIR:
        files = self._collect_files()
        if len(files) > self.max_files:
            raise _error("file count exceeds the hard bound")
        acc = _Accumulator(
            repository_tree=self.repository_tree,
            freshness=self.freshness,
            extractor_identity=self.extractor_identity,
            max_facts=self.max_facts,
        )
        if not files:
            return acc.graph()
        first_path, first_source = next(iter(files.items()))
        first_start, first_end = _file_span_lines(first_source)
        repo_id = acc.add_node(
            NodeKind.REPOSITORY, self.repository_tree, first_path, first_start, first_end
        )
        python_files = {
            path: source
            for path, source in files.items()
            if PurePosixPath(path).suffix in _PYTHON_SUFFIXES
        }
        json_files = {
            path: source
            for path, source in files.items()
            if PurePosixPath(path).suffix in _JSON_SUFFIXES
        }
        module_ids: dict[str, str] = {}
        file_ids: dict[str, str] = {}
        package_ids: dict[str, str] = {}
        in_repo_modules: dict[str, str] = {}
        in_repo_symbols: dict[str, str] = {}
        for path, source in python_files.items():
            start, end = _file_span_lines(source)
            module = module_name_from_path(path)
            file_id = acc.add_node(NodeKind.FILE, path, path, start, end)
            file_ids[path] = file_id
            module_id = acc.add_node(NodeKind.MODULE, module, path, start, end)
            module_ids[path] = module_id
            in_repo_modules[module] = module_id
            acc.add_edge(EdgeKind.CONTAINS, repo_id, module_id, path, start, start)
            acc.add_edge(EdgeKind.CONTAINS, module_id, file_id, path, start, end)
            acc.add_edge(EdgeKind.CONTAINS, repo_id, file_id, path, start, start)
            package = package_name_from_module(module)
            package_path = str(PurePosixPath(path).parent)
            if package and package_path not in {".", ""}:
                pkg_file = (
                    f"{package_path}/__init__.py"
                    if f"{package_path}/__init__.py" in python_files
                    else path
                )
                pkg_source = python_files.get(pkg_file, source)
                pkg_start, pkg_end = _file_span_lines(pkg_source)
                package_id = package_ids.get(package)
                if package_id is None:
                    package_id = acc.add_node(NodeKind.PACKAGE, package, pkg_file, pkg_start, pkg_end)
                    package_ids[package] = package_id
                    acc.add_edge(EdgeKind.CONTAINS, repo_id, package_id, pkg_file, pkg_start, pkg_start)
                acc.add_edge(EdgeKind.CONTAINS, package_id, module_id, path, start, start)
            if _is_test_path(path):
                test_mod = acc.add_node(NodeKind.TEST, f"module:{module}", path, start, end)
                acc.add_edge(EdgeKind.CONTAINS, module_id, test_mod, path, start, start)
                acc.add_edge(EdgeKind.TESTS, test_mod, module_id, path, start, start)
        # Pre-index class/function symbol ids so cross-module imports resolve.
        trees: dict[str, ast.AST] = {}
        extractors: dict[str, _PythonExtractor] = {}
        for path, source in python_files.items():
            try:
                tree = ast.parse(source, filename=path, type_comments=True)
            except (SyntaxError, ValueError):
                start, end = _file_span_lines(source)
                acc.add_node(
                    NodeKind.FILE,
                    f"opaque:{path}",
                    path,
                    start,
                    end,
                    confidence=Confidence.OPAQUE,
                )
                continue
            trees[path] = tree
            extractor = _PythonExtractor(
                acc=acc,
                path=path,
                source=source,
                module_id=module_ids[path],
                file_id=file_ids[path],
                module_name=module_name_from_path(path),
                in_repo_modules=in_repo_modules,
                in_repo_symbols=in_repo_symbols,
            )
            extractors[path] = extractor
            extractor.collect_definitions(tree)
            self._index_symbols(extractor, tree, in_repo_symbols)
        for path, extractor in extractors.items():
            extractor.in_repo_symbols = in_repo_symbols
            extractor.visit(trees[path])
        for path, source in json_files.items():
            self._extract_json(acc, repo_id, path, source, in_repo_symbols, in_repo_modules)
        return acc.graph()

    def _index_symbols(
        self,
        extractor: _PythonExtractor,
        tree: ast.AST,
        in_repo_symbols: dict[str, str],
    ) -> None:
        module = extractor.module_name

        def walk(node: ast.AST, prefix: tuple[str, ...]) -> None:
            body: Sequence[ast.AST] = ()
            if isinstance(node, ast.Module):
                body = node.body
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = node.body
            for child in body:
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualified = ".".join((module, *prefix, child.name))
                    start, end = _line_span(extractor.source, child)
                    symbol_id = extractor.acc.add_node(
                        NodeKind.SYMBOL, qualified, extractor.path, start, end
                    )
                    extractor.local_symbols[qualified] = symbol_id
                    extractor.local_symbols.setdefault(child.name, symbol_id)
                    in_repo_symbols[qualified] = symbol_id
                    walk(child, (*prefix, child.name))

        walk(tree, ())

    def _extract_json(
        self,
        acc: _Accumulator,
        repo_id: str,
        path: str,
        source: str,
        in_repo_symbols: Mapping[str, str],
        in_repo_modules: Mapping[str, str],
    ) -> None:
        start, end = _file_span_lines(source)
        file_id = acc.add_node(NodeKind.FILE, path, path, start, end)
        acc.add_edge(EdgeKind.CONTAINS, repo_id, file_id, path, start, start)
        try:
            payload = json.loads(source) if source.strip() else None
        except json.JSONDecodeError:
            acc.add_node(
                NodeKind.ARTIFACT, f"opaque:{path}", path, start, end, confidence=Confidence.OPAQUE
            )
            return
        is_schema = _is_schema_path(path) or (
            isinstance(payload, dict) and bool({"$schema", "properties", "type"} & set(payload))
        )
        is_proof = _is_proof_path(path) or (
            isinstance(payload, dict)
            and bool({"proves", "proof", "obligation", "proof_obligation"} & set(payload))
        )
        if is_schema:
            schema_id = acc.add_node(NodeKind.SCHEMA, path, path, start, end)
            acc.add_edge(EdgeKind.CONTAINS, file_id, schema_id, path, start, end)
            acc.add_edge(EdgeKind.CONTAINS, repo_id, schema_id, path, start, start)
        if is_proof:
            proof_id = acc.add_node(NodeKind.PROOF, path, path, start, end)
            acc.add_edge(EdgeKind.CONTAINS, file_id, proof_id, path, start, end)
            acc.add_edge(EdgeKind.CONTAINS, repo_id, proof_id, path, start, start)
            target_name = ""
            if isinstance(payload, dict):
                for key in ("proves", "obligation", "proof", "target"):
                    value = payload.get(key)
                    if isinstance(value, str) and value:
                        target_name = value
                        break
            if target_name:
                target_id = in_repo_symbols.get(target_name) or in_repo_modules.get(target_name)
                if target_id is None:
                    target_id = acc.add_node(
                        NodeKind.SYMBOL,
                        f"external:{target_name}",
                        path,
                        start,
                        end,
                        confidence=Confidence.CONSERVATIVE,
                    )
                acc.add_edge(
                    EdgeKind.PROVES,
                    proof_id,
                    target_id,
                    path,
                    start,
                    end,
                    confidence=Confidence.CONSERVATIVE if target_name not in in_repo_symbols else Confidence.EXACT,
                )
        if not is_schema and not is_proof:
            acc.add_node(NodeKind.ARTIFACT, path, path, start, end, confidence=Confidence.CONSERVATIVE)

    def _collect_files(self) -> dict[str, str]:
        files: dict[str, str] = {}
        if self._sources:
            for path, text in sorted(self._sources.items()):
                if self._is_excluded(path):
                    continue
                if self.inclusions and not any(logical_path_under(path, item) for item in self.inclusions):
                    continue
                files[path] = text
            return files
        if self.root is None:
            raise _error("repository_root or sources is required")
        candidates: list[str] = []
        if self.inclusions:
            for inclusion in self.inclusions:
                physical = self._contained_path(inclusion)
                if physical.is_file():
                    candidates.append(inclusion)
                elif physical.is_dir():
                    candidates.extend(self._walk_directory(inclusion, physical))
                else:
                    raise _error(f"inclusion does not exist: {inclusion}")
        else:
            candidates.extend(self._walk_directory("", self.root))
        seen: set[str] = set()
        for relative in candidates:
            if relative in seen or self._is_excluded(relative):
                continue
            seen.add(relative)
            suffix = PurePosixPath(relative).suffix
            if suffix not in _PYTHON_SUFFIXES and suffix not in _JSON_SUFFIXES:
                continue
            physical = self._contained_path(relative)
            files[relative] = self._read_text(relative, physical)
            if len(files) > self.max_files:
                raise _error("file count exceeds the hard bound")
        return dict(sorted(files.items()))

    def _walk_directory(self, prefix: str, directory: Path) -> list[str]:
        found: list[str] = []
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise _error(f"cannot list {prefix or '.'}: {exc}") from exc
        for entry in entries:
            name = entry.name
            relative = f"{prefix}/{name}" if prefix else name
            try:
                normalize_relative_path(relative)
            except ArchitectureGraphEscapeError:
                continue
            if name in DEFAULT_WALK_EXCLUSIONS or self._is_excluded(relative):
                continue
            try:
                st = entry.lstat()
            except OSError as exc:
                raise _error(f"cannot stat {relative}: {exc}") from exc
            if stat.S_ISLNK(st.st_mode):
                raise _error(
                    f"symlink escape rejected before I/O: {relative}",
                    error_type=ArchitectureGraphEscapeError,
                )
            if stat.S_ISDIR(st.st_mode):
                if (entry / ".git").exists():
                    raise _error(
                        f"submodule escape rejected before I/O: {relative}",
                        error_type=ArchitectureGraphEscapeError,
                    )
                found.extend(self._walk_directory(relative, entry))
            elif stat.S_ISREG(st.st_mode):
                found.append(relative)
        return found

    def _is_excluded(self, relative: str) -> bool:
        path = normalize_relative_path(relative)
        parts = PurePosixPath(path).parts
        if any(part in DEFAULT_WALK_EXCLUSIONS for part in parts):
            return True
        for exclusion in self.exclusions:
            if logical_path_under(path, exclusion):
                return True
        return False

    def _reject_logical(self, relative: str) -> None:
        path = normalize_relative_path(relative)
        for protected in self.protected_paths:
            if logical_path_under(path, protected):
                raise _error(
                    f"protected path rejected: {path}",
                    error_type=ArchitectureGraphEscapeError,
                )
        for prefix in self.gitlink_prefixes:
            if logical_path_under(path, prefix):
                raise _error(
                    f"submodule escape rejected before I/O: {path}",
                    error_type=ArchitectureGraphEscapeError,
                )

    def _contained_path(self, relative: str) -> Path:
        if self.root is None:
            raise _error("repository_root is required for filesystem extraction")
        path = normalize_relative_path(relative)
        self._reject_logical(path)
        current = self.root
        parts = tuple(PurePosixPath(path).parts)
        root_resolved = self.root
        for index, part in enumerate(parts):
            current = current / part
            try:
                st = current.lstat()
            except FileNotFoundError as exc:
                raise _error(f"path does not exist: {path}") from exc
            if stat.S_ISLNK(st.st_mode):
                raise _error(
                    f"symlink escape rejected before I/O: {path}",
                    error_type=ArchitectureGraphEscapeError,
                )
            if stat.S_ISDIR(st.st_mode) and (current / ".git").exists() and index + 1 < len(parts):
                raise _error(
                    f"submodule escape rejected before I/O: {path}",
                    error_type=ArchitectureGraphEscapeError,
                )
        if current.exists() and not current.is_symlink():
            try:
                current.resolve().relative_to(root_resolved)
            except ValueError as exc:
                raise _error(
                    f"path escapes repository root: {path}",
                    error_type=ArchitectureGraphEscapeError,
                ) from exc
        return current

    def _read_text(self, relative: str, physical: Path) -> str:
        size = physical.stat().st_size
        if size > self.max_source_bytes:
            raise _error(f"source for {relative!r} exceeds the hard byte bound")
        try:
            text = physical.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise _error(f"source for {relative!r} is not valid UTF-8") from exc
        if len(text.encode("utf-8")) > self.max_source_bytes:
            raise _error(f"source for {relative!r} exceeds the hard byte bound")
        return text


def build_architecture_graph(
    repository_root: str | Path | None = None,
    *,
    repository_tree: str,
    freshness: str = DEFAULT_FRESHNESS,
    sources: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> ArchitectureIR:
    """Extract an ArchitectureIR graph from a repository root or in-memory sources."""

    builder = ArchitectureGraphBuilder(
        repository_root,
        repository_tree=repository_tree,
        freshness=freshness,
        sources=sources,
        **kwargs,
    )
    return builder.build()


def extract_architecture_graph(
    sources: Mapping[str, str],
    *,
    repository_tree: str,
    freshness: str = DEFAULT_FRESHNESS,
    **kwargs: Any,
) -> ArchitectureIR:
    """Extract an ArchitectureIR graph from in-memory repository-relative sources."""

    return ArchitectureGraphBuilder.from_sources(
        sources,
        repository_tree=repository_tree,
        freshness=freshness,
        **kwargs,
    ).build()
