"""Deterministic, bounded static dependency tracing for pytest (PTR-020).

``StaticTestDependencyTrace@1`` is compact evidence derived from an
``AnalysisASTIndex`` and the exact source files represented by that index.  It
is deliberately fail-closed: a construct which cannot be resolved without
executing user code is retained as an :class:`UnknownDependencyFrontier`.

Source is read only while parsing.  Trace rows contain repository-relative
paths, content hashes, AST record identities, symbols, and source spans; they
never contain a source body or an absolute path.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_ast_index import (
    AnalysisASTIndex,
    IndexedASTPath,
)
from ipfs_accelerate_py.agent_supervisor.analysis.test_execution_identity import (
    ContentIdentity,
    mint_content_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
)

STATIC_TEST_DEPENDENCY_TRACE_INTERFACE: Final = "StaticTestDependencyTrace@1"
STATIC_TEST_DEPENDENCY_TRACER_INTERFACE: Final = "StaticTestDependencyTracer@1"
STATIC_TRACE_LIMITS_INTERFACE: Final = "StaticTraceLimits@1"
STATIC_TRACE_ANALYZER_INTERFACE: Final = "StaticTraceAnalyzer@1"

STATIC_TEST_DEPENDENCY_TRACE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/static-test-dependency-trace@1"
)
STATIC_TRACE_LIMITS_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/static-trace-limits@1"
STATIC_TRACE_ANALYZER_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/static-trace-analyzer@1"

_SAFE_TEXT_RE: Final = re.compile(r"^[A-Za-z0-9_.:@/+ <>*=-]{1,512}$")
_CONFIG_FILES: Final = ("pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini")
_REFLECTION_CALLS: Final = frozenset(
    {
        "compile",
        "delattr",
        "eval",
        "exec",
        "getattr",
        "globals",
        "hasattr",
        "locals",
        "setattr",
        "vars",
    }
)
_DYNAMIC_IMPORT_CALLS: Final = frozenset({"__import__", "import_module", "importlib.import_module"})
_EFFECT_CALLS: Final[Mapping[str, frozenset[str]]] = {
    "subprocess": frozenset(
        {
            "Popen",
            "call",
            "check_call",
            "check_output",
            "os.popen",
            "os.spawnl",
            "os.spawnv",
            "os.system",
            "subprocess.Popen",
            "subprocess.call",
            "subprocess.check_call",
            "subprocess.check_output",
            "subprocess.run",
        }
    ),
    "network": frozenset(
        {
            "aiohttp.ClientSession",
            "httpx.get",
            "httpx.post",
            "requests.get",
            "requests.post",
            "socket.create_connection",
            "socket.socket",
            "urllib.request.urlopen",
            "urlopen",
        }
    ),
    "clock": frozenset({"datetime.now", "datetime.utcnow", "time.monotonic", "time.time"}),
    "randomness": frozenset(
        {
            "random.choice",
            "random.random",
            "random.randrange",
            "secrets.token_bytes",
            "secrets.token_hex",
            "uuid.uuid4",
        }
    ),
    "environment": frozenset({"os.environ.get", "os.getenv", "os.putenv", "os.unsetenv"}),
    "hardware": frozenset(
        {
            "cuda.is_available",
            "jax.devices",
            "torch.cuda.is_available",
            "torch.cuda.get_device_name",
        }
    ),
}
_SAFE_DECORATORS: Final = frozenset(
    {
        "abstractmethod",
        "abc.abstractmethod",
        "classmethod",
        "dataclass",
        "functools.cache",
        "functools.lru_cache",
        "property",
        "pytest.fixture",
        "fixture",
        "pytest.mark.asyncio",
        "pytest.mark.filterwarnings",
        "pytest.mark.parametrize",
        "pytest.mark.skip",
        "pytest.mark.skipif",
        "pytest.mark.usefixtures",
        "pytest.mark.xfail",
        "staticmethod",
    }
)


class StaticTraceError(ValueError):
    """Invalid static trace input or identity material."""

    __test__ = False


@dataclass(frozen=True)
class StaticTraceLimits:
    """Hard analysis limits which participate in the trace identity."""

    __test__: ClassVar[bool] = False

    max_files: int = 256
    max_edges: int = 2_048
    max_frontier: int = 512
    max_depth: int = 32
    max_source_bytes: int = 2 * 1_048_576
    max_ast_nodes: int = 100_000
    max_data_bytes: int = 8 * 1_048_576
    max_symbols_per_file: int = 2_048
    max_text_chars: int = 512

    _BOUNDS: ClassVar[Mapping[str, tuple[int, int]]] = {
        "max_files": (1, 8_192),
        "max_edges": (1, 65_536),
        "max_frontier": (1, 16_384),
        "max_depth": (1, 256),
        "max_source_bytes": (1, 32 * 1_048_576),
        "max_ast_nodes": (1, 1_000_000),
        "max_data_bytes": (1, 128 * 1_048_576),
        "max_symbols_per_file": (1, 16_384),
        "max_text_chars": (32, 4_096),
    }

    def __post_init__(self) -> None:
        for name, (minimum, maximum) in self._BOUNDS.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise StaticTraceError(f"{name} must be an integer in [{minimum}, {maximum}]")

    @property
    def interface(self) -> str:
        return STATIC_TRACE_LIMITS_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STATIC_TRACE_LIMITS_SCHEMA,
            "interface": STATIC_TRACE_LIMITS_INTERFACE,
            **{name: getattr(self, name) for name in sorted(self._BOUNDS)},
        }


@dataclass(frozen=True)
class UnknownDependencyFrontier:
    """One typed, compact edge the static analyzer could not close."""

    __test__: ClassVar[bool] = False

    kind: str
    source_path: str
    source_symbol: str = ""
    target: str = ""
    line_start: int = 0
    line_end: int = 0

    def __post_init__(self) -> None:
        kind = _safe_public_text(self.kind, "frontier kind")
        path = _repo_path(self.source_path, allow_empty=True)
        symbol = _safe_public_text(self.source_symbol, "source symbol", allow_empty=True)
        target = _safe_public_text(self.target, "frontier target", allow_empty=True)
        start = _line_number(self.line_start)
        end = _line_number(self.line_end)
        if end and start and end < start:
            raise StaticTraceError("frontier line_end precedes line_start")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source_path", path)
        object.__setattr__(self, "source_symbol", symbol)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "line_start", start)
        object.__setattr__(self, "line_end", end)

    @property
    def frontier_id(self) -> str:
        return (
            "static-frontier:sha256:"
            + hashlib.sha256(canonical_json_bytes(self._content_dict())).hexdigest()
        )

    def _content_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source_path": self.source_path,
            "source_symbol": self.source_symbol,
            "target": self.target,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"frontier_id": self.frontier_id, **self._content_dict()}


@dataclass(frozen=True)
class StaticTestDependencyTrace:
    """Immutable canonical result of one static closure computation."""

    __test__: ClassVar[bool] = False

    content_identity: ContentIdentity
    retained_canonical_bytes: bytes
    unknown_frontier: tuple[UnknownDependencyFrontier, ...]
    analyzed_file_count: int
    dependency_edge_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.content_identity, ContentIdentity):
            raise StaticTraceError("static trace requires a ContentIdentity")
        if type(self.retained_canonical_bytes) is not bytes:
            raise StaticTraceError("retained_canonical_bytes must be exact bytes")
        if self.retained_canonical_bytes != self.content_identity.canonical_bytes:
            raise StaticTraceError("trace bytes do not match ContentIdentity")
        frontiers = tuple(sorted(self.unknown_frontier, key=lambda item: item.frontier_id))
        if len({item.frontier_id for item in frontiers}) != len(frontiers):
            raise StaticTraceError("unknown frontier rows must be unique")
        object.__setattr__(self, "unknown_frontier", frontiers)
        for value in (self.analyzed_file_count, self.dependency_edge_count):
            if type(value) is not int or value < 0:
                raise StaticTraceError("trace counters must be non-negative integers")

    @property
    def interface(self) -> str:
        return STATIC_TEST_DEPENDENCY_TRACE_INTERFACE

    @property
    def schema(self) -> str:
        return STATIC_TEST_DEPENDENCY_TRACE_SCHEMA

    @property
    def complete(self) -> bool:
        return not self.unknown_frontier

    @property
    def completeness(self) -> str:
        return "complete" if self.complete else "incomplete"

    @property
    def cid(self) -> str:
        return self.content_identity.cid

    @property
    def trace_cid(self) -> str:
        return self.cid

    @property
    def root_cid(self) -> str:
        return self.cid

    @property
    def canonical_bytes(self) -> bytes:
        return self.retained_canonical_bytes

    @property
    def unknown_dependency_frontier(self) -> tuple[UnknownDependencyFrontier, ...]:
        return self.unknown_frontier

    def to_dict(self) -> dict[str, Any]:
        value = json.loads(self.retained_canonical_bytes.decode("utf-8"))
        if not isinstance(value, dict):  # pragma: no cover - construction invariant
            raise StaticTraceError("static trace canonical bytes are not an object")
        return value

    def verify(self) -> StaticTestDependencyTrace:
        self.content_identity.verify()
        if canonical_json_bytes(self.to_dict()) != self.retained_canonical_bytes:
            raise StaticTraceError("static trace bytes are not canonical")
        payload_frontier = tuple(
            item["frontier_id"] for item in self.to_dict().get("unknown_frontier", ())
        )
        if payload_frontier != tuple(item.frontier_id for item in self.unknown_frontier):
            raise StaticTraceError("frontier projection does not match canonical trace")
        return self


@dataclass
class _ParsedFile:
    indexed: IndexedASTPath
    tree: ast.Module
    roles: set[str]
    depth: int


@dataclass(frozen=True)
class _FixtureDefinition:
    name: str
    path: str
    symbol: str
    node: ast.FunctionDef | ast.AsyncFunctionDef


def _safe_public_text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    text = " ".join(str(value or "").split())
    if not text and allow_empty:
        return ""
    if not _SAFE_TEXT_RE.fullmatch(text):
        raise StaticTraceError(f"{field} is not bounded public text")
    return text


def _repo_path(value: Any, *, allow_empty: bool = False) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    if not raw and allow_empty:
        return ""
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise StaticTraceError("path must be repository-relative and contained")
    return path.as_posix()


def _line_number(value: Any) -> int:
    if type(value) is not int or value < 0 or value > (1 << 31) - 1:
        raise StaticTraceError("source lines must be non-negative bounded integers")
    return value


def _expression_name(node: ast.AST | None) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expression_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _span(node: ast.AST) -> tuple[int, int]:
    start = int(getattr(node, "lineno", 0) or 0)
    return start, int(getattr(node, "end_lineno", start) or start)


def _literal_strings(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        result: list[str] = []
        for item in node.elts:
            values = _literal_strings(item)
            if values is None:
                return None
            result.extend(values)
        return tuple(result)
    return None


class StaticTestDependencyTracer:
    """Compute a conservative repository-local dependency closure."""

    __test__ = False

    def __init__(
        self,
        analysis_index: AnalysisASTIndex | None = None,
        repository_root: str | os.PathLike[str] | None = None,
        *,
        ast_index: AnalysisASTIndex | None = None,
        limits: StaticTraceLimits | None = None,
        identity_minter: Callable[[Any], ContentIdentity] = mint_content_identity,
    ) -> None:
        if analysis_index is not None and ast_index is not None:
            raise StaticTraceError("provide only one of analysis_index or ast_index")
        index = analysis_index if analysis_index is not None else ast_index
        if not isinstance(index, AnalysisASTIndex):
            raise StaticTraceError("analysis_index must be an AnalysisASTIndex")
        if repository_root is None:
            raise StaticTraceError("repository_root is required")
        try:
            root = Path(os.fspath(repository_root)).resolve(strict=True)
        except (OSError, TypeError, ValueError) as exc:
            raise StaticTraceError("repository_root is unavailable") from exc
        if not root.is_dir():
            raise StaticTraceError("repository_root must be a directory")
        if not callable(identity_minter):
            raise StaticTraceError("identity_minter must be callable")

        self.analysis_index = index
        self.repository_root = root
        self.limits = limits or StaticTraceLimits()
        self._identity_minter = identity_minter
        self._records = {item.path: item for item in index.path_records}
        self._modules: dict[str, list[str]] = {}
        for item in index.path_records:
            for alias in self._module_aliases(item):
                self._modules.setdefault(alias, []).append(item.path)
        for paths in self._modules.values():
            paths.sort()
        self._reset()

    @staticmethod
    def _module_aliases(indexed: IndexedASTPath) -> tuple[str, ...]:
        module = indexed.module
        parts = module.split(".") if module else []
        aliases = {module} if module else set()
        # Source-layout repositories commonly index ``src/pkg/x.py`` while
        # Python imports it as ``pkg.x``.  Only this explicit layout alias is
        # added; arbitrary suffix matching would make resolution ambiguous.
        if parts and parts[0] in {"src", "lib", "python"} and len(parts) > 1:
            aliases.add(".".join(parts[1:]))
        return tuple(sorted(aliases))

    def _reset(self) -> None:
        self._parsed: dict[str, _ParsedFile] = {}
        self._nodes: dict[bytes, dict[str, Any]] = {}
        self._edges: dict[bytes, dict[str, Any]] = {}
        self._frontiers: dict[str, UnknownDependencyFrontier] = {}
        self._fixtures: dict[str, list[_FixtureDefinition]] = {}
        self._root_path = ""
        self._root_symbol = ""
        self._bound_kinds: set[str] = set()
        self._source_hashes_verified = True
        self._parser_healthy = True
        self._result: StaticTestDependencyTrace | None = None

    @property
    def result(self) -> StaticTestDependencyTrace | None:
        return self._result

    @property
    def trace_result(self) -> StaticTestDependencyTrace | None:
        return self._result

    def trace(
        self,
        test_path: str,
        test_symbol: str = "",
        *,
        node_id: str = "",
    ) -> StaticTestDependencyTrace:
        """Trace one indexed test module/symbol and return canonical evidence."""

        self._reset()
        self._root_path = _repo_path(test_path)
        selected = (test_symbol or (node_id.rsplit("::", 1)[-1] if node_id else "")).split("[", 1)[
            0
        ]
        self._root_symbol = _safe_public_text(selected, "test symbol", allow_empty=True)

        # Pytest configuration and ancestor conftests affect collection and
        # fixture/hook semantics even when the test imports none of them.
        self._record_configuration_files()
        conftests = self._ancestor_conftests(self._root_path)
        for path in conftests:
            self._analyze_file(path, role="conftest", depth=0)
        self._analyze_file(self._root_path, role="test", depth=0)
        self._resolve_root_fixtures()
        self._record_conftest_hooks(conftests)

        analyzer, analyzer_cid = self._analyzer_identity()
        frontiers = tuple(sorted(self._frontiers.values(), key=lambda item: item.frontier_id))
        payload = {
            "schema": STATIC_TEST_DEPENDENCY_TRACE_SCHEMA,
            "interface": STATIC_TEST_DEPENDENCY_TRACE_INTERFACE,
            "root": {"path": self._root_path, "symbol": self._root_symbol},
            "analysis_ast_index_id": self.analysis_index.index_id,
            "limits": self.limits.to_dict(),
            "analyzer": analyzer,
            "analyzer_cid": analyzer_cid,
            "dependencies": {
                "nodes": [self._nodes[key] for key in sorted(self._nodes)],
                "edges": [self._edges[key] for key in sorted(self._edges)],
            },
            "unknown_frontier": [item.to_dict() for item in frontiers],
            "health": {
                "complete": not frontiers,
                "source_hashes_verified": self._source_hashes_verified,
                "parser_healthy": self._parser_healthy,
                "analysis_bounds_reached": sorted(self._bound_kinds),
                "analyzed_file_count": len(self._parsed),
                "dependency_edge_count": len(self._edges),
                "unknown_frontier_count": len(frontiers),
            },
        }
        expected = canonical_json_bytes(payload)
        identity = self._identity_minter(payload)
        if not isinstance(identity, ContentIdentity):
            raise StaticTraceError("identity provider did not return ContentIdentity")
        identity.verify()
        if identity.canonical_bytes != expected:
            raise StaticTraceError("identity provider canonical bytes do not match trace")
        result = StaticTestDependencyTrace(
            content_identity=identity,
            retained_canonical_bytes=expected,
            unknown_frontier=frontiers,
            analyzed_file_count=len(self._parsed),
            dependency_edge_count=len(self._edges),
        )
        self._result = result
        return result

    analyze = trace
    build = trace

    def _analyzer_identity(self) -> tuple[dict[str, Any], str]:
        try:
            source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        except OSError as exc:  # pragma: no cover - installed source is expected
            raise StaticTraceError("static analyzer source is unavailable") from exc
        analyzer = {
            "schema": STATIC_TRACE_ANALYZER_SCHEMA,
            "interface": STATIC_TRACE_ANALYZER_INTERFACE,
            "tracer_interface": STATIC_TEST_DEPENDENCY_TRACER_INTERFACE,
            "implementation": "python-ast-index-closure",
            "implementation_source_sha256": source_digest,
            "python_implementation": sys.implementation.name,
            "python_version": ".".join(str(value) for value in sys.version_info[:3]),
            "ast_schema": "python-ast@1",
            "resolution_schema": "repository-local-import-fixture-effect@1",
        }
        identity = self._identity_minter(analyzer)
        if not isinstance(identity, ContentIdentity):
            raise StaticTraceError("identity provider did not return ContentIdentity")
        identity.verify()
        if identity.canonical_bytes != canonical_json_bytes(analyzer):
            raise StaticTraceError("analyzer identity bytes do not match input")
        return analyzer, identity.cid

    def _ancestor_conftests(self, path: str) -> tuple[str, ...]:
        parent = PurePosixPath(path).parent
        candidates: list[str] = []
        current = PurePosixPath()
        for part in parent.parts:
            current /= part
            candidate = (current / "conftest.py").as_posix()
            if candidate in self._records:
                candidates.append(candidate)
        if "conftest.py" in self._records:
            candidates.insert(0, "conftest.py")
        return tuple(dict.fromkeys(candidates))

    def _record_configuration_files(self) -> None:
        for name in _CONFIG_FILES:
            candidate = self.repository_root / name
            if not candidate.exists():
                continue
            if candidate.is_symlink() or not candidate.is_file():
                self._frontier("missing_file", "", target=name)
                continue
            self._record_data_node(candidate, name, role="config")
            self._edge("config", self._root_path, "", name, "", 0, 0)

    def _analyze_file(self, path: str, *, role: str, depth: int) -> None:
        path = _repo_path(path)
        if len(path) > self.limits.max_text_chars:
            self._bound("text", "", target="path")
            return
        prior = self._parsed.get(path)
        if prior is not None:
            prior.roles.add(role)
            self._refresh_code_node(prior)
            return
        if depth > self.limits.max_depth:
            self._bound("depth", path)
            return
        if len(self._parsed) >= self.limits.max_files:
            self._bound("files", path)
            return
        indexed = self._records.get(path)
        if indexed is None:
            self._frontier("missing_file", path, target=path)
            return
        absolute = self.repository_root / Path(path)
        try:
            resolved = absolute.resolve(strict=True)
            resolved.relative_to(self.repository_root)
        except (OSError, ValueError):
            self._frontier("missing_file", path, target=path)
            return
        if absolute.is_symlink() or not resolved.is_file():
            self._frontier("missing_file", path, target=path)
            return
        try:
            data = resolved.read_bytes()
        except OSError:
            self._frontier("missing_file", path, target=path)
            return
        if len(data) > self.limits.max_source_bytes:
            self._bound("source_bytes", path)
            return
        actual_hash = hashlib.sha256(data).hexdigest()
        claimed = indexed.source_sha256.removeprefix("sha256:")
        if not claimed or claimed != actual_hash:
            self._source_hashes_verified = False
            self._frontier("stale_ast_index", path, target=indexed.record_id)
            return
        try:
            source = data.decode("utf-8")
            tree = ast.parse(source, filename=path)
        except (UnicodeDecodeError, SyntaxError, ValueError):
            self._parser_healthy = False
            self._frontier("parse_error", path)
            return
        node_count = sum(1 for _ in ast.walk(tree))
        if node_count > self.limits.max_ast_nodes:
            self._bound("ast_nodes", path)
            return
        if indexed.ast_record.parse_error:
            self._parser_healthy = False
            self._frontier("indexed_parse_error", path, target=indexed.record_id)

        parsed = _ParsedFile(indexed=indexed, tree=tree, roles={role}, depth=depth)
        self._parsed[path] = parsed
        self._refresh_code_node(parsed)
        self._collect_fixtures(parsed)
        self._inspect_decorators(parsed)
        self._inspect_plugin_registration(parsed)
        self._inspect_imports(parsed)
        self._inspect_calls(parsed)

    def _refresh_code_node(self, parsed: _ParsedFile) -> None:
        record = parsed.indexed.ast_record
        symbols = []
        for symbol in sorted(record.qualified_symbols):
            if len(symbols) >= self.limits.max_symbols_per_file:
                self._bound("symbols", parsed.indexed.path)
                break
            if len(symbol) > self.limits.max_text_chars:
                self._bound("text", parsed.indexed.path, target="symbol")
                continue
            start, end = record.symbol_lines.get(symbol, (0, 0))
            symbols.append(
                {
                    "name": symbol,
                    "line_start": start,
                    "line_end": end,
                    "symbol_hash": record.symbol_hashes.get(symbol, ""),
                }
            )
        node = {
            "kind": "source",
            "path": parsed.indexed.path,
            "module": parsed.indexed.module,
            "roles": sorted(parsed.roles),
            "blob_identity": parsed.indexed.blob_identity,
            "source_sha256": parsed.indexed.source_sha256,
            "record_id": parsed.indexed.record_id,
            "symbols": symbols,
        }
        # Replace the older role projection for this path.
        for key, value in tuple(self._nodes.items()):
            if value.get("kind") == "source" and value.get("path") == parsed.indexed.path:
                del self._nodes[key]
        self._put_node(node)

    def _put_node(self, node: dict[str, Any]) -> None:
        encoded = canonical_json_bytes(node)
        self._nodes[encoded] = node

    def _record_data_node(self, absolute: Path, relative: str, *, role: str) -> bool:
        relative = _repo_path(relative)
        if len(relative) > self.limits.max_text_chars:
            self._bound("text", self._root_path, target="data_path")
            return False
        try:
            resolved = absolute.resolve(strict=True)
            resolved.relative_to(self.repository_root)
            if absolute.is_symlink() or not resolved.is_file():
                raise OSError
            size = resolved.stat().st_size
            if size > self.limits.max_data_bytes:
                self._bound("data_bytes", self._root_path, target=relative)
                return False
            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        except (OSError, ValueError):
            self._frontier("missing_file", self._root_path, target=relative)
            return False
        self._put_node(
            {
                "kind": "data",
                "path": relative,
                "roles": [role],
                "content_sha256": f"sha256:{digest}",
                "size_bytes": size,
            }
        )
        return True

    def _inspect_imports(self, parsed: _ParsedFile) -> None:
        for node in ast.walk(parsed.tree):
            imports: list[tuple[str, int, str]] = []
            if isinstance(node, ast.Import):
                imports = [(alias.name, 0, alias.name) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    if alias.name == "*":
                        self._frontier_node(
                            "reflection", parsed.indexed.path, node, target="star_import"
                        )
                        continue
                    candidate = f"{module}.{alias.name}" if module else alias.name
                    imports.append((candidate, int(node.level or 0), module))
            for requested, level, fallback in imports:
                resolved_name = self._absolute_import_name(parsed.indexed, requested, level)
                target = self._resolve_internal_module(resolved_name)
                if target is None and fallback and fallback != requested:
                    base = self._absolute_import_name(parsed.indexed, fallback, level)
                    target = self._resolve_internal_module(base)
                    if target is not None:
                        resolved_name = base
                start, end = _span(node)
                if target is not None:
                    self._edge("import", parsed.indexed.path, "", target, resolved_name, start, end)
                    self._analyze_file(target, role="import", depth=parsed.depth + 1)
                else:
                    kind = (
                        "native_code" if self._is_native_module(resolved_name) else "missing_file"
                    )
                    self._frontier_node(kind, parsed.indexed.path, node, target=resolved_name)

    @staticmethod
    def _absolute_import_name(indexed: IndexedASTPath, requested: str, level: int) -> str:
        if not level:
            return requested.strip(".")
        module_parts = indexed.module.split(".") if indexed.module else []
        if indexed.path.endswith("/__init__.py") or indexed.path == "__init__.py":
            package = module_parts
        else:
            package = module_parts[:-1]
        remove = max(0, level - 1)
        if remove > len(package):
            return requested.strip(".")
        prefix = package[: len(package) - remove] if remove else package
        suffix = requested.strip(".").split(".") if requested.strip(".") else []
        return ".".join([*prefix, *suffix])

    def _resolve_internal_module(self, module: str) -> str | None:
        paths = self._modules.get(module, ())
        if len(paths) == 1:
            return paths[0]
        if len(paths) > 1:
            self._frontier("ambiguous_import", self._root_path, target=module)
        return None

    def _is_native_module(self, module: str) -> bool:
        top = module.split(".", 1)[0]
        candidates = [
            self.repository_root / Path(*module.split(".")),
            self.repository_root / Path(*top.split(".")),
        ]
        suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
        for base in candidates:
            if any(Path(str(base) + suffix).exists() for suffix in suffixes):
                return True
        try:
            spec = importlib.util.find_spec(top)
        except (ImportError, AttributeError, ValueError):
            return False
        if spec is None:
            return False
        if spec.origin in {"built-in", "frozen"}:
            return spec.origin == "built-in"
        return bool(spec.origin and str(spec.origin).endswith(suffixes))

    def _collect_fixtures(self, parsed: _ParsedFile) -> None:
        for node in ast.walk(parsed.tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            fixture_name = ""
            for decorator in node.decorator_list:
                expression = decorator.func if isinstance(decorator, ast.Call) else decorator
                if _expression_name(expression) not in {"fixture", "pytest.fixture"}:
                    continue
                fixture_name = node.name
                if isinstance(decorator, ast.Call):
                    for keyword in decorator.keywords:
                        if keyword.arg == "name":
                            values = _literal_strings(keyword.value)
                            if values and len(values) == 1:
                                fixture_name = values[0]
                            else:
                                self._frontier_node(
                                    "opaque_decorator",
                                    parsed.indexed.path,
                                    decorator,
                                    source_symbol=node.name,
                                    target="fixture:name",
                                )
                break
            if fixture_name:
                try:
                    fixture_name = _safe_public_text(fixture_name, "fixture name")
                except StaticTraceError:
                    self._frontier_node(
                        "opaque_decorator",
                        parsed.indexed.path,
                        node,
                        source_symbol=node.name,
                        target="fixture:name",
                    )
                    continue
                definition = _FixtureDefinition(fixture_name, parsed.indexed.path, node.name, node)
                self._fixtures.setdefault(fixture_name, []).append(definition)

    def _inspect_decorators(self, parsed: _ParsedFile) -> None:
        for node in ast.walk(parsed.tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                expression = decorator.func if isinstance(decorator, ast.Call) else decorator
                name = _expression_name(expression)
                if name in _SAFE_DECORATORS or name.startswith("pytest.mark."):
                    start, end = _span(decorator)
                    self._edge(
                        "decorator",
                        parsed.indexed.path,
                        node.name,
                        parsed.indexed.path,
                        name,
                        start,
                        end,
                    )
                    continue
                target = name if name and _SAFE_TEXT_RE.fullmatch(name) else "dynamic_decorator"
                self._frontier_node(
                    "opaque_decorator",
                    parsed.indexed.path,
                    decorator,
                    source_symbol=node.name,
                    target=target,
                )

    def _inspect_plugin_registration(self, parsed: _ParsedFile) -> None:
        for node in parsed.tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(
                isinstance(target, ast.Name) and target.id == "pytest_plugins" for target in targets
            ):
                continue
            value = node.value
            plugins = _literal_strings(value) if value is not None else None
            if plugins is None:
                self._frontier_node(
                    "reflection", parsed.indexed.path, node, target="pytest_plugins"
                )
                continue
            for plugin in plugins:
                try:
                    safe_plugin = _safe_public_text(plugin, "plugin module")
                except StaticTraceError:
                    self._frontier_node(
                        "reflection", parsed.indexed.path, node, target="pytest_plugins"
                    )
                    continue
                target = self._resolve_internal_module(safe_plugin)
                if target is None:
                    self._frontier_node(
                        "missing_file", parsed.indexed.path, node, target=safe_plugin
                    )
                    continue
                start, end = _span(node)
                self._edge("plugin", parsed.indexed.path, "", target, safe_plugin, start, end)
                self._analyze_file(target, role="plugin", depth=parsed.depth + 1)

    def _inspect_calls(self, parsed: _ParsedFile) -> None:
        aliases = self._import_aliases(parsed.tree)
        parents = {
            child: parent
            for parent in ast.walk(parsed.tree)
            for child in ast.iter_child_nodes(parent)
        }
        owners: list[tuple[ast.AST, str]] = []
        for node in ast.walk(parsed.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                owners.append((node, node.name))
        for node in ast.walk(parsed.tree):
            if not isinstance(node, ast.Call):
                continue
            raw_name = _expression_name(node.func)
            name = self._expand_alias(raw_name, aliases)
            owner = ""
            containing = [
                (candidate, symbol)
                for candidate, symbol in owners
                if _span(candidate)[0] <= _span(node)[0] <= _span(candidate)[1]
            ]
            if containing:
                owner = min(containing, key=lambda item: _span(item[0])[1] - _span(item[0])[0])[1]
            if name in _DYNAMIC_IMPORT_CALLS:
                target = "dynamic_import"
                if node.args:
                    values = _literal_strings(node.args[0])
                    if values and len(values) == 1 and _SAFE_TEXT_RE.fullmatch(values[0]):
                        target = values[0]
                self._frontier_node(
                    "dynamic_import",
                    parsed.indexed.path,
                    node,
                    source_symbol=owner,
                    target=target,
                )
            if name in _REFLECTION_CALLS or name.rsplit(".", 1)[-1] in _REFLECTION_CALLS:
                self._frontier_node(
                    "reflection",
                    parsed.indexed.path,
                    node,
                    source_symbol=owner,
                    target=name.rsplit(".", 1)[-1],
                )
            effect = next((kind for kind, names in _EFFECT_CALLS.items() if name in names), None)
            if effect is not None:
                start, end = _span(node)
                self._edge("effect", parsed.indexed.path, owner, "", effect, start, end)
                self._frontier_node(
                    "uncontrolled_effect",
                    parsed.indexed.path,
                    node,
                    source_symbol=owner,
                    target=effect,
                )
            if name in {"open", "builtins.open"} and not (
                isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call)
            ):
                self._inspect_path_call(parsed, node, owner, name)
            if name in {"Path", "pathlib.Path"}:
                self._inspect_path_constructor(parsed, node, owner, parents, aliases)

    @staticmethod
    def _import_aliases(tree: ast.Module) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name.split(".", 1)[0]
                    aliases[local] = alias.name if alias.asname else local
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    if alias.name != "*":
                        aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        return aliases

    @staticmethod
    def _expand_alias(name: str, aliases: Mapping[str, str]) -> str:
        head, separator, tail = name.partition(".")
        expanded = aliases.get(head, head)
        return f"{expanded}.{tail}" if separator else expanded

    def _inspect_path_constructor(
        self,
        parsed: _ParsedFile,
        node: ast.Call,
        owner: str,
        parents: Mapping[ast.AST, ast.AST],
        aliases: Mapping[str, str],
    ) -> None:
        attribute = parents.get(node)
        invocation = parents.get(attribute) if attribute is not None else None
        if not (
            isinstance(attribute, ast.Attribute)
            and attribute.value is node
            and isinstance(invocation, ast.Call)
            and invocation.func is attribute
        ):
            self._frontier_node(
                "reflection",
                parsed.indexed.path,
                node,
                source_symbol=owner,
                target="dynamic_path",
            )
            return
        method = attribute.attr
        if method in {
            "read_bytes",
            "read_text",
            "stat",
            "lstat",
            "exists",
            "is_dir",
            "is_file",
            "is_symlink",
        }:
            self._inspect_path_call(parsed, node, owner, "pathlib.Path")
            return
        if method == "open":
            self._inspect_path_call(parsed, node, owner, "open", mode_call=invocation)
            return
        if method in {
            "chmod",
            "mkdir",
            "rename",
            "replace",
            "rmdir",
            "symlink_to",
            "touch",
            "unlink",
            "write_bytes",
            "write_text",
        }:
            start, end = _span(invocation)
            self._edge(
                "effect",
                parsed.indexed.path,
                owner,
                "",
                "filesystem_write",
                start,
                end,
            )
            self._frontier_node(
                "uncontrolled_effect",
                parsed.indexed.path,
                invocation,
                source_symbol=owner,
                target="filesystem_write",
            )
            return
        # Unknown Path methods can perform I/O through subclasses or future
        # APIs and therefore cannot close the dependency statically.
        expanded = self._expand_alias(method, aliases)
        self._frontier_node(
            "reflection",
            parsed.indexed.path,
            invocation,
            source_symbol=owner,
            target=expanded if _SAFE_TEXT_RE.fullmatch(expanded) else "dynamic_path",
        )

    def _inspect_path_call(
        self,
        parsed: _ParsedFile,
        node: ast.Call,
        owner: str,
        name: str,
        *,
        mode_call: ast.Call | None = None,
    ) -> None:
        if not node.args:
            self._frontier_node(
                "reflection",
                parsed.indexed.path,
                node,
                source_symbol=owner,
                target="dynamic_path",
            )
            return
        values = _literal_strings(node.args[0])
        if not values or len(values) != 1:
            self._frontier_node(
                "reflection",
                parsed.indexed.path,
                node,
                source_symbol=owner,
                target="dynamic_path",
            )
            return
        raw = values[0]
        mode = "r"
        mode_source = mode_call or node
        mode_arg_index = 0 if mode_call is not None else 1
        if name in {"open", "builtins.open"} and len(mode_source.args) > mode_arg_index:
            modes = _literal_strings(mode_source.args[mode_arg_index])
            mode = modes[0] if modes and len(modes) == 1 else "dynamic"
        for keyword in mode_source.keywords:
            if keyword.arg == "mode":
                modes = _literal_strings(keyword.value)
                mode = modes[0] if modes and len(modes) == 1 else "dynamic"
        if mode == "dynamic" or any(flag in mode for flag in "wax+"):
            start, end = _span(mode_source)
            self._edge(
                "effect",
                parsed.indexed.path,
                owner,
                "",
                "filesystem_write",
                start,
                end,
            )
            self._frontier_node(
                "uncontrolled_effect",
                parsed.indexed.path,
                mode_source,
                source_symbol=owner,
                target="filesystem_write",
            )
            return
        if Path(raw).is_absolute() or ".." in PurePosixPath(raw.replace("\\", "/")).parts:
            self._frontier_node(
                "missing_file",
                parsed.indexed.path,
                node,
                source_symbol=owner,
                target="outside_repository",
            )
            return
        relative = (PurePosixPath(parsed.indexed.path).parent / raw).as_posix()
        candidate = self.repository_root / Path(relative)
        if not candidate.exists():
            root_relative = _repo_path(raw)
            root_candidate = self.repository_root / Path(root_relative)
            if root_candidate.exists():
                relative, candidate = root_relative, root_candidate
        start, end = _span(node)
        if relative in self._records:
            self._edge("data", parsed.indexed.path, owner, relative, "", start, end)
            self._analyze_file(relative, role="data", depth=parsed.depth + 1)
        elif self._record_data_node(candidate, relative, role="data"):
            self._edge("data", parsed.indexed.path, owner, relative, "", start, end)

    def _resolve_root_fixtures(self) -> None:
        parsed = self._parsed.get(self._root_path)
        if parsed is None:
            return
        functions = [
            node
            for node in ast.walk(parsed.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        selected: list[ast.FunctionDef | ast.AsyncFunctionDef]
        if self._root_symbol:
            leaf = self._root_symbol.split("[", 1)[0]
            selected = [node for node in functions if node.name == leaf]
            if not selected:
                self._frontier("missing_test_symbol", self._root_path, target=leaf)
                return
        else:
            selected = [node for node in functions if node.name.startswith("test")]
        fixture_names: set[str] = set()
        for node in selected:
            fixture_names.update(argument.arg for argument in node.args.args)
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                if _expression_name(decorator.func) == "pytest.mark.usefixtures":
                    for argument in decorator.args:
                        values = _literal_strings(argument)
                        if values is None:
                            self._frontier_node(
                                "opaque_decorator",
                                self._root_path,
                                decorator,
                                source_symbol=node.name,
                                target="usefixtures",
                            )
                        else:
                            fixture_names.update(values)
        seen: set[str] = set()
        for fixture in sorted(fixture_names):
            self._resolve_fixture(
                fixture, source_path=self._root_path, source_symbol=self._root_symbol, seen=seen
            )

    def _resolve_fixture(
        self, name: str, *, source_path: str, source_symbol: str, seen: set[str]
    ) -> None:
        if name in seen:
            return
        seen.add(name)
        definitions = self._fixtures.get(name, ())
        if not definitions:
            self._frontier("unresolved_fixture", source_path, source_symbol, name)
            return
        # Local definition wins; otherwise nearest ancestor conftest wins.
        local = [item for item in definitions if item.path == self._root_path]
        candidates = local or sorted(
            definitions,
            key=lambda item: (-len(PurePosixPath(item.path).parts), item.path, item.symbol),
        )
        selected = candidates[0]
        if len(candidates) > 1 and len(PurePosixPath(candidates[0].path).parts) == len(
            PurePosixPath(candidates[1].path).parts
        ):
            self._frontier("ambiguous_fixture", source_path, source_symbol, name)
            return
        start, end = _span(selected.node)
        self._edge(
            "fixture",
            source_path,
            source_symbol,
            selected.path,
            selected.symbol,
            start,
            end,
        )
        parsed = self._parsed.get(selected.path)
        if parsed is not None:
            parsed.roles.add("fixture")
            self._refresh_code_node(parsed)
        for argument in selected.node.args.args:
            self._resolve_fixture(
                argument.arg,
                source_path=selected.path,
                source_symbol=selected.symbol,
                seen=seen,
            )

    def _record_conftest_hooks(self, conftests: Sequence[str]) -> None:
        for path in conftests:
            parsed = self._parsed.get(path)
            if parsed is None:
                continue
            for node in parsed.tree.body:
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and node.name.startswith("pytest_"):
                    start, end = _span(node)
                    self._edge("hook", self._root_path, "", path, node.name, start, end)

    def _edge(
        self,
        kind: str,
        source_path: str,
        source_symbol: str,
        target_path: str,
        target_symbol: str,
        line_start: int,
        line_end: int,
    ) -> None:
        if any(
            len(value) > self.limits.max_text_chars
            for value in (kind, source_path, source_symbol, target_path, target_symbol)
        ):
            self._bound("text", source_path, target="edge")
            return
        try:
            edge = {
                "kind": _safe_public_text(kind, "edge kind"),
                "source_path": _repo_path(source_path, allow_empty=True),
                "source_symbol": _safe_public_text(
                    source_symbol, "edge source symbol", allow_empty=True
                ),
                "target_path": _repo_path(target_path, allow_empty=True),
                "target_symbol": _safe_public_text(
                    target_symbol, "edge target symbol", allow_empty=True
                ),
                "line_start": _line_number(line_start),
                "line_end": _line_number(line_end),
            }
        except StaticTraceError:
            self._bound("text", source_path, target="edge")
            return
        encoded = canonical_json_bytes(edge)
        if encoded in self._edges:
            return
        if len(self._edges) >= self.limits.max_edges:
            self._bound("edges", source_path)
            return
        self._edges[encoded] = edge

    def _frontier_node(
        self,
        kind: str,
        source_path: str,
        node: ast.AST,
        *,
        source_symbol: str = "",
        target: str = "",
    ) -> None:
        start, end = _span(node)
        self._frontier(kind, source_path, source_symbol, target, start, end)

    def _frontier(
        self,
        kind: str,
        source_path: str,
        source_symbol: str = "",
        target: str = "",
        line_start: int = 0,
        line_end: int = 0,
    ) -> None:
        if any(
            len(str(value or "")) > self.limits.max_text_chars
            for value in (kind, source_path, source_symbol, target)
        ):
            self._bound_kinds.add("text")
            kind = "analysis_bound"
            source_path = source_path if len(source_path) <= self.limits.max_text_chars else ""
            source_symbol = ""
            target = "text"
            line_start = 0
            line_end = 0
        try:
            item = UnknownDependencyFrontier(
                kind=kind,
                source_path=source_path,
                source_symbol=source_symbol,
                target=target,
                line_start=line_start,
                line_end=line_end,
            )
        except StaticTraceError:
            self._bound_kinds.add("text")
            item = UnknownDependencyFrontier("analysis_bound", source_path, target="text")
        if item.frontier_id in self._frontiers:
            return
        if len(self._frontiers) >= self.limits.max_frontier:
            self._bound_kinds.add("frontier")
            marker = UnknownDependencyFrontier("analysis_bound", self._root_path, target="frontier")
            if marker.frontier_id not in self._frontiers:
                # The bound marker is authority-relevant and may not itself be
                # discarded by the bound.  Evict a deterministic detail row.
                evicted = max(self._frontiers)
                del self._frontiers[evicted]
                self._frontiers[marker.frontier_id] = marker
            return
        self._frontiers[item.frontier_id] = item

    def _bound(self, kind: str, source_path: str, *, target: str = "") -> None:
        self._bound_kinds.add(kind)
        self._frontier("analysis_bound", source_path or self._root_path, target=target or kind)


def trace_static_dependencies(
    analysis_index: AnalysisASTIndex,
    test_path: str,
    *,
    repository_root: str | os.PathLike[str],
    test_symbol: str = "",
    node_id: str = "",
    limits: StaticTraceLimits | None = None,
    identity_minter: Callable[[Any], ContentIdentity] = mint_content_identity,
) -> StaticTestDependencyTrace:
    """Convenience wrapper for a single static dependency trace."""

    return StaticTestDependencyTracer(
        analysis_index,
        repository_root,
        limits=limits,
        identity_minter=identity_minter,
    ).trace(test_path, test_symbol, node_id=node_id)


__all__ = [
    "STATIC_TEST_DEPENDENCY_TRACE_INTERFACE",
    "STATIC_TEST_DEPENDENCY_TRACE_SCHEMA",
    "STATIC_TEST_DEPENDENCY_TRACER_INTERFACE",
    "STATIC_TRACE_ANALYZER_INTERFACE",
    "STATIC_TRACE_LIMITS_INTERFACE",
    "StaticTestDependencyTrace",
    "StaticTestDependencyTracer",
    "StaticTraceError",
    "StaticTraceLimits",
    "UnknownDependencyFrontier",
    "trace_static_dependencies",
]
