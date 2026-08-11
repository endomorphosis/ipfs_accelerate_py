"""Snapshot-bound typed program dependency graph builder.

Adapts :class:`AnalysisASTIndex`, :class:`CodeImpactIndex`, and
:class:`SemanticDependencyGraph` into a concrete :class:`ProgramGraph` with
typed call, data, state, schema, wiring, ownership, and validation edges.
Incremental rebuild of changed paths is required to equal a clean rebuild of
the same fixture snapshot.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from ..core.conflict_graph import ASTBlobRecord, build_python_ast_blob_record
from .program_graph import (
    Completeness,
    PROGRAM_GRAPH_VERSION,
    ProgramAuthority,
    ProgramEdge,
    ProgramEdgeKind,
    ProgramGraph,
    ProgramGraphError,
    ProgramGraphRoots,
    ProgramGraphSnapshot,
    ProgramNode,
    ProgramNodeKind,
    ProgramProvenance,
    ProgramTrust,
)

PROGRAM_DEPENDENCY_GRAPH_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-dependency-graph@1"
)
PROGRAM_DEPENDENCY_GRAPH_VERSION = "program-dependency-graph@1"
PATH_COMPONENT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-path-component@1"
)

DEFAULT_MAX_SOURCE_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_PATHS = 50_000

_DYNAMIC_CALLEES = frozenset(
    {
        "<dynamic>",
        "getattr",
        "setattr",
        "globals",
        "locals",
        "eval",
        "exec",
        "__import__",
        "importlib.import_module",
    }
)
_FACTORY_NAMES = frozenset(
    {
        "create",
        "make",
        "build",
        "factory",
        "from_dict",
        "from_json",
        "from_config",
        "get_instance",
        "instance",
        "builder",
    }
)
_BUILDER_NAMES = frozenset({"builder", "build", "with_", "set_"})
_REGISTRY_MARKERS = frozenset(
    {"register", "registry", "provide", "bind", "inject", "autoload"}
)
_SERIALIZER_MARKERS = frozenset(
    {
        "serialize",
        "deserialize",
        "to_dict",
        "from_dict",
        "to_json",
        "from_json",
        "dump",
        "load",
        "encode",
        "decode",
    }
)
_TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|testing|test_fixtures?|fixtures?|mocks?|conftest)(/|$)",
    re.IGNORECASE,
)
_GENERATED_PATH_RE = re.compile(
    r"(^|/)(generated|gen|_generated|build|dist|\.tox|node_modules)(/|$)",
    re.IGNORECASE,
)
_NATIVE_PATH_RE = re.compile(
    r"\.(so|dylib|dll|a|o|c|cc|cpp|h|hpp|rs|go)$",
    re.IGNORECASE,
)
_SCHEMA_PATH_RE = re.compile(
    r"(schema|openapi|swagger|protobuf|proto|avro|jsonschema)",
    re.IGNORECASE,
)
_CONFIG_PATH_RE = re.compile(
    r"(config|settings|feature.?flag)",
    re.IGNORECASE,
)
_CLI_MARKERS = frozenset({"cli", "main", "argparse", "click", "typer", "fire"})
_API_MARKERS = frozenset(
    {"route", "api", "endpoint", "get", "post", "put", "delete", "patch", "rpc"}
)


class ProgramDependencyGraphError(ProgramGraphError):
    """Raised when dependency-graph construction is unsafe or incomplete."""


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


def _repo_path(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    if not raw:
        raise ProgramDependencyGraphError("path is required")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ProgramDependencyGraphError(f"repository path escapes its root: {value!r}")
    return path.as_posix()


def _module_name(path: str) -> str:
    value = path[:-3] if path.endswith(".py") else path
    if value.endswith(".pyi"):
        value = value[:-4]
    parts = list(PurePosixPath(value).parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _node_id(*parts: str) -> str:
    cleaned = [str(part).strip() for part in parts if str(part).strip()]
    return "node:" + _identity("program-node-id", cleaned)


def _simple(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    return text.rsplit(".", 1)[-1]


def _is_test_path(path: str) -> bool:
    return bool(_TEST_PATH_RE.search(path)) or PurePosixPath(path).name.startswith(
        "test_"
    )


def _is_generated_path(path: str) -> bool:
    return bool(_GENERATED_PATH_RE.search(path))


def _is_native_path(path: str) -> bool:
    return bool(_NATIVE_PATH_RE.search(path))


def _span_dict(
    line_start: int = 0,
    line_end: int = 0,
    column_start: int = 0,
    column_end: int = 0,
) -> dict[str, int]:
    return {
        "line_start": max(0, int(line_start)),
        "line_end": max(0, int(line_end)),
        "column_start": max(0, int(column_start)),
        "column_end": max(0, int(column_end)),
    }


@dataclass(frozen=True)
class PathSource:
    """One path-bound source document or pre-parsed AST blob."""

    path: str
    source: str = ""
    language: str = "python"
    blob_identity: str = ""
    source_sha256: str = ""
    ast_record: ASTBlobRecord | None = None
    generated: bool = False
    excluded: bool = False
    native: bool = False
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _repo_path(self.path))
        object.__setattr__(self, "source", str(self.source or ""))
        object.__setattr__(
            self, "language", str(self.language or "python").strip() or "python"
        )
        object.__setattr__(self, "blob_identity", str(self.blob_identity or "").strip())
        object.__setattr__(
            self, "source_sha256", str(self.source_sha256 or "").strip()
        )
        object.__setattr__(self, "generated", bool(self.generated))
        object.__setattr__(self, "excluded", bool(self.excluded))
        object.__setattr__(self, "native", bool(self.native))
        attrs = self.attributes or {}
        if not isinstance(attrs, Mapping):
            raise ProgramDependencyGraphError("attributes must be a mapping")
        object.__setattr__(
            self,
            "attributes",
            MappingProxyType({str(key): attrs[key] for key in sorted(attrs)}),
        )
        if self.ast_record is not None and not isinstance(
            self.ast_record, ASTBlobRecord
        ):
            raise ProgramDependencyGraphError("ast_record must be ASTBlobRecord")
        if len(self.source.encode("utf-8")) > DEFAULT_MAX_SOURCE_BYTES:
            raise ProgramDependencyGraphError(
                f"source for {self.path!r} exceeds the hard byte bound"
            )

    @property
    def content_key(self) -> str:
        if self.ast_record is not None:
            return self.ast_record.record_id
        if self.source_sha256:
            return self.source_sha256
        if self.source:
            digest = "sha256:" + hashlib.sha256(
                self.source.encode("utf-8", errors="surrogatepass")
            ).hexdigest()
            return digest
        return self.blob_identity or f"empty:{self.path}"


@dataclass(frozen=True)
class PathComponent:
    """Reusable per-path graph component for incremental rebuild."""

    path: str
    content_key: str
    nodes: tuple[ProgramNode, ...]
    edges: tuple[tuple[str, str, str, Mapping[str, Any]], ...]
    frontier_refs: tuple[str, ...] = ()
    exclusion_refs: tuple[str, ...] = ()
    schema: str = PATH_COMPONENT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _repo_path(self.path))
        object.__setattr__(self, "content_key", str(self.content_key or "").strip())
        if not self.content_key:
            raise ProgramDependencyGraphError("path component requires content_key")
        object.__setattr__(
            self,
            "nodes",
            tuple(sorted(self.nodes, key=lambda item: item.node_id)),
        )
        normalized_edges: list[tuple[str, str, str, Mapping[str, Any]]] = []
        for item in self.edges:
            if len(item) == 3:
                source, target, kind = item
                attributes: Mapping[str, Any] = {}
            else:
                source, target, kind, attributes = item  # type: ignore[misc]
            normalized_edges.append(
                (
                    str(source),
                    str(target),
                    str(kind),
                    MappingProxyType(dict(attributes or {})),
                )
            )
        object.__setattr__(
            self,
            "edges",
            tuple(
                sorted(
                    normalized_edges,
                    key=lambda item: (item[0], item[2], item[1], _canonical_json(dict(item[3]))),
                )
            ),
        )
        object.__setattr__(
            self,
            "frontier_refs",
            tuple(sorted({str(item) for item in self.frontier_refs if str(item)})),
        )
        object.__setattr__(
            self,
            "exclusion_refs",
            tuple(sorted({str(item) for item in self.exclusion_refs if str(item)})),
        )

    @property
    def component_id(self) -> str:
        return _identity(
            "program-path-component",
            {
                "schema": self.schema,
                "path": self.path,
                "content_key": self.content_key,
                "node_ids": [node.node_id for node in self.nodes],
                "node_content_ids": [node.content_id for node in self.nodes],
                "edges": [
                    {
                        "source": source,
                        "target": target,
                        "kind": kind,
                        "attributes": dict(attributes),
                    }
                    for source, target, kind, attributes in self.edges
                ],
                "frontier_refs": list(self.frontier_refs),
                "exclusion_refs": list(self.exclusion_refs),
            },
        )


@dataclass
class _BuildState:
    roots: ProgramGraphRoots
    nodes: dict[str, ProgramNode]
    edge_keys: set[str]
    edges: list[ProgramEdge]
    frontier: set[str]
    exclusions: set[str]
    symbol_index: dict[str, list[str]]
    path_modules: dict[str, str]

    def add_node(self, node: ProgramNode) -> ProgramNode:
        existing = self.nodes.get(node.node_id)
        if existing is not None:
            return existing
        self.nodes[node.node_id] = node
        for key in {node.name, node.qualified_name, _simple(node.qualified_name)}:
            if key:
                self.symbol_index.setdefault(key, []).append(node.node_id)
        return node

    def add_edge(
        self,
        source: str,
        target: str,
        kind: ProgramEdgeKind,
        *,
        provenance: ProgramProvenance = ProgramProvenance.AST,
        trust: ProgramTrust = ProgramTrust.TRUSTED,
        authority: ProgramAuthority = ProgramAuthority.AUTHORITATIVE,
        completeness: Completeness = Completeness.COMPLETE,
        confidence: int = 100,
        attributes: Mapping[str, Any] | None = None,
    ) -> ProgramEdge | None:
        if source not in self.nodes or target not in self.nodes:
            return None
        if source == target and kind not in {
            ProgramEdgeKind.ALIASES,
            ProgramEdgeKind.OVERLOADS,
        }:
            return None
        edge = ProgramEdge(
            source=source,
            target=target,
            kind=kind,
            roots=self.roots,
            provenance=provenance,
            trust=trust,
            authority=authority,
            completeness=completeness,
            confidence=confidence,
            attributes=attributes or {},
        )
        if edge.edge_id in self.edge_keys:
            return None
        self.edge_keys.add(edge.edge_id)
        self.edges.append(edge)
        return edge


class ProgramDependencyGraph:
    """Builder and query façade for the snapshot-bound dependency graph.

    Construction is deterministic: the same roots and path sources always
    produce the same :class:`ProgramGraphSnapshot` identity, whether built
    clean or via incremental path-component reuse.
    """

    def __init__(
        self,
        roots: ProgramGraphRoots | Mapping[str, Any],
        *,
        previous: "ProgramDependencyGraph | None" = None,
    ) -> None:
        if isinstance(roots, Mapping):
            roots = ProgramGraphRoots.from_dict(roots)
        if not isinstance(roots, ProgramGraphRoots):
            raise ProgramDependencyGraphError("roots must be ProgramGraphRoots")
        self._roots = roots
        self._components: dict[str, PathComponent] = {}
        self._snapshot: ProgramGraphSnapshot | None = None
        self._graph: ProgramGraph | None = None
        if previous is not None:
            if previous.roots.roots_id != roots.roots_id:
                # Different roots: discard previous components.
                pass
            else:
                self._components = dict(previous._components)

    @property
    def roots(self) -> ProgramGraphRoots:
        return self._roots

    @property
    def snapshot(self) -> ProgramGraphSnapshot | None:
        return self._snapshot

    @property
    def graph(self) -> ProgramGraph | None:
        return self._graph

    @property
    def components(self) -> Mapping[str, PathComponent]:
        return MappingProxyType(self._components)

    def build(
        self,
        sources: Iterable[PathSource | Mapping[str, Any]] = (),
        *,
        nominations: Iterable[Mapping[str, Any]] = (),
        impact_edges: Mapping[str, Sequence[str]] | None = None,
        previous: "ProgramDependencyGraph | Mapping[str, PathComponent] | None" = None,
    ) -> ProgramGraph:
        """Build a complete graph, reusing unchanged path components when possible."""

        path_sources = [self._coerce_source(item) for item in sources]
        if len(path_sources) > DEFAULT_MAX_PATHS:
            raise ProgramDependencyGraphError("path count exceeds hard bound")
        if len({item.path for item in path_sources}) != len(path_sources):
            raise ProgramDependencyGraphError("source paths must be unique")

        prior_components: dict[str, PathComponent] = {}
        if previous is None:
            prior_components = dict(self._components)
        elif isinstance(previous, ProgramDependencyGraph):
            if previous.roots.roots_id == self._roots.roots_id:
                prior_components = dict(previous._components)
        elif isinstance(previous, Mapping):
            prior_components = {
                str(key): (
                    value
                    if isinstance(value, PathComponent)
                    else self._component_from_dict(value)
                )
                for key, value in previous.items()
            }

        components: dict[str, PathComponent] = {}
        for source in sorted(path_sources, key=lambda item: item.path):
            prior = prior_components.get(source.path)
            if (
                prior is not None
                and prior.content_key == source.content_key
                and prior.path == source.path
            ):
                # Rebind nodes to current roots while preserving structure.
                components[source.path] = self._rebind_component(prior)
            else:
                components[source.path] = self._extract_component(source)

        self._components = components
        snapshot = self._assemble(
            components,
            nominations=nominations,
            impact_edges=impact_edges or {},
        )
        self._snapshot = snapshot
        self._graph = ProgramGraph(snapshot)
        return self._graph

    def rebuild_incremental(
        self,
        sources: Iterable[PathSource | Mapping[str, Any]],
        *,
        nominations: Iterable[Mapping[str, Any]] = (),
        impact_edges: Mapping[str, Sequence[str]] | None = None,
    ) -> ProgramGraph:
        """Incremental rebuild using the current component cache as previous."""

        return self.build(
            sources,
            nominations=nominations,
            impact_edges=impact_edges,
            previous=self,
        )

    def to_dict(self) -> dict[str, Any]:
        if self._snapshot is None:
            raise ProgramDependencyGraphError("graph has not been built")
        return {
            "schema": PROGRAM_DEPENDENCY_GRAPH_SCHEMA,
            "version": PROGRAM_DEPENDENCY_GRAPH_VERSION,
            "roots": self._roots.to_dict(),
            "snapshot": self._snapshot.to_dict(),
            "component_ids": {
                path: component.component_id
                for path, component in sorted(self._components.items())
            },
        }

    @classmethod
    def from_sources(
        cls,
        roots: ProgramGraphRoots | Mapping[str, Any],
        sources: Iterable[PathSource | Mapping[str, Any]],
        **kwargs: Any,
    ) -> "ProgramDependencyGraph":
        graph = cls(roots)
        graph.build(sources, **kwargs)
        return graph

    @classmethod
    def from_python_sources(
        cls,
        roots: ProgramGraphRoots | Mapping[str, Any],
        files: Mapping[str, str],
        **kwargs: Any,
    ) -> "ProgramDependencyGraph":
        sources = [
            PathSource(path=path, source=source, language="python")
            for path, source in sorted(files.items())
        ]
        return cls.from_sources(roots, sources, **kwargs)

    @classmethod
    def from_ast_index(
        cls,
        roots: ProgramGraphRoots | Mapping[str, Any],
        index: Any,
        **kwargs: Any,
    ) -> "ProgramDependencyGraph":
        """Build from an AnalysisASTIndex-like object."""

        path_records = getattr(index, "path_records", None)
        if path_records is None:
            path_records = getattr(index, "records", ())
        sources: list[PathSource] = []
        for item in path_records:
            path = getattr(item, "path", None) or (
                item.get("path") if isinstance(item, Mapping) else None
            )
            record = getattr(item, "ast_record", None) or (
                item.get("ast_record") if isinstance(item, Mapping) else None
            )
            if path is None or record is None:
                continue
            if not isinstance(record, ASTBlobRecord):
                record = ASTBlobRecord.from_dict(record)
            sources.append(
                PathSource(
                    path=str(path),
                    language=getattr(record, "language", "python") or "python",
                    blob_identity=record.blob_identity,
                    source_sha256=record.source_sha256,
                    ast_record=record,
                )
            )
        return cls.from_sources(roots, sources, **kwargs)

    def _coerce_source(self, value: PathSource | Mapping[str, Any]) -> PathSource:
        if isinstance(value, PathSource):
            return value
        if not isinstance(value, Mapping):
            raise ProgramDependencyGraphError("source must be PathSource or mapping")
        record = value.get("ast_record")
        if isinstance(record, Mapping):
            record = ASTBlobRecord.from_dict(record)
        return PathSource(
            path=str(value.get("path") or ""),
            source=str(value.get("source") or ""),
            language=str(value.get("language") or "python"),
            blob_identity=str(value.get("blob_identity") or ""),
            source_sha256=str(value.get("source_sha256") or ""),
            ast_record=record if isinstance(record, ASTBlobRecord) else None,
            generated=bool(value.get("generated", False)),
            excluded=bool(value.get("excluded", False)),
            native=bool(value.get("native", False)),
            attributes=value.get("attributes") or {},
        )

    def _component_from_dict(self, value: Any) -> PathComponent:
        if isinstance(value, PathComponent):
            return value
        raise ProgramDependencyGraphError("invalid path component cache entry")

    def _rebind_component(self, component: PathComponent) -> PathComponent:
        """Re-emit a cached component under the current roots identity."""

        rebound_nodes = []
        for node in component.nodes:
            rebound_nodes.append(
                ProgramNode(
                    node_id=node.node_id,
                    kind=node.kind,
                    name=node.name,
                    roots=self._roots,
                    path=node.path,
                    qualified_name=node.qualified_name,
                    language=node.language,
                    blob_identity=node.blob_identity,
                    source_sha256=node.source_sha256,
                    span=dict(node.span),
                    provenance=node.provenance,
                    provenance_id=node.provenance_id,
                    trust=node.trust,
                    authority=node.authority,
                    completeness=node.completeness,
                    confidence=node.confidence,
                    extractor_id=node.extractor_id,
                    attributes=dict(node.attributes),
                )
            )
        return PathComponent(
            path=component.path,
            content_key=component.content_key,
            nodes=tuple(rebound_nodes),
            edges=component.edges,
            frontier_refs=component.frontier_refs,
            exclusion_refs=component.exclusion_refs,
        )

    def _extract_component(self, source: PathSource) -> PathComponent:
        if source.excluded:
            return PathComponent(
                path=source.path,
                content_key=source.content_key,
                nodes=(),
                edges=(),
                exclusion_refs=(f"excluded:{source.path}",),
            )
        if source.native or _is_native_path(source.path):
            node = ProgramNode(
                node_id=_node_id("native", source.path),
                kind=ProgramNodeKind.NATIVE_BOUNDARY,
                name=PurePosixPath(source.path).name,
                roots=self._roots,
                path=source.path,
                qualified_name=source.path,
                language=source.language or "native",
                blob_identity=source.blob_identity,
                source_sha256=source.source_sha256,
                provenance=ProgramProvenance.EXTRACTOR,
                completeness=Completeness.FRONTIER,
                attributes={"boundary": "native"},
            )
            return PathComponent(
                path=source.path,
                content_key=source.content_key,
                nodes=(node,),
                edges=(),
                frontier_refs=(f"native:{source.path}",),
            )
        if source.generated or _is_generated_path(source.path):
            node = ProgramNode(
                node_id=_node_id("generated", source.path),
                kind=ProgramNodeKind.GENERATED,
                name=PurePosixPath(source.path).name,
                roots=self._roots,
                path=source.path,
                qualified_name=source.path,
                language=source.language,
                blob_identity=source.blob_identity,
                source_sha256=source.source_sha256,
                provenance=ProgramProvenance.EXTRACTOR,
                completeness=Completeness.PARTIAL,
                attributes={"boundary": "generated"},
            )
            return PathComponent(
                path=source.path,
                content_key=source.content_key,
                nodes=(node,),
                edges=(),
                frontier_refs=(f"generated:{source.path}",),
            )

        record = source.ast_record
        if record is None and source.source and source.language.startswith("python"):
            record = build_python_ast_blob_record(
                source.source,
                blob_identity=source.blob_identity,
                source_sha256=source.source_sha256,
            )
        if record is None:
            return PathComponent(
                path=source.path,
                content_key=source.content_key,
                nodes=(),
                edges=(),
                frontier_refs=(f"unsupported:{source.path}",),
            )

        return self._extract_from_record(source, record)

    def _extract_from_record(
        self, source: PathSource, record: ASTBlobRecord
    ) -> PathComponent:
        roots = self._roots
        path = source.path
        module = _module_name(path)
        language = record.language or source.language or "python"
        blob = record.blob_identity
        sha = record.source_sha256
        nodes: dict[str, ProgramNode] = {}
        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        frontier: set[str] = set()
        exclusions: set[str] = set()

        def add_node(
            kind: ProgramNodeKind,
            name: str,
            *,
            qualified: str = "",
            span: Mapping[str, Any] | None = None,
            attributes: Mapping[str, Any] | None = None,
            completeness: Completeness = Completeness.COMPLETE,
        ) -> str:
            qn = qualified or (f"{module}.{name}" if module and name else name)
            node_id = _node_id(kind.value, path, qn or name)
            if node_id in nodes:
                return node_id
            nodes[node_id] = ProgramNode(
                node_id=node_id,
                kind=kind,
                name=_simple(name) or name,
                roots=roots,
                path=path,
                qualified_name=qn or name,
                language=language,
                blob_identity=blob,
                source_sha256=sha,
                span=dict(span or {}),
                provenance=ProgramProvenance.AST,
                trust=ProgramTrust.TRUSTED,
                authority=ProgramAuthority.AUTHORITATIVE,
                completeness=completeness,
                extractor_id=roots.extractor_id,
                attributes=dict(attributes or {}),
            )
            return node_id

        def add_edge(
            source_id: str,
            target_id: str,
            kind: ProgramEdgeKind,
            **attributes: Any,
        ) -> None:
            edges.append((source_id, target_id, kind.value, dict(attributes)))

        file_id = add_node(
            ProgramNodeKind.FILE,
            PurePosixPath(path).name,
            qualified=path,
            attributes={"role": "file"},
        )
        module_id = add_node(
            ProgramNodeKind.MODULE,
            module or PurePosixPath(path).stem,
            qualified=module or path,
            attributes={"role": "module"},
        )
        add_edge(file_id, module_id, ProgramEdgeKind.CONTAINS)

        if _is_test_path(path):
            test_id = add_node(
                ProgramNodeKind.TEST,
                PurePosixPath(path).name,
                qualified=f"test:{path}",
                attributes={"role": "test_module"},
            )
            add_edge(test_id, module_id, ProgramEdgeKind.TESTS)

        if _SCHEMA_PATH_RE.search(path):
            schema_id = add_node(
                ProgramNodeKind.SCHEMA,
                PurePosixPath(path).name,
                qualified=f"schema:{path}",
            )
            add_edge(module_id, schema_id, ProgramEdgeKind.SCHEMA_OF)

        if _CONFIG_PATH_RE.search(path):
            config_id = add_node(
                ProgramNodeKind.CONFIG_PROVIDER,
                PurePosixPath(path).name,
                qualified=f"config:{path}",
            )
            add_edge(module_id, config_id, ProgramEdgeKind.CONFIGURES)

        # Declarations from qualified symbols.
        symbol_ids: dict[str, str] = {}
        for symbol in record.qualified_symbols:
            lines = record.symbol_lines.get(symbol, (0, 0))
            span = _span_dict(lines[0], lines[1])
            kind = self._symbol_kind(symbol, record)
            symbol_id = add_node(
                kind,
                symbol,
                qualified=f"{module}.{symbol}" if module else symbol,
                span=span,
                attributes={
                    "symbol_hash": record.symbol_hashes.get(symbol, ""),
                    "local_symbol": symbol,
                },
            )
            symbol_ids[symbol] = symbol_id
            add_edge(module_id, symbol_id, ProgramEdgeKind.DEFINES)
            add_edge(module_id, symbol_id, ProgramEdgeKind.DECLARES)

            simple = _simple(symbol)
            if simple == "__init__":
                ctor_id = add_node(
                    ProgramNodeKind.CONSTRUCTOR,
                    symbol,
                    qualified=f"{module}.{symbol}" if module else symbol,
                    span=span,
                )
                add_edge(symbol_id, ctor_id, ProgramEdgeKind.CONSTRUCTS)
            if simple.lower() in _FACTORY_NAMES or simple.startswith("create_"):
                factory_id = add_node(
                    ProgramNodeKind.FACTORY,
                    symbol,
                    qualified=f"factory:{module}.{symbol}" if module else f"factory:{symbol}",
                    span=span,
                )
                add_edge(symbol_id, factory_id, ProgramEdgeKind.FACTORY_CREATES)
            if simple.lower() in {"builder"} or simple.startswith("build_"):
                builder_id = add_node(
                    ProgramNodeKind.BUILDER,
                    symbol,
                    qualified=f"builder:{module}.{symbol}" if module else f"builder:{symbol}",
                    span=span,
                )
                add_edge(symbol_id, builder_id, ProgramEdgeKind.BUILDER_BUILDS)
            if simple.lower() in _REGISTRY_MARKERS or "register" in simple.lower():
                reg_id = add_node(
                    ProgramNodeKind.REGISTRY,
                    symbol,
                    qualified=f"registry:{module}.{symbol}" if module else f"registry:{symbol}",
                    span=span,
                )
                add_edge(symbol_id, reg_id, ProgramEdgeKind.REGISTERS)
            if simple.lower() in _SERIALIZER_MARKERS:
                ser_kind = (
                    ProgramNodeKind.DESERIALIZER
                    if "de" in simple.lower() or simple.lower().startswith("from_")
                    else ProgramNodeKind.SERIALIZER
                )
                ser_id = add_node(
                    ser_kind,
                    symbol,
                    qualified=f"serde:{module}.{symbol}" if module else f"serde:{symbol}",
                    span=span,
                )
                edge_kind = (
                    ProgramEdgeKind.DESERIALIZES
                    if ser_kind is ProgramNodeKind.DESERIALIZER
                    else ProgramEdgeKind.SERIALIZES
                )
                add_edge(symbol_id, ser_id, edge_kind)
            simple_l = simple.lower()
            if (
                simple_l in _CLI_MARKERS
                or simple_l.startswith("cli_")
                or simple_l.endswith("_cli")
                or "cli_main" in simple_l
            ):
                cli_id = add_node(
                    ProgramNodeKind.CLI_COMMAND,
                    symbol,
                    qualified=f"cli:{module}.{symbol}" if module else f"cli:{symbol}",
                    span=span,
                )
                add_edge(symbol_id, cli_id, ProgramEdgeKind.SERVES)
            if simple_l in _API_MARKERS or any(
                marker in simple_l for marker in ("endpoint", "handler", "rpc")
            ):
                api_id = add_node(
                    ProgramNodeKind.API_ENDPOINT
                    if "rpc" not in simple.lower()
                    else ProgramNodeKind.RPC_METHOD,
                    symbol,
                    qualified=f"surface:{module}.{symbol}" if module else f"surface:{symbol}",
                    span=span,
                )
                add_edge(symbol_id, api_id, ProgramEdgeKind.SERVES)
            if _is_test_path(path) or simple.startswith("test_"):
                test_id = add_node(
                    ProgramNodeKind.TEST,
                    symbol,
                    qualified=f"test:{module}.{symbol}" if module else f"test:{symbol}",
                    span=span,
                )
                add_edge(test_id, symbol_id, ProgramEdgeKind.TESTS)
            if simple.startswith("mock_") or simple.endswith("Mock") or "fixture" in simple.lower():
                mock_kind = (
                    ProgramNodeKind.FIXTURE
                    if "fixture" in simple.lower()
                    else ProgramNodeKind.MOCK
                )
                mock_id = add_node(
                    mock_kind,
                    symbol,
                    qualified=f"mock:{module}.{symbol}" if module else f"mock:{symbol}",
                    span=span,
                )
                edge_kind = (
                    ProgramEdgeKind.FIXTURES
                    if mock_kind is ProgramNodeKind.FIXTURE
                    else ProgramEdgeKind.MOCKS
                )
                add_edge(mock_id, symbol_id, edge_kind)

            # Ownership / validation markers on public symbols.
            if not simple.startswith("_"):
                owner_id = add_node(
                    ProgramNodeKind.OWNERSHIP,
                    f"owner:{symbol}",
                    qualified=f"ownership:{module}.{symbol}" if module else f"ownership:{symbol}",
                    attributes={"owner_module": module},
                )
                add_edge(owner_id, symbol_id, ProgramEdgeKind.OWNS)
                val_id = add_node(
                    ProgramNodeKind.VALIDATION,
                    f"validate:{symbol}",
                    qualified=f"validation:{module}.{symbol}" if module else f"validation:{symbol}",
                )
                add_edge(val_id, symbol_id, ProgramEdgeKind.VALIDATES)

        # Interfaces / protocols / overrides.
        for interface in record.interfaces:
            if "(" in interface and interface.endswith(")"):
                # Class bases form: Name(Base1,Base2)
                name, _, bases = interface.partition("(")
                base_list = [item.strip() for item in bases[:-1].split(",") if item.strip()]
                class_symbol = name
                class_id = symbol_ids.get(class_symbol)
                if class_id is None:
                    class_id = add_node(
                        ProgramNodeKind.CLASS,
                        class_symbol,
                        qualified=f"{module}.{class_symbol}" if module else class_symbol,
                    )
                    symbol_ids[class_symbol] = class_id
                for base in base_list:
                    base_id = add_node(
                        ProgramNodeKind.INTERFACE
                        if base.endswith("Protocol") or base in {"Protocol", "ABC"}
                        else ProgramNodeKind.CLASS,
                        base,
                        qualified=base,
                        completeness=Completeness.PARTIAL,
                    )
                    add_edge(
                        class_id,
                        base_id,
                        ProgramEdgeKind.IMPLEMENTS
                        if "Protocol" in base or base in {"ABC", "abc.ABC"}
                        else ProgramEdgeKind.OVERRIDES,
                    )
            elif ":" in interface:
                # Method signature form: Name:def ...
                name, _, signature = interface.partition(":")
                method_id = symbol_ids.get(name)
                if method_id is None:
                    method_id = add_node(
                        ProgramNodeKind.METHOD if "." in name else ProgramNodeKind.FUNCTION,
                        name,
                        qualified=f"{module}.{name}" if module else name,
                    )
                    symbol_ids[name] = method_id
                # Parameters / returns from signature text.
                self._add_signature_flow(
                    add_node, add_edge, method_id, name, signature
                )

        # Imports / aliases / re-exports.
        for imported in record.imports:
            import_id = add_node(
                ProgramNodeKind.IMPORT,
                imported,
                qualified=f"import:{path}:{imported}",
                attributes={"statement": imported},
            )
            add_edge(module_id, import_id, ProgramEdgeKind.IMPORTS)
            alias_name = ""
            target_name = ""
            if " as " in imported:
                head, alias_name = imported.rsplit(" as ", 1)
                alias_name = alias_name.strip()
                target_name = head.strip()
            elif imported.startswith("from "):
                # from module import name
                match = re.match(
                    r"from\s+(\S+)\s+import\s+(\S+)", imported
                )
                if match:
                    target_name = f"{match.group(1)}.{match.group(2)}"
                    alias_name = match.group(2)
            elif imported.startswith("import "):
                target_name = imported[len("import ") :].strip()
                alias_name = target_name.split(".", 1)[0]
            if alias_name:
                alias_id = add_node(
                    ProgramNodeKind.ALIAS,
                    alias_name,
                    qualified=f"alias:{module}.{alias_name}" if module else f"alias:{alias_name}",
                    attributes={"target": target_name, "statement": imported},
                )
                add_edge(alias_id, import_id, ProgramEdgeKind.ALIASES)
                if target_name:
                    target_id = add_node(
                        ProgramNodeKind.EXTERNAL
                        if "." in target_name or target_name
                        else ProgramNodeKind.SYMBOL,
                        target_name,
                        qualified=target_name,
                        completeness=Completeness.PARTIAL,
                        attributes={"import_target": True},
                    )
                    add_edge(import_id, target_id, ProgramEdgeKind.IMPORTS)
                    # Re-export heuristic: public alias of imported name.
                    if not alias_name.startswith("_"):
                        export_id = add_node(
                            ProgramNodeKind.EXPORT,
                            alias_name,
                            qualified=f"export:{module}.{alias_name}" if module else f"export:{alias_name}",
                        )
                        add_edge(module_id, export_id, ProgramEdgeKind.EXPORTS)
                        add_edge(export_id, target_id, ProgramEdgeKind.RE_EXPORTS)

        # Calls.
        for call in record.calls:
            owner, separator, callee = call.partition("->")
            if not separator:
                owner, callee = "<module>", call
            owner_symbol = owner if owner != "<module>" else ""
            owner_id = symbol_ids.get(owner_symbol)
            if owner_id is None and owner_symbol:
                owner_id = add_node(
                    ProgramNodeKind.SYMBOL,
                    owner_symbol,
                    qualified=f"{module}.{owner_symbol}" if module else owner_symbol,
                    completeness=Completeness.PARTIAL,
                )
                symbol_ids[owner_symbol] = owner_id
            if owner_id is None:
                owner_id = module_id

            callee_leaf = callee.rsplit(".", 1)[-1]
            if callee in _DYNAMIC_CALLEES or callee_leaf in _DYNAMIC_CALLEES:
                frontier_id = add_node(
                    ProgramNodeKind.FRONTIER,
                    callee,
                    qualified=f"frontier:dynamic:{path}:{callee}",
                    completeness=Completeness.FRONTIER,
                    attributes={"reason": "dynamic_dispatch"},
                )
                add_edge(
                    owner_id,
                    frontier_id,
                    ProgramEdgeKind.CALLS,
                )
                # Mark as incomplete call via attributes on a separate frontier ref.
                frontier.add(f"dynamic:{path}:{callee}")
                continue

            # Prefer in-module symbol target.
            target_id = None
            for key in (callee, callee_leaf, f"{owner_symbol}.{callee_leaf}" if owner_symbol else ""):
                if key and key in symbol_ids:
                    target_id = symbol_ids[key]
                    break
            if target_id is None:
                # Create a partial symbol / external node.
                completeness = Completeness.PARTIAL
                kind = ProgramNodeKind.SYMBOL
                if callee_leaf in {
                    "print",
                    "len",
                    "str",
                    "int",
                    "open",
                    "range",
                } or callee.startswith(
                    ("os.", "sys.", "subprocess.", "requests.")
                ):
                    kind = ProgramNodeKind.EXTERNAL
                    completeness = Completeness.FRONTIER
                    frontier.add(f"external:{path}:{callee}")
                target_id = add_node(
                    kind,
                    callee,
                    qualified=callee,
                    completeness=completeness,
                    attributes={"callee": callee, "unresolved_local": kind is ProgramNodeKind.SYMBOL},
                )
            add_edge(
                owner_id,
                target_id,
                ProgramEdgeKind.CALLS,
                callee=callee,
                owner=owner_symbol,
            )

            # Decorator / callback heuristics.
            if callee_leaf in {"decorator", "wraps"} or callee_leaf.endswith(
                "_decorator"
            ):
                dec_id = add_node(
                    ProgramNodeKind.DECORATOR,
                    callee,
                    qualified=f"decorator:{callee}",
                )
                add_edge(owner_id, dec_id, ProgramEdgeKind.DECORATES)
            if "callback" in callee_leaf.lower() or callee_leaf in {
                "on_success",
                "on_error",
                "listener",
            }:
                cb_id = add_node(
                    ProgramNodeKind.CALLBACK,
                    callee,
                    qualified=f"callback:{callee}",
                )
                add_edge(owner_id, cb_id, ProgramEdgeKind.CALLBACK_TO)
            if callee_leaf in {"register", "provide", "bind"}:
                di_id = add_node(
                    ProgramNodeKind.DI_BINDING,
                    callee,
                    qualified=f"di:{path}:{callee}",
                )
                add_edge(owner_id, di_id, ProgramEdgeKind.INJECTS)
                add_edge(owner_id, di_id, ProgramEdgeKind.REGISTERS)

        # State transitions / data flow.
        for transition in record.state_transitions:
            # owner:name:operation[:value]
            parts = transition.split(":")
            owner = parts[0] if parts else "<module>"
            owner_id = symbol_ids.get(owner, module_id)
            state_id = add_node(
                ProgramNodeKind.STATE,
                transition,
                qualified=f"state:{path}:{transition}",
                attributes={"transition": transition},
            )
            add_edge(owner_id, state_id, ProgramEdgeKind.STATE_FLOW)
            # Field / variable data flow.
            if len(parts) >= 3 and parts[2].startswith("assign"):
                field_name = parts[1]
                field_id = add_node(
                    ProgramNodeKind.FIELD if "." in field_name else ProgramNodeKind.VARIABLE,
                    field_name,
                    qualified=f"field:{module}.{field_name}" if module else f"field:{field_name}",
                )
                add_edge(owner_id, field_id, ProgramEdgeKind.DATA_FLOW)
                add_edge(field_id, owner_id, ProgramEdgeKind.FIELD_OF)

        # Richer Python source facts when source body is available.
        if source.source and language.startswith("python") and not record.parse_error:
            self._enrich_from_python_source(
                source.source,
                module=module,
                path=path,
                symbol_ids=symbol_ids,
                module_id=module_id,
                add_node=add_node,
                add_edge=add_edge,
                frontier=frontier,
            )

        if record.parse_error:
            frontier.add(f"parse_error:{path}")

        # Documentation node for markdown-like names is handled elsewhere;
        # attach a documentation edge for modules that look like docs.
        if path.endswith((".md", ".rst", ".txt")) or "docs/" in path:
            doc_id = add_node(
                ProgramNodeKind.DOCUMENTATION,
                PurePosixPath(path).name,
                qualified=f"docs:{path}",
            )
            add_edge(doc_id, module_id, ProgramEdgeKind.DOCUMENTS)

        return PathComponent(
            path=path,
            content_key=source.content_key,
            nodes=tuple(nodes.values()),
            edges=tuple(edges),
            frontier_refs=tuple(sorted(frontier)),
            exclusion_refs=tuple(sorted(exclusions)),
        )

    def _symbol_kind(self, symbol: str, record: ASTBlobRecord) -> ProgramNodeKind:
        simple = _simple(symbol)
        # Class if any interface entry starts with the symbol and has bases.
        for interface in record.interfaces:
            if interface.startswith(symbol + "("):
                return ProgramNodeKind.CLASS
            if interface.startswith(symbol + ":"):
                return (
                    ProgramNodeKind.METHOD if "." in symbol else ProgramNodeKind.FUNCTION
                )
        if simple == "__init__":
            return ProgramNodeKind.CONSTRUCTOR
        if "." in symbol:
            # Heuristic: nested names are methods when parent is also a symbol.
            parent = symbol.rsplit(".", 1)[0]
            if parent in record.qualified_symbols:
                return ProgramNodeKind.METHOD
        # Default function if appears as callable interface.
        for interface in record.interfaces:
            if interface.startswith(symbol + ":"):
                return ProgramNodeKind.FUNCTION
        # Class without interfaces still listed as qualified symbol.
        # Prefer CLASS when no '.' and capitalized.
        if "." not in symbol and simple[:1].isupper():
            return ProgramNodeKind.CLASS
        if "." not in symbol:
            return ProgramNodeKind.FUNCTION
        return ProgramNodeKind.SYMBOL

    def _add_signature_flow(
        self,
        add_node,
        add_edge,
        method_id: str,
        name: str,
        signature: str,
    ) -> None:
        # signature like "def process(self, a, b) -> R" or "async def ..."
        params_match = re.search(r"\((.*)\)", signature)
        if params_match:
            raw_params = params_match.group(1)
            for index, part in enumerate(raw_params.split(",")):
                part = part.strip()
                if not part or part in {"self", "cls", "*", "/"}:
                    continue
                if part.startswith("*"):
                    part = part.lstrip("*")
                pname = part.split(":", 1)[0].split("=", 1)[0].strip()
                if not pname:
                    continue
                param_id = add_node(
                    ProgramNodeKind.PARAMETER,
                    pname,
                    qualified=f"param:{name}:{pname}",
                    attributes={"position": index, "signature": part},
                )
                add_edge(param_id, method_id, ProgramEdgeKind.PARAMETER_OF)
                add_edge(method_id, param_id, ProgramEdgeKind.DATA_FLOW)
        ret_match = re.search(r"->\s*(.+)$", signature)
        if ret_match:
            ret = ret_match.group(1).strip()
            ret_id = add_node(
                ProgramNodeKind.RETURN,
                ret,
                qualified=f"return:{name}",
                attributes={"annotation": ret},
            )
            add_edge(method_id, ret_id, ProgramEdgeKind.RETURNS)

    def _enrich_from_python_source(
        self,
        source: str,
        *,
        module: str,
        path: str,
        symbol_ids: dict[str, str],
        module_id: str,
        add_node,
        add_edge,
        frontier: set[str],
    ) -> None:
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            frontier.add(f"enrich_parse_error:{path}")
            return

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.scope: list[str] = []

            def _owner(self) -> str:
                return ".".join(self.scope)

            def _owner_id(self) -> str:
                owner = self._owner()
                return symbol_ids.get(owner, module_id)

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self.scope.append(node.name)
                for base in node.bases:
                    base_name = self._expr_name(base)
                    if not base_name:
                        continue
                    owner_id = self._owner_id()
                    # owner currently includes class name; parent class symbol
                    class_symbol = ".".join(self.scope)
                    class_id = symbol_ids.get(class_symbol) or symbol_ids.get(node.name)
                    if class_id is None:
                        class_id = owner_id
                    base_id = add_node(
                        ProgramNodeKind.INTERFACE
                        if base_name.endswith("Protocol")
                        else ProgramNodeKind.CLASS,
                        base_name,
                        qualified=base_name,
                        completeness=Completeness.PARTIAL,
                    )
                    add_edge(
                        class_id,
                        base_id,
                        ProgramEdgeKind.IMPLEMENTS
                        if "Protocol" in base_name or base_name in {"ABC", "abc.ABC"}
                        else ProgramEdgeKind.OVERRIDES,
                    )
                for decorator in node.decorator_list:
                    self._decorator(decorator)
                self.generic_visit(node)
                self.scope.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._function(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._function(node)

            def _function(
                self, node: ast.FunctionDef | ast.AsyncFunctionDef
            ) -> None:
                self.scope.append(node.name)
                fn_symbol = ".".join(self.scope)
                fn_id = symbol_ids.get(fn_symbol) or symbol_ids.get(node.name) or module_id
                for index, arg in enumerate(node.args.args):
                    if arg.arg in {"self", "cls"}:
                        continue
                    param_id = add_node(
                        ProgramNodeKind.PARAMETER,
                        arg.arg,
                        qualified=f"param:{fn_symbol}:{arg.arg}",
                        span=_span_dict(
                            getattr(arg, "lineno", 0),
                            getattr(arg, "end_lineno", getattr(arg, "lineno", 0)),
                            getattr(arg, "col_offset", 0),
                            getattr(arg, "end_col_offset", 0),
                        ),
                        attributes={"position": index},
                    )
                    add_edge(param_id, fn_id, ProgramEdgeKind.PARAMETER_OF)
                    add_edge(fn_id, param_id, ProgramEdgeKind.DATA_FLOW)
                if node.returns is not None:
                    ret_name = self._expr_name(node.returns) or ast.dump(node.returns)
                    ret_id = add_node(
                        ProgramNodeKind.RETURN,
                        ret_name,
                        qualified=f"return:{fn_symbol}",
                    )
                    add_edge(fn_id, ret_id, ProgramEdgeKind.RETURNS)
                for decorator in node.decorator_list:
                    self._decorator(decorator)
                # Context managers inside function body.
                self.generic_visit(node)
                self.scope.pop()

            def _decorator(self, node: ast.AST) -> None:
                name = self._expr_name(node)
                if not name:
                    return
                owner_id = self._owner_id()
                dec_id = add_node(
                    ProgramNodeKind.DECORATOR,
                    name,
                    qualified=f"decorator:{module}.{name}" if module else f"decorator:{name}",
                )
                add_edge(owner_id, dec_id, ProgramEdgeKind.DECORATES)
                if name.rsplit(".", 1)[-1] in {"register", "route", "app", "provider"}:
                    reg_id = add_node(
                        ProgramNodeKind.REGISTRY,
                        name,
                        qualified=f"registry:{name}",
                    )
                    add_edge(owner_id, reg_id, ProgramEdgeKind.REGISTERS)

            def visit_With(self, node: ast.With) -> None:
                self._with(node)
                self.generic_visit(node)

            def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
                self._with(node)
                self.generic_visit(node)

            def _with(self, node: ast.With | ast.AsyncWith) -> None:
                owner_id = self._owner_id()
                for item in node.items:
                    name = self._expr_name(item.context_expr) or "context"
                    cm_id = add_node(
                        ProgramNodeKind.CONTEXT_MANAGER,
                        name,
                        qualified=f"ctx:{module}.{name}" if module else f"ctx:{name}",
                    )
                    add_edge(owner_id, cm_id, ProgramEdgeKind.CONTEXT_MANAGES)

            def visit_Assign(self, node: ast.Assign) -> None:
                self._assign(node.targets, node.value)
                self.generic_visit(node)

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                self._assign([node.target], node.value)
                self.generic_visit(node)

            def _assign(
                self, targets: Sequence[ast.AST], value: ast.AST | None
            ) -> None:
                owner_id = self._owner_id()
                for target in targets:
                    name = self._expr_name(target)
                    if not name:
                        continue
                    field_id = add_node(
                        ProgramNodeKind.FIELD if "." in name else ProgramNodeKind.VARIABLE,
                        name,
                        qualified=f"field:{module}.{name}" if module else f"field:{name}",
                    )
                    add_edge(owner_id, field_id, ProgramEdgeKind.DATA_FLOW)
                    add_edge(field_id, owner_id, ProgramEdgeKind.FIELD_OF)
                    if isinstance(value, ast.Call):
                        callee = self._expr_name(value.func)
                        if callee and _simple(callee).lower() in _FACTORY_NAMES:
                            factory_id = add_node(
                                ProgramNodeKind.FACTORY,
                                callee,
                                qualified=f"factory:{callee}",
                            )
                            add_edge(factory_id, field_id, ProgramEdgeKind.FACTORY_CREATES)

            @staticmethod
            def _expr_name(node: ast.AST | None) -> str:
                if node is None:
                    return ""
                if isinstance(node, ast.Name):
                    return node.id
                if isinstance(node, ast.Attribute):
                    parent = Visitor._expr_name(node.value)
                    return f"{parent}.{node.attr}" if parent else node.attr
                if isinstance(node, ast.Call):
                    return Visitor._expr_name(node.func)
                if isinstance(node, ast.Subscript):
                    return Visitor._expr_name(node.value)
                return ""

        Visitor().visit(tree)

    def _assemble(
        self,
        components: Mapping[str, PathComponent],
        *,
        nominations: Iterable[Mapping[str, Any]],
        impact_edges: Mapping[str, Sequence[str]],
    ) -> ProgramGraphSnapshot:
        state = _BuildState(
            roots=self._roots,
            nodes={},
            edge_keys=set(),
            edges=[],
            frontier=set(),
            exclusions=set(self._roots.excluded_roots),
            symbol_index={},
            path_modules={},
        )

        # Repository root node.
        repo_id = _node_id("repository", self._roots.forest_id, self._roots.tree_id)
        state.add_node(
            ProgramNode(
                node_id=repo_id,
                kind=ProgramNodeKind.REPOSITORY,
                name="repository",
                roots=self._roots,
                qualified_name=self._roots.forest_id,
                provenance=ProgramProvenance.EXTRACTOR,
                attributes={
                    "forest_id": self._roots.forest_id,
                    "tree_id": self._roots.tree_id,
                    "overlay_id": self._roots.overlay_id,
                },
            )
        )

        for path in sorted(components):
            component = components[path]
            state.frontier.update(component.frontier_refs)
            state.exclusions.update(component.exclusion_refs)
            state.path_modules[path] = _module_name(path)
            for node in component.nodes:
                state.add_node(node)
            # File nodes depend on repository.
            for node in component.nodes:
                if node.kind is ProgramNodeKind.FILE:
                    state.add_edge(repo_id, node.node_id, ProgramEdgeKind.CONTAINS)

        # Materialize deferred edges with current node set.
        for path in sorted(components):
            component = components[path]
            for source, target, kind, attributes in component.edges:
                try:
                    edge_kind = ProgramEdgeKind(kind)
                except ValueError:
                    state.frontier.add(f"unknown_edge_kind:{kind}")
                    continue
                state.add_edge(
                    source,
                    target,
                    edge_kind,
                    attributes=dict(attributes),
                )

        # Cross-path import resolution: link import targets to local modules.
        module_nodes = {
            node.qualified_name: node.node_id
            for node in state.nodes.values()
            if node.kind is ProgramNodeKind.MODULE
        }
        for node in list(state.nodes.values()):
            if node.kind is not ProgramNodeKind.IMPORT:
                continue
            statement = str(node.attributes.get("statement") or node.name)
            module_name = ""
            if statement.startswith("import "):
                module_name = statement[len("import ") :].split(" as ", 1)[0].strip()
            elif statement.startswith("from "):
                match = re.match(r"from\s+(\S+)\s+import\s+", statement)
                if match:
                    module_name = match.group(1).lstrip(".")
            if module_name and module_name in module_nodes:
                state.add_edge(
                    node.node_id,
                    module_nodes[module_name],
                    ProgramEdgeKind.DEPENDS_ON,
                    attributes={"resolved_import": True},
                )

        # Impact-index dependency edges (authoritative when provided as reviewed).
        for dependent, providers in sorted((impact_edges or {}).items()):
            dep_ids = state.symbol_index.get(dependent, [])
            if not dep_ids:
                # Try qualified match against node qualified names.
                dep_ids = [
                    node.node_id
                    for node in state.nodes.values()
                    if node.qualified_name == dependent or node.name == dependent
                ]
            for provider in providers:
                prov_ids = state.symbol_index.get(str(provider), [])
                if not prov_ids:
                    prov_ids = [
                        node.node_id
                        for node in state.nodes.values()
                        if node.qualified_name == str(provider)
                        or node.name == str(provider)
                    ]
                for dep_id in dep_ids:
                    for prov_id in prov_ids:
                        state.add_edge(
                            dep_id,
                            prov_id,
                            ProgramEdgeKind.DEPENDS_ON,
                            provenance=ProgramProvenance.IMPACT_INDEX,
                            attributes={"impact": True},
                        )

        # Nominated GraphRAG / runtime / vector edges remain non-authoritative.
        for raw in nominations:
            if not isinstance(raw, Mapping):
                continue
            source = str(raw.get("source") or "")
            target = str(raw.get("target") or "")
            kind_text = str(raw.get("kind") or ProgramEdgeKind.RELATED_TO.value)
            channel = str(raw.get("provenance") or "graphrag").lower()
            provenance = {
                "graphrag": ProgramProvenance.GRAPHRAG,
                "runtime": ProgramProvenance.RUNTIME,
                "vector": ProgramProvenance.VECTOR,
                "history": ProgramProvenance.HISTORY,
                "model": ProgramProvenance.MODEL,
            }.get(channel, ProgramProvenance.GRAPHRAG)
            if source not in state.nodes:
                # Allow lookup by qualified name.
                ids = state.symbol_index.get(source, [])
                source = ids[0] if len(ids) == 1 else source
            if target not in state.nodes:
                ids = state.symbol_index.get(target, [])
                target = ids[0] if len(ids) == 1 else target
            if source not in state.nodes or target not in state.nodes:
                state.frontier.add(
                    f"nominated_unresolved:{source}->{target}"
                )
                continue
            try:
                kind = ProgramEdgeKind(kind_text)
            except ValueError:
                kind = ProgramEdgeKind.RELATED_TO
            # Force nomination authority boundary.
            state.add_edge(
                source,
                target,
                kind if kind is not ProgramEdgeKind.RELATED_TO else ProgramEdgeKind.RELATED_TO,
                provenance=provenance,
                trust=ProgramTrust.NOMINATED,
                authority=ProgramAuthority.NOMINATED,
                completeness=Completeness.FRONTIER,
                confidence=int(raw.get("confidence") or 20),
                attributes={
                    "nominated": True,
                    "channel": channel,
                    **{
                        str(key): value
                        for key, value in raw.items()
                        if key
                        not in {
                            "source",
                            "target",
                            "kind",
                            "provenance",
                            "confidence",
                        }
                    },
                },
            )

        # Included / excluded / generated / native roots from identity.
        for root in self._roots.excluded_roots:
            state.exclusions.add(f"excluded_root:{root}")
        for root in self._roots.generated_roots:
            state.frontier.add(f"generated_root:{root}")
        for root in self._roots.native_roots:
            state.frontier.add(f"native_root:{root}")
        for tombstone in self._roots.tombstones:
            state.exclusions.add(f"tombstone:{tombstone}")

        complete = not state.frontier and not state.exclusions
        return ProgramGraphSnapshot(
            roots=self._roots,
            nodes=tuple(state.nodes.values()),
            edges=tuple(state.edges),
            frontier_refs=tuple(sorted(state.frontier)),
            exclusion_refs=tuple(sorted(state.exclusions)),
            complete=complete,
        )


def build_program_dependency_graph(
    roots: ProgramGraphRoots | Mapping[str, Any],
    sources: Iterable[PathSource | Mapping[str, Any]] | Mapping[str, str] = (),
    **kwargs: Any,
) -> ProgramGraph:
    """Convenience builder returning a :class:`ProgramGraph`."""

    if isinstance(sources, Mapping) and sources and all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in sources.items()
    ):
        # Treat as path -> python source mapping when values are source text.
        sample_key = next(iter(sources))
        sample_val = sources[sample_key]
        if "\n" in sample_val or sample_val.strip().startswith(
            ("def ", "class ", "import ", "from ", '"""', "'''")
        ) or len(sample_val) > 64:
            return ProgramDependencyGraph.from_python_sources(
                roots, sources, **kwargs
            ).graph  # type: ignore[return-value]
    graph = ProgramDependencyGraph(roots)
    return graph.build(sources, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "PATH_COMPONENT_SCHEMA",
    "PROGRAM_DEPENDENCY_GRAPH_SCHEMA",
    "PROGRAM_DEPENDENCY_GRAPH_VERSION",
    "PathComponent",
    "PathSource",
    "ProgramDependencyGraph",
    "ProgramDependencyGraphError",
    "build_program_dependency_graph",
]
