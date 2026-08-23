"""Hermetic PCAR-003 ArchitectureIR graph extraction tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.architecture_ir import (
    ArchitectureIR,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.contracts import (
    Confidence,
    EdgeKind,
    NodeKind,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.graph_builder import (
    ARCHITECTURE_GRAPH_BUILDER_VERSION,
    ARCHITECTURE_GRAPH_EVIDENCE,
    DEFAULT_FRESHNESS,
    DEFAULT_PROTECTED_PATHS,
    EXTRACTOR_IDENTITY,
    GRAPH_EXTRACTOR_IDENTITY,
    TASK_ID,
    ArchitectureGraphBuilder,
    ArchitectureGraphBuilderError,
    ArchitectureGraphEscapeError,
    build_architecture_graph,
    call_targets,
    edges_of,
    extract_architecture_graph,
    module_name_from_path,
    nodes_of,
    normalize_relative_path,
)
from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

_TREE = "pcar-003-fixture-tree"
_FRESHNESS = "pcar-003-graph-fixture"

_PKG_INIT = """from .ops import Handler, run_operation
from .ops import SCHEMA as OPS_SCHEMA

__all__ = ["Handler", "run_operation", "OPS_SCHEMA"]
"""

_PKG_OPS = '''"""Representative operations module."""
import json
from pathlib import Path

SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}}


def run_operation(path: str) -> dict:
    text = Path(path).read_text()
    payload = json.loads(text)
    return payload


class Handler:
    def execute(self, name: str):
        handler = getattr(self, name)
        return handler()

    def persist(self) -> None:
        self.store.commit()

    def observe(self) -> None:
        print("ok")

    def __getattr__(self, item):
        return self.observe


def make_handler() -> Handler:
    return Handler()
'''

_PKG_DYNAMIC = '''class Router:
    def dispatch(self, name: str):
        handler = getattr(self, name)
        return handler()

    def read_state(self):
        return 1

    def write_state(self):
        return 2

    def __getattr__(self, item):
        return self.read_state
'''

_PKG_LITERAL = '''class Router:
    def dispatch(self, name: str):
        handler = getattr(self, "read_state")
        return handler()

    def read_state(self):
        return 1

    def write_state(self):
        return 2
'''

_BOOM = """raise RuntimeError("imported")

def surviving() -> int:
    return 1
"""

_TEST_OPS = '''from pkg.ops import run_operation


def test_run_operation() -> None:
    assert run_operation("fixture.json") == {}
'''


def _sources() -> dict[str, str]:
    return {
        "pkg/__init__.py": _PKG_INIT,
        "pkg/ops.py": _PKG_OPS,
        "pkg/dynamic.py": _PKG_DYNAMIC,
        "test/test_ops.py": _TEST_OPS,
        "proofs/run_operation.proof.json": json.dumps(
            {"proves": "pkg.ops.run_operation", "obligation": "run_operation preserves schema"}
        ),
        "pkg/ops.schema.json": json.dumps(
            {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}
        ),
    }


def _graph(sources: dict[str, str] | None = None) -> ArchitectureIR:
    return extract_architecture_graph(
        sources or _sources(),
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )


def _node(graph: ArchitectureIR, kind: NodeKind, identity: str):
    exact = f"n:{kind.value}:{identity}"
    for node in graph.nodes:
        if node.node_id == exact:
            return node
    matches = [
        node
        for node in graph.nodes
        if node.kind is kind and identity in node.node_id
    ]
    assert matches, (
        f"missing {kind.value} node containing {identity!r}: "
        f"{[n.node_id for n in graph.nodes if n.kind is kind]}"
    )
    return matches[0]


def _symbol(graph: ArchitectureIR, qualified: str):
    return _node(graph, NodeKind.SYMBOL, qualified)


def test_extractor_identity_and_evidence_pins() -> None:
    assert EXTRACTOR_IDENTITY == "pcar-003-architecture-graph-builder"
    assert GRAPH_EXTRACTOR_IDENTITY == EXTRACTOR_IDENTITY
    assert ARCHITECTURE_GRAPH_EVIDENCE == "pcar/architecture-graph@1"
    assert ARCHITECTURE_GRAPH_BUILDER_VERSION == "architecture-graph-builder@1"
    assert TASK_ID == "PCAR-003"
    assert DEFAULT_FRESHNESS == "pcar-003-architecture-graph"
    assert "test/api/architecture_refactorer/test_board.py" in DEFAULT_PROTECTED_PATHS
    assert module_name_from_path("pkg/ops.py") == "pkg.ops"
    assert module_name_from_path("pkg/__init__.py") == "pkg"
    assert normalize_relative_path("./pkg/ops.py") == "pkg/ops.py"
    with pytest.raises(ArchitectureGraphEscapeError):
        normalize_relative_path("../escape.py")
    with pytest.raises(ArchitectureGraphEscapeError):
        normalize_relative_path("/abs/escape.py")


def test_graph_extraction_emits_modules_symbols_schemas_tests_and_proofs() -> None:
    graph = _graph()
    kinds = {node.kind for node in graph.nodes}
    assert {
        NodeKind.REPOSITORY,
        NodeKind.PACKAGE,
        NodeKind.MODULE,
        NodeKind.FILE,
        NodeKind.SYMBOL,
        NodeKind.SCHEMA,
        NodeKind.OPERATION,
        NodeKind.EFFECT,
        NodeKind.TEST,
        NodeKind.PROOF,
    } <= kinds
    assert _node(graph, NodeKind.MODULE, "pkg.ops")
    assert _node(graph, NodeKind.PACKAGE, "pkg")
    assert _node(graph, NodeKind.FILE, "pkg/ops.py")
    assert _symbol(graph, "pkg.ops.run_operation")
    assert _symbol(graph, "pkg.ops.Handler")
    assert _symbol(graph, "pkg.ops.Handler.execute")
    assert _node(graph, NodeKind.OPERATION, "pkg.ops.run_operation")
    schema_ids = {node.node_id for node in nodes_of(graph, NodeKind.SCHEMA)}
    assert any("pkg.ops.SCHEMA" in node_id or "pkg/ops.schema.json" in node_id for node_id in schema_ids)
    assert _node(graph, NodeKind.TEST, "test.test_ops.test_run_operation")
    assert _node(graph, NodeKind.PROOF, "proofs/run_operation.proof.json")
    contains = {(edge.source, edge.target) for edge in edges_of(graph, EdgeKind.CONTAINS)}
    module_id = _node(graph, NodeKind.MODULE, "pkg.ops").node_id
    file_id = _node(graph, NodeKind.FILE, "pkg/ops.py").node_id
    symbol_id = _symbol(graph, "pkg.ops.run_operation").node_id
    assert (module_id, file_id) in contains
    assert any(target == symbol_id for source, target in contains)


def test_import_and_call_edges_bind_in_repo_targets() -> None:
    graph = _graph()
    imports = edges_of(graph, EdgeKind.IMPORTS)
    assert imports
    init_module = _node(graph, NodeKind.MODULE, "pkg").node_id
    ops_module = _node(graph, NodeKind.MODULE, "pkg.ops").node_id
    run_symbol = _symbol(graph, "pkg.ops.run_operation").node_id
    imported_ops = [
        edge
        for edge in imports
        if edge.source == init_module and edge.target in {ops_module, run_symbol}
    ]
    assert imported_ops
    assert all(edge.provenance.confidence in {Confidence.EXACT, Confidence.CONSERVATIVE} for edge in imported_ops)
    reexports = edges_of(graph, EdgeKind.REEXPORTS)
    assert any(edge.source == init_module and edge.target == run_symbol for edge in reexports)
    calls = edges_of(graph, EdgeKind.CALLS)
    test_fn = _symbol(graph, "test.test_ops.test_run_operation").node_id
    assert any(edge.source == test_fn and edge.target == run_symbol for edge in calls)
    handler = _symbol(graph, "pkg.ops.Handler")
    assert handler.kind is NodeKind.SYMBOL
    maker = _symbol(graph, "pkg.ops.make_handler")
    constructs = edges_of(graph, EdgeKind.CONSTRUCTS)
    assert any(edge.source == maker.node_id and edge.target == handler.node_id for edge in constructs)
    execute = _symbol(graph, "pkg.ops.Handler.execute")
    persist = _symbol(graph, "pkg.ops.Handler.persist")
    observe = _symbol(graph, "pkg.ops.Handler.observe")
    execute_targets = set(call_targets(graph, execute.node_id))
    assert persist.node_id in execute_targets
    assert observe.node_id in execute_targets


def test_provenance_binds_extractor_tree_freshness_and_exact_spans() -> None:
    sources = _sources()
    graph = _graph(sources)
    ops_source = sources["pkg/ops.py"]
    import_line = next(
        index
        for index, line in enumerate(ops_source.splitlines(), start=1)
        if line.startswith("import json")
    )
    run_start = next(
        index
        for index, line in enumerate(ops_source.splitlines(), start=1)
        if line.startswith("def run_operation")
    )
    for fact in (*graph.nodes, *graph.edges):
        assert fact.provenance.extractor_identity == EXTRACTOR_IDENTITY
        assert fact.provenance.repository_tree == _TREE
        assert fact.provenance.freshness == _FRESHNESS
        assert fact.provenance.confidence in {
            Confidence.EXACT,
            Confidence.CONSERVATIVE,
            Confidence.HEURISTIC,
            Confidence.OPAQUE,
        }
        span = fact.provenance.span
        assert span.path in sources
        body = sources[span.path]
        last = body.count("\n") + (0 if body.endswith("\n") else 1)
        if not body:
            last = 1
        if body.endswith("\n"):
            last = max(1, body.count("\n"))
        assert 1 <= span.start_line <= span.end_line <= last
    import_edges = [
        edge
        for edge in edges_of(graph, EdgeKind.IMPORTS)
        if edge.provenance.span.path == "pkg/ops.py"
        and edge.provenance.span.start_line == import_line
    ]
    assert import_edges
    run_node = _symbol(graph, "pkg.ops.run_operation")
    assert run_node.provenance.span.path == "pkg/ops.py"
    assert run_node.provenance.span.start_line == run_start
    json_line = ops_source.splitlines()[import_line - 1]
    assert "import json" in json_line


def test_deterministic_graph_root_and_round_trip() -> None:
    first = _graph()
    second = extract_architecture_graph(
        _sources(),
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    third = ArchitectureGraphBuilder.from_sources(
        dict(reversed(list(_sources().items()))),
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    ).build()
    assert first == second == third
    assert first.content_identity == second.content_identity == third.content_identity
    payload = first.to_dict()
    restored = ArchitectureIR.from_mapping(payload)
    assert restored == first
    claimed = payload.pop("content_identity")
    validate_cid(claimed, codecs=("dag-json",))
    assert claimed == cid_for_dag_json(payload)
    assert claimed == first.content_identity
    assert not claimed.startswith("sha256:")


def test_does_not_execute_inspected_modules() -> None:
    graph = extract_architecture_graph(
        {"pkg/boom.py": _BOOM},
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    surviving = _symbol(graph, "pkg.boom.surviving")
    assert surviving.kind is NodeKind.SYMBOL
    assert surviving.provenance.confidence is Confidence.EXACT


def test_syntax_error_is_opaque_not_promoted() -> None:
    graph = extract_architecture_graph(
        {"pkg/broken.py": "def nope(\n"},
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    opaque = [
        node
        for node in graph.nodes
        if node.provenance.confidence is Confidence.OPAQUE and "pkg/broken.py" in node.provenance.span.path
    ]
    assert opaque
    assert not any(
        node.kind is NodeKind.SYMBOL and "nope" in node.node_id for node in graph.nodes
    )


def test_conservative_dynamic_dispatch_is_a_superset_of_literal_getattr() -> None:
    dynamic = extract_architecture_graph(
        {"pkg/router.py": _PKG_DYNAMIC},
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    literal = extract_architecture_graph(
        {"pkg/router.py": _PKG_LITERAL},
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    dynamic_dispatch = _symbol(dynamic, "pkg.router.Router.dispatch")
    literal_dispatch = _symbol(literal, "pkg.router.Router.dispatch")
    dynamic_targets = set(call_targets(dynamic, dynamic_dispatch.node_id))
    literal_targets = set(call_targets(literal, literal_dispatch.node_id))
    read_state = _symbol(dynamic, "pkg.router.Router.read_state").node_id
    write_state = _symbol(dynamic, "pkg.router.Router.write_state").node_id
    assert read_state in dynamic_targets
    assert write_state in dynamic_targets
    assert read_state in literal_targets
    assert dynamic_targets >= literal_targets
    dynamic_calls = [
        edge
        for edge in edges_of(dynamic, EdgeKind.CALLS)
        if edge.source == dynamic_dispatch.node_id
    ]
    assert dynamic_calls
    assert all(edge.provenance.confidence is not Confidence.EXACT for edge in dynamic_calls)
    assert any(
        edge.provenance.confidence in {Confidence.CONSERVATIVE, Confidence.OPAQUE}
        for edge in dynamic_calls
    )
    assert any(edge.provenance.confidence is Confidence.OPAQUE for edge in dynamic_calls)


def test_protected_external_and_escape_paths_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ArchitectureGraphEscapeError, match="repository-relative"):
        extract_architecture_graph({"../escape.py": "x = 1\n"}, repository_tree=_TREE)
    with pytest.raises(ArchitectureGraphEscapeError, match="repository-relative"):
        extract_architecture_graph({"/tmp/escape.py": "x = 1\n"}, repository_tree=_TREE)
    protected = DEFAULT_PROTECTED_PATHS[0]
    with pytest.raises(ArchitectureGraphEscapeError, match="protected path"):
        extract_architecture_graph({protected: "# sealed\n"}, repository_tree=_TREE)
    with pytest.raises(ArchitectureGraphEscapeError, match="submodule"):
        extract_architecture_graph(
            {"ipfs_datasets_py/hidden.py": "x = 1\n"},
            repository_tree=_TREE,
        )
    root = tmp_path / "repo"
    root.mkdir()
    (root / "ok.py").write_text("value = 1\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("stolen = 1\n", encoding="utf-8")
    (root / "link.py").symlink_to(outside)
    with pytest.raises(ArchitectureGraphEscapeError, match="symlink"):
        build_architecture_graph(root, repository_tree=_TREE, freshness=_FRESHNESS)
    vendor = root / "vendor"
    vendor.mkdir()
    (vendor / ".git").write_text("gitdir: /tmp/other\n", encoding="utf-8")
    (vendor / "nested.py").write_text("x = 1\n", encoding="utf-8")
    (root / "link.py").unlink()
    with pytest.raises(ArchitectureGraphEscapeError, match="submodule"):
        build_architecture_graph(root, repository_tree=_TREE, freshness=_FRESHNESS)


def test_filesystem_extraction_matches_in_memory_and_honors_exclusions(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text(_PKG_INIT, encoding="utf-8")
    (root / "pkg" / "ops.py").write_text(_PKG_OPS, encoding="utf-8")
    (root / "pkg" / "skip.py").write_text("def ignored():\n    return 0\n", encoding="utf-8")
    disk = ArchitectureGraphBuilder(
        root,
        repository_tree=_TREE,
        freshness=_FRESHNESS,
        exclusions=("pkg/skip.py",),
    ).build()
    memory = extract_architecture_graph(
        {"pkg/__init__.py": _PKG_INIT, "pkg/ops.py": _PKG_OPS},
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    assert _symbol(disk, "pkg.ops.run_operation")
    assert not any("skip.ignored" in node.node_id for node in disk.nodes)
    assert {node.node_id for node in nodes_of(disk, NodeKind.SYMBOL)} >= {
        node.node_id for node in nodes_of(memory, NodeKind.SYMBOL) if "pkg.ops" in node.node_id
    }


def test_source_byte_bound_fails_closed() -> None:
    with pytest.raises(ArchitectureGraphBuilderError, match="byte bound"):
        extract_architecture_graph(
            {"pkg/huge.py": "x = 1\n"},
            repository_tree=_TREE,
            max_source_bytes=1,
        )


def test_build_architecture_graph_requires_root_or_sources() -> None:
    with pytest.raises(ArchitectureGraphBuilderError):
        ArchitectureGraphBuilder(repository_tree=_TREE).build()
    graph = build_architecture_graph(
        sources={"pkg/mod.py": "def ping():\n    return 1\n"},
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    assert _symbol(graph, "pkg.mod.ping")
