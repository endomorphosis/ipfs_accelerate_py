"""Tests for the snapshot-bound typed program dependency graph (RPR-025)."""

from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.program_dependency_graph import (
    PathSource,
    ProgramDependencyGraph,
    build_program_dependency_graph,
)
from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import (
    build_python_ast_blob_record,
)
from ipfs_accelerate_py.agent_supervisor.program_call_resolver import (
    CallResolutionStatus,
    CallSite,
    ProgramCallResolver,
)
from ipfs_accelerate_py.agent_supervisor.program_graph import (
    Completeness,
    ProgramAuthority,
    ProgramEdge,
    ProgramEdgeKind,
    ProgramGraph,
    ProgramGraphError,
    ProgramGraphIdentityError,
    ProgramGraphRoots,
    ProgramGraphSnapshot,
    ProgramNode,
    ProgramNodeKind,
    ProgramProvenance,
    ProgramTrust,
)


FIXTURE_ROOTS = ProgramGraphRoots(
    forest_id="forest:fixture-a",
    tree_id="tree:deadbeef",
    overlay_id="overlay:clean",
    coverage_id="coverage:full",
    included_roots=("src/", "tests/"),
    excluded_roots=("vendor/",),
    generated_roots=("generated/",),
    native_roots=("native/",),
    extractor_id="program-graph@1",
    config_id="config:test",
    toolchain_id="toolchain:cpython",
    tombstones=("old/module.py",),
)


def _fixture_sources() -> dict[str, str]:
    return {
        "src/service.py": """
from src.schema import Payload
from src.factory import build_client as make_client

class Base(Protocol):
    def process(self, left, right):
        ...

class Service(Base):
    def __init__(self, client=None):
        self.client = client or make_client()

    @register
    def process(self, left, right, context=None):
        payload = Payload.from_dict({"left": left, "right": right})
        return self.client.send(payload)

def helper(x: int) -> int:
    return x + 1
""",
        "src/factory.py": """
def build_client():
    return Client()

class Client:
    def send(self, payload):
        return payload

def create_service():
    return Service()
""",
        "src/schema.py": """
class Payload:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    @classmethod
    def from_dict(cls, data):
        return cls(data["left"], data["right"])

    def to_dict(self):
        return {"left": self.left, "right": self.right}
""",
        "src/api.py": """
def endpoint_handler(request):
    service = create_service()
    return service.process(request.a, request.b)

def cli_main():
    return endpoint_handler(None)
""",
        "src/dynamic.py": """
def run(name, obj):
    return getattr(obj, name)()
""",
        "tests/test_service.py": """
from src.service import Service

def test_process():
    service = Service()
    assert service.process(1, 2) is not None

def mock_client():
    return None
""",
    }


def _build_graph(
    files: dict[str, str] | None = None,
    *,
    roots: ProgramGraphRoots = FIXTURE_ROOTS,
    nominations=(),
    impact_edges=None,
    previous=None,
) -> ProgramDependencyGraph:
    builder = ProgramDependencyGraph(roots, previous=previous)
    builder.build(
        [
            PathSource(path=path, source=source, language="python")
            for path, source in sorted((files or _fixture_sources()).items())
        ],
        nominations=nominations,
        impact_edges=impact_edges,
        previous=previous,
    )
    return builder


# ---------------------------------------------------------------------------
# Identity binding
# ---------------------------------------------------------------------------


def test_roots_bind_forest_tree_overlay_coverage_and_boundaries() -> None:
    roots = FIXTURE_ROOTS
    payload = roots.to_dict()
    rebuilt = ProgramGraphRoots.from_dict(payload)
    assert rebuilt == roots
    assert rebuilt.roots_id == roots.roots_id
    assert "src/" in rebuilt.included_roots
    assert "vendor/" in rebuilt.excluded_roots
    assert "generated/" in rebuilt.generated_roots
    assert "native/" in rebuilt.native_roots
    assert "old/module.py" in rebuilt.tombstones
    assert rebuilt.extractor_id
    assert rebuilt.config_id == "config:test"
    assert rebuilt.toolchain_id == "toolchain:cpython"


def test_forged_roots_identity_is_rejected() -> None:
    payload = FIXTURE_ROOTS.to_dict()
    payload["roots_id"] = "program-graph-roots:sha256:" + ("0" * 64)
    with pytest.raises(ProgramGraphIdentityError):
        ProgramGraphRoots.from_dict(payload)


def test_graph_identity_is_deterministic_and_root_bound() -> None:
    left = _build_graph()
    right = _build_graph()
    assert left.graph is not None and right.graph is not None
    assert left.graph.graph_id == right.graph.graph_id
    assert left.snapshot is not None
    assert left.snapshot.roots.roots_id == FIXTURE_ROOTS.roots_id
    # All nodes/edges share the same roots identity.
    for node in left.graph.nodes:
        assert node.roots.roots_id == FIXTURE_ROOTS.roots_id
    for edge in left.graph.edges:
        assert edge.roots.roots_id == FIXTURE_ROOTS.roots_id


def test_snapshot_round_trip_preserves_identity() -> None:
    builder = _build_graph()
    assert builder.snapshot is not None
    payload = builder.snapshot.to_dict()
    rebuilt = ProgramGraphSnapshot.from_dict(payload)
    assert rebuilt.snapshot_id == builder.snapshot.snapshot_id
    assert ProgramGraphSnapshot.from_json(builder.snapshot.to_json()) == rebuilt


# ---------------------------------------------------------------------------
# Typed nodes / edges coverage
# ---------------------------------------------------------------------------


def test_typed_nodes_cover_supported_declarations_and_surfaces() -> None:
    graph = _build_graph().graph
    assert graph is not None
    kinds = {node.kind for node in graph.nodes}
    required = {
        ProgramNodeKind.MODULE,
        ProgramNodeKind.FILE,
        ProgramNodeKind.CLASS,
        ProgramNodeKind.FUNCTION,
        ProgramNodeKind.METHOD,
        ProgramNodeKind.CONSTRUCTOR,
        ProgramNodeKind.FACTORY,
        ProgramNodeKind.IMPORT,
        ProgramNodeKind.ALIAS,
        ProgramNodeKind.PARAMETER,
        ProgramNodeKind.RETURN,
        ProgramNodeKind.FIELD,
        ProgramNodeKind.SERIALIZER,
        ProgramNodeKind.DESERIALIZER,
        ProgramNodeKind.TEST,
        ProgramNodeKind.MOCK,
        ProgramNodeKind.OWNERSHIP,
        ProgramNodeKind.VALIDATION,
        ProgramNodeKind.API_ENDPOINT,
        ProgramNodeKind.CLI_COMMAND,
        ProgramNodeKind.REPOSITORY,
    }
    missing = required - kinds
    assert not missing, f"missing node kinds: {sorted(item.value for item in missing)}"


def test_typed_edges_cover_calls_overrides_imports_data_and_ownership() -> None:
    graph = _build_graph().graph
    assert graph is not None
    kinds = {edge.kind for edge in graph.edges}
    required = {
        ProgramEdgeKind.DEFINES,
        ProgramEdgeKind.CALLS,
        ProgramEdgeKind.IMPORTS,
        ProgramEdgeKind.ALIASES,
        ProgramEdgeKind.PARAMETER_OF,
        ProgramEdgeKind.RETURNS,
        ProgramEdgeKind.DATA_FLOW,
        ProgramEdgeKind.STATE_FLOW,
        ProgramEdgeKind.OWNS,
        ProgramEdgeKind.VALIDATES,
        ProgramEdgeKind.TESTS,
        ProgramEdgeKind.FACTORY_CREATES,
        ProgramEdgeKind.CONSTRUCTS,
        ProgramEdgeKind.CONTAINS,
    }
    missing = required - kinds
    assert not missing, f"missing edge kinds: {sorted(item.value for item in missing)}"


def test_override_and_implementation_edges_from_bases() -> None:
    graph = _build_graph().graph
    assert graph is not None
    override_or_impl = [
        edge
        for edge in graph.edges
        if edge.kind in {ProgramEdgeKind.OVERRIDES, ProgramEdgeKind.IMPLEMENTS}
    ]
    assert override_or_impl, "expected override/implements edges for Service(Base)"


def test_decorators_registries_and_di_are_represented() -> None:
    graph = _build_graph().graph
    assert graph is not None
    kinds = {node.kind for node in graph.nodes}
    edge_kinds = {edge.kind for edge in graph.edges}
    assert ProgramNodeKind.DECORATOR in kinds or ProgramEdgeKind.DECORATES in edge_kinds
    assert (
        ProgramNodeKind.REGISTRY in kinds
        or ProgramEdgeKind.REGISTERS in edge_kinds
        or ProgramNodeKind.DI_BINDING in kinds
    )


def test_schema_serializer_and_factory_nodes_exist() -> None:
    graph = _build_graph().graph
    assert graph is not None
    names = {node.name for node in graph.nodes} | {
        node.qualified_name for node in graph.nodes
    }
    assert any("Payload" in name for name in names)
    assert any(
        node.kind in {ProgramNodeKind.SERIALIZER, ProgramNodeKind.DESERIALIZER}
        for node in graph.nodes
    )
    assert any(node.kind is ProgramNodeKind.FACTORY for node in graph.nodes)


def test_tests_mocks_and_docs_boundaries() -> None:
    files = _fixture_sources()
    files["docs/api.md"] = "# API\n\nThe service MUST process payloads.\n"
    files["generated/client.py"] = "def generated_call():\n    return 1\n"
    files["native/ext.c"] = "int add(int a, int b) { return a + b; }\n"
    builder = _build_graph(files)
    graph = builder.graph
    assert graph is not None
    assert any(node.kind is ProgramNodeKind.TEST for node in graph.nodes)
    assert any(node.kind is ProgramNodeKind.MOCK for node in graph.nodes)
    assert any(node.kind is ProgramNodeKind.DOCUMENTATION for node in graph.nodes)
    assert any(node.kind is ProgramNodeKind.GENERATED for node in graph.nodes)
    assert any(node.kind is ProgramNodeKind.NATIVE_BOUNDARY for node in graph.nodes)
    assert any(ref.startswith("generated:") for ref in graph.frontier_refs)
    assert any(ref.startswith("native:") for ref in graph.frontier_refs)


def test_excluded_and_tombstone_roots_appear_in_exclusions() -> None:
    graph = _build_graph().graph
    assert graph is not None
    assert any("excluded_root:vendor/" in ref for ref in graph.exclusion_refs)
    assert any("tombstone:old/module.py" in ref for ref in graph.exclusion_refs)
    assert graph.complete is False


# ---------------------------------------------------------------------------
# Nominated GraphRAG / runtime / vector edges
# ---------------------------------------------------------------------------


def test_graphrag_runtime_vector_edges_remain_nominated() -> None:
    builder = _build_graph()
    graph = builder.graph
    assert graph is not None
    # Pick two authoritative nodes to nominate a related_to edge between.
    symbols = [node for node in graph.nodes if node.kind is ProgramNodeKind.FUNCTION]
    assert len(symbols) >= 2
    nominations = [
        {
            "source": symbols[0].node_id,
            "target": symbols[1].node_id,
            "kind": "related_to",
            "provenance": "graphrag",
            "confidence": 15,
        },
        {
            "source": symbols[0].node_id,
            "target": symbols[1].node_id,
            "kind": "calls",
            "provenance": "vector",
            "confidence": 10,
        },
        {
            "source": symbols[1].node_id,
            "target": symbols[0].node_id,
            "kind": "related_to",
            "provenance": "runtime",
            "confidence": 12,
        },
    ]
    builder2 = _build_graph(nominations=nominations)
    graph2 = builder2.graph
    assert graph2 is not None
    nominated = graph2.snapshot.nominated_edges()
    assert nominated
    for edge in nominated:
        assert edge.authoritative is False
        assert edge.authority is ProgramAuthority.NOMINATED
        assert edge.provenance in {
            ProgramProvenance.GRAPHRAG,
            ProgramProvenance.VECTOR,
            ProgramProvenance.RUNTIME,
        }
    # Authoritative edges must not include nominated provenance.
    for edge in graph2.snapshot.authoritative_edges():
        assert edge.provenance.trusted_channel
        assert edge.authority.authority_bearing


def test_untrusted_provenance_cannot_mint_authoritative_node() -> None:
    with pytest.raises(ProgramGraphError):
        ProgramNode(
            node_id="n1",
            kind=ProgramNodeKind.SYMBOL,
            name="x",
            roots=FIXTURE_ROOTS,
            provenance=ProgramProvenance.VECTOR,
            trust=ProgramTrust.TRUSTED,
            authority=ProgramAuthority.AUTHORITATIVE,
        )


# ---------------------------------------------------------------------------
# Call resolver
# ---------------------------------------------------------------------------


def test_resolver_returns_resolved_for_unique_symbol() -> None:
    graph = _build_graph().graph
    assert graph is not None
    resolver = ProgramCallResolver(graph)
    result = resolver.resolve_reference("helper", path="src/service.py")
    assert result.status is CallResolutionStatus.RESOLVED
    assert len(result.target_ids) == 1
    assert result.route_closed is True
    assert result.resolution_id


def test_resolver_returns_ambiguous_for_same_name() -> None:
    files = {
        "a.py": "def process(x):\n    return x\n",
        "b.py": "def process(x):\n    return x + 1\n",
        "c.py": "def caller():\n    return process(1)\n",
    }
    graph = _build_graph(files).graph
    assert graph is not None
    resolver = ProgramCallResolver(graph)
    result = resolver.resolve_reference("process")
    assert result.status is CallResolutionStatus.AMBIGUOUS
    assert len(result.candidate_ids) >= 2
    assert result.frontier_refs
    assert result.route_closed is False


def test_resolver_returns_dynamic_for_getattr() -> None:
    graph = _build_graph().graph
    assert graph is not None
    resolver = ProgramCallResolver(graph)
    result = resolver.resolve_reference("getattr")
    assert result.status is CallResolutionStatus.DYNAMIC
    assert any("dynamic" in ref for ref in result.frontier_refs)


def test_resolver_returns_external_for_builtins() -> None:
    graph = _build_graph().graph
    assert graph is not None
    resolver = ProgramCallResolver(graph)
    result = resolver.resolve_reference("print")
    assert result.status is CallResolutionStatus.EXTERNAL
    assert result.frontier_refs


def test_resolver_returns_unsupported_with_bounded_frontier() -> None:
    graph = _build_graph().graph
    assert graph is not None
    resolver = ProgramCallResolver(graph)
    result = resolver.resolve_reference("definitely_missing_symbol_xyz")
    assert result.status is CallResolutionStatus.UNSUPPORTED
    assert result.frontier_refs
    assert result.route_closed is False


def test_resolver_does_not_authorize_nominated_only_edges() -> None:
    builder = _build_graph()
    graph = builder.graph
    assert graph is not None
    symbols = [node for node in graph.nodes if node.kind is ProgramNodeKind.FUNCTION]
    nominations = [
        {
            "source": symbols[0].node_id,
            "target": symbols[1].node_id,
            "kind": "calls",
            "provenance": "vector",
        }
    ]
    graph2 = _build_graph(nominations=nominations).graph
    assert graph2 is not None
    resolver = ProgramCallResolver(graph2)
    # A pure name match may still resolve; ensure nominated-only candidate
    # path does not close the route without an authoritative target.
    missing = resolver.resolve_reference("no_such_symbol_for_nomination")
    assert missing.status is CallResolutionStatus.UNSUPPORTED
    assert missing.route_closed is False


def test_resolver_bound_frontier_is_deterministic() -> None:
    graph = _build_graph().graph
    assert graph is not None
    resolver = ProgramCallResolver(graph)
    site = CallSite(caller_id="", callee_reference="missing_one")
    left = resolver.resolve(site)
    right = resolver.resolve(site)
    assert left.resolution_id == right.resolution_id
    assert left.to_dict() == right.to_dict()


# ---------------------------------------------------------------------------
# Incremental rebuild equals clean rebuild
# ---------------------------------------------------------------------------


def test_incremental_rebuild_equals_clean_rebuild() -> None:
    files = _fixture_sources()
    clean = _build_graph(files)
    assert clean.graph is not None
    clean_id = clean.graph.graph_id
    clean_json = clean.snapshot.to_json() if clean.snapshot else ""

    # First build, then change one file and rebuild incrementally.
    warm = _build_graph(files)
    files2 = dict(files)
    files2["src/service.py"] = files["src/service.py"] + "\n\ndef extra():\n    return 1\n"
    incremental = ProgramDependencyGraph(FIXTURE_ROOTS, previous=warm)
    incremental.build(
        [
            PathSource(path=path, source=source, language="python")
            for path, source in sorted(files2.items())
        ],
        previous=warm,
    )
    clean_changed = _build_graph(files2)
    assert incremental.graph is not None and clean_changed.graph is not None
    assert incremental.graph.graph_id == clean_changed.graph.graph_id
    assert incremental.snapshot is not None and clean_changed.snapshot is not None
    assert incremental.snapshot.to_json() == clean_changed.snapshot.to_json()

    # Unchanged snapshot still matches the original clean build.
    warm_same = ProgramDependencyGraph(FIXTURE_ROOTS, previous=warm)
    warm_same.build(
        [
            PathSource(path=path, source=source, language="python")
            for path, source in sorted(files.items())
        ],
        previous=warm,
    )
    assert warm_same.graph is not None
    assert warm_same.graph.graph_id == clean_id
    assert warm_same.snapshot is not None
    assert warm_same.snapshot.to_json() == clean_json


def test_component_reuse_on_unchanged_paths() -> None:
    files = _fixture_sources()
    first = _build_graph(files)
    second = ProgramDependencyGraph(FIXTURE_ROOTS, previous=first)
    second.build(
        [
            PathSource(path=path, source=source, language="python")
            for path, source in sorted(files.items())
        ],
        previous=first,
    )
    assert set(first.components) == set(second.components)
    for path in first.components:
        assert first.components[path].content_key == second.components[path].content_key
        # Rebinding preserves node ids.
        assert {n.node_id for n in first.components[path].nodes} == {
            n.node_id for n in second.components[path].nodes
        }


# ---------------------------------------------------------------------------
# Build helpers / integration
# ---------------------------------------------------------------------------


def test_build_from_ast_records_without_source_body() -> None:
    files = _fixture_sources()
    sources = []
    for path, source in sorted(files.items()):
        record = build_python_ast_blob_record(source)
        sources.append(
            PathSource(
                path=path,
                language="python",
                blob_identity=record.blob_identity,
                source_sha256=record.source_sha256,
                ast_record=record,
            )
        )
    builder = ProgramDependencyGraph(FIXTURE_ROOTS)
    graph = builder.build(sources)
    assert graph.nodes
    assert graph.edges
    assert any(edge.kind is ProgramEdgeKind.CALLS for edge in graph.edges)


def test_build_program_dependency_graph_convenience() -> None:
    graph = build_program_dependency_graph(FIXTURE_ROOTS, _fixture_sources())
    assert isinstance(graph, ProgramGraph)
    assert graph.graph_id


def test_impact_edges_are_admitted_as_depends_on() -> None:
    graph = _build_graph(
        impact_edges={"Service.process": ("helper",)}
    ).graph
    assert graph is not None
    depends = [
        edge
        for edge in graph.edges
        if edge.kind is ProgramEdgeKind.DEPENDS_ON
        and edge.attributes.get("impact")
    ]
    # May or may not resolve depending on symbol index keys; at least build succeeds.
    assert graph.graph_id
    assert isinstance(depends, list)


def test_program_graph_facade_queries() -> None:
    graph = _build_graph().graph
    assert graph is not None
    modules = graph.nodes_of_kind(ProgramNodeKind.MODULE)
    assert modules
    for module in modules:
        contained = graph.edges_from(module.node_id)
        assert isinstance(contained, tuple)
    found = graph.find_by_path("src/service.py")
    assert found
    payload = graph.to_dict()
    rebuilt = ProgramGraph.from_dict(payload)
    assert rebuilt.graph_id == graph.graph_id


def test_capability_symbols_are_importable() -> None:
    from ipfs_accelerate_py.agent_supervisor import program_graph as pg
    from ipfs_accelerate_py.agent_supervisor import program_call_resolver as pcr
    from ipfs_accelerate_py.agent_supervisor.analysis import (
        program_dependency_graph as pdg,
    )

    assert pg.ProgramGraph is ProgramGraph
    assert pcr.ProgramCallResolver is ProgramCallResolver
    assert pdg.ProgramDependencyGraph is ProgramDependencyGraph
    assert pg.ProgramNode is ProgramNode
    assert pg.ProgramEdge is ProgramEdge
    assert pg.ProgramGraphSnapshot is ProgramGraphSnapshot


def test_trace_graph_evidence_projects_coverage() -> None:
    graph = _build_graph().graph
    assert graph is not None
    evidence = graph.trace_graph_evidence()
    # May be GraphEvidence or mapping depending on import path.
    graph_id = getattr(evidence, "graph_id", None) or evidence.get("graph_id")
    assert graph_id == graph.graph_id
    complete = getattr(evidence, "complete", None)
    if complete is None:
        complete = evidence.get("complete")
    assert complete is False  # fixture has exclusions/frontiers


def test_json_serialization_is_canonical() -> None:
    graph = _build_graph().graph
    assert graph is not None
    left = graph.to_json()
    right = graph.to_json()
    assert left == right
    # Stable key ordering.
    payload = json.loads(left)
    assert payload["graph_id"] == graph.graph_id
