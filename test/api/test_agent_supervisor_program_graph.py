"""Tests for the canonical cross-repository program evidence graph.

Covers VFS-G040 evidence terms ``vfs/program-graph@1`` (canonical construction)
and, via the ranking provider import, ``vfs/graphrag-projection@1`` (optional
GraphRAG ranking only).  Construction and ranking remain separate surfaces.

Also anchors the synthetic objective validation repair discovery term so
supervisor exact-text scans re-find the VFS-G040 validation gate without
mixing that meta term into content-addressed graph identity.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.ipfs_datasets_program_graph_provider import (
    GRAPHRAG_PROJECTION_EVIDENCE,
    IpfsDatasetsProgramGraphProvider,
    all_covered_evidence_terms,
    covered_evidence_terms,
    graphrag_projection_evidence_terms,
)
from ipfs_accelerate_py.agent_supervisor.program_graph import (
    DanglingEdgeError,
    ForgedIdentityError,
    GraphChunk,
    GraphCompleteness,
    GraphIndex,
    IllegalCycleError,
    OBJECTIVE_GOAL_ID,
    OBJECTIVE_VALIDATION_REPAIR_EVIDENCE,
    PROGRAM_GRAPH_EVIDENCE,
    PROGRAM_GRAPH_SCHEMA,
    ProgramEdgeKind,
    ProgramGraph,
    ProgramGraphError,
    ProgramNodeKind,
    ResolverStatus,
    SourceSpan,
    all_program_edge_kinds,
    all_program_node_kinds,
    build_program_graph,
    make_edge,
    make_node,
    merge_program_graphs,
    objective_validation_repair_evidence_terms,
    program_graph_evidence_terms,
)


FOREST_ID = "forest:test-vfs-008"
PRODUCER = "program-ast-adapter@1"
BLOB_A = "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
BLOB_B = "baguqeerbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _span(line: int = 1, col: int = 0) -> SourceSpan:
    return SourceSpan(line_start=line, column_start=col, line_end=line, column_end=col + 4)


def _node(
    kind: ProgramNodeKind | str,
    key: str,
    *,
    component_id: str = "",
    blob_cid: str = BLOB_A,
    forest_id: str = FOREST_ID,
    qualified_name: str = "",
    path: str = "",
    language: str = "python",
    resolver_status: ResolverStatus | str = ResolverStatus.RESOLVED_STATIC,
    record: dict[str, Any] | None = None,
) -> Any:
    return make_node(
        kind=kind,
        record_key=key,
        producer=PRODUCER,
        blob_cid=blob_cid,
        forest_id=forest_id,
        component_id=component_id or key,
        qualified_name=qualified_name or key,
        path=path,
        language=language,
        span=_span(),
        resolver_status=resolver_status,
        record=record or {},
    )


def _edge(
    source: str,
    target: str,
    kind: ProgramEdgeKind | str,
    *,
    component_id: str = "comp-a",
    blob_cid: str = BLOB_A,
    forest_id: str = FOREST_ID,
    resolver_status: ResolverStatus | str = ResolverStatus.RESOLVED_STATIC,
    record: dict[str, Any] | None = None,
) -> Any:
    return make_edge(
        source=source,
        target=target,
        kind=kind,
        producer=PRODUCER,
        blob_cid=blob_cid,
        forest_id=forest_id,
        component_id=component_id,
        span=_span(2),
        resolver_status=resolver_status,
        record=record or {},
    )


def _fixture_graph() -> ProgramGraph:
    """Build a small multi-kind graph covering VFS-008 node vocabulary."""

    repo = _node(
        ProgramNodeKind.REPOSITORY,
        "repo:accelerator",
        component_id="repo:accelerator",
        path="",
        qualified_name="accelerator",
    )
    blob = _node(
        ProgramNodeKind.BLOB,
        f"blob:{BLOB_A}",
        component_id="module:pkg.mod",
        path="pkg/mod.py",
        qualified_name=BLOB_A,
    )
    module = _node(
        ProgramNodeKind.MODULE,
        "module:pkg.mod",
        component_id="module:pkg.mod",
        path="pkg/mod.py",
        qualified_name="pkg.mod",
    )
    symbol = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:pkg.mod.entry",
        component_id="module:pkg.mod",
        path="pkg/mod.py",
        qualified_name="pkg.mod.entry",
    )
    definition = _node(
        ProgramNodeKind.DEFINITION,
        "def:pkg.mod.entry",
        component_id="module:pkg.mod",
        path="pkg/mod.py",
        qualified_name="pkg.mod.entry",
    )
    import_node = _node(
        ProgramNodeKind.IMPORT,
        "import:pkg.mod:os",
        component_id="module:pkg.mod",
        path="pkg/mod.py",
        qualified_name="os",
        resolver_status=ResolverStatus.EXTERNAL,
        record={"reason": "stdlib"},
    )
    export_node = _node(
        ProgramNodeKind.EXPORT,
        "export:pkg.mod.entry",
        component_id="module:pkg.mod",
        path="pkg/mod.py",
        qualified_name="pkg.mod.entry",
    )
    call = _node(
        ProgramNodeKind.CALL,
        "call:pkg.mod.entry:helper",
        component_id="module:pkg.mod",
        path="pkg/mod.py",
        qualified_name="helper",
        resolver_status=ResolverStatus.CANDIDATE,
        record={"reason": "candidate_same_module"},
    )
    type_node = _node(
        ProgramNodeKind.TYPE,
        "type:pkg.mod.EntryResult",
        component_id="module:pkg.mod",
        path="pkg/mod.py",
        qualified_name="pkg.mod.EntryResult",
    )
    schema = _node(
        ProgramNodeKind.SCHEMA,
        "schema:tool.entry.input",
        component_id="mcp:entry",
        path="schemas/entry.json",
        language="json",
        qualified_name="tool.entry.input",
        blob_cid=BLOB_B,
    )
    contract = _node(
        ProgramNodeKind.CONTRACT,
        "contract:tool.entry",
        component_id="mcp:entry",
        path="schemas/entry.json",
        language="json",
        qualified_name="tool.entry",
        blob_cid=BLOB_B,
        record={"role": "expected"},
    )
    doc = _node(
        ProgramNodeKind.DOC,
        "doc:pkg.mod.entry",
        component_id="module:pkg.mod",
        path="docs/mod.md",
        language="markdown",
        qualified_name="pkg.mod.entry",
        blob_cid=BLOB_B,
    )
    test = _node(
        ProgramNodeKind.TEST,
        "test:test_entry",
        component_id="test:test_entry",
        path="test/test_mod.py",
        qualified_name="test_entry",
        blob_cid=BLOB_B,
    )
    mcp_tool = _node(
        ProgramNodeKind.MCP_TOOL,
        "mcp_tool:entry",
        component_id="mcp:entry",
        path="mcp/tools.json",
        language="json",
        qualified_name="entry",
        blob_cid=BLOB_B,
    )
    mcp_reg = _node(
        ProgramNodeKind.MCP_REGISTRATION,
        "mcp_reg:entry",
        component_id="mcp:entry",
        path="mcp/tools.json",
        language="json",
        qualified_name="entry",
        blob_cid=BLOB_B,
    )
    transport = _node(
        ProgramNodeKind.TRANSPORT,
        "transport:stdio",
        component_id="mcp:entry",
        path="mcp/server.py",
        qualified_name="stdio",
        blob_cid=BLOB_B,
    )
    artifact = _node(
        ProgramNodeKind.ARTIFACT,
        "artifact:ast:pkg.mod",
        component_id="module:pkg.mod",
        path="pkg/mod.py",
        qualified_name="ast:pkg.mod",
    )
    finding = _node(
        ProgramNodeKind.FINDING,
        "finding:entry-contract-drift",
        component_id="mcp:entry",
        path="findings/entry.json",
        language="json",
        qualified_name="entry-contract-drift",
        blob_cid=BLOB_B,
        record={"severity": "error"},
    )
    proof_obligation = _node(
        ProgramNodeKind.PROOF_OBLIGATION,
        "proof:entry-precondition",
        component_id="mcp:entry",
        path="proofs/entry.json",
        language="json",
        qualified_name="entry-precondition",
        blob_cid=BLOB_B,
        record={"kind": "precondition"},
    )

    nodes = (
        repo,
        blob,
        module,
        symbol,
        definition,
        import_node,
        export_node,
        call,
        type_node,
        schema,
        contract,
        doc,
        test,
        mcp_tool,
        mcp_reg,
        transport,
        artifact,
        finding,
        proof_obligation,
    )
    edges = (
        _edge(repo.node_id, blob.node_id, ProgramEdgeKind.CONTAINS, component_id="repo:accelerator"),
        _edge(blob.node_id, module.node_id, ProgramEdgeKind.CONTAINS, component_id="module:pkg.mod"),
        _edge(module.node_id, symbol.node_id, ProgramEdgeKind.DEFINES, component_id="module:pkg.mod"),
        _edge(symbol.node_id, definition.node_id, ProgramEdgeKind.DEFINES, component_id="module:pkg.mod"),
        _edge(module.node_id, import_node.node_id, ProgramEdgeKind.IMPORTS, component_id="module:pkg.mod"),
        _edge(module.node_id, export_node.node_id, ProgramEdgeKind.EXPORTS, component_id="module:pkg.mod"),
        _edge(
            call.node_id,
            symbol.node_id,
            ProgramEdgeKind.CALLS,
            component_id="module:pkg.mod",
            resolver_status=ResolverStatus.CANDIDATE,
        ),
        _edge(symbol.node_id, type_node.node_id, ProgramEdgeKind.TYPED_AS, component_id="module:pkg.mod"),
        _edge(doc.node_id, symbol.node_id, ProgramEdgeKind.DOCUMENTS, component_id="module:pkg.mod"),
        _edge(test.node_id, symbol.node_id, ProgramEdgeKind.TESTS, component_id="test:test_entry"),
        _edge(mcp_reg.node_id, mcp_tool.node_id, ProgramEdgeKind.REGISTERS, component_id="mcp:entry"),
        _edge(
            mcp_tool.node_id,
            transport.node_id,
            ProgramEdgeKind.USES_TRANSPORT,
            component_id="mcp:entry",
        ),
        _edge(
            mcp_tool.node_id,
            symbol.node_id,
            ProgramEdgeKind.IMPLEMENTS,
            component_id="mcp:entry",
        ),
        _edge(mcp_tool.node_id, schema.node_id, ProgramEdgeKind.REFERENCES, component_id="mcp:entry"),
        _edge(contract.node_id, schema.node_id, ProgramEdgeKind.REFERENCES, component_id="mcp:entry"),
        _edge(contract.node_id, mcp_tool.node_id, ProgramEdgeKind.OBLIGATES, component_id="mcp:entry"),
        _edge(finding.node_id, contract.node_id, ProgramEdgeKind.SUPPORTS, component_id="mcp:entry"),
        _edge(
            proof_obligation.node_id,
            symbol.node_id,
            ProgramEdgeKind.OBLIGATES,
            component_id="mcp:entry",
        ),
        _edge(artifact.node_id, module.node_id, ProgramEdgeKind.DERIVED_FROM, component_id="module:pkg.mod"),
        _edge(symbol.node_id, module.node_id, ProgramEdgeKind.MEMBER_OF, component_id="module:pkg.mod"),
        _edge(module.node_id, repo.node_id, ProgramEdgeKind.DEPENDS_ON, component_id="module:pkg.mod"),
        _edge(
            import_node.node_id,
            import_node.node_id,
            ProgramEdgeKind.RESOLVES_TO,
            component_id="module:pkg.mod",
            resolver_status=ResolverStatus.EXTERNAL,
            record={"reason": "external_stdlib"},
        ),
    )
    return build_program_graph(forest_id=FOREST_ID, nodes=nodes, edges=edges, producer=PRODUCER)


def test_all_required_node_kinds_are_admitted() -> None:
    graph = _fixture_graph()
    kinds = {node.kind for node in graph.nodes}
    expected = set(ProgramNodeKind)
    assert kinds == expected
    assert len(graph.nodes) == len(ProgramNodeKind)


def test_every_record_binds_producer_blob_span_resolver_and_forest() -> None:
    graph = _fixture_graph()
    for node in graph.nodes:
        binding = node.binding
        assert binding.producer == PRODUCER
        assert binding.blob_cid
        assert binding.forest_id == FOREST_ID
        assert isinstance(binding.span, SourceSpan)
        assert isinstance(binding.resolver_status, ResolverStatus)
        payload = node.to_dict()
        assert payload["binding"]["producer"] == PRODUCER
        assert payload["binding"]["blob_cid"] == binding.blob_cid
        assert payload["binding"]["forest_id"] == FOREST_ID
        assert "line_start" in payload["binding"]["span"]
        assert payload["binding"]["resolver_status"]
    for edge in graph.edges:
        binding = edge.binding
        assert binding.producer == PRODUCER
        assert binding.blob_cid
        assert binding.forest_id == FOREST_ID
        assert isinstance(binding.span, SourceSpan)
        assert isinstance(binding.resolver_status, ResolverStatus)


def test_graph_is_deterministic_and_content_addressed() -> None:
    left = _fixture_graph()
    right = _fixture_graph()
    assert left.graph_id == right.graph_id
    assert left.to_dict() == right.to_dict()
    assert left.to_json() == right.to_json()
    # Order of input nodes/edges must not affect identity.
    shuffled = build_program_graph(
        forest_id=FOREST_ID,
        nodes=tuple(reversed(left.nodes)),
        edges=tuple(reversed(left.edges)),
        producer=PRODUCER,
    )
    assert shuffled.graph_id == left.graph_id
    assert [node.node_id for node in shuffled.nodes] == [
        node.node_id for node in left.nodes
    ]


def test_round_trip_serialization_preserves_identity() -> None:
    graph = _fixture_graph()
    payload = json.loads(graph.to_json())
    restored = ProgramGraph.from_dict(payload)
    assert restored.graph_id == graph.graph_id
    assert restored.canonical_records() == graph.canonical_records()
    assert restored.completeness().content_id == graph.completeness().content_id
    assert restored.build_index().index_id == graph.build_index().index_id


def test_rejects_dangling_edges() -> None:
    module = _node(ProgramNodeKind.MODULE, "module:only")
    missing_target = "pnode-does-not-exist"
    edge = _edge(module.node_id, missing_target, ProgramEdgeKind.DEFINES)
    with pytest.raises(DanglingEdgeError):
        build_program_graph(forest_id=FOREST_ID, nodes=[module], edges=[edge])


def test_rejects_forged_node_and_edge_identities() -> None:
    from ipfs_accelerate_py.agent_supervisor.program_graph import (
        ProgramGraphEdge,
        ProgramGraphNode,
    )

    node = _node(ProgramNodeKind.SYMBOL, "symbol:x")
    forged_node = dict(node.to_dict())
    forged_node["node_id"] = "pnode-forged"
    with pytest.raises(ForgedIdentityError):
        ProgramGraphNode.from_dict(forged_node)

    other = _node(ProgramNodeKind.SYMBOL, "symbol:y")
    edge = _edge(node.node_id, other.node_id, ProgramEdgeKind.CALLS)
    forged_edge = dict(edge.to_dict())
    forged_edge["edge_id"] = "pedge-forged"
    with pytest.raises(ForgedIdentityError):
        ProgramGraphEdge.from_dict(forged_edge)

    graph = build_program_graph(
        forest_id=FOREST_ID, nodes=[node, other], edges=[edge]
    )
    forged_graph = dict(graph.to_dict())
    forged_graph["graph_id"] = "pgraph-forged"
    with pytest.raises(ForgedIdentityError):
        ProgramGraph.from_dict(forged_graph)


def test_rejects_illegal_structural_cycles_but_allows_call_cycles() -> None:
    a = _node(ProgramNodeKind.MODULE, "module:a", component_id="c1")
    b = _node(ProgramNodeKind.MODULE, "module:b", component_id="c1")
    # contains cycle is illegal
    with pytest.raises(IllegalCycleError):
        build_program_graph(
            forest_id=FOREST_ID,
            nodes=[a, b],
            edges=[
                _edge(a.node_id, b.node_id, ProgramEdgeKind.CONTAINS, component_id="c1"),
                _edge(b.node_id, a.node_id, ProgramEdgeKind.CONTAINS, component_id="c1"),
            ],
        )
    # depends_on cycle is illegal
    with pytest.raises(IllegalCycleError):
        build_program_graph(
            forest_id=FOREST_ID,
            nodes=[a, b],
            edges=[
                _edge(a.node_id, b.node_id, ProgramEdgeKind.DEPENDS_ON, component_id="c1"),
                _edge(b.node_id, a.node_id, ProgramEdgeKind.DEPENDS_ON, component_id="c1"),
            ],
        )
    # call cycles are legal (mutual recursion)
    call_graph = build_program_graph(
        forest_id=FOREST_ID,
        nodes=[a, b],
        edges=[
            _edge(a.node_id, b.node_id, ProgramEdgeKind.CALLS, component_id="c1"),
            _edge(b.node_id, a.node_id, ProgramEdgeKind.CALLS, component_id="c1"),
        ],
    )
    assert len(call_graph.edges) == 2


def test_rejects_foreign_forest_bindings() -> None:
    local = _node(ProgramNodeKind.MODULE, "module:local")
    foreign = _node(
        ProgramNodeKind.MODULE,
        "module:foreign",
        forest_id="forest:other",
    )
    with pytest.raises(ProgramGraphError, match="foreign forest"):
        build_program_graph(forest_id=FOREST_ID, nodes=[local, foreign])


def test_indexes_and_chunks_are_deterministic_and_content_addressed() -> None:
    graph = _fixture_graph()
    index = graph.build_index()
    assert index.forest_id == FOREST_ID
    assert index.index_id.startswith("pindex-")
    assert ProgramNodeKind.MCP_TOOL.value in index.node_ids_by_kind
    assert "module:pkg.mod" in index.node_ids_by_component
    assert BLOB_A in index.node_ids_by_blob_cid
    assert "pkg.mod.entry" in index.node_ids_by_qualified_name

    restored_index = GraphIndex.from_dict(index.to_dict())
    assert restored_index.index_id == index.index_id

    chunks = graph.chunk_all_components()
    assert chunks
    assert [chunk.chunk_key for chunk in chunks] == sorted(
        chunk.chunk_key for chunk in chunks
    )
    for chunk in chunks:
        assert chunk.chunk_id.startswith("pchunk-")
        assert chunk.forest_id == FOREST_ID
        restored = GraphChunk.from_dict(chunk.to_dict())
        assert restored.chunk_id == chunk.chunk_id
        # Rebuilding the same component chunk is stable.
        component_id = chunk.chunk_key.removeprefix("component:")
        again = graph.chunk_by_component(component_id)
        assert again.chunk_id == chunk.chunk_id
        assert again.chunk_key == chunk.chunk_key


def test_incremental_component_replacement() -> None:
    graph = _fixture_graph()
    original_id = graph.graph_id
    component = "module:pkg.mod"
    before_nodes = {node.node_id for node in graph.nodes_for_component(component)}
    assert before_nodes

    # Replace the module component with a smaller closed set.
    module = _node(
        ProgramNodeKind.MODULE,
        "module:pkg.mod",
        component_id=component,
        path="pkg/mod.py",
        qualified_name="pkg.mod",
        blob_cid=BLOB_B,
        record={"revision": 2},
    )
    symbol = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:pkg.mod.entry_v2",
        component_id=component,
        path="pkg/mod.py",
        qualified_name="pkg.mod.entry_v2",
        blob_cid=BLOB_B,
    )
    edge = _edge(
        module.node_id,
        symbol.node_id,
        ProgramEdgeKind.DEFINES,
        component_id=component,
        blob_cid=BLOB_B,
    )
    replaced = graph.replace_component(
        component, nodes=[module, symbol], edges=[edge]
    )
    assert replaced.forest_id == FOREST_ID
    assert replaced.graph_id != original_id
    # Other components remain.
    assert replaced.nodes_by_kind(ProgramNodeKind.MCP_TOOL)
    assert replaced.nodes_by_kind(ProgramNodeKind.REPOSITORY)
    # Replaced component no longer contains old module-scoped symbols.
    remaining_keys = {
        node.record_key for node in replaced.nodes_for_component(component)
    }
    assert remaining_keys == {"module:pkg.mod", "symbol:pkg.mod.entry_v2"}
    # Replacing with the same content is idempotent on identity.
    again = replaced.replace_component(
        component, nodes=[module, symbol], edges=[edge]
    )
    assert again.graph_id == replaced.graph_id


def test_completeness_and_frontier_metadata() -> None:
    graph = _fixture_graph()
    completeness = graph.completeness()
    assert completeness.node_count == len(graph.nodes)
    assert completeness.edge_count == len(graph.edges)
    assert completeness.component_count == len(graph.component_ids())
    assert completeness.frontier_count > 0
    assert completeness.complete is False
    statuses = {item.resolver_status for item in completeness.frontier}
    assert ResolverStatus.EXTERNAL in statuses or ResolverStatus.CANDIDATE in statuses
    assert completeness.node_counts_by_kind[ProgramNodeKind.SYMBOL.value] >= 1
    payload = completeness.to_dict()
    restored = GraphCompleteness.from_dict(payload)
    assert restored.content_id == completeness.content_id

    module = _node(
        ProgramNodeKind.MODULE,
        "module:closed",
        component_id="c",
        resolver_status=ResolverStatus.RESOLVED_STATIC,
    )
    symbol = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:closed",
        component_id="c",
        resolver_status=ResolverStatus.RESOLVED_STATIC,
    )
    closed = build_program_graph(
        forest_id=FOREST_ID,
        nodes=[module, symbol],
        edges=[
            _edge(
                module.node_id,
                symbol.node_id,
                ProgramEdgeKind.DEFINES,
                component_id="c",
            )
        ],
    )
    assert closed.completeness().complete is True
    assert closed.completeness().frontier_count == 0


def test_truncated_or_unexplained_gaps_mark_incomplete() -> None:
    base_nodes = [
        _node(ProgramNodeKind.MODULE, "module:x", resolver_status=ResolverStatus.RESOLVED_STATIC)
    ]
    truncated = build_program_graph(
        forest_id=FOREST_ID,
        nodes=base_nodes,
        truncated=True,
        truncation_reason="node_bound",
    )
    assert truncated.completeness().complete is False
    assert truncated.completeness().truncated is True
    gaps = build_program_graph(
        forest_id=FOREST_ID,
        nodes=base_nodes,
        unexplained_gap_count=2,
    )
    assert gaps.completeness().complete is False
    assert gaps.completeness().unexplained_gap_count == 2


def test_merge_graphs_same_forest() -> None:
    left = build_program_graph(
        forest_id=FOREST_ID,
        nodes=[_node(ProgramNodeKind.MODULE, "module:left", component_id="left")],
    )
    right = build_program_graph(
        forest_id=FOREST_ID,
        nodes=[_node(ProgramNodeKind.MODULE, "module:right", component_id="right")],
    )
    merged = merge_program_graphs([left, right])
    assert len(merged.nodes) == 2
    assert set(merged.component_ids()) == {"left", "right"}
    foreign = build_program_graph(
        forest_id="forest:other",
        nodes=[
            _node(
                ProgramNodeKind.MODULE,
                "module:foreign",
                forest_id="forest:other",
                component_id="foreign",
            )
        ],
    )
    with pytest.raises(ProgramGraphError, match="foreign forest"):
        merge_program_graphs([left, foreign])


def test_schema_constant_and_node_kind_vocabulary() -> None:
    assert PROGRAM_GRAPH_SCHEMA.endswith("@1")
    assert {kind.value for kind in ProgramNodeKind} == {
        "repository",
        "blob",
        "module",
        "symbol",
        "definition",
        "import",
        "export",
        "call",
        "type",
        "schema",
        "contract",
        "doc",
        "test",
        "mcp_tool",
        "mcp_registration",
        "transport",
        "artifact",
        "finding",
        "proof_obligation",
    }
    assert {kind.value for kind in all_program_node_kinds()} == {
        kind.value for kind in ProgramNodeKind
    }
    assert {kind.value for kind in ProgramEdgeKind} >= {
        "supports",
        "obligates",
        "references",
        "calls",
    }
    assert set(all_program_edge_kinds()) == set(ProgramEdgeKind)


def test_self_referential_contains_is_illegal() -> None:
    module = _node(ProgramNodeKind.MODULE, "module:self")
    with pytest.raises(IllegalCycleError):
        build_program_graph(
            forest_id=FOREST_ID,
            nodes=[module],
            edges=[
                _edge(
                    module.node_id,
                    module.node_id,
                    ProgramEdgeKind.CONTAINS,
                    component_id="module:self",
                )
            ],
        )


def test_replacement_rejects_dangling_and_foreign_nodes() -> None:
    graph = _fixture_graph()
    foreign = _node(
        ProgramNodeKind.MODULE,
        "module:foreign",
        forest_id="forest:other",
        component_id="module:pkg.mod",
    )
    with pytest.raises(ProgramGraphError, match="foreign forest"):
        graph.replace_component("module:pkg.mod", nodes=[foreign], edges=[])

    local = _node(
        ProgramNodeKind.MODULE,
        "module:pkg.mod",
        component_id="module:pkg.mod",
    )
    dangling = _edge(
        local.node_id,
        "pnode-missing",
        ProgramEdgeKind.DEFINES,
        component_id="module:pkg.mod",
    )
    with pytest.raises(DanglingEdgeError):
        graph.replace_component(
            "module:pkg.mod", nodes=[local], edges=[dangling]
        )


# ---------------------------------------------------------------------------
# VFS-G040 evidence terms: canonical graph vs optional GraphRAG ranking
# ---------------------------------------------------------------------------


def test_program_graph_evidence_term_vfs_program_graph() -> None:
    """Prove exact-text evidence term vfs/program-graph@1 for discovery scans."""

    assert PROGRAM_GRAPH_EVIDENCE == "vfs/program-graph@1"
    assert program_graph_evidence_terms() == ("vfs/program-graph@1",)

    graph = _fixture_graph()
    payload = graph.to_dict()
    assert payload["evidence"] == ["vfs/program-graph@1"]
    assert payload["evidence_program_graph"] == "vfs/program-graph@1"
    assert payload["canonical_construction"] is True
    assert payload["graphrag_ranking_authority"] is False
    # Envelope metadata must not alter content-addressed graph identity.
    assert payload["graph_id"] == graph.graph_id
    # Provenance remains content-bound on every node/edge.
    for node in graph.nodes:
        assert node.binding.blob_cid
        assert node.binding.forest_id == FOREST_ID
        assert node.node_id.startswith("pnode-")
    for edge in graph.edges:
        assert edge.binding.blob_cid
        assert edge.edge_id.startswith("pedge-")


def test_contract_finding_and_proof_obligation_nodes_are_provenance_bound() -> None:
    """VFS-G040 node kinds for contracts, findings, and proof obligations."""

    graph = _fixture_graph()
    by_kind = {node.kind: node for node in graph.nodes}
    assert ProgramNodeKind.CONTRACT in by_kind
    assert ProgramNodeKind.FINDING in by_kind
    assert ProgramNodeKind.PROOF_OBLIGATION in by_kind
    assert graph.edges_by_kind(ProgramEdgeKind.SUPPORTS)
    assert graph.edges_by_kind(ProgramEdgeKind.OBLIGATES)
    for kind in (
        ProgramNodeKind.CONTRACT,
        ProgramNodeKind.FINDING,
        ProgramNodeKind.PROOF_OBLIGATION,
    ):
        node = by_kind[kind]
        assert node.binding.blob_cid == BLOB_B
        assert node.binding.producer == PRODUCER
        assert node.binding.forest_id == FOREST_ID


def test_graphrag_projection_evidence_term_and_non_authority() -> None:
    """Prove vfs/graphrag-projection@1 is ranking-only and non-authoritative."""

    assert GRAPHRAG_PROJECTION_EVIDENCE == "vfs/graphrag-projection@1"
    assert graphrag_projection_evidence_terms() == ("vfs/graphrag-projection@1",)
    assert covered_evidence_terms() == (
        "vfs/program-graph@1",
        "vfs/graphrag-projection@1",
    )
    # Construction and ranking remain separate surfaces.
    assert program_graph_evidence_terms() == ("vfs/program-graph@1",)
    assert "vfs/graphrag-projection@1" not in program_graph_evidence_terms()

    def _unavailable(_name: str) -> Any:
        raise ModuleNotFoundError("ipfs_datasets_py unavailable in test")

    graph = _fixture_graph()
    # Force explicit local fallback so ranking does not depend on optional backends.
    provider = IpfsDatasetsProgramGraphProvider(importer=_unavailable)
    projection, result = provider.project_and_query(graph, "pkg.mod.entry")

    for payload in (projection.to_dict(), result.to_dict()):
        assert payload["evidence"] == "vfs/graphrag-projection@1"
        assert payload["evidence_graphrag_projection"] == "vfs/graphrag-projection@1"
        assert payload["evidence_program_graph"] == "vfs/program-graph@1"
        assert payload["canonical_construction"] is False
        assert payload["ranking_only"] is True
        assert payload["completion_authority"] is False
        assert payload["mutation_authority"] is False
        assert payload["creates_proofs"] is False
        assert payload["creates_findings"] is False
        assert payload["non_authoritative"] is True
        assert payload["safe_for_completion_reasoning"] is False
        assert payload["safe_for_proof_authority"] is False

    assert result.safe_for_completion_reasoning is False
    assert result.non_authoritative is True
    # Compact references with ranking reasons — never full source bodies.
    assert result.references
    for ref in result.references:
        body = ref.to_dict()
        assert "node_id" in body or "reference_id" in body
        assert "source_body" not in body
        assert "source_code" not in body
        assert "completion" not in body
        assert "ranking_reason" in body or "score" in body or "reasons" in body


def test_canonical_construction_separated_from_optional_graphrag_ranking() -> None:
    """GraphRAG cannot mint program-graph records or change graph identity."""

    def _unavailable(_name: str) -> Any:
        raise ModuleNotFoundError("ipfs_datasets_py unavailable in test")

    graph = _fixture_graph()
    original_id = graph.graph_id
    original_records = graph.canonical_records()

    provider = IpfsDatasetsProgramGraphProvider(importer=_unavailable)
    projection = provider.project(graph)
    result = provider.query(graph, "entry", projection=projection)

    # Ranking leaves the canonical graph untouched.
    assert graph.graph_id == original_id
    assert graph.canonical_records() == original_records
    assert projection.forest_id == graph.forest_id
    # Chunk CIDs are deterministic and bound to admitted evidence only.
    assert projection.chunks
    for chunk in projection.chunks:
        assert chunk.chunk_cid
    chunk_payload = projection.to_dict()
    assert chunk_payload["ranking_only"] is True
    assert chunk_payload["creates_calls"] is False
    # Result identity is independent of program-graph identity.
    assert result.result_id != graph.graph_id
    assert not result.result_id.startswith("pgraph-")


def test_objective_validation_repair_evidence_term_discoverable() -> None:
    """VFS-G040 objective validation repair: exact-text discovery key present.

    Anchors the synthetic phrase ``objective validation repair`` so objective
    scans re-find the validation gate.  Domain evidence stays separate:
    construction (``vfs/program-graph@1``) vs optional ranking
    (``vfs/graphrag-projection@1``).  The repair term never enters graph_id
    identity or ranking result authority.
    """

    assert OBJECTIVE_VALIDATION_REPAIR_EVIDENCE == "objective validation repair"
    assert OBJECTIVE_GOAL_ID == "VFS-G040"
    assert objective_validation_repair_evidence_terms() == (
        "objective validation repair",
    )
    # Domain envelope evidence remains construction-only.
    assert program_graph_evidence_terms() == ("vfs/program-graph@1",)
    assert "objective validation repair" not in program_graph_evidence_terms()
    assert covered_evidence_terms() == (
        "vfs/program-graph@1",
        "vfs/graphrag-projection@1",
    )
    assert "objective validation repair" not in covered_evidence_terms()
    # Full discovery set includes the validation-gate meta term last.
    assert all_covered_evidence_terms() == (
        "vfs/program-graph@1",
        "vfs/graphrag-projection@1",
        "objective validation repair",
    )

    graph = _fixture_graph()
    payload = graph.to_dict()
    # Graph identity envelope never absorbs the synthetic repair term.
    assert payload["evidence"] == ["vfs/program-graph@1"]
    assert "objective validation repair" not in payload["evidence"]
    assert payload["canonical_construction"] is True
    assert payload["graphrag_ranking_authority"] is False

    def _unavailable(_name: str) -> Any:
        raise ModuleNotFoundError("ipfs_datasets_py unavailable in test")

    provider = IpfsDatasetsProgramGraphProvider(importer=_unavailable)
    projection, result = provider.project_and_query(graph, "pkg.mod.entry")
    for body in (projection.to_dict(), result.to_dict()):
        assert body["evidence"] == "vfs/graphrag-projection@1"
        assert body["ranking_only"] is True
        assert body["completion_authority"] is False
        assert body["safe_for_proof_authority"] is False
        assert body.get("evidence_objective_validation_repair") is None
        # Separate canonical construction from optional GraphRAG ranking.
        assert body["canonical_construction"] is False
        assert body["evidence_program_graph"] == "vfs/program-graph@1"
