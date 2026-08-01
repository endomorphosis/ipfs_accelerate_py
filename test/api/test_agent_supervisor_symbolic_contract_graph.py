"""SCA-030 contract tests for the typed graph and bounded GraphRAG view."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_ast_index import (
    AnalysisASTIndex,
    IndexedASTPath,
)
from ipfs_accelerate_py.agent_supervisor.analysis.code_evidence_graph import (
    CodeEvidenceGraph,
    EvidenceEdgeKind,
    EvidenceNode,
    EvidenceNodeKind,
    EvidenceProvenance,
    ProvenanceEdge,
)
from ipfs_accelerate_py.agent_supervisor.analysis.symbolic_contract_graph import (
    BOUNDED_GRAPHRAG_RETRIEVER_INTERFACE,
    GRAPH_VERSION,
    SYMBOLIC_CONTRACT_GRAPH_INTERFACE,
    BoundedGraphRAGRetriever,
    ClosureBounds,
    ContractAuthority,
    ContractEdgeKind,
    ContractGraphEdge,
    ContractGraphNode,
    ContractNodeKind,
    ContractProvenance,
    GraphRAGRetrievalReceipt,
    IncompleteMandatoryClosureError,
    RetrievalBounds,
    SymbolicContractGraph,
    SymbolicContractGraphError,
    build_symbolic_contract_graph,
    canonical_contract_graph_bytes,
    project_symbolic_contract_graph,
)
from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import ASTBlobRecord


SNAPSHOT = "repository-snapshot:sha256:fixture"


def _node(
    key: str,
    *,
    kind: ContractNodeKind = ContractNodeKind.SYMBOL,
    provenance: ContractProvenance = ContractProvenance.AST,
    authority: ContractAuthority = ContractAuthority.SOURCE_OBSERVATION,
    payload=None,
    required_dependencies=(),
) -> ContractGraphNode:
    return ContractGraphNode(
        kind=kind,
        stable_key=key,
        snapshot_id=SNAPSHOT,
        provenance=provenance,
        authority=authority,
        version=GRAPH_VERSION,
        payload=payload or {"label": key},
        source_refs=("ast-record:fixture",),
        required_dependencies=required_dependencies,
    )


def _edge(
    source: ContractGraphNode,
    target: ContractGraphNode,
    *,
    kind: ContractEdgeKind = ContractEdgeKind.DEPENDS_ON,
    provenance: ContractProvenance = ContractProvenance.AST,
    authority: ContractAuthority = ContractAuthority.SOURCE_OBSERVATION,
    mandatory: bool = True,
) -> ContractGraphEdge:
    return ContractGraphEdge(
        kind=kind,
        source=source.node_id,
        target=target.node_id,
        snapshot_id=SNAPSHOT,
        provenance=provenance,
        authority=authority,
        version=GRAPH_VERSION,
        mandatory=mandatory,
        source_refs=("ast-record:fixture",),
    )


def _diamond_graph() -> tuple[
    SymbolicContractGraph, dict[str, ContractGraphNode]
]:
    nodes = {name: _node(f"symbol:{name}") for name in ("a", "b", "c", "d")}
    edges = (
        _edge(nodes["a"], nodes["b"], kind=ContractEdgeKind.CALLS),
        _edge(nodes["a"], nodes["c"], kind=ContractEdgeKind.IMPORTS),
        _edge(nodes["b"], nodes["d"]),
        _edge(nodes["c"], nodes["d"]),
    )
    annotation = _node(
        "context:ranked",
        kind=ContractNodeKind.PROVENANCE,
        provenance=ContractProvenance.GRAPHRAG,
        authority=ContractAuthority.CONTEXT_ONLY,
    )
    context_edge = _edge(
        nodes["a"],
        annotation,
        kind=ContractEdgeKind.RELATED_TO,
        provenance=ContractProvenance.GRAPHRAG,
        authority=ContractAuthority.CONTEXT_ONLY,
        mandatory=False,
    )
    graph = SymbolicContractGraph(
        snapshot_id=SNAPSHOT,
        nodes=(*nodes.values(), annotation),
        edges=(*edges, context_edge),
    )
    return graph, nodes


def test_interface_and_content_identity_are_present_on_every_record() -> None:
    graph, _ = _diamond_graph()

    assert SYMBOLIC_CONTRACT_GRAPH_INTERFACE == "SymbolicContractGraph@1"
    assert graph.to_dict()["interface"] == SYMBOLIC_CONTRACT_GRAPH_INTERFACE
    assert graph.graph_root.startswith("b")
    assert graph.graph_id == graph.root_id == graph.identity.cid
    assert graph.identity.profile == "strict-dag-json-v1"
    assert graph.identity.interface == "ContentIdentity@1"

    for record in (*graph.nodes, *graph.edges):
        projection = record.to_dict()
        assert projection["snapshot_id"] == SNAPSHOT
        assert projection["provenance"]
        assert projection["authority"]
        assert projection["version"] == GRAPH_VERSION
        assert projection["identity"]["cid"].startswith("b")
        assert projection["identity"]["digest"].startswith("sha256:")
        assert projection["identity"]["validated"] is True
        assert projection["identity"]["profile"] == "strict-dag-json-v1"


def test_graph_root_is_order_independent_stable_and_round_trips() -> None:
    graph, _ = _diamond_graph()
    reordered = SymbolicContractGraph(
        snapshot_id=graph.snapshot_id,
        version=graph.version,
        nodes=tuple(reversed(graph.nodes)),
        edges=tuple(reversed(graph.edges)),
    )

    assert reordered.graph_root == graph.graph_root
    assert reordered.to_json() == graph.to_json()
    rebuilt = SymbolicContractGraph.from_json(graph.to_json())
    assert rebuilt == graph
    assert rebuilt.graph_root == graph.graph_root


def test_node_edge_and_root_tampering_is_rejected() -> None:
    graph, _ = _diamond_graph()
    payload = graph.to_dict()
    payload["nodes"][0]["payload"]["tampered"] = True
    with pytest.raises(SymbolicContractGraphError, match="node identity"):
        SymbolicContractGraph.from_dict(payload)

    payload = graph.to_dict()
    payload["graph_root"] = "baguqeeratampered"
    payload["graph_id"] = "baguqeeratampered"
    with pytest.raises(SymbolicContractGraphError, match="graph root"):
        SymbolicContractGraph.from_dict(payload)


def test_cross_snapshot_foreign_version_and_unknown_endpoint_fail_closed() -> None:
    left = _node("symbol:left")
    foreign = replace(
        _node("symbol:foreign"),
        snapshot_id="foreign-snapshot",
        node_id="",
        identity=None,
    )
    with pytest.raises(SymbolicContractGraphError, match="foreign snapshot"):
        SymbolicContractGraph(SNAPSHOT, (left, foreign), ())

    right = _node("symbol:right")
    bad = replace(
        _edge(left, right),
        target="baguqeeraunknown",
        edge_id="",
        identity=None,
    )
    with pytest.raises(SymbolicContractGraphError, match="unknown node"):
        SymbolicContractGraph(SNAPSHOT, (left, right), (bad,))

    foreign_version = replace(left, version="2", node_id="", identity=None)
    with pytest.raises(SymbolicContractGraphError, match="foreign graph version"):
        SymbolicContractGraph(SNAPSHOT, (foreign_version,), ())


def test_graphrag_provenance_cannot_mint_authority_or_mandatory_edges() -> None:
    left = _node("symbol:left")
    right = _node("symbol:right")

    with pytest.raises(SymbolicContractGraphError, match="context_only"):
        _node(
            "context:bad",
            provenance=ContractProvenance.GRAPHRAG,
            authority=ContractAuthority.SOURCE_OBSERVATION,
        )
    with pytest.raises(SymbolicContractGraphError, match="never mandatory"):
        _edge(
            left,
            right,
            provenance=ContractProvenance.GRAPHRAG,
            authority=ContractAuthority.CONTEXT_ONLY,
            mandatory=True,
        )


def test_forward_and_reverse_closure_are_exact_and_deterministic() -> None:
    graph, nodes = _diamond_graph()

    forward = graph.forward_closure(nodes["a"].node_id)
    reverse = graph.reverse_closure(nodes["d"].node_id)
    again = graph.forward_closure(nodes["a"].node_id)

    expected_nodes = {item.node_id for item in nodes.values()}
    expected_edges = {edge.edge_id for edge in graph.edges if edge.mandatory}
    assert set(forward.node_ids) == expected_nodes
    assert set(forward.edge_ids) == expected_edges
    assert set(reverse.node_ids) == expected_nodes
    assert set(reverse.edge_ids) == expected_edges
    assert forward.closure_id == again.closure_id
    assert forward.paths[nodes["d"].node_id][0] == nodes["a"].node_id
    assert reverse.paths[nodes["a"].node_id][0] == nodes["d"].node_id
    assert all(
        graph.node(node_id).authority is not ContractAuthority.CONTEXT_ONLY
        for node_id in forward.node_ids
    )
    assert forward.complete and forward.safe_for_proof
    assert type(forward).from_dict(forward.to_dict()).closure_id == (
        forward.closure_id
    )


def test_typed_closure_filter_is_exact() -> None:
    graph, nodes = _diamond_graph()
    calls_only = graph.forward_closure(
        nodes["a"].node_id,
        edge_kinds=(ContractEdgeKind.CALLS,),
    )
    assert calls_only.node_ids == tuple(
        sorted((nodes["a"].node_id, nodes["b"].node_id))
    )
    assert len(calls_only.edge_ids) == 1


def test_closure_truncation_raises_with_non_authoritative_receipt() -> None:
    graph, nodes = _diamond_graph()
    with pytest.raises(
        IncompleteMandatoryClosureError, match="max_nodes_exceeded"
    ) as caught:
        graph.forward_closure(
            nodes["a"].node_id,
            bounds=ClosureBounds(max_nodes=2, max_edges=20, max_depth=20),
        )

    receipt = caught.value.receipt
    assert receipt.truncated
    assert not receipt.complete
    assert not receipt.safe_for_proof
    assert receipt.reason_code == "max_nodes_exceeded"
    explicit = graph.forward_closure(
        nodes["a"].node_id,
        bounds=ClosureBounds(max_nodes=2, max_edges=20, max_depth=20),
        fail_closed=False,
    )
    assert explicit.to_dict()["authority"] == "none"


def test_missing_mandatory_edge_manifest_fails_closure_closed() -> None:
    left = _node("symbol:left")
    right = _node("symbol:right")
    edge = _edge(left, right)
    missing = "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    graph = SymbolicContractGraph(
        SNAPSHOT,
        (left, right),
        (edge,),
        mandatory_edge_ids=(edge.edge_id, missing),
    )

    assert not graph.complete
    assert graph.missing_mandatory_edge_ids == (missing,)
    with pytest.raises(
        IncompleteMandatoryClosureError, match="missing_mandatory_edges"
    ) as caught:
        graph.forward_closure(left.node_id)
    assert caught.value.receipt.missing_edge_ids == (missing,)


def test_node_required_dependency_without_edge_fails_closure_closed() -> None:
    right = _node("symbol:right")
    left = _node(
        "symbol:left", required_dependencies=(right.stable_key,)
    )
    graph = SymbolicContractGraph(SNAPSHOT, (left, right), ())

    assert graph.missing_dependency_keys == (
        "symbol:left->symbol:right:missing_edge",
    )
    with pytest.raises(
        IncompleteMandatoryClosureError,
        match="missing_mandatory_dependencies",
    ):
        graph.forward_closure(left.node_id)


def test_serialized_mandatory_manifest_detects_removed_edge() -> None:
    graph, nodes = _diamond_graph()
    payload = graph.to_dict()
    payload.pop("graph_root")
    payload.pop("graph_id")
    payload.pop("identity")
    payload.pop("complete")
    payload["edges"] = payload["edges"][1:]
    payload["edge_count"] -= 1
    rebuilt = SymbolicContractGraph.from_dict(payload)

    assert not rebuilt.complete
    with pytest.raises(IncompleteMandatoryClosureError):
        rebuilt.forward_closure(nodes["a"].node_id)


def test_bounded_local_retrieval_has_content_addressed_context_receipt() -> None:
    graph, _ = _diamond_graph()
    retriever = BoundedGraphRAGRetriever(graph)
    bounds = RetrievalBounds(max_candidates=2, max_bytes=4_096)
    receipt = retriever.retrieve("symbol", bounds=bounds)

    assert BOUNDED_GRAPHRAG_RETRIEVER_INTERFACE == (
        "BoundedGraphRAGRetriever@1"
    )
    assert isinstance(receipt, GraphRAGRetrievalReceipt)
    assert len(receipt.candidates) == 2
    assert receipt.truncated
    assert receipt.total_matches >= len(receipt.candidates)
    assert receipt.byte_count <= bounds.max_bytes
    assert receipt.receipt_id.startswith("b")
    assert receipt.identity.cid == receipt.receipt_id
    assert receipt.non_authoritative
    assert not receipt.safe_for_proof
    assert receipt.to_dict()["authority"] == "context_only"
    assert all(
        item.to_dict()["provenance"] == "graphrag"
        for item in receipt.candidates
    )
    rebuilt = GraphRAGRetrievalReceipt.from_dict(receipt.to_dict())
    assert rebuilt.receipt_id == receipt.receipt_id

    tampered = receipt.to_dict()
    tampered["safe_for_proof"] = True
    with pytest.raises(SymbolicContractGraphError, match="claim mismatch"):
        GraphRAGRetrievalReceipt.from_dict(tampered)


def test_retrieval_byte_limit_is_enforced_with_explicit_truncation() -> None:
    nodes = tuple(
        _node(
            f"symbol:searchable:{index}",
            payload={"label": "searchable " + "x" * 128, "index": index},
        )
        for index in range(20)
    )
    graph = SymbolicContractGraph(SNAPSHOT, nodes, ())
    retriever = BoundedGraphRAGRetriever(graph)
    bounds = RetrievalBounds(max_candidates=20, max_bytes=2_048)
    receipt = retriever.retrieve("searchable", bounds=bounds)

    assert receipt.truncated
    assert len(receipt.candidates) < 20
    assert receipt.byte_count <= 2_048
    assert receipt.to_dict()["omitted_candidates"] > 0


def test_optional_datasets_provider_remains_lazy_until_explicit_request() -> None:
    graph, nodes = _diamond_graph()
    calls: list[str] = []

    class Provider:
        def analyze(self, request):
            calls.append("analyze")
            assert request["operation"] == "graph_retrieval"
            assert request["payload"]["graph_root"] == graph.graph_root
            return {
                "status": "completed",
                "results": [{"node_id": nodes["d"].node_id}],
                "truncated": False,
                "receipt_id": "provider-receipt:fixture",
            }

    def factory():
        calls.append("factory")
        return Provider()

    retriever = BoundedGraphRAGRetriever(graph, provider_factory=factory)
    before = retriever.capability()
    local = retriever.retrieve("no-local-token", bounds=RetrievalBounds(max_bytes=4_096))

    assert calls == []
    assert before["provider_loaded"] is False
    assert local.provider_requested is False
    assert local.provider_loaded is False

    provider = retriever.retrieve(
        "no-local-token",
        bounds=RetrievalBounds(max_bytes=4_096),
        use_optional_provider=True,
    )
    assert calls == ["factory", "analyze"]
    assert provider.provider_requested
    assert provider.provider_loaded
    assert provider.provider_status == "completed"
    assert nodes["d"].node_id in provider.candidate_node_ids


def test_optional_provider_failure_degrades_to_local_context() -> None:
    graph, _ = _diamond_graph()

    def factory():
        raise ModuleNotFoundError("ipfs_datasets_py")

    retriever = BoundedGraphRAGRetriever(graph, provider_factory=factory)
    receipt = retriever.retrieve(
        "symbol",
        bounds=RetrievalBounds(max_candidates=3, max_bytes=4_096),
        use_optional_provider=True,
    )

    assert receipt.provider_requested
    assert not receipt.provider_loaded
    assert receipt.provider_status == "degraded:ModuleNotFoundError"
    assert receipt.candidates
    assert receipt.non_authoritative


def test_bounded_view_uses_rank_only_for_seeds_then_exact_mandatory_closure() -> None:
    graph, nodes = _diamond_graph()
    view = BoundedGraphRAGRetriever(graph).view(
        "symbol:a",
        retrieval_bounds=RetrievalBounds(
            max_candidates=1, max_bytes=4_096
        ),
    )

    assert view.retrieval.candidate_node_ids == (nodes["a"].node_id,)
    assert set(view.mandatory_closure.node_ids) == {
        item.node_id for item in nodes.values()
    }
    assert view.mandatory_closure_complete
    assert not view.safe_for_proof
    assert view.to_dict()["authority"] == "context_only"


def test_projection_covers_ast_files_modules_symbols_calls_imports_and_effects() -> None:
    record = ASTBlobRecord(
        blob_identity="blob:sha256:fixture",
        source_sha256="sha256:" + "1" * 64,
        qualified_symbols=("Service.run",),
        imports=("pkg.policy",),
        calls=("Service.run->remote.execute",),
        state_transitions=("Service.run:mutates-cache",),
        interfaces=("Service.run(request)->result",),
        symbol_hashes={"Service.run": "symbol-hash"},
        symbol_lines={"Service.run": (10, 20)},
    )
    ast_index = AnalysisASTIndex(
        path_records=(IndexedASTPath("src/service.py", record),)
    )
    graph = project_symbolic_contract_graph(
        {"snapshot_id": SNAPSHOT, "ast_index": ast_index}
    )

    expected_kinds = {
        ContractNodeKind.REPOSITORY_SNAPSHOT,
        ContractNodeKind.FILE,
        ContractNodeKind.MODULE,
        ContractNodeKind.SYMBOL,
        ContractNodeKind.CALL,
        ContractNodeKind.IMPORT,
        ContractNodeKind.EFFECT,
        ContractNodeKind.INTERFACE,
        ContractNodeKind.UNRESOLVED,
    }
    assert expected_kinds <= {node.kind for node in graph.nodes}
    assert {
        ContractEdgeKind.SOURCED_FROM,
        ContractEdgeKind.DEFINED_IN,
        ContractEdgeKind.CALLS,
        ContractEdgeKind.IMPORTS,
        ContractEdgeKind.DEPENDS_ON,
        ContractEdgeKind.HAS_EFFECT,
        ContractEdgeKind.DECLARES,
    } <= {edge.kind for edge in graph.edges}
    assert graph.complete
    assert all(node.identity and node.source_refs for node in graph.nodes[1:])
    assert all(edge.mandatory for edge in graph.edges)


def test_projection_composes_schema_index_and_code_evidence_graph() -> None:
    record = ASTBlobRecord(
        blob_identity="blob:sha256:schema-fixture",
        source_sha256="sha256:" + "2" * 64,
    )
    ast_index = AnalysisASTIndex(
        path_records=(IndexedASTPath("schemas/tool.json", record),)
    )
    evidence_left = EvidenceNode(
        kind=EvidenceNodeKind.TASK,
        record_key="task:SCA-030",
        provenance=EvidenceProvenance.TASK,
        record={"task_id": "SCA-030"},
        task_id="SCA-030",
    )
    evidence_right = EvidenceNode(
        kind=EvidenceNodeKind.TREE,
        record_key=SNAPSHOT,
        provenance=EvidenceProvenance.TASK,
        record={"tree_id": SNAPSHOT},
        tree_id=SNAPSHOT,
    )
    evidence_edge = ProvenanceEdge(
        source=evidence_left.node_id,
        target=evidence_right.node_id,
        kind=EvidenceEdgeKind.DEPENDS_ON,
        provenance=EvidenceProvenance.TASK,
        provenance_record_id="task-receipt:fixture",
    )
    evidence_graph = CodeEvidenceGraph(
        nodes=(evidence_left, evidence_right),
        edges=(evidence_edge,),
    )

    graph = project_symbolic_contract_graph(
        {"snapshot_id": SNAPSHOT, "ast_index": ast_index},
        schema_index=[
            {
                "schema_id": "schema:tool@1",
                "record_id": "schema-record:fixture",
                "path": "schemas/tool.json",
                "schema_version": "1",
                # Bodies are deliberately not copied into graph rows.
                "schema": {"type": "object", "properties": {"secret": {}}},
            }
        ],
        code_evidence_graph=evidence_graph,
    )

    schema = graph.nodes_by_kind(ContractNodeKind.SCHEMA)
    assert len(schema) == 1
    assert "schema" not in schema[0].payload
    assert schema[0].payload["schema_version"] == "1"
    assert any(
        node.stable_key == "code-evidence:task:SCA-030"
        for node in graph.nodes
    )
    assert any(
        edge.payload.get("code_evidence_kind") == "depends_on"
        and edge.mandatory
        for edge in graph.edges
    )
    assert graph.complete


def test_projection_and_typed_builder_are_stable_across_input_order() -> None:
    left = _node("tool:left", kind=ContractNodeKind.TOOL)
    right = _node("handler:right", kind=ContractNodeKind.HANDLER)
    edge = _edge(left, right, kind=ContractEdgeKind.HANDLED_BY)
    first = build_symbolic_contract_graph(
        snapshot_id=SNAPSHOT, nodes=(left, right), edges=(edge,)
    )
    second = build_symbolic_contract_graph(
        snapshot_id=SNAPSHOT, nodes=(right, left), edges=(edge,)
    )

    assert first.graph_root == second.graph_root
    assert canonical_contract_graph_bytes(first.to_dict()) == (
        canonical_contract_graph_bytes(second.to_dict())
    )
