from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_ast_index import (
    AnalysisASTIndex,
    AnalysisASTIndexStats,
    IndexedASTPath,
)
from ipfs_accelerate_py.agent_supervisor.analysis.symbolic_contract_graph import (
    CONTENT_IDENTITY_INTERFACE,
    SYMBOLIC_CONTRACT_GRAPH_INTERFACE,
    ClosureBounds,
    ContractAuthority,
    ContractEdgeKind,
    ContractGraphEdge,
    ContractGraphNode,
    ContractNodeKind,
    ContractProvenance,
    DatasetsProviderState,
    GraphClosureTruncatedError,
    GraphIntegrityError,
    GraphRAGLimits,
    LazyDatasetsGraphRAGProvider,
    MandatoryEdgeRequirement,
    MissingMandatoryEdgeError,
    SymbolicContractGraph,
    project_repository_index,
)
from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import ASTBlobRecord


SNAPSHOT = "snapshot:fixture-v1"
INDEX = "repository-index:fixture-v1"
VERSION = "fixture-contract@1"


def _node(
    kind: ContractNodeKind,
    key: str,
    *,
    authority: ContractAuthority = ContractAuthority.SOURCE_OBSERVATION,
    provenance: ContractProvenance = ContractProvenance.AST,
    **metadata: object,
) -> ContractGraphNode:
    return ContractGraphNode(
        kind=kind,
        key=key,
        label=key,
        snapshot_id=SNAPSHOT,
        provenance=provenance,
        provenance_id=f"receipt:{key}",
        authority=authority,
        version=VERSION,
        metadata=metadata,
    )


def _edge(
    source: ContractGraphNode,
    target: ContractGraphNode,
    kind: ContractEdgeKind,
    *,
    mandatory: bool = True,
    authority: ContractAuthority = ContractAuthority.SOURCE_OBSERVATION,
    provenance: ContractProvenance = ContractProvenance.AST,
) -> ContractGraphEdge:
    return ContractGraphEdge(
        source=source.node_id,
        target=target.node_id,
        kind=kind,
        snapshot_id=SNAPSHOT,
        provenance=provenance,
        provenance_id=f"receipt:{source.key}:{target.key}",
        authority=authority,
        version=VERSION,
        mandatory=mandatory,
    )


def _graph() -> tuple[
    SymbolicContractGraph,
    ContractGraphNode,
    ContractGraphNode,
    ContractGraphNode,
]:
    module = _node(
        ContractNodeKind.MODULE,
        "module:dispatch",
        path="src/dispatch.py",
        module="src.dispatch",
    )
    tool = _node(
        ContractNodeKind.TOOL,
        "tool:search",
        path="src/dispatch.py",
        symbol="search_tool",
    )
    handler = _node(
        ContractNodeKind.HANDLER,
        "handler:search",
        path="src/handler.py",
        symbol="search_handler",
    )
    effect = _node(
        ContractNodeKind.EFFECT,
        "effect:write",
        path="src/handler.py",
    )
    edges = (
        _edge(module, tool, ContractEdgeKind.DEFINES),
        _edge(tool, handler, ContractEdgeKind.DISPATCHES_TO),
        _edge(handler, effect, ContractEdgeKind.HAS_EFFECT),
    )
    return (
        SymbolicContractGraph(
            snapshot_id=SNAPSHOT,
            source_index_id=INDEX,
            nodes=(effect, handler, tool, module),
            edges=tuple(reversed(edges)),
        ),
        module,
        tool,
        effect,
    )


def test_every_record_binds_identity_provenance_authority_snapshot_and_version() -> None:
    graph, *_ = _graph()

    assert graph.identity["interface"] == CONTENT_IDENTITY_INTERFACE
    assert graph.identity["validated"] is True
    assert graph.to_dict()["interface"] == SYMBOLIC_CONTRACT_GRAPH_INTERFACE
    for record in (*graph.to_dict()["nodes"], *graph.to_dict()["edges"]):
        assert record["identity"]["cid"].startswith("b")
        assert record["identity"]["validated"] is True
        assert record["provenance"]
        assert record["provenance_id"]
        assert record["authority"]
        assert record["snapshot_id"] == SNAPSHOT
        assert record["version"]


def test_graph_root_and_json_are_stable_across_input_order_and_round_trip() -> None:
    graph, *_ = _graph()
    reordered = SymbolicContractGraph(
        snapshot_id=graph.snapshot_id,
        source_index_id=graph.source_index_id,
        nodes=tuple(reversed(graph.nodes)),
        edges=tuple(reversed(graph.edges)),
    )

    assert graph.root_id == reordered.root_id
    assert graph.to_json() == reordered.to_json()
    assert SymbolicContractGraph.from_json(graph.to_json()).to_json() == graph.to_json()

    tampered = json.loads(graph.to_json())
    tampered["nodes"][0]["label"] = "silently changed"
    with pytest.raises(GraphIntegrityError, match="identity"):
        SymbolicContractGraph.from_dict(tampered)


def test_exact_forward_and_reverse_typed_closure() -> None:
    graph, module, tool, effect = _graph()

    forward = graph.forward_closure(module.node_id)
    reverse = graph.reverse_closure(effect.node_id)
    dispatch_only = graph.forward_closure(
        tool.node_id, edge_kinds=(ContractEdgeKind.DISPATCHES_TO,)
    )

    assert forward.complete and not forward.truncated
    assert forward.node_ids == tuple(node.node_id for node in graph.nodes)
    assert reverse.node_ids == forward.node_ids
    assert forward.paths[effect.node_id] == (
        module.node_id,
        tool.node_id,
        next(
            node.node_id
            for node in graph.nodes
            if node.kind is ContractNodeKind.HANDLER
        ),
        effect.node_id,
    )
    assert dispatch_only.node_ids == tuple(
        sorted(
            (
                tool.node_id,
                next(
                    node.node_id
                    for node in graph.nodes
                    if node.kind is ContractNodeKind.HANDLER
                ),
            )
        )
    )


def test_closure_bounds_fail_closed_with_incomplete_receipt() -> None:
    graph, module, *_ = _graph()

    with pytest.raises(GraphClosureTruncatedError) as captured:
        graph.mandatory_closure(
            module.node_id,
            bounds=ClosureBounds(
                max_nodes=2,
                max_edges=10,
                max_depth=10,
                max_bytes=100_000,
            ),
        )

    assert captured.value.reason_code == "max_nodes_exceeded"
    assert captured.value.receipt is not None
    assert captured.value.receipt.complete is False
    assert captured.value.receipt.truncated is True


def test_missing_mandatory_edges_fail_at_graph_and_closure_boundaries() -> None:
    graph, module, tool, effect = _graph()
    missing = MandatoryEdgeRequirement(
        source=module.node_id,
        target=effect.node_id,
        kind=ContractEdgeKind.AUTHORIZES,
    )

    with pytest.raises(MissingMandatoryEdgeError) as build_error:
        SymbolicContractGraph(
            snapshot_id=SNAPSHOT,
            source_index_id=INDEX,
            nodes=graph.nodes,
            edges=graph.edges,
            mandatory_requirements=(missing,),
        )
    assert missing.requirement_id in build_error.value.missing

    with pytest.raises(MissingMandatoryEdgeError) as closure_error:
        graph.mandatory_closure(module.node_id, required_edges=(missing,))
    assert closure_error.value.receipt is not None
    assert closure_error.value.receipt.complete is False
    assert closure_error.value.receipt.missing_mandatory_edges

    dangling = ContractGraphEdge(
        source=module.node_id,
        target="missing-node",
        kind=ContractEdgeKind.DEFINES,
        snapshot_id=SNAPSHOT,
        provenance=ContractProvenance.AST,
        provenance_id="receipt:missing",
        authority=ContractAuthority.SOURCE_OBSERVATION,
        version=VERSION,
    )
    with pytest.raises(MissingMandatoryEdgeError):
        SymbolicContractGraph(
            snapshot_id=SNAPSHOT,
            source_index_id=INDEX,
            nodes=(module, tool),
            edges=(dangling,),
        )


def test_context_only_authority_boundary_is_enforced() -> None:
    graph, module, tool, _ = _graph()

    with pytest.raises(GraphIntegrityError, match="context-only"):
        _edge(
            module,
            tool,
            ContractEdgeKind.RELATED_TO,
            provenance=ContractProvenance.GRAPHRAG,
        )
    with pytest.raises(GraphIntegrityError, match="GraphRAG"):
        _node(
            ContractNodeKind.TOOL,
            "ranked-tool",
            provenance=ContractProvenance.GRAPHRAG,
        )

    receipt = graph.retrieve_candidates(
        "search handler",
        limits=GraphRAGLimits(
            max_candidates=4,
            max_results=4,
            max_bytes=32_768,
            max_hops=2,
        ),
    )
    assert receipt.results
    assert receipt.output_bytes == len(receipt.to_json().encode())
    assert receipt.to_dict()["completion_authoritative"] is False
    assert all(
        edge.authority is ContractAuthority.CONTEXT_ONLY
        and edge.to_dict()["authoritative"] is False
        and edge.to_dict()["mandatory"] is False
        for edge in receipt.context_edges
    )


def test_candidate_retrieval_is_deterministic_bounded_and_receipted() -> None:
    graph, *_ = _graph()
    limits = GraphRAGLimits(
        max_candidates=2,
        max_results=1,
        max_bytes=16_384,
        max_hops=1,
    )

    first = graph.retrieve_candidates("search", limits=limits)
    second = graph.retrieve_candidates("search", limits=limits)

    assert first.to_json() == second.to_json()
    assert first.receipt_id == second.receipt_id
    assert len(first.candidates) == 1
    assert first.truncated
    assert first.dropped_count > 0
    assert len(first.to_json().encode()) <= limits.max_bytes
    assert first.graph_root == graph.root_id
    assert first.snapshot_id == graph.snapshot_id


def test_optional_datasets_provider_is_lazy_and_never_promotes_candidates() -> None:
    graph, _, tool, _ = _graph()
    calls: list[str] = []

    def provider(*, query, graph, limit):
        calls.append(query)
        assert graph["authority"] == ContractAuthority.CONTEXT_ONLY.value
        return [{"node_id": tool.node_id}, {"node_id": "unknown"}][:limit]

    datasets = LazyDatasetsGraphRAGProvider(candidate_callable=provider)
    assert not datasets.loaded

    receipt = graph.retrieve_candidates(
        "no lexical match xyz",
        datasets_provider=datasets,
    )

    assert datasets.loaded
    assert calls == ["no lexical match xyz"]
    assert receipt.provider_state is DatasetsProviderState.HEALTHY
    selected = next(
        candidate for candidate in receipt.candidates if candidate.node_id == tool.node_id
    )
    assert "datasets_candidate" in selected.reason_codes
    assert selected.to_dict()["authority"] == ContractAuthority.CONTEXT_ONLY.value


def test_cold_module_import_does_not_load_optional_datasets_graphrag() -> None:
    script = """
import sys
import ipfs_accelerate_py.agent_supervisor.analysis.symbolic_contract_graph
print(int('ipfs_datasets_py.knowledge_graphs.query.unified_engine' in sys.modules))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                (
                    str(Path(__file__).resolve().parents[2]),
                    os.environ.get("PYTHONPATH", ""),
                )
            ),
        },
    )
    assert completed.stdout.strip() == "0"


@dataclass(frozen=True)
class _Row:
    path: str
    row_id: str
    disposition_kind: str
    content_digest: str
    git_status: str = "clean"
    ast_record_id: str = ""


@dataclass(frozen=True)
class _Index:
    rows: tuple[_Row, ...]
    ast_index: AnalysisASTIndex
    snapshot_id: str = SNAPSHOT
    index_id: str = INDEX
    safe_for_completion_reasoning: bool = True


def _repository_index() -> _Index:
    service = ASTBlobRecord(
        blob_identity="blob-service",
        source_sha256="sha256:" + "a" * 64,
        qualified_symbols=("search_tool", "search_handler", "test_search"),
        imports=("src.policy",),
        calls=("src.policy.authorize", "search_handler", "dynamic_call"),
        state_transitions=("write:results",),
        interfaces=("def search_tool(query: str) -> dict",),
        symbol_hashes={
            "search_tool": "symbol:" + "1" * 64,
            "search_handler": "symbol:" + "2" * 64,
            "test_search": "symbol:" + "3" * 64,
        },
        symbol_lines={
            "search_tool": (1, 2),
            "search_handler": (4, 5),
            "test_search": (7, 8),
        },
    )
    policy = ASTBlobRecord(
        blob_identity="blob-policy",
        source_sha256="sha256:" + "b" * 64,
        qualified_symbols=("authorize",),
    )
    paths = (
        IndexedASTPath("src/service.py", service),
        IndexedASTPath("src/policy.py", policy),
    )
    ast_index = AnalysisASTIndex(
        path_records=paths,
        stats=AnalysisASTIndexStats(
            scanned_path_count=3,
            indexed_blob_count=2,
            new_blob_count=2,
        ),
    )
    rows = (
        _Row(
            "schemas/tool.json",
            "row:schema",
            "structured_data",
            "sha256:" + "c" * 64,
            ast_record_id="ast:schema",
        ),
        _Row(
            "src/policy.py",
            "row:policy",
            "semantic_ast",
            "sha256:" + "b" * 64,
            ast_record_id=policy.record_id,
        ),
        _Row(
            "src/service.py",
            "row:service",
            "semantic_ast",
            "sha256:" + "a" * 64,
            ast_record_id=service.record_id,
        ),
    )
    return _Index(rows=rows, ast_index=ast_index)


def test_repository_projection_covers_typed_facts_and_unresolved_targets() -> None:
    graph = project_repository_index(_repository_index())
    kinds = {node.kind for node in graph.nodes}

    assert {
        ContractNodeKind.REPOSITORY_SNAPSHOT,
        ContractNodeKind.FILE,
        ContractNodeKind.MODULE,
        ContractNodeKind.SYMBOL,
        ContractNodeKind.IMPORT,
        ContractNodeKind.CALL,
        ContractNodeKind.EFFECT,
        ContractNodeKind.SCHEMA,
        ContractNodeKind.INTERFACE,
        ContractNodeKind.TOOL,
        ContractNodeKind.HANDLER,
        ContractNodeKind.TEST,
        ContractNodeKind.POLICY,
        ContractNodeKind.UNRESOLVED,
    }.issubset(kinds)
    assert graph.edges_by_kind(ContractEdgeKind.IMPORTS)
    assert graph.edges_by_kind(ContractEdgeKind.CALLS)
    unresolved_edges = [
        edge
        for edge in graph.edges
        if edge.authority is ContractAuthority.UNRESOLVED
    ]
    assert unresolved_edges
    assert all(not edge.mandatory for edge in unresolved_edges)
    assert graph.to_code_evidence_graph().nodes
    projection = graph.to_datasets_projection()
    assert projection["authority"] == ContractAuthority.CONTEXT_ONLY.value
    assert projection["graph_root"] == graph.root_id


def test_unhealthy_repository_index_and_body_metadata_fail_closed() -> None:
    index = _repository_index()
    unhealthy = _Index(
        rows=index.rows,
        ast_index=index.ast_index,
        safe_for_completion_reasoning=False,
    )
    with pytest.raises(GraphIntegrityError, match="not healthy"):
        project_repository_index(unhealthy)

    with pytest.raises(GraphIntegrityError, match="body key"):
        _node(
            ContractNodeKind.FILE,
            "unsafe",
            source_body="never retain this",
        )
