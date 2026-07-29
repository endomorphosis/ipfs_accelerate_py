"""Real indexed graph GraphRAG retrieval (SCA-605 / SCAEV179INDEXGRAPH)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.contract_assurance_baseline import (
    BaselineStageName,
    CONTRACT_ASSURANCE_INDEXGRAPH_EVIDENCE,
    materialize_contract_assurance_baseline,
)
from ipfs_accelerate_py.agent_supervisor.analysis.symbolic_contract_graph import (
    GRAPH_VERSION,
    BoundedGraphRAGRetriever,
    ContractAuthority,
    ContractEdgeKind,
    ContractGraphEdge,
    ContractGraphNode,
    ContractNodeKind,
    ContractProvenance,
    ExactDatasetsGraphProviderError,
    RetrievalBounds,
    SymbolicContractGraph,
)


SNAPSHOT = "repository-snapshot:sha256:real-graph-fixture"


def _node(
    key: str,
    *,
    kind: ContractNodeKind = ContractNodeKind.TOOL,
    payload: dict | None = None,
) -> ContractGraphNode:
    return ContractGraphNode(
        kind=kind,
        stable_key=key,
        snapshot_id=SNAPSHOT,
        provenance=ContractProvenance.REGISTRY,
        authority=ContractAuthority.SOURCE_OBSERVATION,
        version=GRAPH_VERSION,
        payload=payload or {"label": key, "role": "mcp tool registration"},
        source_refs=("registration:fixture",),
    )


def _edge(
    source: ContractGraphNode,
    target: ContractGraphNode,
    *,
    kind: ContractEdgeKind = ContractEdgeKind.DECLARES,
) -> ContractGraphEdge:
    return ContractGraphEdge(
        kind=kind,
        source=source.node_id,
        target=target.node_id,
        snapshot_id=SNAPSHOT,
        provenance=ContractProvenance.REGISTRY,
        authority=ContractAuthority.SOURCE_OBSERVATION,
        version=GRAPH_VERSION,
        mandatory=False,
        source_refs=("registration:fixture",),
    )


def _real_graph() -> SymbolicContractGraph:
    package = _node(
        "package:ipfs_accelerate_py",
        kind=ContractNodeKind.MODULE,
        payload={"name": "ipfs_accelerate_py", "role": "provider package"},
    )
    tool = _node(
        "tool:demo.echo",
        kind=ContractNodeKind.TOOL,
        payload={"name": "demo.echo", "role": "mcp tool registration"},
    )
    handler = _node(
        "handler:demo.echo",
        kind=ContractNodeKind.HANDLER,
        payload={"name": "echo", "role": "handler surface"},
    )
    return SymbolicContractGraph(
        snapshot_id=SNAPSHOT,
        nodes=(package, tool, handler),
        edges=(
            _edge(package, tool, kind=ContractEdgeKind.DECLARES),
            _edge(tool, handler, kind=ContractEdgeKind.HANDLED_BY),
        ),
    )


def test_local_graphrag_queries_real_graph_not_canary_nodes() -> None:
    graph = _real_graph()
    retriever = BoundedGraphRAGRetriever(graph)
    receipt = retriever.retrieve(
        "mcp tool registration",
        bounds=RetrievalBounds(max_candidates=8, max_bytes=32_000, max_query_bytes=1024),
    )
    assert receipt.graph_root == graph.graph_root
    assert receipt.snapshot_id == graph.snapshot_id
    assert receipt.non_authoritative is True
    assert receipt.safe_for_proof is False
    graph_ids = {node.node_id for node in graph.nodes}
    assert receipt.candidate_node_ids
    assert set(receipt.candidate_node_ids).issubset(graph_ids)
    assert not any("canary" in node_id for node_id in receipt.candidate_node_ids)
    assert len(receipt.candidates) <= 8


def test_graphrag_results_are_capped_by_policy_bounds() -> None:
    graph = _real_graph()
    retriever = BoundedGraphRAGRetriever(graph)
    receipt = retriever.retrieve(
        "demo echo package handler",
        bounds=RetrievalBounds(max_candidates=1, max_bytes=32_000, max_query_bytes=1024),
    )
    assert len(receipt.candidates) <= 1
    if receipt.total_matches > 1:
        assert receipt.truncated is True


def test_require_exact_datasets_fails_closed_when_unavailable() -> None:
    graph = _real_graph()
    retriever = BoundedGraphRAGRetriever(graph)

    def _boom(_name: str) -> object:
        raise ImportError("datasets graphrag unavailable in fixture")

    retriever._exact_datasets_importer = _boom  # type: ignore[attr-defined]
    with pytest.raises((ExactDatasetsGraphProviderError, ImportError, Exception)):
        retriever.retrieve(
            "mcp tool",
            use_exact_datasets=True,
            require_exact_datasets=True,
            bounds=RetrievalBounds(
                max_candidates=4, max_bytes=16_000, max_query_bytes=512
            ),
        )


def test_baseline_projects_real_graph_through_graphrag_stage() -> None:
    graph = _real_graph()
    baseline = materialize_contract_assurance_baseline(
        snapshot_id=graph.snapshot_id,
        graph=graph,
        snapshot={
            "snapshot_id": graph.snapshot_id,
            "scope_policy_id": "policy-fixture",
            "head_tree_id": "tree-fixture",
            "stats": {"tracked_path_count": 1, "disposition_count": 1},
        },
        extract_expected=False,
        project_graph=False,
        run_traces=False,
        run_parity=False,
        run_mismatch=False,
        run_vulnerability=False,
        run_graphrag=True,
        require_actual_package_surfaces=False,
        assess_surface_health=False,
        graphrag_query="mcp tool registration",
    )
    assert baseline.llm_call_count == 0
    stage = next(
        item for item in baseline.stages if item.name is BaselineStageName.GRAPHRAG
    )
    assert stage.completeness.value in {"complete", "partial"}
    assert stage.details.get("non_authoritative") is True
    assert stage.details.get("proof_authority") is False
    assert stage.details.get("canary_fixed_nodes") is False
    assert stage.details.get("graph_root") == graph.graph_root
    assert baseline.findings["index_graph"]["evidence"] == (
        CONTRACT_ASSURANCE_INDEXGRAPH_EVIDENCE
    )
    assert baseline.findings["index_graph"]["graphrag_non_authoritative"] is True
