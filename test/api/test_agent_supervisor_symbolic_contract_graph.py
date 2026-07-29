from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.symbolic_contract_graph import (
    BoundedGraphRAGRetriever,
    ClosureDirection,
    ContractAuthority,
    ContractClosureBounds,
    ContractEdgeKind,
    ContractGraphClosure,
    ContractGraphBoundsError,
    ContractNodeKind,
    ContractProvenance,
    GraphRAGRetrievalBounds,
    GraphRAGRetrievalReceipt,
    MandatoryEdgeRequirement,
    MissingMandatoryEdgeError,
    RetrievalStatus,
    SymbolicContractEdge,
    SymbolicContractGraph,
    SymbolicContractGraphError,
    SymbolicContractNode,
    TruncatedContractClosureError,
    build_symbolic_contract_graph,
)


SNAPSHOT = "repository-snapshot:sha256:fixture"
VERSION = "fixture-contract-producer@1"


def _node(
    node_id: str,
    kind: ContractNodeKind,
    *,
    provenance: ContractProvenance = ContractProvenance.AST,
    authority: ContractAuthority = ContractAuthority.OBSERVATION,
    record: dict[str, object] | None = None,
) -> SymbolicContractNode:
    return SymbolicContractNode(
        node_id=node_id,
        kind=kind,
        snapshot_id=SNAPSHOT,
        provenance=provenance,
        provenance_id=f"fixture-record:{node_id}",
        authority=authority,
        version=VERSION,
        record=record or {"name": node_id},
    )


def _edge(
    source: str,
    target: str,
    kind: ContractEdgeKind,
    *,
    provenance: ContractProvenance = ContractProvenance.AST,
    authority: ContractAuthority = ContractAuthority.OBSERVATION,
    mandatory: bool = True,
) -> SymbolicContractEdge:
    return SymbolicContractEdge(
        source=source,
        target=target,
        kind=kind,
        snapshot_id=SNAPSHOT,
        provenance=provenance,
        provenance_id=f"fixture-edge:{source}:{kind.value}:{target}",
        authority=authority,
        version=VERSION,
        mandatory=mandatory,
    )


def _graph(
    *,
    context: bool = True,
    requirements: tuple[MandatoryEdgeRequirement, ...] = (),
) -> SymbolicContractGraph:
    nodes = (
        _node("file:client", ContractNodeKind.FILE),
        _node("module:client", ContractNodeKind.MODULE),
        _node("tool:add", ContractNodeKind.TOOL),
        _node("schema:add", ContractNodeKind.SCHEMA),
        _node("handler:add", ContractNodeKind.HANDLER),
        _node("policy:write", ContractNodeKind.POLICY),
        _node("transport:stdio", ContractNodeKind.TRANSPORT),
        _node("test:add", ContractNodeKind.TEST),
        *(
            (
                _node(
                    "context:similar",
                    ContractNodeKind.PROVENANCE,
                    provenance=ContractProvenance.GRAPHRAG,
                    authority=ContractAuthority.CONTEXT_ONLY,
                ),
            )
            if context
            else ()
        ),
    )
    edges = (
        _edge("file:client", "module:client", ContractEdgeKind.CONTAINS),
        _edge("module:client", "tool:add", ContractEdgeKind.EXPOSES),
        _edge("tool:add", "schema:add", ContractEdgeKind.BINDS_SCHEMA),
        _edge("tool:add", "handler:add", ContractEdgeKind.HANDLED_BY),
        _edge("handler:add", "policy:write", ContractEdgeKind.GOVERNED_BY),
        _edge(
            "handler:add",
            "transport:stdio",
            ContractEdgeKind.TRANSPORTED_BY,
        ),
        _edge("test:add", "tool:add", ContractEdgeKind.TESTS),
        *(
            (
                _edge(
                    "context:similar",
                    "tool:add",
                    ContractEdgeKind.NOMINATES,
                    provenance=ContractProvenance.GRAPHRAG,
                    authority=ContractAuthority.CONTEXT_ONLY,
                    mandatory=False,
                ),
            )
            if context
            else ()
        ),
    )
    return SymbolicContractGraph(
        snapshot_id=SNAPSHOT,
        nodes=nodes,
        edges=edges,
        mandatory_edge_requirements=requirements,
    )


def test_every_node_and_edge_has_content_identity_and_authority_bindings() -> None:
    graph = _graph()

    for item in (*graph.nodes, *graph.edges):
        record = item.to_dict()
        assert record["content_id"].startswith("b")
        assert record["identity"]["cid"] == record["content_id"]
        assert record["identity"]["validated"] is True
        assert record["identity"]["profile"] == "strict-dag-json-v1"
        assert record["identity"]["digest"].startswith("sha256:")
        assert record["snapshot_id"] == SNAPSHOT
        assert record["provenance"]
        assert record["provenance_id"]
        assert record["authority"]
        assert record["version"] == VERSION

    rendered = graph.to_dict()
    assert rendered["identity"]["cid"] == graph.graph_root
    assert rendered["identity"]["validated"] is True
    assert rendered["node_count"] == len(graph.nodes)
    assert rendered["edge_count"] == len(graph.edges)


def test_graph_root_and_serialization_are_stable_under_input_reordering() -> None:
    graph = _graph()
    rebuilt = build_symbolic_contract_graph(
        snapshot_id=SNAPSHOT,
        nodes=reversed(graph.nodes),
        edges=reversed(graph.edges),
        mandatory_edge_requirements=reversed(
            graph.mandatory_edge_requirements
        ),
    )

    assert rebuilt.graph_root == graph.graph_root
    assert rebuilt.to_json() == graph.to_json()
    assert SymbolicContractGraph.from_json(graph.to_json()) == graph

    changed = replace(
        graph,
        nodes=tuple(
            replace(node, version="fixture-contract-producer@2")
            if node.node_id == "tool:add"
            else node
            for node in graph.nodes
        ),
    )
    assert changed.graph_root != graph.graph_root


def test_graph_covers_required_contract_node_and_edge_families() -> None:
    assert {
        "file",
        "module",
        "symbol",
        "call",
        "import",
        "effect",
        "schema",
        "tool",
        "handler",
        "test",
        "policy",
        "transport",
        "provenance",
    }.issubset({item.value for item in ContractNodeKind})
    assert {
        "contains",
        "defines",
        "imports",
        "calls",
        "has_effect",
        "binds_schema",
        "handled_by",
        "governed_by",
        "transported_by",
        "depends_on",
        "derived_from",
    }.issubset({item.value for item in ContractEdgeKind})


def test_forward_and_reverse_mandatory_closures_are_exact() -> None:
    graph = _graph()
    forward = graph.forward_closure("tool:add")
    reverse = graph.reverse_closure("tool:add")

    assert forward.direction is ClosureDirection.FORWARD
    assert set(forward.node_ids) == {
        "tool:add",
        "schema:add",
        "handler:add",
        "policy:write",
        "transport:stdio",
    }
    assert forward.paths["policy:write"] == (
        "tool:add",
        "handler:add",
        "policy:write",
    )
    assert set(reverse.node_ids) == {
        "tool:add",
        "module:client",
        "file:client",
        "test:add",
    }
    assert reverse.paths["file:client"] == (
        "tool:add",
        "module:client",
        "file:client",
    )
    assert "context:similar" not in forward.node_ids
    assert "context:similar" not in reverse.node_ids
    assert forward.complete and reverse.complete
    assert not forward.truncated and not reverse.truncated
    assert ContractGraphClosure.from_dict(forward.to_dict()) == forward


def test_graphrag_records_are_context_only_and_cannot_be_mandatory() -> None:
    graph = _graph()
    nomination = graph.edges_by_kind(ContractEdgeKind.NOMINATES)[0]
    assert nomination.authority is ContractAuthority.CONTEXT_ONLY
    assert nomination.authoritative is False
    assert nomination.mandatory is False

    with pytest.raises(SymbolicContractGraphError, match="context_only"):
        _node(
            "forged",
            ContractNodeKind.CLAIM,
            provenance=ContractProvenance.GRAPHRAG,
            authority=ContractAuthority.PROOF_INPUT,
        )
    with pytest.raises(SymbolicContractGraphError, match="cannot be mandatory"):
        _edge(
            "context:similar",
            "tool:add",
            ContractEdgeKind.DEPENDS_ON,
            provenance=ContractProvenance.GRAPHRAG,
            authority=ContractAuthority.CONTEXT_ONLY,
            mandatory=True,
        )


def test_missing_declared_mandatory_edge_fails_closed() -> None:
    missing = MandatoryEdgeRequirement(
        "tool:add", ContractEdgeKind.AUTHORIZED_BY, "policy:missing"
    )
    graph = _graph(requirements=(missing,))

    with pytest.raises(MissingMandatoryEdgeError) as raised:
        graph.forward_closure("tool:add")
    assert raised.value.requirements == (missing,)

    view = BoundedGraphRAGRetriever(graph).retrieve(
        "add", candidate_node_ids=("tool:add",)
    )
    assert view.receipt.status is RetrievalStatus.INCOMPLETE
    assert view.receipt.reason_code == "missing_mandatory_edges"
    assert view.receipt.missing_requirement_ids == (missing.requirement_id,)
    assert view.nodes == ()
    assert view.edges == ()
    assert view.forward_closure is None
    assert view.reverse_closure is None
    assert not view.complete
    with pytest.raises(MissingMandatoryEdgeError):
        view.require_complete()


def test_unknown_endpoint_on_mandatory_edge_is_rejected_at_projection() -> None:
    with pytest.raises(MissingMandatoryEdgeError):
        SymbolicContractGraph(
            snapshot_id=SNAPSHOT,
            nodes=(_node("tool:add", ContractNodeKind.TOOL),),
            edges=(
                _edge(
                    "tool:add",
                    "handler:absent",
                    ContractEdgeKind.HANDLED_BY,
                ),
            ),
        )


@pytest.mark.parametrize(
    ("bounds", "reason"),
    [
        (ContractClosureBounds(max_nodes=2), "max_nodes_exceeded"),
        (ContractClosureBounds(max_edges=1), "max_edges_exceeded"),
        (ContractClosureBounds(max_depth=1), "max_depth_exceeded"),
    ],
)
def test_closure_truncation_raises_instead_of_returning_partial_authority(
    bounds: ContractClosureBounds, reason: str
) -> None:
    graph = _graph()
    with pytest.raises(TruncatedContractClosureError) as raised:
        graph.forward_closure("tool:add", bounds=bounds)
    assert raised.value.reason_code == reason


def test_bounded_retrieval_returns_receipt_and_deterministic_both_way_closure() -> None:
    graph = _graph()
    retriever = BoundedGraphRAGRetriever(
        graph,
        bounds=GraphRAGRetrievalBounds(max_candidates=4),
        repository_id="repository:fixture",
        objective_revision="SCA-G030@1",
    )
    view = retriever.retrieve("add", candidate_node_ids=("tool:add",))

    assert view.complete
    assert view.receipt.status is RetrievalStatus.COMPLETE
    assert view.receipt.reason_code == "complete"
    assert view.receipt.graph_root == graph.graph_root
    assert view.receipt.forward_closure_id == view.forward_closure.closure_id
    assert view.receipt.reverse_closure_id == view.reverse_closure.closure_id
    assert view.receipt.receipt_id.startswith("b")
    assert view.receipt.to_dict()["completion_authority"] is False
    assert view.receipt.to_dict()["proof_authority"] is False
    assert not view.safe_for_completion_reasoning
    assert {item.node_id for item in view.nodes} == {
        "file:client",
        "module:client",
        "tool:add",
        "schema:add",
        "handler:add",
        "policy:write",
        "transport:stdio",
        "test:add",
    }
    assert all(edge.mandatory and edge.authoritative for edge in view.edges)

    repeated = retriever.retrieve("add", candidate_node_ids=("tool:add",))
    assert repeated.receipt.receipt_id == view.receipt.receipt_id
    assert repeated.to_dict() == view.to_dict()
    assert (
        GraphRAGRetrievalReceipt.from_json(view.receipt.to_json())
        == view.receipt
    )

    forged_receipt = view.receipt.to_dict()
    forged_receipt["completion_authority"] = True
    with pytest.raises(SymbolicContractGraphError, match="completion authority"):
        GraphRAGRetrievalReceipt.from_dict(forged_receipt)


def test_candidate_and_closure_bounds_produce_fail_closed_receipts() -> None:
    graph = _graph()
    candidate_limited = BoundedGraphRAGRetriever(
        graph, bounds=GraphRAGRetrievalBounds(max_candidates=1)
    ).retrieve(
        "add",
        candidate_node_ids=("handler:add", "schema:add", "tool:add"),
    )
    assert candidate_limited.truncated
    assert candidate_limited.receipt.reason_code == "candidate_bound_exceeded"
    assert len(candidate_limited.candidates) == 1
    assert candidate_limited.nodes == ()
    with pytest.raises(TruncatedContractClosureError):
        candidate_limited.require_complete()

    closure_limited = BoundedGraphRAGRetriever(
        graph,
        bounds=GraphRAGRetrievalBounds(
            max_candidates=1, max_nodes=2, max_edges=16, max_depth=8
        ),
    ).retrieve("add", candidate_node_ids=("tool:add",))
    assert closure_limited.truncated
    assert closure_limited.receipt.reason_code == "max_nodes_exceeded"
    assert closure_limited.forward_closure is None
    assert closure_limited.reverse_closure is None


def test_query_bound_is_enforced_before_provider_dispatch() -> None:
    calls: list[str] = []
    retriever = BoundedGraphRAGRetriever(
        _graph(),
        bounds=GraphRAGRetrievalBounds(max_query_bytes=4),
        provider_factory=lambda: calls.append("factory"),
    )
    with pytest.raises(ContractGraphBoundsError, match="retrieval query exceeds"):
        retriever.retrieve("oversized", use_optional_provider=True)
    assert calls == []


def test_optional_datasets_provider_remains_lazy_until_explicit_dispatch() -> None:
    calls: list[str] = []

    class Provider:
        def build_request(self, payload):
            calls.append("build_request")
            return payload

        def analyze(self, request):
            calls.append("analyze")
            return SimpleNamespace(
                successful=True,
                status=SimpleNamespace(value="completed"),
                truncated=False,
                result_id="provider-result:fixture",
                evidence_references=(
                    {"record_id": "handler:add", "score_millionths": 950_000},
                ),
            )

    def factory():
        calls.append("factory")
        return Provider()

    retriever = BoundedGraphRAGRetriever(_graph(), provider_factory=factory)
    assert calls == []

    local = retriever.retrieve("add", candidate_node_ids=("tool:add",))
    assert local.complete
    assert local.receipt.provider_requested is False
    assert local.receipt.provider_imported is False
    assert calls == []

    offloaded = retriever.retrieve("add", use_optional_provider=True)
    assert offloaded.complete
    assert offloaded.receipt.provider_requested is True
    assert offloaded.receipt.provider_imported is True
    assert offloaded.receipt.provider_result_id == "provider-result:fixture"
    assert "handler:add" in {item.node_id for item in offloaded.candidates}
    assert calls == ["factory", "build_request", "analyze"]


def test_optional_provider_truncation_is_never_accepted_as_complete() -> None:
    class TruncatedProvider:
        def build_request(self, payload):
            return payload

        def analyze(self, request):
            return SimpleNamespace(
                successful=True,
                status=SimpleNamespace(value="completed"),
                truncated=True,
                result_id="provider-result:truncated",
                evidence_references=({"record_id": "tool:add"},),
            )

    view = BoundedGraphRAGRetriever(
        _graph(), provider=TruncatedProvider()
    ).retrieve("add", use_optional_provider=True)

    assert view.receipt.status is RetrievalStatus.TRUNCATED
    assert view.receipt.reason_code == "optional_provider_truncated"
    assert view.forward_closure is None
    assert view.reverse_closure is None
    assert view.nodes == ()


def test_provider_failure_degrades_to_receipted_local_deterministic_retrieval() -> None:
    def broken_factory():
        raise ModuleNotFoundError("ipfs_datasets_py")

    view = BoundedGraphRAGRetriever(
        _graph(), provider_factory=broken_factory
    ).retrieve(
        "not-present",
        use_optional_provider=True,
        candidate_node_ids=("tool:add",),
    )

    assert view.complete
    assert view.receipt.reason_code == "provider_degraded_local_fallback"
    assert view.receipt.provider_status == "provider_dispatch_failed"
    assert view.receipt.provider_requested
    assert view.receipt.provider_imported
    assert not view.receipt.safe_for_completion_reasoning


def test_deserialization_rejects_forged_identity_authority_and_graph_root() -> None:
    graph = _graph()
    payload = json.loads(graph.to_json())
    next(
        node for node in payload["nodes"] if node["node_id"] == "tool:add"
    )["authority"] = "reviewed_contract"
    with pytest.raises(SymbolicContractGraphError, match="identity mismatch"):
        SymbolicContractGraph.from_dict(payload)

    payload = json.loads(graph.to_json())
    payload["graph_root"] = "bafyfaked"
    with pytest.raises(SymbolicContractGraphError, match="graph root mismatch"):
        SymbolicContractGraph.from_dict(payload)
