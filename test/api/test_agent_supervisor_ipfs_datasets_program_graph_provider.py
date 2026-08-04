"""Tests for the bounded ipfs_datasets_py GraphRAG/IPLD projection provider."""

from __future__ import annotations

import types
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.ipfs_datasets_program_graph_provider import (
    IPFS_DATASETS_PROGRAM_GRAPH_PROVIDER_ID,
    CapabilityHealth,
    GraphProjectionBounds,
    GraphProjectionPolicy,
    GraphProjectionQuery,
    IpfsDatasetsProgramGraphProvider,
    ProjectionMode,
    ProjectionStatus,
    ReasonCode,
    inspect_program_graph_provider_capability,
    project_program_graph_local,
    rank_projection_local,
)
from ipfs_accelerate_py.agent_supervisor.program_graph import (
    ProgramEdgeKind,
    ProgramNodeKind,
    ResolverStatus,
    SourceSpan,
    build_program_graph,
    make_edge,
    make_node,
)


FOREST_ID = "forest:test-vfs-012"
PRODUCER = "program-ast-adapter@1"
BLOB_A = "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
BLOB_B = "baguqeerbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _span(line: int = 1, col: int = 0) -> SourceSpan:
    return SourceSpan(
        line_start=line, column_start=col, line_end=line, column_end=col + 4
    )


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
    )


def _edge(
    source: str,
    target: str,
    kind: ProgramEdgeKind | str,
    *,
    component_id: str = "module:pkg.mod",
    blob_cid: str = BLOB_A,
    forest_id: str = FOREST_ID,
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
        resolver_status=ResolverStatus.RESOLVED_STATIC,
    )


def _fixture_graph() -> Any:
    module = _node(
        ProgramNodeKind.MODULE,
        "module:pkg.mod",
        component_id="module:pkg.mod",
        path="pkg/mod.py",
        qualified_name="pkg.mod",
    )
    entry = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:pkg.mod.entry",
        component_id="module:pkg.mod",
        path="pkg/mod.py",
        qualified_name="pkg.mod.entry",
    )
    helper = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:pkg.mod.helper",
        component_id="module:pkg.mod",
        path="pkg/mod.py",
        qualified_name="pkg.mod.helper",
    )
    other = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:pkg.other.run",
        component_id="module:pkg.other",
        path="pkg/other.py",
        qualified_name="pkg.other.run",
        blob_cid=BLOB_B,
    )
    call = _node(
        ProgramNodeKind.CALL,
        "call:pkg.mod.entry->helper",
        component_id="module:pkg.mod",
        path="pkg/mod.py",
        qualified_name="pkg.mod.helper",
    )
    nodes = (module, entry, helper, other, call)
    edges = (
        _edge(module.node_id, entry.node_id, ProgramEdgeKind.DEFINES),
        _edge(module.node_id, helper.node_id, ProgramEdgeKind.DEFINES),
        _edge(entry.node_id, call.node_id, ProgramEdgeKind.CONTAINS),
        _edge(call.node_id, helper.node_id, ProgramEdgeKind.CALLS),
    )
    return build_program_graph(
        forest_id=FOREST_ID,
        nodes=nodes,
        edges=edges,
        producer="program-graph@1",
    )


class _UnavailableImporter:
    def __call__(self, name: str) -> Any:
        raise ModuleNotFoundError(name)


class _IncompatibleBackend:
    """Exposes a query method without any bound parameters."""

    def query(self, query_text: str):  # noqa: ANN001
        return [{"node_id": "invented", "score": 1.0}]


class _UnboundedEngine:
    def query(self, query_text: str, include_everything: bool = True):  # noqa: ANN001
        return []


class _PartialModule(types.ModuleType):
    """Importable module without a compatible query surface."""

    def __init__(self, name: str = "partial") -> None:
        super().__init__(name)
        self.helper = object()


class _HealthyBackend:
    def __init__(self, hits: list[dict[str, Any]] | None = None) -> None:
        self.hits = hits
        self.calls: list[dict[str, Any]] = []

    def query(
        self,
        query_text: str = "",
        *,
        top_k: int = 10,
        max_graph_hops: int = 2,
        max_nodes_visited: int | None = None,
        max_edges_traversed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "query_text": query_text,
                "top_k": top_k,
                "max_graph_hops": max_graph_hops,
                "max_nodes_visited": max_nodes_visited,
                "max_edges_traversed": max_edges_traversed,
                **kwargs,
            }
        )
        if self.hits is not None:
            return self.hits
        return []


class _PoisonBackend(_HealthyBackend):
    """Healthy-shaped surface that returns poisoned ranking payloads."""

    def __init__(self, kind: str = "unknown_node") -> None:
        super().__init__()
        self.kind = kind

    def query(
        self,
        query_text: str = "",
        *,
        top_k: int = 10,
        max_graph_hops: int = 2,
        max_nodes_visited: int | None = None,
        max_edges_traversed: int | None = None,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(
            {
                "query_text": query_text,
                "top_k": top_k,
                "max_graph_hops": max_graph_hops,
                "max_nodes_visited": max_nodes_visited,
                "max_edges_traversed": max_edges_traversed,
                **kwargs,
            }
        )
        if self.kind == "unknown_node":
            return [
                {
                    "node_id": "symbol:invented.by.graphrag",
                    "score": 0.99,
                    "ranking_reason": "fabricated",
                }
            ]
        if self.kind == "forged_chunk":
            graph = _fixture_graph()
            # Real node, forged chunk CID.
            node = next(n for n in graph.nodes if n.kind is ProgramNodeKind.SYMBOL)
            return [
                {
                    "node_id": node.node_id,
                    "chunk_cid": "baguqeerpoisonedchunkcid00000000000000000000000000000000",
                    "score": 0.5,
                }
            ]
        if self.kind == "authority":
            return {
                "results": [
                    {
                        "node_id": next(
                            n.node_id
                            for n in _fixture_graph().nodes
                            if n.kind is ProgramNodeKind.SYMBOL
                        ),
                        "score": 0.5,
                    }
                ],
                "completion_authority": True,
            }
        if self.kind == "forbidden_field":
            return [
                {
                    "node_id": next(
                        n.node_id
                        for n in _fixture_graph().nodes
                        if n.kind is ProgramNodeKind.SYMBOL
                    ),
                    "score": 0.5,
                    "source": "SECRET_SOURCE_BODY",
                }
            ]
        if self.kind == "new_call":
            return [
                {
                    "node_id": "call:fabricated",
                    "kind": "call",
                    "score": 1.0,
                    "creates_calls": True,
                }
            ]
        raise AssertionError(f"unknown poison kind {self.kind}")

# ---------------------------------------------------------------------------
# Capability / inspection
# ---------------------------------------------------------------------------


def test_inspect_capability_never_imports() -> None:
    calls: list[str] = []

    def importer(name: str) -> Any:
        calls.append(name)
        raise AssertionError("inspect must not import")

    provider = IpfsDatasetsProgramGraphProvider(importer=importer)
    cap = provider.capabilities()

    assert calls == []
    assert cap.health is CapabilityHealth.LAZY
    assert cap.non_authoritative is True if hasattr(cap, "non_authoritative") else True
    payload = cap.to_dict()
    assert payload["completion_authority"] is False
    assert payload["mutation_authority"] is False
    assert payload["creates_calls"] is False
    assert payload["creates_proofs"] is False
    assert payload["provider_id"] == IPFS_DATASETS_PROGRAM_GRAPH_PROVIDER_ID
    assert payload["lazy_import"] is True


def test_disabled_provider_is_explicit() -> None:
    provider = IpfsDatasetsProgramGraphProvider(
        GraphProjectionPolicy(enabled=False),
        importer=_UnavailableImporter(),
    )
    cap = provider.probe()
    assert cap.health is CapabilityHealth.UNAVAILABLE
    assert cap.reason_code is ReasonCode.PROVIDER_DISABLED

    graph = _fixture_graph()
    projection = provider.project(graph)
    assert projection.status is ProjectionStatus.DISABLED
    assert projection.chunks  # still emits local projection for ranking data


# ---------------------------------------------------------------------------
# Unavailable
# ---------------------------------------------------------------------------


def test_unavailable_optional_module_uses_local_fallback() -> None:
    provider = IpfsDatasetsProgramGraphProvider(importer=_UnavailableImporter())
    cap = provider.probe()
    assert cap.health is CapabilityHealth.UNAVAILABLE
    assert cap.reason_code is ReasonCode.OPTIONAL_MODULE_UNAVAILABLE
    assert cap.mode is ProjectionMode.LOCAL_FALLBACK

    graph = _fixture_graph()
    projection = provider.project(graph, probe=True)
    assert projection.status is ProjectionStatus.LOCAL_FALLBACK
    assert projection.mode is ProjectionMode.LOCAL_FALLBACK
    assert projection.chunks
    assert all(chunk.chunk_cid.startswith("b") for chunk in projection.chunks)
    assert projection.provenance_links
    assert projection.to_dict()["completion_authority"] is False

    result = provider.query(graph, "pkg.mod.entry", projection=projection)
    assert result.status is ProjectionStatus.LOCAL_FALLBACK
    assert result.reason_code is ReasonCode.LOCAL_FALLBACK_QUERY
    assert result.references
    assert result.safe_for_completion_reasoning is False
    assert result.to_dict()["mutation_authority"] is False
    assert result.to_dict()["creates_findings"] is False


def test_unavailable_without_fallback_is_explicit() -> None:
    provider = IpfsDatasetsProgramGraphProvider(
        GraphProjectionPolicy(allow_local_fallback=False),
        importer=_UnavailableImporter(),
    )
    graph = _fixture_graph()
    projection = provider.project(graph)
    assert projection.status is ProjectionStatus.UNAVAILABLE
    assert projection.reason_code is ReasonCode.OPTIONAL_MODULE_UNAVAILABLE

    result = provider.query(graph, "entry", projection=projection, use_backend=False)
    assert result.status is ProjectionStatus.UNAVAILABLE


# ---------------------------------------------------------------------------
# Incompatible
# ---------------------------------------------------------------------------


def test_incompatible_unbounded_query_surface() -> None:
    provider = IpfsDatasetsProgramGraphProvider(backend=_IncompatibleBackend())
    cap = provider.probe()
    assert cap.health is CapabilityHealth.INCOMPATIBLE
    assert cap.reason_code is ReasonCode.OPTIONAL_API_INCOMPATIBLE

    graph = _fixture_graph()
    projection = provider.project(graph)
    assert projection.status is ProjectionStatus.LOCAL_FALLBACK
    assert "incompatible" in projection.reason.casefold()

    # Even with an incompatible backend, local ranking must still work.
    result = provider.query(graph, "helper", projection=projection, use_backend=False)
    assert result.references
    assert result.to_dict()["creates_contracts"] is False


def test_incompatible_module_import_without_query_api() -> None:
    partial = _PartialModule("ipfs_datasets_py.search.graph_query")

    def importer(name: str) -> Any:
        if name.endswith("graph_query") or name.endswith("knowledge_graphs"):
            return partial
        raise ModuleNotFoundError(name)

    provider = IpfsDatasetsProgramGraphProvider(importer=importer)
    cap = provider.probe()
    # Modules imported, no bounded query -> partial (not fully incompatible).
    assert cap.health in {CapabilityHealth.PARTIAL, CapabilityHealth.INCOMPATIBLE}
    assert cap.imported is True


# ---------------------------------------------------------------------------
# Partial
# ---------------------------------------------------------------------------


def test_partial_probe_status_and_truncation() -> None:
    partial = _PartialModule("ipfs_datasets_py.knowledge_graphs")

    def importer(name: str) -> Any:
        if "knowledge_graphs" in name or "graphrag" in name or "graph_query" in name:
            return partial
        raise ModuleNotFoundError(name)

    provider = IpfsDatasetsProgramGraphProvider(importer=importer)
    cap = provider.probe()
    assert cap.health is CapabilityHealth.PARTIAL
    assert cap.reason_code is ReasonCode.OPTIONAL_API_PARTIAL

    graph = _fixture_graph()
    # Force truncation via tiny item bound.
    tight = GraphProjectionBounds(
        max_items=2,
        max_results=1,
        max_depth=2,
        max_hops=1,
        max_bytes=64 * 1024,
        max_query_bytes=4096,
        max_chunk_nodes=64,
        max_chunk_edges=128,
        max_chunk_count=1,
        max_provenance_links=16,
        timeout_ms=1000,
    )
    projection = provider.project(graph, bounds=tight)
    assert projection.truncated or projection.status in {
        ProjectionStatus.PARTIAL,
        ProjectionStatus.LOCAL_FALLBACK,
    }
    # Query with max_results=1 forces partial ranking truncation.
    result = provider.query(
        graph,
        GraphProjectionQuery(text="pkg", max_results=1),
        projection=projection,
        bounds=tight,
        use_backend=False,
    )
    assert len(result.references) <= 1
    if result.truncated:
        assert result.status is ProjectionStatus.PARTIAL
        assert result.truncation_reason


def test_partial_backend_empty_falls_back_to_local_ranking() -> None:
    backend = _HealthyBackend(hits=[])
    provider = IpfsDatasetsProgramGraphProvider(backend=backend)
    graph = _fixture_graph()
    projection = provider.project(graph)
    assert projection.capability is not None
    assert projection.capability.health is CapabilityHealth.HEALTHY

    result = provider.query(graph, "pkg.mod.entry", projection=projection)
    # Empty backend hits -> local ranking still returns references.
    assert result.references
    assert result.mode is ProjectionMode.LOCAL_FALLBACK
    assert result.to_dict()["completion_authority"] is False


# ---------------------------------------------------------------------------
# Poisoned results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    ["unknown_node", "forged_chunk", "authority", "forbidden_field", "new_call"],
)
def test_poisoned_backend_results_are_rejected(kind: str) -> None:
    provider = IpfsDatasetsProgramGraphProvider(backend=_PoisonBackend(kind=kind))
    graph = _fixture_graph()
    projection = provider.project(graph)
    assert projection.capability is not None
    assert projection.capability.health is CapabilityHealth.HEALTHY

    result = provider.query(graph, "entry", projection=projection, use_backend=True)
    assert result.status is ProjectionStatus.POISONED
    assert result.reason_code is ReasonCode.POISONED_BACKEND_RESULT
    assert result.references == ()
    assert result.safe_for_completion_reasoning is False
    payload = result.to_dict()
    assert payload["creates_calls"] is False
    assert payload["creates_proofs"] is False
    assert payload["completion_authority"] is False
    assert payload["mutation_authority"] is False


def test_healthy_backend_ranks_only_canonical_evidence() -> None:
    graph = _fixture_graph()
    entry = next(
        n for n in graph.nodes if n.qualified_name == "pkg.mod.entry"
    )
    helper = next(
        n for n in graph.nodes if n.qualified_name == "pkg.mod.helper"
    )
    backend = _HealthyBackend(
        hits=[
            {"node_id": helper.node_id, "score": 0.4, "ranking_reason": "graph"},
            {"node_id": entry.node_id, "score": 0.9, "ranking_reason": "vector"},
        ]
    )
    provider = IpfsDatasetsProgramGraphProvider(backend=backend)
    projection = provider.project(graph)
    result = provider.query(graph, "entry", projection=projection, use_backend=True)

    assert result.status is ProjectionStatus.COMPLETED
    assert result.mode is ProjectionMode.GRAPHRAG
    assert result.ranking_method == "backend_filtered_canonical"
    assert [ref.node_id for ref in result.references] == [
        entry.node_id,
        helper.node_id,
    ]
    assert result.references[0].rank == 0
    assert result.references[0].score >= result.references[1].score
    assert result.references[0].ranking_reason.startswith("backend:")
    # Backend must receive bound parameters.
    assert backend.calls
    assert backend.calls[0]["top_k"] >= 1
    assert "max_graph_hops" in backend.calls[0]


# ---------------------------------------------------------------------------
# Deterministic query / projection
# ---------------------------------------------------------------------------


def test_local_projection_chunk_cids_are_deterministic() -> None:
    graph = _fixture_graph()
    first = project_program_graph_local(graph)
    second = project_program_graph_local(graph)

    assert first.projection_id == second.projection_id
    assert first.index_cid == second.index_cid
    assert [c.chunk_cid for c in first.chunks] == [
        c.chunk_cid for c in second.chunks
    ]
    assert [c.chunk_id for c in first.chunks] == [
        c.chunk_id for c in second.chunks
    ]
    # Provenance links bind chunk CIDs to blob CIDs and graph identity.
    kinds = {link.kind for link in first.provenance_links}
    assert "projects_blob" in kinds
    assert "derived_from_graph" in kinds
    assert all(link.source_cid for link in first.provenance_links)
    assert all(link.target_cid for link in first.provenance_links)


def test_deterministic_query_ranking_is_stable() -> None:
    graph = _fixture_graph()
    provider = IpfsDatasetsProgramGraphProvider(importer=_UnavailableImporter())
    projection = provider.project(graph)

    q = GraphProjectionQuery(text="pkg.mod.entry helper")
    first = provider.query(graph, q, projection=projection, use_backend=False)
    second = provider.query(graph, q, projection=projection, use_backend=False)

    assert first.result_id == second.result_id
    assert first.query_id == second.query_id
    assert [r.to_dict() for r in first.references] == [
        r.to_dict() for r in second.references
    ]
    assert first.references
    for ref in first.references:
        assert ref.ranking_reason
        assert ref.chunk_cid
        assert ref.score >= 0.0
        assert ref.node_id in projection.allowed_node_ids()


def test_query_seed_and_hop_expansion_stays_in_projection() -> None:
    graph = _fixture_graph()
    projection = project_program_graph_local(graph)
    entry = next(
        n for n in graph.nodes if n.qualified_name == "pkg.mod.entry"
    )
    result = rank_projection_local(
        graph,
        projection,
        GraphProjectionQuery(
            text="",
            seed_node_ids=(entry.node_id,),
            max_hops=2,
            max_results=16,
        ),
    )
    allowed = projection.allowed_node_ids()
    assert result.references
    for ref in result.references:
        assert ref.node_id in allowed
        assert ref.hop_distance <= 2
        assert "seed_node" in ref.ranking_reason or ref.hop_distance > 0


def test_query_never_ranks_nodes_outside_projection() -> None:
    graph = _fixture_graph()
    # Project with max_chunk_count=1 so one component is dropped.
    bounds = GraphProjectionBounds(
        max_items=256,
        max_results=32,
        max_depth=3,
        max_hops=2,
        max_bytes=128 * 1024,
        max_query_bytes=16 * 1024,
        max_chunk_nodes=64,
        max_chunk_edges=128,
        max_chunk_count=1,
        max_provenance_links=256,
        timeout_ms=2000,
    )
    projection = project_program_graph_local(graph, bounds=bounds)
    assert len(projection.chunks) == 1
    result = rank_projection_local(
        graph,
        projection,
        GraphProjectionQuery(text="pkg"),
        bounds=bounds,
    )
    allowed = projection.allowed_node_ids()
    for ref in result.references:
        assert ref.node_id in allowed
    # The dropped component must not appear.
    dropped = {
        n.node_id
        for n in graph.nodes
        if n.node_id not in allowed
    }
    assert dropped
    assert not any(ref.node_id in dropped for ref in result.references)


def test_authority_claims_are_immutable_on_all_results() -> None:
    provider = IpfsDatasetsProgramGraphProvider(importer=_UnavailableImporter())
    graph = _fixture_graph()
    projection, result = provider.project_and_query(graph, "entry")

    for payload in (projection.to_dict(), result.to_dict()):
        assert payload["creates_calls"] is False
        assert payload["creates_contracts"] is False
        assert payload["creates_findings"] is False
        assert payload["creates_proofs"] is False
        assert payload["completion_authority"] is False
        assert payload["mutation_authority"] is False
        assert payload["ranking_only"] is True
        assert payload["canonical_evidence_only"] is True
        assert payload["non_authoritative"] is True


def test_bounds_validation_fails_closed() -> None:
    with pytest.raises(Exception):
        GraphProjectionBounds(max_results=100, max_items=10)
    with pytest.raises(Exception):
        GraphProjectionBounds(max_hops=5, max_depth=2)
    with pytest.raises(Exception):
        GraphProjectionQuery(text="")  # needs text or seeds


def test_query_input_order_does_not_affect_result_identity() -> None:
    graph = _fixture_graph()
    # Rebuild graph with nodes/edges shuffled.
    nodes = list(graph.nodes)
    edges = list(graph.edges)
    shuffled = build_program_graph(
        forest_id=graph.forest_id,
        nodes=list(reversed(nodes)),
        edges=list(reversed(edges)),
        producer=graph.producer,
    )
    provider = IpfsDatasetsProgramGraphProvider(importer=_UnavailableImporter())
    a = provider.project_and_query(graph, "pkg.mod")
    b = provider.project_and_query(shuffled, "pkg.mod")
    assert a[0].graph_id == b[0].graph_id
    assert a[0].projection_id == b[0].projection_id
    assert a[1].result_id == b[1].result_id


def test_inspect_helper_matches_provider_capabilities() -> None:
    policy = GraphProjectionPolicy(
        bounds=GraphProjectionBounds(max_results=8, max_items=32)
    )
    declared = inspect_program_graph_provider_capability(policy)
    provider = IpfsDatasetsProgramGraphProvider(policy)
    assert provider.capabilities().capability_id == declared.capability_id


def test_chunk_cids_change_when_evidence_changes() -> None:
    graph = _fixture_graph()
    base = project_program_graph_local(graph)
    # Add a new symbol in a new component.
    extra = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:pkg.extra.f",
        component_id="module:pkg.extra",
        path="pkg/extra.py",
        qualified_name="pkg.extra.f",
    )
    extended = build_program_graph(
        forest_id=FOREST_ID,
        nodes=list(graph.nodes) + [extra],
        edges=list(graph.edges),
        producer=graph.producer,
    )
    other = project_program_graph_local(extended)
    assert base.projection_id != other.projection_id
    assert base.index_cid != other.index_cid
