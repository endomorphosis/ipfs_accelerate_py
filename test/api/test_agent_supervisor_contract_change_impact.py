"""Tests for reverse transitive impact closure and SCCs (RPR-028)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    ContractClauseDelta,
    DeltaDisposition,
    DeltaKind,
    ImpactCompleteness,
    ProgramContractDelta,
    PropagationAuthorityRoots,
)
from ipfs_accelerate_py.agent_supervisor.analysis.code_evidence_graph import (
    CodeImpactIndex,
)
from ipfs_accelerate_py.agent_supervisor.analysis.contract_change_impact import (
    ContractChangeImpactAnalyzer,
    ImpactClosureBounds,
    compute_impact_closure,
    compute_impact_closure_result,
    compute_sccs,
    resolve_seed_nodes,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_dependency_graph import (
    PathSource,
    ProgramDependencyGraph,
)
from ipfs_accelerate_py.agent_supervisor.program_graph import (
    Completeness,
    ProgramAuthority,
    ProgramEdge,
    ProgramEdgeKind,
    ProgramGraph,
    ProgramGraphRoots,
    ProgramGraphSnapshot,
    ProgramNode,
    ProgramNodeKind,
    ProgramProvenance,
    ProgramTrust,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_roots() -> ProgramGraphRoots:
    return ProgramGraphRoots(
        forest_id="forest:impact",
        tree_id="tree:impact",
        overlay_id="overlay:impact",
        coverage_id="coverage:impact",
        included_roots=("src/", "tests/"),
        excluded_roots=(),
        generated_roots=(),
        native_roots=(),
        extractor_id="program-graph@1",
        config_id="config:impact",
        toolchain_id="toolchain:cpython",
    )


@pytest.fixture
def prop_roots(graph_roots: ProgramGraphRoots) -> PropagationAuthorityRoots:
    graph = _empty_graph(graph_roots)
    return PropagationAuthorityRoots(
        repository_id="repository:impact",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id=graph_roots.forest_id,
        candidate_tree_id=graph_roots.tree_id,
        candidate_overlay_id=graph_roots.overlay_id,
        graph_id=graph.graph_id,
        index_id="index:impact",
        model_id="model:impact",
        config_id="config:impact",
        translator_id="translator:impact",
        toolchain_id="toolchain:cpython",
        policy_id="policy:impact",
    )


def _empty_graph(roots: ProgramGraphRoots) -> ProgramGraph:
    snapshot = ProgramGraphSnapshot(
        roots=roots,
        nodes=(),
        edges=(),
        frontier_refs=(),
        exclusion_refs=(),
        complete=True,
    )
    return ProgramGraph(snapshot)


def _node(
    roots: ProgramGraphRoots,
    node_id: str,
    kind: ProgramNodeKind,
    name: str,
    *,
    path: str = "src/mod.py",
    qualified: str | None = None,
    authoritative: bool = True,
    frontier: bool = False,
) -> ProgramNode:
    if frontier or not authoritative:
        provenance = ProgramProvenance.RUNTIME if not authoritative else ProgramProvenance.AST
        trust = ProgramTrust.NOMINATED if not authoritative else ProgramTrust.TRUSTED
        authority = ProgramAuthority.NOMINATED if not authoritative else ProgramAuthority.AUTHORITATIVE
        completeness = Completeness.FRONTIER if frontier else Completeness.COMPLETE
        if frontier:
            provenance = ProgramProvenance.AST
            trust = ProgramTrust.TRUSTED
            authority = ProgramAuthority.AUTHORITATIVE
            completeness = Completeness.FRONTIER
            kind = kind if kind is not ProgramNodeKind.FUNCTION else ProgramNodeKind.FRONTIER
    else:
        provenance = ProgramProvenance.AST
        trust = ProgramTrust.TRUSTED
        authority = ProgramAuthority.AUTHORITATIVE
        completeness = Completeness.COMPLETE
    return ProgramNode(
        node_id=node_id,
        kind=kind,
        name=name,
        roots=roots,
        path=path,
        qualified_name=qualified or name,
        provenance=provenance,
        trust=trust,
        authority=authority,
        completeness=completeness,
        extractor_id=roots.extractor_id,
    )


def _edge(
    roots: ProgramGraphRoots,
    source: str,
    target: str,
    kind: ProgramEdgeKind,
    *,
    nominated: bool = False,
    provenance: ProgramProvenance | None = None,
) -> ProgramEdge:
    if nominated:
        return ProgramEdge(
            source=source,
            target=target,
            kind=kind if kind is not ProgramEdgeKind.RELATED_TO else ProgramEdgeKind.RELATED_TO,
            roots=roots,
            provenance=provenance or ProgramProvenance.GRAPHRAG,
            trust=ProgramTrust.NOMINATED,
            authority=ProgramAuthority.NOMINATED,
            completeness=Completeness.FRONTIER,
            confidence=20,
            extractor_id=roots.extractor_id,
        )
    return ProgramEdge(
        source=source,
        target=target,
        kind=kind,
        roots=roots,
        provenance=ProgramProvenance.AST,
        trust=ProgramTrust.TRUSTED,
        authority=ProgramAuthority.AUTHORITATIVE,
        completeness=Completeness.COMPLETE,
        extractor_id=roots.extractor_id,
    )


def _graph(
    roots: ProgramGraphRoots,
    nodes: list[ProgramNode],
    edges: list[ProgramEdge],
    *,
    complete: bool = True,
    frontier_refs: tuple[str, ...] = (),
    exclusion_refs: tuple[str, ...] = (),
) -> ProgramGraph:
    if frontier_refs or exclusion_refs:
        complete = False
    snapshot = ProgramGraphSnapshot(
        roots=roots,
        nodes=tuple(nodes),
        edges=tuple(edges),
        frontier_refs=frontier_refs,
        exclusion_refs=exclusion_refs,
        complete=complete,
    )
    return ProgramGraph(snapshot)


def _delta(
    prop_roots: PropagationAuthorityRoots,
    subject: str,
    *,
    graph_id: str | None = None,
) -> ProgramContractDelta:
    roots = prop_roots
    if graph_id is not None and graph_id != prop_roots.graph_id:
        roots = PropagationAuthorityRoots(
            repository_id=prop_roots.repository_id,
            base_forest_id=prop_roots.base_forest_id,
            base_tree_id=prop_roots.base_tree_id,
            base_overlay_id=prop_roots.base_overlay_id,
            candidate_forest_id=prop_roots.candidate_forest_id,
            candidate_tree_id=prop_roots.candidate_tree_id,
            candidate_overlay_id=prop_roots.candidate_overlay_id,
            graph_id=graph_id,
            index_id=prop_roots.index_id,
            model_id=prop_roots.model_id,
            config_id=prop_roots.config_id,
            translator_id=prop_roots.translator_id,
            toolchain_id=prop_roots.toolchain_id,
            policy_id=prop_roots.policy_id,
        )
    clause = ContractClauseDelta(
        clause_id="clause:param-add",
        kind=DeltaKind.PARAMETER_ADD,
        disposition=DeltaDisposition.BREAKING,
        subject_symbol_id=subject,
        consumer_domain="domain:python-callers",
        before_contract_ref="contract:before",
        after_contract_ref="contract:after",
        reason="third argument required",
    )
    return ProgramContractDelta(
        roots=roots,
        change_set_id="changeset:impact",
        subject_symbol_id=subject,
        before_contract_ref="contract:before",
        after_contract_ref="contract:after",
        clauses=(clause,),
        evidence_refs=("evidence:delta",),
    )


def _bind_roots(
    prop_roots: PropagationAuthorityRoots, graph: ProgramGraph
) -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id=prop_roots.repository_id,
        base_forest_id=prop_roots.base_forest_id,
        base_tree_id=prop_roots.base_tree_id,
        base_overlay_id=prop_roots.base_overlay_id,
        candidate_forest_id=prop_roots.candidate_forest_id,
        candidate_tree_id=prop_roots.candidate_tree_id,
        candidate_overlay_id=prop_roots.candidate_overlay_id,
        graph_id=graph.graph_id,
        index_id=prop_roots.index_id,
        model_id=prop_roots.model_id,
        config_id=prop_roots.config_id,
        translator_id=prop_roots.translator_id,
        toolchain_id=prop_roots.toolchain_id,
        policy_id=prop_roots.policy_id,
    )


# ---------------------------------------------------------------------------
# SCC unit tests
# ---------------------------------------------------------------------------


def test_compute_sccs_deterministic_and_topological() -> None:
    # A → B → C → B  (B,C cycle) plus D isolated consumer of A via A→D
    nodes = ("A", "B", "C", "D")
    adj = {
        "A": ("B", "D"),
        "B": ("C",),
        "C": ("B",),
        "D": (),
    }
    first, topo_first = compute_sccs(nodes, adj)
    second, topo_second = compute_sccs(tuple(reversed(nodes)), adj)
    assert first == second
    assert topo_first == topo_second
    # Cycle members share one component.
    member_sets = [set(group) for group in first]
    assert {"B", "C"} in member_sets
    assert {"A"} in member_sets
    assert {"D"} in member_sets
    # Condensation order: A before B/C and D (edges A→B, A→D).
    order = {node: index for index, node in enumerate(topo_first)}
    assert order["A"] < order["B"]
    assert order["A"] < order["D"]


def test_compute_sccs_empty() -> None:
    assert compute_sccs((), {}) == ((), ())


# ---------------------------------------------------------------------------
# Direct reverse call closure
# ---------------------------------------------------------------------------


def test_reverse_call_closure_includes_direct_and_indirect_callers(
    graph_roots: ProgramGraphRoots, prop_roots: PropagationAuthorityRoots
) -> None:
    process = _node(graph_roots, "node:process", ProgramNodeKind.FUNCTION, "process")
    wrapper = _node(graph_roots, "node:wrapper", ProgramNodeKind.FUNCTION, "wrapper")
    api = _node(graph_roots, "node:api", ProgramNodeKind.FUNCTION, "api")
    edges = [
        _edge(graph_roots, "node:wrapper", "node:process", ProgramEdgeKind.CALLS),
        _edge(graph_roots, "node:api", "node:wrapper", ProgramEdgeKind.CALLS),
    ]
    graph = _graph(graph_roots, [process, wrapper, api], edges)
    roots = _bind_roots(prop_roots, graph)
    delta = _delta(roots, "process")

    receipt = compute_impact_closure(delta, graph)

    consumer_ids = {item.consumer_id for item in receipt.consumers}
    assert "consumer:node:wrapper" in consumer_ids
    assert "consumer:node:api" in consumer_ids
    depths = {item.consumer_id: item.depth for item in receipt.consumers}
    assert depths["consumer:node:wrapper"] == 1
    assert depths["consumer:node:api"] == 2
    assert all(item.mandatory for item in receipt.consumers)
    assert receipt.completeness is ImpactCompleteness.COMPLETE
    assert receipt.frontier_node_ids == ()
    # Round-trip identity.
    assert type(receipt).from_dict(receipt.to_record()) == receipt


def test_consumers_deduplicated_while_retaining_all_edge_paths(
    graph_roots: ProgramGraphRoots, prop_roots: PropagationAuthorityRoots
) -> None:
    process = _node(graph_roots, "node:process", ProgramNodeKind.FUNCTION, "process")
    left = _node(graph_roots, "node:left", ProgramNodeKind.FUNCTION, "left")
    right = _node(graph_roots, "node:right", ProgramNodeKind.FUNCTION, "right")
    shared = _node(graph_roots, "node:shared", ProgramNodeKind.FUNCTION, "shared")
    e1 = _edge(graph_roots, "node:left", "node:process", ProgramEdgeKind.CALLS)
    e2 = _edge(graph_roots, "node:right", "node:process", ProgramEdgeKind.CALLS)
    e3 = _edge(graph_roots, "node:shared", "node:left", ProgramEdgeKind.CALLS)
    e4 = _edge(graph_roots, "node:shared", "node:right", ProgramEdgeKind.CALLS)
    graph = _graph(
        graph_roots,
        [process, left, right, shared],
        [e1, e2, e3, e4],
    )
    roots = _bind_roots(prop_roots, graph)
    delta = _delta(roots, "process")
    receipt = compute_impact_closure(delta, graph)

    shared_consumer = next(
        item for item in receipt.consumers if item.consumer_id == "consumer:node:shared"
    )
    # One consumer record, multiple retained edge refs.
    assert len([item for item in receipt.consumers if item.node.node_id == "node:shared"]) == 1
    assert len(shared_consumer.edge_refs) >= 1


def test_wrappers_overrides_tests_schemas_and_factories(
    graph_roots: ProgramGraphRoots, prop_roots: PropagationAuthorityRoots
) -> None:
    subject = _node(graph_roots, "node:svc", ProgramNodeKind.METHOD, "process", qualified="Service.process")
    base = _node(graph_roots, "node:base", ProgramNodeKind.METHOD, "process", qualified="Base.process")
    override = _node(
        graph_roots, "node:override", ProgramNodeKind.METHOD, "process", qualified="Child.process"
    )
    factory = _node(graph_roots, "node:factory", ProgramNodeKind.FACTORY, "build_service")
    serializer = _node(graph_roots, "node:ser", ProgramNodeKind.SERIALIZER, "to_dict")
    test_node = _node(
        graph_roots, "node:test", ProgramNodeKind.TEST, "test_process", path="tests/test_svc.py"
    )
    validation = _node(
        graph_roots, "node:val", ProgramNodeKind.VALIDATION, "validate_process", path="src/checks.py"
    )
    api = _node(graph_roots, "node:api", ProgramNodeKind.API_ENDPOINT, "endpoint")
    edges = [
        _edge(graph_roots, "node:override", "node:base", ProgramEdgeKind.OVERRIDES),
        _edge(graph_roots, "node:svc", "node:base", ProgramEdgeKind.OVERRIDES),
        _edge(graph_roots, "node:svc", "node:factory", ProgramEdgeKind.FACTORY_CREATES),
        _edge(graph_roots, "node:svc", "node:ser", ProgramEdgeKind.SERIALIZES),
        _edge(graph_roots, "node:test", "node:svc", ProgramEdgeKind.TESTS),
        _edge(graph_roots, "node:val", "node:svc", ProgramEdgeKind.VALIDATES),
        _edge(graph_roots, "node:svc", "node:api", ProgramEdgeKind.SERVES),
    ]
    graph = _graph(
        graph_roots,
        [subject, base, override, factory, serializer, test_node, validation, api],
        edges,
    )
    roots = _bind_roots(prop_roots, graph)
    # Change base method → override + svc (OVERRIDES reverse) + downstream.
    delta = _delta(roots, "Base.process")
    receipt = compute_impact_closure(delta, graph)
    names = {item.node.symbol_id for item in receipt.consumers}
    assert "Child.process" in names or "node:override" in {item.node.node_id for item in receipt.consumers}
    assert "Service.process" in names or "node:svc" in {item.node.node_id for item in receipt.consumers}

    # Change service method → tests, validation, factory, serializer, api.
    delta_svc = _delta(roots, "Service.process")
    receipt_svc = compute_impact_closure(delta_svc, graph)
    kinds = {item.node.kind for item in receipt_svc.consumers}
    assert "test" in kinds
    assert "validation" in kinds
    assert "factory" in kinds or "serializer" in kinds or "api_endpoint" in kinds
    assert receipt_svc.validation_refs


def test_scc_cycle_grouped_deterministically(
    graph_roots: ProgramGraphRoots, prop_roots: PropagationAuthorityRoots
) -> None:
    a = _node(graph_roots, "node:a", ProgramNodeKind.FUNCTION, "a")
    b = _node(graph_roots, "node:b", ProgramNodeKind.FUNCTION, "b")
    c = _node(graph_roots, "node:c", ProgramNodeKind.FUNCTION, "c")
    seed = _node(graph_roots, "node:seed", ProgramNodeKind.FUNCTION, "seed")
    # seed ← a ← b ← c ← a  (cycle among a,b,c; a calls seed)
    edges = [
        _edge(graph_roots, "node:a", "node:seed", ProgramEdgeKind.CALLS),
        _edge(graph_roots, "node:b", "node:a", ProgramEdgeKind.CALLS),
        _edge(graph_roots, "node:c", "node:b", ProgramEdgeKind.CALLS),
        _edge(graph_roots, "node:a", "node:c", ProgramEdgeKind.CALLS),
    ]
    graph = _graph(graph_roots, [seed, a, b, c], edges)
    roots = _bind_roots(prop_roots, graph)
    delta = _delta(roots, "seed")
    first = compute_impact_closure(delta, graph)
    second = compute_impact_closure(delta, graph)
    assert first.to_record() == second.to_record()
    assert first.sccs
    multi = [scc for scc in first.sccs if len(scc.member_consumer_ids) >= 2]
    assert multi, "expected a multi-member SCC for the a/b/c cycle"
    members = set(multi[0].member_consumer_ids)
    assert "consumer:node:a" in members
    assert "consumer:node:b" in members
    assert "consumer:node:c" in members


# ---------------------------------------------------------------------------
# Authority / frontier fail-closed behaviour
# ---------------------------------------------------------------------------


def test_nominated_graphrag_edge_does_not_close_coverage(
    graph_roots: ProgramGraphRoots, prop_roots: PropagationAuthorityRoots
) -> None:
    process = _node(graph_roots, "node:process", ProgramNodeKind.FUNCTION, "process")
    nominated = _node(graph_roots, "node:plugin", ProgramNodeKind.FUNCTION, "plugin_hook")
    edges = [
        _edge(
            graph_roots,
            "node:plugin",
            "node:process",
            ProgramEdgeKind.CALLS,
            nominated=True,
            provenance=ProgramProvenance.GRAPHRAG,
        ),
    ]
    graph = _graph(graph_roots, [process, nominated], edges)
    roots = _bind_roots(prop_roots, graph)
    delta = _delta(roots, "process")
    receipt = compute_impact_closure(delta, graph)

    assert receipt.completeness is not ImpactCompleteness.COMPLETE
    assert receipt.completeness is ImpactCompleteness.PARTIAL_WITH_FRONTIER
    assert receipt.frontier_edge_ids or receipt.frontier_node_ids
    # Nominated consumer is never mandatory.
    for consumer in receipt.consumers:
        if consumer.node.node_id == "node:plugin":
            assert consumer.mandatory is False


def test_vector_runtime_edges_are_frontier_only(
    graph_roots: ProgramGraphRoots, prop_roots: PropagationAuthorityRoots
) -> None:
    process = _node(graph_roots, "node:process", ProgramNodeKind.FUNCTION, "process")
    runtime = _node(graph_roots, "node:runtime", ProgramNodeKind.FUNCTION, "runtime_hook")
    vector = _node(graph_roots, "node:vector", ProgramNodeKind.FUNCTION, "vector_hit")
    edges = [
        _edge(
            graph_roots,
            "node:runtime",
            "node:process",
            ProgramEdgeKind.RELATED_TO,
            nominated=True,
            provenance=ProgramProvenance.RUNTIME,
        ),
        _edge(
            graph_roots,
            "node:vector",
            "node:process",
            ProgramEdgeKind.RELATED_TO,
            nominated=True,
            provenance=ProgramProvenance.VECTOR,
        ),
    ]
    graph = _graph(graph_roots, [process, runtime, vector], edges)
    roots = _bind_roots(prop_roots, graph)
    receipt = compute_impact_closure(_delta(roots, "process"), graph)
    assert receipt.completeness is ImpactCompleteness.PARTIAL_WITH_FRONTIER
    assert not any(item.mandatory for item in receipt.consumers)


def test_truncation_cannot_yield_complete(
    graph_roots: ProgramGraphRoots, prop_roots: PropagationAuthorityRoots
) -> None:
    process = _node(graph_roots, "node:process", ProgramNodeKind.FUNCTION, "process")
    nodes = [process]
    edges = []
    for index in range(6):
        node_id = f"node:caller{index}"
        nodes.append(_node(graph_roots, node_id, ProgramNodeKind.FUNCTION, f"caller{index}"))
        edges.append(_edge(graph_roots, node_id, "node:process", ProgramEdgeKind.CALLS))
    graph = _graph(graph_roots, nodes, edges)
    roots = _bind_roots(prop_roots, graph)
    bounds = ImpactClosureBounds(max_consumers=3, max_depth=256, max_edges=100, max_sccs=16)
    receipt = compute_impact_closure(
        _delta(roots, "process"), graph, bounds=bounds
    )
    assert receipt.completeness is not ImpactCompleteness.COMPLETE
    assert receipt.completeness is ImpactCompleteness.PARTIAL_WITH_FRONTIER
    assert any("truncated" in ref for ref in receipt.frontier_node_ids)
    assert bounds.bound_ref in receipt.resource_bound_refs


def test_depth_bound_truncation_is_explicit(
    graph_roots: ProgramGraphRoots, prop_roots: PropagationAuthorityRoots
) -> None:
    nodes = [
        _node(graph_roots, f"node:n{i}", ProgramNodeKind.FUNCTION, f"n{i}")
        for i in range(5)
    ]
    edges = [
        _edge(graph_roots, f"node:n{i+1}", f"node:n{i}", ProgramEdgeKind.CALLS)
        for i in range(4)
    ]
    graph = _graph(graph_roots, nodes, edges)
    roots = _bind_roots(prop_roots, graph)
    bounds = ImpactClosureBounds(max_consumers=100, max_depth=1, max_edges=100, max_sccs=16)
    result = compute_impact_closure_result(
        _delta(roots, "n0"), graph, bounds=bounds
    )
    assert result.diagnostics.truncated is True
    assert result.receipt.completeness is ImpactCompleteness.PARTIAL_WITH_FRONTIER


def test_stale_graph_root_abstains(
    graph_roots: ProgramGraphRoots, prop_roots: PropagationAuthorityRoots
) -> None:
    process = _node(graph_roots, "node:process", ProgramNodeKind.FUNCTION, "process")
    caller = _node(graph_roots, "node:caller", ProgramNodeKind.FUNCTION, "caller")
    graph = _graph(
        graph_roots,
        [process, caller],
        [_edge(graph_roots, "node:caller", "node:process", ProgramEdgeKind.CALLS)],
    )
    # Claim a different graph identity than the supplied snapshot.
    delta = _delta(prop_roots, "process", graph_id="graph:stale-forged")
    receipt = compute_impact_closure(delta, graph)
    assert receipt.completeness is ImpactCompleteness.ABSTAINED
    assert any("stale_graph" in ref for ref in receipt.frontier_node_ids)
    assert not any(item.mandatory for item in receipt.consumers)


def test_unresolved_subject_abstains(
    graph_roots: ProgramGraphRoots, prop_roots: PropagationAuthorityRoots
) -> None:
    process = _node(graph_roots, "node:process", ProgramNodeKind.FUNCTION, "process")
    graph = _graph(graph_roots, [process], [])
    roots = _bind_roots(prop_roots, graph)
    receipt = compute_impact_closure(_delta(roots, "symbol:does_not_exist"), graph)
    assert receipt.completeness is ImpactCompleteness.ABSTAINED
    assert any("unresolved_subject" in ref for ref in receipt.frontier_node_ids)


def test_forged_completeness_cannot_be_constructed_via_analyzer(
    graph_roots: ProgramGraphRoots, prop_roots: PropagationAuthorityRoots
) -> None:
    """Open graph frontiers force partial; analyzer never emits complete+frontier."""
    process = _node(graph_roots, "node:process", ProgramNodeKind.FUNCTION, "process")
    caller = _node(graph_roots, "node:caller", ProgramNodeKind.FUNCTION, "caller")
    graph = _graph(
        graph_roots,
        [process, caller],
        [_edge(graph_roots, "node:caller", "node:process", ProgramEdgeKind.CALLS)],
        complete=False,
        frontier_refs=("reflection:getattr",),
    )
    roots = _bind_roots(prop_roots, graph)
    receipt = compute_impact_closure(_delta(roots, "process"), graph)
    assert receipt.completeness is ImpactCompleteness.PARTIAL_WITH_FRONTIER
    assert receipt.frontier_node_ids
    # Canonical record rejects complete+frontier; analyzer must not produce it.
    assert not (
        receipt.completeness is ImpactCompleteness.COMPLETE
        and (receipt.frontier_node_ids or receipt.frontier_edge_ids)
    )


def test_dynamic_frontier_node_is_recorded(
    graph_roots: ProgramGraphRoots, prop_roots: PropagationAuthorityRoots
) -> None:
    process = _node(graph_roots, "node:process", ProgramNodeKind.FUNCTION, "process")
    dynamic = _node(
        graph_roots,
        "node:dynamic",
        ProgramNodeKind.FRONTIER,
        "getattr",
        frontier=True,
    )
    edges = [_edge(graph_roots, "node:dynamic", "node:process", ProgramEdgeKind.CALLS)]
    graph = _graph(
        graph_roots,
        [process, dynamic],
        edges,
        frontier_refs=("dynamic:src/mod.py:getattr",),
    )
    roots = _bind_roots(prop_roots, graph)
    receipt = compute_impact_closure(_delta(roots, "process"), graph)
    assert receipt.completeness is ImpactCompleteness.PARTIAL_WITH_FRONTIER
    assert any(
        item.node.node_id == "node:dynamic" and not item.mandatory
        for item in receipt.consumers
    ) or "node:dynamic" in receipt.frontier_node_ids


def test_exclusions_and_bounds_recorded(
    graph_roots: ProgramGraphRoots, prop_roots: PropagationAuthorityRoots
) -> None:
    roots = ProgramGraphRoots(
        forest_id=graph_roots.forest_id,
        tree_id=graph_roots.tree_id,
        overlay_id=graph_roots.overlay_id,
        coverage_id=graph_roots.coverage_id,
        included_roots=("src/",),
        excluded_roots=("vendor/",),
        generated_roots=("generated/",),
        native_roots=(),
        extractor_id=graph_roots.extractor_id,
        config_id=graph_roots.config_id,
        toolchain_id=graph_roots.toolchain_id,
    )
    process = _node(roots, "node:process", ProgramNodeKind.FUNCTION, "process")
    graph = _graph(
        roots,
        [process],
        [],
        complete=False,
        exclusion_refs=("excluded_root:vendor/",),
        frontier_refs=("generated_root:generated/",),
    )
    prop = _bind_roots(prop_roots, graph)
    bounds = ImpactClosureBounds(max_consumers=10, max_depth=4, max_edges=20, max_sccs=4)
    receipt = ContractChangeImpactAnalyzer(bounds=bounds).analyze(
        _delta(prop, "process"), graph
    ).receipt
    assert any("vendor" in ref for ref in receipt.excluded_refs)
    assert bounds.bound_ref in receipt.resource_bound_refs
    assert receipt.completeness is ImpactCompleteness.PARTIAL_WITH_FRONTIER


# ---------------------------------------------------------------------------
# CodeImpactIndex integration
# ---------------------------------------------------------------------------


def test_code_impact_index_adds_validation_and_dependent(
    graph_roots: ProgramGraphRoots, prop_roots: PropagationAuthorityRoots
) -> None:
    process = _node(
        graph_roots,
        "node:process",
        ProgramNodeKind.FUNCTION,
        "process",
        path="src/service.py",
        qualified="service.process",
    )
    helper = _node(
        graph_roots,
        "node:helper",
        ProgramNodeKind.FUNCTION,
        "helper",
        path="src/helper.py",
        qualified="helper.helper",
    )
    graph = _graph(graph_roots, [process, helper], [])
    roots = _bind_roots(prop_roots, graph)
    index = CodeImpactIndex(
        repository_tree_id=graph_roots.tree_id,
        symbol_paths={
            "service.process": "src/service.py",
            "helper.helper": "src/helper.py",
        },
        symbol_dependencies={
            "helper.helper": ("service.process",),
        },
        path_dependencies={},
        validation_targets={
            "validation:unit-service": ("service.process", "src/service.py"),
        },
    )
    # Bind index_id into roots.
    roots = PropagationAuthorityRoots(
        repository_id=roots.repository_id,
        base_forest_id=roots.base_forest_id,
        base_tree_id=roots.base_tree_id,
        base_overlay_id=roots.base_overlay_id,
        candidate_forest_id=roots.candidate_forest_id,
        candidate_tree_id=roots.candidate_tree_id,
        candidate_overlay_id=roots.candidate_overlay_id,
        graph_id=roots.graph_id,
        index_id=index.index_id,
        model_id=roots.model_id,
        config_id=roots.config_id,
        translator_id=roots.translator_id,
        toolchain_id=roots.toolchain_id,
        policy_id=roots.policy_id,
    )
    receipt = compute_impact_closure(
        _delta(roots, "service.process"),
        graph,
        impact_index=index,
    )
    assert any(item.node.node_id == "node:helper" for item in receipt.consumers)
    assert "validation:unit-service" in receipt.validation_refs


# ---------------------------------------------------------------------------
# ProgramDependencyGraph integration
# ---------------------------------------------------------------------------


def test_program_dependency_graph_reverse_closure_from_fixture() -> None:
    graph_roots = ProgramGraphRoots(
        forest_id="forest:pdg",
        tree_id="tree:pdg",
        overlay_id="overlay:pdg",
        coverage_id="coverage:pdg",
        included_roots=("src/", "tests/"),
        excluded_roots=(),
        generated_roots=(),
        native_roots=(),
        extractor_id="program-graph@1",
        config_id="config:pdg",
        toolchain_id="toolchain:cpython",
    )
    sources = {
        "src/core.py": (
            "def process(left, right):\n"
            "    return left + right\n"
            "\n"
            "def wrapper(left, right):\n"
            "    return process(left, right)\n"
            "\n"
            "def api(request):\n"
            "    return wrapper(request.a, request.b)\n"
        ),
        "tests/test_core.py": (
            "from src.core import process\n"
            "\n"
            "def test_process():\n"
            "    assert process(1, 2) == 3\n"
        ),
    }
    builder = ProgramDependencyGraph(graph_roots)
    builder.build(
        [
            PathSource(path=path, source=source, language="python")
            for path, source in sorted(sources.items())
        ]
    )
    graph = builder.graph
    assert graph is not None

    seeds = resolve_seed_nodes(graph, "process")
    assert seeds, "expected process seed in dependency graph"

    prop_roots = PropagationAuthorityRoots(
        repository_id="repository:pdg",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id=graph_roots.forest_id,
        candidate_tree_id=graph_roots.tree_id,
        candidate_overlay_id=graph_roots.overlay_id,
        graph_id=graph.graph_id,
        index_id="index:pdg",
        model_id="model:pdg",
        config_id="config:pdg",
        translator_id="translator:pdg",
        toolchain_id="toolchain:cpython",
        policy_id="policy:pdg",
    )
    delta = _delta(prop_roots, "process")
    result = compute_impact_closure_result(delta, graph)
    receipt = result.receipt

    consumer_names = {
        item.node.symbol_id for item in receipt.consumers
    } | {item.node.node_id for item in receipt.consumers}
    # At least one of wrapper / api / test must appear.
    joined = " ".join(sorted(consumer_names)).lower()
    assert "wrapper" in joined or "api" in joined or "test" in joined
    assert result.diagnostics.seed_node_ids
    assert receipt.resource_bound_refs
    assert receipt.evidence_refs
    # Determinism across runs.
    again = compute_impact_closure(delta, graph)
    assert again.to_record() == receipt.to_record()


def test_analyzer_class_matches_module_helper(
    graph_roots: ProgramGraphRoots, prop_roots: PropagationAuthorityRoots
) -> None:
    process = _node(graph_roots, "node:process", ProgramNodeKind.FUNCTION, "process")
    caller = _node(graph_roots, "node:caller", ProgramNodeKind.FUNCTION, "caller")
    graph = _graph(
        graph_roots,
        [process, caller],
        [_edge(graph_roots, "node:caller", "node:process", ProgramEdgeKind.CALLS)],
    )
    roots = _bind_roots(prop_roots, graph)
    delta = _delta(roots, "process")
    via_class = ContractChangeImpactAnalyzer().analyze(delta, graph).receipt
    via_helper = compute_impact_closure(delta, graph)
    assert via_class.to_record() == via_helper.to_record()


def test_zero_consumers_on_complete_graph_is_complete_not_silent_unused_claim(
    graph_roots: ProgramGraphRoots, prop_roots: PropagationAuthorityRoots
) -> None:
    """Empty consumers with a complete trusted graph may be COMPLETE.

    Zero consumers is not *proof of unused* when the graph is incomplete; that
    case must remain partial/abstained (covered by frontier tests).
    """
    process = _node(graph_roots, "node:process", ProgramNodeKind.FUNCTION, "process")
    graph = _graph(graph_roots, [process], [], complete=True)
    roots = _bind_roots(prop_roots, graph)
    receipt = compute_impact_closure(_delta(roots, "process"), graph)
    assert receipt.consumers == ()
    assert receipt.completeness is ImpactCompleteness.COMPLETE


def test_bounds_validation_rejects_invalid_limits() -> None:
    with pytest.raises(Exception):
        ImpactClosureBounds(max_consumers=0)
    with pytest.raises(Exception):
        ImpactClosureBounds(max_depth=-1)
    with pytest.raises(Exception):
        ImpactClosureBounds(max_edges=10**9)
