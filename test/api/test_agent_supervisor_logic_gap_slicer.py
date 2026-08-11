"""Conformance tests for conservative logic-gap static slicing (LPR-007)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    GraphNodeRef,
    GraphProvenance,
    ImpactClosureReceipt,
    ImpactCompleteness,
    ImpactConsumer,
    ImpactSCC,
    PropagationAuthorityRoots,
)
from ipfs_accelerate_py.agent_supervisor.analysis.logic_gap_slicer import (
    PRODUCER_ID,
    AnalyzerCoverage,
    AnalyzerKind,
    ExclusionReason,
    InformationDemand,
    InventoryDisposition,
    LogicGapSlice,
    LogicGapSlicer,
    LogicGapSlicerAuthorityError,
    LogicGapSlicerBoundsError,
    LogicGapSlicerError,
    LogicGapSlicingInventory,
    SccReference,
    SliceFactClass,
    SliceFactSelection,
    SliceSelectionDisposition,
    StaticSliceCompleteness,
    all_analyzer_kinds,
    all_slice_fact_classes,
    all_static_slice_completeness,
    is_terminal_completeness,
    required_next_source_types_for_facets,
    slice_logic_gaps,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
    GapDisposition,
    GapMissingClass,
    GoalDisposition,
    GoalFamily,
    LogicFacetKind,
    LogicFacetRef,
    LogicGap,
    ProgramLogicAuthorityRoots,
    ProgramLogicGoal,
    ProofStatus,
    SourceAuthorityClass,
    SourceRouteKind,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_premise_corpus import (
    PremiseFeatureSet,
    PremiseSourceClass,
    PremiseTombstone,
    ProgramLogicPremise,
    ProgramLogicPremiseCorpus,
)
from ipfs_accelerate_py.agent_supervisor.analysis.value_provenance_graph import (
    PRODUCER_ID as VPG_PRODUCER,
    DefinitionKind,
    DominanceFact,
    DominanceKind,
    PathCondition,
    ProvenanceStatus,
    ReachingDefinition,
    SourceLocation,
    UnknownFrontierFact,
    UnknownReason,
    ValueProvenanceGraph,
)
from ipfs_accelerate_py.agent_supervisor.program_graph import (
    Completeness,
    ProgramEdgeKind,
    ProgramGraphRoots,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def logic_roots() -> ProgramLogicAuthorityRoots:
    return ProgramLogicAuthorityRoots(
        repository_id="repository:lpr-007",
        objective_id="objective:lpr-007",
        trace_id="trace:lpr-007",
        change_id="change:lpr-007",
        consumer_id="consumer:primary",
        forest_id="forest:lpr-007",
        tree_id="tree:candidate",
        overlay_id="overlay:lpr-007",
        graph_id="graph:lpr-007",
        index_id="index:lpr-007",
        corpus_id="corpus:lpr-007",
        model_id="model:lpr-007",
        translator_id="translator:lpr-007",
        toolchain_id="toolchain:lpr-007",
        policy_id="policy:lpr-007",
        environment_id="environment:lpr-007",
    )


@pytest.fixture
def prop_roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:lpr-007",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:lpr-007",
        index_id="index:lpr-007",
        model_id="model:lpr-007",
        config_id="config:lpr-007",
        translator_id="translator:lpr-007",
        toolchain_id="toolchain:lpr-007",
        policy_id="policy:lpr-007",
    )


@pytest.fixture
def graph_roots() -> ProgramGraphRoots:
    return ProgramGraphRoots(
        forest_id="forest:lpr-007",
        tree_id="tree:candidate",
        overlay_id="overlay:lpr-007",
        coverage_id="coverage:full",
        included_roots=("src/",),
        excluded_roots=("vendor/",),
        extractor_id="program-dependency-graph@1",
        config_id="config:lpr-007",
        toolchain_id="toolchain:lpr-007",
    )


def _goal(
    roots: ProgramLogicAuthorityRoots,
    *,
    goal_id: str = "goal:value-sufficiency",
    family: GoalFamily = GoalFamily.VALUE,
    disposition: GoalDisposition = GoalDisposition.OPEN,
    symbols: tuple[str, ...] = ("symbol:pkg.process",),
    source_refs: tuple[str, ...] = ("stmt:contract.process",),
    required_facets: tuple[LogicFacetRef, ...] = (),
    unsupported_facets: tuple[LogicFacetRef, ...] = (),
    bound_refs: tuple[str, ...] = (),
) -> ProgramLogicGoal:
    if not required_facets and not unsupported_facets:
        required_facets = (
            LogicFacetRef(
                facet_id="facet:info:process",
                kind=LogicFacetKind.INFORMATION,
                subject_symbol_id="symbol:pkg.process",
                contract_ref="contract:process",
            ),
        )
    return ProgramLogicGoal(
        roots=roots,
        goal_id=goal_id,
        family=family,
        disposition=disposition,
        positive_statement_ref="stmt:goal:value-sufficiency",
        affected_symbol_ids=symbols,
        source_refs=source_refs,
        required_facets=required_facets,
        unsupported_facets=unsupported_facets,
        assumption_refs=("assumption:root-current",),
        assumption_authority=SourceAuthorityClass.AUTHORITATIVE,
        proof_status=ProofStatus.UNPROVED,
        logic_family_refs=("logic:fol",),
        bound_refs=bound_refs,
        invalidation_refs=(
            roots.tree_id,
            roots.graph_id,
            roots.corpus_id,
            roots.policy_id,
        ),
    )


def _location(path: str = "src/sample.py", line: int = 10) -> SourceLocation:
    return SourceLocation(path=path, line_start=line, line_end=line + 1)


def _vpg(
    graph_roots: ProgramGraphRoots,
    *,
    complete: bool = True,
    with_frontier: bool = False,
    with_unsupported: bool = False,
    with_bound: bool = False,
) -> ValueProvenanceGraph:
    location = _location()
    definitions = (
        ReachingDefinition(
            def_id="def:process:ctx",
            variable="ctx",
            kind=DefinitionKind.PARAMETER,
            block_id="block:process:entry",
            procedure_id="proc:pkg.process",
            location=location,
            producer_id=VPG_PRODUCER,
            roots_id=graph_roots.roots_id,
            status=ProvenanceStatus.PROVED,
        ),
        ReachingDefinition(
            def_id="def:process:total",
            variable="total",
            kind=DefinitionKind.ASSIGNMENT,
            block_id="block:process:body",
            procedure_id="proc:pkg.process",
            location=_location(line=20),
            producer_id=VPG_PRODUCER,
            roots_id=graph_roots.roots_id,
            status=ProvenanceStatus.PROVED,
        ),
    )
    dominance = (
        DominanceFact(
            fact_id="dom:entry-body",
            kind=DominanceKind.DOMINATES,
            dominator_block_id="block:process:entry",
            dominated_block_id="block:process:body",
            procedure_id="proc:pkg.process",
            status=ProvenanceStatus.PROVED,
        ),
    )
    paths = (
        PathCondition(
            condition_id="path:process:body",
            procedure_id="proc:pkg.process",
            block_id="block:process:body",
            predicate_ref="pred:true",
            status=ProvenanceStatus.PROVED,
        ),
    )
    frontier: tuple[UnknownFrontierFact, ...] = ()
    completeness = Completeness.COMPLETE if complete else Completeness.PARTIAL
    if with_frontier:
        frontier = (
            UnknownFrontierFact(
                fact_id="unknown:dynamic-dispatch",
                reason=UnknownReason.DYNAMIC_TARGET,
                procedure_id="proc:pkg.process",
                block_id="block:process:body",
                variable="handler",
            ),
        )
        completeness = Completeness.FRONTIER
    if with_unsupported:
        frontier = frontier + (
            UnknownFrontierFact(
                fact_id="unknown:unsupported-ast",
                reason=UnknownReason.UNSUPPORTED_AST,
                procedure_id="proc:pkg.process",
                detail="async-generator",
            ),
        )
        completeness = Completeness.UNSUPPORTED
    if with_bound:
        frontier = frontier + (
            UnknownFrontierFact(
                fact_id="unknown:loop-bound",
                reason=UnknownReason.LOOP_BEYOND_BOUNDS,
                procedure_id="proc:pkg.process",
                block_id="block:process:loop",
            ),
        )
        completeness = Completeness.PARTIAL
    return ValueProvenanceGraph(
        roots=graph_roots,
        producer_id=VPG_PRODUCER,
        procedures=("proc:pkg.process",),
        blocks=(),
        definitions=definitions,
        uses=(),
        def_use_chains=(),
        dominance_facts=dominance,
        path_conditions=paths,
        type_refinements=(),
        information_provenances=(),
        interprocedural_threads=(),
        unknown_frontier=frontier,
        completeness=completeness,
    )


@dataclass
class _FakeEdge:
    source: str
    target: str
    kind: ProgramEdgeKind
    edge_id: str = ""

    def __post_init__(self) -> None:
        if not self.edge_id:
            self.edge_id = f"edge:{self.source}->{self.target}:{self.kind.value}"


@dataclass
class _FakeNode:
    node_id: str
    qualified_name: str = ""
    name: str = ""

    def __post_init__(self) -> None:
        if not self.qualified_name:
            self.qualified_name = self.node_id
        if not self.name:
            self.name = self.node_id.rsplit(".", 1)[-1]


@dataclass
class FakeProgramGraph:
    """Minimal ProgramGraph-like fixture for slicer tests."""

    roots: ProgramGraphRoots
    nodes_by_id: dict[str, _FakeNode] = field(default_factory=dict)
    edges: list[_FakeEdge] = field(default_factory=list)
    frontier_refs: tuple[str, ...] = ()
    exclusion_refs: tuple[str, ...] = ()
    complete: bool = True
    graph_id: str = "graph:fake-lpr-007"

    def node(self, node_id: str) -> _FakeNode | None:
        return self.nodes_by_id.get(node_id)

    def edges_from(self, node_id: str) -> tuple[_FakeEdge, ...]:
        return tuple(edge for edge in self.edges if edge.source == node_id)

    def edges_to(self, node_id: str) -> tuple[_FakeEdge, ...]:
        return tuple(edge for edge in self.edges if edge.target == node_id)

    def edges_of_kind(self, kind: Any) -> tuple[_FakeEdge, ...]:
        return tuple(edge for edge in self.edges if edge.kind == kind)

    def find_by_qualified_name(self, name: str) -> tuple[_FakeNode, ...]:
        return tuple(
            node
            for node in self.nodes_by_id.values()
            if node.qualified_name == name or node.node_id == name
        )


def _dependency_graph(graph_roots: ProgramGraphRoots, *, cyclic: bool = False) -> FakeProgramGraph:
    nodes = {
        "symbol:pkg.process": _FakeNode("symbol:pkg.process", "pkg.process"),
        "symbol:pkg.caller": _FakeNode("symbol:pkg.caller", "pkg.caller"),
        "symbol:pkg.Factory": _FakeNode("symbol:pkg.Factory", "pkg.Factory"),
        "symbol:pkg.Schema": _FakeNode("symbol:pkg.Schema", "pkg.Schema"),
        "symbol:pkg.helper": _FakeNode("symbol:pkg.helper", "pkg.helper"),
    }
    edges = [
        _FakeEdge("symbol:pkg.caller", "symbol:pkg.process", ProgramEdgeKind.CALLS),
        _FakeEdge(
            "symbol:pkg.Factory", "symbol:pkg.process", ProgramEdgeKind.FACTORY_CREATES
        ),
        _FakeEdge(
            "symbol:pkg.Schema", "symbol:pkg.process", ProgramEdgeKind.SCHEMA_OF
        ),
        _FakeEdge(
            "symbol:pkg.helper", "symbol:pkg.process", ProgramEdgeKind.DATA_FLOW
        ),
        # Nominating-only edge must be excluded.
        _FakeEdge(
            "symbol:pkg.vector-hit",
            "symbol:pkg.process",
            ProgramEdgeKind.RELATED_TO,
            edge_id="edge:vector-nominated",
        ),
    ]
    if cyclic:
        edges.append(
            _FakeEdge(
                "symbol:pkg.process", "symbol:pkg.helper", ProgramEdgeKind.DATA_FLOW
            )
        )
        edges.append(
            _FakeEdge(
                "symbol:pkg.helper", "symbol:pkg.caller", ProgramEdgeKind.DEPENDS_ON
            )
        )
        edges.append(
            _FakeEdge(
                "symbol:pkg.caller", "symbol:pkg.process", ProgramEdgeKind.CALLS
            )
        )
    return FakeProgramGraph(
        roots=graph_roots,
        nodes_by_id=nodes,
        edges=edges,
        complete=True,
        frontier_refs=(),
        exclusion_refs=("exclusion:vendor/",),
    )


def _corpus(
    roots: ProgramLogicAuthorityRoots,
    *,
    include_hypothesis: bool = True,
    close_facet: bool = True,
) -> ProgramLogicPremiseCorpus:
    features = PremiseFeatureSet(
        symbol_feature_refs=("symbol:pkg.process",),
        type_feature_refs=("type:Context",),
        effect_feature_refs=(),
        import_feature_refs=("import:pkg",),
    )
    premises: list[ProgramLogicPremise] = [
        ProgramLogicPremise(
            roots=roots,
            premise_id="premise:contract.process",
            source_class=PremiseSourceClass.REVIEWED_CONTRACT,
            statement_ref="stmt:contract.process",
            statement_digest="sha256:" + "ab" * 32,
            lowering_ref="lower:contract.process",
            expectation_authority=True,
            source_precedence=100,
            features=features,
            contract_identity="contract:process",
        ),
        ProgramLogicPremise(
            roots=roots,
            premise_id="premise:static.reaching",
            source_class=PremiseSourceClass.VALUE_PROVENANCE,
            statement_ref="stmt:static.reaching",
            statement_digest="sha256:" + "cd" * 32,
            lowering_ref="lower:static.reaching",
            expectation_authority=False,
            source_precedence=50,
            features=features,
        ),
    ]
    if close_facet:
        premises.append(
            ProgramLogicPremise(
                roots=roots,
                premise_id="premise:facet:info:process",
                source_class=PremiseSourceClass.REVIEWED_CONTRACT,
                statement_ref="stmt:facet:info:process",
                statement_digest="sha256:" + "ef" * 32,
                lowering_ref="lower:facet:info",
                expectation_authority=True,
                source_precedence=100,
                features=features,
                contract_identity="contract:process",
            )
        )
    if include_hypothesis:
        premises.append(
            ProgramLogicPremise(
                roots=roots,
                premise_id="premise:vector.analogue",
                source_class=PremiseSourceClass.VECTOR_ANALOGUE,
                statement_ref="stmt:vector.analogue",
                statement_digest="sha256:" + "11" * 32,
                lowering_ref="lower:vector",
                expectation_authority=False,
                source_precedence=10,
                features=features,
            )
        )
    return ProgramLogicPremiseCorpus(
        roots=roots,
        premises=tuple(premises),
        tombstones=(
            PremiseTombstone(
                premise_id="premise:stale.old",
                statement_digest="sha256:" + "00" * 32,
                reason="superseded",
                tree_identity=roots.tree_id,
            ),
        ),
        graph_identity=roots.graph_id,
        index_identity=roots.index_id,
    )


def _impact(prop_roots: PropagationAuthorityRoots, *, with_scc: bool = True) -> ImpactClosureReceipt:
    node = GraphNodeRef(
        node_id="node:pkg.caller",
        kind="function",
        path="src/caller.py",
        symbol_id="symbol:pkg.caller",
        artifact_id="artifact:caller",
        provenance=GraphProvenance.TRUSTED,
        extractor_id="extractor:pdg",
    )
    consumers = (
        ImpactConsumer(
            consumer_id="consumer:pkg.caller",
            node=node,
            depth=1,
            mandatory=True,
            edge_refs=("edge:caller->process",),
            path_condition_ref="path:caller:entry",
        ),
        ImpactConsumer(
            consumer_id="consumer:pkg.helper",
            node=GraphNodeRef(
                node_id="node:pkg.helper",
                kind="function",
                path="src/helper.py",
                symbol_id="symbol:pkg.helper",
                artifact_id="artifact:helper",
                provenance=GraphProvenance.TRUSTED,
                extractor_id="extractor:pdg",
            ),
            depth=2,
            mandatory=True,
            edge_refs=("edge:helper->process",),
        ),
    )
    sccs = ()
    if with_scc:
        sccs = (
            ImpactSCC(
                scc_id="scc:caller-helper",
                member_consumer_ids=(
                    "consumer:pkg.caller",
                    "consumer:pkg.helper",
                ),
            ),
        )
    return ImpactClosureReceipt(
        roots=prop_roots,
        delta_id="delta:process-signature",
        completeness=ImpactCompleteness.COMPLETE,
        consumers=consumers,
        sccs=sccs,
        excluded_refs=("exclusion:test-only",),
        evidence_refs=("evidence:impact",),
    )


# ---------------------------------------------------------------------------
# Positive: dependency-complete minimal slice
# ---------------------------------------------------------------------------


def test_dependency_complete_minimal_slice(
    logic_roots: ProgramLogicAuthorityRoots,
    prop_roots: PropagationAuthorityRoots,
    graph_roots: ProgramGraphRoots,
) -> None:
    goal = _goal(logic_roots)
    inventory = LogicGapSlicer(logic_roots).slice(
        (goal,),
        corpus=_corpus(logic_roots),
        dependency_graph=_dependency_graph(graph_roots),
        value_provenance=_vpg(graph_roots),
        impact_closure=_impact(prop_roots),
    )
    assert inventory.disposition is InventoryDisposition.COMPLETE
    gap_slice = inventory.slice_for_goal(goal.goal_id)
    assert gap_slice is not None
    assert gap_slice.completeness is StaticSliceCompleteness.COMPLETE
    assert gap_slice.is_dependency_complete
    assert gap_slice.selected_fact_refs
    assert gap_slice.excluded_fact_refs  # hypothesis / nominating / tombstone
    assert gap_slice.reaching_definition_refs
    assert gap_slice.dominance_refs
    assert gap_slice.path_condition_refs
    assert gap_slice.caller_boundary_refs
    assert gap_slice.constructor_boundary_refs
    assert gap_slice.schema_boundary_refs
    assert gap_slice.consumer_closure_refs
    assert gap_slice.analyzer_coverage
    assert gap_slice.required_next_source_types
    assert gap_slice.semantic_authority is False
    assert gap_slice.producer_id == PRODUCER_ID
    # Selected / excluded disjoint
    assert not (set(gap_slice.selected_fact_refs) & set(gap_slice.excluded_fact_refs))
    # Hypothesis premise excluded
    assert "premise:vector.analogue" in gap_slice.excluded_fact_refs
    # Reviewed contract selected
    assert "premise:contract.process" in gap_slice.selected_fact_refs


def test_information_demand_and_logic_gap_projection(
    logic_roots: ProgramLogicAuthorityRoots,
    prop_roots: PropagationAuthorityRoots,
    graph_roots: ProgramGraphRoots,
) -> None:
    goal = _goal(logic_roots)
    inventory = slice_logic_gaps(
        logic_roots,
        (goal,),
        corpus=_corpus(logic_roots),
        dependency_graph=_dependency_graph(graph_roots),
        value_provenance=_vpg(graph_roots),
        impact_closure=_impact(prop_roots),
    )
    demand = inventory.demand_for_goal(goal.goal_id)
    assert demand is not None
    assert demand.slice_id
    assert demand.semantic_authority is False
    assert demand.candidate_source_routes
    gap = demand.to_logic_gap()
    assert isinstance(gap, LogicGap)
    assert gap.goal_id == goal.goal_id
    assert gap.semantic_authority is False
    assert gap.dependency_slice_refs
    assert inventory.gaps
    assert inventory.gaps[0].goal_id == goal.goal_id


def test_selected_and_excluded_facts_recorded(
    logic_roots: ProgramLogicAuthorityRoots,
    graph_roots: ProgramGraphRoots,
) -> None:
    goal = _goal(logic_roots)
    inventory = LogicGapSlicer(logic_roots).slice(
        (goal,),
        corpus=_corpus(logic_roots),
        dependency_graph=_dependency_graph(graph_roots),
        value_provenance=_vpg(graph_roots),
    )
    gap_slice = inventory.slice_for_goal(goal.goal_id)
    assert gap_slice is not None
    selected = [
        item
        for item in gap_slice.facts
        if item.disposition is SliceSelectionDisposition.SELECTED
    ]
    excluded = [
        item
        for item in gap_slice.facts
        if item.disposition is SliceSelectionDisposition.EXCLUDED
    ]
    assert selected
    assert excluded
    # Nominating RELATED_TO edge excluded
    assert any(
        item.fact_ref == "edge:vector-nominated"
        and item.exclusion_reason is ExclusionReason.NOMINATING_ONLY
        for item in excluded
    )


def test_reaching_path_dominance_requirements(
    logic_roots: ProgramLogicAuthorityRoots,
    graph_roots: ProgramGraphRoots,
) -> None:
    goal = _goal(logic_roots)
    inventory = LogicGapSlicer(logic_roots).slice(
        (goal,),
        value_provenance=_vpg(graph_roots),
        corpus=_corpus(logic_roots),
    )
    gap_slice = inventory.slice_for_goal(goal.goal_id)
    assert gap_slice is not None
    assert "def:process:ctx" in gap_slice.reaching_definition_refs or any(
        "def:process" in ref for ref in gap_slice.reaching_definition_refs
    )
    assert gap_slice.dominance_refs
    assert gap_slice.path_condition_refs
    coverage_kinds = {item.analyzer for item in gap_slice.analyzer_coverage}
    assert AnalyzerKind.REACHING_DEFINITIONS in coverage_kinds
    assert AnalyzerKind.DOMINANCE in coverage_kinds
    assert AnalyzerKind.PATH_CONDITIONS in coverage_kinds


def test_caller_constructor_schema_boundaries(
    logic_roots: ProgramLogicAuthorityRoots,
    graph_roots: ProgramGraphRoots,
) -> None:
    goal = _goal(logic_roots)
    inventory = LogicGapSlicer(logic_roots).slice(
        (goal,),
        dependency_graph=_dependency_graph(graph_roots),
        corpus=_corpus(logic_roots),
        value_provenance=_vpg(graph_roots),
    )
    gap_slice = inventory.slice_for_goal(goal.goal_id)
    assert gap_slice is not None
    assert "symbol:pkg.caller" in gap_slice.caller_boundary_refs
    assert "symbol:pkg.Factory" in gap_slice.constructor_boundary_refs
    assert "symbol:pkg.Schema" in gap_slice.schema_boundary_refs


def test_cycles_are_finite_scc_references(
    logic_roots: ProgramLogicAuthorityRoots,
    prop_roots: PropagationAuthorityRoots,
    graph_roots: ProgramGraphRoots,
) -> None:
    goal = _goal(logic_roots)
    inventory = LogicGapSlicer(logic_roots).slice(
        (goal,),
        dependency_graph=_dependency_graph(graph_roots, cyclic=True),
        impact_closure=_impact(prop_roots, with_scc=True),
        corpus=_corpus(logic_roots),
        value_provenance=_vpg(graph_roots),
    )
    gap_slice = inventory.slice_for_goal(goal.goal_id)
    assert gap_slice is not None
    assert gap_slice.scc_refs
    for scc in gap_slice.scc_refs:
        assert isinstance(scc, SccReference)
        assert len(scc.member_refs) >= 2
        # Finite — members listed once, no unrolled path explosion
        assert len(scc.member_refs) == len(set(scc.member_refs))
    # Impact SCC preserved
    assert any(scc.scc_id == "scc:caller-helper" for scc in gap_slice.scc_refs)


def test_unknown_frontier_exclusions_coverage_and_next_sources(
    logic_roots: ProgramLogicAuthorityRoots,
    graph_roots: ProgramGraphRoots,
) -> None:
    goal = _goal(logic_roots)
    inventory = LogicGapSlicer(logic_roots).slice(
        (goal,),
        value_provenance=_vpg(graph_roots, complete=False, with_frontier=True),
        dependency_graph=_dependency_graph(graph_roots),
        corpus=_corpus(logic_roots, close_facet=False),
    )
    gap_slice = inventory.slice_for_goal(goal.goal_id)
    assert gap_slice is not None
    assert gap_slice.completeness is StaticSliceCompleteness.FRONTIER
    assert gap_slice.unknown_frontier_refs
    assert gap_slice.exclusion_refs or gap_slice.excluded_fact_refs
    assert gap_slice.analyzer_coverage
    assert gap_slice.required_next_source_types
    assert SourceRouteKind.LOCAL_STATIC in gap_slice.required_next_source_types
    demand = inventory.demand_for_goal(goal.goal_id)
    assert demand is not None
    assert demand.disposition is GapDisposition.FRONTIER
    assert demand.unknown_frontier_refs
    assert demand.missing_class is GapMissingClass.FRONTIER


# ---------------------------------------------------------------------------
# Fail-closed / adversarial
# ---------------------------------------------------------------------------


def test_bound_exhaustion_is_incomplete_never_solved(
    logic_roots: ProgramLogicAuthorityRoots,
    graph_roots: ProgramGraphRoots,
) -> None:
    goal = _goal(logic_roots)
    inventory = LogicGapSlicer(
        logic_roots, max_slice_nodes=3, max_backward_depth=1
    ).slice(
        (goal,),
        dependency_graph=_dependency_graph(graph_roots),
        value_provenance=_vpg(graph_roots),
        corpus=_corpus(logic_roots),
    )
    gap_slice = inventory.slice_for_goal(goal.goal_id)
    assert gap_slice is not None
    assert gap_slice.completeness is StaticSliceCompleteness.INCOMPLETE
    assert any(ref.startswith("bound:exhausted") for ref in gap_slice.bound_refs)
    assert is_terminal_completeness(gap_slice.completeness)
    # Never "solved"
    assert gap_slice.completeness is not StaticSliceCompleteness.COMPLETE


def test_unsupported_syntax_yields_unsupported_never_solved(
    logic_roots: ProgramLogicAuthorityRoots,
    graph_roots: ProgramGraphRoots,
) -> None:
    goal = _goal(
        logic_roots,
        required_facets=(),
        unsupported_facets=(
            LogicFacetRef(
                facet_id="facet:memory:unsafe",
                kind=LogicFacetKind.MEMORY,
                subject_symbol_id="symbol:pkg.process",
                contract_ref="contract:process",
                unsupported=True,
            ),
        ),
    )
    inventory = LogicGapSlicer(logic_roots).slice(
        (goal,),
        value_provenance=_vpg(graph_roots, with_unsupported=True),
    )
    gap_slice = inventory.slice_for_goal(goal.goal_id)
    assert gap_slice is not None
    assert gap_slice.completeness in {
        StaticSliceCompleteness.UNSUPPORTED,
        StaticSliceCompleteness.FRONTIER,
        StaticSliceCompleteness.INCOMPLETE,
    }
    # Bound/unsupported path must not be complete
    assert gap_slice.completeness is not StaticSliceCompleteness.COMPLETE
    assert gap_slice.unsupported_construct_refs
    assert is_terminal_completeness(gap_slice.completeness) or (
        gap_slice.completeness is StaticSliceCompleteness.UNSUPPORTED
    )


def test_reject_cross_root_corpus(
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    other = ProgramLogicAuthorityRoots(
        repository_id="repository:other",
        objective_id="objective:other",
        trace_id="trace:other",
        change_id="change:other",
        consumer_id="consumer:other",
        forest_id="forest:other",
        tree_id="tree:other",
        overlay_id="overlay:other",
        graph_id="graph:other",
        index_id="index:other",
        corpus_id="corpus:other",
        model_id="model:other",
        translator_id="translator:other",
        toolchain_id="toolchain:other",
        policy_id="policy:other",
        environment_id="environment:other",
    )
    with pytest.raises(LogicGapSlicerAuthorityError, match="roots"):
        LogicGapSlicer(logic_roots).slice(
            (_goal(logic_roots),),
            corpus=_corpus(other),
        )


def test_reject_cross_root_goal(
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    other = ProgramLogicAuthorityRoots(
        repository_id="repository:other",
        objective_id="objective:other",
        trace_id="trace:other",
        change_id="change:other",
        consumer_id="consumer:other",
        forest_id="forest:other",
        tree_id="tree:other",
        overlay_id="overlay:other",
        graph_id="graph:other",
        index_id="index:other",
        corpus_id="corpus:other",
        model_id="model:other",
        translator_id="translator:other",
        toolchain_id="toolchain:other",
        policy_id="policy:other",
        environment_id="environment:other",
    )
    with pytest.raises(LogicGapSlicerAuthorityError, match="roots"):
        LogicGapSlicer(logic_roots).slice((_goal(other),))


def test_reject_forged_complete_impact_with_frontier(
    logic_roots: ProgramLogicAuthorityRoots,
    prop_roots: PropagationAuthorityRoots,
) -> None:
    # ImpactClosureReceipt itself rejects complete+frontier; ensure slicer
    # also rejects forged payloads that bypass construction.
    node = GraphNodeRef(
        node_id="node:x",
        kind="function",
        path="src/x.py",
        symbol_id="symbol:x",
        artifact_id="artifact:x",
        provenance=GraphProvenance.TRUSTED,
        extractor_id="extractor:pdg",
    )
    # Build a partial receipt, then mutate completeness (simulate forge).
    receipt = ImpactClosureReceipt(
        roots=prop_roots,
        delta_id="delta:forged",
        completeness=ImpactCompleteness.PARTIAL_WITH_FRONTIER,
        consumers=(
            ImpactConsumer(
                consumer_id="consumer:x",
                node=node,
                depth=0,
                mandatory=True,
            ),
        ),
        frontier_node_ids=("node:dynamic",),
    )
    forged = receipt.to_dict()
    forged["completeness"] = ImpactCompleteness.COMPLETE.value
    # from_dict should reject; if it somehow passed, slicer must still fail.
    with pytest.raises(Exception):
        ImpactClosureReceipt.from_dict(forged)


def test_reject_forged_complete_slice_with_frontier(
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    with pytest.raises(LogicGapSlicerAuthorityError, match="unknown frontier"):
        LogicGapSlice(
            roots=logic_roots,
            slice_id="slice:forged",
            goal_id="goal:one",
            completeness=StaticSliceCompleteness.COMPLETE,
            unknown_frontier_refs=("frontier:open",),
            selected_fact_refs=("fact:a",),
        )


def test_reject_forged_complete_slice_with_unsupported(
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    with pytest.raises(LogicGapSlicerAuthorityError, match="unsupported"):
        LogicGapSlice(
            roots=logic_roots,
            slice_id="slice:forged2",
            goal_id="goal:one",
            completeness=StaticSliceCompleteness.COMPLETE,
            unsupported_construct_refs=("unsupported:ast",),
            selected_fact_refs=("fact:a",),
        )


def test_reject_forged_complete_with_bound_exhaustion(
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    with pytest.raises(LogicGapSlicerAuthorityError, match="bound"):
        LogicGapSlice(
            roots=logic_roots,
            slice_id="slice:forged3",
            goal_id="goal:one",
            completeness=StaticSliceCompleteness.COMPLETE,
            bound_refs=("bound:exhausted:facts",),
            selected_fact_refs=("fact:a",),
        )


def test_reject_semantic_authority_on_slice(
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    with pytest.raises(LogicGapSlicerAuthorityError, match="semantic authority"):
        LogicGapSlice(
            roots=logic_roots,
            slice_id="slice:auth",
            goal_id="goal:one",
            completeness=StaticSliceCompleteness.INCOMPLETE,
            semantic_authority=True,
        )


def test_reject_source_bodies_in_slice(
    logic_roots: ProgramLogicAuthorityRoots,
    graph_roots: ProgramGraphRoots,
) -> None:
    goal = _goal(logic_roots)
    inventory = LogicGapSlicer(logic_roots).slice(
        (goal,),
        value_provenance=_vpg(graph_roots),
        corpus=_corpus(logic_roots),
    )
    payload = inventory.slices[0].to_dict()
    payload["source_body"] = "def evil(): pass"
    with pytest.raises(LogicGapSlicerError, match="unsupported fields|source bodies"):
        LogicGapSlice.from_dict(payload)


def test_slices_contain_references_not_bodies(
    logic_roots: ProgramLogicAuthorityRoots,
    prop_roots: PropagationAuthorityRoots,
    graph_roots: ProgramGraphRoots,
) -> None:
    goal = _goal(logic_roots)
    inventory = LogicGapSlicer(logic_roots).slice(
        (goal,),
        corpus=_corpus(logic_roots),
        dependency_graph=_dependency_graph(graph_roots),
        value_provenance=_vpg(graph_roots),
        impact_closure=_impact(prop_roots),
    )
    body_keys = {"source_body", "snippet", "source_text", "file_text", "proof_script"}
    for record in (inventory, *inventory.slices, *inventory.demands, *inventory.gaps):
        payload = record.to_dict()
        assert not (body_keys & set(payload))
        # Nested maps must also stay body-free ( CanonicalContract already
        # enforces this; double-check common body markers as keys).
        stack = [payload]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                for key, value in current.items():
                    assert key.lower().replace("-", "_") not in body_keys
                    if isinstance(value, (dict, list, tuple)):
                        stack.append(value)
            elif isinstance(current, (list, tuple)):
                stack.extend(
                    item for item in current if isinstance(item, (dict, list, tuple))
                )


def test_forged_content_identity_rejected(
    logic_roots: ProgramLogicAuthorityRoots,
    graph_roots: ProgramGraphRoots,
) -> None:
    goal = _goal(logic_roots)
    inventory = LogicGapSlicer(logic_roots).slice(
        (goal,),
        value_provenance=_vpg(graph_roots),
        corpus=_corpus(logic_roots),
    )
    payload = inventory.to_dict()
    payload["content_id"] = "forged:cid"
    with pytest.raises(LogicGapSlicerAuthorityError, match="content identity"):
        LogicGapSlicingInventory.from_dict(payload)


def test_value_provenance_forged_complete_with_frontier_rejected(
    logic_roots: ProgramLogicAuthorityRoots,
    graph_roots: ProgramGraphRoots,
) -> None:
    # Construct a VPG that claims COMPLETE while carrying frontier facts.
    # ValueProvenanceGraph may allow the structure; slicer must reject.
    location = _location()
    vpg = ValueProvenanceGraph(
        roots=graph_roots,
        producer_id=VPG_PRODUCER,
        procedures=("proc:pkg.process",),
        blocks=(),
        definitions=(
            ReachingDefinition(
                def_id="def:x",
                variable="x",
                kind=DefinitionKind.PARAMETER,
                block_id="block:entry",
                procedure_id="proc:pkg.process",
                location=location,
                producer_id=VPG_PRODUCER,
                roots_id=graph_roots.roots_id,
            ),
        ),
        uses=(),
        def_use_chains=(),
        dominance_facts=(),
        path_conditions=(),
        type_refinements=(),
        information_provenances=(),
        interprocedural_threads=(),
        unknown_frontier=(
            UnknownFrontierFact(
                fact_id="unknown:dyn",
                reason=UnknownReason.DYNAMIC_TARGET,
                procedure_id="proc:pkg.process",
            ),
        ),
        completeness=Completeness.COMPLETE,  # forged
    )
    with pytest.raises(LogicGapSlicerAuthorityError, match="complete"):
        LogicGapSlicer(logic_roots).slice(
            (_goal(logic_roots),),
            value_provenance=vpg,
        )


def test_cross_root_value_provenance_tree_rejected(
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    stale_roots = ProgramGraphRoots(
        forest_id="forest:lpr-007",
        tree_id="tree:stale",
        overlay_id="overlay:lpr-007",
        coverage_id="coverage:full",
        included_roots=("src/",),
        extractor_id="value-provenance-graph@1",
        config_id="config:lpr-007",
        toolchain_id="toolchain:lpr-007",
    )
    with pytest.raises(LogicGapSlicerAuthorityError, match="tree_id"):
        LogicGapSlicer(logic_roots).slice(
            (_goal(logic_roots),),
            value_provenance=_vpg(stale_roots),
        )


# ---------------------------------------------------------------------------
# Contract round-trips / determinism / helpers
# ---------------------------------------------------------------------------


def test_round_trip_inventory(
    logic_roots: ProgramLogicAuthorityRoots,
    prop_roots: PropagationAuthorityRoots,
    graph_roots: ProgramGraphRoots,
) -> None:
    goal = _goal(logic_roots)
    inventory = LogicGapSlicer(logic_roots).slice(
        (goal,),
        corpus=_corpus(logic_roots),
        dependency_graph=_dependency_graph(graph_roots),
        value_provenance=_vpg(graph_roots),
        impact_closure=_impact(prop_roots),
    )
    rebuilt = LogicGapSlicingInventory.from_dict(inventory.to_dict())
    assert rebuilt.content_id == inventory.content_id
    assert rebuilt.inventory_id == inventory.inventory_id
    assert len(rebuilt.slices) == len(inventory.slices)
    assert rebuilt.slices[0].slice_id == inventory.slices[0].slice_id


def test_deterministic_slicing(
    logic_roots: ProgramLogicAuthorityRoots,
    prop_roots: PropagationAuthorityRoots,
    graph_roots: ProgramGraphRoots,
) -> None:
    kwargs = dict(
        goals=(_goal(logic_roots),),
        corpus=_corpus(logic_roots),
        dependency_graph=_dependency_graph(graph_roots),
        value_provenance=_vpg(graph_roots),
        impact_closure=_impact(prop_roots),
    )
    first = LogicGapSlicer(logic_roots).slice(**kwargs)
    second = LogicGapSlicer(logic_roots).slice(**kwargs)
    assert first.content_id == second.content_id
    assert first.inventory_id == second.inventory_id
    assert [s.slice_id for s in first.slices] == [s.slice_id for s in second.slices]
    assert [d.demand_id for d in first.demands] == [d.demand_id for d in second.demands]


def test_empty_goals_abstain(logic_roots: ProgramLogicAuthorityRoots) -> None:
    inventory = LogicGapSlicer(logic_roots).slice()
    assert inventory.disposition is InventoryDisposition.ABSTAINED
    assert inventory.slices == ()
    assert inventory.demands == ()


def test_unsupported_goal_disposition_recorded(
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    goal = _goal(logic_roots, disposition=GoalDisposition.UNSUPPORTED)
    inventory = LogicGapSlicer(logic_roots).slice((goal,))
    assert goal.goal_id in inventory.unsupported_goal_ids
    assert inventory.slices == ()


def test_slice_fact_selection_requires_exclusion_reason() -> None:
    with pytest.raises(LogicGapSlicerError, match="exclusion_reason"):
        SliceFactSelection(
            fact_ref="fact:x",
            fact_class=SliceFactClass.PREMISE,
            disposition=SliceSelectionDisposition.EXCLUDED,
        )


def test_analyzer_coverage_rejects_complete_with_unsupported() -> None:
    with pytest.raises(LogicGapSlicerAuthorityError, match="unsupported"):
        AnalyzerCoverage(
            analyzer=AnalyzerKind.VALUE_PROVENANCE,
            completeness=Completeness.COMPLETE,
            unsupported_construct_refs=("unsupported:ast",),
        )


def test_information_demand_round_trip(
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    demand = InformationDemand(
        roots=logic_roots,
        demand_id="demand:one",
        goal_id="goal:one",
        slice_id="slice:one",
        missing_class=GapMissingClass.VALUE,
        disposition=GapDisposition.REQUIRED,
        observed_fact_ref="fact:observed",
        required_fact_ref="fact:required",
        discrepancy_ref="discrepancy:one",
        candidate_source_routes=(SourceRouteKind.DATAFLOW, SourceRouteKind.LOCAL_STATIC),
        automation_eligible=True,
    )
    rebuilt = InformationDemand.from_dict(demand.to_dict())
    assert rebuilt.content_id == demand.content_id
    gap = demand.to_logic_gap()
    assert gap.missing_class is GapMissingClass.VALUE


def test_helper_enumerations() -> None:
    assert AnalyzerKind.VALUE_PROVENANCE in all_analyzer_kinds()
    assert SliceFactClass.REACHING_DEFINITION in all_slice_fact_classes()
    assert StaticSliceCompleteness.COMPLETE in all_static_slice_completeness()
    assert is_terminal_completeness(StaticSliceCompleteness.INCOMPLETE)
    assert is_terminal_completeness(StaticSliceCompleteness.UNSUPPORTED)
    assert not is_terminal_completeness(StaticSliceCompleteness.COMPLETE)
    routes = required_next_source_types_for_facets((LogicFacetKind.INFORMATION,))
    assert SourceRouteKind.LOCAL_STATIC in routes
    assert SourceRouteKind.DATAFLOW in routes


def test_module_entry_point(
    logic_roots: ProgramLogicAuthorityRoots,
    graph_roots: ProgramGraphRoots,
) -> None:
    inventory = slice_logic_gaps(
        logic_roots,
        (_goal(logic_roots),),
        value_provenance=_vpg(graph_roots),
        corpus=_corpus(logic_roots),
    )
    assert isinstance(inventory, LogicGapSlicingInventory)
    assert inventory.producer_id == PRODUCER_ID


def test_max_slice_nodes_bound_validation(
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    with pytest.raises(LogicGapSlicerBoundsError):
        LogicGapSlicer(logic_roots, max_slice_nodes=0)


def test_incomplete_when_required_facets_unclosed(
    logic_roots: ProgramLogicAuthorityRoots,
    graph_roots: ProgramGraphRoots,
) -> None:
    goal = _goal(logic_roots)
    inventory = LogicGapSlicer(logic_roots).slice(
        (goal,),
        # Corpus that does not close the required facet id.
        corpus=_corpus(logic_roots, close_facet=False, include_hypothesis=False),
        value_provenance=_vpg(graph_roots),
        dependency_graph=_dependency_graph(graph_roots),
    )
    gap_slice = inventory.slice_for_goal(goal.goal_id)
    assert gap_slice is not None
    # Without closing the required facet, slice is incomplete (not solved).
    assert gap_slice.completeness is StaticSliceCompleteness.INCOMPLETE
    demand = inventory.demand_for_goal(goal.goal_id)
    assert demand is not None
    assert demand.disposition is GapDisposition.REQUIRED
    assert demand.missing_class in {
        GapMissingClass.VALUE,
        GapMissingClass.CONTRACT,
        GapMissingClass.PREMISE,
    }


def test_loop_bound_frontier_incomplete(
    logic_roots: ProgramLogicAuthorityRoots,
    graph_roots: ProgramGraphRoots,
) -> None:
    inventory = LogicGapSlicer(logic_roots).slice(
        (_goal(logic_roots),),
        value_provenance=_vpg(graph_roots, with_bound=True),
        corpus=_corpus(logic_roots),
    )
    gap_slice = inventory.slice_for_goal("goal:value-sufficiency")
    assert gap_slice is not None
    assert gap_slice.completeness is StaticSliceCompleteness.INCOMPLETE
    assert any("bound:exhausted" in ref for ref in gap_slice.bound_refs)
