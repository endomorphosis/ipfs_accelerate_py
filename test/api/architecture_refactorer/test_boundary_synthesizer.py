"""Hermetic PCAR-011 interface-boundary synthesis tests."""

from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.architecture_ir import (
    ArchitectureEdge,
    ArchitectureIR,
    ArchitectureNode,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.boundary_synthesizer import (
    BOUNDARY_PROPOSAL_EVIDENCE,
    BOUNDARY_PROPOSAL_SCHEMA,
    BOUNDARY_PROPOSAL_VERSION,
    BOUNDARY_SYNTHESIS_EVIDENCE,
    BOUNDARY_SYNTHESIS_SCHEMA,
    CANDIDATE_TIER_ONLY,
    CLOSED_BOUNDARY_CONCERNS,
    CLOSED_COST_DIMENSIONS,
    CLOSED_HARD_CONSTRAINTS,
    CLOSED_PROPOSAL_DISPOSITIONS,
    CLOSED_PROPOSAL_TIERS,
    COMPLETION_AUTHORITATIVE,
    DEFAULT_FRESHNESS,
    EFFECT_CLASS,
    EXTRACTOR_IDENTITY,
    INITIAL_BOUNDARY_CONCERNS,
    INITIAL_BOUNDARY_SOURCE_BINDINGS,
    MAX_BOUNDARY_MEMBERS,
    RANKING_IS_NON_PROBATIVE,
    REQUIRED_COST_DIMENSIONS,
    REQUIRED_HARD_CONSTRAINTS,
    REQUIRED_PROPOSAL_DECLARATIONS,
    ROLLBACK_DECLARATION,
    SYNTHESIZER_CAN_APPLY_CANDIDATE,
    SYNTHESIZER_CAN_AUTHORIZE_CHANGES,
    SYNTHESIZER_CAN_CREATE_AUTHORITY,
    SYNTHESIZER_CAN_PROMOTE_CANDIDATE,
    SYNTHESIZER_CAN_TRANSFER_AUTHORITY,
    TASK_ID,
    BoundaryConcern,
    BoundaryCostDimension,
    BoundaryProposal,
    BoundarySourceBinding,
    BoundarySynthesisResult,
    BoundarySynthesizerAuthorityError,
    BoundarySynthesizerError,
    HardConstraintKind,
    InterfaceBoundarySynthesizer,
    ProposalDisposition,
    ProposalTier,
    build_boundary_synthesis_result,
    check_hard_constraints,
    hard_constraints_include_non_compensable_invariants,
    initial_boundary_concerns,
    rank_boundary_proposals,
    refuse_authority_creation,
    refuse_authority_transfer,
    refuse_candidate_application,
    refuse_candidate_promotion,
    refuse_change_authorization,
    synthesize_boundaries,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.contract_extractor import (
    AmbiguityKind,
    ContractDimension,
    ContractEvidenceSource,
    ContractEvidenceUnit,
    extract_contract_candidates,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.contracts import (
    Confidence,
    EdgeKind,
    NodeKind,
    SourceFactIdentity,
    SourceSpan,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.entropy import (
    NON_COMPENSABLE_INVARIANTS,
    measure_semantic_entropy,
)
from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

_TREE = "a698da9e4b54e2929adacb613bc61ba3e72eed58"
_FRESHNESS = "pcar-011-fixture"
_EXTRACTOR = "pcar-011-fixture"
_AUTH_PATH = "ipfs_accelerate_py/agent_supervisor/control/control_plane.py"
_CAP_PATH = "ipfs_accelerate_py/agent_supervisor/control/capability_resolver.py"
_EXEC_PATH = "ipfs_accelerate_py/agent_supervisor/contracts/execution.py"
_CTX_PATH = "ipfs_accelerate_py/agent_supervisor/context/context_compiler.py"
_PROOF_PATH = "ipfs_accelerate_py/agent_supervisor/verification/planner.py"
_STATE_PATH = "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py"
_CTRL_PATH = "ipfs_accelerate_py/agent_supervisor/control/control_contracts.py"
_RCPT_PATH = "ipfs_accelerate_py/agent_supervisor/todo_daemon/authoritative_completion.py"
_LEGACY_PATH = "ipfs_accelerate_py/agent_supervisor/todo_daemon/legacy_landed_review.py"
_SIM_PATH = "ipfs_accelerate_py/agent_supervisor/runtime/provider_usage.py"
_TEST_PATH = "test/api/architecture_refactorer/test_boundary_synthesizer.py"


def _span(path: str, start: int, end: int | None = None) -> SourceSpan:
    return SourceSpan(path, start, start if end is None else end)


def _fact(
    path: str,
    start: int,
    *,
    confidence: Confidence = Confidence.EXACT,
    end: int | None = None,
) -> SourceFactIdentity:
    return SourceFactIdentity(
        extractor_identity=_EXTRACTOR,
        span=_span(path, start, end),
        confidence=confidence,
        freshness=_FRESHNESS,
        repository_tree=_TREE,
    )


def _node(
    node_id: str,
    kind: NodeKind,
    path: str,
    start: int,
    *,
    confidence: Confidence = Confidence.EXACT,
) -> ArchitectureNode:
    return ArchitectureNode(
        node_id=node_id,
        kind=kind,
        provenance=_fact(path, start, confidence=confidence),
    )


def _edge(
    edge_id: str,
    kind: EdgeKind,
    source: str,
    target: str,
    path: str,
    start: int,
    *,
    confidence: Confidence = Confidence.EXACT,
) -> ArchitectureEdge:
    return ArchitectureEdge(
        edge_id=edge_id,
        kind=kind,
        source=source,
        target=target,
        provenance=_fact(path, start, confidence=confidence),
    )


def _graph(
    nodes: tuple[ArchitectureNode, ...],
    edges: tuple[ArchitectureEdge, ...] = (),
) -> ArchitectureIR:
    return ArchitectureIR.from_parts(
        repository_tree=_TREE,
        freshness=_FRESHNESS,
        nodes=nodes,
        edges=edges,
    )


def _coherent_graph() -> ArchitectureIR:
    return _graph(
        (
            _node("n-auth", NodeKind.AUTHORITY, _AUTH_PATH, 10),
            _node("n-prov", NodeKind.PROVIDER, _CAP_PATH, 20),
            _node("n-op", NodeKind.OPERATION, _CTRL_PATH, 30),
            _node("n-entry", NodeKind.ENTRYPOINT, _CTRL_PATH, 40),
            _node("n-policy", NodeKind.POLICY, _CTRL_PATH, 50),
            _node("n-receipt", NodeKind.RECEIPT, _RCPT_PATH, 60),
            _node("n-proof", NodeKind.PROOF, _PROOF_PATH, 70),
            _node("n-test", NodeKind.TEST, _TEST_PATH, 80),
            _node("n-state", NodeKind.STATE, _STATE_PATH, 90),
            _node("n-schema", NodeKind.SCHEMA, _CTX_PATH, 100),
            _node("n-mod", NodeKind.MODULE, _CTX_PATH, 110),
            _node("n-iface", NodeKind.INTERFACE, _CTRL_PATH, 120),
            _node("n-compat", NodeKind.COMPATIBILITY, _LEGACY_PATH, 130),
            _node("n-sim", NodeKind.SIMULATION, _SIM_PATH, 140),
            _node("n-exec", NodeKind.OPERATION, _EXEC_PATH, 150),
        ),
        (
            _edge("e-auth-prov", EdgeKind.AUTHORIZES, "n-auth", "n-prov", _CAP_PATH, 20),
            _edge("e-auth-op", EdgeKind.AUTHORIZES, "n-auth", "n-op", _CTRL_PATH, 30),
            _edge("e-auth-entry", EdgeKind.AUTHORIZES, "n-auth", "n-entry", _CTRL_PATH, 40),
            _edge("e-auth-policy", EdgeKind.AUTHORIZES, "n-auth", "n-policy", _CTRL_PATH, 50),
            _edge("e-auth-receipt", EdgeKind.AUTHORIZES, "n-auth", "n-receipt", _RCPT_PATH, 60),
            _edge("e-auth-state", EdgeKind.AUTHORIZES, "n-auth", "n-state", _STATE_PATH, 90),
            _edge("e-auth-schema", EdgeKind.AUTHORIZES, "n-auth", "n-schema", _CTX_PATH, 100),
            _edge("e-auth-mod", EdgeKind.AUTHORIZES, "n-auth", "n-mod", _CTX_PATH, 110),
            _edge("e-auth-exec", EdgeKind.AUTHORIZES, "n-auth", "n-exec", _EXEC_PATH, 150),
            _edge("e-auth-sim", EdgeKind.AUTHORIZES, "n-auth", "n-sim", _SIM_PATH, 140),
            _edge("e-executes", EdgeKind.EXECUTES, "n-entry", "n-op", _CTRL_PATH, 41),
            _edge("e-implements", EdgeKind.IMPLEMENTS, "n-entry", "n-iface", _CTRL_PATH, 42),
            _edge("e-policy", EdgeKind.EVALUATES_POLICY, "n-policy", "n-op", _CTRL_PATH, 51),
            _edge("e-persists", EdgeKind.PERSISTS, "n-op", "n-state", _STATE_PATH, 91),
            _edge("e-observes", EdgeKind.OBSERVES, "n-op", "n-receipt", _RCPT_PATH, 61),
            _edge("e-reads", EdgeKind.READS, "n-op", "n-schema", _CTX_PATH, 101),
            _edge("e-contains", EdgeKind.CONTAINS, "n-mod", "n-schema", _CTX_PATH, 111),
            _edge("e-tests", EdgeKind.TESTS, "n-test", "n-op", _TEST_PATH, 81),
            _edge("e-proves", EdgeKind.PROVES, "n-proof", "n-op", _PROOF_PATH, 71),
            _edge("e-adapts", EdgeKind.ADAPTS, "n-compat", "n-auth", _LEGACY_PATH, 131),
            _edge("e-fallback", EdgeKind.FALLBACKS_TO, "n-auth", "n-sim", _SIM_PATH, 141),
            _edge("e-exec-calls", EdgeKind.CALLS, "n-exec", "n-op", _EXEC_PATH, 151),
        ),
    )


def _synthesize(
    graph: ArchitectureIR | None = None,
    **kwargs: object,
) -> BoundarySynthesisResult:
    architecture = graph if graph is not None else _coherent_graph()
    return synthesize_boundaries(architecture, **kwargs)


def test_closed_vocabulary_and_synthesizer_invariants() -> None:
    assert BOUNDARY_PROPOSAL_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/boundary-proposal@1"
    )
    assert BOUNDARY_PROPOSAL_VERSION == 1
    assert BOUNDARY_PROPOSAL_EVIDENCE == "pcar/boundary-proposal@1"
    assert BOUNDARY_SYNTHESIS_SCHEMA.endswith("boundary-synthesis-result@1")
    assert BOUNDARY_SYNTHESIS_EVIDENCE == "pcar/boundary-proposal@1"
    assert EXTRACTOR_IDENTITY == "pcar-011-interface-boundary-synthesizer"
    assert TASK_ID == "PCAR-011"
    assert DEFAULT_FRESHNESS == "pcar-011-boundary-synthesis"
    assert EFFECT_CLASS == "read_only_planning"
    assert SYNTHESIZER_CAN_APPLY_CANDIDATE is False
    assert SYNTHESIZER_CAN_TRANSFER_AUTHORITY is False
    assert SYNTHESIZER_CAN_CREATE_AUTHORITY is False
    assert SYNTHESIZER_CAN_PROMOTE_CANDIDATE is False
    assert SYNTHESIZER_CAN_AUTHORIZE_CHANGES is False
    assert CANDIDATE_TIER_ONLY is True
    assert RANKING_IS_NON_PROBATIVE is True
    assert COMPLETION_AUTHORITATIVE is False
    assert MAX_BOUNDARY_MEMBERS == 32
    assert tuple(item.value for item in INITIAL_BOUNDARY_CONCERNS) == (
        "provider capability/selection",
        "execution requests/outcomes",
        "analysis/context",
        "proof and verification scheduling",
        "task/objective state",
        "control operations",
        "receipt/evidence queries",
        "legacy compatibility",
        "simulations",
    )
    assert initial_boundary_concerns() == INITIAL_BOUNDARY_CONCERNS
    assert CLOSED_BOUNDARY_CONCERNS == {item.value for item in BoundaryConcern}
    assert CLOSED_PROPOSAL_DISPOSITIONS == {"admitted", "rejected"}
    assert CLOSED_PROPOSAL_TIERS == {"candidate"}
    assert CLOSED_HARD_CONSTRAINTS == {item.value for item in HardConstraintKind}
    assert CLOSED_COST_DIMENSIONS == {item.value for item in BoundaryCostDimension}
    assert tuple(item.value for item in REQUIRED_COST_DIMENSIONS) == (
        "cross_boundary_effects",
        "mutable_sharing",
        "cycles",
        "public_symbols",
        "change_amplification",
        "context_amplification",
        "validation_amplification",
        "dependency_cone",
    )
    assert set(NON_COMPENSABLE_INVARIANTS) <= CLOSED_HARD_CONSTRAINTS
    assert hard_constraints_include_non_compensable_invariants() is True
    assert tuple(item.value for item in REQUIRED_HARD_CONSTRAINTS[:3]) == (
        "NoAuthorityWeakening",
        "NoEffectExpansion",
        "NoHiddenBehaviorChange",
    )
    bindings = {item.concern for item in INITIAL_BOUNDARY_SOURCE_BINDINGS}
    assert bindings == set(INITIAL_BOUNDARY_CONCERNS)
    assert all(isinstance(item, BoundarySourceBinding) for item in INITIAL_BOUNDARY_SOURCE_BINDINGS)
    with pytest.raises(ValueError):
        BoundaryConcern("aesthetic score")
    with pytest.raises(ValueError):
        HardConstraintKind("ignore")
    with pytest.raises(ValueError):
        ProposalTier("requirement")
    with pytest.raises(ValueError):
        ProposalDisposition("maybe")
    with pytest.raises(ValueError):
        BoundaryCostDimension("score")


def test_initial_boundaries_are_complete_and_stable() -> None:
    result = _synthesize()
    assert result.concerns == tuple(item.value for item in INITIAL_BOUNDARY_CONCERNS)
    assert len(result.proposals) == len(INITIAL_BOUNDARY_CONCERNS)
    assert {item.concern for item in result.proposals} == set(INITIAL_BOUNDARY_CONCERNS)
    empty = _synthesize(_graph((_node("n-auth", NodeKind.AUTHORITY, _AUTH_PATH, 1),)))
    assert empty.concerns == result.concerns
    assert len(empty.proposals) == 9
    for proposal in empty.proposals:
        assert proposal.disposition is ProposalDisposition.REJECTED
        assert proposal.constraint(HardConstraintKind.NO_UNRESOLVED_AUTHORITY).satisfied is False


def test_coherent_authority_is_preserved_and_incoherence_is_rejected() -> None:
    result = _synthesize()
    control = result.proposal(BoundaryConcern.CONTROL_OPERATIONS)
    assert control.disposition is ProposalDisposition.ADMITTED
    assert control.canonical_owner == "n-auth"
    assert control.constraint(HardConstraintKind.NO_INCOHERENT_AUTHORITY).satisfied is True
    assert control.constraint(HardConstraintKind.NO_AUTHORITY_WEAKENING).satisfied is True
    provider = result.proposal(BoundaryConcern.PROVIDER_CAPABILITY_SELECTION)
    assert provider.admitted is True
    assert provider.canonical_owner == "n-auth"
    assert "n-prov" in provider.members
    assert "n-sim" not in provider.members
    split = _graph(
        (
            _node("n-auth-a", NodeKind.AUTHORITY, _AUTH_PATH, 10),
            _node("n-auth-b", NodeKind.AUTHORITY, _AUTH_PATH, 11),
            _node("n-op", NodeKind.OPERATION, _CTRL_PATH, 30),
            _node("n-entry", NodeKind.ENTRYPOINT, _CTRL_PATH, 40),
        ),
        (
            _edge("e-a", EdgeKind.AUTHORIZES, "n-auth-a", "n-op", _CTRL_PATH, 30),
            _edge("e-b", EdgeKind.AUTHORIZES, "n-auth-b", "n-entry", _CTRL_PATH, 40),
            _edge("e-exec", EdgeKind.EXECUTES, "n-entry", "n-op", _CTRL_PATH, 41),
        ),
    )
    rejected = _synthesize(split).proposal(BoundaryConcern.CONTROL_OPERATIONS)
    assert rejected.disposition is ProposalDisposition.REJECTED
    assert rejected.constraint(HardConstraintKind.NO_INCOHERENT_AUTHORITY).satisfied is False
    assert HardConstraintKind.NO_INCOHERENT_AUTHORITY.value in rejected.rejection_reasons
    mapped = _synthesize(
        _coherent_graph(),
        ownership={BoundaryConcern.CONTROL_OPERATIONS.value: "n-auth"},
    )
    assert mapped.proposal(BoundaryConcern.CONTROL_OPERATIONS).canonical_owner == "n-auth"


def test_complete_proposal_declares_required_fields() -> None:
    result = _synthesize()
    for proposal in result.proposals:
        payload = proposal.to_dict()
        for field in REQUIRED_PROPOSAL_DECLARATIONS:
            assert field in payload, field
        assert payload["required_interface"]["name"]
        assert payload["rollback"] == ROLLBACK_DECLARATION
        assert payload["tier"] == "candidate"
        assert payload["can_apply"] is False
        assert payload["can_promote"] is False
        assert payload["can_transfer_authority"] is False
        assert payload["can_create_authority"] is False
        assert payload["completion_authoritative"] is False
        assert payload["ranking_is_non_probative"] is True
        assert len(proposal.hard_constraints) == len(REQUIRED_HARD_CONSTRAINTS)
        assert tuple(item.constraint for item in proposal.hard_constraints) == (
            REQUIRED_HARD_CONSTRAINTS
        )
        assert proposal.predicted_cone_reduction == proposal.cost.reductions.dependency_cone
        assert (
            proposal.predicted_context_reduction
            == proposal.cost.reductions.context_amplification
        )
        assert proposal.ranking_inputs is not None
        assert proposal.ranking_inputs.concern == proposal.concern.value
    admitted = result.proposal(BoundaryConcern.CONTROL_OPERATIONS)
    assert admitted.admitted is True
    assert admitted.canonical_owner == "n-auth"
    assert admitted.state_owner == "n-state"
    assert "n-entry" in admitted.members or "n-op" in admitted.members
    assert admitted.tests == ("n-test",)
    assert admitted.proofs == ("n-proof",)
    assert admitted.allowed_effects
    assert "persists:n-state" in admitted.allowed_effects


def test_hard_constraints_reject_cycles_sharing_simulation_and_ambiguity() -> None:
    cyclic = _graph(
        (
            _node("n-auth", NodeKind.AUTHORITY, _AUTH_PATH, 10),
            _node("n-op", NodeKind.OPERATION, _CTRL_PATH, 30),
            _node(
                "n-ext",
                NodeKind.SYMBOL,
                "ipfs_accelerate_py/agent_supervisor/runtime/helper.py",
                31,
            ),
        ),
        (
            _edge("e-auth", EdgeKind.AUTHORIZES, "n-auth", "n-op", _CTRL_PATH, 30),
            _edge(
                "e-out",
                EdgeKind.CALLS,
                "n-op",
                "n-ext",
                "ipfs_accelerate_py/agent_supervisor/runtime/helper.py",
                32,
            ),
            _edge(
                "e-back",
                EdgeKind.CALLS,
                "n-ext",
                "n-op",
                "ipfs_accelerate_py/agent_supervisor/runtime/helper.py",
                33,
            ),
        ),
    )
    cycle_proposal = _synthesize(cyclic).proposal(BoundaryConcern.CONTROL_OPERATIONS)
    assert cycle_proposal.disposition is ProposalDisposition.REJECTED
    assert cycle_proposal.constraint(HardConstraintKind.NO_BOUNDARY_CYCLE).satisfied is False
    shared = _graph(
        (
            _node("n-auth", NodeKind.AUTHORITY, _AUTH_PATH, 10),
            _node("n-op", NodeKind.OPERATION, _CTRL_PATH, 30),
            _node("n-out", NodeKind.SYMBOL, _EXEC_PATH, 31),
            _node("n-state", NodeKind.STATE, _STATE_PATH, 90),
        ),
        (
            _edge("e-auth", EdgeKind.AUTHORIZES, "n-auth", "n-op", _CTRL_PATH, 30),
            _edge("e-mut-a", EdgeKind.MUTATES, "n-op", "n-state", _STATE_PATH, 91),
            _edge("e-mut-b", EdgeKind.MUTATES, "n-out", "n-state", _STATE_PATH, 92),
        ),
    )
    sharing = _synthesize(shared).proposal(BoundaryConcern.CONTROL_OPERATIONS)
    assert sharing.disposition is ProposalDisposition.REJECTED
    assert sharing.constraint(HardConstraintKind.NO_CROSS_BOUNDARY_MUTABLE_SHARING).satisfied is False
    leaked = _graph(
        (
            _node("n-auth", NodeKind.AUTHORITY, _AUTH_PATH, 10),
            _node("n-op", NodeKind.OPERATION, _CTRL_PATH, 30),
            _node("n-sim", NodeKind.SIMULATION, _SIM_PATH, 140),
        ),
        (
            _edge("e-auth", EdgeKind.AUTHORIZES, "n-auth", "n-op", _CTRL_PATH, 30),
            _edge("e-call-sim", EdgeKind.CALLS, "n-op", "n-sim", _SIM_PATH, 141),
        ),
    )
    live = _synthesize(leaked).proposal(BoundaryConcern.SIMULATIONS)
    assert live.disposition is ProposalDisposition.REJECTED
    assert live.constraint(HardConstraintKind.NO_SIMULATED_AS_LIVE).satisfied is False
    graph = _coherent_graph()
    contracts = extract_contract_candidates(
        (
            ContractEvidenceUnit(
                subject="n-op",
                source_kind=ContractEvidenceSource.TYPE,
                dimension=ContractDimension.EFFECTS,
                value="writes:store-a",
                provenance=_fact(_CTRL_PATH, 30),
            ),
            ContractEvidenceUnit(
                subject="n-op",
                source_kind=ContractEvidenceSource.TEST,
                dimension=ContractDimension.EFFECTS,
                value="writes:store-b",
                provenance=_fact(_TEST_PATH, 80),
            ),
        ),
        repository_tree=_TREE,
        freshness=_FRESHNESS,
        architecture=graph,
    )
    assert any(item.kind is AmbiguityKind.CONFLICTING_VALUES for item in contracts.ambiguities)
    ambiguous = _synthesize(graph, contracts=contracts)
    control = ambiguous.proposal(BoundaryConcern.CONTROL_OPERATIONS)
    assert control.disposition is ProposalDisposition.REJECTED
    assert control.constraint(HardConstraintKind.NO_UNRESOLVED_AMBIGUITY).satisfied is False


def test_hard_constraint_counterexamples_are_recorded() -> None:
    graph = _coherent_graph()
    view = synthesize_boundaries(graph)
    control = view.proposal(BoundaryConcern.CONTROL_OPERATIONS)
    from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.boundary_synthesizer import (
        _build_view,
    )

    built = _build_view(graph)
    members = tuple(built.nodes_by_id[node_id] for node_id in control.members)
    expanded = check_hard_constraints(
        concern=BoundaryConcern.CONTROL_OPERATIONS,
        view=built,
        members=members,
        canonical_owner="n-auth",
        competing_owners=("n-auth",),
        blocked_concerns=(),
        state_ids=("n-state",),
        callers=control.allowed_callers,
        effects=control.allowed_effects,
        tests=control.tests,
        proofs=control.proofs,
        adapters=control.migration_adapters,
        deprecations=(),
        proposed_effects=(*control.allowed_effects, "writes:sibling-store"),
        predicted_tests=(),
        predicted_proofs=(),
        ambiguities=(),
        cyclic=(),
        shared_states=(),
    )
    by_kind = {item.constraint: item for item in expanded}
    assert by_kind[HardConstraintKind.NO_EFFECT_EXPANSION].satisfied is False
    assert by_kind[HardConstraintKind.NO_VALIDATION_REDUCTION].satisfied is False
    assert by_kind[HardConstraintKind.NO_PROOF_OBLIGATION_LOSS].satisfied is False
    assert (
        by_kind[
            HardConstraintKind.NO_PUBLIC_CONTRACT_BREAK_WITHOUT_VERSIONED_MIGRATION
        ].satisfied
        is False
    )
    sibling = _graph(
        (
            _node("n-auth", NodeKind.AUTHORITY, _AUTH_PATH, 10),
            _node(
                "n-op",
                NodeKind.OPERATION,
                "ipfs_datasets_py/agent_supervisor/control.py",
                30,
            ),
        ),
        (_edge("e-auth", EdgeKind.AUTHORIZES, "n-auth", "n-op", _AUTH_PATH, 10),),
    )
    leaked = _synthesize(sibling).proposal(BoundaryConcern.CONTROL_OPERATIONS)
    assert leaked.constraint(HardConstraintKind.NO_CROSS_REPOSITORY_WRITE).satisfied is False
    secret = _graph(
        (
            _node("n-auth", NodeKind.AUTHORITY, _AUTH_PATH, 10),
            _node(
                "n-op",
                NodeKind.OPERATION,
                "ipfs_accelerate_py/agent_supervisor/secrets/private_key.py",
                30,
            ),
        ),
        (
            _edge(
                "e-auth",
                EdgeKind.AUTHORIZES,
                "n-auth",
                "n-op",
                "ipfs_accelerate_py/agent_supervisor/secrets/private_key.py",
                30,
            ),
        ),
    )
    exposed = _synthesize(secret).proposal(BoundaryConcern.CONTROL_OPERATIONS)
    assert exposed.constraint(HardConstraintKind.NO_SECRET_OR_PRIVATE_DATA_LEAK).satisfied is False


def test_admitted_proposals_minimize_cost_without_dropping_validation() -> None:
    result = _synthesize()
    control = result.proposal(BoundaryConcern.CONTROL_OPERATIONS)
    assert control.admitted is True
    assert control.cost.after.cross_boundary_effects <= control.cost.before.cross_boundary_effects
    assert control.cost.after.dependency_cone <= control.cost.before.dependency_cone
    assert (
        control.cost.after.validation_amplification
        == control.cost.before.validation_amplification
    )
    assert control.predicted_cone_reduction == control.cost.reductions.dependency_cone
    assert control.cost.reductions.validation_amplification == 0
    for dimension in REQUIRED_COST_DIMENSIONS:
        assert control.cost.before.dimension(dimension) >= 0
        assert control.cost.after.dimension(dimension) >= 0
        assert control.cost.reductions.dimension(dimension) == max(
            0,
            control.cost.before.dimension(dimension) - control.cost.after.dimension(dimension),
        )
    ranked = rank_boundary_proposals(result.proposals)
    assert ranked == tuple(
        sorted(result.admitted_proposals, key=lambda item: item.ranking_inputs.sort_key())
    )
    assert result.ranked_proposal_identities == tuple(item.content_identity for item in ranked)
    reversed_nodes = tuple(reversed(_coherent_graph().nodes))
    reversed_edges = tuple(reversed(_coherent_graph().edges))
    again = _synthesize(_graph(reversed_nodes, reversed_edges))
    assert again.content_identity == result.content_identity
    assert again.ranked_proposal_identities == result.ranked_proposal_identities


def test_legacy_and_simulation_quarantine_are_not_production_owners() -> None:
    result = _synthesize()
    legacy = result.proposal(BoundaryConcern.LEGACY_COMPATIBILITY)
    simulation = result.proposal(BoundaryConcern.SIMULATIONS)
    if legacy.admitted:
        assert legacy.canonical_owner == "n-auth"
        assert "n-compat" in legacy.members
        assert legacy.constraint(HardConstraintKind.NO_SIMULATED_AS_LIVE).satisfied is True
    if simulation.admitted:
        assert simulation.canonical_owner == "n-auth"
        assert "n-sim" in simulation.members
        assert simulation.required_interface.name == "simulation.quarantine"
        assert simulation.constraint(HardConstraintKind.NO_SIMULATED_AS_LIVE).satisfied is True
    assert result.proposal(BoundaryConcern.CONTROL_OPERATIONS).canonical_owner != "n-sim"
    assert result.proposal(BoundaryConcern.CONTROL_OPERATIONS).canonical_owner != "n-compat"


def test_entropy_and_ownership_identities_bind_and_cannot_prove_safety() -> None:
    graph = _coherent_graph()
    entropy = measure_semantic_entropy(graph)
    result = _synthesize(graph, entropy=entropy, ownership={"control operations": "n-auth"})
    assert result.entropy_identity == entropy.content_identity
    assert result.architecture_ir_identity == graph.content_identity
    control = result.proposal(BoundaryConcern.CONTROL_OPERATIONS)
    assert control.ranking_is_non_probative is True
    assert control.constraint(HardConstraintKind.NO_FALSE_COMPLETION).satisfied is True
    mismatched = measure_semantic_entropy(
        _graph((_node("n-other", NodeKind.AUTHORITY, _AUTH_PATH, 1),))
    )
    with pytest.raises(BoundarySynthesizerError, match="entropy architecture_ir_identity"):
        synthesize_boundaries(graph, entropy=mismatched)


def test_synthesizer_cannot_apply_promote_or_transfer() -> None:
    synthesizer = InterfaceBoundarySynthesizer()
    result = synthesizer.synthesize(_coherent_graph())
    assert result == build_boundary_synthesis_result(_coherent_graph())
    proposal = result.proposal(BoundaryConcern.CONTROL_OPERATIONS)
    with pytest.raises(BoundarySynthesizerAuthorityError, match="cannot apply"):
        synthesizer.apply(proposal)
    with pytest.raises(BoundarySynthesizerAuthorityError, match="cannot promote"):
        synthesizer.promote(proposal)
    with pytest.raises(BoundarySynthesizerAuthorityError, match="cannot transfer"):
        synthesizer.transfer_authority("n-auth")
    with pytest.raises(BoundarySynthesizerAuthorityError, match="cannot create"):
        synthesizer.create_authority("n-new")
    with pytest.raises(BoundarySynthesizerAuthorityError, match="cannot authorize"):
        synthesizer.authorize_change("merge")
    with pytest.raises(BoundarySynthesizerAuthorityError, match="cannot apply"):
        result.apply()
    with pytest.raises(BoundarySynthesizerAuthorityError, match="cannot promote"):
        proposal.promote()
    with pytest.raises(BoundarySynthesizerAuthorityError, match="cannot apply"):
        refuse_candidate_application("apply")
    with pytest.raises(BoundarySynthesizerAuthorityError, match="cannot promote"):
        refuse_candidate_promotion("promote")
    with pytest.raises(BoundarySynthesizerAuthorityError, match="cannot transfer"):
        refuse_authority_transfer("transfer")
    with pytest.raises(BoundarySynthesizerAuthorityError, match="cannot create"):
        refuse_authority_creation("create")
    with pytest.raises(BoundarySynthesizerAuthorityError, match="cannot authorize"):
        refuse_change_authorization("change")
    with pytest.raises(BoundarySynthesizerAuthorityError, match="cannot apply"):
        InterfaceBoundarySynthesizer(can_apply=True)
    with pytest.raises(BoundarySynthesizerAuthorityError, match="cannot transfer"):
        InterfaceBoundarySynthesizer(can_transfer_authority=True)
    with pytest.raises(BoundarySynthesizerAuthorityError, match="cannot create"):
        InterfaceBoundarySynthesizer(can_create_authority=True)
    with pytest.raises(BoundarySynthesizerAuthorityError, match="cannot promote"):
        InterfaceBoundarySynthesizer(can_promote=True)
    with pytest.raises(BoundarySynthesizerAuthorityError, match="cannot authorize"):
        InterfaceBoundarySynthesizer(can_authorize_changes=True)


def test_round_trip_and_canonical_identity() -> None:
    result = _synthesize()
    payload = result.to_dict()
    restored = BoundarySynthesisResult.from_mapping(payload)
    assert restored == result
    assert restored.to_dict() == payload
    assert restored.to_json() == json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert BoundarySynthesisResult.from_json(restored.to_json()) == result
    claimed = payload.pop("content_identity")
    validate_cid(claimed, codecs=("dag-json",))
    assert claimed == cid_for_dag_json(payload)
    assert claimed == result.content_identity
    assert not claimed.startswith("sha256:")
    proposal = result.proposal(BoundaryConcern.CONTROL_OPERATIONS)
    proposal_payload = proposal.to_dict()
    assert BoundaryProposal.from_mapping(proposal_payload) == proposal
    assert BoundaryProposal.from_json(proposal.to_json()) == proposal
    for field in REQUIRED_PROPOSAL_DECLARATIONS:
        assert field in proposal_payload


def test_unknown_fields_and_identity_mismatch_are_rejected() -> None:
    payload = _synthesize().to_dict()
    payload["aesthetic"] = 1
    with pytest.raises(BoundarySynthesizerError, match="unknown boundary-proposal field"):
        BoundarySynthesisResult.from_mapping(payload)
    clean = _synthesize().to_dict()
    forged_body = {key: value for key, value in clean.items() if key != "content_identity"}
    forged_body["freshness"] = "pcar-011-forged"
    clean["content_identity"] = cid_for_dag_json(forged_body)
    with pytest.raises(BoundarySynthesizerError, match="content identity mismatch"):
        BoundarySynthesisResult.from_mapping(clean)
    proposal = _synthesize().proposal(BoundaryConcern.ANALYSIS_CONTEXT).to_dict()
    proposal["score"] = 12
    with pytest.raises(BoundarySynthesizerError, match="unknown boundary-proposal field"):
        BoundaryProposal.from_mapping(proposal)
    with pytest.raises(BoundarySynthesizerError, match="must match ArchitectureIR"):
        synthesize_boundaries(_coherent_graph(), freshness="other-freshness")
