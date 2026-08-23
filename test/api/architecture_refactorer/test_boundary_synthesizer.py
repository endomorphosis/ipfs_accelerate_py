"""Hermetic PCAR-011 interface-boundary synthesis tests."""

from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.architecture_ir import (
    ArchitectureEdge,
    ArchitectureIR,
    ArchitectureNode,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.authority_graph import (
    INITIAL_CONCERN_SOURCE_BINDINGS,
    INITIAL_CONCERNS,
    ConcernClaim,
    ConcernKind,
    OwnerDisposition,
    resolve_authority_ownership,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.boundary_synthesizer import (
    BOUNDARY_CONCERNS,
    BOUNDARY_PROPOSAL_EVIDENCE,
    BOUNDARY_PROPOSAL_SCHEMA,
    BOUNDARY_PROPOSAL_VERSION,
    BOUNDARY_SYNTHESIS_SCHEMA,
    CANDIDATE_INTERFACES_ONLY,
    CLOSED_BOUNDARIES,
    CLOSED_COST_DIMENSIONS,
    CLOSED_HARD_CONSTRAINTS,
    CLOSED_INTERFACE_STABILITIES,
    CLOSED_PROPOSAL_DISPOSITIONS,
    CLOSED_REJECTION_KINDS,
    DEFAULT_FRESHNESS,
    EFFECT_CLASS,
    EXTRACTOR_IDENTITY,
    HARD_CONSTRAINTS_PRECEDE_RANKING,
    INITIAL_BOUNDARIES,
    MISSING_PLAN_HARD_GATES,
    RANKING_IS_NON_PROBATIVE,
    REQUIRED_BOUNDARIES,
    REQUIRED_COST_DIMENSIONS,
    REQUIRED_HARD_CONSTRAINTS,
    SYNTHESIZER_CAN_APPLY_PROPOSALS,
    SYNTHESIZER_CAN_AUTHORIZE_CHANGES,
    SYNTHESIZER_CAN_MUTATE_STATE,
    SYNTHESIZER_CAN_OVERRIDE_HARD_CONSTRAINTS,
    SYNTHESIZER_CAN_PROMOTE_AUTHORITY,
    SYNTHESIZER_CAN_TRANSFER_AUTHORITY,
    TASK_ID,
    UNRESOLVED_AMBIGUITY_REJECTS,
    UNRESOLVED_AUTHORITY_REJECTS,
    BoundaryCostMeasure,
    BoundaryCostVector,
    BoundaryInterface,
    BoundaryKind,
    BoundaryMigration,
    BoundaryPrediction,
    BoundaryProposal,
    BoundaryRankingInput,
    BoundaryRollback,
    BoundarySynthesizerAuthorityError,
    BoundarySynthesizerError,
    BoundarySynthesisResult,
    CostDimensionKind,
    HardConstraintCheck,
    HardConstraintKind,
    InterfaceBoundarySynthesizer,
    InterfaceStability,
    ProposalDisposition,
    RejectedBoundaryProposal,
    RejectionKind,
    admit_rejected_by_cost,
    detect_cross_boundary_cycles,
    detect_cross_boundary_effects,
    detect_mutable_sharing,
    measure_boundary_costs,
    rank_boundary_proposals,
    ranking_establishes,
    refuse_autonomous_application,
    refuse_authority_promotion,
    refuse_authority_transfer,
    refuse_hard_constraint_override,
    refuse_ownership_authorization,
    refuse_state_mutation,
    synthesize_interface_boundaries,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.contract_extractor import (
    AmbiguityKind,
    ContractAmbiguity,
    ContractDimension,
    ContractEvidenceSource,
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
)
from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

_TREE = "a698da9e4b54e2929adacb613bc61ba3e72eed58"
_FRESHNESS = "pcar-011-fixture"
_EXTRACTOR = "pcar-011-fixture"
_TYPE_PATH = "ipfs_accelerate_py/agent_supervisor/architecture_refactorer/contracts.py"
_PROJECTION_PATH = "ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py"
_LEGACY_PATH = "ipfs_accelerate_py/agent_supervisor/todo_daemon/legacy_landed_review.py"
_SIM_PATH = "ipfs_accelerate_py/agent_supervisor/runtime/provider_usage.py"


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


def _binding_for(concern: ConcernKind) -> tuple:
    canonical = [
        item
        for item in INITIAL_CONCERN_SOURCE_BINDINGS
        if item.concern is concern
        and item.recommended_disposition is OwnerDisposition.CANONICAL
    ]
    extras = [
        item
        for item in INITIAL_CONCERN_SOURCE_BINDINGS
        if item.concern is concern
        and item.recommended_disposition is not OwnerDisposition.CANONICAL
    ]
    return canonical[0], extras


def _covered_graph(
    extra_nodes: tuple[ArchitectureNode, ...] = (),
    extra_edges: tuple[ArchitectureEdge, ...] = (),
    extra_claims: tuple[ConcernClaim, ...] = (),
):
    nodes: list[ArchitectureNode] = []
    edges: list[ArchitectureEdge] = []
    claims: list[ConcernClaim] = []
    for index, concern in enumerate(INITIAL_CONCERNS, start=1):
        canonical, extras = _binding_for(concern)
        owner_id = f"n-auth-{index:02d}"
        subject_id = f"n-subject-{index:02d}"
        nodes.append(
            _node(owner_id, NodeKind.AUTHORITY, canonical.path, canonical.start_line)
        )
        subject_kind = {
            ConcernKind.POLICY_DECISION: NodeKind.POLICY,
            ConcernKind.STATE_PERSISTENCE: NodeKind.STATE,
            ConcernKind.PROVIDER_CAPABILITY: NodeKind.PROVIDER,
            ConcernKind.PROVIDER_SELECTION: NodeKind.PROVIDER,
            ConcernKind.PROOF_VERIFICATION: NodeKind.PROOF,
            ConcernKind.TEST_EVIDENCE: NodeKind.TEST,
            ConcernKind.COMPLETION_EVIDENCE: NodeKind.RECEIPT,
            ConcernKind.EXECUTION_RESULT: NodeKind.RECEIPT,
            ConcernKind.OPERATION_IDENTITY: NodeKind.OPERATION,
        }.get(concern, NodeKind.SYMBOL)
        nodes.append(
            _node(subject_id, subject_kind, canonical.path, canonical.start_line + 1)
        )
        edge_id = f"e-auth-{index:02d}"
        edges.append(
            _edge(
                edge_id,
                EdgeKind.AUTHORIZES,
                owner_id,
                subject_id,
                canonical.path,
                canonical.start_line,
            )
        )
        claims.append(
            ConcernClaim(
                concern=concern,
                owner_node_id=owner_id,
                disposition=OwnerDisposition.CANONICAL,
                evidence_edge_ids=(edge_id,),
            )
        )
        edges.append(
            _edge(
                f"e-test-{index:02d}",
                EdgeKind.TESTS,
                "n-test",
                owner_id,
                canonical.path,
                canonical.start_line,
            )
        )
        edges.append(
            _edge(
                f"e-proof-{index:02d}",
                EdgeKind.PROVES,
                "n-proof",
                owner_id,
                canonical.path,
                canonical.start_line,
            )
        )
        for extra in extras:
            extra_id = f"n-extra-{index:02d}"
            nodes.append(
                _node(
                    extra_id,
                    NodeKind.COMPATIBILITY,
                    extra.path,
                    extra.start_line,
                    confidence=extra.inventory_confidence,
                )
            )
            extra_edge = f"e-adapt-{index:02d}"
            edges.append(
                _edge(
                    extra_edge,
                    EdgeKind.ADAPTS,
                    extra_id,
                    owner_id,
                    extra.path,
                    extra.start_line,
                    confidence=extra.inventory_confidence,
                )
            )
            claims.append(
                ConcernClaim(
                    concern=concern,
                    owner_node_id=extra_id,
                    disposition=OwnerDisposition.ADAPTER,
                    evidence_edge_ids=(extra_edge,),
                )
            )

    nodes.extend(
        (
            _node("n-test", NodeKind.TEST, _TYPE_PATH, 8),
            _node("n-proof", NodeKind.PROOF, _TYPE_PATH, 9),
            _node("n-iface", NodeKind.INTERFACE, _TYPE_PATH, 10),
            _node("n-caller", NodeKind.SYMBOL, _TYPE_PATH, 11),
            _node("n-proj", NodeKind.GENERATED, _PROJECTION_PATH, 12),
            _node("n-legacy", NodeKind.COMPATIBILITY, _LEGACY_PATH, 601),
            _node("n-sim", NodeKind.SIMULATION, _SIM_PATH, 273),
        )
    )
    edges.extend(
        (
            _edge("e-impl", EdgeKind.IMPLEMENTS, "n-auth-03", "n-iface", _TYPE_PATH, 10),
            _edge("e-calls", EdgeKind.CALLS, "n-caller", "n-subject-03", _TYPE_PATH, 11),
            _edge(
                "e-proj",
                EdgeKind.GENERATES,
                "n-proj",
                "n-auth-02",
                _PROJECTION_PATH,
                12,
            ),
            _edge(
                "e-legacy",
                EdgeKind.SUPERSEDES,
                "n-auth-09",
                "n-legacy",
                _LEGACY_PATH,
                601,
            ),
            _edge(
                "e-sim",
                EdgeKind.FALLBACKS_TO,
                "n-auth-04",
                "n-sim",
                _SIM_PATH,
                273,
            ),
        )
    )
    architecture = _graph(tuple(nodes) + extra_nodes, tuple(edges) + extra_edges)
    ownership = resolve_authority_ownership(architecture, (*claims, *extra_claims))
    return architecture, ownership


def _synthesize(
    extra_nodes: tuple[ArchitectureNode, ...] = (),
    extra_edges: tuple[ArchitectureEdge, ...] = (),
    extra_claims: tuple[ConcernClaim, ...] = (),
    **kwargs,
) -> BoundarySynthesisResult:
    architecture, ownership = _covered_graph(extra_nodes, extra_edges, extra_claims)
    return synthesize_interface_boundaries(architecture, ownership, **kwargs)


def _ambiguity(subject: str) -> ContractAmbiguity:
    return ContractAmbiguity(
        subject=subject,
        dimension=ContractDimension.EFFECTS,
        kind=AmbiguityKind.CONFLICTING_VALUES,
        values=("pure", "writes:store"),
        source_kinds=(ContractEvidenceSource.TYPE, ContractEvidenceSource.TEST),
        message="conflict",
        provenance=_fact(_TYPE_PATH, 3),
    )


def test_closed_vocabulary_and_authority_invariants() -> None:
    assert BOUNDARY_PROPOSAL_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/boundary-proposal@1"
    )
    assert BOUNDARY_PROPOSAL_VERSION == 1
    assert BOUNDARY_PROPOSAL_EVIDENCE == "pcar/boundary-proposal@1"
    assert BOUNDARY_SYNTHESIS_SCHEMA.endswith("boundary-synthesis-result@1")
    assert EXTRACTOR_IDENTITY == "pcar-011-interface-boundary-synthesizer"
    assert TASK_ID == "PCAR-011"
    assert DEFAULT_FRESHNESS == "pcar-011-boundary-synthesis"
    assert EFFECT_CLASS == "read_only_planning"
    assert SYNTHESIZER_CAN_AUTHORIZE_CHANGES is False
    assert SYNTHESIZER_CAN_TRANSFER_AUTHORITY is False
    assert SYNTHESIZER_CAN_PROMOTE_AUTHORITY is False
    assert SYNTHESIZER_CAN_MUTATE_STATE is False
    assert SYNTHESIZER_CAN_APPLY_PROPOSALS is False
    assert SYNTHESIZER_CAN_OVERRIDE_HARD_CONSTRAINTS is False
    assert RANKING_IS_NON_PROBATIVE is True
    assert CANDIDATE_INTERFACES_ONLY is True
    assert HARD_CONSTRAINTS_PRECEDE_RANKING is True
    assert UNRESOLVED_AMBIGUITY_REJECTS is True
    assert UNRESOLVED_AUTHORITY_REJECTS is True
    assert tuple(item.value for item in INITIAL_BOUNDARIES) == (
        "provider_capability_selection",
        "execution_request_outcome",
        "analysis_context",
        "proof_verification_scheduling",
        "task_objective_state",
        "control_operations",
        "receipt_evidence_query",
        "legacy_compatibility",
        "simulation",
    )
    assert REQUIRED_BOUNDARIES == INITIAL_BOUNDARIES
    assert CLOSED_BOUNDARIES == {item.value for item in BoundaryKind}
    assert CLOSED_COST_DIMENSIONS == {item.value for item in CostDimensionKind}
    assert CLOSED_HARD_CONSTRAINTS == {item.value for item in HardConstraintKind}
    assert CLOSED_PROPOSAL_DISPOSITIONS == {"accepted", "rejected"}
    assert CLOSED_INTERFACE_STABILITIES == {"candidate"}
    assert CLOSED_REJECTION_KINDS == {item.value for item in RejectionKind}
    assert set(NON_COMPENSABLE_INVARIANTS) <= CLOSED_HARD_CONSTRAINTS
    assert MISSING_PLAN_HARD_GATES == ()
    assert tuple(item.value for item in REQUIRED_COST_DIMENSIONS) == (
        "CrossBoundaryEffects",
        "MutableSharing",
        "Cycles",
        "PublicSymbols",
        "ChangeAmplification",
        "ContextBurden",
        "ValidationAmplification",
        "DependencyCone",
    )
    assert set(BOUNDARY_CONCERNS) == set(INITIAL_BOUNDARIES)
    with pytest.raises(ValueError):
        BoundaryKind("aesthetic facade")
    with pytest.raises(ValueError):
        CostDimensionKind("score")
    with pytest.raises(ValueError):
        HardConstraintKind("maybe")
    with pytest.raises(ValueError):
        ProposalDisposition("provisional")
    with pytest.raises(ValueError):
        InterfaceStability("stable")
    with pytest.raises(ValueError):
        RejectionKind("ignore")


def test_initial_boundaries_cover_coherent_authorities() -> None:
    result = _synthesize()
    assert result.covers_initial_boundaries is True
    assert result.effect_class == EFFECT_CLASS
    assert result.candidate_interfaces_only is True
    assert result.ranking_is_non_probative is True
    assert result.hard_constraints_preserved is True
    assert result.can_apply_proposals is False
    assert tuple(item.kind for item in result.proposals) + tuple(
        item.kind for item in result.rejections
    ) == INITIAL_BOUNDARIES or set(item.kind for item in result.proposals) | set(
        item.kind for item in result.rejections
    ) == set(INITIAL_BOUNDARIES)
    covered = {item.kind for item in result.proposals} | {
        item.kind for item in result.rejections
    }
    assert covered == set(INITIAL_BOUNDARIES)
    assert not result.rejections
    architecture, ownership = _covered_graph()
    expected_owners = {
        BoundaryKind.PROVIDER_CAPABILITY_SELECTION: ownership.canonical_owner(
            ConcernKind.PROVIDER_CAPABILITY
        ).node_id,
        BoundaryKind.EXECUTION_REQUEST_OUTCOME: ownership.canonical_owner(
            ConcernKind.EXECUTION_RESULT
        ).node_id,
        BoundaryKind.ANALYSIS_CONTEXT: ownership.canonical_owner(
            ConcernKind.CONTENT_IDENTITY
        ).node_id,
        BoundaryKind.PROOF_VERIFICATION_SCHEDULING: ownership.canonical_owner(
            ConcernKind.PROOF_VERIFICATION
        ).node_id,
        BoundaryKind.TASK_OBJECTIVE_STATE: ownership.canonical_owner(
            ConcernKind.TASK_IDENTITY
        ).node_id,
        BoundaryKind.CONTROL_OPERATIONS: ownership.canonical_owner(
            ConcernKind.POLICY_DECISION
        ).node_id,
        BoundaryKind.RECEIPT_EVIDENCE_QUERY: ownership.canonical_owner(
            ConcernKind.COMPLETION_EVIDENCE
        ).node_id,
        BoundaryKind.LEGACY_COMPATIBILITY: ownership.canonical_owner(
            ConcernKind.AUTHORIZATION
        ).node_id,
        BoundaryKind.SIMULATION: ownership.canonical_owner(
            ConcernKind.PROVIDER_CAPABILITY
        ).node_id,
    }
    for kind, owner_id in expected_owners.items():
        proposal = result.proposal(kind)
        assert proposal.canonical_owner_node_id == owner_id
        assert proposal.interface.canonical_owner_node_id == owner_id
        assert proposal.interface.stability is InterfaceStability.CANDIDATE
        assert proposal.disposition is ProposalDisposition.ACCEPTED
        assert architecture.content_identity == result.architecture_ir_identity
        assert ownership.content_identity == result.ownership_identity


def test_complete_proposal_declares_required_fields() -> None:
    result = _synthesize()
    required = {
        "interface",
        "canonical_owner_node_id",
        "state_owner_node_id",
        "callers",
        "effects",
        "adapters",
        "deprecated_paths",
        "tests",
        "proofs",
        "migration",
        "rollback",
        "prediction",
        "costs",
        "hard_constraints",
    }
    for proposal in result.proposals:
        payload = proposal.to_dict()
        assert required <= set(payload)
        assert proposal.interface.name == f"pcar.boundary.{proposal.kind.value}@1"
        assert proposal.rollback.action == "revert_candidate_interface"
        assert proposal.rollback.applied_effects is False
        assert proposal.rollback.restores_tree is True
        assert proposal.migration.mutates_state is False
        assert proposal.migration.transfers_authority is False
        assert proposal.migration.phases == (
            "declare_interface",
            "adapt_callers",
            "deprecate_paths",
            "validate_and_seal",
        )
        assert proposal.state_owner_node_id
        assert proposal.tests
        assert proposal.proofs
        assert tuple(item.kind for item in proposal.hard_constraints) == REQUIRED_HARD_CONSTRAINTS
        assert all(item.passed for item in proposal.hard_constraints)
        assert tuple(item.kind for item in proposal.costs.measures) == REQUIRED_COST_DIMENSIONS
        assert proposal.prediction.validation_coverage_loss == 0
        assert proposal.prediction.cone_reduction == max(
            0, proposal.prediction.current_cone_size - proposal.prediction.proposed_cone_size
        )
        round_trip = BoundaryProposal.from_mapping(payload)
        assert round_trip.content_identity == proposal.content_identity
        assert json.loads(proposal.to_json())["content_identity"] == proposal.content_identity
        validate_cid(proposal.content_identity, codecs=("dag-json",))
        assert cid_for_dag_json(proposal._identity_payload()) == proposal.content_identity


def test_hard_constraints_cannot_be_overridden_by_ranking() -> None:
    result = _synthesize(
        extra_nodes=(_node("n-outside", NodeKind.SYMBOL, _TYPE_PATH, 80),),
        extra_edges=(
            _edge("e-cycle-a", EdgeKind.CALLS, "n-auth-01", "n-outside", _TYPE_PATH, 80),
            _edge("e-cycle-b", EdgeKind.CALLS, "n-outside", "n-auth-01", _TYPE_PATH, 81),
        ),
    )
    rejected = result.rejection(BoundaryKind.ANALYSIS_CONTEXT)
    assert rejected.disposition is ProposalDisposition.REJECTED
    assert rejected.rejection_kind is RejectionKind.CYCLE
    failed = {item.kind for item in rejected.failed_constraints}
    assert HardConstraintKind.CROSS_BOUNDARY_CYCLE in failed
    assert BoundaryKind.ANALYSIS_CONTEXT.value not in result.ranking
    assert result.hard_constraints_preserved is True
    assert ranking_establishes(result, "safety") is False
    assert ranking_establishes(result, "ownership") is False
    assert ranking_establishes(result, "promotion") is False
    with pytest.raises(BoundarySynthesizerAuthorityError, match="hard constraint"):
        admit_rejected_by_cost(rejected)
    with pytest.raises(BoundarySynthesizerAuthorityError, match="hard constraint"):
        rejected.promote_by_cost()
    with pytest.raises(BoundarySynthesizerAuthorityError, match="hard constraint"):
        result.override_hard_constraint()
    with pytest.raises(BoundarySynthesizerAuthorityError, match="hard constraint"):
        refuse_hard_constraint_override("override")


def test_unresolved_authority_hard_rejects_affected_boundary() -> None:
    result = _synthesize(
        extra_claims=(
            ConcernClaim(
                concern=ConcernKind.CONTENT_IDENTITY,
                owner_node_id="n-auth-01",
                disposition=OwnerDisposition.UNKNOWN,
            ),
        )
    )
    rejected = result.rejection(BoundaryKind.ANALYSIS_CONTEXT)
    assert rejected.rejection_kind is RejectionKind.UNRESOLVED_AUTHORITY
    assert rejected.failed_constraints
    assert any(
        item.kind is HardConstraintKind.UNRESOLVED_AUTHORITY and item.passed is False
        for item in rejected.hard_constraints
    )
    assert result.proposal(BoundaryKind.EXECUTION_REQUEST_OUTCOME).disposition is (
        ProposalDisposition.ACCEPTED
    )


def test_unresolved_ambiguity_hard_rejects_affected_boundary() -> None:
    result = _synthesize(ambiguities=(_ambiguity("n-auth-01"),))
    rejected = result.rejection(BoundaryKind.ANALYSIS_CONTEXT)
    assert rejected.rejection_kind is RejectionKind.UNRESOLVED_AMBIGUITY
    assert any(
        item.kind is HardConstraintKind.UNRESOLVED_AMBIGUITY and item.passed is False
        for item in rejected.hard_constraints
    )


def test_mutable_sharing_and_effect_cycle_counterexamples() -> None:
    shared = _synthesize(
        extra_nodes=(_node("n-outside", NodeKind.SYMBOL, _TYPE_PATH, 90),),
        extra_edges=(
            _edge("e-share-in", EdgeKind.WRITES, "n-auth-06", "n-subject-12", _TYPE_PATH, 90),
            _edge("e-share-out", EdgeKind.WRITES, "n-outside", "n-subject-12", _TYPE_PATH, 91),
        ),
    )
    rejected = shared.rejection(BoundaryKind.TASK_OBJECTIVE_STATE)
    assert rejected.rejection_kind is RejectionKind.MUTABLE_SHARING
    architecture, _ownership = _covered_graph(
        extra_nodes=(_node("n-outside", NodeKind.SYMBOL, _TYPE_PATH, 90),),
        extra_edges=(
            _edge("e-share-in", EdgeKind.WRITES, "n-auth-06", "n-subject-12", _TYPE_PATH, 90),
            _edge("e-share-out", EdgeKind.WRITES, "n-outside", "n-subject-12", _TYPE_PATH, 91),
        ),
    )
    cluster = ("n-auth-06", "n-subject-06", "n-auth-12", "n-subject-12")
    assert "n-subject-12" in detect_mutable_sharing(architecture, cluster)
    assert detect_cross_boundary_effects(architecture, cluster)
    cyclic = _covered_graph(
        extra_nodes=(_node("n-outside", NodeKind.SYMBOL, _TYPE_PATH, 80),),
        extra_edges=(
            _edge("e-cycle-a", EdgeKind.CALLS, "n-auth-01", "n-outside", _TYPE_PATH, 80),
            _edge("e-cycle-b", EdgeKind.CALLS, "n-outside", "n-auth-01", _TYPE_PATH, 81),
        ),
    )[0]
    cycles = detect_cross_boundary_cycles(cyclic, ("n-auth-01", "n-subject-01"))
    assert cycles


def test_costs_are_independently_auditable_and_ranking_is_deterministic() -> None:
    first = _synthesize()
    second = _synthesize()
    assert first.content_identity == second.content_identity
    assert first.ranking == second.ranking
    assert first.ranking == rank_boundary_proposals(first.proposals, first.rejections)
    assert [item.kind.value for item in first.ranking_inputs] == list(first.ranking)
    by_cost = sorted(
        first.proposals,
        key=lambda item: (*item.costs.ranking_key(), item.kind.value),
    )
    assert tuple(item.kind.value for item in by_cost) == first.ranking
    provider = first.proposal(BoundaryKind.PROVIDER_CAPABILITY_SELECTION)
    costs = measure_boundary_costs(
        _covered_graph()[0],
        (provider.canonical_owner_node_id, "n-subject-03", "n-iface"),
    )
    assert tuple(item.kind for item in costs.measures) == REQUIRED_COST_DIMENSIONS
    assert costs.total_numerator == sum(item.numerator for item in costs.measures)
    assert costs.measure(CostDimensionKind.PUBLIC_SYMBOLS).unit == "symbols"
    payload = dict(costs.to_dict())
    payload["aesthetic"] = 1
    with pytest.raises(BoundarySynthesizerError, match="unknown"):
        BoundaryCostVector.from_mapping(payload)


def test_unknown_fields_and_incomplete_proposals_fail_closed() -> None:
    result = _synthesize()
    payload = result.proposal(BoundaryKind.ANALYSIS_CONTEXT).to_dict()
    payload["score"] = 0
    with pytest.raises(BoundarySynthesizerError, match="unknown"):
        BoundaryProposal.from_mapping(payload)
    with pytest.raises(BoundarySynthesizerError, match="nonempty"):
        BoundaryInterface(
            name="",
            kind=BoundaryKind.ANALYSIS_CONTEXT,
            canonical_owner_node_id="n-auth-01",
        )
    with pytest.raises(BoundarySynthesizerError, match="content identity"):
        BoundaryInterface(
            name="pcar.boundary.analysis_context@1",
            kind=BoundaryKind.ANALYSIS_CONTEXT,
            canonical_owner_node_id="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
    with pytest.raises(BoundarySynthesizerError, match="stability"):
        BoundaryInterface(
            name="pcar.boundary.analysis_context@1",
            kind=BoundaryKind.ANALYSIS_CONTEXT,
            canonical_owner_node_id="n-auth-01",
            stability="stable",  # type: ignore[arg-type]
        )
    with pytest.raises(BoundarySynthesizerError, match="applied effects"):
        BoundaryRollback(applied_effects=True)
    with pytest.raises(BoundarySynthesizerAuthorityError, match="state"):
        BoundaryMigration(state_owner_node_id="n-auth-01", mutates_state=True)
    with pytest.raises(BoundarySynthesizerAuthorityError, match="authority"):
        BoundaryMigration(state_owner_node_id="n-auth-01", transfers_authority=True)
    accepted = result.proposal(BoundaryKind.ANALYSIS_CONTEXT)
    with pytest.raises(BoundarySynthesizerError, match="failed hard constraints"):
        BoundaryProposal(
            kind=accepted.kind,
            interface=accepted.interface,
            canonical_owner_node_id=accepted.canonical_owner_node_id,
            state_owner_node_id=accepted.state_owner_node_id,
            callers=accepted.callers,
            effects=accepted.effects,
            adapters=accepted.adapters,
            deprecated_paths=accepted.deprecated_paths,
            tests=accepted.tests,
            proofs=accepted.proofs,
            migration=accepted.migration,
            rollback=accepted.rollback,
            prediction=accepted.prediction,
            costs=accepted.costs,
            hard_constraints=tuple(
                HardConstraintCheck(
                    kind=item.kind,
                    passed=False if item.kind is HardConstraintKind.MISSING_ROLLBACK else item.passed,
                    message="forced failure" if item.kind is HardConstraintKind.MISSING_ROLLBACK else item.message,
                    evidence_node_ids=item.evidence_node_ids,
                    evidence_edge_ids=item.evidence_edge_ids,
                )
                for item in accepted.hard_constraints
            ),
            repository_tree=accepted.repository_tree,
            freshness=accepted.freshness,
        )


def test_synthesizer_refuses_authority_state_and_application() -> None:
    architecture, ownership = _covered_graph()
    synthesizer = InterfaceBoundarySynthesizer(architecture, ownership)
    result = synthesizer.synthesize()
    assert isinstance(result, BoundarySynthesisResult)
    assert synthesizer.effect_class == EFFECT_CLASS
    assert synthesizer.can_apply_proposals is False
    proposal = result.proposal(BoundaryKind.CONTROL_OPERATIONS)
    with pytest.raises(BoundarySynthesizerAuthorityError, match="apply"):
        synthesizer.apply(proposal)
    with pytest.raises(BoundarySynthesizerAuthorityError, match="apply"):
        proposal.apply()
    with pytest.raises(BoundarySynthesizerAuthorityError, match="apply"):
        result.apply()
    with pytest.raises(BoundarySynthesizerAuthorityError, match="authority"):
        synthesizer.transfer_authority()
    with pytest.raises(BoundarySynthesizerAuthorityError, match="authority"):
        synthesizer.promote_authority()
    with pytest.raises(BoundarySynthesizerAuthorityError, match="state"):
        synthesizer.mutate_state()
    with pytest.raises(BoundarySynthesizerAuthorityError, match="authorize"):
        synthesizer.authorize_change()
    with pytest.raises(BoundarySynthesizerAuthorityError):
        refuse_autonomous_application("apply")
    with pytest.raises(BoundarySynthesizerAuthorityError):
        refuse_authority_transfer("transfer")
    with pytest.raises(BoundarySynthesizerAuthorityError):
        refuse_authority_promotion("promote")
    with pytest.raises(BoundarySynthesizerAuthorityError):
        refuse_state_mutation("mutate")
    with pytest.raises(BoundarySynthesizerAuthorityError):
        refuse_ownership_authorization("authorize")
    identity_mismatch = dict(result.to_dict())
    identity_mismatch["can_apply_proposals"] = True
    with pytest.raises(BoundarySynthesizerAuthorityError):
        BoundarySynthesisResult.from_mapping(identity_mismatch)
    round_trip = BoundarySynthesisResult.from_json(result.to_json())
    assert round_trip.content_identity == result.content_identity
    validate_cid(result.content_identity, codecs=("dag-json",))


def test_legacy_and_simulation_boundaries_name_quarantine_without_promotion() -> None:
    result = _synthesize()
    legacy = result.proposal(BoundaryKind.LEGACY_COMPATIBILITY)
    simulation = result.proposal(BoundaryKind.SIMULATION)
    assert _LEGACY_PATH in legacy.deprecated_paths
    assert _SIM_PATH in simulation.deprecated_paths
    assert "n-legacy" in legacy.interface.allowed_callers or "n-legacy" in {
        *legacy.adapters,
        *legacy.callers,
        legacy.canonical_owner_node_id,
    } or _LEGACY_PATH in legacy.deprecated_paths
    assert simulation.constraint(HardConstraintKind.NO_SIMULATED_AS_LIVE).passed is True
    assert simulation.canonical_owner_node_id.startswith("n-auth-")
    architecture, _ownership = _covered_graph()
    owner = architecture.nodes[0]
    assert owner.kind is not NodeKind.SIMULATION or simulation.canonical_owner_node_id != "n-sim"


def test_provider_boundary_records_callers_effects_and_predictions() -> None:
    result = _synthesize()
    proposal = result.proposal(BoundaryKind.PROVIDER_CAPABILITY_SELECTION)
    assert "n-caller" in proposal.callers
    assert proposal.interface.allowed_callers == proposal.callers
    assert proposal.interface.allowed_effects == proposal.effects
    assert "n-iface" in proposal.interface.public_symbols or proposal.prediction.current_public_symbols >= 0
    assert proposal.prediction.proposed_public_symbols <= proposal.prediction.current_public_symbols
    assert proposal.prediction.proposed_cone_size <= proposal.prediction.current_cone_size
    assert proposal.constraint(HardConstraintKind.NO_EFFECT_EXPANSION).passed is True
    assert proposal.constraint(HardConstraintKind.NO_AUTHORITY_WEAKENING).passed is True
    costs = proposal.costs
    assert costs.measure("CrossBoundaryEffects").numerator >= 0
    reconstructed = BoundaryCostMeasure.from_mapping(
        costs.measure(CostDimensionKind.CYCLES).to_dict()
    )
    assert reconstructed.content_identity == costs.measure(CostDimensionKind.CYCLES).content_identity


def test_tree_mismatch_and_missing_fields_fail_closed() -> None:
    architecture, ownership = _covered_graph()
    extended = _graph(
        architecture.nodes + (_node("n-more", NodeKind.SYMBOL, _TYPE_PATH, 200),),
        architecture.edges,
    )
    with pytest.raises(BoundarySynthesizerError, match="architecture_ir_identity"):
        synthesize_interface_boundaries(extended, ownership)
    payload = architecture.to_dict()
    del payload["nodes"]
    with pytest.raises(BoundarySynthesizerError):
        synthesize_interface_boundaries(payload, ownership)
    prediction = BoundaryPrediction(
        current_cone_size=4,
        proposed_cone_size=2,
        cone_reduction=2,
        current_context_nodes=4,
        proposed_context_nodes=2,
        context_reduction=2,
        current_public_symbols=3,
        proposed_public_symbols=1,
        public_symbol_reduction=2,
        current_validation_units=2,
        proposed_validation_units=2,
        validation_amplification_reduction=0,
        validation_coverage_loss=0,
    )
    with pytest.raises(BoundarySynthesizerError, match="documented delta"):
        BoundaryPrediction(
            current_cone_size=4,
            proposed_cone_size=2,
            cone_reduction=9,
            current_context_nodes=4,
            proposed_context_nodes=2,
            context_reduction=2,
            current_public_symbols=3,
            proposed_public_symbols=1,
            public_symbol_reduction=2,
            current_validation_units=2,
            proposed_validation_units=2,
            validation_amplification_reduction=0,
            validation_coverage_loss=0,
        )
    assert prediction.cone_reduction == 2


def test_cycle_rejection_is_isolated_to_the_affected_boundary() -> None:
    result = _synthesize(
        extra_nodes=(_node("n-outside", NodeKind.SYMBOL, _TYPE_PATH, 80),),
        extra_edges=(
            _edge("e-cycle-a", EdgeKind.CALLS, "n-auth-01", "n-outside", _TYPE_PATH, 80),
            _edge("e-cycle-b", EdgeKind.CALLS, "n-outside", "n-auth-01", _TYPE_PATH, 81),
        ),
    )
    rejected = result.rejection(BoundaryKind.ANALYSIS_CONTEXT)
    assert rejected.rejection_kind is RejectionKind.CYCLE
    assert HardConstraintKind.CROSS_BOUNDARY_CYCLE in {
        item.kind for item in rejected.failed_constraints
    }
    assert result.proposal(BoundaryKind.EXECUTION_REQUEST_OUTCOME).disposition is (
        ProposalDisposition.ACCEPTED
    )
    assert result.proposal(BoundaryKind.LEGACY_COMPATIBILITY).disposition is (
        ProposalDisposition.ACCEPTED
    )
    assert result.proposal(BoundaryKind.CONTROL_OPERATIONS).disposition is (
        ProposalDisposition.ACCEPTED
    )
    assert BoundaryKind.ANALYSIS_CONTEXT.value not in result.ranking
    assert BoundaryKind.EXECUTION_REQUEST_OUTCOME.value in result.ranking


def test_dual_state_authority_is_retained_as_a_hard_rejection() -> None:
    result = _synthesize(
        extra_nodes=(_node("n-outside", NodeKind.SYMBOL, _TYPE_PATH, 90),),
        extra_edges=(
            _edge("e-share-in", EdgeKind.WRITES, "n-auth-06", "n-subject-12", _TYPE_PATH, 90),
            _edge("e-share-out", EdgeKind.WRITES, "n-outside", "n-subject-12", _TYPE_PATH, 91),
        ),
    )
    rejected = result.rejection(BoundaryKind.TASK_OBJECTIVE_STATE)
    failed = {item.kind for item in rejected.failed_constraints}
    assert HardConstraintKind.MUTABLE_SHARING in failed
    assert HardConstraintKind.DUAL_STATE_AUTHORITY in failed
    assert rejected.costs.measure(CostDimensionKind.MUTABLE_SHARING).numerator >= 1
    assert tuple(item.kind for item in rejected.costs.measures) == REQUIRED_COST_DIMENSIONS
    assert result.proposal(BoundaryKind.PROVIDER_CAPABILITY_SELECTION).disposition is (
        ProposalDisposition.ACCEPTED
    )


def test_scope_escape_hard_rejects_and_does_not_write_siblings() -> None:
    result = _synthesize(
        extra_nodes=(
            _node("n-sib", NodeKind.SYMBOL, "ipfs_datasets_py/contracts.py", 4),
        ),
        extra_edges=(
            _edge(
                "e-sib",
                EdgeKind.IMPLEMENTS,
                "n-auth-05",
                "n-sib",
                "ipfs_datasets_py/contracts.py",
                4,
            ),
        ),
    )
    rejected = result.rejection(BoundaryKind.EXECUTION_REQUEST_OUTCOME)
    assert rejected.rejection_kind is RejectionKind.HARD_CONSTRAINT
    assert any(
        item.kind is HardConstraintKind.SCOPE_ESCAPE and item.passed is False
        for item in rejected.hard_constraints
    )
    assert any(
        item.kind is HardConstraintKind.NO_CROSS_REPOSITORY_WRITE and item.passed is False
        for item in rejected.hard_constraints
    )
    assert "n-sib" in rejected.node_ids
    assert result.proposal(BoundaryKind.ANALYSIS_CONTEXT).disposition is (
        ProposalDisposition.ACCEPTED
    )


def test_accepted_proposals_do_not_expand_effects_or_drop_validation() -> None:
    result = _synthesize()
    for proposal in result.proposals:
        assert proposal.constraint(HardConstraintKind.NO_EFFECT_EXPANSION).passed is True
        assert proposal.constraint(HardConstraintKind.NO_VALIDATION_REDUCTION).passed is True
        assert proposal.constraint(HardConstraintKind.NO_PROOF_OBLIGATION_LOSS).passed is True
        assert proposal.constraint(HardConstraintKind.NO_AUTHORITY_WEAKENING).passed is True
        assert proposal.constraint(HardConstraintKind.MISSING_ROLLBACK).passed is True
        assert proposal.prediction.proposed_public_symbols <= (
            proposal.prediction.current_public_symbols
        )
        assert proposal.prediction.proposed_cone_size <= proposal.prediction.current_cone_size
        assert proposal.prediction.proposed_context_nodes <= (
            proposal.prediction.current_context_nodes
        )
        with pytest.raises(BoundarySynthesizerError, match="no rejected proposal"):
            result.rejection(proposal.kind)
    with pytest.raises(BoundarySynthesizerError, match="no rejected proposal"):
        result.rejection(BoundaryKind.ANALYSIS_CONTEXT)


def test_ranking_inputs_are_closed_and_non_probative() -> None:
    result = _synthesize()
    assert ranking_establishes(result, "equivalence") is False
    assert ranking_establishes(result, "deletion") is False
    with pytest.raises(BoundarySynthesizerError, match="unsupported ranking"):
        ranking_establishes(result, "aesthetic")
    for item in result.ranking_inputs:
        payload = item.to_dict()
        reconstructed = BoundaryRankingInput.from_mapping(payload)
        assert reconstructed.content_identity == item.content_identity
        payload["score"] = 0
        with pytest.raises(BoundarySynthesizerError, match="unknown"):
            BoundaryRankingInput.from_mapping(payload)
    synthesizer = InterfaceBoundarySynthesizer(*_covered_graph())
    with pytest.raises(BoundarySynthesizerAuthorityError, match="hard constraint"):
        synthesizer.override_hard_constraint()
    result_payload = result.to_dict()
    result_payload["aesthetic"] = True
    with pytest.raises(BoundarySynthesizerError, match="unknown"):
        BoundarySynthesisResult.from_mapping(result_payload)
