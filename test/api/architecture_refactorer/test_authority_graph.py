"""Hermetic PCAR-006 authority-ownership graph tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.architecture_ir import (
    ArchitectureEdge,
    ArchitectureIR,
    ArchitectureNode,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.authority_graph import (
    AUTHORITY_OWNERSHIP_EVIDENCE,
    AUTHORITY_OWNERSHIP_SCHEMA,
    AUTHORITY_OWNERSHIP_VERSION,
    CONTENT_IDENTITY_IS_NOT_AUTHORITY,
    EXTRACTOR_IDENTITY,
    INITIAL_CONCERNS,
    INITIAL_CONCERN_SOURCE_BINDINGS,
    OWNERSHIP_GRAPH_CAN_AUTHORIZE_CHANGES,
    OWNERSHIP_GRAPH_CAN_TRANSFER_AUTHORITY,
    REEXPORT_IS_NOT_AUTHORITY,
    SILENT_ARBITRATION_PROHIBITED,
    TASK_ID,
    ArbitrationRationale,
    AuthorityGraphAuthorityError,
    AuthorityGraphError,
    AuthorityOwnershipGraph,
    CLOSED_ARBITRATION_RATIONALES,
    CLOSED_CONCERNS,
    CLOSED_OWNERSHIP_BLOCKERS,
    CLOSED_OWNER_DISPOSITIONS,
    ConcernClaim,
    ConcernKind,
    ConcernOwner,
    FormalArbitration,
    LoserClassification,
    OwnerDisposition,
    OwnershipBlockerKind,
    build_authority_ownership_graph,
    canonical_owners,
    lookup_owner_by_content_identity,
    refuse_authority_transfer,
    refuse_ownership_authorization,
    resolve_authority_ownership,
    silently_select_canonical,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.contracts import (
    Confidence,
    EdgeKind,
    NodeKind,
    SourceFactIdentity,
    SourceSpan,
)
from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

_TREE = "a698da9e4b54e2929adacb613bc61ba3e72eed58"
_FRESHNESS = "pcar-006-fixture"
_EXTRACTOR = "pcar-006-fixture"
_ROOT = Path(__file__).resolve().parents[3]
_INVENTORY = (
    _ROOT
    / "docs/architecture/architecture_refactorer_inventory"
    / "current_repository_inventory.json"
)
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


def _covered_graph() -> tuple[ArchitectureIR, tuple[ConcernClaim, ...]]:
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
            _node("n-proj", NodeKind.GENERATED, _PROJECTION_PATH, 12),
            _node("n-legacy", NodeKind.COMPATIBILITY, _LEGACY_PATH, 601),
            _node("n-sim", NodeKind.SIMULATION, _SIM_PATH, 273),
        )
    )
    edges.extend(
        (
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
    return _graph(tuple(nodes), tuple(edges)), tuple(claims)


def test_closed_concern_and_disposition_vocabulary() -> None:
    assert AUTHORITY_OWNERSHIP_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/authority-ownership-graph@1"
    )
    assert AUTHORITY_OWNERSHIP_VERSION == 1
    assert AUTHORITY_OWNERSHIP_EVIDENCE == "pcar/authority-ownership-graph@1"
    assert EXTRACTOR_IDENTITY == "pcar-006-authority-ownership-graph"
    assert TASK_ID == "PCAR-006"
    assert OWNERSHIP_GRAPH_CAN_AUTHORIZE_CHANGES is False
    assert OWNERSHIP_GRAPH_CAN_TRANSFER_AUTHORITY is False
    assert CONTENT_IDENTITY_IS_NOT_AUTHORITY is True
    assert REEXPORT_IS_NOT_AUTHORITY is True
    assert SILENT_ARBITRATION_PROHIBITED is True
    assert tuple(item.value for item in INITIAL_CONCERNS) == (
        "content identity",
        "operation identity",
        "provider capability",
        "provider selection",
        "execution result",
        "task identity",
        "objective identity",
        "policy decision",
        "authorization",
        "confirmation",
        "lease and fencing",
        "state persistence",
        "proof verification",
        "test evidence",
        "completion evidence",
        "release qualification",
    )
    assert CLOSED_CONCERNS == {item.value for item in ConcernKind}
    assert CLOSED_OWNER_DISPOSITIONS == {
        "canonical",
        "adapter",
        "projection",
        "legacy",
        "simulation",
        "unknown",
    }
    assert CLOSED_ARBITRATION_RATIONALES == {
        "explicit_reviewed_contract",
        "supersedes_edge",
        "deprecates_edge",
        "adapts_edge",
    }
    assert CLOSED_OWNERSHIP_BLOCKERS == {item.value for item in OwnershipBlockerKind}
    assert "unknown_owner" in CLOSED_OWNERSHIP_BLOCKERS
    assert "multiple_production_authorities" in CLOSED_OWNERSHIP_BLOCKERS
    with pytest.raises(ValueError):
        ConcernKind("aesthetic score")
    with pytest.raises(ValueError):
        OwnerDisposition("inferred")
    with pytest.raises(ValueError):
        OwnershipBlockerKind("ignore")


def test_current_tree_source_bindings_cover_every_initial_concern() -> None:
    concerns = {item.concern for item in INITIAL_CONCERN_SOURCE_BINDINGS}
    assert concerns == set(INITIAL_CONCERNS)
    canonical_concerns = {
        item.concern
        for item in INITIAL_CONCERN_SOURCE_BINDINGS
        if item.recommended_disposition is OwnerDisposition.CANONICAL
    }
    assert canonical_concerns == set(INITIAL_CONCERNS)
    inventory = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    nominated = {
        (item["concern"], item["path"], item["nominated_symbol"])
        for item in inventory["authority_candidates"]
    }
    for binding in INITIAL_CONCERN_SOURCE_BINDINGS:
        path = _ROOT / binding.path
        assert path.is_file(), binding.path
        text = path.read_text(encoding="utf-8")
        assert binding.nominated_symbol in text
        lines = text.splitlines()
        assert 1 <= binding.start_line <= binding.end_line <= len(lines)
        assert (
            binding.concern.value,
            binding.path,
            binding.nominated_symbol,
        ) in nominated


def test_initial_concern_coverage_has_exactly_one_owner_and_explicit_non_owners() -> None:
    architecture, claims = _covered_graph()
    graph = resolve_authority_ownership(architecture, claims)
    assert graph.covers_initial_concerns is True
    assert graph.fails_closed is False
    assert tuple(item.concern for item in graph.concerns) == INITIAL_CONCERNS
    assert {item.concern for item in graph.concerns} == set(INITIAL_CONCERNS)
    owners = canonical_owners(graph)
    assert len(owners) == len(INITIAL_CONCERNS)
    assert len({node_id for _concern, node_id in owners}) == len(INITIAL_CONCERNS)
    content = graph.ownership_for(ConcernKind.CONTENT_IDENTITY)
    assert content.has_canonical_owner is True
    assert content.canonical_owner is not None
    assert content.canonical_owner.node_id == "n-auth-01"
    assert content.canonical_owner.disposition is OwnerDisposition.CANONICAL
    assert content.blocker is None
    assert content.adapters
    assert all(item.disposition is OwnerDisposition.ADAPTER for item in content.adapters)
    operation = graph.ownership_for("operation identity")
    assert operation.projections
    assert operation.projections[0].node_id == "n-proj"
    assert operation.projections[0].disposition is OwnerDisposition.PROJECTION
    authorization = graph.ownership_for(ConcernKind.AUTHORIZATION)
    assert authorization.legacy_owners
    assert authorization.legacy_owners[0].disposition is OwnerDisposition.LEGACY
    selection = graph.ownership_for(ConcernKind.PROVIDER_SELECTION)
    assert selection.simulation_owners
    assert selection.simulation_owners[0].disposition is OwnerDisposition.SIMULATION
    assert graph.canonical_owner(ConcernKind.TASK_IDENTITY).node_id == "n-auth-06"
    assert graph.architecture_ir_identity == architecture.content_identity
    assert graph.can_authorize_changes is False
    assert graph.can_transfer_authority is False
    assert build_authority_ownership_graph(architecture, claims) == graph


def test_empty_graph_unknown_fails_closed_for_every_concern() -> None:
    architecture = _graph((), ())
    graph = resolve_authority_ownership(architecture)
    assert graph.covers_initial_concerns is True
    assert graph.fails_closed is True
    assert len(graph.blockers) == len(INITIAL_CONCERNS)
    for record in graph.concerns:
        assert record.canonical_owner is None
        assert record.blocker is not None
        assert record.blocker.kind is OwnershipBlockerKind.UNKNOWN_OWNER
        with pytest.raises(AuthorityGraphError, match="no canonical owner"):
            graph.canonical_owner(record.concern)


def test_unknown_claim_fails_closed_even_with_a_canonical_candidate() -> None:
    architecture, claims = _covered_graph()
    unknown = ConcernClaim(
        concern=ConcernKind.TASK_IDENTITY,
        owner_node_id="n-auth-06",
        disposition=OwnerDisposition.UNKNOWN,
    )
    graph = resolve_authority_ownership(architecture, (*claims, unknown))
    record = graph.ownership_for(ConcernKind.TASK_IDENTITY)
    assert record.canonical_owner is None
    assert record.blocker is not None
    assert record.blocker.kind is OwnershipBlockerKind.CONFLICTING_DISPOSITION


def test_unknown_production_path_fails_closed_beside_a_canonical_claim() -> None:
    architecture, claims = _covered_graph()
    extra_node = _node(
        "n-unknown-prod",
        NodeKind.AUTHORITY,
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/registry.py",
        84,
        confidence=Confidence.CONSERVATIVE,
    )
    graph = resolve_authority_ownership(
        ArchitectureIR.from_parts(
            repository_tree=architecture.repository_tree,
            freshness=architecture.freshness,
            nodes=architecture.nodes + (extra_node,),
            edges=architecture.edges,
        ),
        (
            *claims,
            ConcernClaim(
                concern=ConcernKind.TASK_IDENTITY,
                owner_node_id="n-unknown-prod",
                disposition=OwnerDisposition.UNKNOWN,
            ),
        ),
    )
    record = graph.ownership_for(ConcernKind.TASK_IDENTITY)
    assert record.canonical_owner is None
    assert record.blocker is not None
    assert record.blocker.kind is OwnershipBlockerKind.UNKNOWN_PRODUCTION_OWNER
    assert "n-unknown-prod" in record.blocker.node_ids


def test_explicit_unknown_owner_without_canonical_fails_closed() -> None:
    binding = _binding_for(ConcernKind.CONTENT_IDENTITY)[0]
    architecture = _graph(
        (
            _node("n-unknown", NodeKind.AUTHORITY, binding.path, binding.start_line),
        )
    )
    graph = resolve_authority_ownership(
        architecture,
        (
            ConcernClaim(
                concern=ConcernKind.CONTENT_IDENTITY,
                owner_node_id="n-unknown",
                disposition=OwnerDisposition.UNKNOWN,
            ),
        ),
    )
    record = graph.ownership_for(ConcernKind.CONTENT_IDENTITY)
    assert record.blocker is not None
    assert record.blocker.kind is OwnershipBlockerKind.UNKNOWN_OWNER
    assert record.unknown_owners[0].node_id == "n-unknown"


def test_multiple_production_authorities_without_arbitration_block() -> None:
    binding = _binding_for(ConcernKind.CONTENT_IDENTITY)[0]
    extra = [
        item
        for item in INITIAL_CONCERN_SOURCE_BINDINGS
        if item.concern is ConcernKind.CONTENT_IDENTITY
        and item.recommended_disposition is OwnerDisposition.ADAPTER
    ][0]
    architecture = _graph(
        (
            _node("n-a", NodeKind.AUTHORITY, binding.path, binding.start_line),
            _node("n-b", NodeKind.AUTHORITY, extra.path, extra.start_line),
            _node("n-subject", NodeKind.SYMBOL, binding.path, binding.start_line + 1),
        ),
        (
            _edge(
                "e-a",
                EdgeKind.AUTHORIZES,
                "n-a",
                "n-subject",
                binding.path,
                binding.start_line,
            ),
            _edge(
                "e-b",
                EdgeKind.AUTHORIZES,
                "n-b",
                "n-subject",
                extra.path,
                extra.start_line,
            ),
        ),
    )
    claims = (
        ConcernClaim(
            ConcernKind.CONTENT_IDENTITY, "n-a", OwnerDisposition.CANONICAL, ("e-a",)
        ),
        ConcernClaim(
            ConcernKind.CONTENT_IDENTITY, "n-b", OwnerDisposition.CANONICAL, ("e-b",)
        ),
    )
    graph = resolve_authority_ownership(architecture, claims)
    record = graph.ownership_for(ConcernKind.CONTENT_IDENTITY)
    assert record.canonical_owner is None
    assert record.blocker is not None
    assert record.blocker.kind is OwnershipBlockerKind.MULTIPLE_PRODUCTION_AUTHORITIES
    assert set(record.blocker.node_ids) == {"n-a", "n-b"}
    with pytest.raises(AuthorityGraphError, match="silent arbitration"):
        silently_select_canonical(record.blocker.node_ids)


def test_formal_arbitration_selects_one_owner_and_classifies_losers() -> None:
    binding = _binding_for(ConcernKind.CONTENT_IDENTITY)[0]
    extra = [
        item
        for item in INITIAL_CONCERN_SOURCE_BINDINGS
        if item.concern is ConcernKind.CONTENT_IDENTITY
        and item.recommended_disposition is OwnerDisposition.ADAPTER
    ][0]
    architecture = _graph(
        (
            _node("n-a", NodeKind.AUTHORITY, binding.path, binding.start_line),
            _node("n-b", NodeKind.AUTHORITY, extra.path, extra.start_line),
            _node("n-subject", NodeKind.SYMBOL, binding.path, binding.start_line + 1),
        ),
        (
            _edge(
                "e-a",
                EdgeKind.AUTHORIZES,
                "n-a",
                "n-subject",
                binding.path,
                binding.start_line,
            ),
            _edge(
                "e-b",
                EdgeKind.AUTHORIZES,
                "n-b",
                "n-subject",
                extra.path,
                extra.start_line,
            ),
            _edge(
                "e-adapt",
                EdgeKind.ADAPTS,
                "n-b",
                "n-a",
                extra.path,
                extra.start_line,
            ),
        ),
    )
    claims = (
        ConcernClaim(
            ConcernKind.CONTENT_IDENTITY, "n-a", OwnerDisposition.CANONICAL, ("e-a",)
        ),
        ConcernClaim(
            ConcernKind.CONTENT_IDENTITY, "n-b", OwnerDisposition.CANONICAL, ("e-b",)
        ),
    )
    arbitration = FormalArbitration(
        concern=ConcernKind.CONTENT_IDENTITY,
        canonical_owner_node_id="n-a",
        loser_classifications=(
            LoserClassification("n-b", OwnerDisposition.ADAPTER),
        ),
        arbitrator_identity="pcar-006-content-identity-arbitration",
        rationale=ArbitrationRationale.ADAPTS_EDGE,
        provenance=_fact(extra.path, extra.start_line, confidence=Confidence.EXACT),
        evidence_node_ids=("n-a", "n-b"),
        evidence_edge_ids=("e-adapt",),
    )
    graph = resolve_authority_ownership(
        architecture, claims, arbitrations=(arbitration,)
    )
    record = graph.ownership_for(ConcernKind.CONTENT_IDENTITY)
    assert record.blocker is None
    assert record.canonical_owner is not None
    assert record.canonical_owner.node_id == "n-a"
    assert record.arbitration is not None
    assert record.arbitration.canonical_owner_node_id == "n-a"
    assert [item.node_id for item in record.adapters] == ["n-b"]
    assert record.adapters[0].disposition is OwnerDisposition.ADAPTER


def test_silent_and_content_identity_arbitration_are_rejected() -> None:
    binding = _binding_for(ConcernKind.CONTENT_IDENTITY)[0]
    provenance = _fact(binding.path, binding.start_line)
    with pytest.raises(AuthorityGraphError, match="silent arbitration"):
        FormalArbitration(
            concern=ConcernKind.CONTENT_IDENTITY,
            canonical_owner_node_id="n-a",
            loser_classifications=(
                LoserClassification("n-b", OwnerDisposition.ADAPTER),
            ),
            arbitrator_identity="first_listed",
            rationale=ArbitrationRationale.EXPLICIT_REVIEWED_CONTRACT,
            provenance=provenance,
        )
    with pytest.raises(AuthorityGraphError, match="content identity"):
        FormalArbitration(
            concern=ConcernKind.CONTENT_IDENTITY,
            canonical_owner_node_id="baguqeeraaf2trsznudx7wxyyocgpkaoqf5smketqgebile3aobqxbcvdddbq",
            loser_classifications=(),
            arbitrator_identity="reviewed-contract",
            rationale=ArbitrationRationale.EXPLICIT_REVIEWED_CONTRACT,
            provenance=provenance,
        )
    with pytest.raises(AuthorityGraphError, match="cannot prove ownership"):
        FormalArbitration(
            concern=ConcernKind.CONTENT_IDENTITY,
            canonical_owner_node_id="n-a",
            loser_classifications=(
                LoserClassification("n-b", OwnerDisposition.ADAPTER),
            ),
            arbitrator_identity="reviewed-contract",
            rationale=ArbitrationRationale.EXPLICIT_REVIEWED_CONTRACT,
            provenance=_fact(
                binding.path, binding.start_line, confidence=Confidence.HEURISTIC
            ),
        )
    with pytest.raises(AuthorityGraphError, match="loser cannot remain canonical"):
        LoserClassification("n-b", OwnerDisposition.CANONICAL)
    with pytest.raises(AuthorityGraphError, match="content identity"):
        lookup_owner_by_content_identity("baguqeera" + ("a" * 50))


def test_reexport_is_not_canonical_authority() -> None:
    binding = _binding_for(ConcernKind.OPERATION_IDENTITY)[0]
    architecture = _graph(
        (
            _node("n-reexport", NodeKind.AUTHORITY, binding.path, binding.start_line),
            _node("n-real", NodeKind.AUTHORITY, binding.path, binding.start_line + 1),
        ),
        (
            _edge(
                "e-reexport",
                EdgeKind.REEXPORTS,
                "n-reexport",
                "n-real",
                binding.path,
                binding.start_line,
            ),
        ),
    )
    graph = resolve_authority_ownership(
        architecture,
        (
            ConcernClaim(
                ConcernKind.OPERATION_IDENTITY,
                "n-reexport",
                OwnerDisposition.CANONICAL,
                ("e-reexport",),
            ),
        ),
    )
    record = graph.ownership_for(ConcernKind.OPERATION_IDENTITY)
    assert record.canonical_owner is None
    assert record.blocker is not None
    assert record.blocker.kind is OwnershipBlockerKind.REEXPORT_CLAIMED_AUTHORITY


def test_heuristic_opaque_simulation_and_non_authority_canonical_claims_fail_closed() -> None:
    binding = _binding_for(ConcernKind.PROOF_VERIFICATION)[0]
    heuristic_graph = _graph(
        (
            _node(
                "n-h",
                NodeKind.AUTHORITY,
                binding.path,
                binding.start_line,
                confidence=Confidence.HEURISTIC,
            ),
        )
    )
    heuristic = resolve_authority_ownership(
        heuristic_graph,
        (
            ConcernClaim(
                ConcernKind.PROOF_VERIFICATION, "n-h", OwnerDisposition.CANONICAL
            ),
        ),
    )
    heuristic_record = heuristic.ownership_for(ConcernKind.PROOF_VERIFICATION)
    assert heuristic_record.blocker is not None
    assert heuristic_record.blocker.kind is OwnershipBlockerKind.NON_PROBATIVE_OWNERSHIP
    opaque_graph = _graph(
        (
            _node(
                "n-o",
                NodeKind.AUTHORITY,
                binding.path,
                binding.start_line,
                confidence=Confidence.OPAQUE,
            ),
        )
    )
    opaque = resolve_authority_ownership(
        opaque_graph,
        (
            ConcernClaim(
                ConcernKind.PROOF_VERIFICATION, "n-o", OwnerDisposition.CANONICAL
            ),
        ),
    )
    opaque_record = opaque.ownership_for(ConcernKind.PROOF_VERIFICATION)
    assert opaque_record.blocker is not None
    assert opaque_record.blocker.kind is OwnershipBlockerKind.NON_PROBATIVE_OWNERSHIP
    simulation_graph = _graph(
        (_node("n-s", NodeKind.SIMULATION, _SIM_PATH, 273),)
    )
    simulated = resolve_authority_ownership(
        simulation_graph,
        (
            ConcernClaim(
                ConcernKind.PROVIDER_SELECTION, "n-s", OwnerDisposition.CANONICAL
            ),
        ),
    )
    simulated_record = simulated.ownership_for(ConcernKind.PROVIDER_SELECTION)
    assert simulated_record.blocker is not None
    assert simulated_record.blocker.kind is OwnershipBlockerKind.SIMULATED_AS_LIVE
    assert simulated_record.simulation_owners
    assert simulated_record.simulation_owners[0].node_id == "n-s"
    symbol_graph = _graph(
        (_node("n-sym", NodeKind.SYMBOL, binding.path, binding.start_line),)
    )
    symbol = resolve_authority_ownership(
        symbol_graph,
        (
            ConcernClaim(
                ConcernKind.PROOF_VERIFICATION, "n-sym", OwnerDisposition.CANONICAL
            ),
        ),
    )
    symbol_record = symbol.ownership_for(ConcernKind.PROOF_VERIFICATION)
    assert symbol_record.blocker is not None
    assert symbol_record.blocker.kind is OwnershipBlockerKind.NON_AUTHORITY_CANONICAL_CLAIM


def test_claim_by_content_identity_is_rejected() -> None:
    with pytest.raises(AuthorityGraphError, match="content identity"):
        ConcernClaim(
            ConcernKind.CONTENT_IDENTITY,
            "baguqeeraaf2trsznudx7wxyyocgpkaoqf5smketqgebile3aobqxbcvdddbq",
            OwnerDisposition.CANONICAL,
        )
    architecture, claims = _covered_graph()
    owner = resolve_authority_ownership(architecture, claims).canonical_owner(
        ConcernKind.CONTENT_IDENTITY
    )
    with pytest.raises(AuthorityGraphError, match="content identity"):
        ConcernOwner(
            node_id=owner.content_identity,
            disposition=OwnerDisposition.CANONICAL,
            node_kind=NodeKind.AUTHORITY,
            provenance=owner.provenance,
        )


def test_round_trip_and_canonical_identity() -> None:
    architecture, claims = _covered_graph()
    graph = resolve_authority_ownership(architecture, claims)
    payload = graph.to_dict()
    restored = AuthorityOwnershipGraph.from_mapping(payload)
    assert restored == graph
    assert restored.to_dict() == payload
    assert restored.to_json() == json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert AuthorityOwnershipGraph.from_json(restored.to_json()) == graph
    claimed = payload.pop("content_identity")
    validate_cid(claimed, codecs=("dag-json",))
    assert claimed == cid_for_dag_json(payload)
    assert claimed == graph.content_identity
    assert not claimed.startswith("sha256:")
    reversed_graph = resolve_authority_ownership(
        ArchitectureIR.from_parts(
            repository_tree=architecture.repository_tree,
            freshness=architecture.freshness,
            nodes=tuple(reversed(architecture.nodes)),
            edges=tuple(reversed(architecture.edges)),
        ),
        tuple(reversed(claims)),
    )
    assert reversed_graph.content_identity == graph.content_identity
    assert reversed_graph.to_dict() == graph.to_dict()


def test_unknown_fields_and_identity_mismatch_are_rejected() -> None:
    architecture, claims = _covered_graph()
    payload = resolve_authority_ownership(architecture, claims).to_dict()
    unknown = dict(payload)
    unknown["hidden"] = True
    identity_payload = {
        key: value for key, value in unknown.items() if key != "content_identity"
    }
    unknown["content_identity"] = cid_for_dag_json(identity_payload)
    with pytest.raises(AuthorityGraphError, match="unknown authority-ownership field"):
        AuthorityOwnershipGraph.from_mapping(unknown)
    missing = {key: value for key, value in payload.items() if key != "freshness"}
    with pytest.raises(AuthorityGraphError, match="missing authority-ownership field"):
        AuthorityOwnershipGraph.from_mapping(missing)
    forged = dict(payload)
    forged["content_identity"] = "sha256:" + ("00" * 32)
    with pytest.raises(AuthorityGraphError, match="content identity mismatch"):
        AuthorityOwnershipGraph.from_mapping(forged)
    schema = dict(payload)
    schema["schema"] = schema["schema"] + "-extra"
    identity_payload = {
        key: value for key, value in schema.items() if key != "content_identity"
    }
    schema["content_identity"] = cid_for_dag_json(identity_payload)
    with pytest.raises(AuthorityGraphError, match="unexpected authority-ownership schema"):
        AuthorityOwnershipGraph.from_mapping(schema)


def test_missing_concern_and_authorize_or_transfer_are_rejected() -> None:
    architecture, claims = _covered_graph()
    graph = resolve_authority_ownership(architecture, claims)
    payload = graph.to_dict()
    truncated = dict(payload)
    truncated["concerns"] = payload["concerns"][1:]
    identity_payload = {
        key: value for key, value in truncated.items() if key != "content_identity"
    }
    truncated["content_identity"] = cid_for_dag_json(identity_payload)
    with pytest.raises(AuthorityGraphError, match="missing initial concerns"):
        AuthorityOwnershipGraph.from_mapping(truncated)
    with pytest.raises(AuthorityGraphAuthorityError, match="cannot authorize"):
        graph.authorize_change("refactor")
    with pytest.raises(AuthorityGraphAuthorityError, match="cannot transfer"):
        graph.transfer_authority("content identity")
    with pytest.raises(AuthorityGraphAuthorityError, match="cannot authorize"):
        refuse_ownership_authorization("promotion")
    with pytest.raises(AuthorityGraphAuthorityError, match="cannot transfer"):
        refuse_authority_transfer("transfer")
    forged = dict(payload)
    forged["can_authorize_changes"] = True
    identity_payload = {
        key: value for key, value in forged.items() if key != "content_identity"
    }
    forged["content_identity"] = cid_for_dag_json(identity_payload)
    with pytest.raises(AuthorityGraphError, match="cannot authorize changes"):
        AuthorityOwnershipGraph.from_mapping(forged)


def test_unclassified_competitor_and_unknown_node_claims_fail_closed() -> None:
    binding = _binding_for(ConcernKind.LEASE_AND_FENCING)[0]
    architecture = _graph(
        (
            _node("n-a", NodeKind.AUTHORITY, binding.path, binding.start_line),
            _node("n-b", NodeKind.AUTHORITY, binding.path, binding.start_line + 1),
            _node("n-subject", NodeKind.STATE, binding.path, binding.start_line + 2),
        ),
        (
            _edge(
                "e-a",
                EdgeKind.AUTHORIZES,
                "n-a",
                "n-subject",
                binding.path,
                binding.start_line,
            ),
            _edge(
                "e-b",
                EdgeKind.AUTHORIZES,
                "n-b",
                "n-subject",
                binding.path,
                binding.start_line + 1,
            ),
        ),
    )
    claims = (
        ConcernClaim(
            ConcernKind.LEASE_AND_FENCING, "n-a", OwnerDisposition.CANONICAL, ("e-a",)
        ),
        ConcernClaim(
            ConcernKind.LEASE_AND_FENCING, "n-b", OwnerDisposition.CANONICAL, ("e-b",)
        ),
    )
    arbitration = FormalArbitration(
        concern=ConcernKind.LEASE_AND_FENCING,
        canonical_owner_node_id="n-a",
        loser_classifications=(),
        arbitrator_identity="incomplete-arbitration",
        rationale=ArbitrationRationale.EXPLICIT_REVIEWED_CONTRACT,
        provenance=_fact(binding.path, binding.start_line),
        evidence_node_ids=("n-a",),
    )
    graph = resolve_authority_ownership(
        architecture, claims, arbitrations=(arbitration,)
    )
    record = graph.ownership_for(ConcernKind.LEASE_AND_FENCING)
    assert record.blocker is not None
    assert record.blocker.kind is OwnershipBlockerKind.UNCLASSIFIED_COMPETITOR
    with pytest.raises(AuthorityGraphError, match="not in ArchitectureIR"):
        resolve_authority_ownership(
            architecture,
            (
                ConcernClaim(
                    ConcernKind.LEASE_AND_FENCING,
                    "n-missing",
                    OwnerDisposition.CANONICAL,
                ),
            ),
        )


def test_exactly_one_owner_or_blocker_cardinality() -> None:
    architecture, claims = _covered_graph()
    covered = resolve_authority_ownership(architecture, claims)
    empty = resolve_authority_ownership(_graph((), ()))
    for graph in (covered, empty):
        for record in graph.concerns:
            has_owner = record.canonical_owner is not None
            has_blocker = record.blocker is not None
            assert has_owner ^ has_blocker
            if has_owner:
                assert record.canonical_owner.disposition is OwnerDisposition.CANONICAL
