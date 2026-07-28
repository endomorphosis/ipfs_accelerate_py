from __future__ import annotations

from dataclasses import replace
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_retrieval import (
    BoundRetrievalCandidate,
    RetrievalBindingError,
    RetrievalSnapshotBinding,
)
from ipfs_accelerate_py.agent_supervisor.context.decision_contracts import (
    ActionEnvelope,
    ApplicabilityFact,
    ApplicabilityFactKind,
    AuthorityEnvelope,
    CapabilityEnvelope,
    DecisionAuthority,
    DecisionBudget,
    DecisionRequest,
    DecisionTarget,
    EffectEnvelope,
    EffectKind,
    PinnedArtifactRef,
    ReferenceAuthority,
    SemanticRoot,
    SemanticRootKind,
    WorktreeCoverage,
)
from ipfs_accelerate_py.agent_supervisor.proof.proof_directed_retrieval import (
    CandidateDisposition,
    MissingRequiredIndexError,
    ProofDirectedRetrievalBudgetError,
    ProofDirectedRetrievalReceipt,
    ProofRetrievalBudget,
    RetrievalBackendState,
    embedding_fingerprint,
    retrieve_proof_directed,
)
from ipfs_accelerate_py.agent_supervisor.analysis.semantic_dependency_graph import (
    SemanticAuthority,
    SemanticDependencyGraph,
    SemanticEdge,
    SemanticEdgeKind,
    SemanticNode,
    SemanticNodeKind,
    SemanticProvenance,
    SemanticTrust,
)


ROOT = "decision-runtime-root:fixture"
CONFIGURATION = "retrieval-configuration:fixture"
INDEX_ROOTS = {
    "bm25": "bm25-index:fixture",
    "vector": "vector-index:fixture",
}


def _ref(name: str) -> PinnedArtifactRef:
    return PinnedArtifactRef.from_value(
        {"artifact": name, "version": 1},
        artifact_id=f"artifact:{name}",
        artifact_kind=name,
        artifact_schema=f"example/{name}@1",
        artifact_schema_version="1",
        producer_id="producer:test-suite",
        authority=ReferenceAuthority.VERIFIED,
    )


def _budget(**changes: int) -> DecisionBudget:
    values = {
        "max_input_tokens": 4_096,
        "max_output_tokens": 2_048,
        "max_serialized_bytes": 262_144,
        "max_artifact_bytes": 1_048_576,
        "max_graph_hops": 8,
        "max_retrieval_results": 64,
        "max_proof_attempts": 8,
        "max_latency_ms": 30_000,
        "max_expansions": 32,
        "max_items": 512,
        "max_depth": 16,
        "max_text_bytes": 8_192,
        "max_actions": 8,
        "max_effects": 16,
        "max_facts": 32,
        "max_capabilities": 16,
    }
    values.update(changes)
    return DecisionBudget(**values)


def _request(**changes: object) -> DecisionRequest:
    path = "ipfs_accelerate_py/agent_supervisor/proof_directed_retrieval.py"
    target = DecisionTarget(
        target_id="target:retrieval",
        resource_type="repository-file",
        repository_paths=(path,),
    )
    action = ActionEnvelope(
        action_id="action:implement-retrieval",
        action="write_file",
        tool_id="tool:filesystem-edit",
        authority=DecisionAuthority.MUTATION,
        arguments={"path": path},
        targets=(target,),
    )
    effect = EffectEnvelope(
        effect_id="effect:retrieval",
        kind=EffectKind.WRITE,
        authority=DecisionAuthority.MUTATION,
        target_ids=(target.target_id,),
        repository_paths=(path,),
        description="Implement proof-directed retrieval",
        verification={"command": "python -m pytest"},
    )
    coverage = tuple(sorted(WorktreeCoverage, key=lambda item: item.value))
    roots = tuple(
        sorted(
            (
                SemanticRoot(
                    kind=kind,
                    artifact=_ref(f"root-{kind.value}"),
                    coverage=(
                        coverage
                        if kind is SemanticRootKind.DIRTY_WORKTREE
                        else ()
                    ),
                )
                for kind in SemanticRootKind
            ),
            key=lambda item: item.kind.value,
        )
    )
    capability = CapabilityEnvelope(
        capability_id="tool:filesystem-edit",
        provider_id="provider:local",
        version="1",
        configuration=_ref("filesystem-edit"),
    )
    authority = AuthorityEnvelope(
        principal_id="principal:daemon",
        requested_authority=DecisionAuthority.MUTATION,
        capability_ids=(capability.capability_id,),
        lease_id="lease:1",
        fencing_epoch=1,
        idempotency_key="ASI-132/attempt-1",
        authorization=_ref("authorization"),
    )
    fact = ApplicabilityFact(
        fact_id="fact:effective",
        kind=ApplicabilityFactKind.EFFECTIVE_TIME,
        predicate="policy-effective",
        value={"effective": True},
        source=_ref("fact"),
        jurisdiction="US",
        effective_from_ms=1,
        effective_until_ms=10,
    )
    values: dict[str, object] = {
        "decision_kind": "execute",
        "stage": "implementation",
        "objective_id": "ASI-132",
        "objective_revision": "sha256:objective",
        "acceptance_id": "sha256:acceptance",
        "repository_id": "repository:fixture",
        "repository_path": "/srv/repository",
        "jurisdiction": "US",
        "effective_at_ms": 5,
        "environment_id": "environment:test",
        "model_id": "model:test",
        "toolchain_id": "toolchain:test",
        "authority": authority,
        "budget": _budget(),
        "action": action,
        "expected_effects": (effect,),
        "semantic_roots": roots,
        "applicability_facts": (fact,),
        "capabilities": (capability,),
    }
    values.update(changes)
    return DecisionRequest(**values)


def _node(
    node_id: str,
    kind: SemanticNodeKind,
    *,
    request: DecisionRequest | None = None,
    provenance: SemanticProvenance = SemanticProvenance.SOURCE,
    authority: SemanticAuthority = SemanticAuthority.AUTHORITATIVE,
) -> SemanticNode:
    record = {"id": node_id}
    if request is not None:
        record.update(
            {
                "decision_request_id": request.content_id,
                "objective_id": request.objective_id,
                "action_id": request.action.action_id,
            }
        )
    return SemanticNode(
        node_id=node_id,
        kind=kind,
        root_id=ROOT,
        provenance=provenance,
        trust=SemanticTrust.VERIFIED,
        authority=authority,
        version="fixture@1",
        record=record,
    )


def _edge(
    source: str,
    target: str,
    kind: SemanticEdgeKind,
    *,
    provenance: SemanticProvenance = SemanticProvenance.SOURCE,
) -> SemanticEdge:
    return SemanticEdge(
        source=source,
        target=target,
        kind=kind,
        root_id=ROOT,
        provenance=provenance,
        provenance_id=f"edge:{source}:{kind.value}:{target}",
        trust=SemanticTrust.VERIFIED,
        authority=SemanticAuthority.AUTHORITATIVE,
        version="fixture@1",
        mandatory=True,
    )


def _graph(request: DecisionRequest) -> SemanticDependencyGraph:
    nodes = (
        _node("decision", SemanticNodeKind.DECISION, request=request),
        _node("action", SemanticNodeKind.ACTION),
        _node(
            "security-policy",
            SemanticNodeKind.SECURITY_POLICY,
            provenance=SemanticProvenance.SECURITY_IR,
            authority=SemanticAuthority.POLICY_INPUT,
        ),
        _node(
            "denial",
            SemanticNodeKind.SECURITY_RESOURCE,
            provenance=SemanticProvenance.SECURITY_IR,
            authority=SemanticAuthority.POLICY_INPUT,
        ),
        _node("obligation", SemanticNodeKind.OBLIGATION),
        _node(
            "proof",
            SemanticNodeKind.PROOF,
            provenance=SemanticProvenance.PROOF,
        ),
        _node("optional", SemanticNodeKind.PREMISE),
    )
    edges = (
        _edge("decision", "action", SemanticEdgeKind.REQUIRES),
        _edge("action", "security-policy", SemanticEdgeKind.CONSTRAINED_BY),
        _edge(
            "security-policy",
            "denial",
            SemanticEdgeKind.DENIES,
            provenance=SemanticProvenance.SECURITY_IR,
        ),
        _edge("action", "obligation", SemanticEdgeKind.REQUIRES),
        _edge(
            "obligation",
            "proof",
            SemanticEdgeKind.PROVEN_BY,
            provenance=SemanticProvenance.PROOF,
        ),
    )
    return SemanticDependencyGraph(ROOT, nodes, edges)


def _binding(
    request: DecisionRequest,
    graph: SemanticDependencyGraph,
    *,
    partition_id: str | None = None,
) -> RetrievalSnapshotBinding:
    fingerprint = embedding_fingerprint(
        (),
        model_id=request.model_id,
        configuration_id=CONFIGURATION,
    )
    return RetrievalSnapshotBinding(
        graph_root_id=graph.root_id,
        graph_id=graph.graph_id,
        partition_id=partition_id or request.repository_id,
        configuration_id=CONFIGURATION,
        model_id=request.model_id,
        embedding_fingerprint=fingerprint,
        index_roots=INDEX_ROOTS,
    )


def _candidate(
    request: DecisionRequest,
    graph: SemanticDependencyGraph,
    *,
    node_id: str = "optional",
    source: str = "bm25",
    partition_id: str | None = None,
    score: int = 900_000,
) -> BoundRetrievalCandidate:
    binding = _binding(request, graph, partition_id=partition_id)
    return BoundRetrievalCandidate(
        node_id=node_id,
        source=source,
        score_millionths=score,
        binding=binding,
        index_root_id=INDEX_ROOTS[source],
        rank=0,
    )


def test_receipt_is_canonical_and_closure_is_independent_of_candidates() -> None:
    request = _request()
    graph = _graph(request)

    receipt = retrieve_proof_directed(
        request,
        graph,
        candidates=(_candidate(request, graph),),
        index_roots=INDEX_ROOTS,
        configuration_id=CONFIGURATION,
    )
    restored = ProofDirectedRetrievalReceipt.from_json(receipt.to_json())

    assert restored == receipt
    assert json.loads(receipt.to_json()) == receipt.to_dict()
    assert set(receipt.closure_node_ids) == {
        "decision",
        "action",
        "security-policy",
        "denial",
        "obligation",
        "proof",
    }
    assert receipt.optional_node_ids == ("optional",)
    assert set(receipt.closure_node_ids).issubset(receipt.included_node_ids)
    assert receipt.closure_fixed_point and receipt.closure_complete
    assert not receipt.proof_authority and not receipt.completion_authority
    assert any(seed.selector_kind == "decision_request" for seed in receipt.seeds)


def test_stale_cross_partition_and_poisoned_candidates_cannot_hide_denial() -> None:
    request = _request()
    graph = _graph(request)
    foreign = _candidate(
        request,
        graph,
        node_id="denial",
        source="vector",
        partition_id="repository:foreign",
    )
    poisoned = {
        "source": "vector",
        "node_id": "denial",
        "score": float("nan"),
        "binding": _binding(request, graph).to_dict(),
        "index_root_id": INDEX_ROOTS["vector"],
    }

    receipt = retrieve_proof_directed(
        request,
        graph,
        candidates=(foreign, poisoned),
        index_roots=INDEX_ROOTS,
        configuration_id=CONFIGURATION,
    )

    assert "denial" in receipt.closure_node_ids
    assert "denial" in receipt.included_node_ids
    assert "denial" not in receipt.omitted_node_ids
    assert all(
        item.disposition is CandidateDisposition.REJECTED
        for item in receipt.candidates
    )
    assert receipt.truncation["rejected_candidate_count"] == 2

    poisoned_query = retrieve_proof_directed(
        request,
        graph,
        candidate_providers={"vector": ()},
        index_roots=INDEX_ROOTS,
        configuration_id=CONFIGURATION,
        query_embedding=(float("inf"),),
        required_indexes=("vector",),
    )
    assert "denial" in poisoned_query.closure_node_ids
    assert poisoned_query.backend_states["vector"] == (
        RetrievalBackendState.EXACT_FALLBACK.value
    )
    assert "poisoned_or_non_finite_query_embedding" in (
        poisoned_query.disagreement
    )


def test_bound_candidate_rejects_cross_partition_snapshot() -> None:
    request = _request()
    graph = _graph(request)
    candidate = _candidate(
        request,
        graph,
        partition_id="repository:foreign",
    )

    with pytest.raises(RetrievalBindingError, match="snapshot"):
        candidate.validate_against(
            _binding(request, graph),
            node_ids={item.node_id for item in graph.nodes},
        )


def test_missing_required_index_uses_exact_fallback_or_fails_closed() -> None:
    request = _request()
    graph = _graph(request)

    fallback = retrieve_proof_directed(
        request,
        graph,
        index_roots=INDEX_ROOTS,
        configuration_id=CONFIGURATION,
        required_indexes=("vector",),
    )

    assert fallback.backend_states["vector"] == (
        RetrievalBackendState.EXACT_FALLBACK.value
    )
    assert fallback.fallback == ("vector:deterministic_exact_graph_scan",)
    assert "proof" in fallback.closure_node_ids

    with pytest.raises(MissingRequiredIndexError):
        retrieve_proof_directed(
            request,
            graph,
            index_roots=INDEX_ROOTS,
            configuration_id=CONFIGURATION,
            required_indexes=("vector",),
            allow_exact_fallback=False,
        )


def test_graph_budget_exhaustion_fails_closed_without_partial_receipt() -> None:
    request = _request()
    graph = _graph(request)
    budget = replace(
        ProofRetrievalBudget.from_decision(request),
        max_graph_nodes=2,
    )

    with pytest.raises(ProofDirectedRetrievalBudgetError, match="closure"):
        retrieve_proof_directed(
            request,
            graph,
            index_roots=INDEX_ROOTS,
            configuration_id=CONFIGURATION,
            budget=budget,
        )
