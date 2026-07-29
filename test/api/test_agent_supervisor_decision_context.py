from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContentAddressedContextStore,
    DecisionContextCompiler,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.context.decision_context import (
    ContextCompletenessWitness,
    DecisionContextBindingError,
    DecisionContextCompilation,
    DecisionContextError,
    DecisionContextOverflowError,
    DecisionContextRepresentation,
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


ROOT = "decision-context-root:fixture"


def _artifact(name: str) -> PinnedArtifactRef:
    return PinnedArtifactRef.from_value(
        {"name": name, "version": 1},
        artifact_id=f"artifact:{name}",
        artifact_kind=name,
        artifact_schema=f"example/{name}@1",
        artifact_schema_version="1",
        producer_id="producer:test",
        authority=ReferenceAuthority.VERIFIED,
    )


def _decision_budget() -> DecisionBudget:
    return DecisionBudget(
        max_input_tokens=4_096,
        max_output_tokens=1_024,
        max_serialized_bytes=262_144,
        max_artifact_bytes=1_048_576,
        max_graph_hops=16,
        max_retrieval_results=64,
        max_proof_attempts=8,
        max_latency_ms=30_000,
        max_expansions=32,
        max_items=512,
        max_depth=24,
        max_text_bytes=16_384,
        max_actions=8,
        max_effects=16,
        max_facts=32,
        max_capabilities=16,
    )


def _request() -> DecisionRequest:
    path = "ipfs_accelerate_py/agent_supervisor/decision_context.py"
    target = DecisionTarget(
        target_id="target:decision-context",
        resource_type="repository-file",
        repository_paths=(path,),
    )
    action = ActionEnvelope(
        action_id="action:compile-decision-context",
        action="write_file",
        tool_id="tool:filesystem",
        authority=DecisionAuthority.MUTATION,
        arguments={"path": path},
        targets=(target,),
    )
    effect = EffectEnvelope(
        effect_id="effect:decision-context",
        kind=EffectKind.WRITE,
        authority=DecisionAuthority.MUTATION,
        target_ids=(target.target_id,),
        repository_paths=(path,),
        description="Compile a complete minimal decision context",
        verification={"command": "python -m pytest"},
    )
    coverage = tuple(sorted(WorktreeCoverage, key=lambda item: item.value))
    roots = tuple(
        sorted(
            (
                SemanticRoot(
                    kind=kind,
                    artifact=_artifact(f"root-{kind.value}"),
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
        capability_id="tool:filesystem",
        provider_id="provider:local",
        version="1",
        configuration=_artifact("filesystem"),
    )
    authority = AuthorityEnvelope(
        principal_id="principal:daemon",
        requested_authority=DecisionAuthority.MUTATION,
        capability_ids=(capability.capability_id,),
        lease_id="lease:1",
        fencing_epoch=1,
        idempotency_key="ASI-133/attempt-1",
        authorization=_artifact("authorization"),
    )
    fact = ApplicabilityFact(
        fact_id="fact:effective",
        kind=ApplicabilityFactKind.EFFECTIVE_TIME,
        predicate="policy-effective",
        value={"effective": True},
        source=_artifact("applicability"),
        jurisdiction="US",
        effective_from_ms=1,
        effective_until_ms=20,
    )
    return DecisionRequest(
        decision_kind="execute",
        stage="implementation",
        objective_id="ASI-133",
        objective_revision="sha256:objective",
        acceptance_id="sha256:acceptance",
        repository_id="repository:fixture",
        repository_path="/srv/repository",
        jurisdiction="US",
        effective_at_ms=10,
        environment_id="environment:test",
        model_id="model:test",
        toolchain_id="toolchain:test",
        authority=authority,
        budget=_decision_budget(),
        action=action,
        expected_effects=(effect,),
        semantic_roots=roots,
        applicability_facts=(fact,),
        capabilities=(capability,),
    )


def _node(
    node_id: str,
    kind: SemanticNodeKind,
    *,
    request: DecisionRequest | None = None,
    record: dict[str, object] | None = None,
    provenance: SemanticProvenance = SemanticProvenance.SOURCE,
    authority: SemanticAuthority = SemanticAuthority.AUTHORITATIVE,
) -> SemanticNode:
    value = dict(record or {"id": node_id})
    if request is not None:
        value.update(
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
        record=value,
    )


def _edge(
    source: str,
    target: str,
    kind: SemanticEdgeKind,
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


def _graph(
    request: DecisionRequest, *, large_legal_body: bool = False
) -> SemanticDependencyGraph:
    legal_record: dict[str, object] = {
        "jurisdiction": "US",
        "rule": "applicable",
    }
    if large_legal_body:
        legal_record["canonical_body"] = "legal constraint " * 600
    nodes = (
        _node("decision", SemanticNodeKind.DECISION, request=request),
        _node("intent", SemanticNodeKind.INTENT_ACTION),
        _node(
            "legal",
            SemanticNodeKind.LEGAL_OBLIGATION,
            record=legal_record,
            provenance=SemanticProvenance.LEGAL_IR,
            authority=SemanticAuthority.CONSTRAINT_INPUT,
        ),
        _node(
            "security",
            SemanticNodeKind.SECURITY_POLICY,
            provenance=SemanticProvenance.SECURITY_IR,
            authority=SemanticAuthority.POLICY_INPUT,
        ),
        _node("obligation", SemanticNodeKind.OBLIGATION),
        _node("proof", SemanticNodeKind.PROOF, provenance=SemanticProvenance.PROOF),
        _node(
            "monitor",
            SemanticNodeKind.MONITOR,
            provenance=SemanticProvenance.MONITOR,
        ),
        _node(
            "validation",
            SemanticNodeKind.VALIDATION,
            provenance=SemanticProvenance.VALIDATION,
        ),
    )
    edges = (
        _edge("decision", "intent", SemanticEdgeKind.REQUIRES),
        _edge("decision", "legal", SemanticEdgeKind.CONSTRAINED_BY),
        _edge(
            "decision",
            "security",
            SemanticEdgeKind.CONSTRAINED_BY,
            SemanticProvenance.SECURITY_IR,
        ),
        _edge("decision", "obligation", SemanticEdgeKind.REQUIRES),
        _edge(
            "obligation",
            "proof",
            SemanticEdgeKind.PROVEN_BY,
            SemanticProvenance.PROOF,
        ),
        _edge(
            "obligation",
            "monitor",
            SemanticEdgeKind.MONITORED_BY,
            SemanticProvenance.MONITOR,
        ),
        _edge("decision", "validation", SemanticEdgeKind.REQUIRES),
    )
    return SemanticDependencyGraph(root_id=ROOT, nodes=nodes, edges=edges)


def _context_budget(tokens: int = 20_000) -> ContextBudget:
    return ContextBudget(
        max_input_tokens=tokens,
        reserved_output_tokens=256,
        reserved_tool_tokens=64,
        max_items=512,
        max_item_bytes=16_384,
        max_serialized_bytes=1_048_576,
        max_depth=24,
        max_text_bytes=16_384,
    )


def _tokenizer(text: str) -> int:
    return max(1, len(text.encode("utf-8")) // 8)


def _compile(
    request: DecisionRequest,
    graph: SemanticDependencyGraph,
    **kwargs: object,
) -> DecisionContextCompilation:
    receipt = retrieve_proof_directed(request, graph)
    compiler = DecisionContextCompiler(
        _context_budget(),
        tokenizer=_tokenizer,
    )
    return compiler.compile(request, graph, receipt, **kwargs)


def test_required_core_and_witness_cover_every_mandatory_dependency() -> None:
    request = _request()
    graph = _graph(request)
    result = _compile(request, graph)
    closure = graph.mandatory_closure("decision")

    assert set(result.context.required_core) == {
        "decision",
        "roots",
        "intent_action_contract",
        "legal_constraints",
        "legal_unknowns",
        "security_constraints",
        "security_unknowns",
        "authorization_state",
        "program_scope",
        "effect_scope",
        "assumptions",
        "obligations",
        "proof_state",
        "monitor_state",
        "validation",
        "acceptance",
        "failure_behavior",
    }
    assert result.context.required_core["decision"]["content_id"] == request.content_id
    assert result.witness.mandatory_node_ids == closure.node_ids
    assert result.witness.mandatory_edge_ids == closure.edge_ids
    assert set(result.witness.dependency_paths) == set(closure.node_ids)
    assert all(entry.path == closure.paths[entry.node_id] for entry in result.witness.entries)
    assert not result.required_nodes_participated_in_value_selection
    assert sum(len(item.references) for item in result.contexts) == len(closure.node_ids)


def test_large_required_body_uses_verified_resolvable_expansion_handle() -> None:
    request = _request()
    graph = _graph(request, large_legal_body=True)
    store = ContentAddressedContextStore()
    result = _compile(request, graph, artifact_store=store)
    legal = next(
        reference
        for context in result.contexts
        for reference in context.references
        if reference.node_id == "legal"
    )

    assert legal.representation is DecisionContextRepresentation.EXPANSION
    assert legal.expansion_handle is not None
    assert store.get(legal.expansion_handle.referenced_content_id)
    assert result.witness.entry("legal").reference_content_id == (
        legal.expansion_handle.referenced_content_id
    )

    with pytest.raises(DecisionContextError):
        replace(
            result.witness,
            mandatory_node_ids=tuple(
                node for node in result.witness.mandatory_node_ids if node != "legal"
            ),
        )


def test_provider_remeasurement_splits_or_fails_closed_without_truncation() -> None:
    request = _request()
    graph = _graph(request)
    receipt = retrieve_proof_directed(request, graph)
    compiler = DecisionContextCompiler(
        _context_budget(),
        tokenizer=_tokenizer,
    )

    result = compiler.compile(request, graph, receipt)
    assert result.split
    assert result.expansion_requests
    assert set(result.witness.mandatory_node_ids) == {
        reference.node_id
        for context in result.contexts
        for reference in context.references
    }
    assert all(
        context.provider_input_tokens <= context.effective_input_limit
        for context in result.contexts
    )
    assert compiler.verify(result) is result

    with pytest.raises(DecisionContextOverflowError, match="mandatory closure"):
        compiler.compile(
            request,
            graph,
            receipt,
            overflow_behavior="fail_closed",
        )


def test_receipt_decision_and_graph_bindings_fail_closed() -> None:
    request = _request()
    graph = _graph(request)
    receipt = retrieve_proof_directed(request, graph)
    compiler = DecisionContextCompiler(
        _context_budget(), tokenizer=_tokenizer
    )

    with pytest.raises(DecisionContextBindingError, match="different decision"):
        compiler.compile(
            request,
            graph,
            replace(receipt, decision_request_id="decision:forged"),
        )


def test_contract_round_trip_is_immutable_and_tamper_evident() -> None:
    request = _request()
    result = _compile(request, _graph(request))
    restored = DecisionContextCompilation.from_dict(result.to_record())

    assert restored.content_id == result.content_id
    assert ContextCompletenessWitness.from_dict(
        result.witness.to_record()
    ).content_id == result.witness.content_id

    forged = result.witness.to_record()
    forged["closure_id"] = "mandatory-closure:forged"
    with pytest.raises(DecisionContextBindingError, match="identity"):
        ContextCompletenessWitness.from_dict(forged)


def test_tenfold_irrelevant_growth_changes_only_bounded_index_metadata() -> None:
    request = _request()
    base_graph = _graph(request)
    extra_kinds = (
        SemanticNodeKind.LEGAL_DECLARATION,
        SemanticNodeKind.TOOL,
        SemanticNodeKind.FILE,
        SemanticNodeKind.ANNOTATION,
        SemanticNodeKind.PREMISE,
    )
    extras = tuple(
        _node(
            f"irrelevant-{index}",
            extra_kinds[index % len(extra_kinds)],
            provenance=(
                SemanticProvenance.LEGAL_IR
                if index % len(extra_kinds) == 0
                else SemanticProvenance.SOURCE
            ),
            authority=(
                SemanticAuthority.CONSTRAINT_INPUT
                if index % len(extra_kinds) == 0
                else SemanticAuthority.DESCRIPTIVE_INPUT
            ),
        )
        for index in range(len(base_graph.nodes) * 10)
    )
    grown_graph = SemanticDependencyGraph(
        root_id=ROOT,
        nodes=(*base_graph.nodes, *extras),
        edges=base_graph.edges,
    )

    base = _compile(request, base_graph)
    grown = _compile(request, grown_graph)

    assert grown.context.required_core == base.context.required_core
    assert [
        reference.to_dict()
        for context in grown.contexts
        for reference in context.references
    ] == [
        reference.to_dict()
        for context in base.contexts
        for reference in context.references
    ]
    assert grown.witness.content_id == base.witness.content_id
    assert grown.witness.entries == base.witness.entries
    assert grown.context.index_metadata != base.context.index_metadata
    assert set(grown.context.index_metadata) == set(base.context.index_metadata)
    base_payloads = [context.provider_payload() for context in base.contexts]
    grown_payloads = [context.provider_payload() for context in grown.contexts]
    assert len(grown_payloads) == len(base_payloads)
    for base_payload, grown_payload in zip(base_payloads, grown_payloads):
        base_payload.pop("index_metadata")
        grown_payload.pop("index_metadata")
        assert grown_payload == base_payload
