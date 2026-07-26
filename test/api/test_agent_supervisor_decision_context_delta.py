from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.context_compiler import (
    ContentAddressedContextStore,
    DecisionContextCompiler,
    compile_decision_context_retry,
    expand_decision_context,
    reconstruct_decision_context,
)
from ipfs_accelerate_py.agent_supervisor.decision_context import (
    DecisionContextBindingError,
    DecisionContextChangeKind,
    DecisionContextChangedDependency,
    DecisionContextError,
    DecisionContextExpansionBudget,
    DecisionContextExpansionError,
    DecisionContextExpansionRequest,
    DecisionContextInvalidatedError,
    DecisionContextRepresentation,
    DecisionContextRetryCapsule,
    DecisionContextRetryError,
)
from ipfs_accelerate_py.agent_supervisor.proof_directed_retrieval import (
    retrieve_proof_directed,
)
from ipfs_accelerate_py.agent_supervisor.semantic_dependency_graph import (
    SemanticDependencyGraph,
)
from test.api.test_agent_supervisor_decision_context import (
    _context_budget,
    _graph,
    _request,
    _tokenizer,
)


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _parent():
    request = _request()
    graph = _graph(request, large_legal_body=True)
    store = ContentAddressedContextStore()
    compiler = DecisionContextCompiler(_context_budget(), tokenizer=_tokenizer)
    parent = compiler.compile(
        request,
        graph,
        retrieve_proof_directed(request, graph),
        artifact_store=store,
    )
    context, reference = next(
        (context, reference)
        for context in parent.contexts
        for reference in context.references
        if reference.representation is DecisionContextRepresentation.EXPANSION
    )
    return request, graph, store, compiler, parent, context, reference


def _request_expansion(
    request,
    graph,
    parent,
    context,
    reference,
    *,
    budget: DecisionContextExpansionBudget | None = None,
    question: str | None = None,
    prior_request_ids=(),
):
    return DecisionContextExpansionRequest(
        parent_decision_request_id=request.content_id,
        parent_context_id=context.content_id,
        parent_completeness_witness_id=parent.witness.content_id,
        unresolved_question=question or f"expand mandatory dependency {reference.node_id}",
        expansion_handle=reference.expansion_handle,
        budget=budget
        or DecisionContextExpansionBudget(1, 20_000, 1_048_576, 30_000),
        prior_request_ids=prior_request_ids,
        authority_id=request.authority.content_id,
        semantic_graph_root_id=graph.root_id,
    )


def _diagnostic(name: str = "pytest") -> DecisionContextChangedDependency:
    return DecisionContextChangedDependency(
        kind=DecisionContextChangeKind.DIAGNOSTICS,
        dependency_id=f"diagnostic:{name}",
        previous_content_id=_digest(f"{name}:before"),
        current_content_id=_digest(f"{name}:after"),
        payload={"diagnostic": name},
    )


def _references(compilation):
    return {
        reference.node_id: reference
        for context in compilation.contexts
        for reference in context.references
    }


def test_question_bound_expansion_is_delta_only_and_reconstructs_parent() -> None:
    request, graph, store, compiler, parent, context, reference = _parent()
    expansion = _request_expansion(
        request, graph, parent, context, reference
    )

    result = expand_decision_context(compiler, parent, expansion, store)
    rebuilt = reconstruct_decision_context(parent, result.retry_capsule)

    assert rebuilt == result.reconstructed_compilation == parent
    assert result.retry_capsule.expanded_evidence[0].node_content_id == (
        reference.node_content_id
    )
    assert result.retry_capsule.changed_dependencies[0].kind is (
        DecisionContextChangeKind.EXPANDED_EVIDENCE
    )
    assert {"required_core", "contexts", "witness"}.isdisjoint(
        result.transmitted_payload
    )
    assert result.delta_input_tokens < result.full_replay_input_tokens
    assert DecisionContextExpansionRequest.from_json(expansion.to_json()) == expansion

    with pytest.raises(DecisionContextExpansionError, match="repeated|equivalent"):
        compiler.expand_decision_context(parent, expansion, store)
    with pytest.raises(DecisionContextExpansionError, match="question|closure"):
        compiler.expand_decision_context(
            parent,
            replace(expansion, unresolved_question="browse unrelated corpus"),
            store,
        )


def test_expansion_rejects_unadmitted_and_cross_boundary_handles() -> None:
    request, graph, store, compiler, parent, context, reference = _parent()
    expansion = _request_expansion(
        request, graph, parent, context, reference
    )
    forged_handle = replace(
        reference.expansion_handle,
        reference_id="mandatory:outside-closure",
    )
    with pytest.raises(
        (DecisionContextExpansionError, DecisionContextBindingError),
        match="closure|parent|handle",
    ):
        compiler.expand_decision_context(
            parent,
            replace(expansion, expansion_handle=forged_handle),
            store,
        )

    for field, value in (
        ("current_repository_id", "repository:other"),
        ("current_dirty_worktree_root_id", _digest("dirty:other")),
        ("current_semantic_roots_digest", _digest("roots:other")),
        ("current_authority_id", _digest("authority:other")),
    ):
        with pytest.raises(DecisionContextInvalidatedError):
            compiler.expand_decision_context(
                parent, expansion, store, **{field: value}
            )


@pytest.mark.parametrize(
    ("budget", "elapsed", "pattern"),
    (
        (DecisionContextExpansionBudget(0, 20_000, 1_048_576, 30_000), 0, "count"),
        (DecisionContextExpansionBudget(1, 1, 1_048_576, 30_000), 0, "token"),
        (DecisionContextExpansionBudget(1, 20_000, 1, 30_000), 0, "byte"),
        (DecisionContextExpansionBudget(1, 20_000, 1_048_576, 1), 2, "latency"),
    ),
)
def test_expansion_fails_closed_on_each_budget(
    budget: DecisionContextExpansionBudget,
    elapsed: int,
    pattern: str,
) -> None:
    request, graph, store, compiler, parent, context, reference = _parent()
    expansion = _request_expansion(
        request, graph, parent, context, reference, budget=budget
    )
    with pytest.raises(DecisionContextExpansionError, match=pattern):
        compiler.expand_decision_context(
            parent, expansion, store, elapsed_latency_ms=elapsed
        )


def test_retry_binds_exact_parent_preserves_omissions_and_round_trips() -> None:
    _, _, _, compiler, parent, _, _ = _parent()
    result = compile_decision_context_retry(
        compiler,
        parent,
        changed_dependencies=(_diagnostic(),),
        omission_reasons={"optional:evidence:with:colon": "token_budget"},
    )
    capsule = result.retry_capsule

    assert result.reconstructed_compilation == parent
    assert capsule.omission_reasons == {
        "optional:evidence:with:colon": "token_budget"
    }
    assert capsule.delta_input_tokens < capsule.full_replay_input_tokens
    assert DecisionContextRetryCapsule.from_json(capsule.to_json()) == capsule
    assert reconstruct_decision_context(parent, capsule) == parent
    assert {"required_core", "contexts", "witness"}.isdisjoint(
        result.transmitted_payload
    )

    forged = capsule.to_record()
    forged["parent_context_id"] = "decision-context:forged"
    with pytest.raises(DecisionContextError, match="identity|parent"):
        DecisionContextRetryCapsule.from_dict(forged)
    unknown = capsule.to_dict()
    unknown["corpus_query"] = "browse everything"
    with pytest.raises(DecisionContextError, match="unsupported"):
        DecisionContextRetryCapsule.from_dict(unknown)
    with pytest.raises(DecisionContextInvalidatedError, match="corpus browsing"):
        compiler.compile_retry(
            parent,
            changed_dependencies=(
                replace(_diagnostic(), payload={"corpus_query": "everything"}),
            ),
        )


def test_paired_retries_reduce_tokens_without_coverage_or_safety_loss() -> None:
    for index in range(5):
        _, _, _, compiler, parent, _, _ = _parent()
        result = compiler.compile_retry(
            parent, changed_dependencies=(_diagnostic(str(index)),)
        )
        rebuilt = result.reconstructed_compilation

        assert result.delta_input_tokens * 100 <= (
            result.full_replay_input_tokens * 65
        )
        assert rebuilt.required_core == parent.required_core
        assert rebuilt.witness.mandatory_node_ids == parent.witness.mandatory_node_ids
        assert rebuilt.witness.mandatory_edge_ids == parent.witness.mandatory_edge_ids
        assert rebuilt.witness.dependency_paths == parent.witness.dependency_paths


@pytest.mark.parametrize(
    ("kind", "node_id"),
    (
        (DecisionContextChangeKind.DEPENDENCIES, "legal"),
        (DecisionContextChangeKind.PROOFS, "proof"),
        (DecisionContextChangeKind.POLICIES, "security"),
        (DecisionContextChangeKind.IR_ROOTS, "intent"),
    ),
)
def test_structural_retry_revalidates_declared_change_and_full_closure(
    kind: DecisionContextChangeKind, node_id: str
) -> None:
    request, graph, _, compiler, parent, _, _ = _parent()
    changed_graph = SemanticDependencyGraph(
        root_id=graph.root_id,
        nodes=tuple(
            replace(node, record={**dict(node.record), "revision": 2})
            if node.node_id == node_id
            else node
            for node in graph.nodes
        ),
        edges=graph.edges,
    )
    target = compiler.compile(
        request,
        changed_graph,
        retrieve_proof_directed(request, changed_graph),
        artifact_store=ContentAddressedContextStore(),
    )
    before, after = _references(parent)[node_id], _references(target)[node_id]
    change = DecisionContextChangedDependency(
        kind=kind,
        dependency_id=node_id,
        previous_content_id=before.node_content_id,
        current_content_id=after.node_content_id,
        payload={"diagnostic_receipt_id": _digest(node_id)},
    )
    result = compiler.compile_retry(
        parent,
        changed_dependencies=(change,),
        target_compilation=target,
    )

    assert result.reconstructed_compilation == target
    assert target.stable_core_id == parent.stable_core_id
    assert target.witness.mandatory_node_ids == parent.witness.mandatory_node_ids
    assert reconstruct_decision_context(parent, result.retry_capsule, target) == target
    with pytest.raises(DecisionContextRetryError, match="structural"):
        reconstruct_decision_context(parent, result.retry_capsule)
    with pytest.raises(
        DecisionContextRetryError, match="undeclared|does not match"
    ):
        compiler.compile_retry(
            parent,
            changed_dependencies=(
                replace(change, dependency_id="proof" if node_id != "proof" else "legal"),
            ),
            target_compilation=target,
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"current_repository_id": "repository:other"},
        {"current_dirty_worktree_root_id": "dirty:other"},
        {"current_semantic_roots_digest": _digest("roots:other")},
        {"current_semantic_graph_root_id": "graph:other"},
        {"current_authority_id": "authority:other"},
    ),
)
def test_retry_invalidates_changed_repository_roots_or_authority(kwargs) -> None:
    _, _, _, compiler, parent, _, _ = _parent()
    with pytest.raises(DecisionContextInvalidatedError):
        compiler.compile_retry(
            parent, changed_dependencies=(_diagnostic(),), **kwargs
        )
