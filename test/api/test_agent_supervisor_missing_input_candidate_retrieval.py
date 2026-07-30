"""Adversarial conformance tests for missing-input candidate nomination (RPR-032)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    GraphNodeRef,
    GraphProvenance,
    MissingInputRequirement,
    PropagationAuthorityRoots,
)
from ipfs_accelerate_py.agent_supervisor.analysis.missing_input_candidate_retrieval import (
    ConstructionRouteCandidate,
    ConstructionRouteKind,
    MissingInputCandidateDisposition,
    MissingInputCandidateRetriever,
    MissingInputQuery,
    MissingInputRetrievalBindingError,
    MissingInputRetrievalBounds,
    MissingInputRetrievalBoundsError,
    MissingInputSignal,
    REJECTION_BODY_OR_SECRET,
    REJECTION_COMPATIBILITY_CLAIM,
    REJECTION_FORBIDDEN_CONFIG_ENV,
    REJECTION_FORGED,
    REJECTION_PARTIAL,
    REJECTION_PLACEMENT_CLAIM,
    REJECTION_POISONED,
    REJECTION_SEMANTIC_AUTHORITY_CLAIM,
    REJECTION_STALE_OR_CROSS_ROOT,
    REJECTION_WRITE_SCOPE_CLAIM,
    ValueProvenanceCandidate,
    candidate_set_identity,
    retrieve_missing_input_candidates,
)


ROOTS = PropagationAuthorityRoots(
    repository_id="repository:fixture",
    base_forest_id="forest:base",
    base_tree_id="tree:base",
    base_overlay_id="overlay:base",
    candidate_forest_id="forest:candidate",
    candidate_tree_id="tree:candidate",
    candidate_overlay_id="overlay:candidate",
    graph_id="graph:fixture",
    index_id="index:fixture",
    model_id="model:fixture",
    config_id="config:fixture",
    translator_id="translator:fixture",
    toolchain_id="toolchain:fixture",
    policy_id="policy:fixture",
)


def _requirement(**extra: object) -> MissingInputRequirement:
    values = {
        "roots": ROOTS,
        "requirement_id": "missing:context",
        "obligation_id": "obligation:caller",
        "clause_id": "clause:param-add",
        "parameter_name": "context",
        "type_ref": "type:Context",
        "nullability": "non_null",
        "information_content_ref": "info:request-context",
        "construction_precondition_refs": ("pre:available",),
        "capability_refs": ("cap:context.read",),
        "propagation_depth_bound": 8,
    }
    values.update(extra)
    return MissingInputRequirement(**values)


def _node(
    node_id: str = "node:local-ctx",
    path: str = "pkg/caller.py",
    symbol_id: str = "symbol:local_ctx",
) -> GraphNodeRef:
    return GraphNodeRef(
        node_id=node_id,
        kind="variable",
        path=path,
        symbol_id=symbol_id,
        artifact_id="blob:" + path.replace("/", ":"),
        provenance=GraphProvenance.NOMINATED,
    )


def _candidate(
    expression_ref: str = "expr:local_ctx",
    *,
    path: str = "pkg/caller.py",
    **extra: object,
) -> dict[str, object]:
    value: dict[str, object] = {
        "expression_ref": expression_ref,
        "type_ref": "type:Context",
        "source_node": _node(symbol_id=expression_ref, path=path),
        "evidence_refs": ("evidence:fixture",),
        "history_reviewed": True,
    }
    value.update(extra)
    return value


def test_union_is_deterministic_deduplicated_and_non_authoritative() -> None:
    requirement = _requirement()
    retriever = MissingInputCandidateRetriever(ROOTS)
    signals = {
        "vector": (_candidate(score=0.9, semantic_authority=False),),
        "in_scope_symbol": (_candidate(),),
        "caller_parameter": (_candidate(),),
        "graph": (_candidate(),),
    }

    forward = retriever.retrieve(
        requirement,
        consumer_path="pkg/caller.py",
        candidates_by_signal=signals,
    )
    reverse = retriever.retrieve(
        requirement,
        consumer_path="pkg/caller.py",
        candidates_by_signal=dict(reversed(tuple(signals.items()))),
    )

    assert forward.content_id == reverse.content_id
    assert forward.query_id == forward.query.content_id
    assert len(forward.candidates) == 1
    nomination = forward.candidates[0]
    assert nomination.disposition is MissingInputCandidateDisposition.NOMINATED
    assert nomination.route_kind is ConstructionRouteKind.REUSE
    assert set(signal for signal, _ in nomination.signal_evidence) == {
        MissingInputSignal.VECTOR.value,
        MissingInputSignal.IN_SCOPE_SYMBOL.value,
        MissingInputSignal.CALLER_PARAMETER.value,
        MissingInputSignal.GRAPH.value,
    }
    assert nomination.candidate.semantic_authority is False
    assert nomination.candidate.compatibility_claim is False
    assert nomination.candidate.placement_claim is False
    assert nomination.candidate.write_paths == ()
    assert nomination.write_paths == forward.write_paths == ()
    assert forward.semantic_authority is False
    assert forward.admitted_candidate_id == ""
    assert forward.candidate_set_id == candidate_set_identity(forward.value_candidates)
    assert type(forward).from_dict(forward.to_record()).content_id == forward.content_id
    assert MissingInputQuery.from_dict(forward.query.to_record()).content_id == forward.query.content_id


def test_distinguishes_reuse_thread_convert_construct_and_new_behavior_routes() -> None:
    requirement = _requirement()
    receipt = MissingInputCandidateRetriever(ROOTS).retrieve(
        requirement,
        consumer_path="pkg/caller.py",
        candidates_by_signal={
            "in_scope_symbol": (_candidate("expr:reuse_local"),),
            "reaching_definition": (
                _candidate(
                    "expr:upstream_ctx",
                    thread=True,
                    available_locally=False,
                    path="pkg/mid.py",
                ),
            ),
            "caller_parameter": (
                _candidate(
                    "expr:convert_raw",
                    convert=True,
                    conversion_ref="convert:raw_to_context",
                ),
            ),
            "factory_builder_constructor": (
                _candidate(
                    "expr:ContextFactory.create",
                    factory=True,
                    factory_ref="factory:ContextFactory.create",
                    path="pkg/factories.py",
                ),
            ),
            "schema": (
                _candidate(
                    "expr:schema_default",
                    construct=True,
                    path="pkg/schemas.py",
                ),
            ),
            "authoritative_spec_test": (
                _candidate(
                    "expr:new_support_type",
                    new_behavior=True,
                    path="pkg/support.py",
                ),
            ),
        },
    )

    by_expr = {item.candidate.expression_ref: item for item in receipt.candidates}
    assert by_expr["expr:reuse_local"].route_kind is ConstructionRouteKind.REUSE
    assert by_expr["expr:upstream_ctx"].route_kind is ConstructionRouteKind.THREAD
    assert by_expr["expr:convert_raw"].route_kind is ConstructionRouteKind.CONVERT
    assert by_expr["expr:ContextFactory.create"].route_kind is ConstructionRouteKind.CONSTRUCT
    assert by_expr["expr:schema_default"].route_kind is ConstructionRouteKind.CONSTRUCT
    assert by_expr["expr:new_support_type"].route_kind is ConstructionRouteKind.NEW_BEHAVIOR
    assert all(
        item.disposition is MissingInputCandidateDisposition.NOMINATED
        for item in receipt.candidates
    )
    assert all(item.candidate.semantic_authority is False for item in receipt.candidates)


def test_unions_all_declared_signal_families() -> None:
    requirement = _requirement()
    families = {
        "in_scope_symbol": _candidate("expr:scope"),
        "receiver_state": _candidate("expr:self.ctx", path="pkg/service.py"),
        "caller_parameter": _candidate("expr:param_request"),
        "constant_default": _candidate("expr:DEFAULT_CTX"),
        "request_session_context": _candidate("expr:request.session"),
        "reaching_definition": _candidate("expr:rd_hint", thread_hint=True, available_locally=False),
        "config_env_provider": _candidate("expr:settings.context", path="pkg/config.py"),
        "di_registry_provider": _candidate("expr:container.context", path="pkg/di.py"),
        "factory_builder_constructor": _candidate(
            "expr:build_context", factory=True, path="pkg/factory.py"
        ),
        "schema": _candidate("expr:ContextSchema", construct=True, path="pkg/schema.py"),
        "lineage": _candidate("expr:historical_ctx", path="pkg/history.py"),
        "authoritative_spec_test": _candidate("expr:spec_fixture", path="test/test_ctx.py"),
        "lexical_bm25": _candidate("expr:lexical_hit"),
        "graph": _candidate("expr:graph_node"),
        "vector": _candidate("expr:vector_hit", score=0.42, semantic_authority=False),
    }
    receipt = MissingInputCandidateRetriever(ROOTS).retrieve(
        requirement,
        consumer_path="pkg/caller.py",
        candidates_by_signal={name: (payload,) for name, payload in families.items()},
    )
    assert len(receipt.candidates) == len(families)
    signal_root_names = {signal for signal, _ in receipt.signal_roots}
    assert signal_root_names == set(MissingInputSignal.__members__[m].value for m in MissingInputSignal.__members__)
    # Alias coverage for BM25 / DI / config.
    aliased = MissingInputCandidateRetriever(ROOTS).retrieve(
        requirement,
        consumer_path="pkg/caller.py",
        candidates_by_signal={
            "bm25": (_candidate("expr:bm25"),),
            "di": (_candidate("expr:di"),),
            "env": (_candidate("expr:env"),),
            "history": (_candidate("expr:hist"),),
        },
    )
    signals = {
        signal
        for nomination in aliased.candidates
        for signal, _ in nomination.signal_evidence
    }
    assert MissingInputSignal.LEXICAL_BM25.value in signals
    assert MissingInputSignal.DI_REGISTRY_PROVIDER.value in signals
    assert MissingInputSignal.CONFIG_ENV_PROVIDER.value in signals
    assert MissingInputSignal.LINEAGE.value in signals


def test_adversarial_targets_are_retained_with_stable_diagnostics() -> None:
    requirement = _requirement()
    receipt = MissingInputCandidateRetriever(ROOTS).retrieve(
        requirement,
        consumer_path="pkg/caller.py",
        candidates_by_signal={
            "in_scope_symbol": (
                _candidate("expr:stale", tree_id="tree:other"),
                _candidate("expr:forged", forged_history=True),
                _candidate("expr:forbidden_env", forbidden_config=True),
                {"partial": True, "evidence_refs": ("evidence:partial",)},
                _candidate(
                    "expr:compat_claim",
                    compatible=True,
                ),
                _candidate(
                    "expr:write_claim",
                    write_paths=("pkg/caller.py",),
                ),
                _candidate(
                    "expr:placement_claim",
                    placement="pkg/new_site.py",
                ),
                _candidate(
                    "expr:authority_claim",
                    semantic_authority=True,
                ),
                _candidate(
                    "expr:secret_body",
                    source_body="def leak():\n    return api_key\n",
                    api_key="super-secret-value",
                ),
            ),
            "vector": (
                _candidate(
                    "expr:poison",
                    semantic_authority=True,
                    score=float("nan"),
                ),
            ),
        },
    )

    by_expr = {item.candidate.expression_ref: item for item in receipt.candidates}
    assert REJECTION_STALE_OR_CROSS_ROOT in by_expr["expr:stale"].diagnostics
    assert REJECTION_FORGED in by_expr["expr:forged"].diagnostics
    assert REJECTION_FORBIDDEN_CONFIG_ENV in by_expr["expr:forbidden_env"].diagnostics
    assert any(REJECTION_PARTIAL in item.diagnostics for item in receipt.candidates)
    assert REJECTION_COMPATIBILITY_CLAIM in by_expr["expr:compat_claim"].diagnostics
    assert REJECTION_WRITE_SCOPE_CLAIM in by_expr["expr:write_claim"].diagnostics
    assert REJECTION_PLACEMENT_CLAIM in by_expr["expr:placement_claim"].diagnostics
    assert REJECTION_SEMANTIC_AUTHORITY_CLAIM in by_expr["expr:authority_claim"].diagnostics
    assert REJECTION_BODY_OR_SECRET in by_expr["expr:secret_body"].diagnostics
    assert REJECTION_POISONED in by_expr["expr:poison"].diagnostics
    assert all(
        item.disposition is MissingInputCandidateDisposition.REJECTED
        for item in receipt.candidates
    )
    assert all(item.candidate.semantic_authority is False for item in receipt.candidates)
    assert all(item.write_paths == () for item in receipt.candidates)
    # Bodies/secrets are redacted out of the canonical receipt payload.
    serialized = receipt.to_record()
    blob = str(serialized)
    assert "super-secret-value" not in blob
    assert "def leak" not in blob


def test_bounds_refuse_over_budget_per_signal_and_union() -> None:
    requirement = _requirement()
    retriever = MissingInputCandidateRetriever(
        ROOTS,
        bounds=MissingInputRetrievalBounds(max_candidates=2, max_candidates_per_signal=1),
    )
    with pytest.raises(MissingInputRetrievalBoundsError):
        retriever.retrieve(
            requirement,
            consumer_path="pkg/caller.py",
            candidates_by_signal={
                "in_scope_symbol": (
                    _candidate("expr:a"),
                    _candidate("expr:b"),
                ),
            },
        )
    with pytest.raises(MissingInputRetrievalBoundsError):
        MissingInputCandidateRetriever(
            ROOTS,
            bounds=MissingInputRetrievalBounds(max_candidates=1, max_candidates_per_signal=8),
        ).retrieve(
            requirement,
            consumer_path="pkg/caller.py",
            candidates_by_signal={
                "in_scope_symbol": (_candidate("expr:a"),),
                "graph": (_candidate("expr:b"),),
            },
        )


def test_cross_root_requirement_and_forged_query_bindings_fail_closed() -> None:
    other_roots = PropagationAuthorityRoots(
        repository_id="repository:other",
        base_forest_id="forest:other-base",
        base_tree_id="tree:other-base",
        base_overlay_id="overlay:other-base",
        candidate_forest_id="forest:other-candidate",
        candidate_tree_id="tree:other-candidate",
        candidate_overlay_id="overlay:other-candidate",
        graph_id="graph:other",
        index_id="index:other",
        model_id="model:other",
        config_id="config:other",
        translator_id="translator:other",
        toolchain_id="toolchain:other",
        policy_id="policy:other",
    )
    requirement = _requirement()
    foreign = _requirement(roots=other_roots, requirement_id="missing:foreign")
    retriever = MissingInputCandidateRetriever(ROOTS)
    with pytest.raises(MissingInputRetrievalBindingError):
        retriever.retrieve(foreign, consumer_path="pkg/caller.py")
    with pytest.raises(MissingInputRetrievalBindingError):
        retriever.retrieve(
            requirement,
            query=MissingInputQuery.from_requirement(
                foreign, consumer_path="pkg/caller.py"
            ),
            consumer_path="pkg/caller.py",
        )
    with pytest.raises(MissingInputRetrievalBindingError):
        retriever.retrieve(
            requirement,
            consumer_path="pkg/caller.py",
            graph_id="graph:forged",
        )


def test_empty_signal_set_emits_explicit_partial_diagnostic() -> None:
    requirement = _requirement()
    receipt = retrieve_missing_input_candidates(
        ROOTS,
        requirement,
        consumer_path="pkg/caller.py",
        candidates_by_signal={},
    )
    assert len(receipt.candidates) == 1
    nomination = receipt.candidates[0]
    assert nomination.disposition is MissingInputCandidateDisposition.REJECTED
    assert nomination.diagnostics == (REJECTION_PARTIAL,)
    assert nomination.candidate.semantic_authority is False
    assert receipt.admitted_candidate_id == ""
    assert receipt.write_paths == ()


def test_value_provenance_and_construction_route_records_round_trip() -> None:
    route = ConstructionRouteCandidate(
        route=ConstructionRouteKind.THREAD,
        expression_ref="expr:upstream",
        source_node_id="node:upstream",
        dependency_refs=("dep:caller", "dep:mid"),
    )
    candidate = ValueProvenanceCandidate(
        roots=ROOTS,
        requirement_id="missing:context",
        expression_ref="expr:upstream",
        type_ref="type:Context",
        route=route,
        source_node=_node("node:upstream", path="pkg/mid.py", symbol_id="symbol:upstream"),
        information_content_ref="info:request-context",
    )
    assert candidate.route_kind is ConstructionRouteKind.THREAD
    assert candidate.write_scope == ()
    assert ValueProvenanceCandidate.from_dict(candidate.to_record()) == candidate
    assert ConstructionRouteCandidate.from_dict(route.to_record()) == route

    with pytest.raises(MissingInputRetrievalBindingError):
        ValueProvenanceCandidate(
            roots=ROOTS,
            requirement_id="missing:context",
            expression_ref="expr:bad",
            type_ref="type:Context",
            route=route,
            semantic_authority=True,
        )
    with pytest.raises(MissingInputRetrievalBindingError):
        ValueProvenanceCandidate(
            roots=ROOTS,
            requirement_id="missing:context",
            expression_ref="expr:bad",
            type_ref="type:Context",
            route=route,
            write_paths=("pkg/x.py",),
        )
    with pytest.raises(MissingInputRetrievalBindingError):
        ConstructionRouteCandidate(
            route=ConstructionRouteKind.CONSTRUCT,
            expression_ref="expr:factory",
            semantic_authority=True,
        )


def test_query_requires_consumer_context_and_binds_requirement() -> None:
    requirement = _requirement()
    query = MissingInputQuery.from_requirement(
        requirement,
        consumer_path="pkg/caller.py",
        consumer_node_id="node:caller",
        consumer_context_refs=("consumer:pkg/caller.py",),
    )
    assert query.requirement_id == requirement.requirement_id
    assert query.semantic_authority is False
    with pytest.raises(MissingInputRetrievalBindingError):
        MissingInputQuery(
            roots=ROOTS,
            requirement_id="missing:context",
            obligation_id="obligation:caller",
            clause_id="clause:param-add",
            parameter_name="context",
            type_ref="type:Context",
            information_content_ref="info:request-context",
        )
    with pytest.raises(MissingInputRetrievalBindingError):
        MissingInputQuery.from_requirement(
            requirement,
            consumer_path="pkg/caller.py",
        ).__class__(
            roots=ROOTS,
            requirement_id=requirement.requirement_id,
            obligation_id=requirement.obligation_id,
            clause_id=requirement.clause_id,
            parameter_name=requirement.parameter_name,
            type_ref=requirement.type_ref,
            information_content_ref=requirement.information_content_ref,
            consumer_path="pkg/caller.py",
            semantic_authority=True,
        )


def test_per_signal_refs_and_candidate_set_identity_are_complete() -> None:
    requirement = _requirement()
    receipt = MissingInputCandidateRetriever(ROOTS).retrieve(
        requirement,
        consumer_path="pkg/caller.py",
        candidates_by_signal={
            "graph": (
                _candidate(
                    "expr:g",
                    evidence_refs=("graph:edge:1", "graph:node:1"),
                ),
            ),
            "lexical": (
                _candidate(
                    "expr:g",
                    evidence_refs=("lexical:doc:1",),
                ),
            ),
        },
    )
    assert len(receipt.candidates) == 1
    nomination = receipt.candidates[0]
    evidence_signals = dict(nomination.signal_evidence)
    assert "graph" in evidence_signals
    assert MissingInputSignal.LEXICAL_BM25.value in evidence_signals
    assert nomination.candidate.signal_refs
    assert receipt.candidate_set_id == candidate_set_identity(
        (nomination.candidate,)
    )
    # Signal roots bind every family for replay, not only those that hit.
    assert len(receipt.signal_roots) == len(MissingInputSignal)


def test_stateless_entry_point_matches_retriever() -> None:
    requirement = _requirement()
    signals = {
        "receiver_state": (_candidate("expr:self.context"),),
        "constant_default": (_candidate("expr:DEFAULT"),),
    }
    via_class = MissingInputCandidateRetriever(ROOTS).retrieve(
        requirement,
        consumer_path="pkg/caller.py",
        candidates_by_signal=signals,
    )
    via_fn = retrieve_missing_input_candidates(
        ROOTS,
        requirement,
        consumer_path="pkg/caller.py",
        candidates_by_signal=signals,
    )
    assert via_class.content_id == via_fn.content_id
    assert {item.route_kind for item in via_fn.candidates} == {ConstructionRouteKind.REUSE}
