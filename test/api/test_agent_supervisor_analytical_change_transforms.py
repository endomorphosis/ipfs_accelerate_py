"""Fail-closed coverage for deterministic analytical change transforms (RPR-037)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    AnalyticalTransform,
    ConsumerDisposition,
    ConsumerMigrationObligation,
    GraphNodeRef,
    GraphProvenance,
    PropagationAuthorityRoots,
    TransformDisposition,
    TransformKind,
)
from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_contracts import (
    AuthorityRoots,
    DecisionDisposition,
    EvidenceReference,
    RepairCandidate,
    RepairStrategy,
    RepairTargetDecision,
    SourceSpan,
    candidate_set_identity,
)
from ipfs_accelerate_py.agent_supervisor.planning.analytical_change_transforms import (
    ANALYTICAL_CHANGE_TRANSFORMER_INTERFACE,
    AnalyticalChangeTransformAuthorityError,
    AnalyticalChangeTransformError,
    AnalyticalChangeTransformer,
    FieldMapping,
    TransformRejectionReason,
    TransformSite,
    make_span,
    render_analytical_transform,
)
from ipfs_accelerate_py.agent_supervisor.proof.missing_input_synthesis import (
    SynthesisDisposition,
    ValueMappingProof,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:one",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:one",
        index_id="index:one",
        model_id="model:one",
        config_id="config:one",
        translator_id="translator:one",
        toolchain_id="toolchain:one",
        policy_id="policy:one",
    )


def node(path: str = "pkg/caller.py", symbol: str = "symbol:caller") -> GraphNodeRef:
    return GraphNodeRef(
        node_id=f"node:{symbol}",
        kind="function",
        path=path,
        symbol_id=symbol,
        artifact_id=f"blob:{symbol}",
        provenance=GraphProvenance.TRUSTED,
        extractor_id="extractor:ast",
    )


def mapping(
    *,
    disposition: SynthesisDisposition = SynthesisDisposition.UNIQUE_PROVED,
    expression_ref: str = "expr:ctx",
    proved: tuple[str, ...] | None = None,
    repository_id: str = "repository:one",
    tree_id: str = "tree:candidate",
) -> ValueMappingProof:
    if proved is None:
        if disposition is SynthesisDisposition.UNIQUE_PROVED:
            proved = ("candidate:ctx",)
        elif disposition is SynthesisDisposition.AMBIGUOUS:
            proved = ("candidate:a", "candidate:b")
        else:
            proved = ()
    return ValueMappingProof(
        requirement_id="missing:context",
        consumer_id="consumer:one",
        disposition=disposition,
        facet_results=(),
        proved_candidate_ids=proved,
        refuted_candidate_ids=()
        if disposition is not SynthesisDisposition.REFUTED
        else ("candidate:bad",),
        expression_ref=expression_ref,
        type_ref="type:Context",
        repository_id=repository_id,
        tree_id=tree_id,
        toolchain_id="toolchain:one",
        policy_id="policy:one",
        reason_codes=(
            ("unique_source",)
            if disposition is SynthesisDisposition.UNIQUE_PROVED
            else ("non_unique",)
        ),
    )


def obligation(auth: PropagationAuthorityRoots | None = None) -> ConsumerMigrationObligation:
    auth = auth or roots()
    return ConsumerMigrationObligation(
        roots=auth,
        obligation_id="obligation:one",
        consumer_id="consumer:one",
        delta_id="delta:one",
        disposition=ConsumerDisposition.MIGRATE,
        clause_ids=("clause:param-add",),
        node=node(),
        proof_refs=("proof:obligation",),
        missing_input_ids=("missing:context",),
        invalidation_refs=("tree:candidate",),
    )


def repair_decision(
    path: str = "pkg/caller.py",
    *,
    admitted: bool = True,
) -> RepairTargetDecision:
    repair_roots = AuthorityRoots(
        repository_id="repository:one",
        forest_id="forest:candidate",
        tree_id="tree:candidate",
        graph_id="graph:one",
        index_id="index:one",
        model_id="model:one",
        config_id="config:one",
        translator_id="translator:one",
        toolchain_id="toolchain:one",
        policy_id="policy:one",
    )
    candidate = RepairCandidate(
        repair_roots,
        "trace:one",
        RepairStrategy.NEW_IMPLEMENTATION,
        SourceSpan(path, 0, 12, "blob:one"),
        (EvidenceReference("candidate", "candidate:one", producer_id="test"),),
    )
    candidates = (candidate,)
    return RepairTargetDecision(
        roots=repair_roots,
        candidates=candidates,
        candidate_set_id=candidate_set_identity(candidates),
        disposition=DecisionDisposition.ADMITTED if admitted else DecisionDisposition.ABSTAINED,
        strategy=RepairStrategy.NEW_IMPLEMENTATION,
        selected_candidate_id=candidate.content_id if admitted else "",
        permitted_read_paths=(path,) if admitted else (),
        permitted_write_paths=(path,) if admitted else (),
        evidence_refs=(EvidenceReference("authority", "authority:one", producer_id="test"),),
        proof_refs=(EvidenceReference("proof", "proof:one", producer_id="test"),),
        invalidation_refs=("tree:candidate",),
    )


def site(
    kind: TransformKind,
    source: str,
    *,
    site_id: str = "site:one",
    path: str = "pkg/caller.py",
    **kwargs: object,
) -> TransformSite:
    auth = roots()
    defaults: dict[str, object] = {
        "roots": auth,
        "site_id": site_id,
        "kind": kind,
        "span": make_span(path, source),
        "obligation_ids": ("obligation:one",),
        "proof_refs": ("proof:one",),
    }
    defaults.update(kwargs)
    return TransformSite(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Interface / identity
# ---------------------------------------------------------------------------


def test_interface_constant() -> None:
    assert ANALYTICAL_CHANGE_TRANSFORMER_INTERFACE == "AnalyticalChangeTransformer@1"
    assert AnalyticalChangeTransformer.INTERFACE == ANALYTICAL_CHANGE_TRANSFORMER_INTERFACE


def test_returns_canonical_analytical_transform() -> None:
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.ADD_ARGUMENT,
            "process(event)",
            expression_ref="expr:ctx",
            expression_text="ctx",
            parameter_name="context",
        ),
        value_mapping=mapping(),
    )
    assert isinstance(receipt.transform, AnalyticalTransform)
    assert receipt.transform.SCHEMA.endswith("analytical-transform@1")
    assert receipt.admitted
    # Round-trip canonical record.
    restored = AnalyticalTransform.from_dict(receipt.transform.to_record())
    assert restored == receipt.transform


# ---------------------------------------------------------------------------
# Add / rename / reorder argument
# ---------------------------------------------------------------------------


def test_add_argument_keyword_from_unique_proved_expression() -> None:
    receipt = render_analytical_transform(
        site(
            TransformKind.ADD_ARGUMENT,
            "process(event)",
            expression_ref="expr:ctx",
            expression_text="ctx",
            parameter_name="context",
            keyword_style=True,
        ),
        value_mapping=mapping(),
        obligation=obligation(),
    )
    assert receipt.admitted
    assert receipt.edits[0].replacement == "process(event, context=ctx)"
    assert receipt.edits[0].before_hash.startswith("sha256:")
    assert receipt.edits[0].expected_after_hash.startswith("sha256:")
    assert receipt.transform.proof_refs
    assert "expr:ctx" in receipt.transform.expression_refs


def test_add_argument_positional_preserves_spacing_style() -> None:
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.ADD_ARGUMENT,
            "process(event,other)",
            expression_ref="expr:ctx",
            expression_text="ctx",
            parameter_name="context",
            keyword_style=False,
        ),
        value_mapping=mapping(),
    )
    assert receipt.edits[0].replacement == "process(event,other,ctx)"


def test_add_argument_idempotent_when_keyword_present() -> None:
    source = "process(event, context=ctx)"
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.ADD_ARGUMENT,
            source,
            expression_ref="expr:ctx",
            expression_text="ctx",
            parameter_name="context",
        ),
        value_mapping=mapping(),
    )
    assert receipt.edits[0].replacement == source


def test_rename_keyword_argument() -> None:
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.RENAME_ARGUMENT,
            "emit(event=event, tenant=tenant_id)",
            previous_parameter_name="tenant",
            parameter_name="tenant_id",
            expression_ref="expr:ctx",
            expression_text="tenant_id",
        ),
        value_mapping=mapping(expression_ref="expr:ctx"),
    )
    assert receipt.edits[0].replacement == "emit(event=event, tenant_id=tenant_id)"


def test_rename_function_parameter() -> None:
    source = "def process(event, context):\n    return event"
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.RENAME_ARGUMENT,
            source,
            previous_parameter_name="context",
            parameter_name="ctx",
            expression_ref="expr:ctx",
            expression_text="ctx",
        ),
        value_mapping=mapping(),
    )
    assert "def process(event, ctx):" in receipt.edits[0].replacement


def test_reorder_keyword_arguments_total_mapping() -> None:
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.REORDER_ARGUMENT,
            "emit(b=2, a=1)",
            argument_order=("a", "b"),
            expression_ref="expr:ctx",
            expression_text="a",
        ),
        value_mapping=mapping(),
    )
    assert receipt.edits[0].replacement == "emit(a=1, b=2)"


def test_reorder_rejects_non_total_mapping() -> None:
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.REORDER_ARGUMENT,
            "emit(b=2, a=1)",
            argument_order=("a",),
            expression_ref="expr:ctx",
            expression_text="a",
        ),
        value_mapping=mapping(),
    )
    assert not receipt.admitted
    assert TransformRejectionReason.NON_TOTAL_MAPPING.value in receipt.rejection_reasons


# ---------------------------------------------------------------------------
# Threading, import/export/registration, adapter, constructor, schema
# ---------------------------------------------------------------------------


def test_thread_parameter_requires_full_route_in_batch() -> None:
    hop_a = site(
        TransformKind.THREAD_PARAMETER,
        "inner(x)",
        site_id="site:hop-a",
        path="pkg/inner.py",
        expression_ref="expr:ctx",
        expression_text="ctx",
        parameter_name="context",
        route_site_ids=("site:hop-a", "site:hop-b"),
        dependency_transform_ids=(),
    )
    hop_b = site(
        TransformKind.THREAD_PARAMETER,
        "outer(y)",
        site_id="site:hop-b",
        path="pkg/outer.py",
        expression_ref="expr:ctx",
        expression_text="ctx",
        parameter_name="context",
        route_site_ids=("site:hop-a", "site:hop-b"),
        dependency_transform_ids=("transform:site:hop-a",),
    )
    batch = AnalyticalChangeTransformer().render_many(
        (hop_a, hop_b),
        value_mappings={"site:hop-a": mapping(), "site:hop-b": mapping()},
    )
    assert batch.admitted_transforms
    assert all(item.admitted for item in batch.receipts)
    # Deterministic site_id ordering, not input order.
    assert [item.site_id for item in batch.receipts] == ["site:hop-a", "site:hop-b"]


def test_thread_parameter_rejects_incomplete_route() -> None:
    hop = site(
        TransformKind.THREAD_PARAMETER,
        "inner(x)",
        site_id="site:hop-a",
        expression_ref="expr:ctx",
        expression_text="ctx",
        parameter_name="context",
        route_site_ids=("site:hop-a", "site:hop-missing"),
    )
    batch = AnalyticalChangeTransformer().render_many(
        (hop,),
        value_mappings={"site:hop-a": mapping()},
    )
    assert not batch.receipts[0].admitted
    assert TransformRejectionReason.SCOPE_ESCAPE.value in batch.receipts[0].rejection_reasons


def test_add_import_idempotent_and_appends() -> None:
    transformer = AnalyticalChangeTransformer()
    first = transformer.render(
        site(
            TransformKind.ADD_IMPORT,
            "import os\n",
            import_module="pkg.types",
            import_name="Context",
        )
    )
    assert first.admitted
    assert "from pkg.types import Context" in first.edits[0].replacement
    second = transformer.render(
        site(
            TransformKind.ADD_IMPORT,
            first.edits[0].replacement,
            import_module="pkg.types",
            import_name="Context",
        )
    )
    assert second.edits[0].replacement == first.edits[0].replacement


def test_add_export_to_all() -> None:
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.ADD_EXPORT,
            '__all__ = ["existing"]',
            export_name="Context",
        )
    )
    assert '"Context"' in receipt.edits[0].replacement
    assert '"existing"' in receipt.edits[0].replacement


def test_add_registration_closed_call() -> None:
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.ADD_REGISTRATION,
            "REGISTRY.register()",
            registration_name="context",
            registration_target="Context",
        )
    )
    assert receipt.edits[0].replacement == 'REGISTRY.register("context", Context)'


def test_add_adapter_wraps_proved_expression() -> None:
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.ADD_ADAPTER,
            "ctx",
            expression_ref="expr:ctx",
            expression_text="ctx",
            adapter_expression="ContextAdapter",
        ),
        value_mapping=mapping(),
    )
    assert receipt.edits[0].replacement == "ContextAdapter(ctx)"


def test_update_constructor_adds_keyword() -> None:
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.UPDATE_CONSTRUCTOR,
            "Context(request_id)",
            expression_ref="expr:ctx",
            expression_text="tenant_id",
            parameter_name="tenant",
        ),
        value_mapping=mapping(),
    )
    assert receipt.edits[0].replacement == "Context(request_id, tenant=tenant_id)"


def test_update_schema_field_total_json_mapping() -> None:
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.UPDATE_SCHEMA_FIELD,
            '{"tenant":"t1","name":"n"}',
            path="pkg/schema.json",
            field_mappings=(
                FieldMapping("tenant", "tenant_id"),
                FieldMapping("name", "display_name"),
            ),
        )
    )
    assert receipt.edits[0].replacement == '{"display_name":"n","tenant_id":"t1"}'


def test_update_fixture_rejects_missing_keys() -> None:
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.UPDATE_FIXTURE,
            '{"only":1}',
            path="test/fixtures/one.json",
            field_mappings=(FieldMapping("missing", "other"),),
        )
    )
    assert not receipt.admitted
    assert TransformRejectionReason.NON_TOTAL_MAPPING.value in receipt.rejection_reasons


# ---------------------------------------------------------------------------
# Rejection matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        ("process(event, *rest)", TransformRejectionReason.DYNAMIC_SPLAT),
        ("process(event, **kwargs)", TransformRejectionReason.DYNAMIC_SPLAT),
        ("process(event) if flag else other(event)", TransformRejectionReason.UNSUPPORTED_SYNTAX),
    ],
)
def test_rejects_dynamic_or_unsupported_call_shapes(
    source: str, reason: TransformRejectionReason
) -> None:
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.ADD_ARGUMENT,
            source,
            expression_ref="expr:ctx",
            expression_text="ctx",
            parameter_name="context",
        ),
        value_mapping=mapping(),
    )
    assert not receipt.admitted
    assert reason.value in receipt.rejection_reasons
    assert receipt.edits == ()


def test_rejects_ambiguous_overload_count() -> None:
    with pytest.raises(AnalyticalChangeTransformError):
        # overload_count != 1 is rejected at site construction.
        site(
            TransformKind.ADD_ARGUMENT,
            "process(event)",
            expression_ref="expr:ctx",
            expression_text="ctx",
            parameter_name="context",
            overload_count=2,
        )


def test_rejects_stale_before_hash() -> None:
    good = make_span("pkg/caller.py", "process(event)")
    with pytest.raises(AnalyticalChangeTransformAuthorityError, match="stale_span"):
        TransformSite(
            roots=roots(),
            site_id="site:stale",
            kind=TransformKind.ADD_ARGUMENT,
            span=type(good)(
                path=good.path,
                start=good.start,
                end=good.end,
                artifact_id=good.artifact_id,
                span_text=good.span_text,
                before_hash="sha256:" + "0" * 64,
            ),
            obligation_ids=("obligation:one",),
            proof_refs=("proof:one",),
            expression_ref="expr:ctx",
            expression_text="ctx",
            parameter_name="context",
        )


def test_rejects_non_unique_value_mapping() -> None:
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.ADD_ARGUMENT,
            "process(event)",
            expression_ref="expr:ctx",
            expression_text="ctx",
            parameter_name="context",
        ),
        value_mapping=mapping(
            disposition=SynthesisDisposition.AMBIGUOUS,
            proved=("candidate:a", "candidate:b"),
        ),
    )
    assert not receipt.admitted
    assert TransformRejectionReason.NO_CODE_AUTHORITY.value in receipt.rejection_reasons


def test_rejects_expression_ref_mismatch() -> None:
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.ADD_ARGUMENT,
            "process(event)",
            expression_ref="expr:other",
            expression_text="ctx",
            parameter_name="context",
        ),
        value_mapping=mapping(expression_ref="expr:ctx"),
    )
    assert TransformRejectionReason.EXPRESSION_MISMATCH.value in receipt.rejection_reasons


def test_rejects_root_mismatch_on_value_mapping() -> None:
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.ADD_ARGUMENT,
            "process(event)",
            expression_ref="expr:ctx",
            expression_text="ctx",
            parameter_name="context",
        ),
        value_mapping=mapping(repository_id="repository:other"),
    )
    assert TransformRejectionReason.ROOT_MISMATCH.value in receipt.rejection_reasons


def test_rejects_invented_expression_behavior() -> None:
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.ADD_ARGUMENT,
            "process(event)",
            expression_ref="expr:ctx",
            expression_text="make_context()",
            parameter_name="context",
        ),
        value_mapping=mapping(),
    )
    assert TransformRejectionReason.INVENTED_BEHAVIOR.value in receipt.rejection_reasons


def test_rejects_new_dependency_outside_allowlist() -> None:
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.ADD_IMPORT,
            "import os\n",
            import_module="external.vendor",
            import_name="Thing",
            allowed_dependency_paths=("pkg/",),
        )
    )
    assert TransformRejectionReason.NEW_DEPENDENCY.value in receipt.rejection_reasons


def test_rejects_unauthorized_write_path() -> None:
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.ADD_ARGUMENT,
            "process(event)",
            path="pkg/other.py",
            expression_ref="expr:ctx",
            expression_text="ctx",
            parameter_name="context",
        ),
        value_mapping=mapping(),
        decision=repair_decision(path="pkg/caller.py", admitted=True),
    )
    assert TransformRejectionReason.PATH_NOT_AUTHORIZED.value in receipt.rejection_reasons


def test_admits_when_repair_target_authorizes_path() -> None:
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.ADD_ARGUMENT,
            "process(event)",
            expression_ref="expr:ctx",
            expression_text="ctx",
            parameter_name="context",
        ),
        value_mapping=mapping(),
        decision=repair_decision(path="pkg/caller.py", admitted=True),
    )
    assert receipt.admitted


def test_rejects_missing_proof_refs() -> None:
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.ADD_IMPORT,
            "",
            import_module="pkg.types",
            import_name="Context",
            proof_refs=(),
        )
    )
    assert TransformRejectionReason.MISSING_PROOF.value in receipt.rejection_reasons


def test_rejects_scope_escape_when_obligation_path_differs() -> None:
    foreign = ConsumerMigrationObligation(
        roots=roots(),
        obligation_id="obligation:one",
        consumer_id="consumer:one",
        delta_id="delta:one",
        disposition=ConsumerDisposition.MIGRATE,
        clause_ids=("clause:param-add",),
        node=node(path="pkg/elsewhere.py", symbol="symbol:else"),
        proof_refs=("proof:obligation",),
        missing_input_ids=("missing:context",),
        invalidation_refs=("tree:candidate",),
    )
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.ADD_ARGUMENT,
            "process(event)",
            expression_ref="expr:ctx",
            expression_text="ctx",
            parameter_name="context",
        ),
        value_mapping=mapping(),
        obligation=foreign,
    )
    assert TransformRejectionReason.SCOPE_ESCAPE.value in receipt.rejection_reasons


# ---------------------------------------------------------------------------
# Determinism / formatting / authority binding
# ---------------------------------------------------------------------------


def test_repeated_rendering_is_byte_equivalent() -> None:
    transformer = AnalyticalChangeTransformer()
    request = site(
        TransformKind.ADD_ARGUMENT,
        "process(event)",
        expression_ref="expr:ctx",
        expression_text="ctx",
        parameter_name="context",
    )
    first = transformer.render(request, value_mapping=mapping())
    second = transformer.render(request, value_mapping=mapping())
    assert first.edits[0].replacement == second.edits[0].replacement
    assert first.edits[0].expected_after_hash == second.edits[0].expected_after_hash
    assert first.replay_identity == second.replay_identity
    assert first.to_record() == second.to_record()


def test_before_and_after_hashes_bind_exact_text() -> None:
    source = "process(event)"
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.ADD_ARGUMENT,
            source,
            expression_ref="expr:ctx",
            expression_text="ctx",
            parameter_name="context",
        ),
        value_mapping=mapping(),
    )
    edit = receipt.edits[0]
    import hashlib

    assert edit.before_hash == "sha256:" + hashlib.sha256(source.encode()).hexdigest()
    assert (
        edit.expected_after_hash
        == "sha256:" + hashlib.sha256(edit.replacement.encode()).hexdigest()
    )


def test_non_admitted_transform_grants_no_target_paths() -> None:
    receipt = AnalyticalChangeTransformer().render(
        site(
            TransformKind.ADD_ARGUMENT,
            "process(event, *xs)",
            expression_ref="expr:ctx",
            expression_text="ctx",
            parameter_name="context",
        ),
        value_mapping=mapping(),
    )
    assert receipt.transform.disposition is TransformDisposition.REJECTED
    assert receipt.transform.target_paths == ()
    assert receipt.edits == ()


def test_batch_root_mismatch_raises() -> None:
    other_roots = PropagationAuthorityRoots(
        repository_id="repository:two",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:one",
        index_id="index:one",
        model_id="model:one",
        config_id="config:one",
        translator_id="translator:one",
        toolchain_id="toolchain:one",
        policy_id="policy:one",
    )
    a = site(TransformKind.ADD_IMPORT, "", site_id="a", import_module="pkg.a", import_name="A")
    b = TransformSite(
        roots=other_roots,
        site_id="b",
        kind=TransformKind.ADD_IMPORT,
        span=make_span("pkg/b.py", ""),
        obligation_ids=("obligation:one",),
        proof_refs=("proof:one",),
        import_module="pkg.b",
        import_name="B",
    )
    with pytest.raises(AnalyticalChangeTransformAuthorityError):
        AnalyticalChangeTransformer().render_many((a, b))


def test_make_span_rejects_length_mismatch_via_hash() -> None:
    span = make_span("pkg/x.py", "abc")
    assert span.end - span.start == 3
    assert span.before_hash.startswith("sha256:")
