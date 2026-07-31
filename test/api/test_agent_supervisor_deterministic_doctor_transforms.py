"""Fail-closed coverage for deterministic doctor AST repair transforms (LPR-033)."""

from __future__ import annotations

import hashlib

import pytest

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
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DoctorAuthorityRoots,
    DoctorOperatorKind,
    DoctorRepairDisposition,
    DoctorRepairOperatorSpec,
    is_doctor_tcb_path,
)
from ipfs_accelerate_py.agent_supervisor.planning.analytical_change_transforms import (
    FieldMapping,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_transforms import (
    DOCTOR_REPAIR_OPERATOR_REGISTRY_INTERFACE,
    PRODUCER_ID,
    RENDERER_ID,
    DoctorOperatorProposal,
    DoctorOperatorRejectionReason,
    DoctorRepairOperatorRegistry,
    DoctorTransformAuthorityError,
    DoctorTransformError,
    DoctorTransformUnsupportedError,
    build_default_doctor_operator_registry,
    default_operator_registry_id,
    doctor_roots_to_propagation_roots,
    make_edit_site,
)
from ipfs_accelerate_py.agent_supervisor.proof.missing_input_synthesis import (
    SynthesisDisposition,
    ValueMappingProof,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def roots(**overrides: str) -> DoctorAuthorityRoots:
    base = {
        "repository_id": "repository:fixture",
        "forest_id": "forest:fixture",
        "tree_id": "tree:fixture",
        "overlay_id": "overlay:fixture",
        "file_root_id": "file-root:fixture",
        "ast_root_id": "ast:fixture",
        "graph_id": "graph:fixture",
        "corpus_id": "corpus:fixture",
        "index_id": "index:fixture",
        "model_id": "model:fixture",
        "cache_id": "cache:fixture",
        "operator_registry_id": "operators:fixture",
        "translator_id": "translator:fixture",
        "solver_id": "solver:fixture",
        "kernel_id": "kernel:fixture",
        "toolchain_id": "toolchain:fixture",
        "policy_id": "policy:fixture",
        "sandbox_id": "sandbox:fixture",
        "environment_id": "environment:fixture",
        "lease_id": "lease:fixture",
    }
    base.update(overrides)
    return DoctorAuthorityRoots(**base)


def registry(auth: DoctorAuthorityRoots | None = None) -> DoctorRepairOperatorRegistry:
    return build_default_doctor_operator_registry(auth or roots())


def mapping(
    *,
    disposition: SynthesisDisposition = SynthesisDisposition.UNIQUE_PROVED,
    expression_ref: str = "expr:ctx",
    proved: tuple[str, ...] | None = None,
    repository_id: str = "repository:fixture",
    tree_id: str = "tree:fixture",
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
        toolchain_id="toolchain:fixture",
        policy_id="policy:fixture",
        reason_codes=(
            ("unique_source",)
            if disposition is SynthesisDisposition.UNIQUE_PROVED
            else ("non_unique",)
        ),
    )


def repair_decision(
    path: str = "pkg/caller.py",
    *,
    admitted: bool = True,
) -> RepairTargetDecision:
    repair_roots = AuthorityRoots(
        repository_id="repository:fixture",
        forest_id="forest:fixture",
        tree_id="tree:fixture",
        graph_id="graph:fixture",
        index_id="index:fixture",
        model_id="model:fixture",
        config_id="config:fixture",
        translator_id="translator:fixture",
        toolchain_id="toolchain:fixture",
        policy_id="policy:fixture",
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
        invalidation_refs=("tree:fixture",),
    )


def propose_add_argument(
    reg: DoctorRepairOperatorRegistry,
    source: str = "process(event)",
    *,
    path: str = "pkg/caller.py",
    proof_admitted: bool = True,
    **kwargs: object,
) -> DoctorOperatorProposal:
    site = make_edit_site(path, source)
    defaults: dict[str, object] = {
        "obligation_refs": ("obligation:one",),
        "proof_refs": ("proof:one",),
        "value_source_refs": ("value:ctx",),
        "expression_ref": "expr:ctx",
        "parameter_name": "context",
        "proof_admitted": proof_admitted,
    }
    defaults.update(kwargs)
    return reg.propose(DoctorOperatorKind.ADD_ARGUMENT, site, **defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Registry identity and closed set
# ---------------------------------------------------------------------------


def test_interface_and_default_registry_cover_every_kind() -> None:
    reg = registry()
    assert reg.INTERFACE == DOCTOR_REPAIR_OPERATOR_REGISTRY_INTERFACE
    assert DOCTOR_REPAIR_OPERATOR_REGISTRY_INTERFACE == "DoctorRepairOperatorRegistry@1"
    assert set(reg.kinds()) == set(DoctorOperatorKind)
    assert len(reg.descriptors) == len(DoctorOperatorKind)
    assert reg.producer_id == PRODUCER_ID
    assert reg.registry_id == default_operator_registry_id(roots())
    # Deterministic identity across rebuilds.
    assert registry().registry_id == reg.registry_id


def test_every_operator_declares_closed_contracts() -> None:
    reg = registry()
    for descriptor in reg.descriptors:
        spec = descriptor.spec
        assert isinstance(spec, DoctorRepairOperatorSpec)
        assert spec.supported_languages == ("python",)
        assert spec.precondition_refs
        assert spec.postcondition_refs
        assert spec.frame_condition_refs
        assert spec.proof_template_refs
        assert spec.placement_constraints
        assert spec.forbidden_paths
        assert spec.renderer_id
        assert spec.idempotent is True
        assert spec.inverse_or_compensation_ref
        assert spec.grants_write_authority is False
        assert spec.semantic_authority is False
        assert descriptor.input_type_refs
        assert descriptor.output_type_refs
        assert descriptor.supported_ast_shapes
        # Analytical kinds map to the existing transformer except restore.
        if descriptor.kind is not DoctorOperatorKind.RESTORE_TRACKED_ARTIFACT:
            assert descriptor.analytical_kind
            assert RENDERER_ID in spec.renderer_id or spec.renderer_id == RENDERER_ID


def test_registry_is_immutable_and_body_free() -> None:
    reg = registry()
    payload = reg.to_dict()
    assert "source_text" not in payload
    assert "span_text" not in payload
    assert "replacement" not in payload
    with pytest.raises(Exception):
        reg.descriptors = ()  # type: ignore[misc]


def test_lookup_unknown_operator_fails_closed() -> None:
    reg = registry()
    with pytest.raises(DoctorTransformUnsupportedError, match="unknown_operator"):
        reg.get("operator:does-not-exist")


def test_doctor_roots_bridge_keeps_candidate_binding() -> None:
    auth = roots()
    prop = doctor_roots_to_propagation_roots(auth)
    assert prop.repository_id == auth.repository_id
    assert prop.candidate_tree_id == auth.tree_id
    assert prop.base_tree_id != prop.candidate_tree_id
    assert prop.toolchain_id == auth.toolchain_id


# ---------------------------------------------------------------------------
# Body-free proposals until proof admission
# ---------------------------------------------------------------------------


def test_propose_is_body_free() -> None:
    reg = registry()
    proposal = propose_add_argument(reg, proof_admitted=False)
    payload = proposal.to_dict()
    for forbidden in (
        "source_text",
        "span_text",
        "body",
        "replacement",
        "code",
        "snippet",
    ):
        assert forbidden not in payload
        assert forbidden not in payload["edit_site"]
    assert proposal.grants_write_authority is False
    assert proposal.semantic_authority is False
    assert proposal.proof_admitted is False
    # Round-trip.
    restored = DoctorOperatorProposal.from_dict(payload)
    assert restored.proposal_id == proposal.proposal_id
    assert restored.edit_site.before_hash == proposal.edit_site.before_hash


def test_evaluate_without_proof_admission_abstains() -> None:
    reg = registry()
    proposal = propose_add_argument(reg, proof_admitted=False)
    receipt = reg.evaluate(proposal, value_mapping=mapping())
    assert receipt.disposition is DoctorRepairDisposition.ABSTAIN
    assert DoctorOperatorRejectionReason.PROOF_NOT_ADMITTED.value in receipt.rejection_reasons
    assert receipt.expected_after_hash == ""
    assert not receipt.admitted


def test_evaluate_proof_admitted_still_requires_render() -> None:
    reg = registry()
    proposal = propose_add_argument(reg, proof_admitted=True)
    receipt = reg.evaluate(proposal, value_mapping=mapping())
    assert receipt.disposition is DoctorRepairDisposition.ABSTAIN
    assert DoctorOperatorRejectionReason.RENDER_REQUIRED.value in receipt.rejection_reasons


def test_render_without_proof_admission_abstains() -> None:
    reg = registry()
    source = "process(event)"
    proposal = propose_add_argument(reg, source, proof_admitted=False)
    receipt, render = reg.render_admitted(
        proposal,
        span_text=source,
        expression_text="ctx",
        value_mapping=mapping(),
    )
    assert not receipt.admitted
    assert render is None
    assert DoctorOperatorRejectionReason.PROOF_NOT_ADMITTED.value in receipt.rejection_reasons


# ---------------------------------------------------------------------------
# Admitted renders for each eligible operator family
# ---------------------------------------------------------------------------


def test_add_argument_renders_keyword_from_unique_proved_expression() -> None:
    reg = registry()
    source = "process(event)"
    proposal = propose_add_argument(reg, source, proof_admitted=True)
    receipt, render = reg.render_admitted(
        proposal,
        span_text=source,
        expression_text="ctx",
        value_mapping=mapping(),
    )
    assert receipt.admitted
    assert render is not None
    assert render.edits[0].replacement == "process(event, context=ctx)"
    assert receipt.expected_after_hash == render.edits[0].expected_after_hash
    # Receipt itself stays body-free.
    assert "replacement" not in receipt.to_dict()
    assert "process(event" not in str(receipt.to_dict())


def test_exact_rename_identifier() -> None:
    reg = registry()
    source = "old_name"
    site = make_edit_site("pkg/symbols.py", source)
    proposal = reg.propose(
        DoctorOperatorKind.EXACT_RENAME,
        site,
        obligation_refs=("obligation:rename",),
        proof_refs=("proof:rename",),
        parameter_name="new_name",
        previous_parameter_name="old_name",
        proof_admitted=True,
    )
    receipt, render = reg.render_admitted(proposal, span_text=source)
    assert receipt.admitted
    assert render is not None
    assert render.edits[0].replacement == "new_name"


def test_rename_argument_keyword() -> None:
    reg = registry()
    source = "emit(event=event, tenant=tenant_id)"
    site = make_edit_site("pkg/caller.py", source)
    proposal = reg.propose(
        DoctorOperatorKind.RENAME_ARGUMENT,
        site,
        obligation_refs=("obligation:one",),
        proof_refs=("proof:one",),
        value_source_refs=("value:rename",),
        expression_ref="expr:ctx",
        previous_parameter_name="tenant",
        parameter_name="tenant_id",
        proof_admitted=True,
    )
    receipt, render = reg.render_admitted(
        proposal, span_text=source, expression_text="tenant_id", value_mapping=mapping()
    )
    assert receipt.admitted
    assert render is not None
    assert render.edits[0].replacement == "emit(event=event, tenant_id=tenant_id)"


def test_reorder_argument_total_mapping() -> None:
    reg = registry()
    source = "emit(b=2, a=1)"
    site = make_edit_site("pkg/caller.py", source)
    proposal = reg.propose(
        DoctorOperatorKind.REORDER_ARGUMENT,
        site,
        obligation_refs=("obligation:one",),
        proof_refs=("proof:one",),
        value_source_refs=("value:reorder",),
        expression_ref="expr:ctx",
        argument_order=("a", "b"),
        proof_admitted=True,
    )
    receipt, render = reg.render_admitted(
        proposal, span_text=source, expression_text="a", value_mapping=mapping()
    )
    assert receipt.admitted
    assert render is not None
    assert render.edits[0].replacement == "emit(a=1, b=2)"


def test_add_import_export_registration() -> None:
    reg = registry()
    import_src = "import os\n"
    site = make_edit_site("pkg/mod.py", import_src)
    import_prop = reg.propose(
        DoctorOperatorKind.ADD_IMPORT,
        site,
        obligation_refs=("obligation:one",),
        proof_refs=("proof:one",),
        import_module="pkg.types",
        import_name="Context",
        allowed_dependency_paths=("pkg/",),
        proof_admitted=True,
    )
    receipt, render = reg.render_admitted(import_prop, span_text=import_src)
    assert receipt.admitted
    assert render is not None
    assert "from pkg.types import Context" in render.edits[0].replacement

    export_src = '__all__ = ["existing"]'
    export_site = make_edit_site("pkg/mod.py", export_src)
    export_prop = reg.propose(
        DoctorOperatorKind.ADD_EXPORT,
        export_site,
        obligation_refs=("obligation:one",),
        proof_refs=("proof:one",),
        export_name="Context",
        proof_admitted=True,
    )
    receipt, render = reg.render_admitted(export_prop, span_text=export_src)
    assert receipt.admitted
    assert '"Context"' in render.edits[0].replacement

    reg_src = "REGISTRY.register()"
    reg_site = make_edit_site("pkg/mod.py", reg_src)
    reg_prop = reg.propose(
        DoctorOperatorKind.ADD_REGISTRATION,
        reg_site,
        obligation_refs=("obligation:one",),
        proof_refs=("proof:one",),
        registration_name="context",
        registration_target="Context",
        proof_admitted=True,
    )
    receipt, render = reg.render_admitted(reg_prop, span_text=reg_src)
    assert receipt.admitted
    assert render.edits[0].replacement == 'REGISTRY.register("context", Context)'


def test_constructor_factory_adapter_and_schema() -> None:
    reg = registry()
    ctor_src = "Context(request_id)"
    site = make_edit_site("pkg/caller.py", ctor_src)
    ctor = reg.propose(
        DoctorOperatorKind.ADD_CONSTRUCTOR_ROUTE,
        site,
        obligation_refs=("obligation:one",),
        proof_refs=("proof:one",),
        value_source_refs=("value:tenant",),
        expression_ref="expr:ctx",
        parameter_name="tenant",
        proof_admitted=True,
    )
    receipt, render = reg.render_admitted(
        ctor, span_text=ctor_src, expression_text="tenant_id", value_mapping=mapping()
    )
    assert receipt.admitted
    assert render.edits[0].replacement == "Context(request_id, tenant=tenant_id)"

    factory = reg.propose(
        DoctorOperatorKind.ADD_FACTORY_ROUTE,
        site,
        obligation_refs=("obligation:one",),
        proof_refs=("proof:one",),
        value_source_refs=("value:tenant",),
        expression_ref="expr:ctx",
        parameter_name="tenant",
        proof_admitted=True,
    )
    receipt, render = reg.render_admitted(
        factory, span_text=ctor_src, expression_text="tenant_id", value_mapping=mapping()
    )
    assert receipt.admitted

    adapter_src = "ctx"
    adapter_site = make_edit_site("pkg/caller.py", adapter_src)
    adapter = reg.propose(
        DoctorOperatorKind.FINITE_ADAPTER,
        adapter_site,
        obligation_refs=("obligation:one",),
        proof_refs=("proof:one",),
        value_source_refs=("value:ctx",),
        expression_ref="expr:ctx",
        adapter_expression="ContextAdapter",
        proof_admitted=True,
    )
    receipt, render = reg.render_admitted(
        adapter, span_text=adapter_src, expression_text="ctx", value_mapping=mapping()
    )
    assert receipt.admitted
    assert render.edits[0].replacement == "ContextAdapter(ctx)"

    schema_src = '{"tenant":"t1","name":"n"}'
    schema_site = make_edit_site("pkg/schema.json", schema_src)
    schema = reg.propose(
        DoctorOperatorKind.SCHEMA_PROJECTION,
        schema_site,
        obligation_refs=("obligation:one",),
        proof_refs=("proof:one",),
        field_mapping_refs=("map:tenant->tenant_id", "map:name->display_name"),
        proof_admitted=True,
    )
    receipt, render = reg.render_admitted(
        schema,
        span_text=schema_src,
        field_mappings=(
            FieldMapping("tenant", "tenant_id"),
            FieldMapping("name", "display_name"),
        ),
    )
    assert receipt.admitted
    assert render.edits[0].replacement == '{"display_name":"n","tenant_id":"t1"}'


def test_restore_tracked_artifact_from_verified_preimage() -> None:
    reg = registry()
    stale = "old-content"
    restored = "new-verified-content\n"
    preimage = "sha256:" + hashlib.sha256(restored.encode()).hexdigest()
    site = make_edit_site("pkg/artifact.txt", stale)
    proposal = reg.propose(
        DoctorOperatorKind.RESTORE_TRACKED_ARTIFACT,
        site,
        obligation_refs=("obligation:restore",),
        proof_refs=("proof:restore",),
        value_source_refs=("value:artifact",),
        artifact_cid="cid:artifact:1",
        artifact_preimage_hash=preimage,
        proof_admitted=True,
    )
    receipt, render = reg.render_admitted(
        proposal,
        span_text=stale,
        verified_artifact_bytes=restored.encode(),
    )
    assert receipt.admitted
    assert render is not None
    assert render.edits[0].replacement == restored


def test_restore_rejects_preimage_mismatch() -> None:
    reg = registry()
    stale = "old"
    site = make_edit_site("pkg/artifact.txt", stale)
    proposal = reg.propose(
        DoctorOperatorKind.RESTORE_TRACKED_ARTIFACT,
        site,
        obligation_refs=("obligation:restore",),
        proof_refs=("proof:restore",),
        value_source_refs=("value:artifact",),
        artifact_cid="cid:artifact:1",
        artifact_preimage_hash="sha256:" + "0" * 64,
        proof_admitted=True,
    )
    receipt, render = reg.render_admitted(
        proposal,
        span_text=stale,
        verified_artifact_bytes=b"other",
    )
    assert not receipt.admitted
    assert render is None
    assert DoctorOperatorRejectionReason.RESTORE_CID_MISMATCH.value in receipt.rejection_reasons


# ---------------------------------------------------------------------------
# Rejection matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        ("process(event, *rest)", DoctorOperatorRejectionReason.DYNAMIC_SPLAT),
        ("process(event, **kwargs)", DoctorOperatorRejectionReason.DYNAMIC_SPLAT),
    ],
)
def test_rejects_splats(
    source: str, reason: DoctorOperatorRejectionReason
) -> None:
    reg = registry()
    proposal = propose_add_argument(reg, source, proof_admitted=True)
    receipt, render = reg.render_admitted(
        proposal,
        span_text=source,
        expression_text="ctx",
        value_mapping=mapping(),
    )
    assert not receipt.admitted
    assert reason.value in receipt.rejection_reasons
    assert render is None or not render.admitted


def test_rejects_ambiguous_overload_at_propose() -> None:
    reg = registry()
    site = make_edit_site("pkg/caller.py", "process(event)")
    with pytest.raises(DoctorTransformUnsupportedError, match="ambiguous_overload"):
        reg.propose(
            DoctorOperatorKind.ADD_ARGUMENT,
            site,
            obligation_refs=("obligation:one",),
            proof_refs=("proof:one",),
            value_source_refs=("value:ctx",),
            expression_ref="expr:ctx",
            parameter_name="context",
            overload_count=2,
            proof_admitted=True,
        )


def test_rejects_reflection_and_monkey_patch_markers() -> None:
    reg = registry()
    site = make_edit_site("pkg/caller.py", "process(event)")
    with pytest.raises(DoctorTransformUnsupportedError, match="reflection"):
        reg.propose(
            DoctorOperatorKind.ADD_ARGUMENT,
            site,
            obligation_refs=("obligation:one",),
            proof_refs=("proof:one",),
            value_source_refs=("value:ctx",),
            expression_ref="expr:reflection",
            parameter_name="context",
            proof_admitted=True,
        )
    with pytest.raises(DoctorTransformUnsupportedError, match="monkey_patch"):
        reg.propose(
            DoctorOperatorKind.ADD_ARGUMENT,
            site,
            obligation_refs=("obligation:one",),
            proof_refs=("proof:one",),
            value_source_refs=("value:ctx",),
            expression_ref="expr:ctx",
            parameter_name="monkey_patch",
            proof_admitted=True,
        )


def test_rejects_native_ffi_unsafe_concurrency_generated_markers() -> None:
    reg = registry()
    site = make_edit_site("pkg/caller.py", "process(event)")
    for marker, expected in (
        ("native_helper", "native_or_ffi"),
        ("unsafe_ptr", "unsafe_target"),
        ("threading_local", "concurrency_target"),
        ("generated_stub", "generated_target"),
    ):
        with pytest.raises(DoctorTransformUnsupportedError, match=expected):
            reg.propose(
                DoctorOperatorKind.ADD_ARGUMENT,
                site,
                obligation_refs=("obligation:one",),
                proof_refs=("proof:one",),
                value_source_refs=("value:ctx",),
                expression_ref=f"expr:{marker}",
                parameter_name="context",
                proof_admitted=True,
            )


def test_rejects_stale_span_on_render() -> None:
    reg = registry()
    source = "process(event)"
    proposal = propose_add_argument(reg, source, proof_admitted=True)
    receipt, render = reg.render_admitted(
        proposal,
        span_text="process(other)",  # does not match before_hash
        expression_text="ctx",
        value_mapping=mapping(),
    )
    assert not receipt.admitted
    assert DoctorOperatorRejectionReason.STALE_SPAN.value in receipt.rejection_reasons
    assert render is None


def test_rejects_incomplete_schema_mapping() -> None:
    reg = registry()
    source = '{"only":1}'
    site = make_edit_site("test/fixtures/one.json", source)
    # Missing field_mapping_refs → preflight incomplete_mapping.
    proposal = reg.propose(
        DoctorOperatorKind.SCHEMA_PROJECTION,
        site,
        obligation_refs=("obligation:one",),
        proof_refs=("proof:one",),
        field_mapping_refs=(),
        proof_admitted=True,
    )
    receipt = reg.evaluate(proposal)
    assert DoctorOperatorRejectionReason.INCOMPLETE_MAPPING.value in receipt.rejection_reasons

    # Present refs but non-total mapping → render rejects.
    proposal2 = reg.propose(
        DoctorOperatorKind.SCHEMA_PROJECTION,
        site,
        obligation_refs=("obligation:one",),
        proof_refs=("proof:one",),
        field_mapping_refs=("map:missing->other",),
        proof_admitted=True,
    )
    receipt2, _ = reg.render_admitted(
        proposal2,
        span_text=source,
        field_mappings=(FieldMapping("missing", "other"),),
    )
    assert not receipt2.admitted
    assert DoctorOperatorRejectionReason.INCOMPLETE_MAPPING.value in receipt2.rejection_reasons


def test_rejects_unproved_values() -> None:
    reg = registry()
    source = "process(event)"
    proposal = propose_add_argument(reg, source, proof_admitted=True)
    receipt, _ = reg.render_admitted(
        proposal,
        span_text=source,
        expression_text="ctx",
        value_mapping=mapping(
            disposition=SynthesisDisposition.AMBIGUOUS,
            proved=("candidate:a", "candidate:b"),
        ),
    )
    assert not receipt.admitted
    assert DoctorOperatorRejectionReason.UNPROVED_VALUE.value in receipt.rejection_reasons


def test_rejects_invented_expression_behavior() -> None:
    reg = registry()
    source = "process(event)"
    proposal = propose_add_argument(reg, source, proof_admitted=True)
    receipt, _ = reg.render_admitted(
        proposal,
        span_text=source,
        expression_text="make_context()",
        value_mapping=mapping(),
    )
    assert not receipt.admitted
    assert DoctorOperatorRejectionReason.INVENTED_BEHAVIOR.value in receipt.rejection_reasons


def test_rejects_forbidden_and_tcb_paths() -> None:
    reg = registry()
    tcb = "ipfs_accelerate_py/agent_supervisor/proof/kernel_verification.py"
    assert is_doctor_tcb_path(tcb)
    with pytest.raises(Exception):
        # DoctorEditSite itself rejects TCB paths.
        make_edit_site(tcb, "process(event)")

    # Forbidden non-TCB path under the default registry forbidden set.
    forbidden = "vendor/external_pkg/module.py"
    assert not is_doctor_tcb_path(forbidden)
    site = make_edit_site(forbidden, "process(event)")
    with pytest.raises(DoctorTransformAuthorityError):
        reg.propose(
            DoctorOperatorKind.ADD_ARGUMENT,
            site,
            obligation_refs=("obligation:one",),
            proof_refs=("proof:one",),
            value_source_refs=("value:ctx",),
            expression_ref="expr:ctx",
            parameter_name="context",
            proof_admitted=True,
        )


def test_rejects_cross_root_value_mapping() -> None:
    reg = registry()
    source = "process(event)"
    proposal = propose_add_argument(reg, source, proof_admitted=True)
    receipt, _ = reg.render_admitted(
        proposal,
        span_text=source,
        expression_text="ctx",
        value_mapping=mapping(tree_id="tree:other"),
    )
    assert not receipt.admitted
    assert DoctorOperatorRejectionReason.CROSS_ROOT_WRITE.value in receipt.rejection_reasons


def test_rejects_unauthorized_write_path() -> None:
    reg = registry()
    source = "process(event)"
    proposal = propose_add_argument(
        reg, source, path="pkg/other.py", proof_admitted=True
    )
    receipt, _ = reg.render_admitted(
        proposal,
        span_text=source,
        expression_text="ctx",
        value_mapping=mapping(),
        decision=repair_decision(path="pkg/caller.py", admitted=True),
    )
    assert not receipt.admitted
    assert DoctorOperatorRejectionReason.PATH_NOT_AUTHORIZED.value in receipt.rejection_reasons


def test_rejects_new_dependency_outside_allowlist() -> None:
    reg = registry()
    source = "import os\n"
    site = make_edit_site("pkg/mod.py", source)
    proposal = reg.propose(
        DoctorOperatorKind.ADD_IMPORT,
        site,
        obligation_refs=("obligation:one",),
        proof_refs=("proof:one",),
        import_module="external.vendor",
        import_name="Thing",
        allowed_dependency_paths=("pkg/",),
        proof_admitted=True,
    )
    receipt = reg.evaluate(proposal)
    assert DoctorOperatorRejectionReason.NEW_DEPENDENCY.value in receipt.rejection_reasons


def test_rejects_proposal_with_body_fields() -> None:
    reg = registry()
    proposal = propose_add_argument(reg, proof_admitted=False)
    payload = proposal.to_dict()
    payload["source_text"] = "process(event)"
    with pytest.raises(DoctorTransformAuthorityError, match="body_in_proposal"):
        # Reconstruct via manual fields would still not accept source_text;
        # assert the body-free guard on registry payload validation path.
        from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_transforms import (
            _assert_body_free_mapping,
        )

        _assert_body_free_mapping(payload, "proposal")


def test_rejects_write_or_semantic_authority_on_proposal() -> None:
    reg = registry()
    site = make_edit_site("pkg/caller.py", "process(event)")
    with pytest.raises(DoctorTransformAuthorityError, match="write_authority"):
        DoctorOperatorProposal(
            roots=reg.roots,
            proposal_id="proposal:bad",
            operator_id="operator:add_argument",
            kind=DoctorOperatorKind.ADD_ARGUMENT,
            edit_site=site,
            obligation_refs=("obligation:one",),
            grants_write_authority=True,
        )


# ---------------------------------------------------------------------------
# Idempotency / determinism
# ---------------------------------------------------------------------------


def test_repeated_render_is_byte_equivalent() -> None:
    reg = registry()
    source = "process(event)"
    proposal = propose_add_argument(reg, source, proof_admitted=True)
    first, first_render = reg.render_admitted(
        proposal,
        span_text=source,
        expression_text="ctx",
        value_mapping=mapping(),
    )
    second, second_render = reg.render_admitted(
        proposal,
        span_text=source,
        expression_text="ctx",
        value_mapping=mapping(),
    )
    assert first.admitted and second.admitted
    assert first_render is not None and second_render is not None
    assert first_render.edits[0].replacement == second_render.edits[0].replacement
    assert first.expected_after_hash == second.expected_after_hash
    assert first.replay_identity == second.replay_identity


def test_reapplying_transform_is_noop() -> None:
    reg = registry()
    source = "process(event)"
    proposal = propose_add_argument(reg, source, proof_admitted=True)
    assert reg.render_admitted_repeat_is_noop(
        proposal,
        span_text=source,
        expression_text="ctx",
        value_mapping=mapping(),
    )


def test_already_present_add_argument_is_idempotent_noop() -> None:
    reg = registry()
    source = "process(event, context=ctx)"
    proposal = propose_add_argument(reg, source, proof_admitted=True)
    receipt, render = reg.render_admitted(
        proposal,
        span_text=source,
        expression_text="ctx",
        value_mapping=mapping(),
    )
    assert receipt.admitted
    assert render is not None
    assert render.edits[0].replacement == source
    assert receipt.idempotent_noop is True


def test_before_and_after_hashes_bind_exact_text() -> None:
    reg = registry()
    source = "process(event)"
    proposal = propose_add_argument(reg, source, proof_admitted=True)
    receipt, render = reg.render_admitted(
        proposal,
        span_text=source,
        expression_text="ctx",
        value_mapping=mapping(),
    )
    assert receipt.admitted and render is not None
    edit = render.edits[0]
    assert edit.before_hash == "sha256:" + hashlib.sha256(source.encode()).hexdigest()
    assert (
        edit.expected_after_hash
        == "sha256:" + hashlib.sha256(edit.replacement.encode()).hexdigest()
    )
    assert receipt.replacement_hash == edit.expected_after_hash


def test_thread_argument_requires_route_site_ids() -> None:
    reg = registry()
    source = "inner(x)"
    site = make_edit_site("pkg/caller.py", source)
    proposal = reg.propose(
        DoctorOperatorKind.THREAD_ARGUMENT,
        site,
        obligation_refs=("obligation:one",),
        proof_refs=("proof:one",),
        value_source_refs=("value:ctx",),
        expression_ref="expr:ctx",
        parameter_name="context",
        route_site_ids=(),
        proof_admitted=True,
    )
    receipt = reg.evaluate(proposal, value_mapping=mapping())
    assert DoctorOperatorRejectionReason.INCOMPLETE_MAPPING.value in receipt.rejection_reasons


def test_specs_never_grant_write_authority() -> None:
    reg = registry()
    for spec in reg.specs():
        assert spec.grants_write_authority is False
        with pytest.raises(Exception):
            DoctorRepairOperatorSpec(
                roots=reg.roots,
                operator_id=spec.operator_id + ":bad",
                kind=spec.kind,
                supported_languages=("python",),
                precondition_refs=spec.precondition_refs,
                postcondition_refs=spec.postcondition_refs,
                grants_write_authority=True,
            )


def test_root_mismatch_between_proposal_and_registry() -> None:
    reg = registry()
    other = registry(roots(repository_id="repository:other", tree_id="tree:other"))
    source = "process(event)"
    proposal = propose_add_argument(other, source, proof_admitted=True)
    receipt = reg.evaluate(proposal, value_mapping=mapping(repository_id="repository:other", tree_id="tree:other"))
    assert DoctorOperatorRejectionReason.ROOT_MISMATCH.value in receipt.rejection_reasons
