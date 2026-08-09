"""DCR-045: UI, ORB, IDL, mobile, and projection repair operators.

Acceptance:
* Every edited projection roundtrips to the same semantic IR.
* Live UI actions reach the expected mediated MCP effect.
* Bridge-only, prose-inferred, or missing target projections abstain.
* Operators remain proposal-only and never grant write/proof/semantic authority.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_catalog import (
    MCP_IDL_INTERFACE,
    ORB_INTERFACE,
    UIIR_DOCUMENT_INTERFACE,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorFamily,
    OperatorKind,
    build_default_operator_registry,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.ui_projection_repairs import (
    MCP_IDL_INTERFACE as OP_MCP_IDL,
    ORB_INTERFACE as OP_ORB,
    UI_PROJECTION_REPAIR_OPERATORS_INTERFACE,
    UI_REPAIR_EVIDENCE,
    UI_UX_IR_SCHEMA_VERSION,
    UIIR_DOCUMENT_INTERFACE as OP_UIIR,
    IdlProjectionOperator,
    LiveActionTrace,
    MediationPathClass,
    MobileProjectionOperator,
    OperatorRole,
    OrbBindingOperator,
    ProjectionSurface,
    RepairDisposition,
    SourceAuthority,
    SurfaceProjection,
    UiActionBinding,
    UiComponentNode,
    UiDescriptorOperator,
    UiProjectionRepairError,
    UiProjectionRepairReceipt,
    UiProjectionRepairRequest,
    UIIRSemanticDocument,
    UIProjectionRepairOperators,
    assert_semantic_roundtrip,
    build_ui_projection_repair_operators,
    materialize_ui_projection_operator_vectors,
    project_semantic_ir,
    projection_diff,
    semantic_ir_from_projection,
    verify_mediated_mcp_effects,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


def _cid(label: str) -> str:
    return content_identity({"dcr045": label})


def _action(
    *,
    action_id: str = "action:invoke",
    method: str = "tools/call",
    effect: str = "effect:invoke",
) -> UiActionBinding:
    return UiActionBinding(
        action_id=action_id,
        label=f"Run {method}",
        mcp_method=method,
        interface_cid=_cid(f"iface:{method}"),
        effect_id=effect,
        mediation_path=MediationPathClass.GOVERNED_MEDIATOR,
        argument_schema_cid=_cid(f"arg:{method}"),
        result_schema_cid=_cid(f"result:{method}"),
        confirmation_required=False,
    )


def _document(
    *,
    authority: SourceAuthority = SourceAuthority.FULL_UI_UX_IR,
    actions: tuple[UiActionBinding, ...] | None = None,
) -> UIIRSemanticDocument:
    action = actions[0] if actions else _action()
    bound_actions = actions or (action,)
    root = UiComponentNode(
        component_id="component:root",
        role="panel",
        purpose="Primary surface",
        action_ids=(),
        child_ids=("component:invoke",),
    )
    button = UiComponentNode(
        component_id="component:invoke",
        role="button",
        purpose="Invoke mediated MCP method",
        action_ids=tuple(item.action_id for item in bound_actions),
        child_ids=(),
    )
    return UIIRSemanticDocument(
        document_id="uiir:dcr045:demo",
        title="DCR-045 Demo Surface",
        components=(root, button),
        actions=bound_actions,
        entry_components=("component:root",),
        terminal_outcomes=("success", "failure"),
        authority=authority,
        source_refs=("source:reviewed-descriptor",),
    )


def _traces_for(document: UIIRSemanticDocument) -> tuple[LiveActionTrace, ...]:
    return tuple(
        LiveActionTrace(
            action_id=action.action_id,
            mcp_method=action.mcp_method,
            effect_id=action.effect_id,
            mediation_path=action.mediation_path,
            terminal_state="passed",
            receipt_cid=_cid(f"receipt:{action.action_id}"),
        )
        for action in document.actions
    )


def _drifted_projection(
    document: UIIRSemanticDocument,
    surface: ProjectionSurface,
) -> SurfaceProjection:
    expected = project_semantic_ir(document, surface)
    # Introduce deterministic drift on a presentation-adjacent action label.
    drifted_actions = tuple(
        {
            **dict(action),
            "label": f"STALE::{action['label']}",
        }
        for action in expected.actions
    )
    return SurfaceProjection(
        surface=expected.surface,
        projection_id=expected.projection_id,
        document_id=expected.document_id,
        nodes=expected.nodes,
        actions=drifted_actions,
        source_schema_cid=expected.source_schema_cid,
        target_schema_cid=expected.target_schema_cid,
        authority=SourceAuthority.PRODUCTION,
        mcp_interface_cid=expected.mcp_interface_cid,
        orb_interface_cid=expected.orb_interface_cid,
        semantic_digest="sha256:" + ("0" * 64),  # stale semantic digest
        mediation_path=expected.mediation_path,
    )


# ---------------------------------------------------------------------------
# Interface / registry binding
# ---------------------------------------------------------------------------


def test_interfaces_and_evidence_are_declared() -> None:
    assert UI_PROJECTION_REPAIR_OPERATORS_INTERFACE == "UIProjectionRepairOperators@1"
    assert UI_REPAIR_EVIDENCE == "dcr/ui-repair@1"
    assert UI_UX_IR_SCHEMA_VERSION == "ui-ux-ir/v1"
    assert OP_UIIR == UIIR_DOCUMENT_INTERFACE == "UIIRDocument"
    assert OP_MCP_IDL == MCP_IDL_INTERFACE == "MCP-IDL"
    assert OP_ORB == ORB_INTERFACE == "ORB"
    ops = build_ui_projection_repair_operators()
    assert ops.INTERFACE == UI_PROJECTION_REPAIR_OPERATORS_INTERFACE
    assert ops.EVIDENCE_ID == UI_REPAIR_EVIDENCE
    assert isinstance(ops.ui_descriptor, UiDescriptorOperator)
    assert isinstance(ops.orb_binding, OrbBindingOperator)
    assert isinstance(ops.idl_projection, IdlProjectionOperator)
    assert isinstance(ops.mobile_projection, MobileProjectionOperator)


def test_registry_binds_repair_ui_projection_to_ui_family() -> None:
    reg = build_default_operator_registry()
    descriptor = reg.require_known(OperatorKind.REPAIR_UI_PROJECTION)
    assert descriptor.family is OperatorFamily.UI
    assert descriptor.kind is OperatorKind.REPAIR_UI_PROJECTION
    assert descriptor.proposal_only is True
    assert descriptor.grants_write_authority is False
    assert descriptor.grants_proof_authority is False
    assert descriptor.semantic_authority is False
    assert descriptor.allows_source_generation is False
    assert "scope:closed_ui_projection" in descriptor.write_scope
    assert reg.get("ui_projection").kind is OperatorKind.REPAIR_UI_PROJECTION
    assert reg.get("orb_idl_binding").kind is OperatorKind.REPAIR_UI_PROJECTION


# ---------------------------------------------------------------------------
# Semantic roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "surface",
    (
        ProjectionSurface.DESKTOP,
        ProjectionSurface.WEB,
        ProjectionSurface.CLI,
        ProjectionSurface.MOBILE,
        ProjectionSurface.ORB,
        ProjectionSurface.IDL,
    ),
)
def test_every_surface_projection_roundtrips_to_same_semantic_ir(
    surface: ProjectionSurface,
) -> None:
    document = _document()
    projection = assert_semantic_roundtrip(document, surface)
    restored = semantic_ir_from_projection(
        projection,
        title=document.title,
        entry_components=document.entry_components,
        terminal_outcomes=document.terminal_outcomes,
        authority=document.authority,
    )
    assert restored.semantic_core_digest() == document.semantic_core_digest()
    assert projection.semantic_digest == document.semantic_digest
    assert projection.source_schema_cid == document.schema_cid
    assert projection.document_id == document.document_id


def test_projection_diff_detects_drift_and_alignment() -> None:
    document = _document()
    expected = project_semantic_ir(document, ProjectionSurface.DESKTOP)
    drifted = _drifted_projection(document, ProjectionSurface.DESKTOP)
    aligned = projection_diff(expected, expected)
    assert aligned["kind"] == "aligned"
    assert aligned["changed_paths"] == ()
    drifted_diff = projection_diff(drifted, expected)
    assert drifted_diff["kind"] == "drift"
    assert "actions" in drifted_diff["changed_paths"] or "semantic_digest" in drifted_diff[
        "changed_paths"
    ]
    missing = projection_diff(None, expected)
    assert missing["kind"] == "missing_target"


# ---------------------------------------------------------------------------
# Live mediation
# ---------------------------------------------------------------------------


def test_live_action_reaches_expected_mediated_mcp_effect() -> None:
    document = _document()
    ok, reasons, evidence = verify_mediated_mcp_effects(document, _traces_for(document))
    assert ok is True
    assert "mediated_mcp_effects_verified" in reasons
    assert len(evidence) == len(document.actions)
    assert evidence[0]["observed_effect_id"] == document.actions[0].effect_id


def test_direct_proxy_and_effect_mismatch_fail_closed() -> None:
    document = _document()
    with pytest.raises(UiProjectionRepairError, match="direct_proxy"):
        LiveActionTrace(
            action_id=document.actions[0].action_id,
            mcp_method=document.actions[0].mcp_method,
            effect_id=document.actions[0].effect_id,
            mediation_path=MediationPathClass.DIRECT_PROXY,
            terminal_state="passed",
            receipt_cid=_cid("receipt:proxy"),
        )
    bad = LiveActionTrace(
        action_id=document.actions[0].action_id,
        mcp_method=document.actions[0].mcp_method,
        effect_id="effect:wrong",
        mediation_path=MediationPathClass.TOOLS_CALL,
        terminal_state="passed",
        receipt_cid=_cid("receipt:bad"),
    )
    ok, reasons, _ = verify_mediated_mcp_effects(document, (bad,))
    assert ok is False
    assert any(code.startswith("effect_mismatch:") for code in reasons)


# ---------------------------------------------------------------------------
# Operator apply / abstain / preview / inverse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("surface", "operator_factory"),
    (
        (ProjectionSurface.DESKTOP, lambda ops: ops.ui_descriptor),
        (ProjectionSurface.WEB, lambda ops: ops.ui_descriptor),
        (ProjectionSurface.CLI, lambda ops: ops.ui_descriptor),
        (ProjectionSurface.MOBILE, lambda ops: ops.mobile_projection),
        (ProjectionSurface.ORB, lambda ops: ops.orb_binding),
        (ProjectionSurface.IDL, lambda ops: ops.idl_projection),
    ),
)
def test_edited_projection_roundtrips_and_live_mediation_passes(
    surface: ProjectionSurface,
    operator_factory,
) -> None:
    document = _document()
    ops = build_ui_projection_repair_operators()
    operator = operator_factory(ops)
    current = _drifted_projection(document, surface)
    request = UiProjectionRepairRequest(
        semantic_ir=document,
        surface=surface,
        role=operator.ROLE,
        current_projection=current,
        live_traces=_traces_for(document),
        require_live_mediation=True,
    )
    receipt = operator.apply(request)
    assert receipt.disposition is RepairDisposition.PREVIEW_READY
    assert receipt.semantic_roundtrip_ok is True
    assert receipt.live_mediation_ok is True
    assert receipt.proposal_only is True
    assert receipt.grants_write_authority is False
    assert receipt.grants_proof_authority is False
    assert receipt.semantic_authority is False
    assert receipt.evidence_id == UI_REPAIR_EVIDENCE
    assert receipt.operator_kind == OperatorKind.REPAIR_UI_PROJECTION.value
    assert receipt.source_schema_cid == document.schema_cid
    assert receipt.preview_projection is not None
    # Edited preview roundtrips to the same semantic core.
    restored = semantic_ir_from_projection(
        receipt.preview_projection,
        title=document.title,
        entry_components=document.entry_components,
        terminal_outcomes=document.terminal_outcomes,
        authority=document.authority,
    )
    assert restored.semantic_core_digest() == document.semantic_core_digest()
    # Inverse restores the pre-edit projection identity.
    inverse = operator.inverse(receipt)
    assert inverse is not None
    assert inverse.projection_digest == current.projection_digest
    # Live mediation evidence is retained.
    assert receipt.mediation_evidence
    assert receipt.mediation_evidence[0]["mcp_method"] == document.actions[0].mcp_method


def test_already_aligned_projection_is_idempotent() -> None:
    document = _document()
    expected = project_semantic_ir(document, ProjectionSurface.DESKTOP)
    request = UiProjectionRepairRequest(
        semantic_ir=document,
        surface=ProjectionSurface.DESKTOP,
        role=OperatorRole.UI_DESCRIPTOR,
        current_projection=expected,
        live_traces=_traces_for(document),
    )
    receipt = UiDescriptorOperator().apply(request)
    assert receipt.disposition is RepairDisposition.ALREADY_ALIGNED
    assert receipt.semantic_roundtrip_ok is True
    assert receipt.live_mediation_ok is True
    assert receipt.projection_diff["kind"] == "aligned"


@pytest.mark.parametrize(
    "authority",
    (
        SourceAuthority.BRIDGE_ONLY,
        SourceAuthority.PROSE_INFERRED,
        SourceAuthority.MISSING,
    ),
)
def test_bridge_prose_or_missing_target_abstains(authority: SourceAuthority) -> None:
    document = _document()
    expected = project_semantic_ir(document, ProjectionSurface.WEB)
    current = SurfaceProjection(
        surface=expected.surface,
        projection_id=expected.projection_id,
        document_id=expected.document_id,
        nodes=expected.nodes,
        actions=expected.actions,
        source_schema_cid=expected.source_schema_cid,
        target_schema_cid=expected.target_schema_cid,
        authority=authority,
        mcp_interface_cid=expected.mcp_interface_cid,
        orb_interface_cid=expected.orb_interface_cid,
        semantic_digest=expected.semantic_digest,
        mediation_path=expected.mediation_path,
    )
    receipt = UiDescriptorOperator().apply(
        UiProjectionRepairRequest(
            semantic_ir=document,
            surface=ProjectionSurface.WEB,
            role=OperatorRole.UI_DESCRIPTOR,
            current_projection=current,
            live_traces=_traces_for(document),
        )
    )
    assert receipt.disposition is RepairDisposition.ABSTAIN
    assert "conflict_policy_abstain" in receipt.reason_codes
    assert receipt.preview_projection is None


def test_missing_target_projection_abstains() -> None:
    document = _document()
    receipt = UiDescriptorOperator().apply(
        UiProjectionRepairRequest(
            semantic_ir=document,
            surface=ProjectionSurface.DESKTOP,
            role=OperatorRole.UI_DESCRIPTOR,
            current_projection=None,
            live_traces=_traces_for(document),
        )
    )
    assert receipt.disposition is RepairDisposition.ABSTAIN
    assert "missing_target_projection" in receipt.reason_codes


def test_non_full_ui_ux_ir_semantic_source_abstains() -> None:
    # Build a production IR first so the projection is well-formed, then lower
    # only the semantic source authority for the repair request.
    production = _document(authority=SourceAuthority.FULL_UI_UX_IR)
    current = _drifted_projection(production, ProjectionSurface.DESKTOP)
    weak = UIIRSemanticDocument(
        document_id=production.document_id,
        title=production.title,
        components=production.components,
        actions=production.actions,
        entry_components=production.entry_components,
        terminal_outcomes=production.terminal_outcomes,
        authority=SourceAuthority.BRIDGE_ONLY,
        schema_cid=production.schema_cid,
        orb_interface_cid=production.orb_interface_cid,
        idl_interface_cid=production.idl_interface_cid,
        source_refs=production.source_refs,
    )
    receipt = UiDescriptorOperator().apply(
        UiProjectionRepairRequest(
            semantic_ir=weak,
            surface=ProjectionSurface.DESKTOP,
            role=OperatorRole.UI_DESCRIPTOR,
            current_projection=current,
            live_traces=_traces_for(production),
        )
    )
    assert receipt.disposition is RepairDisposition.ABSTAIN
    assert "semantic_source_not_full_ui_ux_ir" in receipt.reason_codes


def test_live_mediation_failure_rejects_preview() -> None:
    document = _document()
    current = _drifted_projection(document, ProjectionSurface.DESKTOP)
    bad_trace = LiveActionTrace(
        action_id=document.actions[0].action_id,
        mcp_method=document.actions[0].mcp_method,
        effect_id="effect:other",
        mediation_path=MediationPathClass.GOVERNED_MEDIATOR,
        terminal_state="passed",
        receipt_cid=_cid("receipt:mismatch"),
    )
    receipt = UiDescriptorOperator().apply(
        UiProjectionRepairRequest(
            semantic_ir=document,
            surface=ProjectionSurface.DESKTOP,
            role=OperatorRole.UI_DESCRIPTOR,
            current_projection=current,
            live_traces=(bad_trace,),
        )
    )
    assert receipt.disposition is RepairDisposition.REJECTED
    assert "live_mediation_failed" in receipt.reason_codes


def test_orb_and_idl_operators_force_surface() -> None:
    document = _document()
    orb_current = _drifted_projection(document, ProjectionSurface.ORB)
    idl_current = _drifted_projection(document, ProjectionSurface.IDL)
    orb_receipt = OrbBindingOperator().apply(
        UiProjectionRepairRequest(
            semantic_ir=document,
            surface=ProjectionSurface.DESKTOP,  # wrong surface on purpose
            role=OperatorRole.ORB_BINDING,
            current_projection=orb_current,
            live_traces=_traces_for(document),
        )
    )
    # Surface forced to ORB; current projection surface is ORB so apply proceeds.
    assert orb_receipt.surface is ProjectionSurface.ORB
    assert orb_receipt.disposition is RepairDisposition.PREVIEW_READY

    idl_receipt = IdlProjectionOperator().apply(
        UiProjectionRepairRequest(
            semantic_ir=document,
            surface=ProjectionSurface.WEB,
            role=OperatorRole.IDL_PROJECTION,
            current_projection=idl_current,
            live_traces=_traces_for(document),
        )
    )
    assert idl_receipt.surface is ProjectionSurface.IDL
    assert idl_receipt.disposition is RepairDisposition.PREVIEW_READY


def test_facade_repair_all_surfaces() -> None:
    document = _document()
    projections = {
        surface.value: _drifted_projection(document, surface)
        for surface in ProjectionSurface
    }
    ops = build_ui_projection_repair_operators()
    receipts = ops.repair_all_surfaces(
        document, projections, live_traces=_traces_for(document)
    )
    assert len(receipts) == len(ProjectionSurface)
    assert all(item.disposition is RepairDisposition.PREVIEW_READY for item in receipts)
    assert all(item.semantic_roundtrip_ok for item in receipts)
    assert all(item.live_mediation_ok for item in receipts)
    # Every edited preview roundtrips independently.
    for receipt in receipts:
        assert receipt.preview_projection is not None
        restored = semantic_ir_from_projection(
            receipt.preview_projection,
            title=document.title,
            entry_components=document.entry_components,
            terminal_outcomes=document.terminal_outcomes,
            authority=document.authority,
        )
        assert restored.semantic_core_digest() == document.semantic_core_digest()


def test_receipt_roundtrip_and_authority_flags_sealed() -> None:
    document = _document()
    current = _drifted_projection(document, ProjectionSurface.MOBILE)
    receipt = MobileProjectionOperator().apply(
        UiProjectionRepairRequest(
            semantic_ir=document,
            surface=ProjectionSurface.MOBILE,
            role=OperatorRole.MOBILE_PROJECTION,
            current_projection=current,
            live_traces=_traces_for(document),
        )
    )
    restored = UiProjectionRepairReceipt.from_dict(receipt.to_dict())
    assert restored.content_id == receipt.content_id
    assert restored.disposition is receipt.disposition
    with pytest.raises(UiProjectionRepairError, match="proposal-only"):
        replace(receipt, proposal_only=False)
    with pytest.raises(UiProjectionRepairError, match="grants_write_authority"):
        replace(receipt, grants_write_authority=True)


def test_forbidden_payload_fields_fail_closed() -> None:
    with pytest.raises(UiProjectionRepairError, match="forbidden fields"):
        UiActionBinding.from_dict(
            {
                "action_id": "action:x",
                "label": "X",
                "mcp_method": "m",
                "interface_cid": _cid("i"),
                "effect_id": "effect:x",
                "source_body": "print('nope')",
            }
        )
    with pytest.raises(UiProjectionRepairError, match="forbidden fields"):
        UIIRSemanticDocument.from_dict(
            {
                "document_id": "uiir:x",
                "title": "X",
                "components": [],
                "actions": [],
                "entry_components": [],
                "terminal_outcomes": [],
                "llm_prompt": "invent UI",
            }
        )


def test_artifact_projection_is_content_addressed_and_non_authoritative() -> None:
    artifact = materialize_ui_projection_operator_vectors()
    assert artifact["interface"] == UI_PROJECTION_REPAIR_OPERATORS_INTERFACE
    assert artifact["evidence_id"] == UI_REPAIR_EVIDENCE
    assert artifact["operator_kind"] == OperatorKind.REPAIR_UI_PROJECTION.value
    assert artifact["proposal_only"] is True
    assert artifact["grants_write_authority"] is False
    assert artifact["artifact_digest"].startswith("sha256:")
    assert set(artifact["surfaces"]) == {item.value for item in ProjectionSurface}
    assert set(artifact["roles"]) == {item.value for item in OperatorRole}
    # Deterministic re-materialization.
    assert materialize_ui_projection_operator_vectors()["artifact_digest"] == (
        artifact["artifact_digest"]
    )


def test_request_from_dict_and_facade_apply() -> None:
    document = _document()
    current = _drifted_projection(document, ProjectionSurface.CLI)
    payload = {
        "semantic_ir": document.to_dict(),
        "surface": "cli",
        "role": "surface_sync",
        "current_projection": current.to_dict(),
        "live_traces": [item.to_dict() for item in _traces_for(document)],
        "require_live_mediation": True,
    }
    request = UiProjectionRepairRequest.from_dict(payload)
    receipt = UIProjectionRepairOperators().apply(request)
    assert receipt.disposition is RepairDisposition.PREVIEW_READY
    assert receipt.role is OperatorRole.UI_DESCRIPTOR
    assert receipt.surface is ProjectionSurface.CLI
