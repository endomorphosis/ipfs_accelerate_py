"""Focused fail-closed coverage for DCR-060 planner composition."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.default_planner_factory import (
    DcrPlannerCompositionDisposition,
    DcrPlannerCompositionEvidence,
    DcrPlannerRoots,
    DcrPlannerSelfTestReceipt,
    DcrPlannerServiceBinding,
    DcrPlannerServiceKind,
    DefaultPlannerFactoryError,
    assess_dcr_planner_composition,
    build_default_planner_handles,
)


def _ready_handles():
    return build_default_planner_handles(
        optional_provers=("z3",),
        require_proof_carrying=True,
        which=lambda _name: "/reviewed/z3",
    )


def _roots() -> DcrPlannerRoots:
    return DcrPlannerRoots(
        policy_cid="policy-cid",
        forest_cid="forest-cid",
        graph_cid="graph-cid",
        findings_cid="findings-cid",
        logic_candidate_cid="logic-candidate-cid",
        proof_cache_cid="proof-cache-cid",
        operator_registry_cid="operator-registry-cid",
    )


def _service(kind: DcrPlannerServiceKind, roots: DcrPlannerRoots):
    receipt = DcrPlannerSelfTestReceipt(
        service_kind=kind,
        service_identity="service-" + kind.value,
        root_cids=tuple(roots.to_dict().values()),
        input_cids=(roots.logic_candidate_cid, roots.findings_cid),
    )
    return DcrPlannerServiceBinding(
        service_kind=kind,
        service_interface="Reviewed" + kind.value.title().replace("_", ""),
        service_identity=receipt.service_identity,
        self_test=receipt,
    )


def test_dcr060_rejects_untyped_evidence_without_planner_view():
    result = assess_dcr_planner_composition(_ready_handles(), object())

    assert result.disposition is DcrPlannerCompositionDisposition.DEFER_CAPABILITY
    assert "typed_dcr060_evidence_required" in result.reason_codes
    assert result.planner_handles is None
    assert result.planner_view_cid == ""
    assert result.execution_authorized is False
    assert result.completion_authorized is False


def test_dcr060_malformed_typed_inputs_defer_and_never_call_services():
    roots = _roots()
    evidence = DcrPlannerCompositionEvidence(
        doctor_binding=lambda: None,
        doctor_service=lambda: None,
        logic_candidate=object(),
        stage_gate=object(),
        proof_cache_binding=object(),
        operator_registry=object(),
        roots=roots,
        services=tuple(_service(kind, roots) for kind in DcrPlannerServiceKind),
    )

    result = assess_dcr_planner_composition(_ready_handles(), evidence)

    assert result.disposition is DcrPlannerCompositionDisposition.DEFER_CAPABILITY
    assert "legacy_doctor_binding_cannot_satisfy_dcr050" in result.reason_codes
    assert "typed_dcr030_logic_candidate_required" in result.reason_codes
    assert "integration_pending_dcr050_dcr053_live_evidence" in result.reason_codes
    assert result.planner_view_cid == ""
    assert result.model_call_count == result.provider_call_count == result.network_call_count == 0


def test_dcr060_service_receipts_are_closed_and_non_boolean():
    roots = _roots()
    with pytest.raises(DefaultPlannerFactoryError, match="exactly zero"):
        DcrPlannerSelfTestReceipt(
            service_kind=DcrPlannerServiceKind.CANDIDATE,
            service_identity="candidate",
            root_cids=tuple(roots.to_dict().values()),
            input_cids=(roots.logic_candidate_cid, roots.findings_cid),
            model_call_count=1,
        )
    with pytest.raises(DefaultPlannerFactoryError, match="swallow"):
        DcrPlannerSelfTestReceipt(
            service_kind=DcrPlannerServiceKind.SCHEDULER,
            service_identity="scheduler",
            root_cids=tuple(roots.to_dict().values()),
            input_cids=(roots.logic_candidate_cid, roots.findings_cid),
            swallowed_exception=True,
        )


def test_dcr060_synthetic_service_id_is_deferred_not_promoted():
    roots = _roots()
    receipt = DcrPlannerSelfTestReceipt(
        service_kind=DcrPlannerServiceKind.RECEIPT,
        service_identity="synthetic-receipt-service",
        root_cids=tuple(roots.to_dict().values()),
        input_cids=(roots.logic_candidate_cid, roots.findings_cid),
    )
    evidence = DcrPlannerCompositionEvidence(
        doctor_binding=object(),
        doctor_service=object(),
        logic_candidate=object(),
        stage_gate=object(),
        proof_cache_binding=object(),
        operator_registry=object(),
        roots=roots,
        services=(
            DcrPlannerServiceBinding(
                DcrPlannerServiceKind.RECEIPT,
                "ReviewedReceiptService",
                receipt.service_identity,
                receipt,
            ),
            _service(DcrPlannerServiceKind.CANDIDATE, roots),
            _service(DcrPlannerServiceKind.SCHEDULER, roots),
        ),
    )

    result = assess_dcr_planner_composition(_ready_handles(), evidence)

    assert "deterministic_service_self_test_or_identity_invalid" in result.reason_codes
    assert result.planner_view_cid == ""
