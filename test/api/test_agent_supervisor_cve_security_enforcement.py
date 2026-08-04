from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.cve_security_gate import (
    CVESecurityGateOutcome,
    CVESecurityGateResult,
    SecurityFactStream,
    SecurityMappedDecision,
    SecurityRequestContext,
    SecurityRequestMapping,
    SecurityRequestMappingStatus,
)
from ipfs_accelerate_py.agent_supervisor.control.execution_permit import (
    ExecutionAttempt,
    PermitIssuanceError,
    PermitVerificationCode,
    PermitVerificationError,
    issue_cve_execution_permit,
    issue_execution_permit,
    verify_cve_execution_permit,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_constraint_compiler import (
    AdmissionAuthority,
    AdmissionRejectionCode,
    CVESecurityEnforcementEvidence,
    CVESecurityEnforcementStage,
    PlanAdmissionReceipt,
    ValidationResult,
    _plain as _admission_plain,
    compile_cve_merge_admission,
    compile_cve_plan_admission,
    compile_cve_post_generation_validation,
    compile_cve_pre_execution_admission,
    revalidate_cve_merged_tree,
)
from ipfs_accelerate_py.agent_supervisor.proof.security_constraint_adapter import (
    evaluate_security_authorization,
)
from test.api.test_agent_supervisor_execution_permit import (
    NOW,
    _fixture as _permit_fixture,
)
from test.api.test_agent_supervisor_ir_constraint_compiler import (
    TREE,
    _request as _base_admission,
)


def _gate_result(
    admission,
    *,
    outcome: CVESecurityGateOutcome = CVESecurityGateOutcome.PASS,
    stale_decision: bool = False,
) -> CVESecurityGateResult:
    request = admission.security_requests[0]
    policy = admission.security_policy
    context = SecurityRequestContext.from_policy(
        policy,
        principal=request.principal,
        tool=request.tool,
        current_state=request.current_state,
        state_version=request.state_version,
        requested_authority=request.requested_authority,
        evaluated_at_ms=request.evaluated_at_ms,
        source_zone=request.source_zone,
        channel=request.channel,
        target_zone=request.target_zone,
        satisfied_assumption_ids=request.satisfied_assumption_ids,
        accepted_claim_result_ids=request.accepted_claim_result_ids,
    )
    intent_mapping = SecurityRequestMapping(
        stream=SecurityFactStream.INTENT,
        source_id="intent:action:write",
        status=SecurityRequestMappingStatus.EXACT,
        request=request,
        evidence_ids=("intent:evidence:write",),
    )
    code_mapping = SecurityRequestMapping(
        stream=SecurityFactStream.CODE,
        source_id="code:tree:write",
        status=SecurityRequestMappingStatus.EXACT,
        request=request,
        evidence_ids=("code:evidence:write",),
    )
    decision = evaluate_security_authorization(policy, request)
    if stale_decision:
        decision = replace(decision, evaluated_at_ms=decision.evaluated_at_ms + 1)
    return CVESecurityGateResult(
        outcome=outcome,
        policy_receipt_id=policy.content_id,
        context=context,
        intent_mappings=(intent_mapping,),
        code_mappings=(code_mapping,),
        decisions=(
            SecurityMappedDecision(
                intent_mapping.mapping_id,
                SecurityFactStream.INTENT,
                decision,
            ),
            SecurityMappedDecision(
                code_mapping.mapping_id,
                SecurityFactStream.CODE,
                decision,
            ),
        ),
    )


def _gated(
    admission,
    terminal_stage: CVESecurityEnforcementStage,
    *,
    outcome: CVESecurityGateOutcome = CVESecurityGateOutcome.PASS,
    stale_decision: bool = False,
):
    terminal_index = tuple(CVESecurityEnforcementStage).index(terminal_stage)
    gate = _gate_result(
        admission,
        outcome=outcome,
        stale_decision=stale_decision,
    )
    evidence = []
    parent_id = ""
    for stage in tuple(CVESecurityEnforcementStage)[: terminal_index + 1]:
        item = CVESecurityEnforcementEvidence(
            stage=stage,
            repository_tree_id=admission.repository_tree_id,
            gate_result=gate,
            parent_evidence_id=parent_id,
            expires_at_ms=gate.context.evaluated_at_ms + 60_000,
        )
        evidence.append(item)
        parent_id = item.evidence_id
    return replace(
        admission,
        cve_security_evidence=tuple(evidence),
        required_cve_security_stage=terminal_stage,
    )


@pytest.mark.parametrize(
    ("security_decision", "outcome", "expected_code"),
    (
        (
            "deny",
            CVESecurityGateOutcome.REJECT,
            AdmissionRejectionCode.SECURITY_DENY,
        ),
        (
            "conflict",
            CVESecurityGateOutcome.REJECT,
            AdmissionRejectionCode.SECURITY_CONFLICT,
        ),
        (
            "unknown",
            CVESecurityGateOutcome.UNKNOWN,
            AdmissionRejectionCode.SECURITY_UNKNOWN,
        ),
    ),
)
def test_plan_admission_rejects_deny_conflict_unknown_and_stale_gate_evidence(
    security_decision: str,
    outcome: CVESecurityGateOutcome,
    expected_code: AdmissionRejectionCode,
) -> None:
    request = _gated(
        _base_admission(security_decision=security_decision),
        CVESecurityEnforcementStage.PLAN_ADMISSION,
        outcome=outcome,
    )

    receipt = compile_cve_plan_admission(request)

    assert not receipt.admitted
    assert (
        AdmissionRejectionCode.CVE_SECURITY_GATE_REJECTED.value
        in receipt.reason_codes
    )
    assert expected_code.value in receipt.reason_codes

    stale = _gated(
        _base_admission(),
        CVESecurityEnforcementStage.PLAN_ADMISSION,
        stale_decision=True,
    )
    stale_receipt = compile_cve_plan_admission(stale)
    assert not stale_receipt.admitted
    assert (
        AdmissionRejectionCode.CVE_SECURITY_GATE_STALE.value
        in stale_receipt.reason_codes
    )


def test_allow_still_requires_existing_authority_and_declared_generated_effects(
) -> None:
    request = _gated(
        _base_admission(),
        CVESecurityEnforcementStage.POST_GENERATION,
    )
    no_authority = replace(
        request,
        authority=AdmissionAuthority(
            principal="principal:worker",
            requested_authority="mutation",
            grant_principal="principal:other",
            granted_authorities=("mutation",),
            grant_source_ids=("security-grant:fixture",),
        ),
    )
    receipt = compile_cve_post_generation_validation(no_authority)
    assert not receipt.admitted
    assert AdmissionRejectionCode.AUTHORITY_MISMATCH.value in receipt.reason_codes

    broadened_plan = _admission_plain(request.candidate_plan)
    changed_effect = {
        "effect_id": "effect:write",
        "action_id": "action:write",
        "operation": "delete",
        "target": "src/undeclared.py",
    }
    broadened_plan["effects"] = [changed_effect]
    broadened_plan["actions"][0]["effects"] = [changed_effect]
    broadened = replace(request, candidate_plan=broadened_plan)
    receipt = compile_cve_post_generation_validation(broadened)
    assert not receipt.admitted
    assert AdmissionRejectionCode.UNDECLARED_EFFECT.value in receipt.reason_codes


def test_every_cve_runtime_stage_requires_an_unbroken_tree_bound_gate_chain(
) -> None:
    base = _base_admission()
    stage_compilers = (
        (
            CVESecurityEnforcementStage.PLAN_ADMISSION,
            compile_cve_plan_admission,
        ),
        (
            CVESecurityEnforcementStage.PRE_EXECUTION,
            compile_cve_pre_execution_admission,
        ),
        (
            CVESecurityEnforcementStage.POST_GENERATION,
            compile_cve_post_generation_validation,
        ),
        (
            CVESecurityEnforcementStage.MERGE_ADMISSION,
            compile_cve_merge_admission,
        ),
        (
            CVESecurityEnforcementStage.MERGED_TREE_REVALIDATION,
            revalidate_cve_merged_tree,
        ),
    )
    for stage, compiler in stage_compilers:
        receipt = compiler(_gated(base, stage))
        assert receipt.admitted
        assert len(receipt.cve_security_evidence_ids) == (
            tuple(CVESecurityEnforcementStage).index(stage) + 1
        )
        assert PlanAdmissionReceipt.from_dict(receipt.to_dict()) == receipt

    merge = _gated(base, CVESecurityEnforcementStage.MERGE_ADMISSION)
    skipped = replace(
        merge,
        cve_security_evidence=(
            merge.cve_security_evidence[0],
            merge.cve_security_evidence[-1],
        ),
    )
    skipped_receipt = compile_cve_merge_admission(skipped)
    assert not skipped_receipt.admitted
    assert (
        AdmissionRejectionCode.CVE_SECURITY_GATE_MISSING.value
        in skipped_receipt.reason_codes
    )

    merged = _gated(
        base, CVESecurityEnforcementStage.MERGED_TREE_REVALIDATION
    )
    drifted = replace(
        merged,
        repository_tree_id="tree:merged-drift",
        validation_results=tuple(
            ValidationResult(
                item.requirement_id,
                item.status,
                "tree:merged-drift",
                evidence_id=item.evidence_id,
                reason_codes=item.reason_codes,
            )
            for item in merged.validation_results
        ),
    )
    drifted_receipt = revalidate_cve_merged_tree(drifted)
    assert not drifted_receipt.admitted
    assert (
        AdmissionRejectionCode.CVE_SECURITY_GATE_STALE.value
        in drifted_receipt.reason_codes
    )


def test_no_cve_permit_path_bypasses_pre_execution_gate_or_root_recheck(
) -> None:
    admission, _, witness = _permit_fixture()
    plan_only = _gated(
        admission, CVESecurityEnforcementStage.PLAN_ADMISSION
    )
    plan_receipt = compile_cve_plan_admission(plan_only)
    assert plan_receipt.admitted
    with pytest.raises(PermitIssuanceError, match="pre-execution"):
        issue_execution_permit(
            plan_only,
            plan_receipt,
            witness,
            caller="agent-supervisor:implementation-daemon",
            policy_id="policy:implementation-daemon",
            policy_revision="sha256:policy-v1",
            issued_at_ms=NOW,
            expires_at_ms=NOW + 30_000,
        )

    pre_execution = _gated(
        admission, CVESecurityEnforcementStage.PRE_EXECUTION
    )
    pre_receipt = compile_cve_pre_execution_admission(pre_execution)
    permit = issue_cve_execution_permit(
        pre_execution,
        pre_receipt,
        witness,
        caller="agent-supervisor:implementation-daemon",
        policy_id="policy:implementation-daemon",
        policy_revision="sha256:policy-v1",
        issued_at_ms=NOW,
        expires_at_ms=NOW + 30_000,
    )
    assert {
        item.domain for item in permit.evidence_receipts
        if item.domain.startswith("cve_security_gate:")
    } == {
        "cve_security_gate:plan_admission",
        "cve_security_gate:pre_execution",
    }

    valid_attempt = ExecutionAttempt.from_permit(permit, now_ms=NOW + 1)
    use = verify_cve_execution_permit(
        permit,
        valid_attempt,
        trusted_permit_ids=(permit.permit_id,),
    )
    assert use.authorizes_effect

    drifted_attempt = ExecutionAttempt.from_permit(
        permit,
        now_ms=NOW + 2,
        repository_tree_id="tree:changed-after-gate",
    )
    with pytest.raises(PermitVerificationError) as caught:
        verify_cve_execution_permit(
            permit,
            drifted_attempt,
            trusted_permit_ids=(permit.permit_id,),
        )
    assert caught.value.code is PermitVerificationCode.STALE_ROOT
