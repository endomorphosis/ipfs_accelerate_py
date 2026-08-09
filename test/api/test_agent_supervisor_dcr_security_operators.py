"""DCR-046: authorization, effect, and policy repair operators.

Acceptance:
* Policy outage / missing decision denies.
* Stale / revoked / wrong-audience grants fail.
* No server-supplied authorization assertion is trusted.
* Operators restore reviewed bindings only; inventing authority/policy/UCAN/
  effect classifications abstains for review.
* Operators remain proposal-only and never grant write/proof/semantic authority.
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorFamily,
    OperatorKind,
    build_default_operator_registry,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.security_repairs import (
    MCP_PROFILE_C,
    MCP_PROFILE_D,
    MCP_PROFILES_C_D,
    SECURITY_IR_INTERFACE,
    SECURITY_REPAIR_EVIDENCE,
    SECURITY_REPAIR_OPERATORS_INTERFACE,
    AuthorizationBinding,
    AuthorizationBindingOperator,
    AuthoritySource,
    DenialReasonCode,
    EffectAnnotation,
    EffectAnnotationOperator,
    EffectClass,
    GrantStatus,
    OperatorRole,
    PolicyAvailability,
    PolicyGate,
    PolicyGateOperator,
    RepairDisposition,
    SecurityAuthorizationRequest,
    SecurityGrant,
    SecurityIR,
    SecurityRepairError,
    SecurityRepairRequest,
    SecurityVerdict,
    build_security_repair_operators,
    evaluate_security_authorization,
    materialize_security_operator_vectors,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


NOW = 1_000_000
ISSUER = "did:key:issuer-root"
AUDIENCE = "did:key:worker-1"
CAPABILITY = "tools/call"
RESOURCE = "mcp://logic_tools/cec_prove"
ACTION = "invoke"
EFFECT_ID = "effect:dispatch:cec_prove"
GRANT_ID = "grant:root-worker-1"
BINDING_ID = "binding:worker-1"
GATE_ID = "gate:profile-d-primary"
POLICY_ID = "policy:reviewed-dcr046"


def _cid(label: str) -> str:
    return content_identity({"dcr046": label})


def _grant(
    *,
    grant_id: str = GRANT_ID,
    issuer: str = ISSUER,
    audience: str = AUDIENCE,
    capability: str = CAPABILITY,
    resource: str = RESOURCE,
    not_before_ms: int = 0,
    expires_at_ms: int | None = 2_000_000,
    revoked: bool = False,
    revocation_id: str = "",
    proof_cid: str = "",
    obligations: tuple[str, ...] = (),
    effect_ids: tuple[str, ...] = (EFFECT_ID,),
) -> SecurityGrant:
    return SecurityGrant(
        grant_id=grant_id,
        issuer=issuer,
        audience=audience,
        capability=capability,
        resource=resource,
        not_before_ms=not_before_ms,
        expires_at_ms=expires_at_ms,
        revoked=revoked,
        revocation_id=revocation_id,
        proof_cid=proof_cid or _cid(f"proof:{grant_id}"),
        obligations=obligations,
        effect_ids=effect_ids,
        authority=AuthoritySource.REVIEWED,
    )


def _binding(
    *,
    grant: SecurityGrant | None = None,
    binding_id: str = BINDING_ID,
) -> AuthorizationBinding:
    grant = grant or _grant()
    return AuthorizationBinding(
        binding_id=binding_id,
        principal=grant.audience,
        audience=grant.audience,
        capability=grant.capability,
        resource=grant.resource,
        grant_id=grant.grant_id,
        effect_ids=grant.effect_ids,
        authority=AuthoritySource.REVIEWED,
        profile=MCP_PROFILE_C,
    )


def _effect(
    *,
    effect_id: str = EFFECT_ID,
    action: str = ACTION,
    resource: str = RESOURCE,
    effect_class: EffectClass = EffectClass.DISPATCH,
) -> EffectAnnotation:
    return EffectAnnotation(
        effect_id=effect_id,
        action=action,
        resource=resource,
        effect_class=effect_class,
        declared=True,
        authority=AuthoritySource.REVIEWED,
    )


def _gate(
    *,
    gate_id: str = GATE_ID,
    policy_id: str = POLICY_ID,
    availability: PolicyAvailability = PolicyAvailability.AVAILABLE,
    decision: str = "allow",
    obligations: tuple[str, ...] = (),
    justification: str = "reviewed allow",
) -> PolicyGate:
    return PolicyGate(
        gate_id=gate_id,
        policy_id=policy_id,
        availability=availability,
        decision=decision,
        obligations=obligations,
        justification=justification,
        authority=AuthoritySource.REVIEWED,
        profile=MCP_PROFILE_D,
    )


def _security_ir(
    *,
    grants: tuple[SecurityGrant, ...] | None = None,
    bindings: tuple[AuthorizationBinding, ...] | None = None,
    effects: tuple[EffectAnnotation, ...] | None = None,
    policy_gates: tuple[PolicyGate, ...] | None = None,
    revoked_proof_cids: tuple[str, ...] = (),
    authority: AuthoritySource = AuthoritySource.REVIEWED,
) -> SecurityIR:
    grant = (grants or (_grant(),))[0]
    return SecurityIR(
        document_id="security-ir:dcr046:demo",
        trusted_issuers=(ISSUER,),
        grants=grants if grants is not None else (grant,),
        bindings=bindings if bindings is not None else (_binding(grant=grant),),
        effects=effects if effects is not None else (_effect(),),
        policy_gates=policy_gates if policy_gates is not None else (_gate(),),
        revoked_proof_cids=revoked_proof_cids,
        authority=authority,
        profiles=MCP_PROFILES_C_D,
        source_refs=("source:reviewed-security-ir",),
    )


def _request(
    *,
    principal: str = AUDIENCE,
    audience: str = AUDIENCE,
    capability: str = CAPABILITY,
    resource: str = RESOURCE,
    action: str = ACTION,
    expected_effects: tuple[str, ...] = (EFFECT_ID,),
    evaluated_at_ms: int = NOW,
    fulfilled_obligations: tuple[str, ...] = (),
    server_assertions: dict | None = None,
    grant_id: str = GRANT_ID,
    policy_gate_id: str = GATE_ID,
) -> SecurityAuthorizationRequest:
    return SecurityAuthorizationRequest(
        principal=principal,
        audience=audience,
        capability=capability,
        resource=resource,
        action=action,
        expected_effects=expected_effects,
        evaluated_at_ms=evaluated_at_ms,
        fulfilled_obligations=fulfilled_obligations,
        server_assertions=server_assertions or {},
        grant_id=grant_id,
        policy_gate_id=policy_gate_id,
    )


# ---------------------------------------------------------------------------
# Interface / registry binding
# ---------------------------------------------------------------------------


def test_interfaces_and_evidence_are_declared() -> None:
    assert SECURITY_REPAIR_OPERATORS_INTERFACE == "SecurityRepairOperators@1"
    assert SECURITY_REPAIR_EVIDENCE == "dcr/safety-repair@1"
    assert SECURITY_IR_INTERFACE == "SecurityIR"
    assert MCP_PROFILE_C == "mcpplusplus/profile-c-ucan/v1"
    assert MCP_PROFILE_D == "mcpplusplus/profile-d-policy/v1"
    ops = build_security_repair_operators()
    assert ops.INTERFACE == SECURITY_REPAIR_OPERATORS_INTERFACE
    assert ops.EVIDENCE_ID == SECURITY_REPAIR_EVIDENCE
    assert isinstance(ops.authorization_binding, AuthorizationBindingOperator)
    assert isinstance(ops.effect_annotation, EffectAnnotationOperator)
    assert isinstance(ops.policy_gate, PolicyGateOperator)


def test_registry_binds_repair_authorization_guard_to_security_family() -> None:
    reg = build_default_operator_registry()
    descriptor = reg.require_known(OperatorKind.REPAIR_AUTHORIZATION_GUARD)
    assert descriptor.family is OperatorFamily.SECURITY
    assert descriptor.kind is OperatorKind.REPAIR_AUTHORIZATION_GUARD
    assert descriptor.proposal_only is True
    assert descriptor.grants_write_authority is False
    assert descriptor.grants_proof_authority is False
    assert descriptor.semantic_authority is False
    assert descriptor.allows_source_generation is False
    assert "scope:closed_authorization_guard" in descriptor.write_scope
    assert reg.get("authorization_guard").kind is OperatorKind.REPAIR_AUTHORIZATION_GUARD
    assert reg.get("confirmation_check").kind is OperatorKind.REPAIR_AUTHORIZATION_GUARD


# ---------------------------------------------------------------------------
# Execution-time fail-closed evaluation (acceptance core)
# ---------------------------------------------------------------------------


def test_positive_reviewed_grant_and_policy_permit() -> None:
    security_ir = _security_ir()
    decision = evaluate_security_authorization(security_ir, _request())
    assert decision.verdict is SecurityVerdict.PERMIT
    assert decision.reason is DenialReasonCode.ALLOWED
    assert decision.permitted is True
    assert GRANT_ID in decision.matched_grant_ids
    assert decision.establishes_generated_code_correctness is False
    assert "execution_time_check" in decision.reason_codes


def test_policy_outage_denies() -> None:
    security_ir = _security_ir(
        policy_gates=(_gate(availability=PolicyAvailability.OUTAGE, decision=""),)
    )
    decision = evaluate_security_authorization(security_ir, _request())
    assert decision.verdict is SecurityVerdict.DENY
    assert decision.reason is DenialReasonCode.POLICY_OUTAGE
    assert decision.permitted is False
    assert "policy_outage" in decision.reason_codes


def test_missing_decision_denies() -> None:
    security_ir = _security_ir(
        policy_gates=(
            _gate(availability=PolicyAvailability.AVAILABLE, decision=""),
        )
    )
    decision = evaluate_security_authorization(security_ir, _request())
    assert decision.verdict is SecurityVerdict.DENY
    assert decision.reason is DenialReasonCode.MISSING_DECISION
    assert "missing_decision" in decision.reason_codes


def test_missing_policy_gate_denies() -> None:
    security_ir = _security_ir(policy_gates=())
    decision = evaluate_security_authorization(security_ir, _request(policy_gate_id=""))
    assert decision.verdict is SecurityVerdict.DENY
    assert decision.reason is DenialReasonCode.MISSING_DECISION


def test_stale_expired_grant_fails() -> None:
    security_ir = _security_ir(grants=(_grant(expires_at_ms=NOW - 1),))
    decision = evaluate_security_authorization(security_ir, _request())
    assert decision.verdict is SecurityVerdict.DENY
    assert decision.reason is DenialReasonCode.STALE_GRANT
    assert "stale_grant" in decision.reason_codes
    assert "expired" in decision.reason_codes


def test_stale_not_yet_valid_grant_fails() -> None:
    security_ir = _security_ir(grants=(_grant(not_before_ms=NOW + 10_000),))
    decision = evaluate_security_authorization(security_ir, _request())
    assert decision.verdict is SecurityVerdict.DENY
    assert decision.reason is DenialReasonCode.STALE_GRANT
    assert "not_yet_valid" in decision.reason_codes


def test_revoked_grant_fails() -> None:
    security_ir = _security_ir(
        grants=(_grant(revoked=True, revocation_id="revocation:1"),)
    )
    decision = evaluate_security_authorization(security_ir, _request())
    assert decision.verdict is SecurityVerdict.DENY
    assert decision.reason is DenialReasonCode.REVOKED
    assert "revoked" in decision.reason_codes


def test_revoked_proof_cid_fails() -> None:
    proof = _cid("proof:revoked-feed")
    security_ir = _security_ir(
        grants=(_grant(proof_cid=proof),),
        revoked_proof_cids=(proof,),
    )
    decision = evaluate_security_authorization(security_ir, _request())
    assert decision.verdict is SecurityVerdict.DENY
    assert decision.reason is DenialReasonCode.REVOKED


def test_wrong_audience_grant_fails() -> None:
    security_ir = _security_ir(
        grants=(_grant(audience="did:key:other-worker"),),
        bindings=(),
    )
    decision = evaluate_security_authorization(
        security_ir,
        _request(grant_id=GRANT_ID),
    )
    assert decision.verdict is SecurityVerdict.DENY
    assert decision.reason is DenialReasonCode.WRONG_AUDIENCE
    assert "wrong_audience" in decision.reason_codes


def test_server_supplied_authorization_assertion_is_not_trusted() -> None:
    # No applicable grant — only a server claim that the caller is authorized.
    security_ir = _security_ir(grants=(), bindings=())
    decision = evaluate_security_authorization(
        security_ir,
        _request(
            grant_id="",
            server_assertions={
                "authorized": True,
                "authorization_assertion": "permit",
                "server_asserted_permit": True,
                "force_allow": True,
            },
        ),
    )
    assert decision.verdict is SecurityVerdict.DENY
    assert decision.ignored_server_assertions is True
    assert decision.reason is DenialReasonCode.SERVER_ASSERTION_UNTRUSTED
    assert "server_assertion_untrusted" in decision.reason_codes


def test_server_assertion_cannot_override_revocation() -> None:
    security_ir = _security_ir(grants=(_grant(revoked=True, revocation_id="rev:9"),))
    decision = evaluate_security_authorization(
        security_ir,
        _request(
            server_assertions={
                "is_authorized": True,
                "pre_authorized": True,
                "trusted_decision": "permit",
            }
        ),
    )
    assert decision.verdict is SecurityVerdict.DENY
    assert decision.reason is DenialReasonCode.REVOKED
    assert decision.ignored_server_assertions is True
    assert "server_assertion_untrusted" in decision.reason_codes


def test_missing_security_ir_denies() -> None:
    decision = evaluate_security_authorization(None, _request())
    assert decision.verdict is SecurityVerdict.DENY
    assert decision.reason is DenialReasonCode.MISSING_SECURITY_IR


def test_grant_status_matrix() -> None:
    grant = _grant(audience=AUDIENCE, expires_at_ms=NOW + 100, not_before_ms=NOW - 100)
    assert grant.status_at(NOW, expected_audience=AUDIENCE) is GrantStatus.ACTIVE
    assert (
        grant.status_at(NOW, expected_audience="did:key:other")
        is GrantStatus.WRONG_AUDIENCE
    )
    revoked = _grant(revoked=True)
    assert revoked.status_at(NOW, expected_audience=AUDIENCE) is GrantStatus.REVOKED
    expired = _grant(expires_at_ms=NOW)
    assert expired.status_at(NOW, expected_audience=AUDIENCE) is GrantStatus.EXPIRED


def test_decision_cannot_claim_generated_code_correctness() -> None:
    security_ir = _security_ir()
    decision = evaluate_security_authorization(security_ir, _request())
    forged = decision.to_dict()
    forged["establishes_generated_code_correctness"] = True
    with pytest.raises(SecurityRepairError, match="generated-code"):
        type(decision).from_dict(forged)


# ---------------------------------------------------------------------------
# Operator apply / abstain / preview / inverse
# ---------------------------------------------------------------------------


def test_authorization_binding_operator_restores_reviewed_binding() -> None:
    grant = _grant()
    reviewed = _binding(grant=grant)
    # Drifted binding (wrong capability label) currently present.
    drifted = AuthorizationBinding(
        binding_id=BINDING_ID,
        principal=AUDIENCE,
        audience=AUDIENCE,
        capability="tools/list",
        resource=RESOURCE,
        grant_id=GRANT_ID,
        effect_ids=(EFFECT_ID,),
        authority=AuthoritySource.REVIEWED,
    )
    security_ir = _security_ir(grants=(grant,), bindings=(drifted,))
    request = SecurityRepairRequest(
        security_ir=security_ir,
        role=OperatorRole.AUTHORIZATION_BINDING,
        reviewed_binding=reviewed,
        current_binding=drifted,
        authorization_request=_request(),
        require_execution_check=True,
    )
    receipt = AuthorizationBindingOperator().apply(request)
    assert receipt.disposition is RepairDisposition.PREVIEW_READY
    assert receipt.proposal_only is True
    assert receipt.grants_write_authority is False
    assert receipt.grants_proof_authority is False
    assert receipt.semantic_authority is False
    assert receipt.evidence_id == SECURITY_REPAIR_EVIDENCE
    assert receipt.operator_kind == OperatorKind.REPAIR_AUTHORIZATION_GUARD.value
    assert receipt.preview_binding is not None
    assert receipt.preview_binding.content_id == reviewed.content_id
    assert receipt.execution_check_ok is True
    inverse = AuthorizationBindingOperator().inverse(receipt)
    assert inverse is not None
    assert inverse.content_id == drifted.content_id


def test_authorization_binding_already_aligned_is_idempotent() -> None:
    security_ir = _security_ir()
    binding = security_ir.bindings[0]
    receipt = AuthorizationBindingOperator().apply(
        SecurityRepairRequest(
            security_ir=security_ir,
            role=OperatorRole.AUTHORIZATION_BINDING,
            reviewed_binding=binding,
            current_binding=binding,
            authorization_request=_request(),
        )
    )
    assert receipt.disposition is RepairDisposition.ALREADY_ALIGNED
    assert receipt.execution_check_ok is True


def test_authorization_binding_abstains_without_reviewed_binding() -> None:
    security_ir = _security_ir()
    receipt = AuthorizationBindingOperator().apply(
        SecurityRepairRequest(
            security_ir=security_ir,
            role=OperatorRole.AUTHORIZATION_BINDING,
            reviewed_binding=None,
            current_binding=None,
            authorization_request=_request(),
            require_execution_check=False,
        )
    )
    assert receipt.disposition is RepairDisposition.ABSTAIN
    assert "missing_reviewed_binding" in receipt.reason_codes
    assert "conflict_policy_abstain" in receipt.reason_codes


def test_authorization_binding_abstains_when_grant_would_be_invented() -> None:
    security_ir = _security_ir(grants=())
    reviewed = _binding()
    receipt = AuthorizationBindingOperator().apply(
        SecurityRepairRequest(
            security_ir=security_ir,
            role=OperatorRole.AUTHORIZATION_BINDING,
            reviewed_binding=reviewed,
            current_binding=None,
            require_execution_check=False,
        )
    )
    assert receipt.disposition is RepairDisposition.ABSTAIN
    assert "binding_grant_missing" in receipt.reason_codes
    assert DenialReasonCode.INVENTED_GRANT.value in receipt.reason_codes


def test_effect_annotation_operator_restores_reviewed_effect() -> None:
    security_ir = _security_ir(effects=())
    reviewed = _effect()
    receipt = EffectAnnotationOperator().apply(
        SecurityRepairRequest(
            security_ir=security_ir,
            role=OperatorRole.EFFECT_ANNOTATION,
            reviewed_effect=reviewed,
            current_effect=None,
            authorization_request=_request(),
        )
    )
    assert receipt.disposition is RepairDisposition.PREVIEW_READY
    assert receipt.preview_effect is not None
    assert receipt.preview_effect.effect_id == EFFECT_ID
    assert receipt.execution_check_ok is True
    assert receipt.preview_security_ir is not None
    assert receipt.preview_security_ir.effect_by_id(EFFECT_ID) is not None


def test_effect_annotation_abstains_when_effect_would_be_invented() -> None:
    security_ir = _security_ir()
    receipt = EffectAnnotationOperator().apply(
        SecurityRepairRequest(
            security_ir=security_ir,
            role=OperatorRole.EFFECT_ANNOTATION,
            reviewed_effect=None,
            require_execution_check=False,
        )
    )
    assert receipt.disposition is RepairDisposition.ABSTAIN
    assert DenialReasonCode.INVENTED_EFFECT.value in receipt.reason_codes


def test_policy_gate_operator_restores_fail_closed_gate() -> None:
    # Current gate is in outage; reviewed gate is available allow.
    outage = _gate(availability=PolicyAvailability.OUTAGE, decision="")
    reviewed = _gate(availability=PolicyAvailability.AVAILABLE, decision="allow")
    security_ir = _security_ir(policy_gates=(outage,))
    # Evaluation against current IR must deny (outage).
    assert (
        evaluate_security_authorization(security_ir, _request()).reason
        is DenialReasonCode.POLICY_OUTAGE
    )
    receipt = PolicyGateOperator().apply(
        SecurityRepairRequest(
            security_ir=security_ir,
            role=OperatorRole.POLICY_GATE,
            reviewed_policy_gate=reviewed,
            current_policy_gate=outage,
            authorization_request=_request(),
        )
    )
    assert receipt.disposition is RepairDisposition.PREVIEW_READY
    assert receipt.preview_policy_gate is not None
    assert receipt.preview_policy_gate.availability is PolicyAvailability.AVAILABLE
    assert receipt.execution_check_ok is True
    # Preview IR permits after restoration.
    assert receipt.preview_security_ir is not None
    restored = evaluate_security_authorization(
        receipt.preview_security_ir, _request()
    )
    assert restored.permitted is True
    inverse = PolicyGateOperator().inverse(receipt)
    assert inverse is not None
    assert inverse.availability is PolicyAvailability.OUTAGE


def test_policy_gate_operator_rejects_restoring_outage_as_allow() -> None:
    security_ir = _security_ir()
    outage = _gate(availability=PolicyAvailability.OUTAGE, decision="")
    receipt = PolicyGateOperator().apply(
        SecurityRepairRequest(
            security_ir=security_ir,
            role=OperatorRole.POLICY_GATE,
            reviewed_policy_gate=outage,
            require_execution_check=False,
        )
    )
    assert receipt.disposition is RepairDisposition.REJECTED
    assert DenialReasonCode.POLICY_OUTAGE.value in receipt.reason_codes


def test_policy_gate_operator_abstains_without_reviewed_gate() -> None:
    security_ir = _security_ir()
    receipt = PolicyGateOperator().apply(
        SecurityRepairRequest(
            security_ir=security_ir,
            role=OperatorRole.POLICY_GATE,
            reviewed_policy_gate=None,
            require_execution_check=False,
        )
    )
    assert receipt.disposition is RepairDisposition.ABSTAIN
    assert DenialReasonCode.INVENTED_POLICY.value in receipt.reason_codes


def test_non_reviewed_security_ir_source_abstains() -> None:
    # Build a valid IR then lower only the document authority for the request.
    production = _security_ir(authority=AuthoritySource.REVIEWED)
    weak = SecurityIR(
        document_id=production.document_id,
        trusted_issuers=production.trusted_issuers,
        grants=production.grants,
        bindings=production.bindings,
        effects=production.effects,
        policy_gates=production.policy_gates,
        revoked_proof_cids=production.revoked_proof_cids,
        authority=AuthoritySource.SERVER_ASSERTED,
        profiles=production.profiles,
        source_refs=production.source_refs,
    )
    receipt = AuthorizationBindingOperator().apply(
        SecurityRepairRequest(
            security_ir=weak,
            role=OperatorRole.AUTHORIZATION_BINDING,
            reviewed_binding=production.bindings[0],
            current_binding=production.bindings[0],
            require_execution_check=False,
        )
    )
    assert receipt.disposition is RepairDisposition.ABSTAIN
    assert "security_source_not_reviewed" in receipt.reason_codes


def test_facade_operators_are_proposal_only() -> None:
    ops = build_security_repair_operators()
    for operator in (
        ops.authorization_binding,
        ops.effect_annotation,
        ops.policy_gate,
    ):
        assert operator.descriptor.proposal_only is True
        assert operator.descriptor.grants_write_authority is False
        assert operator.operator_id.startswith("dcr-operator:")


def test_materialize_security_operator_vectors_evidence_subset() -> None:
    security_ir = _security_ir()
    request = _request()
    vectors = materialize_security_operator_vectors(security_ir, request)
    assert vectors["evidence_id"] == SECURITY_REPAIR_EVIDENCE
    assert vectors["interface"] == SECURITY_REPAIR_OPERATORS_INTERFACE
    assert vectors["security_ir_interface"] == SECURITY_IR_INTERFACE
    assert vectors["profiles"] == list(MCP_PROFILES_C_D)
    assert vectors["principal"] == AUDIENCE
    assert vectors["audience"] == AUDIENCE
    assert vectors["capability"] == CAPABILITY
    assert vectors["server_assertions_trusted"] is False
    assert vectors["decision"]["verdict"] == SecurityVerdict.PERMIT.value
    assert EFFECT_ID in vectors["effects"]
    assert "execution_time_check" in vectors["execution_time_check"]
    assert vectors["temporal_validity"]["evaluated_at_ms"] == NOW


def test_canonical_roundtrip_for_core_contracts() -> None:
    security_ir = _security_ir()
    request = _request()
    decision = evaluate_security_authorization(security_ir, request)
    assert SecurityIR.from_dict(security_ir.to_dict()).content_id == security_ir.content_id
    assert (
        SecurityAuthorizationRequest.from_dict(request.to_dict()).content_id
        == request.content_id
    )
    assert (
        type(decision).from_dict(decision.to_dict()).content_id == decision.content_id
    )
    grant = security_ir.grants[0]
    assert SecurityGrant.from_dict(grant.to_dict()).content_id == grant.content_id


def test_undeclared_effect_denies() -> None:
    security_ir = _security_ir(effects=())
    decision = evaluate_security_authorization(security_ir, _request())
    assert decision.verdict is SecurityVerdict.DENY
    assert decision.reason is DenialReasonCode.UNDECLARED_EFFECT
