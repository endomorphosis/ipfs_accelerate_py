"""Independent multi-gate plan admission (PDR-027 / PlanAdmissionService@1)."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.plan_admission_service import (
    ADMISSION_STAGE_ORDER,
    PLAN_ADMISSION_SERVICE_INTERFACE,
    PlanAdmissionService,
    PlanAdmissionServiceCode,
    PlanAdmissionServiceError,
    PlanAdmissionServiceReceipt,
    PlanAdmissionServiceRequest,
    PlanAdmissionStage,
    admit_plan_through_service,
    construct_plan_admission_request,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.proof.intent_constraint_adapter import (
    IntentCompilationStatus,
    compile_intent_constraints,
    create_intent_conformance_request,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_constraint_compiler import (
    ActionDomainBinding,
    AdmissionAssumption,
    AdmissionAuthority,
    PlanAdmissionRequest,
    PlanAdmissionVerdict,
    ProgramDependency,
    RootBinding,
    ValidationRequirement,
    ValidationResult,
    ValidationStatus,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_registry import (
    IRFamily,
    IRLoadRequest,
    IRLoadStatus,
    IRRegistry,
    deterministic_ir_fixture,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_adapters import IRAdapterRegistry
from ipfs_accelerate_py.agent_supervisor.proof.legal_constraint_adapter import (
    LegalApplicabilityQuery,
    compile_legal_constraints,
)
from ipfs_accelerate_py.agent_supervisor.proof.security_constraint_adapter import (
    SecurityAuthorizationRequest,
    compile_security_constraints,
)
from ipfs_accelerate_py.agent_supervisor.analysis.semantic_dependency_graph import (
    MandatoryClosure,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_plan_admission import (
    PromptPlanAdmissionCode,
    admit_prompt_plan,
)
from test.api.test_agent_supervisor_prompt_goal_planner import (
    _encoded_proposal,
    _request,
    _scan,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
    parse_prompt_goal_graph,
)


TREE = "tree:plan-admission-service"
SOURCE = ({"source_id": "review:service", "span_id": "section:1"},)
FLOW = {"classification": "source", "direction": "workspace_to_tool"}
LEGAL_SCOPE = {
    "jurisdiction": "US-CA",
    "subject": "source-code",
    "principal": "principal:worker",
    "action": "write",
    "resource": "resource:repository",
    "effect": "file:update",
}


def _normalized(family: IRFamily, **sections: object):
    reference, encoded = deterministic_ir_fixture(family, **sections)
    registry = IRRegistry()
    registry.register_local_artifact(reference, encoded)
    loaded = registry.load(IRLoadRequest(reference=reference, family=family))
    assert loaded.status is IRLoadStatus.VERIFIED
    return IRAdapterRegistry().normalize(loaded).require_artifact()


def _candidate(action_ids: tuple[str, ...] = ("action:write",)) -> dict[str, object]:
    actions = []
    effects = []
    for index, action_id in enumerate(action_ids):
        depends_on = (
            ["action:prepare"]
            if action_id == "action:write" and "action:prepare" in action_ids
            else []
        )
        effect = {
            "effect_id": f"effect:{action_id.removeprefix('action:')}",
            "action_id": action_id,
            "operation": "update",
            "target": f"src/{index}.py",
        }
        actions.append(
            {
                "action_id": action_id,
                "principal": "principal:worker",
                "action": "write",
                "tool": "tool:editor",
                "target": "resource:repository",
                "requested_authority": "mutation",
                "depends_on": depends_on,
                "effects": [effect],
                "proof_obligation_ids": (
                    [f"proof:{action_id}"] if action_id == "action:write" else []
                ),
                "validation_requirement_ids": (
                    [f"validation:{action_id}"]
                    if action_id == "action:write"
                    else []
                ),
                "assumption_ids": (
                    ["assumption:tool"] if action_id == "action:write" else []
                ),
            }
        )
        effects.append(effect)
    return {
        "plan_id": "plan:admission-service",
        "actions": actions,
        "effects": effects,
    }


def _intent_request(candidate: dict[str, object]):
    declarations: list[dict[str, object]] = [
        {"id": "goal:admit", "kind": "goal", "grounded": True}
    ]
    actions = candidate["actions"]
    assert isinstance(actions, list)
    for action in actions:
        action_id = str(action["action_id"])
        declarations.append(
            {
                "id": action_id,
                "kind": "action",
                "goal_id": "goal:admit",
                "depends_on": list(action["depends_on"]),
                "grounded": True,
            }
        )
        effect = action["effects"][0]
        declarations.append(
            {
                "id": str(effect["effect_id"]),
                "kind": "effect",
                "action_id": action_id,
                "operation": effect["operation"],
                "target": effect["target"],
                "grounded": True,
            }
        )
    intent = _normalized(IRFamily.INTENT, declarations=tuple(declarations))
    formalization = _normalized(IRFamily.FORMALIZATION)
    compilation = compile_intent_constraints(intent, formalization)
    assert compilation.status is IntentCompilationStatus.COMPILED
    constraints = compilation.require_constraint_set()
    intent_candidate = copy.deepcopy(candidate)
    intent_candidate["intent_root"] = dict(constraints.intent_root)
    intent_candidate["formalization_root"] = dict(constraints.formalization_root)
    intent_candidate["goal_ids"] = ["goal:admit"]
    return create_intent_conformance_request(
        compilation,
        intent_candidate,
        discharged_obligation_ids=tuple(
            item.obligation_id for item in constraints.proof_obligations
        ),
    )


def _legal_result(*, modality: str = "permission", proof_obligation_id: str = ""):
    declaration: dict[str, object] = {
        "declaration_id": f"norm:{modality}",
        "kind": "norm",
        "modality": modality,
        **LEGAL_SCOPE,
        "effective_from_ms": 100,
        "effective_until_ms": 1000,
        "source_references": SOURCE,
    }
    obligations: tuple[dict[str, object], ...] = ()
    if proof_obligation_id:
        declaration["proof_obligation_ids"] = (proof_obligation_id,)
        obligations = (
            {
                "obligation_id": proof_obligation_id,
                "kind": "proof",
                "provision_ids": (f"norm:{modality}",),
                "required": True,
                "discharged": True,
                "source_references": SOURCE,
            },
        )
    artifact = _normalized(
        IRFamily.LEGAL,
        declarations=(declaration,),
        obligations=obligations,
    )
    query = LegalApplicabilityQuery(
        legal_root_artifact_id=artifact.root_artifact_id,
        legal_root_cid_v1=artifact.root_cid_v1,
        legal_root_supervisor_digest=artifact.root_supervisor_digest,
        **LEGAL_SCOPE,
        effective_at_ms=500,
    )
    return compile_legal_constraints(artifact, query)


def _security(candidate: dict[str, object], *, decision: str = "allow"):
    actions = candidate["actions"]
    assert isinstance(actions, list)
    policies = []
    for action in actions:
        effect = action["effects"][0]
        policies.append(
            {
                "declaration_id": f"policy:{action['action_id']}",
                "kind": "policy",
                "decision": decision,
                "principal": "principal:worker",
                "action": "write",
                "tool": "tool:editor",
                "target": "resource:repository",
                "data_flow": FLOW,
                "expected_effect": {
                    "operation": effect["operation"],
                    "target": effect["target"],
                },
                "requested_authority": "mutation",
                "source_references": SOURCE,
            }
        )
    declarations = (
        {
            "declaration_id": "principal:worker",
            "kind": "principal",
            "source_references": SOURCE,
        },
        {
            "declaration_id": "tool:editor",
            "kind": "resource",
            "resource_type": "tool",
            "source_references": SOURCE,
        },
        {
            "declaration_id": "resource:repository",
            "kind": "resource",
            "resource_type": "repository",
            "source_references": SOURCE,
        },
        *policies,
    )
    artifact = _normalized(IRFamily.SECURITY, declarations=declarations)
    policy = compile_security_constraints(artifact)
    requests = []
    for action in actions:
        effect = action["effects"][0]
        requests.append(
            SecurityAuthorizationRequest(
                security_root_artifact_id=artifact.root_artifact_id,
                security_root_cid_v1=artifact.root_cid_v1,
                security_root_supervisor_digest=artifact.root_supervisor_digest,
                principal="principal:worker",
                action="write",
                tool="tool:editor",
                target="resource:repository",
                data_flow=FLOW,
                expected_effect={
                    "operation": effect["operation"],
                    "target": effect["target"],
                },
                current_state={"repository": "current"},
                requested_authority="mutation",
                evaluated_at_ms=500,
            )
        )
    return artifact, policy, tuple(requests)


def _proof(obligation_id: str, *, tree: str = TREE) -> ProofReceipt:
    evidence = ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:kernel-proof",
        subject_id=obligation_id,
        verifier_id="kernel:lean",
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
    )
    return ProofReceipt(
        obligation_id=obligation_id,
        plan_id="plan:admission-service",
        attempt_id=f"attempt:{obligation_id}",
        repository_id="repository:service",
        repository_tree_id=tree,
        ast_scope_ids=("scope:service",),
        premise_ids=(),
        translator_id="translator:reviewed",
        solver_id="solver:reviewed",
        kernel_id="kernel:lean",
        toolchain_id="toolchain:locked",
        policy_id="policy:proof",
        resource_budget=ResourceBudget(),
        verdict=ProofVerdict.PROVED,
        evidence=(evidence,),
        kernel_receipt_id=f"kernel:{obligation_id}",
        provider_claimed_assurance=AssuranceLevel.UNVERIFIED,
    )


def _closure() -> MandatoryClosure:
    return MandatoryClosure(
        root_id="decision-root:service",
        decision_id="decision:admit",
        node_ids=("decision:admit",),
        edge_ids=(),
        paths={"decision:admit": ("decision:admit",)},
    )


def _materials(
    *,
    action_ids: tuple[str, ...] = ("action:write",),
    security_decision: str = "allow",
    legal_modality: str = "permission",
    include_proof: bool = True,
    include_validation: bool = True,
    execution_plan_id: str = "execution-plan:service",
    policy_ids: tuple[str, ...] = ("policy:admission", "policy:security"),
    provider_claim: dict[str, object] | None = None,
    authority_matched: bool = True,
) -> PlanAdmissionServiceRequest:
    candidate = _candidate(action_ids)
    intent_request = _intent_request(candidate)
    legal = _legal_result(modality=legal_modality)
    _artifact, security_policy, security_requests = _security(
        candidate, decision=security_decision
    )
    actions = candidate["actions"]
    assert isinstance(actions, list)
    effects = candidate["effects"]
    assert isinstance(effects, list)

    proof_ids = {
        obligation_id
        for action in actions
        for obligation_id in action.get("proof_obligation_ids", ()) or ()
    }
    # Intent/legal proof obligations are also mandatory.
    intent_proofs = {
        item.obligation_id
        for item in intent_request.constraint_set.proof_obligations
    }
    all_proofs = proof_ids | intent_proofs
    proof_results = (
        tuple(_proof(obligation_id) for obligation_id in sorted(all_proofs))
        if include_proof
        else ()
    )

    validation_ids = {
        requirement_id
        for action in actions
        for requirement_id in action.get("validation_requirement_ids", ()) or ()
    }
    validation_requirements = tuple(
        ValidationRequirement(
            requirement_id=requirement_id,
            action_ids=tuple(
                str(action["action_id"])
                for action in actions
                if requirement_id
                in (action.get("validation_requirement_ids") or ())
            ),
            command="pytest",
            required=True,
        )
        for requirement_id in sorted(validation_ids)
    )
    validation_results = (
        tuple(
            ValidationResult(
                requirement_id=requirement_id,
                status=ValidationStatus.PASSED,
                repository_tree_id=TREE,
                evidence_id=f"evidence:validation:{requirement_id}",
            )
            for requirement_id in sorted(validation_ids)
        )
        if include_validation
        else ()
    )

    assumption_ids = {
        assumption_id
        for action in actions
        for assumption_id in action.get("assumption_ids", ()) or ()
    }

    authority = AdmissionAuthority(
        principal="principal:worker",
        requested_authority="mutation",
        grant_principal=(
            "principal:worker" if authority_matched else "principal:other"
        ),
        granted_authorities=("mutation",) if authority_matched else ("read",),
        grant_source_ids=("security-grant:service",),
    )

    effect_projections = tuple(
        {
            "operation": effect["operation"],
            "target": effect["target"],
        }
        for effect in effects
    )

    return PlanAdmissionServiceRequest(
        candidate_plan=candidate,
        repository_tree_id=TREE,
        formal_plan_id=str(candidate["plan_id"]),
        formal_source_identity="formal-source:service",
        evidence_bundle_id="evidence-bundle:service",
        execution_plan_id=execution_plan_id,
        policy_ids=policy_ids,
        intent_request=intent_request,
        legal_results=(legal,),
        security_policy=security_policy,
        security_requests=security_requests,
        action_bindings=tuple(
            ActionDomainBinding(
                action_id=str(action["action_id"]),
                legal_result_ids=(legal.content_id,),
                security_request_ids=tuple(
                    request.content_id
                    for request, effect in zip(
                        security_requests, effects, strict=True
                    )
                    if effect["action_id"] == action["action_id"]
                ),
            )
            for action in actions
        ),
        authority=authority,
        root_bindings=(
            RootBinding("intent", "intent-root:service", "intent-root:service"),
            RootBinding("legal", "legal-root:service", "legal-root:service"),
            RootBinding(
                "security", "security-root:service", "security-root:service"
            ),
            RootBinding("program", TREE, TREE),
        ),
        program_dependencies=tuple(
            ProgramDependency(
                dependency_id=f"dependency:{action['action_id']}",
                action_id=str(action["action_id"]),
                depends_on_action_ids=tuple(action["depends_on"]),
                evidence_ids=(f"evidence:dependency:{action['action_id']}",),
            )
            for action in actions
        ),
        assumptions=tuple(
            AdmissionAssumption(
                assumption_id,
                action_ids=tuple(
                    str(action["action_id"])
                    for action in actions
                    if assumption_id in (action.get("assumption_ids") or ())
                ),
                required=True,
                satisfied=True,
                evidence_ids=(f"evidence:assumption:{assumption_id}",),
            )
            for assumption_id in sorted(assumption_ids)
        ),
        proof_results=proof_results,
        validation_requirements=validation_requirements,
        validation_results=validation_results,
        intent_effects=effect_projections,
        code_effects=effect_projections,
        generated_formula_ids=(),
        mandatory_closure=_closure(),
        graph_complete=True,
        provider_admission_claim=provider_claim,
    )


def test_service_interface_and_constructs_own_request_ignoring_provider_claims() -> None:
    materials = _materials(
        provider_claim={
            "admitted": True,
            "verdict": "admitted",
            "score": 99,
            "provider_admitted": True,
            "authorizes_execution": True,
        }
    )
    service = PlanAdmissionService()
    assert service.INTERFACE == PLAN_ADMISSION_SERVICE_INTERFACE

    constructed = service.construct_request(materials)
    assert isinstance(constructed, PlanAdmissionRequest)
    # Provider claims never become part of the constructed request payload.
    payload = constructed.to_dict()
    assert "admitted" not in payload or payload.get("admitted") is not True
    assert payload.get("authorizes_execution") is not True
    # Identity is content-addressed from primitive fields.
    again = construct_plan_admission_request(materials)
    assert again.request_id == constructed.request_id


def test_admits_only_after_every_stage_in_fixed_order() -> None:
    materials = _materials()
    receipt = admit_plan_through_service(materials)

    assert receipt.admitted
    assert receipt.verdict is PlanAdmissionVerdict.ADMITTED
    assert tuple(item.stage for item in receipt.stage_results) == ADMISSION_STAGE_ORDER
    assert all(item.passed for item in receipt.stage_results)
    # Full multi-artifact binding on every admitted receipt.
    assert receipt.candidate_plan_id == materials.candidate_plan_id
    assert receipt.candidate_graph_id == materials.candidate_graph_id
    assert receipt.evidence_bundle_id == "evidence-bundle:service"
    assert receipt.formal_plan_id == materials.formal_plan_id
    assert receipt.formal_source_identity == "formal-source:service"
    assert receipt.execution_plan_id == "execution-plan:service"
    assert receipt.policy_ids == ("policy:admission", "policy:security")
    assert receipt.repository_tree_id == TREE
    assert dict(receipt.semantic_roots) == dict(materials.semantic_roots)
    assert receipt.ir_request_id
    assert receipt.ir_receipt_id
    assert receipt.ir_receipt is not None
    assert receipt.ir_receipt.admitted
    assert receipt.proof_obligation_ids
    assert receipt.proof_result_ids
    assert receipt.authorizes_execution is False


def test_unknown_mandatory_domains_fail_closed() -> None:
    base = _materials()

    missing_execution = replace(base, execution_plan_id="")
    missing_policy = replace(base, policy_ids=())
    missing_proof = replace(base, proof_results=())
    missing_validation = replace(base, validation_results=())
    missing_security = replace(base, security_requests=())
    missing_legal = replace(base, legal_results=())
    bad_authority = replace(
        base,
        authority=AdmissionAuthority(
            principal="principal:worker",
            requested_authority="mutation",
            grant_principal="principal:other",
            granted_authorities=("read",),
            grant_source_ids=("grant:stale",),
        ),
    )
    stale_root = replace(
        base,
        root_bindings=(
            RootBinding("intent", "intent-root:service", "intent-root:other"),
            RootBinding("legal", "legal-root:service", "legal-root:service"),
            RootBinding(
                "security", "security-root:service", "security-root:service"
            ),
            RootBinding("program", TREE, TREE),
        ),
    )

    cases = (
        (missing_execution, PlanAdmissionServiceCode.MISSING_EXECUTION_PLAN.value),
        (missing_policy, PlanAdmissionServiceCode.MISSING_POLICY.value),
        (missing_proof, PlanAdmissionServiceCode.MISSING_PROOF.value),
        (
            missing_validation,
            PlanAdmissionServiceCode.UNKNOWN_MANDATORY_VALIDATION.value,
        ),
        (
            missing_security,
            PlanAdmissionServiceCode.UNKNOWN_MANDATORY_SECURITY.value,
        ),
        (
            missing_legal,
            PlanAdmissionServiceCode.UNKNOWN_MANDATORY_APPLICABILITY.value,
        ),
        (bad_authority, PlanAdmissionServiceCode.AUTHORITY_MISMATCH.value),
        (stale_root, PlanAdmissionServiceCode.STALE_ROOT.value),
    )
    for materials, expected in cases:
        receipt = PlanAdmissionService().admit(materials)
        assert not receipt.admitted, expected
        assert expected in receipt.reason_codes
        assert not receipt.authorizes_execution


def test_security_forbidden_is_checked_against_intent_and_code_effects() -> None:
    materials = _materials(security_decision="deny")
    receipt = PlanAdmissionService().admit(materials)

    assert not receipt.admitted
    assert PlanAdmissionServiceCode.SECURITY_FORBIDDEN.value in receipt.reason_codes
    security_stage = next(
        item
        for item in receipt.stage_results
        if item.stage is PlanAdmissionStage.SECURITY
    )
    assert not security_stage.passed
    assert "intent" in security_stage.message or "code" in security_stage.message


def test_security_stream_gap_fails_when_code_effects_uncovered() -> None:
    materials = _materials()
    # Code effects diverge from security/intent coverage.
    broken = replace(
        materials,
        code_effects=(
            {"operation": "delete", "target": "src/secret.py"},
        ),
    )
    receipt = PlanAdmissionService().admit(broken)
    assert not receipt.admitted
    assert (
        PlanAdmissionServiceCode.SECURITY_STREAM_GAP.value in receipt.reason_codes
        or PlanAdmissionServiceCode.UNDECLARED_EFFECT.value in receipt.reason_codes
    )


def test_receipt_round_trip_and_tampering_fail() -> None:
    materials = _materials()
    receipt = PlanAdmissionService().admit(materials)
    payload = receipt.to_dict()
    restored = PlanAdmissionServiceReceipt.from_dict(payload)
    assert restored.receipt_id == receipt.receipt_id
    assert restored.admitted is receipt.admitted

    tampered = dict(payload)
    tampered["candidate_plan_id"] = "plan:forged"
    with pytest.raises(PlanAdmissionServiceError, match="identity"):
        PlanAdmissionServiceReceipt.from_dict(tampered)

    authority_claim = dict(payload)
    authority_claim["authorizes_execution"] = True
    with pytest.raises(PlanAdmissionServiceError, match="authorize execution"):
        PlanAdmissionServiceReceipt.from_dict(authority_claim)

    provider_claim = dict(payload)
    provider_claim["provider_claims_are_authority"] = True
    with pytest.raises(PlanAdmissionServiceError, match="provider"):
        PlanAdmissionServiceReceipt.from_dict(provider_claim)


def test_replay_rejects_mismatched_or_stale_receipt() -> None:
    materials = _materials()
    service = PlanAdmissionService()
    receipt = service.admit(materials)
    # Exact replay succeeds.
    assert service.replay(materials, receipt).receipt_id == receipt.receipt_id

    other = replace(materials, execution_plan_id="execution-plan:other")
    with pytest.raises(PlanAdmissionServiceError, match="replay"):
        service.replay(other, receipt)


def test_provider_admission_claims_never_admit() -> None:
    materials = _materials(
        security_decision="deny",
        provider_claim={
            "admitted": True,
            "verdict": "admitted",
            "score": 1_000_000,
            "passed": True,
        },
    )
    receipt = PlanAdmissionService().admit(materials)
    assert not receipt.admitted
    assert PlanAdmissionServiceCode.SECURITY_FORBIDDEN.value in receipt.reason_codes


def test_default_prompt_admission_without_ir_request_is_not_binding_mismatch() -> None:
    """Factory absence must not surface as IR_BINDING_MISMATCH."""

    workflow = _request()
    scan = _scan(workflow)
    graph = parse_prompt_goal_graph(_encoded_proposal(scan), workflow, scan)

    result = admit_prompt_plan(
        graph,
        repository_tree_id=scan.dirty_worktree_root,
        ir_request=None,
        workflow_request=workflow,
        scan_receipt=scan,
    )

    assert not result.admitted
    assert (
        PromptPlanAdmissionCode.IR_BINDING_MISMATCH.value
        not in result.reason_codes
    )
    assert (
        PromptPlanAdmissionCode.UNKNOWN_MANDATORY_STATE.value
        in result.reason_codes
    )


def test_stage_order_is_fixed_even_on_early_failure() -> None:
    materials = replace(_materials(), formal_plan_id="plan:different")
    receipt = PlanAdmissionService().admit(materials)
    assert not receipt.admitted
    assert tuple(item.stage for item in receipt.stage_results) == ADMISSION_STAGE_ORDER
    formal = receipt.stage_results[0]
    assert formal.stage is PlanAdmissionStage.FORMAL
    assert not formal.passed
    # Later stages still run (including IR join) so evidence is complete.
    assert any(
        item.stage is PlanAdmissionStage.IR_JOIN for item in receipt.stage_results
    )
