from __future__ import annotations

import copy
from dataclasses import replace

import pytest

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
from ipfs_accelerate_py.agent_supervisor.proof.ir_adapters import IRAdapterRegistry
from ipfs_accelerate_py.agent_supervisor.proof.ir_constraint_compiler import (
    ActionDomainBinding,
    AdmissionAssumption,
    AdmissionAuthority,
    AdmissionRejectionCode,
    PlanAdmissionReceipt,
    PlanAdmissionRequest,
    ProgramDependency,
    RootBinding,
    ValidationRequirement,
    ValidationResult,
    ValidationStatus,
    compile_plan_admission,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_registry import (
    IRFamily,
    IRLoadRequest,
    IRLoadStatus,
    IRRegistry,
    deterministic_ir_fixture,
)
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


TREE = "tree:admission-fixture"
SOURCE = ({"source_id": "review:fixture", "span_id": "section:1"},)
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


def _candidate(action_ids: tuple[str, ...]) -> dict[str, object]:
    actions = []
    effects = []
    for index, action_id in enumerate(action_ids):
        depends_on = (
            ["action:prepare"]
            if action_id == "action:write" and "action:prepare" in action_ids
            else ["action:write"]
            if action_id == "action:validate" and "action:write" in action_ids
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
            }
        )
        effects.append(effect)
    return {
        "plan_id": "plan:admission-fixture",
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


def _legal_result(
    *,
    modality: str = "permission",
    proof_obligation_id: str = "",
):
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


def _security(
    candidate: dict[str, object],
    *,
    decision: str = "allow",
):
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
        plan_id="plan:admission-fixture",
        attempt_id="attempt:proof",
        repository_id="repository:fixture",
        repository_tree_id=tree,
        ast_scope_ids=("scope:fixture",),
        premise_ids=(),
        translator_id="translator:reviewed",
        solver_id="solver:reviewed",
        kernel_id="kernel:lean",
        toolchain_id="toolchain:locked",
        policy_id="policy:proof",
        resource_budget=ResourceBudget(),
        verdict=ProofVerdict.PROVED,
        evidence=(evidence,),
        kernel_receipt_id="kernel-receipt:fixture",
        provider_claimed_assurance=AssuranceLevel.UNVERIFIED,
    )


def _closure(*, irrelevant: int = 0) -> MandatoryClosure:
    return MandatoryClosure(
        root_id="decision-root:fixture",
        decision_id="decision:admit",
        node_ids=("decision:admit",),
        edge_ids=(),
        paths={"decision:admit": ("decision:admit",)},
        annotation_node_ids=tuple(
            f"annotation:irrelevant:{index}" for index in range(irrelevant)
        ),
    )


def _request(
    *,
    action_ids: tuple[str, ...] = ("action:write",),
    legal_modality: str = "permission",
    security_decision: str = "allow",
    proof_obligation_id: str = "",
    proof_results: tuple[ProofReceipt, ...] = (),
    generated_formula_ids: tuple[str, ...] = (),
    closure: MandatoryClosure | None = None,
) -> PlanAdmissionRequest:
    candidate = _candidate(action_ids)
    intent_request = _intent_request(candidate)
    legal = _legal_result(
        modality=legal_modality,
        proof_obligation_id=proof_obligation_id,
    )
    security_artifact, security_policy, security_requests = _security(
        candidate, decision=security_decision
    )
    action_bindings = tuple(
        ActionDomainBinding(
            action_id=str(action["action_id"]),
            legal_result_ids=(legal.content_id,),
            security_request_ids=(security_request.content_id,),
        )
        for action, security_request in zip(
            candidate["actions"], security_requests, strict=True
        )
    )
    return PlanAdmissionRequest(
        candidate_plan=candidate,
        repository_tree_id=TREE,
        intent_request=intent_request,
        legal_results=(legal,),
        security_policy=security_policy,
        security_requests=security_requests,
        action_bindings=action_bindings,
        authority=AdmissionAuthority(
            principal="principal:worker",
            requested_authority="mutation",
            grant_principal="principal:worker",
            granted_authorities=("mutation",),
            grant_source_ids=("security-grant:fixture",),
        ),
        root_bindings=(
            RootBinding("intent", "root:intent", "root:intent"),
            RootBinding("legal", "root:legal", "root:legal"),
            RootBinding(
                "security",
                security_artifact.root_supervisor_digest,
                security_artifact.root_supervisor_digest,
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
            for action in candidate["actions"]
        ),
        assumptions=(
            AdmissionAssumption(
                "assumption:workspace-isolated",
                action_ids=tuple(action_ids),
                satisfied=True,
                evidence_ids=("evidence:assumption",),
            ),
        ),
        proof_results=proof_results,
        validation_requirements=(
            ValidationRequirement(
                "validation:pytest",
                action_ids=("action:write",),
                command="python -m pytest -q",
            ),
        ),
        validation_results=(
            ValidationResult(
                "validation:pytest",
                ValidationStatus.PASSED,
                TREE,
                evidence_id="evidence:pytest",
            ),
        ),
        generated_formula_ids=generated_formula_ids,
        mandatory_closure=closure or _closure(),
    )


def test_admits_exact_cross_domain_request_and_receipt_round_trips() -> None:
    request = _request()

    receipt = compile_plan_admission(request)

    assert receipt.admitted
    assert not receipt.authorizes_execution
    assert receipt.candidate_graph_id == request.candidate_graph_id
    assert receipt.checked_dependency_ids == ("dependency:action:write",)
    assert receipt.checked_assumption_ids == ("assumption:workspace-isolated",)
    assert receipt.checked_validation_ids == ("validation:pytest",)
    assert receipt.legal_permission_ids == ("norm:permission",)
    assert receipt.security_grant_ids
    assert not receipt.reason_codes
    assert PlanAdmissionReceipt.from_dict(receipt.to_dict()) == receipt

    tampered = copy.deepcopy(receipt.to_dict())
    tampered["security_grant_ids"] = []
    with pytest.raises(ValueError, match="identity"):
        PlanAdmissionReceipt.from_dict(tampered)


@pytest.mark.parametrize(
    ("build", "expected"),
    (
        (
            lambda: replace(
                _request(),
                graph_complete=False,
            ),
            AdmissionRejectionCode.INCOMPLETE_GRAPH,
        ),
        (
            lambda: _request(legal_modality="prohibition"),
            AdmissionRejectionCode.LEGAL_PROHIBITION,
        ),
        (
            lambda: _request(security_decision="deny"),
            AdmissionRejectionCode.SECURITY_DENY,
        ),
        (
            lambda: replace(
                _request(),
                authority=AdmissionAuthority(
                    "principal:worker",
                    "mutation",
                    "principal:other",
                    ("mutation",),
                    ("security-grant:fixture",),
                ),
            ),
            AdmissionRejectionCode.AUTHORITY_MISMATCH,
        ),
        (
            lambda: replace(
                _request(),
                root_bindings=(
                    RootBinding("program", TREE, "tree:stale"),
                ),
            ),
            AdmissionRejectionCode.STALE_ROOT,
        ),
        (
            lambda: replace(
                _request(),
                assumptions=(
                    AdmissionAssumption(
                        "assumption:workspace-isolated",
                        ("action:write",),
                        satisfied=False,
                    ),
                ),
            ),
            AdmissionRejectionCode.ASSUMPTION_UNRESOLVED,
        ),
        (
            lambda: replace(
                _request(),
                program_dependencies=(
                    ProgramDependency(
                        "dependency:write",
                        "action:write",
                        required=True,
                        satisfied=False,
                    ),
                ),
            ),
            AdmissionRejectionCode.DEPENDENCY_UNSATISFIED,
        ),
        (
            lambda: replace(_request(), validation_results=()),
            AdmissionRejectionCode.VALIDATION_MISSING,
        ),
    ),
)
def test_every_hard_failure_is_rejected_with_a_counterexample(
    build, expected: AdmissionRejectionCode
) -> None:
    receipt = compile_plan_admission(build())

    assert not receipt.admitted
    assert expected.value in receipt.reason_codes
    matching = [
        item for item in receipt.rejection_reasons if item.code is expected
    ]
    assert matching
    assert all(
        any(example.rejection_id == rejection.rejection_id for example in receipt.counterexamples)
        for rejection in matching
    )


def test_legal_permission_is_not_a_grant_and_security_deny_still_prunes() -> None:
    receipt = compile_plan_admission(_request(security_decision="deny"))

    assert not receipt.admitted
    assert receipt.legal_permission_ids == ("norm:permission",)
    assert not receipt.security_grant_ids
    assert AdmissionRejectionCode.SECURITY_DENY.value in receipt.reason_codes
    assert receipt.to_dict()["permissions_are_grants"] is False


def test_generated_formula_never_substitutes_for_a_typed_proof_receipt() -> None:
    obligation_id = "proof:legal-compliance"
    missing = _request(
        proof_obligation_id=obligation_id,
        generated_formula_ids=(obligation_id,),
    )

    rejected = compile_plan_admission(missing)
    admitted = compile_plan_admission(
        replace(missing, proof_results=(_proof(obligation_id),))
    )

    assert AdmissionRejectionCode.MISSING_PROOF.value in rejected.reason_codes
    proof_rejection = next(
        item
        for item in rejected.rejection_reasons
        if item.code is AdmissionRejectionCode.MISSING_PROOF
    )
    assert proof_rejection.details["generated_formula_present"] is True
    assert rejected.to_dict()["generated_formulas_are_proofs"] is False
    assert admitted.admitted
    assert admitted.proof_result_ids == (_proof(obligation_id).receipt_id,)


def test_stale_proof_and_undeclared_security_effect_fail_closed() -> None:
    obligation_id = "proof:legal-compliance"
    stale = _request(
        proof_obligation_id=obligation_id,
        proof_results=(_proof(obligation_id, tree="tree:old"),),
    )
    base = _request()
    changed_security = replace(
        base.security_requests[0],
        expected_effect={"operation": "delete", "target": "src/undeclared.py"},
    )
    undeclared = replace(
        base,
        security_requests=(changed_security,),
        action_bindings=(
            replace(
                base.action_bindings[0],
                security_request_ids=(changed_security.content_id,),
            ),
        ),
    )

    stale_receipt = compile_plan_admission(stale)
    effect_receipt = compile_plan_admission(undeclared)

    assert AdmissionRejectionCode.INVALID_PROOF.value in stale_receipt.reason_codes
    assert AdmissionRejectionCode.UNDECLARED_EFFECT.value in effect_receipt.reason_codes


def test_complete_reasons_drive_dependency_local_replanning() -> None:
    request = _request(
        action_ids=(
            "action:prepare",
            "action:write",
            "action:validate",
            "action:lint",
        )
    )
    broken = replace(
        request,
        authority=replace(request.authority, grant_principal="principal:other"),
        program_dependencies=tuple(
            replace(item, satisfied=False, evidence_ids=())
            if item.action_id == "action:write"
            else item
            for item in request.program_dependencies
        ),
        validation_results=(),
    )

    receipt = compile_plan_admission(broken)

    assert {
        AdmissionRejectionCode.AUTHORITY_MISMATCH.value,
        AdmissionRejectionCode.DEPENDENCY_UNSATISFIED.value,
        AdmissionRejectionCode.VALIDATION_MISSING.value,
    } <= set(receipt.reason_codes)
    local = next(
        item
        for item in receipt.counterexamples
        if item.witness["code"]
        == AdmissionRejectionCode.DEPENDENCY_UNSATISFIED.value
    )
    assert local.failing_action_ids == ("action:write",)
    assert local.affected_action_ids == ("action:validate", "action:write")
    assert local.fixed_action_ids == ("action:lint", "action:prepare")
    assert {"action:write", "action:validate"} <= set(
        receipt.local_replan_action_ids
    )


def test_candidate_order_and_irrelevant_closure_growth_do_not_change_outcome() -> None:
    action_ids = ("action:prepare", "action:write", "action:validate")
    first = _request(
        action_ids=action_ids,
        closure=_closure(),
    )
    reversed_candidate = _candidate(action_ids)
    reversed_candidate["actions"] = list(reversed(reversed_candidate["actions"]))
    reversed_candidate["effects"] = list(reversed(reversed_candidate["effects"]))
    reordered = replace(
        first,
        candidate_plan=reversed_candidate,
        legal_results=tuple(reversed(first.legal_results)),
        security_requests=tuple(reversed(first.security_requests)),
        action_bindings=tuple(reversed(first.action_bindings)),
        root_bindings=tuple(reversed(first.root_bindings)),
        program_dependencies=tuple(reversed(first.program_dependencies)),
        mandatory_closure=_closure(irrelevant=100),
    )

    first_receipt = compile_plan_admission(first)
    reordered_receipt = compile_plan_admission(reordered)

    assert first_receipt.admitted and reordered_receipt.admitted
    assert first_receipt.candidate_graph_id == reordered_receipt.candidate_graph_id
    assert first_receipt.reason_codes == reordered_receipt.reason_codes == ()
    assert first_receipt.closure_id == reordered_receipt.closure_id
    assert first_receipt.legal_permission_ids == reordered_receipt.legal_permission_ids
    assert first_receipt.security_grant_ids == reordered_receipt.security_grant_ids
