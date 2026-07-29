from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.formal_plan_compiler import (
    CompilationStatus,
    FormalPlanCompiler,
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
    compile_intent_constraints,
    create_intent_conformance_request,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_constraint_compiler import (
    ActionDomainBinding,
    AdmissionAssumption,
    AdmissionAuthority,
    PlanAdmissionRequest,
    ProgramDependency,
    RootBinding,
    ValidationRequirement,
    ValidationResult,
    ValidationStatus,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_registry import IRFamily
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
    parse_prompt_goal_graph,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_plan_admission import (
    PromptPlanAdmissionCode,
    PromptPlanAdmissionPolicy,
    PromptPlanAdmissionReceipt,
    admit_prompt_plan,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    PromptAcceptanceRecord,
    PromptGoalGraph,
    PromptOutputRecord,
    PromptValidationRecord,
)
from ipfs_accelerate_py.agent_supervisor.proof.security_constraint_adapter import (
    SecurityAuthorizationRequest,
    compile_security_constraints,
)
from test.api.test_agent_supervisor_ir_constraint_compiler import (
    _closure,
    _legal_result,
    _normalized,
)
from test.api.test_agent_supervisor_prompt_goal_planner import (
    _encoded_proposal,
    _request,
    _scan,
)


FLOW = {"classification": "source", "direction": "workspace_to_tool"}
SOURCE = ({"source_id": "review:prompt-plan", "span_id": "fixture:1"},)


def _graph_fixture():
    workflow = _request()
    scan = _scan(workflow)
    graph = parse_prompt_goal_graph(
        _encoded_proposal(scan), workflow, scan
    )
    return workflow, scan, graph


def _effect_projection(effect: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in effect.items()
        if key not in {"effect_id", "action_id", "task_id", "metadata"}
    }


def _proof(
    obligation_id: str,
    *,
    plan_id: str,
    tree_id: str,
) -> ProofReceipt:
    evidence = ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id=f"artifact:{obligation_id}",
        subject_id=obligation_id,
        verifier_id="kernel:lean",
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
    )
    return ProofReceipt(
        obligation_id=obligation_id,
        plan_id=plan_id,
        attempt_id=f"attempt:{obligation_id}",
        repository_id="repository:prompt-plan",
        repository_tree_id=tree_id,
        ast_scope_ids=("scope:prompt-plan",),
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


def _ir_request(
    graph: PromptGoalGraph,
    workflow,
    tree_id: str,
    *,
    security_decision: str = "allow",
) -> PlanAdmissionRequest:
    formal = FormalPlanCompiler().compile_prompt_graph(
        graph, repository_tree_id=tree_id
    )
    assert formal.status is CompilationStatus.COMPILED
    projection = formal.admission_projection
    assert projection is not None
    candidate = projection.to_dict()
    actions = list(candidate["actions"])
    effects = list(candidate["effects"])

    declarations: list[dict[str, object]] = [
        {
            "id": graph.root_goal.goal_cid,
            "kind": "goal",
            "grounded": True,
        }
    ]
    for action in actions:
        declarations.append(
            {
                "id": action["action_id"],
                "kind": "action",
                "goal_id": graph.root_goal.goal_cid,
                "depends_on": list(action["depends_on"]),
                "grounded": True,
            }
        )
        for effect in action["effects"]:
            declarations.append(
                {
                    "id": effect["effect_id"],
                    "kind": "effect",
                    "action_id": action["action_id"],
                    **_effect_projection(dict(effect)),
                    "grounded": True,
                }
            )
    intent = _normalized(IRFamily.INTENT, declarations=tuple(declarations))
    formalization = _normalized(IRFamily.FORMALIZATION)
    intent_compilation = compile_intent_constraints(intent, formalization)
    constraints = intent_compilation.require_constraint_set()
    intent_candidate = copy.deepcopy(candidate)
    intent_candidate["intent_root"] = dict(constraints.intent_root)
    intent_candidate["formalization_root"] = dict(
        constraints.formalization_root
    )
    intent_candidate["goal_ids"] = [graph.root_goal.goal_cid]
    intent_request = create_intent_conformance_request(
        intent_compilation,
        intent_candidate,
        discharged_obligation_ids=tuple(
            item.obligation_id for item in constraints.proof_obligations
        ),
    )

    legal = _legal_result()
    security_declarations: list[dict[str, object]] = [
        {
            "declaration_id": "actor:prompt-plan",
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
    ]
    for effect in effects:
        security_declarations.append(
            {
                "declaration_id": f"policy:{effect['effect_id']}",
                "kind": "policy",
                "decision": security_decision,
                "principal": "actor:prompt-plan",
                "action": "write",
                "tool": "tool:editor",
                "target": "resource:repository",
                "data_flow": FLOW,
                "expected_effect": _effect_projection(dict(effect)),
                "requested_authority": "mutation",
                "source_references": SOURCE,
            }
        )
    security_artifact = _normalized(
        IRFamily.SECURITY, declarations=tuple(security_declarations)
    )
    security_policy = compile_security_constraints(security_artifact)
    security_requests = tuple(
        SecurityAuthorizationRequest(
            security_root_artifact_id=security_artifact.root_artifact_id,
            security_root_cid_v1=security_artifact.root_cid_v1,
            security_root_supervisor_digest=(
                security_artifact.root_supervisor_digest
            ),
            principal="actor:prompt-plan",
            action="write",
            tool="tool:editor",
            target="resource:repository",
            data_flow=FLOW,
            expected_effect=_effect_projection(dict(effect)),
            current_state={"repository": "current"},
            requested_authority="mutation",
            evaluated_at_ms=500,
        )
        for effect in effects
    )
    action_bindings = tuple(
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
    )

    proof_ids = {
        obligation_id
        for action in actions
        for obligation_id in action["proof_obligation_ids"]
    }
    validation_ids = {
        requirement_id
        for action in actions
        for requirement_id in action["validation_requirement_ids"]
    }
    assert formal.plan is not None
    formal_requirements = {
        item.requirement_id: item
        for item in formal.plan.evidence_requirements
    }
    assumption_ids = {
        assumption_id
        for action in actions
        for assumption_id in action["assumption_ids"]
    }
    return PlanAdmissionRequest(
        candidate_plan=candidate,
        repository_tree_id=tree_id,
        intent_request=intent_request,
        legal_results=(legal,),
        security_policy=security_policy,
        security_requests=security_requests,
        action_bindings=action_bindings,
        authority=AdmissionAuthority(
            principal="actor:prompt-plan",
            requested_authority="mutation",
            grant_principal="actor:prompt-plan",
            granted_authorities=("mutation",),
            grant_source_ids=("security-grant:prompt-plan",),
        ),
        root_bindings=(
            RootBinding(
                "intent", workflow.intent_ir_root, workflow.intent_ir_root
            ),
            RootBinding("legal", workflow.legal_ir_root, workflow.legal_ir_root),
            RootBinding(
                "security",
                workflow.security_ir_root,
                workflow.security_ir_root,
            ),
            RootBinding("program", tree_id, tree_id),
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
                    if assumption_id in action["assumption_ids"]
                ),
                satisfied=True,
                evidence_ids=("evidence:assumption",),
            )
            for assumption_id in sorted(assumption_ids)
        ),
        proof_results=tuple(
            _proof(
                obligation_id,
                plan_id=projection.plan_id,
                tree_id=tree_id,
            )
            for obligation_id in sorted(proof_ids)
        ),
        validation_requirements=tuple(
            ValidationRequirement(
                requirement_id,
                action_ids=tuple(
                    str(action["action_id"])
                    for action in actions
                    if requirement_id
                    in action["validation_requirement_ids"]
                ),
                command=formal_requirements[
                    requirement_id
                ].fallback_check_ids[0],
            )
            for requirement_id in sorted(validation_ids)
        ),
        validation_results=tuple(
            ValidationResult(
                requirement_id,
                ValidationStatus.PASSED,
                tree_id,
                evidence_id=f"evidence:{requirement_id}",
            )
            for requirement_id in sorted(validation_ids)
        ),
        generated_formula_ids=projection.generated_formula_ids,
        mandatory_closure=_closure(),
    )


def _admit(*, security_decision: str = "allow"):
    workflow, scan, graph = _graph_fixture()
    ir_request = _ir_request(
        graph,
        workflow,
        scan.dirty_worktree_root,
        security_decision=security_decision,
    )
    result = admit_prompt_plan(
        graph,
        repository_tree_id=scan.dirty_worktree_root,
        ir_request=ir_request,
        workflow_request=workflow,
        scan_receipt=scan,
    )
    return workflow, scan, graph, ir_request, result


def test_admits_only_after_quality_formal_ir_proof_and_validation_gates() -> None:
    _workflow, _scan_receipt, graph, ir_request, result = _admit()

    assert result.admitted
    assert result.formal_compilation is not None
    assert result.formal_compilation.status is CompilationStatus.COMPILED
    assert result.ir_receipt is not None and result.ir_receipt.admitted
    assert result.receipt.final_plan_cid
    assert result.receipt.final_task_cids == tuple(
        sorted(task.task_cid for task in graph.tasks)
    )
    assert all(result.receipt.invariants.values())
    assert not result.receipt.authorizes_execution
    assert result.receipt.ir_request_id == ir_request.request_id
    restored = PromptPlanAdmissionReceipt.from_dict(result.receipt.to_dict())
    assert restored == result.receipt


def test_candidate_order_and_irrelevant_corpus_growth_do_not_change_result() -> None:
    workflow, scan, graph, ir_request, first = _admit()
    payload = graph.to_dict()
    payload["goals"].reverse()
    payload["tasks"].reverse()
    payload["evidence"].reverse()
    reordered = PromptGoalGraph.from_dict(payload)

    second = admit_prompt_plan(
        reordered,
        repository_tree_id=scan.dirty_worktree_root,
        ir_request=ir_request,
        workflow_request=workflow,
        scan_receipt=scan,
        irrelevant_corpus=tuple(
            {"path": f"unrelated/{index}.txt"} for index in range(1_000)
        ),
    )

    assert first.admitted and second.admitted
    assert first.receipt == second.receipt
    assert first.receipt.receipt_id == second.receipt.receipt_id
    assert first.formal_compilation is not None
    assert second.formal_compilation is not None
    assert first.formal_compilation.plan_id == second.formal_compilation.plan_id


def test_final_cids_are_withheld_on_exact_ir_binding_or_hard_domain_failure() -> None:
    workflow, scan, graph, ir_request, _result = _admit()
    mismatched = replace(
        ir_request,
        generated_formula_ids=(*ir_request.generated_formula_ids, "formula:foreign"),
    )

    binding_failure = admit_prompt_plan(
        graph,
        repository_tree_id=scan.dirty_worktree_root,
        ir_request=mismatched,
        workflow_request=workflow,
        scan_receipt=scan,
    )
    _workflow, _scan_receipt, _graph, _request_value, security_failure = _admit(
        security_decision="deny"
    )

    assert not binding_failure.admitted
    assert (
        PromptPlanAdmissionCode.IR_BINDING_MISMATCH.value
        in binding_failure.reason_codes
    )
    assert not binding_failure.receipt.final_plan_cid
    assert not binding_failure.receipt.final_task_cids
    assert not security_failure.admitted
    assert "ir.security_deny" in security_failure.reason_codes
    assert security_failure.ir_receipt is not None
    assert security_failure.ir_receipt.counterexamples


@pytest.mark.parametrize(
    ("mutate", "reason"),
    (
        (
            lambda task: replace(
                task,
                validations=(
                    PromptValidationRecord(
                        validation_key="validation:pytest",
                        argv=("bash", "-c", "pytest"),
                    ),
                ),
            ),
            PromptPlanAdmissionCode.SHELL_VALIDATION.value,
        ),
        (
            lambda task: replace(
                task,
                outputs=(
                    PromptOutputRecord(
                        path=(
                            "docs/architecture/"
                            "agent_supervisor_self_improvement.todo.md"
                        ),
                        effect="modify",
                        media_type="text/markdown",
                    ),
                ),
                predicted_files=(
                    "docs/architecture/"
                    "agent_supervisor_self_improvement.todo.md",
                ),
                scope_paths=("docs/architecture",),
            ),
            PromptPlanAdmissionCode.OUTPUT_FORBIDDEN.value,
        ),
        (
            lambda task: replace(
                task,
                resource_class="unbounded-provider",
            ),
            PromptPlanAdmissionCode.RESOURCE_INFEASIBLE.value,
        ),
    ),
)
def test_quality_output_resource_and_shell_failures_are_hard(
    mutate,
    reason: str,
) -> None:
    workflow, scan, graph = _graph_fixture()
    changed_task = mutate(graph.tasks[0])
    changed_graph = replace(graph, tasks=(changed_task,))
    ir_request = _ir_request(
        changed_graph, workflow, scan.dirty_worktree_root
    )

    result = admit_prompt_plan(
        changed_graph,
        repository_tree_id=scan.dirty_worktree_root,
        ir_request=ir_request,
        workflow_request=workflow,
        scan_receipt=scan,
    )

    assert not result.admitted
    assert reason in result.reason_codes
    assert not result.receipt.final_plan_cid


def test_stale_roots_unknown_closure_unbound_paths_and_missing_proofs_fail() -> None:
    workflow, scan, graph, ir_request, _result = _admit()
    stale = replace(
        ir_request,
        root_bindings=tuple(
            replace(item, observed="root:stale")
            if item.kind == "program"
            else item
            for item in ir_request.root_bindings
        ),
        mandatory_closure=None,
        proof_results=(),
    )
    task = graph.tasks[0]
    unbound_task = replace(
        task,
        outputs=(
            PromptOutputRecord(
                path="pkg/unseen.py",
                effect="create",
                media_type="text/x-python",
            ),
        ),
        predicted_files=("pkg/unseen.py",),
    )
    unbound_graph = replace(graph, tasks=(unbound_task,))
    unbound_ir = _ir_request(
        unbound_graph, workflow, scan.dirty_worktree_root
    )

    stale_result = admit_prompt_plan(
        graph,
        repository_tree_id=scan.dirty_worktree_root,
        ir_request=stale,
        workflow_request=workflow,
        scan_receipt=scan,
    )
    unbound_result = admit_prompt_plan(
        unbound_graph,
        repository_tree_id=scan.dirty_worktree_root,
        ir_request=unbound_ir,
        workflow_request=workflow,
        scan_receipt=scan,
    )

    assert PromptPlanAdmissionCode.MISSING_PROOF.value in stale_result.reason_codes
    assert (
        PromptPlanAdmissionCode.UNKNOWN_MANDATORY_STATE.value
        in stale_result.reason_codes
    )
    assert "ir.stale_root" in stale_result.reason_codes
    assert (
        PromptPlanAdmissionCode.EVIDENCE_UNTRACED.value
        in unbound_result.reason_codes
    )


def test_malformed_claimed_graph_identity_returns_an_exact_rejection() -> None:
    _workflow, scan, graph = _graph_fixture()
    payload = graph.to_record()
    payload["content_id"] = "baforeign"

    result = admit_prompt_plan(
        payload,
        repository_tree_id=scan.dirty_worktree_root,
    )

    assert not result.admitted
    assert result.reason_codes == (
        PromptPlanAdmissionCode.MALFORMED_GRAPH.value,
    )
    assert result.receipt.findings[0].counterexample["exception"]


def test_acceptance_granularity_conflicts_and_protected_subtrees_fail_closed() -> None:
    _workflow, scan, graph = _graph_fixture()
    task = graph.tasks[0]

    uncovered_task = replace(
        task,
        acceptance=(
            PromptAcceptanceRecord(
                criterion_key="criterion:uncovered",
                criterion="This criterion has no validation binding.",
                evidence_cids=task.evidence_cids,
                validation_keys=(),
            ),
        ),
    )
    uncovered = admit_prompt_plan(
        replace(graph, tasks=(uncovered_task,)),
        repository_tree_id=scan.dirty_worktree_root,
    )

    broad_outputs = tuple(
        PromptOutputRecord(
            path=f"pkg/generated_{index}.py",
            effect="create",
            media_type="text/x-python",
        )
        for index in range(17)
    )
    broad_task = replace(
        task,
        outputs=broad_outputs,
        predicted_files=tuple(item.path for item in broad_outputs),
    )
    broad = admit_prompt_plan(
        replace(graph, tasks=(broad_task,)),
        repository_tree_id=scan.dirty_worktree_root,
    )

    conflicting_task = replace(
        task,
        task_key="task:conflicting-writer",
        objective="Independently write the same output path.",
    )
    conflict = admit_prompt_plan(
        replace(graph, tasks=(task, conflicting_task)),
        repository_tree_id=scan.dirty_worktree_root,
    )

    protected = admit_prompt_plan(
        graph,
        repository_tree_id=scan.dirty_worktree_root,
        policy=PromptPlanAdmissionPolicy(protected_paths=("pkg",)),
    )

    assert (
        PromptPlanAdmissionCode.ACCEPTANCE_UNCOVERED.value
        in uncovered.reason_codes
    )
    assert PromptPlanAdmissionCode.TASK_TOO_BROAD.value in broad.reason_codes
    assert (
        PromptPlanAdmissionCode.CONFLICT_UNORDERED.value
        in conflict.reason_codes
    )
    assert PromptPlanAdmissionCode.OUTPUT_FORBIDDEN.value in protected.reason_codes


def test_validation_receipt_must_name_the_compiler_bound_structured_check() -> None:
    workflow, scan, graph, ir_request, _result = _admit()
    requirement = ir_request.validation_requirements[0]
    mismatched = replace(
        ir_request,
        validation_requirements=(
            replace(requirement, command="pytest some-other-target.py"),
        ),
    )

    result = admit_prompt_plan(
        graph,
        repository_tree_id=scan.dirty_worktree_root,
        ir_request=mismatched,
        workflow_request=workflow,
        scan_receipt=scan,
    )

    assert not result.admitted
    assert (
        PromptPlanAdmissionCode.IR_BINDING_MISMATCH.value
        in result.reason_codes
    )
    assert any(
        finding.path.endswith(".command")
        for finding in result.receipt.findings
    )


def test_receipt_rejects_tampered_determinism_claims() -> None:
    _workflow, _scan_receipt, _graph, _ir_request_value, result = _admit()
    payload = result.receipt.to_dict()
    payload["candidate_order_independent"] = False

    with pytest.raises(ValueError, match="candidate-order"):
        PromptPlanAdmissionReceipt.from_dict(payload)
