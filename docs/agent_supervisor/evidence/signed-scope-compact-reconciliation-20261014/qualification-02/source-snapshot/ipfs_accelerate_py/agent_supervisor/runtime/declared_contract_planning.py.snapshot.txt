"""Compile an exact signed task declaration without interpreting its prose.

This is mechanical goal/subgoal/task construction, not natural-language intent
formalization. The normal independent local admission still decides whether
the resulting candidate can be materialized and executed.
"""
from __future__ import annotations

from ..prompt.prompt_workflow import (
    LocalFallbackPolicy, PromptAcceptanceRecord, PromptGoalGraph,
    PromptGoalRecord, PromptOutputRecord, PromptTaskRecord,
    PromptValidationRecord, PromptWorkflowRequest,
)
from . import local_planning_admission as local

PLANNING_MODE = "declared-contract@1"
POLICY_ID = "terminal-declared-contract-planner@1"
SCHEMA = "supervisor-declared-contract-planning@1"


def build_declared_contract_plan(*, manifest):
    """Verify one signed task and compile its exact candidate skeleton.

    No provider, source execution, invented tests or inferred requirements are
    used. Source text remains an opaque signed input for the later coding task.
    This helper neither admits nor materializes the resulting candidate.
    """
    declared, _profile, sources = local._manifest(manifest, initial=True)
    if declared["schema"] not in {local.PLANNER_MANIFEST_SCHEMA, local.CREATE_MANIFEST_SCHEMA}:
        raise local.LocalPlanningError("declared-contract planning requires ordinary signed planning inputs")
    if "intent_requirements" in declared or "planning_inputs" not in declared:
        raise local.LocalPlanningError("declared-contract planning cannot replace Intent requirement coverage")
    request = PromptWorkflowRequest.from_dict(declared["planning_inputs"]["request"])
    policy = request.planning_policy
    if (policy.policy_id != POLICY_ID or policy.allow_model is not False
            or policy.fallback_policy is not LocalFallbackPolicy.DISABLED):
        raise local.LocalPlanningError("declared-contract mode differs from signed planning policy")
    if len(declared["tasks"]) != 1 or declared["tasks"][0]["dependencies"]:
        raise local.LocalPlanningError("declared-contract planning supports one independent signed task")
    if request.budget.max_goals < 2 or request.budget.max_tasks < 1:
        raise local.LocalPlanningError("signed budget cannot contain the declared skeleton")
    roots, evidence = local._verify_planning_inputs(declared)
    refs = tuple(row.evidence_cid for row in evidence)
    spec = declared["tasks"][0]
    acceptance = tuple(PromptAcceptanceRecord(**row) for row in spec["acceptance"])
    provenance = dict(planning_mode=PLANNING_MODE,
        interpretation_scope="exact_signed_task_structure_only",
        request_cid=request.request_cid, scan_cid=declared["planning_roots"]["scan_cid"],
        source_semantics_verified=False, semantic_alignment_verified=False,
        proof_authority=False, execution_authority=False, completion_authority=False)
    root = PromptGoalRecord(
        goal_key="TB-GOAL", parent_goal_cid="", dependency_goal_cids=(),
        title="Fulfill the signed public instruction",
        objective="Fulfill the complete public instruction under its exact signed task contract.",
        rationale="The administrator already declared the required scope, outputs and checks.",
        scope_paths=tuple(spec["scope_paths"]), acceptance=acceptance,
        evidence_cids=refs, provenance=provenance,
    )
    child = PromptGoalRecord(
        goal_key="TB-SUBGOAL", parent_goal_cid=root.goal_cid, dependency_goal_cids=(),
        title="Implement the declared task",
        objective="Implement the signed outputs and submit them to the pending declared checks.",
        rationale="Goal aggregation does not interpret the instruction or establish correctness.",
        scope_paths=tuple(spec["scope_paths"]), acceptance=acceptance,
        evidence_cids=refs, provenance=provenance,
    )
    task = PromptTaskRecord(
        task_key=spec["task_key"], goal_cid=child.goal_cid, dependency_task_cids=(),
        objective="Fulfill the complete signed public instruction; the structural check alone is not task success.",
        rationale="Preserve the exact independently declared task; implementation and validation remain pending.",
        scope_paths=tuple(spec["scope_paths"]),
        outputs=tuple(PromptOutputRecord(**row) for row in spec["outputs"]),
        validations=tuple(PromptValidationRecord(**row) for row in spec["validations"]),
        acceptance=acceptance, evidence_cids=refs, policy_roots=roots,
        predicted_files=tuple(row["path"] for row in spec["outputs"]),
        track="declared-contract-planning", resource_class="cpu-medium",
        fallback_behavior="fail_closed", provenance=provenance,
    )
    graph = PromptGoalGraph(**declared["planning_roots"], policy_roots=roots,
        goals=(root, child), tasks=(task,), evidence=evidence)
    if len(graph.canonical_bytes()) > request.budget.max_serialized_bytes:
        raise local.LocalPlanningError("declared skeleton exceeds signed graph byte budget")
    # Reuse the same formal compiler and consistency checks as admission. The
    # admission gate repeats these checks independently before materialization.
    compiled, validation, pending = local._graph_contract(graph, declared, local._tree(sources))
    return {"graph": graph, "receipt": dict(schema=SCHEMA, planning_mode=PLANNING_MODE,
        manifest_cid=local.content_identity(manifest), graph_cid=graph.content_id,
        request_cid=request.request_cid, scan_cid=declared["planning_roots"]["scan_cid"],
        source_tree_id=local._tree(sources), plan_id=compiled.plan.plan_id,
        goals=2, tasks=1, pending_requirements=len(pending),
        formal_plan_consistent=validation["status"] == "consistent",
        provider_calls=0, model_calls=0, source_obligations_discharged=0,
        instruction_interpreted=False, source_semantics_verified=False,
        semantic_alignment_verified=False, proof_authority=False,
        execution_authority=False, completion_authority=False,
        interpretation_scope="exact_signed_task_structure_only")}
