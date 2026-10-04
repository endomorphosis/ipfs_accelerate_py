"""Deterministic selection of reviewed requirement-to-task operations.

This joins the existing obligation compiler, symbolic candidate planner,
critic and formal prompt compiler. Its predicates mean administrative task
coverage. It does not assert that a software effect has occurred or that the
candidate interpretation of a source is true. Admission verifies the signed
manifest and source before consuming or replaying this pure proposal.
"""
from __future__ import annotations

from collections.abc import Mapping
from ..core.multiformats_identity import cid_for_dag_json
from ..proof.formal_verification_contracts import content_identity
from ..prompt.intent_plan_coverage import check_intent_plan_coverage
from ..prompt.prompt_workflow import (
    PromptAcceptanceRecord, PromptEvidenceRecord, PromptGoalGraph,
    PromptGoalRecord, PromptOutputRecord, PromptTaskRecord, PromptValidationRecord,
    PromptWorkflowRequest,
)
from .formal_plan_compiler import FormalPlanCompiler
from .formal_plan_validator import validate_formal_plan
from .intent_requirement_adapter import build_intent_planning_materials
from .obligation_graph_compiler import ObligationGraphCompiler
from .plan_critic import PlanCritic
from .symbolic_candidate_planner import SymbolicCandidateBounds, SymbolicCandidatePlanner

SCHEMA = "intent-symbolic-planning-receipt@1"
_AUTHORITY = {"semantic_alignment_verified": False, "proof_authority": False,
              "execution_authority": False, "completion_authority": False}


class IntentSymbolicPlanningError(ValueError):
    """An operation, selection or effect cannot be projected faithfully."""


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _effect_rows(operations):
    return sorted((operation["task_key"], output["path"], output["effect"], output["media_type"])
                  for operation in operations for output in operation["outputs"])


def _critic_projection(materials, obligation, selected_ids):
    candidates = {item.candidate_id: item for item in obligation.task_candidates}
    operations = {materials.operation_candidate_ids[item["operation_id"]]: item
                  for item in materials.operations}
    tasks, effects = [], []
    for candidate_id in selected_ids:
        operation = operations[candidate_id]
        candidate = candidates[candidate_id]
        tasks.append({"task_id": candidate_id, "action_id": candidate_id,
                      "depends_on": list(candidate.depends_on_candidate_ids),
                      "closes_obligation_ids": list(candidate.closes_obligation_ids),
                      "outputs": [row["path"] for row in operation["outputs"]],
                      "paths": [row["path"] for row in operation["outputs"]],
                      "resource_class": "cpu-medium", "estimated_duration_ms": 1000})
        for output in operation["outputs"]:
            effects.append({"effect_id": cid_for_dag_json({"operation": operation["operation_id"],
                                                         "output": _plain(output)}),
                            "task_id": candidate_id, "operation": "assign",
                            "target_id": output["path"], "value": output["effect"],
                            "media_type": output["media_type"]})
    return {"plan_id": cid_for_dag_json({"intent": materials.intent.to_dict(), "tasks": tasks}),
            "tasks": tasks, "effects": effects}


def _project_graph(materials, selected, manifest, requirements):
    declared = manifest["payload"]
    specs = {item["task_key"]: item for item in declared["tasks"]}
    operations = {materials.operation_candidate_ids[item["operation_id"]]: item
                  for item in materials.operations}
    schedule = selected.symbolic_candidate.schedule
    if set(schedule.task_ids) != set(operations):
        raise IntentSymbolicPlanningError("symbolic selection differs from the signed exact task population")
    policy_roots, evidence = ((content_identity(declared["policy"]),), ())
    if "planning_inputs" in declared:
        inputs = declared["planning_inputs"]
        request = PromptWorkflowRequest.from_dict(inputs["request"])
        policy_roots = tuple(sorted({request.policy_root,
            *(content_identity(row) for row in inputs["domain_declarations"].values())}))
        evidence = tuple(PromptEvidenceRecord.from_dict(row) for row in inputs["selected_evidence"])
    refs = tuple(row.evidence_cid for row in evidence)
    acceptances = {}
    for spec in specs.values():
        for row in spec["acceptance"]:
            acceptance = PromptAcceptanceRecord(**row)
            existing = acceptances.get(acceptance.criterion_key)
            if existing is not None and existing != acceptance:
                raise IntentSymbolicPlanningError("conflicting signed aggregation acceptance keys")
            acceptances[acceptance.criterion_key] = acceptance
    scope = tuple(sorted({path for spec in specs.values() for path in spec["scope_paths"]}))
    provenance = {"operation_contract_cid": cid_for_dag_json(_plain(materials.operation_contract)),
                  "interpretation_scope": "administrative_requirement_task_coverage", **_AUTHORITY}
    root = PromptGoalRecord(
        goal_key="INTENT-GOAL", parent_goal_cid="", dependency_goal_cids=(),
        title="Plan the reviewed public requirements", objective="Cover the signed public task requirements",
        rationale="Selected by the existing symbolic obligation planner",
        scope_paths=scope, acceptance=tuple(acceptances.values()), evidence_cids=refs,
        provenance=provenance,
    )
    child = PromptGoalRecord(
        goal_key="INTENT-SUBGOAL", parent_goal_cid=root.goal_cid, dependency_goal_cids=(),
        title="Execute the selected task contracts", objective="Run the exact signed operations and acceptance checks",
        rationale="Public acceptance remains pending execution", scope_paths=scope,
        acceptance=tuple(acceptances.values()), evidence_cids=refs, provenance=provenance,
    )
    tasks, by_candidate, bindings = [], {}, []
    expected_edges = set()
    for wave in schedule.waves:
        for candidate_id in wave:
            operation = operations[candidate_id]
            spec = specs[operation["task_key"]]
            dependencies = tuple(materials.operation_candidate_ids[name]
                                 for name in operation["dependency_operation_ids"])
            if not set(dependencies) <= set(by_candidate):
                raise IntentSymbolicPlanningError("symbolic schedule violates signed operation ordering")
            expected_edges.update((dependency, candidate_id) for dependency in dependencies)
            task = PromptTaskRecord(
                task_key=operation["task_key"], goal_cid=child.goal_cid,
                dependency_task_cids=tuple(by_candidate[name].task_cid for name in dependencies),
                objective="Cover reviewed requirements for " + operation["operation_id"],
                rationale="Use the independently signed operation, output effects and validation commands",
                scope_paths=tuple(spec["scope_paths"]),
                outputs=tuple(PromptOutputRecord(**row) for row in spec["outputs"]),
                validations=tuple(PromptValidationRecord(**row) for row in spec["validations"]),
                acceptance=tuple(PromptAcceptanceRecord(**row) for row in spec["acceptance"]),
                evidence_cids=refs, policy_roots=policy_roots, track="intent-symbolic",
                predicted_files=tuple(row["path"] for row in spec["outputs"]),
                provenance={**provenance, "operation_id": operation["operation_id"],
                            "requirement_ids": list(materials.candidate_requirement_ids[candidate_id])},
            )
            tasks.append(task)
            by_candidate[candidate_id] = task
            for requirement_id in materials.candidate_requirement_ids[candidate_id]:
                grounding = next(row for row in requirements if row["requirement_id"] == requirement_id)
                bindings.append({"requirement_id": requirement_id, "task_keys": [task.task_key],
                                 "validation_keys": grounding["validation_keys"]})
    if set(schedule.dependency_edges) != expected_edges:
        raise IntentSymbolicPlanningError("symbolic schedule edges differ from the signed operation dependencies")
    graph = PromptGoalGraph(
        **declared["planning_roots"], policy_roots=policy_roots,
        goals=(root, child), tasks=tuple(tasks), evidence=evidence,
    )
    return graph, sorted(bindings, key=lambda row: row["requirement_id"])


def build_intent_symbolic_plan(contract, *, manifest, applicability_timeout_seconds=None,
                               source_applicability_nomination=None):
    """Produce a deterministic graph and replayable nomination receipt.

    The caller must verify the manifest/source independently. Replaying this
    function uses the signed baseline; v3 also replays the captured nomination
    and actual bounded checker. It performs no neural inference or provider calls.
    """
    materials = build_intent_planning_materials(contract, manifest=manifest,
        applicability_timeout_seconds=applicability_timeout_seconds,
        source_applicability_nomination=source_applicability_nomination)
    obligation = ObligationGraphCompiler().compile(
        materials.intent, current_facts=materials.current_facts, producers=materials.producers,
        task_candidates=materials.task_candidates, predicates=materials.predicates,
        current_root_id=materials.intent.current_root_id,
    )
    if obligation.planning_blocked or obligation.review_required:
        error = IntentSymbolicPlanningError("symbolic obligations require review or are blocked")
        error.symbolic_issues = [row.to_dict() for row in obligation.issues]
        raise error
    portfolio = SymbolicCandidatePlanner(bounds=SymbolicCandidateBounds(
        candidate_count=1, max_model_candidates=0, max_tasks_per_candidate=16,
    )).plan(obligation, materials.frozen_goal, materials.candidate_context, allow_model=False)
    if portfolio.selected is None:
        raise IntentSymbolicPlanningError("no symbolic candidate passed the existing planner gates")
    selected = portfolio.selected
    selected_ids = selected.symbolic_candidate.schedule.task_ids
    critic_plan = _critic_projection(materials, obligation, selected_ids)
    critique = PlanCritic().critique(
        critic_plan, obligation_graph=obligation,
        required_goal_ids=obligation.root_obligation_ids,
        expected_effects=[row["effect_id"] for row in critic_plan["effects"]],
    )
    if not critique.accepted:
        error = IntentSymbolicPlanningError("symbolic task projection failed independent critique: " + str(critique.decision.value))
        error.symbolic_issues = [row.to_dict() for row in critique.findings]
        raise error
    graph, bindings = _project_graph(materials, selected, manifest, contract["requirements"])
    coverage = check_intent_plan_coverage(contract, graph=graph, bindings=bindings)
    if not coverage["accepted"]:
        error = IntentSymbolicPlanningError("symbolic task projection does not cover its requirements")
        error.requirement_coverage = coverage
        raise error
    declared = manifest["payload"]
    compiled = FormalPlanCompiler().compile_prompt_graph(
        graph, repository_tree_id=materials.intent.current_root_id,
        actor_id=declared["profile_content_id"],
    )
    if compiled.plan is None or compiled.status.value != "compiled" or compiled.proof_results:
        raise IntentSymbolicPlanningError("symbolic projection did not compile an unproved local formal plan")
    validation = validate_formal_plan(compiled.plan, compiled.formulas)
    if validation.status.value != "consistent":
        raise IntentSymbolicPlanningError("symbolic projection compiled inconsistent formal effects")
    keys = {task.task_cid: task.task_key for task in graph.tasks}
    formal_effects = []
    for effect in compiled.plan.effects:
        source = effect.metadata["source_effect"]
        if effect.task_id not in keys or effect.operation.value != "assign" or effect.fluent_id != "output:" + source["path"]:
            raise IntentSymbolicPlanningError("formal effect is not an exact declared output assignment")
        formal_effects.append((keys[effect.task_id], source["path"], effect.value, source["media_type"]))
    if sorted(formal_effects) != _effect_rows(materials.operations):
        raise IntentSymbolicPlanningError("compiler effects differ from reviewed operation outputs")
    # Snapshot freezes full semantic material values, including the concrete
    # operations; only its digest enters this proof-numeric-safe receipt.
    input_snapshot_cid = ""
    if "planning_inputs" in declared:
        from ..prompt.plan_create_service import (
            PlanCreateMaterials, freeze_plan_create_input_snapshot, plan_create_request_from_workflow,
        )

        request = PromptWorkflowRequest.from_dict(declared["planning_inputs"]["request"])
        create_request = plan_create_request_from_workflow(
            request, dirty_worktree_root=materials.intent.current_root_id,
            scope_paths=tuple(sorted({path for spec in declared["tasks"] for path in spec["scope_paths"]})),
        )
        bound = PlanCreateMaterials(
            intent=materials.intent, producers=materials.producers,
            task_candidates=materials.task_candidates, predicates=materials.predicates,
            current_facts=materials.current_facts, frozen_goal=materials.frozen_goal,
            candidate_context=materials.candidate_context,
            extra={"operation_contract": _plain(materials.operation_contract)},
        )
        input_snapshot_cid = freeze_plan_create_input_snapshot(create_request, materials=bound).snapshot_cid
    receipt = {
        "schema": SCHEMA, "accepted": True,
        "interpretation_scope": "administrative_requirement_task_coverage",
        "contract_cid": materials.contract_cid,
        "operation_contract_cid": cid_for_dag_json(_plain(materials.operation_contract)),
        "materials_cid": cid_for_dag_json(materials.to_dict()),
        "input_snapshot_cid": input_snapshot_cid,
        "obligation_graph_id": obligation.graph_id, "portfolio_id": portfolio.portfolio_id,
        "selected_candidate_id": selected.symbolic_candidate.candidate_id,
        "selected_operation_ids": sorted(row["operation_id"] for row in materials.operations),
        "schedule": selected.symbolic_candidate.schedule.to_dict(),
        "critic_id": critique.critique_id, "critic_decision": critique.decision.value,
        "graph_cid": graph.content_id, "formal_plan_id": compiled.plan.plan_id,
        "formal_effects_cid": cid_for_dag_json([list(row) for row in sorted(formal_effects)]),
        "coverage_cid": cid_for_dag_json(coverage), "provider_calls": 0,
        "observed_facts_supplied": len(materials.current_facts), "source_semantics_verified": False, **_AUTHORITY,
    }
    if materials.source_applicability is not None:
        receipt["source_applicability_nomination"] = _plain(source_applicability_nomination)
        receipt["source_applicability"] = _plain(materials.source_applicability)
    return {"graph": graph, "requirement_bindings": bindings,
            "coverage": coverage, "receipt": receipt}


__all__ = ["SCHEMA", "IntentSymbolicPlanningError", "build_intent_symbolic_plan"]
