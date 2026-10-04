"""Bind current native repository queries to the unchanged symbolic control.

This preserves the declared task population and behavioral residuals. Repository
retrieval and trained latent vectors nominate context; they grant no authority.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .terminal_codebase_planning_snapshot import AUTHORITY, _plain
from .terminal_codebase_supervisor_fixture import _read

SCHEMA = "repository-indexed-planning-snapshot@1"


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return "sha256:" + hashlib.sha256(_wire(value)).hexdigest()


def current_signed_sources(manifest):
    """Read the exact signed forest; ambient files and Git HEAD are not inputs."""
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    declared, _, observed = local._manifest(manifest, initial=True)
    repository = Path(declared["repository"])
    sources = {path: _read(repository / path, 1_048_576) for path in sorted(observed)}
    if any(hashlib.sha256(raw).hexdigest() != observed[path]["sha256"]
           for path, raw in sources.items()):
        raise ValueError("signed source changed during repository capture")
    if len(declared["tasks"]) != 1:
        raise ValueError("indexed qualification requires the complete single declared task")
    return declared, observed, sources


def _reference(path, body, *, schema):
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("canonical independent planning artifact required")
    raw = _read(path, 32 * 1024 * 1024)
    if json.loads(raw) != body:
        raise ValueError("complete planning artifact body differs")
    return {"schema": schema, "path": str(path), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "canonical_body_sha256": _digest(body),
            "scope": "complete_current_context; no_behavioral_or_execution_authority"}


def _select_indexed_plan(*, bound, snapshot, materials, manifest, contract):
    """Run existing compiler/planner/critic against the indexed input identity."""
    from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import ObligationGraphCompiler
    from ipfs_accelerate_py.agent_supervisor.planning.symbolic_candidate_planner import SymbolicCandidatePlanner, SymbolicCandidateBounds
    from ipfs_accelerate_py.agent_supervisor.planning.plan_critic import PlanCritic
    from ipfs_accelerate_py.agent_supervisor.planning.intent_symbolic_planning import _critic_projection, _project_graph, _effect_rows
    from ipfs_accelerate_py.agent_supervisor.planning.formal_plan_compiler import FormalPlanCompiler
    from ipfs_accelerate_py.agent_supervisor.planning.formal_plan_validator import validate_formal_plan
    from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import check_intent_plan_coverage
    from ipfs_accelerate_py.agent_supervisor.planning.ir_logic_hooks import prepare_planning_context

    obligation = ObligationGraphCompiler().compile(bound.intent, current_facts=bound.current_facts,
        producers=bound.producers, task_candidates=bound.task_candidates, predicates=bound.predicates,
        current_root_id=bound.intent.current_root_id)
    if obligation.planning_blocked or obligation.review_required:
        raise ValueError("indexed administrative obligations require review")
    context = {**_plain(bound.candidate_context),
        "input_snapshot_cid": snapshot.snapshot_cid,
        "indexed_materials_sha256": _digest(_plain(bound.extra)),
        "repository_context_scope": "descriptive_and_conditional_only"}
    portfolio = SymbolicCandidatePlanner(bounds=SymbolicCandidateBounds(
        candidate_count=1, max_model_candidates=0, max_tasks_per_candidate=16)).plan(
            obligation, bound.frozen_goal, context, allow_model=False)
    expected_context = _plain(prepare_planning_context(context, domain=str(context["domain"])))
    if portfolio.selected is None or _plain(portfolio.request.context) != expected_context:
        raise ValueError("indexed symbolic portfolio failed or changed frozen input context")
    selected = portfolio.selected
    critic_plan = _critic_projection(materials, obligation, selected.symbolic_candidate.schedule.task_ids)
    critique = PlanCritic().critique(critic_plan, obligation_graph=obligation,
        required_goal_ids=obligation.root_obligation_ids,
        expected_effects=[row["effect_id"] for row in critic_plan["effects"]])
    if not critique.accepted:
        raise ValueError("indexed selection failed independent native critique")
    graph, bindings = _project_graph(materials, selected, manifest, contract["requirements"])
    coverage = check_intent_plan_coverage(contract, graph=graph, bindings=bindings)
    if not coverage["accepted"]:
        raise ValueError("indexed selection changed complete administrative requirement coverage")
    declared = manifest["payload"]
    compiled = FormalPlanCompiler().compile_prompt_graph(graph,
        repository_tree_id=bound.intent.current_root_id, actor_id=declared["profile_content_id"])
    if compiled.plan is None or compiled.status.value != "compiled" or compiled.proof_results:
        raise ValueError("indexed selection did not compile the native unproved formal plan")
    checked = validate_formal_plan(compiled.plan, compiled.formulas)
    if checked.status.value != "consistent":
        raise ValueError("indexed selection has inconsistent formal effects")
    task_keys = {task.task_cid: task.task_key for task in graph.tasks}
    effects = []
    for effect in compiled.plan.effects:
        source = effect.metadata["source_effect"]
        if (effect.task_id not in task_keys or effect.operation.value != "assign"
                or effect.fluent_id != "output:" + source["path"]):
            raise ValueError("indexed formal effect differs from the declared output")
        effects.append((task_keys[effect.task_id], source["path"], effect.value, source["media_type"]))
    if sorted(effects) != _effect_rows(materials.operations):
        raise ValueError("indexed compiler changed reviewed operation effects")
    receipt = {"schema": "repository-indexed-symbolic-planning-receipt@1", "accepted": True,
        "interpretation_scope": "administrative_requirement_task_coverage",
        "input_snapshot_cid": snapshot.snapshot_cid,
        "indexed_materials_sha256": context["indexed_materials_sha256"],
        "candidate_context_sha256": _digest(expected_context),
        "native_context_id": portfolio.request.context_id,
        "obligation_graph_id": obligation.graph_id, "portfolio_id": portfolio.portfolio_id,
        "selected_candidate_id": selected.symbolic_candidate.candidate_id,
        "selected_operation_ids": sorted(row["operation_id"] for row in materials.operations),
        "schedule": selected.symbolic_candidate.schedule.to_dict(),
        "critic_id": critique.critique_id, "critic_decision": critique.decision.value,
        "graph_cid": graph.content_id, "formal_plan_id": compiled.plan.plan_id,
        "formal_effects_sha256": _digest([list(row) for row in sorted(effects)]),
        "coverage_sha256": _digest(coverage), "provider_calls": 0,
        "observed_facts_supplied": 0, **AUTHORITY}
    return {"graph": graph, "requirement_bindings": bindings, "coverage": coverage, "receipt": receipt}


def build_indexed_repository_planning_snapshot(*, intent_experiment, manifest,
        workflow_request, control, proof_index_manifest, match_result,
        repository_index_manifest, repository_query, frozen_learning,
        learning_artifact):
    """Reconstruct native queries and learning before freezing actual materials."""
    from .terminal_codebase_repository_index import query_repository_index
    from .terminal_codebase_catalog_capture import capture_frozen_codebase_learning
    from .terminal_codebase_planning_snapshot import build_repository_proof_planning_snapshot
    from ipfs_accelerate_py.agent_supervisor.planning.intent_requirement_adapter import build_intent_planning_materials
    from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import (
        PlanCreateMaterials, freeze_plan_create_input_snapshot, plan_create_request_from_workflow,
    )
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import PromptWorkflowRequest

    # Detach caller-owned JSON before any native read or reconstruction. Later
    # query/domain checks and the frozen extra must use the same validated body.
    (manifest, workflow_request, control, proof_index_manifest, match_result,
     repository_index_manifest, repository_query, frozen_learning, learning_artifact) = (
        json.loads(_wire(_plain(value))) for value in (manifest, workflow_request, control,
            proof_index_manifest, match_result, repository_index_manifest, repository_query,
            frozen_learning, learning_artifact))
    declared, observed, sources = current_signed_sources(manifest)
    base = build_repository_proof_planning_snapshot(manifest=manifest,
        workflow_request=workflow_request, control=control,
        proof_index_manifest=proof_index_manifest, match_result=match_result)
    inputs = {"source_bytes": sources, "repository_id": declared["repository_cid"],
        "task_spec": declared["tasks"][0], "proof_index_manifest": proof_index_manifest,
        "expected": repository_index_manifest, "output": Path(repository_index_manifest["output"])}
    focus = match_result["query"]
    requested = {"source_path": focus["source_path"], "symbols": focus["symbols"],
                 "property": focus["property"], "limit": 12}
    rebuilt_query = query_repository_index(**inputs, query=requested)
    if _wire(rebuilt_query) != _wire(repository_query):
        raise ValueError("repository query differs from exact native current-source reconstruction")
    matched_ids = {row["entry_id"] for row in match_result["model_nominations"]}
    if set(repository_query["nominated_entry_ids"]) != matched_ids:
        raise ValueError("repository query nominations differ from qualified native intent/domain matching")
    index_ref = _reference(Path(repository_index_manifest["output"]) / "manifest.json",
        repository_index_manifest, schema="repository-native-index-artifact-reference@1")
    learning_ref = _reference(learning_artifact["path"], frozen_learning,
        schema="repository-frozen-learning-artifact-reference@1")
    if {key: learning_ref[key] for key in ("path", "bytes", "sha256")} != learning_artifact:
        raise ValueError("frozen learning artifact pin differs")
    expected_learning = capture_frozen_codebase_learning(
        intent_experiment=Path(intent_experiment), current_source_bytes=sources)
    if _wire(expected_learning) != _wire(frozen_learning):
        raise ValueError("frozen learning differs from exact source/model replay")
    extra = {**json.loads(_wire(base["snapshot"]["full_materials"])),
        "current_repository_index": index_ref,
        "current_repository_query": json.loads(_wire(repository_query)),
        "frozen_codebase_learning": learning_ref,
        "repository_context_scope": "source_bound_descriptive_and_conditional_context_only"}
    materials = build_intent_planning_materials(control["authored_control"]["contract"], manifest=manifest)
    bound = PlanCreateMaterials(intent=materials.intent, producers=materials.producers,
        task_candidates=materials.task_candidates, predicates=materials.predicates,
        current_facts=(), frozen_goal=materials.frozen_goal,
        candidate_context=materials.candidate_context, extra=extra)
    request = plan_create_request_from_workflow(PromptWorkflowRequest.from_dict(workflow_request),
        dirty_worktree_root=materials.intent.current_root_id,
        scope_paths=tuple(sorted({path for spec in declared["tasks"] for path in spec["scope_paths"]})))
    frozen = freeze_plan_create_input_snapshot(request, materials=bound)
    if request.budget.max_model_calls != 0 or not frozen.material_binding.get("reuse_supported", False):
        raise ValueError("indexed materials require exactly reusable model-disabled native planning")
    if frozen.snapshot_cid == base["snapshot"]["native_input_snapshot"]["snapshot_cid"]:
        raise ValueError("current repository index must change the actual planning input identity")
    proposed = _select_indexed_plan(bound=bound, snapshot=frozen, materials=materials,
        manifest=manifest, contract=control["authored_control"]["contract"])
    if proposed["graph"].content_id != base["snapshot"]["native_graph_cid"]:
        raise ValueError("indexed context changed the complete declared administrative graph")
    _, after, after_sources = current_signed_sources(manifest)
    if after != observed or after_sources != sources:
        raise ValueError("signed source forest changed while freezing indexed materials")
    if (index_ref != _reference(Path(repository_index_manifest["output"]) / "manifest.json",
            repository_index_manifest, schema="repository-native-index-artifact-reference@1")
            or learning_ref != _reference(learning_artifact["path"], frozen_learning,
                schema="repository-frozen-learning-artifact-reference@1")):
        raise ValueError("complete context artifact changed while freezing indexed materials")
    snapshot = {"schema": SCHEMA,
        "scope": "current_native_repository_context_for_separate_authored_administrative_control",
        "baseline_planning_snapshot_id": base["snapshot"]["snapshot_id"],
        "signed_manifest_sha256": _digest(_plain(manifest)),
        "current_source_inventory": observed,
        "repository_index_manifest_id": repository_index_manifest["manifest_id"],
        "repository_query_sha256": _digest(repository_query),
        "frozen_learning_sha256": _digest(frozen_learning),
        "full_materials": extra, "full_materials_sha256": _digest(extra),
        "native_input_snapshot": frozen.to_dict(),
        "native_graph_cid": proposed["graph"].content_id,
        "baseline_native_symbolic_receipt": base["symbolic_plan"]["receipt"],
        "native_symbolic_receipt": proposed["receipt"], "indexed_planner_executed": True,
        "current_behavioral_facts": [], "behavioral_satisfied_requirements": [],
        "complete_declared_task_population_preserved": True,
        "public_request_planned": False, "repository_evidence_admitted": False,
        "worker_launched": False, "provider_calls": 0, "training_steps": 0, **AUTHORITY}
    snapshot["snapshot_id"] = _digest(snapshot)
    return {"snapshot": snapshot, "symbolic_plan": proposed}


def replay_indexed_repository_planning_snapshot(*, expected, **inputs):
    observed = build_indexed_repository_planning_snapshot(**inputs)
    if _wire(observed["snapshot"]) != _wire(expected):
        raise ValueError("indexed planning snapshot differs from exact complete replay")
    return observed
