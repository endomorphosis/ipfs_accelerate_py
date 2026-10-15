"""Complete proposal-only binding between repository preview and successor tasks.

This pure companion keeps the two request/scan schemas distinct. It checks
administrative requirement coverage, never source semantics, signatures,
native currentness, checker execution, materialization, or worker permission.
Every native use still needs the separate receiving owner and closing fences.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
import gc
import math
from types import MappingProxyType

from ipfs_datasets_py.logic.software_contracts.content import validate_cid

from ..planning.adaptive_planner import FrozenPlanningGoal
from ..planning.obligation_graph_compiler import (
    InvalidationSelector, InvalidationSelectorKind, ObligationGraphCompiler,
    PredicatePolarity, ProducerRule, SemanticSupport, TaskCandidate, TypedIntent, TypedPredicate,
    obligation_id_for_predicate,
)
from ..planning.plan_evaluator import EvidenceAwarePlanPolicy
from ..planning.plan_revision_contracts import (
    DirtyTreePolicy, FallbackPolicy, PlanAuthorityRoots, PlanCreateRequest,
    PlanRequestBudget, TaskSourceKind,
)
from ..planning.structural_codebase_context import StructuralCodebaseContext
from ..proof.formal_verification_contracts import content_identity
from ..prompt.plan_create_service import (
    PlanCreateMaterials, PlanCreateMode, PlanCreatePreviewReceipt, PlanCreateStage, PlanCreateStageResult,
    PlanCreateVerdict, _assert_body_free, freeze_plan_create_input_snapshot,
)
from ..prompt.prompt_workflow import PromptGoalGraph
from .codebase_successor_dispatch_context import (
    AUTHORITY_NAMES, DECLARATION_SCHEMA, _DECLARATION_FIELDS, _model_pair, _wire,
)

SCHEMA = "supervisor-preview-successor-bridge@1"
MAX_BRIDGE_BYTES = 2 * 1024 * 1024
_PROFILE = "repository-live-root-observation@1"
_RESERVED_CONTEXT = frozenset({"request_cid", "input_snapshot_cid", "mode",
    "bounds_digest", "scope_paths", "structural_codebase_context_cid", "structural_codebase"})
_RESERVED_EXTRA = frozenset({"structural_codebase_context_cid", "structural_codebase"})
_PASSIVE_ENUM_TYPES = {PredicatePolarity, SemanticSupport, InvalidationSelectorKind,
    DirtyTreePolicy, FallbackPolicy, TaskSourceKind, PlanCreateMode, PlanCreateStage, PlanCreateVerdict}
_PASSIVE_RECORD_TYPES = {TypedIntent, TypedPredicate, InvalidationSelector, ProducerRule,
    TaskCandidate, FrozenPlanningGoal, EvidenceAwarePlanPolicy, PlanCreateRequest,
    PlanAuthorityRoots, PlanRequestBudget, PlanCreatePreviewReceipt, PlanCreateStageResult}


class PreviewSuccessorContractError(ValueError):
    """Complete explicit preview/task bindings cannot be reconstructed."""


def _need(condition, message):
    if not condition:
        raise PreviewSuccessorContractError(message)


def _passive(value, *, records=False, depth=0, budget=None):
    """Inspect closed builtin data without evaluating arbitrary record methods.

    A mapping proxy can wrap a user-defined mapping. Inspect its referent
    first, so an exact proxy type alone cannot hide live mapping callbacks.
    """
    budget = [100_000] if budget is None else budget
    budget[0] -= 1
    _need(depth <= 48 and budget[0] >= 0, "bounded passive input population required")
    kind = type(value)
    if kind in {str, int, bool, type(None)}:
        _need(kind is not int or value.bit_length() <= 128, "bounded passive integer required")
        _need(kind is not str or len(value) <= MAX_BRIDGE_BYTES, "bounded passive string required")
        return
    if kind is float:
        _need(math.isfinite(value), "finite passive metadata numbers required")
        return
    if records and kind in _PASSIVE_ENUM_TYPES:
        attributes = object.__getattribute__(value, "__dict__")
        _need(type(attributes) is dict and set(attributes)
              <= {"_value_", "_name_", "__objclass__", "_sort_order_"},
              "passive enum cannot override instance behavior")
        _passive(object.__getattribute__(value, "_value_"), depth=depth + 1, budget=budget)
        return
    if kind in {dict, MappingProxyType}:
        source = value
        if kind is MappingProxyType:
            referents = gc.get_referents(value)
            _need(len(referents) == 1 and type(referents[0]) is dict,
                  "passive mapping proxy must wrap an exact dictionary")
            source = referents[0]
        _need(len(source) <= budget[0] and all(type(key) is str for key in source),
              "exact passive dictionary keys required")
        for key, item in source.items():
            _passive(key, depth=depth + 1, budget=budget)
            _passive(item, records=records, depth=depth + 1, budget=budget)
        return
    if kind in {tuple, list}:
        _need(len(value) <= budget[0], "bounded exact passive sequence required")
        for item in value:
            _passive(item, records=records, depth=depth + 1, budget=budget)
        return
    if records and kind in _PASSIVE_RECORD_TYPES:
        members = fields(kind)
        attributes = object.__getattribute__(value, "__dict__")
        _need(type(attributes) is dict and set(attributes) == {member.name for member in members},
              "passive typed record cannot override instance behavior")
        for member in members:
            # Metadata admits no records, even approved records with methods.
            nested_records = member.name not in {"metadata", "redacted_source_metadata"}
            _passive(object.__getattribute__(value, member.name), records=nested_records,
                     depth=depth + 1, budget=budget)
        return
    raise PreviewSuccessorContractError("exact passive builtin data and known typed records required")


def _passive_materials(materials):
    _need(type(materials) is PlanCreateMaterials, "exact original materials required")
    members = fields(PlanCreateMaterials)
    attributes = object.__getattribute__(materials, "__dict__")
    _need(type(attributes) is dict and set(attributes) == {member.name for member in members},
          "passive materials cannot override instance behavior")
    _need(object.__getattribute__(materials, "query_plan") is None
          and object.__getattribute__(materials, "workflow_request") is None,
          "structural bridge cannot accept query or workflow stages")
    budget = [100_000]
    for member in members:
        _passive(object.__getattribute__(materials, member.name), budget=budget,
            records=member.name in {"intent", "frozen_goal", "producers", "task_candidates", "predicates"})
    _need(type(materials.intent) is TypedIntent
          and type(materials.intent.desired_predicates) in {list, tuple}
          and all(type(item) is TypedPredicate for item in materials.intent.desired_predicates)
          and type(materials.frozen_goal) is FrozenPlanningGoal
          and type(materials.frozen_goal.policy) is EvidenceAwarePlanPolicy,
          "exact typed desired predicates and evaluator policy required")


def require_passive_repository_preview_inputs(request, materials):
    """Check inputs before snapshot reads, including after native fence closure."""
    _need(type(request) is PlanCreateRequest and type(request.roots) is PlanAuthorityRoots
          and type(request.budget) is PlanRequestBudget,
          "exact typed create request, authority roots and request budget required")
    _passive(request, records=True)
    _passive_materials(materials)


def require_passive_repository_preview_receipt(preview):
    """Reject nested receipt serializers before post-fence custody reads."""
    _need(type(preview) is PlanCreatePreviewReceipt, "exact typed preview receipt required")
    _passive(preview, records=True)


def _plain(value, *, depth=0, budget=None):
    budget = [100_000] if budget is None else budget
    budget[0] -= 1
    _need(depth <= 48 and budget[0] >= 0, "bounded bridge structure required")
    if isinstance(value, Mapping):
        _need(len(value) <= budget[0] and all(type(key) is str for key in value),
              "bounded exact bridge keys required")
        return {key: _plain(item, depth=depth + 1, budget=budget) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        _need(len(value) <= budget[0], "bounded bridge array required")
        return [_plain(item, depth=depth + 1, budget=budget) for item in value]
    _need(type(value) in {str, int, bool, type(None)}, "integer-only bridge values required")
    return value


def _bytes(value):
    return _wire(value, max_bytes=MAX_BRIDGE_BYTES)


def _cid(value, name):
    _need(type(value) is str, "exact " + name + " required")
    validate_cid(value, codecs={"dag-json"})
    return value


def _rows(value, fields, name):
    _need(isinstance(value, Sequence) and not isinstance(value, (str, bytes))
          and 1 <= len(value) <= 16, "bounded nonempty " + name + " required")
    result = []
    for row in value:
        _need(isinstance(row, Mapping) and set(row) == fields, "exact " + name + " row required")
        result.append(_plain(row))
    _bytes(result)
    return result


def _identities(value, name):
    _need(type(value) is list and value and len(value) <= 256
          and all(type(item) is str and item for item in value), "nonempty " + name + " required")
    _need(len(value) == len(set(value)), "duplicate " + name)
    return sorted(value)


def bind_repository_preview_materials(materials, structural_context):
    """Reconstruct the exact structural-preview augmentation from original inputs."""
    _need(type(materials) is PlanCreateMaterials and type(structural_context) is StructuralCodebaseContext,
          "exact original materials and structural context required")
    _passive_materials(materials)
    _need(materials.current_roots is None and not materials.current_facts
          and materials.obligation_graph is None and materials.evidence_bundle is None
          and materials.admission_materials is None and materials.model_provider is None
          and not materials.evidence_adapters and not materials.evidence_queries
          and materials.scan is None and materials.parallel_request is None
          and materials.parallel_tasks is None,
          "structural bridge cannot accept facts, static roots or injected stages")
    _need(type(materials.intent) is TypedIntent and type(materials.frozen_goal) is FrozenPlanningGoal
          and not materials.frozen_goal.policy.require_proof
          and type(materials.candidate_context) in {dict, MappingProxyType}
          and type(materials.extra) in {dict, MappingProxyType}
          and not set(materials.candidate_context) & _RESERVED_CONTEXT
          and not set(materials.extra) & _RESERVED_EXTRA,
          "explicit model-off original structural materials required")
    _need(materials.extra.get("root_observation_profile", _PROFILE) == _PROFILE,
          "structural root observation profile differs")
    for rows, cls, maximum, name in (
        (materials.task_candidates, TaskCandidate, 16, "task candidates"),
        (materials.producers, ProducerRule, 256, "producers"),
        (materials.predicates, TypedPredicate, 256, "predicates"),
    ):
        _need(type(rows) in {list, tuple}
              and len(rows) <= maximum and all(type(item) is cls for item in rows),
              "complete typed " + name + " required")
    _need(materials.task_candidates and len(materials.intent.desired_predicates) <= 16,
          "bounded complete task and predicate populations required")
    record = structural_context.to_dict()
    return replace(materials,
        scan={"scan_cid": structural_context.cid, "structural_codebase": record},
        candidate_context={**dict(materials.candidate_context),
            "structural_codebase_context_cid": structural_context.cid, "structural_codebase": record},
        extra={**dict(materials.extra), "structural_codebase_context_cid": structural_context.cid,
            "structural_codebase": record, "root_observation_profile": _PROFILE})


def build_preview_successor_bridge(*, request, materials, preview, structural_context, graph,
        successor_declaration, admission_cid, candidate_bindings, requirement_bindings,
        checker_policy_cid, training_record_cid):
    """Rebuild every proposal binding; supplied identities alone grant no authority."""
    _need(type(request) is PlanCreateRequest and type(preview) is PlanCreatePreviewReceipt
          and type(graph) is PromptGoalGraph, "exact typed create request, preview and task graph required")
    require_passive_repository_preview_inputs(request, materials)
    bound = bind_repository_preview_materials(materials, structural_context)
    # Constructors verify claimed record identities; preserve the full wire,
    # including fields excluded from another schema's content identity.
    request_wire, preview_wire, graph_wire = request.to_dict(), preview.to_dict(), graph.to_dict()
    _need(_bytes(PlanCreateRequest.from_dict(request_wire).to_dict()) == _bytes(request_wire)
          and _bytes(PlanCreatePreviewReceipt.from_dict(preview_wire).to_dict()) == _bytes(preview_wire)
          and _bytes(PromptGoalGraph.from_dict(graph_wire).to_dict()) == _bytes(graph_wire),
          "canonical typed preview inputs required")
    head = structural_context.head
    roots = request.roots
    _need(request.repository_id == head.repository_id
          and roots.repository_root_cid == head.snapshot_cid
          and roots.dirty_worktree_root == head.snapshot_cid
          and roots.program_root == structural_context.semantic_state_cid == graph.program_root
          and materials.intent.current_root_id == head.snapshot_cid
          and materials.frozen_goal.repository_tree_id == head.snapshot_cid,
          "preview source, head and program roots differ")
    _need(preview.request_cid == request.request_cid
          and _bytes(preview.roots.to_dict()) == _bytes(roots.to_dict())
          and preview.scan_cid == structural_context.cid
          and preview.mode is PlanCreateMode.DETERMINISTIC
          and preview.verdict in {PlanCreateVerdict.REVIEW_ONLY, PlanCreateVerdict.ADMITTED}
          and preview.read_only is True and not preview.wrote_effects,
          "read-only deterministic structural preview bindings differ")
    snapshot = freeze_plan_create_input_snapshot(request, materials=bound)
    binding = snapshot.material_binding
    _need(binding and binding["reuse_supported"] is True and not binding["unsupported_fields"]
          and snapshot.to_dict()["schema"].endswith("@2")
          and preview.input_snapshot_cid == snapshot.snapshot_cid,
          "complete supported material-bound preview snapshot required")
    _need(set(materials.frozen_goal.policy.acceptance_criteria)
          == set(materials.intent.goal_predicate_ids), "frozen policy omits or adds desired predicates")
    candidates = _rows(candidate_bindings, {"candidate_id", "task_cid"}, "candidate bindings")
    requirements = _rows(requirement_bindings,
        {"predicate_id", "task_cids", "acceptance_cids"}, "requirement bindings")
    declaration = _plain(successor_declaration)
    _need(type(declaration) is dict and set(declaration) == _DECLARATION_FIELDS
          and declaration["schema"] == DECLARATION_SCHEMA,
          "closed successor declaration required")
    _need(type(declaration["authority"]) is dict and set(declaration["authority"]) == AUTHORITY_NAMES
          and all(flag is False for flag in declaration["authority"].values()),
          "successor declaration cannot acquire authority")
    _bytes({"request": request_wire, "preview": preview_wire, "graph": graph_wire,
            "successor": declaration, "candidate_bindings": candidates, "requirement_bindings": requirements})
    from ipfs_datasets_py.logic.software_contracts.codebase_inventory_resume import _head
    previous, current = _head(declaration["previous_head"]), _head(declaration["current_head"])
    _need(current == head and previous.repository_id == head.repository_id
          and current.generation == previous.generation + 1, "exact immediate successor head required")
    _model_pair(declaration["previous_model"], declaration["model"])
    for name in ("full_context_cid", "selection_cid", "source_delta_cid", "previous_membership_cid",
                 "current_membership_cid", "root_cid", "completion_cid", "inventory_context_cid"):
        _cid(declaration[name], name)
    for value, name in ((admission_cid, "admission CID"), (checker_policy_cid, "checker policy CID"),
                        (training_record_cid, "training record CID")):
        _cid(value, name)
    compiled = ObligationGraphCompiler().compile(materials.intent, current_facts=(),
        producers=materials.producers, task_candidates=materials.task_candidates,
        predicates=materials.predicates, current_root_id=roots.dirty_worktree_root)
    _need(compiled.ready and not compiled.uncovered_leaf_obligation_ids
          and compiled.graph_id == preview.obligation_graph_cid,
          "preview does not bind the complete ready compiled obligation graph")
    stages = {item.stage: item for item in preview.stage_results}
    _need(len(stages) == len(preview.stage_results)
          and all(stage in stages and stages[stage].passed is True
              and not stages[stage].blockers and stages[stage].artifact_cid == cid
              for stage, cid in ((PlanCreateStage.SCAN, structural_context.cid),
                                 (PlanCreateStage.OBLIGATION, compiled.graph_id))),
          "preview structural stages differ from compiled bindings")
    authored = {item.candidate_id: item for item in materials.task_candidates}
    compiled_candidates = {item.candidate_id: item for item in compiled.task_candidates}
    tasks = {item.task_cid: item for item in graph.tasks}
    by_candidate = {}
    for row in candidates:
        _need(type(row["candidate_id"]) is str and type(row["task_cid"]) is str
              and row["candidate_id"] not in by_candidate,
              "duplicate or malformed candidate binding")
        by_candidate[row["candidate_id"]] = row["task_cid"]
    _need(len(authored) == len(materials.task_candidates)
          and set(authored) == set(compiled_candidates) == set(by_candidate)
          and len(set(by_candidate.values())) == len(by_candidate)
          and set(by_candidate.values()) == set(tasks),
          "every authored candidate and signed task needs one exact binding")
    nodes = {item.obligation_id: item for item in compiled.nodes}
    desired = set(materials.intent.goal_predicate_ids)
    required_tasks = {predicate: set() for predicate in desired}
    for key, candidate in compiled_candidates.items():
        task = tasks[by_candidate[key]]
        _need(set(task.dependency_task_cids)
              == {by_candidate[dependency] for dependency in candidate.depends_on_candidate_ids}
              and candidate.depends_on_candidate_ids == authored[key].depends_on_candidate_ids,
              "candidate and signed task dependencies differ")
        pending, visited = list(candidate.closes_obligation_ids), set()
        while pending:
            obligation = pending.pop()
            if obligation in visited:
                continue
            _need(obligation in nodes, "candidate references an unknown compiled obligation")
            visited.add(obligation)
            node = nodes[obligation]
            pending.extend(node.parent_obligation_ids)
        matched = {predicate for predicate in desired
                   if obligation_id_for_predicate(predicate) in visited}
        _need(matched, "candidate is unrelated to every desired predicate")
        for predicate in matched:
            required_tasks[predicate].add(task.task_cid)
    by_predicate = {}
    for row in requirements:
        predicate = row["predicate_id"]
        _need(type(predicate) is str and predicate in desired and predicate not in by_predicate,
              "duplicate or foreign desired predicate binding")
        selected_tasks = _identities(row["task_cids"], "requirement task CIDs")
        acceptance = _identities(row["acceptance_cids"], "requirement acceptance CIDs")
        _need(set(selected_tasks) == required_tasks[predicate],
              "predicate task coverage differs from compiled producer closure")
        expected_acceptance = {item.content_id for task_cid in selected_tasks
                               for item in tasks[task_cid].acceptance}
        _need(set(acceptance) == expected_acceptance,
              "predicate binding omits or adds actual signed task acceptance")
        by_predicate[predicate] = {"predicate_id": predicate, "task_cids": selected_tasks,
                                   "acceptance_cids": acceptance}
    _need(set(by_predicate) == desired, "every desired predicate needs complete explicit coverage")
    result = {"schema": SCHEMA,
        "interpretation_scope": "administrative_preview_requirement_task_coverage",
        "request": request_wire, "input_snapshot": snapshot.to_dict(), "preview": preview_wire,
        "structural_context": structural_context.to_dict(),
        "structural_context_cid": structural_context.cid,
        "graph": graph_wire, "graph_cid": graph.content_id,
        "admission_cid": admission_cid, "successor_declaration": declaration,
        "successor_declaration_cid": content_identity(declaration), "training_record_cid": training_record_cid,
        "checker_policy_cid": checker_policy_cid,
        "evaluator_policy_digest": materials.frozen_goal.policy_digest,
        "compiled_obligation_graph_cid": compiled.graph_id,
        "compiled_obligation_graph_digest": content_identity(compiled.to_dict()),
        "root_mapping": {"create_request_cid": request.request_cid,
            "workflow_request_cid": graph.request_cid, "preview_scan_cid": preview.scan_cid,
            "workflow_scan_cid": graph.scan_cid, "program_root": graph.program_root,
            "preview_plan_root": preview.plan_root_cid, "signed_graph_cid": graph.content_id},
        "candidate_bindings": [{"candidate_id": key, "task_cid": by_candidate[key]}
                               for key in sorted(by_candidate)],
        "requirement_bindings": [by_predicate[key] for key in sorted(by_predicate)],
        "population": {"desired_predicates": len(desired), "authored_candidates": len(authored),
            "compiled_candidates": len(compiled_candidates), "signed_tasks": len(tasks),
            "signed_goals": len(graph.goals), "task_acceptance": sum(len(task.acceptance) for task in graph.tasks),
            "task_validations": sum(len(task.validations) for task in graph.tasks),
            "task_outputs": sum(len(task.outputs) for task in graph.tasks)},
        "authority": {name: False for name in sorted(AUTHORITY_NAMES)},
        "checker_executed": False, "signatures_verified_here": False,
        "native_currentness_verified_here": False, "worker_launched": False,
        "production_admitted": False}
    _assert_body_free(result, "preview successor bridge")
    _bytes(result)
    _need(_bytes(request.to_dict()) == _bytes(request_wire)
          and _bytes(preview.to_dict()) == _bytes(preview_wire)
          and _bytes(graph.to_dict()) == _bytes(graph_wire)
          and _bytes(_plain(successor_declaration)) == _bytes(declaration)
          and freeze_plan_create_input_snapshot(request, materials=
              bind_repository_preview_materials(materials, structural_context)).snapshot_cid == snapshot.snapshot_cid,
          "preview successor inputs changed during reconstruction")
    result["bridge_cid"] = content_identity(result)
    _bytes(result)
    return result


def validate_preview_successor_bridge(record, **inputs):
    """Compare full reconstruction, including identity and closed field population."""
    captured = _bytes(_plain(record))
    expected = build_preview_successor_bridge(**inputs)
    _need(captured == _bytes(expected) and captured == _bytes(_plain(record)),
          "preview successor bridge differs from complete reconstructed inputs")
    return expected


__all__ = ["SCHEMA", "MAX_BRIDGE_BYTES", "PreviewSuccessorContractError",
    "bind_repository_preview_materials", "require_passive_repository_preview_inputs",
    "require_passive_repository_preview_receipt",
    "build_preview_successor_bridge", "validate_preview_successor_bridge"]
