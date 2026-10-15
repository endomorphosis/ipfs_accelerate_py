"""Closed historical replay of the explicit preview-successor worker profile.

The child declaration points to an immutable @6 admission and its signed
proposal companion. Public replay checks retained signatures and administrative
coverage. Material digests omit their original bodies: only the separate live
receiving owner can reconstruct those inputs, observe roots/checker and permit
installation or START. This module provides no currentness or proof authority.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from ..proof.formal_verification_contracts import content_identity
from . import codebase_successor_dispatch_context as successor

DECLARATION_SCHEMA = "supervisor-codebase-preview-successor-declaration@1"
WORKER_SCHEMA = "supervisor-codebase-preview-successor-worker-context@1"
MAX_DECLARATION_BYTES = 8 * 1024 * 1024
_DECLARATION_FIELDS = {"schema", "parent_admission", "parent_admission_cid",
    "signed_bridge", "signed_bridge_cid", "bridge_cid"}
_BRIDGE_FIELDS = {"schema", "interpretation_scope", "request", "input_snapshot", "preview",
    "structural_context", "structural_context_cid", "graph", "graph_cid", "admission_cid",
    "successor_declaration", "successor_declaration_cid", "training_record_cid",
    "checker_policy_cid", "evaluator_policy_digest", "compiled_obligation_graph_cid",
    "compiled_obligation_graph_digest", "root_mapping", "candidate_bindings",
    "requirement_bindings", "population", "authority", "checker_executed",
    "signatures_verified_here", "native_currentness_verified_here", "worker_launched",
    "production_admitted", "bridge_cid"}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _wire(value):
    return successor._wire(value, max_bytes=MAX_DECLARATION_BYTES)


def _same(left, right):
    return _wire(left) == _wire(right)


def _bridge_replay(bridge, parent, declared, graph, profile):
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead, HEAD_SCHEMA
    from ipfs_datasets_py.logic.software_contracts.content import validate_cid
    from ..planning.plan_revision_contracts import PlanCreateRequest
    from ..planning.structural_codebase_context import StructuralCodebaseContext
    from ..prompt.plan_create_service import (
        PlanCreateInputSnapshot, PlanCreateMode, PlanCreatePreviewReceipt,
        PlanCreateStage, PlanCreateVerdict, _assert_body_free,
    )
    from .codebase_preview_successor_contract import SCHEMA, MAX_BRIDGE_BYTES

    value = successor._verify_signature(bridge, profile)
    bridge_wire = successor._wire(value, max_bytes=MAX_BRIDGE_BYTES)
    _assert_body_free(value, "retained preview successor bridge")
    _need(set(value) == _BRIDGE_FIELDS and value["schema"] == SCHEMA,
          "closed signed preview successor bridge required")
    _need(value["interpretation_scope"] == "administrative_preview_requirement_task_coverage",
          "preview successor bridge interpretation differs")
    successor._authority(value["authority"])
    _need(all(value[field] is False for field in ("checker_executed", "signatures_verified_here",
        "native_currentness_verified_here", "worker_launched", "production_admitted")),
        "proposal companion cannot acquire execution or proof authority")
    _need(value["bridge_cid"] == content_identity({key: item for key, item in value.items()
        if key != "bridge_cid"}) and value["admission_cid"] == content_identity(parent),
        "preview bridge identity or immutable parent admission differs")
    _need(_same(value["graph"], graph.to_dict()) and value["graph_cid"] == graph.content_id
        and _same(value["successor_declaration"], declared["codebase_successor_context"])
        and value["successor_declaration_cid"] == content_identity(value["successor_declaration"]),
        "preview bridge signed graph or successor declaration differs")
    for field in ("training_record_cid", "checker_policy_cid", "evaluator_policy_digest",
                  "compiled_obligation_graph_digest"):
        validate_cid(value[field], codecs={"dag-json"})
    request = PlanCreateRequest.from_dict(value["request"])
    preview = PlanCreatePreviewReceipt.from_dict(value["preview"])
    snapshot = PlanCreateInputSnapshot.from_dict(value["input_snapshot"])
    _need(_same(request.to_dict(), value["request"])
          and _same(preview.to_dict(), value["preview"])
          and _same(snapshot.to_dict(), value["input_snapshot"]),
          "canonical full preview request, receipt and material snapshot required")
    expected_snapshot = PlanCreateInputSnapshot.from_request(request).to_dict()
    omitted = {"schema", "snapshot_cid", "material_binding"}
    _need(_same({key: item for key, item in value["input_snapshot"].items() if key not in omitted},
                {key: item for key, item in expected_snapshot.items() if key not in omitted})
          and value["input_snapshot"]["schema"].endswith("@2")
          and snapshot.material_binding.get("reuse_supported") is True
          and not snapshot.material_binding.get("unsupported_fields"),
          "preview successor material snapshot must bind the complete request")
    structural = value["structural_context"]
    _need(type(structural) is dict, "exact retained structural context required")
    projected_head = structural.get("head")
    _need(type(projected_head) is dict and set(projected_head) == {"repository_id", "generation",
        "manifest_cid", "snapshot_cid", "ast_revision_id", "receipt_cid"},
        "closed projected structural head required")
    context = StructuralCodebaseContext(CodebaseHead.from_dict({"schema": HEAD_SCHEMA, **projected_head}),
        structural.get("semantic_state_cid"), structural.get("coverage"))
    _need(_same(context.to_dict(), structural)
          and value["structural_context_cid"] == context.cid
          and _same(context.head.to_dict(), declared["codebase_successor_context"]["current_head"]),
          "preview successor structural context identity or signed current head differs")
    roots = request.roots
    _need(request.repository_root == declared["repository"]
          and request.repository_id == context.head.repository_id
          and roots.repository_root_cid == roots.dirty_worktree_root == context.head.snapshot_cid
          and roots.program_root == context.semantic_state_cid == graph.program_root,
          "preview successor request/source/program roots differ")
    _need(preview.request_cid == request.request_cid
          and _same(preview.roots.to_dict(), roots.to_dict())
          and preview.scan_cid == context.cid
          and preview.input_snapshot_cid == snapshot.snapshot_cid
          and preview.mode is PlanCreateMode.DETERMINISTIC
          and preview.verdict in {PlanCreateVerdict.REVIEW_ONLY, PlanCreateVerdict.ADMITTED}
          and preview.read_only is True and not preview.wrote_effects
          and preview.obligation_graph_cid == value["compiled_obligation_graph_cid"],
          "preview successor receipt/source/material bindings differ")
    stages = {stage.stage: stage for stage in preview.stage_results}
    _need(len(stages) == len(preview.stage_results)
          and all(stage in stages and stages[stage].passed is True
              and not stages[stage].blockers and stages[stage].artifact_cid == cid
              for stage, cid in ((PlanCreateStage.SCAN, context.cid),
                  (PlanCreateStage.OBLIGATION, value["compiled_obligation_graph_cid"]))),
          "preview successor structural stage bindings differ")
    expected_roots = {"create_request_cid": request.request_cid,
        "workflow_request_cid": graph.request_cid, "preview_scan_cid": preview.scan_cid,
        "workflow_scan_cid": graph.scan_cid, "program_root": graph.program_root,
        "preview_plan_root": preview.plan_root_cid, "signed_graph_cid": graph.content_id}
    _need(_same(value["root_mapping"], expected_roots), "preview successor explicit root mapping differs")
    tasks = {task.task_cid: task for task in graph.tasks}
    candidates, requirements = value["candidate_bindings"], value["requirement_bindings"]
    _need(type(candidates) is list and len(candidates) == len(tasks)
          and all(type(row) is dict and set(row) == {"candidate_id", "task_cid"}
              and type(row["candidate_id"]) is str and row["candidate_id"]
              and row["task_cid"] in tasks for row in candidates),
          "preview successor needs the complete candidate/task bijection")
    _need(len({row["candidate_id"] for row in candidates}) == len(candidates)
          and {row["task_cid"] for row in candidates} == set(tasks)
          and candidates == sorted(candidates, key=lambda row: row["candidate_id"]),
          "preview successor candidate/task population differs")
    _need(type(requirements) is list and 1 <= len(requirements) <= 16,
          "bounded complete preview successor requirement bindings required")
    covered, predicates = set(), set()
    for row in requirements:
        _need(type(row) is dict and set(row) == {"predicate_id", "task_cids", "acceptance_cids"}
              and type(row["predicate_id"]) is str and row["predicate_id"]
              and row["predicate_id"] not in predicates,
              "exact unique preview successor predicate binding required")
        selected, acceptance = row["task_cids"], row["acceptance_cids"]
        _need(type(selected) is list and selected and all(type(item) is str for item in selected)
              and selected == sorted(set(selected)) and set(selected) <= set(tasks),
              "exact preview successor predicate task coverage required")
        expected_acceptance = {item.content_id for cid in selected for item in tasks[cid].acceptance}
        _need(type(acceptance) is list and acceptance
              and all(type(item) is str for item in acceptance)
              and acceptance == sorted(expected_acceptance),
              "preview successor predicate omits or adds signed task acceptance")
        covered.update(selected)
        predicates.add(row["predicate_id"])
    _need(covered == set(tasks)
          and requirements == sorted(requirements, key=lambda row: row["predicate_id"]),
          "preview successor requirement population omits an authored task")
    population = {"desired_predicates": len(predicates), "authored_candidates": len(tasks),
        "compiled_candidates": len(tasks), "signed_tasks": len(tasks), "signed_goals": len(graph.goals),
        "task_acceptance": sum(len(task.acceptance) for task in graph.tasks),
        "task_validations": sum(len(task.validations) for task in graph.tasks),
        "task_outputs": sum(len(task.outputs) for task in graph.tasks)}
    _need(_same(value["population"], population), "preview successor complete population differs")
    _need(successor._wire(value, max_bytes=MAX_BRIDGE_BYTES) == bridge_wire,
          "detached signed preview successor bridge changed during replay")
    return value


def preview_successor_declaration(parent_admission, signed_bridge):
    """Construct a noncircular signed-child declaration from retained inputs."""
    value = {"schema": DECLARATION_SCHEMA, "parent_admission": parent_admission,
        "parent_admission_cid": content_identity(parent_admission), "signed_bridge": signed_bridge,
        "signed_bridge_cid": content_identity(signed_bridge),
        "bridge_cid": signed_bridge.get("payload", {}).get("bridge_cid")}
    return validate_preview_successor_declaration(value)


def preview_successor_worker_reference(context):
    """Compact historical references for the signed native execution scope."""
    checked = validate_preview_successor_declaration(context)
    return {"schema": "supervisor-preview-successor-worker-reference@1",
        "codebase_preview_successor_context_cid": content_identity(checked),
        **{key: checked[key] for key in ("parent_admission_cid", "signed_bridge_cid", "bridge_cid")}}


def validate_preview_successor_declaration(value, *, manifest=None, graph=None, profile=None):
    """Replay retained declarations without opening any live source/model owner."""
    from . import local_planning_admission as local

    captured = _wire(value)
    inputs = tuple(None if item is None else _wire(item.to_dict() if type(item) is local.PromptGoalGraph
        else item) for item in (manifest, graph))
    checked = json.loads(captured)
    _need(set(checked) == _DECLARATION_FIELDS and checked["schema"] == DECLARATION_SCHEMA,
          "closed preview successor child declaration required")
    parent = checked["parent_admission"]
    _need(type(parent) is dict and set(parent) == {"manifest", "graph", "receipt"}
          and parent.get("manifest", {}).get("payload", {}).get("schema") == local.SUCCESSOR_MANIFEST_SCHEMA,
          "preview successor child requires an immutable exact @6 admission")
    binding = parent["manifest"].get("binding", {})
    selected_profile = profile or SimpleNamespace(identity_did=binding.get("identity"),
        profile_id=binding.get("profile_id"))
    declared = successor._verify_signature(parent["manifest"], selected_profile)
    local._validate_local_manifest_declarations(declared)
    native = local.PromptGoalGraph.from_dict(parent["graph"])
    _need(_same(native.to_dict(), parent["graph"]), "canonical full immutable parent graph required")
    receipt = successor._verify_signature(parent["receipt"], selected_profile)
    expected = local._plain(successor._planning_payload(native, parent["manifest"], declared,
        selected_profile, declared["sources"]))
    _need(_same(receipt, expected), "preview successor immutable parent planning receipt differs")
    bridge = _bridge_replay(checked["signed_bridge"], parent, declared, native, selected_profile)
    _need(checked["parent_admission_cid"] == content_identity(parent)
          and checked["signed_bridge_cid"] == content_identity(checked["signed_bridge"])
          and checked["bridge_cid"] == bridge["bridge_cid"],
          "preview successor child identities differ")
    if manifest is not None:
        _need(type(manifest) is dict and manifest.get("schema") == local.PREVIEW_SUCCESSOR_MANIFEST_SCHEMA,
              "explicit @7 preview successor child manifest required")
        base = {key: item for key, item in manifest.items() if key != "codebase_preview_successor_context"}
        base["schema"] = local.SUCCESSOR_MANIFEST_SCHEMA
        _need(_same(base, declared) and _same(manifest.get("codebase_preview_successor_context"), checked),
              "preview successor child changed immutable parent source/task declarations")
    if graph is not None:
        supplied = graph.to_dict() if type(graph) is local.PromptGoalGraph else graph
        _need(_same(supplied, native.to_dict()), "preview successor child changed immutable parent task graph")
    _need(_wire(value) == captured and _wire(checked) == captured and tuple(None if item is None else _wire(
        item.to_dict() if type(item) is local.PromptGoalGraph else item)
        for item in (manifest, graph)) == inputs, "preview successor child inputs changed during replay")
    return checked


def build_preview_successor_worker_context(*, manifest_envelope, graph, receipt, task_cid,
        owner_identity, owner_profile_id, inventory_context, successor_context):
    """Render the strict child profile after complete historical public replay."""
    from . import local_planning_admission as local

    inputs = (manifest_envelope, graph, receipt, inventory_context, successor_context)
    captured = tuple(successor._guard(item) for item in inputs)
    result = successor._build_successor_worker_context(manifest_envelope=manifest_envelope,
        graph=graph, receipt=receipt, task_cid=task_cid, owner_identity=owner_identity,
        owner_profile_id=owner_profile_id, inventory_context=inventory_context,
        successor_context=successor_context, manifest_schema=local.PREVIEW_SUCCESSOR_MANIFEST_SCHEMA,
        planning_receipt_schema=local.PREVIEW_SUCCESSOR_PLANNING_RECEIPT_SCHEMA)
    profile = SimpleNamespace(identity_did=owner_identity, profile_id=owner_profile_id)
    manifest = successor._verify_signature(manifest_envelope, profile)
    declaration = validate_preview_successor_declaration(manifest["codebase_preview_successor_context"],
        manifest=manifest, graph=graph, profile=profile)
    bridge = declaration["signed_bridge"]["payload"]
    _need(bridge["training_record_cid"] == successor_context["selection"]["value"]["training_record_cid"],
          "preview successor training record differs from complete selected model context")
    result.pop("context_cid")
    result.update(schema=WORKER_SCHEMA,
        codebase_preview_successor_context_cid=content_identity(declaration),
        **{key: declaration[key] for key in ("parent_admission_cid", "signed_bridge_cid", "bridge_cid")},
        preview_requirement_coverage={key: bridge[key] for key in ("interpretation_scope", "root_mapping",
            "candidate_bindings", "requirement_bindings", "population", "checker_policy_cid",
            "training_record_cid", "evaluator_policy_digest", "compiled_obligation_graph_cid",
            "compiled_obligation_graph_digest", "structural_context_cid")},
        original_material_bodies_reconstructed_here=False,
        native_preview_roots_current_verified_here=False,
        checker_executed_here=False)
    result["context_cid"] = content_identity(result)
    successor._wire(result)
    _need(tuple(successor._guard(item) for item in inputs) == captured,
          "preview successor worker inputs changed during public replay")
    return result


__all__ = ["DECLARATION_SCHEMA", "WORKER_SCHEMA", "MAX_DECLARATION_BYTES",
    "preview_successor_declaration", "validate_preview_successor_declaration",
    "preview_successor_worker_reference", "build_preview_successor_worker_context"]
