"""Fresh 300-member source successor, explicit child and model-off planning.

Two setup epochs are explicit caller actions. The coordinator fits nothing;
two fresh 32-member pages exercise default and opt-out CPU 8D inference.
Cold receiving reopens owners in this process. No completed 300-member scan,
worker admission, decoded formula proof, CUDA or production activation is
claimed. Resource reservations are accounting, not kernel memory limits.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time

from . import qualify_codebase_inventory_resume as authored
from . import qualify_codebase_inventory_scan as base

SCHEMA = "codebase-source-successor-native-qualification@1"
REPOSITORY_ID = "qualification:source-successor"


def _source_owner(index, connection):
    from ipfs_datasets_py.logic.software_contracts.codebase_inventory_projection_replay import ASTS_CATALOG_TABLES
    tables = {}
    for name in ASTS_CATALOG_TABLES:
        rows = connection.execute('SELECT * FROM "' + name + '" ORDER BY ALL').fetchall()
        tables[name] = {"rows": len(rows), "sha256": hashlib.sha256(base._wire(rows)).hexdigest()}
    return {"current_head": index.current(REPOSITORY_ID).to_dict(), "tables": tables}


def _owners(index, registry, connection):
    return {"source": _source_owner(index, connection), "registry": base._registry(registry),
            "model_artifacts": base._files(registry.artifact_root)}


def _pins(output, coordinator):
    modules = set(coordinator._implementation()["files"])
    modules.update({
        "ipfs_accelerate_py.agent_supervisor.planning.codebase_source_delta_context",
        "ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview",
        "ipfs_accelerate_py.agent_supervisor.planning.structural_codebase_context",
        "ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler",
        "ipfs_accelerate_py.agent_supervisor.planning.plan_evaluator",
        "ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner",
        "ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts",
        "ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service",
        authored.__name__, base.__name__, __name__,
    })
    destination = output / "producers"
    destination.mkdir()
    rows = []
    for name in sorted(modules):
        path = Path(importlib.import_module(name).__file__).resolve()
        raw = path.read_bytes()
        copy = destination / (name + ".py")
        with copy.open("xb") as stream:
            stream.write(raw)
        rows.append({"name": name, "path": str(path), "copy": copy.relative_to(output).as_posix(),
                     "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    value = {"schema": "codebase-source-successor-selected-producers@1", "files": rows,
             "scope": "listed_local_files_only", "execution_attestation": False}
    base._write(output / "generation-inputs.json", value)
    return value


def _planning_case(index, head, repository):
    from ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner import FrozenPlanningGoal
    from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
        ProducerRule, TaskCandidate, TypedIntent, TypedPredicate, obligation_id_for_producer)
    from ipfs_accelerate_py.agent_supervisor.planning.plan_evaluator import EvidenceAwarePlanPolicy
    from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
        DirtyTreePolicy, PlanAuthorityRoots, PlanCreateRequest, PlanRequestBudget, TaskSourceKind)
    from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import PlanCreateMaterials
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
    roots = PlanAuthorityRoots(repository_id=head.repository_id, task_source_id="source:authored-runtime",
        **{name: cid_for_structured({"authored_root": name}) for name in (
            "task_source_revision", "policy_root", "intent_ir_root", "legal_ir_root", "security_ir_root",
            "capability_catalog_root", "provider_catalog_root", "usage_policy_root", "configuration_root")},
        repository_root_cid=head.snapshot_cid, dirty_worktree_root=head.snapshot_cid,
        program_root=index.load(head.manifest_cid).semantic_state.state_cid)
    request = PlanCreateRequest(prompt_source_cid=cid_for_structured({"authored_prompt": "runtime-review"}),
        repository_id=head.repository_id, repository_root=str(repository), scope_paths=("calc.py",),
        dirty_tree_policy=DirtyTreePolicy.OBSERVE_AND_BIND, task_source_kind=TaskSourceKind.BOTH,
        board_namespace="source-successor", alias_prefix="SUCCESSOR", roots=roots,
        budget=PlanRequestBudget(max_model_calls=0, max_latency_ms=120000),
        required_analysis_operations=(), optional_analysis_operations=(),
        required_logic_families=(), optional_logic_families=())
    goals = tuple(TypedPredicate("goal:runtime:" + str(number), "reviewed_runtime_requirement", "calc.py",
        object_ref="requirement:" + str(number)) for number in range(2))
    intent = TypedIntent("intent:authored-source-successor", goals, ("source:authored-runtime",),
                        current_root_id=head.snapshot_cid)
    producers = tuple(ProducerRule("producer:runtime:" + str(number), (goal.predicate_id,))
                      for number, goal in enumerate(goals))
    tasks = tuple(TaskCandidate("task:runtime:" + str(number),
        (obligation_id_for_producer(producer.producer_id, goal.predicate_id),), producer_id=producer.producer_id)
        for number, (producer, goal) in enumerate(zip(producers, goals)))
    materials = PlanCreateMaterials(intent=intent, producers=producers, task_candidates=tasks,
        frozen_goal=FrozenPlanningGoal("goal:runtime", cid_for_structured({"authored_goal": "runtime"}),
            head.snapshot_cid, EvidenceAwarePlanPolicy(acceptance_criteria=intent.goal_predicate_ids,
                evidence_terms=intent.source_refs, allowed_scopes=("scope:calc.py",),
                available_resource_classes=("cpu",), require_validation=True, require_proof=False)),
        candidate_context={"repository_paths": ["calc.py"], "task_metadata": {
            item.candidate_id: {"predicted_files": ["calc.py"], "scope_ids": ["scope:calc.py"],
                               "resource_classes": ["cpu"]} for item in tasks}},
        extra={"authored_review": "review:complete-runtime-requirements"})
    return request, materials


def qualify(output, *, overall_seconds=1800.0, reference_seconds=120.0):
    from ipfs_accelerate_py.agent_supervisor.planning import codebase_source_delta_context as planning
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor as delta
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor_model as coordinator
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as scan
    from ipfs_datasets_py.logic.software_contracts import codebase_source_training as training
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import CodebaseScanLimits
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_projection_features as features
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError

    output = Path(output).resolve()
    authored.require(type(reference_seconds) in {int, float} and 0 < reference_seconds <= 600,
                     "explicit finite reference qualification deadline required")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + overall_seconds
    report = {"schema": SCHEMA, "qualified": False, "pid": os.getpid(), "phases": [], "controls": [],
        "scope": "fresh_300_member_successor_and_two_32_member_cpu8d_prefix_pages",
        "setup_training_attempts": [], "new_fitting_epochs": 0, "training_attempts_after_setup": 0,
        "inference_attempts_during_selection_or_cold_receiving": 0,
        "source_property_solver_calls": 0, "source_owner_reopens": 0, "model_owner_reopens": 0,
        "new_scan_pages_created": 0, "inherited_scan_pages": 0, "inherited_setup_epochs": 0,
        "complete_scan_qualified": False, "worker_admission_qualified": False,
        "cuda_qualified": False, "384d_qualified": False, "proof_authority": False,
        "production_default_activated": False, "repository_code_executed": False}
    report["operation_deadline_seconds"] = {"default_selection": 120, "reference_selection": reference_seconds,
        "inference_page": 600, "planning_adapter": 120, "existing_repository_preview": 90,
        "reference_override_is_qualification_only": True}
    index = registry = connection = scheduler = None

    def remaining(cap=120):
        seconds = deadline - time.monotonic()
        authored.require(seconds > 0, "overall successor qualification deadline exceeded")
        return min(seconds, cap)

    def options(cap=120):
        return {"scheduler": scheduler, "timeout_seconds": remaining(cap), "memory_mb": 1024}

    def progress():
        report["elapsed_seconds_so_far"] = time.monotonic() - started
        base._progress(output / "progress.json", report)

    def phase(name, action):
        row = {"name": name, "status": "running"}
        report["phases"].append(row)
        progress()
        began = time.monotonic()
        try:
            value = action()
        except BaseException:
            row.update(status="failed", elapsed_seconds=time.monotonic() - began)
            progress()
            raise
        row.update(status="completed", elapsed_seconds=time.monotonic() - began)
        progress()
        return value

    def persist(name, record):
        value = record.to_dict()
        cid = record.artifact_cid if hasattr(record, "artifact_cid") else record.cid
        base._write(output / name, {"artifact_cid": cid, "value": value})

    def refuse(name, action, allowed):
        began = time.monotonic()
        try:
            action()
        except allowed as error:
            report["controls"].append({"name": name, "refused": True, "error_type": type(error).__name__,
                "error": str(error), "elapsed_seconds": time.monotonic() - began})
            base._assert_clean(scheduler)
            progress()
            return
        raise AssertionError("control was accepted: " + name)

    def forbid_fit(*args, **kwargs):
        report["training_attempts_after_setup"] += 1
        raise AssertionError("coordinator/inference/planning/receiving attempted fitting")

    def forbid_inference(*args, **kwargs):
        report["inference_attempts_during_selection_or_cold_receiving"] += 1
        raise AssertionError("selection/cold receiving attempted forward inference")

    try:
        selected = _pins(output, coordinator)
        scheduler = authored._scheduler(output)
        report["scheduler_configuration"] = scheduler.config.persisted_dict()
        repository = output / "repository"
        phase("author_fresh_300_member_repository", lambda: authored._sources(repository))
        index, registry, connection = authored._open(output)
        preparation = {"repository_id": REPOSITORY_ID, "limits": CodebaseScanLimits(512, 65536),
                       "exclusions": (".runtime",)}
        previous = phase("publish_previous_source", lambda: index.prepare_current(repository,
            operation_id="initial", expected_head=None, **preparation, **options()))
        base._write(output / "previous-publication-receipt.json", previous.to_dict())
        persist("previous-manifest.json", index.load(previous.head.manifest_cid))
        selections = [training.CodebaseTrainingSelection(path, role) for path, role in
            (("calc.py", "train"), ("tune.py", "tune"), ("canary.py", "canary"))]

        def fit(head, name, parent=None):
            attempt = {"name": name, "requested_epochs": 1, "actual_completed_epochs": None,
                       "unknown_actual_epochs_on_failure": True}
            report["setup_training_attempts"].append(attempt)
            old_epochs = 0 if parent is None else authored._state(registry, parent)["completed_epochs"]
            record = training.train_current_codebase_features(index, repository, expected_head=head,
                registry=registry, selections=selections, operation_id=name, parent_version_id=parent,
                epochs=1, learning_rate=.002, seed=1729, **options())
            version = record.to_dict()["version_id"]
            actual = authored._state(registry, version)["completed_epochs"] - old_epochs
            authored.require(actual == 1, "explicit setup epoch delta differs")
            attempt.update(actual_completed_epochs=actual, unknown_actual_epochs_on_failure=False, version_id=version)
            report["new_fitting_epochs"] += actual
            persist(name + "-training-record.json", record)
            return version

        parent = phase("explicit_previous_head_root_training", lambda: fit(previous.head, "root"))
        parent_state = authored._state(registry, parent, include_identity=True)
        (repository / "calc.py").write_text("def increment(n: int) -> int:\n    return n + 2\n")
        (repository / "bulk000.py").unlink()
        (repository / "bulk001.py").rename(repository / "renamed.py")
        (repository / "added.py").write_text("def added(n: int) -> int:\n    return n + 700\n")
        authored._git(repository, "add", "-A")
        authored._git(repository, "commit", "--quiet", "--no-verify", "-m", "Explicit source successor")
        current = phase("publish_current_source", lambda: index.prepare_current(repository,
            operation_id="successor", expected_head=previous.head, **preparation, **options()))
        base._write(output / "current-publication-receipt.json", current.to_dict())
        persist("current-manifest.json", index.load(current.head.manifest_cid))
        transition = phase("build_default_source_delta", lambda: delta.build_current_codebase_source_delta(
            index, repository, previous_head=previous.head, expected_head=current.head, **options()))
        persist("source-delta.json", transition)
        child = phase("explicit_current_head_child_training", lambda: fit(current.head, "child", parent))
        child_state = authored._state(registry, child, include_identity=True)
        authored.require(authored._state(registry, parent, include_identity=True) == parent_state,
                         "explicit child training mutated its parent")
        report.update(previous_head=previous.head.to_dict(), current_head=current.head.to_dict(),
            coverage=transition.to_dict()["coverage"], parent_version_id=parent, child_version_id=child,
            checkpoint_states={"root": parent_state, "child": child_state})
        baseline = _owners(index, registry, connection)
        base._write(output / "owners-before.json", baseline)
        original_cas = base._files(index.artifacts.root)
        base._write(output / "source-artifacts-before.json", original_cas)
        with ExitStack() as guards:
            guards.enter_context(base._patch(training, "train_current_codebase_features", forbid_fit))
            guards.enter_context(base._patch(features, "train_projection_features", forbid_fit))
            limits = scan.CodebaseScanResumeLimits(max_inventory_entries=512, page_entries=32)
            def select(optimized=True, selected_child=child):
                with base._patch(scan, "_worker", forbid_inference), base._patch(features, "infer_projection_features", forbid_inference):
                    return coordinator.start_current_codebase_successor_scan(index, repository,
                        source_delta=transition, registry=registry, previous_version_id=parent,
                        version_id=selected_child, limits=limits, optimized=optimized,
                        **options(120 if optimized else reference_seconds))
            selection = phase("select_default_fresh_successor_root", select)
            persist("successor-selection.json", selection)
            root = scan.load_codebase_scan_resume_root(index.artifacts, selection.to_dict()["root_cid"])
            persist("scan-root.json", root)
            reference_selection = phase("select_opt_out_successor_root", lambda: select(False))
            persist("reference-successor-selection.json", reference_selection)
            reference_root = scan.load_codebase_scan_resume_root(index.artifacts, reference_selection.to_dict()["root_cid"])
            persist("reference-scan-root.json", reference_root)
            first = phase("infer_default_fresh_32_member_prefix", lambda: scan.scan_current_codebase_page(
                index, repository, root=root, registry=registry, **options(600)))
            report["new_scan_pages_created"] += 1
            persist("prefix-page.json", first)
            reference_page = phase("infer_opt_out_fresh_32_member_prefix", lambda: scan.scan_current_codebase_page(
                index, repository, root=reference_root, registry=registry, **options(600)))
            report["new_scan_pages_created"] += 1
            persist("reference-prefix-page.json", reference_page)
            first_value, reference_value = first.to_dict(), reference_page.to_dict()
            authored.require(all(first_value[key] == reference_value[key] for key in ("entries", "coverage", "inference")),
                             "fresh default/reference numerical outcomes differ")
            authored.require(first_value["worker_receipt"] is not None and first_value["coverage"]["inferred_rows"] > 0,
                             "fresh native prefix performed no inference")
            report["opt_out_equivalence"] = {"entries_coverage_and_inference_exact": True,
                "scope": "first32_ordered_members_only", "throughput_qualified": False,
                "optimized_root_cid": root.artifact_cid, "reference_root_cid": reference_root.artifact_cid,
                "optimized_page_cid": first.artifact_cid, "reference_page_cid": reference_page.artifact_cid}
            report["prefix_coverage"] = first_value["coverage"]
            request, materials = _planning_case(index, current.head, repository)
            base._write(output / "authored-planning-inputs.json", {"request": request.to_dict(),
                "materials": materials.to_binding_dict()})
            from dataclasses import fields, replace
            from ipfs_accelerate_py.agent_supervisor.prompt import plan_create_service as service
            semantic_inputs = {}
            def retain_binding(material):
                binding = material.to_semantic_binding()
                preimages = {}
                for member in fields(material):
                    raw = service._semantic_material_wire(getattr(material, member.name),
                        allow_records=member.name not in {"model_provider", "evidence_adapters"})
                    digest = "sha256:" + hashlib.sha256(
                        b"plan-create-semantic-input\n" + member.name.encode() + b"\n" + raw).hexdigest()
                    authored.require(binding["field_digests"][member.name] == digest,
                                     "retained semantic material preimage differs")
                    preimages[member.name] = json.loads(raw)
                key = hashlib.sha256(base._wire(binding)).hexdigest()
                semantic_inputs[key] = {"binding": binding, "preimages": preimages}
            def plan():
                return planning.preview_current_source_delta_plan(transition, index, repository,
                    request=request, materials=materials, policy_observer=lambda observed: observed.roots, **options())
            preview = phase("actual_current_source_delta_plan_preview", plan)
            base._write(output / "planning-preview.json", preview)
            retain_binding(materials)
            source_bound = replace(materials, extra={**materials.extra,
                planning.MATERIAL_KEY: preview[planning.MATERIAL_KEY]})
            retain_binding(source_bound)
            context = preview["repository_preview"]["structural_context"]
            context_cid = preview["repository_preview"]["structural_context_cid"]
            fully_bound = replace(source_bound, scan={"scan_cid": context_cid, "structural_codebase": context},
                candidate_context={**source_bound.candidate_context,
                    "structural_codebase_context_cid": context_cid, "structural_codebase": context},
                extra={**source_bound.extra, "structural_codebase_context_cid": context_cid,
                    "structural_codebase": context, "root_observation_profile": "repository-live-root-observation@1"})
            retain_binding(fully_bound)
            authored.require(fully_bound.to_semantic_binding() == preview["repository_preview"]["input_snapshot"]["material_binding"],
                             "retained final material preimages differ from consumed snapshot")
            base._write(output / "planning-semantic-preimages.json", {"schema": "source-successor-planning-semantic-preimages@1",
                "entries": [semantic_inputs[key] for key in sorted(semantic_inputs)],
                "capture": "explicit_reconstruction_checked_against_native_consumed_material_binding",
                "execution_attestation": False})
            authored.require(preview["current_facts"] == preview["removed_task_ids"] == []
                and len(preview["declared_requirement_ids"]) == len(preview["declared_task_ids"]) == 2
                and len(preview["residual_requirements"]) == len(preview["residual_task_ids"]) == 2,
                "planning dropped an independently authored requirement/task")
            report["planning_result_cid"] = preview["result_cid"]
            phase("refuse_old_model_for_new_head", lambda: refuse("old_model_for_new_head", lambda: select(True, parent),
                (ValueError,)))
            cancelled = threading.Event()
            cancelled.set()
            phase("refuse_precancelled_selection_receiving", lambda: refuse("precancelled_selection_receiving",
                lambda: coordinator.validate_current_codebase_successor_scan(selection, index, repository,
                    registry=registry, cancel_event=cancelled, **options()), (LeaseCancelledError,)))
            after = _owners(index, registry, connection)
            authored.require(after == baseline and authored._state(registry, child, include_identity=True) == child_state,
                             "inference/coordinator/planning changed native model/source owners")
            base._write(output / "owners-after-warm.json", after)
            guards.enter_context(base._patch(scan, "_worker", forbid_inference))
            guards.enter_context(base._patch(features, "infer_projection_features", forbid_inference))
            authored._close(registry, connection)
            registry = connection = None
            index, registry, connection = authored._open(output)
            report["source_owner_reopens"] = report["model_owner_reopens"] = 1
            selection = coordinator.load_codebase_successor_scan(index.artifacts, selection.artifact_cid)
            transition = delta.load_codebase_source_delta(index.artifacts, transition.artifact_cid)
            root = scan.load_codebase_scan_resume_root(index.artifacts, root.artifact_cid)
            first = scan.load_codebase_scan_resume_page(index.artifacts, first.artifact_cid)
            phase("cold_receive_successor_selection", lambda: coordinator.validate_current_codebase_successor_scan(
                selection, index, repository, registry=registry, **options()))
            phase("cold_receive_fresh_prefix_page", lambda: scan.validate_current_codebase_scan_page(
                first, index, repository, root=root, registry=registry, **options(600)))
            replayed = phase("cold_current_source_delta_plan_preview", plan)
            base._write(output / "cold-planning-preview.json", replayed)
            base._write(output / "cold-planning-semantic-preimages.json", {"schema": "source-successor-planning-semantic-preimages@1",
                "entries": [semantic_inputs[key] for key in sorted(semantic_inputs)],
                "capture": "explicit_reconstruction_checked_against_native_consumed_material_binding",
                "execution_attestation": False})
            authored.require(replayed == preview, "cold source planning material/receipt identity differs")
            expected = deepcopy(baseline)
            owner_row = list(expected["registry"]["meta"][0])
            owner_row[5] += 1
            expected["registry"]["meta"][0] = owner_row
            cold_after = _owners(index, registry, connection)
            authored.require(base._wire(cold_after) == base._wire(expected), "cold owner changes exceed explicit generation reopen")
            authored.require(authored._state(registry, parent, include_identity=True) == parent_state
                and authored._state(registry, child, include_identity=True) == child_state,
                "cold receiving changed checkpoint/Adam state")
            base._write(output / "owners-after-cold.json", cold_after)
            base._write(output / "checkpoint-states-after.json", {"root": authored._state(registry, parent, include_identity=True),
                "child": authored._state(registry, child, include_identity=True)})
            ending = {row["path"]: row for row in base._files(index.artifacts.root)}
            authored.require(all(ending[row["path"]] == row for row in original_cas), "source setup CAS bytes changed")
            authored._require_pins(output, selected)
            authored.require(registry.resolve_head(registry.get_version(child)["variant_id"], "main") is None,
                             "private model was promoted")
            authored.require(report["training_attempts_after_setup"] == 0 and report["new_fitting_epochs"] == 2,
                             "explicit/no-fit accounting differs")
            authored.require(report["inference_attempts_during_selection_or_cold_receiving"] == 0,
                             "selection/cold receiving inference accounting differs")
            report.update(qualified=True, source_delta_cid=transition.artifact_cid,
                successor_selection_cid=selection.artifact_cid, root_cid=root.artifact_cid,
                prefix_page_cid=first.artifact_cid, cold_receiving_verified=True,
                native_owner_unchanged_except_explicit_reopen=True, original_source_artifacts_preserved=True,
                selected_producers_unchanged=True, numerical_reuse=False, model_head_promoted=False)
    except BaseException as error:
        report.update(qualified=False, error_type=type(error).__name__, error=str(error))
    finally:
        authored._close(registry, connection)
        if scheduler is not None:
            report["final_resources"] = base._assert_clean(scheduler)
        report["recorded_seconds"] = time.monotonic() - started
        base._write(output / "result.json", report)
        progress()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-seconds", type=float, default=120.0)
    args = parser.parse_args()
    result = qualify(args.output, reference_seconds=args.reference_seconds)
    print(json.dumps({"qualified": result["qualified"], "recorded_seconds": result["recorded_seconds"],
                      "error_type": result.get("error_type"), "error": result.get("error")}))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    sys.exit(main())
