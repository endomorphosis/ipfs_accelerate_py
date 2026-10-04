"""Native signed-context, bounded candidate, lowering and adaptation prerequisite.

The child produces an exact reviewed artifact. An explicit owner fixture applies
it under an unchanged Git HEAD; this is not Portal publication or an isolated
repair worker. Both original task contracts survive the complete experiment.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import time

from .finite_repository_admission_experiment import _git, _native_graph, _sources
from .terminal_codebase_finite_experiment import INTENT, _capture
from .terminal_codebase_finite_service_experiment import _authority_materials, _catalog, _open, _request, _scheduler
from .terminal_codebase_adaptation_experiment import (
    _artifact_bytes, _context, _continuation, _idle, _measure, _owned_file_pins, _phase, _registry_inventory,
)
from .terminal_codebase_finite_index import digest, wire

SCHEMA = "finite-repository-reviewed-candidate-qualification@1"
_MODULE = "benchmarks.agent_supervisor.container_coding.finite_repository_candidate_experiment"
_FALSE = {name: False for name in (
    "native_worker_successor_loop_qualified", "worker_isolation_verified", "signing_key_inaccessibility_verified",
    "portal_publication_performed", "candidate_completed_task", "omission_authority", "production_activated",
    "universal_python_semantics_proved", "parser_correctness_proved", "convergence_proved", "generalization_verified",
    "public_terminal_bench_task_satisfied", "formal_decoder_available", "execution_authority", "completion_authority",
)}
_CRITERIA = {
    "schema": "finite-repository-reviewed-candidate-criteria@1",
    "configuration": {"epochs": 16, "learning_rate": .01, "seed": 1729},
    "training_selections": [["calc.py", "train"], ["known_variant.py", "train"], ["tune.py", "tune"], ["canary.py", "canary"]],
    "scope": "Authored Terminal Bench-compatible ten-file fixture; transductive structural reconstruction.",
    "acceptance": [
        "Signed finite context and native materialization preserve both original administrator tasks.",
        "A freshly reserved bounded child generates exactly the reviewed candidate and leaves canonical source, task rows and model registry unchanged.",
        "An explicit owner fixture applies only the retained reviewed bytes; native successor capture derives invalidation.",
        "Real Lean checks the supported AST lowering theorem and actual native target correspondence before and after the change.",
        "Sixteen root and sixteen child training attempts retain exact native basis and Adam lineage; finite tuning objectives do not increase.",
        "Fresh successor finite context has both clauses observed, retains both task declarations and refuses no-work materialization.",
        "Cold native owner and registry verification does not fit or promote a model; complete metadata survives DuckDB/DuckLake restart.",
    ],
    "claims": _FALSE,
}
_SOURCES = (
    _MODULE,
    "benchmarks.agent_supervisor.container_coding.finite_repository_admission_experiment",
    "benchmarks.agent_supervisor.container_coding.terminal_codebase_adaptation_experiment",
    "benchmarks.agent_supervisor.container_coding.terminal_codebase_finite_service_experiment",
    "benchmarks.agent_supervisor.container_coding.terminal_codebase_finite_experiment",
    "benchmarks.agent_supervisor.container_coding.codebase_ir_metadata",
    "benchmarks.agent_supervisor.container_coding.terminal_codebase_supervisor_fixture",
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_candidate",
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_admission",
    "ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission",
    "ipfs_accelerate_py.agent_supervisor.planning.codebase_feature_context",
    "ipfs_accelerate_py.agent_supervisor.planning.finite_integer_source_custody",
    "ipfs_accelerate_py.agent_supervisor.planning.finite_integer_capacity_preview",
    "ipfs_accelerate_py.agent_supervisor.planning.finite_integer_capacity",
    "ipfs_accelerate_py.agent_supervisor.planning.finite_integer_plan_preview",
    "ipfs_accelerate_py.agent_supervisor.planning.finite_integer_plan_service",
    "ipfs_accelerate_py.agent_supervisor.planning.finite_integer_codebase",
    "ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview",
    "ipfs_accelerate_py.agent_supervisor.planning.structural_codebase_context",
    "ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository",
    "ipfs_accelerate_py.agent_supervisor.control.profile_authority",
    "ipfs_datasets_py.logic.software_contracts.codebase_integer_lowering_lean",
    "ipfs_datasets_py.logic.software_contracts.codebase_integer_model_lean",
    "ipfs_datasets_py.logic.software_contracts.codebase_integer_profile",
    "ipfs_datasets_py.logic.software_contracts.codebase_finite_integer_observation",
    "ipfs_datasets_py.logic.software_contracts.codebase_source_training",
    "ipfs_datasets_py.logic.software_contracts.codebase_ir",
    "ipfs_datasets_py.logic.software_contracts.duckdb_ast_store",
    "ipfs_datasets_py.logic.software_verification.source_adapters",
    "ipfs_datasets_py.logic.backends.process",
    "ipfs_datasets_py.duckdb_control.codebase_catalog",
    "ipfs_datasets_py.duckdb_control.autoencoder_registry",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_projection_features",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_runtime_registry",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.codebase_feature_worker",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.modal_autoencoder_cuda",
)


def _write(path, value):
    with Path(path).open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def _pin(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _tasks(intent, materialized):
    def plain(value):
        if isinstance(value, Mapping):
            return {key: plain(child) for key, child in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(child) for child in value]
        return value
    return {cid: plain(intent.get_task(cid)) for cid in materialized["task_cids"]}


def _projection_rows(cx):
    cursor = cx.execute("SELECT * FROM invalidations ORDER BY invalidation_id")
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _reject(controls, label, operation):
    try:
        operation()
    except ValueError as exc:
        controls.append({"control": label, "rejected": True, "error_type": type(exc).__name__, "error": str(exc)})
    else:
        raise ValueError("required rejection accepted: " + label)


def _with_custody(owner, operation):
    from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_source_custody import capture_source_custody
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseTimeoutError
    deadline = time.monotonic() + owner.timeout_seconds
    def checkpoint():
        if time.monotonic() >= deadline:
            raise LeaseTimeoutError("complete source custody join exceeded its phase deadline")
    custody = capture_source_custody(owner, checkpoint)
    result = operation()
    custody.require_current(checkpoint)
    return result


def _lowering_scope(record, *, matches):
    value = record.to_dict()
    if (value["scope"] != "formal_ast_lowering_correctness"
            or value["status"] != ("model_proved" if matches else "model_refuted")
            or value["requested_model_theorem_proved"] is not matches
            or any(value[name] is not True for name in ("source_ast_semantics_defined",
                "source_ast_lowering_proved", "native_target_correspondence_proved", "kernel_checked_model"))
            or any(value[name] is not False for name in ("source_semantics_verified", "runtime_behavior_verified",
                "source_parser_correctness_proved", "cpython_equivalence_proved", "execution_authority", "completion_authority"))):
        raise ValueError("lowering result exceeded or failed its independently checked scope")


def _capture_complete(index, repository, head, scheduler):
    from ipfs_datasets_py.logic.software_verification.source_adapters import adapt_source_to_software_verification
    values = _capture(index, repository, head, scheduler)
    expected = {"calc.py", "decoy.py", "support.py", "consumer.py", "unsupported.py",
        "check_type.py", "check_offset.py", "known_variant.py", "tune.py", "canary.py"}
    if (set(row["path"] for row in values["sources"]) != expected or len(values["ast"]) != len(expected)
            or not {"imports", "calls"}.issubset({row["relation"] for row in values["kg"]})):
        raise ValueError("native capture lost source, AST or import/call metadata")
    manifest = index.load(head.manifest_cid)
    values["compiled_logic"] = []
    for entry in manifest.snapshot.entries:
        source = index.artifacts.get_bytes(entry.source_cid)
        if source != (repository / entry.path).read_bytes():
            raise ValueError("native frontend disposition lost captured-source correspondence")
        pipeline = adapt_source_to_software_verification(source.decode("utf-8"), path=entry.path,
            language="python", revision="snapshot:" + head.snapshot_cid, preserve_type_annotations=True)
        values["compiled_logic"].append({"schema": "finite-reviewed-candidate-native-unit-disposition@1",
            "head": head.to_dict(), "source_path": entry.path, "source_cid": entry.source_cid,
            "frontend": pipeline.to_dict(), "disposition": "retained_unproved_native_frontend",
            "solver_executed": False, "source_semantics_verified": False, "proof_authority": False})
    return values


def _cold(output):
    from ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview import RepositoryPlanPreviewOwner
    from ipfs_accelerate_py.agent_supervisor.planning.codebase_feature_context import verify_current_context
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_admission as boundary
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_candidate as candidate_module
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_lowering_lean import (
        IntegerOffsetLoweringProof, validate_current_integer_offset_lowering,
    )
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
    request = json.loads((output / "cold-request.json").read_bytes())
    for pin in request["retained_pins"]:
        if _pin(pin["path"]) != pin:
            raise ValueError("parent artifact changed before cold verification")
    scheduler = _scheduler(output / "cold-resource-admission.json")
    cx, index = _open(output)
    registry = AutoencoderRegistry(output / "train.duckdb", output / "model-artifacts")
    try:
        owner = RepositoryPlanPreviewOwner(index=index, repository=output / "repository",
            expected_head=CodebaseHead.from_dict(request["head"]), scheduler=scheduler,
            timeout_seconds=90, memory_mb=1024)
        baseline = _registry_inventory(registry)
        context = _context(output / "child-frozen-context")
        verify_current_context(owner, registry, context)
        if _registry_inventory(registry) != baseline:
            raise ValueError("cold frozen replay changed model registry")
        admission = json.loads((output / "successor-admission.json").read_bytes())
        current = boundary.verify_current_finite_repository_admission(owner=owner, admission=admission,
            output=output / "cold-preview", policy_observer=lambda bound: bound.roots)
        lowering = IntegerOffsetLoweringProof.from_dict(json.loads((output / "successor-lowering.json").read_bytes()))
        validated = _with_custody(owner, lambda: validate_current_integer_offset_lowering(lowering, index, owner.repository,
            expected_head=owner.expected_head, contract=IntegerOffsetContract.from_dict(lowering.to_dict()["contract"]),
            tool_policy=request["tool_policy"], scheduler=scheduler,
            timeout_seconds=90, memory_mb=1024))
        _lowering_scope(validated, matches=True)
        parent = json.loads((output / "before-admission.json").read_bytes())
        boundary.verify_finite_repository_admission(admission=parent)
        reviewed = json.loads((output / "reviewed-candidate.json").read_bytes())
        candidate_module.verify_finite_repository_candidate(admission=parent, candidate=reviewed)
        generated = json.loads((output / "generated-candidate.json").read_bytes())
        candidate_module.verify_generated_finite_repository_candidate(record=generated)
        materialized = json.loads((output / "materialized.json").read_bytes())
        with IntentRepository(output / "intent.duckdb") as intent:
            tasks = _tasks(intent, materialized)
        if tasks != request["task_rows"]:
            raise ValueError("cold native task population/state differs")
        if current["semantic_context"]["residual_requirement_ids"] or admission["local_admission"] is not None:
            raise ValueError("cold successor context acquired residual execution")
        capture = _capture_complete(index, owner.repository, owner.expected_head, scheduler)
        _write(output / "cold-capture.json", capture)
        if _registry_inventory(registry) != baseline:
            raise ValueError("cold proof/context replay fitted or promoted a model")
        _idle(scheduler)
        for pin in request["retained_pins"]:
            if _pin(pin["path"]) != pin:
                raise ValueError("cold validation changed retained parent artifact")
        result = {"schema": "finite-reviewed-candidate-cold-verification@1", "status": "completed",
            "head": owner.expected_head.to_dict(), "current_context": current,
            "lowering_record_cid": validated.cid if hasattr(validated, "cid") else lowering.cid,
            "model_version_id": context.version_id, "registry_before": baseline, "registry_after": _registry_inventory(registry),
            "task_rows": tasks, "training_steps": 0, "historical_parent_preserved": True,
            "active_leases": 0, "waiting_requests": 0, **_FALSE}
        _write(output / "cold-response.json", result)
        return result
    finally:
        registry.close()
        cx.close()


def run(output, *, python_executable, lean_executable):
    from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
    from ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview import RepositoryPlanPreviewOwner
    from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_codebase import build_finite_integer_intent
    from ipfs_accelerate_py.agent_supervisor.planning.codebase_feature_context import prepare_codebase_feature_context, verify_current_context
    from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_source_custody import capture_source_custody
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_admission as boundary
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_candidate as candidate_module
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
    from ipfs_datasets_py.logic.software_contracts.codebase_finite_integer_observation import seal_finite_integer_tools
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_lowering_lean import prove_current_integer_offset_lowering
    from ipfs_datasets_py.logic.software_contracts.codebase_source_training import CodebaseTrainingSelection
    from .codebase_ir_metadata import hydrate_codebase_ir_metadata, validate_codebase_ir_metadata
    from .terminal_codebase_supervisor_fixture import bound_terminal_codebase_metadata_records, reconstruct_terminal_codebase_metadata_records
    output = Path(output).absolute()
    if output.resolve() != output or output.exists() or not output.parent.is_dir():
        raise ValueError("fresh canonical experiment namespace required")
    output.mkdir(mode=0o700)
    phases, controls, records = [], [], {}
    started, clock = datetime.now(timezone.utc).isoformat(), time.monotonic()
    cx = registry = scheduler = None
    def phase(name, operation):
        return _phase(phases, name, operation)
    def extend(values):
        for family, rows in values.items():
            target = records.setdefault(family, [])
            ordinal = len(target)
            target.extend({"schema": "finite-reviewed-candidate-metadata-occurrence@1",
                "occurrence": ordinal + position, "record": deepcopy(row)} for position, row in enumerate(rows))
    try:
        _write(output / "criteria.json", _CRITERIA)
        source_directory = output / "selected-source-snapshot"
        source_directory.mkdir(mode=0o700)
        implementation = []
        for module in _SOURCES:
            original = _pin(importlib.import_module(module).__file__)
            destination = source_directory / (module + ".py")
            with destination.open("xb") as stream:
                stream.write(Path(original["path"]).read_bytes())
            retained = _pin(destination)
            if (retained["sha256"], retained["bytes"]) != (original["sha256"], original["bytes"]):
                raise ValueError("selected producer changed during retention")
            implementation.append({"module": module, "original": original, "retained": retained})
        _write(output / "execution-source-pins.json", implementation)
        repository = output / "repository"
        inventory = _sources(repository)
        for name, content in {
            "known_variant.py": "# vocabulary fixture: offset two\ndef increment(n: int) -> int:\n    return n + 2\n",
            "tune.py": "# fixed tuning fixture\ndef increment(n: int) -> int:\n    return n + 1\n",
            "canary.py": "# fixed diagnostic fixture\ndef increment(n: int) -> int:\n    return n + 1\n",
        }.items():
            (repository / name).write_bytes(content.encode())
            inventory[name] = content
        _git(repository, "add", ".")
        _git(repository, "commit", "-qm", "fixed structural training cohort")
        git_head = _git(repository, "rev-parse", "HEAD")
        source_before = {name: _pin(repository / name) for name in inventory}
        profile, lifecycle = output / "profile", output / "lifecycle"
        Supervisor.init_local(repository=repository, consent=True, profile_dir=profile, lifecycle_dir=lifecycle)
        document, catalog = build_finite_integer_intent(INTENT), _catalog()
        tools = seal_finite_integer_tools(python_executable=Path(python_executable).resolve(strict=True),
            lean_executable=Path(lean_executable).resolve(strict=True))
        _write(output / "tool-policy.json", tools)
        _write(output / "intent-ir.json", document.to_dict())
        _write(output / "operation-catalog.json", catalog.to_dict())
        scheduler = _scheduler(output / "resource-admission.json")
        cx, index = _open(output)
        registry = AutoencoderRegistry(output / "train.duckdb", output / "model-artifacts")
        first_receipt = phase("initial_capture", lambda: index.prepare_current(repository,
            repository_id="repository:reviewed-candidate-qualification", operation_id="initial", expected_head=None,
            scheduler=scheduler))
        first = first_receipt.head
        owner = RepositoryPlanPreviewOwner(index=index, repository=repository, expected_head=first,
            scheduler=scheduler, timeout_seconds=90, memory_mb=1024)
        extend(phase("initial_metadata_capture", lambda: _capture_complete(index, repository, first, scheduler)))
        selections = tuple(CodebaseTrainingSelection(path, role, contracts=()) for path, role in _CRITERIA["training_selections"])
        root = phase("root_training", lambda: prepare_codebase_feature_context(owner=owner, registry=registry,
            mode="train", output=output / "root-training-context", selections=selections,
            operation_id="reviewed-candidate-root", **_CRITERIA["configuration"]))
        root_measure = _measure(root)
        root_measure["metrics_scope"] = _CRITERIA["scope"]
        frozen = phase("root_frozen", lambda: prepare_codebase_feature_context(owner=owner, registry=registry,
            mode="frozen", output=output / "root-frozen-context", version_id=root.version_id))
        contract = IntegerOffsetContract(path="calc.py", function_name="increment", parameter="n", offset=2)
        lowering = phase("initial_lowering", lambda: _with_custody(owner, lambda: prove_current_integer_offset_lowering(index, repository,
            expected_head=first, contract=contract, tool_policy=tools, output=output / "before-lowering-artifacts",
            scheduler=scheduler, timeout_seconds=90, memory_mb=1024)))
        _lowering_scope(lowering, matches=False)
        _write(output / "before-lowering.json", lowering.to_dict())
        request = _request(index, repository, first, document, catalog, _authority_materials())
        graph, manifest, bindings = _native_graph(repository=repository, request=request, profile=profile, lifecycle=lifecycle)
        declaration = boundary.author_finite_repository_declaration(owner=owner, manifest=manifest,
            request=request, intent_document=document, source_text=INTENT, operation_catalog=catalog,
            tool_policy=tools, task_bindings=bindings)
        admission = phase("signed_initial_admission", lambda: boundary.admit_finite_repository_plan(owner=owner,
            declaration=declaration, graph=graph, output=output / "before-preview", policy_observer=lambda bound: bound.roots))
        _write(output / "before-admission.json", admission)
        selected = index.load(first.manifest_cid).snapshot.entries
        selected_source = next(entry.source_cid for entry in selected if entry.path == "calc.py")
        decoy_source = next(entry.source_cid for entry in selected if entry.path == "decoy.py")
        if admission["evidence"]["match"]["source_cid"] != selected_source or selected_source == decoy_source:
            raise ValueError("same-name decoy replaced the requested native source")
        with IntentRepository(output / "intent.duckdb") as intent:
            materialized = phase("full_task_materialization", lambda: boundary.materialize_finite_repository_plan(
                owner=owner, admission=admission, intent=intent, output=output / "materialization-preview",
                policy_observer=lambda bound: bound.roots))
            _write(output / "materialized.json", materialized)
            checks = {}
            for number, task in enumerate(sorted(graph.tasks, key=lambda task: task.task_key, reverse=True)):
                row = intent.get_task(task.task_cid)
                intent.cas_task_status(task_cid=task.task_cid, expected_revision=row["revision"], new_status="in_progress")
                observed = local.run_local_task_validations(intent=intent, task_cid=task.task_cid,
                    attempt_id="reviewed-candidate-public-" + str(number))
                checks[task.task_key] = observed
                if observed["passed"]:
                    row = intent.get_task(task.task_cid)
                    intent.cas_task_status(task_cid=task.task_cid, expected_revision=row["revision"], new_status="completed",
                        evidence_digests=[result["evidence_digest"] for result in observed["results"]])
            if checks["FINITE-TYPE"]["passed"] is not True or checks["FINITE-OFFSET"]["passed"] is not False:
                raise ValueError("owner public checks did not preserve the initial residual")
            _write(output / "public-checks.json", checks)
            task_before = _tasks(intent, materialized)
            _write(output / "task-rows-before-candidate.json", task_before)
            registry_before = _registry_inventory(registry)
            reviewed = candidate_module.author_finite_repository_candidate(owner=owner, admission=admission,
                review_ref="review:authored-exact-offset-replacement")
            _write(output / "reviewed-candidate.json", reviewed)
            generated = phase("reserved_child_candidate", lambda: candidate_module.generate_finite_repository_candidate(
                owner=owner, admission=admission, candidate=reviewed, output=output / "candidate-artifacts",
                policy_observer=lambda bound: bound.roots))
            _write(output / "generated-candidate.json", generated)
            if _tasks(intent, materialized) != task_before or _registry_inventory(registry) != registry_before:
                raise ValueError("candidate child altered task or model owners")
            if {name: _pin(repository / name) for name in inventory} != source_before:
                raise ValueError("candidate child changed canonical repository")
            _write(output / "candidate-owner-invariance.json", {"task_rows_unchanged": True,
                "registry_before": registry_before, "registry_after": _registry_inventory(registry), "canonical_source_unchanged": True})
            custody = capture_source_custody(owner, lambda: None)
            candidate_module.verify_finite_repository_candidate(admission=admission, candidate=reviewed)
            candidate_module.verify_generated_finite_repository_candidate(record=generated)
            # Explicit authored OWNER fixture application. The child has no
            # mutation, task completion or publication API in this profile.
            custody.require_current(lambda: None)
            if (repository / "calc.py").read_bytes() != candidate_module.BEFORE_SOURCE:
                raise ValueError("owner fixture preimage changed")
            (repository / "calc.py").write_bytes(candidate_module.AFTER_SOURCE)
            task_after = _tasks(intent, materialized)
            if task_after != task_before:
                raise ValueError("owner fixture application completed a guarded task")
        if _git(repository, "rev-parse", "HEAD") != git_head:
            raise ValueError("owner fixture application moved Git HEAD")
        parent_pins = _owned_file_pins(output / "before-preview", output / "materialization-preview",
            output / "before-lowering-artifacts", output / "candidate-artifacts", output / "root-training-context", output / "root-frozen-context")
        _reject(controls, "old_signed_current_admission_after_edit", lambda: boundary.verify_current_finite_repository_admission(
            owner=owner, admission=admission, output=output / "stale-preview", policy_observer=lambda bound: bound.roots))
        _reject(controls, "old_feature_context_after_edit", lambda: verify_current_context(owner, registry, frozen))
        _reject(controls, "old_candidate_after_edit", lambda: candidate_module.author_finite_repository_candidate(
            owner=owner, admission=admission, review_ref="review:stale-owner"))
        boundary.verify_finite_repository_admission(admission=admission)
        successor_receipt = phase("successor_capture", lambda: index.prepare_current(repository,
            repository_id=first.repository_id, operation_id="successor", expected_head=first, scheduler=scheduler))
        successor = successor_receipt.head
        if successor.generation != first.generation + 1 or successor.snapshot_cid == first.snapshot_cid:
            raise ValueError("native successor did not publish a new source generation")
        owner = replace(owner, expected_head=successor)
        invalidations = _projection_rows(cx)
        if not any(row.get("reason") == "revision_superseded" for row in invalidations):
            raise ValueError("native catalog did not derive revision invalidation")
        _write(output / "native-source-publications.json", {"initial": first_receipt.to_dict(),
            "successor": successor_receipt.to_dict(), "invalidations": invalidations})
        extend(phase("successor_metadata_capture", lambda: _capture_complete(index, repository, successor, scheduler)))
        _reject(controls, "old_context_under_successor_owner", lambda: verify_current_context(owner, registry, frozen))
        child = phase("child_training", lambda: prepare_codebase_feature_context(owner=owner, registry=registry,
            mode="train", output=output / "child-training-context", selections=selections,
            operation_id="reviewed-candidate-child", parent_version_id=root.version_id, **_CRITERIA["configuration"]))
        child_measure = _measure(child)
        child_measure["metrics_scope"] = _CRITERIA["scope"]
        continuation = _continuation(root, child)
        child_frozen = phase("child_frozen", lambda: prepare_codebase_feature_context(owner=owner, registry=registry,
            mode="frozen", output=output / "child-frozen-context", version_id=child.version_id))
        successor_lowering = phase("successor_lowering", lambda: _with_custody(owner, lambda: prove_current_integer_offset_lowering(index, repository,
            expected_head=successor, contract=contract, tool_policy=tools, output=output / "successor-lowering-artifacts",
            scheduler=scheduler, timeout_seconds=90, memory_mb=1024)))
        _lowering_scope(successor_lowering, matches=True)
        _write(output / "successor-lowering.json", successor_lowering.to_dict())
        next_request = _request(index, repository, successor, document, catalog, _authority_materials())
        next_graph, next_manifest, next_bindings = _native_graph(repository=repository, request=next_request,
            profile=profile, lifecycle=lifecycle)
        if [task.to_dict() for task in next_graph.tasks] != [task.to_dict() for task in graph.tasks] or next_bindings != bindings:
            raise ValueError("successor changed the original complete administrator task population")
        next_declaration = boundary.author_finite_repository_declaration(owner=owner, manifest=next_manifest,
            request=next_request, intent_document=document, source_text=INTENT, operation_catalog=catalog,
            tool_policy=tools, task_bindings=next_bindings)
        next_admission = phase("signed_successor_admission", lambda: boundary.admit_finite_repository_plan(
            owner=owner, declaration=next_declaration, graph=next_graph, output=output / "successor-preview",
            policy_observer=lambda bound: bound.roots))
        _write(output / "successor-admission.json", next_admission)
        if next_admission["local_admission"] is not None or next_admission["receipt"]["payload"]["planning_permitted"]:
            raise ValueError("finite no-work acquired execution grant")
        with IntentRepository(output / "empty-successor.duckdb") as empty:
            _reject(controls, "no_work_cannot_materialize", lambda: boundary.materialize_finite_repository_plan(
                owner=owner, admission=next_admission, intent=empty, output=output / "no-work-preview",
                policy_observer=lambda bound: bound.roots))
            with empty._connection(write=False) as connection:
                population = {table: connection.execute("SELECT count(*) FROM " + table).fetchone()[0]
                    for table in ("objectives", "tasks", "plans", "goals")}
            if any(population.values()):
                raise ValueError("no-work refusal wrote native intent rows")
        _write(output / "no-work-refusal.json", {"native_population": population, "execution_grant": False})
        if any(_pin(pin["path"]) != pin for pin in parent_pins):
            raise ValueError("successor changed historical parent artifact")
        numerical = [{"binding": context.material_binding, "retained": context.retained_artifacts}
            for context in (root, frozen, child, child_frozen)]
        _write(output / "training-measurements.json", {"root": root_measure, "child": child_measure, "continuation": continuation})
        retained_pins = _owned_file_pins(output / "cas", output / "model-artifacts",
            output / "successor-preview", output / "successor-lowering-artifacts", output / "child-training-context", output / "child-frozen-context") + parent_pins
        retained_pins += [_pin(output / name) for name in ("criteria.json", "tool-policy.json", "intent-ir.json",
            "operation-catalog.json", "before-admission.json", "successor-admission.json", "reviewed-candidate.json",
            "generated-candidate.json", "before-lowering.json", "successor-lowering.json", "materialized.json")]
        _write(output / "cold-request.json", {"head": successor.to_dict(), "tool_policy": tools,
            "task_rows": task_after, "retained_pins": retained_pins})
        registry.close()
        registry = None
        cx.close()
        cx = None
        cold_process = phase("cold_native_process", lambda: subprocess.run([str(python_executable), "-m", _MODULE,
            "cold", str(output)], capture_output=True, text=True, timeout=180))
        _write(output / "cold-process.json", {"returncode": cold_process.returncode,
            "stdout": cold_process.stdout, "stderr": cold_process.stderr})
        if cold_process.returncode:
            raise ValueError("cold native verification failed: " + cold_process.stderr)
        cold = json.loads((output / "cold-response.json").read_bytes())
        extend(json.loads((output / "cold-capture.json").read_bytes()))
        artifacts = _owned_file_pins(output / "cas", output / "model-artifacts", output / "cold-preview",
            output / "before-preview", output / "materialization-preview", output / "before-lowering-artifacts",
            output / "candidate-artifacts", output / "successor-preview", output / "successor-lowering-artifacts",
            output / "root-training-context", output / "root-frozen-context", output / "child-training-context", output / "child-frozen-context")
        extend({"artifacts": _artifact_bytes(artifacts), "intent_ir": [document.to_dict()],
            "contracts": [contract.to_dict()], "signed_admissions": [admission, next_admission],
            "reviewed_candidates": [reviewed, generated], "lowering_proofs": [lowering.to_dict(), successor_lowering.to_dict()],
            "current_facts": [fact for item in (admission, next_admission) for fact in item["evidence"]["match"]["current_facts"]],
            "finite_matches": [item["evidence"]["match"] for item in (admission, next_admission)],
            "native_tasks": [task_before, task_after, materialized, checks, population],
            "native_publications": [first_receipt.to_dict(), successor_receipt.to_dict(), *invalidations],
            "feature_contexts": numerical, "training": [*numerical, root_measure, child_measure, continuation],
            "vectors": [{"binding": item["binding"], "inference": item["retained"]["inference"]} for item in numerical],
            "controls": controls, "cold_verification": [cold], "criteria": [_CRITERIA, {"implementation": implementation, "tools": tools}]})
        bounded = bound_terminal_codebase_metadata_records(records)
        if wire(reconstruct_terminal_codebase_metadata_records(bounded)) != wire(records):
            raise ValueError("metadata packaging changed a complete producer record")
        metadata = phase("duckdb_ducklake_hydration", lambda: hydrate_codebase_ir_metadata(records=bounded,
            output=output / "metadata", source_snapshot={"schema": SCHEMA, "heads": [first.to_dict(), successor.to_dict()],
                "producer_sha256": digest(records), "criteria": _pin(output / "criteria.json"), "git_head": git_head}))
        replay = phase("duckdb_ducklake_cold_readback", lambda: validate_codebase_ir_metadata(output=output / "metadata",
            expected=metadata, fresh_process=True))
        restored = {family: [json.loads(line)["payload"] for line in
            (output / "metadata" / descriptor["relative_path"]).read_bytes().splitlines()]
            for family, descriptor in replay["exports"].items()}
        if wire(reconstruct_terminal_codebase_metadata_records(restored)) != wire(records):
            raise ValueError("native metadata readback changed complete producer records")
        if any(_pin(pin["path"]) != pin for pin in retained_pins):
            raise ValueError("cold or metadata work changed retained evidence")
        if any(_pin(item["original"]["path"]) != item["original"] or _pin(item["retained"]["path"]) != item["retained"] for item in implementation):
            raise ValueError("selected producer bytes changed during qualification")
        _idle(scheduler)
        result = {"schema": SCHEMA, "status": "completed", "output": str(output), "started_at": started,
            "completed_at": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.monotonic() - clock,
            "fixture_scope": _CRITERIA["scope"], "inventory_paths": sorted(inventory), "git_head_unchanged": True,
            "original_head": first.to_dict(), "successor_head": successor.to_dict(), "bounded_child_candidate_generated": True,
            "owner_fixture_applied_candidate": True, "complete_original_task_population_preserved": True,
            "task_rows_unchanged_during_candidate_and_successor": True, "task_count": len(graph.tasks),
            "root_measurement": root_measure, "child_measurement": child_measure, "continuation": continuation,
            "actual_attempted_training_epochs": root.actual_training_delta + child.actual_training_delta,
            "lowering_proofs": [lowering.to_dict(), successor_lowering.to_dict()],
            "successor_finite_facts": len(next_admission["evidence"]["match"]["current_facts"]),
            "successor_local_admission": None, "historical_parent_preserved": True, "cold_verification": cold,
            "complete_family_counts": {name: len(rows) for name, rows in records.items()},
            "packaged_family_counts": {name: len(rows) for name, rows in bounded.items()},
            "metadata": metadata, "metadata_replay": replay, "controls": controls,
            "active_leases": 0, "waiting_requests": 0, "phase_budget_seconds": 90,
            "budget_scope": "Each native preparation has its own deadline; the complete experiment and metadata readback are separate phases.",
            "implementation": implementation, **_FALSE}
        _write(output / "metadata-result.json", metadata)
        _write(output / "metadata-replay.json", replay)
        _write(output / "result.json", result)
        return result
    except BaseException as exc:
        failure = {"schema": SCHEMA, "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
            "wall_seconds": time.monotonic() - clock, "phases": phases,
            "scheduler": scheduler.snapshot() if scheduler is not None else None}
        _write(output / "failure.json", failure)
        raise
    finally:
        if registry is not None:
            registry.close()
        if cx is not None:
            cx.close()
        _write(output / "phase-costs.json", phases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "cold"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--python", default="/home/barberb/.local/bin/python")
    parser.add_argument("--lean", default="/home/barberb/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lean")
    args = parser.parse_args()
    result = _cold(args.output) if args.action == "cold" else run(args.output,
        python_executable=args.python, lean_executable=args.lean)
    print(json.dumps({key: result[key] for key in ("status", "output", "wall_seconds") if key in result}, sort_keys=True))


if __name__ == "__main__":
    main()
