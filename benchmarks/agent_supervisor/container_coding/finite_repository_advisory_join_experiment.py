"""Join advisory source training, a trusted proposal, and an isolated native worker.

Eight-dimensional CPU float64 structural reconstruction is advisory. It does
not rank repairs, decode formal logic, or choose the independently reviewed
fixed edit. The unchanged signed model-off admission and native owner retain
execution, publication and completion gates for the complete two-task fixture.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import tempfile

from .finite_repository_admission_experiment import (
    _git, _pin, _sources, _write,
)
from .terminal_codebase_finite_experiment import INTENT
from .terminal_codebase_finite_service_experiment import (
    _authority_materials, _catalog, _open, _request, _scheduler,
)

SCHEMA = "finite-repository-advisory-native-worker-qualification@1"
MODULE = "benchmarks.agent_supervisor.container_coding.finite_repository_advisory_join_experiment"
SOURCES = (MODULE,
    "benchmarks.agent_supervisor.container_coding.finite_repository_admission_experiment",
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_admission",
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_execution",
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_candidate_runner",
    "ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime",
    "ipfs_accelerate_py.agent_supervisor.runtime.candidate_execution",
    "ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge",
    "ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission",
    "ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository",
    "benchmarks.agent_supervisor.container_coding.finite_repository_worker_experiment",
    "benchmarks.agent_supervisor.container_coding.finite_repository_candidate_experiment",
    "benchmarks.agent_supervisor.container_coding.terminal_codebase_adaptation_experiment",
    "ipfs_accelerate_py.agent_supervisor.planning.codebase_feature_context",
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_candidate",
    "ipfs_datasets_py.duckdb_control.autoencoder_registry",
    "ipfs_datasets_py.logic.software_contracts.codebase_source_training",
    "ipfs_datasets_py.logic.software_contracts.codebase_integer_lowering_lean")

from .terminal_codebase_adaptation_experiment import (
    _check_preview, _context, _continuation, _idle, _measure, _owned_file_pins,
    _phase, _preview, _registry_inventory,
)
from .finite_repository_candidate_experiment import (
    _capture_complete, _lowering_scope, _tasks, _with_custody,
)

CRITERIA = {
    "schema": "finite-advisory-native-worker-criteria@1",
    "configuration": {"epochs": 16, "learning_rate": .01, "seed": 1729},
    "training_selections": [["calc.py", "train"], ["known_variant.py", "train"],
        ["tune.py", "tune"], ["canary.py", "canary"]],
    "intended_training_phases": ["root_training", "initial_child_training", "successor_child_training"],
    "intended_total_attempted_epochs": 48,
    "representation": "8D CPU float64 native compiler structural reconstruction",
    "evaluation_scope": "Fixed transductive structural cohort; repeated tuning and diagnostic canary.",
    "features_choose_fixed_edit": False, "latent_ranking_available": False,
    "formal_decoder_available": False, "cuda_qualified": False, "384d_qualified": False,
    "signed_execution_admission_mode": "model_off", "provider_calls": 0,
    "acceptance": [
        "Initial off/root-frozen/initial-child-frozen advisory previews preserve identical complete finite meaning.",
        "Three independently retained training phases attempt sixteen epochs with exact Adam and basis continuation.",
        "Trusted reserved proposal output feeds the public native worker handoff without a canonical owner write.",
        "Actual separate worker identity publishes only the permitted source and completes the original population.",
        "Successor off/frozen previews agree, stale advice refuses, and fresh successor registry replay performs no fitting.",
    ],
}
ADVISORY_FIELDS = ("source_cid", "query", "domain_inputs", "domain_cid", "observations", "eligible_clause_ids",
    "residual_clause_ids", "clause_results", "selected_task_ids", "declared_task_requirement_ids",
    "candidate_task_meaning", "current_facts_count", "operation_catalog_cid")


def _advisory_projection(result):
    match = result["match"]
    return {name: match[name] for name in ("source_cid", "query", "domain_inputs", "domain_cid",
            "eligible_clause_ids", "residual_clause_ids", "clause_results")} | {
        "observations": match["observation"]["observations"],
        "selected_task_ids": result["selected_task_ids"],
        "declared_task_requirement_ids": result["declared_task_requirement_ids"],
        "candidate_task_meaning": result["candidate_plan"]["tasks"],
        "current_facts_count": result["current_facts_count"],
        "operation_catalog_cid": result["operation_catalog_cid"],
    }


def _comparison(previews):
    projection = _advisory_projection(previews[0][1])
    if any(_advisory_projection(value) != projection for _, value in previews[1:]):
        raise ValueError("advisory modes changed complete finite source/domain/clause or task meaning")
    return {"schema": "finite-advisory-mode-comparison@1", "agreement": True,
        "compared_fields": list(ADVISORY_FIELDS), "projection": projection,
        "preview_records": [{"label": label, "result_cid": value["result_cid"],
            "feature_context_cid": value["feature_context"]["context_cid"]} for label, value in previews],
        "features_choose_fixed_edit": False, "execution_authority": False, "completion_authority": False}


def _training_ledger(attempts):
    return {"schema": "finite-advisory-observed-training-attempts@1", "attempts": attempts,
        "known_actual_attempted_epochs": sum(row["actual_attempted_epochs"] or 0 for row in attempts),
        "unknown_fitting_attempt_count": sum(row["actual_attempted_epochs"] is None for row in attempts),
        "count_scope": "Returned native context deltas; a failed fitting phase remains unknown rather than zero."}


def _update_progress(path, value):
    path = Path(path)
    if (path.name not in {"phase-costs.json", "training-attempts.json", "training-measurements.json"}
            or path.parent.resolve(strict=True) != path.parent or path.is_symlink()):
        raise ValueError("only owned progress observations may be replaced")
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                prefix="." + path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _scope_cost(phases, label, manager):
    start = time.monotonic()
    status, error = "completed", None
    try:
        with manager as value:
            yield value
    except BaseException as caught:
        status, error = "failed", repr(caught)
        raise
    finally:
        phases.append({"phase": label, "status": status, "wall_seconds": time.monotonic() - start,
            "error": error, "cost_scope": "inclusive context lifetime; nested phase costs are not additive"})



def _native_graph(*, repository, request, profile, lifecycle):
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
        PromptAcceptanceRecord, PromptGoalGraph, PromptGoalRecord, PromptOutputRecord,
        PromptTaskRecord, PromptValidationRecord,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_codebase import (
        TYPE_STATEMENT_ID, OFFSET_STATEMENT_ID,
    )
    policy = content_identity(local.LOCAL_POLICY)
    scopes = ("calc.py", "check_type.py", "check_offset.py")
    checks, criteria = [], []
    for kind in ("type", "offset"):
        checks.append(PromptValidationRecord(validation_key="public-" + kind,
            argv=("python3", "check_" + kind + ".py"), policy_cid=policy))
        criteria.append(PromptAcceptanceRecord(criterion_key="finite-" + kind,
            criterion="The public " + kind + " check passes", validation_keys=("public-" + kind,)))
    goal = PromptGoalRecord(goal_key="FINITE-GOAL", parent_goal_cid="", dependency_goal_cids=(),
        title="Repair increment", objective="Retain exact integer output and return n plus two",
        rationale="Independently reviewed public finite requirements", scope_paths=scopes,
        acceptance=tuple(criteria))
    tasks, specs = [], []
    for kind, check, criterion in zip(("type", "offset"), checks, criteria):
        dependencies = () if not tasks else (tasks[0].task_cid,)
        task = PromptTaskRecord(task_key="FINITE-" + kind.upper(), goal_cid=goal.goal_cid,
            dependency_task_cids=dependencies, objective="Satisfy the public " + kind + " check",
            rationale="Keep the complete administrator task population", scope_paths=scopes,
            outputs=(PromptOutputRecord(path="calc.py", effect="modify", media_type="text/x-python"),),
            validations=(check,), acceptance=(criterion,), evidence_cids=(),
            policy_roots=(policy,), predicted_files=("calc.py",))
        tasks.append(task)
        specs.append({"task_key": task.task_key, "scope_paths": list(scopes),
            "dependencies": [] if kind == "type" else [tasks[0].task_key],
            "outputs": [{name: getattr(task.outputs[0], name) for name in ("path", "effect", "media_type")}],
            "validations": [{name: local._plain(getattr(check, name)) for name in
                ("validation_key", "argv", "cwd", "expected_exit_codes", "policy_cid")}],
            "acceptance": [{name: local._plain(getattr(criterion, name)) for name in
                ("criterion_key", "criterion", "evidence_cids", "validation_keys")}],
        })
    roots = {"request_cid": request.request_cid, "program_root": request.roots.program_root,
        "scan_cid": content_identity({"sources": local._sources(repository,
            sorted(_git(repository, "ls-files", "-z").split("\0")[:-1]))})}
    graph = PromptGoalGraph(**roots, policy_roots=(policy,), goals=(goal,), tasks=tuple(tasks), evidence=())
    manifest = local.author_local_benchmark_manifest(repository=repository, profile_dir=profile,
        lifecycle_dir=lifecycle, task_specs=specs, planning_roots=roots)
    return graph, manifest, {TYPE_STATEMENT_ID: tasks[0].task_key, OFFSET_STATEMENT_ID: tasks[1].task_key}



def _historical_replay(output):
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_admission as boundary
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import cid_for_structured
    admission = json.loads((output / "before-admission.json").read_bytes())
    verified = boundary.verify_finite_repository_admission(admission=admission)
    materialized = json.loads((output / "materialized.json").read_bytes())
    with IntentRepository(output / "private/intent.duckdb") as intent:
        rows = [intent.get_task(cid) for cid in materialized["task_cids"]]
        all_rows = intent.list_tasks()
        if ({row["task_cid"] for row in all_rows} != set(materialized["task_cids"])
                or len(all_rows) != len(materialized["task_cids"])):
            raise ValueError("fresh replay changed the complete native task population")
        statuses = {row["task_alias"]: row["status"] for row in rows}
        if statuses != {"FINITE-TYPE": "completed", "FINITE-OFFSET": "completed"}:
            raise ValueError("fresh replay lost the complete native task population")
        plan = intent.get_plan(materialized["plan_id"])
        ref = plan["body"]["finite_repository_admission_ref"]
        retained_raw = Path(ref["path"]).read_bytes()
        retained = json.loads(retained_raw)
        if (retained != admission or len(retained_raw) != ref["bytes"]
                or hashlib.sha256(retained_raw).hexdigest() != ref["sha256"]
                or cid_for_structured(retained) != ref["admission_cid"]):
            raise ValueError("native plan lost the complete signed historical context")
        boundary.verify_finite_repository_admission(admission=retained)
    return {"schema": "finite-worker-fresh-process-historical-replay@1",
        "historical_integrity_verified": bool(verified), "task_statuses": statuses,
        "current_freshness_claimed": False, "training_steps": 0}


def _run(output, *, python_executable, lean_executable, handoff_root, worktree_root, phases, training_attempts):
    from .native_quack_qualification import open_existing_native_owner
    from .terminal_container_supervisor import _native_diagnostics
    from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
    from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        WorktreeLifecycleStore, WorkspaceLifecycleRecord, read_process_birth,
    )
    from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_codebase import build_finite_integer_intent
    from ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview import RepositoryPlanPreviewOwner
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_admission as boundary
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_candidate_runner import author_finite_repository_candidate
    from ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_execution import reserve_finite_repository_execution
    from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon
    from ipfs_accelerate_py.agent_supervisor.planning.codebase_feature_context import (
        prepare_codebase_feature_context, verify_current_context,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_candidate as trusted_candidate
    from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
    from ipfs_datasets_py.logic.software_contracts.codebase_source_training import CodebaseTrainingSelection
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_lowering_lean import prove_current_integer_offset_lowering
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes, cid_for_structured
    from ipfs_datasets_py.logic.software_contracts.codebase_finite_integer_observation import seal_finite_integer_tools
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import (
        IntegerOffsetContract, UnsupportedIntegerProfile, compile_integer_offset,
    )

    output = Path(output).absolute()
    if output.exists() or output.parent.resolve() != output.parent:
        raise ValueError("fresh exact qualification output required")
    output.mkdir(mode=0o755)
    private = output / "private"
    private.mkdir(mode=0o700)
    started = time.monotonic()
    _write(output / "criteria.json", CRITERIA)
    criteria_pin = _pin(output / "criteria.json")
    def phase(label, operation):
        try:
            return _phase(phases, label, operation)
        finally:
            _update_progress(output / "phase-costs.json", phases)
    pins = [_pin(importlib.import_module(name).__file__) for name in SOURCES]
    copies = output / "selected-source-snapshot"
    copies.mkdir()
    for name, pin in zip(SOURCES, pins):
        path = copies / (name + ".py")
        path.write_bytes(Path(pin["path"]).read_bytes())
        if _pin(path)["sha256"] != pin["sha256"]:
            raise ValueError("selected producer changed during sequential copy")
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
    original_commit = _git(repository, "rev-parse", "HEAD")
    profile, lifecycle = private / "profile", private / "lifecycle"
    Supervisor.init_local(repository=repository, consent=True, profile_dir=profile, lifecycle_dir=lifecycle)
    document, catalog = build_finite_integer_intent(INTENT), _catalog()
    tools = seal_finite_integer_tools(python_executable=Path(python_executable).resolve(strict=True),
        lean_executable=Path(lean_executable).resolve(strict=True))
    scheduler = _scheduler(private / "resource-admission.json")
    connection, index = _open(output)
    registry = None
    advisory = private / "advisory"
    advisory.mkdir(mode=0o700)
    (advisory / "prompt.txt").write_bytes(INTENT.encode())
    _write(advisory / "authority-materials.json", _authority_materials())
    _write(output / "tool-policy.json", tools)
    _write(output / "intent-ir.json", document.to_dict())
    _write(output / "operation-catalog.json", catalog.to_dict())
    selections = tuple(CodebaseTrainingSelection(path, role, contracts=())
        for path, role in CRITERIA["training_selections"])
    _write(output / "training-selections.json", [item.to_dict() for item in selections])
    measurements, continuations, training_contexts = [], [], []
    def train(label, current_owner, parent_version_id=None):
        row = {"phase": label, "status": "started", "requested_epochs": 16,
            "actual_attempted_epochs": None, "version_id": None,
            "parent_version_id": parent_version_id, "native_operation_id": "advisory-worker-" + label,
            "context_path": str(advisory / (label.replace("_", "-") + "-context"))}
        training_attempts.append(row)
        _update_progress(output / "training-attempts.json", _training_ledger(training_attempts))
        try:
            context = phase(label, lambda: prepare_codebase_feature_context(owner=current_owner,
                registry=registry, mode="train", output=Path(row["context_path"]),
                selections=selections, operation_id=row["native_operation_id"],
                parent_version_id=parent_version_id, **CRITERIA["configuration"]))
            row.update(status="completed", actual_attempted_epochs=context.actual_training_delta,
                version_id=context.version_id, context_cid=context.cid)
            training_contexts.append(context)
            _update_progress(output / "training-attempts.json", _training_ledger(training_attempts))
            measurement = _measure(context)
            measurements.append(measurement)
            _update_progress(output / "training-measurements.json", {"measurements": measurements,
                "continuations": continuations})
            return context
        except BaseException as error:
            if row["actual_attempted_epochs"] is None:
                row["status"] = "failed_unknown_actual_fitting"
            else:
                row["status"] = "native_training_completed_measurement_failed"
            row["error"] = repr(error)
            raise
        finally:
            _update_progress(output / "training-attempts.json", _training_ledger(training_attempts))
    def advice(label, current_owner, context, facts, selected):
        result, request = phase(label + "_preview", lambda: _preview(index, repository,
            current_owner.expected_head, scheduler, registry, context, tools,
            advisory / (label + "-preview-artifacts")))
        _check_preview(result, facts=facts, selected=selected, feature=context)
        _write(advisory / (label + "-preview.json"), result)
        _write(advisory / (label + "-request.json"), request.to_dict())
        _idle(scheduler)
        return result
    report = {"schema": SCHEMA, "status": "incomplete", "worker_launched": False,
        "training_steps_during_admission_and_worker": 0, "planning_model_calls": 0,
        "provider_calls": 0, "production_activated": False,
        "representation": CRITERIA["representation"], "features_choose_fixed_edit": False,
        "latent_ranking_available": False, "formal_decoder_available": False,
        "cuda_qualified": False, "384d_qualified": False,
        "task_omission_authority": False, "universal_python_semantics_proved": False,
        "inventory_paths": sorted(inventory), "original_commit": original_commit}
    try:
        registry = AutoencoderRegistry(private / "train.duckdb", private / "model-artifacts")
        first = phase("initial_source_capture", lambda: index.prepare_current(repository,
            repository_id="repository:finite-advisory-worker-qualification", operation_id="initial",
            expected_head=None, scheduler=scheduler, admission_timeout_seconds=90)).head
        captured = index.load(first.manifest_cid)
        entries = {entry.path: entry for entry in captured.snapshot.entries}
        if set(entries) != set(inventory):
            raise ValueError("captured inventory lost a repository unit")
        ast_units = {}
        for name in ("calc.py", "decoy.py", "unsupported.py"):
            unit = next(item for item in captured.units if item.source_key == entries[name].source_key)
            if unit.ast_cid is None or index.load_ast_artifact(captured, name) is None:
                raise ValueError("captured AST inventory lost " + name)
            ast_units[name] = unit.ast_cid
        try:
            compile_integer_offset((repository / "unsupported.py").read_bytes(),
                IntegerOffsetContract(path="unsupported.py", function_name="dynamic", parameter="value", offset=2),
                revision="snapshot:" + first.snapshot_cid)
        except UnsupportedIntegerProfile as error:
            unsupported = {"rejected": True, "error": str(error)}
        else:
            raise ValueError("unsupported AST gained an integer-model fact")
        _write(output / "captured-inventory.json", {"paths": sorted(entries), "ast_units": ast_units,
            "unsupported_integer_profile": unsupported})
        owner = RepositoryPlanPreviewOwner(index=index, repository=repository, expected_head=first,
            scheduler=scheduler, timeout_seconds=90, memory_mb=1024)
        initial_capture = phase("initial_complete_metadata_capture", lambda:
            _capture_complete(index, repository, first, scheduler))
        _write(output / "initial-capture.json", initial_capture)
        off = phase("initial_model_off_prepare", lambda: prepare_codebase_feature_context(owner=owner,
            registry=registry, mode="model_off", output=advisory / "root-off-context"))
        root = train("root_training", owner)
        root_frozen = phase("root_frozen_prepare", lambda: prepare_codebase_feature_context(owner=owner,
            registry=registry, mode="frozen", output=advisory / "root-frozen-context", version_id=root.version_id))
        initial_child = train("initial_child_training", owner, root.version_id)
        continuations.append(_continuation(root, initial_child))
        initial_child_frozen = phase("initial_child_frozen_prepare", lambda:
            prepare_codebase_feature_context(owner=owner, registry=registry, mode="frozen",
                output=advisory / "initial-child-frozen-context", version_id=initial_child.version_id))
        initial_registry = _registry_inventory(registry)
        initial_previews = [(label, advice(label, owner, context, 1, ["task:finite:offset"]))
            for label, context in (("initial-off", off), ("root-frozen", root_frozen),
                ("initial-child-frozen", initial_child_frozen))]
        initial_comparison = _comparison(initial_previews)
        _write(output / "initial-advisory-comparison.json", initial_comparison)
        if _registry_inventory(registry) != initial_registry:
            raise ValueError("initial advisory previews fitted or promoted a model")
        contract = IntegerOffsetContract(path="calc.py", function_name="increment", parameter="n", offset=2)
        before_lowering = phase("initial_lowering", lambda: _with_custody(owner,
            lambda: prove_current_integer_offset_lowering(index, repository, expected_head=first,
                contract=contract, tool_policy=tools, output=output / "before-lowering-artifacts",
                scheduler=scheduler, timeout_seconds=90, memory_mb=1024)))
        _lowering_scope(before_lowering, matches=False)
        _write(output / "before-lowering.json", before_lowering.to_dict())
        request = _request(index, repository, first, document, catalog, _authority_materials())
        graph, manifest, bindings = _native_graph(repository=repository, request=request,
            profile=profile, lifecycle=lifecycle)
        tasks = {task.task_key: task for task in graph.tasks}
        declaration = boundary.author_finite_repository_declaration(owner=owner, manifest=manifest,
            request=request, intent_document=document, source_text=INTENT, operation_catalog=catalog,
            tool_policy=tools, task_bindings=bindings)
        admission = phase("signed_model_off_initial_admission", lambda:
            boundary.admit_finite_repository_plan(owner=owner, declaration=declaration,
                graph=graph, output=output / "before-preview", policy_observer=lambda bound: bound.roots))
        selected_source = boundary.verify_finite_repository_admission(admission=admission)["semantic_context"]["source_cid"]
        if selected_source != entries["calc.py"].source_cid or selected_source == entries["decoy.py"].source_cid:
            raise ValueError("same-name decoy changed the selected source identity")
        _write(output / "before-admission.json", admission)
        with IntentRepository(private / "intent.duckdb") as intent:
            materialized = phase("complete_original_task_materialization", lambda:
                boundary.materialize_finite_repository_plan(owner=owner, admission=admission,
                    intent=intent, output=output / "materialization-preview", policy_observer=lambda bound: bound.roots))
            _write(output / "materialized.json", materialized)
        (private / "intent.duckdb").chmod(0o600)
        with _scope_cost(phases, "native_typed_owner_lifetime_inclusive",
                open_existing_native_owner(database=private / "intent.duckdb", checkout=repository,
                    state_dir=private / "owner", repository_id=manifest["payload"]["repository_cid"],
                    execution_routes={task.task_key: GROK_CODEX_EXECUTION_MODE for task in graph.tasks})) as native:
            driver = DatabaseImplementationDaemon(database_path=native.database,
                coordination_path=private / "prerequisite-coordination.duckdb",
                execution_path=private / "prerequisite-execution.duckdb", authority_mode="quack",
                task_source_kind="duckdb", owner_session_id="session:finite-worker-prerequisite",
                process_instance_id=native.identity.process_birth_id, quack_uri=native.identity.listen_uri,
                task_source=native.source, close_task_source=False,
                state_owner_bootstrap_credentials=native.credentials, strict_task_sharding=True,
                max_task_attempts=1, lease_ms=120_000, require_real_execution=True).open()
            try:
                attempt = phase("actual_native_prerequisite_claim", lambda: driver.claim_next())
                if attempt is None or attempt.task_cid != tasks["FINITE-TYPE"].task_cid:
                    raise ValueError("actual native prerequisite claim selected a different task")
                prerequisite = native.source.get_task(attempt.task_cid)
                checked = phase("actual_native_prerequisite_public_checks", lambda:
                    run_owner_local_task_validations(server=native.server, task_cid=attempt.task_cid,
                        attempt_id=attempt.attempt_id, expected_revision=prerequisite.revision))
                if checked["passed"] is not True:
                    raise ValueError("actual native public prerequisite did not pass")
                claimed = dict(prerequisite.body["completion_receipt"])
                digest = checked["results"][0]["evidence_digest"]
                native.source.compare_and_set_status(prerequisite.task_cid, prerequisite.revision, "completed",
                    receipt={"operation": "database_complete", "evidence_digest": digest,
                        **{key: claimed[key] for key in ("attempt_id", "claim_id", "lease_id",
                            "owner_session_id", "fencing_token", "fence_epoch")}},
                    expected_control_receipt=claimed, evidence_digests=[digest])
                if native.source.get_task(attempt.task_cid).status != "completed":
                    raise ValueError("native prerequisite completion was not committed")
                _write(output / "prerequisite-native-claim.json", claimed)
                _write(output / "prerequisite-validation.json", checked)
            finally:
                driver.close()
            residual_task = native.source.get_task(tasks["FINITE-OFFSET"].task_cid)
            if residual_task.status != "ready":
                raise ValueError("residual must remain ready for the genuine native claim")
            with native.server._lock:
                with IntentRepository(bound_connection=native.server._connection, install_schema=False) as intent:
                    tasks_before_proposal = _tasks(intent, materialized)
            source_before_proposal = {name: _pin(repository / name) for name in inventory}
            registry_before_proposal = _registry_inventory(registry)
            if registry_before_proposal != initial_registry:
                raise ValueError("signed model-off admission or native materialization fitted or promoted a model")
            _write(output / "admission-no-fit-observation.json", {"registry_before": initial_registry,
                "registry_after": registry_before_proposal, "unchanged": True,
                "training_steps_during_admission_and_materialization": 0, "planning_model_calls": 0})
            reviewed = phase("trusted_proposal_author", lambda:
                trusted_candidate.author_finite_repository_candidate(owner=owner, admission=admission,
                    review_ref="review:advisory-worker-fixed-offset-replacement"))
            _write(output / "reviewed-candidate.json", reviewed)
            generated = phase("actual_reserved_trusted_proposal", lambda:
                trusted_candidate.generate_finite_repository_candidate(owner=owner, admission=admission,
                    candidate=reviewed, output=private / "candidate-artifacts",
                    policy_observer=lambda bound: bound.roots))
            _write(output / "generated-candidate.json", generated)
            trusted_candidate.verify_generated_finite_repository_candidate(record=generated)
            replacement_pin = generated["artifacts"]["replacement"]
            replacement = Path(replacement_pin["path"]).read_bytes()
            if (len(replacement) != replacement_pin["bytes"]
                    or hashlib.sha256(replacement).hexdigest() != replacement_pin["sha256"]
                    or cid_for_bytes(replacement) != generated["replacement_cid"]):
                raise ValueError("trusted proposal replacement lost its retained exact identity")
            with native.server._lock:
                with IntentRepository(bound_connection=native.server._connection, install_schema=False) as intent:
                    tasks_after_proposal = _tasks(intent, materialized)
                    if tasks_after_proposal != tasks_before_proposal:
                        raise ValueError("trusted proposal changed native task population")
                    if (_registry_inventory(registry) != registry_before_proposal
                            or {name: _pin(repository / name) for name in inventory} != source_before_proposal):
                        raise ValueError("trusted proposal changed model registry or canonical source")
                    residual = intent.get_task(residual_task.task_cid)
                    candidate = phase("public_native_handoff_author", lambda:
                        author_finite_repository_candidate(admission=admission, intent=intent,
                            task_cid=residual_task.task_cid, after_bytes=replacement,
                            output=Path(handoff_root) / "candidate.json"))
            _write(output / "candidate-descriptor.json", candidate)
            bridge = {"schema": "finite-advisory-trusted-proposal-native-handoff@1",
                "generated_result_cid": generated["result_cid"],
                "generated_result_pin": _pin(output / "generated-candidate.json"),
                "reviewed_candidate_cid": generated["reviewed_candidate_cid"],
                "reviewed_candidate_pin": _pin(output / "reviewed-candidate.json"),
                "replacement_artifact": replacement_pin, "replacement_raw_cid": cid_for_bytes(replacement),
                "replacement_sha256": hashlib.sha256(replacement).hexdigest(),
                "before_raw_cid": cid_for_bytes(trusted_candidate.BEFORE_SOURCE),
                "before_sha256": candidate["before_sha256"], "after_sha256": candidate["after_sha256"],
                "public_candidate_descriptor": candidate, "public_candidate_pin": _pin(candidate["artifact"]),
                "finite_admission_cid": cid_for_structured(admission),
                "semantic_context_cid": candidate["semantic_context_cid"],
                "task_cid": residual["task_cid"], "task_revision": residual["revision"],
                "advisory_context_cids": [context.cid for context in (off, root_frozen, initial_child_frozen)],
                "advisory_comparison_pin": _pin(output / "initial-advisory-comparison.json"),
                "canonical_source_unchanged": True, "native_task_rows_unchanged": True,
                "model_registry_unchanged": True, "features_choose_fixed_edit": False,
                "execution_authority": False, "completion_authority": False, "publication_authority": False}
            if bridge["after_sha256"] != bridge["replacement_sha256"]:
                raise ValueError("public candidate did not receive the retained trusted replacement")
            _write(output / "candidate-bridge.json", bridge)
            _write(output / "proposal-owner-invariance.json", {"task_rows_before": tasks_before_proposal,
                "task_rows_after": tasks_after_proposal, "source_before": source_before_proposal,
                "registry_before": registry_before_proposal, "registry_after": _registry_inventory(registry)})
            evidence = {str(path): _pin(path) for root in (output / "cas", output / "before-preview",
                output / "materialization-preview") for path in root.rglob("*") if path.is_file()}
            for path in (output / "before-admission.json", output / "materialized.json",
                         Path(materialized["finite_admission_ref"]["path"]), Path(candidate["artifact"])):
                evidence[str(path)] = _pin(path)
            for pin in _owned_file_pins(private / "model-artifacts", advisory, private / "candidate-artifacts",
                    output / "before-lowering-artifacts"):
                evidence[pin["path"]] = pin
            for name in ("criteria.json", "training-selections.json", "initial-capture.json",
                    "initial-advisory-comparison.json", "reviewed-candidate.json", "generated-candidate.json",
                    "candidate-bridge.json", "proposal-owner-invariance.json", "before-lowering.json",
                    "admission-no-fit-observation.json"):
                evidence[str(output / name)] = _pin(output / name)
            _write(output / "historical-parent-artifact-pins.json", list(evidence.values()))
            command = shlex.join(["/opt/ipfs-supervisor/bin/owner-worker",
                "--finite-repository-artifact", candidate["artifact"],
                "--finite-repository-sha256", candidate["sha256"],
                "--finite-repository-task-cid", residual["task_cid"]])
            with _scope_cost(phases, "private_native_execution_scope_inclusive",
                    reserve_finite_repository_execution(owner=owner, admission=admission, candidate=candidate,
                        server=native.server, source=native.source, output=private / "launch-evidence",
                        policy_observer=lambda bound: bound.roots, admission_timeout_seconds=90)) as scope:
                _write(output / "execution-scope.json", scope.to_dict())
                runtime = AdmittedBenchmarkRuntime.create(private / "launch", admission=admission["local_admission"],
                    server=native.server, source=native.source, implement=True, implementation_command=command,
                    candidate_runner_argv=("/opt/ipfs-supervisor/bin/validation-worker",),
                    finite_execution_scope=scope, max_task_attempts=1, lifetime_seconds=300,
                    worker_worktree_root=Path(worktree_root), timeout_ms=30_000)
                try:
                    report["start"] = phase("actual_native_START", lambda: runtime.start()).to_dict()
                    if report["start"]["status"] != "succeeded":
                        raise ValueError("native START failed")
                    report["worker_launched"] = True
                    observations, previous = [], None
                    allocation_store = WorktreeLifecycleStore(repo_root=repository)
                    allocations = {}
                    deadline = time.monotonic() + 120
                    while time.monotonic() < deadline:
                        task = native.source.get_task(residual["task_cid"])
                        for allocation in allocation_store.iter_records():
                            if (allocation.canonical_task_cid == residual["task_cid"]
                                    or allocation.task_id == residual["task_alias"]):
                                allocations[allocation.record_id] = allocation.to_dict()
                        current = (task.status, task.revision)
                        if current != previous:
                            observations.append({"status": task.status, "revision": task.revision,
                                "seconds": time.monotonic() - started})
                            previous = current
                        if task.status in {"completed", "failed", "blocked", "cancelled"}:
                            break
                        if not runtime.process.snapshot(runtime.profile).members:
                            raise ValueError("native supervisor exited before residual completion")
                        time.sleep(0.25)
                    report["task_observations"] = observations
                    report["task"] = {"status": task.status, "revision": task.revision,
                        "body": dict(task.body)}
                    report["observed_worker_allocations"] = list(allocations.values())
                    report["resource_before_stop"] = scheduler.snapshot()
                finally:
                    report["stop"] = phase("actual_native_STOP_and_UID_cleanup", lambda: runtime.stop()).to_dict()
                    report["remaining_processes"] = len(runtime.process.snapshot(runtime.profile).members)
                    report["bootstrap_errors"] = runtime.bootstrap_errors
                    report["native_diagnostics"] = _native_diagnostics(runtime.state)
                    _write(output / "native-lifecycle.json", report)
                    phase("native_runtime_close", lambda: runtime.close())
                if (report["task"]["status"] != "completed" or report["stop"]["status"] != "succeeded"
                        or report["remaining_processes"] or report["bootstrap_errors"]):
                    raise ValueError("isolated native residual worker lifecycle is incomplete")
                report["execution_scope_after_stop"] = scope.to_dict()
        if _registry_inventory(registry) != registry_before_proposal:
            raise ValueError("signed admission or isolated native worker fitted or promoted a model")
        _write(output / "worker-no-fit-observation.json", {"registry_before": registry_before_proposal,
            "registry_after": _registry_inventory(registry), "unchanged": True,
            "training_steps_during_admission_and_worker": 0, "planning_model_calls": 0})
        report["resource_after_owner_close"] = scheduler.snapshot()
        published = _git(repository, "rev-parse", "HEAD")
        parents = _git(repository, "rev-list", "--parents", "-n", "1", published).split()
        if len(parents) != 3 or parents[1] != original_commit:
            raise ValueError("native publication is not one exact baseline two-parent merge")
        changed = _git(repository, "diff", "--name-only", original_commit, published).splitlines()
        if changed != ["calc.py"]:
            raise ValueError("published edit exceeded the original output permission")
        for name in ("check_type.py", "check_offset.py"):
            phase("post_STOP_public_check_" + name, lambda name=name:
                subprocess.run([sys.executable, "-B", name], cwd=repository, check=True,
                    capture_output=True, timeout=10))
        # Canonical database workers retain their allocated Git worktrees.
        # This fixture explicitly cleans only its published, fenced allocation;
        # native crash recovery and automatic completion cleanup remain separate.
        cleanup_records = []
        store = WorktreeLifecycleStore(repo_root=repository)
        linked = _git(repository, "worktree", "list", "--porcelain")
        for observed in allocations.values():
            allocation = WorkspaceLifecycleRecord.from_dict(observed)
            path = Path(allocation.workspace_path)
            if not path.exists():
                continue
            if (allocation.record_id not in allocations or path.resolve(strict=True) != path
                    or not path.is_relative_to(Path(worktree_root)) or path == Path(worktree_root)
                    or "worktree " + str(path) not in linked.splitlines()
                    or _git(path, "status", "--porcelain")):
                raise ValueError("published allocation identity/cleanliness cannot authorize fixture cleanup")
            if read_process_birth(allocation.owner.pid) == allocation.owner:
                raise ValueError("observed allocation still has its live owner birth")
            current_allocation = store.load_workspace(path)
            if current_allocation is not None and any(
                    getattr(current_allocation, field) != getattr(allocation, field)
                    for field in ("record_id", "task_id", "canonical_task_cid", "owner", "lease_id",
                                  "workspace_path", "branch", "state_dir")):
                raise ValueError("published allocation has a different current lifecycle owner")
            subprocess.run(["git", "-C", str(repository), "merge-base", "--is-ancestor",
                _git(path, "rev-parse", "HEAD"), published], check=True, capture_output=True, timeout=10)
            decision = store.authorize_cleanup(workspace_path=path, branch=allocation.branch,
                expected_state_dir=allocation.state_dir)
            cleanup_records.append({"allocation": allocation.to_dict(), "decision": decision.to_dict(),
                "native_lifecycle_record_after_stop": None if current_allocation is None else current_allocation.to_dict(),
                "scope": "explicit owner fixture cleanup after native STOP; not native completion recovery"})
            if not decision.allowed:
                raise ValueError("native lifecycle owner refused published fixture worktree cleanup")
            phase("explicit_owner_fixture_worktree_cleanup", lambda path=path:
                subprocess.run(["git", "-C", str(repository), "worktree", "remove", str(path)],
                    check=True, capture_output=True, timeout=30))
        _write(output / "owner-fixture-worktree-cleanup.json", cleanup_records)
        if (repository / ".git/worktrees").exists():
            if any((repository / ".git/worktrees").iterdir()):
                raise ValueError("unaccounted linked worktree prevents successor custody")
            (repository / ".git/worktrees").rmdir()
        old_current_rejected = False
        try:
            phase("reject_stale_current_admission", lambda:
                boundary.verify_current_finite_repository_admission(owner=owner, admission=admission,
                    output=output / "stale-preview", policy_observer=lambda bound: bound.roots))
        except ValueError:
            old_current_rejected = True
        if not old_current_rejected:
            raise ValueError("published source retained stale current admission")
        # A fresh independent profile preserves old owner-local history and
        # binds the actual published Git baseline; it does not rotate old keys.
        stale_advice = []
        for label, context in (("root-frozen", root_frozen), ("initial-child-frozen", initial_child_frozen)):
            try:
                phase("reject_stale_" + label, lambda context=context: verify_current_context(owner, registry, context))
            except ValueError as error:
                stale_advice.append({"label": label, "rejected": True, "error": str(error)})
            else:
                raise ValueError("published source retained stale initial advisory context")
        _write(output / "stale-advisory-refusals.json", stale_advice)
        successor = phase("published_successor_source_capture", lambda: index.prepare_current(repository,
            repository_id=first.repository_id, operation_id="published-successor", expected_head=first,
            scheduler=scheduler, admission_timeout_seconds=90)).head
        if successor.generation != first.generation + 1 or successor.snapshot_cid == first.snapshot_cid:
            raise ValueError("native publication did not produce the next exact source generation")
        new_profile, new_lifecycle = private / "successor-profile", private / "successor-lifecycle"
        Supervisor.init_local(repository=repository, consent=True, profile_dir=new_profile,
            lifecycle_dir=new_lifecycle)
        successor_owner = replace(owner, expected_head=successor)
        successor_capture = phase("successor_complete_metadata_capture", lambda:
            _capture_complete(index, repository, successor, scheduler))
        _write(output / "successor-capture.json", successor_capture)
        successor_child = train("successor_child_training", successor_owner, initial_child.version_id)
        continuations.append(_continuation(initial_child, successor_child))
        successor_frozen = phase("successor_child_frozen_prepare", lambda:
            prepare_codebase_feature_context(owner=successor_owner, registry=registry, mode="frozen",
                output=advisory / "successor-child-frozen-context", version_id=successor_child.version_id))
        successor_off = phase("successor_model_off_prepare", lambda:
            prepare_codebase_feature_context(owner=successor_owner, registry=registry,
                mode="model_off", output=advisory / "successor-off-context"))
        successor_registry = _registry_inventory(registry)
        successor_previews = [(label, advice(label, successor_owner, context, 2, []))
            for label, context in (("successor-off", successor_off), ("successor-frozen", successor_frozen))]
        successor_comparison = _comparison(successor_previews)
        _write(output / "successor-advisory-comparison.json", successor_comparison)
        if _registry_inventory(registry) != successor_registry:
            raise ValueError("successor advisory preview fitted or promoted a model")
        successor_lowering = phase("successor_lowering", lambda: _with_custody(successor_owner,
            lambda: prove_current_integer_offset_lowering(index, repository, expected_head=successor,
                contract=contract, tool_policy=tools, output=output / "successor-lowering-artifacts",
                scheduler=scheduler, timeout_seconds=90, memory_mb=1024)))
        _lowering_scope(successor_lowering, matches=True)
        _write(output / "successor-lowering.json", successor_lowering.to_dict())
        _update_progress(output / "training-measurements.json", {"measurements": measurements,
            "continuations": continuations})
        next_request = _request(index, repository, successor, document, catalog, _authority_materials())
        next_graph, next_manifest, next_bindings = _native_graph(repository=repository, request=next_request,
            profile=new_profile, lifecycle=new_lifecycle)
        if ([task.to_dict() for task in next_graph.tasks] != [task.to_dict() for task in graph.tasks]
                or next_bindings != bindings):
            raise ValueError("successor changed the complete original administrator task declarations")
        next_declaration = boundary.author_finite_repository_declaration(owner=successor_owner,
            manifest=next_manifest, request=next_request, intent_document=document, source_text=INTENT,
            operation_catalog=catalog, tool_policy=tools, task_bindings=next_bindings)
        next_admission = phase("signed_model_off_successor_no_work", lambda:
            boundary.admit_finite_repository_plan(owner=successor_owner, declaration=next_declaration,
                graph=next_graph, output=output / "successor-preview", policy_observer=lambda bound: bound.roots))
        _write(output / "successor-admission.json", next_admission)
        next_receipt = next_admission["receipt"]["payload"]
        if (next_admission["local_admission"] is not None
                or next_receipt["planning_permitted"] is not False
                or next_receipt["no_work_review_only"] is not True
                or next_receipt["semantic_context"]["residual_requirement_ids"]
                or next_receipt["semantic_context"]["finite_selected_task_ids"]):
            raise ValueError("no-work successor gained an empty task grant")
        with IntentRepository(private / "no-work.duckdb") as empty_intent:
            try:
                phase("reject_no_work_materialization", lambda:
                    boundary.materialize_finite_repository_plan(owner=successor_owner,
                        admission=next_admission, intent=empty_intent,
                        output=output / "no-work-materialization", policy_observer=lambda bound: bound.roots))
            except ValueError:
                no_work_population = empty_intent.list_tasks()
                if no_work_population:
                    raise ValueError("no-work review created native task rows")
            else:
                raise ValueError("no-work successor materialized an empty task grant")
        _write(output / "no-work-refusal.json", {"native_tasks": no_work_population,
            "no_work_materialization_rejected": True, "execution_grant": False})
        cold_root = output / "cold"
        cold_root.mkdir()
        cold_connection, cold_index = _open(cold_root)
        try:
            cold_head = phase("independent_cold_source_capture", lambda:
                cold_index.prepare_current(repository, repository_id=first.repository_id,
                    operation_id="cold-published-capture", expected_head=None, scheduler=scheduler,
                    admission_timeout_seconds=90)).head
            _write(output / "cold-capture.json", phase("independent_cold_complete_metadata_capture", lambda:
                _capture_complete(cold_index, repository, cold_head, scheduler)))
            cold_owner = replace(owner, index=cold_index, expected_head=cold_head)
            cold_request = _request(cold_index, repository, cold_head, document, catalog, _authority_materials())
            cold_graph, cold_manifest, cold_bindings = _native_graph(repository=repository,
                request=cold_request, profile=new_profile, lifecycle=new_lifecycle)
            cold_decl = boundary.author_finite_repository_declaration(owner=cold_owner,
                manifest=cold_manifest, request=cold_request, intent_document=document, source_text=INTENT,
                operation_catalog=catalog, tool_policy=tools, task_bindings=cold_bindings)
            cold_admission = phase("independent_cold_signed_model_off_admission", lambda:
                boundary.admit_finite_repository_plan(owner=cold_owner,
                    declaration=cold_decl, graph=cold_graph, output=output / "cold-preview",
                    policy_observer=lambda bound: bound.roots))
            _write(output / "cold-admission.json", cold_admission)
            fields = ("source_cid", "query", "domain_inputs", "observations",
                "eligible_requirement_ids", "residual_requirement_ids", "finite_selected_task_ids",
                "operation_catalog_cid", "model_status")
            successor_context = boundary.verify_finite_repository_admission(admission=next_admission)["semantic_context"]
            cold_context = boundary.verify_finite_repository_admission(admission=cold_admission)["semantic_context"]
            if any(successor_context[name] != cold_context[name] for name in fields):
                raise ValueError("native successor/cold finite outcomes differ")
            _write(output / "cold-comparison.json", {"agreement": True, "compared_fields": fields,
                "scope": "exact finite source/domain and clause outcomes; generation envelopes remain distinct"})
        finally:
            cold_connection.close()
        final_registry_before_reopen = _registry_inventory(registry)
        if final_registry_before_reopen != successor_registry:
            raise ValueError("successor admission or independent cold capture fitted or promoted a model")
        _write(output / "successor-admission-no-fit-observation.json", {"registry_before": successor_registry,
            "registry_after": final_registry_before_reopen, "unchanged": True,
            "training_steps_during_admission_and_cold_capture": 0, "planning_model_calls": 0})
        replay_pins = _owned_file_pins(private / "model-artifacts", advisory, private / "candidate-artifacts",
            output / "cas", output / "before-lowering-artifacts", output / "successor-lowering-artifacts")
        replay_pins += list(evidence.values())
        _write(output / "cold-request.json", {"schema": "finite-advisory-worker-current-replay-request@1",
            "head": successor.to_dict(), "tool_policy": tools,
            "selected_context": str(successor_frozen.output),
            "selected_context_binding": successor_frozen.material_binding,
            "registry_before_reopen": final_registry_before_reopen,
            "retained_pins": replay_pins})
        registry.close()
        registry = None
        connection.close()
        connection = None
        fresh = phase("fresh_original_successor_catalog_and_registry_replay", lambda:
            subprocess.run([sys.executable, "-m", MODULE, "replay", str(output),
                "--response-file", str(output / "fresh-process-response.json")],
                capture_output=True, text=True, timeout=90))
        _write(output / "fresh-process.json", {"returncode": fresh.returncode,
            "stdout": fresh.stdout, "stderr": fresh.stderr})
        if fresh.returncode:
            raise ValueError("fresh successor registry replay failed: " + fresh.stderr)
        fresh_result = json.loads((output / "fresh-process-response.json").read_bytes())
        _write(output / "fresh-process-replay.json", fresh_result)
        if any(_pin(path) != pin for path, pin in evidence.items()):
            raise ValueError("worker/successor work altered historical evidence")
        if any(_pin(pin["path"]) != pin for pin in pins):
            raise ValueError("selected producer changed during native qualification")
        final_resources = scheduler.snapshot()
        if final_resources["active_lease_count"] or final_resources["waiting_request_count"]:
            raise ValueError("native resource leases or waiters remain")
        observed_training = _training_ledger(training_attempts)
        if (observed_training["known_actual_attempted_epochs"] != 48
                or observed_training["unknown_fitting_attempt_count"]):
            raise ValueError("joined qualification did not account for all three actual training phases")
        report.update(status="completed", elapsed_seconds=time.monotonic() - started,
            actual_attempted_training_epochs=observed_training["known_actual_attempted_epochs"],
            training_measurements=measurements, continuations=continuations,
            selected_initial_model_version_id=initial_child.version_id,
            selected_successor_model_version_id=successor_child.version_id,
            initial_advisory_comparison=initial_comparison,
            successor_advisory_comparison=successor_comparison,
            trusted_proposal_generated=True, exact_trusted_replacement_feeds_native_handoff=True,
            candidate_bridge=bridge,
            frozen_current_successor_replayed_without_fitting=True,
            metadata_hydration_performed=False, previous_head_model_fallback_used=False,
            model_off_advisory_fallback_available=True,
            original_head=first.to_dict(), successor_head=successor.to_dict(),
            audit_artifacts={"contexts": {
                    "root_training": "private/advisory/root-training-context",
                    "root_frozen": "private/advisory/root-frozen-context",
                    "initial_child_training": "private/advisory/initial-child-training-context",
                    "initial_child_frozen": "private/advisory/initial-child-frozen-context",
                    "initial_off": "private/advisory/root-off-context",
                    "successor_child_training": "private/advisory/successor-child-training-context",
                    "successor_child_frozen": "private/advisory/successor-child-frozen-context",
                    "successor_off": "private/advisory/successor-off-context"},
                "previews": {"initial_off": "private/advisory/initial-off-preview.json",
                    "root_frozen": "private/advisory/root-frozen-preview.json",
                    "initial_child_frozen": "private/advisory/initial-child-frozen-preview.json",
                    "successor_off": "private/advisory/successor-off-preview.json",
                    "successor_frozen": "private/advisory/successor-frozen-preview.json"},
                "initial_comparison": "initial-advisory-comparison.json",
                "successor_comparison": "successor-advisory-comparison.json",
                "controls": "stale-advisory-refusals.json",
                "cold_response": "fresh-process-replay.json",
                "no_fit_inventories": ["admission-no-fit-observation.json", "worker-no-fit-observation.json",
                    "successor-admission-no-fit-observation.json",
                    *["private/advisory/" + label + "-preview-artifacts-registry-inventory.json"
                        for label in ("initial-off", "root-frozen", "initial-child-frozen", "successor-off", "successor-frozen")]],"criteria": "criteria.json", "training_attempts": "training-attempts.json",
                "training_measurements": "training-measurements.json", "phase_costs": "phase-costs.json",
                "initial_capture": "initial-capture.json", "successor_capture": "successor-capture.json",
                "initial_advisory_comparison": "initial-advisory-comparison.json",
                "successor_advisory_comparison": "successor-advisory-comparison.json",
                "advisory_directory": "private/advisory", "candidate_bridge": "candidate-bridge.json",
                "reviewed_candidate": "reviewed-candidate.json", "generated_candidate": "generated-candidate.json",
                "public_candidate_descriptor": "candidate-descriptor.json", "native_lifecycle": "native-lifecycle.json",
                "parent_artifact_pins": "historical-parent-artifact-pins.json",
                "successor_admission": "successor-admission.json", "cold_admission": "cold-admission.json",
                "fresh_process_replay": "fresh-process-replay.json",
                "fresh_process_response": "fresh-process-response.json", "cold_request": "cold-request.json"},
            published_commit=published, published_commit_parents=parents[1:], changed_paths=changed,
            complete_task_population_retained=True, actual_public_checks_passed=True,
            stale_current_admission_rejected=True, historical_artifacts_unchanged=True,
            fresh_process_historical_replay=fresh_result,
            successor_cold_finite_outcomes_agree=True, no_work_successor_grants_no_task_omission=True,
            execution_sources=pins, active_leases=0, waiting_requests=0,
            native_worker_successor_loop_qualified=True,
            qualification_scope="one authored ten-file, two-task, five-input advisory structural adaptation and unchanged model-off native isolated worker publication")
        _write(output / "result.json", report)
        return report
    finally:
        if registry is not None:
            # Record partial durable observations even after a failed fitting
            # call; unknown work before a return is never represented as zero.
            try:
                _write(output / "last-native-registry-inventory.json", _registry_inventory(registry))
            finally:
                registry.close()
        if connection is not None:
            connection.close()


def replay(output):
    from ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview import RepositoryPlanPreviewOwner
    from ipfs_accelerate_py.agent_supervisor.planning.codebase_feature_context import verify_current_context
    from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    output = Path(output)
    historical = _historical_replay(output)
    request = json.loads((output / "cold-request.json").read_bytes())
    for pin in request["retained_pins"]:
        if _pin(pin["path"]) != pin:
            raise ValueError("fresh process lost immutable parent artifact")
    scheduler = _scheduler(output / "fresh-current-resource-admission.json")
    connection, index = _open(output)
    registry = AutoencoderRegistry(output / "private/train.duckdb", output / "private/model-artifacts")
    try:
        owner = RepositoryPlanPreviewOwner(index=index, repository=output / "repository",
            expected_head=CodebaseHead.from_dict(request["head"]), scheduler=scheduler,
            timeout_seconds=90, memory_mb=1024)
        before = _registry_inventory(registry)
        parent = request["registry_before_reopen"]
        if (before["owner_generation"] != parent["owner_generation"] + 1
                or before["artifacts"] != parent["artifacts"]
                or {key: value for key, value in before["tables"].items() if key != "meta"}
                    != {key: value for key, value in parent["tables"].items() if key != "meta"}):
            raise ValueError("native registry reopen changed data beyond its legitimate owner generation")
        context = _context(request["selected_context"])
        verified = verify_current_context(owner, registry, context)
        if verified != request["selected_context_binding"]:
            raise ValueError("fresh selected successor model context differs")
        after = _registry_inventory(registry)
        if after != before:
            raise ValueError("fresh frozen successor verification fitted or promoted a model")
        _idle(scheduler)
        for pin in request["retained_pins"]:
            if _pin(pin["path"]) != pin:
                raise ValueError("fresh numerical replay changed immutable parent artifact")
        return historical | {"schema": "finite-advisory-worker-current-and-historical-replay@1",
            "selected_context_binding": verified, "source_head": owner.expected_head.to_dict(),
            "registry_owner_generation_before_reopen": parent["owner_generation"],
            "registry_owner_generation_after_reopen": before["owner_generation"],
            "registry_reopen_scope": "Native owner generation advances once; numerical records and artifacts remain exact.",
            "registry_before_verification": before, "registry_after_verification": after,
            "current_successor_feature_verified": True, "training_steps": 0,
            "fitting_performed": False, "promotion_performed": False,
            "independent_cold_generation_used_for_feature_verification": False}
    finally:
        registry.close()
        connection.close()


def run(output, *, python_executable, lean_executable, handoff_root, worktree_root):
    output = Path(output).absolute()
    if output.exists() or output.parent.resolve() != output.parent:
        raise ValueError("fresh exact qualification output required")
    phases, training_attempts = [], []
    started = time.monotonic()
    try:
        return _run(output, python_executable=python_executable, lean_executable=lean_executable,
            handoff_root=handoff_root, worktree_root=worktree_root,
            phases=phases, training_attempts=training_attempts)
    except BaseException as error:
        if output.is_dir():
            _write(output / "failure.json", {"schema": SCHEMA, "status": "failed",
                "error_type": type(error).__name__, "error": str(error),
                "elapsed_seconds": time.monotonic() - started,
                "training_observation": _training_ledger(training_attempts),
                "worker_completion_claimed": False, "production_activated": False})
        raise
    finally:
        if output.is_dir():
            _update_progress(output / "phase-costs.json", phases)
            _update_progress(output / "training-attempts.json", _training_ledger(training_attempts))
            _write(output / "total-cost.json", {"elapsed_seconds": time.monotonic() - started,
                "phase_cost_scope": "Observed call durations; inclusive scope entries overlap nested phases."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "replay"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--lean", type=Path, default=Path("/toolchains/lean/bin/lean"))
    parser.add_argument("--handoff-root", type=Path, default=Path("/opt/ipfs-supervisor/finite-handoffs"))
    parser.add_argument("--worktree-root", type=Path, default=Path("/opt/ipfs-supervisor/worktrees"))
    parser.add_argument("--response-file", type=Path)
    args = parser.parse_args()
    if args.response_file is not None and (args.action != "replay"
            or args.response_file.absolute() != args.output.absolute() / "fresh-process-response.json"):
        parser.error("response file must be the exact replay output artifact")
    result = replay(args.output) if args.action == "replay" else run(args.output,
        python_executable=args.python, lean_executable=args.lean,
        handoff_root=args.handoff_root, worktree_root=args.worktree_root)
    if args.response_file is not None:
        # Diagnostics from imported native code remain in the captured streams;
        # the parent consumes this exclusive, complete machine response.
        _write(args.response_file, result)
    printed = result if args.action == "replay" else {"schema": result["schema"], "status": result["status"]}
    print(json.dumps(printed, sort_keys=True))


if __name__ == "__main__":
    main()
