"""Bind prepared advisory artifacts at the actual finite native worker spawn.

The signed worker admission stays model-off. Learned features are separately
verified advisory inputs; a verified retained proposal supplies the actual
public worker handoff. Two late callback mutations must refuse before Popen;
positive publication, STOP, successor and cold replay retain the existing path.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import time

from .finite_repository_admission_experiment import (
    _git, _pin, _write,
)
from .terminal_codebase_finite_experiment import INTENT
from .terminal_codebase_finite_service_experiment import (
    _authority_materials, _catalog, _open, _request, _scheduler,
)

SCHEMA = "finite-advisory-spawn-native-worker-qualification@1"
MODULE = "benchmarks.agent_supervisor.container_coding.finite_advisory_spawn_experiment"
from .finite_repository_worker_experiment import SOURCES as WORKER_SOURCES
from .finite_repository_candidate_experiment import (
    _SOURCES as CANDIDATE_SOURCES, _capture_complete, _projection_rows, _tasks,
)
from .finite_repository_advisory_worker_support import (
    AdvisoryWorkerSupport, create_advisory_worker_sources,
)
SOURCES = tuple(dict.fromkeys((MODULE,
    "benchmarks.agent_supervisor.container_coding.run_finite_advisory_spawn_docker",
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_advisory_artifact_closure",
    "ipfs_accelerate_py.agent_supervisor.control.before_popen_refusal",
    "ipfs_accelerate_py.agent_supervisor.control.control_plane",
    "ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator",
    "ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge",
    "ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner",
    "benchmarks.agent_supervisor.container_coding.finite_repository_advisory_worker_support",
    "benchmarks.agent_supervisor.container_coding.finite_repository_sharded_metadata",
    *WORKER_SOURCES, *CANDIDATE_SOURCES)))


from .finite_repository_worker_experiment import _native_graph


def _write_progress(path, value):
    """Update this fresh namespace's bounded in-progress retention manifest."""
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(raw) > 4 * 1024**2:
        raise ValueError("advisory progress manifest exceeds 4 MiB")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)


def _completion_retirement_observation(runtime):
    """Observe actual exact-handle retirement after genuine runtime close."""
    from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import OwnerLocalCompletionService
    with runtime.server._lock:
        service = runtime.completion_service
        gateway = runtime.server._command_gateway
        result = {"service_retired": type(service) is OwnerLocalCompletionService and service.retired,
            "owner_validation_handler_unbound": gateway._local_task_validation_handler is None,
            "owner_validation_binding_unbound": gateway._local_task_validation_binding is None,
            "owner_validation_callbacks_active": gateway._local_task_validation_active,
            "retirement_token_serialized": False}
    if (result["service_retired"] is not True or result["owner_validation_handler_unbound"] is not True
            or result["owner_validation_binding_unbound"] is not True
            or result["owner_validation_callbacks_active"] != 0):
        raise ValueError("native runtime close did not retire its exact inactive owner validation callback")
    return result


def _snapshot_closure_artifacts(closure, *, output, archive, snapshots):
    """Retain original seal bytes before callbacks, including mutable registry."""
    from ipfs_accelerate_py.agent_supervisor.runtime.finite_advisory_artifact_closure import _read
    reference = closure.material_binding
    payload = reference["signed_closure"]["payload"]
    archive.mkdir(mode=0o700, exist_ok=True)
    stored = sum(path.stat().st_size for path in archive.iterdir())
    def snapshot(original):
        observed, raw = _read(original["path"], original)
        target = archive / (observed["sha256"] + ".blob")
        if target.exists():
            if target.is_symlink() or target.read_bytes() != raw:
                raise ValueError("retained advisory artifact snapshot differs")
        else:
            nonlocal stored
            stored += len(raw)
            if stored > 128 * 1024**2:
                raise ValueError("deduplicated advisory artifact snapshot exceeds 128 MiB")
            with target.open("xb") as stream:
                stream.write(raw)
            target.chmod(0o444)
        return {"original": observed, "snapshot": _pin(target)}
    row = {"closure_cid": reference["closure_cid"],
        "files": [snapshot(original) for original in payload["files"]],
        "seal_snapshot": snapshot(reference["artifact"])}
    closure.require_detached(**{key: payload[key] for key in (
        "head", "finite_admission_cid", "semantic_context_cid", "candidate", "administrator_task_cids")})
    snapshots.append(row)
    _write_progress(output / "advisory-artifact-snapshots.json", {
        "schema": "finite-advisory-launch-artifact-snapshots@1", "closures": snapshots})
    return row


def _prepare_advisory_closure(*, owner, admission, candidate, support, output,
                             archive, snapshots, evidence_output):
    from ipfs_accelerate_py.agent_supervisor.runtime.finite_advisory_artifact_closure import (
        prepare_finite_advisory_artifact_closure,
    )
    closure = prepare_finite_advisory_artifact_closure(owner=owner, registry=support.registry,
        admission=admission, candidate=candidate, training_context=support.root_context,
        frozen_context=support.frozen, lowering_proof=support.before_lowering,
        contract=support.contract, tool_policy=support.tools, reviewed_candidate=support.reviewed,
        generated_candidate=support.generated, bridge=support.bridge,
        artifact_paths=support.parent_pins, output=output)
    _snapshot_closure_artifacts(closure, output=evidence_output, archive=archive, snapshots=snapshots)
    return closure


def _mutate_advisory_artifact(path, mutation):
    """Fixture-only mutation after the genuine prelaunch callback returns."""
    from ipfs_accelerate_py.agent_supervisor.runtime.finite_advisory_artifact_closure import _read
    before, raw = _read(path)
    mode = stat.S_IMODE(before["witness"][2])
    if mutation == "append_checkpoint_byte":
        with path.open("ab") as stream:
            stream.write(b"\n")
    elif mutation == "replace_context_inode_same_bytes":
        temporary = path.with_name(path.name + ".late-callback-replacement")
        with temporary.open("xb") as stream:
            stream.write(raw)
        temporary.chmod(mode)
        os.replace(temporary, path)
    else:
        raise ValueError("exact named advisory callback mutation required")
    changed, _ = _read(path)
    if (mutation == "append_checkpoint_byte" and changed["sha256"] == before["sha256"]
            or mutation == "replace_context_inode_same_bytes" and (
                changed["sha256"] != before["sha256"] or changed["size_bytes"] != before["size_bytes"]
                or changed["witness"][1] == before["witness"][1])):
        raise ValueError("actual advisory mutation did not establish its intended byte/inode change")
    return before, changed, raw, mode


def _restore_advisory_artifact(path, raw, mode):
    from ipfs_accelerate_py.agent_supervisor.runtime.finite_advisory_artifact_closure import _read
    with path.open("wb") as stream:
        stream.write(raw)
    path.chmod(mode)
    observed, restored = _read(path)
    if restored != raw:
        raise ValueError("advisory callback fixture restoration differs")
    return observed


def _run_late_callback_control(*, owner, admission, candidate, native, command,
                              worktree_root, output, closure, mutation, artifact_path,
                              task_snapshot, scheduler):
    """Run actual START; mutate at its last callback, then prove no spawn."""
    from ipfs_accelerate_py.agent_supervisor.entrypoints import admitted_benchmark_runtime as native_runtime
    from ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_execution import reserve_finite_repository_execution
    output.mkdir(mode=0o700)
    tasks_before = task_snapshot()
    sql_before = _native_evidence_snapshot(native.server)["tables"]
    publication_before = _git(owner.repository, "rev-parse", "HEAD")
    record = {"schema": "finite-advisory-late-callback-spawn-refusal@1", "case": mutation,
        "closure_reference": closure.material_binding, "artifact_path": str(artifact_path),
        "task_rows_before": tasks_before, "publication_head_before": publication_before,
        "native_effect_rows_before": {key: sql_before[key] for key in (
            "task_claims", "task_attempts", "merge_attempts", "completion_receipts")},
        "armed_actual_popen_callback": False, "after_verified_callback": False,
        "injection": False, "children_count": 0, "workerPopen_count": 0,
        "popen_count_scope": "actual admitted runtime supervisor subprocess.Popen; no supervisor implies no delegated worker",
        "claim_delta": 0, "attempt_delta": 0, "publication_delta": 0,
        "completion_authority": False, "authenticated_process_origin": False,
        "atomicity_attested": False, "production_activated": False}
    original_bytes = original_mode = None
    started = time.monotonic()
    try:
        with reserve_finite_repository_execution(owner=owner, admission=admission, candidate=candidate,
                server=native.server, source=native.source, output=output / "execution-evidence",
                policy_observer=lambda bound: bound.roots, admission_timeout_seconds=90,
                advisory_closure=closure) as scope:
            record["execution_scope"] = scope.to_dict()
            runtime = native_runtime.AdmittedBenchmarkRuntime.create(output / "launch",
                admission=admission["local_admission"], server=native.server, source=native.source,
                implement=True, implementation_command=command,
                candidate_runner_argv=("/opt/ipfs-supervisor/bin/validation-worker",),
                finite_execution_scope=scope, max_task_attempts=1, lifetime_seconds=300,
                worker_worktree_root=Path(worktree_root), timeout_ms=30_000)
            real_callback = runtime.process._popen
            real_require = scope.require_runtime
            real_subprocess = native_runtime.subprocess
            def counted_popen(*args, **kwargs):
                record["workerPopen_count"] += 1
                return real_subprocess.Popen(*args, **kwargs)
            class CountNativePopen:
                Popen = staticmethod(counted_popen)
                def __getattr__(self, name):
                    return getattr(real_subprocess, name)
            def after_verified(runtime_argument, *, before_spawn=False, stopping=False):
                nonlocal original_bytes, original_mode
                result = real_require(runtime_argument, before_spawn=before_spawn, stopping=stopping)
                if before_spawn and not record["injection"]:
                    record["after_verified_callback"] = True
                    before, changed, original_bytes, original_mode = _mutate_advisory_artifact(
                        artifact_path, mutation)
                    record.update(before=before, changed=changed, injection=True)
                return result
            def armed_callback(*args, **kwargs):
                record["armed_actual_popen_callback"] = True
                scope.require_runtime = after_verified
                native_runtime.subprocess = CountNativePopen()
                try:
                    return real_callback(*args, **kwargs)
                finally:
                    scope.require_runtime = real_require
                    native_runtime.subprocess = real_subprocess
            runtime.process._popen = armed_callback
            try:
                try:
                    record["start"] = runtime.start().to_dict()
                    record["start_refused"] = record["start"]["status"] != "succeeded"
                except Exception as error:
                    record.update(start_refused=True, start_exception={
                        "type": type(error).__name__, "message": str(error)[:4096]})
                record["children_count"] = len(runtime._children)
                record["remaining_processes_before_stop"] = len(runtime.process.snapshot(runtime.profile).members)
                sql_after = _native_evidence_snapshot(native.server)["tables"]
                record["claim_delta"] = len(sql_after["task_claims"]) - len(sql_before["task_claims"])
                record["attempt_delta"] = len(sql_after["task_attempts"]) - len(sql_before["task_attempts"])
                record["native_claim_rows_before"] = sql_before["task_claims"]
                record["native_claim_rows_after"] = sql_after["task_claims"]
                record["native_attempt_rows_before"] = sql_before["task_attempts"]
                record["native_attempt_rows_after"] = sql_after["task_attempts"]
                record["native_effect_rows_after"] = {key: sql_after[key] for key in (
                    "task_claims", "task_attempts", "merge_attempts", "completion_receipts")}
                record["publication_head_after"] = _git(owner.repository, "rev-parse", "HEAD")
                record["publication_delta"] = int(record["publication_head_after"] != publication_before)
                record["task_rows_after"] = task_snapshot()
                record["task_population_unchanged"] = record["task_rows_after"] == tasks_before
                record["native_effect_rows_unchanged"] = (
                    record["native_effect_rows_before"] == record["native_effect_rows_after"])
                if (not all(record[key] is True for key in (
                        "armed_actual_popen_callback", "after_verified_callback", "injection",
                        "start_refused", "task_population_unchanged", "native_effect_rows_unchanged"))
                        or any(record[key] != 0 for key in ("children_count", "workerPopen_count",
                            "remaining_processes_before_stop", "claim_delta", "attempt_delta", "publication_delta"))):
                    raise ValueError("late advisory callback did not refuse before actual Popen and native task effects")
            finally:
                runtime.process._popen = real_callback
                record["stop"] = runtime.stop().to_dict()
                record["remaining_processes"] = len(runtime.process.snapshot(runtime.profile).members)
                record["bootstrap_errors"] = list(runtime.bootstrap_errors)
                runtime.close()
                record.update(_completion_retirement_observation(runtime))
            if (record["stop"]["status"] != "succeeded" or record["remaining_processes"]
                    or record["bootstrap_errors"]):
                raise ValueError("late advisory refusal failed native STOP or isolated UID cleanup")
            record["execution_scope_after_stop"] = scope.to_dict()
        record["envelope_released"] = scope.parent_lease.released
        record["native_leases_after"] = scheduler.snapshot()
        if (not record["envelope_released"] or record["native_leases_after"]["active_lease_count"]
                or record["native_leases_after"]["waiting_request_count"]):
            raise ValueError("late advisory refusal retained a resource lease or waiter")
        record["status"] = "completed"
        return record
    finally:
        if original_bytes is not None:
            record["restored"] = _restore_advisory_artifact(artifact_path, original_bytes, original_mode)
        record["elapsed_seconds"] = time.monotonic() - started
        _write(output / "result.json", record)


def _stop_with_advisory_drift(runtime, artifact_path):
    """Actual STOP stays available despite advisory receipt byte drift."""
    before, changed, raw, mode = _mutate_advisory_artifact(artifact_path, "append_checkpoint_byte")
    observation = {"schema": "finite-advisory-drift-during-native-stop@1", "injection": True,
        "artifact_path": str(artifact_path), "before": before, "changed": changed,
        "scope": "actual receipt byte drift before STOP; source publication is separately already stale",
        "authenticated_process_origin": False, "atomicity_attested": False}
    try:
        receipt = runtime.stop().to_dict()
        observation.update(stop_succeeded=receipt["status"] == "succeeded",
            remaining_processes=len(runtime.process.snapshot(runtime.profile).members))
        return receipt, observation
    finally:
        observation["restored"] = _restore_advisory_artifact(artifact_path, raw, mode)


def _native_evidence_snapshot(server):
    """Copy complete, explicitly selected public task/evidence SQL projections."""
    tables = ("objectives", "goals", "plans", "tasks", "task_dependencies",
        "task_outputs", "task_validations", "task_acceptance", "task_attempts",
        "task_claims", "leases", "fencing_epochs", "validation_runs",
        "validation_results", "completion_receipts", "merge_attempts")
    result = {}
    with server._lock:
        for table in tables:
            cursor = server._connection.execute('SELECT * FROM "' + table + '"')
            values = cursor.fetchall()
            if any(not isinstance(row, Mapping) for row in values):
                raise ValueError("native owner evidence requires its complete named row projection")
            rows = [dict(row) for row in values]
            result[table] = sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))
    return {"schema": "finite-advisory-worker-native-sql-evidence@1",
        "scope": "complete selected task/claim/public-validation/completion projections; no credentials or owner vault",
        "tables": result}


def replay(output):
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


def run(output, *, python_executable, lean_executable, handoff_root, worktree_root):
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
    pins = [_pin(importlib.import_module(name).__file__) for name in SOURCES]
    copies = output / "selected-source-snapshot"
    copies.mkdir()
    for name, pin in zip(SOURCES, pins):
        path = copies / (name + ".py")
        path.write_bytes(Path(pin["path"]).read_bytes())
        if _pin(path)["sha256"] != pin["sha256"]:
            raise ValueError("selected producer changed during sequential copy")
    repository = output / "repository"
    inventory = create_advisory_worker_sources(repository)
    original_commit = _git(repository, "rev-parse", "HEAD")
    profile, lifecycle = private / "profile", private / "lifecycle"
    Supervisor.init_local(repository=repository, consent=True, profile_dir=profile, lifecycle_dir=lifecycle)
    document, catalog = build_finite_integer_intent(INTENT), _catalog()
    tools = seal_finite_integer_tools(python_executable=Path(python_executable).resolve(strict=True),
        lean_executable=Path(lean_executable).resolve(strict=True))
    scheduler = _scheduler(private / "resource-admission.json")
    connection, index = _open(output)
    report = {"schema": SCHEMA, "status": "incomplete", "worker_launched": False,
        "training_during_worker_execution": 0, "provider_calls": 0, "production_activated": False,
        "signed_worker_feature_mode": "model_off", "learned_features_authorize_execution": False,
        "advisory_final_spawn_closure_qualified": False,
        "advisory_spawn_scope": "bounded cooperative detached observation after callbacks immediately before actual Popen",
        "authenticated_advisory_process_origin": False, "advisory_atomicity_attested": False,
        "launch_artifact_bytes_hydrated": False,
        "launch_artifact_snapshot_scope": "complete original signed file population retained separately; metadata indexes snapshot bytes",
        "convergence_proved": False, "generalization_verified": False,
        "formal_decoder_available": False, "parser_correctness_proved": False,
        "public_terminal_bench_task_satisfied": False,
        "task_omission_authority": False, "universal_python_semantics_proved": False,
        "inventory_paths": sorted(inventory), "original_commit": original_commit}
    support = AdvisoryWorkerSupport(output=output)
    try:
        initial_publication = index.prepare_current(repository, repository_id="repository:finite-advisory-spawn-qualification",
            operation_id="initial", expected_head=None, scheduler=scheduler,
            admission_timeout_seconds=90)
        first = initial_publication.head
        report["original_head"] = first.to_dict()
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
        report["advisory_initial"] = support.prepare_initial(owner=owner, tool_policy=tools)
        request = _request(index, repository, first, document, catalog, _authority_materials())
        graph, manifest, bindings = _native_graph(repository=repository, request=request,
            profile=profile, lifecycle=lifecycle)
        tasks = {task.task_key: task for task in graph.tasks}
        declaration = boundary.author_finite_repository_declaration(owner=owner, manifest=manifest,
            request=request, intent_document=document, source_text=INTENT, operation_catalog=catalog,
            tool_policy=tools, task_bindings=bindings)
        admission = boundary.admit_finite_repository_plan(owner=owner, declaration=declaration,
            graph=graph, output=output / "before-preview", policy_observer=lambda bound: bound.roots)
        selected_source = boundary.verify_finite_repository_admission(admission=admission)["semantic_context"]["source_cid"]
        if selected_source != entries["calc.py"].source_cid or selected_source == entries["decoy.py"].source_cid:
            raise ValueError("same-name decoy changed the selected source identity")
        _write(output / "before-admission.json", admission)
        with IntentRepository(private / "intent.duckdb") as intent:
            materialized = boundary.materialize_finite_repository_plan(owner=owner, admission=admission,
                intent=intent, output=output / "materialization-preview", policy_observer=lambda bound: bound.roots)
            _write(output / "materialized.json", materialized)
        (private / "intent.duckdb").chmod(0o600)
        with open_existing_native_owner(database=private / "intent.duckdb", checkout=repository,
                state_dir=private / "owner", repository_id=manifest["payload"]["repository_cid"],
                execution_routes={task.task_key: GROK_CODEX_EXECUTION_MODE for task in graph.tasks}) as native:
            driver = DatabaseImplementationDaemon(database_path=native.database,
                coordination_path=private / "prerequisite-coordination.duckdb",
                execution_path=private / "prerequisite-execution.duckdb", authority_mode="quack",
                task_source_kind="duckdb", owner_session_id="session:finite-worker-prerequisite",
                process_instance_id=native.identity.process_birth_id, quack_uri=native.identity.listen_uri,
                task_source=native.source, close_task_source=False,
                state_owner_bootstrap_credentials=native.credentials, strict_task_sharding=True,
                max_task_attempts=1, lease_ms=120_000, require_real_execution=True).open()
            try:
                attempt = driver.claim_next()
                if attempt is None or attempt.task_cid != tasks["FINITE-TYPE"].task_cid:
                    raise ValueError("actual native prerequisite claim selected a different task")
                prerequisite = native.source.get_task(attempt.task_cid)
                checked = run_owner_local_task_validations(server=native.server, task_cid=attempt.task_cid,
                    attempt_id=attempt.attempt_id, expected_revision=prerequisite.revision)
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
            def task_snapshot():
                with native.server._lock:
                    with IntentRepository(bound_connection=native.server._connection, install_schema=False) as intent:
                        return _tasks(intent, materialized)
            report["task_rows_before_candidate"] = task_snapshot()
            after_bytes = support.candidate_bytes(owner=owner, admission=admission,
                task_snapshot=task_snapshot)
            with native.server._lock:
                with IntentRepository(bound_connection=native.server._connection, install_schema=False) as intent:
                    residual = intent.get_task(residual_task.task_cid)
                    candidate = author_finite_repository_candidate(admission=admission, intent=intent,
                        task_cid=residual_task.task_cid, after_bytes=after_bytes,
                        output=Path(handoff_root) / "candidate.json")
            report["proposal_worker_bridge"] = support.bind_worker_descriptor(
                admission=admission, candidate=candidate, task_cid=residual_task.task_cid)
            _write(output / "candidate-descriptor.json", candidate)
            evidence = {str(path): _pin(path) for root in (output / "cas", output / "before-preview",
                output / "materialization-preview") for path in root.rglob("*") if path.is_file()}
            for path in (output / "before-admission.json", output / "materialized.json",
                         Path(materialized["finite_admission_ref"]["path"]), Path(candidate["artifact"])):
                evidence[str(path)] = _pin(path)
            _write(output / "historical-parent-artifact-pins.json", list(evidence.values()))
            command = shlex.join(["/opt/ipfs-supervisor/bin/owner-worker",
                "--finite-repository-artifact", candidate["artifact"],
                "--finite-repository-sha256", candidate["sha256"],
                "--finite-repository-task-cid", residual["task_cid"]])
            report["advisory_prelaunch_verification"] = support.check_prelaunch(owner=owner)
            closure_root = private / "advisory-spawn-closures"
            closure_root.mkdir(mode=0o700)
            archive = private / "advisory-artifact-snapshots"
            snapshots, controls, closure_references = [], [], []
            control_root = private / "advisory-spawn-controls"
            control_root.mkdir(mode=0o700)
            for mutation, path in (
                    ("append_checkpoint_byte", support.frozen.output / "checkpoint.json"),
                    ("replace_context_inode_same_bytes", support.frozen.output / "inference.json")):
                control_closure = support._phase("prepare_spawn_control_" + mutation, lambda mutation=mutation:
                    _prepare_advisory_closure(owner=owner, admission=admission, candidate=candidate,
                        support=support, output=closure_root / mutation, archive=archive,
                        snapshots=snapshots, evidence_output=output))
                closure_references.append(control_closure.material_binding)
                control = support._phase("native_spawn_control_" + mutation, lambda mutation=mutation, path=path:
                    _run_late_callback_control(owner=owner, admission=admission, candidate=candidate,
                        native=native, command=command, worktree_root=worktree_root,
                        output=control_root / mutation, closure=control_closure, mutation=mutation,
                        artifact_path=path, task_snapshot=task_snapshot, scheduler=scheduler))
                controls.append(control)
                _write_progress(output / "advisory-spawn-controls.json", controls)
                support.check_historical_pins()
            advisory_closure = support._phase("prepare_positive_spawn_closure", lambda:
                _prepare_advisory_closure(owner=owner, admission=admission, candidate=candidate,
                    support=support, output=closure_root / "positive", archive=archive,
                    snapshots=snapshots, evidence_output=output))
            closure_references.append(advisory_closure.material_binding)
            report["advisory_closure_reference"] = advisory_closure.material_binding
            report["advisory_spawn_controls"] = controls
            report["launch_artifact_archive_retained"] = True
            _write(output / "advisory-closure-reference.json", advisory_closure.material_binding)
            support._extend({"advisory_spawn_closures": closure_references,
                "advisory_spawn_controls": controls,
                "advisory_launch_artifacts": [{"schema": "finite-advisory-launch-artifact-snapshots@1",
                    "closures": snapshots, "byte_payload_hydrated": False,
                    "scope": "complete separately retained original seal byte populations indexed by native metadata"}]})
            with reserve_finite_repository_execution(owner=owner, admission=admission, candidate=candidate,
                    server=native.server, source=native.source, output=private / "launch-evidence",
                    policy_observer=lambda bound: bound.roots, admission_timeout_seconds=90,
                    advisory_closure=advisory_closure) as scope:
                _write(output / "execution-scope.json", scope.to_dict())
                runtime = AdmittedBenchmarkRuntime.create(private / "launch", admission=admission["local_admission"],
                    server=native.server, source=native.source, implement=True, implementation_command=command,
                    candidate_runner_argv=("/opt/ipfs-supervisor/bin/validation-worker",),
                    finite_execution_scope=scope, max_task_attempts=1, lifetime_seconds=300,
                    worker_worktree_root=Path(worktree_root), timeout_ms=30_000)
                try:
                    report["start"] = runtime.start().to_dict()
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
                    if report.get("task", {}).get("status") == "completed":
                        report["stop"], report["advisory_drift_during_stop"] = _stop_with_advisory_drift(
                            runtime, support.frozen.output / "inference.json")
                        support._extend({"advisory_spawn_controls": [report["advisory_drift_during_stop"]]})
                    else:
                        report["stop"] = runtime.stop().to_dict()
                    report["remaining_processes"] = len(runtime.process.snapshot(runtime.profile).members)
                    report["bootstrap_errors"] = runtime.bootstrap_errors
                    report["native_diagnostics"] = _native_diagnostics(runtime.state)
                    _write(output / "native-lifecycle.json", report)
                    runtime.close()
                    report.update(_completion_retirement_observation(runtime))
                if (report["task"]["status"] != "completed" or report["stop"]["status"] != "succeeded"
                        or report["remaining_processes"] or report["bootstrap_errors"]):
                    raise ValueError("isolated native residual worker lifecycle is incomplete")
                report["execution_scope_after_stop"] = scope.to_dict()
            support.check_historical_pins()
            report["completed_task_rows"] = task_snapshot()
            if (set(report["completed_task_rows"]) != set(report["task_rows_before_candidate"])
                    or any(row["status"] != "completed" for row in report["completed_task_rows"].values())):
                raise ValueError("worker lost the complete original native task population")
            _write(output / "completed-task-rows.json", report["completed_task_rows"])
            _write(output / "native-task-evidence.json", _native_evidence_snapshot(native.server))
            portal_paths = sorted((private / "launch/state/run/admitted_database_portal_attempts").glob(
                "*/database-attempt-binding.json"))
            if len(portal_paths) != 1:
                raise ValueError("exactly one genuine native residual Portal binding required")
            _write(output / "native-portal-binding.json", {
                "artifact": _pin(portal_paths[0]), "record": json.loads(portal_paths[0].read_bytes())})
        report["resource_after_owner_close"] = scheduler.snapshot()
        published = _git(repository, "rev-parse", "HEAD")
        parents = _git(repository, "rev-list", "--parents", "-n", "1", published).split()
        if len(parents) != 3 or parents[1] != original_commit:
            raise ValueError("native publication is not one exact baseline two-parent merge")
        changed = _git(repository, "diff", "--name-only", original_commit, published).splitlines()
        if changed != ["calc.py"]:
            raise ValueError("published edit exceeded the original output permission")
        report.update(published_commit=published, published_commit_parents=parents[1:],
            changed_paths=changed)
        for name in ("check_type.py", "check_offset.py"):
            subprocess.run([sys.executable, "-B", name], cwd=repository, check=True,
                capture_output=True, timeout=10)
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
            subprocess.run(["git", "-C", str(repository), "worktree", "remove", str(path)],
                check=True, capture_output=True, timeout=30)
        _write(output / "owner-fixture-worktree-cleanup.json", cleanup_records)
        if (repository / ".git/worktrees").exists():
            if any((repository / ".git/worktrees").iterdir()):
                raise ValueError("unaccounted linked worktree prevents successor custody")
            (repository / ".git/worktrees").rmdir()
        old_current_rejected = False
        try:
            boundary.verify_current_finite_repository_admission(owner=owner, admission=admission,
                output=output / "stale-preview", policy_observer=lambda bound: bound.roots)
        except ValueError:
            old_current_rejected = True
        if not old_current_rejected:
            raise ValueError("published source retained stale current admission")
        # A fresh independent profile preserves old owner-local history and
        # binds the actual published Git baseline; it does not rotate old keys.
        successor_publication = index.prepare_current(repository, repository_id=first.repository_id,
            operation_id="published-successor", expected_head=first, scheduler=scheduler,
            admission_timeout_seconds=90)
        successor = successor_publication.head
        if successor.generation != first.generation + 1 or successor.snapshot_cid == first.snapshot_cid:
            raise ValueError("actual worker publication did not create a new native source generation")
        invalidations = _projection_rows(connection)
        if not any(row.get("reason") == "revision_superseded" for row in invalidations):
            raise ValueError("native catalog did not invalidate the previous source revision")
        _write(output / "native-source-publications.json", {"initial": initial_publication.to_dict(),
            "successor": successor_publication.to_dict(), "invalidations": invalidations})
        report["successor_head"] = successor.to_dict()
        new_profile, new_lifecycle = private / "successor-profile", private / "successor-lifecycle"
        Supervisor.init_local(repository=repository, consent=True, profile_dir=new_profile,
            lifecycle_dir=new_lifecycle)
        successor_owner = replace(owner, expected_head=successor)
        report["advisory_successor"] = support.prepare_successor(owner=successor_owner)
        next_request = _request(index, repository, successor, document, catalog, _authority_materials())
        next_graph, next_manifest, next_bindings = _native_graph(repository=repository, request=next_request,
            profile=new_profile, lifecycle=new_lifecycle)
        if ([task.to_dict() for task in next_graph.tasks] != [task.to_dict() for task in graph.tasks]
                or next_bindings != bindings):
            raise ValueError("successor changed original administrator task identities or contracts")
        next_declaration = boundary.author_finite_repository_declaration(owner=successor_owner,
            manifest=next_manifest, request=next_request, intent_document=document, source_text=INTENT,
            operation_catalog=catalog, tool_policy=tools, task_bindings=next_bindings)
        next_admission = boundary.admit_finite_repository_plan(owner=successor_owner,
            declaration=next_declaration, graph=next_graph, output=output / "successor-preview",
            policy_observer=lambda bound: bound.roots)
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
                boundary.materialize_finite_repository_plan(owner=successor_owner,
                    admission=next_admission, intent=empty_intent,
                    output=output / "no-work-materialization", policy_observer=lambda bound: bound.roots)
            except ValueError:
                if empty_intent.list_tasks():
                    raise ValueError("no-work review created native task rows")
            else:
                raise ValueError("no-work successor materialized an empty task grant")
        cold_root = output / "cold"
        cold_root.mkdir()
        cold_connection, cold_index = _open(cold_root)
        try:
            cold_head = cold_index.prepare_current(repository, repository_id=first.repository_id,
                operation_id="cold-published-capture", expected_head=None, scheduler=scheduler,
                admission_timeout_seconds=90).head
            _write(output / "cold-capture.json", _capture_complete(cold_index, repository, cold_head, scheduler))
            cold_owner = replace(owner, index=cold_index, expected_head=cold_head)
            cold_request = _request(cold_index, repository, cold_head, document, catalog, _authority_materials())
            cold_graph, cold_manifest, cold_bindings = _native_graph(repository=repository,
                request=cold_request, profile=new_profile, lifecycle=new_lifecycle)
            cold_decl = boundary.author_finite_repository_declaration(owner=cold_owner,
                manifest=cold_manifest, request=cold_request, intent_document=document, source_text=INTENT,
                operation_catalog=catalog, tool_policy=tools, task_bindings=cold_bindings)
            cold_admission = boundary.admit_finite_repository_plan(owner=cold_owner,
                declaration=cold_decl, graph=cold_graph, output=output / "cold-preview",
                policy_observer=lambda bound: bound.roots)
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
        fresh = subprocess.run([sys.executable, "-m", MODULE, "replay", str(output)],
            check=True, capture_output=True, text=True, timeout=60)
        _write(output / "fresh-process-replay.json", json.loads(fresh.stdout))
        if any(_pin(path) != pin for path, pin in evidence.items()):
            raise ValueError("worker/successor work altered historical evidence")
        if any(_pin(pin["path"]) != pin for pin in pins):
            raise ValueError("selected producer changed during native qualification")
        connection.close()
        report["advisory_metadata_and_cold_replay"] = support.finish(owner=successor_owner, worker_report=report)
        final_resources = scheduler.snapshot()
        if final_resources["active_lease_count"] or final_resources["waiting_request_count"]:
            raise ValueError("native resource leases or waiters remain")
        report.update(status="completed", elapsed_seconds=time.monotonic() - started,
            published_commit=published, published_commit_parents=parents[1:], changed_paths=changed,
            complete_task_population_retained=True, actual_public_checks_passed=True,
            stale_current_admission_rejected=True, historical_artifacts_unchanged=True,
            fresh_process_historical_replay=json.loads(fresh.stdout),
            successor_cold_finite_outcomes_agree=True, no_work_successor_grants_no_task_omission=True,
            execution_sources=pins, active_leases=0, waiting_requests=0,
            native_worker_successor_loop_qualified=True,
            advisory_final_spawn_closure_qualified=True,
            actual_attempted_training_epochs=support.actual_training_epochs,
            verified_proposal_consumed_by_native_worker=True,
            qualification_scope="one authored ten-file, two-task, five-input model-off native worker publication with signed advisory byte closure at cooperative final spawn, two late callback refusals, STOP despite advisory byte drift, successor and cold replay")
        _write(output / "result.json", report)
        return report
    except BaseException as exc:
        _write(output / "failure.json", {"schema": SCHEMA, "status": "failed", "error_type": type(exc).__name__,
            "error": str(exc), "elapsed_seconds": time.monotonic() - started,
            "observed_training_epoch_lower_bound": support.actual_training_epochs,
            "training_count_scope": "complete contexts and retained checkpoint requests; failure costs require native ledger inspection",
            "worker_report": report})
        raise
    finally:
        support.close()
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "replay"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--lean", type=Path, default=Path("/toolchains/lean/bin/lean"))
    parser.add_argument("--handoff-root", type=Path, default=Path("/opt/ipfs-supervisor/finite-handoffs"))
    parser.add_argument("--worktree-root", type=Path, default=Path("/opt/ipfs-supervisor/worktrees"))
    args = parser.parse_args()
    result = replay(args.output) if args.action == "replay" else run(args.output,
        python_executable=args.python, lean_executable=args.lean,
        handoff_root=args.handoff_root, worktree_root=args.worktree_root)
    printed = result if args.action == "replay" else {"schema": result["schema"], "status": result["status"]}
    print(json.dumps(printed, sort_keys=True))


if __name__ == "__main__":
    main()
