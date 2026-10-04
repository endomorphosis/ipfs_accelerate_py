"""Qualify one isolated native worker publication from finite source context.

The complete administrator population stays in the owner database. Public
checks complete the prerequisite; the native worker edits the residual task.
This authored, model-off fixture has a five-input observation domain.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from .finite_repository_admission_experiment import (
    _git, _pin, _sources, _write,
)
from .terminal_codebase_finite_experiment import INTENT
from .terminal_codebase_finite_service_experiment import (
    _authority_materials, _catalog, _open, _request, _scheduler,
)

SCHEMA = "finite-repository-native-worker-qualification@1"
MODULE = "benchmarks.agent_supervisor.container_coding.finite_repository_worker_experiment"
SOURCES = (MODULE,
    "benchmarks.agent_supervisor.container_coding.finite_repository_admission_experiment",
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_admission",
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_execution",
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_candidate_runner",
    "ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime",
    "ipfs_accelerate_py.agent_supervisor.runtime.candidate_execution",
    "ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge",
    "ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission",
    "ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository")


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
    inventory = _sources(repository)
    original_commit = _git(repository, "rev-parse", "HEAD")
    profile, lifecycle = private / "profile", private / "lifecycle"
    Supervisor.init_local(repository=repository, consent=True, profile_dir=profile, lifecycle_dir=lifecycle)
    document, catalog = build_finite_integer_intent(INTENT), _catalog()
    tools = seal_finite_integer_tools(python_executable=Path(python_executable).resolve(strict=True),
        lean_executable=Path(lean_executable).resolve(strict=True))
    scheduler = _scheduler(private / "resource-admission.json")
    connection, index = _open(output)
    report = {"schema": SCHEMA, "status": "incomplete", "worker_launched": False,
        "training_steps": 0, "provider_calls": 0, "production_activated": False,
        "task_omission_authority": False, "universal_python_semantics_proved": False,
        "inventory_paths": sorted(inventory), "original_commit": original_commit}
    try:
        first = index.prepare_current(repository, repository_id="repository:finite-worker-qualification",
            operation_id="initial", expected_head=None, scheduler=scheduler,
            admission_timeout_seconds=90).head
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
            with native.server._lock:
                with IntentRepository(bound_connection=native.server._connection, install_schema=False) as intent:
                    residual = intent.get_task(residual_task.task_cid)
                    candidate = author_finite_repository_candidate(admission=admission, intent=intent,
                        task_cid=residual_task.task_cid, after_bytes=b"def increment(n: int) -> int:\n    return n + 2\n",
                        output=Path(handoff_root) / "candidate.json")
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
            with reserve_finite_repository_execution(owner=owner, admission=admission, candidate=candidate,
                    server=native.server, source=native.source, output=private / "launch-evidence",
                    policy_observer=lambda bound: bound.roots, admission_timeout_seconds=90) as scope:
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
                    report["stop"] = runtime.stop().to_dict()
                    report["remaining_processes"] = len(runtime.process.snapshot(runtime.profile).members)
                    report["bootstrap_errors"] = runtime.bootstrap_errors
                    report["native_diagnostics"] = _native_diagnostics(runtime.state)
                    _write(output / "native-lifecycle.json", report)
                    runtime.close()
                if (report["task"]["status"] != "completed" or report["stop"]["status"] != "succeeded"
                        or report["remaining_processes"] or report["bootstrap_errors"]):
                    raise ValueError("isolated native residual worker lifecycle is incomplete")
                report["execution_scope_after_stop"] = scope.to_dict()
        report["resource_after_owner_close"] = scheduler.snapshot()
        published = _git(repository, "rev-parse", "HEAD")
        parents = _git(repository, "rev-list", "--parents", "-n", "1", published).split()
        if len(parents) != 3 or parents[1] != original_commit:
            raise ValueError("native publication is not one exact baseline two-parent merge")
        changed = _git(repository, "diff", "--name-only", original_commit, published).splitlines()
        if changed != ["calc.py"]:
            raise ValueError("published edit exceeded the original output permission")
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
        successor = index.prepare_current(repository, repository_id=first.repository_id,
            operation_id="published-successor", expected_head=first, scheduler=scheduler,
            admission_timeout_seconds=90).head
        new_profile, new_lifecycle = private / "successor-profile", private / "successor-lifecycle"
        Supervisor.init_local(repository=repository, consent=True, profile_dir=new_profile,
            lifecycle_dir=new_lifecycle)
        successor_owner = replace(owner, expected_head=successor)
        next_request = _request(index, repository, successor, document, catalog, _authority_materials())
        next_graph, next_manifest, next_bindings = _native_graph(repository=repository, request=next_request,
            profile=new_profile, lifecycle=new_lifecycle)
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
            qualification_scope="one authored two-task, five-input, model-off native isolated worker publication")
        _write(output / "result.json", report)
        return report
    finally:
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
