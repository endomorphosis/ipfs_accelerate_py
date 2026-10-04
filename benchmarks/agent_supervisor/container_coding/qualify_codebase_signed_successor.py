"""Actual signed successor dispatch on a new copy of a closed complete scan.

A freshly signed full-task formatting request consumes the explicitly trained
+2 successor baseline. The isolated authored worker parenthesizes reordered operands;
public checks retain exact integer and offset semantics. No new model fitting,
scan pages or forward inference occur in this qualification. The original full
scan archive stays closed, and inherited numerical execution is provenance.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from . import qualify_codebase_inventory_resume as native
from . import qualify_codebase_inventory_scan as base
from . import source_successor_full_scan_fixture as full_fixture

SCHEMA = "codebase-signed-successor-native-qualification@1"
MODULE = "benchmarks.agent_supervisor.container_coding.qualify_codebase_signed_successor"
DEFAULT_HOST = Path("/results/container-resource-authority.json")

require = native.require
_git = native._git

def _require_unchanged_owners(observed, before):
    # Native registry rows contain finite floats. Keep exact scalar types
    # through the observation wire; structured CID wire rejects floats.
    require(full_fixture.inert.observation_same(observed, before),
            "receiving changed source/model owners")

def _container_scheduler(authority_path):
    """Attach to the single local pool bounded by the held host envelope.

    Host lease PIDs are provenance here. They never enter the isolated PID
    namespace's scheduler state or owner recovery.
    """
    raw = full_fixture.read_absolute(authority_path)
    value = full_fixture.inert.parse(raw)
    fields = {"schema", "state_path", "persisted_config", "auto_renew_leases",
        "lease_ttl_seconds", "parent_host_envelope", "namespace", "scope",
        "independent_whole_host_pool", "shares_host_pid_state", "proof_authority",
        "initial_resources"}
    require(type(value) is dict and set(value) == fields
        and value["schema"] == "successor-dispatch-container-resource-authority@1"
        and value["state_path"] == "/results/container-resource-admission.json"
        and value["auto_renew_leases"] is True
        and all(value[name] is False for name in
            ("independent_whole_host_pool", "shares_host_pid_state", "proof_authority"))
        and value["scope"] == "local PID accounting inside held host envelope and actual CPU RAM PID cgroup",
        "exact container-local resource authority required")
    require(value["namespace"] == {name: os.readlink("/proc/self/ns/" + name)
        for name in ("pid", "mnt", "net")}, "resource authority belongs to another namespace")
    config = value["persisted_config"]
    require(type(config) is dict and config.get("proof_safety_enabled") is True
        and config.get("lane_reservations") == {}, "native proof-safe local pool required")
    for name, ceiling in (("total_cpu_slots", 12), ("total_memory_mb", 8192),
            ("total_child_process_slots", 12), ("total_unified_memory_mb", 8192)):
        require(type(config.get(name)) is int and 0 < config[name] <= ceiling,
            "local pool exceeds held host/cgroup ceiling: " + name)
    require(config.get("total_gpu_memory_mb") is None
        and config["total_unified_memory_mb"] == config["total_memory_mb"],
        "CPU-only local memory pool required")
    parent = value["parent_host_envelope"]
    require(type(parent) is dict and set(parent) == {"host_scheduler_state_path",
        "host_configuration_pin", "host_reservation", "container_id", "cpu_limit",
        "memory_limit_bytes", "pids_limit"}
        and type(parent["cpu_limit"]) is int and parent["cpu_limit"] == 12
        and type(parent["memory_limit_bytes"]) is int and parent["memory_limit_bytes"] == 8 * 1024**3
        and type(parent["pids_limit"]) is int and parent["pids_limit"] == 512
        and type(parent["host_scheduler_state_path"]) is str
        and parent["host_scheduler_state_path"] != value["state_path"],
        "held host envelope and actual container limits required")
    from ipfs_accelerate_py.agent_supervisor.runtime.candidate_execution import _root_file
    boundary_path = Path("/opt/ipfs-supervisor/container-boundary.json")
    boundary_digest = _root_file(boundary_path)
    boundary_raw = full_fixture.read_absolute(boundary_path, 65_536)
    boundary = full_fixture.inert.parse(boundary_raw)
    require(hashlib.sha256(boundary_raw).hexdigest() == boundary_digest == _root_file(boundary_path)
        and boundary["schema"] == "supervisor-container-worker-boundary@1"
        and boundary["container_id"] == parent["container_id"]
        and boundary["namespaces"] == value["namespace"],
        "root-controlled worker boundary and resource container differ")
    lease = parent["host_reservation"]
    require(type(lease) is dict and lease.get("released") is False
        and lease.get("cancelled") is False and lease.get("requires_gpu") is False,
        "held CPU-only parent host lease required")
    for name, expected in (("cpu_slots", 12), ("memory_mb", 8192), ("child_process_slots", 12)):
        require(type(lease.get(name)) is int and lease[name] == expected,
            "parent host lease does not cover local container: " + name)
    ttl = value["lease_ttl_seconds"]
    require(type(ttl) in (int, float) and math.isfinite(ttl) and ttl > 0,
        "finite native resource lease TTL required")
    scheduler = full_fixture.shared_scheduler(value["state_path"], config)
    require(scheduler.config.lease_ttl_seconds == ttl
        and scheduler.config.auto_renew_leases is True, "resource renewal authority differs")
    initial = value["initial_resources"]
    require(type(initial) is dict and initial.get("state_path") == value["state_path"]
        and all(type(initial.get(name)) is int and initial[name] == 0
            for name in ("active_lease_count", "waiting_request_count")),
        "fresh local pool must have no earlier process owners")
    clean = full_fixture.assert_clean(scheduler)
    require(clean["global_active_lease_count"] == clean["global_waiting_request_count"] == 0,
        "container-local pool was already in use before this operation")
    return scheduler, value, raw

def _run_published_check(repository, name):
    """Retain bounded actual public subprocess observations after the merge."""
    require(name in {"check_type.py", "check_offset.py"}, "fixed authored public check required")
    repository = Path(repository)
    source = repository / name
    require(repository.is_absolute() and repository.resolve(strict=True) == repository
        and source.resolve(strict=True) == source and source.is_file() and not source.is_symlink(),
        "exact published public check path required")
    before = source.read_bytes()
    require(len(before) <= 131_072, "authored public check source exceeds qualification bound")
    argv = [sys.executable, "-B", name]
    checked = subprocess.run(argv, cwd=repository, check=False, capture_output=True, timeout=10)
    require(type(checked.returncode) is int and type(checked.stdout) is bytes
        and type(checked.stderr) is bytes and len(checked.stdout) + len(checked.stderr) <= 65_536,
        "published public check observation exceeds output bound")
    after = source.read_bytes()
    require(before == after, "public check source changed during execution")
    row = {"schema": "source-successor-published-public-check@1", "path": name,
        "argv": argv, "cwd": str(repository), "timeout_seconds": 10,
        "returncode": checked.returncode,
        "stdout": checked.stdout.decode("utf-8"), "stderr": checked.stderr.decode("utf-8"),
        "stdout_bytes": len(checked.stdout), "stderr_bytes": len(checked.stderr),
        "stdout_sha256": hashlib.sha256(checked.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(checked.stderr).hexdigest(),
        "source_before": full_fixture.pin(before), "source_after": full_fixture.pin(after)}
    require(checked.returncode == 0, "published public check failed")
    return row

def _finalize_qualification_report(output, report, *, registry, connection, scheduler,
                                  started, primary_error_active):
    """Retain refusal evidence even when native cleanup cannot prove closure."""
    first_cleanup_error = None

    def failed(stage, error):
        nonlocal first_cleanup_error
        if first_cleanup_error is None:
            first_cleanup_error = error
        report["qualified"] = False
        report.setdefault("cleanup_errors", []).append({"stage": stage,
            "error_type": type(error).__name__, "error": str(error)})

    try:
        native._close(registry, connection)
    except BaseException as error:
        failed("close_native_owners", error)
    report["recorded_seconds"] = time.monotonic() - started
    if scheduler is not None:
        try:
            report["final_resources"] = full_fixture.assert_clean(scheduler)
        except BaseException as error:
            report["final_resource_cleanup_verified"] = False
            failed("verify_named_resource_cleanup", error)
        else:
            report["final_resource_cleanup_verified"] = True
    output = Path(output)
    if output.is_dir() and output.resolve(strict=True) == output:
        try:
            base._write(output / "result.json", report)
        except BaseException as error:
            failed("persist_result_report", error)
    if first_cleanup_error is not None and not primary_error_active:
        raise first_cleanup_error

def _native_graph(index, head, completion):
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
        PromptAcceptanceRecord, PromptGoalGraph, PromptGoalRecord, PromptOutputRecord,
        PromptTaskRecord, PromptValidationRecord)
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    policy = content_identity(local.LOCAL_POLICY)
    scopes = ("calc.py", "check_type.py", "check_offset.py", "README.md")
    checks = tuple(PromptValidationRecord(validation_key="public-" + kind,
        argv=("python3", "-B", "check_" + kind + ".py"), policy_cid=policy) for kind in ("type", "offset"))
    criteria = tuple(PromptAcceptanceRecord(criterion_key="inventory-" + kind,
        criterion="The public " + kind + " check passes", validation_keys=("public-" + kind,))
        for kind in ("type", "offset"))
    goal = PromptGoalRecord(goal_key="SUCCESSOR-GOAL", parent_goal_cid="", dependency_goal_cids=(),
        title="Format increment", objective="Retain exact integer output and offset two while formatting the return expression",
        rationale="Independent administrator task declarations; inventory features are advisory",
        scope_paths=scopes, acceptance=criteria)
    tasks, specs = [], []
    for kind, check, criterion in zip(("type", "offset"), checks, criteria):
        task = PromptTaskRecord(task_key="SUCCESSOR-TYPE" if kind == "type" else "SUCCESSOR-FORMAT", goal_cid=goal.goal_cid,
            dependency_task_cids=() if not tasks else (tasks[0].task_cid,),
            objective="Preserve the public type check" if kind == "type" else "Parenthesize the return expression while preserving the public offset check", rationale="Keep every original administrator task",
            scope_paths=scopes, outputs=(PromptOutputRecord(path="calc.py", effect="modify", media_type="text/x-python"),),
            validations=(check,), acceptance=(criterion,), evidence_cids=(),
            policy_roots=(policy,), predicted_files=("calc.py",))
        tasks.append(task)
        specs.append({"task_key": task.task_key, "scope_paths": list(scopes),
            "dependencies": [] if kind == "type" else [tasks[0].task_key],
            "outputs": [{name: getattr(task.outputs[0], name) for name in ("path", "effect", "media_type")}],
            "validations": [{name: local._plain(getattr(check, name)) for name in
                ("validation_key", "argv", "cwd", "expected_exit_codes", "policy_cid")}],
            "acceptance": [{name: local._plain(getattr(criterion, name)) for name in
                ("criterion_key", "criterion", "evidence_cids", "validation_keys")}]})
    roots = {"request_cid": content_identity({"schema": "authored-successor-format-work-request@1",
        "objective": goal.objective, "task_keys": [task.task_key for task in tasks]}),
        "program_root": index.load(head.manifest_cid).semantic_state.state_cid,
        "scan_cid": content_identity({"head": head.to_dict(), "completed_scan": completion.artifact_cid})}
    graph = PromptGoalGraph(**roots, policy_roots=(policy,), goals=(goal,), tasks=tuple(tasks), evidence=())
    return graph, specs, roots

def run_authored_worker(*, artifact, expected_sha256, task_cid):
    """One explicit offline fixture edit, after the real public context reader.

    This is an authored model-off worker. It is not an LLM provider, learned
    patch proposal, source proof or substitute for native public validation.
    The Docker deployment installs this branch only for this qualification.
    """
    from ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction import _unique, load_public_instruction
    require(os.getuid() == os.geteuid() == 1001, "isolated worker identity required")
    raw = sys.stdin.buffer.read(256_001)
    require(len(raw) <= 256_000, "native worker prompt exceeds bound")
    prompt = raw.decode("utf-8")
    capsule, _ = json.JSONDecoder(object_pairs_hook=_unique).raw_decode(prompt.lstrip())
    require(type(capsule) is dict and capsule.get("objective_id") == "SUCCESSOR-FORMAT",
            "exact authored offset task required")
    workspace = Path.cwd()
    block, receipt = load_public_instruction(artifact=Path(artifact), expected_sha256=expected_sha256,
        task_cid=task_cid, prompt=prompt, workspace=workspace)
    require("CODEBASE INVENTORY ADVISORY" in block and receipt.get("codebase_successor") is not None and receipt.get("schema") == "supervisor-public-instruction-inclusion@4",
            "completed inventory context was not included by the actual public reader")
    path = workspace / "calc.py"
    before = b"def increment(n: int) -> int:\n    return n + 2\n"
    after = b"def increment(n: int) -> int:\n    return (2 + n)\n"
    require(path.resolve(strict=True) == path and not path.is_symlink()
            and path.is_file() and path.read_bytes() == before, "authored patch preimage differs")
    # A second real public replay closes the complete current source/contract
    # observations immediately before this fixed fixture's authorized write.
    _, closing = load_public_instruction(artifact=Path(artifact), expected_sha256=expected_sha256,
        task_cid=task_cid, prompt=prompt, workspace=workspace)
    require(closing == receipt, "public worker context changed before the authored edit")
    descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "r+b") as stream:
        require(stream.read() == before, "authored patch descriptor preimage differs")
        stream.seek(0)
        stream.write(after)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())
    require(path.read_bytes() == after, "authored fixture edit did not persist")
    result = {"schema": "source-successor-authored-native-worker@1", "status": "materialized",
        "pid": os.getpid(), "uid": os.getuid(), "task_cid": task_cid,
        "path": "calc.py", "before_sha256": hashlib.sha256(before).hexdigest(),
        "after_sha256": hashlib.sha256(after).hexdigest(), "public_instruction": receipt,
        "provider_calls": 0, "training_steps": 0, "proof_authority": False,
        "completion_authority": False, "native_completion_recorded_here": False}
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0

def _qualify_admission_worker(output, index, registry, selection, root, completion, scheduler, phase, report, options):
    from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
    from ipfs_accelerate_py.agent_supervisor.runtime import codebase_successor_dispatch_admission as boundary
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.runtime.codebase_successor_dispatch_admission import reserve_current_successor_execution
    from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        WorktreeLifecycleStore, WorkspaceLifecycleRecord, read_process_birth)
    from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository, MAX_BODY_BYTES
    from .native_quack_qualification import open_existing_native_owner
    from .terminal_container_supervisor import _native_diagnostics
    from .audit_codebase_signed_successor import verify_authored_worker_receipt
    from ipfs_accelerate_py.agent_supervisor.runtime.candidate_execution import _root_file

    repository, private = output / "repository", output / "private"
    boundary_path = Path("/opt/ipfs-supervisor/container-boundary.json")
    boundary_sha = _root_file(boundary_path)
    with boundary_path.open("rb") as stream:
        boundary_raw = stream.read(65_537)
    require(len(boundary_raw) <= 65_536 and hashlib.sha256(boundary_raw).hexdigest() == boundary_sha
        and _root_file(boundary_path) == boundary_sha, "exact root-controlled worker boundary bytes required")
    boundary_copy = output / "container-boundary.json"
    boundary_copy.write_bytes(boundary_raw)
    boundary_copy.chmod(0o444)
    original = _git(repository, "rev-parse", "HEAD")
    profile, lifecycle = private / "profile", private / "lifecycle"
    phase("initialize_private_signed_owner", lambda: Supervisor.init_local(repository=repository,
        consent=True, profile_dir=profile, lifecycle_dir=lifecycle))
    graph, specs, roots = _native_graph(index, index.current(root.to_dict()["head"]["repository_id"]), completion)
    tasks = {task.task_key: task for task in graph.tasks}
    current = {"selection": selection, "root": root, "completion": completion, "index": index,
               "repository": repository, "registry": registry}
    manifest = phase("sign_fresh_full_successor_task_manifest", lambda: boundary.author_current_successor_manifest(
        selection, root, completion, index, repository, registry=registry, profile_dir=profile,
        lifecycle_dir=lifecycle, task_specs=specs, planning_roots=roots, **options()))
    base._write(output / "signed-manifest.json", manifest)
    admission = phase("admit_full_native_graph", lambda: boundary.admit_current_successor_plan(
        **current, manifest=manifest, graph=graph, **options()))
    base._write(output / "admission.json", admission)
    received = phase("receive_signed_full_admission", lambda: boundary.verify_current_successor_admission(
        **current, admission=admission, **options()))
    base._write(output / "current-admission-verification.json", received)
    verified = local.verify_local_benchmark_admission(admission)
    require(len(verified["manifest"]["sources"]) == 300 and len(verified["graph"].tasks) == 2
            and not received["current_facts"] and not received["removed_task_cids"],
            "signed context must retain all sources, tasks and pending requirements")
    with IntentRepository(private / "intent.duckdb") as intent:
        materialized = phase("transactional_native_full_task_install", lambda: boundary.materialize_current_successor_plan(
            **current, admission=admission, intent=intent, **options()))
        rows = intent.list_tasks()
        require(sorted(row["task_cid"] for row in rows) == sorted(task.task_cid for task in graph.tasks),
                "native install lost original administrator tasks")
        body_bytes = {row["task_alias"]: len(json.dumps(row["body"], sort_keys=True,
            separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()) for row in rows}
        require(all(size < MAX_BODY_BYTES for size in body_bytes.values()),
                "full native task body exceeds unchanged database byte bound")
        report["native_task_body_bytes"] = body_bytes
    base._write(output / "materialized.json", materialized)
    (private / "intent.duckdb").chmod(0o600)
    allocations = {}
    worktree_root = Path("/opt/ipfs-supervisor/worktrees")
    with open_existing_native_owner(database=private / "intent.duckdb", checkout=repository,
            state_dir=private / "owner", repository_id=manifest["payload"]["repository_cid"],
            execution_routes={task.task_key: GROK_CODEX_EXECUTION_MODE for task in graph.tasks}) as native:
        driver = DatabaseImplementationDaemon(database_path=native.database,
            coordination_path=private / "prerequisite-coordination.duckdb",
            execution_path=private / "prerequisite-execution.duckdb", authority_mode="quack",
            task_source_kind="duckdb", owner_session_id="session:inventory-resume-prerequisite",
            process_instance_id=native.identity.process_birth_id, quack_uri=native.identity.listen_uri,
            task_source=native.source, close_task_source=False,
            state_owner_bootstrap_credentials=native.credentials, strict_task_sharding=True,
            max_task_attempts=1, lease_ms=120_000, require_real_execution=True).open()
        try:
            attempt = phase("claim_native_prerequisite", driver.claim_next)
            require(attempt is not None and attempt.task_cid == tasks["SUCCESSOR-TYPE"].task_cid,
                    "native prerequisite claim selected another task")
            prerequisite = native.source.get_task(attempt.task_cid)
            checked = phase("execute_native_prerequisite_public_check", lambda: run_owner_local_task_validations(
                server=native.server, task_cid=attempt.task_cid, attempt_id=attempt.attempt_id,
                expected_revision=prerequisite.revision))
            require(checked["passed"] is True, "actual public prerequisite failed")
            claimed = dict(prerequisite.body["completion_receipt"])
            digest = checked["results"][0]["evidence_digest"]
            native.source.compare_and_set_status(prerequisite.task_cid, prerequisite.revision, "completed",
                receipt={"operation": "database_complete", "evidence_digest": digest,
                    **{key: claimed[key] for key in ("attempt_id", "claim_id", "lease_id",
                        "owner_session_id", "fencing_token", "fence_epoch")}},
                expected_control_receipt=claimed, evidence_digests=[digest])
            require(native.source.get_task(attempt.task_cid).status == "completed",
                    "native prerequisite completion did not commit")
            base._write(output / "prerequisite-native-claim.json", claimed)
            base._write(output / "prerequisite-validation.json", checked)
        finally:
            driver.close()
        residual = native.source.get_task(tasks["SUCCESSOR-FORMAT"].task_cid)
        require(residual.status == "ready", "original residual task must remain ready")
        prepared = phase("prepare_public_successor_worker_context", lambda: boundary.prepare_current_successor_worker_context(
            **current, admission=admission, task_cid=residual.task_cid, source_path="README.md",
            expected_source_sha256=hashlib.sha256((repository / "README.md").read_bytes()).hexdigest(), **options()))
        base._write(output / "public-worker-context.json", prepared)
        public_raw = Path(prepared["artifact"]).read_bytes()
        base._write(output / "public-worker-artifact.json", json.loads(public_raw))
        (output / "public-worker-artifact.raw.json").write_bytes(public_raw)
        candidate = {"public_instruction": prepared, "task_revision": residual.revision,
            "argv": ["/opt/ipfs-supervisor/bin/owner-worker", "--model", "successor-authored-format-fixture",
                "--purpose", "coding", "--public-instruction-artifact", prepared["artifact"],
                "--public-instruction-sha256", prepared["sha256"],
                "--public-instruction-task-cid", residual.task_cid, "--timeout", "90"]}
        base._write(output / "candidate.json", candidate)
        with reserve_current_successor_execution(**current, admission=admission, candidate=candidate,
                server=native.server, source=native.source, output=private / "launch-evidence",
                scheduler=scheduler, timeout_seconds=options()["timeout_seconds"], memory_mb=1024, cpu_slots=4,
                execution_memory_mb=4096, child_process_slots=8,
                admission_timeout_seconds=options()["admission_timeout_seconds"]) as scope:
            base._write(output / "execution-scope-before.json", scope.to_dict())
            runtime = phase("bind_successor_native_runtime", lambda: AdmittedBenchmarkRuntime.create(
                private / "launch", admission=admission, server=native.server, source=native.source,
                implement=True, implementation_command=scope.to_dict()["payload"]["candidate"]["implementation_command"],
                candidate_runner_argv=("/opt/ipfs-supervisor/bin/validation-worker",),
                inventory_execution_scope=scope, max_task_attempts=1, lifetime_seconds=600,
                worker_worktree_root=worktree_root, timeout_ms=30_000))
            try:
                report["start"] = phase("native_worker_start", lambda: runtime.start().to_dict())
                require(report["start"]["status"] == "succeeded", "native START failed")
                report["worker_launched"] = True
                observations, previous = [], None
                store = WorktreeLifecycleStore(repo_root=repository)
                worker_deadline = time.monotonic() + options()["timeout_seconds"]
                while time.monotonic() < worker_deadline:
                    task = native.source.get_task(residual.task_cid)
                    for allocation in store.iter_records():
                        if allocation.canonical_task_cid == residual.task_cid or allocation.task_id == "SUCCESSOR-FORMAT":
                            allocations[allocation.record_id] = allocation.to_dict()
                    observed = (task.status, task.revision)
                    if observed != previous:
                        observations.append({"status": task.status, "revision": task.revision})
                        previous = observed
                    if task.status in {"completed", "failed", "blocked", "cancelled"}:
                        break
                    require(runtime.process.snapshot(runtime.profile).members,
                            "native supervisor exited before residual completion")
                    time.sleep(.25)
                report["task_observations"] = observations
                report["residual_task"] = {"task_cid": task.task_cid, "status": task.status,
                    "revision": task.revision, "body": dict(task.body)}
                report["observed_worker_allocations"] = list(allocations.values())
                report["resource_before_stop"] = scheduler.snapshot()
            finally:
                report["stop"] = phase("native_worker_stop_and_uid_cleanup", lambda: runtime.stop().to_dict())
                report["remaining_processes"] = len(runtime.process.snapshot(runtime.profile).members)
                report["bootstrap_errors"] = runtime.bootstrap_errors
                report["native_diagnostics"] = _native_diagnostics(runtime.state)
                base._write(output / "native-lifecycle.json", {key: report.get(key) for key in
                    ("start", "stop", "worker_launched", "residual_task", "task_observations",
                     "observed_worker_allocations", "remaining_processes", "bootstrap_errors", "native_diagnostics")})
                runtime.close()
            require(report["residual_task"]["status"] == "completed"
                and report["stop"]["status"] == "succeeded" and not report["remaining_processes"]
                and not report["bootstrap_errors"], "isolated native worker lifecycle is incomplete")
            report["authored_worker_receipt"] = verify_authored_worker_receipt(output,
                prepared=prepared, task_cid=residual.task_cid)
            base._write(output / "authored-worker-receipt-verification.json", report["authored_worker_receipt"])
            base._write(output / "execution-scope-after-stop.json", scope.to_dict())
        report["native_task_statuses"] = {task.task_key: native.source.get_task(task.task_cid).status for task in graph.tasks}
        require(set(report["native_task_statuses"].values()) == {"completed"}, "original full task population not completed")

    published = _git(repository, "rev-parse", "HEAD")
    parents = _git(repository, "rev-list", "--parents", "-n", "1", published).split()
    require(len(parents) == 3 and parents[1] == original, "native publication requires exact baseline two-parent merge")
    require(_git(repository, "diff", "--name-only", original, published).splitlines() == ["calc.py"],
            "published patch exceeded authored output permission")
    public_checks = [phase("check_published_" + name,
        lambda name=name: _run_published_check(repository, name))
        for name in ("check_type.py", "check_offset.py")]
    base._write(output / "published-public-checks.json", public_checks)
    report["publication"] = {"baseline_commit": original, "published_commit": published,
        "parents": parents[1:], "changed_paths": ["calc.py"], "public_checks_passed": True,
        "public_checks": public_checks}
    # The fixture removes only its clean, published allocation after native STOP.
    # This does not qualify automatic cleanup or general crash recovery.
    cleanup = []
    store = WorktreeLifecycleStore(repo_root=repository)
    linked = _git(repository, "worktree", "list", "--porcelain").splitlines()
    for observed in allocations.values():
        allocation = WorkspaceLifecycleRecord.from_dict(observed)
        path = Path(allocation.workspace_path)
        if not path.exists():
            continue
        require(path.resolve(strict=True) == path and path.is_relative_to(worktree_root)
            and path != worktree_root and "worktree " + str(path) in linked
            and not _git(path, "status", "--porcelain"), "allocated worktree identity/cleanliness differs")
        require(read_process_birth(allocation.owner.pid) != allocation.owner, "allocated worker owner still live")
        actual = store.load_workspace(path)
        require(actual is None or all(getattr(actual, field) == getattr(allocation, field) for field in
            ("record_id", "task_id", "canonical_task_cid", "owner", "lease_id", "workspace_path", "branch", "state_dir")),
            "allocated worktree lifecycle owner changed")
        subprocess.run(["git", "-C", str(repository), "merge-base", "--is-ancestor",
            _git(path, "rev-parse", "HEAD"), published], check=True, capture_output=True, timeout=10)
        decision = store.authorize_cleanup(workspace_path=path, branch=allocation.branch,
            expected_state_dir=allocation.state_dir)
        require(decision.allowed, "native lifecycle owner refused explicit fixture cleanup")
        subprocess.run(["git", "-C", str(repository), "worktree", "remove", str(path)],
            check=True, capture_output=True, timeout=30)
        cleanup.append({"allocation": observed, "decision": decision.to_dict(),
            "scope": "explicit owner fixture cleanup after native STOP"})
    base._write(output / "owner-fixture-worktree-cleanup.json", cleanup)
    linked_directory = repository / ".git/worktrees"
    if linked_directory.exists():
        require(not any(linked_directory.iterdir()), "unaccounted linked worktree retained")
        linked_directory.rmdir()
    report["resource_after_native_owner_close"] = full_fixture.assert_clean(scheduler)


def run(output, *, setup_seed, setup_seed_receipt, overall_seconds=3000.0, host_configuration=None):
    from .source_successor_dispatch_fixture import (
        materialize_staged_successor_dispatch, open_materialized_successor_dispatch,
    )
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as scan
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor as delta
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor_model as successor
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor_receiving as paired
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import CodebaseScanLimits
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_projection_features as features
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_runtime_registry as runtimes
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseTimeoutError, ResourceSchedulerError

    require(type(overall_seconds) in (int, float) and math.isfinite(overall_seconds) and 0 < overall_seconds <= 3550,
            "finite qualification deadline must retain container cleanup reserve")
    output = Path(output).absolute()
    require(not output.exists() and output.parent.resolve(strict=True) == output.parent,
            "fresh exact signed qualification output required")
    started, deadline = time.monotonic(), time.monotonic() + overall_seconds
    report = {"schema": SCHEMA, "qualified": False, "pid": os.getpid(), "phases": [], "controls": [],
        "scope": "fresh_signed_complete_300_member_successor_cpu8d_authored_format_worker",
        "inherited_setup_epochs": 2, "new_fitting_epochs": 0, "inherited_scan_pages": 10,
        "inherited_reference_pages": 1, "new_scan_pages": 0, "new_reference_pages": 0,
        "post_setup_fit_attempt_count": 0, "inference_attempt_count": 0, "provider_calls": 0,
        "proof_authority": False, "source_execution_attested": False, "scan_execution_attested": False,
        "cuda_qualified": False, "384d_qualified": False, "production_default_activated": False,
        "complete_scan_reexecuted_here": False, "overall_deadline_seconds": overall_seconds,
        "operation_deadlines": {"default_receiving": 120, "admission": 180, "native_start_ms": 30000,
            "public_receivers_reference_close": 30, "resource_admission": 30, "worker_lifetime": 600},
        "reference_scope": "unchanged_public_receivers_same_exact_default_completion_not_full_optout_pages"}
    index = registry = connection = scheduler = None
    staged = False
    def progress():
        report["elapsed_seconds_so_far"] = time.monotonic() - started
        if staged:
            base._progress(output / "progress.json", report)
    def remaining(cap=120):
        left = deadline - time.monotonic()
        require(left > 0, "signed successor overall deadline exceeded")
        return min(cap, left)
    def options(cap=180):
        # Full native task installation adds signed/catalog work beyond the
        # standalone scan entry. Bound that metadata work separately, while
        # retaining the measured default entry120 and native START30 gates.
        return {"scheduler": scheduler, "timeout_seconds": remaining(cap),
            "admission_timeout_seconds": min(30.0, remaining()), "memory_mb": 1024}
    def phase(name, action):
        row = {"name": name, "status": "running"}
        report["phases"].append(row)
        progress()
        begin = time.monotonic()
        try:
            value = action()
            row["status"] = "completed"
            return value
        except BaseException as error:
            row.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            row["elapsed_seconds"] = time.monotonic() - begin
            progress()
    def persist(name, record):
        base._write(output / name, {"artifact_cid": record.artifact_cid, "value": record.to_dict()})
    def forbidden_inference(*args, **kwargs):
        report["inference_attempt_count"] += 1
        raise AssertionError("signed dispatch attempted forward inference")
    try:
        receipt_raw = full_fixture.read_absolute(setup_seed_receipt, 16 * 1024 * 1024)
        seed = full_fixture.inert.parse(receipt_raw)
        materialized = phase("materialize_independently_audited_complete_successor", lambda:
            materialize_staged_successor_dispatch(Path(setup_seed), output, seed))
        staged = True
        report["materialization"] = materialized
        seed_binding = materialized["staged_receipt"]
        require(full_fixture.inert.same(seed_binding, seed), "materialized successor seed receipt differs")
        base._write(output / "source-dispatch-seed.json", seed)
        host_path = Path(host_configuration) if host_configuration is not None else DEFAULT_HOST
        scheduler, host, host_raw = _container_scheduler(host_path)
        report.update(scheduler_state_path=str(scheduler.state_path),
            scheduler_configuration=scheduler.config.persisted_dict(),
            container_resource_authority_pin=full_fixture.pin(host_raw),
            local_pool_bounded_by_host_envelope=True, shares_host_pid_state=False)
        base._write(output / "container-resource-authority.json", host)
        names = set(successor._implementation()["files"])
        names.update({MODULE, paired.__name__, native.__name__, full_fixture.__name__, base.__name__,
            "benchmarks.agent_supervisor.container_coding.source_successor_dispatch_fixture",
            "benchmarks.agent_supervisor.container_coding.audit_codebase_signed_successor",
            "ipfs_accelerate_py.agent_supervisor.runtime.codebase_successor_dispatch_context",
            "ipfs_accelerate_py.agent_supervisor.runtime.codebase_successor_dispatch_admission",
            "ipfs_accelerate_py.agent_supervisor.runtime.codebase_inventory_execution",
            "ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission",
            "ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction",
            "ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index",
            "ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge",
            "ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime"})
        pins = native._pins(output, sorted(names))
        with full_fixture.no_fit(report), ExitStack() as guard:
            for owner, name in ((scan, "_worker"), (features, "infer_projection_features"),
                    (runtimes.SourceBoundCodebaseFeatureRuntime, "infer")):
                guard.enter_context(base._patch(owner, name, forbidden_inference))
            index, registry, connection = phase("open_only_new_copied_native_owners", lambda:
                open_materialized_successor_dispatch(output))
            selection = successor.load_codebase_successor_scan(index.artifacts, seed_binding["selection_cid"])
            root = scan.load_codebase_scan_resume_root(index.artifacts, seed_binding["root_cid"])
            completion = scan.load_codebase_scan_resume_completion(index.artifacts, seed_binding["completion_cid"])
            require(selection.to_dict()["root_cid"] == root.artifact_cid
                and completion.to_dict()["coverage"]["inventory_entries"] == len(root.to_dict()["members"]) == 300
                and completion.to_dict()["coverage"]["pages"] == 10 and root.to_dict()["optimized"] is True,
                "full exact default successor completion required")
            versions = {"root": selection.to_dict()["previous_model"]["version_id"],
                        "child": selection.to_dict()["model"]["version_id"]}
            frozen = {name: native._state(registry, version, include_identity=True) for name, version in versions.items()}
            require(full_fixture.inert.same(frozen, seed_binding["checkpoint_states"]),
                    "materialized model checkpoint state differs from closed setup")
            before = full_fixture.owners(index, registry, connection)
            base._write(output / "owners-before.json", before)
            base._write(output / "checkpoint-states-before.json", frozen)
            persist("successor-selection.json", selection); persist("scan-root.json", root); persist("scan-completion.json", completion)
            transition = delta.load_codebase_source_delta(index.artifacts, selection.to_dict()["source_delta_cid"])
            persist("source-delta.json", transition)
            original_cas = base._files(index.artifacts.root)
            base._write(output / "source-artifacts-before.json", original_cas)
            report.update(root_cid=root.artifact_cid, completion_cid=completion.artifact_cid,
                selection_cid=selection.artifact_cid, source_delta_cid=transition.artifact_cid,
                scan_coverage=completion.to_dict()["coverage"], previous_head=selection.to_dict()["previous_head"],
                current_head=root.to_dict()["head"], selected_version_id=versions["child"], previous_version_id=versions["root"])
            def receive_pair():
                begin = time.monotonic()
                signal = threading.Event()
                with paired._paired_current_codebase_successor_completion(selection, completion, index, output / "repository",
                        root=root, registry=registry, cancel_event=signal, **options(120)) as close:
                    report["paired_entry_seconds"] = time.monotonic() - begin
                    closing = time.monotonic()
                    timer = threading.Timer(min(30.0, remaining(30)), signal.set)
                    timer.start()
                    try:
                        require(close() is completion, "paired receiver selected another completed successor")
                    finally:
                        timer.cancel(); timer.join()
                        report["paired_close_seconds"] = time.monotonic() - closing
                    require(report["paired_close_seconds"] <= 30,
                            "paired default closing exceeded unchanged native START30 budget")
            phase("receive_full_successor_default_pair_120_then30", receive_pair)
            def reference_closing():
                until = time.monotonic() + min(30.0, remaining(30))
                def args():
                    left = until - time.monotonic()
                    if left <= 0:
                        raise LeaseTimeoutError("public successor reference closing30 deadline exceeded")
                    return {"scheduler": scheduler, "timeout_seconds": left,
                        "admission_timeout_seconds": min(30, left), "memory_mb": 1024}
                require(successor.validate_current_codebase_successor_scan(selection, index, output / "repository",
                    registry=registry, **args()) is selection, "reference selected another successor")
                require(scan.validate_current_codebase_scan_completion(completion, index, output / "repository",
                    root=root, registry=registry, **args()) is completion, "reference selected another completion")
            reference_begin = time.monotonic()
            try:
                reference_closing()
            except (LeaseTimeoutError, ResourceSchedulerError) as error:
                report["public_receivers_reference_close"] = {"completed": False,
                    "budget_refused": True, "integrity_refusal_claimed": False,
                    "error_type": type(error).__name__, "error": str(error),
                    "elapsed_seconds": time.monotonic() - reference_begin, "deadline_seconds": 30}
            else:
                report["public_receivers_reference_close"] = {"completed": True,
                    "budget_refused": False, "integrity_refusal_claimed": False,
                    "elapsed_seconds": time.monotonic() - reference_begin, "deadline_seconds": 30}
            _require_unchanged_owners(full_fixture.owners(index, registry, connection), before)
            _qualify_admission_worker(output, index, registry, selection, root, completion, scheduler, phase, report, options)
            old_head = scan._head(root.to_dict()["head"])
            captured = index.load(old_head.manifest_cid).snapshot
            publication = phase("publish_formatted_source_generation_three", lambda: index.prepare_current(
                output / "repository", repository_id=old_head.repository_id, operation_id="signed-successor-format-publication",
                expected_head=old_head, limits=CodebaseScanLimits(captured.max_entries, captured.max_file_bytes),
                exclusions=captured.exclusions, **options()))
            require(publication.previous_head == old_head and publication.head.generation == old_head.generation + 1,
                    "format publication did not advance exact source head")
            base._write(output / "published-source-receipt.json", publication.to_dict())
            def stale():
                try:
                    scan.validate_current_codebase_scan_completion(completion, index, output / "repository",
                        root=root, registry=registry, **options())
                except (LeaseTimeoutError, ResourceSchedulerError):
                    raise AssertionError("resource failure cannot qualify stale completion refusal")
                except ValueError as error:
                    report["controls"].append({"name": "published_source_rejects_old_completion", "refused": True,
                        "error_type": type(error).__name__, "error": str(error), "integrity_refusal_claimed": True})
                    return
                raise AssertionError("published successor accepted historical completed scan")
            phase("published_source_rejects_old_completion", stale)
            after = {name: native._state(registry, version, include_identity=True) for name, version in versions.items()}
            require(full_fixture.inert.same(after, frozen), "signed dispatch changed parent or child numerical states")
            for row in original_cas:
                path = index.artifacts.root / row["path"]
                require(path.stat().st_size == row["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"],
                        "preexisting source/AST/scan immutable artifact changed")
            native._require_pins(output, pins)
            base._write(output / "checkpoint-states-after.json", after)
            base._write(output / "owners-after-publication.json", full_fixture.owners(index, registry, connection))
            report.update(qualified=True, numerical_before=frozen, numerical_after=after,
                original_source_artifacts_preserved=True, native_worker_qualified=True,
                full_administrator_population_completed=True, native_source_generation_advanced=True)
    except BaseException as error:
        report.update(qualified=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        _finalize_qualification_report(output, report, registry=registry, connection=connection,
            scheduler=scheduler, started=started, primary_error_active=sys.exc_info()[0] is not None)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--setup-seed", required=True)
    parser.add_argument("--setup-seed-receipt", required=True)
    parser.add_argument("--overall-seconds", type=float, default=3000.0)
    parser.add_argument("--host-configuration")
    options = parser.parse_args()
    run(Path(options.output), setup_seed=Path(options.setup_seed),
        setup_seed_receipt=Path(options.setup_seed_receipt), overall_seconds=options.overall_seconds,
        host_configuration=options.host_configuration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
