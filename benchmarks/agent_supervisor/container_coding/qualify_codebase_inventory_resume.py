"""Native qualification of resumable inventory and signed advisory context.

The fixture has 300 tracked files, exceeding the legacy 256-entry scan profile.
Setup trains an explicitly selected private model. Scan, receiving, admission
and worker-context replay then fit nothing. Numerical features remain advisory.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shlex
import threading

from . import qualify_codebase_inventory_scan as base

SCHEMA = "codebase-inventory-resume-native-qualification@1"
MODULE = "benchmarks.agent_supervisor.container_coding.qualify_codebase_inventory_resume"
REPOSITORY_ID = "qualification:inventory-resume"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def _git(repository, *arguments):
    result = subprocess.run(["git", *arguments], cwd=repository,
        env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull},
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    return result.stdout.decode().strip()


def _sources(repository):
    repository.mkdir()
    files = {
        "calc.py": "def increment(n: int) -> int:\n    return n + 1\n",
        "tune.py": "def increment(n: int) -> int:\n    return n + 101\n",
        "canary.py": "def increment(n: int) -> int:\n    return n + 103\n",
        "check_type.py": "from calc import increment\nfor n in (-2, -1, 0, 1, 2):\n    assert type(increment(n)) is int\n",
        "check_offset.py": "from calc import increment\nfor n in (-2, -1, 0, 1, 2):\n    assert increment(n) == n + 2\n",
        "README.md": "Update increment in calc.py to return n + 2 while retaining exact integer output. Keep both public checks and every declared task.\n",
        "malformed.py": "def malformed(:\n",
    }
    for number in range(291):
        files[f"bulk{number:03d}.py"] = f"def increment(n: int) -> int:\n    return n + {number + 1}\n"
    for name, text in files.items():
        (repository / name).write_bytes(text.encode())
    (repository / "non_utf8.py").write_bytes(b"\xff\xfe\x00invalid\n")
    (repository / "oversized.dat").write_bytes(b"z" * (64 * 1024 + 1))
    for arguments in (("init", "--quiet"), ("config", "user.name", "Inventory resume qualification"),
            ("config", "user.email", "inventory-resume@example.invalid"), ("add", "."),
            ("commit", "--quiet", "--no-verify", "-m", "Authored 300-member fixture")):
        _git(repository, *arguments)
    count = len([x for x in _git(repository, "ls-files", "-z").split("\0") if x])
    require(count == 300, "qualification source population must be exactly 300")


def _open(output):
    import duckdb
    from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog, CodebaseCatalogLimits
    from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
    from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
    from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
    private = output / "private"
    private.mkdir(mode=0o700, exist_ok=True)
    require(private.resolve() == private and not private.is_symlink()
            and private.stat().st_mode & 0o077 == 0, "exact owner-private source/model directory required")
    connection = duckdb.connect(str(private / "source.duckdb"), config={"threads": 1, "memory_limit": "128MB"})
    try:
        store = DuckDBASTStore(connection=connection)
        artifacts = ImmutableCAS(private / "source-artifacts")
        index = RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=artifacts,
            catalog=CodebaseCatalog(store, artifacts, limits=CodebaseCatalogLimits(max_entries=512)))
        registry = AutoencoderRegistry(private / "model.duckdb", private / "model-artifacts")
    except BaseException:
        connection.close()
        raise
    return index, registry, connection


def _close(registry, connection):
    if registry is not None:
        registry.close()
    if connection is not None:
        connection.close()


def _state(registry, version, *, include_identity=False):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_projection_features as features
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_runtime_registry as runtimes
    selected = registry.get_version(version)
    saved = runtimes._read_candidate(registry, selected)
    state = {"state_sha256": features.digest(saved["state"]),
        "completed_epochs": saved["state"]["completed_epochs"],
        "adam_steps": [item["step"] for item in saved["state"]["adam"]],
        "latent_width": saved["state"]["latent_width"],
        "feature_columns": len(saved["feature_space"]["columns"])}
    if include_identity:
        state.update(artifact=selected["artifact"], report_sha256=features.digest(saved["report"]))
    return state


def _owners(index, registry):
    return {"source_head": index.current(REPOSITORY_ID).to_dict(), "registry": base._registry(registry),
        "model_artifacts": base._files(registry.artifact_root)}


def _owner_generation(owners):
    rows = owners["registry"]["meta"]
    require(type(rows) is list and len(rows) == 1 and len(rows[0]) == 6
        and type(rows[0][5]) is int and rows[0][5] > 0, "exact native registry owner generation required")
    return rows[0][5]


def _owners_after_reopens(owners, count):
    require(type(count) is int and 0 <= count <= 5, "bounded known registry reopens required")
    expected = deepcopy(owners)
    original = expected["registry"]["meta"][0]
    row = list(original)
    row[5] = _owner_generation(owners) + count
    expected["registry"]["meta"][0] = type(original)(row)
    return expected


def _scheduler(output):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
        GlobalResourceScheduler, ResourceSchedulerConfig)
    return GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=output / "resource-admission.json", lane_reservations={}, auto_renew_leases=True))


def _pins(output, modules):
    paths = [(module, Path(importlib.import_module(module).__file__).resolve()) for module in modules]
    paths.append(("qualification_harness", Path(__file__).resolve()))
    destination = output / "producers"
    destination.mkdir()
    rows = []
    for name, path in paths:
        raw = path.read_bytes()
        copy = destination / (name + ".py")
        with copy.open("xb") as file:
            file.write(raw)
        rows.append({"name": name, "path": str(path), "copy": copy.relative_to(output).as_posix(),
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    record = {"schema": "codebase-inventory-resume-selected-producers@1", "files": rows,
        "execution_attestation": False, "scope": "listed local source bytes only"}
    base._write(output / "generation-inputs.json", record)
    return record


def _require_pins(output, pins):
    for row in pins["files"]:
        expected = row["sha256"]
        require(hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == expected,
                "selected producer changed: " + row["name"])
        require(hashlib.sha256((output / row["copy"]).read_bytes()).hexdigest() == expected,
                "retained producer changed: " + row["name"])


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
    goal = PromptGoalRecord(goal_key="INVENTORY-GOAL", parent_goal_cid="", dependency_goal_cids=(),
        title="Repair increment", objective="Retain exact integer output and return n plus two",
        rationale="Independent administrator task declarations; inventory features are advisory",
        scope_paths=scopes, acceptance=criteria)
    tasks, specs = [], []
    for kind, check, criterion in zip(("type", "offset"), checks, criteria):
        task = PromptTaskRecord(task_key="INVENTORY-" + kind.upper(), goal_cid=goal.goal_cid,
            dependency_task_cids=() if not tasks else (tasks[0].task_cid,),
            objective="Satisfy the public " + kind + " check", rationale="Keep every original administrator task",
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
    roots = {"request_cid": content_identity({"schema": "authored-inventory-work-request@1",
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
    from ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction import load_public_instruction
    require(os.getuid() == os.geteuid() == 1001, "isolated worker identity required")
    raw = sys.stdin.buffer.read(256_001)
    require(len(raw) <= 256_000, "native worker prompt exceeds bound")
    prompt = raw.decode("utf-8")
    capsule = json.loads(prompt)
    require(type(capsule) is dict and capsule.get("objective_id") == "INVENTORY-OFFSET",
            "exact authored offset task required")
    workspace = Path.cwd()
    block, receipt = load_public_instruction(artifact=Path(artifact), expected_sha256=expected_sha256,
        task_cid=task_cid, prompt=prompt, workspace=workspace)
    require("CODEBASE INVENTORY ADVISORY" in block and receipt.get("codebase_inventory") is not None,
            "completed inventory context was not included by the actual public reader")
    path = workspace / "calc.py"
    before = b"def increment(n: int) -> int:\n    return n + 1\n"
    after = b"def increment(n: int) -> int:\n    return n + 2\n"
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
    result = {"schema": "inventory-resume-authored-native-worker@1", "status": "materialized",
        "pid": os.getpid(), "uid": os.getuid(), "task_cid": task_cid,
        "path": "calc.py", "before_sha256": hashlib.sha256(before).hexdigest(),
        "after_sha256": hashlib.sha256(after).hexdigest(), "public_instruction": receipt,
        "provider_calls": 0, "training_steps": 0, "proof_authority": False,
        "completion_authority": False, "native_completion_recorded_here": False}
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


class PostSetupFitAttempt(AssertionError):
    pass


def _qualify_admission_worker(output, index, registry, root, completion, scheduler, phase, report, options):
    from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
    from ipfs_accelerate_py.agent_supervisor.runtime import codebase_inventory_evidence_admission as boundary
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.runtime.codebase_inventory_execution import reserve_inventory_execution
    from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        WorktreeLifecycleStore, WorkspaceLifecycleRecord, read_process_birth)
    from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository, MAX_BODY_BYTES
    from .native_quack_qualification import open_existing_native_owner
    from .terminal_container_supervisor import _native_diagnostics
    from .audit_codebase_inventory_resume_worker import verify_authored_worker_receipt
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
    graph, specs, roots = _native_graph(index, index.current(REPOSITORY_ID), completion)
    tasks = {task.task_key: task for task in graph.tasks}
    current = {"root": root, "completion": completion, "index": index,
               "repository": repository, "registry": registry}
    manifest = phase("sign_full_inventory_task_manifest", lambda: boundary.author_current_inventory_manifest(
        root, completion, index, repository, registry=registry, profile_dir=profile,
        lifecycle_dir=lifecycle, task_specs=specs, planning_roots=roots, **options()))
    base._write(output / "signed-manifest.json", manifest)
    admission = phase("admit_full_native_graph", lambda: boundary.admit_current_inventory_plan(
        **current, manifest=manifest, graph=graph, **options()))
    base._write(output / "admission.json", admission)
    received = phase("receive_signed_full_admission", lambda: boundary.verify_current_inventory_admission(
        **current, admission=admission, **options()))
    base._write(output / "current-admission-verification.json", received)
    verified = local.verify_local_benchmark_admission(admission)
    require(len(verified["manifest"]["sources"]) == 300 and len(verified["graph"].tasks) == 2
            and not received["current_facts"] and not received["removed_task_cids"],
            "signed context must retain all sources, tasks and pending requirements")
    with IntentRepository(private / "intent.duckdb") as intent:
        materialized = phase("transactional_native_full_task_install", lambda: boundary.materialize_current_inventory_plan(
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
            require(attempt is not None and attempt.task_cid == tasks["INVENTORY-TYPE"].task_cid,
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
        residual = native.source.get_task(tasks["INVENTORY-OFFSET"].task_cid)
        require(residual.status == "ready", "original residual task must remain ready")
        prepared = phase("prepare_public_inventory_worker_context", lambda: boundary.prepare_current_inventory_worker_context(
            **current, admission=admission, task_cid=residual.task_cid, source_path="README.md",
            expected_source_sha256=hashlib.sha256((repository / "README.md").read_bytes()).hexdigest(), **options()))
        base._write(output / "public-worker-context.json", prepared)
        public_raw = Path(prepared["artifact"]).read_bytes()
        base._write(output / "public-worker-artifact.json", json.loads(public_raw))
        (output / "public-worker-artifact.raw.json").write_bytes(public_raw)
        candidate = {"public_instruction": prepared, "task_revision": residual.revision,
            "argv": ["/opt/ipfs-supervisor/bin/owner-worker", "--model", "inventory-authored-fixture",
                "--purpose", "coding", "--public-instruction-artifact", prepared["artifact"],
                "--public-instruction-sha256", prepared["sha256"],
                "--public-instruction-task-cid", residual.task_cid, "--timeout", "90"]}
        base._write(output / "candidate.json", candidate)
        with reserve_inventory_execution(**current, admission=admission, candidate=candidate,
                server=native.server, source=native.source, output=private / "launch-evidence",
                scheduler=scheduler, timeout_seconds=options()["timeout_seconds"], memory_mb=1024, cpu_slots=4,
                execution_memory_mb=4096, child_process_slots=8,
                admission_timeout_seconds=options()["admission_timeout_seconds"]) as scope:
            base._write(output / "execution-scope-before.json", scope.to_dict())
            runtime = phase("bind_inventory_native_runtime", lambda: AdmittedBenchmarkRuntime.create(
                private / "launch", admission=admission, server=native.server, source=native.source,
                implement=True, implementation_command=scope.to_dict()["payload"]["candidate"]["implementation_command"],
                candidate_runner_argv=("/opt/ipfs-supervisor/bin/validation-worker",),
                inventory_execution_scope=scope, max_task_attempts=1, lifetime_seconds=300,
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
                        if allocation.canonical_task_cid == residual.task_cid or allocation.task_id == "INVENTORY-OFFSET":
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
    for name in ("check_type.py", "check_offset.py"):
        phase("check_published_" + name, lambda name=name: subprocess.run([sys.executable, "-B", name],
            cwd=repository, check=True, capture_output=True, timeout=10))
    report["publication"] = {"baseline_commit": original, "published_commit": published,
        "parents": parents[1:], "changed_paths": ["calc.py"], "public_checks_passed": True}
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
    report["resource_after_native_owner_close"] = base._assert_clean(scheduler)


def _fit_guard(report):
    from ipfs_datasets_py.logic.software_contracts import codebase_source_training as training
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_projection_features as features
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_runtime_registry as runtimes
    def refuse(*args, **kwargs):
        report["post_setup_fit_attempt_count"] += 1
        raise PostSetupFitAttempt("scan/admission/worker receiving attempted fitting")
    stack = ExitStack()
    stack.enter_context(base._patch(training, "train_current_codebase_features", refuse))
    stack.enter_context(base._patch(features, "train_projection_features", refuse))
    stack.enter_context(base._patch(runtimes.SourceBoundCodebaseFeatureRuntime, "train", refuse))
    return stack


def resume_in_fresh_process(output):
    """Reopen real owners for one bounded chunk of a saved root/cursor."""
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as scanner
    output = Path(output).resolve(strict=True)
    request = json.loads((output / "resume-request.json").read_bytes())
    require(type(request["max_pages"]) is int and 1 <= request["max_pages"] <= 2,
            "fresh resume chunk must contain at most two pages")
    require(type(request["run_number"]) is int and 1 <= request["run_number"] <= 4,
            "fixture resume chunk number exceeds bound")
    receipt = output / f"fresh-process-run-{request['run_number']:02d}.json"
    require(not receipt.exists(), "fresh chunk receipt already exists")
    scheduler = _scheduler(output)
    index = registry = connection = None
    report = {"schema": "inventory-resume-fresh-process-chunk@1", "qualified": False,
        "complete": False, "pid": os.getpid(), "post_setup_fit_attempt_count": 0,
        "pages_created": [], "root_cid": request["root_cid"], "version_id": request["version_id"],
        "request_cursor": request["cursor"], "run_number": request["run_number"],
        "max_pages": request["max_pages"], "timeout_seconds": request["timeout_seconds"],
        "fit_guard_scope": "three named owner-process training APIs; child source/counters and frozen state checked separately"}
    started = time.monotonic()
    try:
        index, registry, connection = _open(output)
        root = scanner.load_codebase_scan_resume_root(index.artifacts, request["root_cid"])
        cursor = scanner.CodebaseScanResumeCursor.from_dict(request["cursor"])
        before = _owners(index, registry)
        state = _state(registry, request["version_id"])
        report["registry_owner_generation_before"] = _owner_generation(before)
        deadline = started + request["timeout_seconds"]
        def options():
            remaining = deadline - time.monotonic()
            require(remaining > 0, "fresh-process resume exceeded declared deadline")
            return {"scheduler": scheduler, "timeout_seconds": min(120.0, remaining),
                "admission_timeout_seconds": min(30.0, remaining), "memory_mb": 1024}
        with _fit_guard(report):
            while cursor is not None and len(report["pages_created"]) < request["max_pages"]:
                page = scanner.scan_current_codebase_page(index, output / "repository",
                    root=root, registry=registry, cursor=cursor, **options())
                report["pages_created"].append(page.artifact_cid)
                cursor = page.next_cursor
                base._progress(output / f"fresh-process-progress-{request['run_number']:02d}.json", {**report,
                    "next_cursor": None if cursor is None else cursor.to_dict(),
                    "recorded_seconds": time.monotonic() - started})
            report["next_cursor"] = None if cursor is None else cursor.to_dict()
            report["prefix_tail_cid"] = page.artifact_cid
            if cursor is None:
                completion = scanner.complete_current_codebase_scan(index, output / "repository",
                    root=root, registry=registry, tail_page_cid=page.artifact_cid, **options())
                require(scanner.validate_current_codebase_scan_completion(completion, index, output / "repository",
                    root=root, registry=registry, **options()) is completion, "fresh receiver selected another completion")
                report.update(complete=True, completion_cid=completion.artifact_cid,
                    coverage=completion.to_dict()["coverage"])
        require(_owners(index, registry) == before, "fresh-process resume mutated model/catalog owners")
        require(_state(registry, request["version_id"]) == state, "fresh-process resume changed weights/Adam")
        report.update(qualified=True, numerical_before=state, numerical_after=state,
            source_model_owner_preservation=True,
            registry_owner_generation_after=_owner_generation(_owners(index, registry)),
            final_resources=base._assert_clean(scheduler))
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        _close(registry, connection)
        report["recorded_seconds"] = time.monotonic() - started
        base._write(receipt, report)
    return report


def _resume_chunks(output, *, root, cursor, version, owner_generation, deadline, phase):
    """Run four separate bounded owner processes; accept only a full final scan."""
    runs, pages, pids = [], [], []
    started = time.monotonic()
    for number in range(1, 5):
        require(cursor is not None, "fixture completed before all eight remaining pages")
        remaining = deadline - time.monotonic()
        require(remaining > 0, "overall qualification deadline exceeded before resume chunk")
        request = {"root_cid": root.artifact_cid, "cursor": cursor, "version_id": version,
            "timeout_seconds": min(420.0, remaining), "max_pages": 2, "run_number": number}
        base._progress(output / "resume-request.json", request)
        base._write(output / f"resume-request-{number:02d}.json", request)
        with (output / f"fresh-process-{number:02d}.stdout").open("x") as stdout, \
                (output / f"fresh-process-{number:02d}.stderr").open("x") as stderr:
            def invoke_child():
                child = subprocess.run([sys.executable, "-B", "-m", MODULE, "resume", str(output)],
                    stdout=stdout, stderr=stderr, timeout=min(450.0, deadline - time.monotonic()))
                require(child.returncode == 0,
                    "fresh-process chunk failed; retained numbered stdout/stderr identify boundary")
                return child
            phase(f"restart_owners_and_resume_chunk_{number:02d}", invoke_child)
        result = json.loads((output / f"fresh-process-run-{number:02d}.json").read_bytes())
        require(result["qualified"] is True and type(result["post_setup_fit_attempt_count"]) is int
            and result["post_setup_fit_attempt_count"] == 0 and type(result["pid"]) is int
            and result["pid"] > 1 and result["pid"] != os.getpid() and result["pid"] not in pids
            and result["root_cid"] == root.artifact_cid and result["version_id"] == version
            and result["request_cursor"] == cursor and len(result["pages_created"]) == 2
            and result["source_model_owner_preservation"] is True
            and type(result["registry_owner_generation_before"]) is int
            and result["registry_owner_generation_before"] == owner_generation + number
            and result["registry_owner_generation_after"] == result["registry_owner_generation_before"]
            and result["numerical_before"] == result["numerical_after"],
            "fresh owner chunk failed exact request/no-fitting/model-preservation checks")
        require(all(type(value) is int and value == 0 for value in result["final_resources"].values()),
                "fresh owner chunk retained reservations")
        if number < 4:
            require(result["complete"] is False and result["next_cursor"] is not None
                and result["next_cursor"]["next_offset"] > cursor["next_offset"],
                "incomplete resume chunk failed strict progress")
        else:
            require(result["complete"] is True and result["next_cursor"] is None,
                    "final resume chunk did not return a complete scan")
        runs.append(result); pages.extend(result["pages_created"]); pids.append(result["pid"])
        cursor = result["next_cursor"]
    last = runs[-1]
    aggregate = {"schema": "inventory-resume-fresh-process@2", "qualified": True, "complete": True,
        "pid": last["pid"], "pids": pids, "process_runs": runs, "pages_created": pages,
        "root_cid": root.artifact_cid, "version_id": version, "completion_cid": last["completion_cid"],
        "coverage": last["coverage"], "post_setup_fit_attempt_count": 0,
        "numerical_before": runs[0]["numerical_before"], "numerical_after": last["numerical_after"],
        "source_model_owner_preservation": True, "final_resources": last["final_resources"],
        "registry_owner_generation_before": runs[0]["registry_owner_generation_before"],
        "registry_owner_generation_after": last["registry_owner_generation_after"],
        "recorded_seconds": time.monotonic() - started,
        "fit_guard_scope": last["fit_guard_scope"]}
    require(aggregate["numerical_before"] == aggregate["numerical_after"],
            "separate owner chunks changed frozen model")
    base._write(output / "fresh-process-resume.json", aggregate)
    return aggregate


def run(output, *, overall_seconds=850.0, setup_seed=None, setup_seed_receipt=None, scan_seed=None, scan_seed_receipt=None):
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as scanner
    from ipfs_datasets_py.logic.software_contracts import codebase_source_training as training
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import CodebaseScanLimits
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import collect_proof_host_resources
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
        LeaseCancelledError, LeaseTimeoutError, ResourceSchedulerError)
    require(type(overall_seconds) in (int, float) and 0 < overall_seconds <= 3550,
            "qualification deadline must fit container cleanup reserve")
    output = Path(output).absolute()
    require(not output.exists() and output.parent.resolve(strict=True) == output.parent,
            "fresh exact qualification output required")
    require((setup_seed is None) == (setup_seed_receipt is None), "setup seed requires exact receipt")
    require((scan_seed is None) == (scan_seed_receipt is None)
            and (scan_seed is None or setup_seed is not None),
            "completed scan seed requires its receipt and the verified source/model setup")
    output.mkdir(mode=0o755)
    started = time.monotonic()
    deadline = started + overall_seconds
    report = {"schema": SCHEMA, "qualified": False, "scope": "300_file_8d_cpu_resumable_scan_signed_model_off_native_worker",
        "setup_training_attempts": [], "phases": [], "controls": [], "post_setup_fit_attempt_count": 0,
        "proof_authority": False, "source_execution_attested": False, "scan_execution_attested": False,
        "production_default_activated": False, "cuda_qualified": False, "384d_qualified": False,
        "provider_calls": 0, "overall_deadline_seconds": overall_seconds}
    report["fit_guard_scope"] = "three named owner-process training APIs; child source/counters and frozen state checked separately"
    index = registry = connection = scheduler = None
    def progress():
        report["elapsed_seconds_so_far"] = time.monotonic() - started
        base._progress(output / "progress.json", report)
    def phase(name, operation):
        row = {"name": name, "status": "running"}
        report["phases"].append(row)
        progress()
        began = time.monotonic()
        print("Inventory resume qualification: " + name, file=sys.stderr, flush=True)
        try:
            value = operation()
            row["status"] = "completed"
            return value
        except BaseException as error:
            row.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            row["elapsed_seconds"] = time.monotonic() - began
            progress()
    def options():
        remaining = deadline - time.monotonic()
        require(remaining > 0, "overall qualification deadline exceeded")
        return {"scheduler": scheduler, "timeout_seconds": min(120.0, remaining),
            "admission_timeout_seconds": min(30.0, remaining), "memory_mb": 1024}
    def refuse(name, operation, expected, *, cancellation=False):
        try:
            operation()
        except BaseException as error:
            chain, seen = [], set()
            item = error
            while item is not None and id(item) not in seen:
                seen.add(id(item)); chain.append(item); item = item.__cause__ or item.__context__
            require(not any(isinstance(e, (ResourceSchedulerError, LeaseTimeoutError, PostSetupFitAttempt))
                and not (cancellation and isinstance(e, LeaseCancelledError)) for e in chain),
                "resource/fit failure cannot qualify integrity refusal")
            require(not any("numerical child failed" in str(e) or "worker failed" in str(e) for e in chain),
                    "failed native worker cannot qualify integrity refusal")
            if cancellation:
                require(any(type(e) is LeaseCancelledError for e in chain), "actual cancellation type required")
            else:
                require(any(word in " ".join(str(e) for e in chain).lower() for word in expected),
                        "integrity refusal did not identify intended boundary")
            value = {"name": name, "refused": True, "error_type": type(error).__name__,
                "error": str(error), "accepted_resource_or_fit_failure": False,
                "resources_after": base._assert_clean(scheduler)}
            report["controls"].append(value)
            return value
        raise AssertionError("integrity control accepted: " + name)
    try:
        modules = tuple(dict.fromkeys((*base.PRODUCER_MODULES,
            "ipfs_datasets_py.logic.software_contracts.codebase_inventory_resume",
            "ipfs_datasets_py.logic.software_contracts.codebase_inventory_projection_replay",
            "ipfs_datasets_py.logic.software_contracts.codebase_inventory_lineage",
            "ipfs_datasets_py.logic.software_contracts.codebase_inventory_receiving",
            "ipfs_datasets_py.optimizers.logic_theorem_optimizer.codebase_inventory_resume_worker",
            "ipfs_accelerate_py.agent_supervisor.runtime.codebase_inventory_evidence_admission",
            "ipfs_accelerate_py.agent_supervisor.runtime.codebase_inventory_evidence_worker_context",
            "ipfs_accelerate_py.agent_supervisor.runtime.codebase_inventory_execution",
            "ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission",
            "ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction",
            "ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge",
            "benchmarks.agent_supervisor.container_coding.audit_codebase_inventory_resume_worker",
            "benchmarks.agent_supervisor.container_coding.inventory_resume_setup_fixture",
            "benchmarks.agent_supervisor.container_coding.inventory_resume_scan_fixture",
            "ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime")))
        seed = None
        if setup_seed is not None:
            from .inventory_resume_setup_fixture import materialize_staged_setup
            seed = json.loads(Path(setup_seed_receipt).read_bytes())
            began = time.monotonic()
            materialization = materialize_staged_setup(Path(setup_seed), output, seed)
            report["phases"].append({"name": "materialize_verified_closed_setup", "status": "completed",
                "elapsed_seconds": time.monotonic() - began})
            base._write(output / "source-setup-seed.json", seed)
            base._write(output / "setup-seed-materialization.json", materialization)
            report["setup_reuse"] = materialization
            report["inherited_actual_setup_epochs"] = seed["inherited_actual_setup_epochs"]
        else:
            report["inherited_actual_setup_epochs"] = 0
        pins = _pins(output, modules)
        if seed is not None:
            for name, expected in seed["producer_pins"].items():
                matches = [row for row in pins["files"] if row["name"] == name]
                require(len(matches) == 1 and all(matches[0][key] == expected[key]
                    for key in ("bytes", "sha256")), "reused checkpoint producer differs in deployed generation")
        scheduler = _scheduler(output)
        report["genuine_host_resources_before"] = asdict(collect_proof_host_resources())
        report["scheduler_configuration"] = scheduler.config.persisted_dict()
        repository = output / "repository"
        if seed is None:
            phase("create_300_tracked_sources", lambda: _sources(output / "repository"))
            repository = output / "repository"
            index, registry, connection = _open(output)
            head = phase("publish_large_native_source_head", lambda: index.prepare_current(repository,
                repository_id=REPOSITORY_ID, operation_id="initial", expected_head=None,
                limits=CodebaseScanLimits(512, 64 * 1024), exclusions=(".runtime",), **options()).head)
            report["head"] = head.to_dict()
            require(len(index.load(head.manifest_cid).snapshot.entries) == 300, "native capture omitted source entries")
            selections = [training.CodebaseTrainingSelection(path, role) for path, role in
                (("calc.py", "train"), ("tune.py", "tune"), ("canary.py", "canary"))]
            def fit(name, parent=None):
                attempt = {"name": name, "requested_epochs": 1, "actual_completed_epochs": None,
                    "unknown_actual_epochs_on_failure": True}
                report["setup_training_attempts"].append(attempt)
                previous_epochs = 0 if parent is None else _state(registry, parent)["completed_epochs"]
                record = training.train_current_codebase_features(index, repository, expected_head=head,
                    registry=registry, selections=selections, operation_id=name, parent_version_id=parent,
                    epochs=1, learning_rate=.002, seed=1729, **options())
                current = _state(registry, record.to_dict()["version_id"])
                actual = current["completed_epochs"] - previous_epochs
                require(actual == 1, "native setup epoch delta differs")
                attempt.update(actual_completed_epochs=actual, unknown_actual_epochs_on_failure=False,
                    version_id=record.to_dict()["version_id"])
                base._write(output / (name + ".json"), record.to_dict())
                return record.to_dict()["version_id"]
            parent = phase("fit_private_one_epoch_root", lambda: fit("root"))
            version = phase("fit_private_same_head_one_epoch_child", lambda: fit("child", parent))
        else:
            with _fit_guard(report):
                index, registry, connection = _open(output)
                head = index.current(REPOSITORY_ID)
                require(head.to_dict() == seed["head"], "reused native source head differs")
                parent, version = seed["root_version_id"], seed["child_version_id"]
                require(_state(registry, parent, include_identity=True) == seed["checkpoint_states"]["root"]
                    and _state(registry, version, include_identity=True) == seed["checkpoint_states"]["child"],
                    "reused full native model state differs")
                require(_git(repository, "status", "--porcelain") == ""
                    and len(_git(repository, "ls-files", "-z").split("\0")) - 1 == 300,
                    "reused native checkout is not the complete clean source fixture")
                report["head"] = head.to_dict()
                report["reused_setup_state_native_checked"] = True
        report["selected_version_id"] = version
        baseline = _owners(index, registry)
        before = _state(registry, version)
        base._write(output / "owners-before-scans.json", baseline)
        original_cas = base._files(index.artifacts.root)
        base._write(output / "captured-artifacts-before-scans.json", original_cas)
        with _fit_guard(report):
            if scan_seed is not None:
                require(seed is not None and scan_seed_receipt is not None,
                        "completed scan reuse requires the verified model/source setup")
                from .inventory_resume_scan_fixture import materialize_staged_scan
                scan_receipt = json.loads(Path(scan_seed_receipt).read_bytes())
                materialization = phase("materialize_verified_closed_scan", lambda:
                    materialize_staged_scan(Path(scan_seed), index.artifacts.root, scan_receipt))
                base._write(output / "source-scan-seed.json", scan_receipt)
                base._write(output / "scan-seed-materialization.json", materialization)
                require(materialization["head"] == head.to_dict()
                        and materialization["selected_version_id"] == version,
                        "reused completed scan belongs to another source/model")
                root = scanner.load_codebase_scan_resume_root(index.artifacts, materialization["root_cid"])
                completion = scanner.load_codebase_scan_resume_completion(index.artifacts, materialization["completion_cid"])
                phase("receive_reused_completed_chain", lambda: scanner.validate_current_codebase_scan_completion(
                    completion, index, repository, root=root, registry=registry, **options()))
                require(completion.to_dict()["coverage"] == materialization["coverage"]
                        and materialization["coverage"]["inventory_entries"] == 300
                        and len(completion.to_dict()["pages"]) == 10,
                        "reused completed scan coverage differs")
                report.update(scan_reuse=materialization, source_scan_seed=scan_receipt,
                    new_scan_pages_created=0, inherited_scan_pages=10,
                    scan_coverage=completion.to_dict()["coverage"], completed_scan_cid=completion.artifact_cid,
                    scan_root_cid=root.artifact_cid, opt_out_equivalence=materialization["opt_out_equivalence"])
                reused_owners = _owners(index, registry)
                require(reused_owners == baseline and _state(registry, version) == before,
                        "receiving reused completion changed native source/model owners")
                base._write(output / "scan-root.json", root.to_dict())
                base._write(output / "scan-completion.json", completion.to_dict())
                base._write(output / "owners-after-resume.json", reused_owners)
                base._write(output / "opt-out-equivalence.json", report["opt_out_equivalence"])
                _require_pins(output, pins)
            else:
                from ipfs_datasets_py.logic.software_contracts import codebase_inventory_scan as legacy
                phase("legacy_256_profile_refuses_large_head", lambda: refuse("legacy_256_profile_refuses_large_head",
                    lambda: legacy.scan_current_codebase_features(index, repository, expected_head=head,
                        registry=registry, version_id=version, **options()), ("profile", "bound", "inventory")))
                root = phase("start_durable_large_scan", lambda: scanner.start_current_codebase_scan(index, repository,
                    expected_head=head, registry=registry, version_id=version,
                    limits=scanner.CodebaseScanResumeLimits(max_inventory_entries=512, page_entries=32), **options()))
                base._write(output / "scan-root.json", root.to_dict())
                first = phase("first_scan_page", lambda: scanner.scan_current_codebase_page(index, repository,
                    root=root, registry=registry, **options()))
                second = phase("second_scan_page", lambda: scanner.scan_current_codebase_page(index, repository,
                    root=root, registry=registry, cursor=first.next_cursor, **options()))
                require(all(page.to_dict()["worker_receipt"] is not None
                    and page.to_dict()["coverage"]["inferred_rows"] > 0 for page in (first, second)),
                    "qualification requires actual numerical inference in both pre-restart pages")
                reference_root = phase("start_opt_out_reference_root", lambda: scanner.start_current_codebase_scan(
                    index, repository, expected_head=head, registry=registry, version_id=version,
                    limits=scanner.CodebaseScanResumeLimits(max_inventory_entries=512, page_entries=32),
                    optimized=False, **options()))
                reference_page = phase("opt_out_reference_first_page", lambda: scanner.scan_current_codebase_page(
                    index, repository, root=reference_root, registry=registry, **options()))
                optimized_value, reference_value = first.to_dict(), reference_page.to_dict()
                require(all(optimized_value[key] == reference_value[key] for key in ("entries", "coverage", "inference")),
                        "default optimization differs from complete opt-out reference outcomes")
                report["opt_out_equivalence"] = {"optimized_root_cid": root.artifact_cid,
                    "reference_root_cid": reference_root.artifact_cid, "optimized_page_cid": first.artifact_cid,
                    "reference_page_cid": reference_page.artifact_cid, "entries_coverage_and_inference_exact": True,
                    "comparison_scope": "first 32 ordered source members of one frozen 8D CPU model",
                    "throughput_qualified": False}
                base._write(output / "opt-out-reference-root.json", reference_root.to_dict())
                base._write(output / "opt-out-equivalence.json", report["opt_out_equivalence"])
                base._write(output / "prefix-pages.json", [first.artifact_cid, second.artifact_cid])
                phase("incomplete_prefix_cannot_complete", lambda: refuse("incomplete_prefix_cannot_complete",
                    lambda: scanner.complete_current_codebase_scan(index, repository, root=root,
                        registry=registry, tail_page_cid=second.artifact_cid, **options()), ("incomplete", "complete")))
                wrong = deepcopy(second.next_cursor.to_dict()); wrong["next_offset"] += 1
                phase("cursor_offset_forgery", lambda: refuse("cursor_offset_forgery",
                    lambda: scanner.scan_current_codebase_page(index, repository, root=root, registry=registry,
                        cursor=scanner.CodebaseScanResumeCursor.from_dict(wrong), **options()), ("cursor", "offset", "prefix")))
                cancelled = threading.Event(); cancelled.set()
                phase("pre_cancelled_resume", lambda: refuse("pre_cancelled_resume",
                    lambda: scanner.scan_current_codebase_page(index, repository, root=root, registry=registry,
                        cursor=second.next_cursor, **{**options(), "cancel_event": cancelled}), ("cancel",), cancellation=True))
                _close(registry, connection); registry = connection = None
                resumed = _resume_chunks(output, root=root, cursor=second.next_cursor.to_dict(),
                    version=version, owner_generation=_owner_generation(baseline), deadline=deadline, phase=phase)
                report["fresh_process_resume"] = resumed
                index, registry, connection = _open(output)
                root = scanner.load_codebase_scan_resume_root(index.artifacts, root.artifact_cid)
                completion = scanner.load_codebase_scan_resume_completion(index.artifacts, resumed["completion_cid"])
                phase("receive_complete_resumed_chain", lambda: scanner.validate_current_codebase_scan_completion(
                    completion, index, repository, root=root, registry=registry, **options()))
                require(completion.to_dict()["coverage"]["inventory_entries"] == 300
                    and completion.to_dict()["coverage"]["inferred_rows"] > 0,
                    "completed numerical scan requires complete ledger and positive inference")
                report["scan_coverage"] = completion.to_dict()["coverage"]
                report["completed_scan_cid"] = completion.artifact_cid
                report["scan_root_cid"] = root.artifact_cid
                base._write(output / "scan-completion.json", completion.to_dict())
                _require_pins(output, pins)
                reopened = _owners(index, registry)
                require(reopened == _owners_after_reopens(baseline, 5) and _state(registry, version) == before,
                        "resumable scans changed frozen model/catalog state outside exact owner reopens")
                base._write(output / "owners-after-resume.json", reopened)
                report["registry_owner_reopen_transition"] = {
                    "schema": "inventory-resume-registry-owner-reopen@1", "before": _owner_generation(baseline),
                    "after": _owner_generation(reopened), "reopens": 5,
                    "child_generations": [row["registry_owner_generation_before"] for row in resumed["process_runs"]],
                    "other_owner_fields_unchanged": True}
                def receive():
                    return scanner.validate_current_codebase_scan_completion(completion, index, repository,
                        root=root, registry=registry, **options())
                original_calc = (repository / "calc.py").read_bytes()
                try:
                    (repository / "calc.py").write_bytes(original_calc + b"# source drift\n")
                    phase("current_source_drift", lambda: refuse("current_source_drift", receive,
                        ("current source", "snapshot", "repository", "stale")))
                finally:
                    (repository / "calc.py").write_bytes(original_calc)
                page_path = index.artifacts.path_for(first.artifact_cid, source=True)
                original_page = page_path.read_bytes()
                try:
                    page_path.write_bytes(original_page + b" ")
                    phase("durable_page_byte_tamper", lambda: refuse("durable_page_byte_tamper", receive,
                        ("cid", "digest", "identity", "artifact", "canonical")))
                finally:
                    page_path.write_bytes(original_page)
                native_store = index.ingestor.store
                projection = index.lookup(index.load(head.manifest_cid), "calc.py")
                with native_store._lock, native_store._transaction():
                    symbol = connection.execute("SELECT symbol_row_id,name FROM symbols WHERE blob_id=? ORDER BY symbol_row_id LIMIT 1",
                        [projection.blob_id]).fetchone()
                    require(symbol is not None, "native fixture symbol required for relational corruption")
                    connection.execute("UPDATE symbols SET name=? WHERE symbol_row_id=?", ["tampered", symbol[0]])
                try:
                    phase("native_projection_row_tamper", lambda: refuse("native_projection_row_tamper", receive,
                        ("symbol", "projection", "canonical")))
                finally:
                    with native_store._lock, native_store._transaction():
                        connection.execute("UPDATE symbols SET name=? WHERE symbol_row_id=?", [symbol[1], symbol[0]])
                phase("receive_after_exact_fault_restoration", receive)
            _qualify_admission_worker(output, index, registry, root, completion, scheduler, phase, report, options)
            phase("published_source_rejects_old_completion", lambda: refuse("published_source_rejects_old_completion",
                lambda: scanner.validate_current_codebase_scan_completion(completion, index, repository,
                    root=root, registry=registry, **options()), ("source", "repository", "snapshot", "current", "stale")))
        require(_state(registry, version) == before, "worker/signed replay changed frozen model weights")
        _require_pins(output, pins)
        for row in original_cas:
            path = index.artifacts.root / row["path"]
            require(path.stat().st_size == row["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"],
                    "captured immutable source/AST artifacts changed")
        report.update(qualified=True, numerical_before=before, numerical_after=_state(registry, version),
            original_source_artifacts_preserved=True, final_resources=base._assert_clean(scheduler),
            genuine_host_resources_after=asdict(collect_proof_host_resources()))
    except BaseException as error:
        report.update(qualified=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["known_actual_setup_epochs"] = sum(attempt["actual_completed_epochs"] or 0 for attempt in report["setup_training_attempts"])
        report["unknown_fitting_epochs"] = any(attempt["unknown_actual_epochs_on_failure"] for attempt in report["setup_training_attempts"])
        if scheduler is not None:
            resources = scheduler.snapshot()
            report["final_resources"] = {key: resources[key] for key in
                ("active_lease_count", "waiting_request_count")}
        _close(registry, connection)
        report["recorded_seconds"] = time.monotonic() - started
        base._write(output / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "resume"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--overall-seconds", type=float, default=850.0)
    parser.add_argument("--setup-seed", type=Path)
    parser.add_argument("--setup-seed-receipt", type=Path)
    parser.add_argument("--scan-seed", type=Path)
    parser.add_argument("--scan-seed-receipt", type=Path)
    args = parser.parse_args()
    if not 850.0 <= args.overall_seconds <= 3550.0:
        parser.error("overall qualification budget must be between 850 and 3550 seconds")
    try:
        result = run(args.output, overall_seconds=args.overall_seconds,
            setup_seed=args.setup_seed, setup_seed_receipt=args.setup_seed_receipt,
            scan_seed=args.scan_seed, scan_seed_receipt=args.scan_seed_receipt) if args.mode == "run" else resume_in_fresh_process(args.output)
    except BaseException as error:
        print(json.dumps({"qualified": False, "error_type": type(error).__name__, "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({"qualified": result["qualified"], "recorded_seconds": result["recorded_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
