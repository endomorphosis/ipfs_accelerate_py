"""Retain a real local finite-context admission and successor qualification.

This exercises native source/index/checker/profile/task owners. Source edits are
explicit fixture actions; it does not launch a worker or activate a service.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

from .terminal_codebase_finite_experiment import INTENT
from .terminal_codebase_finite_service_experiment import (
    _authority_materials, _catalog, _open, _request, _scheduler,
)

SCHEMA = "finite-repository-admission-qualification@1"
_MODULE = "benchmarks.agent_supervisor.container_coding.finite_repository_admission_experiment"
_SOURCES = (_MODULE, "ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_admission",
    "ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission",
    "ipfs_accelerate_py.agent_supervisor.planning.finite_integer_capacity_preview",
    "ipfs_accelerate_py.agent_supervisor.planning.finite_integer_source_custody",
    "ipfs_accelerate_py.agent_supervisor.control.profile_authority",
    "ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository",
    "ipfs_datasets_py.logic.software_contracts.codebase_ir",
    "ipfs_datasets_py.logic.software_contracts.codebase_finite_integer_observation",
    "ipfs_datasets_py.logic.software_contracts.codebase_integer_model_lean")


def _wire(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(_wire(value))


def _pin(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _git(repository, *args):
    return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()


def _sources(repository):
    files = {
        "calc.py": "def increment(n: int) -> int:\n    return n + 1\n",
        "decoy.py": "def increment(n: int) -> int:\n    return n + 99\n",
        "support.py": "DEFAULT_OFFSET = 1\n",
        "consumer.py": "from calc import increment\n\ndef apply(n):\n    return increment(n)\n",
        "unsupported.py": "def dynamic(value):\n    return getattr(value, 'unknown', None)\n",
        "check_type.py": (
            "from calc import increment\n"
            "for n in (-2, -1, 0, 1, 2):\n    assert type(increment(n)) is int\n"
        ),
        "check_offset.py": (
            "from calc import increment\n"
            "for n in (-2, -1, 0, 1, 2):\n    assert increment(n) == n + 2\n"
        ),
    }
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Finite Admission Qualification")
    _git(repository, "config", "user.email", "fixture@example.invalid")
    for name, content in files.items():
        (repository / name).write_bytes(content.encode())
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "authored multiunit repository")
    return files


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
            argv=(sys.executable, "check_" + kind + ".py"), policy_cid=policy))
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
    admission = json.loads((output / "before-admission.json").read_bytes())
    verified = boundary.verify_finite_repository_admission(admission=admission)
    materialized = json.loads((output / "materialized.json").read_bytes())
    with IntentRepository(output / "intent.duckdb") as intent:
        rows = [intent.get_task(cid) for cid in materialized["task_cids"]]
        if any(row is None for row in rows):
            raise ValueError("native historical replay lost an administrator task")
        statuses = {row["task_alias"]: row["status"] for row in rows}
        if statuses != {"FINITE-TYPE": "completed", "FINITE-OFFSET": "in_progress"}:
            raise ValueError("native replay changed the independently observed task lifecycle")
        plan = intent.get_plan(materialized["plan_id"])
        retained = json.loads(Path(plan["body"]["finite_repository_admission_ref"]["path"]).read_bytes())
        if retained != admission:
            raise ValueError("native plan reference does not retain the complete signed finite context")
        boundary.verify_finite_repository_admission(admission=retained)
    return {"schema": "finite-admission-fresh-process-replay@1", "verified": bool(verified),
        "tasks_present": all(row is not None for row in rows),
        "task_statuses": statuses,
        "training_steps": 0, "worker_launched": False, "current_freshness_claimed": False}


def run(output, *, python_executable, lean_executable):
    from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
    from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_codebase import build_finite_integer_intent
    from ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview import RepositoryPlanPreviewOwner
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_admission as boundary
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository, IntentCompletionError
    from ipfs_datasets_py.logic.software_contracts.codebase_finite_integer_observation import seal_finite_integer_tools
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import (
        IntegerOffsetContract, UnsupportedIntegerProfile, compile_integer_offset,
    )
    output = Path(output).absolute()
    if output.exists() or not output.parent.is_dir() or output.parent.resolve() != output.parent:
        raise ValueError("fresh canonical output under an existing parent is required")
    output.mkdir(mode=0o700)
    started = time.monotonic()
    execution_sources = [_pin(importlib.import_module(name).__file__) for name in _SOURCES]
    source_copies = []
    source_directory = output / "selected-source-snapshot"
    source_directory.mkdir(mode=0o700)
    for name, pin in zip(_SOURCES, execution_sources):
        destination = source_directory / (name + ".py")
        with destination.open("xb") as stream:
            stream.write(Path(pin["path"]).read_bytes())
        retained = _pin(destination)
        if (retained["sha256"], retained["bytes"]) != (pin["sha256"], pin["bytes"]):
            raise ValueError("selected source changed while being retained")
        source_copies.append({"module": name, "original": pin, "retained": retained})
    repository = output / "repository"
    inventory = _sources(repository)
    source_before = {name: _pin(repository / name) for name in inventory}
    git_head = _git(repository, "rev-parse", "HEAD")
    profile, lifecycle = output / "profile", output / "lifecycle"
    Supervisor.init_local(repository=repository, consent=True, profile_dir=profile, lifecycle_dir=lifecycle)
    document, catalog = build_finite_integer_intent(INTENT), _catalog()
    tools = seal_finite_integer_tools(python_executable=Path(python_executable).resolve(strict=True),
        lean_executable=Path(lean_executable).resolve(strict=True))
    scheduler = _scheduler(output / "resource-admission.json")
    cx, index = _open(output)
    controls = []
    try:
        first = index.prepare_current(repository, repository_id="repository:finite-admission-qualification",
            operation_id="initial", expected_head=None, scheduler=scheduler).head
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
        except UnsupportedIntegerProfile as exc:
            controls.append({"control": "unsupported_unit_cannot_become_integer_fact", "rejected": True,
                "error": str(exc)})
        else:
            raise ValueError("unsupported unit was silently included in the integer model")
        owner = RepositoryPlanPreviewOwner(index=index, repository=repository, expected_head=first,
            scheduler=scheduler, timeout_seconds=90, memory_mb=1024)
        request = _request(index, repository, first, document, catalog, _authority_materials())
        graph, manifest, bindings = _native_graph(repository=repository, request=request, profile=profile,
            lifecycle=lifecycle)
        by_key = {task.task_key: task for task in graph.tasks}
        ordered_tasks = (by_key["FINITE-TYPE"], by_key["FINITE-OFFSET"])
        declaration = boundary.author_finite_repository_declaration(owner=owner, manifest=manifest,
            request=request, intent_document=document, source_text=INTENT, operation_catalog=catalog,
            tool_policy=tools, task_bindings=bindings)
        _write(output / "before-declaration.json", declaration)
        admission = boundary.admit_finite_repository_plan(owner=owner, declaration=declaration, graph=graph,
            output=output / "before-preview", policy_observer=lambda bound: bound.roots)
        selected_source = boundary.verify_finite_repository_admission(admission=admission)["semantic_context"]["source_cid"]
        if selected_source != entries["calc.py"].source_cid or selected_source == entries["decoy.py"].source_cid:
            raise ValueError("same-name decoy changed the selected source identity")
        _write(output / "before-admission.json", admission)
        with IntentRepository(output / "intent.duckdb") as intent:
            materialized = boundary.materialize_finite_repository_plan(owner=owner, admission=admission,
                intent=intent, output=output / "materialization-preview", policy_observer=lambda bound: bound.roots)
            if set(materialized["task_cids"]) != {task.task_cid for task in graph.tasks}:
                raise ValueError("native materialization omitted administrator tasks")
            _write(output / "materialized.json", materialized)
            offset = by_key["FINITE-OFFSET"].task_cid
            row = intent.get_task(offset)
            if local.CONTRACT_KEY not in row["body"]:
                raise ValueError("native task does not carry the old immutable completion contract")
            altered = deepcopy(row["body"])
            altered.pop(local.CONTRACT_KEY)
            try:
                intent.upsert_task(task_cid=offset, task_alias=row["task_alias"], goal_cid=row["goal_cid"],
                    identity=row["identity"], body=altered, expected_revision=row["revision"])
            except local.LocalPlanningError as exc:
                controls.append({"control": "cannot_strip_native_guard", "rejected": True, "error": str(exc)})
            else:
                raise ValueError("native completion guard was stripped")
            first_task = by_key["FINITE-TYPE"].task_cid
            first_row = intent.get_task(first_task)
            intent.cas_task_status(task_cid=first_task, expected_revision=first_row["revision"],
                new_status="in_progress")
            first_row = intent.get_task(first_task)
            try:
                intent.cas_task_status(task_cid=first_task, expected_revision=first_row["revision"],
                    new_status="completed", receipt=admission["receipt"]["payload"],
                    allow_completion_without_evidence=True)
            except (IntentCompletionError, ValueError) as exc:
                controls.append({"control": "finite_context_cannot_complete_task", "rejected": True, "error": str(exc)})
            else:
                raise ValueError("bounded finite context completed a native task")
            observations = {}
            for number, task in enumerate(ordered_tasks):
                current = intent.get_task(task.task_cid)
                if current["status"] == "ready":
                    intent.cas_task_status(task_cid=task.task_cid, expected_revision=current["revision"],
                        new_status="in_progress")
                observed = local.run_local_task_validations(intent=intent, task_cid=task.task_cid,
                    attempt_id="qualification-public-" + str(number))
                observations[task.task_key] = observed
                if observed["passed"] is True:
                    current = intent.get_task(task.task_cid)
                    intent.cas_task_status(task_cid=task.task_cid, expected_revision=current["revision"],
                        new_status="completed", evidence_digests=[row["evidence_digest"] for row in observed["results"]])
            if (observations["FINITE-TYPE"]["passed"] is not True
                    or observations["FINITE-OFFSET"]["passed"] is not False):
                raise ValueError("independent public checks did not retain the finite counterexample")
            _write(output / "public-checks.json", observations)
        evidence = {str(path): _pin(path) for root in (output / "cas", output / "before-preview",
            output / "materialization-preview") for path in root.rglob("*") if path.is_file()}
        replay = subprocess.run([sys.executable, "-m", _MODULE, "replay", str(output)],
            check=True, capture_output=True, text=True, timeout=60)
        _write(output / "fresh-process-replay.json", json.loads(replay.stdout))
        if any(_pin(path) != pin for path, pin in evidence.items()):
            raise ValueError("historical replay changed evidence bytes")
        (repository / "calc.py").write_text("def increment(n: int) -> int:\n    return n + 2\n")
        try:
            boundary.verify_current_finite_repository_admission(owner=owner, admission=admission,
                output=output / "stale-preview", policy_observer=lambda bound: bound.roots)
        except ValueError as exc:
            controls.append({"control": "same_head_source_drift_rejected", "rejected": True, "error": str(exc)})
        else:
            raise ValueError("changed working bytes reused old current admission")
        historical = boundary.verify_finite_repository_admission(admission=admission)
        if not historical or _git(repository, "rev-parse", "HEAD") != git_head:
            raise ValueError("historical identity or authored same-HEAD control failed")
        successor = index.prepare_current(repository, repository_id=first.repository_id,
            operation_id="successor", expected_head=first, scheduler=scheduler).head
        successor_owner = replace(owner, expected_head=successor)
        next_request = _request(index, repository, successor, document, catalog, _authority_materials())
        next_graph, next_manifest, next_bindings = _native_graph(repository=repository, request=next_request,
            profile=profile, lifecycle=lifecycle)
        next_declaration = boundary.author_finite_repository_declaration(owner=successor_owner,
            manifest=next_manifest, request=next_request, intent_document=document, source_text=INTENT,
            operation_catalog=catalog, tool_policy=tools, task_bindings=next_bindings)
        next_admission = boundary.admit_finite_repository_plan(owner=successor_owner, declaration=next_declaration,
            graph=next_graph, output=output / "successor-preview", policy_observer=lambda bound: bound.roots)
        _write(output / "successor-declaration.json", next_declaration)
        _write(output / "successor-admission.json", next_admission)
        with IntentRepository(output / "empty-successor.duckdb") as intent:
            try:
                boundary.materialize_finite_repository_plan(owner=successor_owner, admission=next_admission,
                    intent=intent, output=output / "no-work-materialization", policy_observer=lambda bound: bound.roots)
            except ValueError as exc:
                controls.append({"control": "no_work_is_review_only", "rejected": True, "error": str(exc)})
            else:
                raise ValueError("no-work preview created an execution grant")
        cold_root = output / "cold"
        cold_root.mkdir()
        cold_cx, cold_index = _open(cold_root)
        try:
            cold_head = cold_index.prepare_current(repository, repository_id=first.repository_id,
                operation_id="cold-full-capture", expected_head=None, scheduler=scheduler).head
            cold_owner = replace(owner, index=cold_index, expected_head=cold_head)
            cold_request = _request(cold_index, repository, cold_head, document, catalog, _authority_materials())
            cold_graph, cold_manifest, cold_bindings = _native_graph(repository=repository, request=cold_request,
                profile=profile, lifecycle=lifecycle)
            cold_declaration = boundary.author_finite_repository_declaration(owner=cold_owner,
                manifest=cold_manifest, request=cold_request, intent_document=document, source_text=INTENT,
                operation_catalog=catalog, tool_policy=tools, task_bindings=cold_bindings)
            cold_admission = boundary.admit_finite_repository_plan(owner=cold_owner, declaration=cold_declaration,
                graph=cold_graph, output=output / "cold-preview", policy_observer=lambda bound: bound.roots)
            _write(output / "cold-admission.json", cold_admission)
            projection_fields = ("source_cid", "query", "domain_inputs", "observations",
                "eligible_requirement_ids", "residual_requirement_ids", "finite_selected_task_ids",
                "operation_catalog_cid", "model_status")
            successor_context = boundary.verify_finite_repository_admission(admission=next_admission)["semantic_context"]
            cold_context = boundary.verify_finite_repository_admission(admission=cold_admission)["semantic_context"]
            if any(successor_context[name] != cold_context[name] for name in projection_fields):
                raise ValueError("incremental successor does not agree with independent cold observation")
            _write(output / "cold-comparison.json", {"agreement": True, "compared_fields": projection_fields,
                "incremental_head": successor.to_dict(), "cold_head": cold_head.to_dict(),
                "scope": "finite clause outcomes and source/domain identity; generation envelopes remain distinct"})
        finally:
            cold_cx.close()
        state = scheduler.snapshot()
        if state["active_lease_count"] or state["waiting_request_count"]:
            raise ValueError("shared resource owner was not released")
        final_evidence = {str(path): _pin(path) for evidence_root in (output / "cas", output / "before-preview",
            output / "materialization-preview") for path in evidence_root.rglob("*") if path.is_file()}
        # A successor may add its own CAS entries; every historical file must
        # retain its original bytes across replay, successor and cold work.
        if any(final_evidence.get(path) != pin for path, pin in evidence.items()):
            raise ValueError("successor or cold work changed historical parent evidence")
        if any(_pin(pin["path"]) != pin for pin in execution_sources):
            raise ValueError("selected producer bytes changed during qualification")
        if any(_pin(row["retained"]["path"]) != row["retained"] for row in source_copies):
            raise ValueError("retained selected source snapshot changed during qualification")
        result = {"schema": SCHEMA, "status": "completed", "elapsed_seconds": time.monotonic() - started,
            "output": str(output), "inventory_paths": sorted(inventory), "selected_symbol": "calc.py::increment",
            "retained_ast_units": ast_units,
            "same_name_decoy": "decoy.py::increment", "unsupported_integer_profile_unit": "unsupported.py::dynamic",
            "original_head": first.to_dict(), "successor_head": successor.to_dict(), "git_head_unchanged": True,
            "source_before": source_before, "source_after": {name: _pin(repository / name) for name in inventory},
            "complete_task_population_committed": True, "task_cids": materialized["task_cids"],
            "public_type_check_passed": True, "public_offset_check_passed": False,
            "type_task_completed_only_after_actual_owner_checks": True,
            "fresh_process_historical_replay": json.loads(replay.stdout), "historical_artifacts_unchanged": True,
            "incremental_cold_finite_outcomes_agree": True,
            "controls": controls, "active_leases": 0, "waiting_requests": 0,
            "training_steps": 0, "worker_launched": False, "production_activated": False,
            "task_omission_authority": False, "universal_python_semantics_proved": False,
            "native_worker_successor_loop_qualified": False,
            "execution_sources": execution_sources,
            "selected_source_copies": source_copies,
        }
        _write(output / "result.json", result)
        return result
    finally:
        cx.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "replay"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--python", default="/home/barberb/.local/bin/python")
    parser.add_argument("--lean", default="/home/barberb/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lean")
    args = parser.parse_args()
    result = (_historical_replay(args.output) if args.action == "replay" else
        run(args.output, python_executable=args.python, lean_executable=args.lean))
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
