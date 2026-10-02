"""Repository facts, residual planning, native worker publication and fresh successor checks.

The fixture has an explicit controlled-language specification and finite domain.
It measures a model-off integration, not learned interpretation or Terminal-Bench.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

INPUTS = [-2, -1, 0, 1, 2]
INSTRUCTION = ("Under python-integer-offset-finite@1, calc.py::increment(n) must return an exact int for inputs [-2,-1,0,1,2].\n"
               "Under python-integer-offset-finite@1, calc.py::increment(n) must return n + 2 for inputs [-2,-1,0,1,2].")
SOURCE = "def increment(n: int) -> int:\n    return n + 1\n"
CHECK = "from calc import increment\nvalues = [increment(n) for n in [-2,-1,0,1,2]]\nassert all(type(v) is int for v in values)\nassert values == [0,1,2,3,4], values\n"


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _git(root, *args):
    return subprocess.check_output(["/usr/bin/git", "-C", str(root), *args], stderr=subprocess.DEVNULL).decode().strip()


def _index(connection, artifacts):
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog
    from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
    from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
    from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
    store = DuckDBASTStore(connection=connection)
    cas = ImmutableCAS(artifacts)
    return RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=cas,
                                  catalog=CodebaseCatalog(store, cas))


def finite_successor_semantics(match):
    """Compare source/requirements/outcomes while retaining separate run receipts.

    Predicate IDs include the catalog head generation. A cold generation-1
    rebuild and incremental generation-2 rebuild must keep those identities
    distinct even when the exact source, complete query and observations agree.
    """
    result = {key: match[key] for key in ("status", "source_cid", "domain_cid", "domain_inputs",
        "eligible_clause_ids", "residual_clause_ids", "finite_counterexamples", "query")}
    result["source_snapshot_cid"] = match["head"]["snapshot_cid"]
    result["clause_results"] = [{key: value for key, value in row.items() if key != "predicate_id"}
                               for row in match["clause_results"]]
    run_receipts = {"artifacts", "head", "lean_certificate", "output", "python_process", "result_cid"}
    result["observation_semantics"] = {key: value for key, value in match["observation"].items()
                                       if key not in run_receipts}
    return result


def prepare_request(*, index, head, repository, intent, declared, tools, scheduler):
    """Derive roots from actual source, signed task/policy and explicit absent domains."""
    from ipfs_accelerate_py.agent_supervisor.planning import finite_integer_codebase as matcher
    from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_plan_preview import (
        FiniteIntegerOperationCatalog, ReviewedFiniteIntegerOperation,
        finite_integer_prompt_cid, finite_integer_intent_cid,
    )
    from ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview import RepositoryPlanPreviewOwner
    from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
        PlanAuthorityRoots, PlanCreateRequest, PlanRequestBudget, DirtyTreePolicy,
        TaskSourceKind, plan_revision_cid,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    document = matcher.build_finite_integer_intent(INSTRUCTION)
    catalog = FiniteIntegerOperationCatalog(tuple(ReviewedFiniteIntegerOperation(
        requirement_id=requirement, task_id=task, producer_id="producer:" + task,
        path="calc.py", function_name="increment", parameter="n", review_ref="authored:bounded-offset-operation@1")
        for requirement, task in ((matcher.TYPE_STATEMENT_ID, "task:finite:type"),
                                  (matcher.OFFSET_STATEMENT_ID, "task:finite:offset"))))
    native_manifest = index.load(head.manifest_cid)
    selection = dict(tools=tools, operations=catalog.to_dict(), model="disabled",
                     native_task_cid=declared["task_cid"])

    def observed_roots():
        admission = local.verify_local_benchmark_admission(declared["admission"], initial=True)
        projection = intent.plan_projection(task_cids=[declared["task_cid"]])
        return PlanAuthorityRoots(repository_id=head.repository_id,
            task_source_id="native-intent:" + str(intent.database_path),
            repository_root_cid=head.snapshot_cid, dirty_worktree_root=head.snapshot_cid,
            task_source_revision=plan_revision_cid(projection), policy_root=plan_revision_cid(local.LOCAL_POLICY),
            intent_ir_root=finite_integer_intent_cid(document),
            legal_ir_root=plan_revision_cid({"legal_constraints": "not_selected"}),
            security_ir_root=plan_revision_cid({"security_constraints": "not_selected"}),
            program_root=native_manifest.semantic_state.state_cid, capability_catalog_root=catalog.cid,
            provider_catalog_root=plan_revision_cid({"model_calls": "disabled"}),
            usage_policy_root=plan_revision_cid({"native_manifest_cid": admission["receipt"]["manifest_cid"]}),
            configuration_root=plan_revision_cid(selection))

    roots = observed_roots()
    request = PlanCreateRequest(prompt_source_cid=finite_integer_prompt_cid(INSTRUCTION),
        repository_id=head.repository_id, repository_root=str(repository), scope_paths=("calc.py",),
        dirty_tree_policy=DirtyTreePolicy.OBSERVE_AND_BIND, task_source_kind=TaskSourceKind.DUCKDB,
        board_namespace="finite-native-qualification", alias_prefix="FINITE", roots=roots,
        budget=PlanRequestBudget(max_tasks=2, max_goals=2, max_model_calls=0, max_latency_ms=90000),
        required_analysis_operations=(), optional_analysis_operations=(),
        required_logic_families=(), optional_logic_families=(), observe_roots=True)

    def policy_observer(bound):
        if bound != request:
            raise ValueError("native finite request changed")
        return observed_roots()

    return dict(owner=RepositoryPlanPreviewOwner(index=index, repository=repository,
        expected_head=head, scheduler=scheduler, timeout_seconds=90, memory_mb=1024),
        request=request, intent_document=document, source_text=INSTRUCTION,
        operation_catalog=catalog, tool_policy=tools, policy_observer=policy_observer)


def qualify(*, output: Path, python: Path, lean: Path):
    import duckdb
    from ipfs_datasets_py.logic.software_contracts.codebase_finite_integer_observation import seal_finite_integer_tools
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import get_global_resource_scheduler
    from ipfs_accelerate_py.agent_supervisor.planning import finite_integer_codebase as matcher
    from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_finite_handoff import prepare_finite_repository_handoff
    from ipfs_accelerate_py.agent_supervisor.runtime.supervised_task_context import prepare_supervised_task_context
    from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
    from .local_planning_qualification import prepare_local_task
    from .native_quack_qualification import open_existing_native_owner
    from .terminal_container_supervisor import _native_diagnostics
    from .terminal_codebase_finite_index import persist_finite_evidence_index, query_finite_evidence_index

    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("fresh exact qualification directory required")
    output.mkdir(parents=True)
    started = time.monotonic()
    report = dict(schema="native-repository-finite-qualification@1", qualified=False,
        provider_calls=0, provider_tokens=0, training_steps=0, benchmark_result=False,
        intent_origin="complete explicitly scoped controlled-language IntentIR",
        learned_interpretation=False, proof_cache_bypass=False, production_activation=False,
        run_owner_pid=os.getpid(),
        observations=[])
    try:
        repository = output / "repository"; repository.mkdir()
        for name, body in {"calc.py": SOURCE, "instruction.txt": INSTRUCTION,
                "decoy.py": "def increment(n: int) -> int:\n    return n + 200\n",
                "unsupported.py": "def dynamic(n):\n    return eval(str(n))\n",
                "public_check.py": CHECK}.items():
            (repository / name).write_text(body)
        for args in (("init", "-q"), ("config", "user.name", "Repository finite qualification"),
                ("config", "user.email", "qualification@example.invalid"),
                ("add", "."), ("commit", "-qm", "Independent complete finite acceptance")):
            _git(repository, *args)
        (repository / ".git/info/exclude").write_text(".runtime/\n__pycache__/\n")
        (repository / ".runtime").mkdir(mode=0o755)
        baseline = _git(repository, "rev-parse", "HEAD")
        check = ["python3", "-B", "public_check.py"]
        report["initial_public_check_exit_code"] = subprocess.run(check, cwd=repository,
            capture_output=True, timeout=10).returncode
        if not report["initial_public_check_exit_code"]:
            raise RuntimeError("independent public acceptance must initially fail")
        tools = seal_finite_integer_tools(python_executable=python, lean_executable=lean)
        scheduler = get_global_resource_scheduler()
        with duckdb.connect(str(output / "repository.duckdb"), config={"threads": 1, "memory_limit": "64MB"}) as cx:
            index = _index(cx, output / "artifacts")
            head = index.prepare_current(repository, repository_id="repository:finite-native-qualification",
                operation_id="initial", expected_head=None, scheduler=scheduler).head
            with IntentRepository(output / "intent.duckdb") as intent:
                declared = prepare_local_task(repository=repository, state=output / "policy", intent=intent,
                    scope_paths=["calc.py", "instruction.txt", "public_check.py"], output_path="calc.py",
                    validation_argv=check, objective=INSTRUCTION.replace("\n", " "))
                cid = declared["task_cid"]
                options = prepare_request(index=index, head=head, repository=repository, intent=intent,
                    declared=declared, tools=tools, scheduler=scheduler)
                candidate = prepare_finite_repository_handoff(**options, admission=declared["admission"],
                    intent=intent, task_cid=cid, state=output / "finite-repair", instruction_path="instruction.txt")
                _write(output / "candidate-result.json", candidate)
                if candidate["status"] != "candidate_ready":
                    raise RuntimeError("native repository plan did not select a checked candidate")
                historical = persist_finite_evidence_index(match=candidate["preview"]["match"], output=output / "finite-index")
                report["index_query"] = query_finite_evidence_index(index=index, repository=repository,
                    expected_head=head, expected=historical, scheduler=scheduler)
                context = prepare_supervised_task_context(repository=repository, intent=intent, task_cid=cid,
                    paths=["calc.py", "instruction.txt", "public_check.py"], required_raw_paths=["calc.py"],
                    output=repository / ".runtime/initial")
                bundle = write_task_context_bundle(repository=repository, prepared=[context],
                    output=repository / ".runtime/context-bundle.json")
                report.update(task_cid=cid, baseline_commit=baseline, initial_head=head.to_dict(),
                    captured_paths=[entry.path for entry in index.load(head.manifest_cid).snapshot.entries],
                    initial_fact_count=candidate["preview"]["current_facts_count"],
                    initial_residuals=candidate["preview"]["match"]["residual_clause_ids"],
                    selected_operations=candidate["preview"]["selected_task_ids"],
                    retained_preview_execution_debt=candidate["preview"]["execution_plan"],
                    context_bundle=bundle, handoff_sha256=candidate["handoff_sha256"])
        binding = candidate["signed_evidence"]["binding"]
        command = shlex.join([sys.executable, "-B", "-P", "-m",
            "ipfs_accelerate_py.agent_supervisor.runtime.repository_finite_runner",
            "--artifact", candidate["handoff_path"], "--sha256", candidate["handoff_sha256"],
            "--task-cid", cid, "--owner-did", binding["identity"], "--profile-id", binding["profile_id"]])
        worktrees = output / "worktrees"; worktrees.mkdir(mode=0o750)
        with open_existing_native_owner(database=output / "intent.duckdb", checkout=repository,
                state_dir=output / "owner", repository_id=declared["manifest"]["payload"]["repository_cid"],
                execution_routes={declared["task_id"]: GROK_CODEX_EXECUTION_MODE}) as owner:
            runtime = AdmittedBenchmarkRuntime.create(output / "launch", admission=declared["admission"],
                server=owner.server, source=owner.source, implement=True, implementation_command=command,
                context_bundle=bundle, refresh_context_on_completion=True, max_task_attempts=1,
                timeout_ms=30000, lifetime_seconds=180, worker_worktree_root=worktrees)
            try:
                report["start"] = runtime.start().to_dict()
                if report["start"]["status"] != "succeeded":
                    raise RuntimeError("native START failed")
                deadline, previous = time.monotonic() + 90, None
                while time.monotonic() < deadline:
                    task = owner.source.get_task(cid)
                    current = (task.status, task.revision)
                    if current != previous:
                        report["observations"].append(dict(status=task.status, revision=task.revision,
                                                         seconds=time.monotonic() - started))
                        _write(output / "progress.json", report); previous = current
                    if task.status in {"completed", "failed", "blocked", "cancelled"}:
                        break
                    if not runtime.process.snapshot(runtime.profile).members:
                        raise RuntimeError("native supervisor ended before task completion")
                    time.sleep(.25)
                report["task"] = dict(status=task.status, revision=task.revision)
            finally:
                try:
                    report["stop"] = runtime.stop().to_dict()
                    report["remaining_processes"] = len(runtime.process.snapshot(runtime.profile).members)
                    report["native_diagnostics"] = _native_diagnostics(runtime.state)
                    report["bootstrap_errors"] = runtime.bootstrap_errors
                    if report["stop"]["status"] == "succeeded" and report["remaining_processes"] == 0:
                        report["after_stop"] = runtime.observe()
                finally:
                    runtime.close()
        report["final_public_check_exit_code"] = subprocess.run(check, cwd=repository,
            capture_output=True, timeout=10).returncode
        report["published_commit"] = _git(repository, "rev-parse", "HEAD")
        report["published_source_matches_checked_candidate"] = hashlib.sha256((repository / "calc.py").read_bytes()).hexdigest() == candidate["signed_evidence"]["payload"]["edit"]["after_sha256"]
        from ipfs_accelerate_py.agent_supervisor.runtime.repository_finite_runner import materialize_finite_candidate
        stale_workspace = output / "stale-allocated"
        _git(repository, "worktree", "add", "--detach", str(stale_workspace), baseline)
        try:
            try:
                materialize_finite_candidate(artifact=Path(candidate["handoff_path"]),
                    expected_sha256=candidate["handoff_sha256"], task_cid=cid,
                    owner_did=binding["identity"], profile_id=binding["profile_id"],
                    prompt=json.dumps({"objective_id": declared["task_id"]}), workspace=stale_workspace)
            except ValueError:
                report["stale_dispatch_refused"] = True
            else:
                report["stale_dispatch_refused"] = False
        finally:
            _git(repository, "worktree", "remove", "--force", str(stale_workspace))
        # Reopen the durable owner and capture the actual published successor.
        with duckdb.connect(str(output / "repository.duckdb"), config={"threads": 1, "memory_limit": "64MB"}) as cx:
            index = _index(cx, output / "artifacts")
            prior = index.current(head.repository_id)
            successor = index.prepare_current(repository, repository_id=head.repository_id,
                operation_id="published-successor", expected_head=prior, scheduler=scheduler).head
            try:
                query_finite_evidence_index(index=index, repository=repository, expected_head=head,
                                           expected=historical, scheduler=scheduler)
            except ValueError:
                report["stale_index_refused"] = True
            else:
                report["stale_index_refused"] = False
            matched = matcher.match_finite_integer_intent(index=index, repository=repository,
                repository_id=head.repository_id, expected_head=successor,
                intent_document=matcher.build_finite_integer_intent(INSTRUCTION), source_text=INSTRUCTION,
                output=output / "successor-observation", tool_policy=tools, scheduler=scheduler)
            report["successor"] = dict(head=successor.to_dict(), fact_count=len(matched["current_facts"]),
                residuals=matched["residual_clause_ids"], observation_cid=matched["observation_cid"])
            _write(output / "successor-match.json", matched)
        with duckdb.connect(str(output / "cold.duckdb"), config={"threads": 1, "memory_limit": "64MB"}) as cx:
            cold = _index(cx, output / "cold-artifacts")
            cold_head = cold.prepare_current(repository, repository_id=head.repository_id,
                operation_id="independent-cold-successor", expected_head=None, scheduler=scheduler).head
            cold_match = matcher.match_finite_integer_intent(index=cold, repository=repository,
                repository_id=head.repository_id, expected_head=cold_head,
                intent_document=matcher.build_finite_integer_intent(INSTRUCTION), source_text=INSTRUCTION,
                output=output / "cold-observation", tool_policy=tools, scheduler=scheduler)
            report["cold_successor_agrees"] = finite_successor_semantics(cold_match) == finite_successor_semantics(matched)
            _write(output / "successor-semantic-comparison.json", {
                "incremental": finite_successor_semantics(matched),
                "cold": finite_successor_semantics(cold_match),
                "generation_bound_receipts_remain_distinct": matched["observation_cid"] != cold_match["observation_cid"]})
            _write(output / "cold-match.json", cold_match)
        report["resource_state"] = scheduler.snapshot()
        report["owned_remaining_leases"] = [row for row in scheduler.active_leases()
                                             if row["owner_pid"] == os.getpid()]
        report["qualified"] = bool(report.get("task", {}).get("status") == "completed"
            and report["stop"]["status"] == "succeeded" and report["remaining_processes"] == 0
            and not report["bootstrap_errors"] and report["final_public_check_exit_code"] == 0
            and report["published_commit"] != baseline and report["published_source_matches_checked_candidate"]
            and report["stale_index_refused"] and report["stale_dispatch_refused"]
            and report["cold_successor_agrees"] and report["successor"]["fact_count"] == 2
            and not report["successor"]["residuals"]
            and not report["owned_remaining_leases"])
    except Exception as error:
        report["error"] = dict(type=type(error).__name__, message=str(error)[:4096])
        if "scheduler" in locals():
            report["resource_state"] = scheduler.snapshot()
        report["host_pressure"] = {kind: Path("/proc/pressure", kind).read_text()
            for kind in ("cpu", "memory", "io") if Path("/proc/pressure", kind).is_file()}
    finally:
        report["seconds"] = time.monotonic() - started
        _write(output / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--lean", required=True, type=Path)
    result = qualify(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ("qualified", "seconds", "provider_calls")}))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
