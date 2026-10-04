"""Actual bounded behavior qualification; separate from the Bottle benchmark.

This executes only closed, independently guarded authored integer functions.
Lean checks finite recorded arithmetic, not CPython semantics. There is no
training, worker dispatch, official reward or production admission here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from .terminal_codebase_finite_index import (
    wire, digest, persist_finite_evidence_index, query_finite_evidence_index,
)

SCHEMA = "terminal-codebase-finite-observation-experiment@1"
INPUTS = [-2, -1, 0, 1, 2]
INTENT = (
    "Under python-integer-offset-finite@1, calc.py::increment(n) must return an exact int for inputs [-2,-1,0,1,2].\n"
    "Under python-integer-offset-finite@1, calc.py::increment(n) must return n + 2 for inputs [-2,-1,0,1,2]."
)


def _write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(wire(value) + b"\n")


def _pin(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _git(repository, *args):
    return subprocess.check_output(["git", "-C", str(repository), *args], text=True,
        stderr=subprocess.STDOUT, timeout=15).strip()


def _repo(path, files):
    path.mkdir(mode=0o700)
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Finite IR Qualification")
    _git(path, "config", "user.email", "finite-ir@example.invalid")
    for name, raw in files.items():
        (path / name).write_bytes(raw)
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "exact authored finite observation input")


def _source(offset):
    return f"def increment(n: int) -> int:\n    return n + {offset}\n".encode()


def _fresh_query(output, label, repository, head, expected):
    """Cold Python process opens the durable owner and complete typed index."""
    request = {"database": str(output / "codebase.duckdb"), "artifacts": str(output / "cas"),
        "repository": str(repository), "head": head.to_dict(), "expected": expected,
        "resource_state": str(output / (label + "-restart-resource-admission.json"))}
    request_path = output / (label + "-restart-request.json")
    _write(request_path, request)
    script = '''import json,sys
from pathlib import Path
import duckdb
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog,CodebaseHead
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import GlobalResourceScheduler,ResourceSchedulerConfig
from benchmarks.agent_supervisor.container_coding.terminal_codebase_finite_index import query_finite_evidence_index
r=json.loads(Path(sys.argv[1]).read_bytes())
owner=GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(state_path=Path(r['resource_state']),lane_reservations={},auto_renew_leases=False))
with duckdb.connect(r['database'],config={'threads':1,'memory_limit':'64MB'}) as cx:
 store=DuckDBASTStore(connection=cx); artifacts=ImmutableCAS(r['artifacts'])
 index=RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store),artifacts=artifacts,catalog=CodebaseCatalog(store,artifacts))
 result=query_finite_evidence_index(index=index,repository=r['repository'],expected_head=CodebaseHead.from_dict(r['head']),expected=r['expected'],scheduler=owner)
 assert owner.snapshot()['active_lease_count']==owner.snapshot()['waiting_request_count']==0
 print(json.dumps(result,sort_keys=True))
'''
    script_path = output / "fresh-query.py"
    if not script_path.exists():
        script_path.write_text(script)
    elif script_path.read_text() != script:
        raise ValueError("fresh finite query implementation changed")
    run = subprocess.run([sys.executable, str(script_path), str(request_path)],
        capture_output=True, text=True, timeout=60)
    process = {"command": [sys.executable, str(script_path), str(request_path)],
        "returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr,
        "script": _pin(script_path), "request": _pin(request_path)}
    _write(output / (label + "-restart-process.json"), process)
    if run.returncode != 0:
        raise ValueError("fresh finite evidence query failed: " + run.stderr)
    return json.loads(run.stdout)


def _capture(index, repository, head, scheduler):
    observed = index.observe_current(repository, expected_head=head, scheduler=scheduler)
    manifest = observed.manifest
    state = manifest.semantic_state.to_dict()
    ast = [index.load_ast_artifact(manifest, entry.path).to_dict()
        for entry in manifest.snapshot.entries if not entry.is_opaque
        and entry.path.endswith(".py") and index.lookup(manifest, entry.path) is not None]
    return {"heads": [head.to_dict()], "structural_manifests": [manifest.to_dict()],
        "sources": [entry.to_dict() for entry in manifest.snapshot.entries], "ast": ast,
        "symbols": state["symbols"], "kg": state["edges"], "artifacts": state["artifacts"],
        "vectors": [], "training": []}


def _native_plan(match):
    """Compile both declared goals and run the existing model-disabled planner."""
    from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
        ObligationGraphCompiler, TypedIntent, ProducerRule, TaskCandidate,
        obligation_id_for_producer, obligation_id_for_predicate,
    )
    from ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner import FrozenPlanningGoal
    from ipfs_accelerate_py.agent_supervisor.planning.plan_evaluator import EvidenceAwarePlanPolicy
    from ipfs_accelerate_py.agent_supervisor.planning.symbolic_candidate_planner import (
        SymbolicCandidatePlanner, SymbolicCandidateBounds, SymbolicCandidatePlanningError,
    )
    from ipfs_accelerate_py.agent_supervisor.planning.plan_critic import PlanCritic
    intent = TypedIntent.from_dict(match["typed_intent"])
    if not intent.current_root_id or len(intent.desired_predicates) != 2:
        raise ValueError("complete source-rooted two-clause finite intent required")
    producers, candidates = [], []
    satisfied = {fact["predicate"]["predicate_id"] for fact in match["current_facts"]}
    predicate_by_id = {predicate.predicate_id: predicate for predicate in intent.desired_predicates}
    requirement_map = dict(intent.metadata["requirement_predicate_ids"])
    if (set(requirement_map) != set(match["query"]["requirement_ids"])
            or len(requirement_map) != 2 or set(requirement_map.values()) != set(predicate_by_id)):
        raise ValueError("complete finite requirement-to-predicate map required")
    task_requirements = {}
    for ordinal, requirement_id in enumerate(sorted(requirement_map)):
        predicate = predicate_by_id[requirement_map[requirement_id]]
        producer_id, task_id = f"finite-fixture-producer:{ordinal}", f"finite-fixture-task:{ordinal}"
        task_requirements[task_id] = requirement_id
        producers.append(ProducerRule(producer_id=producer_id,
            effect_predicate_ids=(predicate.predicate_id,), provenance_refs=intent.source_refs,
            task_candidate_ids=(task_id,)))
        candidates.append(TaskCandidate(candidate_id=task_id, producer_id=producer_id,
            closes_obligation_ids=((obligation_id_for_predicate(predicate.predicate_id)
                if predicate.predicate_id in satisfied else
                obligation_id_for_producer(producer_id, predicate.predicate_id)),),
            provenance_refs=intent.source_refs))
    graph = ObligationGraphCompiler().compile(intent, current_facts=match["current_facts"],
        producers=producers, task_candidates=candidates, current_root_id=intent.current_root_id)
    if graph.planning_blocked or graph.review_required:
        raise ValueError("bounded finite intent graph requires review")
    goal = FrozenPlanningGoal(goal_id="finite-fixture-goal", goal_content_id=digest(intent.to_dict()),
        repository_tree_id=intent.current_root_id, policy=EvidenceAwarePlanPolicy(
            acceptance_criteria=intent.goal_predicate_ids, evidence_terms=intent.source_refs,
            supported_semantics=("python-integer-offset-finite@1",),
            allowed_scopes=("scope:calc.py",), available_resource_classes=("cpu",),
            require_validation=True, require_proof=False))
    context = {"domain": "finite-integer-fixture", "repository_paths": ["calc.py"],
        "finite_match_sha256": digest(match), "evidence_scope": "exact_declared_finite_inputs",
        "task_metadata": {task.candidate_id: {"predicted_files": ["calc.py"],
            "predicted_symbols": ["increment"], "scope_ids": ["scope:calc.py"],
            "resource_classes": ["cpu"]} for task in candidates}}
    portfolio, selected = None, None
    try:
        portfolio = SymbolicCandidatePlanner(bounds=SymbolicCandidateBounds(
            candidate_count=1, max_model_candidates=0)).plan(graph, goal, context, allow_model=False)
    except SymbolicCandidatePlanningError as error:
        roots = [node for node in graph.nodes if node.obligation_id in graph.root_obligation_ids]
        if (str(error) != "obligation graph is already complete; no task candidate is invented"
                or len(roots) != 2 or any(node.status.value != "discharged" for node in roots)):
            raise
    else:
        if portfolio.selected is None:
            raise ValueError("native finite symbolic portfolio did not select a candidate")
        selected = portfolio.selected.symbolic_candidate
    selected_ids = list(selected.schedule.task_ids) if selected else []
    by_id = {task.candidate_id: task for task in candidates}
    critic_plan = {"plan_id": selected.candidate_id if selected else digest(graph.to_dict()), "tasks": [{"task_id": task_id,
        "depends_on": [], "closes_obligation_ids": list(by_id[task_id].closes_obligation_ids)}
        for task_id in selected_ids], "effects": []}
    critique = PlanCritic().critique(critic_plan, obligation_graph=graph,
        required_goal_ids=graph.root_obligation_ids)
    if not critique.accepted:
        raise ValueError("native finite symbolic selection failed critique: " + str(critique.to_dict()))
    return {"schema": "terminal-codebase-finite-symbolic-planning@1",
        "typed_intent": intent.to_dict(), "obligation_graph": graph.to_dict(),
        "portfolio": portfolio.to_dict() if portfolio else None, "critic": critique.to_dict(),
        "planner_status": "selected" if selected else "already_complete_in_finite_domain",
        "declared_task_candidate_ids": [task.candidate_id for task in candidates],
        "declared_task_requirement_ids": task_requirements,
        "selected_task_ids": selected_ids,
        "both_declared_clause_roots_preserved": len(graph.root_obligation_ids) == 2,
        "provider_calls": 0, "worker_launched": False, "production_admitted": False,
        "omission_authority": False, "completion_authority": False,
        "evidence_scope": "authored_finite_domain_only"}


def run_finite_experiment(*, output: Path, parent: Path, python_executable: Path,
        lean_executable: Path):
    import duckdb
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog
    from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex, StaleCodebaseError
    from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
    from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
    from ipfs_datasets_py.logic.software_contracts.codebase_finite_integer_observation import (
        seal_finite_integer_tools, observe_finite_integer_source,
    )
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
        GlobalResourceScheduler, ResourceSchedulerConfig,
    )
    from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_codebase import (
        build_finite_integer_intent, match_finite_integer_intent,
    )
    from ipfs_accelerate_py.agent_supervisor.planning.structural_codebase_context import run_with_structural_codebase_context
    from .codebase_ir_metadata import hydrate_codebase_ir_metadata, validate_codebase_ir_metadata
    from .terminal_codebase_supervisor_fixture import (
        bound_terminal_codebase_metadata_records, reconstruct_terminal_codebase_metadata_records,
    )
    output, parent = Path(output), Path(parent)
    if not output.is_absolute() or output.resolve() != output or output.exists() or not output.parent.is_dir():
        raise ValueError("new canonical experiment namespace required")
    started, clock = datetime.now(timezone.utc).isoformat(), time.monotonic()
    parent_pin = _pin(parent / "result.json")
    old = json.loads((parent / "result.json").read_bytes())
    if old.get("status") != "completed":
        raise ValueError("completed repository qualification required")
    old_audit = json.loads((parent / "audit.json").read_bytes())
    parent_artifacts = [_pin(parent / row["relative_path"]) for row in old_audit["artifacts"]]
    if any({key: pin[key] for key in ("bytes", "sha256")} !=
            {key: row[key] for key in ("bytes", "sha256")}
            for pin, row in zip(parent_artifacts, old_audit["artifacts"])):
        raise ValueError("prior repository qualification artifact drift")
    import importlib
    implementation_names = (
        __name__, "benchmarks.agent_supervisor.container_coding.terminal_codebase_finite_index",
        "ipfs_datasets_py.logic.software_contracts.codebase_finite_integer_observation",
        "ipfs_accelerate_py.agent_supervisor.planning.finite_integer_codebase",
        "ipfs_datasets_py.logic.software_contracts.codebase_integer_profile",
        "ipfs_datasets_py.logic.software_contracts.codebase_ir",
        "ipfs_datasets_py.logic.software_verification.pipeline",
        "ipfs_datasets_py.logic.software_verification.source_adapters",
        "ipfs_datasets_py.logic.backends.process",
    )
    implementation_pins = [_pin(importlib.import_module(name).__file__) for name in implementation_names]
    capture_path = Path(old["parent"]["intent_signed_capture"]["path"])
    capture_pin = _pin(capture_path)
    capture = json.loads(capture_path.read_bytes())
    original = Path(capture["manifest"]["payload"]["repository"]) / "bottle.py"
    original_pin = _pin(original)
    if original_pin["sha256"] != "761756ce31753e526c48d28ccbca13a5d2493b16fe37aff3e1e4d2efaf3a2bba" or original_pin["bytes"] != 175565:
        raise ValueError("exact original Terminal Bench Bottle source required")
    output.mkdir(mode=0o700)
    repository = output / "authored-repository"
    _repo(repository, {"calc.py": _source(1), "intent.txt": INTENT.encode()})
    benchmark_repository = output / "unsupported-benchmark-repository"
    _repo(benchmark_repository, {"bottle.py": original.read_bytes()})
    policy = seal_finite_integer_tools(python_executable=python_executable, lean_executable=lean_executable)
    _write(output / "tool-policy.json", policy)
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=output / "resource-admission.json", lane_reservations={}, auto_renew_leases=False))
    cx = duckdb.connect(str(output / "codebase.duckdb"), config={"threads": 1, "memory_limit": "64MB"})
    store = DuckDBASTStore(connection=cx)
    artifacts = ImmutableCAS(output / "cas")
    index = RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=artifacts,
        catalog=CodebaseCatalog(store, artifacts))
    records = {}
    def extend(values):
        for family, rows in values.items():
            target = records.setdefault(family, [])
            start = len(target)
            target.extend({"schema": "terminal-finite-metadata-occurrence@1",
                "occurrence": start + ordinal, "record": row} for ordinal, row in enumerate(rows))
    repository_id = "repository:authored-finite-integer-qualification"
    native = build_finite_integer_intent(INTENT)
    doc = native.to_dict()
    _write(output / "intent-ir.json", doc)
    results, evidence_indexes, queries, fresh_queries, plans, captures = [], [], [], [], [], []
    def plan_owned(head, matched):
        return run_with_structural_codebase_context(index, repository,
            lambda context: _native_plan(matched), repository_id=repository_id,
            expected_head=head, scheduler=scheduler)
    try:
        first = index.prepare_current(repository, repository_id=repository_id,
            operation_id="finite-initial", expected_head=None, scheduler=scheduler).head
        head = first
        git_head = _git(repository, "rev-parse", "HEAD")
        stale = []
        for label in ("before", "after"):
            current = _capture(index, repository, head, scheduler)
            captures.append(current)
            extend(current)
            matched = match_finite_integer_intent(index=index, repository=repository,
                repository_id=repository_id, intent_document=doc, source_text=INTENT,
                expected_head=head, output=output / (label + "-observation"), tool_policy=policy,
                scheduler=scheduler)
            _write(output / (label + "-match.json"), matched)
            evidence = persist_finite_evidence_index(match=matched, output=output / (label + "-index"))
            query = query_finite_evidence_index(index=index, repository=repository, expected_head=head,
                expected=evidence, scheduler=scheduler)
            cx.close()
            restarted = _fresh_query(output, label, repository, head, evidence)
            if restarted != query:
                raise ValueError("cold finite evidence index query differs")
            cx = duckdb.connect(str(output / "codebase.duckdb"), config={"threads": 1, "memory_limit": "64MB"})
            store = DuckDBASTStore(connection=cx)
            index = RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=artifacts,
                catalog=CodebaseCatalog(store, artifacts))
            plan = plan_owned(head, matched)
            index.observe_current(repository, expected_head=head, scheduler=scheduler)
            _write(output / (label + "-plan.json"), plan)
            results.append(matched); evidence_indexes.append(evidence); queries.append(query); fresh_queries.append(restarted); plans.append(plan)
            if label == "before":
                (repository / "calc.py").write_bytes(_source(2))
                if _git(repository, "rev-parse", "HEAD") != git_head:
                    raise ValueError("successor must retain the same Git HEAD")
                operations = {
                    "old_observation": lambda: match_finite_integer_intent(index=index, repository=repository,
                        repository_id=repository_id, intent_document=doc, source_text=INTENT, expected_head=first,
                        output=output / "stale-observation", tool_policy=policy, scheduler=scheduler),
                    "old_index": lambda: query_finite_evidence_index(index=index, repository=repository,
                        expected_head=first, expected=evidence, scheduler=scheduler),
                    "old_planning_materials": lambda: plan_owned(first, matched),
                }
                for kind, call in operations.items():
                    try:
                        call()
                    except StaleCodebaseError as error:
                        stale.append({"control": kind, "rejected": True, "error": str(error)})
                    else:
                        raise ValueError("same-HEAD source edit accepted historical finite evidence: " + kind)
                head = index.prepare_current(repository, repository_id=repository_id,
                    operation_id="finite-successor", expected_head=first, scheduler=scheduler).head
        if len(results[0]["current_facts"]) != 1 or len(results[1]["current_facts"]) != 2:
            raise ValueError("actual two-clause finite observation gate differs")
        if plans[0]["declared_task_candidate_ids"] != plans[1]["declared_task_candidate_ids"]:
            raise ValueError("declared finite fixture task population changed")
        if plans[0]["declared_task_requirement_ids"] != plans[1]["declared_task_requirement_ids"]:
            raise ValueError("declared finite fixture task meanings changed")
        benchmark_head = index.prepare_current(benchmark_repository,
            repository_id="repository:unsupported-original-bottle", operation_id="bottle-control",
            expected_head=None, scheduler=scheduler).head
        unsupported = observe_finite_integer_source(index=index, repository=benchmark_repository,
            expected_head=benchmark_head, contract=IntegerOffsetContract("bottle.py", "increment", "n", 2),
            inputs=INPUTS, output=output / "unsupported-bottle-observation", tool_policy=policy, scheduler=scheduler)
        if unsupported["status"] != "unsupported" or unsupported["runtime_observation_coverage_complete"]:
            raise ValueError("original Bottle incorrectly accepted by finite integer profile")
        extend(_capture(index, benchmark_repository, benchmark_head, scheduler))
        extend({"intent_ir": [doc], "contracts": [result["observation"]["contract"] for result in results],
            "observations": [result["observation"] for result in results],
            "current_facts": [fact for result in results for fact in result["current_facts"]],
            "runtime_traces": [result["observation"]["trace"] for result in results],
            "lean_certificates": [result["observation"]["lean_certificate"] for result in results],
            "compiled_logic": [json.loads(Path(result["observation"]["artifacts"]["compiled"]["path"]).read_bytes()) for result in results],
            "finite_matches": results, "finite_indexes": evidence_indexes, "finite_queries": queries,
            "fresh_finite_queries": fresh_queries,
            "finite_plans": plans, "stale_controls": stale, "unsupported_controls": [unsupported],
            "tool_policies": [policy]})
        cx.close()
        bounded = bound_terminal_codebase_metadata_records(records)
        if wire(reconstruct_terminal_codebase_metadata_records(bounded)) != wire(records):
            raise ValueError("complete finite metadata packaging changed records")
        metadata = hydrate_codebase_ir_metadata(records=bounded, output=output / "metadata",
            source_snapshot={"schema": "terminal-finite-observation-source-binding@1",
                "parent": parent_pin, "original_terminal_bench_source": original_pin,
                "heads": [first.to_dict(), head.to_dict(), benchmark_head.to_dict()],
                "complete_producer_sha256": digest(records), "training_steps": 0})
        replay = validate_codebase_ir_metadata(output=output / "metadata", expected=metadata, fresh_process=True)
        restored = {family: [json.loads(line)["payload"] for line in
            (output / "metadata" / desc["relative_path"]).read_bytes().splitlines()]
            for family, desc in replay["exports"].items()}
        if wire(reconstruct_terminal_codebase_metadata_records(restored)) != wire(records):
            raise ValueError("complete native finite metadata readback differs")
        if _pin(parent_pin["path"]) != parent_pin or _pin(capture_pin["path"]) != capture_pin or _pin(original_pin["path"]) != original_pin:
            raise ValueError("prior qualification or original public source changed")
        if any(_pin(row["path"]) != row for row in parent_artifacts + implementation_pins):
            raise ValueError("prior artifact or finite implementation changed during execution")
        if scheduler.snapshot()["active_lease_count"] or scheduler.snapshot()["waiting_request_count"]:
            raise ValueError("finite experiment leaked a resource reservation")
        result = {"schema": SCHEMA, "status": "completed", "output": str(output),
            "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.monotonic() - clock, "parent": parent_pin,
            "execution_sources": implementation_pins, "parent_artifacts_unchanged": parent_artifacts,
            "original_terminal_bench_source": original_pin, "intent_ir": doc,
            "before": results[0], "after": results[1], "indexes": evidence_indexes,
            "queries": queries, "fresh_queries": fresh_queries, "plans": plans, "same_git_head": git_head,
            "stale_controls": stale, "unsupported_bottle": unsupported,
            "metadata": metadata, "metadata_replay": replay,
            "complete_producer_sha256": digest(records),
            "complete_family_counts": {family: len(rows) for family, rows in records.items()},
            "zero_truncation": True, "training_steps": 0, "provider_calls": 0,
            "source_semantics_verified": False, "universal_integer_correctness": False,
            "public_bottle_task_satisfied": False, "production_admitted": False,
            "worker_launched": False, "omission_authority": False, "completion_authority": False}
        _write(output / "metadata-result.json", metadata)
        _write(output / "metadata-replay.json", replay)
        _write(output / "result.json", result)
        return result
    except BaseException as error:
        cx.close()
        _write(output / "failure.json", {"schema": SCHEMA, "status": "failed", "error": repr(error),
            "started_at": started, "wall_seconds": time.monotonic() - clock})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable).resolve())
    parser.add_argument("--lean", type=Path, required=True)
    args = parser.parse_args()
    result = run_finite_experiment(output=args.output, parent=args.parent,
        python_executable=args.python, lean_executable=args.lean)
    print(json.dumps({key: result[key] for key in ("status", "output", "wall_seconds", "complete_family_counts")}, sort_keys=True))
