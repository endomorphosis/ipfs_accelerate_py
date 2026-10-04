"""Retained native finite observations through the real strict planning service.

Only an authored closed integer fixture executes. This records finite arithmetic,
complete task meanings, historical index readback, and real metadata storage. It
does not run a worker, authenticate signed evidence, fit a model, or prove Python
semantics, broad benchmark success, or optimizer convergence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

from .terminal_codebase_finite_experiment import INTENT, _capture, _git, _pin, _repo, _source, _write
from .terminal_codebase_finite_index import (
    digest, persist_finite_evidence_index, query_finite_evidence_index, wire,
)

SCHEMA = "terminal-codebase-finite-service-experiment@1"
_MODULE = "benchmarks.agent_supervisor.container_coding.terminal_codebase_finite_service_experiment"
_FALSE = {name: False for name in (
    "source_semantics_verified", "runtime_behavior_verified", "behavior_authority",
    "proof_authority", "execution_authority", "completion_authority", "mutation_authority",
    "production_admitted", "worker_launched", "omission_authority", "signed_evidence_admitted",
    "universal_integer_correctness", "public_terminal_bench_task_satisfied", "convergence_proved")}


def _catalog():
    from ipfs_accelerate_py.agent_supervisor.planning import finite_integer_codebase as matcher
    from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_plan_preview import (
        FiniteIntegerOperationCatalog, ReviewedFiniteIntegerOperation,
    )
    return FiniteIntegerOperationCatalog(operations=tuple(ReviewedFiniteIntegerOperation(
        requirement_id=requirement, task_id=task, producer_id=producer,
        path="calc.py", function_name="increment", parameter="n",
        review_ref="review:authored-finite-service-operations")
        for requirement, task, producer in (
            (matcher.TYPE_STATEMENT_ID, "task:finite:type", "producer:finite:type"),
            (matcher.OFFSET_STATEMENT_ID, "task:finite:offset", "producer:finite:offset"))))


def _authority_materials():
    """Actual unsigned fixture policy records; none advertises production authority."""
    return {name: {"schema": "authored-finite-service-authority-material@1", "role": name,
                   "profile": "python-integer-offset-finite@1", "production_activated": False,
                   "worker_allowed": False, "model_calls": 0}
            for name in ("policy", "legal", "security", "provider", "usage", "configuration")}


def _request(index, repository, head, document, catalog, authority):
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
    from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_plan_preview import (
        finite_integer_intent_cid, finite_integer_prompt_cid,
    )
    from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
        DirtyTreePolicy, PlanAuthorityRoots, PlanCreateRequest, PlanRequestBudget, TaskSourceKind,
    )
    cid = lambda role: cid_for_structured(authority[role])
    roots = PlanAuthorityRoots(repository_id=head.repository_id,
        repository_root_cid=head.snapshot_cid, dirty_worktree_root=head.snapshot_cid,
        task_source_id=cid_for_structured({"schema": "authored-finite-service-task-source@1",
            "intent_ir_root": finite_integer_intent_cid(document), "operation_catalog_cid": catalog.cid}),
        task_source_revision=catalog.cid, policy_root=cid("policy"),
        intent_ir_root=finite_integer_intent_cid(document), legal_ir_root=cid("legal"),
        security_ir_root=cid("security"), program_root=index.load(head.manifest_cid).semantic_state.state_cid,
        capability_catalog_root=catalog.cid, provider_catalog_root=cid("provider"),
        usage_policy_root=cid("usage"), configuration_root=cid("configuration"))
    return PlanCreateRequest(prompt_source_cid=finite_integer_prompt_cid(INTENT),
        repository_id=head.repository_id, repository_root=str(repository), scope_paths=("calc.py",),
        dirty_tree_policy=DirtyTreePolicy.OBSERVE_AND_BIND, task_source_kind=TaskSourceKind.BOTH,
        board_namespace="authored-finite-service", alias_prefix="CFS", roots=roots,
        budget=PlanRequestBudget(max_goals=2, max_tasks=2, max_model_calls=0, max_latency_ms=90_000),
        required_analysis_operations=(), optional_analysis_operations=(),
        required_logic_families=(), optional_logic_families=(), observe_roots=True,
        supervisor_profile="authored-finite-service-preview", caller="principal:authored-qualification")


def _open(output):
    import duckdb
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog
    from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
    from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
    from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
    cx = duckdb.connect(str(output / "codebase.duckdb"), config={"threads": 1, "memory_limit": "64MB"})
    store, artifacts = DuckDBASTStore(connection=cx), ImmutableCAS(output / "cas")
    return cx, RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=artifacts,
                                     catalog=CodebaseCatalog(store, artifacts))


def _scheduler(path):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
        GlobalResourceScheduler, ResourceSchedulerConfig,
    )
    return GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=path, lane_reservations={}, auto_renew_leases=False))


def _preview(index, repository, head, scheduler, document, catalog, policy, authority_path, output):
    from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_plan_preview import preview_finite_integer_plan
    from ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview import RepositoryPlanPreviewOwner
    authority = json.loads(authority_path.read_bytes())
    prompt_path = authority_path.parent / "prompt.txt"
    if prompt_path.read_bytes() != INTENT.encode("utf-8"):
        raise ValueError("exact retained authored prompt bytes differ")
    request = _request(index, repository, head, document, catalog, authority)

    def observe_policy(bound):
        if bound != request:
            raise ValueError("finite service changed the exact authored request")
        if prompt_path.read_bytes() != INTENT.encode("utf-8"):
            raise ValueError("authored prompt changed during finite service preview")
        live = _request(index, repository, head, document, catalog, json.loads(authority_path.read_bytes()))
        live.roots.require_current(request.roots)
        return live.roots

    owner = RepositoryPlanPreviewOwner(index=index, repository=repository, expected_head=head,
                                      scheduler=scheduler, timeout_seconds=90, memory_mb=1024)
    result = preview_finite_integer_plan(owner=owner, request=request, intent_document=document,
        source_text=INTENT, operation_catalog=catalog, output=output,
        tool_policy=policy, policy_observer=observe_policy)
    return result, request


def _validate_result(result, *, facts, selected):
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
    from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import ObligationGraph
    from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_codebase import OFFSET_STATEMENT_ID
    from ipfs_accelerate_py.agent_supervisor.planning.plan_critic import PlanCritic, PlanCritique
    from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import (
        PlanCreateInputSnapshot, PlanCreatePreviewReceipt,
    )
    if result["result_cid"] != cid_for_structured({key: value for key, value in result.items() if key != "result_cid"}):
        raise ValueError("complete native service result identity differs")
    preview = PlanCreatePreviewReceipt.from_dict(result["preview"])
    snapshot = PlanCreateInputSnapshot.from_dict(result["input_snapshot"])
    graph = ObligationGraph.from_dict(result["obligation_graph"])
    critique = PlanCritique.from_dict(result["critique"])
    plan = result["candidate_plan"]
    replay = PlanCritic().critique(plan, obligation_graph=graph, evidence=result["critic_evidence"],
        required_goal_ids=graph.root_obligation_ids, expected_effects=plan["expected_effect_ids"])
    if (result["current_facts_count"] != facts or len(graph.facts) != facts
            or len(graph.root_obligation_ids) != 2 or result["selected_task_ids"] != selected
            or result["match"]["residual_clause_ids"] != ([OFFSET_STATEMENT_ID] if selected else [])
            or [task["task_id"] for task in plan["tasks"]] != selected
            or preview.input_snapshot_cid != snapshot.snapshot_cid
            or not critique.accepted or critique.truncated or replay != critique
            or preview.verdict.value != "review_only" or not preview.read_only or preview.wrote_effects
            or result["model_calls"] != 0 or result["training_steps"] != 0
            or any(result.get(name) is not False for name in (
                "production_admitted", "worker_launched", "source_semantics_verified",
                "proof_authority", "execution_authority", "completion_authority"))):
        raise ValueError("actual finite service result, complete roots, critic or authorities differ")
    stages = {stage.stage.value: stage for stage in preview.stage_results}
    if any(not stages[name].passed for name in ("scan", "query", "evidence", "obligation", "candidate", "critique")):
        raise ValueError("required actual finite service stage failed")
    if stages["admission"].passed or "ir_admission_materials_absent" not in stages["admission"].blockers:
        raise ValueError("finite proposal incorrectly acquired admission")
    if selected:
        if (stages["parallel_plan"].passed or result["execution_plan"]["admitted"] is not False
                or "parallel:stale_capacity" not in stages["parallel_plan"].blockers):
            raise ValueError("real nonempty parallel capacity debt was hidden")
    elif (result["planner_status"] != "already_complete_in_finite_domain" or plan["effects"]
            or result["execution_plan"]["status"] != "no_execution_requested"
            or result["execution_plan"]["admitted"] is not False):
        raise ValueError("complete finite roots did not retain explicit no-work proposal")


def _cold_validate(request_path):
    """Cold process opens the real owner and makes another fresh native preview."""
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_codebase import build_finite_integer_intent
    request = json.loads(Path(request_path).read_bytes())
    output, repository = Path(request["output"]), Path(request["repository"])
    head = CodebaseHead.from_dict(request["head"])
    scheduler = _scheduler(output / "restart-resource-admission.json")
    cx, index = _open(output)
    try:
        stored = []
        for row in request["preview_artifacts"]:
            if _pin(row["path"]) != row:
                raise ValueError("retained service proposal changed before cold validation")
            stored.append(json.loads(Path(row["path"]).read_bytes()))
        _validate_result(stored[0], facts=1, selected=["task:finite:offset"])
        _validate_result(stored[1], facts=2, selected=[])
        query = query_finite_evidence_index(index=index, repository=repository, expected_head=head,
            expected=request["expected_index"], scheduler=scheduler)
        if query["current_facts"] or query["cache_authority"] != "historical_candidate_only":
            raise ValueError("cold historical query advertised current facts")
        index.observe_current(repository, expected_head=head, scheduler=scheduler)
        fresh, current = _preview(index, repository, head, scheduler,
            build_finite_integer_intent(INTENT), _catalog(), request["tool_policy"],
            output / "authority-materials.json", output / "restart-observation")
        _validate_result(fresh, facts=2, selected=[])
        if current.to_dict() != request["expected_request"]:
            raise ValueError("cold request reconstruction changed authority roots")
        if fresh["match"]["observation_cid"] == stored[1]["match"]["observation_cid"]:
            raise ValueError("cold preview reused a historical observation instead of executing freshly")
        _write(output / "restart-preview.json", fresh)
        if any(_pin(row["path"]) != row for row in request["preview_artifacts"]):
            raise ValueError("cold validation changed a retained proposal")
        state = scheduler.snapshot()
        if state["active_lease_count"] or state["waiting_request_count"]:
            raise ValueError("cold service preview leaked a resource reservation")
        return {"schema": "terminal-codebase-finite-service-cold-validation@1", "status": "completed",
            "head": head.to_dict(), "historical_index_query": query, "current_facts_from_index": 0,
            "fresh_observation_executed": True, "fresh_preview": _pin(output / "restart-preview.json"),
            "fresh_request": current.to_dict(),
            "fresh_current_facts_count": fresh["current_facts_count"],
            "fresh_selected_task_ids": fresh["selected_task_ids"], "retained_critics_replayed": 2,
            "active_lease_count": 0, "waiting_request_count": 0, "model_calls": 0, "training_steps": 0,
            **_FALSE}
    finally:
        cx.close()


def run_finite_service_experiment(*, output: Path, python_executable: Path,
                                  lean_executable: Path, parent: Path | None = None):
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import StaleCodebaseError
    from ipfs_datasets_py.logic.software_contracts.codebase_finite_integer_observation import seal_finite_integer_tools
    from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_codebase import build_finite_integer_intent
    from .codebase_ir_metadata import hydrate_codebase_ir_metadata, validate_codebase_ir_metadata
    from .terminal_codebase_supervisor_fixture import (
        bound_terminal_codebase_metadata_records, reconstruct_terminal_codebase_metadata_records,
    )
    output = Path(output)
    if (not output.is_absolute() or output.resolve() != output or output.exists()
            or not output.parent.is_dir()):
        raise ValueError("new canonical finite-service experiment directory required")
    started, clock = datetime.now(timezone.utc).isoformat(), time.monotonic()
    parent_pins = []
    if parent is not None:
        parent = Path(parent).resolve(strict=True)
        for name in ("result.json", "audit.json"):
            if (parent / name).is_file():
                parent_pins.append(_pin(parent / name))
        if not parent_pins or not (parent / "result.json").is_file():
            raise ValueError("existing parent result required when a parent is supplied")
    modules = (_MODULE, "benchmarks.agent_supervisor.container_coding.terminal_codebase_finite_experiment",
        "benchmarks.agent_supervisor.container_coding.terminal_codebase_finite_index",
        "benchmarks.agent_supervisor.container_coding.codebase_ir_metadata",
        "benchmarks.agent_supervisor.container_coding.terminal_codebase_supervisor_fixture",
        "ipfs_accelerate_py.agent_supervisor.planning.finite_integer_plan_preview",
        "ipfs_accelerate_py.agent_supervisor.planning.finite_integer_plan_service",
        "ipfs_accelerate_py.agent_supervisor.planning.finite_integer_codebase",
        "ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service",
        "ipfs_accelerate_py.agent_supervisor.planning.plan_critic",
        "ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler",
        "ipfs_accelerate_py.agent_supervisor.planning.symbolic_candidate_planner",
        "ipfs_accelerate_py.agent_supervisor.planning.plan_evaluator",
        "ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts",
        "ipfs_accelerate_py.agent_supervisor.planning.structural_codebase_context",
        "ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview",
        "ipfs_datasets_py.logic.intent_ir.canonicalize",
        "ipfs_datasets_py.logic.intent_ir.schema",
        "ipfs_datasets_py.logic.software_contracts.codebase_finite_integer_observation",
        "ipfs_datasets_py.logic.software_contracts.codebase_integer_profile",
        "ipfs_datasets_py.logic.software_contracts.codebase_ir",
        "ipfs_datasets_py.logic.backends.process")
    sources = [_pin(importlib.import_module(name).__file__) for name in modules]
    output.mkdir(mode=0o700)
    repository = output / "authored-repository"
    cx = None
    try:
        _repo(repository, {"calc.py": _source(1)})
        git_head = _git(repository, "rev-parse", "HEAD")
        document, catalog = build_finite_integer_intent(INTENT), _catalog()
        policy = seal_finite_integer_tools(python_executable=python_executable, lean_executable=lean_executable)
        with (output / "prompt.txt").open("xb") as stream:
            stream.write(INTENT.encode("utf-8"))
        prompt_pin = _pin(output / "prompt.txt")
        _write(output / "intent-ir.json", document.to_dict())
        _write(output / "operation-catalog.json", catalog.to_dict())
        _write(output / "tool-policy.json", policy)
        _write(output / "authority-materials.json", _authority_materials())
        authority_pin = _pin(output / "authority-materials.json")
        scheduler = _scheduler(output / "resource-admission.json")
        cx, index = _open(output)
        repository_id = "repository:authored-finite-service-qualification"
        first = index.prepare_current(repository, repository_id=repository_id,
            operation_id="finite-service-initial", expected_head=None, scheduler=scheduler).head
        head, results, requests, indexes, queries, controls, records = first, [], [], [], [], [], {}

        def extend(values):
            for family, rows in values.items():
                target = records.setdefault(family, [])
                start = len(target)
                target.extend({"schema": "terminal-finite-service-metadata-occurrence@1",
                    "occurrence": start + ordinal, "record": row} for ordinal, row in enumerate(rows))

        for label in ("before", "after"):
            extend(_capture(index, repository, head, scheduler))
            result, request = _preview(index, repository, head, scheduler, document, catalog, policy,
                                      output / "authority-materials.json", output / (label + "-observation"))
            _validate_result(result, facts=1 if label == "before" else 2,
                             selected=["task:finite:offset"] if label == "before" else [])
            _write(output / (label + "-preview.json"), result)
            _write(output / (label + "-request.json"), request.to_dict())
            evidence = persist_finite_evidence_index(match=result["match"], output=output / (label + "-index"))
            query = query_finite_evidence_index(index=index, repository=repository, expected_head=head,
                                              expected=evidence, scheduler=scheduler)
            if query["current_facts"]:
                raise ValueError("historical index incorrectly supplied current planning facts")
            results.append(result)
            requests.append(request.to_dict())
            indexes.append(evidence)
            queries.append(query)
            if label == "before":
                (repository / "calc.py").write_bytes(_source(2))
                if _git(repository, "rev-parse", "HEAD") != git_head:
                    raise ValueError("authored successor must retain the same Git HEAD")
                calls = {
                    "old_owner_and_request_after_same_head_edit": lambda: _preview(
                        index, repository, first, scheduler, document, catalog, policy,
                        output / "authority-materials.json", output / "stale-observation"),
                    "old_historical_index_after_same_head_edit": lambda: query_finite_evidence_index(
                        index=index, repository=repository, expected_head=first, expected=evidence, scheduler=scheduler),
                }
                for control, call in calls.items():
                    try:
                        call()
                    except StaleCodebaseError as error:
                        controls.append({"control": control, "rejected": True, "error": str(error)})
                    else:
                        raise ValueError("stale same-HEAD finite service control was accepted: " + control)
                if (output / "stale-observation").exists():
                    raise ValueError("stale service request executed a native observer")
                head = index.prepare_current(repository, repository_id=repository_id,
                    operation_id="finite-service-successor", expected_head=first, scheduler=scheduler).head
                try:
                    _preview(index, repository, first, scheduler, document, catalog, policy,
                             output / "authority-materials.json", output / "superseded-observation")
                except StaleCodebaseError as error:
                    controls.append({"control": "superseded_owner_generation", "rejected": True, "error": str(error)})
                else:
                    raise ValueError("superseded owner generation was accepted")
        if results[0]["declared_task_requirement_ids"] != results[1]["declared_task_requirement_ids"]:
            raise ValueError("captured successor changed complete reviewed task meanings")
        cx.close()
        cx = None
        restart_request = {"schema": "terminal-finite-service-restart-request@1", "output": str(output),
            "repository": str(repository), "head": head.to_dict(), "tool_policy": policy,
            "expected_index": indexes[1], "expected_request": request.to_dict(),
            "preview_artifacts": [_pin(output / (label + "-preview.json")) for label in ("before", "after")]}
        _write(output / "restart-request.json", restart_request)
        command = [sys.executable, "-m", _MODULE, "--cold-request", str(output / "restart-request.json")]
        child = subprocess.run(command, capture_output=True, text=True, timeout=120)
        _write(output / "restart-process.json", {"command": command, "returncode": child.returncode,
            "stdout": child.stdout, "stderr": child.stderr, "request": _pin(output / "restart-request.json")})
        if child.returncode != 0:
            raise ValueError("cold finite service validation failed: " + child.stderr)
        cold = json.loads(child.stdout)
        if cold["status"] != "completed":
            raise ValueError("cold finite service validation did not complete")
        _write(output / "restart-validation.json", cold)
        fresh = json.loads((output / "restart-preview.json").read_bytes())
        _validate_result(fresh, facts=2, selected=[])
        results.append(fresh)
        requests.append(cold["fresh_request"])
        queries.append(cold["historical_index_query"])
        cx, index = _open(output)
        extend(_capture(index, repository, head, scheduler))
        cx.close()
        cx = None
        matches = [result["match"] for result in results]
        extend({"intent_ir": [document.to_dict()], "contracts": [match["observation"]["contract"] for match in matches],
            "observations": [match["observation"] for match in matches],
            "current_facts": [fact for match in matches for fact in match["current_facts"]],
            "runtime_traces": [match["observation"]["trace"] for match in matches],
            "lean_certificates": [match["observation"]["lean_certificate"] for match in matches],
            "compiled_logic": [json.loads(Path(match["observation"]["artifacts"]["compiled"]["path"]).read_bytes()) for match in matches],
            "finite_matches": matches, "finite_indexes": indexes, "finite_queries": queries,
            "service_results": [*results, cold], "service_input_snapshots": [result["input_snapshot"] for result in results],
            "request_roots": requests,
            "obligation_graphs": [result["obligation_graph"] for result in results],
            "candidate_portfolios": [result["portfolio"] for result in results],
            "candidate_plans": [result["candidate_plan"] for result in results],
            "plan_critiques": [result["critique"] for result in results],
            "critic_evidence": [result["critic_evidence"] for result in results],
            "execution_plans": [result["execution_plan"] for result in results],
            "operation_catalogs": [catalog.to_dict()], "stale_controls": controls,
            "authority_materials": [{"schema": "authored-finite-service-authority-inputs@1",
                "authored_materials": json.loads((output / "authority-materials.json").read_bytes()),
                "native_tool_policy": policy, "production_activated": False}]})
        bounded = bound_terminal_codebase_metadata_records(records)
        if wire(reconstruct_terminal_codebase_metadata_records(bounded)) != wire(records):
            raise ValueError("complete service metadata packaging changed producer records")
        metadata = hydrate_codebase_ir_metadata(records=bounded, output=output / "metadata",
            source_snapshot={"schema": "terminal-finite-service-source-binding@1", "parent_artifacts": parent_pins,
                "heads": [first.to_dict(), head.to_dict()], "complete_producer_sha256": digest(records),
                "prompt": prompt_pin,
                "profile": "python-integer-offset-finite@1", "training_steps": 0, "model_calls": 0})
        replay = validate_codebase_ir_metadata(output=output / "metadata", expected=metadata, fresh_process=True)
        restored = {family: [json.loads(line)["payload"] for line in
            (output / "metadata" / description["relative_path"]).read_bytes().splitlines()]
            for family, description in replay["exports"].items()}
        if wire(reconstruct_terminal_codebase_metadata_records(restored)) != wire(records):
            raise ValueError("cold complete DuckDB/DuckLake metadata readback differs")
        if any(_pin(row["path"]) != row for row in sources + parent_pins + [authority_pin, prompt_pin]):
            raise ValueError("implementation, parent, or authority materials changed during qualification")
        state = scheduler.snapshot()
        if state["active_lease_count"] or state["waiting_request_count"]:
            raise ValueError("finite service harness leaked a resource reservation")
        result = {"schema": SCHEMA, "status": "completed", "output": str(output),
            "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.monotonic() - clock, "profile": "python-integer-offset-finite@1",
            "fixture_scope": "authored_closed_integer_function_only", "domain_inputs": [-2, -1, 0, 1, 2],
            "same_git_head": git_head, "heads": [first.to_dict(), head.to_dict()],
            "before": results[0], "after": results[1], "cold_fresh_preview": results[2], "prompt": prompt_pin,
            "indexes": indexes, "historical_queries": queries, "cold_validation": cold,
            "stale_controls": controls, "metadata": metadata, "metadata_replay": replay,
            "complete_producer_sha256": digest(records),
            "complete_family_counts": {family: len(rows) for family, rows in records.items()},
            "zero_truncation": True, "execution_sources": sources, "parent_artifacts_unchanged": parent_pins,
            "active_lease_count": 0, "waiting_request_count": 0, "training_steps": 0, "provider_calls": 0,
            **_FALSE}
        _write(output / "metadata-result.json", metadata)
        _write(output / "metadata-replay.json", replay)
        _write(output / "result.json", result)
        return result
    except BaseException as error:
        if cx is not None:
            cx.close()
        _write(output / "failure.json", {"schema": SCHEMA, "status": "failed", "error": repr(error),
            "started_at": started, "wall_seconds": time.monotonic() - clock})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable).resolve())
    parser.add_argument("--lean", type=Path)
    parser.add_argument("--cold-request", type=Path)
    args = parser.parse_args()
    if args.cold_request is not None:
        print(json.dumps(_cold_validate(args.cold_request), sort_keys=True))
    else:
        if args.output is None or args.lean is None:
            parser.error("--output and --lean are required for a new retained experiment")
        result = run_finite_service_experiment(output=args.output, python_executable=args.python,
                                             lean_executable=args.lean, parent=args.parent)
        print(json.dumps({key: result[key] for key in (
            "status", "output", "wall_seconds", "complete_family_counts")}, sort_keys=True))
