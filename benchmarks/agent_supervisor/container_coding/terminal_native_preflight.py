"""Provider-free original-task admission, complete indexing and prompt preflight.

The graph is explicitly authored from the independently declared public task.
This checks native storage/context bounds; it is not model planning, coding,
supervisor execution, or a Terminal-Bench result. No verifier input is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import time

from . import terminal_indexed_preparation as prep
from .local_live_planner import preflight_proposal


def _authored_graph(prepared):
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
        _select_evidence, parse_prompt_goal_graph,
    )
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
        PromptWorkflowRequest, DirectoryScanReceipt,
    )
    request = PromptWorkflowRequest.from_dict(prepared["request"])
    scan = DirectoryScanReceipt.from_dict(prepared["scan"])
    config = prep._config(Path(prepared["repository"]))
    authored = preflight_proposal({"spec": prepared["spec"],
        "evidence": _select_evidence(request, scan, config)})
    authored["root_goal_key"] = "TB-GOAL"
    authored["goals"][0].update(goal_key="TB-GOAL", title="Public task",
        objective="Fulfill the public instruction")
    authored["goals"][1].update(goal_key="TB-SUBGOAL", parent_goal_key="TB-GOAL",
        title="Repair", objective="Fulfill the public instruction")
    authored["tasks"][0].update(task_key="TB-CODE-TASK", goal_key="TB-SUBGOAL",
        objective="Fulfill the public instruction", predicted_files=["bottle.py", "report.jsonl"])
    return parse_prompt_goal_graph(json.dumps(authored), request, scan, config=config,
        constraint_summaries=prepared["constraints"])


def qualify(*, repository: Path, instruction: Path, state: Path,
            model_snapshot: Path | None = None, model_revision: str = "") -> dict:
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        TodoImplementationDaemon, PortalTask,
    )
    from .indexed_doctor_lifecycle import _context

    started = time.monotonic()
    prepared = prep.prepare(repository=repository, instruction=instruction, state=state)
    repository, state = Path(prepared["repository"]), Path(prepared["state"])
    graph = _authored_graph(prepared)
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=prepared["manifest"])
    prep._write(state / "admission.json", admission)
    with IntentRepository(state / "intent.duckdb") as intent:
        materialized = local.materialize_local_benchmark_plan(admission=admission, intent=intent)
        plan_body = intent.get_plan(materialized["plan_id"])["body"]
        reference = plan_body["local_planning_receipt_ref"]
        if local.load_local_planning_receipt(reference, manifest=prepared["manifest"]) != admission["receipt"]:
            raise RuntimeError("native plan lost its complete signed receipt")
    phases = {"prepare_admit_materialize_seconds": time.monotonic() - started}
    indexed = prep.context(state=state, model_snapshot=model_snapshot, model_revision=model_revision)
    phases["context_seconds"] = indexed["seconds"]
    prompt_started = time.monotonic()
    with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
        task = intent.get_task(materialized["task_cids"][0])
        spec = prepared["spec"]
        # Read-only projection of the actual pending task. No fabricated claim,
        # attempt, dispatch capability or proof receipt is created by preflight.
        portal = PortalTask(task_id=task["task_alias"], title=task["body"]["title"],
            canonical_task_cid=task["task_cid"], status=task["status"],
            completion="manual", priority="P2", track="implementation",
            outputs=[row["path"] for row in spec["outputs"]],
            validation=[shlex.join(row["argv"]) for row in spec["validations"]],
            acceptance=" ; ".join(row["criterion"] for row in spec["acceptance"]),
            metadata={"database task cid": task["task_cid"],
                "local planning contract cid": local.content_identity(task["body"][local.CONTRACT_KEY])})
        daemon = TodoImplementationDaemon(todo_path=repository / ".runtime/preflight-tasks.todo.md",
            state_path=state / "worker/tasks.json", strategy_path=state / "worker/strategy.json",
            events_path=state / "worker/events.jsonl", repo_root=repository, task_header_prefix="## TB-")
        daemon._world_intent_repository = intent
        daemon._task_context_nomination_bundle = indexed["context_bundle"]
        try:
            prompt = daemon._build_implementation_prompt(portal, attempt=1)
            retrieval = _context(prompt, "code-retrieval-context")
            world = _context(prompt, "intent-world-context")
            if (retrieval["status"] != "current" or retrieval["index_id"] != indexed["index_id"]
                    or world["semantic_root_cid"] != indexed["semantic_root_cid"]
                    or indexed["semantic_root_cid"] not in prompt):
                raise RuntimeError("native prompt did not consume the exact current contexts")
            (state / "worker-prompt.txt").write_text(prompt)
        finally:
            daemon.close_event_runtime()
    phases["native_prompt_seconds"] = time.monotonic() - prompt_started
    result = {
        "schema": "terminal-native-provider-free-preflight@1", "qualified": True,
        "graph_provenance": "explicitly authored public task; no model proposal",
        "text_generation_calls": 0, "official_verifier_executed": False,
        "benchmark_success": None, "supervisor_execution_qualified": False,
        "completion_authority": False, "source_count": len(prepared["manifest"]["payload"]["sources"]),
        "bottle_sha256": prepared["manifest"]["payload"]["sources"]["bottle.py"]["sha256"],
        "query_sha256": hashlib.sha256(prepared["query"].encode()).hexdigest(),
        "query_is_exact_public_instruction": prepared["query"] == instruction.read_text(),
        "signed_receipt_bytes": len(local._receipt_bytes(admission["receipt"])),
        "native_plan_body_bytes": len(local._receipt_bytes(plan_body)),
        "native_task_body_bytes": len(local._receipt_bytes(task["body"])),
        "native_prompt_bytes": len(prompt.encode()),
        "native_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "context": indexed, "phases": phases, "seconds": time.monotonic() - started,
    }
    prep._write(state / "native-preflight-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--instruction", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--model-revision", default="")
    print(json.dumps(qualify(**vars(parser.parse_args())), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
