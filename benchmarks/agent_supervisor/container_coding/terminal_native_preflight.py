"""Provider-free original-task admission, complete indexing and prompt preflight.

The graph is explicitly authored from the independently declared public task.
This checks native storage/context bounds; it is not model planning, coding,
supervisor execution, or a Terminal-Bench result. No verifier input is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import time

from . import terminal_indexed_preparation as prep
from .local_live_planner import preflight_proposal
from .terminal_task_profile import MULTITASK_SCHEMA, validate_task_profile


def _authored_graph(prepared):
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
        _select_evidence, parse_prompt_goal_graph,
    )
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
        PromptWorkflowRequest, DirectoryScanReceipt,
    )
    request = PromptWorkflowRequest.from_dict(prepared["request"])
    scan = DirectoryScanReceipt.from_dict(prepared["scan"])
    if "specs" in prepared or (prepared.get("task_profile") or {}).get("schema") == MULTITASK_SCHEMA:
        raise ValueError("multi-task preflight is unsupported; exactly one declared task is required")
    config = prep._config(Path(prepared["repository"]),
        timeout_seconds=prepared.get("planner_timeout_seconds", 90),
        provider_profile=prepared.get("provider_profile"))
    authored = preflight_proposal({"spec": prepared["spec"],
        "evidence": _select_evidence(request, scan, config)})
    authored["root_goal_key"] = "TB-GOAL"
    authored["goals"][0].update(goal_key="TB-GOAL", title="Public task",
        objective="Fulfill the public instruction")
    authored["goals"][1].update(goal_key="TB-SUBGOAL", parent_goal_key="TB-GOAL",
        title="Public task implementation" if prepared.get("task_profile") else "Repair",
        objective="Fulfill the public instruction")
    authored["tasks"][0].update(task_key=prepared["spec"]["task_key"], goal_key="TB-SUBGOAL",
        objective="Fulfill the public instruction",
        predicted_files=[row["path"] for row in prepared["spec"]["outputs"]])
    return parse_prompt_goal_graph(json.dumps(authored), request, scan, config=config,
        constraint_summaries=prepared["constraints"])


def qualify(*, repository: Path, instruction: Path, state: Path,
            model_snapshot: Path | None = None, model_revision: str = "",
            task_profile: dict | None = None, provider_profile: str | None = None,
            resource_profile: str | None = None, source384_config: Path | None = None,
            source384_timeout_seconds: float | None = None,
            intent_checkpoint_descriptor: Path | None = None,
            intent_action_384_config: Path | None = None,
            intent_projection_request: Path | None = None,
            intent_projection_request_sha256: str | None = None,
            disable_intent_autoencoder: bool = False,
            enable_source_unit_autoencoder: bool = False,
            source_unit_security_decoder_descriptor: Path | None = None,
            source_unit_project_logic_families: bool = False,
            source_unit_intent_family_context: Path | None = None,
            source_unit_intent_logic_families: list[str] | None = None,
            intent_requirement_contract: Path | None = None) -> dict:
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        TodoImplementationDaemon, PortalTask,
    )
    from .indexed_doctor_lifecycle import _context
    from .benchmark_resource_profile import execution_budget

    started = time.monotonic()
    # Refuse unsupported declarations before prepare commits public input files
    # or creates signed state. This authored graph does not fabricate a symbolic
    # requirement-to-task coverage witness or discard additional declared tasks.
    if type(task_profile) is dict and task_profile.get("schema") == MULTITASK_SCHEMA:
        raise ValueError("multi-task preflight is unsupported; exactly one declared task is required")
    if intent_requirement_contract is not None:
        raise ValueError("authored preflight does not support Intent requirement-contract planning")
    if bool(model_snapshot) != bool(model_revision):
        raise ValueError("learned index needs both the pinned local snapshot and revision")
    if task_profile is not None:
        task_profile = validate_task_profile(task_profile, instruction=Path(instruction).read_text(encoding="utf-8"))
    budget = execution_budget(resource_profile)
    source384_limit = budget["source384_seconds"]
    if source384_timeout_seconds is not None:
        if (source384_config is None or type(source384_timeout_seconds) not in (int, float)
                or not math.isfinite(source384_timeout_seconds)
                or not 0 < source384_timeout_seconds <= source384_limit):
            raise ValueError("Source384 timeout requires a config and a positive bound within the resource profile")
        source384_limit = source384_timeout_seconds
    prepared = prep.prepare(repository=repository, instruction=instruction, state=state,
        task_profile=task_profile, provider_profile=provider_profile, resource_profile=resource_profile,
        intent_checkpoint_descriptor=intent_checkpoint_descriptor,
        intent_action_384_config=intent_action_384_config,
        intent_projection_request=intent_projection_request,
        intent_projection_request_sha256=intent_projection_request_sha256,
        disable_intent_autoencoder=disable_intent_autoencoder,
        enable_source_unit_autoencoder=enable_source_unit_autoencoder,
        source_unit_security_decoder_descriptor=source_unit_security_decoder_descriptor,
        source_unit_project_logic_families=source_unit_project_logic_families,
        source_unit_intent_family_context=source_unit_intent_family_context,
        source_unit_intent_logic_families=source_unit_intent_logic_families)
    repository, state = Path(prepared["repository"]), Path(prepared["state"])
    initial = None
    initial_seconds = 0.0
    if source384_config is not None:
        entered = time.monotonic()
        initial = prep.initial_context(state=state, model_snapshot=model_snapshot,
            model_revision=model_revision, source384_config=source384_config,
            train_autoencoder=False, source384_timeout_seconds=source384_limit)
        initial_seconds = time.monotonic() - entered
    graph = _authored_graph(prepared)
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=prepared["manifest"])
    prep._write(state / "admission.json", admission)
    with IntentRepository(state / "intent.duckdb") as intent:
        materialized = local.materialize_local_benchmark_plan(admission=admission, intent=intent)
        plan_body = intent.get_plan(materialized["plan_id"])["body"]
        reference = plan_body["local_planning_receipt_ref"]
        if local.load_local_planning_receipt(reference, manifest=prepared["manifest"]) != admission["receipt"]:
            raise RuntimeError("native plan lost its complete signed receipt")
    if len(materialized["task_cids"]) != 1:
        raise RuntimeError("provider-free preflight requires one materialized task")
    phases = {"prepare_admit_materialize_seconds": time.monotonic() - started - initial_seconds}
    if initial is not None:
        phases["initial_context_seconds"] = initial_seconds
    indexed = prep.context(state=state, model_snapshot=model_snapshot, model_revision=model_revision)
    if indexed["task_cid"] != materialized["task_cids"][0]:
        raise RuntimeError("native context does not belong to the exact admitted task")
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
                    or world["world_snapshot_cid"] != indexed["world_snapshot_cid"]
                    or {row["task_cid"] for row in world["tasks"]} != {task["task_cid"]}
                    or indexed["semantic_root_cid"] not in prompt):
                raise RuntimeError("native prompt did not consume the exact current contexts")
            (state / "worker-prompt.txt").write_text(prompt)
        finally:
            daemon.close_event_runtime()
    phases["native_prompt_seconds"] = time.monotonic() - prompt_started
    result = {
        "schema": "terminal-native-provider-free-preflight@1", "qualified": True,
        "graph_provenance": "explicitly authored public task; no model proposal",
        "text_generation_calls": 0, "provider_calls": 0, "official_verifier_executed": False,
        "benchmark_success": None, "supervisor_execution_qualified": False,
        "completion_authority": False, "source_count": len(prepared["manifest"]["payload"]["sources"]),
        "input_sha256": {name: row["sha256"] for name, row in
            prepared["manifest"]["payload"]["sources"].items()},
        "worker_inputs": prepared["worker_inputs"],
        "declared_outputs": prepared["spec"]["outputs"],
        "task_cid": task["task_cid"],
        "provider_profile": prepared.get("provider_profile"),
        "provider": prepared["provider"], "model": prepared["model"],
        "resource_profile": prepared["resource_profile"],
        "source384_enabled": source384_config is not None,
        "intent_preplanning": prepared["intent_preplanning"],
        "source_unit_preplanning": prepared["source_unit_preplanning"],
        "query_sha256": hashlib.sha256(prepared["query"].encode()).hexdigest(),
        "query_is_exact_public_instruction": prepared["query"] == instruction.read_text(),
        "signed_receipt_bytes": len(local._receipt_bytes(admission["receipt"])),
        "native_plan_body_bytes": len(local._receipt_bytes(plan_body)),
        "native_task_body_bytes": len(local._receipt_bytes(task["body"])),
        "native_prompt_bytes": len(prompt.encode()),
        "native_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "context": indexed, "phases": phases, "seconds": time.monotonic() - started,
    }
    if "bottle.py" in result["input_sha256"]:
        result["bottle_sha256"] = result["input_sha256"]["bottle.py"]
    if initial is not None:
        result["initial_context"] = initial
        result["source384_timeout_seconds"] = source384_limit
    prep._write(state / "native-preflight-result.json", result)
    return result


def main():
    from .benchmark_provider_profile import PROVIDER_PROFILES
    from .benchmark_resource_profile import PROFILES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--instruction", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--task-profile", type=Path)
    parser.add_argument("--provider-profile", choices=PROVIDER_PROFILES)
    parser.add_argument("--resource-profile", choices=PROFILES)
    parser.add_argument("--source384-config", type=Path)
    parser.add_argument("--source384-timeout-seconds", type=float)
    parser.add_argument("--intent-checkpoint-descriptor", type=Path)
    parser.add_argument("--intent-action-384-config", type=Path)
    parser.add_argument("--intent-projection-request", type=Path)
    parser.add_argument("--intent-projection-request-sha256")
    parser.add_argument("--disable-intent-autoencoder", action="store_true")
    parser.add_argument("--enable-source-unit-autoencoder", action="store_true")
    parser.add_argument("--source-unit-security-decoder-descriptor", type=Path)
    parser.add_argument("--source-unit-project-logic-families", action="store_true")
    parser.add_argument("--source-unit-intent-family-context", type=Path)
    parser.add_argument("--source-unit-intent-logic-family", action="append", dest="source_unit_intent_logic_families")
    args = vars(parser.parse_args())
    if args["task_profile"] is not None:
        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError(f"duplicate task profile key: {key}")
                value[key] = item
            return value

        with args["task_profile"].open("rb") as stream:
            raw = stream.read(262145)
        if len(raw) > 262144:
            parser.error("task profile exceeds the public declaration byte bound")
        args["task_profile"] = json.loads(raw, object_pairs_hook=unique)
    print(json.dumps(qualify(**args), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
