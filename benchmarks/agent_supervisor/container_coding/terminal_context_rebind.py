"""Reuse verified public source indexes and capture the actual new native owner.

This is a warm-context launch qualification helper, not a cold benchmark arm.
It neither rewrites canonical tasks nor grants scheduling/completion authority.
The sealed world observation can become historical after claim admission.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from . import terminal_indexed_preparation as prep
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.code_retrieval_context import load_code_retrieval_context
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import load_semantic_worker_context
from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import (
    load_task_context_nomination, write_task_context_bundle,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import QuackStateServer
from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import IpfsDatasetsSemanticStateProvider
from ipfs_accelerate_py.agent_supervisor.semantic_state.intent_world_snapshot import (
    capture_intent_world_snapshot, load_intent_world_context, persist_intent_world_snapshot,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


def rebind_full_context(*, prepared_state: Path, admission: dict,
                        server: QuackStateServer, output: Path) -> dict:
    """Recapture current native task state while reusing source-verified indexes.

    The native owner must be started and idle before supervisor launch. Caller
    supplies its actual server, not an alternate connection to its database.
    No model calls or whole-source re-embedding occur; this is explicitly warm
    reuse and must not be counted as a cold benchmark context initialization.
    """
    started = time.monotonic()
    if type(server) is not QuackStateServer or server.identity is None:
        raise ValueError("actual live native owner required")
    state = Path(prepared_state).resolve(strict=True)
    prepared = prep._load_prepared(state)
    verified = local.verify_local_benchmark_admission(admission)
    if admission["manifest"] != prepared["manifest"] or len(verified["graph"].tasks) != 1:
        raise ValueError("rebind requires this exact independently prepared task")
    root = Path(prepared["repository"])
    output = Path(output).absolute()
    if output.exists() or output.resolve() != output or not output.is_relative_to(root / ".runtime"):
        raise ValueError("rebind output must be new and inside repository runtime artifacts")
    task_spec = verified["graph"].tasks[0]
    original = json.loads((state / "context-result.json").read_text())
    if original.get("task_cid") != task_spec.task_cid:
        raise ValueError("original context belongs to another canonical task")
    bundle = original["context_bundle"]
    metadata = load_task_context_nomination(repository=root, artifact=bundle["artifact"],
        expected_sha256=bundle["sha256"], task_cid=task_spec.task_cid, task_id=task_spec.task_key)

    def source_contexts():
        semantic_text = load_semantic_worker_context(repository=root,
            artifact=metadata["semantic context artifact"],
            expected_sha256=metadata["semantic context sha256"], task_id=task_spec.task_key)
        retrieval_text = load_code_retrieval_context(repository=root,
            artifact=metadata["code retrieval artifact"],
            expected_sha256=metadata["code retrieval sha256"], task_id=task_spec.task_key)
        semantic, retrieval = json.loads(semantic_text), json.loads(retrieval_text)
        if (retrieval["status"] != "current" or retrieval["query_text"] != prepared["query"]
                or semantic["semantic_root_cid"] != original["semantic_root_cid"]
                or retrieval["index_id"] != original["index_id"]
                or set(semantic["manifest"]) != set(prepared["worker_inputs"])):
            raise ValueError("reused indexes differ from current public input scope/query")
        return semantic, retrieval

    semantic, retrieval = source_contexts()
    blocks = (root / metadata["semantic context artifact"]).parent / "blocks"
    def get_block(cid):
        if not isinstance(cid, str) or not cid or Path(cid).name != cid or cid in {".", ".."}:
            raise ValueError("invalid semantic block identity")
        path = blocks / cid
        if path.is_symlink() or path.resolve() != path:
            raise ValueError("semantic block path must be canonical")
        return path.read_bytes()
    view = IpfsDatasetsSemanticStateProvider().open_verified_view(semantic["semantic_root_cid"], get_block)
    identity = server.identity
    with server._lock:
        if server.identity != identity or server.ready().get("ready") is not True:
            raise ValueError("native owner changed or is not ready")
        connection = server._connection
        connection.execute("BEGIN TRANSACTION")
        try:
            intent = IntentRepository(bound_connection=connection, install_schema=False)
            task = intent.get_task(task_spec.task_cid)
            if task is None or task["status"] != "ready":
                raise ValueError("launch rebind requires the exact unclaimed native task")
            contract, _, _, _ = local._contract(task["body"], task_spec.task_cid)
            if (contract["graph_cid"] != verified["graph"].content_id
                    or contract["manifest"] != admission["manifest"]):
                raise ValueError("native task belongs to a different admission")
            capture = capture_intent_world_snapshot(intent, repository_id=view.root.repository_id,
                task_cids=[task_spec.task_cid], semantic_root_cid=semantic["semantic_root_cid"],
                get_semantic_block=get_block, transaction_owned_by_caller=True)
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    world = persist_intent_world_snapshot(capture, output=output / "world", task_id=task_spec.task_key)
    if source_contexts() != (semantic, retrieval):
        raise ValueError("source context changed during native world capture")
    with server._lock:
        if server.identity != identity:
            raise ValueError("native owner changed during capture persistence")
        intent = IntentRepository(bound_connection=server._connection, install_schema=False)
        load_intent_world_context(artifact=Path(world["artifact"]),
            expected_sha256=world["artifact_sha256"], task_id=task_spec.task_key,
            repository_id=view.root.repository_id, intent=intent)
    metadata.update({"world context artifact": Path(world["artifact"]).relative_to(root).as_posix(),
        "world context sha256": world["artifact_sha256"],
        "world context repository": view.root.repository_id})
    result = {"schema": "supervisor-task-context-preparation@1",
        "task_cid": task_spec.task_cid, "task_id": task_spec.task_key,
        "task_revision": task["revision"], "metadata": metadata,
        "semantic_root_cid": semantic["semantic_root_cid"],
        "world_snapshot_cid": capture["snapshot"]["snapshot_cid"],
        "plan_projection_cid": capture["plan_projection"]["projection_cid"],
        "event_watermark": capture["planning_context"]["event_watermark"],
        "index_id": retrieval["index_id"], "query_id": retrieval["query_id"],
        "retrieval_result_id": retrieval["result_id"],
        "source_sha256": retrieval["source_sha256"],
        "public_query_sha256": hashlib.sha256(prepared["query"].encode()).hexdigest(),
        "local_contract_cid": local.content_identity(task["body"][local.CONTRACT_KEY]),
        "preparation_mode": "verified-warm-source-reuse-and-new-owner-world-capture",
        "intent_fresh_at_capture": True, "intent_fresh_after_claim": False,
        "world_after_claim": "sealed historical observation; no live intent authority",
        "new_embedding_calls": 0, "text_generation_calls": 0,
        "execution_authority": False, "completion_authority": False,
        "canonical_task_mutated": False, "seconds": time.monotonic() - started}
    result["context_bundle"] = write_task_context_bundle(repository=root, prepared=[result],
        output=output / "context-bundle.json")
    prep._write(output / "result.json", result)
    return result


def verify_worker_context_prompt(*, prompt: str, rebound: dict) -> dict:
    """Check actual worker stdin evidence identities without granting authority."""
    wire, _ = json.JSONDecoder().raw_decode(prompt)
    contexts = {}
    for kind in ("semantic-context", "code-retrieval-context", "intent-world-context"):
        refs = sorted((row for row in wire["evidence"] if row["kind"] == kind),
            key=lambda row: row["reference_id"])
        if not refs or not all(row["metadata"]["required"] is True for row in refs):
            raise ValueError("native worker prompt omitted a required context: " + kind)
        contexts[kind] = json.loads("".join(row["summary"] for row in refs))
    semantic, retrieval, world = (contexts[kind] for kind in (
        "semantic-context", "code-retrieval-context", "intent-world-context"))
    tasks = [task for task in world.get("tasks", []) if task.get("task_cid") == rebound["task_cid"]]
    if (semantic["task_id"] != rebound["task_id"]
            or wire.get("objective_id") != rebound["task_id"]
            or len(tasks) != 1
            or tasks[0].get("body", {}).get("local_planning_contract_cid") != rebound["local_contract_cid"]
            or semantic["semantic_root_cid"] != rebound["semantic_root_cid"]
            or retrieval["status"] != "current" or retrieval["index_id"] != rebound["index_id"]
            or retrieval["result_id"] != rebound["retrieval_result_id"]
            or retrieval["source_sha256"] != rebound["source_sha256"]
            or hashlib.sha256(retrieval["query_text"].encode()).hexdigest() != rebound["public_query_sha256"]
            or world["world_snapshot_cid"] != rebound["world_snapshot_cid"]
            or world["plan_projection_cid"] != rebound["plan_projection_cid"]
            or world["semantic_root_cid"] != rebound["semantic_root_cid"]
            or world.get("completion_authority") is not False
            or world.get("execution_authority") is not False):
        raise ValueError("native worker context identity/status differs from nomination")
    return {"schema": "terminal-rebound-worker-context-observation@1",
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_bytes": len(prompt.encode()), "required_context_kinds": sorted(contexts),
        "semantic_root_cid": rebound["semantic_root_cid"], "index_id": rebound["index_id"],
        "world_snapshot_cid": rebound["world_snapshot_cid"],
        "task_cid": rebound["task_cid"], "local_contract_cid": rebound["local_contract_cid"],
        "source_sha256": retrieval["source_sha256"],
        "public_query_sha256": rebound["public_query_sha256"],
        "intent_freshness_checked_by_worker": world.get("intent_freshness_checked") is True,
        "world_scope": "sealed capture; task claim can supersede its intent revision",
        "task_completed": False, "completion_authority": False, "text_generation_calls": 0}
