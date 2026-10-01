"""Join local pending admission, indexed context, actual Doctor and completion.

This bounded, authored qualification has no language-model planning or coding
calls. It exercises a reviewed keyword repair, not arbitrary synthesis or a
Terminal-Bench trial. The learned mode uses a pinned local embedding model.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _context(prompt: str, kind: str) -> dict:
    wire, _ = json.JSONDecoder().raw_decode(prompt)
    rows = sorted((row for row in wire["evidence"] if row["kind"] == kind),
                  key=lambda row: row["reference_id"])
    if not rows or not all(row["metadata"]["required"] for row in rows):
        raise RuntimeError("required native prompt evidence missing: " + kind)
    return json.loads("".join(row["summary"] for row in rows))


def qualify(output: Path, *, model_snapshot: Path | None = None,
            model_revision: str = "") -> dict:
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import (
        CodeVectorIndexSnapshot, CodeVectorSearchResult,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import run_local_task_validations
    from ipfs_accelerate_py.agent_supervisor.runtime.supervised_task_context import prepare_supervised_task_context
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository, IntentCompletionError
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTask, TodoImplementationDaemon
    from benchmarks.agent_supervisor.container_coding import doctor_composition_smoke as doctor
    from benchmarks.agent_supervisor.container_coding.local_planning_qualification import prepare_local_task

    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("qualification needs a new directory without symlink ancestors")
    if bool(model_snapshot) != bool(model_revision):
        raise ValueError("learned mode needs the exact model snapshot and revision")
    output.mkdir(parents=True)
    started = time.monotonic()
    runtime, inputs = doctor.prepare(output / "doctor")
    repository = runtime.checkout_root
    with (repository / ".git/info/exclude").open("a") as stream:
        stream.write("\n.runtime/\n")
    objective = "Repair the direct local caller keyword so pkg.use(7) returns 7"
    validation = [sys.executable, "-B", "-c", "import pkg; assert pkg.use(7) == 7"]

    def index_and_context(intent, cid, stage):
        if model_snapshot:
            from benchmarks.agent_supervisor.container_coding.learned_vector_preflight import qualify as build
            indexed = build(repository, output / ("vectors-" + stage), ["pkg.py"],
                            "process use", model_snapshot, model_revision)
        else:
            from benchmarks.agent_supervisor.container_coding.vector_index_preflight import qualify as build
            indexed = build(repository, output / ("vectors-" + stage), ["pkg.py"], "process use")
        with duckdb.connect(str(output / ("vectors-" + stage) / "vectors.duckdb"),
                            read_only=True, config={"threads": 1}) as connection:
            row = connection.execute("SELECT payload FROM snapshots WHERE id=?", [indexed["index_id"]]).fetchone()
            snapshot = CodeVectorIndexSnapshot.from_dict(json.loads(row[0]))
        return prepare_supervised_task_context(
            repository=repository, intent=intent, task_cid=cid,
            paths=["pkg.py"], required_raw_paths=["pkg.py"],
            output=repository / ".runtime" / stage, code_vector_snapshot=snapshot,
            code_vector_result=CodeVectorSearchResult.from_dict(indexed["hits"]),
            code_query_text=indexed["query"],
        )

    with IntentRepository(output / "intent.duckdb") as intent:
        prepared = prepare_local_task(
            repository=repository, state=output / "local-policy", intent=intent,
            scope_paths=["pkg.py"], output_path="pkg.py", validation_argv=validation,
            objective=objective,
        )
        cid = prepared["task_cid"]
        task = intent.get_task(cid)
        intent.cas_task_status(task_cid=cid, expected_revision=task["revision"], new_status="in_progress")
        failed = run_local_task_validations(intent=intent, task_cid=cid, attempt_id="indexed-doctor:before")
        if failed["passed"]:
            raise RuntimeError("authored repair did not exhibit its expected public failure")
        try:
            intent.cas_task_status(task_cid=cid, expected_revision=intent.get_task(cid)["revision"],
                                   new_status="completed", allow_completion_without_evidence=True)
        except IntentCompletionError:
            blocked_before = True
        else:
            raise RuntimeError("pending acceptance was bypassed")
        before = index_and_context(intent, cid, "before")
        daemon = TodoImplementationDaemon(
            todo_path=repository / ".runtime/tasks.todo.md", state_path=output / "worker/tasks.json",
            strategy_path=output / "worker/strategy.json", events_path=output / "worker/events.jsonl",
            repo_root=repository, task_header_prefix="## LOCAL-",
        )
        daemon._world_intent_repository = intent
        portal = PortalTask(task_id=before["task_id"], title=objective, status="ready", completion="manual",
                            priority="P0", track="indexed-doctor", outputs=["pkg.py"],
                            validation=[], acceptance="The signed public check passes",
                            metadata=before["metadata"])
        try:
            prompt = daemon._build_implementation_prompt(portal, attempt=1)
            retrieval = _context(prompt, "code-retrieval-context")
            world = _context(prompt, "intent-world-context")
            if (retrieval["status"] != "current" or not retrieval["hits"]
                    or world["semantic_root_cid"] != before["semantic_root_cid"]):
                raise RuntimeError("initial native prompt did not join current indexes and world")
            (output / "initial-prompt.json").write_text(prompt)
            doctor.run_prepared(runtime, inputs, output / "doctor")
            transaction = runtime.composition_transaction
            committed = transaction.merge_cas.desired_ref
            # This disposable task retains its signed baseline HEAD. Materialize
            # exactly the native transaction's committed output into its worktree.
            subprocess.run(["git", "-C", str(repository), "restore", "--source=" + committed,
                            "--worktree", "--", "pkg.py"], check=True)
            expected = subprocess.check_output(["git", "-C", str(repository), "show", committed + ":pkg.py"])
            if (repository / "pkg.py").read_bytes() != expected:
                raise RuntimeError("worktree differs from the verified published candidate")
            passed = run_local_task_validations(intent=intent, task_cid=cid, attempt_id="indexed-doctor:after")
            if not passed["passed"] or passed["source_tree_id"] == failed["source_tree_id"]:
                raise RuntimeError("actual repaired acceptance did not pass on changed source")
            after = index_and_context(intent, cid, "after")
            portal = replace(portal, metadata=after["metadata"])
            fresh_prompt = daemon._build_implementation_prompt(portal, attempt=2)
            refreshed = _context(fresh_prompt, "code-retrieval-context")
            fresh_world = _context(fresh_prompt, "intent-world-context")
            if (refreshed["status"] != "current" or refreshed["index_id"] == retrieval["index_id"]
                    or after["semantic_root_cid"] == before["semantic_root_cid"]
                    or fresh_world["semantic_root_cid"] != after["semantic_root_cid"]):
                raise RuntimeError("post-repair context did not refresh matching native roots")
            (output / "refreshed-prompt.json").write_text(fresh_prompt)
            intent.cas_task_status(task_cid=cid, expected_revision=intent.get_task(cid)["revision"],
                                   new_status="completed",
                                   evidence_digests=[row["evidence_digest"] for row in passed["results"]])
            completed = intent.get_task(cid)
            result = {
                "schema": "indexed-doctor-local-lifecycle-qualification@1", "status": "qualified",
                "task_cid": cid, "task_status": completed["status"], "task_revision": completed["revision"],
                "pending_acceptance_enforced": blocked_before, "before_validation": failed,
                "after_validation": passed, "before_context": before, "after_context": after,
                "native_prompt_context_consumed": True, "native_doctor_transaction_committed": True,
                "doctor_commit": committed, "source_after_sha256": hashlib.sha256(expected).hexdigest(),
                "actual_native_completion": completed["status"] == "completed",
                "learned_embeddings": bool(model_snapshot),
                "index_result_paths": ["vectors-before/result.json", "vectors-after/result.json"],
                "text_generation_calls": 0, "provider_tokens": None, "token_savings": None,
                "full_supervisor_started": False, "quack_owner_used": False,
                "planning_source": "independently authored qualification task graph",
                "proof_scope": inputs.proof_scope, "terminal_bench_result": False,
                "seconds": time.monotonic() - started,
            }
            _json(output / "result.json", result)
            return result
        finally:
            daemon.close_event_runtime()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--model-revision", default="")
    args = parser.parse_args()
    result = qualify(args.output, model_snapshot=args.model_snapshot, model_revision=args.model_revision)
    print(json.dumps({key: result[key] for key in ("status", "task_status", "seconds", "learned_embeddings")}))
