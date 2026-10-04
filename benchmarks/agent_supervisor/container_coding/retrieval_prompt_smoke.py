"""Hydrate a qualified learned index and consume it in a real native prompt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def qualify(source: Path, qualification: Path, output: Path) -> dict:
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import (
        CodeVectorIndexSnapshot, CodeVectorSearchResult,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.code_retrieval_context import prepare_code_retrieval_context
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTask, TodoImplementationDaemon

    source = source.resolve(strict=True)
    qualification = qualification.resolve(strict=True)
    if output.exists():
        raise ValueError("output must be a new directory")
    raw_result = (qualification / "result.json").read_bytes()
    prior = json.loads(raw_result)
    if (prior.get("schema") != "native-local-learned-vector-qualification@1"
            or prior.get("status") != "qualified" or prior.get("learned_embeddings") is not True
            or prior.get("canary", {}).get("disposition") != "passed"):
        raise ValueError("an actual learned-index qualification is required")
    sources = {}
    for name, digest in prior["source_sha256"].items():
        relative = Path(name)
        path = source / relative
        if (relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name
                or path.is_symlink() or not path.resolve().is_relative_to(source)):
            raise ValueError("qualified input escapes source root")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("qualified source changed")
        sources[name] = raw
    started = time.monotonic()
    with duckdb.connect(str(qualification / "vectors.duckdb"), read_only=True, config={"threads": 1}) as connection:
        row = connection.execute("SELECT payload FROM snapshots WHERE id=?", [prior["index_id"]]).fetchone()
        snapshot = CodeVectorIndexSnapshot.from_dict(json.loads(row[0]))
    search = CodeVectorSearchResult.from_dict(prior["hits"])
    output.mkdir(parents=True)
    repository = output / "repository"
    repository.mkdir()
    for name, raw in sources.items():
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (repository / "tasks.todo.md").write_text("# Native learned retrieval qualification\n")
    for args in [("init", "-q"), ("add", "."),
                 ("-c", "user.name=Qualification", "-c", "user.email=local@example.invalid", "commit", "-qm", "permitted inputs")]:
        subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True)
    metadata = prepare_code_retrieval_context(
        repository=repository, task_id="VECTOR-001", query_text=prior["query"], snapshot=snapshot,
        result=search, output=repository / ".runtime/code-retrieval.json",
    )
    daemon = TodoImplementationDaemon(
        todo_path=repository / "tasks.todo.md", state_path=output / "state/tasks.json",
        strategy_path=output / "state/strategy.json", events_path=output / "state/events.jsonl",
        repo_root=repository, task_header_prefix="## VECTOR-",
    )
    task = PortalTask(task_id="VECTOR-001", title="Inspect " + prior["query"], status="ready",
                      completion="manual", priority="P0", track="retrieval",
                      outputs=list(sources), validation=[], acceptance="Review the nominated source",
                      metadata=metadata["metadata"])
    try:
        prompt = daemon._build_implementation_prompt(task, attempt=1)
        capsule = daemon._last_implementation_context.capsule
        refs = [item for item in capsule.evidence if item.kind == "code-retrieval-context"]
        context = json.loads("".join(item.summary for item in sorted(refs, key=lambda item: item.reference_id)))
        if (not refs or not all(item.required for item in refs) or context["status"] != "current"
                or context["result_id"] != search.result_id or len(context["hits"]) != len(search.hits)
                or snapshot.index_id not in prompt):
            raise RuntimeError("native prompt did not consume exact current retrieval")
        result = {
            "schema": "native-learned-retrieval-prompt-qualification@1", "status": "qualified",
            "qualified_index_result_sha256": hashlib.sha256(raw_result).hexdigest(),
            "index_id": snapshot.index_id, "query_id": search.query.query_id, "result_id": search.result_id,
            "context_artifact_sha256": metadata["sha256"], "source_sha256": prior["source_sha256"],
            "native_worker_prompt_consumed": True, "nominated_symbols": [hit["symbol"] for hit in context["hits"]],
            "prompt_bytes": len(prompt.encode()), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "seconds": time.monotonic() - started, "text_generation_calls": 0, "new_embedding_calls": 0,
            "provider_tokens": None, "token_savings": None, "task_completion_authorized": False,
            "full_supervisor_started": False, "terminal_bench_result": False,
        }
        (output / "worker-prompt.json").write_text(prompt)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        daemon.close_event_runtime()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(qualify(args.source, args.qualification, args.output), indent=2))
