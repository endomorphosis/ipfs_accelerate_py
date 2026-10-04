"""Native retrieval must reach prompts without stale or fabricated hits."""
from dataclasses import replace
import hashlib
import json
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.program_ast_adapters import build_program_evidence_index
from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import build_code_symbol_vector_index
from ipfs_accelerate_py.agent_supervisor.runtime.code_retrieval_context import (
    prepare_code_retrieval_context, load_code_retrieval_context,
)


def prepare(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = "def calculate(x):\n    return x + 1\n\ndef unrelated():\n    return 0\n"
    (repo / "worker.py").write_text(source)
    (repo / "tasks.todo.md").write_text("# Retrieval qualification\n")
    for args in [("init", "-q"), ("add", "."),
                 ("-c", "user.name=Qualification", "-c", "user.email=local@example.invalid", "commit", "-qm", "seed")]:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    ast = build_program_evidence_index({"worker.py": source}).ast_index
    index = build_code_symbol_vector_index(
        ast, forest_id="forest:explicit-input", tree_id="tree:explicit-input", coverage_id=ast.index_id,
        dimensions=2, model_id="test-vectors", model_revision="1", configuration_id="test:2d",
        vectors=lambda row: (1.0, 0.0) if row.symbol == "calculate" else (0.0, 1.0),
    )
    hits = index.search((1.0, 0.0), max_results=1)
    result = prepare_code_retrieval_context(repository=repo, task_id="RET-001", query_text="calculate",
                                            snapshot=index, result=hits, output=repo / "retrieval.json")
    args = dict(repository=repo, task_id="RET-001", artifact="retrieval.json", expected_sha256=result["sha256"])
    return repo, index, hits, result, args


def test_real_index_nominations_reach_native_prompt_and_disappear_when_source_changes(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTask, TodoImplementationDaemon

    repo, index, _, result, args = prepare(tmp_path)
    context = json.loads(load_code_retrieval_context(**args))
    assert context["status"] == "current" and context["hits"][0]["symbol"] == "worker.calculate"
    assert context["nomination_only"] is True and context["semantic_authority"] is False
    daemon = TodoImplementationDaemon(todo_path=repo / "tasks.todo.md", state_path=tmp_path / "state/tasks.json",
                                     strategy_path=tmp_path / "state/strategy.json", events_path=tmp_path / "state/events.jsonl",
                                     repo_root=repo, task_header_prefix="## RET-")
    task = PortalTask(task_id="RET-001", title="Repair calculate", status="ready", completion="manual", priority="P0",
                      track="retrieval", outputs=["worker.py"], validation=["python3 -m py_compile worker.py"],
                      acceptance="Calculation works", metadata=result["metadata"])
    prompt = daemon._build_implementation_prompt(task, attempt=1)
    compiled = daemon._last_implementation_context
    assert index.index_id in prompt and "worker.calculate" in prompt
    assert any(item.kind == "code-retrieval-context" and item.required for item in compiled.capsule.evidence)
    failure = daemon.record_implementation_failure_context(
        task, {"kind": "validation_failure", "returncode": 1},
        changed_files=("worker.py",), unresolved_requirements=("calculation",),
    )
    (repo / "worker.py").write_text("def calculate(x):\n    return x - 1\n")
    stale = json.loads(load_code_retrieval_context(**args))
    assert stale["status"] == "stale" and stale["hits"] == [] and stale["stale_paths"] == ["worker.py"]
    prompt = daemon._build_implementation_prompt(task, attempt=2)
    retried = daemon._last_implementation_context.capsule
    retrieved = json.loads("".join(item.summary for item in retried.evidence if item.kind == "code-retrieval-context"))
    assert retrieved["status"] == "stale" and "worker.calculate" not in prompt
    assert failure.failure_id in prompt
    (repo / "worker.py").unlink()
    assert json.loads(load_code_retrieval_context(**args))["status"] == "stale"
    daemon.close_event_runtime()


def test_foreign_task_and_modified_artifact_are_rejected(tmp_path):
    repo, _, _, _, args = prepare(tmp_path)
    with pytest.raises(ValueError, match="task or schema"):
        load_code_retrieval_context(**{**args, "task_id": "FOREIGN"})
    (repo / "retrieval.json").write_text("{}")
    with pytest.raises(ValueError, match="digest"):
        load_code_retrieval_context(**args)


def test_rehashed_forged_hit_does_not_replay(tmp_path):
    repo, index, hits, _, args = prepare(tmp_path)
    forged = replace(hits, hits=(replace(hits.hits[0], score=0.5),))
    payload = json.loads((repo / "retrieval.json").read_text())
    payload["result"] = forged.to_dict()
    raw = json.dumps(payload).encode()
    (repo / "retrieval.json").write_bytes(raw)
    with pytest.raises(ValueError, match="does not replay"):
        load_code_retrieval_context(**{**args, "expected_sha256": hashlib.sha256(raw).hexdigest()})
    missing = replace(index, rows=index.rows[:1])
    with pytest.raises(ValueError, match="omits or duplicates"):
        prepare_code_retrieval_context(repository=repo, task_id="RET-002", query_text="calculate",
                                       snapshot=missing, result=missing.search((1.0, 0.0), max_results=1),
                                       output=repo / "missing.json")


def test_large_index_uses_exact_bounded_persistence_reference(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import code_retrieval_context as runtime

    repo, index, hits, _, _ = prepare(tmp_path)
    payload = json.loads((repo / "retrieval.json").read_bytes())
    # Exercise the real spill boundary without constructing a huge test index.
    limit = len(runtime._json(payload).encode()) - len(runtime._json(payload["snapshot"]).encode()) // 2
    monkeypatch.setattr(runtime, "MAX_ARTIFACT_BYTES", limit)
    prepared = prepare_code_retrieval_context(
        repository=repo, task_id="RET-002", query_text="calculate", snapshot=index, result=hits,
        output=repo / "bounded.json",
    )
    args = dict(repository=repo, task_id="RET-002", artifact="bounded.json", expected_sha256=prepared["sha256"])
    referenced = json.loads((repo / "bounded.json").read_bytes())
    assert referenced["schema"] == runtime.REFERENCED_SCHEMA and "snapshot" not in referenced
    assert (repo / "bounded.json").stat().st_size <= limit
    context = json.loads(load_code_retrieval_context(**args))
    assert context["index_id"] == index.index_id and context["hits"][0]["symbol"] == "worker.calculate"
    snapshot_path = repo / referenced["snapshot_ref"]["path"]
    snapshot_raw = snapshot_path.read_bytes()
    snapshot_path.write_bytes(b" " + snapshot_raw[1:])
    with pytest.raises(ValueError, match="snapshot digest"):
        load_code_retrieval_context(**args)
    snapshot_path.write_bytes(snapshot_raw)
    referenced["snapshot_ref"]["path"] = "../outside.json"
    raw = runtime._json(referenced).encode()
    (repo / "bounded.json").write_bytes(raw)
    with pytest.raises(ValueError, match="escapes"):
        load_code_retrieval_context(**{**args, "expected_sha256": hashlib.sha256(raw).hexdigest()})
