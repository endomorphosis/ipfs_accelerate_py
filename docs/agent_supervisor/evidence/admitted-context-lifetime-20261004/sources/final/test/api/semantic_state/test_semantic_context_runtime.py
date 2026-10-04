"""Live producer-to-supervisor context integration (optional datasets checkout)."""

import json
import subprocess

import pytest

pytest.importorskip("ipfs_datasets_py.logic.software_contracts.semantic_state")

from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import (
    prepare_semantic_context,
    load_semantic_worker_context,
)


def test_live_capsules_reach_native_daemon_and_stale_sources_are_refused(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalTask,
        TodoImplementationDaemon,
    )
    from ipfs_accelerate_py.agent_supervisor.context.context_compiler import render_context_capsule

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "target.py").write_text(
        "from dependency import add\ndef calculate(a,b):\n    return add(a,b)\n"
    )
    (repo / "dependency.py").write_text("def add(a: int,b: int)->int:\n    return a+b\n")
    (repo / "test_target.py").write_text(
        "from target import calculate\ndef test_add():\n    assert calculate(1,2)==3\n"
    )
    (repo / "tasks.todo.md").write_text("# Local integration\n")
    for args in [
        ("init", "-q"),
        ("add", "."),
        (
            "-c",
            "user.name=Integration",
            "-c",
            "user.email=local@example.invalid",
            "commit",
            "-qm",
            "seed",
        ),
    ]:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    task_id = "CTX-001"
    result = prepare_semantic_context(
        repository=repo,
        paths=["target.py", "dependency.py", "test_target.py"],
        required_raw_paths=["target.py", "test_target.py"],
        objective="Preserve addition",
        task_id=task_id,
        output=tmp_path / "context",
    )
    artifact = repo / "semantic-context.json"
    artifact.write_bytes((tmp_path / "context/worker-context.json").read_bytes())
    payload = json.loads(artifact.read_bytes())
    assert (tmp_path / "context/blocks" / result["scope_cid"]).is_file()
    for path, binding in payload["manifest"].items():
        assert (tmp_path / "context/blocks" / binding["source_cid"]).read_bytes() == (
            repo / path
        ).read_bytes()
    assert result["capsules"] == 6
    assert result["ducklake"]["status"] == "projected"
    assert payload["raw_sources"]["target.py"] == (repo / "target.py").read_text()
    assert any(a["admission"] == "exact_substitute" for a in payload["admissions"])
    assert result["compact_bytes"] < result["pretty_json_bytes"]
    daemon = TodoImplementationDaemon(
        todo_path=repo / "tasks.todo.md",
        state_path=tmp_path / "state/tasks.json",
        strategy_path=tmp_path / "state/strategy.json",
        events_path=tmp_path / "state/events.jsonl",
        repo_root=repo,
        task_header_prefix="## CTX-",
    )
    task = PortalTask(
        task_id=task_id,
        title="Preserve addition",
        status="ready",
        completion="manual",
        priority="P0",
        track="context",
        outputs=["target.py"],
        validation=["python3 -m pytest test_target.py"],
        acceptance="Addition is preserved",
        metadata={
            "Provider role": "deterministic-only",
            "Semantic context artifact": "semantic-context.json",
            "Semantic context sha256": result["worker_payload_sha256"],
        },
    )
    compiled = daemon._compile_implementation_context(task, attempt=1)
    prompt = render_context_capsule(compiled.capsule)
    assert result["semantic_root_cid"] in prompt
    assert "supervisor-semantic-worker-context@1" in prompt
    args = dict(
        repository=repo,
        artifact="semantic-context.json",
        expected_sha256=result["worker_payload_sha256"],
        task_id=task_id,
    )
    with pytest.raises(ValueError, match="task or schema"):
        load_semantic_worker_context(**{**args, "task_id": "OTHER"})
    with pytest.raises(ValueError, match="digest"):
        load_semantic_worker_context(**{**args, "expected_sha256": "0" * 64})
    (repo / "dependency.py").write_text("def add(a,b): return a-b\n")
    with pytest.raises(ValueError, match="stale"):
        daemon._compile_implementation_context(task, attempt=1)
    task.metadata["Semantic context refresh"] = "true"
    refreshed = daemon._compile_implementation_context(task, attempt=2)
    refreshed_prompt = render_context_capsule(refreshed.capsule)
    refreshed_paths = list((repo / ".runtime/semantic-refresh").glob("*/worker-context.json"))
    assert len(refreshed_paths) == 1
    fresh_payload = json.loads(refreshed_paths[0].read_bytes())
    assert fresh_payload["semantic_root_cid"] != result["semantic_root_cid"]
    assert fresh_payload["semantic_root_cid"] in refreshed_prompt
    assert fresh_payload["refresh_lineage"]["attempt_id"] == "CTX-001:2"
    assert (
        fresh_payload["refresh_lineage"]["previous_payload_sha256"]
        == result["worker_payload_sha256"]
    )
    assert artifact.read_bytes() == (tmp_path / "context/worker-context.json").read_bytes()


@pytest.mark.parametrize("path", ["../escape.py", "/tmp/escape.py", "a/../b.py"])
def test_input_scope_cannot_escape_repository(tmp_path, path):
    with pytest.raises(ValueError, match="canonical"):
        prepare_semantic_context(
            repository=tmp_path,
            paths=[path],
            required_raw_paths=[path],
            objective="Task",
            task_id="CTX-001",
            output=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()
