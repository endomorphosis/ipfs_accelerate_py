"""Live producer-to-supervisor context integration (optional datasets checkout)."""

import hashlib
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


def _program_context(tmp_path, *, source="def target(value): return value + 1\n", program=("target.py",), query=""):
    root = tmp_path / "program-repository"
    root.mkdir()
    (root / "instruction.md").write_text("Inspect the real program without inferring proof authority.\n")
    (root / "smoke.py").write_text("def harness_only_symbol():\n    return unknown_harness_dependency()\n")
    (root / "target.py").write_text(source)
    output = root / ".semantic/initial"
    prepared = prepare_semantic_context(repository=root,
        paths=["instruction.md", "smoke.py", "target.py"], required_raw_paths=["instruction.md"],
        program_paths=program, objective="Inspect the program.", task_id="PROGRAM-001",
        output=output, worker_query=query)
    return root, output, prepared


@pytest.mark.parametrize("source,program,count", [
    ("def target(value): return value + 1\n", (), 0),
    ("# A genuine comment-only module.\n", ("target.py",), 1),
    ("def target(value): return value + 1\n", ("target.py",), 2),
])
def test_explicit_program_scope_keeps_native_facts_and_full_support_binding(tmp_path, source, program, count):
    root, output, prepared = _program_context(tmp_path, source=source, program=program)
    payload = json.loads((output / "worker-context.json").read_bytes())
    assert prepared["schema"] == "supervisor-semantic-context-preparation@2"
    assert payload["schema"] == "supervisor-semantic-worker-context@2"
    assert prepared["program_paths"] == payload["program_paths"] == list(program)
    assert prepared["capsules"] == len(payload["capsules"]) == count
    assert payload["reconstruction"]["semantic_symbol_count"] == count
    assert payload["reconstruction"]["doctor_source_paths"] == list(program)
    assert payload["reconstruction"]["captured_source_count"] == 3
    assert payload["reconstruction"]["support_source_count"] == 3 - len(program)
    assert set(payload["manifest"]) == {"instruction.md", "smoke.py", "target.py"}
    assert payload["raw_sources"]["smoke.py"] == (root / "smoke.py").read_text()
    assert "harness_only_symbol" not in json.dumps(payload["capsules"])
    assert "smoke.py" not in (output / "doctor.json").read_text()
    assert payload["completion_authority"] is False
    assert payload["reconstruction"]["semantic_acceptance_authority"] is False
    assert load_semantic_worker_context(repository=root, artifact=".semantic/initial/worker-context.json",
        expected_sha256=prepared["worker_payload_sha256"], task_id="PROGRAM-001") == (
        output / "worker-context.json").read_text()
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import _scan_scoped_sources
    from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import IpfsDatasetsSemanticStateProvider
    from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload
    state = _scan_scoped_sources({name: (root / name).read_bytes() for name in program},
        repository_id=cid_for_payload({"repository": str(root)}), max_symbols=256)
    producer = IpfsDatasetsSemanticStateProvider()
    assert producer.view_semantic_state_bundle(producer.build_semantic_state(state)).root.root_cid == payload["semantic_root_cid"]


def test_default_program_scope_preserves_legacy_wire_bytes(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "target.py").write_text("def target(): return 1\n")
    args = dict(repository=root, paths=["target.py"], required_raw_paths=["target.py"],
        objective="Inspect target", task_id="LEGACY-001")
    prepare_semantic_context(**args, output=root / ".semantic/default")
    prepare_semantic_context(**args, program_paths=None, output=root / ".semantic/none")
    assert (root / ".semantic/default/worker-context.json").read_bytes() == (
        root / ".semantic/none/worker-context.json").read_bytes()


@pytest.mark.parametrize("program", [["target.py", "target.py"], ["target.py", "smoke.py"],
    ["absent.py"], ["./target.py"], "target.py"])
def test_explicit_program_subset_requires_canonical_population(tmp_path, program):
    with pytest.raises(ValueError, match="program paths"):
        _program_context(tmp_path, program=program)
    assert not (tmp_path / "program-repository/.semantic/initial").exists()


@pytest.mark.parametrize("mutation", ["subset", "extra_path", "drop_capsule", "drop_projected_capsule",
    "support_raw", "downgrade", "untyped_authority", "symbol_count"])
def test_saved_explicit_program_replays_subset_native_capsules_and_support(tmp_path, mutation):
    root, output, _ = _program_context(tmp_path, query="target" if mutation == "drop_projected_capsule" else "")
    artifact = output / "worker-context.json"
    payload = json.loads(artifact.read_bytes())
    if mutation == "subset":
        from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload
        payload["program_paths"] = []
        payload["scope_cid"] = cid_for_payload({"schema": "supervisor-source-scope@2",
            "sources": payload["manifest"], "program_paths": []})
        payload["reconstruction"].update(scope_cid=payload["scope_cid"], program_paths=[],
            doctor_source_paths=[], program_source_count=0, support_source_count=3)
        payload["raw_sources"]["target.py"] = (root / "target.py").read_text()
    elif mutation == "extra_path":
        payload["program_paths"].append("outside.py")
    elif mutation.startswith("drop_"):
        payload["capsules"].pop()
        payload["admissions"].pop()
        if mutation == "drop_projected_capsule":
            projection = payload["worker_projection"]
            projection["selected_symbols"].pop()
            projection["selected_capsule_count"] -= 1
            projection["omitted_capsule_count"] += 1
    elif mutation == "support_raw":
        payload["raw_sources"].pop("smoke.py")
    elif mutation == "downgrade":
        payload["schema"] = "supervisor-semantic-worker-context@1"
    elif mutation == "untyped_authority":
        payload["reconstruction"]["completion_authority"] = 0
    else:
        payload["reconstruction"]["semantic_symbol_count"] += 1
    artifact.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError):
        load_semantic_worker_context(repository=root, artifact=".semantic/initial/worker-context.json",
            expected_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(), task_id="PROGRAM-001")


@pytest.mark.parametrize("name", ["target.py", "smoke.py"])
def test_program_and_support_drift_both_refuse_saved_dispatch(tmp_path, name):
    root, _, prepared = _program_context(tmp_path)
    (root / name).write_text((root / name).read_text() + "# changed source\n")
    with pytest.raises(ValueError, match="stale"):
        load_semantic_worker_context(repository=root, artifact=".semantic/initial/worker-context.json",
            expected_sha256=prepared["worker_payload_sha256"], task_id="PROGRAM-001")
