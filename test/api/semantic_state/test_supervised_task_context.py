"""One real task preparation joins native intent, source capsules and world."""

import json
from dataclasses import replace
import subprocess

import pytest

pytest.importorskip("ipfs_datasets_py.logic.software_contracts.semantic_state")

from ipfs_accelerate_py.agent_supervisor.runtime import supervised_task_context as runtime
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
)


@pytest.fixture
def inputs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "target.py").write_text("from dependency import add\ndef target(x): return add(x,1)\n")
    (repo / "dependency.py").write_text("def add(a,b): return a+b\n")
    (repo / "tasks.todo.md").write_text("# Supervised task context\n")
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
    intent = IntentRepository(tmp_path / "intent.duckdb")
    intent.upsert_objective(
        objective_id="objective:add", objective_alias="OBJ-1", title="Preserve addition"
    )
    intent.upsert_goal(
        goal_cid="goal:add",
        goal_alias="GOAL-1",
        title="Preserve addition",
        objective_id="objective:add",
    )
    intent.upsert_plan(
        plan_cid="plan:add", plan_alias="PLAN-1", goal_cid="goal:add", status="active"
    )
    intent.upsert_task(
        task_cid="task:add",
        task_alias="CONTEXT-001",
        goal_cid="goal:add",
        plan_cid="plan:add",
        objective_id="objective:add",
        body={"title": "Preserve addition"},
        outputs=[{"path": "target.py", "effect": "modify"}],
        acceptance=[{"criterion": "Addition holds", "evidence_kind": "validation"}],
        validations=[["python3", "-m", "pytest", "test_target.py"]],
    )
    return dict(
        repository=repo,
        intent=intent,
        task_cid="task:add",
        paths=["target.py", "dependency.py"],
        required_raw_paths=["target.py"],
        output=repo / ".runtime/prepared",
    )


def daemon_and_task(arguments, prepared):
    repo = arguments["repository"]
    state = repo.parent / "state"
    daemon = TodoImplementationDaemon(
        repo_root=repo,
        todo_path=repo / "tasks.todo.md",
        state_path=state / "state.json",
        strategy_path=state / "strategy.json",
        events_path=state / "events.jsonl",
        implementation_log_dir=state / "logs",
        task_header_prefix="## CONTEXT-",
        merge_queue_dir=state / "merge-queue",
        worktree_root=state / "worktrees",
        validation_cache_dir=state / "validation-cache",
    )
    daemon._world_intent_repository = arguments["intent"]
    task = PortalTask(
        task_id=prepared["task_id"],
        title=prepared["task_title"],
        status="ready",
        completion="manual",
        priority="P1",
        track="context",
        outputs=["target.py"],
        validation=["python3 -m pytest test_target.py"],
        acceptance="Addition holds",
        metadata=prepared["metadata"],
    )
    return daemon, task


def context_from_prompt(prompt, kind):
    wire, _ = json.JSONDecoder().raw_decode(prompt)
    refs = sorted(
        (row for row in wire["evidence"] if row["kind"] == kind),
        key=lambda row: row["reference_id"],
    )
    assert refs and all(row["metadata"]["required"] for row in refs)
    return json.loads("".join(row["summary"] for row in refs))


def test_launch_bundle_feeds_context_without_self_referential_native_task_mutation(inputs):
    from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import (
        write_task_context_bundle, load_task_context_nomination,
    )

    before = inputs["intent"].plan_projection()
    prepared = runtime.prepare_supervised_task_context(**inputs)
    bundle = write_task_context_bundle(repository=inputs["repository"], prepared=[prepared],
                                      output=inputs["repository"] / ".runtime/launch-context.json")
    daemon, task = daemon_and_task(inputs, prepared)
    task = replace(task, metadata={}, canonical_task_cid=inputs["task_cid"])
    daemon._task_context_nomination_bundle = bundle
    try:
        prompt = daemon._build_implementation_prompt(task, attempt=1)
        world = context_from_prompt(prompt, "intent-world-context")
        assert world["semantic_root_cid"] == prepared["semantic_root_cid"]
        assert task.metadata == {} and inputs["intent"].plan_projection() == before
        with pytest.raises(ValueError, match="foreign task"):
            load_task_context_nomination(repository=inputs["repository"], artifact=bundle["artifact"],
                                         expected_sha256=bundle["sha256"], task_cid="foreign", task_id=task.task_id)
        (inputs["repository"] / bundle["artifact"]).write_text("{}")
        with pytest.raises(ValueError, match="digest"):
            daemon._build_implementation_prompt(task, attempt=2)
    finally:
        daemon.close_event_runtime()


def test_prepare_task_context_feeds_matching_native_semantic_and_world_roots(inputs):
    intent = inputs["intent"]
    before = intent.plan_projection()
    watermark = intent.event_watermark()
    prepared = runtime.prepare_supervised_task_context(**inputs)
    json.dumps(prepared)
    assert intent.plan_projection() == before
    assert intent.event_watermark() == watermark
    assert prepared["semantic"]["ducklake"]["status"] == "projected"
    assert prepared["world"]["metadata"]["status"] == "projected"
    assert prepared["canonical_task_mutated"] is False
    daemon, task = daemon_and_task(inputs, prepared)
    prompt = daemon._build_implementation_prompt(task, attempt=1)
    semantic = context_from_prompt(prompt, "semantic-context")
    world = context_from_prompt(prompt, "intent-world-context")
    assert (
        world["semantic_root_cid"] == semantic["semantic_root_cid"] == prepared["semantic_root_cid"]
    )
    assert world["world_snapshot_cid"] == prepared["world_snapshot_cid"]
    assert world["tasks"][0]["task_cid"] == prepared["task_cid"]
    assert semantic["task_id"] == prepared["task_id"]
    assert world["semantic_coherence"]["status"] == "current"
    assert world["execution_authority"] is False
    assert world["completion_authority"] is False


def test_native_retrieval_joins_the_same_task_capsules_and_world(inputs):
    from ipfs_accelerate_py.agent_supervisor.analysis.program_ast_adapters import build_program_evidence_index
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import build_code_symbol_vector_index

    source = {name: (inputs["repository"] / name).read_text() for name in inputs["paths"]}
    ast = build_program_evidence_index(source).ast_index
    index = build_code_symbol_vector_index(
        ast, forest_id="forest:explicit", tree_id="tree:explicit", coverage_id=ast.index_id,
        dimensions=2, model_id="test-vectors", model_revision="1", configuration_id="test:2d",
        vectors=lambda row: (1.0, 0.0) if row.symbol == "add" else (0.0, 1.0),
    )
    hits = index.search((1.0, 0.0), max_results=1)
    prepared = runtime.prepare_supervised_task_context(
        **inputs, code_vector_snapshot=index, code_vector_result=hits, code_query_text="addition",
    )
    daemon, task = daemon_and_task(inputs, prepared)
    prompt = daemon._build_implementation_prompt(task, attempt=1)
    retrieval = context_from_prompt(prompt, "code-retrieval-context")
    semantic = context_from_prompt(prompt, "semantic-context")
    world = context_from_prompt(prompt, "intent-world-context")
    assert retrieval["task_id"] == semantic["task_id"] == prepared["task_id"]
    assert retrieval["result_id"] == hits.result_id
    assert retrieval["hits"][0]["symbol"] == "dependency.add"
    assert retrieval["status"] == "current" and retrieval["completion_authority"] is False
    assert world["semantic_root_cid"] == semantic["semantic_root_cid"]
    daemon.close_event_runtime()


def test_refresh_marks_captured_world_semantics_stale_and_rejects_nomination_tampering(inputs):
    prepared = runtime.prepare_supervised_task_context(**inputs)
    daemon, task = daemon_and_task(inputs, prepared)
    daemon._build_implementation_prompt(task, attempt=1)
    daemon.record_implementation_failure_context(
        task, {"kind": "validation_failure", "returncode": 1}, changed_files=["dependency.py"]
    )
    (inputs["repository"] / "dependency.py").write_text("def add(a,b): return a-b\n")
    prompt = daemon._build_implementation_prompt(task, attempt=2)
    semantic = context_from_prompt(prompt, "semantic-context")
    world = context_from_prompt(prompt, "intent-world-context")
    assert world["schema"] == "supervisor-intent-world-dispatch-observation@1"
    assert world["semantic_coherence"]["status"] == "stale"
    assert (
        world["semantic_coherence"]["dispatch_semantic_root_cid"] == semantic["semantic_root_cid"]
    )
    assert (
        world["semantic_coherence"]["captured_semantic_root_cid"] == prepared["semantic_root_cid"]
    )
    assert world["component_status"]["capsule_index"] == "stale"
    assert world["captured_component_status"]["capsule_index"] == "current"
    assert world["schedulable"] is False
    assert world["execution_authority"] is False
    assert world["completion_authority"] is False
    original = inputs["repository"] / prepared["metadata"]["Semantic context artifact"]
    original.write_bytes(original.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        daemon._build_implementation_prompt(task, attempt=3)


@pytest.mark.parametrize("changed_owner", ["source", "intent"])
def test_preparation_refuses_owner_drift_before_publishing_nominations(
    inputs, monkeypatch, changed_owner
):
    original = runtime.persist_intent_world_snapshot

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if changed_owner == "source":
            (inputs["repository"] / "dependency.py").write_text("def add(a,b): return a-b\n")
        else:
            inputs["intent"].upsert_task(
                task_cid="task:add",
                task_alias="CONTEXT-001",
                goal_cid="goal:add",
                status="in_progress",
                body={"title": "Preserve addition"},
            )
        return result

    monkeypatch.setattr(runtime, "persist_intent_world_snapshot", changed)
    with pytest.raises(ValueError, match="stale|changed"):
        runtime.prepare_supervised_task_context(**inputs)
    assert not (inputs["output"] / "result.json").exists()


def test_preparation_refuses_output_outside_repository(inputs):
    inputs["output"] = inputs["repository"].parent / "outside"
    with pytest.raises(ValueError, match="repository-contained"):
        runtime.prepare_supervised_task_context(**inputs)
    assert not inputs["output"].exists()


def test_worker_world_projection_omits_internal_contract_without_mutating_capture():
    from ipfs_accelerate_py.agent_supervisor.semantic_state.intent_world_snapshot import minify_intent_world_worker_context
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
    original = {
        'tasks': [{'task_cid': 'task-1', 'body': {
            'title': 'Repair public behavior',
            'local_planning_contract': {'manifest': {'profile_dir': '/private/profile'}, 'signature': 'signed'},
        }, 'validations': [{'argv': ['python3', '-B', 'test_answer.py']}]}],
        'execution_authority': False, 'completion_authority': False,
        'plan_projection_cid': 'captured-plan', 'world_snapshot_cid': 'captured-world',
    }
    before = json.dumps(original, sort_keys=True)
    projected = minify_intent_world_worker_context(original)
    assert json.dumps(original, sort_keys=True) == before
    assert '/private/profile' not in json.dumps(projected)
    assert projected['tasks'][0]['validations'] == original['tasks'][0]['validations']
    assert projected['tasks'][0]['body']['local_planning_contract_cid'] == content_identity(original['tasks'][0]['body']['local_planning_contract'])
    assert projected['worker_projection']['source_context_cid'] == content_identity(original)
    assert projected['worker_projection']['full_canonical_projection'] is False
    assert projected['execution_authority'] is False and projected['completion_authority'] is False
    assert projected['plan_projection_cid'] == original['plan_projection_cid']


def test_optional_security_source_program_runs_in_existing_task_context(inputs, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import security_source_program_advisor_384 as advisor
    calls = []
    selected = {"explicit": "local-config"}
    before = inputs["intent"].plan_projection()

    def prepare(**options):
        calls.append(options)
        return {"status": "source_candidate_advice", "proof_authority": False,
                "completion_authority": False, "continue_planning": True}

    monkeypatch.setattr(advisor, "prepare_repository_source_program_advice", prepare)
    prepared = runtime.prepare_supervised_task_context(**inputs, security_source_program_config=selected)
    assert calls == [{"repository": inputs["repository"].resolve(), "paths": inputs["paths"],
                      "config": selected, "output": inputs["output"] / "security-source-program-advice.json"}]
    assert prepared["security_source_program_advice"]["status"] == "source_candidate_advice"
    assert prepared["canonical_task_mutated"] is False and inputs["intent"].plan_projection() == before
    assert "Security" not in " ".join(prepared["metadata"])


def test_optional_security_missing_checkpoint_does_not_block_task_context(inputs):
    from ipfs_accelerate_py.agent_supervisor.runtime.security_source_program_advisor_384 import CONFIG_SCHEMA
    selected = {"schema": CONFIG_SCHEMA, "checkpoint_path": str(inputs["repository"].parent / "absent-checkpoint.json"),
        "checkpoint_sha256": "a" * 64, "decoder": "structured", "embedding_snapshot_path": None, "lake": None}
    prepared = runtime.prepare_supervised_task_context(**inputs, security_source_program_config=selected)
    advice = prepared["security_source_program_advice"]
    assert advice["status"] == "fail_open_unavailable" and advice["failure_stage"] == "checkpoint_loading"
    assert advice["continue_planning"] is True and advice["proof_authority"] is False
    assert prepared["canonical_task_mutated"] is False
    assert (inputs["output"] / "security-source-program-advice.json").is_file()
