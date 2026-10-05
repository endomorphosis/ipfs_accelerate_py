"""Retained native semantic @1 assets remain usable with generic profiles.

Only the planner proposal and CLI version are authored fixtures. Semantic and
vector production, admission, native owner capture and worker verification run
their actual implementations.
"""
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
from benchmarks.agent_supervisor.container_coding import learned_vector_preflight, vector_index_preflight
from benchmarks.agent_supervisor.container_coding.test_terminal_task_profile import original, prepare, git
from benchmarks.agent_supervisor.container_coding.test_terminal_empty_context import authored_proposal
from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _version
from benchmarks.agent_supervisor.container_coding.terminal_context_rebind import (
    rebind_full_context, verify_worker_context_prompt,
)
from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import supervised_task_context
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTask, TodoImplementationDaemon


def _retained_native_v1(monkeypatch):
    native = initial.prepare_semantic_context

    def produce(**kwargs):
        # Emulate the retained producer contract, not fabricated @1 output:
        # the real native helper derives facts from the complete manifest.
        kwargs.pop("program_paths", None)
        result = native(**kwargs)
        assert result["schema"] == "supervisor-semantic-context-preparation@1"
        assert "program_paths" not in result
        return result

    monkeypatch.setattr(initial, "prepare_semantic_context", produce)


def test_retained_generic_v1_assets_reuse_through_native_owner_and_worker(tmp_path, monkeypatch):
    args = original(tmp_path)
    root, _, state, _ = args
    prepared = prepare(args)
    _retained_native_v1(monkeypatch)
    monkeypatch.setattr(learned_vector_preflight, "qualify",
        lambda *a, **k: pytest.fail("retained lexical assets consumed an embedding model"))
    receipt = prep.initial_context(state=state)
    loaded = initial.load_initial_context(state=state, prepared=prepared, require_empty_owner=True)
    assert loaded["semantic"]["schema"] == "supervisor-semantic-worker-context@1"
    assert "program_paths" not in loaded["semantic"]
    assert set(loaded["semantic"]["manifest"]) == set(prepared["worker_inputs"])
    assert loaded["retrieval"]["status"] == "current"
    assert receipt["index_id"] is not None and receipt["indexed_symbols"] == 1
    assert receipt["learned_embeddings"] is False
    assert set(loaded["retrieval"]["source_sha256"]) == {"source.py"}
    paths = [root / loaded["descriptor"]["metadata"]["Semantic context artifact"],
        root / ".runtime/terminal-vectors/result.json", root / ".runtime/terminal-vectors/vectors.duckdb"]
    retained_bytes = {path: path.read_bytes() for path in paths}
    calls = []

    def router(prompt, **kwargs):
        calls.append(prompt)
        assert receipt["semantic_root_cid"] in prompt and receipt["index_id"] in prompt
        return {"text": authored_proposal(prepared), "observation": {}, "execution_receipt": None}

    _version(monkeypatch)
    planned = prep.plan(state=state, provider_callable=router)
    assert planned["qualified"], planned
    assert len(calls) == 1

    def cold_build(*args, **kwargs):
        pytest.fail("warm context reuse rebuilt retained native assets")

    monkeypatch.setattr(initial, "prepare_semantic_context", cold_build)
    monkeypatch.setattr(supervised_task_context, "prepare_semantic_context", cold_build)
    monkeypatch.setattr(vector_index_preflight, "qualify", cold_build)
    context = prep.context(state=state)
    assert context["initial_indexes_reused"] is True
    assert context["new_embedding_calls"] == 0 and context["learned_embeddings"] is False
    assert context["semantic_root_cid"] == receipt["semantic_root_cid"]
    assert context["index_id"] == receipt["index_id"]
    assert {path: path.read_bytes() for path in paths} == retained_bytes
    admission = json.loads((state / "admission.json").read_bytes())
    task = local.verify_local_benchmark_admission(admission)["graph"].tasks[0]
    clone = state / "live-owner.duckdb"
    with IntentRepository(clone) as intent:
        local.materialize_local_benchmark_plan(admission=admission, intent=intent)
    with open_existing_native_owner(database=clone, checkout=root, state_dir=state / "owner",
            repository_id=prepared["manifest"]["payload"]["repository_cid"],
            execution_routes={task.task_key: GROK_CODEX_EXECUTION_MODE}) as owner:
        before = owner.source.get_task(task.task_cid)
        rebound = rebind_full_context(prepared_state=state, admission=admission,
            server=owner.server, output=root / ".runtime/rebound")
        assert owner.source.get_task(task.task_cid) == before
        assert rebound["index_id"] == receipt["index_id"]
        assert rebound["semantic_root_cid"] == receipt["semantic_root_cid"]
        assert rebound["new_embedding_calls"] == rebound["text_generation_calls"] == 0
        assert {path: path.read_bytes() for path in paths} == retained_bytes
        portal = PortalTask(task_id=task.task_key, canonical_task_cid=task.task_cid,
            title=before.body["title"], status="ready", completion="manual", priority="P2",
            track="implementation", outputs=["result.py"], validation=[], acceptance="Public shape and syntax",
            metadata={"database task cid": task.task_cid})
        daemon = TodoImplementationDaemon(todo_path=root / ".runtime/test.todo.md",
            state_path=state / "prompt/tasks.json", strategy_path=state / "prompt/strategy.json",
            events_path=state / "prompt/events.jsonl", repo_root=root, task_header_prefix="## TB-")
        daemon._task_context_nomination_bundle = rebound["context_bundle"]
        try:
            prompt = daemon._build_implementation_prompt(portal, attempt=1)
            observed = verify_worker_context_prompt(prompt=prompt, rebound=rebound)
            assert observed["index_id"] == receipt["index_id"]
            assert observed["source_sha256"] == loaded["retrieval"]["source_sha256"]
            assert observed["intent_freshness_checked_by_worker"] is False
            assert observed["completion_authority"] is False
        finally:
            daemon.close_event_runtime()
    assert not (root / "result.py").exists()


@pytest.mark.parametrize("comment_only", [False, True])
def test_null_index_empty_lane_cannot_reuse_unscoped_legacy_v1(tmp_path, monkeypatch, comment_only):
    args = original(tmp_path, empty=not comment_only)
    root, _, state, _ = args
    if comment_only:
        (root / "source.py").write_text("# A program input without qualified declarations.\n")
        git(root, "add", "source.py")
        git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "commit", "-qm", "comment-only program baseline")
    prepare(args)
    _retained_native_v1(monkeypatch)
    monkeypatch.setattr(vector_index_preflight, "qualify",
        lambda *a, **k: pytest.fail("empty scope manufactured a vector index"))
    monkeypatch.setattr(learned_vector_preflight, "qualify",
        lambda *a, **k: pytest.fail("empty scope consumed an embedding model"))
    with pytest.raises(ValueError, match="empty program population"):
        prep.initial_context(state=state)
    assert not (state / "initial-context-result.json").exists()
    assert not (state / "planner-invoked.json").exists()
    assert not (state / "admission.json").exists()
    assert not (root / "result.py").exists()
