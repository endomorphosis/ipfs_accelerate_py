"""Native empty-scope preparation, admission, owner, worker and historical audit.

Planner output and router receipts are authored fixtures. Source observations,
semantic state, admission, storage, worker compilation and audit are real.
"""
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from benchmarks.agent_supervisor.container_coding.test_terminal_task_profile import original, prepare, git
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import _proposal_json
from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _version
from benchmarks.agent_supervisor.container_coding.terminal_context_rebind import rebind_full_context, verify_worker_context_prompt
from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTask, TodoImplementationDaemon


def authored_proposal(prepared):
    proposal = json.loads(_proposal_json(prepared))
    proposal["tasks"][0]["predicted_files"] = [item["path"] for item in prepared["spec"]["outputs"]]
    return json.dumps(proposal)


def scenario(tmp_path, comment_only):
    args = original(tmp_path, empty=not comment_only)
    if comment_only:
        (args[0] / "source.py").write_text("# Native Python module, without qualified code symbols.\n")
        git(args[0], "add", "source.py")
        git(args[0], "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "commit", "-qm", "comment-only program baseline")
    return args, prepare(args)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize("comment_only", [False, True])
@pytest.mark.parametrize("initial_reuse", [False, True])
def test_native_empty_context_reaches_planner_owner_worker_and_historical_audit(
        tmp_path, monkeypatch, comment_only, initial_reuse):
    args, prepared = scenario(tmp_path, comment_only)
    root, _, state, _ = args
    from benchmarks.agent_supervisor.container_coding import learned_vector_preflight, vector_index_preflight
    monkeypatch.setattr(learned_vector_preflight, "qualify", lambda *a, **k: pytest.fail("empty scope consumed embeddings"))
    monkeypatch.setattr(vector_index_preflight, "qualify", lambda *a, **k: pytest.fail("empty scope manufactured lexical vectors"))
    # Explicitly authored selection tests the unconsumed policy branch. It is
    # not a trained model or an assertion that its weights were verified.
    selected_model = tmp_path / "authored-unused-model-selection"
    selected_model.mkdir()
    model = {"model_snapshot": selected_model, "model_revision": "authored-unused-revision"}
    before = None
    if initial_reuse:
        before = prep.initial_context(state=state, **model)
        observed = initial.load_initial_context(state=state, prepared=prepared, require_empty_owner=True)
        assert observed["retrieval"]["hits"] == []
        assert observed["retrieval"]["support_hashes"] == {
            name: {"role": role, "sha256": prepared["manifest"]["payload"]["sources"][name]["sha256"]}
            for name, role in ((prep.INSTRUCTION, "instruction"),
                (".supervisor-task-profile.json", "task_profile"), (prep.SMOKE, "structural_smoke"))}
    calls = []
    def router(prompt, **kwargs):
        calls.append(prompt)
        if before is not None:
            assert before["source_population_cid"] in prompt
            assert before["disposition"] in prompt
        return {"text": authored_proposal(prepared), "observation": {}, "execution_receipt": None}
    _version(monkeypatch)
    planned = prep.plan(state, provider_callable=router)
    assert planned["qualified"], planned
    assert len(calls) == 1
    context = prep.context(state=state, **model)
    assert context["initial_indexes_reused"] is initial_reuse
    assert context["index_id"] is None
    assert context["indexed_symbols"] == context["new_embedding_calls"] == context["embedding_calls"] == 0
    assert context["learned_embeddings"] is False
    assert context["disposition"] == ("zero_qualified_symbols" if comment_only else "no_program_inputs")
    assert context["full_capsules"] == int(comment_only)
    assert not (root / ".runtime/terminal-vectors/vectors.duckdb").exists()
    assert not (root / "result.py").exists()
    if before is not None:
        assert context["source_population_cid"] == before["source_population_cid"]
        assert context["semantic_root_cid"] == before["semantic_root_cid"]
        assert context["world_snapshot_cid"] != before["world_snapshot_cid"]
    admission = json.loads((state / "admission.json").read_text())
    verified = local.verify_local_benchmark_admission(admission)
    task = verified["graph"].tasks[0]
    clone = state / "live-owner.duckdb"
    with IntentRepository(clone) as intent:
        local.materialize_local_benchmark_plan(admission=admission, intent=intent)
    with open_existing_native_owner(database=clone, checkout=root, state_dir=state / "owner",
            repository_id=prepared["manifest"]["payload"]["repository_cid"],
            execution_routes={task.task_key: GROK_CODEX_EXECUTION_MODE}) as owner:
        native_before = owner.source.get_task(task.task_cid)
        rebound = rebind_full_context(prepared_state=state, admission=admission,
            server=owner.server, output=root / ".runtime/rebound")
        assert owner.source.get_task(task.task_cid) == native_before
        assert rebound["source_population_cid"] == context["source_population_cid"]
        assert rebound["new_embedding_calls"] == rebound["text_generation_calls"] == 0
        portal = PortalTask(task_id=task.task_key, canonical_task_cid=task.task_cid,
            title=native_before.body["title"], status="ready", completion="manual", priority="P2",
            track="implementation", outputs=["result.py"], validation=[], acceptance="Public shape and syntax",
            metadata={"database task cid": task.task_cid})
        daemon = TodoImplementationDaemon(todo_path=root / ".runtime/test.todo.md",
            state_path=state / "prompt/tasks.json", strategy_path=state / "prompt/strategy.json",
            events_path=state / "prompt/events.jsonl", implementation_log_dir=state / "launch/implementation-logs",
            repo_root=root, task_header_prefix="## TB-")
        daemon._task_context_nomination_bundle = rebound["context_bundle"]
        try:
            prompt = daemon._build_implementation_prompt(portal, attempt=1)
            daemon._persist_implementation_context_receipt(portal, attempt=1)
            worker = verify_worker_context_prompt(prompt=prompt, rebound=rebound)
            assert worker["index_id"] is None
            assert worker["source_population_cid"] == context["source_population_cid"]
            assert worker["intent_freshness_checked_by_worker"] is False
            for mutation in ("schema", "program_subset", "support_hash", "support_text"):
                wire, end = json.JSONDecoder().raw_decode(prompt)
                refs = sorted((row for row in wire["evidence"] if row["kind"] == "semantic-context"),
                    key=lambda row: row["reference_id"])
                semantic = json.loads("".join(row["summary"] for row in refs))
                if mutation == "schema": semantic["schema"] = "supervisor-semantic-worker-context@1"
                elif mutation == "program_subset": semantic["program_paths"] = [prep.SMOKE]
                elif mutation == "support_hash": semantic["manifest"][prep.SMOKE]["sha256"] = "0" * 64
                else: semantic["raw_sources"][prep.SMOKE] = "changed support text\n"
                refs[0]["summary"] = json.dumps(semantic)
                for row in refs[1:]: row["summary"] = ""
                # Authored altered wire exercises the identity observer; final
                # capsule audit independently checks its original chunk CIDs.
                changed_prompt = json.dumps(wire) + prompt[end:]
                with pytest.raises(ValueError):
                    verify_worker_context_prompt(prompt=changed_prompt, rebound=rebound)
            with pytest.raises(ValueError, match="empty population"):
                verify_worker_context_prompt(prompt=prompt,
                    rebound={**rebound, "source_population_cid": "sha256:" + "0" * 64})
            workspace_root = tmp_path / "removed-worker-workspaces"
            workspace = workspace_root / "authored-worker"
            model_prompt, header = render_model_prompt(prompt=prompt, purpose="coding", workspace=workspace)
            receipt = {"schema": "router-implementation-invocation@1", "phase": "coding",
                "invocation_id": "authored-empty-context-invocation", "purpose": "coding", "workspace": str(workspace),
                "prompt_sha256": sha(prompt), "prompt_bytes": len(prompt.encode()),
                "native_prompt_sha256": sha(prompt), "native_prompt_bytes": len(prompt.encode()),
                "model_prompt_sha256": sha(model_prompt), "model_prompt_bytes": len(model_prompt.encode()),
                "workspace_advisory_sha256": sha(header), "workspace_advisory_bytes": len(header.encode())}
            # This is a historical audit: subsequent source drift does not
            # make the stored prompt falsely claim a fresh live observation.
            (root / prep.INSTRUCTION).write_text("changed after prompt capture\n")
            historical = audit.audit_terminal_context(state=state, receipts=[receipt], workspace_root=workspace_root)
            assert historical["all_observed_coding_inputs_verified"] is True, historical
            assert historical["capsules"][0]["index_id"] is None
            assert historical["capsules"][0]["source_population_cid"] == context["source_population_cid"]
            assert historical["completion_authority"] is historical["task_correctness_established"] is False
            assert not workspace.exists()
            with pytest.raises(ValueError):
                rebind_full_context(prepared_state=state, admission=admission,
                    server=owner.server, output=root / ".runtime/stale-rebound")
            assert not (root / ".runtime/stale-rebound/context-bundle.json").exists()
        finally:
            daemon.close_event_runtime()
