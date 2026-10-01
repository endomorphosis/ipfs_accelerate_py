"""Actual native-owner context recapture preserves historical world semantics."""
import json
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.terminal_native_preflight import _authored_graph
from benchmarks.agent_supervisor.container_coding.terminal_context_rebind import rebind_full_context, verify_worker_context_prompt
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original as original
from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTask, TodoImplementationDaemon


def test_rebind_actual_owner_and_native_prompt_checks_fresh_sources_without_world_authority(original):
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    graph = _authored_graph(prepared)
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=prepared["manifest"])
    prep._write(state / "admission.json", admission)
    with IntentRepository(state / "intent.duckdb") as intent:
        local.materialize_local_benchmark_plan(admission=admission, intent=intent)
    prep.context(state=state)
    clone = state / "clone.duckdb"
    with IntentRepository(clone) as intent:
        result = local.materialize_local_benchmark_plan(admission=admission, intent=intent)
    task_spec = graph.tasks[0]
    with open_existing_native_owner(database=clone, checkout=root, state_dir=state / "owner",
            repository_id=prepared["manifest"]["payload"]["repository_cid"],
            execution_routes={task_spec.task_key: GROK_CODEX_EXECUTION_MODE}) as owner:
        before = owner.source.get_task(task_spec.task_cid)
        rebound = rebind_full_context(prepared_state=state, admission=admission,
            server=owner.server, output=root / ".runtime/rebound")
        assert owner.source.get_task(task_spec.task_cid) == before
        assert rebound["new_embedding_calls"] == rebound["text_generation_calls"] == 0
        assert rebound["intent_fresh_at_capture"] is True
        assert rebound["intent_fresh_after_claim"] is False
        portal = PortalTask(task_id=task_spec.task_key, canonical_task_cid=task_spec.task_cid,
            title=before.body["title"], status="ready", completion="manual", priority="P2",
            track="implementation", outputs=["bottle.py", "report.jsonl"],
            validation=[], acceptance="Public shape and syntax", metadata={"database task cid": task_spec.task_cid})
        daemon = TodoImplementationDaemon(todo_path=root / ".runtime/test.todo.md",
            state_path=state / "prompt/tasks.json", strategy_path=state / "prompt/strategy.json",
            events_path=state / "prompt/events.jsonl", repo_root=root, task_header_prefix="## TB-")
        daemon._task_context_nomination_bundle = rebound["context_bundle"]
        try:
            # Same production behavior: sealed world evidence without claiming
            # to read a live owner from this worker-facing Portal instance.
            prompt = daemon._build_implementation_prompt(portal, attempt=1)
            observed = verify_worker_context_prompt(prompt=prompt, rebound=rebound)
            assert observed["intent_freshness_checked_by_worker"] is False
            assert observed["completion_authority"] is False
            assert len(observed["required_context_kinds"]) == 3
            assert observed["source_sha256"] == {"bottle.py": prepared["manifest"]["payload"]["sources"]["bottle.py"]["sha256"]}
            foreign_query = {**rebound, "public_query_sha256": "0" * 64}
            with pytest.raises(ValueError, match="identity/status"):
                verify_worker_context_prompt(prompt=prompt, rebound=foreign_query)
            # Genuine task-state advancement makes the earlier world capture
            # historical; it cannot turn into fresh owner state by reuse.
            owner.source.compare_and_set_status(before.task_cid, before.revision, "retrying")
            with pytest.raises(ValueError, match="unclaimed native task"):
                rebind_full_context(prepared_state=state, admission=admission,
                    server=owner.server, output=root / ".runtime/rebound-rejected")
            foreign = {**rebound, "world_snapshot_cid": "foreign"}
            with pytest.raises(ValueError, match="identity/status"):
                verify_worker_context_prompt(prompt=prompt, rebound=foreign)
            # Byte changes invalidate reused semantic nominations before any
            # new context bundle is published.
            (root / "bottle.py").write_text("def application():\n    return 'changed'\n")
            with pytest.raises(ValueError):
                rebind_full_context(prepared_state=state, admission=admission,
                    server=owner.server, output=root / ".runtime/rebound-stale")
            assert not (root / ".runtime/rebound-stale/context-bundle.json").exists()
        finally:
            daemon.close_event_runtime()
