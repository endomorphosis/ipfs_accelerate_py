"""Real native STOP precedes automatic completed-context reconstruction."""
from __future__ import annotations

import json
import pytest

from ipfs_accelerate_py.agent_supervisor.control import profile_authority
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_local_completion_bridge import complete, native_published_transition, typed_claim
from test.api.semantic_state.test_published_task_context import initial_bundle


@pytest.fixture(autouse=True)
def isolated_lifecycle_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_authority, "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE", tmp_path / "account")


def lexical_vectors(repository):
    import duckdb
    from benchmarks.agent_supervisor.container_coding.vector_index_preflight import qualify
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import (
        CodeVectorIndexSnapshot, CodeVectorSearchResult,
    )
    output = repository.parent / "initial-lexical-index"
    result = qualify(repository, output, ["answer.py"], "answer")
    with duckdb.connect(str(output / "vectors.duckdb"), read_only=True, config={"threads": 1}) as connection:
        snapshot = CodeVectorIndexSnapshot.from_dict(json.loads(connection.execute(
            "SELECT payload FROM snapshots WHERE id=?", [result["index_id"]]).fetchone()[0]))
    return snapshot, CodeVectorSearchResult.from_dict(result["hits"])


@pytest.mark.parametrize("lexical", [False, True])
def test_opted_in_runtime_refreshes_only_after_native_stop(scenario, tmp_path, lexical):
    admission = local.admit_local_benchmark_plan(graph=scenario["graph"], manifest=scenario["manifest"])
    with typed_claim(scenario, tmp_path) as (owner, _daemon, attempt):
        bundle = initial_bundle(scenario, owner, **({"vector_builder": lexical_vectors} if lexical else {}))
        original_bundle = (scenario["repository"] / bundle["artifact"]).read_bytes()
        runtime = AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission=admission,
            server=owner.server, source=owner.source, implement=False, timeout_ms=30_000,
            context_bundle=bundle, refresh_context_on_completion=True,
            published_retrieval_policy="lexical-tfidf-symbols@1" if lexical else None)
        try:
            assert runtime.manifest["context_refresh_policy"]["trigger"] == "after_native_stop"
            assert runtime.start().succeeded
            assert runtime.observe()["published_context"][0]["status"] == "pending_completion"
            task = owner.source.get_task(attempt.task_cid)
            transition = native_published_transition(scenario, tmp_path, owner, attempt)
            passed = run_owner_local_task_validations(server=owner.server, task_cid=task.task_cid,
                attempt_id=attempt.attempt_id, expected_revision=task.revision, source_transition=transition)
            assert passed["passed"] is True
            complete(owner, attempt, passed["results"][0]["evidence_digest"])
            before = owner.source.get_task(task.task_cid)
            # Completion alone cannot start expensive derivative work while
            # an actual admitted worker process tree remains active.
            assert len(runtime.process.snapshot(runtime.profile).members) >= 2
            assert runtime.observe()["published_context"][0]["status"] == "pending_native_stop"
            assert runtime._context_refresh_attempts == {}
            with pytest.raises(ValueError, match="native STOP"):
                runtime.refresh_after_stop()
            stopped = runtime.stop()
            assert stopped.succeeded
            assert stopped.data["old_tree_fenced"] is True
            assert not runtime.process.snapshot(runtime.profile).members
            assert runtime._context_refresh_attempts == {}
            refreshed = runtime.observe()["published_context"][0]
            assert refreshed["status"] == "refreshed"
            assert refreshed["task_revision"] == 4
            assert refreshed["completion_authority"] is False
            assert refreshed["retrieval_status"] == ("current" if lexical else "unavailable")
            assert refreshed["context_bundle"] != bundle
            assert runtime.refresh_after_stop()[0] == refreshed
            assert runtime._context_refresh_attempts == {task.task_cid: 1}
            if lexical:
                current = runtime._published_context[task.task_cid]
                policy = runtime.manifest["published_retrieval_policy"]
                assert current["retrieval"]["previous_index_id"] == policy["index_id"]
                assert current["retrieval"]["index_id"] != policy["index_id"]
                assert current["retrieval"]["config_id"] == policy["config_id"]
                assert current["retrieval"]["ducklake"]["status"] == "projected"
            assert owner.source.get_task(task.task_cid) == before
            assert runtime.context_bundle == bundle
            assert (scenario["repository"] / bundle["artifact"]).read_bytes() == original_bundle
            # Cached derivatives are never returned as current after an
            # owner intent event supersedes the captured watermark.
            with owner.server._lock:
                intent = IntentRepository(bound_connection=owner.server._connection, install_schema=False)
                intent.upsert_objective(objective_id="next-objective", objective_alias="NEXT", title="Next task")
            stale = runtime.observe()["published_context"][0]
            assert stale["status"] == "unavailable"
            assert "context_bundle" not in stale
            assert runtime.stop().succeeded
            # Source drift may reject observation itself; shutdown remains
            # governed by the original signed lifecycle grant.
            (scenario["repository"] / "answer.py").write_text("def answer(): return 9\n")
            with pytest.raises(ValueError):
                runtime.observe()
            assert runtime.stop().succeeded
            assert not runtime.process.snapshot(runtime.profile).members
        finally:
            if runtime.process.snapshot(runtime.profile).members:
                assert runtime.stop().succeeded
            runtime.close()
