"""Actual owner completion and fenced STOP precede learned index renewal."""
import json
import pytest

from ipfs_accelerate_py.agent_supervisor.control import profile_authority
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations
from test.api.test_agent_supervisor_local_planning_admission import scenario
from test.api.test_local_completion_bridge import complete, native_published_transition, typed_claim
from test.api.semantic_state.test_published_task_context import initial_bundle
from test.api.semantic_state.test_published_learned_retrieval import actual_learned


@pytest.fixture(autouse=True)
def isolated_lifecycle_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_authority, "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE", tmp_path / "account")


def test_signed_learned_profile_rebuilds_after_actual_publication_and_stop(scenario, tmp_path, actual_learned):
    import duckdb
    from benchmarks.agent_supervisor.container_coding.learned_vector_preflight import qualify
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import CodeVectorIndexSnapshot, CodeVectorSearchResult

    root = scenario["repository"]
    initial = {}
    def vectors(repository):
        output = repository / ".runtime/learned-vectors"
        initial.update(qualify(repository, output, ["answer.py"], "answer", actual_learned["model"], actual_learned["model"].name))
        with duckdb.connect(str(output / "vectors.duckdb"), read_only=True, config={"threads": 1}) as db:
            snapshot = CodeVectorIndexSnapshot.from_dict(json.loads(db.execute("SELECT payload FROM snapshots").fetchone()[0]))
        return snapshot, CodeVectorSearchResult.from_dict(initial["hits"])

    admission = local.admit_local_benchmark_plan(graph=scenario["graph"], manifest=scenario["manifest"])
    with typed_claim(scenario, tmp_path) as (owner, _daemon, attempt):
        bundle = initial_bundle(scenario, owner, vector_builder=vectors)
        runtime = AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission=admission,
            server=owner.server, source=owner.source, implement=False, timeout_ms=30_000,
            context_bundle=bundle, refresh_context_on_completion=True,
            published_retrieval_policy="local-safetensors-symbols@1",
            published_learned_artifacts={"result": ".runtime/learned-vectors/result.json",
                "manifest": ".runtime/learned-vectors/model-manifest.json",
                "model_snapshot": str(actual_learned["model"])})
        try:
            binding = runtime.manifest["published_retrieval_policy"]
            assert binding["pinned_policy_id"] == initial["canary"]["policy_id"]
            assert runtime.start().succeeded
            task = owner.source.get_task(attempt.task_cid)
            transition = native_published_transition(scenario, tmp_path, owner, attempt)
            checked = run_owner_local_task_validations(server=owner.server, task_cid=task.task_cid,
                attempt_id=attempt.attempt_id, expected_revision=task.revision, source_transition=transition)
            assert checked["passed"] is True
            complete(owner, attempt, checked["results"][0]["evidence_digest"])
            retained_task = owner.source.get_task(task.task_cid)
            assert runtime.observe()["published_context"][0]["status"] == "pending_native_stop"
            assert runtime._context_refresh_attempts == {}
            stop = runtime.stop()
            assert stop.succeeded and stop.data["old_tree_fenced"] is True
            assert not runtime.process.snapshot(runtime.profile).members
            row = runtime.observe()["published_context"][0]
            assert row["status"] == "refreshed" and row["retrieval_status"] == "current"
            assert row["task_revision"] == 4
            refreshed = runtime._published_context[task.task_cid]
            assert row["goal_progress"] == refreshed["goal_progress"]
            retrieval = refreshed["retrieval"]
            assert retrieval["index_id"] != initial["index_id"]
            assert retrieval["reason"] == "rebuilt_with_pinned_configuration"
            assert retrieval["ducklake"]["status"] == "projected"
            lineage = retrieval["policy_lineage"]
            assert lineage["model_configuration_preserved"] is True
            assert lineage["previous_policy"]["model_artifact_id"] == lineage["current_policy"]["model_artifact_id"]
            receipt = lineage["embedding_receipt"]
            assert row["embedding_receipt"] == receipt
            assert receipt["local_embedding_calls"] == 3
            assert receipt["remote_embedding_calls"] == receipt["text_generation_calls"] == 0
            assert receipt["completion_authority"] is False
            assert runtime.refresh_after_stop()[0] == row
            assert runtime._context_refresh_attempts == {task.task_cid: 1}
            assert owner.source.get_task(task.task_cid) == retained_task
        finally:
            if runtime.process.snapshot(runtime.profile).members:
                assert runtime.stop().succeeded
            runtime.close()
