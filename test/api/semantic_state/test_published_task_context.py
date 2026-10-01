"""Real publication/completion refreshes evidence without mutating authority."""
from dataclasses import replace
import json
import math

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_local_completion_bridge import (
    typed_claim, native_published_transition, complete,
)
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations
from ipfs_accelerate_py.agent_supervisor.runtime.published_task_context import (
    refresh_published_task_context, load_published_task_context,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle
from ipfs_accelerate_py.agent_supervisor.semantic_state.intent_world_snapshot import (
    capture_intent_world_snapshot, persist_intent_world_snapshot,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


def vectors(repository, previous_snapshot=None, previous_result=None, query_text="answer", output=None):
    """Explicit local character-count test index; no learned-model claim."""
    from ipfs_accelerate_py.agent_supervisor.analysis.program_ast_adapters import build_program_evidence_index
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import (
        build_code_symbol_vector_index, search_code_symbol_vector_index,
    )
    sources = {"answer.py": (repository / "answer.py").read_text()}
    ast = build_program_evidence_index(sources).ast_index
    root = local.content_identity({"sources": sources})
    def vector(text):
        values = (float(text.count("a") + 1), float(len(text) + 1))
        norm = math.sqrt(sum(value * value for value in values))
        return tuple(value / norm for value in values)
    snapshot = build_code_symbol_vector_index(ast, forest_id=root, tree_id=root,
        model_id="explicit-character-count-test", model_revision="1", dimensions=2,
        configuration_id="character-count-v1", vectors=lambda row: vector(row.qualified_symbol))
    return snapshot, search_code_symbol_vector_index(snapshot, vector(query_text), max_results=5)


def initial_bundle(scenario, owner, *, retrieval=True, vector_builder=vectors):
    from ipfs_accelerate_py.agent_supervisor.runtime.code_retrieval_context import prepare_code_retrieval_context
    from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import IpfsDatasetsSemanticStateProvider
    root = scenario["repository"]
    out = root / ".runtime/predecessor"
    alias = scenario["graph"].tasks[0].task_key
    semantic = prepare_semantic_context(repository=root, paths=["answer.py", "test_answer.py"],
        required_raw_paths=["test_answer.py"], objective="Repair the answer", task_id=alias,
        output=out / "semantic")
    get_block = lambda cid: (out / "semantic/blocks" / cid).read_bytes()
    view = IpfsDatasetsSemanticStateProvider().open_verified_view(semantic["semantic_root_cid"], get_block)
    with owner.server._lock:
        connection = owner.server._connection
        connection.execute("BEGIN TRANSACTION")
        try:
            intent = IntentRepository(bound_connection=connection, install_schema=False)
            capture = capture_intent_world_snapshot(intent, repository_id=view.root.repository_id,
                task_cids=[scenario["task_cid"]], semantic_root_cid=semantic["semantic_root_cid"],
                get_semantic_block=get_block, transaction_owned_by_caller=True)
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    world = persist_intent_world_snapshot(capture, output=out / "world", task_id=alias)
    metadata = {
        "Semantic context artifact": (out / "semantic/worker-context.json").relative_to(root).as_posix(),
        "Semantic context sha256": semantic["worker_payload_sha256"],
        "Semantic context refresh": "true",
        "World context artifact": (out / "world/intent-world.json").relative_to(root).as_posix(),
        "World context sha256": world["artifact_sha256"],
        "World context repository": view.root.repository_id,
    }
    if retrieval:
        snapshot, hits = vector_builder(root)
        metadata.update(prepare_code_retrieval_context(repository=root, task_id=alias,
            query_text="answer", snapshot=snapshot, result=hits, output=out / "retrieval.json")["metadata"])
    prepared = {"schema": "supervisor-task-context-preparation@1", "task_cid": scenario["task_cid"],
                "task_id": alias, "metadata": metadata}
    return write_task_context_bundle(repository=root, prepared=[prepared], output=out / "bundle.json")


@pytest.mark.parametrize("rebuild", [False, True])
def test_actual_completed_publication_refreshes_roots_and_keeps_authority(scenario, tmp_path, rebuild):
    admission = local.admit_local_benchmark_plan(graph=scenario["graph"], manifest=scenario["manifest"])
    with typed_claim(scenario, tmp_path) as (owner, _daemon, attempt):
        bundle = initial_bundle(scenario, owner)
        old_bundle_bytes = (scenario["repository"] / bundle["artifact"]).read_bytes()
        arguments = dict(server=owner.server, admission=admission, predecessor_bundle=bundle,
            task_cid=attempt.task_cid, output=scenario["repository"] / ".runtime/published",
            retrieval_rebuilder=vectors if rebuild else None)
        with pytest.raises(ValueError, match="accepted source transition"):
            refresh_published_task_context(**arguments)
        task = owner.source.get_task(attempt.task_cid)
        transition = native_published_transition(scenario, tmp_path, owner, attempt)
        with pytest.raises(local.LocalPlanningError, match="no exact native owner observation"):
            refresh_published_task_context(**arguments)
        passed = run_owner_local_task_validations(server=owner.server, task_cid=task.task_cid,
            attempt_id=attempt.attempt_id, expected_revision=task.revision, source_transition=transition)
        assert passed["passed"] is True
        with pytest.raises(ValueError, match="completed native task"):
            refresh_published_task_context(**arguments)
        complete(owner, attempt, passed["results"][0]["evidence_digest"])
        canonical_before = owner.source.get_task(task.task_cid)
        refreshed = refresh_published_task_context(**arguments)
        loaded = load_published_task_context(server=owner.server, admission=admission,
            artifact=refreshed["refresh_artifact"], expected_sha256=refreshed["refresh_sha256"])
        assert loaded["task_revision"] == 4
        assert loaded["task_status"] == "completed"
        assert loaded["intent_freshness_checked"] is True
        assert loaded["semantic_root_cid"] != loaded["predecessor_semantic_root_cid"]
        assert loaded["world_snapshot_cid"] != loaded["predecessor_world_snapshot_cid"]
        assert loaded["accepted_source_tree_id"] == passed["source_tree_id"]
        assert loaded["semantic"]["ducklake"]["status"] == "projected"
        assert loaded["world"]["metadata"]["status"] == "projected"
        assert loaded["planning_context"]["tasks"][0]["status"] == "completed"
        progress = loaded["goal_progress"]
        assert progress["status"] == "observed"
        assert progress["population_scope"] == "selected-tasks"
        assert progress["task_population_complete"] is False
        assert progress["goal_completion_authority"] == "unresolved"
        assert progress["goal_contracts_evaluated"] is False
        assert progress["tasks"][0]["status"] == "completed"
        assert progress["tasks"][0]["revision"] == 4
        assert progress["tasks"][0]["current_completion_receipt"]["receipt_cid"]
        assert all(goal["status"] == "open" for goal in progress["goals"])
        assert all(goal["observed_counts"]["completed_with_current_receipt"] == 1 for goal in progress["goals"])
        assert loaded["unavailable_components"]
        assert all(loaded[name] is False for name in ("execution_authority", "completion_authority",
            "canonical_task_mutated", "launch_nomination_mutated"))
        assert owner.source.get_task(task.task_cid) == canonical_before
        assert (scenario["repository"] / bundle["artifact"]).read_bytes() == old_bundle_bytes
        assert loaded["retrieval"]["status"] == ("current" if rebuild else "unavailable")
        if rebuild:
            assert loaded["retrieval"]["index_id"] != loaded["retrieval"]["previous_index_id"]
            assert loaded["retrieval"]["ducklake"]["status"] == "projected"
        else:
            assert "Code retrieval artifact" not in loaded["metadata"]
        # Caller-provided stale hashes, source changes and task mismatches do
        # not become valid via the cached context observation.
        with pytest.raises(ValueError, match="digest"):
            load_published_task_context(server=owner.server, admission=admission,
                artifact=refreshed["refresh_artifact"], expected_sha256="0" * 64)
        predecessor = json.loads(old_bundle_bytes)["tasks"][0]["metadata"]
        predecessor_artifact = scenario["repository"] / predecessor["semantic context artifact"]
        original = predecessor_artifact.read_bytes()
        predecessor_artifact.write_text("{}")
        try:
            with pytest.raises(ValueError, match="digest"):
                load_published_task_context(server=owner.server, admission=admission,
                    artifact=refreshed["refresh_artifact"], expected_sha256=refreshed["refresh_sha256"])
        finally:
            predecessor_artifact.write_bytes(original)
        # A different canonical intent event invalidates the old world even
        # when the completed task and source themselves are unchanged.
        with owner.server._lock:
            intent = IntentRepository(bound_connection=owner.server._connection, install_schema=False)
            intent.upsert_objective(objective_id="later-objective", objective_alias="LATER", title="Later work")
        with pytest.raises(ValueError, match="intent|watermark|stale"):
            load_published_task_context(server=owner.server, admission=admission,
                artifact=refreshed["refresh_artifact"], expected_sha256=refreshed["refresh_sha256"])
        (scenario["repository"] / "answer.py").write_text("def answer():\n    return 3\n")
        with pytest.raises(ValueError):
            load_published_task_context(server=owner.server, admission=admission,
                artifact=refreshed["refresh_artifact"], expected_sha256=refreshed["refresh_sha256"])


def test_foreign_owner_is_never_a_publication_refresh_authority(tmp_path):
    with pytest.raises(TypeError, match="actual native owner"):
        refresh_published_task_context(server=object(), admission={}, predecessor_bundle={},
            task_cid="foreign", output=tmp_path / "result")


def test_native_pinned_embedding_policy_allows_only_root_rebinding(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import search_code_symbol_vector_index
    from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_embedding_provider import PinnedEmbeddingPolicy
    from ipfs_accelerate_py.agent_supervisor.runtime.published_task_context import (
        PinnedRetrievalRebuild, _verify_retrieval_rebuild,
    )
    root = tmp_path / "source"
    root.mkdir()
    (root / "answer.py").write_text("def answer(): return 1\n")
    old, old_hits = vectors(root)
    (root / "answer.py").write_text("def answer(): return 2\n")
    new, _ = vectors(root)
    def pin(snapshot):
        config = snapshot.config
        policy = PinnedEmbeddingPolicy(provider_id="explicit-local-character-count-test",
            model_artifact_id=config.model_id, model_revision=config.model_revision,
            dimensions=config.dimensions, chunker_id=config.chunker_id,
            normalizer=config.normalization, distance=config.metric,
            corpus_root_id=snapshot.tree_id, index_root_id=snapshot.ast_index_id,
            forest_id=snapshot.forest_id, tree_id=snapshot.tree_id, config_id="same-static-model-config")
        snapshot = replace(snapshot, config=replace(config, configuration_id=policy.policy_id))
        return policy, snapshot, search_code_symbol_vector_index(snapshot, old_hits.query.query_vector, max_results=5)
    old_policy, old, old_hits = pin(old)
    new_policy, new, new_hits = pin(new)
    rebuilt = PinnedRetrievalRebuild(new, new_hits, old_policy, new_policy)
    assert _verify_retrieval_rebuild(rebuilt, old, old_hits)[2]["model_configuration_preserved"] is True
    for changed in (replace(new_policy, model_revision="different"),
                    replace(new_policy, config_id="different"),
                    replace(new_policy, allow_remote=True, remote_endpoint_id="remote"),
                    replace(new_policy, tree_id=old.tree_id)):
        with pytest.raises(ValueError, match="pinned"):
            _verify_retrieval_rebuild(replace(rebuilt, current_policy=changed), old, old_hits)
