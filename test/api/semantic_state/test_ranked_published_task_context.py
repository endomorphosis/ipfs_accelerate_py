"""Fresh source-bound rankings precede real completed-publication capsules.

The local vectors are explicit character counts, not learned embeddings. Native
Quack ownership, Git publication, validation/completion, AST/query replay,
capsules and DuckLake persistence execute normally without remote providers.
"""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_local_completion_bridge import typed_claim, native_published_transition, complete
from test.api.semantic_state.test_published_task_context import initial_bundle, vectors
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import published_task_context as published
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations
from ipfs_accelerate_py.agent_supervisor.runtime.code_retrieval_context import load_code_retrieval_context


@contextmanager
def completed_ranked(scenario, tmp_path, *, current=False, vector_builder=vectors):
    admission = local.admit_local_benchmark_plan(graph=scenario["graph"], manifest=scenario["manifest"])
    with typed_claim(scenario, tmp_path) as (owner, _daemon, attempt):
        bundle = None if current else initial_bundle(scenario, owner, ranked_capsules=True, vector_builder=vector_builder)
        task = owner.source.get_task(attempt.task_cid)
        transition = native_published_transition(scenario, tmp_path, owner, attempt)
        if current:
            # A generation captured after publication but before completion
            # remains current; refreshing it must not invoke an embedding call.
            bundle = initial_bundle(scenario, owner, ranked_capsules=True, vector_builder=vector_builder)
        passed = run_owner_local_task_validations(server=owner.server, task_cid=task.task_cid,
            attempt_id=attempt.attempt_id, expected_revision=task.revision, source_transition=transition)
        assert passed["passed"] is True
        complete(owner, attempt, passed["results"][0]["evidence_digest"])
        yield owner, dict(server=owner.server, admission=admission, predecessor_bundle=bundle,
            task_cid=attempt.task_cid, output=scenario["repository"] / ".runtime/ranked-successor")


def reload(arguments, result):
    return published.load_published_task_context(server=arguments["server"], admission=arguments["admission"],
        artifact=result["refresh_artifact"], expected_sha256=result["refresh_sha256"])


def write_json(path, value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def pinned_policy(snapshot):
    from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_embedding_provider import PinnedEmbeddingPolicy
    config = snapshot.config
    return PinnedEmbeddingPolicy(provider_id="explicit-local-character-count-test",
        model_artifact_id=config.model_id, model_revision=config.model_revision,
        dimensions=config.dimensions, chunker_id=config.chunker_id,
        normalizer=config.normalization, distance=config.metric,
        corpus_root_id=snapshot.tree_id, index_root_id=snapshot.ast_index_id,
        forest_id=snapshot.forest_id, tree_id=snapshot.tree_id, config_id="same-static-model-config")


def pinned_vectors(repository, **kwargs):
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import search_code_symbol_vector_index
    snapshot, hits = vectors(repository, **kwargs)
    snapshot = replace(snapshot, config=replace(snapshot.config, configuration_id=pinned_policy(snapshot).policy_id))
    return snapshot, search_code_symbol_vector_index(snapshot, hits.query.query_vector, max_results=5)


def test_ranked_successor_replays_native_pinned_policy_root_rebinding(scenario, tmp_path):
    with completed_ranked(scenario, tmp_path, vector_builder=pinned_vectors) as (_owner, arguments):
        def rebuild(**kwargs):
            previous = kwargs["previous_snapshot"]
            snapshot, hits = pinned_vectors(**kwargs)
            return published.PinnedRetrievalRebuild(snapshot, hits, pinned_policy(previous), pinned_policy(snapshot))
        result = published.refresh_published_task_context(**arguments, retrieval_rebuilder=rebuild)
        observed = reload(arguments, result)
        lineage = observed["retrieval"]["policy_lineage"]
        assert lineage["model_configuration_preserved"] is True
        assert lineage["previous_policy"]["tree_id"] != lineage["current_policy"]["tree_id"]
        assert observed["retrieval"]["previous_config_id"] != observed["retrieval"]["config_id"]
        assert lineage["previous_policy"]["model_revision"] == lineage["current_policy"]["model_revision"]
        forged = json.loads(json.dumps(observed))
        forged["retrieval"]["policy_lineage"]["current_policy"]["model_revision"] = "foreign"
        path = arguments["output"] / "forged-policy-result.json"
        with pytest.raises(ValueError):
            published.load_published_task_context(server=arguments["server"], admission=arguments["admission"],
                artifact=path.relative_to(scenario["repository"]).as_posix(), expected_sha256=write_json(path, forged))


@pytest.mark.parametrize("current", [False, True])
def test_native_ranked_successor_rebuilds_or_reuses_retrieval_before_capsules(scenario, tmp_path, monkeypatch, current):
    root = scenario["repository"]
    with completed_ranked(scenario, tmp_path, current=current) as (owner, arguments):
        before = owner.source.get_task(arguments["task_cid"])
        bundle_before = (root / arguments["predecessor_bundle"]["artifact"]).read_bytes()
        calls = []
        real_semantic = published.prepare_semantic_context
        def rebuild(**kwargs):
            calls.append("retrieval")
            assert not (arguments["output"] / "semantic").exists()
            assert not (arguments["output"] / "world").exists()
            return vectors(**kwargs)
        def semantic(**kwargs):
            calls.append("semantic")
            selected = kwargs["worker_retrieval"]
            context = json.loads(load_code_retrieval_context(repository=root, task_id=kwargs["task_id"],
                artifact=selected["artifact"], expected_sha256=selected["sha256"]))
            assert context["status"] == "current"
            assert context["source_sha256"]["answer.py"] == hashlib.sha256((root / "answer.py").read_bytes()).hexdigest()
            return real_semantic(**kwargs)
        monkeypatch.setattr(published, "prepare_semantic_context", semantic)
        result = published.refresh_published_task_context(**arguments, retrieval_rebuilder=rebuild)
        assert calls == (["semantic"] if current else ["retrieval", "semantic"])
        observed = reload(arguments, result)
        assert observed["task_status"] == "completed"
        assert observed["retrieval"]["reason"] == ("verified_unchanged_source_scope" if current else "rebuilt_with_pinned_configuration")
        projection = observed["semantic"]["worker_projection"]
        assert projection["selector"] == published.RANKED_CAPSULE_SELECTOR
        assert projection["retrieval"] == {"artifact": observed["metadata"]["Code retrieval artifact"],
                                         "sha256": observed["metadata"]["Code retrieval sha256"]}
        assert projection["selected_symbols"][0]["stable_symbol_id"] == projection["ranked_symbol_ids"][0]
        payload = json.loads((root / observed["metadata"]["Semantic context artifact"]).read_bytes())
        assert payload["raw_sources"]["test_answer.py"] == (root / "test_answer.py").read_text()
        assert observed["semantic"]["ducklake"]["status"] == "projected"
        assert observed["world"]["metadata"]["status"] == "projected"
        assert owner.source.get_task(arguments["task_cid"]) == before
        assert (root / arguments["predecessor_bundle"]["artifact"]).read_bytes() == bundle_before
        assert all(observed[key] is False for key in ("execution_authority", "completion_authority",
            "canonical_task_mutated", "launch_nomination_mutated"))


@pytest.mark.parametrize("mutation", ["foreign_reference", "foreign_digest", "unknown_rank", "query"])
def test_rehashed_ranked_predecessor_cannot_change_selected_retrieval(scenario, tmp_path, monkeypatch, mutation):
    root = scenario["repository"]
    with completed_ranked(scenario, tmp_path) as (_owner, arguments):
        bundle = json.loads((root / arguments["predecessor_bundle"]["artifact"]).read_bytes())
        metadata = bundle["tasks"][0]["metadata"]
        path = root / metadata["semantic context artifact"]
        semantic = json.loads(path.read_bytes())
        policy = semantic["worker_projection"]
        if mutation == "foreign_reference":
            policy["retrieval"]["artifact"] += ".foreign"
        elif mutation == "foreign_digest":
            policy["retrieval"]["sha256"] = "0" * 64
        elif mutation == "unknown_rank":
            policy["ranked_symbol_ids"] = ["unknown"]
        else:
            policy["query"] = "other task"
        forged = path.with_name("forged-context.json")
        metadata["semantic context artifact"] = forged.relative_to(root).as_posix()
        metadata["semantic context sha256"] = write_json(forged, semantic)
        bundle_path = root / ".runtime/predecessor/forged-bundle.json"
        reference = {"artifact": bundle_path.relative_to(root).as_posix(), "sha256": write_json(bundle_path, bundle)}
        monkeypatch.setattr(published, "prepare_semantic_context", lambda **kwargs: pytest.fail("partial semantic successor"))
        with pytest.raises(ValueError):
            published.refresh_published_task_context(**{**arguments, "predecessor_bundle": reference}, retrieval_rebuilder=vectors)
        assert not arguments["output"].exists()


@pytest.mark.parametrize("failure", ["unavailable", "stale_result", "owner_drift", "deadline"])
def test_failed_ranked_rebuild_refuses_before_semantic_or_world(scenario, tmp_path, monkeypatch, failure):
    from ipfs_accelerate_py.agent_supervisor.runtime.published_retrieval import RetrievalRefreshUnavailable
    with completed_ranked(scenario, tmp_path) as (owner, arguments):
        before = owner.source.get_task(arguments["task_cid"])
        clock = [1000.]
        if failure == "deadline":
            monkeypatch.setattr(published, "time", SimpleNamespace(monotonic=lambda: clock[0]))
            arguments["deadline_monotonic"] = 1001.
        def rebuild(**kwargs):
            if failure == "unavailable":
                raise RetrievalRefreshUnavailable("model_unavailable")
            if failure == "stale_result":
                return kwargs["previous_snapshot"], kwargs["previous_result"]
            value = vectors(**kwargs)
            if failure == "owner_drift":
                # Deliberate native-store revision corruption tests the owner
                # recheck; restore it below rather than claim an authorized edit.
                with owner.server._lock:
                    owner.server._connection.execute("UPDATE tasks SET revision=revision+1 WHERE task_cid=?", [arguments["task_cid"]])
            else:
                clock[0] = 1002.
            return value
        monkeypatch.setattr(published, "prepare_semantic_context", lambda **kwargs: pytest.fail("partial semantic successor"))
        try:
            with pytest.raises((ValueError, TimeoutError)):
                published.refresh_published_task_context(**arguments, retrieval_rebuilder=rebuild)
            assert not (arguments["output"] / "semantic").exists()
            assert not (arguments["output"] / "world").exists()
            assert not (arguments["output"] / "result.json").exists()
        finally:
            if failure == "owner_drift":
                with owner.server._lock:
                    owner.server._connection.execute("UPDATE tasks SET revision=? WHERE task_cid=?", [before.revision, arguments["task_cid"]])
        assert owner.source.get_task(arguments["task_cid"]) == before


@pytest.mark.parametrize("mutation", ["receipt_projection", "result_identity", "model_configuration"])
def test_rehashed_successor_receipt_rechecks_ranked_retrieval_lineage(scenario, tmp_path, mutation):
    root = scenario["repository"]
    with completed_ranked(scenario, tmp_path) as (_owner, arguments):
        result = published.refresh_published_task_context(**arguments, retrieval_rebuilder=vectors)
        payload = json.loads((root / result["refresh_artifact"]).read_bytes())
        if mutation == "receipt_projection":
            payload["semantic"]["worker_projection"]["max_capsules"] = 0
        elif mutation == "result_identity":
            payload["retrieval"]["result_id"] = "other-result"
        else:
            payload["retrieval"]["config_id"] = "other-config"
        path = arguments["output"] / "forged-result.json"
        with pytest.raises(ValueError):
            published.load_published_task_context(server=arguments["server"], admission=arguments["admission"],
                artifact=path.relative_to(root).as_posix(), expected_sha256=write_json(path, payload))


@pytest.mark.parametrize("changed", ["edited", "removed"])
def test_ranked_successor_router_replays_history_without_current_authority(scenario, tmp_path, changed):
    from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
        ContextCompiler, build_text_context_references, render_context_capsule,
    )
    from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec

    root = scenario["repository"]
    with completed_ranked(scenario, tmp_path) as (_owner, arguments):
        result = published.refresh_published_task_context(**arguments, retrieval_rebuilder=vectors)
        observed = reload(arguments, result)
        artifact = observed["metadata"]["Semantic context artifact"]
        text = (root / artifact).read_text()
        refs = build_text_context_references(text, reference_prefix="semantic-context", kind="semantic-context",
            path=artifact, repository_id="test:repository", tree_id="test:tree", required=True, chunk_bytes=1201)
        compiled = ContextCompiler(ContextBudget(max_input_tokens=32768, max_items=128,
            max_item_bytes=16384, max_serialized_bytes=262144)).compile(
            repository_id="test:repository", tree_id="test:tree", objective_id=observed["task_id"],
            objective_revision="sha256:task", policy_id="policy:test", policy_revision="sha256:policy",
            caller="supervisor:test", stage="implementation", goal={"id": observed["task_id"]},
            authority={"mode": "candidate_only", "completion_authority": False},
            scope={"allowed_paths": ["answer.py"]}, acceptance={"criteria": ["pending validation"]}, evidence=refs)
        prompt = render_context_capsule(compiled.capsule)
        transports = ("supervisor-semantic-router-input@1", "supervisor-semantic-router-input@2")
        encoded = {transport: codec.encode_semantic_router_prompt(prompt=prompt, repository=root,
            transport_schema=transport) for transport in transports}
        if changed == "edited":
            (root / "answer.py").write_text("def answer():\n    return 7\n")
        else:
            (root / "answer.py").unlink()
            (root / "test_answer.py").unlink()
        with pytest.raises((ValueError, FileNotFoundError)):
            reload(arguments, result)
        for transport, original in encoded.items():
            with pytest.raises((ValueError, FileNotFoundError)):
                codec.encode_semantic_router_prompt(prompt=prompt, repository=root, transport_schema=transport)
            replay = codec.replay_semantic_router_prompt_for_audit(prompt=prompt, repository=root,
                transport_schema=transport)
            assert replay.provider_prompt == original.provider_prompt
            assert replay.table == original.table
            assert replay.native_prompt == original.native_prompt
            assert replay.freshness_checked is replay.receipt["freshness_checked"] is False
            assert replay.receipt["completion_authority"] is replay.receipt["execution_authority"] is False
            with pytest.raises(ValueError, match="historical"):
                codec.decode_semantic_router_response(response='{"schema":"supervisor-semantic-router-reply@1"}',
                    encoded=replay, repository=root)
