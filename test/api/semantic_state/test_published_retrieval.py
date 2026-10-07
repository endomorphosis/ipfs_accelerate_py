"""Closed lexical rebuild retains the native policy and reports drift."""
from dataclasses import replace
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import published_retrieval as runtime
from ipfs_accelerate_py.agent_supervisor.runtime.code_retrieval_context import (
    prepare_code_retrieval_context, load_code_retrieval_context,
)
from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle
from test.integration.test_admitted_context_refresh import lexical_vectors


def initial(tmp_path, *, query_text="answer"):
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "answer.py").write_text("def answer(): return 1\n")
    snapshot, hits = lexical_vectors(repository)
    if query_text != "answer":
        vocabulary, weights, _ = runtime._verify_lexical_snapshot(snapshot)
        hits = snapshot.search(runtime._vector(query_text, vocabulary, weights),
            max_results=hits.query.max_results)
    metadata = prepare_code_retrieval_context(repository=repository, task_id="TASK",
        query_text=query_text, snapshot=snapshot, result=hits, output=repository / ".runtime/retrieval.json")["metadata"]
    bundle = write_task_context_bundle(repository=repository, prepared=[{
        "schema": "supervisor-task-context-preparation@1", "task_cid": "task:one",
        "task_id": "TASK", "metadata": metadata}], output=repository / ".runtime/bundle.json")
    return repository, snapshot, hits, bundle


def test_native_lexical_descriptor_rebuild_and_tamper_refusal(tmp_path):
    root, old, old_hits, bundle = initial(tmp_path)
    binding = runtime.bind_published_retrieval_policy(repository=root, bundle=bundle,
        task_cid="task:one", task_id="TASK")
    (root / "answer.py").write_text("def answer(): return 2\n")
    rebuild = runtime.published_retrieval_rebuilder(repository=root, bundle=bundle, binding=binding)
    new, hits = rebuild(repository=root, previous_snapshot=old, previous_result=old_hits,
        query_text="answer", output=root / ".runtime/new")
    assert new.index_id != old.index_id and new.config == old.config
    assert hits.index_id == new.index_id and hits.hits
    for field, value in (("index_id", "foreign"), ("query_sha256", "0" * 64),
                         ("policy", "learned"), ("scope_paths", ["foreign.py"]),
                         ("completion_authority", True)):
        with pytest.raises(ValueError):
            runtime.published_retrieval_rebuilder(repository=root, bundle=bundle, binding={**binding, field: value})
    with pytest.raises(ValueError, match="predecessor/query"):
        rebuild(repository=root, previous_snapshot=old, previous_result=old_hits,
            query_text="foreign", output=root / ".runtime/new")
    (root / "answer.py").write_text("def different_name(): return 2\n")
    with pytest.raises(runtime.RetrievalRefreshUnavailable, match="lexical_vocabulary_changed"):
        rebuild(repository=root, previous_snapshot=old, previous_result=old_hits,
            query_text="answer", output=root / ".runtime/new")


def test_lexical_policy_never_relabels_another_model(tmp_path):
    _root, original, _hits, _bundle = initial(tmp_path)
    wrong = replace(original, config=replace(original.config, model_id="different-pinned-local-model"))
    with pytest.raises(ValueError, match="exact native lexical"):
        runtime._verify_lexical_snapshot(wrong)


def test_no_overlap_query_rebuilds_current_index_with_empty_nominations(tmp_path):
    query_text = "Schedule independent intervals under capacity constraints"
    root, old, old_hits, bundle = initial(tmp_path, query_text=query_text)
    assert old.rows and old_hits.hits == () and not any(old_hits.query.query_vector)
    binding = runtime.bind_published_retrieval_policy(repository=root, bundle=bundle,
        task_cid="task:one", task_id="TASK")
    assert binding["execution_authority"] is binding["completion_authority"] is False
    (root / "answer.py").write_text("def answer(): return 2\n")
    rebuild = runtime.published_retrieval_rebuilder(repository=root, bundle=bundle, binding=binding)
    new, hits = rebuild(repository=root, previous_snapshot=old, previous_result=old_hits,
        query_text=query_text, output=root / ".runtime/new")
    assert new.index_id != old.index_id and new.config == old.config
    assert len(new.rows) == len(old.rows) == 1
    assert hits.index_id == new.index_id and hits.hits == ()
    assert hits.complete is True and hits.searched_row_count == len(new.rows)
    assert hits.semantic_authority is False and not any(hits.query.query_vector)
    prepared = prepare_code_retrieval_context(repository=root, task_id="TASK", query_text=query_text,
        snapshot=new, result=hits, output=root / ".runtime/rebuilt.json")
    context = json.loads(load_code_retrieval_context(repository=root, task_id="TASK",
        artifact=".runtime/rebuilt.json", expected_sha256=prepared["sha256"]))
    assert context["status"] == "current" and context["hits"] == []
    assert context["index_id"] == new.index_id and context["stale_paths"] == []
    assert context["semantic_authority"] is context["execution_authority"] is context["completion_authority"] is False
    # Absence of overlap does not authorize silently changing the pinned
    # vocabulary, accepting a foreign query or overlooking source freshness.
    with pytest.raises(ValueError, match="predecessor/query"):
        rebuild(repository=root, previous_snapshot=old, previous_result=old_hits,
            query_text="foreign query", output=root / ".runtime/foreign")
    (root / "answer.py").write_text("def changed_name(): return 2\n")
    with pytest.raises(runtime.RetrievalRefreshUnavailable, match="lexical_vocabulary_changed"):
        rebuild(repository=root, previous_snapshot=old, previous_result=old_hits,
            query_text=query_text, output=root / ".runtime/changed")
    stale = json.loads(load_code_retrieval_context(repository=root, task_id="TASK",
        artifact=".runtime/rebuilt.json", expected_sha256=prepared["sha256"]))
    assert stale["status"] == "stale" and stale["hits"] == [] and stale["stale_paths"] == ["answer.py"]
