"""Closed lexical rebuild retains the native policy and reports drift."""
from dataclasses import replace
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import published_retrieval as runtime
from ipfs_accelerate_py.agent_supervisor.runtime.code_retrieval_context import prepare_code_retrieval_context
from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle
from test.integration.test_admitted_context_refresh import lexical_vectors


def initial(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "answer.py").write_text("def answer(): return 1\n")
    snapshot, hits = lexical_vectors(repository)
    metadata = prepare_code_retrieval_context(repository=repository, task_id="TASK",
        query_text="answer", snapshot=snapshot, result=hits, output=repository / ".runtime/retrieval.json")["metadata"]
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
