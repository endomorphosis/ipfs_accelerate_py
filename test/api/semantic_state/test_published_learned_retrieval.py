"""Actual offline MiniLM refresh preserves native model and source contracts."""
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import CodeVectorIndexSnapshot, CodeVectorSearchResult
from ipfs_accelerate_py.agent_supervisor.runtime.code_retrieval_context import prepare_code_retrieval_context
from ipfs_accelerate_py.agent_supervisor.runtime import published_learned_retrieval as learned
from ipfs_accelerate_py.agent_supervisor.runtime.published_task_context import _verify_retrieval_rebuild, _verify_embedding_observation
from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle

REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL = Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots" / REVISION
SOURCE = "def answer():\n    return 1\n\ndef count_characters(text):\n    return len(text)\n"


@pytest.fixture(scope="module")
def actual_learned(tmp_path_factory):
    if not (MODEL / "model.safetensors").is_file():
        pytest.skip("requires existing local MiniLM assets; never downloads")
    pytest.importorskip("sentence_transformers")
    import duckdb
    from benchmarks.agent_supervisor.container_coding.learned_vector_preflight import qualify
    home = tmp_path_factory.mktemp("pinned-learned-refresh")
    model = home / REVISION
    shutil.copytree(MODEL, model, symlinks=False)
    for path in (model, *model.rglob("*")):
        path.chmod(0o700 if path.is_dir() else 0o600)
    repository = home / "initial"
    repository.mkdir()
    (repository / "answer.py").write_text(SOURCE)
    with pytest.MonkeyPatch.context() as env:
        env.setenv("HF_HUB_OFFLINE", "1")
        env.setenv("TRANSFORMERS_OFFLINE", "1")
        env.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
        env.setenv("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE", "0")
        result = qualify(repository, repository / ".runtime/vectors", ["answer.py"], "answer", model, REVISION)
        with duckdb.connect(str(repository / ".runtime/vectors/vectors.duckdb"), read_only=True, config={"threads": 1}) as db:
            snapshot = CodeVectorIndexSnapshot.from_dict(json.loads(db.execute("SELECT payload FROM snapshots").fetchone()[0]))
        yield {"root": repository, "model": model, "result": result, "snapshot": snapshot,
               "hits": CodeVectorSearchResult.from_dict(result["hits"])}


@pytest.fixture
def initial(tmp_path, actual_learned):
    root = tmp_path / "repository"
    root.mkdir()
    (root / "answer.py").write_text(SOURCE)
    vectors = root / ".runtime/vectors"
    vectors.mkdir(parents=True)
    for name in ("result.json", "model-manifest.json"):
        shutil.copyfile(actual_learned["root"] / ".runtime/vectors" / name, vectors / name)
    prepared = prepare_code_retrieval_context(repository=root, task_id="TASK", query_text="answer",
        snapshot=actual_learned["snapshot"], result=actual_learned["hits"], output=root / ".runtime/retrieval.json")
    bundle = write_task_context_bundle(repository=root, prepared=[{"schema": "supervisor-task-context-preparation@1",
        "task_cid": "task:one", "task_id": "TASK", "metadata": prepared["metadata"]}], output=root / ".runtime/bundle.json")
    artifacts = {"result": ".runtime/vectors/result.json", "manifest": ".runtime/vectors/model-manifest.json",
                 "model_snapshot": str(actual_learned["model"])}
    binding = learned.bind_published_learned_retrieval_policy(repository=root, bundle=bundle,
        task_cid="task:one", task_id="TASK", artifacts=artifacts)
    return {**actual_learned, "root": root, "bundle": bundle, "artifacts": artifacts, "binding": binding}


def _callback(initial):
    return learned.published_learned_retrieval_rebuilder(repository=initial["root"],
        bundle=initial["bundle"], binding=initial["binding"])


def _rebuild(initial, callback):
    return callback(repository=initial["root"], previous_snapshot=initial["snapshot"],
        previous_result=initial["hits"], query_text="answer", output=initial["root"] / ".runtime/new")


def test_actual_local_model_rebuild_rebinds_only_source_roots(initial):
    callback = _callback(initial)
    (initial["root"] / "answer.py").write_text(SOURCE.replace("return 1", "return 2"))
    value = _rebuild(initial, callback)
    snapshot, hits, lineage = _verify_retrieval_rebuild(value, initial["snapshot"], initial["hits"])
    assert snapshot.index_id != initial["snapshot"].index_id
    assert hits.index_id == snapshot.index_id and hits.hits
    roots = {"corpus_root_id", "index_root_id", "forest_id", "tree_id"}
    old, new = value.previous_policy._payload(), value.current_policy._payload()
    assert {k: v for k, v in old.items() if k not in roots} == {k: v for k, v in new.items() if k not in roots}
    assert snapshot.config.configuration_id == value.current_policy.policy_id
    assert value.current_policy.model_artifact_id == initial["result"]["model_artifact_id"]
    receipt = lineage["embedding_receipt"]
    assert receipt["canary"]["disposition"] == "passed"
    assert receipt["local_embedding_calls"] == 3
    assert receipt["local_embedding_texts"] == 6
    assert receipt["remote_embedding_calls"] == receipt["text_generation_calls"] == 0
    assert receipt["native_fact_rows_replayed"] == 2
    assert receipt["completion_authority"] is receipt["semantic_authority"] is False
    assert (initial["root"] / ".runtime/new/snapshot.json").is_file()
    for field, wrong in (("index_id", "foreign"), ("remote_embedding_calls", 1), ("completion_authority", True),
                         ("source_sha256", {"answer.py": "0" * 64}), ("local_embedding_calls", 0),
                         ("local_embedding_texts", 0), ("embedding_receipts", [])):
        with pytest.raises(ValueError, match="embedding receipt"):
            _verify_retrieval_rebuild(replace(value, embedding_receipt={**value.embedding_receipt, field: wrong}),
                initial["snapshot"], initial["hits"])


@pytest.mark.parametrize("drift", ["result", "manifest", "policy", "query", "runtime", "source_symlink", "source_fifo"])
def test_pin_or_source_drift_abstains_before_loading_model(initial, monkeypatch, drift):
    if drift == "result":
        (initial["root"] / initial["artifacts"]["result"]).write_text("{}")
    elif drift == "manifest":
        (initial["root"] / initial["artifacts"]["manifest"]).write_text("{}")
    elif drift == "policy":
        initial["binding"]["pinned_policy_id"] = "foreign-policy"
    elif drift == "query":
        initial["binding"]["query_sha256"] = "0" * 64
    elif drift == "runtime":
        monkeypatch.setattr(learned, "_versions", lambda: {"torch": "different"})
    else:
        path = initial["root"] / "answer.py"
        path.unlink()
        if drift == "source_symlink":
            target = initial["root"] / "other.py"
            target.write_text(SOURCE)
            path.symlink_to(target)
        else:
            os.mkfifo(path)
    monkeypatch.setattr(learned, "_LocalRouterModel", lambda *a, **k: pytest.fail("must not load a model"))
    with pytest.raises(learned.LearnedRetrievalUnavailable) as caught:
        _rebuild(initial, _callback(initial))
    assert caught.value.receipt["local_embedding_calls"] == 0
    assert caught.value.receipt["remote_embedding_calls"] == 0
    assert caught.value.receipt["status"] == "unavailable"


def test_owner_model_refuses_worker_writable_or_symlink_assets(tmp_path):
    model, repository = tmp_path / "model", tmp_path / "repository"
    model.mkdir(mode=0o700)
    repository.mkdir()
    file = model / "model.safetensors"
    file.write_bytes(b"unloaded bounded test inventory")
    file.chmod(0o666)
    with pytest.raises(ValueError, match="owner-controlled"):
        learned._owner_model(model, repository)
    file.chmod(0o600)
    saved = model / "retained"
    file.rename(saved)
    file.symlink_to(saved)
    with pytest.raises(ValueError, match="owner-controlled"):
        learned._owner_model(model, repository)


def test_owner_model_refuses_replaceable_ancestor(tmp_path):
    parent = tmp_path / "writable"
    model, repository = parent / "model", tmp_path / "repository"
    model.mkdir(parents=True, mode=0o700)
    repository.mkdir()
    parent.chmod(0o777)
    with pytest.raises(ValueError, match="ancestors"):
        learned._owner_model(model, repository)


def test_actual_model_canary_failure_keeps_observed_local_embedding_cost(initial, monkeypatch):
    original = learned._LocalRouterModel.embed_texts
    def corrupt_after_actual_inference(self, texts, **kwargs):
        vectors = original(self, texts, **kwargs)
        return [[1.0] * len(vector) for vector in vectors]
    monkeypatch.setattr(learned._LocalRouterModel, "embed_texts", corrupt_after_actual_inference)
    (initial["root"] / "answer.py").write_text(SOURCE.replace("return 1", "return 2"))
    with pytest.raises(learned.LearnedRetrievalUnavailable, match="learned_canary_failed") as caught:
        _rebuild(initial, _callback(initial))
    assert caught.value.receipt["canary"]["disposition"] == "failed"
    assert caught.value.receipt["local_embedding_calls"] == 1
    assert caught.value.receipt["local_embedding_texts"] == 3
    assert caught.value.receipt["text_generation_calls"] == 0
    receipt = caught.value.receipt
    assert _verify_embedding_observation(receipt, initial["snapshot"]) == receipt
    for field, wrong in (("local_embedding_calls", 0), ("previous_index_id", "foreign"),
                         ("current_policy_id", "foreign"), ("model_artifact_id", "foreign")):
        with pytest.raises(ValueError, match="embedding receipt"):
            _verify_embedding_observation({**receipt, field: wrong}, initial["snapshot"])
