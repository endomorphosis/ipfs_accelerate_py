"""Historical retrieval nomination keeps immutable Source384 identity checks.

The Source384 numerical/config seams are authored by successor_case; lexical
AST/index/query reconstruction is real. The integration counterpart exercises
actual checkpoint inference, Z3, owner publication and the same lexical path.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from test.api.semantic_state.test_source384_successor_context import successor_case  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.runtime import published_retrieval as lexical
from ipfs_accelerate_py.agent_supervisor.runtime import published_learned_retrieval as learned
from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as source384
from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as bundles


@pytest.fixture
def selected(successor_case):
    import duckdb
    from benchmarks.agent_supervisor.container_coding.vector_index_preflight import qualify
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import (
        CodeVectorIndexSnapshot, CodeVectorSearchResult,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.code_retrieval_context import prepare_code_retrieval_context
    c = successor_case
    c.source.write_text("def answer(): return 1\n")
    output = c.repository / ".runtime/initial"
    result = qualify(c.repository, output / "vectors", ["module.py"], "answer")
    with duckdb.connect(str(output / "vectors/vectors.duckdb"), read_only=True, config={"threads": 1}) as db:
        snapshot = CodeVectorIndexSnapshot.from_dict(json.loads(db.execute(
            "SELECT payload FROM snapshots WHERE id=?", [result["index_id"]]).fetchone()[0]))
    hits = CodeVectorSearchResult.from_dict(result["hits"])
    metadata = prepare_code_retrieval_context(repository=c.repository, task_id="TASK", query_text="answer",
        snapshot=snapshot, result=hits, output=output / "retrieval.json")["metadata"]
    row = {"schema": "supervisor-task-context-preparation@1", "task_cid": "task:one", "task_id": "TASK",
           "metadata": metadata}
    # Bind the real lexical policy before adding the authored numerical receipt.
    lexical_bundle = bundles.write_task_context_bundle(repository=c.repository, prepared=[row],
        output=output / "lexical-only.json")
    binding = lexical.bind_published_retrieval_policy(repository=c.repository, bundle=lexical_bundle,
        task_cid="task:one", task_id="TASK")
    bundle = bundles.write_task_context_bundle(repository=c.repository,
        prepared=[{**row, "source384_context": c.predecessor}], output=output / "bundle.json")
    c.source.write_text("def answer(): return 2\n")
    return SimpleNamespace(case=c, bundle=bundle, binding=binding, snapshot=snapshot, hits=hits, metadata=metadata)


def _rebuild(s):
    return lexical.published_retrieval_rebuilder(repository=s.case.repository, bundle=s.bundle, binding=s.binding)


def test_historical_source384_allows_current_lexical_reconstruction(selected):
    s = selected
    rebuild = _rebuild(s)
    snapshot, hits = rebuild(repository=s.case.repository, previous_snapshot=s.snapshot,
        previous_result=s.hits, query_text="answer", output=s.case.repository / ".runtime/new")
    assert snapshot.index_id != s.snapshot.index_id
    assert snapshot.config == s.snapshot.config and hits.hits
    assert source384.validate_historical_source384_selection(repository=s.case.repository,
        expected_receipt=s.case.predecessor) == s.case.predecessor
    assert not any(key == "infer" for key, _ in s.case.observations)
    # Initial policy binding, launch and worker reads still require current source.
    with pytest.raises(ValueError, match="differs from signed population"):
        lexical.bind_published_retrieval_policy(repository=s.case.repository, bundle=s.bundle,
            task_cid="task:one", task_id="TASK")
    with pytest.raises(ValueError, match="differs from signed population"):
        bundles.load_task_context_nomination(repository=s.case.repository, artifact=s.bundle["artifact"],
            expected_sha256=s.bundle["sha256"], task_cid="task:one", task_id="TASK")


@pytest.mark.parametrize("damage", ["bundle", "receipt", "config", "inference", "producer", "header_consumer", "scope", "query", "task"])
def test_historical_lexical_refresh_refuses_changed_identity_or_assets(selected, monkeypatch, damage):
    s, c = selected, selected.case
    if damage == "bundle":
        path = c.repository / s.bundle["artifact"]
        path.write_bytes(path.read_bytes() + b" ")
    elif damage == "receipt":
        path = c.previous / "receipt.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif damage == "config":
        c.config_path.write_bytes(c.config_path.read_bytes() + b" ")
    elif damage == "inference":
        (c.previous / "inference.json").write_bytes(b"{}")
    elif damage == "producer":
        monkeypatch.setattr(source384, "_pins", lambda: {"producer": "changed"})
    elif damage == "header_consumer":
        monkeypatch.setattr(source384, "_header_pin", lambda: "0" * 64)
    elif damage == "scope":
        s.binding["scope_paths"] = ["foreign.py"]
    elif damage == "query":
        s.binding["query_sha256"] = "0" * 64
    else:
        s.binding["task_cid"] = "task:foreign"
    with pytest.raises(ValueError):
        _rebuild(s)
    assert not any(key == "infer" for key, _ in c.observations)


@pytest.mark.parametrize("tampered", [False, True])
def test_learned_refresh_historical_selection_precedes_model_loading(selected, monkeypatch, tampered):
    """Authored learned-pin stop proves the handoff; no learned model is loaded."""
    s, c = selected, selected.case
    binding = {name: "authored" for name in learned.FIELDS}
    binding.update(schema=learned.SCHEMA, policy=learned.POLICY, learned_embeddings=True,
        completion_authority=False, execution_authority=False, task_cid="task:one", task_id="TASK")
    calls = []
    def pins(repository, metadata, task_id, artifacts):
        calls.append((repository, metadata, task_id))
        raise ValueError("authored learned pin stop")
    monkeypatch.setattr(learned, "_load_pins", pins)
    monkeypatch.setattr(learned, "_LocalRouterModel", lambda *args: pytest.fail("model loaded"))
    if tampered:
        (c.previous / "inference.json").write_bytes(b"{}")
    callback = learned.published_learned_retrieval_rebuilder(repository=c.repository,
        bundle=s.bundle, binding=binding)
    with pytest.raises(learned.LearnedRetrievalUnavailable) as caught:
        callback(repository=c.repository, previous_snapshot=s.snapshot, previous_result=s.hits,
            query_text="answer", output=c.repository / ".runtime/new")
    assert bool(calls) is (not tampered)
    if not tampered:
        assert calls == [(c.repository, bundles._normalize(s.metadata), "TASK")]
        assert str(caught.value.__cause__) == "authored learned pin stop"
    assert caught.value.receipt["local_embedding_calls"] == 0
