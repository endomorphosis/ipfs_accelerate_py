"""Real offline initial indexes are reused after an authored planner proposal."""
import json

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original, _proposal_json
from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _version
from test.api.semantic_state.test_published_learned_retrieval import actual_learned


def test_learned_initial_index_reaches_planning_then_reuses_exact_pins(original, actual_learned, monkeypatch):
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    model = actual_learned["model"]
    indexed = prep.initial_context(state=state, model_snapshot=model, model_revision=model.name)
    assert indexed["learned_embeddings"] is True
    assert indexed["provider_calls"] == 0
    before = {name: (root / ".runtime/terminal-vectors" / name).read_bytes()
              for name in ("result.json", "model-manifest.json", "vectors.duckdb")}
    calls = []
    def authored_provider(prompt, **kwargs):
        calls.append(prompt)
        assert indexed["index_id"] in prompt
        assert indexed["semantic_root_cid"] in prompt
        return {"text": _proposal_json(prepared), "observation": {}, "execution_receipt": None}
    _version(monkeypatch)
    assert prep.plan(state, provider_callable=authored_provider)["qualified"] is True
    assert len(calls) == 1  # Explicit authored fixture; no LLM request.
    from benchmarks.agent_supervisor.container_coding import learned_vector_preflight
    monkeypatch.setattr(learned_vector_preflight, "qualify", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("postadmission context must not repeat embeddings")))
    monkeypatch.setattr(initial, "prepare_semantic_context", lambda **k: (_ for _ in ()).throw(
        AssertionError("postadmission context must not repeat semantic producer")))
    context = prep.context(state=state, model_snapshot=model, model_revision=model.name)
    assert context["initial_indexes_reused"] is True
    assert context["new_embedding_calls"] == 0
    assert context["index_id"] == indexed["index_id"]
    assert context["semantic_root_cid"] == indexed["semantic_root_cid"]
    assert context["world_snapshot_cid"] != indexed["world_snapshot_cid"]
    assert before == {name: (root / ".runtime/terminal-vectors" / name).read_bytes() for name in before}
    result = json.loads(before["result.json"])
    assert result["local_model_calls"] > 0
    assert result["policy"]["model_revision"] == model.name
    assert result["remote_provider_calls"] == 0
