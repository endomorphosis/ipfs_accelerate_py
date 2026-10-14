"""Long literal instructions fit bounded indexed context without being truncated."""
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
from benchmarks.agent_supervisor.container_coding import terminal_semantic_context_policy as policy
from benchmarks.agent_supervisor.container_coding import terminal_task_profile as profiles
from benchmarks.agent_supervisor.container_coding.test_terminal_data_profiles import mixed


def long_instruction_case(tmp_path):
    args = mixed(tmp_path, media="application/x-ndjson", name="requests.jsonl",
        raw=b'{"request_id":"public","size":128}\n' * 2800)
    root, instruction, state, profile = args
    text = ("Preserve every original request and all exact values.\n" * 84
            + "Write result.py using public_source and the public request data.\n")
    assert 4300 < len(text.encode()) < 5000
    instruction.write_text(text)
    profile["instruction_sha256"] = profiles.instruction_sha256(text)
    (root / "source.py").write_text("\n".join(
        f"def public_source_{i}(value):\n    return value + {i}\n" for i in range(12)))
    return root, instruction, state, profile


@pytest.mark.parametrize("mode", ["indexed-default@1", "indexed-compact@1"])
def test_long_public_instruction_keeps_literal_body_and_bounded_context(tmp_path, mode):
    root, instruction, state, profile = long_instruction_case(tmp_path)
    prepared = prep.prepare(repository=root, instruction=instruction, state=state,
        task_profile=profile, disable_intent_autoencoder=True, semantic_context_profile=mode)
    receipt = prep.initial_context(state=state)
    loaded = initial.load_initial_context(state=state, prepared=prepared, require_empty_owner=True)
    semantic = loaded["semantic"]
    assert receipt["worker_payload_bytes"] <= 32768
    if mode == "indexed-compact@1":
        assert receipt["worker_capsules"] == 2
    else:
        assert 2 < receipt["worker_capsules"] <= 8
    assert semantic["raw_sources"][prep.INSTRUCTION] == instruction.read_text()
    assert semantic["worker_projection"]["query"] == instruction.read_text()
    assert semantic["raw_sources"][profiles.SMOKE] == (root / profiles.SMOKE).read_text()
    assert semantic["raw_sources"][profiles.PROFILE] == (root / profiles.PROFILE).read_text()
    assert semantic["objective"] == policy.semantic_context_objective(mode, instruction.read_text())
    assert semantic["pack"]["pack"]["objective"] == semantic["objective"]
    assert len(semantic["objective"].encode()) < 160
    assert hashlib.sha256(instruction.read_bytes()).hexdigest() in semantic["objective"]
    assert "requests.jsonl" not in semantic["raw_sources"]
    assert receipt["provider_calls"] == 0


def test_initial_replay_rejects_valid_capsules_with_foreign_objective(tmp_path, monkeypatch):
    root, instruction, state, profile = long_instruction_case(tmp_path)
    prepared = prep.prepare(repository=root, instruction=instruction, state=state,
        task_profile=profile, disable_intent_autoencoder=True, semantic_context_profile="indexed-compact@1")
    original = policy.semantic_context_objective
    calls = []
    def nominate_foreign_objective(mode, query):
        calls.append((mode, query))
        return "Foreign objective" if len(calls) == 1 else original(mode, query)
    monkeypatch.setattr(policy, "semantic_context_objective", nominate_foreign_objective)
    with pytest.raises(ValueError, match="objective differs from its literal public instruction"):
        prep.initial_context(state=state)
    assert len(calls) == 2
    semantic_path = root / ".runtime/terminal-initial-context/semantic/worker-context.json"
    descriptor_path = root / ".runtime/terminal-initial-context/descriptor.json"
    raw = semantic_path.read_bytes()
    descriptor = json.loads(descriptor_path.read_bytes())
    assert descriptor["metadata"]["Semantic context sha256"] == hashlib.sha256(raw).hexdigest()
    assert json.loads(raw)["objective"] == "Foreign objective"
    assert json.loads(raw)["pack"]["pack"]["objective"] == "Foreign objective"
    assert not (state / "initial-context-result.json").exists()
    assert not (state / "planner-invoked.json").exists()


def test_legacy_objective_retains_whitespace_normalization_only():
    query = "Public task.\n\nPreserve literal café and numbers 0.055.\n"
    assert policy.semantic_context_objective("legacy-bounded@1", query) == " ".join(query.split())
    indexed = policy.semantic_context_objective("indexed-default@1", query)
    assert indexed == policy.semantic_context_objective("indexed-compact@1", query)
    assert indexed != policy.semantic_context_objective("indexed-default@1", query + "\n")


def test_initial_replay_rejects_coherently_rewritten_ranking_query(tmp_path, monkeypatch):
    root, instruction, state, profile = long_instruction_case(tmp_path)
    prepared = prep.prepare(repository=root, instruction=instruction, state=state,
        task_profile=profile, disable_intent_autoencoder=True, semantic_context_profile="indexed-compact@1")
    produce = initial.prepare_semantic_context
    def foreign_query(**options):
        return produce(**{**options, "worker_query": "Different public_source ranking request"})
    monkeypatch.setattr(initial, "prepare_semantic_context", foreign_query)
    with pytest.raises(ValueError, match="query differs from its literal public instruction"):
        prep.initial_context(state=state)
    semantic_path = root / ".runtime/terminal-initial-context/semantic/worker-context.json"
    descriptor = json.loads((root / ".runtime/terminal-initial-context/descriptor.json").read_bytes())
    raw = semantic_path.read_bytes()
    assert descriptor["metadata"]["Semantic context sha256"] == hashlib.sha256(raw).hexdigest()
    value = json.loads(raw)
    assert value["raw_sources"][prep.INSTRUCTION] == instruction.read_text()
    assert value["objective"] == policy.semantic_context_objective("indexed-compact@1", instruction.read_text())
    assert value["worker_projection"]["query"] != instruction.read_text()
    assert not (state / "initial-context-result.json").exists()
    assert not (state / "planner-invoked.json").exists()
