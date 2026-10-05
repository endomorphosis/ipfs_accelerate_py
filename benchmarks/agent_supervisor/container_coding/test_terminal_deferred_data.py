"""Bounded data fetch references retain exact bytes, replay and immutable scope."""
from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding.test_terminal_data_profiles import mixed
from benchmarks.agent_supervisor.container_coding.test_terminal_task_profile import prepare
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import _proposal_json
from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _version
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
from benchmarks.agent_supervisor.container_coding import terminal_task_profile as profiles
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_context_runtime as semantic


@pytest.fixture
def captured(tmp_path):
    raw = b'{"request_id":"example","size":128}\n' * 2800
    args = mixed(tmp_path, media="application/x-ndjson", name="requests.jsonl", raw=raw)
    prepared = prepare(args)
    root, _, state, _ = args
    context = prep.initial_context(state=state)
    loaded = initial.load_initial_context(state=state, prepared=prepared, require_empty_owner=True)
    descriptor = loaded["descriptor"]
    return dict(root=root, state=state, prepared=prepared, context=context,
        payload=loaded["semantic"], raw=raw,
        load=dict(repository=root, artifact=descriptor["metadata"]["Semantic context artifact"],
            expected_sha256=descriptor["metadata"]["Semantic context sha256"], task_id=descriptor["task_alias"]))


def test_large_public_data_stays_exact_but_outside_bounded_prompt(captured):
    c = captured
    payload = c["payload"]
    text = semantic.load_semantic_worker_context(**c["load"])
    assert len(c["raw"]) > 90000 and len(text.encode()) <= 32768
    assert "requests.jsonl" not in payload["raw_sources"]
    reference = payload["task_data_projection"]["raw_source_fetch_required"]["requests.jsonl"]
    assert reference == dict(payload["manifest"]["requests.jsonl"], role="task_data", media_type="application/x-ndjson")
    assert reference["sha256"] == hashlib.sha256(c["raw"]).hexdigest()
    assert "requests.jsonl" in payload["worker_projection"]["raw_source_fetch_required"]
    blocks = (c["root"] / c["load"]["artifact"]).parent / "blocks"
    assert (blocks / reference["source_cid"]).read_bytes() == c["raw"]
    assert {profiles.INSTRUCTION, profiles.SMOKE, profiles.PROFILE} <= set(payload["raw_sources"])
    assert payload["task_data_projection"]["semantic_equivalence_claimed"] is False
    assert c["context"]["provider_calls"] == 0


@pytest.mark.parametrize("name", ["requests.jsonl", profiles.PROFILE, profiles.SMOKE, profiles.INSTRUCTION])
def test_immutable_data_and_support_drift_cannot_be_refreshed(captured, name):
    c = captured
    path = c["root"] / name
    path.write_bytes(path.read_bytes().replace(b"128", b"129") if name == "requests.jsonl"
        else path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        semantic.resolve_semantic_worker_context(**c["load"],
            refresh_output=c["root"] / ".runtime/refresh", attempt_id="attempt:drift")
    assert not (c["root"] / ".runtime/refresh").exists()


def test_program_refresh_preserves_identical_data_refs_and_required_bytes(captured):
    c = captured
    (c["root"] / "source.py").write_text("def public_source():\n    return 2\n")
    refreshed = semantic.resolve_semantic_worker_context(**c["load"],
        refresh_output=c["root"] / ".runtime/refresh", attempt_id="attempt:program")
    payload = json.loads(refreshed["text"])
    assert refreshed["refreshed"] is True
    assert payload["task_data_projection"] == c["payload"]["task_data_projection"]
    assert payload["raw_sources"] == c["payload"]["raw_sources"]
    assert set(payload["refresh_lineage"]["source_delta"]) == {"source.py"}
    assert len(refreshed["text"].encode()) <= 32768


@pytest.mark.parametrize("damage", ["missing_projection", "wrong_hash", "wrong_role", "missing_data", "program_data", "authority"])
def test_rehashed_payload_cannot_forge_or_omit_deferred_bindings(captured, damage):
    c = captured
    payload = deepcopy(c["payload"])
    projection = payload["task_data_projection"]
    refs = projection["raw_source_fetch_required"]
    if damage == "missing_projection": del payload["task_data_projection"]
    elif damage == "wrong_hash": refs["requests.jsonl"]["sha256"] = "0" * 64
    elif damage == "wrong_role": refs["requests.jsonl"]["role"] = "instruction"
    elif damage == "missing_data": refs.clear()
    elif damage == "program_data": refs["source.py"] = refs.pop("requests.jsonl")
    elif damage == "authority": projection["proof_authority"] = True
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    (c["root"] / c["load"]["artifact"]).write_bytes(raw)
    with pytest.raises(ValueError):
        semantic.load_semantic_worker_context(**{**c["load"], "expected_sha256":hashlib.sha256(raw).hexdigest()})


@pytest.mark.parametrize("reused", [False, True])
def test_admitted_context_preserves_deferred_data_with_or_without_initial_reuse(tmp_path, monkeypatch, reused):
    args = mixed(tmp_path, media="application/x-ndjson", name="requests.jsonl",
        raw=b'{"request_id":"example","size":128}\n' * 2800)
    prepared = prepare(args)
    if reused:
        prep.initial_context(state=args[2])
    _version(monkeypatch)
    proposal = json.loads(_proposal_json(prepared))
    proposal["tasks"][0]["predicted_files"] = [item["path"] for item in prepared["spec"]["outputs"]]
    planned = prep.plan(args[2], provider_callable=lambda *a, **k: dict(text=json.dumps(proposal),
        observation={}, execution_receipt=None))
    assert planned["qualified"], planned
    context = prep.context(state=args[2])
    assert context["initial_indexes_reused"] is reused
    semantic_file = args[0] / (".runtime/terminal-initial-context/semantic/worker-context.json"
        if reused else ".runtime/terminal-context/semantic/worker-context.json")
    payload = json.loads(semantic_file.read_bytes())
    assert "requests.jsonl" in payload["task_data_projection"]["raw_source_fetch_required"]
    assert "requests.jsonl" not in payload["raw_sources"]
    assert len(semantic_file.read_bytes()) <= 32768


def test_required_raw_data_cannot_be_deferred(captured):
    c = captured
    with pytest.raises(ValueError, match="exact public profile"):
        semantic.prepare_semantic_context(repository=c["root"], paths=c["prepared"]["worker_inputs"],
            required_raw_paths=[profiles.INSTRUCTION, profiles.SMOKE, "requests.jsonl"],
            program_paths=["source.py"], defer_task_data=True, worker_query="public task",
            objective="Public task", task_id="task", output=c["root"] / ".runtime/new-semantic")
    assert not (c["root"] / ".runtime/new-semantic").exists()
