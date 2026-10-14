"""Bound actual selected capsules, preserving native indexes and signed inputs."""
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original
from benchmarks.agent_supervisor.container_coding.test_terminal_task_profile import profile
from benchmarks.agent_supervisor.container_coding.terminal_native_preflight import _authored_graph
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import resolve_semantic_worker_context
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


def prepared_budget(original, budget):
    root, instruction, state = original
    (root / "bottle.py").write_text("\n".join(
        f"def operation_{i}(value):\n    return value + {i}\n" for i in range(4)))
    return prep.prepare(repository=root, instruction=instruction, state=state,
        capsule_budget=budget,
        task_profile=profile(instruction.read_text(), inputs=("bottle.py",), output="bottle.py"))


@pytest.mark.parametrize("budget,limit", [("legacy", 8), ("one-capsule@1", 1), ("two-capsules@1", 2)])
def test_actual_initial_reuse_and_cold_refresh_preserve_cap_and_full_scope(original, budget, limit):
    root, instruction, state = original
    prepared = prepared_budget(original, budget)
    receipt = prep.initial_context(state=state, capsule_budget=budget)
    loaded = initial.load_initial_context(state=state, prepared=prepared, require_empty_owner=True)
    semantic = loaded["semantic"]
    assert semantic["worker_projection"]["max_capsules"] == limit
    assert len(semantic["capsules"]) == min(5, limit)
    # Semantic state includes the module; retrieval indexes its four functions.
    assert receipt["full_capsules"] == 5
    assert receipt["indexed_symbols"] == 4
    assert semantic["program_paths"] == ["bottle.py"]
    assert set(semantic["manifest"]) == set(prepared["worker_inputs"])
    assert semantic["raw_sources"][prep.INSTRUCTION] == prepared["query"]
    assert semantic["raw_sources"][prep.SMOKE] == (root / prep.SMOKE).read_text()
    assert receipt["provider_calls"] == 0
    assert receipt.get("capsule_budget", "legacy") == budget
    if budget == "legacy":
        assert all("capsule_budget" not in item for item in (prepared, receipt, loaded["descriptor"]))
    artifact = root / loaded["descriptor"]["metadata"]["Semantic context artifact"]
    original_payload = artifact.read_bytes()
    vector_artifact = root / ".runtime/terminal-vectors/result.json"
    original_index = vector_artifact.read_bytes()
    admission = local.admit_local_benchmark_plan(graph=_authored_graph(prepared), manifest=prepared["manifest"])
    prep._write(state / "admission.json", admission)
    with IntentRepository(state / "intent.duckdb") as intent:
        local.materialize_local_benchmark_plan(admission=admission, intent=intent)
    reused = prep.context(state=state)
    assert reused["initial_indexes_reused"] is True
    assert reused["new_embedding_calls"] == reused["provider_calls"] == 0
    assert reused["full_capsules"] == 5 and reused["worker_capsules"] == min(5, limit)
    assert reused.get("capsule_budget", "legacy") == budget
    assert artifact.read_bytes() == original_payload
    assert vector_artifact.read_bytes() == original_index
    assert prep._load_prepared(state)["manifest"] == prepared["manifest"]

    (root / "bottle.py").write_text((root / "bottle.py").read_text().replace("value + 0", "value + 10"))
    refreshed = resolve_semantic_worker_context(repository=root,
        artifact=artifact.relative_to(root).as_posix(), expected_sha256=hashlib.sha256(original_payload).hexdigest(),
        task_id=prepared["spec"]["task_key"], refresh_output=root / ".runtime/cold-refresh", attempt_id="authored-test")
    current = json.loads(refreshed["text"])
    assert refreshed["refreshed"] is True
    assert current["worker_projection"]["max_capsules"] == limit
    assert current["worker_projection"]["full_capsule_count"] == 5
    assert len(current["capsules"]) == min(5, limit)
    assert current["raw_sources"] == semantic["raw_sources"]
    assert current["required_raw_paths"] == semantic["required_raw_paths"]
    assert current["program_paths"] == semantic["program_paths"]
    assert set(current["manifest"]) == set(semantic["manifest"])
    assert current["semantic_root_cid"] != semantic["semantic_root_cid"]
    assert current["completion_authority"] is False
    assert artifact.read_bytes() == original_payload


@pytest.mark.parametrize("mutation", ["prepared", "receipt", "descriptor", "strip-all"])
def test_budget_declaration_drift_is_rejected_even_with_rehashed_descriptor(original, mutation):
    root, instruction, state = original
    prepared = prepared_budget(original, "one-capsule@1")
    receipt = prep.initial_context(state=state, capsule_budget="one-capsule@1")
    path = root / receipt["descriptor"]["artifact"]
    descriptor = json.loads(path.read_bytes())
    if mutation == "prepared":
        prepared["capsule_budget"] = "two-capsules@1"
    elif mutation == "receipt":
        receipt["capsule_budget"] = "two-capsules@1"
    elif mutation == "descriptor":
        descriptor["capsule_budget"] = "two-capsules@1"
    else:
        for item in (prepared, receipt, descriptor):
            del item["capsule_budget"]
    path.write_bytes(initial._bytes(descriptor))
    receipt["descriptor"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="capsule budget"):
        initial._load_initial_context(state=state, prepared=prepared, require_empty_owner=True, result=receipt)
    assert not (state / "planner-invoked.json").exists()


@pytest.mark.parametrize("requested", ["legacy", "two-capsules@1"])
def test_initial_cannot_silently_change_frozen_budget_before_indexing(original, monkeypatch, requested):
    root, instruction, state = original
    prepared_budget(original, "one-capsule@1")
    monkeypatch.setattr(initial, "prepare_initial_context", lambda **kw: pytest.fail("started indexing"))
    with pytest.raises(ValueError, match="frozen preparation"):
        prep.initial_context(state=state, capsule_budget=requested)
    assert not (root / ".runtime/terminal-vectors").exists()


def test_nonlegacy_context_cannot_skip_frozen_initial_projection(original):
    root, instruction, state = original
    prepared = prepared_budget(original, "one-capsule@1")
    admission = local.admit_local_benchmark_plan(graph=_authored_graph(prepared), manifest=prepared["manifest"])
    prep._write(state / "admission.json", admission)
    with IntentRepository(state / "intent.duckdb") as intent:
        local.materialize_local_benchmark_plan(admission=admission, intent=intent)
    with pytest.raises(ValueError, match="frozen initial context"):
        prep.context(state=state)
    assert not (root / ".runtime/terminal-vectors").exists()


def test_multitask_budget_refuses_before_repository_or_model_access(tmp_path):
    from benchmarks.agent_supervisor.container_coding.terminal_task_profile import MULTITASK_SCHEMA
    with pytest.raises(ValueError, match="one independent signed task"):
        prep.prepare(repository=tmp_path / "absent", instruction=tmp_path / "absent-instruction",
            state=tmp_path / "state", capsule_budget="one-capsule@1", task_profile={"schema": MULTITASK_SCHEMA})
    assert not (tmp_path / "state").exists()
