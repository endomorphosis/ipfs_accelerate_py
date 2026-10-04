"""Closed refresh diagnostics at the real observation routing boundaries.

Owner checks, graph construction and derivative work are explicit authored
seams here. These controls do not qualify native proof or model execution.
"""
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import PromptGoalGraph
from ipfs_accelerate_py.agent_supervisor.runtime import published_task_context as contexts
from ipfs_accelerate_py.agent_supervisor.runtime import published_retrieval as lexical
from ipfs_accelerate_py.agent_supervisor.runtime import published_learned_retrieval as learned


@pytest.fixture
def observation_route(tmp_path, monkeypatch):
    runtime = object.__new__(AdmittedBenchmarkRuntime)
    runtime.admission = {"manifest": {"payload": {"schema": "authored-observation-route"}}, "graph": {}}
    runtime.manifest = {"context_refresh_policy": {"trigger": "after_native_stop",
        "max_attempts_per_task": 1, "output_root": ".runtime/published"}}
    runtime.source = SimpleNamespace(get_task=lambda cid: SimpleNamespace(status="completed"))
    runtime._context_refresh_stopped = lambda: True
    runtime._published_context = {}
    runtime._context_refresh_attempts = {}
    runtime.repository = tmp_path
    runtime.context_bundle = {"artifact": ".runtime/predecessor", "sha256": "a" * 64}
    runtime.server = object()
    calls = []
    monkeypatch.setattr(AdmittedBenchmarkRuntime.__mro__[1], "observe",
        lambda self: calls.append("owner_observation") or {"canonical_task_mutated": False})
    monkeypatch.setattr(PromptGoalGraph, "from_dict", classmethod(lambda cls, value:
        SimpleNamespace(tasks=[SimpleNamespace(task_cid="authored-task")])))
    runtime._record = lambda name, value: calls.append((name, value))
    return runtime, calls


@pytest.mark.parametrize("boundary,policy,error_class", [
    ("rebuilder_selection", "lexical-tfidf-symbols@1", ValueError),
    ("rebuilder_selection", "local-safetensors-symbols@1", RuntimeError),
    ("cache_reload", None, TimeoutError),
    ("successor_refresh", None, ValueError),
])
def test_refresh_failure_preserves_closed_phase_and_original_class(
        observation_route, monkeypatch, boundary, policy, error_class):
    runtime, calls = observation_route
    private_body = "authored private exception body that must not be published"
    def refusal(**kwargs):
        calls.append(boundary)
        raise error_class(private_body)
    def forbidden(**kwargs):
        pytest.fail("work continued beyond the refused boundary")
    monkeypatch.setattr(contexts, "refresh_published_task_context", forbidden)
    monkeypatch.setattr(contexts, "load_published_task_context", forbidden)
    if policy is not None:
        runtime.manifest["published_retrieval_policy"] = {"policy": policy}
        owner, name = ((lexical, "published_retrieval_rebuilder")
            if policy.startswith("lexical") else (learned, "published_learned_retrieval_rebuilder"))
        monkeypatch.setattr(owner, name, refusal)
    elif boundary == "cache_reload":
        previous = {"refresh_artifact": ".runtime/cached", "refresh_sha256": "b" * 64}
        runtime._published_context["authored-task"] = previous
        monkeypatch.setattr(contexts, "load_published_task_context", refusal)
    else:
        monkeypatch.setattr(contexts, "refresh_published_task_context", refusal)
    result = runtime._observe_in_replay_scope()
    assert result["published_context"] == [{"task_cid": "authored-task", "status": "unavailable",
        "error_type": error_class.__name__, "error_phase": boundary, "completion_authority": False}]
    assert calls[:2] == ["owner_observation", boundary]
    assert calls[-1][0] == "published-context"
    assert private_body not in repr(result) and private_body not in repr(calls[-1])
    assert result["canonical_task_mutated"] is False
    if boundary == "cache_reload":
        assert runtime._published_context == {"authored-task": previous}
        assert runtime._context_refresh_attempts == {}
    else:
        assert runtime._published_context == {}
        assert runtime._context_refresh_attempts == {"authored-task": 1}
        again = runtime._observe_in_replay_scope()
        assert again["published_context"] == [{"task_cid": "authored-task", "status": "retry_budget_exhausted"}]
        assert calls.count(boundary) == 1


@pytest.mark.parametrize("status,stopped,expected", [
    ("in_progress", True, "pending_completion"),
    ("completed", False, "pending_native_stop"),
])
def test_unfinished_or_unstopped_work_does_not_enter_derivative_boundary(
        observation_route, monkeypatch, status, stopped, expected):
    runtime, calls = observation_route
    runtime.source.get_task = lambda cid: SimpleNamespace(status=status)
    runtime._context_refresh_stopped = lambda: stopped
    def forbidden(**kwargs):
        pytest.fail("derivative work ran before completion and native STOP")
    monkeypatch.setattr(contexts, "refresh_published_task_context", forbidden)
    monkeypatch.setattr(contexts, "load_published_task_context", forbidden)
    row = runtime._observe_in_replay_scope()["published_context"][0]
    assert row["status"] == expected and "error_phase" not in row
    assert runtime._context_refresh_attempts == {} and runtime._published_context == {}
    assert calls[0] == "owner_observation"


def test_nonordinary_cancellation_is_not_turned_into_unavailable(observation_route, monkeypatch):
    runtime, calls = observation_route
    class Cancelled(BaseException):
        pass
    def cancelled(**kwargs):
        raise Cancelled()
    monkeypatch.setattr(contexts, "refresh_published_task_context", cancelled)
    with pytest.raises(Cancelled):
        runtime._observe_in_replay_scope()
    assert calls == ["owner_observation"]
    assert runtime._published_context == {}
