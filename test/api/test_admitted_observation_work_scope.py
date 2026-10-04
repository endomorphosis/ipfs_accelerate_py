"""The owner's full observation retains its original local work deadline.

These are deadline/factory controls, with no model or proof acceptance seam.
Native publication and successor inference are qualified separately.
"""
from concurrent.futures import ThreadPoolExecutor

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.runtime import header_intent_applicability as budget


@pytest.mark.parametrize('flag,refresh,bundle', [
    (True, False, None), (True, True, {}), (1, False, None), ('true', False, None),
])
def test_successor_requires_explicit_captured_local_work_before_state(tmp_path, flag, refresh, bundle):
    with pytest.raises(ValueError, match='Source384 refresh requires'):
        AdmittedBenchmarkRuntime.create(tmp_path / 'launch', admission=None, server=None, source=None,
            refresh_source384_on_completion=flag, refresh_context_on_completion=refresh,
            context_bundle=bundle)
    assert not (tmp_path / 'launch').exists()


def _runtime(scope, deadline, observer):
    runtime = object.__new__(AdmittedBenchmarkRuntime)
    runtime._replay_work_scope = scope
    runtime._replay_lifetime_deadline = deadline
    runtime._observe_in_replay_scope = observer
    return runtime


def test_full_observation_resumes_original_scope_in_new_thread(monkeypatch):
    monkeypatch.setenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE', 'local-benchmark@1')
    clock = [100.]
    monkeypatch.setattr(budget.time, 'monotonic', lambda: clock[0])
    with budget.local_benchmark_applicability_budget(deadline_monotonic=200.):
        scope = budget.capture_applicability_budget()
    seen = []
    def observer():
        seen.append((budget.applicability_replay_timeout(),
                     budget.capture_applicability_budget().deadline_monotonic))
        return {'published_context': []}
    runtime = _runtime(scope, 180., observer)
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(runtime.observe).result() == {'published_context': []}
        clock[0] = 150.
        assert pool.submit(runtime.observe).result() == {'published_context': []}
    assert seen == [(120., 180.), (120., 180.)]
    assert budget.capture_applicability_budget() is None
    assert budget.applicability_replay_timeout() == 45.


def test_observation_cannot_return_after_scope_expires(monkeypatch):
    monkeypatch.setenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE', 'local-benchmark@1')
    clock = [100.]
    monkeypatch.setattr(budget.time, 'monotonic', lambda: clock[0])
    with budget.local_benchmark_applicability_budget(deadline_monotonic=200.):
        scope = budget.capture_applicability_budget()
    def late_observer():
        clock[0] = 201.
        return {'published_context': ['must not be returned']}
    with pytest.raises(TimeoutError, match='before publication'):
        _runtime(scope, 250., late_observer).observe()
    assert budget.capture_applicability_budget() is None


def test_shorter_enclosing_observation_scope_remains_binding(monkeypatch):
    monkeypatch.setenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE', 'local-benchmark@1')
    monkeypatch.setattr(budget.time, 'monotonic', lambda: 100.)
    with budget.local_benchmark_applicability_budget(deadline_monotonic=200.):
        scope = budget.capture_applicability_budget()
    runtime = _runtime(scope, 180., lambda: budget.capture_applicability_budget().deadline_monotonic)
    with budget.applicability_budget(10.):
        assert runtime.observe() == 110.


def test_ordinary_observation_keeps_default_replay_ceiling():
    runtime = _runtime(None, 0., lambda: budget.applicability_replay_timeout())
    assert runtime.observe() == 45.
