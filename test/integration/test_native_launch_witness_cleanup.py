"""Native launch witnesses survive private /proc without creating authority."""
from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import Operation
from ipfs_accelerate_py.agent_supervisor.control.control_plane import PartialMutationError
from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import LifecycleOrchestrator
from ipfs_accelerate_py.agent_supervisor.entrypoints.isolated_benchmark_runtime import (
    IsolatedBenchmarkRuntime, NativeSupervisorHealthAdapter,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedSupervisorHealthAdapter
from test.api.test_agent_supervisor_lifecycle_orchestrator import _profile, _request


@pytest.fixture
def launched(tmp_path):
    children = []
    def popen(*args, **kwargs):
        child = subprocess.Popen(*args, **kwargs)
        children.append(child)
        return child
    profile = _profile(tmp_path, argv=(sys.executable, '-c', 'import time; time.sleep(30)'))
    adapter = AdmittedSupervisorHealthAdapter(popen=popen)
    try:
        yield profile, adapter, children
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


def hide_after_verified_launch(adapter, monkeypatch):
    hidden = set()
    launch, environ = adapter.launch, adapter._environ
    def observed_launch(profile, *, fencing_epoch):
        identity = launch(profile, fencing_epoch=fencing_epoch)
        hidden.add(identity.pid)
        return identity
    def private(pid):
        if pid in hidden:
            raise PermissionError('authored private process boundary')
        return environ(pid)
    monkeypatch.setattr(adapter, 'launch', observed_launch)
    monkeypatch.setattr(adapter, '_environ', private)


def test_failed_start_fences_launch_witness_when_environment_becomes_private(launched, monkeypatch):
    profile, adapter, children = launched
    hide_after_verified_launch(adapter, monkeypatch)
    orchestrator = LifecycleOrchestrator(state_root=profile.state_root, profiles=(profile,),
        process_adapter=adapter, clock_ms=lambda: 1000, poll_interval_ms=5, stop_grace_ms=5)
    with pytest.raises(PartialMutationError, match='startup did not prove sustained health'):
        orchestrator.start(_request(profile, operation=Operation.START, deadline_ms=200))
    assert len(children) == 1
    children[0].wait(timeout=2)
    assert not adapter.snapshot(profile).members


def test_private_launch_is_visible_but_never_substitutes_for_child_bootstrap(launched, monkeypatch):
    profile, adapter, _children = launched
    hide_after_verified_launch(adapter, monkeypatch)
    identity = adapter.launch(profile, fencing_epoch=9)
    tree = adapter.snapshot(profile)
    assert tree.members == (identity,)
    assert not adapter.healthy(profile, tree, fencing_epoch=9, now_ms=1000)
    assert not getattr(adapter, '_authenticated_children', {})


@pytest.mark.parametrize('field', ['parent', 'group', 'session', 'birth', 'argv'])
def test_private_launch_witness_rejects_identity_drift(launched, monkeypatch, field):
    profile, adapter, _children = launched
    hide_after_verified_launch(adapter, monkeypatch)
    identity = adapter.launch(profile, fencing_epoch=9)
    stat = adapter._stat
    if field == 'argv':
        monkeypatch.setattr(adapter, '_argv', lambda pid: ('different executable',))
    else:
        def changed(pid):
            observed = list(stat(pid))
            if pid == identity.pid:
                observed[['parent', 'group', 'session', 'birth'].index(field)] += 1
            return tuple(observed)
        monkeypatch.setattr(adapter, '_stat', changed)
    assert not adapter.snapshot(profile).members


def test_close_refuses_hidden_live_owned_child_before_releasing_lease():
    released = []
    runtime = IsolatedBenchmarkRuntime.__new__(IsolatedBenchmarkRuntime)
    runtime._children = [SimpleNamespace(poll=lambda: None)]
    runtime.profile = object()
    runtime.process = SimpleNamespace(snapshot=lambda profile: SimpleNamespace(members=()))
    runtime.lease = SimpleNamespace(fencing_token='fence', fence_epoch=1)
    runtime.coordinator = SimpleNamespace(release=lambda *a, **k: released.append('release'),
                                          close=lambda: released.append('close'))
    with pytest.raises(RuntimeError, match='live launched child'):
        runtime.close()
    assert released == []
