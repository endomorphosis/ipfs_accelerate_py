"""Regression windows observed in the real interrupted supervisor run."""
from pathlib import Path
import os
import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import ordinary_provider_cleanup as module

@pytest.mark.parametrize('error',[FileNotFoundError,ProcessLookupError])
def test_exited_process_stat_is_absent(monkeypatch,error):
    def gone(*args,**kwargs):raise error('owned process exited')
    monkeypatch.setattr(Path,'read_text',gone)
    assert module.OrdinaryProviderCleanupObserver.birth(99999991) is None

def test_process_permission_failure_is_not_an_absence(monkeypatch):
    def denied(*args,**kwargs):raise PermissionError('identity cannot be observed')
    monkeypatch.setattr(Path,'read_text',denied)
    with pytest.raises(PermissionError):module.OrdinaryProviderCleanupObserver.birth(99999991)

@pytest.mark.parametrize('error',[FileNotFoundError,ProcessLookupError])
def test_exited_thread_children_are_ignored_without_sticky_error(tmp_path,monkeypatch,error):
    observer=module.OrdinaryProviderCleanupObserver.__new__(module.OrdinaryProviderCleanupObserver)
    observer.owned={99999991:123};observer.bindings={};observer.unresolved=None
    observer.attempt=tmp_path;observer.temporary_root=tmp_path/'provider'
    monkeypatch.setattr(observer,'birth',lambda pid:(1,123,'S'))
    children=Path('/proc/99999991/task/99999991/children')
    monkeypatch.setattr(Path,'glob',lambda self,pattern:iter([children]))
    def gone(*args,**kwargs):raise error('owned thread exited')
    monkeypatch.setattr(Path,'read_text',gone)
    observer.observe()
    assert observer.unresolved is None and observer.bindings=={}
    assert (tmp_path/'provider-docker-observations.json').is_file()

def test_thread_permission_failure_remains_fail_closed(tmp_path,monkeypatch):
    observer=module.OrdinaryProviderCleanupObserver.__new__(module.OrdinaryProviderCleanupObserver)
    observer.owned={99999991:123};observer.bindings={};observer.unresolved=None
    observer.attempt=tmp_path;observer.temporary_root=tmp_path/'provider'
    monkeypatch.setattr(observer,'birth',lambda pid:(1,123,'S'))
    monkeypatch.setattr(Path,'glob',lambda self,pattern:iter([Path('/proc/99999991/task/1/children')]))
    def denied(*args,**kwargs):raise PermissionError('cannot observe child ownership')
    monkeypatch.setattr(Path,'read_text',denied)
    with pytest.raises(PermissionError):observer.observe()

def test_reused_pid_is_skipped_before_child_or_commandline_reads(tmp_path,monkeypatch):
    observer=module.OrdinaryProviderCleanupObserver.__new__(module.OrdinaryProviderCleanupObserver)
    observer.owned={99999991:123};observer.bindings={};observer.unresolved=None
    observer.attempt=tmp_path;observer.temporary_root=tmp_path/'provider'
    monkeypatch.setattr(observer,'birth',lambda pid:(1,124,'S'))
    def forbidden(*args,**kwargs):raise AssertionError('foreign PID metadata was traversed')
    monkeypatch.setattr(Path,'glob',forbidden)
    observer.observe()
    assert observer.bindings=={}
