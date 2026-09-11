"""Cooperative original-parent retention never escalates incomplete work."""
from pathlib import Path
import ast
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.source_maintenance_drain import SourceMaintenanceDrain
from scripts import run_agent_supervisor_efficiency_state_hardening as native

ROOT=Path(__file__).resolve().parents[2]
FIXTURE=Path(__file__).parent/'fixtures/aseh_source_maintenance_process.py'


def wait_until(predicate,timeout=15):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if predicate(): return
        time.sleep(.03)
    raise AssertionError('fixture deadline')


@pytest.mark.parametrize('mode',['slow','permission','interrupt','closed_refusal','publish','finalizer'])
def test_actual_native_parent_retains_incomplete_owned_cohort(tmp_path,mode):
    with (tmp_path/'parent.stdout').open('w') as stdout,(tmp_path/'parent.stderr').open('w') as stderr:
        parent=subprocess.Popen([sys.executable,'-B',str(FIXTURE),str(tmp_path),'parent',mode],
            cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True,
            env={**os.environ,'PYTHONPATH':str(ROOT),'PYTHONDONTWRITEBYTECODE':'1'})
        descriptors=[]
        try:
            wait_until(lambda:(tmp_path/'admitted.json').exists() or parent.poll() is not None)
            assert parent.poll() is None,(tmp_path/'parent.stderr').read_text()
            identities=json.loads((tmp_path/'admitted.json').read_text())
            descriptors=[os.pidfd_open(identities[key]) for key in ('owner','worker')]
            wait_until(lambda:(tmp_path/'heartbeat.json').exists())
            old_heartbeat=json.loads((tmp_path/'heartbeat.json').read_text())['at']
            started=time.monotonic()
            # Real native 10-second outer deadline is crossed with healthy
            # work still running; do not accelerate or replace the native clock.
            hold_seconds=11.0 if mode=='slow' else .4
            while time.monotonic()-started<hold_seconds:
                os.kill(parent.pid,signal.SIGINT)
                os.kill(parent.pid,signal.SIGTERM)
                assert parent.poll() is None
                assert (tmp_path/'retained-capsule').is_file()
                assert not (tmp_path/'returned.json').exists()
                assert not (tmp_path/'cleanup.json').exists()
                assert not (tmp_path/'forced-cleanup.json').exists()
                time.sleep(.1)
            assert json.loads((tmp_path/'heartbeat.json').read_text())['at']>old_heartbeat
            assert not (tmp_path/'unexpected-signal.json').exists()
            (tmp_path/'finish-worker').touch()
            wait_until(lambda:(tmp_path/'owner-closed.json').exists())
            if mode in {'permission','interrupt','closed_refusal','publish'}:
                time.sleep(.3)
                assert parent.poll() is None
                assert (tmp_path/'retained-capsule').is_file()
                assert not (tmp_path/'cleanup.json').exists()
            (tmp_path/'allow-observation').touch()
            result=parent.wait(timeout=10)
            assert result==(1 if mode=='finalizer' else 0),(tmp_path/'parent.stderr').read_text()
            assert json.loads((tmp_path/'cleanup.json').read_text())=={'complete':True,'owner_returncode':0}
            assert not (tmp_path/'retained-capsule').exists()
            assert (tmp_path/'unknown-callback').read_text()=='started_outcome_unknown'
            assert not (tmp_path/'forced-cleanup.json').exists()
            assert not (tmp_path/'unexpected-signal.json').exists()
            if mode=='finalizer': assert not (tmp_path/'returned.json').exists()
            for descriptor in descriptors:
                poller=select.poll();poller.register(descriptor,select.POLLIN)
                assert poller.poll(0)
            for key in ('owner','worker'):
                assert not Path(f'/proc/{identities[key]}').exists()
        finally:
            (tmp_path/'finish-worker').touch()
            (tmp_path/'allow-observation').touch()
            try: parent.wait(timeout=10)
            except subprocess.TimeoutExpired:
                parent.kill();parent.wait(timeout=5)
            for descriptor in descriptors: os.close(descriptor)


def test_request_races_shutdown_without_stealing_admission(tmp_path):
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],start_new_session=True)
    drain=SourceMaintenanceDrain(admission_gate=lambda _:None,closed_gate=lambda _:None,publish=lambda _:None)
    try:
        drain.bind(child,start_time_ticks=native._dedicated_process_group_birth(child),source_head='a'*40,source_tree='b'*40)
        barrier=threading.Barrier(3)
        outcomes={}
        def request():
            barrier.wait()
            try: drain.request();outcomes['request']=True
            except RuntimeError: outcomes['request']=False
        def stop():
            barrier.wait();outcomes['stop']=drain.reserve_ordinary_shutdown()
        threads=[threading.Thread(target=request),threading.Thread(target=stop)]
        for thread in threads: thread.start()
        barrier.wait()
        for thread in threads: thread.join()
        assert outcomes['request'] != outcomes['stop']
        if drain.requested:
            with pytest.raises(RuntimeError,match='incomplete'):
                drain.close()
        child.terminate();child.wait(timeout=5)
        if drain.requested: assert drain.observe()['complete'] is True
        drain.close()
        with pytest.raises(RuntimeError,match='cannot admit'):
            drain.request()
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=5)


def test_old_native_helpers_and_unrelated_stop_semantics_are_preserved():
    path='scripts/run_agent_supervisor_efficiency_state_hardening.py'
    old=ast.parse(subprocess.run(['git','show',f'aa3a5b840acd5013007f243ccaa6368cd8605d2e:{path}'],cwd=ROOT,check=True,text=True,capture_output=True).stdout)
    current=ast.parse((ROOT/path).read_text())
    select_names=lambda tree:{node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
    old_nodes,current_nodes=select_names(old),select_names(current)
    for name in ('_stop_signal_handlers','_call_stop_signal_handlers','_dedicated_process_group_birth','_signal_dedicated_process_group','_terminate_dedicated_process_group','_terminate_scheduler'):
        assert ast.dump(old_nodes[name])==ast.dump(current_nodes[name])
    delegated=current_nodes['_run_supervisor_with_retained_child']
    main_try=next(node for node in delegated.body if isinstance(node,ast.Try))
    first_cleanup_call=main_try.finalbody[1]
    assert isinstance(first_cleanup_call,ast.Expr)
    assert isinstance(first_cleanup_call.value,ast.Call)
    assert first_cleanup_call.value.func.id=='_finish_source_maintenance_before_cleanup'


def test_ordinary_parent_wrapper_keeps_original_call_contract(monkeypatch):
    calls=[]
    monkeypatch.setattr(native,'_run_supervisor_with_retained_child',lambda *a,**kw:calls.append((a,kw)) or 7)
    assert native.run_supervisor(Path('config.json'),implement=True,duration=2)==7
    assert calls==[((Path('config.json'),),{'implement':True,'duration':2})]


def test_observation_publisher_cannot_mutate_completion_authority():
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],start_new_session=True)
    def publish(row):
        row['complete']=True
        row['callback_settlement_authority']=True
    drain=SourceMaintenanceDrain(admission_gate=lambda _:None,closed_gate=lambda _:None,publish=publish)
    try:
        drain.bind(child,start_time_ticks=native._dedicated_process_group_birth(child),source_head='a'*40,source_tree='b'*40)
        drain.request()
        observation=drain.observe()
        assert observation['complete'] is False
        assert observation['callback_settlement_authority'] is False
        assert drain.complete is False
        with pytest.raises(RuntimeError,match='incomplete'):
            drain.close()
        child.terminate();child.wait(timeout=5)
        assert drain.observe()['complete'] is True
        drain.close()
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=5)
