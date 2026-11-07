"""Actual detached cleanup/hold fixtures; never touch a real lane or operator."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import repair_dispatcher as d


DIAGNOSTIC = r'''
from contextlib import contextmanager
import json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);stubborn=sys.argv[2]=='stubborn'
def identity():
 fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
 return {'pid':os.getpid(),'start_ticks':int(fields[19]),'uid':os.getuid()}
(root/'diagnostic.pid.json').write_text(json.dumps(identity()))
def stop(sig,frame):raise InterruptedError('owned diagnostic interrupted')
signal.signal(signal.SIGTERM,signal.SIG_IGN if stubborn else stop)
signal.signal(signal.SIGINT,stop)
@contextmanager
def hold():
 marker=root/'hold-marker';raw=b'owned private diagnostic fixture\n';marker.write_bytes(raw)
 info=marker.stat();expected=(info.st_dev,info.st_ino);path=root/'HOLD';os.link(marker,path)
 try:yield
 finally:
  previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM,signal.SIGINT})
  try:
   released=False
   if path.exists():
    current=path.lstat()
    if (current.st_dev,current.st_ino)==expected and path.read_bytes()==raw:
     path.unlink();released=True
   (root/'diagnostic-cleanup.json').write_text(json.dumps({'owned_hold_released':released,'fixture_only':True}))
  finally:signal.pthread_sigmask(signal.SIG_SETMASK,previous)
native=r"""
import json,os,signal,sys,time
from pathlib import Path
root=Path(sys.argv[1]);signal.signal(signal.SIGTERM,signal.SIG_IGN)
fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
(root/'native.pid.json').write_text(json.dumps({'pid':os.getpid(),'start_ticks':int(fields[19]),'uid':os.getuid()}))
(root/'native-ready').write_text('actual private process ready')
time.sleep(60)
"""
with hold():
 child=subprocess.Popen([sys.executable,'-c',native,str(root)],start_new_session=True,
                        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 try:child.wait()
 finally:
  previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM,signal.SIGINT})
  try:
   child.terminate()
   try:child.wait(timeout=.2)
   except subprocess.TimeoutExpired:child.kill();child.wait(timeout=3)
  finally:signal.pthread_sigmask(signal.SIG_SETMASK,previous)
'''


WORKER = r'''
import subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);mode=sys.argv[2]
subprocess.Popen([sys.executable,str(root/'diagnostic.py'),str(root),mode],start_new_session=True,
                 stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
until=time.monotonic()+10
while not (root/'native-ready').exists():
 if time.monotonic()>until:raise RuntimeError('fixture child did not start')
 time.sleep(.01)
# Remain observable across multiple actual tree snapshots before exit/timeout.
time.sleep(.5 if mode=='normal-exit' else 60)
'''


def config_for(root,mode='cooperative'):
    root.mkdir(parents=True,exist_ok=True)
    (root/'diagnostic.py').write_text(DIAGNOSTIC)
    (root/'worker.py').write_text(WORKER)
    (root/'prompt.txt').write_text('Fixture only; no task or proof authority.\n')
    config=d.Config(root/'manifest.json',root,root,prompt_path=root/'prompt.txt',
                    worker_argv=[sys.executable,str(root/'worker.py'),str(root),mode],worker_timeout=1.2)
    return config


def wait_ready(root,timeout=4):
    until=time.monotonic()+timeout
    while not (root/'native-ready').exists() and time.monotonic()<until:time.sleep(.01)
    assert (root/'native-ready').is_file()


def cleanup_fixture(root):
    for path in root.glob('*.pid.json'):
        record=json.loads(path.read_text());pid=record['pid'];current=d.probe_process(pid)
        if current.get('alive') and current.get('start_ticks')==record['start_ticks']:
            assert Path('/proc',str(pid)).stat().st_uid==os.getuid()==record['uid']
            try:
                fd=os.pidfd_open(pid)
                try:
                    if d.probe_process(pid).get('start_ticks')==record['start_ticks']:
                        signal.pidfd_send_signal(fd,signal.SIGKILL)
                finally:os.close(fd)
            except ProcessLookupError:pass


def assert_owned_children_gone(root):
    until=time.monotonic()+.5
    while True:
        alive=[]
        for name in ['diagnostic.pid.json','native.pid.json']:
            record=json.loads((root/name).read_text());current=d.probe_process(record['pid'])
            if current.get('alive') and current.get('start_ticks')==record['start_ticks']:alive.append(record)
        if not alive:return
        assert time.monotonic()<until,alive
        time.sleep(.01)


def test_worker_deadline_allows_exact_owned_hold_cleanup_and_preserves_unrelated_actor(tmp_path):
    config=config_for(tmp_path/'worker-timeout')
    sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
    try:
        result=d.run_worker(config,config.state_dir/'worker.log',lambda _:None)
        assert result['outcome']=='timed_out' and config.worker_timeout==1.2
        assert_owned_children_gone(config.repo_root)
        assert not (config.repo_root/'HOLD').exists()
        assert json.loads((config.repo_root/'diagnostic-cleanup.json').read_text())['owned_hold_released'] is True
        assert sentinel.poll() is None
    finally:
        cleanup_fixture(config.repo_root)
        sentinel.terminate();sentinel.wait(timeout=3)


def test_adopted_worker_deadline_has_same_bounded_descendant_cleanup(tmp_path):
    config=config_for(tmp_path/'adopted-timeout')
    worker=subprocess.Popen(config.worker_argv,cwd=config.repo_root,start_new_session=True,
                            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        wait_ready(config.repo_root)
        identity={**d.probe_process(worker.pid),'owned_session':True}
        assert d.stop_adopted_worker(identity)
        worker.wait(timeout=3)
        assert_owned_children_gone(config.repo_root)
        assert not (config.repo_root/'HOLD').exists()
        assert json.loads((config.repo_root/'diagnostic-cleanup.json').read_text())['owned_hold_released'] is True
    finally:
        if worker.poll() is None:worker.kill();worker.wait(timeout=3)
        cleanup_fixture(config.repo_root)


def test_natural_worker_exit_still_allows_observed_descendant_finally(tmp_path):
    config=config_for(tmp_path/'natural-exit','normal-exit')
    try:
        result=d.run_worker(config,config.state_dir/'worker.log',lambda _:None)
        assert result['outcome']=='awaiting_observation' and result['exit_code']==0
        assert_owned_children_gone(config.repo_root)
        assert not (config.repo_root/'HOLD').exists()
        assert json.loads((config.repo_root/'diagnostic-cleanup.json').read_text())['owned_hold_released'] is True
    finally:cleanup_fixture(config.repo_root)


def test_foreign_replaced_hold_is_preserved_by_real_diagnostic_finally(tmp_path):
    config=config_for(tmp_path/'foreign-hold');config.interval=.01
    foreign=[]
    def poll(_identity):
        hold=config.repo_root/'HOLD'
        if (config.repo_root/'native-ready').exists() and not foreign:
            hold.unlink();hold.write_text('independent user hold\n');foreign.append(hold.stat().st_ino)
    try:
        d.run_worker(config,config.state_dir/'worker.log',lambda _:None,on_poll=poll)
        hold=config.repo_root/'HOLD'
        assert foreign and hold.stat().st_ino==foreign[0] and hold.read_text()=='independent user hold\n'
        assert json.loads((config.repo_root/'diagnostic-cleanup.json').read_text())['owned_hold_released'] is False
        assert_owned_children_gone(config.repo_root)
    finally:cleanup_fixture(config.repo_root)


def test_uncooperative_descendant_is_bounded_without_heuristically_deleting_its_hold(tmp_path,monkeypatch):
    config=config_for(tmp_path/'uncooperative','stubborn')
    monkeypatch.setattr(d,'_DESCENDANT_CLEANUP_GRACE_SECONDS',.4)
    started=time.monotonic()
    try:
        result=d.run_worker(config,config.state_dir/'worker.log',lambda _:None)
        assert result['outcome']=='timed_out' and time.monotonic()-started<3
        assert_owned_children_gone(config.repo_root)
        assert (config.repo_root/'HOLD').is_file()
        assert not (config.repo_root/'diagnostic-cleanup.json').exists()
    finally:cleanup_fixture(config.repo_root)


def test_absent_descendants_do_not_pay_a_blind_grace_sleep(tmp_path):
    config=config_for(tmp_path/'no-descendants')
    config.worker_argv=[sys.executable,'-c','pass'];start=time.monotonic()
    result=d.run_worker(config,config.state_dir/'worker.log',lambda _:None)
    assert result['outcome']=='awaiting_observation' and time.monotonic()-start<1


def test_failed_exit_poll_still_kills_only_recorded_fixture_descendants(tmp_path,monkeypatch):
    config=config_for(tmp_path/'failed-exit-poll')
    def failed():raise OSError('injected private pidfd readiness failure')
    monkeypatch.setattr(d.select,'poll',failed)
    try:
        with pytest.raises(OSError,match='private pidfd readiness failure'):
            d.run_worker(config,config.state_dir/'worker.log',lambda _:None)
        assert_owned_children_gone(config.repo_root)
        # Forced failure cannot invent the diagnostic's finally/release receipt.
        assert not (config.repo_root/'diagnostic-cleanup.json').exists()
    finally:cleanup_fixture(config.repo_root)
