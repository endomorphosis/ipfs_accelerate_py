"""Real owned watchdog identity fixtures, never real lanes/providers."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import socket
import tempfile
import time
import pytest
from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume,watchdog_guardian
from ipfs_accelerate_py.agent_supervisor.rescue import repair_dispatcher as d
source_root=Path(lane_resume.__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('_lease_rotation_fixture',source_root/'test/api/test_agent_supervisor_repair_dispatcher.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
rig=f.rig

@pytest.fixture
def rotation(rig,monkeypatch):
    config,now,lane,live,original_probe=rig
    package=config.repo_root/'private-watchdog-source'
    directory=package
    for name in ['ipfs_accelerate_py','agent_supervisor','rescue']:
        directory/=name;directory.mkdir(parents=True);(directory/'__init__.py').write_text('')
    (directory/'supervisor_watchdog.py').write_text("import pathlib,time\npathlib.Path('fixture-ready').write_text('ready')\nwhile True: time.sleep(60)\n")
    command=[sys.executable,'-m',watchdog_guardian.WATCHDOG_MODULE,'--manifest-path',str(config.manifest_path),
             '--repo-root',str(config.repo_root),'--check-interval','120']
    env={**os.environ,'PYTHONPATH':str(package),'PYTHONDONTWRITEBYTECODE':'1'}
    processes=[]
    def spawn(*, omit_interval=False, poll_env=None):
        argv=command[:-2] if omit_interval else command
        environment=dict(env)
        if poll_env is not None:environment["WATCHDOG_CHECK_INTERVAL_SECONDS"]=poll_env
        child=subprocess.Popen(argv,cwd=config.repo_root,env=environment,start_new_session=True,
                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        processes.append(child);return child
    owner=spawn();deadline=time.monotonic()+3
    while not (config.repo_root/'fixture-ready').exists() and time.monotonic()<deadline:time.sleep(.01)
    assert owner.poll() is None and (config.repo_root/'fixture-ready').exists()
    marker=config.manifest_path.parent/'watchdog.pid';marker.write_text(str(owner.pid))
    live[200]=False
    f.status(lane,state='stopped',phase='lane_resume_exited',active_task_id='',terminal_reason='resume_duration_elapsed',heartbeat_at_ms=now*1000)
    report=config.repo_root/'runtime/report.json'
    def fresh(offset=0,**changes):
        value={'status':{'pid':owner.pid,'state':'healthy'},'reports':[]};value.update(changes)
        f.write_json(report,value);os.utime(report,(now+offset,now+offset))
    fresh()
    def probe(pid):
        return d.probe_process(pid) if pid in [p.pid for p in processes] else original_probe(pid)
    monkeypatch.setattr(d,'discover_watchdogs',watchdog_guardian.discover_watchdogs)
    try:yield config,now,lane,live,probe,owner,fresh,spawn
    finally:
        for child in processes:
            if child.poll() is None:child.terminate()
            child.wait(timeout=3)


def test_normal_expiry_waits_for_real_watchdog_without_spending_or_refunding_attempts(rotation):
    config,now,lane,live,probe,owner,fresh,spawn=rotation
    state={'attempts':{'old':{'count':2}},'unresolved_signature':'old','next_dispatch_at':now+500}
    result=d.tick(config,state,apply=True,now=now,probe=probe,worker=lambda *_:pytest.fail('premature maintenance'))
    assert result['action']=='native_lease_rotation_wait' and not result['incidents']
    wait=result['native_rotation_grace'][0]
    assert wait['watchdog_pid']==owner.pid and str(wait['watchdog_start_ticks'])==str(d.probe_process(owner.pid)['start_ticks'])
    assert wait['deadline']==pytest.approx(now+180,abs=.001) and state['attempts']['old']['count']==2
    assert state['unresolved_signature']=='old' and state['next_dispatch_at']==now+500
    assert not result['completion_authority'] and owner.poll() is None


def test_deadline_is_anchored_across_report_heartbeat_churn_and_dispatcher_restart(rotation):
    config,now,lane,live,probe,owner,fresh,spawn=rotation
    state={};first=d.observe(config,state,now=now,probe=probe)['native_rotation_grace'][0]
    state=json.loads(json.dumps(state));fresh(179)
    f.status(lane,heartbeat_at_ms=(now+170)*1000)
    result=d.observe(config,state,now=now+179,probe=probe)
    assert result['native_rotation_grace'][0]['deadline']==first['deadline']
    fresh(181)
    result=d.observe(config,state,now=now+181,probe=probe)
    assert not result.get('native_rotation_grace', []) and 'lane_dead_pending' in f.kinds(result)


def test_grace_expires_into_one_real_bounded_maintenance_child(rotation):
    config,now,lane,live,probe,owner,fresh,spawn=rotation
    config.worker_argv=[sys.executable,'-c',"from pathlib import Path;Path('maintenance-started').write_text('actual child')"]
    config.max_attempts=1;state={};d.tick(config,state,apply=True,now=now,probe=probe)
    fresh(181);result=d.tick(config,state,apply=True,now=now+181,probe=probe)
    assert result['action']=='worker_finished' and result['worker']['exit_code']==0
    assert (config.repo_root/'maintenance-started').read_text()=='actual child'
    assert state['attempts'][result['signature']]['count']==1
    fresh(182);assert d.tick(config,state,apply=True,now=now+182,probe=probe)['action']=='attention_required'
    assert '- Status: pending' in (config.repo_root/'todo.md').read_text()


def test_successive_expired_pid_has_its_own_bounded_rotation(rotation):
    config,now,lane,live,probe,owner,fresh,spawn=rotation
    state={};d.observe(config,state,now=now,probe=probe);fresh(120)
    (lane/'vog_bundle_supervisor.pid').write_text('201')
    f.status(lane,pid=201,heartbeat_at_ms=(now+120)*1000)
    result=d.observe(config,state,now=now+120,probe=probe)
    assert result['native_rotation_grace'][0]['expired_pid']==201
    assert result['native_rotation_grace'][0]['deadline']==pytest.approx(now+300,abs=.001)


def test_real_native_recovery_clears_rotation_observation(rotation):
    config,now,lane,live,probe,owner,fresh,spawn=rotation
    state={};d.observe(config,state,now=now,probe=probe);fresh(36)
    live[200]=True;f.status(lane,state='healthy',phase='deterministic_command',terminal_reason='',active_task_id='VOG-001',heartbeat_at_ms=(now+36)*1000)
    result=d.observe(config,state,now=now+36,probe=probe)
    assert not result.get('native_rotation_grace', []) and not state.get('lease_rotation_observations', {})
    assert 'lane_dead_pending' not in f.kinds(result)


def test_completed_idle_never_earns_rotation_or_dispatch(rotation):
    config,now,lane,live,probe,owner,fresh,spawn=rotation
    (config.repo_root/'todo.md').write_text('## VOG-001 Done\n- Status: completed\n- Depends on:\n')
    result=d.tick(config,{},apply=True,now=now,probe=probe,worker=lambda *_:pytest.fail('idle maintenance'))
    assert result['action']=='healthy' and not result.get('native_rotation_grace', [])


@pytest.mark.parametrize('reason',['crashed','','unexpected_exit'])
def test_crash_or_unknown_exit_has_no_rotation_grace(rotation,reason):
    config,now,lane,live,probe,owner,fresh,spawn=rotation
    f.status(lane,terminal_reason=reason)
    result=d.tick(config,{},now=now,probe=probe)
    assert not result.get('native_rotation_grace', []) and result['action']=='would_dispatch'
    assert 'lane_dead_pending' in f.kinds(result)


@pytest.mark.parametrize('failure',['dead','stale','wrong_report_pid','bad_start_ticks','failed_restart','missing_report','duplicate'])
def test_dead_stale_mismatched_or_failed_native_watchdog_does_not_delay_maintenance(rotation,monkeypatch,failure):
    config,now,lane,live,probe,owner,fresh,spawn=rotation
    if failure=='dead':owner.terminate();owner.wait(timeout=3)
    if failure=='stale':os.utime(config.repo_root/'runtime/report.json',(now-1000,now-1000))
    if failure=='wrong_report_pid':fresh(status={'pid':owner.pid+1,'state':'healthy'})
    if failure=='failed_restart':fresh(reports=[{'bundle_key':'vog','action':'restart_failed','pid_check':{'pid':200}}])
    if failure=='missing_report':(config.repo_root/'runtime/report.json').unlink()
    if failure=='duplicate':
        other=spawn();deadline=time.monotonic()+3
        while not d.probe_process(other.pid).get('argv') and time.monotonic()<deadline:time.sleep(.01)
    if failure=='bad_start_ticks':
        actual=watchdog_guardian.discover_watchdogs
        def changed(*args,**kwargs):
            values,errors=actual(*args,**kwargs)
            return [{**p,'start_ticks':str(int(p['start_ticks'])+1)} for p in values],errors
        monkeypatch.setattr(d,'discover_watchdogs',changed)
    result=d.tick(config,{},now=now,probe=probe)
    assert not result.get('native_rotation_grace', []) and result['action']=='would_dispatch'
    assert 'lane_dead_pending' in f.kinds(result)


def test_already_running_maintenance_is_not_canceled_or_refunded_by_rotation_grace(rotation):
    config,now,lane,live,probe,owner,fresh,spawn=rotation
    worker=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    identity=d.probe_process(worker.pid)
    state={'active_worker':{**identity,'started_at':now,'deadline':now+60},'attempts':{'spent':{'count':1}}}
    def actual(pid):return d.probe_process(pid) if pid==worker.pid else probe(pid)
    try:
        result=d.tick(config,state,apply=True,now=now,probe=actual)
        assert result['action']=='worker_already_running' and worker.poll() is None
        assert state['attempts']['spent']['count']==1
    finally:
        worker.terminate();worker.wait(timeout=3)


@pytest.mark.parametrize('setting,expected',[('300',345),('600',600),('nan',None)])
def test_actual_watchdog_environment_poll_setting_is_bounded_and_used_without_logging_secrets(rotation,setting,expected):
    config,now,lane,live,probe,owner,fresh,spawn=rotation
    owner.terminate();owner.wait(timeout=3)
    replacement=spawn(omit_interval=True,poll_env=setting)
    (config.manifest_path.parent/'watchdog.pid').write_text(str(replacement.pid))
    fresh(status={'pid':replacement.pid,'state':'healthy'})
    result=d.tick(config,{},now=now,probe=probe)
    if expected is None:
        assert not result.get('native_rotation_grace', []) and 'lane_dead_pending' in f.kinds(result)
    else:
        assert result['action']=='native_lease_rotation_wait'
        assert result['native_rotation_grace'][0]['deadline']==pytest.approx(now+expected,abs=.001)
    assert 'WATCHDOG_CHECK_INTERVAL_SECONDS' not in json.dumps(result)


def test_malformed_persisted_rotation_clock_cannot_crash_or_extend_terminal_heartbeat_grace(rotation):
    config,now,lane,live,probe,owner,fresh,spawn=rotation
    result=d.tick(config,{'lease_rotation_observations':None},now=now,probe=probe)
    assert result['action']=='native_lease_rotation_wait'
    assert result['native_rotation_grace'][0]['deadline']==pytest.approx(now+180,abs=.001)


@pytest.fixture
def notify_socket(monkeypatch):
    with tempfile.TemporaryDirectory(prefix='repair-notify-') as temporary:
        with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as receiver:
            path=str(Path(temporary)/'notify.sock');receiver.bind(path);receiver.settimeout(.04)
            monkeypatch.setenv('NOTIFY_SOCKET',path)
            monkeypatch.setenv('WATCHDOG_PID',str(os.getpid()))
            monkeypatch.setenv('WATCHDOG_USEC','120000000')
            yield receiver,path


def test_real_private_systemd_datagram_targets_only_own_pid(notify_socket):
    receiver,path=notify_socket
    assert d.SERVICE_WATCHDOG_VERSION==1 and d._notify_control_watchdog()
    assert receiver.recv(1024)==b'WATCHDOG=1'


@pytest.mark.parametrize('name,value',[
    ('WATCHDOG_PID',None),('WATCHDOG_PID','wrong'),('WATCHDOG_PID','1'),
    ('WATCHDOG_USEC',None),('WATCHDOG_USEC','0'),('WATCHDOG_USEC','nan'),
    ('NOTIFY_SOCKET',None),('NOTIFY_SOCKET','relative.sock')])
def test_absent_wrong_pid_or_invalid_notify_configuration_never_sends(notify_socket,monkeypatch,name,value):
    receiver,path=notify_socket
    if value is None:monkeypatch.delenv(name,raising=False)
    else:monkeypatch.setenv(name,value)
    assert not d._notify_control_watchdog()
    with pytest.raises(socket.timeout):receiver.recv(1024)


def test_optional_notification_delivery_failure_does_not_raise(notify_socket,monkeypatch):
    receiver,path=notify_socket
    monkeypatch.setenv('NOTIFY_SOCKET',path+'.absent')
    assert not d._notify_control_watchdog()


def test_adopted_active_worker_tick_publishes_control_heartbeat_without_refunding_or_extending(rotation,monkeypatch):
    config,now,lane,live,probe,owner,fresh,spawn=rotation
    worker=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)
    identity=d.probe_process(worker.pid)
    state={'active_worker':{**identity,'started_at':now,'deadline':now+90},
           'attempts':{'spent':{'count':2}},'unresolved_signature':'spent','next_dispatch_at':now+30}
    def actual(pid):return d.probe_process(pid) if pid==worker.pid else probe(pid)
    clock=[now];monkeypatch.setattr(d.time,'time',lambda:clock[0])
    try:
        first=d.tick(config,state,apply=True,now=now,probe=actual)
        clock[0]=now+1
        second=d.tick(config,state,apply=True,now=now+1,probe=actual)
        saved=json.loads((config.state_dir/'status.json').read_text())
        assert first['control_heartbeat_at']<second['control_heartbeat_at']==saved['control_heartbeat_at']
        assert second['action']=='worker_already_running' and second['last_full_observation_at']==now+1
        assert state['active_worker']['deadline']==now+90 and state['attempts']['spent']['count']==2
        assert state['unresolved_signature']=='spent' and state['next_dispatch_at']==now+30
        assert 'recovery_observed_at' not in state and worker.poll() is None
    finally:
        worker.terminate();worker.wait(timeout=3)


@pytest.mark.parametrize('mode',['held','cooldown'])
def test_held_and_cooldown_completed_ticks_publish_control_heartbeat(rig,monkeypatch,mode):
    config,now,lane,live,probe=rig
    live[200]=False
    observation=d.observe(config,{},now=now,probe=probe)
    signature=observation['signature']
    state={'attempts':{signature:{'count':1,'next_dispatch_at':now+80}},
           'unresolved_signature':signature,'next_dispatch_at':now+80}
    if mode=='held':
        config.state_dir.mkdir(parents=True);(config.state_dir/'hold').write_text('operator')
    clock=[now];monkeypatch.setattr(d.time,'time',lambda:clock[0])
    first=d.tick(config,state,apply=True,now=now,probe=probe)
    clock[0]+=1
    second=d.tick(config,state,apply=True,now=now+1,probe=probe)
    assert second['action']==mode and second['control_heartbeat_at']>first['control_heartbeat_at']
    assert state['attempts'][signature]['count']==1 and state['next_dispatch_at']==now+80


def test_real_worker_poll_heartbeat_retains_last_full_observation_and_original_budget(rig,notify_socket,monkeypatch):
    config,now,lane,live,probe=rig;receiver,path=notify_socket
    live[200]=False;config.interval=.01;config.worker_timeout=2
    config.worker_argv=[sys.executable,'-c','import time;time.sleep(.6)']
    state={};snapshots=[];notify=d._notify_control_watchdog
    def notified():
        snapshots.append((json.loads((config.state_dir/'status.json').read_text()),json.loads(json.dumps(state))))
        return notify()
    monkeypatch.setattr(d,'_notify_control_watchdog',notified)
    result=d.tick(config,state,apply=True,now=now,probe=probe,
                  worker=lambda cfg,log,started:d._control_worker(cfg,state,log,started))
    assert len(snapshots)>=2
    for telemetry,durable in snapshots:
        assert receiver.recv(1024)==b'WATCHDOG=1'
        assert telemetry['action']=='worker_running' and telemetry['control_phase']=='worker_poll'
        assert telemetry['observed_at']==telemetry['last_full_observation_at']==now
        assert durable['active_worker']['deadline']==now+2
        assert durable['attempts'][result['signature']]['count']==1
        assert durable['unresolved_signature']==result['signature'] and 'recovery_observed_at' not in durable
    assert snapshots[-1][0]['control_heartbeat_at']>snapshots[0][0]['control_heartbeat_at']
    assert result['action']=='worker_finished' and result['worker']['exit_code']==0
    assert not state.get('active_worker') and 'recovery_observed_at' not in state


def test_blocked_worker_poll_has_no_background_heartbeat(rig,notify_socket,monkeypatch):
    config,now,lane,live,probe=rig;receiver,path=notify_socket
    config.interval=.01;config.worker_timeout=2
    config.worker_argv=[sys.executable,'-c','import time;time.sleep(.5)']
    config.state_dir.mkdir(parents=True)
    original=subprocess.Popen.wait;checks=[];messages=[]
    def blocked(process,*args,**kwargs):
        if not checks:
            checks.append(True);time.sleep(.08)
            with pytest.raises(socket.timeout):receiver.recv(1024)
        return original(process,*args,**kwargs)
    monkeypatch.setattr(subprocess.Popen,'wait',blocked)
    result=d.run_worker(config,config.state_dir/'worker.log',lambda _:None,
                        on_poll=lambda _:messages.append(d._notify_control_watchdog()))
    assert checks and any(messages) and result['exit_code']==0


def test_observation_error_preserves_prior_full_observation_timestamp(rig,monkeypatch):
    config,now,lane,live,probe=rig;state={}
    d.tick(config,state,now=now,probe=probe)
    def failed(*args,**kwargs):raise ValueError('actual observation did not complete')
    monkeypatch.setattr(d,'observe',failed)
    result=d.tick(config,state,now=now+20,probe=probe)
    assert result['action']=='observation_error' and result['last_full_observation_at']==now
    assert result['control_heartbeat_at'] and not result['completion_authority']


def test_actual_main_sigkill_preserves_owned_worker_for_deadline_bound_adoption(rig,notify_socket):
    config,now,lane,live,probe=rig;receiver,path=notify_socket
    source=Path(d.__file__).resolve()
    script='''
import importlib.util,json,os,sys
from pathlib import Path
import ipfs_accelerate_py.agent_supervisor.rescue as package
name=package.__name__+'.repair_dispatcher'
spec=importlib.util.spec_from_file_location(name,sys.argv[1]);d=importlib.util.module_from_spec(spec)
sys.modules[name]=d;package.repair_dispatcher=d;spec.loader.exec_module(d)
root=Path(sys.argv[2]);manifest=Path(sys.argv[3]);real_probe=d.probe_process
def probe(pid):
 if pid==100:return {'pid':100,'alive':True,'age_seconds':5000,'start_ticks':1000,'cwd':str(root),'argv':['python','-m','ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog','--manifest-path',str(manifest),'--repo-root',str(root)]}
 if pid==200:return {'pid':200,'alive':False}
 return real_probe(pid)
d.discover_watchdogs=lambda *_:([],[])
tick=d.tick;d.tick=lambda cfg,state,**kw:tick(cfg,state,probe=probe,**kw)
os.environ['WATCHDOG_PID']=str(os.getpid())
raise SystemExit(d.main(['--manifest-path',str(manifest),'--repo-root',str(root),'--state-dir',sys.argv[4],
 '--apply','--prompt-path',sys.argv[5],'--worker-argv-json',json.dumps([sys.executable,'-c','import time;time.sleep(20)']),
 '--worker-timeout','10','--interval','.02']))
'''
    environment={**os.environ,'PYTHONPATH':str(source_root),'PYTHONDONTWRITEBYTECODE':'1'}
    child=subprocess.Popen([sys.executable,'-c',script,str(source),str(config.repo_root),str(config.manifest_path),
                            str(config.state_dir),str(config.prompt_path)],env=environment,
                           stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,start_new_session=True)
    active=None
    try:
        receiver.settimeout(3);assert receiver.recv(1024)==b'WATCHDOG=1'
        durable=json.loads((config.state_dir/'state.json').read_text());active=durable['active_worker']
        assert active['owned_session'] and d.probe_process(active['pid'])['alive']
        child.kill();child.wait(timeout=3)
        assert d.probe_process(active['pid'])['alive']
        def actual(pid):return d.probe_process(pid) if pid==active['pid'] else probe(pid)
        result=d.tick(config,durable,apply=True,now=time.time(),probe=actual,
                      worker=lambda *_:pytest.fail('duplicate replacement worker'))
        assert result['action']=='worker_already_running' and durable['active_worker']==active
        assert sum(record['count'] for record in durable['attempts'].values())==1
        assert 'recovery_observed_at' not in durable and result['control_heartbeat_at']
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=3)
        if active:d.stop_adopted_worker(active)
