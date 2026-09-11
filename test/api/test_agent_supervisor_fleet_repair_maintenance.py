"""Maintenance observations must never dispatch work or invent an idle claim."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import fleet_repair as repair
from ipfs_accelerate_py.agent_supervisor.rescue import fleet_repair_maintenance as maintenance
from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import write_json


def config(tmp_path):
    return {"state_dir": str(tmp_path), "boards": [
        {"id": "spar", "cwd": str(tmp_path), "hold_files": []},
        {"id": "sawm", "cwd": str(tmp_path), "hold_files": []}],
        "runtime_release": "/configured/normal/release", "repair_worker": {"cwd": str(tmp_path)}}


def inactive_runner(spec, **kwargs):
    unit = spec["argv"][3]
    return {"returncode": 0, "stdout": f"Id={unit}\nLoadState=loaded\nActiveState=inactive\nSubState=dead\nMainPID=0\n"}


def queued(cfg, identifier="spar", **extra):
    path = Path(cfg["state_dir"]) / "repairs" / identifier / "job.json"
    write_json(path, {"board_id": identifier, "status": "queued", **extra})
    return path


def test_due_unstarted_job_truthfully_waits_without_claim_or_selection(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    path = queued(cfg)
    before = path.read_bytes()
    for name in ("next_job", "run_job", "reconcile_queued_jobs", "enqueue"):
        monkeypatch.setattr(repair, name, lambda *a, **k: pytest.fail("maintenance called repair mutation"))
    result = maintenance.observe(cfg, runner=inactive_runner)
    assert result["status"] == "waiting" and result["no_unfinished_claims"]
    assert result["waiting"] == [{"board_id": "spar", "next_attempt_at": 0}]
    assert path.read_bytes() == before


@pytest.mark.parametrize("extra", [
    {"status": "running", "attempts": 1, "last_started_at": 1},
    {"attempts": 1, "last_started_at": 1},
    {"attempts": 1, "last_started_at": 2, "finished_at": 1},
    {"attempts": True}, {"attempts": 1}, {"finished_at": 1},
    {"attempts": 0, "last_started_at": 1, "finished_at": float('inf')},
    {"next_attempt_at": float('nan')}, {"status": "unknown"},
])
def test_unfinished_or_unknown_claim_cannot_become_idle(tmp_path, extra):
    cfg = config(tmp_path); path = queued(cfg, **extra); before = path.read_bytes()
    with pytest.raises(maintenance.ObservationBlocked):
        maintenance.observe(cfg, runner=inactive_runner)
    assert path.read_bytes() == before


def test_closed_attempt_and_held_due_job_are_waiting(tmp_path):
    cfg = config(tmp_path); hold = tmp_path / "HOLD"; hold.touch()
    cfg['boards'][0]['hold_files'] = [str(hold)]
    queued(cfg, attempts=2, last_started_at=1, finished_at=2, next_attempt_at=3)
    result = maintenance.observe(cfg, runner=inactive_runner)
    assert result['status'] == 'waiting' and not result['waiting'] and result['held'][0]['board_id'] == 'spar'


def test_active_and_unknown_units_never_idle(tmp_path):
    cfg = config(tmp_path)
    def running(spec, **kw):
        result = inactive_runner(spec, **kw)
        if spec['argv'][3] == maintenance.UNITS['repair_job']:
            result['stdout'] = result['stdout'].replace('ActiveState=inactive','ActiveState=active').replace('SubState=dead','SubState=running').replace('MainPID=0','MainPID=123')
        return result
    assert maintenance.observe(cfg, runner=running)['status'] == 'running'
    def unknown(spec, **kw):return {'returncode':1,'stdout':''}
    with pytest.raises(maintenance.ObservationBlocked):maintenance.observe(cfg, runner=unknown)


def test_changed_queue_and_symlink_cannot_supply_idle(tmp_path):
    cfg = config(tmp_path);path=queued(cfg);calls=[]
    def racing(spec, **kw):
        calls.append(1)
        if len(calls)==3:queued(cfg, next_attempt_at=10)
        return inactive_runner(spec, **kw)
    with pytest.raises(maintenance.ObservationBlocked, match='changed'):
        maintenance.observe(cfg, runner=racing)
    path.unlink();path.symlink_to(tmp_path/'missing')
    with pytest.raises(OSError):maintenance.observe(cfg, runner=inactive_runner)


def test_real_process_worker_flock_exclusion_and_inode_replacement(tmp_path):
    path=tmp_path/'repair-worker.lock';path.touch()
    code="import fcntl,os,sys;f=open(sys.argv[1],'r+');fcntl.flock(f,fcntl.LOCK_EX);print('ready',flush=True);sys.stdin.read()"
    child=subprocess.Popen([sys.executable,'-I','-c',code,str(path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=='ready'
        with pytest.raises(BlockingIOError):
            with maintenance.retained_dispatcher_lock(path):pytest.fail('stole dispatcher mutex')
    finally:
        child.stdin.close();child.wait(timeout=5);child.stdout.close()
    with maintenance.retained_dispatcher_lock(path) as current:
        code="import fcntl,sys;f=open(sys.argv[1],'r+');\ntry:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError:sys.exit(23)"
        assert subprocess.run([sys.executable,'-I','-c',code,str(path)],check=False).returncode==23
        replacement=tmp_path/'replacement';replacement.touch();replacement.replace(path)
        with pytest.raises(maintenance.ObservationBlocked,match='changed'):current()


def prepare_run(tmp_path, monkeypatch):
    cfg=config(tmp_path);path=tmp_path/'fleet.json';write_json(path,cfg)
    cfg['_config_path']=str(path.resolve())
    monkeypatch.setattr(maintenance,'load_config',lambda p:json.loads(p.read_text()))
    (tmp_path/'repair-worker.lock').touch()
    return cfg,path


def test_bounded_mode_reports_real_waiting_then_stopped_and_preserves_jobs(tmp_path, monkeypatch, capsys):
    cfg,path=prepare_run(tmp_path,monkeypatch);job=queued(cfg);before=job.read_bytes()
    for name in ('next_job','run_job','reconcile_queued_jobs'):
        monkeypatch.setattr(repair,name,lambda *a,**k:pytest.fail('unexpected dispatch'))
    assert maintenance.run(cfg,config_path=path,duration_seconds=1,interval_seconds=.2,
                           stop=threading.Event(),runner=inactive_runner)==0
    records=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records)>=3 and all(v['status']=='waiting' for v in records[:-1])
    assert records[-1]['status']=='maintenance_observer_stopped'
    assert all(v['mode']=='maintenance_observation' and v['dispatch_enabled'] is False for v in records)
    assert records[0]['configured_normal_runtime_release']=='/configured/normal/release'
    assert records[0]['observed_at']<records[-2]['observed_at']
    assert job.read_bytes()==before


def test_observation_error_publishes_blocked_and_busy_mutex_cannot_overwrite(tmp_path, monkeypatch, capsys):
    cfg,path=prepare_run(tmp_path,monkeypatch)
    def failed(*a,**kw):raise RuntimeError('unit read failed')
    maintenance.run(cfg,config_path=path,duration_seconds=1,interval_seconds=.2,stop=threading.Event(),runner=failed)
    records=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all(v['status']=='blocked' for v in records[:-1])
    output=tmp_path/'repair-worker.json';before=output.read_bytes()
    with maintenance.retained_dispatcher_lock(tmp_path/'repair-worker.lock'):
        with pytest.raises(BlockingIOError):
            maintenance.run(cfg,config_path=path,duration_seconds=1,stop=threading.Event(),runner=inactive_runner)
    assert output.read_bytes()==before


def test_config_loaded_object_cannot_rebind_new_bytes(tmp_path, monkeypatch):
    cfg,path=prepare_run(tmp_path,monkeypatch)
    changed=json.loads(path.read_text());changed['boards']=[];write_json(path,changed)
    with pytest.raises(maintenance.ObservationBlocked,match='retained bytes'):
        maintenance.run(cfg,config_path=path,duration_seconds=1,stop=threading.Event(),runner=inactive_runner)
    assert not (tmp_path/'repair-worker.json').exists()


@pytest.mark.skipif(os.environ.get('RUN_FLEET_MAINTENANCE_HOST_TESTS')!='1',reason='explicit disposable user-unit test')
def test_actual_active_user_unit_refuses_maintenance_idle(tmp_path):
    unit='fleet-maintenance-observer-test-'+uuid.uuid4().hex+'.service'
    subprocess.run(['systemd-run','--user','--quiet','--collect','--unit='+unit,
                    '--property=RuntimeMaxSec=10','/bin/sleep','3'],check=True)
    cfg=config(tmp_path)
    try:
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            if maintenance._unit_snapshot(unit)['MainPID']!='0':break
            time.sleep(.05)
        result=maintenance.observe(cfg,unit_names={'dispatcher':unit,'repair_job':unit})
        assert result['status']=='running' and result['units']['repair_job']['MainPID']!='0'
        # A separate absent transient job is allowed as closed; active normal dispatcher is not.
        absent='fleet-maintenance-observer-absent-'+uuid.uuid4().hex+'.service'
        result=maintenance.observe(cfg,unit_names={'dispatcher':unit,'repair_job':absent})
        assert result['status']=='blocked' and result['reason']=='normal_dispatcher_not_positively_inactive'
    finally:
        # The disposable unit exits naturally; no production service or signal is touched.
        deadline=time.monotonic()+12
        while time.monotonic()<deadline:
            result=subprocess.run(['systemctl','--user','show',unit,'--property=MainPID','--value'],capture_output=True,text=True)
            if result.stdout.strip()=='0':break
            time.sleep(.1)
        assert result.stdout.strip()=='0'


def test_actual_cli_selects_only_bounded_observer_with_normal_config_loader(tmp_path, monkeypatch, capsys):
    cfg=config(tmp_path)
    for board in cfg['boards']:board['probe']={'argv':['/bin/false']}
    path=tmp_path/'fleet.json';write_json(path,cfg);(tmp_path/'repair-worker.lock').touch()
    job=queued(cfg);original=job.read_bytes()
    for name in ('next_job','run_job','reconcile_queued_jobs'):
        monkeypatch.setattr(repair,name,lambda *a,**k:pytest.fail('CLI selected repair work'))
    original_run=maintenance.run
    monkeypatch.setattr(maintenance,'run',lambda c,**kw:original_run(c,**kw,runner=inactive_runner))
    assert repair.main(['observe-maintenance','--config',str(path),'--maintenance-duration-seconds','1',
                        '--maintenance-interval-seconds','.2'])==0
    records=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0]['status']=='waiting' and records[-1]['status']=='maintenance_observer_stopped'
    assert job.read_bytes()==original and path.read_text()==json.dumps(cfg,indent=2,sort_keys=True)+'\n'
    with pytest.raises(SystemExit):repair.main(['observe-maintenance','--config',str(path),'--once'])
