"""Opted-in queue expiry never grants authority over unknown/manual holds."""
import copy
import errno
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.rescue import repair_dispatcher as d,queue_hold_lease as q


def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));path.chmod(0o600)


@pytest.fixture
def rig(tmp_path,monkeypatch):
 now=10000.0;root=tmp_path;runtime=root/'runtime';lane=runtime/'lane';lane.mkdir(parents=True)
 manifest=runtime/'bundle_lanes.json';write(manifest,{'todo_path':'todo.md','task_header_prefix':'## VOG-','report_path':'runtime/report.json','lanes':[{'bundle_key':'vog','state_dir':'runtime/lane','state_prefix':'vog'}]})
 (root/'todo.md').write_text('## VOG-001 Work\n- Status: pending\n- Depends on:\n')
 (runtime/'watchdog.pid').write_text('100');(lane/'vog_bundle_supervisor.pid').write_text('200')
 write(lane/'vog_status.json',dict(pid=200,state='healthy',phase='deterministic_command',active_task_id='VOG-001',heartbeat_at_ms=now*1000,last_task_transition_at_ms=now*1000))
 (lane/'supervisor.log').write_text('fixture\n');write(runtime/'report.json',{'reports':[{'bundle_key':'vog','action':'none'}]})
 cfg=d.Config(manifest,root,runtime/'dispatcher',repair_lock_path=runtime/'fixer.lock',cooldown_seconds=10)
 state={'schema':d.SCHEMA,'attempts':{'original':{'count':2,'next_dispatch_at':0}},'next_dispatch_at':0}
 write(cfg.state_dir/'state.json',state)
 def probe(pid):
  argv=(['python','-m','ipfs_accelerate_py.agent_supervisor.rescue.lane_resume','--repo-root',str(root),'--todo-path','todo.md','--bundle-key','vog','--status-path',str(lane/'vog_status.json'),'--pid-path',str(lane/'vog_bundle_supervisor.pid')] if pid==200 else ['python','-m','ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog','--manifest-path',str(manifest),'--repo-root',str(root)])
  return dict(pid=pid,alive=pid in (100,200,300),start_ticks=pid*10,age_seconds=5000,argv=argv,cwd=str(root),pgid=pid,session_id=pid)
 monkeypatch.setattr(d,'discover_watchdogs',lambda *_:([],[]))
 real_cap=q.require_live_queue_hold_capability
 monkeypatch.setattr(q,'require_live_queue_hold_capability',lambda *_:dict(version=1,identity_verified=True))
 stopped=[];monkeypatch.setattr(d,'stop_adopted_worker',lambda a:stopped.append(a) or True)
 return cfg,now,state,probe,stopped,real_cap


def register(rig,ttl=10):
 cfg,now,*_=rig
 return q.register_queue_hold(cfg,'bounded fixture maintenance',ttl,clock=lambda:now)


def adoption(rig):
 cfg,now,state,probe,stopped,_=rig
 state['active_worker']=dict(pid=300,start_ticks=3000,owned_session=True,pgid=300,session_id=300,started_at=now-100,deadline=now-1)
 write(cfg.state_dir/'state.json',state)
 return state


def test_expired_optin_hold_is_archived_and_no_longer_blocks_dispatch(rig):
 cfg,now,state,probe,stopped,_=rig;lease=register(rig)
 result=d.tick(cfg,state,apply=True,now=now+11,probe=probe)
 assert not (cfg.state_dir/'hold').exists() and not (cfg.state_dir/'hold.lease.json').exists()
 assert result['held'] is False and result['action']!='held'
 assert Path(lease['archive_hold_path']).stat().st_ino==lease['inode']
 import hashlib
 assert hashlib.sha256(Path(lease['archive_hold_path']).read_bytes()).hexdigest()==lease['body_sha256']
 assert json.loads(Path(lease['archive_lease_path']).read_text())['token']==lease['token']
 assert Path(lease['archive_lease_path']).is_file() and not stopped


def test_expired_queue_cannot_extend_adopted_original_deadline(rig):
 cfg,now,state,probe,stopped,_=rig;register(rig);adoption(rig);before=copy.deepcopy(state['attempts'])
 result=d.tick(cfg,state,apply=True,now=now+11,probe=probe)
 assert result['action']=='adopted_worker_timed_out' and len(stopped)==1
 assert stopped[0]['deadline']==now-1 and state['attempts']==before and not state.get('active_worker')
 assert (cfg.state_dir/'hold').exists() # physical mutation waits until the worker is gone
 assert state['next_dispatch_at']==now+11+cfg.cooldown_seconds


@pytest.mark.parametrize('other',['root','lane','owner_uncertain'])
def test_expired_queue_does_not_logically_ignore_any_other_hold(rig,monkeypatch,other):
 cfg,now,state,probe,stopped,_=rig;register(rig);adoption(rig)
 if other=='root':(cfg.repo_root/'HOLD').write_text('manual\n')
 elif other=='lane':(cfg.manifest_path.parent/'lane/hold').write_text('manual\n')
 else:
  (cfg.manifest_path.parent/'watchdog.pid').unlink()
  monkeypatch.setattr(d,'discover_watchdogs',lambda *_:([],['uncertain fixture']))
 result=d.tick(cfg,state,apply=True,now=now+11,probe=probe)
 assert result['action']=='held' and not stopped and state['active_worker']['deadline']==now-1
 assert (cfg.state_dir/'hold').exists()


def test_unexpired_lease_and_observational_expiry_do_not_clear_marker(rig):
 cfg,now,state,probe,stopped,_=rig;lease=register(rig);raw=(cfg.state_dir/'hold').read_bytes()
 value=d.tick(cfg,state,apply=True,now=now+9,probe=probe)
 assert value['held'] is True and value['action']=='held'
 result=q.expire_queue_hold(cfg,state,now+11,apply=False)
 assert result['authenticated_expired'] and (cfg.state_dir/'hold').read_bytes()==raw
 assert json.loads((cfg.state_dir/'hold.lease.json').read_text())['expires_at']==lease['expires_at']


@pytest.mark.parametrize('fault',['schema','scope','token','bool_device','inode','sha','mtime','extra','long_ttl','future','nan','huge'])
def test_invalid_or_future_lease_never_clears_or_hides_hold(rig,fault):
 cfg,now,state,probe,stopped,_=rig;register(rig);path=cfg.state_dir/'hold.lease.json';value=json.loads(path.read_text())
 if fault=='schema':value['schema']='manual'
 elif fault=='scope':value['scope']='worker_pause'
 elif fault=='token':value['token']='../unsafe'
 elif fault=='bool_device':value['device']=True
 elif fault=='inode':value['inode']+=1
 elif fault=='sha':value['body_sha256']='f'*64
 elif fault=='mtime':value['mtime_ns']+=1
 elif fault=='extra':value['archive_path']='/tmp/untrusted'
 elif fault=='long_ttl':value['expires_at']=value['created_at']+1801
 elif fault=='future':value['created_at']=now+100;value['expires_at']=now+101
 elif fault=='nan':value['created_at']=float('nan')
 else:value['created_at']=10**400
 write(path,value);identity=(cfg.state_dir/'hold').stat().st_ino
 result=d.tick(cfg,state,apply=True,now=now+11,probe=probe)
 assert result['held'] and result['action']=='held' and (cfg.state_dir/'hold').stat().st_ino==identity
 assert not stopped


@pytest.mark.parametrize('fault',['replace','mutate','extra_link','sidecar_symlink','sidecar_link','fifo_lease','fifo_hold'])
def test_replaced_unknown_link_symlink_or_fifo_is_preserved_without_blocking_read(rig,fault):
 cfg,now,state,probe,stopped,_=rig;register(rig);hold=cfg.state_dir/'hold';lease=cfg.state_dir/'hold.lease.json'
 if fault=='replace':hold.unlink();hold.write_text('manual replacement\n')
 elif fault=='mutate':hold.write_text('manual changed same inode\n')
 elif fault=='extra_link':os.link(hold,cfg.state_dir/'unknown-extra-link')
 elif fault=='sidecar_symlink':raw=lease.read_bytes();lease.unlink();target=cfg.state_dir/'foreign';target.write_bytes(raw);target.chmod(0o600);lease.symlink_to(target)
 elif fault=='sidecar_link':os.link(lease,cfg.state_dir/'unknown-descriptor-link')
 elif fault=='fifo_lease':lease.unlink();os.mkfifo(lease,0o600)
 else:hold.unlink();os.mkfifo(hold,0o600)
 started=time.monotonic();value=q.review_queue_hold(cfg,now+11);assert time.monotonic()-started<1
 assert value['status']=='invalid' and not value['authenticated_expired']
 assert (hold.exists() or hold.is_symlink()) and lease.exists()


def test_legacy_manual_hold_without_sidecar_is_never_upgraded(rig):
 cfg,now,state,probe,stopped,_=rig;hold=cfg.state_dir/'hold';hold.write_text('old manual hold\n')
 identity=hold.stat();value=d.tick(cfg,state,apply=True,now=now+100000,probe=probe)
 assert value['held'] and hold.read_text()=='old manual hold\n' and hold.stat().st_ino==identity.st_ino
 assert not (cfg.state_dir/'hold.lease.json').exists() and not stopped


@pytest.mark.parametrize('fault',['active_key','spawn_key','missing_state','bad_state','wrong_schema','fifo_state','repair_busy'])
def test_physical_expiry_requires_valid_actual_quiescence_and_native_lock(rig,fault):
 cfg,now,state,probe,stopped,_=rig;register(rig);path=cfg.state_dir/'state.json'
 if fault=='active_key':write(path,dict(state,active_worker={}))
 elif fault=='spawn_key':write(path,dict(state,spawn_intent={}))
 elif fault=='missing_state':path.unlink()
 elif fault=='bad_state':path.write_text('{broken')
 elif fault=='wrong_schema':write(path,{'schema':'unknown'})
 elif fault=='fifo_state':path.unlink();os.mkfifo(path,0o600)
 if fault=='repair_busy':
  with d.repair_lock(cfg.repair_lock_path):value=q.expire_queue_hold(cfg,state,now+11,apply=True)
 else:value=q.expire_queue_hold(cfg,state,now+11,apply=True)
 assert value['retirement']!='archived' and (cfg.state_dir/'hold').exists()
 assert not stopped


def test_missing_configured_native_marker_cannot_register_or_logically_optin(rig):
 cfg,now,state,probe,stopped,_=rig;register(rig);cfg.repair_lock_path=None
 assert not q.review_queue_hold(cfg,now+11)['authenticated_expired']
 with pytest.raises(ValueError):q.register_queue_hold(cfg,'invalid config',10,clock=lambda:now)


@pytest.mark.parametrize('ttl',[0,1801,True,float('nan'),10**400])
def test_invalid_registration_bound_has_no_marker_or_lease(rig,ttl):
 cfg,now,*_=rig
 with pytest.raises(ValueError):q.register_queue_hold(cfg,'bad bound',ttl,clock=lambda:now)
 assert not (cfg.state_dir/'hold').exists() and not (cfg.state_dir/'hold.lease.json').exists()


@pytest.mark.parametrize('stage',['before_sidecar','after_sidecar','after_live'])
def test_interrupted_registration_never_leaves_live_unregistered_hold(rig,monkeypatch,stage):
 cfg,now,state,probe,stopped,_=rig;real=os.link
 proxy=SimpleNamespace(**vars(os))
 def link(source,target,*args,**kwargs):
  target=Path(target)
  if stage=='before_sidecar' and target.name=='hold.lease.json':raise InterruptedError('fixture before sidecar')
  result=real(source,target,*args,**kwargs)
  if (stage=='after_sidecar' and target.name=='hold.lease.json') or (stage=='after_live' and target.name=='hold'):
   raise InterruptedError('fixture after durable namespace operation')
  return result
 proxy.link=link;monkeypatch.setattr(q,'os',proxy)
 with pytest.raises(InterruptedError):register(rig)
 live=cfg.state_dir/'hold';sidecar=cfg.state_dir/'hold.lease.json'
 if stage=='before_sidecar':
  assert not live.exists() and not sidecar.exists()
 else:
  assert sidecar.exists() and q.review_queue_hold(cfg,now+11)['authenticated_expired']
  if live.exists():assert live.stat().st_nlink==2
  monkeypatch.setattr(q,'os',os)
  value=q.expire_queue_hold(cfg,state,now+11,apply=True)
  assert value['retirement']=='archived' and not live.exists() and not sidecar.exists()


def test_expired_before_live_link_is_refused_and_reserved_state_is_recoverable(rig):
 cfg,now,state,probe,stopped,_=rig;clock=iter([now,now+10])
 with pytest.raises(ValueError,match='expired'):q.register_queue_hold(cfg,'late creation',10,clock=lambda:next(clock))
 assert not (cfg.state_dir/'hold').exists()
 assert q.review_queue_hold(cfg,now+11)['authenticated_expired']
 assert q.expire_queue_hold(cfg,state,now+11,apply=True)['retirement']=='archived'
 assert not (cfg.state_dir/'hold.lease.json').exists()


@pytest.mark.parametrize('stage',['archive_hold','archive_lease','after_hold_unlink'])
def test_interrupted_retirement_keeps_exact_archives_and_finishes_on_retry(rig,monkeypatch,stage):
 cfg,now,state,probe,stopped,_=rig;lease=register(rig);real=os.link;proxy=SimpleNamespace(**vars(os));failed=[]
 def link(source,target,*args,**kwargs):
  result=real(source,target,*args,**kwargs)
  if not failed and ((stage=='archive_hold' and str(target)==lease['archive_hold_path']) or (stage=='archive_lease' and str(target)==lease['archive_lease_path'])):
   failed.append(True);raise InterruptedError('fixture after archive link')
  return result
 proxy.link=link;monkeypatch.setattr(q,'os',proxy)
 old=q._unlink_registered
 def unlink(path,review,descriptor=False):
  result=old(path,review,descriptor)
  if stage=='after_hold_unlink' and not failed and path.name=='hold':failed.append(True);raise InterruptedError('fixture after live unlink')
  return result
 monkeypatch.setattr(q,'_unlink_registered',unlink)
 first=q.expire_queue_hold(cfg,state,now+11,apply=True)
 assert first['retirement']=='refused' and failed
 assert Path(lease['archive_hold_path']).stat().st_ino==lease['inode']
 assert q.review_queue_hold(cfg,now+11)['authenticated_expired']
 second=q.expire_queue_hold(cfg,state,now+11,apply=True)
 assert second['retirement'] in ('archived','released_descriptor_retired') and not (cfg.state_dir/'hold').exists() and not (cfg.state_dir/'hold.lease.json').exists()


def test_other_old_archives_preserved_and_current_token_collision_refuses_creation(rig,monkeypatch):
 cfg,now,*_=rig;archive=cfg.state_dir/'expired-queue-holds';archive.mkdir(mode=0o700)
 old=archive/('b'*32+'.hold');old.write_text('old unrelated archive\n');old.chmod(0o600)
 token='a'*32;collision=archive/(token+'.hold');collision.write_text('manual archive conflict\n');collision.chmod(0o600)
 monkeypatch.setattr(q,'uuid',SimpleNamespace(uuid4=lambda:SimpleNamespace(hex=token)))
 with pytest.raises(FileExistsError):register(rig)
 assert collision.read_text()=='manual archive conflict\n' and old.read_text()=='old unrelated archive\n'
 assert not (cfg.state_dir/'hold').exists() and not (cfg.state_dir/'hold.lease.json').exists()


def test_unknown_replaced_archive_is_not_overwritten_or_live_hold_cleared(rig):
 cfg,now,state,*_=rig;lease=register(rig);target=Path(lease['archive_hold_path']);target.write_text('foreign archive\n');target.chmod(0o600)
 assert not q.review_queue_hold(cfg,now+11)['authenticated_expired']
 result=q.expire_queue_hold(cfg,state,now+11,apply=True)
 assert (cfg.state_dir/'hold').exists() and target.read_text()=='foreign archive\n'


def test_early_legacy_publisher_release_does_not_block_next_registration(rig):
 cfg,now,state,*_=rig;first=register(rig)
 proof=q.publisher_queue_hold_proof(cfg,first,now=now+1)
 assert set(proof)=={'schema','path','device','inode','sha256','token','release_archive'}
 assert proof['schema']=='jevops.repair-queue-hold-lease.v1'
 live=cfg.repo_root/proof['path'];target=cfg.repo_root/proof['release_archive'];os.link(live,target);live.unlink()
 # Descriptor-only finalization cannot change an active worker's native presence.
 with d.repair_lock(cfg.repair_lock_path):
  raw=cfg.repair_lock_path.read_bytes()
  second=register(rig)
  assert cfg.repair_lock_path.read_bytes()==raw
 assert second['token']!=first['token'] and Path(first['archive_lease_path']).exists()
 assert target.stat().st_ino==first['inode']


def live_capability_fixture(rig,monkeypatch):
 import sys
 cfg,now,state,probe,stopped,real=rig;pid=os.getpid();ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
 argv=[sys.executable,'-m','ipfs_accelerate_py.agent_supervisor.rescue.repair_dispatcher','--manifest-path',str(cfg.manifest_path),'--repo-root',str(cfg.repo_root),'--state-dir',str(cfg.state_dir),'--repair-lock-path',str(cfg.repair_lock_path),'--apply']
 status=dict(schema=d.SCHEMA,control_pid=pid,control_start_ticks=ticks,control_heartbeat_at=now,queue_hold_lease_version=1,apply=True)
 current=dict(pid=pid,alive=True,start_ticks=ticks,argv=argv,cwd=str(cfg.repo_root))
 monkeypatch.setattr(d,'probe_process',lambda _:dict(current))
 write(cfg.state_dir/'status.json',status)
 return cfg,now,status,current,real


def test_live_capability_requires_bound_current_applying_controller(rig,monkeypatch):
 cfg,now,status,current,real=live_capability_fixture(rig,monkeypatch)
 value=real(cfg,now)
 assert value['identity_verified'] and value['version']==1


@pytest.mark.parametrize('damage',['old_version','bool_version','stale','future','birth','cwd','module','repo','manifest','state','repairlock','no_apply','status_observational','once','script_data','command_data','duplicate','uid'])
def test_live_capability_rejects_old_unverified_or_nonapplying_process(rig,monkeypatch,damage):
 cfg,now,status,current,real=live_capability_fixture(rig,monkeypatch)
 if damage=='old_version':status['queue_hold_lease_version']=0
 elif damage=='bool_version':status['queue_hold_lease_version']=True
 elif damage=='stale':status['control_heartbeat_at']=now-121
 elif damage=='future':status['control_heartbeat_at']=now+1
 elif damage=='birth':current['start_ticks']+=1
 elif damage=='cwd':current['cwd']='/foreign'
 elif damage=='module':current['argv'][2]='unrelated.module'
 elif damage in ('repo','manifest','state','repairlock'):
  key={'repo':'--repo-root','manifest':'--manifest-path','state':'--state-dir','repairlock':'--repair-lock-path'}[damage]
  current['argv'][current['argv'].index(key)+1]='/foreign'
 elif damage=='no_apply':current['argv'].remove('--apply')
 elif damage=='status_observational':status['apply']=False
 elif damage=='once':current['argv'].append('--once')
 elif damage=='script_data':current['argv'].insert(1,'unrelated.py')
 elif damage=='command_data':current['argv'][1:1]=['-c','pass']
 elif damage=='duplicate':current['argv']+=['--state-dir',str(cfg.state_dir)]
 else:status['control_pid']=1;current['pid']=1
 write(cfg.state_dir/'status.json',status)
 with pytest.raises((ValueError,OSError)):real(cfg,now)


def test_registration_default_gate_refuses_before_creating_any_queue_files(rig,monkeypatch):
 cfg,now,state,probe,stopped,real=rig
 monkeypatch.setattr(q,'require_live_queue_hold_capability',real)
 write(cfg.state_dir/'status.json',dict(schema=d.SCHEMA,queue_hold_lease_version=0))
 before={p.name for p in cfg.state_dir.iterdir()}
 with pytest.raises(ValueError):register(rig)
 assert {p.name for p in cfg.state_dir.iterdir()}==before


def test_live_capability_is_rechecked_after_lease_before_marker_link(rig,monkeypatch):
 cfg,now,state,*_=rig;calls=[]
 def check(*args):
  calls.append(1)
  if len(calls)==2:raise ValueError('controller changed before link')
  return dict(version=1,identity_verified=True)
 monkeypatch.setattr(q,'require_live_queue_hold_capability',check)
 with pytest.raises(ValueError,match='controller changed'):register(rig)
 assert len(calls)==2 and not (cfg.state_dir/'hold').exists()
 assert (cfg.state_dir/'hold.lease.json').exists() and q.review_queue_hold(cfg,now+11)['authenticated_expired']


def test_status_advertises_runtime_feature_and_tiny_unrepresentable_ttl_is_refused(rig):
 cfg,now,state,probe,*_=rig;value=d.tick(cfg,state,now=now,probe=probe)
 assert value['queue_hold_lease_version']==1
 with pytest.raises(ValueError):q.register_queue_hold(cfg,'sub-ulp bound',1e-300,clock=lambda:now)
 assert not (cfg.state_dir/'hold.lease.json').exists()


def test_touch_before_unlink_preserves_changed_hold_custody(rig,monkeypatch):
 cfg,now,state,*_=rig;register(rig);old=q._unlink_registered;changed=[]
 def touch(path,review,descriptor=False):
  if path.name=='hold' and not changed:
   info=path.stat();os.utime(path,ns=(info.st_atime_ns,info.st_mtime_ns+1));changed.append(True)
  return old(path,review,descriptor)
 monkeypatch.setattr(q,'_unlink_registered',touch)
 value=q.expire_queue_hold(cfg,state,now+11,apply=True)
 assert value['retirement']=='refused' and changed and (cfg.state_dir/'hold').exists()
 assert not q.review_queue_hold(cfg,now+11)['authenticated_expired']


def test_capability_verification_cannot_link_a_lease_that_expires_during_check(rig):
 cfg,now,state,*_=rig;clock=iter([now,now+9,now+10])
 with pytest.raises(ValueError,match='expired during live capability'):
  q.register_queue_hold(cfg,'slow capability check',10,clock=lambda:next(clock))
 assert not (cfg.state_dir/'hold').exists()
 assert q.review_queue_hold(cfg,now+11)['authenticated_expired']
