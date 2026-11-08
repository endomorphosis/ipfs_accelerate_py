"""Bounded malformed JSON remains invalid metadata, not a controller crash."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.rescue import repair_dispatcher as d, queue_hold_lease as q

fixture_path=Path('/home/barberb/lift_coding/external/ipfs_accelerate/test/api/test_agent_supervisor_queue_hold_lease.py')
spec=importlib.util.spec_from_file_location('_canonical_queue_fixture',fixture_path);prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
rig=prior.rig


def nested():
 # This build's C decoder can exceed Python's nominal recursionlimit. An
 # unclosed12000-level input genuinely exhausts it within the16KiB body bound.
 raw=b'['*12000
 assert len(raw)<=16384
 return raw


def register(rig):
 cfg,now,*_=rig
 return q.register_queue_hold(cfg,'private parser fixture',10,clock=lambda:now)


def snapshot(directory):
 return {p.name:(p.stat().st_ino,hashlib.sha256(p.read_bytes()).hexdigest()) for p in directory.iterdir() if p.is_file()}


@pytest.mark.parametrize('manual',[False,True])
def test_nested_sidecar_is_invalid_and_tick_keeps_running_without_deleting_manual_hold(rig,manual):
 cfg,now,state,probe,stopped,_=rig;path=cfg.state_dir/'hold.lease.json';path.write_bytes(b'['*4000);path.chmod(0o600)
 hold=cfg.state_dir/'hold'
 if manual:hold.write_text('unknown manual hold\n')
 before=path.read_bytes();counters=copy.deepcopy(state['attempts'])
 result=d.tick(cfg,state,apply=True,now=now,probe=probe)
 assert result['queue_hold_lease']['status']=='invalid' and result['held'] is manual
 assert result['action']==('held' if manual else 'healthy') and not stopped
 assert path.read_bytes()==before and state['attempts']==counters
 if manual:assert hold.read_text()=='unknown manual hold\n'


@pytest.mark.parametrize('consumer',['review','tick'])
def test_nested_body_is_invalid_before_any_expiry_or_counter_mutation(rig,consumer):
 cfg,now,state,probe,stopped,_=rig;register(rig);hold=cfg.state_dir/'hold';hold.write_bytes(nested());before=snapshot(cfg.state_dir)
 if consumer=='review':
  value=q.review_queue_hold(cfg,now+11)
  assert value['status']=='invalid' and not value['authenticated_expired']
  assert snapshot(cfg.state_dir)==before
 else:
  counters=copy.deepcopy(state['attempts']);value=d.tick(cfg,state,apply=True,now=now+11,probe=probe)
  assert value['action']=='held' and value['queue_hold_lease']['status']=='invalid'
  assert state['attempts']==counters and hold.read_bytes()==nested()
 assert not stopped


def test_nested_actual_state_refuses_physical_retirement_and_releases_only_own_fixer(rig):
 cfg,now,state,probe,stopped,_=rig;register(rig);path=cfg.state_dir/'state.json';path.write_bytes(nested());before=snapshot(cfg.state_dir);saved=copy.deepcopy(state)
 value=q.expire_queue_hold(cfg,state,now+11,apply=True)
 assert value['retirement']=='refused' and (cfg.state_dir/'hold').exists()
 assert state==saved and snapshot(cfg.state_dir)==before
 assert not cfg.repair_lock_path.exists() and not stopped


def test_nested_live_capability_status_refuses_before_registration(rig,monkeypatch):
 cfg,now,state,probe,stopped,real=rig;status=cfg.state_dir/'status.json';status.write_bytes(nested());status.chmod(0o600)
 monkeypatch.setattr(q,'require_live_queue_hold_capability',real)
 before=snapshot(cfg.state_dir)
 with pytest.raises(ValueError):q.register_queue_hold(cfg,'must refuse before effects',10,clock=lambda:now)
 assert snapshot(cfg.state_dir)==before and not (cfg.state_dir/'hold').exists()


def test_dispatcher_json_normalizes_bounded_recursion_without_guessing_state(rig):
 cfg,*_=rig;path=cfg.state_dir/'state.json';path.write_bytes(nested());before=path.read_bytes()
 with pytest.raises(ValueError):d._json(path)
 assert path.read_bytes()==before


class PhaseDeadline(BaseException):pass


@pytest.mark.parametrize('control',[KeyboardInterrupt,SystemExit,PhaseDeadline])
@pytest.mark.parametrize('surface',['sidecar','body','state','capability','dispatcher'])
def test_metadata_decode_never_swallows_original_control_object(rig,monkeypatch,control,surface):
 cfg,now,state,probe,stopped,real=rig;register(rig)
 original=control('original control object')
 def interrupt(*args,**kwargs):raise original
 proxy=SimpleNamespace(**vars(json));proxy.loads=interrupt
 if surface=='dispatcher':
  monkeypatch.setattr(d,'json',proxy);invoke=lambda:d._json(cfg.state_dir/'state.json')
 elif surface=='sidecar':
  monkeypatch.setattr(q,'json',proxy);invoke=lambda:q.review_queue_hold(cfg,now+11)
 elif surface=='body':
  old=json.loads;calls=[]
  def body(*args,**kwargs):
   calls.append(1)
   if len(calls)==2:raise original
   return old(*args,**kwargs)
  proxy.loads=body;monkeypatch.setattr(q,'json',proxy);invoke=lambda:q.review_queue_hold(cfg,now+11)
 elif surface=='state':
  old=json.loads
  def actual(raw,*args,**kwargs):
   if raw==(cfg.state_dir/'state.json').read_bytes():raise original
   return old(raw,*args,**kwargs)
  proxy.loads=actual;monkeypatch.setattr(q,'json',proxy);invoke=lambda:q.expire_queue_hold(cfg,state,now+11,apply=True)
 else:
  status=cfg.state_dir/'status.json';status.write_text('{}');status.chmod(0o600)
  monkeypatch.setattr(q,'json',proxy);invoke=lambda:real(cfg,now)
 before=(cfg.state_dir/'hold').read_bytes();counters=copy.deepcopy(state)
 with pytest.raises(control) as caught:invoke()
 assert caught.value is original and (cfg.state_dir/'hold').read_bytes()==before and state==counters
 assert not cfg.repair_lock_path.exists() and not stopped


def test_sidecar_decoder_recursion_failure_is_contained_at_the_decode_boundary(rig,monkeypatch):
 cfg,now,state,probe,stopped,_=rig;register(rig)
 original=RecursionError('explicit decoder boundary fixture')
 def fail(*args,**kwargs):raise original
 proxy=SimpleNamespace(**vars(json));proxy.loads=fail;monkeypatch.setattr(q,'json',proxy)
 before=(cfg.state_dir/'hold').read_bytes();counters=copy.deepcopy(state)
 value=q.review_queue_hold(cfg,now+11)
 assert value['status']=='invalid' and not value['authenticated_expired']
 assert (cfg.state_dir/'hold').read_bytes()==before and state==counters and not stopped


def test_nested_manifest_reports_observation_error_and_continues_heartbeat_without_holds(rig):
 cfg,now,state,probe,stopped,_=rig;cfg.manifest_path.write_bytes(nested());before=cfg.manifest_path.read_bytes();counters=copy.deepcopy(state['attempts'])
 value=d.tick(cfg,state,apply=True,now=now,probe=probe)
 assert value['action']=='observation_error' and value['queue_hold_lease_version']==1
 assert value['control_heartbeat_at']>0 and state['attempts']==counters and not stopped
 assert cfg.manifest_path.read_bytes()==before and not (cfg.state_dir/'hold').exists()
