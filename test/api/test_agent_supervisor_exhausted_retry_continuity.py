"""Exhausted retries must retain bounded escalation and truthful health."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest
from ipfs_accelerate_py.agent_supervisor.rescue import repair_dispatcher as dispatcher
from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume
import ipfs_accelerate_py.agent_supervisor.rescue as rescue_package

# Reuse unchanged actual fixture/provenance builders from the existing suite.
source_root=Path(lane_resume.__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('_current_dispatcher_test_fixtures',source_root/'test/api/test_agent_supervisor_repair_dispatcher.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
rig=f.rig
status=f.status
exhausted_wait=f.exhausted_wait
write_json=f.write_json
kinds=f.kinds


def test_native_cooling_failure_is_retry_wait_and_does_not_refund_maintenance_budget(rig):
    config,now,lane,live,probe=rig
    exhausted_wait(rig,exit_code=1)
    state={'unresolved_signature':'previous','attempts':{'previous':{'count':2}},'next_dispatch_at':now+100}
    result=dispatcher.tick(config,state,now=now,probe=probe)
    assert result['action']=='retry_wait' and not result['incidents']
    assert state['attempts']['previous']['count']==2 and state['unresolved_signature']=='previous'
    assert state['next_dispatch_at']==now+100


@pytest.mark.parametrize('phase',['deterministic_command','validation','lane_resume'])
def test_real_failed_retry_does_not_erase_first_failure_clock(rig,phase):
    config,now,lane,live,probe=rig;config.quiet_seconds=30
    (config.repo_root/'source.py').write_text('raise SystemExit(1)\n')
    assert subprocess.run([sys.executable,'source.py'],cwd=config.repo_root,timeout=5).returncode==1
    ledger=exhausted_wait(rig,exit_code=1)
    state={};dispatcher.observe(config,state,now=now,probe=probe)
    status(lane,phase=phase,active_task_id='VOG-001')
    assert not dispatcher.observe(config,state,now=now+10,probe=probe)['incidents']
    assert state['exhausted_wait_observations']['vog:VOG-001']['first_seen']==now
    assert subprocess.run([sys.executable,'source.py'],cwd=config.repo_root,timeout=5).returncode==1
    ledger['VOG-001']+=1;ledger['_retry_after_ms']['VOG-001']=int((now+920)*1000)
    write_json(lane/'lane_resume_attempts.json',ledger)
    status(lane,phase='attempt_retry_wait',active_task_id='',heartbeat_at_ms=(now+31)*1000)
    result=dispatcher.observe(config,state,now=now+31,probe=probe)
    assert 'exhausted_hard_failure' in kinds(result)
    assert state['exhausted_wait_observations']['vog:VOG-001']['first_seen']==now
    assert state['exhausted_wait_observations']['vog:VOG-001']['retry_deadline']==int((now+900)*1000)/1000


def test_unrelated_control_source_change_does_not_restart_failure_grace(rig):
    config,now,lane,live,probe=rig;config.quiet_seconds=30
    extra=config.repo_root/'unrelated_monitor.py';extra.write_text('version=1\n')
    config.evidence_paths=(*config.evidence_paths,extra)
    exhausted_wait(rig,exit_code=1);state={};dispatcher.observe(config,state,now=now,probe=probe)
    extra.write_text('version=2\n')
    result=dispatcher.observe(config,state,now=now+31,probe=probe)
    assert 'exhausted_hard_failure' in kinds(result)
    assert state['exhausted_wait_observations']['vog:VOG-001']['first_seen']==now


def test_new_completed_work_really_resets_failure_grace(rig):
    config,now,lane,live,probe=rig;config.quiet_seconds=30
    exhausted_wait(rig,exit_code=1);state={};dispatcher.observe(config,state,now=now,probe=probe)
    status(lane,last_task_transition_at_ms=(now+20)*1000)
    assert not dispatcher.observe(config,state,now=now+31,probe=probe)['incidents']
    assert state['exhausted_wait_observations']['vog:VOG-001']['first_seen']==now+31


def test_actual_completed_task_clears_failure_and_allows_recovery_observation(rig):
    config,now,lane,live,probe=rig
    exhausted_wait(rig,exit_code=1);state={'unresolved_signature':'prior','attempts':{'prior':{'count':2}}}
    dispatcher.tick(config,state,now=now,probe=probe)
    text=(config.repo_root/'todo.md').read_text().replace('- Status: pending','- Status: completed')
    (config.repo_root/'todo.md').write_text(text)
    status(lane,phase='lane_resume',active_task_id='')
    result=dispatcher.tick(config,state,now=now+10,probe=probe)
    assert result['action']=='recovery_observed' and not state['exhausted_wait_observations']
    assert not state['attempts']


def semantic(rig,**changes):
    config,now,lane,live,probe=rig
    ledger=exhausted_wait(rig,attempts=6,exit_code=1)
    contract=ledger['_command_evidence']['VOG-001']['contract']
    progress={'schema':'agent-supervisor/execution-progress@1','contract':contract,'semantic_failures':6,
              'state':'failed','last_exit_code':1,'last_stage':'command','finished_at_ms':int((now-10)*1000)}
    progress.update(changes);ledger['_validation_fixed_points']['VOG-001']['execution_progress']=progress
    ledger['_retry_after_ms'].clear();write_json(lane/'lane_resume_attempts.json',ledger)
    status(lane,state='degraded',phase='semantic_failure_wait',active_task_id='VOG-001',
           backpressure_reasons=['VOG-001:unchanged_semantic_failure_budget_exhausted'])
    return ledger


def test_semantic_failure_wait_is_verified_and_escalates_without_a_future_retry_deadline(rig):
    config,now,lane,live,probe=rig
    semantic(rig);state={}
    first=dispatcher.tick(config,state,now=now,probe=probe)
    assert first['action']=='retry_wait' and not first['incidents']
    result=dispatcher.tick(config,state,now=now+1,probe=probe)
    assert result['action']=='would_dispatch' and 'exhausted_hard_failure' in kinds(result)
    assert result['incidents'][0]['evidence']['command_exit_code']==1


@pytest.mark.parametrize('change',[{'semantic_failures':5},{'semantic_failures':'6'},
    {'contract':'f'*64},{'state':'validated'},{'schema':'unknown'}])
def test_unqualified_semantic_wait_is_visible_but_not_accepted_as_failure_proof(rig,change):
    config,now,lane,live,probe=rig
    semantic(rig,**change);result=dispatcher.tick(config,{},now=now+100,probe=probe)
    assert result['action']=='retry_wait' and not result['incidents']
    assert not result['retry_waits']


def test_changed_source_receipt_is_review_needed_not_healthy_or_current_proof(rig):
    config,now,lane,live,probe=rig
    exhausted_wait(rig,exit_code=1)
    (config.repo_root/'source.py').write_text('print("real repaired producer")\n')
    result=dispatcher.tick(config,{},now=now,probe=probe)
    assert result['action']=='source_change_review_needed' and not result['incidents']
    assert result['source_provenance_reviews'][0]['current_proof_accepted'] is False


def test_real_cached_contract_module_cannot_silently_reject_new_policy_receipt(rig,monkeypatch,tmp_path):
    config,now,lane,live,probe=rig
    ledger=exhausted_wait(rig,exit_code=1)
    source=tmp_path/'cached_lane_resume.py';source.write_bytes(Path(lane_resume.__file__).read_bytes())
    name=lane_resume.__name__;spec=importlib.util.spec_from_file_location(name,source)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    # Disk policy advances after the old code has really been imported.
    source.write_text(source.read_text()+'\ndef _command_contract(*args, **kwargs):\n    return "f" * 64\n')
    ledger['_command_evidence']['VOG-001']['retry_contract']='f'*64
    write_json(lane/'lane_resume_attempts.json',ledger)
    monkeypatch.setitem(sys.modules,name,old);monkeypatch.setattr(rescue_package,'lane_resume',old)
    result=dispatcher.tick(config,{},now=now,probe=probe)
    assert result['action']=='would_dispatch' and kinds(result)=={'source_change_review_needed'}
    review=result['source_provenance_reviews'][0]
    assert review['loaded_policy_matches_disk'] is False
    assert '_command_contract' in review['mismatched_functions']
    assert review['current_proof_accepted'] is False and not result['retry_waits']


@pytest.mark.parametrize('last_exit_code',[1,0])
def test_semantic_validation_failure_requires_actual_failed_validation_receipt(rig,last_exit_code):
    config,now,lane,live,probe=rig
    ledger=semantic(rig,last_stage='validation',last_exit_code=last_exit_code)
    ledger['_command_evidence']['VOG-001']['exit_code']=0
    ledger['_validation_fixed_points']['VOG-001']['failures']=1
    write_json(lane/'lane_resume_attempts.json',ledger)
    state={};dispatcher.observe(config,state,now=now,probe=probe)
    result=dispatcher.tick(config,state,now=now+1,probe=probe)
    assert bool(result['incidents']) is (last_exit_code==1)
    if last_exit_code==1:
        assert result['incidents'][0]['evidence']['failure_stage']=='validation'
    else:
        assert result['action']=='retry_wait' and not result['retry_waits']
