"""MODELED protocol/control regressions; no actual Main/native/health credit.

Profiles/managers/cycles below are explicitly modeled. No private _CycleHandle
is manufactured, no private _CYCLES is edited and no fake Popen/wait can produce
a positive acknowledgment. Genuine42e/e243 negative and real Main API/epoch
qualification are separate original-control gates described in the packet.
"""
import copy
import json
from pathlib import Path
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import native_executor_health as health
from ipfs_accelerate_py.agent_supervisor.rescue import scientific_executor as executor


def put_model(path,value):
    path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    raw=value if isinstance(value,bytes) else (json.dumps(value,sort_keys=True)+'\n').encode()
    path.write_bytes(raw);path.chmod(0o600)
    return health.descriptor(path)


@pytest.fixture
def idle_model(tmp_path,monkeypatch):
    root=tmp_path/'proof';root.mkdir(mode=0o700)
    owner=dict(path=str(tmp_path/'modeled-owner.py'),bytes=1,sha256=health.SOURCE_LEASE_OWNER_SHA)
    profile=dict(owner_source=owner,max_cycle_seconds=300)
    profile_ref=put_model(tmp_path/'modeled-profile.json',profile)
    monkeypatch.setattr(health,'_profile',lambda ref:copy.deepcopy(profile))
    request=dict(schema='legal-ir-original-native-control-request/v1',purpose='native_control_metadata',
        source_sha256={owner['path']:owner['sha256'],health.SOURCE_LEASE_MODULE_PATH:health.SOURCE_LEASE_MODULE_SHA},
        source_roots=[str(Path(health.SOURCE_LEASE_MODULE_PATH).parents[3]),
                      str(Path(health.SOURCE_LEASE_MODULE_PATH).parents[3].parent/'ipfs_accelerate')])
    request_ref=put_model(root/'request.json',request)
    terminal=dict(schema='legal-ir-original-native-control-terminal/v1',request_sha256=request_ref['sha256'],
        completed=False,child_launch_attempted=False,child_reaped=False,child_returncode=None,
        control_effects='not_started',storage_status='not_requested',original_native_compute_closed=True,
        source_READ_closed=True,absolute_resource_operation_deadline_closed=True,cleanup_errors=[],
        received_signals=[],error_type='SourceLeaseTimeout',error='timed out waiting for admission source lease',
        task_completion_authority_granted_by_resource_owner=False,owner_pid=222,
        owner_boot_id='modeled-same-boot',elapsed_seconds=60.05,
        resource_profile='original_orchestration_parented_disk_sampled_birth_bound_descendants',
        external_file_growth_enforcement='polled_size_caps_and_original_conservative_precharge_not_sandbox',
        partial_or_failed_control_must_be_reconciled=True)
    return dict(root=root,profile=profile,profile_ref=profile_ref,request=request,request_ref=request_ref,terminal=terminal)


def test_exact_prestart_model_does_not_invent_absent_facts_or_health(idle_model):
    case=idle_model;before=copy.deepcopy(case['terminal'])
    result=health.classify_original_source_lease_deferral(case['profile_ref'],case['request_ref'],case['terminal'],kind='genuine_scoped_idle')
    assert result['eligible'] is True and result['health_credit'] is False
    assert result['native_or_storage_release_authority'] is False and case['terminal']==before
    assert not any(key in case['terminal']for key in ('owner_birth','database_preimage_before_grants','reservation'))


@pytest.mark.parametrize('key,value',[
    ('completed',True),('child_launch_attempted',True),('child_reaped',True),('child_returncode',0),
    ('control_effects','started'),('storage_status','retained'),('original_native_compute_closed',False),
    ('source_READ_closed',False),('absolute_resource_operation_deadline_closed',False),
    ('cleanup_errors',['unknown']),('received_signals',[15]),('error_type','ResourceAdmissionDeadlineError'),
    ('error','timed out waiting for read source lease'),('task_completion_authority_granted_by_resource_owner',True),
    ('owner_pid',True),('elapsed_seconds',float('nan')),('request_sha256','f'*64),
    ('owner_birth','123'),('database_preimage_before_grants',{}),('reservation',{}),
    ('execution_request',{}),('external_precharge',{}),('process_spawn_receipts',[]),
    ('unknown_extra_field',True),
],ids=['completed','launch','child_reaped','child_returncode','control_effect','storage','native','source','deadline',
       'cleanup','signal','other_error','other_lease_stage','task_authority','PID_bool','nonfinite','request',
       'fabricated_birth','DB_point','reservation','execution_request','charge','spawns','unknown_extra'])
def test_source_lease_model_rejects_unknown_or_started_outcome(idle_model,key,value):
    case=idle_model;case['terminal'][key]=value
    with pytest.raises(health.ExecutorHealthError):
        health.classify_original_source_lease_deferral(case['profile_ref'],case['request_ref'],case['terminal'],kind='genuine_scoped_idle')


def test_native_execution_cannot_enter_idle_retry_model(idle_model):
    with pytest.raises(health.ExecutorHealthError):
        health.classify_original_source_lease_deferral(idle_model['profile_ref'],idle_model['request_ref'],
            idle_model['terminal'],kind='native_execution')


def test_unqualified_source_lease_implementation_is_not_admitted_model(idle_model):
    case=idle_model;request=copy.deepcopy(case['request'])
    request['source_sha256'][health.SOURCE_LEASE_MODULE_PATH]='f'*64
    ref=put_model(Path(case['request_ref']['path']),request)
    terminal=dict(case['terminal'],request_sha256=ref['sha256'])
    with pytest.raises(health.ExecutorHealthError,match='lease_source'):
        health.classify_original_source_lease_deferral(case['profile_ref'],ref,terminal,kind='genuine_scoped_idle')


def test_serialized_or_fake_handle_never_consumes_private_capability(idle_model):
    before_requests=set(health._USED_REQUESTS);before_handles=tuple(health._CYCLES)
    with pytest.raises(health.ExecutorHealthError,match='capability_missing'):
        health.acknowledge_original_nonstarted_cycle(idle_model['profile_ref'],{},idle_model['request_ref'],
            sequence=1,cycle_handle={'serialized':'not-a-capability'})
    assert set(health._USED_REQUESTS)==before_requests and tuple(health._CYCLES)==before_handles


class ModeledStop:
    def __init__(self,stop_during_cooldown=False):self.waits=[];self.stop_during_cooldown=stop_during_cooldown
    def is_set(self):return False
    def wait(self,seconds):
        self.waits.append(seconds)
        return self.stop_during_cooldown and seconds==60


@pytest.fixture
def serve_model(tmp_path,monkeypatch):
    """Control-only outcomes; no original owner or health publisher is mocked positive."""
    root=tmp_path/'proof';root.mkdir(mode=0o700)
    profile={'proof_root':str(root),'receipt_path':str(root/'health.json')}
    config={'max_cycles':3,'max_proof_bytes':1024**2,'poll_seconds':1}
    config_ref=put_model(tmp_path/'config.json',config);profile_ref=put_model(tmp_path/'profile.json',profile)
    monkeypatch.setattr(executor,'load_configuration',lambda ref:(config,profile_ref,profile))
    monkeypatch.setattr(health,'_profile',lambda ref:profile)
    monkeypatch.setattr(executor,'require_main_process',lambda *a:{'model_identity':True})
    monkeypatch.setattr(executor,'check_proof_budget',lambda *a,**k:{'bytes':0,'files':0})
    monkeypatch.setattr(executor,'observe_native',lambda config:{'modeled_quiet_scope':True})
    monkeypatch.setattr(executor,'select_template',lambda *a:({},'genuine_scoped_idle'))
    seen=[]
    def bind(outcomes):
        choices=iter(outcomes)
        def modeled_cycle(config,pref,profile,template,kind,sequence,stop,*,cycle_number=None,source_retry=None,profile_transition_ref=None):
            cycle_number=sequence if cycle_number is None else cycle_number
            seen.append((cycle_number,sequence,copy.deepcopy(source_retry)))
            path=root/('cycle-%08d-modeled'%cycle_number);path.mkdir()
            put_model(path/'intent.json',{'kind':kind,'sequence':sequence,'cycle_number':cycle_number})
            if next(choices)=='defer':
                ref=put_model(path/'prestart-deferral.json',{'MODELED_ONLY':True,'health_credit':False})
                return dict(deferred_source_lease=True,deferral_ref=ref,cycle_root=str(path))
            # Control-flow result only, not a forged receipt/proof/health write.
            return {'MODELED_SUCCESS_RETURN_VALUE_ONLY':True}
        monkeypatch.setattr(executor,'run_cycle',modeled_cycle)
    return dict(root=root,config=config,config_ref=config_ref,seen=seen,bind=bind)


def test_three_model_deferrals_spend_three_positions_two_slots_and_no_sequence(serve_model):
    case=serve_model;case['bind'](['defer','defer','defer']);stop=ModeledStop()
    assert executor.serve(case['config_ref'],stop)==executor.HOLD_EXIT
    assert [(row[0],row[1])for row in case['seen']]==[(1,1),(2,1),(3,1)]
    assert [row[2]['additional_slot'] if row[2]else None for row in case['seen']]==[None,1,2]
    assert stop.waits.count(60)==2 and not(case['root']/'health.json').exists()
    hold=json.loads((case['root']/'executor-held.json').read_bytes())
    assert hold['reason']=='source_lease_retry_exhausted'


def test_real_success_control_result_alone_advances_modeled_sequence(serve_model):
    case=serve_model;case['bind'](['defer','success','success']);stop=ModeledStop()
    executor.serve(case['config_ref'],stop)
    assert [(row[0],row[1])for row in case['seen']]==[(1,1),(2,1),(3,2)]
    assert stop.waits.count(60)==1 and not(case['root']/'health.json').exists()


def test_stop_during_original_cooldown_never_starts_additional_model_cycle(serve_model):
    case=serve_model;case['bind'](['defer']);stop=ModeledStop(stop_during_cooldown=True)
    assert executor.serve(case['config_ref'],stop)==0
    assert len(case['seen'])==1 and stop.waits==[60]


@pytest.fixture
def profile_transition_model(tmp_path,monkeypatch):
    """Data-only source-profile contract; no model is admitted as a real Main."""
    root=tmp_path/'proof';root.mkdir(mode=0o700)
    rows=[put_model(tmp_path/(name+'.py'),b'# modeled producer source data\n')for name in ('main','health')]
    old_rows=[put_model(tmp_path/(name+'-old.py'),b'# retained modeled source data\n')for name in ('main','health')]
    fixed=dict(schema=health.PROFILE_SCHEMA,catalogue={'scope':'modeled'},receipt_path=str(root/'health.json'),
        task_scope=['modeled-task'],max_age_seconds=300,max_cycle_seconds=300,proof_root=str(root),
        cycle_contract={'modeled':True},owner_source={'unchanged-owner':True},driver_source={'unchanged-driver':True})
    prior={**fixed,'main_entrypoint':old_rows[0],'health_module':old_rows[1],
        'source_descriptors':old_rows,'unit':dict(name='modeled.service',interpreter='/modeled/python',
            working_directory=str(tmp_path),argv=['/modeled/python',old_rows[0]['path']],environment={'PYTHONPATH':'/old','PYTHONDONTWRITEBYTECODE':'1'})}
    current={**fixed,'main_entrypoint':rows[0],'health_module':rows[1],
        'source_descriptors':old_rows+rows,'unit':dict(name='modeled.service',interpreter='/modeled/python',
            working_directory=str(tmp_path),argv=['/modeled/python',rows[0]['path'],'--known-source-lease-recovery',str(tmp_path/'grant.json')],environment={'PYTHONPATH':'/new','PYTHONDONTWRITEBYTECODE':'1'})}
    old_ref=put_model(tmp_path/'old-profile.json',prior);new_ref=put_model(tmp_path/'new-profile.json',current)
    profiles={old_ref['path']:prior,new_ref['path']:current}
    monkeypatch.setattr(health,'_profile',lambda ref:copy.deepcopy(profiles[ref['path']]))
    config=dict(schema=executor.CONFIG_SCHEMA,catalogue={'scope':'modeled'},idle_template_ref={'same':True},
        execution_templates=[],poll_seconds=60,max_cycles=10000,max_proof_bytes=1073741824)
    old_config=put_model(tmp_path/'old-config.json',{**config,'profile_path':old_ref['path']})
    new_config=put_model(tmp_path/'new-config.json',{**config,'profile_path':new_ref['path']})
    source=put_model(tmp_path/'source-successor.json',dict(schema='agent-supervisor/scientific-producer-source-successor@1',
        prior_profile_ref=old_ref,successor_profile_ref=new_ref,producer_sources=rows,
        owner_environment=prior['unit']['environment'],prior_unit_argv=prior['unit']['argv'],successor_unit_argv=current['unit']['argv']))
    grant={key:None for key in health.SOURCE_LEASE_RECOVERY_KEYS}
    grant.update(schema=health.SOURCE_LEASE_RECOVERY_SCHEMA,authorization='one_exact_same_generation_source_lease_resume',
        incident='cycle48-source-lease-timeout-v1',prior_profile_ref=old_ref,successor_profile_ref=new_ref,
        prior_config_ref=old_config,successor_config_ref=new_config,source_successor_ref=source)
    return dict(prior=prior,current=current,old_ref=old_ref,new_ref=new_ref,grant=grant,root=root)


def test_profile_transition_model_preserves_payload_roles_and_original_limits(profile_transition_model):
    case=profile_transition_model
    _,ref,prior=health.validate_source_profile_transition_value(case['grant'],case['new_ref'],case['current'])
    assert ref==case['old_ref'] and prior['owner_source']==case['current']['owner_source']


@pytest.mark.parametrize('field',['catalogue','task_scope','proof_root','receipt_path','max_cycle_seconds','owner_source','driver_source'])
def test_profile_transition_model_refuses_semantic_or_resource_role_change(profile_transition_model,field):
    case=profile_transition_model;current=copy.deepcopy(case['current']);current[field]='changed'
    with pytest.raises(health.ExecutorHealthError):
        health.validate_source_profile_transition_value(case['grant'],case['new_ref'],current)


@pytest.mark.parametrize('sequence',[1,2])
def test_first_epoch_transition_model_checks_old_profile_at_original_clocks(profile_transition_model,tmp_path,monkeypatch,sequence):
    case=profile_transition_model;root=case['root'];failed=root/'cycle48';failed.mkdir()
    old_identity={'invocation_id':'old-modeled-invocation'};new_identity={'invocation_id':'new-modeled-invocation'}
    old_proof={'native_diagnostic_ref':{'MODELED_DATA_ONLY':True},'request_ref':{'sha256':'old-request'}}
    old_proof_ref=put_model(root/'old-cycle-proof.json',old_proof)
    old=dict(profile_ref=case['old_ref'],sequence=34,identity=old_identity,observed_at=100.0,
             observed_monotonic_ns=100000,cycle_proof_ref=old_proof_ref)
    old_ref=put_model(root/'health.json',old)
    grant=dict(case['grant'],last_health_receipt_ref=old_ref,last_closed_cycle_ref=old_proof_ref,
               prior_main_identity=old_identity,hold_ref=put_model(root/'hold.json',{'cycle_root':str(failed)}))
    grant_ref=put_model(tmp_path/'grant.json',grant)
    put_model(failed/'source-lease-reconciliation.json',dict(
        schema='agent-supervisor/scientific-source-lease-reconciliation@1',recovery_ref=grant_ref,
        consuming_main_identity=new_identity,health_credit=False,spent_cycle_count=48))
    monkeypatch.setenv(health.SOURCE_LEASE_RECOVERY_ENV,grant_ref['sha256'])
    seen=[]
    def historical_receipt(ref,pref,profile,native,identity,*,now,monotonic_ns):
        seen.append((pref,now,monotonic_ns,native))
        return old,old_proof
    monkeypatch.setattr(health,'_receipt',historical_receipt)
    monkeypatch.setattr(health,'_linked_native',lambda *a,**k:({'MODELED_OLD_NATIVE':True},{}))
    monkeypatch.setattr(health,'_definitely_dead_identity',lambda *a:True)
    proof=dict(sequence=sequence,request_ref={'sha256':'new-request'})
    if sequence==2:
        with pytest.raises(health.ExecutorHealthError):
            health._source_transition_previous_receipt(grant_ref,old_ref,case['new_ref'],case['current'],
                {'MODELED_CURRENT_NATIVE':True},new_identity,proof,now=200.0)
    else:
        result=health._source_transition_previous_receipt(grant_ref,old_ref,case['new_ref'],case['current'],
            {'MODELED_CURRENT_NATIVE':True},new_identity,proof,now=200.0)
        assert seen==[(case['old_ref'],100.0,100000,{'MODELED_OLD_NATIVE':True})]
        assert result['profile_transition_ref']==grant_ref and result['archive_raw']==health._bound(old_ref)
        assert not Path(result['archive_path']).exists()


def test_typed_first_receipt_model_can_advance_to_ordinary_sequence_two_without_replaying_grant(profile_transition_model,tmp_path,monkeypatch):
    case=profile_transition_model;profile=case['current'];pref=case['new_ref'];unit=profile['unit']
    identity=dict(unit=unit['name'],pid=12345,birth_ticks=67890,boot_id='modeled-boot',invocation_id='a'*32,
        argv=unit['argv'],exe=unit['interpreter'],cwd=unit['working_directory'],
        environment={**unit['environment'],'INVOCATION_ID':'a'*32,health.ENV_PROFILE_SHA:pref['sha256']})
    old_proof={'request_ref':{'sha256':'old-real-work-shaped-data'}}
    old=dict(schema=health.PROFILE_TRANSITION_RECEIPT_SCHEMA,profile_ref=pref,sequence=1,identity=identity,
        observed_at=100.0,observed_monotonic_ns=100000,cycle_proof_ref={'MODELED_ONLY':True},catalogue={},origins={},
        previous_receipt_sha256='f'*64,profile_transition_ref={'MODELED_ALREADY_CONSUMED_GRANT':True})
    old_ref=put_model(Path(profile['receipt_path']),old);seen=[]
    def authenticate_previous(ref,actual_pref,actual_profile,native,actual_identity,*,now,monotonic_ns):
        seen.append((ref,actual_pref,now,monotonic_ns))
        assert actual_profile==profile and actual_identity==identity
        return old,old_proof
    def replay_forbidden(*args,**kwargs):pytest.fail('same-profile seq2 cannot replay old profile transition')
    monkeypatch.setattr(health,'_receipt',authenticate_previous)
    monkeypatch.setattr(health,'_source_transition_previous_receipt',replay_forbidden)
    result=health._previous_receipt_transition(pref,profile,{},identity,
        {'sequence':2,'request_ref':{'sha256':'new-real-work-shaped-data'},'started_monotonic_ns':100001},
        now=101.0,monotonic_ns=101000,profile_transition_ref={'MODELED_ALREADY_CONSUMED_GRANT':True})
    assert seen==[(old_ref,pref,100.0,100000)]
    assert result['previous_ref']==old_ref and result['archive_path'] is None
    assert 'profile_transition_ref' not in result


@pytest.mark.parametrize('changed_binding',['config','profile'])
def test_cooldown_binding_change_model_prevents_any_additional_cycle(serve_model,monkeypatch,changed_binding):
    case=serve_model;case['bind'](['defer'])
    original_profile=health._profile(None)
    class ChangeDuringCooldown(ModeledStop):
        def wait(self,seconds):
            result=super().wait(seconds)
            if seconds==60:
                if changed_binding=='config':put_model(Path(case['config_ref']['path']),{'changed_during_cooldown':True})
                else:monkeypatch.setattr(health,'_profile',lambda ref:{**original_profile,'changed_during_cooldown':True})
            return result
    stop=ChangeDuringCooldown()
    assert executor.serve(case['config_ref'],stop)==executor.HOLD_EXIT
    assert len(case['seen'])==1 and stop.waits==[60]
    assert not(case['root']/'health.json').exists()
    assert (case['root']/'executor-held.json').exists()
