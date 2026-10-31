"""Owning metadata/loop cases for exact closed source-admission deferrals.

Closed native/proof calls in these fixtures are explicitly modeled. They grant
no current manager, original cycle, native, runtime or task-completion credit.
"""
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import os
import threading
import time
import pytest
from ipfs_accelerate_py.agent_supervisor.rescue import scientific_executor as science


def pin(path):
    raw=Path(path).read_bytes();return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def write(path,value,mode=0o400):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.chmod(0o600)
    raw=value if type(value)is bytes else (json.dumps(value,sort_keys=True)+'\n').encode()
    path.write_bytes(raw);path.chmod(mode);return pin(path)

@pytest.fixture
def health(monkeypatch):
    def keys(value,expected,reason):science.need(type(value)is dict and set(value)==set(expected),reason)
    h=SimpleNamespace(_integer=lambda v,lo=0,hi=None:type(v)is int and v>=lo and (hi is None or v<=hi),
        _text=lambda v:type(v)is str and bool(v),_number=lambda v:type(v)in(int,float)and float('-inf')<v<float('inf'),
        _path=lambda p:Path(p),_same_birth=lambda *args:False,_definitely_dead_identity=lambda *args:True,descriptor=pin,_raw=lambda p:Path(p).read_bytes(),
        _decode=lambda raw:json.loads(raw),_document=lambda ref:json.loads(Path(ref['path']).read_bytes()),_keys=keys)
    monkeypatch.setattr(science,'health_owner',lambda:h);return h

@pytest.fixture
def negative(tmp_path,health):
    root=tmp_path/('cycle-00000003-'+'a'*32)
    root.mkdir();request={'schema':'legal-ir-original-native-control-request/v1','purpose':'native_control_metadata',
        'request_id':'scientific-executor:'+root.name,'attempt_directory':str(root/'attempt'),
        'cpu_slots':1,'memory_mb':1536,'child_process_slots':8,'storage_bytes':384*1024**2,'timeout_seconds':300,
        'resource_ledger':str(science.ORIGINAL_RESOURCE_LEDGER)}
    ref=write(root/'request.json',request)
    terminal={'schema':'legal-ir-original-native-control-terminal/v1','request_sha256':ref['sha256'],
        'completed':False,'error_type':'SourceLeaseTimeout','error':'timed out waiting for admission source lease',
        'control_effects':'not_started','storage_status':'not_requested','child_launch_attempted':False,'child_reaped':False,
        'child_returncode':None,'source_READ_closed':True,'original_native_compute_closed':True,
        'absolute_resource_operation_deadline_closed':True,'task_completion_authority_granted_by_resource_owner':False,
        'cleanup_errors':[],'received_signals':[],'owner_pid':2**30+71,
        'owner_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    return dict(root=root,request=request,request_ref=ref,terminal=terminal)


def test_exact_closed_birthless_source_timeout_is_negative_only(negative):
    n=negative;science.source_admission_terminal(n['terminal'],n['request_ref'],n['request'],n['root'])
    assert 'owner_birth'not in n['terminal'] and not(n['root']/'closed.json').exists()

@pytest.mark.parametrize('field,value',[
    ('completed',True),('child_launch_attempted',True),('child_reaped',True),('child_returncode',0),
    ('source_READ_closed',False),('original_native_compute_closed',False),('absolute_resource_operation_deadline_closed',False),
    ('control_effects','started'),('storage_status','released'),('error_type','DeadlineExceeded'),
    ('error','timed out waiting for source lock'),('cleanup_errors',['unknown']),('received_signals',[15]),
    ('owner_boot_id','other-boot'),('owner_pid',True),('task_completion_authority_granted_by_resource_owner',True),
])
def test_partial_failed_or_foreign_terminal_never_defers(negative,field,value):
    n=negative;n['terminal'][field]=value
    with pytest.raises(science.ExecutorHold):science.source_admission_terminal(n['terminal'],n['request_ref'],n['request'],n['root'])

@pytest.mark.parametrize('field',['reservation','execution_request','process_spawn_receipts','sampled_process_custody_final',
    'sampled_process_custody','external_precharge','external_before','external_after','external_output_parent_identities',
    'parent_capacity_handoff','database_preimage_before_grants','database_preimage_observed','database_postimage','allowed_subprocess_events'])
def test_any_original_effect_field_refuses_automatic_deferral(negative,field):
    n=negative;n['terminal'][field]={}
    with pytest.raises(science.ExecutorHold,match='effects'):science.source_admission_terminal(n['terminal'],n['request_ref'],n['request'],n['root'])

@pytest.mark.parametrize('field,value',[('purpose','native_execution'),('storage_bytes',1),('cpu_slots',True),('memory_mb',2048),('child_process_slots',9),('timeout_seconds',301)])
def test_original_native_scope_and_envelope_cannot_be_changed(negative,field,value):
    n=negative;n['request'][field]=value
    with pytest.raises(science.ExecutorHold):science.source_admission_terminal(n['terminal'],n['request_ref'],n['request'],n['root'])


def test_birthless_historical_owner_requires_literal_PID_absence(negative):
    n=negative;n['terminal']['owner_pid']=os.getpid()
    with pytest.raises(science.ExecutorHold,match='PID_not_absent'):science.source_admission_terminal(n['terminal'],n['request_ref'],n['request'],n['root'])


def test_attempt_payload_or_positive_cycle_evidence_refuses_deferral(negative):
    n=negative;attempt=Path(n['request']['attempt_directory']);attempt.mkdir(mode=0o700);(attempt/'unknown').write_text('effect')
    with pytest.raises(science.ExecutorHold,match='attempt'):science.source_admission_terminal(n['terminal'],n['request_ref'],n['request'],n['root'])
    (attempt/'unknown').unlink();write(n['root']/'cycle-proof.json',{})
    with pytest.raises(science.ExecutorHold,match='positive'):science.source_admission_terminal(n['terminal'],n['request_ref'],n['request'],n['root'])


def test_captured_owner_binding_does_not_fabricate_terminal_birth(negative):
    n=negative;captured=dict(pid=n['terminal']['owner_pid'],boot_id=n['terminal']['owner_boot_id'],birth_ticks=123)
    science.source_admission_terminal(n['terminal'],n['request_ref'],n['request'],n['root'],captured_owner=captured)
    assert 'owner_birth'not in n['terminal'];captured['pid']+=1
    with pytest.raises(science.ExecutorHold,match='captured_owner'):science.source_admission_terminal(n['terminal'],n['request_ref'],n['request'],n['root'],captured_owner=captured)


def test_only_producer_config_and_explicit_recovery_argv_slots_may_move():
    a=['python','-B','/old/main.py','serve','--config','/old/config','--config-sha256','a','--bounded-generation-handover','/old/auth']
    b=['python','-B','/new/main.py','serve','--config','/new/config','--config-sha256','b','--known-source-admission-recovery','/new/auth']
    assert science.source_admission_unit_projection(a,'/old/main.py')==science.source_admission_unit_projection(b,'/new/main.py')
    b.insert(1,'-O');assert science.source_admission_unit_projection(a,'/old/main.py')!=science.source_admission_unit_projection(b,'/new/main.py')

@pytest.mark.parametrize('field',['--config','--config-sha256','--bounded-generation-handover'])
def test_duplicate_critical_unit_argument_refuses(field):
    argv=['python','/main','--config','x','--config-sha256','y','--bounded-generation-handover','z',field,'duplicate']
    with pytest.raises(science.ExecutorHold):science.source_admission_unit_projection(argv,'/main')

@pytest.fixture
def inventory_case(tmp_path,health):
    root=tmp_path/'producer';rels=['ipfs_accelerate_py/__init__.py','ipfs_accelerate_py/agent_supervisor/__init__.py',
        'ipfs_accelerate_py/agent_supervisor/rescue/__init__.py','ipfs_accelerate_py/agent_supervisor/rescue/scientific_executor.py',
        'ipfs_accelerate_py/agent_supervisor/rescue/native_executor_health.py']
    pins=[write(root/r,b'"""modeled source metadata"""\n')for r in rels[:4]]
    native_health=write(tmp_path/'retained-native-health.py',b'"""modeled retained H metadata"""\n')
    inv={'schema':'legal-ir-scientific-source-inventory/v1','source_root':str(root),'source_commit':'a'*40,'source_tree':'b'*40,
        'file_count':len(pins),'files':copy.deepcopy(pins)}
    invref=write(tmp_path/'inventory.json',inv)
    commitment={'schema':'legal-ir-scientific-source-commitment/v1','source_root':str(root),'source_commit':'a'*40,
        'source_tree':'b'*40,'source_inventory':invref,'runtime_activation_or_task_authority':False}
    ref=write(tmp_path/'source.json',commitment);profile={'main_entrypoint':pins[3],'health_module':native_health,'source_descriptors':[*pins,native_health]}
    old=science.ORIGINAL_NAMED_ROOTS;science.ORIGINAL_NAMED_ROOTS=(tmp_path,)
    yield dict(ref=ref,commitment=commitment,inventory=inv,profile=profile,root=tmp_path)
    science.ORIGINAL_NAMED_ROOTS=old


def test_committed_producer_and_three_real_parent_descriptors_are_required(inventory_case):
    c=inventory_case;science.source_admission_source_inventory(c['ref'],c['profile'])
    extra=write(c['root']/'producer/ipfs_accelerate_py/extra.py',b'# unapproved producer role\n')
    c['profile']['source_descriptors'].append(extra)
    with pytest.raises(science.ExecutorHold,match='parent'):science.source_admission_source_inventory(c['ref'],c['profile'])
    c['profile']['source_descriptors'].pop()
    extra_DATA=write(c['root']/'producer/extra.json',{})
    c['profile']['source_descriptors'].append(extra_DATA)
    with pytest.raises(science.ExecutorHold,match='parent'):science.source_admission_source_inventory(c['ref'],c['profile'])
    c['profile']['source_descriptors'].pop()
    c['profile']['source_descriptors']=c['profile']['source_descriptors'][1:]
    with pytest.raises(science.ExecutorHold,match='parent'):science.source_admission_source_inventory(c['ref'],c['profile'])

@pytest.mark.parametrize('mutation',['foreign_root','foreign_tree','missing_main','duplicate_path','not_source_only'])
def test_source_commitment_and_inventory_cannot_be_an_unjoined_document(inventory_case,mutation):
    c=inventory_case
    if mutation=='foreign_root':c['commitment']['source_root']+='/elsewhere'
    elif mutation=='foreign_tree':c['inventory']['source_tree']='c'*40
    elif mutation=='missing_main':c['inventory']['files'][3]['sha256']='f'*64
    elif mutation=='duplicate_path':c['inventory']['files'][0]=copy.deepcopy(c['inventory']['files'][1])
    else:c['commitment']['runtime_activation_or_task_authority']=True
    c['commitment']['source_inventory']=write(c['root']/'inventory.json',c['inventory']);ref=write(c['root']/'source.json',c['commitment'])
    with pytest.raises(science.ExecutorHold):science.source_admission_source_inventory(ref,c['profile'])

@pytest.mark.parametrize('outcomes,expected',[
    (['defer','defer','defer'],[(1,1),(1,2),(1,3)]),
    (['defer','success','defer','defer','defer'],[(1,1),(1,2),(2,3),(2,4),(2,5)]),
])
def test_physical_budget_counts_deferrals_and_health_sequence_only_counts_success(tmp_path,health,monkeypatch,outcomes,expected):
    root=tmp_path/'proof';root.mkdir(mode=0o700);config={'max_cycles':len(outcomes),'max_proof_bytes':1024**3,'poll_seconds':1}
    profile={'proof_root':str(root),'receipt_path':str(root/'health.json')};configref={'path':'model-config'};profileref={'path':'model-profile'}
    health._document=lambda ref:config;health._profile=lambda ref:profile
    monkeypatch.setattr(science,'load_configuration',lambda ref:(config,profileref,profile))
    monkeypatch.setattr(science,'require_main_process',lambda *args:{'pid':123,'birth_ticks':456})
    monkeypatch.setattr(science,'check_proof_budget',lambda *args,**kwargs:None)
    monkeypatch.setattr(science,'observe_native',lambda config:{})
    monkeypatch.setattr(science,'select_template',lambda *args:({'path':'template'},'genuine_scoped_idle'))
    calls=[]
    def run(*args,cycle_number=None):
        sequence=args[5];cycle_number=sequence if cycle_number is None else cycle_number;calls.append((sequence,cycle_number));cycle=root/('cycle-%08d-%s'%(cycle_number,'a'*32));cycle.mkdir()
        science.put(cycle/'intent.json',dict(kind='genuine_scoped_idle',sequence=sequence,cycle_number=cycle_number))
        if outcomes[len(calls)-1]=='defer':raise science.SourceAdmissionDeferred(str(cycle/'admission-deferred.json'))
    monkeypatch.setattr(science,'run_cycle',run)
    stop=SimpleNamespace(is_set=lambda:False,wait=lambda seconds:False)
    assert science.serve(configref,stop)==science.HOLD_EXIT
    assert calls==expected
    hold=json.loads((root/'executor-held.json').read_bytes());assert hold['reason']=='source_admission_consecutive_defer_bound'
    assert not(root/'health.json').exists() and len(list(root.glob('cycle-*')))==len(outcomes)


def test_recovery_mode_is_explicitly_profile_and_environment_pinned(tmp_path,health):
    ref=write(tmp_path/'authorization.json',{})
    with pytest.raises(science.ExecutorHold,match='unit_pinned'):
        science.reconcile_source_admission_recovery(ref,{}, {},{}, {})


@pytest.mark.parametrize('manager,age',[
    ({'MainPID':'12','ActiveState':'failed'},0),
    ({'MainPID':'0','ActiveState':'active'},0),
    ({'MainPID':'0','ActiveState':'failed'},-1),
    ({'MainPID':'0','ActiveState':'failed'},301),
    ({'MainPID':'0','ActiveState':'failed'},float('nan')),
])
def test_recovery_rejects_live_manager_future_stale_or_nonfinite_admin_facts(health,tmp_path,manager,age):
    profile={'max_age_seconds':300,'unit':{'name':'model-science.service'}};ref={'path':'model-profile'};health._profile=lambda r:profile
    value={key:None for key in science.ADMISSION_KEYS};value.update(schema=science.ADMISSION_SCHEMA,
        authorization='resume_idle_after_closed_source_admission_timeout',profile_ref=ref,
        manager_before={'Id':'model-science.service',**manager},recorded_at=time.time()-age)
    with pytest.raises(science.ExecutorHold,match='admin_clock_or_manager'):
        science._validate_source_admission_recovery(value,ref,profile,{})


def test_recovery_control_snapshot_must_be_frozen(health,tmp_path):
    profile={'max_age_seconds':300,'unit':{'name':'model-science.service'}};ref={'path':'model-profile'};health._profile=lambda r:profile
    value={key:None for key in science.ADMISSION_KEYS};value.update(schema=science.ADMISSION_SCHEMA,
        authorization='resume_idle_after_closed_source_admission_timeout',profile_ref=ref,
        manager_before={'Id':'model-science.service','MainPID':'0','ActiveState':'failed'},recorded_at=time.time())
    foreign=copy.deepcopy(value);foreign['manager_before']['Id']='foreign.service'
    with pytest.raises(science.ExecutorHold,match='admin_clock_or_manager'):
        science._validate_source_admission_recovery(foreign,ref,profile,{})
    for field in ('prior_profile_snapshot_ref','prior_config_snapshot_ref','negative_audit_ref','scheduler_snapshot_ref','current_native_ref','source_commitment_ref'):
        value[field]=write(tmp_path/(field+'.json'),{},0o400)
    Path(value['source_commitment_ref']['path']).chmod(0o600)
    with pytest.raises(science.ExecutorHold,match='not_frozen'):
        science._validate_source_admission_recovery(value,ref,profile,{})


def test_proc_permission_unknown_is_never_inferred_as_PID_absence(monkeypatch):
    original=Path.lstat;pid=2**30+993;target=Path('/proc')/str(pid)
    def denied(path,*args,**kwargs):
        if path==target:raise PermissionError('modeled unreadable proc entry')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'lstat',denied)
    with pytest.raises(science.ExecutorHold,match='observation_unknown'):
        science.source_admission_PID_absent(pid)


@pytest.mark.parametrize('missing_role',[None,'owner_source','driver_source'])
def test_original_waited_idle_deferral_binds_both_native_roles_without_health(negative,health,monkeypatch,missing_role):
    n=negative;profile={'unit':{'argv':['/python']},'owner_source':{'path':'/original-owner.py','sha256':'a'*64},
        'driver_source':{'path':'/original-driver.py','sha256':'b'*64}}
    n['request']['source_sha256']={profile[role]['path']:profile[role]['sha256']for role in ('owner_source','driver_source')}
    if missing_role:n['request']['source_sha256'].pop(profile[missing_role]['path'])
    n['request_ref']=write(n['root']/'request.json',n['request'])
    n['terminal']['request_sha256']=n['request_ref']['sha256'];write(n['root']/'owner.stdout',n['terminal'])
    class ModeledWaitedOriginalOwner:
        returncode=1
        def poll(self):return self.returncode
    monkeypatch.setattr(science.subprocess,'Popen',ModeledWaitedOriginalOwner)
    fields={'sequence':7,'process':{'pid':123},'owner_observation':dict(pid=n['terminal']['owner_pid'],
        boot_id=n['terminal']['owner_boot_id'],birth_ticks=123,argv=['/python','-B',profile['owner_source']['path'],
        '--request',n['request_ref']['path'],'--expected-request-sha256',n['request_ref']['sha256']])}
    if missing_role:
        with pytest.raises(science.ExecutorHold,match='source_argv'):
            science.source_admission_deferral(profile,n['request_ref'],n['request'],n['root'],'genuine_scoped_idle',ModeledWaitedOriginalOwner(),fields,1)
        assert not(n['root']/'admission-deferred.json').exists()
    else:
        ref=science.source_admission_deferral(profile,n['request_ref'],n['request'],n['root'],'genuine_scoped_idle',ModeledWaitedOriginalOwner(),fields,1)
        result=json.loads(Path(ref['path']).read_bytes())
        assert result['health_sequence_advanced']is False and result['health_or_task_completion_authority']is False
        assert result['automatic_retry_of_original_request']is False
        assert not(n['root']/'closed.json').exists()and not(n['root']/'cycle-proof.json').exists()


def test_recovered_epoch_rechecks_retained_census_before_each_new_attempt(tmp_path,health,monkeypatch):
    root=tmp_path/'proof';root.mkdir(mode=0o700);old_hold=write(root/'executor-held.json',{'old_negative':True})
    config={'max_cycles':50,'max_proof_bytes':1024**3,'poll_seconds':1}
    profile={'proof_root':str(root),'receipt_path':str(root/'new-health.json')};configref={'path':'model-config'};profileref={'path':'model-profile'}
    recovery={'schema':science.ADMISSION_SCHEMA,'prior_generation_authorization_ref':{'path':'old-generation'}}
    census={'path':'immutable-old-census'};generation={'retained_generations':[census]}
    health._document=lambda ref:recovery if ref['path']=='recovery'else generation if ref['path']=='old-generation'else config
    health._profile=lambda ref:profile
    monkeypatch.setattr(science,'load_configuration',lambda ref:(config,profileref,profile))
    monkeypatch.setattr(science,'require_main_process',lambda *args:{'pid':123,'birth_ticks':456})
    monkeypatch.setattr(science,'reconcile_source_admission_recovery',lambda *args:(48,1))
    monkeypatch.setattr(science,'check_proof_budget',lambda *args,**kwargs:None)
    monkeypatch.setattr(science,'observe_native',lambda config:{})
    monkeypatch.setattr(science,'select_template',lambda *args:({'path':'template'},'genuine_scoped_idle'))
    checks=[];calls=[];monkeypatch.setattr(science,'_generation_census',lambda ref:checks.append(ref))
    def deferred(*args,cycle_number):
        calls.append((args[5],cycle_number));cycle=root/('cycle-%08d-%s'%(cycle_number,'b'*32));cycle.mkdir()
        science.put(cycle/'intent.json',dict(kind='genuine_scoped_idle',sequence=args[5],cycle_number=cycle_number))
        raise science.SourceAdmissionDeferred(str(cycle/'admission-deferred.json'))
    monkeypatch.setattr(science,'run_cycle',deferred)
    stop=SimpleNamespace(is_set=lambda:False,wait=lambda seconds:False)
    assert science.serve(configref,stop,recovery_ref={'path':'recovery'})==science.HOLD_EXIT
    assert checks==[census,census]and calls==[(1,49),(1,50)]
    assert pin(root/'executor-held.json')==old_hold and not(root/'new-health.json').exists()
