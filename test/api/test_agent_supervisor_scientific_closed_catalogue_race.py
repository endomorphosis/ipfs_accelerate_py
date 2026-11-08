"""Owning protocol fixtures, with no genuine native/manager/closure credit."""
from pathlib import Path
from types import SimpleNamespace
import copy,json,threading
import pytest
from ipfs_accelerate_py.agent_supervisor.rescue import scientific_executor as science


@pytest.fixture
def race(monkeypatch):
    class OriginalVerifierError(ValueError):pass
    docs={
        'proof':{'kind':'genuine_scoped_idle','finished_at':1002.0,'finished_monotonic_ns':1002000,
            'native_diagnostic_ref':{'path':'native'},'result_ref':{'path':'result'},
            'request_ref':{'path':'request'},'terminal_ref':{'path':'terminal'}},
        'native':{'event_cursor':271,'tasks':[{'task_cid':'one','status':'completed','revision':3}]},
        'result':{'schema':'native-embedded-daemon-observation@1','task_scope':['one'],
            'ready_task_count':0,'selection_idle_reason':'no_scoped_ready_tasks','write_count':0,
            'unchanged':True,'provider_dispatched':False,'attempt_consumed':False,
            'completion_authority':False,'catalogue_binding':{'event_cursor':269,
                'task_statuses':[{'task_cid':'one','status':'blocked','revision':2}]}},
        'request':{'database_preimage':{'path':'db','bytes':1,'sha256':'a'*64}},
        'terminal':{'absolute_resource_operation_deadline_closed':True,
            'task_completion_authority_granted_by_resource_owner':False,
            'database_preimage_before_grants':{'path':'db','bytes':1,'sha256':'a'*64},
            'database_preimage_observed':{'path':'db','bytes':1,'sha256':'a'*64},
            'database_postimage':{'path':'db','bytes':1,'sha256':'a'*64}}}
    calls=[]
    def keys(v,expected,reason):science.need(type(v)is dict and set(v)==set(expected),reason)
    def reject(*args,**kwargs):
        calls.append((args,kwargs));raise OriginalVerifierError('embedded_idle_catalogue_cursor_changed')
    h=SimpleNamespace(ExecutorHealthError=OriginalVerifierError,_document=lambda r:docs[r['path']],
        _native=lambda n,p,**kw:n,_cycle=reject,_keys=keys,
        _integer=lambda v,lo=0,hi=2**63-1:type(v)is int and lo<=v<=hi)
    monkeypatch.setattr(science,'health_owner',lambda:h)
    return dict(docs=docs,health=h,calls=calls,profile={'task_scope':['one']},identity={'pid':42},
        proof_ref={'path':'proof'})


def recognize(r):
    return science.closed_catalogue_race_rejection(r['proof_ref'],r['profile'],r['identity'])


def test_exact_closed_quiet_cursor_race_stays_rejected_negative(race):
    r=race;before=copy.deepcopy(r['docs']);out=recognize(r)
    assert out['rejection']=='embedded_idle_catalogue_cursor_changed'
    assert out['from_cursor']==269 and out['to_cursor']==271
    assert out['task_changes']==[{'task_cid':'one','before':{'task_cid':'one','status':'blocked','revision':2},
        'after':{'task_cid':'one','status':'completed','revision':3}}]
    assert out['health_or_task_credit']is False and out['original_proof_still_rejected']is True
    assert r['docs']==before
    assert r['calls'][0][1]=={'now':1002.0,'monotonic_ns':1002000}


@pytest.mark.parametrize('reason',['original_cycle_not_closed','original_process_closure_missing',
    'original_owner_process_join','original_owner_request_join','original_driver_source_or_argv_changed',
    'original_import_origin_unbound','original_idle_result_not_quiet','descriptor_changed'])
def test_any_other_original_verifier_failure_stops(race,reason):
    def refused(*a,**k):raise race['health'].ExecutorHealthError(reason)
    race['health']._cycle=refused
    with pytest.raises(science.ExecutorHold,match='not_exact'):recognize(race)


@pytest.mark.parametrize('error',[KeyboardInterrupt(),SystemExit(15),RuntimeError('unknown')])
def test_signal_or_unknown_original_exception_is_never_a_deferral(race,error):
    def refused(*a,**k):raise error
    race['health']._cycle=refused
    with pytest.raises(type(error)):recognize(race)


def test_a_valid_original_proof_is_not_relabeled_as_a_negative(race):
    race['health']._cycle=lambda *a,**k:race['docs']['proof']
    with pytest.raises(science.ExecutorHold,match='proof_was_valid'):recognize(race)


@pytest.mark.parametrize('where,status',[('native','ready'),('native','running'),('native','claimed'),
    ('embedded','ready'),('embedded','running'),('embedded','unknown')])
def test_runnable_busy_or_unknown_status_stops(race,where,status):
    if where=='native':race['docs']['native']['tasks'][0]['status']=status
    else:race['docs']['result']['catalogue_binding']['task_statuses'][0]['status']=status
    with pytest.raises(science.ExecutorHold):recognize(race)


@pytest.mark.parametrize('field,value',[('ready_task_count',True),('ready_task_count',1),
    ('write_count',True),('write_count',1),('provider_dispatched',True),('attempt_consumed',True),
    ('completion_authority',True),('unchanged',False),('selection_idle_reason','busy')])
def test_control_effects_or_nonquiet_read_are_not_races(race,field,value):
    race['docs']['result'][field]=value
    with pytest.raises(science.ExecutorHold,match='not_quiet_readonly'):recognize(race)


@pytest.mark.parametrize('change',['same_cursor','regressed_cursor','same_revision','regressed_revision',
    'missing_task','duplicate_task','foreign_task','boolean_revision'])
def test_unproved_status_cursor_or_scope_change_stops(race,change):
    n=race['docs']['native']
    if change=='same_cursor':n['event_cursor']=269
    elif change=='regressed_cursor':n['event_cursor']=268
    elif change=='same_revision':n['tasks'][0]['revision']=2
    elif change=='regressed_revision':n['tasks'][0]['revision']=1
    elif change=='missing_task':n['tasks']=[]
    elif change=='duplicate_task':n['tasks']*=2
    elif change=='foreign_task':n['tasks'][0]['task_cid']='foreign'
    else:n['tasks'][0]['revision']=True
    with pytest.raises(science.ExecutorHold):recognize(race)


@pytest.mark.parametrize('field,value',[('absolute_resource_operation_deadline_closed',False),
    ('task_completion_authority_granted_by_resource_owner',True),
    ('database_preimage_before_grants',None),('database_preimage_observed',None),('database_postimage',None)])
def test_original_db_or_deadline_closure_cannot_be_inferred(race,field,value):
    race['docs']['terminal'][field]=value
    with pytest.raises(science.ExecutorHold,match='not_closed'):recognize(race)


def test_only_explicit_new_mode_and_existing_config_producer_slots_may_move():
    old=['python','-B','/old/main.py','serve','--config','/old/config','--config-sha256','a',
        '--known-source-admission-recovery','/old/grant']
    new=['python','-B','/new/main.py','serve','--config','/new/config','--config-sha256','b',
        '--known-closed-catalogue-race-recovery','/new/grant']
    assert science._catalogue_race_unit_projection(old,'/old/main.py')==science._catalogue_race_unit_projection(new,'/new/main.py')
    new.insert(1,'-O')
    assert science._catalogue_race_unit_projection(old,'/old/main.py')!=science._catalogue_race_unit_projection(new,'/new/main.py')


@pytest.fixture
def loop_case(tmp_path,monkeypatch):
    root=tmp_path/'proof';root.mkdir(mode=0o700)
    profile={'proof_root':str(root),'receipt_path':str(root/'absent-health')}
    config={'max_cycles':5,'max_proof_bytes':1000,'poll_seconds':1}
    class Keys:
        @staticmethod
        def _path(x):return Path(x)
        @staticmethod
        def _document(x):return config
        @staticmethod
        def _profile(x):return profile
    monkeypatch.setattr(science,'health_owner',lambda:Keys)
    monkeypatch.setattr(science,'load_configuration',lambda r:(config,{},profile))
    monkeypatch.setattr(science,'require_main_process',lambda *a:{'pid':42})
    monkeypatch.setattr(science,'check_proof_budget',lambda *a,**k:{})
    monkeypatch.setattr(science,'observe_native',lambda c:{})
    monkeypatch.setattr(science,'select_template',lambda *a:({},'genuine_scoped_idle'))
    held=[];failures=[];seen=[]
    monkeypatch.setattr(science,'persistent_hold',lambda *a:held.append(a))
    monkeypatch.setattr(science,'_recovery_failure',lambda *a:failures.append(a))
    class Stop:
        def is_set(self):return False
        def wait(self,s):return False
    return dict(root=root,held=held,failures=failures,seen=seen,stop=Stop())


def test_three_closed_race_negatives_spend_positions_without_advancing_health(loop_case,monkeypatch):
    c=loop_case
    def run(*args,**kwargs):
        c['seen'].append((args[5],kwargs.get('cycle_number')))
        raise science.CatalogueRaceDeferred(str(c['root']/'negative.json'))
    monkeypatch.setattr(science,'run_cycle',run)
    assert science.serve({},c['stop'])==78
    assert c['seen']==[(1,None),(1,2),(1,3)]
    assert c['held'][-1][-1]=='catalogue_race_consecutive_defer_bound'
    assert c['failures'][-1][1:3]==(3,1)


def test_success_resets_race_bound_and_only_then_advances_sequence(loop_case,monkeypatch):
    c=loop_case;outcomes=iter(['race','ok','race','race','race'])
    def run(*args,**kwargs):
        c['seen'].append((args[5],kwargs.get('cycle_number')))
        if next(outcomes)=='race':raise science.CatalogueRaceDeferred(str(c['root']/'negative.json'))
    monkeypatch.setattr(science,'run_cycle',run)
    assert science.serve({},c['stop'])==78
    assert c['seen']==[(1,None),(1,2),(2,3),(2,4),(2,5)]
    assert c['failures'][-1][1:3]==(5,2)


def test_unknown_cycle_failure_does_not_poll_again(loop_case,monkeypatch):
    c=loop_case
    def run(*args,**kwargs):c['seen'].append(args[5]);raise RuntimeError('unknown original failure')
    monkeypatch.setattr(science,'run_cycle',run)
    assert science.serve({},c['stop'])==78
    assert c['seen']==[1]
    assert c['held'][-1][-1]=='unknown original failure'


@pytest.fixture
def parser_case(tmp_path,monkeypatch):
    import hashlib,os,sys
    path=tmp_path/'native_catalogue_probe.py'
    def write(body):
        if path.exists():path.chmod(0o600)
        path.write_bytes(body);path.chmod(0o400)
        return {'path':str(path),'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}
    raw=b"def observe_native_catalogue(database, population, **binding):\n    return {'details': {'native_catalogue': {'source': 'authenticated-raw', 'binding': binding}}}\n"
    ref=write(raw)
    def bound(pin):
        data=Path(pin['path']).read_bytes()
        science.need(len(data)==pin['bytes']and hashlib.sha256(data).hexdigest()==pin['sha256'],
            'descriptor_changed')
        return data
    monkeypatch.setattr(science,'health_owner',lambda:SimpleNamespace(_bound=bound))
    names=set(sys.modules)
    yield dict(path=path,raw=raw,ref=ref,write=write,bound=bound)
    for name in set(sys.modules)-names:
        if name.startswith('_ipfs_scientific_catalogue_parser_'):sys.modules.pop(name,None)


def test_selected_parser_compiles_real_raw_source_and_preserves_opaque_origin(parser_case):
    c=parser_case;parser=science.load_explicit_catalogue_parser(c['ref'])
    assert parser.module.__file__==parser.observer.__code__.co_filename==str(c['path'])
    assert parser.module.__spec__.origin==str(c['path'])and parser.module.__package__==''
    assert parser.descriptor==c['ref']


def test_cached_pyc_and_default_loader_get_code_cannot_replace_parser_source(parser_case,monkeypatch):
    import importlib.machinery,importlib.util,importlib._bootstrap_external
    c=parser_case;cache=c['path'].parent/'__pycache__'/(c['path'].stem+'.cpython-312.pyc')
    cache.parent.mkdir()
    evil=compile("raise AssertionError('cached parser body executed')",str(c['path']),'exec')
    cache.write_bytes(importlib._bootstrap_external._code_to_timestamp_pyc(evil,
        mtime=int(c['path'].stat().st_mtime),source_size=len(c['raw'])))
    def forbidden(*args,**kwargs):raise AssertionError('default cache-capable get_code used')
    monkeypatch.setattr(importlib.machinery.SourceFileLoader,'get_code',forbidden)
    parser=science.load_explicit_catalogue_parser(c['ref'])
    assert parser.observer(None,None)['details']['native_catalogue']['source']=='authenticated-raw'
    assert cache.exists()


def test_parser_source_change_between_admission_and_raw_compile_refuses(parser_case,monkeypatch):
    c=parser_case;calls=[]
    def bound(pin):
        calls.append(True)
        if len(calls)==2:c['write'](c['raw']+b'\n# changed source\n')
        return c['bound'](pin)
    monkeypatch.setattr(science,'health_owner',lambda:SimpleNamespace(_bound=bound))
    with pytest.raises(science.ExecutorHold,match='descriptor_changed'):
        science.load_explicit_catalogue_parser(c['ref'])


def test_parser_source_change_during_module_execution_refuses(parser_case):
    c=parser_case
    raw=("from pathlib import Path\n"
         "p=Path(__file__)\np.chmod(0o600)\np.write_bytes(p.read_bytes()+b'\\n# raced\\n')\n"
         "def observe_native_catalogue(*args, **kwargs):\n    return {}\n").encode()
    ref=c['write'](raw)
    with pytest.raises(science.ExecutorHold,match='descriptor_changed'):
        science.load_explicit_catalogue_parser(ref)


def test_preexisting_opaque_module_is_not_parser_source_authority(parser_case,monkeypatch):
    import sys,types
    c=parser_case;name='_ipfs_scientific_catalogue_parser_'+c['ref']['sha256']
    monkeypatch.setitem(sys.modules,name,types.ModuleType(name))
    with pytest.raises(science.ExecutorHold,match='already_present'):
        science.load_explicit_catalogue_parser(c['ref'])


def test_selected_parser_receives_only_unchanged_catalogue_kwargs(parser_case):
    c=parser_case;parser=science.load_explicit_catalogue_parser(c['ref'])
    binding={'database_path':'/db','population_path':'/pop','board_id':'legal-ir-scientific',
        'board_namespace':'board','plan_root_cid':'plan','repository_tree_id':'tree',
        'population_sha256':'a'*64,'max_age_seconds':180,'timeout_seconds':15}
    observed=science.observe_native({'catalogue':binding},catalogue_parser=parser)
    assert observed['binding']=={k:v for k,v in binding.items()if k not in ('database_path','population_path')}


def test_selected_parser_source_change_after_constructor_is_not_hidden_by_cached_module(parser_case):
    c=parser_case;parser=science.load_explicit_catalogue_parser(c['ref'])
    c['write'](c['raw']+b'\n# changed after constructor\n')
    with pytest.raises(science.ExecutorHold,match='descriptor_changed'):
        science.observe_native({'catalogue':{}},catalogue_parser=parser)


def test_arbitrary_callable_cannot_enter_selected_parser_route():
    with pytest.raises(science.ExecutorHold,match='unselected_callable'):
        science.observe_native({'catalogue':{}},catalogue_parser=lambda *a: {})


def test_new_parser_pair_is_projected_only_as_an_explicit_pair():
    old=['python','-B','/old/main.py','serve','--config','/old/config','--config-sha256','a',
        '--known-source-admission-recovery','/old/grant']
    new=['python','-B','/new/main.py','serve','--config','/new/config','--config-sha256','b',
        '--known-closed-catalogue-race-recovery','/new/grant','--catalogue-parser-source','/new/native_catalogue_probe.py',
        '--catalogue-parser-sha256','f'*64]
    assert science._catalogue_race_unit_projection(old,'/old/main.py')==science._catalogue_race_unit_projection(new,'/new/main.py')
    new.remove('--catalogue-parser-sha256')
    with pytest.raises(science.ExecutorHold,match='parser_pair'):
        science._catalogue_race_unit_projection(new,'/new/main.py')


@pytest.fixture
def reconciliation(tmp_path,monkeypatch):
    import hashlib
    root=tmp_path/'proof';root.mkdir(mode=0o700)
    def pin(path):
        raw=Path(path).read_bytes()
        return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    grant=root/'grant.json'
    value={'global_hold_ref':{'path':'oldhold','bytes':1,'sha256':'a'*64},
        'failed_cycle_hold_ref':{'path':'failedhold','bytes':1,'sha256':'b'*64},
        'prior_source_admission_ack_ref':{'path':'oldack','bytes':1,'sha256':'c'*64}}
    grant.write_text(json.dumps(value));grant.chmod(0o400);ref=pin(grant)
    profile={'proof_root':str(root),'unit':{'argv':['python','serve',
        '--known-closed-catalogue-race-recovery',str(grant)]}}
    h=SimpleNamespace(_document=lambda r:json.loads(Path(r['path']).read_text()),descriptor=pin)
    monkeypatch.setattr(science,'health_owner',lambda:h)
    monkeypatch.setenv(science.CATALOGUE_RACE_SHA_ENV,ref['sha256'])
    validations=[]
    def validated(*args):validations.append(True);return 162,[]
    monkeypatch.setattr(science,'_validate_closed_catalogue_race_recovery',validated)
    return dict(root=root,ref=ref,grant=grant,value=value,profile=profile,validations=validations)


def test_one_use_negative_ack_carries162_and_never_writes_health_or_old_hold(reconciliation):
    c=reconciliation;before=c['grant'].read_bytes()
    spent,censuses=science.reconcile_closed_catalogue_race_recovery(c['ref'],{},c['profile'],{}, {'pid':42})
    assert spent==162 and censuses==[]and c['grant'].read_bytes()==before
    ack=json.loads((c['root']/'catalogue-race-reconciliation-v01.json').read_text())
    assert ack['spent_cycle_count']==162 and ack['next_cycle_number']==163
    assert ack['generation_number']==ack['max_generations']==2
    assert ack['no_health_or_task_credit_for_failed_cycle']is True
    assert ack['prior_source_admission_ack_ref']==c['value']['prior_source_admission_ack_ref']
    assert sorted(p.name for p in c['root'].iterdir())==['catalogue-race-reconciliation-v01.json','grant.json']


def test_consumed_negative_ack_cannot_be_replayed_or_replaced(reconciliation):
    c=reconciliation
    science.reconcile_closed_catalogue_race_recovery(c['ref'],{},c['profile'],{}, {'pid':42})
    ack=c['root']/'catalogue-race-reconciliation-v01.json';before=ack.read_bytes()
    with pytest.raises(science.ExecutorHold,match='already_consumed'):
        science.reconcile_closed_catalogue_race_recovery(c['ref'],{},c['profile'],{}, {'pid':43})
    assert ack.read_bytes()==before and c['validations']==[True]


@pytest.mark.parametrize('fault',['environment','other_argv_path','crossed_mode','mutable_grant'])
def test_unpinned_or_crossed_grant_never_reaches_validation_or_ack(reconciliation,monkeypatch,fault):
    c=reconciliation
    if fault=='environment':monkeypatch.setenv(science.CATALOGUE_RACE_SHA_ENV,'d'*64)
    elif fault=='other_argv_path':c['profile']['unit']['argv'][-1]='/foreign/grant'
    elif fault=='crossed_mode':c['profile']['unit']['argv']+=['--known-source-admission-recovery','/old/grant']
    else:c['grant'].chmod(0o600)
    with pytest.raises(science.ExecutorHold,match='not_unit_pinned'):
        science.reconcile_closed_catalogue_race_recovery(c['ref'],{},c['profile'],{}, {'pid':42})
    assert not c['validations']and not (c['root']/'catalogue-race-reconciliation-v01.json').exists()


@pytest.mark.parametrize('reason',['prior_Main_live_or_unknown','prior_owner_live_or_unknown',
    'unreviewed_population_change','original_event_capture_join','counter_or_authority_path_changed',
    'spent_history_reset','original_backend_or_source_removed','original_native_lease_unknown'])
def test_failed_original_admission_never_writes_negative_ack(reconciliation,monkeypatch,reason):
    c=reconciliation
    def refused(*args):raise science.ExecutorHold(reason)
    monkeypatch.setattr(science,'_validate_closed_catalogue_race_recovery',refused)
    with pytest.raises(science.ExecutorHold,match=reason):
        science.reconcile_closed_catalogue_race_recovery(c['ref'],{},c['profile'],{}, {'pid':42})
    assert not (c['root']/'catalogue-race-reconciliation-v01.json').exists()


@pytest.fixture
def owner_transition(tmp_path,monkeypatch):
    import hashlib,stat
    def pin(path):
        data=Path(path).read_bytes()
        return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    def write(path,value):
        path=Path(path);path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        if path.exists():path.chmod(0o600)
        data=value if type(value)is bytes else(json.dumps(value,sort_keys=True)+'\n').encode()
        path.write_bytes(data);path.chmod(0o400);return pin(path)
    def bound(ref):
        science.need(pin(ref['path'])==ref,'modeled_source_descriptor_changed')
        return Path(ref['path']).read_bytes()
    def keys(value,expected,reason):science.need(type(value)is dict and set(value)==set(expected),reason)
    h=SimpleNamespace(_document=lambda r:json.loads(bound(r)),_bound=bound,_keys=keys)
    monkeypatch.setattr(science,'health_owner',lambda:h)
    monkeypatch.setattr(science,'ORIGINAL_NAMED_ROOTS',(tmp_path,))
    # Qualification receipt itself is a separate real-native gate. This
    # fixture models that admitted source gate while testing OW role structure.
    monkeypatch.setattr(science,'catalogue_race_native_owner_qualification',lambda *a: {'modeled_source_gate':True})
    root=tmp_path/'OW';prefix='ipfs_accelerate_py/agent_supervisor/rescue/'
    relatives=['ipfs_accelerate_py/__init__.py','ipfs_accelerate_py/agent_supervisor/__init__.py',
        prefix+'__init__.py',prefix+'original_embedded_observation_owner.py']
    pins=[write(root/name,('# fixture '+name+'\n').encode())for name in relatives]
    native=tmp_path/'H';producer=tmp_path/'D'
    prior_owner={'path':str(native/prefix/'original_embedded_observation_owner.py'),'bytes':72407,
        'sha256':'e243cdf7b22a63f55e3c55101a2a7aeeee31492a98c8dbfe9f05b4d43b7fdc06'}
    driver={'path':str(native/prefix/'native_embedded_observation_driver.py'),'bytes':28162,
        'sha256':'2e9ba7653bbc0939bb41f1fbbdd14f5f0e9a51fd78577dbbb5aa044fc14e3ea6'}
    health={'path':str(native/prefix/'native_executor_health.py'),'bytes':57074,'sha256':'a'*64}
    prior={'owner_source':prior_owner,'driver_source':driver,'health_module':health}
    profile={'owner_source':pins[-1],'driver_source':dict(driver),'health_module':dict(health),
        'main_entrypoint':{'path':str(producer/prefix/'scientific_executor.py')},'source_descriptors':pins}
    inventory={'schema':'legal-ir-scientific-source-inventory/v1','source_root':str(root),
        'source_commit':'b'*40,'source_tree':'c'*40,'file_count':4,
        'files':[dict(p,mode=stat.S_IMODE(Path(p['path']).stat().st_mode))for p in pins]}
    inv=write(tmp_path/'owner-inventory.json',inventory)
    value={'schema':'agent-supervisor/scientific-native-owner-policy-source-commitment@1',
        'source_root':str(root),'source_commit':'b'*40,'source_tree':'c'*40,'source_inventory':inv,
        'source_operation_plan':write(tmp_path/'source-operation.json',{'scope':'modeled_source_only'}),
        'source_qualification':write(tmp_path/'source-qualification.json',{'scope':'modeled_unit_fixture'}),
        'owner_source':pins[-1],'prior_owner_source':prior_owner,'native_backend_source':driver,
        'scheduler_policy_transition':{'prior_sha256':'4662bd8fe6caf6f2c048e3d7cff7abb4d80e5c875b21e02e204b1b89371b8286',
            'target_sha256':'f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661'},
        'runtime_activation_or_task_authority':False}
    ref=write(tmp_path/'owner-commitment.json',value)
    return dict(prior=prior,profile=profile,value=value,inventory=inventory,ref=ref,pins=pins,write=write,
        inventory_path=tmp_path/'owner-inventory.json',commitment_path=tmp_path/'owner-commitment.json')


def test_only_explicit_disjoint_fourfile_owner_transition_is_accepted(owner_transition):
    c=owner_transition
    value,pins=science.catalogue_race_native_owner_inventory(c['ref'],c['prior'],c['profile'])
    assert value==c['value']and set(pins)=={p['path']for p in c['pins']}
    assert c['profile']['driver_source']==c['prior']['driver_source']
    assert c['profile']['health_module']==c['prior']['health_module']


@pytest.mark.parametrize('fault',['crossed_owner','crossed_prior_owner','crossed_driver','crossed_health',
    'wrong_target_policy','wrong_prior_policy','producer_root_alias','H_root_alias',
    'wrong_source_generation','missing_parent','extra_source','mutable_owner','changed_owner_bytes',
    'extra_authority_field','renamed_owner'])
def test_crossed_owner_profile_policy_or_backend_never_broadens_legacy(owner_transition,fault):
    c=owner_transition
    if fault=='crossed_owner':c['profile']['owner_source']=dict(c['profile']['owner_source'],sha256='d'*64)
    elif fault=='crossed_prior_owner':c['value']['prior_owner_source']=dict(c['value']['prior_owner_source'],sha256='e'*64)
    elif fault=='crossed_driver':c['profile']['driver_source']=dict(c['profile']['driver_source'],sha256='f'*64)
    elif fault=='crossed_health':c['profile']['health_module']=dict(c['profile']['health_module'],sha256='f'*64)
    elif fault=='wrong_target_policy':c['value']['scheduler_policy_transition']['target_sha256']='f'*64
    elif fault=='wrong_prior_policy':c['value']['scheduler_policy_transition']['prior_sha256']='e'*64
    elif fault=='producer_root_alias':c['value']['source_root']=str(Path(c['profile']['main_entrypoint']['path']).parents[3])
    elif fault=='H_root_alias':c['value']['source_root']=str(Path(c['prior']['owner_source']['path']).parents[3])
    elif fault=='wrong_source_generation':c['inventory']['source_commit']='d'*40
    elif fault=='missing_parent':c['inventory']['files']=c['inventory']['files'][1:]
    elif fault=='extra_source':c['inventory']['files'].append(dict(c['inventory']['files'][0],path='/foreign/source.py'))
    elif fault=='mutable_owner':Path(c['profile']['owner_source']['path']).chmod(0o600)
    elif fault=='changed_owner_bytes':c['write'](c['profile']['owner_source']['path'],b'# changed after qualification\n')
    elif fault=='extra_authority_field':c['value']['provider_authority']=True
    else:
        previous=Path(c['pins'][-1]['path']);renamed=previous.with_name('renamed_owner.py')
        previous.rename(renamed);c['pins'][-1]['path']=str(renamed)
        c['inventory']['files'][-1]['path']=str(renamed)
    c['value']['source_inventory']=c['write'](c['inventory_path'],c['inventory'])
    ref=c['write'](c['commitment_path'],c['value'])
    with pytest.raises(science.ExecutorHold):
        science.catalogue_race_native_owner_inventory(ref,c['prior'],c['profile'])


@pytest.mark.parametrize('fault',['old_175_summary','extra_acceptance_field','cases_bool','phase_bool',
    'wrong_mode','unclosed_root','authority','not_all_raw','crossed_qualification'])
def test_owner128_requires_exact_shared_closed_receipt_before_deeper_original_reads(monkeypatch,fault):
    # Models only the first DATA/shape boundary. Actual original closure and
    #128/384 collection are independently required by the complete reader.
    keys=['schema','accepted','original_root_closeout','plan','request','scope','observation',
        'owner_source','owning_test_source','cases','natural_phases','selector_contract_mode',
        'raw_subjects','all_raw_subjects_compiled_and_executed','original_source_native_process_storage_closed',
        'runtime_activation_or_task_authority']
    ref={'path':'modeled-acceptance'}
    value={key:{}for key in keys}
    value.update(schema='legal-ir-native-owner128-source-acceptance/v1',accepted=True,cases=128,
        natural_phases=384,selector_contract_mode='whole_owner_file_128',all_raw_subjects_compiled_and_executed=True,
        original_source_native_process_storage_closed=True,runtime_activation_or_task_authority=False)
    owner={'source_qualification':ref}
    if fault=='old_175_summary':value={'accepted':True,'cases':175}
    elif fault=='extra_acceptance_field':value['other_authority']=False
    elif fault=='cases_bool':value['cases']=True
    elif fault=='phase_bool':value['natural_phases']=True
    elif fault=='wrong_mode':value['selector_contract_mode']='selected_functions'
    elif fault=='unclosed_root':value['original_source_native_process_storage_closed']=False
    elif fault=='authority':value['runtime_activation_or_task_authority']=True
    elif fault=='not_all_raw':value['all_raw_subjects_compiled_and_executed']=False
    else:owner['source_qualification']={'path':'other-acceptance'}
    def deeper(*args,**kwargs):raise AssertionError('invalid receipt passed the initial exact shared contract')
    h=SimpleNamespace(_document=lambda pin:value,_bound=deeper,descriptor=deeper,_raw=deeper)
    monkeypatch.setattr(science,'health_owner',lambda:h)
    with pytest.raises(science.ExecutorHold,match='actual exact Owner128'):
        science.catalogue_race_native_owner_qualification(ref,owner)


@pytest.fixture
def boundary_race(race):
    # Surrounding actual original source/native proof remains modeled by race;
    # these controls exercise only selected-G diagnostic scope and quiet rows.
    race['health']._text=lambda v:type(v)is str and bool(v)
    race['profile']['catalogue']={'population_sha256':'f'*64}
    graph={'schema':'native-observational-dependency-graph/v1',
        'population_envelope_ref':{'path':'/selected/envelope.json','bytes':500,'sha256':'f'*64},
        'prerequisite_body_semantics_qualified':False,'completion_authority':False,
        'dispatch_authority':False,'observed_task_completion_established':False,
        'observed_task_count':1,'node_count':2,'observed_tasks':[{'task_cid':'child','status':'blocked','revision':1}]}
    race['docs']['native'].update(population_schema='legal-ir-native-catalogue-boundary/v1',dependency_boundary=graph)
    return race


@pytest.mark.parametrize('status',['blocked','completed','skipped'])
def test_closed_negative_boundary_deferral_requires_quiet_observed_dependencies(boundary_race,status):
    c=boundary_race;c['docs']['native']['dependency_boundary']['observed_tasks'][0]['status']=status
    result=recognize(c)
    assert result['original_proof_still_rejected']is True and result['health_or_task_credit']is False


@pytest.mark.parametrize('fault',['ready','running','claimed','in_progress','unknown','wrong_graph_schema',
    'wrong_population_binding','authority','missing_graph','partial_population','duplicate_or_core_task'])
def test_nonquiet_unknown_or_crossed_boundary_observation_is_never_a_deferral(boundary_race,fault):
    c=boundary_race;native=c['docs']['native'];graph=native['dependency_boundary']
    if fault in ('ready','running','claimed','in_progress','unknown'):graph['observed_tasks'][0]['status']=fault
    elif fault=='wrong_graph_schema':graph['schema']='old-unqualified-graph/v0'
    elif fault=='wrong_population_binding':graph['population_envelope_ref']['sha256']='a'*64
    elif fault=='authority':graph['observed_task_completion_established']=True
    elif fault=='missing_graph':native.pop('dependency_boundary')
    elif fault=='partial_population':graph['observed_task_count']=2
    else:graph['observed_tasks'][0]['task_cid']='one'
    with pytest.raises(science.ExecutorHold):recognize(c)
