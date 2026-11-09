"""Pure source/constructor contracts; no scheduler, database, daemon or provider run.

The synthetic import modules below exercise the retained source guard. Constructor
success is only request parsing, and carries no actual resource/executor credit.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import pytest

BASE = Path(__file__).resolve().parents[2]
REL = Path('ipfs_accelerate_py/agent_supervisor/rescue')


def _load(name, filename):
    path=BASE/REL/filename
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def driver():
    return _load('_qualified_embedded_driver','native_embedded_observation_driver.py')


@pytest.fixture
def owner():
    return _load('_qualified_original_embedded_owner','original_embedded_observation_owner.py')


@pytest.fixture
def guard():
    return _load('_qualified_original_project_guard','project_import_source_guard.py')


def _pin(path):
    raw=path.read_bytes()
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def _write(path,value):
    path.write_text(json.dumps(value,sort_keys=True)+'\n')
    return _pin(path)


def _import_git_fixture(root, route):
    relative='ipfs_accelerate_py/agent_implementation_route.py'
    sql=['ipfs_accelerate_py/agent_supervisor/task_sources/sql/'+name for name in (
        '0001_control_plane.sql','0002_causal_event_federation_core.sql',
        '0003_state_server_restart_identity.sql','0004_hash_observations.sql','0005_tasks_index_repair.sql')]
    blobs=[{'relative_path':relative,'object_id':'0'*40,'bytes':route['bytes']}]
    blobs.extend({'relative_path':name,'object_id':str(index)*40,'bytes':1} for index,name in enumerate(sql,2))
    blobs.append({'relative_path':'ipfs_accelerate_py/agent_supervisor/task_sources/sql/0002_tasks_index_repair.sql',
        'object_id':'7'*40,'bytes':1})
    return {'blobs':blobs,
        'ls_tree_tail':['ls-tree','-r','-z','1'*40,'--','ipfs_accelerate_py',
            'scripts/ops/agent_supervisor/configured_board_scheduler.py',
            'scripts/ops/agent_supervisor/implementation_supervisor_entry.py'],
        'payload_bytes':route['bytes']+6,'required_relative_files':[relative,*sql],'route_source':route,
        'scope':'immutable HEAD bytes archived/hashed by original import capsule; no import or execution of omitted live source bodies',
        'selected_file_count':7,'source_head':'1'*40,'source_root':str(root),'unique_blob_count':7}


def _plan(driver,tmp_path):
    attempt=tmp_path/'attempt';attempt.mkdir(mode=0o700)
    scope=['baguqeeraa0001','baguqeeraa0002']
    root=driver.RUNTIME
    helper=root/'ipfs_accelerate_py/agent_supervisor/todo_daemon/native_owner_bootstrap.py'
    guard=root/REL/'project_import_source_guard.py'
    pins={str(p):_pin(p)['sha256'] for p in (helper,guard,Path(driver.__file__))}
    route={'path':str(root/'ipfs_accelerate_py/agent_implementation_route.py'),'bytes':1,'sha256':'0'*64}
    pins[route['path']]=route['sha256']
    prior={'path':str(driver.DATABASE),'bytes':1,'sha256':'0'*64}
    binding={'schema':'native-embedded-daemon-observation-binding@1','database_preimage':prior,
        'task_scope':scope,'plan_root_cid':'baguqeeraa-plan-root','repository_tree_id':'tree-v1',
        'observer_source':_pin(helper),'source_schema':'ipfs_accelerate_py/agent-supervisor/database-task-source@1'}
    return {'schema':'native-embedded-scoped-observation-plan/v1',
        'operation':'native_embedded_scoped_observation_v1','accelerate_root':str(root),'task_scope':scope,
        'native_metadata_cli':{'schema':'native-embedded-scoped-observation-cli/v2','runtime_root':str(root),
            'cwd':str(driver.WORKSPACE),'attempt_directory':str(attempt),'temporary_directory':str(attempt/'temporary'),
            'project_import_plan':{'accelerate_root':str(root),'datasets_root':str(driver.DATASETS),
                'publication_root':str(driver.DATASETS),'source_pins':[_pin(p) for p in (helper,guard)]},
            'pre_body_guard':_pin(guard),'result_path':str(attempt/'exports/native-cli-driver-result.json'),
            'source_sha256':pins,'observation_binding':binding,'import_head_capsule_Git':_import_git_fixture(root,route)}}


def test_closed_driver_constructor(driver,tmp_path,monkeypatch):
    monkeypatch.setattr(driver,'DATASETS',tmp_path.parent.parent.parent)
    plan=_plan(driver,tmp_path)
    # Only the constructor's path boundary is widened to this isolated fixture.
    attempt=Path(plan['native_metadata_cli']['attempt_directory'])
    monkeypatch.setattr(driver,'DATASETS',attempt.parents[2])
    # A declaration outside the canonical test-log tree must still be refused.
    with pytest.raises(RuntimeError,match='owned observation paths'):
        driver.configuration(plan)


@pytest.mark.parametrize('change',[
    'schema','operation','accelerate_root','scope_empty','scope_duplicate','scope_bad_type',
    'cli_schema','cli_runtime','cli_cwd','temporary','result','extra_cli','bad_pin',
    'binding_schema','binding_scope','database_path','observer_path','source_schema','guard_path',
])
def test_driver_rejects_wrong_closed_contract(driver,tmp_path,monkeypatch,change):
    plan=_plan(driver,tmp_path)
    c=plan['native_metadata_cli'];b=c['observation_binding']
    # Put the fixture attempt in the exact declaration tree without invoking it.
    monkeypatch.setattr(driver,'DATASETS',tmp_path)
    attempt=tmp_path/'workspace/test-logs/fresh-attempt'
    c['attempt_directory']=str(attempt);c['temporary_directory']=str(attempt/'temporary')
    c['result_path']=str(attempt/'exports/native-cli-driver-result.json')
    if change in ('schema','operation','accelerate_root'):plan[change]='bad'
    elif change=='scope_empty':plan['task_scope']=[]
    elif change=='scope_duplicate':plan['task_scope']=['same','same']
    elif change=='scope_bad_type':plan['task_scope']=[[]]
    elif change=='cli_schema':c['schema']='bad'
    elif change=='cli_runtime':c['runtime_root']='/wrong-root'
    elif change=='cli_cwd':c['cwd']='/wrong-cwd'
    elif change=='temporary':c['temporary_directory']=str(attempt/'wrong')
    elif change=='result':c['result_path']=str(attempt/'wrong.json')
    elif change=='extra_cli':c['provider']='enabled'
    elif change=='bad_pin':c['source_sha256']={'relative':'0'*64}
    elif change=='binding_schema':b['schema']='bad'
    elif change=='binding_scope':b['task_scope']=['different']
    elif change=='database_path':b['database_preimage']['path']='/wrong-db'
    elif change=='observer_path':b['observer_source']['path']='/wrong-helper'
    elif change=='source_schema':b['source_schema']=''
    elif change=='guard_path':c['pre_body_guard']['path']=str(tmp_path/'workspace-copy-guard.py')
    with pytest.raises(RuntimeError):driver.configuration(plan)


def test_valid_declaration_does_not_dispatch(driver,tmp_path,monkeypatch):
    plan=_plan(driver,tmp_path);c=plan['native_metadata_cli'];monkeypatch.setattr(driver,'DATASETS',tmp_path)
    attempt=tmp_path/'workspace/test-logs/fresh-attempt'
    c.update(attempt_directory=str(attempt),temporary_directory=str(attempt/'temporary'),
        result_path=str(attempt/'exports/native-cli-driver-result.json'))
    assert driver.configuration(plan) is c
    with pytest.raises(RuntimeError,match='does not dispatch'):
        driver.authorize_native_control_popen(['/usr/bin/true'],(),{},plan)
    assert driver.authorize_native_control_fd_event('os.chmod',(3,0o500,-1),plan) is None
    assert not attempt.exists()


def _request(owner,driver,tmp_path,monkeypatch):
    plan=_plan(driver,tmp_path)
    c=plan['native_metadata_cli']
    pins=c['source_sha256'];pins[str(Path(owner.__file__))]=_pin(Path(owner.__file__))['sha256']
    # Constructor tests use isolated source/JSON files and a declared descriptor;
    # no actual DB body, resource kernel, task claim or native owner is entered.
    p=tmp_path/'plan.json';ref=_write(p,plan)
    request={'schema':owner.SCHEMA,'purpose':'native_control_metadata','request_id':'fixture-observation',
        'argv':[sys.executable,'-B',str(owner.ACTOR_PATH),'--plan',str(p),'--plan-sha256',ref['sha256']],
        'cwd':str(owner.WORKSPACE),'attempt_directory':c['attempt_directory'],
        'source_sha256':{**pins,str(p):ref['sha256']},'cpu_slots':1,'memory_mb':1536,
        'child_process_slots':8,'storage_bytes':384*1024**2,'timeout_seconds':300,
        'scheduler_state_path':str(owner.STATE),'expected_scheduler_policy_sha256':owner.POLICY,
        'resource_ledger':str(owner.LEDGER),'source_roots':[str(owner.R),str(owner.A)],
        'external_files':copy.deepcopy(owner.EXTERNAL_FILES),'database_preimage':c['observation_binding']['database_preimage'],
        'allowed_subprocess_argv':copy.deepcopy(owner.READONLY_PROFILE_GIT)}
    monkeypatch.setattr(owner,'ROOTS',(*owner.ROOTS,tmp_path))
    return request,plan,p


def test_original_request_parse_has_no_database_or_native_effect(owner,driver,tmp_path,monkeypatch):
    value,plan,p=_request(owner,driver,tmp_path,monkeypatch)
    request_path=tmp_path/'request.json';ref=_write(request_path,value)
    paths=[];actual=owner.regular
    def recorded(path,*args):paths.append(str(path));return actual(path,*args)
    monkeypatch.setattr(owner,'regular',recorded)
    parsed,_=owner.request(request_path,ref['sha256'])
    assert parsed==value
    assert str(owner.DATABASE) not in paths
    assert list(Path(value['attempt_directory']).iterdir())==[]
    assert not (tmp_path/'execution-request.json').exists()


@pytest.mark.parametrize('change',['old_a_only','cpu','timeout','external_extra','database_pin',
    'missing_owner_pin','wrong_actor','nonempty_scope','guard_ref','observer_ref','runtime_root','guard_path'])
def test_original_request_rejects_wrong_binding(owner,driver,tmp_path,monkeypatch,change):
    value,plan,p=_request(owner,driver,tmp_path,monkeypatch)
    if change=='old_a_only':value['source_roots']=[str(owner.A)]
    elif change=='cpu':value['cpu_slots']=2
    elif change=='timeout':value['timeout_seconds']=301
    elif change=='external_extra':value['external_files'].append({'path':str(tmp_path/'extra'),'max_bytes':1})
    elif change=='database_pin':value['source_sha256'][str(owner.DATABASE)]='0'*64
    elif change=='missing_owner_pin':value['source_sha256'].pop(str(Path(owner.__file__)))
    elif change=='wrong_actor':value['argv'][2]='/wrong-driver.py'
    else:
        if change=='nonempty_scope':plan['task_scope']=[]
        elif change=='guard_ref':plan['native_metadata_cli']['pre_body_guard']['sha256']='0'*64
        elif change=='observer_ref':plan['native_metadata_cli']['observation_binding']['observer_source']['bytes']+=1
        elif change=='runtime_root':plan['accelerate_root']='/wrong-root'
        elif change=='guard_path':
            alias=tmp_path/'workspace-copy-guard.py'
            alias.write_bytes(Path(plan['native_metadata_cli']['pre_body_guard']['path']).read_bytes())
            ref=_pin(alias);plan['native_metadata_cli']['pre_body_guard']=ref
            plan['native_metadata_cli']['source_sha256'][str(alias)]=ref['sha256']
            value['source_sha256'][str(alias)]=ref['sha256']
        ref=_write(p,plan);value['argv'][6]=ref['sha256'];value['source_sha256'][str(p)]=ref['sha256']
    request_path=tmp_path/'request.json';ref=_write(request_path,value)
    with pytest.raises(RuntimeError):owner.request(request_path,ref['sha256'])


def test_exact_guard_nodes_execute_authenticated_source(guard,tmp_path):
    root=tmp_path/'project';root.mkdir();exports=tmp_path/'exports';exports.mkdir()
    source=root/'proof.py';source.write_text('answer = 47\n')
    plan={'accelerate_root':str(root),'datasets_root':str(root),'publication_root':str(root),
        'source_pins':[_pin(source)]}
    finder=guard.GuardedProjectImports(plan,exports)
    finder.path_finder=types.SimpleNamespace(find_spec=lambda *args:importlib.util.spec_from_file_location('ipfs_accelerate_py.proof',source))
    spec=finder.find_spec('ipfs_accelerate_py.proof');module=types.ModuleType(spec.name);module.__spec__=spec
    spec.loader.exec_module(module)
    assert module.answer==47
    assert finder.project_origins[spec.name]['execution_profile']=='compiled_exact_authenticated_source_bytes'
    source.write_text('raise RuntimeError("changed body must not run")\n')
    with pytest.raises(ValueError,match='input commitment differs'):spec.loader.exec_module(module)


@pytest.mark.parametrize('case',['uncatalogued','foreign_accelerate','provider'])
def test_guard_denies_before_project_body(guard,tmp_path,case):
    root=tmp_path/'project';root.mkdir();foreign=tmp_path/'foreign.py';foreign.write_text('raise AssertionError("body must not execute")\n')
    source=root/'proof.py';source.write_bytes(foreign.read_bytes())
    finder=guard.GuardedProjectImports({'accelerate_root':str(root),'datasets_root':str(root),
        'publication_root':str(root),'source_pins':[]},tmp_path)
    location=foreign if case=='foreign_accelerate' else source
    finder.path_finder=types.SimpleNamespace(find_spec=lambda *args:importlib.util.spec_from_file_location('ipfs_accelerate_py.proof',location))
    with pytest.raises(ImportError):finder.find_spec('openai' if case=='provider' else 'ipfs_accelerate_py.proof')
    assert finder.denials and finder.project_origins=={}


@pytest.mark.parametrize('case',['hash','bytes','symlink','hardlink'])
def test_original_guard_rejects_wrong_file_custody(guard,tmp_path,case):
    path=tmp_path/'source.py';path.write_text('value=1\n');ref=_pin(path)
    alias=tmp_path/'alias.py' if case in ('symlink','hardlink') else None
    try:
        if case=='hash':ref['sha256']='0'*64
        elif case=='bytes':ref['bytes']+=1
        elif case=='symlink':alias.symlink_to(path);ref['path']=str(alias)
        elif case=='hardlink':os.link(path,alias)
        with pytest.raises(ValueError):guard.read_pin(ref)
    finally:
        # The original resource finalizer requires ordinary single-link files.
        # Keep the real nofollow/nlink rejection, then remove our exact alias
        # even when that assertion fails; this grants no special-file custody.
        if alias is not None:alias.unlink(missing_ok=True)
        assert path.lstat().st_nlink==1


@pytest.fixture
def import_git_case(driver,tmp_path,monkeypatch):
    """Explicit frame/transport unit fixtures: never SDK, Git or health evidence."""
    import subprocess
    import time
    plan=_plan(driver,tmp_path);c=plan['native_metadata_cli']
    monkeypatch.setattr(driver,'DATASETS',tmp_path)
    attempt=tmp_path/'workspace/test-logs/fresh-git-admission-declaration'
    c.update(attempt_directory=str(attempt),temporary_directory=str(attempt/'temporary'),
        result_path=str(attempt/'exports/native-cli-driver-result.json'))
    capsule=c['import_head_capsule_Git']
    tokens={name:object() for name in ('capture','batch','query')}
    route=types.SimpleNamespace(_capture=None,_agent_control_plane_head_payloads=tokens['batch'],
        _agent_git_output=tokens['query'],_AGENT_CONTROL_PLANE_GIT_TIMEOUT_SECONDS=30,
        _AGENT_CONTROL_PLANE_RELATIVE_FILES=capsule['required_relative_files'],
        _AGENT_CONTROL_PLANE_REQUIRED_SQL_FILES=[name for name in capsule['required_relative_files'] if name.endswith('.sql')])
    daemon=types.SimpleNamespace(_capture_imported_control_plane_generation=tokens['capture'])
    frames={tokens['capture']:types.SimpleNamespace(f_locals={'source_root':driver.RUNTIME}),
        tokens['batch']:types.SimpleNamespace(f_locals={'root':driver.RUNTIME,'source_head':capsule['source_head'],
            'blobs':{row['relative_path']:row['object_id']for row in capsule['blobs']},'deadline':time.monotonic()+30}),
        tokens['query']:types.SimpleNamespace(f_locals={'root':driver.RUNTIME,'arguments':('rev-parse','--show-toplevel')}),
        subprocess.run:types.SimpleNamespace(f_locals={'timeout':30,'check':False,'input':None,'capture_output':False})}
    monkeypatch.setattr(driver,'module_owner',lambda name,relative,c:
        (route,capsule['route_source']) if name=='ipfs_accelerate_py.agent_implementation_route' else (daemon,{}))
    monkeypatch.setattr(driver,'frame_for',lambda module,fn:frames[fn])
    prefix=['git','-c','core.quotepath=false','-c','core.fsmonitor=false','-c','core.hooksPath=/dev/null']
    kwargs={'cwd':driver.RUNTIME,'env':None,'stdin':subprocess.PIPE,'stdout':subprocess.PIPE,
        'stderr':subprocess.DEVNULL,'close_fds':True,'bufsize':0}
    return {'plan':plan,'argv':prefix+['cat-file','--batch'],'kwargs':kwargs,'frames':frames,'tokens':tokens,
        'route':route,'daemon':daemon,'prefix':prefix}


def _refresh_import_git_environment_for_call(case,monkeypatch):
    """Match the real SDK's immediate-before-Popen environment construction.

    Pytest changes PYTEST_CURRENT_TEST between fixture setup and the call phase;
    the parent harness may restore PATH at that boundary too. No production
    environment check is relaxed, and negative mutations happen after refresh.
    """
    monkeypatch.setenv('PATH','/usr/bin:/bin')
    environment={name:value for name,value in os.environ.items() if not name.startswith('GIT_')}
    environment.update(GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL='/dev/null',GIT_TERMINAL_PROMPT='0',
        GIT_NO_REPLACE_OBJECTS='1',GIT_OPTIONAL_LOCKS='0',GIT_NO_LAZY_FETCH='1',LC_ALL='C',LANG='C')
    case['kwargs']['env']=environment


def test_exact_import_git_batch_constructor_shape_is_admitted_without_spawning(driver,import_git_case,monkeypatch):
    case=import_git_case
    _refresh_import_git_environment_for_call(case,monkeypatch)
    row=driver.authorize_native_control_popen(case['argv'],(),case['kwargs'],case['plan'])
    assert row['kind']=='immutable_source_capsule_read'
    assert row['original_frame']=='_agent_control_plane_head_payloads'
    assert row['start_new_session'] is False and row['sealed_launcher_fds']==[]
    capsule=case['plan']['native_metadata_cli']['import_head_capsule_Git']
    legacy='ipfs_accelerate_py/agent_supervisor/task_sources/sql/0002_tasks_index_repair.sql'
    assert any(item['relative_path']==legacy for item in capsule['blobs'])
    assert legacy not in capsule['required_relative_files']


def test_exact_import_git_metadata_constructor_shape_is_admitted_without_spawning(driver,import_git_case,monkeypatch):
    import subprocess
    case=import_git_case
    _refresh_import_git_environment_for_call(case,monkeypatch)
    kwargs=dict(case['kwargs'])
    kwargs.pop('close_fds');kwargs.pop('bufsize')
    kwargs.update(stdin=subprocess.DEVNULL,stderr=subprocess.PIPE)
    row=driver.authorize_native_control_popen(case['prefix']+['rev-parse','--show-toplevel'],(),kwargs,case['plan'])
    assert row['kind']=='immutable_source_capsule_read' and row['original_frame']=='_agent_git_output'


@pytest.mark.parametrize('mutation',['provider','tail','unknown_sql','environment','stdin','missing_sql',
    'close_fds','blob_scope','deadline'])
def test_import_git_admission_refuses_unprovided_process_shapes(driver,import_git_case,mutation,monkeypatch):
    import subprocess
    import time
    case=import_git_case
    _refresh_import_git_environment_for_call(case,monkeypatch)
    argv=list(case['argv']);kwargs=copy.deepcopy(case['kwargs'])
    if mutation=='provider':argv[0]=sys.executable
    elif mutation=='tail':argv[-1]='--filters'
    elif mutation=='unknown_sql':
        blob={'relative_path':'ipfs_accelerate_py/agent_supervisor/task_sources/sql/9999_unreviewed.sql','object_id':'9'*40,'bytes':1}
        capsule=case['plan']['native_metadata_cli']['import_head_capsule_Git']
        capsule['blobs'].append(blob);capsule['selected_file_count']+=1
        capsule['unique_blob_count']+=1;capsule['payload_bytes']+=1
    elif mutation=='environment':kwargs['env']['GIT_CONFIG_GLOBAL']='/foreign'
    elif mutation=='stdin':kwargs['stdin']=subprocess.DEVNULL
    elif mutation=='missing_sql':
        capsule=case['plan']['native_metadata_cli']['import_head_capsule_Git']
        removed=next(row for row in capsule['blobs'] if row['relative_path'].endswith('/0005_tasks_index_repair.sql'))
        capsule['blobs'].remove(removed);capsule['required_relative_files'].remove(removed['relative_path'])
        capsule['selected_file_count']-=1;capsule['unique_blob_count']-=1;capsule['payload_bytes']-=1
    elif mutation=='close_fds':kwargs['close_fds']=False
    elif mutation=='blob_scope':case['frames'][case['tokens']['batch']].f_locals['blobs']={}
    else:case['frames'][case['tokens']['batch']].f_locals['deadline']=time.monotonic()-1
    reasons={'provider':'does not dispatch','tail':'bounded Git query frame',
        'unknown_sql':'unselected original HEAD source family','environment':'noninteractive import Git environment',
        'stdin':'batch pipe/custody','missing_sql':'required source/SQL population',
        'close_fds':'batch pipe/custody','blob_scope':'immutable HEAD batch selection',
        'deadline':'batch deadline'}
    with pytest.raises(RuntimeError,match=reasons[mutation]):
        driver.authorize_native_control_popen(argv,(),kwargs,case['plan'])


def test_swallowed_normal_import_failure_cannot_be_reported_as_idle(driver,import_git_case):
    case=import_git_case
    # These types are explicit unit fixtures; absent actual capsule stops before
    # any verifier/native-entrypoint import, even if an outer import swallowed it.
    case['route'].AgentImplementationControlPlanePin=type('FixturePin',(),{})
    case['route'].AgentImplementationSealedControlPlane=type('FixtureLaunch',(),{})
    with pytest.raises(RuntimeError,match='genuine current eager'):
        driver.require_original_import_capability(case['daemon'],case['plan']['native_metadata_cli'])


def _resolution_case(guard,tmp_path,monkeypatch,body):
    """Explicit checkpoint fixture, never an installed native import guard."""
    root=tmp_path/'project';root.mkdir();exports=tmp_path/'exports';exports.mkdir()
    source=root/'probe.py';source.write_text(body)
    finder=guard.GuardedProjectImports({'accelerate_root':str(root),'datasets_root':str(root),
        'publication_root':str(root),'source_pins':[_pin(source)]},exports)
    finder.path_finder=types.SimpleNamespace(find_spec=lambda *args:
        importlib.util.spec_from_file_location('ipfs_accelerate_py.probe',source))
    # checkpoint() checks registration; no unremovable global audit hook is
    # installed inside a unit case. Only direct finder/loader calls run below.
    finder.registered=True
    monkeypatch.setattr(sys,'meta_path',[*sys.meta_path,finder])
    return finder,source


def _resolution_module(spec):
    module=types.ModuleType(spec.name);module.__spec__=spec;module.__file__=spec.origin
    return module


def test_spec_only_resolution_is_separate_until_actual_authenticated_execution(guard,tmp_path,monkeypatch):
    finder,source=_resolution_case(guard,tmp_path,monkeypatch,'answer = 47\n')
    spec=finder.find_spec('ipfs_accelerate_py.probe')
    before=json.loads(Path(finder.checkpoint('resolved-only')['path']).read_bytes())
    assert before['project_imports']==[] and len(before['project_resolutions'])==1
    row=before['project_resolutions'][0]
    assert row['execution_status']=='resolution_only' and 'execution_profile' not in row
    assert all(row[key]==_pin(source)[key] for key in ('path','bytes','sha256'))
    module=_resolution_module(spec);spec.loader.exec_module(module)
    after=json.loads(Path(finder.checkpoint('executed')['path']).read_bytes())
    assert module.answer==47 and after['project_resolutions']==[] and len(after['project_imports'])==1
    assert after['project_imports'][0]['execution_status']=='execution_completed'
    assert after['project_imports'][0]['execution_profile']=='compiled_exact_authenticated_source_bytes'


def test_started_execution_is_not_misreported_as_resolution_or_completed(guard,tmp_path,monkeypatch):
    finder,_=_resolution_case(guard,tmp_path,monkeypatch,'inside = checkpoint_inside()\nanswer = 47\n')
    spec=finder.find_spec('ipfs_accelerate_py.probe');module=_resolution_module(spec)
    module.checkpoint_inside=lambda:finder.checkpoint('inside-execution')
    spec.loader.exec_module(module)
    inside=json.loads(Path(module.inside['path']).read_bytes())
    assert inside['project_resolutions']==[] and len(inside['project_imports'])==1
    assert inside['project_imports'][0]['execution_status']=='execution_started'
    assert 'execution_profile' not in inside['project_imports'][0]


def test_failed_execution_remains_unqualified_import_provenance(guard,tmp_path,monkeypatch):
    finder,_=_resolution_case(guard,tmp_path,monkeypatch,'started = True\nraise RuntimeError("explicit fixture failure")\n')
    spec=finder.find_spec('ipfs_accelerate_py.probe');module=_resolution_module(spec)
    with pytest.raises(RuntimeError,match='explicit fixture failure'):spec.loader.exec_module(module)
    value=json.loads(Path(finder.checkpoint('failed-execution')['path']).read_bytes())
    assert module.started is True and value['project_resolutions']==[] and len(value['project_imports'])==1
    assert value['project_imports'][0]['execution_status']=='execution_failed'
    assert 'execution_profile' not in value['project_imports'][0]


def test_reresolving_same_pin_preserves_actual_completed_execution(guard,tmp_path,monkeypatch):
    finder,_=_resolution_case(guard,tmp_path,monkeypatch,'events.append("executed")\n')
    spec=finder.find_spec('ipfs_accelerate_py.probe');module=_resolution_module(spec);module.events=[]
    spec.loader.exec_module(module)
    finder.find_spec('ipfs_accelerate_py.probe')
    value=json.loads(Path(finder.checkpoint('reresolved')['path']).read_bytes())
    assert module.events==['executed'] and value['project_resolutions']==[]
    assert value['project_imports'][0]['execution_status']=='execution_completed'
    assert value['project_imports'][0]['execution_profile']=='compiled_exact_authenticated_source_bytes'


def test_reresolving_different_authenticated_pin_is_refused(guard,tmp_path,monkeypatch):
    finder,source=_resolution_case(guard,tmp_path,monkeypatch,'answer = 47\n')
    spec=finder.find_spec('ipfs_accelerate_py.probe');module=_resolution_module(spec);spec.loader.exec_module(module)
    original=dict(finder.project_origins[spec.name])
    source.write_text('answer = 99\n');finder.pins[str(source)]=_pin(source)
    with pytest.raises(ImportError,match='project_resolution_changed'):
        finder.find_spec('ipfs_accelerate_py.probe')
    assert finder.project_origins[spec.name]==original and module.answer==47


@pytest.mark.parametrize('ambient',[False,True],ids=['present-before-exec','ambient-body-outside-guard'])
def test_resolution_only_present_module_cannot_escape_execution_provenance(guard,tmp_path,monkeypatch,ambient):
    finder,source=_resolution_case(guard,tmp_path,monkeypatch,'outside_guard = True\n')
    spec=finder.find_spec('ipfs_accelerate_py.probe');module=_resolution_module(spec)
    # Both a premature module registration and actual ambient execution must
    # remain unqualified. This explicit private fixture is not native health.
    with monkeypatch.context() as context:
        context.setitem(sys.modules,spec.name,module)
        if ambient:
            exec(compile(source.read_bytes(),str(source),'exec'),module.__dict__)
            assert module.outside_guard is True
        with pytest.raises(ValueError,match='loaded project module lacks authenticated execution'):
            finder.checkpoint('present-resolution-only')
    assert finder.project_origins[spec.name]['execution_status']=='resolution_only'
    assert 'execution_profile' not in finder.project_origins[spec.name]
    assert not (finder.directory/'project-import-origins-present-resolution-only.json').exists()


def _reuse_cache_fixture(c,tmp_path):
    root=tmp_path/'retained-cache'/('c'*64);root.mkdir(parents=True)
    runner='ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py'
    accepted=dict(schema='ipfs_accelerate_py.agent_supervisor.accepted-control-plane@2',runner_path=str(root/runner),
        runner_sha256='sha256:'+'a'*64,capsule_root=str(root),capsule_id='sha256:'+'c'*64,
        source_head=c['import_head_capsule_Git']['source_head'],source_tree='2'*40,archive_sha256='sha256:'+'b'*64)
    manifest=dict(schema='ipfs_accelerate_py.agent_supervisor.materialized-control-plane@1',source_head=accepted['source_head'],
        source_tree=accepted['source_tree'],files={runner:accepted['runner_sha256']},capsule_id=accepted['capsule_id'])
    commitment=dict(schema='legal-ir-scientific-source-commitment/v1',source_root=c['runtime_root'],
        source_commit=accepted['source_head'],source_tree=accepted['source_tree'],runtime_activation_or_task_authority=False)
    m=_write(root/'.agent-control-plane-manifest.json',manifest);z=_write(tmp_path/'source-generation.json',commitment)
    data=dict(schema='agent-supervisor/scientific-source-capsule-reuse@1',accepted_pin=accepted,manifest_ref=m,source_commitment_ref=z)
    path=tmp_path/'source-capsule-ref.json'
    def seal():
        ref=_write(path,data);c['schema']='native-embedded-scoped-observation-cli/v3';c['source_capsule_ref']=ref
        for pin in (ref,m,z):c['source_sha256'][pin['path']]=pin['sha256']
        return ref
    seal();return data,seal


@pytest.mark.parametrize('target',['driver','owner'])
def test_reuse_metadata_is_closed_source_input_without_native_or_cache_writes(driver,owner,tmp_path,monkeypatch,target):
    plan=_plan(driver,tmp_path);c=plan['native_metadata_cli'];data,_=_reuse_cache_fixture(c,tmp_path)
    module=driver if target=='driver' else owner;monkeypatch.setattr(module,'REUSE_ROOTS',(tmp_path,))
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert module.capsule_reuse_binding(c)==data
    assert {str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}==before


@pytest.mark.parametrize('mutation',['extra','head','tree','commitment_root','source_pin','inside_attempt','manifest_path'])
def test_reuse_metadata_rejects_cache_scope_and_generation_drift(driver,tmp_path,monkeypatch,mutation):
    c=_plan(driver,tmp_path)['native_metadata_cli'];data,seal=_reuse_cache_fixture(c,tmp_path)
    monkeypatch.setattr(driver,'REUSE_ROOTS',(tmp_path,))
    if mutation=='extra':data['write_on_miss']=True
    elif mutation=='head':data['accepted_pin']['source_head']='3'*40
    elif mutation=='tree':data['accepted_pin']['source_tree']='3'*40
    elif mutation=='commitment_root':
        p=Path(data['source_commitment_ref']['path']);x=json.loads(p.read_bytes());x['source_root']='/arbitrary/root'
        data['source_commitment_ref']=_write(p,x)
    elif mutation=='inside_attempt':data['accepted_pin']['capsule_root']=str(Path(c['attempt_directory'])/'cache'/('c'*64))
    elif mutation=='manifest_path':data['manifest_ref']=_write(tmp_path/'other-manifest.json',json.loads(Path(data['manifest_ref']['path']).read_bytes()))
    seal()
    if mutation=='source_pin':c['source_sha256'].pop(data['manifest_ref']['path'])
    with pytest.raises(RuntimeError):driver.capsule_reuse_binding(c)


def test_original_owner_cli3_transport_keeps_exact_grants_and_empty_attempt(owner,driver,tmp_path,monkeypatch):
    value,plan,path=_request(owner,driver,tmp_path,monkeypatch);c=plan['native_metadata_cli']
    _reuse_cache_fixture(c,tmp_path);monkeypatch.setattr(owner,'REUSE_ROOTS',(tmp_path,))
    value['source_sha256'].update(c['source_sha256']);ref=_write(path,plan)
    value['argv'][6]=ref['sha256'];value['source_sha256'][str(path)]=ref['sha256']
    req=tmp_path/'v3-request.json';ref=_write(req,value)
    actual,_=owner.request(req,ref['sha256'])
    assert actual==value and {k:actual[k] for k in ('cpu_slots','memory_mb','child_process_slots','storage_bytes','timeout_seconds')}==dict(cpu_slots=1,memory_mb=1536,child_process_slots=8,storage_bytes=384*1024**2,timeout_seconds=300)
    assert list(Path(value['attempt_directory']).iterdir())==[]


@pytest.mark.parametrize('files',[276,277])
def test_reuse_parent_file_bound_is_enforced_independently_of_byte_usage(owner,tmp_path,files):
    for n in range(files):(tmp_path/str(n)).write_bytes(b'')
    if files==276:assert owner.reuse_attempt_file_bound(tmp_path)==276
    else:
        with pytest.raises(RuntimeError,match='total attempt file bound'):owner.reuse_attempt_file_bound(tmp_path)


@pytest.mark.parametrize('ordinal',[32,33])
def test_reuse_process_constructor_cap_precedes_any_subprocess(owner,ordinal):
    if ordinal==32:owner.reuse_process_constructor_bound(ordinal)
    else:
        with pytest.raises(RuntimeError,match='constructor bound'):owner.reuse_process_constructor_bound(ordinal)


# Policy successor controls extend the existing owner. The original tests above
# remain byte-exact; these constructors and inert seams grant no native custody.
def _policy_request(owner,driver,tmp_path,monkeypatch,policy_name):
    value,plan,p=_request(owner,driver,tmp_path,monkeypatch)
    assert value['expected_scheduler_policy_sha256']==owner.POLICY
    value['expected_scheduler_policy_sha256']=getattr(owner,policy_name)
    if policy_name=='POLICY':return value,plan,p
    old_root=str(driver.RUNTIME);new_root=str(owner.QUALIFIED_F07_ACTOR.parents[3])
    owner_path=str(Path(owner.__file__))
    def relocated(item):
        if isinstance(item,dict):return {relocated(k):relocated(v) for k,v in item.items()}
        if isinstance(item,list):return [relocated(v) for v in item]
        if isinstance(item,str) and item!=owner_path and item.startswith(old_root+'/'):
            return new_root+item[len(old_root):]
        if item==old_root:return new_root
        return item
    value=relocated(value);plan=relocated(plan)
    ref=_write(p,plan)
    value['argv'][6]=ref['sha256'];value['source_sha256'][str(p)]=ref['sha256']
    return value,plan,p


@pytest.mark.parametrize('policy_name',['POLICY','QUALIFIED_F07_POLICY'])
def test_known_policy_request_preserves_legacy_shape_without_effects(owner,driver,tmp_path,monkeypatch,policy_name):
    value,_,_=_policy_request(owner,driver,tmp_path,monkeypatch,policy_name)
    p=tmp_path/'policy-request.json';ref=_write(p,value)
    reads=[];actual=owner.regular
    def recorded(path,*args):
        reads.append(str(path));return actual(path,*args)
    def forbidden(*args,**kwargs):
        raise AssertionError('request parsing must not enter a resource owner')
    monkeypatch.setattr(owner,'regular',recorded)
    monkeypatch.setattr(owner,'_open_original_scheduler',forbidden)
    monkeypatch.setattr(owner,'load_source_lease',forbidden)
    parsed,raw=owner.request(p,ref['sha256'])
    assert parsed==value and raw==p.read_bytes()
    assert parsed['schema']=='legal-ir-original-native-control-request/v1'
    assert set(parsed)==owner.FIELDS
    assert str(owner.DATABASE) not in reads
    assert list(Path(value['attempt_directory']).iterdir())==[]


@pytest.mark.parametrize('selected',[None,False,7,{},[], 'f07', '0'*64,
    '4662bd8fe6caf6f2c048e3d7cff7abb4d80e5c875b21e02e204b1b89371b8286'.upper(),
    'f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661'.upper()])
def test_unknown_policy_refuses_before_plan_database_or_admission(owner,driver,tmp_path,monkeypatch,selected):
    value,_,_=_request(owner,driver,tmp_path,monkeypatch)
    value['expected_scheduler_policy_sha256']=selected
    p=tmp_path/'unknown-policy.json';ref=_write(p,value)
    reads=[];actual=owner.regular
    def recorded(path,*args):
        reads.append(str(path));return actual(path,*args)
    monkeypatch.setattr(owner,'regular',recorded)
    with pytest.raises(RuntimeError,match='explicit qualified scheduler policy'):
        owner.request(p,ref['sha256'])
    assert reads==[str(p)]
    assert list(Path(value['attempt_directory']).iterdir())==[]


@pytest.mark.parametrize('change',['missing_expected','extra_profile'])
def test_policy_selection_never_adds_a_field_or_missing_field_default(owner,driver,tmp_path,monkeypatch,change):
    value,_,_=_request(owner,driver,tmp_path,monkeypatch)
    if change=='missing_expected':value.pop('expected_scheduler_policy_sha256')
    else:value['scheduler_policy_profile']='current'
    p=tmp_path/'closed-policy.json';ref=_write(p,value)
    with pytest.raises(RuntimeError,match='closed control request'):
        owner.request(p,ref['sha256'])


@pytest.mark.parametrize('policy_name',['POLICY','QUALIFIED_F07_POLICY'])
@pytest.mark.parametrize('field,bad',[('cpu_slots',2),('memory_mb',1537),
    ('child_process_slots',9),('storage_bytes',384*1024**2+1),('timeout_seconds',301)])
def test_policy_choice_does_not_widen_any_resource_bound(owner,driver,tmp_path,monkeypatch,policy_name,field,bad):
    value,_,_=_policy_request(owner,driver,tmp_path,monkeypatch,policy_name)
    value[field]=bad
    p=tmp_path/'policy-envelope.json';ref=_write(p,value)
    with pytest.raises(RuntimeError,match='reviewed finite control envelope'):
        owner.request(p,ref['sha256'])


@pytest.mark.parametrize('selected_name',['POLICY','QUALIFIED_F07_POLICY'])
@pytest.mark.parametrize('actual_name',['POLICY','QUALIFIED_F07_POLICY'])
def test_original_scheduler_receives_selected_digest_and_crossing_refuses_before_acquire(owner,selected_name,actual_name):
    selected=getattr(owner,selected_name);actual=getattr(owner,actual_name)
    calls=[];sentinel=object()
    def original_opener(state,expected):
        calls.append((state,expected))
        if expected!=actual:raise RuntimeError('original scheduler policy mismatch')
        return sentinel
    value={'expected_scheduler_policy_sha256':selected}
    if selected==actual:
        assert owner._open_original_scheduler(value,original_opener) is sentinel
    else:
        with pytest.raises(RuntimeError,match='original scheduler policy mismatch'):
            owner._open_original_scheduler(value,original_opener)
    assert calls==[(owner.STATE,selected)]


def test_unknown_digest_never_calls_original_scheduler_opener(owner):
    calls=[]
    with pytest.raises(RuntimeError,match='explicit qualified scheduler policy'):
        owner._open_original_scheduler({'expected_scheduler_policy_sha256':'0'*64},
            lambda *args:calls.append(args))
    assert calls==[]


@pytest.mark.parametrize('selected_name',['POLICY','QUALIFIED_F07_POLICY'])
@pytest.mark.parametrize('actual_name',['POLICY','QUALIFIED_F07_POLICY'])
def test_private_capability_requires_matching_admitted_record_policy(owner,selected_name,actual_name):
    selected=getattr(owner,selected_name);actual=getattr(owner,actual_name)
    value={'expected_scheduler_policy_sha256':selected}
    record={'scheduler_policy_sha256':actual}
    if selected==actual:
        assert owner._parent_capacity_scheduler_policy(value,record)==selected
    else:
        with pytest.raises(RuntimeError,match='admitted parent scheduler policy differs'):
            owner._parent_capacity_scheduler_policy(value,record)


@pytest.mark.parametrize('policy_name',['POLICY','QUALIFIED_F07_POLICY'])
def test_private_capability_does_not_invent_a_missing_record_policy(owner,policy_name):
    with pytest.raises(RuntimeError,match='admitted parent scheduler policy differs'):
        owner._parent_capacity_scheduler_policy(
            {'expected_scheduler_policy_sha256':getattr(owner,policy_name)},{})


@pytest.mark.parametrize('policy_name',['POLICY','QUALIFIED_F07_POLICY'])
@pytest.mark.parametrize('admitted_name',['POLICY','QUALIFIED_F07_POLICY'])
def test_parent_seal_carries_selected_policy_with_inert_fd_operations(owner,tmp_path,monkeypatch,policy_name,admitted_name):
    from dataclasses import dataclass
    @dataclass
    class Token:
        lease_id:str
        lease_key:str
        state_path:str
    policy=getattr(owner,policy_name);pid=os.getpid();child_pid=123;child_birth='456'
    record={'status':'active','owner_pid':pid,'native_lease_id':'fixture-native',
        'child':{'pid':child_pid,'birth':child_birth},'parent_lease_id':'fixture-phase',
        'attempt_directory':{'path':str(tmp_path),'device':0,'inode':0},
        'scheduler_policy_sha256':getattr(owner,admitted_name)}
    native=types.SimpleNamespace(owner_pid=pid,released=False,cancelled=False,
        token=Token('fixture-native','fixture-key',str(owner.STATE)))
    fake_owner=types.SimpleNamespace(native_lease=native,
        to_dict=lambda:{'status':'active','reservation_id':'fixture-disk','record':record})
    writes=[];seals=owner.fcntl.F_SEAL_WRITE|owner.fcntl.F_SEAL_GROW|owner.fcntl.F_SEAL_SHRINK|owner.fcntl.F_SEAL_SEAL
    fake_os=types.SimpleNamespace(getpid=lambda:pid,geteuid=os.geteuid,
        SEEK_END=os.SEEK_END,SEEK_SET=os.SEEK_SET,lseek=lambda *args:0,
        write=lambda fd,data:(writes.append(bytes(data)) or len(data)))
    fake_fcntl=types.SimpleNamespace(**{name:getattr(owner.fcntl,name) for name in
        ('F_SEAL_WRITE','F_SEAL_GROW','F_SEAL_SHRINK','F_SEAL_SEAL','F_ADD_SEALS','F_GET_SEALS')},
        fcntl=lambda *args:seals)
    monkeypatch.setattr(owner,'os',fake_os)
    monkeypatch.setattr(owner,'fcntl',fake_fcntl)
    monkeypatch.setattr(owner,'time',types.SimpleNamespace(monotonic=lambda:1.0))
    monkeypatch.setattr(owner,'_parent_capacity_process_identity',lambda pid:{'pid':pid,'birth':'fixture'})
    monkeypatch.setattr(owner,'pin',lambda p:{'path':str(p),'bytes':1,'sha256':'0'*64})
    value={'expected_scheduler_policy_sha256':policy,'cpu_slots':1,'memory_mb':1536,
        'child_process_slots':8,'storage_bytes':384*1024**2}
    kwargs=dict(owner=fake_owner,value=value,request_path=tmp_path/'request.json',
        execution_request={'path':'fixture'},work_deadline=2.0,native_deadline=3.0,
        child_pid=child_pid,child_birth=child_birth)
    if policy!=getattr(owner,admitted_name):
        with pytest.raises(RuntimeError,match='admitted parent scheduler policy differs'):
            owner._seal_parent_capacity_fd(99,**kwargs)
        assert writes==[]
        return
    receipt=owner._seal_parent_capacity_fd(99,**kwargs)
    payload=json.loads(b''.join(writes))
    assert payload['scheduler_policy_sha256']==policy
    assert payload['schema']=='native-metadata-sealed-parent-capacity/v1'
    assert payload['authority']==owner.PARENT_CAPACITY_AUTHORITY
    assert receipt['sealed'] is True and receipt['original_child_bound'] is True


def test_invalid_policy_main_refuses_before_any_resource_or_child_effect(owner,driver,tmp_path,monkeypatch):
    value,_,_=_request(owner,driver,tmp_path,monkeypatch)
    value['expected_scheduler_policy_sha256']='0'*64
    p=tmp_path/'main-unknown-policy.json';ref=_write(p,value)
    calls=[]
    def forbidden(*args,**kwargs):
        calls.append('effect');raise AssertionError('unexpected resource or child effect')
    monkeypatch.setattr(owner,'load_source_lease',forbidden)
    monkeypatch.setattr(owner,'pin',forbidden)
    monkeypatch.setattr(owner,'_open_original_scheduler',forbidden)
    monkeypatch.setattr(owner,'_new_parent_capacity_fd',forbidden)
    monkeypatch.setattr(owner,'subprocess',types.SimpleNamespace(Popen=forbidden))
    monkeypatch.setattr(sys,'argv',[owner.__file__,'--request',str(p),'--expected-request-sha256',ref['sha256']])
    with pytest.raises(RuntimeError,match='explicit qualified scheduler policy'):
        owner.main()
    assert calls==[]
    assert list(Path(value['attempt_directory']).iterdir())==[]


def test_legacy_backend_selection_keeps_original_sibling_without_new_pins(owner):
    assert owner._observation_actor({'expected_scheduler_policy_sha256':owner.POLICY})==owner.ACTOR_PATH


@pytest.mark.parametrize('mutation',['wrong_path','missing_pin','wrong_pin'])
def test_qualified_backend_rejects_unknown_or_copied_actor_before_file_read(owner,mutation,monkeypatch):
    actor=str(owner.QUALIFIED_F07_ACTOR)
    value={'expected_scheduler_policy_sha256':owner.QUALIFIED_F07_POLICY,
        'argv':[sys.executable,'-B',actor],'source_sha256':{actor:owner.QUALIFIED_F07_ACTOR_SHA}}
    if mutation=='wrong_path':
        copied='/tmp/copied-native_embedded_observation_driver.py'
        value['argv'][2]=copied;value['source_sha256'][copied]=owner.QUALIFIED_F07_ACTOR_SHA
    elif mutation=='missing_pin':value['source_sha256'].pop(actor)
    else:value['source_sha256'][actor]='0'*64
    calls=[]
    monkeypatch.setattr(owner,'pin',lambda *args:calls.append(args))
    with pytest.raises(RuntimeError,match='explicit qualified f07 observation backend'):
        owner._observation_actor(value)
    assert calls==[]


def test_qualified_backend_rejects_changed_original_bytes(owner,monkeypatch):
    actor=str(owner.QUALIFIED_F07_ACTOR)
    value={'expected_scheduler_policy_sha256':owner.QUALIFIED_F07_POLICY,
        'argv':[sys.executable,'-B',actor],'source_sha256':{actor:owner.QUALIFIED_F07_ACTOR_SHA}}
    monkeypatch.setattr(owner,'pin',lambda p:{'path':str(p),'bytes':28162,'sha256':'0'*64})
    with pytest.raises(RuntimeError,match='qualified f07 observation backend changed'):
        owner._observation_actor(value)


@pytest.mark.parametrize('mutation',['accelerate_root','runtime_root','observer','guard'])
def test_relocated_owner_cannot_rebind_retained_backend_graph(owner,driver,tmp_path,monkeypatch,mutation):
    value,plan,p=_policy_request(owner,driver,tmp_path,monkeypatch,'QUALIFIED_F07_POLICY')
    c=plan['native_metadata_cli']
    if mutation=='accelerate_root':plan['accelerate_root']=str(driver.RUNTIME)
    elif mutation=='runtime_root':c['runtime_root']=str(driver.RUNTIME)
    elif mutation=='observer':c['observation_binding']['observer_source']['path']=str(tmp_path/'other-helper.py')
    else:c['pre_body_guard']['path']=str(tmp_path/'other-guard.py')
    ref=_write(p,plan);value['argv'][6]=ref['sha256'];value['source_sha256'][str(p)]=ref['sha256']
    q=tmp_path/'rebound-backend.json';ref=_write(q,value)
    with pytest.raises(RuntimeError):owner.request(q,ref['sha256'])
    assert list(Path(value['attempt_directory']).iterdir())==[]


@pytest.mark.parametrize('selected_name',['POLICY','QUALIFIED_F07_POLICY'])
@pytest.mark.parametrize('actual_name',['POLICY','QUALIFIED_F07_POLICY'])
def test_parented_constructor_receives_same_policy_and_unchanged_bounds(owner,tmp_path,selected_name,actual_name):
    selected=getattr(owner,selected_name);actual=getattr(owner,actual_name)
    calls=[];sentinel=object();parent=object()
    def original_constructor(*args,**kwargs):
        calls.append((args,kwargs))
        if kwargs['expected_scheduler_policy_sha256']!=actual:
            raise RuntimeError('original parented policy mismatch')
        return sentinel
    value={'expected_scheduler_policy_sha256':selected,'storage_bytes':384*1024**2}
    if selected==actual:
        assert owner._new_original_parented_reservation(value,original_constructor,
            attempt=tmp_path,parent_lease=parent,timeout_seconds=17.5) is sentinel
    else:
        with pytest.raises(RuntimeError,match='original parented policy mismatch'):
            owner._new_original_parented_reservation(value,original_constructor,
                attempt=tmp_path,parent_lease=parent,timeout_seconds=17.5)
    assert calls==[((owner.LEDGER,),dict(roots=owner.ROOTS,attempt_directory=tmp_path,
        parent_lease=parent,storage_bytes=384*1024**2,memory_mb=1536,cpu_slots=1,
        child_process_slots=8,scheduler_state_path=owner.STATE,
        expected_scheduler_policy_sha256=selected,timeout_seconds=17.5,
        ledger_lock_timeout_seconds=None))]


def test_unknown_policy_never_constructs_parented_reservation(owner,tmp_path):
    calls=[]
    with pytest.raises(RuntimeError,match='explicit qualified scheduler policy'):
        owner._new_original_parented_reservation(
            {'expected_scheduler_policy_sha256':'0'*64,'storage_bytes':384*1024**2},
            lambda *args,**kwargs:calls.append((args,kwargs)),attempt=tmp_path,
            parent_lease=object(),timeout_seconds=17.5)
    assert calls==[]
