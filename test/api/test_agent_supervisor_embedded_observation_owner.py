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
