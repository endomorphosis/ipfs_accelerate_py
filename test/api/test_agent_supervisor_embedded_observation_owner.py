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


def _plan(driver,tmp_path):
    attempt=tmp_path/'attempt';attempt.mkdir(mode=0o700)
    scope=['baguqeeraa0001','baguqeeraa0002']
    root=driver.RUNTIME
    helper=root/'ipfs_accelerate_py/agent_supervisor/todo_daemon/native_owner_bootstrap.py'
    guard=root/REL/'project_import_source_guard.py'
    pins={str(p):_pin(p)['sha256'] for p in (helper,guard,Path(driver.__file__))}
    prior={'path':str(driver.DATABASE),'bytes':1,'sha256':'0'*64}
    binding={'schema':'native-embedded-daemon-observation-binding@1','database_preimage':prior,
        'task_scope':scope,'plan_root_cid':'baguqeeraa-plan-root','repository_tree_id':'tree-v1',
        'observer_source':_pin(helper),'source_schema':'ipfs_accelerate_py/agent-supervisor/database-task-source@1'}
    return {'schema':'native-embedded-scoped-observation-plan/v1',
        'operation':'native_embedded_scoped_observation_v1','accelerate_root':str(root),'task_scope':scope,
        'native_metadata_cli':{'schema':'native-embedded-scoped-observation-cli/v1','runtime_root':str(root),
            'cwd':str(driver.WORKSPACE),'attempt_directory':str(attempt),'temporary_directory':str(attempt/'temporary'),
            'project_import_plan':{'accelerate_root':str(root),'datasets_root':str(driver.DATASETS),
                'publication_root':str(driver.DATASETS),'source_pins':[_pin(p) for p in (helper,guard)]},
            'pre_body_guard':_pin(guard),'result_path':str(attempt/'exports/native-cli-driver-result.json'),
            'source_sha256':pins,'observation_binding':binding}}


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
