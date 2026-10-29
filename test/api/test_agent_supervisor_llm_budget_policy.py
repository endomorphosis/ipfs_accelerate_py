"""Opt-in freeze tests with actual owned adapter children and no provider/network."""
from __future__ import annotations
import ast
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2]
PACKAGE = 'ipfs_accelerate_py.agent_supervisor.todo_daemon'
sys.path.insert(0, '/home/barberb/lift_coding/external/ipfs_accelerate')
# Load real package prerequisites, then the reviewed private files themselves.
import ipfs_accelerate_py.agent_supervisor.todo_daemon.llm


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


guard=load(PACKAGE+'.budget_policy',SOURCE/'ipfs_accelerate_py/agent_supervisor/todo_daemon/budget_policy.py')
adapter=load(PACKAGE+'._private_budget_llm',SOURCE/'ipfs_accelerate_py/agent_supervisor/todo_daemon/llm.py')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True)+'\n');path.chmod(0o600)


@pytest.fixture
def campaign(tmp_path,monkeypatch):
    root=tmp_path/'campaign';root.mkdir()
    goals=root/'implementation_plan/goals.json';write(goals,{'budgets':{'router_calls_total':24}})
    entries=[]
    for i,calls in enumerate([22,1,1,1]):
        ledger=root/f'budget/ledger-{i}.json'
        schema=None if i==0 else 'jevops.lean-refactor.scientific-review-ledger.v1'
        value={'calls':calls,'contracts':{'retained-bootstrap':calls}} if i==0 else {
            'schema':schema,'calls':calls,'max_calls':1,'status':'RESPONSE_RETAINED'}
        write(ledger,value);entries.append({'path':str(ledger.relative_to(root)),
            'sha256':sha(ledger),'schema':schema,'calls':calls})
    policy={'schema':guard.SCHEMA,'env_prefix':'JEVOPS_REFACTOR_LLM','campaign_root':str(root),
            'new_calls_allowed':False,'original_goals':{'path':str(goals.relative_to(root)),
            'sha256':sha(goals),'router_calls_total':24},'ledgers':entries,
            'aggregate_reserved':25,'reason':'operator_provider_freeze'}
    path=tmp_path/'freeze.json';write(path,policy)
    fake=tmp_path/'owned-fixture-router';package=fake/'ipfs_accelerate_py';package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    marker=tmp_path/'router-child-was-executed'
    (package/'llm_router.py').write_text('from pathlib import Path\n'
      'def generate_text(prompt,**kwargs):\n    Path('+repr(str(marker))+').write_text("owned fixture; no provider")\n    return "FIXTURE_OK"\n'
      'def get_last_generation_trace():\n    return {"effective_provider_name":"grok_cli"}\n')
    monkeypatch.setattr(adapter,'_canonical_accelerator_source_root',lambda:fake)
    monkeypatch.setattr(adapter,'apply_efficient_llm_route',lambda config:config)
    monkeypatch.setenv('JEVOPS_REFACTOR_LLM_BUDGET_POLICY',str(path))
    config=adapter.LlmRouterInvocation(repo_root=root,provider='grok_cli',model_name='grok-4.7',
      env_prefix='JEVOPS_REFACTOR_LLM',python_executable=sys.executable,timeout_seconds=5,
      timeout_grace_seconds=1,required_effective_providers=('grok_cli',),
      allow_local_fallback=False,allow_cross_provider_fallback=False)
    return SimpleNamespace(root=root,path=path,policy=policy,config=config,marker=marker)


def denied(f):
    with pytest.raises(guard.BudgetPolicyDenied,match='no router child|No router child'):
        adapter.call_llm_router_with_receipt('private fixture; never a model request',f.config)
    assert not f.marker.exists()


def test_actual_exhausted_policy_denies_before_any_tempfile_or_router_child(campaign,monkeypatch):
    f=campaign
    before={p:p.read_bytes() for p in f.root.rglob('*.json')}
    audit=guard.inspect_budget_freeze(f.path,env_prefix=f.config.env_prefix,repo_root=f.root)
    assert audit['aggregate_reserved']==25 and audit['original_cap']==24
    monkeypatch.setattr(adapter.tempfile,'NamedTemporaryFile',lambda *a,**k:pytest.fail('no provider artifact may be created'))
    denied(f)
    assert all(p.read_bytes()==raw for p,raw in before.items())


def test_unrelated_prefix_uses_actual_owned_child_despite_invalid_campaign_policy(campaign):
    f=campaign;f.path.write_text('invalid policy body')
    config=replace(f.config,env_prefix='UNRELATED_LLM')
    text,result=adapter.call_llm_router_with_receipt('owned fixture',config)
    assert text=='FIXTURE_OK' and f.marker.read_text()=='owned fixture; no provider'
    assert result.exit_code==0 and result.effective_provider=='grok_cli'


def test_unconfigured_existing_prefix_retains_real_child_transport(campaign,monkeypatch):
    f=campaign;monkeypatch.delenv('JEVOPS_REFACTOR_LLM_BUDGET_POLICY')
    text,result=adapter.call_llm_router_with_receipt('owned fixture',f.config)
    assert text=='FIXTURE_OK' and result.exit_code==0 and f.marker.exists()


def test_freeze_never_grants_a_call_even_if_audited_count_is_below_cap(campaign):
    f=campaign;entry=f.policy['ledgers'][0];path=f.root/entry['path']
    value=json.loads(path.read_text());value['calls']=value['contracts']['retained-bootstrap']=20;write(path,value)
    entry.update(calls=20,sha256=sha(path));f.policy['aggregate_reserved']=23;write(f.path,f.policy)
    assert guard.inspect_budget_freeze(f.path,env_prefix=f.config.env_prefix,repo_root=f.root)['aggregate_reserved']==23
    denied(f)


@pytest.mark.parametrize('fault',['missing','invalid_json','empty','symlink','hardlink','writable','duplicate_json','extra_field','allow_true','wrong_prefix','outside_scope'])
def test_missing_malformed_or_unsafe_policy_fails_closed(campaign,monkeypatch,tmp_path,fault):
    f=campaign
    if fault=='missing':f.path.unlink()
    elif fault=='invalid_json':f.path.write_text('{')
    elif fault=='empty':monkeypatch.setenv('JEVOPS_REFACTOR_LLM_BUDGET_POLICY','')
    elif fault=='symlink':
        copy=f.path.with_name('actual.json');f.path.rename(copy);f.path.symlink_to(copy)
    elif fault=='hardlink':os.link(f.path,f.path.with_name('same-inode.json'))
    elif fault=='writable':f.path.chmod(0o666)
    elif fault=='duplicate_json':f.path.write_text('{"schema":"x","schema":"y"}')
    elif fault=='extra_field':f.policy['allow_once']=True;write(f.path,f.policy)
    elif fault=='allow_true':f.policy['new_calls_allowed']=True;write(f.path,f.policy)
    elif fault=='wrong_prefix':f.policy['env_prefix']='OTHER';write(f.path,f.policy)
    else:
        outside=tmp_path/'outside';outside.mkdir();f.config=replace(f.config,repo_root=outside)
    denied(f)


@pytest.mark.parametrize('fault',['missing','symlink','raw_change','count_boolean','count_mismatch','schema_mismatch','duplicate_path','escape','goals_change','cap_change','aggregate_change','fewer_ledgers'])
def test_all_four_immutable_ledger_anchors_and_original_cap_are_required(campaign,fault):
    f=campaign;entry=f.policy['ledgers'][1];path=f.root/entry['path']
    if fault=='missing':path.unlink()
    elif fault=='symlink':
        real=path.with_name('actual.json');path.rename(real);path.symlink_to(real)
    elif fault=='raw_change':path.write_text(path.read_text()+' ')
    elif fault=='count_boolean':entry['calls']=True;write(f.path,f.policy)
    elif fault=='count_mismatch':entry['calls']=0;write(f.path,f.policy)
    elif fault=='schema_mismatch':entry['schema']='wrong';write(f.path,f.policy)
    elif fault=='duplicate_path':f.policy['ledgers'][2]=entry;write(f.path,f.policy)
    elif fault=='escape':entry['path']='../elsewhere.json';write(f.path,f.policy)
    elif fault=='goals_change':(f.root/f.policy['original_goals']['path']).write_text('{}')
    elif fault=='cap_change':f.policy['original_goals']['router_calls_total']=100;write(f.path,f.policy)
    elif fault=='aggregate_change':f.policy['aggregate_reserved']=24;write(f.path,f.policy)
    else:f.policy['ledgers'].pop();write(f.path,f.policy)
    denied(f)


def test_freeze_is_before_provider_adaptation_or_model_switch(campaign,monkeypatch):
    f=campaign;f.config=replace(f.config,provider='llama_cpp')
    monkeypatch.setattr(adapter,'apply_efficient_llm_route',lambda *_:pytest.fail('no route adaptation before freeze'))
    denied(f)


def test_denial_clears_stale_previous_success_receipt(campaign,monkeypatch):
    f=campaign;monkeypatch.delenv('JEVOPS_REFACTOR_LLM_BUDGET_POLICY')
    adapter.call_llm_router_with_receipt('owned fixture',f.config)
    assert adapter.last_llm_child_result() is not None
    f.marker.unlink();monkeypatch.setenv('JEVOPS_REFACTOR_LLM_BUDGET_POLICY',str(f.path));denied(f)
    assert adapter.last_llm_child_result() is None


def test_adapter_capability_is_one_literal_integer_not_boolean_or_duplicate():
    tree=ast.parse(Path(adapter.__file__).read_text());nodes=[node for node in tree.body
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROVIDER_BUDGET_POLICY_VERSION' for t in node.targets)]
    assert len(nodes)==1 and isinstance(nodes[0].value,ast.Constant)
    assert type(nodes[0].value.value) is int and nodes[0].value.value==1
