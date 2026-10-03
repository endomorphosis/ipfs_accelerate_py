"""Actual signed indexes/worlds with authored numerical-currentness boundaries."""
from copy import deepcopy
import hashlib
import json
import time
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original
from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _proposal_json, _version
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as numerical


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@pytest.fixture
def selected(original, monkeypatch):
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    result = prep.initial_context(state=state)
    config = state / 'authored-config.json'
    config.write_bytes(b'{"scope":"authored-currentness-boundary"}')
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    output = state / 'source384-context'; output.mkdir()
    receipt = dict(schema='terminal-source384-repository-context@1',repository=str(root),output=str(output),
        config_path=str(config),config_sha256=config_sha,
        source_hashes={name: row['sha256'] for name,row in prepared['manifest']['payload']['sources'].items()},
        summary=dict(schema='terminal-source384-planning-summary@1',checkpoint_sha256='c'*64,
            inference_sha256='d'*64,source_files=3,training_steps=0,provider_calls=0,
            proof_authority=False,execution_authority=False,completion_authority=False),
        completion_authority=False,execution_authority=False)
    receipt_raw=raw(receipt); (output/'receipt.json').write_bytes(receipt_raw)
    prep._write(state/'source384-selection.json',dict(config_path=str(config),config_sha256=config_sha))
    descriptor_path=root/result['descriptor']['artifact']; descriptor=json.loads(descriptor_path.read_bytes())
    descriptor['source384_context']=receipt; descriptor_path.write_bytes(raw(descriptor))
    result.update(source384_context=receipt,descriptor=initial._reference(root,descriptor_path))
    prep._write(state/'initial-context-result.json',result)
    calls=[]
    def check(*, repository, expected_receipt):
        calls.append(dict(repository=repository,expected_receipt=deepcopy(expected_receipt)))
        if (config.read_bytes()!=b'{"scope":"authored-currentness-boundary"}'
                or (output/'receipt.json').read_bytes()!=receipt_raw
                or expected_receipt!=receipt):
            raise ValueError('authored numerical currentness changed')
        return expected_receipt
    monkeypatch.setattr(numerical,'validate_source384_context',check)
    _version(monkeypatch)
    return SimpleNamespace(root=root,state=state,prepared=prepared,receipt=receipt,result=result,
        descriptor_path=descriptor_path,config=config,output=output,calls=calls,checker=check)


def task_rows(case):
    with IntentRepository(case.state/'intent.duckdb',install_schema=False) as intent:
        return intent.plan_projection()['tasks']


def plan(case, provider=None):
    return prep.plan(case.state,provider_callable=provider or (lambda *a,**k:
        dict(text=_proposal_json(case.prepared),observation={},execution_receipt=None)))


def test_staging_has_no_numerical_currentness_and_full_gate_remains(selected):
    c=selected
    staged=initial.stage_initial_context_nomination(state=c.state,prepared=c.prepared,require_empty_owner=True)
    assert c.calls==[] and staged['source384_currentness']=='not_established_by_nomination_staging'
    checked=initial.load_initial_context(state=c.state,prepared=c.prepared,require_empty_owner=True)
    assert len(c.calls)==1 and staged['summaries']==checked['summaries']
    assert not task_rows(c)


def test_direct_plan_preserves_predispatch_and_independent_admission_gates(selected):
    c=selected; provider=[]
    def router(*a,**k):
        assert len(c.calls)==1
        assert not task_rows(c)
        provider.append(True)
        return dict(text=_proposal_json(c.prepared),observation={},execution_receipt=None)
    result=plan(c,router)
    assert result['qualified'] and provider==[True] and len(c.calls)==2
    assert len(task_rows(c))==1


@pytest.mark.parametrize('mutation',['config','receipt'])
def test_mutation_after_staging_refuses_before_provider_and_materialization(selected,monkeypatch,mutation):
    c=selected; original_stage=initial.stage_initial_context_nomination; provider=[]
    def staged(**kwargs):
        value=original_stage(**kwargs)
        (c.config if mutation=='config' else c.output/'receipt.json').write_bytes(b'changed')
        return value
    monkeypatch.setattr(initial,'stage_initial_context_nomination',staged)
    result=plan(c,lambda *a,**k:provider.append(True))
    assert not result['qualified'] and not provider and len(c.calls)==1
    assert not task_rows(c) and not (c.state/'admission.json').exists()


def test_mutation_during_provider_refuses_independent_admission(selected):
    c=selected
    def router(*a,**k):
        c.config.write_bytes(b'changed')
        return dict(text=_proposal_json(c.prepared),observation={},execution_receipt=None)
    result=plan(c,router)
    assert not result['qualified'] and result['provider_calls']==1 and len(c.calls)==2
    assert not task_rows(c) and not (c.state/'admission.json').exists()


def test_admitted_world_defers_numerical_check_to_final_publication_gate(selected):
    c=selected; assert plan(c)['qualified']; c.calls.clear(); before=task_rows(c)
    result=prep.context(state=c.state)
    assert len(c.calls)==1 and result['context_bundle'] and task_rows(c)==before
    from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import read_task_context_historical_selection
    task=task_rows(c)[0]
    bundle=result['context_bundle']
    assert read_task_context_historical_selection(repository=c.root,artifact=bundle['artifact'],
        expected_sha256=bundle['sha256'],task_cid=result['task_cid'],task_id=c.prepared['spec']['task_key'])['source384_context']==c.receipt


@pytest.mark.parametrize('moment',['before','during'])
def test_world_build_never_publishes_stale_numerical_nomination(selected,monkeypatch,moment):
    c=selected; assert plan(c)['qualified']; c.calls.clear(); before=task_rows(c)
    if moment=='before': c.config.write_bytes(b'changed')
    else:
        persist=initial.persist_intent_world_snapshot
        def mutate(*a,**k):
            value=persist(*a,**k);c.config.write_bytes(b'changed');return value
        monkeypatch.setattr(initial,'persist_intent_world_snapshot',mutate)
    with pytest.raises(ValueError,match='numerical currentness changed'):
        prep.context(state=c.state)
    assert len(c.calls)==1 and task_rows(c)==before
    assert not (c.root/'.runtime/terminal-context-bundle.json').exists()
    assert not (c.root/'.runtime/terminal-context/result.json').exists()


def test_coherently_rewritten_nomination_during_world_capture_is_refused(selected,monkeypatch):
    c=selected; assert plan(c)['qualified']; before=task_rows(c)
    persist=initial.persist_intent_world_snapshot
    def mutate(*a,**k):
        value=persist(*a,**k)
        descriptor=json.loads(c.descriptor_path.read_bytes())
        descriptor['extra_nomination']='changed after capture'
        c.descriptor_path.write_bytes(raw(descriptor))
        result=json.loads((c.state/'initial-context-result.json').read_bytes())
        result['descriptor']=initial._reference(c.root,c.descriptor_path)
        prep._write(c.state/'initial-context-result.json',result)
        return value
    monkeypatch.setattr(initial,'persist_intent_world_snapshot',mutate)
    with pytest.raises(ValueError,match='nomination changed during admitted world'):
        prep.context(state=c.state)
    assert task_rows(c)==before and not (c.root/'.runtime/terminal-context-bundle.json').exists()


def test_staging_still_rejects_current_signed_source_drift(selected):
    c=selected;(c.root/'bottle.py').write_text('def other(): return 7\n')
    with pytest.raises(ValueError):
        initial.stage_initial_context_nomination(state=c.state,prepared=c.prepared,require_empty_owner=True)
    assert not c.calls and not task_rows(c)


def test_coherent_descriptor_replacement_after_staging_refuses_before_provider(selected,monkeypatch):
    c=selected; stage=initial.stage_initial_context_nomination; provider=[]
    def changed(**kwargs):
        value=stage(**kwargs)
        descriptor=json.loads(c.descriptor_path.read_bytes())
        descriptor['extra_nomination']='new selection with identical summaries'
        c.descriptor_path.write_bytes(raw(descriptor))
        result=json.loads((c.state/'initial-context-result.json').read_bytes())
        result['descriptor']=initial._reference(c.root,c.descriptor_path)
        prep._write(c.state/'initial-context-result.json',result)
        return value
    monkeypatch.setattr(initial,'stage_initial_context_nomination',changed)
    result=plan(c,lambda *a,**k:provider.append(True))
    assert not result['qualified'] and not provider and len(c.calls)==1
    assert not task_rows(c) and not (c.state/'admission.json').exists()


@pytest.mark.parametrize('mutation',['symlink','fifo','hardlink','oversize'])
def test_nomination_marker_requires_bounded_canonical_regular_bytes(selected,mutation):
    import os
    c=selected; marker=c.state/'initial-context-result.json'
    if mutation=='symlink':
        other=c.state/'saved-marker.json';marker.rename(other);marker.symlink_to(other)
    elif mutation=='fifo':
        marker.unlink();os.mkfifo(marker)
    elif mutation=='hardlink':os.link(marker,c.state/'linked-marker.json')
    else:marker.write_bytes(b'x'*(initial.MAX_BYTES+1))
    with pytest.raises(ValueError):
        initial.stage_initial_context_nomination(state=c.state,prepared=c.prepared,require_empty_owner=True)
    assert not c.calls and not task_rows(c)
