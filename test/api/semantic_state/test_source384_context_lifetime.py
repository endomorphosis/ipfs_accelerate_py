"""Logical input lifetime and unchanged fences; no model or RSS claims.

The production validator runs against small declared JSON/files. Native replay,
inventory and model configuration are isolated seams, not qualification doubles.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as owner
from ipfs_datasets_py.logic.software_contracts import codebase_source_units_384 as units
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead


@pytest.fixture
def context_case(tmp_path, monkeypatch):
    repository=tmp_path/'repo'; repository.mkdir()
    source=repository/'module.py'; source.write_text('def add(a, b): return a + b\n')
    output=tmp_path/'state'; output.mkdir()
    config_path=tmp_path/'config.json'; config_path.write_text('{}')
    checkpoint='c'*64
    config={'schema':'terminal-source384-config@1',
        'checkpoint_sha256':checkpoint,'embedding_snapshot':'declared-test-snapshot'}
    cid=cid_for_structured({'declared_fixture':True})
    head=CodebaseHead('fixture',1,cid,cid,f'rev:fixture:snapshot:{cid}',cid).to_dict()
    inference={'report':{'key':{'source_head':head,'version_id':'fixture-parent',
        'original_checkpoint_sha256':checkpoint},'preparation':{'paths':['module.py'],
        'max_functions':1024,'max_selected_units':128}}}
    inference_path=output/'inference.json'; inference_path.write_bytes(owner._raw(inference))
    inventory=[{'path':'module.py','disposition':'captured'}]
    producer={'declared_fixture':True}; summary={'declared_fixture_summary':True}
    receipt={'schema':owner.SCHEMA,'output':str(output),'repository':str(repository),
        'source_hashes':{'module.py':owner._sha(source.read_bytes())},'producer':producer,
        'training_steps':0,'config_path':str(config_path),'config_sha256':owner._sha(config_path.read_bytes()),
        'checkpoint_sha256':checkpoint,'inference_sha256':owner._sha(inference_path.read_bytes()),
        'source_head':head,'version_id':'fixture-parent','source_inventory':inventory,
        'summary':summary,**owner.AUTHORITY}
    (output/'receipt.json').write_bytes(owner._raw(receipt))
    events=[]; calls=[]; first_reads=[]
    class TrackedBytes(bytes):
        def __del__(self): events.append('initial_raw_released')
    original_read=owner._read
    def read(path, maximum):
        data=original_read(path,maximum)
        if path==inference_path and not first_reads:
            first_reads.append(True)
            return TrackedBytes(data)
        return data
    @contextmanager
    def owners(path):
        assert path==output
        yield object(), object()
    case=SimpleNamespace(repository=repository,output=output,receipt=receipt,
        inference=inference,inference_path=inference_path,events=events,calls=calls,
        source=source,producer=producer,on_validate=None)
    def validate(index,root,actual,**kwargs):
        assert root==repository and actual==inference
        calls.append({'raw_released':events==['initial_raw_released'],
                      'timeout':kwargs['timeout_seconds'],'memory':kwargs['memory_mb']})
        if case.on_validate: case.on_validate()
    monkeypatch.setattr(owner,'_read',read)
    monkeypatch.setattr(owner,'_pins',lambda:dict(producer))
    monkeypatch.setattr(owner,'load_source384_config',lambda path:dict(config))
    monkeypatch.setattr(owner,'_owners',owners)
    monkeypatch.setattr(owner,'_inventory',lambda *args:deepcopy(inventory))
    monkeypatch.setattr(owner,'_summary',lambda *args,**kwargs:deepcopy(summary))
    monkeypatch.setattr(units,'validate_shared_parent_units',validate)
    case.run=lambda:owner._validate_source384_context(repository=repository,
        expected_receipt=receipt,parent_lease=object(),deadline=time.monotonic()+30)
    return case


def test_source384_raw_bytes_released_before_native_replay(context_case):
    case=context_case
    before=deepcopy(case.receipt)
    result=case.run()
    assert case.calls==[{'raw_released':True,'timeout':case.calls[0]['timeout'],'memory':4096}]
    assert 0<case.calls[0]['timeout']<=30
    assert result==before and case.receipt==before and result is not case.receipt
    assert case.events==['initial_raw_released']


@pytest.mark.parametrize('damage',['initial_digest','initial_selection','final_digest','final_source','final_producer'])
def test_source384_input_and_fresh_fences_still_refuse(context_case,damage):
    case=context_case
    if damage=='initial_digest':
        case.inference_path.write_bytes(case.inference_path.read_bytes()+b' ')
    elif damage=='initial_selection':
        changed=deepcopy(case.inference); changed['report']['key']['version_id']='different'
        case.inference_path.write_bytes(owner._raw(changed))
        case.receipt['inference_sha256']=owner._sha(case.inference_path.read_bytes())
        (case.output/'receipt.json').write_bytes(owner._raw(case.receipt))
    else:
        def change():
            if damage=='final_digest':case.inference_path.write_bytes(case.inference_path.read_bytes()+b' ')
            elif damage=='final_source':case.source.write_text('changed source')
            else:case.producer['changed']=True
        case.on_validate=change
    with pytest.raises(ValueError):case.run()
    assert len(case.calls)==int(damage.startswith('final'))


def test_source384_replay_exception_is_preserved(context_case):
    failure=RuntimeError('declared native replay refusal')
    def reject():raise failure
    context_case.on_validate=reject
    with pytest.raises(RuntimeError) as result:context_case.run()
    assert result.value is failure


def test_source384_final_deadline_still_checked(context_case,monkeypatch):
    def expire():monkeypatch.setattr(owner,'_remaining',lambda deadline:owner._require(False,'expired fixture'))
    context_case.on_validate=expire
    with pytest.raises(ValueError,match='expired fixture'):context_case.run()
