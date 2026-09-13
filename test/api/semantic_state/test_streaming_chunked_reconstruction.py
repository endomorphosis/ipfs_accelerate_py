"""Explicit streaming integration preserves paged roots and admission guards."""
from dataclasses import replace
import json
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state import chunked_reconstruction as r
from ipfs_datasets_py.logic.software_contracts.semantic_index import chunked_snapshot as c
from ipfs_datasets_py.logic.software_contracts.semantic_index import streaming_scanner as s
from ipfs_datasets_py.logic.software_contracts.semantic_state.models import SemanticStateBundle
from ipfs_datasets_py.logic.software_contracts.content import canonical_dag_json_bytes, cid_for_structured


def git(root,*args):
    return subprocess.check_output(['git','-c','core.hooksPath=/dev/null','-C',str(root),*args],stderr=subprocess.DEVNULL).decode().strip()


@pytest.fixture
def source(tmp_path):
    root=tmp_path/'source'
    root.mkdir()
    git(root,'init','-b','main')
    (root/'module.py').write_text('def add(a,b): return a+b\n')
    (root/'test_module.py').write_text('from module import add\ndef test_add(): assert add(1,2)==3\n')
    git(root,'add','.')
    git(root,'-c','user.name=Fixture','-c','user.email=test@example.invalid','commit','-qm','fixture')
    request=dict(repository_id='fixture:stream-consumer',expected_commit=git(root,'rev-parse','HEAD'),
                 expected_tree=git(root,'rev-parse','HEAD^{tree}'))
    chunked=c.snapshot_chunked_repository(root,**request)
    return root,request,chunked


def reconstruct(source,**kwargs):
    root,request,chunked=source
    return r.reconstruct_chunked_semantic_state(root,chunked,**request,
                expected_chunked_snapshot_cid=chunked.snapshot_cid,**kwargs)


def test_explicit_streaming_preserves_complete_paged_bundle_and_changes_configuration(source):
    old=reconstruct(source)
    result=reconstruct(source,streaming_limits=s.StreamingScanLimits())
    assert result.snapshot_cid==old.snapshot_cid
    assert result.state_cid==old.state_cid
    assert result.bundle.blocks==old.bundle.blocks
    assert result.configuration_digest!=old.configuration_digest
    assert old.streaming_scan_observation is None
    assert old.observation()['schema'].endswith('@2')
    assert result.observation()['schema'].endswith('@3')
    assert result.streaming_scan_observation['retained_snapshot_source_bytes']==0
    assert result.observation()['complete_analysis_authority'] is False
    assert reconstruct(source,streaming_limits=s.StreamingScanLimits(),nominated_bundle=result.bundle).nomination_matched


def test_streaming_limits_and_worker_profile_are_configuration_bound(source,monkeypatch):
    first=reconstruct(source,streaming_limits=s.StreamingScanLimits())
    changed=reconstruct(source,streaming_limits=s.StreamingScanLimits(max_ast_nodes=50000))
    assert first.bundle.root.root_cid==changed.bundle.root.root_cid
    assert first.configuration_digest!=changed.configuration_digest
    original=s.analysis_process_profile
    monkeypatch.setattr(s,'analysis_process_profile',lambda:{**original(),'qualification_fixture':'changed'})
    profiled=reconstruct(source,streaming_limits=s.StreamingScanLimits())
    assert profiled.configuration_digest!=first.configuration_digest


def test_streaming_ast_refusal_retains_structured_diagnostics(source):
    with pytest.raises(r.StreamingReconstructionError) as failure:
        reconstruct(source,streaming_limits=s.StreamingScanLimits(max_ast_nodes=1))
    assert failure.value.observation()['code']=='ast_node_budget'
    assert failure.value.observation()['complete_analysis_authority'] is False


def test_streaming_nomination_missing_snapshot_page_is_refused(source):
    result=reconstruct(source,streaming_limits=s.StreamingScanLimits())
    blocks=dict(result.bundle.blocks)
    root=json.loads(blocks[result.paged_snapshot_cid])
    del blocks[root['entry_pages'][0]]
    with pytest.raises(r.ReconstructionError):
        reconstruct(source,streaming_limits=s.StreamingScanLimits(),
                    nominated_bundle=SemanticStateBundle(result.bundle.root,blocks))


def test_streaming_still_refuses_individual_oversized_bundle_block(source,monkeypatch):
    from ipfs_datasets_py.logic.software_contracts import semantic_state
    original=semantic_state.build_semantic_state
    def oversized(*args,**kwargs):
        result=original(*args,**kwargs)
        blocks=dict(result.blocks)
        payload={'oversized':'x'*(r.MAX_BUNDLE_BLOCK_BYTES+1)}
        blocks[cid_for_structured(payload)]=canonical_dag_json_bytes(payload)
        return SemanticStateBundle(result.root,blocks)
    monkeypatch.setattr(semantic_state,'build_semantic_state',oversized)
    with pytest.raises(r.StreamingReconstructionError) as failure:
        reconstruct(source,streaming_limits=s.StreamingScanLimits())
    assert failure.value.observation()['code']=='semantic_bundle_frame'


def test_full_streaming_consumer_accepts_real_ordinary_population_over_128_mib(source):
    root,_,_=source
    (root/'module.py').unlink()
    (root/'test_module.py').unlink()
    padding=b'#'+b'x'*(2*1024*1024-32)+b'\n'
    expected={}
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
    for i in range(66):
        raw=f'# file {i:05}\n'.encode()+padding
        name=f'file_{i:03}.py'
        (root/name).write_bytes(raw)
        expected[name]=(len(raw),cid_for_bytes(raw))
    del raw,padding
    git(root,'add','-A')
    git(root,'-c','user.name=Fixture','-c','user.email=test@example.invalid','commit','-qm','large ordinary fixture')
    request=dict(repository_id='fixture:stream-consumer',expected_commit=git(root,'rev-parse','HEAD'),
                 expected_tree=git(root,'rev-parse','HEAD^{tree}'))
    admission=c.ChunkedSnapshotLimits(max_stream_bytes=160*1024*1024)
    chunked=c.snapshot_chunked_repository(root,**request,limits=admission)
    total=sum(size for size,_ in expected.values())
    assert total>c.MAX_RETAINED_SOURCE_BYTES
    result=reconstruct((root,request,chunked),streaming_limits=s.StreamingScanLimits(),admission_limits=admission)
    assert result.streaming_scan_observation['ordinary_source_bytes_submitted']==total
    assert result.streaming_scan_observation['unique_blob_bytes_verified']==total
    assert result.streaming_scan_observation['retained_snapshot_source_bytes']==0
    assert not result.analysis_limitation_cids
    assert result.observation()['analysis_coverage']=='not_established'
    descriptor=json.loads(result.bundle.blocks[result.paged_snapshot_cid])
    entries=[entry for cid in descriptor['entry_pages'] for entry in json.loads(result.bundle.blocks[cid])['records']]
    assert len(entries)==len(expected)
    assert all(entry['kind']=='python' and entry['opaque_reason'] is None and
               (entry['size_bytes'],entry['source_cid'])==expected[entry['path']] for entry in entries)
    assert all(len(data)<=r.MAX_BUNDLE_BLOCK_BYTES for data in result.bundle.blocks.values())
