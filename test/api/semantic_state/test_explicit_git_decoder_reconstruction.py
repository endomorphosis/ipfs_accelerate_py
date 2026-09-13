"""Consumer request/budget cannot be substituted by a serialized decoder claim."""
import json
import subprocess

import pytest
from ipfs_accelerate_py.agent_supervisor.semantic_state import chunked_reconstruction as R
from ipfs_datasets_py.logic.software_contracts.semantic_index import chunked_snapshot as C
from ipfs_datasets_py.logic.software_contracts.semantic_index import streaming_scanner as S
from ipfs_datasets_py.logic.software_contracts.semantic_index import git_decoder_profile as D


@pytest.fixture
def source(tmp_path):
    root=tmp_path/'source';root.mkdir()
    def git(*args):
        return subprocess.check_output(['git','-c','core.hooksPath=/dev/null','-C',str(root),*args],
                                       stderr=subprocess.DEVNULL).decode().strip()
    git('init','-b','main')
    (root/'module.py').write_bytes(b'def add(a,b): return a+b\n')
    (root/'test_module.py').write_bytes(b'from module import add\ndef test_add(): assert add(1,2)==3\n')
    git('add','-A');git('-c','user.name=Fixture','-c','user.email=test@example.invalid','commit','-qm','fixture')
    request=dict(repository_id='fixture:explicit-consumer',expected_commit=git('rev-parse','HEAD'),
                 expected_tree=git('rev-parse','HEAD^{tree}'))
    return root,request,C.snapshot_chunked_repository(root,**request)

PROFILE=D.GitBlobDecoderProfile(D.EXPLICIT_DECODER_ADDRESS_BYTES)
BUDGET=D.GitBlobDecoderBudget(D.EXPLICIT_DECODER_ADDRESS_BYTES)


def acquired(source):
    root,request,_=source
    return C.snapshot_chunked_repository(root,**request,decoder_profile=PROFILE,decoder_budget=BUDGET)


def reconstruct(source,snapshot,**kwargs):
    root,request,_=source
    return R.reconstruct_chunked_semantic_state(root,snapshot,**request,
        expected_chunked_snapshot_cid=snapshot.snapshot_cid,**kwargs)


@pytest.mark.parametrize('streaming',[False,True])
def test_explicit_profile_consumes_full_content_and_is_configuration_bound(source,streaming):
    snapshot=acquired(source)
    options={'streaming_limits':S.StreamingScanLimits()} if streaming else {}
    result=reconstruct(source,snapshot,decoder_profile=PROFILE,decoder_budget=BUDGET,**options)
    assert result.observation()['schema'].endswith('@4')
    assert result.git_decoder_observation=={'profile':PROFILE.payload(),'caller_budget':{'max_address_space_bytes':256*1024**2}}
    assert not result.observation()['complete_analysis_authority']
    root=json.loads(result.bundle.blocks[result.chunked_snapshot_cid])
    assert root['schema'].endswith('@2') and root['git_decoder_profile']==PROFILE.payload()
    assert all(len(raw)<=R.MAX_BUNDLE_BLOCK_BYTES for raw in result.bundle.blocks.values())
    if streaming:
        assert result.streaming_scan_observation['git_decoder_profile']==PROFILE.payload()
        assert result.streaming_scan_observation['retained_snapshot_source_bytes']==0
        assert result.streaming_scan_observation['limits']==S.StreamingScanLimits().__dict__
    assert reconstruct(source,snapshot,decoder_profile=PROFILE,decoder_budget=BUDGET,
                       nominated_bundle=result.bundle,**options).nomination_matched


@pytest.mark.parametrize('serialized',[False,True])
@pytest.mark.parametrize('request_kind',['absent','unbudgeted','wrong_profile','untyped_budget'])
def test_manifest_does_not_self_authorize_decoder_resources_or_profile(source,monkeypatch,serialized,request_kind):
    snapshot=acquired(source);root,identity,_=source
    cid,blocks=snapshot.manifest_blocks()
    value=blocks if serialized else snapshot
    monkeypatch.setattr(C,'_hash_blob',lambda *a,**kw:pytest.fail('unadmitted content read'))
    monkeypatch.setattr(S,'_hash_blob',lambda *a,**kw:pytest.fail('unadmitted streaming analysis'))
    options={}
    if request_kind=='unbudgeted':options={'decoder_profile':PROFILE}
    elif request_kind=='wrong_profile':options={'decoder_profile':D.DEFAULT_DECODER_PROFILE,'decoder_budget':BUDGET}
    elif request_kind=='untyped_budget':options={'decoder_profile':PROFILE,'decoder_budget':{'max_address_space_bytes':256*1024**2}}
    with pytest.raises(R.ReconstructionError,match='profile|budget'):
        R.reconstruct_chunked_semantic_state(root,value,**identity,
            expected_chunked_snapshot_cid=cid,streaming_limits=S.StreamingScanLimits(),**options)


def test_default_v1_stays_default_and_explicit_budget_changes_configuration_only(source):
    _,_,snapshot=source
    old=reconstruct(source,snapshot)
    admitted=reconstruct(source,snapshot,decoder_budget=BUDGET)
    assert old.observation()['schema'].endswith('@2') and old.git_decoder_observation is None
    assert old.bundle==admitted.bundle and old.configuration_digest!=admitted.configuration_digest
    assert admitted.git_decoder_observation['profile']['address_space_bytes']==128*1024**2
    assert json.loads(old.bundle.blocks[old.chunked_snapshot_cid])['schema'].endswith('@1')


def test_default_profile_bundle_cannot_nominate_an_explicit_profile_acquisition(source):
    _,_,default=source
    nominated=reconstruct(source,default).bundle
    with pytest.raises(R.ReconstructionError):
        reconstruct(source,acquired(source),decoder_profile=PROFILE,decoder_budget=BUDGET,
                    nominated_bundle=nominated,streaming_limits=S.StreamingScanLimits())
