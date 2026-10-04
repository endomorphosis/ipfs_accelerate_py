"""Exact opaque artifact admission using real Git collection and native task envelopes."""
import hashlib,importlib.util,io,json,subprocess,zipfile
from dataclasses import replace
from pathlib import Path
import pytest
from ipfs_accelerate_py.agent_supervisor.validation import proposal_validation as P
from ipfs_accelerate_py.agent_supervisor.proof.code_proof_obligations import collect_git_candidate_diff
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as D
# Reuse durable identity fixtures; production functions above are not stubbed.
spec=importlib.util.spec_from_file_location('artifact_identity_fixture',Path(__file__).with_name('test_agent_supervisor_untrusted_proposal.py'));T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
PDF=b'%PDF-1.4\n\x00qualification fixture only\n%%EOF\n'
def zip_bytes():
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w') as z:z.writestr('README.txt','Inert bounded archive fixture\n')
 return b.getvalue()
def git(root,*args):return subprocess.check_output(['git','-c','core.autocrlf=false',*args],cwd=root).decode()
def collect(tmp_path,files):
 git(tmp_path,'init','-q');git(tmp_path,'config','user.name','Qualification');git(tmp_path,'config','user.email','test@invalid');git(tmp_path,'commit','--allow-empty','-qm','Baseline')
 for name,data in files.items():p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 git(tmp_path,'add','-A');entries=collect_git_candidate_diff(tmp_path,base_revision='HEAD');patch=git(tmp_path,'diff','--no-ext-diff','--no-color','--full-index','HEAD')
 ops=tuple(P.ProposalOperation(operation=e.change_kind.value,path=e.path,old_path=e.old_path,rationale_refs=(T.RATIONALE,)) for e in entries)
 effects=tuple(P.ProposalExpectedEffect(operation=e.change_kind.value,path=e.path,before_sha256=T._sha(e.before_source),after_sha256=T._sha(e.after_source)) for e in entries)
 return replace(T._proposal(),candidate_diff=entries,declared_paths=tuple(files),operations=ops,expected_effects=effects,patch_text=patch,proposal_id='')
def policy(paths=(),**kw):return T._policy(allowed_paths=('out/',),task_owned_paths=('out/',),binary_artifact_paths=paths,max_findings=30,**kw)
def codes(p,pol):return {f.code.value for f in P.validate_implementation_proposal(p,policy=pol).findings}
def task(paths):
 env={'schema':D.PROPOSAL_SCOPED_BINARY_ARTIFACT_ENVELOPE_SCHEMA,'binary_paths':list(paths),'max_file_bytes':16_000_000,'max_patch_bytes':16_000_000,'max_output_bytes':24_000_000}
 return D.PortalTask(task_id=T.TASK_ID,title='Fixture',status='ready',completion='',priority='1',track='test',outputs=['out'],metadata={D.PROPOSAL_ARTIFACT_ENVELOPE_METADATA_KEY:json.dumps(env)})
@pytest.mark.parametrize('name,data',[('paper.pdf',PDF),('supplement.zip',zip_bytes())])
def test_actual_git_exact_artifact_and_dynamic_text_admitted(tmp_path,name,data):
 path='out/'+name;p=collect(tmp_path,{path:data,'out/snapshots/dynamic-123/log.txt':b'Actual fixture log\n'})
 limits=D.PortalImplementationDaemon._proposal_local_envelope_limits(p,task=task([path]));assert limits['binary_artifact_paths']==(path,)
 pol=policy((path,),**{k:v for k,v in limits.items() if k.startswith('max_')})
 r=P.validate_implementation_proposal(p,policy=pol);assert r.accepted,[(f.code.value,f.message) for f in r.findings]
 assert not pol.allow_binary and not pol.allow_archives
 assert 'binary_change_forbidden' in codes(p,policy())
 e=next(e for e in p.candidate_diff if e.binary);assert e.metadata['after_sha256']==hashlib.sha256(data).hexdigest();assert e.metadata['after_size_bytes']==len(data)
@pytest.mark.parametrize('name,data',[('other.pdf',PDF),('main.tex',PDF),('receipt.json',PDF),('other.zip',zip_bytes())])
def test_exact_authority_does_not_admit_unlisted_binary(tmp_path,name,data):
 p=collect(tmp_path,{'out/'+name:data});assert 'binary_change_forbidden' in codes(p,policy(('out/paper.pdf',)))
@pytest.mark.parametrize('path',['/out/paper.pdf','out/../paper.pdf','out/*.pdf','out//paper.pdf','other/paper.pdf','out/model.pt'])
def test_authority_path_must_be_exact_supported_task_owned(path):
 with pytest.raises(P.ProposalValidationError):policy((path,))
@pytest.mark.parametrize('data',[b'not a pdf\x00',b'PK\x03\x04not-pdf\x00'])
def test_disguised_pdf_rejected(tmp_path,data):
 p=collect(tmp_path,{'out/paper.pdf':data});assert 'binary_change_forbidden' in codes(p,policy(('out/paper.pdf',)))
def test_oversize_artifact_aggregate_not_hidden_by_binary_diff(tmp_path):
 p=collect(tmp_path,{'out/paper.pdf':PDF});e=p.candidate_diff[0];e=replace(e,metadata={**e.metadata,'after_size_bytes':16_000_001});p=replace(p,candidate_diff=(e,),proposal_id='')
 assert 'binary_artifact_paths' not in D.PortalImplementationDaemon._proposal_local_envelope_limits(p,task=task(['out/paper.pdf']))
 assert 'binary_change_forbidden' in codes(p,policy(('out/paper.pdf',)))
def test_policy_identity_roundtrip_and_default_identity_unchanged():
 old=policy();assert 'binary_artifact_paths' not in old.to_dict();new=policy(('out/paper.pdf',));assert new.policy_id!=old.policy_id;assert P.ProposalValidationPolicy.from_dict(new.to_dict())==new
@pytest.mark.parametrize('field,value',[('binary_paths',['out/*.pdf']),('binary_paths',['other/paper.pdf']),('binary_paths',['out/model.pt']),('allow_binary',True),('paths',['out/paper.pdf'])])
def test_invalid_declared_envelope_fails_closed(tmp_path,field,value):
 p=collect(tmp_path,{'out/paper.pdf':PDF});t=task(['out/paper.pdf']);v=json.loads(t.metadata[D.PROPOSAL_ARTIFACT_ENVELOPE_METADATA_KEY]);v[field]=value;t.metadata[D.PROPOSAL_ARTIFACT_ENVELOPE_METADATA_KEY]=json.dumps(v)
 assert 'binary_artifact_paths' not in D.PortalImplementationDaemon._proposal_local_envelope_limits(p,task=t)
def test_opaque_untrusted_json_stays_rejected(tmp_path):
 p=collect(tmp_path,{'out/paper.pdf':PDF});baseline=tmp_path/'baseline';baseline.mkdir();r=P.validate_untrusted_implementation_proposal(p.to_dict(),policy=policy(('out/paper.pdf',)),repository_root=baseline);assert not r.accepted
@pytest.mark.parametrize('kwargs',[{'generated_path_patterns':('out/*',)},{'protected_paths':('out/paper.pdf',)},{'symlink_paths':('out/',)}])
def test_exact_binary_authority_keeps_other_protections(tmp_path,kwargs):
 p=collect(tmp_path,{'out/paper.pdf':PDF});assert not P.validate_implementation_proposal(p,policy=policy(('out/paper.pdf',),**kwargs)).accepted


def test_opaque_sides_count_toward_policy_aggregate(tmp_path):
    p=collect(tmp_path,{'out/paper.pdf':PDF,'out/second.pdf':PDF})
    entries=tuple(replace(e,metadata={**e.metadata,'after_size_bytes':600_000}) for e in p.candidate_diff)
    p=replace(p,candidate_diff=entries,proposal_id='')
    assert 'patch_too_large' in codes(p,policy(('out/paper.pdf','out/second.pdf'),max_patch_bytes=1_000_000))


