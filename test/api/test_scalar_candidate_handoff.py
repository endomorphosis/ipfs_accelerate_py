"""Actual owner admission and allocated Git writes with explicit model test doubles."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding.local_planning_qualification import prepare_local_task
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.runtime import scalar_candidate_handoff as author
from ipfs_accelerate_py.agent_supervisor.runtime import scalar_candidate_runner as worker
from ipfs_accelerate_py.agent_supervisor.runtime import scalar_repair_advisor as advisor
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity

SOURCE='def derive(capacity: int, threshold: int) -> int:\n    return capacity + threshold\n'
AFTER=SOURCE.replace(' + ',' * ')
INSTRUCTION='the runner must compute result; requires true; ensures result = old(left) * old(right) and returned.'


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.DEVNULL).decode().strip()


@pytest.fixture
def scenario(tmp_path,monkeypatch):
    root=tmp_path/'repository';root.mkdir()
    (root/'source.py').write_text(SOURCE)
    (root/'instruction.txt').write_text(INSTRUCTION)
    (root/'check.py').write_text('from source import derive\nassert derive(2,3)==6\n')
    for argv in [('init','-q'),('config','user.name','Scalar tests'),('config','user.email','scalar@example.invalid'),
                 ('add','.'),('commit','-qm','independent baseline')]:git(root,*argv)
    (root/'.git/info/exclude').write_text('.runtime/\n')
    with IntentRepository(tmp_path/'intent.duckdb') as intent:
        declared=prepare_local_task(repository=root,state=tmp_path/'policy',intent=intent,
            scope_paths=['source.py','check.py','instruction.txt'],output_path='source.py',
            validation_argv=['python3','-B','check.py'],objective=INSTRUCTION)
        candidate=dict(id='replace:mul',status='checked_candidate',effect_status='satisfied',
            enabled_case_count=9,bounded_effects_satisfied=True,live_build_verified=True,
            before_source_sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),source_text=AFTER,
            source_sha256=hashlib.sha256(AFTER.encode()).hexdigest())
        evidence=dict(schema='supervisor-scalar-operator-repair-advice/v1',status='candidate_evidence',
            initial_refutation_live_verified=True,input_pins_rechecked=True,original_source_unchanged=True,
            original_source=dict(id='source.py',source_text=SOURCE,source_sha256=candidate['before_source_sha256']),
            candidates=[candidate,{**candidate,'id':'replace:sub','effect_status':'refuted','bounded_effects_satisfied':False}],
            satisfied_candidate_ids=['replace:mul'],**advisor.FALSE)
        calls=[]
        def fresh(**kwargs):calls.append(deepcopy(kwargs));return deepcopy(evidence)
        monkeypatch.setattr(advisor,'prepare_scalar_repair_advice',fresh)
        options=dict(repository=root,admission=declared['admission'],intent=intent,task_cid=declared['task_cid'],
            state=tmp_path/'candidate',instruction=INSTRUCTION,intent_config={},security_config={},effect_config={})
        yield dict(root=root,intent=intent,declared=declared,options=options,evidence=evidence,calls=calls)


def materialization(scenario,tmp_path):
    report=author.prepare_scalar_candidate_handoff(**scenario['options'])
    worktree=tmp_path/'allocated'
    git(scenario['root'],'worktree','add','--detach',str(worktree),'HEAD')
    request=dict(artifact=Path(report['handoff_path']),expected_sha256=report['handoff_sha256'],
        task_cid=report['task_cid'],prompt=json.dumps({'objective_id':report['task_id']}),workspace=worktree)
    return report,request


def test_fresh_advisor_called_then_native_worktree_only_is_changed(scenario,tmp_path):
    report,request=materialization(scenario,tmp_path)
    assert len(scenario['calls'])==1
    assert scenario['calls'][0]['source_rows']==[scenario['evidence']['original_source']]
    assert report['status']=='candidate_ready'
    assert all(report[key] is False for key in author.FALSE)
    assert Path(report['handoff_path']).stat().st_mode & 0o222==0
    assert Path(report['handoff_path']).is_relative_to(scenario['root']/'.runtime/scalar-handoffs')
    task_before=scenario['intent'].get_task(report['task_cid'])
    observed=worker.materialize_scalar_candidate(**request)
    assert observed['status']=='candidate_materialized'
    assert (request['workspace']/'source.py').read_text()==AFTER
    assert (scenario['root']/'source.py').read_text()==SOURCE
    assert git(request['workspace'],'rev-parse','HEAD')==report['handoff']['baseline_commit']
    assert git(request['workspace'],'diff','--name-only')=='source.py'
    assert scenario['intent'].get_task(report['task_cid'])==task_before


@pytest.mark.parametrize('change',['no_positive','ambiguous','no_enabled','no_live','no_refutation','stale_pins','starting_satisfied','budget_limited'])
def test_no_candidate_handoff_from_incomplete_or_ambiguous_evidence(scenario,change):
    evidence=scenario['evidence'];candidate=evidence['candidates'][0]
    if change=='no_positive':candidate['effect_status']='refuted'
    if change=='ambiguous':evidence['candidates'][1]={**candidate,'id':'second'}
    if change=='no_enabled':candidate['enabled_case_count']=0
    if change=='no_live':candidate['live_build_verified']=False
    if change=='no_refutation':evidence['initial_refutation_live_verified']=False
    if change=='stale_pins':evidence['input_pins_rechecked']=False
    if change=='starting_satisfied':evidence['status']='no_starting_counterexample'
    if change=='budget_limited':evidence['candidates'][1]['status']='not_checked_candidate_budget'
    report=author.prepare_scalar_candidate_handoff(**scenario['options'])
    assert report['status']=='residual' and report['handoff'] is None
    assert len(report['repair_advice']['candidates'])==len(evidence['candidates'])
    assert not (scenario['root']/'.runtime/scalar-handoffs').exists()


@pytest.mark.parametrize('change',['foreign_instruction','source_drift','foreign_repository','output_instruction'])
def test_independent_admission_and_instruction_required_before_inference(scenario,tmp_path,change):
    options=deepcopy({k:v for k,v in scenario['options'].items() if k!='intent'});options['intent']=scenario['intent']
    if change=='foreign_instruction':options['instruction']='unbound instruction'
    if change=='source_drift':(scenario['root']/'source.py').write_text(AFTER)
    if change=='foreign_repository':options['repository']=tmp_path
    if change=='output_instruction':options['instruction_path']='source.py'
    with pytest.raises(ValueError):author.prepare_scalar_candidate_handoff(**options)
    assert scenario['calls']==[]


def test_immutable_source_instruction_binding_and_task_drift_recheck(scenario,monkeypatch):
    report=author.prepare_scalar_candidate_handoff(**{**scenario['options'],'instruction_path':'instruction.txt'})
    assert report['handoff']['instruction_binding']['kind']=='signed_source'
    def changed(**kwargs):
        task=scenario['intent'].get_task(scenario['declared']['task_cid'])
        scenario['intent'].cas_task_status(task_cid=task['task_cid'],expected_revision=task['revision'],new_status='in_progress')
        return deepcopy(scenario['evidence'])
    monkeypatch.setattr(advisor,'prepare_scalar_repair_advice',changed)
    with pytest.raises(ValueError,match='task'):
        author.prepare_scalar_candidate_handoff(**{**scenario['options'],'state':scenario['options']['state'].with_name('changed')})


@pytest.mark.parametrize('change',['wrong_source','wrong_after','authority','wrong_population'])
def test_incompatible_fresh_consumer_response_cannot_be_exported(scenario,change):
    evidence=scenario['evidence']
    if change=='wrong_source':evidence['original_source']['source_text']='foreign'
    if change=='wrong_after':evidence['candidates'][0]['source_sha256']='f'*64
    if change=='authority':evidence['mutation_authority']=True
    if change=='wrong_population':evidence['satisfied_candidate_ids']=[]
    with pytest.raises(ValueError):author.prepare_scalar_candidate_handoff(**scenario['options'])
    assert not scenario['options']['state'].exists()


@pytest.mark.parametrize('change',['bad_digest','bad_task','bad_prompt','canonical_workspace','writable_artifact','source_drift','linked_source','linked_artifact','writable_directory','canonical_source_drift','copied_artifact'])
def test_runner_refuses_tamper_without_canonical_publication(scenario,tmp_path,change):
    report,request=materialization(scenario,tmp_path)
    if change=='bad_digest':request['expected_sha256']='f'*64
    if change=='bad_task':request['task_cid']='foreign-task'
    if change=='bad_prompt':request['prompt']='{"objective_id":"foreign"}'
    if change=='canonical_workspace':request['workspace']=scenario['root']
    if change=='writable_artifact':request['artifact'].chmod(0o644)
    if change=='source_drift':(request['workspace']/'source.py').write_text(SOURCE+'\n')
    if change=='linked_source':
        p=request['workspace']/'source.py';p.unlink();p.symlink_to(scenario['root']/'source.py')
    if change=='writable_directory':request['artifact'].parent.chmod(0o777)
    if change=='canonical_source_drift':(scenario['root']/'source.py').write_text(SOURCE+'# changed\n')
    if change=='copied_artifact':
        p=tmp_path/'copy.json';p.write_bytes(request['artifact'].read_bytes());p.chmod(0o444);request['artifact']=p
    if change=='linked_artifact':
        link=tmp_path/'linked.json';link.symlink_to(request['artifact']);request['artifact']=link
    with pytest.raises((ValueError,OSError,subprocess.CalledProcessError)):worker.materialize_scalar_candidate(**request)
    expected=SOURCE+'# changed\n' if change=='canonical_source_drift' else SOURCE
    assert (scenario['root']/'source.py').read_text()==expected
    assert scenario['intent'].get_task(report['task_cid'])['status']=='ready'


def test_rehashed_candidate_traversal_and_claimed_authority_are_rejected(scenario,tmp_path):
    report,request=materialization(scenario,tmp_path)
    for index,change in enumerate(['traversal','authority','empty_domain','extra_field']):
        payload=deepcopy(report['handoff'])
        if change=='traversal':payload['edit']['path']=payload['permitted_output']['path']='../source.py'
        if change=='authority':payload['publication_authority']=True
        if change=='empty_domain':payload['evidence']['enabled_case_count']=0
        if change=='extra_field':payload['proof_receipt_id']='invented'
        payload['artifact_cid']=content_identity({k:v for k,v in payload.items() if k!='artifact_cid'})
        path=tmp_path/f'changed-{index}.json';raw=author._wire(payload);path.write_bytes(raw);path.chmod(0o444)
        with pytest.raises(ValueError):worker.materialize_scalar_candidate(**{**request,'artifact':path,'expected_sha256':author._sha(raw)})
    assert (request['workspace']/'source.py').read_text()==SOURCE
