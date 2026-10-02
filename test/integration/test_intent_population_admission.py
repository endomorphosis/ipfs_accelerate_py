"""Independent full-universe policy and bounded native population decisions."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import os
import site
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import intent_population_admission as population
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import repository_successor_context as successor
from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    PromptGoalRecord,PromptTaskRecord,PromptGoalGraph,PromptAcceptanceRecord,PromptValidationRecord,PromptOutputRecord)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from test.api.test_finite_integer_codebase import finite_tools,finite_git


@pytest.fixture
def profile(tmp_path):
    repository=tmp_path/'repository';repository.mkdir()
    for name in ('calc.py','other.py'):(repository/name).write_text('def increment(n: int) -> int:\n    return n + 1\n')
    (repository/'.gitignore').write_text('.runtime/\n')
    for argv in [('init','-q'),('config','user.name','Population fixture'),('config','user.email','fixture@example.invalid'),
        ('add','.'),('commit','-qm','Independent population baseline')]:finite_git(repository,*argv)
    (repository/'.runtime').mkdir(mode=0o755)
    Supervisor.init_local(repository=repository,consent=True,profile_dir=tmp_path/'profile',lifecycle_dir=tmp_path/'lifecycle')
    return tmp_path,repository


def universe(profile,text):
    state,repository=profile;policy=content_identity(local.LOCAL_POLICY)
    scopes=('calc.py','other.py');acceptance=[];validations=[]
    for key in ('calc','other'):
        validations.append(PromptValidationRecord(validation_key='check:'+key,argv=('python3','-c','import '+key),policy_cid=policy))
        acceptance.append(PromptAcceptanceRecord(criterion_key='accept:'+key,criterion='Explicit '+key+' acceptance',validation_keys=('check:'+key,)))
    root=PromptGoalRecord(goal_key='root',parent_goal_cid='',dependency_goal_cids=(),title='Reviewed requirement',objective=text,
        rationale='Independent original universe',scope_paths=scopes,acceptance=tuple(acceptance))
    child=replace(root,goal_key='work',parent_goal_cid=root.goal_cid,title='Potential execution tasks')
    tasks=tuple(PromptTaskRecord(task_key=key,goal_cid=child.goal_cid,dependency_task_cids=(),objective='Modify '+key,
        rationale='Independent potential work',scope_paths=(key+'.py',),
        outputs=(PromptOutputRecord(path=key+'.py',effect='modify',media_type='text/x-python'),),
        validations=(validations[i],),acceptance=(acceptance[i],),evidence_cids=(),policy_roots=(policy,),predicted_files=(key+'.py',))
        for i,key in enumerate(('calc','other')))
    roots={name:content_identity({'population-fixture':name,'text':text}) for name in ('request_cid','scan_cid','program_root')}
    graph=PromptGoalGraph(**roots,policy_roots=(policy,),goals=(root,child),tasks=tasks,evidence=())
    manifest=local.author_local_benchmark_manifest(repository=repository,profile_dir=state/'profile',lifecycle_dir=state/'lifecycle',
        task_specs=successor._specs(graph),planning_roots=roots)
    return local.admit_local_benchmark_plan(graph=graph,manifest=manifest)


def query(path='calc.py',offset=1):
    return dict(contract=population.IntegerOffsetContract(path,'increment','n',offset).to_dict(),inputs=[-1,0,1])


def declaration(profile,tools,text='agent must modify calc or agent must modify other.',*,proofs=False,guard=None):
    admission=universe(profile,text);ast=population.rich_grammar.parse_instruction(text);atoms=population._atoms(ast)
    rows=[]
    for path,atom in atoms.items():
        key=atom['object'];keys=['calc','other'] if len(atoms)==1 else [key]
        prohibited=atom['modality']=='prohibited'
        rows.append(dict(atom_path=path,atom=atom,task_keys=keys,
            proof_request=query('calc.py') if proofs and not prohibited else None,
            forbidden_outputs=[dict(path=k+'.py',effect='modify',media_type='text/x-python') for k in keys] if prohibited else []))
    arguments=dict(base_admission=admission,instruction=text,groundings=rows,
        guard_grounding=dict(guard=ast['guard'],proof_request=guard) if ast['kind']=='if' else None,
        allowed_task_sets=[[],['calc'],['other'],['calc','other']],review_ref='review:explicit-fixture',tool_policy=tools)
    return arguments,population.author_population_policy(**arguments)


def test_signed_alternative_retains_full_universe_and_independent_allowed_sets(profile,finite_tools):
    args,policy=declaration(profile,finite_tools)
    value,base,verified=population._load(policy)
    assert value['universe']==['calc','other'] and value['ast']['kind']=='or'
    decision=population._coverage(value,['calc'],{})
    assert decision['accepted'] and decision['complete_atom_paths']==['left','right']
    assert decision['requirements'][1]['disposition']=='not_selected_alternative_or_uncovered'
    assert all(decision[k] is False for k in population.FALSE)
    graph,mapping=population._project(verified['graph'],['calc'],content_identity(policy),type('Head',(),{'to_dict':lambda self:{'generation':1}})())
    assert len(graph.tasks)==1 and len(mapping)==1
    original=next(t for t in verified['graph'].tasks if t.task_key=='calc');new=graph.tasks[0]
    assert replace(new,task_key=original.task_key,goal_cid=original.goal_cid).to_dict()==original.to_dict()
    with pytest.raises(ValueError):replace(verified['graph'],tasks=())


@pytest.mark.parametrize('mutation',['source','atom','universe','producers','unknown','authority'])
def test_even_resigned_policy_cannot_drop_or_rewrite_declared_meaning(profile,finite_tools,mutation):
    args,policy=declaration(profile,finite_tools);value=deepcopy(policy['payload'])
    if mutation=='source':value['instruction']+=' ignored sentence.'
    elif mutation=='atom':value['groundings'][0]['atom']['modality']='permitted'
    elif mutation=='universe':value['universe']=['calc']
    elif mutation=='producers':value['producers']={}
    elif mutation=='unknown':value['claimed_coverage']=True
    else:value['execution_authority']=True
    verified=local.verify_local_benchmark_admission(args['base_admission'])
    signed=local._signed(value,verified['manifest'])
    with pytest.raises(ValueError):population._load(signed)


def test_unknown_conditional_guard_never_justifies_omission(profile,finite_tools):
    _,policy=declaration(profile,finite_tools,'if feature is enabled, agent must modify calc.')
    result=population._coverage(policy['payload'],[],{})
    assert not result['accepted'] and result['unresolved']==['guard']
    assert result['requirements'][0]['active'] is None
    assert result['requirements'][0]['disposition']=='unresolved_guard'


def test_prohibition_covers_all_grounded_future_effects(profile,finite_tools):
    _,policy=declaration(profile,finite_tools,'agent must not modify calc.')
    assert population._coverage(policy['payload'],[],{})['accepted']
    invalid=population._coverage(policy['payload'],['calc'],{})
    assert not invalid['accepted'] and invalid['prohibited_atom_paths']==['root']


@pytest.mark.parametrize('source',['agent must modify calc then agent must modify other.','agent may modify calc.'])
def test_temporal_or_permitted_truth_is_not_inferred_from_current_code(profile,finite_tools,source):
    with pytest.raises(ValueError):declaration(profile,finite_tools,source)


def test_nonagent_actor_cannot_inherit_native_agent_authorization(profile,finite_tools):
    with pytest.raises(ValueError,match='actor'):
        declaration(profile,finite_tools,'operator must modify calc.')


def test_selected_write_cannot_overwrite_a_property_used_to_elide_other_tasks(profile,finite_tools):
    _,policy=declaration(profile,finite_tools,proofs=True)
    value,_,verified=population._load(policy)
    evidence=dict(truth=True,status='positive',query=query(),source_head={'fixture':'only for pure coverage control'})
    coverage=population._coverage(value,['calc'],{'right':evidence})
    assert coverage['accepted']
    with pytest.raises(ValueError,match='proof-elided'):
        population._preserved_properties(coverage,verified['graph'],['calc'])
    safe=population._coverage(value,['other'],{'left':evidence})
    assert population._preserved_properties(safe,verified['graph'],['other'])[0]['path']=='calc.py'
    selected_all=population._coverage(value,['calc','other'],{'left':evidence,'right':evidence})
    assert population._preserved_properties(selected_all,verified['graph'],['calc','other'])==[]


@pytest.fixture
def owners(profile):
    """Actual owners; explicit controlled sampling is separately reportable."""
    import duckdb
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog,CodebaseHead
    from ipfs_datasets_py.duckdb_control.intent_codebase_catalog import IntentCodebaseCatalog
    from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
    from ipfs_datasets_py.logic.software_contracts.codebase_scan_policy import prepare_policy_current
    from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
    from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
    from ipfs_accelerate_py.agent_supervisor.proof.finite_checked_cache import FiniteCheckedCache
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import FormalVerificationCache
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    root,repository=profile
    scheduler=host_scheduler(root)
    with duckdb.connect(str(root/'source.duckdb'),config={'threads':1,'memory_limit':'64MB'}) as cx:
        store=DuckDBASTStore(connection=cx);artifacts=ImmutableCAS(root/'source-cas');artifacts.root.chmod(0o700)
        index=RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store),artifacts=artifacts,
            catalog=CodebaseCatalog(store,artifacts))
        receipt=prepare_policy_current(index,repository,repository_id='population:fixture',operation_id='population-source',
            expected_head=None,timeout_seconds=60,scheduler=scheduler)
        catalog=IntentCodebaseCatalog(index);cache=FiniteCheckedCache(FormalVerificationCache(root/'proof'),artifacts)
        Path(catalog.catalog._database_path).chmod(0o600);cache.cache.path.chmod(0o600)
        with IntentRepository(root/'intent.duckdb') as intent:
            Path(intent.database_path).chmod(0o600)
            yield dict(catalog=catalog,checked_cache=cache,expected_head=CodebaseHead.from_dict(receipt['head']),
                intent=intent,root=root,repository=repository,scheduler=scheduler)


def host_scheduler(root):
    if os.environ.get('RPI021_CONTROLLED_HOST')!='1':return None
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import GlobalResourceScheduler,ResourceSchedulerConfig
    return GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(state_path=root/'controlled-resources.json',
        proof_resource_sampler=lambda:ProofHostResources(8,8192,8192),lane_reservations={},auto_renew_leases=False))


def prepare(owners,policy,selected,state='population-state',**options):
    return population.prepare_population_admission(policy=policy,selected_task_keys=selected,
        **{k:owners[k] for k in ('catalog','checked_cache','expected_head','intent','scheduler')},
        state=owners['root']/state,**options)


def test_actual_choice_cannot_accumulate_disjoint_authorized_alternatives(profile,finite_tools,owners,monkeypatch):
    args,_=declaration(profile,finite_tools)
    args['allowed_task_sets']=[['calc'],['other']]
    policy=population.author_population_policy(**args)
    result=prepare(owners,policy,['calc'])
    assert result['status']=='population_admitted' and not result['decision_replayed']
    _require_no_original=lambda: all(owners['intent'].get_task(t.task_cid) is None
        for t in local.verify_local_benchmark_admission(args['base_admission'])['graph'].tasks)
    assert _require_no_original()
    def forbidden(*a,**k):pytest.fail('conflicting selection launched proof work')
    monkeypatch.setattr(owners['checked_cache'],'check_and_store',forbidden)
    with pytest.raises(ValueError,match='another task set'):
        prepare(owners,policy,['other'],state='alternative-state')
    with owners['intent']._connection(write=False) as cx:
        tasks=cx.execute('SELECT task_cid FROM tasks').fetchall()
    assert {row[0] for row in tasks}==set(result['materialized']['task_cids'])


def test_actual_lost_publication_reply_recovers_exact_native_population_and_decision(profile,finite_tools,owners,monkeypatch):
    _,policy=declaration(profile,finite_tools)
    with monkeypatch.context() as control:
        control.setattr(population,'_publish_decision',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('lost publication reply')))
        with pytest.raises(RuntimeError,match='lost publication reply'):prepare(owners,policy,['calc'])
    result=prepare(owners,policy,['calc'])
    assert result['decision_replayed'] and not result['saved_observations_used_as_proof']
    before={cid:dict(owners['intent'].get_task(cid)) for cid in result['materialized']['task_cids']}
    again=prepare(owners,policy,['calc'],state='new-retry-state')
    assert again['decision']==result['decision'] and again['materialized']==result['materialized']
    assert before=={cid:dict(owners['intent'].get_task(cid)) for cid in before}
    assert Path(again['decision_path']).is_file()


def test_actual_fresh_process_replays_file_backed_choice_without_new_task_revisions(profile,finite_tools,owners):
    _,policy=declaration(profile,finite_tools);before=prepare(owners,policy,['calc'])
    data=dict(policy=policy,roots=population.evidence_owner._roots(owners['catalog'],owners['checked_cache']),
        source_head=owners['expected_head'].to_dict(),intent_database=str(owners['intent'].database_path),root=str(owners['root']))
    owners['intent'].close();owners['catalog'].catalog.store._connection.close()
    script='''import json,sys
sys.path[:0]=json.loads(sys.argv[1])
from pathlib import Path
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
from ipfs_accelerate_py.agent_supervisor.runtime import intent_population_admission as owner
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from test.integration.test_intent_population_admission import host_scheduler
r=json.loads(sys.argv[2]);root=Path(r['root'])
with IntentRepository(r['intent_database']) as intent:
 with owner.evidence_owner._owners(r['roots']) as (catalog,cache):
  result=owner.prepare_population_admission(policy=r['policy'],selected_task_keys=['calc'],catalog=catalog,checked_cache=cache,
   expected_head=CodebaseHead.from_dict(r['source_head']),intent=intent,state=root/'population-state',scheduler=host_scheduler(root))
  assert all(intent.get_task(cid)['revision']==1 for cid in result['materialized']['task_cids'])
  print('POPULATION_REPLAY='+json.dumps(dict(decision_sha256=result['decision_sha256'],materialized=result['materialized'],replayed=result['decision_replayed'])))
'''
    roots=[str(Path(population.__file__).parents[3]),'/tmp/ir-release-datasets-20261001',site.getusersitepackages()]
    child=subprocess.run([sys.executable,'-I','-c',script,json.dumps(roots),json.dumps(data)],capture_output=True,text=True,timeout=90)
    assert child.returncode==0,child.stderr
    after=json.loads(next(line.split('=',1)[1] for line in child.stdout.splitlines() if line.startswith('POPULATION_REPLAY=')))
    assert after==dict(decision_sha256=before['decision_sha256'],materialized=before['materialized'],replayed=True)


def test_actual_choice_transaction_rolls_back_tasks_if_record_publication_fails(profile,finite_tools,owners,monkeypatch):
    _,policy=declaration(profile,finite_tools)
    actual=population.IntentRepository.upsert_plan
    def failed(owner,**kw):
        if kw['body'].get('schema')=='reviewed-task-population-choice-record@1':
            raise RuntimeError('choice record failure')
        return actual(owner,**kw)
    with monkeypatch.context() as control:
        control.setattr(population.IntentRepository,'upsert_plan',failed)
        with pytest.raises(RuntimeError,match='choice record failure'):prepare(owners,policy,['calc'])
    with owners['intent']._connection(write=False) as cx:
        assert cx.execute('SELECT count(*) FROM tasks').fetchone()[0]==0
        assert cx.execute('SELECT count(*) FROM plans').fetchone()[0]==0
        assert cx.execute('SELECT count(*) FROM goals').fetchone()[0]==0
    assert prepare(owners,policy,['calc'])['status']=='population_admitted'


@pytest.mark.parametrize('mutation',['task','choice_goal','choice_plan','choice_body'])
def test_actual_changed_native_choice_or_selected_population_refuses_replay(profile,finite_tools,owners,mutation):
    _,policy=declaration(profile,finite_tools);result=prepare(owners,policy,['calc'])
    with owners['intent']._connection(write=True) as cx:
        if mutation=='task':cx.execute('UPDATE tasks SET revision=revision+1 WHERE task_cid=?',[result['materialized']['task_cids'][0]])
        elif mutation=='choice_goal':cx.execute('UPDATE goals SET revision=revision+1 WHERE goal_cid=?',[result['choice']['goal_cid']])
        elif mutation=='choice_plan':cx.execute('UPDATE plans SET status=? WHERE plan_cid=?',['active',result['choice']['plan_cid']])
        else:cx.execute('UPDATE plans SET body_json=? WHERE plan_cid=?',['{}',result['choice']['plan_cid']])
    with pytest.raises(ValueError):prepare(owners,policy,['calc'])


def test_actual_native_proof_zero_edit_and_retry_keep_full_universe(profile,finite_tools,owners):
    _,policy=declaration(profile,finite_tools,proofs=True)
    result=prepare(owners,policy,[])
    again=prepare(owners,policy,[])
    assert result['status']=='zero_edit_admitted' and again['decision_replayed']
    assert again['decision']==result['decision'] and again['materialized']['task_cids']==[]
    payload=result['decision']['payload']
    assert payload['policy']['payload']['universe']==['calc','other']
    assert payload['omitted_task_keys']==['calc','other'] and len(payload['coverage']['requirements'])==2
    assert all(v['fresh_native_observation'] and v['truth'] for v in again['fresh_observations'].values())
    with owners['intent']._connection(write=False) as cx:
        assert cx.execute('SELECT count(*) FROM tasks').fetchone()[0]==0
        assert [row[0] for row in cx.execute("SELECT status FROM plans").fetchall()]==['proposed']
    assert all(payload[k] is False for k in population.FALSE)


def test_actual_native_proof_elision_cannot_authorize_overwriting_its_source(profile,finite_tools,owners):
    _,policy=declaration(profile,finite_tools,proofs=True)
    with pytest.raises(ValueError,match='proof-elided'):prepare(owners,policy,['calc'])
    with owners['intent']._connection(write=False) as cx:assert cx.execute('SELECT count(*) FROM tasks').fetchone()[0]==0
    safe=prepare(owners,policy,['other'],state='preserved-property-state')
    assert safe['decision']['payload']['protected_property_sources'][0]['path']=='calc.py'


@pytest.mark.parametrize('offset,selected,truth',[(2,[],False),(1,['calc','other'],True)])
def test_actual_conditional_guard_binds_finite_initial_source_truth(profile,finite_tools,owners,offset,selected,truth):
    _,policy=declaration(profile,finite_tools,'if feature is enabled, agent must modify calc.',guard=query(offset=offset))
    result=prepare(owners,policy,selected)
    payload=result['decision']['payload']
    assert payload['coverage']['guard_truth'] is truth
    assert payload['coverage']['requirements'][0]['active'] is truth
    assert payload['policy']['payload']['guard_scope']=='signed_finite_property_at_initial_source_snapshot'
    assert len(result['materialized']['task_cids'])==len(selected)
    assert result['fresh_observations']['guard']['fresh_native_observation']


def test_actual_prohibition_rejects_effect_and_admits_inert_zero_population(profile,finite_tools,owners):
    _,policy=declaration(profile,finite_tools,'agent must not modify calc.')
    denied=prepare(owners,policy,['calc'])
    assert denied['status']=='unresolved_population'
    assert denied['decision']['coverage']['prohibited_atom_paths']==['root']
    allowed=prepare(owners,policy,[],state='zero-effect-state')
    assert allowed['status']=='zero_edit_admitted' and allowed['materialized']['task_cids']==[]
    with owners['intent']._connection(write=False) as cx:assert cx.execute('SELECT count(*) FROM tasks').fetchone()[0]==0


def test_actual_public_artifact_staging_interruption_leaves_no_partial_public_name(profile,finite_tools,owners,monkeypatch):
    _,policy=declaration(profile,finite_tools);actual=population.finite._persist
    def interrupt(path,body):
        actual(path,body)
        if path.name.startswith('.population-'):raise RuntimeError('interrupted after complete artifact staging')
    with monkeypatch.context() as control:
        control.setattr(population.finite,'_persist',interrupt)
        with pytest.raises(RuntimeError,match='artifact staging'):prepare(owners,policy,['calc'])
    directory=owners['repository']/'.runtime'/'repository-finite-handoffs'
    assert not list(directory.glob('*.json'))
    result=prepare(owners,policy,['calc'])
    assert result['decision_replayed'] and Path(result['decision_path']).is_file()
    assert population.finite._wire(result['decision'])==Path(result['decision_path']).read_bytes()
