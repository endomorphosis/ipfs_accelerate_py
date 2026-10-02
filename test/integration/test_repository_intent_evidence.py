"""Closed query populations, real native proof lookup, and cold reconstruction."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis import repository_intent_evidence as typed
from ipfs_accelerate_py.agent_supervisor.analysis.repository_code_evidence import UnitEvidenceQuery
from ipfs_accelerate_py.agent_supervisor.planning import finite_integer_codebase as finite
from ipfs_accelerate_py.agent_supervisor.proof import finite_checked_cache as cache_owner
from test.integration.test_repository_code_evidence import prepared, check, positive
from ipfs_datasets_py.duckdb_control.intent_codebase_catalog import IntentCodebaseCatalog
from ipfs_accelerate_py.agent_supervisor.analysis.repository_code_evidence import RepositoryCodeEvidence
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import FormalVerificationCache
from test.api.test_finite_integer_codebase import finite_tools, finite_text


@pytest.fixture(scope='module')
def owner(tmp_path_factory):
    retained=os.environ.get('RPI_INTENT_QUALIFICATION_FIXTURE')
    if retained:
        from ipfs_accelerate_py.agent_supervisor.runtime import repository_behavioral_admission as gate
        saved=json.loads((Path(retained)/'native-test-fixture.json').read_text())
        payload=json.loads(Path(saved['controls']['artifact']).read_text())['payload']
        with gate._owners(payload['owner_roots']) as (catalog,cache):
            from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
            yield RepositoryCodeEvidence(catalog,cache),dict(repository=Path(payload['repository']),
                expected_head=CodebaseHead.from_dict(payload['source_head']),semantic_manifest_cid=payload['semantic_manifest_cid'])
        return
    root=tmp_path_factory.mktemp('typed-intent-evidence')
    generator=prepared.__wrapped__(root)
    index,repository,head,descriptor=next(generator)
    catalog=IntentCodebaseCatalog(index)
    catalog.publish(repository,expected_head=head,manifest_cid=descriptor['manifest_cid'],operation_id='typed-query-inventory')
    cache=cache_owner.FiniteCheckedCache(FormalVerificationCache(root/'proof'),index.artifacts)
    yield RepositoryCodeEvidence(catalog,cache),dict(repository=repository,expected_head=head,semantic_manifest_cid=descriptor['manifest_cid'])
    try:next(generator)
    except StopIteration:pass


def query(kind, tools, **changes):
    values=dict(source_text=finite_text(), intent_document=finite.build_finite_integer_intent(finite_text()), tool_policy=tools)
    values.update(changes)
    return kind.from_native(**values)


@pytest.mark.parametrize('change',['empty','large','duplicate_json','extra','noncanonical','bool_bound','overflow','empty_population'])
def test_closed_query_refuses_ambiguous_unbounded_inputs(finite_tools,change):
    good=query(typed.IntentResolutionQuery,finite_tools)
    fields={k:v for k,v in good.to_dict().items() if k!='kind'}
    if change=='empty': fields['source_text']=''
    elif change=='large':fields['source_text']='a'*65537
    elif change=='duplicate_json':fields['tool_policy_json']='{"a":1,"a":1}'
    elif change=='extra':fields['intent_document_json']=typed._wire({**json.loads(fields['intent_document_json']),'fake_fact':True})
    elif change=='noncanonical':fields['intent_document_json']=' '+fields['intent_document_json']
    elif change=='bool_bound':fields['max_requirements']=True
    elif change=='overflow':fields['max_requirements']=17
    else:fields['max_requirements']=0
    with pytest.raises((ValueError,TypeError)):typed.IntentResolutionQuery(**fields)


def test_actual_typed_intent_and_residual_queries_preserve_all_requirements(owner,finite_tools,tmp_path):
    plane,args=owner;facade=typed.RepositoryIntentEvidence(plane)
    complete=facade.resolve_intent(query=query(typed.IntentResolutionQuery,finite_tools),output=tmp_path/'complete',**args)
    residual=facade.residual_obligations(query=query(typed.ResidualObligationsQuery,finite_tools),output=tmp_path/'residual',**args)
    expected=[finite.OFFSET_STATEMENT_ID,finite.TYPE_STATEMENT_ID]
    assert complete['complete_requirement_ids']==residual['complete_requirement_ids']==sorted(expected)
    assert len(complete['rows'])==2 and [row['statement_id'] for row in residual['rows']]==[finite.OFFSET_STATEMENT_ID]
    assert complete['selection_complete'] and residual['selection_complete']
    assert complete['next_cursor'] is residual['next_cursor'] is None
    assert complete['root']!=residual['root']
    assert not residual['reduced_task_population_authorized']
    assert residual['complete_native_result']['match']['fresh_match']['observation']['kernel_checked_model_table']
    assert residual['root_material']['consumed_record_commitments']['discovery_record_cids']
    with pytest.raises(ValueError,match='query type'):
        facade.resolve_intent(query=query(typed.ResidualObligationsQuery,finite_tools),output=tmp_path/'wrong-type',**args)
    with pytest.raises(ValueError,match='population'):
        facade.resolve_intent(query=query(typed.IntentResolutionQuery,finite_tools,max_requirements=1),output=tmp_path/'overflow',**args)


def test_actual_wrong_premise_and_tool_cannot_reuse_positive(owner,finite_tools):
    positive(owner);plane,args=owner;saved=check(owner,finite_tools,1)
    q=UnitEvidenceQuery('calc.py',(-2,-1,0,1,2),64)
    alternate=cache_owner.native.seal_finite_integer_tools(python_executable=Path('/usr/bin/false'),lean_executable=Path(finite_tools['lean']['path']))
    missing=plane.unit_evidence(query=q,tool_policy=alternate,**args)
    assert missing['cache_status']=='miss' and not missing['positive_reuse_eligible']
    forged=deepcopy(saved['evidence'])
    forged['correspondence']['materials']['assumptions'][0]['text']='Unreviewed stronger premise'
    forged_cid=plane.index.artifacts.put(forged)
    with plane.checked_cache._transaction() as cx:
        cx.execute('UPDATE finite_checked_cache_records SET record_cid=? WHERE request_key=?',[forged_cid,saved['request_key']])
    try:
        with pytest.raises(ValueError):plane.unit_evidence(query=q,tool_policy=finite_tools,**args)
    finally:
        with plane.checked_cache._transaction() as cx:
            cx.execute('UPDATE finite_checked_cache_records SET record_cid=? WHERE request_key=?',[saved['record_cid'],saved['request_key']])


def test_fresh_process_positive_query_rechecks_native_body_and_preserves_key(owner,finite_tools,tmp_path):
    positive(owner);plane,args=owner;saved=check(owner,finite_tools,1)
    request=dict(repository=str(args['repository']),head=args['expected_head'].to_dict(),
        semantic_manifest_cid=args['semantic_manifest_cid'],source_database=str(plane.catalog.catalog._database_path),
        cas=str(plane.index.artifacts.root),proof=str(plane.checked_cache.cache.path.parent),tools=finite_tools)
    path=tmp_path/'cold-request.json';path.write_text(json.dumps(request));plane.catalog._cx.close()
    script='''
import json,sys,duckdb
from pathlib import Path
from benchmarks.agent_supervisor.container_coding.native_repository_finite_supervision import _index
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
from ipfs_datasets_py.duckdb_control.intent_codebase_catalog import IntentCodebaseCatalog
from ipfs_accelerate_py.agent_supervisor.analysis.repository_code_evidence import RepositoryCodeEvidence,UnitEvidenceQuery
from ipfs_accelerate_py.agent_supervisor.proof.finite_checked_cache import FiniteCheckedCache
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import FormalVerificationCache
v=json.loads(Path(sys.argv[1]).read_text())
cx=duckdb.connect(v['source_database'],config={'threads':1,'memory_limit':'64MB'})
index=_index(cx,Path(v['cas']))
plane=RepositoryCodeEvidence(IntentCodebaseCatalog(index),FiniteCheckedCache(FormalVerificationCache(v['proof']),index.artifacts))
result=plane.unit_evidence(query=UnitEvidenceQuery('calc.py',(-2,-1,0,1,2),64),repository=Path(v['repository']),expected_head=CodebaseHead.from_dict(v['head']),semantic_manifest_cid=v['semantic_manifest_cid'],tool_policy=v['tools'])
print(json.dumps(result));cx.close()
'''
    replay=subprocess.run([sys.executable,'-c',script,str(path)],capture_output=True,text=True,timeout=180)
    assert replay.returncode==0,replay.stderr
    result=json.loads(replay.stdout)
    assert result['positive_reuse_eligible'] and result['fresh_native_observation']
    assert result['commitments']['cache']['record_cid']==saved['record_cid']
    assert result['commitments']['canonical_key']==saved['evidence']['correspondence']['canonical_key']
    assert result['commitments']['receipt_id'] and not result['proof_authority']
