"""Actual durable owners and finite checkers behind bounded local evidence queries."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis import repository_code_evidence as evidence
from ipfs_accelerate_py.agent_supervisor.proof.finite_checked_cache import FiniteCheckedCache
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import FormalVerificationCache
from ipfs_accelerate_py.agent_supervisor.planning import finite_integer_codebase as finite
from ipfs_datasets_py.duckdb_control.intent_codebase_catalog import IntentCodebaseCatalog
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
from ipfs_datasets_py.logic.software_contracts.codebase_semantic_manifest import build_codebase_semantic_manifest
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from test.integration.test_terminal_codebase_semantic_index import prepared
from test.api.test_finite_integer_codebase import finite_tools, finite_text


@pytest.fixture
def owner(prepared, tmp_path):
    index, repository, head, descriptor = prepared
    catalog = IntentCodebaseCatalog(index)
    catalog.publish(repository, expected_head=head, manifest_cid=descriptor['manifest_cid'], operation_id='query-inventory')
    cache = FiniteCheckedCache(FormalVerificationCache(tmp_path/'proof'), index.artifacts)
    plane = evidence.RepositoryCodeEvidence(catalog, cache)
    return plane, dict(repository=repository, expected_head=head, semantic_manifest_cid=descriptor['manifest_cid'])


def check(owner, tools, offset=2):
    plane, args = owner
    contract = IntegerOffsetContract('calc.py','increment','n',offset)
    return plane.checked_cache.check_and_store(owner_inputs=dict(index=plane.index,
        repository=args['repository'], expected_head=args['expected_head'], contract=contract,
        inputs=[-2,-1,0,1,2], tool_policy=tools))


def positive(owner):
    plane, args = owner
    manifest = plane.index.artifacts.get(args['semantic_manifest_cid'])
    result = build_codebase_semantic_manifest(plane.index, policy_receipt_cid=manifest['policy_receipt_cid'],
        contracts=(IntegerOffsetContract('calc.py','increment','n',1),))
    args['semantic_manifest_cid'] = result['manifest_cid']
    plane.catalog.publish(args['repository'], expected_head=args['expected_head'],
        manifest_cid=result['manifest_cid'], operation_id='positive-declaration')
    return owner


def test_actual_ast_inventory_pages_bind_every_consumed_row(owner):
    plane, args = owner
    query = evidence.RelevantUnitsQuery(page_size=2)
    first = plane.relevant_units(query=query, **args)
    second = plane.relevant_units(query=query, cursor=first['next_cursor'], **args)
    assert first['total_rows']==second['total_rows']==4
    assert first['root']==second['root'] and first['offset']==0 and second['offset']==2
    assert first['selection_complete'] is second['selection_complete'] is False
    assert not first['page_is_last'] and second['page_is_last']
    allrows=first['rows']+second['rows']
    assert {r['path'] for r in allrows}=={'.gitignore','README.md','calc.py','unsupported.py'}
    assert first['consumed_row_cids']+second['consumed_row_cids']==[cid_for_structured(r) for r in allrows]
    source=next(r for r in allrows if r['path']=='calc.py')
    assert source['ast_summary']['source_sha256'] and source['ast_cid']
    assert source['ast_revision']==args['expected_head'].ast_revision_id
    assert source['model_status']=='source_bound_model'
    assert all(not row['semantic_relevance_inferred'] and not row['proof_authority'] for row in allrows)
    assert plane.relevant_units(query=evidence.RelevantUnitsQuery(),**args)['selection_complete']


@pytest.mark.parametrize('change',['root','prefix_cid','offset','query','extra'])
def test_resealed_or_incomplete_cursor_refuses(owner, change):
    plane,args=owner
    query=evidence.RelevantUnitsQuery(page_size=2)
    cursor=plane.relevant_units(query=query,**args)['next_cursor']
    if change=='root':cursor['root']=args['semantic_manifest_cid']
    elif change=='prefix_cid':cursor['prefix_cid']=args['semantic_manifest_cid']
    elif change=='offset':cursor['offset']=3
    elif change=='query':cursor['query']['page_size']=4
    else:cursor['assume_consumed']=True
    with pytest.raises(ValueError,match='cursor'):
        plane.relevant_units(query=query,cursor=cursor,**args)


@pytest.mark.parametrize('bad',[dict(paths=['calc.py']),dict(paths=('../calc.py',)),dict(paths=('calc.py','calc.py')),
    dict(page_size=True),dict(page_size=65)])
def test_closed_relevant_query(bad):
    with pytest.raises(ValueError):evidence.RelevantUnitsQuery(**bad)


@pytest.mark.parametrize('bad',[(),(True,), (2,1),(1,1),(2**32,)])
def test_closed_explicit_finite_domain(bad):
    with pytest.raises(ValueError):evidence.UnitEvidenceQuery('calc.py',bad)


def test_query_domain_matches_native_thirty_two_case_bound():
    assert len(evidence.UnitEvidenceQuery('calc.py',tuple(range(32))).inputs)==32
    with pytest.raises(ValueError):evidence.UnitEvidenceQuery('calc.py',tuple(range(33)))


def test_unknown_path_and_dirty_source_cannot_hide_inventory(owner):
    plane,args=owner
    with pytest.raises(ValueError,match='outside'):
        plane.relevant_units(query=evidence.RelevantUnitsQuery(('missing.py',)),**args)
    first=plane.relevant_units(query=evidence.RelevantUnitsQuery(page_size=2),**args)
    commit=subprocess.check_output(['git','-C',str(args['repository']),'rev-parse','HEAD'])
    (args['repository']/'calc.py').write_text('def increment(n: int) -> int:\n    return n + 2\n')
    assert subprocess.check_output(['git','-C',str(args['repository']),'rev-parse','HEAD'])==commit
    with pytest.raises(ValueError):
        plane.relevant_units(query=evidence.RelevantUnitsQuery(page_size=2),cursor=first['next_cursor'],**args)


def test_inactive_parser_producer_drift_refuses_cursor(owner, monkeypatch):
    plane,args=owner
    query=evidence.RelevantUnitsQuery(page_size=2)
    first=plane.relevant_units(query=query,**args)
    original=evidence._pins
    monkeypatch.setattr(evidence,'_pins',lambda:{**original(),'declared-parser':'different'})
    with pytest.raises(ValueError,match='cursor'):
        plane.relevant_units(query=query,cursor=first['next_cursor'],**args)


def test_missing_selector_or_captured_body_refuses_empty_success(owner):
    plane,args=owner
    plane.catalog._cx.execute('DELETE FROM intent_codebase.selectors')
    with pytest.raises(ValueError,match='selector'):
        plane.relevant_units(query=evidence.RelevantUnitsQuery(),**args)


def test_real_negative_history_and_reverse_dependency_projection(owner, finite_tools):
    plane,args=owner
    stored=check(owner,finite_tools)
    query=evidence.UnitEvidenceQuery('calc.py',(-2,-1,0,1,2),64)
    page=plane.unit_evidence(query=query,tool_policy=finite_tools,**args)
    assert page['cache_status']=='refuted' and not page['positive_reuse_eligible']
    assert not page['fresh_native_observation'] and page['commitments']['cache']['record_cid']==stored['record_cid']
    assert {'source','expression','formalization','slice','obligation','assumptions','bounds','translation',
        'provider','environment','policy','schema','checker','network_policy','evidence_kind','authority_ceiling'} <= set(page['commitments']['canonical_key'])
    assert page['commitments']['reverse_references'][0]['historical_only']
    assert page['reverse_source_dependencies']['historical_only']
    assert not page['reverse_source_dependencies']['truncated']
    assert all(not r['value']['authoritative'] for r in page['rows'])
    assert any(e['kind']=='historically_depended_on' for e in page['reverse_source_dependencies']['edges'])


def test_actual_positive_cache_requires_fresh_native_checker_on_each_page(owner, finite_tools, monkeypatch):
    positive(owner)
    plane,args=owner
    stored=check(owner,finite_tools,1)
    from ipfs_accelerate_py.agent_supervisor.proof import finite_checked_cache as cache_owner
    original=cache_owner.native.observe_finite_integer_source
    calls=[]
    def fresh(**kwargs):calls.append(True);return original(**kwargs)
    monkeypatch.setattr(cache_owner.native,'observe_finite_integer_source',fresh)
    query=evidence.UnitEvidenceQuery('calc.py',(-2,-1,0,1,2),64)
    first=plane.unit_evidence(query=query,tool_policy=finite_tools,**args)
    second=plane.unit_evidence(query=query,tool_policy=finite_tools,cursor=first['next_cursor'],**args)
    assert len(calls)==2
    assert first['positive_reuse_eligible'] and second['positive_reuse_eligible']
    assert first['fresh_native_observation'] and first['positive_scope']=='exact_generated_finite_integer_table_theorems_only'
    assert first['commitments']['receipt_id'] and first['root']==second['root']
    assert first['proof_authority'] is first['source_equivalence_proved'] is False
    assert all(not r['value']['authoritative'] for r in first['rows']+second['rows'])
    bundle=plane.index.artifacts.get(stored['evidence']['bundle_cid'])
    plane.index.artifacts.path_for(bundle['body_cids']['lean_olean'],source=True).unlink()
    with pytest.raises((ValueError,OSError)):
        plane.unit_evidence(query=query,tool_policy=finite_tools,**args)


def test_model_off_ignored_asset_change_keeps_key_but_domain_or_receipt_revocation_does_not(owner,finite_tools):
    positive(owner)
    plane,args=owner
    stored=check(owner,finite_tools,1)
    ignored=args['repository']/'.runtime/model-version.txt'
    ignored.parent.mkdir();ignored.write_text('unconsumed-model-version-two')
    query=evidence.UnitEvidenceQuery('calc.py',(-2,-1,0,1,2),64)
    result=plane.unit_evidence(query=query,tool_policy=finite_tools,**args)
    assert result['commitments']['cache']['record_cid']==stored['record_cid']
    assert result['positive_reuse_eligible']
    missed=plane.unit_evidence(query=evidence.UnitEvidenceQuery('calc.py',(-1,0,1)),tool_policy=finite_tools,**args)
    assert missed['cache_status']=='miss' and not missed['positive_reuse_eligible']
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import ProofCacheKey
    assert plane.checked_cache.cache.delete(ProofCacheKey.from_dict(stored['evidence']['formal_key']))
    with pytest.raises(ValueError,match='reconstruction'):
        plane.unit_evidence(query=query,tool_policy=finite_tools,**args)


def test_formatting_recapture_changes_exact_spans_and_rejects_old_page(owner):
    plane,args=owner
    query=evidence.RelevantUnitsQuery(page_size=2)
    first=plane.relevant_units(query=query,**args)
    old=plane.relevant_units(query=evidence.RelevantUnitsQuery(('calc.py',)),**args)['rows'][0]
    (args['repository']/'calc.py').write_text('# source-only formatting\n\ndef increment(n: int) -> int:\n    return n + 1\n')
    from benchmarks.agent_supervisor.container_coding.terminal_codebase_semantic_index import prepare_semantic_index
    newhead,desc=prepare_semantic_index(index=plane.index,repository=args['repository'],
        repository_id=args['expected_head'].repository_id,expected_head=args['expected_head'],
        operation_id='format-recapture',contract=IntegerOffsetContract('calc.py','increment','n',2))
    args.update(expected_head=newhead,semantic_manifest_cid=desc['manifest_cid'])
    plane.catalog.publish(args['repository'],expected_head=newhead,manifest_cid=desc['manifest_cid'],operation_id='format-discovery')
    with pytest.raises(ValueError,match='cursor'):
        plane.relevant_units(query=query,cursor=first['next_cursor'],**args)
    new=plane.relevant_units(query=evidence.RelevantUnitsQuery(('calc.py',)),**args)['rows'][0]
    assert new['source_cid']!=old['source_cid'] and new['ast_cid']!=old['ast_cid']
    assert new['ast_summary']!=old['ast_summary']


def test_missing_cache_never_synthesizes_positive_or_runs_checker(owner,finite_tools,monkeypatch):
    plane,args=owner
    from ipfs_accelerate_py.agent_supervisor.proof import finite_checked_cache as cache_owner
    monkeypatch.setattr(cache_owner.native,'observe_finite_integer_source',lambda **kw:pytest.fail('missing cache ran checker'))
    result=plane.unit_evidence(query=evidence.UnitEvidenceQuery('calc.py',(-1,0,1)),tool_policy=finite_tools,**args)
    assert result['cache_status']=='miss' and not result['positive_reuse_eligible']
    unsupported=plane.unit_evidence(query=evidence.UnitEvidenceQuery('unsupported.py',(-1,0,1)),tool_policy=finite_tools,**args)
    assert unsupported['cache_status'] is None and unsupported['unit_disposition']!='source_bound_model'


def test_intent_resolution_preserves_exact_residual_and_all_clause_ids(owner,finite_tools,tmp_path):
    plane,args=owner
    text=finite_text()
    result=plane.resolve_intent(**args,intent_document=finite.build_finite_integer_intent(text),
        source_text=text,output=tmp_path/'intent',tool_policy=finite_tools)
    assert result['complete_requirement_ids']==sorted([finite.TYPE_STATEMENT_ID,finite.OFFSET_STATEMENT_ID])
    assert [r['statement_id'] for r in result['residual_obligations']]==[finite.OFFSET_STATEMENT_ID]
    assert result['consumed_record_commitments']['checked_cache']['status']=='refuted'
    assert result['selection_complete'] and not result['reduced_task_population_authorized']


def test_local_plane_cannot_claim_dqp_release():
    plane=evidence.LocalCodeEvidencePlane('revision',(),())
    assert not plane.to_dict()['dqp_release_verified']
    with pytest.raises(ValueError,match='DQP'):plane.project_dqp()


def test_cold_process_rebuilds_identical_ast_page_without_python_cache(owner,tmp_path):
    plane,args=owner
    query=evidence.RelevantUnitsQuery(page_size=2)
    first=plane.relevant_units(query=query,**args)
    request=dict(repository=str(args['repository']),head=args['expected_head'].to_dict(),
        semantic_manifest_cid=args['semantic_manifest_cid'],source_database=str(tmp_path/'source.duckdb'),
        cas=str(tmp_path/'cas'),proof=str(tmp_path/'proof'),cursor=first['next_cursor'])
    path=tmp_path/'replay.json';path.write_text(json.dumps(request))
    plane.catalog._cx.close()
    script='''
import json,sys,duckdb
from pathlib import Path
from benchmarks.agent_supervisor.container_coding.native_repository_finite_supervision import _index
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
from ipfs_datasets_py.duckdb_control.intent_codebase_catalog import IntentCodebaseCatalog
from ipfs_accelerate_py.agent_supervisor.analysis.repository_code_evidence import *
from ipfs_accelerate_py.agent_supervisor.proof.finite_checked_cache import FiniteCheckedCache
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import FormalVerificationCache
v=json.loads(Path(sys.argv[1]).read_text())
cx=duckdb.connect(v['source_database'],config={'threads':1,'memory_limit':'64MB'})
index=_index(cx,Path(v['cas']))
plane=RepositoryCodeEvidence(IntentCodebaseCatalog(index),FiniteCheckedCache(FormalVerificationCache(v['proof']),index.artifacts))
result=plane.relevant_units(query=RelevantUnitsQuery(page_size=2),repository=Path(v['repository']),expected_head=CodebaseHead.from_dict(v['head']),semantic_manifest_cid=v['semantic_manifest_cid'],cursor=v['cursor'])
print(json.dumps(result));cx.close()
'''
    result=subprocess.run([sys.executable,'-c',script,str(path)],capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stderr
    second=json.loads(result.stdout)
    assert first['root']==second['root'] and second['offset']==2 and second['page_is_last']
    assert not second['selection_complete']
