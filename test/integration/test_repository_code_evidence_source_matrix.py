"""Join actual source changes to the public bounded evidence queries."""
import subprocess

import pytest

from test.integration.test_repository_code_evidence import owner, prepared, finite_tools
from ipfs_accelerate_py.agent_supervisor.analysis.repository_code_evidence import RelevantUnitsQuery, UnitEvidenceQuery
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
from benchmarks.agent_supervisor.container_coding.terminal_codebase_semantic_index import prepare_semantic_index


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.DEVNULL).decode().strip()


def refresh(plane,args,name,*,path='calc.py',repository=None,repository_id=None):
    repository=repository or args['repository']
    repository_id=repository_id or args['expected_head'].repository_id
    head,descriptor=prepare_semantic_index(index=plane.index,repository=repository,
        repository_id=repository_id,expected_head=plane.index.current(repository_id),operation_id=name,
        contract=IntegerOffsetContract(path,'increment','n',2))
    plane.catalog.publish(repository,expected_head=head,manifest_cid=descriptor['manifest_cid'],operation_id=name+':discovery')
    return dict(repository=repository,expected_head=head,semantic_manifest_cid=descriptor['manifest_cid'])


def test_two_dirty_successors_at_same_git_head_never_reuse_old_rows(owner):
    plane,args=owner
    original_head=git(args['repository'],'rev-parse','HEAD')
    before=plane.relevant_units(query=RelevantUnitsQuery(('calc.py',)),**args)
    for offset in (2,3):
        (args['repository']/'calc.py').write_text('def increment(n: int) -> int:\n    return n + '+str(offset)+'\n')
        changed=refresh(plane,args,'dirty:'+str(offset))
        assert git(args['repository'],'rev-parse','HEAD')==original_head
        with pytest.raises(ValueError):plane.relevant_units(query=RelevantUnitsQuery(),**args)
        after=plane.relevant_units(query=RelevantUnitsQuery(('calc.py',)),**changed)
        assert after['rows'][0]['source_cid']!=before['rows'][0]['source_cid']
        assert after['rows'][0]['ast_cid']!=before['rows'][0]['ast_cid']
        args,before=changed,after


def test_rename_changes_logical_identity_and_old_path_is_not_satisfied(owner,finite_tools):
    plane,args=owner
    before=plane.relevant_units(query=RelevantUnitsQuery(('calc.py',)),**args)['rows'][0]
    git(args['repository'],'mv','calc.py','renamed.py')
    new=refresh(plane,args,'renamed',path='renamed.py')
    after=plane.relevant_units(query=RelevantUnitsQuery(('renamed.py',)),**new)['rows'][0]
    assert after['logical_unit_id']!=before['logical_unit_id']
    # The complete dirty inventory preserves the removed tracked path as an
    # opaque entry until commit; it must not silently reuse its former AST.
    removed=plane.relevant_units(query=RelevantUnitsQuery(('calc.py',)),**new)['rows'][0]
    assert removed['model_status']=='opaque' and removed['ast_cid'] is None and removed['source_cid'] is None
    old_evidence=plane.unit_evidence(query=UnitEvidenceQuery('calc.py',(-1,0,1)),tool_policy=finite_tools,**new)
    assert old_evidence['cache_status'] is None and not old_evidence['positive_reuse_eligible']
    result=plane.unit_evidence(query=UnitEvidenceQuery('renamed.py',(-1,0,1)),tool_policy=finite_tools,**new)
    assert result['cache_status']=='miss' and not result['positive_reuse_eligible']


def test_parallel_worktrees_keep_distinct_current_rows(owner,tmp_path):
    plane,args=owner
    second=tmp_path/'parallel-worktree'
    git(args['repository'],'worktree','add','--detach',str(second),'HEAD')
    try:
        alternate=refresh(plane,args,'parallel',repository=second,repository_id=args['expected_head'].repository_id+':parallel')
        unchanged=plane.relevant_units(query=RelevantUnitsQuery(('calc.py',)),**alternate)
        (args['repository']/'calc.py').write_text('def increment(n: int) -> int:\n    return n + 3\n')
        changed=refresh(plane,args,'first-only')
        result=plane.relevant_units(query=RelevantUnitsQuery(('calc.py',)),**changed)
        assert result['rows'][0]['ast_cid']!=unchanged['rows'][0]['ast_cid']
        assert plane.relevant_units(query=RelevantUnitsQuery(('calc.py',)),**alternate)==unchanged
    finally:git(args['repository'],'worktree','remove','--force',str(second))


def test_real_submodule_is_retained_as_opaque_and_cannot_supply_ast_or_proof(owner,tmp_path,finite_tools):
    plane,args=owner
    child=tmp_path/'nested';child.mkdir()
    (child/'nested.py').write_text('def nested():\n    return 1\n')
    git(child,'init','-q');git(child,'add','.')
    git(child,'-c','user.name=Evidence fixture','-c','user.email=evidence@example.invalid','commit','-qm','nested source')
    git(args['repository'],'-c','protocol.file.allow=always','submodule','add','-q',str(child),'component')
    new=refresh(plane,args,'submodule')
    rows=plane.relevant_units(query=RelevantUnitsQuery(),**new)['rows']
    row=next(r for r in rows if r['path']=='component')
    assert row['ast_cid'] is None and row['source_cid'] is None
    assert row['model_status']!='source_bound_model'
    assert all(r['path']!='component/nested.py' for r in rows)
    result=plane.unit_evidence(query=UnitEvidenceQuery('component',(-1,0,1)),tool_policy=finite_tools,**new)
    assert result['cache_status'] is None and not result['positive_reuse_eligible']


def test_source_change_after_row_construction_refuses_returned_page(owner,monkeypatch):
    plane,args=owner
    original=plane._page
    def change(*values,**kwargs):
        result=original(*values,**kwargs)
        (args['repository']/'calc.py').write_text('def increment(n: int) -> int:\n    return n + 3\n')
        return result
    monkeypatch.setattr(plane,'_page',change)
    with pytest.raises(ValueError):plane.relevant_units(query=RelevantUnitsQuery(),**args)


def test_ignore_configuration_change_refuses_frozen_page(owner):
    plane,args=owner
    first=plane.relevant_units(query=RelevantUnitsQuery(page_size=2),**args)
    with (args['repository']/'.gitignore').open('a') as out:out.write('unsupported.py\n')
    with pytest.raises(ValueError):
        plane.relevant_units(query=RelevantUnitsQuery(page_size=2),cursor=first['next_cursor'],**args)
