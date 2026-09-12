"""A lane restart ignores unrelated stale indexes in the shared Git store."""
import dataclasses
import json

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    WorktreeLifecycleStore, ProcessBirthIdentity, OwnershipError,
)


def fixture(tmp_path):
    root=tmp_path/'current';root.mkdir();state=root/'lane-1'
    store=WorktreeLifecycleStore(root,store_dir=tmp_path/'shared-lifecycle')
    dead=ProcessBirthIdentity(pid=2**30-9,start_time_ticks=1,boot_id='dead-test-owner')
    def seed(s,name,owner=dead,lane_state=state):
        return s.begin_preparing(task_id=name,canonical_task_cid='task:'+name,attempt=1,
            lane_id='lane-1',workspace_path=tmp_path/name,branch='implementation/'+name,
            merge_target='main',state_dir=str(lane_state),owner=owner)
    return store,state,seed


def index_for(store,record):
    return store.task_index_path_for(canonical_task_cid=record.canonical_task_cid,task_id=record.task_id,attempt=record.attempt)


def break_index(path,mode):
    if mode=='missing':path.unlink()
    else:
        value=json.loads(path.read_text());value['record_id']='another-workspace';path.write_text(json.dumps(value))


@pytest.mark.parametrize('scope',['other-repository','other-lane'])
@pytest.mark.parametrize('damage',['missing','rebound'])
def test_unrelated_broken_index_cannot_block_current_lane(tmp_path,scope,damage):
    store,state,seed=fixture(tmp_path);own=seed(store,'current-task')
    other=WorktreeLifecycleStore(tmp_path/'historical',store_dir=store.store_dir) if scope=='other-repository' else store
    foreign=seed(other,'historical-task',lane_state=state if scope=='other-repository' else tmp_path/'other-lane')
    index=index_for(store,foreign);break_index(index,damage)
    before=store.workspace_path_for(foreign.workspace_path).read_bytes();pointer=index.read_bytes() if index.exists() else None
    result=store.reclaim_dead_owners_for_controlled_restart(expected_state_dir=state)
    assert [r.task_id for r in result]==[own.task_id]
    assert store.workspace_path_for(foreign.workspace_path).read_bytes()==before
    assert (index.read_bytes() if index.exists() else None)==pointer


@pytest.mark.parametrize('damage',['missing','rebound'])
def test_selected_lane_still_refuses_broken_routing_index(tmp_path,damage):
    store,state,seed=fixture(tmp_path);record=seed(store,'selected-task')
    break_index(index_for(store,record),damage);p=store.workspace_path_for(record.workspace_path);before=p.read_bytes()
    with pytest.raises(OwnershipError,match='index changed'):
        store.reclaim_dead_owners_for_controlled_restart(expected_state_dir=state)
    assert p.read_bytes()==before


def test_live_owner_is_not_a_controlled_restart_candidate(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import current_process_birth
    store,state,seed=fixture(tmp_path);record=seed(store,'live-task',owner=current_process_birth())
    index_for(store,record).unlink();p=store.workspace_path_for(record.workspace_path);before=p.read_bytes()
    assert store.reclaim_dead_owners_for_controlled_restart(expected_state_dir=state)==[]
    assert p.read_bytes()==before


def test_selected_record_is_compared_again_under_effect_guards(tmp_path,monkeypatch):
    store,state,seed=fixture(tmp_path);record=seed(store,'selected-task')
    actual=store.reclaim_dead_owner_for_controlled_restart
    def changed(workspace,**kwargs):
        assert kwargs['expected_record']==record
        newer=dataclasses.replace(record,fence=record.fence+1)
        store.workspace_path_for(workspace).write_text(json.dumps(newer.to_dict()))
        return actual(workspace,**kwargs)
    monkeypatch.setattr(store,'reclaim_dead_owner_for_controlled_restart',changed)
    assert store.reclaim_dead_owners_for_controlled_restart(expected_state_dir=state)==[]
    assert store.load_workspace(record.workspace_path).is_nonterminal
