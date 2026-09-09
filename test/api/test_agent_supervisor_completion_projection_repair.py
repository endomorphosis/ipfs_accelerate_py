"""Legacy receipt repair must prove exact admitted history, never new success."""
import json
from copy import deepcopy

import pytest

from test.api.test_agent_supervisor_intent_repository import _repo, _seed_graph
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    IntentRepositoryIntegrityError, _canonical,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.task_sources.completion_projection_repair import (
    prepare_repair, apply_projection, EVENT, COUNTER,
)


def legacy_task(repo):
    ids = _seed_graph(repo)
    cid = ids['task_a']
    t = repo.get_task(cid)
    repo.cas_task_status(task_cid=cid, expected_revision=t['revision'], new_status='in_progress',
                         receipt={'operation':'claim',COUNTER:2})
    repo.record_validation_result(task_cid=cid, outcome='passed', evidence_digest=ids['evidence_digest'])
    t=repo.get_task(cid)
    repo.cas_task_status(task_cid=cid, expected_revision=t['revision'], new_status='completed',
        receipt={'operation':'completed','validation':'passed'}, evidence_digests=[ids['evidence_digest']])
    t=repo.get_task(cid)
    body=deepcopy(t['body']);body.pop(COUNTER);body['completion_receipt'][COUNTER]=2
    with repo._connection(write=True) as c:
        c.execute('UPDATE tasks SET body_json=? WHERE task_cid=?',[_canonical(body),cid])
        c.execute('UPDATE task_revisions SET body_json=? WHERE task_cid=? AND revision=?',
                  [_canonical(body),cid,t['revision']])
    return cid,t['revision'],body


def test_exact_repair_preserves_revision_receipt_history_and_replays(tmp_path):
    with _repo(tmp_path) as repo:
        cid,rev,body=legacy_task(repo)
        with repo._connection(write=True) as c:
            before=c.execute('SELECT * FROM completion_receipts').fetchall()
            history=c.execute('SELECT * FROM task_revisions').fetchall()
            repair=prepare_repair(c,task_cid=cid,expected_revision=rev,expected_body_cid=content_identity(body))
            apply_projection(c,repair)
            repo._append_event(c,event_type=EVENT,subject_id=cid,task_cid=cid,body=repair)
            assert c.execute('SELECT * FROM completion_receipts').fetchall()==before
            assert c.execute('SELECT * FROM task_revisions').fetchall()==history
        for replay in (False,True):
            if replay: repo.rebuild_projections_from_events()
            task=repo.get_task(cid)
            assert task['revision']==rev and task['status']=='completed'
            assert task['body'][COUNTER]==2 and COUNTER not in task['body']['completion_receipt']
            with repo._connection(write=True) as c:
                apply_projection(c,repair) # crash-after-commit replay is idempotent
                rows=c.execute('SELECT receipt_cid,task_cid,goal_cid,attempt_id,claim_cid,fencing_token,completed_at,validation_run_id,evidence_digest,body_json FROM completion_receipts WHERE task_cid=?',[cid]).fetchall()
                binding,reasons=repo._current_task_completion_binding(task,rows)
                assert binding and not reasons


@pytest.mark.parametrize('bad',['revision','cas','body','event','receipt','history','counter'])
def test_rejects_nonexact_preimages(tmp_path,bad):
    with _repo(tmp_path) as repo:
        cid,rev,body=legacy_task(repo)
        with repo._connection(write=True) as c:
            if bad=='body':
                body['unrelated']='changed'
                c.execute('UPDATE tasks SET body_json=? WHERE task_cid=?',[_canonical(body),cid])
            if bad=='event':
                c.execute("UPDATE domain_events SET event_id='forged' WHERE task_cid=? AND event_type='intent.completion_recorded'",[cid])
            if bad=='receipt':
                c.execute("UPDATE completion_receipts SET evidence_digest='foreign' WHERE task_cid=?",[cid])
            if bad=='history':
                c.execute('DELETE FROM task_revisions WHERE task_cid=? AND revision=?',[cid,rev-1])
            if bad=='counter':
                body['completion_receipt'][COUNTER]=3
                c.execute('UPDATE tasks SET body_json=? WHERE task_cid=?',[_canonical(body),cid])
                c.execute('UPDATE task_revisions SET body_json=? WHERE task_cid=? AND revision=?',[_canonical(body),cid,rev])
            with pytest.raises(IntentRepositoryIntegrityError):
                prepare_repair(c,task_cid=cid,expected_revision=rev+1 if bad=='revision' else rev,
                    expected_body_cid='wrong' if bad=='cas' else content_identity(body))


@pytest.mark.parametrize('bad',['', 'generation','birth','effect','lease','runtime','rollback'])
def test_native_owner_repair_fences_quiescence_and_transaction_replay(tmp_path,bad):
    from ipfs_accelerate_py.agent_supervisor.task_sources.completion_projection_repair import recover_on_owner
    with _repo(tmp_path) as repo:
        cid,rev,body=legacy_task(repo)
        identity=dict(database_uuid='db:repair',generation=1,fence_epoch=1,process_birth_id='birth:repair')
        with repo._connection(write=True) as c:
            c.execute("INSERT INTO store_generations (generation,schema_revision,fence_epoch,revision,database_uuid,birth_id,created_at) VALUES (1,3,1,0,'db:repair','birth:repair','now')")
            if bad=='generation':identity['generation']=2
            if bad=='birth':identity['process_birth_id']='birth:other'
            if bad=='effect':
                c.execute("INSERT INTO effect_claims (effect_id,task_cid,attempt_id,effect_kind,target_path,state,claimed_at,body_json) VALUES ('effect:one',?,'attempt:one','write','source','active','now','{}')",[cid])
            if bad=='lease':
                c.execute("INSERT INTO leases (task_cid,claim_cid,resolution_cid,claimant_did,logical_epoch,fencing_token,expires_at_ms,attempt,state,started_at_ms) VALUES (?,'claim','resolution','worker',1,1,9999999999999,1,'accepted',1)",[cid])
            if bad=='runtime':
                c.execute("INSERT INTO task_assignments (assignment_id,task_cid,owner_session_id,daemon_id,assigned_at,state,revision) VALUES ('assignment:one',?,'session','daemon','now','unknown',1)",[cid])
        if bad in ('generation','birth','effect','lease','runtime'):
            with repo._connection(write=True) as c:
                with pytest.raises(IntentRepositoryIntegrityError):
                    recover_on_owner(c,owner_identity=identity,task_cids=[cid])
            assert repo.get_task(cid)['body']==body
            return
        if bad=='rollback':
            with pytest.raises(RuntimeError,match='crash'):
                with repo._connection(write=True) as c:
                    assert recover_on_owner(c,owner_identity=identity,task_cids=[cid])[0]['changed']
                    raise RuntimeError('crash before commit')
            assert repo.get_task(cid)['body']==body
        with repo._connection(write=True) as c:
            result=recover_on_owner(c,owner_identity=identity,task_cids=[cid])
            assert len(result)==1 and result[0]['changed']
            assert c.execute('SELECT revision FROM store_generations').fetchone()[0] == 1
        with repo._connection(write=True) as c:
            assert recover_on_owner(c,owner_identity=identity,task_cids=[cid])==[]
            assert c.execute('SELECT revision FROM store_generations').fetchone()[0] == 1
        assert COUNTER not in repo.get_task(cid)['body']['completion_receipt']


def test_replay_cannot_replace_unrelated_task_data(tmp_path):
    with _repo(tmp_path) as repo:
        cid,rev,body=legacy_task(repo)
        with repo._connection(write=True) as c:
            repair=prepare_repair(c,task_cid=cid,expected_revision=rev,expected_body_cid=content_identity(body))
            repair['body']['unrelated']='forged'
            repair['after_body_cid']=content_identity(repair['body'])
            with pytest.raises(IntentRepositoryIntegrityError,match='unrelated'):
                apply_projection(c,repair)
