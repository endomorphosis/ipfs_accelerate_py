"""Normal control reopening, stale-source refusal and immutable history checks."""
import json
import pytest
from test.api.test_agent_supervisor_database_implementation_daemon import _open_daemon, _population
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import DatabaseCoordinationStaleFenceError, DatabaseCoordinationExpiredError


def rows(c,table):
    r=c.execute('SELECT * FROM '+table)
    return sorted((dict(v) for v in r.fetchall()),key=lambda x:json.dumps(x,sort_keys=True,default=str))


def complete(d):
    d.materialize_population(_population(2))
    d.run_once()
    t=d.task_source.get('task:cid:001')
    assert t.status=='completed'
    return t


def reopen(d,t):
    return d.task_source.compare_and_set_status(t.task_cid,expected_revision=t.revision,status='ready',receipt={'operation':'operator_corrected_scope','prior_history_preserved':True}).task


def test_reopening_preserves_history_and_claims_next_attempt(tmp_path):
    calls=[]
    with _open_daemon(tmp_path,provider_calls=calls)as d:
        done=complete(d);t=reopen(d,done);c=d.coordinator._require()
        preserved={name:rows(c,name)for name in ['task_attempts','task_claims','fenced_leases','token_history','task_dependencies']}
        old=rows(c,'task_completions');events=rows(c,'lease_events')
        result=d.coordinator.reconcile_reopened_task(task_cid=t.task_cid,expected_control_revision=t.revision,read_control_task=d.task_source.get)
        assert result['reopened'] and result['attempt_budget_reset']is False
        for name,want in preserved.items():assert rows(c,name)==want
        assert rows(c,'task_completions')==[]
        extra=[r for r in rows(c,'lease_events')if r not in events]
        assert len(extra)==1 and json.loads(extra[0]['body_json'])['archived_completion_row']==old[0]
        assert d.coordinator.reconcile_reopened_task(task_cid=t.task_cid,expected_control_revision=t.revision,read_control_task=d.task_source.get)['reopened']is False
        assert len(rows(c,'lease_events'))==len(events)+1
        claim=d.claim_next();assert claim.task_cid==t.task_cid and claim.attempt_number==2
        d.resume_attempt(claim)
        assert calls==['task:cid:001','task:cid:001']
        assert d.task_source.get(t.task_cid).status=='completed'
        assert d.coordinator.claimability(t.task_cid)['claimable']is False


@pytest.mark.parametrize('delta',[-1,0])
def test_old_or_equal_revision_rejected_without_mutation(tmp_path,delta):
    with _open_daemon(tmp_path)as d:
        t=complete(d);c=d.coordinator._require();before={n:rows(c,n)for n in ['task_completions','coordination_tasks','lease_events']}
        with pytest.raises(DatabaseCoordinationStaleFenceError):
            d.coordinator.reconcile_reopened_task(task_cid=t.task_cid,expected_control_revision=t.revision+delta,read_control_task=d.task_source.get)
        assert before=={n:rows(c,n)for n in before}


@pytest.mark.parametrize('kind',['completed','missing','changed_revision'])
def test_current_control_must_still_be_exact_ready(tmp_path,kind):
    with _open_daemon(tmp_path)as d:
        t=complete(d);expected=t.revision+1
        if kind!='completed':t=reopen(d,t);expected=t.revision
        reader=d.task_source.get
        if kind=='missing':reader=lambda cid:None
        if kind=='changed_revision':expected+=1
        c=d.coordinator._require();before={n:rows(c,n)for n in ['task_completions','coordination_tasks','lease_events']}
        with pytest.raises(DatabaseCoordinationStaleFenceError):d.coordinator.reconcile_reopened_task(task_cid=t.task_cid,expected_control_revision=expected,read_control_task=reader)
        assert before=={n:rows(c,n)for n in before}


def test_unsettled_live_lease_refuses_reopening(tmp_path,monkeypatch):
    with _open_daemon(tmp_path)as d:
        d.materialize_population(_population(1))
        def pending(*args,**kwargs):raise RuntimeError('constructed interruption before settlement')
        monkeypatch.setattr(d.coordinator,'settle_task_claim',pending)
        with pytest.raises(RuntimeError,match='before settlement'):d.run_once()
        t=reopen(d,d.task_source.get('task:cid:001'));c=d.coordinator._require();before={n:rows(c,n)for n in ['task_completions','coordination_tasks','lease_events','task_claims','fenced_leases']}
        with pytest.raises(DatabaseCoordinationExpiredError,match='not released'):d.coordinator.reconcile_reopened_task(task_cid=t.task_cid,expected_control_revision=t.revision,read_control_task=d.task_source.get)
        assert before=={n:rows(c,n)for n in before}


def test_daemon_automatically_reconciles_only_eligible_task(tmp_path):
    with _open_daemon(tmp_path)as d:
        t=complete(d);t=reopen(d,t)
        claim=d.claim_next()
        assert claim.task_cid==t.task_cid and claim.attempt_number==2
        assert d.task_source.get(t.task_cid).revision==t.revision+1
