"""Exact recorded canonical history plus constructed physical denial checks."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon, DatabaseImplementationAuthorityError
from ipfs_accelerate_py.agent_supervisor.todo_daemon.retained_callback_suffix import IDENTITY, CALLBACK_REASON, PREFLIGHT_REASON, verified_suffix


def history():
    return json.loads((Path(__file__).parent / 'fixtures/retained_callback_preflight_history.json').read_text())


def rehash(h):
    h.pop('projection_cid', None)
    h['projection_cid'] = content_identity(h)


@pytest.mark.parametrize('mutation', [None, 'hash', 'gap', 'semantic', 'source_reason', 'terminal_reason', 'extra_field', 'foreign_route', 'foreign_claim', 'foreign_admission', 'budget_count', 'budget_hash', 'provider_consumed', 'active', 'wrong_fence', 'wrong_task'])
def test_exact_recorded_callback_preflight_suffix(mutation):
    h = history(); rows = h['revisions']; source = rows[3]['body']['completion_receipt']; terminal = rows[-1]['body']['completion_receipt']
    if mutation == 'gap': rows.pop(4)
    elif mutation == 'semantic': rows[7]['body']['title'] = 'changed'
    elif mutation == 'source_reason': source['reason'] = 'arbitrary'
    elif mutation == 'terminal_reason': terminal['reason'] = 'arbitrary'
    elif mutation == 'extra_field': terminal['authority'] = True
    elif mutation == 'foreign_route': rows[7]['body']['completion_receipt']['execution_route_binding']['task_cid'] = 'foreign'
    elif mutation == 'foreign_claim': rows[5]['body']['completion_receipt']['claim_id'] = 'foreign'
    elif mutation == 'foreign_admission': rows[6]['body']['completion_receipt']['claim_id'] = 'foreign'
    elif mutation == 'budget_count': terminal['retry_budget']['verified_typed_deferral_count'] = 1
    elif mutation == 'budget_hash': terminal['retry_budget']['observation_id'] = 'foreign'
    elif mutation == 'provider_consumed': terminal['attempt_consumed'] = True
    elif mutation == 'active': rows[-1]['status'] = 'in_progress'
    elif mutation == 'wrong_fence': terminal['fence_epoch'] += 1
    elif mutation == 'wrong_task': h['task_cid'] = 'foreign'
    rehash(h)
    if mutation == 'hash': h['projection_cid'] = 'foreign'
    result = verified_suffix(h, task_cid=history()['task_cid'], task_alias='DOEP-053', control_revision=11)
    assert (result is not None) == (mutation is None)
    if result:
        assert result['source_task_revision'] == 4
        assert result['source_receipt']['reason'] == CALLBACK_REASON
        assert result['preflight_exhaustion'] is True


def physical_fixture():
    # These phases are constructed for unit verification, not captured native evidence.
    h = history(); rows = h['revisions']; cid = h['task_cid']
    d = object.__new__(DatabaseImplementationDaemon)
    d.open = lambda: d
    receipts = [rows[i]['body']['completion_receipt'] for i in (3, 7, 10)]
    attempts = {}; phases = {}
    for index, receipt in enumerate(receipts):
        a = SimpleNamespace(**{k:receipt[k] for k in IDENTITY}, task_cid=cid, task_alias='DOEP-053', status='failed', committed_phase='failed', revision=3, finished_at_ms=receipt['execution_finished_at_ms'])
        attempts[a.attempt_id] = a
        if index == 0:
            failed = {'attempt_consumed':'unknown', 'backoff_seconds':0, 'deferred':False, 'portal_retryable_failure':False, 'portal_terminal_failure':True, 'provider_dispatched':'unknown', 'reason':CALLBACK_REASON, 'typed_deferral_slot_consumed':'unknown'}
        else:
            gen = {'schema':'ipfs_accelerate_py/agent-supervisor/database-portal-typed-deferral@1', 'task_cid':cid, 'task_generation':cid, 'state_schema_revision':'fixture:current'}
            typed = {**gen, 'reason':PREFLIGHT_REASON, 'attempt_consumed':False, 'provider_dispatched':False, 'typed_deferral_slot_consumed':True, 'generation_fingerprint':d._database_portal_evidence_digest(gen)}
            typed['deferral_fingerprint'] = d._database_portal_evidence_digest(typed)
            typed['attempt_id'] = a.attempt_id
            failed = {'reason':PREFLIGHT_REASON, 'deferred':True, 'portal_retryable_failure':True, 'portal_terminal_failure':False, 'attempt_consumed':False, 'provider_dispatched':False, 'typed_deferral_slot_consumed':True, 'typed_deferral':typed}
        phases[a.attempt_id] = [{'revision':i, 'phase':phase, 'body':body, 'fencing_token':a.fencing_token, 'fence_epoch':a.fence_epoch, 'committed_at_ms':a.finished_at_ms} for i,phase,body in [(1,'claimed',{}),(2,'context',{'resumed':True}),(3,'failed',failed)]]
    d.get_attempt = attempts.get; d.phase_history = phases.__getitem__
    d._local_attempt_is_exact_latest = lambda a: True
    d._failed_attempt_coordination_successor = lambda a: None
    d._terminal_coordination_reproduces_read_only = lambda a, **kw: True
    d.list_running_attempts = list
    d._coordinator = SimpleNamespace(get_prepared_task_completion=lambda cid: None)
    d._task_source = SimpleNamespace(task_revision_history_projection=lambda cid: h)
    budget_calls = []
    def reproduced_budget(a):
        budget_calls.append(a.attempt_id)
        return copy.deepcopy(receipts[-1]['retry_budget'])
    d._typed_deferral_budget_observation = reproduced_budget
    task = SimpleNamespace(task_cid=cid, task_alias='DOEP-053', revision=11, status='blocked', body=rows[-1]['body'])
    return d, task, attempts, phases, budget_calls


@pytest.mark.parametrize('mutation', [None, 'provider_phase', 'provider_true', 'missing_source', 'changed_terminal', 'active_attempt', 'prepared', 'newer_cursor', 'live_fence', 'budget_mismatch'])
def test_physical_callback_deferral_gate(mutation):
    d, task, attempts, phases, budget_calls = physical_fixture()
    source, middle, current = list(attempts.values())
    if mutation == 'provider_phase': phases[current.attempt_id][1]['phase'] = 'provider'
    elif mutation == 'provider_true': phases[current.attempt_id][-1]['body']['provider_dispatched'] = True
    elif mutation == 'missing_source': attempts.pop(source.attempt_id)
    elif mutation == 'changed_terminal': phases[current.attempt_id][-1]['body']['reason'] = 'foreign'
    elif mutation == 'active_attempt': d.list_running_attempts = lambda:[current]
    elif mutation == 'prepared': d.coordinator.get_prepared_task_completion = lambda cid:{'prepared':True}
    elif mutation == 'newer_cursor': d._local_attempt_is_exact_latest = lambda a:False
    elif mutation == 'live_fence': d._terminal_coordination_reproduces_read_only = lambda a,**kw:False
    elif mutation == 'budget_mismatch': d._typed_deferral_budget_observation = lambda a:{}
    try:
        context = d._retained_callback_suffix_context(task)
    except DatabaseImplementationAuthorityError:
        context = None
    assert (context is not None) == (mutation is None)
    if context:
        assert context['source_attempt'] == source
        assert context['current_attempt'] == current
        assert budget_calls == [current.attempt_id]
        assert d._post_merge_completion_crash_recovery_context(
            task, require_current_blocked=True)['context_id'] == context['context_id']
