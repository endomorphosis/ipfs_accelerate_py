"""Post-STOP work cannot swallow its deadline or borrow finalization time."""
import signal
import time
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver


def report():
    return {'arm': 'full', 'task_state': {'status': 'completed'},
        'stop': {'status': 'succeeded'}, 'remaining_processes': 0, 'phases': {}}


def test_refresh_deadline_escapes_optional_exception_handlers_and_restores_outer_alarm(monkeypatch):
    observed = []
    prior = signal.getsignal(signal.SIGALRM)
    # Trigger the actual installed refresh handler. A component's ordinary
    # exception handler must not convert this deadline into optional absence.
    def rebuild():
        try:
            signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)
        except Exception:
            pytest.fail('refresh deadline was swallowed')
        observed.append('continued after expiry')
    timers = []
    monkeypatch.setattr(signal, 'setitimer', lambda kind, seconds: timers.append(seconds))
    value = report()
    driver._refresh_completed_context(SimpleNamespace(refresh_after_stop=rebuild), value,
                                     deadline=time.monotonic() + 30)
    assert value['post_publication_context']['status'] == 'unavailable'
    assert value['post_publication_context']['error_type'] == '_PostStopRefreshExpired'
    assert not observed and signal.getsignal(signal.SIGALRM) is prior
    assert 14 < timers[0] <= 15 and timers[1] == 0 and 27 < timers[2] <= 28
    assert value['task_state']['status'] == 'completed'


@pytest.mark.parametrize('change', [{'stop': {'status': 'failed'}}, {'remaining_processes': 1},
    {'task_state': {'status': 'in_progress'}}, {'arm': 'no-index'}])
def test_unsettled_or_unrequested_work_never_rebuilds(change):
    value = {**report(), **change}
    driver._refresh_completed_context(SimpleNamespace(refresh_after_stop=lambda: pytest.fail('must not rebuild')),
                                     value, deadline=time.monotonic() + 30)
    assert 'post_publication_context' not in value


def test_finalization_reserve_and_current_retrieval_are_required(monkeypatch):
    value = report()
    driver._refresh_completed_context(SimpleNamespace(refresh_after_stop=lambda: pytest.fail('no budget')),
                                     value, deadline=time.monotonic() + 18)
    assert value['post_publication_context']['status'] == 'deferred'
    monkeypatch.setattr(signal, 'setitimer', lambda *args: None)
    runtime = SimpleNamespace(refresh_after_stop=lambda: [{'status': 'refreshed', 'retrieval_status': 'unavailable'}])
    driver._refresh_completed_context(runtime, value, deadline=time.monotonic() + 30)
    assert value['post_publication_context']['status'] == 'incomplete'


def test_original_work_cutoff_bounds_refresh_without_shortening_cleanup(monkeypatch):
    monkeypatch.setattr(time, 'monotonic', lambda: 100.)
    timers = []
    monkeypatch.setattr(signal, 'setitimer', lambda kind, seconds: timers.append(seconds))
    value = report()
    driver._refresh_completed_context(SimpleNamespace(refresh_after_stop=lambda: []), value,
                                     deadline=200., work_deadline=120.)
    assert value['post_publication_context']['budget_seconds'] == 20.
    assert timers == [20., 0, 98.]
    assert value['task_state']['status'] == 'completed'


def test_expired_work_does_not_borrow_available_cleanup_time(monkeypatch):
    monkeypatch.setattr(time, 'monotonic', lambda: 100.)
    value = report()
    driver._refresh_completed_context(
        SimpleNamespace(refresh_after_stop=lambda: pytest.fail('cleanup time is not work')),
        value, deadline=160., work_deadline=99.)
    assert value['post_publication_context']['status'] == 'deferred'
    assert value['post_publication_context']['budget_seconds'] == 0.


@pytest.mark.skipif(not hasattr(signal, 'setitimer'), reason='requires POSIX alarms')
def test_real_refresh_alarm_interrupts_component_without_consuming_cleanup_reserve():
    prior = signal.getsignal(signal.SIGALRM)
    continued = []
    def rebuild():
        try:
            time.sleep(30)
        except Exception:
            continued.append('optional exception handler swallowed deadline')
        continued.append('continued past deadline')
    value = report()
    started = time.monotonic()
    try:
        driver._refresh_completed_context(SimpleNamespace(refresh_after_stop=rebuild), value,
                                         deadline=started + 20.1)
        elapsed = time.monotonic() - started
        assert 4.9 <= elapsed < 10
        assert value['post_publication_context']['error_type'] == '_PostStopRefreshExpired'
        assert value['post_publication_context']['embedding_calls'] is None
        assert not continued and signal.getsignal(signal.SIGALRM) is prior
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 10
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior)


@pytest.mark.parametrize('missing', [False, True])
def test_failed_learned_refresh_keeps_actual_inference_cost_and_unknown_totals(monkeypatch, missing):
    monkeypatch.setattr(signal, 'setitimer', lambda *args: None)
    receipt = {'schema': 'supervisor-published-learned-embedding-receipt@1',
        'status': 'unavailable', 'local_embedding_calls': 1, 'local_embedding_texts': 3,
        'remote_embedding_calls': 0, 'text_generation_calls': 0}
    rows = [{'status': 'refreshed', 'retrieval_status': 'unavailable', 'embedding_receipt': receipt}]
    if missing:
        rows.append({'status': 'unavailable'})
    value = report()
    driver._refresh_completed_context(SimpleNamespace(refresh_after_stop=lambda: rows), value,
                                     deadline=time.monotonic() + 30)
    observation = value['post_publication_context']
    assert observation['status'] == 'incomplete'
    assert observation['embedding_calls'] == (None if missing else 1)
    accounting = observation['embedding_accounting']
    assert accounting['all_refreshes_receipted'] is not missing
    assert accounting['known_subtotals']['local_embedding_calls'] == 1
    assert accounting['totals']['local_embedding_texts'] == (None if missing else 3)
