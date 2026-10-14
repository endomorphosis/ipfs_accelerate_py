"""Content-free cumulative growth telemetry and exact completion boundaries."""
from copy import deepcopy
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import codex_usage_receipt as usage

THREAD = 'growth-thread-123456'


def counters(input=100, cached=50, output=10):
    return dict(input_tokens=input, cached_input_tokens=cached,
                output_tokens=output, total_tokens=input+output)


def event(kind, **fields):
    return dict(type='event_msg', payload=dict(type=kind, **fields))


def token(total, last=None):
    info = dict(total_token_usage=total)
    if last is not None:
        info['last_token_usage'] = last
    return event('token_count', info=info)


def recover(tmp_path, records, *, complete=True, raw_tail=''):
    home = tmp_path / 'home'
    folder = home / 'sessions/2026/10/13'
    folder.mkdir(parents=True, exist_ok=True)
    workspace = tmp_path / 'work'
    workspace.mkdir(exist_ok=True)
    rows = [dict(type='session_meta', payload=dict(id=THREAD, cwd=str(workspace)))] + records
    if complete:
        rows += [event('task_complete')]
    (folder / f'rollout-{THREAD}.jsonl').write_text(
        ''.join(json.dumps(row)+'\n' for row in rows) + raw_tail)
    return usage.recover_codex_usage(home=home, thread_id=THREAD, workspace=workspace)


def test_final_cumulative_remains_last_and_first_baseline_is_unknown(tmp_path):
    first, last = counters(), counters(130, 70, 15)
    delta = counters(30, 20, 5)
    r = recover(tmp_path, [token(first, first), token(last, delta)])
    assert r['usage'] == last and r['observed_token_count_records'] == 2
    assert r['task_complete_observed'] is True
    growth = r['usage_growth']
    one, two = growth['records']
    assert one['status'] == 'initial_baseline_unobserved'
    assert one['delta_usage'] is one['delta_uncached_input_tokens'] is None
    assert one['reported_last_usage']['input_tokens'] == 100
    assert two['delta_usage']['input_tokens'] == 30
    assert two['delta_uncached_input_tokens'] == 10
    assert two['reported_last_matches_delta'] is True
    assert growth['cache_included_in_input'] is True
    assert growth['truncated'] is False and growth['omitted_token_events'] == 0
    assert 'not necessarily distinct model requests' in growth['scope']


def test_repeated_snapshot_is_zero_delta_not_another_request(tmp_path):
    a, b = counters(), counters(150, 75, 20)
    r = recover(tmp_path, [token(a), token(b), token(b, counters(50, 25, 10))])
    row = r['usage_growth']['records'][2]
    assert row['status'] == 'repeated_snapshot'
    assert all(row['delta_usage'][k] == 0 for k in usage._USAGE_KEYS)
    assert row['reported_last_matches_delta'] is False
    assert row['delta_uncached_input_tokens'] == 0
    assert r['usage'] == b and r['usage']['total_tokens'] == 170


@pytest.mark.parametrize('key', usage._USAGE_KEYS)
@pytest.mark.parametrize('bad', [None, True, -1, '10', 1.5, 1 << 63])
def test_invalid_category_does_not_invent_a_delta_or_baseline(tmp_path, key, bad):
    invalid = counters(); invalid[key] = bad
    r = recover(tmp_path, [token(counters()), token(invalid), token(counters(200, 100, 20)),
                           token(counters(210, 100, 22))])
    rows = r['usage_growth']['records']
    assert rows[1]['status'] == 'missing_or_invalid_cumulative_usage'
    assert rows[1]['delta_usage'] is rows[1]['delta_uncached_input_tokens'] is None
    assert rows[2]['status'] == 'previous_baseline_unavailable'
    assert rows[2]['delta_usage'] is None
    assert rows[3]['delta_usage']['input_tokens'] == 10


@pytest.mark.parametrize('mutate', [
    lambda x: x.update(total_tokens=0),
    lambda x: x.update(cached_input_tokens=101),
    lambda x: x.update(reasoning_output_tokens=11),
    lambda x: x.update(cache_write_input_tokens=-1),
])
def test_arithmetic_and_optional_categories_must_be_coherent(tmp_path, mutate):
    value = counters(); mutate(value)
    r = recover(tmp_path, [token(value)])
    row = r['usage_growth']['records'][0]
    assert row['status'] == 'missing_or_invalid_cumulative_usage'
    assert row['cumulative_uncached_input_tokens'] is None
    assert r['usage_growth']['final_snapshot_valid'] is False


def test_nonmonotone_reset_requires_new_baseline(tmp_path):
    r = recover(tmp_path, [token(counters()), token(counters(90, 40, 8)),
                           token(counters(110, 50, 10)), token(counters(120, 50, 12))])
    rows = r['usage_growth']['records']
    assert rows[1]['status'] == 'nonmonotone_cumulative_usage'
    assert rows[1]['delta_usage'] is None
    assert rows[2]['status'] == 'previous_baseline_unavailable'
    assert rows[3]['delta_uncached_input_tokens'] == 10
    assert r['usage'] == counters(120, 50, 12)


def test_cache_reclassification_cannot_be_negative_uncached_delta(tmp_path):
    r = recover(tmp_path, [token(counters(100, 50)), token(counters(101, 60)),
                           token(counters(110, 65))])
    rows = r['usage_growth']['records']
    assert rows[1]['status'] == 'inconsistent_usage_delta'
    assert rows[1]['delta_usage'] is rows[1]['delta_uncached_input_tokens'] is None
    assert rows[2]['status'] == 'previous_baseline_unavailable'


def test_missing_optional_baseline_is_not_zero(tmp_path):
    one = counters()
    two = dict(counters(110, 55, 12), reasoning_output_tokens=2)
    r = recover(tmp_path, [token(one), token(two, dict(counters(10, 5, 2), reasoning_output_tokens=2))])
    row = r['usage_growth']['records'][1]
    assert row['delta_usage']['reasoning_output_tokens'] is None
    assert row['reported_last_matches_delta'] is None


def test_missing_token_info_is_counted_but_not_fabricated(tmp_path):
    r = recover(tmp_path, [token(counters()), event('token_count', info=None),
                           token(counters(110, 55, 12))])
    assert r['usage_growth']['observed_token_events'] == 3
    assert r['observed_token_count_records'] == 2
    assert r['usage_growth']['records'][1]['status'] == 'missing_or_invalid_cumulative_usage'
    assert r['usage_growth']['records'][2]['status'] == 'previous_baseline_unavailable'
    assert r['usage'] == counters(110, 55, 12)


@pytest.mark.parametrize('record', [event('task_started'), event('turn_started'), event('turn_aborted'),
    event('task_aborted'), event('user_message', message='PRIVATE'), token(counters(120, 60, 12)),
    event('token_count', info=None), dict(type='turn_context', payload={'private': 'PRIVATE'})])
def test_completion_resets_after_new_work_or_abort(tmp_path, record):
    r = recover(tmp_path, [token(counters()), event('task_complete'), record], complete=False)
    assert r['task_complete_observed'] is False
    assert r['tool_diagnostics']['tool_event_counts_complete'] is False
    r = recover(tmp_path, [token(counters()), event('task_complete'), record])
    assert r['task_complete_event_observed'] is True
    assert r['task_complete_observed'] is (record != event('token_count', info=None))


@pytest.mark.parametrize('bad', [[], 12, dict(type='event_msg', payload='PRIVATE')])
def test_malformed_records_break_baseline_and_prevent_complete_claim(tmp_path, bad):
    r = recover(tmp_path, [token(counters()), bad, token(counters(130, 60, 13))])
    assert r['task_complete_observed'] is False
    assert r['usage_growth']['unreadable_record_gaps'] == 1
    assert r['usage_growth']['records'][1]['status'] == 'previous_baseline_unavailable'
    assert r['usage'] == counters(130, 60, 13)


def test_malformed_json_tail_after_completion_is_not_complete(tmp_path):
    r = recover(tmp_path, [token(counters())], raw_tail='{private-partial')
    assert r['task_complete_observed'] is False
    assert r['usage_growth']['unreadable_record_gaps'] == 1
    assert r['tool_diagnostics']['complete_tool_output_utf8_bytes'] is None


@pytest.mark.parametrize('limit', ['records', 'bytes'])
def test_telemetry_is_bounded_and_final_cumulative_is_still_current(tmp_path, monkeypatch, limit):
    if limit == 'records':
        monkeypatch.setattr(usage, 'MAX_USAGE_GROWTH_RECORDS', 2)
    else:
        monkeypatch.setattr(usage, 'MAX_USAGE_GROWTH_BYTES', 1000)
    totals = [counters(i * 100, i * 50, i * 10) for i in range(1, 21)]
    r = recover(tmp_path, [token(x) for x in totals]);g = r['usage_growth']
    assert r['usage'] == totals[-1] and r['observed_token_count_records'] == 20
    assert g['observed_token_events'] == 20 and g['truncated'] is True
    assert len(g['records']) == g['retained_token_events'] <= 2
    assert g['omitted_token_events'] == 20 - len(g['records'])
    encoded = json.dumps(g['records'], sort_keys=True, separators=(',', ':')).encode()
    assert len(encoded) == g['retained_rows_canonical_bytes'] <= g['max_retained_rows_bytes']
    assert sum(g['status_counts'].values()) == 20


def test_private_text_and_unknown_token_fields_never_enter_telemetry(tmp_path):
    marker = 'PRIVATE_TOKEN_GROWTH_BODY'
    value = counters(); value.update(prompt=marker, secret=marker)
    rows = [token(value, value), dict(type='response_item', payload=dict(type='reasoning', text=marker)),
            dict(type='response_item', payload=dict(type='message', content=marker)),
            dict(type='response_item', payload=dict(type='custom_tool_call', input=marker)),
            dict(type='response_item', payload=dict(type='custom_tool_call_output', output=marker))]
    r = recover(tmp_path, rows)
    assert marker not in json.dumps(r)
    assert r['tool_diagnostics']['observed_tool_output_utf8_bytes'] == len(marker.encode())
    assert r['tool_diagnostics']['complete_tool_output_utf8_bytes'] == len(marker.encode())


def test_tool_bytes_have_explicit_utf8_and_canonical_json_scope(tmp_path):
    structured = {'z': ['é'], 'a': 1}
    rows = []
    for output in ['é', structured]:
        rows += [dict(type='response_item', payload=dict(type='custom_tool_call', input='authored')),
                 dict(type='response_item', payload=dict(type='custom_tool_call_output', output=output))]
    r = recover(tmp_path, rows);d = r['tool_diagnostics']
    expected = len('é'.encode()) + len(json.dumps(structured, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())
    assert d['observed_tool_output_utf8_bytes'] == d['complete_tool_output_utf8_bytes'] == expected
    assert d['outputs_without_byte_measurement'] == 0
    assert 'not tokenized or provider-input bytes' in d['tool_output_byte_scope']


@pytest.mark.parametrize('output', [None, 1, '\ud800', {'bad': float('inf')}])
def test_unmeasurable_output_does_not_invent_complete_bytes(tmp_path, output):
    rows = [dict(type='response_item', payload=dict(type='custom_tool_call', input='authored')),
            dict(type='response_item', payload=dict(type='custom_tool_call_output', output=output))]
    d = recover(tmp_path, rows)['tool_diagnostics']
    assert d['observed_tool_output_utf8_bytes'] == 0
    assert d['complete_tool_output_utf8_bytes'] is None
    assert d['outputs_without_byte_measurement'] == 1


@pytest.mark.parametrize('kind', ['function_call', 'custom_tool_call', 'function_call_output', 'custom_tool_call_output'])
def test_later_tool_activity_revokes_completion_without_turn_marker(tmp_path, kind):
    row = dict(type='response_item', payload=dict(type=kind, input='authored', arguments='authored', output='authored'))
    r = recover(tmp_path, [token(counters()), event('task_complete'), row], complete=False)
    assert r['task_complete_observed'] is False


@pytest.mark.parametrize('bad', [event('token_count', info=None), token({'input_tokens': 100}),
                               token(dict(counters(), total_tokens=0))])
def test_task_complete_cannot_promote_stale_counts_after_invalid_token_event(tmp_path, bad):
    r = recover(tmp_path, [token(counters()), bad])
    assert r['task_complete_observed'] is False
    assert r['usage_growth']['final_snapshot_valid'] is False
    assert r['usage_available'] is True  # Historical observation remains visible.


def test_optional_counter_change_is_not_a_repeated_snapshot(tmp_path):
    first = dict(counters(), cache_write_input_tokens=0)
    second = dict(counters(), cache_write_input_tokens=1)
    r = recover(tmp_path, [token(first), token(second)])
    assert r['usage_growth']['records'][1]['status'] == 'observed_consecutive_delta'
    assert r['usage_growth']['records'][1]['delta_usage']['cache_write_input_tokens'] == 1


@pytest.mark.parametrize('tail', [counters(90, 40, 8), counters(101, 60, 10)])
def test_completion_cannot_promote_an_inconsistent_cumulative_sequence(tmp_path, tail):
    r = recover(tmp_path, [token(counters()), token(tail)])
    assert r['task_complete_event_observed'] is True
    assert r['task_complete_observed'] is False
    assert r['usage_growth']['final_snapshot_valid'] is True
    assert r['usage_growth']['cumulative_sequence_consistent'] is False
    assert r['usage'] == tail
    from benchmarks.agent_supervisor.container_coding.native_usage_accounting import native_usage_contract
    assert native_usage_contract({'provider': 'codex_cli'}, r) == (False, 'native_session_cumulative', True)


def test_no_usage_completion_marker_remains_distinct_from_unknown_usage(tmp_path):
    r = recover(tmp_path, [])
    assert r['task_complete_observed'] is r['task_complete_event_observed'] is True
    assert r['usage_available'] is False and r['usage'] == {}
    assert r['usage_growth']['final_snapshot_valid'] is None
    assert r['usage_growth']['observed_token_events'] == 0
    assert r['tool_diagnostics']['complete_tool_output_utf8_bytes'] == 0
