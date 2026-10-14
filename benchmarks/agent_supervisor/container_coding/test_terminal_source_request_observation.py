"""Versioned request attribution with unchanged historical source accounting."""
import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_request_observation as attribution
from benchmarks.agent_supervisor.container_coding.test_terminal_source_view_observation import (
    evidence, _collect, _fixture, _typed_audit, _mixed_pages,  # noqa: F401
)
from benchmarks.agent_supervisor.container_coding.test_source_edit_worker_wiring import _editing


def _upgrade(audit):
    audit['schema'] = attribution.AUDIT_SCHEMA
    for row in audit['requests']:
        if row['request_sha256'] is None:
            tags = attribution.UNKNOWN
        elif row.get('source_edit_page'):
            op = row['source_edit_operation']
            tags = ('edit_' + op, 'working-candidate', 'compact' if op == 'prepare' else 'not_applicable')
        elif row.get('candidate_context_page'):
            tags = ('candidate_context', 'working-candidate', 'not_applicable')
        elif row.get('discovery_page'):
            tags = ('discover', 'not_applicable', 'not_applicable')
        else:
            tags = ('read', 'baseline', 'compact')
        row.update(zip(attribution.FIELDS, tags))


@pytest.mark.parametrize('shared', [False, True])
def test_new_aggregate_retains_refused_and_malformed_requests(evidence, shared):
    _typed_audit(evidence[3], evidence[4], shared=shared)
    _upgrade(evidence[3]); evidence[5]()
    result = _collect(evidence)
    assert result['status'] == 'observed'
    assert result['counts']['accepted_requests'] == 3 and result['counts']['response_bytes'] == 2281
    counts = result['request_operation_counts']
    assert counts['accepted_requests'] == 3 and counts['unparsed_requests'] == 1
    assert counts['rows'] == [dict(operation='read', source_state='baseline', view='compact',
        status_counts={'ok': 1, 'retrieval_refused': 1}), dict(operation='unknown', source_state='unknown',
        view='unknown', status_counts={'invalid_request': 1})]
    assert result['request_operation_coverage'] == dict(observed_sessions=1, historical_sessions=0,
        observed_requests=3, historical_unattributed_requests=0, all_sessions_observed=True)
    assert counts['raw_query_fields_exported'] is counts['admission_authority'] is False


def test_mixed_old_and_new_audits_explicitly_preserve_unknown_historical_requests(evidence):
    _typed_audit(evidence[3], evidence[4]); _upgrade(evidence[3]); evidence[5]()
    old, row = _fixture('e' * 32)
    evidence[2]['provider_invocations'].append(row); evidence[5](old)
    result = _collect(evidence)
    assert result['status'] == 'observed' and result['counts']['accepted_requests'] == 6
    assert result['request_operation_coverage'] == dict(observed_sessions=1, historical_sessions=1,
        observed_requests=3, historical_unattributed_requests=3, all_sessions_observed=False)
    assert result['request_operation_counts']['accepted_requests'] == 3
    assert 'request_operation_counts' not in result['sessions'][1]


@pytest.mark.parametrize('schema', [1, 2, 3, 4])
def test_historical_audits_do_not_gain_fabricated_attribution(evidence, schema):
    if schema in (2, 3): _typed_audit(evidence[3], evidence[4], shared=schema == 3)
    if schema == 4: _editing(evidence)
    evidence[5]()
    result = _collect(evidence)
    assert result['status'] == 'observed'
    assert 'request_operation_counts' not in result and 'request_operation_coverage' not in result
    assert 'request_operation_counts' not in result['sessions'][0]


@pytest.mark.parametrize('mode', ['mixed', 'editing', 'empty'])
def test_all_success_page_types_and_empty_current_session_are_supported(evidence, mode):
    if mode == 'mixed': _mixed_pages(evidence)
    elif mode == 'editing': _editing(evidence)
    else:
        _typed_audit(evidence[3], evidence[4])
        evidence[3].update(requests=[], accepted_requests=0, response_bytes=0)
    _upgrade(evidence[3]); evidence[5]()
    result = _collect(evidence)
    assert result['status'] == 'observed'
    assert result['request_operation_counts']['accepted_requests'] == result['counts']['accepted_requests']
    assert result['request_operation_coverage']['all_sessions_observed']
    assert result['discovery_counts'] is not None


@pytest.mark.parametrize('change', ['missing', 'numeric', 'dynamic', 'impossible', 'unknown_hashed',
    'known_unhashed', 'success_mismatch', 'old_schema', 'missing_discovery', 'wrong_counter', 'private_field'])
def test_tampered_attribution_or_accounting_remains_unavailable(evidence, change):
    _typed_audit(evidence[3], evidence[4]); _upgrade(evidence[3])
    row = evidence[3]['requests'][0]
    if change == 'missing': row.pop('request_operation')
    elif change == 'numeric': row['request_operation'] = 1
    elif change == 'dynamic': row['request_operation'] = '/private/path'
    elif change == 'impossible': row['request_view'] = 'not_applicable'
    elif change == 'unknown_hashed': row.update(zip(attribution.FIELDS, attribution.UNKNOWN))
    elif change == 'known_unhashed': evidence[3]['requests'][2].update(zip(attribution.FIELDS, ('read', 'baseline', 'compact')))
    elif change == 'success_mismatch': row.update(zip(attribution.FIELDS, ('edit_prepare', 'working-candidate', 'compact')))
    elif change == 'old_schema': evidence[3]['schema'] = 'supervisor-worker-source-session@2'
    elif change == 'missing_discovery': row.pop('discovery_page'); row.pop('returned_discovery_rows')
    elif change == 'wrong_counter': evidence[3]['accepted_requests'] += 1
    else: row['private_query'] = 'source body'
    evidence[5]()
    result = _collect(evidence)
    assert result['status'] == 'unavailable' and result['counts'] is None
    assert 'request_operation_counts' not in result
