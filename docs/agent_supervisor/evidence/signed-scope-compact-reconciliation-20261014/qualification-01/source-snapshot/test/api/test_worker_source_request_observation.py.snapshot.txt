"""Actual socket attribution without granting refused requests source effects."""
import json
import socket

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as protocol
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_edit_client as edit
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_request_observation as observation
from test.api.test_worker_source_broker import service, _request  # noqa: F401
from test.api.test_semantic_router_integration import allocated, native  # noqa: F401
from test.api.test_worker_source_edit_broker import case, read_working  # noqa: F401


def _exchange(service, request):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(30)
        channel.connect(str(service.socket_path))
        protocol.send_frame(channel, request)
        return protocol.receive_frame(channel)


@pytest.mark.parametrize('failure,reason', [('foreign_path', 'retrieval_refused'),
    ('peer', 'peer_refused'), ('context', 'stale_context')])
def test_valid_refusal_keeps_requested_read_shape_without_mutation(service, monkeypatch, failure, reason):
    before = (service.workspace / 'mod.py').read_bytes()
    monkeypatch.setattr(service, '_peer', lambda _: None)
    request = _request(service, view='original')
    if failure == 'foreign_path':
        request['query']['path'] = 'private-marker.py'
    elif failure == 'peer':
        def refuse(_):
            raise ValueError('authored peer refusal')
        monkeypatch.setattr(service, '_peer', refuse)
    else:
        def refuse(**kwargs):
            raise ValueError('authored context refusal')
        monkeypatch.setattr(service, '_current_context', refuse)
    response = _exchange(service, request)
    assert response['ok'] is False and response['reason_code'] == reason
    service.close()
    audit = json.loads(service.audit_path.read_bytes())
    record, = audit['requests']
    assert observation.validate_record(record) == ('read', 'baseline', 'original')
    assert record['returned_source_bytes'] == 0 and record['status'] == reason
    assert (service.workspace / 'mod.py').read_bytes() == before
    assert b'private-marker' not in protocol.canonical_json(audit)


def test_malformed_socket_request_remains_unattributed(service, monkeypatch):
    monkeypatch.setattr(service, '_peer', lambda _: pytest.fail('malformed request reached admission'))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(30)
        channel.connect(str(service.socket_path))
        protocol.send_frame(channel, dict(schema=protocol.REQUEST_SCHEMA, private_body='secret-marker'))
        assert channel.recv(1) == b''
    service.close()
    audit = json.loads(service.audit_path.read_bytes())
    record, = audit['requests']
    assert observation.validate_record(record) == observation.UNKNOWN
    assert record['request_sha256'] is record['response_sha256'] is None
    assert record['status'] == 'invalid_request' and record['response_bytes'] == 0
    assert b'secret-marker' not in protocol.canonical_json(audit)


def test_successful_mapping_and_refused_stale_mapping_are_distinguishable(case):
    service = case['service']
    _, change = read_working(case)
    before = (case['workspace'] / 'mod.py').read_bytes()
    stale = dict(change, expected_source_sha256='0' * 64)
    refusal = _exchange(service, edit._request(service.descriptor, operation='prepare', **stale))
    assert refusal['ok'] is False and refusal['reason_code'] == 'retrieval_refused'
    assert (case['workspace'] / 'mod.py').read_bytes() == before
    result = edit.edit_source(service.descriptor_path, service.descriptor_sha256, edit=change)
    assert result['ok'] and result['validation_pending']
    service.close()
    audit = json.loads(service.audit_path.read_bytes())
    assert [(record['request_operation'], record['status']) for record in audit['requests']] == [
        ('read', 'ok'), ('edit_prepare', 'retrieval_refused'), ('edit_prepare', 'ok'), ('edit_confirm', 'ok')]
    counts = observation.operation_counts(audit['requests'])
    assert counts['accepted_requests'] == 4 and counts['unparsed_requests'] == 0
    assert next(row for row in counts['rows'] if row['operation'] == 'edit_prepare')['status_counts'] == {
        'ok': 1, 'retrieval_refused': 1}
    assert (case['root'] / 'mod.py').read_bytes() == before
    assert b'expected_view_preimage' not in protocol.canonical_json(audit)
    assert all(record['owner_writes_performed'] is False for record in audit['requests'])
