"""Portable metadata/kernel-I/O checks only; no native capabilities or handoff doubles."""
import fcntl
import hashlib
import os
import subprocess
import sys
import time

import pytest
import ipfs_accelerate_py.cli_runtime.grok_private_m1_offline_worker as worker
import ipfs_accelerate_py.cli_runtime.grok_private_m1_offline_check as checker


def inert_request(tmp_path):
    """Syntactically complete inert metadata; never a claim or native grant."""
    root = tmp_path / 'accelerator'
    pin = lambda path: dict(path=str(path), bytes=1, sha256='0' * 64)
    result = dict(schema=worker.SCHEMA, mode='offline-ACK-only', one_shot=True, attempt_cap=1,
        provider_dispatch_allowed=False, private_task_alias='private-M1a-' + 'a' * 32,
        database_task_cid='inert-task', allocation_session_id='inert-session',
        accelerator_root=str(root), workspace=str(tmp_path / 'workspace'),
        receipt_directory=str(tmp_path / 'receipts'),
        source_generation='grok-automatic-named-session/private-M1a-' + 'a' * 32,
        work_deadline_monotonic=1000.0, native_deadline_monotonic=1060.0,
        worker_pin=pin(root / worker.WORKER_RELATIVE), owner_client_pin=pin(root / worker.CLIENT_RELATIVE),
        source_profile_pin=pin(tmp_path / 'source-profile.json'), capsule_pin=pin(tmp_path / 'capsule.json'))
    result['owner_client_pin']['sha256'] = worker.REQUIRED_CLIENT_SHA256
    return result


def sealed(raw, *, seal=True):
    descriptor = os.memfd_create('private-M1a-inert-test', os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    try:
        assert os.write(descriptor, raw) == len(raw)
        if seal:
            fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, worker.SEALS)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def test_actual_fully_sealed_inert_request_roundtrip_and_write_refusal(tmp_path):
    value = inert_request(tmp_path)
    raw = worker.canonical(value)
    descriptor = sealed(raw)
    try:
        assert worker.read_sealed_request(descriptor, hashlib.sha256(raw).hexdigest()) == value
        with pytest.raises(OSError):
            os.pwrite(descriptor, b'x', 0)
        assert fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & worker.SEALS == worker.SEALS
    finally:
        os.close(descriptor)


@pytest.mark.parametrize('seal,hash_value', [(False, None), (True, 'f' * 64)])
def test_real_unsealed_or_wrong_digest_request_refuses(tmp_path, seal, hash_value):
    raw = worker.canonical(inert_request(tmp_path))
    descriptor = sealed(raw, seal=seal)
    try:
        with pytest.raises(worker.OfflineRefusal):
            worker.read_sealed_request(descriptor, hash_value or hashlib.sha256(raw).hexdigest())
    finally:
        os.close(descriptor)


@pytest.mark.parametrize('cap', [True, 1.0, 0, 2])
def test_cap1_does_not_accept_boolean_float_or_other_integer(tmp_path, cap):
    value = inert_request(tmp_path)
    value['attempt_cap'] = cap
    with pytest.raises(worker.OfflineRefusal, match='offline_one_shot_cap1_only'):
        worker.validate_request(value)


def test_closed_request_rejects_provider_and_extra_authority_fields(tmp_path):
    value = inert_request(tmp_path)
    value['provider_dispatch_allowed'] = True
    with pytest.raises(worker.OfflineRefusal):
        worker.validate_request(value)
    value = inert_request(tmp_path)
    value['native_lease'] = {'claimed': True}
    with pytest.raises(worker.OfflineRefusal, match='closed_private_request_schema_required'):
        worker.validate_request(value)


def test_default_off_main_does_not_read_invalid_descriptor(capsys):
    assert worker.main(['--private-m1-request-fd', '999999']) == 2
    error = worker.decode(worker.canonical(__import__('json').loads(capsys.readouterr().err)))
    assert error == dict(schema='private-M1a-offline-worker-failure/v1', error_type='OfflineRefusal',
                         provider_calls=0, native_qualified=False)


def pipe_writer(script):
    compile(script, "<private-M1a-pipe-writer>", "exec")
    return subprocess.Popen([sys.executable, '-I', '-B', '-c', script], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_complete_original_stdin_keeps_all_real_fragments_until_eof():
    child = pipe_writer(r"import os,time; os.write(1,b'first '); time.sleep(.03); os.write(1,b'second\n')")
    try:
        assert worker.read_complete_stdin(child.stdout.fileno(), time.monotonic() + 4) == b'first second\n'
        assert child.wait(timeout=2) == 0
    finally:
        child.communicate(timeout=2)


def test_original_absolute_stdin_clock_refuses_open_pipe_without_eof():
    child = pipe_writer("import os,time; os.write(1,b'prefix'); time.sleep(.2)")
    try:
        started = time.monotonic()
        with pytest.raises(worker.OfflineRefusal, match='original_stdin_work_deadline_expired'):
            worker.read_complete_stdin(child.stdout.fileno(), started + 3.05)
    finally:
        child.communicate(timeout=2)


def test_real_empty_stdin_eof_cannot_pass_complete_request():
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    try:
        with pytest.raises(worker.OfflineRefusal, match='complete_original_stdin_required'):
            worker.read_complete_stdin(read_fd, time.monotonic() + 4)
    finally:
        os.close(read_fd)


def test_offline_observation_is_exclusive_private_and_exact(tmp_path):
    directory = tmp_path / 'receipts'
    directory.mkdir(mode=0o700)
    value = dict(metadata_only=True, completion_authority=False, native_qualified=False)
    pin = worker.durable_observation(directory, value)
    assert worker.decode(worker.read_pin(pin, maximum=worker.MAX_RECEIPT, private=True)) == value
    assert (directory / 'offline-ACK.json').stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        worker.durable_observation(directory, value)


def inert_ACK_metadata(tmp_path):
    request = inert_request(tmp_path)
    context = dict(portal_task_id=request['private_task_alias'], database_task_cid=request['database_task_cid'],
        database_attempt_number=1, portal_attempt=1, allocation_session_id=request['allocation_session_id'],
        workspace=request['workspace'], source_generation=request['source_generation'],
        work_deadline_monotonic=1000.0, native_deadline_monotonic=1060.0,
        request_sha256='1' * 64, context_sha256='2' * 64, nonce='3' * 48,
        owner_pid=20, owner_birth=100, worker_pid=21, worker_birth=101, worker_process_group=21,
        worker_source_sha256='0' * 64, owner_source_sha256=worker.REQUIRED_CLIENT_SHA256,
        source_bindings_sha256='0' * 64, capsule_sha256='0' * 64,
        database_attempt_id='inert-attempt', database_claim_id='inert-claim', coordination_lease_id='inert-coordination',
        owner_session_id='inert-owner', fencing_token='inert-fence', fence_epoch=1,
        root_lease_id='inert-root', phase_lease_id='inert-phase',
        disk_native_lease_id='inert-disk-native', disk_reservation_id='inert-disk')
    clients, owners, timings = [], [], []
    for sequence, stage in enumerate(worker.STAGES, 1):
        started = 100.0 + sequence
        response = dict(schema='grok-production-owner-currentness/v1', status='current', sequence=sequence, stage=stage,
            **{key: context[key] for key in ('request_sha256', 'context_sha256', 'nonce', 'owner_pid',
                'owner_birth', 'worker_pid', 'worker_birth', 'worker_process_group',
                'work_deadline_monotonic', 'native_deadline_monotonic')}, observed_monotonic=started + .01)
        common = dict(sequence=sequence, stage=stage, request_sha256=context['request_sha256'],
            context_sha256=context['context_sha256'], response_sha256=hashlib.sha256(worker.canonical(response)).hexdigest(),
            client_pid=21, client_birth=101, client_process_group=21, owner_pid=20, owner_birth=100,
            observed_monotonic=started + .01)
        clients.append(dict(common, owner_process_group=20, received_monotonic=started + .02,
                            actual_kernel_peer_owner_verified=True, observation_is_not_native_authority=True))
        owners.append(dict(common, source_bindings_sha256='0' * 64,
            claim_binding={key: context[key] for key in checker.CLAIM_FIELDS},
            resource_binding={key: context[key] for key in checker.RESOURCE_FIELDS}))
        timings.append(dict(stage=stage, started_monotonic=started, returned_monotonic=started + .03))
    observation = dict(schema=worker.RECEIPT_SCHEMA, private_task_alias=request['private_task_alias'],
        database_task_cid=request['database_task_cid'], allocation_session_id=request['allocation_session_id'],
        source_generation=request['source_generation'], sealed_request_sha256=hashlib.sha256(worker.canonical(request)).hexdigest(),
        request_sha256=context['request_sha256'], context_sha256=context['context_sha256'],
        worker_source_sha256='0' * 64, owner_source_sha256=worker.REQUIRED_CLIENT_SHA256,
        source_bindings_sha256='0' * 64, capsule_sha256='0' * 64,
        original_stdin_bytes=12, original_stdin_sha256='4' * 64,
        owner_rpc_receipts=clients, call_timings=timings, response_monotonic=104.0,
        offline_only=True, provider_calls=0, allocation_capture_written=False, metadata_only=True,
        completion_authority=False, native_qualified=False, Lean_admitted=False)
    return request, context, observation, owners


def test_inert_metadata_join_never_grants_native_or_task_acceptance(tmp_path):
    request, context, observation, owners = inert_ACK_metadata(tmp_path)
    result = checker.metadata_joins(request=request, context=context, observation=observation, owner_rows=owners,
        expected_stdin=dict(bytes=12, sha256='4' * 64))
    assert result['three_ACK_metadata_joins'] is True
    assert result['native_qualified'] is result['completion_authority'] is result['Lean_admitted'] is False


@pytest.mark.parametrize('change', ['sequence', 'digest', 'claim', 'deadline', 'authority', 'stdin'])
def test_metadata_tampering_cannot_become_offline_observation(tmp_path, change):
    request, context, observation, owners = inert_ACK_metadata(tmp_path)
    if change == 'sequence':
        observation['owner_rpc_receipts'][0]['sequence'] = True
    elif change == 'digest':
        owners[1]['response_sha256'] = 'f' * 64
    elif change == 'claim':
        owners[0]['claim_binding']['database_claim_id'] = 'another-claim'
    elif change == 'deadline':
        observation['response_monotonic'] = request['work_deadline_monotonic']
    elif change == 'authority':
        observation['native_qualified'] = True
    else:
        observation['original_stdin_sha256'] = 'f' * 64
    with pytest.raises(checker.OfflineCheckRefusal):
        checker.metadata_joins(request=request, context=context, observation=observation, owner_rows=owners,
        expected_stdin=dict(bytes=12, sha256='4' * 64))
