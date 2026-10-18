"""Private offline ACK metadata joins and FIRST-call handoff refusal observer.

Nothing in this module creates native authority or accepts a completed task.
The first-call function is a composition seam; the admitted M1 runner which
owns that call does not exist yet. Do not invoke it after normal Portal has
already invoked after_stream_return.
"""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import ipfs_accelerate_py.cli_runtime.grok_private_m1_offline_worker as worker

CHECK_SCHEMA = 'private-M1a-offline-check/v1'
CLIENT_FIELDS = {'sequence', 'stage', 'request_sha256', 'context_sha256', 'owner_pid',
    'owner_birth', 'owner_process_group', 'client_pid', 'client_birth', 'client_process_group',
    'observed_monotonic', 'received_monotonic', 'response_sha256',
    'actual_kernel_peer_owner_verified', 'observation_is_not_native_authority'}
OWNER_FIELDS = {'sequence', 'stage', 'request_sha256', 'context_sha256', 'response_sha256',
    'client_pid', 'client_birth', 'client_process_group', 'owner_pid', 'owner_birth',
    'observed_monotonic', 'source_bindings_sha256', 'claim_binding', 'resource_binding'}
CLAIM_FIELDS = ('database_task_cid', 'database_attempt_id', 'database_claim_id',
    'database_attempt_number', 'coordination_lease_id', 'owner_session_id', 'fencing_token', 'fence_epoch')
RESOURCE_FIELDS = ('root_lease_id', 'phase_lease_id', 'disk_native_lease_id', 'disk_reservation_id')
MISSING_CAPTURE = 'handoff_independent_store_capture_readback_missing'


class OfflineCheckRefusal(RuntimeError):
    pass


def need(value, reason):
    if not value:
        raise OfflineCheckRefusal(reason)


def exact(left, right):
    """JSON booleans must never satisfy integer identities or sequence values."""
    return type(left) is type(right) and left == right


def metadata_joins(*, request, context, observation, owner_rows, expected_stdin):
    """Validate additive copies only; this never grants authority from JSON."""
    worker.validate_request(request)
    need(type(context) is dict and type(observation) is dict and
         type(owner_rows) is list and len(owner_rows) == len(worker.STAGES), 'bounded_actual_metadata_required')
    expected_fields = {'schema', 'private_task_alias', 'database_task_cid', 'sealed_request_sha256',
        'request_sha256', 'context_sha256', 'allocation_session_id', 'source_generation',
        'worker_source_sha256', 'owner_source_sha256', 'source_bindings_sha256', 'capsule_sha256',
        'original_stdin_bytes', 'original_stdin_sha256', 'owner_rpc_receipts', 'call_timings',
        'response_monotonic', 'offline_only', 'provider_calls', 'allocation_capture_written',
        'metadata_only', 'completion_authority', 'native_qualified', 'Lean_admitted'}
    need(set(observation) == expected_fields and observation['schema'] == worker.RECEIPT_SCHEMA,
         'closed_offline_observation_required')
    for field, value in dict(offline_only=True, provider_calls=0, allocation_capture_written=False,
            metadata_only=True, completion_authority=False, native_qualified=False, Lean_admitted=False).items():
        need(exact(observation[field], value), 'offline_observation_cannot_claim_authority')
    for field in ('private_task_alias', 'database_task_cid', 'allocation_session_id', 'source_generation'):
        need(exact(observation[field], request[field]), 'offline_request_identity_changed')
    need(observation['sealed_request_sha256'] == hashlib.sha256(worker.canonical(request)).hexdigest(),
         'sealed_request_metadata_digest_changed')
    for field in ('request_sha256', 'context_sha256'):
        need(exact(observation[field], context[field]), 'offline_context_commitment_changed')
    for field, pin in (('worker_source_sha256', 'worker_pin'), ('owner_source_sha256', 'owner_client_pin'),
                       ('source_bindings_sha256', 'source_profile_pin'), ('capsule_sha256', 'capsule_pin')):
        need(exact(observation[field], request[pin]['sha256']) and
             exact(context[field], request[pin]['sha256']), 'offline_source_binding_changed')
    for field, value in dict(portal_task_id=request['private_task_alias'], database_task_cid=request['database_task_cid'],
            database_attempt_number=1, portal_attempt=1, allocation_session_id=request['allocation_session_id'],
            workspace=request['workspace'], source_generation=request['source_generation'],
            work_deadline_monotonic=request['work_deadline_monotonic'],
            native_deadline_monotonic=request['native_deadline_monotonic']).items():
        need(exact(context[field], value), 'offline_actual_context_request_join_changed')
    need(type(observation['original_stdin_bytes']) is int and
         0 < observation['original_stdin_bytes'] <= worker.MAX_STDIN and
         type(observation['original_stdin_sha256']) is str and
         worker.re.fullmatch('[0-9a-f]{64}', observation['original_stdin_sha256']),
         'complete_original_stdin_observation_required')
    need(type(expected_stdin) is dict and set(expected_stdin) == {'bytes', 'sha256'} and
         exact(expected_stdin['bytes'], observation['original_stdin_bytes']) and
         exact(expected_stdin['sha256'], observation['original_stdin_sha256']),
         'independent_original_stdin_commitment_changed')
    clients, timings = observation['owner_rpc_receipts'], observation['call_timings']
    need(type(clients) is list and type(timings) is list and
         len(clients) == len(timings) == len(worker.STAGES), 'exact_three_offline_ACKs_required')
    previous_returned = 0
    for sequence, (stage, client, owner, timing) in enumerate(zip(worker.STAGES, clients, owner_rows, timings), 1):
        need(type(client) is dict and set(client) == CLIENT_FIELDS and
             type(owner) is dict and set(owner) == OWNER_FIELDS and
             type(timing) is dict and set(timing) == {'stage', 'started_monotonic', 'returned_monotonic'},
             'closed_ACK_observation_rows_required')
        for row in (client, owner):
            need(exact(row['sequence'], sequence) and exact(row['stage'], stage) and
                 exact(row['request_sha256'], context['request_sha256']) and
                 exact(row['context_sha256'], context['context_sha256']), 'ACK_order_or_context_changed')
            for key, context_key in (('owner_pid', 'owner_pid'), ('owner_birth', 'owner_birth'),
                    ('client_pid', 'worker_pid'), ('client_birth', 'worker_birth'),
                    ('client_process_group', 'worker_process_group')):
                need(exact(row[key], context[context_key]), 'ACK_actual_process_join_changed')
        need(exact(client['actual_kernel_peer_owner_verified'], True) and
             exact(client['observation_is_not_native_authority'], True) and
             type(client['owner_process_group']) is int and client['owner_process_group'] > 1,
             'ACK_must_retain_kernel_peer_and_non_authority_labels')
        need(exact(owner['source_bindings_sha256'], request['source_profile_pin']['sha256']) and
             owner['claim_binding'] == {key: context[key] for key in CLAIM_FIELDS} and
             owner['resource_binding'] == {key: context[key] for key in RESOURCE_FIELDS} and
             set(owner['claim_binding']) == set(CLAIM_FIELDS) and
             set(owner['resource_binding']) == set(RESOURCE_FIELDS), 'ACK_claim_or_resource_join_changed')
        for field in CLAIM_FIELDS:
            need(exact(owner['claim_binding'][field], context[field]), 'ACK_typed_claim_join_changed')
        response = dict(schema='grok-production-owner-currentness/v1', status='current', sequence=sequence, stage=stage,
            **{key: context[key] for key in ('request_sha256', 'context_sha256', 'nonce', 'owner_pid',
                'owner_birth', 'worker_pid', 'worker_birth', 'worker_process_group',
                'work_deadline_monotonic', 'native_deadline_monotonic')},
            observed_monotonic=owner['observed_monotonic'])
        digest = hashlib.sha256(worker.canonical(response)).hexdigest()
        need(exact(client['response_sha256'], digest) and exact(owner['response_sha256'], digest) and
             exact(client['observed_monotonic'], owner['observed_monotonic']), 'ACK_response_digest_join_changed')
        for value in (timing['started_monotonic'], timing['returned_monotonic'], owner['observed_monotonic'],
                      client['received_monotonic'], observation['response_monotonic']):
            need(type(value) in (int, float) and worker.math.isfinite(value), 'finite_ACK_clocks_required')
        need(exact(timing['stage'], stage) and previous_returned <= timing['started_monotonic'] <=
             owner['observed_monotonic'] <= client['received_monotonic'] <= timing['returned_monotonic'] <=
             observation['response_monotonic'] < request['work_deadline_monotonic'] - 3.0,
             'ACK_original_clock_or_monotonic_order_changed')
        previous_returned = timing['returned_monotonic']
    return dict(schema=CHECK_SCHEMA, three_ACK_metadata_joins=True, provider_calls=0,
                metadata_only=True, completion_authority=False, native_qualified=False, Lean_admitted=False)


def first_after_stream_missing_capture(*, handoff, completed, request, observation_pin, expected_stdin):
    """Own the FIRST real after_stream_return call in a future private runner.

    Ordinary Portal already owns this call and must not call this helper after
    its own callback. This function never marks the task or attempt complete.
    It neither closes unrelated owners nor suppresses different refusals.
    """
    root = Path(request['accelerator_root'])
    worker.validate_request(request)
    module_name = 'ipfs_accelerate_py.cli_runtime.grok_production_handoff'
    module = sys.modules.get(module_name)
    need(type(module) is types.ModuleType and module.__file__ == module.__spec__.origin ==
         str(root / 'ipfs_accelerate_py/cli_runtime/grok_production_handoff.py') and
         type(handoff) is module.ProductionProviderHandoff, 'real_normal_handoff_required')
    handoff._lifetime()
    need(handoff._prepared is True and handoff._closed is False and handoff._completed is False and
         handoff._capture is None and type(completed) is subprocess.CompletedProcess and
         type(completed.returncode) is int and completed.returncode == 0 and
         handoff._process is not None and handoff._process.returncode == 0 and
         tuple(completed.args) == handoff.command, 'first_actual_successful_stream_return_required')
    for path, digest in handoff._pins.items():
        need(handoff.factory.source_digest(path) == digest, 'actual_complete_producer_source_changed')
    for own in (__file__, worker.__file__):
        need(handoff._pins.get(str(Path(own))) == handoff.factory.source_digest(own),
             'offline_checker_and_worker_must_be_in_actual_producer_pins')
    need(type(handoff.allocation_store) is handoff._allocation_module.AllocationStore and
         handoff.allocation_store._ready is True and
         str(handoff.allocation_store.path) == handoff.binding['allocation_store_path'],
         'genuine_initialized_allocation_store_required')
    need(request['allocation_session_id'] == handoff.binding['allocation_session_id'] and
         request['database_task_cid'] == handoff.binding['database_task_cid'] and
         request['private_task_alias'] == handoff.binding['portal_task_id'] and
         request['work_deadline_monotonic'] == handoff.binding['work_deadline_monotonic'] and
         request['native_deadline_monotonic'] == handoff.binding['native_deadline_monotonic'],
         'actual_handoff_private_request_join_required')
    directory = Path(request['receipt_directory'])
    need(directory.resolve(strict=True) == directory and handoff.owner._disk.directory in directory.parents and
         handoff.server.context_path.parent == directory, 'actual_charged_private_observation_directory_required')
    context = handoff.core.validate_context(handoff.core.decode(handoff.server._context_bytes))
    observation = worker.decode(worker.read_pin(observation_pin, maximum=worker.MAX_RECEIPT, private=True))
    need(Path(observation_pin['path']) == Path(request['receipt_directory']) / 'offline-ACK.json',
         'exact_actual_offline_observation_path_required')
    need(time.monotonic() < request['native_deadline_monotonic'], 'original_native_callback_clock_expired')
    refusal_observed = False
    try:
        handoff.after_stream_return(completed)
    except module.ProductionHandoffRefusal as error:
        need(error.args == (MISSING_CAPTURE,), 'different_handoff_refusal_is_not_offline_success')
        refusal_observed = True
    need(refusal_observed and handoff._closed is True and handoff.server.closed is True and
         handoff._completed is False and handoff._capture is None,
         'genuine_missing_capture_refusal_and_closed_listener_required')
    result = metadata_joins(request=request, context=context, observation=observation,
                            owner_rows=handoff.server.observations(), expected_stdin=expected_stdin)
    # The public method returns {} for no row AND for missing DuckDB. Record
    # its real result; {} alone is deliberately not a durable absence proof.
    row = handoff.allocation_store.get_cli_session(request['allocation_session_id'], 'grok_cli')
    need(type(row) is dict and not row, 'offline_worker_must_not_produce_an_allocation_capture')
    need(time.monotonic() < request['native_deadline_monotonic'], 'original_native_readback_clock_expired')
    for path, digest in handoff._pins.items():
        need(handoff.factory.source_digest(path) == digest, 'actual_producer_source_changed_after_readback')
    need(time.monotonic() < request['native_deadline_monotonic'], 'original_native_final_check_clock_expired')
    result.update(actual_first_handoff_missing_capture_refusal=True, actual_listener_closed=True,
        public_allocation_readback_empty=True, durable_allocation_absence_proved=False,
        actual_capture_accepted=False, task_completion_accepted=False)
    return result
