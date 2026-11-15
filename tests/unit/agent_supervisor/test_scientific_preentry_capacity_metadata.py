"""Authored SOURCE controls for negative DATA recognition only; not executed.

No scheduler, census, Popen, actual request, grant, health or service action is
performed. A recognized fixture still explicitly requires original custody.
"""
import copy

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import scientific_executor as executor


@pytest.fixture
def negative():
    root = '/fixture/proof/cycle-00000200-' + 'a' * 32
    request_ref = dict(path=root+'/request.json', bytes=123, sha256='b'*64)
    profile = dict(proof_root='/fixture/proof',
        owner_source=dict(path='/fixture/owner.py', sha256='c'*64),
        driver_source=dict(path='/fixture/driver.py', sha256='d'*64))
    request = dict(schema='legal-ir-original-native-control-request/v1',
        purpose='native_control_metadata', request_id='scientific-executor:'+root.rsplit('/', 1)[1],
        resource_ledger=str(executor.ORIGINAL_RESOURCE_LEDGER), attempt_directory=root+'/attempt',
        cpu_slots=1, memory_mb=1536, child_process_slots=8, storage_bytes=384*1024**2,
        timeout_seconds=300, source_sha256={'/fixture/owner.py':'c'*64, '/fixture/driver.py':'d'*64},
        database_preimage=dict(path='/fixture/control.duckdb', bytes=100, sha256='e'*64))
    intent = dict(kind='genuine_scoped_idle', cycle_number=200, sequence=38,
                  automatic_retry_admitted=False, request_ref=request_ref)
    terminal = dict(schema='legal-ir-original-native-control-terminal/v1',
        request_sha256=request_ref['sha256'], completed=False, error_type='DaemonResourceError',
        error='storage reservation capacity exceeded', storage_status='not_entered',
        control_effects='not_started', child_launch_attempted=False, child_reaped=False,
        child_returncode=None, source_READ_closed=True, original_native_compute_closed=True,
        absolute_resource_operation_deadline_closed=True,
        task_completion_authority_granted_by_resource_owner=False, cleanup_errors=[], received_signals=[],
        owner_pid=1234, owner_birth='123456', owner_boot_id='00000000-0000-4000-8000-000000000001',
        database_preimage_before_grants=request['database_preimage'])
    terminal['reservation'] = dict(authority='one_actual_child_of_existing_phase_no_additional_host_root',
        cleanup_error=None, enforcement='fixture cooperative observation', external_file_growth_monitored=False,
        last_local_usage=None, ledger_path=request['resource_ledger'], profile='parented-phase-sampled/v1',
        record=None, reservation_id='f'*32, resource_lease=None,
        roots=[dict(path=str(p),device=1,inode=i+1)for i,p in enumerate(executor.ORIGINAL_NAMED_ROOTS)],
        scheduler_lane='orchestration', schema='daemon-resource-reservations-v1', status='not_entered',
        storage_limit_bytes=220000000000, workload='canonical_parented_phase')
    terminal['resource_capacity_observation'] = dict(schema='daemon-resource-capacity-observation/v1',
        observed_apparent_bytes=174941688482, outstanding_full_reservations_bytes=44656169792,
        additional_requested_bytes=402653184, charged_bytes=220000511458, limit_bytes=220000000000,
        minimum_filesystem_free_bytes=563440861184, storage_capacity_deficit_bytes=511458,
        filesystem_free_deficit_bytes=0, scope='named_roots_only_apparent_bytes',
        storage_capacity_exceeded=True, filesystem_free_insufficient=False)
    return terminal, request_ref, request, profile, intent, root


def test_recognized_negative_has_no_health_retry_or_dispatch_credit(negative):
    result = executor._closed_capacity_terminal_metadata(*negative)
    assert result['cycle_number'] == 200 and result['attempted_health_sequence'] == 38
    assert result['capacity_deficit_bytes'] == 511458
    for key in ('health_credit', 'completion_authority', 'dispatch_authority', 'automatic_retry_admitted'):
        assert result[key] is False
    assert result['original_Popen_wait_source_and_no_claim_proof_still_required'] is True


@pytest.mark.parametrize('changes', [
    {'error_type':'SourceLeaseTimeout'}, {'error':'disk ledger lock unavailable'},
    {'storage_status':'retained'}, {'child_launch_attempted':True},
    {'child_reaped':True}, {'source_READ_closed':False},
    {'original_native_compute_closed':False}, {'cleanup_errors':['unknown']},
    {'execution_request':{}}, {'owner_birth':''},
])
def test_unknown_started_or_unclean_terminal_is_not_capacity_deferral(negative, changes):
    values = list(copy.deepcopy(negative))
    values[0].update(changes)
    with pytest.raises(ValueError):
        executor._closed_capacity_terminal_metadata(*values)


@pytest.mark.parametrize('key,value', [
    ('record', {}), ('resource_lease', {}), ('status','retained'),
    ('last_local_usage', {}), ('scheduler_lane','validation'), ('child', {'pid':1234}),
])
def test_entered_or_foreign_reservation_refuses(negative, key, value):
    values = list(copy.deepcopy(negative))
    values[0]['reservation'][key] = value
    with pytest.raises(ValueError):
        executor._closed_capacity_terminal_metadata(*values)


@pytest.mark.parametrize('key,value', [
    ('charged_bytes',220000511459), ('limit_bytes',220000000001),
    ('additional_requested_bytes',402653185), ('storage_capacity_deficit_bytes',1),
    ('filesystem_free_insufficient',True), ('observed_apparent_bytes',True),
])
def test_capacity_shape_and_math_cannot_weaken_original_limit(negative, key, value):
    values = list(copy.deepcopy(negative))
    values[0]['resource_capacity_observation'][key] = value
    with pytest.raises(ValueError):
        executor._closed_capacity_terminal_metadata(*values)


def test_request_is_neither_replayed_nor_relabelled_as_execution(negative):
    values = list(copy.deepcopy(negative))
    values[4]['kind'] = 'approved_execution'
    with pytest.raises(ValueError):
        executor._closed_capacity_terminal_metadata(*values)
    values = list(copy.deepcopy(negative))
    values[2]['storage_bytes'] -= 1
    with pytest.raises(ValueError):
        executor._closed_capacity_terminal_metadata(*values)
