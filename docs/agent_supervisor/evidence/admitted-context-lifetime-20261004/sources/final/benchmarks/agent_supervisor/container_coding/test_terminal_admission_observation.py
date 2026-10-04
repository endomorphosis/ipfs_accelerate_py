"""Actual shared scheduler errors reach both supervisor diagnostic consumers."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import signal

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_resource_diagnostics as diagnostics
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from benchmarks.agent_supervisor.container_coding import terminal_source384_qualification as qualifier
from ipfs_datasets_py.optimizers.logic_theorem_optimizer import resource_scheduler as schedulers
from ipfs_datasets_py.optimizers.logic_theorem_optimizer import proof_resource_safety as safety


@pytest.fixture
def refused(tmp_path):
    healthy = safety.ProofHostResources(8, 8192, 8192)
    current = [healthy]
    owner = schedulers.GlobalResourceScheduler(schedulers.ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / 'resources.json', proof_resource_sampler=lambda: current[0],
        lane_reservations={}, auto_renew_leases=False))
    current[0] = replace(healthy, memory_stall_percent=10)
    with pytest.raises(schedulers.LeaseTimeoutError) as caught:
        owner.acquire('orchestration', memory_mb=512, timeout=0, request_id='PRIVATE_REQUEST')
    current[0] = healthy
    return caught.value


def test_consumer_preserves_admission_sample_after_pressure_changes(refused, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('attached observation collection must not sample or open an owner')
    monkeypatch.setattr(safety, 'collect_proof_host_resources', forbidden)
    monkeypatch.setattr(schedulers, 'get_global_resource_scheduler', forbidden)
    result = diagnostics.collect_failure_admission(refused)
    assert result['status'] == 'observed'
    assert result['last_sample']['host']['memory_stall_percent'] == 10
    assert result['primary_gate']['reason'] == 'proof_memory_stall'
    assert result['complete_admission_decision'] is result['causal_proof'] is False
    assert 'PRIVATE' not in json.dumps(result)
    assert len(json.dumps(result)) < diagnostics.MAX_BYTES


@pytest.mark.parametrize('mutation', ['extra', 'nan', 'bool', 'huge', 'reason', 'authority', 'list'])
def test_malformed_attached_observation_is_unavailable_without_raw_export(refused, mutation):
    record = deepcopy(refused.admission_observation)
    if mutation == 'extra':record['PRIVATE'] = ['PRIVATE'] * 10000
    elif mutation == 'reason':record['primary_gate']['reason'] = 'PRIVATE_REASON'
    elif mutation == 'authority':record['complete_admission_decision'] = True
    elif mutation == 'list':record['last_sample']['host'] = []
    else:record['last_sample']['host']['available_memory_mb'] = {'nan':float('nan'), 'bool':True, 'huge':2**100}[mutation]
    refused.admission_observation = record
    result = diagnostics.collect_failure_admission(refused)
    assert result['status'] == 'unavailable'
    assert 'PRIVATE' not in json.dumps(result, allow_nan=False)


def test_older_scheduler_and_nonadmission_errors_are_explicitly_unavailable():
    assert diagnostics.collect_failure_admission(schedulers.LeaseTimeoutError('PRIVATE'))['reason'] == 'no_attached_observation'
    assert diagnostics.collect_failure_admission(ValueError('PRIVATE'))['reason'] == 'no_native_admission_error'


def test_full_driver_retains_attached_sample_separate_from_later_host_sample(refused, monkeypatch):
    monkeypatch.setattr(safety, 'collect_proof_host_resources', lambda:safety.ProofHostResources(8,8192,8192))
    monkeypatch.setattr(diagnostics, 'collect_failure_scheduler', lambda:{'status':'unavailable'})
    result = driver._failure_diagnostics(refused, phase='initial_context')
    assert result['failure_resources']['memory_stall_percent'] == 0
    assert result['failure_admission']['last_sample']['host']['memory_stall_percent'] == 10


def test_qualifier_executes_error_export_with_actual_native_timeout(refused, tmp_path, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as preparation
    from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as source
    output = tmp_path / 'result.json'; signals = []
    monkeypatch.setattr(signal, 'signal', lambda *args:None)
    monkeypatch.setattr(signal, 'setitimer', lambda *args:signals.append(args))
    monkeypatch.setattr(preparation, 'prepare', lambda **kwargs:{})
    def initial(**kwargs):raise refused
    monkeypatch.setattr(preparation, 'initial_context', initial)
    monkeypatch.setattr(source, '_pins', lambda:{'authored_fixture':True})
    monkeypatch.setattr(safety, 'collect_proof_host_resources', lambda:safety.ProofHostResources(8,8192,8192))
    monkeypatch.setattr(diagnostics, 'collect_failure_scheduler', lambda:{'status':'unavailable'})
    native_open, native_read = Path.open, Path.read_text
    def opened(path, *args, **kwargs):
        return native_open(output if str(path) == qualifier.RESULT_PATH else path, *args, **kwargs)
    def read(path, *args, **kwargs):
        if str(path) == '/opt/ipfs-supervisor/source384-public-instruction.md':return 'authored instruction'
        return native_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', opened); monkeypatch.setattr(Path, 'read_text', read)
    with pytest.raises(SystemExit) as ended:
        exec(compile(qualifier.CONTEXT_PROBE, 'context-probe', 'exec'), {})
    assert ended.value.code == 1
    result = json.loads(output.read_bytes())
    assert result['error_type'] == 'LeaseTimeoutError' and result['qualified'] is False
    assert result['failure_admission']['last_sample']['host']['memory_stall_percent'] == 10
    assert (signal.ITIMER_REAL, 0) in signals
    assert result['provider_calls'] == 0
