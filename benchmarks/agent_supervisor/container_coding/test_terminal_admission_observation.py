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


@pytest.fixture
def attributed_refusal(tmp_path):
    """A real sampler and scheduler keep the host/container distinction."""
    proc, cgroup = tmp_path / 'proc', tmp_path / 'cgroup'
    (proc / 'self').mkdir(parents=True)
    (proc / 'pressure').mkdir()
    (cgroup / 'PRIVATE_CONTAINER').mkdir(parents=True)
    (proc / 'meminfo').write_text('MemTotal: 8388608 kB\nMemAvailable: 8388608 kB\n')
    (proc / 'self/cgroup').write_text('0::/PRIVATE_CONTAINER\n')
    def sample():
        return safety.collect_proof_host_resources(proc_root=proc, cgroup_root=cgroup)
    owner = schedulers.GlobalResourceScheduler(schedulers.ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / 'attributed-resources.json', proof_resource_sampler=sample,
        lane_reservations={}, auto_renew_leases=False))
    (proc / 'pressure/memory').write_text('full avg10=10.00 avg60=0.00 avg300=0.00 total=1\n')
    (cgroup / 'PRIVATE_CONTAINER/memory.pressure').write_text(
        'full avg10=4.00 avg60=0.00 avg300=0.00 total=1\n')
    with pytest.raises(schedulers.LeaseTimeoutError) as caught:
        owner.acquire('orchestration', memory_mb=512, timeout=0, request_id='PRIVATE_REQUEST')
    (proc / 'pressure/memory').write_text('full avg10=0.00\n')
    return caught.value


def test_actual_pressure_attribution_preserves_host_and_container_samples(attributed_refusal, monkeypatch):
    test_consumer_preserves_admission_sample_after_pressure_changes(attributed_refusal, monkeypatch)
    result = diagnostics.collect_failure_admission(attributed_refusal)
    sources = result['last_sample']['pressure_sources']
    assert sources['samples'][0]['memory'] == {'avg10': 10., 'status': 'observed'}
    assert sources['samples'][1]['memory'] == {'avg10': 4., 'status': 'observed'}
    assert sources['samples'][1]['depth'] == 0
    assert sources['samples'][2]['memory'] == {'avg10': None, 'status': 'unavailable'}
    assert sources['omitted_cgroup_scopes'] == 0
    assert 'PRIVATE' not in json.dumps(result)


def test_full_driver_exports_same_sample_pressure_sources(attributed_refusal, monkeypatch):
    test_full_driver_retains_attached_sample_separate_from_later_host_sample(attributed_refusal, monkeypatch)
    result = driver._failure_diagnostics(attributed_refusal, phase='initial_context')
    assert result['failure_admission']['last_sample']['pressure_sources']['samples'][0]['memory']['avg10'] == 10


def test_qualifier_exports_same_sample_pressure_sources(attributed_refusal, tmp_path, monkeypatch):
    test_qualifier_executes_error_export_with_actual_native_timeout(attributed_refusal, tmp_path, monkeypatch)
    result = json.loads((tmp_path / 'result.json').read_bytes())
    assert result['failure_admission']['last_sample']['pressure_sources']['samples'][0]['memory']['avg10'] == 10


@pytest.mark.parametrize('mutation', [
    'extra', 'too_many', 'empty', 'scope', 'depth', 'bool_depth', 'path',
    'negative', 'nan', 'over100', 'missing_value', 'false_zero', 'status',
    'aggregate', 'omitted_count', 'omitted_max', 'schema',
    'schema_subclass', 'scope_subclass',
])
def test_malformed_pressure_attribution_is_not_exported(attributed_refusal, mutation):
    class PretendString(str):
        def __eq__(self, other):return True
        def __ne__(self, other):return False
    record = deepcopy(attributed_refusal.admission_observation)
    source = record['last_sample']['pressure_sources']
    sample = source['samples'][1]
    metric = sample['memory']
    if mutation == 'extra':source['PRIVATE'] = 'PRIVATE'
    elif mutation == 'too_many':source['samples'] *= 10
    elif mutation == 'empty':source['samples'] = []
    elif mutation == 'scope':sample['scope'] = 'PRIVATE'
    elif mutation == 'depth':sample['depth'] = 1
    elif mutation == 'bool_depth':sample['depth'] = False
    elif mutation == 'path':sample['path'] = 'PRIVATE'
    elif mutation == 'negative':metric['avg10'] = -1
    elif mutation == 'nan':metric['avg10'] = float('nan')
    elif mutation == 'over100':metric['avg10'] = 101
    elif mutation == 'missing_value':metric['avg10'] = None
    elif mutation == 'false_zero':metric.update(status='unavailable', avg10=0)
    elif mutation == 'status':metric['status'] = 'PRIVATE'
    elif mutation == 'aggregate':metric['avg10'] = 50
    elif mutation == 'omitted_count':source['omitted_cgroup_scopes'] = 1
    elif mutation == 'omitted_max':source['omitted_maxima']['memory'] = 1
    elif mutation == 'schema_subclass':source['schema'] = PretendString('PRIVATE_SCHEMA')
    elif mutation == 'scope_subclass':sample['scope'] = PretendString('PRIVATE_SCOPE')
    else:source['schema'] = 'PRIVATE'
    attributed_refusal.admission_observation = record
    result = diagnostics.collect_failure_admission(attributed_refusal)
    assert result['status'] == 'unavailable'
    assert 'PRIVATE' not in json.dumps(result, allow_nan=False)


def test_maximum_pressure_inventory_fits_existing_export_bound(refused):
    # Exercise the largest schema inventory and long scalar representations.
    metrics = ('memory', 'cpu', 'io')
    maximum = 99.12345678901234
    record = deepcopy(refused.admission_observation)
    record['last_sample']['host'].update({m + '_stall_percent': maximum for m in metrics})
    record['last_sample']['pressure_sources'] = dict(schema='proof-pressure-sources@1',
        samples=[dict(scope='host' if i == 0 else 'cgroup', depth=None if i == 0 else i - 1,
            **{m: dict(avg10=maximum, status='observed') for m in metrics}) for i in range(9)],
        omitted_cgroup_scopes=diagnostics.MAX_NUMBER,
        omitted_maxima={m: maximum for m in metrics})
    refused.admission_observation = record
    result = diagnostics.collect_failure_admission(refused)
    assert result['status'] == 'observed'
    assert len(json.dumps(result, allow_nan=False).encode()) <= diagnostics.MAX_BYTES
