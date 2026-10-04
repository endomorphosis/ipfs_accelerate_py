"""Native finite TLC checks, exact evidence joins, and shared-owner isolation."""
from dataclasses import replace
from pathlib import Path
import shutil
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.proof import datasets_tla_tasks as tasks
from ipfs_accelerate_py.agent_supervisor.proof.datasets_kernel_tasks import make_datasets_lean_nat_task
from ipfs_accelerate_py.agent_supervisor.proof.datasets_hammer_tasks import make_datasets_smt_task
from ipfs_accelerate_py.agent_supervisor.proof.datasets_prover_resources import open_datasets_prover_lease
from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_resources import (
    BundleProverSupervisor, ExecutionStatus, MultiProverResourceBudget,
    MultiProverResourceLease, ProverTaskExecutor,
)
from ipfs_datasets_py.logic.backends import process
from ipfs_datasets_py.logic.backends.tla import runners
from ipfs_datasets_py.logic.backends.installers import state_model
from ipfs_datasets_py.logic.hammers import semantic_routing as routing
from ipfs_datasets_py.logic.ir_core.claims import FrozenMap
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceSchedulerConfig,
)


def budget():
    return MultiProverResourceBudget(cpu_slots=2, process_slots=2, thread_slots=2,
        memory_bytes=2048 * 1024**2, max_portfolio_width=2)


@pytest.fixture
def native(tmp_path):
    assert (state_model.expand_user_local_root() / 'tlc' / state_model.TLC_VERSION / 'tla2tools.jar').is_file()
    healthy = ProofHostResources(8, 8192, 8192)
    pressure = [healthy]
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / 'leases.json', proof_resource_sampler=lambda: pressure[0],
        total_cpu_slots=4, total_memory_mb=4096, total_child_process_slots=4,
        lane_reservations={}, proof_backoff_seconds=.03, poll_interval_seconds=.005))
    yield scheduler, pressure, healthy
    deadline = time.monotonic() + 3
    while scheduler.snapshot()['active_lease_count'] and time.monotonic() < deadline:
        time.sleep(.005)
    assert scheduler.snapshot()['active_lease_count'] == 0
    assert scheduler.snapshot()['waiting_request_count'] == 0


def execute(scheduler, task):
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        return ProverTaskExecutor(lease).execute(task)


@pytest.mark.parametrize('bound', [1, 2, 5, 16, 64])
def test_real_native_model_explores_all_generated_states_with_bounded_authority(native, bound):
    scheduler, _, _ = native
    task = tasks.make_datasets_tlc_counter_task(task_id=f'counter-{bound}', counter_bound=bound, invariant_max=bound)
    receipt = execute(scheduler, task)
    assert receipt.status is ExecutionStatus.SUCCEEDED, receipt.to_dict()
    payload = receipt.result
    assert payload['bindings_match'] and payload['model_check_passed']
    assert payload['native_checks'] == 1
    assert payload['max_steps'] == bound and payload['distinct_states'] == bound + 1
    assert payload['queue_empty'] and payload['typed_result']['authority'] == 'model_check'
    assert payload['typed_result']['translation_ceiling'] == 'bounded'
    assert payload['typed_result']['bounds']['max_steps'] == bound
    assert payload['model_check_receipt']['checked_liveness_properties'] == []
    assert 'CHECK_DEADLOCK FALSE' in payload['model_check_receipt']['configuration_text']
    assert all(payload[key] is False for key in ('proof_authority', 'kernel_authority',
        'source_semantics_verified', 'behavior_authority', 'execution_authority', 'completion_authority'))
    assert payload['native_runtime']['tlc_jar_sha256'] == state_model.TLC_SHA256
    assert payload['native_runtime']['java_major'] >= 11
    assert not task.deterministic and not receipt.cache_bypassed_execution


def test_real_counterexample_preserved_and_dependent_refused(native):
    scheduler, _, _ = native
    first = tasks.make_datasets_tlc_counter_task(task_id='violated', counter_bound=1, invariant_max=0)
    after = tasks.make_datasets_tlc_counter_task(task_id='after', counter_bound=1, invariant_max=1,
                                               dependencies=('violated',))
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        bundle = BundleProverSupervisor(lease).execute((first, after))
    failed, blocked = bundle.receipts
    assert failed.status is ExecutionStatus.FAILED
    assert failed.reasons == ('tlc_invariant_not_established',)
    assert failed.result['bindings_match'] and not failed.result['model_check_passed']
    assert failed.result['status'] == 'counterexample'
    assert failed.result['typed_result']['status'] == 'violated'
    states = failed.result['model_check_receipt']['counterexample']['states']
    assert [row['assignments']['count'] for row in states] == ['0', '1']
    assert blocked.status is ExecutionStatus.BLOCKED


def test_real_smt_lean_tlc_share_one_owner_with_bounded_native_probes(native, monkeypatch):
    scheduler, _, _ = native
    assert shutil.which('lean') and shutil.which('z3')
    captured = []
    original = process.SubprocessExecutor.execute
    def observe(self, invocation, cancellation=None):
        state = scheduler.snapshot()
        captured.append((state['active_root_lease_count'], state['active_child_lease_count'], invocation))
        return original(self, invocation, cancellation)
    monkeypatch.setattr(process.SubprocessExecutor, 'execute', observe)
    # No constructor or installer subprocess is permitted before the private probes.
    monkeypatch.setattr(state_model, 'probe_java_runtime', lambda *a, **k: pytest.fail('unleased Java probe'))
    from ipfs_datasets_py.logic.external_provers import lazy_installer
    monkeypatch.setattr(lazy_installer, 'ensure_prover_executable', lambda *a, **k: pytest.fail('implicit install'))
    parsed = routing.modal.parse_modal('p and not p', routing.modal.profile_k())
    target = dict(request_id='smt', source_construct='authored-mixed-fixture', logic_family='propositional',
        ast_format='shared_logic', printed=parsed.printed, native_ast=parsed.root.to_dict(), solver_names=('z3',))
    smt = make_datasets_smt_task(target=target, expected_verdict='unsat')
    lean = make_datasets_lean_nat_task(task_id='lean', offset=7)
    tlc = tasks.make_datasets_tlc_counter_task(task_id='tlc', counter_bound=5, invariant_max=5,
                                             dependencies=('smt', 'lean'))
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        bundle = BundleProverSupervisor(lease).execute((smt, lean, tlc))
    assert bundle.successful_task_ids == ('smt', 'lean', 'tlc'), bundle.to_dict()
    receipts = {item.task_id: item for item in bundle.receipts}
    assert receipts['tlc'].started_at_ms >= max(receipts['smt'].finished_at_ms, receipts['lean'].finished_at_ms)
    assert len(captured) == 5  # Lean probe+kernel; Java probe+TLC check+TLC help.
    assert all(roots == 1 and children >= 1 for roots, children, _ in captured)
    java = [item for _, _, item in captured if Path(item.argv[0]).name == 'java']
    assert len(java) == 3
    for invocation in java:
        assert invocation.limits.resident_memory_bytes == 512 * 1024**2
        assert invocation.limits.memory_bytes == 4096 * 1024**2
        assert invocation.limits.cpu_seconds > 0
        assert '-XX:ActiveProcessorCount=1' in invocation.argv
        assert not any(key in invocation.environment for key in ('JAVA_TOOL_OPTIONS', 'JDK_JAVA_OPTIONS', '_JAVA_OPTIONS'))
        assert '-Djava.io.tmpdir=.' in invocation.argv
        if '-config' in invocation.argv:
            assert invocation.argv[invocation.argv.index('-workers') + 1] == '1'
            assert invocation.argv[invocation.argv.index('-fpmem') + 1] == '0.0625'


@pytest.mark.parametrize('values', [
    {'counter_bound': True}, {'counter_bound': 0}, {'counter_bound': 65}, {'counter_bound': '1'},
    {'invariant_max': True}, {'invariant_max': -1}, {'invariant_max': 2},
    {'task_id': ''}, {'task_id': ' x '}, {'task_id': 'x' * 1025},
    {'timeout_seconds': 0}, {'timeout_seconds': float('inf')}, {'timeout_seconds': True},
    {'memory_mb': 255}, {'memory_mb': 4097}, {'memory_mb': True},
    {'dependencies': 'one'}, {'dependencies': ['x'] * 65},
])
def test_invalid_closed_inputs_rejected_without_execution(values):
    with pytest.raises(ValueError):
        tasks.make_datasets_tlc_counter_task(**{**dict(task_id='safe', counter_bound=1, invariant_max=1), **values})


def test_missing_runtime_refuses_without_native_launch_or_install(native, monkeypatch):
    scheduler, _, _ = native
    launches = []
    monkeypatch.setattr(tasks, '_runtime', lambda _: (_ for _ in ()).throw(FileNotFoundError('not installed')))
    monkeypatch.setattr(process.SubprocessExecutor, 'execute', lambda *a, **k: launches.append(a))
    result = execute(scheduler, tasks.make_datasets_tlc_counter_task(task_id='missing', counter_bound=1, invariant_max=1))
    assert result.status is ExecutionStatus.FAILED and result.result['status'] == 'unavailable'
    assert not launches


def test_standalone_lease_cannot_launch_native_tlc():
    with MultiProverResourceLease(budget()) as lease:
        result = ProverTaskExecutor(lease).execute(tasks.make_datasets_tlc_counter_task(
            task_id='unbridged', counter_bound=1, invariant_max=1))
    assert result.status is ExecutionStatus.FAILED and 'datasets-backed' in result.diagnostics


def test_repacked_underfunded_native_envelope_is_refused(native, monkeypatch):
    scheduler, _, _ = native
    launches = []
    monkeypatch.setattr(process.SubprocessExecutor, 'execute', lambda *a, **k: launches.append(a))
    task = tasks.make_datasets_tlc_counter_task(task_id='underfunded', counter_bound=1, invariant_max=1)
    task = replace(task, resources=replace(task.resources, memory_bytes=128 * 1024**2))
    result = execute(scheduler, task)
    assert result.reasons == ('tlc_resource_envelope',) and not launches


@pytest.mark.parametrize('mutation', ['witness', 'backend', 'result_id', 'bool', 'configuration', 'request'])
def test_post_native_contradictory_result_cannot_be_accepted(native, monkeypatch, mutation):
    scheduler, _, _ = native
    original = runners.TLCBackend.check
    def mutate(self, *args, **kwargs):
        actual = original(self, *args, **kwargs)
        if mutation == 'request':
            return replace(actual, request_digest='0' * 64)
        if mutation == 'configuration':
            return replace(actual, receipt=replace(actual.receipt, configuration_text='SPECIFICATION Other\n'))
        if mutation == 'backend':
            result = replace(actual.result, backend_id='different')
        elif mutation == 'result_id':
            result = replace(actual.result, result_id='stale')
        else:
            witness = actual.result.witness.to_dict()
            witness['bounded' if mutation == 'bool' else 'model_digest'] = 1 if mutation == 'bool' else 'f' * 64
            result = replace(actual.result, witness=FrozenMap(witness))
        return replace(actual, result=result)
    monkeypatch.setattr(runners.TLCBackend, 'check', mutate)
    result = execute(scheduler, tasks.make_datasets_tlc_counter_task(task_id='tampered', counter_bound=2, invariant_max=2))
    assert result.status is ExecutionStatus.FAILED and result.reasons == ('tlc_binding_mismatch',)
    assert result.result['native_checks'] == 1 and not result.result['model_check_passed']


def test_forged_pass_after_real_counterexample_cannot_unlock_dependency(native, monkeypatch):
    scheduler, _, _ = native
    original = runners.TLCBackend.check
    def mutate(self, artifact, **kwargs):
        actual = original(self, artifact, **kwargs)
        fake = replace(actual.receipt, status=runners.ModelCheckOutcomeStatus.PASSED, counterexample=None)
        fake_result = self._result_from_receipt(fake, request=kwargs['request'], bounds=kwargs['request'].bounds)
        return replace(actual, receipt=fake, result=fake_result)
    monkeypatch.setattr(runners.TLCBackend, 'check', mutate)
    first = tasks.make_datasets_tlc_counter_task(task_id='forged', counter_bound=1, invariant_max=0)
    after = tasks.make_datasets_tlc_counter_task(task_id='after', counter_bound=1, invariant_max=1, dependencies=('forged',))
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        result = BundleProverSupervisor(lease).execute((first, after))
    assert result.receipts[0].status is ExecutionStatus.FAILED
    assert result.receipts[0].result['status'] == 'counterexample'
    assert result.receipts[0].reasons == ('tlc_binding_mismatch',)
    assert result.receipts[1].status is ExecutionStatus.BLOCKED


def test_stale_real_outcome_without_current_native_check_is_refused(native, monkeypatch):
    scheduler, _, _ = native
    original = runners.TLCBackend.check
    previous = []
    def replay(self, *args, **kwargs):
        if previous:
            return previous[0]
        actual = original(self, *args, **kwargs)
        previous.append(actual)
        return actual
    monkeypatch.setattr(runners.TLCBackend, 'check', replay)
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        executor = ProverTaskExecutor(lease)
        first = executor.execute(tasks.make_datasets_tlc_counter_task(task_id='first', counter_bound=2, invariant_max=2))
        stale = executor.execute(tasks.make_datasets_tlc_counter_task(task_id='second', counter_bound=2, invariant_max=2))
    assert first.status is ExecutionStatus.SUCCEEDED
    assert stale.status is ExecutionStatus.FAILED and stale.reasons == ('tlc_binding_mismatch',)
    assert stale.result['native_checks'] == 0


@pytest.mark.parametrize('stop', ['cancel', 'deadline'])
def test_pressure_wait_stops_before_native_probe_without_leak(native, monkeypatch, stop):
    scheduler, pressure, healthy = native
    launches, results = [], []
    monkeypatch.setattr(process.SubprocessExecutor, 'execute', lambda *a, **k: launches.append(a))
    cancel = threading.Event()
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget(), admission_timeout_seconds=2) as lease:
        pressure[0] = replace(healthy, available_memory_mb=10)
        task = tasks.make_datasets_tlc_counter_task(task_id='waiting', counter_bound=1, invariant_max=1,
                                                  timeout_seconds=.1 if stop == 'deadline' else 2)
        thread = threading.Thread(target=lambda: results.append(ProverTaskExecutor(lease).execute(task, cancellation=cancel)))
        thread.start()
        if stop == 'cancel':
            deadline = time.monotonic() + 1
            while not scheduler.snapshot()['waiting_request_count'] and time.monotonic() < deadline:
                time.sleep(.005)
            cancel.set()
        thread.join(2)
        assert not thread.is_alive()
    assert not launches
    assert results[0].status is (ExecutionStatus.CANCELLED if stop == 'cancel' else ExecutionStatus.TIMED_OUT)


def test_live_tlc_cancellation_reaps_native_work_before_release(native, monkeypatch):
    scheduler, _, _ = native
    original = process.SubprocessExecutor.execute
    started, finished, cancel = threading.Event(), threading.Event(), threading.Event()
    pids, results = [], []
    def observe(self, invocation, cancellation=None):
        if '-config' not in invocation.argv:
            return original(self, invocation, cancellation)
        old_popen = self._popen
        def capture(*args, **kwargs):
            child = old_popen(*args, **kwargs)
            pids.append(child.pid)
            started.set()
            return child
        self._popen = capture
        try:
            return original(self, invocation, cancellation)
        finally:
            finished.set()
    monkeypatch.setattr(process.SubprocessExecutor, 'execute', observe)
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        thread = threading.Thread(target=lambda: results.append(ProverTaskExecutor(lease).execute(
            tasks.make_datasets_tlc_counter_task(task_id='cancel-live', counter_bound=64, invariant_max=64), cancellation=cancel)))
        thread.start()
        try:
            assert started.wait(5)
            assert scheduler.snapshot()['active_child_lease_count'] >= 1
            cancel.set()
            thread.join(2)
            assert not thread.is_alive() and results[0].status is ExecutionStatus.CANCELLED
            assert finished.wait(2)
        finally:
            cancel.set()
            thread.join(3)
    assert pids and all(not Path(f'/proc/{pid}').exists() for pid in pids)


def test_runtime_reader_rejects_fifo_without_waiting_for_writer(tmp_path):
    import os
    fifo = tmp_path / 'runtime.jar'
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match='regular file'):
        tasks._read_bounded(fifo, 1024, threading.Event())


def test_runtime_reader_rejects_oversized_regular_file(tmp_path):
    oversized = tmp_path / 'runtime.jar'
    with oversized.open('wb') as stream:
        stream.truncate(1 << 30)
    with pytest.raises(ValueError, match='byte bound'):
        tasks._read_bounded(oversized, 1024, threading.Event())
