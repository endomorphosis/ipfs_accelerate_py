"""Fresh closed Isabelle checks with shared budgets and exact kernel joins."""
from dataclasses import replace
from pathlib import Path
import shutil
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.proof import datasets_isabelle_tasks as tasks
from ipfs_accelerate_py.agent_supervisor.proof.datasets_kernel_tasks import make_datasets_lean_nat_task
from ipfs_accelerate_py.agent_supervisor.proof.datasets_hammer_tasks import make_datasets_smt_task
from ipfs_accelerate_py.agent_supervisor.proof.datasets_prover_resources import open_datasets_prover_lease
from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_resources import (
    BundleProverSupervisor, ExecutionStatus, MultiProverResourceBudget,
    MultiProverResourceLease, ProverTaskExecutor,
)
from ipfs_datasets_py.logic.backends import process
from ipfs_datasets_py.logic.backends.kernel import isabelle
from ipfs_datasets_py.logic.hammers import semantic_routing as routing
from ipfs_datasets_py.logic.ir_core.claims import FrozenMap
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceSchedulerConfig,
)


def budget():
    return MultiProverResourceBudget(cpu_slots=6, process_slots=24, thread_slots=6,
        memory_bytes=6144 * 1024**2, max_portfolio_width=2)


@pytest.fixture
def native(tmp_path):
    healthy = ProofHostResources(32, 32768, 32768)
    pressure = [healthy]
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / 'leases.json', proof_resource_sampler=lambda: pressure[0],
        total_cpu_slots=8, total_memory_mb=12288, total_child_process_slots=32,
        lane_reservations={}, proof_backoff_seconds=.03, poll_interval_seconds=.005))
    yield scheduler, pressure, healthy
    deadline = time.monotonic() + 5
    while scheduler.snapshot()['active_lease_count'] and time.monotonic() < deadline:
        time.sleep(.01)
    assert scheduler.snapshot()['active_lease_count'] == 0
    assert scheduler.snapshot()['waiting_request_count'] == 0


def execute(scheduler, task):
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        return ProverTaskExecutor(lease).execute(task)


def test_real_default_task_sized_path_uses_distinct_native_phase_leases():
    from ipfs_accelerate_py.agent_supervisor.proof.datasets_native_bundle import execute_datasets_native_bundle
    task = tasks.make_datasets_isabelle_nat_task(task_id='default-prepared-isabelle', offset=41)
    execution = execute_datasets_native_bundle([task], timeout_seconds=120)
    receipt = execution.receipt.receipts[0]
    assert receipt.status is ExecutionStatus.SUCCEEDED, receipt.to_dict()
    assert execution.plan.proof_safety_enabled
    assert receipt.result['kernel_accepted'] and receipt.result['bindings_match']
    assert receipt.result['native_checks'] == 1
    phases = receipt.result['native_phases']
    assert [row['phase'] for row in phases] == ['version', 'kernel']
    assert len({row['child_lease_id'] for row in phases}) == 2
    assert all(row['observation']['workspace_cleaned'] for row in phases)


@pytest.mark.parametrize('offset', [0, 7, 65535])
def test_real_closed_nat_theorem_has_exact_kernel_authority_only(native, offset):
    scheduler, _, _ = native
    task = tasks.make_datasets_isabelle_nat_task(task_id=f'isabelle-{offset}', offset=offset)
    result = execute(scheduler, task)
    assert result.status is ExecutionStatus.SUCCEEDED, result.to_dict()
    payload = result.result
    assert payload['kernel_accepted'] and payload['kernel_authority'] and payload['bindings_match']
    assert payload['native_checks'] == 1 and payload['runtime_unchanged']
    assert payload['statement'] == f'forall n : Isabelle.HOL.nat, n + {offset} = {offset} + n'
    assert payload['typed_result']['authority'] == 'theorem'
    assert payload['kernel_receipt']['imports'] == ['Main']
    assert payload['kernel_receipt']['translation'] is None
    assert payload['kernel_receipt']['axiom_report']['residual_axioms'] == []
    assert payload['kernel_receipt']['axiom_report']['contains_sorry'] is False
    assert payload['kernel_receipt']['axiom_report']['contains_unreviewed_axiomatization'] is False
    assert all(payload[key] is False for key in ('source_semantics_verified', 'cross_family_correspondence_verified',
        'behavior_authority', 'execution_authority', 'completion_authority'))
    assert not task.deterministic and not result.cache_bypassed_execution
    assert task.resources.memory_bytes == 2304 * 1024**2
    assert task.resources.cpu_slots == 3 and task.resources.process_slots == 12


def test_real_smt_lean_isabelle_share_owner_and_isolate_native_settings(native, monkeypatch):
    scheduler, _, _ = native
    assert shutil.which('lean') and shutil.which('z3')
    captured = []
    original = process.SubprocessExecutor.execute
    def observe(self, invocation, cancellation=None):
        state = scheduler.snapshot()
        captured.append((state['active_root_lease_count'], state['active_child_lease_count'], invocation))
        return original(self, invocation, cancellation)
    monkeypatch.setattr(process.SubprocessExecutor, 'execute', observe)
    from ipfs_datasets_py.logic.hammers.frontends.isabelle import IsabelleFrontend
    from ipfs_datasets_py.logic.external_provers import lazy_installer
    monkeypatch.setattr(IsabelleFrontend, 'capability', lambda *a, **k: pytest.fail('unleased frontend probe'))
    monkeypatch.setattr(lazy_installer, 'ensure_prover_executable', lambda *a, **k: pytest.fail('implicit installer'))
    parsed = routing.modal.parse_modal('p and not p', routing.modal.profile_k())
    target = dict(request_id='smt', source_construct='authored-mixed-fixture', logic_family='propositional',
        ast_format='shared_logic', printed=parsed.printed, native_ast=parsed.root.to_dict(), solver_names=('z3',))
    smt = make_datasets_smt_task(target=target, expected_verdict='unsat')
    lean = make_datasets_lean_nat_task(task_id='lean', offset=7)
    checked = tasks.make_datasets_isabelle_nat_task(task_id='isabelle', offset=7, dependencies=('smt', 'lean'))
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        bundle = BundleProverSupervisor(lease).execute((smt, lean, checked))
    assert bundle.successful_task_ids == ('smt', 'lean', 'isabelle'), bundle.to_dict()
    receipts = {item.task_id: item for item in bundle.receipts}
    assert receipts['isabelle'].started_at_ms >= max(receipts['smt'].finished_at_ms, receipts['lean'].finished_at_ms)
    assert len(captured) == 4  # Lean probe+check; Isabelle version+check.
    assert all(roots == 1 and children >= 1 for roots, children, _ in captured)
    invocations = [item for _, _, item in captured if Path(item.argv[0]).name == 'isabelle']
    assert len(invocations) == 2
    for item in invocations:
        assert item.limits.resident_memory_bytes == 2048 * 1024**2
        assert item.limits.memory_bytes == 32 * 1024**3 and item.limits.cpu_seconds > 0
        assert item.environment['HOME'] == item.environment['TMPDIR'] == str(item.cwd)
        assert 'USER_HOME' not in item.environment and 'ISABELLE_SETTINGS_PRESENT' not in item.environment
        assert '-XX:ActiveProcessorCount=1' in item.environment['JDK_JAVA_OPTIONS']
        assert '-Xmx256m' in item.environment['JDK_JAVA_OPTIONS']
        assert not Path(item.cwd).exists()
        if 'process_theories' in item.argv:
            assert 'threads=1' in item.argv and 'parallel_proofs=0' in item.argv and 'quick_and_dirty=false' in item.argv


@pytest.mark.parametrize('values', [
    {'offset': True}, {'offset': -1}, {'offset': 65536}, {'offset': '1'},
    {'task_id': ''}, {'task_id': ' x '}, {'task_id': 'x' * 1025},
    {'timeout_seconds': 0}, {'timeout_seconds': float('inf')}, {'timeout_seconds': True},
    {'memory_mb': 1023}, {'memory_mb': 4097}, {'memory_mb': True},
    {'dependencies': 'one'}, {'dependencies': ['x'] * 65},
])
def test_invalid_closed_input_fails_before_execution(values):
    with pytest.raises(ValueError):
        tasks.make_datasets_isabelle_nat_task(**{**dict(task_id='safe', offset=1), **values})


def test_standalone_lease_cannot_run_native_isabelle():
    with MultiProverResourceLease(budget()) as lease:
        result = ProverTaskExecutor(lease).execute(tasks.make_datasets_isabelle_nat_task(task_id='unbridged', offset=1))
    assert result.status is ExecutionStatus.FAILED and 'datasets-backed' in result.diagnostics


def test_missing_runtime_refuses_and_blocks_dependent_without_installer(native, monkeypatch):
    scheduler, _, _ = native
    launches = []
    monkeypatch.setattr(tasks, '_runtime', lambda _: (_ for _ in ()).throw(FileNotFoundError('not installed')))
    monkeypatch.setattr(process.SubprocessExecutor, 'execute', lambda *a, **k: launches.append(a))
    first = tasks.make_datasets_isabelle_nat_task(task_id='missing', offset=1)
    after = tasks.make_datasets_isabelle_nat_task(task_id='after', offset=1, dependencies=('missing',))
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        bundle = BundleProverSupervisor(lease).execute((first, after))
    assert bundle.receipts[0].status is ExecutionStatus.FAILED
    assert bundle.receipts[0].result['status'] == 'unavailable'
    assert bundle.receipts[1].status is ExecutionStatus.BLOCKED and not launches


@pytest.mark.parametrize('resource', ['memory_bytes', 'cpu_slots', 'process_slots', 'thread_slots'])
def test_repacked_underfunded_resource_envelope_never_launches(native, monkeypatch, resource):
    scheduler, _, _ = native
    launches = []
    monkeypatch.setattr(process.SubprocessExecutor, 'execute', lambda *a, **k: launches.append(a))
    task = tasks.make_datasets_isabelle_nat_task(task_id='underfunded', offset=1)
    changes = {resource: 1}
    if resource == 'cpu_slots':
        # The bridge reserves at least as many CPUs as requested worker threads.
        changes['thread_slots'] = 1
    task = replace(task, resources=replace(task.resources, **changes))
    result = execute(scheduler, task)
    assert result.status is ExecutionStatus.FAILED and result.reasons == ('isabelle_resource_envelope',)
    assert not launches


def test_real_kernel_rejects_perturbed_generated_proof(native, monkeypatch):
    scheduler, _, _ = native
    original = tasks._profile_source
    def invalid_proof(offset):
        name, statement, source = original(offset)
        return name, statement, source.replace('by (rule add.commute)', 'by (rule refl)')
    monkeypatch.setattr(tasks, '_profile_source', invalid_proof)
    result = execute(scheduler, tasks.make_datasets_isabelle_nat_task(task_id='rejected', offset=7))
    assert result.status is ExecutionStatus.FAILED
    assert result.result['native_checks'] == 1 and not result.result['kernel_accepted']
    assert not result.result['kernel_authority'] and result.result['native_output_accepted'] is False


@pytest.mark.parametrize('mutation', ['witness', 'backend', 'request', 'bool'])
def test_post_native_inconsistent_typed_metadata_is_refused(native, monkeypatch, mutation):
    scheduler, _, _ = native
    original = isabelle.IsabelleKernelBackend.run
    def mutate(self, *args, **kwargs):
        actual = original(self, *args, **kwargs)
        if mutation == 'request':
            binding = replace(actual.source_binding, request_digest='f' * 64)
            receipt = replace(actual.receipt, request_digest='f' * 64, source_binding=binding)
            return replace(actual, request_digest='f' * 64, source_binding=binding, receipt=receipt)
        if mutation == 'backend':
            result = replace(actual.result, backend_id='unrelated')
        else:
            witness = actual.result.witness.to_dict()
            if mutation == 'witness':
                witness['theorem_digest'] = '0' * 64
            else:
                witness['axiom_report']['contains_sorry'] = 0
            result = replace(actual.result, witness=FrozenMap(witness))
        return replace(actual, result=result)
    monkeypatch.setattr(isabelle.IsabelleKernelBackend, 'run', mutate)
    result = execute(scheduler, tasks.make_datasets_isabelle_nat_task(task_id='mutated', offset=7))
    assert result.status is ExecutionStatus.FAILED and result.reasons == ('isabelle_binding_mismatch',)
    assert result.result['native_checks'] == 1 and not result.result['kernel_authority']


def test_stale_accepted_result_without_fresh_native_check_is_refused(native, monkeypatch):
    scheduler, _, _ = native
    original = isabelle.IsabelleKernelBackend.run
    previous = []
    def replay(self, *args, **kwargs):
        if previous:
            return previous[0]
        outcome = original(self, *args, **kwargs)
        previous.append(outcome)
        return outcome
    monkeypatch.setattr(isabelle.IsabelleKernelBackend, 'run', replay)
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        executor = ProverTaskExecutor(lease)
        first = executor.execute(tasks.make_datasets_isabelle_nat_task(task_id='first', offset=1))
        stale = executor.execute(tasks.make_datasets_isabelle_nat_task(task_id='stale', offset=1))
    assert first.status is ExecutionStatus.SUCCEEDED, first.to_dict()
    assert stale.status is ExecutionStatus.FAILED and stale.reasons == ('isabelle_binding_mismatch',)
    assert stale.result['native_checks'] == 0 and not stale.result['kernel_authority']


@pytest.mark.parametrize('stop', ['cancel', 'deadline'])
def test_external_pressure_wait_stops_before_native_probe(native, monkeypatch, stop):
    scheduler, pressure, healthy = native
    launches, results = [], []
    monkeypatch.setattr(process.SubprocessExecutor, 'execute', lambda *a, **k: launches.append(a))
    cancel = threading.Event()
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget(), admission_timeout_seconds=2) as lease:
        pressure[0] = replace(healthy, available_memory_mb=10)
        task = tasks.make_datasets_isabelle_nat_task(task_id='waiting', offset=1,
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


@pytest.mark.parametrize('stage', ['version', 'process_theories'])
def test_live_native_cancel_reaps_process_group_before_resources_release(native, monkeypatch, stage):
    scheduler, _, _ = native
    original = process.SubprocessExecutor.execute
    started, finished, cancel = threading.Event(), threading.Event(), threading.Event()
    pids, results = [], []
    def observe(self, invocation, cancellation=None):
        if stage not in invocation.argv:
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
            tasks.make_datasets_isabelle_nat_task(task_id='cancel-live', offset=7), cancellation=cancel)))
        thread.start()
        try:
            assert started.wait(15)
            assert scheduler.snapshot()['active_child_lease_count'] >= 1
            if stage == 'process_theories':
                deadline = time.monotonic() + 12
                descendants = {}
                while time.monotonic() < deadline:
                    descendants = _native_descendants(pids[0])
                    if 'poly' in descendants.values():
                        break
                    time.sleep(.01)
                assert 'poly' in descendants.values(), descendants
                pids.extend(descendants)
            cancel.set()
            thread.join(3)
            assert not thread.is_alive() and results[0].status is ExecutionStatus.CANCELLED
            assert finished.wait(3)
        finally:
            cancel.set()
            thread.join(5)
    deadline = time.monotonic() + 3
    while any(_native_alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(.02)
    assert pids and not any(_native_alive(pid) for pid in pids)


def test_runtime_reader_refuses_fifo_without_blocking(tmp_path):
    import os
    path = tmp_path / 'launcher'
    os.mkfifo(path)
    with pytest.raises(ValueError, match='regular file'):
        tasks._read_bounded(path, 1024, threading.Event())


@pytest.mark.parametrize('stop', ['resume', 'cancel', 'deadline'])
def test_controlled_pressure_between_version_and_kernel_requires_fresh_child(native, monkeypatch, stop):
    """Synthetic native outcomes isolate admission timing, not theorem truth."""
    scheduler, pressure, healthy = native
    signal = threading.Event()
    launches, timers = [], []
    runtime = {'version': 'Isabelle2025-2', 'executable': '/fixture/bin/isabelle',
               'runtime_root': '/fixture', 'selected_file_sha256': {}}
    monkeypatch.setattr(tasks, '_runtime', lambda _: ('/fixture/bin/isabelle', runtime))

    def controlled(self, request, **kwargs):
        rows = scheduler.active_leases()
        assert len(rows) == 3  # shared root, admitted task and fresh native phase
        leaf = next(row for row in rows if not any(other.get('parent_lease_id') == row['lease_id'] for other in rows))
        launches.append((request.argv, leaf['lease_id'], time.monotonic()))
        version = request.argv[-1] == 'version'
        if version:
            pressure[0] = replace(healthy, available_memory_mb=100)
            if stop != 'deadline':
                action = signal.set if stop == 'cancel' else lambda: pressure.__setitem__(0, healthy)
                timer = threading.Timer(.15, action)
                timers.append(timer)
                timer.start()
        return process.ToolRunResult(interface_version=process.BOUNDED_TOOL_RUNNER_VERSION,
            runtime=request.runtime, command=request.argv, returncode=0,
            stdout='Isabelle2025-2' if version else 'IPFS_ISABELLE_KERNEL_CHECKED',
            stderr='', elapsed_seconds=.001, output_files={})

    monkeypatch.setattr(process.BoundedToolRunner, 'run', controlled)
    task = tasks.make_datasets_isabelle_nat_task(task_id='interphase-pressure', offset=7,
        timeout_seconds=.3 if stop == 'deadline' else 3)
    try:
        with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
            result = ProverTaskExecutor(lease).execute(task, cancellation=signal)
    finally:
        for timer in timers:
            timer.cancel()
            timer.join(1)
    if stop == 'resume':
        assert result.status is ExecutionStatus.SUCCEEDED, result.to_dict()
        assert len(launches) == 2 and launches[0][1] != launches[1][1]
        assert launches[1][2] - launches[0][2] >= .14
        assert [row['phase'] for row in result.result['native_phases']] == ['version', 'kernel']
    else:
        assert len(launches) == 1
        assert result.status is (ExecutionStatus.CANCELLED if stop == 'cancel' else ExecutionStatus.TIMED_OUT), result.to_dict()


def _native_descendants(leader):
    pending, result = [leader], {}
    while pending:
        pid = pending.pop()
        if pid in result:
            continue
        root = Path('/proc') / str(pid)
        try:
            result[pid] = (root / 'comm').read_text().strip()
            for task in (root / 'task').iterdir():
                try:
                    pending.extend(int(child) for child in (task / 'children').read_text().split())
                except (OSError, ValueError):
                    pass
        except OSError:
            pass
    return result


def _native_alive(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z'
    except (OSError, IndexError):
        return False
