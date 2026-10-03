"""Injected ownership/fault controls are separate from actual-host acceptance."""
from contextlib import contextmanager
from pathlib import Path
import json
import subprocess
import gc
import weakref
import sys
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_resource_bridge as bridge
from ipfs_accelerate_py.agent_supervisor.runtime import repository_pipeline_resources as pipeline
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    ResourceScheduler, ResourcePolicy, HostResourceSnapshot,
)
from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_daemon_resources as daemon
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceSchedulerConfig, ResourceLane, get_global_resource_scheduler,
)


@pytest.fixture
def injected_owners(tmp_path, monkeypatch):
    """Real file-backed owners, explicitly injected host telemetry, not acceptance."""
    shared = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / 'injected-native.json',
        proof_resource_sampler=lambda: ProofHostResources(8, 8192, 8192),
        poll_interval_seconds=.005, proof_backoff_seconds=.02))
    monkeypatch.setattr(bridge, 'get_global_resource_scheduler', lambda: shared)
    monkeypatch.setattr(daemon, 'get_global_resource_scheduler', lambda: shared)
    supervisor = ResourceScheduler(ResourcePolicy(max_lanes=8), host_sampler=lambda *args, **kw:
        HostResourceSnapshot(memory_total_bytes=8*1024**3, memory_available_bytes=8*1024**3,
            disk_total_bytes=16*1024**3, disk_available_bytes=16*1024**3,
            worker_limit=8, available_worker_capacity=8))
    yield supervisor, shared
    assert not supervisor.active_leases
    assert shared.snapshot()['active_lease_count'] == shared.snapshot()['waiting_request_count'] == 0


def budget(**changes):
    return bridge.RepositoryResourceBudget(**dict(dict(cpu_slots=2, memory_mb=512,
        process_slots=2, disk_bytes=16*bridge.MIB, wall_time_ms=10000), **changes))


def demand(phase, **changes):
    return bridge.RepositoryPhaseDemand(phase, **dict(dict(memory_mb=128,
        disk_bytes=bridge.MIB), **changes))


@contextmanager
def reserve(tmp_path, supervisor, *, policy=None, limits=None, cancel=None):
    roots = tmp_path / 'disk'
    roots.mkdir(exist_ok=True)
    with pipeline.RepositoryPipelineResources(supervisor).reserve(repository_id='rpi022-pipeline',
            workspace=tmp_path, budget=limits or budget(), policy=policy or pipeline.PipelineResourcePolicy(),
            ledger_path=tmp_path / 'disk-ledger.json', roots=[roots], cancel_event=cancel) as parent:
        yield parent


def attempt(tmp_path, name):
    path = tmp_path / 'disk' / name
    path.mkdir()
    return path


def eventually(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, 'condition was not observed'
        time.sleep(.01)


@pytest.mark.parametrize('change', [dict(protected_cpu_slots=True), dict(protected_memory_mb=0),
    dict(protected_queue_slots=8), dict(protected_phase_slots=128), dict(protected_cpu_slots=2),
    dict(protected_memory_mb=512), dict(protected_process_slots=2), dict(protected_disk_bytes=16*bridge.MIB),
    dict(protected_queued_payload_bytes=4*bridge.MIB), dict(maximum_retained_payload_bytes=512*bridge.MIB)])
def test_policy_refuses_unbounded_or_empty_partitions(change):
    with pytest.raises(bridge.RepositoryResourceError):
        pipeline.PipelineResourcePolicy(**change).validate(budget())


def test_injected_exact_nested_authority_and_explicit_durability(injected_owners, tmp_path):
    supervisor, shared = injected_owners
    with reserve(tmp_path, supervisor) as parent:
        with parent.phase(demand('sql'), payload=b'bounded-input', attempt_directory=attempt(tmp_path, 'sql')) as phase:
            options = phase.native_options()
            with options['parent_lease'].acquire_child(lane=ResourceLane.PERSISTENCE,
                    cpu_slots=1, memory_mb=64, child_process_slots=1, timeout=1) as consumer:
                assert consumer.parent_lease_id == phase.daemon.native_lease.lease_id
                assert phase.daemon.native_lease.parent_lease_id == phase.native.native.lease_id
                snapshot = shared.snapshot()
                assert snapshot['active_root_lease_count'] == 1
                assert snapshot['active_child_lease_count'] == 3
            assert phase.payload == b'bounded-input'
            with pytest.raises(daemon.DaemonResourceError, match='durability'):
                phase.finalize()
            phase.finalize(artifacts_durable=True)
        receipt = parent.receipt()
        assert receipt['retained_payload_bytes'] == receipt['owned_disk_bytes'] == 0
        assert receipt['events'][0]['disk']['status'] == 'released'
        assert 'lease_key' not in json.dumps(receipt)
        receipt['events'].clear()
        assert len(parent.receipt()['events']) == 1
    assert parent.receipt()['root']['owned_active_root_count'] == 0


def test_injected_full_training_cannot_take_validation_or_cleanup_compartment(injected_owners, tmp_path):
    supervisor, shared = injected_owners
    entered, finish, errors = threading.Event(), threading.Event(), []
    with reserve(tmp_path, supervisor) as parent:
        training_attempt = attempt(tmp_path, 'training')
        def train():
            try:
                with parent.phase(demand('training', memory_mb=384, disk_bytes=8*bridge.MIB),
                                  attempt_directory=training_attempt) as phase:
                    entered.set()
                    assert finish.wait(5)
                    phase.finalize(artifacts_durable=True)
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=train)
        thread.start()
        try:
            assert entered.wait(3)
            for name in ('validation', 'cleanup'):
                with parent.phase(demand(name), attempt_directory=attempt(tmp_path, name)) as phase:
                    assert parent.receipt()['active_phases'] == 2
                    assert shared.snapshot()['active_root_lease_count'] == 1
                    phase.finalize(artifacts_durable=True)
        finally:
            finish.set()
            thread.join(5)
        assert not thread.is_alive() and not errors


def test_injected_payload_queue_bytes_count_actual_immutable_inputs(injected_owners, tmp_path):
    supervisor, _ = injected_owners
    policy = pipeline.PipelineResourcePolicy(maximum_queued_payload_bytes=16,
        protected_queued_payload_bytes=4)
    errors = []
    with reserve(tmp_path, supervisor, policy=policy) as parent:
        queued_attempt = attempt(tmp_path, 'queued')
        def queue():
            try:
                with parent.phase(demand('scan'), payload=b'123456789012', attempt_directory=queued_attempt) as phase:
                    phase.finalize(artifacts_durable=True)
            except BaseException as error:
                errors.append(error)
        with parent.phase(demand('training'), attempt_directory=attempt(tmp_path, 'train')) as first:
            thread = threading.Thread(target=queue)
            thread.start()
            eventually(lambda: parent.receipt()['queued_payload_bytes'] == 12)
            with pytest.raises(bridge.RepositoryResourceError, match='queued payload byte'):
                with parent.phase(demand('scan'), payload=b'x', attempt_directory=attempt(tmp_path, 'overflow')):
                    pytest.fail('payload overflow admitted')
            with pytest.raises(bridge.RepositoryResourceError, match='immutable bytes'):
                with parent.phase(demand('scan'), payload=bytearray(b'x'), attempt_directory=attempt(tmp_path, 'mutable')):
                    pytest.fail('mutable queue retained')
            with parent.phase(demand('validation'), payload=b'1234', attempt_directory=attempt(tmp_path, 'validation')) as validation:
                validation.finalize(artifacts_durable=True)
            first.finalize(artifacts_durable=True)
        thread.join(5)
        assert not thread.is_alive() and not errors


def test_injected_retained_payload_covers_active_inputs_and_capture_allowance(injected_owners, tmp_path):
    supervisor, _ = injected_owners
    policy = pipeline.PipelineResourcePolicy(maximum_queued_payload_bytes=1024,
        protected_queued_payload_bytes=128, maximum_retained_payload_bytes=300000,
        protected_retained_payload_bytes=140000)
    with reserve(tmp_path, supervisor, policy=policy) as parent:
        with parent.phase(demand('training'), payload=b'x'*512, attempt_directory=attempt(tmp_path, 'train')) as first:
            with pytest.raises(bridge.RepositoryResourceError, match='retained payload byte'):
                with parent.phase(demand('scan'), payload=b'y', attempt_directory=attempt(tmp_path, 'queued')):
                    pytest.fail('active retained bytes ignored')
            first.finalize(artifacts_durable=True)


def test_injected_failed_disk_claim_stays_charged_until_protected_cleanup(injected_owners, tmp_path):
    supervisor, _ = injected_owners
    with reserve(tmp_path, supervisor) as parent:
        with pytest.raises(RuntimeError, match='consumer failed'):
            with parent.phase(demand('training', disk_bytes=8*bridge.MIB), attempt_directory=attempt(tmp_path, 'failed')):
                raise RuntimeError('consumer failed')
        receipt = parent.receipt()
        assert receipt['owned_disk_bytes'] == 8*bridge.MIB
        assert len(receipt['retained_disk_reservations']) == 1
        retained = receipt['retained_disk_reservations'][0]
        with parent.phase(demand('cleanup'), attempt_directory=attempt(tmp_path, 'cleanup')) as cleanup:
            with pytest.raises(daemon.DaemonResourceError, match='durability'):
                cleanup.recover_retained(retained)
            cleanup.recover_retained(retained, artifacts_durable=True)
            cleanup.finalize(artifacts_durable=True)
        assert parent.receipt()['retained_disk_reservations'] == []
        assert parent.receipt()['owned_disk_bytes'] == 0


def test_injected_missing_finalize_retains_claim_and_refuses_reused_attempt(injected_owners, tmp_path):
    supervisor, _ = injected_owners
    with reserve(tmp_path, supervisor) as parent:
        path = attempt(tmp_path, 'failed')
        with pytest.raises(bridge.RepositoryResourceError, match='explicit durable finalization'):
            with parent.phase(demand('sql'), attempt_directory=path):
                pass
        with pytest.raises(bridge.RepositoryResourceError, match='never reused'):
            with parent.phase(demand('sql'), attempt_directory=path):
                pytest.fail('reused stale attempt')
        assert parent.receipt()['owned_disk_bytes'] == bridge.MIB


def test_injected_precharge_bounds_paths_and_final_disk_bytes(injected_owners, tmp_path):
    supervisor, _ = injected_owners
    with reserve(tmp_path, supervisor) as parent:
        with parent.phase(demand('persistence'), attempt_directory=attempt(tmp_path, 'persist')) as phase:
            output = tmp_path/'disk'/'output'
            with pytest.raises(bridge.RepositoryResourceError, match='outside named'):
                phase.charge_external(tmp_path/'unowned', 1)
            phase.charge_external(output, 1024)
            assert phase.charge_external(output, 1024)['already_charged']
            with pytest.raises(daemon.DaemonResourceError, match='amount changed'):
                phase.charge_external(output, 2048)
            with pytest.raises(daemon.DaemonResourceError, match='storage byte limit'):
                phase.charge_external(tmp_path/'disk'/'overflow', bridge.MIB)
            output.write_bytes(b'x'*1024)
            phase.finalize(artifacts_durable=True)
        assert parent.receipt()['owned_disk_bytes'] == 1024


def test_injected_bounded_process_uses_actual_payload_and_reaps_before_release(injected_owners, tmp_path):
    supervisor, _ = injected_owners
    with reserve(tmp_path, supervisor) as parent:
        with parent.phase(demand('validation'), payload=b'hello', attempt_directory=attempt(tmp_path, 'process')) as phase:
            result = phase.run([str(Path(sys.executable).resolve()), '-I', '-B', '-c',
                'import sys;print(sys.stdin.buffer.read().decode())'], timeout_seconds=3)
            assert result.returncode == 0 and result.stdout.strip() == 'hello'
            assert result.workspace_cleaned
            phase.finalize(artifacts_durable=True)
        disk = parent.receipt()['events'][0]['disk']
        assert disk['status'] == 'released' and disk['record']['child']['pid'] == result.pid
        assert disk['record']['last_usage']['group_rss']['live_processes'] == 0


def test_injected_process_cancellation_reaps_and_retains_disk(injected_owners, tmp_path):
    supervisor, shared = injected_owners
    signal = threading.Event()
    timer = None
    results = []
    with pytest.raises(bridge.LeaseCancelledError):
        with reserve(tmp_path, supervisor, cancel=signal) as parent:
            with parent.phase(demand('training'), attempt_directory=attempt(tmp_path, 'cancel')) as phase:
                timer = threading.Timer(.2, signal.set)
                timer.start()
                try:
                    phase.run([str(Path(sys.executable).resolve()), '-I', '-B', '-c', 'import time;time.sleep(10)'], timeout_seconds=3)
                finally:
                    results.append(phase.last_process_result)
    if timer is not None:
        timer.join()
    disk = parent.receipt()['events'][0]['disk']
    assert disk['status'] == 'retained'
    assert phase.last_process_result is None
    assert results[0].cancelled and results[0].process_tree_terminated
    assert shared.snapshot()['active_root_lease_count'] == 0


def test_live_pipeline_composes_actual_host_and_named_disk_owners(tmp_path):
    """This can be refused by current host pressure; never substitute telemetry."""
    supervisor = ResourceScheduler(ResourcePolicy(max_lanes=8))
    shared = get_global_resource_scheduler()
    with reserve(tmp_path, supervisor, limits=budget(wall_time_ms=15000)) as parent:
        with parent.phase(demand('validation'), payload=b'live', attempt_directory=attempt(tmp_path, 'live')) as phase:
            result = phase.run([str(Path(sys.executable).resolve()), '-I', '-B', '-c',
                'import sys;print(sys.stdin.buffer.read().decode())'], timeout_seconds=3)
            assert result.returncode == 0 and result.stdout.strip() == 'live'
            assert phase.daemon.native_lease.parent_lease_id == phase.native.native.lease_id
            assert parent.receipt()['root']['host_authority']['state_path'] == str(shared.state_path)
            phase.finalize(artifacts_durable=True)
    assert not supervisor.active_leases and parent.receipt()['root']['owned_active_root_count'] == 0


def test_injected_ordinary_history_exhaustion_preserves_cleanup(injected_owners, tmp_path):
    supervisor, _ = injected_owners
    policy = pipeline.PipelineResourcePolicy(protected_phase_slots=1)
    with reserve(tmp_path, supervisor, policy=policy, limits=budget(maximum_phases=2)) as parent:
        with parent.phase(demand('scan'), attempt_directory=attempt(tmp_path, 'first')) as phase:
            phase.finalize(artifacts_durable=True)
        with pytest.raises(bridge.RepositoryResourceError, match='history exhausted'):
            with parent.phase(demand('training'), attempt_directory=attempt(tmp_path, 'overflow')):
                pytest.fail('ordinary history borrowed protected slot')
        with parent.phase(demand('cleanup'), attempt_directory=attempt(tmp_path, 'cleanup')) as phase:
            phase.finalize(artifacts_durable=True)


@pytest.mark.parametrize('name,value', [('cpu_slots', 2), ('memory_mb', 385),
    ('process_slots', 2), ('disk_bytes', 8*bridge.MIB+1)])
def test_injected_ordinary_demand_cannot_borrow_protected_capacity(injected_owners, tmp_path, name, value):
    supervisor, _ = injected_owners
    with reserve(tmp_path, supervisor) as parent:
        changes = {name: value}
        if name == 'process_slots':
            changes['cpu_slots'] = 2
        with pytest.raises(bridge.RepositoryResourceError, match='protected or ordinary'):
            with parent.phase(demand('training', **changes), attempt_directory=attempt(tmp_path, name)):
                pytest.fail('protected compartment borrowed')


def test_injected_process_cannot_escape_owner_thread_or_capture_bound(injected_owners, tmp_path):
    supervisor, _ = injected_owners
    errors = []
    with reserve(tmp_path, supervisor) as parent:
        with parent.phase(demand('validation'), attempt_directory=attempt(tmp_path, 'phase')) as phase:
            def other_thread():
                try:
                    phase.native_options()
                except BaseException as error:
                    errors.append(error)
            thread = threading.Thread(target=other_thread)
            thread.start(); thread.join(2)
            assert len(errors) == 1 and 'synchronous' in str(errors[0])
            with pytest.raises(bridge.RepositoryResourceError, match='capture exceeds'):
                phase.run(['/bin/true'], max_output_bytes=100000, timeout_seconds=1)
            phase.finalize(artifacts_durable=True)
        with pytest.raises(bridge.RepositoryResourceError, match='closed or finalized'):
            phase.native_options()


def test_injected_actual_disk_overshoot_reaps_process_and_retains_claim(injected_owners, tmp_path):
    supervisor, _ = injected_owners
    with reserve(tmp_path, supervisor) as parent:
        path = attempt(tmp_path, 'overshoot')
        with pytest.raises(daemon.DaemonResourceError, match='storage byte limit'):
            with parent.phase(demand('training'), attempt_directory=path) as phase:
                phase.run([str(Path(sys.executable).resolve()), '-I', '-B', '-c',
                    'import pathlib,sys,time;p=pathlib.Path(sys.argv[1]);'
                    '(p/"a").write_bytes(b"x"*600000);(p/"b").write_bytes(b"y"*600000);time.sleep(10)',
                    str(path)], timeout_seconds=3)
        assert phase.last_process_result is None
        receipt = parent.receipt()
        assert receipt['retained_disk_reservations'] and receipt['owned_disk_bytes'] == bridge.MIB
        child = receipt['events'][0]['disk']['record']['child']
        assert child is not None and daemon._group_usage(child)['live_processes'] == 0


def test_injected_cancelled_queue_releases_payload_references(injected_owners, tmp_path):
    supervisor, _ = injected_owners
    errors = []
    signal = threading.Event()
    with pytest.raises(bridge.LeaseCancelledError):
        with reserve(tmp_path, supervisor, cancel=signal) as parent:
            queued_attempt = attempt(tmp_path, 'queued')
            def queue():
                try:
                    with parent.phase(demand('scan'), payload=b'held', attempt_directory=queued_attempt):
                        pytest.fail('cancelled queue admitted')
                except BaseException as error:
                    errors.append(error)
            with parent.phase(demand('training'), attempt_directory=attempt(tmp_path, 'train')):
                thread = threading.Thread(target=queue)
                thread.start()
                eventually(lambda: parent.receipt()['queued_payload_bytes'] == 4)
                signal.set(); thread.join(3)
                assert not thread.is_alive() and len(errors) == 1
                assert isinstance(errors[0], bridge.LeaseCancelledError)
                parent.parent.remaining()
    assert parent.receipt()['queued_payload_bytes'] == parent.receipt()['retained_payload_bytes'] == 0


def test_injected_live_registered_child_retains_bridge_until_reaped(injected_owners, tmp_path):
    supervisor, shared = injected_owners
    child = None
    with reserve(tmp_path, supervisor) as parent:
        try:
            with pytest.raises(bridge.RepositoryResourceError, match='explicit durable finalization'):
                with parent.phase(demand('training'), attempt_directory=attempt(tmp_path, 'live')) as phase:
                    child = subprocess.Popen([sys.executable, '-I', '-B', '-c', 'import time;time.sleep(30)'],
                                             start_new_session=True)
                    phase.daemon.check_usage(phase.attempt, child_pid=child.pid)
            receipt = parent.receipt()
            assert receipt['retained_host_phase_count'] == 1 and receipt['active_phases'] == 1
            assert phase.native.native.lease_id in {r['lease_id'] for r in shared.active_leases()}
            retained = receipt['retained_disk_reservations'][0]
            with parent.phase(demand('cleanup'), attempt_directory=attempt(tmp_path, 'cleanup')) as cleanup:
                with pytest.raises(daemon.DaemonResourceError, match='still alive'):
                    cleanup.recover_retained(retained, artifacts_durable=True)
                child.terminate(); child.wait(timeout=3)
                cleanup.recover_retained(retained, artifacts_durable=True)
                assert phase.native.native.released
                cleanup.finalize(artifacts_durable=True)
            assert parent.receipt()['retained_host_phase_count'] == 0
        finally:
            if child is not None and child.poll() is None:
                child.kill(); child.wait(timeout=3)


def test_injected_outstanding_native_consumer_cannot_release_parent(injected_owners, tmp_path):
    supervisor, shared = injected_owners
    consumer = None
    with reserve(tmp_path, supervisor) as parent:
        try:
            with pytest.raises(bridge.RepositoryResourceError, match='native consumers must finish'):
                with parent.phase(demand('training'), attempt_directory=attempt(tmp_path, 'consumer')) as phase:
                    consumer = phase.native_options()['parent_lease'].acquire_child(lane=ResourceLane.TRAINER,
                        cpu_slots=1, memory_mb=64, child_process_slots=1, timeout=1)
                    phase.finalize(artifacts_durable=True)
            receipt = parent.receipt()
            assert receipt['retained_host_phase_count'] == 1
            retained = receipt['retained_disk_reservations'][0]
            with parent.phase(demand('cleanup'), attempt_directory=attempt(tmp_path, 'cleanup')) as cleanup:
                with pytest.raises(bridge.RepositoryResourceError, match='native consumers must release'):
                    cleanup.recover_retained(retained, artifacts_durable=True)
                consumer.release()
                cleanup.recover_retained(retained, artifacts_durable=True)
                cleanup.finalize(artifacts_durable=True)
            assert parent.receipt()['retained_host_phase_count'] == 0
        finally:
            if consumer is not None:
                consumer.release()


def test_injected_validation_requires_explicit_compatible_memory_declaration(injected_owners, tmp_path):
    supervisor, _ = injected_owners
    with reserve(tmp_path, supervisor) as parent:
        with pytest.raises(bridge.RepositoryResourceError, match='protected or ordinary memory_mb'):
            with parent.phase(bridge.RepositoryPhaseDemand('validation'), attempt_directory=attempt(tmp_path, 'invalid')):
                pytest.fail('512 MiB default demand entered 128 MiB protected compartment')


def test_injected_unsafe_outer_exit_survives_gc_until_new_admitted_cleanup(injected_owners, tmp_path):
    supervisor, shared = injected_owners
    child = None
    old = None
    try:
        with pytest.raises(bridge.RepositoryResourceError, match='active phases did not drain'):
            with reserve(tmp_path, supervisor) as old:
                with old.phase(demand('training'), attempt_directory=attempt(tmp_path, 'unsafe')) as phase:
                    child = subprocess.Popen([sys.executable, '-I', '-B', '-c', 'import time;time.sleep(30)'],
                                             start_new_session=True)
                    phase.daemon.check_usage(phase.attempt, child_pid=child.pid)
        root_id = old.parent.native.lease_id
        reference = weakref.ref(old)
        del old, phase
        gc.collect()
        assert reference() in pipeline.pending_pipeline_recoveries()
        assert root_id in {row['lease_id'] for row in shared.active_leases()}
        child.terminate(); child.wait(timeout=3)
        recovered = reference()
        retained = recovered.receipt()['retained_disk_reservations'][0]
        replacement_supervisor = ResourceScheduler(ResourcePolicy(max_lanes=8),
                                                   host_sampler=supervisor.host_sampler)
        with reserve(tmp_path, replacement_supervisor) as replacement:
            with replacement.phase(demand('cleanup'), attempt_directory=attempt(tmp_path, 'recovery')) as cleanup:
                cleanup.recover_retained(retained, artifacts_durable=True, owner=recovered)
                cleanup.finalize(artifacts_durable=True)
        assert recovered not in pipeline.pending_pipeline_recoveries()
        assert recovered.parent.receipt()['closed']
        assert root_id not in {row['lease_id'] for row in shared.active_leases()}
    finally:
        if child is not None and child.poll() is None:
            child.kill(); child.wait(timeout=3)
        # Failed assertions must not abandon this test's injected owner scope.
        for pending in pipeline.pending_pipeline_recoveries():
            if pending.parent.shared.state_path == shared.state_path:
                for reservation_id in list(pending._retained):
                    pending._recover(reservation_id, artifacts_durable=True)
