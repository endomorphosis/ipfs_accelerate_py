"""Separate live shared-authority acceptance from injected telemetry controls."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_resource_bridge as bridge
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import ResourceScheduler, ResourcePolicy, HostResourceSnapshot
from ipfs_datasets_py.logic.software_contracts.codebase_resources import acquire_codebase_resources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import GlobalResourceScheduler, ResourceSchedulerConfig, get_global_resource_scheduler


def local_supervisor():
    return ResourceScheduler(ResourcePolicy(max_lanes=8))


@pytest.fixture
def injected_owners(tmp_path, monkeypatch):
    """Fault-control fixture, explicitly not live host acceptance evidence."""
    shared = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path/'injected-state.json',
        proof_resource_sampler=lambda: ProofHostResources(8, 8192, 8192),
        poll_interval_seconds=.005, proof_backoff_seconds=.02))
    monkeypatch.setattr(bridge, 'get_global_resource_scheduler', lambda: shared)
    def sample(*args, **kwargs):
        return HostResourceSnapshot(memory_total_bytes=8*1024**3, memory_available_bytes=8*1024**3,
            disk_total_bytes=16*1024**3, disk_available_bytes=16*1024**3,
            worker_limit=8, available_worker_capacity=8)
    supervisor = ResourceScheduler(ResourcePolicy(max_lanes=8), host_sampler=sample)
    yield supervisor, shared
    assert not supervisor.active_leases
    assert shared.snapshot()['active_lease_count'] == shared.snapshot()['waiting_request_count'] == 0


@pytest.mark.parametrize('values', [dict(device='cuda'), dict(enforcement='hard'), dict(cpu_slots=True),
    dict(memory_mb=0), dict(wall_time_ms=0), dict(process_slots=2), dict(threads_per_process=2),
    dict(maximum_queued_phases=257), dict(maximum_queued_metadata_bytes=0)])
def test_closed_cpu_profile_refuses_unknown_or_unbounded_demands(values):
    with pytest.raises(bridge.RepositoryResourceError):
        bridge.RepositoryResourceBudget(**values)


def test_boundary_rejects_invalid_cancellation_before_admission(tmp_path):
    with pytest.raises(bridge.RepositoryResourceError, match='callable is_set'):
        with bridge.RepositoryResourceBridge(local_supervisor()).reserve(repository_id='invalid',
            workspace=tmp_path, budget=bridge.RepositoryResourceBudget(), cancel_event=object()):
            pytest.fail('invalid cancellation admitted')


def test_injected_nested_owner_accounting_and_detached_receipts(injected_owners, tmp_path):
    supervisor, shared = injected_owners
    with bridge.RepositoryResourceBridge(supervisor).reserve(repository_id='fixture',workspace=tmp_path,
            budget=bridge.RepositoryResourceBudget()) as parent:
        with parent.phase(bridge.RepositoryPhaseDemand('semantic_index')) as phase:
            with acquire_codebase_resources(**phase.native_options()) as child:
                snapshot=shared.snapshot()
                assert snapshot['active_root_lease_count']==1 and snapshot['active_child_lease_count']==2
                assert child.parent_lease_id==phase.native.lease_id
        receipt=parent.receipt()
        assert receipt['owned_active_root_count']==1
        assert receipt['host_authority']['state_path']==str(shared.state_path)
        assert 'lease_key' not in json.dumps(receipt)
        receipt['events'][0]['demand']['phase']='forged'
        assert parent.receipt()['events'][0]['demand']['phase']=='semantic_index'
    assert parent.receipt()['closed'] and parent.receipt()['owned_active_root_count']==0


@pytest.mark.parametrize('missing', [False, True])
def test_injected_pressure_and_missing_telemetry_refuse_without_leaks(injected_owners, tmp_path, missing):
    supervisor, shared=injected_owners
    def pressure():
        if missing: raise RuntimeError('telemetry unavailable')
        return ProofHostResources(8,8192,8192,memory_stall_percent=90)
    shared.config.proof_resource_sampler=pressure
    with pytest.raises(bridge.LeaseTimeoutError):
        with bridge.RepositoryResourceBridge(supervisor).reserve(repository_id='pressure',workspace=tmp_path,
                budget=bridge.RepositoryResourceBudget(wall_time_ms=100)):
            pytest.fail('pressure refused by native owner')
    reason=shared.snapshot()['proof_backoff']['reason']
    assert reason==('proof_resource_telemetry_unknown' if missing else 'proof_memory_stall')


def test_injected_global_reserve_does_not_imply_parent_validation_capacity(injected_owners,tmp_path):
    supervisor,shared=injected_owners
    signal=threading.Event()
    seen=[]
    def validation(parent):
        try:
            with parent.phase(bridge.RepositoryPhaseDemand('validation')):
                seen.append('unexpected')
        except bridge.LeaseCancelledError:
            seen.append('cancelled')
    with pytest.raises(bridge.LeaseCancelledError):
        with bridge.RepositoryResourceBridge(supervisor).reserve(repository_id='siblings',workspace=tmp_path,
                budget=bridge.RepositoryResourceBudget(maximum_queued_phases=1),cancel_event=signal) as parent:
            with parent.phase(bridge.RepositoryPhaseDemand('training',memory_mb=1024)):
                worker=threading.Thread(target=validation,args=(parent,));worker.start()
                end=time.monotonic()+3
                while shared.snapshot()['waiting_request_count']==0 and time.monotonic()<end:time.sleep(.01)
                assert shared.snapshot()['waiting_request_count']==1
                with pytest.raises(bridge.RepositoryResourceError,match='phase queue is full'):
                    with parent.phase(bridge.RepositoryPhaseDemand('scan')):pass
                assert parent.receipt()['validation_reservation_scope']=='global_only_siblings_share_parent_envelope'
                signal.set();worker.join(3)
                assert not worker.is_alive() and seen==['cancelled']


def test_injected_queue_and_phase_budget_bounds(injected_owners,tmp_path):
    supervisor,_=injected_owners
    with bridge.RepositoryResourceBridge(supervisor).reserve(repository_id='bounds',workspace=tmp_path,
            budget=bridge.RepositoryResourceBudget(maximum_phases=1)) as parent:
        with pytest.raises(bridge.RepositoryResourceError,match='phase exceeds parent'):
            with parent.phase(bridge.RepositoryPhaseDemand('scan',memory_mb=2048)):pass
        with parent.phase(bridge.RepositoryPhaseDemand('sql')):pass
        with pytest.raises(bridge.RepositoryResourceError,match='history capacity'):
            with parent.phase(bridge.RepositoryPhaseDemand('validation')):pass
    with bridge.RepositoryResourceBridge(supervisor).reserve(repository_id='bytes',workspace=tmp_path,
            budget=bridge.RepositoryResourceBudget(maximum_queued_metadata_bytes=1)) as parent:
        with pytest.raises(bridge.RepositoryResourceError,match='metadata exceeds'):
            with parent.phase(bridge.RepositoryPhaseDemand('scan')):pass


def test_live_two_repository_parents_share_actual_default_host_authority(tmp_path):
    shared=get_global_resource_scheduler()
    barrier=threading.Barrier(2)
    root_ids=[]
    def run(index):
        supervisor=local_supervisor()
        with bridge.RepositoryResourceBridge(supervisor).reserve(repository_id='rpi022-live-'+str(index),
                workspace=tmp_path,budget=bridge.RepositoryResourceBudget(wall_time_ms=120000)) as parent:
            root_ids.append(parent.native.lease_id)
            with parent.phase(bridge.RepositoryPhaseDemand('semantic_index')) as phase:
                with acquire_codebase_resources(**phase.native_options()) as grandchild:
                    barrier.wait(timeout=120)
                    leases=shared.active_leases()
                    ours=[row for row in leases if row['lease_id'] in root_ids]
                    assert len(ours)==2 and all(row.get('parent_lease_id') is None for row in ours)
                    assert grandchild.parent_lease_id==phase.native.lease_id
                    receipt=parent.receipt()
                    assert receipt['host_authority']['state_path']==str(shared.state_path)
                    assert receipt['owned_active_root_count']==1 and receipt['validation_reserve']['cpu_slots']>=1
                    barrier.wait(timeout=120)
            result=parent.receipt()
        assert not supervisor.active_leases and parent.receipt()['closed']
        return result
    with ThreadPoolExecutor(max_workers=2) as executor:
        rows=list(executor.map(run,range(2)))
    assert len({row['host_authority']['root_lease_id'] for row in rows})==2
    assert not set(root_ids)&{row['lease_id'] for row in shared.active_leases()}


def test_live_killed_parent_is_reaped_by_actual_shared_owner(tmp_path):
    script=r'''
import json,sys,time
from ipfs_accelerate_py.agent_supervisor.runtime.repository_resource_bridge import *
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import ResourceScheduler,ResourcePolicy
with RepositoryResourceBridge(ResourceScheduler(ResourcePolicy(max_lanes=8))).reserve(repository_id='rpi022-crash',workspace=sys.argv[1],budget=RepositoryResourceBudget(wall_time_ms=120000)) as parent:
 with parent.phase(RepositoryPhaseDemand('scan')) as phase:
  print(json.dumps({'root':parent.native.lease_id,'child':phase.native.lease_id,'state':str(parent.shared.state_path)}),flush=True)
  time.sleep(120)
'''
    process=subprocess.Popen([sys.executable,'-c',script,str(tmp_path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        import select
        ready,_,_=select.select([process.stdout],[],[],130)
        assert ready,'live native admission did not complete in the declared deadline'
        line=process.stdout.readline()
        assert line,process.stderr.read()
        identity=json.loads(line)
        shared=get_global_resource_scheduler()
        assert identity['state']==str(shared.state_path)
        process.kill();process.wait(timeout=10)
        recovered=shared.recover_stale_leases()
        assert {identity['root'],identity['child']}<=set(recovered)
        assert not {identity['root'],identity['child']}&{r['lease_id'] for r in shared.active_leases()}
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=10)


def test_live_cancellation_terminates_actual_bounded_child_and_releases_owners(tmp_path):
    from ipfs_datasets_py.logic.backends.codebase_process import BoundedToolRunner, ToolRunLimits
    supervisor=local_supervisor()
    results=[]
    ready=tmp_path/'actual-child-ready'
    def work(parent):
        try:
            with parent.phase(bridge.RepositoryPhaseDemand('validation')) as phase:
                result=BoundedToolRunner().run([str(Path(sys.executable).resolve()),'-I','-B','-c',
                    'import pathlib,sys,time;pathlib.Path(sys.argv[1]).write_text("ready");time.sleep(120)',str(ready)],
                    cancellation=parent.cancellation,limits=ToolRunLimits(timeout_seconds=15,
                        memory_bytes=512*bridge.MIB,resident_memory_bytes=512*bridge.MIB,cpu_seconds=15),
                    environment=phase.thread_environment())
                results.append(result)
        except bridge.LeaseCancelledError:
            results.append('cancelled')
        except BaseException as error:
            results.append(error)
    with pytest.raises(bridge.LeaseCancelledError):
        with bridge.RepositoryResourceBridge(supervisor).reserve(repository_id='rpi022-live-cancel',workspace=tmp_path,
                budget=bridge.RepositoryResourceBudget(wall_time_ms=120000)) as parent:
            thread=threading.Thread(target=work,args=(parent,));thread.start()
            deadline=time.monotonic()+15
            while not ready.exists() and thread.is_alive() and time.monotonic()<deadline:time.sleep(.02)
            assert ready.exists(),('bounded native child did not launch',results)
            parent.cancel();thread.join(10)
            assert not thread.is_alive()
            assert len(results)==2 and results[1]=='cancelled'
            assert results[0].cancelled and results[0].process_tree_terminated and results[0].workspace_cleaned
    assert parent.receipt()['closed'] and not supervisor.active_leases
    assert parent.receipt()['owned_active_root_count']==0
