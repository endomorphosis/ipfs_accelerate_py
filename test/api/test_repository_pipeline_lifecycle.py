"""Process-local lifecycle controls; native telemetry injection stays explicit."""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import json
import subprocess
import sys
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_pipeline_resources as pipeline
from ipfs_accelerate_py.agent_supervisor.runtime import repository_resource_bridge as bridge
from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_daemon_resources as daemon
from test.api.test_repository_pipeline_resources import injected_owners, budget, demand, reserve, attempt, eventually


@pytest.fixture
def small_inventory(monkeypatch):
    assert pipeline.pipeline_lifecycle_inventory()['reserved']==0
    monkeypatch.setattr(pipeline,'MAX_PIPELINE_LIFECYCLES',3)
    yield
    assert pipeline.pipeline_lifecycle_inventory()['reserved']==0
    assert pipeline.pending_pipeline_recoveries()==()


def args(tmp_path,**changes):
    roots=tmp_path/'disk';roots.mkdir(exist_ok=True)
    return dict(repository_id='lifecycle-cap',workspace=tmp_path,budget=budget(),
        policy=pipeline.PipelineResourcePolicy(),ledger_path=tmp_path/'ledger.json',roots=[roots],**changes)


def test_cap_precedes_concurrent_host_admission_and_failed_entries_release(small_inventory,injected_owners,tmp_path,monkeypatch):
    supervisor,_=injected_owners;adapter=pipeline.RepositoryPipelineResources(supervisor)
    entered=[];errors=[];release=threading.Event()
    @contextmanager
    def admission(**kwargs):
        entered.append(kwargs['repository_id'])
        assert release.wait(3)
        raise RuntimeError('injected native admission refusal')
        yield
    monkeypatch.setattr(adapter.bridge,'reserve',admission)
    options=args(tmp_path)
    def call():
        try:
            with adapter.reserve(**options):pytest.fail('refused host entry yielded')
        except BaseException as error:errors.append(error)
    threads=[threading.Thread(target=call) for _ in range(8)]
    for thread in threads:thread.start()
    try:
        eventually(lambda:len(entered)==2 and len(errors)==6)
        state=pipeline.pipeline_lifecycle_inventory()
        assert state['ordinary']==state['awaiting_host_admission']==state['reserved']==2
        assert all('lifecycle capacity exhausted' in str(error) for error in errors)
    finally:
        release.set()
        for thread in threads:thread.join(4)
    assert all(not thread.is_alive() for thread in threads)
    assert len(errors)==8 and sum(isinstance(e,RuntimeError) and str(e)=='injected native admission refusal' for e in errors)==2
    assert pipeline.pipeline_lifecycle_inventory()['reserved']==0


@pytest.mark.parametrize('value',[1,None,'yes'])
def test_recovery_mode_requires_exact_boolean_before_host_admission(small_inventory,injected_owners,tmp_path,value):
    supervisor,_=injected_owners
    with pytest.raises(bridge.RepositoryResourceError,match='Boolean'):
        with pipeline.RepositoryPipelineResources(supervisor).reserve(**args(tmp_path,recovery_only=value)):
            pytest.fail('nonboolean recovery mode admitted')


def test_injected_native_expiry_does_not_free_retained_slots_and_reserved_cleanup_reaps(small_inventory,injected_owners,tmp_path):
    supervisor,shared=injected_owners
    # Real file-backed native leases expire. Only host telemetry and this short
    # test lease lifetime differ; old Python/native contexts remain reachable.
    shared.config=replace(shared.config,lease_ttl_seconds=.2,auto_renew_leases=False)
    children=[];olds=[]
    try:
        for number in range(2):
            with pytest.raises(bridge.RepositoryResourceError,match='active phases did not drain'):
                with reserve(tmp_path,supervisor if number==0 else type(supervisor)(supervisor.policy,host_sampler=supervisor.host_sampler)) as old:
                    olds.append(old)
                    with old.phase(demand('training'),attempt_directory=attempt(tmp_path,'unsafe-'+str(number))) as phase:
                        child=subprocess.Popen([sys.executable,'-I','-B','-c','import time;time.sleep(45)'],start_new_session=True)
                        children.append(child)
                        phase.daemon.check_usage(phase.attempt,child_pid=child.pid)
            assert child.poll() is None
            assert shared.snapshot()['active_lease_count']==0
            assert old in pipeline.pending_pipeline_recoveries()
            assert pipeline.pipeline_lifecycle_inventory()['ordinary']==number+1
        with pytest.raises(bridge.RepositoryResourceError,match='lifecycle capacity exhausted'):
            with reserve(tmp_path,supervisor):pytest.fail('expired native capacity hid retained Python owners')
        shared.config=replace(shared.config,lease_ttl_seconds=120,auto_renew_leases=True)
        options=args(tmp_path,recovery_only=True);options['ledger_path']=tmp_path/'disk-ledger.json'
        recovery_supervisor=type(supervisor)(supervisor.policy,host_sampler=supervisor.host_sampler)
        with pipeline.RepositoryPipelineResources(recovery_supervisor).reserve(**options) as recovery:
            assert pipeline.pipeline_lifecycle_inventory()['reserved']==3
            with pytest.raises(bridge.RepositoryResourceError,match='lifecycle capacity exhausted'):
                with pipeline.RepositoryPipelineResources(recovery_supervisor).reserve(**options):
                    pytest.fail('second cleanup lifecycle borrowed an ordinary slot')
            for kind in ('training','validation','scan'):
                with pytest.raises(bridge.RepositoryResourceError,match='cleanup phases only'):
                    with recovery.phase(demand(kind),attempt_directory=attempt(tmp_path,'forbidden-'+kind)):
                        pytest.fail('recovery compartment used for workload phase')
            with recovery.phase(demand('cleanup'),attempt_directory=attempt(tmp_path,'cleanup')) as cleanup:
                with pytest.raises(bridge.RepositoryResourceError,match='delegate consumer'):
                    cleanup.native_options()
                with pytest.raises(bridge.RepositoryResourceError,match='workload processes'):
                    cleanup.run(['/bin/true'],timeout_seconds=1)
                for old,child in zip(olds,children):
                    retained=old.receipt()['retained_disk_reservations'][0]
                    with pytest.raises(daemon.DaemonResourceError,match='still alive'):
                        cleanup.recover_retained(retained,artifacts_durable=True,owner=old)
                    child.terminate();child.wait(timeout=3)
                    cleanup.recover_retained(retained,artifacts_durable=True,owner=old)
                assert pipeline.pipeline_lifecycle_inventory()['ordinary']==0
                cleanup.finalize(artifacts_durable=True)
        assert pipeline.pipeline_lifecycle_inventory()['reserved']==0
        assert all(old.parent._closed and not old.parent.supervisor.active_leases for old in olds)
        assert not recovery_supervisor.active_leases
    finally:
        for child in children:
            if child.poll() is None:child.kill();child.wait(timeout=3)
        for old in olds:
            for reservation in list(old._retained):old._recover(reservation,artifacts_durable=True)
            if old._outer_exited and not old._active and not old._queued and not old._retained_contexts:
                old.reap_lifecycle()


def test_injected_idle_native_close_failure_retains_slot_until_explicit_reap(small_inventory,injected_owners,tmp_path,monkeypatch):
    supervisor,shared=injected_owners;old=None;close=None
    try:
        with pytest.raises(RuntimeError,match='injected close failure'):
            with reserve(tmp_path,supervisor) as old:
                close=old.parent.close
                monkeypatch.setattr(old.parent,'close',lambda:(_ for _ in ()).throw(RuntimeError('injected close failure')))
        assert old in pipeline.pending_pipeline_recoveries()
        assert pipeline.pipeline_lifecycle_inventory()['ordinary']==1
        assert shared.snapshot()['active_root_lease_count']==1
        monkeypatch.setattr(old.parent,'close',close)
        old.reap_lifecycle()
        assert pipeline.pipeline_lifecycle_inventory()['reserved']==0
        assert shared.snapshot()['active_root_lease_count']==0
    finally:
        if old is not None and close is not None:
            monkeypatch.setattr(old.parent,'close',close)
            if old._outer_exited:old.reap_lifecycle()


def test_injected_normal_repeated_reservations_release_lifecycle_without_changing_disk_authority(small_inventory,injected_owners,tmp_path):
    supervisor,_=injected_owners
    for number in range(5):
        with reserve(tmp_path,supervisor) as owner:
            assert pipeline.pipeline_lifecycle_inventory()['ordinary']==1
            with owner.phase(demand('validation'),attempt_directory=attempt(tmp_path,'normal-'+str(number))) as phase:
                phase.finalize(artifacts_durable=True)
        assert pipeline.pipeline_lifecycle_inventory()['reserved']==0
        receipt=owner.receipt()
        assert receipt['lifecycle']['host_resource_authority'] is False
        assert 'lease_key' not in json.dumps(receipt)


def test_injected_queued_scope_stays_counted_while_outer_close_waits_for_active_scope(small_inventory,injected_owners,tmp_path):
    supervisor,shared=injected_owners
    entered=threading.Event();finish=threading.Event();errors=[];closed=[]
    context=reserve(tmp_path,supervisor);owner=context.__enter__()
    first_attempt=attempt(tmp_path,'race-active');second_attempt=attempt(tmp_path,'race-queued')
    def active():
        try:
            with owner.phase(demand('training'),attempt_directory=first_attempt) as phase:
                entered.set();assert finish.wait(4)
                phase.finalize(artifacts_durable=True)
        except BaseException as error:errors.append(error)
    def queued():
        try:
            with owner.phase(demand('scan'),payload=b'queued',attempt_directory=second_attempt):
                pytest.fail('cancelled queued scope was admitted')
        except BaseException as error:errors.append(error)
    def close():
        try:context.__exit__(None,None,None)
        except BaseException as error:closed.append(error)
    first=threading.Thread(target=active);second=threading.Thread(target=queued);closer=threading.Thread(target=close)
    first.start()
    try:
        assert entered.wait(2);second.start()
        eventually(lambda:owner.receipt()['queued_phases']==1)
        assert pipeline.pipeline_lifecycle_inventory()['ordinary']==1
        closer.start()
        eventually(lambda:owner._closed and len(errors)==1)
        # The queued thread has observed native cancellation, while the active
        # scope deliberately has not drained yet. No slot can be recycled.
        assert owner.receipt()['active_phases']==1
        assert pipeline.pipeline_lifecycle_inventory()['ordinary']==1
        assert not owner._outer_exited and closer.is_alive()
    finally:
        finish.set();first.join(3)
        if second.ident is not None:second.join(3)
        if closer.ident is not None:closer.join(3)
        else:context.__exit__(None,None,None)
        if owner._outer_exited and not owner._active and not owner._queued and not owner._retained_contexts:
            owner.reap_lifecycle()
    assert not first.is_alive() and not second.is_alive() and not closer.is_alive()
    assert not closed and len(errors)==2 and all(isinstance(error,bridge.LeaseCancelledError) for error in errors)
    assert owner.receipt()['retained_disk_reservations']  # Durable claims were not silently released.
    assert pipeline.pipeline_lifecycle_inventory()['reserved']==0
    assert shared.snapshot()['active_lease_count']==0
