"""Real file-backed consumers; controlled telemetry is explicitly injected."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.agent_supervisor.container_coding import repository_benchmark_preparation as prep
from ipfs_accelerate_py.agent_supervisor.runtime import repository_resource_handoff as handoff
from ipfs_accelerate_py.agent_supervisor.runtime.repository_pipeline_resources import (
    RepositoryPipelineResources, PipelineResourcePolicy,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_resource_bridge import (
    RepositoryResourceBudget, RepositoryPhaseDemand, LeaseCancelledError,
)
from ipfs_datasets_py.logic.software_contracts.codebase_scan_policy import CodebaseScanPolicy
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLane
from test.api.test_repository_pipeline_resources import injected_owners


@contextmanager
def managed(tmp_path, injected_owners, monkeypatch):
    supervisor, shared = injected_owners
    monkeypatch.setattr(handoff, 'get_global_resource_scheduler', lambda: shared)
    disk = tmp_path/'disk'; disk.mkdir(exist_ok=True)
    attempts = disk/'attempts'; attempts.mkdir(exist_ok=True)
    with RepositoryPipelineResources(supervisor).reserve(repository_id='repository:managed', workspace=tmp_path,
            budget=RepositoryResourceBudget(cpu_slots=2, memory_mb=3072, process_slots=2,
                disk_bytes=256*1024**2, wall_time_ms=20000),
            policy=PipelineResourcePolicy(protected_memory_mb=1024, protected_disk_bytes=64*1024**2),
            ledger_path=tmp_path/'ledger.json', roots=[disk]) as owner:
        yield owner, attempts


def native_index(tmp_path):
    import duckdb
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog
    from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
    from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
    from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
    repository=tmp_path/'repository'; repository.mkdir()
    (repository/'calc.py').write_text('def add(n: int) -> int:\n    return n + 1\n')
    (repository/'opaque.bin').write_bytes(b'\x00\xff')
    for arguments in (('init','-q'),('config','user.name','Managed fixture'),
            ('config','user.email','fixture@example.invalid'),('add','.'),('commit','-qm','fixture')):
        subprocess.run(['git','-C',str(repository),*arguments],check=True,capture_output=True)
    cx=duckdb.connect(str(tmp_path/'disk'/'source.duckdb'),config={'threads':1,'memory_limit':'64MB'})
    store=DuckDBASTStore(connection=cx)
    artifacts=ImmutableCAS(tmp_path/'disk'/'cas')
    index=RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=artifacts,
                                 catalog=CodebaseCatalog(store,artifacts))
    return index, repository, cx


def arguments(index, repository, owner, attempts):
    return dict(index=index, repository=repository, repository_id=owner.parent.repository_id,
        expected_head=None, operation_id='managed-model-off',
        selection=prep.RepositoryPreparationSelection(CodebaseScanPolicy()),
        envelope=owner.parent, budget=prep.PreparationBudget(memory_mb=1024),
        pipeline=owner, pipeline_attempt_root=attempts)


def test_managed_preparation_runs_actual_scan_catalog_and_validation(injected_owners,tmp_path,monkeypatch):
    with managed(tmp_path,injected_owners,monkeypatch) as (owner,attempts):
        index, repository, cx=native_index(tmp_path)
        try:
            result=prep.prepare_repository_benchmark(**arguments(index,repository,owner,attempts))
            assert result['qualified'] and result['model']['identity']=='explicit-model-off@1'
            assert result['complete_inventory']['inventory_entries']==2
            assert [s['phase'] for s in result['stages']]==['scan','semantic_index','sql','validation']
            receipt=result['managed_pipeline']['resource_receipt']
            assert receipt['root']['owned_active_root_count']==1
            assert receipt['owned_disk_bytes']==4*16*1024**2
            assert all(s['managed_resources']['payload_bytes']>0 for s in result['stages'])
            assert all(s['managed_resources']['sampled_named_root_growth_bytes']<=16*1024**2 for s in result['stages'])
            assert all(e['disk']['status']=='released' for e in receipt['events'])
            assert result['training'] is result['inference'] is None
            assert result['managed_pipeline']['external_process_rss_captured'] is False
            assert owner.parent.remaining()>0
        finally:
            cx.close()


@pytest.mark.parametrize('damage',['wrong_owner','outside_attempt','outside_cas','zero_write','boolean_write',
    'unselected_model','unselected_proof'])
def test_managed_preparation_refuses_scope_before_source_publication(injected_owners,tmp_path,monkeypatch,damage):
    with managed(tmp_path,injected_owners,monkeypatch) as (owner,attempts):
        index, repository, cx=native_index(tmp_path)
        try:
            values=arguments(index,repository,owner,attempts)
            original_root=index.artifacts.root
            if damage=='wrong_owner':values['pipeline']=object()
            elif damage=='outside_attempt':values['pipeline_attempt_root']=repository
            elif damage=='outside_cas':index.artifacts.root=repository
            elif damage=='zero_write':values['pipeline_write_budget_bytes']=0
            elif damage=='unselected_model':values['model']=object()
            elif damage=='unselected_proof':values['checked_cache']=object()
            else:values['pipeline_write_budget_bytes']=True
            with pytest.raises(ValueError):prep.prepare_repository_benchmark(**values)
            index.artifacts.root=original_root
            assert index.current(owner.parent.repository_id) is None
            assert owner.receipt()['events']==[]
        finally:
            cx.close()


def test_managed_failure_retains_precharged_native_disk_claim(injected_owners,tmp_path,monkeypatch):
    with managed(tmp_path,injected_owners,monkeypatch) as (owner,attempts):
        index, repository, cx=native_index(tmp_path)
        try:
            def fail(*args,**kwargs):
                raise RuntimeError('injected scan failure')
            monkeypatch.setattr(prep,'prepare_policy_current',fail)
            with pytest.raises(RuntimeError,match='injected scan failure'):
                prep.prepare_repository_benchmark(**arguments(index,repository,owner,attempts))
            receipt=owner.receipt()
            assert len(receipt['retained_disk_reservations'])==1
            assert receipt['owned_disk_bytes']==16*1024**2
            assert receipt['events'][0]['disk']['record']['external_charges']
            assert receipt['root']['owned_active_root_count']==1
        finally:
            cx.close()


@contextmanager
def worker_phase(owner,attempts):
    path=attempts/'worker';path.mkdir()
    with owner.phase(RepositoryPhaseDemand('validation',memory_mb=1024,disk_bytes=16*1024**2),
                     payload=b'actual worker input',attempt_directory=path) as phase:
        yield phase
        phase.finalize(artifacts_durable=True)


def grant(phase,tmp_path):
    private=tmp_path/'private';private.mkdir(mode=0o700)
    return handoff.write_pipeline_resource_grant(phase=phase,directory=private,task_cid='task:managed')


def grant_args(saved):
    return dict(artifact=saved['artifact'],expected_sha256=saved['sha256'],
                repository_id='repository:managed',task_cid='task:managed')


def reseal(saved,mutate):
    path=Path(saved['artifact'])
    value=json.loads(path.read_bytes())
    mutate(value)
    raw=handoff._wire(value)
    digest=hashlib.sha256(raw).hexdigest()
    replacement=path.parent/(digest+'.json')
    replacement.write_bytes(raw);replacement.chmod(0o400)
    return {**saved,'artifact':str(replacement),'sha256':digest}


def test_v2_actual_consumer_authenticates_daemon_and_all_native_ancestors(injected_owners,tmp_path,monkeypatch):
    with managed(tmp_path,injected_owners,monkeypatch) as (owner,attempts):
        with worker_phase(owner,attempts) as phase:
            saved=grant(phase,tmp_path)
            with handoff.delegated_supported_repository_phase(**grant_args(saved)) as consumer:
                assert consumer.native.parent_lease_id==phase.daemon.native_lease.lease_id
                with consumer.native.acquire_child(lane=ResourceLane.VALIDATION,memory_mb=128,
                        cpu_slots=1,child_process_slots=1,timeout=1) as grandchild:
                    assert grandchild.parent_lease_id==consumer.native.lease_id
                assert owner.parent.shared.snapshot()['active_root_lease_count']==1
            receipt=consumer.receipt()
            assert receipt['lease']['released']
            assert receipt['root_lease_id']==owner.parent.native.lease_id
            assert receipt['bridge_phase_lease_id']==phase.native.native.lease_id
            assert receipt['daemon_reservation_id']==phase.daemon.reservation_id
            assert 'parent_token' not in json.dumps(receipt) and 'lease_key' not in json.dumps(receipt)
            assert phase.daemon.native_lease.released is False


@pytest.mark.parametrize('damage',['root','bridge','daemon_id','token','phase','capacity','task',
    'repository','owner','expires','version','unknown','mode','hardlink','directory'])
def test_v2_resealed_or_unsafe_grant_cannot_admit_native_consumer(injected_owners,tmp_path,monkeypatch,damage):
    with managed(tmp_path,injected_owners,monkeypatch) as (owner,attempts):
        with worker_phase(owner,attempts) as phase:
            saved=grant(phase,tmp_path)
            def mutate(value):
                if damage=='root':value['root_lease_id']='unowned'
                elif damage=='bridge':value['bridge_phase_lease_id']=value['root_lease_id']
                elif damage=='daemon_id':value['daemon_reservation_id']='wrong'
                elif damage=='token':value['parent_token']['lease_key']='0'*64
                elif damage=='phase':value['phase']['phase']='training'
                elif damage=='capacity':value['phase']['memory_mb']=512
                elif damage=='task':value['task_cid']='task:foreign'
                elif damage=='repository':value['repository_id']='repository:foreign'
                elif damage=='owner':value['owner_pid']=True
                elif damage=='expires':value['expires_monotonic']=True
                elif damage=='version':value['schema']='repository-private-resource-grant@99'
                elif damage=='unknown':value['additional']=True
            if damage=='mode':Path(saved['artifact']).chmod(0o600)
            elif damage=='hardlink':os.link(saved['artifact'],str(saved['artifact'])+'.link')
            elif damage=='directory':Path(saved['artifact']).parent.chmod(0o750)
            else:saved=reseal(saved,mutate)
            before=owner.parent.shared.snapshot()['active_lease_count']
            with pytest.raises(Exception):
                with handoff.delegated_supported_repository_phase(**grant_args(saved)):
                    pytest.fail('damaged grant acquired native capacity')
            assert owner.parent.shared.snapshot()['active_lease_count']==before


def test_v2_cancellation_and_released_phase_refuse_delegation(injected_owners,tmp_path,monkeypatch):
    with managed(tmp_path,injected_owners,monkeypatch) as (owner,attempts):
        with worker_phase(owner,attempts) as phase:
            saved=grant(phase,tmp_path)
        with pytest.raises(ValueError,match='no longer live'):
            with handoff.delegated_supported_repository_phase(**grant_args(saved)):
                pytest.fail('released daemon was reused')


def test_v1_dispatch_retains_legacy_exact_root_behavior(injected_owners,tmp_path,monkeypatch):
    with managed(tmp_path,injected_owners,monkeypatch) as (owner,attempts):
        private=tmp_path/'private';private.mkdir(mode=0o700)
        saved=handoff.write_repository_resource_grant(envelope=owner.parent,directory=private,
            task_cid='task:managed',demand=RepositoryPhaseDemand('validation',memory_mb=512))
        with handoff.delegated_supported_repository_phase(**grant_args(saved)) as consumer:
            assert type(consumer) is handoff.DelegatedRepositoryPhase
            assert consumer.native.parent_lease_id==owner.parent.native.lease_id
        assert consumer.native.released


def test_managed_sampled_growth_over_ceiling_retains_failed_claim(injected_owners,tmp_path,monkeypatch):
    with managed(tmp_path,injected_owners,monkeypatch) as (owner,attempts):
        index,repository,cx=native_index(tmp_path)
        original=prep.prepare_policy_current
        def grow(*args,**kwargs):
            result=original(*args,**kwargs)
            (tmp_path/'disk'/'oversized-output').write_bytes(b'x'*(2*1024**2))
            return result
        monkeypatch.setattr(prep,'prepare_policy_current',grow)
        try:
            values=arguments(index,repository,owner,attempts)
            values['pipeline_write_budget_bytes']=1024**2
            with pytest.raises(ValueError,match='sampled named-root growth'):
                prep.prepare_repository_benchmark(**values)
            assert owner.receipt()['retained_disk_reservations']
            assert owner.receipt()['events'][0]['disk']['status']=='retained'
        finally:
            cx.close()


def test_v2_actual_cross_process_child_joins_same_controlled_native_owner(injected_owners,tmp_path,monkeypatch):
    with managed(tmp_path,injected_owners,monkeypatch) as (owner,attempts):
        with worker_phase(owner,attempts) as phase:
            saved=grant(phase,tmp_path)
            request=tmp_path/'request.json'; request.write_text(json.dumps(grant_args(saved)))
            script='''
import json,sys
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.runtime import repository_resource_handoff as handoff
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import GlobalResourceScheduler,ResourceSchedulerConfig
shared=GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(state_path=Path(sys.argv[2]),
 proof_resource_sampler=lambda:ProofHostResources(8,8192,8192),poll_interval_seconds=.005,proof_backoff_seconds=.02))
handoff.get_global_resource_scheduler=lambda:shared
with handoff.delegated_supported_repository_phase(**json.loads(Path(sys.argv[1]).read_text())) as phase:
 with phase.native.acquire_child(memory_mb=128,cpu_slots=1,child_process_slots=1,timeout=2) as child:
  assert child.parent_lease_id==phase.native.lease_id
receipt=phase.receipt()
print(json.dumps(receipt))
'''
            result=subprocess.run([sys.executable,'-B','-P','-c',script,str(request),str(owner.parent.shared.state_path)],
                                  capture_output=True,text=True,timeout=15)
            assert result.returncode==0,result.stderr[-2048:]
            receipt=json.loads(result.stdout)
            assert receipt['root_lease_id']==owner.parent.native.lease_id
            assert receipt['lease']['released'] and receipt['private_token_disclosed'] is False
            assert 'lease_key' not in result.stdout and 'parent_token' not in result.stdout


def test_v2_ancestor_cancel_propagates_to_actual_consumer(injected_owners,tmp_path,monkeypatch):
    with pytest.raises(LeaseCancelledError):
        with managed(tmp_path,injected_owners,monkeypatch) as (owner,attempts):
            with worker_phase(owner,attempts) as phase:
                saved=grant(phase,tmp_path)
                with handoff.delegated_supported_repository_phase(**grant_args(saved)) as consumer:
                    owner.parent.cancel()
                    assert consumer.is_set()
                    consumer.remaining()
