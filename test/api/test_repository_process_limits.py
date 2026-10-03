"""Kernel outcomes use real children; controlled host sampling is opt-in only."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import sys
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_pipeline_resources as pipeline
from ipfs_accelerate_py.agent_supervisor.runtime import repository_resource_bridge as bridge
from ipfs_accelerate_py.agent_supervisor.runtime import repository_process_limits as owner
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    ResourceScheduler, ResourcePolicy, HostResourceSnapshot,
)
from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_daemon_resources as daemon
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceSchedulerConfig, get_global_resource_scheduler,
)
from ipfs_datasets_py.logic.backends import codebase_process

MIB = 1024 * 1024
PYTHON = str(Path(sys.executable).resolve())


@pytest.mark.parametrize('cpu,memory', [(True,64*MIB),(0,64*MIB),(1.0,64*MIB),
    (3601,64*MIB),(1,True),(1,15*MIB),(1,64.0*MIB),(1,2**40+1)])
def test_closed_profile_refuses_invalid_dimensions(cpu,memory):
    with pytest.raises(ValueError):owner.RepositoryProcessLimits(cpu,memory)


@pytest.mark.parametrize('changes', [dict(memory_mb=32),dict(timeout_seconds=.1,threads_per_process=1),
    dict(memory_mb=True),dict(timeout_seconds=float('nan')),dict(disk_bytes=0)])
def test_profile_cannot_exceed_phase_or_accept_unknown_dimensions(changes):
    arguments=dict(memory_mb=128,threads_per_process=1,timeout_seconds=5,disk_bytes=MIB)
    arguments.update(changes)
    with pytest.raises(ValueError):owner.RepositoryProcessLimits(2,64*MIB).for_phase(**arguments)


def test_explicit_profile_refuses_nonlinux(monkeypatch):
    monkeypatch.setattr(owner.sys,'platform','unsupported')
    with pytest.raises(ValueError,match='Linux'):
        owner.RepositoryProcessLimits(1,64*MIB).for_phase(memory_mb=128,
            threads_per_process=1,timeout_seconds=5,disk_bytes=MIB)


@pytest.fixture
def resource_owners(tmp_path,monkeypatch):
    controlled=os.environ.get('RPI_PROCESS_LIMITS_CONTROLLED_HOST')=='1'
    if controlled:
        shared=GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
            state_path=tmp_path/'controlled-native.json',
            proof_resource_sampler=lambda:ProofHostResources(8,8192,8192),
            poll_interval_seconds=.005,proof_backoff_seconds=.02))
        monkeypatch.setattr(bridge,'get_global_resource_scheduler',lambda:shared)
        monkeypatch.setattr(daemon,'get_global_resource_scheduler',lambda:shared)
        supervisor=ResourceScheduler(ResourcePolicy(max_lanes=8),host_sampler=lambda *a,**kw:
            HostResourceSnapshot(memory_total_bytes=8*1024**3,memory_available_bytes=8*1024**3,
                disk_total_bytes=16*1024**3,disk_available_bytes=16*1024**3,
                worker_limit=8,available_worker_capacity=8))
    else:
        shared=get_global_resource_scheduler()
        supervisor=ResourceScheduler(ResourcePolicy(max_lanes=8))
    yield supervisor,shared
    assert not supervisor.active_leases
    assert not [r for r in shared.active_leases()
        if r.get('request_id','').startswith('repository:rpi022-process-limits:')
        and r['owner_pid']==os.getpid()]


@contextmanager
def admitted_phase(tmp_path,resource_owners):
    supervisor,shared=resource_owners
    roots=tmp_path/'disk';roots.mkdir()
    attempt=roots/'attempt';attempt.mkdir()
    budget=bridge.RepositoryResourceBudget(cpu_slots=3,process_slots=3,memory_mb=512,
        disk_bytes=16*MIB,wall_time_ms=60000)
    with pipeline.RepositoryPipelineResources(supervisor).reserve(repository_id='rpi022-process-limits',
            workspace=tmp_path,budget=budget,policy=pipeline.PipelineResourcePolicy(),
            ledger_path=tmp_path/'disk-ledger.json',roots=[roots]) as parent:
        with parent.phase(bridge.RepositoryPhaseDemand('training',cpu_slots=2,
                process_slots=2,memory_mb=128,disk_bytes=MIB),attempt_directory=attempt) as phase:
            yield phase
            phase.finalize(artifacts_durable=True)
        receipt=parent.receipt()
        assert receipt['active_phases']==receipt['retained_host_phase_count']==0
        assert receipt['retained_disk_reservations']==[]
        assert not list(attempt.iterdir())


LIMITS="""import json,resource
print(json.dumps({name:list(resource.getrlimit(getattr(resource,'RLIMIT_'+name))) for name in ('CPU','AS','FSIZE','CORE')}),flush=True)
"""


def retain(phase,result,name='hard'):
    """Public authored-fixture output; no native owner keys or environment."""
    value=dict(result=result.to_dict(),enforcement=phase.last_process_enforcement,
        host_sampler='injected' if os.environ.get('RPI_PROCESS_LIMITS_CONTROLLED_HOST')=='1' else 'actual_default')
    (phase.attempt.parent.parent/('process-'+name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


def test_native_pipeline_inherits_exact_limits_and_default_opt_out(tmp_path,resource_owners):
    with admitted_phase(tmp_path,resource_owners) as phase:
        hard=owner.RepositoryProcessLimits(2,64*MIB)
        result=phase.run([PYTHON,'-I','-c',LIMITS],timeout_seconds=5,hard_limits=hard)
        assert result.ok,result
        assert json.loads(result.stdout)==dict(CPU=[2,2],AS=[64*MIB]*2,FSIZE=[MIB]*2,CORE=[0,0])
        receipt=phase.last_process_enforcement
        assert receipt['native_process_started'] and receipt['workspace_cleaned']
        assert receipt['kernel_per_process']['cpu_seconds']==2
        assert not receipt['aggregate_kernel_enforcement'] and not receipt['execution_authority']
        retain(phase,result)
        default=phase.run([PYTHON,'-I','-c',LIMITS],timeout_seconds=5)
        assert default.ok and phase.last_process_enforcement is None
        assert json.loads(default.stdout)['AS']==[-1,-1]
        retain(phase,default,'default')


def test_native_cpu_limit_stops_actual_busy_loop(tmp_path,resource_owners):
    with admitted_phase(tmp_path,resource_owners) as phase:
        result=phase.run([PYTHON,'-I','-c',LIMITS+'while True: pass\n'],timeout_seconds=8,
            hard_limits=owner.RepositoryProcessLimits(1,64*MIB))
        assert result.returncode==-signal.SIGKILL and not result.timed_out,result
        assert json.loads(result.stdout)['CPU']==[1,1]
        assert result.workspace_cleaned and phase.check_usage()['group_rss']['live_processes']==0
        retain(phase,result)


def test_native_address_space_refuses_actual_allocation(tmp_path,resource_owners):
    script=LIMITS+'''try:
 value=bytearray(128*1024*1024)
except MemoryError:
 print('ADDRESS_SPACE_REFUSED',flush=True)
else:
 raise SystemExit('unbounded allocation')
'''
    with admitted_phase(tmp_path,resource_owners) as phase:
        result=phase.run([PYTHON,'-I','-c',script],timeout_seconds=5,
            hard_limits=owner.RepositoryProcessLimits(2,64*MIB))
        assert result.ok and 'ADDRESS_SPACE_REFUSED' in result.stdout,result
        assert json.loads(result.stdout.splitlines()[0])['AS']==[64*MIB]*2
        retain(phase,result)


def test_native_file_size_refuses_actual_write(tmp_path,resource_owners):
    script=LIMITS+'''import os,signal,errno
signal.signal(signal.SIGXFSZ,signal.SIG_IGN)
fd=os.open('bounded-file',os.O_WRONLY|os.O_CREAT,0o600)
try:
 for _ in range(32): os.write(fd,b'x'*65536)
except OSError as error:
 assert error.errno==errno.EFBIG
 assert os.fstat(fd).st_size==1024*1024
 print('FILE_SIZE_REFUSED',flush=True)
else:
 raise SystemExit('unbounded file')
finally:
 os.close(fd)
'''
    with admitted_phase(tmp_path,resource_owners) as phase:
        result=phase.run([PYTHON,'-I','-c',script],timeout_seconds=5,
            hard_limits=owner.RepositoryProcessLimits(2,64*MIB))
        assert result.ok and 'FILE_SIZE_REFUSED' in result.stdout,result
        assert result.workspace_cleaned
        retain(phase,result)


def test_native_wall_timeout_retains_process_cleanup(tmp_path,resource_owners):
    with admitted_phase(tmp_path,resource_owners) as phase:
        script=LIMITS+'''import os,time
child=os.fork()
if child==0:
 time.sleep(30)
else:
 print('CHILD='+str(child),flush=True)
 time.sleep(30)
'''
        result=phase.run([PYTHON,'-I','-c',script],timeout_seconds=.5,
            hard_limits=owner.RepositoryProcessLimits(1,64*MIB))
        assert result.timed_out and result.process_tree_terminated and result.workspace_cleaned,result
        assert phase.check_usage()['group_rss']['live_processes']==0
        assert phase.last_process_enforcement['timed_out']
        child=int(next(row.split('=',1)[1] for row in result.stdout.splitlines() if row.startswith('CHILD=')))
        observed=daemon._process(child)
        assert observed is None or observed['state']=='Z'
        retain(phase,result)


def test_missing_native_helper_refuses_instead_of_sampled_fallback(tmp_path,resource_owners,monkeypatch):
    def missing():raise codebase_process.ToolProcessError('required native prlimit unavailable')
    with admitted_phase(tmp_path,resource_owners) as phase:
        monkeypatch.setattr(codebase_process,'_linux_prlimit_path',missing)
        result=phase.run([PYTHON,'-I','-c','raise SystemExit("must not launch")'],timeout_seconds=5,
            hard_limits=owner.RepositoryProcessLimits(2,64*MIB))
        assert result.pid is None and result.returncode is None and 'prlimit unavailable' in result.error
        assert phase.last_process_enforcement['native_process_started'] is False
        retain(phase,result)
