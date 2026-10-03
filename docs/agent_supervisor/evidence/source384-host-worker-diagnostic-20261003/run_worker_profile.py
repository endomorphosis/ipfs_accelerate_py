from pathlib import Path
import hashlib,json,os,site,sys,time
B=Path(__file__).resolve().parent;W=B.parent.parent;D=W/'.worktrees/ir-release-datasets-20261002'
sys.path.insert(0,str(D))
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceSchedulerConfig,GlobalResourceScheduler,ResourceLane
from ipfs_datasets_py.logic.backends.codebase_process import BoundedToolRunner,ToolRunLimits,run_bounded_stdin_tool
from ipfs_datasets_py.logic.software_contracts import codebase_source_units_384 as native
from ipfs_datasets_py.logic.formalization.autoencoder.source_program_runtime_384_v2 import _read_view
from importlib.metadata import version
state=Path('/tmp/ipfs-datasets-resource-scheduler-1000.json')
config=json.loads(state.read_text())['config'];scheduler=GlobalResourceScheduler(ResourceSchedulerConfig(state_path=state,**config))
read=lambda p:json.loads(p.read_text())
reference=read(W/'artifacts/source384-header-docker-20261003/qualification-01/native-inference.json')['report']
checkpoint=W/'artifacts/distributed384-20261001/run-01/security_ir/coordinator/checkpoints/2ca38dfcc05536315fc3e2c0647b710b930ef4066b474061a7b4e5bfb9a258c5.json'
snapshot='/home/barberb/.cache/huggingface/hub/models--thenlper--gte-small/snapshots/17e1f347d17fe144873b1201da91788898c639cd'
roots=[str(D),*site.getsitepackages(),site.getusersitepackages()]
argv=[sys.executable,'-I','-B',str(B/'profile_worker.py'),json.dumps(roots)]
environment=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
command=dict(argv=argv,environment=environment,scope='host worker-only diagnostic under existing shared scheduler; no codebase observation or Docker qualification',scheduler_config=config,checkpoint_sha256=checkpoint.stem,helper_sha256=hashlib.sha256((B/'profile_worker.py').read_bytes()).hexdigest(),source_rows=128,memory_mb=4096,timeout_seconds=90)
(B/'worker-profile-command.json').write_text(json.dumps(command,indent=2)+'\n')
start=time.monotonic()
with scheduler.acquire(lane=ResourceLane.ORCHESTRATION,cpu_slots=1,memory_mb=4096,child_process_slots=1,timeout=15,request_id='source384-worker-profile') as lease:
 _,_,view,_=_read_view(checkpoint,checkpoint.stem)
 key=reference['key'].copy();key.update(python=sys.version,runtime_versions={n:version(n) for n in ('torch','numpy','transformers','sentence-transformers','tokenizers')},embedding_snapshot=snapshot,producer=native.pins(),runtime_checkpoint_sha256=native.sha(native.raw(view)))
 payload=dict(schema=native.WORKER_SCHEMA,key=key,checkpoint=view,rows=reference['preparation']['selected_inputs'])
 result=run_bounded_stdin_tool(argv,native.raw(payload),runner=BoundedToolRunner(base_environment=environment),limits=ToolRunLimits(timeout_seconds=90,resident_memory_bytes=4096*1024**2,max_input_bytes=native.MAX_BYTES,max_output_bytes=native.MAX_BYTES,max_workspace_bytes=2*native.MAX_BYTES),cancellation=lease.combined_cancellation_signal(None))
 (B/'worker-profile.stdout').write_text(result.stdout);(B/'worker-profile.stderr').write_text(result.stderr)
 receipt=dict(returncode=result.returncode,seconds=time.monotonic()-start,elapsed_ms=result.elapsed_ms,timed_out=result.timed_out,cancelled=result.cancelled,resource_exhausted=result.resource_exhausted,workspace_cleaned=result.workspace_cleaned,termination_reason=result.termination_reason,source_proof_claim=False)
 (B/'worker-profile-exit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));assert result.returncode==0 and not result.timed_out
