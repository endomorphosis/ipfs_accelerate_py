
from contextlib import redirect_stdout
import hashlib,json,pathlib,signal,sys,time
started=time.monotonic(); phase='runtime_load';phase_started=started
result=dict(schema='terminal-source384-original-container-qualification@1',qualified=False,
 provider_calls=0,official_verifier_executed=False,benchmark_result=False,
 source_qualified_proof_claimed=False,training_steps=0,download_calls=0)
def expired(*args): raise TimeoutError('bounded Source384 container qualification expired')
signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,270)
try:
 with redirect_stdout(sys.stderr):
  from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
  from ipfs_accelerate_py.agent_supervisor.runtime.source384_repository_context import validate_source384_context, _pins, _read, MAX_INFERENCE_BYTES
  root=pathlib.Path('/app'); state=pathlib.Path('/opt/ipfs-supervisor/state/source384-qualification')
  config_path=pathlib.Path('/opt/ipfs-supervisor/models/source384/config.json')
  result['producer']=_pins()
  # Diagnostic wrappers only; all native calls, leases and deadlines execute.
  from ipfs_datasets_py.logic.software_contracts import codebase_ir as diagnostic_index
  from ipfs_datasets_py.logic.software_contracts import codebase_source_units_384 as diagnostic_units
  from ipfs_datasets_py.logic.software_contracts import codebase_source_384 as diagnostic_parent
  result['diagnostic_only']=True
  result['diagnostic_stage_events']=[]
  def diagnostic_timed(target,name):
   original=getattr(target,name)
   def invoke(*args,**kwargs):
    began=time.monotonic();status='returned'
    try:return original(*args,**kwargs)
    except BaseException:
     status='raised';raise
    finally:
     event=dict(stage=target.__name__+'.'+name,start_seconds=began-started,seconds=time.monotonic()-began,status=status)
     if name=='_worker':event['requested_worker_timeout_seconds']=kwargs['timeout']
     if len(result['diagnostic_stage_events'])<128:result['diagnostic_stage_events'].append(event)
     else:result['diagnostic_stage_overflow']=True
   setattr(target,name,invoke)
  for target,name in [(diagnostic_index.RepositoryCodebaseIndex,'prepare_current'),(diagnostic_index.RepositoryCodebaseIndex,'observe_current'),(diagnostic_units,'_context'),(diagnostic_units,'_worker'),(diagnostic_units,'load_source_unit_inference'),(diagnostic_parent,'register_shared_parent')]:diagnostic_timed(target,name)
  phase='prepare';phase_started=before=time.monotonic()
  prepared=prep.prepare(repository=root,instruction=pathlib.Path('/opt/ipfs-supervisor/source384-public-instruction.md'),state=state,
   **({"intent_requirement_contract":pathlib.Path(sys.argv[1])} if len(sys.argv)>1 else {}))
  result['prepare_seconds']=time.monotonic()-before
  phase='initial_context';phase_started=before=time.monotonic()
  context=prep.initial_context(state=state,source384_config=config_path,train_autoencoder=False)
  result['initial_context_seconds']=time.monotonic()-before
  receipt=context['source384_context']
  if len(sys.argv)>1:
   from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
   result['intent_requirement_contract_cid']=cid_for_dag_json(prepared['intent_requirement_contract'])
  if receipt.get('schema')=='terminal-source384-repository-context@2':
   result['header_consumer_sha256']=receipt['header_consumer_sha256']
  inference_path=pathlib.Path(receipt['output'])/'inference.json'
  raw=_read(inference_path,MAX_INFERENCE_BYTES);inference=json.loads(raw)
  if hashlib.sha256(raw).hexdigest()!=receipt['inference_sha256']:raise ValueError('native inference digest differs')
  if inference['native_worker_executed'] is not True or inference['inference_executed'] is not True:
   raise ValueError('qualification requires actual pinned parent inference')
  if inference['report']['output']['model_loads']!=1:raise ValueError('one actual GTE model load required')
  if any(receipt[k] is not False for k in ('proof_authority','execution_authority','completion_authority','formalization_authority')):
   raise ValueError('Source384 preparation cannot grant authority')
  if receipt['config_path']!=str(config_path):raise ValueError('relocated model config was not consumed')
  # Diagnostic intervention on this process's unreachable/unused allocations.
  # Context bytes stay live and equal; no cache is cleared and no limit changes.
  import gc,ctypes,os
  from dataclasses import asdict
  from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import collect_proof_host_resources
  def diagnostic_memory():
   fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in pathlib.Path('/proc/self/smaps_rollup').read_text().splitlines() if ':' in line}
   return dict(smaps={k:fields.get(k) for k in ('Rss','Anonymous','Private_Clean','Private_Dirty')},resources=asdict(collect_proof_host_resources()))
  context_wire=lambda:json.dumps(context,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
  digest_before=hashlib.sha256(context_wire()).hexdigest()
  release=dict(scope='this_process_only',limits_changed=False,source_checks_skipped=False,context_sha256_before=digest_before,before=diagnostic_memory())
  begin_release=time.monotonic()
  release['collected_objects']=gc.collect()
  release['after_gc']=diagnostic_memory()
  release['libc']=os.confstr('CS_GNU_LIBC_VERSION')
  trim=ctypes.CDLL(None).malloc_trim;trim.argtypes=[ctypes.c_size_t];trim.restype=ctypes.c_int
  release['malloc_trim_returncode']=trim(0)
  release['after_trim']=diagnostic_memory()
  release['seconds']=time.monotonic()-begin_release
  release['context_sha256_after']=hashlib.sha256(context_wire()).hexdigest()
  if release['context_sha256_after']!=digest_before:raise ValueError('diagnostic reclamation changed retained context')
  result['diagnostic_reclamation']=release
  phase='warm_observation';phase_started=before=time.monotonic()
  validate_source384_context(repository=root,expected_receipt=receipt)
  result['warm_observation_seconds']=time.monotonic()-before
  result.update(qualified=True,checkpoint_sha256=receipt['checkpoint_sha256'],
   config_sha256=receipt['config_sha256'],config_path=receipt['config_path'],
   inference_sha256=receipt['inference_sha256'],source_head=receipt['source_head'],
   signed_source_hashes=receipt['source_hashes'],coverage=receipt['summary']['coverage'],
   source384_summary=receipt['summary'],source384_resource_profile=receipt['resource_profile'],
   native_worker_receipt=inference['report']['worker_receipt'],
   native_inference_key=inference['report']['key'],native_worker_executed=True,
   inference_executed=True,neural_inference_replayed=False,
   source384_seconds=receipt['seconds'],context_seconds=context['seconds'],
   context_nonoverlapping_seconds=context['nonoverlapping_seconds'])
except BaseException as exc:
 import traceback
 traceback.print_exc(file=sys.stderr,limit=20)
 result.update(error_type=type(exc).__name__,error=str(exc)[:2048],error_phase=phase,phase_seconds=time.monotonic()-phase_started)
 try:
  from dataclasses import asdict
  from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import collect_proof_host_resources
  result['failure_resources']=asdict(collect_proof_host_resources())
 except Exception as diagnostic_error: result['failure_resource_error']=type(diagnostic_error).__name__
 try:
  from benchmarks.agent_supervisor.container_coding.terminal_resource_diagnostics import collect_failure_scheduler
  result['failure_scheduler']=collect_failure_scheduler()
 except Exception: result['failure_scheduler_error']='collection_unavailable'
finally:
 signal.setitimer(signal.ITIMER_REAL,0)
 result['seconds']=time.monotonic()-started
raw=json.dumps(result,sort_keys=True,allow_nan=False).encode()
if len(raw)>1024*1024: raise ValueError('bounded qualification result required')
with pathlib.Path('/opt/ipfs-supervisor/state/source384-qualification-result.json').open('xb') as stream:stream.write(raw)
print(raw.decode())
raise SystemExit(0 if result['qualified'] else 1)
