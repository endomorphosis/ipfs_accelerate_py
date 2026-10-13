"""Public-profile output observer qualification; lifecycle records are authored fixtures."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,os,shutil,subprocess,sys
ROOT=Path('/opt/ipfs-supervisor');sys.path[:0]=[str(ROOT/n) for n in ('source','datasets','kit')]
STATE=ROOT/'state';FIXTURE=STATE/'observer-fixture.json'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(path,value):path.write_text(json.dumps(value,sort_keys=True)+'\n')
assert os.geteuid()==1000
from benchmarks.agent_supervisor.container_coding import terminal_output_observation as observer
if sys.argv[1]=='prepare':
 import duckdb
 conn=duckdb.connect(':memory:');platform=conn.execute('PRAGMA platform').fetchone()[0]
 extensions=Path.home()/'.duckdb/extensions'/('v'+duckdb.__version__)/platform;extensions.mkdir(parents=True,exist_ok=True)
 for name in ('ducklake','quack','httpfs'):
  src=Path('/extensions')/(name+'.duckdb_extension');target=extensions/src.name
  if target.exists():assert sha(target.read_bytes())==sha(src.read_bytes())
  else:shutil.copyfile(src,target)
  conn.execute('LOAD '+name)
 conn.close()
 repository=ROOT/'repository'
 for name in ('eigen.py','eval.py'):shutil.copyfile(Path('/permitted')/name,repository/name)
 for argv in (['init','-q'],['add','.'],['-c','user.name=Output Observation Fixture','-c','user.email=fixture@example.invalid','commit','-qm','permitted public input']):
  subprocess.run(['git','-C',str(repository),*argv],check=True,capture_output=True)
 from benchmarks.agent_supervisor.container_coding.terminal_native_preflight import qualify
 preflight=STATE/'preflight'
 result=qualify(repository=repository,instruction=Path('/permitted/instruction.md'),state=preflight,task_profile=json.loads(Path('/permitted/profile.json').read_text()),disable_intent_autoencoder=True)
 assert result['qualified'] and result['provider_calls']==0
 from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
 verified=local.verify_local_benchmark_admission(json.loads((preflight/'admission.json').read_text()),initial=True)
 task=verified['graph'].tasks[0];scope=observer.capture_scope(verified,task)
 assert [x['path'] for x in scope['outputs']]==['eigen.py']
 from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
 with IntentRepository(preflight/'intent.duckdb',install_schema=False) as intent:record=SimpleNamespace(**intent.get_task(task.task_cid))
 from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge as Bridge,_atomic_write
 native=STATE/'native';native.mkdir(mode=0o700);(native/'run').mkdir(mode=0o700)
 fixture_attempt=SimpleNamespace(attempt_id='authored:output-observer-attempt',claim_id='authored:output-observer-claim',lease_id='authored:output-observer-lease',task_cid=task.task_cid,task_alias=task.task_key,attempt_number=1,owner_session_id='authored:output-observer-owner',fencing_token=1,fence_epoch=1,body={})
 bridge=object.__new__(Bridge);bridge.attempt_root=native/'run/admitted_database_portal_attempts'
 paths=bridge._paths(fixture_attempt);identity=bridge._seal_attempt_directory(paths,attempt_id=fixture_attempt.attempt_id,create=True)
 seed=Bridge._render_projection_seed(fixture_attempt,record)
 binding=Bridge._binding(fixture_attempt,record,seed)
 _atomic_write(paths.binding,json.dumps(binding).encode(),sealed_directory_identity=identity)
 assert Bridge._strict_binding(paths.binding)==binding
 workspace=ROOT/'worktrees/owned';workspace.mkdir(mode=0o775)
 shutil.copyfile(repository/'eigen.py',workspace/'eigen.py');(workspace/'eigen.py').chmod(0o666)
 event=dict(task_id=task.task_key,task_cid=task.task_cid,canonical_task_cid=task.task_cid,worktree_path=str(workspace),baseline_ref=scope['baseline_commit'],attempt=1,returncode=0,provider_dispatched=True,attempt_consumed=True,lifecycle_finalize={'finalized':True})
 from ipfs_accelerate_py.agent_supervisor.runtime.event_log import append_jsonl_event
 prior=os.umask(0o022)
 try:append_jsonl_event(paths.events,'implementation_finished',event)
 finally:os.umask(prior)
 assert paths.events.stat().st_mode&0o777==0o644
 completion={'operation':'database_complete',**{key+'_sha256':sha(getattr(fixture_attempt,key).encode()) for key in ('attempt_id','claim_id','lease_id')},'fencing_token':1,'fence_epoch':1}
 terminal=dict(availability='observed',task_cid=task.task_cid,status='completed',revision=record.revision+1,completion_receipt=completion)
 report=dict(task_state={k:terminal[k] for k in ('task_cid','status','revision')},native_progress={'latest':{'task':terminal}},stop={'status':'succeeded'},remaining_processes=0,runtime_close={'succeeded':True},worker_cleanup_returncode=0,provider_invocations=[dict(schema='router-implementation-invocation@1',phase='coding',purpose='coding',invocation_id='authored:output-observer-invocation',workspace=str(workspace),public_instruction={'task_cid':task.task_cid})])
 write(FIXTURE,dict(native_state=str(native),workspace_root=str(ROOT/'worktrees'),repository=str(repository),scope=scope,report=report));FIXTURE.chmod(0o600)
 write(Path('/results/preparation.json'),dict(schema='public-output-observer-authored-preparation@1',public_profile='largest-eigenval',profile_sha256=sha(Path('/permitted/profile.json').read_bytes()),input_sha256={name:sha((Path('/permitted')/name).read_bytes()) for name in ('eigen.py','eval.py','instruction.md')},signed_scope_captured_before_edit=True,output_paths=['eigen.py'],event_mode='0644',attempt_directory_mode='0700',native_binding_writer_used=True,native_event_writer_used=True,authored_lifecycle_fixture=True,supervisor_start_executed=False,provider_calls=0,official_verifier_executed=False))
 print('prepared')
else:
 fixture=json.loads(FIXTURE.read_text());result=observer.collect(**fixture,timeout_seconds=5)
 assert result['status']=='observed',result
 row,=result['outputs'];baseline=Path('/permitted/eigen.py').read_bytes();candidate=baseline+b'\n# Authored output-observer fixture; no benchmark solution.\n'
 assert row['candidate']['sha256']==sha(candidate) and row['canonical']['sha256']==sha(baseline)
 assert row['candidate']['equals_admitted_baseline'] is False and row['canonical']['equals_admitted_baseline'] is True
 assert all(result[key] is False for key in ('patch_application_established','writer_identity_established','task_correctness_established','completion_authority','raw_source_exported'))
 result.update(authored_lifecycle_fixture=True,supervisor_start_executed=False,official_verifier_executed=False)
 write(Path('/results/observation.json'),result)
 print(json.dumps({'status':result['status'],'outputs':len(result['outputs'])}))
