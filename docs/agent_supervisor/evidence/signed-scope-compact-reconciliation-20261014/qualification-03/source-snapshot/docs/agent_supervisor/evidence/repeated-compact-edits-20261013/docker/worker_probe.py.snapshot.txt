"""Authored provider hook exercising actual cross-UID source/edit CLI transport."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/opt/ipfs-supervisor');CONTROL=ROOT/'fixture-ipc'
CLIENT=ROOT/'source/ipfs_accelerate_py/agent_supervisor/runtime/worker_source_client.py'
ENV='IPFS_SUPERVISOR_SOURCE_VIEW_DESCRIPTOR'
def wire(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def install():
 assert os.getuid()==os.geteuid()==1001
 from ipfs_accelerate_py import llm_router
 from ipfs_accelerate_py.llm_allocation import intelligence_index
 from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as router
 intelligence_index.discover_available_providers=lambda:['codex_cli']
 intelligence_index.select_efficient_route=lambda **kw:SimpleNamespace(provider='codex_cli',model_name='source-view-ipc-authored',catalog_revision='authored-fixture')
 llm_router.get_llm_provider=lambda *a,**kw:object()
 calls=[]
 def generate(prompt,**kwargs):
  assert not calls and kwargs['allow_local_fallback'] is False and kwargs['allow_cross_provider_fallback'] is False
  calls.append(dict(prompt_bytes=len(prompt.encode()),prompt_sha256=sha(prompt.encode())))
  descriptor=os.environ[ENV];digest=os.environ[ENV+'_SHA256'];desc=json.loads(Path(descriptor).read_bytes())
  assert desc['source_edit_mode']=='mapped-span@2'
  status=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
  assert status['NoNewPrivs'].strip()=='1' and all(int(status[x].strip(),16)==0 for x in ('CapEff','CapPrm','CapAmb'))
  try:(ROOT/'state/private-control').read_bytes()
  except PermissionError:private_denied=True
  else:raise AssertionError('owner state readable by worker')
  logs=[]
  def invoke(client,args,body=None):
   proc=subprocess.run([sys.executable,'-I','-B',str(client),*args],input=body,capture_output=True,timeout=45)
   logs.append(dict(returncode=proc.returncode,stdout_bytes=len(proc.stdout),stderr_bytes=len(proc.stderr),stdout_sha256=sha(proc.stdout),stderr_sha256=sha(proc.stderr)))
   assert proc.returncode==0 and not proc.stderr, (proc.returncode,proc.stderr.decode()[:200])
   return json.loads(proc.stdout)
  current=invoke(CLIENT,['--path','mod.py','--view','compact','--working','--edit-ready'])
  assert current['ok'] and current['exhausted'] and '# ordinary' not in current['content']
  initial_generation=current['workspace_generation']
  steps=[]
  for old_value,new_value in [('1','2'),('2','30'),('30','4')]:
   position=current['content'].encode().index(('value = '+old_value).encode())+len(b'value = ')
   edit=dict(path='mod.py',expected_workspace_generation=current['workspace_generation'],expected_source_sha256=current['source_sha256'],expected_view_sha256=current['view_sha256'],expected_manifest_sha256=current['manifest_sha256'],view_start=position,view_end=position+len(old_value),expected_view_preimage=old_value,replacement=new_value)
   before=Path('mod.py').stat()
   result=invoke(CLIENT.with_name('worker_source_edit_client.py'),['--edit-file','-'],wire(edit))
   after=Path('mod.py').stat()
   assert result['ok'] and result['status']=='applied_validation_pending' and result['owner_postimage_verified'] and result['validation_pending']
   assert not result['automatic_retry_performed'] and not result['atomic_application_claimed'] and not result['completion_authority']
   assert (before.st_dev,before.st_ino)==(after.st_dev,after.st_ino) and before.st_uid==after.st_uid==1000
   assert result['write_mode']=='preserved_owner_inode'
   refreshed=invoke(CLIENT,['--path','mod.py','--view','compact','--working','--edit-ready','--expected-workspace-generation',result['workspace_generation'],'--expected-source-sha256',result['source_sha256'],'--expected-view-sha256',result['view_sha256']])
   assert refreshed['workspace_generation']==result['workspace_generation']!=current['workspace_generation']
   assert 'value = '+new_value in refreshed['content']
   assert refreshed['manifest_sha256']==result['manifest_sha256']!=current['manifest_sha256']
   steps.append(dict(ordinal=len(steps)+1,before_generation=current['workspace_generation'],before_source_sha256=current['source_sha256'],before_view_sha256=current['view_sha256'],before_manifest_sha256=current['manifest_sha256'],result=result,refreshed_source_sha256=refreshed['source_sha256'],refreshed_view_sha256=refreshed['view_sha256'],refreshed_manifest_sha256=refreshed['manifest_sha256']))
   current=refreshed
  assert len({step['result']['proposal_sha256'] for step in steps})==3
  assert len({initial_generation,*[step['result']['workspace_generation'] for step in steps]})==4
  outcome=dict(schema='repeated-compact-edit-cross-uid-worker@1',uid=os.getuid(),gid=os.getgid(),task_alias=desc['task_id'],descriptor_path=descriptor,descriptor_sha256=digest,source_hash=sha(Path('mod.py').read_bytes()),initial_generation=initial_generation,final_generation=current['workspace_generation'],result=result,edit_steps=steps,confirmed_edit_count=len(steps),cli_calls=logs,private_read_denied=private_denied,no_new_privileges=True,capabilities_zero=True,retained_owner_inode=True,owner_uid=after.st_uid,mode=after.st_mode&0o777,original_source_expansion_requests=0,external_model_calls=0)
  (CONTROL/'worker-result.json').write_bytes(wire(outcome))
  from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
  schema=wire(kwargs['codex_output_schema'])
  set_last_cli_observation('codex_cli',dict(exit_code=0,authored_offline_fixture=True,codex_output_schema_sha256=sha(schema),codex_output_schema_bytes=len(schema)))
  return '{"schema":"supervisor-coding-completion@1","status":"candidate_ready"}'
 llm_router.generate_text=generate
 actual=router.run
 def capture(*args,**kwargs):
  output,receipt=actual(*args,**kwargs)
  assert len(calls)==1 and ENV not in os.environ and ENV+'_SHA256' not in os.environ
  retained={k:v for k,v in receipt.items() if k in ('schema','source_view_retrieval','semantic_translation','completion_authority','usage','provider','model','model_prompt_bytes','model_prompt_sha256','invocation_id','purpose','workspace','external_container_boundary','status','coding_reply_contract')}
  (CONTROL/'router-observation.json').write_bytes(wire(dict(receipt=retained,authored_generate_calls=calls,descriptor_environment_restored=True)))
  return output,receipt
 router.run=capture
