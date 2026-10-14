"""Authored source context; real native owner, worker, broker and validation launchers."""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/opt/ipfs-supervisor');PRIVATE=ROOT/'state';CONTROL=ROOT/'fixture-ipc'
sys.path[:0]=[str(ROOT/name) for name in ('source','datasets','kit')]
def wire(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def write(path,x):Path(path).write_bytes(wire(x))
def main():
 started=time.monotonic();result=dict(schema='repeated-compact-edit-cross-uid-owner@1',complete=False,provider_calls=0,external_model_calls=0,official_verifier_executed=False,full_supervisor_start_qualified=False,source_obligations_discharged=0,provider_token_usage_measured=False)
 try:
  assert os.getuid()==os.geteuid()==1000
  assert not any(os.environ.get(n) for n in ('OPENAI_API_KEY','XAI_API_KEY','ANTHROPIC_API_KEY','HF_TOKEN'))
  (PRIVATE/'private-control').write_bytes(b'authored owner-only state\n');(PRIVATE/'private-control').chmod(0o600)
  repository=ROOT/'repository';workspace=ROOT/'worktrees/task'
  originals={'mod.py':b'def answer():\n    value = 1  # ordinary explanation\n    return value\n'+b'# ordinary fixture explanation retained privately by the owner\n'*40,'public.txt':b'Authored public input.\n'}
  for name,body in originals.items():(repository/name).write_bytes(body)
  def git(*args):return subprocess.check_output(['git','-C',str(repository),*args],stderr=subprocess.PIPE)
  git('init','-q');git('add','--all');git('-c','user.name=Qualification','-c','user.email=qualification@example.invalid','commit','-qm','authored input')
  git('worktree','add','--detach','-q',str(workspace))
  from ipfs_accelerate_py.agent_supervisor.context.context_compiler import ContextCompiler,build_text_context_references,render_context_capsule
  from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
  from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
  output=repository/'.runtime/semantic'
  prepare_semantic_context(repository=repository,paths=sorted(originals),program_paths=['mod.py'],required_raw_paths=['public.txt'],objective='Set the answer to four through three sequential mapped edits.',task_id='TASK-EDIT',output=output)
  artifact=output/'worker-context.json'
  refs=build_text_context_references(artifact.read_text(),reference_prefix='semantic-context',kind='semantic-context',path=artifact.relative_to(repository).as_posix(),repository_id='repo:test',tree_id='tree:test',required=True,chunk_bytes=1201)
  compiled=ContextCompiler(ContextBudget(max_input_tokens=32768,max_items=128,max_item_bytes=16384,max_serialized_bytes=262144)).compile(repository_id='repo:test',tree_id='tree:test',objective_id='TASK-EDIT',objective_revision='sha256:task',policy_id='policy:test',policy_revision='sha256:policy',caller='supervisor:test',stage='implementation',goal={'id':'TASK-EDIT'},authority={'mode':'candidate_only','completion_authority':False},scope={'allowed_paths':['mod.py']},acceptance={'criteria':['pending native validation']},evidence=refs)
  prompt=render_context_capsule(compiled.capsule).encode();(PRIVATE/'owner-prompt.txt').write_bytes(prompt)
  before=(workspace/'mod.py').stat()
  argv=[str(ROOT/'bin/owner-worker'),'--provider','codex_cli','--model','source-view-ipc-authored','--timeout','180','--max-output-tokens','128','--purpose','coding','--semantic-repository',str(repository),'--coding-reply-mode','ordinary-completion@1','--source-view-profile','python-comments@1','--source-edit-mode','mapped-span@2']
  write(PRIVATE/'owner-launch.json',dict(argv=argv,prompt_sha256=sha(prompt),prompt_bytes=len(prompt)))
  proc=subprocess.run(argv,cwd=workspace,input=prompt,capture_output=True,timeout=240)
  (PRIVATE/'owner-entry.stdout').write_bytes(proc.stdout);(PRIVATE/'owner-entry.stderr').write_bytes(proc.stderr)
  assert proc.returncode==0,'owner entry failed; private diagnostics retained'
  worker=json.loads((CONTROL/'worker-result.json').read_bytes());router=json.loads((CONTROL/'router-observation.json').read_bytes())
  expected=originals['mod.py'].replace(b'value = 1',b'value = 4',1);after=(workspace/'mod.py').stat()
  assert (workspace/'mod.py').read_bytes()==expected and worker['source_hash']==sha(expected)
  assert worker['confirmed_edit_count']==3 and len(worker['edit_steps'])==3
  assert all(step['result']['ok'] and step['result']['status']=='applied_validation_pending' for step in worker['edit_steps'])
  assert all(worker['edit_steps'][index]['before_generation']==worker['edit_steps'][index-1]['result']['workspace_generation'] for index in (1,2))
  assert (before.st_dev,before.st_ino)==(after.st_dev,after.st_ino) and after.st_uid==1000 and after.st_mode&0o777==0o660
  assert all((repository/name).read_bytes()==raw for name,raw in originals.items())
  assert (workspace/'public.txt').read_bytes()==originals['public.txt']
  assert not Path(worker['descriptor_path']).exists() and router['descriptor_environment_restored']
  validation=[str(ROOT/'bin/validation-worker'),str(ROOT/'venv/bin/python'),'-I','-B','-c',"import os,runpy; assert os.getuid()==1001; assert runpy.run_path('mod.py')['answer']()==4"]
  check=subprocess.run(validation,cwd=workspace,capture_output=True,timeout=60)
  write(PRIVATE/'validation.json',dict(argv=validation,returncode=check.returncode,stdout_bytes=len(check.stdout),stderr_bytes=len(check.stderr)))
  (PRIVATE/'validation.stdout').write_bytes(check.stdout);(PRIVATE/'validation.stderr').write_bytes(check.stderr)
  assert check.returncode==0,'candidate validation failed'
  remaining=[]
  for path in Path('/proc').glob('[0-9]*/status'):
   try:fields=dict(line.split(':',1) for line in path.read_text().splitlines() if ':' in line)
   except (FileNotFoundError,PermissionError,ProcessLookupError):continue
   if fields['Uid'].split()[0]=='1001' and not fields['State'].strip().startswith('Z'):remaining.append(int(path.parent.name))
  assert not remaining
  from benchmarks.agent_supervisor.container_coding.terminal_source_view_observation import collect
  observation=collect(parent=ROOT/'worker-observations',scope=dict(task_cid='authored:TASK-EDIT',task_alias='TASK-EDIT'),report=dict(worker_cleanup_returncode=0,source_view_profile='python-comments@1',source_edit_mode='mapped-span@2',provider_invocations=[dict(router['receipt'],phase='coding')]))
  assert observation['status']=='observed',observation['reason']
  assert observation['counts']['accepted_requests']==10
  assert observation['source_edit_counts']['delivered_mapping_pages']==3 and observation['source_edit_counts']['delivered_confirmation_pages']==3
  write('/results/session-observation.json',observation)
  result.update(complete=True,owner_uid=1000,worker_uid=1001,worker_no_new_privileges=worker['no_new_privileges'],worker_capabilities_zero=worker['capabilities_zero'],owner_private_read_denied=worker['private_read_denied'],retained_owner_inode=True,retained_owner_uid=after.st_uid,retained_source_mode=after.st_mode&0o777,source_before_sha256=sha(originals['mod.py']),candidate_sha256=sha(expected),canonical_unchanged=True,support_input_unchanged=True,descriptor_removed=True,remaining_worker_pids=remaining,validation_returncode=check.returncode,validation_worker_uid=1001,validation_scope='Authored answer()==4 smoke via production VALIDATION_ENTRY/WORKER_ENTRY --validate; not benchmark verifier or formal proof',confirmed_edit_count=3,sequential_generations_verified=True,source_session_accounting_verified=True,source_edit_counts=observation['source_edit_counts'],source_counts=observation['counts'],original_expansion_requests=0,edit_validation_pending=True,model_body_not_retained_publicly=True,authored_generate_hook_calls=1,cli_call_count=len(worker['cli_calls']),cli_stdout_bytes=sum(x['stdout_bytes'] for x in worker['cli_calls']),cli_stderr_bytes=sum(x['stderr_bytes'] for x in worker['cli_calls']),initial_generation_sha256=worker['initial_generation'],final_generation_sha256=worker['final_generation'],owner_writes_performed=False,task_completion_claimed=False)
 except BaseException as error:
  result.update(error_type=type(error).__name__,error=str(error)[:200]);(PRIVATE/'probe-failure.txt').write_text(traceback.format_exc())
 finally:
  result['seconds']=time.monotonic()-started;write('/results/probe-qualification.json',result);print(json.dumps({k:result.get(k) for k in ('complete','error_type','seconds')}))
 return 0 if result['complete'] else 1
if __name__=='__main__':raise SystemExit(main())
