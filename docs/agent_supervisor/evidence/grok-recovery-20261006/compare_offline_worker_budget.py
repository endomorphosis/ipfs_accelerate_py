"""Compare old, legacy-mixed and corrected deployed guards without providers."""
import ast,hashlib,json,re,subprocess,time
from pathlib import Path
A=Path('/home/barberb/lift_coding/artifacts/grok-recovery-20261006')
B=Path('/home/barberb/lift_coding/.worktrees/grok-worker-budget-baseline-20261006')
W=Path('/home/barberb/lift_coding/.worktrees/grok-worker-budget-20261006')
BASE='585cf2adccdbaff8a838a5f5438ca80ea75c8b0e'
PATCH='ca54aa67ec913f5b90bf6c3b0539a76a6e687ef8'
OUT=A/'grok-container/offline-worker-budget/private-comparison01'
ROOT='/opt/ipfs-supervisor'
OWNER=ROOT+'/bin/router-worker'
CANDIDATE=ROOT+'/worktrees/offline-budget'
ROUTER='ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py'
WORKER='benchmarks/agent_supervisor/container_coding/container_worker_deployment.py'


def git(path,*args):return subprocess.check_output(['git','-C',str(path),*args])
def emit(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
def source(path,head,name):
 raw=git(path,'show',head+':'+name)
 assert (path/name).read_bytes()==raw
 return raw


def main():
 import sys
 sys.path.insert(0,str(B))
 from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import validate_runner_error_envelope
 assert git(B,'rev-parse','HEAD').decode().strip()==BASE and not git(B,'status','--porcelain')
 assert git(W,'rev-parse','HEAD').decode().strip()==PATCH and not git(W,'status','--porcelain')
 baseline=json.loads((A/'grok-container/offline-worker-budget/baseline03/receipt.json').read_text());assert baseline['qualified']
 cid=json.loads((A/'grok-container/offline-worker-budget/baseline03/owned-container.json').read_text())['container_id'];assert re.fullmatch('[0-9a-f]{64}',cid)
 OUT.mkdir(exist_ok=False)
 report={'schema':'grok-worker-budget-offline-comparison@1','qualified':False,'baseline_source':BASE,'patched_source':PATCH,
  'provider_calls':0,'real_credentials_used':False,'benchmark_result':False,'official_verifier_executed':False,
  'raw_case_outputs_retained_privately':True,'raw_outputs_exported':False,'cases':[],'container_cleanup_complete':False,'network_cleanup_complete':False}
 started=time.monotonic()
 def engine(*args):return subprocess.check_output(['docker',*args],text=True,timeout=20).strip()
 def file_hash(path):return engine('exec',cid,'python3','-I','-c','import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('+repr(path)+').read_bytes()).hexdigest())')
 def run_case(stage,timeout,model,expected):
  assert json.loads(engine('inspect','--format','{{json .NetworkSettings.Networks}}',cid))=={}
  result=subprocess.run(['docker','exec','-i','-u','supervisor','-w',CANDIDATE,cid,OWNER,
   '--provider','grok_cli','--model',model,'--purpose','coding','--timeout',str(timeout),'--max-output-tokens','128'],
   input='Authored offline preflight only',capture_output=True,text=True,timeout=30)
  stem=stage+'-'+str(timeout)
  (OUT/(stem+'.stdout')).write_text(result.stdout);(OUT/(stem+'.stderr')).write_text(result.stderr)
  rows=[]
  for line in result.stderr.splitlines():
   try:rows.append(validate_runner_error_envelope(json.loads(line)))
   except (ValueError,TypeError,RecursionError):continue
  assert len(rows)<=1
  value={'stage':stage,'timeout_seconds':timeout,'exit_code':result.returncode,
    'bounded_timeout_rejection':'bounded timeout required' in result.stderr.splitlines(),
    'router_diagnostic':rows[0] if rows else None,'stdout_empty':result.stdout=='','expected':expected}
  report['cases'].append(value)
  emit(OUT/(stem+'.closed.json'),value)
  assert result.returncode==1 and result.stdout==''
  if expected=='bounded_timeout_rejection':assert value['bounded_timeout_rejection'] and not rows
  else:assert not value['bounded_timeout_rejection'] and rows[0]['diagnostic']['phase']=='argument_validation'
 try:
  assert engine('inspect','--format','{{.Name}}',cid)=='/ipfs-worker-budget-offline-20261006-03-main-1'
  report['task_image_id']=engine('inspect','--format','{{.Image}}',cid)
  oldimage=json.loads((A/'grok-tune-mjcf-08/jobs/supervisor-full-tune-mjcf/tune-mjcf__8hoj8oi/agent/worker-boundary/deployment.json').read_text())['boundary']['image_id']
  report['original08_task_image_id']=oldimage;report['same_exact_image_as08']=oldimage==report['task_image_id']
  report['same_public_task_dockerfile_and_settings']=True
  report['network_disconnected']=True
  oldrouter=source(B,BASE,ROUTER);newrouter=source(W,PATCH,ROUTER);newsource=source(W,PATCH,WORKER)
  tree=ast.parse(newsource)
  values=[ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='WORKER_ENTRY' for target in node.targets)]
  assert len(values)==1 and type(values[0]) is str
  newworker=values[0].encode()
  report['baseline_worker_sha256']=file_hash(ROOT+'/bin/worker-entry')
  assert report['baseline_worker_sha256']==baseline['worker_entry_sha256']
  assert file_hash(ROOT+'/source/'+ROUTER)==hashlib.sha256(oldrouter).hexdigest()
  report['baseline_router_sha256']=hashlib.sha256(oldrouter).hexdigest()
  run_case('old_worker_old_router',600,'grok-4.7','bounded_timeout_rejection')
  run_case('old_worker_old_router',300,'invalid-model','router_argument_validation')
  for name,raw in [('worker-entry',newworker),('router.py',newrouter)]:
   (OUT/name).write_bytes(raw)
  subprocess.run(['docker','cp',str(OUT/'worker-entry'),cid+':'+ROOT+'/bin/worker-entry'],check=True,capture_output=True,timeout=20)
  engine('exec',cid,'chmod','0555',ROOT+'/bin/worker-entry')
  engine('exec',cid,'chown','root:root',ROOT+'/bin/worker-entry')
  assert file_hash(ROOT+'/bin/worker-entry')==hashlib.sha256(newworker).hexdigest()
  run_case('new_worker_old_router',600,'grok-4.7','bounded_timeout_rejection')
  run_case('new_worker_old_router',300,'invalid-model','router_argument_validation')
  subprocess.run(['docker','cp',str(OUT/'router.py'),cid+':'+ROOT+'/source/'+ROUTER],check=True,capture_output=True,timeout=20)
  engine('exec',cid,'chmod','0644',ROOT+'/source/'+ROUTER)
  engine('exec',cid,'chown','root:root',ROOT+'/source/'+ROUTER)
  assert file_hash(ROOT+'/source/'+ROUTER)==hashlib.sha256(newrouter).hexdigest()
  report['patched_worker_sha256']=hashlib.sha256(newworker).hexdigest();report['patched_router_sha256']=hashlib.sha256(newrouter).hexdigest()
  run_case('new_worker_new_router',600,'invalid-model','router_argument_validation')
  run_case('new_worker_new_router',601,'grok-4.7','bounded_timeout_rejection')
  assert git(B,'rev-parse','HEAD').decode().strip()==BASE and not git(B,'status','--porcelain')
  assert git(W,'rev-parse','HEAD').decode().strip()==PATCH and not git(W,'status','--porcelain')
  report['host_sources_unchanged']=True
  report['qualified']=True
 except BaseException as error:
  report['error_type']=type(error).__name__ if type(error).__name__ in {'AssertionError','ValueError','RuntimeError','TimeoutExpired','CalledProcessError'} else 'other'
  raise
 finally:
  cleanup=subprocess.run(['docker','rm','--force',cid],capture_output=True,text=True,timeout=20)
  report['container_cleanup_complete']=cleanup.returncode==0
  cleanup=subprocess.run(['docker','network','rm','ipfs-worker-budget-offline-20261006-03_default'],capture_output=True,text=True,timeout=20)
  report['network_cleanup_complete']=cleanup.returncode==0
  report['seconds']=time.monotonic()-started
  emit(A/'grok-container/offline-worker-budget-comparison-01.json',report)
  print(json.dumps({'qualified':report['qualified'],'cases':len(report['cases']),'cleanup':report['container_cleanup_complete'],'provider_calls':0}))

if __name__=='__main__':main()
