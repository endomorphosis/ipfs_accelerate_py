"""Provider-free reproduction of audited worker budget admission in the task image."""
import asyncio,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
A=Path('/home/barberb/lift_coding/artifacts/grok-recovery-20261006')
P=Path('/home/barberb/lift_coding/.worktrees/grok-worker-budget-baseline-20261006')
HEAD='585cf2adccdbaff8a838a5f5438ca80ea75c8b0e'
OUT=A/'grok-container/offline-worker-budget/baseline03'

def emit(path,value):
 with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')

async def main():
 from harbor.environments.docker.docker import DockerEnvironment
 from harbor.models.task.task import Task
 from harbor.models.trial.paths import TrialPaths
 from benchmarks.agent_supervisor.container_coding.terminal_deployment import deploy_supervisor,runtime_environment,PYTHON
 from benchmarks.agent_supervisor.container_coding.terminal_task_bootstrap import bootstrap_task_repository
 from benchmarks.agent_supervisor.container_coding.container_worker_deployment import deploy_worker_boundary,WORKER_ENTRY
 from benchmarks.agent_supervisor.container_coding.terminal_source384_qualification import resource_options
 from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import validate_runner_error_envelope
 assert subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip()==HEAD
 assert not subprocess.check_output(['git','-C',str(P),'status','--porcelain'],text=True)
 review=json.loads((A/'grok-container/archive-review-08.json').read_text());assert review['qualified']
 OUT.mkdir(exist_ok=False)
 auth=OUT/'synthetic-auth.json';auth.write_text('{}\n');auth.chmod(0o600)
 task=Task(Path('/home/barberb/lift_coding/.benchmarks/terminal-bench-2/tune-mjcf'))
 paths=TrialPaths(OUT/'harbor');paths.mkdir()
 env=DockerEnvironment(environment_dir=task.paths.environment_dir,environment_name=task.short_name,
   session_id='ipfs-worker-budget-offline-20261006-03',trial_paths=paths,task_env_config=task.config.environment,
   keep_containers=True,**resource_options('source384-5cpu-20gib-coding600@1'))
 report={'schema':'grok-worker-budget-offline-probe@1','source_head':HEAD,'archive_sha256':review['archive_sha256_actual'],
   'provider_calls':0,'real_credentials_used':False,'official_verifier_executed':False,'benchmark_result':False,'qualified':False,'cleanup_pending':True,'cases':[]}
 started=time.monotonic();cid=None
 try:
  await env.start(force_build=True)
  cid=await env._platform._resolve_service_container('main')
  cid=subprocess.check_output(['docker','inspect','--format','{{.Id}}',cid],text=True).strip()
  emit(OUT/'owned-container.json',{'container_id':cid,'source_head':HEAD,'created_for':'provider_free_worker_budget_probe'})
  print(json.dumps({'stage':'container_created','container_id':cid}),flush=True)
  profile=json.loads((A/'grok-tune-mjcf-08-profile.json').read_text())
  await bootstrap_task_repository(env,profile=profile,output=OUT/'task-bootstrap')
  await deploy_supervisor(env,archive_dir=A/'grok-bundle-08',output=OUT/'deployment',auth_json=auth,
    install_codex=False,provider='grok_cli',isolated_uv_bootstrap=True)
  boundary=await deploy_worker_boundary(env,output=OUT/'worker-boundary',provider='grok_cli')
  assert boundary['qualified']
  networks=json.loads(subprocess.check_output(['docker','inspect','--format','{{json .NetworkSettings.Networks}}',cid],text=True))
  for network in networks:subprocess.run(['docker','network','disconnect',network,cid],check=True,capture_output=True,timeout=15)
  assert json.loads(subprocess.check_output(['docker','inspect','--format','{{json .NetworkSettings.Networks}}',cid],text=True))=={}
  report['network_disconnected_before_cases']=True
  remote=r'''import hashlib,json,pathlib,subprocess
root=pathlib.Path('/opt/ipfs-supervisor');candidate=root/'worktrees/offline-budget'
subprocess.run(['git','-C','/app','worktree','add','--detach',str(candidate),'HEAD'],check=True,capture_output=True)
rows=[]
for timeout,model in ((600,'grok-4.7'),(300,'invalid-model')):
 result=subprocess.run([str(root/'bin/router-worker'),'--provider','grok_cli','--model',model,'--purpose','coding','--timeout',str(timeout),'--max-output-tokens','128'],cwd=candidate,input='Authored offline preflight only',capture_output=True,text=True,timeout=20)
 from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import validate_runner_error_envelope
 diagnostic=None
 for line in result.stderr.splitlines():
  try:value=validate_runner_error_envelope(json.loads(line))
  except (ValueError,TypeError,RecursionError):continue
  if diagnostic is not None:raise ValueError('ambiguous authored diagnostic')
  diagnostic=value
 rows.append({'timeout_seconds':timeout,'exit_code':result.returncode,'bounded_timeout_rejection':'bounded timeout required' in result.stderr.splitlines(),'router_diagnostic':diagnostic,'stdout_empty':result.stdout==''})
print(json.dumps({'cases':rows,'worker_entry_sha256':hashlib.sha256((root/'bin/worker-entry').read_bytes()).hexdigest()}))
'''
  result=await env.exec(command=PYTHON+' -P -c '+shlex.quote(remote),cwd='/app',user='supervisor',env=runtime_environment(),timeout_sec=60)
  if result.return_code:raise RuntimeError('offline worker probe execution failed')
  observed=json.loads(result.stdout)
  report.update(observed)
  emit(OUT/'case-observations.json',observed)
  assert observed['worker_entry_sha256']==hashlib.sha256(WORKER_ENTRY.encode()).hexdigest()
  first,second=observed['cases'];assert first['timeout_seconds']==600 and first['exit_code']==1 and first['bounded_timeout_rejection'] and first['router_diagnostic'] is None
  assert second['timeout_seconds']==300 and second['exit_code']==1 and not second['bounded_timeout_rejection']
  second['router_diagnostic']=validate_runner_error_envelope(second['router_diagnostic'])
  assert second['router_diagnostic']['diagnostic']['phase']=='argument_validation'
  report.update(observed,qualified=True,boundary_qualified=True)
  print(json.dumps({'stage':'baseline_qualified','qualified':True,'provider_calls':0,'cases':observed['cases']}),flush=True)
 except BaseException as error:
  report['error_type']=type(error).__name__ if type(error).__name__ in {'ValueError','RuntimeError','AssertionError','TimeoutExpired','TimeoutError'} else 'other'
  raise
 finally:
  auth.unlink(missing_ok=True)
  if not report['qualified']:
   try:
    await env.stop(delete=True)
    report['cleanup_pending']=False
   except Exception:
    report['cleanup_pending']=True
  report['seconds']=time.monotonic()-started
  emit(OUT/'receipt.json',report)

if __name__=='__main__':asyncio.run(main())
