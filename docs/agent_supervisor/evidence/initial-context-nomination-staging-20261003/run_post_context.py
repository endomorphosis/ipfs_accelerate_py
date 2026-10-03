"""Bounded authored native lifecycle/prompt timing; no numerical inference."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,threading,time
B=Path(__file__).resolve().parent;W=B.parents[1];A=W/'.worktrees/ir-release-accelerate-20261002';D=W/'.worktrees/ir-release-datasets-20261002'
if '--child' not in sys.argv:
 env=dict(os.environ,PYTHONPATH=f'{A}:{D}',PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=str(B/'post-context-seal.duckdb'))
 cmd=[sys.executable,'-B',str(Path(__file__)),'--child']
 (B/'post-context-command.json').write_text(json.dumps(dict(argv=cmd,cwd=str(A),environment={k:env[k] for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','PYTEST_DISABLE_PLUGIN_AUTOLOAD','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB')}),indent=2)+'\n')
 started=time.monotonic()
 with (B/'post-context.log').open('wb') as out:r=subprocess.run(cmd,cwd=A,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=180)
 receipt=dict(exit_code=r.returncode,seconds=time.monotonic()-started)
 (B/'post-context-exit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));raise SystemExit(r.returncode)
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import TodoImplementationDaemon
import pytest
cases=['test/integration/test_admitted_context_refresh.py::test_source384_publication_reports_historical_selection_without_downgrade_or_dispatch',
 'benchmarks/agent_supervisor/container_coding/test_terminal_context_rebind.py::test_rebind_actual_owner_and_native_prompt_checks_fresh_sources_without_world_authority[selected]']
owners={AdmittedBenchmarkRuntime:('_verify','start','stop'),TodoImplementationDaemon:('_compile_implementation_context',)}
events=[];lock=threading.Lock();originals=[];started=time.monotonic()
def wrapped(original,label):
 def call(*a,**k):
  begin=time.monotonic();outcome='returned'
  try:return original(*a,**k)
  except BaseException:
   outcome='raised';raise
  finally:
   event=dict(method=label,thread=threading.current_thread().name,entered_seconds=begin-started,seconds=time.monotonic()-begin,outcome=outcome)
   with lock:
    if len(events)<256:events.append(event)
 return call
for cls,names in owners.items():
 for name in names:
  original=getattr(cls,name);originals.append((cls,name,original));setattr(cls,name,wrapped(original,cls.__name__+'.'+name))
source_paths=[A/'ipfs_accelerate_py/agent_supervisor/runtime/task_context_bundle.py',A/'ipfs_accelerate_py/agent_supervisor/entrypoints/admitted_benchmark_runtime.py',A/'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py',*[A/c.split('::')[0] for c in cases],Path(__file__)]
pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
try:
 code=int(pytest.main(['-q',*cases,'--junitxml='+str(B/'post-context.xml')]))
finally:
 for cls,name,original in originals:setattr(cls,name,original)
 (B/'post-context-events.json').write_text(json.dumps(dict(schema='authored-post-context-lifecycle-timing@1',events=events,
  source_pins=pins,scope='External method timers on actual native START/bootstrap/STOP and prompt compilation controls. Source384 validator is explicitly authored by those tests; fixture is small. No model or provider. Not a full benchmark latency estimate.',
  wrappers_restored=True,model_calls=0,provider_calls=0,docker_runs=0,
  runtime_bounds_unchanged=True,independent_native_source384_latency_qualified=False),indent=2)+'\n')
raise SystemExit(code)
