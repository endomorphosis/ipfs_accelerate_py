from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,sys,time
B=Path(__file__).resolve().parent;W=B.parents[1];A=W/'.worktrees/ir-release-accelerate-20261002';D=W/'.worktrees/ir-release-datasets-20261002'
if '--child' in sys.argv:
 sys.path[:0]=[str(A),str(D)]
 if os.environ['NOMINATION_MODE']=='candidate':
  import benchmarks.agent_supervisor.container_coding as package
  for leaf in ('terminal_initial_context','terminal_indexed_preparation'):
   name=package.__name__+'.'+leaf;path=B/'proposed-A/benchmarks/agent_supervisor/container_coding'/(leaf+'.py')
   spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);setattr(package,leaf,module)
 import pytest
 raise SystemExit(pytest.main(sys.argv[sys.argv.index('--child')+1:]))
mode,label=sys.argv[1:3]
env=dict(os.environ,PYTHONPATH=f'{A}:{D}',PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NOMINATION_MODE=mode,IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=str(B/(label+'-seal.duckdb')))
test=B/'proposed-A/benchmarks/agent_supervisor/container_coding/test_terminal_nomination_staging.py'
args=['-q',str(test),*sys.argv[3:],'--junitxml='+str(B/(label+'.xml'))]
cmd=[sys.executable,'-B',str(Path(__file__)),'--child',*args]
files=list((B/'proposed-A').rglob('*.py'))+[Path(__file__)]
pins=lambda:{str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
before=pins();snapshot=B/(label+'-sources');snapshot.mkdir()
for p in files:
 q=snapshot/p.relative_to(B);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
(B/(label+'-command.json')).write_text(json.dumps(dict(argv=cmd,cwd=str(A),mode=mode,source_pins=before,environment={k:env[k] for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','PYTEST_DISABLE_PLUGIN_AUTOLOAD','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NOMINATION_MODE','IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB')},scope='Whole-module off-tree overlay; authored numerical observer, native signed manifests/indexes/world/intent. No model/provider/Docker.'),indent=2)+'\n')
start=time.monotonic()
with (B/(label+'.log')).open('wb') as out:r=subprocess.run(cmd,cwd=A,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=240)
receipt=dict(exit_code=r.returncode,seconds=time.monotonic()-start,source_pins_unchanged=pins()==before)
(B/(label+'-exit.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));raise SystemExit(r.returncode)
