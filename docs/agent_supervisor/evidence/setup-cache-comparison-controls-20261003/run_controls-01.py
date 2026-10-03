from pathlib import Path
import hashlib,json,os,subprocess,time
B=Path(__file__).resolve().parent;W=B.parents[1];A=W/'.worktrees/ir-release-accelerate-20261002';D=W/'.worktrees/ir-release-datasets-20261002'
files=[A/'benchmarks/agent_supervisor/container_coding'/name for name in ['benchmark_controls.py','test_benchmark_controls.py','test_setup_cache_comparison_controls.py']]
argv=[str(W/'.venvs/terminal-bench-harbor/bin/python'),'-m','pytest','-q',str(files[1]),str(files[2]),'--junitxml='+str(B/'controls-01.xml')]
env={'PYTHONPATH':str(A)+':'+str(D),'PYTHONDONTWRITEBYTECODE':'1','PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB':str(B/'controls-01-seal.duckdb'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
record=dict(argv=argv,cwd=str(A),environment=env,source_pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},scope='host-only declaration validation; no task execution, provider/model calls or Docker')
(B/'controls-01-command.json').write_text(json.dumps(record,indent=2)+'\n');started=time.monotonic()
with (B/'controls-01.log').open('w') as out:
 result=subprocess.run(argv,cwd=A,env=dict(os.environ,**env),stdout=out,stderr=subprocess.STDOUT)
(B/'controls-01-exit.json').write_text(json.dumps(dict(exit_code=result.returncode,seconds=time.monotonic()-started,source_pins_after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
raise SystemExit(result.returncode)
