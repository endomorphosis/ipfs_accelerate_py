from pathlib import Path
import hashlib,json,os,subprocess,sys,time
B=Path(__file__).resolve().parent
W=B.parents[1]; A=W/'.worktrees/ir-release-accelerate-20261002'; D=W/'.worktrees/ir-release-datasets-20261002'
prefix=sys.argv[1]
files=['ipfs_accelerate_py/agent_supervisor/runtime/task_context_bundle.py','test/api/semantic_state/test_source384_task_context_bundle.py','test/api/semantic_state/test_source384_receipt_reference.py']
tests=files[1:]+sys.argv[2:]
files += [t.split('::')[0] for t in sys.argv[2:]]
env=dict(os.environ,PYTHONPATH=f'{A}:{D}',PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=str(B/(prefix+'-seal.duckdb')))
cmd=[sys.executable,'-m','pytest','-q',*[str(A/t) for t in tests],'--junitxml='+str(B/(prefix+'.xml'))]
pins=lambda:{f:hashlib.sha256((A/f).read_bytes()).hexdigest() for f in files}
before=pins();snap=B/(prefix+'-sources');snap.mkdir()
for f in files:
 p=snap/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((A/f).read_bytes())
(B/(prefix+'-command.json')).write_text(json.dumps(dict(argv=cmd,cwd=str(A),source_pins=before,environment={k:env[k] for k in ['PYTHONPATH','PYTHONDONTWRITEBYTECODE','PYTEST_DISABLE_PLUGIN_AUTOLOAD','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB']}),indent=2)+'\n')
start=time.monotonic()
with (B/(prefix+'.log')).open('wb') as log:
 result=subprocess.run(cmd,cwd=A,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
receipt=dict(exit_code=result.returncode,seconds=time.monotonic()-start,source_pins_after=pins(),source_pins_unchanged=before==pins())
(B/(prefix+'-exit.json')).write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt));raise SystemExit(result.returncode)
