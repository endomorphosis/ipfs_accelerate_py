from pathlib import Path
import json,os,subprocess,sys,time
B=Path(__file__).resolve().parent
W=B.parents[1];A=W/'.worktrees/ir-release-accelerate-20261002'
prior=json.loads((W/'artifacts/source384-replay-lifetime-20261003/actual-a-context-command.json').read_bytes())
overrides=prior['environment_overrides']
for k in ('IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB','IPFS_DATASETS_PY_SEAL_KEY_STORE_PATH'):
    overrides[k]=str(B/Path(overrides[k]).name)
env={**os.environ,**overrides}
argv=[sys.executable,'-B',str(B/'run_configured_a.py')]
(B/'actual-a-context-controller-command.json').write_text(json.dumps(dict(argv=argv,cwd=str(A),environment_overrides=overrides),indent=2)+'\n')
t=time.monotonic()
with (B/'actual-a-context.log').open('w') as out:
    p=subprocess.run(argv,cwd=A,env=env,stdout=out,stderr=subprocess.STDOUT)
receipt=dict(returncode=p.returncode,elapsed_seconds=time.monotonic()-t)
(B/'actual-a-context-controller-exit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt));raise SystemExit(p.returncode)
