from pathlib import Path
import hashlib,json,os,subprocess,sys,time
B=Path(__file__).resolve().parent;W=B.parent.parent;D=W/'.worktrees/ir-admission-observation-datasets-20261004'
mode,label=sys.argv[1:]
assert mode in {'before','after'} and label.startswith('renewed-') and '/' not in label
script=B/'profile_observation_renewed.py'
env=dict(PYTHONPATH=str(D),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
argv=['/home/barberb/.local/bin/python','-B',str(script),mode,label]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with (B/(label+'-controller-command.json')).open('x') as f:json.dump(dict(argv=argv,cwd=str(D),environment_overrides=env,script_sha256=sha(script),controller_sha256=sha(Path(__file__)),timeout_seconds=180.,observation_timeout_seconds=120.,only_isolated_ledger=True),f,indent=2)
start=time.monotonic()
with (B/(label+'.stdout')).open('x') as out,(B/(label+'.stderr')).open('x') as err:
 try:
  run=subprocess.run(argv,cwd=D,env={**os.environ,**env},stdout=out,stderr=err,timeout=180.)
  result=dict(returncode=run.returncode,controller_timed_out=False)
 except subprocess.TimeoutExpired:
  result=dict(returncode=124,controller_timed_out=True)
result['seconds']=time.monotonic()-start
with (B/(label+'-exit.json')).open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True);raise SystemExit(result['returncode'])
