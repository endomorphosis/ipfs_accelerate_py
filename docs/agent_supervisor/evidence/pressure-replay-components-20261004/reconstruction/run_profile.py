import json, os, pathlib, subprocess, time
b=pathlib.Path(__file__).resolve().parent
d=pathlib.Path('/home/barberb/lift_coding/.worktrees/ir-pressure-attribution-datasets-20261004')
env={'PYTHONPATH':str(d),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','NUMEXPR_MAX_THREADS':'1','CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
argv=['/home/barberb/.local/bin/python','-B',str(b/'profile_observation.py')]
(b/'paired-command.json').write_text(json.dumps({'argv':argv,'cwd':str(d),'environment_overrides':env,'timeout_seconds':75},indent=2)+'\n')
t=time.monotonic();exit={}
try:
 with (b/'paired-stdout.txt').open('w') as o,(b/'paired-stderr.txt').open('w') as e:
  result=subprocess.run(argv,cwd=d,env=dict(os.environ,**env),stdout=o,stderr=e,timeout=75)
 exit['returncode']=result.returncode
except subprocess.TimeoutExpired:exit={'returncode':None,'timeout':True}
exit['elapsed_seconds']=time.monotonic()-t
(b/'paired-exit.json').write_text(json.dumps(exit,indent=2)+'\n');print(json.dumps(exit))
