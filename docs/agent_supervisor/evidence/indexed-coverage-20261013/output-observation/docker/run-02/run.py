"""Fresh offline Docker qualification; only public inputs and closed observations retained."""
from pathlib import Path
import hashlib,json,os,subprocess,time,shutil,sys
R=Path(__file__).resolve().parent;W=Path('/home/barberb/lift_coding');P=W/'.worktrees/supervisor-comparison-contracts-20261011';D=W/'.worktrees/indexed-coverage-datasets-20261013';K=W/'.worktrees/publication-main-20261004/kit'
ROOT='/opt/ipfs-supervisor';PY=ROOT+'/venv/bin/python';IMAGE='sha256:771900c18b663b33a5bf96e066346b2d5c842506523efb167aebe8d3a99b5c07'
O=R/sys.argv[1];assert O.parent==R and not O.exists();O.mkdir();results=O/'results';results.mkdir();results.chmod(0o777)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def run(argv,stage,**kwargs):
 with (O/(stage+'.stdout')).open('w') as stdout,(O/(stage+'.stderr')).open('w') as stderr:
  p=subprocess.run(argv,stdout=stdout,stderr=stderr,**kwargs)
 p.check_returncode();return p
sources=[P/'benchmarks/agent_supervisor/container_coding/terminal_output_observation.py',P/'ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py',P/'ipfs_accelerate_py/agent_supervisor/runtime/event_log.py',R/'owner_probe.py',Path(__file__)]
before={str(p):sha(p) for p in sources}
for name in ('owner_probe.py','run.py'):shutil.copyfile(R/name,O/name)
env=dict(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='',IPFS_KIT_DISABLE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONHASHSEED='0',IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR=ROOT+'/state/orchestration',IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=ROOT+'/state/seal.duckdb',PYTHONPATH=':'.join(ROOT+'/'+n for n in ('source','datasets','kit')))
source=W/'.benchmarks/terminal-bench-2/largest-eigenval';mounts=[(P,ROOT+'/source',True),(D,ROOT+'/datasets',True),(K,ROOT+'/kit',True),(Path('/home/barberb/.duckdb/extensions/v1.5.5/linux_arm64'),'/extensions',True),(R/'owner_probe.py','/qualification/owner_probe.py',True),(W/'artifacts/terminal-suite-eigenval-pilot-20261004/largest-eigenval-profile.json','/permitted/profile.json',True),(results,'/results',False)]
mounts += [(source/rel,'/permitted/'+name,True) for rel,name in [('environment/src/eigen.py','eigen.py'),('environment/src/eval.py','eval.py'),('instruction.md','instruction.md')]]
name='output-observation-'+str(time.time_ns());argv=['docker','create','--name',name,'--network','none','--cpus','5','--memory','12g','--pids-limit','128','--user','0']
for host,target,readonly in mounts:argv += ['--mount',f'type=bind,src={host},dst={target}'+(',readonly' if readonly else '')]
for k,v in env.items():argv += ['--env',k+'='+v]
argv += ['--entrypoint','/bin/sleep',IMAGE,'600']
write(O/'command.json',dict(command=argv,environment=env,source_before=before,scope='Real public-profile admission/event writer and bounded observer; authored settlement and invocation fixtures, no full START or provider.'))
setup="""import os,pathlib,pwd
root=pathlib.Path('/opt/ipfs-supervisor')
assert pwd.getpwnam('supervisor').pw_uid==1000 and pwd.getpwnam('benchmarkworker').pw_uid==1001
for name,mode in [('state',0o700),('repository',0o755),('worktrees',0o751)]:
 p=root/name;p.mkdir(exist_ok=True);os.chown(p,1000,1000);p.chmod(mode)
for name in ['.local','.duckdb','.ipfs_accelerate']:
 p=pathlib.Path('/home/supervisor')/name;p.mkdir(exist_ok=True);os.chown(p,1000,1000);p.chmod(0o700)
"""
worker="""import os,pathlib
assert os.geteuid()==1001
try:pathlib.Path('/opt/ipfs-supervisor/state/observer-fixture.json').read_bytes()
except PermissionError:pass
else:raise AssertionError('worker read owner state')
p=pathlib.Path('/opt/ipfs-supervisor/worktrees/owned/eigen.py')
with p.open('ab') as f:f.write(b'\\n# Authored output-observer fixture; no benchmark solution.\\n')
print('authored edit completed; owner state inaccessible')
"""
record=dict(schema='declared-output-observation-docker-qualification@1',complete=False,image_id=IMAGE,network='none',provider_calls=0,official_verifier_executed=False,supervisor_start_executed=False,authored_lifecycle_fixture=True,private_state_retained=False)
container=None;started=time.monotonic()
try:
 container=subprocess.check_output(argv,text=True,timeout=30).strip();record['container_id']=container
 run(['docker','start',container],'start',timeout=30)
 run(['docker','exec','--user','0',container,PY,'-I','-B','-c',setup],'setup',timeout=30)
 run(['docker','exec','--user','1000:1000','--env','HOME=/home/supervisor',container,PY,'-B','/qualification/owner_probe.py','prepare'],'prepare',timeout=300)
 run(['docker','exec','--user','1001:1001',container,PY,'-I','-B','-c',worker],'worker',timeout=30)
 run(['docker','exec','--user','1000:1000','--env','HOME=/home/supervisor',container,PY,'-B','/qualification/owner_probe.py','observe'],'observe',timeout=30)
 record['complete']=json.loads((results/'observation.json').read_text())['status']=='observed'
except Exception as error:record['error_type']=type(error).__name__
finally:
 if container:
  removed=subprocess.run(['docker','rm','-f',container],capture_output=True,timeout=30);record['container_removed']=removed.returncode==0
 record['source_after']={str(p):sha(p) for p in sources};record['source_unchanged']=record['source_after']==before;record['seconds']=time.monotonic()-started
 write(O/'qualification.json',record);print(json.dumps(record))
