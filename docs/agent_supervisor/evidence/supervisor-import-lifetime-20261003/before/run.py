from pathlib import Path
import json,os,subprocess,time,hashlib
B=Path(__file__).resolve().parent;W=B.parent.parent;A=W/'.worktrees/ir-release-accelerate-20261002';D=A.parent/'ir-release-datasets-20261002'
program='''import importlib,json,sys,time,pathlib,resource,gc
started=time.monotonic()
for name in sys.argv[1:]:importlib.import_module(name)
gc.collect()
status={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in pathlib.Path('/proc/self/status').read_text().splitlines() if ':' in line}
rollup={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in pathlib.Path('/proc/self/smaps_rollup').read_text().splitlines() if ':' in line}
print(json.dumps(dict(seconds=time.monotonic()-started,rss=status.get('VmRSS'),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,rollup=rollup,modules=sorted(sys.modules),imports=sys.argv[1:])))
'''
env={'PYTHONPATH':str(A)+':'+str(D),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','NUMEXPR_MAX_THREADS':'1','CUDA_VISIBLE_DEVICES':'','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1'}
for tag,name in [('prepare','terminal_indexed_preparation'),('driver','terminal_container_supervisor')]:
 argv=[str(Path.home()/'.local/bin/python'),'-B','-c',program,'benchmarks.agent_supervisor.container_coding.'+name]
 (B/(tag+'-command.json')).write_text(json.dumps(dict(argv=argv,cwd=str(A),environment_overrides=env,source_sha256=hashlib.sha256((A/'benchmarks/agent_supervisor/container_coding'/ (name+'.py')).read_bytes()).hexdigest(),native_container_claim=False),indent=2)+'\n')
 started=time.monotonic()
 with (B/(tag+'.stdout')).open('x') as out,(B/(tag+'.stderr')).open('x') as err:r=subprocess.run(argv,cwd=A,env={**os.environ,**env},stdout=out,stderr=err,timeout=120)
 (B/(tag+'-exit.json')).write_text(json.dumps(dict(returncode=r.returncode,seconds=time.monotonic()-started),indent=2)+'\n')
 assert r.returncode==0
 data=json.loads((B/(tag+'.stdout')).read_text().splitlines()[-1]);(B/(tag+'.json')).write_text(json.dumps(data,indent=2)+'\n');print(tag,{k:v for k,v in data.items() if k in {'seconds','rss','peak_rss_kib'}})
