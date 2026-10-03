"""Bounded host diagnostic, not container or source-authority qualification."""
from pathlib import Path
from contextlib import redirect_stdout
import cProfile,hashlib,json,os,pstats,sys,time
sys.path[:0]=json.loads(sys.argv[1])
data=json.loads(sys.stdin.buffer.read(32*1024**2+1))
def execute():
 from ipfs_datasets_py.logic.software_contracts.codebase_source_units_384_worker import execute
 return execute(data)
profile=cProfile.Profile();start=time.monotonic()
with redirect_stdout(sys.stderr):output=profile.runcall(execute)
stats=pstats.Stats(profile)
rows=[dict(file=file,line=line,function=function,primitive_calls=cc,total_calls=nc,self_seconds=tt,cumulative_seconds=ct)
 for (file,line,function),(cc,nc,tt,ct,callers) in stats.stats.items()]
rows.sort(key=lambda r:r['cumulative_seconds'],reverse=True)
raw=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
print(json.dumps(dict(schema='source384-worker-host-profile@1',seconds=time.monotonic()-start,
 input_sha256=hashlib.sha256(raw(data)).hexdigest(),output_sha256=hashlib.sha256(raw(output)).hexdigest(),
 model_loads=output['model_loads'],rows=len(output['rows']),
 candidates=sum(r['candidate'] is not None for r in output['rows']),
 profile_top_cumulative=rows[:65],native_container_claim=False,source_proof_claim=False)))
