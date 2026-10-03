from pathlib import Path
import hashlib,json,os,subprocess,sys,time
B=Path(__file__).resolve().parent
A=Path('/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002');D=A.parent/'ir-release-datasets-20261002'
record=json.loads((B/'integration-preimages.json').read_text())
def pins():
    return {tag:{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in record[tag]['files']} for tag,root in [('A',A),('D',D)]}
tag=sys.argv[1];selected=sys.argv[2:]
env={'PYTHONPATH':str(A)+':'+str(D),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','NUMEXPR_MAX_THREADS':'1','CUDA_VISIBLE_DEVICES':'','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB':str(B/(tag+'-seal.duckdb')),'IPFS_DATASETS_PY_SEAL_KEY_STORE_PATH':str(B/(tag+'-keys.db'))}
if os.environ.get('HEADER_TRANSPORT_SITE'):env['PYTHONPATH']+=':'+str(Path(os.environ['HEADER_TRANSPORT_SITE']).resolve(strict=True))
env.update({k:os.environ[k] for k in ('CODEBASE384_CHECKPOINT','CODEBASE384_EMBEDDING_SNAPSHOT','IPFS_DATASETS_PUBLIC_BOTTLE_FIXTURE') if k in os.environ})
argv=[str(Path.home()/'.local/bin/python'),'-B','-m','pytest','-q',*selected,'--junitxml='+str(B/(tag+'.xml'))]
before=pins()
(B/(tag+'-command.json')).write_text(json.dumps(dict(argv=argv,cwd=str(D),environment_overrides=env,source_pins=before,source_overlay=False,executed=True),indent=2)+'\n')
started=time.monotonic()
with (B/(tag+'.log')).open('xb') as log:
    result=subprocess.run(argv,cwd=D,env={**os.environ,**env},stdout=log,stderr=subprocess.STDOUT,timeout=300)
after=pins()
receipt=dict(returncode=result.returncode,seconds=time.monotonic()-started,source_pins_unchanged=before==after,after_source_pins=after)
(B/(tag+'-exit.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='after_source_pins'}))
raise SystemExit(result.returncode if before==after else 99)
