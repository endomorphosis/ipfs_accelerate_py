from pathlib import Path
import hashlib,json,os,subprocess,sys,time
B=Path(__file__).resolve().parent
A=Path('/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002')
D=Path('/home/barberb/lift_coding/.worktrees/ir-release-datasets-20261002')
label='actual-01'
env=dict(os.environ,PYTHONPATH=f'{A}:{D}',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',NUMEXPR_MAX_THREADS='1',CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=str(B/(label+'-seal.duckdb')),IPFS_DATASETS_PY_SEAL_KEY_STORE_PATH=str(B/(label+'-key-seal.db')))
paths=['benchmarks/agent_supervisor/container_coding/test_terminal_initial_context_lifetime.py','benchmarks/agent_supervisor/container_coding/test_terminal_initial_context.py','test/integration/test_initial_learned_context_reuse.py']
av=[sys.executable,'-B','-m','pytest','-q',*paths,'--basetemp='+str(B/(label+'-temp')),'--junitxml='+str(B/(label+'.xml'))]
pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [A/'benchmarks/agent_supervisor/container_coding/terminal_initial_context.py',*[A/p for p in paths]]}
(B/(label+'-command.json')).write_text(json.dumps(dict(argv=av,cwd=str(A),environment_overrides={k:v for k,v in env.items() if k not in os.environ or os.environ[k]!=v},source_pins=pins,scope='Actual on-disk native module and tests. No module overlay, no model/provider calls. One authored Source384 seam isolates only map lifetime.'),indent=2)+'\n')
t=time.monotonic()
with (B/(label+'.log')).open('w') as out: p=subprocess.run(av,cwd=A,env=env,stdout=out,stderr=subprocess.STDOUT)
assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in pins.items())
receipt=dict(returncode=p.returncode,elapsed_seconds=time.monotonic()-t,source_pins_unchanged=True)
(B/(label+'-exit.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));raise SystemExit(p.returncode)
