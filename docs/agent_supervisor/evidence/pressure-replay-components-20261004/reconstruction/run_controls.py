import hashlib, json, os, pathlib, subprocess, sys, time
base=pathlib.Path(__file__).resolve().parent
label=sys.argv[1]
d=pathlib.Path('/home/barberb/lift_coding/.worktrees/ir-pressure-attribution-datasets-20261004')
test='tests/integration/logic/software_contracts/test_codebase_observation_reconstruction.py'
private=base/'private'/label
private.mkdir(parents=True,exist_ok=False)
paths=[test,'ipfs_datasets_py/logic/software_contracts/codebase_ir.py','ipfs_datasets_py/logic/software_contracts/ast_ir.py','ipfs_datasets_py/logic/software_contracts/cache.py','ipfs_datasets_py/logic/software_contracts/duckdb_ast_store.py','ipfs_datasets_py/logic/software_contracts/schema_versions.py','ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py','ipfs_datasets_py/optimizers/logic_theorem_optimizer/proof_resource_safety.py','tests/integration/logic/software_contracts/test_codebase_current.py']
def pins(): return {p:hashlib.sha256((d/p).read_bytes()).hexdigest() for p in paths}
overrides={'PYTHONPATH':str(d),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','NUMEXPR_MAX_THREADS':'1','CUDA_VISIBLE_DEVICES':'','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB':str(private/'seals.duckdb'),'IPFS_DATASETS_PY_SEAL_KEY_STORE_PATH':str(private/'keys.db')}
argv=['/home/barberb/.local/bin/python','-B','-m','pytest','-q',test,'--junitxml='+str(base/(label+'.xml'))]
record={'argv':argv,'cwd':str(d),'environment_overrides':overrides,'source_pins':pins()}
(base/(label+'-command.json')).write_text(json.dumps(record,indent=2)+'\n')
started=time.monotonic()
with (base/(label+'-stdout.txt')).open('w') as out, (base/(label+'-stderr.txt')).open('w') as err:
 result=subprocess.run(argv,cwd=d,env=dict(os.environ,**overrides),stdout=out,stderr=err)
exit={'returncode':result.returncode,'elapsed_seconds':time.monotonic()-started,'source_pins_after':pins(),'source_pins_unchanged':record['source_pins']==pins()}
(base/(label+'-exit.json')).write_text(json.dumps(exit,indent=2)+'\n')
print(json.dumps(exit))
sys.exit(result.returncode)
