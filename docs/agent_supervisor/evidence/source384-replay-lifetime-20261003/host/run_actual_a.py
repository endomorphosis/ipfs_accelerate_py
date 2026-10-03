from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
W=ROOT.parents[1];A=W/'.worktrees/ir-release-accelerate-20261002';D=W/'.worktrees/ir-release-datasets-20261002'
mode=sys.argv[1];assert mode in ('lifetime','context')
name='actual-a-'+mode
paths={'lifetime':'test/api/semantic_state/test_source384_context_lifetime.py','context':'benchmarks/agent_supervisor/container_coding/test_terminal_source384_context.py'}
command=[sys.executable,'-B','-m','pytest','-q',paths[mode],'--basetemp='+str(ROOT/(name+'-temp')),'--junitxml='+str(ROOT/(name+'.xml'))]
selected=json.loads((W/'artifacts/terminal-source384-context-20261003/qualified-manifest-final-command.json').read_text())['env']
selected.update(PYTHONPATH=str(A)+os.pathsep+str(D),IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=str(ROOT/(name+'-seal.duckdb')),IPFS_DATASETS_PY_SEAL_KEY_STORE_PATH=str(ROOT/(name+'-key-seal.db')))
files=[A/'ipfs_accelerate_py/agent_supervisor/runtime/source384_repository_context.py',D/'ipfs_datasets_py/logic/software_contracts/codebase_source_units_384.py',D/'ipfs_datasets_py/logic/software_contracts/codebase_ir.py',D/'ipfs_datasets_py/logic/software_contracts/cache.py',A/paths[mode]]
pins=lambda:{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
before=pins();scope={'argv':command,'cwd':str(A),'environment_overrides':selected,'source_pins':before,'scope':'Actual on-disk owners/tests, no import overlay. Native worker uses same interpreter site roots; no Harbor dependency path.'}
(ROOT/(name+'-command.json')).write_text(json.dumps(scope,indent=2)+'\n')
start=time.monotonic()
with (ROOT/(name+'.log')).open('w') as log:r=subprocess.run(command,cwd=A,env=dict(os.environ,**selected),stdout=log,stderr=subprocess.STDOUT)
after=pins();receipt={'returncode':r.returncode,'elapsed_seconds':time.monotonic()-start,'source_pins_after':after,'sources_unchanged':after==before}
(ROOT/(name+'-exit.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt));print((ROOT/(name+'.log')).read_text()[-7000:]);assert before==after
raise SystemExit(r.returncode)
