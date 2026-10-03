from pathlib import Path
import hashlib,json,os,subprocess,time
B=Path(__file__).resolve().parent;W=B.parent.parent;A=W/'.worktrees/ir-release-accelerate-20261002';D=A.parent/'ir-release-datasets-20261002'
names=['test_terminal_import_lifetime.py','test_terminal_failure_diagnostics.py','test_terminal_resource_diagnostics.py','test_cleanup_deadline.py','test_post_stop_refresh_budget.py']
paths=['benchmarks/agent_supervisor/container_coding/'+n for n in names]
source=paths+['benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py']
pins=lambda:{p:hashlib.sha256((A/p).read_bytes()).hexdigest() for p in source}
env=json.loads((B.parent/'header-full-import-census-20261003/prepare-command.json').read_text())['environment_overrides']
env.update(IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=str(B/'actual-seal.duckdb'),IPFS_DATASETS_PY_SEAL_KEY_STORE_PATH=str(B/'actual-keys.db'))
argv=[str(Path.home()/'.local/bin/python'),'-B','-m','pytest','-q',*paths,'--junitxml='+str(B/'actual.xml')]
before=pins();(B/'actual-command.json').write_text(json.dumps(dict(argv=argv,cwd=str(A),environment_overrides=env,source_pins=before),indent=2)+'\n')
started=time.monotonic()
with (B/'actual.log').open('x') as out:r=subprocess.run(argv,cwd=A,env={**os.environ,**env},stdout=out,stderr=subprocess.STDOUT,timeout=300)
receipt=dict(returncode=r.returncode,seconds=time.monotonic()-started,source_pins_unchanged=pins()==before)
(B/'actual-exit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt);assert r.returncode==0 and receipt['source_pins_unchanged']
