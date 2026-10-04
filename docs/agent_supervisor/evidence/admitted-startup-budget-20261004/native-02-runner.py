import hashlib,json,os,subprocess,sys,time
from pathlib import Path
B=Path(__file__).resolve().parent
A=Path('/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002')
recipe=json.loads(Path('/home/barberb/lift_coding/artifacts/native-completion-binding-20261004/regression-02-command.json').read_text())
label=sys.argv[1];targets=sys.argv[2:];private=B/('private-'+label);private.mkdir()
basetemp=Path('/tmp/ipfs-asb-'+label)
assert not basetemp.exists(), 'test basetemp must be fresh'
env=recipe['environment_overrides'].copy()
for key,name in [('IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB','seal.duckdb'),('IPFS_DATASETS_PY_SEAL_KEY_STORE_PATH','keys.db'),('IPFS_DATASETS_RESOURCE_SCHEDULER_PATH','scheduler.json')]:env[key]=str(private/name)
argv=['/home/barberb/.local/bin/python','-B','-m','pytest','-q','-o','cache_dir='+str(private/'pytest-cache'),'--basetemp='+str(basetemp),*targets,'--junitxml='+str(B/(label+'.xml'))]
files=list(json.loads((B/'original-pins.json').read_text()))+['ipfs_accelerate_py/agent_supervisor/control/control_plane.py','test/integration/test_admitted_startup_budget.py']
files += [t.split('::')[0] for t in targets if (A/t.split('::')[0]).is_file() and t.split('::')[0] not in files]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pins={f:sha(A/f) for f in files}
heads={name:subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip() for name,path in [('source',A),('datasets',Path('/home/barberb/lift_coding/.worktrees/ir-pressure-attribution-datasets-20261004'))]}
command=dict(argv=argv,cwd=str(A),environment_overrides=env,source_pins=pins,repository_heads=heads,runner_sha256=sha(Path(__file__)),providers_called=False,production_scheduler_used=False,hidden_verifier_used=False)
(B/(label+'-command.json')).write_text(json.dumps(command,indent=2)+'\n')
t=time.monotonic()
with (B/(label+'.stdout')).open('x') as out,(B/(label+'.stderr')).open('x') as err:r=subprocess.run(argv,cwd=A,env={**os.environ,**env},stdout=out,stderr=err)
receipt=dict(returncode=r.returncode,seconds=time.monotonic()-t,source_pins_unchanged=all(sha(A/f)==h for f,h in pins.items()))
(B/(label+'-exit.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));raise SystemExit(r.returncode)
