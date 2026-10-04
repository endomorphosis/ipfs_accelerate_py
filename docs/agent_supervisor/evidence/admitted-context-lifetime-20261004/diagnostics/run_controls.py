import hashlib,json,os,subprocess,sys,time
from pathlib import Path
out=Path(__file__).resolve().parent
root=Path('/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002')
d=Path('/home/barberb/lift_coding/.worktrees/ir-admission-observation-datasets-20261004')
phase=sys.argv[1]
targets=sys.argv[2:]
env_overrides=json.loads(Path('/home/barberb/lift_coding/artifacts/semantic-preparation-profile-20261004/accelerate-command.json').read_text())['environment_overrides']
private=out/'private-controls-state'/phase
private.mkdir(parents=True,exist_ok=False)
env_overrides.update(IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=str(private/'seal.duckdb'),IPFS_DATASETS_PY_SEAL_KEY_STORE_PATH=str(private/'keys.db'),IPFS_DATASETS_RESOURCE_SCHEDULER_PATH=str(private/'scheduler.sqlite3'))
if phase in ('baseline-ownership','baseline-capture-ownership'):
 env_overrides['PYTHONPATH']=str(out)+':'+env_overrides['PYTHONPATH']
 env_overrides['LIFETIME_BASELINE_BIND_SOURCE']=str(out/('before' if phase=='baseline-ownership' else 'minimal-generation')/'terminal_initial_context.py')
 targets=['-p','before_bind_plugin',*targets]
argv=['/home/barberb/.local/bin/python','-B','-m','pytest','-q','-o','cache_dir='+str(private/'pytest-cache'),*targets,'--junitxml='+str(out/(phase+'.xml'))]
source_paths=[root/'benchmarks/agent_supervisor/container_coding/terminal_initial_context.py',root/'benchmarks/agent_supervisor/container_coding/test_terminal_initial_context_lifetime.py']+[root/t for t in targets if t.endswith('.py')]
sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
source_dir=out/(phase+'-sources')
source_dir.mkdir()
for name in sources:
 target=source_dir/name
 target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes((root/name).read_bytes())
command={'argv':argv,'cwd':str(root),'environment_overrides':env_overrides,'source_sha256':sources,'production_scheduler_ledger_mutated':False,'provider_calls':0,'hidden_verifier_used':False}
if phase in ('baseline-ownership','baseline-capture-ownership'):
 command['archived_bind_override']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'before_bind_plugin.py',Path(env_overrides['LIFETIME_BASELINE_BIND_SOURCE']))}
(out/(phase+'-command.json')).write_text(json.dumps(command,indent=2)+'\n')
started=time.monotonic()
with (out/(phase+'-stdout.txt')).open('w') as stdout,(out/(phase+'-stderr.txt')).open('w') as stderr:
 result=subprocess.run(argv,cwd=root,env={**os.environ,**env_overrides},stdout=stdout,stderr=stderr)
status={'returncode':result.returncode,'seconds':time.monotonic()-started,'source_pins_unchanged':all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in sources.items())}
(out/(phase+'-exit.json')).write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status))
