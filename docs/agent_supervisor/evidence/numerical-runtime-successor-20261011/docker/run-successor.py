import os, sys, json, hashlib, subprocess, time
from pathlib import Path
root = Path(__file__).parent
reference=json.loads((root/'helper-01/command.json').read_text())
out=root/sys.argv[1]; out.mkdir()
env=dict(reference['environment_overrides']); env['IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB']=str(out/'seal.duckdb')
paths=list(reference['source_before'])+[str(Path(reference['cwd'])/name) for name in ['test/integration/test_admitted_learned_numerical_successor.py', 'test/api/test_admitted_observation_work_scope.py', 'test/integration/test_admitted_context_refresh.py', 'test/api/semantic_state/test_published_task_context.py', 'test/integration/test_admitted_numerical_successor.py', 'test/integration/test_admitted_source384_successor.py', 'test/api/semantic_state/test_learned_numerical_task_context_bundle.py']]
def hashes(): return {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
command=[reference['command'][0],'-B','-m','pytest','-q','--color=no',*sys.argv[2:],'--junitxml='+str(out/'junit.xml'),'-o','cache_dir='+str(out/'pytest-cache')]
before=hashes(); (out/'command.json').write_text(json.dumps(dict(command=command,cwd=reference['cwd'],environment_overrides=env,source_before=before,scope='Real cached learned checkpoint, local publication, native START/STOP; no providers/downloads/training. Optional Source384 and elapsed phase clock explicitly authored.'),indent=2,sort_keys=True)+'\n')
t=time.monotonic()
with (out/'pytest.log').open('w') as f: result=subprocess.run(command,cwd=reference['cwd'],env={**os.environ,**env},stdout=f,stderr=subprocess.STDOUT)
after=hashes(); (out/'exit.json').write_text(json.dumps(dict(exit_code=result.returncode,wall_seconds=time.monotonic()-t,source_after=after,stable_sources=before==after),indent=2,sort_keys=True)+'\n')
print(out, result.returncode, flush=True)
sys.exit(result.returncode)
