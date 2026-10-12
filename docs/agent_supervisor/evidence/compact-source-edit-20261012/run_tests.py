import hashlib,json,os,subprocess,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent
P=Path('/home/barberb/lift_coding/.worktrees/supervisor-comparison-contracts-20261011')
run=R/sys.argv[1];run.mkdir(exist_ok=False)
base=json.loads((R/'test-environment.json').read_text());env=dict(base['environment'])
env.update(IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=str(run/'seal.duckdb'),IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR=str(run/'orchestration'))
files=sys.argv[2:]
command=[base['python'],'-B','-m','pytest','-q','--color=no',*files,'--junitxml='+str(run/'junit.xml'),'-o','cache_dir='+str(run/'pytest-cache'),'-o','log_cli=false','--tb=short']
paths=set(files)
for args in (['diff','--name-only','HEAD','--','*.py'],['ls-files','--others','--exclude-standard','--','*.py']):
 paths.update(subprocess.check_output(['git',*args],cwd=P,text=True).splitlines())
paths.update(['ipfs_accelerate_py/agent_supervisor/runtime/reversible_source_view.py','ipfs_accelerate_py/agent_supervisor/runtime/worker_source_views.py','ipfs_accelerate_py/agent_supervisor/semantic_state/worktree.py'])
paths={P/name for name in paths};paths.update([Path(__file__).resolve(),R/'test-environment.json'])
def hashes():return {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
before=hashes();(run/'command.json').write_text(json.dumps(dict(command=command,cwd=str(P),environment_overrides=env,source_before=before),indent=2)+'\n')
t=time.monotonic()
with (run/'pytest.log').open('w') as stream:
 proc=subprocess.run(command,cwd=P,env={**os.environ,**env},stdout=stream,stderr=subprocess.STDOUT)
counts={k:sum(int(s.attrib.get(k,0)) for s in ET.parse(run/'junit.xml').getroot().iter('testsuite')) for k in ('tests','failures','errors','skipped')} if (run/'junit.xml').exists() else None
after=hashes();result=dict(exit_code=proc.returncode,seconds=time.monotonic()-t,counts=counts,source_unchanged=before==after,source_after=after,provider_calls=0,provider_tokens=None,total_token_savings_measured=False)
(run/'qualification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_after'}))
if proc.returncode:print((run/'pytest.log').read_text()[-16000:])
raise SystemExit(proc.returncode)
