import hashlib,json,os,subprocess,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
P=Path('/home/barberb/lift_coding/.worktrees/supervisor-comparison-contracts-20261011')
R=Path(__file__).parent/sys.argv[1]; R.mkdir(exist_ok=False)
previous=json.loads(Path('/home/barberb/lift_coding/artifacts/working-source-refresh-20261012/final-tests-01/command.json').read_text())
env=dict(previous['environment_overrides'])
env.update(IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=str(R/'seal.duckdb'), IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR=str(R/'orchestration'), DOCTOR_COMPOSITION_LEAN='/home/barberb/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lean',ELAN_TOOLCHAIN='leanprover/lean4:v4.34.1')
files=sys.argv[2:]
command=[previous['command'][0],'-B','-m','pytest','-q','--color=no',*files,'--junitxml='+str(R/'junit.xml'),'-o','cache_dir='+str(R/'pytest-cache'),'-o','log_cli=false','--tb=short']
paths=set(files)
for args in (['diff','--name-only','--','*.py'],['ls-files','--others','--exclude-standard','--','*.py']):
 paths.update(subprocess.check_output(['git',*args],cwd=P,text=True).splitlines())
paths={P/path for path in paths}
paths.update(Path(path) for path in previous['source_before'] if Path(path).exists())
paths.add(Path(__file__))
def hashes():return {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
before=hashes(); (R/'command.json').write_text(json.dumps(dict(command=command,cwd=str(P),environment_overrides=env,source_before=before),indent=2)+'\n')
t=time.monotonic()
with (R/'pytest.log').open('w') as stream:r=subprocess.run(command,cwd=P,env={**os.environ,**env},stdout=stream,stderr=subprocess.STDOUT)
counts={k:sum(int(s.attrib.get(k,0)) for s in ET.parse(R/'junit.xml').getroot().iter('testsuite')) for k in ('tests','failures','errors','skipped')}
after=hashes(); result=dict(exit_code=r.returncode,seconds=time.monotonic()-t,counts=counts,source_unchanged=before==after,source_after=after,provider_calls=0,provider_tokens=None,total_token_savings_measured=False)
(R/'qualification.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({k:v for k,v in result.items() if k!='source_after'}))
if r.returncode:print((R/'pytest.log').read_text()[-16000:])
raise SystemExit(r.returncode)
