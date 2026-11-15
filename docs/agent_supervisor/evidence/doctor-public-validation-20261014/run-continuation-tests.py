"""Offline authored qualification only. No campaign or model invocation."""
import hashlib,importlib.util,json,subprocess,time,sys,xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
W=Path('/home/barberb/lift_coding/.worktrees/supervisor-public-validation-20261014')
A=Path('/home/barberb/lift_coding/artifacts/public-validation-continuation-20261014')
PY='/home/barberb/lift_coding/.venvs/terminal-bench-harbor/bin/python'
label=sys.argv[1]
assert label and all(c.isalnum() or c=='-' for c in label)
selection=sys.argv[2:]; files=[value for value in selection if value.endswith('.py')]; extra=[value for value in selection if not value.endswith('.py')]
assert files and all(p.endswith('.py') for p in files)

changed=subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=W,text=True).splitlines()
new=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=W,text=True).splitlines()
paths=sorted({p for p in changed+new+files if p.endswith('.py')})
def pins():return {p:hashlib.sha256((W/p).read_bytes()).hexdigest() for p in paths}
source=pins()
dependencies={'ipfs_datasets_py':Path('/home/barberb/lift_coding/.worktrees/supervisor-bounded-rounds-datasets-20261014'), 'ipfs_kit_py':Path('/home/barberb/lift_coding/.worktrees/publication-main-20261004/kit')}
def dependency_pins():
 return {name:dict(path=str(path),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip(),tracked_status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=no'],cwd=path,text=True)) for name,path in dependencies.items()}
dependency_before=dependency_pins()
def distribution_pins():
 code="import importlib.metadata as m,hashlib,json; print(json.dumps({n:dict(version=m.version(n),metadata_sha256=hashlib.sha256(m.distribution(n).read_text('METADATA').encode()).hexdigest()) for n in ('multiformats','multiformats-config','bases','typing-validation')}))"
 return json.loads(subprocess.check_output([PY,'-I','-c',code],text=True))
distribution_before=distribution_pins()

(A/(label+'-source-pins.json')).write_text(json.dumps({'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip(),'files':source},indent=2)+'\n')
spec=importlib.util.spec_from_file_location('safe_env','/home/barberb/lift_coding/artifacts/capsule-budget-live-20261014/campaign_common.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
env=m.safe_environment();env['PYTHONPATH']=env.get('PYTHONPATH','').replace(str(m.Q),str(W))
env['PYTHONPATH']=env['PYTHONPATH'].replace('/home/barberb/lift_coding/.worktrees/indexed-coverage-datasets-20261013','/home/barberb/lift_coding/.worktrees/supervisor-bounded-rounds-datasets-20261014')
env['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
env['PYTHONUNBUFFERED']='1'
env['IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB']=str(A/(label+'-seal.duckdb'))
argv=[PY,'-m','pytest','-q','--tb=short','--color=no','-o','log_cli=false','-o','addopts=',*files,*extra,f'--junitxml={A}/{label}.xml']
origins=json.loads(subprocess.check_output([PY,'-c',"import importlib.util,json;print(json.dumps({n:importlib.util.find_spec(n).origin for n in ['ipfs_datasets_py','ipfs_kit_py']}))"],cwd=W,env=env,text=True))
started=time.monotonic()
with (A/(label+'.log')).open('w') as log:result=subprocess.run(argv,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT)
cases=list(ET.parse(A/(label+'.xml')).getroot().iter('testcase'))
counts=Counter('failed' if t.find('failure') is not None else 'error' if t.find('error') is not None else 'skipped' if t.find('skipped') is not None else 'passed' for t in cases)
record=dict(exit_code=result.returncode,seconds=time.monotonic()-started,live_provider_calls=0,source_pins_unchanged=source==pins(),argv=argv,counts=dict(counts),dependencies=dependency_before,dependencies_unchanged=dependency_before==dependency_pins(),dependency_import_origins=origins,distribution_pins=distribution_before,distribution_pins_unchanged=distribution_before==distribution_pins())
(A/(label+'-result.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
for t in cases:
 for failure in t.findall('failure')+t.findall('error'):print(t.attrib['name'],failure.text[:6000])
print((A/(label+'.log')).read_text()[-2000:])
