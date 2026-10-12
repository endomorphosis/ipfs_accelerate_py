import hashlib,json,shutil,subprocess,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path('/home/barberb/lift_coding')
P=ROOT/'.worktrees/supervisor-comparison-contracts-20261011'
R=ROOT/'artifacts/supervisor-token-efficiency-20261012'
A=R/'archive-01'
attempt=sys.argv[1]
assert attempt.isdecimal() and len(attempt)==2
run=R/('docker-instruction-'+attempt)
run.mkdir(exist_ok=False)
results=run/'results'
results.mkdir();results.chmod(0o777)
source=A/'extracted/source'
files=['conftest.py','pytest.ini','test/pytest.ini','test/conftest.py','test/__init__.py','test/api/__init__.py','test/api/test_semantic_instruction_dispatch.py','test/api/test_router_public_instruction.py','test/api/test_semantic_router_integration.py','test/api/test_semantic_router_translation.py','test/api/test_agent_supervisor_local_planning_admission.py']
files += [str(p.relative_to(P)) for p in (P/'test/common').rglob('*.py') if '__pycache__' not in p.parts]
for relative in files:
 target=source/relative
 if target.exists():
  assert target.read_bytes()==(P/relative).read_bytes()
  continue
 target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(P/relative,target)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
entry=run/'entry.py'
entry.write_text(r"""import hashlib,importlib,json,os,sys
from pathlib import Path
sys.path[:0]=['/code/source','/code/datasets','/code/kit']
os.chdir('/tmp')
from benchmarks.agent_supervisor.container_coding.terminal_semantic_metadata_policy import require_semantic_metadata_archive
manifest=json.loads(Path('/qualification/archive-manifest.json').read_text())
for mode in ('common-bindings@1','common-bindings-instruction@1'):require_semantic_metadata_archive(manifest,mode)
modules={}
for name in ('ipfs_accelerate_py.agent_supervisor.runtime.semantic_instruction_view','ipfs_accelerate_py.agent_supervisor.runtime.semantic_router_translation','ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner','ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime','benchmarks.agent_supervisor.container_coding.terminal_context_audit'):
 module=importlib.import_module(name)
 path=Path(module.__file__)
 assert path.is_relative_to('/code/source')
 modules[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
Path('/qualification-output/module-origins.json').write_text(json.dumps(modules,indent=2)+'\n')
import pytest
raise SystemExit(pytest.main(['-q','--color=no','-x','/code/source/test/api/test_semantic_instruction_dispatch.py','--basetemp=/tmp/instruction-pytest','--junitxml=/qualification-output/junit.xml','-o','cache_dir=/tmp/pytest-cache','-o','junit_family=xunit1']))
""")
compile(entry.read_text(),str(entry),'exec')
name='instruction-metadata-smoke-20261012-'+attempt
image=subprocess.check_output(['docker','image','inspect','ipfs-supervisor-numerical-dev:20261011-r2','--format','{{.Id}}'],text=True).strip()
command=['docker','run','--name',name,'--rm','--network','none','--read-only','--tmpfs','/tmp:rw,nosuid,nodev,mode=1777,size=2147483648','--tmpfs','/home/supervisor/.local:rw,nosuid,nodev,uid=1000,gid=1000,mode=0700,size=67108864','--memory','12g','--cpus','5','--pids-limit','128','--cap-drop','ALL','--security-opt','no-new-privileges']
mounts=[(A/'extracted'/folder,'/code/'+folder,True) for folder in ('source','datasets','kit')]
mounts += [(entry,'/qualification/entry.py',True),(A/'archive/manifest.json','/qualification/archive-manifest.json',True),(results,'/qualification-output',False)]
for host,target,readonly in mounts:
 command += ['--mount',f'type=bind,src={host},dst={target}'+(',readonly' if readonly else '')]
env=dict(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='',IPFS_KIT_DISABLE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB='/tmp/pytest-seal.duckdb',PYTHONHASHSEED='0')
for key,value in env.items():command += ['--env',key+'='+value]
command += ['--entrypoint','/opt/ipfs-supervisor/venv/bin/python',image,'-I','-B','/qualification/entry.py']
manifest=json.loads((A/'archive/manifest.json').read_text())
tracked=[entry,A/'archive/manifest.json']+[source/p for p in files]
tracked += [A/'extracted'/row['path'] for row in manifest['files']]
before={str(p):sha(p) for p in tracked}
record=dict(command=command,image_id=image,environment=env,archive_sha256=manifest['archive_sha256'],source_and_test_file_count=len(tracked),all_source_hashes_match_archive_before=all(before[str(A/'extracted'/row['path'])]==row['sha256'] for row in manifest['files']),test_harness_sha256={p:sha(source/p) for p in files},entry_sha256=sha(entry),scope='Offline Docker smoke executes actual extracted archive production source; test harness copied separately, not included in archive.')
(run/'command.json').write_text(json.dumps(record,indent=2)+'\n')
start=time.monotonic();code=None;error=None
try:
 with (run/'stdout.txt').open('w') as out,(run/'stderr.txt').open('w') as err:
  code=subprocess.run(command,stdout=out,stderr=err,timeout=240).returncode
except subprocess.TimeoutExpired:error='docker_smoke_timeout_240_seconds'
finally:subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
after={str(p):sha(p) for p in tracked}
remaining=subprocess.check_output(['docker','container','ls','-a','--filter','name='+name,'--format','{{.Names}}'],text=True).strip()
counts={}
observations=[]
if (results/'junit.xml').is_file():
 tree=ET.parse(results/'junit.xml')
 counts={key:sum(int(s.attrib.get(key,0)) for s in tree.iter('testsuite')) for key in ('tests','errors','failures','skipped')}
 for case in tree.iter('testcase'):
  props={p.attrib['name']:p.attrib['value'] for p in case.findall('./properties/property')}
  if props:
   for key in ('native_complete_bytes','candidate_complete_bytes','selected_complete_bytes'):props[key]=int(props[key])
   props['live_token_savings_measured']=False
   if props['fallback_reason']=='None':props['fallback_reason']=None
   observations.append(dict(test=case.attrib['name'],**props))
qualification=dict(schema='instruction-metadata-offline-docker-smoke@1',returncode=code,error=error,seconds=time.monotonic()-start,counts=counts,source_hashes_unchanged=before==after,source_and_test_file_count=len(tracked),container_removed=remaining=='',image_id=image,archive_sha256=manifest['archive_sha256'],full_runtime_deployment_qualified=False,official_terminal_bench_score=None,live_provider_calls=0,model_downloads=0,training_steps=0,input_byte_observations=observations,results={p.name:sha(p) for p in results.iterdir() if p.is_file()})
(run/'qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
print(json.dumps(qualification,indent=2))
print((run/'stdout.txt').read_text()[-6000:]);print((run/'stderr.txt').read_text()[-4000:])
raise SystemExit(code if code is not None else 1)
