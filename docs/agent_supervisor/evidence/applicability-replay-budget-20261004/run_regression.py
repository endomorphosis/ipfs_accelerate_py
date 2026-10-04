import hashlib,json,os,subprocess,time,xml.etree.ElementTree as ET
from pathlib import Path
B=Path(__file__).resolve().parent
A=Path('/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002')
D=Path('/home/barberb/lift_coding/.worktrees/ir-pressure-attribution-datasets-20261004')
label='regression-01'
paths=[
 'benchmarks/agent_supervisor/container_coding/test_extended_resource_profile.py',
 'benchmarks/agent_supervisor/container_coding/test_doctor_dispatch_budget.py',
 'test/api/test_header_intent_applicability.py',
 'test/api/test_agent_supervisor_local_planning_admission.py',
 'test/api/test_intent_symbolic_planning.py',
 'test/api/test_intent_symbolic_planning_ordered.py',
 'benchmarks/agent_supervisor/container_coding/test_terminal_intent_requirement_planning.py',
]
prior=json.loads(Path('/home/barberb/lift_coding/artifacts/header-applicability-replay-budget-20261004/fast-03-command.json').read_text())['source_pins']
pinpaths={key:(A if key.startswith('source/') else D)/key.split('/',1)[1] for key in prior}
for name in paths:pinpaths['source/'+name]=A/name
for name in ['benchmarks/agent_supervisor/container_coding/benchmark_resource_profile.py','ipfs_accelerate_py/agent_supervisor/runtime/local_completion_bridge.py']:
 pinpaths['source/'+name]=A/name
def pins():return {key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in sorted(pinpaths.items())}
private=Path('/tmp/hrb-reg-01');private.mkdir(mode=0o700,exist_ok=False)
env={
 'PYTHONPATH':str(A)+':'+str(D)+':/home/barberb/lift_coding/.venvs/terminal-bench-harbor/lib/python3.12/site-packages',
 'PYTHONDONTWRITEBYTECODE':'1','PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1',
 'IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB':str(private/'seal.duckdb'),
 'IPFS_DATASETS_PY_SEAL_KEY_STORE_PATH':str(private/'keys.db'),
 'IPFS_DATASETS_RESOURCE_SCHEDULER_PATH':str(private/'scheduler.json'),
 'IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR':str(private/'orchestration'),
 'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
 'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
}
argv=['/home/barberb/.local/bin/python','-B','-m','pytest','-q','-p','no:cacheprovider','--basetemp='+str(private/'tests'),'--junitxml='+str(B/(label+'.xml')),*paths]
before=pins()
command=dict(argv=argv,cwd=str(A),environment_overrides=env,source_pins=before,
 source_revisions={key:subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip() for key,root in [('source',A),('datasets',D)]},
 harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),selected_source_pins_complete_repository=False)
(B/(label+'-command.json')).write_text(json.dumps(command,sort_keys=True,indent=2)+'\n')
start=time.monotonic()
with (B/(label+'.log')).open('x') as log:r=subprocess.run(argv,cwd=A,env={**os.environ,**env},stdout=log,stderr=subprocess.STDOUT)
after=pins();root=ET.parse(B/(label+'.xml')).getroot()
counts={key:sum(int(s.get(key,'0')) for s in root.iter('testsuite')) for key in ['tests','failures','errors','skipped']}
receipt=dict(returncode=r.returncode,seconds=time.monotonic()-start,source_pins_unchanged=before==after,after_pins=after,xml_counts=counts)
(B/(label+'-exit.json')).write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
print(json.dumps({key:value for key,value in receipt.items() if key!='after_pins'}))
raise SystemExit(r.returncode)
