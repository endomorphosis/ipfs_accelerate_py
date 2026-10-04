"""Bounded host controls for profile deadlines and signed native policy inheritance."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET

B=Path(__file__).resolve().parent
import sys
label=sys.argv[1]
W=B.parent.parent
A=W/'.worktrees/ir-release-accelerate-20261002'
recipe=json.loads((W/'artifacts/ast-text-validation-native-20261004/source-02-command.json').read_text())
env=dict(recipe['environment_overrides'])
private=B/('private-'+label)
private.mkdir(mode=0o700,exist_ok=False)
env.update(IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=str(private/'seal.duckdb'),
           IPFS_DATASETS_PY_SEAL_KEY_STORE_PATH=str(private/'keys.db'),
           IPFS_DATASETS_RESOURCE_SCHEDULER_PATH=str(private/'scheduler.json'))
tests=sys.argv[2:]
owners=['ipfs_accelerate_py/agent_supervisor/runtime/local_completion_bridge.py',
        'ipfs_accelerate_py/agent_supervisor/task_sources/typed_state_owner.py',
        'ipfs_accelerate_py/agent_supervisor/entrypoints/admitted_benchmark_runtime.py',
        'test/integration/test_admitted_benchmark_runtime.py',
        'test/api/test_owner_completion_service_retirement.py']
def pins():return {p:hashlib.sha256((A/p).read_bytes()).hexdigest() for p in owners}
before=pins()
argv=[recipe['argv'][0],'-B','-m','pytest','-q','-o','cache_dir='+str(private/'pytest-cache'),'--basetemp='+str(private/'tmp'),*tests,
      '--junitxml='+str(B/(label+'.xml'))]
command=dict(argv=argv,cwd=str(A),environment_overrides=env,source_pins=before,
    hidden_verifier_used=False,production_scheduler_ledger_mutated=False,provider_calls=0)
(B/(label+'-command.json')).write_text(json.dumps(command,indent=2,sort_keys=True)+'\n')
started=time.monotonic()
with (B/(label+'.stdout')).open('x') as out,(B/(label+'.stderr')).open('x') as err:
    result=subprocess.run(argv,cwd=A,env={**os.environ,**env},stdout=out,stderr=err)
counts=dict(passed=0,failed=0,errors=0,skipped=0,executions=0)
for case in ET.parse(B/(label+'.xml')).getroot().iter('testcase'):
    counts['executions']+=1
    category='failed' if case.find('failure') is not None else 'errors' if case.find('error') is not None else 'skipped' if case.find('skipped') is not None else 'passed'
    counts[category]+=1
receipt=dict(returncode=result.returncode,seconds=time.monotonic()-started,
    counts=counts,source_pins_unchanged=before==pins())
(B/(label+'-exit.json')).write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt))
raise SystemExit(result.returncode)
