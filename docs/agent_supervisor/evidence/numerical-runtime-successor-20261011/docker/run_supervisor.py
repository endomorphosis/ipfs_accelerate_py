import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/barberb/lift_coding')
P = ROOT/'.worktrees/supervisor-comparison-contracts-20261011'
D = ROOT/'.worktrees/formalization-datasets-20261011'
R = ROOT/'artifacts/supervisor-numerical-runtime-20261011'
run = R/sys.argv[1]
run.mkdir(parents=True,exist_ok=False)
prior = json.loads((ROOT/'artifacts/supervisor-formalization-audit-20261011/postmerge-comparison/command.json').read_text())
env = prior['environment_overrides']
env['NUMERICAL_SELECTION384_CHECKPOINT'] = str(R.parent/'supervisor-numerical-training-20261011/release/checkpoint.json')
env['IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB'] = str(run/'seal.duckdb')
paths = [
    'benchmarks/agent_supervisor/container_coding/terminal_numerical_refresh_accounting.py',
    'benchmarks/agent_supervisor/container_coding/test_terminal_numerical_refresh_accounting.py',
    'test/integration/test_admitted_learned_numerical_successor.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/published_numerical_selection.py',
    'test/api/test_published_numerical_selection.py',
    'benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py',
    'benchmarks/agent_supervisor/container_coding/terminal_numerical_selection_deployment.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/numerical_selection_config.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/numerical_selection_context.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/task_context_bundle.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/source_formalization_context.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/published_task_context.py',
    'ipfs_accelerate_py/agent_supervisor/entrypoints/admitted_benchmark_runtime.py',
    'benchmarks/agent_supervisor/container_coding/terminal_initial_context.py',
    'benchmarks/agent_supervisor/container_coding/terminal_indexed_preparation.py',
    'benchmarks/agent_supervisor/container_coding/terminal_context_audit.py',
    'benchmarks/agent_supervisor/container_coding/benchmark_comparison.py',
    'benchmarks/agent_supervisor/container_coding/test_terminal_numerical_selection_integration.py',
    'test/api/test_numerical_selection_config.py',
]
tracked = [P/p for p in paths] + [D/'ipfs_datasets_py/logic/software_contracts'/p for p in (
    'codebase_numerical_selection_384.py','codebase_numerical_selection_384_worker.py')]
tracked.append(D/'ipfs_datasets_py/logic/backends/codebase_process.py')
def hashes():
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
cmd = [prior['command'][0],'-B','-m','pytest','-q','--color=no',*sys.argv[2:],
    '--junitxml='+str(run/'junit.xml'),'-o','cache_dir='+str(run/'pytest-cache')]
before = hashes()
meta = dict(command=cmd,cwd=str(P),environment_overrides=env,source_before=before,
    scope='Cached local experimental numerical checkpoint; no training, downloads, or remote providers. Authored coding/planning provider responses are explicit in tests.')
(run/'command.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
start=time.monotonic()
with (run/'pytest.log').open('w') as stream:
    result=subprocess.run(cmd,cwd=P,env={**os.environ,**env},stdout=stream,stderr=subprocess.STDOUT)
after=hashes()
report=dict(exit_code=result.returncode,seconds=time.monotonic()-start,source_after=after,source_unchanged=before==after)
(run/'qualification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='source_after'}))
raise SystemExit(result.returncode)
