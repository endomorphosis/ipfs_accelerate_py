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
R = ROOT/'artifacts/supervisor-ranked-successor-20261012'
run = R/sys.argv[1]
run.mkdir(parents=True,exist_ok=False)
profile = os.environ.get('TOKEN_EFFICIENCY_TEST_PROFILE', 'native')
prior_path = ('artifacts/supervisor-numerical-runtime-20261011/deployment-final-01/command.json'
    if profile == 'harbor' else 'artifacts/supervisor-formalization-audit-20261011/postmerge-comparison/command.json')
prior = json.loads((ROOT/prior_path).read_text())
env = prior['environment_overrides'].copy()
env['PYTHONPATH'] = env['PYTHONPATH'].replace(str(D), str(ROOT/'.worktrees/ranked-current-datasets-20261012'))
env['NUMERICAL_SELECTION384_CHECKPOINT'] = str(R.parent/'supervisor-numerical-training-20261011/release/checkpoint.json')
env['IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB'] = str(run/'seal.duckdb')
paths = [str(p.relative_to(P)) for p in (P/'ipfs_accelerate_py/agent_supervisor/runtime').glob('semantic*py')]
paths += ['ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py',
          'benchmarks/agent_supervisor/container_coding/terminal_context_audit.py',
          'benchmarks/agent_supervisor/container_coding/benchmark_comparison.py',
          'benchmarks/agent_supervisor/container_coding/terminal_initial_context.py',
          'benchmarks/agent_supervisor/container_coding/terminal_semantic_metadata_policy.py']
paths += subprocess.check_output(['git','diff','HEAD','--name-only','--','*.py'], cwd=P, text=True).splitlines()
paths += subprocess.check_output(['git','ls-files','--others','--exclude-standard','--','*.py'], cwd=P, text=True).splitlines()
paths += [arg for arg in sys.argv[2:] if (P/arg).is_file()]
paths += ['benchmarks/agent_supervisor/container_coding/terminal_live_metadata_census.py', 'ipfs_accelerate_py/agent_supervisor/runtime/formal_status_census.py', 'ipfs_accelerate_py/agent_supervisor/runtime/doctor_header_cache_bridge.py']
tracked = [P/p for p in sorted(set(paths))]
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
