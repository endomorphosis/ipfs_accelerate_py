import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

P = Path('/home/barberb/lift_coding/.worktrees/supervisor-comparison-contracts-20261011')
R = Path(__file__).resolve().parent / sys.argv[1]
R.mkdir(parents=True, exist_ok=False)
previous = json.loads(Path('/home/barberb/lift_coding/artifacts/source-view-worker-20261012/final-tests-01/command.json').read_text())
environment = dict(previous['environment_overrides'])
environment['IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB'] = str(R / 'seal.duckdb')
environment['IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR'] = str(R / 'orchestration')
files = [value for value in previous['command'] if value.endswith('.py')]
files += [
    'test/api/test_worker_source_working_views.py',
    'test/api/test_worker_source_working_broker.py',
    'test/api/test_worker_source_working_client.py',
    'test/api/test_source_lookup_worker_channel.py',
    'test/api/test_source_lookup_worker_router.py',
    'test/api/test_worker_source_views.py',
    'test/api/test_worker_source_broker.py',
    'test/api/test_worker_source_client.py',
    'test/api/test_reversible_source_view.py',
    'test/api/test_semantic_router_integration.py',
    'test/api/test_terminal_doctor_dispatch.py',
    'benchmarks/agent_supervisor/container_coding/test_source_view_worker_wiring.py',
    'benchmarks/agent_supervisor/container_coding/test_terminal_source_view_policy.py',
    'benchmarks/agent_supervisor/container_coding/test_terminal_source_view_observation.py',
    'benchmarks/agent_supervisor/container_coding/test_terminal_context_audit.py',
    'benchmarks/agent_supervisor/container_coding/test_terminal_semantic_metadata_policy.py',
    'benchmarks/agent_supervisor/container_coding/test_benchmark_controls.py',
]
files = list(dict.fromkeys(files))
command = [previous['command'][0], '-B', '-m', 'pytest', '-q', '--color=no', *files,
    '--junitxml=' + str(R / 'junit.xml'), '-o', 'cache_dir=' + str(R / 'pytest-cache'),
    '-o', 'log_cli=false', '--tb=short']
paths = set(files)
for git_args in (['diff', '--name-only', '--', '*.py'], ['ls-files', '--others', '--exclude-standard', '--', '*.py']):
    paths.update(subprocess.check_output(['git', *git_args], cwd=P, text=True).splitlines())
paths = {P / path for path in paths}
paths.update(Path(path) for path in previous['source_before'] if '/ipfs_datasets_py/' in path)
paths.add(Path(__file__).resolve())
for path in previous['source_before']:
    selected = Path(path)
    if selected.exists(): paths.add(selected)
for name in ('source_lookup_worker_client.py', 'source_lookup_worker_broker.py'):
    paths.add(P / 'ipfs_accelerate_py/agent_supervisor/runtime' / name)
def hashes():
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
before = hashes()
(R / 'command.json').write_text(json.dumps(dict(command=command, cwd=str(P), environment_overrides=environment,
    source_before=before, scope='Offline source/capsule/owner/router and benchmark configuration regression; authored providers only'), indent=2) + '\n')
started = time.monotonic()
with (R / 'pytest.log').open('w') as stream:
    result = subprocess.run(command, cwd=P, env={**os.environ, **environment}, stdout=stream, stderr=subprocess.STDOUT)
after = hashes()
suites = ET.parse(R / 'junit.xml').getroot()
counts = {name: sum(int(suite.attrib.get(name, 0)) for suite in suites.iter('testsuite'))
          for name in ('tests', 'failures', 'errors', 'skipped')}
qualification = dict(exit_code=result.returncode, seconds=time.monotonic()-started,
    counts=counts, source_unchanged=before == after, source_after=after,
    provider_calls=0, provider_tokens=None, full_supervisor_start=False,
    official_verifier_executed=False, total_token_savings_measured=False)
(R / 'qualification.json').write_text(json.dumps(qualification, indent=2) + '\n')
print(json.dumps({key: value for key, value in qualification.items() if key != 'source_after'}, indent=2))
if result.returncode:
    print((R / 'pytest.log').read_text()[-14000:])
raise SystemExit(result.returncode or int(before != after))
