"""Run bounded isolated native integration controls, retaining exact source pins."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

B = Path(__file__).resolve().parent
W = B.parent.parent
A = W / '.worktrees/ir-release-accelerate-20261002'
D = W / '.worktrees/ir-pressure-attribution-datasets-20261004'
label = sys.argv[1]
assert label in {'before', 'after'}
recipe = json.loads((W / 'artifacts/ast-text-validation-native-20261004/source-02-command.json').read_text())
env = dict(recipe['environment_overrides'])
private = B / 'private-controls-state' / label
private.mkdir(parents=True, mode=0o700, exist_ok=False)
env.update(IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=str(private / 'seal.duckdb'),
    IPFS_DATASETS_PY_SEAL_KEY_STORE_PATH=str(private / 'keys.db'),
    IPFS_DATASETS_RESOURCE_SCHEDULER_PATH=str(private / 'scheduler.sqlite3'))
test = 'test/integration/test_admitted_benchmark_runtime.py'
if label == 'before':
    test += '::test_signed_run_owns_orchestration_path_without_legacy_account_scan'
argv = [recipe['argv'][0], '-B', '-m', 'pytest', '-q', '-o', 'addopts=', test,
    '--junitxml=' + str(B / (label + '.xml'))]
paths = [A / p for p in (
    'ipfs_accelerate_py/agent_supervisor/entrypoints/admitted_benchmark_runtime.py',
    'ipfs_accelerate_py/agent_supervisor/entrypoints/isolated_benchmark_runtime.py',
    'ipfs_accelerate_py/agent_supervisor/task_sources/board_control_plane.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/local_planning_admission.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/local_completion_bridge.py',
    'test/integration/test_admitted_benchmark_runtime.py',
    'benchmarks/agent_supervisor/container_coding/local_planning_qualification.py',
    'benchmarks/agent_supervisor/container_coding/native_quack_qualification.py')]
paths += [D / p for p in (
    'ipfs_datasets_py/optimizers/logic_theorem_optimizer/proof_resource_safety.py',
    'ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py')]


def pins():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


before = pins()
command = dict(argv=argv, cwd=str(A), environment_overrides=env, source_pins=before,
    hidden_verifier_used=False, production_scheduler_ledger_mutated=False, provider_calls=0)
with (B / (label + '-command.json')).open('x') as stream:
    stream.write(json.dumps(command, indent=2, sort_keys=True) + '\n')
start = time.monotonic()
with (B / (label + '.stdout')).open('x') as out, (B / (label + '.stderr')).open('x') as err:
    completed = subprocess.run(argv, cwd=A, env={**os.environ, **env}, stdout=out, stderr=err)
counts = dict(passed=0, failed=0, errors=0, skipped=0, executions=0)
xml = B / (label + '.xml')
if xml.exists():
    for case in ET.parse(xml).getroot().iter('testcase'):
        counts['executions'] += 1
        category = 'failed' if case.find('failure') is not None else 'errors' if case.find('error') is not None else 'skipped' if case.find('skipped') is not None else 'passed'
        counts[category] += 1
receipt = dict(returncode=completed.returncode, seconds=time.monotonic() - start,
    counts=counts, source_pins_unchanged=before == pins())
(B / (label + '-exit.json')).write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
print(json.dumps(receipt))
raise SystemExit(completed.returncode)
