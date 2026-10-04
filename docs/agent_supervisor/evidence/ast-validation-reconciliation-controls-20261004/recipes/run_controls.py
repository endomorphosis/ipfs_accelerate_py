"""Fresh isolated regression runs for the semantic serialization generation."""
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
P = W / 'artifacts/pressure-replay-native-20261004'
A = W / '.worktrees/ir-release-accelerate-20261002'
D = W / '.worktrees/ir-pressure-attribution-datasets-20261004'
kind, label = sys.argv[1:]
assert kind in ('source', 'integration', 'native', 'focused')
assert label.replace('-', '').isalnum()
recipe_name = 'controls-03-command.json' if kind == 'source' else kind + '-command.json'
recipe = json.loads(((W / 'artifacts/cold-publication-projection-20261004/focused-01-command.json') if kind == 'focused' else (P / recipe_name)).read_text())
root = A if kind == 'source' else D
env = recipe['environment_overrides'].copy()
private = B / 'private-controls-state' / label
private.mkdir(parents=True, exist_ok=False)
for key, name in (
    ('IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB', 'seal.duckdb'),
    ('IPFS_DATASETS_PY_SEAL_KEY_STORE_PATH', 'keys.db'),
    ('IPFS_DATASETS_RESOURCE_SCHEDULER_PATH', 'scheduler.sqlite3'),
):
    env[key] = str(private / name)
targets = [arg for arg in recipe['argv'] if arg.endswith('.py')]
if kind == 'source' and label in ('source-03', 'source-04'):
    target = 'test/integration/test_admitted_benchmark_runtime.py'
    assert target not in targets
    targets.append(target)
if kind == 'integration':
    # This exact module already passed in the current producer's focused run.
    targets.remove('tests/unit/logic/software_contracts/test_codebase_manifest_memo.py')
    # These current-owner controls passed in the AST focused group.
    targets.remove('tests/unit/logic/software_contracts/test_duckdb_ast_store.py')
    targets.remove('tests/unit/logic/software_contracts/test_duckdb_ast_store_persistence.py')
    targets += ['tests/integration/logic/software_contracts/test_codebase_observation_reconstruction.py']
argv = [recipe['argv'][0], '-B', '-m', 'pytest', '-q', '-o',
        'cache_dir=' + str(private / 'pytest-cache'), *targets,
        '--junitxml=' + str(B / (label + '.xml'))]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

owners = {root / p for p in targets}
for path in json.loads((P / 'frozen-production-pins.json').read_text()):
    prefix, name = path.split('/', 1)
    owners.add((A if prefix == 'source' else D) / name)
owners.add(D / 'ipfs_datasets_py/logic/software_contracts/semantic_index/models.py')
owners.add(A / 'ipfs_accelerate_py/agent_supervisor/entrypoints/admitted_benchmark_runtime.py')
pins = {str(p): sha(p) for p in sorted(owners)}
def write(name, value):
    with (B / name).open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')

write(label + '-command.json', dict(argv=argv, cwd=str(root),
    environment_overrides=env, source_pins=pins, provider_calls=0,
    hidden_verifier_used=False, production_scheduler_ledger_mutated=False))
started = time.monotonic()
with (B / (label + '.stdout')).open('x') as out, (B / (label + '.stderr')).open('x') as err:
    run = subprocess.run(argv, cwd=root, env={**os.environ, **env},
                         stdout=out, stderr=err, timeout=600)
tree = ET.parse(B / (label + '.xml'))
result = dict(returncode=run.returncode, seconds=time.monotonic() - started,
    tests=len(tree.findall('.//testcase')), failed=len(tree.findall('.//failure')),
    errors=len(tree.findall('.//error')), skipped=len(tree.findall('.//skipped')),
    source_pins_unchanged=all(sha(Path(p)) == v for p, v in pins.items()))
write(label + '-exit.json', result)
print(json.dumps(result), flush=True)
assert result['source_pins_unchanged'] and not any(result[k] for k in ('returncode', 'failed', 'errors', 'skipped'))
