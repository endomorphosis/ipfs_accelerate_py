"""Rerun the unchanged 237-case recipe after consumer wiring/diagnostic fixes."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET

B = Path(__file__).resolve().parent
D = Path('/home/barberb/lift_coding/.worktrees/ir-pressure-attribution-datasets-20261004')
A = Path('/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002')
expected_heads = {'datasets': 'da853fa837b89a4ef01fa981127cfc446158371e',
                  'accelerate': 'd2de765bc5307cd728a9c9099759e812ee092381'}
def heads():
    return {name: subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
            for name, root in [('datasets', D), ('accelerate', A)]}

assert heads() == expected_heads
label = 'focused-03'
old = json.loads((B / 'focused-01-command.json').read_bytes())
keys = list(old['source_sha256'])
sha = lambda raw: hashlib.sha256(raw).hexdigest()
def pins():
    return {name: sha((D / name).read_bytes()) for name in keys}

private = B / 'private-controls-state' / label
private.mkdir(mode=0o700, parents=True, exist_ok=False)
source_dir = B / (label + '-sources')
source_dir.mkdir(exist_ok=False)
for name in keys:
    target = source_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((D / name).read_bytes())
argv = list(old['argv'])
argv[-1] = '--junitxml=' + str(B / (label + '.xml'))
overrides = {**old['environment_overrides'],
    'IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB': str(private / 'seal.duckdb'),
    'IPFS_DATASETS_PY_SEAL_KEY_STORE_PATH': str(private / 'keys.db'),
    'IPFS_DATASETS_RESOURCE_SCHEDULER_PATH': str(private / 'scheduler.sqlite3'),
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'CUDA_VISIBLE_DEVICES': '',
    'TOKENIZERS_PARALLELISM': 'false', 'NUMEXPR_NUM_THREADS': '1', 'NUMEXPR_MAX_THREADS': '1'}
initial = pins()
command = {'argv': argv, 'cwd': str(D), 'environment_overrides': overrides,
           'source_sha256': initial, 'source_heads': expected_heads, 'timeout_seconds': 180,
           'controller_sha256': sha(Path(__file__).read_bytes()),
           'previous_recipe_sha256': sha((B / 'focused-01-command.json').read_bytes()),
           'hidden_verifier_used': False, 'provider_calls': 0, 'production_scheduler_ledger_mutated': False}
(B / (label + '-command.json')).write_text(json.dumps(command, indent=2) + '\n')
started = time.monotonic()
with (B / (label + '-stdout.txt')).open('w') as out, (B / (label + '-stderr.txt')).open('w') as err:
    result = subprocess.run(argv, cwd=D, env={**os.environ, **overrides}, stdout=out, stderr=err, timeout=180)
final = pins()
exit = {'returncode': result.returncode, 'wall_seconds': time.monotonic() - started,
        'source_pins_after': final, 'source_pins_unchanged': initial == final,
        'source_heads_after': heads(), 'source_heads_unchanged': heads() == expected_heads}
(B / (label + '-exit.json')).write_text(json.dumps(exit, indent=2) + '\n')
cases = list(ET.parse(B / (label + '.xml')).getroot().iter('testcase'))
prior_cases = list(ET.parse(B / 'focused-01.xml').getroot().iter('testcase'))
identities = lambda items: sorted((case.get('classname'), case.get('name')) for case in items)
report = {'schema': 'ast-printability-reconciled-controls@1', 'source_heads': expected_heads,
          'tests': len(cases), 'failures': sum(c.find('failure') is not None for c in cases),
          'errors': sum(c.find('error') is not None for c in cases),
          'skips': sum(c.find('skipped') is not None for c in cases),
          'same_test_population_as_focused_01': identities(cases) == identities(prior_cases),
          'source_pins_unchanged': initial == final, 'source_heads_unchanged': exit['source_heads_unchanged'],
          'wall_seconds': exit['wall_seconds'], 'source_snapshots': str(source_dir),
          'previous_sealed_package_mutated': False, 'new_performance_measurement': False,
          'private_stores_exported': False,
          'records': {name: sha((B / name).read_bytes()) for name in [
              label + '-command.json', label + '-exit.json', label + '.xml',
              label + '-stdout.txt', label + '-stderr.txt']}}
(B / (label + '-reconciliation.json')).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
assert result.returncode == 0 and len(cases) == 237 and not any(report[key] for key in ['errors','failures','skips'])
assert report['same_test_population_as_focused_01'] and report['source_pins_unchanged'] and report['source_heads_unchanged']
