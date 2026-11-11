"""Copy only closed qualification facts and authored audit scripts into docs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

W = Path('/home/barberb/lift_coding/.worktrees/supervisor-tool-event-coverage-20261015')
B = Path('/home/barberb/lift_coding/artifacts/grok-pruning-profiles-20261015')
E = W / 'docs/agent_supervisor/evidence/grok-pruning-20261015'
labels = ['core-router-final03', 'outer-final01', 'subprocess-final01', 'metadata-doctor-final01']

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

sources, records = {}, []
for label in labels:
    result = json.loads((B / (label + '-result.json')).read_text())
    assert result['exit_code'] == 0 and set(result['counts']) == {'passed'}
    assert all(result[key] is True for key in ('source_pins_unchanged', 'dependencies_unchanged', 'distribution_pins_unchanged'))
    assert result['live_provider_calls'] == 0
    pins = json.loads((B / (label + '-source-pins.json')).read_text())['files']
    for name, expected in pins.items():
        assert digest(W / name) == expected, name
        assert name not in sources or sources[name] == expected, name
        sources[name] = expected
    records.append({'label': label, 'counts': result['counts'], 'seconds': result['seconds'],
                    'result_sha256': digest(B / (label + '-result.json')),
                    'source_pins_sha256': digest(B / (label + '-source-pins.json'))})

inventory = json.loads((B / 'closed-audit-files-inventory-draft01.json').read_text())
names = ['closed-audit-files-inventory-draft01.json', 'run-tests.py', 'assemble-evidence.py',
         'authored-e2e/test_pruning_preflight.py']
for row in inventory['files']:
    source = B / row['path']
    assert source.stat().st_size == row['bytes'] and digest(source) == row['sha256']
    names.append(row['path'])
for label in labels:
    names.extend(label + suffix for suffix in ('-result.json', '-source-pins.json'))
for label in ('core-router-dev01', 'core-router-final01', 'core-router-final02'):
    names.extend(label + suffix for suffix in ('-result.json', '-source-pins.json'))
assert not E.exists(), 'Evidence destination already exists'
E.mkdir(parents=True)
for name in names:
    target = E / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(B / name, target)

free = shutil.disk_usage(W).free
qualification = {
    'schema': 'grok-pruning-offline-qualification@1',
    'observed_at_utc': datetime.now(timezone.utc).isoformat(),
    'runtime_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip(),
    'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=W, text=True).strip(),
    'source_pins_current': True, 'source_files': sources,
    'batches': records, 'passed': sum(row['counts']['passed'] for row in records),
    'tests_provider_calls': 0,
    'native_authored_inspection_calls': 5, 'inspection_containers_cleaned': 5,
    'live_benchmark_launched': False, 'new_runtime_archive_built': False,
    'live_admission': {'status': 'blocked' if free < 32 * 1024**3 else 'not_attempted',
        'reason': 'disk_floor' if free < 32 * 1024**3 else 'qualification_only',
        'observed_free_bytes': free, 'minimum_free_bytes': 32 * 1024**3,
        'admission_floor_changed': False, 'retained_artifacts_deleted': False},
    'effective_settings_verified': False, 'config_file_consumption_verified': False,
    'token_savings_established': False, 'benchmark_score': None,
    'corrected_findings': [
        'Reserved structured preflight receipt was stringified by CLI metadata; validated preservation and redaction added.',
        'Runner success now requires selected preflight before acceptance; failures retain observed usage.',
        'Configuration comparison eligibility is independent of outcome equality.',
        'Authored timeout fixture now leaves absent completion evidence unknown.',
        'Authored native observation fixtures now include the separate generic successful exit field.'
    ],
    'remaining': ['Build and qualify updated runtime archive after admission permits it.',
                  'Run matched on/off tasks with fixed controls and official correctness measurement.',
                  'Native effective pruning and config consumption evidence is unavailable in pinned inspection.'],
}
write(E / 'qualification-final01.json', qualification)
manifest = {'schema': 'grok-pruning-closed-evidence-manifest@1',
    'raw_prompts_included': False, 'raw_native_reports_included': False,
    'credentials_included': False, 'private_owner_artifacts_included': False,
    'files': [{'path': str(path.relative_to(E)), 'bytes': path.stat().st_size, 'sha256': digest(path)}
              for path in sorted(E.rglob('*')) if path.is_file()]}
write(E / 'manifest.json', manifest)
print(json.dumps({'passed': qualification['passed'], 'files': len(manifest['files']),
                  'qualification_sha256': digest(E / 'qualification-final01.json'),
                  'live_admission': qualification['live_admission']}))
