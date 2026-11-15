"""Package reviewed offline qualification; never open provider/owner archives."""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

W = Path('/home/barberb/lift_coding/.worktrees/supervisor-public-validation-20261014')
A = Path('/home/barberb/lift_coding/artifacts/doctor-public-validation-20261014')
B = Path('/home/barberb/lift_coding/artifacts/public-validation-continuation-20261014')
E = W / 'docs/agent_supervisor/evidence/doctor-public-validation-20261014'
FINAL = [(A, 'public-final01'), (A, 'public-wiring-final'), (B, 'continuation-green01'),
         (B, 'continuation-native-green01')]
NATIVE_TEST = 'test/api/test_public_validation_continuation.py'
NATIVE_CLASS = 'api.test_public_validation_continuation'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

copies = []

def copy(source, name=None):
    target = E / (name or source.name)
    raw = source.read_bytes()
    target.write_bytes(raw)
    copies.append(dict(source=str(source), destination=target.name, bytes=len(raw), sha256=sha(source)))

all_pins, tests, runs, test_pin_replacements = {}, set(), [], []
for root, label in FINAL:
    result = read(root / (label + '-result.json'))
    legacy_subset = label == 'continuation-green01'
    assert result['exit_code'] == (1 if legacy_subset else 0) and result['live_provider_calls'] == 0
    assert all(result[k] is True for k in ('source_pins_unchanged', 'dependencies_unchanged', 'distribution_pins_unchanged'))
    assert set(result['counts']) == ({'passed', 'failed'} if legacy_subset else {'passed'})
    if legacy_subset:
        assert result['counts']['failed'] == 1
    pins = read(root / (label + '-source-pins.json'))
    for name, digest in pins['files'].items():
        current = sha(W / name)
        if current != digest:
            assert name == NATIVE_TEST and label != 'continuation-native-green01'
            test_pin_replacements.append(dict(run=label, path=name, observed_sha256=digest, current_sha256=current,
                reason='test-only ordering correction; whole native module requalified separately'))
        assert name not in all_pins or all_pins[name] == current
        all_pins[name] = current
    cases = list(ET.parse(root / (label + '.xml')).getroot().iter('testcase'))
    assert len(cases) == sum(result['counts'].values())
    selected = [c for c in cases if not legacy_subset or c.attrib['classname'] != NATIVE_CLASS]
    if legacy_subset:
        excluded = [c for c in cases if c.attrib['classname'] == NATIVE_CLASS]
        assert len(excluded) == 3 and sum(c.find('failure') is not None for c in excluded) == 1
    if label == 'continuation-native-green01':
        assert len(selected) == 3 and all(c.attrib['classname'] == NATIVE_CLASS for c in selected)
    for case in selected:
        assert not any(case.find(kind) is not None for kind in ('failure', 'error', 'skipped'))
        identity = (case.attrib['classname'], case.attrib['name'])
        assert identity not in tests
        tests.add(identity)
    runs.append(dict(label=label, original_counts=result['counts'], qualified_passed=len(selected),
        excluded_module=NATIVE_CLASS if legacy_subset else None, seconds=result['seconds'], base_commit=pins['base_commit'],
        result=label+'-result.json', source_pins=label+'-source-pins.json', junit=label+'.xml'))

reader = read(A / 'reader-public-qualification.json')
assert reader['passed'] and reader['failures'] == reader['errors'] == reader['skipped'] == 0
assert reader['provider_calls'] == 0 and reader['runtime_modules_imported'] == []
assert all(reader[k] for k in ('source_pins_unchanged', 'artifact_pins_unchanged',
    'retained_c_projection_identical', 'old_c_evidence_unchanged'))
for source, binding in reader['source_files'].items():
    assert sha(Path(source)) == binding['sha256']
for source, binding in reader['artifact_files'].items():
    assert sha(Path(source)) == binding['sha256']
review = read(A / 'independent-review.json')

developmental = []
for root in (A, B):
    for result_path in sorted(root.glob('*-result.json')):
        label = result_path.name.removesuffix('-result.json')
        result = read(result_path)
        copy(result_path)
        copy(root / (label + '-source-pins.json'))
        final = (root, label) in FINAL
        if final or result['exit_code']:
            copy(root / (label + '.xml'))
            copy(root / (label + '.log'))
        if not final:
            developmental.append(dict(label=label, counts=result['counts'], source_pins_unchanged=result['source_pins_unchanged'],
                qualification=False, reason='superseded developmental run; excluded from final count'))
copy(A / 'run-tests.py', 'run-profile-tests.py')
copy(B / 'run-tests.py', 'run-continuation-tests.py')
for source in reader['artifact_files']:
    copy(Path(source))
for name in ('reader-public-qualification.json', 'reader-public-tests-final.log',
             'retained-c-public-reader-replay.json', 'independent-review.json'):
    copy(A / name)
copy(Path(__file__))

qualification = dict(schema='doctor-public-validation-qualification@1',
    base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip(),
    worktree=str(W), repository_tests=len(tests), artifact_reader_tests=reader['tests_run'],
    total_distinct_tests=len(tests)+reader['tests_run'], runs=runs, source_files=all_pins,
    source_files_match_current=True, test_pin_replacements=test_pin_replacements,
    live_provider_calls=0, fresh_live_benchmark_run=False,
    provider_tokens_saved=None, benchmark_reward=None,
    authored_boundaries=['provider responses', 'host UID/isolation boundary',
        'native continuation public-check outputs', 'timing classifier clocks'],
    actual_paths_exercised=['generated numerical checker subprocesses', 'signed native task admission',
        'two separate coding subprocesses', 'source broker', 'Doctor handoff',
        'candidate validation custody', 'historical input replay', 'attempt accounting'],
    developmental_runs=developmental,
    limitations=['empirical public checks do not prove correctness or speed on all inputs',
        'explicit largest-eigenval public profile only', 'no new live provider or container qualification',
        'native internal model calls, total tokens and dollars are not bounded by owner attempt policy',
        'previous missing coding usage remains unknown'])
(E / 'qualification.json').write_text(json.dumps(qualification, indent=2, sort_keys=True)+'\n')
(E / 'copy-provenance.json').write_text(json.dumps(dict(schema='reviewed-evidence-copy@1', files=copies), indent=2, sort_keys=True)+'\n')
manifest = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(E.iterdir()) if p.is_file() and p.name != 'manifest.json'}
(E / 'manifest.json').write_text(json.dumps(dict(schema='evidence-manifest@1', files=manifest), indent=2, sort_keys=True)+'\n')
print(json.dumps(dict(total_distinct_tests=qualification['total_distinct_tests'], repository_tests=len(tests),
    reader_tests=reader['tests_run'], source_files=len(all_pins), evidence_files=len(manifest))))
