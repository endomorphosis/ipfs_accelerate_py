"""Finite custody correspondence controls under root's genuine charged owner.

These tests use necessary-condition dictionaries and refuse fake objects before
native imports. They do not qualify actual custody, Docker cleanup or providers.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import re
import stat
import sys
import tempfile

PHASE = Path('/home/barberb/lift_coding/artifacts/legal-ir-handoff-native-20261017-01')
SOURCE = PHASE / 'ordinary-custody-hold-v01'
ATTEMPT = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs/legal-ir-ordinary-custody-controls-20261017-r01')
TEST = 'test_custody_hold_memory.py'
EXPECTED_FILES = {'README.md', 'ordinary_custody_hold.py', TEST}
NATIVE = Path('/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets/ipfs_datasets_py/optimizers/logic_theorem_optimizer')
EXPECTED_INPUTS = {str(NATIVE / (name + '.py')) for name in (
    'parented_phase_resources', 'autoencoder_daemon_resources',
    'existing_resource_scheduler', 'resource_scheduler')}


def need(value, reason):
    if not value:
        raise ValueError(reason)


def stable(path, maximum=8 * 1024 * 1024):
    path = Path(path)
    need(path.is_absolute() and os.path.abspath(path) == str(path) and '..' not in path.parts
         and not any(p.is_symlink() for p in (path, *path.parents)), 'canonical unaliased source required')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid in {0, os.getuid()}
             and before.st_mode & 0o002 == 0 and 0 < before.st_size <= maximum,
             'bounded owned trusted source required')
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns,
                          s.st_mode, s.st_uid, s.st_gid, s.st_nlink)
    need(len(raw) == before.st_size and identity(before) == identity(after) == identity(path.lstat()),
         'frozen source changed while reading')
    return raw


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate manifest JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def main():
    need(len(sys.argv) == 2 and re.fullmatch('[0-9a-f]{64}', sys.argv[1]),
         'usage: qualify_custody_controls.py EXPECTED_HELPER_MANIFEST_SHA256')
    expected = sys.argv[1]
    attempt = Path(os.environ['TMPDIR']).absolute()
    info = attempt.lstat()
    need(attempt == ATTEMPT and stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
         and info.st_mode & 0o077 == 0 and not any(p.is_symlink() for p in (attempt, *attempt.parents)),
         'original exact private charged attempt required')
    manifest_raw = stable(SOURCE / 'SOURCE_MANIFEST.json', 1024 * 1024)
    need(hashlib.sha256(manifest_raw).hexdigest() == expected, 'frozen custody manifest differs')
    manifest = strict_json(manifest_raw)
    need(manifest.get('schema') == 'legal-ir-ordinary-custody-hold-source@1'
         and type(manifest.get('controlled_testcases')) is int and manifest['controlled_testcases'] == 35,
         'exact 35-case frozen custody source required')
    rows, input_rows = manifest.get('files'), manifest.get('input_pins')
    need(type(rows) is list and len(rows) == 3 and type(input_rows) is list and len(input_rows) == 4,
         'complete finite custody source and native input closure required')
    pinned = {}
    for row in rows:
        need(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'closed source manifest row required')
        spelling = row['path']
        need(type(spelling) is str, 'source relative spelling required')
        pure = PurePosixPath(spelling)
        need(not pure.is_absolute() and '..' not in pure.parts and pure.as_posix() == spelling
             and spelling in EXPECTED_FILES, 'exact custody source row required')
        need(spelling not in pinned and type(row['bytes']) is int and 0 < row['bytes'] <= 8 * 1024 * 1024
             and type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}', row['sha256']),
             'unique bounded custody source row required')
        pinned[spelling] = row
    need(set(pinned) == EXPECTED_FILES, 'all frozen custody source rows required')
    inputs = {}
    for row in input_rows:
        need(type(row) is dict and set(row) == {'path', 'bytes', 'sha256', 'role'}
             and row['role'] == 'held_native_public_API_source', 'closed exact native input row required')
        spelling = row['path']
        need(type(spelling) is str and spelling in EXPECTED_INPUTS and spelling not in inputs
             and type(row['bytes']) is int and 0 < row['bytes'] <= 8 * 1024 * 1024
             and type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}', row['sha256']),
             'unique bounded exact native input required')
        inputs[spelling] = row
    need(set(inputs) == EXPECTED_INPUTS, 'all held native public API pins required')
    self_raw = stable(Path(__file__).absolute())

    def guard():
        need(stable(SOURCE / 'SOURCE_MANIFEST.json', 1024 * 1024) == manifest_raw
             and stable(Path(__file__).absolute()) == self_raw, 'custody manifest or harness changed')
        for relative, row in pinned.items():
            raw = stable(SOURCE / relative)
            need(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'],
                 'frozen custody source changed')
        for spelling, row in inputs.items():
            raw = stable(Path(spelling))
            need(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'],
                 'held native input source changed')

    guard()
    os.environ.update(TMPDIR=str(attempt), TMP=str(attempt), TEMP=str(attempt),
        PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_ADDOPTS='',
        IPFS_DATASETS_PY_MINIMAL_IMPORTS='1', IPFS_AUTO_INSTALL='false',
        IPFS_DATASETS_AUTO_INSTALL='false', IPFS_ACCEL_SKIP_CORE='1')
    tempfile.tempdir = str(attempt)
    sys.dont_write_bytecode = True
    import pytest

    class Counts:
        collected = failures = errors = skipped = passed_call_reports = 0
        def __init__(self):
            self.passed_nodeids = set()
        def pytest_sessionfinish(self, session, exitstatus):
            self.collected = session.testscollected
        def pytest_runtest_logreport(self, report):
            if report.skipped:
                self.skipped += 1
            elif report.failed:
                if report.when == 'call':
                    self.failures += 1
                else:
                    self.errors += 1
            elif report.when == 'call' and report.passed:
                self.passed_nodeids.add(report.nodeid)
                self.passed_call_reports += 1

    counts = Counts()
    code = int(pytest.main(['-q', '-s', '--assert=plain', '-c', '/dev/null',
        '--confcutdir', str(SOURCE), '--rootdir', str(SOURCE), '--import-mode=importlib',
        '-p', 'no:cacheprovider', '--basetemp', str(attempt / 'pytest'),
        '--junitxml', str(attempt / 'ordinary-custody-controls.xml'), str(SOURCE / TEST)], plugins=[counts]))
    guard()
    passed_cases = len(counts.passed_nodeids)
    passed = code == 0 and counts.collected == passed_cases == 35 and counts.failures == counts.errors == counts.skipped == 0
    result = {'schema': 'legal-ir-ordinary-custody-memory-controls@1', 'passed': passed,
        'returncode': code, 'source_manifest_sha256': expected, 'controlled_testcases': counts.collected,
        'expected_controlled_testcases': 35, 'passed_testcases': passed_cases,
        'additional_passed_call_reports': counts.passed_call_reports - passed_cases,
        'case_count_policy': 'Unique pytest node IDs; successful subtest reports do not duplicate their parent case.',
        'failures': counts.failures, 'errors': counts.errors, 'skipped': counts.skipped,
        'all_test_outputs_within_original_native_attempt': True, 'external_test_aliases_created': False,
        'test_scope': 'Pure correspondence and early-refusal controls only; no genuine native binder or cleanup qualification.',
        'test_control_native_admissions': 0, 'Docker_calls_performed': False,
        'provider_calls_performed': False, 'canonical_tasks_accepted': 0,
        'actual_custody_binding_qualified': False, 'actual_provider_execution_qualified': False,
        'production_hook_deployed': False, 'acceptance_authority': False, 'completion_authority': False}
    raw = (json.dumps(result, indent=2, sort_keys=True) + '\n').encode()
    descriptor = os.open(attempt / 'ordinary-custody-controls.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(attempt, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
