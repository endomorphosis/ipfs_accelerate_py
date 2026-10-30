"""Finite source/gate/builder controls; no actual provider or hook deployment."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import re
import stat
import sys
import tempfile

PHASE = Path('/home/barberb/lift_coding/artifacts/legal-ir-real-handoff-20261017-01')
SOURCE = PHASE / 'ordinary-success-hook-v01'
EXTERNAL = Path('/tmp/pytest-of-barberb/legal-ir-ordinary-hook-controls-20261017-r01')
TESTS = ('test_handoff_memory.py', 'test_gate_memory.py', 'test_builder_callsites.py')
HARNESS = 'qualify_ordinary_success_hook_v01.py'


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
             and before.st_mode & 0o002 == 0 and 0 < before.st_size <= maximum, 'bounded owned trusted source required')
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
         'usage: qualify_ordinary_success_hook_v01.py EXPECTED_MANIFEST_SHA256')
    expected = sys.argv[1]
    # Capture the real wrapper's private attempt before test aliases redirect
    # TMPDIR. The harness cannot establish or replace native authority itself.
    attempt = Path(os.environ['TMPDIR']).absolute()
    info = attempt.lstat()
    need(attempt.is_dir() and info.st_uid == os.geteuid() and info.st_mode & 0o077 == 0
         and attempt != EXTERNAL and attempt not in EXTERNAL.parents and EXTERNAL not in attempt.parents
         and not any(p.is_symlink() for p in (attempt, *attempt.parents)), 'original private charged attempt required')
    need(not EXTERNAL.exists() and not any(p.is_symlink() for p in (EXTERNAL, *EXTERNAL.parents)),
         'fresh separately precharged test alias directory required')
    manifest_raw = stable(SOURCE / 'SOURCE_MANIFEST.json', 1024 * 1024)
    need(hashlib.sha256(manifest_raw).hexdigest() == expected, 'frozen ordinary hook manifest differs')
    manifest = strict_json(manifest_raw)
    count = manifest.get('controlled_testcases')
    need(type(count) is int and 0 < count <= 256, 'exact manifest-controlled unique case count required')
    rows = manifest.get('files')
    need(type(rows) is list and 1 <= len(rows) <= 64, 'finite complete hook source closure required')
    pinned = {}
    for row in rows:
        need(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'closed source manifest row required')
        spelling = row['path']
        need(type(spelling) is str, 'source relative spelling required')
        if spelling.startswith('/'):
            path = Path(spelling)
            need(SOURCE in path.parents, 'absolute row must remain in the frozen hook bundle')
            relative = path.relative_to(SOURCE).as_posix()
        else:
            pure = PurePosixPath(spelling)
            need(not pure.is_absolute() and '..' not in pure.parts and pure.as_posix() == spelling,
                 'canonical relative source row required')
            relative = spelling
        need(relative not in pinned and relative != 'SOURCE_MANIFEST.json' and type(row['bytes']) is int
             and 0 < row['bytes'] <= 8 * 1024 * 1024 and type(row['sha256']) is str
             and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'unique bounded source row required')
        pinned[relative] = row
    need(set(TESTS) | {HARNESS} <= set(pinned), 'actual tests and exact archived harness must be pinned before freeze')
    self_raw = stable(Path(__file__).absolute())
    need(hashlib.sha256(self_raw).hexdigest() == pinned[HARNESS]['sha256']
         and len(self_raw) == pinned[HARNESS]['bytes'], 'executed harness differs from its archived exact copy')

    def guard():
        need(stable(SOURCE / 'SOURCE_MANIFEST.json', 1024 * 1024) == manifest_raw
             and stable(Path(__file__).absolute()) == self_raw, 'hook manifest or harness changed')
        for relative, row in pinned.items():
            raw = stable(SOURCE / relative)
            need(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'],
                 'frozen hook source changed')

    guard()
    EXTERNAL.mkdir(mode=0o700)
    os.environ.update(TMPDIR=str(EXTERNAL), TMP=str(EXTERNAL), TEMP=str(EXTERNAL),
        PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_ADDOPTS='',
        IPFS_DATASETS_PY_MINIMAL_IMPORTS='1', IPFS_AUTO_INSTALL='false',
        IPFS_DATASETS_AUTO_INSTALL='false', IPFS_ACCEL_SKIP_CORE='1')
    tempfile.tempdir = str(EXTERNAL)
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
        '-p', 'no:cacheprovider', '--basetemp', str(EXTERNAL / 'pytest'),
        '--junitxml', str(attempt / 'ordinary-hook-controls.xml'),
        *(str(SOURCE / name) for name in TESTS)], plugins=[counts]))
    guard()
    passed_cases = len(counts.passed_nodeids)
    passed = code == 0 and counts.collected == passed_cases == count and counts.failures == counts.errors == counts.skipped == 0
    result = {'schema': 'legal-ir-ordinary-success-hook-controls@1', 'passed': passed,
        'returncode': code, 'source_manifest_sha256': expected,
        'controlled_testcases': counts.collected, 'expected_controlled_testcases': count,
        'passed_testcases': passed_cases, 'additional_passed_call_reports': counts.passed_call_reports - passed_cases,
        'case_count_policy': 'Unique pytest node IDs; successful subtest reports do not duplicate their parent case.',
        'failures': counts.failures, 'errors': counts.errors, 'skipped': counts.skipped,
        'external_test_root': str(EXTERNAL), 'test_aliases_outside_native_attempt': True,
        'provider_calls_performed': False, 'canonical_tasks_accepted': 0,
        'actual_provider_execution_qualified': False, 'production_hook_deployed': False,
        'acceptance_authority': False, 'completion_authority': False}
    raw = (json.dumps(result, indent=2, sort_keys=True) + '\n').encode()
    descriptor = os.open(attempt / 'ordinary-hook-controls.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
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
