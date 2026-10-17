"""Run exactly47 cold/source cases inside the separate actual V05 owner.

This worker starts no provider, Docker, Git or test-native owner. The enclosing
V05 entry owns real CPU/disk admission and finalization; test inputs grant none.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import re
import stat
import sys
import tempfile

HERE = Path(__file__).absolute().parent
ATTEMPT = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs/legal-ir-freezer-board-controls-20261017-r01')


def need(value, reason):
    if not value:
        raise ValueError(reason)


def stable(path, maximum=256 * 1024 * 1024):
    path = Path(path)
    need(path.is_absolute() and str(path) == os.path.abspath(path)
         and not any(p.is_symlink() for p in (path, *path.parents)), 'unaliased_source_required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_uid in {0, os.getuid()} and before.st_nlink == 1
             and before.st_mode & 0o002 == 0 and 0 <= before.st_size <= maximum, 'bounded_owned_source_required')
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    witness = lambda m: (m.st_dev, m.st_ino, m.st_size, m.st_mode, m.st_uid, m.st_gid,
                         m.st_nlink, m.st_mtime_ns, m.st_ctime_ns)
    need(len(raw) == before.st_size and witness(before) == witness(after) == witness(path.lstat()), 'stable_source_required')
    return raw


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate_json_key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_json')))


def main():
    need(len(sys.argv) == 2 and re.fullmatch('[0-9a-f]{64}', sys.argv[1]), 'exact_source_manifest_argument_required')
    expected = sys.argv[1]
    attempt = Path(os.environ['TMPDIR'])
    meta = attempt.lstat()
    need(attempt == ATTEMPT and stat.S_ISDIR(meta.st_mode) and meta.st_uid == os.getuid()
         and meta.st_mode & 0o077 == 0 and not any(p.is_symlink() for p in (attempt, *attempt.parents)),
         'exact_private_charged_native_attempt_required')
    raw = stable(HERE / 'SOURCE_MANIFEST.json', 1024 * 1024)
    need(hashlib.sha256(raw).hexdigest() == expected, 'frozen_source_manifest_changed')
    manifest = strict(raw)
    need(manifest.get('schema') == 'legal-ir-report-freezer-dual-board-source@4'
         and manifest.get('authored_control_cases') == 47
         and manifest.get('authored_control_cases_by_file') == {'test_board_controls.py': 47}
         and type(manifest.get('files')) is list and 1 <= len(manifest['files']) <= 16
         and type(manifest.get('input_pins')) is dict, 'exact_47_case_source_generation_required')
    guards, names = {str(HERE / 'SOURCE_MANIFEST.json'): expected}, set()
    for row in manifest['files']:
        relative = PurePosixPath(row['path'])
        need(not relative.is_absolute() and '..' not in relative.parts and relative.as_posix() == row['path']
             and row['path'] not in names, 'unique_source_relative_path_required')
        names.add(row['path'])
        data = stable(HERE / row['path'])
        need(type(row['bytes']) is int and len(data) == row['bytes']
             and hashlib.sha256(data).hexdigest() == row['sha256'], 'frozen_source_row_changed')
        guards[str(HERE / row['path'])] = row['sha256']
    need({'test_board_controls.py', 'qualify_board_controls.py',
          'proposed/ipfs_accelerate_py/agent_supervisor/todo_daemon/report_only_candidate_freezer.py'} <= names,
         'actual_test_worker_and_helper_rows_required')
    guards.update(manifest['input_pins'])
    def guard():
        for spelling, digest in guards.items():
            need(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest)
                 and hashlib.sha256(stable(spelling)).hexdigest() == digest, 'complete_source_closure_changed')
    guard()
    os.environ.update(TMPDIR=str(attempt), TMP=str(attempt), TEMP=str(attempt), PYTHONDONTWRITEBYTECODE='1',
        PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_ADDOPTS='', IPFS_DATASETS_PY_MINIMAL_IMPORTS='1',
        IPFS_AUTO_INSTALL='false', IPFS_DATASETS_AUTO_INSTALL='false', IPFS_ACCEL_SKIP_CORE='1')
    tempfile.tempdir = str(attempt)
    sys.dont_write_bytecode = True
    import pytest
    class Counts:
        collected = failures = errors = skipped = passed_call_reports = 0
        def __init__(self): self.passed = set()
        def pytest_sessionfinish(self, session, exitstatus): self.collected = session.testscollected
        def pytest_runtest_logreport(self, report):
            if report.skipped: self.skipped += 1
            elif report.failed:
                if report.when == 'call': self.failures += 1
                else: self.errors += 1
            elif report.when == 'call' and report.passed:
                self.passed.add(report.nodeid)
                self.passed_call_reports += 1
    counts = Counts()
    code = int(pytest.main(['-q', '-s', '--assert=plain', '-c', '/dev/null', '--confcutdir', str(HERE),
        '--rootdir', str(HERE), '--import-mode=importlib', '-p', 'no:cacheprovider',
        '--basetemp', str(attempt / 'pytest'), '--junitxml', str(attempt / 'freezer-board-controls.xml'),
        str(HERE / 'test_board_controls.py')], plugins=[counts]))
    guard()
    junit = attempt / 'freezer-board-controls.xml'
    junit_raw = stable(junit, 16 * 1024 * 1024)
    os.chmod(junit, 0o600, follow_symlinks=False)
    with junit.open('rb') as stream: os.fsync(stream.fileno())
    passed_cases = len(counts.passed)
    passed = code == 0 and counts.collected == passed_cases == 47 and counts.failures == counts.errors == counts.skipped == 0
    result = {'schema': 'legal-ir-report-freezer-dual-board-controls@1', 'passed': passed, 'returncode': code,
        'source_manifest_sha256': expected, 'private_junit_path': str(junit),
        'private_junit_sha256': hashlib.sha256(junit_raw).hexdigest(), 'controlled_testcases': counts.collected,
        'expected_controlled_testcases': 47, 'passed_testcases': passed_cases,
        'additional_passed_call_reports': counts.passed_call_reports - passed_cases,
        'failures': counts.failures, 'errors': counts.errors, 'skipped': counts.skipped,
        'case_count_policy': 'Unique pytest node IDs; extra call subreports recorded separately.',
        'all_outputs_within_original_native_attempt': True, 'external_test_aliases_created': False,
        'test_scope': 'Pure necessary conditions and exact source correspondence only.',
        'test_control_native_admissions': 0, 'Git_calls_performed': False, 'Docker_calls_performed': False,
        'provider_calls_performed': False, 'actual_public_preflight_qualified': False,
        'public_report_freezer_qualified': False, 'actual_provider_execution_qualified': False,
        'production_hook_deployed': False, 'canonical_tasks_accepted': 0,
        'acceptance_authority': False, 'completion_authority': False}
    data = (json.dumps(result, indent=2, sort_keys=True) + '\n').encode()
    need(len(data) <= 256 * 1024, 'bounded_control_result_required')
    fd = os.open(attempt / 'freezer-board-controls.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(attempt, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: os.fsync(directory)
    finally: os.close(directory)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
