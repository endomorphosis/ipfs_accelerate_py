"""Finite cold controls under the genuine externally admitted owner process.

These tests import stdlib-only H/C or AST-selected G definitions. They never
establish actual native custody, Docker transport or provider success.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import unittest

ROOT = Path('/home/barberb/lift_coding/artifacts/legal-ir-callable-handoff-20261017-01')
ATTEMPT = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs/legal-ir-callable-memory-controls-20261017-r03')
MANIFESTS = {
    ROOT / 'handoff-v03/SOURCE_MANIFEST.json': '6ca1a9ca25f1bb477841a006e2b77ba34c0c647d5930b3eaabf7537e6dbf6879',
    ROOT / 'runner-v03/SOURCE_MANIFEST.json': '1a2d28a952f0bac6a7af951b4754085dcb1110f86fb46ebf319014f2e66062e2',
}
TESTS = (
    (ROOT / 'handoff-v03/test_handoff_memory.py', 80),
    (ROOT / 'handoff-v03/test_custody_hold_memory.py', 35),
    (ROOT / 'handoff-v03/test_callable_handoff_refusals.py', 17),
    (ROOT / 'runner-v03/test_callable_contract.py', 42),
)

def need(value, reason):
    if not value:
        raise ValueError(reason)

def digest(path):
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path
         and not any(p.is_symlink() for p in (path, *path.parents)), 'canonical source required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    value, size = hashlib.sha256(), 0
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
             and before.st_uid in (0, os.geteuid()) and not before.st_mode & 0o002
             and 0 < before.st_size <= 256 * 1024 * 1024, 'bounded owned source required')
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            size += len(chunk)
            value.update(chunk)
        after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode, s.st_nlink)
    need(size == before.st_size and identity(before) == identity(after) == identity(path.lstat()), 'source changed')
    return size, value.hexdigest()

def pins():
    result = {}
    for path, expected in MANIFESTS.items():
        need(digest(path)[1] == expected, 'source manifest differs')
        result[path] = expected
        manifest = json.loads(path.read_bytes())
        for row in manifest['files']:
            target = path.parent / row['path']
            need(target.is_relative_to(path.parent) and '..' not in target.parts, 'manifest path escaped')
            need(digest(target) == (row['bytes'], row['sha256']), 'manifest source differs')
            result[target] = row['sha256']
        for row in manifest.get('input_pins', []) + manifest.get('input_sources', []):
            target = Path(row['path'])
            need(digest(target) == (row['bytes'], row['sha256']), 'context source differs')
            result[target] = row['sha256']
    result[Path(__file__).absolute()] = digest(Path(__file__).absolute())[1]
    return result

def main():
    need(sys.dont_write_bytecode and Path(os.environ['TMPDIR']) == ATTEMPT,
         'original genuine charged no-bytecode attempt required')
    info = ATTEMPT.lstat()
    need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and not info.st_mode & 0o077,
         'private charged output root required')
    protected = pins()
    suite, counts = unittest.TestSuite(), []
    for index, (path, expected) in enumerate(TESTS):
        spec = importlib.util.spec_from_file_location('callable_memory_suite_' + str(index), path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        selected = unittest.defaultTestLoader.loadTestsFromModule(module)
        counts.append(selected.countTestCases())
        need(counts[-1] == expected, 'finite unique control count differs')
        suite.addTests(selected)
    outcome = unittest.TextTestRunner(verbosity=1).run(suite)
    for path, expected in protected.items():
        need(digest(path)[1] == expected, 'frozen source changed during tests')
    passed = outcome.wasSuccessful() and outcome.testsRun == 174 and not outcome.skipped
    result = {'schema': 'legal-ir-callable-memory-controls@1', 'passed': passed,
              'tests_run': outcome.testsRun, 'expected_tests': 174, 'source_suite_counts': counts,
              'failures': [case.id() for case, _ in outcome.failures],
              'errors': [case.id() for case, _ in outcome.errors], 'skipped': len(outcome.skipped),
              'scope': 'cold input/source and necessary-condition controls only',
              'provider_calls': 0, 'Docker_calls': 0, 'test_native_admissions': 0,
              'canonical_task_attempts': 0, 'canonical_tasks_accepted': 0,
              'native_custody_qualified': False, 'provider_execution_qualified': False,
              'production_hook_deployed': False, 'acceptance_authority': False}
    target = ATTEMPT / 'callable-memory-controls.json'
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(ATTEMPT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if passed else 1

if __name__ == '__main__':
    raise SystemExit(main())
