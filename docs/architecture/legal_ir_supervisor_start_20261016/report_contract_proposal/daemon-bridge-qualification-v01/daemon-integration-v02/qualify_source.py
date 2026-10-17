"""Finite source-only worker; launch under the original native orchestrator."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import stat
import sys
import unittest

HERE = Path(__file__).absolute().parent


def read(path):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('unaliased_source_required')
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > 16 * 1024**2:
            raise ValueError('bounded_regular_source_required')
        raw = stream.read()
        after = os.fstat(stream.fileno())
    witness = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
    if witness(before) != witness(after) or witness(after) != witness(path.stat()):
        raise ValueError('stable_source_required')
    return raw


def main():
    if len(sys.argv) != 2:
        raise ValueError('exact_manifest_digest_required')
    raw = read(HERE / 'SOURCE_MANIFEST.json')
    if hashlib.sha256(raw).hexdigest() != sys.argv[1]:
        raise ValueError('source_manifest_changed')
    manifest = json.loads(raw)
    guards = dict(manifest['input_pins'])
    for row in manifest['files']:
        guards[str(HERE / row['path'])] = row['sha256']
    guards[str(HERE / 'SOURCE_MANIFEST.json')] = sys.argv[1]

    def guard():
        for name, digest in guards.items():
            if hashlib.sha256(read(name)).hexdigest() != digest:
                raise ValueError('source_closure_changed: ' + name)
    guard()
    attempt = Path(os.environ['TMPDIR'])
    if not attempt.is_dir() or any(p.is_symlink() for p in (attempt, *attempt.parents)):
        raise ValueError('original_private_native_attempt_required')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location('bridge_source_controls', HERE / 'test_integration_contract.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    with (attempt / 'daemon-source-controls.log').open('x') as stream:
        os.chmod(stream.name, 0o600)
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        stream.flush()
        os.fsync(stream.fileno())
    guard()
    passed = result.wasSuccessful() and result.testsRun == 15 and not result.skipped
    payload = {'schema': 'legal-ir-daemon-source-qualification@1', 'passed': passed,
               'controlled_testcases': result.testsRun, 'failures': len(result.failures),
               'errors': len(result.errors), 'skipped': len(result.skipped),
               'source_manifest_sha256': sys.argv[1], 'runtime_enabled': False,
               'provider_execution_qualified': False, 'native_host_qualified': False,
               'canonical_tasks_accepted': 0, 'acceptance_authority': False,
               'scope': 'actual composed daemon AST and preserved producers only'}
    with (attempt / 'daemon-source-controls.json').open('x') as stream:
        os.chmod(stream.name, 0o600)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(attempt, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
