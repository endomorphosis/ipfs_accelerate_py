"""Charged filesystem reader controls; no observer/provider/acceptance run."""
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile

HERE = Path(__file__).absolute().parent
SOURCE = HERE / 'real-candidate-binding-v02'
EXPECTED = '1699b077d97ba65edb354826123d1785d38631d35651140f919432bb61b4827e'
EXTERNAL = Path('/tmp/pytest-of-barberb/legal-ir-candidate-readers-20261017-qualification01')


def main():
    attempt = Path(os.environ['TMPDIR']).absolute()
    assert attempt.is_dir() and attempt != EXTERNAL and attempt not in EXTERNAL.parents
    assert not EXTERNAL.exists() and not any(p.is_symlink() for p in (EXTERNAL, *EXTERNAL.parents))
    manifest_raw = (SOURCE / 'SOURCE_MANIFEST.json').read_bytes()
    assert hashlib.sha256(manifest_raw).hexdigest() == EXPECTED
    manifest = json.loads(manifest_raw)

    def guard():
        assert (SOURCE / 'SOURCE_MANIFEST.json').read_bytes() == manifest_raw
        for row in manifest['files']:
            raw = (SOURCE / row['path']).read_bytes()
            assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']

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
        collected = passed = failures = errors = skipped = 0
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
                self.passed += 1

    counts = Counts()
    code = int(pytest.main(['-q', '-s', '--assert=plain', '-c', '/dev/null',
        '--confcutdir', str(SOURCE), '--rootdir', str(SOURCE), '--import-mode=importlib',
        '-p', 'no:cacheprovider', '--basetemp', str(EXTERNAL / 'pytest'),
        '--junitxml', str(attempt / 'candidate-reader-controls.xml'),
        str(SOURCE / 'test_binding_memory.py'), str(SOURCE / 'test_binding_readers.py')], plugins=[counts]))
    guard()
    passed = code == 0 and counts.collected == counts.passed == 54 and counts.failures == counts.errors == counts.skipped == 0
    result = {'schema': 'legal-ir-candidate-reader-qualification@1', 'passed': passed,
        'returncode': code, 'controlled_testcases': counts.collected, 'passed_testcases': counts.passed,
        'failures': counts.failures, 'errors': counts.errors, 'skipped': counts.skipped,
        'source_manifest_sha256': EXPECTED, 'external_test_root': str(EXTERNAL),
        'test_aliases_outside_native_attempt': True, 'actual_public_observer_qualified': False,
        'provider_calls_performed': False, 'canonical_tasks_accepted': 0}
    raw = (json.dumps(result, indent=2, sort_keys=True) + '\n').encode()
    with (attempt / 'candidate-reader-qualification.json').open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
