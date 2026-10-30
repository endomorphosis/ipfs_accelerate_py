"""Charged60 reader/inventory/public-parser controls; no Git/provider/acceptance run."""
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile

HERE = Path(__file__).absolute().parent
SOURCE = HERE / 'real-candidate-binding-v03'
EXPECTED = '829d35224fc21fa40bd63e50b8d9073e3e66e58dd332710fc85d29caa214be97'
EXTERNAL = Path('/tmp/pytest-of-barberb/legal-ir-candidate-readers-20261017-qualification03')


def main():
    attempt = Path(os.environ['TMPDIR']).absolute()
    assert attempt.is_dir() and attempt != EXTERNAL and attempt not in EXTERNAL.parents
    assert not EXTERNAL.exists() and not any(p.is_symlink() for p in (EXTERNAL, *EXTERNAL.parents))
    manifest_raw = (SOURCE / 'SOURCE_MANIFEST.json').read_bytes()
    assert hashlib.sha256(manifest_raw).hexdigest() == EXPECTED
    manifest = json.loads(manifest_raw)
    assert manifest['controlled_testcases'] == 60

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
        '--junitxml', str(attempt / 'candidate-reader-controls.xml'),
        str(SOURCE / 'test_binding_memory.py'), str(SOURCE / 'test_binding_readers.py'),
        str(SOURCE / 'test_public_parser_regression.py')], plugins=[counts]))
    guard()
    passed_cases = len(counts.passed_nodeids)
    passed = code == 0 and counts.collected == passed_cases == 60 and counts.failures == counts.errors == counts.skipped == 0
    result = {'schema': 'legal-ir-candidate-reader-qualification@1', 'passed': passed,
        'returncode': code, 'controlled_testcases': counts.collected, 'passed_testcases': passed_cases,
        'additional_passed_call_reports': counts.passed_call_reports - passed_cases,
        'case_count_policy': 'Unique pytest node IDs; successful subtest reports do not duplicate their parent case.',
        'failures': counts.failures, 'errors': counts.errors, 'skipped': counts.skipped,
        'source_manifest_sha256': EXPECTED, 'external_test_root': str(EXTERNAL),
        'test_aliases_outside_native_attempt': True, 'actual_public_observer_qualified': False,
        'actual_public_parser_regression_passed': passed, 'native_git_fixture_qualified': False,
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
