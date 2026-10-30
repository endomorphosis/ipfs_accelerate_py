"""Finite in-memory handoff controls under genuine native CPU ownership."""
from pathlib import Path
import hashlib
import json
import os
import sys

HERE = Path(__file__).absolute().parent
DATASETS = Path('/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets')


def main():
    request_path = Path(sys.argv[1])
    raw = request_path.read_bytes()
    request = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == sys.argv[2]
    assert request['schema'] == 'legal-ir-handoff-native-stdout-controls-request@1'
    assert type(request['expected_testcases']) is int and 0 < request['expected_testcases'] <= 1000
    assert 0 < len(request['test_paths']) <= 8
    assert all(HERE in Path(path).parents for path in request['test_paths'])

    def guard():
        assert request_path.read_bytes() == raw
        for spelling, digest in request['source_sha256'].items():
            assert hashlib.sha256(Path(spelling).read_bytes()).hexdigest() == digest

    guard()
    sys.dont_write_bytecode = True
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
                      PYTEST_ADDOPTS='', IPFS_DATASETS_PY_MINIMAL_IMPORTS='1',
                      IPFS_AUTO_INSTALL='false', IPFS_DATASETS_AUTO_INSTALL='false',
                      IPFS_ACCEL_SKIP_CORE='1')
    sys.path.insert(0, str(DATASETS))
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import open_existing_scheduler
    scheduler = open_existing_scheduler('/tmp/ipfs-datasets-resource-scheduler-1000.json',
        'f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661')
    with scheduler.acquire('orchestration', cpu_slots=1, memory_mb=1024, child_process_slots=1,
            requires_gpu=False, timeout=60, request_id=request['name'] + ':root') as root:
        with root.acquire_child(cpu_slots=1, memory_mb=1024, child_process_slots=1,
                requires_gpu=False, timeout=60, request_id=request['name'] + ':phase') as phase:
            guard()
            import pytest

            class Counts:
                collected = 0
                passed = 0
                failures = 0
                errors = 0
                skipped = 0
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
            rc = int(pytest.main(['-q', '-s', '--assert=plain', '-c', '/dev/null',
                '--confcutdir', str(HERE), '--rootdir', str(HERE), '--import-mode=importlib',
                '-p', 'no:cacheprovider', '-p', 'no:tmpdir', *request['test_paths']], plugins=[counts]))
            guard()
            assert root.renew() and phase.renew()
            ids = {'root_lease_id': root.lease_id, 'phase_lease_id': phase.lease_id}
    passed = (rc == 0 and counts.collected == counts.passed == request['expected_testcases']
              and counts.failures == counts.errors == counts.skipped == 0)
    print(json.dumps({'schema': 'legal-ir-handoff-native-stdout-controls-terminal@1',
        'passed': passed, 'returncode': rc, 'controlled_testcases': counts.collected,
        'passed_testcases': counts.passed, 'failures': counts.failures,
        'errors': counts.errors, 'skipped': counts.skipped,
        'expected_testcases': request['expected_testcases'],
        'source_request_sha256': hashlib.sha256(raw).hexdigest(),
        'actual_root_released': root.released, 'actual_phase_released': phase.released,
        'filesystem_test_outputs_requested': False, 'native_disk_grant_claimed': False,
        'real_git_or_docker_runtime_qualified': False, 'packet_materialized': False,
        'provider_calls_performed': False, 'canonical_tasks_accepted': 0, **ids}), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
