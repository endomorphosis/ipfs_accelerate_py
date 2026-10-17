"""Finite source/refusal controls after original V05 parent attachment.

No target daemon is imported or instantiated. No provider, Git, Docker,
database connection or additional native owner is started by this worker.
"""
from pathlib import Path, PurePosixPath
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import tempfile

HERE = Path(__file__).absolute().parent
ROOT = HERE.parent
ATTEMPT = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs/legal-ir-settlement-source-controls-20261017-r02')
REQUEST = HERE/'request-r02.json'
NAME = 'legal-ir-settlement-source-controls-r02'
STORAGE = 128*1024*1024
EXPECTED_CASES = 38
GATE = ROOT/'repository-fixture-v02/parent_attachment_gate.py'
GATE_SHA = 'dba5662443619358e2225d0cf7ccaa2e2b6c5bf51b0a60d2d1809004d1667fb3'


def need(value, reason):
    if not value:
        raise ValueError(reason)


def stable(path, maximum=256*1024*1024):
    path = Path(path)
    need(path.is_absolute() and str(path) == os.path.abspath(path)
         and not any(item.is_symlink() for item in (path, *path.parents)), 'unaliased_source_required')
    with os.fdopen(os.open(path, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK), 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid in {0,os.getuid()}
             and not before.st_mode & 0o002 and 0 <= before.st_size <= maximum,
             'bounded_owned_regular_source_required')
        raw = stream.read(maximum+1); after = os.fstat(stream.fileno())
    witness = lambda row: (row.st_dev,row.st_ino,row.st_size,row.st_mode,row.st_uid,
                           row.st_gid,row.st_nlink,row.st_mtime_ns,row.st_ctime_ns)
    need(len(raw) == before.st_size and witness(before) == witness(after) == witness(path.lstat()),
         'stable_source_required')
    return raw


def strict(raw):
    def pairs(items):
        result = {}
        for key,value in items:
            need(key not in result,'duplicate_JSON_key'); result[key] = value
        return result
    return json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_JSON')))


def main():
    need(len(sys.argv) == 2 and re.fullmatch('[0-9a-f]{64}',sys.argv[1]),'exact_manifest_argument_required')
    expected = sys.argv[1]
    sys.dont_write_bytecode = True
    raw = stable(HERE/'SOURCE_MANIFEST.json',1024*1024)
    need(hashlib.sha256(raw).hexdigest() == expected,'source_manifest_changed')
    manifest = strict(raw)
    need(manifest.get('schema') == 'legal-ir-settlement-source-controls@1'
         and manifest.get('expected_testcases') == EXPECTED_CASES
         and manifest.get('expected_testcases_by_file') == {'test_settlement_source.py':21,'../settlement-audit-v01/test_continuation_refusals.py':17},
         'exact_source_control_generation_required')
    guards = {str(HERE/'SOURCE_MANIFEST.json'):expected}
    names = set()
    for row in manifest['files']:
        relative = PurePosixPath(row['path'])
        need(not relative.is_absolute() and '..' not in relative.parts
             and relative.as_posix() == row['path'] and row['path'] not in names,'unique_relative_source_required')
        names.add(row['path'])
        data = stable(HERE/row['path'])
        need(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'],'source_row_changed')
        guards[str(HERE/row['path'])] = row['sha256']
    guards.update(manifest['input_pins'])
    def guard():
        for spelling,digest in guards.items():
            need(type(digest) is str and re.fullmatch('[0-9a-f]{64}',digest)
                 and hashlib.sha256(stable(spelling)).hexdigest() == digest,'complete_source_closure_changed')
    guard()
    need(hashlib.sha256(stable(GATE)).hexdigest() == GATE_SHA,'original_parent_gate_source_changed')
    spec = importlib.util.spec_from_file_location('actual_original_parent_attachment_gate',GATE)
    gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
    attachment = gate.wait_for_original_generic_parent(REQUEST,attempt_directory=ATTEMPT,
        expected_request_name=NAME,storage_bytes=STORAGE,
        worker_command=['/usr/bin/python3.12','-B',str(HERE/'qualify_controls.py'),expected],timeout_seconds=60)
    # No pytest, product helper or source assembler import occurs before the gate.
    attempt = Path(os.environ['TMPDIR'])
    need(attempt == ATTEMPT,'original_charged_attempt_required')
    os.environ.update(TMPDIR=str(attempt),TMP=str(attempt),TEMP=str(attempt),PYTHONDONTWRITEBYTECODE='1',
        PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTEST_ADDOPTS='',IPFS_DATASETS_PY_MINIMAL_IMPORTS='1',
        IPFS_AUTO_INSTALL='false',IPFS_DATASETS_AUTO_INSTALL='false',IPFS_ACCEL_SKIP_CORE='1')
    tempfile.tempdir = str(attempt)
    import pytest
    class Counts:
        collected = failures = errors = skipped = call_reports = 0
        def __init__(self):self.passed = set()
        def pytest_sessionfinish(self,session,exitstatus):self.collected = session.testscollected
        def pytest_runtest_logreport(self,report):
            if report.skipped:self.skipped += 1
            elif report.failed:
                if report.when == 'call':self.failures += 1
                else:self.errors += 1
            elif report.when == 'call' and report.passed:
                self.passed.add(report.nodeid); self.call_reports += 1
    counts = Counts()
    junit = attempt/'settlement-source-controls.xml'
    code = int(pytest.main(['-q','-s','--assert=plain','-c','/dev/null','--confcutdir',str(HERE),
        '--rootdir',str(HERE),'--import-mode=importlib','-p','no:cacheprovider',
        '--basetemp',str(attempt/'pytest'),'--junitxml',str(junit),
        str(HERE/'test_settlement_source.py'),str(ROOT/'settlement-audit-v01/test_continuation_refusals.py')],plugins=[counts]))
    guard()
    junit_raw = stable(junit,16*1024*1024)
    os.chmod(junit,0o600,follow_symlinks=False)
    with junit.open('rb') as stream:os.fsync(stream.fileno())
    passed = code == 0 and counts.collected == len(counts.passed) == EXPECTED_CASES and counts.failures == counts.errors == counts.skipped == 0
    result = dict(schema='legal-ir-settlement-source-control-result@1',passed=passed,returncode=code,
        source_manifest_sha256=expected,expected_testcases=EXPECTED_CASES,
        controlled_testcases=counts.collected,passed_testcases=len(counts.passed),
        additional_passed_call_reports=counts.call_reports-len(counts.passed),
        failures=counts.failures,errors=counts.errors,skipped=counts.skipped,
        original_parent_attachment_order_observed=attachment['original_parent_attachment_order_observed'],
        private_junit_path=str(junit),private_junit_sha256=hashlib.sha256(junit_raw).hexdigest(),
        source_assembly_only=True,positive_original_daemon_callback_cases=0,
        additional_native_admissions=0,Git_calls_performed=False,Docker_calls_performed=False,
        provider_calls_performed=False,database_connections_opened=False,
        full_original_host_qualified=False,original_attempt_disposition_settled=False,
        runtime_enabled=False,github_qualification='skipped_by_user',canonical_tasks_accepted=0,
        acceptance_authority=False,completion_authority=False)
    data = (json.dumps(result,indent=2,sort_keys=True)+'\n').encode()
    need(len(data) <= 256*1024,'bounded_result_required')
    fd = os.open(attempt/'settlement-source-controls.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    directory = os.open(attempt,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(directory)
    finally:os.close(directory)
    print(json.dumps(result,sort_keys=True),flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
