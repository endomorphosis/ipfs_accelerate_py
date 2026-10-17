"""Finite actual-private-DuckDB qualifier; root runs under generic admission.

Only this fixed private fixture changes. It is not a daemon/provider attempt,
and this child does not claim the parent owner's later native finalization.
"""
import ast
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import types
from typing import Any
import unittest
from xml.etree import ElementTree

HERE = Path(__file__).absolute().parent
CONTROLLER = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller')
RELATIVE = Path('ipfs_accelerate_py/agent_supervisor/todo_daemon/portal_task_state_control_plane.py')
ATTEMPT = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs/legal-ir-borrowed-state-20261017-r01')
SITE = Path('/home/barberb/.local/lib/python3.12/site-packages')
DAEMON = CONTROLLER / 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py'
EXPECTED_SOURCE_CASES, EXPECTED_REAL_CASES = 8, 18


def need(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def checked(path, expected=None, maximum=64 * 1024 * 1024):
    path = Path(path)
    need(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'unaliased source required')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid in {0, os.getuid()}
             and before.st_mode & 0o002 == 0 and 0 < before.st_size <= maximum, 'bounded owned regular source required')
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    witness = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_uid, row.st_nlink,
                           row.st_size, row.st_mtime_ns, row.st_ctime_ns)
    need(len(raw) == before.st_size and witness(before) == witness(after) == witness(path.lstat()), 'source changed')
    if expected is not None:
        need(hashlib.sha256(raw).hexdigest() == expected, 'source SHA changed')
    return raw


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, 'exact module loader required')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_private(name, raw):
    descriptor = os.open(ATTEMPT / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
        self._by_case = {}

    def startTest(self, test):
        super().startTest(test)
        row = {'case': test.id(), 'passed': True}
        self._by_case[test.id()] = row
        self.records.append(row)

    def addSuccess(self, test):
        super().addSuccess(test)

    def addFailure(self, test, error):
        super().addFailure(test, error)
        self._by_case[test.id()].update(passed=False, failure='failure')

    def addError(self, test, error):
        super().addError(test, error)
        self._by_case[test.id()].update(passed=False, failure='error')

    def addSubTest(self, test, subtest, error):
        super().addSubTest(test, subtest, error)
        if error is not None:
            self._by_case[test.id()].update(passed=False, failure='failure')

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._by_case[test.id()].update(passed=False, failure='skipped')


def main():
    need(len(sys.argv) == 2 and len(sys.argv[1]) == 64
         and all(c in '0123456789abcdef' for c in sys.argv[1]), 'frozen manifest SHA argument required')
    manifest_raw = checked(HERE / 'SOURCE_MANIFEST.json', sys.argv[1])
    manifest = json.loads(manifest_raw)
    need(manifest.get('schema') == 'legal-ir-borrowed-state-source@1', 'exact source schema required')
    for row in manifest['files']:
        raw = checked(HERE / row['path'], row['sha256'])
        need(len(raw) == row['bytes'], 'frozen source byte count required')
    for path, digest in manifest['input_pins'].items():
        checked(path, digest)
    need(ATTEMPT.is_dir() and not ATTEMPT.is_symlink()
         and ATTEMPT.stat().st_uid == os.getuid(), 'already-admitted original private attempt required')
    fixture_root = ATTEMPT / 'actual-private-databases'
    need(not fixture_root.exists() and not fixture_root.is_symlink(), 'fresh private fixture required')
    # This source cannot grant admission from a path or manifest. Root invokes
    # it only inside its retained genuine native generic-owner context.
    fixture_root.mkdir(mode=0o700)
    for key in list(os.environ):
        if key.startswith('IPFS_ACCELERATE_AGENT_QUACK_') or key in {'IPFS_ACCELERATE_AGENT_CONTROL_DUCKDB', 'IPFS_ACCELERATE_AGENT_CONTROL_DUCKLAKE'}:
            del os.environ[key]
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', IPFS_ACCEL_SKIP_CORE='1',
                      IPFS_DATASETS_PY_MINIMAL_IMPORTS='1', IPFS_AUTO_INSTALL='false',
                      IPFS_DATASETS_AUTO_INSTALL='false', OMP_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    sys.path.insert(0, str(CONTROLLER))
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as state_module
    import duckdb
    import _duckdb
    need(Path(state_module.__file__).absolute() == CONTROLLER / 'ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py',
         'actual pinned original connection producer required')
    need(Path(duckdb.__file__).absolute() == SITE / 'duckdb/__init__.py'
         and Path(_duckdb.__file__).absolute() == SITE / '_duckdb.cpython-312-aarch64-linux-gnu.so',
         'actual pinned installed DuckDB payload required')
    control_name = 'ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane'
    control = load_file(control_name, HERE / 'proposed' / RELATIVE)

    # Select the entire original PortalTaskState class AST verbatim, including
    # its real save/load methods. The huge daemon module is never imported.
    daemon_raw = checked(DAEMON, manifest['input_pins'][str(DAEMON)])
    tree = ast.parse(daemon_raw)
    selected = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PortalTaskState')
    selected_ast_sha = hashlib.sha256(ast.dump(selected, include_attributes=False).encode()).hexdigest()
    selected_module = types.ModuleType('ipfs_accelerate_py.agent_supervisor.todo_daemon._selected_fixture_state')
    selected_module.__package__ = 'ipfs_accelerate_py.agent_supervisor.todo_daemon'
    selected_module.__dict__.update({'dataclass': dataclass, 'field': field, 'asdict': asdict,
                                    'Path': Path, 'Any': Any, 'json': json})
    sys.modules[selected_module.__name__] = selected_module
    selected_tree = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    exec(compile(selected_tree, str(DAEMON), 'exec'), selected_module.__dict__)
    cold = load_file('legal_ir_borrowed_source_controls', HERE / 'test_borrowed_state_source.py')
    real = load_file('legal_ir_borrowed_real_cases', HERE / 'test_borrowed_state_real.py')
    real.configure(fixture_root, control, state_module, selected_module.PortalTaskState)
    loader = unittest.TestLoader()
    cold_suite, real_suite = loader.loadTestsFromModule(cold), loader.loadTestsFromModule(real)
    need(cold_suite.countTestCases() == EXPECTED_SOURCE_CASES and real_suite.countTestCases() == EXPECTED_REAL_CASES,
         'exact cold and real fixture populations required')
    capture = io.StringIO()
    result = unittest.TextTestRunner(stream=capture, verbosity=2, resultclass=RecordedResult).run(
        unittest.TestSuite([cold_suite, real_suite]))
    raw_capture = capture.getvalue().encode()
    need(len(raw_capture) <= 1024 * 1024, 'bounded actual fixture diagnostics required')
    write_private('borrowed-state-controls.log', raw_capture)
    junit = ElementTree.Element('testsuite', tests=str(result.testsRun), failures=str(len(result.failures)),
                               errors=str(len(result.errors)), skipped=str(len(result.skipped)))
    for row in result.records:
        case = ElementTree.SubElement(junit, 'testcase', name=row['case'])
        if not row['passed']:
            ElementTree.SubElement(case, row['failure'])
    write_private('borrowed-state-controls.junit.xml', ElementTree.tostring(junit, encoding='utf-8', xml_declaration=True))
    passed = (result.wasSuccessful() and result.testsRun == EXPECTED_SOURCE_CASES + EXPECTED_REAL_CASES
              and len(result.records) == result.testsRun and not result.skipped)
    report = {'schema': 'legal-ir-borrowed-state-private-database-qualification@1',
              'passed': passed, 'source_manifest_sha256': sys.argv[1],
              'controlled_source_cases': EXPECTED_SOURCE_CASES, 'controlled_real_database_cases': EXPECTED_REAL_CASES,
              'tests_run': result.testsRun, 'passed_testcases': sum(row['passed'] for row in result.records),
              'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
              'cases': result.records, 'actual_fixture_owner_closeout_observations': real.OBSERVATIONS,
              'original_PortalTaskState_class_AST_sha256': selected_ast_sha,
              'full_daemon_module_imported': False, 'original_class_selected_verbatim': True,
              'native_parent_finalization_claimed': False, 'production_state_written': False,
              'provider_calls_performed': False, 'canonical_tasks_accepted': 0,
              'acceptance_authority': False, 'completion_authority': False,
              'production_hook_deployed': False, 'real_provider_path_qualified': False}
    encoded = (json.dumps(report, indent=2, sort_keys=True) + '\n').encode()
    write_private('borrowed-state-qualification.json', encoded)
    descriptor = os.open(ATTEMPT, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps({'schema': 'legal-ir-borrowed-state-qualification-terminal@1',
                      'passed': passed, 'result_path': str(ATTEMPT / 'borrowed-state-qualification.json'),
                      'result_sha256': hashlib.sha256(encoded).hexdigest(),
                      'source_manifest_sha256': sys.argv[1], 'provider_calls_performed': False,
                      'canonical_tasks_accepted': 0, 'acceptance_authority': False}, sort_keys=True), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
