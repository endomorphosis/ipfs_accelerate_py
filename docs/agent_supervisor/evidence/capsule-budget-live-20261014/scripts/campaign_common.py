"""Artifact-only helpers. Importing this module performs no external action."""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat

A = Path(__file__).resolve().parent
Q = Path('/home/barberb/lift_coding/.worktrees/supervisor-token-growth-20261013')
LABELS = ('legacy', 'one-capsule')
BUDGETS = dict(zip(LABELS, ('legacy', 'one-capsule@1')))
SCRIPT_FILES = ('campaign_common.py', 'prepare_pair.py', 'verify_runtime.py', 'run_locked_trial.py',
                'provider-readiness/probe_grok_metadata.py')


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def binding(path):
    path = Path(path).absolute()
    info = path.lstat()
    need(path.resolve(strict=True) == path and stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
         'canonical regular single-link artifact required')
    return dict(path=str(path), bytes=info.st_size, sha256=sha(path))


def read(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 64 * 1024**2,
         'bounded regular JSON artifact required')
    def pairs(rows):
        value = {}
        for key, item in rows:
            need(key not in value, 'duplicate JSON field')
            value[key] = item
        return value
    def invalid(_):
        raise ValueError('nonfinite JSON value')
    return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def write_numbered(directory, stem, value):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1000):
        path = directory / f'{stem}-{index:03d}.json'
        try:
            write_new(path, value)
            return path
        except FileExistsError:
            continue
    raise ValueError('bounded observation sequence exhausted')


def recipe():
    value = read(A / 'recipe.json')
    need(value.get('schema') == 'capsule-budget-live-recipe@1' and value.get('cwd') == str(Q),
         'known campaign recipe required')
    need(set(value.get('arms', {})) == set(LABELS), 'exact two-arm campaign required')
    need(value.get('allowed_config_difference_paths') == ['/agents/0/kwargs/capsule_budget', '/jobs_dir'],
         'unexpected treatment differences')
    need(value.get('provider_profile') == 'grok-4.7-cli-1.0.46@1' and value.get('model') == 'grok-4.7',
         'fixed Grok route required')
    for label in LABELS:
        row = value['arms'][label]
        need(row.get('capsule_budget') == BUDGETS[label] and row.get('arm') == 'full', 'arm treatment differs')
        out = A / 'live-trials' / f'grok-{label}-01'
        need(row.get('output') == str(out), 'fresh isolated campaign output required')
        for operation in ('prepare', 'execute', 'collect'):
            argv = row[operation + '_argv']
            need(type(argv) is list and all(type(v) is str for v in argv)
                 and argv[:4] == [value['python'], '-B', '-m',
                     'benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark']
                 and argv[4] == operation and argv[argv.index('--output') + 1] == str(out),
                 'bounded declared benchmark command required')
    return value


def safe_environment():
    value = read(A / 'live-environment.json')
    env = dict(os.environ)
    for key in value['scrub_inherited_names']:
        env.pop(key, None)
    env.update(value['environment_overrides'])
    return env


def differences(left, right, path=''):
    missing = object()
    if type(left) is dict and type(right) is dict:
        return sum((differences(left.get(key, missing), right.get(key, missing), path + '/' + key)
                    for key in sorted(left.keys() | right.keys())), [])
    if type(left) is list and type(right) is list and len(left) == len(right):
        return sum((differences(a, b, path + '/' + str(i)) for i, (a, b) in enumerate(zip(left, right))), [])
    return [] if type(left) is type(right) and left == right else [path]


def check_binding(row):
    need(binding(row['path']) == row, 'frozen artifact changed: ' + row['path'])


def benchmark_source_pins():
    return {str(path): sha(path) for path in sorted((Q / 'benchmarks/agent_supervisor/container_coding').glob('*.py'))}


def disk_admission_snapshot():
    """Match the current scheduler's rounded disk signal without importing it.

    The general policy threshold is diagnostic; ordinary admitted-benchmark
    START does not use it as a disk gate. This does not reserve capacity or replace the
    container's later admission after archive/setup expansion.
    """
    source = Q / 'ipfs_accelerate_py/agent_supervisor/runtime/resource_scheduler.py'
    tree = ast.parse(source.read_text())
    policy = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ResourcePolicy')
    field = next(node for node in policy.body if isinstance(node, ast.AnnAssign)
                 and isinstance(node.target, ast.Name) and node.target.id == 'disk_high_watermark_percent')
    threshold = ast.literal_eval(field.value)
    need(type(threshold) is int and 0 < threshold <= 100, 'known disk admission threshold required')
    try:
        import psutil
        disk = psutil.disk_usage(str(Q))
        percent = int(round(float(disk.percent)))
        native_percent = float(disk.percent)
        available, total = int(disk.free), int(disk.total)
        method = 'psutil_disk_usage_then_scheduler_round'
    except (ImportError, AttributeError, OSError):
        info = os.statvfs(Q)
        available, total = int(info.f_bavail) * int(info.f_frsize), int(info.f_blocks) * int(info.f_frsize)
        need(total > 0, 'disk capacity unavailable')
        percent = max(0, min(100, 100 - (available * 100 // total)))
        native_percent = None
        method = 'scheduler_statvfs_fallback'
    return dict(schema='capsule-budget-host-disk-admission@1', observed_at_utc=now(),
        path=str(Q), method=method, native_disk_percent=native_percent, scheduler_disk_percent=percent,
        disk_high_watermark_percent=threshold, disk_available_bytes=available, disk_total_bytes=total,
        default_threshold_exceeded=percent >= threshold, active_policy_verified=False,
        native_start_admission_observed=False,
        default_percent_threshold_applies_to_benchmark_start=False,
        scheduler_source_sha256=sha(source), provider_calls=0, filesystem_mutation_by_observer=False,
        reserves_capacity=False, container_admission_still_required=True)


def record_admission_gate(directory, selection):
    need(type(selection) is dict and selection.get('mode') in {
        'observation-only', 'minimum-free-bytes', 'default-high-watermark'}, 'explicit root-local gate mode required')
    minimum = selection.get('minimum_available_bytes', 0)
    need(type(minimum) is int and minimum >= 0, 'nonnegative declared disk-byte prerequisite required')
    observed = disk_admission_snapshot()
    mode = selection['mode']
    if mode == 'minimum-free-bytes':
        passed = observed['disk_available_bytes'] >= minimum
    elif mode == 'default-high-watermark':
        passed = not observed['default_threshold_exceeded']
    else:
        passed = None
    observed.update(root_local_gate_mode=mode, minimum_available_bytes=minimum,
        root_local_prerequisite_passed=passed,
        root_local_prerequisite_scope='Frozen campaign choice, not proof of native START admission.')
    path = write_numbered(directory, 'host-disk', observed)
    need(passed is not False, 'root-local disk prerequisite refused; retained ' + str(path))
    return binding(path)


def reporting_freeze_binding(path):
    value = read(path)
    need(value.get('schema') == 'capsule-budget-live-reporting-code-freeze@1',
         'explicit reporting implementation freeze required')
    rows = value.get('files')
    need(type(rows) is list and 1 <= len(rows) <= 64, 'bounded reporting implementation closure required')
    paths = set()
    for row in rows:
        need(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'} and row['path'] not in paths,
             'unique reporting source binding required')
        check_binding(row); paths.add(row['path'])
    need(str(A / 'read_live_pair.py') in paths, 'actual capsule-budget reader must be frozen')
    validation = value.get('validation')
    need(type(validation) is dict and type(validation.get('test_result')) is dict,
         'reporting validation binding required')
    check_binding(validation['test_result'])
    return binding(path)
