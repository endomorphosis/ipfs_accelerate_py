#!/usr/bin/env python3
"""Qualify ASEH commands in isolated clones without replacing recorded evidence.

This produces independent source qualification only. It never admits a native
validation receipt, completes a task/goal, drains an owner, or authorizes a merge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

REPOSITORIES = ('ipfs_accelerate_py', 'ipfs_datasets_py', 'ipfs_kit_py')
COHORTS = {'ASEH-070': 'hermetic', 'ASEH-071': 'historical',
           'ASEH-072': 'live-shadow', 'ASEH-073': 'canary'}
HARNESS = 'benchmarks/agent_supervisor/efficiency_state_hardening/paired_harness.py'


def rewritten_command(task, output):
    """Preserve validator semantics while giving new cohort evidence new paths."""
    argv = shlex.split(task['body']['validation'])
    alias = task['task_alias']
    if argv[:4] == ['python3', '-m', 'pytest', '-q']:
        for path in argv[4:]:
            if not path.endswith('.py') or Path(path).is_absolute() or '..' in Path(path).parts:
                raise ValueError('unexpected pytest path')
        return argv
    if alias not in COHORTS or argv[:2] != ['python3', HARNESS]:
        raise ValueError('unsupported ASEH validation command')
    if argv[argv.index('--cohort') + 1] != COHORTS[alias]:
        raise ValueError('cohort does not match task')
    for flag, destination in (
        ('--output', output / (alias + '-results.json')),
        ('--qualification-output', output / (alias + '-qualification.json')),
    ):
        if argv.count(flag) != 1:
            raise ValueError('cohort output flag absent or duplicated')
        argv[argv.index(flag) + 1] = str(destination)
    if alias == 'ASEH-073':
        flag = '--require-shadow-receipt'
        if argv.count(flag) != 1:
            raise ValueError('canary shadow prerequisite absent or duplicated')
        argv[argv.index(flag) + 1] = str(output / 'ASEH-072-results.json')
    return argv


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def install_readonly_guard(roots):
    """Guard Python writes, including descriptor-relative filesystem APIs."""
    open_context = threading.local()
    original_open = os.open
    def descriptor_open(path, flags, mode=0o777, *, dir_fd=None):
        previous = getattr(open_context, 'dir_fd', None)
        open_context.dir_fd = dir_fd
        try:
            return original_open(path, flags, mode, dir_fd=dir_fd)
        finally:
            open_context.dir_fd = previous
    os.open = descriptor_open
    def audit(event, args):
        paths = []
        if event == 'open':
            path, mode, flags = args
            if isinstance(path, (str, bytes, os.PathLike)) and (
                (isinstance(mode, str) and any(c in mode for c in 'wax+'))
                or int(flags or 0) & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            ):
                paths = [(path, getattr(open_context, 'dir_fd', None))]
        elif event in {'os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.truncate'}:
            fd_index = 1 if event in {'os.remove', 'os.rmdir'} else 2
            paths = [(args[0], args[fd_index] if len(args) > fd_index else None)]
        elif event in {'os.rename', 'os.link'}:
            paths = [(args[0], args[2]), (args[1], args[3])]
        elif event == 'os.symlink':
            # Creating a symlink mutates its destination entry, not its target.
            paths = [(args[1], args[2])]
        for value, dir_fd in paths:
            if isinstance(value, int):
                value = os.readlink('/proc/self/fd/' + str(value))
            if isinstance(value, (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(value))
                if not path.is_absolute() and dir_fd is not None and dir_fd >= 0:
                    path = Path(os.readlink('/proc/self/fd/' + str(dir_fd))) / path
                resolved = (path.parent.resolve() / path.name
                            if event in {'os.remove', 'os.rmdir', 'os.mkdir', 'os.rename', 'os.symlink'}
                            else path.resolve())
                candidates = (path.absolute(), resolved)
                if any(p == r or r in p.parents for p in candidates for r in roots):
                    raise PermissionError('qualification cannot mutate source checkout: ' + str(path))
    sys.addaudithook(audit)


def child(root, owner, argv):
    """Execute one suite with exact package origins and Python write guards."""
    install_readonly_guard([root, root / 'ipfs_datasets_py', root / 'ipfs_kit_py'])
    import ipfs_accelerate_py, ipfs_datasets_py, ipfs_kit_py
    expected = {'ipfs_accelerate_py': root / 'ipfs_accelerate_py',
                'ipfs_datasets_py': root / 'ipfs_datasets_py',
                'ipfs_kit_py': root / 'ipfs_kit_py/ipfs_kit_py'}
    for module in (ipfs_accelerate_py, ipfs_datasets_py, ipfs_kit_py):
        if Path(module.__file__).resolve().parent != expected[module.__name__]:
            raise RuntimeError('package origin mismatch: ' + module.__name__)
    print(json.dumps({'package_origins': {m.__name__: m.__file__ for m in
        (ipfs_accelerate_py, ipfs_datasets_py, ipfs_kit_py)}}), flush=True)
    if argv[1:3] == ['-m', 'pytest']:
        import pytest
        return pytest.main([*argv[3:], '--noconftest', '--import-mode=importlib',
                           '-p', 'no:cacheprovider', '-o', 'addopts=', '-o', 'log_cli=false'])
    import runpy
    sys.argv = argv[1:]
    runpy.run_path(str(root / argv[1]), run_name='__main__')
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--task', action='append')
    parser.add_argument('--child-owner')
    parser.add_argument('--child-argv')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.child_owner:
        return child(root, args.child_owner, json.loads(args.child_argv))
    if not (root / '.git').is_dir():
        raise ValueError('qualification requires an isolated clone, not a live linked worktree')
    capture = args.capture.resolve(strict=True)
    seal = json.loads((capture / 'capture-receipt.json').read_text())
    for filename, digest in seal['files'].items():
        if hashlib.sha256((capture / filename).read_bytes()).hexdigest() != digest:
            raise ValueError('capture file digest mismatch: ' + filename)
    plan = json.loads((capture / 'plan-projection.json').read_text())
    tasks = sorted(plan['tasks'], key=lambda t: t['task_alias'])
    if len(tasks) != 40 or len({t['task_alias'] for t in tasks}) != 40:
        raise ValueError('exact 40-task ASEH population required')
    selected = set(args.task or [t['task_alias'] for t in tasks])
    if not selected <= {t['task_alias'] for t in tasks}:
        raise ValueError('unknown task selection')
    output = args.output.resolve()
    if root == output or root in output.parents:
        raise ValueError('qualification output must be outside source clone')
    output.mkdir(parents=True, exist_ok=False)
    roots = {'ipfs_accelerate_py': root, 'ipfs_datasets_py': root / 'ipfs_datasets_py',
             'ipfs_kit_py': root / 'ipfs_kit_py'}
    identities = {name: {'head': git(path, 'rev-parse', 'HEAD'),
                         'tree': git(path, 'rev-parse', 'HEAD^{tree}')}
                  for name, path in roots.items()}
    config = json.loads((root / 'config/agent_supervisor_efficiency_state_hardening_scheduler.json').read_text())
    env = {**os.environ, 'GIT_OPTIONAL_LOCKS': '0', 'PYTHONDONTWRITEBYTECODE': '1',
           'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'PYTHONPATH': os.pathsep.join([str(root), str(root / 'ipfs_datasets_py'),
                str(root / 'ipfs_kit_py'), *config['validation_runtime']['pythonpath_entries']])}
    evidence_paths = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z',
        'docs/architecture/agent_supervisor_efficiency_state_hardening_inventory',
        'docs/architecture/AGENT_SUPERVISOR_EFFICIENCY_AND_STATE_HARDENING_FINAL_REPORT.md',
        'benchmarks/agent_supervisor/efficiency_state_hardening']).split(b'\0')
    def evidence_hashes():
        return {p.decode(): hashlib.sha256((root / p.decode()).read_bytes()).hexdigest()
                for p in evidence_paths if p}
    before = evidence_hashes()
    results = []
    for task in tasks:
        if task['task_alias'] not in selected:
            continue
        owner = task['body']['owning_repository']
        if owner not in roots:
            raise ValueError('unknown task repository')
        argv = rewritten_command(task, output)
        started = time.time()
        log = output / (task['task_alias'] + '.log')
        with log.open('x') as handle:
            try:
                completed = subprocess.run([config['validation_runtime']['python_executable'], '-B',
                    __file__, '--root', str(root), '--child-owner', owner, '--child-argv', json.dumps(argv)],
                    cwd=roots[owner], env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=480)
                code = completed.returncode
            except subprocess.TimeoutExpired:
                code = 124
        result = {'task_alias': task['task_alias'], 'owner': owner, 'argv': argv,
                  'exit_code': code, 'duration_seconds': time.time() - started,
                  'log': str(log), 'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()}
        results.append(result)
        print(json.dumps(result), flush=True)
    after = evidence_hashes()
    unchanged = before == after
    report = {'schema': 'aseh/isolated-current-source-qualification@1', 'authority': False,
              'native_receipt_admitted': False, 'goal_acceptance': False, 'merge_authority': False,
              'source_identities': identities, 'capture_sha256': hashlib.sha256((capture / 'capture-receipt.json').read_bytes()).hexdigest(),
              'recorded_evidence_unchanged': unchanged, 'before': before, 'after': after,
              'conftest_scope': 'unrelated repository conftests excluded; suite fixtures and exact package origins retained',
              'results': results, 'passed': sum(r['exit_code'] == 0 for r in results),
              'failed': sum(r['exit_code'] != 0 for r in results)}
    with (output / 'qualification.json').open('x') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return 0 if unchanged and all(r['exit_code'] == 0 for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
