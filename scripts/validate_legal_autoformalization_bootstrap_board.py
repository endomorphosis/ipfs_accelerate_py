#!/usr/bin/env python3
"""Read-only validation of the bounded LAC inventory execution projection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-all', action='store_true')
    parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for key, value in {
        'PYTHONDONTWRITEBYTECODE': '1', 'IPFS_ACCEL_SKIP_CORE': '1',
        'IPFS_AUTO_INSTALL': 'false', 'IPFS_DATASETS_AUTO_INSTALL': 'false',
        'IPFS_DATASETS_PY_MINIMAL_IMPORTS': '1',
    }.items():
        os.environ.setdefault(key, value)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import parse_task_file
    from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import parse_goal_heap
    plan = root / 'docs/architecture/legal_autoformalization_convergence'
    config_path = root / 'config/legal_autoformalization_bootstrap_scheduler.json'
    config = json.loads(config_path.read_text())
    sources_path = root / 'config/legal_autoformalization_bootstrap_sources.json'
    sources = json.loads(sources_path.read_text())
    tasks = parse_task_file(plan / 'bootstrap.todo.md', '## LAC-')
    goals = parse_goal_heap((plan / 'bootstrap.objectives.md').read_text())
    expected = {'LAC-001', 'LAC-002', 'LAC-003'}
    errors = []
    if len(tasks) != 3 or {t.task_id for t in tasks} != expected:
        errors.append('bootstrap task population differs from the exact allowlist')
    if len(goals) != 4:
        errors.append('bootstrap goal hierarchy differs from the four expected records')
    for task in tasks:
        output = f'data/legal_autoformalization_convergence/{task.task_id}/result.json'
        validation = f'python3 scripts/legal_autoformalization_bootstrap.py materialize --task {task.task_id}'
        if task.depends_on:
            errors.append(f'{task.task_id}: inventory bootstrap must have no dependencies')
        if task.metadata.get('outputs') != output or task.metadata.get('predicted files') != output:
            errors.append(f'{task.task_id}: scope/output mismatch')
        if task.metadata.get('validation') != validation:
            errors.append(f'{task.task_id}: executable validation mismatch')
        if task.metadata.get('provider role') != 'deterministic-only' or task.metadata.get('context budget tokens') != '0':
            errors.append(f'{task.task_id}: inventory requires deterministic-only zero-provider context')
    full_plan = plan / 'full_plan/tasks.json'
    if hashlib.sha256(full_plan.read_bytes()).hexdigest() != sources['original_plan_sha256']:
        errors.append('original plan copy does not match its frozen identity')
    if config.get('max_lanes') != 3 or config.get('objective_refill_enabled') or config.get('codebase_refill_enabled'):
        errors.append('bootstrap concurrency/refill policy mismatch')
    if set(config['bootstrap_scope']['allowed_task_ids']) != expected:
        errors.append('scheduler bootstrap allowlist differs from the actual board')
    if any(not (root / name).is_file() for name in config['protected_paths']):
        errors.append('missing protected bootstrap source')
    print(json.dumps({'schema': 'legal-autoformalization-bootstrap-board-validation/v1',
        'valid': not errors, 'task_count': len(tasks), 'goal_count': len(goals),
        'ready_task_ids': sorted(expected), 'errors': errors,
        'original_plan_sha256': sources['original_plan_sha256'],
        'scope': 'Inventory-only execution projection; original task acceptance remains separately evidenced.'}, indent=2))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
