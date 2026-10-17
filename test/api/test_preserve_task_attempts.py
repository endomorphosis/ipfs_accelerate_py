"""Ordinary in-memory policy controls, not native/provider qualification.

Execute exact function ASTs from reviewed source without importing the product
module's owner/bootstrap effects. The sole AST change lifts selected methods to
module scope; their bodies and arguments are unchanged. Fixture state/queue are
in-memory test doubles, never live ledger objects or simulated native grants.
Set PRESERVE_ATTEMPTS_SOURCE_ROOT to qualify a publisher-applied source tree.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import logging
import os
import re
import shlex
import time
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(os.environ.get('PRESERVE_ATTEMPTS_SOURCE_ROOT', str(Path(__file__).resolve().parents[2])))
D = 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py'
S = 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py'
R = 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py'
C = 'ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py'
REARMS = ('_rearm_attempt_limited_retry_repairs', '_rearm_attempt_limited_stale_proposal_tasks', '_rearm_attempt_limited_obsoleted_validation_tasks')
COOLDOWNS = ('_release_repaired_dependency_preflight_cooldowns', '_release_repaired_read_only_validation_cooldowns', '_release_obsolete_provider_repair_cooldowns', '_release_obsolete_venv_launcher_cooldowns')


@lru_cache(maxsize=4)
def source_tree(path):
    return ast.parse((ROOT / path).read_text())


def node(path, name, cls=None):
    tree = source_tree(path)
    body = next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls) if cls else tree.body
    return next(n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


def load_function(path, name, cls=None, **globals_):
    fn = copy.deepcopy(node(path, name, cls))
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    unit = ast.fix_missing_locations(ast.Module(body=[future, fn], type_ignores=[]))
    ns = {'Mapping': Mapping, 'json': json, 'hashlib': hashlib, 'Path': Path, 'shlex': shlex, 'time': time, 'logging': logging, **globals_}
    exec(compile(unit, str(ROOT / path), 'exec'), ns)
    return ns[name]


class Queue:
    def __init__(self):
        self.records = {'cid:owned': {'failures': 3, 'cooldown': 500}, 'cid:foreign': {'failures': 7, 'cooldown': 900}}
        self.resets = []
        self.saves = 0
        self.entries = {}

    def reset_retry_state(self, cid):
        self.resets.append(cid)
        self.records[cid] = {'failures': 0, 'cooldown': 0}
        return True

    def save(self):
        self.saves += 1


class State(NS):
    def save(self, _path):
        self.saves += 1


def state():
    return State(implementation_attempts={'OWN-1': 3, 'FOREIGN-1': 7}, implementation_attempts_by_cid={'cid:owned': 3, 'cid:foreign': 7}, task_identities={'OWN-1': {'canonical_task_cid': 'cid:owned'}}, retry_budget_repair_receipts={}, retry_budget_repair_rearm_receipts={}, stale_proposal_replay_rearm_receipts={}, validation_obsolescence_rearm_receipts={}, protected_implementation_attempts={}, implementation_in_progress=False, active_task_id='', active_task_cid='cid:owned', last_implementation_task_id='OWN-1', last_implementation_task_cid='cid:owned', saves=0)


def daemon(preserve):
    globals_ = dict(normalize_status=lambda v: v, retry_budget_repair_source=lambda t: (getattr(t, 'source', ''), 'validation'), is_retry_budget_repair_task=lambda t: bool(getattr(t, 'source', '')), utc_now=lambda: 'fixture-clock', RETRY_BUDGET_REPAIR_REARM_POLICY_REVISION='fixture-repair@1', STALE_PROPOSAL_REPLAY_REARM_POLICY_REVISION='fixture-stale@1', VALIDATION_OBSOLESCENCE_REARM_POLICY_REVISION='fixture-validation@1', DETERMINISTIC_DECLARED_OUTPUT_TARGET_REQUIRED='fixture-output-target-required', task_is_declared_read_only_validation=lambda t, **kw: True, preflight_validation_project_dependencies=lambda *a, **kw: {'passed': True}, task_declared_output_paths=lambda t: ('data/report.json',), ExecutionMode=NS(DETERMINISTIC_ONLY=NS(value='deterministic-only')))
    names = (*REARMS, *COOLDOWNS, '_reset_attempt_budgets_for_completed_retry_repairs', '_release_unfinished_active_attempt', '_restore_task_attempt', '_task_attempt_count', '_task_attempt', '_partition_tasks_at_attempt_limit')
    methods = {name: load_function(D, name, 'PortalImplementationDaemon', **globals_) for name in names}
    obj = type('InMemoryOrdinaryDaemon', (), methods)()
    obj.preserve_task_attempts = preserve
    obj.max_task_attempts = 3
    obj.task_shard_count = 1
    obj.strict_task_sharding = True
    obj.task_queue = Queue()
    obj.state_path = Path('/never-opened/fixture-state')
    obj.repo_root = Path('/never-opened/fixture-repo')
    obj.events = []
    obj._record_event = lambda *a: obj.events.append(a)
    obj._identity_for_task = lambda t: NS(canonical_task_key='owned-key', canonical_task_cid='cid:owned', board_namespace='finite-fixture')
    obj._canonical_ref = lambda t: 'cid:owned'
    obj._durable_protected_recovery_attempt = lambda *a: None
    obj._stale_proposal_replay_recovery_evidence = lambda *a, **kw: {'kind': 'independent-unit-evidence'}
    obj._task_has_validation_command_failure_evidence = lambda t: True
    obj._protected_attempt_latched = lambda *a, **kw: False
    obj._protected_attempt_event_latch = lambda *a, **kw: None
    obj._iter_events = lambda: ()
    obj._scoped_recovery_attempts = {}
    obj._provider_isolation_failure_is_obsolete = lambda t: True
    obj._task_declared_implementation_provider = lambda t: 'operator-only'
    obj._validation_uses_venv_argv0 = lambda t: True
    obj._venv_launcher_failure_is_obsolete = lambda t: True
    return obj


@pytest.mark.parametrize('method', REARMS)
def test_preserve_mode_keeps_exhausted_counters_and_retry_queue(method):
    obj, st = daemon(True), state()
    task = NS(task_id='OWN-1', source='SOURCE-1' if method == REARMS[0] else '')
    before, queue_before = copy.deepcopy(vars(st)), copy.deepcopy(vars(obj.task_queue))
    for revision in ('old-runtime', 'new-runtime', 'another-runtime'):
        assert getattr(obj, method)(st, [task], {'OWN-1': 'ready'}, recovery_revision=revision) == []
        assert vars(st) == before
        assert vars(obj.task_queue) == queue_before
    selected, limited = obj._partition_tasks_at_attempt_limit([task], {'OWN-1': 'ready'}, st)
    assert selected == [] and limited[0]['attempt_count'] == 3


@pytest.mark.parametrize('method', REARMS)
def test_default_mode_retains_existing_rearm_behavior(method):
    obj, st = daemon(False), state()
    task = NS(task_id='OWN-1', source='SOURCE-1' if method == REARMS[0] else '')
    result = getattr(obj, method)(st, [task], {'OWN-1': 'ready'}, recovery_revision='new-runtime')
    assert len(result) == 1
    expected = 2 if method == REARMS[1] else 0
    assert st.implementation_attempts.get('OWN-1', 0) == expected
    assert st.implementation_attempts_by_cid.get('cid:owned', 0) == expected
    assert st.implementation_attempts['FOREIGN-1'] == 7
    assert st.implementation_attempts_by_cid['cid:foreign'] == 7
    assert obj.task_queue.records['cid:foreign'] == {'failures': 7, 'cooldown': 900}
    assert obj.task_queue.resets == ['cid:owned']
    assert getattr(obj, method)(st, [task], {'OWN-1': 'ready'}, recovery_revision='new-runtime') == []


@pytest.mark.parametrize('preserve', [False, True])
def test_completed_repair_reset_policy(preserve):
    obj, st = daemon(preserve), state()
    tasks = [NS(task_id='OWN-1', status='todo', source=''), NS(task_id='REPAIR-1', status='completed', source='OWN-1')]
    before = copy.deepcopy(vars(st))
    result, deferred = obj._reset_attempt_budgets_for_completed_retry_repairs(st, tasks)
    assert deferred == []
    if preserve:
        assert result == [] and vars(st) == before
        assert obj.task_queue.resets == [] and obj.task_queue.saves == 0
    else:
        assert len(result) == 1 and st.implementation_attempts.get('OWN-1', 0) == 0
        assert st.implementation_attempts_by_cid.get('cid:owned', 0) == 0
        assert obj.task_queue.resets == ['cid:owned']
    assert st.implementation_attempts['FOREIGN-1'] == 7


@pytest.mark.parametrize('preserve', [False, True])
def test_unknown_vanished_worker_cannot_erase_consumed_attempt_in_preserve_mode(preserve):
    obj, st = daemon(preserve), state()
    result = obj._release_unfinished_active_attempt(st, task_id='OWN-1', attempt=3)
    expected = 3 if preserve else 2
    assert st.implementation_attempts['OWN-1'] == expected
    assert st.implementation_attempts_by_cid['cid:owned'] == expected
    assert result['consumed'] is preserve and result['released'] is not preserve
    assert st.implementation_attempts['FOREIGN-1'] == 7


@pytest.mark.parametrize('method', COOLDOWNS)
@pytest.mark.parametrize('preserve', [False, True])
def test_automatic_cooldown_repair_policy(method, preserve):
    obj = daemon(preserve)
    resets = []
    note = ('validation_project_dependency_preflight_failed' if method == COOLDOWNS[0] else 'ValueError: fixture-output-target-required' if method == COOLDOWNS[1] else 'validation_or_implementation_failed')
    entry = NS(task_id='OWN-1', cooldown_until=time.time() + 1000, notes=note, reset_retry_state=lambda: resets.append('OWN-1'))
    obj.task_queue.entries = {'cid:owned': entry}
    task = NS(task_id='OWN-1', source='SOURCE-1', validation=('fixture-check',))
    result = getattr(obj, method)({'OWN-1': task})
    assert result == ([] if preserve else ['OWN-1'])
    assert resets == ([] if preserve else ['OWN-1'])
    assert obj.task_queue.saves == (0 if preserve else 1)


@pytest.mark.parametrize('preserve', [False, True])
def test_zero_provider_prelaunch_restore_keeps_existing_behavior(preserve):
    obj, st = daemon(preserve), state()
    obj._restore_task_attempt(st, NS(task_id='OWN-1'), 2, zero_provider_execution=True)
    assert st.implementation_attempts['OWN-1'] == 2
    assert st.implementation_attempts_by_cid['cid:owned'] == 2
    assert st.implementation_attempts['FOREIGN-1'] == 7


@pytest.mark.parametrize('proof', [False, None, 0, 1, 'true'])
@pytest.mark.parametrize('restored', [0, 2])
def test_productive_or_unknown_execution_restore_cannot_erase_consumed_ordinal(proof, restored):
    obj, st = daemon(True), state()
    before, queue_before = copy.deepcopy(vars(st)), copy.deepcopy(vars(obj.task_queue))
    obj._restore_task_attempt(st, NS(task_id='OWN-1'), restored, zero_provider_execution=proof)
    assert vars(st) == before and vars(obj.task_queue) == queue_before


def test_legacy_unknown_restore_behavior_is_unchanged():
    obj, st = daemon(False), state()
    obj._restore_task_attempt(st, NS(task_id='OWN-1'), 2)
    assert st.implementation_attempts['OWN-1'] == st.implementation_attempts_by_cid['cid:owned'] == 2


def test_two_consumed_attempts_allow_only_ordinal_three_then_stop_across_runtime_revisions():
    obj, st = daemon(True), state()
    st.implementation_attempts['OWN-1'] = st.implementation_attempts_by_cid['cid:owned'] = 2
    task = NS(task_id='OWN-1', source='')
    assert obj._task_attempt(st, task) == 3
    selected, limited = obj._partition_tasks_at_attempt_limit([task], {'OWN-1': 'ready'}, st)
    assert selected == [task] and limited == []
    # Independent fixture terminal outcome: one actual ordinal was consumed.
    st.implementation_attempts['OWN-1'] = st.implementation_attempts_by_cid['cid:owned'] = 3
    for revision in ('runtime-A', 'runtime-B'):
        for method in REARMS:
            assert getattr(obj, method)(st, [task], {'OWN-1': 'ready'}, recovery_revision=revision) == []
        assert obj._partition_tasks_at_attempt_limit([task], {'OWN-1': 'ready'}, st)[0] == []
    assert st.implementation_attempts_by_cid['cid:owned'] == 3


@pytest.mark.parametrize('path', [D, S])
def test_both_cli_parsers_default_to_legacy_and_accept_explicit_preservation(path):
    fn = node(path, 'parse_args')
    statements = [n for n in fn.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and n.value.args and isinstance(n.value.args[0], ast.Constant) and n.value.args[0].value == '--preserve-task-attempts']
    assert len(statements) == 1
    parser = argparse.ArgumentParser()
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(statements), type_ignores=[])), '<actual-cli-option>', 'exec'), {'parser': parser})
    assert parser.parse_args([]).preserve_task_attempts is False
    assert parser.parse_args(['--preserve-task-attempts']).preserve_task_attempts is True


@pytest.mark.parametrize('preserve', [False, True])
def test_profile_to_supervisor_cli_policy(preserve):
    fn = load_function(S, 'supervisor_scheduler_config_cli_defaults', re=re, REPO_ROOT=Path('/fixture'))
    profile = dict(board_namespace='finite-fixture', taskboard_path='tasks.todo.md', task_prefix='## OWN-', stale_seconds=120, check_interval_seconds=5, max_restarts=1, max_task_attempts=3, daemon_interval_seconds=5, implementation_timeout_seconds=120, validation_max_workers=1, merge_target_branch='fixture', objectives_path='goals.md', worktree_submodule_paths=(), protected_paths=(), manual_completion_authority_task_ids=(), manual_completion_authority_required_task_ids=(), preserve_task_attempts=preserve)
    args = fn(profile, repo_root=Path('/fixture'))
    assert ('--preserve-task-attempts' in args) is preserve
    assert args[args.index('--max-task-attempts') + 1] == '3'


@pytest.mark.parametrize('preserve', [False, True])
def test_configured_board_to_supervisor_cli_policy(preserve):
    fn = load_function(C, 'configured_board_common_args', _objective_refill_controls=lambda p: None, _plan_bound_profile=lambda b: False)
    payload = {name: 1 for name in ('stale_seconds', 'check_interval_seconds', 'watchdog_startup_grace_seconds', 'max_restarts', 'daemon_interval_seconds', 'implementation_timeout_seconds', 'implementation_max_timeout_seconds', 'implementation_log_stall_seconds', 'implementation_retry_budget', 'validation_retry_budget', 'merge_retry_budget')}
    payload.update(max_task_attempts=3, preserve_task_attempts=preserve)
    board = NS(payload=payload, resolved_database_program=lambda: NS(worktree_root=''), path=lambda p: Path('/fixture') / p, runtime_paths={'worktrees': 'worktrees', 'merge_queue': 'queue'}, taskboard_path='tasks.todo.md', task_header_prefix='## OWN-', merge_target_branch='fixture', database_program=None, strict_task_sharding=False, idle_lane_work_stealing=False, worktree_submodule_paths=(), protected_paths=())
    args = fn(board, implement=True)
    assert ('--preserve-task-attempts' in args) is preserve
    assert '--implement' in args
    assert args[args.index('--max-task-attempts') + 1] == '3'


@pytest.mark.parametrize('preserve', [False, True])
@pytest.mark.parametrize('flags', [(), ('--preserve-task-attempts',), ('--preserve-task-attempt',), ('--preserve-task-attempts', '--preserve-task-attempts'), ('--preserve-task-attempts=true',)])
def test_adoption_requires_exact_preservation_policy_and_spelling(preserve, flags):
    fn = load_function(S, '_managed_daemon_matches_command_line', 'PortalImplementationSupervisor')
    cfg = NS(daemon_script_path=None, state_dir='/fixture/state', state_prefix='fixture', todo_path='/fixture/tasks.todo.md', implement=True, max_task_attempts=3, validation_max_workers=1, implementation_protected_paths=(), production_provider_policy='', production_provider_context_budget_tokens=None, production_provider_timeout_seconds=None, production_provider_review_authority_key_path=None, production_provider_launch_authority_receipt_path=None, production_provider_launch_authority_receipt_content_id='', manual_completion_authority_task_ids=(), manual_completion_authority_required_task_ids=(), manual_completion_authority_epoch_id='', manual_completion_authority_revalidation_only=False, preserve_task_attempts=preserve, execution_slice_task_ids=(), execution_slice_task_cids=(), merge_target_branch='', merge_queue_dir=None)
    obj = NS(config=cfg, board_namespace='fixture')
    args = ['python', '-m', 'ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon', '--state-dir', '/fixture/state', '--state-prefix', 'fixture', '--todo-path', '/fixture/tasks.todo.md', '--implement', '--max-task-attempts', '3', '--validation-max-workers', '1', '--board-namespace', 'fixture', *flags]
    expected = flags == (('--preserve-task-attempts',) if preserve else ())
    assert fn(obj, shlex.join(args)) is expected


def test_all_direct_portal_and_database_constructor_boundaries_forward_the_option():
    expected_counts = {D: 2, R: 4}
    for path, count in expected_counts.items():
        tree = ast.parse((ROOT / path).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {'PortalImplementationDaemon', 'DatabaseImplementationDaemon', 'portal_daemon_class'} and any(k.arg == 'max_task_attempts' for k in n.keywords)]
        # Include the standalone database helper whose cap uses its default.
        if path == R:
            calls += [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'DatabaseImplementationDaemon' and not any(k.arg == 'max_task_attempts' for k in n.keywords)]
        assert len(calls) == count
        for call in calls:
            bindings = [kw for kw in call.keywords if kw.arg == 'preserve_task_attempts']
            assert len(bindings) == 1
            expression = ast.Expression(body=copy.deepcopy(bindings[0].value))
            for preserve in (False, True):
                assert eval(compile(ast.fix_missing_locations(expression), '<constructor-policy>', 'eval'), {'parsed': NS(preserve_task_attempts=preserve), 'args': NS(preserve_task_attempts=preserve)}) is preserve


def test_restore_authority_is_only_wired_to_proven_prelaunch_paths():
    tree = ast.parse((ROOT / D).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PortalImplementationDaemon')
    calls = []
    for fn in cls.body:
        if isinstance(fn, ast.FunctionDef):
            for n in ast.walk(fn):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == '_restore_task_attempt':
                    calls.append((fn.name, n))
    assert len(calls) == 9
    explicit = [(name, call, next((k.value for k in call.keywords if k.arg == 'zero_provider_execution'), None)) for name, call in calls]
    assert sum(proof is not None for _, _, proof in explicit) == 5
    assert sum(proof is None for _, _, proof in explicit) == 4
    # Exact nominal terminal paths get no authority. None of the completion,
    # timeout, exception or post-completion durability restores can refund.
    terminal = [(name, call) for name, call, proof in explicit if proof is None]
    assert [name for name, _ in terminal].count('_run_implementation') == 3
    assert [name for name, _ in terminal].count('_run_implementation_in_ephemeral_worktree') == 1
    # The capacity helper has already returned unless both independent
    # pre-dispatch refusals agree; its proof must remain a literal True.
    capacity = [(name, proof) for name, _, proof in explicit if name == '_record_provider_capacity_deferral']
    assert len(capacity) == 1 and isinstance(capacity[0][1], ast.Constant) and capacity[0][1].value is True
    # Every ephemeral proof is gated by its actual provider_dispatched join.
    for name, _, proof in explicit:
        if proof is not None and name == '_run_implementation_in_ephemeral_worktree':
            assert any(isinstance(n, ast.Compare) and isinstance(n.left, ast.Name) and n.left.id == 'provider_dispatched' and len(n.ops) == 1 and isinstance(n.ops[0], ast.Is) and isinstance(n.comparators[0], ast.Constant) and n.comparators[0].value is False for n in ast.walk(proof))


def test_supervisor_config_and_child_argv_forward_the_explicit_option():
    fn = node(S, 'supervisor_config_from_args')
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'PortalSupervisorConfig']
    assert len(calls) == 1
    expr = next(k.value for k in calls[0].keywords if k.arg == 'preserve_task_attempts')
    tree = ast.parse((ROOT / S).read_text())
    append_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == 'self.config.preserve_task_attempts' and any(isinstance(c, ast.Constant) and c.value == '--preserve-task-attempts' for c in ast.walk(n))]
    assert len(append_nodes) == 1
    for preserve in (False, True):
        cfg = eval(compile(ast.fix_missing_locations(ast.Expression(body=copy.deepcopy(expr))), '<actual-config-binding>', 'eval'), {'args': NS(preserve_task_attempts=preserve)})
        assert cfg is preserve
        command = []
        unit = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(append_nodes[0])], type_ignores=[]))
        exec(compile(unit, '<actual-child-option>', 'exec'), {'self': NS(config=NS(preserve_task_attempts=preserve)), 'command': command})
        assert command == (['--preserve-task-attempts'] if preserve else [])
