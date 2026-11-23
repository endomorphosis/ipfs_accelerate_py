"""Normal-import component controls for provisional closeout store selection.

The maintained supervisor method runs with declared task state and an owner
spy. No supervisor constructor or real database owner is used in these tests.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as daemon_module,
    implementation_supervisor as supervisor_module,
)
from ipfs_accelerate_py.agent_supervisor.objectives import objective_graph
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    database_task_source,
    duckdb_state,
)


class ProvisionalStoreAbsenceTests(unittest.TestCase):
    def setUp(self):
        self.private_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.private_directory.cleanup)
        self.directory = Path(self.private_directory.name)
        self.objective_path = self.directory / 'declared-objectives.md'
        self.objective_path.write_text('# Declared test objective\n', encoding='utf-8')
        self.repo_root = self.directory / 'declared-repository-NOT-A-STORE'

    def invoke(self, *, store_id='state/owner.duckdb', authority_mode='legacy_markdown',
               endpoint='', active_task_id='', in_progress=False, statuses=None,
               task_counts=None, goal_status='active', revision=7,
               owner_error=False, cas_error=False, token='', parse_error=False,
               absent_program=False, missing_objective=False):
        events = []
        counts = {'completed': 1, 'ready': 0, 'blocked': 0, 'waiting': 0}
        if task_counts is not None:
            counts = task_counts
        state = SimpleNamespace(
            active_task_id=active_task_id,
            implementation_in_progress=in_progress,
            task_statuses={'T1': 'completed'} if statuses is None else statuses,
            task_count=sum(counts.values()),
            **{key + '_count': value for key, value in counts.items()},
        )
        program = None if absent_program else SimpleNamespace(
            store_id=store_id, authority_mode=authority_mode, quack_endpoint=endpoint,
        )
        declared_self = SimpleNamespace(
            board_namespace='declared-test-board',
            config=SimpleNamespace(
                state_path=self.directory / 'declared-state',
                objective_path=None if missing_objective else self.objective_path,
                database_program=program,
                repo_root=self.repo_root,
                task_shard_index=0,
            ),
        )

        class DeclaredOwner:
            def __init__(self, target, **kwargs):
                events.append(('constructor', target, kwargs))
                if owner_error:
                    raise RuntimeError('declared owner refusal')

            def __enter__(self):
                events.append(('enter',))
                return self

            def __exit__(self, exc_type, exc, traceback):
                events.append(('exit', exc_type))

            def get_goal(self, goal_id):
                events.append(('get_goal', goal_id))
                return {'status': goal_status, 'revision': revision}

            def compare_and_set_goal_status(self, goal_id, expected_revision, status, metadata):
                events.append(('CAS', goal_id, expected_revision, status, metadata))
                if cas_error:
                    raise RuntimeError('declared CAS refusal')

        def discover(store):
            events.append(('discover', store))
            return SimpleNamespace(token=token)

        def parse(text):
            events.append(('parse', text))
            if parse_error:
                raise ValueError('declared parse refusal')
            return [SimpleNamespace(goal_id='G1')]

        with (
            patch.object(daemon_module.PortalTaskState, 'load', return_value=state),
            patch.object(objective_graph, 'parse_goal_heap', side_effect=parse),
            patch.object(database_task_source, 'DatabaseTaskSource', DeclaredOwner),
            patch.object(duckdb_state, 'discover_live_quack_endpoint', side_effect=discover),
        ):
            result = supervisor_module.PortalImplementationSupervisor.provisionally_complete_terminal_goals(declared_self)
        return result, events

    def assert_no_owner_effects(self, events):
        self.assertFalse(any(event[0] in {'constructor', 'enter', 'get_goal', 'CAS', 'discover'} for event in events))

    def test_empty_string_is_absent_without_owner_attempt(self):
        result, events = self.invoke(store_id='')
        self.assertEqual(result['reason'], 'database_store_absent')
        self.assertIs(result['attempted'], False)
        self.assertIs(result['completion_authority'], False)
        self.assertEqual(result['changed_goal_ids'], [])
        self.assert_no_owner_effects(events)

    def test_none_is_absent_without_owner_attempt(self):
        result, events = self.invoke(store_id=None)
        self.assertEqual(result['reason'], 'database_store_absent')
        self.assertIs(result['attempted'], False)
        self.assertIs(result['completion_authority'], False)
        self.assertEqual(result['changed_goal_ids'], [])
        self.assert_no_owner_effects(events)

    def test_nonempty_store_paths_preserve_configured_legacy_owner(self):
        cases = (
            ('state/owner.duckdb', self.repo_root / 'state/owner.duckdb'),
            (str(self.directory / 'owner.duckdb'), self.directory / 'owner.duckdb'),
            (' ', self.repo_root / ' '),
            ('.', self.repo_root),
            (Path('state/owner.duckdb'), self.repo_root / 'state/owner.duckdb'),
        )
        for store_id, expected_target in cases:
            with self.subTest(store_id=store_id):
                result, events = self.invoke(store_id=store_id)
                constructors = [event for event in events if event[0] == 'constructor']
                self.assertEqual(len(constructors), 1)
                self.assertEqual(constructors[0][1], expected_target)
                self.assertEqual(constructors[0][2], {
                    'owner_id': 'implementation-supervisor-provisional-goal:declared-test-board:0',
                    'install_schema': False,
                })
                self.assertEqual(result['reason'], 'applied')
                self.assertIs(result['completion_authority'], False)

    def test_supported_active_goal_statuses_remain_only_provisional(self):
        for status in ('active', 'reopened', 'analysis_inconclusive'):
            with self.subTest(status=status):
                result, events = self.invoke(goal_status=status)
                changes = [event for event in events if event[0] == 'CAS']
                self.assertEqual(len(changes), 1)
                self.assertEqual(changes[0][1:4], ('G1', 7, 'provisionally_complete'))
                self.assertEqual(changes[0][4], {
                    'schema': 'ipfs_accelerate_py/agent-supervisor/provisional-goal-completion@1',
                    'completion_authority': False,
                    'tasks_complete': True,
                    'goal_alias': 'G1',
                    'state': 'provisionally_complete',
                })
                self.assertEqual(result['changed_goal_ids'], ['G1'])
                self.assertIs(result['completion_authority'], False)

    def test_existing_active_or_nonterminal_task_refusals_precede_owner(self):
        cases = (
            ({'active_task_id': 'T1'}, 'active_implementation'),
            ({'in_progress': True}, 'active_implementation'),
            ({'statuses': {'T1': 'ready'}}, 'tasks_not_all_terminal'),
            ({'statuses': {}, 'task_counts': {'completed': 1, 'ready': 1, 'blocked': 0, 'waiting': 0}}, 'tasks_not_all_terminal'),
        )
        for options, reason in cases:
            with self.subTest(options=options):
                result, events = self.invoke(**options)
                self.assertEqual(result['reason'], reason)
                self.assertIs(result['attempted'], False)
                self.assert_no_owner_effects(events)

    def test_existing_missing_objective_program_and_parse_refusals(self):
        cases = (
            ({'missing_objective': True}, 'objective_path_missing'),
            ({'absent_program': True}, 'database_program_absent'),
            ({'parse_error': True}, 'objective_parse_failed:ValueError'),
        )
        for options, reason in cases:
            with self.subTest(options=options):
                result, events = self.invoke(**options)
                self.assertEqual(result['reason'], reason)
                self.assertIs(result['attempted'], False)
                self.assert_no_owner_effects(events)

    def test_existing_terminal_goal_or_invalid_revision_does_not_CAS(self):
        for options in ({'goal_status': 'verified_complete'}, {'goal_status': 'blocked'},
                        {'revision': True}, {'revision': '7'}):
            with self.subTest(options=options):
                result, events = self.invoke(**options)
                self.assertEqual(result['reason'], 'no_active_goals')
                self.assertIs(result['attempted'], True)
                self.assertEqual(result['changed_goal_ids'], [])
                self.assertFalse(any(event[0] == 'CAS' for event in events))

    def test_existing_CAS_refusal_is_not_reported_as_a_change(self):
        result, events = self.invoke(cas_error=True)
        self.assertEqual(result['reason'], 'no_active_goals')
        self.assertIs(result['attempted'], True)
        self.assertEqual(result['changed_goal_ids'], [])
        self.assertEqual(sum(event[0] == 'CAS' for event in events), 1)
        self.assertEqual(sum(event[0] == 'exit' for event in events), 1)

    def test_existing_owner_refusal_restores_attachment_token(self):
        with patch.dict(os.environ, {duckdb_state.QUACK_TOKEN_ENV: 'declared-original-token'}):
            result, events = self.invoke(owner_error=True, token='declared-attach-token')
            self.assertEqual(os.environ[duckdb_state.QUACK_TOKEN_ENV], 'declared-original-token')
        self.assertEqual(result['reason'], 'owner_cas_failed:RuntimeError')
        self.assertIs(result['attempted'], False)
        self.assertFalse(any(event[0] == 'CAS' for event in events))

    def test_existing_successful_owner_exit_restores_attachment_token(self):
        with patch.dict(os.environ, {duckdb_state.QUACK_TOKEN_ENV: 'declared-original-token'}):
            result, events = self.invoke(token='declared-attach-token')
            self.assertEqual(os.environ[duckdb_state.QUACK_TOKEN_ENV], 'declared-original-token')
        self.assertEqual(result['reason'], 'applied')
        self.assertEqual(sum(event[0] == 'exit' for event in events), 1)

    def test_quack_endpoint_contract_is_unchanged(self):
        absent, absent_events = self.invoke(authority_mode='quack', endpoint='', store_id='')
        self.assertEqual(absent['reason'], 'quack_endpoint_absent')
        self.assert_no_owner_effects(absent_events)
        present, events = self.invoke(authority_mode='quack', endpoint='declared://owner', store_id='')
        self.assertEqual(present['reason'], 'applied')
        constructor = next(event for event in events if event[0] == 'constructor')
        self.assertEqual(constructor[1], 'declared://owner')

    def test_existing_missing_projection_and_terminal_counts_bridge(self):
        for counts in ({'completed': 0, 'ready': 0, 'blocked': 0, 'waiting': 0},
                       {'completed': 1, 'ready': 0, 'blocked': 0, 'waiting': 0}):
            with self.subTest(counts=counts):
                result, events = self.invoke(statuses={}, task_counts=counts)
                self.assertEqual(result['reason'], 'applied')
                self.assertIs(result['completion_authority'], False)
                self.assertEqual(sum(event[0] == 'CAS' for event in events), 1)


if __name__ == '__main__':
    unittest.main()
