"""Actual private DuckDB component cases; executed only by admitted qualifier.

No mocked or constructed-looking connection is accepted as a positive. Each
positive handle comes from the pinned original local producer. Invalid actual
handles below are deliberately closed, retired or poisoned by actual APIs.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import threading
import unittest

ROOT = CONTROL = STATE = PORTAL_CLASS = None
OBSERVATIONS = []


def configure(root, control, state_module, portal_class):
    global ROOT, CONTROL, STATE, PORTAL_CLASS
    ROOT, CONTROL, STATE, PORTAL_CLASS = root, control, state_module, portal_class


@contextmanager
def observe_no_owned_effects(connection, *, allow_internal_refusal=False):
    """Observe original Python producer code, without replacing its behavior."""
    if sys.getprofile() is not None:
        raise RuntimeError('fixture requires an unused current-thread profiler')
    forbidden = {STATE.open_duckdb_connection.__code__, STATE.DuckDBConnection.close.__code__,
                 STATE.DuckDBConnection.reconnect_exclusive_owner.__code__, CONTROL._project_ducklake.__code__}
    if not allow_internal_refusal:
        forbidden.add(CONTROL._connect.__code__)
    observed = []
    original_raw, epoch = connection._connection, connection._reconnect_epoch

    def profile(frame, event, _arg):
        if event != 'call':
            return
        if frame.f_code in forbidden:
            raise AssertionError('borrowed branch reached an owned/open/close/recovery/history producer')
        if frame.f_code is STATE.DuckDBConnection.execute.__code__:
            sql = str(frame.f_locals['sql'])
            normalized = ' '.join(sql.strip().upper().split())
            if not normalized.startswith(('SELECT ', 'INSERT ')):
                raise AssertionError('borrowed branch executed unexpected SQL: ' + normalized[:120])
            observed.append(normalized)

    sys.setprofile(profile)
    try:
        yield observed
    finally:
        sys.setprofile(None)
        if connection._connection is not original_raw or connection._reconnect_epoch != epoch:
            raise AssertionError('borrowed branch replaced the original actual native handle')


class BorrowedRealDatabaseCases(unittest.TestCase):
    def setUp(self):
        if ROOT is None:
            raise RuntimeError('only the source-pinned admitted worker may configure this fixture')
        self.root = ROOT / self._testMethodName
        self.root.mkdir(mode=0o700)
        self.state_path = self.root / 'lane-state.json'
        self.target = self.root / 'lane.duckdb'
        self.connection = STATE.open_duckdb_connection(self.target, prefer_quack=False,
                                                      memory_limit='64MB', threads=1)
        self.assertIs(type(self.connection), STATE.DuckDBConnection)
        self.connection.execute(CONTROL.DDL)
        self.connection.execute(CONTROL.HANDOFF_DDL)
        self.body = {'active_task_id': 'fixture-only', 'active_attempt': 2,
                     'implementation_attempts': {'fixture-only': 2}, 'completion_authority': False}
        self.connection.execute(
            'INSERT INTO portal_task_state VALUES (?, ?, ?, FALSE, ?)',
            [self.state_path.stem, json.dumps(self.body), 1, 'fixture-baseline'])
        CONTROL.bind_task_state_control_plane(self.state_path, str(self.target))

    def tearDown(self):
        self.connection.close()
        OBSERVATIONS.append({'case': self._testMethodName, 'original_connection_closed_by_fixture_owner': True,
                             'provider_calls_performed': False, 'canonical_tasks_accepted': 0})

    def borrow(self):
        return CONTROL.borrow_task_state_connection(self.state_path, connection=self.connection, target=self.target)

    def require(self):
        return CONTROL.require_borrowed_task_state_connection(self.state_path, connection=self.connection)

    def test_lane_original_sql_roundtrip_without_second_open_or_close(self):
        with observe_no_owned_effects(self.connection) as statements:
            with self.borrow() as borrowed:
                self.assertIs(borrowed, self.connection)
                self.assertIs(self.require(), self.connection)
                initial = CONTROL.load_portal_task_state(self.state_path.stem, state_path=self.state_path)
                self.assertEqual(initial['active_attempt'], 2)
                body = dict(initial, heartbeat_at='fixture-heartbeat')
                changed = CONTROL.save_portal_task_state(self.state_path.stem, body, state_path=self.state_path)
                self.assertTrue(changed['changed'])
                self.assertEqual(changed['revision'], 2)
                self.assertEqual(changed['ducklake']['status'], 'skipped_borrowed_state_connection')
                self.assertFalse(CONTROL.save_portal_task_state(self.state_path.stem, body, state_path=self.state_path)['changed'])
                self.assertEqual(CONTROL.load_portal_task_state(self.state_path.stem, state_path=self.state_path)['heartbeat_at'],
                                 'fixture-heartbeat')
        self.assertTrue(any(s.startswith('INSERT INTO PORTAL_TASK_STATE') for s in statements))
        self.assertEqual(self.connection.execute('SELECT 37').fetchone()[0], 37)
        self.assertFalse(self.connection._closed)

    def test_board_original_sql_roundtrip_uses_same_handle(self):
        self.connection.close()
        board = self.root / 'board-control-plane'
        board.mkdir(mode=0o700)
        self.target = board / 'control.duckdb'
        self.connection = STATE.open_duckdb_connection(self.target, prefer_quack=False, memory_limit='64MB', threads=1)
        self.connection.execute('''CREATE TABLE artefact_generic (
            board_namespace VARCHAR, kind VARCHAR, artefact_id VARCHAR, payload_json VARCHAR,
            updated_at_ms BIGINT, PRIMARY KEY (board_namespace, kind, artefact_id))''')
        self.connection.execute('INSERT INTO artefact_generic VALUES (?, ?, ?, ?, ?)',
                                [CONTROL._BOARD_NAMESPACE, CONTROL._BOARD_KIND, self.state_path.stem,
                                 json.dumps(self.body), 1])
        CONTROL.bind_task_state_control_plane(self.state_path, str(self.target))
        with observe_no_owned_effects(self.connection) as statements:
            with self.borrow():
                before = CONTROL.load_portal_task_state(self.state_path.stem, state_path=self.state_path)
                body = dict(before, heartbeat_at='board-heartbeat')
                changed = CONTROL.save_portal_task_state(self.state_path.stem, body, state_path=self.state_path)
                self.assertTrue(changed['changed'])
                self.assertEqual(changed['store'], 'board-control-plane')
                self.assertFalse(CONTROL.save_portal_task_state(self.state_path.stem, body, state_path=self.state_path)['changed'])
                self.assertEqual(CONTROL.load_portal_task_state(self.state_path.stem, state_path=self.state_path)['heartbeat_at'],
                                 'board-heartbeat')
        self.assertTrue(any(s.startswith('INSERT INTO ARTEFACT_GENERIC') for s in statements))
        self.assertEqual(self.connection.execute('SELECT 41').fetchone()[0], 41)

    def test_original_portal_class_save_load_preserves_counter(self):
        with observe_no_owned_effects(self.connection):
            with self.borrow():
                state = PORTAL_CLASS.load(self.state_path)
                self.assertEqual(state.implementation_attempts['fixture-only'], 2)
                state.heartbeat_at = 'class-heartbeat'
                self.assertTrue(state.save(self.state_path))
                again = PORTAL_CLASS.load(self.state_path)
                self.assertEqual(again.heartbeat_at, 'class-heartbeat')
                self.assertEqual(again.implementation_attempts['fixture-only'], 2)

    def test_missing_original_row_refuses_legacy_json_migration(self):
        self.connection.execute('DELETE FROM portal_task_state')
        sentinel = b'{"active_attempt":999,"implementation_attempts":{"fixture-only":999}}'
        self.state_path.write_bytes(sentinel)
        with observe_no_owned_effects(self.connection):
            with self.borrow():
                with self.assertRaisesRegex(CONTROL.PortalTaskStateControlPlaneError, 'original_authoritative_row'):
                    PORTAL_CLASS.load(self.state_path)
        self.assertEqual(self.state_path.read_bytes(), sentinel)
        self.assertEqual(self.connection.execute('SELECT count(*) FROM portal_task_state').fetchone()[0], 0)

    def test_default_owned_empty_producer_behavior_is_preserved(self):
        self.connection.execute('DELETE FROM portal_task_state')
        self.connection.close()
        sentinel = b'{"active_attempt":999}'
        self.state_path.write_bytes(sentinel)
        self.assertEqual(CONTROL.load_portal_task_state(self.state_path.stem, state_path=self.state_path, create=False), {})
        self.assertEqual(self.state_path.read_bytes(), sentinel)
        # The caller, rather than a borrowed scope, explicitly reacquires its
        # own fixture handle after the original owned producer has closed.
        self.connection = STATE.open_duckdb_connection(self.target, prefer_quack=False, memory_limit='64MB', threads=1)

    def test_missing_explicit_route_refuses_without_implicit_resolution(self):
        key = str(self.state_path)
        selected = CONTROL._BOUND_STATE_TARGETS.pop(key)
        try:
            with observe_no_owned_effects(self.connection):
                with self.assertRaisesRegex(CONTROL.PortalTaskStateControlPlaneError, 'original_explicit_target'):
                    with self.borrow():
                        self.fail('implicit route admitted')
        finally:
            CONTROL._BOUND_STATE_TARGETS[key] = selected

    def test_exception_identity_and_caller_lifetime_survive_scope(self):
        original = RuntimeError('fixture-original-exception')
        try:
            with observe_no_owned_effects(self.connection):
                with self.borrow():
                    raise original
        except RuntimeError as caught:
            self.assertIs(caught, original)
        else:
            self.fail('original exception missing')
        self.assertEqual(self.connection.execute('SELECT 43').fetchone()[0], 43)
        with self.assertRaisesRegex(CONTROL.PortalTaskStateControlPlaneError, 'active_original_scope'):
            self.require()

    def test_offthread_actual_state_access_refuses_without_reopen(self):
        observed = []
        with self.borrow():
            def other_thread():
                try:
                    CONTROL.load_portal_task_state(self.state_path.stem, state_path=self.state_path)
                except BaseException as exc:
                    observed.append(exc)
            thread = threading.Thread(target=other_thread)
            thread.start(); thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(observed), 1)
            self.assertIsInstance(observed[0], CONTROL.PortalTaskStateControlPlaneError)
            self.assertIn('original_process_and_thread', str(observed[0]))
            self.assertIs(self.require(), self.connection)

    def test_other_state_path_same_target_refuses(self):
        other = self.root / 'another-lane.json'
        CONTROL.bind_task_state_control_plane(other, str(self.target))
        with self.borrow():
            with self.assertRaisesRegex(CONTROL.PortalTaskStateControlPlaneError, 'exact_original_state_path'):
                CONTROL.load_portal_task_state(other.stem, state_path=other)

    def test_nonidentical_object_cannot_supply_connection_authority(self):
        with self.borrow():
            with self.assertRaisesRegex(CONTROL.PortalTaskStateControlPlaneError, 'identical_original_connection'):
                CONTROL.require_borrowed_task_state_connection(self.state_path, connection=object())

    def test_wrong_declared_actual_target_refuses(self):
        foreign = self.root / 'foreign.duckdb'
        other = STATE.open_duckdb_connection(foreign, prefer_quack=False, memory_limit='64MB', threads=1)
        try:
            with self.assertRaisesRegex(CONTROL.PortalTaskStateControlPlaneError, 'original_explicit_target'):
                with CONTROL.borrow_task_state_connection(self.state_path, connection=self.connection, target=foreign):
                    self.fail('wrong target admitted')
        finally:
            other.close()

    def test_rebinding_active_target_refuses(self):
        with self.borrow():
            with self.assertRaisesRegex(CONTROL.PortalTaskStateControlPlaneError, 'binding_mutation'):
                CONTROL.bind_task_state_control_plane(self.state_path, str(self.target))

    def test_actual_transaction_refuses_state_io_without_reconnect(self):
        with self.borrow():
            self.connection.execute('BEGIN TRANSACTION')
            try:
                with self.assertRaisesRegex(CONTROL.PortalTaskStateControlPlaneError, 'nontransactional'):
                    self.require()
            finally:
                self.connection.rollback()
        self.assertEqual(self.connection.execute('SELECT 47').fetchone()[0], 47)

    def test_actual_poisoned_handle_refuses_automatic_recovery(self):
        with self.borrow():
            self.connection.execute('BEGIN TRANSACTION')
            # The actual raw rollback makes the wrapper's subsequent real
            # COMMIT fail; original public execute poisons that native handle.
            self.connection._connection.execute('ROLLBACK')
            with self.assertRaises(Exception):
                self.connection.commit()
            self.assertTrue(self.connection._poisoned)
            original_epoch = self.connection._reconnect_epoch
            with self.assertRaises(CONTROL.PortalTaskStateControlPlaneError):
                self.require()
            self.assertEqual(self.connection._reconnect_epoch, original_epoch)
            self.assertIsNone(self.connection._connection)

    def test_actual_closed_handle_refuses_reopen(self):
        with self.borrow():
            self.connection.close()
            with self.assertRaises(CONTROL.PortalTaskStateControlPlaneError):
                self.require()
            self.assertTrue(self.connection._closed)
            self.assertIsNone(self.connection._connection)

    def test_actual_catalog_switch_refuses(self):
        self.connection.execute("ATTACH ':memory:' AS foreign_fixture")
        self.connection.execute('USE foreign_fixture')
        with self.assertRaisesRegex(CONTROL.PortalTaskStateControlPlaneError, 'current_database_and_schema'):
            with self.borrow():
                self.fail('catalog switch admitted')

    def test_same_target_merge_open_refuses_before_owned_effects(self):
        with observe_no_owned_effects(self.connection, allow_internal_refusal=True):
            with self.borrow():
                with self.assertRaisesRegex(CONTROL.PortalTaskStateControlPlaneError, 'same_target_reopen'):
                    CONTROL.load_merge_handoff('fixture-only', target=str(self.target))

    def test_legacy_source_profile_and_unreviewed_backends(self):
        profile = dict(task_source=None, plan_revision_store=None, require_active_plan_revision=False,
                       configured_parallel_execution_plan=None, active_plan_binding=None)
        with self.borrow():
            CONTROL.require_borrowed_task_state_source_profile(self.state_path, **profile)
            for key in profile:
                changed = dict(profile)
                changed[key] = True if key == 'require_active_plan_revision' else object()
                with self.subTest(unsupported_field=key):
                    with self.assertRaisesRegex(CONTROL.PortalTaskStateControlPlaneError, 'task_source_or_plan'):
                        CONTROL.require_borrowed_task_state_source_profile(self.state_path, **changed)
