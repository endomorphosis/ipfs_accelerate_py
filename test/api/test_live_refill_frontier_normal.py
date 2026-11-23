"""Normal-import owning controls for the existing live refill implementation.

Run only under the existing owner profile. LIVE_REFILL_CAPTURED_BOARD must bind
an admitted copy of the exact historical board fixture to obtain all17 cases
without skips. No board, database, model or generator operation is invoked.
Static whole-module preservation remains a separate private AST control.
"""
import copy
from datetime import datetime as RealDatetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from ipfs_accelerate_py.agent_supervisor.objectives import backlog_refinery

BOARD_SHA = '8cbb552a3e8d899a7dd01d4530f0115c9b2ab6fefbb651feb122fdb274ccf75b'
NOW = RealDatetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
class FixedDatetime(RealDatetime):
    @classmethod
    def now(cls, tz=None): return NOW if tz is None else NOW.astimezone(tz)


def captured_board_bytes():
    selected = os.environ.get('LIVE_REFILL_CAPTURED_BOARD')
    if not selected:
        raise unittest.SkipTest('Existing owner must bind LIVE_REFILL_CAPTURED_BOARD for full17/no-skip qualification')
    raw = Path(selected).read_bytes()
    if len(raw) != 493809 or hashlib.sha256(raw).hexdigest() != BOARD_SHA:
        raise AssertionError('captured145board fixture identity mismatch')
    return raw


_HELPER_NAMES = (
    'task_id_prefix', 'task_id_pattern', 'task_ids_from_todo_text',
    'task_statuses_from_todo_text', 'state_statuses_match_todo_statuses',
    'open_task_count', 'effective_open_task_count', 'refill_state_counts',
    'load_json_dict', 'parse_iso_timestamp', 'should_refill_backlog',
    'task_header_prefix', '_refill_task_statuses_from_todo_text', '_refill_open_task_count',
)
HELPERS = {name: getattr(backlog_refinery, name) for name in _HELPER_NAMES}


class MemoryState:
    def __init__(self, value): self.raw = json.dumps(value, sort_keys=True)
    def exists(self): return True
    def read_text(self, **kwargs): return self.raw

def board(statuses, labels=None):
    if labels is None: labels = ['Status'] * len(statuses)
    return ''.join(f'## AUTO-{i+1:03d} fixture\n- {label}: {status}\n'
                   for i, (status, label) in enumerate(zip(statuses, labels)))

def strategy(count, time=None):
    return {'drained_count': count, 'last_scan': NOW.isoformat() if time is None else time,
            'spent_probes': 7, 'attempts': 3, 'mission_fingerprint': 'retained',
            'seen_fingerprints': ['a', 'b'], 'unrelated': {'budget': 11}}

class RefillRegressionTests(unittest.TestCase):
    def setUp(self):
        clock = patch.object(backlog_refinery, 'datetime', FixedDatetime)
        clock.start()
        self.addCleanup(clock.stop)

    def decide(self, text, saved, state=None, force=False, min_open=0):
        snapshot = copy.deepcopy(saved)
        state_before = None if state is None else state.raw
        result = HELPERS['should_refill_backlog'](
            todo_text=text, state_path=state, strategy=saved,
            last_scan_key='last_scan', last_drained_scan_task_count_key='drained_count',
            task_prefix='AUTO-', min_open_tasks=min_open, cooldown_seconds=60, force=force)
        self.assertEqual(saved, snapshot)
        if state is not None: self.assertEqual(state.raw, state_before)
        return result

    def test_saved_zero_is_a_real_count_and_respects_recent_cooldown(self):
        self.assertEqual(self.decide('', strategy(0)), (False, 'cooldown', 0, 0))

    def test_same_145_completed_count_respects_recent_cooldown(self):
        text = board(['completed'] * 145)
        self.assertEqual(self.decide(text, strategy(145)), (False, 'cooldown', 0, 145))

    def test_missing_count_still_scans_drained_frontier(self):
        saved = strategy(0); del saved['drained_count']
        self.assertEqual(self.decide('', saved), (True, 'drained_exhaustive', 0, 0))

    def test_changed_count_still_scans_and_keeps_probe_history(self):
        self.assertEqual(self.decide(board(['completed'] * 146), strategy(145)),
                         (True, 'drained_exhaustive', 0, 146))

    def test_force_and_expired_cooldown_semantics_are_unchanged(self):
        self.assertEqual(self.decide(board(['ready']), strategy(1), force=True),
                         (True, 'force', 1, 1))
        self.assertEqual(self.decide('', strategy(0, '2026-11-22T11:59:00+00:00')),
                         (True, 'low_backlog', 0, 0))
        self.assertEqual(self.decide('', strategy(0, '2026-11-22T11:59:59+00:00')),
                         (False, 'cooldown', 0, 0))

    def test_ready_and_todo_aliases_count_when_counters_are_missing(self):
        text = board(['todo', 'ready', 'waiting', 'completed'])
        statuses = {f'AUTO-{i+1:03d}': status for i, status in enumerate(['todo', 'READY', 'waiting', 'completed'])}
        memory = MemoryState({'task_statuses': statuses, 'task_count': 4})
        before = memory.raw
        result = HELPERS['refill_state_counts'](text, state_path=memory, task_prefix='AUTO-')
        for name in ('ready_count', 'selectable_ready_count', 'eligible_ready_count'):
            self.assertEqual(result[name], 2)
        self.assertEqual(memory.raw, before)

    def test_explicit_zero_ready_counters_remain_authoritative(self):
        text = board(['todo', 'ready', 'waiting', 'completed'])
        memory = MemoryState({'task_statuses': {'AUTO-001': 'todo', 'AUTO-002': 'ready',
                'AUTO-003': 'waiting', 'AUTO-004': 'completed'}, 'task_count': 4,
                'ready_count': 0, 'selectable_ready_count': 0, 'eligible_ready_count': 0})
        result = HELPERS['refill_state_counts'](text, state_path=memory, task_prefix='AUTO-')
        self.assertEqual([result[n] for n in ('ready_count', 'selectable_ready_count', 'eligible_ready_count')], [0, 0, 0])
        self.assertEqual(self.decide(text, strategy(-1), state=memory),
                         (True, 'runnable_drained_exhaustive', 3, 4))

    def test_ready_alias_prevents_false_runnable_drained_refill(self):
        text = board(['ready', 'waiting', 'completed'])
        memory = MemoryState({'task_statuses': {'AUTO-001': 'ready', 'AUTO-002': 'waiting',
                              'AUTO-003': 'completed'}, 'task_count': 3})
        self.assertEqual(self.decide(text, strategy(-1), state=memory),
                         (False, 'open_task_threshold', 2, 3))

    def test_mixed_case_status_lines_use_existing_numeric_heading_parser(self):
        statuses = ['completed', 'ready', 'todo', 'waiting']
        text = board(statuses, ['status', 'Status', 'STATUS', 'sTaTuS'])
        result = HELPERS['task_statuses_from_todo_text'](text, task_prefix='AUTO-')
        self.assertEqual(result, {f'AUTO-{i+1:03d}': status for i, status in enumerate(statuses)})
        self.assertEqual(HELPERS['open_task_count'](text, task_prefix='AUTO-'), 3)

    def test_status_parser_does_not_capture_other_sections_or_status_like_fields(self):
        text = '## AUTO-001 fixture\n- status-note: ready\n- status: completed\n## Goal overview\n- Status: ready\n'
        self.assertEqual(HELPERS['task_statuses_from_todo_text'](text, task_prefix='AUTO-'),
                         {'AUTO-001': 'completed'})

    def test_actual_145_board_numeric_subsets_and_prefix_limit_are_explicit(self):
        raw = captured_board_bytes()
        self.assertEqual(len(raw), 493809)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), BOARD_SHA)
        text = raw.decode()
        # Fixture census, NOT a replacement parser for the production API.
        physical = re.findall(r'^## (LIRAE-\S+)(?:\s|$)', text, re.MULTILINE)
        self.assertEqual(len(physical), 145)
        self.assertEqual(HELPERS['task_ids_from_todo_text'](text, task_prefix='LIRAE-'), [])
        for prefix, count in [('LIRAE-USC-T', 33), ('LIRAE-CONST-A', 7)]:
            with self.subTest(prefix=prefix):
                ids = HELPERS['task_ids_from_todo_text'](text, task_prefix=prefix)
                result = HELPERS['task_statuses_from_todo_text'](text, task_prefix=prefix)
                self.assertEqual(len(ids), count)
                self.assertEqual(set(result), set(ids))
                self.assertEqual(set(result.values()), {'completed'})
        self.assertEqual(hashlib.sha256(raw).hexdigest(), BOARD_SHA)


    def test_full_actual_145_refill_projection_and_same_count_cooldown(self):
        raw = captured_board_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), BOARD_SHA)
        text = raw.decode()
        result = HELPERS['_refill_task_statuses_from_todo_text'](text, task_prefix='LIRAE-')
        self.assertEqual(len(result), 145)
        self.assertEqual(set(result.values()), {'completed'})
        self.assertEqual(HELPERS['effective_open_task_count'](text, state_path=None, task_prefix='LIRAE-'), 0)
        saved = strategy(145); before = copy.deepcopy(saved)
        outcome = HELPERS['should_refill_backlog'](todo_text=text, state_path=None, strategy=saved,
            last_scan_key='last_scan', last_drained_scan_task_count_key='drained_count',
            task_prefix='LIRAE-', min_open_tasks=0, cooldown_seconds=60)
        self.assertEqual(outcome, (False, 'cooldown', 0, 145))
        self.assertEqual(saved, before)

    def test_refill_missing_status_and_duplicate_task_records_stay_unknown_open(self):
        text = ('## LIRAE-A-1 one\n- status: completed\n'
                '## LIRAE-B-1 missing status\n'
                '## LIRAE-A-1 repeated record\n- Status: completed\n')
        result = HELPERS['_refill_task_statuses_from_todo_text'](text, task_prefix='LIRAE-')
        self.assertEqual(result, {'LIRAE-A-1': 'unknown', 'LIRAE-B-1': 'unknown'})
        self.assertEqual(HELPERS['effective_open_task_count'](text, state_path=None, task_prefix='LIRAE-'), 2)

    def test_duplicate_status_fields_do_not_silently_close_uncertain_work(self):
        text = '## LIRAE-X one\n- Status: completed\n- status: ready\n'
        self.assertEqual(HELPERS['_refill_task_statuses_from_todo_text'](text, task_prefix='LIRAE-'),
                         {'LIRAE-X': 'unknown'})
        self.assertEqual(HELPERS['effective_open_task_count'](text, state_path=None, task_prefix='LIRAE-'), 1)

    def test_cached_completed_state_does_not_hide_missing_or_duplicate_board_status(self):
        texts = ['## LIRAE-X missing\n',
                 '## LIRAE-X first\n- Status: completed\n## LIRAE-X second\n- Status: completed\n',
                 '## LIRAE-X unknown\n- Status: unknown\n',
                 '## LIRAE-X malformed\n- status: unrecognized-status\n']
        for text in texts:
            with self.subTest(text=text):
                state = MemoryState({'task_count': 1, 'task_statuses': {'LIRAE-X': 'completed'}})
                before = state.raw
                self.assertEqual(HELPERS['effective_open_task_count'](text, state_path=state, task_prefix='LIRAE-'), 1)
                self.assertEqual(HELPERS['refill_state_counts'](text, state_path=state, task_prefix='LIRAE-'), {})
                self.assertEqual(state.raw, before)

    def test_valid_nonterminal_to_completed_cached_override_is_retained(self):
        text = '## LIRAE-X existing ready row\n- Status: ready\n'
        state = MemoryState({'task_count': 1, 'task_statuses': {'LIRAE-X': 'completed'},
                             'ready_count': 0, 'selectable_ready_count': 0, 'eligible_ready_count': 0})
        before = state.raw
        self.assertEqual(HELPERS['effective_open_task_count'](text, state_path=state, task_prefix='LIRAE-'), 0)
        counts = HELPERS['refill_state_counts'](text, state_path=state, task_prefix='LIRAE-')
        self.assertEqual(counts['completed_count'], 1)
        self.assertEqual([counts[n] for n in ('ready_count', 'selectable_ready_count', 'eligible_ready_count')], [0, 0, 0])
        self.assertEqual(state.raw, before)

    def test_numeric_allocator_retains_strict_id_contract(self):
        # Full pre/post AST preservation belongs to the separate sealed private
        # control. This normal-import test checks the maintained public API.
        self.assertEqual(HELPERS['task_ids_from_todo_text']('## AUTO-001 a\n## AUTO-001-P01 partition\n',
                                                         task_prefix='AUTO-'), ['AUTO-001'])
