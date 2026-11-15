"""Pure retained-artifact cases; no supervisor, Docker or provider imports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import read_pilot as reader


def receipt(n=1, phase='coding'):
    return dict(invocation_id=str(n)*32, phase=phase, provider='grok_cli', model='grok-4.7',
        schema='router-implementation-invocation@1', router_calls=1,
        native_rollout_usage=dict(schema='native-grok-final-usage@1', usage_available=True,
            usage_complete_observed=True, task_complete_observed=True, totals_scope='native_final_envelope',
            cache_included_in_input=False, billing_total_verified=False,
            usage=dict(input_tokens=10, cached_input_tokens=20, cache_write_input_tokens=3,
                output_tokens=4, total_tokens=37, reasoning_output_tokens=2)))


def two_round_report():
    first, second = receipt(), receipt(2)
    first['source_view_retrieval'] = dict(session_id='a'*32, descriptor_sha256='b'*64,
        task_id='task', native_prompt_sha256='c'*64)
    first['doctor_residual_context'] = dict(artifact_sha256='f'*64, advisory_sha256='e'*64)
    second['candidate_round_dispatch'] = dict(schema='supervisor-candidate-dispatch-observation@2',
        round_number=2, session_id='d'*32, prior_session_id='a'*32, prior_descriptor_sha256='b'*64,
        task_id='task', native_prompt_sha256='c'*64,
        doctor_candidate_dispatch=dict(policy_id='doctor-coding-two-attempts@1',
            coding_policy_sha256='d'*64, doctor_artifact_sha256='f'*64, doctor_advisory_sha256='e'*64))
    accounting = dict(schema='terminal-candidate-attempt-accounting@1', status='matched_round_receipts',
        potential_unreceipted_attempt=False, doctor_attempts_reserved=[1, 2],
        doctor_coding_policy='doctor-coding-two-attempts@1', doctor_coding_policy_sha256='d'*64,
        complete_usage_established=False, rounds=[dict(round_number=n, process_started=True,
            receipt_observed=True, possible_unreceipted_attempt=False) for n in (1, 2)])
    return dict(provider_invocations=[receipt(3, 'planning'), first, second], candidate_attempt_accounting=accounting)


def complete_aggregate():
    return dict(input_tokens=30, cached_input_tokens=60, cache_write_input_tokens=9,
        output_tokens=12, total_tokens=111, native_scoped_usage_complete_observed=True,
        all_invocations_receipted=True)


class ReaderTests(unittest.TestCase):
    def test_phase_timings_are_closed_numeric_fields(self):
        result = reader.summarize_report(dict(phases={
            'AUTHORED raw prompt text with spaces': 1,
            'planning_seconds': 2.5, 'doctor_seconds': 'PRIVATE',
            'prepare_seconds': True, 'context_seconds': float('inf')}))
        self.assertEqual(result['phase_seconds'], {'planning_seconds': 2.5})
        self.assertNotIn('AUTHORED', json.dumps(result))
        self.assertNotIn('PRIVATE', json.dumps(result))

    def test_no_receipts_are_unknown_not_zero(self):
        result = reader.summarize_report({})
        self.assertIsNone(result['all_phase_usage']['observed_subtotals']['total_tokens'])
        self.assertIsNone(result['all_phase_usage']['outer_router_calls'])
        self.assertFalse(result['all_phase_usage']['all_native_contracts_complete'])

    def test_missing_and_null_categories_keep_known_subtotal(self):
        for missing in ('absent', 'null', 'bool'):
            row = receipt(); usage = row['native_rollout_usage']['usage']
            if missing == 'absent': usage.pop('input_tokens')
            else: usage['input_tokens'] = None if missing == 'null' else True
            result = reader.aggregate([row, receipt(2)])
            self.assertEqual(result['observed_subtotals']['input_tokens'], 10)
            self.assertEqual(result['missing_categories']['input_tokens'], 1)
            self.assertIsNone(result['complete_native_totals']['total_tokens'])

    def test_final_envelope_does_not_fill_omitted_completeness(self):
        row = receipt(); row['native_rollout_usage'].pop('usage_complete_observed')
        result = reader.aggregate([row])
        self.assertEqual(result['observed_subtotals']['total_tokens'], 37)
        self.assertIsNone(reader.native(row)['usage_complete_under_native_contract'])
        self.assertIsNone(result['complete_native_totals']['total_tokens'])

    def test_identical_duplicates_count_once(self):
        row = receipt(); result = reader.summarize_report(dict(provider_invocations=[row, copy.deepcopy(row)]))
        self.assertEqual(result['duplicate_receipts_ignored'], 1)
        self.assertEqual(result['all_phase_usage']['observed_subtotals']['total_tokens'], 37)

    def test_conflicting_duplicate_refuses_aggregation(self):
        a, b = receipt(), receipt(); b['native_rollout_usage']['usage']['output_tokens'] = 99
        with self.assertRaisesRegex(ValueError, 'conflicting_invocation_receipts'):
            reader.summarize_report(dict(provider_invocations=[a, b]))

    def test_two_rounds_and_planning_count_separately(self):
        result = reader.summarize_report(two_round_report(), complete_aggregate())
        self.assertEqual(result['usage_by_phase']['coding']['outer_router_calls'], 2)
        self.assertEqual(result['usage_by_phase']['coding']['observed_subtotals']['total_tokens'], 74)
        self.assertEqual(result['all_phase_usage']['complete_native_totals']['total_tokens'], 111)
        self.assertEqual(result['all_phase_usage']['observed_subtotals']['reasoning_output_tokens'], 6)
        self.assertTrue(result['continuation']['continuation_supported_by_retained_report'])
        self.assertFalse(result['continuation']['independent_owner_replay_performed'])
        self.assertFalse(result['continuation']['native_accounting']['complete_usage_established'])

    def test_aggregate_completeness_disagreement_keeps_totals_unknown(self):
        for variant in ('missing', 'unknown', 'incomplete', 'counter_mismatch', 'unreceipted'):
            aggregate = complete_aggregate()
            if variant == 'missing': aggregate = None
            elif variant == 'unknown': aggregate['native_scoped_usage_complete_observed'] = None
            elif variant == 'incomplete': aggregate['native_scoped_usage_complete_observed'] = False
            elif variant == 'counter_mismatch': aggregate['total_tokens'] += 1
            else: aggregate['all_invocations_receipted'] = False
            result = reader.summarize_report(two_round_report(), aggregate)
            self.assertEqual(result['all_phase_usage']['observed_subtotals']['total_tokens'], 111)
            self.assertIsNone(result['all_phase_usage']['complete_native_totals']['total_tokens'])
            self.assertIsNone(result['usage_by_phase']['coding']['complete_native_totals']['total_tokens'])

    def test_reservations_alone_and_preprovider_failure_are_not_two_calls(self):
        for change in ('no_second_call', 'no_join', 'missing_first', 'bool_reservation', 'foreign_doctor'):
            report = two_round_report()
            if change == 'no_second_call': report['provider_invocations'][2]['router_calls'] = 0
            elif change == 'no_join': report['provider_invocations'][2]['candidate_round_dispatch']['prior_descriptor_sha256'] = None
            elif change == 'missing_first': report['provider_invocations'].pop(1)
            elif change == 'bool_reservation': report['candidate_attempt_accounting']['doctor_attempts_reserved'] = [True, 2]
            else: report['provider_invocations'][1]['doctor_residual_context']['artifact_sha256'] = 'a'*64
            self.assertFalse(reader.summarize_report(report)['continuation']['continuation_supported_by_retained_report'])

    def test_unreceipted_attempt_suppresses_complete_not_observed(self):
        report = two_round_report(); report['unreceipted_provider_attempt'] = dict(usage=None)
        result = reader.summarize_report(report)['all_phase_usage']
        self.assertEqual(result['observed_subtotals']['total_tokens'], 111)
        self.assertIsNone(result['complete_native_totals']['total_tokens'])

    def test_failure_usage_retained_and_raw_text_excluded(self):
        row = receipt(); row.update(status='failed', error='PRIVATE PROVIDER TEXT', prompt='PRIVATE PROMPT')
        row['native_rollout_usage']['task_complete_observed'] = False
        report = dict(provider_invocations=[row], auth={'secret': 'PRIVATE CREDENTIAL'})
        result = reader.summarize_report(report)
        self.assertNotIn('PRIVATE', json.dumps(result))
        self.assertEqual(result['usage_by_phase']['coding']['observed_subtotals']['total_tokens'], 37)
        self.assertIsNone(result['usage_by_phase']['coding']['complete_native_totals']['total_tokens'])

    def test_actual_reward_is_independent_from_supervisor_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); trial = root/'jobs/supervisor-full-largest-eigenval/largest-eigenval__sample'
            (trial/'agent').mkdir(parents=True)
            (trial/'result.json').write_text(json.dumps(dict(verifier_result=dict(rewards=dict(reward=0)))))
            (trial/'agent/supervisor-result.json').write_text(json.dumps(dict(task_completed=True, provider_invocations=[receipt()])))
            row = reader.read_run(root)['trials'][0]
            self.assertEqual(row['official_reward'], 0)
            self.assertTrue(row['task_completed_reported'])


if __name__ == '__main__':
    unittest.main()
