"""Authored reports only; no providers, repository imports or private artifacts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import read_planning_ablation as added


def report():
    receipt = dict(schema=added.SCHEMA, planning_mode=added.MODE,
        **{key: 'cid:authored-' + key for key in added.IDS}, goals=2, tasks=1,
        pending_requirements=3, formal_plan_consistent=True, provider_calls=0, model_calls=0,
        source_obligations_discharged=0, **{key: False for key in added.FLAGS},
        interpretation_scope='exact_signed_task_structure_only')
    return dict(planning_mode=added.MODE, provider_invocations=[], planning=dict(
        schema='terminal-indexed-planner-result@1', qualified=True, goals=2, tasks=1,
        planning_mode=added.MODE, planning_strategy='declared_contract', provider_calls=0,
        provider_receipt=None, execution_receipt=None, model_request_sha256=None,
        provider_observation={}, task_cids=['cid:authored-task'],
        **{key: False for key in added.FLAGS}, declared_contract_planning=receipt,
        failure={'message': 'PRIVATE MODEL TEXT'}, provider='grok_cli', model='grok-4.7'))


class PlanningReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = added.baseline_reader()

    def observe(self, value, mode=added.MODE):
        return added.observe_planning(value, mode, self.base)

    def test_valid_declared_report_is_observation_only(self):
        value = report()
        result = self.observe(value)
        self.assertEqual(result['status'], 'validated_retained_declared_plan')
        self.assertTrue(result['declared_route_supported_by_retained_report'])
        self.assertEqual((result['goals'], result['tasks'], result['provider_calls_reported']), (2, 1, 0))
        self.assertEqual(result['receipt_sha256'], added.digest(self.base.canonical(value['planning']['declared_contract_planning'])))
        for key in ('proof_authority', 'execution_authority', 'completion_authority',
                    'independent_admission_replay_performed', 'total_token_savings_established'):
            self.assertIs(result[key], False)
        self.assertIsNone(result['planning_tokens_saved'])
        encoded = json.dumps(result)
        for forbidden in ('PRIVATE MODEL TEXT', 'cid:authored-', 'provider_observation', 'task_cids'):
            self.assertNotIn(forbidden, encoded)

    def test_no_planning_invocations_alone_cannot_establish_declared_execution(self):
        for value in (None, {}, {'provider_invocations': []}, {'provider_invocations': [], 'planning': {}}):
            with self.subTest(value=value):
                self.assertFalse(self.observe(value)['declared_route_supported_by_retained_report'])

    def test_unqualified_report_never_becomes_declared_success(self):
        value = report(); value['planning']['qualified'] = False
        result = self.observe(value)
        self.assertEqual(result['status'], 'unqualified')
        self.assertFalse(result['declared_route_supported_by_retained_report'])

    def test_closed_receipt_rejects_missing_extra_wrong_or_nonfinite_fields(self):
        mutations = [('remove', 'model_calls', None), ('set', 'extra', False),
            ('set', 'schema', 'unknown'), ('set', 'goals', True), ('set', 'tasks', 2),
            ('set', 'model_calls', False), ('set', 'provider_calls', 1),
            ('set', 'pending_requirements', True), ('set', 'pending_requirements', 0),
            ('set', 'pending_requirements', float('nan')), ('set', 'proof_authority', True),
            ('set', 'source_semantics_verified', True), ('set', 'formal_plan_consistent', False),
            ('set', 'manifest_cid', 'line\nbreak')]
        for action, key, replacement in mutations:
            with self.subTest(key=key, replacement=replacement):
                value = report(); receipt = value['planning']['declared_contract_planning']
                if action == 'remove': receipt.pop(key)
                else: receipt[key] = replacement
                self.assertEqual(self.observe(value)['status'], 'invalid')

    def test_plan_counts_modes_custody_and_zero_calls_must_agree(self):
        for key, replacement in [('qualified', 1), ('goals', True), ('tasks', 2), ('provider_calls', False),
            ('planning_mode', 'model'), ('planning_strategy', 'direct'), ('provider_receipt', {}),
            ('execution_receipt', {}), ('model_request_sha256', 'a'*64), ('provider_observation', {'usage': 0}),
            ('task_cids', []), ('completion_authority', True)]:
            with self.subTest(key=key):
                value = report(); value['planning'][key] = replacement
                self.assertEqual(self.observe(value)['status'], 'invalid')
        value = report(); value.pop('planning_mode')
        self.assertEqual(self.observe(value)['status'], 'invalid')
        value = report(); value['planning'].pop('execution_receipt')
        self.assertEqual(self.observe(value)['status'], 'invalid')

    def test_contradictory_or_conflicting_native_receipts_reject(self):
        value = report(); value['provider_invocations'] = [dict(invocation_id='a'*32, phase='planning')]
        self.assertEqual(self.observe(value)['status'], 'invalid')
        for phase in (None, 'unknown', True):
            value['provider_invocations'][0]['phase'] = phase
            self.assertEqual(self.observe(value)['status'], 'invalid')
        value['provider_invocations'] = [dict(invocation_id='a'*32, phase='coding', router_calls=1),
            dict(invocation_id='a'*32, phase='coding', router_calls=2)]
        self.assertEqual(self.observe(value)['status'], 'invalid')

    def test_coding_receipts_do_not_imply_planning_generation(self):
        value = report(); value['provider_invocations'] = [dict(invocation_id='a'*32, phase='coding')]
        self.assertTrue(self.observe(value)['declared_route_supported_by_retained_report'])

    def test_model_default_and_invalid_controls_are_explicit(self):
        self.assertEqual(added.planning_control({'agents': [{'kwargs': {}}]}), 'model')
        for value in (None, {}, {'agents': []}, {'agents': [{'kwargs': {'planning_mode': True}}]},
                      {'agents': [{'kwargs': {'planning_mode': 'private text'}}]}):
            self.assertIsNone(added.planning_control(value))
        self.assertEqual(self.observe({}, 'model')['status'], 'not_selected')
        self.assertEqual(self.observe(report(), 'model')['status'], 'invalid')
        self.assertFalse(self.observe(report(), None)['declared_route_supported_by_retained_report'])

    def test_additive_projection_preserves_baseline_and_exposes_changed_control(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def write_run(name, mode, value):
                run = root / name
                parent = run / 'jobs/supervisor-full-largest-eigenval/largest-eigenval__authored'
                (parent / 'agent').mkdir(parents=True)
                kwargs = {} if mode == 'model' else {'planning_mode': mode}
                (run / 'config.json').write_text(json.dumps({'agents': [{'kwargs': kwargs}]}))
                (parent / 'result.json').write_text(json.dumps({'verifier_result': {'rewards': {'reward': 0}}}))
                (parent / 'agent/supervisor-result.json').write_text(json.dumps(value))
                return run
            pilot = write_run('declared', added.MODE, report())
            prior = write_run('model', 'model', {'provider_invocations': []})
            recipe = root / 'recipe.json'
            recipe.write_text(json.dumps({'schema': 'doctor-dispatch-live-pilot@1', 'output': str(pilot)}))
            result = added.read_comparison(recipe, prior_directory=prior)
            self.assertEqual(result['baseline'], {'pilot': self.base.read_run(pilot), 'prior': self.base.read_run(prior)})
            self.assertEqual(result['differing_planning_control'], {'planning_mode': {'pilot': added.MODE, 'prior': 'model'}})
            self.assertTrue(result['planning']['pilot']['trials'][0]['planning']['declared_route_supported_by_retained_report'])
            self.assertIs(result['total_token_savings_established'], False)
            self.assertNotIn('PRIVATE MODEL TEXT', json.dumps(result))

    def test_file_change_between_baseline_and_additive_reads_rejects(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root / 'config.json'
            path.write_text(json.dumps({'agents': [{'kwargs': {}}]}))
            baseline = self.base.read_run(root)
            path.write_text(json.dumps({'agents': [{'kwargs': {'planning_mode': added.MODE}}]}))
            with self.assertRaisesRegex(ValueError, 'retained_artifact_changed'):
                added.planning_run(root, baseline, self.base)


if __name__ == '__main__':
    unittest.main()
