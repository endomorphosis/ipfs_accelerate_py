"""Authored local orchestration contracts, no models, Docker or preparation."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

A = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A))
import campaign_common as c
import run_locked_trial as runner


class CampaignTests(unittest.TestCase):
    def test_recipe_has_exact_budget_treatment_and_fresh_outputs(self):
        recipe = c.recipe()
        self.assertEqual(recipe['host_disk_prerequisite']['minimum_available_bytes'], 32 * 1024**3)
        self.assertEqual(recipe['host_disk_prerequisite']['mode'], 'minimum-free-bytes')
        a, b = [recipe['arms'][label] for label in c.LABELS]
        self.assertNotIn('--capsule-budget', a['prepare_argv'])
        self.assertEqual(b['prepare_argv'][-2:], ['--capsule-budget', 'one-capsule@1'])
        def normalized(row):
            command = list(row['prepare_argv'])
            command[command.index('--output')+1] = '<out>'
            if '--capsule-budget' in command:
                del command[-2:]
            return command
        self.assertEqual(normalized(a), normalized(b))
        self.assertEqual(a['source_view_profile'], b['source_view_profile'])

    def test_safe_environment_scrubs_provider_keys_without_logging_values(self):
        with patch.dict(c.os.environ, {'OPENAI_API_KEY':'authored-secret','XAI_API_KEY':'authored-secret',
            'HF_TOKEN':'authored-secret','OPENAI_BASE_URL':'authored-secret'}, clear=False):
            value = c.safe_environment()
        for key in ('OPENAI_API_KEY','XAI_API_KEY','HF_TOKEN','OPENAI_BASE_URL'):
            self.assertNotIn(key, value)
        self.assertEqual(value['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(value['HF_HUB_OFFLINE'], '1')

    def test_disk_rounding_observation_does_not_claim_active_native_gate(self):
        fake = SimpleNamespace(disk_usage=lambda _: SimpleNamespace(percent=94.8,free=100 * 1024**3,total=4 * 1024**4))
        with patch.dict(sys.modules, {'psutil':fake}):
            value = c.disk_admission_snapshot()
        self.assertEqual(value['scheduler_disk_percent'], 95)
        self.assertTrue(value['default_threshold_exceeded'])
        self.assertFalse(value['default_percent_threshold_applies_to_benchmark_start'])
        self.assertFalse(value['native_start_admission_observed'])
        self.assertFalse(value['reserves_capacity'])

    def test_physical_storage_gate_accepts_high_percent_with_enough_bytes(self):
        with tempfile.TemporaryDirectory(dir=A / 'tests') as directory:
            fake = dict(disk_available_bytes=32 * 1024**3,default_threshold_exceeded=True)
            with patch.object(c, 'disk_admission_snapshot', return_value=fake):
                row = c.record_admission_gate(Path(directory), {'mode':'minimum-free-bytes',
                    'minimum_available_bytes':32 * 1024**3})
            self.assertTrue(c.read(row['path'])['root_local_prerequisite_passed'])

    def test_low_available_bytes_refuse_and_retain_observation(self):
        with tempfile.TemporaryDirectory(dir=A / 'tests') as directory:
            fake = dict(disk_available_bytes=32 * 1024**3-1,default_threshold_exceeded=False)
            with patch.object(c, 'disk_admission_snapshot', return_value=fake):
                with self.assertRaisesRegex(ValueError, 'root-local disk prerequisite refused'):
                    c.record_admission_gate(Path(directory), {'mode':'minimum-free-bytes',
                        'minimum_available_bytes':32 * 1024**3})
            self.assertFalse(c.read(Path(directory) / 'host-disk-001.json')['root_local_prerequisite_passed'])

    def test_gate_observations_cannot_overwrite_prior_failures(self):
        with tempfile.TemporaryDirectory(dir=A / 'tests') as directory:
            with patch.object(c, 'disk_admission_snapshot', return_value=dict(disk_available_bytes=10,default_threshold_exceeded=False)):
                first = c.record_admission_gate(Path(directory), {'mode':'observation-only'})
                second = c.record_admission_gate(Path(directory), {'mode':'observation-only'})
            self.assertNotEqual(first['path'], second['path'])
            c.check_binding(first)
            self.assertIsNone(c.read(first['path'])['root_local_prerequisite_passed'])

    def test_no_frozen_lock_cannot_launch(self):
        with tempfile.TemporaryDirectory(dir=A / 'tests') as directory:
            with patch.object(runner, 'A', Path(directory)), patch.object(runner.subprocess, 'run') as process:
                with self.assertRaises(ValueError):
                    runner.execute('legacy')
                process.assert_not_called()

    def test_full_configuration_difference_detects_new_control(self):
        a = {'agents':[{'kwargs':{'source_view_profile':'identity@1'}}], 'jobs_dir':'a'}
        b = {'agents':[{'kwargs':{'source_view_profile':'identity@1','capsule_budget':'one-capsule@1'}}], 'jobs_dir':'b'}
        self.assertEqual(c.differences(a,b), ['/agents/0/kwargs/capsule_budget','/jobs_dir'])
        b['agents'][0]['kwargs']['planning_mode'] = 'declared-contract@1'
        self.assertIn('/agents/0/kwargs/planning_mode', c.differences(a,b))

    def test_binding_change_is_refused(self):
        with tempfile.TemporaryDirectory(dir=A / 'tests') as directory:
            path = Path(directory) / 'value.json'; c.write_new(path, {'authored':1})
            before = c.binding(path); path.write_text('{"authored":2}\n')
            with self.assertRaisesRegex(ValueError, 'frozen artifact changed'):
                c.check_binding(before)

    def test_closed_json_refuses_duplicates_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory(dir=A / 'tests') as directory:
            path = Path(directory) / 'value.json'
            for value in ('{"x":1,"x":2}', '{"x":NaN}'):
                path.write_text(value)
                with self.assertRaises(ValueError):
                    c.read(path)

    def test_reporting_freeze_requires_exact_reader_and_current_helper_bytes(self):
        with tempfile.TemporaryDirectory(dir=A / 'tests') as directory:
            root = Path(directory)
            reader = root / 'read_live_pair.py'; reader.write_text('# authored reader fixture\n')
            helper = root / 'helper.py'; helper.write_text('# authored helper fixture\n')
            result = root / 'test-result.json'; c.write_new(result, {'exit_code':0})
            manifest = root / 'reporting-freeze.json'
            c.write_new(manifest, {'schema':'capsule-budget-live-reporting-code-freeze@1',
                'files':[c.binding(reader),c.binding(helper)],'validation':{'test_result':c.binding(result)}})
            with patch.object(c, 'A', root):
                self.assertEqual(c.reporting_freeze_binding(manifest), c.binding(manifest))
                helper.write_text('# changed helper\n')
                with self.assertRaisesRegex(ValueError, 'frozen artifact changed'):
                    c.reporting_freeze_binding(manifest)

    def test_reporting_freeze_missing_validation_refuses_before_launch(self):
        with tempfile.TemporaryDirectory(dir=A / 'tests') as directory:
            root = Path(directory)
            reader = root / 'read_live_pair.py'; reader.write_text('# authored reader fixture\n')
            manifest = root / 'reporting-freeze.json'
            c.write_new(manifest, {'schema':'capsule-budget-live-reporting-code-freeze@1',
                'files':[c.binding(reader)]})
            with patch.object(c, 'A', root):
                with self.assertRaisesRegex(ValueError, 'reporting validation binding required'):
                    c.reporting_freeze_binding(manifest)


if __name__ == '__main__':
    unittest.main()
