"""Artifact-only regression checks: original historical receipts and authored mutations."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pair_collector',HERE/'collect_pair.py')
collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)
OLD=Path('/home/barberb/lift_coding/artifacts/source-context-substitution-20261013/live-trials')

class CollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt,cls.binding=collector.read(OLD/'codex-legacy-01/receipt.json')
        cls.report=cls.receipt['trials'][0]['supervisor']

    def test_original_historical_pairs_recompute_native_usage_without_relabelling(self):
        for name,total,reward in [('codex-legacy-01',313979,0.0),('codex-projected-01',313980,1.0)]:
            with self.subTest(name=name):
                receipt,binding=collector.read(OLD/name/'receipt.json')
                row=collector.project_trial(receipt,receipt['trials'][0],binding)
                self.assertEqual(row['arm'],'full')
                self.assertEqual(row['official_reward'],reward)
                self.assertEqual(row['usage']['tokens']['total_tokens'],total)
                self.assertEqual(row['usage']['tokens']['total_tokens'],row['usage']['tokens']['input_tokens']+row['usage']['tokens']['output_tokens'])
                self.assertTrue(row['usage']['cache_included_in_input'])
                self.assertEqual(row['source_retrieval']['counts']['accepted_requests'],0)
                self.assertEqual(row['source384']['coverage']['selected_units'],4)
                self.assertEqual(row['source384']['coverage']['candidate_statuses'],{'fail_open_source_contract_unsupported':4})
                self.assertEqual(row['declared_output_observation']['status'],'unavailable')
                self.assertEqual(row['usage']['observed_router_sessions'],2)

    def test_original_infrastructure_failure_retained_with_unknown_usage(self):
        receipt,binding=collector.read(OLD/'grok-legacy-02/receipt.json')
        row=collector.project_trial(receipt,receipt['trials'][0],binding)
        self.assertEqual(row['exception_type'],'GrokCredentialReadinessError')
        self.assertEqual(row['usage']['observed_router_sessions'],0)
        self.assertIsNone(row['usage']['tokens']['total_tokens'])
        self.assertIsNone(row['official_reward'])

    def test_identical_duplicate_receipt_is_counted_once(self):
        report=deepcopy(self.report);report['provider_invocations'].append(deepcopy(report['provider_invocations'][0]))
        value=collector.usage_projection(report)
        self.assertEqual(value['observed_router_sessions'],2)
        self.assertEqual(value['tokens']['total_tokens'],313979)

    def test_conflicting_duplicate_identity_refused(self):
        report=deepcopy(self.report);duplicate=deepcopy(report['provider_invocations'][0]);duplicate['seconds']+=1
        report['provider_invocations'].append(duplicate)
        with self.assertRaisesRegex(ValueError,'conflicting duplicate'):collector.usage_projection(report)

    def test_inconsistent_codex_usage_never_promoted_or_max_selected(self):
        for change in ['total','cache','bool','missing','unreceipted']:
            with self.subTest(change=change):
                report=deepcopy(self.report);usage=report['provider_invocations'][0]['native_rollout_usage']['usage']
                if change=='total':usage['total_tokens']+=1
                elif change=='cache':usage['cached_input_tokens']=usage['input_tokens']+1
                elif change=='bool':usage['input_tokens']=True
                elif change=='missing':del usage['output_tokens']
                else:report['unreceipted_provider_attempt']={'phase':'coding'}
                value=collector.usage_projection(report)
                self.assertIsNone(value['tokens']['total_tokens'])
                self.assertFalse(value['native_usage_complete_observed'])

    def test_grok_cache_semantics_and_missing_completeness_preserved(self):
        row=dict(invocation_id='a'*32,provider='grok_cli',native_rollout_usage=dict(
            schema='native-grok-final-usage@1',task_complete_observed=True,usage_complete_observed=True,
            usage_available=True,totals_scope='native_final_envelope',cache_included_in_input=False,
            usage=dict(input_tokens=10,cached_input_tokens=5,cache_write_input_tokens=3,output_tokens=2,total_tokens=20)))
        value=collector.usage_projection({'provider_invocations':[row]})
        self.assertEqual(value['tokens']['total_tokens'],20);self.assertFalse(value['cache_included_in_input'])
        del row['native_rollout_usage']['usage_complete_observed']
        self.assertIsNone(collector.usage_projection({'provider_invocations':[row]})['tokens']['total_tokens'])

    def test_invocation_route_mismatch_refused(self):
        receipt=deepcopy(self.receipt);receipt['trials'][0]['supervisor']['provider_invocations'][0]['model']='other'
        with self.assertRaisesRegex(ValueError,'invocation route differs'):collector.project_trial(receipt,receipt['trials'][0],self.binding)

    def test_wrong_task_or_arm_refused(self):
        for field in ['task','arm','exact_trial_task_matches']:
            with self.subTest(field=field):
                receipt=deepcopy(self.receipt);trial=receipt['trials'][0]
                if field=='arm':trial['supervisor']['arm']='no-index'
                elif field=='task':trial['task']='other'
                else:trial[field]=False
                with self.assertRaises(ValueError):collector.project_trial(receipt,trial,self.binding)

    def test_absent_retrieval_is_unknown_not_zero_and_raw_fields_not_exported(self):
        self.assertIsNone(collector.retrieval_projection(None))
        observation=dict(schema='terminal-numerical-retrieval-observation@1',status='not_selected',counts=None,sessions=[],raw_query='private text',error='private path')
        projected=collector.retrieval_projection(observation)
        self.assertIsNone(projected['counts']);self.assertNotIn('raw_query',projected);self.assertNotIn('error',projected)

    def test_retrieval_counter_types_refused(self):
        for value in [True,-1,'5']:
            with self.subTest(value=value),self.assertRaises(ValueError):
                collector.retrieval_projection(dict(counts={'accepted_requests':value}))

    def test_duplicate_json_symlinks_and_size_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'receipt.json';path.write_text('{"a":1,"a":2}')
            with self.assertRaisesRegex(ValueError,'duplicate'):collector.read(path)
            path.write_text('{}');(root/'link.json').symlink_to(path)
            with self.assertRaisesRegex(ValueError,'canonical'):collector.read(root/'link.json')
            with self.assertRaisesRegex(ValueError,'bounded'):collector.read(path,maximum=1)

    def test_frozen_preparation_mismatch_refused_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);prep,_=collector.read(OLD/'codex-legacy-01/preparation.json');config,_=collector.read(OLD/'codex-legacy-01/config.json')
            config['agents'][0]['kwargs']['resource_profile']='changed'
            (root/'config.json').write_text(json.dumps(config));(root/'preparation.json').write_text(json.dumps(prep))
            with self.assertRaisesRegex(ValueError,'configuration differs'):collector.prepared(root,'full')

    def test_historical_run_cannot_be_frozen_as_fresh(self):
        with self.assertRaisesRegex(ValueError,'freeze must precede launch'):
            collector.prepared(OLD/'codex-legacy-01','full',require_unlaunched=True)

    def test_scope_diff_exposes_unintended_changes(self):
        self.assertEqual(collector.differences({'provider':'a'},{'provider':'b'}),['/provider'])
        self.assertNotIn('/provider',collector.CONFIG_DIFFS)

    def test_unknown_numerical_scalar_and_control_fields_not_exported(self):
        receipt=deepcopy(self.receipt);report=receipt['trials'][0]['supervisor']
        report.setdefault('initial_context',{})['numerical_formalization']={'summary':{'schema':'terminal-numerical-formalization-planning-summary@1','private_source_text':'do not export me','files':2}}
        row=collector.project_trial(receipt,receipt['trials'][0],self.binding)
        self.assertNotIn('private_source_text',row['numerical_coverage']['summary'])
        prep,_=collector.read(OLD/'codex-legacy-01/preparation.json')
        control=deepcopy(receipt['comparison_controls']);control['private_source_text']='do not export me'
        result=collector.controls_projection(control,prep['comparison_controls'])
        self.assertTrue(result['matched_controls']);self.assertNotIn('private_source_text',result)
        control['configuration_unchanged']=False
        self.assertFalse(collector.controls_projection(control,prep['comparison_controls'])['matched_controls'])

    def test_actual_unlaunched_pair_freezes_only_declared_differences(self):
        lock=collector.freeze([HERE/'live-trials/codex-full-01'],[HERE/'live-trials/codex-no-index-01'])
        self.assertEqual(set(lock['launch_inputs']),{'runner','recipe','environment'})
        self.assertEqual(lock['attempts'][1]['configuration_difference_paths'],['/agents/0/kwargs/arm','/job_name','/jobs_dir'])
        result=collector.collect(lock)
        self.assertEqual(len(result['attempts']),2)
        self.assertFalse(result['matched_controls'])
        self.assertTrue(all(v is None for v in result['full_minus_no_index_tokens'].values()))
        self.assertTrue(all(a['status']=='awaiting_receipt' for a in result['attempts']))

    def test_launch_start_must_join_exact_lock_recipe_runner_environment(self):
        launch_inputs=collector.launch_inputs();recipe,_=collector.read(launch_inputs['recipe']['path'])
        prep,config,row=collector.prepared(HERE/'live-trials/codex-full-01','full')
        lock={'launch_inputs':launch_inputs,'common':{'provider_profile':prep['provider_profile']}}
        binding={'sha256':'1'*64}
        start=dict(schema='numerical-live-pair-launch@1',arm='full',pair_lock_sha256=binding['sha256'],
            recipe_sha256=launch_inputs['recipe']['sha256'],script_sha256=launch_inputs['runner']['sha256'],
            environment_manifest_sha256=launch_inputs['environment']['sha256'],preparation_sha256=row['preparation']['sha256'],
            inherited_environment_values_retained=False,cwd=recipe['cwd'],argv=recipe['commands_by_provider'][prep['provider_profile']]['full']['execute_argv'])
        collector.validate_launch_start(start,lock=lock,lock_binding=binding,declared=row)
        for key in ('pair_lock_sha256','recipe_sha256','script_sha256','environment_manifest_sha256','preparation_sha256','arm','argv'):
            with self.subTest(key=key),self.assertRaises(ValueError):
                changed=deepcopy(start);changed[key]='changed'
                collector.validate_launch_start(changed,lock=lock,lock_binding=binding,declared=row)

    def _authored_collection(self, *, reused_identity=False, mismatched_controls=False):
        # Authored in-memory mutations; original historical artifacts stay unchanged.
        lock=collector.freeze([HERE/'live-trials/codex-full-01'],[HERE/'live-trials/codex-no-index-01'])
        lock_binding={'sha256':'1'*64};recipe,_=collector.read(lock['launch_inputs']['recipe']['path'])
        original_read=collector.read;fake={}
        for arm_index,declared in enumerate(lock['attempts']):
            base=Path(declared['directory']);prep,_=collector.read(base/'preparation.json')
            receipt=deepcopy(self.receipt);receipt.update(arm=declared['arm'],
                numerical_formalization_profile=prep['numerical_formalization_profile'],numerical_formalization_config_unchanged=True,
                comparison_controls={'status':'observed','reason':None,'declared':prep['comparison_controls'],'configuration_unchanged':True})
            trial=receipt['trials'][0];trial['trial']='authored-'+declared['arm'];trial['supervisor']['arm']=declared['arm']
            if arm_index and not reused_identity:
                for index,row in enumerate(trial['supervisor']['provider_invocations']):row['invocation_id']=format(index+20,'032x')
            if mismatched_controls and arm_index:receipt['original_task_inputs_unchanged']=False
            digest=hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest()
            start=dict(schema='numerical-live-pair-launch@1',arm=declared['arm'],pair_lock_sha256=lock_binding['sha256'],
                recipe_sha256=lock['launch_inputs']['recipe']['sha256'],script_sha256=lock['launch_inputs']['runner']['sha256'],
                environment_manifest_sha256=lock['launch_inputs']['environment']['sha256'],preparation_sha256=declared['preparation']['sha256'],
                inherited_environment_values_retained=False,cwd=recipe['cwd'],argv=recipe['commands_by_provider'][prep['provider_profile']][declared['arm']]['execute_argv'])
            result=dict(schema='numerical-live-pair-launch-result@1',arm=declared['arm'],receipt_sha256=digest,preparation_unchanged=True)
            for name,value in [('receipt.json',receipt),('launch-start.json',start),('launch-result.json',result)]:
                fake[str(base/name)]=(value,{'path':str(base/name),'availability':'observed','sha256':digest,'bytes':1})
        def read(path,**kwargs):return fake[str(path)] if str(path) in fake else original_read(path,**kwargs)
        with patch.object(collector,'read',read):return collector.collect(lock,lock_binding=lock_binding)

    def test_cross_arm_reused_invocation_identity_refused(self):
        with self.assertRaisesRegex(ValueError,'across registered pair'):self._authored_collection(reused_identity=True)

    def test_control_mismatch_retains_usage_but_suppresses_pair_delta(self):
        value=self._authored_collection(mismatched_controls=True)
        self.assertFalse(value['matched_controls'])
        self.assertEqual(value['arms']['full']['tokens']['total_tokens'],313979)
        self.assertEqual(value['arms']['no-index']['tokens']['total_tokens'],313979)
        self.assertTrue(all(x is None for x in value['full_minus_no_index_tokens'].values()))

    def test_matching_authored_pair_retains_all_trials_and_delta(self):
        value=self._authored_collection()
        self.assertTrue(value['matched_controls']);self.assertEqual(len(value['attempts']),2)
        self.assertEqual(value['full_minus_no_index_tokens']['total_tokens'],0)

if __name__=='__main__':unittest.main(verbosity=2)
