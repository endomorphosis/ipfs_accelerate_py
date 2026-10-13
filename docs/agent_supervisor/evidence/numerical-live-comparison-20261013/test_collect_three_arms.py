"""Historical accounting and adversarial metadata tests; no generation or transcripts."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('three',HERE/'collect_three_arms.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
OLD=HERE.parent/'terminal-codex-eigenvalue-baseline-20261005/native-01/receipt.json'

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.receipt=json.loads(OLD.read_bytes());cls.trial=cls.receipt['trials'][0]
  cls.native=json.loads((HERE/'next-native-baseline/native-codex-01/preparation.json').read_bytes())['comparison_controls']
  cls.supervisor=json.loads((HERE/'live-trials/codex-full-01/preparation.json').read_bytes())['comparison_controls']
 def usage(self,raw):return c.native_usage(raw,model='gpt-6.1-sol',effort='high',cli_version='0.160.0')
 def test_actual_historical_cumulative_usage_no_cache_double_count(self):
  row=c.native_trial(self.receipt,self.trial)
  self.assertTrue(row['usage']['native_usage_complete_observed'])
  self.assertEqual(row['usage']['tokens'],dict(input_tokens=154292,cached_input_tokens=132608,output_tokens=7458,total_tokens=161750))
  self.assertEqual(row['usage']['observed_native_sessions'],1)
 def test_event_snapshots_do_not_multiply(self):
  raw=deepcopy(self.trial['raw_usage']);raw['sessions'][0]['token_count_events']=99999
  self.assertEqual(self.usage(raw)['tokens']['total_tokens'],161750)
 def test_missing_usage_remains_unknown(self):
  v=self.usage({});self.assertEqual(v['observed_native_sessions'],0);self.assertTrue(all(x is None for x in v['tokens'].values()))
 def test_incomplete_native_session_retains_observed_counts(self):
  raw=deepcopy(self.trial['raw_usage']);raw['sessions'][0]['task_complete_observed']=False
  v=self.usage(raw);self.assertIsNone(v['tokens']['total_tokens']);self.assertEqual(v['reported_counts']['total_tokens'],161750)
 def test_missing_exact_completion_event_unknown(self):
  raw=deepcopy(self.trial['raw_usage']);raw['sessions'][0].pop('task_complete_event_observed')
  self.assertFalse(self.usage(raw)['native_usage_complete_observed'])
 def test_malformed_lines_prevent_completion(self):
  raw=deepcopy(self.trial['raw_usage']);raw['sessions'][0]['malformed_lines']=1
  self.assertFalse(self.usage(raw)['native_usage_complete_observed'])
 def test_duplicate_native_session_rejected(self):
  raw=deepcopy(self.trial['raw_usage']);raw['sessions'].append(deepcopy(raw['sessions'][0]))
  with self.assertRaisesRegex(ValueError,'duplicate native session'):self.usage(raw)
 def test_mismatched_aggregate_not_final(self):
  raw=deepcopy(self.trial['raw_usage']);raw['usage']['total_tokens']+=1
  self.assertIsNone(self.usage(raw)['tokens']['total_tokens'])
 def test_invalid_counts_not_tokens(self):
  for bad in (True,-1,'10',None):
   for field in c.TOKENS:
    with self.subTest(bad=bad,field=field):
     raw=deepcopy(self.trial['raw_usage']);raw['sessions'][0]['usage'][field]=bad
     self.assertIsNone(self.usage(raw)['tokens']['total_tokens'])
 def test_incoherent_session_totals(self):
  for key,value in [('total_tokens',1),('cached_input_tokens',999999)]:
   with self.subTest(key=key):
    raw=deepcopy(self.trial['raw_usage']);raw['sessions'][0]['usage'][key]=value
    self.assertFalse(self.usage(raw)['native_usage_complete_observed'])
 def test_wrong_session_route_unknown(self):
  for key,value in [('cli_version','other'),('observed_models',['other']),('observed_reasoning_efforts',['low'])]:
   with self.subTest(key=key):
    raw=deepcopy(self.trial['raw_usage']);raw['sessions'][0][key]=value
    self.assertIsNone(self.usage(raw)['tokens']['total_tokens'])
 def test_reward_not_completion_or_usage_authority(self):
  trial=deepcopy(self.trial);trial['reward']={'reward':0}
  row=c.native_trial(self.receipt,trial)
  self.assertEqual(row['outcome'],'official_verifier_not_passed');self.assertEqual(row['usage']['tokens']['total_tokens'],161750)
 def test_raw_fields_and_session_identifiers_not_exported(self):
  trial=deepcopy(self.trial);trial['raw_usage']['prompt']='PRIVATE_PROMPT_CANARY';trial['raw_usage']['sessions'][0]['path']='PRIVATE_PATH_CANARY';trial['exception_type']='PRIVATE\nMESSAGE'
  row=c.native_trial(self.receipt,trial);encoded=json.dumps(row)
  self.assertNotIn('PRIVATE',encoded);self.assertNotIn(trial['raw_usage']['sessions'][0]['session_id'],encoded)
  self.assertIsNone(row['exception_type'])
 def test_exact_intentional_control_difference_keeps_generic_mismatch(self):
  v=c.experiment_audit(self.native,self.supervisor)
  self.assertTrue(v['explicit_three_arm_declaration_matches']);self.assertFalse(v['production_comparator']['matches'])
  self.assertEqual(v['production_comparator']['differences'],[c.EXPERIMENT])
 def test_unexpected_common_control_differences_not_allowed(self):
  for section,key in [('identity','model'),('task_input_sha256','instruction.md'),('job_controls_sha256','environment'),('agent_controls_sha256','max_timeout_sec')]:
   with self.subTest(section=section,key=key):
    other=deepcopy(self.supervisor);other[section][key]='other' if section=='identity' else 'a'*64
    other.pop('sha256');other['sha256']=c.controls._digest(other)
    self.assertFalse(c.experiment_audit(self.native,other)['explicit_three_arm_declaration_matches'])
 def test_different_numerical_profile_not_allowed(self):
  other=deepcopy(self.supervisor);other['experimental_controls_sha256']['numerical_formalization_profile']='a'*64
  other.pop('sha256');other['sha256']=c.controls._digest(other)
  self.assertFalse(c.experiment_audit(self.native,other)['explicit_three_arm_declaration_matches'])
 def test_missing_numerical_profile_not_allowed(self):
  self.assertFalse(c.experiment_audit(self.native,self.native)['explicit_three_arm_declaration_matches'])
 def test_pending_attempt_suppresses_arm_totals(self):
  row=c.native_trial(self.receipt,self.trial)
  v=c.aggregate([dict(arm='native-codex',status='observed',trials=[row]),dict(arm='native-codex',status='awaiting_receipt',trials=[])],'native-codex')
  self.assertEqual(v['retained_attempts'],2);self.assertIsNone(v['tokens']['total_tokens'])
 def test_failed_attempt_tokens_retained(self):
  trial=deepcopy(self.trial);trial['reward']={'reward':0};row=c.native_trial(self.receipt,trial)
  v=c.aggregate([dict(arm='native-codex',status='observed',trials=[row])],'native-codex')
  self.assertEqual(v['official_passes'],0);self.assertEqual(v['tokens']['total_tokens'],161750)
 def test_native_wrong_trial_identity_visible(self):
  trial=deepcopy(self.trial);trial['exact_trial_profile_matches']=False
  self.assertFalse(c.native_trial(self.receipt,trial)['identity_matches'])
 def test_native_lock_requires_prelaunch_provenance(self):
  lock=HERE/'next-native-baseline/execution-lock.json';value,b=c.read(lock);value=deepcopy(value);value['frozen_before_launch']=False
  original=c.read
  def fake(path,**kwargs):return (value,b) if Path(path)==lock else original(path,**kwargs)
  with patch.object(c,'read',side_effect=fake),self.assertRaisesRegex(ValueError,'prelaunch'):c.native_attempt(lock)

 def test_native_launch_binds_every_frozen_input(self):
  expected=dict(command=['python','execute'],cwd='/public',lock_sha256='1'*64,runner_sha256='2'*64,recipe_sha256='3'*64,environment_sha256='4'*64)
  start=dict(schema='numerical-native-baseline-launch@1',argv=expected['command'],cwd=expected['cwd'],execution_lock_sha256=expected['lock_sha256'],script_sha256=expected['runner_sha256'],recipe_sha256=expected['recipe_sha256'],environment_sha256=expected['environment_sha256'])
  c.validate_native_launch(start,**expected)
  for key in start:
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,'launch binding'):
    changed=deepcopy(start);changed[key]='changed';c.validate_native_launch(changed,**expected)
 def test_two_distinct_sessions_sum_once(self):
  raw=deepcopy(self.trial['raw_usage']);other=deepcopy(raw['sessions'][0]);other['session_id']='other-session';raw['sessions'].append(other)
  for key in c.TOKENS:raw['usage'][key]*=2
  self.assertEqual(self.usage(raw)['tokens']['total_tokens'],323500)
 def test_native_lock_unexpected_path_rejected_before_open(self):
  lock=HERE/'next-native-baseline/execution-lock.json';value,b=c.read(lock);value=deepcopy(value)
  value['files'][str(HERE/'private-auth.json')]='a'*64
  original=c.read
  def fake(path,**kwargs):
   self.assertNotEqual(Path(path),HERE/'private-auth.json')
   return (value,b) if Path(path)==lock else original(path,**kwargs)
  with patch.object(c,'read',side_effect=fake),self.assertRaisesRegex(ValueError,'lock file inventory'):c.native_attempt(lock)
 def test_wrong_native_launch_receipt_binding_rejected(self):
  root=HERE/'next-native-baseline';base=root/'native-codex-01';original=c.read
  receipt=deepcopy(self.receipt)
  fake_result={'schema':'numerical-native-baseline-launch-result@1','receipt_sha256':'a'*64,'exit_code':0}
  def fake(path,**kwargs):
   if Path(path)==base/'receipt.json':return receipt,{'availability':'observed','sha256':'b'*64}
   if Path(path)==base/'launch-result.json':return fake_result,{'availability':'observed','sha256':'c'*64}
   return original(path,**kwargs)
  with patch.object(c,'read',side_effect=fake),self.assertRaisesRegex(ValueError,'receipt digest'):c.native_attempt(root/'execution-lock.json')
 def test_changed_saved_pair_projection_refused(self):
  saved_path=HERE/'pair-observation-final.json';saved,sb=c.read(saved_path);changed=deepcopy(saved)
  changed['arms']['full']['tokens']['total_tokens']+=1
  original=c.read
  def fake(path,**kwargs):return (changed,sb) if Path(path)==saved_path else original(path,**kwargs)
  with patch.object(c,'read',side_effect=fake),self.assertRaisesRegex(ValueError,'saved pair observation differs'):
   c.collect(saved_path,HERE/'pair-lock.json',HERE/'next-native-baseline/execution-lock.json')

 def test_complete_three_arm_authored_join_uses_historical_native_counts(self):
  root=HERE/'next-native-baseline';base=root/'native-codex-01';original=c.read
  receipt=deepcopy(self.receipt)
  receipt['comparison_controls']={'status':'observed','configuration_unchanged':True,'declared':self.native}
  encoded=json.dumps(receipt,sort_keys=True).encode();digest=c.pair.sha(encoded)
  result={'schema':'numerical-native-baseline-launch-result@1','receipt_sha256':digest,'exit_code':0,'seconds':1.0}
  def fake(path,**kwargs):
   if Path(path)==base/'receipt.json':return receipt,{'availability':'observed','sha256':digest,'bytes':len(encoded)}
   if Path(path)==base/'launch-result.json':return result,{'availability':'observed','sha256':'c'*64,'bytes':10}
   return original(path,**kwargs)
  with patch.object(c,'read',side_effect=fake):
   value=c.collect(HERE/'pair-observation-final.json',HERE/'pair-lock.json',root/'execution-lock.json')
  self.assertTrue(value['all_registered_attempts_control_integrity'])
  self.assertFalse(value['production_generic_comparator_match'])
  self.assertEqual(value['arms']['native-codex']['tokens']['total_tokens'],161750)
  self.assertEqual(value['arms']['full']['tokens']['total_tokens'],333973)
  self.assertEqual(value['supervisor_minus_native_tokens']['full']['total_tokens'],172223)
  self.assertEqual(value['full_minus_no_index_tokens']['total_tokens'],-24519)
  self.assertEqual(len(value['attempts']),3)
  self.assertNotIn(self.trial['raw_usage']['sessions'][0]['session_id'],json.dumps(value))

if __name__=='__main__':unittest.main(verbosity=2)
