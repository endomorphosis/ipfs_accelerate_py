from pathlib import Path
import importlib.util,json,hashlib
p=Path('/home/barberb/lift_coding/artifacts/source-context-substitution-20261013/native-observations/observe_results.py')
spec=importlib.util.spec_from_file_location('qualified_observer',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rows=[]
for name,expected in [('baseline-01',246992),('compact-01',327117)]:
 receipt=Path('/home/barberb/lift_coding/artifacts/terminal-reply-contract-results-20261007')/name/'receipt.json'
 value,binding=module.read(receipt);projected,_=module.row_projection(value,value['trials'][0],binding)
 assert projected['tokens']['total_tokens']==expected and projected['native_usage_complete_observed'] is True
 assert projected['tokens']['input_tokens']+projected['tokens']['output_tokens']==expected
 rows.append(dict(historical_arm=name,receipt_sha256=binding['sha256'],total_tokens=expected,cached_input_included=True,official_reward=projected['official_reward']))
print(json.dumps(dict(schema='native-result-observer-historical-replay@1',rows=rows,observer_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),historical_only=True,current_trial_results_established=False,provider_calls=0),sort_keys=True))
