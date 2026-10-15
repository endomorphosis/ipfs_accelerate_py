"""Extract a compact outcome from the qualified closed reader only."""
from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parent
P=B/'live/readout-final.json'
r=json.loads(P.read_text());base=r['baseline']
assert r['raw_prompts_or_provider_text_exported'] is False and r['provider_calls_by_reader']==0
assert r['total_token_savings_established'] is False and r['causal_savings_claimed'] is False
assert len(base['baseline']['pilot']['trials'])==len(base['baseline']['prior']['trials'])==1
new=base['baseline']['pilot']['trials'][0];old=base['baseline']['prior']['trials'][0]
def summarize(t):
 return dict(trial=t['trial'],official_reward=t['official_reward'],task_completed_reported=t['task_completed_reported'],native_supervisor_seconds=t['seconds'],
  phase_seconds=t['phase_seconds'],observed_usage=t['all_phase_usage']['observed_subtotals'],
  full_usage_completeness_established=t['all_phase_usage']['all_native_contracts_complete'],billing_total_verified=t['all_phase_usage']['billing_total_verified'],
  invocations=[dict(phase=i['phase'],status=i['status'],provider=i['provider'],model=i['model'],model_prompt_bytes=i['model_prompt_bytes'],native_cli=i['native_cli'],
   observed_usage=i['native_usage']['counters'],native_usage_complete=i['native_usage']['usage_complete_under_native_contract']) for i in t['invocations']],
  public_validation=t['public_validation'],input_audit=t['input_audit'],continuation=t['continuation'])
value=dict(schema='candidate-prompt-projection-live-outcome@1',runtime_commit='cef1bb3764aa2ea9ed453a2feec5c240cca2dc54',
 pilot=summarize(new),prior=summarize(old),planning=base['planning'],projection=r['candidate_projection'],
 source_readout_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),services_changed=[],
 independent_owner_replay_performed=False,total_token_savings_established=False,causal_savings_claimed=False,raw_content_exported=False)
with (B/'live-outcome.json').open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({k:value[k] for k in ['schema','pilot','prior','projection']},indent=2))
