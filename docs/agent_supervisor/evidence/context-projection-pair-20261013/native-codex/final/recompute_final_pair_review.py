"""Read retained numeric receipts only; never inspect benchmark solution bodies."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
R=Path(__file__).resolve().parent
P=Path('/home/barberb/lift_coding/.worktrees/supervisor-comparison-contracts-20261011')
def sha(path):
 with path.open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def diff(a,b,path=''):
 if type(a)is not type(b):return[path]
 if isinstance(a,dict):return[change for k in sorted(a.keys()|b.keys())for change in ([path+'/'+k]if k not in a or k not in b else diff(a[k],b[k],path+'/'+k))]
 if isinstance(a,list):return[path]if len(a)!=len(b)else[change for i,(x,y)in enumerate(zip(a,b))for change in diff(x,y,path+'/'+str(i))]
 return[]if a==b else[path]
records={};configs={};preps={}
for label,name in [('legacy','codex-legacy-01'),('projected','codex-projected-01')]:
 root=R/'live-trials'/name;path=root/'receipt.json';v=json.loads(path.read_text());prep=json.loads((root/'preparation.json').read_text());config=json.loads((root/'config.json').read_text());configs[label]=config;preps[label]=prep
 assert v['complete_single_trial_receipt']is True and v['trial_count']==1 and v['original_task_inputs_unchanged']is True
 assert sha(root/'config.json')==prep['config_sha256']
 trial=v['trials'][0];s=trial['supervisor'];invocations=s['provider_invocations'];assert len(invocations)==2 and sorted(x['purpose']for x in invocations)==['coding','planning']
 assert len({x['invocation_id']for x in invocations})==2
 rows=[];totals={k:0 for k in ('input_tokens','cached_input_tokens','output_tokens','total_tokens')}
 for invocation in invocations:
  native=invocation['native_rollout_usage'];usage=native['usage']
  assert native['schema']=='native-codex-cumulative-usage@1'and native['usage_available']is True and native['cache_included_in_input']is True
  assert native['billing_total_verified']is False and native['dollar_cost']is None
  assert usage['total_tokens']==usage['input_tokens']+usage['output_tokens']
  assert 0<=usage['cached_input_tokens']<=usage['input_tokens']
  for k in totals:totals[k]+=usage[k]
  rows.append(dict(purpose=invocation['purpose'],usage={k:usage[k]for k in totals},rollout_sha256=native['rollout_sha256'],observed_token_count_records=native['observed_token_count_records']))
 totals['uncached_input_tokens']=totals['input_tokens']-totals['cached_input_tokens']
 source=s['initial_context']['source384_context']['summary'];coding=next(x for x in invocations if x['purpose']=='coding');metadata=coding.get('semantic_metadata_view')
 records[label]=dict(receipt_sha256=sha(path),preparation_sha256=sha(root/'preparation.json'),config_sha256=sha(root/'config.json'),reward=trial['reward'],exception_type=trial['exception_type'],durations_seconds=trial['durations_seconds'],invocations=rows,totals=totals,supervisor_internal_completion=s['task_completed'],intent_status=s['intent_preplanning']['status'],source384=dict(checkpoint_sha256=source['checkpoint_sha256'],mode=source['mode'],training_steps=source['training_steps'],candidate_count=source['candidate_count'],coverage=source['coverage'],source_aligned_candidate_count=source['source_aligned_candidate_count'],source_semantics_verified=source['source_semantics_verified'],proof_authority=source['proof_authority']),indexed_reader_requests=s['source_view_retrieval_observation']['counts']['accepted_requests'],selected_metadata=(None if metadata is None else {k:metadata[k]for k in ('selected_mode','native_complete_bytes','selected_complete_bytes','native_complete_sha256','selected_complete_sha256')}))
 assert all(sha(Path(name))==digest for name,digest in prep['host_source_sha256'].items())
changes=diff(configs['legacy'],configs['projected']);assert changes==['/agents/0/kwargs/semantic_metadata_view','/jobs_dir']
fixed=('host_source_sha256','task_input_sha256','source384','intent_action_384','agent_timeout_seconds','archive_sha256','manifest_sha256','provider_profile','source_view_profile','resource_profile','model','reasoning_effort','cli_version')
assert all(preps['legacy'][k]==preps['projected'][k]for k in fixed)
archive=Path(preps['legacy']['archive']);assert sha(archive/'runtime.tar.gz')==preps['legacy']['archive_sha256'];assert sha(archive/'manifest.json')==preps['legacy']['manifest_sha256']
reg=json.loads((R/'regressions-01/qualification.json').read_text());source_unchanged=all(sha(P/name)==digest for name,digest in reg['source_after'].items());assert source_unchanged
result=dict(schema='context-projection-final-independent-recomputation@1',observed_at_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(Path(__file__)),arms=records,config_changed_paths=changes,equal_preparation_controls=list(fixed),host_preparation_sources_still_unchanged=True,runtime_archive_and_manifest_still_unchanged=True,regression_source_closure_still_unchanged=source_unchanged,projected_minus_legacy={k:records['projected']['totals'][k]-records['legacy']['totals'][k]for k in records['legacy']['totals']},usage_scope='Sum final native cumulative usage once per unique planning/coding invocation. Cached input is a subset and reasoning output is not added again.',billing_verified=False,causal_advantage_established=False,total_token_reduction=False,hidden_test_or_solution_bodies_read=False)
(R/'context-projection-final-recomputation.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(totals={k:v['totals']for k,v in records.items()},config_changed_paths=changes,source_closure_unchanged=source_unchanged)))
