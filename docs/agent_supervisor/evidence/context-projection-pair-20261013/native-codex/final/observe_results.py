"""Bounded read-only metrics from retained owner/Harbor result receipts.

No model transcript, native response, task/verifier/solution body, database,
credential, or process environment is opened. Missing usage is never zero.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse,hashlib,json,math,os,re,stat
from benchmarks.agent_supervisor.container_coding import benchmark_comparison as comparison
from benchmarks.agent_supervisor.container_coding import native_usage_accounting as usage
from benchmarks.agent_supervisor.container_coding.benchmark_controls import compare_controls,validate_controls
ROOT=Path('/home/barberb/lift_coding/artifacts/source-context-substitution-20261013')
MAX_BYTES=64*1024*1024

def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):
 path=Path(path)
 if not path.exists():return None,dict(path=str(path),availability='missing')
 before=path.lstat()
 if not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_size>MAX_BYTES:raise ValueError('bounded regular receipt required')
 def unique(pairs):
  out={}
  for k,v in pairs:
   if k in out:raise ValueError('duplicate receipt field')
   out[k]=v
  return out
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 with os.fdopen(fd,'rb') as f:
  raw=f.read(MAX_BYTES+1);after=os.fstat(f.fileno())
 facts=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 if len(raw)>MAX_BYTES or facts(before)!=facts(after) or facts(after)!=facts(path.lstat()):return None,dict(path=str(path),availability='changing')
 data=json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _value:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
 if type(data) is not dict:raise ValueError('object receipt required')
 return data,dict(path=str(path),availability='observed',sha256=sha(raw),bytes=len(raw))
def obj(x):return x if type(x)is dict else {}
def scalar(x):return type(x)in(str,bool,int,type(None)) or type(x)is float and math.isfinite(x)
def fields(x,keys):x=obj(x);return {k:x[k] for k in keys if k in x and scalar(x[k])}
def counts(x):return {k:v for k,v in obj(x).items() if type(k)is str and re.fullmatch('[a-zA-Z0-9_:@.-]{1,128}',k) and type(v)is int and v>=0}
def summaries(report):
 initial=obj(report.get('initial_context'))
 ctx=obj(initial.get('source384_context'))
 summary=obj(ctx.get('summary'))
 coverage=obj(summary.get('coverage'))
 intent=obj(report.get('intent_preplanning'))
 return {
  'intent_ir':fields(intent,['status','seconds','before_goal_decomposition','instruction_sha256','advice_sha256','artifact_sha256','execution_authority','completion_authority']),
  'source384':{
   'context':fields(ctx,['schema','seconds','checkpoint_sha256','inference_sha256','neural_inference_replayed','training_steps','formalization_authority','proof_authority']),
   'summary':fields(summary,['schema','mode','source_files','program_source_files','harness_support_files','inference_python_files','excluded_source_files','training_steps','provider_calls','nomination_only','model_promotion_performed','omitted_candidates','formalization_authority','proof_authority']),
   'coverage':{**fields(coverage,['files','functions','selected_units']),**{k:counts(coverage.get(k)) for k in ['candidate_statuses','file_dispositions','unit_dispositions']}}
  },
  'source_unit_preplanning':fields(obj(report.get('planning')).get('source_unit_preplanning'),['status','before_goal_decomposition','supplied_to_router','source_semantics_verified'])
 }
def row_projection(receipt,trial,binding):
 row=comparison._trial(receipt,trial,source=binding)
 report=obj(trial.get('supervisor'))
 result={k:row[k] for k in ['trial','task','model','reasoning_effort','cli_version','runtime_archive_sha256','original_task_inputs_unchanged','native_result_sha256','official_reward','native_task_completed','outcome','exception_type','supervisor_error_type','durations_seconds','native_total_seconds','supervisor_seconds','supervisor_phases_seconds','initial_indexed_context_observation','indexed_context_observation','doctor_observation','code_learning_observation','post_publication_proof_index','context_input_audit','post_publication_context','provider_calls','observed_provider_invocations','unreceipted_provider_attempt','native_usage_complete_observed','cache_included_in_input','observed_counts_are_lower_bounds','billing_total_verified','tokens','known_token_subtotals','reported_cost_usd','known_reported_cost_subtotal_usd','invocations']}
 result.update(summaries(report))
 source_read=obj(report.get('source_view_retrieval_observation'))
 result['source_reader']=dict(**fields(source_read,['schema','status','reason','total_token_savings_measured','source_obligations_discharged']),counts=counts(source_read.get('counts')),candidate_context_counts=counts(source_read.get('candidate_context_counts')))
 raw_invocations={v['invocation_id']:v for v in report.get('provider_invocations',[]) if type(v)is dict and type(v.get('invocation_id'))is str}
 for invocation in result['invocations']:
  raw=raw_invocations[invocation['invocation_id']]
  actual=usage.native_usage_contract(raw,obj(raw.get('native_rollout_usage')))
  assert actual==(invocation['native_usage_complete_observed'],invocation['token_count_scope'],invocation['cache_included_in_input'])
  invocation['source_view_profile']=obj(raw.get('source_view_retrieval')).get('profile')
  d=obj(raw.get('representation_displacement'))
  invocation['metadata_displacement']=fields(d,['schema','requested_mode','selected_mode','treatment','selection_basis','fallback_reason','selected_complete_byte_reduction','candidate_complete_byte_reduction','required_source_bytes_preserved','source_preservation','provider_delivery_verified','total_token_savings_measured'])
 return result,row

def observe_arm(name):
 base=ROOT/'live-trials'/name;prepared,pbind=read(base/'preparation.json');receipt,rbind=read(base/'receipt.json')
 result=dict(arm_label=name,preparation=pbind,receipt=rbind,availability='pending',attempts=[],provider_calls_by_observer=0)
 if not prepared:return result,[],None
 result['declared_profile']=fields(prepared,['provider_profile','model','reasoning_effort','cli_version','resource_profile','source_view_profile','semantic_metadata_view','coding_reply_mode','archive_sha256','manifest_sha256','source384_enabled'])
 result['declared_profile'].setdefault('semantic_metadata_view','legacy');result['declared_profile'].setdefault('coding_reply_mode','legacy')
 config,cbind=read(base/'config.json');result['configuration']=cbind
 result['config_matches_preparation']=cbind.get('sha256')==prepared.get('config_sha256')
 checked={}
 for path,expected in prepared.get('host_source_sha256',{}).items():
  source=Path(path)
  if not source.is_relative_to(Path('/home/barberb/lift_coding/.worktrees/metadata-live-20261013')):raise ValueError('unowned host source path')
  checked[path]=source.is_file() and not source.is_symlink() and sha(source.read_bytes())==expected
 result['host_source_files_checked']=len(checked);result['host_source_closure_unchanged']=bool(checked)and all(checked.values())
 result['prepared_controls_valid']=validate_controls(prepared.get('comparison_controls'))
 if receipt is None:return result,[],None
 if receipt.get('schema')!=comparison.SUPERVISOR or receipt.get('task')!='largest-eigenval':raise ValueError('unexpected native receipt')
 result.update(availability='observed',receipt_summary=fields(receipt,['harbor_returncode','complete_single_trial_receipt','trial_count','invocation_seconds','original_task_inputs_unchanged','native_job_result_present','source_view_config_unchanged','semantic_metadata_config_unchanged','coding_reply_config_unchanged','intent_selection_config_unchanged']),declared_control_observation=receipt.get('comparison_controls'))
 trials=receipt.get('trials');assert type(trials)is list and len(trials)<=128
 rows=[];seen=set()
 for trial in trials:
  assert type(trial)is dict and type(trial.get('trial'))is str and trial['trial']not in seen
  seen.add(trial['trial']);projected,row=row_projection(receipt,trial,rbind);result['attempts'].append(projected);rows.append(row)
 result['totals']=comparison._aggregate(rows)
 return result,rows,receipt

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 output=args.output.absolute();assert output.is_relative_to(ROOT/'native-observations') and not output.exists()
 names=['codex-legacy-01','codex-projected-01'];arms={};all_rows=[];receipts={}
 for name in names:
  arm,rows,receipt=observe_arm(name);arms[name]=arm;all_rows.extend(rows)
  if receipt:receipts[name]=receipt
 grok,gbind=read(ROOT/'live-trials/grok-legacy-02/receipt.json');probe,pbind=read(ROOT/'live-preparation/grok-isolated-copy-comparison.json')
 failed=[]
 if grok:
  for trial in grok.get('trials',[]):
   projected,_=row_projection(grok,trial,gbind);failed.append(projected)
 comparisons=compare_controls(receipts[names[0]].get('comparison_controls'),receipts[names[1]].get('comparison_controls')) if len(receipts)==2 else None
 value=dict(schema='source-context-native-result-observation@1',observed_at_utc=datetime.now(timezone.utc).isoformat(),arms=arms,codex_declared_controls_comparison=comparisons,grok_failed_setup_attempts=failed,grok_failure_receipt=gbind,grok_metadata_probe_evidence=pbind,grok_metadata_probe_count=probe.get('models_metadata_commands')if probe else None,grok_metadata_probe_generation_calls=probe.get('provider_generation_calls')if probe else None,observer=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__).read_bytes()),comparison_module_sha256=sha(Path(comparison.__file__).read_bytes()),native_usage_contract_sha256=sha(Path(usage.__file__).read_bytes()),provider_calls=0,hidden_verifier_or_solution_contents_read=False,raw_model_transcripts_read=False,credential_contents_read=False),accounting=dict(authority='Existing retained native receipt projection and native_usage_contract; no transcript summation',native_cumulative_counted_once_per_invocation=True,cached_tokens_not_added_to_codex_input=True,incomplete_usage_remains_unknown=True,observed_subtotals_separate=True,stage_times_not_added_to_nested_agent_time=True,billing_total_verified=False,causal_advantage_claimed=False))
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(output=str(output),arms={name:dict(availability=v['availability'],attempts=len(v['attempts']),metrics=[dict(reward=a['official_reward'],complete_usage=a['native_usage_complete_observed'],total_tokens=a['tokens']['total_tokens'],known_total_subtotal=a['known_token_subtotals']['total_tokens'])for a in v['attempts']])for name,v in arms.items()}),sort_keys=True))
if __name__=='__main__':main()
