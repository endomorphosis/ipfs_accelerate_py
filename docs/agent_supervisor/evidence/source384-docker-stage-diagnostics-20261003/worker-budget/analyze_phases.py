"""Reproduce stage totals from bounded diagnostic events, without source data."""
import collections,hashlib,json,math
from pathlib import Path
B=Path(__file__).parent
raw=(B/'phase-events.jsonl').read_bytes();s=json.loads((B/'phase-summary.json').read_text());events=[json.loads(line) for line in raw.splitlines()]
assert len(raw)==s['event_bytes']<=s['max_event_bytes'] and len(events)==s['events']<=s['max_events']
assert s['dropped_events']==0 and s['stack_empty']
assert s['after_context']['manifest_producer_recognized'] and s['after_context']['manifest_producer_unchanged']
entered={};closed={};order=[];totals=collections.defaultdict(lambda:dict(calls=0,inclusive_seconds=0.,exclusive_seconds=0.))
for row in events:
 if row['event']=='enter':
  assert row['call_id'] not in entered
  assert row['parent_call_id']==(order[-1] if order else None)
  entered[row['call_id']]=row;order.append(row['call_id'])
 elif row['event']=='exit':
  assert order.pop()==row['call_id']
  assert row['stage']==entered[row['call_id']]['stage']
  assert math.isfinite(row['seconds']) and row['seconds']>=0
  closed[row['call_id']]=row
assert not order and set(entered)==set(closed) and len(closed)==s['calls']
children=collections.defaultdict(float)
for row in closed.values():
 if row['parent_call_id'] is not None:children[row['parent_call_id']]+=row['seconds']
for key,row in closed.items():
 t=totals[row['stage']];t['calls']+=1;t['inclusive_seconds']+=row['seconds'];t['exclusive_seconds']+=row['seconds']-children[key]
selected=[]
for key,row in entered.items():
 if row['stage'] in ('index.prepare_current','index.observe_current','source_units._context','source_units._worker'):
  selected.append(dict(stage=row['stage'],at_seconds=row['at_seconds'],
   passed_timeout_seconds=row.get('timeout_seconds',row.get('timeout')),
   timeout_scope=row.get('timeout_scope'),elapsed_seconds=closed[key]['seconds'],error_type=closed[key]['error_type']))
context=json.loads((B/'docker-01/source384-context.json').read_text());deployment=json.loads((B/'docker-01/deployment/deployment.json').read_text())
assert deployment['original_inputs']==deployment['retained_inputs'] and len(deployment['original_inputs']['files'])==218
summary=dict(schema='source384-stage-timing-analysis@1',diagnostic_only=True,
 production_qualification_claimed=False,benchmark_result=False,events_sha256=hashlib.sha256(raw).hexdigest(),
 events=len(events),dropped_events=0,paired_calls=len(closed),stage_calls=selected,totals=dict(totals),
 cache_before=s['before_wrappers'],cache_after=s['after_context'],
 original_files_preserved=218,resource_profile='source384-5cpu-12gib@1',
 deployment_seconds=deployment['seconds'],initial_context_seconds=context['phase_seconds'],
 probe_seconds=context['seconds'],bounded_worker_invoked=True,
 successful_model_load_qualified=False,completed_neural_inference_qualified=False,
 findings=['Cold preparation consumed79.330941s before the worker.',
 'Worker received0.917290s before child admission, then timed out after0.955463s.',
 'Manifest memo was active with3hits,1miss,0bypasses and0evictions.',
 'Nested stage inclusive totals overlap; exclusive totals subtract immediate timed children only.',
 'Instrumentation changes runtime callables in memory; this is not production qualification or a speed claim.',
 'These events do not split cold preparation into scanner, canonical reconstruction and SQL costs.'])
(B/'phase-analysis.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(deployment_seconds=deployment['seconds'],stage_calls=selected,events=len(events),dropped_events=0,totals=dict(totals)),indent=2))
