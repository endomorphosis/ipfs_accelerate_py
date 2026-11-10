"""Arithmetic over the fresh closed diagnostic only; no owner report or logs."""
from pathlib import Path
import hashlib
import json

B=Path(__file__).resolve().parent
INPUT=B/'live/actual-public-diagnostic-01.json'
OUTPUT=B/'closed-arithmetic-analysis-01.json'
TOOLS=('read_file','search_replace','grep','list_dir','todo_write','run_terminal_cmd','unknown')
LANES=('input','output','content')
STATES=('missing','null','empty','valid','invalid','oversized')


def analyze(data):
    assert data['schema']=='terminal-public-tool-event-diagnostic@1'
    observation=data['observation'];assert observation['schema']=='terminal-public-tool-event-readout@1'
    summaries=[]
    for invocation in observation['invocations']:
        progress=invocation['native_grok_progress'];assert progress['schema']=='native-grok-stream-progress@4'
        events=progress['tool_event_observations'];assert events['schema']=='native-grok-tool-events@1'
        rows=events['rows'];assert len(rows)==events['retained_events']
        assert sum(row['frame_bytes'] for row in rows)==events['retained_frame_bytes']
        calls={}
        for row in rows:
            if row['event_type']=='tool_call' and row['call_id_sha256'] is not None:
                calls.setdefault(row['call_id_sha256'],set()).add(row['tool'])
        by_tool={tool:{lane:0 for lane in LANES} for tool in TOOLS}
        joined={tool:{lane:0 for lane in LANES} for tool in TOOLS}
        joined_events={tool:0 for tool in TOOLS}
        lanes={}
        for lane in LANES:
            stats=events['payloads'][lane]
            measured=[row[lane] for row in rows if row[lane]['canonical_json_bytes'] is not None]
            assert len(measured)==stats['measured_fields']
            assert sum(item['canonical_json_bytes'] for item in measured)==stats['canonical_json_bytes']
            repeated=[item for item in measured if item['repetition'] not in ('first_observed','not_measured')]
            assert len(repeated)==stats['repeated_events']
            assert sum(item['canonical_json_bytes'] for item in repeated)==stats['repeated_bytes']
            states={state:sum(row[lane]['state']==state for row in rows) for state in STATES}
            lanes[lane]={**stats,'states':states,'repeated_bytes_percent':
                100*stats['repeated_bytes']/stats['canonical_json_bytes'] if stats['canonical_json_bytes'] else None}
        for row in rows:
            identity_tools=calls.get(row['call_id_sha256'],set())
            joined_tool=next(iter(identity_tools)) if len(identity_tools)==1 else 'unknown'
            if row['tool_name_present'] and row['tool']!=joined_tool:
                joined_tool='unknown'
            joined_events[joined_tool]+=1
            for lane in LANES:
                size=row[lane]['canonical_json_bytes']
                if size is not None:
                    by_tool[row['tool']][lane]+=size;joined[joined_tool][lane]+=size
        for lane in LANES:
            assert sum(stats[lane] for stats in by_tool.values())==events['payloads'][lane]['canonical_json_bytes']
            assert sum(stats[lane] for stats in joined.values())==events['payloads'][lane]['canonical_json_bytes']
        summaries.append(dict(stream_complete=progress['stream_complete'],response_usable=progress['response_usable'],
            final_event_type=progress['final_event_type'],final_reason_code=progress['final_reason_code'],
            returncode=progress['returncode'],timed_out=progress['timed_out'],duration_seconds=progress['duration_seconds'],
            stdout_bytes=progress['stdout_bytes'],stderr_bytes=progress['stderr_bytes'],frames=progress['frames'],
            tool_call_counts=progress['tool_name_counts'],tool_status_event_counts=progress['tool_status_counts'],
            observed_tool_events=events['observed_events'],retained_tool_events=events['retained_events'],
            omitted_tool_events=events['omitted_events'],retained_tool_frame_bytes=events['retained_frame_bytes'],
            call_identities=events['call_identities'],call_relation_counts=events['call_relation_counts'],
            observation_complete=events['observation_complete'],incompleteness=events['incompleteness'],
            payload_lanes=lanes,payload_bytes_by_tool_name_on_event=by_tool,
            tool_name_on_event_presence_counts={str(value).lower():sum(row['tool_name_present'] is value for row in rows) for value in (True,False)},
            payload_bytes_by_resolved_call_tool=joined,resolved_call_tool_event_counts=joined_events,
            resolved_call_tool_assignment_basis='unique-retained-tool-call-label-for-call-hash',
            conflicting_explicit_event_tool_kept_unknown=True,call_join_origin_authenticated=False,
            payload_lanes_may_overlap=True,unique_output_bytes_measured=False,
            tool_result_coverage_complete=events['tool_result_coverage_complete'],
            repetition_scope=events['repetition_scope'],repetition_categories_overlap=True,
            replacement_or_delta_semantics='unverified',
            event_repetition_does_not_establish_history_resends=True,
            provider_request_bytes=None,history_resend_bytes=None,request_history_complete=False,
            tokens=invocation['tokens'],known_token_subtotals=invocation['known_token_subtotals'],
            native_usage_complete_observed=invocation['native_usage_complete_observed'],
            cache_included_in_input=invocation['cache_included_in_input'],billing_total_verified=False,
            reported_cost_usd=invocation['reported_cost_usd'],
            known_reported_cost_subtotal_usd=invocation['known_reported_cost_subtotal_usd']))
    return dict(schema='terminal-tool-event-closed-arithmetic-analysis@1',diagnostic_qualified=data['qualified'],
        observation_only=True,arithmetic_checked_against_retained_rows=True,invocations=summaries,
        planning_provider_calls=observation['planning_provider_calls'],coding_provider_calls=observation['coding_provider_calls'],
        provider_internal_model_calls=None,official_verifier_executed=False,official_reward=None,
        task_success_claimed=False,causal_token_savings_claimed=False,raw_content_exported=False,
        payload_lane_totals_not_added=True,source384=observation.get('source384'),phases=observation['phases'],
        worker_cleanup_returncode=observation.get('worker_cleanup_returncode'),
        exact_container_absence_observations=data['exact_container_absence_observations'],
        container_stop_returned=data['container_stop_returned'],frozen_inputs_unchanged=data['frozen_inputs_unchanged'],
        seconds=data['seconds'])


if __name__=='__main__':
    raw=INPUT.read_bytes();result=analyze(json.loads(raw));result['closed_export_sha256']=hashlib.sha256(raw).hexdigest()
    with OUTPUT.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(path=str(OUTPUT),bytes=OUTPUT.stat().st_size,sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest()),sort_keys=True))
