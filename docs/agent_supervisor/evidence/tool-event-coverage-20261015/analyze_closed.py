"""Analyze the fresh closed @5/@2 receipt; never open private run artifacts."""
from pathlib import Path
import hashlib
import json
import math
import sys

B=Path(__file__).resolve().parent
INPUT=B/'live/actual-public-diagnostic-01.json'
OUTPUT=B/'closed-arithmetic-analysis-01.json'
TOOLS=('read_file','search_replace','grep','list_dir','todo_write','run_terminal_cmd','unknown')
LANES=('input','output','content')
STATES=('missing','null','empty','valid','invalid','oversized')
SHAPES=('missing','null','boolean','integer','number','string','object','array','invalid')
REASONS=('bytes','nodes','depth','integer_bits')
EVENTS=('tool_call','tool_call_update','unknown')
TOKENS=('input_tokens','cached_input_tokens','output_tokens','total_tokens')
PHASES=('prepare_seconds','initial_context_seconds','planning_seconds','context_seconds','admission_seconds','coding_seconds','cleanup_seconds')


def scalar(value,kind,*,nullable=False):
    assert value is None and nullable or type(value) is kind
    if kind in (int,float) and value is not None:assert math.isfinite(value)
    return value


def number(value):
    assert value is None or type(value) in (int,float) and 0<=value<2**63 and math.isfinite(value)
    return value


def boolean(value,*,nullable=False):
    return scalar(value,bool,nullable=nullable)


def digest(value):
    assert type(value) is str and len(value)==64 and set(value)<=set('0123456789abcdef')
    return value


def token_counts(value):
    assert type(value) is dict and set(value)==set(TOKENS)
    return {key:scalar(value[key],int,nullable=True) for key in TOKENS}


def analyze(data):
    from ipfs_accelerate_py.cli_runtime.grok_stream_capture import validate_grok_stream_progress
    assert data['schema']=='terminal-public-tool-event-diagnostic@1'
    assert data['expected_progress_schema']=='native-grok-stream-progress@5'
    assert data['expected_tool_event_schema']=='native-grok-tool-events@2'
    observation=data['observation'];assert observation['schema']=='terminal-public-tool-event-readout@1'
    assert type(observation['invocations']) is list and len(observation['invocations'])<=1
    summaries=[]
    for invocation in observation['invocations']:
        progress=validate_grok_stream_progress(invocation['native_grok_progress'])
        assert progress['schema']=='native-grok-stream-progress@5'
        events=progress['tool_event_observations'];assert events['schema']=='native-grok-tool-events@2'
        rows=events['rows'];assert len(rows)==events['retained_events']
        assert sum(row['frame_bytes'] for row in rows)==events['retained_frame_bytes']
        legacy_bytes={tool:{lane:0 for lane in LANES} for tool in TOOLS}
        resolved_bytes={tool:{lane:0 for lane in LANES} for tool in TOOLS}
        lanes={}
        for lane in LANES:
            stats=events['payloads'][lane]
            measured=[row[lane] for row in rows if row[lane]['canonical_json_bytes'] is not None]
            assert len(measured)==stats['measured_fields']
            assert sum(item['canonical_json_bytes'] for item in measured)==stats['canonical_json_bytes']
            repeated=[item for item in measured if item['repetition'] not in ('first_observed','not_measured')]
            assert len(repeated)==stats['repeated_events']
            assert sum(item['canonical_json_bytes'] for item in repeated)==stats['repeated_bytes']
            for prefix in ('same_call','cross_call'):
                repeats=[item for item in measured if item['repetition'] in (prefix,'same_and_cross_call')]
                assert len(repeats)==stats[prefix+'_repeated_events']
                assert sum(item['canonical_json_bytes'] for item in repeats)==stats[prefix+'_repeated_bytes']
            states={state:sum(row[lane]['state']==state for row in rows) for state in STATES}
            bounds={reason:sum(row[lane]['bound_reason']==reason for row in rows) for reason in REASONS}
            assert bounds==events['classification_counts']['payload_bounds'][lane]
            frames_at_bound=[row['frame_bytes'] for row in rows if row[lane]['bound_reason'] is not None]
            lanes[lane]={**stats,'states':states,'bound_reasons':bounds,
                'max_measured_field_bytes':max((item['canonical_json_bytes'] for item in measured),default=None),
                'bounded_payload_frame_min_bytes':min(frames_at_bound,default=None),
                'bounded_payload_frame_max_bytes':max(frames_at_bound,default=None),
                'repeated_bytes_percent':100*stats['repeated_bytes']/stats['canonical_json_bytes'] if stats['canonical_json_bytes'] else None}
        for row in rows:
            for lane in LANES:
                size=row[lane]['canonical_json_bytes']
                if size is not None:
                    legacy_bytes[row['tool']][lane]+=size
                    resolved_bytes[row['resolved_tool']][lane]+=size
        for lane in LANES:
            assert sum(stats[lane] for stats in legacy_bytes.values())==events['payloads'][lane]['canonical_json_bytes']
            assert sum(stats[lane] for stats in resolved_bytes.values())==events['payloads'][lane]['canonical_json_bytes']
        kinds={kind:[row for row in rows if row['event_type']==kind] for kind in EVENTS}
        status_by_event={kind:{'present':sum(row['status_present'] for row in selected),
            'missing':sum(not row['status_present'] for row in selected),
            'unknown_supplied':sum(row['status_present'] and row['status']=='unknown' for row in selected)}
            for kind,selected in kinds.items()}
        name_by_event={kind:{'present':sum(row['tool_name_present'] for row in selected),
            'missing':sum(not row['tool_name_present'] for row in selected),
            'resolved_exact':sum(row['tool_resolution_basis']=='exact' for row in selected),
            'resolved_documented_alias':sum(row['tool_resolution_basis']=='documented_alias' for row in selected),
            'supplied_unresolved':sum(row['tool_name_present'] and row['resolved_tool']=='unknown' for row in selected),
            'supplied_unresolved_shapes':{shape:sum(row['tool_name_present'] and row['resolved_tool']=='unknown' and row['tool_name_shape']==shape for row in selected) for shape in SHAPES}}
            for kind,selected in kinds.items()}
        summaries.append(dict(stream_complete=progress['stream_complete'],response_usable=progress['response_usable'],
            final_event_type=progress['final_event_type'],final_reason_code=progress['final_reason_code'],
            returncode=progress['returncode'],timed_out=progress['timed_out'],duration_seconds=progress['duration_seconds'],
            stdout_bytes=progress['stdout_bytes'],stderr_bytes=progress['stderr_bytes'],frames=progress['frames'],
            events_after_final=progress['events_after_final'],retained_tool_events_after_final=sum(row['after_final'] for row in rows),
            legacy_tool_call_name_counts=progress['tool_name_counts'],legacy_tool_status_event_counts=progress['tool_status_counts'],
            observed_tool_events=events['observed_events'],retained_tool_events=events['retained_events'],
            omitted_tool_events=events['omitted_events'],retained_tool_frame_bytes=events['retained_frame_bytes'],
            max_retained_tool_frame_bytes=max((row['frame_bytes'] for row in rows),default=None),
            call_identities=events['call_identities'],call_relation_counts=events['call_relation_counts'],
            observation_complete=events['observation_complete'],incompleteness=events['incompleteness'],
            payload_lanes=lanes,payload_bytes_by_legacy_tool_name_on_event=legacy_bytes,
            payload_bytes_by_resolved_tool_name_on_event=resolved_bytes,
            classification_counts=events['classification_counts'],status_presence_by_event_type=status_by_event,
            tool_name_presence_and_resolution_by_event_type=name_by_event,
            payload_limits={key:events[key] for key in ('payload_byte_limit','payload_node_limit','payload_depth_limit','payload_integer_bit_limit')},
            tool_name_classification_scope='each-retained-event-present-bounded-name-digest',
            tool_name_inheritance_performed=False,call_hash_attribution_performed=False,
            missing_update_names_preserved_as_unknown=True,
            missing_update_name_alone_revokes_observation_coverage=False,
            missing_update_status_alone_revokes_observation_coverage=False,
            missing_call_name_or_call_status_revokes_observation_coverage=True,
            payload_lanes_may_overlap=True,unique_output_bytes_measured=False,
            tool_result_coverage_complete=events['tool_result_coverage_complete'],
            repetition_scope=events['repetition_scope'],repetition_categories_overlap=True,
            replacement_or_delta_semantics='unverified',
            event_repetition_does_not_establish_history_resends=True,
            provider_request_bytes=None,history_resend_bytes=None,request_history_complete=False,
            tokens=token_counts(invocation['tokens']),known_token_subtotals=token_counts(invocation['known_token_subtotals']),
            native_usage_complete_observed=boolean(invocation['native_usage_complete_observed'],nullable=True),
            native_task_complete_observed=boolean(invocation['native_task_complete_observed'],nullable=True),
            cache_included_in_input=boolean(invocation['cache_included_in_input'],nullable=True),billing_total_verified=False,
            reported_cost_usd=number(invocation['reported_cost_usd']),
            known_reported_cost_subtotal_usd=number(invocation['known_reported_cost_subtotal_usd'])))
    source384=None
    if observation.get('source384') is not None:
        source384={key:digest(observation['source384'][key]) for key in ('checkpoint_sha256','config_sha256','inference_sha256')}
        source384.update(seconds=number(observation['source384']['seconds']),
            preparation_inference_record_retained=boolean(observation['source384']['preparation_inference_record_retained']),
            worker_consumption_independently_verified=boolean(observation['source384']['worker_consumption_independently_verified']))
    phases=observation['phases'];assert type(phases) is dict and set(phases)<=set(PHASES)
    absence=data['exact_container_absence_observations'];assert type(absence) is list and len(absence)==2
    return dict(schema='terminal-tool-event-closed-arithmetic-analysis@2',diagnostic_qualified=boolean(data['qualified']),
        observation_only=True,validated_progress_schema='native-grok-stream-progress@5',
        validated_tool_event_schema='native-grok-tool-events@2',arithmetic_checked_against_retained_rows=True,invocations=summaries,
        planning_provider_calls=scalar(observation['planning_provider_calls'],int,nullable=True),
        coding_provider_calls=scalar(observation['coding_provider_calls'],int,nullable=True),
        provider_internal_model_calls=None,official_verifier_executed=False,official_reward=None,
        driver_task_completed_observed=boolean(observation.get('driver_task_completed_observed'),nullable=True),
        task_success_claimed=False,causal_token_savings_claimed=False,raw_content_exported=False,
        payload_lane_totals_not_added=True,source384=source384,phases={key:number(value) for key,value in phases.items()},
        worker_cleanup_returncode=scalar(observation.get('worker_cleanup_returncode'),int,nullable=True),
        exact_container_absence_observations=[boolean(value,nullable=True) for value in absence],
        container_stop_returned=boolean(data['container_stop_returned']),frozen_inputs_unchanged=boolean(data['frozen_inputs_unchanged']),
        seconds=number(data['seconds']),private_owner_or_provider_output_read=False,docker_calls=0,provider_calls=0)


if __name__=='__main__':
    freeze=json.loads((B/'live/source-freeze.json').read_bytes())
    source=Path(freeze['roots']['source']['path'])
    for name in ('ipfs_accelerate_py/cli_runtime/grok_stream_capture.py','ipfs_accelerate_py/cli_runtime/grok_tool_events.py'):
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==freeze['source_sha256'][name]
    sys.path.insert(0,str(source))
    raw=INPUT.read_bytes();assert len(raw)<=262144
    result=analyze(json.loads(raw));result['closed_export_sha256']=hashlib.sha256(raw).hexdigest()
    result['analysis_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with OUTPUT.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(path=str(OUTPUT),bytes=OUTPUT.stat().st_size,sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest()),sort_keys=True))
