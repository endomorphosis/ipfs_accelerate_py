"""Closed retained diagnostics; no authority, runtime imports, logs, or providers."""
import json
from diagnostic_bridge_contract import validate_bridge_failure_diagnostic
from diagnostic_runner_contract import validate_runner_error_envelope
from diagnostic_grok_progress_contract import validate_grok_stream_progress, grok_stream_progress_summary

FLAGS={'observation_only':True,'completion_authority':False,'retry_authority':False,'settlement_authority':False}

def failure_observations(value):
    if value is None:return {'status':'unavailable',**FLAGS}
    try:
        if (type(value) is not dict or set(value)!={'schema','bridge','planner_child',*FLAGS}
            or value['schema']!='terminal-native-failure-observations@1'
            or any(value[k] is not v for k,v in FLAGS.items())):raise ValueError()
        projected={}
        for name,scope in [('bridge','exact_admitted_task_and_attempt'),('planner_child','planner_child_stderr')]:
            row=value[name]
            if (type(row) is not dict or set(row)!={'status','diagnostic','scope','matched_record_count','provider_dispatch_observed',*FLAGS}
                or type(row['status']) is not str or row['status'] not in {'observed','missing','ambiguous','invalid','unavailable'}
                or row['scope']!=scope or row['provider_dispatch_observed'] is not None
                or type(row['matched_record_count']) is not int or not 0<=row['matched_record_count']<=4096
                or any(row[k] is not v for k,v in FLAGS.items())):raise ValueError()
            diagnostic=None
            if row['status']=='observed':
                diagnostic=(validate_bridge_failure_diagnostic(row['diagnostic']) if name=='bridge'
                    else validate_runner_error_envelope(row['diagnostic']))
                if diagnostic is None or row['matched_record_count']!=1:raise ValueError()
            elif row['diagnostic'] is not None:raise ValueError()
            projected[name]={**row,'diagnostic':diagnostic}
        return {'status':'validated_retained_observation','schema':value['schema'],**projected,**FLAGS}
    except (ValueError,TypeError,KeyError,RecursionError):
        return {'status':'invalid',**FLAGS}

def grok_progress(value):
    if value is None:return {'status':'unavailable'}
    try:
        validated=validate_grok_stream_progress(value)
        result=grok_stream_progress_summary(validated)
        usage=validated.get('usage_observations')
        if usage is not None:
            result['usage_event_observation']={key:usage[key] for key in ('schema','observed_events','retained_events','omitted_events',
                'counter_semantics','ordinals_are_turn_ids','summation_permitted','billing_total_verified','raw_provider_data_exported')}
        projected = {'status':'validated_retained_observation','progress':result,'usage_event_values_exported':False}
        if 'terminal_rejection' in validated:
            projected['terminal_rejection'] = validated['terminal_rejection']
        return projected
    except (ValueError,TypeError,KeyError,RecursionError):
        return {'status':'invalid'}
