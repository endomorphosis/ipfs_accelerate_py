"""Export validated tool telemetry and numeric usage; never provider text."""
import hashlib
import json
import math
import re

PHASES=('prepare_seconds','initial_context_seconds','planning_seconds','context_seconds',
    'admission_seconds','coding_seconds','cleanup_seconds')


def _need(value):
    if value is not True:
        raise ValueError('closed coding diagnostic contract differs')


def _numeric(value):
    return value if type(value) in (int,float) and 0 <= value < 2**63 and math.isfinite(value) else None


def export_report(report):
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _invocations, TOKENS
    from ipfs_accelerate_py.cli_runtime.grok_stream_capture import validate_grok_stream_progress
    _need(type(report) is dict)
    calls=report.get('provider_invocations')
    _need(type(calls) is list and len(calls)<=1)
    _need(all(type(row) is dict and row.get('phase')=='coding' and row.get('provider')=='grok_cli' for row in calls))
    result=dict(schema='terminal-public-tool-event-readout@1',qualified=False,
        observation_only=True,planning_provider_calls=None,coding_provider_calls=None,
        provider_generation_calls=None,provider_internal_model_calls=None,unreceipted_provider_attempt_observed=
            report.get('unreceipted_provider_attempt') not in (None,False),
        task_success_claimed=False,official_reward=None,official_verifier_executed=False,
        complete_run_token_savings_claimed=False,provider_input_bytes=None,history_resend_bytes=None,
        request_history_complete=False,raw_prompt_or_provider_content_exported=False,
        billing_total_verified=False,invocations=[])
    if not result['unreceipted_provider_attempt_observed']:
        result.update(planning_provider_calls=0,coding_provider_calls=len(calls),provider_generation_calls=len(calls))
    if type(report.get('worker_cleanup_returncode')) is int:
        result['worker_cleanup_returncode']=report['worker_cleanup_returncode']
    if type(report.get('task_completed')) is bool:
        result['driver_task_completed_observed']=report['task_completed']
    result['driver_error_observed']=report.get('error') is not None
    if result['driver_error_observed']:
        error=report['error']
        if type(error) is dict and type(error.get('type')) is str:
            result['driver_error_type_sha256']=hashlib.sha256(error['type'].encode()).hexdigest()
    phases=report.get('phases',{})
    _need(type(phases) is dict)
    result['phases']={name:_numeric(phases[name]) for name in PHASES if name in phases and _numeric(phases[name]) is not None}
    for metric in _invocations(report):
        progress=validate_grok_stream_progress(metric['native_grok_progress'])
        _need(progress['schema']=='native-grok-stream-progress@4')
        tools=progress['tool_event_observations']
        _need(tools['provider_request_bytes'] is None and tools['history_resend_bytes'] is None)
        _need(tools['request_history_complete'] is False and tools['raw_provider_data_exported'] is False)
        row=dict(native_grok_progress=progress,billing_total_verified=False)
        for key in ('tokens','known_token_subtotals'):
            value=metric[key];_need(type(value) is dict and set(value)==set(TOKENS))
            _need(all(v is None or type(v) is int and 0<=v<2**63 for v in value.values()))
            row[key]=value
        for key in ('native_usage_complete_observed','native_task_complete_observed','cache_included_in_input'):
            value=metric[key];_need(value is None or type(value) is bool);row[key]=value
        for key in ('seconds','timeout_seconds','reported_cost_usd','known_reported_cost_subtotal_usd'):
            row[key]=_numeric(metric[key])
        result['invocations'].append(row)
    initial=report.get('initial_context')
    if type(initial) is dict and type(initial.get('source384_context')) is dict:
        native=initial['source384_context'];source384={}
        for key in ('checkpoint_sha256','config_sha256','inference_sha256'):
            _need(type(native.get(key)) is str and re.fullmatch('[0-9a-f]{64}',native[key]) is not None)
            source384[key]=native[key]
        _need(_numeric(native.get('seconds')) is not None)
        source384.update(seconds=_numeric(native['seconds']),preparation_inference_record_retained=True,
            worker_consumption_independently_verified=False)
        result['source384']=source384
    result['qualified']=(result['provider_generation_calls']==1 and len(result['invocations'])==1
        and result.get('worker_cleanup_returncode')==0 and not result['unreceipted_provider_attempt_observed'])
    _need(len(json.dumps(result,sort_keys=True,allow_nan=False).encode())<=262144)
    return result
