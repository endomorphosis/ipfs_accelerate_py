"""Authored reporting-boundary tests; no retained run or provider reads."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('authored_tool_diagnostic_export',Path(__file__).with_name('closed_export.py'))
export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export)


@pytest.fixture
def report():
    from ipfs_accelerate_py.cli_runtime.grok_stream_capture import GrokStreamParser
    from ipfs_accelerate_py.cli_runtime.grok_native_usage import native_grok_observation,grok_usage_receipt
    parser=GrokStreamParser(tool_event_census=True)
    final=dict(type='end',stopReason='end_turn',usage_is_incomplete=False,usage=dict(input_tokens=12,output_tokens=4))
    for frame in [dict(type='tool_call',toolName='read_file',toolCallId='AUTHORED_ONLY_CALL',status='in_progress',rawInput={'path':'AUTHORED_ONLY_PATH'}),
        dict(type='tool_call_update',toolCallId='AUTHORED_ONLY_CALL',status='completed',rawOutput='AUTHORED_ONLY_OUTPUT'),final]:
        parser.feed((json.dumps(frame)+'\n').encode(),elapsed_seconds=.1)
    parser.eof('stdout',elapsed_seconds=.2);parser.eof('stderr',elapsed_seconds=.2);parser.process_exit(0,elapsed_seconds=.2)
    progress=parser.snapshot(elapsed_seconds=.2)
    return dict(provider_invocations=[dict(invocation_id='AUTHORED_ONLY_ID',phase='coding',provider='grok_cli',
        model='AUTHORED_ONLY_MODEL',native_grok_progress=progress,
        native_rollout_usage=grok_usage_receipt(native_grok_observation(final,exit_code=0)),
        extra_private='AUTHORED_ONLY_UNUSED')],worker_cleanup_returncode=0,task_completed=False,
        phases={'prepare_seconds':.25,'AUTHORED_ONLY_PHASE':1})


def test_closed_success_exports_validated_telemetry_only(report):
    out=export.export_report(report)
    assert out['qualified'] is True and out['provider_generation_calls']==1
    assert out['planning_provider_calls']==0 and out['coding_provider_calls']==1
    assert out['task_success_claimed'] is out['complete_run_token_savings_claimed'] is False
    assert out['invocations'][0]['known_token_subtotals']['input_tokens']==12
    assert all(value is None for value in out['invocations'][0]['tokens'].values())
    assert out['invocations'][0]['native_usage_complete_observed'] is None
    assert out['provider_input_bytes'] is out['history_resend_bytes'] is None
    assert 'AUTHORED_ONLY' not in json.dumps(out)


def test_native_failed_outcome_is_reported_without_claiming_task_success(report):
    report['error']={'type':'AuthoredFailure','message':'AUTHORED_ONLY_ERROR'}
    out=export.export_report(report)
    assert out['qualified'] is True and out['driver_error_observed'] is True
    assert out['task_success_claimed'] is False and 'AUTHORED_ONLY' not in json.dumps(out)


@pytest.mark.parametrize('field,value',[('provider','codex_cli'),('phase','planning')])
def test_unexpected_generation_route_rejected(report,field,value):
    report['provider_invocations'][0][field]=value
    with pytest.raises(ValueError):export.export_report(report)


def test_more_than_one_coding_invocation_rejected(report):
    report['provider_invocations'].append(copy.deepcopy(report['provider_invocations'][0]))
    with pytest.raises(ValueError):export.export_report(report)


def test_unreceipted_attempt_prevents_qualified_or_invented_count(report):
    report['unreceipted_provider_attempt']={'body':'AUTHORED_ONLY_ATTEMPT'}
    out=export.export_report(report)
    assert out['qualified'] is False and out['provider_generation_calls'] is None
    assert out['coding_provider_calls'] is None and 'AUTHORED_ONLY' not in json.dumps(out)


def test_zero_calls_retained_without_measured_tool_census(report):
    report['provider_invocations']=[]
    out=export.export_report(report)
    assert out['provider_generation_calls']==0 and out['qualified'] is False


def test_incomplete_cleanup_prevents_qualification(report):
    report['worker_cleanup_returncode']=1
    assert export.export_report(report)['qualified'] is False


def test_legacy_progress_cannot_qualify_fresh_tool_measurement(report):
    from ipfs_accelerate_py.cli_runtime.grok_stream_capture import grok_stream_progress_summary
    report['provider_invocations'][0]['native_grok_progress']=grok_stream_progress_summary(report['provider_invocations'][0]['native_grok_progress'])
    with pytest.raises(ValueError):export.export_report(report)


def test_raw_field_injection_rejected(report):
    report['provider_invocations'][0]['native_grok_progress']['tool_event_observations']['raw_body']='AUTHORED_ONLY_INJECTED'
    with pytest.raises((ValueError,TypeError)):export.export_report(report)


def test_maximum_tool_and_usage_rows_fit_closed_export_bound(report):
    from ipfs_accelerate_py.cli_runtime.grok_stream_capture import GrokStreamParser
    from ipfs_accelerate_py.cli_runtime.grok_tool_events import MAX_TOOL_EVENTS
    from ipfs_accelerate_py.cli_runtime.grok_stream_usage import MAX_USAGE_EVENTS
    parser=GrokStreamParser(tool_event_census=True)
    payload='AUTHORED_ONLY_LARGE_λ'*1024
    for index in range(MAX_TOOL_EVENTS):
        event=dict(type='tool_call',toolCallId='AUTHORED_ONLY_'+str(index),toolName='read_file',
            status='completed',rawInput={'path':payload},rawOutput=payload,
            content=[{'type':'text','text':payload}])
        parser.feed((json.dumps(event,ensure_ascii=False)+'\n').encode(),elapsed_seconds=.1)
    for index in range(MAX_USAGE_EVENTS):
        event=dict(type='usage',stopReason='tool_use',usage={field:2**63-1 for field in
            ('input_tokens','cache_read_input_tokens','cache_creation_input_tokens','output_tokens','reasoning_tokens','total_tokens')})
        parser.feed((json.dumps(event)+'\n').encode(),elapsed_seconds=.2)
    parser.feed(b'{"type":"end","stopReason":"end_turn"}\n',elapsed_seconds=.3)
    parser.eof('stdout',elapsed_seconds=.4);parser.eof('stderr',elapsed_seconds=.4);parser.process_exit(0,elapsed_seconds=.4)
    report['provider_invocations'][0]['native_grok_progress']=parser.snapshot(elapsed_seconds=.4)
    out=export.export_report(report)
    progress=out['invocations'][0]['native_grok_progress']
    assert len(progress['tool_event_observations']['rows'])==MAX_TOOL_EVENTS==64
    assert len(progress['usage_observations']['rows'])==MAX_USAGE_EVENTS==128
    assert out['qualified'] is True
    assert len(json.dumps(out,sort_keys=True,allow_nan=False).encode())<=262144
    assert 'AUTHORED_ONLY' not in json.dumps(out)
